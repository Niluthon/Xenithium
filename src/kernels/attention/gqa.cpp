// Xenithium · dense gated GQA decode (see gqa.hpp for the semantics, the cache layout and the split cut).
// The prep and the per-tile score / P V body follow QSA's decode kernels (qsa.cpp); particular to this file are
// the context splits with a running softmax, the 6-head DPAS shape and the W4 projections.
#include "xenithium/kernels/attention/gqa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/math.hpp>

#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/rows.hpp"

#include "attention/gqa_seg.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = gqa_bf16;

constexpr int kD = kQsaDim, kTile = kQsaTile, kPage = kQsaPage;
constexpr int kPartStride = kD + 16;  // o[256] | max, sum (16-float aligned rows)

inline float bfr(float v) { return static_cast<float>(bf16(v)); }


struct SegW {
  const int32_t *rec = nullptr;
  int stride = 0, seg_rows = 0;
  const bf16 *sinks = nullptr;  // merge: the card's q heads' sink logits (arch flint)
};

// The cut of a row's context (gqa.hpp): tiles per split and the number of splits.
inline int cut_tps(int ntiles) { return (ntiles + kGqaSplits - 1) / kGqaSplits; }
inline int cut_nsp(int ntiles, int tps) { return (ntiles + tps - 1) / tps; }
#ifndef XEN_TRAINED_SEL_TPS
#define XEN_TRAINED_SEL_TPS 4
#endif
// a trained selected list (gather plans: ~34 tiles) runs at least this many tiles per split: one work-group per tile
// paid its q load, partial store and merge read for 64 tokens of work
inline int sel_tps(int ntiles) { const int t = cut_tps(ntiles); return t > XEN_TRAINED_SEL_TPS ? t : XEN_TRAINED_SEL_TPS; }

// ---- prep: norms, rope, cache writes (SIMT, one work-group per (row, unit); units: q heads, kv heads) ----
template <QsaKv KV>
class GqaPrepK;

inline int8_t i8_enc(float v, float inv) {
  const float t = sycl::rint(v * inv);
  return static_cast<int8_t>(t > 127.f ? 127.f : (t < -127.f ? -127.f : t));
}
// int4 code (int8-v4's V, the same as qsa.cpp's i4_enc): rne(v (1 / s)) clamped to +-7, == qsa_ref_v4.
inline int i4_enc(float v, float inv) {
  const float t = sycl::rint(v * inv);
  return static_cast<int>(t > 7.f ? 7.f : (t < -7.f ? -7.f : t));
}

template <QsaKv KV>
sycl::event prep(sycl::queue &q, const GqaShape &s, int rows, const GqaWeights &w, const GqaCache &c, const bf16 *rope, const SegW &sw,
                 const GqaScratch &sc) {
  const int hq = s.hq(), hkv = s.hkv(), units = hq + hkv, ldy = w.qkv_n(), qn = s.q_n();
  const int D = s.head_dim, half = s.rot / 2, rot = s.rot, qstride = (s.gate ? 2 : 1) * s.head_dim;
  const bool norm = s.qk_norm;
  const float nadd = s.norm_plus1 ? 1.f : 0.f;
  const int ring = c.ring_pages;
  const float eps = s.eps;
  const float *y = sc.y;
  bf16 *qo = sc.q;
  uint8_t *kp = c.k, *vp = c.v;
  float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  const bf16 *qnw = w.q_norm, *knw = w.k_norm;
  const bool mrope = s.mrope;
  const int rdelta = s.rope_delta;
  const bf16 *rrows = s.rope_rows;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> buf(sycl::range<1>(kD), h);
    sycl::local_accessor<float, 1> qd(sycl::range<1>(4), h);
    sycl::local_accessor<float, 1> vg(sycl::range<1>(KV == QsaKv::i8v4 ? kD : 1), h);            // int8-v4: |v| per sub-group
    sycl::local_accessor<float, 1> qv(sycl::range<1>(KV == QsaKv::i8v4 ? kQsaV4Scales : 1), h);  // int8-v4: 1 / s per group
    h.parallel_for<GqaPrepK<KV>>(sycl::nd_range<1>(static_cast<size_t>(rows) * units * D, D), [=](sycl::nd_item<1> it) {
      const int g = static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
      const int r = g / units, u = g % units;
      const SegRow sr = seg_row(rec, rstride, seg_rows, r);
      const int p = sr.pos;
      const int rpos = seg_vpos(rec, rstride, sr);  // a tree row's rope at its depth (its K / V slot stays p)
      const int32_t *pages = pages0 + static_cast<size_t>(sr.slot) * slot_pages;
      const auto grp = it.get_group();
      // the row's rope row (GqaShape::mrope / rope_rows)
      const bf16 *rrow = rrows ? rrows + static_cast<size_t>(r) * rot
                               : rope + static_cast<size_t>(rpos + (mrope ? (rstride >= kSegRec ? rec[sr.seg * rstride + kSegRope] : rdelta) : 0)) * rot;
      // qwen35: the (1 + w) norm then the rope over the first 64 dims; llama: no norm (n = the bf16 projection), the rope
      // over all D dims (pairs d, d + half; the table row = half cos | half sin)
      auto norm_rope = [&](float v, const bf16 *nw) {
        float n = v;
        if (norm) {
          const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
          n = bfr(v * sycl::rsqrt(ss / static_cast<float>(D) + eps) * (nadd + static_cast<float>(nw[d])));
        }
        buf[d] = n;
        sycl::group_barrier(grp);
        float o = n;
        if (d < rot) {
          const int i = d < half ? d : d - half;
          const float cs = static_cast<float>(rrow[i]);
          const float sn = static_cast<float>(rrow[half + i]);
          o = d < half ? bfr(n * cs - buf[d + half] * sn) : bfr(n * cs + buf[d - half] * sn);
        }
        return o;
      };
      const float *yr = y + static_cast<size_t>(r) * ldy;
      if (u < hq) {
        const float o = norm_rope(bfr(yr[u * qstride + d]), qnw);
        qo[(static_cast<size_t>(r) * hq + u) * D + d] = bf16(o);
        return;
      }
      const int kh = u - hq;
      const float kn = norm_rope(bfr(yr[qn + kh * D + d]), knw);
      const float vv = bfr(yr[qn + (hkv + kh) * D + d]);
      const size_t slot = (static_cast<size_t>(pages[ring ? (p / kPage) % ring : p / kPage]) * hkv + kh) * kPage + (p % kPage);
      if constexpr (KV != QsaKv::bf16) {
        // qsa.hpp's int8 codes: the IEEE divisions once per work-group (fdiv_rn), broadcast through SLM
        namespace imf = sycl::ext::intel::math;
        constexpr bool V4 = KV == QsaKv::i8v4;
        const float ka = sycl::reduce_over_group(grp, sycl::fabs(kn), sycl::maximum<float>());
        const float va = sycl::reduce_over_group(grp, sycl::fabs(vv), sycl::maximum<float>());
        const bool kz = !(ka > 0.f), vz = !(va > 0.f);
        if (d == 0) qd[0] = kz ? 0.f : imf::fdiv_rn(127.f, ka);
        if (!V4 && d == D / 4) qd[1] = vz ? 0.f : imf::fdiv_rn(127.f, va);
        if (d == D / 2) qd[2] = kz ? 0.f : imf::fdiv_rn(ka, 127.f);
        if (!V4 && d == 3 * D / 4) qd[3] = vz ? 0.f : imf::fdiv_rn(va, 127.f);
        sycl::group_barrier(grp);
        reinterpret_cast<int8_t *>(kp)[slot * D + d] = kz ? int8_t(0) : i8_enc(kn, qd[0]);
        if constexpr (V4) {  // qsa.cpp's int8-v4 V (== qsa_ref_v4), head_dim 256: two codes per byte, a bf16
          // scale per 32 dims (each sub-group's amax to SLM, lane 32 g + 16 runs group g's two IEEE divisions)
          const auto sg = it.get_sub_group();
          const int S = static_cast<int>(sg.get_max_local_range()[0]);
          const float sm = sycl::reduce_over_group(sg, sycl::fabs(vv), sycl::maximum<float>());
          if (sg.get_local_linear_id() == 0) vg[d / S] = sm;
          sycl::group_barrier(grp);
          if (d % kQsaV4Group == kQsaV4Group / 2) {
            const int gq = d / kQsaV4Group;
            float am = 0.f;
            for (int j = gq * kQsaV4Group / S; j < (gq + 1) * kQsaV4Group / S; ++j) am = sycl::fmax(am, vg[j]);
            const bf16 sb = am > 0.f ? bf16(imf::fdiv_rn(am, 7.f)) : bf16(0.f);
            const float sf = static_cast<float>(sb);
            qv[gq] = sf > 0.f ? imf::fdiv_rn(1.f, sf) : 0.f;
            reinterpret_cast<bf16 *>(vsp)[slot * kQsaV4Scales + gq] = sb;
          }
          sycl::group_barrier(grp);
          const int cv = i4_enc(vv, qv[d / kQsaV4Group]);
          const int c1 = sycl::permute_group_by_xor(sg, cv, 1);
          if ((d & 1) == 0) vp[slot * (D / 2) + d / 2] = static_cast<uint8_t>((cv & 0xF) | ((c1 & 0xF) << 4));
        } else {
          reinterpret_cast<int8_t *>(vp)[slot * D + d] = vz ? int8_t(0) : i8_enc(vv, qd[1]);
        }
        if (d == 0) {
          ksp[slot] = qd[2];
          if constexpr (!V4) vsp[slot] = qd[3];
        }
      } else {
        reinterpret_cast<bf16 *>(kp)[slot * D + d] = bf16(kn);
        reinterpret_cast<bf16 *>(vp)[slot * D + d] = bf16(vv);
      }
    });
  });
}

// Prep as ESIMD for decode steps, one thread per (row, unit) with the unit's D dims in registers: no group
// reductions or barriers. The same arithmetic as the SIMT prep above: the sum of squares in its work-group
// reduction's order (an xor butterfly within each 16-lane sub-group, then across the sub-groups), the rotary as
// written, 1 / sqrt by IEEE divide and square root where the SIMT kernel calls sycl::rsqrt (not an ESIMD
// function), so a q element can differ from the SIMT prep's by one bf16 ulp. Prefill (gqa_rows) keeps the SIMT
// prep: a prefill call's bits do not depend on its size.
template <QsaKv KV, int D>
class GqaPrepEK;

template <int N>
inline float bfly(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {  // x[i] + x[i + N / 2] first (xor N / 2 down to 1)
  if constexpr (N == 1) {
    return x[0];
  } else {
    esimd::simd<float, N / 2> h = x.template select<N / 2, 1>(0).read() + x.template select<N / 2, 1>(N / 2).read();
    return bfly<N / 2>(h);
  }
}
// the SIMT work-group reduction's order: lanes by xor butterfly within each 16-lane sub-group, then the sub-groups' sums
// by xor butterfly
template <int D>
inline float sumsq(esimd::simd<float, D> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, D> q = v * v;
  constexpr int G = D / 16;
  esimd::simd<float, G> g;
#pragma unroll
  for (int i = 0; i < G; ++i) g[i] = bfly<16>(q.template select<16, 1>(i * 16).read());
  return bfly<G>(g);
}

template <QsaKv KV, int D>
sycl::event prep_e(sycl::queue &q, const GqaShape &s, int rows, const GqaWeights &w, const GqaCache &c, const bf16 *rope, const SegW &sw,
                   const GqaScratch &sc) {
  const int hq = s.hq(), hkv = s.hkv(), units = hq + hkv, ldy = w.qkv_n(), qn = s.q_n();
  const int rot = s.rot, qstride = (s.gate ? 2 : 1) * D;
  const bool norm = s.qk_norm;
  const float nadd = s.norm_plus1 ? 1.f : 0.f;
  const int ring = c.ring_pages;
  const float eps = s.eps;
  const float *y = sc.y;
  bf16 *qo = sc.q;
  uint8_t *kp = c.k, *vp = c.v;
  float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  const bf16 *qnw = w.q_norm, *knw = w.k_norm;
  const bool mrope = s.mrope;  // the row's rope row at position + delta (GqaShape::mrope; no rope_rows here)
  const int rdelta = s.rope_delta;
  return q.parallel_for<GqaPrepEK<KV, D>>(sycl::nd_range<1>(static_cast<size_t>(rows) * units, 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int g = static_cast<int>(it.get_group(0));
    const int r = g / units, u = g % units;
    const SegRow sr = seg_row(rec, rstride, seg_rows, r);
    const int p = sr.pos;
    const int rp = p + (mrope ? (rstride >= kSegRec ? rec[sr.seg * rstride + kSegRope] : rdelta) : 0);
    const int32_t *pages = pages0 + static_cast<size_t>(sr.slot) * slot_pages;
    auto norm_rope = [&](esimd::simd<float, D> v, const bf16 *nw) SYCL_ESIMD_FUNCTION {
      esimd::simd<float, D> n = v;
      if (norm) {
        const float ss = sumsq<D>(v);
        const esimd::simd<float, 1> mv(ss / static_cast<float>(D) + eps);
        // 1 / sqrt: IEEE divide of the IEEE root
        const float rs = esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(mv))[0];
        const esimd::simd<float, D> wv = esimd::block_load<bf16, D>(nw);
        const esimd::simd<float, D> t = v * rs;
        n = esimd::simd<bf16, D>(t * (wv + nadd));
      }
      if (rot == D) {  // NeoX over every dim (arch flint's sliding layers; the D 128 shapes): pairs d, d + D / 2
        constexpr int H2 = D / 2;
        const esimd::simd<float, H2> cs = esimd::block_load<bf16, H2>(rope + static_cast<size_t>(rp) * D);
        const esimd::simd<float, H2> sn = esimd::block_load<bf16, H2>(rope + static_cast<size_t>(rp) * D + H2);
        const esimd::simd<float, H2> lo = n.template select<H2, 1>(0), hi = n.template select<H2, 1>(H2);
        const esimd::simd<float, H2> olo = lo * cs - hi * sn, ohi = hi * cs + lo * sn;
        n.template select<H2, 1>(0) = esimd::simd<float, H2>(esimd::simd<bf16, H2>(olo));
        n.template select<H2, 1>(H2) = esimd::simd<float, H2>(esimd::simd<bf16, H2>(ohi));
      }
      return n;
    };
    const float *yr = y + static_cast<size_t>(r) * ldy;
    if (u < hq) {
      const esimd::simd<float, D> v = esimd::simd<bf16, D>(esimd::block_load<float, D>(yr + u * qstride));
      const esimd::simd<bf16, D> o = norm_rope(v, qnw);
      o.copy_to(qo + (static_cast<size_t>(r) * hq + u) * D);
      return;
    }
    const int kh = u - hq;
    const esimd::simd<float, D> kn = norm_rope(esimd::simd<bf16, D>(esimd::block_load<float, D>(yr + qn + kh * D)), knw);
    const esimd::simd<float, D> vv = esimd::simd<bf16, D>(esimd::block_load<float, D>(yr + qn + (hkv + kh) * D));
    const size_t slot = (static_cast<size_t>(pages[ring ? (p / kPage) % ring : p / kPage]) * hkv + kh) * kPage + (p % kPage);
    if constexpr (KV == QsaKv::i8) {
      const float ka = esimd::hmax<float>(esimd::abs(kn)), va = esimd::hmax<float>(esimd::abs(vv));
      const bool kz = !(ka > 0.f), vz = !(va > 0.f);
      const float ki = kz ? 0.f : esimd::div_ieee(esimd::simd<float, 1>(127.f), esimd::simd<float, 1>(ka))[0];
      const float vi = vz ? 0.f : esimd::div_ieee(esimd::simd<float, 1>(127.f), esimd::simd<float, 1>(va))[0];
      auto enc = [&](esimd::simd<float, D> x, float inv, bool z) SYCL_ESIMD_FUNCTION {
        esimd::simd<float, D> t = esimd::rnde<float, D>(x * inv);
        t = esimd::max(esimd::min(t, 127.f), -127.f);
        esimd::simd<int8_t, D> c8 = t;
        if (z) c8 = 0;
        return c8;
      };
      esimd::block_store<int8_t, D>(reinterpret_cast<int8_t *>(kp) + slot * D, enc(kn, ki, kz));
      esimd::block_store<int8_t, D>(reinterpret_cast<int8_t *>(vp) + slot * D, enc(vv, vi, vz));
      ksp[slot] = kz ? 0.f : esimd::div_ieee(esimd::simd<float, 1>(ka), esimd::simd<float, 1>(127.f))[0];
      vsp[slot] = vz ? 0.f : esimd::div_ieee(esimd::simd<float, 1>(va), esimd::simd<float, 1>(127.f))[0];
    } else {
      esimd::block_store<bf16, D>(reinterpret_cast<bf16 *>(kp) + slot * D, esimd::simd<bf16, D>(kn));
      esimd::block_store<bf16, D>(reinterpret_cast<bf16 *>(vp) + slot * D, esimd::simd<bf16, D>(vv));
    }
  });
}

int g_prep_e = 1;  // 0 = the SIMT prep in decode too (gqa_set_prep: tests)
}  // namespace
void gqa_set_prep(int esimd) { g_prep_e = esimd; }
namespace {

// ---- attention: one work-group per (row, kv head, context split). 16 items x 16 dims: each K / V
// byte is read once; per 64-token tile the partial q.k per item are summed in SLM in item order, the
// tile's softmax updates the head's running max / sum (items 0..G-1, one head each), every item rescales
// its o accumulators by exp(m_old - m_new) and adds P V on DPAS. (o, max, sum) per head to `part`.
template <QsaKv KV, int G>
class GqaAttnK;

template <QsaKv KV, int G>
sycl::event attn(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const SegW &sw, const GqaScratch &sc, int ns) {
  constexpr bool I8 = KV == QsaKv::i8;
  constexpr int kI = 16, kE = I8 ? 1 : 2;
  constexpr int GA = G > 8 ? 8 : G, GB = G - GA, GBn = GB > 0 ? GB : 1;  // DPAS repeat counts: heads per call
  static_assert(G <= 16 && G >= 1, "1..16 q heads per kv head");
  constexpr int kTokOff = 0, kKsOff = kTokOff + kTile, kVsOff = kKsOff + kTile;
  constexpr int kSpOff = kVsOff + kTile, kSpN = kI * G * 16 + 32;  // + the last item's window
  constexpr int kScOff = kSpOff + kSpN, kPvOff = kScOff + G * kTile, kAlOff = kPvOff + G * kTile;
  constexpr int kSlm = (kAlOff + 16) * 4;
  const int hq = s.hq(), hkv = s.hkv();
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  const bf16 *qb = sc.q;
  float *part = sc.part;
  return q.parallel_for<GqaAttnK<KV, G>>(
      sycl::nd_range<1>(static_cast<size_t>(rows) * hkv * ns * kI, kI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlm>();
        const int g = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
        const int r = g / (hkv * ns), kvh = (g / ns) % hkv, sp = g % ns;
        const SegRow srw = seg_row(rec, rstride, seg_rows, r);
        const int32_t *pages = pages0 + static_cast<size_t>(srw.slot) * slot_pages;
        const int ntok = srw.pos + 1, ntiles = (ntok + kTile - 1) / kTile;
        const int tps = cut_tps(ntiles), nsp = cut_nsp(ntiles, tps);
        if (sp >= nsp) return;
        const int tile0 = sp * tps, tile1 = tile0 + tps < ntiles ? tile0 + tps : ntiles;
        auto so = [](int f) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>(f * 4); };
        esimd::simd<bf16, G * 16> qa;  // A operand: [head][16 dims of this item]
#pragma unroll
        for (int h = 0; h < G; ++h)
          qa.template select<16, 1>(h * 16) = esimd::block_load<bf16, 16>(qb + (static_cast<size_t>(r) * hq + kvh * G + h) * kD + i * 16);
        auto row16b = [&](const uint8_t *base, int slot) SYCL_ESIMD_FUNCTION {
          const uint8_t *src = base + (static_cast<size_t>(slot) * kD + i * 16) * kE;
          esimd::simd<bf16, 16> v;
          if constexpr (I8) {
            esimd::simd<uint8_t, 16> raw = esimd::block_load<uint8_t, 16>(src);
            const esimd::simd<int8_t, 16> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<float, 16> f = sb;
            v = f;
          } else {
            v = esimd::block_load<bf16, 16>(reinterpret_cast<const bf16 *>(src));
          }
          return v;
        };
        auto tile16 = [&](const uint8_t *base, esimd::simd<int32_t, 16> sl) SYCL_ESIMD_FUNCTION {
          esimd::simd<bf16, 256> out;
          if constexpr (I8) {
            esimd::simd<uint8_t, 256> raw;
#pragma unroll
            for (int tl = 0; tl < 16; ++tl)
              raw.template select<16, 1>(tl * 16) = esimd::block_load<uint8_t, 16>(base + static_cast<size_t>(sl[tl]) * kD + i * 16);
            esimd::simd<int8_t, 256> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<float, 256> f = sb;
            out = f;
          } else {
#pragma unroll
            for (int tl = 0; tl < 16; ++tl) out.template select<16, 1>(tl * 16) = row16b(base, sl[tl]);
          }
          return out;
        };
        const float kmul = 1.f / 16.f;  // 1 / sqrt(256)
        esimd::simd<float, GA * 16> oa(0.f);
        esimd::simd<float, GBn * 16> ob(0.f);
        float m_run = -INFINITY, l_run = 0.f;  // items 0..G-1: head i's running max / sum
#pragma unroll 1
        for (int tile = tile0; tile < tile1; ++tile) {
          const int t0 = tile * kTile, nt = ntok - t0 < kTile ? ntok - t0 : kTile;
          esimd::barrier();  // the previous tile's readers of the SLM rows are done
          {  // token slots of the tile (dead slots repeat the last token)
            constexpr int kPer = kTile / kI;
            esimd::simd<int32_t, kPer> sl;
            esimd::simd<float, kPer> kss(1.f), vss(1.f);
#pragma unroll
            for (int j = 0; j < kPer; ++j) {
              int t = i * kPer + j;
              t = t < nt ? t : nt - 1;
              const int tok = t0 + t;
              const int slot = (pages[tok / kPage] * hkv + kvh) * kPage + (tok % kPage);
              sl[j] = slot;
              if constexpr (I8) {
                kss[j] = ksp[slot];
                vss[j] = vsp[slot];
              }
            }
            esimd::slm_block_store<int32_t, kPer>(so(kTokOff + i * kPer), sl);
            esimd::slm_block_store<float, kPer>(so(kKsOff + i * kPer), kss);
            esimd::slm_block_store<float, kPer>(so(kVsOff + i * kPer), vss);
          }
          esimd::barrier();
          // ---- scores, 16 tokens at a time (B = K^T in dword pairs, A = q of the heads) ----
#pragma unroll 1
          for (int sub = 0; sub < kTile / 16; ++sub) {
            if (sub * 16 >= nt) break;
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
            esimd::simd<bf16, 256> kb = tile16(kp, sl);
            esimd::simd<uint32_t, 128> bd;
            {
              esimd::simd<uint32_t, 128> kd = kb.template bit_cast_view<uint32_t>();
#pragma unroll
              for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
            }
            const esimd::simd<bf16, 256> bk = bd.template bit_cast_view<bf16>();
            esimd::simd<float, G * 16> spd;
            {
              const esimd::simd<bf16, GA * 16> aa = qa.template select<GA * 16, 1>(0);
              esimd::simd<float, GA * 16> ca(0.f);
              ca = xmx::dpas<8, GA, float, float, bf16, bf16>(ca, bk, aa);
              spd.template select<GA * 16, 1>(0) = ca;
              if constexpr (GB > 0) {
                const esimd::simd<bf16, GB * 16> ab = qa.template select<GB * 16, 1>(GA * 16);
                esimd::simd<float, GB * 16> cb(0.f);
                cb = xmx::dpas<8, GB, float, float, bf16, bf16>(cb, bk, ab);
                spd.template select<GB * 16, 1>(GA * 16) = cb;
              }
            }
            esimd::slm_block_store<float, G * 16>(so(kSpOff + i * G * 16), spd);
            esimd::barrier();
            // item i sums outputs [i G, i G + G) of (h, tl) over the 16 items, in item order (a 16-byte aligned
            // 32-float window: i G is not a multiple of 4 floats when G = 6)
            esimd::simd<float, 16> acc(0.f);
            const int wb = (i * G) & ~3, wo = i * G - wb;
#pragma unroll
            for (int it2 = 0; it2 < kI; ++it2) {
              esimd::simd<float, 32> win = esimd::slm_block_load<float, 32>(so(kSpOff + it2 * G * 16 + wb));
              acc += win.template select<16, 1>(wo);
            }
            const esimd::simd<float, 16> ks = esimd::slm_block_load<float, 16>(so(kKsOff + sub * 16));
#pragma unroll
            for (int j = 0; j < G; ++j) {
              const int o = i * G + j, h = o / 16, tl = o % 16;
              float sv = acc[j] * kmul;
              if constexpr (I8) sv *= static_cast<float>(ks[tl]);
              esimd::slm_scalar_store<float>(so(kScOff + h * kTile + sub * 16 + tl), sv);
            }
            esimd::barrier();
          }
          // ---- running softmax per head: m' = max(m, tile max), alpha = exp(m - m'), l' = l alpha + sum p ----
          if (i < G) {
            esimd::simd<float, kTile> sv = esimd::slm_block_load<float, kTile>(so(kScOff + i * kTile));
            esimd::simd<int32_t, kTile> ti(0, 1);
            sv.merge(esimd::simd<float, kTile>(-INFINITY), ti >= nt);
            const float mt = esimd::hmax<float>(sv);
            const float mn = m_run > mt ? m_run : mt;
            const esimd::simd<float, 1> al = esimd::exp(esimd::simd<float, 1>(m_run - mn));  // 0 on the first tile
            esimd::simd<float, kTile> pe = esimd::exp(sv - mn);
            pe.merge(esimd::simd<float, kTile>(0.f), ti >= nt);
            l_run = l_run * static_cast<float>(al[0]) + esimd::reduce<float>(pe, std::plus<>());
            m_run = mn;
            if constexpr (I8) pe *= esimd::slm_block_load<float, kTile>(so(kVsOff));
            esimd::slm_block_store<float, kTile>(so(kPvOff + i * kTile), pe);
            esimd::slm_scalar_store<float>(so(kAlOff + i), al[0]);
          }
          esimd::barrier();
          {
            const esimd::simd<float, 16> al = esimd::slm_block_load<float, 16>(so(kAlOff));
#pragma unroll
            for (int h = 0; h < GA; ++h) oa.template select<16, 1>(h * 16) *= static_cast<float>(al[h]);
            if constexpr (GB > 0) {
#pragma unroll
              for (int h = 0; h < GB; ++h) ob.template select<16, 1>(h * 16) *= static_cast<float>(al[GA + h]);
            }
          }
          // ---- P V on DPAS: C (heads x 16 dims) += P (heads x 16 tokens) . V (16 tokens x 16 dims),
          // V as B in token-pair VNNI order; P = hi + lo (two bf16 parts, ~16 significant bits).
#pragma unroll 1
          for (int sub = 0; sub < kTile / 16; ++sub) {
            if (sub * 16 >= nt) break;
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
            esimd::simd<bf16, 256> vb = tile16(vp, sl);
            esimd::simd<bf16, 256> bv;
#pragma unroll
            for (int qq = 0; qq < 8; ++qq) {
              bv.template select<16, 2>(qq * 32) = vb.template select<16, 1>(2 * qq * 16);
              bv.template select<16, 2>(qq * 32 + 1) = vb.template select<16, 1>((2 * qq + 1) * 16);
            }
            esimd::simd<float, G * 16> pw;
#pragma unroll
            for (int h = 0; h < G; ++h) pw.template select<16, 1>(h * 16) = esimd::slm_block_load<float, 16>(so(kPvOff + h * kTile + sub * 16));
            esimd::simd<bf16, G * 16> ph = pw;
            esimd::simd<bf16, G * 16> pl = pw - esimd::simd<float, G * 16>(ph);
            {
              const esimd::simd<bf16, GA * 16> h8 = ph.template select<GA * 16, 1>(0), l8 = pl.template select<GA * 16, 1>(0);
              oa = xmx::dpas<8, GA, float, float, bf16, bf16>(oa, bv, h8);
              oa = xmx::dpas<8, GA, float, float, bf16, bf16>(oa, bv, l8);
            }
            if constexpr (GB > 0) {
              const esimd::simd<bf16, GB * 16> h4 = ph.template select<GB * 16, 1>(GA * 16), l4 = pl.template select<GB * 16, 1>(GA * 16);
              ob = xmx::dpas<8, GB, float, float, bf16, bf16>(ob, bv, h4);
              ob = xmx::dpas<8, GB, float, float, bf16, bf16>(ob, bv, l4);
            }
          }
        }
        float *pb = part + (static_cast<size_t>(r * hkv + kvh) * kGqaSplits + sp) * G * kPartStride;
        if (i < G) {
          esimd::simd<float, 2> ml;
          ml[0] = m_run;
          ml[1] = l_run;
          esimd::block_store<float, 2>(pb + static_cast<size_t>(i) * kPartStride + kD, ml);
        }
#pragma unroll
        for (int h = 0; h < GA; ++h) esimd::block_store<float, 16>(pb + static_cast<size_t>(h) * kPartStride + i * 16, oa.template select<16, 1>(h * 16));
        if constexpr (GB > 0) {
#pragma unroll
          for (int h = 0; h < GB; ++h)
            esimd::block_store<float, 16>(pb + static_cast<size_t>(GA + h) * kPartStride + i * 16, ob.template select<16, 1>(h * 16));
        }
      });
}

// ---- merge: LSE over the splits, bf16, output gate (one 16-item work-group per (row, q head)) ----
// ESIMD, each item 16 dims; every item gathers the splits' (max, sum) in two 32-lane messages. The sums run in
// split order (fp32, from split 0).
template <int D, bool Gate>
class GqaMergeK;
class GqaTrainedMergeK;
// sel_n = the page selection's counts per (segment, kv head) (gqa_select): a segment with a count
// cut its splits by it; 0 / null = by the row's position. D: head_dim (part rows D + 16 floats); Gate: the sigmoid gate (llama: none).
template <int D, bool Gate, bool Trained = false>
sycl::event merge_d(sycl::queue &q, const GqaShape &s, int rows, int ldy, const SegW &sw, const GqaScratch &sc, const int32_t *sel_n) {
  constexpr int kMI = 16, kMD = D / kMI, kL = 32, kPartStride = D + 16;  // items, dims per item, gather lanes
  static_assert(kGqaSplits == 2 * kL, "two 32-lane gathers cover the splits");
  const int hq = s.hq(), hkv = s.hkv(), G = s.grp();
  const float *part = sc.part, *y = sc.y;
  const int32_t *rec = sw.rec;
  const int rstride = sw.stride, seg_rows = sw.seg_rows;
  bf16 *out = sc.attn;
  const int win = s.window;
  const bf16 *sinks = sw.sinks;
  using Kernel = std::conditional_t<Trained, GqaTrainedMergeK, GqaMergeK<D, Gate>>;
  return q.parallel_for<Kernel>(sycl::nd_range<1>(static_cast<size_t>(rows) * hq * kMI, kMI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int g = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
    const int r = g / hq, h = g % hq, kvh = h / G, hh = h % G;
    const SegRow sr = seg_row(rec, rstride, seg_rows, r);
    const int nsel = sel_n ? sel_n[(Trained ? r : sr.seg) * hkv + kvh] : 0;
    const int ntok = seg_vpos(rec, rstride, sr) + 1, ntiles = nsel > 0 ? nsel : (ntok + kTile - 1) / kTile;  // a tree row's splits by its depth
    // windowed: the row's tiles lo .. its last, tile t in split t % kGqaWinSplits (in tile order)
    const int wlo = win > 0 ? (ntok > win ? ntok - win : 0) / kTile : 0;
    const int nt = win > 0 ? (ntok - 1) / kTile - wlo + 1 : cut_nsp(ntiles, (Trained && nsel > 0) ? sel_tps(ntiles) : cut_tps(ntiles));
    auto split_of = [=](int t) SYCL_ESIMD_FUNCTION { return win > 0 ? (wlo + t) % kGqaWinSplits : t; };
    const float *pb = part + (static_cast<size_t>(r * hkv + kvh) * kGqaSplits * G + hh) * kPartStride;
    const uint32_t ts = static_cast<uint32_t>(G * kPartStride);  // floats between splits
    esimd::simd<float, kGqaSplits> mv, lv;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const esimd::simd<uint32_t, kL> t = esimd::simd<uint32_t, kL>(b * kL, 1);
      const esimd::simd_mask<kL> live = t < static_cast<uint32_t>(nt);
      esimd::simd<uint32_t, kL> tw = t;
      if (win > 0) tw = (t + static_cast<uint32_t>(wlo)) % static_cast<uint32_t>(kGqaWinSplits);
      const esimd::simd<uint32_t, kL> off = esimd::merge(tw, esimd::simd<uint32_t, kL>(0), live) * (ts * 4) + D * 4;
      mv.template select<kL, 1>(b * kL) = esimd::gather<float, kL>(pb, off);
      lv.template select<kL, 1>(b * kL) = esimd::gather<float, kL>(pb, off + 4);
    }
    const esimd::simd<int32_t, kGqaSplits> ti(0, 1);
    mv.merge(esimd::simd<float, kGqaSplits>(-INFINITY), ti >= nt);
    float M = esimd::hmax<float>(mv);
    float L = 0.f;
    if (sinks) {  // the sink joins the softmax's denominator: m = max(m, sink), L starts at exp(sink - m)
      const float sk = static_cast<float>(sinks[h]);
      M = M > sk ? M : sk;
      const esimd::simd<float, 1> es = esimd::exp(esimd::simd<float, 1>(sk - M));
      L = es[0];
    }
    esimd::simd<float, kGqaSplits> e = esimd::exp(mv - M);
    e.merge(esimd::simd<float, kGqaSplits>(0.f), ti >= nt);
    esimd::simd<float, kMD> O(0.f);
    for (int t0 = 0; t0 < nt; t0 += 8) {
      esimd::simd<float, 8 * kMD> ov;
#pragma unroll
      for (int j = 0; j < 8; ++j) {  // the round's loads first (dead slots re-read split nt - 1 with weight 0)
        const int t = split_of(t0 + j < nt ? t0 + j : nt - 1);
        ov.template select<kMD, 1>(j * kMD) = esimd::block_load<float, kMD>(pb + static_cast<size_t>(t) * ts + i * kMD);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float ej = t0 + j < nt ? static_cast<float>(e[t0 + j < kGqaSplits ? t0 + j : 0]) : 0.f;
        const float lj = t0 + j < nt ? static_cast<float>(lv[t0 + j < kGqaSplits ? t0 + j : 0]) : 0.f;
        L += lj * ej;
        O += ov.template select<kMD, 1>(j * kMD) * ej;
      }
    }
    const esimd::simd<float, kMD> a = esimd::simd<bf16, kMD>(O / L);
    if constexpr (Gate) {
      const esimd::simd<float, kMD> gt = esimd::simd<bf16, kMD>(esimd::block_load<float, kMD>(y + static_cast<size_t>(r) * ldy + h * 2 * D + D + i * kMD));
      const esimd::simd<float, kMD> sg = esimd::simd<bf16, kMD>(1.f / (1.f + esimd::exp(-gt)));
      esimd::block_store<bf16, kMD>(out + (static_cast<size_t>(r) * hq + h) * D + i * kMD, esimd::simd<bf16, kMD>(a * sg));
    } else {
      (void)y;
      (void)ldy;
      esimd::block_store<bf16, kMD>(out + (static_cast<size_t>(r) * hq + h) * D + i * kMD, esimd::simd<bf16, kMD>(a));
    }
  });
}

template <int D>
class GqaMergeSpK;
// GqaShape::split_merge: one work-group of 16 items per (row, q head); item i merges splits i, i + 16, i + 32, i + 48
// over all D dims (fp32, ascending), then the 16 partial o / sums are summed in item order through SLM (+ the sink).
template <int D>
sycl::event merge_sp(sycl::queue &q, const GqaShape &s, int rows, const SegW &sw, const GqaScratch &sc) {
  constexpr int kI = 16, kMD = D / kI, kPS = D + 16, kPer = kGqaSplits / kI;
  const int hq = s.hq(), hkv = s.hkv(), G = s.grp();
  const float *part = sc.part;
  const int32_t *rec = sw.rec;
  const int rstride = sw.stride, seg_rows = sw.seg_rows, win = s.window;
  const bf16 *sinks = sw.sinks;
  bf16 *out = sc.attn;
  return q.parallel_for<GqaMergeSpK<D>>(sycl::nd_range<1>(static_cast<size_t>(rows) * hq * kI, kI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::slm_init<(kI * D + 2 * kI) * 4>();
    const int g = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
    const int r = g / hq, h = g % hq, kvh = h / G, hh = h % G;
    const SegRow sr = seg_row(rec, rstride, seg_rows, r);
    const int ntok = sr.pos + 1;
    const int wlo = win > 0 ? (ntok > win ? ntok - win : 0) / kTile : 0;
    const int ntiles = (ntok + kTile - 1) / kTile;
    const int nt = win > 0 ? (ntok - 1) / kTile - wlo + 1 : cut_nsp(ntiles, cut_tps(ntiles));
    const float *pb = part + (static_cast<size_t>(r * hkv + kvh) * kGqaSplits * G + hh) * kPS;
    const size_t ts = static_cast<size_t>(G) * kPS;
    float mj[kPer], lj[kPer], mloc = -INFINITY;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int t = i + kI * k;
      mj[k] = -INFINITY, lj[k] = 0.f;
      if (t < nt) {
        const int sp = win > 0 ? (wlo + t) % kGqaWinSplits : t;
        const esimd::simd<float, 4> ml = esimd::block_load<float, 4>(pb + static_cast<size_t>(sp) * ts + D);
        mj[k] = ml[0], lj[k] = ml[1];
        mloc = mloc > mj[k] ? mloc : mj[k];
      }
    }
    esimd::slm_scalar_store<float>(static_cast<uint32_t>((kI * D + i) * 4), mloc);
    esimd::barrier();
    float M = esimd::hmax<float>(esimd::slm_block_load<float, kI>(static_cast<uint32_t>(kI * D * 4)));
    float sk = -INFINITY;
    if (sinks) {
      sk = static_cast<float>(sinks[h]);
      M = M > sk ? M : sk;
    }
    esimd::simd<float, D> o(0.f);
    float l = 0.f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
      const int t = i + kI * k;
      if (t < nt) {
        const int sp = win > 0 ? (wlo + t) % kGqaWinSplits : t;
        const esimd::simd<float, 1> e = esimd::exp(esimd::simd<float, 1>(mj[k] - M));
        o += esimd::block_load<float, D>(pb + static_cast<size_t>(sp) * ts) * e[0];
        l += lj[k] * e[0];
      }
    }
#pragma unroll
    for (int c = 0; c < D; c += 64) esimd::slm_block_store<float, 64>(static_cast<uint32_t>((i * D + c) * 4), o.template select<64, 1>(c));
    esimd::barrier();  // the max's readers are done before the sums overwrite nothing: separate slots
    esimd::slm_scalar_store<float>(static_cast<uint32_t>((kI * D + kI + i) * 4), l);
    esimd::barrier();
    esimd::simd<float, kMD> acc(0.f);
#pragma unroll
    for (int j = 0; j < kI; ++j) acc += esimd::slm_block_load<float, kMD>(static_cast<uint32_t>((j * D + i * kMD) * 4));
    const esimd::simd<float, kI> ls = esimd::slm_block_load<float, kI>(static_cast<uint32_t>((kI * D + kI) * 4));
    float L = 0.f;
#pragma unroll
    for (int j = 0; j < kI; ++j) L += ls[j];
    if (sinks) {
      const esimd::simd<float, 1> es = esimd::exp(esimd::simd<float, 1>(sk - M));
      L += es[0];
    }
    esimd::block_store<bf16, kMD>(out + (static_cast<size_t>(r) * hq + h) * D + i * kMD, esimd::simd<bf16, kMD>(acc / L));
  });
}

sycl::event merge(sycl::queue &q, const GqaShape &s, int rows, int ldy, const SegW &sw, const GqaScratch &sc, const int32_t *sel_n = nullptr) {
  if (s.split_merge) {
    if (s.head_dim != 128 || s.gate || sel_n) throw std::invalid_argument("gqa merge: the split merge is built for head_dim 128 without the gate or selection");
    return merge_sp<128>(q, s, rows, sw, sc);
  }
  if (s.head_dim == 256 && s.gate) return merge_d<256, true>(q, s, rows, ldy, sw, sc, sel_n);
  if (s.head_dim == 128 && !s.gate) return merge_d<128, false>(q, s, rows, ldy, sw, sc, sel_n);
  throw std::invalid_argument("gqa merge: head_dim 256 with the gate or 128 without");
}

sycl::event prep_kv(sycl::queue &q, const GqaShape &s, int rows, const GqaWeights &w, const GqaCache &c, const bf16 *rope, const SegW &sw,
                    const GqaScratch &sc, bool decode = false) {
  // decode steps: the ESIMD prep at D 128 without gate, rotary over every dim or none (arch flint's and llama's shapes)
  if (s.kv == QsaKv::i8v4 && s.head_dim != kD) throw std::invalid_argument("gqa: int8-v4 KV (4-bit V) is built for head_dim 256");
  if (decode && g_prep_e && s.head_dim == 128 && !s.gate && (s.rot == 128 || s.rot == 0) && !s.rope_rows)
    return s.kv == QsaKv::i8 ? prep_e<QsaKv::i8, 128>(q, s, rows, w, c, rope, sw, sc) : prep_e<QsaKv::bf16, 128>(q, s, rows, w, c, rope, sw, sc);
  switch (s.kv) {
    case QsaKv::i8: return prep<QsaKv::i8>(q, s, rows, w, c, rope, sw, sc);
    case QsaKv::i8v4: return prep<QsaKv::i8v4>(q, s, rows, w, c, rope, sw, sc);
    default: return prep<QsaKv::bf16>(q, s, rows, w, c, rope, sw, sc);
  }
}
sycl::event attn_kv(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const SegW &sw, const GqaScratch &sc, int ns) {
  const bool i8 = s.kv == QsaKv::i8;
  if (s.kv == QsaKv::i8v4) throw std::invalid_argument("gqa_decode: the per-row attention kernel takes int8 | bf16 KV (int8-v4: the group kernel)");
  if (s.head_dim != kD) throw std::invalid_argument("gqa_decode: the per-row attention kernel is built for head_dim 256 (use the group kernel)");
  switch (s.grp()) {
    case 6: return i8 ? attn<QsaKv::i8, 6>(q, s, rows, c, sw, sc, ns) : attn<QsaKv::bf16, 6>(q, s, rows, c, sw, sc, ns);
    case 12: return i8 ? attn<QsaKv::i8, 12>(q, s, rows, c, sw, sc, ns) : attn<QsaKv::bf16, 12>(q, s, rows, c, sw, sc, ns);
    default: throw std::invalid_argument("gqa_decode: the attention kernel is built for 6 or 12 q heads per kv head");
  }
}

// the group attention at 128 GRF (gqa_seg.hpp; the 256-GRF instances are gqa_seg.cpp's)
sycl::event attn_seg128(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const SegW &sw, const GqaScratch &sc, int ns, int rm, bool pf) {
  using namespace gqa_detail;
  const bool i8 = s.kv == QsaKv::i8;
  if (s.kv == QsaKv::i8v4) {  // int8-v4 KV, grp 6 at head_dim 256 (the dense 27B)
    if (s.grp() != 6 || s.head_dim != 256) throw std::invalid_argument("gqa_decode: int8-v4 KV is built for 6 q heads per kv head at head_dim 256");
    const int32_t *rec = sw.rec;
#define XEN_V4(RM, QK, LO) return attn_seg<QsaKv::i8v4, 6, RM, 0, false, 256, QK, LO>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
    if (!s.dec_exact()) {
      if (rm != 1) throw std::invalid_argument("gqa_decode: the decode attention numerics run 1-row groups at 128 GRF");
      if (s.dec_qk == 0) XEN_V4(1, 0, false)
      if (s.dec_qk == 2) XEN_V4(1, 2, false)
      if (s.dec_p_lo) XEN_V4(1, 1, true)
      XEN_V4(1, 1, false)
    }
    if (rm == 1) XEN_V4(1, 0, true)
    if (rm == 2) XEN_V4(2, 0, true)
    if (rm == 4) XEN_V4(4, 0, true)
    XEN_V4(8, 0, true)
#undef XEN_V4
  }
  if (!s.dec_exact()) {  // the decode attention numerics, one row per group (the 128-GRF pick: gqa_attn_group)
    if (rm != 1) throw std::invalid_argument("gqa_decode: the decode attention numerics run 1-row groups at 128 GRF");
    const int32_t *rec = sw.rec;
    if (!i8) return attn_seg<QsaKv::bf16, 6, 1, 0, false, 256, 0, false>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
    if (s.dec_qk == 0) return attn_seg<QsaKv::i8, 6, 1, 0, false, 256, 0, false>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
    if (s.dec_qk == 2) return attn_seg<QsaKv::i8, 6, 1, 0, false, 256, 2, false>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
    return s.dec_p_lo ? attn_seg<QsaKv::i8, 6, 1, 0, false, 256, 1, true>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf)
                      : attn_seg<QsaKv::i8, 6, 1, 0, false, 256, 1, false>(q, s, rows, c, rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
  }
#define XEN_SEG(G, RM, D)                                                                                                  \
  if (s.grp() == G && rm == RM && s.head_dim == D)                                                                         \
    return i8 ? attn_seg<QsaKv::i8, G, RM, 0, false, D>(q, s, rows, c, sw.rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf)  \
              : attn_seg<QsaKv::bf16, G, RM, 0, false, D>(q, s, rows, c, sw.rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, nullptr, nullptr, pf);
  XEN_SEG(6, 1, 256)
  XEN_SEG(6, 2, 256)
  XEN_SEG(6, 4, 256)
  XEN_SEG(6, 8, 256)
  XEN_SEG(12, 1, 256)
  XEN_SEG(12, 2, 256)
  XEN_SEG(12, 4, 256)
  XEN_SEG(7, 1, 128)
  XEN_SEG(7, 2, 128)
  XEN_SEG(7, 4, 128)
  XEN_SEG(7, 8, 128)
  XEN_SEG(8, 1, 128)
  XEN_SEG(8, 2, 128)
  XEN_SEG(8, 4, 128)
  XEN_SEG(8, 8, 128)
#undef XEN_SEG
  throw std::invalid_argument("gqa_decode: attention groups of 1 / 2 / 4 / 8 rows (grp 6, grp 7 / 8 at head_dim 128) or 1 / 2 / 4 (grp 12)");
}

// The attention over a step's selected pages, in the group gqa_attn_group picked: gqa_seg.cpp's 256-GRF
// instances, and for int8-fast 1-row groups the 128-GRF instance, as the dense launch runs them (same bits at
// either register file).
sycl::event attn_sel(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const SegW &sw, const GqaScratch &sc, int ns, const GqaGroup &group) {
  using namespace gqa_detail;
  const int group_rows = group.rows > 0 ? group.rows : 1;
  if (s.dec_int8_fast() && s.kv == QsaKv::i8 && group_rows == 1 && group.grf == 128)
    return attn_seg<QsaKv::i8, 6, 1, 0, true, 256, 1, false>(q, s, rows, c, sw.rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, sc.sel, sc.sel_n, group.pf);
  return gqa_attn_seg_sel_xl(q, s, rows, c, sw.rec, sw.stride, sw.seg_rows, sc.q, sc.part, ns, group_rows, sc.sel, sc.sel_n, group.pf);
}

}  // namespace

// Groups of 8 rows spill at either register file: wider segments run as groups of 4 (grp 6) / 2 (grp 12).
static bool gqa_weights_ok(const GqaShape &s, const GqaWeights &w) {
  if (s.qk_norm && (!w.q_norm || !w.k_norm)) return false;
  // the built shapes: qwen35 (256, gate, (1 + w) norms), llama (128, no gate, no norms, rot 128), flint (128, no gate,
  // plain norms, rot 128 or 0, window, sinks)
  if (s.head_dim != kD && (s.head_dim != 128 || s.gate || (s.qk_norm && s.norm_plus1) || (s.rot != 128 && s.rot != 0))) return false;
  if (s.window < 0 || (s.window > 0 && (s.window + kStepRows8 - 2) / kTile + 2 > kGqaWinSplits)) return false;  // a group's tiles fit the splits
  if (w.dense())
    return w.o_d.valid() && w.qkv_d.n >= s.qkv_n() && w.qkv_d.k == s.hidden && w.o_d.n == s.hidden && w.o_d.k == s.o_k();
  return w.qkv.valid() && w.o.valid() && w.qkv.n >= s.qkv_n() && w.qkv.k == s.hidden && w.o.n == s.hidden && w.o.k == s.o_k();
}

GqaEvents gqa_kv_rows(sycl::queue &q, const GqaShape &s, int m, const bf16 *x, const GqaWeights &w, const GqaCache &c, const bf16 *rope,
                      const int32_t *pos4, int pos0, const GqaRowsScratch &sc) {
  if (m < 1 || pos0 < 0) throw std::invalid_argument("gqa_kv_rows: rows / position");
  if (!gqa_weights_ok(s, w)) throw std::invalid_argument("gqa_kv_rows: weight shapes do not match the shape");
  if (c.max_tokens % kPage || pos0 + m > c.max_tokens) throw std::invalid_argument("gqa_kv_rows: the rows run past the cache");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("gqa_kv_rows: int8 KV needs scale pools");
  GqaEvents e;
  const int ldy = w.qkv_n();
  if (w.dense()) e.qkv = pf_dense(q, m, w.qkv_d, x, s.hidden, PfOut{sc.y, nullptr, ldy});
  else e.qkv = w4a16_gemm_pf(q, m, x, s.hidden, w.qkv.w, w.qkv.scales, sc.y, ldy, w.qkv.n, w.qkv.k);
  e.prep = prep_kv(q, s, m, w, c, rope, SegW{pos4, 0, 0}, GqaScratch{sc.y, sc.q, sc.part, sc.attn, sc.partial});
  if (c.kmin) gqa_bounds_range(q, s, c, pos0, m);
  return e;
}

GqaGroup gqa_attn_group(const GqaShape &s, int seg, int pos_bound) {
  // flint (grp 8, D 128): 1-row groups at 128 GRF for one row at any context and for windowed
  // segments up to 4 rows (the window's K / V sits in L2), else 2-row groups at 128 GRF (an 8-row group spills)
  if (s.head_dim == 128 && s.grp() == 8) {
    if (seg <= 1 || (s.window > 0 && seg <= 4)) return {1, 128};
    return {2, 128};
  }
  if (s.head_dim == 128) {  // llama (grp 7): o is half the size of qwen35's at the same rows
    if (seg <= 1) return {1, pos_bound >= 32768 ? 256 : 128};
    if (seg <= 2) return {2, 256};
    if (seg <= 4) return {4, 256};
    return {8, 256};
  }
  // The decode attention numerics run 1-row groups at 128 GRF at every context. The L2 prefetch of the next tile
  // (gqa_seg.hpp pf, same bits): segments of 5-8 rows (two 4-row groups: the second reads the first's prefetched
  // tiles) from 32K in every mode; segments of 2-4 rows with int8 q.k (dec_qk 1) from 131K, with hi + lo q codes
  // (dec_qk 2) from 196K; exact q.k never on 2-4 rows.
  const bool pf_seg = (seg > 4 && pos_bound >= 32768) || (seg > 1 && ((s.dec_qk == 1 && pos_bound >= 131072) || (s.dec_qk == 2 && pos_bound >= 196608)));
  if (seg <= 1) return {1, pos_bound >= 32768 && s.dec_exact() ? 256 : 128, false};
  // bf16 KV, 3-4 rows below 64K: 2-row groups (4-row groups from there); 8-row segments keep 4-row groups
  const bool bf2 = s.kv == QsaKv::bf16 && seg <= 4 && pos_bound < 65536;
  if (seg <= 2 || s.grp() > 6 || bf2) return {2, 256, pf_seg && s.grp() == 6};
  return {4, 256, pf_seg};
}

size_t gqa_page_kv_bytes(const GqaShape &s) { return static_cast<size_t>(s.hkv()) * kPage * s.head_dim * s.kv_elem(); }
size_t gqa_page_scale_floats(const GqaShape &s) { return static_cast<size_t>(s.hkv()) * kPage; }
size_t gqa_page_v_bytes(const GqaShape &s) { return static_cast<size_t>(s.hkv()) * kPage * s.v_row_bytes(); }
size_t gqa_page_vscale_floats(const GqaShape &s) { return static_cast<size_t>(s.hkv()) * kPage * s.v_scale_bytes() / 4; }
size_t gqa_page_total_bytes(const GqaShape &s) {
  return gqa_page_kv_bytes(s) + gqa_page_v_bytes(s) + (qsa_kv_scaled(s.kv) ? (gqa_page_scale_floats(s) + gqa_page_vscale_floats(s)) * 4 : 0);
}
size_t gqa_part_floats(const GqaShape &s, int rows) { return static_cast<size_t>(rows) * s.hkv() * kGqaSplits * s.grp() * (s.head_dim + 16); }
int gqa_splits(int p) {
  const int ntiles = (p + 1 + kTile - 1) / kTile;
  return cut_nsp(ntiles, cut_tps(ntiles));
}

sycl::event gqa_pack_trained_qsa(sycl::queue &q, int rows, int hkv, const GqaStep &step,
                                const int32_t *sel, const int32_t *sel_n,
                                int32_t *entries, int32_t *counts, int32_t *status, uint32_t dense_rows, bool gather) {
  if (!q.is_in_order() || rows < 1 || rows > kMaxRows || hkv < 1 || hkv > 4 || !step.pos || step.stride < kSegRec ||
      (step.tree && seg_size(rows,step.seg_rows)>kTreeRows) || !seg_rows_ok(rows, step.seg_rows, step.stride) ||
      !sel || !sel_n || !entries || !counts || !status)
    throw std::invalid_argument("gqa_pack_trained_qsa: expected valid records, in-order queue and selection buffers");
  const int32_t *rec = step.pos;
  const int stride = step.stride, seg_rows = step.seg_rows;
  const bool tree = step.tree;
  // One work-group per row, one item per sequence element (the <= 512 ascending selected blocks, then the
  // mandatory own block / ancestor-path blocks): validity is a group any-of, a page entry starts where the page
  // changes (group scan for its index), and the starting item ORs the <= 16 blocks of its page.
  constexpr int kWg = 1024;
  static_assert(kTrainedQsaTopBlocks + 16 <= kWg);
  return q.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * kWg, kWg), [=](sycl::nd_item<1> it) {
    const auto grp = it.get_group();
    const int r = static_cast<int>(it.get_group(0)), li = static_cast<int>(it.get_local_id(0));
    const SegRow sr = seg_row(rec, stride, seg_rows, r);
    const int pos = sr.pos;
    // group-uniform: mode 0 = decided (packed), 1 = compact
    int packed = -1, mode = 0, n = 0, total = 0, limit = 0, extra0 = 0;
    const int32_t *ids = sel + static_cast<size_t>(r) * kTrainedQsaTopBlocks;
    if (pos >= 0 && pos < kQsaMaxTokens) {
      if (r < 32 && ((dense_rows >> r) & 1u)) packed = 0;
      else if ((pos + 1) / kQsaCompress <= kTrainedQsaTopBlocks) packed = 0;
      else if (tree) {
        const int start = rec[sr.seg * stride + kSegPos], vpos = seg_vpos(rec, stride, sr);
        n = sel_n[r];
        const int prior = start / kQsaCompress, last = vpos / kQsaCompress;
        if (start < 0 || vpos < start || vpos >= kQsaMaxTokens || vpos - start >= 16 ||
            n != (prior < kTrainedQsaTopBlocks ? prior : kTrainedQsaTopBlocks)) packed = -1;
        else mode = 1, limit = prior, extra0 = prior, total = n + last - prior + 1;
      } else {
        n = sel_n[r];
        const int complete = (pos + 1) / kQsaCompress, own = pos / kQsaCompress;
        if (n != (complete < kTrainedQsaTopBlocks ? complete : kTrainedQsaTopBlocks)) packed = -1;
        else {
          const int prev = n > 0 ? ids[n - 1] : -1;  // checked below; a malformed list fails the any-of
          const int add_own = prev != own;
          mode = 1, limit = complete, extra0 = own, total = n + add_own;
          if (n + add_own == own + 1) mode = 2;  // fully visible unless the list is malformed
        }
      }
    }
    // validity (every item of the group takes part in the collectives)
    bool bad = false;
    if (mode && li < n) bad = ids[li] >= limit || ids[li] < 0 || (li > 0 && ids[li] <= ids[li - 1]);
    const bool any_bad = sycl::any_of_group(grp, bad);
    if (mode && any_bad) mode = 0, packed = -1;
    if (mode == 2) mode = 0, packed = 0;
    if (mode == 1 && tree && n == limit) mode = 0, packed = 0;  // every committed block and the path are visible
    auto blk = [&](int i) { return i < n ? ids[i] : extra0 + (i - n); };
    const bool in = mode == 1 && li < total;
    const int b = in ? blk(li) : 0, page = b / kQsaPageBlocks;
    const bool first = in && (li == 0 || blk(li - 1) / kQsaPageBlocks != page);
    const int at = sycl::exclusive_scan_over_group(grp, first ? 1 : 0, sycl::plus<int>());
    const int count = sycl::reduce_over_group(grp, first ? 1 : 0, sycl::plus<int>());
    int32_t *dst = entries + static_cast<size_t>(r) * kTrainedQsaPages;
    if (first) {
      uint32_t mask = 0;
      for (int j = li; j < total; ++j) {
        const int bj = blk(j);
        if (bj / kQsaPageBlocks != page) break;
        mask |= 1u << (bj % kQsaPageBlocks);
      }
      dst[at] = static_cast<int32_t>((static_cast<uint32_t>(page) << 16) | mask);
    }
    if (mode == 1 && gather) {
      // gather plan: the n selected ids after the entries, ceil(n / 16) gather tiles, then the extra blocks
      // (own block / tree path) as page entries; the page-start entries above are overwritten
      sycl::group_barrier(grp);
      const int ng = (n + 15) / 16;
      if (li < n) dst[kTrainedQsaPlanEntries + li] = ids[li];
      if (li < ng) {
        const int nb = n - 16 * li < 16 ? n - 16 * li : 16;
        dst[li] = kTrainedQsaGather | (li << 16) | static_cast<int32_t>(nb == 16 ? 0xffffu : (1u << nb) - 1u);
      }
      int ne = ng;
      if (li == 0)
        for (int i = n; i < total; ++i) {
          const int bi = blk(i), pi = bi / kQsaPageBlocks, bit = 1 << (bi % kQsaPageBlocks);
          if (ne == ng || trained_qsa_page(dst[ne - 1]) != pi) dst[ne++] = pi << 16;
          dst[ne - 1] |= bit;
        }
      packed = sycl::group_broadcast(grp, ne, 0);
    } else if (mode == 1) packed = count;
    if (li == 0) {
      status[r] = packed;
      for (int h = 0; h < hkv; ++h) counts[r * hkv + h] = packed < 0 ? 0 : packed;
    }
  });
}

GqaEvents gqa_decode(sycl::queue &q, const GqaShape &s, int rows, const bf16 *x, const GqaWeights &w, const GqaCache &c, const bf16 *rope,
                     const GqaStep &step, const GqaScratch &sc, const PairWire *wire, int wire_parity, bf16 *out, int qkv_ks, int o_ks,
                     const LoraLayer *lora, LoraScratch *lora_sc) {
  if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("gqa_decode: rows 1..32");
  if (s.q_heads % s.kv_heads || s.q_heads % s.cards || s.kv_heads % s.cards || s.hidden % 64)
    throw std::invalid_argument("gqa_decode: heads must split over the cards and kv heads, hidden % 64 == 0");
  if (!gqa_weights_ok(s, w)) throw std::invalid_argument("gqa_decode: weight shapes do not match the shape");
  if (c.max_tokens % kPage || c.max_tokens > kQsaMaxTokens || step.pos_bound < rows || step.pos_bound > c.max_tokens)
    throw std::invalid_argument("gqa_decode: pos_bound must lie in [rows, cache.max_tokens], capacity a multiple of 64 up to 1,048,576");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gqa_decode: a wire is needed exactly with two cards");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("gqa_decode: int8 KV needs scale pools");
  // seg_het layouts too (k segments of B rows, then one-row segments; seg_span / seg_row)
  if (!seg_rows_ok(rows, step.seg_rows, step.stride) || (step.seg_rows > 0 && step.stride < kSegRec))
    throw std::invalid_argument("gqa_decode: segments need records (stride >= kSegRec) and 1..rows rows each (or a seg_het layout)");
  const SegW sw{step.pos, step.stride, step.seg_rows, w.sinks};
  const bool trained = step.trained_qsa.entries != nullptr;
  if (trained != (step.trained_qsa.counts != nullptr) ||
      (trained && (s.kv != QsaKv::i8v4 || s.grp() != 6 || s.head_dim != 256 || !s.gate || s.split_merge ||
                   s.window || c.ring_pages || !c.pages || c.slot_pages < c.max_tokens / kPage || !step.pos ||
                   step.stride < kSegRec || (step.tree && seg_size(rows,step.seg_rows)>kTreeRows) ||
                   step.kv_only || step.select || step.select_slots || step.attn_rows < 0)))
    throw std::invalid_argument("gqa_decode: trained QSA needs gated K8V4, explicit pooled slot records, tree segments of at most 16 rows, no window/other selection");
  if (c.ring_pages > 0 && (s.window <= 0 || c.ring_pages < (s.window + rows - 2) / kTile + 2))
    throw std::invalid_argument("gqa_decode: a KV ring needs a window and ring_pages >= (window + rows - 2) / 64 + 2");
  if (s.window > 0 && (step.select > 0 || step.attn_rows < 0)) throw std::invalid_argument("gqa_decode: a window runs on the group kernel without selection");
  if (!s.dec_exact()) {  // the decode attention numerics (GqaShape::dec_qk / dec_p_lo)
    if (s.grp() != 6 || s.head_dim != kD || s.window > 0) throw std::invalid_argument("gqa_decode: the decode attention numerics are built for grp 6 at head_dim 256");
    if (s.dec_qk < 0 || s.dec_qk > 2 || (s.dec_qk && !qsa_kv_scaled(s.kv))) throw std::invalid_argument("gqa_decode: dec_qk 0 / 1 / 2, int8 q.k on the int8 KV");
    if (s.dec_qk == 2 && s.dec_p_lo) throw std::invalid_argument("gqa_decode: dec_qk 2 runs with P hi only");
    if (step.attn_rows < 0) throw std::invalid_argument("gqa_decode: the decode attention numerics run on the group kernel");
    if (step.select > 0 && !s.dec_int8_fast())  // the selected instances are built for exact and int8-fast
      throw std::invalid_argument("gqa_decode: page selection runs the exact or the int8-fast decode attention numerics");
  }
  // tree segments (records' kSegTree) run on the dense group kernel, with the decode attention numerics too (the
  // chain instances' side-row units read their path's K codes / V rows)
  if (step.tree && (step.stride < kSegRec || step.select > 0 || step.attn_rows < 0 || s.window > 0 || s.head_dim != kD ||
                    seg_size(rows, step.seg_rows) > kTreeRows))
    throw std::invalid_argument("gqa_decode: a tree step runs on records, the dense group kernel, int8 / int8-v4 / bf16 KV at head_dim 256, <= 16 rows per segment");
  // grid splits: the most any row can use (positions < pos_bound); idle work-groups return at once. Windowed: one per
  // tile slot (kGqaWinSplits)
  const int ns = s.window > 0 ? kGqaWinSplits : std::min(kGqaSplits, (step.pos_bound + kTile - 1) / kTile);
  GqaEvents e;
  e.qkv = w.dense() ? dense_gemm(q, rows, x, w.qkv_d, sc.y, qkv_ks) : w4a16_gemm_rows(q, rows, x, w.qkv.w, w.qkv.scales, sc.y, w.qkv.n, w.qkv.k, qkv_ks);
  if (lora) {  // the q | k | v deltas into the fp32 qkv projection (before the norms / rope); the B rows sit at the
    // q | k | v offsets of sc.y (out_off), the row stride w.qkv_n()
    if (lora->q.present()) e.qkv = lora_apply(q, rows, x, lora->q, sc.y, w.qkv_n(), *lora_sc, 0);
    if (lora->k.present()) e.qkv = lora_apply(q, rows, x, lora->k, sc.y, w.qkv_n(), *lora_sc, 0);
    if (lora->v.present()) e.qkv = lora_apply(q, rows, x, lora->v, sc.y, w.qkv_n(), *lora_sc, 0);
  }
  e.prep = prep_kv(q, s, rows, w, c, rope, sw, sc, true);
  if (step.kv_only) return e;
  // page selection once a segment's context can pass the budget (below it every segment keeps all
  // its pages: the dense launch, bit for bit)
  const bool sel = step.select > 0 && step.pos_bound > step.select * kPage;
  if (step.select > 0 && s.head_dim != kD) throw std::invalid_argument("gqa_decode: page selection is built for head_dim 256");
  if (sel && (!c.kmin || !sc.sel_key)) throw std::invalid_argument("gqa_decode: page selection needs the cache's bounds and the selection scratch");
  if (sel) gqa_select(q, s, rows, c, step.pos, step.stride, step.seg_rows, sc.q, step.select, step.pos_bound, sc, step.select_slots);
  {
    const int seg = seg_size(rows, step.seg_rows);
    GqaGroup gg = gqa_attn_group(s, seg, step.pos_bound);
    if (step.attn_rows != 0) gg.rows = step.attn_rows;
    if (step.attn_grf > 0) gg.grf = step.attn_grf;
    if (step.attn_pf >= 0) gg.pf = step.attn_pf != 0;
    if (trained) e.attn = gqa_attn_seg_trained_xl(q, s, rows, c, step, sc.q, sc.part, ns);
    else if (sel) e.attn = attn_sel(q, s, rows, c, sw, sc, ns, gg);
    else if (gg.rows < 0) e.attn = attn_kv(q, s, rows, c, sw, sc, ns);
    else if (gg.grf == 128) e.attn = attn_seg128(q, s, rows, c, sw, sc, ns, gg.rows, gg.pf);
    else e.attn = gqa_attn_seg_xl(q, s, rows, c, step.pos, step.stride, step.seg_rows, sc.q, sc.part, ns, gg.rows, gg.pf, step.tree);
  }
  e.merge = trained ? merge_d<256, true, true>(q, s, rows, w.qkv_n(), sw, sc, step.trained_qsa.counts)
                    : merge(q, s, rows, w.qkv_n(), sw, sc, sel ? sc.sel_n : nullptr);
  e.o = w.dense() ? dense_gemm(q, rows, sc.attn, w.o_d, sc.partial, o_ks) : w4a16_gemm_rows(q, rows, sc.attn, w.o.w, w.o.scales, sc.partial, w.o.n, w.o.k, o_ks);
  if (lora && lora->o.present()) e.o = lora_apply(q, rows, sc.attn, lora->o, sc.partial, s.hidden, *lora_sc, 0);  // the o delta before the sum
  e.reduce = pair_reduce(q, rows, s.hidden, sc.partial, wire, wire_parity, out);
  return e;
}

// ---- draft trees: the accepted path's K / V rows to their slots ----
namespace {
class GqaKvMoveK;
struct MoveBufs {
  uint8_t *k[kGqaMoveCaches], *v[kGqaMoveCaches];
  float *ks[kGqaMoveCaches], *vs[kGqaMoveCaches];
  const int32_t *pages[kGqaMoveCaches];
};
}  // namespace

sycl::event gqa_kv_move(sycl::queue &q, const GqaShape &s, const GqaCache *caches, int n_caches, int slot, int p0, const GqaMove &mv) {
  if (n_caches < 1 || n_caches > kGqaMoveCaches || mv.n < 1 || mv.n > kTreeRows || slot < 0)
    throw std::invalid_argument("gqa_kv_move: 1..16 caches, 1..16 rows, int8 / int8-v4 / bf16 KV");
  for (int i = 0; i < mv.n; ++i)
    if (mv.to[i] < 0 || mv.from[i] < mv.to[i] || mv.from[i] >= kTreeRows || (i > 0 && (mv.to[i] <= mv.to[i - 1] || mv.from[i] <= mv.from[i - 1])))
      throw std::invalid_argument("gqa_kv_move: ascending sources and destinations, each destination at or below its source");
  MoveBufs b{};
  for (int l = 0; l < n_caches; ++l) {
    const GqaCache &c = caches[l];
    if (c.ring_pages > 0 || p0 < 0 || p0 + mv.from[mv.n - 1] >= c.max_tokens) throw std::invalid_argument("gqa_kv_move: no rings, rows inside the capacity");
    b.k[l] = c.k, b.v[l] = c.v, b.ks[l] = c.ks, b.vs[l] = c.vs, b.pages[l] = c.pages + static_cast<size_t>(slot) * c.slot_pages;
  }
  const int hkv = s.hkv(), krow = s.head_dim * s.kv_elem(), vrow = s.v_row_bytes();
  const int kch = krow / 16, vch = vrow / 16, ch = kch > vch ? kch : vch;
  const int vsc = s.v_scale_bytes() / 4;  // floats per token-head: 1 (int8) or 4 (int8-v4: 8 bf16 scales)
  const bool scaled = qsa_kv_scaled(s.kv);
  return q.parallel_for<GqaKvMoveK>(sycl::range<3>(static_cast<size_t>(n_caches), static_cast<size_t>(hkv), static_cast<size_t>(ch)), [=](sycl::item<3> it) {
    const int l = static_cast<int>(it[0]), h = static_cast<int>(it[1]), c = static_cast<int>(it[2]);
    auto slot_of = [&](int p) { return (static_cast<size_t>(b.pages[l][p / kPage]) * hkv + h) * kPage + p % kPage; };
    for (int i = 0; i < mv.n; ++i) {
      const size_t src = slot_of(p0 + mv.from[i]), dst = slot_of(p0 + mv.to[i]);
      if (src == dst) continue;
      if (c < kch) reinterpret_cast<sycl::uint4 *>(b.k[l] + dst * krow)[c] = reinterpret_cast<const sycl::uint4 *>(b.k[l] + src * krow)[c];
      if (c < vch) reinterpret_cast<sycl::uint4 *>(b.v[l] + dst * vrow)[c] = reinterpret_cast<const sycl::uint4 *>(b.v[l] + src * vrow)[c];
      if (scaled && c == 0) {
        b.ks[l][dst] = b.ks[l][src];
        for (int j = 0; j < vsc; ++j) b.vs[l][dst * static_cast<size_t>(vsc) + j] = b.vs[l][src * static_cast<size_t>(vsc) + j];
      }
    }
  });
}

// ---- the oneDNN attention body (gqa_attn_sdpa); the oneDNN call is gqa_sdpa_exec (gqa_sdpa.cpp) ----
namespace {

using half = sycl::half;

size_t align4k(size_t b) { return (b + 4095) / 4096 * 4096; }

template <bool I8>
class SdpaGatherK;
// One item per context token: kv head h's K and V rows (256 each) to f16 at [t] and [ld + t]; items past the context
// convert q (one (row, head) of 256 per item) when qn > 0.
template <bool I8>
sycl::event sdpa_gather(sycl::queue &q, const GqaShape &s, const GqaCache &c, int h, int S, half *kv, int ld, const bf16 *qb, half *q16, int qn) {
  const int hkv = s.hkv();
  const uint8_t *K = c.k, *V = c.v;
  const float *KS = c.ks, *VS = c.vs;
  const int32_t *P = c.pages;
  const size_t items = static_cast<size_t>(S) + static_cast<size_t>(qn);
  return q.parallel_for<SdpaGatherK<I8>>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t id = it.get_global_id(0);
    if (id < static_cast<size_t>(S)) {
      const int t = static_cast<int>(id);
      const size_t slot = (static_cast<size_t>(P[t / kPage]) * hkv + h) * kPage + t % kPage;
      half *ko = kv + static_cast<size_t>(t) * kD, *vo = kv + (static_cast<size_t>(ld) + t) * kD;
      if constexpr (I8) {
        const float ks = KS[slot], vs = VS[slot];
        const esimd::simd<int8_t, kD> kc = esimd::block_load<int8_t, kD>(reinterpret_cast<const int8_t *>(K) + slot * kD);
        const esimd::simd<int8_t, kD> vc = esimd::block_load<int8_t, kD>(reinterpret_cast<const int8_t *>(V) + slot * kD);
        esimd::block_store<half, kD>(ko, esimd::simd<half, kD>(esimd::simd<float, kD>(kc) * ks));
        esimd::block_store<half, kD>(vo, esimd::simd<half, kD>(esimd::simd<float, kD>(vc) * vs));
      } else {
        const bf16 *kb = reinterpret_cast<const bf16 *>(K) + slot * kD, *vb = reinterpret_cast<const bf16 *>(V) + slot * kD;
        esimd::block_store<half, kD>(ko, esimd::simd<half, kD>(esimd::simd<float, kD>(esimd::block_load<bf16, kD>(kb))));
        esimd::block_store<half, kD>(vo, esimd::simd<half, kD>(esimd::simd<float, kD>(esimd::block_load<bf16, kD>(vb))));
      }
    } else if (id < items) {
      const size_t r = id - static_cast<size_t>(S);
      esimd::block_store<half, kD>(q16 + r * kD, esimd::simd<half, kD>(esimd::simd<float, kD>(esimd::block_load<bf16, kD>(qb + r * kD))));
    }
  });
}

class SdpaGateK;
// attn_fa's finish on o (f16): attn = bf16(bf16(o) x bf16(1 / (1 + exp(-bf16(gate))))); two (row, head) blocks of 256 per
// item, every load of a block first
sycl::event sdpa_gate(sycl::queue &q, int m, int hq, const half *o, const float *y, int ldy, bf16 *attn) {
  constexpr int NB = 2;
  const size_t blocks = static_cast<size_t>(m) * hq, items = (blocks + NB - 1) / NB;
  return q.parallel_for<SdpaGateK>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t id = it.get_global_id(0);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const size_t rh = id * NB + b;
      if (rh >= blocks) return;
      const size_t row = rh / hq;
      const int hh = static_cast<int>(rh % hq);
      const esimd::simd<half, kD> ov = esimd::block_load<half, kD>(o + rh * kD);
      const esimd::simd<float, kD> gv = esimd::block_load<float, kD>(y + row * ldy + static_cast<size_t>(hh) * 2 * kD + kD);
      const esimd::simd<float, kD> a = esimd::simd<bf16, kD>(esimd::simd<float, kD>(ov));
      const esimd::simd<float, kD> gt = esimd::simd<bf16, kD>(gv);
      const esimd::simd<float, kD> sg = esimd::simd<bf16, kD>(esimd::simd<float, kD>(1.f) / (esimd::simd<float, kD>(1.f) + esimd::exp(-gt)));
      esimd::block_store<bf16, kD>(attn + rh * kD, esimd::simd<bf16, kD>(a * sg));
    }
  });
}

}  // namespace

size_t gqa_sdpa_scratch_bytes(const GqaShape &s, int rows_max, int tokens) {
  const size_t qo = static_cast<size_t>(rows_max) * s.hq() * kD * sizeof(half);
  return align4k(2 * static_cast<size_t>(tokens) * kD * sizeof(half)) + 2 * align4k(qo);
}

void gqa_sdpa_scratch_bind(const GqaShape &s, int rows_max, int tokens, void *base, GqaRowsScratch &sc) {
  auto *b = static_cast<uint8_t *>(base);
  const size_t kvb = align4k(2 * static_cast<size_t>(tokens) * kD * sizeof(half));
  const size_t qo = align4k(static_cast<size_t>(rows_max) * s.hq() * kD * sizeof(half));
  sc.sdpa_kv = reinterpret_cast<half *>(b);
  sc.sdpa_q = reinterpret_cast<half *>(b + kvb);
  sc.sdpa_o = reinterpret_cast<half *>(b + kvb + qo);
  sc.sdpa_tokens = tokens;
}

sycl::event gqa_attn_sdpa(sycl::queue &q, const GqaShape &s, int m, int pos0, const GqaCache &c, const float *y, int ldy, const bf16 *qb,
                          bf16 *attn, const GqaRowsScratch &sc) {
  if (s.head_dim != kD || !s.gate || s.window > 0) throw std::invalid_argument("gqa_attn_sdpa: head_dim 256 with the gate, no window");
  if (c.ring_pages > 0) throw std::invalid_argument("gqa_attn_sdpa: no KV ring");
  if (s.kv != QsaKv::i8 && s.kv != QsaKv::bf16) throw std::invalid_argument("gqa_attn_sdpa: int8 or bf16 KV");
  if (m < 1 || pos0 < 0) throw std::invalid_argument("gqa_attn_sdpa: rows / position");
  const int S = pos0 + m, hq = s.hq(), hkv = s.hkv(), grp = s.grp(), ld = sc.sdpa_tokens;
  if (!sc.sdpa_kv || !sc.sdpa_q || !sc.sdpa_o || S > ld)
    throw std::invalid_argument("gqa_attn_sdpa: the sdpa scratch is missing or holds fewer tokens than pos0 + m");
  const bool i8 = s.kv == QsaKv::i8;
  // per kv head: gather (the first head's launch also converts q), oneDNN; the next head's gather reuses the K | V scratch
  // after the previous oneDNN call on the in-order queue
  for (int h = 0; h < hkv; ++h) {
    const int qn = h == 0 ? m * hq : 0;
    if (i8) sdpa_gather<true>(q, s, c, h, S, sc.sdpa_kv, ld, qb, sc.sdpa_q, qn);
    else sdpa_gather<false>(q, s, c, h, S, sc.sdpa_kv, ld, qb, sc.sdpa_q, qn);
    gqa_sdpa_exec(q, m, S, grp, hq, sc.sdpa_q + static_cast<size_t>(h) * grp * kD, sc.sdpa_kv, sc.sdpa_kv + static_cast<size_t>(ld) * kD,
                  sc.sdpa_o + static_cast<size_t>(h) * grp * kD);
  }
  return sdpa_gate(q, m, hq, sc.sdpa_o, y, ldy, attn);
}

GqaEvents gqa_rows(sycl::queue &q, const GqaShape &s, int m, const bf16 *x, const GqaWeights &w, const GqaCache &c, const bf16 *rope,
                   const int32_t *pos4, int pos0, const GqaRowsScratch &sc, const PairRowsWire *wire, int wire_parity, bf16 *out, int attn,
                   int rows_per_group, bool p_lo, int qk, const GqaRowsSelect &select, const GqaTrainedPrefill *trained, const LoraLayer *lora,
                   LoraScratch *lora_sc) {
  if (trained && (attn != kGqaAttnFa || select.pages || w.sinks || qk != 1 || p_lo || gqa_attn_fa_rows(s, rows_per_group) != trained->rows_per_group))
    throw std::invalid_argument("gqa_rows: trained prefill requires the stock K8V4 int8-fast FA shape");
  if (attn < kGqaAttnBatch || attn > kGqaAttnSdpa) throw std::invalid_argument("gqa_rows: attention body 0..4");
  if (select.pages > 0 && attn != kGqaAttnFa) throw std::invalid_argument("gqa_rows: page selection runs the fa attention body (attention 3)");
  if (select.pages > 0 && (!c.kmin || !sc.sel.valid())) throw std::invalid_argument("gqa_rows: page selection needs the cache's key bounds and the page-set scratch");
  if (qk && attn != kGqaAttnFa) throw std::invalid_argument("gqa_rows: int8 q.k is the fa body's (attention 3)");
  if (s.head_dim != kD && attn != kGqaAttnBatch && attn != kGqaAttnFa)  // (kGqaAttnSdpa checks its own shape)
    throw std::invalid_argument("gqa_rows: the staged / block attention bodies are built for head_dim 256 (attention 0 or 3 at 128)");
  const bool block = attn != kGqaAttnBatch;
  if (m < 1 || pos0 < 0) throw std::invalid_argument("gqa_rows: rows / position");
  if (s.q_heads % s.kv_heads || s.q_heads % s.cards || s.kv_heads % s.cards || s.hidden % 64)
    throw std::invalid_argument("gqa_rows: heads must split over the cards and kv heads, hidden % 64 == 0");
  if (w.dense() || !gqa_weights_ok(s, w)) throw std::invalid_argument("gqa_rows: W4 weight shapes matching the shape");
  if (c.max_tokens % kPage || pos0 + m > c.max_tokens) throw std::invalid_argument("gqa_rows: the rows run past the cache");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gqa_rows: a wire is needed exactly with two cards");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("gqa_rows: int8 KV needs scale pools");
  if (qsa_kv_v4(s.kv) && attn != kGqaAttnFa && attn != kGqaAttnBatch)  // the fa body and the decode-kernel batch path
    throw std::invalid_argument("gqa_rows: int8-v4 KV runs the fa body (attention 3) or the batch path (0)");
  const int ldy = w.qkv.n, hq = s.hq(), ok = s.o_k();
  GqaEvents e;
  if (!sc.qkv_ready) {
    e.qkv = w4a16_gemm_pf(q, m, x, s.hidden, w.qkv.w, w.qkv.scales, sc.y, ldy, w.qkv.n, w.qkv.k);
    if (lora) {  // the q | k | v deltas into the fp32 qkv projection (before the norms / rope); the B rows sit at the
      // q | k | v offsets of sc.y (out_off), the row stride ldy
      if (lora->q.present()) e.qkv = lora_apply(q, m, x, lora->q, sc.y, ldy, *lora_sc, 0);
      if (lora->k.present()) e.qkv = lora_apply(q, m, x, lora->k, sc.y, ldy, *lora_sc, 0);
      if (lora->v.present()) e.qkv = lora_apply(q, m, x, lora->v, sc.y, ldy, *lora_sc, 0);
    }
  }
  const GqaScratch all{sc.y, sc.q, sc.part, sc.attn, sc.partial};
  e.prep = prep_kv(q, s, m, w, c, rope, SegW{pos4, 0, 0}, all);
  if (c.kmin) gqa_bounds_range(q, s, c, pos0, m);  // the chunk's pages' key bounds
  const bool fl = s.head_dim == 128 && s.grp() == 8;  // arch flint: the fa body takes the window / ring / sinks
  if (block && (s.window > 0 || w.sinks) && !(fl && attn == kGqaAttnFa))
    throw std::invalid_argument("gqa_rows: a window / sinks run on the batch body (attention 0) or, at grp 8, the fa body (3)");
  if (c.ring_pages > 0 && (s.window <= 0 || c.ring_pages < (s.window + m - 2) / kTile + 2))
    throw std::invalid_argument("gqa_rows: the KV ring must hold the window and the chunk (ring_pages >= (window + m - 2) / 64 + 2)");
  // page sets once the call's context passes the budget (under it every row block would keep all its
  // pages: the dense body, bit for bit)
  const GqaPageSets *page_sets = nullptr;
  if (select.pages > 0 && (pos0 + m + kPage - 1) / kPage > select.pages) {
    gqa_select_rows(q, s, m, pos0, gqa_attn_fa_rows(s, rows_per_group), c, sc.q, select, sc.sel);
    page_sets = &sc.sel;
  }
  if (attn == kGqaAttnSdpa) e.attn = e.merge = gqa_attn_sdpa(q, s, m, pos0, c, sc.y, ldy, sc.q, sc.attn, sc);
  else if (attn == kGqaAttnFa) {
    e.attn = e.merge = trained ? gqa_attn_fa_trained(q, s, m, c, pos4, sc.y, ldy, sc.q, sc.attn, *trained)
                               : gqa_attn_fa(q, s, m, c, pos4, sc.y, ldy, sc.q, sc.attn, rows_per_group, p_lo, qk, fl ? w.sinks : nullptr, page_sets);
  }
  else if (attn == kGqaAttnStaged) e.attn = e.merge = gqa_attn_staged(q, s, m, c, pos4, sc.y, ldy, sc.q, sc.attn, rows_per_group);
  else if (attn == kGqaAttnBlock) e.attn = e.merge = gqa_attn_rows(q, s, m, c, pos4, sc.y, ldy, sc.q, sc.attn);
  for (int b0 = 0; b0 < m && !block; b0 += kGqaRowsBatch) {
    const int rows = std::min(kGqaRowsBatch, m - b0);
    const GqaScratch v{sc.y + static_cast<size_t>(b0) * ldy, sc.q + static_cast<size_t>(b0) * hq * s.head_dim, sc.part, sc.attn + static_cast<size_t>(b0) * ok,
                       sc.partial};
    const SegW sw{pos4 + b0 / 4, 0, 0, w.sinks};
    const int ns = s.window > 0 ? kGqaWinSplits : std::min(kGqaSplits, (pos0 + b0 + rows + kTile - 1) / kTile);
    // the decode path's group attention (a row's bits = a decode step's for the same projections)
    const GqaGroup gg = gqa_attn_group(s, rows, pos0 + b0 + rows);
    e.attn = gg.grf == 128 ? attn_seg128(q, s, rows, c, sw, v, ns, gg.rows, gg.pf) : gqa_attn_seg_xl(q, s, rows, c, sw.rec, 0, 0, v.q, v.part, ns, gg.rows, gg.pf);
    e.merge = merge(q, s, rows, ldy, sw, v);
  }
  if (sc.o_later) return e;  // the caller runs gqa_rows_o over the rows of several calls
  const GqaEvents t = gqa_rows_o(q, s, m, w, sc, wire, wire_parity, out, lora, lora_sc);
  e.o = t.o, e.send = t.send, e.reduce = t.reduce;
  return e;
}

GqaEvents gqa_rows_o(sycl::queue &q, const GqaShape &s, int m, const GqaWeights &w, const GqaRowsScratch &sc, const PairRowsWire *wire, int wire_parity,
                     bf16 *out, const LoraLayer *lora, LoraScratch *lora_sc) {
  if (m < 1) throw std::invalid_argument("gqa_rows_o: rows");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gqa_rows_o: a wire is needed exactly with two cards");
  const int ok = s.o_k();
  GqaEvents e;
  const bool o_lora = lora && lora->o.present();
  auto o_gemm = [&](int lo, int rows) {
    return w4a16_gemm_pf(q, rows, sc.attn + static_cast<size_t>(lo) * ok, ok, w.o.w, w.o.scales, sc.partial + static_cast<size_t>(lo) * s.hidden, s.hidden,
                         w.o.n, w.o.k);
  };
  if (wire && wire->split && wire->send && !o_lora) {  // the peer's rows first, their copy beside the own rows (a LoRA o: the plain sum)
    int lo = 0, hi = 0;
    pair_split_owner_rows(1 - s.card, m, &lo, &hi);
    if (hi > lo) o_gemm(lo, hi - lo);
    e.send = pair_split_send(q, wire->send->side, s.card, m, s.hidden, sc.partial, *wire->send->sp, wire->send->seq);
    pair_split_owner_rows(s.card, m, &lo, &hi);
    if (hi > lo) e.o = o_gemm(lo, hi - lo);
  } else {
    e.o = o_gemm(0, m);
  }
  if (o_lora) e.o = lora_apply(q, m, sc.attn, lora->o, sc.partial, s.hidden, *lora_sc, 0);  // the o delta before the sum
  if (!wire || !wire->split) e.reduce = pair_reduce_rows(q, m, s.hidden, sc.partial, wire, wire_parity, out);  // split: the caller sums sc.partial
  return e;
}

}  // namespace xenithium::kernels
