#include <cstdlib>
// Xenithium · the DFlash 2 drafter's own kernels (see dflash.hpp for the semantics).
#include "xenithium/kernels/drafter/dflash.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/util/gumbel.hpp"

namespace xenithium::kernels {
namespace {

using bf16 = df_bf16;

inline float bfr(float v) { return static_cast<float>(bf16(v)); }

class DfCopyCols;
class DfMeanCols;
class DfCtxStashK;
class DfRowsGatherK;
class DfRecSetK;
class DfConvK;
class DfPrepK;
class DfAttnK;
class DfAttnE;
class DfAttnMergeK;
class DfTopkChunks;
class DfTopkRows;
class DfTopkMerge;
class DfTopkMerge2;
class DfTopkMergeLists2;
class DfTopkMergeLists;
class DfSelectK;
class DfTreeK;

constexpr int kTopkItems = 256;
constexpr int kTopkChunk = 4096;  // columns per stage-1 work-group (16 per item)
constexpr int kTopkPer = kTopkChunk / kTopkItems;
constexpr int kPartStride = kDfDim + 2;  // o[128] | max | sum

// greedy_pick's key (head.cpp): order-preserving 16-bit image of the bf16 logit, ties to the lower index
inline uint32_t ordered_bf16(uint16_t b) {
  if ((b & 0x7fff) > 0x7f80) return 0;
  if (b == 0x8000) b = 0;
  return (b & 0x8000) ? static_cast<uint32_t>(~b & 0xffff) : static_cast<uint32_t>(b | 0x8000);
}
inline uint64_t make_key(float logit, uint32_t index) {
  const bf16 h(logit);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return (static_cast<uint64_t>(ordered_bf16(b)) << 32) | (0xFFFFFFFEu - index);
}

int ctx_chunks(int window) { return (window - 1 + kDfChunk - 1) / kDfChunk; }

}  // namespace

// ---- features -------------------------------------------------------------------------------------
sycl::event df_copy_cols(sycl::queue &q, int rows, const df_bf16 *src, int lds, int src_col, df_bf16 *dst, int ldd, int dst_col, int n) {
  if (rows < 1 || n < 1) throw std::invalid_argument("df_copy_cols: rows / columns");
  return q.parallel_for<DfCopyCols>(sycl::range<1>(static_cast<size_t>(rows) * n), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]) / n, i = static_cast<int>(id[0]) % n;
    dst[static_cast<size_t>(r) * ldd + dst_col + i] = src[static_cast<size_t>(r) * lds + src_col + i];
  });
}

sycl::event df_mean_cols(sycl::queue &q, int rows, const df_bf16 *src, int streams, int hidden, int src_col, df_bf16 *dst, int ldd, int dst_col, int n) {
  if (rows < 1 || n < 1 || streams < 1 || src_col < 0 || src_col + n > hidden) throw std::invalid_argument("df_mean_cols: rows / columns");
  const float inv = 1.f / static_cast<float>(streams);  // exact for 4 (a power of two); the division's rounding otherwise
  const bool pow2 = (streams & (streams - 1)) == 0;
  return q.parallel_for<DfMeanCols>(sycl::range<1>(static_cast<size_t>(rows) * n), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]) / n, i = static_cast<int>(id[0]) % n;
    const df_bf16 *p = src + static_cast<size_t>(r) * streams * hidden + src_col + i;
    float acc = 0.f;
    for (int s = 0; s < streams; ++s) acc += static_cast<float>(p[static_cast<size_t>(s) * hidden]);
    dst[static_cast<size_t>(r) * ldd + dst_col + i] = df_bf16(pow2 ? acc * inv : acc / static_cast<float>(streams));
  });
}

// ---- conv -----------------------------------------------------------------------------------------
sycl::event df_conv(sycl::queue &q, int rows, int seg_rows, int hidden, int group, const df_bf16 *x, const float *dyn, int ld_dyn, int which,
                    const df_bf16 *base, df_bf16 *out) {
  if (rows < 1 || hidden < 1 || group < 1 || hidden % group || (which != 0 && which != 1) || seg_rows < 0)
    throw std::invalid_argument("df_conv: shape");
  const int G = hidden / group, sr = seg_rows > 0 ? seg_rows : rows;
  return q.parallel_for<DfConvK>(sycl::range<1>(static_cast<size_t>(rows) * hidden), [=](sycl::id<1> id) {
    const int t = static_cast<int>(id[0]) / hidden, c = static_cast<int>(id[0]) % hidden, g = c / group;
    const int t0 = t / sr * sr;
    const float x0 = static_cast<float>(x[static_cast<size_t>(t) * hidden + c]);
    const float x1 = t > t0 ? static_cast<float>(x[static_cast<size_t>(t - 1) * hidden + c]) : 0.f;
    const float *dr = dyn + static_cast<size_t>(t) * ld_dyn;
    const float d0 = bfr(dr[(which * 2 + 0) * G + g]), d1 = bfr(dr[(which * 2 + 1) * G + g]);
    const float b0 = static_cast<float>(base[c]), b1 = static_cast<float>(base[hidden + c]);
    float o = bfr(b0 * x0);
    o = bfr(o + d0 * x0);
    o = bfr(o + bfr(b1 * x1));
    o = bfr(o + d1 * x1);
    out[static_cast<size_t>(t) * hidden + c] = bf16(o);
  });
}

void df_conv_ref(int rows, int seg_rows, int hidden, int group, const uint16_t *x, const float *dyn, int ld_dyn, int which, const uint16_t *base,
                 uint16_t *out) {
  const int G = hidden / group, sr = seg_rows > 0 ? seg_rows : rows;
  auto r16 = [](double v) { return bf16_value(bf16_bits(static_cast<float>(v))); };
  for (int t = 0; t < rows; ++t)
    for (int c = 0; c < hidden; ++c) {
      const int t0 = t / sr * sr, g = c / group;
      const float x0 = bf16_value(x[static_cast<size_t>(t) * hidden + c]);
      const float x1 = t > t0 ? bf16_value(x[static_cast<size_t>(t - 1) * hidden + c]) : 0.f;
      const float d0 = r16(dyn[static_cast<size_t>(t) * ld_dyn + (which * 2 + 0) * G + g]);
      const float d1 = r16(dyn[static_cast<size_t>(t) * ld_dyn + (which * 2 + 1) * G + g]);
      float o = r16(static_cast<double>(bf16_value(base[c])) * x0);
      o = r16(static_cast<double>(o) + static_cast<double>(d0) * x0);
      o = r16(static_cast<double>(o) + r16(static_cast<double>(bf16_value(base[hidden + c])) * x1));
      o = r16(static_cast<double>(o) + static_cast<double>(d1) * x1);
      out[static_cast<size_t>(t) * hidden + c] = bf16_bits(o);
    }
}

// ---- q / k / v prep -------------------------------------------------------------------------------
sycl::event df_prep(sycl::queue &q, int rows, int seg_rows, const int32_t *rec, int hq, int hkv, const float *yq, const float *ykv, const df_bf16 *qn,
                    const df_bf16 *kn, float eps, const DfRope &rope, df_bf16 *qo, df_bf16 *kb, df_bf16 *vb, const DfRing &ring) {
  const bool block = yq != nullptr;
  if (rows < 1 || hkv < 1 || (block && (hq < hkv || !qo || !kb || !vb)) || (!block && (!ring.k || !ring.v || ring.ring < 1 || ring.hkv != hkv)) || !ykv ||
      !rope.cos)
    throw std::invalid_argument("df_prep: shape / buffers");
  const int sr = seg_rows > 0 ? seg_rows : rows, heads = (block ? hq : 0) + 2 * hkv, D = kDfDim, R = ring.ring;
  const int ldq = hq * D, ldkv = 2 * hkv * D;
  const bf16 *cs = rope.cos, *sn = rope.sin;
  bf16 *rk = ring.k, *rv = ring.v;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sh(sycl::range<1>(D), h);
    h.parallel_for<DfPrepK>(sycl::nd_range<1>(static_cast<size_t>(rows) * heads * D, D), [=](sycl::nd_item<1> it) {
      const int g = static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
      const int r = g / heads, hs = g % heads, seg = r / sr, j = r % sr;
      const int32_t *w = rec + seg * kSegRec;
      const int pos = (block ? w[kDfRecStart] : w[kDfRecCtx]) + j;
      const int slot = w[kDfRecSlot];
      // which head: q heads [0, hq) (block only), k heads, v heads
      const int qh = block ? hq : 0;
      const bool isq = hs < qh, isk = !isq && hs < qh + hkv;
      const int hh = isq ? hs : (isk ? hs - qh : hs - qh - hkv);
      const float xv = isq ? bfr(yq[static_cast<size_t>(r) * ldq + hh * D + d]) : bfr(ykv[static_cast<size_t>(r) * ldkv + (isk ? hh : hkv + hh) * D + d]);
      float o = xv;
      if (isq || isk) {
        const float ss = sycl::reduce_over_group(it.get_group(), xv * xv, sycl::plus<float>());
        const float rr = 1.f / sycl::sqrt(ss / static_cast<float>(D) + eps);
        const float wv = static_cast<float>((isq ? qn : kn)[d]);
        o = bfr(wv * bfr(xv * rr));
        sh[d] = o;
        sycl::group_barrier(it.get_group());
        const int i = d % (D / 2);
        const float c = static_cast<float>(cs[static_cast<size_t>(pos) * (D / 2) + i]), s = static_cast<float>(sn[static_cast<size_t>(pos) * (D / 2) + i]);
        const float rot = d < D / 2 ? -sh[d + D / 2] : sh[d - D / 2];
        o = bfr(bfr(o * c) + bfr(rot * s));
      }
      if (block) {
        if (isq) qo[(static_cast<size_t>(r) * hq + hh) * D + d] = bf16(o);
        else if (isk) kb[(static_cast<size_t>(r) * hkv + hh) * D + d] = bf16(o);
        else vb[(static_cast<size_t>(r) * hkv + hh) * D + d] = bf16(o);
      } else {
        const size_t at = ((static_cast<size_t>(slot) * R + static_cast<size_t>(pos % R)) * hkv + hh) * D + d;
        if (isk) rk[at] = bf16(o);
        else rv[at] = bf16(o);
      }
    });
  });
}

// ---- attention, the ESIMD form ---------------------------------------------------------------------
// One launch, a work-group per (segment, kv head, 8 queries = block rows x q heads of the group), its items = (key chunk,
// half of the 128 dims): 14 chunks of the window's 16-key blocks + the block's own keys.
//   scores  S[8 q][16 keys]: bf16 DPAS, A = the queries (copied to SLM once), B = K^T of 16 keys x 16 dims by one
//           transposed u32 2D load (a u32 = a pair of dims: the DPAS operand's layout as it comes off the load)
//   softmax online over the blocks (row max / sum by stride-2 folds; fp32, the running max rescales O and the sum)
//   P . V   O[8 q][16 d] += bf16 DPAS, A = P split in bf16 hi + lo (two chains: P to 16 bits), B = V of 16 keys x 16
//           dims by one "transformed" 2D load (row pairs interleaved: the operand's layout again)
//   merge   the chunks' (O, max, sum) meet in SLM (4 KB per chunk); one item per query writes its bf16 row
// An item takes half of the dims because one item per chunk over all 128 dims spills. Against the work-group form
// below (a group per 32-key chunk, 3 barriers, fp32 dots over SLM, then a merge launch) output words may differ by an
// ulp: both are fp32 sums in their own order. Needs the ring a multiple of 16 keys and K / V bases 64-byte aligned
// (the 2D loads).
namespace {
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
constexpr int kD = kDfDim, kQ = 8, kCtxChunks = 14, kNC = kCtxChunks + 1, kHalves = 2, kAeItems = kNC * kHalves;
constexpr int kTiles = kD / 16 / kHalves;                  // d tiles per item
constexpr uint32_t kSlmQ = 0;                               // [call 8][q 8][16 d] bf16: 2 KB
constexpr uint32_t kSlmO = 2048;                            // [chunk][d tile 8][q 8][16 d] fp32: 4 KB per chunk
constexpr uint32_t kSlmML = kSlmO + kNC * 4096;             // [chunk][max 8 | sum 8] fp32
constexpr uint32_t kSlmBytes = kSlmML + kNC * 64;

template <int N>
inline esimd::simd<float, N / 2> fold_max(esimd::simd<float, N> v) SYCL_ESIMD_FUNCTION {
  return esimd::max<float, N / 2>(v.template select<N / 2, 2>(0).read(), v.template select<N / 2, 2>(1).read());
}
template <int N>
inline esimd::simd<float, N / 2> fold_sum(esimd::simd<float, N> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, N / 2> a = v.template select<N / 2, 2>(0), b = v.template select<N / 2, 2>(1);
  return a + b;
}

sycl::event attn_e(sycl::queue &q, int rows, int sr, const int32_t *rec, int hq, int hkv, const bf16 *qb, const bf16 *kb, const bf16 *vb, const DfRing &ring, int window,
                   bf16 *out, bool causal) {
  const int segs = (rows + sr - 1) / sr, grp = hq / hkv, Q = sr * grp, QC = (Q + kQ - 1) / kQ, R = ring.ring;
  const float scale = 1.f / std::sqrt(static_cast<float>(kD));
  const bf16 *rk = ring.k, *rv = ring.v;
  const float ninf = -std::numeric_limits<float>::infinity();
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<DfAttnE>(sycl::nd_range<1>(static_cast<size_t>(segs) * hkv * QC * kAeItems, kAeItems), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
      esimd::slm_init<kSlmBytes>();
      const int g0 = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int qc = g0 % QC, kvh = (g0 / QC) % hkv, seg = g0 / QC / hkv;
      const int32_t *w = rec + seg * kSegRec;
      const int start = w[kDfRecStart], slot = w[kDfRecSlot];
      const int nr = std::min(sr, rows - seg * sr);
      // the queries of this group into SLM, [call][q][16 d]
      if (lid < kQ) {
        const int qi = qc * kQ + lid, j = qi / grp, hh = qi % grp;
        esimd::simd<uint16_t, kD> v(0);
        if (j < nr) v = esimd::block_load<uint16_t, kD>(reinterpret_cast<const uint16_t *>(qb) + ((static_cast<size_t>(seg) * sr + j) * hq + kvh * grp + hh) * kD);
#pragma unroll
        for (int c = 0; c < 8; ++c) esimd::slm_block_store<uint16_t, 16>(kSlmQ + static_cast<uint32_t>((c * kQ + lid) * 32), v.template select<16, 1>(c * 16).read());
      }
      esimd::barrier();
      {
        const int c = lid / kHalves, dh = lid % kHalves;
        const bool blk = c == kCtxChunks;
        // per-query bounds of the visible key positions [lo, hi]
        esimd::simd<int, kQ> jq;
#pragma unroll
        for (int i = 0; i < kQ; ++i) jq[i] = (qc * kQ + i) / grp;
        esimd::simd<int, kQ> lo, hi;
        int b0, nb;
        if (blk) {
          lo = 0;
          if (causal) hi = esimd::min<int, kQ>(jq, esimd::simd<int, kQ>(nr - 1));
          else hi = nr - 1;
          b0 = 0;
          nb = (nr + 15) / 16;
        } else {
          lo = jq + (start - window + 1);
          hi = start - 1;
          // the segment's visible 16-key blocks split over the chunk items (by the blocks it has, not by the window's, so
          // a context shorter than the window still spreads over every item)
          const int base = std::max(0, start - window + 1), first0 = base / 16, last = (start - 1) / 16;
          const int nblk = start > 0 ? last - first0 + 1 : 0, bpc = (nblk + kCtxChunks - 1) / kCtxChunks, first = first0 + c * bpc;
          b0 = first;
          nb = std::max(0, std::min(bpc, last - first + 1));
        }
        const esimd::simd<int, kQ * 16> lov = lo.template replicate_vs_w_hs<kQ, 1, 16, 0>(0), hiv = hi.template replicate_vs_w_hs<kQ, 1, 16, 0>(0);
        const esimd::simd<int, 16> iota(0, 1);
        esimd::simd<float, kQ> M(ninf), L(0.f);
        esimd::simd<float, kTiles * kQ * 16> O(0.f);
        const size_t hp = static_cast<size_t>(hkv) * kD;  // words per key row
        for (int bi = 0; bi < nb; ++bi) {
          const int b = b0 + bi;
          const uint16_t *kp, *vp;
          int sh;  // surface rows - 1
          if (blk) {
            const size_t o = (static_cast<size_t>(seg) * sr + b * 16) * hp + static_cast<size_t>(kvh) * kD;
            kp = reinterpret_cast<const uint16_t *>(kb) + o;
            vp = reinterpret_cast<const uint16_t *>(vb) + o;
            sh = nr - b * 16 - 1;
          } else {
            const size_t o = (static_cast<size_t>(slot) * R + static_cast<size_t>((b * 16) % R)) * hp + static_cast<size_t>(kvh) * kD;
            kp = reinterpret_cast<const uint16_t *>(rk) + o;
            vp = reinterpret_cast<const uint16_t *>(rv) + o;
            sh = 15;
          }
          const unsigned pitch = static_cast<unsigned>(hp * 2 - 1);
          esimd::simd<float, kQ * 16> S(0.f);
#pragma unroll
          for (int cl = 0; cl < 8; ++cl) {
            esimd::simd<uint32_t, 128> kt = esimd::load_2d<uint32_t, 8, 16, 1, true>(reinterpret_cast<const uint32_t *>(kp), kD * 2 - 1, sh, pitch, cl * 8, 0);
            esimd::simd<bf16, 256> bk = kt.template bit_cast_view<bf16>().read();
            const esimd::simd<bf16, kQ * 16> aq = esimd::slm_block_load<bf16, kQ * 16>(kSlmQ + static_cast<uint32_t>(cl * kQ * 32));
            S = xmx::dpas<8, 8, float, float, bf16, bf16>(S, bk, aq);
          }
          S *= scale;
          const esimd::simd<int, kQ * 16> pos = (iota + b * 16).template replicate_w<kQ, 16>(0);
          const esimd::simd_mask<kQ * 16> vis = (pos >= lov) & (pos <= hiv);
          S.merge(esimd::simd<float, kQ * 16>(ninf), !vis);
          const esimd::simd<float, kQ> mb = fold_max<16>(fold_max<32>(fold_max<64>(fold_max<128>(S))));
          const esimd::simd<float, kQ> Mn = esimd::max<float, kQ>(M, mb);
          if ((Mn > M).any()) {
            esimd::simd<float, kQ> f = esimd::exp(M - Mn);
            f.merge(esimd::simd<float, kQ>(0.f), M == ninf);
            L *= f;
            const esimd::simd<float, kQ * 16> fv = f.template replicate_vs_w_hs<kQ, 1, 16, 0>(0);
#pragma unroll
            for (int t = 0; t < kTiles; ++t) O.template select<kQ * 16, 1>(t * kQ * 16) *= fv;
            M = Mn;
          }
          esimd::simd<float, kQ> Ms = M;
          Ms.merge(esimd::simd<float, kQ>(0.f), M == ninf);
          esimd::simd<float, kQ * 16> P = esimd::exp(S - Ms.template replicate_vs_w_hs<kQ, 1, 16, 0>(0));
          P.merge(esimd::simd<float, kQ * 16>(0.f), !vis);
          L += fold_sum<16>(fold_sum<32>(fold_sum<64>(fold_sum<128>(P))));
          const esimd::simd<bf16, kQ * 16> ph = P;
          const esimd::simd<float, kQ * 16> phf = ph;
          const esimd::simd<bf16, kQ * 16> pl = P - phf;
#pragma unroll
          for (int t = 0; t < kTiles; ++t) {
            esimd::simd<uint16_t, 256> vt = esimd::load_2d<uint16_t, 16, 16, 1, false, true>(vp, kD * 2 - 1, sh, pitch, (dh * kTiles + t) * 16, 0);
            esimd::simd<bf16, 256> bv = vt.template bit_cast_view<bf16>().read();
            esimd::simd<float, kQ * 16> o = O.template select<kQ * 16, 1>(t * kQ * 16);
            o = xmx::dpas<8, 8, float, float, bf16, bf16>(o, bv, ph);
            o = xmx::dpas<8, 8, float, float, bf16, bf16>(o, bv, pl);
            O.template select<kQ * 16, 1>(t * kQ * 16) = o;
          }
        }
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
          esimd::slm_block_store<float, kQ * 16>(kSlmO + static_cast<uint32_t>(c * 4096 + (dh * kTiles + t) * 512), O.template select<kQ * 16, 1>(t * kQ * 16).read());
        if (dh == 0) {
          esimd::slm_block_store<float, kQ>(kSlmML + static_cast<uint32_t>(c * 64), M);
          esimd::slm_block_store<float, kQ>(kSlmML + static_cast<uint32_t>(c * 64 + 32), L);
        }
      }
      esimd::barrier();
      if (lid < kQ) {  // merge: query lid
        const int qi = qc * kQ + lid, j = qi / grp, hh = qi % grp;
        if (j < nr) {
          float Mx = ninf;
          for (int c = 0; c < kNC; ++c) Mx = std::max(Mx, esimd::slm_scalar_load<float>(kSlmML + static_cast<uint32_t>(c * 64 + lid * 4)));
          esimd::simd<float, kD> acc(0.f);
          float Ls = 0.f;
          for (int c = 0; c < kNC; ++c) {
            const float m = esimd::slm_scalar_load<float>(kSlmML + static_cast<uint32_t>(c * 64 + lid * 4));
            if (m == ninf) continue;
            const float e = esimd::exp(esimd::simd<float, 1>(m - Mx))[0];
            Ls += e * esimd::slm_scalar_load<float>(kSlmML + static_cast<uint32_t>(c * 64 + 32 + lid * 4));
#pragma unroll
            for (int t = 0; t < 8; ++t)
              acc.template select<16, 1>(t * 16) += esimd::slm_block_load<float, 16>(kSlmO + static_cast<uint32_t>(c * 4096 + t * 512 + lid * 64)) * e;
          }
          const float inv = Ls > 0.f ? 1.f / Ls : 0.f;
          const esimd::simd<bf16, kD> ov = acc * inv;
          ov.copy_to(out + ((static_cast<size_t>(seg) * sr + j) * hq + kvh * grp + hh) * kD);
        }
      }
    });
  });
}

inline bool attn_al64(const void *p) { return reinterpret_cast<uintptr_t>(p) % 64 == 0; }
}  // namespace

// ---- attention ------------------------------------------------------------------------------------
size_t df_attn_part_floats(int rows, int seg_rows, int hq, int window) {
  const int sr = seg_rows > 0 ? seg_rows : rows, segs = (rows + sr - 1) / sr;
  return static_cast<size_t>(segs) * hq * sr * (ctx_chunks(window) + 1) * kPartStride;
}

sycl::event df_attn(sycl::queue &q, int rows, int seg_rows, const int32_t *rec, int hq, int hkv, const df_bf16 *qb, const df_bf16 *kb, const df_bf16 *vb,
                    const DfRing &ring, int window, float *part, df_bf16 *out, bool causal) {
  const int sr = seg_rows > 0 ? seg_rows : rows, segs = (rows + sr - 1) / sr, grp = hq / hkv;
  if (rows < 1 || hkv < 1 || hq % hkv || grp > kDfMaxGroup || sr > kMaxRows || window < 2 || ring.ring < window + kMaxRows || ring.hkv != hkv)
    throw std::invalid_argument("df_attn: shape");
  if (!std::getenv("XEN_DF_REFERENCE_ATTN") && ring.ring % 16 == 0 && attn_al64(kb) && attn_al64(vb) && attn_al64(ring.k) && attn_al64(ring.v))
    return attn_e(q, rows, sr, rec, hq, hkv, qb, kb, vb, ring, window, out, causal);
  const int D = kDfDim, NC = ctx_chunks(window) + 1, Q = sr * grp, R = ring.ring;
  // DSpine's 16-row block: queries past 32 per kv head go to further work-groups of 32 (query tiles)
  const int QT = (Q + kDfMaxQ - 1) / kDfMaxQ;
  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  const bf16 *rk = ring.k, *rv = ring.v;
  // A work-group per (segment, kv head, 32-key chunk), 128 items in three phases with short serial chains -
  //   scores   item = (key, quarter of the queries): 8 dots of 128 over the key row (bf16 x 8 loads), q broadcast from SLM
  //   softmax  item = (query, quarter of the keys): 8 keys, max / sum over the 4 lanes by sub-group xor shuffles
  //   P . V    item = dim: every query over the chunk's keys (V rows coalesced across the items, P broadcast)
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> qs(sycl::range<1>(static_cast<size_t>(kDfMaxQ) * D), h);
    sycl::local_accessor<float, 1> sc(sycl::range<1>(static_cast<size_t>(kDfMaxQ) * kDfChunk), h);
    h.parallel_for<DfAttnK>(sycl::nd_range<1>(static_cast<size_t>(segs) * hkv * NC * QT * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
      const int g0 = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int qt = g0 % QT, g = g0 / QT;
      const int c = g % NC, kvh = (g / NC) % hkv, seg = g / NC / hkv;
      const int qb0 = qt * kDfMaxQ, Qt = sycl::min(kDfMaxQ, Q - qb0);  // this tile's queries [qb0, qb0 + Qt)
      const int32_t *w = rec + seg * kSegRec;
      const int start = w[kDfRecStart], slot = w[kDfRecSlot];
      const int nr = sycl::min(sr, rows - seg * sr);  // live rows of this segment
      const bool blk = c == NC - 1;
      const int base = sycl::max(0, start - window + 1);
      const int k0 = base + c * kDfChunk;  // ctx chunk: first key position
      const int nk = blk ? nr : sycl::max(0, sycl::min(kDfChunk, start - k0));
      float *pp = part + ((static_cast<size_t>(seg) * hkv + kvh) * NC + c) * static_cast<size_t>(Q) * kPartStride + static_cast<size_t>(qb0) * kPartStride;
      if (nk == 0) {  // empty chunk: max -inf, sum 0 (the merge skips it)
        if (lid < Qt) {
          pp[static_cast<size_t>(lid) * kPartStride + D] = -std::numeric_limits<float>::infinity();
          pp[static_cast<size_t>(lid) * kPartStride + D + 1] = 0.f;
        }
        return;
      }
      // queries: (row j, head hh of the group) -> qi = j * grp + hh
      for (int e = lid; e < Qt * D; e += 128) {
        const int qi = e / D, d = e % D, j = (qb0 + qi) / grp, hh = (qb0 + qi) % grp;
        qs[e] = j < nr ? static_cast<float>(qb[((static_cast<size_t>(seg) * sr + j) * hq + kvh * grp + hh) * D + d]) : 0.f;
      }
      sycl::group_barrier(it.get_group());
      {  // scores
        const int kk = lid % kDfChunk, q0 = (lid / kDfChunk) * 8;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (kk < nk && q0 < Qt) {
          const bf16 *kr = blk ? kb + ((static_cast<size_t>(seg) * sr + kk) * hkv + kvh) * D
                               : rk + ((static_cast<size_t>(slot) * R + static_cast<size_t>((k0 + kk) % R)) * hkv + kvh) * D;
          for (int d0 = 0; d0 < D; d0 += 8) {
            const sycl::vec<uint16_t, 8> raw = *reinterpret_cast<const sycl::vec<uint16_t, 8> *>(reinterpret_cast<const uint16_t *>(kr) + d0);
            float kv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) kv[i] = sycl::bit_cast<float>(static_cast<uint32_t>(raw[i]) << 16);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              float sacc = acc[u];
#pragma unroll
              for (int i = 0; i < 8; ++i) sacc += qs[(q0 + u) * D + d0 + i] * kv[i];
              acc[u] = sacc;
            }
          }
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int qi = q0 + u, j = (qb0 + qi) / grp;
          if (qi < Qt) {
            const bool vis = kk < nk && (blk ? !causal || kk <= j : k0 + kk > start + j - window);
            sc[qi * kDfChunk + kk] = vis ? acc[u] * scale : -std::numeric_limits<float>::infinity();
          }
        }
      }
      sycl::group_barrier(it.get_group());
      {  // softmax partials: 4 lanes per query, 8 keys each
        auto sg = it.get_sub_group();
        const int qi = lid / 4, part4 = lid % 4;
        float m = -std::numeric_limits<float>::infinity();
        if (qi < Qt)
#pragma unroll
          for (int i = 0; i < 8; ++i) m = sycl::fmax(m, sc[qi * kDfChunk + part4 * 8 + i]);
        m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, 1));
        m = sycl::fmax(m, sycl::permute_group_by_xor(sg, m, 2));
        float l = 0.f;
        if (qi < Qt)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int at = qi * kDfChunk + part4 * 8 + i;
            const float sv = sc[at];
            const float pv = sv == -std::numeric_limits<float>::infinity() ? 0.f : sycl::exp(sv - m);
            sc[at] = pv;
            l += pv;
          }
        l += sycl::permute_group_by_xor(sg, l, 1);
        l += sycl::permute_group_by_xor(sg, l, 2);
        if (qi < Qt && part4 == 0) {
          pp[static_cast<size_t>(qi) * kPartStride + D] = m;
          pp[static_cast<size_t>(qi) * kPartStride + D + 1] = l;
        }
      }
      sycl::group_barrier(it.get_group());
      {  // P . V: item = dim
        const int d = lid;
        float o[kDfMaxQ];
#pragma unroll
        for (int i = 0; i < kDfMaxQ; ++i) o[i] = 0.f;
        for (int k = 0; k < nk; ++k) {
          const bf16 *vr = blk ? vb + ((static_cast<size_t>(seg) * sr + k) * hkv + kvh) * D
                               : rv + ((static_cast<size_t>(slot) * R + static_cast<size_t>((k0 + k) % R)) * hkv + kvh) * D;
          const float v = static_cast<float>(vr[d]);
#pragma unroll
          for (int qi = 0; qi < kDfMaxQ; ++qi)
            if (qi < Qt) o[qi] += sc[qi * kDfChunk + k] * v;
        }
#pragma unroll
        for (int qi = 0; qi < kDfMaxQ; ++qi)
          if (qi < Qt) pp[static_cast<size_t>(qi) * kPartStride + d] = o[qi];
      }
    });
  });
  return q.parallel_for<DfAttnMergeK>(sycl::nd_range<1>(static_cast<size_t>(rows) * hq * D, D), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
    const int r = g / hq, h = g % hq, seg = r / sr, j = r % sr, kvh = h / grp, qi = j * grp + h % grp;
    const float *pb = part + (static_cast<size_t>(seg) * hkv + kvh) * NC * static_cast<size_t>(Q) * kPartStride + static_cast<size_t>(qi) * kPartStride;
    const size_t cs = static_cast<size_t>(Q) * kPartStride;
    float M = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < NC; ++c) M = sycl::fmax(M, pb[c * cs + D]);
    float L = 0.f, O = 0.f;
    for (int c = 0; c < NC; ++c) {
      const float m = pb[c * cs + D];
      if (m == -std::numeric_limits<float>::infinity()) continue;
      const float e = sycl::exp(m - M);
      L += e * pb[c * cs + D + 1];
      O += e * pb[c * cs + d];
    }
    out[(static_cast<size_t>(r) * hq + h) * D + d] = bf16(L > 0.f ? O / L : 0.f);
  });
}

void df_attn_ref(int seg_rows, int start, int hq, int hkv, const uint16_t *q, const uint16_t *kb, const uint16_t *vb, const uint16_t *ring_k,
                 const uint16_t *ring_v, int ring, int window, uint16_t *out, bool causal) {
  const int D = kDfDim, grp = hq / hkv;
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<double> s;
  for (int j = 0; j < seg_rows; ++j)
    for (int h = 0; h < hq; ++h) {
      const int kvh = h / grp;
      const uint16_t *qr = q + (static_cast<size_t>(j) * hq + h) * D;
      std::vector<const uint16_t *> ks, vs;
      for (int kp = std::max(0, start + j - window + 1); kp < start; ++kp) {
        ks.push_back(ring_k + (static_cast<size_t>(kp % ring) * hkv + kvh) * D);
        vs.push_back(ring_v + (static_cast<size_t>(kp % ring) * hkv + kvh) * D);
      }
      for (int b = 0; b < (causal ? j + 1 : seg_rows); ++b) {
        ks.push_back(kb + (static_cast<size_t>(b) * hkv + kvh) * D);
        vs.push_back(vb + (static_cast<size_t>(b) * hkv + kvh) * D);
      }
      s.assign(ks.size(), 0.0);
      double m = -1e300;
      for (size_t i = 0; i < ks.size(); ++i) {
        double a = 0;
        for (int d = 0; d < D; ++d) a += static_cast<double>(bf16_value(qr[d])) * bf16_value(ks[i][d]);
        s[i] = a * scale;
        m = std::max(m, s[i]);
      }
      double l = 0;
      for (auto &v : s) l += (v = std::exp(v - m));
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (size_t i = 0; i < ks.size(); ++i) o += s[i] * bf16_value(vs[i][d]);
        out[(static_cast<size_t>(j) * hq + h) * D + d] = bf16_bits(static_cast<float>(o / l));
      }
    }
}

// ---- top-k ----------------------------------------------------------------------------------------
uint64_t df_key(float logit, int64_t index) { return make_key(logit, static_cast<uint32_t>(index)); }

float df_key_value(uint64_t key) {
  const uint32_t o = static_cast<uint32_t>(key >> 32);
  const uint16_t b = (o & 0x8000) ? static_cast<uint16_t>(o & 0x7fff) : static_cast<uint16_t>(~o & 0xffff);
  return bf16_value(b);
}

sycl::event df_topk_wire_init(sycl::queue &q, uint32_t *inbox, size_t words) { return q.fill(inbox, kMailEmpty, words); }

size_t df_topk_scratch_bytes(int n) {
  const int chunks = (n + kTopkChunk - 1) / kTopkChunk;
  return (static_cast<size_t>(kMaxRows) * chunks * kDfTopK + 3 * static_cast<size_t>(kMaxRows) * kDfTopK) * sizeof(uint64_t);  // + hop 1's merge (four cards)
}

// df_topk's mailbox hop: this card's [rows][16] list (src) out to the peer's inbox, the peer's into dst (zero + error
// bit on a timeout). bump: the step counter moves after this hop (the last hop of a call).
template <class Name>
void topk_exchange(sycl::queue &q, const DfTopkWire &w, int rows, const uint64_t *src, uint64_t *dst, uint32_t *inbox, uint32_t *peer_inbox, bool bump) {
  namespace esimd = sycl::ext::intel::esimd;
  int32_t *step = w.step;
  uint32_t *error = w.error;
  const uint64_t wait_cycles = w.wait_cycles;
  q.parallel_for<Name>(sycl::nd_range<1>(kMaxRows, kMaxRows), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_local_id(0));
    constexpr int W = 2 * kDfTopK;
    const int parity = step[0] & 1;
    if (r < rows) {
      const esimd::simd<uint32_t, W> mine = esimd::block_load<uint32_t, W>(reinterpret_cast<const uint32_t *>(src + static_cast<size_t>(r) * kDfTopK));
      constexpr auto uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                            esimd::alignment<16>};
      const size_t slot = (static_cast<size_t>(parity) * kMaxRows + r) * W;
      esimd::block_store<uint32_t, W>(peer_inbox + slot, mine, uc);
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
      esimd::simd<uint32_t, W> got = esimd::block_load<uint32_t, W>(inbox + slot, uc);
      bool ok = (got != kMailEmpty).all();
      if (!ok) {
        const uint64_t t0 = esimd::rdtsc();
        while (!ok) {
          esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
          got = esimd::block_load<uint32_t, W>(inbox + slot, uc);
          ok = (got != kMailEmpty).all();
          if (!ok && esimd::rdtsc() - t0 > wait_cycles) break;
        }
      }
      if (ok) {
        esimd::block_store<uint32_t, W>(inbox + slot, esimd::simd<uint32_t, W>(kMailEmpty), uc);
      } else {
        got = 0u;
        error[0] = 1u;
      }
      esimd::block_store<uint32_t, W>(reinterpret_cast<uint32_t *>(dst + static_cast<size_t>(r) * kDfTopK), got);
    }
    esimd::barrier();
    if (bump && r == 0) step[0] = step[0] + 1;
  });
}

// 4: merge two descending lists into the 16 best (every card: the same bits); b unread when !wired
template <class Name>
sycl::event topk_merge_lists(sycl::queue &q, int rows, const uint64_t *a0, const uint64_t *b0, bool wired, uint64_t *keys) {
  return q.parallel_for<Name>(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]);
    const uint64_t *a = a0 + static_cast<size_t>(r) * kDfTopK, *b = b0 + static_cast<size_t>(r) * kDfTopK;
    int ia = 0, ib = 0;
    for (int i = 0; i < kDfTopK; ++i) {
      const uint64_t x = a[ia];  // ia + ib = i < 16: both indexes stay in range
      const uint64_t y = wired ? b[ib] : 0ull;
      if (x >= y) {
        keys[static_cast<size_t>(r) * kDfTopK + i] = x;
        ++ia;
      } else {
        keys[static_cast<size_t>(r) * kDfTopK + i] = y;
        ++ib;
      }
    }
  });
}

sycl::event df_topk(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, void *scratch, const DfTopkWire *wire, uint64_t *keys) {
  if (rows < 1 || rows > kMaxRows || n < kDfTopK || ld < n) throw std::invalid_argument("df_topk: shape");
  const int chunks = (n + kTopkChunk - 1) / kTopkChunk;
  if (chunks * kDfTopK > kTopkItems * 8) throw std::invalid_argument("df_topk: too many columns");
  auto *ck = static_cast<uint64_t *>(scratch);                                // [rows][chunks][16]
  uint64_t *rk = ck + static_cast<size_t>(kMaxRows) * chunks * kDfTopK;       // [rows][16] this card's best
  // 1: per (row, chunk) the 16 best keys
  q.parallel_for<DfTopkChunks>(sycl::nd_range<1>(static_cast<size_t>(rows) * chunks * kTopkItems, kTopkItems), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0)), r = g / chunks, ch = g % chunks, lid = static_cast<int>(it.get_local_id(0));
    uint64_t v[kTopkPer];
#pragma unroll
    for (int i = 0; i < kTopkPer; ++i) {
      const int c = ch * kTopkChunk + i * kTopkItems + lid;
      v[i] = c < n ? make_key(logits[static_cast<size_t>(r) * ld + c], static_cast<uint32_t>(col_base + c)) : 0ull;
    }
    for (int k = 0; k < kDfTopK; ++k) {
      uint64_t mine = 0;
#pragma unroll
      for (int i = 0; i < kTopkPer; ++i) mine = v[i] > mine ? v[i] : mine;
      const uint64_t best = sycl::reduce_over_group(it.get_group(), mine, sycl::maximum<uint64_t>());
#pragma unroll
      for (int i = 0; i < kTopkPer; ++i)
        if (v[i] == best) v[i] = 0;  // keys are unique (the index part), zero never wins over a real key
      if (lid == 0) ck[(static_cast<size_t>(r) * chunks + ch) * kDfTopK + k] = best;
    }
  });
  // 2: per row the 16 best of its chunks' keys
  q.parallel_for<DfTopkRows>(sycl::nd_range<1>(static_cast<size_t>(rows) * kTopkItems, kTopkItems), [=](sycl::nd_item<1> it) {
    const int r = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
    const int cand = chunks * kDfTopK;
    uint64_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = i * kTopkItems + lid;
      v[i] = e < cand ? ck[static_cast<size_t>(r) * cand + e] : 0ull;
    }
    for (int k = 0; k < kDfTopK; ++k) {
      uint64_t mine = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) mine = v[i] > mine ? v[i] : mine;
      const uint64_t best = sycl::reduce_over_group(it.get_group(), mine, sycl::maximum<uint64_t>());
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (v[i] == best) v[i] = 0;
      if (lid == 0) rk[static_cast<size_t>(r) * kDfTopK + k] = best;
    }
  });
  // 3: the peer card's list through the mailbox (uncached, every word its own flag: head.cpp's pick merge) into pk
  uint64_t *pk = rk + static_cast<size_t>(kMaxRows) * kDfTopK;  // [rows][16] the peer's best (zero without a wire)
  const bool wired = wire != nullptr;
  if (!wired) return topk_merge_lists<DfTopkMergeLists>(q, rows, rk, pk, false, keys);
  const DfTopkWire w = *wire;
  if (!w.peer2_inbox) {  // two cards
    topk_exchange<DfTopkMerge>(q, w, rows, rk, pk, w.inbox, w.peer_inbox, true);
    return topk_merge_lists<DfTopkMergeLists>(q, rows, rk, pk, true, keys);
  }
  // four cards: hop 1 with card ^ 1 into m1, hop 2 with card ^ 2 (its own slots, the same parity:
  // the step counter moves after hop 2)
  uint64_t *m1 = pk + static_cast<size_t>(kMaxRows) * kDfTopK;
  topk_exchange<DfTopkMerge>(q, w, rows, rk, pk, w.inbox, w.peer_inbox, false);
  topk_merge_lists<DfTopkMergeLists>(q, rows, rk, pk, true, m1);
  topk_exchange<DfTopkMerge2>(q, w, rows, m1, pk, w.inbox + kDfTopkWireWords, w.peer2_inbox + kDfTopkWireWords, true);
  return topk_merge_lists<DfTopkMergeLists2>(q, rows, m1, pk, true, keys);
}

void df_topk_ref(const float *row, int n, int col_base, uint64_t *keys) {
  std::vector<uint64_t> v(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) v[static_cast<size_t>(i)] = make_key(row[i], static_cast<uint32_t>(col_base + i));
  std::partial_sort(v.begin(), v.begin() + kDfTopK, v.end(), std::greater<uint64_t>());
  std::copy(v.begin(), v.begin() + kDfTopK, keys);
}

class DfPackDraftRowsK;

sycl::event df_pack_draft_rows(sycl::queue &q, int segs, int block, int hidden, const df_bf16 *src, df_bf16 *dst) {
  constexpr int cols = 64;
  const int depth = block - 1, tiles = hidden / cols;
  if (segs < 1 || depth < 1 || hidden < 1 || hidden % cols) throw std::invalid_argument("df_pack_draft_rows: shape");
  return q.parallel_for<DfPackDraftRowsK>(sycl::range<1>(static_cast<size_t>(segs) * depth * tiles), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    const size_t row = id[0] / tiles, col = (id[0] % tiles) * cols;
    const size_t source_row = row + row / depth + 1;
    esimd::simd<df_bf16, cols> v;
    v.copy_from(src + source_row * hidden + col);
    v.copy_to(dst + row * hidden + col);
  });
}

// ---- selector -------------------------------------------------------------------------------------
sycl::event df_select(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *hp, int rank, const df_bf16 *pred_cb, const df_bf16 *succ_cb,
                      const int32_t *anchors, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise, bool compact) {
  constexpr int kParts = 16, kItems = kDfTopK * kParts;
  if (segs < 1 || seg_rows < 2 || segs * seg_rows > kMaxRows || rank % kParts) throw std::invalid_argument("df_select: shape");
  const int per = rank / kParts;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(kItems), h);
    sycl::local_accessor<float, 1> score(sycl::range<1>(kDfTopK), h);
    sycl::local_accessor<int32_t, 1> pred(sycl::range<1>(1), h);
    h.parallel_for<DfSelectK>(sycl::nd_range<1>(static_cast<size_t>(segs) * kItems, kItems), [=](sycl::nd_item<1> it) {
      const int s = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int b = lid / kParts, p = lid % kParts;
      const DraftNoiseRec nz = noise ? noise[s] : DraftNoiseRec{};  // coupled drafts (inv_t > 0)
      if (lid == 0) pred[0] = anchors[s];
      sycl::group_barrier(it.get_group());
      for (int j = 1; j < seg_rows; ++j) {
        const int row = compact ? s * (seg_rows - 1) + j - 1 : s * seg_rows + j;
        const uint64_t key = keys[static_cast<size_t>(row) * kDfTopK + b];
        const int32_t v = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(key));
        const int32_t tok = sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v;
        const int32_t pr = pred[0];
        const bf16 *P = pred_cb + static_cast<size_t>(pr) * rank, *S = succ_cb + static_cast<size_t>(tok) * rank;
        const float *hr = hp + static_cast<size_t>(row) * rank;
        float acc = 0.f;
        for (int i = p * per; i < (p + 1) * per; ++i) acc += bfr(static_cast<float>(P[i]) * bfr(hr[i])) * static_cast<float>(S[i]);
        part[lid] = acc;
        sycl::group_barrier(it.get_group());
        if (lid < kDfTopK) {
          float t = 0.f;
          for (int i = 0; i < kParts; ++i) t += part[lid * kParts + i];
          const uint64_t kl = keys[static_cast<size_t>(row) * kDfTopK + lid];
          const uint32_t o = static_cast<uint32_t>(kl >> 32);
          const uint16_t bb = (o & 0x8000) ? static_cast<uint16_t>(o & 0x7fff) : static_cast<uint16_t>(~o & 0xffff);
          float sc = sycl::bit_cast<float>(static_cast<uint32_t>(bb) << 16) + t;
          if (nz.inv_t > 0.0f) {
            const int32_t vl = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(kl));
            const int32_t tl = sub_k > 0 && vl >= sub_k ? vl - sub_k + sub_s0 : vl;
            sc = sc * nz.inv_t + util::gumbel(nz.key, nz.pos + static_cast<uint64_t>(j - 1), tl);
          }
          score[lid] = sc;
        }
        sycl::group_barrier(it.get_group());
        if (lid == 0) {
          int best = 0;
          for (int i = 1; i < kDfTopK; ++i)
            if (score[i] > score[best]) best = i;
          const uint64_t kb = keys[static_cast<size_t>(row) * kDfTopK + best];
          const int32_t vb = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(kb));
          const int32_t tb = sub_k > 0 && vb >= sub_k ? vb - sub_k + sub_s0 : vb;
          pred[0] = tb;
          drafts[s * (seg_rows - 1) + j - 1] = tb;
        }
        sycl::group_barrier(it.get_group());
      }
    });
  });
}

void df_select_ref(int seg_rows, const uint64_t *keys, const float *hp, int rank, const uint16_t *pred_cb, const uint16_t *succ_cb, int32_t anchor, int sub_k,
                   int sub_s0, int32_t *drafts, const DraftNoiseRec *noise) {
  auto tok_of = [&](uint64_t key) {
    const auto v = static_cast<int32_t>(df_key_index(key));
    return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v;
  };
  int32_t pred = anchor;
  for (int j = 1; j < seg_rows; ++j) {
    const uint64_t *kr = keys + static_cast<size_t>(j) * kDfTopK;
    const float *hr = hp + static_cast<size_t>(j) * rank;
    double best = -1e300;
    int32_t bt = -1;
    for (int b = 0; b < kDfTopK; ++b) {
      const int32_t t = tok_of(kr[b]);
      double a = 0;
      for (int i = 0; i < rank; ++i)
        a += static_cast<double>(bf16_value(bf16_bits(bf16_value(pred_cb[static_cast<size_t>(pred) * rank + i]) * bf16_value(bf16_bits(hr[i]))))) *
             bf16_value(succ_cb[static_cast<size_t>(t) * rank + i]);
      double sc = df_key_value(kr[b]) + a;
      if (noise && noise->inv_t > 0.0f)  // the device's fp32 form (score x inv_t + noise)
        sc = static_cast<float>(static_cast<float>(sc) * noise->inv_t) + util::gumbel(noise->key, noise->pos + static_cast<uint64_t>(j - 1), t);
      if (sc > best) best = sc, bt = t;
    }
    drafts[j - 1] = pred = bt;
  }
}

// ---- DDTree: the best-first draft tree ----
sycl::event df_tree(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *hp, int rank, const bf16 *pred_cb, const bf16 *succ_cb,
                    const int32_t *anchors, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp) {
  constexpr int kParts = 16, kItems = kDfTopK * kParts, kN = kTreeRows;
  if (segs < 1 || seg_rows < 2 || segs * seg_rows > kMaxRows || rank % kParts || budget < 1 || budget >= kN) throw std::invalid_argument("df_tree: shape");
  const int per = rank / kParts, D = seg_rows - 1;
  const float lcut = cut > 0.f ? sycl::log(cut) : -std::numeric_limits<float>::infinity();
  const float tscale = inv_temp;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(kItems), h);
    sycl::local_accessor<float, 1> score(sycl::range<1>(kDfTopK), h);
    sycl::local_accessor<int32_t, 1> ctok(sycl::range<1>(kN * kDfTopK), h);  // node i's candidates, best first
    sycl::local_accessor<float, 1> clp(sycl::range<1>(kN * kDfTopK), h);
    sycl::local_accessor<float, 1> clp1(sycl::range<1>(kN * kDfTopK), h);  // the cut's lp (temperature 1)
    sycl::local_accessor<float, 1> score1(sycl::range<1>(kDfTopK), h);
    sycl::local_accessor<int32_t, 1> ntok(sycl::range<1>(kN), h), npar(sycl::range<1>(kN), h), ndep(sycl::range<1>(kN), h), nnext(sycl::range<1>(kN), h);
    sycl::local_accessor<float, 1> nw(sycl::range<1>(kN), h), nw1(sycl::range<1>(kN), h);
    sycl::local_accessor<int32_t, 1> ctl(sycl::range<1>(2), h);  // [0] the node whose candidates come next (-1 none), [1] nodes
    h.parallel_for<DfTreeK>(sycl::nd_range<1>(static_cast<size_t>(segs) * kItems, kItems), [=](sycl::nd_item<1> it) {
      const int s = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int b = lid / kParts, p = lid % kParts;
      auto tok_of = [&](uint64_t key) {
        const int32_t v = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(key));
        return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v;
      };
      auto val_of = [&](uint64_t key) {
        const uint32_t o = static_cast<uint32_t>(key >> 32);
        const uint16_t bb = (o & 0x8000) ? static_cast<uint16_t>(o & 0x7fff) : static_cast<uint16_t>(~o & 0xffff);
        return sycl::bit_cast<float>(static_cast<uint32_t>(bb) << 16);
      };
      if (lid == 0) ntok[0] = anchors[s], npar[0] = -1, ndep[0] = 0, nnext[0] = 0, nw[0] = 0.f, nw1[0] = 0.f, ctl[0] = 0, ctl[1] = 1;
      sycl::group_barrier(it.get_group());
      for (;;) {
        const int node = ctl[0];
        sycl::group_barrier(it.get_group());  // every item read the word before item 0 rewrites it
        if (node >= 0) {  // node's 16 candidates (df_select's score, bit for bit), log-softmax, best first
          const int row = s * seg_rows + ndep[node] + 1;
          const uint64_t key = keys[static_cast<size_t>(row) * kDfTopK + b];
          const int32_t tok = tok_of(key), pr = ntok[node];
          const bf16 *P = pred_cb + static_cast<size_t>(pr) * rank, *S = succ_cb + static_cast<size_t>(tok) * rank;
          const float *hr = hp + static_cast<size_t>(row) * rank;
          float acc = 0.f;
          for (int i = p * per; i < (p + 1) * per; ++i) acc += bfr(static_cast<float>(P[i]) * bfr(hr[i])) * static_cast<float>(S[i]);
          part[lid] = acc;
          sycl::group_barrier(it.get_group());
          if (lid < kDfTopK) {
            float t = 0.f;
            for (int i = 0; i < kParts; ++i) t += part[lid * kParts + i];
            score1[lid] = val_of(keys[static_cast<size_t>(row) * kDfTopK + lid]) + t;
            score[lid] = score1[lid] * tscale;  // the tree's temperature
          }
          sycl::group_barrier(it.get_group());
          if (lid < kDfTopK) {
            float mx = score[0];
            for (int i = 1; i < kDfTopK; ++i) mx = sycl::fmax(mx, score[i]);
            float se = 0.f;
            for (int i = 0; i < kDfTopK; ++i) se += sycl::exp(score[i] - mx);
            const float me = score[lid];
            int rk = 0;
            for (int i = 0; i < kDfTopK; ++i) rk += score[i] > me || (score[i] == me && i < lid);
            ctok[node * kDfTopK + rk] = tok_of(keys[static_cast<size_t>(row) * kDfTopK + lid]);
            clp[node * kDfTopK + rk] = me - mx - sycl::log(se);
            if (cut > 0.f) {  // the cut's weights at temperature 1 (the same order: rk)
              float mx1 = score1[0];
              for (int i = 1; i < kDfTopK; ++i) mx1 = sycl::fmax(mx1, score1[i]);
              float se1 = 0.f;
              for (int i = 0; i < kDfTopK; ++i) se1 += sycl::exp(score1[i] - mx1);
              clp1[node * kDfTopK + rk] = score1[lid] - mx1 - sycl::log(se1);
            }
          }
          sycl::group_barrier(it.get_group());
        }
        if (lid == 0) {  // the frontier's best joins the tree
          const int n = ctl[1];
          int bi = -1;
          float bw = 0.f;
          float bw1 = 0.f;
          if (n - 1 < budget)
            for (int i = 0; i < n; ++i)
              if (ndep[i] < D && nnext[i] < kDfTopK) {
                const float w = nw[i] + clp[i * kDfTopK + nnext[i]];
                const float w1 = cut > 0.f ? nw1[i] + clp1[i * kDfTopK + nnext[i]] : 0.f;
                if (w1 >= lcut && (bi < 0 || w > bw)) bi = i, bw = w, bw1 = w1;  // the cut filters the frontier
              }
          if (bi < 0) {
            ctl[0] = -2;
          } else {
            ntok[n] = ctok[bi * kDfTopK + nnext[bi]], npar[n] = bi, ndep[n] = ndep[bi] + 1, nnext[n] = 0, nw[n] = bw, nw1[n] = bw1;
            ++nnext[bi];
            ctl[0] = ndep[n] < D ? n : -1;
            ctl[1] = n + 1;
          }
        }
        sycl::group_barrier(it.get_group());
        if (ctl[0] == -2) break;
      }
      if (lid < kN) {
        const int n = ctl[1] - 1;
        int32_t *o = tree + static_cast<size_t>(s) * kDfTreeInts;
        if (lid == 0) o[0] = n;
        o[1 + lid] = lid < n ? ntok[lid + 1] : 0;
        o[1 + kN + lid] = lid < n ? npar[lid + 1] : 0;
      }
    });
  });
}

void df_tree_ref(int seg_rows, const uint64_t *keys, const float *hp, int rank, const uint16_t *pred_cb, const uint16_t *succ_cb, int32_t anchor, int sub_k,
                 int sub_s0, int budget, float cut, int32_t *tree, float inv_temp) {
  constexpr int kN = kTreeRows, kParts = 16;
  const int D = seg_rows - 1, per = rank / kParts;
  auto tok_of = [&](uint64_t key) {
    const auto v = static_cast<int32_t>(df_key_index(key));
    return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v;
  };
  int32_t ntok[kN] = {anchor}, npar[kN] = {-1}, ndep[kN] = {0}, nnext[kN] = {0};
  float nw[kN] = {0.f}, clp[kN][kDfTopK] = {}, nw1[kN] = {0.f}, clp1[kN][kDfTopK] = {};
  int32_t ctok[kN][kDfTopK] = {};
  auto cands = [&](int node) {  // the device's fp32 sums in its order
    const int row = ndep[node] + 1;
    float sc[kDfTopK], sc1[kDfTopK];
    for (int b = 0; b < kDfTopK; ++b) {
      const uint64_t key = keys[static_cast<size_t>(row) * kDfTopK + b];
      const int32_t t = tok_of(key);
      float tot = 0.f;
      for (int pp = 0; pp < kParts; ++pp) {
        float acc = 0.f;
        for (int i = pp * per; i < (pp + 1) * per; ++i)
          acc += bf16_value(bf16_bits(bf16_value(pred_cb[static_cast<size_t>(ntok[node]) * rank + i]) * bf16_value(bf16_bits(hp[static_cast<size_t>(row) * rank + i])))) *
                 bf16_value(succ_cb[static_cast<size_t>(t) * rank + i]);
        tot += acc;
      }
      sc1[b] = static_cast<float>(df_key_value(key)) + tot;
      sc[b] = sc1[b] * inv_temp;
    }
    float mx1 = sc1[0];
    for (int i = 1; i < kDfTopK; ++i) mx1 = std::fmax(mx1, sc1[i]);
    float se1 = 0.f;
    for (int i = 0; i < kDfTopK; ++i) se1 += std::exp(sc1[i] - mx1);
    float mx = sc[0];
    for (int i = 1; i < kDfTopK; ++i) mx = std::fmax(mx, sc[i]);
    float se = 0.f;
    for (int i = 0; i < kDfTopK; ++i) se += std::exp(sc[i] - mx);
    for (int b = 0; b < kDfTopK; ++b) {
      int rk = 0;
      for (int i = 0; i < kDfTopK; ++i) rk += sc[i] > sc[b] || (sc[i] == sc[b] && i < b);
      ctok[node][rk] = tok_of(keys[static_cast<size_t>(row) * kDfTopK + b]);
      clp[node][rk] = sc[b] - mx - std::log(se);
      clp1[node][rk] = sc1[b] - mx1 - std::log(se1);  // the cut's weights (temperature 1)
    }
  };
  const float lcut = cut > 0.f ? std::log(cut) : -std::numeric_limits<float>::infinity();
  int n = 1;
  if (D >= 1) cands(0);
  while (n - 1 < budget) {
    int bi = -1;
    float bw = 0.f, bw1 = 0.f;
    for (int i = 0; i < n; ++i)
      if (ndep[i] < D && nnext[i] < kDfTopK) {
        const float w = nw[i] + clp[i][nnext[i]];
        const float w1 = cut > 0.f ? nw1[i] + clp1[i][nnext[i]] : 0.f;
        if (w1 >= lcut && (bi < 0 || w > bw)) bi = i, bw = w, bw1 = w1;
      }
    if (bi < 0) break;
    ntok[n] = ctok[bi][nnext[bi]], npar[n] = bi, ndep[n] = ndep[bi] + 1, nnext[n] = 0, nw[n] = bw, nw1[n] = bw1;
    ++nnext[bi];
    if (ndep[n] < D) cands(n);
    ++n;
  }
  std::fill(tree, tree + kDfTreeInts, 0);
  tree[0] = n - 1;
  for (int x = 1; x < n; ++x) tree[x] = ntok[x], tree[kN + x] = npar[x];  // node x: [1 + (x - 1)], [1 + kN + (x - 1)]
}

// ---- the deferred context ----
sycl::event df_ctx_stash(sycl::queue &q, int segs, int seg_rows, int cols, const bf16 *src, bf16 *dst, const int64_t (&dst_row)[kMaxSegs]) {
  if (segs < 1 || segs > kMaxSegs || seg_rows < 1 || cols % 8) throw std::invalid_argument("df_ctx_stash: segments / rows / columns");
  struct Rows {
    int64_t r[kMaxSegs];
  } dr;
  for (int i = 0; i < kMaxSegs; ++i) dr.r[i] = dst_row[i];
  const int c8 = cols / 8, rows = segs * seg_rows;
  const sycl::uint4 *s4 = reinterpret_cast<const sycl::uint4 *>(src);
  sycl::uint4 *d4 = reinterpret_cast<sycl::uint4 *>(dst);
  return q.parallel_for<DfCtxStashK>(sycl::range<2>(static_cast<size_t>(rows), static_cast<size_t>(c8)), [=](sycl::item<2> it) {
    const int row = static_cast<int>(it[0]), c = static_cast<int>(it[1]);
    const int i = row / seg_rows, r = row % seg_rows;
    d4[static_cast<size_t>(dr.r[i] + r) * c8 + c] = s4[static_cast<size_t>(row) * c8 + c];
  });
}

sycl::event df_rows_gather(sycl::queue &q, int n, int cols, const bf16 *src, bf16 *dst, const int (&map)[kMaxRows]) {
  if (n < 1 || n > kMaxRows || cols % 8) throw std::invalid_argument("df_rows_gather: rows / columns");
  struct Map {
    int r[kMaxRows];
  } m;
  for (int i = 0; i < kMaxRows; ++i) m.r[i] = map[i];
  const int c8 = cols / 8;
  const sycl::uint4 *s4 = reinterpret_cast<const sycl::uint4 *>(src);
  sycl::uint4 *d4 = reinterpret_cast<sycl::uint4 *>(dst);
  return q.parallel_for<DfRowsGatherK>(sycl::range<2>(static_cast<size_t>(n), static_cast<size_t>(c8)), [=](sycl::item<2> it) {
    const int row = static_cast<int>(it[0]), c = static_cast<int>(it[1]);
    d4[static_cast<size_t>(row) * c8 + c] = s4[static_cast<size_t>(m.r[row]) * c8 + c];
  });
}

sycl::event df_rec_set(sycl::queue &q, int32_t *rec, int64_t ctx_p0, int slot) {
  const int32_t p0 = static_cast<int32_t>(ctx_p0);
  return q.parallel_for<DfRecSetK>(sycl::range<1>(kSegRec), [=](sycl::id<1> i) {
    const int w = static_cast<int>(i[0]);
    rec[w] = w == kDfRecCtx ? p0 : w == kDfRecSlot ? slot : 0;
  });
}

}  // namespace xenithium::kernels

namespace xenithium::kernels {
// XEN_DF_STAGE witness: set bit `bit` in *flag if any of the n halves is non-finite
sycl::event df_nanwatch(sycl::queue &q, int n, const void *p, unsigned *flag, unsigned bit, bool fp32) {
  const auto *h = static_cast<const uint16_t *>(p);
  const auto *f = static_cast<const uint32_t *>(p);
  return q.parallel_for<class DfNanWatch>(sycl::range<1>(256), [=](sycl::id<1> i) {
    for (int k = static_cast<int>(i[0]); k < n; k += 256) {
      const uint32_t magnitude = fp32 ? f[k] & 0x7fffffffu : h[k] & 0x7fffu;
      if (magnitude >= (fp32 ? 0x7f800000u : 0x7f80u)) {
        sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device>(*flag).fetch_or(bit);
        return;
      }
    }
  });
}

}  // namespace xenithium::kernels
