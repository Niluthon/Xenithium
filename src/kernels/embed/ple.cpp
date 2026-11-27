#include "xenithium/kernels/embed/ple.hpp"

#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {
namespace {

constexpr int kGateMaxPer = 16;  // H / items held in registers
// Items of the gate's work-group (one per (row, stream)): H / 4 when it fits, else the widest
// split that does.
int gate_items(int H) {
  for (const int per : {4, 2, 5, 8, 10, 16, 1}) {
    const int items = H / per;
    if (H % per == 0 && items <= 1024 && items % 16 == 0) return items;
  }
  return 0;
}
constexpr int kConvItems = 256;

template <bool ROWS>
class PleGate;
template <bool TREE>
class PleConv;
class PleMove;
class PleConvRows;
class PleRing;

inline float bfround(float v) { return static_cast<float>(ple_bf16(v)); }

}  // namespace

size_t ple_ring_elems(const PleShape &s) { return static_cast<size_t>(kPleRing) * s.width(); }

class PleWaitK;

sycl::event ple_wait_rows(sycl::queue &q, const PleWait &w, int rows, int emb, const ple_bf16 *emb_host, ple_bf16 *emb_dev) {
  namespace esimd = sycl::ext::intel::esimd;
  if (!w.want || !w.word || !w.error || rows < 1 || emb % 128) throw std::invalid_argument("ple_wait_rows: words / rows / emb");
  constexpr int kItems = 16, kV = 64;  // 64 dwords (256 B) per load
  const int n32 = rows * emb / 2;
  const uint32_t *want = w.want, *word = w.word;
  uint32_t *err = w.error;
  const uint64_t cycles = w.wait_cycles;
  const uint32_t *src = reinterpret_cast<const uint32_t *>(emb_host);
  uint32_t *dst = reinterpret_cast<uint32_t *>(emb_dev);
  return q.parallel_for<PleWaitK>(sycl::nd_range<1>(kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::slm_init<64>();
    const int tid = static_cast<int>(it.get_local_id(0));
    constexpr auto uc64 = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                            esimd::alignment<64>};
    constexpr auto uc256 = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                             esimd::alignment<256>};
    if (tid == 0) {
      const uint32_t wv = esimd::block_load<uint32_t, 16>(want, uc64)[0];
      esimd::simd<uint32_t, 16> d = esimd::block_load<uint32_t, 16>(word, uc64);
      bool ok = d[0] == wv;
      if (!ok) {
        const uint64_t t0 = esimd::rdtsc();
        while (!ok) {
          esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
          d = esimd::block_load<uint32_t, 16>(word, uc64);
          ok = d[0] == wv;
          if (!ok && esimd::rdtsc() - t0 > cycles) break;
        }
      }
      if (!ok) esimd::block_store<uint32_t, 16>(err, esimd::simd<uint32_t, 16>(1u), uc64);
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
    }
    esimd::barrier();
    for (int i = tid * kV; i < n32; i += kItems * kV) {
      const esimd::simd<uint32_t, kV> v = esimd::block_load<uint32_t, kV>(src + i, uc256);
      v.copy_to(dst + i);
    }
  });
}

PleEvents ple_decode(sycl::queue &q, const PleShape &s, int rows, const ple_bf16 *emb_host, const PleWeights &w, const PleState &st,
                     const int32_t *pos, const PleScratch &sc, ple_bf16 *res, int kv_ks, int rec_stride, int seg_rows, bool tree) {
  const int S = s.streams, H = s.hidden, C = s.width(), KV = s.kv_n();
  if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("ple_decode: rows must be 1..8");
  const int gate_n = gate_items(H);
  if (!gate_n) throw std::invalid_argument("ple_decode: no gate work-group shape for this hidden size");
  if (!w.kv.valid() || w.kv.n < KV || w.kv.k != s.emb) throw std::invalid_argument("ple_decode: kv weights do not match the shape");
  if (seg_size(rows, seg_rows) + kPleDil * (kPleTaps - 1) > kPleRing) throw std::invalid_argument("ple_decode: the ring is too short for these rows");
  if (!seg_rows_ok(rows, seg_rows, rec_stride) || (seg_rows > 0 && rec_stride < kSegRec)) throw std::invalid_argument("ple_decode: segments need records");
  if (tree && rec_stride < kSegRec) throw std::invalid_argument("ple_decode: tree rows need segment records");
  PleEvents ev;
  if (emb_host) ev.up = q.memcpy(sc.emb, emb_host, static_cast<size_t>(rows) * s.emb * sizeof(ple_bf16));
  ev.kv = dense_gemm(q, rows, sc.emb, w.kv, sc.kv, kv_ks);
  const int kv_ld = w.kv.n;
  const float eps = s.eps, inv_sqrt_h = 1.f / sycl::sqrt(static_cast<float>(H));
  const ple_bf16 *nk = w.nk, *nq = w.nq, *ncw = w.ncw, *cw = w.conv;
  const float *kv = sc.kv;
  ple_bf16 *gated = sc.gated, *ring0 = st.ring;
  const size_t slot_elems = st.slot_elems;

  ev.gate = q.parallel_for<PleGate<false>>(
      sycl::nd_range<1>(static_cast<size_t>(rows) * S * gate_n, gate_n), [=](sycl::nd_item<1> it) {
        const int g = static_cast<int>(it.get_group(0)), row = g / S, st_ = g % S, lid = static_cast<int>(it.get_local_id(0));
        const int per = H / gate_n;
        const float *krow = kv + static_cast<size_t>(row) * kv_ld + static_cast<size_t>(st_) * H;
        const float *vrow = kv + static_cast<size_t>(row) * kv_ld + C;
        const ple_bf16 *qrow = res + static_cast<size_t>(row) * C + static_cast<size_t>(st_) * H;
        float kk[kGateMaxPer], qq[kGateMaxPer];
        float sk = 0.f, sq = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = lid + j * gate_n;
            kk[j] = bfround(krow[i]);  // the projection's bf16 output
            qq[j] = static_cast<float>(qrow[i]);
            sk += kk[j] * kk[j];
            sq += qq[j] * qq[j];
          }
        sk = sycl::reduce_over_group(it.get_group(), sk, sycl::plus<float>());
        sq = sycl::reduce_over_group(it.get_group(), sq, sycl::plus<float>());
        const float rk = sycl::rsqrt(sk / static_cast<float>(H) + eps), rq = sycl::rsqrt(sq / static_cast<float>(H) + eps);
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = st_ * H + lid + j * gate_n;
            const float kn = bfround(kk[j] * rk * (1.f + static_cast<float>(nk[i])));
            const float qn = bfround(qq[j] * rq * (1.f + static_cast<float>(nq[i])));
            dot += bfround(kn * qn);
          }
        dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>());
        const float d = bfround(bfround(dot) * inv_sqrt_h);
        const float sign = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
        const float mag = bfround(sycl::sqrt(sycl::fmax(sycl::fabs(d), 1e-6f)));
        const float gt = bfround(1.f / (1.f + sycl::exp(-(sign * mag))));
        float gv[kGateMaxPer], sg = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            gv[j] = bfround(gt * bfround(vrow[lid + j * gate_n]));
            sg += gv[j] * gv[j];
          }
        sg = sycl::reduce_over_group(it.get_group(), sg, sycl::plus<float>());
        const float rg = sycl::rsqrt(sg / static_cast<float>(H) + eps);
        const SegRow sr = seg_row(pos, rec_stride, seg_rows, row);
        const int p = sr.pos;
        ple_bf16 *xrow = ring0 + sr.slot * slot_elems + static_cast<size_t>(p % kPleRing) * C + static_cast<size_t>(st_) * H;
        ple_bf16 *grow = gated + static_cast<size_t>(row) * C + static_cast<size_t>(st_) * H;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = lid + j * gate_n;
            grow[i] = ple_bf16(gv[j]);
            xrow[i] = ple_bf16(gv[j] * rg * (1.f + static_cast<float>(ncw[st_ * H + i])));
          }
      });

  const size_t items = static_cast<size_t>(rows) * C;
  // a tree segment's row j (rows.hpp) reads its taps at its virtual position p0 + depth(j): a tap at
  // t >= p0 is its ancestor anc(j, t - p0), whose input sits at that row's slot p0 + anc; the gate above wrote row j's
  // at its own slot p0 + j. tree_t false: the chain instance.
  auto conv = [&](auto tree_t) {
    constexpr bool TREE = decltype(tree_t)::value;
    return q.parallel_for<PleConv<TREE>>(
        sycl::nd_range<1>((items + kConvItems - 1) / kConvItems * kConvItems, kConvItems), [=](sycl::nd_item<1> it) {
          const size_t gi = it.get_global_id(0);
          if (gi >= items) return;
          const int row = static_cast<int>(gi / C), c = static_cast<int>(gi % C);
          const SegRow sr = seg_row(pos, rec_stride, seg_rows, row);
          int p = sr.pos, p0 = 0;
          uint64_t anc = 0;
          bool tr = false;
          if constexpr (TREE) {
            const int32_t *b = pos + sr.seg * rec_stride;
            const uint64_t tw = rec_word(b, kSegTree, kSegTreeHi);
            if (tw & kTreeFlag) {
              int dd = 0;
              anc = tree_anc_word(tw, sr.j, &dd);
              p0 = b[kSegPos], p = p0 + dd, tr = true;
            }
          }
          const ple_bf16 *ring = ring0 + sr.slot * slot_elems;
          float acc = 0.f;
#pragma unroll
          for (int k = 0; k < kPleTaps; ++k) {
            const int t = p - kPleDil * (kPleTaps - 1 - k);
            int ts = t;
            if constexpr (TREE)
              if (tr && t >= p0) ts = p0 + static_cast<int>((anc >> (4 * (t - p0))) & 15u);
            if (t >= 0) acc += static_cast<float>(cw[c * kPleTaps + k]) * static_cast<float>(ring[static_cast<size_t>(ts % kPleRing) * C + c]);
          }
          const float cv = bfround(acc);
          const float y = bfround(cv / (1.f + sycl::exp(-cv)));
          const size_t o = static_cast<size_t>(row) * C + c;
          res[o] = ple_bf16(static_cast<float>(res[o]) + bfround(static_cast<float>(gated[o]) + y));
        });
  };
  ev.conv = tree ? conv(std::true_type{}) : conv(std::false_type{});
  return ev;
}

PleEvents ple_rows_gate(sycl::queue &q, const PleShape &s, int lo, int hi, const ple_bf16 *emb_host, const PleWeights &w, const PleRowsScratch &sc,
                        const ple_bf16 *res) {
  const int S = s.streams, H = s.hidden, C = s.width(), KV = s.kv_n();
  const int gate_n = gate_items(H), m = hi - lo;
  if (lo < 0 || m < 1 || !gate_n) throw std::invalid_argument("ple_rows_gate: rows / hidden");
  if (!w.kv.valid() || w.kv.n < KV || w.kv.k != s.emb) throw std::invalid_argument("ple_rows_gate: kv weights do not match the shape");
  PleEvents ev;
  const size_t emb_at = static_cast<size_t>(lo) * s.emb;
  ev.up = q.memcpy(sc.emb + emb_at, emb_host + emb_at, static_cast<size_t>(m) * s.emb * sizeof(ple_bf16));
  const int kv_ld = w.kv.n;
  // a W8A8 row's result does not depend on the call around it (pf_dense.hpp), nor a W8A16 one's: the rows' bits as one call
  ev.kv = pf_dense(q, m, w.kv, sc.emb + emb_at, s.emb, PfOut{sc.kv + static_cast<size_t>(lo) * kv_ld, nullptr, w.kv.n}, sc.a8.on());
  const float eps = s.eps, inv_sqrt_h = 1.f / sycl::sqrt(static_cast<float>(H));
  const ple_bf16 *nk = w.nk, *nq = w.nq, *ncw = w.ncw;
  const float *kv = sc.kv;
  ple_bf16 *gated = sc.gated, *xs = sc.x;
  // The gate: decode's arithmetic, conv inputs into the chunk buffer.
  ev.gate = q.parallel_for<PleGate<true>>(
      sycl::nd_range<1>(static_cast<size_t>(m) * S * gate_n, gate_n), [=](sycl::nd_item<1> it) {
        const int g = static_cast<int>(it.get_group(0)), row = lo + g / S, st_ = g % S, lid = static_cast<int>(it.get_local_id(0));
        const int per = H / gate_n;
        const float *krow = kv + static_cast<size_t>(row) * kv_ld + static_cast<size_t>(st_) * H;
        const float *vrow = kv + static_cast<size_t>(row) * kv_ld + C;
        const ple_bf16 *qrow = res + static_cast<size_t>(row) * C + static_cast<size_t>(st_) * H;
        float kk[kGateMaxPer], qq[kGateMaxPer];
        float sk = 0.f, sq = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = lid + j * gate_n;
            kk[j] = bfround(krow[i]);
            qq[j] = static_cast<float>(qrow[i]);
            sk += kk[j] * kk[j];
            sq += qq[j] * qq[j];
          }
        sk = sycl::reduce_over_group(it.get_group(), sk, sycl::plus<float>());
        sq = sycl::reduce_over_group(it.get_group(), sq, sycl::plus<float>());
        const float rk = sycl::rsqrt(sk / static_cast<float>(H) + eps), rq = sycl::rsqrt(sq / static_cast<float>(H) + eps);
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = st_ * H + lid + j * gate_n;
            const float kn = bfround(kk[j] * rk * (1.f + static_cast<float>(nk[i])));
            const float qn = bfround(qq[j] * rq * (1.f + static_cast<float>(nq[i])));
            dot += bfround(kn * qn);
          }
        dot = sycl::reduce_over_group(it.get_group(), dot, sycl::plus<float>());
        const float d = bfround(bfround(dot) * inv_sqrt_h);
        const float sign = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
        const float mag = bfround(sycl::sqrt(sycl::fmax(sycl::fabs(d), 1e-6f)));
        const float gt = bfround(1.f / (1.f + sycl::exp(-(sign * mag))));
        float gv[kGateMaxPer], sg = 0.f;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            gv[j] = bfround(gt * bfround(vrow[lid + j * gate_n]));
            sg += gv[j] * gv[j];
          }
        sg = sycl::reduce_over_group(it.get_group(), sg, sycl::plus<float>());
        const float rg = sycl::rsqrt(sg / static_cast<float>(H) + eps);
        ple_bf16 *xrow = xs + static_cast<size_t>(row) * C + static_cast<size_t>(st_) * H;
        ple_bf16 *grow = gated + static_cast<size_t>(row) * C + static_cast<size_t>(st_) * H;
#pragma unroll
        for (int j = 0; j < kGateMaxPer; ++j)
          if (j < per) {
            const int i = lid + j * gate_n;
            grow[i] = ple_bf16(gv[j]);
            xrow[i] = ple_bf16(gv[j] * rg * (1.f + static_cast<float>(ncw[st_ * H + i])));
          }
      });
  return ev;
}

PleEvents ple_rows_conv(sycl::queue &q, const PleShape &s, int lo, int hi, int m, int64_t pos0, const PleWeights &w, const PleState &st,
                        const PleRowsScratch &sc, ple_bf16 *res, bool ring_tail, const int32_t *rec) {
  const int C = s.width();
  if (lo < 0 || hi <= lo || hi > m) throw std::invalid_argument("ple_rows_conv: rows");
  PleEvents ev;
  const ple_bf16 *cw = w.conv, *xs = sc.x, *gated = sc.gated;
  ple_bf16 *ring0 = st.ring;
  const size_t slot_elems = st.slot_elems;
  const size_t items = static_cast<size_t>(hi - lo) * C;
  const int64_t host_p0 = pos0;
  ev.conv = q.parallel_for<PleConvRows>(
      sycl::nd_range<1>((items + kConvItems - 1) / kConvItems * kConvItems, kConvItems), [=](sycl::nd_item<1> it) {
        const size_t gi = it.get_global_id(0);
        if (gi >= items) return;
        const int64_t p0 = rec ? static_cast<int64_t>(rec[kSegPos]) : host_p0;
        const ple_bf16 *ring = rec ? ring0 + static_cast<size_t>(rec[kSegSlot]) * slot_elems : ring0;
        const int row = lo + static_cast<int>(gi / C), c = static_cast<int>(gi % C);
        const int64_t p = p0 + row;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < kPleTaps; ++k) {
          const int64_t t = p - kPleDil * (kPleTaps - 1 - k);
          if (t < 0) continue;
          const ple_bf16 xv = t >= p0 ? xs[static_cast<size_t>(t - p0) * C + c] : ring[static_cast<size_t>(t % kPleRing) * C + c];
          acc += static_cast<float>(cw[c * kPleTaps + k]) * static_cast<float>(xv);
        }
        const float cv = bfround(acc);
        const float y = bfround(cv / (1.f + sycl::exp(-cv)));
        const size_t o = static_cast<size_t>(row) * C + c;
        res[o] = ple_bf16(static_cast<float>(res[o]) + bfround(static_cast<float>(gated[o]) + y));
      });
  if (!ring_tail) return ev;
  // The chunk's last kPleRing positions into the ring (the next steps' conv history).
  const int first = m > kPleRing ? m - kPleRing : 0, n = m - first;
  ev.kv = q.parallel_for<PleRing>(sycl::range<1>(static_cast<size_t>(n) * C), [=](sycl::id<1> id) {
    const int64_t p0 = rec ? static_cast<int64_t>(rec[kSegPos]) : host_p0;
    ple_bf16 *ring = rec ? ring0 + static_cast<size_t>(rec[kSegSlot]) * slot_elems : ring0;
    const int r = first + static_cast<int>(id[0] / C), c = static_cast<int>(id[0] % C);
    ring[static_cast<size_t>((p0 + r) % kPleRing) * C + c] = xs[static_cast<size_t>(r) * C + c];
  });
  return ev;
}

PleEvents ple_rows(sycl::queue &q, const PleShape &s, int m, int64_t pos0, const ple_bf16 *emb_host, const PleWeights &w, const PleState &st,
                   const PleRowsScratch &sc, ple_bf16 *res, const int32_t *rec) {
  if (m < 1) throw std::invalid_argument("ple_rows: rows");
  PleEvents ev = ple_rows_gate(q, s, 0, m, emb_host, w, sc, res);
  ev.conv = ple_rows_conv(q, s, 0, m, m, pos0, w, st, sc, res, true, rec).conv;
  return ev;
}

sycl::event ple_move(sycl::queue &q, const PleShape &s, const PleState &st, int slot, int p0, const int *from, int n) {
  if (n < 1 || n > kTreeRows || p0 < 0 || from[0] != 0) throw std::invalid_argument("ple_move: 1..16 rows from the root");
  uint32_t mv = 0;  // bit x: row x moves (from[x] != x)
  int fr[kTreeRows] = {};
  for (int x = 0; x < n; ++x) {
    if (x > 0 && (from[x] <= from[x - 1] || from[x] >= kTreeRows)) throw std::invalid_argument("ple_move: rows descend the tree");
    fr[x] = from[x];
    if (from[x] != x) mv |= 1u << x;
  }
  if (!mv) return sycl::event{};
  const int C = s.width();
  ple_bf16 *ring = st.ring + static_cast<size_t>(slot) * st.slot_elems;
  struct Fr {
    int v[kTreeRows];
  } frs;
  for (int x = 0; x < kTreeRows; ++x) frs.v[x] = fr[x];
  // one item per channel: every source read before any destination written (path 0, 2, 3 reads slot 2 and writes it)
  return q.parallel_for<PleMove>(sycl::range<1>(static_cast<size_t>(C)), [=](sycl::id<1> id) {
    const int c = static_cast<int>(id[0]);
    ple_bf16 v[kTreeRows];
#pragma unroll
    for (int x = 0; x < kTreeRows; ++x)
      if ((mv >> x) & 1u) v[x] = ring[static_cast<size_t>((p0 + frs.v[x]) % kPleRing) * C + c];
#pragma unroll
    for (int x = 0; x < kTreeRows; ++x)
      if ((mv >> x) & 1u) ring[static_cast<size_t>((p0 + x) % kPleRing) * C + c] = v[x];
  });
}

}  // namespace xenithium::kernels
