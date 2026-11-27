// Xenithium · the hyper-connection mixer's fast path (ESIMD): combine + norm in one launch,
// silu + up GEMM + gate mix in another. Same rounding points as the unfused kernels in hc.cpp;
// the transcendental functions are the hardware ones (exp, sqrt), so the results stay within
// the mixer bound of the host reference rather than bit-equal to the SYCL path.
#include <stdexcept>
#include <type_traits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>

#include "xenithium/kernels/residual/hc.hpp"

#include "comm/split_sum_esimd.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx = esimd::xmx;
using bf16 = hc_bf16;

constexpr int kW = 16;  // columns per tile
constexpr int kKCall = 16;
constexpr int kCallElems = kDenseCallElems;  // 256
constexpr int kGroup = 128;
constexpr int kNormParts = 8;  // work-items per (row, stream)
constexpr int kNormChunk = 64;  // bf16 per load
constexpr int kStreams = 4;  // the fused up-mix is built for 4 streams
constexpr int kMaxLowrank = 512;
// Default k split of the fused up-mix: one k split for every row count, so batch invariance holds.
constexpr int kUpMixKSplit = 2;

template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}

template <bool COMBINE>
class HcNormFast;
template <bool COMBINE, int PER>
class HcNormReg;
template <int M, DenseKind KIND, int KS, int MC>
class HcUpMix;

// One work-group per (row, stream), kNormParts items each holding H / kNormParts contiguous
// elements. Pass 1 forms the combined values and the partial sum of squares; the partials
// meet in SLM and are summed in part order by every item (all items get the same rrms).
// Pass 2 recomputes the combined values from the untouched residual (identical bits), stores
// them and the normalised stream. Nothing is read back after being written.
// For hidden 2560 (PER = 320 values per item) norm_reg keeps the combined values in registers
// between the passes and loads the norm weights before the barrier: the same arithmetic in the same order
// (bit-identical), one load round less.
template <bool COMBINE, int PER>
sycl::event norm_reg(sycl::queue &q, const HcShape &s, int rows, bf16 *res, const bf16 *block, const float *inj, int inj_ld, int inj_col,
                     const bf16 *w, bf16 *xn);
template <bool COMBINE>
sycl::event norm_fast(sycl::queue &q, const HcShape &s, int rows, bf16 *res, const bf16 *block, const float *inj, int inj_ld, int inj_col,
                      const bf16 *w, bf16 *xn) {
  const int S = s.streams, H = s.hidden, per = H / kNormParts;
  const float eps = s.eps;
  return q.parallel_for<HcNormFast<COMBINE>>(
      sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(S) * kNormParts}, {1, kNormParts}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kNormParts * sizeof(float)>();
        // a (row, stream) grid: the ids come from the launch, not from a division by the run-time stream count in every item
        const int row = static_cast<int>(it.get_group(0)), st = static_cast<int>(it.get_group(1)), p = static_cast<int>(it.get_local_id(1));
        const size_t base = (static_cast<size_t>(row) * S + st) * H + static_cast<size_t>(p) * per;
        const size_t bbase = static_cast<size_t>(row) * H + static_cast<size_t>(p) * per;
        float m = 0.f;
        if constexpr (COMBINE) {
          const float l = static_cast<float>(bf16(inj[static_cast<size_t>(row) * inj_ld + inj_col + st]));
          esimd::simd<float, 1> lv(l / static_cast<float>(S));
          m = 2.f * static_cast<float>(sigmoid_v<1>(lv)[0]);
        }
        auto combined = [&](int c) SYCL_ESIMD_FUNCTION {
          esimd::simd<float, kNormChunk> v = esimd::block_load<bf16, kNormChunk>(res + base + c);
          if constexpr (COMBINE) {
            const esimd::simd<float, kNormChunk> b = esimd::block_load<bf16, kNormChunk>(block + bbase + c);
            const esimd::simd<bf16, kNormChunk> r = b * m + v;  // one mad per element
            v = r;
          }
          return v;
        };
        esimd::simd<float, kNormChunk> acc(0.f);
        for (int c = 0; c < per; c += kNormChunk) {
          const esimd::simd<float, kNormChunk> v = combined(c);
          acc += v * v;
        }
        esimd::slm_block_store<float, 1>(static_cast<uint32_t>(p * sizeof(float)), esimd::simd<float, 1>(esimd::reduce<float>(acc, std::plus<>())));
        esimd::barrier();
        const esimd::simd<float, kNormParts> parts = esimd::slm_block_load<float, kNormParts>(0);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < kNormParts; ++i) ss += parts[i];
        const esimd::simd<float, 1> mean(ss / static_cast<float>(H) + eps);
        const float rr = esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(mean))[0];
        for (int c = 0; c < per; c += kNormChunk) {
          const esimd::simd<float, kNormChunk> v = combined(c);
          if constexpr (COMBINE) esimd::block_store<bf16, kNormChunk>(res + base + c, esimd::simd<bf16, kNormChunk>(v));
          const esimd::simd<float, kNormChunk> y = v * rr;
          const esimd::simd<float, kNormChunk> wv = esimd::block_load<bf16, kNormChunk>(w + static_cast<size_t>(st) * H + p * per + c);
          esimd::block_store<bf16, kNormChunk>(xn + base + c, esimd::simd<bf16, kNormChunk>(y * wv + y));  // Gemma (1 + w), one mad
        }
      });
}

// The combine's scale of stream st's block on a row: 2 sigmoid(bf16(inj_s) / S).
inline float combine_scale(const float *inj_row, int inj_col, int st, int S) SYCL_ESIMD_FUNCTION {
  const float l = static_cast<float>(bf16(inj_row[inj_col + st]));
  esimd::simd<float, 1> lv(l / static_cast<float>(S));
  return 2.f * static_cast<float>(sigmoid_v<1>(lv)[0]);
}

// norm_reg's body for one item: PER values of one (row, stream) at res + base (and xn + base), the stream's norm
// weights at w_st. block_chunk(j) gives the block's j-th kNormChunk values of the item's part (COMBINE only). The
// kNormParts items of the stream store their partial sums of squares at SLM byte slm_parts + 4 p and every item adds
// them in part order after a barrier of the whole work-group (all its items must call this).
template <bool COMBINE, int PER, class BlockChunk>
inline void norm_part(bf16 *res, size_t base, BlockChunk block_chunk, float m, const bf16 *w_st, uint32_t slm_parts, int p, int H, float eps,
                      bf16 *xn) SYCL_ESIMD_FUNCTION {
  static_assert(PER % kNormChunk == 0, "whole chunks per item");
  constexpr int kCh = PER / kNormChunk;
  esimd::simd<float, PER> vals;
  esimd::simd<float, kNormChunk> acc(0.f);
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    esimd::simd<float, kNormChunk> v = esimd::block_load<bf16, kNormChunk>(res + base + j * kNormChunk);
    if constexpr (COMBINE) {
      const esimd::simd<float, kNormChunk> b = block_chunk(j);
      const esimd::simd<bf16, kNormChunk> r = b * m + v;  // one mad per element, as norm_fast
      v = r;
    }
    vals.template select<kNormChunk, 1>(j * kNormChunk) = v;
    acc += v * v;
  }
  esimd::simd<bf16, PER> wv;
#pragma unroll
  for (int j = 0; j < kCh; ++j)
    wv.template select<kNormChunk, 1>(j * kNormChunk) = esimd::block_load<bf16, kNormChunk>(w_st + p * PER + j * kNormChunk);
  esimd::slm_block_store<float, 1>(slm_parts + static_cast<uint32_t>(p * sizeof(float)), esimd::simd<float, 1>(esimd::reduce<float>(acc, std::plus<>())));
  esimd::barrier();
  const esimd::simd<float, kNormParts> parts = esimd::slm_block_load<float, kNormParts>(slm_parts);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kNormParts; ++i) ss += parts[i];
  const esimd::simd<float, 1> mean(ss / static_cast<float>(H) + eps);
  const float rr = esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(mean))[0];
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    const esimd::simd<float, kNormChunk> v = vals.template select<kNormChunk, 1>(j * kNormChunk);
    if constexpr (COMBINE) esimd::block_store<bf16, kNormChunk>(res + base + j * kNormChunk, esimd::simd<bf16, kNormChunk>(v));
    const esimd::simd<float, kNormChunk> y = v * rr;
    const esimd::simd<float, kNormChunk> wf = wv.template select<kNormChunk, 1>(j * kNormChunk);
    esimd::block_store<bf16, kNormChunk>(xn + base + j * kNormChunk, esimd::simd<bf16, kNormChunk>(y * wf + y));  // Gemma (1 + w)
  }
}

template <bool COMBINE, int PER>
sycl::event norm_reg(sycl::queue &q, const HcShape &s, int rows, bf16 *res, const bf16 *block, const float *inj, int inj_ld, int inj_col,
                     const bf16 *w, bf16 *xn) {
  const int S = s.streams, H = s.hidden;
  const float eps = s.eps;
  return q.parallel_for<HcNormReg<COMBINE, PER>>(
      sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(S) * kNormParts}, {1, kNormParts}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kNormParts * sizeof(float)>();
        // a (row, stream) grid: the ids come from the launch, not from a division by the run-time stream count in every item
        const int row = static_cast<int>(it.get_group(0)), st = static_cast<int>(it.get_group(1)), p = static_cast<int>(it.get_local_id(1));
        const size_t base = (static_cast<size_t>(row) * S + st) * H + static_cast<size_t>(p) * PER;
        const size_t bbase = static_cast<size_t>(row) * H + static_cast<size_t>(p) * PER;
        const float m = COMBINE ? combine_scale(inj + static_cast<size_t>(row) * inj_ld, inj_col, st, S) : 0.f;
        auto block_chunk = [&](int j) SYCL_ESIMD_FUNCTION { return esimd::simd<float, kNormChunk>(esimd::block_load<bf16, kNormChunk>(block + bbase + j * kNormChunk)); };
        norm_part<COMBINE, PER>(res, base, block_chunk, m, w + static_cast<size_t>(st) * H, 0, p, H, eps, xn);
      });
}

// ---- the combine + norm whose block is a pending four-card owner sum (hc_sum_combine_norm) ----
// One work-group per own row: kStreams x kNormParts items. Item 0 waits for the other cards' partials (the flag lines,
// split::wait_flag_line_quad: every work-group waits for itself), then every item sums its kSumCols-column chunks of the
// row (split::owner_value: pair_split_finish's arithmetic, the copied rows read past L2), writes them to the sum's rows
// and to SLM, and after a barrier each (stream, part) item runs norm_reg's body with its block values from SLM.
constexpr int kSumNormHidden = 2560, kSumNormPer = kSumNormHidden / kNormParts;  // norm_reg's 320 values per item
constexpr int kSumNormItems = kStreams * kNormParts;
constexpr int kSumCols = split::kCols, kSumChunks = kSumNormHidden / kSumCols;
constexpr uint32_t kSumNormBlockBytes = kSumNormHidden * sizeof(bf16);  // SLM: the row's block, then the partials
static_assert(kSumNormHidden % kSumCols == 0 && kSumNormPer % kNormChunk == 0 && kSumCols % kNormChunk == 0, "whole chunks");

template <bool ADD, bool B16>
class HcSumNorm;

template <bool ADD, bool B16>
sycl::event sum_norm(sycl::queue &q, const HcShape &s, const QuadOwnerSum &os, bf16 *res, const float *inj, int inj_ld, int inj_col, const bf16 *w,
                     bf16 *xn) {
  constexpr int S = kStreams, H = kSumNormHidden;
  const float eps = s.eps;
  const float *own = os.own, *b0 = os.slot[0], *b1 = os.slot[1], *b2 = os.slot[2], *b3 = os.slot[3], *ap = os.add_p, *as = os.add_scale;
  const uint32_t *flags = os.recv_flags;
  const int card = os.card;
  const uint32_t seq = os.seq;
  const uint64_t wait_cycles = os.wait_cycles;
  uint32_t *error = os.error;
  uint64_t *span = os.wait_span;
  bf16 *out = os.out;
  const uint32_t zero = static_cast<uint32_t>(os.rows < 0);  // 0: a kernel argument, unknown to the compiler (mail::opaque)
  return q.parallel_for<HcSumNorm<ADD, B16>>(
      sycl::nd_range<1>(static_cast<size_t>(os.rows) * kSumNormItems, kSumNormItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSumNormBlockBytes + kSumNormItems * sizeof(float)>();
        const int row = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
        if (lid == 0) {
          const uint64_t t0 = split::wait_flag_line_quad(flags, card, seq, wait_cycles, error);
          if (span && row == 0) split::write_span(span, t0);
        }
        esimd::barrier();
        const float *const b[kWireCardsMax] = {b0, b1, b2, b3};
        for (int j = lid; j < kSumChunks; j += kSumNormItems) {
          const size_t off = static_cast<size_t>(row) * H + static_cast<size_t>(j) * kSumCols;
          const esimd::simd<bf16, kSumCols> v = split::owner_value<ADD, B16, true>(card, own, b, ap, ADD ? as[row] : 0.f, off, zero);
          v.copy_to(out + off);
          esimd::slm_block_store<bf16, kSumCols>(static_cast<uint32_t>(j * kSumCols * sizeof(bf16)), v);
        }
        esimd::barrier();
        const int st = lid / kNormParts, p = lid % kNormParts;
        const size_t base = (static_cast<size_t>(row) * S + st) * H + static_cast<size_t>(p) * kSumNormPer;
        const float m = combine_scale(inj + static_cast<size_t>(row) * inj_ld, inj_col, st, S);
        auto block_chunk = [&](int j) SYCL_ESIMD_FUNCTION {
          const uint32_t at = static_cast<uint32_t>((p * kSumNormPer + j * kNormChunk) * sizeof(bf16));
          return esimd::simd<float, kNormChunk>(esimd::slm_block_load<bf16, kNormChunk>(at));
        };
        norm_part<true, kSumNormPer>(res, base, block_chunk, m, w + static_cast<size_t>(st) * H,
                                     kSumNormBlockBytes + static_cast<uint32_t>(st * kNormParts * sizeof(float)), p, H, eps, xn);
      });
}

template <DenseKind KIND>
inline esimd::simd<bf16, kCallElems> load_call(const void *w, size_t elem) SYCL_ESIMD_FUNCTION {
  if constexpr (KIND == DenseKind::bf16) {
    return esimd::block_load<bf16, kCallElems>(static_cast<const bf16 *>(w) + elem);
  } else {
    const esimd::simd<int8_t, kCallElems> c = esimd::block_load<int8_t, kCallElems>(static_cast<const int8_t *>(w) + elem);
    const esimd::simd<float, kCallElems> f = c;
    return f;
  }
}

// silu(lora) for M rows x N columns starting at column k0, as the bf16 A operand rows.
template <int kM, int N>
inline esimd::simd<bf16, kM * N> silu_rows(const float *lora, int ld, int k0, float inv_s) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kM * N> a;
#pragma unroll
  for (int r = 0; r < kM; ++r) {
    const esimd::simd<float, N> raw = esimd::block_load<float, N>(lora + static_cast<size_t>(r) * ld + k0);
    const esimd::simd<float, N> x = esimd::simd<float, N>(esimd::simd<bf16, N>(raw)) * inv_s;
    a.template select<N, 1>(r * N) = x * sigmoid_v<N>(x);
  }
  return a;
}

// One work-group per 16 hidden columns; per stream st, KS work-items split the up GEMM's
// 16-k calls of the tile of columns st * H + [c0, c0 + 16) (calls dealt in contiguous runs,
// silu of the matching lora columns formed in registers per call). The parts meet in SLM and
// part 0 sums them in part order, applies the row scales, rounds the gates to bf16 and leaves
// sigmoid(gate) in SLM; item r then mixes row r over the streams in order (rows r, r + items...
// when the shape has more rows than the work-group has items: M 8 at KS 1).
template <int kMl, DenseKind KIND, int KS, int MC = kMl>
sycl::event up_mix(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, const DenseWeights &up, const bf16 *xn,
                   bf16 *out) {
  const int H = s.hidden, R = s.lowrank, ctiles = H / kW, tiles = up.n / kW, groups = R / kGroup, tail = (R % kGroup) / kKCall;
  const int calls = groups * kDenseCalls + tail;
  const void *w = up.w;
  const float *scale = up.scale;
  const float inv_s = 1.f / static_cast<float>(kStreams);
  constexpr int kM = MC;  // rows computed (MC = rows for 1-3 live rows at k split 1; the mix writes rows < rows)
  // 16 rows = two 8-row halves per item, each call's weights loaded once for both, the 8-row DPAS chain per half (the
  // same bits per row as two 8-row launches); 32 rows = four 8-row quarters
  constexpr int kHS = kM > 8 ? kM / 8 : 1, kR = kM / kHS;
  constexpr int kAcc = kM * kW, kAccH = kR * kW;
  return q.parallel_for<HcUpMix<kMl, KIND, KS, MC>>(
      sycl::nd_range<1>(static_cast<size_t>(ctiles) * kStreams * KS, kStreams * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<(kStreams * KS + kStreams) * kAcc * sizeof(float)>();
        const int ct = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
        const int st = lid / KS, part = lid % KS, t = st * ctiles + ct;
        const int c0 = part * calls / KS, c1 = (part + 1) * calls / KS;
        const size_t tbase = static_cast<size_t>(groups) * tiles * kDenseCalls * kCallElems + static_cast<size_t>(t) * tail * kCallElems;
        esimd::simd<float, kAcc> acc(0.f);  // [half][kR][16]
        auto dpas_h = [&](int hh, esimd::simd<bf16, kCallElems> bw, esimd::simd<bf16, kR * kKCall> at) SYCL_ESIMD_FUNCTION {
          esimd::simd<float, kAccH> ah = acc.template select<kAccH, 1>(hh * kAccH);
          ah = xmx::dpas<8, kR, float, float, bf16, bf16>(ah, bw, at);
          acc.template select<kAccH, 1>(hh * kAccH) = ah;
        };
        if constexpr (KS == 1) {
          // One item per stream: silu a whole 128-k group at once (fewer, larger loads).
          for (int g = 0; g < groups; ++g) {
            const size_t gb = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems);
#pragma unroll
            for (int hh = 0; hh < kHS; ++hh) {
              esimd::simd<bf16, kR * kGroup> a = silu_rows<kR, kGroup>(lora + static_cast<size_t>(hh * kR) * lora_ld, lora_ld, g * kGroup, inv_s);
#pragma unroll
              for (int j = 0; j < kDenseCalls; ++j) {
                esimd::simd<bf16, kR * kKCall> at;
#pragma unroll
                for (int r = 0; r < kR; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * kGroup + j * kKCall);
                dpas_h(hh, load_call<KIND>(w, gb + j * kCallElems), at);
              }
            }
          }
          for (int j = 0; j < tail; ++j) {
            const esimd::simd<bf16, kCallElems> bw = load_call<KIND>(w, tbase + j * kCallElems);
#pragma unroll
            for (int hh = 0; hh < kHS; ++hh)
              dpas_h(hh, bw, silu_rows<kR, kKCall>(lora + static_cast<size_t>(hh * kR) * lora_ld, lora_ld, groups * kGroup + j * kKCall, inv_s));
          }
        } else
        for (int ci = c0; ci < c1; ++ci) {
          size_t base;
          int k0;
          if (ci < groups * kDenseCalls) {
            const int g = ci / kDenseCalls, j = ci % kDenseCalls;
            base = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems) + static_cast<size_t>(j) * kCallElems;
            k0 = g * kGroup + j * kKCall;
          } else {
            const int j = ci - groups * kDenseCalls;
            base = tbase + static_cast<size_t>(j) * kCallElems;
            k0 = groups * kGroup + j * kKCall;
          }
          const esimd::simd<bf16, kCallElems> bw = load_call<KIND>(w, base);
#pragma unroll
          for (int hh = 0; hh < kHS; ++hh) dpas_h(hh, bw, silu_rows<kR, kKCall>(lora + static_cast<size_t>(hh * kR) * lora_ld, lora_ld, k0, inv_s));
        }
        constexpr uint32_t kSig = kStreams * KS * kAcc * sizeof(float);  // sigmoid(gate) slots after the parts
        if constexpr (KS > 1) {
#pragma unroll
          for (int hh = 0; hh < kHS; ++hh)
            esimd::slm_block_store<float, kAccH>(static_cast<uint32_t>((lid * kAcc + hh * kAccH) * sizeof(float)), acc.template select<kAccH, 1>(hh * kAccH));
          esimd::barrier();
          if (part == 0)
#pragma unroll
            for (int pp = 1; pp < KS; ++pp)
#pragma unroll
              for (int hh = 0; hh < kHS; ++hh)
                acc.template select<kAccH, 1>(hh * kAccH) +=
                    esimd::slm_block_load<float, kAccH>(static_cast<uint32_t>(((lid + pp) * kAcc + hh * kAccH) * sizeof(float)));
        }
        if (part == 0) {
          if constexpr (KIND == DenseKind::i8) {
            const esimd::simd<float, kW> sc = esimd::block_load<float, kW>(scale + t * kW);
#pragma unroll
            for (int r = 0; r < kM; ++r) acc.template select<kW, 1>(r * kW) = acc.template select<kW, 1>(r * kW) * sc;
          }
#pragma unroll
          for (int hh = 0; hh < kHS; ++hh) {
            const esimd::simd<float, kAccH> gate = esimd::simd<bf16, kAccH>(acc.template select<kAccH, 1>(hh * kAccH));
            esimd::slm_block_store<float, kAccH>(kSig + static_cast<uint32_t>((st * kAcc + hh * kAccH) * sizeof(float)), sigmoid_v<kAccH>(gate));
          }
        }
        esimd::barrier();
        for (int r = lid; r < rows; r += kStreams * KS) {  // items mix one row each
          esimd::simd<float, kW> mix(0.f);
#pragma unroll
          for (int ss = 0; ss < kStreams; ++ss) {
            const esimd::simd<float, kW> sg = esimd::slm_block_load<float, kW>(kSig + static_cast<uint32_t>((ss * kAcc + r * kW) * sizeof(float)));
            const esimd::simd<float, kW> x = esimd::block_load<bf16, kW>(xn + (static_cast<size_t>(r) * kStreams + ss) * H + ct * kW);
            mix = sg * x + mix;
          }
          esimd::block_store<bf16, kW>(out + static_cast<size_t>(r) * H + ct * kW, esimd::simd<bf16, kW>(mix * inv_s));
        }
      });
}

template <int M, DenseKind KIND>
sycl::event up_mix_ks(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, const DenseWeights &up, const bf16 *xn,
                      bf16 *out, int ks) {
  if constexpr (M == 4) {  // only the live rows (bit-identical per row)
    if (ks <= 1 && rows == 1) return up_mix<4, KIND, 1, 1>(q, s, rows, lora, lora_ld, up, xn, out);
    if (ks <= 1 && rows == 2) return up_mix<4, KIND, 1, 2>(q, s, rows, lora, lora_ld, up, xn, out);
    if (ks <= 1 && rows == 3) return up_mix<4, KIND, 1, 3>(q, s, rows, lora, lora_ld, up, xn, out);
  }
  if constexpr (M == 8) {  // 5-7 live rows on the 8-row shape
    auto mc = [&](auto kst) {
      constexpr int KS = decltype(kst)::value;
      if (rows == 5) return up_mix<8, KIND, KS, 5>(q, s, rows, lora, lora_ld, up, xn, out);
      if (rows == 6) return up_mix<8, KIND, KS, 6>(q, s, rows, lora, lora_ld, up, xn, out);
      return up_mix<8, KIND, KS, 7>(q, s, rows, lora, lora_ld, up, xn, out);
    };
    if (rows >= 5 && rows <= 7) {
      if (ks >= 4) return mc(std::integral_constant<int, 4>{});
      if (ks >= 2) return mc(std::integral_constant<int, 2>{});
      return mc(std::integral_constant<int, 1>{});
    }
  }
  if (ks >= 4) return up_mix<M, KIND, 4>(q, s, rows, lora, lora_ld, up, xn, out);
  if (ks >= 2) return up_mix<M, KIND, 2>(q, s, rows, lora, lora_ld, up, xn, out);
  return up_mix<M, KIND, 1>(q, s, rows, lora, lora_ld, up, xn, out);
}
template <DenseKind KIND>
sycl::event up_mix_shape(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, const DenseWeights &up, const bf16 *xn,
                         bf16 *out, int ks) {
  if (rows > kStepRows16)  // the 32-row shape, four 8-row quarters per item
    return up_mix_ks<32, KIND>(q, s, rows, lora, lora_ld, up, xn, out, ks);
  if (rows > kStepRows8)  // the 16-row shape, two 8-row halves per item
    return up_mix_ks<16, KIND>(q, s, rows, lora, lora_ld, up, xn, out, ks);
  return row_shape(rows) == 4 ? up_mix_ks<4, KIND>(q, s, rows, lora, lora_ld, up, xn, out, ks) : up_mix_ks<8, KIND>(q, s, rows, lora, lora_ld, up, xn, out, ks);
}

// ---- many rows: the up GEMM + gate mix in one launch ------------------------------------------------------
// hc.cpp's sigmoidf, 1 / (1 + sycl:exp(-x)), as IGC lowers it under the build's fast math on bmg (read from the asm):
// n = trunc(x * -log2 e), r = -x - n ln2_hi - n ln2_lo (two fused mads), 2^n * 2^(r log2 e) fused into the + 1, a hardware
// reciprocal, 1 above x = 105 and 0 below -105. Equal bit for bit to the SYCL function on all 65,536 bf16 inputs (the gate is
// bf16), so the fused mixer keeps hc_gate_mix16's bits; test-head gates it end to end against the two-launch path.
template <int N>
inline esimd::simd<float, N> sigmoid_sycl_bits(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  const esimd::simd<float, N> n = esimd::rndz<float>(x * sycl::bit_cast<float>(0xBFB8AA3Bu));
  esimd::simd<float, N> r = xesimd::fma(n, esimd::simd<float, N>(sycl::bit_cast<float>(0xBF317200u)), -x);
  r = xesimd::fma(n, esimd::simd<float, N>(sycl::bit_cast<float>(0xB5BFBE8Eu)), r);
  const esimd::simd<float, N> r2 = r * sycl::bit_cast<float>(0x3FB8AA3Bu);
  esimd::simd<float, N> s = esimd::inv(xesimd::fma(esimd::exp2(n), esimd::exp2(r2), esimd::simd<float, N>(1.f)));
  s.merge(esimd::simd<float, N>(1.f), x > 105.f);
  s.merge(esimd::simd<float, N>(0.f), x < -105.f);
  return s;
}

constexpr int kRowsMB = 32;  // rows per work-item (4 DPAS row blocks per widened weight call)
constexpr int kRowsRB = 2;  // row blocks per work-group: work-group = 4 streams x 2 row blocks = 8 items, 64 rows

template <DenseKind KIND>
class HcUpMixRows;

// Work-group = one 16-column tile c of the hidden x 64 rows; item (stream st, row block rb) runs the up GEMM for the
// tile st * H + c over its 32 rows exactly as pf_dense's bodies do per row (dpas 8 x 8 from zero, the 16-k calls in k
// order, the int8 column scale on the finished sum, bf16 rounding), widening its weight calls straight from memory
// (the work-group's two row blocks of a stream share them through L1). The items load their mix rows of xn before the
// k loop, leave sigmoid(gate) in SLM, and after one barrier mix rows over the streams in stream order with
// hc_gate_mix16's arithmetic (fma from zero, / S).
template <DenseKind KIND>
sycl::event up_mix_rows(sycl::queue &q, const HcShape &s, int m, const DenseWeights &up, const bf16 *sv, const bf16 *xn, bf16 *out) {
  constexpr int kItems = kStreams * kRowsRB, kAcc = kRowsMB * kW, P = kRowsMB / kStreams;  // P mix rows per item
  const int H = s.hidden, R = s.lowrank, ctiles = H / kW, tiles = up.n / kW, full = R / kGroup, tail = (R % kGroup) / kKCall;
  const int rgs = (m + kRowsRB * kRowsMB - 1) / (kRowsRB * kRowsMB);
  const void *wp = up.w;
  const float *scale = up.scale;
  return q.parallel_for<HcUpMixRows<KIND>>(
      sycl::nd_range<2>({static_cast<size_t>(rgs), static_cast<size_t>(ctiles) * kItems}, {1, kItems}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kStreams * kRowsRB * kAcc * sizeof(float)>();
        // a (row group, column tile) grid: the ids come from the launch, not from a division by the run-time tile count in every item
        const int rg = static_cast<int>(it.get_group(0)), ct = static_cast<int>(it.get_group(1)), lid = static_cast<int>(it.get_local_id(1));
        const int st = lid / kRowsRB, rb = lid % kRowsRB;
        const int m0 = (rg * kRowsRB + rb) * kRowsMB, w0 = rg * kRowsRB * kRowsMB;
        esimd::simd<bf16, P * kStreams * kW> xp;  // mix row w0 + lid + i * kItems
#pragma unroll
        for (int i = 0; i < P; ++i) {
          const int r0 = w0 + lid + i * kItems, row = r0 < m ? r0 : m - 1;
#pragma unroll
          for (int ss = 0; ss < kStreams; ++ss)
            xp.template select<kW, 1>((i * kStreams + ss) * kW) =
                esimd::block_load<bf16, kW>(xn + (static_cast<size_t>(row) * kStreams + ss) * H + ct * kW);
        }
        if (m0 < m) {
          const uint8_t *wb = static_cast<const uint8_t *>(wp);
          const uint16_t *a16 = reinterpret_cast<const uint16_t *>(sv);
          const unsigned sw = static_cast<unsigned>(R * 2 - 1), sh = static_cast<unsigned>(m - 1);
          const int tt = st * ctiles + ct;
          esimd::simd<float, kAcc> acc(0.f);
          for (int g = 0; g < full + (tail > 0 ? 1 : 0); ++g) {
            const bool is_tail = g >= full;
            const int calls = is_tail ? tail : kDenseCalls;
            for (int j = 0; j < calls; ++j) {
              esimd::simd<uint16_t, kRowsMB * kKCall> au =
                  esimd::load_2d<uint16_t, kKCall, kRowsMB>(a16, sw, sh, sw, g * kGroup + j * kKCall, m0);
              esimd::simd<bf16, kRowsMB * kKCall> av = au.template bit_cast_view<bf16>();
              const size_t e = is_tail ? static_cast<size_t>(full) * tiles * kDenseCalls * kCallElems +
                                             (static_cast<size_t>(tt) * tail + j) * kCallElems
                                       : ((static_cast<size_t>(g) * tiles + tt) * kDenseCalls + j) * kCallElems;
              esimd::simd<bf16, kCallElems> b;
              if constexpr (KIND == DenseKind::i8) {
                // |code| <= 127: float(code) has 16 zero low bits, so its bf16 is the high half (exact)
                esimd::simd<float, kCallElems> f = esimd::block_load<int8_t, kCallElems>(static_cast<const int8_t *>(wp) + e);
                esimd::simd<uint16_t, kCallElems> hi = f.template bit_cast_view<uint16_t>().template select<kCallElems, 2>(1);
                b = hi.template bit_cast_view<bf16>();
              } else {
                b = esimd::block_load<bf16, kCallElems>(reinterpret_cast<const bf16 *>(wb) + e);
              }
#pragma unroll
              for (int r = 0; r < kRowsMB / 8; ++r) {
                auto cv = acc.template select<128, 1>(r * 8 * kW);
                const esimd::simd<float, 128> c = cv;
                cv = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b, av.template select<128, 1>(r * 128).read());
              }
            }
          }
          esimd::simd<float, kW> sc(1.f);
          if constexpr (KIND == DenseKind::i8) sc = esimd::block_load<float, kW>(scale + tt * kW);
#pragma unroll
          for (int r = 0; r < kRowsMB; ++r) {
            esimd::simd<float, kW> v = acc.template select<kW, 1>(r * kW);
            if constexpr (KIND == DenseKind::i8) v *= sc;
            acc.template select<kW, 1>(r * kW) = sigmoid_sycl_bits<kW>(esimd::simd<float, kW>(esimd::simd<bf16, kW>(v)));
          }
          esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>((st * kRowsRB + rb) * kAcc * sizeof(float)), acc);
        }
        esimd::barrier();
#pragma unroll
        for (int i = 0; i < P; ++i) {
          const int rr = lid + i * kItems, row = w0 + rr;
          esimd::simd<float, kW> mix(0.f);
#pragma unroll
          for (int ss = 0; ss < kStreams; ++ss) {
            const esimd::simd<float, kW> sg = esimd::slm_block_load<float, kW>(
                static_cast<uint32_t>(((ss * kRowsRB + rr / kRowsMB) * kAcc + (rr % kRowsMB) * kW) * sizeof(float)));
            mix = xesimd::fma(sg, esimd::simd<float, kW>(xp.template select<kW, 1>((i * kStreams + ss) * kW)), mix);
          }
          if (row < m) esimd::block_store<bf16, kW>(out + static_cast<size_t>(row) * H + ct * kW, esimd::simd<bf16, kW>(mix / static_cast<float>(kStreams)));
        }
      });
}

}  // namespace

sycl::event hc_up_mix_rows(sycl::queue &q, const HcShape &s, int m, const DenseWeights &up, const hc_bf16 *sv, const hc_bf16 *xn,
                           hc_bf16 *out) {
  if (!hc_fast_supported(s) || !up.valid() || up.n != s.width() || up.k != s.lowrank || m < 1 ||
      (up.kind != DenseKind::i8 && up.kind != DenseKind::bf16))
    throw std::invalid_argument("hc_up_mix_rows: unsupported shape");
  return up.kind == DenseKind::i8 ? up_mix_rows<DenseKind::i8>(q, s, m, up, sv, xn, out) : up_mix_rows<DenseKind::bf16>(q, s, m, up, sv, xn, out);
}

bool hc_fast_supported(const HcShape &s) {
  return s.streams == kStreams && s.hidden % (kNormParts * kNormChunk) == 0 && s.lowrank % kKCall == 0 &&
         s.lowrank <= kMaxLowrank;
}

sycl::event hc_combine_norm_fast(sycl::queue &q, const HcShape &s, int rows, hc_bf16 *res, const hc_bf16 *block, const float *inj,
                                 int inj_ld, int inj_col, const hc_bf16 *norm_w, hc_bf16 *xn) {
  if (s.hidden % (kNormParts * kNormChunk)) throw std::invalid_argument("hc_combine_norm_fast: hidden % 512");
  if (block && !inj) throw std::invalid_argument("hc_combine_norm_fast: a pending block needs its injection logits");
  if (s.hidden == 2560)  // 320 values per item in registers
    return block ? norm_reg<true, 320>(q, s, rows, res, block, inj, inj_ld, inj_col, norm_w, xn)
                 : norm_reg<false, 320>(q, s, rows, res, nullptr, nullptr, 0, 0, norm_w, xn);
  return block ? norm_fast<true>(q, s, rows, res, block, inj, inj_ld, inj_col, norm_w, xn)
               : norm_fast<false>(q, s, rows, res, nullptr, nullptr, 0, 0, norm_w, xn);
}

bool hc_sum_norm_supported(const HcShape &s) { return s.streams == kStreams && s.hidden == kSumNormHidden; }

sycl::event hc_sum_combine_norm(sycl::queue &q, const HcShape &s, const QuadOwnerSum &sum, hc_bf16 *res, const float *inj, int inj_ld, int inj_col,
                                const hc_bf16 *norm_w, hc_bf16 *xn) {
  if (!hc_sum_norm_supported(s) || sum.n != s.hidden) throw std::invalid_argument("hc_sum_combine_norm: 4 streams of 2560, the sum over hidden");
  if (sum.rows < 1 || !sum.own || !sum.out || !sum.recv_flags || !sum.error || !inj || !res || !norm_w || !xn)
    throw std::invalid_argument("hc_sum_combine_norm: rows, the sum's buffers, the injection logits, res, the norm weights and xn");
  for (int c = 0; c < kWireCardsMax; ++c)
    if (c != sum.card && !sum.slot[c]) throw std::invalid_argument("hc_sum_combine_norm: every source card's slot");
  const bool add = sum.add_p != nullptr;
  if (add != (sum.add_scale != nullptr)) throw std::invalid_argument("hc_sum_combine_norm: the shared term's rows and scales together");
  if (add) return sum.bf16 ? sum_norm<true, true>(q, s, sum, res, inj, inj_ld, inj_col, norm_w, xn) : sum_norm<true, false>(q, s, sum, res, inj, inj_ld, inj_col, norm_w, xn);
  return sum.bf16 ? sum_norm<false, true>(q, s, sum, res, inj, inj_ld, inj_col, norm_w, xn) : sum_norm<false, false>(q, s, sum, res, inj, inj_ld, inj_col, norm_w, xn);
}

sycl::event hc_up_mix(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, const DenseWeights &up, const hc_bf16 *xn,
                      hc_bf16 *out, int ksplit) {
  if (ksplit <= 0) ksplit = kUpMixKSplit;
  if (!hc_fast_supported(s) || !up.valid() || up.n != s.width() || up.k != s.lowrank || rows < 1 || rows > kStepRows32)
    throw std::invalid_argument("hc_up_mix: unsupported shape");  // 32 rows at most
  return up.kind == DenseKind::i8 ? up_mix_shape<DenseKind::i8>(q, s, rows, lora, lora_ld, up, xn, out, ksplit)
                                  : up_mix_shape<DenseKind::bf16>(q, s, rows, lora, lora_ld, up, xn, out, ksplit);
}

}  // namespace xenithium:kernels

