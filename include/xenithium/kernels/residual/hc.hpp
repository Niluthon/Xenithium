// A position carries S residual streams of `hidden` each, laid out [row][stream][hidden].
// A mixer turns them into one block input (written from the model's definition):
// 1. combine + norm (per row, stream):
// res' = bf16(res + block * 2 sigmoid(bf16(inj_s) / S)) (only with a pending block)
// xn = bf16(fma(y, w, y)), y = res' * rsqrt(mean(res'^2) + eps) (Gemma (1 + w) affine,
// w per stream element)
// 2. lora = bf16(xn. Wdown^T) (dense GEMM, k = S * hidden)
// s = bf16(silu(lora / S)) (hc_silu)
// 3. gate = bf16(s. Wup^T) (dense GEMM, n = S * hidden)
// The GEMMs are kernels:dense_gemm_m4; this header holds the element-wise launches and the
// host reference of the whole mixer with the same rounding points.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/kernels/gemm/dense.hpp"

namespace xenithium::kernels {

struct HcShape {
  int streams = 4;
  int hidden = 2560;
  int lowrank = 320;
  float eps = 1e-6f;
  int width() const { return streams * hidden; }
};

using hc_bf16 = sycl::ext::oneapi::bfloat16;

// Step 1. `block` null = no pending combine (res is only read). inj: fp32 down-GEMM output,
// row stride inj_ld, the stream logits at columns [inj_col, inj_col + S). xn: [rows][S*hidden].
sycl::event hc_combine_norm(sycl::queue &q, const HcShape &s, int rows, hc_bf16 *res, const hc_bf16 *block, const float *inj,
                            int inj_ld, int inj_col, const hc_bf16 *norm_w, hc_bf16 *xn);
// lora fp32 [rows][lora_ld] (first `lowrank` columns) -> s bf16 [rows][lowrank].
sycl::event hc_silu(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, hc_bf16 *out);
// Many-row down epilogue: lora[r][c] = (sum over p in order of partials[p][r][c]) * scale[c]
// for c < n (pf_splitk_sum, row stride n, part stride m * n) and s = hc_silu of its first lowrank
// columns, in one launch.
sycl::event hc_sum_silu(sycl::queue &q, const HcShape &s, int m, int n, int ks, const float *partials, const float *scale, float *lora,
                        hc_bf16 *out);
// gate fp32 [rows][S*hidden], xn bf16 [rows][S*hidden] -> out bf16 [rows][hidden].
sycl::event hc_gate_mix(sycl::queue &q, const HcShape &s, int rows, const float *gate, const hc_bf16 *xn, hc_bf16 *out);

// Fast path (ESIMD, hc_fused.cpp): 4 streams, hidden % 512 == 0, lowrank % 16 == 0 and <= 512.
bool hc_fast_supported(const HcShape &s);
sycl::event hc_combine_norm_fast(sycl::queue &q, const HcShape &s, int rows, hc_bf16 *res, const hc_bf16 *block, const float *inj,
                                 int inj_ld, int inj_col, const hc_bf16 *norm_w, hc_bf16 *xn);
// silu + up GEMM + gate mix in one launch: lora fp32 [4][lora_ld] -> out bf16 [rows][hidden].
// The up matrix is the dense-packed [S*hidden][lowrank]; rows <= 4 (the GEMM always runs 4).
// ksplit: work-items per stream sharing the k calls (1, 2 or 4; 0 = the default).
sycl::event hc_up_mix(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, const DenseWeights &up, const hc_bf16 *xn,
                      hc_bf16 *out, int ksplit = 0);
// hc_combine_norm_fast with a pending block whose rows are still a four-card owner sum (comm/wire.hpp QuadOwnerSum):
// one launch waits for the other cards' partials, writes the sum's own rows (pair_split_finish's bits; its other readers
// find them as before) and combines + norms those rows, the same bits as pair_split_finish + hc_combine_norm_fast on
// res / inj / xn at the sum's first own row. Saves the sum's wait and sum launches. 4 streams of hidden 2560.
bool hc_sum_norm_supported(const HcShape &s);
sycl::event hc_sum_combine_norm(sycl::queue &q, const HcShape &s, const QuadOwnerSum &sum, hc_bf16 *res, const float *inj, int inj_ld, int inj_col,
                                const hc_bf16 *norm_w, hc_bf16 *xn);
// ---- one whole mixer ------------------------------------------------------------------------
struct HcWeights {
  const hc_bf16 *norm = nullptr;  // [S * hidden]
  DenseWeights down;  // [lowrank (+ S injection rows), padded to 16][S * hidden]
  DenseWeights up;  // [S * hidden][lowrank]
};
// A position's multi-stream state on one card. The down output is double buffered: a mixer's
// down output (with its injection logits) lands in lora[lcur ^ 1] while the logits it
// consumes stay readable in lora[lcur], so no launch has to read them before another
// overwrites them. The residual is updated in place by the mixers; the standalone combine
// writes res[cur ^ 1] and flips. Indices flip on the host at submission, which matches the
// device order on an in-order queue.
struct HcState {
  hc_bf16 *res[2] = {nullptr, nullptr};  // [4][S * hidden] each
  float *lora[2] = {nullptr, nullptr};  // [4][>= any down.n] each
  int cur = 0, lcur = 0;
  int lora_ld = 0;  // row stride of lora[lcur] (logits at [lowrank, lowrank + S))
  hc_bf16 *residual() const { return res[cur]; }
  float *down_out() const { return lora[lcur]; }
};
struct HcScratch {
  hc_bf16 *xn = nullptr;  // [4][S * hidden]
  hc_bf16 *s = nullptr;  // unfused path only: [4][lowrank]
  float *gate = nullptr;  // unfused path only: [4][S * hidden]
};
enum class HcPath : uint8_t {
  unfused,  // 5 launches: combine+norm, down, silu, up, gate mix (SYCL element-wise)
  three,  // 3: combine+norm (ESIMD), down, silu+up+gate mix
};
struct HcEvents {
  sycl::event norm, down, up;  // up = the last launch
};
// [combine with the pending block (non-null) and the injection logits in st.lora[st.lcur]
// +] norm -> down(|inject) -> silu -> up -> gate mix -> out. Falls back to the unfused path
// when the shape does not support the fast one.
HcEvents hc_mixer(sycl::queue &q, const HcShape &s, int rows, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcScratch &scratch,
                  hc_bf16 *out, HcPath path = HcPath::three, int down_ksplit = 0, int up_ksplit = 0);
// The same mixer on rows `own` of the state (res, the injection logits, the down output), of block and of out; the
// scratch from its start. For a card that runs only the rows of a step it owns (comm/wire.hpp, rows by owner): each row
// gets hc_mixer's bits. The down GEMM writes its whole row shape, so the state's buffers must hold
// own.lo + row_shape(own.rows) rows, and the rows past own.hi it overwrites must be this card's to lose.
HcEvents hc_mixer_at(sycl::queue &q, const HcShape &s, RowRange own, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcScratch &scratch,
                     hc_bf16 *out, HcPath path = HcPath::three, int down_ksplit = 0, int up_ksplit = 0);
// Standalone combine (no norm): a block that adds to the multi-stream state itself (PLE)
// needs the pending combine materialised first. Writes res[cur ^ 1] and flips. Any row count
// (decode: rows past `rows` up to 4 are carried over; prefill: exactly `rows`).
sycl::event hc_combine(sycl::queue &q, const HcShape &s, int rows, HcState &st, const hc_bf16 *block);
// The same on rows `own` only (hc_mixer_at's use): the other rows of res[cur ^ 1] are left as they are.
sycl::event hc_combine_at(sycl::queue &q, const HcShape &s, RowRange own, HcState &st, const hc_bf16 *block);

// ---- many rows (prefill) ---------------------------------------------------------------
// The same mixer over m rows of a chunk (any m): combine + norm (the fast ESIMD launch), the down
// GEMM as pf_dense split-K + an in-order sum with the silu (hc_sum_silu), the up GEMM + gate mix (one launch above
// kHcUpMixRowsMin rows, else pf_dense + gate mix). Rounding points as hc_mixer; the GEMMs sum k in another order than
// the decode GEMMs (fp32 order only).
// st's buffers hold m rows (res: [m][S * hidden], lora: [m][>= down.n] at row stride down.n).
struct HcRowsScratch {
  hc_bf16 *xn = nullptr;  // [m][S * hidden]
  hc_bf16 *s = nullptr;  // [m][lowrank]
  float *gate = nullptr;  // [m][S * hidden] fp32-sized; holds bf16 (only the two-launch up + mix)
  float *part = nullptr;  // [kHcDownKs][m][down.n]
  PfA8 a8;  // set = the down and up GEMMs run W8A8 (pf_dense.hpp)
  bool split_up_mix = false;  // true = the up GEMM and the gate mix as two launches at every row count (tests)
};
// Calls of more than kHcUpMixRowsMin rows (bf16 activations) run the up GEMM + gate mix as one launch
// (hc_up_mix_rows): the same bits as pf_dense + hc_gate_mix16, without the gate's write and read; shorter calls keep the
// two launches (pf_dense's direct kernel is faster there).
inline constexpr int kHcUpMixRowsMin = 16;
// The up GEMM (pf_dense's per-row DPAS order, the int8 column scale, bf16 gate) + gate mix over m rows in one launch:
// s bf16 [m][lowrank], xn [m][S * hidden] -> out [m][hidden]. 4 streams, bf16 or int8 weights.
sycl::event hc_up_mix_rows(sycl::queue &q, const HcShape &s, int m, const DenseWeights &up, const hc_bf16 *sv, const hc_bf16 *xn,
                           hc_bf16 *out);
// down GEMM k split of the many-row mixer (n = 336 alone gives 6 column blocks).
inline constexpr int kHcDownKs = 4;
HcEvents hc_mixer_rows(sycl::queue &q, const HcShape &s, int m, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcRowsScratch &sc,
                       hc_bf16 *out);
// The same mixer in row parts: rows are independent and the down split is fixed by shape,
// so parts in any order == one call, bit for bit. hc_mixer_rows_part runs rows [r0, r1) of the
// chunk's buffers (res, lora, block, out at row r0; the scratch from its start, so parts must run in
// order on one queue) and leaves st as it is; after the last part, hc_mixer_rows_flip(st, w) once.
// Use: start on the rows a card already holds while the peer's rows of `block` still arrive.
// block_sum: null, or the pending owner sum that writes block's rows [r0, r1) (its out at block + r0 x hidden, its rows
// r1 - r0): the first launch is then hc_sum_combine_norm (hc_sum_norm_supported shapes), the same bits.
HcEvents hc_mixer_rows_part(sycl::queue &q, const HcShape &s, int r0, int r1, const HcState &st, const hc_bf16 *block, const HcWeights &w,
                            const HcRowsScratch &sc, hc_bf16 *out, const QuadOwnerSum *block_sum = nullptr);
void hc_mixer_rows_flip(HcState &st, const HcWeights &w);

// ---- host reference ------------------------------------------------------------------------
// A matrix as the device holds it, for the reference: value(n, k) = w[n * k_dim + k] * scale[n]
// summed as (sum_k x * w) * scale. bf16 weights: w = the bf16 values, scale = 1.
struct RefMatrix {
  const float *w = nullptr;
  const float *scale = nullptr;  // null = 1
  int n = 0, k = 0;
};
// Whole mixer for `rows` rows (bf16 tensors as uint16 bits). Mirrors steps 1-3, GEMM sums in
// double then rounded to bf16 (the device rounds an fp32 sum: they can differ by one ulp).
// res is updated in place when block is non-null. inj: [rows][S] fp32.
// down may carry more rows than lowrank (the injection rows): lora_out then gets all down.n
// columns per row ([rows][down.n], the injection logits at [lowrank, lowrank + S)).
void hc_mix_ref(const HcShape &s, int rows, uint16_t *res, const uint16_t *block, const float *inj, const uint16_t *norm_w,
                const RefMatrix &down, const RefMatrix &up, uint16_t *xn_out, float *lora_out, uint16_t *out);

}  // namespace xenithium:kernels
