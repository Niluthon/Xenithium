// Xenithium · dense decode GEMM: y (4 x n fp32) = x (4 x k bf16) . W^T, W the checkpoint's
// [n][k] matrix packed by pack_dense_bf16 / pack_dense_i8 (pack.hpp).
//
//   bf16 weights: bf16 DPAS on the stored values.
//   int8 weights: codes widened to bf16 in registers (exact, |q| <= 127), bf16 DPAS into an
//                 fp32 sum, the row's fp32 scale applied once to the finished sum.
// One work-group per 16-column tile; `ksplit` work-items share the full 128-k groups, the
// last one also takes a k % 128 tail (calls of 16 k, no padding read); partials meet in SLM
// and are summed in part order (deterministic). The split depends only on (n, k), never on
// which card runs it, so a TP slice and the whole matrix round alike.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {

// i2 (the MTP draft head): 2-bit codes q in 0..3 meaning (q - 1.5) x the (row, 128-k group)
// fp32 scale; codes at the int8 layout's element positions, 4 per byte (element e: bits 2 (e % 4) of
// byte e / 4); scales [k / 128][n]. k % 128 == 0.
enum class DenseKind : uint8_t { bf16, i8, i2 };

struct DenseWeights {
  const void *w = nullptr;        // packed tiles
  const float *scale = nullptr;   // int8: one fp32 per output row; i2: [k / 128][n]; bf16: null
  DenseKind kind = DenseKind::bf16;
  int n = 0, k = 0;
  bool valid() const {
    return w && n > 0 && k > 0 && n % 16 == 0 && k % 16 == 0 && (kind == DenseKind::bf16 || scale) && (kind != DenseKind::i2 || k % 128 == 0);
  }
  size_t bytes() const {
    const size_t nk = static_cast<size_t>(n) * k;
    switch (kind) {
      case DenseKind::i8: return nk + n * 4ull;
      case DenseKind::i2: return nk / 4 + static_cast<size_t>(n) * (k / 128) * 4;
      default: return nk * 2;
    }
  }
};

// W8A8 prefill (pf_dense.hpp): the caller-owned scratch of a many-row GEMM's int8 activations, q
// [m][k] int8 (row stride k) and s = the fp32 group scales, group-major [ceil(k / 128)][pf_a8_ld(m)] (the
// GEMM bodies block-load a row block's 16 scales of a group; the stride is m rounded up to 16, the pad never
// read into a stored row). pf_a8_scales(m, k) floats. A null q = bf16 activations.
// One per queue is enough: every GEMM quantises into it right before reading it.
struct PfA8 {
  int8_t *q = nullptr;
  float *s = nullptr;
  // W8A8 per row: one scale per activation row over the whole k (s[r], r < m; pf_a8_ld(m) floats suffice):
  // the int32 sum of a (row, column) over all of k is exact (k x 127 x 127 < 2^31 for k <= 133,000), so the body keeps
  // one int32 accumulator, no per-group fold, and out = float(sum) x s[r] x w_scale[col] - the same bits whatever the
  // body, the tiles per work-group or the rows of the call. Not with split-K (the many-row HC down keeps per group).
  bool row = false;
  // q / s already hold the call's rows quantised (pf_a8_quantize, or another card's copy of its rows): the GEMM skips
  // its quantiser. Per row only (a row's scale does not depend on the call; the group scales' stride does).
  bool ready = false;
  const PfA8 *on() const { return q ? this : nullptr; }
};
inline constexpr int pf_a8_ld(int m) { return (m + 15) / 16 * 16; }
inline constexpr size_t pf_a8_scales(int m, int k) { return static_cast<size_t>((k + 127) / 128) * static_cast<size_t>(pf_a8_ld(m)); }

inline constexpr int kDenseRows = kDecodeRows;  // M of dense_gemm_m4 (one decode step of 1 + 3 drafts)
int dense_auto_ksplit(int n, int k);
sycl::event dense_gemm_m4(sycl::queue &q, const sycl::ext::oneapi::bfloat16 *x, const DenseWeights &w, float *y, int ksplit = 0);
// The same GEMM at the row shape of `rows` live rows (rows.hpp: 1-4 -> M 4, 5-8 -> M 8, 9-16 -> M 16; 17-32 ->
// M 32: four 8-row quarters per work-group, or two launches at k split 32 / i2 weights):
// x and y hold row_shape(rows) rows (rows past `rows` are never used). Row r's
// bits do not depend on the shape, and the k split depends only on (n, k). 33-64 rows = one 64-row launch or two
// launches (rows 0-31, then the rest); x and y then hold 64 rows.
sycl::event dense_gemm(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *x, const DenseWeights &w, float *y, int ksplit = 0);
// (the two-stage lm_head): y[r][t*16 .. t*16 + 16) for the 16-column tiles t = tiles[r * per_row + i]
// (i < per_row) of each row r < rows only, every other y element untouched. One work-group per (row, listed tile) runs
// dense_gemm's tile body for that one row: the same bits as dense_gemm(q, any rows, x, w, y) at those positions, which
// requires dense_auto_ksplit(n, k) == 1 (the whole k in one item; true for the lm_head slices: thousands of tiles) - throws
// otherwise. bf16 / int8 weights.
sycl::event dense_gemm_tiles(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *x, const DenseWeights &w, float *y, const int32_t *tiles,
                             int per_row);
// dense_gemm (bf16 weights) whose launch also runs a rotate + int8 quantise job in extra
// work-groups (rot_quant.hpp; the GEMM's bits are those of dense_gemm with the same ksplit).
struct RotQuantJob;
sycl::event dense_gemm_rq(sycl::queue &q, int rows, const sycl::ext::oneapi::bfloat16 *x, const DenseWeights &w, float *y, const RotQuantJob &rq,
                          int ksplit = 0);

}  // namespace xenithium::kernels
