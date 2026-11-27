// Xenithium · prefill GEMM on XMX: many rows (a prompt chunk) against the decode kernels' packed
// dense weights, so a chunk reads every weight once instead of once per 4 rows.
// C (m x n) = A (m x k, bf16 row-major, row stride lda). W^T, W the checkpoint's [n][k]
// W is dense.hpp's packing (pack_dense_bf16 / pack_dense_i8): 128-k x 16-column blocks of 8
// VNNI calls, full groups [g][t], then a compact k % 128 tail [t][calls]. int8 codes widen to
// bf16 exactly; the output column's fp32 scale multiplies the finished sum (as decode does).
// m <= 32 with >= 256 (tile, part) items: a direct kernel, one work-item per tile (and k part), no SLM; bit-identical.
// Otherwise work-group = 16 work-items on one column block (NT tiles: 4, or 2 / 1 when that leaves too few
// work-groups) and 16 consecutive row blocks of MB rows. Per 128-k group the work-group widens the column block's weights once into SLM
// (bf16, double-buffered, one barrier per group), shared by its 16 row blocks. A arrives by 2D
// block loads (rows past m read as zero). fp32 accumulation, k summed in order (groups in order,
// 16 k per DPAS call) whatever m is or where a row sits in the chunk: a row's result does not
// depend on the chunk around it. The decode GEMM splits k over work-items (dense_auto_ksplit),
// so prefill and decode outputs differ by fp32 summation order only.
#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"

namespace xenithium::kernels {

inline constexpr int kPfRows = 2048;  // most rows a prefill chunk carries (buffer sizing)

struct PfOut {
  float *f32 = nullptr;  // m x ld
  sycl::ext::oneapi::bfloat16 *b16 = nullptr;  // m x ld
  int ld = 0;  // 0 = n
};

// W8A8. The activations quantised to int8 per (row, 128-k group; a k % 128 tail is its own
// group) with one fp32 scale each (row_quant.hpp's rule, the W4A8 experts' quantiser), run on s8 x s8
// DPAS against the same int8 weight bytes: per group the int32 sum is exact, acc += float(sum) x a_scale
// with the groups in order, the column scale on the finished sum as the bf16-activation path. A numerics
// setting (not the bf16-activation path's bits); a row's result does not depend on the call around it. The caller
// owns the scratch (dense.hpp's PfA8: q [m][k] int8, s [m][ceil(k / 128)] fp32; one per queue). Only
// int8 weights with k % 32 == 0 take it; bf16 weights ignore it (the bf16 path).

// The quantiser a W8A8 GEMM runs on its m rows of a (row stride lda, k columns) into a8.q / a8.s (a8.row: one scale per
// row, else per 128-k group): the same bits, so a GEMM given a8 with PfA8::ready takes them as its own.
sycl::event pf_a8_quantize(sycl::queue &q, int m, const sycl::ext::oneapi::bfloat16 *a, int lda, int k, const PfA8 &a8);

// C = A W^T. A: m rows of stride lda (multiple of 8 elements; 64-byte aligned base).
sycl::event pf_dense(sycl::queue &q, int m, const DenseWeights &w, const sycl::ext::oneapi::bfloat16 *a, int lda,
                     const PfOut &out, const PfA8 *a8 = nullptr);

// A W8A16 call of an int8 matrix with >= min_m rows runs on oneDNN (onednn_pf.hpp onednn_i8_pf; pf_dense, the
// pair's matrices each on their own, pf_dense_splitk; not pf_dense_tiles) only where oneDNN gives the engine's bits: the
// first call of each (n, k, m, strides, output) runs both and compares every output word, and the engine's body stays for a
// key that differs. Bf16 matrices and W8A8 keep the engine's bodies. 0 = off (the default); the initial value comes from
// XEN_PF_DENSE_ONEDNN (probes), servers set it (qwen38: --prefill-gemm with chunks >= 2,048). Process-wide.
// XEN_PF_DENSE_ONEDNN_LOG=1 prints each key's verdict.
void pf_dense_set_onednn(int min_m);
int pf_dense_onednn();
// The largest matrix (n x k codes bytes) the route takes (0 = any): the model sizes the repack slots by it (qwen38: its
// largest trunk matrix, so the lm_head can never grow a slot past the VRAM budget).
void pf_dense_set_onednn_cap(size_t max_codes_bytes);

// pf_dense with a forced kernel: nt 1, 2 or 4 = the SLM kernel with nt 16-column tiles per
// work-group, nt -1 = the direct kernel (m <= 32); pf_dense picks by m and occupancy.
// Bit-identical to pf_dense for every nt: for tests.
sycl::event pf_dense_tiles(sycl::queue &q, int m, const DenseWeights &w, const sycl::ext::oneapi::bfloat16 *a, int lda,
                           const PfOut &out, int nt, const PfA8 *a8 = nullptr);

// An optional third job in the pair's launch, the W4A8 activation quantiser (row_quant.hpp,
// 128-k groups) over m rows of src (row stride cols): it reads only src, so it runs beside the two
// GEMMs; the same body as moe_rows' own launch, the same bits.
struct PfRowQuant {
  const sycl::ext::oneapi::bfloat16 *src = nullptr;  // [m][cols]
  int cols = 0;  // multiple of 128
  int8_t *dst = nullptr;  // [m][cols]
  float *scl = nullptr;  // [m][cols / 128]
  bool rot = false;  // quantise bf16(H_128 src) (rot_quant.hpp), the rotated experts' input
};

// Two independent GEMMs with the same m in ONE launch (C0 = A0 W0^T, C1 = A1 W1^T; the
// outputs must not alias the inputs of either). Each matrix runs the kernel body pf_dense would pick for
// it, so both outputs are bit-identical to two pf_dense calls; the gain is one latency-bound launch
// instead of two (a short prompt's router + shared gate|up). Pairs only what the pair
// kernel carries (the 16-row direct i8 body, the one-tile SLM body); otherwise, or with pair = false, it is two
// pf_dense launches. rq: the quantiser rides the same launch (or its own after the two).
// Returns the last launch's event. a8_0 / a8_1: that matrix on W8A8 (its quantiser launches
// first; an int8 matrix at <= 32 rows then runs the W8A8 direct body, at more rows the SLM body).
sycl::event pf_dense_pair(sycl::queue &q, int m, const DenseWeights &w0, const sycl::ext::oneapi::bfloat16 *a0, int lda0, const PfOut &out0,
                          const DenseWeights &w1, const sycl::ext::oneapi::bfloat16 *a1, int lda1, const PfOut &out1, bool pair = true,
                          const PfRowQuant *rq = nullptr, const PfA8 *a8_0 = nullptr, const PfA8 *a8_1 = nullptr);
// One GEMM (C = A W^T, bf16 activations) and the row quantiser in ONE launch: the GEMM runs pf_dense's SLM body at one
// tile per work-group (pf_dense's bits at every body), the quantiser beside it in the same launch (moe_rows_route: the
// router and the W4A8 x of a row owner's rows).
sycl::event pf_dense_quant(sycl::queue &q, int m, const DenseWeights &w, const sycl::ext::oneapi::bfloat16 *a, int lda, const PfOut &out,
                           const PfRowQuant &rq);
// Whether pf_dense_pair(m, w0, w1) runs as one launch (for tests and the probe).
bool pf_dense_pairable(int m, const DenseWeights &w0, const DenseWeights &w1, bool a8_0 = false, bool a8_1 = false);

// Split-K for `ks` parts by whole 128-groups (the tail rides the last part): raw fp32 partials,
// the int8 scale NOT applied, part p at partials + p * m * ld (row stride ld). The caller sums the
// parts in order and scales (pf_splitk_sum).
sycl::event pf_dense_splitk(sycl::queue &q, int m, const DenseWeights &w, const sycl::ext::oneapi::bfloat16 *a, int lda,
                            float *partials, int ld, int ks, int nt = 0, const PfA8 *a8 = nullptr);  // nt as pf_dense_tiles (0 = auto)
// out[r][c] = (sum over p in order of partials[p][r][c]) * scale[c] (scale null = 1), for r < m, c < n;
// partials as pf_dense_splitk left them (row stride ld, part stride m * ld), out row stride ldo.
sycl::event pf_splitk_sum(sycl::queue &q, int m, int n, int ks, const float *partials, int ld, const float *scale, float *out, int ldo);

}  // namespace xenithium:kernels
