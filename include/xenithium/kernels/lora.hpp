// Xenithium · LoRA adapter deltas: out += scale · B·(A·x) at the existing GEMM outputs.
// A and B are the PEFT adapter matrices (bf16) packed in the dense bf16 layout (pack_dense_bf16):
// A is [r16][k] (DenseWeights n=r16, k) and B is [n][r16] (n, k=r16). r is padded to r16 = ceil(r/16)·16
// (the dense body needs n % 16 == 0). The mid (A·x) is formed in fp32, rounded to bf16 once (as the
// engine's activations), then B·mid runs the same bf16 DPAS body. The delta accumulates into the caller's
// fp32 buffer (the existing part / partial / y) BEFORE the engine's rounding / reduce, so no new wire.
// Column-parallel modules (q/k/v, gate/up) carry A full and B row-split over the card's output slice;
// row-parallel (o, down) carry A col-split over the card's input slice and B full, the partial added to the
// card's part buffer before the pair_reduce.
#pragma once

#include <cstddef>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"

namespace xenithium::kernels {

using bf16 = sycl::ext::oneapi::bfloat16;

// One (card, layer, module) LoRA delta. a / b are the packed matrices (pack_dense_bf16); k is A's input dim
// (this card's slice), n is B's output dim (this card's slice for column-parallel, full for row-parallel),
// out_off the column offset of this module's B rows within the target buffer, scale = lora_alpha / r.
struct LoraModule {
  const uint16_t *a = nullptr;
  const uint16_t *b = nullptr;
  int r16 = 0;
  int k = 0;
  int n = 0;
  int out_off = 0;
  float scale = 1.f;
  bool present() const { return a && b && r16 > 0 && k > 0 && n > 0; }
};

// One layer's LoRA deltas for one card: the GQA (q/k/v/o) and MLP (gate/up/down) modules. A module is
// present when the adapter carries its tensor; an absent module (a layer the adapter does not touch, or a
// module it does not target) keeps nulls and lora_apply skips it.
struct LoraLayer {
  LoraModule q, k, v, o;      // GQA: column-parallel q/k/v, row-parallel o
  LoraModule gate, up, down;  // MLP: column-parallel gate/up, row-parallel down
  bool any() const { return q.present() || k.present() || v.present() || o.present() || gate.present() || up.present() || down.present(); }
};

// Per-card scratch reused across every layer and module: the mid (A·x, fp32 then bf16) and the delta (B·mid).
// Sized to the adapter's r16 and the largest module output (hidden); one per card.
struct LoraScratch {
  float *mid_f32 = nullptr;  // [m][r16]
  bf16 *mid_b16 = nullptr;   // [m][r16]
  float *delta = nullptr;    // [m][n_max]
  int r16 = 0;
  int n_max = 0;
};

// dst[r][c] = bf16(src[r][c]) for r < m, c < n (round to nearest even, as the engine's activations).
sycl::event lora_f32_to_bf16(sycl::queue &q, int m, int n, const float *src, bf16 *dst);
// out[r*ld + c] += scale · delta[r][c] for r < m, c < n (delta row stride n, out row stride ld).
sycl::event lora_accum(sycl::queue &q, int m, int n, int ld, const float *delta, float *out, float scale);
// mid = A·x, mid_b16 = bf16(mid), delta = B·mid_b16, out[out_off .. out_off + n) += scale·delta (out row stride out_ld:
// the target buffer's stride, which differs from n when the module's columns sit inside a wider row, e.g. the q/k/v
// section of the qkv projection or gate | up of the MLP). m <= 64 runs the decode dense_gemm; m > 64 the prefill
// pf_dense. Returns the last launch's event.
sycl::event lora_apply(sycl::queue &q, int m, const bf16 *x, const LoraModule &lm, float *out, int out_ld, LoraScratch &sc, int ksplit);

}  // namespace xenithium::kernels
