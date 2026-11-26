// Xenithium · LoRA adapter deltas (kernels/lora.hpp). The two GEMMs reuse the engine's bf16 DPAS bodies
// (dense_gemm for a decode step, pf_dense for a prefill chunk); only the mid's bf16 rounding and the
// scaled accumulate into the caller's fp32 buffer are new (the trivial elementwise launches below).
#include "xenithium/kernels/lora.hpp"

#include <stdexcept>

#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {

sycl::event lora_f32_to_bf16(sycl::queue &q, int m, int n, const float *src, bf16 *dst) {
  if (m < 1 || n < 1) throw std::invalid_argument("lora_f32_to_bf16: m, n >= 1");
  const size_t total = static_cast<size_t>(m) * n;
  return q.parallel_for(total, [=](sycl::id<1> i) { dst[i[0]] = static_cast<bf16>(src[i[0]]); });
}

sycl::event lora_accum(sycl::queue &q, int m, int n, int ld, const float *delta, float *out, float scale) {
  if (m < 1 || n < 1) throw std::invalid_argument("lora_accum: m, n >= 1");
  if (ld < n) throw std::invalid_argument("lora_accum: the out stride is shorter than the module");
  const size_t total = static_cast<size_t>(m) * n;
  return q.parallel_for(total, [=](sycl::id<1> i) {
    const size_t idx = i[0];
    const int r = static_cast<int>(idx / n), c = static_cast<int>(idx % n);
    out[static_cast<size_t>(r) * ld + c] += scale * delta[idx];
  });
}

sycl::event lora_apply(sycl::queue &q, int m, const bf16 *x, const LoraModule &lm, float *out, int out_ld, LoraScratch &sc, int ksplit) {
  if (m < 1) throw std::invalid_argument("lora_apply: m >= 1");
  if (!lm.present()) return sycl::event{};
  if (lm.r16 > sc.r16 || lm.n > sc.n_max) throw std::invalid_argument("lora_apply: the scratch is short of the module");
  if (out_ld < lm.n + lm.out_off) throw std::invalid_argument("lora_apply: the out stride is short of the module's columns");
  DenseWeights a{reinterpret_cast<const void *>(lm.a), nullptr, DenseKind::bf16, lm.r16, lm.k};
  DenseWeights b{reinterpret_cast<const void *>(lm.b), nullptr, DenseKind::bf16, lm.n, lm.r16};
  if (!a.valid() || !b.valid()) throw std::invalid_argument("lora_apply: A / B do not pack (n and k multiples of 16)");
  sycl::event e;
  if (m <= kMaxShape) {  // a decode step (rows 1..64): the dense GEMM
    e = dense_gemm(q, m, x, a, sc.mid_f32, ksplit);
    e = lora_f32_to_bf16(q, m, lm.r16, sc.mid_f32, sc.mid_b16);
    e = dense_gemm(q, m, sc.mid_b16, b, sc.delta, 0);
    e = lora_accum(q, m, lm.n, out_ld, sc.delta, out + lm.out_off, lm.scale);
  } else {  // a prefill chunk (many rows): the many-row GEMM
    PfOut mid{sc.mid_f32, nullptr, lm.r16};
    e = pf_dense(q, m, a, x, lm.k, mid);
    e = lora_f32_to_bf16(q, m, lm.r16, sc.mid_f32, sc.mid_b16);
    PfOut dl{sc.delta, nullptr, lm.n};
    e = pf_dense(q, m, b, sc.mid_b16, lm.r16, dl);
    e = lora_accum(q, m, lm.n, out_ld, sc.delta, out + lm.out_off, lm.scale);
  }
  return e;
}

}  // namespace xenithium::kernels
