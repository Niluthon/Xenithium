#include "xenithium/kernels/residual/hc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"

namespace xenithium::kernels {
namespace {

constexpr int kNormItems = 256;  // one work-group per (row, stream)

inline float sigmoidf(float x) { return 1.f / (1.f + sycl::exp(-x)); }

template <bool COMBINE>
class HcCombineNorm;
class HcSilu;
class HcSumSilu;
class HcCombine;
class HcCombineAt;
class HcGateMix;

template <bool COMBINE>
sycl::event combine_norm(sycl::queue &q, const HcShape &s, int rows, hc_bf16 *res, const hc_bf16 *block, const float *inj, int inj_ld,
                         int inj_col, const hc_bf16 *w, hc_bf16 *xn) {
  const int S = s.streams, H = s.hidden;
  const float eps = s.eps;
  return q.parallel_for<HcCombineNorm<COMBINE>>(
      sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(S) * kNormItems}, {1, kNormItems}), [=](sycl::nd_item<2> it) {
        const int row = static_cast<int>(it.get_group(0)), st = static_cast<int>(it.get_group(1)), lid = static_cast<int>(it.get_local_id(1));  // a (row, stream) grid
        const size_t base = (static_cast<size_t>(row) * S + st) * H;
        float m = 0.f;
        if constexpr (COMBINE) {
          const float l = static_cast<float>(hc_bf16(inj[static_cast<size_t>(row) * inj_ld + inj_col + st]));
          m = 2.f * sigmoidf(l / static_cast<float>(S));
        }
        float ss = 0.f;
        for (int i = lid; i < H; i += kNormItems) {
          float v = static_cast<float>(res[base + i]);
          if constexpr (COMBINE) {
            const hc_bf16 r = hc_bf16(sycl::fma(static_cast<float>(block[static_cast<size_t>(row) * H + i]), m, v));
            res[base + i] = r;
            v = static_cast<float>(r);
          }
          ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float rr = sycl::rsqrt(ss / static_cast<float>(H) + eps);
        for (int i = lid; i < H; i += kNormItems) {
          // COMBINE: re-read what this item stored (same item, same address: ordered).
          const float y = static_cast<float>(res[base + i]) * rr;
          xn[base + i] = hc_bf16(sycl::fma(y, static_cast<float>(w[st * H + i]), y));  // Gemma (1 + w) as one FMA
        }
      });
}

}  // namespace

sycl::event hc_combine_norm(sycl::queue &q, const HcShape &s, int rows, hc_bf16 *res, const hc_bf16 *block, const float *inj, int inj_ld,
                            int inj_col, const hc_bf16 *norm_w, hc_bf16 *xn) {
  if (block && !inj) throw std::invalid_argument("hc_combine_norm: a pending block needs its injection logits");
  return block ? combine_norm<true>(q, s, rows, res, block, inj, inj_ld, inj_col, norm_w, xn)
               : combine_norm<false>(q, s, rows, res, nullptr, nullptr, 0, 0, norm_w, xn);
}

sycl::event hc_combine(sycl::queue &q, const HcShape &s, int rows, HcState &st, const hc_bf16 *block) {
  const int S = s.streams, H = s.hidden, inj_ld = st.lora_ld, inj_col = s.lowrank;
  const hc_bf16 *res = st.res[st.cur];
  hc_bf16 *dst = st.res[st.cur ^ 1];
  const float *inj = st.lora[st.lcur];
  const size_t live = static_cast<size_t>(rows) * S * H, all = std::max(live, static_cast<size_t>(row_shape(rows)) * S * H);
  sycl::event e = q.parallel_for<HcCombine>(sycl::range<1>(all), [=](sycl::id<1> id) {
    const size_t i = id[0];
    if (i >= live) {  // dead rows: carried over unchanged
      dst[i] = res[i];
      return;
    }
    const int c = static_cast<int>(i % H), sti = static_cast<int>(i / H % S), row = static_cast<int>(i / H / S);
    const float l = static_cast<float>(hc_bf16(inj[static_cast<size_t>(row) * inj_ld + inj_col + sti]));
    const float m = 2.f * sigmoidf(l / static_cast<float>(S));
    dst[i] = hc_bf16(sycl::fma(static_cast<float>(block[static_cast<size_t>(row) * H + c]), m, static_cast<float>(res[i])));
  });
  st.cur ^= 1;
  return e;
}

sycl::event hc_combine_at(sycl::queue &q, const HcShape &s, RowRange own, HcState &st, const hc_bf16 *block) {
  if (own.lo < 0 || own.rows() < 1) throw std::invalid_argument("hc_combine_at: row range");
  const int S = s.streams, H = s.hidden, inj_ld = st.lora_ld, inj_col = s.lowrank, row0 = own.lo;
  const size_t at = static_cast<size_t>(row0) * S * H;
  const hc_bf16 *res = st.res[st.cur] + at;
  hc_bf16 *dst = st.res[st.cur ^ 1] + at;
  const float *inj = st.lora[st.lcur];
  // hc_combine's arithmetic per element, on the range's rows only (no dead rows to carry)
  sycl::event e = q.parallel_for<HcCombineAt>(sycl::range<1>(static_cast<size_t>(own.rows()) * S * H), [=](sycl::id<1> id) {
    const size_t i = id[0];
    const int c = static_cast<int>(i % H), sti = static_cast<int>(i / H % S), row = row0 + static_cast<int>(i / H / S);
    const float l = static_cast<float>(hc_bf16(inj[static_cast<size_t>(row) * inj_ld + inj_col + sti]));
    const float m = 2.f * sigmoidf(l / static_cast<float>(S));
    dst[i] = hc_bf16(sycl::fma(static_cast<float>(block[static_cast<size_t>(row) * H + c]), m, static_cast<float>(res[i])));
  });
  st.cur ^= 1;
  return e;
}

HcEvents hc_mixer(sycl::queue &q, const HcShape &s, int rows, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcScratch &sc,
                  hc_bf16 *out, HcPath path, int down_ksplit, int up_ksplit) {
  return hc_mixer_at(q, s, RowRange{0, rows}, st, block, w, sc, out, path, down_ksplit, up_ksplit);
}

HcEvents hc_mixer_at(sycl::queue &q, const HcShape &s, RowRange own, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcScratch &sc,
                     hc_bf16 *out, HcPath path, int down_ksplit, int up_ksplit) {
  if (w.down.k != s.width() || w.down.n < s.lowrank || w.up.n != s.width() || w.up.k != s.lowrank)
    throw std::invalid_argument("hc_mixer: weight shapes do not match the mixer");
  if (block && st.lora_ld < s.lowrank + s.streams) throw std::invalid_argument("hc_mixer: no injection logits pending for the combine");
  if (own.lo < 0 || own.rows() < 1) throw std::invalid_argument("hc_mixer: row range");
  if (path != HcPath::unfused && !hc_fast_supported(s)) path = HcPath::unfused;
  HcEvents e;
  // The range's rows of the state, the block and out; the scratch from its start.
  const int rows = own.rows();
  const size_t r0 = static_cast<size_t>(own.lo);
  hc_bf16 *res = st.res[st.cur] + r0 * s.width();
  const hc_bf16 *blk = block ? block + r0 * s.hidden : nullptr;
  const float *inj = block ? st.lora[st.lcur] + r0 * st.lora_ld : nullptr;
  float *lora = st.lora[st.lcur ^ 1] + r0 * w.down.n;
  hc_bf16 *o = out + r0 * s.hidden;
  e.norm = path == HcPath::three ? hc_combine_norm_fast(q, s, rows, res, blk, inj, st.lora_ld, s.lowrank, w.norm, sc.xn)
                                 : hc_combine_norm(q, s, rows, res, blk, inj, st.lora_ld, s.lowrank, w.norm, sc.xn);
  e.down = dense_gemm(q, rows, sc.xn, w.down, lora, down_ksplit);
  st.lcur ^= 1;
  st.lora_ld = w.down.n;
  if (path != HcPath::unfused) {
    // 33-64 rows (the MTP layer over a 64-row step, which is not split by owner) as two up + mix
    // launches of rows 0-31 and the rest: each row's operations are its own (the same bits)
    if (rows > kStepRows32) {
      const size_t h = kStepRows32;
      hc_up_mix(q, s, kStepRows32, lora, w.down.n, w.up, sc.xn, o, up_ksplit);
      e.up = hc_up_mix(q, s, rows - kStepRows32, lora + h * w.down.n, w.down.n, w.up, sc.xn + h * s.width(), o + h * s.hidden, up_ksplit);
    } else {
      e.up = hc_up_mix(q, s, rows, lora, w.down.n, w.up, sc.xn, o, up_ksplit);
    }
  } else {
    if (!sc.s || !sc.gate) throw std::invalid_argument("hc_mixer: the unfused path needs s and gate scratch");
    hc_silu(q, s, rows, lora, w.down.n, sc.s);
    dense_gemm(q, rows, sc.s, w.up, sc.gate);
    e.up = hc_gate_mix(q, s, rows, sc.gate, sc.xn, o);
  }
  return e;
}

namespace {
class HcGateMix16;
// hc_gate_mix on a bf16 gate: the same operations on the same bf16 values.
sycl::event hc_gate_mix16(sycl::queue &q, const HcShape &s, int rows, const hc_bf16 *gate, const hc_bf16 *xn, hc_bf16 *out) {
  const int H = s.hidden, S = s.streams;
  return q.parallel_for<HcGateMix16>(sycl::range<1>(static_cast<size_t>(rows) * H), [=](sycl::id<1> id) {
    const int row = static_cast<int>(id[0]) / H, c = static_cast<int>(id[0]) % H;
    float acc = 0.f;
    for (int st = 0; st < S; ++st) {
      const size_t at = (static_cast<size_t>(row) * S + st) * H + c;
      acc = sycl::fma(sigmoidf(static_cast<float>(gate[at])), static_cast<float>(xn[at]), acc);
    }
    out[static_cast<size_t>(row) * H + c] = hc_bf16(acc / static_cast<float>(S));
  });
}
}  // namespace

HcEvents hc_mixer_rows_part(sycl::queue &q, const HcShape &s, int r0, int r1, const HcState &st, const hc_bf16 *block, const HcWeights &w,
                            const HcRowsScratch &sc, hc_bf16 *out, const QuadOwnerSum *block_sum) {
  if (w.down.k != s.width() || w.down.n < s.lowrank || w.up.n != s.width() || w.up.k != s.lowrank)
    throw std::invalid_argument("hc_mixer_rows: weight shapes do not match the mixer");
  if (block && st.lora_ld < s.lowrank + s.streams) throw std::invalid_argument("hc_mixer_rows: no injection logits pending for the combine");
  if (!hc_fast_supported(s) || !sc.xn || !sc.s || !sc.gate || !sc.part) throw std::invalid_argument("hc_mixer_rows: shape or scratch");
  if (r0 < 0 || r1 <= r0) throw std::invalid_argument("hc_mixer_rows: row range");
  HcEvents e;
  const int W = s.width(), H = s.hidden, dn = w.down.n, m = r1 - r0;
  // Rows [r0, r1) of the chunk's buffers; the scratch is per call (from its start).
  const float *inj = block ? st.lora[st.lcur] + static_cast<size_t>(r0) * st.lora_ld : nullptr;
  float *lora = st.lora[st.lcur ^ 1] + static_cast<size_t>(r0) * dn;
  hc_bf16 *res = st.res[st.cur] + static_cast<size_t>(r0) * W;
  const hc_bf16 *blk = block ? block + static_cast<size_t>(r0) * H : nullptr;
  if (block_sum) {
    if (!blk || block_sum->out != blk || block_sum->rows != m) throw std::invalid_argument("hc_mixer_rows: the pending sum is not block's rows [r0, r1)");
    e.norm = hc_sum_combine_norm(q, s, *block_sum, res, inj, st.lora_ld, s.lowrank, w.norm, sc.xn);
  } else {
    e.norm = hc_combine_norm_fast(q, s, m, res, blk, inj, st.lora_ld, s.lowrank, w.norm, sc.xn);
  }
  const int groups = (w.down.k + 127) / 128;
  const int ks = std::min(kHcDownKs, groups);  // by shape, never by rows: any row split keeps every bit
  pf_dense_splitk(q, m, w.down, sc.xn, W, sc.part, dn, ks, 0, sc.a8.on());
  e.down = hc_sum_silu(q, s, m, dn, ks, sc.part, w.down.kind == DenseKind::i8 ? w.down.scale : nullptr, lora, sc.s);
  // The up GEMM + gate mix in one launch (same bits; the gate never goes through memory).
  if (!sc.split_up_mix && !sc.a8.on() && m > kHcUpMixRowsMin) {
    e.up = hc_up_mix_rows(q, s, m, w.up, sc.s, sc.xn, out + static_cast<size_t>(r0) * H);
    return e;
  }
  // The gate in bf16 (the mix rounds it to bf16 first: bit-identical), half its bytes both ways.
  auto *g16 = reinterpret_cast<hc_bf16 *>(sc.gate);
  pf_dense(q, m, w.up, sc.s, s.lowrank, PfOut{nullptr, g16, W}, sc.a8.on());
  e.up = hc_gate_mix16(q, s, m, g16, sc.xn, out + static_cast<size_t>(r0) * H);
  return e;
}

void hc_mixer_rows_flip(HcState &st, const HcWeights &w) {
  st.lcur ^= 1;
  st.lora_ld = w.down.n;
}

HcEvents hc_mixer_rows(sycl::queue &q, const HcShape &s, int m, HcState &st, const hc_bf16 *block, const HcWeights &w, const HcRowsScratch &sc,
                       hc_bf16 *out) {
  const HcEvents e = hc_mixer_rows_part(q, s, 0, m, st, block, w, sc, out);
  hc_mixer_rows_flip(st, w);
  return e;
}

// pf_splitk_sum + hc_silu in one launch (the same operations per element: bit-identical).
sycl::event hc_sum_silu(sycl::queue &q, const HcShape &s, int m, int n, int ks, const float *partials, const float *scale, float *lora,
                        hc_bf16 *out) {
  const int R = s.lowrank, S = s.streams;
  const size_t pstride = static_cast<size_t>(m) * n;
  return q.parallel_for<HcSumSilu>(sycl::range<1>(static_cast<size_t>(m) * n), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0] / n), c = static_cast<int>(id[0] % n);
    const size_t at = static_cast<size_t>(r) * n + c;
    float v = partials[at];
    for (int p = 1; p < ks; ++p) v += partials[p * pstride + at];
    if (scale) v *= scale[c];
    lora[at] = v;
    if (c < R) {
      const float x = static_cast<float>(hc_bf16(v)) / static_cast<float>(S);
      out[static_cast<size_t>(r) * R + c] = hc_bf16(x * sigmoidf(x));
    }
  });
}

sycl::event hc_silu(sycl::queue &q, const HcShape &s, int rows, const float *lora, int lora_ld, hc_bf16 *out) {
  const int R = s.lowrank, S = s.streams;
  return q.parallel_for<HcSilu>(sycl::range<1>(static_cast<size_t>(rows) * R), [=](sycl::id<1> id) {
    const int row = static_cast<int>(id[0]) / R, c = static_cast<int>(id[0]) % R;
    const float x = static_cast<float>(hc_bf16(lora[static_cast<size_t>(row) * lora_ld + c])) / static_cast<float>(S);
    out[static_cast<size_t>(row) * R + c] = hc_bf16(x * sigmoidf(x));
  });
}

sycl::event hc_gate_mix(sycl::queue &q, const HcShape &s, int rows, const float *gate, const hc_bf16 *xn, hc_bf16 *out) {
  const int H = s.hidden, S = s.streams;
  return q.parallel_for<HcGateMix>(sycl::range<1>(static_cast<size_t>(rows) * H), [=](sycl::id<1> id) {
    const int row = static_cast<int>(id[0]) / H, c = static_cast<int>(id[0]) % H;
    float acc = 0.f;
    for (int st = 0; st < S; ++st) {
      const size_t at = (static_cast<size_t>(row) * S + st) * H + c;
      acc = sycl::fma(sigmoidf(static_cast<float>(hc_bf16(gate[at]))), static_cast<float>(xn[at]), acc);
    }
    out[static_cast<size_t>(row) * H + c] = hc_bf16(acc / static_cast<float>(S));
  });
}

// ---- host reference ----------------------------------------------------------------------
namespace {
inline float bf(uint16_t b) { return bf16_value(b); }
inline uint16_t rb(float v) { return bf16_bits(v); }
inline float sig(float x) { return 1.f / (1.f + std::exp(-x)); }
void gemm_ref(const RefMatrix &m, const float *x, int rows, float *y) {  // y [rows][m.n], bf16-rounded
  for (int r = 0; r < rows; ++r)
    for (int n = 0; n < m.n; ++n) {
      double acc = 0;
      const float *w = m.w + static_cast<size_t>(n) * m.k;
      for (int k = 0; k < m.k; ++k) acc += static_cast<double>(x[static_cast<size_t>(r) * m.k + k]) * w[k];
      if (m.scale) acc *= m.scale[n];
      y[static_cast<size_t>(r) * m.n + n] = bf(rb(static_cast<float>(acc)));
    }
}
}  // namespace

void hc_mix_ref(const HcShape &s, int rows, uint16_t *res, const uint16_t *block, const float *inj, const uint16_t *norm_w,
                const RefMatrix &down, const RefMatrix &up, uint16_t *xn_out, float *lora_out, uint16_t *out) {
  const int S = s.streams, H = s.hidden, W = s.width(), R = s.lowrank;
  if (down.k != W || down.n < R || up.k != R || up.n != W) throw std::invalid_argument("hc_mix_ref: matrix shapes");
  std::vector<float> xn(static_cast<size_t>(rows) * W), lora(static_cast<size_t>(rows) * down.n), sv(static_cast<size_t>(rows) * R),
      gate(static_cast<size_t>(rows) * W);
  for (int r = 0; r < rows; ++r)
    for (int st = 0; st < S; ++st) {
      uint16_t *x = res + (static_cast<size_t>(r) * S + st) * H;
      if (block) {
        const float m = 2.f * sig(bf(rb(inj[static_cast<size_t>(r) * S + st])) / static_cast<float>(S));
        for (int i = 0; i < H; ++i) x[i] = rb(std::fma(bf(block[static_cast<size_t>(r) * H + i]), m, bf(x[i])));
      }
      double ss = 0;
      for (int i = 0; i < H; ++i) ss += static_cast<double>(bf(x[i])) * bf(x[i]);
      const float rr = 1.f / std::sqrt(static_cast<float>(ss / H) + s.eps);
      for (int i = 0; i < H; ++i) {
        const float y = bf(x[i]) * rr;
        const uint16_t b = rb(std::fma(y, bf(norm_w[st * H + i]), y));
        xn[(static_cast<size_t>(r) * S + st) * H + i] = bf(b);
        if (xn_out) xn_out[(static_cast<size_t>(r) * S + st) * H + i] = b;
      }
    }
  gemm_ref(down, xn.data(), rows, lora.data());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < R; ++c) {
      const float x = lora[static_cast<size_t>(r) * down.n + c] / static_cast<float>(S);
      sv[static_cast<size_t>(r) * R + c] = bf(rb(x * sig(x)));
    }
  if (lora_out) std::memcpy(lora_out, lora.data(), lora.size() * sizeof(float));
  gemm_ref(up, sv.data(), rows, gate.data());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < H; ++c) {
      float acc = 0.f;
      for (int st = 0; st < S; ++st) {
        const size_t at = (static_cast<size_t>(r) * S + st) * H + c;
        acc = std::fma(sig(gate[at]), xn[at], acc);
      }
      out[static_cast<size_t>(r) * H + c] = rb(acc / static_cast<float>(S));
    }
}

}  // namespace xenithium:kernels
