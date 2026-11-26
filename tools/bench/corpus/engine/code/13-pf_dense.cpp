#include "xenithium/kernels/gemm/pf_dense.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "xenithium/kernels/gemm/onednn_pf.hpp"

#include "xenithium/kernels/tile/dpas_int8.hpp"
#include "xenithium/kernels/tile/row_quant.hpp"
#include "xenithium/kernels/tile/rot_quant.hpp"
#include "xenithium/rt/card.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kWg = 16;  // work-items per work-group (row blocks sharing one column block)
constexpr int kCall = kDenseCallElems;  // 256 bf16 elements of one DPAS call's B operand (16 k x 16 columns)
constexpr int kW = 16;  // columns per tile
constexpr int kGroup = 128;

enum Epi { kStore = 0, kSplit = 1 };

struct Args {
  const void *w = nullptr;
  const float *scale = nullptr;
  int n = 0, full = 0, tail = 0, tiles = 0;
  const bf16 *a = nullptr;
  int lda = 0, m = 0;
  float *f32 = nullptr;
  bf16 *b16 = nullptr;
  int ld = 0;
  int ks = 1;
  size_t part_stride = 0;
  // W8A8: the quantised activations [m][k] int8 (row stride k) and their group-major [gpr][lds] scales
  // (dense.hpp PfA8); null = bf16 activations.
  const int8_t *aq = nullptr;
  const float *as = nullptr;
  int lds = 0;
  bool row = false;  // as = one scale per row (PfA8:row)
};

// SLM bytes of the SLM kernel's work-group with NT tiles (double-buffered widened groups).
template <int NT>
constexpr int slm_bytes() {
  return 2 * NT * kDenseCalls * kCall * 2;
}

// The SLM kernel's work-group `gid` (of rgs * cbs * ks), work-item `lid`; SLM set up by the kernel
// (slm_bytes<NT>). A function so that pf_dense_pair runs the same body.
template <int MB, int NT, DenseKind KIND, int EPI>
inline void slm_body(const Args &a, int gid, int lid) SYCL_ESIMD_FUNCTION {
  constexpr int kEB = KIND == DenseKind::i8 ? 1 : 2;  // bytes per packed element
  constexpr int kBB = kDenseCalls * kCall * kEB;  // bytes per (group, tile) block
  constexpr int kCB = kCall * kEB;  // bytes per DPAS call in a block
  constexpr int kAcc = NT * MB * kW;
  const int rgs = (a.m + kWg * MB - 1) / (kWg * MB);
  const int part = gid % a.ks, rest = gid / a.ks, rb = rest % rgs, cb = rest / rgs;
  const int m0 = (rb * kWg + lid) * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const int g_begin = part * groups / a.ks, g_end = (part + 1) * groups / a.ks;
  auto tile_of = [&](int t) SYCL_ESIMD_FUNCTION -> int {
    const int tt = cb * NT + t;
    return tt < a.tiles ? tt : -1;
  };
  esimd::simd<float, kAcc> acc(0.f);
  const uint16_t *a16 = reinterpret_cast<const uint16_t *>(a.a);
  const unsigned sw = static_cast<unsigned>(a.lda * 2 - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  for (int g = g_begin; g < g_end; ++g) {
    const int buf = (g - g_begin) & 1;
    const bool is_tail = g >= a.full;
    const int calls = is_tail ? a.tail : kDenseCalls;
    // Widen this group's blocks into SLM: unit u = (tile t, call j), 16 items.
    for (int u = lid; u < NT * kDenseCalls; u += kWg) {
      const int t = u / kDenseCalls, j = u % kDenseCalls, tt = tile_of(t);
      if (tt < 0 || j >= calls) continue;
      const uint8_t *src = is_tail ? wb + static_cast<size_t>(a.full) * a.tiles * kBB +
                                         static_cast<size_t>(tt) * a.tail * kCB + static_cast<size_t>(j) * kCB
                                   : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB + static_cast<size_t>(j) * kCB;
      const uint32_t off = static_cast<uint32_t>(((buf * NT + t) * kDenseCalls + j) * kCall * 2);
      if constexpr (KIND == DenseKind::i8) {
        const esimd::simd<int8_t, kCall> c = esimd::block_load<int8_t, kCall>(reinterpret_cast<const int8_t *>(src));
        const esimd::simd<float, kCall> f = c;  // exact
        const esimd::simd<bf16, kCall> b = f;  // exact (|q| <= 127)
        esimd::slm_block_store<bf16, kCall>(off, b);
      } else {
        esimd::slm_block_store<bf16, kCall>(off, esimd::block_load<bf16, kCall>(reinterpret_cast<const bf16 *>(src)));
      }
    }
    esimd::barrier();
    if (m0 < a.m) {
      for (int j = 0; j < calls; ++j) {
        const int k0 = g * kGroup + j * 16;
        esimd::simd<uint16_t, MB * 16> au = esimd::load_2d<uint16_t, 16, MB>(a16, sw, sh, sw, k0, m0);
        esimd::simd<bf16, MB * 16> av = au.template bit_cast_view<bf16>();
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const esimd::simd<bf16, kCall> b = esimd::slm_block_load<bf16, kCall>(
              static_cast<uint32_t>(((buf * NT + t) * kDenseCalls + j) * kCall * 2));
#pragma unroll
          for (int r = 0; r < MB / 8; ++r) {
            auto cv = acc.template select<128, 1>((t * MB + r * 8) * kW);
            esimd::simd<float, 128> c = cv;
            esimd::simd<bf16, 128> ar = av.template select<128, 1>(r * 128);
            cv = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b, ar);
          }
        }
      }
    }
  }
  if (m0 >= a.m) return;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tt = tile_of(t);
    if (tt < 0) continue;
    esimd::simd<float, kW> s(1.f);
    if constexpr (KIND == DenseKind::i8 && EPI == kStore) s = esimd::block_load<float, kW>(a.scale + tt * kW);
#pragma unroll
    for (int r = 0; r < MB; ++r) {
      const int row = m0 + r;
      if (row >= a.m) break;
      esimd::simd<float, kW> v = acc.template select<kW, 1>((t * MB + r) * kW);
      const size_t at = static_cast<size_t>(row) * a.ld + static_cast<size_t>(tt) * kW;
      if constexpr (EPI == kSplit) {
        v.copy_to(a.f32 + part * a.part_stride + at);
      } else {
        if constexpr (KIND == DenseKind::i8) v *= s;
        if (a.f32) v.copy_to(a.f32 + at);
        if (a.b16) {
          const esimd::simd<bf16, kW> vb = v;
          vb.copy_to(a.b16 + at);
        }
      }
    }
  }
}

template <int MB, int NT, DenseKind KIND, int EPI>
class PfKern;

template <int MB, int NT, DenseKind KIND, int EPI>
size_t slm_wgs(const Args &A) {
  return static_cast<size_t>((A.tiles + NT - 1) / NT) * ((A.m + kWg * MB - 1) / (kWg * MB)) * A.ks;
}

template <int MB, int NT, DenseKind KIND, int EPI>
sycl::event launch(sycl::queue &q, const Args &A) {
  static_assert(MB % 8 == 0 && MB <= 32 && NT >= 1 && NT <= 8, "pf_dense tile shape");
  const size_t wgs = slm_wgs<MB, NT, KIND, EPI>(A);
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfKern<MB, NT, KIND, EPI>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<slm_bytes<NT>()>();
      slm_body<MB, NT, KIND, EPI>(a, static_cast<int>(it.get_group(0)), static_cast<int>(it.get_local_id(0)));
    });
  });
}

// ---- W8A8, int8 activations on s8 x s8 DPAS ----------------------------------------------------------------
// The weights are the same int8 codes in their bf16 VNNI blocks; the activations are int8 per (row, 128-k
// group; the k % 128 tail is its own group) with one fp32 scale each (row_quant.hpp). An s8 DPAS call
// takes 32 k (dpas_int8.hpp: dword k4 * 16 + lane holds k = 4 k4.. 4 k4 + 3 of column `lane`), so two
// stored 16-k calls (pair P = kin / 2 of 16 lanes at u16 P * 16 + lane) are reordered in registers into one
// operand: u16 (k4 * 16 + lane) * 2 + h = pair 2 k4 + h. Per group the int32 sum of a tile is exact
// (128 x 127 x 127 < 2^24, so float(sum) is too); acc += float(sum) x a_scale with the groups in order, the
// column scale on the finished sum as the bf16-activation bodies. Every body (SLM with NT tiles, direct,
// pair) runs the same DPAS calls per tile in the same order: bit-identical across them, and a row's result
// does not depend on the call around it. The SLM kernel widens NT x 4 operands of 512 B per group (half the
// bf16 path's SLM bytes), shared by its 16 row blocks.
constexpr int kOp8 = 2 * kCall;  // 512 bytes: one s8 operand, 32 k x 16 columns
constexpr int kOps8 = kDenseCalls / 2;  // 4 operands per full 128-k group

inline esimd::simd<uint16_t, kOp8 / 2> to_s8_operand(esimd::simd<uint16_t, kOp8 / 2> raw) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint16_t, kOp8 / 2> out;
#pragma unroll
  for (int k4 = 0; k4 < 8; ++k4)
#pragma unroll
    for (int h = 0; h < 2; ++h) out.template select<16, 2>(k4 * 32 + h) = raw.template select<16, 1>((2 * k4 + h) * 16);
  return out;
}

// SLM body, j-outer / tile-inner: per 32-k operand the row block's A is loaded once and multiplied into NT
// tiles' int32 accumulators; the group's fold into the fp32 sums reads the row block's 16 scales as one block
// load from the group-major table. NT <= 2: the int32 + fp32 accumulators of 4 tiles x 16 rows fill the GRF.
template <int MB, int NT, int EPI>
inline void slm_body_a8(const Args &a, int gid, int lid) SYCL_ESIMD_FUNCTION {
  static_assert(NT <= 2, "the W8A8 SLM body holds at most 2 tiles' accumulators");
  constexpr int kBB = kDenseCalls * kCall;  // int8 bytes per (group, tile) block
  constexpr int kCB = kCall;  // bytes per stored 16-k call
  constexpr int kAcc = NT * MB * kW;
  const int rgs = (a.m + kWg * MB - 1) / (kWg * MB);
  const int part = gid % a.ks, rest = gid / a.ks, rb = rest % rgs, cb = rest / rgs;
  const int m0 = (rb * kWg + lid) * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const int g_begin = part * groups / a.ks, g_end = (part + 1) * groups / a.ks;
  auto tile_of = [&](int t) SYCL_ESIMD_FUNCTION -> int {
    const int tt = cb * NT + t;
    return tt < a.tiles ? tt : -1;
  };
  esimd::simd<float, kAcc> acc(0.f);
  esimd::simd<int32_t, kAcc> ai(0);
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(a.aq);
  const int k = a.full * kGroup + a.tail * 16;  // the quantised rows are contiguous: stride k bytes
  const unsigned sw = static_cast<unsigned>(k - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  for (int g = g_begin; g < g_end; ++g) {
    const int buf = (g - g_begin) & 1;
    const bool is_tail = g >= a.full;
    const int ops = is_tail ? a.tail / 2 : kOps8;
    // Reorder this group's blocks into SLM as s8 operands: unit u = (tile t, operand j), 16 items.
    for (int u = lid; u < NT * kOps8; u += kWg) {
      const int t = u / kOps8, j = u % kOps8, tt = tile_of(t);
      if (tt < 0 || j >= ops) continue;
      const uint8_t *src = is_tail ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB +
                                         static_cast<size_t>(j) * kOp8
                                   : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB + static_cast<size_t>(j) * kOp8;
      const esimd::simd<uint16_t, kOp8 / 2> raw = esimd::block_load<uint16_t, kOp8 / 2>(reinterpret_cast<const uint16_t *>(src));
      esimd::slm_block_store<uint16_t, kOp8 / 2>(static_cast<uint32_t>(((buf * NT + t) * kOps8 + j) * kOp8), to_s8_operand(raw));
    }
    esimd::barrier();
    if (m0 < a.m) {
      for (int j = 0; j < ops; ++j) {
        const int k0 = g * kGroup + j * 32;
        esimd::simd<uint32_t, MB * 8> au = esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, k0 / 4, m0);
        esimd::simd<int8_t, MB * 32> av = au.template bit_cast_view<int8_t>();
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          if (tile_of(t) < 0) continue;
          const esimd::simd<int8_t, kOp8> b =
              esimd::slm_block_load<int8_t, kOp8>(static_cast<uint32_t>(((buf * NT + t) * kOps8 + j) * kOp8));
#pragma unroll
          for (int rr = 0; rr < MB / 8; ++rr) {
            auto cv = ai.template select<128, 1>((t * MB + rr * 8) * kW);
            const esimd::simd<int32_t, 128> c = cv;
            const esimd::simd<int8_t, 256> ar = av.template select<256, 1>(rr * 256);
            cv = dpas::accumulate<8>(c, b, ar);
          }
        }
      }
      const esimd::simd<float, MB> sg = esimd::block_load<float, MB>(a.as + static_cast<size_t>(g) * a.lds + m0);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        if (tile_of(t) < 0) continue;
#pragma unroll
        for (int r = 0; r < MB; ++r) {
          const float sv = sg[r];
          const esimd::simd<int32_t, kW> fi = ai.template select<kW, 1>((t * MB + r) * kW);
          const esimd::simd<float, kW> f = fi;  // exact
          auto cv = acc.template select<kW, 1>((t * MB + r) * kW);
          cv = esimd::simd<float, kW>(cv) + f * sv;
        }
      }
      ai = 0;
    }
  }
  if (m0 >= a.m) return;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tt = tile_of(t);
    if (tt < 0) continue;
    esimd::simd<float, kW> s(1.f);
    if constexpr (EPI == kStore) s = esimd::block_load<float, kW>(a.scale + tt * kW);
#pragma unroll
    for (int r = 0; r < MB; ++r) {
      const int row = m0 + r;
      if (row >= a.m) break;
      esimd::simd<float, kW> v = acc.template select<kW, 1>((t * MB + r) * kW);
      const size_t at = static_cast<size_t>(row) * a.ld + static_cast<size_t>(tt) * kW;
      if constexpr (EPI == kSplit) {
        v.copy_to(a.f32 + part * a.part_stride + at);
      } else {
        v *= s;
        if (a.f32) v.copy_to(a.f32 + at);
        if (a.b16) {
          const esimd::simd<bf16, kW> vb = v;
          vb.copy_to(a.b16 + at);
        }
      }
    }
  }
}

template <int MB, int NT, int EPI>
class PfKernA8;

template <int MB, int NT, int EPI>
sycl::event launch_a8(sycl::queue &q, const Args &A) {
  static_assert(MB % 8 == 0 && MB <= 32 && NT >= 1 && NT <= 8, "pf_dense tile shape");
  const size_t wgs = slm_wgs<MB, NT, DenseKind::i8, EPI>(A);
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfKernA8<MB, NT, EPI>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<2 * NT * kOps8 * kOp8>();
      slm_body_a8<MB, NT, EPI>(a, static_cast<int>(it.get_group(0)), static_cast<int>(it.get_local_id(0)));
    });
  });
}

// The direct W8A8 body: one work-item per (16-row block, tile, k part), the group's block reordered into
// registers (2 KB) with the next group prefetched; no SLM. A 17-32-row call takes two row blocks (the bf16
// direct body's 32-row shape would spill beside the int32 sums). Items past the range return.
constexpr int kMbA8 = 16;
inline int direct_rbs_a8(const Args &a) { return (a.m + kMbA8 - 1) / kMbA8; }

template <int EPI>
inline void direct_body_a8(const Args &a, int gid) SYCL_ESIMD_FUNCTION {
  constexpr int MB = kMbA8;
  constexpr int kBB = kDenseCalls * kCall;
  constexpr int kCB = kCall;
  const int rbs = direct_rbs_a8(a);
  if (gid >= a.tiles * a.ks * rbs) return;
  const int part = gid % a.ks, rest = gid / a.ks, tt = rest % a.tiles, rb = rest / a.tiles;
  const int m0 = rb * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const int g_begin = part * groups / a.ks, g_end = (part + 1) * groups / a.ks;
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(a.aq);
  const int k = a.full * kGroup + a.tail * 16;
  const unsigned sw = static_cast<unsigned>(k - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  esimd::simd<float, MB * kW> acc(0.f);
  auto block = [&](int g) SYCL_ESIMD_FUNCTION -> const uint8_t * {
    return g >= a.full ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB
                       : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB;
  };
  for (int g = g_begin; g < g_end; ++g) {
    const int ops = g >= a.full ? a.tail / 2 : kOps8;
    const uint8_t *src = block(g);
    if (g + 1 < g_end) {
      const uint8_t *nx = block(g + 1);
      const int nops = g + 1 >= a.full ? a.tail / 2 : kOps8;
#pragma unroll
      for (int j = 0; j < 2 * kOps8; ++j)  // 256-byte pieces (the dense prefetch's shape)
        if (j < 2 * nops)
          esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nx + static_cast<size_t>(j) * 256),
                                        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
    }
    esimd::simd<uint16_t, kOps8 * kOp8 / 2> r;
#pragma unroll
    for (int j = 0; j < kOps8; ++j)
      if (j < ops)
        r.template select<kOp8 / 2, 1>(j * (kOp8 / 2)) =
            to_s8_operand(esimd::block_load<uint16_t, kOp8 / 2>(reinterpret_cast<const uint16_t *>(src + static_cast<size_t>(j) * kOp8)));
    const esimd::simd<float, MB> sg = esimd::block_load<float, MB>(a.as + static_cast<size_t>(g) * a.lds + m0);
    esimd::simd<int32_t, MB * kW> ai(0);
    for (int j = 0; j < ops; ++j) {
      const int k0 = g * kGroup + j * 32;
      esimd::simd<uint32_t, MB * 8> au = esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, k0 / 4, m0);
      esimd::simd<int8_t, MB * 32> av = au.template bit_cast_view<int8_t>();
      esimd::simd<uint16_t, kOp8 / 2> bu = r.template select<kOp8 / 2, 1>(j * (kOp8 / 2));
      const esimd::simd<int8_t, kOp8> b = bu.template bit_cast_view<int8_t>();
#pragma unroll
      for (int rr = 0; rr < MB / 8; ++rr) {
        auto cv = ai.template select<128, 1>(rr * 128);
        const esimd::simd<int32_t, 128> c = cv;
        const esimd::simd<int8_t, 256> ar = av.template select<256, 1>(rr * 256);
        cv = dpas::accumulate<8>(c, b, ar);
      }
    }
#pragma unroll
    for (int rr = 0; rr < MB; ++rr) {
      const float s = sg[rr];
      const esimd::simd<int32_t, kW> fi = ai.template select<kW, 1>(rr * kW);
      const esimd::simd<float, kW> f = fi;
      auto cv = acc.template select<kW, 1>(rr * kW);
      cv = esimd::simd<float, kW>(cv) + f * s;
    }
  }
  esimd::simd<float, kW> s(1.f);
  if constexpr (EPI == kStore) s = esimd::block_load<float, kW>(a.scale + tt * kW);
#pragma unroll
  for (int rr = 0; rr < MB; ++rr) {
    const int row = m0 + rr;
    if (row >= a.m) break;
    esimd::simd<float, kW> v = acc.template select<kW, 1>(rr * kW);
    const size_t at = static_cast<size_t>(row) * a.ld + static_cast<size_t>(tt) * kW;
    if constexpr (EPI == kSplit) {
      v.copy_to(a.f32 + part * a.part_stride + at);
    } else {
      v *= s;
      if (a.f32) v.copy_to(a.f32 + at);
      if (a.b16) {
        const esimd::simd<bf16, kW> vb = v;
        vb.copy_to(a.b16 + at);
      }
    }
  }
}

// ---- W8A8 per row (PfA8:row) ------------------------------------------------------------------------------------------
// One int32 accumulator per (row, column) over the whole k and no per-group fold: the registers the fold's fp32 sums held
// take 4 tiles x 16 rows at 128 GRF (the per-group body holds 2), and the group loop has no FMA tail. The s8 operand is
// made while staging by the dword form of to_s8_operand (u32 (k4 x 16 + lane) = u16 (2 k4 x 16 + lane) | u16 ((2 k4 + 1)
// x 16 + lane) << 16: the same bytes, no strided writes). KB 128-k groups are staged per barrier. Integer sums are exact,
// so every body and shape gives the same bits: out = float(sum) x a_scale[row] x w_scale[col] (row block's scales by one
// block load). No split-K (the store epilogue only).
inline esimd::simd<uint16_t, kOp8 / 2> to_s8_operand_dw(esimd::simd<uint16_t, kOp8 / 2> raw) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, kOp8 / 4> out;
#pragma unroll
  for (int k4 = 0; k4 < 8; ++k4) {
    const esimd::simd<uint32_t, 16> lo = raw.template select<16, 1>(2 * k4 * 16);
    const esimd::simd<uint32_t, 16> hi = raw.template select<16, 1>((2 * k4 + 1) * 16);
    out.template select<16, 1>(k4 * 16) = lo | (hi << 16);
  }
  return out.template bit_cast_view<uint16_t>();
}

template <int MB, int NT>
inline void store_a8r(const Args &a, int m0, int tt, int t, esimd::simd<int32_t, NT * MB * kW> &ai, esimd::simd<float, MB> sr) SYCL_ESIMD_FUNCTION {
  const esimd::simd<float, kW> s = esimd::block_load<float, kW>(a.scale + tt * kW);
#pragma unroll
  for (int r = 0; r < MB; ++r) {
    const int row = m0 + r;
    if (row < a.m) {
      const float srv = sr[r];
      const esimd::simd<int32_t, kW> fi = ai.template select<kW, 1>((t * MB + r) * kW);
      esimd::simd<float, kW> v = esimd::simd<float, kW>(fi) * srv;
      v *= s;
      const size_t at = static_cast<size_t>(row) * a.ld + static_cast<size_t>(tt) * kW;
      if (a.f32) v.copy_to(a.f32 + at);
      if (a.b16) {
        const esimd::simd<bf16, kW> vb = v;
        vb.copy_to(a.b16 + at);
      }
    }
  }
}

template <int MB, int NT, int KB>
inline void slm_body_a8r(const Args &a, int gid, int lid) SYCL_ESIMD_FUNCTION {
  constexpr int kBB = kDenseCalls * kCall, kCB = kCall, kAcc = NT * MB * kW;
  const int rgs = (a.m + kWg * MB - 1) / (kWg * MB);
  const int rb = gid % rgs, cb = gid / rgs;
  const int m0 = (rb * kWg + lid) * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  auto tile_of = [&](int t) SYCL_ESIMD_FUNCTION -> int {
    const int tt = cb * NT + t;
    return tt < a.tiles ? tt : -1;
  };
  esimd::simd<int32_t, kAcc> ai(0);
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(a.aq);
  const int k = a.full * kGroup + a.tail * 16;
  const unsigned sw = static_cast<unsigned>(k - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  for (int g0 = 0; g0 < groups; g0 += KB) {
    const int buf = (g0 / KB) & 1;
    for (int u = lid; u < KB * NT * kOps8; u += kWg) {
      const int gi = u / (NT * kOps8), t = (u / kOps8) % NT, j = u % kOps8, tt = tile_of(t), g = g0 + gi;
      const bool is_tail = g >= a.full;
      const int ops = g >= groups ? 0 : is_tail ? a.tail / 2 : kOps8;
      if (tt < 0 || j >= ops) continue;
      const uint8_t *src = is_tail ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB + static_cast<size_t>(j) * kOp8
                                   : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB + static_cast<size_t>(j) * kOp8;
      const esimd::simd<uint16_t, kOp8 / 2> raw = esimd::block_load<uint16_t, kOp8 / 2>(reinterpret_cast<const uint16_t *>(src));
      esimd::slm_block_store<uint16_t, kOp8 / 2>(static_cast<uint32_t>((((buf * KB + gi) * NT + t) * kOps8 + j) * kOp8), to_s8_operand_dw(raw));
    }
    esimd::barrier();
    if (m0 < a.m) {
#pragma unroll
      for (int gi = 0; gi < KB; ++gi) {
        const int g = g0 + gi;
        const int ops = g >= groups ? 0 : g >= a.full ? a.tail / 2 : kOps8;
        for (int j = 0; j < ops; ++j) {
          const int k0 = g * kGroup + j * 32;
          esimd::simd<uint32_t, MB * 8> au = esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, k0 / 4, m0);
          esimd::simd<int8_t, MB * 32> av = au.template bit_cast_view<int8_t>();
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            if (tile_of(t) < 0) continue;
            const esimd::simd<int8_t, kOp8> b =
                esimd::slm_block_load<int8_t, kOp8>(static_cast<uint32_t>((((buf * KB + gi) * NT + t) * kOps8 + j) * kOp8));
#pragma unroll
            for (int rr = 0; rr < MB / 8; ++rr) {
              auto cv = ai.template select<128, 1>((t * MB + rr * 8) * kW);
              const esimd::simd<int32_t, 128> c = cv;
              const esimd::simd<int8_t, 256> ar = av.template select<256, 1>(rr * 256);
              cv = dpas::accumulate<8>(c, b, ar);
            }
          }
        }
      }
    }
  }
  if (m0 >= a.m) return;
  const esimd::simd<float, MB> sr = esimd::block_load<float, MB>(a.as + m0);
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tt = tile_of(t);
    if (tt >= 0) store_a8r<MB, NT>(a, m0, tt, t, ai, sr);
  }
}

// resident 16-item work-groups per Xe core at 128 GRF (a B70's 32 cores: 128 per wave); the cores are rt:xe_cores (the
// device's, the emulated card's under XEN_EMU_CARD), read once (the cards of a fabric are the same model)
constexpr long kA8RowWgsPerCore = 4;
long a8r_wave_wgs(sycl::queue &q) {
  static const long wgs = kA8RowWgsPerCore * rt::xe_cores(q.get_device());
  return wgs;
}

template <int MB, int NT, int KB>
class PfKernA8R;

template <int MB, int NT, int KB>
sycl::event launch_a8r(sycl::queue &q, const Args &A) {
  const size_t wgs = static_cast<size_t>((A.tiles + NT - 1) / NT) * ((A.m + kWg * MB - 1) / (kWg * MB));
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfKernA8R<MB, NT, KB>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<2 * KB * NT * kOps8 * kOp8>();
      slm_body_a8r<MB, NT, KB>(a, static_cast<int>(it.get_group(0)), static_cast<int>(it.get_local_id(0)));
    });
  });
}

// The direct per-row body (m <= 32 as two 16-row blocks): direct_body_a8's item (16 rows x one tile, the group's block in
// registers, the next group prefetched) with one int32 accumulator over the whole k.
inline void direct_body_a8r(const Args &a, int gid) SYCL_ESIMD_FUNCTION {
  constexpr int MB = kMbA8;
  constexpr int kBB = kDenseCalls * kCall;
  constexpr int kCB = kCall;
  const int rbs = direct_rbs_a8(a);
  if (gid >= a.tiles * rbs) return;
  const int tt = gid % a.tiles, rb = gid / a.tiles;
  const int m0 = rb * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(a.aq);
  const int k = a.full * kGroup + a.tail * 16;
  const unsigned sw = static_cast<unsigned>(k - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  auto block = [&](int g) SYCL_ESIMD_FUNCTION -> const uint8_t * {
    return g >= a.full ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB
                       : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB;
  };
  esimd::simd<int32_t, MB * kW> ai(0);
  for (int g = 0; g < groups; ++g) {
    const int ops = g >= a.full ? a.tail / 2 : kOps8;
    const uint8_t *src = block(g);
    if (g + 1 < groups) {
      const uint8_t *nx = block(g + 1);
      const int nops = g + 1 >= a.full ? a.tail / 2 : kOps8;
#pragma unroll
      for (int j = 0; j < 2 * kOps8; ++j)
        if (j < 2 * nops)
          esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nx + static_cast<size_t>(j) * 256),
                                        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
    }
    esimd::simd<uint16_t, kOps8 * kOp8 / 2> r;
#pragma unroll
    for (int j = 0; j < kOps8; ++j)
      if (j < ops)
        r.template select<kOp8 / 2, 1>(j * (kOp8 / 2)) =
            to_s8_operand_dw(esimd::block_load<uint16_t, kOp8 / 2>(reinterpret_cast<const uint16_t *>(src + static_cast<size_t>(j) * kOp8)));
    for (int j = 0; j < ops; ++j) {
      const int k0 = g * kGroup + j * 32;
      esimd::simd<uint32_t, MB * 8> au = esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, k0 / 4, m0);
      esimd::simd<int8_t, MB * 32> av = au.template bit_cast_view<int8_t>();
      esimd::simd<uint16_t, kOp8 / 2> bu = r.template select<kOp8 / 2, 1>(j * (kOp8 / 2));
      const esimd::simd<int8_t, kOp8> b = bu.template bit_cast_view<int8_t>();
#pragma unroll
      for (int rr = 0; rr < MB / 8; ++rr) {
        auto cv = ai.template select<128, 1>(rr * 128);
        const esimd::simd<int32_t, 128> c = cv;
        const esimd::simd<int8_t, 256> ar = av.template select<256, 1>(rr * 256);
        cv = dpas::accumulate<8>(c, b, ar);
      }
    }
  }
  const esimd::simd<float, MB> sr = esimd::block_load<float, MB>(a.as + m0);
  store_a8r<MB, 1>(a, m0, tt, 0, ai, sr);
}

class PfDirectA8R;
sycl::event launch_direct_a8r(sycl::queue &q, const Args &A) {
  const size_t global = (static_cast<size_t>(A.tiles) * direct_rbs_a8(A) + 15) / 16 * 16;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfDirectA8R>(sycl::nd_range<1>(global, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      direct_body_a8r(a, static_cast<int>(it.get_global_id(0)));
    });
  });
}

// The direct per-row body with a k split for narrow GEMMs at small m (GDN out, QSA o: 160 tiles, 24 groups). Item (tile,
// 16-row block, part) sums its groups' int32 products; pf_a8r_ks_sum adds the parts (int32: exact in any order) and
// applies store_a8r's epilogue (float(sum) x row scale, x column scale, the same two IEEE multiplies): the same bits as
// the unsplit bodies.
inline void direct_body_a8r_ks(const Args &a, int gid, int32_t *part) SYCL_ESIMD_FUNCTION {
  constexpr int MB = kMbA8;
  constexpr int kBB = kDenseCalls * kCall;
  constexpr int kCB = kCall;
  const int rbs = direct_rbs_a8(a);
  if (gid >= a.tiles * rbs * a.ks) return;
  const int p = gid % a.ks, rest = gid / a.ks, tt = rest % a.tiles, rb = rest / a.tiles;
  const int m0 = rb * MB;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const int g_begin = p * groups / a.ks, g_end = (p + 1) * groups / a.ks;
  const uint32_t *a32 = reinterpret_cast<const uint32_t *>(a.aq);
  const int k = a.full * kGroup + a.tail * 16;
  const unsigned sw = static_cast<unsigned>(k - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  auto block = [&](int g) SYCL_ESIMD_FUNCTION -> const uint8_t * {
    return g >= a.full ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB
                       : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB;
  };
  esimd::simd<int32_t, MB * kW> ai(0);
  for (int g = g_begin; g < g_end; ++g) {
    const int ops = g >= a.full ? a.tail / 2 : kOps8;
    const uint8_t *src = block(g);
    if (g + 1 < g_end) {
      const uint8_t *nx = block(g + 1);
      const int nops = g + 1 >= a.full ? a.tail / 2 : kOps8;
#pragma unroll
      for (int j = 0; j < 2 * kOps8; ++j)
        if (j < 2 * nops)
          esimd::prefetch<uint32_t, 64>(reinterpret_cast<const uint32_t *>(nx + static_cast<size_t>(j) * 256),
                                        esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>});
    }
    esimd::simd<uint16_t, kOps8 * kOp8 / 2> r;
#pragma unroll
    for (int j = 0; j < kOps8; ++j)
      if (j < ops)
        r.template select<kOp8 / 2, 1>(j * (kOp8 / 2)) =
            to_s8_operand_dw(esimd::block_load<uint16_t, kOp8 / 2>(reinterpret_cast<const uint16_t *>(src + static_cast<size_t>(j) * kOp8)));
    for (int j = 0; j < ops; ++j) {
      const int k0 = g * kGroup + j * 32;
      esimd::simd<uint32_t, MB * 8> au = esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, k0 / 4, m0);
      esimd::simd<int8_t, MB * 32> av = au.template bit_cast_view<int8_t>();
      esimd::simd<uint16_t, kOp8 / 2> bu = r.template select<kOp8 / 2, 1>(j * (kOp8 / 2));
      const esimd::simd<int8_t, kOp8> b = bu.template bit_cast_view<int8_t>();
#pragma unroll
      for (int rr = 0; rr < MB / 8; ++rr) {
        auto cv = ai.template select<128, 1>(rr * 128);
        const esimd::simd<int32_t, 128> c = cv;
        const esimd::simd<int8_t, 256> ar = av.template select<256, 1>(rr * 256);
        cv = dpas::accumulate<8>(c, b, ar);
      }
    }
  }
  // part p's sums: [p][row][n] (rows of the block past m are not stored)
#pragma unroll
  for (int r = 0; r < MB; ++r)
    if (m0 + r < a.m)
      esimd::block_store<int32_t, kW>(part + (static_cast<size_t>(p) * a.m + m0 + r) * a.n + static_cast<size_t>(tt) * kW,
                                      ai.template select<kW, 1>(r * kW));
}

class PfDirectA8RKs;
class PfA8RKsSum;
// The parts' scratch per SYCL context (grown on demand; prefill calls are host-ordered per card).
std::mutex g_ks_mu;
int32_t *ks_scratch(sycl::queue &q, size_t ints) {
  std::lock_guard<std::mutex> lk(g_ks_mu);
  static std::map<size_t, std::pair<int32_t *, size_t>> bufs;  // by device
  auto &b = bufs[std::hash<sycl::device>{}(q.get_device())];
  if (b.second < ints) {
    if (b.first) sycl::free(b.first, q);
    b.first = sycl::malloc_device<int32_t>(ints, q);
    if (!b.first) throw std::runtime_error("pf_dense: the k-split parts scratch");
    b.second = ints;
  }
  return b.first;
}
constexpr int kA8rKs = 4;  // k parts of the split direct per-row body
sycl::event launch_direct_a8r_ks(sycl::queue &q, const Args &A0, int ks) {
  Args A = A0;
  A.ks = ks;
  int32_t *part = ks_scratch(q, static_cast<size_t>(ks) * A.m * A.n);
  const size_t global = (static_cast<size_t>(A.tiles) * direct_rbs_a8(A) * ks + 15) / 16 * 16;
  q.submit([&](sycl::handler &cgh) {
    const Args a = A;
    cgh.parallel_for<PfDirectA8RKs>(sycl::nd_range<1>(global, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      direct_body_a8r_ks(a, static_cast<int>(it.get_global_id(0)), part);
    });
  });
  const int m = A.m, n = A.n, ld = A.ld;
  const float *rs = A.as, *cs = A.scale;
  float *f32 = A.f32;
  bf16 *b16 = A.b16;
  return q.parallel_for<PfA8RKsSum>(sycl::range<1>(static_cast<size_t>(m) * n), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0] / n), c = static_cast<int>(id[0] % n);
    int32_t t = 0;
    for (int p = 0; p < ks; ++p) t += part[(static_cast<size_t>(p) * m + r) * n + c];
    float v = static_cast<float>(t) * rs[r];
    v *= cs[c];
    const size_t at = static_cast<size_t>(r) * ld + c;
    if (f32) f32[at] = v;
    if (b16) b16[at] = bf16(v);
  });
}

class PfA8Quant;

// The activation quantiser (row_quant_rows: 128-k groups with a short tail, 4 rows x 1 group per item) over m
// rows of a (stride lda) into the caller's scratch (scales group-major, stride pf_a8_ld(m)); k = the weights' k.
constexpr int kQuantRows = 4;
sycl::event launch_a8_quant(sycl::queue &q, const bf16 *a, int lda, int m, int k, int8_t *dst, float *scl) {
  const int gpr = (k + kGroup - 1) / kGroup, lds = pf_a8_ld(m);
  const size_t n = static_cast<size_t>((m + kQuantRows - 1) / kQuantRows) * gpr;
  return q.parallel_for<PfA8Quant>(sycl::nd_range<1>((n + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    row_quant_rows<kGroup, kQuantRows>(a, lda, m, k, dst, scl, lds, it.get_global_id(0));
  });
}

// The per-row quantiser (PfA8:row): one work-group of 16 items per row; item i holds the row's 32-wide chunks
// i, i + 16... (CPI of them at most) in registers, the 16 partial maxima meet in SLM, then every item quantises its
// chunks with the row's scale: s = amax / 127, q = rne(v x (127 / amax)) clamped to +-127 (row_quant.hpp's rule over the
// whole row; an all-zero row gives q = 0, s = 0). k % 32 == 0.
template <int CPI>
class PfA8QuantRow;
template <int CPI>
sycl::event quant_row(sycl::queue &q, const bf16 *a, int lda, int m, int k, int8_t *dst, float *scl) {
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfA8QuantRow<CPI>>(sycl::nd_range<1>(static_cast<size_t>(m) * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<64>();
      const int r = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
      const int nch = k / 32;
      esimd::simd<float, 32 * CPI> v(0.f);
#pragma unroll
      for (int c = 0; c < CPI; ++c) {
        const int ch = i + c * 16;
        if (ch < nch) v.template select<32, 1>(c * 32) = esimd::block_load<bf16, 32>(a + static_cast<size_t>(r) * lda + ch * 32);
      }
      esimd::slm_scalar_store<float>(static_cast<uint32_t>(i * 4), esimd::hmax<float>(esimd::abs(v)));
      esimd::barrier();
      const float amax = esimd::hmax<float>(esimd::slm_block_load<float, 16>(0));
      esimd::simd<int8_t, 32 * CPI> qv(0);
      if (amax > 0.f) {
        const float inv = esimd::div_ieee(127.f, amax);
        const esimd::simd<float, 32 * CPI> t = esimd::rnde<float, 32 * CPI>(v * inv);
        qv = esimd::simd<int8_t, 32 * CPI>(esimd::max(esimd::min(t, 127.f), -127.f));
      }
#pragma unroll
      for (int c = 0; c < CPI; ++c) {
        const int ch = i + c * 16;
        if (ch < nch) esimd::block_store<int8_t, 32>(dst + static_cast<size_t>(r) * k + ch * 32, qv.template select<32, 1>(c * 32));
      }
      if (i == 0) scl[r] = amax > 0.f ? esimd::div_ieee(amax, 127.f) : 0.f;
    });
  });
}
sycl::event launch_a8_quant_row(sycl::queue &q, const bf16 *a, int lda, int m, int k, int8_t *dst, float *scl) {
  if (k % 32 || k > 16 * 32 * 20) throw std::invalid_argument("pf_dense: W8A8 per row needs k % 32 == 0, k <= 10,240");
  if (k <= 16 * 32 * 2) return quant_row<2>(q, a, lda, m, k, dst, scl);
  if (k <= 16 * 32 * 6) return quant_row<6>(q, a, lda, m, k, dst, scl);
  return quant_row<20>(q, a, lda, m, k, dst, scl);
}

// Direct variant for a call of one row block (m <= MB): no SLM, no barrier. One
// work-item per (tile, k part) streams its tile's weights straight into registers, the next group's
// raw block loading while the current one computes. The SLM path at such m has one computing item
// per work-group and one memory latency plus a barrier per 128-k group. Same widening, the same
// DPAS calls in the same order and the same epilogue: bit-identical to the SLM path. Compiled at 128
// GRF: the i8 variants do not spill; the bf16 ones spill 8 KB but never run in the model (no bf16
// weight has the 256 items the direct path needs).
// `gid`: the work-item's (tile, part) index; items past tiles * ks return (no barrier here).
template <int MB, DenseKind KIND, int EPI>
inline void direct_body(const Args &a, int gid) SYCL_ESIMD_FUNCTION {
  using Raw = std::conditional_t<KIND == DenseKind::i8, int8_t, bf16>;
  constexpr int kEB = KIND == DenseKind::i8 ? 1 : 2;
  constexpr int kBB = kDenseCalls * kCall * kEB;
  constexpr int kCB = kCall * kEB;
  if (gid >= a.tiles * a.ks) return;
  const int part = gid % a.ks, tt = gid / a.ks;
  const int groups = a.full + (a.tail > 0 ? 1 : 0);
  const int g_begin = part * groups / a.ks, g_end = (part + 1) * groups / a.ks;
  const uint16_t *a16 = reinterpret_cast<const uint16_t *>(a.a);
  const unsigned sw = static_cast<unsigned>(a.lda * 2 - 1), sh = static_cast<unsigned>(a.m - 1);
  const uint8_t *wb = static_cast<const uint8_t *>(a.w);
  esimd::simd<float, MB * kW> acc(0.f);
  auto load = [&](int g, esimd::simd<Raw, kDenseCalls * kCall> &r) SYCL_ESIMD_FUNCTION {
    const bool is_tail = g >= a.full;
    const int calls = is_tail ? a.tail : kDenseCalls;
    const uint8_t *src = is_tail ? wb + static_cast<size_t>(a.full) * a.tiles * kBB + static_cast<size_t>(tt) * a.tail * kCB
                                 : wb + (static_cast<size_t>(g) * a.tiles + tt) * kBB;
#pragma unroll
    for (int j = 0; j < kDenseCalls; ++j)
      if (j < calls) r.template select<kCall, 1>(j * kCall) = esimd::block_load<Raw, kCall>(reinterpret_cast<const Raw *>(src + static_cast<size_t>(j) * kCB));
  };
  auto compute = [&](int g, esimd::simd<Raw, kDenseCalls * kCall> &r) SYCL_ESIMD_FUNCTION {
    const int calls = g >= a.full ? a.tail : kDenseCalls;
    for (int j = 0; j < calls; ++j) {
      const int k0 = g * kGroup + j * 16;
      esimd::simd<uint16_t, MB * 16> au = esimd::load_2d<uint16_t, 16, MB>(a16, sw, sh, sw, k0, 0);
      esimd::simd<bf16, MB * 16> av = au.template bit_cast_view<bf16>();
      esimd::simd<bf16, kCall> b;
      if constexpr (KIND == DenseKind::i8) {
        const esimd::simd<float, kCall> f = r.template select<kCall, 1>(j * kCall);  // exact
        b = f;  // exact (|q| <= 127)
      } else {
        b = r.template select<kCall, 1>(j * kCall);
      }
#pragma unroll
      for (int rr = 0; rr < MB / 8; ++rr) {
        auto cv = acc.template select<128, 1>(rr * 8 * kW);
        esimd::simd<float, 128> c = cv;
        esimd::simd<bf16, 128> ar = av.template select<128, 1>(rr * 128);
        cv = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b, ar);
      }
    }
  };
  esimd::simd<Raw, kDenseCalls * kCall> r0, r1;
  if (g_begin < g_end) load(g_begin, r0);
  for (int g = g_begin; g < g_end; g += 2) {
    if (g + 1 < g_end) load(g + 1, r1);
    compute(g, r0);
    if (g + 1 >= g_end) break;
    if (g + 2 < g_end) load(g + 2, r0);
    compute(g + 1, r1);
  }
  esimd::simd<float, kW> s(1.f);
  if constexpr (KIND == DenseKind::i8 && EPI == kStore) s = esimd::block_load<float, kW>(a.scale + tt * kW);
#pragma unroll
  for (int r = 0; r < MB; ++r) {
    if (r >= a.m) break;
    esimd::simd<float, kW> v = acc.template select<kW, 1>(r * kW);
    const size_t at = static_cast<size_t>(r) * a.ld + static_cast<size_t>(tt) * kW;
    if constexpr (EPI == kSplit) {
      v.copy_to(a.f32 + part * a.part_stride + at);
    } else {
      if constexpr (KIND == DenseKind::i8) v *= s;
      if (a.f32) v.copy_to(a.f32 + at);
      if (a.b16) {
        const esimd::simd<bf16, kW> vb = v;
        vb.copy_to(a.b16 + at);
      }
    }
  }
}

constexpr int kLs = 16;  // direct kernel: work-items per work-group
size_t direct_wgs(const Args &A) { return (static_cast<size_t>(A.tiles) * A.ks + kLs - 1) / kLs; }
size_t direct_wgs_a8(const Args &A) { return (static_cast<size_t>(A.tiles) * A.ks * direct_rbs_a8(A) + kLs - 1) / kLs; }

template <int EPI>
class PfDirectA8;

template <int EPI>
sycl::event launch_direct_a8(sycl::queue &q, const Args &A) {
  const size_t global = direct_wgs_a8(A) * kLs;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfDirectA8<EPI>>(sycl::nd_range<1>(global, kLs), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      direct_body_a8<EPI>(a, static_cast<int>(it.get_global_id(0)));
    });
  });
}

template <int MB, DenseKind KIND, int EPI>
class PfDirect;

template <int MB, DenseKind KIND, int EPI>
sycl::event launch_direct(sycl::queue &q, const Args &A) {
  static_assert(MB % 8 == 0 && MB <= 32, "pf_dense direct tile shape");
  const size_t global = direct_wgs(A) * kLs;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfDirect<MB, KIND, EPI>>(sycl::nd_range<1>(global, kLs), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      direct_body<MB, KIND, EPI>(a, static_cast<int>(it.get_global_id(0)));
    });
  });
}

template <int MB, int EPI>
sycl::event dispatch_direct(sycl::queue &q, const Args &A, DenseKind kind) {
  if (kind == DenseKind::i8) return launch_direct<MB, DenseKind::i8, EPI>(q, A);
  return launch_direct<MB, DenseKind::bf16, EPI>(q, A);
}

Args base(int m, const DenseWeights &w, const bf16 *a, int lda, const PfA8 *a8 = nullptr) {
  if (!w.valid()) throw std::invalid_argument("pf_dense: invalid weights");
  if (m < 1 || lda < w.k || lda % 8 != 0) throw std::invalid_argument("pf_dense: bad m / lda");
  Args A;
  A.w = w.w;
  A.scale = w.scale;
  A.n = w.n;
  A.full = w.k / kGroup;
  A.tail = (w.k % kGroup) / 16;
  A.tiles = w.n / kW;
  A.a = a;
  A.lda = lda;
  A.m = m;
  if (a8 && w.kind == DenseKind::i8) {  // W8A8 (bf16 weights keep the bf16 path)
    if (w.k % 32) throw std::invalid_argument("pf_dense: W8A8 needs k % 32 == 0");
    if (!a8->q || !a8->s) throw std::invalid_argument("pf_dense: W8A8 scratch");
    A.aq = a8->q;
    A.as = a8->s;
    A.lds = pf_a8_ld(m);
    A.row = a8->row;
  }
  return A;
}

// The W8A8 quantiser of A's activations into its scratch (before the GEMM that reads them).
sycl::event quant_a8(sycl::queue &q, const Args &A) {
  if (A.row) return launch_a8_quant_row(q, A.a, A.lda, A.m, A.full * kGroup + A.tail * 16, const_cast<int8_t *>(A.aq), const_cast<float *>(A.as));
  return launch_a8_quant(q, A.a, A.lda, A.m, A.full * kGroup + A.tail * 16, const_cast<int8_t *>(A.aq), const_cast<float *>(A.as));
}

template <int MB, int NT, int EPI>
sycl::event dispatch_nt(sycl::queue &q, const Args &A, DenseKind kind) {
  if (kind == DenseKind::i8) return launch<MB, NT, DenseKind::i8, EPI>(q, A);
  return launch<MB, NT, DenseKind::bf16, EPI>(q, A);
}

// Tiles per work-group: 4 (the column block of every chunk-sized call) unless that
// leaves the cards short of work-groups, as a short prompt's call does (one row group: a router of
// 32 tiles is 8 work-groups); then 2 or 1. Every tile's k runs in the same order whatever the
// work-group holds, so the result is bit-identical for any choice. m <= kPfDirectRows
// takes the direct kernel. nt > 0 forces the SLM path with nt tiles, nt < 0 the direct one (tests).
constexpr int kPfMinWgs = 64;
constexpr int kPfDirectRows = 32;
constexpr int kPfDirectItems = 256;  // the direct kernel has one item per (tile, part): few tiles starve it
// The kernel pf_dense picks for A: -16 / -32 = direct with MB 16 / 32, 1 / 2 / 4 = SLM with nt tiles.
// a8: the direct W8A8 body is always 16-row blocks (-16, up to 32 rows).
int choose(const Args &A, int nt, bool a8 = false) {
  if (nt < 0 || (nt == 0 && A.m <= kPfDirectRows && static_cast<long>(A.tiles) * A.ks >= kPfDirectItems)) {
    if (A.m > kPfDirectRows) throw std::invalid_argument("pf_dense: the direct kernel takes m <= 32");
    return a8 || A.m <= 16 ? -16 : -32;
  }
  if (nt == 0) {
    const int rgs = (A.m + kWg * 16 - 1) / (kWg * 16);
    auto wgs = [&](int t) { return static_cast<long>((A.tiles + t - 1) / t) * rgs * A.ks; };
    nt = wgs(4) >= kPfMinWgs ? 4 : wgs(2) >= kPfMinWgs ? 2 : 1;
  }
  if (nt != 1 && nt != 2 && nt != 4) throw std::invalid_argument("pf_dense: tiles per work-group 1, 2 or 4");
  return nt;
}

template <int MB, int EPI>
sycl::event dispatch(sycl::queue &q, const Args &A, DenseKind kind, int nt = 0) {
  static_assert(MB == 16, "the SLM path runs 16-row blocks");
  const bool a8 = A.aq != nullptr;
  const int c = choose(A, nt, a8);
  if (a8 && A.row) {  // per-row scales (every body the same bits)
    if constexpr (EPI != kStore) {
      throw std::invalid_argument("pf_dense: W8A8 per row has no split-K");
    } else {
      if (c == -16) return launch_direct_a8r(q, A);
      // m <= 32 on fewer than kPfDirectItems tiles (GDN out, QSA o, shared down): the direct body split over k in kA8rKs
      // parts (the same bits)
      if (nt == 0 && A.m <= kPfDirectRows && A.n == A.tiles * kW) {
        const int groups = A.full + (A.tail > 0 ? 1 : 0), items = A.tiles * direct_rbs_a8(A);
        const int ks = std::min(kA8rKs, groups);
        if (ks >= 2 && items < kPfDirectItems) return launch_direct_a8r_ks(q, A, ks);
      }
      if (nt == 0) {
        // 4 tiles x 1 group per barrier or 2 tiles x 2, by whole waves of work-groups (128 resident per B70 at 128 GRF:
        // a8r_wave_wgs): a 4-tile work-group counts as 1.6 2-tile ones, so 4 tiles where 8 x waves(4) < 5 x waves(2).
        const int rgs = (A.m + kWg * MB - 1) / (kWg * MB);
        const long wave = a8r_wave_wgs(q);
        auto waves = [&](int t) { return (static_cast<long>((A.tiles + t - 1) / t) * rgs + wave - 1) / wave; };
        return 8 * waves(4) < 5 * waves(2) ? launch_a8r<MB, 4, 1>(q, A) : launch_a8r<MB, 2, 2>(q, A);
      }
      if (c == 4) return launch_a8r<MB, 4, 1>(q, A);
      if (c == 2) return launch_a8r<MB, 2, 2>(q, A);
      return launch_a8r<MB, 1, 1>(q, A);
    }
  }
  if (a8) {  // the W8A8 SLM body holds 2 tiles at most: 4 runs as 2 (bit-identical either way)
    if (c == -16) return launch_direct_a8<EPI>(q, A);
    if (c >= 2) return launch_a8<MB, 2, EPI>(q, A);
    return launch_a8<MB, 1, EPI>(q, A);
  }
  if (c == -16) return dispatch_direct<16, EPI>(q, A, kind);
  if (c == -32) return dispatch_direct<32, EPI>(q, A, kind);
  if (c == 4) return dispatch_nt<MB, 4, EPI>(q, A, kind);
  if (c == 2) return dispatch_nt<MB, 2, EPI>(q, A, kind);
  return dispatch_nt<MB, 1, EPI>(q, A, kind);
}

// ---- two independent GEMMs in one launch -------------------------------------------------------------
// Each matrix keeps the kernel pf_dense would pick for it (direct 16 / 32 rows, or SLM with one tile
// per work-group); work-groups [0, w0) run matrix 0's body, the rest matrix 1's. Every output element
// runs the same body code as its own launch: bit-identical. The W8A8 bodies (direct 16-row
// blocks, SLM one tile) run int8 matrices on int8 activations; a pair's int8 matrices share the choice.
enum PairBody { kBodyD16 = 0, kBodyS1 = 2, kBodyD16A8 = 4, kBodyS1A8 = 5 };

template <DenseKind K, int B>
size_t pair_wgs(const Args &A) {
  if constexpr (B == kBodyS1 || B == kBodyS1A8) return slm_wgs<16, 1, K, kStore>(A);
  if constexpr (B == kBodyD16A8) return direct_wgs_a8(A);
  return direct_wgs(A);
}

template <DenseKind K, int B>
inline void pair_run(const Args &a, int wg, int lid) SYCL_ESIMD_FUNCTION {
  if constexpr (B == kBodyS1)
    slm_body<16, 1, K, kStore>(a, wg, lid);
  else if constexpr (B == kBodyS1A8)
    slm_body_a8<16, 1, kStore>(a, wg, lid);
  else if constexpr (B == kBodyD16A8)
    direct_body_a8<kStore>(a, wg * kLs + lid);
  else
    direct_body<16, K, kStore>(a, wg * kLs + lid);
}

template <DenseKind K0, int B0, DenseKind K1, int B1, bool Q>
class PfPair;

// Q = a third job, the W4A8 row quantiser (row_quant.hpp), in work-groups [w0 + w1...).
struct PairQuant {
  const bf16 *src = nullptr;
  int cols = 0;
  int8_t *dst = nullptr;
  float *scl = nullptr;
  size_t n = 0;  // items: rows x cols / 128
  bool rot = false;  // bf16(H_128 src) first (rot_quant.hpp)
};
constexpr int kQuantGroup = 128;
inline void pair_quant_item(const PairQuant &pq, size_t id) SYCL_ESIMD_FUNCTION {
  if (pq.rot) {
    if (id < pq.n) rot_quant_item<kQuantGroup, true>(pq.src, pq.cols, pq.dst, pq.scl, id, static_cast<size_t>(pq.cols / kQuantGroup));
  } else {
    row_quant_item<kQuantGroup>(pq.src, pq.cols, pq.dst, pq.scl, id, pq.n);
  }
}

template <DenseKind K0, int B0, DenseKind K1, int B1, bool Q>
sycl::event launch_pair(sycl::queue &q, const Args &A0, const Args &A1, const PairQuant &Pq) {
  static_assert(kLs == kWg, "the pair kernel's work-group serves both bodies");
  const size_t w0 = pair_wgs<K0, B0>(A0), w1 = pair_wgs<K1, B1>(A1), wq = Q ? (Pq.n + kWg - 1) / kWg : 0;
  const Args a0 = A0, a1 = A1;
  const PairQuant pq = Pq;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<PfPair<K0, B0, K1, B1, Q>>(sycl::nd_range<1>((w0 + w1 + wq) * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      if constexpr (B0 == kBodyS1 || B1 == kBodyS1 || B0 == kBodyS1A8 || B1 == kBodyS1A8) esimd::slm_init<slm_bytes<1>()>();
      const int wg = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      if (wg < static_cast<int>(w0)) {  // uniform per work-group: the SLM body's barriers stay legal
        pair_run<K0, B0>(a0, wg, lid);
      } else if (!Q || wg < static_cast<int>(w0 + w1)) {
        pair_run<K1, B1>(a1, wg - static_cast<int>(w0), lid);
      } else if constexpr (Q) {
        pair_quant_item(pq, static_cast<size_t>(wg - static_cast<int>(w0 + w1)) * kWg + lid);
      }
    });
  });
}

// A8: the int8 matrices of the pair run the W8A8 bodies (both or neither: one switch per pair).
template <DenseKind K0, int B0, DenseKind K1, bool Q, bool A8>
sycl::event pair_b1(sycl::queue &q, const Args &A0, const Args &A1, int b1, const PairQuant &pq) {
  if constexpr (A8) {
    if (b1 == kBodyD16A8) return launch_pair<K0, B0, K1, kBodyD16A8, Q>(q, A0, A1, pq);
    return launch_pair<K0, B0, K1, kBodyS1A8, Q>(q, A0, A1, pq);
  } else {
    if (b1 == kBodyD16) return launch_pair<K0, B0, K1, kBodyD16, Q>(q, A0, A1, pq);
    return launch_pair<K0, B0, K1, kBodyS1, Q>(q, A0, A1, pq);
  }
}

template <DenseKind K0, int B0, bool Q, bool A8>
sycl::event pair_k1(sycl::queue &q, const Args &A0, const Args &A1, DenseKind k1, int b1, const PairQuant &pq) {
  if (k1 == DenseKind::i8) return pair_b1<K0, B0, DenseKind::i8, Q, A8>(q, A0, A1, b1, pq);
  return launch_pair<K0, B0, DenseKind::bf16, kBodyS1, Q>(q, A0, A1, pq);  // bf16: the SLM body only (pair_body)
}

template <DenseKind K0, bool Q, bool A8>
sycl::event pair_b0(sycl::queue &q, const Args &A0, const Args &A1, int b0, DenseKind k1, int b1, const PairQuant &pq) {
  if constexpr (A8) {
    if (b0 == kBodyD16A8) return pair_k1<K0, kBodyD16A8, Q, A8>(q, A0, A1, k1, b1, pq);
    return pair_k1<K0, kBodyS1A8, Q, A8>(q, A0, A1, k1, b1, pq);
  } else {
    if (b0 == kBodyD16) return pair_k1<K0, kBodyD16, Q, A8>(q, A0, A1, k1, b1, pq);
    return pair_k1<K0, kBodyS1, Q, A8>(q, A0, A1, k1, b1, pq);
  }
}

template <bool Q, bool A8>
sycl::event pair_any(sycl::queue &q, const Args &A0, DenseKind k0, int b0, const Args &A1, DenseKind k1, int b1, const PairQuant &pq) {
  if (k0 == DenseKind::i8) return pair_b0<DenseKind::i8, Q, A8>(q, A0, A1, b0, k1, b1, pq);
  return pair_k1<DenseKind::bf16, kBodyS1, Q, A8>(q, A0, A1, k1, b1, pq);  // bf16: the SLM body only (pair_body)
}

class PfQuant;

// The quantiser alone (pf_dense_pair with rq when the GEMMs do not pair): moe_rows' launch shape.
sycl::event launch_quant(sycl::queue &q, const PairQuant &Pq) {
  const PairQuant pq = Pq;
  return q.parallel_for<PfQuant>(sycl::nd_range<1>((pq.n + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    pair_quant_item(pq, it.get_global_id(0));
  });
}

// The pair body for pf_dense's choice c, or -1 when the pair kernel does not carry it: SLM with
// 2 / 4 tiles per work-group (chunk-sized calls, which fill the cards alone), a bf16 direct body (it
// spills; no model weight takes it), or the 32-row direct body (beside another body it spills 0.9-4.5
// KB at 128 GRF).
int pair_body(int c, DenseKind kind, bool a8 = false, bool row = false) {
  if (kind == DenseKind::i8 && a8 && row) return -1;  // the per-row bodies run as their own launches
  if (kind == DenseKind::i8 && a8) {
    if (c == 1) return kBodyS1A8;
    if (c == -16) return kBodyD16A8;
    return -1;
  }
  if (c == 1) return kBodyS1;
  if (kind == DenseKind::i8 && c == -16) return kBodyD16;
  return -1;
}

// ---- int8 matrices on oneDNN ----------------------------------------------------------------------------
// A W8A16 call of an int8 matrix with >= g_dn_min_m rows runs on oneDNN's matmul (onednn_pf.hpp onednn_i8_pf: the [n][k]
// s8 copy repacked per call into an LRU scratch, the stored weights untouched): bf16 activations, fp32 sums, the column
// scale on the finished sum, and the split-K raw partials (equal k slices as one batched matmul). oneDNN sums a slice's k
// in order on most shapes, which gives the engine's bits; on some (n, k, m) it does not (k 10,240 unsplit from 512
// rows; qwen38's PLE kv, n 12,800, at 512+ rows), which changes them and makes a row's bits depend on the call. So the
// first call of each (n, k, m, lda, ld, output, ks) runs both and compares every output word (the ESIMD result is the
// call's result): equal -> oneDNN from then on, else the engine's body for that key (also when the check's transient
// output copy finds no VRAM). A row's bits therefore never depend on which path ran it. Bf16 matrices and W8A8 keep the
// engine's bodies (oneDNN's W8A8 fold rounds apart from the engine's fast-math products on some outputs).
int g_dn_min_m = [] {
  const char *v = std::getenv("XEN_PF_DENSE_ONEDNN");  // the probe's switch; servers call pf_dense_set_onednn
  return v ? std::atoi(v) : 0;
}();
size_t g_dn_cap = 0;  // largest matrix (codes bytes) the route takes; 0 = any
std::mutex g_dn_mu;
std::map<std::array<long long, 8>, int> g_dn_state;  // 1 = oneDNN gives the engine's bits, -1 = it does not
int g_dn_log = [] { const char *v = std::getenv("XEN_PF_DENSE_ONEDNN_LOG"); return v ? std::atoi(v) : 0; }();

bool dn_eligible(const Args &A, DenseKind kind, int epi) {
  if (g_dn_min_m <= 0 || kind != DenseKind::i8 || A.m < g_dn_min_m || A.aq) return false;
  if (g_dn_cap && static_cast<size_t>(A.n) * (A.full * kGroup + A.tail * 16) > g_dn_cap) return false;  // past the budgeted slots
  if ((A.f32 != nullptr) == (A.b16 != nullptr)) return false;  // one output
  if (epi == kSplit) return A.f32 && A.tail == 0 && A.full % A.ks == 0;
  return true;
}

// Runs the call on oneDNN when it gives the engine's bits (see above); false = the caller dispatches the engine's body.
template <int EPI>
bool dn_route(sycl::queue &q, const Args &A, DenseKind kind, sycl::event &ev) {
  if (!dn_eligible(A, kind, EPI)) return false;
  const int k = A.full * kGroup + A.tail * 16, ks = EPI == kSplit ? A.ks : 1;
  const float *wscale = EPI == kSplit ? nullptr : A.scale;
  const auto *w = static_cast<const int8_t *>(A.w);
  const std::array<long long, 8> key{A.n, k, A.m, A.lda, A.ld, A.f32 ? 4 : 2, ks, EPI};
  int st = 0;
  {
    std::lock_guard<std::mutex> lk(g_dn_mu);
    const auto it = g_dn_state.find(key);
    if (it != g_dn_state.end()) st = it->second;
  }
  if (st < 0) return false;
  if (st > 0) {
    ev = onednn_i8_pf(q, A.m, A.n, k, w, wscale, A.a, A.lda, A.f32, A.b16, A.ld, ks);
    return true;
  }
  // first call of this key: the engine's body into the output, oneDNN into a scratch, every word compared
  ev = dispatch<16, EPI>(q, A, kind);
  const size_t elem = A.f32 ? 4 : 2, part = static_cast<size_t>(A.m) * A.ld * elem, bytes = part * ks;
  void *scr = sycl::malloc_device(bytes, q);  // transient (outside the model's VRAM budget): no room = the engine's body
  if (!scr) {
    std::lock_guard<std::mutex> lk(g_dn_mu);
    g_dn_state[key] = -1;
    return true;
  }
  onednn_i8_pf(q, A.m, A.n, k, w, wscale, A.a, A.lda, A.f32 ? static_cast<float *>(scr) : nullptr,
               A.f32 ? nullptr : static_cast<bf16 *>(scr), A.ld, ks);
  const void *out = A.f32 ? static_cast<const void *>(A.f32) : static_cast<const void *>(A.b16);
  const size_t diff = onednn_words_differ(q, out, scr, ks, A.m, static_cast<size_t>(A.n) * elem, static_cast<size_t>(A.ld) * elem, part);
  sycl::free(scr, q);
  if (diff) onednn_dense_forget(q, w);  // its [n][k] copy is never read again
  if (g_dn_log)
    std::fprintf(stderr, "pf_dense onednn: n %d k %d m %d lda %d ld %d %s ks %d: %s (%zu words differ)\n", A.n, k, A.m, A.lda, A.ld, A.f32 ? "f32" : "bf16", ks,
                 diff ? "engine body" : "oneDNN", diff);
  std::lock_guard<std::mutex> lk(g_dn_mu);
  g_dn_state[key] = diff ? -1 : 1;
  return true;
}

template <int EPI>
sycl::event run_one(sycl::queue &q, const Args &A, DenseKind kind, int nt = 0) {
  sycl::event ev;
  if (nt == 0 && dn_route<EPI>(q, A, kind, ev)) return ev;
  return dispatch<16, EPI>(q, A, kind, nt);
}

}  // namespace

sycl::event pf_dense(sycl::queue &q, int m, const DenseWeights &w, const bf16 *a, int lda, const PfOut &out, const PfA8 *a8) {
  Args A = base(m, w, a, lda, a8);
  A.f32 = out.f32;
  A.b16 = out.b16;
  A.ld = out.ld > 0 ? out.ld : w.n;
  if (A.ld < w.n) throw std::invalid_argument("pf_dense: ld < n");
  if (A.aq) quant_a8(q, A);
  return run_one<kStore>(q, A, w.kind);
}

void pf_dense_set_onednn(int min_m) {
  std::lock_guard<std::mutex> lk(g_dn_mu);
  g_dn_min_m = min_m;
}
int pf_dense_onednn() { return g_dn_min_m; }
void pf_dense_set_onednn_cap(size_t max_codes_bytes) {
  std::lock_guard<std::mutex> lk(g_dn_mu);
  g_dn_cap = max_codes_bytes;
}

sycl::event pf_dense_tiles(sycl::queue &q, int m, const DenseWeights &w, const bf16 *a, int lda, const PfOut &out, int nt, const PfA8 *a8) {
  Args A = base(m, w, a, lda, a8);
  A.f32 = out.f32;
  A.b16 = out.b16;
  A.ld = out.ld > 0 ? out.ld : w.n;
  if (A.ld < w.n) throw std::invalid_argument("pf_dense: ld < n");
  if (A.aq) quant_a8(q, A);
  return dispatch<16, kStore>(q, A, w.kind, nt);
}

bool pf_dense_pairable(int m, const DenseWeights &w0, const DenseWeights &w1, bool a8_0, bool a8_1) {
  if (!w0.valid() || !w1.valid() || m < 1) return false;
  const bool u0 = a8_0 && w0.kind == DenseKind::i8, u1 = a8_1 && w1.kind == DenseKind::i8;
  if (w0.kind == DenseKind::i8 && w1.kind == DenseKind::i8 && u0 != u1) return false;
  const auto args = [m](const DenseWeights &w) {
    Args A;
    A.m = m;
    A.tiles = w.n / kW;
    return A;
  };
  return pair_body(choose(args(w0), 0, u0), w0.kind, u0) >= 0 && pair_body(choose(args(w1), 0, u1), w1.kind, u1) >= 0;
}

sycl::event pf_dense_pair(sycl::queue &q, int m, const DenseWeights &w0, const bf16 *a0, int lda0, const PfOut &out0, const DenseWeights &w1,
                          const bf16 *a1, int lda1, const PfOut &out1, bool pair, const PfRowQuant *rq, const PfA8 *a8_0, const PfA8 *a8_1) {
  Args A[2] = {base(m, w0, a0, lda0, a8_0), base(m, w1, a1, lda1, a8_1)};
  const PfOut *o[2] = {&out0, &out1};
  const DenseWeights *w[2] = {&w0, &w1};
  for (int i = 0; i < 2; ++i) {
    A[i].f32 = o[i]->f32;
    A[i].b16 = o[i]->b16;
    A[i].ld = o[i]->ld > 0 ? o[i]->ld : w[i]->n;
    if (A[i].ld < w[i]->n) throw std::invalid_argument("pf_dense_pair: ld < n");
    // one input in one scratch (the GDN in_proj's qkv | z): quantised once
    const bool same = i == 1 && A[1].aq == A[0].aq && a1 == a0 && lda1 == lda0 && A[1].full == A[0].full && A[1].tail == A[0].tail && A[1].row == A[0].row;
    if (A[i].aq && !same) quant_a8(q, A[i]);
  }
  PairQuant pq;
  if (rq) {
    if (!rq->src || !rq->dst || !rq->scl || rq->cols <= 0 || rq->cols % kQuantGroup) throw std::invalid_argument("pf_dense_pair: row quantiser");
    pq = PairQuant{rq->src, rq->cols, rq->dst, rq->scl, static_cast<size_t>(m) * (rq->cols / kQuantGroup), rq->rot};
  }
  const bool u0 = A[0].aq != nullptr, u1 = A[1].aq != nullptr;
  const bool mixed = w0.kind == DenseKind::i8 && w1.kind == DenseKind::i8 && u0 != u1;
  const int b0 = pair_body(choose(A[0], 0, u0), w0.kind, u0, A[0].row), b1 = pair_body(choose(A[1], 0, u1), w1.kind, u1, A[1].row);
  const bool dn = dn_eligible(A[0], w0.kind, kStore) || dn_eligible(A[1], w1.kind, kStore);  // each its own call
  if (!pair || mixed || b0 < 0 || b1 < 0 || dn) {
    run_one<kStore>(q, A[0], w0.kind);
    const sycl::event e = run_one<kStore>(q, A[1], w1.kind);
    return rq ? launch_quant(q, pq) : e;
  }
  const bool a8 = u0 || u1;
  if (rq) return a8 ? pair_any<true, true>(q, A[0], w0.kind, b0, A[1], w1.kind, b1, pq) : pair_any<true, false>(q, A[0], w0.kind, b0, A[1], w1.kind, b1, pq);
  return a8 ? pair_any<false, true>(q, A[0], w0.kind, b0, A[1], w1.kind, b1, pq) : pair_any<false, false>(q, A[0], w0.kind, b0, A[1], w1.kind, b1, pq);
}

sycl::event pf_dense_splitk(sycl::queue &q, int m, const DenseWeights &w, const bf16 *a, int lda, float *partials, int ld,
                            int ks, int nt, const PfA8 *a8) {
  Args A = base(m, w, a, lda, a8);
  const int groups = A.full + (A.tail > 0 ? 1 : 0);
  if (ks < 1 || ks > groups) throw std::invalid_argument("pf_dense_splitk: ks");
  if (A.row) throw std::invalid_argument("pf_dense_splitk: W8A8 per row has no split-K");
  A.f32 = partials;
  A.ld = ld > 0 ? ld : w.n;
  A.ks = ks;
  A.part_stride = static_cast<size_t>(m) * A.ld;
  if (A.aq) quant_a8(q, A);
  return run_one<kSplit>(q, A, w.kind, nt);
}

class PfSplitSum;

sycl::event pf_splitk_sum(sycl::queue &q, int m, int n, int ks, const float *partials, int ld, const float *scale, float *out, int ldo) {
  if (m < 1 || n < 1 || ks < 1 || ld < n || ldo < n) throw std::invalid_argument("pf_splitk_sum: shape");
  const size_t pstride = static_cast<size_t>(m) * ld;
  return q.parallel_for<PfSplitSum>(sycl::range<1>(static_cast<size_t>(m) * n), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0] / n), c = static_cast<int>(id[0] % n);
    const size_t at = static_cast<size_t>(r) * ld + c;
    float s = partials[at];
    for (int p = 1; p < ks; ++p) s += partials[p * pstride + at];
    if (scale) s *= scale[c];
    out[static_cast<size_t>(r) * ldo + c] = s;
  });
}

}  // namespace xenithium:kernels
