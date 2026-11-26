// Xenithium · W4A16 GEMM over many rows (prefill). See w4a16_gemm.hpp.
// Work-group = 16 work-items on one column block of NT 16-column tiles and 16 row blocks of MB rows. Per 128-k
// group the work-group widens the block's NT x 1024 packed bytes (q - 8, exact in bf16) once into SLM as NT x 8 bf16
// VNNI B operands (double-buffered, one barrier per group), shared by its 16 row blocks; A arrives by 2D block
// loads (rows past m read as zero). Per group each item sums its tiles' fp32 group dot (8 DPAS calls of 16 k) and
// adds it times the group's fp16 scale: decode's per-group arithmetic, the groups in order. A row's result depends
// neither on m nor on where the row sits in the call. Built at 256 GRF (xenithium_kernels_xl): the per-group sums
// sit beside the accumulators.
// Dequantised mode (w4a16_pf_set_dequant): the widen writes bf16((q - 8) x scale) per column, the
// DPAS accumulate over the whole k in fp32 and there is no per-group fp32 sum or fold. That frees the 64 GRF of group sums,
// so the tile grows to 32 rows x 4 tiles. Numerics: each weight rounded to bf16 once, not the decode path's exact group
// sums. Every dq shape gives the same bits.
// Zero points (arch llama, w4_set_zero_points: int4 g32 asymmetric): always the dequantised mode, the widen writing
// bf16((q - z) x s) with z and s of the column's 32-k group (two 16-k calls per group): each weight rounded to bf16 once
// (the exact group-sum mode would apply 4 scales per 128-k block in the epilogue loop).
// W4A8 and the exact mode do not take these matrices.
// Affine groups (arch flint, w4_set_affine: int4 g64, w = q x s + b, bf16 s and b): the widen writes the raw codes
// (exact in bf16). Exact mode: per 64-k group (4 of a 128-k block's 8 calls) the group sum of q a is folded as
// acc += gs x s + asum x b, asum = A's row sum over the group from a small launch before the GEMM (aff_rowsum): decode's
// affine expression. Dequantised mode (w4a16_pf_set_dequant): the widen writes bf16(q x s + b), no fold. W4A8 does not
// take these matrices.
// GLU (w4a16_gemm_pf_swiglu): the matrix is [gate | up] (n = 2 x n_half); a column block holds NT / 2 gate
// tiles and the NT / 2 up tiles of the same columns, so its epilogue writes h = swiglu(gate, up) in bf16 (the swiglu
// kernel's rounding points) instead of both fp32 halves. Each column's sums are the plain kernel's, bit for bit.
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <sycl/ext/intel/esimd.hpp>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "common/exp_cw.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx = esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kWg = 16;  // work-items per work-group
constexpr int kW = 16;  // columns per tile
constexpr int kG = 128;  // k per group
constexpr int kCalls = 8;  // 16-k DPAS calls per group
constexpr int kOp = 256;  // bf16 elements of one B operand (16 k x 16 columns)

struct Args {
  const bf16 *a = nullptr;
  int lda = 0, m = 0, n = 0, k = 0;
  const uint8_t *w = nullptr;
  const sycl::half *s = nullptr;
  float *c = nullptr;
  int ldc = 0;
  bf16 *h = nullptr;  // GLU: [m][ldh] bf16
  int ldh = 0;
  const uint8_t *z = nullptr;  // zero points [k/64][n] (two 32-k groups per byte); s is then fp16 [k/32][n]
  // Affine (g64): biases bf16 [k/64][n] (s is then bf16 bits [k/64][n]); the exact mode also reads A's row sums per
  // 64-k group, asum[g][lds] (aff_rowsum)
  const uint16_t *b = nullptr;
  const float *asum = nullptr;
  int lds = 0;
};

using expdev::exp_cw;  // exp_cw.hpp

template <int NT, int CG>
constexpr int slm_bytes() {
  return 2 * CG * NT * kCalls * kOp * 2;
}

// Work-group = 16 items as R row blocks x CG = 16 / R column groups: the
// group's tile is R x MB rows by CG x NT 16-column tiles. A smaller R reads each A row block for more columns (A is the
// L2 traffic that bounds the 16 x 1 layout at 2,048 rows) at the price of widening more weights per row.
template <int MB, int NT, int R, bool GLU, bool DQ, bool ZP, bool AFF>
class W4Pf;

// AFF (affine g64, w = q s + b, q raw 0..15): dequantised = the widen writes bf16(q s + b); exact = per 64-k group
// the fp32 sum of q a (4 DPAS calls) folded as gs x s + asum x b (decode's affine expression, block_dot_aff), the groups
// in order.
template <int MB, int NT, int R, bool GLU, bool DQ = false, bool ZP = false, bool AFF = false>
sycl::event launch(sycl::queue &q, const Args &A) {
  static_assert(MB % 8 == 0 && MB <= 32 && kWg % R == 0, "w4a16_gemm_pf tile shape");
  static_assert(!ZP || DQ, "zero points: the dequantised mode");
  static_assert(!(ZP && AFF), "zero points or affine");
  static_assert(!GLU || NT % 2 == 0, "GLU: gate and up tiles in pairs");
  constexpr int CG = kWg / R;
  constexpr int NH = GLU ? NT / 2 : NT;  // an item's tiles in the output's column space
  const int tiles = A.n / kW, otiles = GLU ? tiles / 2 : tiles;
  const int cbs = (otiles + CG * NH - 1) / (CG * NH), rgs = (A.m + R * MB - 1) / (R * MB);
  const size_t wgs = static_cast<size_t>(cbs) * rgs;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<W4Pf<MB, NT, R, GLU, DQ, ZP, AFF>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<slm_bytes<NT, CG>()>();
      const int gid = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int rb = gid % rgs, cb = gid / rgs;  // row groups of one column block adjacent: they share its weights in L2
      const int rl = lid % R, cl = lid / R;
      const int m0 = (rb * R + rl) * MB, G = a.k / kG;
      const int tiles_all = a.n / kW, otiles_all = GLU ? tiles_all / 2 : tiles_all;
      // column group g's t-th tile: plain = (cb CG + g) NT + t; GLU = gate tile (cb CG + g) NH + t, then the up tiles of
      // the same columns
      auto tile_of = [&](int g, int t) SYCL_ESIMD_FUNCTION {
        const int o = (cb * CG + g) * NH;
        if constexpr (GLU) return t < NH ? o + t : otiles_all + o + (t - NH);
        else return o + t;
      };
      auto tile_ok = [&](int g, int t) SYCL_ESIMD_FUNCTION { return (cb * CG + g) * NH + (GLU && t >= NH ? t - NH : t) < otiles_all; };
      auto slot = [&](int buf, int g, int t, int j) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>((((buf * CG + g) * NT + t) * kCalls + j) * kOp * 2); };
      const uint16_t *a16 = reinterpret_cast<const uint16_t *>(a.a);
      const unsigned sw = static_cast<unsigned>(a.lda * 2 - 1), sh = static_cast<unsigned>(a.m - 1);
      esimd::simd<float, NT * MB * kW> acc(0.f);
      for (int g = 0; g < G; ++g) {
        const int buf = g & 1;
        // widen: unit u = (column group, tile t, call j); 128 packed bytes -> one VNNI operand (even k low nibble, odd k high)
        for (int u = lid; u < CG * NT * kCalls; u += kWg) {
          const int ug = u / (NT * kCalls), t = (u / kCalls) % NT, j = u % kCalls;
          if (!tile_ok(ug, t)) continue;
          const int tt = tile_of(ug, t);
          const esimd::simd<uint8_t, 128> bytes =
              esimd::block_load<uint8_t, 128>(a.w + (static_cast<size_t>(g) * tiles_all + tt) * kW4BlockBytes + static_cast<size_t>(j) * 128);
          esimd::simd<int16_t, 128> lo, hi;
          if constexpr (ZP) {  // the column's zero of 32-k group sg (byte i of a call's 128 holds column i % 16)
            const int sg = g * (kG / kW4ZpGroup) + j / 2;
            const esimd::simd<uint8_t, kW> zb = esimd::block_load<uint8_t, kW>(a.z + static_cast<size_t>(sg / 2) * a.n + tt * kW);
            const esimd::simd<int16_t, kW> zv = (sg & 1) ? esimd::simd<int16_t, kW>(zb >> 4) : esimd::simd<int16_t, kW>(zb & 15);
            esimd::simd<int16_t, 128> zz;
#pragma unroll
            for (int x = 0; x < 8; ++x) zz.template select<16, 1>(x * 16) = zv;
            lo = esimd::simd<int16_t, 128>(bytes & 15) - zz;
            hi = esimd::simd<int16_t, 128>(bytes >> 4) - zz;
          } else if constexpr (AFF) {  // raw codes
            lo = esimd::simd<int16_t, 128>(bytes & 15);
            hi = esimd::simd<int16_t, 128>(bytes >> 4);
          } else {
            lo = esimd::simd<int16_t, 128>(bytes & 15) - kW4Zero;
            hi = esimd::simd<int16_t, 128>(bytes >> 4) - kW4Zero;
          }
          esimd::simd<bf16, kOp> b;
          if constexpr (DQ && AFF) {  // bf16(q x s + b) with s, b of the column's 64-k group
            const size_t srow = static_cast<size_t>(g * 2 + j / 4);
            const esimd::simd<uint32_t, kW> su = esimd::block_load<uint16_t, kW>(reinterpret_cast<const uint16_t *>(a.s) + srow * a.n + tt * kW);
            const esimd::simd<uint32_t, kW> bu = esimd::block_load<uint16_t, kW>(a.b + srow * a.n + tt * kW);
            esimd::simd<uint32_t, kW> sw32 = su << 16, bw32 = bu << 16;
            const esimd::simd<float, kW> sv = sw32.template bit_cast_view<float>().read(), bv = bw32.template bit_cast_view<float>().read();
            esimd::simd<float, 128> scl, bia;
#pragma unroll
            for (int x = 0; x < 8; ++x) scl.template select<16, 1>(x * 16) = sv, bia.template select<16, 1>(x * 16) = bv;
            b.template select<128, 2>(0) = esimd::simd<float, 128>(lo) * scl + bia;
            b.template select<128, 2>(1) = esimd::simd<float, 128>(hi) * scl + bia;
          } else if constexpr (DQ) {  // bf16((q - 8) x scale); byte i of a call's 128 holds column i % 16
            const size_t srow = ZP ? static_cast<size_t>(g * (kG / kW4ZpGroup) + j / 2) : static_cast<size_t>(g);  // ZP: s [k/32][n]
            const esimd::simd<float, kW> sv = esimd::block_load<sycl::half, kW>(a.s + srow * a.n + tt * kW);
            esimd::simd<float, 128> scl;
#pragma unroll
            for (int x = 0; x < 8; ++x) scl.template select<16, 1>(x * 16) = sv;
            b.template select<128, 2>(0) = esimd::simd<float, 128>(lo) * scl;
            b.template select<128, 2>(1) = esimd::simd<float, 128>(hi) * scl;
          } else {
            b.template select<128, 2>(0) = esimd::simd<float, 128>(lo);
            b.template select<128, 2>(1) = esimd::simd<float, 128>(hi);
          }
          esimd::slm_block_store<bf16, kOp>(slot(buf, ug, t, j), b);
        }
        esimd::barrier();
        if (m0 < a.m) {
          esimd::simd<float, DQ ? kW : NT * MB * kW> gs(0.f);  // dq: unused (the DPAS accumulate into acc)
#pragma unroll
          for (int j = 0; j < kCalls; ++j) {
            esimd::simd<uint16_t, MB * 16> au = esimd::load_2d<uint16_t, 16, MB>(a16, sw, sh, sw, g * kG + j * 16, m0);
            esimd::simd<bf16, MB * 16> av = au.template bit_cast_view<bf16>();
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(slot(buf, cl, t, j));
#pragma unroll
              for (int r = 0; r < MB / 8; ++r) {
                esimd::simd<bf16, 128> ar = av.template select<128, 1>(r * 128);
                if constexpr (DQ) {
                  auto cv = acc.template select<128, 1>((t * MB + r * 8) * kW);
                  esimd::simd<float, 128> cc = cv;
                  cv = xmx::dpas<8, 8, float, float, bf16, bf16>(cc, b, ar);
                } else {
                  auto cv = gs.template select<128, 1>((t * MB + r * 8) * kW);
                  esimd::simd<float, 128> cc = cv;
                  cv = xmx::dpas<8, 8, float, float, bf16, bf16>(cc, b, ar);
                }
              }
            }
            if constexpr (AFF && !DQ) {
              if (j % 4 == 3) {  // the 64-k group's fold: acc += gs x s + asum x b, then a fresh group sum
                const int sg = g * 2 + j / 4;
                const esimd::simd<float, MB> as = esimd::block_load<float, MB>(a.asum + static_cast<size_t>(sg) * a.lds + m0);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                  const int tt = tile_ok(cl, t) ? tile_of(cl, t) : tiles_all - 1;
                  const esimd::simd<uint32_t, kW> su = esimd::block_load<uint16_t, kW>(reinterpret_cast<const uint16_t *>(a.s) + static_cast<size_t>(sg) * a.n + tt * kW);
                  const esimd::simd<uint32_t, kW> bu = esimd::block_load<uint16_t, kW>(a.b + static_cast<size_t>(sg) * a.n + tt * kW);
                  esimd::simd<uint32_t, kW> sw32 = su << 16, bw32 = bu << 16;
                  const esimd::simd<float, kW> sv = sw32.template bit_cast_view<float>().read(), bv = bw32.template bit_cast_view<float>().read();
                  // explicit fmas: acc = fma(b, asum, fma(gs, s, acc)), the same bits at every tile shape (the plain
                  // expression contracts differently per shape)
#pragma unroll
                  for (int r = 0; r < MB; ++r) {
                    auto av = acc.template select<kW, 1>((t * MB + r) * kW);
                    const esimd::simd<float, kW> t1 = xesimd::fma(gs.template select<kW, 1>((t * MB + r) * kW).read(), sv, av.read());
                    av = xesimd::fma(bv, esimd::simd<float, kW>(static_cast<float>(as[r])), t1);
                  }
                }
                gs = 0.f;
              }
            }
          }
          if constexpr (!DQ && !AFF) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const int tt = tile_ok(cl, t) ? tile_of(cl, t) : tiles_all - 1;
              const esimd::simd<float, kW> sv = esimd::block_load<sycl::half, kW>(a.s + static_cast<size_t>(g) * a.n + tt * kW);
#pragma unroll
              for (int r = 0; r < MB; ++r)
                acc.template select<kW, 1>((t * MB + r) * kW) += gs.template select<kW, 1>((t * MB + r) * kW) * sv;
            }
          }
        }
      }
      if (m0 >= a.m) return;
      // GLU h: a row's adjacent tiles as one dword block store of bf16 pairs.
      auto store16 = [&]<int E>(bf16 *row, esimd::simd<bf16, E> h) SYCL_ESIMD_FUNCTION {
        const esimd::simd<uint32_t, E / 2> d = h.template bit_cast_view<uint32_t>();
        d.copy_to(reinterpret_cast<uint32_t *>(row));
      };
      if constexpr (GLU) {
        if (!tile_ok(cl, NH - 1)) {  // the column block's last tiles: per tile
#pragma unroll
          for (int t = 0; t < NH; ++t) {
            if (!tile_ok(cl, t)) continue;
            const int ot = (cb * CG + cl) * NH + t;
#pragma unroll
            for (int r = 0; r < MB; ++r) {
              if (m0 + r >= a.m) break;
              // swiglu's rounding points (mlp.cpp): g, u -> bf16; h = bf16(g sigmoid(g) u)
              const esimd::simd<float, kW> gv = esimd::simd<bf16, kW>(acc.template select<kW, 1>((t * MB + r) * kW));
              const esimd::simd<float, kW> uv = esimd::simd<bf16, kW>(acc.template select<kW, 1>(((t + NH) * MB + r) * kW));
              const esimd::simd<float, kW> sg = gv * (esimd::simd<float, kW>(1.f) / (esimd::simd<float, kW>(1.f) + exp_cw(-gv)));
              const esimd::simd<bf16, kW> hv = sg * uv;
              store16(a.h + static_cast<size_t>(m0 + r) * a.ldh + static_cast<size_t>(ot) * kW, hv);
            }
          }
        } else {
          const int ot0 = (cb * CG + cl) * NH;
#pragma unroll
          for (int r = 0; r < MB; ++r) {
            if (m0 + r >= a.m) break;
            esimd::simd<bf16, NH * kW> hr;
#pragma unroll
            for (int t = 0; t < NH; ++t) {
              const esimd::simd<float, kW> gv = esimd::simd<bf16, kW>(acc.template select<kW, 1>((t * MB + r) * kW));
              const esimd::simd<float, kW> uv = esimd::simd<bf16, kW>(acc.template select<kW, 1>(((t + NH) * MB + r) * kW));
              const esimd::simd<float, kW> sg = gv * (esimd::simd<float, kW>(1.f) / (esimd::simd<float, kW>(1.f) + exp_cw(-gv)));
              hr.template select<kW, 1>(t * kW) = sg * uv;
            }
            store16(a.h + static_cast<size_t>(m0 + r) * a.ldh + static_cast<size_t>(ot0) * kW, hr);
          }
        }
      } else {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          if (!tile_ok(cl, t)) continue;
          const int tt = tile_of(cl, t);
#pragma unroll
          for (int r = 0; r < MB; ++r) {
            if (m0 + r >= a.m) break;
            const esimd::simd<float, kW> v = acc.template select<kW, 1>((t * MB + r) * kW);
            v.copy_to(a.c + static_cast<size_t>(m0 + r) * a.ldc + static_cast<size_t>(tt) * kW);
          }
        }
      }
    });
  });
}

// The default shape by rows and columns: 8 x 4 for a
// few rows; between 96 and 512 rows the shape that keeps enough work-groups (a narrow n wants NT 2); from 512 rows
// 32 x 2 on 8 x 2 items (MB 32 halves the SLM B reads per DPAS).
int pf_shape(int m, int n) {
  if (m <= 96) return 2;
  if (m < 256) return n >= 8192 ? 4 : 0;
  if (m < 512) return n >= 8192 ? 7 : 0;
  return 7;
}

bool g_dequant = false;

// Dequantised shapes:
// 32 x 4 on R 8 from 192 rows (and for n < 8,192 below it), 16 x 4 on R 16 for wide n under 192 rows. shape: 13 = 16 x 4
// R 16, 14 = 32 x 2 R 8, 15 = 8 x 4 R 16, 12 / other = 32 x 4 R 8; -1 = by m and n.
// ZP (the zero-point matrices): 8 x 4 (15) up to 96 rows, 16 x 4 (13) below 192, then 12.
template <bool GLU, bool ZP = false, bool AFF = false>
sycl::event launch_dq(sycl::queue &q, const Args &A, int shape) {
  if (shape < 0 && (ZP || AFF)) shape = A.m <= 96 ? 15 : A.m < 192 ? 13 : 12;
  if (shape < 0) shape = A.m < 192 && A.n >= 8192 ? 13 : 12;
  switch (shape) {
    case 13: return launch<16, 4, 16, GLU, true, ZP, AFF>(q, A);
    case 14: return launch<32, 2, 8, GLU, true, ZP, AFF>(q, A);
    case 15: return launch<8, 4, 16, GLU, true, ZP, AFF>(q, A);
    default: return launch<32, 4, 8, GLU, true, ZP, AFF>(q, A);
  }
}

// ---- Affine g64 (arch flint): A's row sums per 64-k group, then the exact-mode GEMM ------------------------------
// item = 16 rows x one 64-k group (two 16 x 32 bf16 2D loads; rows past m read zero); asum[g][lds] = the row's fp32 sum
// of the group, esimd:reduce over the 64 values as the decode tile (block_dot_aff) sums them.
class W4AffRowSum;
struct AffScr {
  float *s = nullptr;
  int max_m = 0, max_k = 0;
};
std::mutex g_affs_mu;
std::unordered_map<sycl::device, AffScr> g_affs;
inline int aff_ld(int m) { return (m + 31) / 32 * 32; }

sycl::event aff_rowsum(sycl::queue &q, const bf16 *a, int lda, int m, int k, float *asum, int lds) {
  const int gpr = k / kW4AffGroup;
  const size_t items = static_cast<size_t>((m + 15) / 16) * gpr;
  return q.parallel_for<W4AffRowSum>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t id = it.get_global_id(0);
    if (id >= items) return;
    const int rb = static_cast<int>(id / gpr), g = static_cast<int>(id % gpr), r0 = rb * 16;
    const uint16_t *a16 = reinterpret_cast<const uint16_t *>(a);
    const unsigned sw = static_cast<unsigned>(lda * 2 - 1), sh = static_cast<unsigned>(m - 1);
    esimd::simd<uint16_t, 512> h0 = esimd::load_2d<uint16_t, 32, 16>(a16, sw, sh, sw, g * 64, r0);
    esimd::simd<uint16_t, 512> h1 = esimd::load_2d<uint16_t, 32, 16>(a16, sw, sh, sw, g * 64 + 32, r0);
    esimd::simd<float, 16> out;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      esimd::simd<uint16_t, 64> u;
      u.template select<32, 1>(0) = h0.template select<32, 1>(i * 32);
      u.template select<32, 1>(32) = h1.template select<32, 1>(i * 32);
      const esimd::simd<float, 64> v = u.template bit_cast_view<bf16>().read();
      out[i] = esimd::reduce<float>(v, std::plus<>());
    }
    esimd::block_store<float, 16>(asum + static_cast<size_t>(g) * lds + r0, out);
  });
}

// the exact-mode shapes (pf_shape's numbers: 2 = 8 x 4, 4 = 16 x 4, 7 = 32 x 2 R 8, other = 16 x 2)
template <bool GLU>
sycl::event launch_aff(sycl::queue &q, Args A, int shape) {
  if (g_dequant) return launch_dq<GLU, false, true>(q, A, shape >= 12 && shape <= 15 ? shape : -1);
  AffScr sc;
  {
    std::lock_guard<std::mutex> lk(g_affs_mu);
    const auto it = g_affs.find(q.get_device());
    if (it == g_affs.end()) throw std::logic_error("w4a16_gemm_pf: MLX affine without a row-sum scratch on this device (w4a16_pf_aff_reserve)");
    sc = it->second;
  }
  if (A.m > sc.max_m || A.k > sc.max_k) throw std::invalid_argument("w4a16_gemm_pf: MLX affine call larger than the reserved row-sum scratch");
  A.lds = aff_ld(A.m);
  A.asum = sc.s;
  aff_rowsum(q, A.a, A.lda, A.m, A.k, sc.s, A.lds);
  if (shape < 0 || shape >= 12) shape = pf_shape(A.m, A.n);
  switch (shape) {
    case 2: return launch<8, 4, 16, GLU, false, false, true>(q, A);
    case 4: return launch<16, 4, 16, GLU, false, false, true>(q, A);
    case 7: return launch<32, 2, 8, GLU, false, false, true>(q, A);
    default: return launch<16, 2, 16, GLU, false, false, true>(q, A);
  }
}


// ---- W4A8: s8 activations x s4 DPAS on the int4 blocks --------------------------------------
// The activations are quantised per (row, 128-k group) with row_quant_rows' rule (s = amax / 127, rne, +-127; the experts'
// and the dense trunk's W4A8 / W8A8 rule; quant_2d16 below, 16 rows per item by 2D blocks) into a per-device scratch:
// int8 [m][k] + fp32 scales group-major [k/128][lds]. The GEMM
// work-group is W4Pf's (R row blocks x CG column groups of NT tiles, MB rows per item) but stages the column block's int4
// blocks into SLM as s4 DPAS B operands (no widen to bf16; reordered from the nib layout, below); A arrives per 32-k call as int8 rows
// by 2D block loads (rows past m read zero). Per group: four s8 x s4 DPAS calls per (tile, 8 rows) into int32 group sums
// (exact), then acc += float(sum) x (w_scale[col] x a_scale[row]) in fp32, the groups in order. Epilogues as W4Pf (plain
// fp32 / GLU swiglu). The weights' and activations' scales enter once per group: a numerics change against W4A16.
struct A8Scr {
  int8_t *q = nullptr;
  float *s = nullptr;
  int max_m = 0, max_k = 0;
};
struct A8State {
  bool on = false;
  std::unordered_set<const void *> keep;  // matrices kept on W4A16 while W4A8 is on
  std::mutex mu;
  std::unordered_map<sycl::device, A8Scr> scr;
  struct In {  // a producer's codes (w4a16_pf_a8_input)
    const bf16 *src = nullptr;
    int lda = 0, rows = 0, k = 0, lds = 0;
    const int8_t *q = nullptr;
    const float *s = nullptr;
    const float *s_rows = nullptr;  // the scales row-major [rows][k / 128] too (oneDNN's layout), or null
  };
  std::unordered_map<sycl::device, In> in;
  // The codes of GLU outputs h that the oneDNN GLU path quantised with the swiglu (glu_q8), at most kHin per
  // device (the split MLP fills h in row ranges); an entry dies when anything rewrites rows of its h (every
  // w4a16_gemm_pf_swiglu call drops the entries its output overlaps first) or with w4a16_pf_a8_input_clear.
  static constexpr int kHin = 4;
  std::unordered_map<sycl::device, std::vector<In>> hin;
  struct HScr {
    int8_t *q = nullptr;
    float *s = nullptr;
    size_t qcap = 0, scap = 0, qused = 0, sused = 0;
  };
  std::unordered_map<sycl::device, HScr> hscr;
} g_a8;
inline int a8_ld(int m) { return (m + 31) / 32 * 32; }

class W4A8Quant;
template <int MB, int NT, int R, bool GLU, int KB, int HALF>
class W4A8Pf;

// The blocks as packed (the nib layout, the decode GEMMs' and W4A16's); each 256-byte call is reordered into an
// s4 DPAS B operand while it is staged (per 64 bytes: s4 dword `lane` = byte `lane` of the 4 k-pair rows, low row first;
// then q ^ 8 per nibble = q - 8 as a two's-complement nibble).
// The pipeline: a stage = KB groups behind one barrier, double-buffered in SLM. Each item owns at most one
// 256-byte unit per group (UPG = CG x NT x 4 <= 16 on every shape) and loads the NEXT stage's units into registers before
// this stage's DPAS, reorders and stores them after (the load latency hidden behind the DPAS, one barrier per KB groups);
// A arrives one 32-k call ahead (register double buffer, across groups too).
// The per-item operand traffic through the LSC is A 256 / NT + B 256 / (MB / 8) bytes per DPAS: 192 at 32 x 2 or 16 x 4,
// 149 at 24 x 4.
// HALF 1 = the 24 x 4 HALF body: the group's A held (4 calls), the
// tiles processed in two halves with a half-size int32 sum buffer (24 x 4 fits the 256-GRF file without spilling: A 64 + B
// 85 = 149 bytes per DPAS vs 192 at 32 x 2), the next group's A loaded during the second half. Same group sums and fold
// expression: the same outputs bit for bit. Stages of 4 groups for the plain GEMMs and 2 for the GLU (the reverse of the
// 32 x 2 body's). Taken where a8_half_pays.
template <int MB, int NT, int R, bool GLU, int KB, int HALF = 0>
sycl::event launch_a8(sycl::queue &q, const Args &A, const int8_t *aq, const float *as, int lds) {
  static_assert(MB % 8 == 0 && MB <= 32 && kWg % R == 0, "w4a8 tile shape");
  static_assert(!GLU || NT % 2 == 0, "GLU: gate and up tiles in pairs");
  constexpr int CG = kWg / R;
  constexpr int NH = GLU ? NT / 2 : NT;
  constexpr int kOp8 = 256, kCalls8 = 4;  // s4 B operand bytes; 32-k calls per group
  constexpr int UPG = CG * NT * kCalls8;  // staged 256-byte units per group
  static_assert(UPG <= kWg, "w4a8: at most one staged unit per item and group");
  const int tiles = A.n / kW, otiles = GLU ? tiles / 2 : tiles;
  const int cbs = (otiles + CG * NH - 1) / (CG * NH), rgs = (A.m + R * MB - 1) / (R * MB);
  const size_t wgs = static_cast<size_t>(cbs) * rgs;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<W4A8Pf<MB, NT, R, GLU, KB, HALF>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<2 * KB * CG * NT * kW4BlockBytes>();
      const int gid = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int rb = gid % rgs, cb = gid / rgs;
      const int rl = lid % R, cl = lid / R;
      const int m0 = (rb * R + rl) * MB, G = a.k / kG, S = (G + KB - 1) / KB;
      const int tiles_all = a.n / kW, otiles_all = GLU ? tiles_all / 2 : tiles_all;
      auto tile_of = [&](int g, int t) SYCL_ESIMD_FUNCTION {
        const int o = (cb * CG + g) * NH;
        if constexpr (GLU) return t < NH ? o + t : otiles_all + o + (t - NH);
        else return o + t;
      };
      auto tile_ok = [&](int g, int t) SYCL_ESIMD_FUNCTION { return (cb * CG + g) * NH + (GLU && t >= NH ? t - NH : t) < otiles_all; };
      auto slot = [&](int buf, int j, int g, int t, int c) SYCL_ESIMD_FUNCTION {
        return static_cast<uint32_t>(((((buf * KB + j) * CG + g) * NT + t) * kCalls8 + c) * kOp8);
      };
      const uint32_t *a32 = reinterpret_cast<const uint32_t *>(aq);
      const unsigned sw = static_cast<unsigned>(a.k - 1), sh = static_cast<unsigned>(a.m - 1);
      // this item's staged unit: (column group, tile, 32-k call)
      const int ug = lid / (NT * kCalls8), ut = (lid / kCalls8) % NT, uc = lid % kCalls8;
      const bool uok = lid < UPG && tile_ok(ug, ut);
      const size_t wcol = static_cast<size_t>(tile_of(ug, ut)) * kW4BlockBytes + uc * kOp8;
      auto wload = [&](int g) SYCL_ESIMD_FUNCTION {
        return esimd::block_load<uint32_t, kOp8 / 4>(reinterpret_cast<const uint32_t *>(a.w + static_cast<size_t>(g) * tiles_all * kW4BlockBytes + wcol));
      };
      auto wstore = [&](esimd::simd<uint32_t, kOp8 / 4> v, int buf, int j) SYCL_ESIMD_FUNCTION {
        esimd::simd<uint8_t, kOp8> nb = v.template bit_cast_view<uint8_t>();  // s4 dword (lane) = the lane's byte of its 4 pair rows
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          esimd::simd<uint32_t, 16> w = esimd::simd<uint32_t, 16>(nb.template select<16, 1>(q4 * 64));
#pragma unroll
          for (int b = 1; b < 4; ++b) w |= esimd::simd<uint32_t, 16>(nb.template select<16, 1>(q4 * 64 + b * 16)) << (8 * b);
          v.template select<16, 1>(q4 * 16) = w ^ 0x88888888u;
        }
        esimd::slm_block_store<uint32_t, kOp8 / 4>(slot(buf, j, ug, ut, uc), v);
      };
      auto aload = [&](int g, int c) SYCL_ESIMD_FUNCTION { return esimd::load_2d<uint32_t, 8, MB>(a32, sw, sh, sw, (g * kG + c * 32) / 4, m0); };
      esimd::simd<float, NT * MB * kW> acc(0.f);
      esimd::simd<uint32_t, 8 * MB> anext = aload(0, 0);
      esimd::simd<uint32_t, 8 * MB> ag[4];  // HALF: the group's A (4 calls)
      if constexpr (HALF != 0) { ag[0] = anext; ag[1] = aload(0, 1); ag[2] = aload(0, 2); ag[3] = aload(0, 3); }
      if (uok) {
#pragma unroll
        for (int j = 0; j < KB; ++j)
          if (j < G) wstore(wload(j), 0, j);
      }
      esimd::barrier();
      for (int s = 0; s < S; ++s) {
        const int buf = s & 1;
        esimd::simd<uint32_t, kOp8 / 4> wn[KB];
        if (uok) {
#pragma unroll
          for (int j = 0; j < KB; ++j)
            if ((s + 1) * KB + j < G) wn[j] = wload((s + 1) * KB + j);
        }
        if constexpr (HALF != 0) {
          static_assert(NT % 2 == 0, "HALF: tile pairs");
#pragma unroll
          for (int j = 0; j < KB; ++j) {
            const int g = s * KB + j;
            if (g >= G || m0 >= a.m) break;
            const esimd::simd<float, MB> sa = esimd::block_load<float, MB>(as + static_cast<size_t>(g) * lds + m0);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              esimd::simd<int32_t, NT / 2 * MB * kW> gh(0);
#pragma unroll
              for (int c = 0; c < kCalls8; ++c) {
                esimd::simd<uint8_t, 32 * MB> av = ag[c].template bit_cast_view<uint8_t>();
#pragma unroll
                for (int th = 0; th < NT / 2; ++th) {
                  const int t = h * (NT / 2) + th;
                  const esimd::simd<uint8_t, kOp8> b = esimd::slm_block_load<uint8_t, kOp8>(slot(buf, j, cl, t, c));
#pragma unroll
                  for (int r = 0; r < MB / 8; ++r) {
                    auto gv = gh.template select<128, 1>((th * MB + r * 8) * kW);
                    esimd::simd<int32_t, 128> cc = gv;
                    gv = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, uint8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8, 128, kOp8, 256>(
                        cc, b, av.template select<256, 1>(r * 256));
                  }
                }
                if (h == 1 && g + 1 < G) ag[c] = aload(g + 1, c);  // the next group's A into the slot just consumed
              }
#pragma unroll
              for (int th = 0; th < NT / 2; ++th) {
                const int t = h * (NT / 2) + th;
                const int tt = tile_ok(cl, t) ? tile_of(cl, t) : tiles_all - 1;
                const esimd::simd<float, kW> sv = esimd::block_load<sycl::half, kW>(a.s + static_cast<size_t>(g) * a.n + tt * kW);
#pragma unroll
                for (int r = 0; r < MB; ++r) {
                  const esimd::simd<float, kW> f = gh.template select<kW, 1>((th * MB + r) * kW);
                  acc.template select<kW, 1>((t * MB + r) * kW) += f * (sv * sa[r]);
                }
              }
            }
          }
        } else {
#pragma unroll
        for (int j = 0; j < KB; ++j) {
          const int g = s * KB + j;
          if (g >= G || m0 >= a.m) break;
          esimd::simd<int32_t, NT * MB * kW> gs(0);
#pragma unroll
          for (int c = 0; c < kCalls8; ++c) {
            esimd::simd<uint32_t, 8 * MB> av32 = anext;
            if (c + 1 < kCalls8) anext = aload(g, c + 1);
            else if (g + 1 < G) anext = aload(g + 1, 0);
            esimd::simd<uint8_t, 32 * MB> av = av32.template bit_cast_view<uint8_t>();
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const esimd::simd<uint8_t, kOp8> b = esimd::slm_block_load<uint8_t, kOp8>(slot(buf, j, cl, t, c));
#pragma unroll
              for (int r = 0; r < MB / 8; ++r) {
                auto gv = gs.template select<128, 1>((t * MB + r * 8) * kW);
                esimd::simd<int32_t, 128> cc = gv;
                gv = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, uint8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8, 128, kOp8, 256>(
                    cc, b, av.template select<256, 1>(r * 256));
              }
            }
          }
          const esimd::simd<float, MB> sa = esimd::block_load<float, MB>(as + static_cast<size_t>(g) * lds + m0);
          esimd::simd<float, NT * kW> svt;
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const int tt = tile_ok(cl, t) ? tile_of(cl, t) : tiles_all - 1;
            svt.template select<kW, 1>(t * kW) = esimd::block_load<sycl::half, kW>(a.s + static_cast<size_t>(g) * a.n + tt * kW);
          }
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            const esimd::simd<float, kW> sv = svt.template select<kW, 1>(t * kW);
#pragma unroll
            for (int r = 0; r < MB; ++r) {
              const esimd::simd<float, kW> f = gs.template select<kW, 1>((t * MB + r) * kW);
              acc.template select<kW, 1>((t * MB + r) * kW) += f * (sv * sa[r]);
            }
          }
        }
        }  // HALF
        if (uok) {
#pragma unroll
          for (int j = 0; j < KB; ++j)
            if ((s + 1) * KB + j < G) wstore(wn[j], buf ^ 1, j);
        }
        esimd::barrier();
      }
      if (m0 >= a.m) return;
      auto store16 = [&]<int E>(bf16 *row, esimd::simd<bf16, E> h) SYCL_ESIMD_FUNCTION {
        const esimd::simd<uint32_t, E / 2> d = h.template bit_cast_view<uint32_t>();
        d.copy_to(reinterpret_cast<uint32_t *>(row));
      };
      if constexpr (GLU) {
#pragma unroll
        for (int t = 0; t < NH; ++t) {
          if (!tile_ok(cl, t)) continue;
          const int ot = (cb * CG + cl) * NH + t;
#pragma unroll
          for (int r = 0; r < MB; ++r) {
            if (m0 + r >= a.m) break;
            const esimd::simd<float, kW> gv = esimd::simd<bf16, kW>(acc.template select<kW, 1>((t * MB + r) * kW));
            const esimd::simd<float, kW> uv = esimd::simd<bf16, kW>(acc.template select<kW, 1>(((t + NH) * MB + r) * kW));
            const esimd::simd<float, kW> sg = gv * (esimd::simd<float, kW>(1.f) / (esimd::simd<float, kW>(1.f) + exp_cw(-gv)));
            const esimd::simd<bf16, kW> hv = sg * uv;
            store16(a.h + static_cast<size_t>(m0 + r) * a.ldh + static_cast<size_t>(ot) * kW, hv);
          }
        }
      } else {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          if (!tile_ok(cl, t)) continue;
          const int tt = tile_of(cl, t);
#pragma unroll
          for (int r = 0; r < MB; ++r) {
            if (m0 + r >= a.m) break;
            const esimd::simd<float, kW> v = acc.template select<kW, 1>((t * MB + r) * kW);
            v.copy_to(a.c + static_cast<size_t>(m0 + r) * a.ldc + static_cast<size_t>(tt) * kW);
          }
        }
      }
    });
  });
}


// The same arithmetic without SLM or barriers: weights and A by direct loads with L1 prefetch, 2-call messages.
// Work-group = 16 items on ONE row block of MB
// rows (R 1) x 16 column groups of NT tiles: every item owns its tiles' columns, so each 256-byte nib unit is loaded by
// exactly one item and reordered into the s4 DPAS operand in registers (the staging's arithmetic, done once per unit);
// A (MB rows x 32 bytes per call) by 2-block 2D loads of two calls, shared by the 16 items through L1. The next group's
// units and A rows are prefetched into L1 at the top of each group. Per group the exact int32 sums are folded as the
// pipelined body folds them (acc += float(sum) x (s_w x s_a)): the same outputs bit for bit.
// MB 16 x NT 2 up to 64 rows, else 32 x 2.
template <int MB, int NT, bool GLU, int R = 1>
class W4A8Pn;
template <int MB, int NT, bool GLU, int R = 1>
sycl::event launch_a8n(sycl::queue &q, const Args &A, const int8_t *aq, const float *as, int lds) {
  static_assert(MB % 8 == 0 && MB <= 32 && kWg % R == 0, "w4a8n tile shape");
  static_assert(!GLU || NT % 2 == 0, "GLU: gate and up tiles in pairs");
  constexpr int CG = kWg / R, NH = GLU ? NT / 2 : NT;
  constexpr int kOp8 = 256, kCalls8 = 4;
  const int tiles = A.n / kW, otiles = GLU ? tiles / 2 : tiles;
  const int cbs = (otiles + CG * NH - 1) / (CG * NH), rgs = (A.m + R * MB - 1) / (R * MB);
  const size_t wgs = static_cast<size_t>(cbs) * rgs;
  const Args a = A;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<W4A8Pn<MB, NT, GLU, R>>(sycl::nd_range<1>(wgs * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int rb = gid % rgs, cb = gid / rgs, rl = lid % R, cl = lid / R;
      const int m0 = (rb * R + rl) * MB, G = a.k / kG;
      if (m0 >= a.m) return;
      const int tiles_all = a.n / kW, otiles_all = GLU ? tiles_all / 2 : tiles_all;
      const int o = (cb * CG + cl) * NH;  // this item's first output tile
      if (o >= otiles_all) return;
      int tt[NT];  // the matrix tiles (GLU: NH gate tiles, then the NH up tiles of the same columns), clamped in range
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int ot = o + (GLU && t >= NH ? t - NH : t);
        const int c = ot < otiles_all ? ot : otiles_all - 1;
        tt[t] = GLU && t >= NH ? otiles_all + c : c;
      }
      const uint32_t *a32 = reinterpret_cast<const uint32_t *>(aq);
      const unsigned sw = static_cast<unsigned>(a.k - 1), sh = static_cast<unsigned>(a.m - 1);
      auto unit = [&](int g, int t, int c) SYCL_ESIMD_FUNCTION {
        return reinterpret_cast<const uint32_t *>(a.w + (static_cast<size_t>(g) * tiles_all + tt[t]) * kW4BlockBytes + c * kOp8);
      };
      // s4 dword (lane) = the lane's byte of its 4 pair rows, low row first; q ^ 8 per nibble (as the pipelined body's staging)
      auto reorder = [&](esimd::simd<uint32_t, kOp8 / 4> v) SYCL_ESIMD_FUNCTION {
        esimd::simd<uint8_t, kOp8> nb = v.template bit_cast_view<uint8_t>();
        esimd::simd<uint32_t, kOp8 / 4> r;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          esimd::simd<uint32_t, 16> w = esimd::simd<uint32_t, 16>(nb.template select<16, 1>(q4 * 64));
#pragma unroll
          for (int b = 1; b < 4; ++b) w |= esimd::simd<uint32_t, 16>(nb.template select<16, 1>(q4 * 64 + b * 16)) << (8 * b);
          r.template select<16, 1>(q4 * 16) = w ^ 0x88888888u;
        }
        return r;
      };
      constexpr auto pfp = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>};
      esimd::simd<float, NT * MB * kW> acc(0.f);
      for (int g = 0; g < G; ++g) {
        if (g + 1 < G) {  // the next group into L1: this item's units and the row block's A (128 bytes per row)
#pragma unroll
          for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int c = 0; c < kCalls8; ++c) esimd::prefetch<uint32_t, 64>(unit(g + 1, t, c), pfp);
          esimd::prefetch_2d<uint32_t, 8, MB, 2>(a32, sw, sh, sw, ((g + 1) * kG) / 4, m0, pfp);
          esimd::prefetch_2d<uint32_t, 8, MB, 2>(a32, sw, sh, sw, ((g + 1) * kG) / 4 + 16, m0, pfp);
        }
        esimd::simd<int32_t, NT * MB * kW> gs(0);
#pragma unroll
        for (int c2 = 0; c2 < kCalls8; c2 += 2) {  // two 32-k calls per message
          esimd::simd<uint32_t, 16 * MB> av2 = esimd::load_2d<uint32_t, 8, MB, 2>(a32, sw, sh, sw, (g * kG + c2 * 32) / 4, m0);
          esimd::simd<uint32_t, kOp8 / 2> braw[NT];
#pragma unroll
          for (int t = 0; t < NT; ++t) {
            braw[t] = esimd::block_load<uint32_t, kOp8 / 2>(unit(g, t, c2));
            braw[t].template select<kOp8 / 4, 1>(0) = reorder(braw[t].template select<kOp8 / 4, 1>(0).read());
            braw[t].template select<kOp8 / 4, 1>(kOp8 / 4) = reorder(braw[t].template select<kOp8 / 4, 1>(kOp8 / 4).read());
          }
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            esimd::simd<uint8_t, 32 * MB> av = av2.template select<8 * MB, 1>(h * 8 * MB).template bit_cast_view<uint8_t>();
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              esimd::simd<uint8_t, kOp8> b = braw[t].template select<kOp8 / 4, 1>(h * kOp8 / 4).template bit_cast_view<uint8_t>();
#pragma unroll
              for (int r = 0; r < MB / 8; ++r) {
                auto gv = gs.template select<128, 1>((t * MB + r * 8) * kW);
                esimd::simd<int32_t, 128> cc = gv;
                gv = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, uint8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8, 128, kOp8, 256>(
                    cc, b, av.template select<256, 1>(r * 256));
              }
            }
          }
        }
        const esimd::simd<float, MB> sa = esimd::block_load<float, MB>(as + static_cast<size_t>(g) * lds + m0);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const esimd::simd<float, kW> sv = esimd::block_load<sycl::half, kW>(a.s + static_cast<size_t>(g) * a.n + tt[t] * kW);
#pragma unroll
          for (int r = 0; r < MB; ++r) {
            const esimd::simd<float, kW> f = gs.template select<kW, 1>((t * MB + r) * kW);
            acc.template select<kW, 1>((t * MB + r) * kW) += f * (sv * sa[r]);
          }
        }
      }
      auto store16 = [&]<int E>(bf16 *row, esimd::simd<bf16, E> h) SYCL_ESIMD_FUNCTION {
        const esimd::simd<uint32_t, E / 2> d = h.template bit_cast_view<uint32_t>();
        d.copy_to(reinterpret_cast<uint32_t *>(row));
      };
#pragma unroll
      for (int t = 0; t < NH; ++t) {
        const int ot = o + t;
        if (ot >= otiles_all) continue;
#pragma unroll
        for (int r = 0; r < MB; ++r) {
          if (m0 + r >= a.m) break;
          if constexpr (GLU) {
            const esimd::simd<float, kW> gv = esimd::simd<bf16, kW>(acc.template select<kW, 1>((t * MB + r) * kW));
            const esimd::simd<float, kW> uv = esimd::simd<bf16, kW>(acc.template select<kW, 1>(((t + NH) * MB + r) * kW));
            const esimd::simd<float, kW> sg = gv * (esimd::simd<float, kW>(1.f) / (esimd::simd<float, kW>(1.f) + exp_cw(-gv)));
            const esimd::simd<bf16, kW> hv = sg * uv;
            store16(a.h + static_cast<size_t>(m0 + r) * a.ldh + static_cast<size_t>(ot) * kW, hv);
          } else {
            const esimd::simd<float, kW> v = acc.template select<kW, 1>((t * MB + r) * kW);
            v.copy_to(a.c + static_cast<size_t>(m0 + r) * a.ldc + static_cast<size_t>(ot) * kW);
          }
        }
      }
    });
  });
}


// The activation quantiser for W4A8 by 2D blocks (row_quant_rows' expressions, row_quant.hpp): item = 16 rows x one 128-k group, read as four 16 x 32 bf16
// 2D loads and written as four 8 x 64-byte 2D stores (rows past m: read as zero, stores clipped; their scales are written
// into the padded scale rows). The arithmetic is row_quant_rows' (s = amax / 127, rne, +-127), so the same bits.
inline void quant_2d16(const bf16 *src, int lda, int m, int k, int8_t *dst, float *scl, int lds, size_t id) SYCL_ESIMD_FUNCTION {
  const int gpr = k / kG;
  const int rb = static_cast<int>(id / gpr), g = static_cast<int>(id % gpr), r0 = rb * 16;
  if (r0 >= m) return;
  const uint32_t *s32 = reinterpret_cast<const uint32_t *>(src);
  const unsigned sw = static_cast<unsigned>(lda * 2 - 1), sh = static_cast<unsigned>(m - 1);
  esimd::simd<uint32_t, 256> blk[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) blk[c] = esimd::load_2d<uint32_t, 16, 16>(s32, sw, sh, sw, g * 64 + c * 16, r0);
  esimd::simd<uint32_t, 128> out[4];  // [half of the 128 bytes][8-row group]: 2D stores take <= 8 rows x 64 bytes
  esimd::simd<float, 16> sv(0.f);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    esimd::simd<float, kG> v;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      esimd::simd<uint32_t, 16> w = blk[c].template select<16, 1>(i * 16);
      esimd::simd<bf16, 32> b = w.template bit_cast_view<bf16>();
      v.template select<32, 1>(c * 32) = b;
    }
    const float amax = esimd::hmax<float>(esimd::abs(v));
    esimd::simd<int8_t, kG> qv(0);
    if (amax > 0.f) {
      const float inv = esimd::div_ieee(127.f, amax);
      sv[i] = esimd::div_ieee(amax, 127.f);
      const esimd::simd<float, kG> t = esimd::rnde<float, kG>(v * inv);
      qv = esimd::simd<int8_t, kG>(esimd::max(esimd::min(t, 127.f), -127.f));
    }
    esimd::simd<uint32_t, 32> qd = qv.template bit_cast_view<uint32_t>();
    out[i / 8].template select<16, 1>((i % 8) * 16) = qd.template select<16, 1>(0);
    out[2 + i / 8].template select<16, 1>((i % 8) * 16) = qd.template select<16, 1>(16);
  }
  uint32_t *d32 = reinterpret_cast<uint32_t *>(dst);
  const unsigned dw = static_cast<unsigned>(k - 1);
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) esimd::store_2d<uint32_t, 16, 8>(d32, dw, sh, dw, g * 32 + h * 16, r0 + rg * 8, out[h * 2 + rg]);
  esimd::block_store<float, 16>(scl + static_cast<size_t>(g) * lds + r0, sv);
}

// The oneDNN GLU's epilogue with down's quantiser in it. h = bf16(tab[bits(g)] x float(u)) on the bf16 [gate | up]
// rows of gu (onednn_pf.cpp's swiglu_bf16 expression), written to h, then quantised per (row, 128-k group) by quant_2d16's
// expressions (s = amax / 127, rne, +-127) into codes [m][nh] + scales group-major [nh / 128][lds]: down then reads them
// (A8State:hin) instead of re-reading h in its own quantiser launch. The same bits as swiglu_bf16 + quant_2d16. Item = 16
// rows x one 128-k group (rows past m: no h / code stores; their scales land in the padded scale rows).
class W4A8GluQ8;
sycl::event glu_q8(sycl::queue &q, int m, int nh, const bf16 *gu, const float *tab, bf16 *h, int ldh, int8_t *hq, float *hs, int lds) {
  const int gpr = nh / kG;
  const size_t items = static_cast<size_t>((m + 15) / 16) * gpr;
  return q.parallel_for<W4A8GluQ8>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t id = it.get_global_id(0);
    if (id >= items) return;
    const int rb = static_cast<int>(id / gpr), g = static_cast<int>(id % gpr), r0 = rb * 16;
    esimd::simd<float, 16> sv(0.f);
    for (int i = 0; i < 16; ++i) {
      const int r = r0 + i;
      if (r >= m) break;
      const bf16 *row = gu + static_cast<size_t>(r) * 2 * nh + static_cast<size_t>(g) * kG;
      esimd::simd<bf16, kG> hb;
#pragma unroll
      for (int c = 0; c < kG / 32; ++c) {
        esimd::simd<bf16, 32> gv = esimd::block_load<bf16, 32>(row + c * 32);
        const esimd::simd<bf16, 32> uv = esimd::block_load<bf16, 32>(row + nh + c * 32);
        const esimd::simd<uint16_t, 32> bits = gv.template bit_cast_view<uint16_t>().read();
        const esimd::simd<uint32_t, 32> off = esimd::simd<uint32_t, 32>(bits) * 4u;
        const esimd::simd<float, 32> t = esimd::gather<float, 32>(tab, off);
        const esimd::simd<float, 32> hv = t * esimd::simd<float, 32>(uv);
        hb.template select<32, 1>(c * 32) = esimd::simd<bf16, 32>(hv);
      }
      esimd::block_store<bf16, kG>(h + static_cast<size_t>(r) * ldh + static_cast<size_t>(g) * kG, hb);
      const esimd::simd<float, kG> v = hb;
      const float amax = esimd::hmax<float>(esimd::abs(v));
      esimd::simd<int8_t, kG> qv(0);
      if (amax > 0.f) {
        const float inv = esimd::div_ieee(127.f, amax);
        sv[i] = esimd::div_ieee(amax, 127.f);
        const esimd::simd<float, kG> t = esimd::rnde<float, kG>(v * inv);
        qv = esimd::simd<int8_t, kG>(esimd::max(esimd::min(t, 127.f), -127.f));
      }
      esimd::block_store<int8_t, kG>(hq + static_cast<size_t>(r) * nh + static_cast<size_t>(g) * kG, qv);
    }
    esimd::block_store<float, 16>(hs + static_cast<size_t>(g) * lds + r0, sv);
  });
}

// drop the hin entries whose h rows overlap [h, h + rows x ldh)
void hin_drop(const sycl::device &dev, const bf16 *h, int rows, int ldh) {
  std::lock_guard<std::mutex> lk(g_a8.mu);
  const auto it = g_a8.hin.find(dev);
  if (it == g_a8.hin.end()) return;
  auto &v = it->second;
  const bf16 *e = h + static_cast<size_t>(rows) * ldh;
  v.erase(std::remove_if(v.begin(), v.end(), [&](const A8State::In &x) { return x.src < e && h < x.src + static_cast<size_t>(x.rows) * x.lda; }), v.end());
  if (v.empty()) g_a8.hscr[dev].qused = g_a8.hscr[dev].sused = 0;
}

int g_a8_shape = -1;  // w4a16_pf_set_a8_shape (bench): -1 = by m
int g_a8_body = 0;  // w4a16_pf_set_a8_body: 0 = the pipelined SLM body (default), 1 = the SLM-free body, 2 = SLM-free on n >= kA8nMinN only,
                      // 3 = SLM-free up to kA8nSmallM rows, the SLM body above

// Stages of 4 groups for gate|up (GLU), 2 for the rest; default shape 32 x 2 R 8 from 97 rows, 8 x 4 R 16 below.
// Body 1 = the SLM-free body (launch_a8n: 16 x 2 up to 64 rows, 32 x 2 above) on every W4A8 call; the same outputs
// bit for bit. In the TP2 model, from 512 rows the split-sum producers send the peer's rows on the side queue and the
// consumers' own-row GEMMs run under that transfer, so the body is a per-setup choice (ModelOptions:prefill_a8_body,
// server --prefill-a8-body).
constexpr int kA8nMinN = 6144;
// Body 3 (qwen35's auto at TP2) = the SLM-free body on calls of at most 128 rows. There the SLM body's small shape
// (8 x 4 R 16, a 128-row work-group) has most row items idle at the stage barriers, and no split-sum sends run under
// 512 rows.
constexpr int kA8nSmallM = 128;
// The 24 x 4 HALF body (shape 5) where its 384-row work-groups pad the call at most 15 % more than the 32 x 2
// body's 256-row ones: 300 / 384 / 768 / 1,024 / 1,536 / 2,048 rows yes, 512 / 1,280 no.
inline bool a8_half_pays(int m) { return 20 * ((m + 383) / 384 * 384) < 23 * ((m + 255) / 256 * 256); }
sycl::event a8_shape(sycl::queue &q, const Args &A, const int8_t *dq, const float *ds, int lds, bool glu, int shape) {
  if (g_a8_shape < 0 && (g_a8_body == 1 || (g_a8_body == 2 && A.n >= kA8nMinN) || (g_a8_body == 3 && A.m <= kA8nSmallM))) {
    if (A.m <= 64) return glu ? launch_a8n<16, 2, true>(q, A, dq, ds, lds) : launch_a8n<16, 2, false>(q, A, dq, ds, lds);
    return glu ? launch_a8n<32, 2, true>(q, A, dq, ds, lds) : launch_a8n<32, 2, false>(q, A, dq, ds, lds);
  }
  if (glu) {
    switch (shape) {
      case 5: return launch_a8<24, 4, 16, true, 2, 1>(q, A, dq, ds, lds);  // 24 x 4 HALF, 2 groups per stage
      case 1: return launch_a8<8, 4, 16, true, 4>(q, A, dq, ds, lds);
      case 2: return launch_a8<16, 4, 16, true, 4>(q, A, dq, ds, lds);
      case 3: return launch_a8<32, 2, 16, true, 4>(q, A, dq, ds, lds);
      default: return launch_a8<32, 2, 8, true, 4>(q, A, dq, ds, lds);
    }
  }
  switch (shape) {
    case 5: return launch_a8<24, 4, 16, false, 4, 1>(q, A, dq, ds, lds);  // 24 x 4 HALF, 4 groups per stage
    case 1: return launch_a8<8, 4, 16, false, 2>(q, A, dq, ds, lds);
    case 2: return launch_a8<16, 4, 16, false, 2>(q, A, dq, ds, lds);
    case 3: return launch_a8<32, 2, 16, false, 2>(q, A, dq, ds, lds);
    default: return launch_a8<32, 2, 8, false, 2>(q, A, dq, ds, lds);
  }
}

// The oneDNN GLU with down's quantiser fused into its swiglu (glu_q8); the codes are registered for h's rows.
sycl::event onednn_glu_q8(sycl::queue &q, const Args &A, const int8_t *aq, const float *as, int lds, const float *as_rows = nullptr) {
  bf16 *gu = nullptr;
  const float *tab = nullptr;
  onednn_w4a8_pf(q, A.m, A.n, A.k, aq, as, lds, A.w, A.s, A.c, A.ldc, A.h, A.ldh, &gu, &tab, as_rows);
  const int nh = A.n / 2, hlds = a8_ld(A.m);
  const size_t qb = (static_cast<size_t>(A.m) * nh + 4095) / 4096 * 4096, sb = (static_cast<size_t>(nh / kG) * hlds + 63) / 64 * 64;
  int8_t *hq = nullptr;
  float *hs = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_a8.mu);
    auto &v = g_a8.hin[q.get_device()];
    A8State::HScr &hc = g_a8.hscr[q.get_device()];
    if (qb > hc.qcap || sb > hc.scap) {  // two calls' worth of the largest so far (the split MLP registers h in row ranges);
      q.wait();  // the first served-size call (the warm-up) sizes it
      if (hc.q) sycl::free(hc.q, q);
      if (hc.s) sycl::free(hc.s, q);
      hc.qcap = std::max(hc.qcap, 2 * qb), hc.scap = std::max(hc.scap, 2 * sb);
      hc.q = static_cast<int8_t *>(sycl::aligned_alloc_device(4096, hc.qcap, q));
      hc.s = static_cast<float *>(sycl::aligned_alloc_device(4096, hc.scap * 4, q));
      v.clear(), hc.qused = hc.sused = 0;
    }
    if (hc.qused + qb > hc.qcap || hc.sused + sb > hc.scap || static_cast<int>(v.size()) >= A8State::kHin) v.clear(), hc.qused = hc.sused = 0;
    hq = hc.q + hc.qused, hs = hc.s + hc.sused;
    hc.qused += qb, hc.sused += sb;
    v.push_back(A8State::In{A.h, A.ldh, A.m, nh, hlds, hq, hs, nullptr});
  }
  return glu_q8(q, A.m, nh, gu, tab, A.h, A.ldh, hq, hs, hlds);
}

sycl::event w4a8_pf(sycl::queue &q, const Args &A, bool glu) {
  A8Scr sc;
  A8State::In pre;
  std::vector<A8State::In> hin;
  {
    std::lock_guard<std::mutex> lk(g_a8.mu);
    const auto it = g_a8.scr.find(q.get_device());
    if (it == g_a8.scr.end()) throw std::logic_error("w4a16_gemm_pf: W4A8 without a scratch on this device (w4a16_pf_a8_reserve)");
    sc = it->second;
    const auto ip = g_a8.in.find(q.get_device());
    if (ip != g_a8.in.end()) pre = ip->second;
    const auto ih = g_a8.hin.find(q.get_device());
    if (!glu && ih != g_a8.hin.end()) hin = ih->second;
  }
  auto from = [&](const A8State::In &p) -> int {  // A's first row inside p's codes, or -1
    if (!p.src || A.a < p.src || A.lda != p.lda || A.k != p.k) return -1;
    const std::ptrdiff_t d = A.a - p.src;
    const int r0 = static_cast<int>(d / p.lda);
    return d % p.lda == 0 && r0 + A.m <= p.rows && r0 % 4 == 0 && (static_cast<size_t>(r0) * p.k) % 64 == 0 ? r0 : -1;
  };
  int r0 = from(pre);
  if (r0 < 0)
    for (const auto &p : hin)
      if ((r0 = from(p)) >= 0) {
        pre = p;
        break;
      }
  if (r0 >= 0) {  // the producer's codes
    int shape = g_a8_shape;
    if (shape < 0) shape = A.m <= 96 ? 1 : a8_half_pays(A.m) ? 5 : 0;
    const int8_t *pq = pre.q + static_cast<size_t>(r0) * pre.k;
    if (onednn_pf_takes(A.m, A.n)) {  // the wide matrices on oneDNN
      const float *sr = pre.s_rows ? pre.s_rows + static_cast<size_t>(r0) * (pre.k / kG) : nullptr;  // no transpose
      if (glu) return onednn_glu_q8(q, A, pq, pre.s + r0, pre.lds, sr);
      return onednn_w4a8_pf(q, A.m, A.n, A.k, pq, pre.s + r0, pre.lds, A.w, A.s, A.c, A.ldc, nullptr, A.ldh, nullptr, nullptr, sr);
    }
    return a8_shape(q, A, pq, pre.s + r0, pre.lds, glu, shape);
  }
  if (A.m > sc.max_m || A.k > sc.max_k) throw std::invalid_argument("w4a16_gemm_pf: W4A8 call larger than the reserved scratch");
  const int lds = a8_ld(A.m), gpr = A.k / kG;
  const size_t items = static_cast<size_t>((A.m + 15) / 16) * gpr;
  const bf16 *src = A.a;
  const int lda = A.lda, m = A.m, k = A.k;
  int8_t *dq = sc.q;
  float *ds = sc.s;
  q.parallel_for<W4A8Quant>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    quant_2d16(src, lda, m, k, dq, ds, lds, it.get_global_id(0));
  });
  if (onednn_pf_takes(A.m, A.n)) {
    if (glu) return onednn_glu_q8(q, A, dq, ds, lds);
    return onednn_w4a8_pf(q, A.m, A.n, A.k, dq, ds, lds, A.w, A.s, A.c, A.ldc, nullptr, A.ldh);
  }
  int shape = g_a8_shape;
  if (shape < 0) shape = A.m <= 96 ? 1 : a8_half_pays(A.m) ? 5 : 0;  // (pipelined body): 32 x 2 R 8 (24 x 4 HALF where it pays)
  return a8_shape(q, A, dq, ds, lds, glu, shape);
}

}  // namespace

void w4a16_pf_set_dequant(bool on) { g_dequant = on; }
bool w4a16_pf_dequant() { return g_dequant; }
void w4a16_pf_set_a8(bool on) { g_a8.on = on; }
bool w4a16_pf_a8() { return g_a8.on; }
void w4a16_pf_set_a8_shape(int shape) { g_a8_shape = shape; }
void w4a16_pf_set_a8_body(int body) { g_a8_body = body; }
int w4a16_pf_a8_body() { return g_a8_body; }
void w4a16_pf_a8_keep_clear() { g_a8.keep.clear(); }
void w4a16_pf_a8_keep(const uint8_t *w_packed) { g_a8.keep.insert(w_packed); }
void w4a16_pf_a8_input(sycl::queue &q, const bf16 *src, int lda, int rows, int k, const int8_t *codes, const float *scales, int lds, const float *scales_rows) {
  if (!src || !codes || !scales || rows < 1 || k % kG || lda < k || lds < rows || lds % 4) throw std::invalid_argument("w4a16_pf_a8_input");
  std::lock_guard<std::mutex> lk(g_a8.mu);
  g_a8.in[q.get_device()] = A8State::In{src, lda, rows, k, lds, codes, scales, scales_rows};
}
void w4a16_pf_a8_input_clear(sycl::queue &q) {
  std::lock_guard<std::mutex> lk(g_a8.mu);
  g_a8.in.erase(q.get_device());
  g_a8.hin.erase(q.get_device());
  g_a8.hscr[q.get_device()].qused = g_a8.hscr[q.get_device()].sused = 0;
}
void w4a16_pf_a8_reserve(sycl::queue &q, int max_m, int max_k) {
  std::lock_guard<std::mutex> lk(g_a8.mu);
  A8Scr &sc = g_a8.scr[q.get_device()];
  if (sc.max_m >= max_m && sc.max_k >= max_k) return;
  if (sc.q) sycl::free(sc.q, q);
  if (sc.s) sycl::free(sc.s, q);
  max_m = std::max(max_m, sc.max_m), max_k = std::max(max_k, sc.max_k);
  sc.q = static_cast<int8_t *>(sycl::aligned_alloc_device(4096, std::max<size_t>(4096, static_cast<size_t>(max_m) * max_k), q));
  const size_t ns = static_cast<size_t>(max_k / kG + 1) * a8_ld(max_m);
  sc.s = static_cast<float *>(sycl::aligned_alloc_device(4096, std::max<size_t>(4096, ns * 4), q));
  q.memset(sc.s, 0, ns * 4).wait();  // the pad rows' scales read by the last row block stay finite
  sc.max_m = max_m, sc.max_k = max_k;
}

// The W4A8 decode registry and scratch (w4a16_gemm.hpp): here, in the shared xl library, as the
// zero-point registry below.
namespace {
struct A8Rows {
  std::mutex mu;
  std::unordered_map<const uint8_t *, int> mats;
  struct Scr {
    int8_t *codes = nullptr;
    float *scales = nullptr;
    int max_k = 0;
  };
  std::unordered_map<sycl::device, Scr> scr;
};
A8Rows g_a8r;
std::atomic<bool> g_a8r_on{false};
}  // namespace

void w4a8_rows_register(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lk(g_a8r.mu);
  g_a8r.mats[w_packed] = 1;
}
void w4a8_rows_unregister(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lk(g_a8r.mu);
  g_a8r.mats.erase(w_packed);
}
void w4a8_rows_set(bool on) { g_a8r_on.store(on, std::memory_order_relaxed); }
bool w4a8_rows_on() { return g_a8r_on.load(std::memory_order_relaxed); }
bool w4a8_rows_of(const uint8_t *w_packed) {
  if (!g_a8r_on.load(std::memory_order_relaxed)) return false;
  std::lock_guard<std::mutex> lk(g_a8r.mu);
  return g_a8r.mats.find(w_packed) != g_a8r.mats.end();
}
void w4a8_rows_reserve(sycl::queue &q, int max_k) {
  constexpr int kRows = 32;
  std::lock_guard<std::mutex> lk(g_a8r.mu);
  A8Rows::Scr &sc = g_a8r.scr[q.get_device()];
  if (sc.max_k >= max_k) return;
  if (sc.codes) sycl::free(sc.codes, q);
  if (sc.scales) sycl::free(sc.scales, q);
  sc.codes = static_cast<int8_t *>(sycl::aligned_alloc_device(4096, std::max<size_t>(4096, static_cast<size_t>(kRows) * max_k), q));
  const size_t ns = static_cast<size_t>(max_k / kG + 1) * kRows;
  sc.scales = static_cast<float *>(sycl::aligned_alloc_device(4096, std::max<size_t>(4096, ns * 4), q));
  q.memset(sc.codes, 0, static_cast<size_t>(kRows) * max_k).wait();
  q.memset(sc.scales, 0, ns * 4).wait();
  sc.max_k = max_k;
}
void w4a8_rows_scratch(sycl::queue &q, int k, int8_t **codes, float **scales) {
  std::lock_guard<std::mutex> lk(g_a8r.mu);
  const auto it = g_a8r.scr.find(q.get_device());
  if (it == g_a8r.scr.end() || it->second.max_k < k) throw std::logic_error("w4a8 rows: no scratch for k on this device (w4a8_rows_reserve)");
  *codes = it->second.codes, *scales = it->second.scales;
}

// The zero-point registry (w4a16_gemm.hpp): packed-weight pointer -> its zero-point nibbles. Here, in the shared xl
// library, which every GEMM caller links (the decode GEMMs in the static kernels library query it too).
namespace {
std::mutex g_zp_mu;
std::unordered_map<const uint8_t *, const uint8_t *> g_zp;
}  // namespace

void w4_set_zero_points(const uint8_t *w_packed, const uint8_t *zeros) {
  std::lock_guard<std::mutex> lock(g_zp_mu);
  g_zp[w_packed] = zeros;
}

void w4_clear_zero_points(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lock(g_zp_mu);
  g_zp.erase(w_packed);
}

const uint8_t *w4_zero_points(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lock(g_zp_mu);
  if (g_zp.empty()) return nullptr;
  const auto it = g_zp.find(w_packed);
  return it == g_zp.end() ? nullptr : it->second;
}

// The affine registry (w4a16_gemm.hpp, int4 g64: arch flint): packed-weight pointer -> its bf16 group biases.
namespace {
std::mutex g_aff_mu;
std::unordered_map<const uint8_t *, const uint16_t *> g_aff;
}  // namespace

void w4_set_affine(const uint8_t *w_packed, const uint16_t *biases) {
  std::lock_guard<std::mutex> lock(g_aff_mu);
  g_aff[w_packed] = biases;
}

void w4_clear_affine(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lock(g_aff_mu);
  g_aff.erase(w_packed);
}

const uint16_t *w4_affine(const uint8_t *w_packed) {
  std::lock_guard<std::mutex> lock(g_aff_mu);
  if (g_aff.empty()) return nullptr;
  const auto it = g_aff.find(w_packed);
  return it == g_aff.end() ? nullptr : it->second;
}

sycl::event w4aff_rowsum(sycl::queue &q, const bf16 *a, int lda, int m, int k, float *asum, int lds) {
  if (m < 1 || k % kW4AffGroup || lda < k || (lda * 2) % 16 || lds < (m + 15) / 16 * 16) throw std::invalid_argument("w4aff_rowsum: shape");
  return aff_rowsum(q, a, lda, m, k, asum, lds);
}

void w4a16_pf_aff_reserve(sycl::queue &q, int max_m, int max_k) {
  std::lock_guard<std::mutex> lk(g_affs_mu);
  AffScr &sc = g_affs[q.get_device()];
  if (sc.max_m >= max_m && sc.max_k >= max_k) return;
  if (sc.s) sycl::free(sc.s, q);
  max_m = std::max(max_m, sc.max_m), max_k = std::max(max_k, sc.max_k);
  const size_t n = static_cast<size_t>(max_k / kW4AffGroup) * aff_ld(max_m);
  sc.s = static_cast<float *>(sycl::aligned_alloc_device(4096, std::max<size_t>(4096, n * 4), q));
  q.memset(sc.s, 0, n * 4).wait();
  sc.max_m = max_m, sc.max_k = max_k;
}

sycl::event w4aff_gemm_pf(sycl::queue &q, int m, const bf16 *a, int lda, const uint8_t *w, const uint16_t *scales, const uint16_t *biases, float *c, int ldc,
                          int n, int k, int shape) {
  if (m < 1 || n % kW || k % kG || lda < k || ldc < n || (lda * 2) % 16 || lda * 2 < 64 || !biases)
    throw std::invalid_argument("w4aff_gemm_pf: shape (n % 16, k % 128, lda >= k with a 16-byte pitch of >= 64 bytes), biases");
  Args A{a, lda, m, n, k, w, reinterpret_cast<const sycl::half *>(scales), c, ldc};
  A.b = biases;
  return launch_aff<false>(q, A, shape);
}

sycl::event w4aff_gemm_pf_swiglu(sycl::queue &q, int m, const bf16 *a, int lda, const uint8_t *w, const uint16_t *scales, const uint16_t *biases, bf16 *h,
                                 int ldh, int n_half, int k, int shape) {
  if (m < 1 || n_half % kW || k % kG || lda < k || ldh < n_half || (lda * 2) % 16 || lda * 2 < 64 || !biases)
    throw std::invalid_argument("w4aff_gemm_pf_swiglu: shape (n_half % 16, k % 128, lda >= k with a 16-byte pitch of >= 64 bytes), biases");
  Args A{a, lda, m, 2 * n_half, k, w, reinterpret_cast<const sycl::half *>(scales), nullptr, 0};
  A.h = h;
  A.ldh = ldh;
  A.b = biases;
  return launch_aff<true>(q, A, shape);
}

namespace {
// Calls of at most this many rows run on the decode GEMMs (w4a16_pf_set_short_rows)
std::atomic<int> g_short_rows{0};
// the decode GEMMs live in libxenithium_kernels (which links this library): it registers them (w4a16_pf_short_hooks)
W4ShortGemm g_short_gemm = nullptr;
W4ShortGlu g_short_glu = nullptr;
W4SiluTable g_short_tab = nullptr;
// the decode swiglu's s(g) table, one per (device, context) of the queues that ask
struct SiluTab {
  sycl::context ctx;
  sycl::device dev;
  float *tab = nullptr;
};
std::mutex g_silu_mu;
std::vector<SiluTab> g_silu;
const float *silu_table_for(sycl::queue &q) {
  std::lock_guard<std::mutex> lk(g_silu_mu);
  for (const auto &t : g_silu)
    if (t.ctx == q.get_context() && t.dev == q.get_device()) return t.tab;
  float *tab = sycl::malloc_device<float>(65536, q);  // kernels:kSiluTabSize (gemm/mlp.hpp)
  g_short_tab(q, tab).wait();
  g_silu.push_back(SiluTab{q.get_context(), q.get_device(), tab});
  return tab;
}
}  // namespace

void w4a16_pf_set_short_rows(int rows) {
  if (rows < 0 || rows > kStepRows32) throw std::invalid_argument("w4a16_pf_set_short_rows: 0..32 rows");
  g_short_rows = rows;
}
int w4a16_pf_short_rows() { return g_short_gemm ? g_short_rows.load() : 0; }
void w4a16_pf_short_hooks(W4ShortGemm gemm, W4ShortGlu glu, W4SiluTable tab) {
  g_short_gemm = gemm;
  g_short_glu = glu;
  g_short_tab = tab;
}

sycl::event w4a16_gemm_pf(sycl::queue &q, int m, const bf16 *a, int lda, const uint8_t *w_packed, const sycl::half *scales, float *c, int ldc, int n,
                          int k, int shape) {
  if (const uint16_t *bs = w4_affine(w_packed)) return w4aff_gemm_pf(q, m, a, lda, w_packed, reinterpret_cast<const uint16_t *>(scales), bs, c, ldc, n, k, shape);
  if (m < 1 || n % kW || k % kG || lda < k || ldc < n || (lda * 2) % 16 || lda * 2 < 64)
    throw std::invalid_argument("w4a16_gemm_pf: shape (n % 16, k % 128, lda >= k with a 16-byte pitch of >= 64 bytes)");
  // a short call on the decode GEMM (W4A16 numerics, decode's k split; the call's A8 codes unused)
  if (g_short_gemm && m <= g_short_rows.load(std::memory_order_relaxed) && lda == k && ldc == n && !w4_zero_points(w_packed))
    return g_short_gemm(q, m, a, w_packed, scales, c, n, k, 0);
  Args A{a, lda, m, n, k, w_packed, scales, c, ldc};
  if (const uint8_t *z = w4_zero_points(w_packed)) {  // int4 g32 + zero points: the dequantised mode always
    A.z = z;
    return launch_dq<false, true>(q, A, shape >= 12 && shape <= 15 ? shape : -1);
  }
  if (g_a8.on && !g_a8.keep.count(w_packed)) return w4a8_pf(q, A, false);
  if (g_dequant) return launch_dq<false>(q, A, shape);
  if (shape < 0) shape = pf_shape(m, n);
  switch (shape) {
    case 1: return launch<16, 1, 16, false>(q, A);
    case 2: return launch<8, 4, 16, false>(q, A);
    case 3: return launch<32, 1, 16, false>(q, A);
    case 4: return launch<16, 4, 16, false>(q, A);
    case 7: return launch<32, 2, 8, false>(q, A);  // 256 rows x 64 columns per work-group (8 x 2 items)
    case 11: return launch<32, 2, 16, false>(q, A);  // 512 x 32
    default: return launch<16, 2, 16, false>(q, A);
  }
}

sycl::event w4a16_gemm_pf_swiglu(sycl::queue &q, int m, const bf16 *a, int lda, const uint8_t *w_packed, const sycl::half *scales, bf16 *h, int ldh,
                                 int n_half, int k, int shape) {
  if (h && m > 0) hin_drop(q.get_device(), h, m, ldh);  // h's rows are rewritten; codes registered for them die
  if (m < 1 || n_half % kW || k % kG || lda < k || ldh < n_half || (lda * 2) % 16 || lda * 2 < 64)
    throw std::invalid_argument("w4a16_gemm_pf_swiglu: shape (n_half % 16, k % 128, lda >= k with a 16-byte pitch of >= 64 bytes)");
  if (const uint16_t *bs = w4_affine(w_packed))
    return w4aff_gemm_pf_swiglu(q, m, a, lda, w_packed, reinterpret_cast<const uint16_t *>(scales), bs, h, ldh, n_half, k, shape);
  if (g_short_glu && m <= g_short_rows.load(std::memory_order_relaxed) && lda == k && ldh == n_half && !w4_zero_points(w_packed))
    return g_short_glu(q, m, a, w_packed, scales, silu_table_for(q), h, n_half, k, 0);
  Args A{a, lda, m, 2 * n_half, k, w_packed, scales, nullptr, 0};
  A.h = h;
  A.ldh = ldh;
  if (const uint8_t *z = w4_zero_points(w_packed)) {  // int4 g32 + zero points: the dequantised mode always
    A.z = z;
    return launch_dq<true, true>(q, A, shape >= 12 && shape <= 15 ? shape : -1);
  }
  if (g_a8.on && !g_a8.keep.count(w_packed)) return w4a8_pf(q, A, true);
  if (g_dequant) return launch_dq<true>(q, A, shape);
  if (shape < 0) shape = pf_shape(m, 2 * n_half);  // by the whole [gate | up] width
  switch (shape) {
    case 2: return launch<8, 4, 16, true>(q, A);
    case 4: return launch<16, 4, 16, true>(q, A);
    case 7: return launch<32, 2, 8, true>(q, A);
    default: return launch<16, 2, 16, true>(q, A);
  }
}

}  // namespace xenithium:kernels
