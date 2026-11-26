// Xenithium · moe_rows' W4A16 gate_up and down. Their own TU because they need 256 GRF (gate_up
// spills at 128): built in xenithium_kernels_xl (-doubleGRF). gate_up is one item for both projections, down two tiles
// per item.
#include <functional>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/moe/moe_rows.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = moe_bf16;

constexpr int kW = 16;
constexpr int kGroup = 128;
constexpr int kSeg = kMoeSegRows;

class MrGateUpBoth;
template <int PK>
class MrDown2;

template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}

// ---- 4: gate_up -> h --------------------------------------------------------------------------------
// One item per (segment, tile) computing both the gate and the up tile, so each x row is loaded
// once per group for both.
sycl::event gate_up_both(sycl::queue &q, const MoeShape &s, int max_segs, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_local(), tiles_i = I / kW, G = H / kGroup;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n, *row = sc.row;
  bf16 *hbuf = sc.h;
  const size_t items = static_cast<size_t>(max_segs) * tiles_i;
  constexpr int kWg = 4;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrGateUpBoth>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / tiles_i, j = gid % tiles_i;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
      int rr[kSeg];
#pragma unroll
      for (int r = 0; r < kSeg; ++r) rr[r] = row[p0 + (r < n ? r : n - 1)];
      esimd::simd<float, kSeg * kW> ag(0.f), au(0.f);
      const bool two = n > 8;
      for (int g = 0; g < G; ++g) {
        const auto rg = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + j) * kW4BlockBytes);
        const auto ru = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + tiles_i + j) * kW4BlockBytes);
        const esimd::simd<float, kW> sg = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + j * kW);
        const esimd::simd<float, kW> su = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + (tiles_i + j) * kW);
        esimd::simd<bf16, 8 * kGroup> ah0, ah1;
#pragma unroll
        for (int r = 0; r < 8; ++r) ah0.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[r]) * H + g * kGroup);
        if (two)
#pragma unroll
          for (int r = 0; r < 8; ++r) ah1.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[8 + r]) * H + g * kGroup);
        esimd::simd<float, 8 * kW> gg0(0.f), gu0(0.f), gg1(0.f), gu1(0.f);
#pragma unroll
        for (int jc = 0; jc < w4::kCalls; ++jc) {
          const esimd::simd<bf16, 2 * w4::kCallBytes> bg = w4::call_codes(rg, jc), bu = w4::call_codes(ru, jc);
          esimd::simd<bf16, 8 * w4::kKPerCall> at;
#pragma unroll
          for (int r = 0; r < 8; ++r) at.template select<w4::kKPerCall, 1>(r * w4::kKPerCall) = ah0.template select<w4::kKPerCall, 1>(r * kGroup + jc * w4::kKPerCall);
          gg0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg0, bg, at);
          gu0 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu0, bu, at);
          if (two) {
#pragma unroll
            for (int r = 0; r < 8; ++r) at.template select<w4::kKPerCall, 1>(r * w4::kKPerCall) = ah1.template select<w4::kKPerCall, 1>(r * kGroup + jc * w4::kKPerCall);
            gg1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gg1, bg, at);
            gu1 = xmx::dpas<8, 8, float, float, bf16, bf16>(gu1, bu, at);
          }
        }
        esimd::simd<float, 8 * kW> t = ag.template select<8 * kW, 1>(0);
        w4::apply_scale<8>(t, gg0, sg);
        ag.template select<8 * kW, 1>(0) = t;
        t = au.template select<8 * kW, 1>(0);
        w4::apply_scale<8>(t, gu0, su);
        au.template select<8 * kW, 1>(0) = t;
        if (two) {
          t = ag.template select<8 * kW, 1>(8 * kW);
          w4::apply_scale<8>(t, gg1, sg);
          ag.template select<8 * kW, 1>(8 * kW) = t;
          t = au.template select<8 * kW, 1>(8 * kW);
          w4::apply_scale<8>(t, gu1, su);
          au.template select<8 * kW, 1>(8 * kW) = t;
        }
      }
      const esimd::simd<float, kSeg * kW> gb = esimd::simd<bf16, kSeg * kW>(ag);
      const esimd::simd<float, kSeg * kW> ub = esimd::simd<bf16, kSeg * kW>(au);
      esimd::simd<bf16, kSeg * kW> hv = gb * sigmoid_v<kSeg * kW>(gb) * ub;
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        const esimd::simd<bf16, kW> v = hv.template select<kW, 1>(r * kW);
        v.copy_to(hbuf + static_cast<size_t>(p0 + r) * I + j * kW);
      }
    });
  });
}

// moe_rows' W4A16 down with one item per (segment, two adjacent 16-column tiles): each h row is
// loaded once per 64-k half group for both tiles (the one-tile kernel in moe_rows.cpp loads it per tile), the second 8-row
// half skipped on <= 8-row segments. Each tile's DPAS chain is the one-tile kernel's: the same bits.
// PK: the TP blob's down piece (moe_tp_piece): 64-k halves, or 32-k quarters at TP4's 160 columns per card.
template <int PK>
sycl::event down2(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int kPB = PK * kW4BlockBytes / kGroup, PG = kGroup / PK, kCB = w4::kCallBytes, kKC = w4::kKPerCall;
  const int H = s.hidden, I = s.inter_local(), tiles = H / kW, pairs = tiles / 2, NP = I / PK;
  const int pg0 = s.card * NP;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n;
  const bf16 *hbuf = sc.h;
  bf16 *y = sc.y;
  constexpr int kWg = 8;
  const size_t items = static_cast<size_t>(max_segs) * pairs;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDown2<PK>>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / pairs, t0 = (gid % pairs) * 2;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      esimd::simd<float, kSeg * kW> acc0(0.f), acc1(0.f);
      const int halves = n > 8 ? 2 : 1;
      for (int g = 0; g < NP; ++g) {
        esimd::simd<uint8_t, kPB> raw0 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0) * kPB);
        esimd::simd<uint8_t, kPB> raw1 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t0 + 1) * kPB);
        const int srow = (pg0 + g) / PG - pg0 / PG;
        const esimd::simd<float, kW> sc0 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t0 * kW);
        const esimd::simd<float, kW> sc1 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + (t0 + 1) * kW);
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          if (hb >= halves) break;
          esimd::simd<bf16, 8 * PK> a;
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int rr = hb * 8 + r < n ? hb * 8 + r : n - 1;
            a.template select<PK, 1>(r * PK) = esimd::block_load<bf16, PK>(hbuf + static_cast<size_t>(p0 + rr) * I + g * PK);
          }
          esimd::simd<float, 8 * kW> s0(0.f), s1(0.f);
#pragma unroll
          for (int jc = 0; jc < PK / kKC; ++jc) {
            esimd::simd<bf16, 8 * kKC> at;
#pragma unroll
            for (int r = 0; r < 8; ++r) at.template select<kKC, 1>(r * kKC) = a.template select<kKC, 1>(r * PK + jc * kKC);
#pragma unroll
            for (int tt = 0; tt < 2; ++tt) {
              esimd::simd<uint8_t, kCB> bytes;
              if (tt) bytes = raw1.template select<kCB, 1>(jc * kCB);
              else bytes = raw0.template select<kCB, 1>(jc * kCB);
              const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
              const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
              esimd::simd<bf16, 2 * kCB> b;
              b.template select<kCB, 2>(0) = esimd::simd<float, kCB>(lo);
              b.template select<kCB, 2>(1) = esimd::simd<float, kCB>(hi);
              if (tt) s1 = xmx::dpas<8, 8, float, float, bf16, bf16>(s1, b, at);
              else s0 = xmx::dpas<8, 8, float, float, bf16, bf16>(s0, b, at);
            }
          }
          esimd::simd<float, 8 * kW> ac = acc0.template select<8 * kW, 1>(hb * 8 * kW);
          w4::apply_scale<8>(ac, s0, sc0);
          acc0.template select<8 * kW, 1>(hb * 8 * kW) = ac;
          ac = acc1.template select<8 * kW, 1>(hb * 8 * kW);
          w4::apply_scale<8>(ac, s1, sc1);
          acc1.template select<8 * kW, 1>(hb * 8 * kW) = ac;
        }
      }
      constexpr auto hint = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<32>};
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        esimd::simd<bf16, 2 * kW> v;
        v.template select<kW, 1>(0) = acc0.template select<kW, 1>(r * kW);
        v.template select<kW, 1>(kW) = acc1.template select<kW, 1>(r * kW);
        esimd::block_store<bf16, 2 * kW>(y + static_cast<size_t>(p0 + r) * H + t0 * kW, v, hint);  // y in bf16
      }
    });
  });
}


// ---- a work-group per block of <= kMoeBlkRows list rows of one expert ------------------------------------------------
//
// widen: one DPAS call's 128 code bytes -> the bf16 VNNI B operand (256 values; dword i = byte i's low / high nibble,
// an adjacent k pair of one column). It is most of these kernels' instructions, so it runs on 32-bit lanes with bfn
// instead of int -> float -> bf16 conversions and the strided moves of the odd halves, in 16-lane pieces (a whole
// call's 32-bit temporaries would hold 8 GRF each):
//  BIAS false: q - 8, exact. f32 bits 0x4B000000 | q are 2^23 + q, so minus 2^23 + 8 leaves q - 8 (the high nibble,
//              read as 16 q, takes one mad by 1/16). A small integer's bf16 bits are its f32 bits >> 16, so the two
//              halves join by shift and bfn. The same values as w4::call_codes, hence the same bits.
//  BIAS true:  128 + q: the bf16 bits 0x4300 | q, a whole dword by or, shl and one bfn. The -8 leaves the operand: the
//              chain starts from -136 x the row's sum of A over the chain's k (MoeShape:w4_bias).
template <bool BIAS>
inline esimd::simd<bf16, 2 * w4::kCallBytes> widen_call(esimd::simd<uint8_t, w4::kCallBytes> bytes) SYCL_ESIMD_FUNCTION {
  using esimd::bfn_t;
  constexpr int N = w4::kCallBytes;
  esimd::simd<uint32_t, N> d;
#pragma unroll
  for (int c = 0; c < N / 16; ++c) {
    const esimd::simd<uint32_t, 16> u = bytes.template select<16, 1>(c * 16);
    if constexpr (!BIAS) {
      const esimd::simd<uint32_t, 16> two23(0x4B000000u);
      esimd::simd<uint32_t, 16> lb = esimd::bfn<(bfn_t::x & bfn_t::y) | bfn_t::z>(u, esimd::simd<uint32_t, 16>(0x0Fu), two23);
      esimd::simd<uint32_t, 16> hb = esimd::bfn<(bfn_t::x & bfn_t::y) | bfn_t::z>(u, esimd::simd<uint32_t, 16>(0xF0u), two23);
      esimd::simd<float, 16> lo = lb.template bit_cast_view<float>() - 8388616.f;  // 2^23 + 8
      esimd::simd<float, 16> hi = hb.template bit_cast_view<float>() * 0.0625f - 524296.f;  // (2^23 + 16 q) / 16 - (2^19 + 8)
      const esimd::simd<uint32_t, 16> lo16 = lo.template bit_cast_view<uint32_t>() >> 16;
      d.template select<16, 1>(c * 16) =
          esimd::bfn<(bfn_t::x & bfn_t::z) | bfn_t::y>(hi.template bit_cast_view<uint32_t>().read(), lo16, esimd::simd<uint32_t, 16>(0xFFFF0000u));
    } else {
      // ((u | K) & M) | ((u << 12) & ~M): the low nibble at bits 0-3, the high one at 16-19, K = two bf16 128.0
      d.template select<16, 1>(c * 16) = esimd::bfn<(bfn_t::x & bfn_t::z) | (bfn_t::y & ~bfn_t::z)>(
          esimd::simd<uint32_t, 16>(u | 0x43004300u), esimd::simd<uint32_t, 16>(u << 12), esimd::simd<uint32_t, 16>(0xFFF0FF0Fu));
    }
  }
  return d.template bit_cast_view<bf16>();
}
inline constexpr float kW4Bias = -(128.f + kW4Zero);  // BIAS: the operand is (q - 8) + 136

// gate_up: TI threads (one per 16-column tile pair of this card's gate and up columns). Per KG 128-k groups the block's
// x rows are staged once in SLM (double-buffered, so one barrier per KG groups) as the DPAS A tiles [call][8-row
// block][8][16]: one call's tiles of every block load in one message. Each thread loads its two int4 tiles per group,
// widens each call's codes once and runs NRB 8-row blocks x (gate, up) independent DPAS chains. NRB (the block's 8-row
// blocks) is a compile-time count, one body per count, so no branch splits a group's DPAS and the compiler overlaps
// loads, widens and DPAS of neighbouring calls. Each (row, column) keeps gate_up_both's chain - per group the eight
// dpas<8, 8> calls in k order, w4:apply_scale with the group's fp16 scale, bf16 g and u, the same swiglu - so h has the
// same bits (BIAS false). BIAS: the stager also writes each row's group sum beside the tiles, the chain start.
inline constexpr int kGuABytes = kMoeBlkRows * kGroup * 2;

template <bool BIAS>
inline constexpr int kGuSlot = kGuABytes + (BIAS ? kMoeBlkRows * 4 : 0);  // one group's A tiles (+ the rows' sums)

template <int TI, int NRB, bool BIAS, int KG>
inline void gate_up_blk_body(int j, int n, int p0, int H, int I, int G, const uint8_t *codes, const sycl::half *scales, const int32_t *row,
                             const bf16 *x, bf16 *hbuf) SYCL_ESIMD_FUNCTION {
  constexpr int R = NRB * 8, kMine = (R + TI - 1) / TI, kTile = 8 * w4::kKPerCall, kSlot = kGuSlot<BIAS>;
  int rr[kMine];  // this thread's staged rows r = j, j + TI... below R (past n: the last row)
#pragma unroll
  for (int i = 0; i < kMine; ++i) {
    const int r = j + i * TI;
    rr[i] = r < R ? row[p0 + (r < n ? r : n - 1)] : -1;
  }
  auto stage = [&](int g, int slot) SYCL_ESIMD_FUNCTION {
#pragma unroll
    for (int i = 0; i < kMine; ++i) {
      const int r = j + i * TI;
      if (rr[i] < 0) break;
      esimd::simd<bf16, kGroup> a = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(rr[i]) * H + g * kGroup);
      const int rb = r / 8, rl = r % 8;
#pragma unroll
      for (int jc = 0; jc < w4::kCalls; ++jc)
        esimd::slm_block_store<bf16, w4::kKPerCall>(static_cast<uint32_t>(slot * kSlot + (((jc * NRB + rb) * 8 + rl) * w4::kKPerCall) * 2),
                                                     a.template select<w4::kKPerCall, 1>(jc * w4::kKPerCall));
      if constexpr (BIAS)
        esimd::slm_scalar_store<float>(static_cast<uint32_t>(slot * kSlot + kGuABytes + r * 4), esimd::reduce<float>(esimd::simd<float, kGroup>(a), std::plus<>()));
    }
  };
  esimd::simd<float, R * kW> ag(0.f), au(0.f);
#pragma unroll
  for (int gi = 0; gi < KG; ++gi) stage(gi, gi);
  esimd::barrier();
  for (int g = 0; g < G; ++g) {
    if (g % KG == 0 && g + KG < G)  // the next KG groups into the other half of the slots
#pragma unroll
      for (int gi = 0; gi < KG; ++gi) stage(g + KG + gi, ((g / KG + 1) & 1) * KG + gi);
    auto rg = w4::load_block(codes + (static_cast<size_t>(g) * 2 * TI + j) * kW4BlockBytes);
    auto ru = w4::load_block(codes + (static_cast<size_t>(g) * 2 * TI + TI + j) * kW4BlockBytes);
    const esimd::simd<float, kW> sg = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + j * kW);
    const esimd::simd<float, kW> su = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + (TI + j) * kW);
    const uint32_t base = static_cast<uint32_t>((((g / KG) & 1) * KG + g % KG) * kSlot);
    esimd::simd<float, R * kW> pg(0.f), pu(0.f);
    if constexpr (BIAS) {
#pragma unroll
      for (int rb = 0; rb < NRB; ++rb) {
        const esimd::simd<float, 8> sb = esimd::slm_block_load<float, 8>(base + kGuABytes + rb * 32) * kW4Bias;
#pragma unroll
        for (int r = 0; r < 8; ++r) pg.template select<kW, 1>((rb * 8 + r) * kW) = sb[r];
      }
      pu = pg;
    }
#pragma unroll
    for (int jc = 0; jc < w4::kCalls; ++jc) {
      const esimd::simd<bf16, 2 * w4::kCallBytes> bg = widen_call<BIAS>(rg.template select<w4::kCallBytes, 1>(jc * w4::kCallBytes));
      const esimd::simd<bf16, 2 * w4::kCallBytes> bu = widen_call<BIAS>(ru.template select<w4::kCallBytes, 1>(jc * w4::kCallBytes));
      esimd::simd<bf16, NRB * kTile> at = esimd::slm_block_load<bf16, NRB * kTile>(base + static_cast<uint32_t>(jc * NRB * kTile * 2));
#pragma unroll
      for (int rb = 0; rb < NRB; ++rb) {
        esimd::simd<float, 8 * kW> c = pg.template select<8 * kW, 1>(rb * 8 * kW);
        c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, bg, at.template select<kTile, 1>(rb * kTile).read());
        pg.template select<8 * kW, 1>(rb * 8 * kW) = c;
        esimd::simd<float, 8 * kW> d = pu.template select<8 * kW, 1>(rb * 8 * kW);
        d = xmx::dpas<8, 8, float, float, bf16, bf16>(d, bu, at.template select<kTile, 1>(rb * kTile).read());
        pu.template select<8 * kW, 1>(rb * 8 * kW) = d;
      }
    }
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb) {
      esimd::simd<float, 8 * kW> t = ag.template select<8 * kW, 1>(rb * 8 * kW);
      w4::apply_scale<8>(t, pg.template select<8 * kW, 1>(rb * 8 * kW), sg);
      ag.template select<8 * kW, 1>(rb * 8 * kW) = t;
      t = au.template select<8 * kW, 1>(rb * 8 * kW);
      w4::apply_scale<8>(t, pu.template select<8 * kW, 1>(rb * 8 * kW), su);
      au.template select<8 * kW, 1>(rb * 8 * kW) = t;
    }
    if ((g + 1) % KG == 0) esimd::barrier();  // these KG groups' SLM reads are done and the next KG are staged
  }
#pragma unroll
  for (int rb = 0; rb < NRB; ++rb) {
    const esimd::simd<float, 8 * kW> gb = esimd::simd<bf16, 8 * kW>(ag.template select<8 * kW, 1>(rb * 8 * kW));
    const esimd::simd<float, 8 * kW> ub = esimd::simd<bf16, 8 * kW>(au.template select<8 * kW, 1>(rb * 8 * kW));
    esimd::simd<bf16, 8 * kW> hv = gb * sigmoid_v<8 * kW>(gb) * ub;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (rb * 8 + r >= n) break;
      const esimd::simd<bf16, kW> v = hv.template select<kW, 1>(r * kW);
      v.copy_to(hbuf + static_cast<size_t>(p0 + rb * 8 + r) * I + j * kW);
    }
  }
}

template <int TI, bool BIAS, int KG>
class MrGateUpBlkT;
template <int TI, bool BIAS, int KG>
sycl::event gate_up_blk(sycl::queue &q, const MoeShape &s, int max_blks, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_local(), G = H / kGroup;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *blk = sc.blk, *blk_n = sc.blk_n, *row = sc.row;
  bf16 *hbuf = sc.h;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrGateUpBlkT<TI, BIAS, KG>>(sycl::nd_range<1>(static_cast<size_t>(max_blks) * TI, TI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<2 * KG * kGuSlot<BIAS>>();
      const int b = static_cast<int>(it.get_group(0)), j = static_cast<int>(it.get_local_id(0));
      if (b >= blk_n[0]) return;  // the whole work-group
      const esimd::simd<int32_t, 4> bp = esimd::block_load<int32_t, 4>(blk + static_cast<size_t>(b) * 4);
      const int e = bp[0], p0 = bp[1], n = bp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
      switch ((n + 7) / 8) {  // uniform over the work-group
        case 1: gate_up_blk_body<TI, 1, BIAS, KG>(j, n, p0, H, I, G, codes, scales, row, x, hbuf); break;
        case 2: gate_up_blk_body<TI, 2, BIAS, KG>(j, n, p0, H, I, G, codes, scales, row, x, hbuf); break;
        case 3: gate_up_blk_body<TI, 3, BIAS, KG>(j, n, p0, H, I, G, codes, scales, row, x, hbuf); break;
        default: gate_up_blk_body<TI, 4, BIAS, KG>(j, n, p0, H, I, G, codes, scales, row, x, hbuf); break;
      }
    });
  });
}

// Two groups per barrier when the groups pair up (four lost: 64 KB of SLM per work-group cost cores' occupancy).
template <bool BIAS>
sycl::event gate_up_blk_ti(sycl::queue &q, const MoeShape &s, int max_blks, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  const bool two = (s.hidden / kGroup) % 2 == 0;
  switch (s.inter_local() / kW) {
    case 8: return two ? gate_up_blk<8, BIAS, 2>(q, s, max_blks, x, w, sc) : gate_up_blk<8, BIAS, 1>(q, s, max_blks, x, w, sc);
    case 10: return two ? gate_up_blk<10, BIAS, 2>(q, s, max_blks, x, w, sc) : gate_up_blk<10, BIAS, 1>(q, s, max_blks, x, w, sc);
    case 16: return two ? gate_up_blk<16, BIAS, 2>(q, s, max_blks, x, w, sc) : gate_up_blk<16, BIAS, 1>(q, s, max_blks, x, w, sc);
    case 20: return two ? gate_up_blk<20, BIAS, 2>(q, s, max_blks, x, w, sc) : gate_up_blk<20, BIAS, 1>(q, s, max_blks, x, w, sc);
    default: throw std::invalid_argument("moe_rows_gate_up_blk: tile pairs per card");
  }
}

// down: 16 threads per block. The block's h rows (NRB x 8 x this card's k, bf16) are staged once in SLM as the DPAS A
// tiles [k piece][call][8-row block][8][16]; each thread walks the output tile pairs p = j, j + 16... and runs, per pair,
// two tiles x NRB 8-row blocks of independent DPAS chains, each call's codes widened once for the blocks and the next
// piece's codes loading meanwhile. Each (row, column) keeps down2's chain (per k piece: its calls from zero,
// w4:apply_scale with the piece's group scale) and its bf16 store: the same bits. NPMAX: the most k pieces per card
// (NP = I / PK at run time).
template <int PK, int NPMAX, int NRB>
inline void down_blk_body(int j, int n, int p0, int H, int I, int NP, int pg0, const uint8_t *codes, const sycl::half *scales, const bf16 *hbuf,
                          bf16 *y) SYCL_ESIMD_FUNCTION {
  constexpr int kPB = PK * kW4BlockBytes / kGroup, PG = kGroup / PK, kCB = w4::kCallBytes, kKC = w4::kKPerCall, kJC = PK / kKC;
  constexpr int R = NRB * 8, T = 16, kTile = 8 * kKC;
  const int tiles = H / kW, pairs = tiles / 2;
  // stage: row r (past n: the last row), piece g, call jc -> [g][jc][rb][rl][16]
  for (int r = j; r < R; r += T) {
    const bf16 *src = hbuf + static_cast<size_t>(p0 + (r < n ? r : n - 1)) * I;
    const int rb = r / 8, rl = r % 8;
    for (int g = 0; g < NP; ++g) {
      esimd::simd<bf16, PK> a = esimd::block_load<bf16, PK>(src + g * PK);
#pragma unroll
      for (int jc = 0; jc < kJC; ++jc)
        esimd::slm_block_store<bf16, kKC>(static_cast<uint32_t>((((g * kJC + jc) * NRB + rb) * 8 + rl) * kKC * 2), a.template select<kKC, 1>(jc * kKC));
    }
  }
  esimd::barrier();
  constexpr auto hint = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<32>};
  for (int pr = j; pr < pairs; pr += T) {
    const int t0 = pr * 2;
    esimd::simd<float, R * kW> acc0(0.f), acc1(0.f);
    esimd::simd<uint8_t, kPB> nx0 = esimd::block_load<uint8_t, kPB>(codes + static_cast<size_t>(t0) * kPB);
    esimd::simd<uint8_t, kPB> nx1 = esimd::block_load<uint8_t, kPB>(codes + static_cast<size_t>(t0 + 1) * kPB);
    for (int g = 0; g < NP; ++g) {
      esimd::simd<uint8_t, kPB> raw0 = nx0, raw1 = nx1;
      if (g + 1 < NP) {  // the next piece's codes
        nx0 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g + 1) * tiles + t0) * kPB);
        nx1 = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g + 1) * tiles + t0 + 1) * kPB);
      }
      const int srow = (pg0 + g) / PG - pg0 / PG;
      const esimd::simd<float, kW> sc0 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t0 * kW);
      const esimd::simd<float, kW> sc1 = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + (t0 + 1) * kW);
      esimd::simd<float, R * kW> s0(0.f), s1(0.f);
#pragma unroll
      for (int jc = 0; jc < kJC; ++jc) {
        const esimd::simd<bf16, 2 * kCB> bb0 = widen_call<false>(raw0.template select<kCB, 1>(jc * kCB));
        const esimd::simd<bf16, 2 * kCB> bb1 = widen_call<false>(raw1.template select<kCB, 1>(jc * kCB));
        esimd::simd<bf16, NRB * kTile> at = esimd::slm_block_load<bf16, NRB * kTile>(static_cast<uint32_t>(((g * kJC + jc) * NRB) * kTile * 2));
#pragma unroll
        for (int rb = 0; rb < NRB; ++rb) {
          esimd::simd<float, 8 * kW> c = s0.template select<8 * kW, 1>(rb * 8 * kW);
          c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, bb0, at.template select<kTile, 1>(rb * kTile).read());
          s0.template select<8 * kW, 1>(rb * 8 * kW) = c;
          esimd::simd<float, 8 * kW> d = s1.template select<8 * kW, 1>(rb * 8 * kW);
          d = xmx::dpas<8, 8, float, float, bf16, bf16>(d, bb1, at.template select<kTile, 1>(rb * kTile).read());
          s1.template select<8 * kW, 1>(rb * 8 * kW) = d;
        }
      }
#pragma unroll
      for (int rb = 0; rb < NRB; ++rb) {
        esimd::simd<float, 8 * kW> ac = acc0.template select<8 * kW, 1>(rb * 8 * kW);
        w4::apply_scale<8>(ac, s0.template select<8 * kW, 1>(rb * 8 * kW), sc0);
        acc0.template select<8 * kW, 1>(rb * 8 * kW) = ac;
        ac = acc1.template select<8 * kW, 1>(rb * 8 * kW);
        w4::apply_scale<8>(ac, s1.template select<8 * kW, 1>(rb * 8 * kW), sc1);
        acc1.template select<8 * kW, 1>(rb * 8 * kW) = ac;
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r >= n) break;
      esimd::simd<bf16, 2 * kW> v;
      v.template select<kW, 1>(0) = acc0.template select<kW, 1>(r * kW);
      v.template select<kW, 1>(kW) = acc1.template select<kW, 1>(r * kW);
      esimd::block_store<bf16, 2 * kW>(y + static_cast<size_t>(p0 + r) * H + t0 * kW, v, hint);  // y in bf16
    }
  }
}

template <int PK, int NPMAX>
class MrDownBlk;
template <int PK, int NPMAX>
sycl::event down_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int T = 16;
  constexpr int kABytes = kMoeBlkRows * PK * NPMAX * 2;
  const int H = s.hidden, I = s.inter_local(), NP = I / PK;
  const int pg0 = s.card * NP;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *blk = sc.blk, *blk_n = sc.blk_n;
  const bf16 *hbuf = sc.h;
  bf16 *y = sc.y;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDownBlk<PK, NPMAX>>(sycl::nd_range<1>(static_cast<size_t>(max_blks) * T, T), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<kABytes>();
      const int b = static_cast<int>(it.get_group(0)), j = static_cast<int>(it.get_local_id(0));
      if (b >= blk_n[0]) return;
      const esimd::simd<int32_t, 4> bp = esimd::block_load<int32_t, 4>(blk + static_cast<size_t>(b) * 4);
      const int e = bp[0], p0 = bp[1], n = bp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      switch ((n + 7) / 8) {  // uniform over the work-group
        case 1: down_blk_body<PK, NPMAX, 1>(j, n, p0, H, I, NP, pg0, codes, scales, hbuf, y); break;
        case 2: down_blk_body<PK, NPMAX, 2>(j, n, p0, H, I, NP, pg0, codes, scales, hbuf, y); break;
        case 3: down_blk_body<PK, NPMAX, 3>(j, n, p0, H, I, NP, pg0, codes, scales, hbuf, y); break;
        default: down_blk_body<PK, NPMAX, 4>(j, n, p0, H, I, NP, pg0, codes, scales, hbuf, y); break;
      }
    });
  });
}

}  // namespace

sycl::event moe_rows_gate_up(sycl::queue &q, const MoeShape &s, int max_segs, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  return gate_up_both(q, s, max_segs, x, w, sc);
}
sycl::event moe_rows_down2(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  return moe_tp_piece(s.inter_local()) == 32 ? down2<32>(q, s, max_segs, w, sc) : down2<64>(q, s, max_segs, w, sc);
}

bool moe_rows_blk_shape(const MoeShape &s) {
  const int I = s.inter_local(), ti = I / kW, PK = moe_tp_piece(I);
  return I % kW == 0 && (ti == 8 || ti == 10 || ti == 16 || ti == 20) && s.hidden % 32 == 0 && s.hidden % kGroup == 0 && I % PK == 0 && I / PK <= 10;
}
sycl::event moe_rows_gate_up_blk(sycl::queue &q, const MoeShape &s, int max_blks, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc) {
  return s.w4_bias ? gate_up_blk_ti<true>(q, s, max_blks, x, w, sc) : gate_up_blk_ti<false>(q, s, max_blks, x, w, sc);
}
sycl::event moe_rows_down_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int I = s.inter_local(), PK = moe_tp_piece(I), NP = I / PK;
  if (PK == 32 && NP <= 5) return down_blk<32, 5>(q, s, max_blks, w, sc);
  if (PK == 64 && NP <= 5) return down_blk<64, 5>(q, s, max_blks, w, sc);
  if (PK == 64 && NP <= 10) return down_blk<64, 10>(q, s, max_blks, w, sc);
  throw std::invalid_argument("moe_rows_down_blk: k pieces per card");
}

}  // namespace xenithium:kernels
