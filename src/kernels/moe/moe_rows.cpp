// Xenithium · MoE block over many rows.
#include "xenithium/kernels/moe/moe_rows.hpp"

#include <algorithm>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/tile/row_quant.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/kernels/tile/w4a8_tile.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "comm/mailbox_esimd.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = moe_bf16;

constexpr int kW = 16;  // columns per tile
constexpr int kKCall = 16;
constexpr int kGroup = 128;
constexpr int kSeg = kMoeSegRows;  // 16
constexpr int kHalf = 64, kHalfBytes = kW4BlockBytes / 2;
constexpr int kLanes = 16;  // top-k sub-group
constexpr int kGroupItems = 1024;
constexpr int kCombineCols = 64;

class MrTopk;
class MrTopkSwiglu;
class MrGroup;
template <int PK>
class MrDown;
class MrSwiglu;
template <bool WIRE, bool PART, bool QUAD>
class MrCombine;

inline uint32_t ordered16(uint32_t b) {  // as moe.cpp: order-preserving image of bf16 bits
  if ((b & 0x7fffu) > 0x7f80u) return 0u;
  if (b == 0x8000u) b = 0u;
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}

// 8 rows x PK k against a PK-k piece of the TP blob's down codes (moe_tp_layout: a 64-k half block of 512 bytes, or at
// TP4's 160 columns per card a 32-k quarter of 256 bytes): fp32 [8][16] before the group's scale.
template <int PK>
inline constexpr int kPieceBytes = PK * kW4BlockBytes / kGroup;
constexpr int kQuarter = 32;
template <int PK = kHalf>
inline esimd::simd<float, 8 * kW> piece_dot8(esimd::simd<uint8_t, kPieceBytes<PK>> raw, esimd::simd<bf16, 8 * PK> a) SYCL_ESIMD_FUNCTION {
  constexpr int kCB = w4::kCallBytes;  // 128
  esimd::simd<float, 8 * kW> gsum(0.f);
#pragma unroll
  for (int j = 0; j < PK / kKCall; ++j) {
    const esimd::simd<uint8_t, kCB> bytes = raw.template select<kCB, 1>(j * kCB);
    const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
    const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
    esimd::simd<bf16, 2 * kCB> b;
    b.template select<kCB, 2>(0) = esimd::simd<float, kCB>(lo);
    b.template select<kCB, 2>(1) = esimd::simd<float, kCB>(hi);
    esimd::simd<bf16, 8 * kKCall> at;
#pragma unroll
    for (int r = 0; r < 8; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * PK + j * kKCall);
    gsum = xmx::dpas<8, 8, float, float, bf16, bf16>(gsum, b, at);
  }
  return gsum;
}

// ---- 2: top-k per row -----------------------------------------------------------------------------
// Row r's top-k on one 16-lane sub-group (a function, so the topk + swiglu launch runs the
// same code). Keys are unique (the low half is the expert id), so the picks do not depend on which
// lane holds which key. A lane holds experts x * 16 + lane, read
// by sub-group block reads when the row has PER x 16 floats (a few block messages instead of 32
// gathers); the keys sit in two halves with their maxima, and only the lane holding a pick
// rescans the half it came from (one 16-key pass per pick instead of two 32-key passes in every lane).
template <int PER>
inline void topk_row(const sycl::sub_group &sg, int r, int E, int K, const float *logits, int ld, int32_t *ids, float *topw, float *sgo) {
  namespace syex = sycl::ext::oneapi::experimental;
  constexpr int kHalfK = PER / 2;
  const int lane = static_cast<int>(sg.get_local_linear_id());
  const float *lr = logits + static_cast<size_t>(r) * ld;
  float lv[PER];
  if (ld >= PER * kLanes) {  // uniform: the whole sub-group takes one branch
    auto gp = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(lr);
    syex::group_load(sg, gp, sycl::span<float, PER>(lv, PER), syex::properties{syex::data_placement_striped, syex::full_group, syex::contiguous_memory});
  } else {
#pragma unroll
    for (int x = 0; x < PER; ++x) lv[x] = x * kLanes + lane < E ? lr[x * kLanes + lane] : 0.f;
  }
  uint32_t key[PER], hm[2] = {0u, 0u};
#pragma unroll
  for (int x = 0; x < PER; ++x) {
    const int e = x * kLanes + lane;
    const uint32_t b = sycl::bit_cast<uint16_t>(bf16(lv[x]));
    key[x] = e < E ? ((ordered16(b) << 16) | (0xffffu - static_cast<uint32_t>(e))) : 0u;
    hm[x / kHalfK] = key[x] > hm[x / kHalfK] ? key[x] : hm[x / kHalfK];
  }
  uint32_t lm = hm[0] > hm[1] ? hm[0] : hm[1];
  int my_e = 0;
  float my_v = 0.f, top = 0.f;
  for (int j = 0; j < K; ++j) {
    const uint32_t mx = sycl::reduce_over_group(sg, lm, sycl::maximum<uint32_t>());
    if (lm == mx) {  // the one lane holding the pick
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (hm[h] == mx) {
          uint32_t nm = 0;
#pragma unroll
          for (int x = 0; x < kHalfK; ++x) {
            const uint32_t k = key[h * kHalfK + x] == mx ? 0u : key[h * kHalfK + x];
            key[h * kHalfK + x] = k;
            nm = k > nm ? k : nm;
          }
          hm[h] = nm;
        }
      }
      lm = hm[0] > hm[1] ? hm[0] : hm[1];
    }
    const uint32_t o = mx >> 16;
    const uint16_t vb = static_cast<uint16_t>((o & 0x8000u) ? (o ^ 0x8000u) : (~o & 0xffffu));
    const float v = static_cast<float>(sycl::bit_cast<bf16>(vb));
    if (j == 0) top = v;
    if (lane == j) {
      my_e = static_cast<int>(0xffffu - (mx & 0xffffu));
      my_v = v;
    }
  }
  const float ex = lane < K ? sycl::exp(my_v - top) : 0.f;
  const float z = sycl::reduce_over_group(sg, ex, sycl::plus<float>());
  if (lane < K) {
    ids[static_cast<size_t>(r) * K + lane] = my_e;
    topw[static_cast<size_t>(r) * K + lane] = ex / z;
  }
  if (lane == 0) {
    const float g = static_cast<float>(bf16(logits[static_cast<size_t>(r) * ld + E]));
    sgo[r] = 1.f / (1.f + sycl::exp(-g));
  }
}

// The shared expert's h for one (row, column): bf16(silu(bf16 g) * bf16 u).
inline void swiglu_item(const float *sgu, bf16 *sh, int SI, size_t id) {
  const int r = static_cast<int>(id / SI), c = static_cast<int>(id % SI);
  const float g = static_cast<float>(bf16(sgu[static_cast<size_t>(r) * 2 * SI + c]));
  const float u = static_cast<float>(bf16(sgu[static_cast<size_t>(r) * 2 * SI + SI + c]));
  sh[static_cast<size_t>(r) * SI + c] = bf16(g * (1.f / (1.f + sycl::exp(-g))) * u);
}

template <int PER>
sycl::event topk(sycl::queue &q, const MoeShape &s, int m, const float *logits, int ld, const MoeRowsScratch &sc) {
  const int E = s.experts, K = s.topk;
  int32_t *ids = sc.ids;
  float *topw = sc.topw, *sgo = sc.sg;
  return q.parallel_for<MrTopk>(sycl::nd_range<1>(static_cast<size_t>(m) * kLanes, kLanes), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kLanes)]] {
    topk_row<PER>(it.get_sub_group(), static_cast<int>(it.get_group(0)), E, K, logits, ld, ids, topw, sgo);
  });
}

// Topk and the shared swiglu in one launch, when the shared gate|up rode the router's launch
// (both read only that launch's outputs). Work-groups [0, m) run topk, the rest the swiglu items, 16
// per work-group (uniform per work-group).
template <int PER>
sycl::event topk_swiglu(sycl::queue &q, const MoeShape &s, int m, const float *logits, int ld, const MoeRowsScratch &sc) {
  const int E = s.experts, K = s.topk, SI = s.shared_inter;
  int32_t *ids = sc.ids;
  float *topw = sc.topw, *sgo = sc.sg;
  const float *sgu = sc.sgu;
  bf16 *sh = sc.sh;
  const size_t nsw = static_cast<size_t>(m) * SI, wsw = (nsw + kLanes - 1) / kLanes;
  return q.parallel_for<MrTopkSwiglu>(sycl::nd_range<1>((static_cast<size_t>(m) + wsw) * kLanes, kLanes), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kLanes)]] {
    const int g = static_cast<int>(it.get_group(0));
    if (g < m) {
      topk_row<PER>(it.get_sub_group(), g, E, K, logits, ld, ids, topw, sgo);
      return;
    }
    const size_t id = static_cast<size_t>(g - m) * kLanes + it.get_local_id(0);
    if (id < nsw) swiglu_item(sgu, sh, SI, id);
  });
}

// ---- 3: group by expert ---------------------------------------------------------------------------
sycl::event group(sycl::queue &q, const MoeShape &s, int m, const MoeRowsScratch &sc) {
  const int E = s.experts, K = s.topk, A = m * s.topk;
  const int32_t *ids = sc.ids;
  int32_t *slot = sc.slot, *row = sc.row, *seg = sc.seg, *seg_n = sc.seg_n;
  int32_t *blk = sc.blk, *blk_n = sc.blk_n;  // the blocks too, when set
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kMoeMaxExperts), h);
    sycl::local_accessor<int32_t, 1> off(sycl::range<1>(kMoeMaxExperts), h);
    sycl::local_accessor<int32_t, 1> cur(sycl::range<1>(kMoeMaxExperts), h);
    h.parallel_for<MrGroup>(sycl::nd_range<1>(kGroupItems, kGroupItems), [=](sycl::nd_item<1> it) {
      const int i = static_cast<int>(it.get_local_linear_id());
      using LA = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
      if (i < E) {
        cnt[i] = 0;
        cur[i] = 0;
      }
      sycl::group_barrier(it.get_group());
      for (int a = i; a < A; a += kGroupItems) LA(cnt[ids[a]]).fetch_add(1);
      sycl::group_barrier(it.get_group());
      const int c = i < E ? cnt[i] : 0;
      const int ns = (c + kSeg - 1) / kSeg;
      const int o = sycl::exclusive_scan_over_group(it.get_group(), c, sycl::plus<int>());
      const int so = sycl::exclusive_scan_over_group(it.get_group(), ns, sycl::plus<int>());
      if (i < E) off[i] = o;
      for (int j = 0; j < ns; ++j) {
        int32_t *sp = seg + static_cast<size_t>(so + j) * 4;
        sp[0] = i;
        sp[1] = o + j * kSeg;
        sp[2] = sycl::min(kSeg, c - j * kSeg);
      }
      if (i == kGroupItems - 1) seg_n[0] = so + ns;
      if (blk) {
        const int nb = (c + kMoeBlkRows - 1) / kMoeBlkRows;
        const int bo = sycl::exclusive_scan_over_group(it.get_group(), nb, sycl::plus<int>());
        for (int j = 0; j < nb; ++j) {
          int32_t *bp = blk + static_cast<size_t>(bo + j) * 4;
          bp[0] = i;
          bp[1] = o + j * kMoeBlkRows;
          bp[2] = sycl::min(kMoeBlkRows, c - j * kMoeBlkRows);
        }
        if (i == kGroupItems - 1) blk_n[0] = bo + nb;
      }
      sycl::group_barrier(it.get_group());
      for (int a = i; a < A; a += kGroupItems) {
        const int e = ids[a];
        const int p = off[e] + LA(cur[e]).fetch_add(1);
        row[p] = a / K;
        slot[a] = p;
      }
    });
  });
}

// ---- 4: gate_up -> h: moe_rows_gate_up (moe_rows_gu.cpp, 256 GRF;)

// ---- 3b: stage the multi-segment cold experts in VRAM -----------------------------------
// One work-group: rows per expert (as group), then each cold expert with >= min_segs segments claims a
// stage slot, most segments first (passes at >= 8, >= 4, >= min_segs), at most cap. tab2[e] = its stage
// slot or its own blob; list = [count, experts...].
sycl::event stage_plan(sycl::queue &q, const MoeShape &s, int m, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int E = s.experts, A = m * s.topk, cap = sc.stage_cap, mins = sc.stage_min_segs;
  const int32_t *ids = sc.ids;
  const uint8_t *cold = w.cold;
  const uint8_t *const *blobs = w.blobs;
  const uint8_t **tab2 = sc.tab2;
  uint8_t *stage = sc.stage;
  const size_t stride = sc.stage_stride;
  int32_t *list = sc.stage_list;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kMoeMaxExperts), h);
    sycl::local_accessor<int32_t, 1> used(sycl::range<1>(1), h);
    h.parallel_for<class MrStagePlan>(sycl::nd_range<1>(kGroupItems, kGroupItems), [=](sycl::nd_item<1> it) {
      const int i = static_cast<int>(it.get_local_linear_id());
      using LA = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
      if (i < E) cnt[i] = 0;
      if (i == 0) used[0] = 0;
      sycl::group_barrier(it.get_group());
      for (int a = i; a < A; a += kGroupItems) LA(cnt[ids[a]]).fetch_add(1);
      sycl::group_barrier(it.get_group());
      const int ns = i < E ? (cnt[i] + kSeg - 1) / kSeg : 0;
      const bool want = i < E && cold[i] != 0 && ns >= mins;
      int slot = -1;
      const int lo[3] = {8, 4, mins};
      int hi = 1 << 30;
      for (int p = 0; p < 3; ++p) {
        if (want && slot < 0 && ns >= lo[p] && ns < hi) {
          const int k = LA(used[0]).fetch_add(1);
          if (k < cap) slot = k;
        }
        hi = sycl::min(hi, lo[p]);
        sycl::group_barrier(it.get_group());
      }
      if (i < E) tab2[i] = slot >= 0 ? stage + static_cast<size_t>(slot) * stride : blobs[i];
      if (slot >= 0) list[1 + slot] = i;
      if (i == 0) list[0] = sycl::min(used[0], cap);
    });
  });
}

// cap x kStageParts work-groups; group (slot, part) copies its part of the slot's blob, host -> VRAM, in
// 16-byte words, consecutive lanes on consecutive words.
constexpr int kStageParts = 16, kStageWg = 256;
sycl::event stage_copy(sycl::queue &q, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int cap = sc.stage_cap;
  const size_t words = sc.stage_stride / 16, per = (words + kStageParts - 1) / kStageParts;
  const uint8_t *const *blobs = w.blobs;
  uint8_t *stage = sc.stage;
  const int32_t *list = sc.stage_list;
  const size_t stride = sc.stage_stride;
  return q.parallel_for<class MrStageCopy>(sycl::nd_range<1>(static_cast<size_t>(cap) * kStageParts * kStageWg, kStageWg), [=](sycl::nd_item<1> it) {
    const int g = static_cast<int>(it.get_group(0));
    const int slot = g / kStageParts, part = g % kStageParts;
    if (slot >= list[0]) return;  // uniform per work-group
    const auto *src = reinterpret_cast<const sycl::vec<uint32_t, 4> *>(blobs[list[1 + slot]]);
    auto *dst = reinterpret_cast<sycl::vec<uint32_t, 4> *>(stage + static_cast<size_t>(slot) * stride);
    const size_t b = static_cast<size_t>(part) * per, e = sycl::min(words, b + per);
    for (size_t k = b + it.get_local_id(0); k < e; k += kStageWg) dst[k] = src[k];
  });
}

// ---- 5: down -> y -----------------------------------------------------------------------------------
// y is written once and read once by combine, so its store streams past L1
// (L2 write-back).
inline void y_store(bf16 *p, esimd::simd<bf16, kW> v) SYCL_ESIMD_FUNCTION {
  constexpr auto hint = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::write_back>, esimd::alignment<32>};
  esimd::block_store<bf16, kW>(p, v, hint);
}

template <int PK>
sycl::event down(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  constexpr int kPB = kPieceBytes<PK>, PG = kGroup / PK;
  const int H = s.hidden, I = s.inter_local(), tiles = H / kW, NP = I / PK;
  const int pg0 = s.card * NP;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n;
  const bf16 *hbuf = sc.h;
  bf16 *y = sc.y;
  constexpr int kWg = 8;
  const size_t items = static_cast<size_t>(max_segs) * tiles;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDown<PK>>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / tiles, t = gid % tiles;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      esimd::simd<float, kSeg * kW> acc(0.f);
      const int halves = n > 8 ? 2 : 1;  // a <= 8-row segment's second half skipped (rows past n are not stored)
      for (int g = 0; g < NP; ++g) {
        const esimd::simd<uint8_t, kPB> raw = esimd::block_load<uint8_t, kPB>(codes + (static_cast<size_t>(g) * tiles + t) * kPB);
        const int srow = (pg0 + g) / PG - pg0 / PG;
        const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t * kW);
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          if (hb >= halves) break;
          esimd::simd<bf16, 8 * PK> a;
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int rr = hb * 8 + r < n ? hb * 8 + r : n - 1;
            a.template select<PK, 1>(r * PK) = esimd::block_load<bf16, PK>(hbuf + static_cast<size_t>(p0 + rr) * I + g * PK);
          }
          esimd::simd<float, 8 * kW> ac = acc.template select<8 * kW, 1>(hb * 8 * kW);
          w4::apply_scale<8>(ac, piece_dot8<PK>(raw, a), scale);
          acc.template select<8 * kW, 1>(hb * 8 * kW) = ac;
        }
      }
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        const esimd::simd<bf16, kW> v = acc.template select<kW, 1>(r * kW);  // y in bf16
        y_store(y + static_cast<size_t>(p0 + r) * H + t * kW, v);
      }
    });
  });
}

// ---- W4A8: int8 activations x the int4 expert codes on s8 x s4 DPAS -------------------
// The activations are quantised per (row, group): s = amax / 127, q = rne(v * (127 / amax)) (an
// all-zero group: q = 0, s = 0); x per 128-k group (gate_up), h per 64-k group (down: one half
// block). The expert weights are the decode bytes: each 32-k call's 256 bytes are reordered in
// registers from the bf16 layout (byte p * 16 + lane holds k 2p, 2p + 1) to the s4 operand (dword
// (k / 8) * 16 + lane, nibble k % 8 LSB first), and q ^ 8 = q - 8 as s4. Per row and group the
// int32 sum becomes float: acc += (float(sum) * s_a) * s_w. Every row's arithmetic depends only on
// its own row, as in the bf16 path, so calls of any size give the same rows.
// The s4 reorder and the s8 x s4 dot live in w4a8_tile.hpp (shared with the decode MoE).
constexpr int kK8 = w4a8::kK8;  // k per s8 x s4 call
constexpr int kOp8 = w4a8::kOp8;  // bytes: the s4 B operand (32 k x 16 columns) = the s8 A operand (8 rows x 32 k)
template <int G>
class MrQuant;
class MrGateUp8;
class MrDown8;

template <int G>
sycl::event quant_rows(sycl::queue &q, int rows, int cols, const bf16 *src, int8_t *dst, float *scl) {
  const int gpr = cols / G;
  const size_t n = static_cast<size_t>(rows) * gpr;
  return q.parallel_for<MrQuant<G>>(sycl::nd_range<1>((n + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    row_quant_item<G>(src, cols, dst, scl, it.get_global_id(0), n);  // row_quant.hpp (shared with pf_dense_pair)
  });
}

using w4a8::to_s4;
// 8 rows x CALLS * 32 k (row r at r * CALLS * 32 in `a`) against the s4 operands: int32 [8][16].
template <int CALLS>
inline esimd::simd<int32_t, 8 * kW> dot8_s8(esimd::simd<uint8_t, CALLS * kOp8> b, esimd::simd<uint8_t, 8 * CALLS * kK8> a) SYCL_ESIMD_FUNCTION {
  return w4a8::dot_s8<8, CALLS>(b, a);
}

// gate_up on W4A8: the work split of gate_up (2 items per (segment, tile), the up sums through SLM),
// x as int8 rows with one scale per (row, 128-group).
sycl::event gate_up8(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_local(), tiles_i = I / kW, G = H / kGroup;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n, *row = sc.row;
  const uint8_t *xq = reinterpret_cast<const uint8_t *>(sc.xq);
  const float *xs = sc.xs;
  bf16 *hbuf = sc.h;
  const size_t wgs = static_cast<size_t>(max_segs) * tiles_i;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrGateUp8>(sycl::nd_range<1>(wgs * 2, 2), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      esimd::slm_init<kSeg * kW * sizeof(float)>();
      const int grp = static_cast<int>(it.get_group(0)), proj = static_cast<int>(it.get_local_id(0));
      const int sid = grp / tiles_i, j = grp % tiles_i;
      if (sid >= seg_n[0]) return;  // uniform per work-group
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.gu_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
      const int nt = proj * tiles_i + j;
      int rr[kSeg];
      esimd::simd<uint32_t, kSeg> soff;
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        rr[r] = row[p0 + (r < n ? r : n - 1)];
        soff[r] = static_cast<uint32_t>(rr[r] * G) * 4u;
      }
      esimd::simd<float, kSeg * kW> acc(0.f);
      for (int g = 0; g < G; ++g) {
        const esimd::simd<uint8_t, kW4BlockBytes> b = to_s4<4>(w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + nt) * kW4BlockBytes));
        const esimd::simd<float, kW> sv = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + nt * kW);
        const esimd::simd<float, kSeg> sa = esimd::gather<float, kSeg>(xs + g, soff);
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          esimd::simd<uint8_t, 8 * kGroup> ah;
#pragma unroll
          for (int r = 0; r < 8; ++r)
            ah.template select<kGroup, 1>(r * kGroup) = esimd::block_load<uint8_t, kGroup>(xq + static_cast<size_t>(rr[hb * 8 + r]) * H + g * kGroup);
          esimd::simd<int32_t, 8 * kW> gs = dot8_s8<4>(b, ah);
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const esimd::simd<float, kW> v = esimd::simd<float, kW>(gs.template select<kW, 1>(r * kW)) * static_cast<float>(sa[hb * 8 + r]);
            acc.template select<kW, 1>((hb * 8 + r) * kW) += v * sv;
          }
        }
      }
      if (proj == 1) esimd::slm_block_store<float, kSeg * kW>(0, acc);
      esimd::barrier();
      if (proj == 1) return;
      const esimd::simd<float, kSeg * kW> gb = esimd::simd<bf16, kSeg * kW>(acc);
      const esimd::simd<float, kSeg * kW> ub = esimd::simd<bf16, kSeg * kW>(esimd::slm_block_load<float, kSeg * kW>(0));
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

// down on W4A8: the work split of down, h as int8 with one scale per (list row, 64-k half block).
sycl::event down8(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc) {
  const int H = s.hidden, I = s.inter_local(), tiles = H / kW, NH = I / kHalf;
  const int hg0 = s.card * NH;
  const MoeBlobLayout lay = w.blob;
  const uint8_t *const *blobs = w.blobs;
  const int32_t *seg = sc.seg, *seg_n = sc.seg_n;
  const uint8_t *hq = reinterpret_cast<const uint8_t *>(sc.hq);
  const float *hs = sc.hs;
  bf16 *y = sc.y;
  constexpr int kWg = 8;
  const size_t items = static_cast<size_t>(max_segs) * tiles;
  return q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<MrDown8>(sycl::nd_range<1>((items + kWg - 1) / kWg * kWg, kWg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const int gid = static_cast<int>(it.get_global_id(0));
      const int sid = gid / tiles, t = gid % tiles;
      if (sid >= seg_n[0]) return;
      const esimd::simd<int32_t, 4> sp = esimd::block_load<int32_t, 4>(seg + static_cast<size_t>(sid) * 4);
      const int e = sp[0], p0 = sp[1], n = sp[2];
      const uint8_t *blob = blobs[e];
      const uint8_t *codes = blob + lay.dn_codes;
      const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
      esimd::simd<uint32_t, kSeg> soff;
#pragma unroll
      for (int r = 0; r < kSeg; ++r) soff[r] = static_cast<uint32_t>((p0 + (r < n ? r : n - 1)) * NH) * 4u;
      esimd::simd<float, kSeg * kW> acc(0.f);
      for (int g = 0; g < NH; ++g) {
        const esimd::simd<uint8_t, kHalfBytes> b = to_s4<2>(esimd::block_load<uint8_t, kHalfBytes>(codes + (static_cast<size_t>(g) * tiles + t) * kHalfBytes));
        const int srow = (hg0 + g) / 2 - hg0 / 2;
        const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(srow) * H + t * kW);
        const esimd::simd<float, kSeg> sa = esimd::gather<float, kSeg>(hs + g, soff);
#pragma unroll
        for (int hb = 0; hb < 2; ++hb) {
          esimd::simd<uint8_t, 8 * kHalf> a;
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int rr = hb * 8 + r < n ? hb * 8 + r : n - 1;
            a.template select<kHalf, 1>(r * kHalf) = esimd::block_load<uint8_t, kHalf>(hq + static_cast<size_t>(p0 + rr) * I + g * kHalf);
          }
          esimd::simd<int32_t, 8 * kW> gs = dot8_s8<2>(b, a);
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const esimd::simd<float, kW> v = esimd::simd<float, kW>(gs.template select<kW, 1>(r * kW)) * static_cast<float>(sa[hb * 8 + r]);
            acc.template select<kW, 1>((hb * 8 + r) * kW) += v * scale;
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kSeg; ++r) {
        if (r >= n) break;
        const esimd::simd<bf16, kW> v = acc.template select<kW, 1>(r * kW);  // y in bf16
        y_store(y + static_cast<size_t>(p0 + r) * H + t * kW, v);
      }
    });
  });
}

// ---- 6: combine (+ the two-card exchange) -----------------------------------------------------------
// PART (split mode): no exchange and no shared term; the card's fp32 expert sum goes to sc.part.
// QUAD (TP4): the four-card exchange (MoeRowsWire:quad), ((e0 + e1) + (e2 + e3)) on every card.
template <bool WIRE, bool PART = false, bool QUAD = false>
// base / rows: the launch's rows [base, base + rows) (split mode finishes the peer's rows first).
sycl::event combine(sycl::queue &q, const MoeShape &s, int base, int rows, int r0, int r1, const MoeRowsScratch &sc, MoeRowsWire wire, int parity, bf16 *out) {
  const int H = s.hidden, K = s.topk, chunks = H / kCombineCols;
  const int32_t *ids = sc.ids, *slot = sc.slot;
  const float *topw = sc.topw, *sgv = sc.sg, *sy = sc.sy;
  const bf16 *y = sc.y;
  float *part = sc.part;
  const mail::QuadPeers qp = QUAD ? mail::quad_peers(wire.quad) : mail::QuadPeers{};
  const size_t hop_words = 2ull * wire.rows_max * static_cast<size_t>(H);  // (quad) hop 2's slot region after hop 1's
  const uint32_t opaque0 = static_cast<uint32_t>(wire.rows_max < 0);  // (quad) 0, a kernel argument (mail:opaque)
  return q.parallel_for<MrCombine<WIRE, PART, QUAD>>(
      sycl::nd_range<1>(static_cast<size_t>(rows) * chunks, chunks % 8 == 0 ? 8 : 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const int id = static_cast<int>(it.get_global_id(0)), r = base + id / chunks, c0 = (id % chunks) * kCombineCols;
        // The row's experts in ascending id (decode's slot order), insertion sort of <= 16.
        int ei[kMoeMaxTopk], pi[kMoeMaxTopk];
        float wi[kMoeMaxTopk];
        for (int j = 0; j < K; ++j) {
          const size_t a = static_cast<size_t>(r) * K + j;
          int e = ids[a], p = slot[a];
          float wv = topw[a];
          int x = j;
          while (x > 0 && ei[x - 1] > e) {
            ei[x] = ei[x - 1];
            pi[x] = pi[x - 1];
            wi[x] = wi[x - 1];
            --x;
          }
          ei[x] = e;
          pi[x] = p;
          wi[x] = wv;
        }
        esimd::simd<float, kCombineCols> acc(0.f);
        for (int j = 0; j < K; ++j) {
          const esimd::simd<float, kCombineCols> v = esimd::block_load<bf16, kCombineCols>(y + static_cast<size_t>(pi[j]) * H + c0);
          acc = v * wi[j] + acc;
        }
        if constexpr (PART) {
          acc.copy_to(part + static_cast<size_t>(r) * H + c0);
          return;
        }
        // The shared expert is added after the two cards' expert partials meet (e0 + e1 is the same float on both
        // cards), and every card has it for every row, so a row's sum does not depend on the call's row count.
        if constexpr (QUAD) {
          const size_t at = (static_cast<size_t>(parity) * wire.rows_max + r) * H + c0;
          acc = mail::quad_sum<kCombineCols>(acc, qp, at, hop_words, kMoeMailEmpty, wire.wait_cycles, wire.error);
          // then bf16(opq(p x scale) + opq(sum)), the four-card split sum's (pair_split_finish) bits
          if (r >= r0 && r < r1) {
            const esimd::simd<float, kCombineCols> p = esimd::block_load<float, kCombineCols>(sy + static_cast<size_t>(r - r0) * H + c0);
            acc = mail::opaque<kCombineCols>(p * sgv[r], opaque0) + mail::opaque<kCombineCols>(acc, opaque0);
          }
          const esimd::simd<bf16, kCombineCols> o = acc;
          o.copy_to(out + static_cast<size_t>(r) * H + c0);
          return;
        } else if constexpr (WIRE) {
          constexpr auto uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                                esimd::cache_hint_L2<esimd::cache_hint::uncached>, esimd::alignment<256>};
          const size_t at = (static_cast<size_t>(parity) * wire.rows_max + r) * H + c0;
          esimd::simd<uint32_t, kCombineCols> mine = acc.template bit_cast_view<uint32_t>();
          mine.merge(esimd::simd<uint32_t, kCombineCols>(0x7fc00000u), mine == kMoeMailEmpty);
          esimd::block_store<uint32_t, kCombineCols>(wire.peer_inbox + at, mine, uc);
          esimd::simd<uint32_t, kCombineCols> got = esimd::block_load<uint32_t, kCombineCols>(wire.inbox + at, uc);
          bool ok = !(got == kMoeMailEmpty).any();
          if (!ok) {
            const uint64_t t0 = esimd::rdtsc();
            while (!ok) {
              esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
              got = esimd::block_load<uint32_t, kCombineCols>(wire.inbox + at, uc);
              ok = !(got == kMoeMailEmpty).any();
              if (!ok && esimd::rdtsc() - t0 > wire.wait_cycles) break;
            }
          }
          if (!ok) {
            wire.error[0] = 1u;
            got = 0x7fc00000u;
          }
          esimd::block_store<uint32_t, kCombineCols>(wire.inbox + at, esimd::simd<uint32_t, kCombineCols>(kMoeMailEmpty), uc);
          acc = acc + got.template bit_cast_view<float>();
        }
        if (r >= r0 && r < r1) {
          const esimd::simd<float, kCombineCols> p = esimd::block_load<float, kCombineCols>(sy + static_cast<size_t>(r - r0) * H + c0);
          acc = p * sgv[r] + acc;
        }
        const esimd::simd<bf16, kCombineCols> o = acc;
        o.copy_to(out + static_cast<size_t>(r) * H + c0);
      });
}

}  // namespace

int moe_rows_max_segs(const MoeShape &s, int rows_max) { return s.experts + rows_max * s.topk / kSeg + 1; }
int moe_rows_max_blks(const MoeShape &s, int rows_max) { return s.experts + rows_max * s.topk / kMoeBlkRows + 1; }

MoeRowsSizes moe_rows_sizes(const MoeShape &s, int rows_max, int router_n) {
  const size_t R = static_cast<size_t>(rows_max), A = R * s.topk, Rs = R;  // shared expert: every row
  MoeRowsSizes z;
  z.logits = R * router_n;
  z.ids = A;
  z.sg = R;
  z.row = A;
  z.seg = static_cast<size_t>(moe_rows_max_segs(s, rows_max)) * 4;
  z.h = A * s.inter_local();
  z.y = A * s.hidden;
  z.sgu = Rs * 2 * s.shared_inter;
  z.sh = Rs * s.shared_inter;
  z.sy = Rs * s.hidden;
  z.xq = R * s.hidden;
  z.xs = R * (s.hidden / kGroup);
  z.hq = A * s.inter_local();
  z.hs = A * std::max(s.inter_local() / kHalf, 2);  // (the TP4 W4A8 blocks: two segments per row)
  z.xr = R * s.hidden;
  z.blk = static_cast<size_t>(moe_rows_max_blks(s, rows_max)) * 4;
  return z;
}

size_t moe_rows_wire_words(const MoeShape &s, int rows_max) { return 2ull * rows_max * s.hidden * wire_hops(s.cards); }
sycl::event moe_rows_wire_init(sycl::queue &q, const MoeShape &s, int rows_max, uint32_t *inbox) {
  return q.fill(inbox, kMoeMailEmpty, moe_rows_wire_words(s, rows_max));
}

MoeEvents moe_rows(sycl::queue &q, const MoeShape &s, int m, const bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc, const MoeRowsWire *wire,
                   int parity, bf16 *out, bool a8, bool fuse) {
  if (!s.tp) throw std::invalid_argument("moe_rows: expert TP layout only");
  if (s.experts > kGroupItems || s.experts > 32 * kLanes || s.topk > kMoeMaxTopk || s.topk > kLanes) throw std::invalid_argument("moe_rows: experts / topk");
  if (s.hidden % kCombineCols || s.inter_local() % kQuarter || s.shared_inter % kW) throw std::invalid_argument("moe_rows: shape");
  // W4A8 at 160 columns per card: the block kernels with per-segment h scales (64-wide h groups would cut the card)
  const bool blocks8 = a8 && sc.blk && sc.blk_n && s.inter_local() % kHalf && moe_rows_blk8_shape(s);
  if (((a8 && !blocks8) || s.rot) && s.inter_local() % kHalf)  // the h quant groups / H_64 blocks are 64 wide
    throw std::invalid_argument("moe_rows: W4A8 needs 64-column cards or the TP4 block list; rot needs 64-column cards");
  if (a8 && (!sc.xq || !sc.xs || !sc.hq || !sc.hs)) throw std::invalid_argument("moe_rows: W4A8 without its scratch");
  if (s.rot && !sc.xr) throw std::invalid_argument("moe_rows: rot without sc.xr");
  if (wire && m > wire->rows_max) throw std::invalid_argument("moe_rows: more rows than the wire holds");
  if (wire && wire->split && !sc.part) throw std::invalid_argument("moe_rows: split mode without its partial buffer");
  if (wire && wire->quad.on() && (s.cards != kWireCardsMax || !wire->quad.complete()))
    throw std::invalid_argument("moe_rows: the four-card wire needs MoeShape::cards 4 and every card's inbox");
  // The split's parts (one over every row for a single send)
  const MoeSplitPart whole{0, m, wire ? wire->send : nullptr};
  const std::span<const MoeSplitPart> parts = !wire || !wire->split ? std::span<const MoeSplitPart>{}
                                              : !wire->parts.empty() ? wire->parts
                                              : wire->send           ? std::span<const MoeSplitPart>(&whole, 1)
                                                                     : std::span<const MoeSplitPart>{};
  int covered = 0;
  for (const MoeSplitPart &p : parts) {
    if (!p.send || !p.send->sp || p.r0 != covered || p.m < 1) throw std::invalid_argument("moe_rows: the split's parts cover the rows in order");
    if (p.send->sp->cards != (wire->quad.on() ? kWireCardsMax : 2)) throw std::invalid_argument("moe_rows: the split's cards differ from the wire's");
    covered += p.m;
  }
  if (!parts.empty() && covered != m) throw std::invalid_argument("moe_rows: the split's parts cover every row");
  const int max_segs = moe_rows_max_segs(s, m);
  MoeEvents ev;
  // Split mode with a send: the peer's rows of the expert sum first, their copy to the peer
  // on the side queue (the copy engine) while the shared expert and the own rows run here.
  const bool overlap = !parts.empty();
  const int SI = s.shared_inter;
  // Without the overlap the shared expert's gate|up rides the router's launch (both read x; a short call's two narrow
  // GEMMs are latency-bound, one launch costs about one of them; the same bits). With fuse the W4A8 x quantiser rides
  // it too (it reads only x), and topk runs the shared swiglu beside it in one launch (both read only the router
  // launch's outputs). The same bits.
  const bool shared_early = !overlap && pf_dense_pairable(m, w.router, w.shared_gu, false, sc.da8.q != nullptr);
  // Under rot the early quantiser rotates first (rot_quant.hpp), so it rides the launch too.
  const bool quant_early = shared_early && a8 && fuse;
  if (shared_early) {
    const PfRowQuant rq{x, s.hidden, sc.xq, sc.xs, s.rot};
    ev.router = pf_dense_pair(q, m, w.router, x, s.hidden, PfOut{sc.logits, nullptr, w.router.n}, w.shared_gu, x, s.hidden, PfOut{sc.sgu, nullptr, 2 * SI}, true,
                              quant_early ? &rq : nullptr, nullptr, sc.da8.on());  // the shared gate|up on W8A8 when set
  } else {
    ev.router = pf_dense(q, m, w.router, x, s.hidden, PfOut{sc.logits, nullptr, w.router.n});
  }
  const bool swiglu_early = shared_early && fuse && m > 0;
  ev.topk = swiglu_early ? topk_swiglu<32>(q, s, m, sc.logits, w.router.n, sc) : topk<32>(q, s, m, sc.logits, w.router.n, sc);
  ev.route = group(q, s, m, sc);
  // The multi-segment cold experts read from a VRAM copy (the same bytes)
  MoeWeights ws = w;
  if (sc.stage_cap > 0 && w.cold && sc.stage && sc.tab2 && sc.stage_list && m >= sc.stage_min_rows && m > kSeg * (sc.stage_min_segs - 1)) {
    if (sc.stage_stride % 16) throw std::invalid_argument("moe_rows: stage stride");
    stage_plan(q, s, m, w, sc);
    stage_copy(q, w, sc);
    ws.blobs = sc.tab2;
  }
  // MoeShape:rot: the routed experts read bf16(H_128 x) and bf16(H_64 h).
  // a8 + rot: rotation and quantiser in one launch each (moe_rot_quant_rows: the same bits as
  // moe_hadamard_rows then quant_rows, without the bf16 round trip); no rotated x copy is needed.
  if (a8) {
    if (!quant_early) {
      if (s.rot) moe_rot_quant_rows(q, m, s.hidden, s.hidden, x, sc.xq, sc.xs, kGroup, true);
      else quant_rows<kGroup>(q, m, s.hidden, x, sc.xq, sc.xs);
    }
    if (blocks8) {  // h leaves gate_up quantised
      const int max_blks = moe_rows_max_blks(s, m);
      ev.gate_up = moe_rows_gate_up8_blk(q, s, max_blks, ws, sc);
      ev.down = moe_rows_down8_blk(q, s, max_blks, ws, sc);
    } else {
      ev.gate_up = gate_up8(q, s, max_segs, ws, sc);
      if (s.rot) moe_rot_quant_rows(q, m * s.topk, s.inter_local(), s.inter_local(), sc.h, sc.hq, sc.hs, kHalf, true);
      else quant_rows<kHalf>(q, m * s.topk, s.inter_local(), sc.h, sc.hq, sc.hs);
      ev.down = down8(q, s, max_segs, ws, sc);
    }
  } else {
    const bf16 *xe = x;
    if (s.rot && m > 0) {
      moe_hadamard_rows(q, m, s.hidden, s.hidden, x, sc.xr, kGroup);
      xe = sc.xr;
    }
    // the work-group-per-block kernels when the caller gave the block list
    const bool blocks = sc.blk && sc.blk_n && moe_rows_blk_shape(s);
    const int max_blks = moe_rows_max_blks(s, m);
    ev.gate_up = blocks ? moe_rows_gate_up_blk(q, s, max_blks, xe, ws, sc) : moe_rows_gate_up(q, s, max_segs, xe, ws, sc);
    if (s.rot && m > 0) moe_hadamard_rows(q, m * s.topk, s.inter_local(), s.inter_local(), sc.h, sc.h, kHalf);
    const bool quarters = moe_tp_piece(s.inter_local()) == kQuarter;  // TP4's 160 columns per card
    ev.down = blocks                ? moe_rows_down_blk(q, s, max_blks, ws, sc)
              : s.hidden % 32 == 0 ? moe_rows_down2(q, s, max_segs, ws, sc)
              : quarters           ? down<kQuarter>(q, s, max_segs, ws, sc)
                                   : down<kHalf>(q, s, max_segs, ws, sc);
  }
  // A part's own rows on this card, as rows of the call.
  auto own_rows = [&](const MoeSplitPart &p, int *lo, int *hi) {
    pair_split_owner_rows(s.card, p.m, lo, hi, p.send->sp->cards);
    *lo += p.r0;
    *hi += p.r0;
  };
  for (const MoeSplitPart &p : parts) {
    // (four cards = every other card's rows, [0, own_lo) and [own_hi, m); two: the peer's, the same launch)
    int lo = 0, hi = 0;
    own_rows(p, &lo, &hi);
    if (lo > p.r0) combine<false, true>(q, s, p.r0, lo - p.r0, 0, m, sc, MoeRowsWire{}, 0, out);
    if (hi < p.r0 + p.m) combine<false, true>(q, s, hi, p.r0 + p.m - hi, 0, m, sc, MoeRowsWire{}, 0, out);
    ev.send = pair_split_send(q, p.send->side, s.card, p.m, s.hidden, sc.part + static_cast<size_t>(p.r0) * s.hidden, *p.send->sp, p.send->seq);
  }
  // Shared expert on every row, on each card (added after the cards' exchange, so a row's
  // sum does not depend on which card owns it; see combine). On the card's own rows only when the
  // caller's sum reads no other row's (MoeRowsWire:shared_own_rows), each buffer at the rows' own places.
  const bool shared_own = overlap && wire->shared_own_rows;
  const int r0 = 0, r1 = m;  // the rows whose shared term combine may add
  auto shared_rows = [&](int s0, int mc) {  // the shared expert on rows [s0, s0 + mc)
    if (mc <= 0) return;
    const float *sgu = sc.sgu + static_cast<size_t>(s0) * 2 * SI;
    bf16 *sh = sc.sh + static_cast<size_t>(s0) * SI;
    if (!shared_early) ev.shared_gu = pf_dense(q, mc, w.shared_gu, x + static_cast<size_t>(s0) * s.hidden, s.hidden, PfOut{sc.sgu + static_cast<size_t>(s0) * 2 * SI, nullptr, 2 * SI}, sc.da8.on());
    sycl::event sw;
    if (!swiglu_early) sw = q.parallel_for<MrSwiglu>(sycl::range<1>(static_cast<size_t>(mc) * SI), [=](sycl::id<1> id) { swiglu_item(sgu, sh, SI, id[0]); });
    ev.shared_down = pf_dense(q, mc, w.shared_down, sh, SI, PfOut{sc.sy + static_cast<size_t>(s0) * s.hidden, nullptr, s.hidden}, sc.da8.on());
    // profile: the paired gate|up counts under the router; shared = swiglu.. down (swiglu in
    // the topk launch: shared = down only, the swiglu counts under topk)
    if (shared_early) ev.shared_gu = swiglu_early ? ev.shared_down : sw;
  };
  if (shared_own) {
    for (const MoeSplitPart &p : parts) {
      int lo = 0, hi = 0;
      own_rows(p, &lo, &hi);
      shared_rows(lo, hi - lo);
    }
  } else {
    shared_rows(0, m);
  }
  if (overlap) {
    // the card's own rows after the shared expert (the other cards' went first, their copy runs beside)
    for (const MoeSplitPart &p : parts) {
      int lo = 0, hi = 0;
      own_rows(p, &lo, &hi);
      ev.combine = combine<false, true>(q, s, lo, hi - lo, r0, r1, sc, MoeRowsWire{}, 0, out);
    }
  } else if (wire && wire->split) {
    ev.combine = combine<false, true>(q, s, 0, m, r0, r1, sc, MoeRowsWire{}, 0, out);
  } else {
    if (!wire) ev.combine = combine<false>(q, s, 0, m, r0, r1, sc, MoeRowsWire{}, 0, out);
    else if (wire->quad.on()) ev.combine = combine<true, false, true>(q, s, 0, m, r0, r1, sc, *wire, parity & 1, out);
    else ev.combine = combine<true>(q, s, 0, m, r0, r1, sc, *wire, parity & 1, out);
  }
  return ev;
}

}  // namespace xenithium:kernels
