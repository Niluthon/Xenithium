// Xenithium · MoE decode block.
#include "xenithium/kernels/moe/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/comm/wire.hpp"
#include "comm/mailbox_esimd.hpp"
#include "xenithium/kernels/tile/w4_tile.hpp"
#include "xenithium/kernels/tile/w4a8_tile.hpp"
#include "xenithium/kernels/tile/rot_quant.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = moe_bf16;

// kM: the launch's row shape (rows.hpp: 4, 8 or 16), a template parameter of every row-dependent
// piece below. The 16-row shape's GEMM launches run two 8-row halves in each item (one load of each
// weight block, the 8-row body's DPAS chain per half: the same bits per row; two work-groups per tile read
// every tile twice) and the CPU path's fallback in two passes. Buffers inside one step (h, partial) are laid out by the
// shape; the routing record and the mailbox by the capacity kMoeRows, so steps of either shape can follow each other.
constexpr int kW = 16;  // columns per tile
constexpr int kKCall = 16;
constexpr int kGroup = 128;
constexpr int kCallElems = kDenseCallElems;  // 256
constexpr int kCombineCols = 64;  // columns per combine work-item
// The four-card owner combine's item moves 128 columns (its partial, the peers' mailbox words and the output row). Each
// column keeps its sum, so no bit moves.
constexpr int kOwnerCombineCols = 128;
constexpr int kRouteLanes = 16;
// Prefetching the next
constexpr int kGateUpKSplit = 8;
constexpr int kDownKSplit = 2;  // (TP4's quarters: 1, moe_decode)

template <int M, int PER, bool CPU, bool FULL>
class MoeRouteK;
template <int M, int KS, bool A8, int MC>
class MoeGateUpK;
template <int M, int KS, bool A8, bool FH, int MC, int PK, int TW>
class MoeDownK;
// How the combine's launch ends: the card's partial rounded (one card), the two-card exchange, the four-card two-hop
// sum, or the four-card owner sum (comm/wire.hpp, rows by owner: the card's own rows only).
enum class CombineSum { local, pair, quad, owner };
template <int M, CombineSum SUM>
class MoeCombineK;

// Order-preserving 16-bit image of bf16 bits: -0 -> +0, NaN -> lowest.
inline uint32_t ordered16(uint32_t b) {  // Selects only (the early return cost the route ~0.9 us per launch)
  b = b == 0x8000u ? 0u : b;
  const uint32_t o = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
  return (b & 0x7fffu) > 0x7f80u ? 0u : o;
}

template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}

// The rows a slot computes, packed into dense kR-row blocks. At 16-32 rows a routed expert gets
// 1.5-2.1 rows on average, which sit in different 8-row quarters of the step: computing every quarter that holds one
// re-read x (the A operand) per quarter and stored every computed row. Packed, a routed slot computes ceil(n / kR)
// blocks of its own rows (one block for n <= kR) and stores only those n rows; every buffer keeps its per-row layout
// (h [slot][row], the partial [slot][row]), so the combine, the CPU path and the tests read them as before. Pack entry i
// = the i-th row of the slot's mask in ascending order; the last block's tail repeats the last routed row (computed,
// never stored). A DPAS row's sums do not depend on the other rows of its call, so a row's bits are the same in any
// block position (test-moe: 17-32-row steps == each row alone). The shared slot packs every row in order.
template <int kN>
struct RowPack {
  esimd::simd<int32_t, kN> row;  // one vector, not kN scalars: 32 scalar entries spilled the 32-row kernels
  int n = 0;  // rows stored: entries [0, n) of row
};
template <int kN>
inline RowPack<kN> pack_rows(uint32_t mask) SYCL_ESIMD_FUNCTION {
  RowPack<kN> p;
  int last = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i) {
    const bool more = mask != 0u;
    last = more ? __builtin_ctz(mask) : last;
    p.row[i] = last;
    p.n += more ? 1 : 0;
    mask &= mask - 1u;  // 0 stays 0
  }
  return p;
}
// The 64-row shape's rows: lo = rows 0-31, hi = rows 32-63, in ascending row order as above. Two
// 32-bit words, not one 64-bit one: every item builds a pack, and 64-bit integer operations are emulated.
template <int kN>
inline RowPack<kN> pack_rows(uint32_t lo, uint32_t hi) SYCL_ESIMD_FUNCTION {
  RowPack<kN> p;
  int last = 0;
#pragma unroll
  for (int i = 0; i < kN; ++i) {
    const bool in_lo = lo != 0u, more = in_lo || hi != 0u;
    const int at = in_lo ? __builtin_ctz(lo) : 32 + __builtin_ctz(hi | 0x80000000u);  // (the guard bit: hi may be 0 when !more)
    last = more ? at : last;
    p.row[i] = last;
    p.n += more ? 1 : 0;
    lo &= lo - 1u;  // 0 stays 0
    hi &= in_lo ? hi : hi - 1u;  // hi's turn only once lo is empty
  }
  return p;
}
// Pack entries [w kN, w kN + kN) of a wider pack: the w-th window of a slot's rows (64 rows = two 32-row windows).
template <int kN, int kAll>
inline RowPack<kN> pack_window(const RowPack<kAll> &all, int w) SYCL_ESIMD_FUNCTION {
  RowPack<kN> p;
  esimd::simd<int32_t, kAll> rows = all.row;  // (select needs a non-const vector)
  p.row = rows.template select<kN, 1>(w * kN);
  const int left = all.n - w * kN;
  p.n = left < 0 ? 0 : left > kN ? kN : left;
  return p;
}
// kN consecutive rows from `first` (the shared slot's rows of one window).
template <int kN>
inline RowPack<kN> row_run(int first) SYCL_ESIMD_FUNCTION {
  RowPack<kN> p;
  p.row = esimd::simd<int32_t, kN>(first, 1);
  p.n = kN;
  return p;
}
template <int kN>
inline RowPack<kN> all_rows() SYCL_ESIMD_FUNCTION {
  RowPack<kN> p;
#pragma unroll
  for (int i = 0; i < kN; ++i) p.row[i] = i;
  p.n = kN;
  return p;
}
// The blocks of kR packed rows that hold stored rows (bit b: pack entries [b kR, b kR + kR)).
template <int kR>
inline uint32_t pack_blocks(int n) SYCL_ESIMD_FUNCTION {
  return (1u << ((n + kR - 1) / kR)) - 1u;
}

// x rows (pack entries [at, at + kM)) -> the A operand of one 128-k group.
template <int kM, int kN>
inline esimd::simd<bf16, kM * kGroup> a_group(const bf16 *x, int ld, int k0, const RowPack<kN> &p, int at = 0) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kM * kGroup> a;
#pragma unroll
  for (int r = 0; r < kM; ++r) a.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(p.row[at + r]) * ld + k0);
  return a;
}
// x rows [kM][k] -> the A operand of one 128-k group (rows in order: the CPU path's fallback).
template <int kM>
inline esimd::simd<bf16, kM * kGroup> a_group(const bf16 *x, int ld, int k0) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kM * kGroup> a;
#pragma unroll
  for (int r = 0; r < kM; ++r) a.template select<kGroup, 1>(r * kGroup) = esimd::block_load<bf16, kGroup>(x + static_cast<size_t>(r) * ld + k0);
  return a;
}

// A k piece of a TP blob's down codes (moe_tp_layout): PK = 64, a half block (512 bytes = the first or second half of a
// 1024-byte block); PK = 32, a quarter block (256 bytes). Byte (kg/2)*16 +
// lane; PK / 16 bf16 DPAS calls give the piece's fp32 sum before its group's scale.
constexpr int kHalf = 64, kHalfBytes = kW4BlockBytes / 2, kQuarter = 32;
template <int PK>
inline constexpr int kPieceBytes = PK * kW4BlockBytes / kGroup;
template <int kM, int PK, int kN>
inline esimd::simd<bf16, kM * PK> a_piece(const bf16 *x, int ld, int k0, const RowPack<kN> &p) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kM * PK> a;
#pragma unroll
  for (int r = 0; r < kM; ++r) a.template select<PK, 1>(r * PK) = esimd::block_load<bf16, PK>(x + static_cast<size_t>(p.row[r]) * ld + k0);
  return a;
}
template <int kM, int PK = kHalf>
inline esimd::simd<float, kM * kW> piece_dot(esimd::simd<uint8_t, kPieceBytes<PK>> raw, esimd::simd<bf16, kM * PK> a) SYCL_ESIMD_FUNCTION {
  constexpr int kAcc = kM * kW;
  constexpr int kCB = w4::kCallBytes;  // 128
  esimd::simd<float, kAcc> gsum(0.f);
#pragma unroll
  for (int j = 0; j < PK / kKCall; ++j) {
    const esimd::simd<uint8_t, kCB> bytes = raw.template select<kCB, 1>(j * kCB);
    const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
    const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
    esimd::simd<bf16, 2 * kCB> b;
    b.template select<kCB, 2>(0) = esimd::simd<float, kCB>(lo);
    b.template select<kCB, 2>(1) = esimd::simd<float, kCB>(hi);
    esimd::simd<bf16, kM * kKCall> at;
#pragma unroll
    for (int r = 0; r < kM; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * PK + j * kKCall);
    gsum = xmx::dpas<8, kM, float, float, bf16, bf16>(gsum, b, at);
  }
  return gsum;
}

// The 16-row shape's W4 x bf16 tiles. NC calls' B operands (q - 8 as bf16, VNNI: group_dot's /
// piece_dot's widening, call by call) from raw bytes, made once and shared by both 8-row halves; each half's DPAS chain
// then runs in group_dot's call order on its own sums: the same bits as the 8-row tile, one widening for 16 rows.
template <int NC, int NB>
inline esimd::simd<bf16, NC * 2 * w4::kCallBytes> b_calls(esimd::simd<uint8_t, NB> raw, int c0) SYCL_ESIMD_FUNCTION {
  constexpr int kCB = w4::kCallBytes;
  esimd::simd<bf16, NC * 2 * kCB> b;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const esimd::simd<uint8_t, kCB> bytes = raw.template select<kCB, 1>((c0 + c) * kCB);
    const esimd::simd<int16_t, kCB> lo = esimd::simd<int16_t, kCB>(bytes & 15) - kW4Zero;
    const esimd::simd<int16_t, kCB> hi = esimd::simd<int16_t, kCB>(bytes >> 4) - kW4Zero;
    b.template select<kCB, 2>(c * 2 * kCB) = esimd::simd<float, kCB>(lo);
    b.template select<kCB, 2>(c * 2 * kCB + 1) = esimd::simd<float, kCB>(hi);
  }
  return b;
}
// gsum [kHS][kR][16] += x rows (the pack's [kHS x kR] entries)[k0, k0 + NC x 16) * B, per block in call order (the
// blocks of hm only).
template <int kR, int kHS, int NC>
inline void dot_calls(esimd::simd<float, kHS * kR * kW> &gsum, esimd::simd<bf16, NC * 2 * w4::kCallBytes> b, const bf16 *x, int ld, int k0,
                      uint32_t hm, const RowPack<kHS * kR> &p) SYCL_ESIMD_FUNCTION {
  constexpr int kCB = w4::kCallBytes, kK = NC * kKCall;
#pragma unroll
  for (int hh = 0; hh < kHS; ++hh) {
    if (!((hm >> hh) & 1u)) continue;
    esimd::simd<bf16, kR * kK> a;
#pragma unroll
    for (int r = 0; r < kR; ++r) a.template select<kK, 1>(r * kK) = esimd::block_load<bf16, kK>(x + static_cast<size_t>(p.row[hh * kR + r]) * ld + k0);
    esimd::simd<float, kR * kW> gs = gsum.template select<kR * kW, 1>(hh * kR * kW);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      esimd::simd<bf16, kR * kKCall> at;
#pragma unroll
      for (int r = 0; r < kR; ++r) at.template select<kKCall, 1>(r * kKCall) = a.template select<kKCall, 1>(r * kK + c * kKCall);
      const esimd::simd<bf16, 2 * kCB> bc = b.template select<2 * kCB, 1>(c * 2 * kCB);
      gs = xmx::dpas<8, kR, float, float, bf16, bf16>(gs, bc, at);
    }
    gsum.template select<kR * kW, 1>(hh * kR * kW) = gs;
  }
}

// the 16- / 32-row shapes: 8-row blocks per item.
// The 64-row shape: one body still computes at most four blocks (eight blocks of accumulators do not
// fit the registers), so a slot with more than 32 rows runs the four-block body once per 32-row window of its pack.
inline constexpr int kBodyBlocksMax = 4;
template <int MC>
inline constexpr int kBlockRows = MC > 8 ? 8 : MC;  // rows of one DPAS chain
template <int MC>
inline constexpr int kHalvesOf = MC > 8 ? std::min(MC / 8, kBodyBlocksMax) : 1;  // blocks one body computes
template <int MC>
inline constexpr int kWindowsOf = MC / (kHalvesOf<MC> * kBlockRows<MC>);  // packed-row windows of a slot (2 at 64 rows)
// On the 64-row shape a routed slot of 9-16 rows runs the two-block body (the 16-row shape's): with every multi-block
// slot in the window loop, gate_up + down cost more per layer than with this tier.
template <int MC>
inline constexpr bool kTwoBlockTier = kWindowsOf<MC> > 1;
// A routed slot's row count from its mask words (hi: the 64-row shape's rows 32-63, 0 below it).
inline int slot_row_count(uint32_t lo, uint32_t hi) { return __builtin_popcount(lo) + __builtin_popcount(hi); }
// The first row of block hh of the shared slot's pack (its rows are consecutive): hh x 8 on the shapes of one window (a
// constant), the pack's entry on the 64-row shape, whose second window starts at row 32.
template <int MC, int kN>
inline int block_row0(const RowPack<kN> &pk, int hh) SYCL_ESIMD_FUNCTION {
  if constexpr (kWindowsOf<MC> > 1) return pk.row[hh * kBlockRows<MC>];
  else return hh * kBlockRows<MC>;
}

// int8 dense tile (layout of pack_dense_i8, full groups only), calls [c0, c1) of k:
// fp32 [kM][16] sum before the row scale.
// The calls' loads go out NP at a time before their DPAS. The shared
// expert's items walk 20 (gate_up, k split 8) / 10 (down, k split 2) calls, and as a load -> DPAS -> load chain
// that was the launch's critical path (the routed W4A8 items walk 2-3 blocks; a synthetic 55 x 64-item launch
// with a 3-deep chain already runs at the card's bandwidth. The DPAS order is
// unchanged: the same bits.
template <int kM, int NP = 2>
inline esimd::simd<float, kM * kW> i8_dot(const int8_t *w, int tiles, int t, const bf16 *x, int ld, int c0, int c1) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, kM * kW> acc(0.f);
  for (int cb = c0; cb < c1; cb += NP) {
    const int n = c1 - cb < NP ? c1 - cb : NP;
    esimd::simd<int8_t, NP * kCallElems> wv;
    esimd::simd<bf16, NP * kM * kKCall> av;
#pragma unroll
    for (int i = 0; i < NP; ++i)
      if (i < n) {
        const int ci = cb + i, g = ci / kDenseCalls, j = ci % kDenseCalls;
        const size_t base = (static_cast<size_t>(g) * tiles + t) * (kDenseCalls * kCallElems) + static_cast<size_t>(j) * kCallElems;
        wv.template select<kCallElems, 1>(i * kCallElems) = esimd::block_load<int8_t, kCallElems>(w + base);
#pragma unroll
        for (int r = 0; r < kM; ++r)
          av.template select<kKCall, 1>((i * kM + r) * kKCall) = esimd::block_load<bf16, kKCall>(x + static_cast<size_t>(r) * ld + ci * kKCall);
      }
#pragma unroll
    for (int i = 0; i < NP; ++i)
      if (i < n) {
        const esimd::simd<int8_t, kCallElems> c = wv.template select<kCallElems, 1>(i * kCallElems);
        const esimd::simd<float, kCallElems> f = c;  // exact
        const esimd::simd<bf16, kCallElems> b(f);
        const esimd::simd<bf16, kM * kKCall> at = av.template select<kM * kKCall, 1>(i * kM * kKCall);
        acc = xmx::dpas<8, kM, float, float, bf16, bf16>(acc, b, at);
      }
  }
  return acc;
}

// ---- launch 2: route ----------------------------------------------------------------------------
// One work-group, one 16-lane sub-group per row; latency-bound, so no dependent chains:
// - each lane loads its keys as contiguous vectors (experts [lane PER, lane PER + PER));
// - every lane derives each pass's winner from the reduced key itself, lane j keeps pass j;
// - the list is built through a presence table in SLM indexed by expert id: lane j of row r ORs
// bit r into mask[e] and stores w into wt[e][r] (a row names an expert once); after one
// barrier, item i owns experts [i P, i P + P) and the list order is ascending expert id over
// this card's list range: u = exclusive scan of present-expert counts over the work-group. A per-slot
// first-appearance scan over SLM or over sub-group broadcasts is slower than this table + scan.
// Hits (optional, device [experts]) += 1 per expert of the list: the decode reads the server's hot-expert
// ranking counts (one read per expert per step). Each list entry has one owner item, so a plain add is race-free.
// Per-row rule a row hands the layer's pool experts
// it routes to (cold[] == 2) to the host pool when it routes to at least cpu.min_cold of them (a sub-group sum over its
// top-k lanes), so a (row, expert) pair's unit is a function of that row alone. Experts with pool rows go to cpu_slot /
// cpu_blob / cpu_mask (the pool's rows) and the mailbox in list order; an expert whose every row is the pool's gets a null
// blob[] entry (gate_up / down skip it), one that other rows need stays on the card (the wait launch overwrites only the
// pool's rows of its partial). Then x's rows to cpu.x and the request number (a device counter: unique per graph replay)
// to the mailbox after a system-scope release fence. A per-row rule keeps a sequence's bits from depending on its
// batch-mates' routes.
template <int kM, int PER, bool kCpu, bool kFull>
sycl::event route(sycl::queue &q, const MoeShape &s, int rows, const float *logits, int ld, const uint8_t *const *blobs, MoeRoute *rt,
                  uint32_t *hits = nullptr, const bf16 *xin = nullptr, const uint8_t *cold = nullptr, MoeCpuLink cpu = {}) {
  const int H = s.hidden;
  const int E = s.experts, K = s.topk, card = s.card, cards = s.cards, per_ep = E / s.cards;
  const int l0 = s.tp ? 0 : card * per_ep, l1 = s.tp ? E : (card + 1) * per_ep;  // the list's experts
  const int T = s.shared_inter / kW;
  const int m0 = l0, m1 = l1;  // the presence table covers the list's experts
  constexpr int kItems = kM * kRouteLanes;
  // The 64-row shape (kWide): an expert's rows are two mask words (rows 0-31, 32-63), and the table of
  // weights per (expert, row) is gone - 512 experts x 64 rows would be all of the SLM - so each lane writes its own
  // weight into the record once its expert's slot is known (rt->weight then holds a row's own slots only).
  constexpr bool kWide = kM > kStepRows32;
  constexpr int kMaskWords = kWide ? 2 : 1;
  static_assert(!kWide || !kCpu, "moe route: the CPU path stops at 32 rows");
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> mask(sycl::range<1>(kMoeMaxExperts * kMaskWords), h);  // [expert][word]
    sycl::local_accessor<uint32_t, 1> cmask(sycl::range<1>(kCpu ? kMoeMaxExperts : 1), h);  // The pool's rows per expert
    // The list's experts x kM (32 rows x 1,024 experts would be all of the SLM)
    sycl::local_accessor<float, 1> wt(sycl::range<1>(kWide ? 1 : static_cast<size_t>(m1 - m0) * kM), h);
    sycl::local_accessor<int16_t, 1> slot_of(sycl::range<1>(kMoeMaxExperts), h);  // A listed expert's slot
    h.parallel_for<MoeRouteK<kM, PER, kCpu, kFull>>(
        sycl::nd_range<1>(kItems, kItems), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kRouteLanes)]] {
          const auto sg = it.get_sub_group();
          const int r = static_cast<int>(sg.get_group_linear_id());
          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int i = static_cast<int>(it.get_local_linear_id());
          const int P = (m1 - m0 + kItems - 1) / kItems;  // table entries per item
          for (int x = 0; x < P; ++x) {
            const int e = i * P + x;
            if (e < m1 - m0) {
              mask[e * kMaskWords] = 0u;
              if constexpr (kWide) mask[e * kMaskWords + 1] = 0u;
              if constexpr (kCpu) cmask[e] = 0u;
            }
          }
          uint32_t key[PER];
          const float *lr = logits + static_cast<size_t>(r) * ld + lane * PER;
          if constexpr (kFull) {  // E == kRouteLanes x PER, every lane's PER keys exist: no guards, -1.7 us per launch
#pragma unroll
            for (int x = 0; x < PER; x += 4) {
              const sycl::vec<float, 4> v4 = *reinterpret_cast<const sycl::vec<float, 4> *>(lr + x);
#pragma unroll
              for (int c = 0; c < 4; ++c)
                key[x + c] = (ordered16(sycl::bit_cast<uint16_t>(bf16(v4[c]))) << 16) | (0xffffu - static_cast<uint32_t>(lane * PER + x + c));
            }
          } else {
#pragma unroll
            for (int x = 0; x < PER; x += 4) {
              const int e0 = lane * PER + x;
              sycl::vec<float, 4> v4(0.f);
              if (e0 + 3 < E) v4 = *reinterpret_cast<const sycl::vec<float, 4> *>(lr + x);
#pragma unroll
              for (int c = 0; c < 4; ++c) {
                const int e = e0 + c;
                const uint32_t b = sycl::bit_cast<uint16_t>(bf16(v4[c]));
                key[x + c] = e < E ? ((ordered16(b) << 16) | (0xffffu - static_cast<uint32_t>(e))) : 0u;
              }
            }
          }
          int my_e = 0;
          float my_v = 0.f, top = 0.f;
          for (int j = 0; j < K; ++j) {
            uint32_t m = 0;
#pragma unroll
            for (int x = 0; x < PER; ++x) m = key[x] > m ? key[x] : m;
            m = sycl::reduce_over_group(sg, m, sycl::maximum<uint32_t>());
#pragma unroll
            for (int x = 0; x < PER; ++x) key[x] = key[x] == m ? 0u : key[x];
            const uint32_t o = m >> 16;
            const uint16_t vb = static_cast<uint16_t>((o & 0x8000u) ? (o ^ 0x8000u) : (~o & 0xffffu));
            const float v = static_cast<float>(sycl::bit_cast<bf16>(vb));
            if (j == 0) top = v;
            if (lane == j) {
              my_e = static_cast<int>(0xffffu - (m & 0xffffu));
              my_v = v;
            }
          }
          const float ex = lane < K ? sycl::exp(my_v - top) : 0.f;
          const float z = sycl::reduce_over_group(sg, ex, sycl::plus<float>());
          const float wv = ex / z;
          if (lane < K) {
            rt->ids[r][lane] = my_e;
            rt->topw[r][lane] = wv;
          }
          if (lane == 0) {
            const float g = static_cast<float>(bf16(logits[static_cast<size_t>(r) * ld + E]));
            rt->shared_gate[r] = 1.f / (1.f + sycl::exp(-g));
          }
          sycl::group_barrier(it.get_group());  // mask zeroed
          // the CPU path (uniform: every item reads the same words)
          const bool cpu_on = kCpu && cpu.mail != nullptr && rows >= cpu.min_rows && cpu.state[1] != 0u;
          // The list covers this card's experts (EP) or all of them (TP).
          const bool in = lane < K && r < rows && my_e >= m0 && my_e < m1;
          const bool pool_e = cpu_on && in && my_e >= l0 && my_e < l1 && cold[my_e - l0] == 2;
          const int n_pool = cpu_on ? sycl::reduce_over_group(sg, pool_e ? 1 : 0, sycl::plus<int>()) : 0;  // this row's pool experts (cpu_on uniform)
          // The row's listed experts' count (its slot list: below, the combine's)
          const int n_in = sycl::reduce_over_group(sg, in ? 1 : 0, sycl::plus<int>());
          if (lane == 0) rt->row_n[r] = n_in;
          if (in) {
            const int el = my_e - m0;
            const int word = kWide ? r >> 5 : 0;
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(
                mask[el * kMaskWords + word])
                .fetch_or(1u << (kWide ? r & 31 : r));
            if constexpr (kCpu) {
              if (pool_e && n_pool >= cpu.min_cold)
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(cmask[el])
                    .fetch_or(1u << r);
            }
            if constexpr (!kWide) wt[el * kM + r] = wv;
          }
          sycl::group_barrier(it.get_group());
          int cnt = 0, ncold = 0;  // ncold: experts with pool rows
          // an expert's rows: its mask word(s) from the table
          auto rows_lo = [&](int e) { return mask[(e - m0) * kMaskWords]; };
          auto rows_hi = [&](int e) { return kWide ? mask[(e - m0) * kMaskWords + kMaskWords - 1] : 0u; };
          for (int x = 0; x < P; ++x) {
            const int e = m0 + i * P + x;
            if (e < m1 && (rows_lo(e) | rows_hi(e)) != 0u) {
              ++cnt;
              if constexpr (kCpu) {
                if (cmask[e - m0] != 0u) ++ncold;
              }
            }
          }
          int u = sycl::exclusive_scan_over_group(it.get_group(), cnt, sycl::plus<int>());
          if (i == kItems - 1) rt->count = u + cnt;
          if (i == 0) {
            rt->shared_lo = card * T / cards;
            rt->shared_hi = (card + 1) * T / cards;
          }
          int cr = 0, n_cpu = 0;
          if (cpu_on) {
            cr = sycl::exclusive_scan_over_group(it.get_group(), ncold, sycl::plus<int>());
            n_cpu = sycl::group_broadcast(it.get_group(), cr + ncold, kItems - 1);  // <= rows x topk <= kMoeCpuMax (setup checks)
          }
          for (int x = 0; x < P; ++x) {
            const int e = m0 + i * P + x;
            if (e >= l1) break;
            if (e < l0) continue;
            const uint32_t mk = rows_lo(e), mk_hi = rows_hi(e);
            if (!(mk | mk_hi)) continue;
            const uint8_t *b = blobs ? blobs[e - l0] : nullptr;
            uint32_t cm = 0u;
            if constexpr (kCpu) cm = cmask[e - m0];
            if (kCpu && cm != 0u && cr < kMoeCpuMax) {  // the pool's rows of it
              rt->cpu_slot[cr] = u;
              rt->cpu_blob[cr] = b;
              rt->cpu_mask[cr] = cm;
              cpu.mail->mask[cr] = cm;
              cpu.mail->blob[cr] = reinterpret_cast<uint64_t>(b);
              ++cr;
              if (cm == mk) b = nullptr;  // every row the pool's: the card skips it
            }
            rt->expert[u] = e;
            rt->blob[u] = b;
            rt->mask[u] = mk;
            if constexpr (kWide) rt->mask_hi[u] = mk_hi;
            if (hits) hits[e] += 1u;
            if constexpr (!kWide) {
#pragma unroll
              for (int rr = 0; rr < kM; ++rr) rt->weight[rr][u] = (mk >> rr) & 1u ? wt[(e - m0) * kM + rr] : 0.f;
            }
            slot_of[e - m0] = static_cast<int16_t>(u);
            ++u;
          }
          // Each row's slots in ascending order: lane k's slot at its expert's rank among the row's listed
          // experts (ascending id = ascending slot), the ranks from the sub-group's other lanes
          sycl::group_barrier(it.get_group());
          {
            const int me = in ? my_e : -1;
            int rank = 0;
            for (int k2 = 0; k2 < K; ++k2) {
              const int o = sycl::select_from_group(sg, me, k2);
              rank += o >= 0 && o < me ? 1 : 0;
            }
            if (in) {
              const int slot = slot_of[my_e - m0];
              rt->row_slot[r][rank] = static_cast<int16_t>(slot);
              if constexpr (kWide) rt->weight[r][slot] = wv;
            }
          }
          n_cpu = sycl::min(n_cpu, kMoeCpuMax);
          if (cpu.mail && i == 0) rt->cpu_n = n_cpu;
          if (kCpu && n_cpu > 0) {  // uniform: x's rows to the pool, then the request
            using v4 = sycl::vec<uint32_t, 4>;
            const int chunks = rows * (H / 8);  // 16-byte chunks of the bf16 rows
            const v4 *src = reinterpret_cast<const v4 *>(xin);
            v4 *dst = reinterpret_cast<v4 *>(cpu.x);
            for (int w = i; w < chunks; w += kItems) dst[w] = src[w];
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
            sycl::group_barrier(it.get_group());
            if (i == 0) {
              cpu.mail->n = n_cpu;
              cpu.mail->rows = rows;
              const uint32_t c = cpu.state[0] + 1u;
              cpu.state[0] = c;
              rt->cpu_req = c;
              sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system, sycl::access::address_space::global_space>(cpu.mail->req)
                  .store(c);
            }
          }
        });
  });
}

// The shared expert's work-groups lead the gate_up / down grids. The shared slot computes every
// row (int8 chains of a k part, 1-4 blocks) and, dispatched last, was the launch's tail. Work-group slot g: 0 = the shared slot
// (`slots`), then the routed slots in list order.
inline int slot_of_group(int g, int slots) { return g == 0 ? slots : g - 1; }

// The routed experts' weights are read once per step, so their loads stream
// past L1 and do not allocate in L2 (L1 streaming + L2 uncached), which keeps the 24 MiB L2 for what a step reuses.
// The hints change no bits.
template <int N>
inline esimd::simd<uint8_t, N> wload(const uint8_t *p) SYCL_ESIMD_FUNCTION {
  constexpr auto kStream = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::streaming>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                             esimd::alignment<64>};
  return esimd::block_load<uint8_t, N>(p, kStream);
}
inline esimd::simd<uint8_t, kW4BlockBytes> wload_block(const uint8_t *p) SYCL_ESIMD_FUNCTION {  // w4:load_block, streamed
  esimd::simd<uint8_t, kW4BlockBytes> raw;
  raw.template select<512, 1>(0) = wload<512>(p);
  raw.template select<512, 1>(512) = wload<512>(p + 512);
  return raw;
}

// ---- launch 3: gate_up -> h ---------------------------------------------------------------------
// A8 (MoeShape:a8): the routed experts read x as int8 (sc.xq, one scale per (row, 128-group)
// in sc.xs) on s8 x s4 DPAS; the shared expert keeps bf16 x.
// `live` routed slots launched (rows x topk of the step: the route record never lists more) beside the shared
// slot's work-groups (the grid's first: slot_of_group); a 1-row step launched kM x topk slots, 3 of 4 of them
// work-groups that only read the count and returned. -1 = all.
// MC rows computed (1-2 for a 1-2-row step; h keeps the kM layout, rows >= MC not written), bf16 x only.
// The operands of one gate_up launch, shared by its work-group bodies.
struct GateUpArgs {
  int H, I, SI, slots, tiles, tiles_i, HW, G, sgu_tiles;
  const MoeRoute *rt;
  bf16 *h;
  MoeBlobLayout lay;
  const int8_t *sgu;
  const float *sgu_scale;
  const uint8_t *xq;
  const float *xs;
  const bf16 *x, *xr;
};

// One gate_up work-group (slot u, tile j): its k parts' sums for NB packed blocks of kR rows (the shape's every block, or
// one), then h for the stored rows. A slot with one block runs the NB = 1 body - its registers are the
// 8-row body's, where the four-block 32-row body spills at 128 GRF. The body is a function over an operand struct,
// not a lambda in the kernel.
template <int kM, int KS, bool A8, int MC, int NB, int kN>
inline void gate_up_group(const GateUpArgs &a, int lid, int u, int j, const uint8_t *blob, const RowPack<kN> &pk) SYCL_ESIMD_FUNCTION {
  constexpr int kHS = kHalvesOf<MC>, kR = kBlockRows<MC>, kAcc = kR * kW;
  static_assert(NB == 1 || NB == kHS || (NB == 2 && kTwoBlockTier<MC>), "gate_up: one block, a body's every block, or the two-block tier");
  const int proj = lid / KS, part = lid % KS;
  const bool shared = u == a.slots;
  const uint32_t hm = pack_blocks<kR>(pk.n);
  const int s_lo = a.rt->shared_lo;
  esimd::simd<float, NB * kAcc> acc(0.f);  // [block][kR][16]
  int col;  // h column of this tile
  if (!shared) {
    const uint8_t *codes = blob + a.lay.gu_codes;
    const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + a.lay.gu_scales);
    const int nt = proj * a.tiles_i + j;
    for (int g = part * a.G / KS; g < (part + 1) * a.G / KS; ++g) {
      const auto raw = wload_block(codes + (static_cast<size_t>(g) * 2 * a.tiles_i + nt) * kW4BlockBytes);
      const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * a.I + nt * kW);
      if constexpr (A8) {
        const auto b4 = w4a8::to_s4<4>(raw);
#pragma unroll
        for (int hh = 0; hh < NB; ++hh) {
          if (!((hm >> hh) & 1u)) continue;  // a block with no packed rows
          esimd::simd<uint8_t, kR * kGroup> ah;
#pragma unroll
          for (int r = 0; r < kR; ++r)
            ah.template select<kGroup, 1>(r * kGroup) = esimd::block_load<uint8_t, kGroup>(a.xq + static_cast<size_t>(pk.row[hh * kR + r]) * a.H + g * kGroup);
          esimd::simd<int32_t, kAcc> gs = w4a8::dot_s8<kR, 4>(b4, ah);
          esimd::simd<float, kAcc> gf;
#pragma unroll
          for (int r = 0; r < kR; ++r) gf.template select<kW, 1>(r * kW) = esimd::simd<float, kW>(gs.template select<kW, 1>(r * kW)) * a.xs[pk.row[hh * kR + r] * a.G + g];
          esimd::simd<float, kAcc> ab = acc.template select<kAcc, 1>(hh * kAcc);
          w4::apply_scale<kR>(ab, gf, scale);
          acc.template select<kAcc, 1>(hh * kAcc) = ab;
        }
      } else if constexpr (NB == 1) {
        w4::apply_scale<kR>(acc, w4::group_dot<kR>(raw, a_group<kR>(a.xr, a.H, g * kGroup, pk)), scale);  // routed: the rotated x under MoeShape:rot
      } else {  // Each call's B once for every block (two 64-k passes: fewer live registers)
        esimd::simd<float, NB * kAcc> gsum(0.f);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) dot_calls<kR, NB, 4>(gsum, b_calls<4>(raw, kk * 4), a.xr, a.H, g * kGroup + kk * 4 * kKCall, hm, pk);
#pragma unroll
        for (int hh = 0; hh < NB; ++hh) {
          if (!((hm >> hh) & 1u)) continue;
          esimd::simd<float, kAcc> ab = acc.template select<kAcc, 1>(hh * kAcc);
          w4::apply_scale<kR>(ab, gsum.template select<kAcc, 1>(hh * kAcc), scale);
          acc.template select<kAcc, 1>(hh * kAcc) = ab;
        }
      }
    }
    col = j * kW;
  } else {
    const int jt = s_lo + j, nt = proj * (a.SI / kW) + jt, calls = a.H / kKCall;
#pragma unroll
    for (int hh = 0; hh < NB; ++hh)
      acc.template select<kAcc, 1>(hh * kAcc) =
          i8_dot<kR>(a.sgu, a.sgu_tiles, nt, a.x + static_cast<size_t>(block_row0<MC>(pk, hh)) * a.H, a.H, part * calls / KS, (part + 1) * calls / KS);
    col = jt * kW;
    if (KS == 1) {
      const esimd::simd<float, kW> sc16 = esimd::block_load<float, kW>(a.sgu_scale + nt * kW);
#pragma unroll
      for (int r = 0; r < NB * kR; ++r) acc.template select<kW, 1>(r * kW) *= sc16;
    }
  }
#pragma unroll
  for (int hh = 0; hh < NB; ++hh) {
    if (!((hm >> hh) & 1u)) continue;
    esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>((lid * NB + hh) * kAcc * sizeof(float)), acc.template select<kAcc, 1>(hh * kAcc));
  }
  esimd::barrier();
  if (lid != 0) return;
#pragma unroll
  for (int hh = 0; hh < NB; ++hh) {
    if (!((hm >> hh) & 1u)) continue;
    esimd::simd<float, kAcc> gs(0.f), us(0.f);
#pragma unroll
    for (int p = 0; p < KS; ++p) {
      gs += esimd::slm_block_load<float, kAcc>(static_cast<uint32_t>((p * NB + hh) * kAcc * sizeof(float)));
      us += esimd::slm_block_load<float, kAcc>(static_cast<uint32_t>(((KS + p) * NB + hh) * kAcc * sizeof(float)));
    }
    if (shared && KS > 1) {
      const int jt = s_lo + j;
      const esimd::simd<float, kW> sg16 = esimd::block_load<float, kW>(a.sgu_scale + jt * kW);
      const esimd::simd<float, kW> su16 = esimd::block_load<float, kW>(a.sgu_scale + (a.SI / kW + jt) * kW);
#pragma unroll
      for (int r = 0; r < kR; ++r) {
        gs.template select<kW, 1>(r * kW) *= sg16;
        us.template select<kW, 1>(r * kW) *= su16;
      }
    }
    const esimd::simd<float, kAcc> gb = esimd::simd<bf16, kAcc>(gs);
    const esimd::simd<float, kAcc> ub = esimd::simd<bf16, kAcc>(us);
    esimd::simd<bf16, kAcc> hv = gb * sigmoid_v<kAcc>(gb) * ub;
    bf16 *hp = a.h + static_cast<size_t>(u) * kM * a.HW + col;
#pragma unroll
    for (int r = 0; r < kR; ++r) {
      if (hh * kR + r >= pk.n) continue;  // A block's tail (a repeated row) is not stored
      const esimd::simd<bf16, kW> row = hv.template select<kW, 1>(r * kW);
      row.copy_to(hp + static_cast<size_t>(pk.row[hh * kR + r]) * a.HW);
    }
  }
}

template <int kM, int KS, bool A8, int MC = kM>
sycl::event gate_up(sycl::queue &q, const MoeShape &s, const bf16 *x_in, const bf16 *xr_in, const MoeWeights &w, const MoeScratch &sc, int live = -1) {
  static_assert(!A8 || MC == kM, "gate_up: fewer computed rows on the bf16 path only");
  // The 16-row shape = two 8-row halves in each item (kHS): one weight block load, the 8-row body's DPAS
  // chain per half on its rows (same bits per row). The halves ran as two work-groups before, which read every tile
  // twice. The 32-row shape = four blocks alike, packed rows.
  constexpr int kHS = kHalvesOf<MC>, kR = kBlockRows<MC>, kWin = kWindowsOf<MC>;
  constexpr int kAcc = kR * kW;
  GateUpArgs a{};
  a.H = s.hidden, a.I = s.inter_local(), a.SI = s.shared_inter, a.slots = kM * s.topk;
  // A card's shared tiles come from the route record: a fixed 1/cards.
  const int tps_max = a.SI / kW / s.cards;
  a.tiles_i = a.I / kW, a.tiles = std::max(a.tiles_i, tps_max);
  a.HW = std::max(s.inter, a.SI), a.G = a.H / kGroup, a.sgu_tiles = 2 * a.SI / kW;
  a.rt = sc.route, a.h = sc.h, a.lay = w.blob;
  a.sgu = static_cast<const int8_t *>(w.shared_gu.w), a.sgu_scale = w.shared_gu.scale;
  a.xq = reinterpret_cast<const uint8_t *>(sc.xq), a.xs = sc.xs, a.x = x_in, a.xr = xr_in;
  const int nl = live < 0 || live > a.slots ? a.slots : live, tiles = a.tiles;
  return q.parallel_for<MoeGateUpK<kM, KS, A8, MC>>(
      sycl::nd_range<2>({static_cast<size_t>(nl + 1), static_cast<size_t>(tiles) * 2 * KS}, {1, static_cast<size_t>(2 * KS)}),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<2 * KS * kHS * kAcc * sizeof(float)>();
        // a grid of (slot, tile): the ids come from the launch, not from a division by the tile count in every item
        const int j = static_cast<int>(it.get_group(1)), u = slot_of_group(static_cast<int>(it.get_group(0)), a.slots);
        const int lid = static_cast<int>(it.get_local_id(1));
        const bool shared = u == a.slots;
        // count, the blob pointer and the row mask are independent loads (one latency, not a chain).
        const MoeRoute *rt = a.rt;
        const int count = rt->count, s_n = rt->shared_hi - rt->shared_lo;
        const uint8_t *blob = rt->blob[u < a.slots ? u : 0];
        const uint32_t mask = rt->mask[u < a.slots ? u : 0];
        // A null blob = the host pool's expert (MoeCpuLink)
        if (shared ? j >= s_n : (u >= count || j >= a.tiles_i || !blob)) return;  // uniform per work-group
        const uint32_t mask_hi = kWin > 1 ? rt->mask_hi[u < a.slots ? u : 0] : 0u;  // , the 64-row shape: rows 32-63
        auto pack = [&]<int kN>() SYCL_ESIMD_FUNCTION {  // the slot's first kN rows
          if constexpr (kWin > 1) return pack_rows<kN>(mask, mask_hi);
          else return pack_rows<kN>(mask);
        };
        // The body by the slot's row count (uniform per work-group); each packs only the entries it uses.
        // A routed slot with one block packs only that block's kR entries. Every work-group of a
        // 32-row step used to build the 32-entry pack (a serial bit scan per item) to use 8 of them.
        const int c = shared ? MC : slot_row_count(mask, mask_hi);
        if (!shared && c <= kR) {
          gate_up_group<kM, KS, A8, MC, 1>(a, lid, u, j, blob, pack.template operator()<kR>());
          return;
        }
        if constexpr (kTwoBlockTier<MC>) {
          if (!shared && c <= 2 * kR) {
            gate_up_group<kM, KS, A8, MC, 2>(a, lid, u, j, blob, pack.template operator()<2 * kR>());
            return;
          }
        }
        if constexpr (kWin > 1) {  // the 64-row shape: the four-block body, once per 32-row window of the slot's rows
          if (!shared && c <= kHS * kR) {  // one window, outside the loop
            gate_up_group<kM, KS, A8, MC, kHS>(a, lid, u, j, blob, pack.template operator()<kHS * kR>());
            return;
          }
          for (int w = 0; w * kHS * kR < c; ++w) {
            if (w > 0) esimd::barrier();  // item 0 has read the last window's sums: its SLM slots are free again
            const RowPack<kHS * kR> pk = shared ? row_run<kHS * kR>(w * kHS * kR) : pack_window<kHS * kR>(pack.template operator()<MC>(), w);
            gate_up_group<kM, KS, A8, MC, kHS>(a, lid, u, j, blob, pk);
          }
        } else {
          const RowPack<kHS * kR> pk = shared ? all_rows<kHS * kR>() : pack.template operator()<kHS * kR>();
          if constexpr (kHS == 1) gate_up_group<kM, KS, A8, MC, 1>(a, lid, u, j, blob, pk);
          else gate_up_group<kM, KS, A8, MC, kHS>(a, lid, u, j, blob, pk);
        }
      });
}

// Gate_up on W4A8 with h's rotate + quantise in its epilogue. A work-group covers
// one 64-column group (4 tiles x gate|up x KS parts), so after the parts are summed the group's h is whole
// in SLM and kM items rotate (under rot) + quantise its rows straight into sc.hq / sc.hs (rot_quant.hpp:
// the bits of the separate launch). The routed slots write no bf16 h; the shared slot's tiles run exactly
// as in gate_up (bf16 h, its own scales). Per tile the sums are formed as in gate_up: the same bits.
template <int M, int KS, int MC>
class MoeGateUp8GK;
// MC: rows computed (1 for a 1-row step; the buffers keep the kM layout, rows >= MC untouched)
template <int kM, int KS, int MC = kM>
sycl::event gate_up8g(sycl::queue &q, const MoeShape &s, const bf16 *x_in, const MoeWeights &w, const MoeScratch &sc) {
  constexpr int kHS = kHalvesOf<MC>, kR = MC / kHS;  // 16 rows = two 8-row halves per item (gate_up); 32: four
  constexpr int kAcc = kR * kW, kT = kHalf / kW, kItems = kT * 2 * KS;  // 4 tiles per 64-group
  const int H = s.hidden, I = s.inter_local(), SI = s.shared_inter, slots = kM * s.topk;
  const int tps_max = SI / kW / s.cards;
  const int tiles_i = I / kW, tiles = std::max(tiles_i, tps_max), groups = (tiles + kT - 1) / kT;
  const int HW = std::max(s.inter, SI), G = H / kGroup, sgu_tiles = 2 * SI / kW, NH = I / kHalf;
  const MoeRoute *rt = sc.route;
  bf16 *hbuf = sc.h;
  int8_t *hq = sc.hq;
  float *hs = sc.hs;
  const bool rot = s.rot;
  const MoeBlobLayout lay = w.blob;
  const int8_t *sgu = static_cast<const int8_t *>(w.shared_gu.w);
  const float *sgu_scale = w.shared_gu.scale;
  const uint8_t *xq = reinterpret_cast<const uint8_t *>(sc.xq);
  const float *xs = sc.xs;
  return q.parallel_for<MoeGateUp8GK<kM, KS, MC>>(
      sycl::nd_range<1>(static_cast<size_t>(slots + 1) * groups * kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        constexpr uint32_t kHvAt = kItems * kHS * kAcc * sizeof(float);  // the group's h: [kT][kHS][kR][16] floats
        esimd::slm_init<kHvAt + kT * kHS * kAcc * sizeof(float)>();
        const int grp = static_cast<int>(it.get_group(0)), u = slot_of_group(grp / groups, slots), jg = grp % groups;
        const bf16 *x = x_in;
        const int lid = static_cast<int>(it.get_local_id(0)), tt = lid / (2 * KS), rem = lid % (2 * KS), proj = rem / KS, part = rem % KS;
        const int j = jg * kT + tt;
        const bool shared = u == slots;
        const int count = rt->count, s_lo = rt->shared_lo, s_hi = rt->shared_hi;
        const uint8_t *blob = rt->blob[u < slots ? u : 0];
        const RowPack<kHS * kR> pk = shared ? all_rows<kHS * kR>() : pack_rows<kHS * kR>(rt->mask[u < slots ? u : 0]);
        const uint32_t hm = pack_blocks<kR>(pk.n);  // The packed blocks to compute
        if (shared ? jg * kT >= s_hi - s_lo : (u >= count || jg * kT >= tiles_i || !blob)) return;  // uniform per work-group
        const bool active = shared ? j < s_hi - s_lo : j < tiles_i;  // the routed groups are always whole (I % 64 == 0)
        esimd::simd<float, kHS * kAcc> acc(0.f);  // [half][kR][16]
        if (active) {
          if (!shared) {
            const uint8_t *codes = blob + lay.gu_codes;
            const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
            const int nt = proj * tiles_i + j;
            for (int g = part * G / KS; g < (part + 1) * G / KS; ++g) {
              const auto raw = w4::load_block(codes + (static_cast<size_t>(g) * 2 * tiles_i + nt) * kW4BlockBytes);
              const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * 2 * I + nt * kW);
              const auto b4 = w4a8::to_s4<4>(raw);
#pragma unroll
              for (int hh = 0; hh < kHS; ++hh) {
                if (!((hm >> hh) & 1u)) continue;
                esimd::simd<uint8_t, kR * kGroup> ah;
#pragma unroll
                for (int r = 0; r < kR; ++r)
                  ah.template select<kGroup, 1>(r * kGroup) = esimd::block_load<uint8_t, kGroup>(xq + static_cast<size_t>(pk.row[hh * kR + r]) * H + g * kGroup);
                esimd::simd<int32_t, kAcc> gs = w4a8::dot_s8<kR, 4>(b4, ah);
                esimd::simd<float, kAcc> gf;
#pragma unroll
                for (int r = 0; r < kR; ++r) gf.template select<kW, 1>(r * kW) = esimd::simd<float, kW>(gs.template select<kW, 1>(r * kW)) * xs[pk.row[hh * kR + r] * G + g];
                esimd::simd<float, kAcc> a = acc.template select<kAcc, 1>(hh * kAcc);
                w4::apply_scale<kR>(a, gf, scale);
                acc.template select<kAcc, 1>(hh * kAcc) = a;
              }
            }
          } else {
            const int jt = s_lo + j, nt = proj * (SI / kW) + jt, calls = H / kKCall;
#pragma unroll
            for (int hh = 0; hh < kHS; ++hh)
              acc.template select<kAcc, 1>(hh * kAcc) = i8_dot<kR>(sgu, sgu_tiles, nt, x + static_cast<size_t>(hh * kR) * H, H, part * calls / KS, (part + 1) * calls / KS);
            if (KS == 1) {
              const esimd::simd<float, kW> sc16 = esimd::block_load<float, kW>(sgu_scale + nt * kW);
#pragma unroll
              for (int r = 0; r < kHS * kR; ++r) acc.template select<kW, 1>(r * kW) *= sc16;
            }
          }
        }
#pragma unroll
        for (int hh = 0; hh < kHS; ++hh) {
          if (!((hm >> hh) & 1u)) continue;
          esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>((lid * kHS + hh) * kAcc * sizeof(float)), acc.template select<kAcc, 1>(hh * kAcc));
        }
        esimd::barrier();
        if (rem == 0 && active) {  // one item per tile: gate_up's epilogue
#pragma unroll
          for (int hh = 0; hh < kHS; ++hh) {
            if (!((hm >> hh) & 1u)) continue;
            esimd::simd<float, kAcc> gs(0.f), us(0.f);
#pragma unroll
            for (int p = 0; p < KS; ++p) {
              gs += esimd::slm_block_load<float, kAcc>(static_cast<uint32_t>(((tt * 2 * KS + p) * kHS + hh) * kAcc * sizeof(float)));
              us += esimd::slm_block_load<float, kAcc>(static_cast<uint32_t>(((tt * 2 * KS + KS + p) * kHS + hh) * kAcc * sizeof(float)));
            }
            if (shared && KS > 1) {
              const int jt = s_lo + j;
              const esimd::simd<float, kW> sg16 = esimd::block_load<float, kW>(sgu_scale + jt * kW);
              const esimd::simd<float, kW> su16 = esimd::block_load<float, kW>(sgu_scale + (SI / kW + jt) * kW);
#pragma unroll
              for (int r = 0; r < kR; ++r) {
                gs.template select<kW, 1>(r * kW) *= sg16;
                us.template select<kW, 1>(r * kW) *= su16;
              }
            }
            const esimd::simd<float, kAcc> gb = esimd::simd<bf16, kAcc>(gs);
            const esimd::simd<float, kAcc> ub = esimd::simd<bf16, kAcc>(us);
            esimd::simd<bf16, kAcc> hv = gb * sigmoid_v<kAcc>(gb) * ub;
            if (shared) {
              bf16 *hp = hbuf + (static_cast<size_t>(u) * kM + hh * kR) * HW + (s_lo + j) * kW;
#pragma unroll
              for (int r = 0; r < kR; ++r) {
                const esimd::simd<bf16, kW> row = hv.template select<kW, 1>(r * kW);
                row.copy_to(hp + static_cast<size_t>(r) * HW);
              }
            } else {
              esimd::slm_block_store<float, kAcc>(kHvAt + static_cast<uint32_t>((tt * kHS + hh) * kAcc * sizeof(float)), esimd::simd<float, kAcc>(hv));
            }
          }
        }
        if (shared) return;  // uniform per work-group
        esimd::barrier();
#pragma unroll
        for (int r = 0; r < MC; ++r) {  // one item per stored pack entry: the group's 64 h values of its row
          if (r % kItems != lid || r >= pk.n) continue;
          const int hh = r / kR, rr = r % kR;
          esimd::simd<float, kHalf> v;
#pragma unroll
          for (int t4 = 0; t4 < kT; ++t4)
            v.template select<kW, 1>(t4 * kW) = esimd::slm_block_load<float, kW>(kHvAt + static_cast<uint32_t>(((t4 * kHS + hh) * kAcc + rr * kW) * sizeof(float)));
          if (rot) rot_block<kHalf>(v);
          float sv;
          const esimd::simd<int8_t, kHalf> qb = quant_block<kHalf>(v, sv);
          const size_t row = static_cast<size_t>(u) * kM + pk.row[r];
          esimd::block_store<int8_t, kHalf>(hq + row * HW + jg * kHalf, qb);
          hs[row * NH + jg] = sv;
        }
      });
}

// ---- launch 4: down -> partial ------------------------------------------------------------------
// A8: the routed slots' h as int8 (sc.hq, laid out as h; one scale per (slot row, 64-k half)
// in sc.hs); tp only (checked by moe_decode). The shared slot keeps its bf16 h.
// PK: the tp blob's down piece (kHalf; kQuarter at TP4's 160 columns per card, moe_tp_piece). A8 runs on halves only.
// TW: output tiles per item of a routed slot with one packed block. TW = 2 (tp pieces, one k part):
// the work-group of an even tile computes its tile and the next one, the odd tile's work-group returns at once. A
// quarter piece is 256 bytes and a tile's five pieces lie 40 KB apart, so an item per tile streamed five 256-byte
// loads; two tiles' pieces are adjacent, so the pair reads 512 bytes per load and the slot's h rows once for both.
// Each tile's sums are formed exactly as before (the same pieces in the same order): the same bits. Micro on recorded
// routes (down per layer, us): 8 / 16 / 32 rows 29.95 / 49.13 / 85.67 -> 28.62 / 44.42 / 76.76; 1 row 5.73 -> 6.04
// so the caller picks it by the step's rows (kDownPairMinRows). Four tiles per item: 32 rows 75.7, 16 rows 45.6, 1 row
// 9.5: no better than two.
template <int kM, int KS, bool A8, bool FH = false, int MC = kM, int PK = kHalf, int TW = 1>
sycl::event down(sycl::queue &q, const MoeShape &s, const MoeWeights &w, const MoeScratch &sc, int live = -1) {  // live: as gate_up
  static_assert(PK == kHalf || !A8, "W4A8 down runs on 64-k half blocks");
  static_assert(TW == 1 || (TW == 2 && !A8 && KS == 1), "tile pairs: bf16 h, one k part");
  if (TW > 1 && (!s.tp || (s.hidden / kW) % TW != 0)) throw std::invalid_argument("moe down: tile pairs need the tp piece layout and an even tile count");
  constexpr int kHS = kHalvesOf<MC>, kR = kBlockRows<MC>, kWin = kWindowsOf<MC>;  // 16 rows = two 8-row halves per item (gate_up); 32: four
  constexpr int kAcc = kR * kW;
  const int H = s.hidden, I = s.inter_local(), SI = s.shared_inter, slots = kM * s.topk;
  const int tiles = H / kW, HW = std::max(s.inter, SI), G = I / kGroup, NH = I / kHalf;
  const bool tp = s.tp;
  const int hg0 = s.card * NH;  // tp: this card's first global half block (A8)
  const int NP = I / PK, PG = kGroup / PK, pg0 = s.card * NP;  // tp: pieces per card, per 128-k group, this card's first
  const MoeRoute *rt = sc.route;
  const bf16 *hbuf = sc.h;
  float *part = sc.partial;
  const MoeBlobLayout lay = w.blob;
  const int8_t *sdn = static_cast<const int8_t *>(w.shared_down.w);
  const float *sdn_scale = w.shared_down.scale;
  const bool round_y = s.down_bf16;
  const uint8_t *hq = reinterpret_cast<const uint8_t *>(sc.hq);
  const float *hs = sc.hs;
  const bool rot = s.rot;
  const int nl = live < 0 || live > slots ? slots : live;
  return q.parallel_for<MoeDownK<kM, KS, A8, FH, MC, PK, TW>>(
      sycl::nd_range<2>({static_cast<size_t>(nl + 1), static_cast<size_t>(tiles) * KS}, {1, static_cast<size_t>(KS)}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        if constexpr (KS > 1) esimd::slm_init<KS * kHS * kAcc * sizeof(float)>();
        // a grid of (slot, tile): the ids come from the launch, not from a division by the tile count in every item
        const int t = static_cast<int>(it.get_group(1)), pt = static_cast<int>(it.get_local_id(1));
        const int u = slot_of_group(static_cast<int>(it.get_group(0)), slots);
        const bool shared = u == slots;
        const int count = rt->count;
        const uint8_t *blob = rt->blob[u < slots ? u : 0];
        const uint32_t mask = rt->mask[u < slots ? u : 0];
        const uint32_t mask_hi = kWin > 1 ? rt->mask_hi[u < slots ? u : 0] : 0u;  // the 64-row shape's rows 32-63
        auto pack = [&]<int kN>() SYCL_ESIMD_FUNCTION {  // the slot's first kN rows
          if constexpr (kWin > 1) return pack_rows<kN>(mask, mask_hi);
          else return pack_rows<kN>(mask);
        };
        const int c = shared ? MC : slot_row_count(mask, mask_hi);  // the slot's rows
        const bool one_block = !shared && c <= kR;  // uniform per work-group
        if constexpr (TW > 1) {  // a one-block routed slot's odd tile belongs to its even neighbour's work-group (below)
          if (one_block && t % TW != 0) return;
        }
        if (!shared && (u >= count || !blob)) return;  // Null = the host pool's expert
        if constexpr (TW > 1) {
          if (one_block) {  // a routed slot with one block, at its pair's first tile
            const RowPack<kR> pk = pack.template operator()<kR>();
            constexpr int kPB = kPieceBytes<PK>;
            const bf16 *hp = hbuf + static_cast<size_t>(u) * kM * HW;
            const uint8_t *codes = blob + lay.dn_codes;
            const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
            esimd::simd<float, TW * kAcc> acc(0.f);  // [tile][kR][16]
            for (int g = 0; g < NP; ++g) {
              esimd::simd<uint8_t, TW * kPB> raw = wload<TW * kPB>(codes + (static_cast<size_t>(g) * tiles + t) * kPB);
              const int row = (pg0 + g) / PG - pg0 / PG;  // the piece's 128-k group, from this card's first
              esimd::simd<float, TW * kW> scale = esimd::block_load<sycl::half, TW * kW>(scales + static_cast<size_t>(row) * H + t * kW);
              const esimd::simd<bf16, kR * PK> a = a_piece<kR, PK>(hp, HW, g * PK, pk);
#pragma unroll
              for (int tt = 0; tt < TW; ++tt) {
                esimd::simd<float, kAcc> at = acc.template select<kAcc, 1>(tt * kAcc);
                const esimd::simd<uint8_t, kPB> piece = raw.template select<kPB, 1>(tt * kPB);
                const esimd::simd<float, kW> piece_scale = scale.template select<kW, 1>(tt * kW);
                w4::apply_scale<kR>(at, piece_dot<kR, PK>(piece, a), piece_scale);
                acc.template select<kAcc, 1>(tt * kAcc) = at;
              }
            }
            if (round_y) acc = esimd::simd<bf16, TW * kAcc>(acc);  // y_e in bf16 (MoeShape:down_bf16)
#pragma unroll
            for (int tt = 0; tt < TW; ++tt) {
              float *pp = part + static_cast<size_t>(u) * kM * H + (t + tt) * kW;
#pragma unroll
              for (int r = 0; r < kR; ++r) {
                if (r >= pk.n) continue;  // the stored pack entries only
                const esimd::simd<float, kW> out = acc.template select<kW, 1>((tt * kR + r) * kW);
                out.copy_to(pp + static_cast<size_t>(pk.row[r]) * H);
              }
            }
            return;
          }
        }
        // The group's body for NB computed blocks (as gate_up_group: the shape's every block, or one). A lambda in the
        // kernel: as a function over an operand struct (gate_up's form) the 1-item down groups ran +0.4 us each (micro).
        auto group = [&]<int NB, int kN>(const RowPack<kN> &pk) SYCL_ESIMD_FUNCTION {
        const uint32_t hm = pack_blocks<kR>(pk.n);  // The packed blocks to compute
        const bf16 *hp = hbuf + static_cast<size_t>(u) * kM * HW;
        esimd::simd<float, NB * kAcc> acc(0.f);  // [half][kR][16]
        if (!shared) {
          const uint8_t *codes = blob + lay.dn_codes;
          const sycl::half *scales = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
          if constexpr (A8) {  // tp: 64-k half blocks, h as int8 per (slot row, half)
            const uint8_t *hqp = hq + static_cast<size_t>(u) * kM * HW;
            const float *hsp = hs + static_cast<size_t>(u) * kM * NH;
            for (int g = pt * NH / KS; g < (pt + 1) * NH / KS; ++g) {
              const esimd::simd<uint8_t, kHalfBytes> raw = esimd::block_load<uint8_t, kHalfBytes>(codes + (static_cast<size_t>(g) * tiles + t) * kHalfBytes);
              const int row = (hg0 + g) / 2 - hg0 / 2;
              const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(row) * H + t * kW);
              const auto b2 = w4a8::to_s4<2>(raw);
#pragma unroll
              for (int hh = 0; hh < NB; ++hh) {
                if (!((hm >> hh) & 1u)) continue;
                esimd::simd<uint8_t, kR * kHalf> ah;
                esimd::simd<float, kR> sa;
                if constexpr (FH) {  // h's rotate + quantise here, on the bf16 h (rot_quant.hpp: the launch's bits)
#pragma unroll
                  for (int r = 0; r < kR; ++r) {
                    esimd::simd<float, kHalf> v = esimd::block_load<bf16, kHalf>(hp + static_cast<size_t>(pk.row[hh * kR + r]) * HW + g * kHalf);
                    if (rot) rot_block<kHalf>(v);
                    float sv;
                    esimd::simd<int8_t, kHalf> qb = quant_block<kHalf>(v, sv);
                    ah.template select<kHalf, 1>(r * kHalf) = qb.template bit_cast_view<uint8_t>();
                    sa[r] = sv;
                  }
                } else {
#pragma unroll
                  for (int r = 0; r < kR; ++r) {
                    ah.template select<kHalf, 1>(r * kHalf) = esimd::block_load<uint8_t, kHalf>(hqp + static_cast<size_t>(pk.row[hh * kR + r]) * HW + g * kHalf);
                    sa[r] = hsp[pk.row[hh * kR + r] * NH + g];
                  }
                }
                esimd::simd<int32_t, kAcc> gs = w4a8::dot_s8<kR, 2>(b2, ah);
                esimd::simd<float, kAcc> gf;
#pragma unroll
                for (int r = 0; r < kR; ++r) gf.template select<kW, 1>(r * kW) = esimd::simd<float, kW>(gs.template select<kW, 1>(r * kW)) * static_cast<float>(sa[r]);
                esimd::simd<float, kAcc> a = acc.template select<kAcc, 1>(hh * kAcc);
                w4::apply_scale<kR>(a, gf, scale);
                acc.template select<kAcc, 1>(hh * kAcc) = a;
              }
            }
          } else if (tp) {  // PK-k pieces (moe_tp_layout), each with its group's scale row
            constexpr int kPB = kPieceBytes<PK>;
            for (int g = pt * NP / KS; g < (pt + 1) * NP / KS; ++g) {
              const esimd::simd<uint8_t, kPB> raw = wload<kPB>(codes + (static_cast<size_t>(g) * tiles + t) * kPB);
              const int row = (pg0 + g) / PG - pg0 / PG;  // the piece's 128-k group, from this card's first
              const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(row) * H + t * kW);
              if constexpr (NB == 1) {
                w4::apply_scale<kR>(acc, piece_dot<kR, PK>(raw, a_piece<kR, PK>(hp, HW, g * PK, pk)), scale);
              } else {  // Each call's B once for every block
                esimd::simd<float, NB * kAcc> gsum(0.f);
                dot_calls<kR, NB, PK / kKCall>(gsum, b_calls<PK / kKCall>(raw, 0), hp, HW, g * PK, hm, pk);
#pragma unroll
                for (int hh = 0; hh < NB; ++hh) {
                  if (!((hm >> hh) & 1u)) continue;
                  esimd::simd<float, kAcc> a = acc.template select<kAcc, 1>(hh * kAcc);
                  w4::apply_scale<kR>(a, gsum.template select<kAcc, 1>(hh * kAcc), scale);
                  acc.template select<kAcc, 1>(hh * kAcc) = a;
                }
              }
            }
          } else {
            for (int g = pt * G / KS; g < (pt + 1) * G / KS; ++g) {
              const auto raw = wload_block(codes + (static_cast<size_t>(g) * tiles + t) * kW4BlockBytes);
              const esimd::simd<float, kW> scale = esimd::block_load<sycl::half, kW>(scales + static_cast<size_t>(g) * H + t * kW);
              if constexpr (NB == 1) {
                w4::apply_scale<kR>(acc, w4::group_dot<kR>(raw, a_group<kR>(hp, HW, g * kGroup, pk)), scale);
              } else {  // Each call's B once for every block
                esimd::simd<float, NB * kAcc> gsum(0.f);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) dot_calls<kR, NB, 4>(gsum, b_calls<4>(raw, kk * 4), hp, HW, g * kGroup + kk * 4 * kKCall, hm, pk);
#pragma unroll
                for (int hh = 0; hh < NB; ++hh) {
                  if (!((hm >> hh) & 1u)) continue;
                  esimd::simd<float, kAcc> a = acc.template select<kAcc, 1>(hh * kAcc);
                  w4::apply_scale<kR>(a, gsum.template select<kAcc, 1>(hh * kAcc), scale);
                  acc.template select<kAcc, 1>(hh * kAcc) = a;
                }
              }
            }
          }
        } else {
          const int s_lo = rt->shared_lo, n_sh = rt->shared_hi - s_lo;  // k calls (16 shared columns each)
#pragma unroll
          for (int hh = 0; hh < NB; ++hh)
            acc.template select<kAcc, 1>(hh * kAcc) =
                i8_dot<kR>(sdn, tiles, t, hp + static_cast<size_t>(block_row0<MC>(pk, hh)) * HW, HW, s_lo + pt * n_sh / KS, s_lo + (pt + 1) * n_sh / KS);
        }
        if constexpr (KS > 1) {
#pragma unroll
          for (int hh = 0; hh < NB; ++hh) {
            if (!((hm >> hh) & 1u)) continue;
            esimd::slm_block_store<float, kAcc>(static_cast<uint32_t>((pt * NB + hh) * kAcc * sizeof(float)), acc.template select<kAcc, 1>(hh * kAcc));
          }
          esimd::barrier();
          if (pt != 0) return;
#pragma unroll
          for (int hh = 0; hh < NB; ++hh) {
            if (!((hm >> hh) & 1u)) continue;
            esimd::simd<float, kAcc> a(0.f);
#pragma unroll
            for (int p2 = 0; p2 < KS; ++p2) a += esimd::slm_block_load<float, kAcc>(static_cast<uint32_t>((p2 * NB + hh) * kAcc * sizeof(float)));
            acc.template select<kAcc, 1>(hh * kAcc) = a;
          }
        }
        if (shared) {
          const esimd::simd<float, kW> sc16 = esimd::block_load<float, kW>(sdn_scale + t * kW);
#pragma unroll
          for (int r = 0; r < NB * kR; ++r) acc.template select<kW, 1>(r * kW) *= sc16;
        } else if (round_y) {
          acc = esimd::simd<bf16, NB * kAcc>(acc);  // y_e in bf16 (MoeShape:down_bf16)
        }
        float *pp = part + static_cast<size_t>(u) * kM * H + t * kW;
#pragma unroll
        for (int r = 0; r < NB * kR; ++r) {
          if (r >= pk.n) continue;  // The stored pack entries only (a block's tail repeats a row)
          const esimd::simd<float, kW> row = acc.template select<kW, 1>(r * kW);
          row.copy_to(pp + static_cast<size_t>(pk.row[r]) * H);
        }
        };
        // The body by the slot's row count, as gate_up.
        if (one_block) {  // Only the block's kR entries are packed
          group.template operator()<1>(pack.template operator()<kR>());
          return;
        }
        if constexpr (kTwoBlockTier<MC>) {
          if (!shared && c <= 2 * kR) {
            group.template operator()<2>(pack.template operator()<2 * kR>());
            return;
          }
        }
        if constexpr (kWin > 1) {  // , the 64-row shape: the four-block body per 32-row window (a 17-32-row
                                   // slot's one window stays in the loop: its own call read +0.4 us per layer here)
          for (int w = 0; w * kHS * kR < c; ++w) {
            if constexpr (KS > 1) {
              if (w > 0) esimd::barrier();  // item 0 has read the last window's sums: its SLM slots are free again
            }
            const RowPack<kHS * kR> pk = shared                ? row_run<kHS * kR>(w * kHS * kR)
                                         : c <= kHS * kR       ? pack.template operator()<kHS * kR>()
                                                               : pack_window<kHS * kR>(pack.template operator()<MC>(), w);
            group.template operator()<kHS>(pk);
          }
        } else {
          const RowPack<kHS * kR> pk = shared ? all_rows<kHS * kR>() : pack.template operator()<kHS * kR>();
          if constexpr (kHS == 1) group.template operator()<1>(pk);
          else group.template operator()<kHS>(pk);
        }
      });
}

// ---- the CPU path's wait launch ----------------------------------------------------------------------
// One work-group per pool expert (kMoeCpuMax groups; the ones past rt->cpu_n return). Item 0 polls the mailbox's done
// word with uncached loads (the only load that sees the host's store inside a recorded graph), bounded by
// cpu.wait_cycles; then the group copies the pool's y rows of its expert into the step's partial (uncached loads of
// host memory), so the combine runs unchanged. On a timeout the group computes the expert itself (gate_up into SLM, then
// down, the GPU path's tiles without its k split), clears the device's on word (state[1]: the next layers keep their
// cold experts) and raises the mailbox's timeouts word for the host.
template <int M>
class MoeCpuWaitK;
constexpr int kWaitItems = 16;
constexpr int kWaitHMax = 1024;  // inter_local columns the fallback's SLM h holds
template <int kM>
sycl::event cpu_wait(sycl::queue &q, const MoeShape &s, const bf16 *x, const MoeWeights &w, const MoeScratch &sc, MoeCpuLink cpu) {
  const int H = s.hidden, Ic = s.inter_local(), tic = Ic / kW, tiles = H / kW, NH = Ic / kHalf, G = H / kGroup, hg0 = s.card * NH;
  const MoeRoute *rt = sc.route;
  float *part = sc.partial;
  const MoeBlobLayout lay = w.blob;
  return q.parallel_for<MoeCpuWaitK<kM>>(
      sycl::nd_range<1>(static_cast<size_t>(kMoeCpuMax) * kWaitItems, kWaitItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        constexpr uint32_t kHAt = 64;  // SLM: [0] the poll's verdict, [64...) h [kM][kWaitHMax] bf16
        esimd::slm_init<kHAt + kM * kWaitHMax * 2>();
        const int i = static_cast<int>(it.get_group(0)), tid = static_cast<int>(it.get_local_id(0));
        if (i >= rt->cpu_n) return;  // uniform per work-group
        const uint32_t req = rt->cpu_req;
        const int u = rt->cpu_slot[i];
        const uint32_t mk = rt->cpu_mask[i];  // The pool's rows (the card wrote the others)
        constexpr auto uc64 = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                                esimd::alignment<64>};
        constexpr auto uc256 = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                                 esimd::alignment<256>};
        if (tid == 0) {
          const uint32_t *dp = &cpu.mail->done;
          esimd::simd<uint32_t, 16> d = esimd::block_load<uint32_t, 16>(dp, uc64);
          bool ok = d[0] == req;
          if (!ok) {
            const uint64_t t0 = esimd::rdtsc();
            while (!ok) {
              esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
              d = esimd::block_load<uint32_t, 16>(dp, uc64);
              ok = d[0] == req;
              if (!ok && esimd::rdtsc() - t0 > cpu.wait_cycles) break;
            }
          }
          esimd::slm_block_store<uint32_t, 4>(0, esimd::simd<uint32_t, 4>(ok ? 1u : 0u));
          if (!ok) {
            cpu.state[1] = 0u;  // the next layers keep their cold experts on the card
            esimd::block_store<uint32_t, 16>(&cpu.mail->timeouts, esimd::simd<uint32_t, 16>(1u), uc64);
          }
        }
        esimd::barrier();
        const bool ok = esimd::slm_block_load<uint32_t, 4>(0)[0] != 0u;
        float *pp = part + static_cast<size_t>(u) * kM * H;
        if (ok) {  // the pool's y rows -> the partial
#pragma unroll
          for (int r = 0; r < kM; ++r) {
            if (!((mk >> r) & 1u)) continue;
            const float *yr = cpu.y + (static_cast<size_t>(i) * kMoeCpuRows + r) * H;
            for (int c = tid * 64; c < H; c += kWaitItems * 64) {
              const esimd::simd<float, 64> v = esimd::block_load<float, 64>(yr + c, uc256);
              v.copy_to(pp + static_cast<size_t>(r) * H + c);
            }
          }
          return;
        }
        // timed out: this expert on the card
        const uint8_t *blob = rt->cpu_blob[i];
        const sycl::half *gsc = reinterpret_cast<const sycl::half *>(blob + lay.gu_scales);
        // The 16-row shape in two passes of 8 rows (kR: the 8-row body's bits)
        constexpr int kR = kM > 8 ? 8 : kM;
        for (int j = tid; j < tic; j += kWaitItems)
          for (int rb = 0; rb < kM; rb += kR) {
            esimd::simd<float, kR * kW> ag(0.f), au(0.f);
            for (int g = 0; g < G; ++g) {
              const auto a = a_group<kR>(x + static_cast<size_t>(rb) * H, H, g * kGroup);
              const auto rg = w4::load_block(blob + lay.gu_codes + (static_cast<size_t>(g) * 2 * tic + j) * kW4BlockBytes);
              const auto ru = w4::load_block(blob + lay.gu_codes + (static_cast<size_t>(g) * 2 * tic + tic + j) * kW4BlockBytes);
              const esimd::simd<float, kW> sg = esimd::block_load<sycl::half, kW>(gsc + static_cast<size_t>(g) * 2 * Ic + j * kW);
              const esimd::simd<float, kW> su = esimd::block_load<sycl::half, kW>(gsc + static_cast<size_t>(g) * 2 * Ic + Ic + j * kW);
              w4::apply_scale<kR>(ag, w4::group_dot<kR>(rg, a), sg);
              w4::apply_scale<kR>(au, w4::group_dot<kR>(ru, a), su);
            }
            const esimd::simd<float, kR * kW> gb = esimd::simd<bf16, kR * kW>(ag);
            const esimd::simd<float, kR * kW> ub = esimd::simd<bf16, kR * kW>(au);
            esimd::simd<bf16, kR * kW> hv = gb * sigmoid_v<kR * kW>(gb) * ub;
#pragma unroll
            for (int r = 0; r < kR; ++r) {
              const esimd::simd<bf16, kW> hr = hv.template select<kW, 1>(r * kW);
              esimd::slm_block_store<bf16, kW>(kHAt + static_cast<uint32_t>(((rb + r) * kWaitHMax + j * kW) * 2), hr);
            }
          }
        esimd::barrier();
        const sycl::half *dsc = reinterpret_cast<const sycl::half *>(blob + lay.dn_scales);
        for (int t = tid; t < tiles; t += kWaitItems)
          for (int rb = 0; rb < kM; rb += kR) {
            esimd::simd<float, kR * kW> acc(0.f);
            for (int g = 0; g < NH; ++g) {
              const esimd::simd<uint8_t, kHalfBytes> raw = esimd::block_load<uint8_t, kHalfBytes>(blob + lay.dn_codes + (static_cast<size_t>(g) * tiles + t) * kHalfBytes);
              esimd::simd<bf16, kR * kHalf> a;
#pragma unroll
              for (int r = 0; r < kR; ++r)
                a.template select<kHalf, 1>(r * kHalf) = esimd::slm_block_load<bf16, kHalf>(kHAt + static_cast<uint32_t>(((rb + r) * kWaitHMax + g * kHalf) * 2));
              const int row = (hg0 + g) / 2 - hg0 / 2;
              const esimd::simd<float, kW> sc16 = esimd::block_load<sycl::half, kW>(dsc + static_cast<size_t>(row) * H + t * kW);
              w4::apply_scale<kR>(acc, piece_dot<kR>(raw, a), sc16);
            }
#pragma unroll
            for (int r = 0; r < kR; ++r) {
              if (!((mk >> (rb + r)) & 1u)) continue;  // Rows the card computed keep its bits
              const esimd::simd<float, kW> rowv = acc.template select<kW, 1>(r * kW);
              rowv.copy_to(pp + static_cast<size_t>(rb + r) * H + t * kW);
            }
          }
      });
}

// ---- launch 5: combine (+ all-reduce) -----------------------------------------------------------
template <int kM, CombineSum SUM>
sycl::event combine(sycl::queue &q, const MoeShape &s, int rows, const MoeScratch &sc, MoeWire wire, int parity, bf16 *out) {
  constexpr int COLS = SUM == CombineSum::owner ? kOwnerCombineCols : kCombineCols;  // the item's columns
  static_assert(kGroup % COLS == 0, "the combine's item columns divide a 128-column group");
  const int H = s.hidden, slots = kM * s.topk, chunks = H / COLS;
  const MoeRoute *rt = sc.route;
  const float *part = sc.partial;
  const mail::QuadPeers qp = SUM == CombineSum::quad ? mail::quad_peers(wire.quad) : mail::QuadPeers{};
  const size_t hop_words = 2ull * kMoeRows * static_cast<size_t>(H);  // (quad) hop 2's slot region after hop 1's
  const mail::OwnerBoxes boxes = SUM == CombineSum::owner ? mail::owner_boxes(wire.owner) : mail::OwnerBoxes{};
  const uint32_t opaque0 = static_cast<uint32_t>(rows < 0);  // 0: a kernel argument the compiler cannot know
  return q.parallel_for<MoeCombineK<kM, SUM>>(
      sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks)}, {1, static_cast<size_t>(chunks % 8 == 0 ? 8 : 1)}),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * COLS;  // a (row, chunk) grid
        // The row's own slots (the route's list: ascending slot), its partials 8 at a time: the 8 loads go
        // out before the accumulation, a short tail re-reads the last slot with a zero weight, so the sum is still in
        // ascending slot order over the slots weighing the row (every slot of the step read, the others weighed 0 and
        // skipped): the same terms in the same order, so the same bits).
        const int n = rt->row_n[r];
        const int16_t *rs = &rt->row_slot[r][0];
        const float *wrow = &rt->weight[r][0];
        esimd::simd<float, COLS> acc(0.f);
        const float *pr = part + static_cast<size_t>(r) * H + c0;
        for (int k0 = 0; k0 < n; k0 += 8) {
          esimd::simd<float, COLS> pv[8];
          float wv[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int u = rs[k0 + j < n ? k0 + j : n - 1];
            pv[j] = esimd::block_load<float, COLS>(pr + static_cast<size_t>(u) * kM * H);
            wv[j] = k0 + j < n ? wrow[u] : 0.f;
          }
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if (wv[j] != 0.f) acc = pv[j] * wv[j] + acc;
        }
        {
          const float g = rt->shared_gate[r];
          const esimd::simd<float, COLS> p = esimd::block_load<float, COLS>(part + (static_cast<size_t>(slots) * kM + r) * H + c0);
          acc = p * g + acc;
        }
        if constexpr (SUM == CombineSum::owner) {  // The row's owner sums, the other cards send
          if (!mail::owner_sum<COLS>(acc, boxes, r, rows, H, c0, parity, kMoeMailEmpty, wire.wait_cycles, wire.error, opaque0)) return;
        } else if constexpr (SUM == CombineSum::quad) {  // Four cards, ((p0 + p1) + (p2 + p3)) on every card
          const size_t at = (static_cast<size_t>(parity) * kMoeRows + r) * H + c0;
          acc = mail::quad_sum<COLS>(acc, qp, at, hop_words, kMoeMailEmpty, wire.wait_cycles, wire.error);
        } else if constexpr (SUM == CombineSum::pair) {
          constexpr auto uc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                                esimd::cache_hint_L2<esimd::cache_hint::uncached>, esimd::alignment<256>};
          const size_t at = (static_cast<size_t>(parity) * kMoeRows + r) * H + c0;  // by capacity: shapes may alternate
          esimd::simd<uint32_t, COLS> mine = acc.template bit_cast_view<uint32_t>();
          mine.merge(esimd::simd<uint32_t, COLS>(0x7fc00000u), mine == kMoeMailEmpty);
          esimd::block_store<uint32_t, COLS>(wire.peer_inbox + at, mine, uc);
          // No fence here: the words are their own flags, so nothing orders this store against
          // the load below (a system-scope fence at this point waited for the PCIe write).
          esimd::simd<uint32_t, COLS> got = esimd::block_load<uint32_t, COLS>(wire.inbox + at, uc);
          bool ok = !(got == kMoeMailEmpty).any();
          if (!ok) {
            const uint64_t t0 = esimd::rdtsc();
            while (!ok) {
              esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
              got = esimd::block_load<uint32_t, COLS>(wire.inbox + at, uc);
              ok = !(got == kMoeMailEmpty).any();
              if (!ok && esimd::rdtsc() - t0 > wire.wait_cycles) break;
            }
          }
          if (!ok) {
            wire.error[0] = 1u;
            got = 0x7fc00000u;
          }
          esimd::block_store<uint32_t, COLS>(wire.inbox + at, esimd::simd<uint32_t, COLS>(kMoeMailEmpty), uc);
          const esimd::simd<float, COLS> other = got.template bit_cast_view<float>();
          acc = acc + other;
        }
        const esimd::simd<bf16, COLS> o = acc;
        o.copy_to(out + static_cast<size_t>(r) * H + c0);
      });
}

}  // namespace

MoeBlobLayout moe_tp_layout(int hidden, int inter, int cards, size_t *stride) {
  const size_t H = static_cast<size_t>(hidden), Ic = static_cast<size_t>(inter / cards);
  if (cards < 1 || inter % (kQuarter * cards) || hidden % kGroup)
    throw std::invalid_argument("moe_tp_layout: inter / cards must be a multiple of 32 (inter " + std::to_string(inter) + ", cards " + std::to_string(cards) + ")");
  MoeBlobLayout l;
  l.gu_codes = 0;
  l.dn_codes = H * 2 * Ic / 2;
  l.gu_scales = l.dn_codes + Ic * H / 2;
  l.dn_scales = l.gu_scales + (H / kGroup) * 2 * Ic * 2;
  // Down scales: one row per 128-k group the card's pieces touch (cards = 2: three rows each; 4: two).
  const size_t rows = static_cast<size_t>(moe_tp_scale_rows(inter, cards));
  const size_t bytes = l.dn_scales + rows * H * 2;
  if (stride) *stride = (bytes + 1023) / 1024 * 1024;  // 1 KiB: every block stays 1024-aligned
  return l;
}

void moe_split_tp(const MoeBlobLayout &fl, int hidden, int inter, int cards, int card, const uint8_t *full, const MoeBlobLayout &tl, uint8_t *out) {
  const int H = hidden, I = inter, Ic = inter / cards, G = H / kGroup, ti = I / kW, tic = Ic / kW, tiles = H / kW;
  const int PK = moe_tp_piece(Ic), NP = Ic / PK, PG = kGroup / PK;
  const size_t PB = static_cast<size_t>(PK) * kW4BlockBytes / kGroup;  // a piece's bytes
  // gate|up codes: per k group, this card's gate tiles then its up tiles.
  for (int g = 0; g < G; ++g)
    for (int proj = 0; proj < 2; ++proj)
      std::memcpy(out + tl.gu_codes + (static_cast<size_t>(g) * 2 * tic + proj * tic) * kW4BlockBytes,
                  full + fl.gu_codes + (static_cast<size_t>(g) * 2 * ti + proj * ti + card * tic) * kW4BlockBytes, static_cast<size_t>(tic) * kW4BlockBytes);
  // gate|up scales: [g][2 I] -> [g][2 Ic].
  for (int g = 0; g < G; ++g)
    for (int proj = 0; proj < 2; ++proj)
      std::memcpy(out + tl.gu_scales + (static_cast<size_t>(g) * 2 * Ic + proj * Ic) * 2, full + fl.gu_scales + (static_cast<size_t>(g) * 2 * I + proj * I + card * Ic) * 2,
                  static_cast<size_t>(Ic) * 2);
  // down: local piece h = global piece card * NP + h = piece (gp % PG) of the full block gp / PG (contiguous bytes: a
  // block's k rows are in order, 16 bytes per k pair).
  const int g0 = card * NP / PG;  // first 128-k group this card touches
  for (int h = 0; h < NP; ++h) {
    const int gp = card * NP + h, g = gp / PG;
    const size_t off = static_cast<size_t>(gp % PG) * PB;
    for (int t = 0; t < tiles; ++t)
      std::memcpy(out + tl.dn_codes + (static_cast<size_t>(h) * tiles + t) * PB, full + fl.dn_codes + (static_cast<size_t>(g) * tiles + t) * kW4BlockBytes + off, PB);
  }
  // every scale row the layout holds is written (rows past the full expert's groups as zeros): callers rely on the split
  // writing every byte up to dn_scales + moe_tp_scale_rows x H x 2
  const int groups = I / kGroup;
  for (int r = 0; r < moe_tp_scale_rows(inter, cards); ++r) {
    uint8_t *dst = out + tl.dn_scales + static_cast<size_t>(r) * H * 2;
    if (g0 + r < groups) std::memcpy(dst, full + fl.dn_scales + static_cast<size_t>(g0 + r) * H * 2, static_cast<size_t>(H) * 2);
    else std::memset(dst, 0, static_cast<size_t>(H) * 2);
  }
}

namespace {
template <int kM>
sycl::event route_shape(sycl::queue &q, const MoeShape &s, int rows, const float *logits, int ld, const uint8_t *const *blobs, MoeRoute *rt,
                        uint32_t *hits = nullptr, const bf16 *xin = nullptr, const uint8_t *cold = nullptr, const MoeCpuLink &cpu = {}) {
  // The CPU path and the key guards compiled only where needed; with the select-only ordered16 the 512-expert
  constexpr int kBig = kMoeMaxExperts / kRouteLanes;
  const bool full = s.experts == 32 * kRouteLanes || s.experts == kBig * kRouteLanes;
  auto go = [&](auto per, auto cpu_t, auto full_t) {
    return route<kM, decltype(per)::value, decltype(cpu_t)::value, decltype(full_t)::value>(q, s, rows, logits, ld, blobs, rt, hits, xin, cold, cpu);
  };
  using T = std::true_type;
  using F = std::false_type;
  using P32 = std::integral_constant<int, 32>;
  using PB = std::integral_constant<int, kBig>;
  if constexpr (kM <= kMoeCpuRows) {  // the CPU path's launches
    if (cpu.mail) {
      if (s.experts <= 32 * kRouteLanes) return full ? go(P32{}, T{}, T{}) : go(P32{}, T{}, F{});
      return full ? go(PB{}, T{}, T{}) : go(PB{}, T{}, F{});
    }
  }
  if (s.experts <= 32 * kRouteLanes) return full ? go(P32{}, F{}, T{}) : go(P32{}, F{}, F{});
  return full ? go(PB{}, F{}, T{}) : go(PB{}, F{}, F{});
}
}  // namespace

sycl::event moe_cpu_state_init(sycl::queue &q, uint32_t *state) {
  return q.single_task([=] {
    for (int i = 0; i < 16; ++i) state[i] = i == 1 ? 1u : 0u;
  });
}

sycl::event moe_route(sycl::queue &q, const MoeShape &s, int rows, const float *logits, int ld, const uint8_t *const *blobs, MoeRoute *rt) {
  if (rows < 1 || rows > kMoeRows) throw std::invalid_argument("moe_route: rows");
  const int sh = row_shape(rows);
  if (sh == kMaxShape) return route_shape<kMaxShape>(q, s, rows, logits, ld, blobs, rt);  // 64 rows
  if (sh == 32) return route_shape<32>(q, s, rows, logits, ld, blobs, rt);
  return sh == 4 ? route_shape<4>(q, s, rows, logits, ld, blobs, rt) : sh == 8 ? route_shape<8>(q, s, rows, logits, ld, blobs, rt) : route_shape<16>(q, s, rows, logits, ld, blobs, rt);
}

// [slot][row] blocks at the model's largest step (MoeShape:rows_cap; a launch lays them out by its own shape)
size_t moe_h_elems(const MoeShape &s) { return static_cast<size_t>(s.slots() + 1) * s.rows_cap * std::max(s.inter, s.shared_inter); }
size_t moe_hs_elems(const MoeShape &s) { return static_cast<size_t>(s.slots()) * s.rows_cap * std::max(1, s.inter_local() / kHalf); }
size_t moe_partial_elems(const MoeShape &s) { return static_cast<size_t>(s.slots() + 1) * s.rows_cap * s.hidden; }
size_t moe_wire_words(const MoeShape &s) { return 2ull * kMoeRows * s.hidden * wire_hops(s.cards); }

sycl::event moe_wire_init(sycl::queue &q, const MoeShape &s, uint32_t *inbox) { return q.fill(inbox, kMoeMailEmpty, moe_wire_words(s)); }

// ----: block Hadamard of bf16 rows ------------------------------------------------------------
namespace {
template <int N>
struct MoeHadamardK;
// One ESIMD thread per N-block, the whole block in registers, each butterfly stage two
// N/2-wide vector ops (was: one SYCL item per block, a serial loop over a private array: ~22 us per
// launch, 2.1 ms per decode step over 48 layers x 2 launches). The same butterflies (a + c, a - c on
// the same pairs, stage by stage) and the same final scale + bf16 rounding: bit-identical.
template <int N, int Hh>
inline void hadamard_stage(esimd::simd<float, N> &v) SYCL_ESIMD_FUNCTION {
  auto mv = v.template bit_cast_view<float, N / (2 * Hh), 2 * Hh>();
  const esimd::simd<float, N / 2> a = mv.template select<N / (2 * Hh), 1, Hh, 1>(0, 0);
  const esimd::simd<float, N / 2> c = mv.template select<N / (2 * Hh), 1, Hh, 1>(0, Hh);
  mv.template select<N / (2 * Hh), 1, Hh, 1>(0, 0) = a + c;
  mv.template select<N / (2 * Hh), 1, Hh, 1>(0, Hh) = a - c;
  if constexpr (2 * Hh < N) hadamard_stage<N, 2 * Hh>(v);
}
template <int N>
sycl::event hadamard_rows(sycl::queue &q, int rows, int cols, int ld, const bf16 *src, bf16 *dst) {
  const size_t nb = static_cast<size_t>(cols / N), items = static_cast<size_t>(rows) * nb;
  return q.parallel_for<MoeHadamardK<N>>(sycl::range<1>(items), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    const size_t r = id[0] / nb, b = id[0] % nb;
    const bf16 *sp = src + r * static_cast<size_t>(ld) + b * N;
    bf16 *dp = dst + r * static_cast<size_t>(ld) + b * N;
    esimd::simd<float, N> v = esimd::block_load<bf16, N>(sp);
    hadamard_stage<N, 1>(v);
    const float sc = N == 64 ? 0.125f : 0.08838834764831845f;  // 1 / sqrt(N)
    v *= sc;
    esimd::block_store<bf16, N>(dp, esimd::simd<bf16, N>(v));
  });
}
}  // namespace

sycl::event moe_hadamard_rows(sycl::queue &q, int rows, int cols, int ld, const bf16 *src, bf16 *dst, int n) {
  if (rows < 0 || cols % n || ld < cols) throw std::invalid_argument("moe_hadamard_rows: shape");
  if (n == 64) return hadamard_rows<64>(q, rows, cols, ld, src, dst);
  if (n == 128) return hadamard_rows<128>(q, rows, cols, ld, src, dst);
  throw std::invalid_argument("moe_hadamard_rows: n must be 64 or 128");
}

// ----: (rotate +) int8 quantise per N-block, the decode W4A8 activations -----------------
// One ESIMD thread per (row, N-block): ROT = bf16(H_N v) exactly as hadamard_rows (same butterflies,
// scale, bf16 rounding), then row_quant.hpp's rule on those bf16 values (s = amax / 127, q = rne(v x
// 127 / amax) clamped to +-127, an all-zero block q = 0, s = 0). The rotation block is the quant group
// (128 for x, 64 for h), so the vector stays in registers: one launch where W4A16 under rot had one.
// dst at stride ld (like src), scl [rows][cols / N].
namespace {
template <int N, bool ROT>
struct MoeRotQuantK;
template <int N, bool ROT>
sycl::event rot_quant_rows(sycl::queue &q, int rows, int cols, int ld, const bf16 *src, int8_t *dst, float *scl) {
  const size_t nb = static_cast<size_t>(cols / N), items = static_cast<size_t>(rows) * nb;
  return q.parallel_for<MoeRotQuantK<N, ROT>>(sycl::range<1>(items), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    rot_quant_item<N, ROT>(src, ld, dst, scl, id[0], nb);  // rot_quant.hpp
  });
}
template <int N>
sycl::event rot_quant(sycl::queue &q, bool rot, int rows, int cols, int ld, const bf16 *src, int8_t *dst, float *scl) {
  return rot ? rot_quant_rows<N, true>(q, rows, cols, ld, src, dst, scl) : rot_quant_rows<N, false>(q, rows, cols, ld, src, dst, scl);
}
}  // namespace
sycl::event moe_rot_quant_rows(sycl::queue &q, int rows, int cols, int ld, const bf16 *src, int8_t *dst, float *scl, int n, bool rot) {
  if (rows < 0 || cols % n || ld < cols) throw std::invalid_argument("moe_rot_quant_rows: shape");
  if (rows == 0) return sycl::event{};
  if (n == 64) return rot_quant<64>(q, rot, rows, cols, ld, src, dst, scl);
  if (n == 128) return rot_quant<128>(q, rot, rows, cols, ld, src, dst, scl);
  throw std::invalid_argument("moe_rot_quant_rows: n must be 64 or 128");
}
namespace {
template <int kM, bool A8, int MC = kM>
sycl::event gate_up_ks(sycl::queue &q, const MoeShape &s, const bf16 *x, const bf16 *xr, const MoeWeights &w, const MoeScratch &sc, int ks, int live = -1) {
  return ks >= 16  ? gate_up<kM, 16, A8, MC>(q, s, x, xr, w, sc, live)  // for 1-row steps (A/B)
         : ks >= 8 ? gate_up<kM, 8, A8, MC>(q, s, x, xr, w, sc, live)
         : ks >= 4 ? gate_up<kM, 4, A8, MC>(q, s, x, xr, w, sc, live)
         : ks >= 2 ? gate_up<kM, 2, A8, MC>(q, s, x, xr, w, sc, live)
                   : gate_up<kM, 1, A8, MC>(q, s, x, xr, w, sc, live);
}
template <int kM, int MC = kM>
sycl::event gate_up8g_ks(sycl::queue &q, const MoeShape &s, const bf16 *x, const MoeWeights &w, const MoeScratch &sc, int ks) {
  return ks >= 8   ? gate_up8g<kM, 8, MC>(q, s, x, w, sc)
         : ks >= 4 ? gate_up8g<kM, 4, MC>(q, s, x, w, sc)
         : ks >= 2 ? gate_up8g<kM, 2, MC>(q, s, x, w, sc)
                   : gate_up8g<kM, 1, MC>(q, s, x, w, sc);
}
// A step of this many rows or more runs the quarter-piece down in tile pairs (down, TW = 2): the 16- and 32-row
// shapes. From 5 rows the 8-row shape reads the same as without this split, and a 1-row step loses.
constexpr int kDownPairMinRows = 9;

template <int kM, bool A8, bool FH = false, int MC = kM>
sycl::event down_ks(sycl::queue &q, const MoeShape &s, const MoeWeights &w, const MoeScratch &sc, int dks, int live = -1, int rows = 0) {
  auto by_ks = [&](auto pk) {
    constexpr int PK = decltype(pk)::value;
    return dks >= 4 ? down<kM, 4, A8, FH, MC, PK>(q, s, w, sc, live) : dks >= 2 ? down<kM, 2, A8, FH, MC, PK>(q, s, w, sc, live)
                                                                         : down<kM, 1, A8, FH, MC, PK>(q, s, w, sc, live);
  };
  if constexpr (!A8) {  // TP4's 160 columns per card come in 32-k quarter blocks
    if (s.tp && moe_tp_piece(s.inter_local()) == kQuarter) {
      if (dks < 2 && rows >= kDownPairMinRows) return down<kM, 1, A8, FH, MC, kQuarter, 2>(q, s, w, sc, live);
      return by_ks(std::integral_constant<int, kQuarter>{});
    }
  }
  return by_ks(std::integral_constant<int, kHalf>{});
}
}  // namespace

namespace {
// The combine with the step's sum: local (one card), the owner sum, the four-card or the two-card wire.
template <int kM>
sycl::event combine_by_wire(sycl::queue &q, const MoeShape &s, int rows, const MoeScratch &sc, const MoeWire *wire, int parity, bf16 *out) {
  if (!wire) return combine<kM, CombineSum::local>(q, s, rows, sc, MoeWire{}, 0, out);
  if (wire->owner.on()) return combine<kM, CombineSum::owner>(q, s, rows, sc, *wire, parity & 1, out);
  if (wire->quad.on()) return combine<kM, CombineSum::quad>(q, s, rows, sc, *wire, parity & 1, out);
  return combine<kM, CombineSum::pair>(q, s, rows, sc, *wire, parity & 1, out);
}

template <int kM>
MoeEvents decode_shape(sycl::queue &q, const MoeShape &s, int rows, const bf16 *x, const MoeWeights &w, const MoeScratch &sc, const MoeWire *wire,
                       int parity, bf16 *out, int gu_ksplit, int dn_ksplit, const MoeCpuLink *cpu) {
  MoeEvents e;
  const bool fuse8 = s.a8 && s.a8_fuse;
  if (fuse8) {  // x's rotate + quantise in the router's launch
    RotQuantJob j;
    j.src = x, j.dst = sc.xq, j.scl = sc.xs, j.rows = kM, j.cols = s.hidden, j.ld = s.hidden, j.rot = s.rot;
    e.router = dense_gemm_rq(q, rows, x, w.router, sc.logits, j);
  } else {
    e.router = dense_gemm(q, rows, x, w.router, sc.logits);
  }
  if (sc.stamps) stamp(q, sc.stamps);
  const MoeCpuLink link = cpu ? *cpu : MoeCpuLink{};
  e.route = route_shape<kM>(q, s, rows, sc.logits, w.router.n, w.blobs, sc.route, sc.hits, x, w.cold, link);
  if (sc.stamps) stamp(q, sc.stamps + 1);
  const int ks = gu_ksplit > 0 ? gu_ksplit : kGateUpKSplit;
  // TP4's 160-column quarters read no faster with another split; gate_up ks 4 / 16 are no better than 8.
  const int dks = dn_ksplit > 0 ? dn_ksplit : s.tp && moe_tp_piece(s.inter_local()) == kQuarter ? 1 : kDownKSplit;
  const int HW = std::max(s.inter, s.shared_inter), hrows = kM * s.topk * kM;  // the routed slots x kM rows
  // A 1-2-row step runs the W4A8 gate_up / down at compute shape 2 (the h / partial layout keeps kM rows; rows >= 2
  // are not written), bit-identical to shape 8. Shape 1 (s8 x s4 DPAS repeat count 1, 2 and 3) changed row 0's bits,
  // unexplained, so it is not used. The 32-row shape's h epilogue would need 128 KiB of SLM at k split 8: h by its
  // own launch (rot_quant: the epilogue's bits).
  bool h8g = false;
  if constexpr (kM <= 16) {
    h8g = true;
    if (kM == 4 && fuse8 && s.a8_h_in_gate_up && rows <= 2) {
      e.gate_up = gate_up8g_ks<kM, 2>(q, s, x, w, sc, ks);
      e.down = down_ks<kM, true, false, 2>(q, s, w, sc, dks);
    } else if (fuse8 && s.a8_h_in_gate_up) {  // x in the router's launch, h in gate_up's epilogue: no extra launch
      e.gate_up = gate_up8g_ks<kM>(q, s, x, w, sc, ks);
      e.down = down_ks<kM, true>(q, s, w, sc, dks);
    } else {
      h8g = false;
    }
  }
  if (h8g) {
  } else if (fuse8) {  // x quantised by the router's launch; h by its own launch or inside down
    e.gate_up = gate_up_ks<kM, true>(q, s, x, x, w, sc, ks);
    if (s.a8_h_in_down && kM <= 16) {
      e.down = down_ks<kM, true, true>(q, s, w, sc, dks);
    } else {
      rot_quant<kHalf>(q, s.rot, hrows, s.inter_local(), HW, sc.h, sc.hq, sc.hs);
      e.down = down_ks<kM, true>(q, s, w, sc, dks);
    }
  } else if (s.a8) {  // x (rotated under rot) -> int8 per 128-group, all kM rows; h likewise per 64-group
    rot_quant<kGroup>(q, s.rot, kM, s.hidden, s.hidden, x, sc.xq, sc.xs);
    e.gate_up = gate_up_ks<kM, true>(q, s, x, x, w, sc, ks);
    rot_quant<kHalf>(q, s.rot, hrows, s.inter_local(), HW, sc.h, sc.hq, sc.hs);
    e.down = down_ks<kM, true>(q, s, w, sc, dks);
  } else {
    const bf16 *xr = x;
    if (s.rot) {  // the routed experts' input in the rotated basis
      moe_hadamard_rows(q, rows, s.hidden, s.hidden, x, sc.xr, kGroup);
      xr = sc.xr;
    }
    // The step's live slots only (rows x topk; a 1-row step used to launch 4 x topk), and a 1- / 2-row step
    // computes 1 / 2 rows (DPAS repeat count), bit-identical.
    const int live = std::min(rows, kM) * s.topk;
    // 5-7 rows on the 8-row shape likewise
    const int mc = (kM == 4 && rows <= 2) || (kM == 8 && rows >= 5 && rows <= 7) ? rows : 0;
    bool done = false;
    if constexpr (kM == 4) {
      if (mc == 1) e.gate_up = gate_up_ks<4, false, 1>(q, s, x, xr, w, sc, ks, live), done = true;
      if (mc == 2) e.gate_up = gate_up_ks<4, false, 2>(q, s, x, xr, w, sc, ks, live), done = true;
    } else if constexpr (kM == 8) {
      if (mc == 5) e.gate_up = gate_up_ks<8, false, 5>(q, s, x, xr, w, sc, ks, live), done = true;
      if (mc == 6) e.gate_up = gate_up_ks<8, false, 6>(q, s, x, xr, w, sc, ks, live), done = true;
      if (mc == 7) e.gate_up = gate_up_ks<8, false, 7>(q, s, x, xr, w, sc, ks, live), done = true;
    }
    if (!done) e.gate_up = gate_up_ks<kM, false>(q, s, x, xr, w, sc, ks, live);
    if (s.rot)  // the routed slots' h, this card's inter_local columns ([slot][row][max(inter, shared_inter)])
      moe_hadamard_rows(q, hrows, s.inter_local(), HW, sc.h, sc.h, kHalf);
    done = false;
    if constexpr (kM == 4) {
      if (mc == 1) e.down = down_ks<4, false, false, 1>(q, s, w, sc, dks, live, rows), done = true;
      if (mc == 2) e.down = down_ks<4, false, false, 2>(q, s, w, sc, dks, live, rows), done = true;
    } else if constexpr (kM == 8) {
      if (mc == 5) e.down = down_ks<8, false, false, 5>(q, s, w, sc, dks, live, rows), done = true;
      if (mc == 6) e.down = down_ks<8, false, false, 6>(q, s, w, sc, dks, live, rows), done = true;
      if (mc == 7) e.down = down_ks<8, false, false, 7>(q, s, w, sc, dks, live, rows), done = true;
    }
    if (!done) e.down = down_ks<kM, false>(q, s, w, sc, dks, live, rows);
    if (link.mail) e.cpu_wait = cpu_wait<kM>(q, s, x, w, sc, link);  // The pool's experts into the partials
  }
  if (sc.stamps) stamp(q, sc.stamps + 2);
  e.combine = combine_by_wire<kM>(q, s, rows, sc, wire, parity, out);
  return e;
}

// A step of 33-64 rows on the 64-row shape (bf16 x; moe_decode has refused rot, a8 and the CPU path).
// The router runs as two 32-row launches (dense_gemm); route, gate_up, down and the combine are one launch each over the
// step's unique experts, which is where a 64-row step saves against two 32-row ones (202 against 2 x 148 unique
// experts per layer at 16 streams x 4 rows, recorded routes).
MoeEvents decode_wide(sycl::queue &q, const MoeShape &s, int rows, const bf16 *x, const MoeWeights &w, const MoeScratch &sc, const MoeWire *wire, int parity,
                      bf16 *out, int gu_ksplit, int dn_ksplit) {
  constexpr int kM = kMaxShape;
  MoeEvents e;
  e.router = dense_gemm(q, rows, x, w.router, sc.logits);
  if (sc.stamps) stamp(q, sc.stamps);
  e.route = route_shape<kM>(q, s, rows, sc.logits, w.router.n, w.blobs, sc.route, sc.hits);
  if (sc.stamps) stamp(q, sc.stamps + 1);
  const int ks = gu_ksplit > 0 ? gu_ksplit : kGateUpKSplit;
  const int dks = dn_ksplit > 0 ? dn_ksplit : s.tp && moe_tp_piece(s.inter_local()) == kQuarter ? 1 : kDownKSplit;
  const int live = std::min(rows * s.topk, s.per_card());  // the route lists no more slots than the card has experts
  e.gate_up = gate_up_ks<kM, false>(q, s, x, x, w, sc, ks, live);
  e.down = down_ks<kM, false>(q, s, w, sc, dks, live, rows);
  if (sc.stamps) stamp(q, sc.stamps + 2);
  e.combine = combine_by_wire<kM>(q, s, rows, sc, wire, parity, out);
  return e;
}
}  // namespace

MoeEvents moe_decode(sycl::queue &q, const MoeShape &s, int rows, const bf16 *x, const MoeWeights &w, const MoeScratch &sc, const MoeWire *wire,
                     int parity, bf16 *out, int gu_ksplit, int dn_ksplit, const MoeCpuLink *cpu) {
  if (rows < 1 || rows > kMoeRows || rows > s.rows_cap)
    throw std::invalid_argument("moe_decode: " + std::to_string(rows) + " rows, the scratch holds steps of 1.." + std::to_string(std::min(s.rows_cap, kMoeRows)) +
                                " (MoeShape::rows_cap)");
  if (rows > kStepRows32 && (s.rot || s.a8 || (cpu && cpu->mail)))
    throw std::invalid_argument("moe_decode: a step above 32 rows runs bf16 x only (no rot, no a8) and without the CPU path");
  if (s.rot && !s.a8 && (!sc.xr || s.hidden % kGroup || s.inter_local() % kHalf)) throw std::invalid_argument("moe_decode: rot needs sc.xr and 128 / 64-aligned shapes");
  if (s.a8 && (!s.tp || !sc.xq || !sc.xs || !sc.hq || !sc.hs || s.hidden % kGroup || s.inter_local() % kHalf))
    throw std::invalid_argument("moe_decode: a8 needs tp, sc.xq / xs / hq / hs and 128 / 64-aligned shapes");
  if ((s.cards != 1 && s.cards != 2 && s.cards != kWireCardsMax) || s.card < 0 || s.card >= s.cards || s.experts % s.cards ||
      s.experts > kMoeMaxExperts || s.topk < 1 || s.topk > kMoeMaxTopk || s.experts % 4)
    throw std::invalid_argument("moe_decode: shape");
  if (s.cards == kWireCardsMax && !s.tp) throw std::invalid_argument("moe_decode: four cards run the expert TP layout only (no EP4)");
  if (s.hidden % kGroup || s.inter % kGroup || s.shared_inter % (kKCall * s.cards) || (s.shared_inter / kW) % s.cards ||
      s.hidden % kCombineCols)
    throw std::invalid_argument("moe_decode: hidden and inter must be multiples of 128 (shared inter split by 16-k calls)");
  if (w.router.n < s.experts + 1 || w.router.k != s.hidden || w.router.kind != DenseKind::bf16)
    throw std::invalid_argument("moe_decode: router must be bf16 [experts + 1][hidden]");
  if (w.shared_gu.kind != DenseKind::i8 || w.shared_gu.n != 2 * s.shared_inter || w.shared_gu.k != s.hidden ||
      w.shared_down.kind != DenseKind::i8 || w.shared_down.n != s.hidden || w.shared_down.k != s.shared_inter)
    throw std::invalid_argument("moe_decode: shared expert must be int8 gate|up [2 si][hidden] and down [hidden][si]");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("moe_decode: a wire is needed exactly with more than one card");
  if (wire && (wire->quad.on() != (s.cards == kWireCardsMax) || (wire->quad.on() && !wire->quad.complete())))
    throw std::invalid_argument("moe_decode: four cards take the four-card wire (MoeWire::quad, every card's inbox), two cards the pair wire");
  // the owner mailbox holds kOwnerRowsMax rows per card, kPairRows / 4: the quad wire carries a step
  // above kPairRows rows until that capacity grows)
  if (wire && wire->owner.on() && rows > kPairRows)
    throw std::invalid_argument("moe_decode: an owner sum carries steps of up to " + std::to_string(kPairRows) + " rows, got " + std::to_string(rows));
  if (wire && wire->owner.on() && (s.cards != kWireCardsMax || !wire->owner.complete() || wire->owner.card != s.card || rows < kOwnerMinRows))
    throw std::invalid_argument("moe_decode: an owner sum (MoeWire::owner) needs four cards, this card's index, every card's inbox and 4+ rows");
  if (s.tp && s.inter % (kQuarter * s.cards))
    throw std::invalid_argument("moe_decode: tp needs inter a multiple of 32 x cards");
  if (cpu && cpu->mail &&
      (!s.tp || s.rot || s.a8 || !w.cold || !cpu->x || !cpu->y || !cpu->state || s.inter_local() > kWaitHMax || s.inter_local() % kHalf || s.hidden % 64 ||
       kMoeCpuRows * s.topk > kMoeCpuMax))
    throw std::invalid_argument("moe_decode: the CPU path needs tp, no rot / a8, MoeWeights::cold, the link's buffers, inter_local <= 1024 and 8 x topk <= kMoeCpuMax");
  const int sh = row_shape(rows);
  if (sh == 4) return decode_shape<4>(q, s, rows, x, w, sc, wire, parity, out, gu_ksplit, dn_ksplit, cpu);
  if (sh == 8) return decode_shape<8>(q, s, rows, x, w, sc, wire, parity, out, gu_ksplit, dn_ksplit, cpu);
  if (sh == 16) return decode_shape<16>(q, s, rows, x, w, sc, wire, parity, out, gu_ksplit, dn_ksplit, cpu);
  if (sh == 32) return decode_shape<32>(q, s, rows, x, w, sc, wire, parity, out, gu_ksplit, dn_ksplit, cpu);
  return decode_wide(q, s, rows, x, w, sc, wire, parity, out, gu_ksplit, dn_ksplit);  // 64 rows
}

// ---- host reference -------------------------------------------------------------------------------
namespace {
float bf(float v) { return static_cast<float>(bf16(v)); }
}  // namespace

void moe_route_ref(const MoeShape &s, int rows, const float *x, const MoeRefWeights &w, MoeRefOut &o) {
  const int E = s.experts, H = s.hidden, K = s.topk;
  o.ids.assign(static_cast<size_t>(rows) * K, 0);
  o.topw.assign(static_cast<size_t>(rows) * K, 0.f);
  o.logits.assign(static_cast<size_t>(rows) * E, 0.f);
  for (int r = 0; r < rows; ++r) {
    std::vector<std::pair<float, int>> v(E);
    for (int e = 0; e < E; ++e) {
      double acc = 0;
      for (int k = 0; k < H; ++k) acc += static_cast<double>(x[static_cast<size_t>(r) * H + k]) * w.router[static_cast<size_t>(e) * H + k];
      const float l = bf(static_cast<float>(acc));
      o.logits[static_cast<size_t>(r) * E + e] = l;
      v[e] = {l == 0.f ? 0.f : l, e};  // -0 == +0
    }
    std::stable_sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
      const bool an = std::isnan(a.first), bn = std::isnan(b.first);
      if (an != bn) return bn;  // NaN last
      return a.first > b.first;
    });
    float z = 0.f;
    std::vector<float> ex(K);
    for (int j = 0; j < K; ++j) {
      ex[j] = std::exp(v[j].first - v[0].first);
      z += ex[j];
    }
    for (int j = 0; j < K; ++j) {
      o.ids[static_cast<size_t>(r) * K + j] = v[j].second;
      o.topw[static_cast<size_t>(r) * K + j] = ex[j] / z;
    }
  }
}

void moe_ref(const MoeShape &s, int rows, const float *x, const MoeRefWeights &w, MoeRefOut &o, const MoeRefH *force) {
  moe_route_ref(s, rows, x, w, o);
  o.h.clear();
  const int H = s.hidden, I = s.inter, SI = s.shared_inter, K = s.topk, C = s.cards, per = s.experts / s.cards;
  o.out.assign(static_cast<size_t>(rows) * H, 0.f);
  auto silu = [](float g) { return g / (1.f + std::exp(-g)); };
  // block Hadamard (host), bf16-rounded like the device
  auto hadamard_bf = [](float *v, int len, int n) {
    for (int b = 0; b < len; b += n) {
      float *p = v + b;
      for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += h << 1)
          for (int j = i; j < i + h; ++j) {
            const float a = p[j], c = p[j + h];
            p[j] = a + c, p[j + h] = a - c;
          }
      const float sc = n == 64 ? 0.125f : 0.08838834764831845f;
      for (int i = 0; i < n; ++i) p[i] = bf(p[i] * sc);
    }
  };
  for (int r = 0; r < rows; ++r) {
    const float *xr = x + static_cast<size_t>(r) * H;
    std::vector<float> xrot;
    const float *xe = xr;  // the routed experts' input
    if (s.rot) {
      xrot.assign(xr, xr + H);
      hadamard_bf(xrot.data(), H, 128);
      xe = xrot.data();
    }
    double sgl = 0;
    for (int k = 0; k < H; ++k) sgl += static_cast<double>(xr[k]) * w.shared_gate[k];
    const float sgate = 1.f / (1.f + std::exp(-bf(static_cast<float>(sgl))));
    std::vector<float> total(H, 0.f);
    for (int c = 0; c < C; ++c) {
      std::vector<float> p(H, 0.f);
      // Routed: the device's list order, ascending expert id among this card's experts.
      std::vector<int> order;
      for (int rr = 0; rr < rows; ++rr)
        for (int j = 0; j < K; ++j) {
          const int e = o.ids[static_cast<size_t>(rr) * K + j];
          if ((s.tp || e / per == c) && std::find(order.begin(), order.end(), e) == order.end()) order.push_back(e);
        }
      std::sort(order.begin(), order.end());
      for (const int e : order) {
        int slot = -1;
        for (int j = 0; j < K; ++j)
          if (o.ids[static_cast<size_t>(r) * K + j] == e) slot = j;
        if (slot < 0) continue;
        const MoeRefExpert *ex = w.experts.at(e);
        if (!ex) throw std::runtime_error("moe_ref: expert not provided");
        std::vector<float> h(I);
        for (int n = 0; n < I; ++n) {
          double g = 0, u = 0;
          for (int k = 0; k < H; ++k) {
            g += static_cast<double>(xe[k]) * ex->gate[static_cast<size_t>(k) * I + n];
            u += static_cast<double>(xe[k]) * ex->up[static_cast<size_t>(k) * I + n];
          }
          const float gb = bf(static_cast<float>(g)), ub = bf(static_cast<float>(u));
          h[n] = bf(silu(gb) * ub);
        }
        if (s.rot) hadamard_bf(h.data(), I, 64);  // 64-blocks: each card's TP slice is block-aligned
        o.h[{r, e}] = h;
        if (force) h = force->at({r, e});
        const float wv = o.topw[static_cast<size_t>(r) * K + slot];
        for (int n = 0; n < H; ++n) {
          double y = 0;
          const int k0 = s.tp ? c * I / C : 0, k1 = s.tp ? (c + 1) * I / C : I;  // tp: this card's k slice
          for (int k = k0; k < k1; ++k) y += static_cast<double>(h[k]) * ex->down[static_cast<size_t>(k) * H + n];
          const float yf = s.down_bf16 ? bf(static_cast<float>(y)) : static_cast<float>(y);
          p[n] = std::fma(yf, wv, p[n]);
        }
      }
      // Shared: this card's columns (a fixed 1/C, in 16-column tiles as the device's route).
      const int T = SI / 16, i0 = c * T / C * 16, i1 = (c + 1) * T / C * 16;
      std::vector<float> h(SI, 0.f);
      for (int n = i0; n < i1; ++n) {
        double g = 0, u = 0;
        for (int k = 0; k < H; ++k) {
          g += static_cast<double>(xr[k]) * w.s_gate[static_cast<size_t>(n) * H + k];
          u += static_cast<double>(xr[k]) * w.s_up[static_cast<size_t>(n) * H + k];
        }
        h[n] = bf(silu(bf(static_cast<float>(g))) * bf(static_cast<float>(u)));
      }
      o.h[{r, -1 - c}] = h;
      if (force) h = force->at({r, -1 - c});
      for (int n = 0; n < H; ++n) {
        double y = 0;
        for (int k = i0; k < i1; ++k) y += static_cast<double>(h[k]) * w.s_down[static_cast<size_t>(n) * SI + k];
        p[n] = std::fma(static_cast<float>(y), sgate, p[n]);
      }
      for (int n = 0; n < H; ++n) total[n] += p[n];
    }
    for (int n = 0; n < H; ++n) o.out[static_cast<size_t>(r) * H + n] = bf(total[n]);
  }
}

}  // namespace xenithium:kernels
