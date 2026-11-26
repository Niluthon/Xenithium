// Xenithium · the many-row gated GQA attention, one work-item per (q head, 8 rows) with all 256 dims of o (prefill,
// see gqa.hpp, gqa_attn_fa).
//
// The staged body (gqa_staged.cpp) gives each (8 pairs) block two items, one per half of the 256 o dims; both
// compute the full q.k and the online softmax of the same 8 pairs, so a third of its DPAS and half of its softmax are
// duplicated, and a staged K / V tile serves only 16 rows x G heads. Here item i of a work-group of QB / 8 x G items owns
// the pairs (head i % G, rows 8 (i / G) .. + 7) and all 256 dims of their o (128 GRF of fp32 accumulators):
//   - per 64-token tile (one KV page of this kv head) the items stage K and V once into SLM as bf16 DPAS operands
//     (int8 codes widen exactly; K dword-transposed for q.k, V token-paired for P.V; 64 KB);
//   - each item computes the full 256-dim q.k of its 8 pairs (DPAS over 16-dim steps, q in registers), their online
//     softmax, and P.V on all 256 dims;
//   - the o rescale is skipped on a tile where no pair's running max moved (exp(0) = 1: the same bits).
// Every pair's arithmetic is the staged body's in the same order (q.k over dc 0..15 per 16-token block, the softmax per
// pair, P = hi + lo bf16 x the V scale, o accumulated sub by sub, tile by tile), so the output is bit-identical to
// gqa_attn_staged; only the work split differs. p_lo = false drops the lo half of P (P.V on bf16 P alone): a numerics
// switch (half the P.V DPAS), off by default.
// Int8 q.k on the int8 KV (QK > 0, a numerics switch). q is quantised per (row, q
// head) when the item loads it (s = amax / 127, rne, row_quant.hpp's rule), K is staged as its raw codes in s8 DPAS B
// operands (32 dims x 16 tokens = one dword transpose of the 16 x 32-byte block, no widen; 16 KB), and q.k runs as s8 x s8
// DPAS (K = 32 per call: half the calls and half the SLM operand bytes of the bf16 form); S = float(sum) x (s_q / 16) x
// the K scale. QK 1: q as one int8 code per element. QK 2: q as hi + lo codes (q / s = hi + lo / 127, |lo| <= 64; the
// lo codes in SLM, 2 KB per item: in registers the body would spill), two DPAS per B operand into two int32 sums,
// S = float(hi sum) + float(lo sum) / 127: q.k within fp32-ulp noise of the bf16 form.
// Every QK keeps 40 rows per group. QK 2 is built with P hi only (with P as hi + lo it spills). Int8 KV only.
// The page-selected instances (SEL; gqa.hpp, GqaPageSets). A row block walks its selected pages
// in ascending order instead of every tile of its context; a tile's arithmetic is the dense body's (the causal and
// range masks take any tile), so a block that keeps every page computes the dense body's values.
// Its own TU at 256 GRF (xenithium_kernels_xl).
#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>

#include <cmath>
#include <stdexcept>

#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/rt/card.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = gqa_bf16;

constexpr int kTile = kQsaTile, kPage = kQsaPage;
static_assert(kTile == 64 && kPage == 64, "a staged tile is one KV page");
constexpr int kOp = 256;                     // bf16 elements of one B operand (16 k x 16 n)
constexpr int kOpBytes = kOp * 2;
constexpr int kSubs = kTile / 16;            // 16-token sub-tiles per tile
constexpr int kKOff = 0;                     // K operands [dc][sub]
constexpr int kOp8 = 512;                    // bytes of one s8 B operand (32 dims x 16 tokens)

// D: head_dim (256: qwen35 with the sigmoid output gate; 128: arch llama, no gate, q.k scale 1 / sqrt(128)); the
// per-D sizes below follow it (SLM 64 KB at D 256, 32 KB at 128; o accumulators D / 2 GRF)
template <QsaKv KV, int G, int QB, bool LO, int QK, int D, bool FL, bool SEL, bool TRAINED = false, bool SKIP = false>
class GqaAttnFaK;

// 16 tokens x 16 dims bf16 (row-major [tok][dim]) -> the q.k B operand (k = dims, n = tokens): dword pairs transposed
inline esimd::simd<bf16, kOp> k_operand(esimd::simd<bf16, kOp> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, 128> kd = v.template bit_cast_view<uint32_t>();
  esimd::simd<uint32_t, 128> bd;
#pragma unroll
  for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
  return bd.template bit_cast_view<bf16>();
}
// 16 tokens x 16 dims -> the P.V B operand (k = tokens, n = dims): token pairs interleaved
inline esimd::simd<bf16, kOp> v_operand(esimd::simd<bf16, kOp> v) SYCL_ESIMD_FUNCTION {
  esimd::simd<bf16, kOp> bv;
#pragma unroll
  for (int qq = 0; qq < 8; ++qq) {
    bv.template select<16, 2>(qq * 32) = v.template select<16, 1>(2 * qq * 16);
    bv.template select<16, 2>(qq * 32 + 1) = v.template select<16, 1>((2 * qq + 1) * 16);
  }
  return bv;
}

// FL (arch flint): the layer's window (GqaShape::window: a row attends to its last `win` tokens; the block walks
// the tiles from its first row's window on, a per-row mask), the KV ring (GqaCache::ring_pages: logical page t % ring) and
// the learned sinks (one logit per q head in the denominator: a = o e / (l e + exp(sink - M)), M = max(m, sink),
// e = exp(m - M): the merge's expression). A tile where a row has no live token leaves its state (the running max stays
// -inf until its first live tile). false = the other archs' instances.
// Tail split (wave quantisation): the call's W = blocks x hkv work-groups run in waves of C (one work-group per
// Xe core at 256 GRF); W = 104 at 2,048 rows on 32 cores is 3.25 waves, and the 4th wave would run 75 % idle.
// Plan tail: the last `tail` work-groups each run as
// `split` units over contiguous ranges of the block's KV tiles (unit groups after the W - tail whole ones), writing the
// unnormalised o, the running max and sum per item to tail_buf; attn_fa_merge combines them (the online-softmax merge)
// and applies the same o / l, bf16 and gate roundings. Whole groups keep their bits. tail = 0: no split.
// a short call's split: up to 16 units per work-group; a tail's remainder keeps 8
inline constexpr int kFaSplitMax = 16;
// a decode call (gqa_attn_fa_decode) on this thread: the slot's page row from a device word, its own tail buffer
struct FaDec { const int32_t *slot_word = nullptr; int slot_pages = 0; bool on = false; };
thread_local FaDec g_fa_dec;
struct FaTail {
  int tail = 0, split = 1;
  float *buf = nullptr;  // [tail][split][NI][8 x D o + 8 m + 8 l] fp32
};

template <QsaKv KV, int G, int QB, bool LO, int QK = 0, int D = 256, bool FL = false, bool SEL = false, bool TRAINED = false, bool SKIP = false>
sycl::event attn_fa(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy, const bf16 *qb,
                    bf16 *attn, const bf16 *sinks = nullptr, FaTail tp = {}, const GqaPageSets *sets = nullptr,
                    GqaTrainedPrefill trained = {}) {
  static_assert(!SKIP || (TRAINED && QK==1 && QB==40));
  static_assert(!TRAINED || (KV == QsaKv::i8v4 && !FL && !SEL && D == 256 && G == 6));
  static_assert(D == 256 || D == 128, "head_dim 256 (gated) or 128");
  static_assert(!SEL || (D == 256 && !FL), "page sets: head_dim 256, no window / ring");
  constexpr int kD = D;
  constexpr bool kGate = D == 256;                    // qwen35's sigmoid output gate; llama has none
  constexpr int kDc = kD / 16;                        // 16-dim chunks per head
  constexpr int kVOff = kDc * kSubs * kOpBytes;       // V operands [sub][dc]
  constexpr int kSlm = 2 * kDc * kSubs * kOpBytes;    // 64 KB (D 256), 32 KB (D 128)
  constexpr int kC8 = kD / 32;                        // 32-dim s8 calls per head (QK > 0)
  constexpr float kmul = D == 256 ? 1.f / 16.f : 0.08838834764831845f;  // 1 / sqrt(D)
  constexpr bool V4 = KV == QsaKv::i8v4;  // K int8, V int4 codes + a bf16 scale per 32 dims (head_dim 256)
  constexpr bool I8 = KV != QsaKv::bf16;  // K int8 (V int8 unless V4)
  static_assert(!V4 || (D == 256 && !FL), "int8-v4 KV: head_dim 256, no window");
  static_assert(QK == 0 || I8, "int8 q.k reads the int8 KV codes");
  static_assert(QK >= 0 && QK <= 2, "QK 0 / 1 / 2");
  // SLM: K operands (bf16 [dc][sub], 32 KB; QK > 0: s8 [call][sub], 16 KB), V operands (bf16 [sub][dc], 32 KB), QK 2: the
  // items' lo q codes (2 KB each)
  constexpr int kKOffQ = 0, kVOffQ = QK ? kC8 * kSubs * kOp8 : kVOff;
  constexpr int kQlOff = kVOffQ + kSubs * kDc * kOpBytes;
  static_assert(QB % 8 == 0, "rows per work-group: a multiple of 8");
  constexpr int NI = (QB / 8) * G;  // items: (8-row group, q head)
  static_assert(NI <= 32, "ESIMD work-groups stay at <= 32 items");
  constexpr int kSlmQ = QK == 0 ? kSlm : kQlOff + (QK == 2 ? NI * kC8 * 256 : 0);
  constexpr int kE = I8 ? 1 : 2;           // KV element bytes
  constexpr int kUw = 64 / kE;             // dims per staging unit (a 16-token x 64-byte 2D block)
  constexpr int kUdc = kUw / 16;           // 16-dim chunks per unit
  constexpr int kUnits = kSubs * (kD / kUw);  // units per K (and per V) tile
  const int hq = s.hq(), hkv = s.hkv(), blocks = (m + QB - 1) / QB;
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages, *slot_w = g_fa_dec.slot_word;
  const int slot_pg = g_fa_dec.slot_pages;
  const int win = FL ? s.window : 0, ring = FL ? c.ring_pages : 0;
  const int W = blocks * hkv, tail = FL ? 0 : tp.tail, split = tail ? tp.split : 1, whole = W - tail;
  float *tbuf = tp.buf;
  const int32_t *sel_pages = SEL ? sets->pages : nullptr, *sel_counts = SEL ? sets->counts : nullptr;  // [block][kv head]..., as g below
  constexpr int kPart = 8 * kD + 16;  // floats per item's partial: o [dc][row][16], m [8], l [8]
  return q.parallel_for<GqaAttnFaK<KV, G, QB, LO, QK, D, FL, SEL, TRAINED, SKIP>>(
      sycl::nd_range<1>(static_cast<size_t>(whole + tail * split) * NI, NI), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlmQ>();
        const int gg = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0));
        const bool part = gg >= whole;  // a tail unit (uniform per work-group)
        const int tu = part ? (gg - whole) / split : 0, sp = part ? (gg - whole) % split : 0;
        const int g = part ? whole + tu : gg;
        const int blk = g / hkv, kvh = g % hkv, r0 = blk * QB;
        const int nrows = m - r0 < QB ? m - r0 : QB;
        const int p0 = pos_word[0] + r0;  // position of the block's first row
        const int32_t *pages = slot_w ? pages0 + static_cast<size_t>(slot_w[0]) * slot_pg : pages0;
        const int ntok = p0 + nrows;
        int ntiles = (ntok + kTile - 1) / kTile;
        // FL: the first tile any row of the block attends to (its first row's window), the ring's logical page
        int tlo = 0;
        if constexpr (FL) tlo = win > 0 && p0 + 1 > win ? (p0 + 1 - win) / kTile : 0;
        // The walk: indexes tlo .. ntiles - 1, index w = tile w - or, SEL with a page set for this (block, kv head), the
        // set's w-th page (ascending, so the block's own pages come last, as in the dense walk).
        const int32_t *walk = nullptr;
        if constexpr (SEL) {
          const int count = sel_counts[g];
          if (count > 0) walk = sel_pages + static_cast<size_t>(g) * kGqaSelMax, ntiles = count;
        }
        if constexpr (TRAINED) {
          // Share physical K/V staging, never share a query's selected masks.
          walk = trained.pages + static_cast<size_t>(blk) * trained.pitch;
          ntiles = trained.counts[blk];
        }
        if (part) {  // this unit's contiguous share of the block's walk
          const int nt_all = ntiles;
          tlo = nt_all * sp / split;
          ntiles = nt_all * (sp + 1) / split;
        }
        auto tile_at = [&](int w) SYCL_ESIMD_FUNCTION {
          if constexpr (SEL || TRAINED) {
            if (walk) return static_cast<int>(walk[w]);
          }
          return w;
        };
        auto page_of = [&](int t) SYCL_ESIMD_FUNCTION { return FL && ring > 0 ? pages[t % ring] : pages[t]; };
        const int rg = i / G, h = i % G;
        // q of the item's 8 rows as A operands [dc][row][16 dims] (QK > 0: s8 codes [call][row][32 dims], scales [row]); a
        // dead row repeats the block's first row
        esimd::simd<bf16, QK ? 16 : kDc * 128> qa;
        esimd::simd<int8_t, QK ? kC8 * 256 : 32> qa8;
        esimd::simd<float, 8> sq;
        esimd::simd<int32_t, 8> rpos;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int j = rg * 8 + r, jj = j < nrows ? j : 0;
          rpos[r] = p0 + jj;
          esimd::simd<bf16, kD> row = esimd::block_load<bf16, kD>(qb + (static_cast<size_t>(r0 + jj) * hq + kvh * G + h) * kD);
          if constexpr (QK == 0) {
#pragma unroll
            for (int dc = 0; dc < kDc; ++dc) qa.template select<16, 1>(dc * 128 + r * 16) = row.template select<16, 1>(dc * 16);
          } else {
            esimd::simd<float, kD> f = row;
            const float amax = esimd::hmax<float>(esimd::abs(f));
            float sv = 0.f, inv = 0.f;
            if (amax > 0.f) inv = esimd::div_ieee(127.f, amax), sv = esimd::div_ieee(amax, 127.f);
            const esimd::simd<float, kD> x = f * inv;
            esimd::simd<float, kD> th = esimd::rnde<float, kD>(x);
            th = esimd::max(esimd::min(th, 127.f), -127.f);
            esimd::simd<int8_t, kD> qv = th;
#pragma unroll
            for (int cc = 0; cc < kC8; ++cc) qa8.template select<32, 1>(cc * 256 + r * 32) = qv.template select<32, 1>(cc * 32);
            if constexpr (QK == 2) {  // lo codes: (q / s - hi) x 127, |.| <= 64, into the item's SLM rows
              esimd::simd<float, kD> tl = esimd::rnde<float, kD>((x - th) * 127.f);
              tl = esimd::max(esimd::min(tl, 127.f), -127.f);
              esimd::simd<int8_t, kD> lv = tl;
#pragma unroll
              for (int cc = 0; cc < kC8; ++cc)
                esimd::slm_block_store<int8_t, 32>(static_cast<uint32_t>(kQlOff + (i * kC8 + cc) * 256 + r * 32), lv.template select<32, 1>(cc * 32));
            }
            sq[r] = sv;
          }
        }
        esimd::simd<float, kDc * 128> oa(0.f);  // [dc][row][16 dims]
        esimd::simd<float, 8> m_run(-INFINITY), l_run(0.f);
        const esimd::simd<int32_t, kTile> ti(0, 1);
        const esimd::simd<int32_t, kOp> rowi = esimd::simd<int32_t, kOp>(0, 1) >> 4;
#pragma unroll 1
        for (int w = tlo; w < ntiles; ++w) {
          const uint32_t work=SKIP?trained.subtiles[size_t(blk)*trained.pitch+w]:0xffffffffu;
          const uint32_t active=(work>>(4+4*rg))&15u;
          const int tile = tile_at(w);
          const int t0 = tile * kTile, nt = ntok - t0 < kTile ? ntok - t0 : kTile;
          const size_t slot = (static_cast<size_t>(page_of(tile)) * hkv + kvh) * kPage;  // the tile's first slot
          esimd::barrier();  // the previous tile's readers are done
#pragma unroll 1
          for (int u = i; u < 2 * kUnits; u += NI) {
            const bool isv = u >= kUnits;
            const int uu = isv ? u - kUnits : u, sub = uu / (kD / kUw), gw = uu % (kD / kUw);
            if constexpr(SKIP) if(!(work & (1u<<sub))) continue;
            const int live = nt - sub * 16;  // rows of this 16-token block holding tokens of the block's range
            if constexpr (V4) {
              if (isv) {  // a V unit = this sub's 16 tokens x 64 dims = 32 bytes of each 128-byte row; nibbles widened exactly
                esimd::simd<uint8_t, 16 * kUw / 2> raw = esimd::load_2d<uint8_t, kUw / 2, 16>(vp + slot * (kD / 2), kD / 2 - 1, kPage - 1, kD / 2 - 1, gw * kUw / 2, sub * 16);
                auto r2 = raw.template bit_cast_view<uint8_t, 16, kUw / 2>();
#pragma unroll
                for (int d4 = 0; d4 < kUdc; ++d4) {
                  const esimd::simd<int16_t, 128> bb = esimd::simd<uint8_t, 128>(r2.template select<16, 1, 8, 1>(0, d4 * 8));
                  const esimd::simd<int16_t, 128> lo = (bb & 15) - ((bb & 8) << 1), hi = (bb >> 4) - ((bb & 128) >> 3);
                  esimd::simd<bf16, kOp> b;
                  b.template select<128, 2>(0) = esimd::simd<float, 128>(lo);
                  b.template select<128, 2>(1) = esimd::simd<float, 128>(hi);
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kVOffQ + (sub * kDc + gw * kUdc + d4) * kOpBytes), v_operand(b));
                }
                continue;
              }
            }
            if constexpr (I8) {
              const uint8_t *base = (isv ? vp : kp) + slot * kD;
              esimd::simd<uint8_t, 16 * kUw> raw = esimd::load_2d<uint8_t, kUw, 16>(base, kD - 1, kPage - 1, kD - 1, gw * kUw, sub * 16);
              if (QK && !isv) {  // s8 B operands: dword (d / 4) x 16 + token = 4 dims of one token, the 16 x 8-dword block transposed
                auto r32 = raw.template bit_cast_view<uint32_t, 16, 16>();
#pragma unroll
                for (int c2 = 0; c2 < 2; ++c2) {
                  esimd::simd<uint32_t, 128> kd = r32.template select<16, 1, 8, 1>(0, c2 * 8), bd;
#pragma unroll
                  for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
                  esimd::slm_block_store<uint32_t, 128>(static_cast<uint32_t>(kKOffQ + ((gw * 2 + c2) * kSubs + sub) * kOp8), bd);
                }
                continue;
              }
              auto r2 = raw.template bit_cast_view<int8_t, 16, kUw>();
#pragma unroll
              for (int d4 = 0; d4 < kUdc; ++d4) {
                const esimd::simd<int8_t, kOp> sb = r2.template select<16, 1, 16, 1>(0, d4 * 16);
                const esimd::simd<float, kOp> f = sb;
                esimd::simd<bf16, kOp> b = f;
                const int dc = gw * kUdc + d4;
                if (isv) {
                  b.merge(esimd::simd<bf16, kOp>(0.f), rowi >= live);
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kVOffQ + (sub * kDc + dc) * kOpBytes), v_operand(b));
                } else {
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes), k_operand(b));
                }
              }
            } else {
              const uint16_t *base = reinterpret_cast<const uint16_t *>(isv ? vp : kp) + slot * kD;
              esimd::simd<uint16_t, 16 * kUw> raw = esimd::load_2d<uint16_t, kUw, 16>(base, kD * 2 - 1, kPage - 1, kD * 2 - 1, gw * kUw, sub * 16);
              auto r2 = raw.template bit_cast_view<uint16_t, 16, kUw>();
#pragma unroll
              for (int d2 = 0; d2 < kUdc; ++d2) {
                esimd::simd<uint16_t, kOp> sb = r2.template select<16, 1, 16, 1>(0, d2 * 16);
                const int dc = gw * kUdc + d2;
                if (isv) {
                  sb.merge(esimd::simd<uint16_t, kOp>(0), rowi >= live);
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kVOff + (sub * kDc + dc) * kOpBytes), v_operand(sb.template bit_cast_view<bf16>()));
                } else {
                  esimd::slm_block_store<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes), k_operand(sb.template bit_cast_view<bf16>()));
                }
              }
            }
          }
          esimd::barrier();  // the tile is staged
          if (w + 1 < ntiles) {  // the next tile's units into the caches while this one computes
            const size_t nslot = (static_cast<size_t>(page_of(tile_at(w + 1))) * hkv + kvh) * kPage;
            constexpr auto pfp = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>, esimd::cache_hint_L2<esimd::cache_hint::cached>};
#pragma unroll 1
            for (int u = i; u < 2 * kUnits; u += NI) {
              const bool isv = u >= kUnits;
              const int uu = isv ? u - kUnits : u, sub = uu / (kD / kUw), gw = uu % (kD / kUw);
              if constexpr(SKIP) if(!(trained.subtiles[size_t(blk)*trained.pitch+w+1] & (1u<<sub))) continue;
              if constexpr (V4) {
                if (isv) esimd::prefetch_2d<uint8_t, kUw / 2, 16>(vp + nslot * (kD / 2), kD / 2 - 1, kPage - 1, kD / 2 - 1, gw * kUw / 2, sub * 16, pfp);
                else esimd::prefetch_2d<uint8_t, kUw, 16>(kp + nslot * kD, kD - 1, kPage - 1, kD - 1, gw * kUw, sub * 16, pfp);
              } else if constexpr (I8) {
                esimd::prefetch_2d<uint8_t, kUw, 16>((isv ? vp : kp) + nslot * kD, kD - 1, kPage - 1, kD - 1, gw * kUw, sub * 16, pfp);
              } else {
                esimd::prefetch_2d<uint16_t, kUw, 16>(reinterpret_cast<const uint16_t *>(isv ? vp : kp) + nslot * kD, kD * 2 - 1, kPage - 1, kD * 2 - 1, gw * kUw, sub * 16, pfp);
              }
            }
          }
          // Every item still participates in both page barriers and staging.
          // Only this bundle's private score/softmax/PV state can be skipped.
          if constexpr(SKIP) if(!active) continue;
          esimd::simd<float, kTile> ks(1.f), vs(1.f);
          if constexpr (I8) {
            ks = esimd::block_load<float, kTile>(ksp + slot);
            if constexpr (!V4) vs = esimd::block_load<float, kTile>(vsp + slot);  // V4: the scales are folded in P.V
          }
          // ---- q.k: S [sub][row][16 tokens] ----
          esimd::simd<float, kSubs * 128> S(-INFINITY);
#pragma unroll
          for (int sub = 0; sub < kSubs; ++sub) {
            if (sub * 16 < nt && (!SKIP || (active & (1u<<sub)))) {
              if constexpr (QK == 0) {
                esimd::simd<float, 128> cc(0.f);
#pragma unroll
                for (int dc = 0; dc < kDc; ++dc) {
                  const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(static_cast<uint32_t>(kKOff + (dc * kSubs + sub) * kOpBytes));
                  const esimd::simd<bf16, 128> aa = qa.template select<128, 1>(dc * 128);
                  cc = xmx::dpas<8, 8, float, float, bf16, bf16>(cc, b, aa);
                }
                S.template select<128, 1>(sub * 128) = cc;
              } else if constexpr (QK == 1) {
                esimd::simd<int32_t, 128> cc(0);
#pragma unroll
                for (int c8 = 0; c8 < kC8; ++c8) {
                  const esimd::simd<int8_t, kOp8> b = esimd::slm_block_load<int8_t, kOp8>(static_cast<uint32_t>(kKOffQ + (c8 * kSubs + sub) * kOp8));
                  cc = xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8, 128, kOp8, 256>(
                      cc, b, qa8.template select<256, 1>(c8 * 256));
                }
                S.template select<128, 1>(sub * 128) = cc;
              } else {
                esimd::simd<int32_t, 128> ch(0), cl(0);
#pragma unroll
                for (int c8 = 0; c8 < kC8; ++c8) {
                  const esimd::simd<int8_t, kOp8> b = esimd::slm_block_load<int8_t, kOp8>(static_cast<uint32_t>(kKOffQ + (c8 * kSubs + sub) * kOp8));
                  const esimd::simd<int8_t, 256> al = esimd::slm_block_load<int8_t, 256>(static_cast<uint32_t>(kQlOff + (i * kC8 + c8) * 256));
                  ch = xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8, 128, kOp8, 256>(
                      ch, b, qa8.template select<256, 1>(c8 * 256));
                  cl = xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8, 128, kOp8, 256>(cl, b, al);
                }
                S.template select<128, 1>(sub * 128) = esimd::simd<float, 128>(ch) + esimd::simd<float, 128>(cl) * (1.f / 127.f);
              }
            }
          }
          // ---- online softmax per row; P (x the V scale) back into S ----
          esimd::simd<float, 8> alv;
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            esimd::simd<float, kTile> sv;
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) sv.template select<16, 1>(sub * 16) = S.template select<16, 1>(sub * 128 + r * 16);
            if constexpr (QK) sv = sv * (static_cast<float>(sq[r]) * kmul);
            else sv = sv * kmul;
            if constexpr (I8) sv *= ks;
            const int rp = rpos[r];
            esimd::simd_mask<kTile> dead = (ti + t0 > rp) | (ti >= nt);
            if constexpr (TRAINED) {
              const int j = rg * 8 + r;
              const size_t rr = r0 + (j < nrows ? j : 0);
              const uint32_t bits = trained.masks[rr * trained.pitch + tile];
              const esimd::simd<uint32_t, kTile> block = esimd::simd<uint32_t, kTile>(0, 1) >> 2;
              dead |= ((esimd::simd<uint32_t, kTile>(bits) >> block) & 1u) == 0u;
            }
            if constexpr (FL) {
              if (win > 0) dead |= ti + t0 <= rp - win;  // before the row's window
            }
            sv.merge(esimd::simd<float, kTile>(-INFINITY), dead);
            const float mt = esimd::hmax<float>(sv);
            const float mo = m_run[r];
            float mn = mo > mt ? mo : mt;
            if constexpr (FL || TRAINED) mn = mn == -INFINITY ? 0.f : mn;  // no live token yet: the state stays (exp(-inf) = 0)
            else if (part) mn = mn == -INFINITY ? 0.f : mn;     // a tail unit's range may hold none of a row's tokens
            const esimd::simd<float, 1> al = esimd::exp(esimd::simd<float, 1>(mo - mn));  // 0 on the first tile
            esimd::simd<float, kTile> pe = esimd::exp(sv - mn);
            pe.merge(esimd::simd<float, kTile>(0.f), dead);
            l_run[r] = l_run[r] * static_cast<float>(al[0]) + esimd::reduce<float>(pe, std::plus<>());
            if constexpr (FL || TRAINED) m_run[r] = mo > mt ? mo : mt;
            else m_run[r] = part ? (mo > mt ? mo : mt) : mn;
            if constexpr (I8 && !V4) {
              pe *= vs;
              pe.merge(esimd::simd<float, kTile>(0.f), dead);  // an unwritten slot's scale may be anything
            }
#pragma unroll
            for (int sub = 0; sub < kSubs; ++sub) S.template select<16, 1>(sub * 128 + r * 16) = pe.template select<16, 1>(sub * 16);
            alv[r] = al[0];
          }
          if ((alv != 1.f).any()) {  // o rescaled only when a running max moved (x 1 changes no bits)
            esimd::simd<float, 128> ar;
#pragma unroll
            for (int r = 0; r < 8; ++r) ar.template select<16, 1>(r * 16) = static_cast<float>(alv[r]);
#pragma unroll
            for (int dc = 0; dc < kDc; ++dc) oa.template select<128, 1>(dc * 128) *= ar;
          }
          // ---- P.V on all 256 dims ----
#pragma unroll
          for (int sub = 0; sub < kSubs; ++sub) {
            if (sub * 16 >= nt || (SKIP && !(active & (1u<<sub)))) continue;
            const esimd::simd<float, 128> pw = S.template select<128, 1>(sub * 128);
            if constexpr (V4) {
              // P x each token's scale of a 32-dim group (fp32, then the hi / lo split: the decode kernel's fold),
              // groups g = chunks 2 g, 2 g + 1; scales of tokens past the block's range masked (unwritten slots: any bits)
              esimd::simd<uint32_t, 64> sw = esimd::block_load<uint32_t, 64>(reinterpret_cast<const uint32_t *>(vsp) + (slot + sub * 16) * 4);
              const esimd::simd<int32_t, 16> tk = esimd::simd<int32_t, 16>(0, 1) + sub * 16;
#pragma unroll
              for (int g = 0; g < kD / 32; ++g) {
                esimd::simd<uint32_t, 16> dw = sw.template select<16, 4>(g / 2);
                esimd::simd<uint32_t, 16> sb = (g & 1) ? esimd::simd<uint32_t, 16>(dw & 0xFFFF0000u) : esimd::simd<uint32_t, 16>(dw << 16);
                esimd::simd<float, 16> sc = sb.template bit_cast_view<float>();
                sc.merge(esimd::simd<float, 16>(0.f), tk >= nt);
                esimd::simd<float, 128> pg, pwv = pw;
#pragma unroll
                for (int r = 0; r < 8; ++r) pg.template select<16, 1>(r * 16) = pwv.template select<16, 1>(r * 16) * sc;
                const esimd::simd<bf16, 128> gh = pg;
                const esimd::simd<bf16, 128> gl = pg - esimd::simd<float, 128>(gh);
#pragma unroll
                for (int dc = 2 * g; dc < 2 * g + 2; ++dc) {
                  const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(static_cast<uint32_t>(kVOffQ + (sub * kDc + dc) * kOpBytes));
                  esimd::simd<float, 128> o8 = oa.template select<128, 1>(dc * 128);
                  o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, gh);
                  if constexpr (LO) o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, gl);
                  oa.template select<128, 1>(dc * 128) = o8;
                }
              }
              continue;
            }
            const esimd::simd<bf16, 128> ph = pw;
            const esimd::simd<bf16, 128> pl = pw - esimd::simd<float, 128>(ph);
#pragma unroll
            for (int dc = 0; dc < kDc; ++dc) {
              const esimd::simd<bf16, kOp> b = esimd::slm_block_load<bf16, kOp>(static_cast<uint32_t>(kVOffQ + (sub * kDc + dc) * kOpBytes));
              esimd::simd<float, 128> o8 = oa.template select<128, 1>(dc * 128);
              o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, ph);
              if constexpr (LO) o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, b, pl);
              oa.template select<128, 1>(dc * 128) = o8;
            }
          }
        }
        if (part) {  // the unit's partial: o unnormalised, m, l (merged by attn_fa_merge)
          float *pb = tbuf + (static_cast<size_t>(tu * split + sp) * NI + i) * kPart;
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) esimd::block_store<float, 128>(pb + dc * 128, oa.template select<128, 1>(dc * 128));
          esimd::block_store<float, 8>(pb + 8 * kD, m_run);
          esimd::block_store<float, 8>(pb + 8 * kD + 8, l_run);
          return;
        }
        // ---- finish: o / l, bf16, gate ----
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int j = rg * 8 + r;
          if (j >= nrows) continue;
          const int row = r0 + j, hh = kvh * G + h;
          float L = l_run[r], E = 1.f;
          if constexpr (FL) {
            if (sinks) {  // the sink in the denominator (merge_sp's expression): M = max(m, sink)
              const float sk = static_cast<float>(sinks[hh]), mr = m_run[r], M = mr > sk ? mr : sk;
              esimd::simd<float, 2> ex;
              ex[0] = mr - M;
              ex[1] = sk - M;
              const esimd::simd<float, 2> ee = esimd::exp(ex);
              E = ee[0];
              L = L * E + static_cast<float>(ee[1]);
            }
          }
          bf16 *op = attn + (static_cast<size_t>(row) * hq + hh) * kD;
#pragma unroll
          for (int dc = 0; dc < kDc; ++dc) {
            const esimd::simd<float, 16> ov = oa.template select<16, 1>(dc * 128 + r * 16);
            esimd::simd<float, 16> a;
            if constexpr (FL) a = esimd::simd<bf16, 16>(ov * E / L);
            else a = esimd::simd<bf16, 16>(ov / L);
            if constexpr (kGate) {
              const float *gp = y + static_cast<size_t>(row) * ldy + hh * 2 * kD + kD;
              const esimd::simd<float, 16> gt = esimd::simd<bf16, 16>(esimd::block_load<float, 16>(gp + dc * 16));
              const esimd::simd<float, 16> sg = esimd::simd<bf16, 16>(esimd::simd<float, 16>(1.f) / (esimd::simd<float, 16>(1.f) + esimd::exp(-gt)));
              esimd::block_store<bf16, 16>(op + dc * 16, esimd::simd<bf16, 16>(a * sg));
            } else {
              esimd::block_store<bf16, 16>(op + dc * 16, esimd::simd<bf16, 16>(a));
            }
          }
        }
      });
}

template <int G, int QB, int D>
class GqaFaMergeK;
// The tail units' partials -> attn: per (tail group, item, row) M = max m_s, L = sum l_s exp(m_s - M), o = sum o_s exp(m_s - M),
// then attn_fa's finish (o / L to bf16, x the bf16 sigmoid gate, bf16). One item per (tail group, item, row).
template <int G, int QB, int D>
sycl::event attn_fa_merge(sycl::queue &q, const GqaShape &s, int m, const int32_t *pos_word, const float *y, int ldy, bf16 *attn, FaTail tp) {
  constexpr int NI = (QB / 8) * G, kDc = D / 16, kPart = 8 * D + 16;
  constexpr bool kGate = D == 256;
  const int hq = s.hq(), hkv = s.hkv(), blocks = (m + QB - 1) / QB, W = blocks * hkv, whole = W - tp.tail, split = tp.split;
  const float *tbuf = tp.buf;
  const size_t items = static_cast<size_t>(tp.tail) * NI * 8;
  return q.parallel_for<GqaFaMergeK<G, QB, D>>(sycl::nd_range<1>((items + 15) / 16 * 16, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t id = it.get_global_id(0);
    if (id >= items) return;
    const int r = static_cast<int>(id % 8), i = static_cast<int>((id / 8) % NI), tu = static_cast<int>(id / (8 * NI));
    const int g = whole + tu, blk = g / hkv, kvh = g % hkv, r0 = blk * QB;
    const int nrows = m - r0 < QB ? m - r0 : QB;
    const int rg = i / G, h = i % G, j = rg * 8 + r;
    if (j >= nrows) return;
    const int row = r0 + j, hh = kvh * G + h;
    esimd::simd<float, kFaSplitMax> ms(-INFINITY), ls(0.f);
    float M = -INFINITY;
    for (int sp = 0; sp < split && sp < kFaSplitMax; ++sp) {
      const float *pb = tbuf + (static_cast<size_t>(tu * split + sp) * NI + i) * kPart;
      ms[sp] = pb[8 * D + r], ls[sp] = pb[8 * D + 8 + r];
      M = ms[sp] > M ? static_cast<float>(ms[sp]) : M;
    }
    esimd::simd<float, kFaSplitMax> e = esimd::exp(ms - M);
    e.merge(esimd::simd<float, kFaSplitMax>(0.f), ms == -INFINITY);
    float L = 0.f;
    for (int sp = 0; sp < split && sp < kFaSplitMax; ++sp) L += static_cast<float>(ls[sp]) * static_cast<float>(e[sp]);
    bf16 *op = attn + (static_cast<size_t>(row) * hq + hh) * D;
#pragma unroll 1
    for (int dc = 0; dc < kDc; ++dc) {
      esimd::simd<float, 16> ov(0.f);
      for (int sp = 0; sp < split && sp < kFaSplitMax; ++sp) {
        const float *pb = tbuf + (static_cast<size_t>(tu * split + sp) * NI + i) * kPart;
        ov += esimd::block_load<float, 16>(pb + dc * 128 + r * 16) * static_cast<float>(e[sp]);
      }
      const esimd::simd<float, 16> a = esimd::simd<bf16, 16>(ov / L);
      if constexpr (kGate) {
        const float *gp = y + static_cast<size_t>(row) * ldy + hh * 2 * D + D;
        const esimd::simd<float, 16> gt = esimd::simd<bf16, 16>(esimd::block_load<float, 16>(gp + dc * 16));
        const esimd::simd<float, 16> sg = esimd::simd<bf16, 16>(esimd::simd<float, 16>(1.f) / (esimd::simd<float, 16>(1.f) + esimd::exp(-gt)));
        esimd::block_store<bf16, 16>(op + dc * 16, esimd::simd<bf16, 16>(a * sg));
      } else {
        esimd::block_store<bf16, 16>(op + dc * 16, esimd::simd<bf16, 16>(a));
      }
    }
  });
}

// Per device: the cores and the partials' buffer (sized for the largest plan: <= 64 units x 32 items)
struct FaTailDev {
  sycl::device dev;
  int cores = 0;
  float *buf = nullptr, *dbuf = nullptr;  // dbuf: decode calls'
};
std::mutex g_tail_mu;
std::vector<FaTailDev> g_tail_devs;
int g_tail_on = [] { const char *v = std::getenv("XEN_ATTN_TAIL"); return v ? std::atoi(v) : 2; }();

// The plan: the last W mod C work-groups split so that they fill the last wave (C = cores x work-groups per core: 32
// threads per core at 256 GRF); none when the remainder is over half a wave (the split would not save a wave's time).
template <int NI>
FaTail fa_tail_plan(sycl::queue &q, int W) {
  FaTail t;
  if (!g_tail_on) return t;
  std::lock_guard<std::mutex> lk(g_tail_mu);
  FaTailDev *d = nullptr;
  for (auto &x : g_tail_devs)
    if (x.dev == q.get_device()) d = &x;
  if (!d) {
    FaTailDev x;
    x.dev = q.get_device();
    x.cores = rt::xe_cores(x.dev);  // the emulated card's under XEN_EMU_CARD
    x.buf = sycl::aligned_alloc_device<float>(4096, static_cast<size_t>(64) * 32 * (8 * 256 + 16), q);
    g_tail_devs.push_back(x);
    d = &g_tail_devs.back();
  }
  if (g_fa_dec.on && !d->dbuf) d->dbuf = sycl::aligned_alloc_device<float>(4096, static_cast<size_t>(64) * 32 * (8 * 256 + 16), q);
  float *const tbuf = g_fa_dec.on ? d->dbuf : d->buf;
  const int C = d->cores * std::max(1, 32 / NI), r = W % C;
  if (W < C && g_tail_on >= 2) {  // a short call (one partial wave): every group split over its tiles
    t.tail = W;
    t.split = std::min({kFaSplitMax, C / W, 64 / W});
    if (t.split < 2) return FaTail{};
    t.buf = tbuf;
    return t;
  }
  if (W < C) return t;  // mode 1: calls under one wave unsplit (their bits independent of the call's size)
  if (r == 0 || 2 * r > C) return t;
  t.tail = r;
  t.split = std::min(8, C / r);
  if (t.split < 2 || r * t.split > 64) return FaTail{};
  t.buf = tbuf;
  return t;
}

// arch flint (grp 8, D 128, FL): P hi + lo or hi only; on the int8 KV also int8 q.k as hi + lo q codes with P hi only
// (QK 2)
template <int QB>
sycl::event fa_flint(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy, const bf16 *qb,
                     bf16 *attn, bool p_lo, int qk, const bf16 *sinks) {
  if (s.kv == QsaKv::i8) {
    if (qk == 2) return attn_fa<QsaKv::i8, 8, QB, false, 2, 128, true>(q, s, m, c, pos_word, y, ldy, qb, attn, sinks);
    return p_lo ? attn_fa<QsaKv::i8, 8, QB, true, 0, 128, true>(q, s, m, c, pos_word, y, ldy, qb, attn, sinks)
                : attn_fa<QsaKv::i8, 8, QB, false, 0, 128, true>(q, s, m, c, pos_word, y, ldy, qb, attn, sinks);
  }
  return p_lo ? attn_fa<QsaKv::bf16, 8, QB, true, 0, 128, true>(q, s, m, c, pos_word, y, ldy, qb, attn, sinks)
              : attn_fa<QsaKv::bf16, 8, QB, false, 0, 128, true>(q, s, m, c, pos_word, y, ldy, qb, attn, sinks);
}

// The page-selected instances, in the numerics page selection is built for: exact (bf16 q.k, P hi + lo) on
// either KV, int8-fast (int8 q.k, P hi only) on the int8 and int8-v4 KV and P hi only on the bf16 KV: XEN_PREFILL_ATTN's
// exact and what its auto resolves to. On int8-v4 the walk over the selected pages is the same; only the V loads differ.
template <int QB>
sycl::event fa_selected(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy, const bf16 *qb,
                        bf16 *attn, bool p_lo, int qk, FaTail tp, const GqaPageSets &sets) {
  const bool exact = qk == 0 && p_lo, hi_only = qk == 0 && !p_lo, int8_fast = qk == 1 && !p_lo;
  if (s.kv == QsaKv::i8 && exact) return attn_fa<QsaKv::i8, 6, QB, true, 0, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  if (s.kv == QsaKv::i8 && int8_fast) return attn_fa<QsaKv::i8, 6, QB, false, 1, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  if (s.kv == QsaKv::i8v4 && exact) return attn_fa<QsaKv::i8v4, 6, QB, true, 0, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  if (s.kv == QsaKv::i8v4 && int8_fast) return attn_fa<QsaKv::i8v4, 6, QB, false, 1, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  if (s.kv == QsaKv::bf16 && exact) return attn_fa<QsaKv::bf16, 6, QB, true, 0, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  if (s.kv == QsaKv::bf16 && hi_only) return attn_fa<QsaKv::bf16, 6, QB, false, 0, 256, false, true>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, &sets);
  throw std::invalid_argument("gqa_attn_fa: page sets run the exact numerics (qk 0, P hi + lo) on the int8 or bf16 KV, int8 q.k with P hi only (qk 1, "
                              "p_lo false) on the int8 / int8-v4 KV, or P hi only on the bf16 KV");
}

template <int G, int QB, int D = 256>
sycl::event fa_kv(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy, const bf16 *qb,
                  bf16 *attn, bool p_lo, int qk, const GqaPageSets *sets = nullptr) {
  constexpr int NI = (QB / 8) * G;
  const FaTail tp = fa_tail_plan<NI>(q, (m + QB - 1) / QB * s.hkv());
  auto body = [&]() -> sycl::event {
    if constexpr (D == 256 && G == 6) {
      if (sets) return fa_selected<QB>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, tp, *sets);
    }
    if constexpr (D == 256) {
      if (s.kv == QsaKv::i8v4) {
        if (qk == 1)
          return p_lo ? attn_fa<QsaKv::i8v4, G, QB, true, 1, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp)
                      : attn_fa<QsaKv::i8v4, G, QB, false, 1, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
        if (qk == 2) return attn_fa<QsaKv::i8v4, G, QB, false, 2, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
        return p_lo ? attn_fa<QsaKv::i8v4, G, QB, true, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp)
                    : attn_fa<QsaKv::i8v4, G, QB, false, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
      }
    }
    if (s.kv == QsaKv::i8) {
      if (qk == 1)
        return p_lo ? attn_fa<QsaKv::i8, G, QB, true, 1, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp)
                    : attn_fa<QsaKv::i8, G, QB, false, 1, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
      if (qk == 2) return attn_fa<QsaKv::i8, G, QB, false, 2, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
      return p_lo ? attn_fa<QsaKv::i8, G, QB, true, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp)
                  : attn_fa<QsaKv::i8, G, QB, false, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
    }
    return p_lo ? attn_fa<QsaKv::bf16, G, QB, true, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp)
                : attn_fa<QsaKv::bf16, G, QB, false, 0, D>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp);
  };
  sycl::event e = body();
  if (tp.tail) e = attn_fa_merge<G, QB, D>(q, s, m, pos_word, y, ldy, attn, tp);
  return e;
}

}  // namespace

sycl::event gqa_attn_fa_trained(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word,
                               const float *y, int ldy, const gqa_bf16 *qb, gqa_bf16 *attn, const GqaTrainedPrefill &plan) {
  if (s.kv != QsaKv::i8v4 || s.head_dim != 256 || s.grp() != 6 || !s.gate || s.window || s.rope_rows || c.slot_pages || c.ring_pages ||
      !plan.pages || !plan.counts || !plan.masks || plan.pitch != c.max_tokens / kPage)
    throw std::invalid_argument("gqa_attn_fa_trained: expected one text K8V4 slot and a valid per-row plan");
  auto run = [&]<int QB,bool Skip=false>() {
    const FaTail tp = fa_tail_plan<(QB / 8) * 6>(q, (m + QB - 1) / QB * s.hkv());
    auto e = attn_fa<QsaKv::i8v4, 6, QB, false, 1, 256, false, false, true,Skip>(q, s, m, c, pos_word, y, ldy, qb, attn, nullptr, tp, nullptr, plan);
    if (tp.tail) e = attn_fa_merge<6, QB, 256>(q, s, m, pos_word, y, ldy, attn, tp);
    return e;
  };
  if(plan.subtiles) {
    if(plan.rows_per_group!=40) throw std::invalid_argument("subtile sparse FA needs 40-row dispatch");
    return run.template operator()<40,true>();
  }
  switch (plan.rows_per_group) {
    case 16: return run.template operator()<16>();
    case 32: return run.template operator()<32>();
    case 40: return run.template operator()<40>();
  }
  throw std::invalid_argument("gqa_attn_fa_trained: expected 16, 32 or 40 rows per group");
}

sycl::event gqa_attn_fa_decode(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *rec, int stride, const float *y, int ldy,
                               const gqa_bf16 *qb, gqa_bf16 *attn, bool p_lo, int qk) {
  if (s.head_dim != 256 || s.grp() != 6 || !s.gate || s.window > 0 || c.ring_pages > 0 || m < 1 || m > 16)
    throw std::invalid_argument("gqa_attn_fa_decode: grp 6 at head_dim 256 with the gate, 1..16 rows of one segment");
  if (qk && !qsa_kv_scaled(s.kv)) throw std::invalid_argument("gqa_attn_fa_decode: int8 q.k needs the int8 KV");
  struct Guard {
    Guard(const int32_t *w, int sp) { g_fa_dec = FaDec{w, sp, true}; }
    ~Guard() { g_fa_dec = FaDec{}; }
  } guard(stride >= kSegRec && c.slot_pages ? rec + kSegSlot : nullptr, c.slot_pages);
  GqaCache c0 = c;
  c0.slot_pages = 0;
  return fa_kv<6, 16>(q, s, m, c0, rec + kSegPos, y, ldy, qb, attn, p_lo, qk);
}

void gqa_attn_fa_set_tail(int mode) { g_tail_on = std::clamp(mode, 0, 2); }
int gqa_attn_fa_tail() { return g_tail_on; }

int gqa_attn_fa_rows(const GqaShape &s, int rows_per_group) {
  if (rows_per_group > 0) return rows_per_group;
  if (s.head_dim == 128) return 32;  // llama (grp 7) and flint (grp 8)
  return s.grp() == 6 ? 40 : 16;
}

sycl::event gqa_attn_fa(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                        const gqa_bf16 *qb, gqa_bf16 *attn, int rows_per_group, bool p_lo, int qk, const gqa_bf16 *sinks, const GqaPageSets *page_sets) {
  if (c.slot_pages) throw std::invalid_argument("gqa_attn_fa: one slot (page table without slot stride)");
  if (page_sets) {
    if (s.head_dim != 256 || s.grp() != 6 || s.window > 0 || sinks || c.ring_pages > 0)
      throw std::invalid_argument("gqa_attn_fa: page sets are built for 6 q heads per kv head at head_dim 256 (no window / ring / sinks)");
    const int blocks = (m + gqa_attn_fa_rows(s, rows_per_group) - 1) / gqa_attn_fa_rows(s, rows_per_group);
    if (!page_sets->valid() || blocks > page_sets->blocks) throw std::invalid_argument("gqa_attn_fa: the page sets hold fewer row blocks than the call has");
  }
  if (s.kv == QsaKv::i8v4 && (s.head_dim != 256 || s.grp() != 6)) throw std::invalid_argument("gqa_attn_fa: int8-v4 KV is built for 6 q heads per kv head at head_dim 256");
  if (s.head_dim == 128 && s.grp() == 8) {  // arch flint: window, ring, sinks; 2 / 3 / 4 x 8 = 16 / 24 / 32 items
    if (s.gate) throw std::invalid_argument("gqa_attn_fa: grp 8 (flint) runs without gate");
    if (qk != 0 && (qk != 2 || p_lo || s.kv != QsaKv::i8)) throw std::invalid_argument("gqa_attn_fa: grp 8 (flint): qk 0, or 2 on the int8 KV with P hi only");
    if (c.ring_pages > 0 && (s.window <= 0 || c.ring_pages < (s.window + m - 2) / kTile + 2))
      throw std::invalid_argument("gqa_attn_fa: the KV ring must hold the window and the call's rows");
    if (rows_per_group == 0 || rows_per_group == 32) return fa_flint<32>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, sinks);
    if (rows_per_group == 16) return fa_flint<16>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, sinks);
    if (rows_per_group == 24) return fa_flint<24>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, sinks);
    throw std::invalid_argument("gqa_attn_fa: grp 8 runs 16 / 24 / 32 rows per group");
  }
  if (s.window > 0 || sinks || c.ring_pages > 0) throw std::invalid_argument("gqa_attn_fa: windows / sinks / rings are built for grp 8 at head_dim 128");
  if (qk < 0 || qk > 2) throw std::invalid_argument("gqa_attn_fa: qk 0 (bf16), 1 (int8 q), 2 (int8 hi + lo q)");
  if (qk && !qsa_kv_scaled(s.kv)) throw std::invalid_argument("gqa_attn_fa: int8 q.k needs the int8 KV");
  if (qk == 2 && p_lo) throw std::invalid_argument("gqa_attn_fa: qk 2 runs with P hi only (p_lo false)");
  if (s.head_dim == 128) {  // arch llama: grp 7 (Nanbeige4.3), no gate; 4 x 7 = 28 items at 32 rows, 2 x 7 at 16
    if (s.grp() != 7 || s.gate || s.qk_norm) throw std::invalid_argument("gqa_attn_fa: head_dim 128 is built for 7 q heads per kv head without gate");
    if (rows_per_group == 0 || rows_per_group == 32) return fa_kv<7, 32, 128>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk);
    if (rows_per_group == 16) return fa_kv<7, 16, 128>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk);
    if (rows_per_group == 24) return fa_kv<7, 24, 128>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk);
    throw std::invalid_argument("gqa_attn_fa: head_dim 128 runs 16 / 24 / 32 rows per group");
  }
  if (s.grp() == 6) {
    if (rows_per_group == 0 || rows_per_group == 40) return fa_kv<6, 40>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, page_sets);
    if (rows_per_group == 32) return fa_kv<6, 32>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, page_sets);
    if (rows_per_group == 16) return fa_kv<6, 16>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk, page_sets);
  }
  if (s.grp() == 12) {
    if (rows_per_group == 0 || rows_per_group == 16) return fa_kv<12, 16>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk);
    if (rows_per_group == 8) return fa_kv<12, 8>(q, s, m, c, pos_word, y, ldy, qb, attn, p_lo, qk);
  }
  throw std::invalid_argument("gqa_attn_fa: built for 6 (16 / 32 / 40 rows per group) or 12 (8 / 16) q heads per kv head");
}

}  // namespace xenithium::kernels
