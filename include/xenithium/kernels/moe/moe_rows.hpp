// Xenithium · MoE block over many rows (prefill): the same semantics and rounding points
// as moe_decode (moe.hpp), for m rows of a chunk, expert TP layout (MoeShape:tp) on 1 or 2 cards.
// 1 router pf_dense over [router | shared gate] (bf16) -> logits fp32 [m][router.n]
// 2 topk one 16-lane sub-group per row: top-k ids (ties -> lowest id), softmax weights over the
// k bf16 logits, sigmoid of the shared gate (moe_route's arithmetic)
// 3 group one work-group: rows counted per expert, an expert-ordered list of the (row, k)
// assignments, and segments of <= 16 list rows of one expert. List order inside an
// expert is by atomics (not deterministic), which changes no result: every DPAS row is
// independent of the rows beside it.
// 4 gate_up per (segment, 16-column tile): gate and up of this card's half columns for the
// segment's rows (W4A16, rows gathered from x), h = bf16(silu(bf16 g) bf16 u)
// 5 down per (segment, 16-column output tile): y = h. Wd^T over this card's 64-k half blocks
// W4A8 (a8) x quantised to int8 per (row, 128-group) before 4, h per (list row,
// 64-group) between 4 and 5; 4 and 5 run s8 x s4 DPAS on the same weight bytes
// shared every row on each card (split the rows over the cards), the whole
// shared MLP on pf_dense (decode splits it by columns): s = sigmoid(sg). y_shared.
// Without the split send its gate|up runs in the router's launch
// (pf_dense_pair, the same bits) whenever both bodies pair (short calls); then
// the W4A8 x quantiser rides that launch too, and the swiglu runs in the topk launch
// 6 combine per (row, 64 columns): sum over the row's experts in ascending id of w y, fp32; the two
// cards exchange their sums through a chunk-sized mailbox (decode's protocol) and write
// bf16((p_0 + p_1) + s). Split mode (MoeRowsWire:split): combine writes the card's
// fp32 sum to `part` instead, and the caller finishes with pair_split_rows (add = s), which
// gives the same bits. Every row's arithmetic is independent of m and of the row's
// place in the call, so a prompt prefilled in calls of any size gives the same rows.
// The partial sums group differently from decode (the shared term), so prefill and decode outputs
// differ by fp32 summation order.
#pragma once

#include <span>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/moe/moe.hpp"

namespace xenithium::kernels {

struct PairSplit;  // wire.hpp
struct PairSplitSend;

inline constexpr int kMoeSegRows = 16;  // list rows per segment (DPAS M = 2 x 8)
// List rows per block of the work-group-per-block W4A16 gate_up / down (MoeRowsScratch blk): four 8-row DPAS blocks per
// thread.
inline constexpr int kMoeBlkRows = 32;
// W4A8: x columns per int8 scale (MoeRowsScratch:xs holds hidden / kMoeXGroup scales per row).
inline constexpr int kMoeXGroup = 128;

struct MoeRowsScratch {
  float *logits = nullptr;  // [rows_max][router.n]
  int32_t *ids = nullptr;  // [rows_max][topk]
  float *topw = nullptr;  // [rows_max][topk]
  float *sg = nullptr;  // [rows_max] sigmoid of the shared gate logit
  int32_t *slot = nullptr;  // [rows_max][topk]: the assignment's list position
  int32_t *row = nullptr;  // [rows_max * topk]: list position -> row
  int32_t *seg = nullptr;  // [moe_rows_max_segs][4]: expert, first list position, rows, -
  int32_t *seg_n = nullptr;  // [16]: [0] = segments
  // With blk set (and blk_n), the group launch also lists blocks of <= kMoeBlkRows list rows of one expert, and the
  // W4A16 gate_up and down run a work-group per block (the block's x / h rows staged once in SLM per k group, every
  // weight tile loaded once per 32 rows instead of per 16): the same bits (each (row, column) keeps the per-segment
  // kernels' DPAS chain, scale and rounding; test-moe compares them; MoeShape:w4_bias trades that for speed in gate_up).
  // Null = the segment kernels.
  int32_t *blk = nullptr;  // [moe_rows_max_blks][4]: expert, first list position, rows, -
  int32_t *blk_n = nullptr;  // [16]: [0] = blocks
  moe_bf16 *h = nullptr;  // [rows_max * topk][inter_local]
  moe_bf16 *y = nullptr;  // [rows_max * topk][hidden]: each expert's down output rounded to bf16.
                             // Under tp it is this card's partial of y_e (MoeShape:down_bf16 in moe_decode rounds
                             // the same value).
  float *sgu = nullptr;  // [rows_max][2 shared_inter]
  moe_bf16 *sh = nullptr;  // [rows_max][shared_inter]
  float *sy = nullptr;  // [rows_max][hidden]
  // W4A8: x and h as int8, one fp32 scale per (row, 128-k group) / (list row, 64-k group)
  int8_t *xq = nullptr;  // [rows_max][hidden]
  float *xs = nullptr;  // [rows_max][hidden / 128]
  int8_t *hq = nullptr;  // [rows_max * topk][inter_local]
  float *hs = nullptr;  // [rows_max * topk][inter_local / 64] (the W4A8 blocks: [..][moe_rows_blk8_hsegs])
  moe_bf16 *xr = nullptr;  // [rows_max][hidden]: the rotated x (MoeShape:rot only;)
  PfA8 da8;  // set = the shared expert's gate|up and down GEMMs run W8A8 (its own
                             // quantisation of the plain x: xq is the routed experts' input, rotated under rot)
  float *part = nullptr;  // [rows_max][hidden]: the card's expert sum, split mode only
  // A cold expert is read over PCIe once per 16-row segment. With stage_cap > 0
  // and MoeWeights:cold, every cold expert with >= stage_min_segs segments (most segments first, at
  // most stage_cap) is copied into `stage` once per call and gate_up / down read it there (the same
  // bytes, the same bits). tab2 [experts] is the call's blob table; stage_list [1 + stage_cap]. Calls of
  // fewer than stage_min_rows rows skip it and its two launches per layer.
  const uint8_t **tab2 = nullptr;
  uint8_t *stage = nullptr;
  int32_t *stage_list = nullptr;
  int stage_cap = 0, stage_min_segs = 2, stage_min_rows = 64;
  size_t stage_stride = 0;
};
int moe_rows_max_segs(const MoeShape &s, int rows_max);
int moe_rows_max_blks(const MoeShape &s, int rows_max);  // MoeRowsScratch:blk entries
// Element counts of the scratch buffers for rows_max rows.
struct MoeRowsSizes {
  size_t logits, ids, sg, row, seg, h, y, sgu, sh, sy, xq, xs, hq, hs, xr, blk;
};
MoeRowsSizes moe_rows_sizes(const MoeShape &s, int rows_max, int router_n);

// The chunk-sized two-card exchange: inbox / peer_inbox 2 (parity) x rows_max x hidden words,
// initialised to kMoeMailEmpty (moe_rows_wire_init).
// Rows [r0, r0 + m) of a moe_rows call as a split of their own (its owner rows, recv and flag lines).
struct MoeSplitPart {
  int r0 = 0, m = 0;
  const PairSplitSend *send = nullptr;
};
struct MoeRowsWire {
  uint32_t *inbox = nullptr;
  uint32_t *peer_inbox = nullptr;
  uint32_t *error = nullptr;
  int rows_max = 0;
  uint64_t wait_cycles = 11200000000ull;  // >= 4 s at 2.8 GHz (a chunk's launches are long)
  bool split = false;  // no exchange here; combine leaves the card's fp32 sum in MoeRowsScratch:part
  // Split only: the peer's rows of part are combined first and sent (pair_split_send) while
  // the shared expert and the own rows run; the caller then calls pair_split_finish.
  const PairSplitSend *send = nullptr;
  // With send: the caller's sum stays with each row's owner (pair_split_finish SplitRows:owner), which adds the shared
  // term for its own rows only, so the shared expert runs on the card's own rows only. A row's bits are the whole
  // call's (the shared expert's GEMMs keep a row's bits whatever the rows beside it).
  bool shared_own_rows = false;
  // Instead of send: the call's rows as several splits, in row order and covering every row. One call then serves
  // several callers' row ranges (the routed experts are cheaper per row the more rows a call has), each range's partial
  // sent as its own (its rows at part + r0 * hidden).
  std::span<const MoeSplitPart> parts{};
  // With parts (W4A8 blocks only): every row's routing (ids, topw, sg) and int8 x (xq, xs) are already in the scratch -
  // moe_rows_route on the row's owner, which shares them with the other cards - so the router, topk and the x quantiser
  // are skipped. x is then read on the card's own rows only (the shared expert).
  bool routed = false;
  // TP4: four cards (comm/wire.hpp WireQuad; not with split): ((e0 + e1) + (e2 + e3)) on every card, the inbox a slot
  // region per hop (moe_rows_wire_words with MoeShape:cards 4). inbox / peer_inbox unused.
  WireQuad quad{};
};
size_t moe_rows_wire_words(const MoeShape &s, int rows_max);
sycl::event moe_rows_wire_init(sycl::queue &q, const MoeShape &s, int rows_max, uint32_t *inbox);

// x bf16 [m][hidden]; out bf16 [m][hidden]. parity alternates per call on both cards. w.blobs: the
// card's TP blob table of this layer ([experts]). a8: gate_up / down on W4A8 (x and h
// quantised to int8 per row and group, s8 x s4 DPAS on the same weight bytes); false = W4A16 (bf16
// activations, the numerics). fuse: when the shared gate|up rides the router's
// launch, the W4A8 x quantiser rides it too and the shared swiglu runs in the topk launch; false =
// the launches. The same bits either way (tests compare them).
MoeEvents moe_rows(sycl::queue &q, const MoeShape &s, int m, const moe_bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc,
                   const MoeRowsWire *wire, int parity, moe_bf16 *out, bool a8 = false, bool fuse = true);

// The routing steps of moe_rows on rows [lo, hi) of x only - router, topk and (a8) the x quantiser - into the same rows
// of sc's logits, ids, topw, sg, xq and xs. Four cards run it on each card's own rows of a split call and copy those rows
// of ids / topw / xq / xs to the other cards (int8 x: about half the bytes of the bf16 rows), then moe_rows with
// MoeRowsWire:routed. Every step is per row, so a row gets the bits the whole call's steps give it. Returns the last
// launch's event in `route` (the router's in `router`, topk's in `topk`).
MoeEvents moe_rows_route(sycl::queue &q, const MoeShape &s, int lo, int hi, const moe_bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc, bool a8);

// Internal to moe_rows: the W4A16 gate_up launch -> sc.h, in moe_rows_gu.cpp. It spills at 128 GRF, so it is built in
// the 256-GRF library (CMake xenithium_kernels_xl). The other moe_rows kernels do not spill and stay at 128.
sycl::event moe_rows_gate_up(sycl::queue &q, const MoeShape &s, int max_segs, const moe_bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc);
// The W4A16 down on two tiles per item (moe_rows_gu.cpp, 256 GRF); hidden % 32.
sycl::event moe_rows_down2(sycl::queue &q, const MoeShape &s, int max_segs, const MoeWeights &w, const MoeRowsScratch &sc);
// The work-group-per-block W4A16 gate_up and down over MoeRowsScratch:blk (moe_rows_gu.cpp, 256 GRF).
// moe_rows_blk_shape: this card's columns and k take them (16-column tile pairs per card 8, 10, 16 or 20; hidden % 32;
// down pieces of at most 10 per card).
bool moe_rows_blk_shape(const MoeShape &s);
sycl::event moe_rows_gate_up_blk(sycl::queue &q, const MoeShape &s, int max_blks, const moe_bf16 *x, const MoeWeights &w, const MoeRowsScratch &sc);
sycl::event moe_rows_down_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc);
// The work-group-per-block W4A8 gate_up and down at 160 (TP4) or 320 (TP2) columns per card (moe_rows_gu.cpp, 256 GRF): s8 x s4 DPAS
// on xq / xs (one scale per (row, 128-k group)); gate_up writes h straight as hq with one scale per (list row, segment)
// into hs ([rows][moe_rows_blk8_hsegs]), a segment = the card's k range cut at the down scales' 128-k group bounds;
// down reads them. Not rotated experts.
bool moe_rows_blk8_shape(const MoeShape &s);
int moe_rows_blk8_hsegs(const MoeShape &s);
sycl::event moe_rows_gate_up8_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc);
sycl::event moe_rows_down8_blk(sycl::queue &q, const MoeShape &s, int max_blks, const MoeWeights &w, const MoeRowsScratch &sc);

}  // namespace xenithium:kernels
