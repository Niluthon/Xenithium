// Xenithium · dense gated GQA decode: the full-attention layers of the dense
// Qwen3.8-27B (arch qwen35), rows = 1..8 speculative tokens of one or several sequences, one card's heads.
//
// Semantics (HF qwen3_5 `full_attention` layers, from config.json and the checkpoint), per token at
// position p (rows r = 0..R-1 of a segment sit at p = pos + r):
//   [q_h | gate_h] (per q head), k, v = bf16(x . W^T)       head_dim 256, W int4 (W4, group 128)
//   q, k      = bf16(rope(bf16(rmsnorm_(1+w)(.))))          x rsqrt(mean x^2 + eps) (1 + w), fp32
//   rope      NeoX halves over the first 64 dims (partial_rotary 0.25), the QSA host table
//             (qsa_rope_table, theta 1e7); text positions: the interleaved M-RoPE sections collapse to 1-D (image
//             rows: GqaShape::mrope / rope_rows)
//   attn      softmax(q . K[0..p] / 16) . V[0..p] per q head over the whole context (fp32; q head h reads
//             kv head h / grp), no selection
//   out       bf16(bf16(attn) bf16(sigmoid(gate))) . Wo^T: per card a k slice -> fp32 partials, the
//             two-card sum (wire.hpp).
// The same rounding points as QSA's attention (qsa.hpp) without the indexer: the host references are
// qsa_ref_norm_rope / qsa_ref_i8 / qsa_ref_attend / qsa_ref_gate.
//
// KV cache: QSA's page layout (pages of 64 tokens named by a per-slot device page table), K / V
// [pages][hkv][64][256] in bf16 or int8 (+ one fp32 scale per (token, kv head), qsa.hpp's codes); no
// compressed keys, no raw-key ring.
//
// Attention is split along the context (flash decoding): a row's ntiles = ceil((p + 1) / 64) tiles
// are cut into nsp = ceil(ntiles / tps) splits of tps = ceil(ntiles / kGqaSplits) tiles; one
// work-group per (row, kv head, split) walks its tiles with a running max / sum (online softmax),
// and the merge combines the splits by LSE and applies the gate. The cut depends only on the row's
// own position, so a row's bits do not depend on the step's other rows or on pos_bound.
//
// Speculative rows need no accept word: everything is addressed by position, so a rejected row's
// K / V is rewritten when the real token at that position comes; prep writes every row's K / V
// before any row attends.
//
// Launches: qkv GEMM (W4), prep (norms, rope, cache writes), attn (splits), merge (LSE, gate),
// o GEMM (W4, fp32 partial), pair_reduce.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/attention/qsa.hpp"  // QsaKv, the page / rope constants, the host references
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/comm/wire.hpp"

namespace xenithium::kernels {

using gqa_bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kGqaSplits = 64;  // most context splits per (row, kv head)
inline constexpr int kGqaWinSplits = 16;  // a windowed layer's splits (GqaShape::window)
// Page-selected decode attention (opt-in). A page's keys are bounded per kv head by their
// elementwise min / max (GqaCache::kmin / kmax, kept by every K write when allocated); a step with GqaStep::select = B pages
// scores every page of a segment's context per kv head by the largest over its rows' q heads of the page score below,
// keeps the first kGqaSelSink pages, the last kGqaSelLocal (up to the segment's last row) and the best B - sink - local
// of the rest, ascending, and the attention walks only those (gqa_seg.hpp, Sel). A segment whose context is B pages or
// fewer keeps them all.
// The page score of a q head: q . mid + spread x |q| . rad, with mid = (kmax + kmin) / 2 and
// rad = (kmax - kmin) / 2 per coordinate. Spread 1 is the upper bound of q . k over the page's keys:
// it finds a page with one outstanding key but ranks the many pages of evenly spread attention by their
// extremes. Spread 0 is the logit of the page's middle key: best for the spread-out mass, blind to a single key. The
// spread is gqa_sel.cpp's kSelSpread, a trade between the two; the scores of a step are one bf16 GEMM of its
// q heads against the pages' bounds (gqa_sel.cpp, score_pages).
inline constexpr int kGqaSelMax = 4096;    // most pages a selection keeps (B)
inline constexpr int kGqaSelSink = 1, kGqaSelLocal = 16;
inline constexpr int kGqaSelPagesMax = kQsaMaxTokens / kQsaPage;  // pages a segment's context may span (score keys per (segment, kv head))

// Arch llama (Nanbeige4.3): the same kernels at head_dim 128 with 7 q heads per kv head, no output gate (q_proj = query
// only), no q / k norms (q, k = bf16(rope(bf16(x . W^T)))), the rope over all 128 dims (NeoX halves, the rope table's
// row = 64 cos | 64 sin), softmax scale 1 / sqrt(128). The KV pages hold [64][128] per kv head.
struct GqaShape {
  int hidden = 0, q_heads = 0, kv_heads = 0;  // the whole model
  int cards = 1, card = 0;
  float eps = 1e-6f;
  QsaKv kv = QsaKv::i8;
  int head_dim = kQsaDim;  // 256 (qwen35) or 128 (llama)
  bool gate = true;        // q_proj rows = query | gate per head, sigmoid(gate) applied after the attention
  bool qk_norm = true;     // (1 + w) RMS norms of q and k before the rope
  int rot = kQsaRot;       // rotary dims (NeoX halves over the first rot dims); the rope table row = rot bf16 (cos | sin)
  // Arch flint (Recursive Flint: 32 q / 4 kv heads x 128, grp 8): plain q / k norms (w x norm(x), not (1 + w)); rot 128
  // on its sliding layers, 0 (no positional encoding) on its full ones; window = the sliding layers' tokens (513: the
  // row's own token and the 512 before it; 0 = the whole context); learned sinks (GqaWeights::sinks).
  // A windowed layer's splits are its tiles: tile t's partial goes to split t % kGqaWinSplits (a row's window spans
  // <= kGqaWinSplits - 1 tiles, a group of 8 rows one more), so a row's bits stay its own; its KV may be a ring
  // (GqaCache::ring_pages).
  bool norm_plus1 = true;
  int window = 0;
  // Arch flint: the merge with the splits across its 16 items (each item a quarter of the splits over all
  // 128 dims, the partial o / sums reduced through SLM in item order) instead of every item walking all splits for its
  // 8 dims. head_dim 128 without the gate only. Other bits than the walk.
  bool split_merge = false;
  // The interleaved M-RoPE (qwen35 image input). A row's rope row is, in order: rope_rows + r x rot when
  // rope_rows is set (a many-row call's own rows, bf16 cos | sin: image rows mix three positions per frequency); else,
  // with mrope, the 1-D table's row at position + delta, delta = the segment record's word kSegRope (records) or
  // rope_delta (a bare position word); else the row at the position. All off = the rope by position.
  bool mrope = false;
  int rope_delta = 0;
  const gqa_bf16 *rope_rows = nullptr;
  // The decode attention's numerics (the group kernel, gqa_seg.hpp; grp 6 at head_dim 256 only,
  // dense walk: no page selection). dec_qk 0 = q.k on bf16 (q x the widened K); 1 = int8 q per (row, q head) x the int8
  // K codes on s8 DPAS; 2 = q as hi + lo int8 codes (q.k within fp32 noise of 0); dec_qk > 0 needs the int8 KV.
  // dec_p_lo false = P.V on P as one bf16 (no lo half). The defaults are the exact decode numerics.
  int dec_qk = 0;
  bool dec_p_lo = true;
  bool dec_exact() const { return dec_qk == 0 && dec_p_lo; }
  bool dec_int8_fast() const { return dec_qk == 1 && !dec_p_lo; }  // int8 q.k on the K codes, P hi only (the served auto)
  int hq() const { return q_heads / cards; }
  int hkv() const { return kv_heads / cards; }
  int grp() const { return q_heads / kv_heads; }
  int q_n() const { return hq() * (gate ? 2 : 1) * head_dim; }  // q (| gate) per head
  int qkv_n() const { return q_n() + 2 * hkv() * head_dim; }    // ... | k | v
  int o_k() const { return hq() * head_dim; }
  int kv_elem() const { return kv == QsaKv::bf16 ? 2 : 1; }  // K's element bytes (int8-v4: K int8)
  // int8-v4 = K int8 as int8 KV, V as int4 codes (two per byte, dim 2j
  // in the low nibble) with one bf16 scale per 32 dims (qsa_ref_v4's rule: s = bf16(amax / 7), code = rne(v / s) clamped
  // to +-7). head_dim 256 only (the dense 27B). A V row is 128 B + 16 B of scales (int8: 256 + 4).
  int v_row_bytes() const { return kv == QsaKv::i8v4 ? head_dim / 2 : head_dim * kv_elem(); }
  int v_scale_bytes() const { return kv == QsaKv::i8v4 ? head_dim / 32 * 2 : 4; }
};

struct GqaWeights {
  // Arch flint: this card's q heads' sink logits (bf16 [hq], device): softmax over the scores and the sink, which
  // attends to nothing (o = sum p v / (sum p + exp(sink - m))). Null: none.
  const gqa_bf16 *sinks = nullptr;
  W4Mat qkv;  // [q|gate of each head | k | v] of this card's heads x hidden
  W4Mat o;    // [hidden][hq x 256]: this card's k slice of o_proj
  const gqa_bf16 *q_norm = nullptr, *k_norm = nullptr;  // [256]
  // Dense projections of the same layout (the qwen35 MTP drafter's bf16 layer, stored int8 per row or bf16) in
  // place of qkv / o when valid (dense_gemm in gqa_decode, pf_dense in gqa_kv_rows); the W4 fields then stay empty.
  DenseWeights qkv_d, o_d;
  bool dense() const { return qkv_d.valid(); }
  int qkv_n() const { return dense() ? qkv_d.n : qkv.n; }  // the projection's columns = its row stride in the scratch
  int qkv_k() const { return dense() ? qkv_d.k : qkv.k; }
};

struct GqaCache {
  uint8_t *k = nullptr, *v = nullptr;  // K [pages][hkv][64][256] x kv_elem bytes; V [pages][hkv][64][v_row_bytes]
  float *ks = nullptr, *vs = nullptr;  // [pages][hkv][64] (int8 KV scales; unused for bf16); int8-v4's vs: bf16 [..][64][8]
  const int32_t *pages = nullptr;      // [slots][slot_pages]: logical page -> physical page (device)
  int max_tokens = 0;                  // capacity per slot: logical pages x 64
  int slot_pages = 0;                  // page table ints per slot (0: one slot)
  // Arch flint, windowed layers: > 0 = the slot's KV is a ring of this many logical pages (position p lives in logical
  // page (p / 64) % ring_pages; the page table then needs ring_pages entries per slot). It must hold a window plus the
  // rows of a step / chunk: ring_pages >= (window + rows - 2) / 64 + 2 (checked; window 513 at 8 rows: 10 pages).
  int ring_pages = 0;
  // Per physical page and kv head the elementwise min / max of its written keys, bf16 [pages][hkv][256]
  // (null: no bounds; then GqaStep::select must be 0). A prefill's K write (gqa_rows, gqa_kv_rows) keeps them; a decode step's pages get theirs from gqa_bounds_page.
  gqa_bf16 *kmin = nullptr, *kmax = nullptr;
};
size_t gqa_page_bound_bytes(const GqaShape &s);  // kmin + kmax bytes of one page (all hkv heads)
size_t gqa_page_kv_bytes(const GqaShape &s);      // K bytes of one page (all hkv heads; V too unless int8-v4)
size_t gqa_page_scale_floats(const GqaShape &s);  // K scales of one page (V too unless int8-v4)
size_t gqa_page_v_bytes(const GqaShape &s);  // V bytes of one page (int8-v4: half of K's)
size_t gqa_page_vscale_floats(const GqaShape &s); // V scale floats of one page (int8-v4: 4 per token-head, bf16 pairs)
size_t gqa_page_total_bytes(const GqaShape &s);   // K + V + their scales of one page (every pool's per-page price)

// Decode scratch for up to kMaxRows rows. Rows past the step's live rows of x and attn are read by the
// M = 4 GEMMs: keep them finite (zero once).
struct GqaScratch {
  float *y = nullptr;        // [8][qkv_n] projections
  gqa_bf16 *q = nullptr;     // [8][hq][256] normed, roped
  float *part = nullptr;     // gqa_part_floats: [kMaxRows][hkv][kGqaSplits][grp][272] o | max, sum
  gqa_bf16 *attn = nullptr;  // [8][hq x 256] gated attention = o_proj input
  float *partial = nullptr;  // [8][hidden]
  // GqaStep::select: score keys [kMaxSegs][hkv][kGqaSelPagesMax], the selection [kMaxSegs][hkv][kGqaSelMax]
  // and its counts [kMaxSegs][hkv]
  uint32_t *sel_key = nullptr;
  int32_t *sel = nullptr, *sel_n = nullptr;
};
size_t gqa_part_floats(const GqaShape &s, int rows = kMaxRows);  // the decode step's rows

struct GqaStep {
  const int32_t *pos = nullptr;  // device word: position of row 0; or segment records (rows.hpp)
  int pos_bound = 0;             // host bound: every row's position < pos_bound <= cache.max_tokens (grid sizes)
  int stride = 0;                // ints between segment records (kSegRec), 0 = a bare position word
  int seg_rows = 0;              // rows per segment (0 = one segment)
  // Rows per attention work-group. 0 = auto (gqa_attn_group); 1 / 2 / 4 / 8 = one K / V pass per
  // group of that many rows of a segment (gqa_seg.hpp; the same bits for every value and register file); -1 = the
  // per-row kernel (one work-group per row: R rows read the K / V R times; it differs from the group kernel by fp
  // order in a small share of the attention outputs: fast math, another loop structure).
  int attn_rows = 0;
  int attn_grf = 0;  // A/B: the group kernel's register file, 128 or 256 (0 = auto)
  int attn_pf = -1;  // A/B: the group kernel's L2 prefetch of the next tile (-1 = auto: gqa_attn_group, 0 off, 1 on)
  bool kv_only = false;  // projections + prep (the K / V rows) only; no attention, o or sum (out unused)
  // Pages of context per (segment, kv head) the attention walks (0 = every page: dense); needs the
  // cache's bounds and the scratch's sel buffers; used once pos_bound passes select pages (below that every segment would
  // keep all its pages: the dense path runs, bit for bit)
  int select = 0;
  // A page count per (slot, kv head) instead of `select` for all (heads differ in how spread
  // their attention is: ModelOptions::select_self), device int32 [slot][hkv()], every entry in kGqaSelSink +
  // kGqaSelLocal + 1 .. kGqaSelMax (the caller's duty: a kernel cannot refuse a device value). A unit is still
  // selected only once its context passes `select`; a head whose count its context does not pass keeps every page.
  // Two sequences of one step may keep different counts. Null = `select` for every head. The decode selection's; a
  // prefill call's row blocks keep the common count.
  const int32_t *select_slots = nullptr;
  // Segments may be draft trees (records' kSegTree, rows.hpp). Row j of a tree
  // sits at position p0 + depth(j) for its rope and its attention (its K / V slot stays p0 + j): it attends to the
  // context below p0 and its ancestors' keys in path order at p0 .. p0 + depth, exactly what row depth(j) of a chain over
  // its ancestors attends to, bit for bit (gqa_seg.hpp: the context tiles shared by the group, a side row's tiles from
  // p0 on gathered in path order). The group kernel only (exact numerics, int8 / bf16 KV, no selection, no window).
  bool tree = false;
  // Opt-in eager K8V4 trained-QSA handoff (trained_qsa_pages.hpp). Device
  // arrays from one plan for these segment records, kept live through merge.
  // One selection per query row, shared slot record for a verification chain.
  // Leave both null for an all-visible plan when the caller knows it.
  struct {
    const int32_t *entries = nullptr;  // [rows][kTrainedQsaPages]
    const int32_t *counts = nullptr;   // [rows][local_kv_heads], 0 = dense row
  } trained_qsa{};
};

void gqa_set_prep(int esimd);  // the decode steps' ESIMD prep (D 128) on / off (tests)

struct GqaEvents {
  sycl::event qkv, prep, attn, merge, o, reduce;
  sycl::event send;  // gqa_rows with PairRowsWire:send: the peer rows' copy
};

// Splits of a row at position p (host mirror of the device cut, for tests and floors).
int gqa_splits(int p);

// The attention group gqa_decode runs for a segment of seg rows by default, as (rows, grf), at a step's
// pos_bound (the register file follows the context: same bits either way).
struct GqaGroup {
  int rows = 1, grf = 0;
  bool pf = false;  // the next tile's K / V into L2 by the segment's first row group (gqa_seg.hpp)
};
GqaGroup gqa_attn_group(const GqaShape &s, int seg, int pos_bound);
// The 256-GRF instances of the group attention (gqa_seg.cpp); gqa_decode calls it. rm 1 / 2 / 4 / 8. tree (GqaStep::tree):
// the step may hold draft trees; 2- / 4-row groups then run the tree instance (same bits, side rows apart).
sycl::event gqa_attn_seg_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows,
                            const gqa_bf16 *qb, float *part, int ns, int rm, bool pf = false, bool tree = false);
// The page-selected instances (grp 6), and the stages before them (gqa_sel.cpp): the bounds of the pages
// of positions [pos0, pos0 + m) of one sequence (prefill), of one whole page of a slot in up to kGqaMoveCaches caches
// (decode, below), and the score + select of every segment's pages per kv head.
// A page's bounds in decode: a step does not write them. The selection never reads the bounds of a
// sequence's last kGqaSelLocal pages (it keeps those pages unscored), so the page a sequence is writing needs none until
// the sequence leaves it; the model calls gqa_bounds_page then, once per page, after the accept (and after a tree's
// gqa_kv_move: the page's keys are final).
sycl::event gqa_attn_seg_sel_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows,
                                const gqa_bf16 *qb, float *part, int ns, int rm, const int32_t *sel, const int32_t *sel_n, bool pf = false);
sycl::event gqa_attn_seg_trained_xl(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const GqaStep &step,
                                   const gqa_bf16 *qb, float *part, int ns);
// Adapt native qsa_select's device buffers (ascending IDs [rows][512]) in the
// same in-order queue. Unscored below-budget rows may have undefined IDs.
// status[r] is the page count (0 dense), or -1 for invalid native selection.
// The producer must guarantee valid inputs; do not use a malformed plan.
sycl::event gqa_pack_trained_qsa(sycl::queue &q, int rows, int hkv, const GqaStep &step,
                                const int32_t *sel, const int32_t *sel_n,
                                int32_t *entries, int32_t *counts, int32_t *status, uint32_t dense_rows = 0, bool gather = false);
sycl::event gqa_bounds_page(sycl::queue &q, const GqaShape &s, const GqaCache *caches, int n_caches, int slot, int page);
sycl::event gqa_bounds_range(sycl::queue &q, const GqaShape &s, const GqaCache &c, int pos0, int m);
sycl::event gqa_select(sycl::queue &q, const GqaShape &s, int rows, const GqaCache &c, const int32_t *rec, int stride, int seg_rows, const gqa_bf16 *qb,
                       int budget, int pos_bound, const GqaScratch &sc, const int32_t *slot_heads = nullptr);
// host reference of a page's score for one q head (tests): q . mid + spread x |q| . rad on the device's bf16 operands
float gqa_page_score_ref(const float *q, const float *kmin, const float *kmax);

// x bf16 [8][hidden] (rows < `rows` live, the rest finite); out bf16 [rows][hidden]. rope: device bf16
// [>= cache.max_tokens][64]. wire: null with one card. Every row computes what a step of its own
// sequence alone computes for it, bit for bit. grp() 6 (Qwen3.8-27B: 24 q / 4 kv heads) or 12.
GqaEvents gqa_decode(sycl::queue &q, const GqaShape &s, int rows, const gqa_bf16 *x, const GqaWeights &w, const GqaCache &c,
                     const gqa_bf16 *rope, const GqaStep &step, const GqaScratch &sc, const PairWire *wire, int wire_parity, gqa_bf16 *out,
                     int qkv_ks = 0, int o_ks = 0);

// After a tree step, the accepted path's K / V rows to their final slots. In each listed cache (one
// card's attention layers), for every kv head, the token at slot position p0 + from[i] moves to p0 + to[i], i ascending
// (to[i] = i < from[i]: a path's depth i; a source slot is never a destination written before it). Byte copies of the
// K / V codes and their scales (int8 / int8-v4 / bf16 KV; int8-v4 V is half the K row, with 8 bf16 scales). One launch for up to kGqaMoveCaches caches.
inline constexpr int kGqaMoveCaches = 16;
struct GqaMove {
  int n = 0;
  int from[kTreeRows] = {}, to[kTreeRows] = {};
};
sycl::event gqa_kv_move(sycl::queue &q, const GqaShape &s, const GqaCache *caches, int n_caches, int slot, int p0, const GqaMove &mv);

// ---- many rows (prefill) ---------------------------------------------------------
// Page-selected prefill attention (opt-in, lossy; gqa_rows' `select`). A many-row call whose
// context passes `select` pages attends per (attention row block, kv head) to `select` pages chosen exactly as a decode
// step's (the first kGqaSelSink, the last kGqaSelLocal up to the block's last row, the best page scores of the rest;
// a page's score = the largest over the block's rows' q heads). The K / V of the call's rows, and every later layer's
// input, are then built on the selected attention. A block whose context is `select` pages or fewer keeps them all,
// and a call that ends under the budget runs the dense body, bit for bit.
// Trained prefill: sorted union of logical pages per FA row group, with a
// separate 16-bit block mask for every (query row, logical page).
struct GqaTrainedPrefill {
  const int32_t *pages = nullptr, *counts = nullptr;
  const uint16_t *masks = nullptr;
  int pitch = 0, rows_per_group = 0;
  const uint32_t *subtiles = nullptr; // low 4 group bits, then 4 per bundle of eight rows; indexed by union walk
};
struct GqaRowsSelect {
  int pages = 0;  // pages per (row block, kv head) once the call's context passes that many; 0 = dense
  // The block holding the call's last row keeps every page. For the call that ends a prompt: its last rows produce the
  // first reply token, and a shared page set is weakest where one row needs one far page (an exact repeat of an earlier
  // span is copied by one head from one page).
  bool dense_last_block = false;
};
struct GqaPageSets {
  uint32_t *keys = nullptr;   // scratch: the pages' score keys, [blocks][hkv][key_stride]
  int32_t *pages = nullptr;   // per (row block, kv head) its pages, ascending: [blocks][hkv][kGqaSelMax]
  int32_t *counts = nullptr;  // [blocks][hkv]: the pages selected; 0 = every page (the dense walk)
  int key_stride = 0;         // score keys per (block, kv head): the cache's pages rounded up to 16
  int blocks = 0;             // row blocks the buffers hold
  bool valid() const { return keys && pages && counts && key_stride > 0 && blocks > 0; }
};
// The buffers' bytes for calls of up to rows_max rows in blocks of block_rows on a cache of max_tokens (layout: the
// stride and block count they give); bind carves them from `base`.
size_t gqa_page_sets_bytes(const GqaShape &s, int rows_max, int block_rows, int max_tokens, GqaPageSets *layout = nullptr);
void gqa_page_sets_bind(const GqaShape &s, int rows_max, int block_rows, int max_tokens, void *base, GqaPageSets &sets);
// Score + select of every row block of m rows of one sequence at pos0 (blocks of block_rows rows, the attention body's;
// q [m][hq][256] as prep wrote it; the cache's bounds must cover the call's own pages: gqa_bounds_range first).
sycl::event gqa_select_rows(sycl::queue &q, const GqaShape &s, int m, int pos0, int block_rows, const GqaCache &c, const gqa_bf16 *qb,
                            const GqaRowsSelect &select, const GqaPageSets &sets);
// m consecutive rows of one sequence at positions pos0 .. pos0 + m - 1 (slot 0). Stage A: q|k|v and o on the
// many-row W4A16 GEMM (w4a16_gemm_pf) over the chunk, one prep over all m rows (every row's K / V written before any
// row attends), then attention + merge per batch of kGqaRowsBatch rows (the decode kernels: a row's bits are the
// decode path's for the same projections), the two-card sum on the chunk-sized wire. pos4: device words, word j =
// pos0 + 4 j (the model's per-slice words; batch b reads word 16 b).
inline constexpr int kGqaRowsBatch = 64;
struct GqaRowsScratch {
  float *y = nullptr;        // [rows_max][qkv.n]
  gqa_bf16 *q = nullptr;     // [rows_max][hq][256]
  float *part = nullptr;     // gqa_part_floats(s, kGqaRowsBatch)
  gqa_bf16 *attn = nullptr;  // [rows_max][hq x 256]
  float *partial = nullptr;  // [rows_max][hidden]
  bool qkv_ready = false;    // y already holds x's q|k|v projection (the caller ran it in row parts)
  bool o_later = false;  // gqa_rows stops with the attention rows in attn; the caller runs gqa_rows_o
  // kGqaAttnSdpa only (gqa_sdpa_scratch_bytes): one kv head's context as contiguous f16 K | V [2][sdpa_tokens][256]
  // (gathered from the pages per call), q and o in f16 [rows_max][hq][256]
  sycl::half *sdpa_kv = nullptr, *sdpa_q = nullptr, *sdpa_o = nullptr;
  int sdpa_tokens = 0;
  GqaPageSets sel;  // gqa_rows' select > 0 only: the row blocks' page sets (gqa_page_sets_bind)
};
// wire->split: the fp32 partial stays in sc.partial and the caller finishes the sum (pair_split_rows), as gdn_rows.
// attn: which attention body (the A/B and the gates):
//   kGqaAttnBatch  (0) the decode attention + merge per 64-row batch (a row's bits = the decode path's for the same projections);
// kGqaAttnBlock (1) gqa_attn_rows, (blocks of rows share every K / V tile, dims split over the items);
// kGqaAttnStaged (2) gqa_attn_staged, (K / V tiles staged in SLM, pairs split over the items);
// kGqaAttnFa (3, default) gqa_attn_fa, (the staging, one item per (head, 8 rows) with all of o:
//                  the same bits as kGqaAttnStaged with p_lo).
// send (wire->split only): null, or the peer's rows of the o projection first and their copy sent on the side
// queue beside the own rows (as gdn_rows does); the event of the copy is returned in GqaEvents::send.
// kGqaAttnSdpa (4) gqa_attn_sdpa (oneDNN's fused GQA attention in f16; head_dim 256 with the gate only; needs
//                  the sdpa_* scratch).
// rows_per_group: the staged / fa body's row block (0 = its default). p_lo: the fa body's P as hi + lo bf16 (true, the
// other bodies' numerics) or hi only (false: a numerics switch). qk (fa body only, int8 KV): the q.k form, see
// gqa_attn_fa.
// select (above): the page selection; pages 0 = dense. The fa body at grp 6 / head_dim 256 only, int8 or bf16
// KV, with the cache's bounds and sc.sel.
inline constexpr int kGqaAttnBatch = 0, kGqaAttnBlock = 1, kGqaAttnStaged = 2, kGqaAttnFa = 3, kGqaAttnSdpa = 4;
GqaEvents gqa_rows(sycl::queue &q, const GqaShape &s, int m, const gqa_bf16 *x, const GqaWeights &w, const GqaCache &c, const gqa_bf16 *rope,
                   const int32_t *pos4, int pos0, const GqaRowsScratch &sc, const PairRowsWire *wire, int wire_parity, gqa_bf16 *out,
                   int attn = kGqaAttnFa, int rows_per_group = 0, bool p_lo = true, int qk = 0, const GqaRowsSelect &select = {},
                   const GqaTrainedPrefill *trained = nullptr);
// gqa_rows' last stage alone: the o projection of sc.attn's m rows into sc.partial and the
// two-card sum (or, wire->split, the partial left for the caller). One prefill call that carries the rows of several
// sequences runs gqa_rows per sequence with sc.o_later (its own cache and positions, its rows of y / attn at their
// offset, the other scratch shared: the calls run in order on the queue) and this once over all rows.
GqaEvents gqa_rows_o(sycl::queue &q, const GqaShape &s, int m, const GqaWeights &w, const GqaRowsScratch &sc, const PairRowsWire *wire, int wire_parity,
                     gqa_bf16 *out);
// Only the K / V of m consecutive rows (the MTP drafter's prefill pass): the q|k|v projection over the rows
// (pf_dense for dense weights, w4a16_gemm_pf for W4) and prep (norms, rope, the cache writes of every row; q to sc.q);
// no attention. Slot 0, positions pos0 + r (pos4: gqa_rows' words). Returns the qkv and prep events.
GqaEvents gqa_kv_rows(sycl::queue &q, const GqaShape &s, int m, const gqa_bf16 *x, const GqaWeights &w, const GqaCache &c, const gqa_bf16 *rope,
                      const int32_t *pos4, int pos0, const GqaRowsScratch &sc);
// The block attention (gqa_rows.cpp, 256 GRF): one work-group per (8 rows, kv head) for grp 6 (4 rows for grp 12) reads
// every K / V tile once for its rows x grp pairs; the decode attention's per-tile arithmetic, a causal mask per row, no
// context splits, the gate fused: attn [m][hq x 256] = bf16(bf16(o / l) bf16(sigmoid(bf16(gate)))). q [m][hq][256]
// (prep's), y the projections (gate columns), rows at positions pos_word[0] + r. Differs from the split decode path by
// fp32 order (one softmax pass instead of splits + LSE merge).
sycl::event gqa_attn_rows(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                          const gqa_bf16 *qb, gqa_bf16 *attn);
// The staged body (gqa_staged.cpp, GRF): one work-group per (rows_per_group rows, kv head) of 2 x pairs / 8
// items; per 64-token tile (one KV page) K and V are staged once into SLM as bf16 DPAS operands (64 KB: two work-groups
// per core), every item computes the full q.k of its 8 (row, q head) pairs, their online softmax in registers, and P.V
// for half of the 256 dims. Same semantics and finish as gqa_attn_rows (causal mask, no splits, gate fused); fp32 order
// differs (DPAS over 16-dim steps). rows_per_group: grp 6 -> 8 or 16 (0 = 16); grp 12 -> 4 or 8 (0 = 8). One slot.
sycl::event gqa_attn_staged(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                            const gqa_bf16 *qb, gqa_bf16 *attn, int rows_per_group = 0);
// gqa_attn_fa (gqa_fa.cpp, 256 GRF): the staged body's tiles and arithmetic with one item per (q head, 8 rows) owning all
// 256 dims of o (no duplicated q.k / softmax; a staged tile serves rows_per_group x grp pairs). With p_lo the same bits
// as gqa_attn_staged; p_lo = false: P.V on bf16 P only (numerics switch). rows_per_group: grp 6 -> 16, 32 or 40
// (0 = 40); grp 12 -> 8 or 16 (0 = 16). One slot. qk (numerics switch, int8 KV only): 0 = q.k as bf16 q x the
// widened K codes (the other bodies' arithmetic); 1 = q quantised to int8 per (row, q head) (s = amax / 127, rne) and s8 x s8
// DPAS on the K codes; 2 = q as hi + lo int8 codes (q / s = hi + lo / 127), two s8 DPAS per K operand: q.k within fp32
// noise of qk 0, only with p_lo = false.
// Arch flint (grp 8, head_dim 128): the layer's window (s.window), a KV ring (c.ring_pages, which must hold the
// window and the call's rows) and the sinks (one bf16 logit per q head of this card, null = none); rows_per_group 16 / 24 /
// 32 (0 = 32), bf16 q.k only. The other shapes take no window / ring / sinks.
// The tail split of gqa_attn_fa (default on, XEN_ATTN_TAIL=0 off): the call's last W mod C work-groups (C = one
// wave) run as context-split units merged after, so the last wave is full (gqa_fa.cpp FaTail). Outputs of the other rows
// are unchanged; the tail rows' o is summed in another order (online-softmax merge of the units).
// the fa body's context splits (process-wide; XEN_ATTN_TAIL, default 2): 0 = none, 1 = the last-wave tail of calls of
// one wave or more (a call's rows keep their bits at every call size below that), 2 = also every work-group of a call under
// one wave (those rows' bits then depend on the call's size)
void gqa_attn_fa_set_tail(int mode);
int gqa_attn_fa_tail();
// page_sets (null = every page): each row block walks its pages of gqa_select_rows instead of the
// whole context (the SEL instances: grp 6 at head_dim 256, 16 / 32 / 40 rows per group; exact numerics on either KV,
// int8 q.k + P hi only on the int8 KV, P hi only on the bf16 KV). A block whose count is 0 walks every page. The tail
// split's units then share a block's selected pages instead of its tiles.
sycl::event gqa_attn_fa(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word, const float *y, int ldy,
                        const gqa_bf16 *qb, gqa_bf16 *attn, int rows_per_group = 0, bool p_lo = true, int qk = 0, const gqa_bf16 *sinks = nullptr,
                        const GqaPageSets *page_sets = nullptr);
// gqa_attn_fa (16 rows per group) on a decode step's one segment of m <= 16 rows: positions and slot from the
// segment record rec (stride >= kSegRec; stride 0: rec is a bare position word, slot 0) over c's slot page table.
// tree: the segment may be a DDTree (record kSegTree): each row reads the context below the segment and its ancestors.
sycl::event gqa_attn_fa_decode(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *rec, int stride, const float *y, int ldy,
                               const gqa_bf16 *qb, gqa_bf16 *attn, bool p_lo, int qk, bool tree = false);
// The rows per work-group gqa_attn_fa runs for rows_per_group (0 = the shape's default): the row block that shares a page set.
int gqa_attn_fa_rows(const GqaShape &s, int rows_per_group);
sycl::event gqa_attn_fa_trained(sycl::queue &q, const GqaShape &s, int m, const GqaCache &c, const int32_t *pos_word,
                               const float *y, int ldy, const gqa_bf16 *qb, gqa_bf16 *attn, const GqaTrainedPrefill &plan);
// gqa_attn_sdpa (gqa_sdpa.cpp, oneDNN Graph): the prefill attention on oneDNN's fused GQA kernel
// (`sdpa,ocl:micro`), one kv head per call: gather the head's context [0, pos0 + m) from the pages into contiguous f16
// K | V (int8: f16(code x scale); bf16: f16(value)), q to f16, then softmax(q . K^T / 16) . V with oneDNN's bottom-right
// causal mask (row r sits at pos0 + r), f32 scores, f16 P and o; the gate as attn_fa's finish:
// attn = bf16(bf16(o) x bf16(sigmoid(bf16(gate)))). Head_dim 256 with the gate, one slot (pages row 0), no window /
// sinks. pos0 is the host's copy of the chunk's first position (the causal mask's lengths are host scalars).
// Compiled partitions are cached per (device, m, tokens).
size_t gqa_sdpa_scratch_bytes(const GqaShape &s, int rows_max, int tokens);  // the sdpa_* buffers' bytes
void gqa_sdpa_scratch_bind(const GqaShape &s, int rows_max, int tokens, void *base, GqaRowsScratch &sc);  // carve them
sycl::event gqa_attn_sdpa(sycl::queue &q, const GqaShape &s, int m, int pos0, const GqaCache &c, const float *y, int ldy,
                          const gqa_bf16 *qb, gqa_bf16 *attn, const GqaRowsScratch &sc);
// Its oneDNN call (gqa_sdpa.cpp, XL library): one kv head, q16 / o16 = the head group's first q head in [m][hq][256],
// k16 / v16 = [S][256], rows at the last m of the S context positions.
void gqa_sdpa_exec(sycl::queue &q, int m, int S, int grp, int hq, const sycl::half *q16, const sycl::half *k16, const sycl::half *v16,
                   sycl::half *o16);

}  // namespace xenithium::kernels
