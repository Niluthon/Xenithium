// Xenithium · the DFlash 2 block drafter's own kernels.
//
// DFlash 2 (z-lab / Inco AI, `DFlash2DraftModel`) drafts a whole block in one pass: block row 0 holds the last verified
// token (the anchor), rows 1..B-1 the mask token; five Qwen3-style decoder layers whose attention reads the block itself
// (non-causal) plus the "context": the fc-projected target hidden states of every verified position in a sliding window.
// Each row j >= 1 of the final hidden gives logits (the target's lm_head) for the token AT position start + j; the top
// candidates per row and a pairwise path selector pick the drafts. Semantics from the checkpoint's config.json and
// tensors. The GEMMs run on the engine's dense kernels (dense.hpp, pf_dense.hpp); this file holds what is particular to
// the drafter:
//
//   df_copy_cols   a column range of residual rows into the feature matrix (the target layers' outputs, concatenated)
//   df_conv        the two-tap grouped dynamic convolution over a segment's rows (before / after each sublayer):
//                    out[t][c] = sum_{o=0,1} (base[o][c] x[t-o][c]  (bf16 product, bf16 sum)
//                                            + dyn[t][o][c / group] x[t-o][c])  (fp32 fma, one bf16 rounding)
//                  x[t-1] = 0 at a segment's first row; dyn = the kernel projection's output rounded to bf16 (a
//                  bf16 `addcmul` chain, offsets in order)
//   df_prep        q / k / v from the projections: per head the Qwen3 RMSNorm (bf16(w x bf16(x rsqrt(mean x^2 + eps))))
//                  and the rotary embedding (rotate_half over the full 128 dims, bf16 cos / sin of fp32 angles, bf16
//                  products and sum), K / V of context rows into the slot's ring
//   df_attn        a segment's block rows over [the ring keys of the window | the block's own keys], softmax in fp32,
//                  in 32-key chunks per (segment, kv head) + a merge; out bf16 (<= 32 queries = rows x group per kv head)
//   df_topk        the 16 best logits of every row over both cards (bf16 keys as greedy_pick's: ties to the lower index),
//                  exchanged through a mailbox; both cards end with the same 16 per row, best first
//   df_select      the candidate path selector (card 0): for row j, score(b) = logit(b) + sum_r bf16(P[pred][r] bf16(h_j[r]))
//                  S[b][r] (fp32 sum), pred = the anchor for j = 1, then the pick of row j - 1; greedy argmax (ties to the
//                  better logit); a segment with a noise record (inv_t > 0) picks argmax over its
//                  candidates of score(b) x inv_t + gumbel(key, pos + j - 1, token(b)) instead (coupled drafts for a
//                  sampled request: the target's draw uses the same noise)
// Rings: per (slot, layer) K and V bf16 [ring][hkv][128], position p at p % ring; the window (2,048) + a block fit, so a
// pass never reads a slot it writes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/head/head.hpp"  // DraftNoiseRec
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {

using df_bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kDfDim = 128;      // head dim (kernel constant)
inline constexpr int kDfTopK = 16;      // candidates per row (kernel constant: the checkpoint's selector_top_k)
inline constexpr int kDfChunk = 32;  // ring keys per attention work-group
inline constexpr int kDfMaxQ = 32;      // block rows x q heads per kv head, per attention work-group
inline constexpr int kDfMaxGroup = 8;   // q heads per kv head (attention SLM sizing)
// The draft pass's segment records (kSegRec ints each, rows.hpp): block start position, context row 0's position, the slot.
inline constexpr int kDfRecStart = 0, kDfRecCtx = 1, kDfRecSlot = 4;

// dst[r][dst_col + i] = src[r][src_col + i] for r < rows, i < n (row strides lds / ldd elements).
sycl::event df_copy_cols(sycl::queue &q, int rows, const df_bf16 *src, int lds, int src_col, df_bf16 *dst, int ldd, int dst_col, int n);
// qwen38 (a residual of `streams` hyper-connection streams, [row][stream][hidden]): the streams' mean,
// dst[r][dst_col + i] = bf16((src[r][0][col + i] + ... + src[r][S-1][col + i]) / S) (fp32 sum in stream order, one
// rounding), for r < rows, i < n; src row stride streams x hidden.
sycl::event df_mean_cols(sycl::queue &q, int rows, const df_bf16 *src, int streams, int hidden, int src_col, df_bf16 *dst, int ldd, int dst_col, int n);

// rows in segments of seg_rows (0 = one segment); x / out bf16 [rows][hidden]; dyn fp32 [rows][ld_dyn], the tap o, group g
// weight of side `which` at column (which x 2 + o) x (hidden / group) + g; base bf16 [2][hidden] (this side's taps).
sycl::event df_conv(sycl::queue &q, int rows, int seg_rows, int hidden, int group, const df_bf16 *x, const float *dyn, int ld_dyn, int which,
                    const df_bf16 *base, df_bf16 *out);
// host reference: bf16 tensors as uint16 bits, the same rounding points
void df_conv_ref(int rows, int seg_rows, int hidden, int group, const uint16_t *x, const float *dyn, int ld_dyn, int which, const uint16_t *base,
                 uint16_t *out);

struct DfRing {
  df_bf16 *k = nullptr, *v = nullptr;  // [slots][ring][hkv][128] (this layer)
  int ring = 0;                        // tokens per slot
  int hkv = 0;
};

// rope: cos / sin bf16 [positions][64] (dims i and i + 64 share them).
struct DfRope {
  const df_bf16 *cos = nullptr, *sin = nullptr;
  int positions = 0;
};

// Block rows (yq != null): yq fp32 [rows][hq x 128], ykv fp32 [rows][2 hkv x 128] (k | v) -> qo bf16 [rows][hq][128] (normed,
// roped at start + j), kb / vb bf16 [rows][hkv][128] (k normed + roped; v rounded). Context rows (yq null): ykv of `rows` rows
// in segments of seg_rows at ctx position + j -> the slot's ring (kb / vb unused). rec: the pass's segment records.
sycl::event df_prep(sycl::queue &q, int rows, int seg_rows, const int32_t *rec, int hq, int hkv, const float *yq, const float *ykv, const df_bf16 *qn,
                    const df_bf16 *kn, float eps, const DfRope &rope, df_bf16 *qo, df_bf16 *kb, df_bf16 *vb, const DfRing &ring);

// Attention of rows in segments of seg_rows: block row j of a segment (start s) reads the ring keys at positions
// (s + j - window, s) clipped at 0, then the segment's seg_rows block keys (kb / vb); causal (DSpine): block
// row j reads block keys 0..j only. part: df_attn_part_floats.
size_t df_attn_part_floats(int rows, int seg_rows, int hq, int window);
sycl::event df_attn(sycl::queue &q, int rows, int seg_rows, const int32_t *rec, int hq, int hkv, const df_bf16 *qb, const df_bf16 *kb, const df_bf16 *vb,
                    const DfRing &ring, int window, float *part, df_bf16 *out, bool causal = false);
// host reference of one segment (q / kb / vb / ring as uint16 bits; ring_k / ring_v [ring][hkv][128] of the slot)
void df_attn_ref(int seg_rows, int start, int hq, int hkv, const uint16_t *q, const uint16_t *kb, const uint16_t *vb, const uint16_t *ring_k,
                 const uint16_t *ring_v, int ring, int window, uint16_t *out, bool causal = false);

// Top-16 over both cards. Card c holds columns [col_base, col_base + n) of every row (fp32 logits, row stride ld).
// Four cards = two hops (card ^ 1, then card ^ 2 with the merged list; peer2_inbox set, the inbox
// kDfTopkWireWords4): the merge is exact, so every card holds the same 16 keys.
struct DfTopkWire {
  uint32_t *inbox = nullptr;        // this card's, kDfTopkWireWords (kDfTopkWireWords4 at four cards; kMailEmpty when free)
  uint32_t *peer_inbox = nullptr;   // card ^ 1's
  uint32_t *peer2_inbox = nullptr;  // card ^ 2's (four cards; null: two)
  int32_t *step = nullptr;          // device counter (parity)
  uint32_t *error = nullptr;
  uint64_t wait_cycles = 2800000000ull;
};
inline constexpr size_t kDfTopkWireWords = 2 * kMaxRows * kDfTopK * 2;
inline constexpr size_t kDfTopkWireWords4 = 2 * kDfTopkWireWords;  // hop 2's slots after hop 1's
sycl::event df_topk_wire_init(sycl::queue &q, uint32_t *inbox, size_t words = kDfTopkWireWords);
size_t df_topk_scratch_bytes(int n);
// keys out uint64 [rows][16], best first (greedy_pick's key: ordered bf16 << 32 | 0xFFFFFFFE - index); wire null = one card.
sycl::event df_topk(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, void *scratch, const DfTopkWire *wire, uint64_t *keys);
// key -> (bf16 logit value, global index)
float df_key_value(uint64_t key);
inline int64_t df_key_index(uint64_t key) { return static_cast<int64_t>(0xFFFFFFFEu - static_cast<uint32_t>(key)); }
// host reference: the 16 best keys of one row of n logits at col_base (best first)
void df_topk_ref(const float *row, int n, int col_base, uint64_t *keys);
uint64_t df_key(float logit, int64_t index);

// Selector over the rows of `segs` segments of seg_rows rows: keys [rows][16] (best first), hp fp32 [rows][rank] (the hidden
// projection; rounded to bf16 on use), codebooks bf16 [vocab][rank], anchors int32 [segs]; candidate index -> token =
// v < sub_k ? v : v - sub_k + sub_s0 (sub_k 0: identity). drafts int32 [segs][seg_rows - 1].
// noise (optional, device [segs]): segment i's record (key, the output position of its first draft, 1 / T);
// inv_t 0 or a null table = the argmax pick, bit for bit.
sycl::event df_select(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *hp, int rank, const df_bf16 *pred_cb, const df_bf16 *succ_cb,
                      const int32_t *anchors, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise = nullptr);
void df_select_ref(int seg_rows, const uint64_t *keys, const float *hp, int rank, const uint16_t *pred_cb, const uint16_t *succ_cb, int32_t anchor, int sub_k,
                   int sub_s0, int32_t *drafts, const DraftNoiseRec *noise = nullptr);

// DDTree (Ringel & Romano 2026): a best-first draft tree per segment instead of the selector's chain. Node 0 = the
// anchor (depth 0, log-weight 0); a node at depth d < seg_rows - 1 has the 16 candidates of block row d + 1, each
// scored as df_select scores it given the node's token (logit + the selector term), as log-softmax lp over the 16; best
// first per node (ties: the lower candidate). Then up to `budget` times: the frontier's best (each node's next unused
// candidate, weight = the node's log-weight + its lp; ties: the lower node) becomes a node (log-weight = that weight);
// with cut > 0 only candidates whose path probability at temperature 1 (the product of their own and their ancestors'
// softmax probabilities over their rows' 16 scores: the chain's conf_cut measure) is >= cut take part.
// tree int32 [segs][kDfTreeInts]: [0] = n nodes past the root, [1 + i] = node i + 1's token, [1 + kTreeRows + i] = its
// parent (0 = the root), in the order they joined (a parent before its children). budget 1 .. kTreeRows - 1. Greedy
// only (no noise). inv_temp scales the 16 scores before the log-softmax (the tree's weights only: each node's
// best-first order is unchanged; 1 = no scaling; DFlashOptions::tree_temp).
inline constexpr int kDfTreeInts = 1 + 2 * kTreeRows;
sycl::event df_tree(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *hp, int rank, const df_bf16 *pred_cb, const df_bf16 *succ_cb,
                    const int32_t *anchors, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp = 1.0f);
void df_tree_ref(int seg_rows, const uint64_t *keys, const float *hp, int rank, const uint16_t *pred_cb, const uint16_t *succ_cb, int32_t anchor, int sub_k,
                 int sub_s0, int budget, float cut, int32_t *tree, float inv_temp = 1.0f);

// The deferred context. segs segments of seg_rows feature rows each (row i x seg_rows + r
// of src, stride cols bf16) to dst rows dst_row[i] + r (stride cols): one launch for a step's context rows.
sycl::event df_ctx_stash(sycl::queue &q, int segs, int seg_rows, int cols, const df_bf16 *src, df_bf16 *dst, const int64_t (&dst_row)[kMaxSegs]);
// dst row i = src row map[i] (i < n <= kMaxRows, stride cols bf16), one launch: a step's feature
// rows re-laid for the passes after a verify of unequal segments.
sycl::event df_rows_gather(sycl::queue &q, int n, int cols, const df_bf16 *src, df_bf16 *dst, const int (&map)[kMaxRows]);
// rec[0 .. kSegRec) = a context-only record (kDfRecCtx = ctx_p0, kDfRecSlot = slot, the rest 0), from kernel arguments.
sycl::event df_rec_set(sycl::queue &q, int32_t *rec, int64_t ctx_p0, int slot);

// XEN_DF_STAGE witness
sycl::event df_nanwatch(sycl::queue &q, int n, const void *p, unsigned *flag, unsigned bit, bool fp32 = false);
}  // namespace xenithium::kernels
