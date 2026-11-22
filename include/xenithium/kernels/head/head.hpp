// Xenithium · the ends of a decode step: embedding in, greedy token out.
// embed_gather: res[row][s][:] = table[ids[row]][:] for every residual stream s. The table
// may be pinned host memory (a step reads rows x hidden x 2 bytes over PCIe) or VRAM.
// Greedy pick over vocab-parallel logits: card c holds columns [col_base, col_base + n) of
// every row, as fp32 sums of its lm_head slice. A logit is rounded to bf16 (the logits' dtype)
// and keyed as one 64-bit unsigned value
// key = ordered(bf16) << 32 | (0xFFFFFFFE - global_index)
// with ordered order-preserving on the 16 bits (-0 == +0, NaN lowest), so the plain max of
// keys is the argmax: largest value, ties to the lowest index. Both words differ from the
// mailbox sentinel 0xFFFFFFFF by construction (high word < 2^16, index < 2^32 - 1).
// launch 1 (pick_chunks): one work-group per (row, 4096-column chunk) -> the chunk's key;
// launch 2 (pick_merge): one work-group; per row the max over chunks, then with a wire the
// card stores its key into the peer's inbox, spins (time-bounded) on its own inbox, takes
// the max of both and resets its inbox. Inboxes are double-buffered by step parity (a
// device counter the launch advances), so a card running a step ahead cannot overwrite a
// key the other card has not consumed yet. Both cards end with the same tokens.
// No queue ever waits on the other card's event (that loses the device on B70): the order
// across cards is carried by the mailbox words only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/rows.hpp"  // kMaxRows (kPickMaxRows)

namespace xenithium::kernels {

using head_bf16 = sycl::ext::oneapi::bfloat16;

sycl::event embed_gather(sycl::queue &q, const head_bf16 *table, const int32_t *ids, int rows, int hidden, int streams, head_bf16 *res);
// Image input: ids >= base read row (id - base) of `extra` (arch:VisionStore's image rows on this card)
// instead of the table. extra null = embed_gather.
sycl::event embed_gather_ext(sycl::queue &q, const head_bf16 *table, const int32_t *ids, int rows, int hidden, int streams, head_bf16 *res, const head_bf16 *extra,
                             int32_t base);

inline constexpr int kPickChunk = 4096;
inline constexpr int kPickMaxRows = kMaxRows;
inline constexpr uint32_t kMailEmpty = 0xFFFFFFFFu;

int pick_chunks(int n);  // chunks per row for n columns
size_t pick_scratch_bytes(int n);

// One card's end of the cross-card pick. inbox lives on this card (peer-resident, the peer
// stores into it), peer_inbox on the other card; both 2 x kPickMaxRows x 2 words (four cards: kPickWireWords4), initialised
// to kMailEmpty (pick_wire_init). step: one device int, initialised 0. error: device word,
// bit 1 set when the wait timed out (the row then keeps this card's own key).
// Four cards (TP4) = two hops (recursive doubling, the max is exact): hop 1 as above with peer_inbox =
// card ^ 1's inbox, hop 2 the pair's best with peer2_inbox = card ^ 2's inbox, into the inbox's second region (the inbox
// then kPickWireWords4 words). Every card ends with the max of the four keys.
struct PickWire {
  uint32_t *inbox = nullptr;
  uint32_t *peer_inbox = nullptr;
  int32_t *step = nullptr;
  uint32_t *error = nullptr;
  uint64_t wait_cycles = 2800000000ull;  // >= 1 s at 2.8 GHz
  uint32_t *peer2_inbox = nullptr;  // four cards: card ^ 2's inbox (null = two cards)
};
inline constexpr size_t kPickWireWords = 2 * kPickMaxRows * 2;
inline constexpr size_t kPickWireWords4 = 2 * kPickWireWords;  // four cards: a region per hop

// logits fp32 [rows][ld]; tokens int32 [rows] (global indices). wire null = one card.
// Returns the merge launch; chunks_event (optional) receives the first launch's event.
sycl::event greedy_pick(sycl::queue &q, const float *logits, int ld, int rows, int n, int col_base, void *scratch, const PickWire *wire,
                        int32_t *tokens, sycl::event *chunks_event = nullptr, uint64_t *local_keys = nullptr);

// Coupled drafting for sampled requests. A slot's record: the Gumbel key, the output position its first
// draft predicts, 1 / T (0 = the slot keeps its argmax draft). util/gumbel.hpp is the shared noise.
struct DraftNoiseRec {
  uint64_t key = 0;
  uint64_t pos = 0;
  float inv_t = 0;
  float pad_[3] = {0, 0, 0};
};
// In place before greedy_pick on a drafter pass's logits [rows][ld] (n columns of this card, column c's token id =
// cv < sub_k ? cv: cv - sub_k + sub_s0 with cv = col_base + c; sub_k 0 = cv itself): row r of a slot with inv_t > 0
// becomes logit * inv_t + gumbel(key, pos + pass, token); other rows are untouched (their bits and picks unchanged).
// The slot of row r: seg[(r / seg_rows) * seg_stride + kSegSlot] (segment records), or slot_fixed when seg is null.
void draft_gumbel(sycl::queue &q, float *logits, int ld, int rows, int n, int col_base, int sub_k, int sub_s0, const int32_t *seg, int seg_stride,
                  int seg_rows, int slot_fixed, const DraftNoiseRec *noise, int pass);

// ---- the two-stage lm_head's shortlist ---------------------------------
// head_select_tiles: per row r < rows of logits [rows][ld] (n columns on this card), the T 16-column tiles (tile t = columns
// [16 t, 16 t + 16)) with the largest maxima, exactly (ties to the lowest tile), into tiles[r * T + i] in ascending order.
// Two launches: every tile's maximum as an order-preserving key plus the maximum of each run of 16 tiles (a super tile);
// then one 16-lane sub-group per row takes the top T super tiles and the top T tiles inside them (the top T tiles always
// lie in the top T super tiles: fewer than T super tiles can exceed a top-T tile's maximum). Sub-group reductions only
// (a work-group-wide radix select over the tile keys of a card's head slice would have its
// histogram bins all hit by keys sharing their high bits). n % 16 == 0, 16 T <= n / 16, T 1..kHeadTilesMax.
// Every tile of a row's top T token logits is among its top T tiles (a token's tile maximum is >= the token).
// A row past 8,192 tiles (TP1: the whole 248,320-column vocabulary on one card) runs as P = head_select_parts(n) equal
// parts, each its own top T (the same argument per part, as per card under TP2): tiles holds P x T per row,
// tiles[r * P T + p T + i], part p's tiles ascending (absolute tile ids). P = 1 up to 8,192 tiles (TP2's 7,760).
inline constexpr int kHeadTilesMax = 32;
int head_select_parts(int n);
size_t head_select_scratch_bytes(int n);
// Returns the second launch; max_event (optional) receives the first's.
sycl::event head_select_tiles(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles,
                              sycl::event *max_event = nullptr);

// head_cand_tiles: the host sampler's candidates of a sampled step without copying the
// rows. head_select_tiles' selection on the bf16-ROUNDED tile maxima (the values the host sampler compares; -0 == +0, NaN
// never selected; ties to the lowest tile = the lowest ids), then the T tiles' raw fp32 values gathered into
// vals [rows][T][16] (tile tiles[r * T + i], column j at vals[(r * T + i) * 16 + j]; P x T per row past 8,192 tiles).
// Exactness: under the host's order
// every token outside a card's T tiles is below each selected tile's best
// token, so for k <= T the k best of the union of both cards' candidates are the row's k best. Returns the gather;
// select_event (optional) receives the selection's last launch.
sycl::event head_cand_tiles(sycl::queue &q, const float *logits, int ld, int rows, int n, int T, void *scratch, int32_t *tiles, float *vals,
                            sycl::event *select_event = nullptr);

// ---- host reference --------------------------------------------------------------------------
// head_select_tiles' set for one row: the T tiles with the largest fp32 maxima (ties to the lowest tile), ascending (per part).
void head_select_ref(const float *row, int n, int T, int32_t *tiles);
// head_cand_tiles' set for one row: the T tiles with the largest bf16-rounded maxima (-0 == +0; ties to the lowest tile), ascending (per part).
void head_cand_ref(const float *row, int n, int T, int32_t *tiles);
uint64_t pick_key(float logit, int64_t global_index);  // the device key of an fp32 logit
// argmax of bf16-rounded logits over one row, ties to the lowest index.
int64_t pick_ref(const float *row, int64_t n, int64_t col_base = 0);

}  // namespace xenithium:kernels
