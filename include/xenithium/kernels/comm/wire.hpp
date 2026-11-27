// Xenithium · two-card sum of fp32 partials through the mailbox (generic all-reduce).
//
// out = bf16(p_self + p_peer) for `rows` x n columns. fp32 addition commutes, so both cards
// write the same bits. Protocol (same as the MoE combine): every word is its own flag. A card
// stores its partial into the peer's inbox (uncached), spins on its own inbox until no word is
// kPairMailEmpty (time-bounded: `error` is set and the words become NaN), then empties it.
// Slots are double-buffered by `parity`; alternate it per call on the same inbox.
//
// With wire == nullptr (one card) it only rounds: out = bf16(p_self).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/rows.hpp"  // kMaxRows (kPairRows)

namespace xenithium::kernels {

inline constexpr uint32_t kPairMailEmpty = 0xFFFFFFFFu;
inline constexpr int kPairRows = kMaxRows;  // rows a decode step can carry (rows.hpp): the inbox layout

// The four-card form of the mailbox sums (qwen38 TP4). A wire whose quad is on sums over four cards by
// recursive doubling: hop 1 with the partner (card ^ 1), hop 2 the pair's sum with the other pair (card ^ 2). Every card
// writes bf16((p0 + p1) + (p2 + p3)), the same bits on all four whatever the rows or the columns per item. An inbox then
// holds a slot region per hop (its words x kQuadHops, hop 2's region after hop 1's).
inline constexpr int kWireCardsMax = 4;
inline constexpr int kQuadHops = 2;
struct WireQuad {
  int card = -1;                                  // this card in [0, 4); -1 = off (the two-card wire: peer_inbox)
  std::array<uint32_t *, kWireCardsMax> inbox{};  // every card's inbox, this card's included
  bool on() const { return card >= 0; }
  // on, with this card in range and every card's inbox named (a null peer inbox would fault the device)
  bool complete() const {
    if (card < 0 || card >= kWireCardsMax) return false;
    for (const uint32_t *ib : inbox)
      if (!ib) return false;
    return true;
  }
};
// The slot regions an inbox needs for `cards` cards (1 or 2: one; 4: one per hop).
inline constexpr int wire_hops(int cards) { return cards == kWireCardsMax ? kQuadHops : 1; }

// ---- a decode step's rows by owner (four cards) --------------------------------------------
// Work that is the same on every card (qwen38's residual mixers) runs each row of a step on one card,
// the row's owner: card c owns rows [rows c / 4, rows (c + 1) / 4) (owner_rows; a kernel finds a row's card with
// row_owner). Two mailbox launches carry it. Both are replay-safe, unlike the prefill's split (whose flag lines hold
// sequence numbers the host bakes into the launch): every word is its own flag, as in the sums above.
//   owner sum    pair_reduce on a wire whose `owner` is on (the decode MoE's combine likewise): a card stores its fp32
//                partial of each row it does not own into the owner's inbox slot [parity][source card][the owner's row],
//                the owner takes its three sources and writes bf16((p0 + p1) + (p2 + p3)) - the four-card mailbox's order
//                and bits - for its own rows only. One hop and 3/8 of the two-hop sum's bytes.
//   owner_share  the owner stores its rows of a bf16 buffer (a mixer's output, which every card needs as the next module's
//                input) into every other card's inbox slot [parity][row]; the others take them into the same rows of
//                their own buffer.
// The rows of a `rows`-row step card `card` owns (rows.hpp RowRange), and the card that owns row r (inverse of each other).
constexpr RowRange owner_rows(int card, int rows) { return {rows * card / kWireCardsMax, rows * (card + 1) / kWireCardsMax}; }
constexpr int row_owner(int r, int rows) { return (kWireCardsMax * r + kWireCardsMax - 1) / rows; }
// One use's mailbox on every card (a sum's, or one shared buffer's): each use needs its own, because a slot holds one
// pending message. Off by default (card -1).
struct OwnerMail {
  int card = -1;                                  // this card in [0, 4)
  std::array<uint32_t *, kWireCardsMax> inbox{};  // every card's inbox for this use, this card's included
  bool on() const { return card >= 0; }
  bool complete() const {  // on, this card in range, every inbox named (a null peer inbox would fault the device)
    if (card < 0 || card >= kWireCardsMax) return false;
    for (const uint32_t *ib : inbox)
      if (!ib) return false;
    return true;
  }
};

// ---- the copy path of the owner sums and shares ---------------------------------------------------------------------
// From kOwnerCopyMinRows rows an exchange moves faster as copies than through a mailbox (512-byte stores per item):
// a card copies the rows another card needs straight into that card's memory (three peer copies on its queue), one
// launch then stores its send count to its flag line on each of them, and one one-item launch waits until this
// card's three lines hold its own send count (every card sends once per exchange of a use, so the counts agree). The
// count is a device word, so a recorded step replays the launches; the lines are mailbox memory.
// Rows that arrived through the link are not in this card's L1 caches' view (a core may still hold the bytes it read
// there before): a kernel reads them past L1 once - the sum itself, or the share's take launch, which stores them
// back so every later reader sees them.
struct OwnerFlags {  // one use's
  std::array<uint32_t *, kWireCardsMax> lines{};  // every card's flag lines: kOwnerCopyLineBytes of mailbox memory, zeroed once
  uint32_t *count = nullptr;  // this card: kOwnerCopyCountBytes of device memory, zeroed once (its sends)
  bool on() const { return count != nullptr; }
};
// Measured in the step: 16 rows +0.2 %, 20 rows -0.3 %, 24 rows -0.7 %, 28 rows -1.2 %, 64 rows -3.1 %.
inline constexpr int kOwnerCopyMinRows = 24;
inline constexpr size_t kOwnerCopyLineBytes = kWireCardsMax * 64, kOwnerCopyCountBytes = 64;
// A share by copies. The owner writes its rows into the same rows of every other card's buffer, so the caller's order
// must keep a card from writing while a peer still reads the last share: between two shares into one buffer every card
// finishes an exchange that blocks on every peer after that peer's readers of it (the owner sum of the module that
// reads the buffer). Off by default.
struct OwnerCopy {
  std::array<sycl::ext::oneapi::bfloat16 *, kWireCardsMax> buf{};  // this call: every card's buffer, this card's included
  OwnerFlags flags{};
  bool on() const { return flags.on(); }
};
// An owner sum by copies: each card copies its partial's rows of another owner into that owner's receive buffer (a
// region per parity and source card: a card's next sum cannot land where its peer still reads), the owner sums its own
// rows with the three received ones in the mailbox's order: the same bits. Off by default.
struct OwnerSumCopy {
  std::array<float *, kWireCardsMax> recv{};  // every card's receive buffer of this use: owner_sum_copy_elems(n_max) fp32
  OwnerFlags flags{};
  bool on() const { return flags.on(); }
};
struct PairWire {
  uint32_t *inbox = nullptr;       // this card's (peer-written), 2 x 8 x n_max words
  uint32_t *peer_inbox = nullptr;  // the other card's
  uint32_t *error = nullptr;       // device word, 1 after a timed-out wait
  uint64_t wait_cycles = 2800000000ull;
  // The bf16 wire: each card sends bf16(p_self) and both write bf16(bf16(p_self) + bf16(p_peer))
  // (fp32 add): an all-reduce of bf16 partials, half the bytes. Both cards must agree.
  // Four cards (quad): hop 1 sends bf16(p), hop 2 the pair's fp32 sum: bf16((bf16 p0 + bf16 p1) + (bf16 p2 + bf16 p3)).
  bool bf16 = false;
  // The int8 wire (pair_reduce only; it rules over bf16). Per 32 columns each card sends 8-bit
  // codes of its partial and their scale (bf16(absmax / 127)), and both write bf16(codes x scale (own) + codes x scale
  // (peer)): 72 bytes per 64 columns against the bf16 wire's 128. Lossy (each partial rounded to 8 bits per
  // 32-column group): opt-in. Both cards must agree. Two cards only (refused with quad).
  bool i8 = false;
  WireQuad quad{};  // four cards (inbox / peer_inbox unused)
  // On = the owner sum (above; it rules over quad): out gets this card's own rows only. fp32 partials
  // (bf16 / i8 refused); rows >= 4, so every card owns one. Each inbox: owner_sum_words(n_max), owner_mail_init.
  OwnerMail owner{};
  OwnerSumCopy copy{};  // on (with owner): sums of kOwnerCopyMinRows rows and more go by copies
};
// words of one card's inbox for `cards` cards (pair_wire_init fills them)
size_t pair_wire_words(int n_max, int cards = 2);
sycl::event pair_wire_init(sycl::queue &q, int n_max, uint32_t *inbox, int cards = 2);

// part fp32 [rows][n] (rows < `rows` read); out bf16 [rows][n]; rows 1..8. n % 64 == 0, n <= the inbox's n_max.
sycl::event pair_reduce(sycl::queue &q, int rows, int n, const float *part, const PairWire *wire, int parity,
                        sycl::ext::oneapi::bfloat16 *out);

// Rows by owner (above). The fewest rows an owner launch takes: every card owns at least one.
inline constexpr int kOwnerMinRows = kWireCardsMax;
inline constexpr int kOwnerRowsMax = (kPairRows + kWireCardsMax - 1) / kWireCardsMax;  // the most rows one card owns
// words of one card's inbox for an owner sum of up to n_max columns / for shared rows of up to n_max bf16 values
size_t owner_sum_words(int n_max);
size_t owner_share_words(int n_max);
size_t owner_sum_copy_elems(int n_max);  // fp32 elements of an owner sum's receive buffer (OwnerSumCopy::recv)
sycl::event owner_mail_init(sycl::queue &q, uint32_t *inbox, size_t words);
struct OwnerShare {
  OwnerMail mail;             // inboxes of owner_share_words(n_max) each
  uint32_t *error = nullptr;  // device word, 1 after a timed-out wait
  uint64_t wait_cycles = 2800000000ull;
  OwnerCopy copy{};           // on: shares of kOwnerCopyMinRows rows and more go by copies (the mailbox is not touched)
};
// buf bf16 [rows][n] on every card: this card's own rows (owner_rows) go to the other cards, their rows arrive in buf.
// rows kOwnerMinRows..kPairRows, n % 128 == 0. parity: alternate per call on the same mailbox (or leave a sum between two
// calls: a card's next owner sum completes only after every other card took the rows).
sycl::event owner_share(sycl::queue &q, int rows, int n, sycl::ext::oneapi::bfloat16 *buf, const OwnerShare &share, int parity);
// The columns one work-item of pair_reduce / pair_reduce_rows exchanges: its slot bytes
// grow with the rows (64 columns at 1-2 rows, 512 B at 3-5, 1 KiB from 6; halved while n is not a multiple). Both cards
// compute it from the same (rows, n, wire): their items line up.
int pair_cols(int rows, int n, bool bf16);
int pair_cols_i8(int rows, int n);  // the int8 wire's (PairWire::i8): 64 columns to 3 rows, 256 to 5, 512 to 19, 1,024 from 20; halved while n is not a multiple

// ---- many rows (prefill) ---------------------------------------------------------------
// The same exchange for up to rows_max rows: inbox / peer_inbox 2 (parity) x rows_max x n_max words.
struct PairSplitSend;  // below
struct PairRowsWire {
  uint32_t *inbox = nullptr;
  uint32_t *peer_inbox = nullptr;
  uint32_t *error = nullptr;
  int rows_max = 0, n_max = 0;
  uint64_t wait_cycles = 11200000000ull;  // >= 4 s at 2.8 GHz
  bool split = false;  // see pair_reduce_rows
  const PairSplitSend *send = nullptr;  // (split only): see PairSplitSend
  // the bf16 wire (bf16(own) sent packed, bf16(own) + bf16(peer) summed in fp32), as
  // PairWire::bf16; both cards alike
  bool bf16 = false;
  WireQuad quad{};  // four cards, as PairWire::quad (split stays two-card)
};
size_t pair_rows_wire_words(int rows_max, int n_max, int cards = 2);
sycl::event pair_rows_wire_init(sycl::queue &q, int rows_max, int n_max, uint32_t *inbox, int cards = 2);
// part fp32 [m][n]; out bf16 [m][n]. n % 64 == 0, n <= n_max, m <= rows_max.
// split: the kernels that take this wire (gdn_rows, qsa_rows) leave their fp32 partial in
// their scratch and skip the sum; the caller finishes both cards with pair_split_rows.
sycl::event pair_reduce_rows(sycl::queue &q, int m, int n, const float *part, const PairRowsWire *wire, int parity,
                             sycl::ext::oneapi::bfloat16 *out);

// ---- many rows, split over the cards ---------------------------------------------------
// The mailbox above moves every partial word through uncached peer stores; a copy moves more bytes per
// second. Split: card 0 finishes rows [0, m/2), card 1 the rest. Each card copies the peer's rows of its
// fp32 partial into the peer's `recv`, the owner writes bf16(own + peer) (+ add) for its rows and copies
// them into the peer's `out`. Both cards end with every row, bit-identical to pair_reduce_rows (fp32
// addition commutes). Cross-card order is by flag lines in the mailbox (sequence numbers, no reset): SYCL
// events from the other card's queue as dependencies fault the device. Slower than the mailbox for few
// rows (two copies and four small launches): use it from kPairSplitMinRows rows. That is the generic
// default; an arch may set its own threshold (qwen35 / qwen38 kPfSplitMinRows).
inline constexpr int kPairSplitMinRows = 64;
// The producers send (side queue, overlapped) from this many rows; under it the halved
// projections and the barriers cost more than the overlap hides.
inline constexpr int kPairSplitSendMinRows = 128;
// The consumer takes the peer's rows late (pair_split_finish without its wait) from this many
// rows; under it its mixer halves cost more than the back copy they hide.
inline constexpr int kPairSplitDeferMinRows = 256;
inline constexpr int kPairSplitRecvWord = 0, kPairSplitOutWord = 16;  // flag lines (words) in `flags`
inline constexpr size_t kPairSplitFlagBytes = 128;
inline constexpr uint32_t kPairSplitSeqs = 4096;  // seq in [1, kPairSplitSeqs): callers wrap (pair_split_next_seq)
inline uint32_t pair_split_next_seq(uint32_t seq) { return seq % (kPairSplitSeqs - 1) + 1; }
inline constexpr size_t kPairSplitSeqTabBytes = kPairSplitSeqs * 16;
sycl::event pair_split_seq_tab_init(sycl::queue &q, uint32_t *tab);
struct PairSplit {
  float *recv = nullptr;          // this card: pair_split_recv_elems fp32 (written by the peer's copy)
  uint32_t *flags = nullptr;      // this card's mailbox, kPairSplitFlagBytes, zeroed once (peer-written)
  float *peer_recv = nullptr;     // the other card's recv
  uint32_t *peer_flags = nullptr; // the other card's flags
  uint32_t *error = nullptr;      // device word, 1 after a timed-out wait
  // Side queues only: this card's table of flag lines (pair_split_seq_tab_init), line i
  // holds i. A flag on the side queue is a 16-byte copy from it: a fill there runs as a kernel and
  // stalls the compute queue behind the copy.
  const uint32_t *seq_tab = nullptr;
  int rows_max = 0, n_max = 0;
  uint64_t wait_cycles = 11200000000ull;
  // Diagnostic: null, or 4 words per split sum for the waits' own spans, as
  // GT-clock cycles (rdtsc): [0, 1] the wait for the peer's partial (entry, exit), [2, 3] the wait for
  // the peer's rows (pair_split_wait). Written by the wait kernels themselves: no events, no barriers.
  uint64_t *wait_span = nullptr;
  // The bf16 wire. The peer's rows go as bf16 (a convert into `stage`, this card,
  // pair_split_recv_elems bf16, then half the bytes) and the owner sums bf16(own) + bf16(peer) in fp32, as the decode's
  // bf16 wire (PairWire::bf16). Both cards set it alike; recv then holds bf16.
  bool bf16 = false;
  sycl::ext::oneapi::bfloat16 *stage = nullptr;
  // cards 4 (TP4): card c owns rows [m c / 4, m (c + 1) / 4); each card copies every other card's rows of
  // its partial into that card's recv (a slot per source card), the owner writes bf16((p0 + p1) + (p2 + p3)) (+ add), the
  // four-card mailbox's order and bits (bf16 wire: each operand bf16), and copies its rows into every other card's out.
  // Flag lines per source card: recv at kPairSplitRecvWord + 4 s, out at kPairSplitOutWord + 4 s. recv_of / flags_of:
  // every card's (by the host); peer_recv / peer_flags unused. stage (bf16): pair_split_stage_elems.
  int cards = 2, card = 0;
  float *recv_of[kWireCardsMax] = {};
  uint32_t *flags_of[kWireCardsMax] = {};
};
size_t pair_split_recv_elems(int rows_max, int n_max, int cards = 2);
size_t pair_split_stage_elems(int rows_max, int n_max, int cards = 2);  // the bf16 wire's stage
// out = bf16(p[r] x scale[r] + (own + peer)) per row: the MoE combine's shared-expert term.
struct PairSplitAdd {
  const float *p = nullptr;      // [m][n] (this card's copy)
  const float *scale = nullptr;  // [m]
};
struct PairSplitEvents {
  sycl::event sum[kWireCardsMax], done[kWireCardsMax];  // done: out complete on that card (its later kernels are ordered after)
  sycl::event copy[kWireCardsMax], recv[kWireCardsMax], back[kWireCardsMax];  // per card: its partial copy, its wait for the peer's, its rows' copy back
};
// Every card in one call (sp[0].cards of them): q[c], part[c], out[c], sp[c] per card; seq: a number never used before on
// these flags (pair_split_next_seq from 0, so it differs from the line's last value). add: null, or one per card.
// side: null, or per card a second in-order queue of the same device. The copies and their
// flags (fills) go there, behind a barrier on q's work so far, so they run on the copy engine beside
// q's next kernels (a peer copy on q runs on the compute engine and stalls it).
PairSplitEvents pair_split_rows(sycl::queue *const q[2], int m, int n, const float *const part[2], sycl::ext::oneapi::bfloat16 *const out[2],
                                const PairSplit sp[2], uint32_t seq, const PairSplitAdd *add = nullptr, sycl::queue *const side[2] = nullptr);
// The same in two halves, for a producer that finishes the peer's rows first and overlaps
// the copy with the rest of its work: pair_split_send on each card once its peer-owned rows of part are
// on q (returns the copy's event), then pair_split_finish for both. Rows: pair_split_owner_rows.
void pair_split_owner_rows(int card, int m, int *lo, int *hi, int cards = 2);  // [m c / cards, m (c + 1) / cards)
// A producer that sends (gdn_rows, qsa_rows out projection; moe_rows combine): it finishes the peer's
// rows of its partial first, calls pair_split_send, then its own rows while the copy runs; the caller
// then calls pair_split_finish.
struct PairSplitSend {
  sycl::queue *side = nullptr;  // the card's second queue (null: the copy on q, no overlap)
  const PairSplit *sp = nullptr;
  uint32_t seq = 0;
};
sycl::event pair_split_send(sycl::queue &q, sycl::queue *side, int card, int m, int n, const float *part, const PairSplit &sp, uint32_t seq);
// Four cards: the order a card sends its copies (and its rows back) in; the sum's bits do not depend on it.
inline int pair_split_send_order(int card, int k, int cards) { return (card + k) % cards; }  // k = 1 .. cards - 1
// What pair_split_finish leaves in `out` on each card.
enum class SplitRows : uint8_t {
  all,       // every row: the owners copy their rows to the other cards, and each card waits for them
  deferred,  // every row, later: the copies are sent, the caller waits (pair_split_wait) before it reads another owner's rows
  owner,     // the card's own rows only (pair_split_owner_rows): nothing is copied back
};
PairSplitEvents pair_split_finish(sycl::queue *const q[2], sycl::queue *const side[2], int m, int n, const float *const part[2],
                                  sycl::ext::oneapi::bfloat16 *const out[2], const PairSplit sp[2], uint32_t seq, const PairSplitAdd *add = nullptr,
                                  SplitRows rows = SplitRows::all);
// One card's four-card sum with SplitRows::owner, resolved for a consumer kernel that runs the wait for the other
// cards' partials and the sum of its own rows itself (hc_sum_combine_norm), instead of pair_split_finish's two launches
// (a wait kernel, then the sum kernel). The consumer writes bf16((p0 + p1) + (p2 + p3)) (+ the shared term) into `out`
// exactly as pair_split_finish does, so the sum's other readers find the same rows.
struct QuadOwnerSum {
  const float *own = nullptr;                           // this card's partial at its first own row (row stride n)
  const float *slot[kWireCardsMax] = {};                // each source card's recv slot at the same row (bf16 wire: bf16)
  const uint32_t *recv_flags = nullptr;                 // this card's recv flag lines (one per source card)
  const float *add_p = nullptr, *add_scale = nullptr;  // the shared term at the first own row, or null
  sycl::ext::oneapi::bfloat16 *out = nullptr;          // the sum at the first own row
  int card = 0, rows = 0, n = 0;                        // rows: the card's own rows of the m-row sum
  uint32_t seq = 0;
  bool bf16 = false;
  uint64_t wait_cycles = 0;
  uint32_t *error = nullptr;
  uint64_t *wait_span = nullptr;  // as PairSplit::wait_span: [0, 1] = the wait for the partials
};
// The QuadOwnerSum of card sp.card for the m-row sum pair_split_finish(..., SplitRows::owner) would finish with the
// same arguments; refuses what that call refuses, and recv flag lines that do not start a 64-byte line.
QuadOwnerSum pair_split_owner_sum(int m, int n, const float *part, sycl::ext::oneapi::bfloat16 *out, const PairSplit &sp, uint32_t seq,
                                  const PairSplitAdd *add = nullptr);
// With SplitRows::deferred, pair_split_finish leaves out's peer rows pending (done[] unset): a
// card may run its consumer on its own rows (pair_split_owner_rows) first, then must call this on
// its queue before anything reads the peer's rows. Its event is that card's done.
sycl::event pair_split_wait(sycl::queue &q, const PairSplit &sp, uint32_t seq);
// Rows shared by their owner: a card copies its own rows (pair_split_owner_rows) of each buffer set into the same rows
// of every other card's buffer, then raises its out flag line on that card (one flag per destination, after the last
// set). A card calls pair_split_wait(seq) on its queue before it reads another owner's rows. Use: work that is the same
// on every card (the residual mixers) runs each row on one card only, and its result goes to the others.
// side: null, or the card's second queue: the copies then run on the copy engine, beside q's next kernels, after
// `after` (the event of q's last kernel that writes the shared rows; null = after everything on q so far). The sends
// rotate as pair_split_send's. seq: as pair_split_finish's (never used before on these flags).
struct SplitShare {
  void *const *bufs = nullptr;  // every card's buffer, by card: [m] rows of row_bytes each
  size_t row_bytes = 0;
};
// tail > 0: only the last `tail` of the card's own rows go (the rows a neighbour's window reads; every row still flags).
sycl::event pair_split_share(sycl::queue &q, sycl::queue *side, int card, int m, std::span<const SplitShare> sets, const PairSplit &sp,
                             uint32_t seq, const sycl::event *after = nullptr, int tail = 0);
// Diagnostic: one work-item writes the card's GT-clock cycle counter (ESIMD
// rdtsc; it runs at the GT frequency, so convert with a calibration over a known span) into *at, in
// order on q. No event or barrier is needed to read it.
sycl::event stamp(sycl::queue &q, uint64_t *at);

}  // namespace xenithium::kernels
