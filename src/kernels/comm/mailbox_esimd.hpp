// Xenithium · device side of the two-card mailbox (private to src/kernels).
//
// Every 32-bit word of a slot is its own flag: `empty` until the peer's (uncached) store lands. A
// receiver spins on its own inbox until no word is empty (time-bounded: on timeout it sets *error
// and returns NaN words), then empties the slot for the next call on this parity.
//
// The many-row sums use it as a reduce-scatter + all-gather. Per (row, 64-column chunk)
// one card owns the chunk (mail_owner): the other card sends its fp32 partial, the owner adds it to
// its own, finishes the row piece and sends the bf16 result back (32 words of bf16 pairs). fp32 addition
// commutes, so the result does not depend on which card owns a chunk; each card sends 64 + 32 words per 2 chunks.
#pragma once

#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/comm/wire.hpp"  // WireQuad

namespace xenithium::kernels::mail {

namespace esimd = sycl::ext::intel::esimd;

inline constexpr uint32_t kNanWord = 0x7fc00000u;      // fp32 quiet NaN: never the empty pattern
inline constexpr uint32_t kNanPairWord = 0x7fc07fc0u;  // two bf16 quiet NaNs

inline constexpr auto kUc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                              esimd::alignment<128>};

// The card that finishes chunk `chunk` (both cards agree; alternate chunks balance the work).
inline bool mail_owner(int chunk, int card) SYCL_ESIMD_FUNCTION { return (chunk & 1) == (card & 1); }

template <int N>
inline void mail_send(uint32_t *peer_slot, esimd::simd<uint32_t, N> words, uint32_t empty, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  words.merge(esimd::simd<uint32_t, N>(nan_word), words == empty);
  esimd::block_store<uint32_t, N>(peer_slot, words, kUc);
}

// How many times a message is re-read before a slot that never becomes every-word-confirmed is a hard error.
inline constexpr int kTakeConfirmTries = 64;

// Take one message. "No word is the empty pattern" is not enough on this host: a peer store can be taken as partly
// visible -- some 64-byte lane runs reading as zeroes while the rest of the message is there -- and a caller that trusts
// the first all-non-empty read sums a partial with a zero and never sees an error (probe wiremin; the fp32 pair_reduce
// cases of the wire gate, 231 -> 71 of 303). So every word is acknowledged: the message is read again and the two reads
// must agree word for word, up to kTakeConfirmTries times. A slot that never settles is error 2, a slot that never
// fills is error 1. The clear is the take's last act, so this parity can be reused.
template <int N>
inline esimd::simd<uint32_t, N> mail_take(uint32_t *slot, uint32_t empty, uint64_t wait_cycles, uint32_t *error, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  const uint64_t t0 = esimd::rdtsc();
  esimd::simd<uint32_t, N> got = esimd::block_load<uint32_t, N>(slot, kUc);
  while ((got == esimd::simd<uint32_t, N>(empty)).any()) {
    if (esimd::rdtsc() - t0 > wait_cycles) {
      error[0] = 1u;
      esimd::block_store<uint32_t, N>(slot, esimd::simd<uint32_t, N>(empty), kUc);
      return esimd::simd<uint32_t, N>(nan_word);
    }
    got = esimd::block_load<uint32_t, N>(slot, kUc);
  }
  for (int t = 0; t < kTakeConfirmTries; ++t) {
    // System scope is what the acknowledgement needs: a fence scope sweep on the wire gate measured none 238, group 138,
    // local 142, system 71 of 303 failing. The fence is also why the acknowledgement loop must stay inside its spin:
    // per message it costs the prefill rows wire time it does not have.
    esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
    const esimd::simd<uint32_t, N> again = esimd::block_load<uint32_t, N>(slot, kUc);
    if (!(again != got).any()) {
      esimd::block_store<uint32_t, N>(slot, esimd::simd<uint32_t, N>(empty), kUc);
      return got;
    }
    got = again;
  }
  error[0] = 2u;
  esimd::block_store<uint32_t, N>(slot, esimd::simd<uint32_t, N>(empty), kUc);
  return got;
}

// bf16 values packed two per word (element 2i in the low half), and back.
template <int N>
inline esimd::simd<uint32_t, N / 2> pack_bf16(esimd::simd<sycl::ext::oneapi::bfloat16, N> v) SYCL_ESIMD_FUNCTION {
  return v.template bit_cast_view<uint32_t>().read();
}
template <int N>
inline esimd::simd<sycl::ext::oneapi::bfloat16, N> unpack_bf16(esimd::simd<uint32_t, N / 2> w) SYCL_ESIMD_FUNCTION {
  return w.template bit_cast_view<sycl::ext::oneapi::bfloat16>().read();
}

// N words of a slot, sent / taken in messages of at most 128 words (the LSC block limit).
inline constexpr int kMailMsgWords = 128;
template <int N>
inline void mail_send_words(uint32_t *peer_slot, esimd::simd<uint32_t, N> words, uint32_t empty, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  constexpr int P = N < kMailMsgWords ? N : kMailMsgWords;
#pragma unroll
  for (int p = 0; p < N / P; ++p) mail_send<P>(peer_slot + p * P, esimd::simd<uint32_t, P>(words.template select<P, 1>(p * P)), empty, nan_word);
}
template <int N>
inline esimd::simd<uint32_t, N> mail_take_words(uint32_t *slot, uint32_t empty, uint64_t wait_cycles, uint32_t *error, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  constexpr int P = N < kMailMsgWords ? N : kMailMsgWords;
  esimd::simd<uint32_t, N> got;
#pragma unroll
  for (int p = 0; p < N / P; ++p) got.template select<P, 1>(p * P) = mail_take<P>(slot + p * P, empty, wait_cycles, error, nan_word);
  return got;
}

// One hop of the four-card sum (TP4; comm/wire.hpp, WireQuad): this card's fp32 values to the peer's
// slot, the peer's from this card's slot, and their sum. fp32 addition commutes, so both cards of the hop hold the same
// bits; two hops (partner card ^ 1, then the other pair card ^ 2) give (p0 + p1) + (p2 + p3) on every card.
template <int N>
inline esimd::simd<float, N> quad_hop(esimd::simd<float, N> mine, uint32_t *peer_slot, uint32_t *slot, uint32_t empty, uint64_t wait_cycles,
                                      uint32_t *error) SYCL_ESIMD_FUNCTION {
  mail_send_words<N>(peer_slot, mine.template bit_cast_view<uint32_t>().read(), empty, kNanWord);
  esimd::simd<uint32_t, N> got = mail_take_words<N>(slot, empty, wait_cycles, error, kNanWord);  // (bit_cast_view needs a non-const lvalue)
  return mine + esimd::simd<float, N>(got.template bit_cast_view<float>().read());
}

// The three inboxes one card of a four-card wire touches, resolved on the host (a kernel indexes no array by its card).
struct QuadPeers {
  uint32_t *own = nullptr, *partner = nullptr, *other = nullptr;  // card, card ^ 1, card ^ 2
};
inline QuadPeers quad_peers(const WireQuad &qd) {
  return QuadPeers{qd.inbox[static_cast<size_t>(qd.card)], qd.inbox[static_cast<size_t>(qd.card ^ 1)], qd.inbox[static_cast<size_t>(qd.card ^ 2)]};
}
// Hop 2 of the fp32 four-card sum: the pair's sum `pair` with the other pair (slot region at + hop_words).
template <int N>
inline esimd::simd<float, N> quad_hop2(esimd::simd<float, N> pair, QuadPeers p, size_t at, size_t hop_words, uint32_t empty, uint64_t wait_cycles,
                                       uint32_t *error) SYCL_ESIMD_FUNCTION {
  return quad_hop<N>(pair, p.other + at + hop_words, p.own + at + hop_words, empty, wait_cycles, error);
}
// The whole fp32 four-card sum of this card's values v: (p0 + p1) + (p2 + p3).
template <int N>
inline esimd::simd<float, N> quad_sum(esimd::simd<float, N> v, QuadPeers p, size_t at, size_t hop_words, uint32_t empty, uint64_t wait_cycles,
                                      uint32_t *error) SYCL_ESIMD_FUNCTION {
  const esimd::simd<float, N> pair = quad_hop<N>(v, p.partner + at, p.own + at, empty, wait_cycles, error);
  return quad_hop2<N>(pair, p, at, hop_words, empty, wait_cycles, error);
}

// v unchanged, but opaque to the compiler's floating-point rewrites: an XOR of its bits with zero,
// a runtime zero the caller passes in as a kernel argument. The build's fast floating point may reassociate a
// four-operand sum and fold a later add into it; with this the four-card sums keep the mailbox's
// (p0 + p1) + (p2 + p3), then + the shared term, exactly, in every kernel that writes them.
template <int N>
inline esimd::simd<float, N> opaque(esimd::simd<float, N> v, uint32_t zero) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, N> b = v.template bit_cast_view<uint32_t>();
  b ^= zero;
  return b.template bit_cast_view<float>();
}

// ---- a decode step's rows by owner (comm/wire.hpp) --------------------------------------------------------------------
// Every card's inbox of one use and this card's index, resolved on the host.
struct OwnerBoxes {
  uint32_t *of[kWireCardsMax] = {};
  int card = 0;
};
inline OwnerBoxes owner_boxes(const OwnerMail &m) {
  OwnerBoxes b;
  for (int c = 0; c < kWireCardsMax; ++c) b.of[c] = m.inbox[static_cast<size_t>(c)];
  b.card = m.card;
  return b;
}
// The owner sum's slot of (source card, the owner's row `own_row`) in the owner's inbox; n = the sum's columns.
inline size_t owner_sum_at(int parity, int src, int own_row, int n) SYCL_ESIMD_FUNCTION {
  return ((static_cast<size_t>(parity) * kWireCardsMax + src) * kOwnerRowsMax + own_row) * static_cast<size_t>(n);
}
// One item of an owner sum: this card's fp32 partial `acc` of row r, columns [c0, c0 + N) of n. A card that does not
// own the row sends it to the owner and returns false. The owner takes the three other cards' partials, leaves
// (p0 + p1) + (p2 + p3) in acc - the four-card mailbox's order, operands by card index, each pair's sum opaque so the
// build's fast floating point cannot reassociate the four - and returns true. zero: a runtime 0 (see opaque).
template <int N>
inline bool owner_sum(esimd::simd<float, N> &acc, const OwnerBoxes &b, int r, int rows, int n, int c0, int parity, uint32_t empty, uint64_t wait_cycles,
                      uint32_t *error, uint32_t zero) SYCL_ESIMD_FUNCTION {
  const int owner = row_owner(r, rows), own_row = r - rows * owner / kWireCardsMax;
  if (owner != b.card) {
    mail_send_words<N>(b.of[owner] + owner_sum_at(parity, b.card, own_row, n) + c0, acc.template bit_cast_view<uint32_t>().read(), empty, kNanWord);
    return false;
  }
  esimd::simd<float, N> p[kWireCardsMax];
#pragma unroll
  for (int s = 0; s < kWireCardsMax; ++s) {
    if (s == b.card) {
      p[s] = acc;
    } else {
      esimd::simd<uint32_t, N> got = mail_take_words<N>(b.of[b.card] + owner_sum_at(parity, s, own_row, n) + c0, empty, wait_cycles, error, kNanWord);
      p[s] = got.template bit_cast_view<float>().read();
    }
  }
  acc = opaque<N>(p[0] + p[1], zero) + opaque<N>(p[2] + p[3], zero);
  return true;
}

}  // namespace xenithium::kernels::mail
