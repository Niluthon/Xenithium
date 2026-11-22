// Xenithium · two-card sum of fp32 partials (see wire.hpp).
#include "xenithium/kernels/comm/wire.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <sycl/ext/intel/esimd.hpp>

#include "comm/mailbox_esimd.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kM = kPairRows;
constexpr int kCols = 64;  // columns per work-item

template <bool WIRE, bool B16, int COLS, bool QUAD>
class PairReduceK;

// An item's slot words go to the peer in messages of at most 128 words (the LSC
// block limit) and come back the same way. The words are their own flags (mail_send / mail_take), so a wider item is
// the same protocol on a longer contiguous run of the slot.
template <bool B16, int COLS>
inline esimd::simd<float, COLS> exchange(esimd::simd<float, COLS> acc, uint32_t *peer_slot, uint32_t *slot, uint64_t wait_cycles, uint32_t *error) SYCL_ESIMD_FUNCTION {
  constexpr int W = B16 ? COLS / 2 : COLS, P = W < 128 ? W : 128;
  const uint32_t nan_word = B16 ? mail::kNanPairWord : mail::kNanWord;
  esimd::simd<float, COLS> mine_f = acc;
  esimd::simd<uint32_t, W> mine;
  if constexpr (B16) {
    const esimd::simd<bf16, COLS> mb = acc;
    mine_f = mb;
    mine = mail::pack_bf16<COLS>(mb);
  } else {
    mine = acc.template bit_cast_view<uint32_t>();
  }
#pragma unroll
  for (int p = 0; p < W / P; ++p) mail::mail_send<P>(peer_slot + p * P, esimd::simd<uint32_t, P>(mine.template select<P, 1>(p * P)), kPairMailEmpty, nan_word);
  esimd::simd<uint32_t, W> got;
#pragma unroll
  for (int p = 0; p < W / P; ++p) got.template select<P, 1>(p * P) = mail::mail_take<P>(slot + p * P, kPairMailEmpty, wait_cycles, error, nan_word);
  if constexpr (B16) return mine_f + esimd::simd<float, COLS>(mail::unpack_bf16<COLS>(got));
  else return acc + got.template bit_cast_view<float>();
}

// The four-card exchange (TP4, WireQuad). Hop 1 at `at` with the partner, hop 2 at at + hop_words
// with the other pair. B16: hop 1 sends bf16(own) (the bf16 wire's operand), hop 2 the pair's fp32 sum.
template <bool B16, int COLS>
inline esimd::simd<float, COLS> exchange_quad(esimd::simd<float, COLS> acc, mail::QuadPeers p, size_t at, size_t hop_words, uint64_t wait_cycles,
                                              uint32_t *error) SYCL_ESIMD_FUNCTION {
  if constexpr (B16) {
    const esimd::simd<bf16, COLS> mb = acc;
    mail::mail_send_words<COLS / 2>(p.partner + at, mail::pack_bf16<COLS>(mb), kPairMailEmpty, mail::kNanPairWord);
    const esimd::simd<uint32_t, COLS / 2> got = mail::mail_take_words<COLS / 2>(p.own + at, kPairMailEmpty, wait_cycles, error, mail::kNanPairWord);
    const esimd::simd<float, COLS> pair = esimd::simd<float, COLS>(mb) + esimd::simd<float, COLS>(mail::unpack_bf16<COLS>(got));
    return mail::quad_hop2<COLS>(pair, p, at, hop_words, kPairMailEmpty, wait_cycles, error);
  } else {
    return mail::quad_sum<COLS>(acc, p, at, hop_words, kPairMailEmpty, wait_cycles, error);
  }
}

// One item per work-group (the peer stores want every core); B16 = the bf16
// wire: each card sends bf16(own) and both sum bf16(own) + bf16(peer) in fp32 (an all-reduce of bf16
// partials; half the bytes).
// COLS columns per item by the slot bytes an item sends (pair_cols): 64 columns at 1-2
// rows, 512 B at 3-5, 1 KiB from 6. Every element's sum is the same whatever COLS.
template <bool WIRE, bool B16, int COLS, bool QUAD = false>
sycl::event reduce(sycl::queue &q, int rows, int n, const float *part, PairWire wire, int parity, bf16 *out) {
  const int chunks = n / COLS;
  const mail::QuadPeers qp = QUAD ? mail::quad_peers(wire.quad) : mail::QuadPeers{};
  const size_t hop_words = 2ull * kM * static_cast<size_t>(n);  // (quad) hop 2's slot region after hop 1's
  return q.parallel_for<PairReduceK<WIRE, B16, COLS, QUAD>>(
      sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks)}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        // a (row, chunk) grid: the ids come from the launch, not from a division by the run-time chunk count in every item
        const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * COLS;
        esimd::simd<float, COLS> acc = esimd::block_load<float, COLS>(part + static_cast<size_t>(r) * n + c0);
        if constexpr (QUAD) {
          const size_t at = (static_cast<size_t>(parity) * kM + r) * n + c0;
          acc = exchange_quad<B16, COLS>(acc, qp, at, hop_words, wire.wait_cycles, wire.error);
        } else if constexpr (WIRE) {
          const size_t at = (static_cast<size_t>(parity) * kM + r) * n + c0;  // bf16: the fp32 layout's slot, its first half used
          acc = exchange<B16, COLS>(acc, wire.peer_inbox + at, wire.inbox + at, wire.wait_cycles, wire.error);
        }
        const esimd::simd<bf16, COLS> o = acc;
        o.copy_to(out + static_cast<size_t>(r) * n + c0);
      });
}

// The int8 wire (PairWire::i8). Per kI8Group columns a card rounds its partial to 8-bit codes
// c = rne(p / s) in [-127, 127] with s = bf16(absmax / 127), sends the codes (as c + 127: no word is the empty pattern)
// and the scales (one fp32 word each), and both cards sum c x s (own) + c x s (peer) in fp32. A code times a bf16 scale
// is exact in fp32 (7 + 8 mantissa bits), so both cards add the same two values: the same bits on both. An item's slot
// (COLS words of the fp32 layout) holds its COLS / 4 code words, then its COLS / kI8Group scale words: 72 bytes per 64
// columns against the bf16 wire's 128. COLS by the rows (pair_cols_i8).
constexpr int kI8Group = 32;
template <int COLS>
class PairReduceI8K;
template <int COLS>
sycl::event reduce_i8(sycl::queue &q, int rows, int n, const float *part, PairWire wire, int parity, bf16 *out) {
  constexpr int G = kI8Group, NG = COLS / G, CW = COLS / 4, P = CW < 128 ? CW : 128;
  const int chunks = n / COLS;
  return q.parallel_for<PairReduceI8K<COLS>>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks)}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * COLS;  // a (row, chunk) grid
    esimd::simd<float, COLS> acc = esimd::block_load<float, COLS>(part + static_cast<size_t>(r) * n + c0);
    const size_t at = (static_cast<size_t>(parity) * kM + r) * n + c0;
    esimd::simd<uint32_t, CW> mine;
    esimd::simd<float, NG> scales;
    esimd::simd<float, COLS> own;
#pragma unroll
    for (int b = 0; b < NG; ++b) {
      const esimd::simd<float, G> v = acc.template select<G, 1>(b * G);
      const esimd::simd<float, 8> s8(esimd::hmax<float>(esimd::abs(v)) * (1.f / 127.f));
      const esimd::simd<bf16, 8> sb = s8;
      const esimd::simd<float, 8> sf = sb;
      const float sc = sf[0] > 0.f ? static_cast<float>(sf[0]) : 1.f;  // (an all-zero group: codes 0)
      esimd::simd<float, G> c = esimd::rnde<float, G>(v * (1.f / sc));
      c = esimd::min<float, G>(esimd::max<float, G>(c, esimd::simd<float, G>(-127.f)), esimd::simd<float, G>(127.f));
      own.template select<G, 1>(b * G) = c * sc;
      esimd::simd<uint8_t, G> cb = esimd::convert<uint8_t>(esimd::convert<int>(c) + 127);
      mine.template select<G / 4, 1>(b * (G / 4)) = cb.template bit_cast_view<uint32_t>().read();
      scales[b] = sc;
    }
    esimd::simd<uint32_t, NG> sw = scales.template bit_cast_view<uint32_t>().read();
    uint32_t *peer_slot = wire.peer_inbox + at, *slot = wire.inbox + at;
#pragma unroll
    for (int p = 0; p < CW / P; ++p) mail::mail_send<P>(peer_slot + p * P, esimd::simd<uint32_t, P>(mine.template select<P, 1>(p * P)), kPairMailEmpty, mail::kNanWord);
    mail::mail_send<NG>(peer_slot + CW, sw, kPairMailEmpty, mail::kNanWord);
    esimd::simd<uint32_t, CW> got;
#pragma unroll
    for (int p = 0; p < CW / P; ++p) got.template select<P, 1>(p * P) = mail::mail_take<P>(slot + p * P, kPairMailEmpty, wire.wait_cycles, wire.error, mail::kNanWord);
    esimd::simd<uint32_t, NG> gsw = mail::mail_take<NG>(slot + CW, kPairMailEmpty, wire.wait_cycles, wire.error, mail::kNanWord);
    const esimd::simd<float, NG> gs = gsw.template bit_cast_view<float>().read();
#pragma unroll
    for (int b = 0; b < NG; ++b) {
      esimd::simd<uint32_t, G / 4> cw = got.template select<G / 4, 1>(b * (G / 4));
      const esimd::simd<uint8_t, G> cb = cw.template bit_cast_view<uint8_t>().read();
      const esimd::simd<float, G> c = esimd::convert<float>(esimd::convert<int>(cb) - 127);
      const float sc = gs[b];
      acc.template select<G, 1>(b * G) = esimd::simd<float, G>(own.template select<G, 1>(b * G)) + c * sc;
    }
    const esimd::simd<bf16, COLS> o = acc;
    o.copy_to(out + static_cast<size_t>(r) * n + c0);
  });
}

template <bool WIRE, bool B16, int COLS, bool QUAD>
class PairRowsK;

// B16 (PairRowsWire::bf16): each card sends bf16(own) packed two per word (half the mailbox words) and both
// sum bf16(own) + bf16(peer) in fp32, as the decode's bf16 wire.
// COLS by the slot bytes per item (pair_cols), one item per work-group.
template <bool WIRE, bool B16, int COLS, bool QUAD = false>
sycl::event reduce_rows(sycl::queue &q, int m, int n, const float *part, PairRowsWire wire, int parity, bf16 *out) {
  const int chunks = n / COLS;
  const size_t wg = COLS == kCols && chunks % 8 == 0 ? 8 : 1;
  const mail::QuadPeers qp = QUAD ? mail::quad_peers(wire.quad) : mail::QuadPeers{};
  const size_t hop_words = 2ull * wire.rows_max * static_cast<size_t>(wire.n_max);  // (quad) hop 2's slot region
  return q.parallel_for<PairRowsK<WIRE, B16, COLS, QUAD>>(
      sycl::nd_range<2>({static_cast<size_t>(m), static_cast<size_t>(chunks)}, {1, wg}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * COLS;  // a (row, chunk) grid
        esimd::simd<float, COLS> acc = esimd::block_load<float, COLS>(part + static_cast<size_t>(r) * n + c0);
        if constexpr (QUAD) {
          const size_t at = (static_cast<size_t>(parity) * wire.rows_max + r) * wire.n_max + c0;
          acc = exchange_quad<B16, COLS>(acc, qp, at, hop_words, wire.wait_cycles, wire.error);
        } else if constexpr (WIRE) {
          const size_t at = (static_cast<size_t>(parity) * wire.rows_max + r) * wire.n_max + c0;  // bf16: the fp32 slot, its first half used
          acc = exchange<B16, COLS>(acc, wire.peer_inbox + at, wire.inbox + at, wire.wait_cycles, wire.error);
        }
        const esimd::simd<bf16, COLS> o = acc;
        o.copy_to(out + static_cast<size_t>(r) * n + c0);
      });
}

// ---- split ----
class PairFlagK;
class PairWaitK;
template <bool ADD, bool B16, int W>
class PairSumK;
template <bool ADD, bool B16, int W>
class QuadSumK;  // the four-card owner sum
class QuadWaitK;  // the wait for the other cards' flag lines
constexpr int kSumCols = 256;  // the split sum's columns per item
// the peer-written split buffers: past L1 (incoherent per core), through L2 like every other peer-copied buffer
constexpr auto kPeerLd = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::cached>,
                                           esimd::alignment<128>};
class PairSendBf16K;
class QuadSendBf16K;

// One word of the peer's flag line = seq, after this queue's earlier copies (in order).
sycl::event flag_store(sycl::queue &q, uint32_t *peer_line, uint32_t seq) {
  return q.parallel_for<PairFlagK>(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
    esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
    esimd::block_store<uint32_t, 4>(peer_line, esimd::simd<uint32_t, 4>(seq), mail::kUc);
  });
}
// One work-item spins until this card's flag line holds seq (bounded: error + give up), so the
// kernels after it on the in-order queue see the peer's copy (one item polls the line, not every
// item of the sum kernel). span (diagnostic): null, or [entry, exit] rdtsc of this wait.
sycl::event flag_wait(sycl::queue &q, const uint32_t *line, uint32_t seq, uint64_t wait_cycles, uint32_t *error, uint64_t *span = nullptr) {
  return q.parallel_for<PairWaitK>(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
    esimd::simd<uint32_t, 4> v = esimd::block_load<uint32_t, 4>(line, mail::kUc);
    const uint64_t t0 = esimd::rdtsc();
    while (v[0] != seq) {
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
      v = esimd::block_load<uint32_t, 4>(line, mail::kUc);
      if (v[0] != seq && esimd::rdtsc() - t0 > wait_cycles) {
        error[0] = 1u;
        break;
      }
    }
    if (span) {
      esimd::simd<uint64_t, 2> s;
      s[0] = t0;
      s[1] = esimd::rdtsc();
      s.copy_to(span);
    }
  });
}

// Four cards: spin until every other card's flag line (base + 4 s, s != card) holds seq, one line after the
// other (bounded: error + give up); span: [entry, exit] of the whole wait.
sycl::event flag_wait_others(sycl::queue &q, const uint32_t *base, int card, int cards, uint32_t seq, uint64_t wait_cycles, uint32_t *error,
                             uint64_t *span = nullptr) {
  return q.parallel_for<QuadWaitK>(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
    const uint64_t t0 = esimd::rdtsc();
    bool timed_out = false;
    for (int s = 0; s < cards; ++s) {
      if (s == card) continue;
      const uint32_t *line = base + 4 * s;
      esimd::simd<uint32_t, 4> v = esimd::block_load<uint32_t, 4>(line, mail::kUc);
      while (v[0] != seq && !timed_out) {
        esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
        v = esimd::block_load<uint32_t, 4>(line, mail::kUc);
        if (v[0] != seq && esimd::rdtsc() - t0 > wait_cycles) {
          error[0] = 1u;
          timed_out = true;
        }
      }
    }
    if (span) {
      esimd::simd<uint64_t, 2> sp;
      sp[0] = t0;
      sp[1] = esimd::rdtsc();
      sp.copy_to(span);
    }
  });
}

class StampK;
}  // namespace

sycl::event stamp(sycl::queue &q, uint64_t *at) {
  return q.parallel_for<StampK>(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
    esimd::simd<uint64_t, 1> v(esimd::rdtsc());
    v.copy_to(at);
  });
}

namespace {
}  // namespace

size_t pair_split_recv_elems(int rows_max, int n_max, int cards) {
  if (cards <= 2) return static_cast<size_t>((rows_max + 1) / 2) * n_max;
  return static_cast<size_t>(cards) * ((rows_max + cards - 1) / cards) * n_max;  // a slot per source card
}
size_t pair_split_stage_elems(int rows_max, int n_max, int cards) {
  return cards <= 2 ? pair_split_recv_elems(rows_max, n_max) : static_cast<size_t>(rows_max) * n_max;  // (four: rows in place)
}

class PairSeqTabK;
sycl::event pair_split_seq_tab_init(sycl::queue &q, uint32_t *tab) {
  return q.parallel_for<PairSeqTabK>(sycl::range<1>(kPairSplitSeqs * 4), [=](sycl::id<1> i) { tab[i[0]] = static_cast<uint32_t>(i[0] / 4); });
}

void pair_split_owner_rows(int card, int m, int *lo, int *hi, int cards) {
  if (cards <= 2) {
    *lo = card == 0 ? 0 : m / 2;
    *hi = card == 0 ? m / 2 : m;
    return;
  }
  *lo = m * card / cards;
  *hi = m * (card + 1) / cards;
}

namespace {
void check_split(int m, int n, const PairSplit &sp) {
  if (m < 1 || n % kCols) throw std::invalid_argument("pair_split_rows: shape");
  if (sp.cards == kWireCardsMax) {
    bool named = sp.card >= 0 && sp.card < kWireCardsMax;
    for (int c = 0; c < kWireCardsMax; ++c) named = named && sp.recv_of[c] && sp.flags_of[c];
    if (!named || !sp.recv || !sp.flags || !sp.error || n > sp.n_max || m > sp.rows_max || (sp.bf16 && (!sp.stage || n % kSumCols)))
      throw std::invalid_argument("pair_split_rows: four-card split buffers (every card's recv and flags, this card's index)");
    return;
  }
  if (sp.cards != 2) throw std::invalid_argument("pair_split_rows: two or four cards");
  if (!sp.recv || !sp.flags || !sp.peer_flags || !sp.peer_recv || !sp.error || n > sp.n_max || m > sp.rows_max || (sp.bf16 && (!sp.stage || n % kSumCols)))
    throw std::invalid_argument("pair_split_rows: split buffers");
}
// rows [lo, hi) of part as bf16 into the same rows of st (the bf16 wire's stage; n % 256 == 0)
sycl::event to_bf16(sycl::queue &q, const float *part, bf16 *st, int lo, int hi, int n) {
  const size_t items = static_cast<size_t>(hi - lo) * n / (4 * kCols);
  const float *src = part + static_cast<size_t>(lo) * n;
  bf16 *dst = st + static_cast<size_t>(lo) * n;
  return q.parallel_for<QuadSendBf16K>(sycl::nd_range<1>(items, items % 16 == 0 ? 16 : 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const size_t base = it.get_global_id(0) * (4 * kCols);
    esimd::simd<float, kCols> v[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) v[p] = esimd::block_load<float, kCols>(src + base + p * kCols);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const esimd::simd<bf16, kCols> w = v[p];
      w.copy_to(dst + base + p * kCols);
    }
  });
}
// a source card's slot in a four-card recv
size_t quad_slot(const PairSplit &sp, int src) { return static_cast<size_t>(src) * ((sp.rows_max + sp.cards - 1) / sp.cards) * sp.n_max; }
// The flag after a copy: on the side queue a 16-byte copy of the table's line (the copy engine, in
// order after the data's copy); on the compute queue the store kernel.
// after (side only): an event of another queue the copy follows.
sycl::event flag_after(sycl::queue &q, uint32_t *peer_line, uint32_t seq, bool side, const uint32_t *tab, const sycl::event *after = nullptr) {
  if (!side) return flag_store(q, peer_line, seq);
  return after ? q.memcpy(peer_line, tab + static_cast<size_t>(seq) * 4, 16, *after) : q.memcpy(peer_line, tab + static_cast<size_t>(seq) * 4, 16);
}

}  // namespace

sycl::event pair_split_send(sycl::queue &q, sycl::queue *side, int card, int m, int n, const float *part, const PairSplit &sp, uint32_t seq) {
  check_split(m, n, sp);
  if (seq == 0 || seq >= kPairSplitSeqs || (side && !sp.seq_tab)) throw std::invalid_argument("pair_split_send: seq / its table");
  if (sp.cards == kWireCardsMax) {  // every other card's rows into its recv slot `card`, then its flag line
    if (card != sp.card) throw std::invalid_argument("pair_split_send: the card is not the split's");
    int lo = 0, hi = 0;
    pair_split_owner_rows(card, m, &lo, &hi, sp.cards);
    sycl::queue &cq = side ? *side : q;
    if (sp.bf16) {  // the other cards' rows as bf16 into the stage (same rows)
      if (lo > 0) to_bf16(q, part, sp.stage, 0, lo, n);
      if (hi < m) to_bf16(q, part, sp.stage, hi, m, n);
    }
    // The copies follow everything on q so far through the first copy's own wait list, not through a
    // barrier on the side queue: behind such a barrier q's next kernel does not start until the side queue has finished
    // its earlier copies, whatever they are, so a send submitted while the previous lane's copies are on the link
    // would stall the compute queue for their rest.
    sycl::event after;
    bool after_open = side != nullptr;
    if (side) after = q.ext_oneapi_submit_barrier();
    sycl::event e;
    // card + 1, + 2, + 3 (pair_split_send_order): no card is every sender's last
    for (int k = 1; k < sp.cards; ++k) {
      const int d = pair_split_send_order(card, k, sp.cards);
      int r0 = 0, r1 = 0;
      pair_split_owner_rows(d, m, &r0, &r1, sp.cards);
      if (r1 > r0) {
        const void *src = sp.bf16 ? static_cast<const void *>(sp.stage + static_cast<size_t>(r0) * n) : static_cast<const void *>(part + static_cast<size_t>(r0) * n);
        void *dst = sp.bf16 ? static_cast<void *>(reinterpret_cast<bf16 *>(sp.recv_of[d]) + quad_slot(sp, card)) : static_cast<void *>(sp.recv_of[d] + quad_slot(sp, card));
        const size_t bytes = static_cast<size_t>(r1 - r0) * n * (sp.bf16 ? 2 : 4);
        e = after_open ? cq.memcpy(dst, src, bytes, after) : cq.memcpy(dst, src, bytes);
        after_open = false;
      }
      flag_after(cq, sp.flags_of[d] + kPairSplitRecvWord + 4 * card, seq, side != nullptr, sp.seq_tab, after_open ? &after : nullptr);
      after_open = false;
    }
    return e;
  }
  int lo = 0, hi = 0;
  pair_split_owner_rows(1 - card, m, &lo, &hi);  // the peer's rows
  sycl::queue &cq = side ? *side : q;
  sycl::event e;
  if (side && !(sp.bf16 && hi > lo)) side->ext_oneapi_submit_barrier({q.ext_oneapi_submit_barrier()});  // after everything on q so far
  if (sp.bf16 && hi > lo) {  // the peer's rows as bf16 into the stage first (on q, after their producer), then half the bytes
    const size_t items = static_cast<size_t>(hi - lo) * n / (4 * kCols);  // 256 columns per item (n % 256 == 0: check_split)
    const float *src = part + static_cast<size_t>(lo) * n;
    bf16 *st = sp.stage;
    const sycl::event cv = q.parallel_for<PairSendBf16K>(sycl::nd_range<1>(items, items % 16 == 0 ? 16 : 1), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const size_t base = it.get_global_id(0) * (4 * kCols);
      esimd::simd<float, kCols> v[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) v[p] = esimd::block_load<float, kCols>(src + base + p * kCols);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const esimd::simd<bf16, kCols> w = v[p];
        w.copy_to(st + base + p * kCols);
      }
    });
    if (side) side->ext_oneapi_submit_barrier({cv});
  }
  if (hi > lo) {
    if (sp.bf16)
      e = cq.memcpy(sp.peer_recv, sp.stage, static_cast<size_t>(hi - lo) * n * 2);
    else
      e = cq.memcpy(sp.peer_recv, part + static_cast<size_t>(lo) * n, static_cast<size_t>(hi - lo) * n * 4);
  }
  flag_after(cq, sp.peer_flags + kPairSplitRecvWord, seq, side != nullptr, sp.seq_tab);
  return e;
}

namespace {
// One operand of the four-card owner sum: this card's fp32 partial (bf16 wire: rounded to bf16), or a source card's slot
// (peer-copied: past L1; bf16 wire: packed bf16).
template <bool B16>
inline esimd::simd<float, kCols> quad_operand(bool own, const float *a, const float *slot, size_t off) SYCL_ESIMD_FUNCTION {
  if (own) {
    const esimd::simd<float, kCols> mine = esimd::block_load<float, kCols>(a + off);
    if constexpr (B16) {
      const esimd::simd<bf16, kCols> mb = mine;
      return esimd::simd<float, kCols>(mb);
    } else {
      return mine;
    }
  }
  if constexpr (B16) {
    const esimd::simd<uint32_t, kCols / 2> got =
        esimd::block_load<uint32_t, kCols / 2>(reinterpret_cast<const uint32_t *>(reinterpret_cast<const bf16 *>(slot) + off), kPeerLd);
    return esimd::simd<float, kCols>(mail::unpack_bf16<kCols>(got));
  } else {
    return esimd::block_load<float, kCols>(slot + off, kPeerLd);
  }
}

// pair_split_finish for four cards (PairSplit::cards 4).
PairSplitEvents quad_split_finish(sycl::queue *const q[], sycl::queue *const side[], int m, int n, const float *const part[], bf16 *const out[],
                                  const PairSplit sp[], uint32_t seq, const PairSplitAdd *add, SplitRows rows) {
  constexpr int C = kWireCardsMax;
  for (int c = 0; c < C; ++c) {
    check_split(m, n, sp[c]);
    if (sp[c].cards != C || sp[c].card != c) throw std::invalid_argument("pair_split_finish: four cards in order");
    if (seq == 0 || seq >= kPairSplitSeqs || (side && side[c] && !sp[c].seq_tab)) throw std::invalid_argument("pair_split_finish: seq / its table");
  }
  PairSplitEvents ev;
  for (int c = 0; c < C; ++c) {
    int r0 = 0, r1 = 0;
    pair_split_owner_rows(c, m, &r0, &r1, C);
    const int rn = r1 - r0;
    ev.recv[c] = flag_wait_others(*q[c], sp[c].flags + kPairSplitRecvWord, c, C, seq, sp[c].wait_cycles, sp[c].error, sp[c].wait_span);
    sycl::queue &cq = side && side[c] ? *side[c] : *q[c];
    if (rn > 0) {
      const float *a = part[c] + static_cast<size_t>(r0) * n;
      const float *b[C];
      for (int s2 = 0; s2 < C; ++s2)  // (bf16 wire: bf16 slots at the same element offsets, as the sender writes them)
        b[s2] = sp[c].bf16 ? reinterpret_cast<const float *>(reinterpret_cast<const bf16 *>(sp[c].recv) + quad_slot(sp[c], s2)) : sp[c].recv + quad_slot(sp[c], s2);
      const float *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
      bf16 *o = out[c] + static_cast<size_t>(r0) * n;
      const float *ap = add ? add[c].p + static_cast<size_t>(r0) * n : nullptr;
      const float *as = add ? add[c].scale + r0 : nullptr;
      const int own = c;
      const uint32_t opaque0 = static_cast<uint32_t>(m < 0);  // 0 (m >= 1): a kernel argument, unknown to the compiler
      // (p0 + p1) + (p2 + p3) per element, the four-card mailbox's order (bf16 wire: each p bf16, the sums fp32)
      auto body = [&](auto ADD, auto B16, auto WIDE) {
        constexpr bool kAdd = decltype(ADD)::value, kB16 = decltype(B16)::value;
        constexpr int kW = decltype(WIDE)::value ? kSumCols : kCols, kP = kW / kCols;
        const int wch = n / kW;
        const size_t items = static_cast<size_t>(rn) * wch;
        const size_t wg = items % 16 == 0 ? 16 : 1;
        ev.sum[c] = q[c]->parallel_for<QuadSumK<kAdd, kB16, kW>>(sycl::nd_range<1>(items, wg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const size_t id = it.get_global_id(0);
          const int r = static_cast<int>(id / wch);
          const size_t base = static_cast<size_t>(r) * n + (id % wch) * kW;
#pragma unroll
          for (int p = 0; p < kP; ++p) {
            const size_t off = base + p * kCols;
            const esimd::simd<float, kCols> x0 = quad_operand<kB16>(own == 0, a, b0, off), x1 = quad_operand<kB16>(own == 1, a, b1, off),
                                            x2 = quad_operand<kB16>(own == 2, a, b2, off), x3 = quad_operand<kB16>(own == 3, a, b3, off);
            // the mailbox's order exactly: the build's fast floating point reassociates four register operands (a clang fp
            // pragma does not stop it), so each pair's sum passes through an XOR with a runtime zero the compiler cannot
            // see through, then the two are added
            esimd::simd<float, kCols> acc = mail::opaque<kCols>(x0 + x1, opaque0) + mail::opaque<kCols>(x2 + x3, opaque0);
            if constexpr (kAdd) {  // bf16(opq(p x scale) + opq(sum)): moe_rows' four-card combine writes the same
              const esimd::simd<float, kCols> pv = esimd::block_load<float, kCols>(ap + off);
              acc = mail::opaque<kCols>(pv * as[r], opaque0) + mail::opaque<kCols>(acc, opaque0);
            }
            const esimd::simd<bf16, kCols> v = acc;
            v.copy_to(o + off);
          }
        });
      };
      auto run = [&](auto ADD, auto B16) {
        if (n % kSumCols == 0) body(ADD, B16, std::true_type{});
        else body(ADD, B16, std::false_type{});
      };
      if (add) {
        sp[c].bf16 ? run(std::true_type{}, std::true_type{}) : run(std::true_type{}, std::false_type{});
      } else {
        sp[c].bf16 ? run(std::false_type{}, std::true_type{}) : run(std::false_type{}, std::false_type{});
      }
      if (rows == SplitRows::owner) continue;  // the sum stays with its owner
      if (&cq != q[c]) cq.ext_oneapi_submit_barrier({ev.sum[c]});
      for (int k = 1; k < C; ++k) {  // (rotated as the sends)
        const int d = pair_split_send_order(c, k, C);
        ev.back[c] = cq.memcpy(out[d] + static_cast<size_t>(r0) * n, o, static_cast<size_t>(rn) * n * 2);
      }
    } else if (rows == SplitRows::owner) {
      continue;
    } else if (&cq != q[c]) {
      cq.ext_oneapi_submit_barrier({q[c]->ext_oneapi_submit_barrier()});
    }
    for (int k = 1; k < C; ++k) flag_after(cq, sp[c].flags_of[pair_split_send_order(c, k, C)] + kPairSplitOutWord + 4 * c, seq, &cq != q[c], sp[c].seq_tab);
  }
  if (rows == SplitRows::all)
    for (int c = 0; c < C; ++c) ev.done[c] = pair_split_wait(*q[c], sp[c], seq);
  return ev;
}
}  // namespace

PairSplitEvents pair_split_finish(sycl::queue *const q[2], sycl::queue *const side[2], int m, int n, const float *const part[2], bf16 *const out[2], const PairSplit sp[2],
                                  uint32_t seq, const PairSplitAdd *add, SplitRows rows) {
  if (sp[0].cards == kWireCardsMax) return quad_split_finish(q, side, m, n, part, out, sp, seq, add, rows);
  for (int c = 0; c < 2; ++c) {
    check_split(m, n, sp[c]);
    if (seq == 0 || seq >= kPairSplitSeqs || (side && side[c] && !sp[c].seq_tab)) throw std::invalid_argument("pair_split_finish: seq / its table");
  }
  PairSplitEvents ev;
  // each card waits for the peer's rows, finishes its own (bf16(own + peer), then + add), copies them
  // into the peer's out and flags that
  for (int c = 0; c < 2; ++c) {
    int r0 = 0, r1 = 0;
    pair_split_owner_rows(c, m, &r0, &r1);
    const int rn = r1 - r0;
    ev.recv[c] = flag_wait(*q[c], sp[c].flags + kPairSplitRecvWord, seq, sp[c].wait_cycles, sp[c].error, sp[c].wait_span);
    sycl::queue &cq = side && side[c] ? *side[c] : *q[c];
    if (rn > 0) {
      const float *a = part[c] + static_cast<size_t>(r0) * n, *b = sp[c].recv;
      bf16 *o = out[c] + static_cast<size_t>(r0) * n;
      const float *ap = add ? add[c].p + static_cast<size_t>(r0) * n : nullptr;
      const float *as = add ? add[c].scale + r0 : nullptr;
      // 256 columns per item in work-groups of 16, the peer's copy read past L1 only (it is
      // ordered after the flag wait's kernel boundary; the consumers read the peer-copied out rows through L2 too).
      auto body = [&](auto ADD, auto B16, auto WIDE) {
        constexpr bool kAdd = decltype(ADD)::value, kB16 = decltype(B16)::value;
        constexpr int kW = decltype(WIDE)::value ? kSumCols : kCols, kP = kW / kCols;
        const int wch = n / kW;
        const size_t items = static_cast<size_t>(rn) * wch;
        const size_t wg = items % 16 == 0 ? 16 : 1;
        ev.sum[c] = q[c]->parallel_for<PairSumK<kAdd, kB16, kW>>(sycl::nd_range<1>(items, wg), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
          const size_t id = it.get_global_id(0);
          const int r = static_cast<int>(id / wch);
          const size_t base = static_cast<size_t>(r) * n + (id % wch) * kW;
          esimd::simd<float, kCols> acc[kP];
#pragma unroll
          for (int p = 0; p < kP; ++p) acc[p] = esimd::block_load<float, kCols>(a + base + p * kCols);
#pragma unroll
          for (int p = 0; p < kP; ++p) {
            const size_t off = base + p * kCols;
            if constexpr (kB16) {  // bf16(own) + bf16(peer) in fp32, as the decode's bf16 wire (both rows' halves alike)
              const esimd::simd<bf16, kCols> mine = acc[p];
              const esimd::simd<uint32_t, kCols / 2> got = esimd::block_load<uint32_t, kCols / 2>(reinterpret_cast<const uint32_t *>(b) + off / 2, kPeerLd);
              acc[p] = esimd::simd<float, kCols>(mine) + esimd::simd<float, kCols>(mail::unpack_bf16<kCols>(got));
            } else {
              acc[p] = acc[p] + esimd::block_load<float, kCols>(b + off, kPeerLd);
            }
            if constexpr (kAdd) {
              const esimd::simd<float, kCols> pv = esimd::block_load<float, kCols>(ap + off);
              acc[p] = pv * as[r] + acc[p];
            }
          }
#pragma unroll
          for (int p = 0; p < kP; ++p) {
            const esimd::simd<bf16, kCols> v = acc[p];
            v.copy_to(o + base + p * kCols);
          }
        });
      };
      auto run = [&](auto ADD, auto B16) {
        if (n % kSumCols == 0) body(ADD, B16, std::true_type{});
        else body(ADD, B16, std::false_type{});
      };
      if (add) {
        sp[c].bf16 ? run(std::true_type{}, std::true_type{}) : run(std::true_type{}, std::false_type{});
      } else {
        sp[c].bf16 ? run(std::false_type{}, std::true_type{}) : run(std::false_type{}, std::false_type{});
      }
      if (rows == SplitRows::owner) continue;  // the sum stays with its owner
      if (&cq != q[c]) cq.ext_oneapi_submit_barrier({ev.sum[c]});
      ev.back[c] = cq.memcpy(out[1 - c] + static_cast<size_t>(r0) * n, o, static_cast<size_t>(rn) * n * 2);
    } else if (rows == SplitRows::owner) {
      continue;
    } else if (&cq != q[c]) {
      cq.ext_oneapi_submit_barrier({q[c]->ext_oneapi_submit_barrier()});
    }
    flag_after(cq, sp[c].peer_flags + kPairSplitOutWord, seq, &cq != q[c], sp[c].seq_tab);
  }
  // the consumers of out wait for the peer's rows (or later: pair_split_wait)
  if (rows == SplitRows::all)
    for (int c = 0; c < 2; ++c) ev.done[c] = pair_split_wait(*q[c], sp[c], seq);
  return ev;
}

sycl::event pair_split_share(sycl::queue &q, sycl::queue *side, int card, int m, std::span<const SplitShare> sets, const PairSplit &sp, uint32_t seq,
                             const sycl::event *after) {
  const bool quad = sp.cards == kWireCardsMax;
  if (sp.cards != 2 && !quad) throw std::invalid_argument("pair_split_share: two or four cards");
  if (card < 0 || card >= sp.cards || (quad && card != sp.card)) throw std::invalid_argument("pair_split_share: the card is not the split's");
  if (m < 1 || m > sp.rows_max || sets.empty()) throw std::invalid_argument("pair_split_share: rows / buffer sets");
  if (seq == 0 || seq >= kPairSplitSeqs || (side && !sp.seq_tab)) throw std::invalid_argument("pair_split_share: seq / its table");
  for (const SplitShare &set : sets) {
    bool named = set.bufs != nullptr && set.row_bytes > 0;
    for (int c = 0; named && c < sp.cards; ++c) named = set.bufs[c] != nullptr;
    if (!named) throw std::invalid_argument("pair_split_share: every card's buffer and the row size");
  }
  int lo = 0, hi = 0;
  pair_split_owner_rows(card, m, &lo, &hi, sp.cards);
  sycl::queue &cq = side ? *side : q;
  // (the first copy's wait list, not a barrier on the side queue: see pair_split_send)
  sycl::event dep;
  bool dep_open = side != nullptr;
  if (side) dep = after ? *after : q.ext_oneapi_submit_barrier();
  sycl::event last;
  for (int k = 1; k < sp.cards; ++k) {
    const int d = pair_split_send_order(card, k, sp.cards);
    for (const SplitShare &set : sets) {
      if (hi <= lo) break;
      const size_t at = static_cast<size_t>(lo) * set.row_bytes, bytes = static_cast<size_t>(hi - lo) * set.row_bytes;
      const std::byte *src = static_cast<const std::byte *>(set.bufs[card]) + at;
      std::byte *dst = static_cast<std::byte *>(set.bufs[d]) + at;
      last = dep_open ? cq.memcpy(dst, src, bytes, dep) : cq.memcpy(dst, src, bytes);
      dep_open = false;
    }
    uint32_t *line = quad ? sp.flags_of[d] + kPairSplitOutWord + 4 * card : sp.peer_flags + kPairSplitOutWord;
    if (!line) throw std::invalid_argument("pair_split_share: the destination card's flag line");
    last = flag_after(cq, line, seq, side != nullptr, sp.seq_tab, dep_open ? &dep : nullptr);
    dep_open = false;
  }
  return last;
}

sycl::event pair_split_wait(sycl::queue &q, const PairSplit &sp, uint32_t seq) {
  if (seq == 0 || seq >= kPairSplitSeqs || !sp.flags || !sp.error) throw std::invalid_argument("pair_split_wait: seq / flags");
  if (sp.cards == kWireCardsMax)  // every other card's rows
    return flag_wait_others(q, sp.flags + kPairSplitOutWord, sp.card, sp.cards, seq, sp.wait_cycles, sp.error, sp.wait_span ? sp.wait_span + 2 : nullptr);
  return flag_wait(q, sp.flags + kPairSplitOutWord, seq, sp.wait_cycles, sp.error, sp.wait_span ? sp.wait_span + 2 : nullptr);
}

PairSplitEvents pair_split_rows(sycl::queue *const q[2], int m, int n, const float *const part[2], bf16 *const out[2], const PairSplit sp[2], uint32_t seq,
                                const PairSplitAdd *add, sycl::queue *const side[2]) {
  const int C = sp[0].cards == kWireCardsMax ? kWireCardsMax : 2;
  sycl::event copy[kWireCardsMax];
  for (int c = 0; c < C; ++c) copy[c] = pair_split_send(*q[c], side ? side[c] : nullptr, c, m, n, part[c], sp[c], seq);
  PairSplitEvents ev = pair_split_finish(q, side, m, n, part, out, sp, seq, add);
  for (int c = 0; c < C; ++c) ev.copy[c] = copy[c];
  return ev;
}

namespace {

}  // namespace

size_t pair_rows_wire_words(int rows_max, int n_max, int cards) { return 2ull * rows_max * static_cast<size_t>(n_max) * wire_hops(cards); }
sycl::event pair_rows_wire_init(sycl::queue &q, int rows_max, int n_max, uint32_t *inbox, int cards) {
  return q.fill(inbox, kPairMailEmpty, pair_rows_wire_words(rows_max, n_max, cards));
}

namespace {
// The wire's instance by its slot columns (pair_cols): the two-card or the four-card exchange.
template <bool QUAD>
sycl::event reduce_rows_wire(sycl::queue &q, int m, int n, const float *part, const PairRowsWire &w, int p, bf16 *out) {
  const int cols = pair_cols(m, n, w.bf16);
  if (w.bf16) {
    if (cols == 512) return reduce_rows<true, true, 512, QUAD>(q, m, n, part, w, p, out);
    if (cols == 256) return reduce_rows<true, true, 256, QUAD>(q, m, n, part, w, p, out);
    if (cols == 128) return reduce_rows<true, true, 128, QUAD>(q, m, n, part, w, p, out);
    return reduce_rows<true, true, kCols, QUAD>(q, m, n, part, w, p, out);
  }
  if (cols == 256) return reduce_rows<true, false, 256, QUAD>(q, m, n, part, w, p, out);
  if (cols == 128) return reduce_rows<true, false, 128, QUAD>(q, m, n, part, w, p, out);
  return reduce_rows<true, false, kCols, QUAD>(q, m, n, part, w, p, out);
}
template <bool QUAD>
sycl::event reduce_wire(sycl::queue &q, int rows, int n, const float *part, const PairWire &w, int p, bf16 *out) {
  const int cols = pair_cols(rows, n, w.bf16);
  if (w.bf16) {
    if (cols == 512) return reduce<true, true, 512, QUAD>(q, rows, n, part, w, p, out);
    if (cols == 256) return reduce<true, true, 256, QUAD>(q, rows, n, part, w, p, out);
    if (cols == 128) return reduce<true, true, 128, QUAD>(q, rows, n, part, w, p, out);
    return reduce<true, true, kCols, QUAD>(q, rows, n, part, w, p, out);
  }
  if (cols == 256) return reduce<true, false, 256, QUAD>(q, rows, n, part, w, p, out);
  if (cols == 128) return reduce<true, false, 128, QUAD>(q, rows, n, part, w, p, out);
  return reduce<true, false, kCols, QUAD>(q, rows, n, part, w, p, out);
}
// A four-card wire names this card and all four inboxes.
void check_quad(const WireQuad &qd, const char *who) {
  if (!qd.complete())
    throw std::invalid_argument(std::string(who) + ": a four-card wire needs its card in 0..3 (got " + std::to_string(qd.card) + ") and every card's inbox");
}
}  // namespace

sycl::event pair_reduce_rows(sycl::queue &q, int m, int n, const float *part, const PairRowsWire *wire, int parity, bf16 *out) {
  if (m < 1 || n % kCols || (wire && (m > wire->rows_max || n > wire->n_max))) throw std::invalid_argument("pair_reduce_rows: shape");
  if (!wire) return reduce_rows<false, false, kCols>(q, m, n, part, PairRowsWire{}, 0, out);
  const int p = parity & 1;
  if (wire->quad.on()) {
    check_quad(wire->quad, "pair_reduce_rows");
    if (wire->split) throw std::invalid_argument("pair_reduce_rows: the split form is two-card only");
    if (wire->quad.copy && !wire->bf16)
      return step_copy_reduce(q, m, n, part, *wire->quad.copy, wire->quad.card, p, wire->error, wire->wait_cycles, out);
    return reduce_rows_wire<true>(q, m, n, part, *wire, p, out);
  }
  return reduce_rows_wire<false>(q, m, n, part, *wire, p, out);
}

size_t pair_wire_words(int n_max, int cards) { return 2ull * kM * static_cast<size_t>(n_max) * wire_hops(cards); }
sycl::event pair_wire_init(sycl::queue &q, int n_max, uint32_t *inbox, int cards) { return q.fill(inbox, kPairMailEmpty, pair_wire_words(n_max, cards)); }

int pair_cols(int rows, int n, bool bf16) {
  if (rows <= 2) return kCols;  // 64 columns (128 B bf16, 256 B fp32)
  int cols = (rows <= 5 ? 512 : 1024) / (bf16 ? 2 : 4);  // slot bytes per item
  while (cols > kCols && n % cols) cols /= 2;
  return cols;
}

int pair_cols_i8(int rows, int n) {
  int cols = rows <= 3 ? kCols : rows <= 5 ? 256 : rows < 20 ? 512 : 1024;
  while (cols > kCols && n % cols) cols /= 2;
  return cols;
}

// ---- a decode step's rows by owner (wire.hpp) -------------------------------------------------------------------------
namespace {
template <int COLS>
class OwnerSumK;
template <int WORDS>
class OwnerShareK;

// The owner sum as pair_reduce's launch: one item per (row, COLS columns), one item per work-group (the peer stores
// want every core, as the two-card sum's).
constexpr int kOwnerSumCols = 128;
template <int COLS>
sycl::event owner_reduce(sycl::queue &q, int rows, int n, const float *part, const PairWire &wire, int parity, bf16 *out) {
  const int chunks = n / COLS;
  const mail::OwnerBoxes boxes = mail::owner_boxes(wire.owner);
  const uint64_t wait_cycles = wire.wait_cycles;
  uint32_t *error = wire.error;
  const uint32_t zero = static_cast<uint32_t>(rows < 0);  // 0 (rows >= 4): a kernel argument, unknown to the compiler
  return q.parallel_for<OwnerSumK<COLS>>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks)}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * COLS;  // a (row, chunk) grid
    esimd::simd<float, COLS> acc = esimd::block_load<float, COLS>(part + static_cast<size_t>(r) * n + c0);
    if (!mail::owner_sum<COLS>(acc, boxes, r, rows, n, c0, parity, kPairMailEmpty, wait_cycles, error, zero)) return;
    const esimd::simd<bf16, COLS> o = acc;
    o.copy_to(out + static_cast<size_t>(r) * n + c0);
  });
}

// Shared rows: one item per (row, WORDS 32-bit words of it = 2 WORDS bf16 values), one item per work-group. The owner's
// items store into the three other cards' slots, the others' items take their slot into the buffer.
// The buffer is read and written as bf16 and packed for the mailbox: the same launch on uint32 loads of the buffer
// crashes the device compiler at 128 words per item (ocloc segfault, oneAPI 2026.2).
template <int WORDS>
sycl::event share_rows(sycl::queue &q, int rows, int n, bf16 *buf, const OwnerShare &share, int parity) {
  constexpr int kVals = 2 * WORDS;
  const int row_words = n / 2, chunks = row_words / WORDS;
  const mail::OwnerBoxes boxes = mail::owner_boxes(share.mail);
  const uint64_t wait_cycles = share.wait_cycles;
  uint32_t *error = share.error;
  return q.parallel_for<OwnerShareK<WORDS>>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(chunks)}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), w0 = static_cast<int>(it.get_global_id(1)) * WORDS;  // a (row, chunk) grid
    bf16 *at = buf + static_cast<size_t>(r) * n + 2 * w0;
    const size_t slot = (static_cast<size_t>(parity) * kM + r) * row_words + w0;
    if (row_owner(r, rows) == boxes.card) {
      esimd::simd<bf16, kVals> mine;
      mine.copy_from(at);
      const esimd::simd<uint32_t, WORDS> words = mail::pack_bf16<kVals>(mine);
#pragma unroll
      for (int k = 1; k < kWireCardsMax; ++k)
        mail::mail_send_words<WORDS>(boxes.of[(boxes.card + k) % kWireCardsMax] + slot, words, kPairMailEmpty, mail::kNanPairWord);
      return;
    }
    const esimd::simd<uint32_t, WORDS> got = mail::mail_take_words<WORDS>(boxes.of[boxes.card] + slot, kPairMailEmpty, wait_cycles, error, mail::kNanPairWord);
    const esimd::simd<bf16, kVals> theirs = mail::unpack_bf16<kVals>(got);
    theirs.copy_to(at);
  });
}
constexpr int kOwnerShareWideRows = 17;  // 256 words per item from here
}  // namespace

size_t owner_sum_words(int n_max) { return 2ull * kWireCardsMax * kOwnerRowsMax * static_cast<size_t>(n_max); }
size_t owner_share_words(int n_max) { return 2ull * kM * static_cast<size_t>(n_max / 2); }
sycl::event owner_mail_init(sycl::queue &q, uint32_t *inbox, size_t words) { return q.fill(inbox, kPairMailEmpty, words); }

sycl::event owner_share(sycl::queue &q, int rows, int n, bf16 *buf, const OwnerShare &share, int parity) {
  if (rows < kOwnerMinRows || rows > kM || n < 128 || n % 128) throw std::invalid_argument("owner_share: rows 4..kPairRows (32), n % 128 == 0");
  if (!share.mail.complete() || !share.error)
    throw std::invalid_argument("owner_share: the mailbox needs its card in 0..3 (got " + std::to_string(share.mail.card) + "), every card's inbox and an error word");
  const int p = parity & 1;
  if (rows >= kOwnerShareWideRows && n % 512 == 0) return share_rows<256>(q, rows, n, buf, share, p);
  if (n % 256 == 0) return share_rows<128>(q, rows, n, buf, share, p);
  return share_rows<64>(q, rows, n, buf, share, p);
}

namespace {
class StepCopyWait;
class StepCopySum;
constexpr auto kStepFlag = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                               esimd::cache_hint_L2<esimd::cache_hint::uncached>, esimd::alignment<16>};
class StepCopyShareWait;
class StepCopyDirectWait;
class StepCopyShareOut;

template <bool SHARE, bool DIRECT = false>
sycl::event step_copy_sync(sycl::queue &q, uint32_t *state, uint32_t *ready, int card, uint32_t *error, uint64_t wait_cycles,
                           bf16 *out = nullptr, size_t items = 0) {
  constexpr int C = kWireCardsMax, kFlagWords = 16;
  using Wait = std::conditional_t<DIRECT, StepCopyDirectWait, std::conditional_t<SHARE, StepCopyShareWait, StepCopyWait>>;
  return q.single_task<Wait>([=]() SYCL_ESIMD_KERNEL {
    uint32_t seq = state[0];
    if constexpr (!SHARE) {
      state[0] = ++seq;
      state[1] = 0;
    }
    esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
    esimd::block_store<uint32_t, 4>(ready + card * kFlagWords, esimd::simd<uint32_t, 4>(seq), kStepFlag);
    const uint64_t start = esimd::rdtsc();
    constexpr auto poll = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                             esimd::cache_hint_L2<esimd::cache_hint::uncached>, esimd::alignment<4>};
    const esimd::simd<uint32_t, C> offsets(0, kFlagWords * sizeof(uint32_t));
    while (true) {
      const auto seen = esimd::gather<uint32_t, C>(ready, offsets, esimd::simd_mask<C>(1), poll);
      if ((seen == seq).all()) break;
      if (esimd::rdtsc() - start > wait_cycles) {
        error[0] = state[1] = 1u;
        break;
      }
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
    }
    if constexpr (DIRECT) {
      if (state[1]) {
        const esimd::simd<bf16, kCols> bad = bf16(__builtin_nanf(""));
        for (size_t off = 0; off < items; off += kCols) bad.copy_to(out + off);
      }
    }
  });
}
}  // namespace

sycl::event step_copy_reduce(sycl::queue &q, int rows, int n, const float *part, const StepCopy &buffers, int card, int parity,
                             uint32_t *error, uint64_t wait_cycles, bf16 *out) {
  constexpr int C = kWireCardsMax, kFlagWords = 16;
  const size_t items = static_cast<size_t>(rows) * n;
  if (card < 0 || card >= C || rows < 1 || n < 1 || n % kCols || items > buffers.stride || !buffers.ready)
    throw std::invalid_argument("step copy: mailbox shape");
  // A rank cannot start reusing a parity until the other ranks have published
  // the intervening operation, which follows their reads of this one.
  const size_t base = static_cast<size_t>(parity & 1) * C * buffers.stride;
  const size_t flag_base = static_cast<size_t>(parity & 1) * C * kFlagWords;
  const bool split = buffers.shared_ready && rows >= buffers.split_min_rows;
  const auto row_begin = [=](int rank) { return static_cast<size_t>(rows * rank / C) * n; };
  for (int hop = 1; hop < C; ++hop) {
    const int dest = (card + hop) % C;
    const size_t start = split ? row_begin(dest) : 0;
    const size_t count = split ? row_begin(dest + 1) - start : items;
    q.memcpy(buffers.recv[dest] + base + card * buffers.stride + start, part + start, count * sizeof(float));
  }
  auto *state = buffers.state[card];
  step_copy_sync<false>(q, state, buffers.ready + flag_base, card, error, wait_cycles);
  const size_t begin = split ? row_begin(card) : 0;
  const size_t count = split ? row_begin(card + 1) - begin : items;
  const size_t shared_at = static_cast<size_t>(parity & 1) * buffers.stride;
  bool direct = split && buffers.outputs[card] == out;
  for (const auto *output : buffers.outputs) direct = direct && output != nullptr;
  bf16 *const reduced = split && !direct ? buffers.shared[card] + shared_at : out;
  const float *src[C];
  for (int i = 0; i < C; ++i) src[i] = i == card ? part : buffers.recv[card] + base + i * buffers.stride;
  const uint32_t zero = static_cast<uint32_t>(rows < 0);
  auto sum = q.parallel_for<StepCopySum>(sycl::range<1>(count / kCols), [=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
    const size_t off = begin + item[0] * kCols;
    esimd::simd<bf16, kCols> result;
    if (state[1]) {
      result = bf16(__builtin_nanf(""));
    } else {
      const auto p0 = esimd::block_load<float, kCols>(src[0] + off, mail::kUc);
      const auto p1 = esimd::block_load<float, kCols>(src[1] + off, mail::kUc);
      const auto p2 = esimd::block_load<float, kCols>(src[2] + off, mail::kUc);
      const auto p3 = esimd::block_load<float, kCols>(src[3] + off, mail::kUc);
      result = mail::opaque<kCols>(p0 + p1, zero) + mail::opaque<kCols>(p2 + p3, zero);
    }
    result.copy_to(reduced + off);
  });
  if (!split) return sum;
  // Only the assigned rows are reduced on each rank, then shared at BF16.
  // The second flag bank separates partial arrival from output arrival.
  for (int hop = 1; hop < C; ++hop) {
    const int dest = (card + hop) % C;
    bf16 *const target = direct ? buffers.outputs[dest] : buffers.shared[dest] + shared_at;
    q.memcpy(target + begin, reduced + begin, count * sizeof(bf16));
  }
  if (direct) {
    // All peers passed the first barrier after their previous consumers of out.
    // The second barrier publishes the BF16 copies before this rank consumes them.
    return step_copy_sync<true, true>(q, state, buffers.shared_ready + flag_base, card, error, wait_cycles, out, items);
  }
  step_copy_sync<true>(q, state, buffers.shared_ready + flag_base, card, error, wait_cycles);
  return q.parallel_for<StepCopyShareOut>(sycl::range<1>(items / kCols), [=](sycl::id<1> item) SYCL_ESIMD_KERNEL {
    const size_t off = item[0] * kCols;
    esimd::simd<bf16, kCols> result = bf16(__builtin_nanf(""));
    if (!state[1]) result.copy_from(reduced + off, esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>,
                                                                     esimd::cache_hint_L2<esimd::cache_hint::cached>,
                                                                     esimd::alignment<128>});
    result.copy_to(out + off);
  });
}

sycl::event pair_reduce(sycl::queue &q, int rows, int n, const float *part, const PairWire *wire, int parity, bf16 *out) {
  if (rows < 1 || rows > kM || n % kCols) throw std::invalid_argument("pair_reduce: rows 1..kPairRows (32), n % 64 == 0");
  if (!wire) return reduce<false, false, kCols>(q, rows, n, part, PairWire{}, 0, out);
  const int p = parity & 1;
  if (wire->owner.on()) {  // this card's own rows only
    if (!wire->owner.complete() || !wire->error)
      throw std::invalid_argument("pair_reduce: an owner sum needs its card in 0..3 (got " + std::to_string(wire->owner.card) + "), every card's inbox and an error word");
    if (rows < kOwnerMinRows || wire->bf16 || wire->i8) throw std::invalid_argument("pair_reduce: an owner sum takes 4+ rows of fp32 partials");
    return n % kOwnerSumCols == 0 ? owner_reduce<kOwnerSumCols>(q, rows, n, part, *wire, p, out) : owner_reduce<kCols>(q, rows, n, part, *wire, p, out);
  }
  if (wire->quad.on()) {
    check_quad(wire->quad, "pair_reduce");
    if (wire->i8) throw std::invalid_argument("pair_reduce: the int8 wire is two-card only");
    if (wire->quad.copy && !wire->bf16)
      return step_copy_reduce(q, rows, n, part, *wire->quad.copy, wire->quad.card, p, wire->error, wire->wait_cycles, out);
    return reduce_wire<true>(q, rows, n, part, *wire, p, out);
  }
  if (wire->i8) {
    const int ci = pair_cols_i8(rows, n);
    if (ci == 1024) return reduce_i8<1024>(q, rows, n, part, *wire, p, out);
    if (ci == 512) return reduce_i8<512>(q, rows, n, part, *wire, p, out);
    if (ci == 256) return reduce_i8<256>(q, rows, n, part, *wire, p, out);
    if (ci == 128) return reduce_i8<128>(q, rows, n, part, *wire, p, out);
    return reduce_i8<kCols>(q, rows, n, part, *wire, p, out);
  }
  return reduce_wire<false>(q, rows, n, part, *wire, p, out);
}

}  // namespace xenithium::kernels
