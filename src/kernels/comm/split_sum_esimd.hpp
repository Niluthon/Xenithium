// Xenithium · device side of the four-card split sum (private to src/kernels): the wait for the other cards' flag
// lines and the owner's per-element sum. pair_split_finish's own kernels and the consumers that take a pending sum into
// their first launch (comm/wire.hpp QuadOwnerSum) share these, so both write the same bits.
#pragma once

#include <cstdint>

#include <sycl/ext/intel/esimd.hpp>

#include "comm/mailbox_esimd.hpp"
#include "xenithium/kernels/comm/wire.hpp"

namespace xenithium::kernels::split {

namespace esimd = sycl::ext::intel::esimd;

inline constexpr int kCols = 64;  // columns per operand load
// the peer-written split buffers: past L1 (incoherent per core), through L2 like every other peer-copied buffer
inline constexpr auto kPeerLd = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::cached>,
                                                  esimd::alignment<128>};
// The same past L2 too: the rows a kernel reads once right after its own wait (wait_flag_line_quad). Measured on the
// four-card micro at 1,024-row sums: -6 us a call against kPeerLd (the copied rows do not displace L2's other lines).
inline constexpr auto kPeerLdUc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                                    esimd::alignment<128>};
template <bool L2_UC, class T, int N>
inline esimd::simd<T, N> peer_load(const T *p) SYCL_ESIMD_FUNCTION {
  if constexpr (L2_UC) return esimd::block_load<T, N>(p, kPeerLdUc);
  else return esimd::block_load<T, N>(p, kPeerLd);
}
// Spins until every other card's flag line (base + 4 s, s != card) holds seq, one line after the other; on a timeout
// sets *error and gives up. One work-item polls; the caller orders the rest of its work after it. Returns the entry
// time (rdtsc) for a diagnostic span.
inline uint64_t wait_flag_lines(const uint32_t *base, int card, int cards, uint32_t seq, uint64_t wait_cycles, uint32_t *error) SYCL_ESIMD_FUNCTION {
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
  return t0;
}

// The four cards' flag lines of one use sit in one 64-byte line (QuadOwnerSum::recv_flags, 64-byte aligned).
inline constexpr int kQuadFlagWords = 4 * kWireCardsMax;
static_assert(kQuadFlagWords == 16 && kPairSplitRecvWord % kQuadFlagWords == 0, "a use's four flag lines in one 64-byte line");
// GT cycles a polling item idles between two loads (~0.1 us): many work-groups may poll at once.
inline constexpr uint64_t kQuadPollBackoff = 256;

// wait_flag_lines for four cards in a kernel where every work-group waits for itself (a consumer that takes a pending
// sum into its first launch): one load of the 64-byte line reads every source's line, through L2 (not past it, as the
// one-item wait kernel's loads): a peer's copy into this card's memory lands in L2, so a line held there updates, and a
// poll that stays in L2 keeps hundreds of polling work-groups off the memory (past L2 they cost ~0.1 us each a
// work-group).
inline uint64_t wait_flag_line_quad(const uint32_t *base, int card, uint32_t seq, uint64_t wait_cycles, uint32_t *error) SYCL_ESIMD_FUNCTION {
  const uint64_t t0 = esimd::rdtsc();
  auto arrived = [&](esimd::simd<uint32_t, kQuadFlagWords> v) SYCL_ESIMD_FUNCTION {
    bool all = true;
#pragma unroll
    for (int s = 0; s < kWireCardsMax; ++s) all = all && (s == card || v[4 * s] == seq);
    return all;
  };
  esimd::simd<uint32_t, kQuadFlagWords> v = esimd::block_load<uint32_t, kQuadFlagWords>(base, kPeerLd);
  while (!arrived(v)) {
    const uint64_t t1 = esimd::rdtsc();
    while (esimd::rdtsc() - t1 < kQuadPollBackoff) {
    }
    v = esimd::block_load<uint32_t, kQuadFlagWords>(base, kPeerLd);
    if (!arrived(v) && esimd::rdtsc() - t0 > wait_cycles) {
      error[0] = 1u;
      break;
    }
  }
  return t0;
}

inline void write_span(uint64_t *span, uint64_t t0) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint64_t, 2> sp;
  sp[0] = t0;
  sp[1] = esimd::rdtsc();
  sp.copy_to(span);
}

// One operand of the four-card owner sum: this card's fp32 partial (bf16 wire: rounded to bf16), or a source card's slot
// (peer-copied: past L1; bf16 wire: packed bf16).
template <bool B16, bool L2_UC>
inline esimd::simd<float, kCols> operand(bool own, const float *a, const float *slot, size_t off) SYCL_ESIMD_FUNCTION {
  if (own) {
    const esimd::simd<float, kCols> mine = esimd::block_load<float, kCols>(a + off);
    if constexpr (B16) {
      const esimd::simd<sycl::ext::oneapi::bfloat16, kCols> mb = mine;
      return esimd::simd<float, kCols>(mb);
    } else {
      return mine;
    }
  }
  if constexpr (B16) {
    const esimd::simd<uint32_t, kCols / 2> got =
        peer_load<L2_UC, uint32_t, kCols / 2>(reinterpret_cast<const uint32_t *>(reinterpret_cast<const sycl::ext::oneapi::bfloat16 *>(slot) + off));
    return esimd::simd<float, kCols>(mail::unpack_bf16<kCols>(got));
  } else {
    return peer_load<L2_UC, float, kCols>(slot + off);
  }
}

// The owner's fp32 value of kCols elements at `off` (row r of its own rows): (p0 + p1) + (p2 + p3), then with ADD
// opq(p x scale) + opq(sum). The build's fast floating point reassociates four register operands (a clang fp pragma
// does not stop it), so each pair's sum passes through an XOR with a runtime zero the compiler cannot see through
// (mail::opaque), then the two are added. zero: a kernel argument that is 0.
template <bool ADD, bool B16, bool L2_UC = false>
inline esimd::simd<float, kCols> owner_value(int card, const float *a, const float *const b[kWireCardsMax], const float *ap, float as_r, size_t off,
                                             uint32_t zero) SYCL_ESIMD_FUNCTION {
  static_assert(kWireCardsMax == 4, "the four-card order");
  const esimd::simd<float, kCols> x0 = operand<B16, L2_UC>(card == 0, a, b[0], off), x1 = operand<B16, L2_UC>(card == 1, a, b[1], off),
                                  x2 = operand<B16, L2_UC>(card == 2, a, b[2], off), x3 = operand<B16, L2_UC>(card == 3, a, b[3], off);
  esimd::simd<float, kCols> acc = mail::opaque<kCols>(x0 + x1, zero) + mail::opaque<kCols>(x2 + x3, zero);
  if constexpr (ADD) {  // bf16(opq(p x scale) + opq(sum)): moe_rows' four-card combine writes the same
    const esimd::simd<float, kCols> pv = esimd::block_load<float, kCols>(ap + off);
    acc = mail::opaque<kCols>(pv * as_r, zero) + mail::opaque<kCols>(acc, zero);
  }
  return acc;
}

}  // namespace xenithium::kernels::split
