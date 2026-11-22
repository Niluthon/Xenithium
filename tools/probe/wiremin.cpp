// xenithium-probe wiremin (scratch): bisection of the two-card pair_reduce mismatch on this host.
//
// Findings so far, from this tool: the fp32 wire loses the peer's payload (the sum carries the owner's partial alone, no
// timeout, no error word) only on the rank whose exchange is launched first, in 64-byte sectors; the bf16 wire (32-word
// messages, half a slot) never loses it; the slot's pre-fill is in memory (--checkinit), and neither a kernel-written
// fill (--kfill) nor a system-scope clean of it (--flushinit) changes anything.
//
// --sendonly isolates the store: one card loads its partial and block-stores it into the peer's slot with the wire's own
// access (64 uncached words, alignment<128>), no receiver; the host then reads the slot back and compares.
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/ext/intel/esimd.hpp>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/rt/fabric.hpp"

#include "probe.hpp"

namespace esimd = sycl::ext::intel::esimd;
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr auto kUc = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>,
                                       esimd::alignment<128>};
constexpr uint16_t kQNaN = 0x7fc0u;

uint16_t b16bits(bf16 v) {
  uint16_t b;
  std::memcpy(&b, &v, 2);
  return b;
}
uint16_t fb16(float f) { return b16bits(static_cast<bf16>(f)); }
float decode(uint16_t bf) {
  const uint32_t u = static_cast<uint32_t>(bf) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

class FillK;
// ---- the wire's own plumbing, copied verbatim from src/kernels/comm/mailbox_esimd.hpp + wire.cpp ----
template <int N>
inline void wire_send(uint32_t *peer_slot, esimd::simd<uint32_t, N> words, uint32_t empty, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  words.merge(esimd::simd<uint32_t, N>(nan_word), words == empty);
  esimd::block_store<uint32_t, N>(peer_slot, words, kUc);
}
template <int N>
inline esimd::simd<uint32_t, N> wire_take(uint32_t *slot, uint32_t empty, uint64_t wait_cycles, uint32_t *error, uint32_t nan_word) SYCL_ESIMD_FUNCTION {
  esimd::simd<uint32_t, N> got = esimd::block_load<uint32_t, N>(slot, kUc);
  bool ok = !(got == empty).any();
  if (!ok) {
    const uint64_t t0 = esimd::rdtsc();
    while (!ok) {
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
      got = esimd::block_load<uint32_t, N>(slot, kUc);
      ok = !(got == empty).any();
      if (!ok && esimd::rdtsc() - t0 > wait_cycles) break;
    }
  }
  if (!ok) {
    error[0] = 1u;
    got = nan_word;
  }
  esimd::block_store<uint32_t, N>(slot, esimd::simd<uint32_t, N>(empty), kUc);
  return got;
}
template <bool B16, int COLS>
inline esimd::simd<float, COLS> wire_exchange(esimd::simd<float, COLS> acc, uint32_t *peer_slot, uint32_t *slot, uint64_t wait_cycles, uint32_t *error) {
  constexpr int W = B16 ? COLS / 2 : COLS, P = W < 128 ? W : 128;
  const uint32_t nan_word = B16 ? 0x7fc07fc0u : 0x7fc00000u;
  esimd::simd<float, COLS> mine_f = acc;
  esimd::simd<uint32_t, W> mine;
  if constexpr (B16) {
    const esimd::simd<bf16, COLS> mb = acc;
    mine_f = mb;
    mine = mb.template bit_cast_view<uint32_t>().read();
  } else {
    mine = acc.template bit_cast_view<uint32_t>();
  }
#pragma unroll
  for (int p = 0; p < W / P; ++p) wire_send<P>(peer_slot + p * P, esimd::simd<uint32_t, P>(mine.template select<P, 1>(p * P)), k::kPairMailEmpty, nan_word);
  esimd::simd<uint32_t, W> got;
#pragma unroll
  for (int p = 0; p < W / P; ++p) got.template select<P, 1>(p * P) = wire_take<P>(slot + p * P, k::kPairMailEmpty, wait_cycles, error, nan_word);
  if constexpr (B16) return mine_f + esimd::simd<float, COLS>(got.template bit_cast_view<bf16>());
  else return acc + got.template bit_cast_view<float>();
}

class ExactK;

// reduce<true, false, 64, false>'s body for the two-card fp32 wire: the engine's own plumbing, with the sum left as raw
// fp32 words instead of bf16 so the host can see exactly what the exchange produced.
void exact_exchange(sycl::queue &q, const float *part, uint32_t *sum, uint32_t *peer_slot, uint32_t *slot, uint32_t *error, int rows, int n, int parity) {
  const size_t chunks = static_cast<size_t>(n) / 64;
  q.parallel_for<ExactK>(sycl::nd_range<2>({static_cast<size_t>(rows), chunks}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * 64;
    const size_t at = (static_cast<size_t>(parity) * 64 + r) * n + c0;
    esimd::simd<float, 64> acc = esimd::block_load<float, 64>(part + static_cast<size_t>(r) * n + c0);
    acc = wire_exchange<false, 64>(acc, peer_slot + at, slot + at, 2800000000ull, error);
    esimd::block_store<float, 64>(reinterpret_cast<float *>(sum) + static_cast<size_t>(r) * n + c0, acc, kUc);
  });
}

class SendOnlyK;

// The empty pattern written by a kernel (the wire's own store) instead of q.fill.
void kfill_empty(sycl::queue &q, uint32_t *buf, size_t words) {
  q.parallel_for<FillK>(sycl::range<1>(words / 64), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
    esimd::block_store<uint32_t, 64>(buf + static_cast<size_t>(id[0]) * 64, esimd::simd<uint32_t, 64>(k::kPairMailEmpty), kUc);
  });
}

// mail_send's job for one 64-word message, without the receiver: load the partial the same way `reduce` does, then store
// it into the peer's slot with the wire's access.
void send_only(sycl::queue &q, const float *part, uint32_t *peer_slot, int rows, int n, int cols) {
  const size_t chunks = static_cast<size_t>(n) / static_cast<size_t>(cols);
  q.parallel_for<SendOnlyK>(sycl::nd_range<2>({static_cast<size_t>(rows), chunks}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * cols;
    esimd::simd<float, 64> acc = esimd::block_load<float, 64>(part + static_cast<size_t>(r) * n + c0);
    esimd::simd<uint32_t, 64> w = acc.template bit_cast_view<uint32_t>();
    esimd::block_store<uint32_t, 64>(peer_slot + static_cast<size_t>(r) * n + c0, w, kUc);
  });
}

class RecvOnlyK;

// mail_take's job, with a dump of the words it read: spin until no word is the empty pattern, write them to `dump`, then
// empty the slot. What the receiver actually saw, which the sum alone cannot show.
void recv_only(sycl::queue &q, uint32_t *slot, uint32_t *dump, uint32_t *err, int rows, int n, int cols) {
  const size_t chunks = static_cast<size_t>(n) / static_cast<size_t>(cols);
  q.parallel_for<RecvOnlyK>(sycl::nd_range<2>({static_cast<size_t>(rows), chunks}, {1, 1}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    const int r = static_cast<int>(it.get_global_id(0)), c0 = static_cast<int>(it.get_global_id(1)) * cols;
    uint32_t *at = slot + static_cast<size_t>(r) * n + c0;
    esimd::simd<uint32_t, 64> got = esimd::block_load<uint32_t, 64>(at, kUc);
    bool ok = !(got == k::kPairMailEmpty).any();
    const uint64_t t0 = esimd::rdtsc();
    while (!ok) {
      esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::none, esimd::fence_scope::system>();
      got = esimd::block_load<uint32_t, 64>(at, kUc);
      ok = !(got == k::kPairMailEmpty).any();
      if (!ok && esimd::rdtsc() - t0 > 2800000000ull) {
        err[0] = 1u;
        break;
      }
    }
    esimd::block_store<uint32_t, 64>(dump + static_cast<size_t>(r) * n + c0, got, kUc);
    esimd::block_store<uint32_t, 64>(at, esimd::simd<uint32_t, 64>(k::kPairMailEmpty), kUc);
  });
}

}  // namespace

int cmd_wiremin(int argc, char **argv) {
  int rows = 1, n = 2560, calls = 3, b16 = 0, parmode = 0, fresh = 0, barrier = 0, outfill = 0, swap = 0, launch_rev = 0, inboxfill = 0,
      checkinit = 0, flushinit = 0, kfill = 0, sendonly = 0, recvonly = 0, exact = 0;
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has = i + 1 < argc;
    if (a == "--rows" && has) rows = std::stoi(argv[++i]);
    else if (a == "--n" && has) n = std::stoi(argv[++i]);
    else if (a == "--calls" && has) calls = std::stoi(argv[++i]);
    else if (a == "--b16") b16 = 1;
    else if (a == "--parmode") parmode = 1;        // keep the parity fixed instead of alternating
    else if (a == "--fresh") fresh = 1;            // re-allocate and re-init the inboxes before every call
    else if (a == "--barrier") barrier = 1;        // wait for both cards between calls
    else if (a == "--outfill") outfill = 1;        // pre-fill out[] with 0xFFFF (a NaN bf16)
    else if (a == "--swap") swap = 1;              // the fabric's ranks {1, 0}
    else if (a == "--launch-rev") launch_rev = 1;  // launch the exchanges in reverse rank order
    else if (a == "--inboxfill") inboxfill = 1;    // fill the inboxes with 0x7FC00000 (a non-empty NaN word)
    else if (a == "--checkinit") checkinit = 1;    // read the inboxes back after pair_wire_init
    else if (a == "--flushinit") flushinit = 1;    // clean (write back) the fill before every call
    else if (a == "--kfill") kfill = 1;            // empty the inboxes with a kernel write instead of q.fill
    else if (a == "--sendonly") sendonly = 1;      // only the store half of the exchange, then verify the slot
    else if (a == "--recvonly") recvonly = 1;      // store half on one card, take half on the other, then verify the take's dump
    else if (a == "--exact") exact = 1;            // the engine's own exchange plumbing, sum kept as fp32 words
    else {
      std::printf("wiremin: unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  const auto loader = xenithium::l0::Loader::open();
  if (loader.size() < 2) {
    std::printf("wiremin: needs two cards\n");
    return 2;
  }
  xenithium::Fabric f(loader, swap ? std::vector<size_t>{1, 0} : std::vector<size_t>{0, 1}, false);
  constexpr int kC = 2, kRowsMax = 64, kNMax = 5120;
  const size_t words = k::pair_wire_words(kNMax);
  const size_t elems = static_cast<size_t>(rows) * n;
  float *part[kC];
  bf16 *out[kC];
  uint32_t *err[kC], *inbox[kC];
  for (int c = 0; c < kC; ++c) {
    auto &q = f.compute(c);
    part[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * kNMax * 4));
    out[c] = static_cast<bf16 *>(f.alloc(c, size_t(kRowsMax) * kNMax * 2));
    err[c] = static_cast<uint32_t *>(f.alloc(c, 64));
    q.memset(err[c], 0, 64).wait();
    inbox[c] = static_cast<uint32_t *>(f.alloc_mailbox(c, words * 4));
    k::pair_wire_init(q, kNMax, inbox[c]).wait();
  }
  std::printf("wiremin: rows %d, n %d, %d calls, %s wire, cols/item %d, parity %s, inbox %s, barrier %s%s%s%s%s%s\n", rows, n, calls,
              b16 ? "bf16" : "fp32", k::pair_cols(rows, n, b16 != 0), parmode ? "fixed" : "alternating",
              fresh ? "re-allocated per call" : "reused", barrier ? "between calls" : "none", swap ? ", ranks {1,0}" : "",
              launch_rev ? ", launched in reverse" : "", sendonly ? ", store only" : "", kfill ? ", kernel fill" : "", flushinit ? ", fill cleaned" : "");

  if (checkinit) {
    for (int c = 0; c < kC; ++c) {
      std::vector<uint32_t> back(words);
      f.compute(c).wait();
      f.copy(c).copy(back.data(), inbox[c], words * 4, true);
      size_t wrong = 0, zero = 0, first = words;
      for (size_t i = 0; i < words; ++i) {
        if (back[i] == k::kPairMailEmpty) continue;
        ++wrong;
        if (back[i] == 0) ++zero;
        if (first == words) first = i;
      }
      std::printf("wiremin: card %d inbox after pair_wire_init: %zu of %zu words not kPairMailEmpty (%zu zero), first at word %zu\n", c, wrong, words, zero,
                  first);
    }
  }

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int parity = 0;
  size_t wrong_total = 0;
  std::vector<float> prev[kC];
  for (int call = 0; call < calls; ++call) {
    std::vector<float> h[kC];
    for (int c = 0; c < kC; ++c) {
      h[c].resize(elems);
      for (auto &v : h[c]) v = nd(rng) * (call + 1);
      f.compute(c).memcpy(part[c], h[c].data(), elems * 4).wait();
    }
    if (fresh) {
      for (int c = 0; c < kC; ++c) {
        f.compute(c).wait();
        f.release(inbox[c]);
        inbox[c] = static_cast<uint32_t *>(f.alloc_mailbox(c, words * 4));
        k::pair_wire_init(f.compute(c), kNMax, inbox[c]).wait();
      }
    }
    if (kfill) {
      for (int c = 0; c < kC; ++c) kfill_empty(f.compute(c), inbox[c], words);
      for (int c = 0; c < kC; ++c) f.compute(c).wait();
    }
    if (flushinit) {
      for (int c = 0; c < kC; ++c)
        f.compute(c)
            .submit([&](sycl::handler &hh) {
              hh.single_task([=]() SYCL_ESIMD_KERNEL {
                esimd::fence<esimd::memory_kind::global, esimd::fence_flush_op::clean, esimd::fence_scope::system>();
              });
            })
            .wait();
    }
    if (inboxfill) {
      const std::vector<uint32_t> nanw(words, 0x7fc00000u);
      for (int c = 0; c < kC; ++c) f.compute(c).memcpy(inbox[c], nanw.data(), words * 4).wait();
    }
    if (outfill) {
      std::vector<uint16_t> nanb(elems, 0xffffu);
      for (int c = 0; c < kC; ++c) f.compute(c).memcpy(out[c], nanb.data(), elems * 2).wait();
    }

    if (exact) {
      uint32_t *sums[kC];
      for (int c = 0; c < kC; ++c) sums[c] = static_cast<uint32_t *>(f.alloc(c, elems * 4));
      for (int i = 0; i < kC; ++i) {
        const int c = launch_rev ? 1 - i : i;
        exact_exchange(f.compute(c), part[c], sums[c], inbox[1 - c], inbox[c], err[c], rows, n, parity);
      }
      if (!parmode) parity ^= 1;
      std::printf("call %d parity %d:", call, parmode ? 0 : (call & 1));
      for (int c = 0; c < kC; ++c) {
        std::vector<float> got(elems);
        f.compute(c).memcpy(got.data(), sums[c], elems * 4).wait();
        size_t bad = 0, zero_contrib = 0, nan = 0;
        for (size_t e = 0; e < elems; ++e) {
          const float want = h[c][e] + h[1 - c][e];
          if (got[e] == want) continue;
          ++bad;
          if (got[e] == h[c][e]) ++zero_contrib;
          if (got[e] != got[e]) ++nan;
        }
        uint32_t ec = 0;
        f.compute(c).memcpy(&ec, err[c], 4).wait();
        wrong_total += bad;
        std::printf(" | c%d bad %zu (own-only %zu, NaN %zu) err %u", c, bad, zero_contrib, nan, ec);
      }
      std::printf("\n");
      for (int c = 0; c < kC; ++c) {
        f.compute(c).wait();
        f.release(sums[c]);
      }
      for (int c = 0; c < kC; ++c) prev[c] = h[c];
      continue;
    }
    if (recvonly) {
      const int cols = k::pair_cols(rows, n, b16 != 0);
      uint32_t *dump = static_cast<uint32_t *>(f.alloc(1, elems * 4));
      // both halves at once, as the wire runs them: rank 0 stores into rank 1's slot, rank 1 takes from it
      send_only(f.compute(0), part[0], inbox[1], rows, n, cols);
      recv_only(f.compute(1), inbox[1], dump, err[1], rows, n, cols);
      f.compute(0).wait();
      f.compute(1).wait();
      std::vector<uint32_t> seen(elems);
      f.copy(1).copy(seen.data(), dump, elems * 4, true);
      uint32_t e1 = 0;
      f.compute(1).memcpy(&e1, err[1], 4).wait();
      size_t bad = 0, zero = 0, empty = 0, first = elems;
      for (size_t e = 0; e < elems; ++e) {
        uint32_t want;
        std::memcpy(&want, &h[0][e], 4);
        if (seen[e] == want) continue;
        ++bad;
        if (seen[e] == 0) ++zero;
        if (seen[e] == k::kPairMailEmpty) ++empty;
        if (first == elems) first = e;
      }
      wrong_total += bad;
      std::printf("call %d: take-only: %zu of %zu words read differ from what was stored (%zu zero, %zu still empty), first at %zu, err %u\n", call, bad, elems,
                  zero, empty, first, e1);
      f.release(dump);
      continue;
    }
    if (sendonly) {
      // the store half only: rank 0 stores its partial into rank 1's slot; the host verifies rank 1's slot afterwards
      const int cols = k::pair_cols(rows, n, b16 != 0);
      send_only(f.compute(0), part[0], inbox[1], rows, n, cols);
      f.compute(0).wait();
      std::vector<uint32_t> got(elems);
      f.copy(1).copy(got.data(), inbox[1], elems * 4, true);
      const float *a = h[0].data();
      size_t bad = 0, zero = 0;
      for (size_t e = 0; e < elems; ++e) {
        float want = a[e];
        uint32_t w;
        std::memcpy(&w, &want, 4);
        if (got[e] == w) continue;
        ++bad;
        if (got[e] == 0) ++zero;
      }
      wrong_total += bad;
      std::printf("call %d: store-only verification: %zu of %zu words in the peer's slot differ from what was sent (%zu of them zero)\n", call, bad, elems, zero);
      continue;
    }

    for (int i = 0; i < kC; ++i) {
      const int c = launch_rev ? 1 - i : i;
      k::PairWire w{inbox[c], inbox[1 - c], err[c], 2800000000ull, b16 != 0};
      k::pair_reduce(f.compute(c), rows, n, part[c], &w, parity, out[c]);
    }
    if (!parmode) parity ^= 1;
    if (barrier) {
      f.compute(0).wait();
      f.compute(1).wait();
    }
    std::printf("call %d parity %d:\n", call, parmode ? 0 : (call & 1));
    for (int c = 0; c < kC; ++c) {
      std::vector<uint16_t> got(elems);
      f.compute(c).memcpy(got.data(), out[c], elems * 2).wait();
      size_t ok = 0, own = 0, peer = 0, nan = 0, other = 0, shown = 0;
      std::string map;
      for (size_t e = 0; e < elems; ++e) {
        const float a = b16 ? static_cast<float>(static_cast<bf16>(h[c][e])) : h[c][e];
        const float b = b16 ? static_cast<float>(static_cast<bf16>(h[1 - c][e])) : h[1 - c][e];
        const float g = decode(got[e]);
        char kind = '.';
        if (fb16(a + b) == got[e]) {
          ++ok;
        } else if (got[e] == kQNaN) {
          ++nan, kind = 'n';
        } else if (g == a) {
          ++own, kind = 'a';
        } else if (g == b) {
          ++peer, kind = 'b';
        } else {
          ++other, kind = '?';
          if (shown < 3) {
            ++shown;
            std::printf("    c%d e%zu (chunk %zu lane %zu): a %.9g b %.9g out %.9g want %.9g\n", c, e, e / static_cast<size_t>(k::pair_cols(rows, n, b16 != 0)),
                        e % static_cast<size_t>(k::pair_cols(rows, n, b16 != 0)), a, b, g, static_cast<float>(static_cast<bf16>(a + b)));
          }
        }
        if (kind != '.' && map.size() < 160) map += kind;
      }
      wrong_total += own + peer + nan + other;
      std::printf("  c%d ok %zu own %zu peer %zu nan %zu other %zu%s%s\n", c, ok, own, peer, nan, other, map.empty() ? "" : " lanes: ", map.c_str());
    }
    uint32_t ec[2] = {0, 0};
    for (int c = 0; c < kC; ++c) f.compute(c).memcpy(&ec[c], err[c], 4).wait();
    std::printf("  err %u/%u\n", ec[0], ec[1]);
    for (int c = 0; c < kC; ++c) prev[c] = h[c];
  }
  for (int c = 0; c < kC; ++c) {
    f.compute(c).wait();
    f.release(inbox[c]);
    f.release(part[c]);
    f.release(out[c]);
    f.release(err[c]);
  }
  if (wrong_total) {
    std::printf("wiremin: FAIL: %zu element(s) do not match the host sum\n", wrong_total);
    return 1;
  }
  std::printf("wiremin: PASS: every card's sum matches the host on every call\n");
  return 0;
}
