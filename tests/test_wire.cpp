// The two-card sums' bits on both cards for every row count and slot width.
// The four-card forms (test_quad: the mailbox; test_quad_split: the split sum) when four cards are visible.
// pair_reduce (1-8 rows) and pair_reduce_rows (9-63 rows) on the fp32 and bf16 wires, n 2,560 / 5,120 / 1,152 (the
// widths pair_cols picks, including a halved one), several calls back to back alternating the parity with new data each
// call: out == the host's bf16(p0 + p1) (fp32 wire) or bf16(bf16(p0) + bf16(p1)) (bf16 wire) on both cards, no timeout.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/rt/fabric.hpp"

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

static int g_checks = 0, g_fail = 0;
static void expect(bool ok, const std::string &what) {
  ++g_checks;
  if (!ok) ++g_fail;
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
}
static uint16_t bits(bf16 v) {
  uint16_t b;
  std::memcpy(&b, &v, 2);
  return b;
}
// XEN_WIRE_CALLS overrides a mailbox test's call count: the paths' repeat loops are where a partly visible or stale take
// shows up, so a stress run raises it (XEN_WIRE_CALLS=200). The copy paths (pair_split) ignore it. Unset or 0: default.
static int wire_calls(int def) {
  const char *s = std::getenv("XEN_WIRE_CALLS");
  const int v = s ? std::atoi(s) : 0;
  return v > 0 ? v : def;
}

// The four-card wires (WireQuad) on cards 0-3. Every card must write bf16((p0 + p1) + (p2 + p3))
// (fp32 wire) or bf16((bf16 p0 + bf16 p1) + (bf16 p2 + bf16 p3)) (bf16 wire): the host sums in that order through
// volatile temporaries (no reassociation). pair_reduce at 1-16 rows, pair_reduce_rows at 17-63, three calls each with the
// parity alternating; the int8 wire is refused.
static void test_quad(const xenithium::l0::Loader &loader, bool copy = false) {
  constexpr int kC = 4, kRowsMax = 64, kNMax = 5120;
  const int kCalls = wire_calls(3);
  xenithium::Fabric f(loader, {0, 1, 2, 3}, false);
  float *part[kC];
  bf16 *out[kC];
  uint32_t *err[kC];
  k::WireQuad dq, rq;  // the decode and the rows inboxes
  k::StepCopy copy_buffers;
  std::vector<void *> copy_allocations;
  if (copy) {
    f.alloc_step_copy(copy_buffers, kRowsMax, kNMax, false, copy_allocations);
    dq.copy = &copy_buffers;
  }
  const size_t words = k::pair_wire_words(kNMax, kC);
  const size_t row_words = k::pair_rows_wire_words(kRowsMax, kNMax, kC);
  for (int c = 0; c < kC; ++c) {
    auto &q = f.compute(c);
    part[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * kNMax * 4));
    const bool direct_out = copy && std::getenv("XEN_Q38_COPY_DIRECT_OUT");
    const size_t output_bytes = size_t(kRowsMax) * kNMax * sizeof(bf16);
    out[c] = static_cast<bf16 *>(direct_out ? f.alloc_mailbox(c, output_bytes) : f.alloc(c, output_bytes));
    if (direct_out) copy_buffers.outputs[c] = out[c];
    err[c] = static_cast<uint32_t *>(f.alloc(c, 4096));
    q.memset(err[c], 0, 64).wait();
    dq.inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(words * sizeof(uint32_t)));
    k::owner_mail_init(q, dq.inbox[c], words).wait();
    rq.inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(row_words * sizeof(uint32_t)));
    k::owner_mail_init(q, rq.inbox[c], row_words).wait();
  }
  constexpr uint64_t wait_cycles = 2800000000ull;
  if (copy) {
    // Load the eager kernels on every rank before measuring inter-rank waits.
    // Publish the fixture flags in advance so each warmup can run alone.
    const int warm_rows = copy_buffers.shared_ready ? copy_buffers.split_min_rows : 1;
    const size_t flag_bytes = 2 * kC * 64;
    for (int c = 0; c < kC; ++c) {
      auto &q = f.compute(c);
      q.memset(part[c], 0, size_t(warm_rows) * 2560 * sizeof(float)).wait();
      q.memset(copy_buffers.recv[c], 0, kC * copy_buffers.stride * sizeof(float)).wait();
      if (copy_buffers.shared_ready) q.memset(copy_buffers.shared[c], 0, copy_buffers.stride * sizeof(bf16)).wait();
    }
    auto warm = [&](int rows) {
      for (int c = 0; c < kC; ++c) {
        copy_buffers.ready[c * 16] = 1;
        if (copy_buffers.shared_ready) copy_buffers.shared_ready[c * 16] = 1;
      }
      for (int c = 0; c < kC; ++c) {
        auto &q = f.compute(c);
        k::step_copy_reduce(q, rows, 2560, part[c], copy_buffers, c, 0, err[c], wait_cycles, out[c]).wait_and_throw();
        q.memset(copy_buffers.state[c], 0, 64).wait_and_throw();
      }
      std::memset(copy_buffers.ready, 0, flag_bytes);
      if (copy_buffers.shared_ready) std::memset(copy_buffers.shared_ready, 0, flag_bytes);
    };
    warm(1);
    if (copy_buffers.shared_ready) warm(warm_rows);
  }
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  int parity = 0;
  for (const bool b16 : (copy ? std::vector<bool>{false} : std::vector<bool>{false, true}))
    for (const int n : {2560, 5120, 1152})
      for (const int rows : {1, 2, 3, 4, 5, 8, 12, 16, 17, 40, 63}) {
        if (copy)
          for (int c = 0; c < kC; ++c) f.compute(c).memset(err[c], 0, sizeof(uint32_t)).wait();
        size_t bad = 0, differ = 0;
        for (int call = 0; call < kCalls; ++call) {
          std::vector<float> h[kC];
          for (int c = 0; c < kC; ++c) {
            h[c].resize(size_t(rows) * n);
            for (auto &v : h[c]) v = nd(rng) * (call + 1);
            f.compute(c).memcpy(part[c], h[c].data(), h[c].size() * 4).wait();
          }
          for (int c = 0; c < kC; ++c) {
            if (rows <= k::kPairRows) {
              k::PairWire w{nullptr, nullptr, err[c], wait_cycles, b16};
              w.quad = dq, w.quad.card = c;
              k::pair_reduce(f.compute(c), rows, n, part[c], &w, parity, out[c]);
            } else if (copy) {
              k::step_copy_reduce(f.compute(c), rows, n, part[c], copy_buffers, c, parity, err[c], wait_cycles, out[c]);
            } else {
              k::PairRowsWire w;
              w.error = err[c], w.rows_max = kRowsMax, w.n_max = kNMax, w.bf16 = b16;
              w.quad = rq, w.quad.card = c;
              k::pair_reduce_rows(f.compute(c), rows, n, part[c], &w, parity, out[c]);
            }
          }
          parity ^= 1;
          std::vector<uint16_t> got[kC];
          for (int c = 0; c < kC; ++c) {
            got[c].resize(size_t(rows) * n);
            f.compute(c).memcpy(got[c].data(), out[c], got[c].size() * 2).wait();
          }
          for (size_t e = 0; e < got[0].size(); ++e) {
            float v[kC];
            for (int c = 0; c < kC; ++c) v[c] = b16 ? static_cast<float>(bf16(h[c][e])) : h[c][e];
            volatile float lo = v[0] + v[1], hi = v[2] + v[3];
            volatile float sum = lo + hi;
            for (int c = 0; c < kC; ++c) bad += bits(bf16(static_cast<float>(sum))) != got[c][e], differ += got[c][e] != got[0][e];
          }
        }
        uint32_t e = 0;
        for (int c = 0; c < kC; ++c) {
          uint32_t x = 0;
          f.compute(c).memcpy(&x, err[c], 4).wait();
          e |= x;
        }
        expect(bad == 0 && differ == 0 && e == 0, std::string(rows <= k::kPairRows ? "pair_reduce" : "pair_reduce_rows") + " four cards, " + (b16 ? "bf16" : "fp32") +
                                                      " wire, n " + std::to_string(n) + ", " + std::to_string(rows) + " rows, " + std::to_string(kCalls) +
                                                      " calls: every card == host (p0 + p1) + (p2 + p3)" + (bad ? " (" + std::to_string(bad) + " differ)" : "") +
                                                      (differ ? " (cards disagree in " + std::to_string(differ) + ")" : "") + (e ? " TIMEOUT" : ""));
      }
  bool refused = false;
  try {
    k::PairWire w{nullptr, nullptr, err[0], 2800000000ull, true};
    w.i8 = true;
    w.quad = dq, w.quad.card = 0;
    k::pair_reduce(f.compute(0), 1, 2560, part[0], &w, 0, out[0]);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  expect(refused, "pair_reduce four cards: the int8 wire is refused");
  for (int c = 0; c < kC; ++c) {
    f.release(dq.inbox[c]), f.release(rq.inbox[c]);
    f.release(part[c]), f.release(out[c]), f.release(err[c]);
  }
  for (void *ptr : copy_allocations) f.release(ptr);
}

// The split four-card sum (PairSplit:cards 4): card c owns rows [m c / 4, m (c + 1) / 4), the
// others' rows arrive by copies, the owner's rows go back by copies. Every card must hold the four-card mailbox's bits:
// bf16((p0 + p1) + (p2 + p3)) (bf16 wire: bf16 operands), with add bf16(p x scale + that sum) (fused or not: either
// rounding accepted). Two paths: pair_split_rows (copies on the compute queue) and the producers' form (pair_split_send
// on a side queue, pair_split_finish without the rows' wait, then pair_split_wait)., a third path: the
// sum left with its owner (SplitRows:owner: a card's other rows of out stay as they were), then pair_split_share of out
// + pair_split_wait: every card holds every row again.
static void test_quad_split(const xenithium::l0::Loader &loader) {
  constexpr int kC = 4, kRowsMax = 1040, kNMax = 2560, kCalls = 2;
  xenithium::Fabric f(loader, {0, 1, 2, 3}, false);
  float *part[kC], *addp[kC], *adds[kC];
  bf16 *out[kC];
  k::PairSplit sp[kC];
  std::vector<sycl::queue> side;
  for (int c = 0; c < kC; ++c) side.emplace_back(f.card_context(c), f.device(c), xenithium::Fabric::queue_props(false));
  for (int c = 0; c < kC; ++c) {
    auto &q = f.compute(c);
    part[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * kNMax * 4));
    addp[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * kNMax * 4));
    adds[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * 4));
    out[c] = static_cast<bf16 *>(f.alloc(c, size_t(kRowsMax) * kNMax * 2));
    k::PairSplit &s = sp[c];
    s.cards = kC, s.card = c, s.rows_max = kRowsMax, s.n_max = kNMax;
    s.recv = static_cast<float *>(f.alloc(c, k::pair_split_recv_elems(kRowsMax, kNMax, kC) * 4));
    s.flags = static_cast<uint32_t *>(f.alloc_mailbox(c, k::kPairSplitFlagBytes));
    q.memset(s.flags, 0, k::kPairSplitFlagBytes).wait();
    s.error = static_cast<uint32_t *>(f.alloc(c, 4096));
    q.memset(s.error, 0, 64).wait();
    s.stage = static_cast<bf16 *>(f.alloc(c, k::pair_split_stage_elems(kRowsMax, kNMax, kC) * 2));
    auto *tab = static_cast<uint32_t *>(f.alloc(c, k::kPairSplitSeqTabBytes));
    k::pair_split_seq_tab_init(q, tab).wait();
    s.seq_tab = tab;
  }
  for (int c = 0; c < kC; ++c)
    for (int d = 0; d < kC; ++d) sp[c].recv_of[d] = sp[d].recv, sp[c].flags_of[d] = sp[d].flags;
  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.f, 1.f);
  uint32_t seq = 0;
  struct Mode {
    bool b16, add;
    const char *name;
  };
  for (const Mode md : {Mode{false, false, "fp32"}, Mode{true, false, "bf16"}, Mode{false, true, "fp32 + add"}})
    for (const int n : {2560, 1152})
      for (const int rows : {24, 61, 256, 1031})
        for (const int path : {0, 1, 2}) {  // pair_split_rows; send + finish + wait; send + finish (owner) + share + wait
          const bool producers = path >= 1, owner = path == 2;
          if (md.b16 && n % 256) continue;  // the bf16 wire takes n % 256 == 0
          constexpr uint16_t kPoison = 0x7fc1;  // what the rows no copy may touch hold before an owner-only sum
          size_t bad = 0, differ = 0, touched = 0;
          for (int call = 0; call < kCalls; ++call) {
            std::vector<float> h[kC], ap[kC], as[kC];
            for (int c = 0; c < kC; ++c) {
              h[c].resize(size_t(rows) * n);
              for (auto &v : h[c]) v = nd(rng) * (call + 1);
              f.compute(c).memcpy(part[c], h[c].data(), h[c].size() * 4).wait();
            }
            // the add term is the same on every card (each card's own copy of the shared expert)
            ap[0].resize(size_t(rows) * n), as[0].resize(size_t(rows));
            for (auto &v : ap[0]) v = nd(rng);
            for (auto &v : as[0]) v = nd(rng);
            for (int c = 0; c < kC; ++c) {
              f.compute(c).memcpy(addp[c], ap[0].data(), ap[0].size() * 4).wait();
              f.compute(c).memcpy(adds[c], as[0].data(), as[0].size() * 4).wait();
            }
            seq = k::pair_split_next_seq(seq);
            sycl::queue *qs[kC], *sd[kC];
            const float *pp[kC];
            bf16 *oo[kC];
            k::PairSplit ss[kC];
            k::PairSplitAdd ad[kC];
            for (int c = 0; c < kC; ++c) {
              qs[c] = &f.compute(c), sd[c] = &side[static_cast<size_t>(c)], pp[c] = part[c], oo[c] = out[c];
              ss[c] = sp[c], ss[c].bf16 = md.b16;
              ad[c] = k::PairSplitAdd{addp[c], adds[c]};
            }
            if (!producers) {
              k::pair_split_rows(qs, rows, n, pp, oo, ss, seq, md.add ? ad : nullptr);
            } else if (!owner) {
              for (int c = 0; c < kC; ++c) k::pair_split_send(*qs[c], sd[c], c, rows, n, pp[c], ss[c], seq);
              k::pair_split_finish(qs, sd, rows, n, pp, oo, ss, seq, md.add ? ad : nullptr, k::SplitRows::deferred);
              for (int c = 0; c < kC; ++c) k::pair_split_wait(*qs[c], ss[c], seq);
            } else {
              const std::vector<uint16_t> poison(size_t(rows) * n, kPoison);
              for (int c = 0; c < kC; ++c) f.compute(c).memcpy(out[c], poison.data(), poison.size() * 2).wait();
              for (int c = 0; c < kC; ++c) k::pair_split_send(*qs[c], sd[c], c, rows, n, pp[c], ss[c], seq);
              k::pair_split_finish(qs, sd, rows, n, pp, oo, ss, seq, md.add ? ad : nullptr, k::SplitRows::owner);
              for (int c = 0; c < kC; ++c) {  // the owner's rows only: every other row is still the poison
                f.compute(c).wait_and_throw();
                side[static_cast<size_t>(c)].wait_and_throw();
                std::vector<uint16_t> now(poison.size());
                f.compute(c).memcpy(now.data(), out[c], now.size() * 2).wait();
                int lo = 0, hi = 0;
                k::pair_split_owner_rows(c, rows, &lo, &hi, kC);
                for (int r = 0; r < rows; ++r) {
                  if (r >= lo && r < hi) continue;
                  for (int e = 0; e < n; ++e) touched += now[size_t(r) * n + e] != kPoison;
                }
              }
              seq = k::pair_split_next_seq(seq);
              void *shared[kC];
              for (int c = 0; c < kC; ++c) shared[c] = out[c];
              const k::SplitShare sets[] = {{shared, size_t(n) * 2}};
              for (int c = 0; c < kC; ++c) k::pair_split_share(*qs[c], sd[c], c, rows, sets, ss[c], seq);
              for (int c = 0; c < kC; ++c) k::pair_split_wait(*qs[c], ss[c], seq);
            }
            std::vector<uint16_t> got[kC];
            for (int c = 0; c < kC; ++c) {
              f.compute(c).wait_and_throw();
              got[c].resize(size_t(rows) * n);
              f.compute(c).memcpy(got[c].data(), out[c], got[c].size() * 2).wait();
            }
            for (size_t e = 0; e < got[0].size(); ++e) {
              float v[kC];
              for (int c = 0; c < kC; ++c) v[c] = md.b16 ? static_cast<float>(bf16(h[c][e])) : h[c][e];
              volatile float lo = v[0] + v[1], hi = v[2] + v[3];
              volatile float sum = lo + hi;
              uint16_t want = bits(bf16(static_cast<float>(sum))), want2 = want;
              if (md.add) {
                const float pv = ap[0][e], sc = as[0][e / static_cast<size_t>(n)];
                volatile float prod = pv * sc;
                volatile float unfused = prod + sum;
                want = bits(bf16(static_cast<float>(unfused)));
                want2 = bits(bf16(std::fma(pv, sc, static_cast<float>(sum))));
              }
              for (int c = 0; c < kC; ++c) bad += got[c][e] != want && got[c][e] != want2, differ += got[c][e] != got[0][e];
            }
          }
          uint32_t e = 0;
          for (int c = 0; c < kC; ++c) {
            uint32_t x = 0;
            f.compute(c).memcpy(&x, sp[c].error, 4).wait();
            e |= x;
          }
          static const char *const kPaths[] = {"pair_split_rows", "send + finish + wait, side queues", "send + finish (owner rows) + share + wait"};
          expect(bad == 0 && differ == 0 && touched == 0 && e == 0, std::string("pair_split four cards (") + kPaths[path] +
                                                        "), " + md.name + ", n " + std::to_string(n) + ", " + std::to_string(rows) + " rows, " + std::to_string(kCalls) +
                                                        " calls: every card == host (p0 + p1) + (p2 + p3)" + (bad ? " (" + std::to_string(bad) + " differ)" : "") +
                                                        (differ ? " (cards disagree in " + std::to_string(differ) + ")" : "") +
                                                        (touched ? " (an owner-only sum wrote " + std::to_string(touched) + " words of other owners' rows)" : "") +
                                                        (e ? " TIMEOUT" : ""));
        }
  for (int c = 0; c < kC; ++c) {
    f.compute(c).wait();
    side[static_cast<size_t>(c)].wait();
  }
}

// A decode step's rows by owner. pair_reduce on an owner wire: card c's own rows (owner_rows) of out ==
// the four-card mailbox's bits, bf16((p0 + p1) + (p2 + p3)); its other rows stay as they were (a sentinel). owner_share
// of out: every card then holds every owner's rows. Calls back to back with new data, the parity alternating, then all
// on one parity (as the model's once-a-step mailboxes use it); row counts with unequal owners (5, 9, 30); n 2,560 and
// 10,240 (the widest items), 1,152 (the share's narrowest) and 1,088 (the sum's narrowest; no share: n % 128). Row
// counts under kOwnerMinRows, the bf16 / int8 wires and an incomplete mailbox are refused.
static void test_owner(const xenithium::l0::Loader &loader) {
  constexpr int kC = 4, kNMax = 10240;
  const int kCalls = wire_calls(3);
  constexpr uint16_t kSentinel = 0x1234;
  xenithium::Fabric f(loader, {0, 1, 2, 3}, false);
  float *part[kC];
  bf16 *out[kC];
  uint32_t *err[kC];
  k::OwnerMail sum_mail, share_mail;
  for (int c = 0; c < kC; ++c) {
    auto &q = f.compute(c);
    part[c] = static_cast<float *>(f.alloc(c, size_t(k::kPairRows) * kNMax * 4));
    out[c] = static_cast<bf16 *>(f.alloc(c, size_t(k::kPairRows) * kNMax * 2));
    err[c] = static_cast<uint32_t *>(f.alloc(c, 4096));
    q.memset(err[c], 0, 64).wait();
    sum_mail.inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::owner_sum_words(kNMax) * 4));
    k::owner_mail_init(q, sum_mail.inbox[c], k::owner_sum_words(kNMax)).wait();
    share_mail.inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::owner_share_words(kNMax) * 4));
    k::owner_mail_init(q, share_mail.inbox[c], k::owner_share_words(kNMax)).wait();
  }
  auto sum_wire = [&](int c) {
    k::PairWire w;
    w.error = err[c];
    w.owner = sum_mail, w.owner.card = c;
    return w;
  };
  auto share_of = [&](int c) {
    k::OwnerShare s;
    s.mail = share_mail, s.mail.card = c;
    s.error = err[c];
    return s;
  };
  bool inverse = true;  // owner_rows and row_owner agree for every row count
  for (int rows = k::kOwnerMinRows; rows <= k::kPairRows; ++rows)
    for (int c = 0; c < kC; ++c) {
      const k::RowRange own = k::owner_rows(c, rows);
      inverse = inverse && own.rows() >= 1 && own.rows() <= k::kOwnerRowsMax;
      for (int r = own.lo; r < own.hi; ++r) inverse = inverse && k::row_owner(r, rows) == c;
    }
  expect(inverse, "owner_rows / row_owner: every card owns 1.." + std::to_string(k::kOwnerRowsMax) + " rows of 4.." + std::to_string(k::kPairRows) +
                      ", and each row's owner is the card whose range holds it");
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.f, 1.f);
  int parity = 0;
  for (const bool same_parity : {false, true})
    for (const int n : {2560, 10240, 1152, 1088})
      for (const int rows : {4, 5, 8, 9, 16, 17, 30, 32}) {
        size_t bad_sum = 0, touched = 0, bad_share = 0;
        for (int call = 0; call < kCalls; ++call) {
          std::vector<float> h[kC];
          const std::vector<uint16_t> sentinel(size_t(rows) * n, kSentinel);
          for (int c = 0; c < kC; ++c) {
            h[c].resize(size_t(rows) * n);
            for (auto &v : h[c]) v = nd(rng) * (call + 1);
            f.compute(c).memcpy(part[c], h[c].data(), h[c].size() * 4).wait();
            f.compute(c).memcpy(out[c], sentinel.data(), sentinel.size() * 2).wait();
          }
          const int p = same_parity ? 0 : parity;
          for (int c = 0; c < kC; ++c) {
            const k::PairWire w = sum_wire(c);
            k::pair_reduce(f.compute(c), rows, n, part[c], &w, p, out[c]);
          }
          std::vector<uint16_t> summed[kC], shared[kC];
          for (int c = 0; c < kC; ++c) {
            summed[c].resize(size_t(rows) * n);
            f.compute(c).memcpy(summed[c].data(), out[c], summed[c].size() * 2).wait();
          }
          if (n % 128 == 0) {
            for (int c = 0; c < kC; ++c) k::owner_share(f.compute(c), rows, n, out[c], share_of(c), p);
            for (int c = 0; c < kC; ++c) {
              shared[c].resize(size_t(rows) * n);
              f.compute(c).memcpy(shared[c].data(), out[c], shared[c].size() * 2).wait();
            }
          }
          parity ^= 1;
          for (int r = 0; r < rows; ++r)
            for (int x = 0; x < n; ++x) {
              const size_t e = size_t(r) * n + x;
              volatile float lo = h[0][e] + h[1][e], hi = h[2][e] + h[3][e];
              volatile float sum = lo + hi;
              const uint16_t want = bits(bf16(static_cast<float>(sum)));
              for (int c = 0; c < kC; ++c) {
                if (k::row_owner(r, rows) == c) bad_sum += summed[c][e] != want;
                else touched += summed[c][e] != kSentinel;
                if (!shared[c].empty()) bad_share += shared[c][e] != want;
              }
            }
        }
        uint32_t e = 0;
        for (int c = 0; c < kC; ++c) {
          uint32_t x = 0;
          f.compute(c).memcpy(&x, err[c], 4).wait();
          e |= x;
        }
        expect(bad_sum == 0 && touched == 0 && bad_share == 0 && e == 0,
               std::string("owner sum") + (n % 128 == 0 ? " + owner_share" : "") + ", n " + std::to_string(n) + ", " + std::to_string(rows) + " rows, " +
                   std::to_string(kCalls) + " calls" + (same_parity ? " on one parity" : "") +
                   ": own rows == host (p0 + p1) + (p2 + p3), other rows untouched" + (n % 128 == 0 ? ", every row on every card after the share" : "") +
                   (bad_sum ? " (" + std::to_string(bad_sum) + " sum elements differ)" : "") + (touched ? " (" + std::to_string(touched) + " foreign elements written)" : "") +
                   (bad_share ? " (" + std::to_string(bad_share) + " shared elements differ)" : "") + (e ? " TIMEOUT" : ""));
      }
  auto refuses = [&](auto &&call) {
    try {
      call();
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  expect(refuses([&] {
           const k::PairWire w = sum_wire(0);
           k::pair_reduce(f.compute(0), k::kOwnerMinRows - 1, 2560, part[0], &w, 0, out[0]);
         }) && refuses([&] { k::owner_share(f.compute(0), k::kOwnerMinRows - 1, 2560, out[0], share_of(0), 0); }),
         "owner sum / owner_share: fewer rows than cards are refused");
  expect(refuses([&] {
           k::PairWire w = sum_wire(0);
           w.bf16 = true;
           k::pair_reduce(f.compute(0), 8, 2560, part[0], &w, 0, out[0]);
         }) && refuses([&] {
           k::PairWire w = sum_wire(0);
           w.i8 = true;
           k::pair_reduce(f.compute(0), 8, 2560, part[0], &w, 0, out[0]);
         }),
         "owner sum: the bf16 and the int8 wire are refused");
  expect(refuses([&] {
           k::PairWire w = sum_wire(0);
           w.owner.inbox[2] = nullptr;
           k::pair_reduce(f.compute(0), 8, 2560, part[0], &w, 0, out[0]);
         }) && refuses([&] {
           k::OwnerShare s = share_of(0);
           s.mail.inbox[3] = nullptr;
           k::owner_share(f.compute(0), 8, 2560, out[0], s, 0);
         }),
         "owner sum / owner_share: a mailbox without every card's inbox is refused");
  for (int c = 0; c < kC; ++c) {
    f.compute(c).wait();
    f.release(sum_mail.inbox[c]), f.release(share_mail.inbox[c]);
    f.release(part[c]), f.release(out[c]), f.release(err[c]);
  }
}

int main() {
  const auto loader = xenithium::l0::Loader::open();
  if (std::getenv("XEN_TEST_COPY_WIRE")) {
    if (loader.size() < 4) {
      std::printf("[SKIP] copy wire requires four cards\n");
      return 77;
    }
    test_quad(loader, true);
    std::printf("copy wire: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
  }
  xenithium::Fabric f(loader, {0, 1}, false);
  constexpr int kRowsMax = 64, kNMax = 5120;
  const int kCalls = wire_calls(3);
  float *part[2];
  bf16 *out[2];
  uint32_t *inbox[2], *rinbox[2], *err[2];
  for (int c = 0; c < 2; ++c) {
    auto &q = f.compute(c);
    part[c] = static_cast<float *>(f.alloc(c, size_t(kRowsMax) * kNMax * 4));
    out[c] = static_cast<bf16 *>(f.alloc(c, size_t(kRowsMax) * kNMax * 2));
    err[c] = static_cast<uint32_t *>(f.alloc(c, 64));
    q.memset(err[c], 0, 64).wait();
    inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_wire_words(kNMax) * 4));
    k::pair_wire_init(q, kNMax, inbox[c]).wait();
    rinbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_rows_wire_words(kRowsMax, kNMax) * 4));
    k::pair_rows_wire_init(q, kRowsMax, kNMax, rinbox[c]).wait();
  }
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  int parity = 0;
  for (const bool b16 : {false, true})
    for (const int n : {2560, 5120, 1152})
      for (const int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 23, 40, 63}) {
        size_t bad = 0;
        for (int call = 0; call < kCalls; ++call) {
          std::vector<float> h[2];
          for (int c = 0; c < 2; ++c) {
            h[c].resize(size_t(rows) * n);
            for (auto &v : h[c]) v = nd(rng) * (call + 1);
            f.compute(c).memcpy(part[c], h[c].data(), h[c].size() * 4).wait();
          }
          for (int c = 0; c < 2; ++c) {
            if (rows <= k::kPairRows) {  // the decode wire: up to a step's rows
              k::PairWire w{inbox[c], inbox[1 - c], err[c], 2800000000ull, b16};
              k::pair_reduce(f.compute(c), rows, n, part[c], &w, parity, out[c]);
            } else {
              k::PairRowsWire w;
              w.inbox = rinbox[c], w.peer_inbox = rinbox[1 - c], w.error = err[c], w.rows_max = kRowsMax, w.n_max = kNMax, w.bf16 = b16;
              k::pair_reduce_rows(f.compute(c), rows, n, part[c], &w, parity, out[c]);
            }
          }
          parity ^= 1;
          for (int c = 0; c < 2; ++c) {
            std::vector<uint16_t> got(size_t(rows) * n);
            f.compute(c).memcpy(got.data(), out[c], got.size() * 2).wait();
            for (size_t e = 0; e < got.size(); ++e) {
              const float a = b16 ? static_cast<float>(bf16(h[0][e])) : h[0][e], b = b16 ? static_cast<float>(bf16(h[1][e])) : h[1][e];
              bad += bits(bf16(a + b)) != got[e];
            }
          }
        }
        uint32_t e[2];
        for (int c = 0; c < 2; ++c) f.compute(c).memcpy(&e[c], err[c], 4).wait();
        expect(bad == 0 && e[0] == 0 && e[1] == 0, std::string(rows <= k::kPairRows ? "pair_reduce" : "pair_reduce_rows") + " " + (b16 ? "bf16" : "fp32") + " wire, n " +
                                                       std::to_string(n) + ", " + std::to_string(rows) + " rows (" + std::to_string(k::pair_cols(rows, n, b16)) +
                                                       " columns per item), " + std::to_string(kCalls) + " calls: both cards == host" +
                                                       (bad ? " (" + std::to_string(bad) + " differ)" : "") + (e[0] | e[1] ? " TIMEOUT" : ""));
      }
  // The int8 wire (PairWire:i8, pair_reduce at 1-32 rows). Both cards write the same bits; an
  // element is the sum of the two partials each rounded to 8-bit codes of its 32-column group (scale s = bf16(absmax /
  // 127)): |out - (p0 + p1)| <= (s0 + s1) / 2 + the bf16 rounding of the sum; the sum of two all-zero groups is 0; an
  // outlier column does not disturb another group.
  for (const int n : {2560, 5120, 1152})
    for (const int rows : {1, 2, 5, 8, 9, 16, 20, 32}) {
      size_t differ = 0, over = 0;
      double se = 0, sv = 0;
      for (int call = 0; call < kCalls; ++call) {
        std::vector<float> h[2];
        for (int c = 0; c < 2; ++c) {
          h[c].resize(size_t(rows) * n);
          for (auto &v : h[c]) v = nd(rng) * (call + 1);
          for (size_t e = 0; e < h[c].size(); e += 97) h[c][e] *= 40.f;  // outlier columns
          for (int e = 0; e < 64 && call == 1; ++e) h[c][size_t(64) + e] = 0.f;  // two all-zero groups (both cards)
          f.compute(c).memcpy(part[c], h[c].data(), h[c].size() * 4).wait();
        }
        for (int c = 0; c < 2; ++c) {
          k::PairWire w{inbox[c], inbox[1 - c], err[c], 2800000000ull, true};
          w.i8 = true;
          k::pair_reduce(f.compute(c), rows, n, part[c], &w, parity, out[c]);
        }
        parity ^= 1;
        std::vector<uint16_t> got[2];
        for (int c = 0; c < 2; ++c) {
          got[c].resize(size_t(rows) * n);
          f.compute(c).memcpy(got[c].data(), out[c], got[c].size() * 2).wait();
        }
        for (size_t g0 = 0; g0 < got[0].size(); g0 += 32) {
          float s[2];
          for (int c = 0; c < 2; ++c) {
            float mx = 0;
            for (int i = 0; i < 32; ++i) mx = std::max(mx, std::fabs(h[c][g0 + i]));
            s[c] = static_cast<float>(bf16(mx * (1.f / 127.f)));
          }
          for (int i = 0; i < 32; ++i) {
            differ += got[0][g0 + i] != got[1][g0 + i];
            bf16 o;
            std::memcpy(&o, &got[0][g0 + i], 2);
            const double ex = static_cast<double>(h[0][g0 + i]) + h[1][g0 + i], d = std::fabs(static_cast<double>(static_cast<float>(o)) - ex);
            over += !(d <= 0.5 * (static_cast<double>(s[0]) + s[1]) * 1.02 + std::fabs(ex) / 128.0 + 1e-30);
            se += d * d, sv += ex * ex;
          }
        }
      }
      uint32_t e[2];
      for (int c = 0; c < 2; ++c) f.compute(c).memcpy(&e[c], err[c], 4).wait();
      char rel[32];
      std::snprintf(rel, sizeof rel, "%.4f", std::sqrt(se / sv));
      expect(differ == 0 && over == 0 && e[0] == 0 && e[1] == 0, "pair_reduce int8 wire, n " + std::to_string(n) + ", " + std::to_string(rows) + " rows (" +
                                                                    std::to_string(k::pair_cols_i8(rows, n)) + " columns per item), " + std::to_string(kCalls) +
                                                                    " calls: both cards equal, every element within its groups' half steps (rel rms err " + rel + ")" +
                                                                    (differ ? " (" + std::to_string(differ) + " words differ between the cards)" : "") +
                                                                    (over ? " (" + std::to_string(over) + " over the bound)" : "") + (e[0] | e[1] ? " TIMEOUT" : ""));
    }
  if (loader.size() >= 4) test_quad(loader), test_quad_split(loader), test_owner(loader);
  else std::printf("[SKIP] four-card wires: %zu card(s) visible\n", loader.size());
  std::printf("wire: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
