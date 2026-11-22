// Gate for the Gated DeltaNet decode block (kernels/linear_attention/gdn.hpp): projections, conv window, gates,
// delta-rule recurrence, gated norm, out_proj and the two-card sum; and the speculative-row
// protocol (rows are pending until the next step's accept count).
// Synthetic weights at the model's dims (hidden 2560, 16 key / 48 value heads of 128) and at a
// small shape. The reference is teacher-forced at the projections: it consumes the device's
// bf16 projection values for every token of the history, so what is compared is the conv,
// recurrence and norm arithmetic (then out_proj from the device's y). A free-running reference
// only gets a magnitude bound: one bf16 flip of a projection moves everything after it by a
// fraction of an ulp.
// y: within 2^-6 max(|ref|, head rms), <= 0.5 % of elements off. y is rounded twice
// in a chain (o = bf16(S q), then y), so a flip of o plus y's own rounding is 2 ulps.
// Not ulps of y alone either: o = S q can cancel to a small value whose fp32
// sum-order error is many of its own ulps.
// S: max |dS| <= 1e-4 max |S| (fp32 state vs a double reference)
// out: within 2^-7 max(|ref|, row rms), <= 1 % off, from the device's y
// protocol: "R rows then accept c" == "c rows then the next step", bit for bit
// two cards: bit-identical outputs; each card's y equals the one-card y of its heads; out
// vs the sum of the per-card references from the device's y (each card quantises
// its own out_proj slice, so a one-card run is a different model: no comparison)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/linear_attention/gdn_rows.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
float bfr(float v) { return k::bf16_value(k::bf16_bits(v)); }

struct BoundCmp {
  double scaled = 0;
  size_t off = 0, n = 0;
  void add_row(const float *dev, const double *ref, int len) {
    double ss = 0;
    for (int i = 0; i < len; ++i) ss += ref[i] * ref[i];
    const double rms = std::sqrt(ss / len);
    for (int i = 0; i < len; ++i) {
      ++n;
      off += dev[i] != bfr(static_cast<float>(ref[i]));
      scaled = std::max(scaled, std::fabs(dev[i] - ref[i]) / std::max({std::fabs(ref[i]), rms, 1e-30}));
    }
  }
  bool ok() const { return scaled <= 1.0 / 128 && off * 100 <= n; }
  std::string text() const {
    char b[128];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms), %zu / %zu differ (%.3f %%)", scaled, off, n, 100.0 * off / std::max<size_t>(1, n));
    return b;
  }
};

// ---- the whole model's weights (bf16 values) and one card's slice ----
struct Model {
  k::GdnShape s;  // cards = 1
  std::vector<float> qkv, z, a, b, out, conv, a_log, dt_bias, norm;
};
Model make_model(int H, int KH, int VH, uint32_t seed) {
  Model m;
  m.s = k::GdnShape{H, KH, VH, 128, 128, 1, 0, 1e-6f};
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  const int CC = m.s.conv_ch(), VW = m.s.val_w();
  auto fill = [&](std::vector<float> &v, size_t n, float sd) {
    v.resize(n);
    for (auto &x : v) x = bfr(nd(rng) * sd);
  };
  const float sw = 1.f / std::sqrt(static_cast<float>(H));
  fill(m.qkv, static_cast<size_t>(CC) * H, sw);
  fill(m.z, static_cast<size_t>(VW) * H, sw);
  fill(m.a, static_cast<size_t>(VH) * H, sw);
  fill(m.b, static_cast<size_t>(VH) * H, sw);
  fill(m.out, static_cast<size_t>(H) * VW, 1.f / std::sqrt(static_cast<float>(VW)));
  fill(m.conv, static_cast<size_t>(CC) * 4, 0.4f);
  m.a_log.resize(VH);
  m.dt_bias.resize(VH);
  for (int h = 0; h < VH; ++h) {
    m.a_log[h] = bfr(std::log(1.f + 15.f * ud(rng)));
    m.dt_bias[h] = bfr(-3.f + 4.f * ud(rng));
  }
  m.norm.resize(128);
  for (auto &v : m.norm) v = bfr(0.5f + ud(rng));
  return m;
}

// One card's buffers: weights, state, scratch.
struct Card {
  sycl::queue *q = nullptr;
  k::GdnShape s;
  k::GdnWeights w;
  k::GdnRefWeights ref;
  k::GdnState st;
  k::GdnScratch sc;
  k::PairWire wire;
  int32_t *accept = nullptr;
  bf16 *x = nullptr, *out = nullptr;
  std::vector<void *> mem;
};
template <typename T>
T *up(xenithium::Fabric &f, int c, const std::vector<T> &v, std::vector<void *> &mem) {
  auto *p = static_cast<T *>(f.alloc(c, std::max<size_t>(64, v.size() * sizeof(T))));
  f.compute(c).memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  mem.push_back(p);
  return p;
}
std::vector<uint16_t> bits(const std::vector<float> &v) {
  std::vector<uint16_t> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = k::bf16_bits(v[i]);
  return b;
}
// int8 matrix [n][kk] -> device weights + dequantised reference (rows padded to 16).
k::DenseWeights i8_matrix(xenithium::Fabric &f, int c, const std::vector<float> &w, int n, int kk, std::vector<float> &deq, std::vector<void *> &mem) {
  const int np = (n + 15) / 16 * 16;
  std::vector<uint16_t> wb(static_cast<size_t>(np) * kk, 0);
  for (size_t i = 0; i < w.size(); ++i) wb[i] = k::bf16_bits(w[i]);
  std::vector<int8_t> codes(wb.size());
  std::vector<float> sc(np);
  k::quantize_i8_rows(wb.data(), np, kk, codes.data(), sc.data(), 1);
  for (int r = n; r < np; ++r) sc[r] = 1.f;
  deq.resize(static_cast<size_t>(n) * kk);
  for (int r = 0; r < n; ++r)
    for (int j = 0; j < kk; ++j) deq[static_cast<size_t>(r) * kk + j] = codes[static_cast<size_t>(r) * kk + j] * sc[r];
  std::vector<int8_t> packed(k::dense_packed_elems(np, kk));
  k::pack_dense_i8(codes.data(), np, kk, packed.data());
  return k::DenseWeights{up(f, c, packed, mem), up(f, c, sc, mem), k::DenseKind::i8, np, kk};
}

Card make_card(xenithium::Fabric &f, int c, const Model &m, int cards) {
  Card d;
  d.q = &f.compute(c);
  d.s = m.s;
  d.s.cards = cards;
  d.s.card = c;
  const auto &s = d.s;
  const int H = s.hidden, D = 128, KWa = m.s.key_w(), kh0 = c * s.kh(), vh0 = c * s.vh();
  // q | k | v | z rows of this card, conv rows q | k | v, a | b, out_proj columns.
  std::vector<float> qkvz, ab, out, conv;
  auto rows_of = [&](const std::vector<float> &src, int r0, int n, int cols, std::vector<float> &dst) {
    dst.insert(dst.end(), src.begin() + static_cast<size_t>(r0) * cols, src.begin() + static_cast<size_t>(r0 + n) * cols);
  };
  rows_of(m.qkv, kh0 * D, s.key_w(), H, qkvz);
  rows_of(m.qkv, KWa + kh0 * D, s.key_w(), H, qkvz);
  rows_of(m.qkv, 2 * KWa + vh0 * D, s.val_w(), H, qkvz);
  rows_of(m.z, vh0 * D, s.val_w(), H, qkvz);
  rows_of(m.conv, kh0 * D, s.key_w(), 4, conv);
  rows_of(m.conv, KWa + kh0 * D, s.key_w(), 4, conv);
  rows_of(m.conv, 2 * KWa + vh0 * D, s.val_w(), 4, conv);
  rows_of(m.a, vh0, s.vh(), H, ab);
  rows_of(m.b, vh0, s.vh(), H, ab);
  const int VWa = m.s.val_w();
  out.resize(static_cast<size_t>(H) * s.val_w());
  for (int n = 0; n < H; ++n)
    std::memcpy(&out[static_cast<size_t>(n) * s.val_w()], &m.out[static_cast<size_t>(n) * VWa + vh0 * D], s.val_w() * 4ull);
  d.w.qkvz = i8_matrix(f, c, qkvz, s.qkvz_n(), H, d.ref.qkvz, d.mem);
  d.w.a = reinterpret_cast<const bf16 *>(up(f, c, bits(std::vector<float>(ab.begin(), ab.begin() + static_cast<size_t>(s.vh()) * H)), d.mem));
  d.w.b = reinterpret_cast<const bf16 *>(up(f, c, bits(std::vector<float>(ab.begin() + static_cast<size_t>(s.vh()) * H, ab.end())), d.mem));
  d.ref.ab = ab;
  d.w.out = i8_matrix(f, c, out, H, s.val_w(), d.ref.out, d.mem);
  d.ref.conv = conv;
  d.w.conv = reinterpret_cast<const bf16 *>(up(f, c, bits(conv), d.mem));
  d.ref.a_log.assign(m.a_log.begin() + vh0, m.a_log.begin() + vh0 + s.vh());
  d.ref.dt_bias.assign(m.dt_bias.begin() + vh0, m.dt_bias.begin() + vh0 + s.vh());
  d.ref.norm = m.norm;
  d.w.a_log = reinterpret_cast<const bf16 *>(up(f, c, bits(d.ref.a_log), d.mem));
  d.w.dt_bias = reinterpret_cast<const bf16 *>(up(f, c, bits(d.ref.dt_bias), d.mem));
  d.w.norm = reinterpret_cast<const bf16 *>(up(f, c, bits(d.ref.norm), d.mem));
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(c, bytes);
    d.mem.push_back(p);
    return p;
  };
  d.st.S = static_cast<float *>(dev(k::gdn_S_elems(s) * 4));
  for (int p = 0; p < 2; ++p) {
    d.st.win[p] = static_cast<bf16 *>(dev(k::gdn_win_elems(s) * 2));
    d.st.y[p] = static_cast<float *>(dev(4ull * d.w.qkvz.n * 4));
    d.st.ab[p] = static_cast<float *>(dev(4ull * s.ab_n() * 4));
  }
  d.accept = static_cast<int32_t *>(dev(64));
  d.st.accept = d.accept;
  constexpr size_t kR = k::kStepRows32;  // the 32-row shape's rows
  d.sc.core = static_cast<bf16 *>(dev(kR * s.val_w() * 2));
  d.sc.partial = static_cast<float *>(dev(kR * H * 4));
  d.x = static_cast<bf16 *>(dev(kR * H * 2));
  d.out = static_cast<bf16 *>(dev(kR * H * 2));
  d.q->memset(d.sc.core, 0, kR * s.val_w() * 2).wait();
  if (cards > 1) {
    d.wire.inbox = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_wire_words(H, cards) * 4));
    d.mem.push_back(d.wire.inbox);
    d.wire.error = static_cast<uint32_t *>(dev(4096));
    k::pair_wire_init(*d.q, H, d.wire.inbox, cards).wait();
    d.q->memset(d.wire.error, 0, 4).wait();
  }
  return d;
}
void free_card(xenithium::Fabric &f, Card &d) {
  for (void *p : d.mem) f.release(p);
  d.mem.clear();
}
// The cards' decode wires: the pair's peer inboxes, or the four-card wire.
void link_wires(Card *d, int C) {
  for (int c = 0; c < C; ++c) {
    if (C == 2) {
      d[c].wire.peer_inbox = d[1 - c].wire.inbox;
    } else if (C == 4) {
      d[c].wire.quad.card = c;
      for (int o = 0; o < C; ++o) d[c].wire.quad.inbox[o] = d[o].wire.inbox;
    }
  }
}
void reset(Card &d) {
  k::gdn_state_reset(*d.q, d.s, d.st).wait();
  d.q->memset(d.accept, 0, 4).wait();
}

template <typename T>
std::vector<T> rd(sycl::queue &q, const T *p, size_t n) {
  std::vector<T> v(n);
  q.memcpy(v.data(), p, n * sizeof(T)).wait();
  return v;
}
std::vector<float> rd16(sycl::queue &q, const bf16 *p, size_t n) {
  const auto b = rd(q, reinterpret_cast<const uint16_t *>(p), n);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
  return v;
}

struct Step {
  int rows, accept;  // accept: how many of this step's rows the next step commits
};
const std::vector<Step> kPlan = {{1, 1}, {4, 2}, {4, 4}, {2, 1}, {3, 3}, {1, 1}, {4, 0}, {4, 3}, {1, 1}, {2, 2}, {4, 1}, {1, 1}};

std::vector<float> make_x(int H, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(4ull * H);
  for (auto &v : x) v = bfr(nd(rng));
  return x;
}

// What a run produced per step (rows < R): out bf16 values, core y, committed S after the step.
struct Trace {
  std::vector<std::vector<float>> out, y;
  std::vector<std::vector<float>> S;
  std::vector<std::vector<float>> proj, ab;  // device projections [4][n] fp32 per step
};

// One card through the plan, waiting after each step; the trace for the checks.
Trace run_one(Card &d, const std::vector<Step> &plan, uint32_t seed, int qks) {
  reset(d);
  Trace t;
  const int H = d.s.hidden;
  for (size_t i = 0; i < plan.size(); ++i) {
    const int prev = i ? plan[i - 1].accept : 0;
    d.q->memcpy(d.accept, &prev, 4).wait();
    const auto x = make_x(H, seed + static_cast<uint32_t>(i));
    const auto xb = bits(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    k::gdn_decode(*d.q, d.s, plan[i].rows, d.x, d.w, d.st, static_cast<int>(i), d.sc, nullptr, 0, d.out, qks);
    d.q->wait_and_throw();
    const int R = plan[i].rows;
    t.out.push_back(rd16(*d.q, d.out, static_cast<size_t>(R) * H));
    t.y.push_back(rd16(*d.q, d.sc.core, static_cast<size_t>(R) * d.s.val_w()));
    t.S.push_back(rd(*d.q, d.st.S, k::gdn_S_elems(d.s)));
    t.proj.push_back(rd(*d.q, d.st.y[i & 1], 4ull * d.w.qkvz.n));
    t.ab.push_back(rd(*d.q, d.st.ab[i & 1], 4ull * d.s.ab_n()));
  }
  return t;
}

// the same plan with the host hints the qwen35 model passes - accept_max = the exact accept word (the
// core stages that many previous rows) and commit on 1-row steps (S and the window committed after the row; the next step
// then replays nothing: accept 0). Outputs must equal run_one's bit for bit (1-row steps of kPlan always accept 1).
Trace run_hint(Card &d, const std::vector<Step> &plan, uint32_t seed, int qks) {
  reset(d);
  Trace t;
  const int H = d.s.hidden;
  bool committed = false;
  for (size_t i = 0; i < plan.size(); ++i) {
    const int prev = i && !committed ? plan[i - 1].accept : 0;
    d.q->memcpy(d.accept, &prev, 4).wait();
    const auto x = make_x(H, seed + static_cast<uint32_t>(i));
    const auto xb = bits(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    k::GdnState st = d.st;
    st.accept_max = prev;
    st.commit = plan[i].rows == 1;
    k::gdn_decode(*d.q, d.s, plan[i].rows, d.x, d.w, st, static_cast<int>(i), d.sc, nullptr, 0, d.out, qks);
    d.q->wait_and_throw();
    committed = st.commit;
    t.out.push_back(rd16(*d.q, d.out, static_cast<size_t>(plan[i].rows) * H));
  }
  return t;
}

void test_one(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d = make_card(f, 0, m, 1);
  const auto &s = d.s;
  const int H = s.hidden, VW = s.val_w(), CC = s.conv_ch();
  const Trace t = run_one(d, kPlan, 100, 0);
  char what[512];
  // Reference, teacher-forced at the device projections; committed state advances by accept.
  k::GdnRefState st;
  st.reset(s);
  BoundCmp yc, oc, fc;
  double s_err = 0, s_max = 0;
  k::GdnRefState fr;  // free-running
  fr.reset(s);
  for (size_t i = 0; i < kPlan.size(); ++i) {
    const int R = kPlan[i].rows, ldy = d.w.qkvz.n, ldab = s.ab_n();
    const auto x = make_x(H, 100 + static_cast<uint32_t>(i));
    k::GdnRefState run = st, frun = fr;
    std::vector<k::GdnRefState> after;
    for (int r = 0; r < R; ++r) {
      std::vector<float> pq(s.qkvz_n()), pa(s.ab_n()), o(VW), y(VW), fo(VW), fy(VW), fq(s.qkvz_n()), fa(s.ab_n());
      for (int n = 0; n < s.qkvz_n(); ++n) pq[n] = bfr(t.proj[i][static_cast<size_t>(r) * ldy + n]);
      for (int n = 0; n < s.ab_n(); ++n) pa[n] = bfr(t.ab[i][static_cast<size_t>(r) * ldab + n]);
      k::gdn_ref_core(s, d.ref, run, pq.data(), pa.data(), o.data(), y.data());
      after.push_back(run);
      for (int h = 0; h < s.vh(); ++h) {
        std::vector<double> yr(y.begin() + h * 128, y.begin() + (h + 1) * 128);
        yc.add_row(&t.y[i][static_cast<size_t>(r) * VW + h * 128], yr.data(), 128);
      }
      std::vector<double> part(H);
      k::gdn_ref_out(s, d.ref, &t.y[i][static_cast<size_t>(r) * VW], part.data());
      oc.add_row(&t.out[i][static_cast<size_t>(r) * H], part.data(), H);
      k::gdn_ref_proj(s, d.ref, &x[static_cast<size_t>(r) * H], fq.data(), fa.data());
      k::gdn_ref_core(s, d.ref, frun, fq.data(), fa.data(), fo.data(), fy.data());
      k::gdn_ref_out(s, d.ref, fy.data(), part.data());
      fc.add_row(&t.out[i][static_cast<size_t>(r) * H], part.data(), H);
      if (r + 1 == kPlan[i].accept) fr = frun;
    }
    if (kPlan[i].accept > 0) st = after[kPlan[i].accept - 1];
    // Device S after step i is the state committed at its start (after replaying step i-1's
    // accepted rows): compare against the reference's state before this step's commit.
    (void)CC;
  }
  // S: the device's committed state after the last step = the reference before the last commit.
  {
    k::GdnRefState st2;
    st2.reset(s);
    for (size_t i = 0; i + 1 < kPlan.size(); ++i) {
      k::GdnRefState run = st2;
      for (int r = 0; r < kPlan[i].accept; ++r) {
        std::vector<float> pq(s.qkvz_n()), pa(s.ab_n()), o(VW), y(VW);
        for (int n = 0; n < s.qkvz_n(); ++n) pq[n] = bfr(t.proj[i][static_cast<size_t>(r) * d.w.qkvz.n + n]);
        for (int n = 0; n < s.ab_n(); ++n) pa[n] = bfr(t.ab[i][static_cast<size_t>(r) * s.ab_n() + n]);
        k::gdn_ref_core(s, d.ref, run, pq.data(), pa.data(), o.data(), y.data());
      }
      st2 = run;
    }
    const auto &S = t.S.back();
    for (size_t j = 0; j < S.size(); ++j) {
      s_err = std::max(s_err, std::fabs(S[j] - st2.S[j]));
      s_max = std::max(s_max, std::fabs(st2.S[j]));
    }
  }
  std::snprintf(what, sizeof what, "%s one card, %zu steps (rows 1-4, accept 0-4): y vs reference from the device's projections %s", tag,
                kPlan.size(), yc.text().c_str());
  expect(yc.scaled <= 1.0 / 64 && yc.off * 200 <= yc.n, what);
  std::snprintf(what, sizeof what, "%s one card: committed S vs reference: max |dS| %.2e of max |S| %.2e", tag, s_err, s_max);
  expect(s_err <= 1e-4 * s_max && s_max > 0, what);
  std::snprintf(what, sizeof what, "%s one card: out vs reference from the device's y %s (free-running: %s)", tag, oc.text().c_str(),
                fc.text().c_str());
  expect(oc.ok() && fc.scaled <= 1.0 / 32, what);

  // Determinism.
  const Trace t2 = run_one(d, kPlan, 100, 0);
  std::snprintf(what, sizeof what, "%s one card: run-to-run bit-identical (outputs and state)", tag);
  expect(t2.out == t.out && t2.S.back() == t.S.back(), what);
  const Trace th = run_hint(d, kPlan, 100, 0);
  std::snprintf(what, sizeof what, "%s one card: accept_max = the accept word and commit on 1-row steps == the plain protocol, bit for bit", tag);
  expect(th.out == t.out, what);

  // Protocol: "4 rows then accept c" == "c rows then the next step".
  bool proto = true;
  for (int c = 0; c <= 4; ++c) {
    const std::vector<Step> pa = {{1, 1}, {4, 2}, {4, c}, {3, 3}, {2, 2}};
    std::vector<Step> pb = pa;
    pb[2].rows = std::max(c, 1);  // c = 0: one row, none accepted
    const Trace ta = run_one(d, pa, 300, 0), tb = run_one(d, pb, 300, 0);
    bool same = ta.S.back() == tb.S.back();
    for (size_t i = 0; i < pa.size(); ++i) {
      const size_t n = static_cast<size_t>(std::min(pa[i].rows, pb[i].rows)) * H;
      same &= std::equal(ta.out[i].begin(), ta.out[i].begin() + n, tb.out[i].begin());
    }
    if (!same) std::printf("  protocol differs at accept %d\n", c);
    proto &= same;
  }
  std::snprintf(what, sizeof what, "%s one card: 4 rows then accept c == c rows then the next step, bit for bit (c = 0..4)", tag);
  expect(proto, what);
  free_card(f, d);
}

// Two cards (heads split, out_proj k split, pair_reduce) vs one card, steps back to back.
// Every card of the fabric (2, or 4: the four-card wire; 4 key / 12 value heads per card).
void test_cards(xenithium::Fabric &f, const Model &m, const char *tag) {
  const int C = static_cast<int>(f.size());
  const std::string who = std::string(tag) + (C == 2 ? " two cards" : " four cards");
  Card one = make_card(f, 0, m, 1);
  Card d[4];
  for (int c = 0; c < C; ++c) d[c] = make_card(f, c, m, C);
  link_wires(d, C);
  const int H = m.s.hidden;
  const int qks = k::dense_auto_ksplit(d[0].w.qkvz.n, H);
  const Trace t1 = run_one(one, kPlan, 500, qks);
  // Two cards: submit every step on both cards without a host wait (accept words set by a
  // queued copy), keep each step's out / core.
  std::vector<int32_t> acc(kPlan.size());
  for (size_t i = 0; i < kPlan.size(); ++i) acc[i] = i ? kPlan[i - 1].accept : 0;
  std::vector<bf16 *> outs[4], cores[4];
  std::vector<int32_t *> accs[4];
  for (int c = 0; c < C; ++c) {
    reset(d[c]);
    for (size_t i = 0; i < kPlan.size(); ++i) {
      outs[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * H * 2)));
      cores[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * d[c].s.val_w() * 2)));
      accs[c].push_back(static_cast<int32_t *>(f.alloc(c, 64)));
      f.compute(c).memcpy(accs[c].back(), &acc[i], 4).wait();
    }
  }
  std::vector<std::vector<uint16_t>> xs;
  for (size_t i = 0; i < kPlan.size(); ++i) xs.push_back(bits(make_x(H, 500 + static_cast<uint32_t>(i))));
  std::vector<bf16 *> xb[4];
  for (int c = 0; c < C; ++c)
    for (size_t i = 0; i < kPlan.size(); ++i) {
      xb[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * H * 2)));
      f.compute(c).memcpy(xb[c].back(), xs[i].data(), xs[i].size() * 2).wait();
    }
  for (size_t i = 0; i < kPlan.size(); ++i)
    for (int c = 0; c < C; ++c) {
      k::GdnState st = d[c].st;
      st.accept = accs[c][i];
      k::GdnScratch sc = d[c].sc;
      sc.core = cores[c][i];
      k::gdn_decode(f.compute(c), d[c].s, kPlan[i].rows, xb[c][i], d[c].w, st, static_cast<int>(i), sc, &d[c].wire, static_cast<int>(i),
                    outs[c][i], qks);
    }
  for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  bool same = true, yeq = true;
  BoundCmp tp;
  for (size_t i = 0; i < kPlan.size(); ++i) {
    const int R = kPlan[i].rows;
    const auto a = rd16(f.compute(0), outs[0][i], static_cast<size_t>(R) * H);
    for (int c = 1; c < C; ++c) same &= rd16(f.compute(c), outs[c][i], a.size()) == a;
    std::vector<double> ref(static_cast<size_t>(R) * H, 0.0), part(H);
    for (int c = 0; c < C; ++c) {
      const int VWc = d[c].s.val_w(), VW1 = one.s.val_w();
      const auto y = rd16(f.compute(c), cores[c][i], static_cast<size_t>(R) * VWc);
      for (int r = 0; r < R; ++r) {
        yeq &= std::equal(y.begin() + static_cast<size_t>(r) * VWc, y.begin() + static_cast<size_t>(r + 1) * VWc,
                          t1.y[i].begin() + static_cast<size_t>(r) * VW1 + c * VWc);
        k::gdn_ref_out(d[c].s, d[c].ref, &y[static_cast<size_t>(r) * VWc], part.data());
        for (int n = 0; n < H; ++n) ref[static_cast<size_t>(r) * H + n] += part[n];
      }
    }
    for (int r = 0; r < R; ++r) tp.add_row(&a[static_cast<size_t>(r) * H], &ref[static_cast<size_t>(r) * H], H);
  }
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
    err |= e;
  }
  std::vector<uint32_t> inbox = rd(f.compute(0), d[0].wire.inbox, k::pair_wire_words(H, C));
  const bool empty = std::all_of(inbox.begin(), inbox.end(), [](uint32_t v) { return v == k::kPairMailEmpty; });
  char what[512];
  std::snprintf(what, sizeof what, "%s, %zu steps back to back: every card bit-identical", who.c_str(), kPlan.size());
  expect(same, what);
  std::snprintf(what, sizeof what, "%s: each card's y == the one-card y of its heads, bit for bit", who.c_str());
  expect(yeq, what);
  std::snprintf(what, sizeof what, "%s: out vs the sum of the per-card references from the device's y %s", who.c_str(), tp.text().c_str());
  expect(tp.ok(), what);
  std::snprintf(what, sizeof what, "%s: no mailbox timeout, inbox empty after the run", who.c_str());
  expect(!err && empty, what);
  for (int c = 0; c < C; ++c) {
    for (auto *p : outs[c]) f.release(p);
    for (auto *p : cores[c]) f.release(p);
    for (auto *p : accs[c]) f.release(p);
    for (auto *p : xb[c]) f.release(p);
    free_card(f, d[c]);
  }
  free_card(f, one);
}

// gdn_rows (prefill) against gdn_decode on 4-row slices of the same rows (the decode
// block is gated against the reference above). History: a 4-row decode step whose first 3 rows the
// chunk accepts (gdn_rows replays them from the pending log). Then one 4-row decode step after the
// chunk (the handoff: accept 0, window from win_out). Both keep S in fp32 through the rows, so y /
// out / S differ by fp32 summation order only (projections on pf_dense).
struct RowsRun {
  std::vector<float> out, y, out_after, y_after, S, ycard1;
  std::vector<std::vector<float>> ycards;  // every card's y; empty when cards differ
  std::vector<float> S_call, win_call, snap_S, snap_win;  // the state right after the call / at snap_pos
};
RowsRun run_path(xenithium::Fabric &f, Card *d, int C, int M, bool slices, k::GdnRowsScratch *rs, k::PairRowsWire *rw, bf16 *const *xd,
                 bf16 *const *xa, bf16 *const *xn, const k::PairSplit *sp = nullptr, uint32_t seq = 0, sycl::queue *const *side = nullptr,
                 int pos = -1, int snap_pos = -1, int qks = 0) {  // qks: the decode qkvz k split (0 = the card shape's own)
  const int H = d[0].s.hidden, VW = d[0].s.val_w();
  RowsRun o;
  std::vector<void *> tmp[4];
  auto word = [&](int c, int v) {
    auto *p = static_cast<int32_t *>(f.alloc(c, 64));
    f.compute(c).memcpy(p, &v, 4).wait();
    tmp[c].push_back(p);
    return p;
  };
  int wp = 0;  // the decode pair wire's parity (per call)
  auto dec = [&](int rows, bf16 *const *x, int step, int accept) {
    for (int c = 0; c < C; ++c) {
      k::GdnState st = d[c].st;
      st.accept = word(c, accept);
      k::gdn_decode(f.compute(c), d[c].s, rows, x[c], d[c].w, st, step, d[c].sc, C > 1 ? &d[c].wire : nullptr, wp, d[c].out, qks);
    }
    ++wp;
    for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  };
  for (int c = 0; c < C; ++c) reset(d[c]);
  dec(4, xa, 0, 0);  // history: 4 pending rows
  int last = 3, step = 1;
  if (slices) {
    for (int sl = 0; sl * 4 < M; ++sl) {
      const int rows = std::min(4, M - 4 * sl);
      bf16 *xs[4] = {};
      for (int c = 0; c < C; ++c) xs[c] = xd[c] + static_cast<size_t>(sl) * 4 * H;
      dec(rows, xs, step, last);
      const auto a = rd16(f.compute(0), d[0].out, static_cast<size_t>(rows) * H);
      const auto y = rd16(f.compute(0), d[0].sc.core, static_cast<size_t>(rows) * VW);
      o.out.insert(o.out.end(), a.begin(), a.end());
      o.y.insert(o.y.end(), y.begin(), y.end());
      last = rows;
      ++step;
    }
  } else {
    const int sp0 = 1;  // the parity the step after the history would have
    bf16 *out[4] = {};
    float *snap_S0 = nullptr;
    bf16 *snap_W0 = nullptr;
    for (int c = 0; c < C; ++c) {
      out[c] = static_cast<bf16 *>(f.alloc(c, static_cast<size_t>(M + 4) * H * 2));
      tmp[c].push_back(out[c]);
      k::GdnRowsState st;
      st.S = d[c].st.S;
      st.win_in = d[c].st.win[sp0 ^ 1];
      st.win_out = d[c].st.win[sp0];  // read by the next step (parity sp0 ^ 1) as win[its parity ^ 1]
      st.log_y = d[c].st.y[sp0 ^ 1];
      st.log_ab = d[c].st.ab[sp0 ^ 1];
      st.accept = word(c, last);
      st.pos = pos;  // >= 0 = full 64-row chunks on the chunked scan
      if (snap_pos >= 0) {  // the state at snap_pos too
        st.snap_pos = snap_pos;
        st.snap_S = static_cast<float *>(f.alloc(c, k::gdn_S_elems(d[c].s) * 4));
        st.snap_win = static_cast<bf16 *>(f.alloc(c, k::gdn_win_elems(d[c].s) * 2));
        tmp[c].push_back(st.snap_S);
        tmp[c].push_back(st.snap_win);
        if (c == 0) snap_S0 = st.snap_S, snap_W0 = st.snap_win;
      }
      k::PairRowsWire w = C > 1 ? rw[c] : k::PairRowsWire{};
      w.split = sp != nullptr;
      k::PairSplitSend snd{side ? side[c] : nullptr, sp ? &sp[c] : nullptr, seq};
      if (sp && side && seq % 2 == 1) w.send = &snd;  // the out projection sends the peer's rows first
      k::gdn_rows(f.compute(c), d[c].s, M, xd[c], d[c].w, st, rs[c], C > 1 ? &w : nullptr, 0, out[c]);
    }
    if (sp) {  // finish the two-card sum split over the cards
      sycl::queue *qs[2] = {&f.compute(0), &f.compute(1)};
      const float *part[2] = {rs[0].partial, rs[1].partial};
      if (side && seq % 2 == 1) {
        k::pair_split_finish(qs, side, M, H, part, out, sp, seq);
      } else {
        k::pair_split_rows(qs, M, H, part, out, sp, seq, nullptr, side);
      }
      if (side)
        for (int c = 0; c < 2; ++c) side[c]->wait_and_throw();
    }
    for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
    o.S_call = rd(f.compute(0), d[0].st.S, k::gdn_S_elems(d[0].s));
    o.win_call = rd16(f.compute(0), d[0].st.win[sp0], k::gdn_win_elems(d[0].s));
    if (snap_S0) {
      o.snap_S = rd(f.compute(0), snap_S0, k::gdn_S_elems(d[0].s));
      o.snap_win = rd16(f.compute(0), snap_W0, k::gdn_win_elems(d[0].s));
    }
    o.out = rd16(f.compute(0), out[0], static_cast<size_t>(M) * H);
    o.y = rd16(f.compute(0), rs[0].y, static_cast<size_t>(M) * VW);
    if (C > 1) {
      o.ycard1 = rd16(f.compute(1), rs[1].y, static_cast<size_t>(M) * d[1].s.val_w());
      if (rd16(f.compute(1), out[1], o.out.size()) != o.out) o.ycard1.clear();  // marks "cards differ"
      bool differ = false;
      for (int c = 0; c < C; ++c) {
        o.ycards.push_back(rd16(f.compute(c), rs[c].y, static_cast<size_t>(M) * d[c].s.val_w()));
        differ |= rd16(f.compute(c), out[c], o.out.size()) != o.out;
      }
      if (differ) o.ycards.clear();
    }
    last = 0;
    step = 2;  // a chunk advances the step parity by one
  }
  dec(4, xn, step, last);  // the step after: commits the rest, reads the window the rows left
  o.out_after = rd16(f.compute(0), d[0].out, 4ull * H);
  o.y_after = rd16(f.compute(0), d[0].sc.core, 4ull * VW);
  o.S = rd(f.compute(0), d[0].st.S, k::gdn_S_elems(d[0].s));
  for (int c = 0; c < C; ++c)
    for (void *p : tmp[c]) f.release(p);
  return o;
}

BoundCmp bound(const std::vector<float> &dev, const std::vector<float> &ref, int len) {
  BoundCmp b;
  std::vector<double> r(len);
  for (size_t r0 = 0; r0 + len <= ref.size(); r0 += len) {
    for (int i = 0; i < len; ++i) r[i] = ref[r0 + i];
    b.add_row(&dev[r0], r.data(), len);
  }
  return b;
}

void test_rows(xenithium::Fabric &f, const Model &m, const char *tag, int M) {
  const int C = f.size() >= 4 ? 4 : f.size() > 1 ? 2 : 1, H = m.s.hidden;
  std::vector<uint16_t> xs = bits(make_x(H, 7000)), xa = bits(make_x(H, 7001)), xn = bits(make_x(H, 7002));
  std::vector<float> xchunk(static_cast<size_t>(M + 4) * H);
  {
    std::mt19937 rng(7003u + M);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &v : xchunk) v = bfr(nd(rng));
  }
  const auto xcb = bits(xchunk);
  (void)xs;
  auto setup = [&](int cards, Card *d, k::GdnRowsScratch *rs, k::PairRowsWire *rw, bf16 **xd, bf16 **xap, bf16 **xnp, std::vector<void *> *mem) {
    for (int c = 0; c < cards; ++c) {
      d[c] = make_card(f, c, m, cards);
      auto dev = [&](size_t bytes) {
        void *p = f.alloc(c, std::max<size_t>(bytes, 64));
        mem[c].push_back(p);
        return p;
      };
      const auto z = k::gdn_rows_sizes(d[c].s, d[c].w.qkvz.n, M);
      rs[c].yq = static_cast<float *>(dev(z.yq * 4));
      rs[c].ab = static_cast<float *>(dev(z.ab * 4));
      rs[c].cv = static_cast<float *>(dev(z.cv * 4));
      rs[c].qn = static_cast<float *>(dev(z.qk * 4));
      rs[c].kn = static_cast<float *>(dev(z.qk * 4));
      rs[c].gate = static_cast<float *>(dev(z.gate * 4));
      rs[c].lgate = static_cast<float *>(dev(z.lgate * 4));
      {
        const auto zs = k::gdn_scan_sizes(d[c].s, M);
        rs[c].scan.wk = static_cast<bf16 *>(dev(zs.wk * 2));
        rs[c].scan.qg = static_cast<bf16 *>(dev(zs.wk * 2));
        rs[c].scan.kt = static_cast<bf16 *>(dev(zs.wk * 2));
        rs[c].scan.kt_lo = static_cast<bf16 *>(dev(zs.wk * 2));
        rs[c].scan.mt = static_cast<bf16 *>(dev(zs.mt * 2));
        rs[c].scan.uv = static_cast<float *>(dev(zs.uv * 4));
        rs[c].scan.gc = static_cast<float *>(dev(zs.gc * 4));
      }
      rs[c].o = static_cast<bf16 *>(dev(z.o * 2));
      rs[c].y = static_cast<bf16 *>(dev(z.o * 2));
      rs[c].partial = static_cast<float *>(dev(z.partial * 4));
      if (cards > 1) {
        rw[c].rows_max = M;
        rw[c].n_max = H;
        rw[c].inbox = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_rows_wire_words(M, H, cards) * 4));
        mem[c].push_back(rw[c].inbox);
        rw[c].error = d[c].wire.error;
        k::pair_rows_wire_init(f.compute(c), M, H, rw[c].inbox, cards).wait();
      }
      xd[c] = static_cast<bf16 *>(dev(xcb.size() * 2));
      f.compute(c).memcpy(xd[c], xcb.data(), xcb.size() * 2).wait();
      xap[c] = static_cast<bf16 *>(dev(xa.size() * 2));
      f.compute(c).memcpy(xap[c], xa.data(), xa.size() * 2).wait();
      xnp[c] = static_cast<bf16 *>(dev(xn.size() * 2));
      f.compute(c).memcpy(xnp[c], xn.data(), xn.size() * 2).wait();
    }
    link_wires(d, cards);
    for (int c = 0; c < cards; ++c) {
      if (cards == 2) rw[c].peer_inbox = rw[1 - c].inbox;
      if (cards == 4) {
        rw[c].quad.card = c;
        for (int o = 0; o < cards; ++o) rw[c].quad.inbox[o] = rw[o].inbox;
      }
    }
  };
  auto teardown = [&](int cards, Card *d, std::vector<void *> *mem) {
    for (int c = 0; c < cards; ++c) {
      for (void *p : mem[c]) f.release(p);
      free_card(f, d[c]);
    }
  };
  char what[512];
  // one card: rows vs slices
  RowsRun a, b;
  {
    Card d[1];
    k::GdnRowsScratch rs[1];
    k::PairRowsWire rw[1];
    bf16 *xd[1], *xap[1], *xnp[1];
    std::vector<void *> mem[1];
    setup(1, d, rs, rw, xd, xap, xnp, mem);
    a = run_path(f, d, 1, M, true, rs, rw, xd, xap, xnp);
    b = run_path(f, d, 1, M, false, rs, rw, xd, xap, xnp);
    const RowsRun b2 = run_path(f, d, 1, M, false, rs, rw, xd, xap, xnp);
    const int VW = d[0].s.val_w();
    const BoundCmp yc = bound(b.y, a.y, 128), oc = bound(b.out, a.out, H), ya = bound(b.y_after, a.y_after, 128), oa = bound(b.out_after, a.out_after, H);
    double s_err = 0, s_max = 0;
    for (size_t j = 0; j < a.S.size(); ++j) {
      s_err = std::max<double>(s_err, std::fabs(b.S[j] - a.S[j]));
      s_max = std::max<double>(s_max, std::fabs(a.S[j]));
    }
    (void)VW;
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows (after 3 replayed): y vs the decode path on 4-row slices %s", tag, M, yc.text().c_str());
    expect(yc.scaled <= 1.0 / 64 && yc.off * 20 <= yc.n, what);
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows: out vs the decode path %s", tag, M, oc.text().c_str());
    expect(oc.scaled <= 1.0 / 64 && oc.off * 20 <= oc.n, what);
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows: the next decode step (accept 0, win_out): y %s, out %s; committed S max |dS| %.2e of %.2e", tag,
                  M, ya.text().c_str(), oa.text().c_str(), s_err, s_max);
    expect(ya.scaled <= 1.0 / 64 && oa.scaled <= 1.0 / 64 && s_err <= 1e-4 * s_max && s_max > 0, what);
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows: run-to-run bit-identical (out, next step, S)", tag, M);
    expect(b2.out == b.out && b2.out_after == b.out_after && b2.S == b.S, what);
    // The chunked scan (full 64-row chunks at absolute multiples of 64, the rest on the row
    // core). Its numerics differ (S enters as a bf16 operand): bounded against the decode path, run-to-run
    // bit-identical. pos 3: row 0 (3 replayed rows) at position 0, chunks from row 0; pos 40: row 0 at
    // position 37, 27 rows on the row core before the first boundary.
    if (M + 3 >= k::kGdnChunk) {
      auto rel = [](const std::vector<float> &x, const std::vector<float> &r) {
        double num = 0, den = 0;
        for (size_t j = 0; j < r.size(); ++j) num += (double(x[j]) - r[j]) * (double(x[j]) - r[j]), den += double(r[j]) * r[j];
        return std::sqrt(num / std::max(den, 1e-300));
      };
      for (int pos : {3, 40}) {
        const RowsRun c = run_path(f, d, 1, M, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, pos);
        const RowsRun c2 = run_path(f, d, 1, M, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, pos);
        const BoundCmp cy = bound(c.y, a.y, 128), cya = bound(c.y_after, a.y_after, 128);
        const double ry = rel(c.y, a.y), ro = rel(c.out, a.out), rya = rel(c.y_after, a.y_after), rs_ = rel(c.S, a.S), rrow = rel(b.y, a.y);
        std::snprintf(what, sizeof what,
                      "%s gdn_rows %d rows, chunked scan from position %d: vs the decode path rel rms y %.2e (row core %.2e), out %.2e, next step y %.2e, S %.2e; "
                      "y worst %.2e, next step worst %.2e",
                      tag, M, pos - 3, ry, rrow, ro, rya, rs_, cy.scaled, cya.scaled);
        // The worst-element bound of max(|ref|, the vector's rms) holds to the sizes (<= 300 rows). Past that, near-zero
        // output vectors make it meaningless: there the denominator has a floor of 1 % of the global rms.
        auto worst_abs = [](const std::vector<float> &x, const std::vector<float> &r) {  // per 128-vector, floor 1 % of the global rms
          double gs = 0, m = 0;
          for (float v : r) gs += double(v) * v;
          const double floor = 0.01 * std::sqrt(gs / std::max<size_t>(1, r.size()));
          for (size_t v0 = 0; v0 + 128 <= r.size(); v0 += 128) {
            double ss = 0;
            for (int i = 0; i < 128; ++i) ss += double(r[v0 + i]) * r[v0 + i];
            const double rms = std::sqrt(ss / 128);
            for (int i = 0; i < 128; ++i) m = std::max(m, std::fabs(double(x[v0 + i]) - r[v0 + i]) / std::max({std::fabs(double(r[v0 + i])), rms, floor}));
          }
          return m;
        };
        const double wy = worst_abs(c.y, a.y), wya = worst_abs(c.y_after, a.y_after);
        const bool worst_ok = M > 300 ? (wy <= 1.0 / 8 && wya <= 1.0 / 8) : (cy.scaled <= 1.0 / 8 && cya.scaled <= 1.0 / 8);
        if (M > 300) std::snprintf(what + std::strlen(what), sizeof what - std::strlen(what), "; worst with a 1 %% global-rms floor: y %.2e, next step %.2e", wy, wya);
        expect(ry <= 1e-2 && ro <= 1e-2 && rya <= 1e-2 && rs_ <= 1e-2 && worst_ok, what);
        std::snprintf(what, sizeof what, "%s gdn_rows %d rows, chunked scan from position %d: run-to-run bit-identical (out, next step, S)", tag, M, pos - 3);
        expect(c2.out == c.out && c2.out_after == c.out_after && c2.S == c.S, what);
        // a snapshot inside the call (the scan in two parts around it) changes none of the
        // call's bits, and holds what a call ending there leaves (S, window)
        const int base = pos - 3;
        for (int sn = (base / 64 + 1) * 64; sn < base + M + 3 - 64; sn += 128) {
          if (sn - pos < 64) continue;  // the shorter call keeps 64+ new rows (the engine's calls)
          const RowsRun cs = run_path(f, d, 1, M, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, pos, sn);
          const RowsRun cc = run_path(f, d, 1, sn - pos, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, pos);
          std::snprintf(what, sizeof what, "%s gdn_rows %d rows from position %d, snapshot at %d: the call's out / next step / S unchanged, bit for bit", tag, M,
                        base, sn);
          expect(cs.out == c.out && cs.out_after == c.out_after && cs.S == c.S && cs.S_call == c.S_call && cs.win_call == c.win_call, what);
          std::snprintf(what, sizeof what, "%s gdn_rows snapshot at %d == the state a call ending there leaves (S, window), bit for bit", tag, sn);
          expect(!cs.snap_S.empty() && cs.snap_S == cc.S_call && cs.snap_win == cc.win_call, what);
        }
      }
    }
    teardown(1, d, mem);
  }
  if (C == 4) {  // TP4: 4 key / 12 value heads per card, the four-card rows wire
    Card d[4];
    k::GdnRowsScratch rs[4];
    k::PairRowsWire rw[4];
    bf16 *xd[4], *xap[4], *xnp[4];
    std::vector<void *> mem[4];
    setup(4, d, rs, rw, xd, xap, xnp, mem);
    // the one-card run's decode k split (its history rows' y, replayed by the rows): each card's y == the one-card y of
    // its heads needs the same split (the card shape's own split sums in another order: the decode test's contract)
    k::GdnShape s1 = m.s;
    s1.cards = 1;
    const int qks1 = k::dense_auto_ksplit(s1.qkvz_n(), H);
    const RowsRun t = run_path(f, d, 4, M, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, -1, -1, qks1);
    const RowsRun t2 = run_path(f, d, 4, M, false, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, -1, -1, qks1);
    const RowsRun ts = run_path(f, d, 4, M, true, rs, rw, xd, xap, xnp, nullptr, 0, nullptr, -1, -1, qks1);
    const int VWc = d[0].s.val_w(), VW1 = 4 * VWc;
    bool yeq = t.ycards.size() == 4;
    for (int c = 0; c < 4 && yeq; ++c)
      for (int r = 0; r < M && yeq; ++r)
        yeq &= std::equal(t.ycards[c].begin() + static_cast<size_t>(r) * VWc, t.ycards[c].begin() + static_cast<size_t>(r + 1) * VWc,
                          b.y.begin() + static_cast<size_t>(r) * VW1 + c * VWc);
    // out against the four-card decode path (the per-card int8 out_proj slices: one card's matrix quantises differently)
    const BoundCmp oc = bound(t.out, ts.out, H), oa = bound(t.out_after, ts.out_after, H);
    uint32_t err = 0;
    for (int c = 0; c < 4; ++c) {
      uint32_t e = 0;
      f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
      err |= e;
    }
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows, four cards: every card's out identical; each card's y == the one-card rows y of its heads, bit for bit", tag, M);
    expect(yeq, what);
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows, four cards: out vs the four-card decode path %s, next step %s; run-to-run bit-identical; no mailbox error",
                  tag, M, oc.text().c_str(), oa.text().c_str());
    expect(oc.scaled <= 1.0 / 64 && oc.off * 20 <= oc.n && oa.scaled <= 1.0 / 64 && t2.out == t.out && t2.out_after == t.out_after && err == 0, what);
    teardown(4, d, mem);
  }
  if (C == 2) {
    Card d[2];
    k::GdnRowsScratch rs[2];
    k::PairRowsWire rw[2];
    bf16 *xd[2], *xap[2], *xnp[2];
    std::vector<void *> mem[2];
    setup(2, d, rs, rw, xd, xap, xnp, mem);
    const RowsRun t = run_path(f, d, 2, M, false, rs, rw, xd, xap, xnp);
    const int VW1 = d[0].s.val_w();
    bool yeq = !t.ycard1.empty();
    for (int r = 0; r < M && yeq; ++r) {
      yeq &= std::equal(t.y.begin() + static_cast<size_t>(r) * VW1, t.y.begin() + static_cast<size_t>(r + 1) * VW1,
                        b.y.begin() + static_cast<size_t>(r) * 2 * VW1);
      yeq &= std::equal(t.ycard1.begin() + static_cast<size_t>(r) * VW1, t.ycard1.begin() + static_cast<size_t>(r + 1) * VW1,
                        b.y.begin() + static_cast<size_t>(r) * 2 * VW1 + VW1);
    }
    // each card quantises its own k slice of out_proj: compare with the two-card decode path
    const RowsRun ts = run_path(f, d, 2, M, true, rs, rw, xd, xap, xnp);
    const BoundCmp oc = bound(t.out, ts.out, H), oa = bound(t.out_after, ts.out_after, H);
    uint32_t err[2];
    for (int c = 0; c < 2; ++c) f.compute(c).memcpy(&err[c], d[c].wire.error, 4).wait();
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows, two cards: both cards' out identical; each card's y == the one-card rows y of its heads", tag, M);
    expect(yeq, what);
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows, two cards: out vs the two-card decode path %s, next step %s; no mailbox error", tag, M,
                  oc.text().c_str(), oa.text().c_str());
    expect(oc.scaled <= 1.0 / 64 && oc.off * 20 <= oc.n && oa.scaled <= 1.0 / 64 && !err[0] && !err[1], what);
    // The split two-card sum (pair_split_rows) == the mailbox, bit for bit, twice on the same
    // buffers (fresh sequence numbers; a stale line would show)
    k::PairSplit sp[2];
    for (int c = 0; c < 2; ++c) {
      sp[c].rows_max = M;
      sp[c].n_max = H;
      sp[c].recv = static_cast<float *>(f.alloc(c, k::pair_split_recv_elems(M, H) * 4));
      mem[c].push_back(sp[c].recv);
      sp[c].flags = static_cast<uint32_t *>(f.alloc_mailbox(c, k::kPairSplitFlagBytes));
      mem[c].push_back(sp[c].flags);
      f.compute(c).memset(sp[c].flags, 0, k::kPairSplitFlagBytes).wait();
      sp[c].error = d[c].wire.error;
      {
        auto *tab = static_cast<uint32_t *>(f.alloc(c, k::kPairSplitSeqTabBytes));
        mem[c].push_back(tab);
        k::pair_split_seq_tab_init(f.compute(c), tab).wait();
        sp[c].seq_tab = tab;
      }
    }
    for (int c = 0; c < 2; ++c) {
      sp[c].peer_recv = sp[1 - c].recv;
      sp[c].peer_flags = sp[1 - c].flags;
    }
    bool split_ok = true;
    sycl::property_list sprops{sycl::property::queue::in_order()};
    sycl::queue sideq[2] = {sycl::queue(f.card_context(0), f.device(0), sprops), sycl::queue(f.card_context(1), f.device(1), sprops)};
    sycl::queue *side[2] = {&sideq[0], &sideq[1]};
    for (uint32_t seq = 1; seq <= 4; ++seq) {  // call 2 on side queues, 3 with the projection's send
      const RowsRun t2 = run_path(f, d, 2, M, false, rs, rw, xd, xap, xnp, sp, seq, seq == 2 || seq == 3 ? side : nullptr);
      split_ok &= t2.out == t.out && t2.out_after == t.out_after && t2.ycard1 == t.ycard1 && !t2.ycard1.empty();
    }
    for (int c = 0; c < 2; ++c) f.compute(c).memcpy(&err[c], d[c].wire.error, 4).wait();
    std::snprintf(what, sizeof what, "%s gdn_rows %d rows, two cards: the split sum (pair_split_rows) == the mailbox sum on both cards, next step too, 4 calls (compute queues, side queues, the out projection sending first)", tag, M);
    expect(split_ok && !err[0] && !err[1], what);
    // the bf16 wires (PairRowsWire:bf16 / PairSplit:bf16): the split sum == the rows wire on
    // bf16, both cards, side queues too; close to the fp32 sums (bf16 partials)
    {
      for (int c = 0; c < 2; ++c) {
        rw[c].bf16 = true;
        sp[c].bf16 = true;
        sp[c].stage = static_cast<bf16 *>(f.alloc(c, k::pair_split_recv_elems(M, H) * 2));
        mem[c].push_back(sp[c].stage);
      }
      const RowsRun t16 = run_path(f, d, 2, M, false, rs, rw, xd, xap, xnp);
      bool ok16 = !t16.out.empty();
      for (uint32_t seq = 5; seq <= 8; ++seq) {
        const RowsRun t2 = run_path(f, d, 2, M, false, rs, rw, xd, xap, xnp, sp, seq, seq == 6 || seq == 7 ? side : nullptr);
        ok16 &= t2.out == t16.out && t2.out_after == t16.out_after;
      }
      const BoundCmp ob = bound(t.out, t16.out, H);
      for (int c = 0; c < 2; ++c) f.compute(c).memcpy(&err[c], d[c].wire.error, 4).wait();
      std::snprintf(what, sizeof what, "%s gdn_rows %d rows, two cards, bf16 wires: the split sum == the rows wire bit for bit (4 calls, side queues); vs fp32 %s", tag, M,
                    ob.text().c_str());
      expect(ok16 && ob.scaled <= 1.0 / 64 && !err[0] && !err[1], what);
      for (int c = 0; c < 2; ++c) rw[c].bf16 = sp[c].bf16 = false;
    }
    teardown(2, d, mem);
  }
}

// Several sequences in one step (segment records, rows.hpp) against each sequence alone
// on the one-sequence path, bit for bit: every segment's out and core y rows, and every slot's
// committed S after the run. Sequences skip steps, sit on slots in any row order, and commit
// fewer rows than they ran (the next own step replays its accept count from the slot's log).
struct SegStep {
  std::vector<int> seqs;  // segment i = sequence seqs[i] on slot seqs[i]
  int rows;  // rows per segment
  std::vector<int> rr = {};  // rows per segment when they differ (seg_het: B rows, then 1-row segments)
  int of(size_t i) const { return rr.empty() ? rows : rr[i]; }
  int all() const {
    int n = 0;
    for (size_t i = 0; i < seqs.size(); ++i) n += of(i);
    return n;
  }
  int code() const { return rr.empty() ? rows : k::seg_het(rr[0], static_cast<int>(std::count(rr.begin(), rr.end(), rr[0])), static_cast<int>(seqs.size())); }
};
void test_segments(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d = make_card(f, 0, m, 1);
  const auto &s = d.s;
  const int H = s.hidden, VW = s.val_w();
  constexpr int NS = 16;  // 16 slots (16-row steps of one-row segments)
  // steps of 5-8 rows (the 8-row shape) among them: 2 x 4, 3 x 2, 2 x 3; and of 9-16 rows (the 16-row shape):
  // 16 x 1, 8 x 2, 4 x 4, 3 x 4, unequal (2 x 4 + 8 x 1, 3 x 3 + 3 x 1), skipped slots
  const std::vector<SegStep> plan = {{{0, 1, 2}, 1}, {{0, 1}, 2}, {{2}, 4}, {{1, 0}, 2}, {{0, 1, 2}, 1}, {{2, 0}, 2},
                                     {{1}, 3},       {{0, 1, 2}, 1}, {{0, 2}, 2}, {{1, 2}, 2}, {{2, 1, 0}, 1}, {{0}, 4},
                                     {{0, 1}, 4},    {{2, 0, 1}, 2}, {{1, 2}, 3}, {{2, 0}, 4}, {{0, 1, 2}, 2}, {{1, 0}, 4},
                                     {{0, 1, 2}, 0, {3, 3, 1}}, {{2, 1}, 0, {4, 1}}, {{1, 0, 2}, 0, {4, 1, 1}}, {{0, 2, 1}, 0, {2, 1, 1}},
                                     {{2, 0, 1}, 0, {2, 2, 1}}, {{0, 1, 2}, 1},
                                     {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1},
                                     {{15, 3, 7, 1, 9, 0, 12, 5}, 2},
                                     {{4, 13, 2, 10}, 4},
                                     {{6, 11, 14}, 4},
                                     {{8, 0, 1, 2, 3, 4, 5, 6, 7, 9}, 0, {4, 4, 1, 1, 1, 1, 1, 1, 1, 1}},
                                     {{2, 9, 15, 0, 11, 13}, 0, {3, 3, 3, 1, 1, 1}},
                                     {{14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15}, 1},
                                     {{1, 0, 2}, 3}};
  auto commit_of = [](int k, int rows) { return k % 2 == 0 ? rows : std::max(1, rows - 1); };  // own step k
  auto x_of = [&](int q, int k) { return make_x(H, 9000u + 100u * static_cast<uint32_t>(q) + static_cast<uint32_t>(k)); };
  // solo: each sequence through its own steps on the one-sequence path
  std::vector<std::vector<std::vector<float>>> sout(NS), sy(NS);
  std::vector<std::vector<float>> sS(NS);
  for (int q = 0; q < NS; ++q) {
    reset(d);
    int k = 0, prev = 0;
    for (const auto &st : plan) {
      const auto at = std::find(st.seqs.begin(), st.seqs.end(), q);
      if (at == st.seqs.end()) continue;
      const int rq = st.of(static_cast<size_t>(at - st.seqs.begin()));
      d.q->memcpy(d.accept, &prev, 4).wait();
      const auto xb = bits(x_of(q, k));
      d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
      k::gdn_decode(*d.q, s, rq, d.x, d.w, d.st, k, d.sc, nullptr, 0, d.out);
      d.q->wait_and_throw();
      sout[q].push_back(rd16(*d.q, d.out, static_cast<size_t>(rq) * H));
      sy[q].push_back(rd16(*d.q, d.sc.core, static_cast<size_t>(rq) * VW));
      prev = commit_of(k, rq);
      ++k;
    }
    sS[q] = rd(*d.q, d.st.S, k::gdn_S_elems(s));
  }
  // batched: one state with NS slots, segment records
  std::vector<void *> mem;
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(0, std::max<size_t>(bytes, 64));
    mem.push_back(p);
    d.q->memset(p, 0, std::max<size_t>(bytes, 64)).wait();
    return p;
  };
  k::GdnState g;
  g.slot_S = k::gdn_S_elems(s);
  g.slot_win = k::gdn_win_elems(s);
  g.slot_y = 4ull * d.w.qkvz.n;
  g.slot_ab = 4ull * s.ab_n();
  g.S = static_cast<float *>(dev(NS * g.slot_S * 4));
  for (int p = 0; p < 2; ++p) {
    g.win[p] = static_cast<bf16 *>(dev(NS * g.slot_win * 2));
    g.y[p] = static_cast<float *>(dev(NS * g.slot_y * 4));
    g.ab[p] = static_cast<float *>(dev(NS * g.slot_ab * 4));
  }
  auto *rec = static_cast<int32_t *>(dev(k::kMaxSegs * k::kSegRec * 4));
  g.seg = rec;
  k::GdnScratch sc = d.sc;
  sc.y = static_cast<float *>(dev(static_cast<size_t>(k::kStepRows32) * d.w.qkvz.n * 4));
  int own[NS] = {}, prev[NS] = {};
  bool same = true;
  size_t rows_cmp = 0;
  for (const auto &st : plan) {
    const int ns = static_cast<int>(st.seqs.size()), R = st.all();
    std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
    std::vector<float> x(static_cast<size_t>(k::kStepRows32) * H, 0.f);
    for (int i = 0, r0 = 0; i < ns; r0 += st.of(static_cast<size_t>(i)), ++i) {
      const int q = st.seqs[static_cast<size_t>(i)], rq = st.of(static_cast<size_t>(i));
      w[static_cast<size_t>(i) * k::kSegRec + k::kSegAccept] = prev[q];
      w[static_cast<size_t>(i) * k::kSegRec + k::kSegParity] = own[q] & 1;
      w[static_cast<size_t>(i) * k::kSegRec + k::kSegSlot] = q;
      const auto xq = x_of(q, own[q]);
      std::copy(xq.begin(), xq.begin() + static_cast<size_t>(rq) * H, x.begin() + static_cast<size_t>(r0) * H);
    }
    d.q->memcpy(rec, w.data(), w.size() * 4).wait();
    const auto xb = bits(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    g.seg_rows = st.code();
    k::gdn_decode(*d.q, s, R, d.x, d.w, g, 0, sc, nullptr, 0, d.out);
    d.q->wait_and_throw();
    const auto out = rd16(*d.q, d.out, static_cast<size_t>(R) * H), y = rd16(*d.q, d.sc.core, static_cast<size_t>(R) * VW);
    for (int i = 0, r0 = 0; i < ns; r0 += st.of(static_cast<size_t>(i)), ++i) {
      const int q = st.seqs[static_cast<size_t>(i)], rq = st.of(static_cast<size_t>(i));
      same &= std::equal(out.begin() + static_cast<size_t>(r0) * H, out.begin() + static_cast<size_t>(r0 + rq) * H, sout[q][own[q]].begin());
      same &= std::equal(y.begin() + static_cast<size_t>(r0) * VW, y.begin() + static_cast<size_t>(r0 + rq) * VW, sy[q][own[q]].begin());
      rows_cmp += rq;
      prev[q] = commit_of(own[q], rq);
      ++own[q];
    }
  }
  // S as committed so far (each sequence's last own step is still pending on both paths)
  bool s_same = true;
  const auto Sall = rd(*d.q, g.S, NS * g.slot_S);
  for (int q = 0; q < NS; ++q) s_same &= std::equal(sS[q].begin(), sS[q].end(), Sall.begin() + static_cast<size_t>(q) * g.slot_S);
  char what[512];
  std::snprintf(what, sizeof what, "%s segments (%d slots, 1-4 rows each, unequal segments, steps of up to 16 rows, skipped steps, partial commits): out and y == each sequence alone, bit for bit (%zu rows)",
                tag, NS, rows_cmp);
  expect(same, what);
  std::snprintf(what, sizeof what, "%s segments: every slot's committed S == its sequence alone, bit for bit", tag);
  expect(s_same, what);
  for (void *p : mem) f.release(p);
  free_card(f, d);
}

// segments of up to 8 rows on a state whose logs hold 8 rows (GdnState:log_rows, the 256-GRF core
// instances), as the DFlash drafter's 8-row verify of one sequence runs them, against the same accepted rows run as steps of
// at most 4 rows on a 4-row log (the 4-row core): every accepted row's out and y bit for bit, then the committed S after a
// final committing 1-row step. Rows past the accept count are computed and dropped (a rejected draft).
void test_rows8(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d = make_card(f, 0, m, 1);
  const auto &s = d.s;
  const int H = s.hidden, VW = s.val_w();
  struct P8 {
    int rows, accept;
  };
  const std::vector<P8> plan = {{1, 1}, {8, 6}, {8, 8}, {5, 2}, {8, 1}, {3, 3}, {8, 7}, {2, 1}, {7, 5}, {8, 4}, {6, 6}, {1, 1}};
  auto x8 = [&](uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(8ull * H);
    for (auto &v : x) v = bfr(nd(rng));
    return x;
  };
  std::vector<void *> mem;
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(0, std::max<size_t>(bytes, 64));
    mem.push_back(p);
    d.q->memset(p, 0, std::max<size_t>(bytes, 64)).wait();
    return p;
  };
  auto make_state = [&](int log_rows) {
    k::GdnState g;
    g.slot_S = k::gdn_S_elems(s);
    g.slot_win = k::gdn_win_elems(s);
    g.slot_y = static_cast<size_t>(log_rows) * d.w.qkvz.n;
    g.slot_ab = static_cast<size_t>(log_rows) * s.ab_n();
    g.log_rows = log_rows;
    g.S = static_cast<float *>(dev(g.slot_S * 4));
    for (int p = 0; p < 2; ++p) {
      g.win[p] = static_cast<bf16 *>(dev(g.slot_win * 2));
      g.y[p] = static_cast<float *>(dev(g.slot_y * 4));
      g.ab[p] = static_cast<float *>(dev(g.slot_ab * 4));
    }
    g.seg = static_cast<int32_t *>(dev(k::kMaxSegs * k::kSegRec * 4));
    return g;
  };
  k::GdnScratch sc = d.sc;
  sc.y = static_cast<float *>(dev(8ull * d.w.qkvz.n * 4));
  // one step of one sequence on records: rows, the previous step's accept, own step parity, commit on 1-row steps
  auto step = [&](k::GdnState g, int rows, int prev, int k_own, const std::vector<float> &x) {
    std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
    w[k::kSegAccept] = prev;
    w[k::kSegParity] = k_own & 1;
    d.q->memcpy(const_cast<int32_t *>(g.seg), w.data(), w.size() * 4).wait();
    const auto xb = bits(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    g.seg_rows = rows;
    g.accept_max = prev;
    g.commit = rows == 1;
    k::gdn_decode(*d.q, s, rows, d.x, d.w, g, 0, sc, nullptr, 0, d.out);
    d.q->wait_and_throw();
    return std::make_pair(rd16(*d.q, d.out, static_cast<size_t>(rows) * H), rd16(*d.q, d.sc.core, static_cast<size_t>(rows) * VW));
  };
  // A: the 8-row plan on an 8-row log
  k::GdnState ga = make_state(k::kGdnMaxRows);
  std::vector<float> aout, ay, rows_x;  // accepted rows only, and their inputs in order
  int prev = 0;
  for (size_t i = 0; i < plan.size(); ++i) {
    const auto x = x8(7000u + static_cast<uint32_t>(i));
    const auto [o, y] = step(ga, plan[i].rows, prev, static_cast<int>(i), x);
    const int a = plan[i].accept;
    aout.insert(aout.end(), o.begin(), o.begin() + static_cast<size_t>(a) * H);
    ay.insert(ay.end(), y.begin(), y.begin() + static_cast<size_t>(a) * VW);
    rows_x.insert(rows_x.end(), x.begin(), x.begin() + static_cast<size_t>(a) * H);
    prev = plan[i].rows == 1 ? 0 : a;  // a 1-row step committed its row
  }
  const auto xlast = x8(7999u);
  const auto fa = step(ga, 1, prev, static_cast<int>(plan.size()), xlast);
  const auto Sa = rd(*d.q, ga.S, ga.slot_S);
  // B: the same accepted rows as steps of <= 4 rows, each kept whole, on a 4-row log
  k::GdnState gb = make_state(k::kGdnRows);
  std::vector<float> bout, by;
  const int total = static_cast<int>(rows_x.size() / H);
  int kb = 0;
  prev = 0;
  for (int r0 = 0; r0 < total; r0 += 4, ++kb) {
    const int n = std::min(4, total - r0);
    std::vector<float> x(8ull * H, 0.f);
    std::copy(rows_x.begin() + static_cast<size_t>(r0) * H, rows_x.begin() + static_cast<size_t>(r0 + n) * H, x.begin());
    const auto [o, y] = step(gb, n, prev, kb, x);
    bout.insert(bout.end(), o.begin(), o.end());
    by.insert(by.end(), y.begin(), y.end());
    prev = n == 1 ? 0 : n;
  }
  const auto fb = step(gb, 1, prev, kb, xlast);
  const auto Sb = rd(*d.q, gb.S, gb.slot_S);
  char what[512];
  std::snprintf(what, sizeof what, "%s 8-row log: %zu steps of 1-8 rows, accepts 1-8 (%d accepted rows): out and y == the same rows in steps of <= 4 rows, bit for bit",
                tag, plan.size(), total);
  expect(aout == bout && ay == by, what);
  std::snprintf(what, sizeof what, "%s 8-row log: the committing 1-row step after it (out) and the committed S == the 4-row path, bit for bit", tag);
  expect(fa.first == fb.first && Sa == Sb, what);
  for (void *p : mem) f.release(p);
  free_card(f, d);
}

// Steps whose segment is a draft TREE (parents in the record, any topological order), the next step replaying the
// accepted root-to-node path through the record's path mask. Gate: every row's out and y == the last row of a chain
// step over its ancestors on a state that ran the accepted history as chain steps, and the committed S after a final
// committing 1-row step == the chain history's, bit for bit.
// wide = a 16-row log (kGdnWideRows) and trees of up to 16 rows (the 16-row tree core, 64-bit record words), the
// steps after them replaying up to 16 path rows; the chain references run in steps of <= 8 rows on an 8-row log.
void test_tree8(xenithium::Fabric &f, const Model &m, const char *tag, bool flag_chains = false, bool wide = false) {
  Card d = make_card(f, 0, m, 1);
  const auto &s = d.s;
  const int H = s.hidden, VW = s.val_w();
  const int LR = wide ? k::kGdnWideRows : k::kGdnMaxRows;  // the tree state's log rows
  struct T8 {
    std::vector<int> par;  // parent of each row (row 0: -1)
    int node;  // the accepted path ends here
  };
  const std::vector<T8> plan16 = {
      {{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 15}, {{-1, 0, 0, 2, 2, 4, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 15},
      {{-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 5, 10, 11, 12, 13, 14}, 9},  {{-1, 0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13}, 14},
      {{-1, 0, 1, 2, 0, 4, 1, 6}, 3},                                {{-1}, 0},
      {{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 4},  {{-1, 0, 0, 2, 3, 4, 2, 6, 7, 8, 9, 10}, 11},
      {{-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 10, 11, 8, 13, 14}, 12},  {{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 15},
      {{-1, 0}, 1},                                                  {{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 13},
      {{-1, 0, 1, 2, 3, 4, 5, 6}, 7}};
  const std::vector<T8> plan8 = {
      {{-1, 0, 1, 2, 0, 4, 1, 6}, 3}, {{-1, 0, 0, 0, 3, 3, 5, 0}, 6}, {{-1, 0, 1, 1, 3, 3}, 5}, {{-1, 0, 1, 2, 3, 4, 5, 6}, 7},
      {{-1, 0, 0, 0, 0}, 0},          {{-1, 0, 1, 0, 3, 4, 3}, 6},    {{-1}, 0},                {{-1, 0, 1, 2, 2, 1, 0, 6}, 7},
      {{-1, 0, 1, 2, 3, 4, 5, 4}, 5}, {{-1, 0, 0, 1, 1, 2, 2}, 2},    {{-1, 0}, 1},            {{-1, 0, 1, 2, 3, 0, 5, 6}, 7},
      {{-1, 0, 0, 2, 3, 3, 5, 5}, 7}, {{-1, 0, 0, 2, 2, 4, 5, 6}, 3}, {{-1, 0, 1, 1, 0, 4, 4, 1}, 6}, {{-1, 0, 0, 2, 2, 4, 4, 6}, 7},
      // a chain-shaped tree, a branch, two chain-shaped trees
      {{-1, 0, 1, 2, 3, 4, 5, 6}, 7}, {{-1, 0, 1, 2, 2, 4, 2, 6}, 5}, {{-1, 0, 1, 2, 3, 4, 5, 6}, 7}, {{-1, 0, 1, 2, 3, 4, 5, 6}, 2}};
  const std::vector<T8> &plan = wide ? plan16 : plan8;
  auto xr = [&](uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(static_cast<size_t>(LR) * H);
    for (auto &v : x) v = bfr(nd(rng));
    return x;
  };
  std::vector<void *> mem;
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(0, std::max<size_t>(bytes, 64));
    mem.push_back(p);
    d.q->memset(p, 0, std::max<size_t>(bytes, 64)).wait();
    return p;
  };
  auto make_state = [&](int lr = k::kGdnMaxRows) {
    k::GdnState g;
    g.slot_S = k::gdn_S_elems(s);
    g.slot_win = k::gdn_win_elems(s);
    g.slot_y = static_cast<size_t>(lr) * d.w.qkvz.n;
    g.slot_ab = static_cast<size_t>(lr) * s.ab_n();
    g.log_rows = lr;
    g.S = static_cast<float *>(dev(g.slot_S * 4));
    for (int p = 0; p < 2; ++p) {
      g.win[p] = static_cast<bf16 *>(dev(g.slot_win * 2));
      g.y[p] = static_cast<float *>(dev(g.slot_y * 4));
      g.ab[p] = static_cast<float *>(dev(g.slot_ab * 4));
    }
    g.seg = static_cast<int32_t *>(dev(k::kMaxSegs * k::kSegRec * 4));
    return g;
  };
  auto zero = [&](const k::GdnState &g) {
    d.q->memset(g.S, 0, g.slot_S * 4);
    for (int p = 0; p < 2; ++p) {
      d.q->memset(g.win[p], 0, g.slot_win * 2);
      d.q->memset(g.y[p], 0, g.slot_y * 4);
      d.q->memset(g.ab[p], 0, g.slot_ab * 4);
    }
    d.q->wait();
  };
  k::GdnScratch sc = d.sc;
  sc.y = static_cast<float *>(dev(static_cast<size_t>(k::kMaxRows) * d.w.qkvz.n * 4));
  // one step on records: rows, the previous step's accept (+ path mask), own parity, the tree word
  auto step = [&](k::GdnState g, int rows, int prev, uint64_t pmask, int k_own, uint64_t tw, const std::vector<float> &x, uint64_t aux = 0) {
    std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
    w[k::kSegAccept] = prev;
    w[k::kSegParity] = k_own & 1;
    k::rec_word_set(w.data(), k::kSegTree, k::kSegTreeHi, tw);
    k::rec_word_set(w.data(), k::kSegPath, k::kSegPathHi, pmask);
    k::rec_word_set(w.data(), k::kSegTreeAux, k::kSegTreeAux2, aux);
    d.q->memcpy(const_cast<int32_t *>(g.seg), w.data(), w.size() * 4).wait();
    const auto xb = bits(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    g.seg_rows = rows;
    g.accept_max = prev;
    g.commit = rows == 1;
    k::gdn_decode(*d.q, s, rows, d.x, d.w, g, 0, sc, nullptr, 0, d.out);
    d.q->wait_and_throw();
    return std::make_pair(rd16(*d.q, d.out, static_cast<size_t>(rows) * H), rd16(*d.q, d.sc.core, static_cast<size_t>(rows) * VW));
  };
  // a chain over `hist` (accepted rows, in steps of <= 8 rows, each kept whole) then `tail` rows: the last row's out and y
  k::GdnState gc = make_state();
  auto chain = [&](const std::vector<float> &hist, const std::vector<float> &tail, bool commit_tail, std::vector<float> *S) {
    zero(gc);
    const int total = static_cast<int>(hist.size() / H);
    int kc = 0, prev = 0;
    for (int r0 = 0; r0 < total; r0 += 8, ++kc) {
      const int n = std::min(8, total - r0);
      std::vector<float> x(8ull * H, 0.f);
      std::copy(hist.begin() + static_cast<size_t>(r0) * H, hist.begin() + static_cast<size_t>(r0 + n) * H, x.begin());
      step(gc, n, prev, 0, kc, 0, x);
      prev = n == 1 ? 0 : n;
    }
    int n = static_cast<int>(tail.size() / H), t0 = 0;
    while (n > 8) {  // a tail past 8 rows (a 16-row tree's deep row) in kept 8-row steps first
      std::vector<float> x(8ull * H, 0.f);
      std::copy(tail.begin() + static_cast<size_t>(t0) * H, tail.begin() + static_cast<size_t>(t0 + 8) * H, x.begin());
      step(gc, 8, prev, 0, kc++, 0, x);
      prev = 8, t0 += 8, n -= 8;
    }
    std::vector<float> x(8ull * H, 0.f);
    std::copy(tail.begin() + static_cast<size_t>(t0) * H, tail.end(), x.begin());
    if (commit_tail && n != 1) throw std::logic_error("test_tree8: a committing tail is one row");
    const auto [o, y] = step(gc, n, prev, 0, kc, 0, x);
    if (S) *S = rd(*d.q, gc.S, gc.slot_S);
    return std::make_pair(std::vector<float>(o.end() - H, o.end()), std::vector<float>(y.end() - VW, y.end()));
  };
  k::GdnState gt = make_state(LR);
  gt.tree = true;
  gt.tree_S = static_cast<float *>(dev(k::gdn_tree_S_elems(s) * 4));
  std::vector<float> hist;  // the accepted rows' inputs, in order
  int prev = 0, bad_rows = 0, rows_all = 0;
  uint64_t pmask = 0;
  for (size_t i = 0; i < plan.size(); ++i) {
    const auto &t = plan[i];
    const int R = static_cast<int>(t.par.size());
    uint64_t tw = 0;
    bool chain_only = true;
    for (int j = 1; j < R; ++j) tw |= static_cast<uint64_t>(t.par[static_cast<size_t>(j)]) << (4 * j), chain_only &= t.par[static_cast<size_t>(j)] == j - 1;
    // flag_chains: every multi-row step carries its tree words, chain-shaped or not (qwen35 Model:step_tree's words)
    const bool flagged = flag_chains ? R > 1 : !chain_only;
    if (flagged) tw |= k::kTreeFlag;
    const auto x = xr(9100u + static_cast<uint32_t>(i));
    const uint64_t aux = flagged ? k::tree_aux_word(t.par.data(), R) : 0ull;
    const auto [o, y] = step(gt, R, prev, pmask, static_cast<int>(i), tw, x, aux);
    for (int j = 0; j < R; ++j) {  // row j vs a chain over its ancestors
      std::vector<int> path;
      for (int a = j; a >= 0; a = t.par[static_cast<size_t>(a)]) path.insert(path.begin(), a);
      std::vector<float> tail;
      for (int a : path) tail.insert(tail.end(), x.begin() + static_cast<size_t>(a) * H, x.begin() + static_cast<size_t>(a + 1) * H);
      const auto ref = chain(hist, tail, false, nullptr);
      const bool same = std::equal(ref.first.begin(), ref.first.end(), o.begin() + static_cast<size_t>(j) * H) &&
                        std::equal(ref.second.begin(), ref.second.end(), y.begin() + static_cast<size_t>(j) * VW);
      bad_rows += !same;
      if (!same) {
        int no = 0, ny = 0;
        for (int e = 0; e < H; ++e) no += ref.first[static_cast<size_t>(e)] != o[static_cast<size_t>(j) * H + e];
        for (int e = 0; e < VW; ++e) ny += ref.second[static_cast<size_t>(e)] != y[static_cast<size_t>(j) * VW + e];
        std::printf("  tree step %zu row %d (aux %012llx): out %d / y %d values differ\n", i, j, static_cast<unsigned long long>(aux), no, ny);
      }
      ++rows_all;
    }
    std::vector<int> path;
    for (int a = t.node; a >= 0; a = t.par[static_cast<size_t>(a)]) path.insert(path.begin(), a);
    pmask = k::tree_path_word(path.data(), static_cast<int>(path.size()));  // 0 for a chain prefix: the plain accept count
    for (int a : path) hist.insert(hist.end(), x.begin() + static_cast<size_t>(a) * H, x.begin() + static_cast<size_t>(a + 1) * H);
    prev = R == 1 ? 0 : static_cast<int>(path.size());
    if (R == 1) pmask = 0;
  }
  const auto xlast = xr(9999u);
  const std::vector<float> x1(xlast.begin(), xlast.begin() + H);
  const auto fa = step(gt, 1, prev, pmask, static_cast<int>(plan.size()), 0, xlast);
  const auto Sa = rd(*d.q, gt.S, gt.slot_S);
  std::vector<float> Sb;
  const auto fb = chain(hist, x1, true, &Sb);
  char what[512];
  std::snprintf(what, sizeof what, "%s tree core: %zu tree steps (%d rows, %d accepted): every row's out and y == a chain over its ancestors, bit for bit (%d differ)",
                tag, plan.size(), rows_all, static_cast<int>(hist.size() / H), bad_rows);
  expect(bad_rows == 0, what);
  std::snprintf(what, sizeof what, "%s tree core: the committing 1-row step after the accepted paths (out) and the committed S == the chain's, bit for bit", tag);
  expect(std::equal(fb.first.begin(), fb.first.end(), fa.first.begin()) && Sa == Sb, what);
  // the tree core on chain records == the chain cores (1-row commits: the 4-row library's 1 / 1 instance; 4- and 8-row
  // verifies, partial accepts), every row, then S: a plain greedy decode and a chain verify under the tree core
  if (!wide) {
    zero(gc);
    zero(gt);
    struct Sp {
      int rows, keep;
    };
    const std::vector<Sp> seq = {{1, 1}, {1, 1}, {1, 1}, {8, 8}, {8, 3}, {1, 1}, {4, 4}, {4, 2}, {8, 8}, {1, 1}, {1, 1}, {8, 5}, {2, 2}, {1, 1}};
    int pc = 0, bad = 0;
    for (size_t i = 0; i < seq.size(); ++i) {
      const auto x = xr(7700u + static_cast<uint32_t>(i));
      const int n = seq[i].rows;
      const auto a = step(gc, n, pc, 0, static_cast<int>(i), 0, x);
      const auto b = step(gt, n, pc, 0, static_cast<int>(i), 0, x);
      bad += !(a.first == b.first && a.second == b.second);
      pc = n == 1 ? 0 : seq[i].keep;
    }
    bad += rd(*d.q, gc.S, gc.slot_S) != rd(*d.q, gt.S, gt.slot_S);
    std::snprintf(what, sizeof what, "%s tree core on chain records == the chain cores: 14 steps of 1 / 2 / 4 / 8 rows (1-row commits, partial accepts), out, y and S bit for bit (%d differ)", tag, bad);
    expect(bad == 0, what);
  }
  for (void *p : mem) f.release(p);
  free_card(f, d);
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const bool two = loader.size() > 1;
    xenithium::Fabric f(loader, two ? std::vector<size_t>{0, 1} : std::vector<size_t>{0});
    const Model small = make_model(512, 2, 8, 11), model = make_model(2560, 16, 48, 12);
    test_one(f, small, "small (h 512, 2 key / 8 value heads)");
    test_one(f, model, "model dims (h 2560, 16 key / 48 value heads)");
    {  // the swish output gate (y = norm(o) x silu(z)) through the same protocol
      Model swish = small;
      swish.s.swish = true;
      test_one(f, swish, "small, swish gate");
      if (f.size() > 1) test_cards(f, swish, "small, swish gate");
      for (int M : {37, 300}) test_rows(f, swish, "small, swish gate", M);  // gdn_rows + scan with swish
    }
    if (f.size() > 1) {
      test_cards(f, small, "small");
      test_cards(f, model, "model dims");
    }
    test_segments(f, small, "small");
    test_segments(f, model, "model dims");
    test_rows8(f, small, "small");
    test_rows8(f, model, "model dims");
    {
      Model swish = model;
      swish.s.swish = true;
      test_rows8(f, swish, "model dims, swish gate");
      test_tree8(f, swish, "model dims, swish gate");
      // the 27B's per-card GDN (hidden 5120, 8 key / 24 value heads: its own core instance)
      Model card = make_model(5120, 8, 24, 21);
      card.s.swish = true;
      test_tree8(f, card, "27B card dims (h 5120, 8 key / 24 value heads), swish gate");
      test_tree8(f, card, "27B card dims, swish gate, every step's tree words", true);
      // 16-row trees and 16-row replays (the 16-row tree core)
      test_tree8(f, card, "27B card dims, swish gate, 16-row trees", false, true);
      test_tree8(f, card, "27B card dims, swish gate, 16-row trees, every step's tree words", true, true);
    }
    test_tree8(f, small, "small");
    // qwen38's per-card GDN (hidden 2560, no swish gate) at TP2 (8 key / 24 value heads) and TP4 (4 / 12)
    test_tree8(f, make_model(2560, 8, 24, 31), "qwen38 TP2 card dims (h 2560, 8 key / 24 value heads)");
    test_tree8(f, make_model(2560, 4, 12, 32), "qwen38 TP4 card dims (h 2560, 4 key / 12 value heads)");
    test_tree8(f, make_model(2560, 4, 12, 32), "qwen38 TP4 card dims, every step's tree words", true);
    for (int M : {1, 37, 300}) test_rows(f, small, "small", M);
    test_rows(f, model, "model dims", 64);
    test_rows(f, model, "model dims", 1030);  // the 8 x 8 a | b blocks from 1,024 rows
    if (loader.size() >= 4) {  // TP4 (4 key / 12 value heads per card)
      xenithium::Fabric f4(loader, {0, 1, 2, 3});
      test_cards(f4, model, "model dims");
      Model swish = model;
      swish.s.swish = true;
      test_cards(f4, swish, "model dims, swish gate");
      for (int M : {64, 300}) test_rows(f4, model, "model dims", M);
    } else {
      std::printf("[SKIP] four-card GDN: %zu card(s) visible\n", loader.size());
    }
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::printf("FAILED: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("OK: 0 failure(s)\n");
  return 0;
}
