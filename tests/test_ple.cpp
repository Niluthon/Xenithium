// Gate for the PLE decode block (kernels/embed/ple.hpp): kv GEMM, gate (norms, signed-sqrt sigmoid,
// gated value, conv-input norm), the dilated conv over the position ring, the residual update,
// and the speculative-row protocol (rows are addressed by position: no accept word).
// Synthetic weights at the model's dims (hidden 2560, 4 streams, E 2560) and at a small shape,
// a plan of steps with 1-4 rows from position 0 (the conv start: taps before 0 are zero) to past
// the ring's wrap. The reference is teacher-forced stage by stage on the device's own values:
// kv: within 2^-7 max(|ref|, row rms) of the int8 GEMM reference, <= 1 % off
// gate: gated and the conv input from the device's kv and residual, within 2^-6 (each is
// rounded twice in a chain: g, then g v; the norm, then x) of max(|ref|, row rms),
// <= 0.5 % off
// conv: the residual from the device's conv inputs (the ring) and gated, 2^-6, <= 0.5 % off
// protocol: "4 rows at p, continue at p + c" == "c rows at p, continue at p + c", bit for bit
// run-to-run and two cards: bit-identical
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/embed/ple.hpp"
#include "xenithium/kernels/embed/ple_ref.hpp"
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
  double scaled = 0, lim = 0;
  size_t off = 0, n = 0, pct = 1;
  BoundCmp(double l, size_t p) : lim(l), pct(p) {}
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
  bool ok() const { return scaled <= lim && off * 1000 <= n * pct; }
  std::string text() const {
    char b[128];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms), %zu / %zu differ (%.3f %%)", scaled, off, n, 100.0 * off / std::max<size_t>(1, n));
    return b;
  }
};

struct Model {
  k::PleShape s;
  std::vector<float> kv;  // [kv_n][E] float weights
  std::vector<float> nk, nq, ncw, cw;  // bf16 values
};
Model make_model(int H, int E, unsigned seed) {
  Model m;
  m.s = k::PleShape{H, 4, E, 1e-6f};
  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  const int C = m.s.width();
  m.kv.resize(static_cast<size_t>(m.s.kv_n()) * E);
  for (auto &x : m.kv) x = bfr(n01(rng) * 0.02f);
  auto vec = [&](size_t n, float sc) {
    std::vector<float> v(n);
    for (auto &x : v) x = bfr(n01(rng) * sc);
    return v;
  };
  m.nk = vec(C, 0.2f);
  m.nq = vec(C, 0.2f);
  m.ncw = vec(C, 0.2f);
  m.cw = vec(static_cast<size_t>(C) * k::kPleTaps, 0.4f);
  return m;
}

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

struct Card {
  sycl::queue *q = nullptr;
  k::PleWeights w;
  std::vector<float> deq;  // dequantised kv [kv_n][E]
  k::PleState st;
  k::PleScratch sc;
  int32_t *pos = nullptr;
  bf16 *res = nullptr;
  std::vector<void *> mem;
};

Card make_card(xenithium::Fabric &f, int c, const Model &m) {
  Card d;
  d.q = &f.compute(c);
  const int n = m.s.kv_n(), E = m.s.emb, C = m.s.width();
  std::vector<int8_t> codes(static_cast<size_t>(n) * E);
  std::vector<float> sc(n);
  const auto wb = bits(m.kv);
  k::quantize_i8_rows(wb.data(), n, E, codes.data(), sc.data(), 1);
  d.deq.resize(codes.size());
  for (int r = 0; r < n; ++r)
    for (int j = 0; j < E; ++j) d.deq[static_cast<size_t>(r) * E + j] = codes[static_cast<size_t>(r) * E + j] * sc[r];
  std::vector<int8_t> packed(k::dense_packed_elems(n, E));
  k::pack_dense_i8(codes.data(), n, E, packed.data());
  d.w.kv = k::DenseWeights{up(f, c, packed, d.mem), up(f, c, sc, d.mem), k::DenseKind::i8, n, E};
  d.w.nk = reinterpret_cast<const bf16 *>(up(f, c, bits(m.nk), d.mem));
  d.w.nq = reinterpret_cast<const bf16 *>(up(f, c, bits(m.nq), d.mem));
  d.w.ncw = reinterpret_cast<const bf16 *>(up(f, c, bits(m.ncw), d.mem));
  d.w.conv = reinterpret_cast<const bf16 *>(up(f, c, bits(m.cw), d.mem));
  auto alloc = [&](size_t bytes) {
    void *p = f.alloc(c, bytes);
    d.q->memset(p, 0, bytes).wait();
    d.mem.push_back(p);
    return p;
  };
  d.st.ring = static_cast<bf16 *>(alloc(k::ple_ring_elems(m.s) * 2));
  d.q->memset(d.st.ring, 0x7f, k::ple_ring_elems(m.s) * 2).wait();  // garbage (NaN-ish): positions < 0 must not be read
  d.sc.emb = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kMaxRows) * E * 2));
  d.sc.kv = static_cast<float *>(alloc(static_cast<size_t>(k::kMaxRows) * n * 4));
  d.sc.gated = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kMaxRows) * C * 2));
  d.pos = static_cast<int32_t *>(alloc(64));
  d.res = static_cast<bf16 *>(alloc(static_cast<size_t>(k::kMaxRows) * C * 2));
  return d;
}
void free_card(xenithium::Fabric &f, Card &d) {
  for (auto *p : d.mem) f.release(p);
  d.mem.clear();
}

// Inputs of one step: rows of embeddings and residuals (bf16 bits), random.
struct StepIn {
  int pos = 0, rows = 0;
  std::vector<uint16_t> emb, res;
};
StepIn make_step(const Model &m, int pos, int rows, std::mt19937 &rng) {
  StepIn s;
  s.pos = pos;
  s.rows = rows;
  std::normal_distribution<float> n01(0.f, 1.f);
  s.emb.resize(static_cast<size_t>(rows) * m.s.emb);
  for (auto &x : s.emb) x = k::bf16_bits(n01(rng) * 0.3f);
  s.res.resize(static_cast<size_t>(k::kDenseRows) * m.s.width(), 0);
  for (size_t i = 0; i < static_cast<size_t>(rows) * m.s.width(); ++i) s.res[i] = k::bf16_bits(n01(rng) * 0.7f);
  return s;
}

// Runs a step on a card; returns the residual after it.
std::vector<uint16_t> run(Card &d, const Model &m, const StepIn &in, uint16_t *pinned) {
  std::memcpy(pinned, in.emb.data(), in.emb.size() * 2);
  d.q->memcpy(d.pos, &in.pos, 4).wait();
  d.q->memcpy(d.res, in.res.data(), in.res.size() * 2).wait();
  k::ple_decode(*d.q, m.s, in.rows, reinterpret_cast<const bf16 *>(pinned), d.w, d.st, d.pos, d.sc, d.res).conv.wait();
  std::vector<uint16_t> out(in.res.size());
  d.q->memcpy(out.data(), d.res, out.size() * 2).wait();
  return out;
}

std::vector<float> vals(const uint16_t *b, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
  return v;
}

void test_one(xenithium::Fabric &f, const Model &m, const char *tag) {
  const int C = m.s.width(), H = m.s.hidden, E = m.s.emb, KV = m.s.kv_n();
  Card d = make_card(f, 0, m);
  auto *pinned = static_cast<uint16_t *>(f.alloc_host(static_cast<size_t>(k::kDenseRows) * E * 2));
  std::mt19937 rng(77);
  // Plan: 1, 2, 3, 4, 1, 4, 4, 2, 1, 3 rows from 0 (25 positions: wraps the 16-slot ring).
  const int plan[] = {1, 2, 3, 4, 1, 4, 4, 2, 1, 3};
  std::vector<std::vector<float>> x_hist;  // the device's conv inputs by position
  BoundCmp kvc(1.0 / 128, 10), gc(1.0 / 64, 5), xc(1.0 / 64, 5), oc(1.0 / 64, 5);
  int pos = 0;
  std::vector<StepIn> steps;
  std::vector<std::vector<uint16_t>> outs;
  for (const int R : plan) {
    const StepIn in = make_step(m, pos, R, rng);
    const auto out = run(d, m, in, pinned);
    steps.push_back(in);
    outs.push_back(out);
    std::vector<float> kvd(static_cast<size_t>(k::kDenseRows) * KV);
    std::vector<uint16_t> gd(static_cast<size_t>(k::kDenseRows) * C), ring(k::ple_ring_elems(m.s));
    d.q->memcpy(kvd.data(), d.sc.kv, kvd.size() * 4).wait();
    d.q->memcpy(gd.data(), d.sc.gated, gd.size() * 2).wait();
    d.q->memcpy(ring.data(), d.st.ring, ring.size() * 2).wait();
    for (int r = 0; r < R; ++r) {
      const int p = pos + r;
      // kv vs the int8 reference
      std::vector<double> ref(KV);
      const auto ev = vals(in.emb.data() + static_cast<size_t>(r) * E, E);
      for (int n = 0; n < KV; ++n) {
        double acc = 0;
        for (int j = 0; j < E; ++j) acc += static_cast<double>(d.deq[static_cast<size_t>(n) * E + j]) * ev[j];
        ref[n] = acc;
      }
      // fp32 GEMM output: count flips of its bf16 rounding (what the block consumes)
      std::vector<float> kvr(&kvd[static_cast<size_t>(r) * KV], &kvd[static_cast<size_t>(r) * KV] + KV);
      for (auto &v : kvr) v = bfr(v);
      kvc.add_row(kvr.data(), ref.data(), C);
      kvc.add_row(kvr.data() + C, ref.data() + C, H);
      // gate from the device's kv (bf16) and the residual
      std::vector<float> key(C), val(H), g(C), x(C);
      for (int i = 0; i < C; ++i) key[i] = bfr(kvd[static_cast<size_t>(r) * KV + i]);
      for (int i = 0; i < H; ++i) val[i] = bfr(kvd[static_cast<size_t>(r) * KV + C + i]);
      const auto rv = vals(in.res.data() + static_cast<size_t>(r) * C, C);
      k::ple_ref_gate(H, 4, key.data(), val.data(), rv.data(), m.nk.data(), m.nq.data(), m.ncw.data(), m.s.eps, g.data(), x.data());
      const auto gdev = vals(gd.data() + static_cast<size_t>(r) * C, C), xdev = vals(ring.data() + static_cast<size_t>(p % k::kPleRing) * C, C);
      for (int s = 0; s < 4; ++s) {
        std::vector<double> gr(g.begin() + s * H, g.begin() + (s + 1) * H), xr(x.begin() + s * H, x.begin() + (s + 1) * H);
        gc.add_row(gdev.data() + s * H, gr.data(), H);
        xc.add_row(xdev.data() + s * H, xr.data(), H);
      }
      if (static_cast<int>(x_hist.size()) <= p) x_hist.resize(p + 1);
      x_hist[p] = xdev;
      // conv + residual from the device's ring and gated
      const float *taps[k::kPleTaps];
      for (int j = 0; j < k::kPleTaps; ++j) {
        const int t = p - k::kPleDil * (k::kPleTaps - 1 - j);
        taps[j] = t >= 0 ? x_hist[t].data() : nullptr;
      }
      std::vector<float> o = rv;
      k::ple_ref_conv(C, taps, m.cw.data(), gdev.data(), o.data());
      const auto odev = vals(out.data() + static_cast<size_t>(r) * C, C);
      std::vector<double> od(o.begin(), o.end());
      oc.add_row(odev.data(), od.data(), C);
    }
    // dead rows untouched
    bool dead = std::equal(out.begin() + static_cast<size_t>(R) * C, out.end(), in.res.begin() + static_cast<size_t>(R) * C);
    if (!dead) expect(false, std::string(tag) + " rows past `rows` left unchanged");
    pos += R;
  }
  expect(kvc.ok(), std::string(tag) + " kv GEMM vs the int8 reference: " + kvc.text());
  expect(gc.ok(), std::string(tag) + " gated vs the reference from the device's kv: " + gc.text());
  expect(xc.ok(), std::string(tag) + " conv input (ring) vs the reference: " + xc.text());
  expect(oc.ok(), std::string(tag) + " residual vs conv + add from the device's ring and gated (10 steps, 25 positions, ring wrapped): " + oc.text());

  // Run-to-run: a fresh ring (garbage-filled), the same steps, bit-identical outputs.
  {
    d.q->memset(d.st.ring, 0x3c, k::ple_ring_elems(m.s) * 2).wait();
    bool same = true;
    for (size_t i = 0; i < steps.size(); ++i) same &= run(d, m, steps[i], pinned) == outs[i];
    expect(same, std::string(tag) + " run-to-run (ring refilled with other garbage): bit-identical");
  }
  // Protocol: 4 speculative rows at p, continue at p + c == c rows at p, continue at p + c.
  {
    bool same = true;
    const int p0 = pos;
    for (int c = 1; c <= 3; ++c) {
      StepIn spec = make_step(m, p0, 4, rng), next = make_step(m, p0 + c, 3, rng);
      // run A: 4 rows, then the next step
      std::vector<uint16_t> ring0(k::ple_ring_elems(m.s));
      d.q->memcpy(ring0.data(), d.st.ring, ring0.size() * 2).wait();
      run(d, m, spec, pinned);
      const auto a = run(d, m, next, pinned);
      // run B: the same history, c rows only
      d.q->memcpy(d.st.ring, ring0.data(), ring0.size() * 2).wait();
      StepIn cr = spec;
      cr.rows = c;
      cr.emb.resize(static_cast<size_t>(c) * E);
      std::fill(cr.res.begin() + static_cast<size_t>(c) * C, cr.res.end(), 0);
      run(d, m, cr, pinned);
      const auto b = run(d, m, next, pinned);
      same &= a == b;
      d.q->memcpy(d.st.ring, ring0.data(), ring0.size() * 2).wait();
    }
    expect(same, std::string(tag) + " 4 speculative rows, continue at p + c == c rows (c = 1..3): bit-identical");
  }
  f.release(pinned);
  free_card(f, d);
}

void test_two(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d[2] = {make_card(f, 0, m), make_card(f, 1, m)};
  auto *pinned = static_cast<uint16_t *>(f.alloc_host(static_cast<size_t>(k::kDenseRows) * m.s.emb * 2));
  std::mt19937 rng(5);
  bool same = true;
  int pos = 0;
  for (const int R : {4, 4, 1, 3, 4}) {
    const StepIn in = make_step(m, pos, R, rng);
    same &= run(d[0], m, in, pinned) == run(d[1], m, in, pinned);
    pos += R;
  }
  expect(same, std::string(tag) + " two cards (each its own copy): bit-identical residuals, 5 steps");
  f.release(pinned);
  free_card(f, d[0]);
  free_card(f, d[1]);
}

// Compare prefill with 4-row decode slices using the same KV accumulation order. Prefill sums K in
// order, while decode can auto-select split-K and round a projection differently. Pin decode to one
// split so the gate and convolution can be checked bit for bit. The chunk reads the ring left by
// two decode steps, and the first decode step after the chunk reads the ring left by prefill.
void test_rows(xenithium::Fabric &f, const Model &m, const char *tag, int M) {
  const int C = m.s.width(), E = m.s.emb;
  Card d = make_card(f, 0, m);
  auto *pinned = static_cast<uint16_t *>(f.alloc_host(static_cast<size_t>(M + 4) * E * 2));
  k::PleRowsScratch rs;
  auto alloc = [&](size_t bytes) {
    void *p = f.alloc(0, std::max<size_t>(bytes, 64));
    d.mem.push_back(p);
    return p;
  };
  rs.emb = static_cast<bf16 *>(alloc(static_cast<size_t>(M) * E * 2));
  rs.kv = static_cast<float *>(alloc(static_cast<size_t>(M) * d.w.kv.n * 4));
  rs.gated = static_cast<bf16 *>(alloc(static_cast<size_t>(M) * C * 2));
  rs.x = static_cast<bf16 *>(alloc(static_cast<size_t>(M) * C * 2));
  auto *res = static_cast<bf16 *>(alloc(static_cast<size_t>(M + 4) * C * 2));
  std::mt19937 rng(77u + M);
  const StepIn h0 = make_step(m, 0, 4, rng), h1 = make_step(m, 4, 2, rng);
  const int p0 = 6;
  std::vector<uint16_t> emb(static_cast<size_t>(M) * E), r0(static_cast<size_t>(M + 4) * C, 0);
  std::normal_distribution<float> n01(0.f, 1.f);
  for (auto &x : emb) x = k::bf16_bits(n01(rng) * 0.3f);
  for (size_t i = 0; i < static_cast<size_t>(M) * C; ++i) r0[i] = k::bf16_bits(n01(rng) * 0.7f);
  const StepIn after = make_step(m, p0 + M, 4, rng);
  auto path = [&](bool slices, std::vector<uint16_t> &chunk, std::vector<uint16_t> &next) {
    d.q->memset(d.st.ring, 0x7f, k::ple_ring_elems(m.s) * 2).wait();
    run(d, m, h0, pinned);
    run(d, m, h1, pinned);
    d.q->memcpy(res, r0.data(), r0.size() * 2).wait();
    std::memcpy(pinned, emb.data(), emb.size() * 2);
    if (slices) {
      for (int sl = 0; sl * 4 < M; ++sl) {
        const int rows = std::min(4, M - 4 * sl), pos = p0 + 4 * sl;
        d.q->memcpy(d.pos, &pos, 4).wait();
        k::ple_decode(*d.q, m.s, rows, reinterpret_cast<const bf16 *>(pinned + static_cast<size_t>(sl) * 4 * E), d.w, d.st, d.pos, d.sc,
                      res + static_cast<size_t>(sl) * 4 * C, /*kv_ks=*/1)
            .conv.wait();
      }
    } else {
      k::ple_rows(*d.q, m.s, M, p0, reinterpret_cast<const bf16 *>(pinned), d.w, d.st, rs, res).conv.wait();
    }
    chunk.resize(static_cast<size_t>(M) * C);
    d.q->memcpy(chunk.data(), res, chunk.size() * 2).wait();
    next = run(d, m, after, pinned);
  };
  std::vector<uint16_t> ca, na, cb, nb;
  path(true, ca, na);
  path(false, cb, nb);
  char what[256];
  std::snprintf(what, sizeof what, "%s ple_rows %d rows at position 6: residuals == the decode path on 4-row slices, bit for bit", tag, M);
  expect(ca == cb, what);
  std::snprintf(what, sizeof what, "%s ple_rows %d rows: the next decode step (reads the ring the rows left) bit-identical", tag, M);
  expect(na == nb, what);
  f.release(pinned);
  free_card(f, d);
}

// Several sequences in one step (segment records, rows.hpp) against each sequence alone,
// bit for bit: slots with their own rings at different positions (wrapping the 16-slot ring),
// skipped steps, 1-4 rows per segment, partial commits (a rejected row's ring entry is rewritten).
void test_segments(xenithium::Fabric &f, const Model &m, const char *tag) {
  constexpr int NS = 16;  // 16 slots (16-row steps of one-row segments)
  const int C = m.s.width(), E = m.s.emb;
  Card d = make_card(f, 0, m);
  auto *pinned = static_cast<uint16_t *>(f.alloc_host(static_cast<size_t>(k::kMaxRows) * E * 2));
  const size_t RE = k::ple_ring_elems(m.s);
  auto *ring = static_cast<bf16 *>(f.alloc(0, NS * RE * 2));
  auto *rec = static_cast<int32_t *>(f.alloc(0, k::kMaxSegs * k::kSegRec * 4));
  struct SegStep {
    std::vector<int> seqs;
    int rows;
    std::vector<int> rr = {};  // rows per segment when they differ (seg_het: B rows, then 1-row segments)
    int of(size_t j) const { return rr.empty() ? rows : rr[j]; }
    int all() const {
      int n = 0;
      for (size_t j = 0; j < seqs.size(); ++j) n += of(j);
      return n;
    }
    int code() const { return rr.empty() ? rows : k::seg_het(rr[0], static_cast<int>(std::count(rr.begin(), rr.end(), rr[0])), static_cast<int>(seqs.size())); }
  };
  const std::vector<SegStep> plan = {{{0, 1, 2}, 1}, {{0, 1}, 2}, {{2}, 4}, {{1, 0}, 2}, {{0, 1, 2}, 1}, {{2, 0}, 2}, {{1}, 3}, {{0, 1, 2}, 1},
                                     {{0, 2}, 2},    {{1, 2}, 2}, {{2, 1, 0}, 1}, {{0}, 4}, {{1, 0}, 2},    {{2}, 3},    {{0, 1, 2}, 1},
                                     {{0, 1}, 4},    {{2, 0, 1}, 2}, {{1, 2}, 3}, {{2, 0}, 4},  // 5-8-row steps
                                     {{0, 1, 2}, 0, {3, 3, 1}}, {{2, 0}, 0, {4, 1}}, {{1, 2, 0}, 0, {4, 1, 1}}, {{0, 1, 2}, 0, {2, 2, 1}},
                                     // 9-16 rows (the 16-row shape)
                                     {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1}, {{15, 3, 7, 1, 9, 0, 12, 5}, 2},
                                     {{4, 13, 2, 10}, 4}, {{6, 11, 14}, 4}, {{8, 0, 1, 2, 3, 4, 5, 6, 7, 9}, 0, {4, 4, 1, 1, 1, 1, 1, 1, 1, 1}},
                                     {{2, 9, 15, 0, 11, 13}, 0, {3, 3, 3, 1, 1, 1}}, {{14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15}, 1}, {{1, 0, 2}, 3}};
  auto commit_of = [](int k2, int rows) { return k2 % 2 == 0 ? rows : std::max(1, rows - 1); };
  auto in_of = [&](int q, int k2, int pos, int rows) {
    std::mt19937 rng(5000u + 100u * static_cast<uint32_t>(q) + static_cast<uint32_t>(k2));
    return make_step(m, pos, rows, rng);
  };
  const int start[NS] = {0, 5, 20, 1, 2, 3, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37};
  auto clear = [&]() { d.q->memset(ring, 0x7f, NS * RE * 2).wait(); };  // garbage: positions < 0 must not be read
  std::vector<std::vector<std::vector<uint16_t>>> solo(NS);
  clear();
  for (int q = 0; q < NS; ++q) {
    k::PleState st;
    st.ring = ring + static_cast<size_t>(q) * RE;
    int pos = start[q], k2 = 0;
    for (const auto &sp : plan) {
      const auto at = std::find(sp.seqs.begin(), sp.seqs.end(), q);
      if (at == sp.seqs.end()) continue;
      const int rq = sp.of(static_cast<size_t>(at - sp.seqs.begin()));
      const StepIn in = in_of(q, k2, pos, rq);
      std::memcpy(pinned, in.emb.data(), in.emb.size() * 2);
      d.q->memcpy(d.pos, &pos, 4).wait();
      d.q->memcpy(d.res, in.res.data(), in.res.size() * 2).wait();
      k::ple_decode(*d.q, m.s, rq, reinterpret_cast<const bf16 *>(pinned), d.w, st, d.pos, d.sc, d.res).conv.wait();
      solo[q].push_back(std::vector<uint16_t>(static_cast<size_t>(rq) * C));
      d.q->memcpy(solo[q].back().data(), d.res, solo[q].back().size() * 2).wait();
      pos += commit_of(k2, rq);
      ++k2;
    }
  }
  clear();
  k::PleState all;
  all.ring = ring;
  all.slot_elems = RE;
  int pos[NS], own[NS] = {};
  std::copy(start, start + NS, pos);
  bool same = true;
  size_t rows_cmp = 0;
  for (const auto &sp : plan) {
    const int ns = static_cast<int>(sp.seqs.size()), R = sp.all();
    std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
    std::vector<uint16_t> res(static_cast<size_t>(k::kMaxRows) * C, 0);
    for (int j = 0, r0 = 0; j < ns; r0 += sp.of(static_cast<size_t>(j)), ++j) {
      const int q = sp.seqs[static_cast<size_t>(j)], rq = sp.of(static_cast<size_t>(j));
      w[static_cast<size_t>(j) * k::kSegRec + k::kSegPos] = pos[q];
      w[static_cast<size_t>(j) * k::kSegRec + k::kSegSlot] = q;
      const StepIn in = in_of(q, own[q], pos[q], rq);
      std::memcpy(pinned + static_cast<size_t>(r0) * E, in.emb.data(), in.emb.size() * 2);
      std::copy(in.res.begin(), in.res.begin() + static_cast<size_t>(rq) * C, res.begin() + static_cast<size_t>(r0) * C);
    }
    d.q->memcpy(rec, w.data(), w.size() * 4).wait();
    d.q->memcpy(d.res, res.data(), res.size() * 2).wait();
    k::ple_decode(*d.q, m.s, R, reinterpret_cast<const bf16 *>(pinned), d.w, all, rec, d.sc, d.res, 0, k::kSegRec, sp.code()).conv.wait();
    std::vector<uint16_t> out(static_cast<size_t>(R) * C);
    d.q->memcpy(out.data(), d.res, out.size() * 2).wait();
    for (int j = 0, r0 = 0; j < ns; r0 += sp.of(static_cast<size_t>(j)), ++j) {
      const int q = sp.seqs[static_cast<size_t>(j)], rq = sp.of(static_cast<size_t>(j));
      same &= std::equal(out.begin() + static_cast<size_t>(r0) * C, out.begin() + static_cast<size_t>(r0 + rq) * C, solo[q][own[q]].begin());
      rows_cmp += rq;
      pos[q] += commit_of(own[q], rq);
      ++own[q];
    }
  }
  char what[512];
  std::snprintf(what, sizeof what, "%s segments (%d slots, own rings, 1-4 rows each, unequal segments, up to 16 a step, skipped steps, partial commits): residuals == each sequence alone, bit for bit "
                "(%zu rows)", tag, NS, rows_cmp);
  expect(same, what);
  f.release(ring);
  f.release(rec);
  f.release(pinned);
  free_card(f, d);
}

// A draft tree's rows (records with kSegTree, rows.hpp) against chains of their paths, bit for bit:
// row j of the tree == row depth(j) of a chain step of j's path (root.. j) alone from the same ring; after the accept
// of a path and ple_move, the next steps (1 and 4 rows) == those after the chain step of the path. Positions wrap the
// 32-slot ring; trees of 8 rows (a chain as a tree, wide at the root, deep with side rows).
void test_trees(xenithium::Fabric &f, const Model &m, const char *tag) {
  const int C = m.s.width(), E = m.s.emb;
  Card d = make_card(f, 0, m);
  auto *pinned = static_cast<uint16_t *>(f.alloc_host(static_cast<size_t>(k::kMaxRows) * E * 2));
  const size_t RE = k::ple_ring_elems(m.s);
  auto *snap = static_cast<bf16 *>(f.alloc(0, RE * 2));
  auto *rec = static_cast<int32_t *>(f.alloc(0, k::kSegRec * 4));
  struct Case {
    int p0;
    std::vector<int> par, path;
  };
  const std::vector<Case> cases = {
      {13, {-1, 0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3}},  // a chain written as a tree (moves nothing)
      {30, {-1, 0, 1, 2, 0, 4, 1, 6}, {0, 4, 5}},  // wraps the ring
      {9, {-1, 0, 0, 0, 1, 2, 3, 4}, {0, 3, 6}},
      {40, {-1, 0, 1, 2, 3, 4, 5, 1}, {0, 1, 7}},
      {63, {-1, 0, 1, 1, 2, 4, 5, 6}, {0, 1, 3}},
      {21, {-1, 0, 1, 1, 3, 4, 5, 6}, {0, 1, 3, 4, 5, 6, 7}},
  };
  auto run = [&](int pos, int rows, const std::vector<uint16_t> &emb, const std::vector<uint16_t> &res, const std::vector<int32_t> *w, bool tree) {
    std::memcpy(pinned, emb.data(), emb.size() * 2);
    d.q->memcpy(d.res, res.data(), res.size() * 2).wait();
    if (w) {
      d.q->memcpy(rec, w->data(), w->size() * 4).wait();
      k::ple_decode(*d.q, m.s, rows, reinterpret_cast<const bf16 *>(pinned), d.w, d.st, rec, d.sc, d.res, 0, k::kSegRec, 0, tree).conv.wait();
    } else {
      d.q->memcpy(d.pos, &pos, 4).wait();
      k::ple_decode(*d.q, m.s, rows, reinterpret_cast<const bf16 *>(pinned), d.w, d.st, d.pos, d.sc, d.res).conv.wait();
    }
    std::vector<uint16_t> out(static_cast<size_t>(rows) * C);
    d.q->memcpy(out.data(), d.res, out.size() * 2).wait();
    return out;
  };
  bool rows_same = true, next_same = true;
  int nrows = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const Case &cs = cases[ci];
    const int R = static_cast<int>(cs.par.size()), n = static_cast<int>(cs.path.size());
    d.q->memset(d.st.ring, 0x7f, RE * 2).wait();
    std::mt19937 rng(7100u + static_cast<uint32_t>(ci));
    for (int p = 0; p < cs.p0; p += 4) {  // history: 4-row steps up to p0
      const int rr = std::min(4, cs.p0 - p);
      const StepIn in = make_step(m, p, rr, rng);
      run(p, rr, in.emb, in.res, nullptr, false);
    }
    d.q->memcpy(snap, d.st.ring, RE * 2).wait();
    StepIn tin;  // the tree's R rows (make_step keeps kDenseRows residual rows)
    {
      std::normal_distribution<float> n01(0.f, 1.f);
      tin.emb.resize(static_cast<size_t>(R) * E);
      for (auto &x : tin.emb) x = k::bf16_bits(n01(rng) * 0.3f);
      tin.res.resize(static_cast<size_t>(R) * C);
      for (auto &x : tin.res) x = k::bf16_bits(n01(rng) * 0.7f);
    }
    std::vector<int32_t> w(k::kSegRec, 0);
    w[k::kSegPos] = cs.p0;
    k::rec_word_set(w.data(), k::kSegTree, k::kSegTreeHi, k::tree_word(cs.par.data(), R));
    const std::vector<uint16_t> tout = run(cs.p0, R, tin.emb, tin.res, &w, true);
    auto chain_of = [&](const std::vector<int> &rows) {  // the inputs of rows `rows` of the tree, in order
      std::vector<uint16_t> emb, res;
      for (int r : rows) {
        emb.insert(emb.end(), tin.emb.begin() + static_cast<size_t>(r) * E, tin.emb.begin() + static_cast<size_t>(r + 1) * E);
        res.insert(res.end(), tin.res.begin() + static_cast<size_t>(r) * C, tin.res.begin() + static_cast<size_t>(r + 1) * C);
      }
      return std::make_pair(emb, res);
    };
    // the next steps after the path (both ways)
    const StepIn n1 = make_step(m, cs.p0 + n, 1, rng), n4 = make_step(m, cs.p0 + n + 1, 4, rng);
    k::ple_move(*d.q, m.s, d.st, 0, cs.p0, cs.path.data(), n).wait();
    const std::vector<uint16_t> ta = run(cs.p0 + n, 1, n1.emb, n1.res, nullptr, false), tb = run(cs.p0 + n + 1, 4, n4.emb, n4.res, nullptr, false);
    for (int j = 0; j < R; ++j) {
      std::vector<int> pth;
      for (int x = j; x >= 0; x = cs.par[static_cast<size_t>(x)]) pth.insert(pth.begin(), x);
      d.q->memcpy(d.st.ring, snap, RE * 2).wait();
      const auto in = chain_of(pth);
      const int L = static_cast<int>(pth.size());
      const std::vector<uint16_t> cout = run(cs.p0, L, in.first, in.second, nullptr, false);
      rows_same &= std::equal(tout.begin() + static_cast<size_t>(j) * C, tout.begin() + static_cast<size_t>(j + 1) * C, cout.begin() + static_cast<size_t>(L - 1) * C);
      ++nrows;
    }
    d.q->memcpy(d.st.ring, snap, RE * 2).wait();
    const auto pin = chain_of(cs.path);
    run(cs.p0, n, pin.first, pin.second, nullptr, false);
    const std::vector<uint16_t> ca = run(cs.p0 + n, 1, n1.emb, n1.res, nullptr, false), cb = run(cs.p0 + n + 1, 4, n4.emb, n4.res, nullptr, false);
    next_same &= ta == ca && tb == cb;
  }
  char what[512];
  std::snprintf(what, sizeof what, "%s trees (%zu trees of 8 rows, positions wrapping the ring): each row == the last row of its path's chain step, bit for bit (%d rows)", tag,
                cases.size(), nrows);
  expect(rows_same, what);
  std::snprintf(what, sizeof what, "%s trees: after ple_move of the accepted path the next 1- and 4-row steps == those after the path's chain step", tag);
  expect(next_same, what);
  f.release(snap);
  f.release(rec);
  f.release(pinned);
  free_card(f, d);
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const bool two = loader.size() > 1;
    xenithium::Fabric f(loader, two ? std::vector<size_t>{0, 1} : std::vector<size_t>{0});
    const Model small = make_model(512, 512, 3), model = make_model(2560, 2560, 4);
    test_one(f, small, "small (h 512, E 512)");
    test_one(f, model, "model dims (h 2560, E 2560)");
    if (f.size() > 1) test_two(f, model, "model dims");
    test_segments(f, small, "small");
    test_segments(f, model, "model dims");
    test_trees(f, small, "small");
    test_trees(f, model, "model dims");
    for (int M : {1, 13, 300}) test_rows(f, small, "small", M);
    test_rows(f, model, "model dims", 64);
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
