// Gate for QSA decode (kernels/attention/qsa.hpp; the host-only rotary / YaRN / int8 code checks are in
// test_qsa_host.cpp): the exact top-k select (one launch and chunked + merge, to 262,144 blocks), and the whole decode block over
// a history past the 2,048-token budget, teacher-forced stage by stage:
// q, iq vs norm + rope of the device's projections: 1 ulp flips only (<= 1 % of elements)
// K / V the rows written this step, dequantised, vs the reference from the device's projections
// kc blocks closed this step vs the reference compression of the raw keys
// score every complete block vs the reference from the device's iq and kc (fp32 sum order)
// select == the reference top-k of the device's keys, exactly (ids and count)
// attn vs softmax(q K / 16) V over the device's selection, K / V as stored (2 ulps: bf16(o)
// then the gate)
// out vs the reference o_proj of the device's attn (2^-7)
// and bit-exact identities: run-to-run; identity page table == a permuted table in a pool twice the
// size (unused pages poisoned); an eager pos_bound == the capacity bound; "R rows, then continue at
// pos + c" == "c rows" (speculative rows need no accept word); two cards: both cards' out
// bit-identical, each card's attn == the one-card attn of its heads, same selection. Draft-tree steps
// == chain steps of their paths, and qsa_kv_move (test_trees).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/attention/qsa.hpp"
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
const char *kv_name(k::QsaKv kv) { return kv == k::QsaKv::i8 ? "int8" : kv == k::QsaKv::i8v4 ? "int8-v4" : "bf16"; }

struct BoundCmp {
  double scaled = 0, w_dev = 0, w_ref = 0;
  size_t off = 0, n = 0;
  void add_row(const float *dev, const double *ref, int len) {
    double ss = 0;
    for (int i = 0; i < len; ++i) ss += ref[i] * ref[i];
    const double rms = std::sqrt(ss / len);
    for (int i = 0; i < len; ++i) {
      ++n;
      off += dev[i] != bfr(static_cast<float>(ref[i]));
      const double sc = std::fabs(dev[i] - ref[i]) / std::max({std::fabs(ref[i]), rms, 1e-30});
      if (!(sc <= scaled)) {
        scaled = sc;
        w_dev = dev[i];
        w_ref = ref[i];
      }
    }
  }
  void add_rowf(const float *dev, const float *ref, int len) {
    std::vector<double> r(ref, ref + len);
    add_row(dev, r.data(), len);
  }
  bool ok(double bound, double frac) const { return scaled <= bound && off <= frac * n; }
  std::string text() const {
    char b[160];
    std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms) (dev %.6g ref %.6g), %zu / %zu differ (%.3f %%)", scaled, w_dev, w_ref, off, n,
                  100.0 * off / std::max<size_t>(1, n));
    return b;
  }
};

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
std::vector<uint16_t> bits(const std::vector<float> &v) {
  std::vector<uint16_t> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = k::bf16_bits(v[i]);
  return b;
}

// ---------------------------------------------------------------- select
void test_select(xenithium::Fabric &f) {
  auto &q = f.compute(0);
  const int cap = k::kQsaMaxBlocks;
  k::QsaScratch sc{};
  sc.score = static_cast<uint32_t *>(f.alloc(0, 4ull * cap * 4));
  sc.cand_key = static_cast<uint32_t *>(f.alloc(0, 4ull * 16 * 512 * 4));
  sc.cand_id = static_cast<int32_t *>(f.alloc(0, 4ull * 16 * 512 * 4));
  sc.cand_n = static_cast<int32_t *>(f.alloc(0, 4096));
  sc.sel = static_cast<int32_t *>(f.alloc(0, 4ull * 512 * 4));
  sc.sel_n = static_cast<int32_t *>(f.alloc(0, 4096));
  int32_t *nb = static_cast<int32_t *>(f.alloc(0, 4096));
  std::mt19937 rng(9);
  struct Case {
    int n[4];
    int kind;  // 0 random floats, 1 few distinct values (ties), 2 all equal, 3 one binade, 4 zeros + few
  };
  const std::vector<Case> cases = {{{100, 512, 513, 1}, 0},           {{2048, 3000, 777, 600}, 1},       {{5000, 4096, 1000, 513}, 2},
                                   {{16384, 9000, 16000, 12000}, 3},  {{16384, 16384, 16384, 16384}, 0}, {{16385, 20000, 513, 30000}, 1},
                                   {{65536, 40000, 65000, 1}, 3},      {{131072, 100000, 2, 70000}, 4},  {{262144, 262144, 200000, 262143}, 0},
                                   {{262144, 150000, 262144, 9000}, 1},
                                   // chunks of max(4,096, bound / 16) around the one-launch edge, early stops
                                   {{4096, 4097, 8192, 2049}, 0},      {{4097, 32768, 12000, 8191}, 1},  {{65536, 65535, 4100, 50000}, 0}};
  bool ok = true;
  for (const auto &cs : cases) {
    const int bound = *std::max_element(cs.n, cs.n + 4);
    std::vector<uint32_t> keys(4ull * cap);
    for (int r = 0; r < 4; ++r)
      for (int i = 0; i < cs.n[r]; ++i) {
        float v;
        switch (cs.kind) {
          case 0: v = std::uniform_real_distribution<float>(0.f, 50.f)(rng); break;
          case 1: v = static_cast<float>(rng() % 7); break;
          case 2: v = 3.5f; break;
          case 3: v = 8.f + std::uniform_real_distribution<float>(0.f, 1e-3f)(rng); break;
          default: v = (rng() % 1000 == 0) ? 1.f + (rng() % 3) : 0.f; break;
        }
        std::memcpy(&keys[static_cast<size_t>(r) * cap + i], &v, 4);
      }
    q.memcpy(sc.score, keys.data(), keys.size() * 4).wait();
    q.memcpy(nb, cs.n, 16).wait();
    for (const int chunk : {0, k::kQsaChunk}) {  // decode's chunks and prefill's 16,384
      q.memset(sc.sel, 0xA5, 4ull * 512 * 4).wait();
      k::qsa_select(q, 4, sc.score, cap, nb, bound, sc, nullptr, chunk).wait_and_throw();
      q.wait_and_throw();
      const auto sel = rd(q, sc.sel, 4ull * 512);
      const auto seln = rd(q, sc.sel_n, 4);
      bool c_ok = true;
      for (int r = 0; r < 4; ++r) {
        const auto want = k::qsa_ref_select(&keys[static_cast<size_t>(r) * cap], cs.n[r], 512);
        c_ok &= seln[r] == static_cast<int>(want.size()) && std::equal(want.begin(), want.end(), sel.begin() + r * 512);
      }
      if (!c_ok) std::printf("  select mismatch: n = %d %d %d %d kind %d chunk %d\n", cs.n[0], cs.n[1], cs.n[2], cs.n[3], cs.kind, chunk);
      ok &= c_ok;
    }
  }
  expect(ok, "select: top-512 == reference (ties to the lowest block, ascending) for 1 .. 262,144 blocks: random, 7 values, all equal, one binade, "
             "mostly zero; one launch and chunked + merge");
  for (void *p : {(void *)sc.score, (void *)sc.cand_key, (void *)sc.cand_id, (void *)sc.cand_n, (void *)sc.sel, (void *)sc.sel_n, (void *)nb})
    f.release(p);
}

// ---------------------------------------------------------------- the decode block
struct Model {
  k::QsaShape s;  // cards = 1
  std::vector<float> qkv;  // [24 x (q 256 | gate 256) | k 2 x 256 | v 2 x 256][H]
  std::vector<float> idx;  // [640][H]
  std::vector<float> o;  // [H][24 x 256]
  std::vector<float> qn, kn, iqn, ikn;
};
Model make_model(int H, uint32_t seed) {
  Model m;
  m.s = k::QsaShape{H, 24, 2, 1, 0, 1e-6f, k::QsaKv::i8};
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto fill = [&](std::vector<float> &v, size_t n, float sd) {
    v.resize(n);
    for (auto &x : v) x = bfr(nd(rng) * sd);
  };
  const float sw = 1.f / std::sqrt(static_cast<float>(H));
  fill(m.qkv, static_cast<size_t>(m.s.qkv_n()) * H, sw);
  fill(m.idx, static_cast<size_t>(m.s.idx_n()) * H, sw);
  fill(m.o, static_cast<size_t>(H) * m.s.o_k(), 1.f / std::sqrt(static_cast<float>(m.s.o_k())));
  fill(m.qn, 256, 0.2f);
  fill(m.kn, 256, 0.2f);
  fill(m.iqn, 128, 0.2f);
  fill(m.ikn, 128, 0.2f);
  return m;
}

struct Card {
  sycl::queue *q = nullptr;
  k::QsaShape s;
  k::QsaWeights w;
  std::vector<float> qkv_deq, o_deq;  // reference weights of this card
  k::QsaCache c;
  k::QsaScratch sc;
  k::PairWire wire;
  bf16 *rope = nullptr, *x = nullptr, *out = nullptr;
  int32_t *pos = nullptr;
  std::vector<int32_t> pages;  // host copy of the page table
  int pool_pages = 0;
  std::vector<void *> mem;
  void *dev(xenithium::Fabric &f, int card, size_t bytes) {
    void *p = f.alloc(card, std::max<size_t>(bytes, 64));
    mem.push_back(p);
    return p;
  }
};
template <typename T>
T *up(xenithium::Fabric &f, int c, const std::vector<T> &v, std::vector<void *> &mem) {
  auto *p = static_cast<T *>(f.alloc(c, std::max<size_t>(64, v.size() * sizeof(T))));
  f.compute(c).memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  mem.push_back(p);
  return p;
}
k::DenseWeights i8_matrix(xenithium::Fabric &f, int c, const std::vector<float> &w, int n, int kk, std::vector<float> &deq, std::vector<void *> &mem) {
  const auto wb = bits(w);
  std::vector<int8_t> codes(wb.size());
  std::vector<float> sc(n);
  k::quantize_i8_rows(wb.data(), n, kk, codes.data(), sc.data(), 1);
  deq.resize(static_cast<size_t>(n) * kk);
  for (int r = 0; r < n; ++r)
    for (int j = 0; j < kk; ++j) deq[static_cast<size_t>(r) * kk + j] = codes[static_cast<size_t>(r) * kk + j] * sc[r];
  std::vector<int8_t> packed(k::dense_packed_elems(n, kk));
  k::pack_dense_i8(codes.data(), n, kk, packed.data());
  return k::DenseWeights{up(f, c, packed, mem), up(f, c, sc, mem), k::DenseKind::i8, n, kk};
}

constexpr int kCap = 4096;  // tokens of the test cache

// pool_mul: physical pages = pool_mul x logical; perm: random page table (else identity); cap: tokens per slot.
Card make_card(xenithium::Fabric &f, int c, const Model &m, int cards, k::QsaKv kv, int pool_mul, bool perm, uint32_t seed, int cap = kCap) {
  Card d;
  d.q = &f.compute(c);
  d.s = m.s;
  d.s.cards = cards;
  d.s.card = c;
  d.s.kv = kv;
  const auto &s = d.s;
  const int H = s.hidden, D = 256, hq = s.hq(), hkv = s.hkv();
  std::vector<float> qkv;
  auto rows_of = [&](int r0, int n) { qkv.insert(qkv.end(), m.qkv.begin() + static_cast<size_t>(r0) * H, m.qkv.begin() + static_cast<size_t>(r0 + n) * H); };
  rows_of(c * hq * 2 * D, hq * 2 * D);
  rows_of(m.s.q_n() + s.kv0() * D, hkv * D);  // this card's kv heads (TP4's pair of cards share one)
  rows_of(m.s.q_n() + m.s.kv_heads * D + s.kv0() * D, hkv * D);
  d.w.qkv = i8_matrix(f, c, qkv, s.qkv_n(), H, d.qkv_deq, d.mem);
  {
    std::vector<uint16_t> packed(k::dense_packed_elems(s.idx_n(), H));
    k::pack_dense_bf16(bits(m.idx).data(), s.idx_n(), H, packed.data());
    d.w.idx = k::DenseWeights{up(f, c, packed, d.mem), nullptr, k::DenseKind::bf16, s.idx_n(), H};
  }
  std::vector<float> o(static_cast<size_t>(H) * s.o_k());
  for (int n = 0; n < H; ++n)
    std::memcpy(&o[static_cast<size_t>(n) * s.o_k()], &m.o[static_cast<size_t>(n) * m.s.o_k() + c * s.o_k()], s.o_k() * 4ull);
  d.w.o = i8_matrix(f, c, o, H, s.o_k(), d.o_deq, d.mem);
  d.w.q_norm = reinterpret_cast<const bf16 *>(up(f, c, bits(m.qn), d.mem));
  d.w.k_norm = reinterpret_cast<const bf16 *>(up(f, c, bits(m.kn), d.mem));
  d.w.iq_norm = reinterpret_cast<const bf16 *>(up(f, c, bits(m.iqn), d.mem));
  d.w.ik_norm = reinterpret_cast<const bf16 *>(up(f, c, bits(m.ikn), d.mem));
  // cache: logical pages -> physical pages of a pool
  const int lp = cap / k::kQsaPage;
  d.pool_pages = lp * pool_mul;
  d.pages.resize(lp);
  std::iota(d.pages.begin(), d.pages.end(), 0);
  if (perm) {
    std::vector<int32_t> all(d.pool_pages);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), std::mt19937(seed));
    std::copy(all.begin(), all.begin() + lp, d.pages.begin());
  }
  const size_t P = d.pool_pages;
  d.c.k = static_cast<uint8_t *>(d.dev(f, c, P * k::qsa_page_kv_bytes(s)));
  d.c.v = static_cast<uint8_t *>(d.dev(f, c, P * k::qsa_page_v_bytes(s)));
  d.c.ks = static_cast<float *>(d.dev(f, c, P * k::qsa_page_scale_floats(s) * 4));
  d.c.vs = static_cast<float *>(d.dev(f, c, P * k::qsa_page_vscale_floats(s) * 4));
  d.c.kc = static_cast<bf16 *>(d.dev(f, c, P * k::qsa_page_kc_elems() * 2));
  d.c.ring = static_cast<bf16 *>(d.dev(f, c, k::kQsaRing * 128 * 2));
  d.c.pages = up(f, c, d.pages, d.mem);
  d.c.max_tokens = cap;
  d.q->memset(d.c.k, 0xA5, P * k::qsa_page_kv_bytes(s));
  d.q->memset(d.c.v, 0xA5, P * k::qsa_page_v_bytes(s));
  d.q->memset(d.c.ks, 0xA5, P * k::qsa_page_scale_floats(s) * 4);
  d.q->memset(d.c.vs, 0xA5, P * k::qsa_page_vscale_floats(s) * 4);
  d.q->memset(d.c.kc, 0xA5, P * k::qsa_page_kc_elems() * 2);
  d.q->memset(d.c.ring, 0xA5, k::kQsaRing * 128 * 2).wait();
  auto &sc = d.sc;
  constexpr size_t kR = k::kStepRows32;  // the 32-row shape's rows
  sc.y = static_cast<float *>(d.dev(f, c, kR * d.w.qkv.n * 4));
  sc.yi = static_cast<float *>(d.dev(f, c, kR * d.w.idx.n * 4));
  sc.q = static_cast<bf16 *>(d.dev(f, c, kR * hq * D * 2));
  sc.iq = static_cast<bf16 *>(d.dev(f, c, kR * 4 * 128 * 2));
  sc.nblk = static_cast<int32_t *>(d.dev(f, c, kR * 4));
  sc.score = static_cast<uint32_t *>(d.dev(f, c, k::qsa_score_words(cap, static_cast<int>(kR)) * 4));
  sc.cand_key = static_cast<uint32_t *>(d.dev(f, c, kR * 16 * 512 * 4));
  sc.cand_id = static_cast<int32_t *>(d.dev(f, c, kR * 16 * 512 * 4));
  sc.cand_n = static_cast<int32_t *>(d.dev(f, c, 4096));
  sc.sel = static_cast<int32_t *>(d.dev(f, c, kR * 512 * 4));
  sc.sel_n = static_cast<int32_t *>(d.dev(f, c, kR * 4));
  sc.part = static_cast<float *>(d.dev(f, c, k::qsa_part_floats(s, static_cast<int>(kR)) * 4));
  sc.attn = static_cast<bf16 *>(d.dev(f, c, kR * s.o_k() * 2));
  sc.partial = static_cast<float *>(d.dev(f, c, kR * H * 4));
  d.q->memset(sc.attn, 0, kR * s.o_k() * 2).wait();
  d.q->memset(sc.y, 0, kR * d.w.qkv.n * 4).wait();
  d.q->memset(sc.yi, 0, kR * d.w.idx.n * 4).wait();
  d.rope = reinterpret_cast<bf16 *>(up(f, c, k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{}), d.mem));
  d.x = static_cast<bf16 *>(d.dev(f, c, kR * H * 2));
  d.out = static_cast<bf16 *>(d.dev(f, c, kR * H * 2));
  d.pos = static_cast<int32_t *>(d.dev(f, c, 64));
  if (cards > 1) {
    d.wire.inbox = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_wire_words(H, cards) * 4));
    d.mem.push_back(d.wire.inbox);
    d.wire.error = static_cast<uint32_t *>(d.dev(f, c, 4096));
    k::pair_wire_init(*d.q, H, d.wire.inbox, cards).wait();
    d.q->memset(d.wire.error, 0, 4).wait();
  }
  return d;
}
void free_card(xenithium::Fabric &f, Card &d) {
  for (void *p : d.mem) f.release(p);
  d.mem.clear();
}

struct Step {
  int pos, rows, bound;  // bound: 0 = eager (pos + rows), else the cache capacity
};
// Mostly 4-row steps (accept 4) to pass 2,048 tokens fast, with partial accepts and short steps.
std::vector<Step> make_plan(int target, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<Step> v;
  int pos = 0;
  while (pos < target) {
    const int kind = static_cast<int>(rng() % 8);
    const int rows = kind == 0 ? 1 : (kind == 1 ? 2 + static_cast<int>(rng() % 2) : 4);
    int acc = kind == 2 ? static_cast<int>(rng() % rows) : rows;  // some steps keep fewer rows
    if (pos + rows > kCap) break;
    v.push_back({pos, rows, (rng() % 3 == 0) ? 1 : 0});
    pos += acc;
  }
  return v;
}
std::vector<float> make_x(int H, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(4ull * H);
  for (auto &v : x) v = bfr(nd(rng));
  return x;
}

bool g_every = false;  // run_step asks for QsaStep:every wherever the step's bound allows it
void run_step(Card &d, const Step &st, const std::vector<uint16_t> &xb, const k::PairWire *wire, int wire_parity, int qks) {
  d.q->memcpy(d.pos, &st.pos, 4);
  d.q->memcpy(d.x, xb.data(), xb.size() * 2);
  k::QsaStep step{d.pos, st.bound ? kCap : st.pos + st.rows};
  step.every = g_every && step.pos_bound <= k::kQsaTopBlocks * k::kQsaCompress;
  k::qsa_decode(*d.q, d.s, st.rows, d.x, d.w, d.c, d.rope, step, d.sc, wire, wire_parity, d.out, qks);
}

struct Trace {
  std::vector<std::vector<uint16_t>> out, attn;
  std::vector<std::vector<int32_t>> sel;
};
// One card through the plan (host wait each step); `check` = steps to verify against the reference.
Trace run_plan(Card &d, const Model &m, const std::vector<Step> &plan, uint32_t seed, int qks, const std::vector<int> &check, BoundCmp *cmp,
               bool *sel_ok, double *score_err, BoundCmp *kvc) {
  Trace t;
  const auto &s = d.s;
  const int H = s.hidden, hq = s.hq(), hkv = s.hkv(), D = 256, ldy = d.w.qkv.n, ldi = d.w.idx.n;
  const bool i8 = k::qsa_kv_scaled(s.kv), v4 = k::qsa_kv_v4(s.kv);  // int8 K (and V, or int4 V)
  std::vector<std::vector<float>> raw(kCap, std::vector<float>(128));  // latest raw index key per position
  const auto rope = k::qsa_rope_table(kCap, 64, 1e7, k::QsaYarn{});
  std::vector<float> qnw(m.qn), knw(m.kn), iqw(m.iqn), ikw(m.ikn);
  for (size_t i = 0; i < plan.size(); ++i) {
    const auto &st = plan[i];
    const auto xb = bits(make_x(H, seed + static_cast<uint32_t>(i)));
    run_step(d, st, xb, nullptr, 0, qks);
    d.q->wait_and_throw();
    t.out.push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), static_cast<size_t>(st.rows) * H));
    t.attn.push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.sc.attn), static_cast<size_t>(st.rows) * s.o_k()));
    auto sel = rd(*d.q, d.sc.sel, 4ull * 512);
    const auto seln = rd(*d.q, d.sc.sel_n, 4);
    std::vector<int32_t> sv;
    for (int r = 0; r < st.rows; ++r) {
      sv.push_back(seln[r]);
      sv.insert(sv.end(), sel.begin() + r * 512, sel.begin() + r * 512 + seln[r]);
    }
    t.sel.push_back(sv);
    const auto yi = rd(*d.q, d.sc.yi, 4ull * ldi);
    for (int r = 0; r < st.rows; ++r)
      for (int j = 0; j < 128; ++j) raw[st.pos + r][j] = bfr(yi[static_cast<size_t>(r) * ldi + 512 + j]);
    if (!cmp || !std::binary_search(check.begin(), check.end(), static_cast<int>(i))) continue;
    // ---- teacher-forced check of this step ----
    const auto y = rd(*d.q, d.sc.y, 4ull * ldy);
    const auto qd = rd16(*d.q, d.sc.q, 4ull * hq * D), iqd = rd16(*d.q, d.sc.iq, 4ull * 512);
    const auto attn = rd16(*d.q, d.sc.attn, 4ull * s.o_k()), out = rd16(*d.q, d.out, 4ull * H);
    const auto keys = rd(*d.q, d.sc.score, k::qsa_score_words(kCap));
    const int stride = static_cast<int>(k::qsa_score_words(kCap) / 4);
    const size_t P = d.pool_pages;
    const auto kpool = rd(*d.q, d.c.k, P * k::qsa_page_kv_bytes(s)), vpool = rd(*d.q, d.c.v, P * k::qsa_page_v_bytes(s));
    const auto kspool = rd(*d.q, d.c.ks, P * k::qsa_page_scale_floats(s)), vspool = rd(*d.q, d.c.vs, P * k::qsa_page_vscale_floats(s));
    const auto kcpool = rd16(*d.q, d.c.kc, P * k::qsa_page_kc_elems());
    auto slot_of = [&](int tok, int kh) { return (static_cast<size_t>(d.pages[tok / 64]) * hkv + kh) * 64 + tok % 64; };
    auto kv_val = [&](const std::vector<uint8_t> &pool, const std::vector<float> &sp, size_t slot, int dd) {
      if (i8) return static_cast<float>(static_cast<int8_t>(pool[slot * D + dd])) * sp[slot];
      uint16_t b;
      std::memcpy(&b, &pool[(slot * D + dd) * 2], 2);
      return k::bf16_value(b);
    };
    auto v_val = [&](size_t slot, int dd) {  // int8-v4: two int4 codes per byte, dim 2j low
      if (!v4) return kv_val(vpool, vspool, slot, dd);
      const int byte = vpool[slot * (D / 2) + dd / 2], nib = dd & 1 ? byte >> 4 : byte & 0xF;
      uint16_t sb;  // int8-v4: bf16 scales, one per 32 dims
      std::memcpy(&sb, reinterpret_cast<const uint8_t *>(vspool.data()) + (slot * k::kQsaV4Scales + dd / k::kQsaV4Group) * 2, 2);
      return static_cast<float>(nib >= 8 ? nib - 16 : nib) * k::bf16_value(sb);
    };
    auto kc_of = [&](int b) { return &kcpool[(static_cast<size_t>(d.pages[b / 16]) * 16 + b % 16) * 128]; };
    for (int r = 0; r < st.rows; ++r) {
      const int p = st.pos + r;
      const uint16_t *rr = &rope[static_cast<size_t>(p) * 64];
      std::vector<float> v(D), o(D);
      for (int h = 0; h < hq; ++h) {  // q
        for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + j]);
        k::qsa_ref_norm_rope(v.data(), qnw.data(), D, rr, s.eps, o.data());
        cmp[0].add_rowf(&qd[(static_cast<size_t>(r) * hq + h) * D], o.data(), D);
      }
      for (int h = 0; h < 4; ++h) {  // iq
        for (int j = 0; j < 128; ++j) v[j] = bfr(yi[static_cast<size_t>(r) * ldi + h * 128 + j]);
        k::qsa_ref_norm_rope(v.data(), iqw.data(), 128, rr, s.eps, o.data());
        cmp[1].add_rowf(&iqd[(static_cast<size_t>(r) * 4 + h) * 128], o.data(), 128);
      }
      for (int kh = 0; kh < hkv; ++kh) {  // K / V of this row, as stored
        for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + kh * D + j]);
        k::qsa_ref_norm_rope(v.data(), knw.data(), D, rr, s.eps, o.data());
        std::vector<float> vv(D), kref(D), vref(D), kdev(D), vdev(D);
        for (int j = 0; j < D; ++j) vv[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + (hkv + kh) * D + j]);
        if (i8) {
          std::vector<int8_t> cd(D);
          float sc;
          k::qsa_ref_i8(o.data(), D, cd.data(), &sc);
          for (int j = 0; j < D; ++j) kref[j] = static_cast<float>(cd[j]) * sc;
          if (v4) {
            uint16_t gs[k::kQsaV4Scales];
            k::qsa_ref_v4(vv.data(), cd.data(), gs);
            for (int j = 0; j < D; ++j) vref[j] = static_cast<float>(cd[j]) * k::bf16_value(gs[j / k::kQsaV4Group]);
          } else {
            k::qsa_ref_i8(vv.data(), D, cd.data(), &sc);
            for (int j = 0; j < D; ++j) vref[j] = static_cast<float>(cd[j]) * sc;
          }
        } else {
          kref = o;
          vref = vv;
        }
        const size_t sl = slot_of(p, kh);
        for (int j = 0; j < D; ++j) {
          kdev[j] = kv_val(kpool, kspool, sl, j);
          vdev[j] = v_val(sl, j);
        }
        kvc[0].add_rowf(kdev.data(), kref.data(), D);
        kvc[1].add_rowf(vdev.data(), vref.data(), D);
      }
      if ((p + 1) % 4 == 0) {  // block closed by this row
        const int b = p / 4;
        std::vector<float> raw4(512);
        for (int j = 0; j < 4; ++j) std::copy(raw[b * 4 + j].begin(), raw[b * 4 + j].end(), raw4.begin() + j * 128);
        k::qsa_ref_compress(raw4.data(), ikw.data(), &rope[static_cast<size_t>(b * 4) * 64], s.eps, o.data());
        cmp[2].add_rowf(kc_of(b), o.data(), 128);
      }
      // scores and selection
      const int nblk = (p + 1) / 4;
      for (int b = 0; b < nblk; ++b) {
        float dv;
        std::memcpy(&dv, &keys[static_cast<size_t>(r) * stride + b], 4);
        const double ref = k::qsa_ref_score(&iqd[static_cast<size_t>(r) * 512], kc_of(b));
        double mag = 0;  // sum of |products|: the scale of fp32 sum-order error
        for (int j = 0; j < 512; ++j) mag += std::fabs(static_cast<double>(iqd[static_cast<size_t>(r) * 512 + j]) * kc_of(b)[j % 128]);
        *score_err = std::max(*score_err, std::fabs(dv - ref) / (mag / std::sqrt(128.0) + 1e-30));
      }
      const auto want = k::qsa_ref_select(&keys[static_cast<size_t>(r) * stride], nblk, 512);
      const std::vector<int> got(sel.begin() + r * 512, sel.begin() + r * 512 + seln[r]);
      *sel_ok &= got == want;
      // attention over the device's selection, K / V as stored
      const auto toks = k::qsa_ref_tokens(got, p);
      std::vector<double> part(H, 0.0);
      for (int kh = 0; kh < hkv; ++kh) {
        std::vector<float> K(toks.size() * D), V(toks.size() * D);
        for (size_t t2 = 0; t2 < toks.size(); ++t2)
          for (int j = 0; j < D; ++j) {
            K[t2 * D + j] = kv_val(kpool, kspool, slot_of(toks[t2], kh), j);
            V[t2 * D + j] = v_val(slot_of(toks[t2], kh), j);
          }
        for (int hh = 0; hh < s.grp(); ++hh) {
          const int h = kh * s.grp() + hh;
          std::vector<double> oo(D);
          k::qsa_ref_attend(&qd[(static_cast<size_t>(r) * hq + h) * D], static_cast<int>(toks.size()), K.data(), V.data(), oo.data());
          std::vector<float> gate(D), ga(D);
          for (int j = 0; j < D; ++j) gate[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + D + j]);
          k::qsa_ref_gate(oo.data(), gate.data(), ga.data());
          cmp[3].add_rowf(&attn[(static_cast<size_t>(r) * hq + h) * D], ga.data(), D);
        }
      }
      for (int n = 0; n < H; ++n) {
        double a = 0;
        for (int j = 0; j < s.o_k(); ++j) a += static_cast<double>(attn[static_cast<size_t>(r) * s.o_k() + j]) * d.o_deq[static_cast<size_t>(n) * s.o_k() + j];
        part[n] = a;
      }
      cmp[4].add_row(&out[static_cast<size_t>(r) * H], part.data(), H);
    }
  }
  return t;
}

void test_decode(xenithium::Fabric &f, const Model &m, k::QsaKv kv, const char *tag) {
  const auto plan = make_plan(2400, 77);
  // checked steps: the first ones, around the budget (2,048 tokens = 512 blocks) and the last ones
  std::vector<int> check;
  for (size_t i = 0; i < plan.size(); ++i) {
    const int p = plan[i].pos;
    if (i < 6 || i + 4 >= plan.size() || (p >= 2040 && p <= 2056)) check.push_back(static_cast<int>(i));
  }
  Card d = make_card(f, 0, m, 1, kv, 1, false, 0);
  BoundCmp cmp[5], kvc[2];
  bool sel_ok = true;
  double serr = 0;
  const Trace t = run_plan(d, m, plan, 1000, 0, check, cmp, &sel_ok, &serr, kvc);
  char what[600];
  const bool i8 = k::qsa_kv_scaled(kv);
  std::printf("%s, %s KV: %zu steps to position %d, %zu checked\n", tag, kv_name(kv), plan.size(), plan.back().pos + plan.back().rows,
              check.size());
  std::snprintf(what, sizeof what, "%s q (norm + rope) vs reference from the device's projections: %s", tag, cmp[0].text().c_str());
  expect(cmp[0].ok(1.0 / 128, 0.01), what);
  std::snprintf(what, sizeof what, "%s iq: %s", tag, cmp[1].text().c_str());
  expect(cmp[1].ok(1.0 / 128, 0.01), what);
  std::snprintf(what, sizeof what, "%s K as stored vs reference: %s; V: %s", tag, kvc[0].text().c_str(), kvc[1].text().c_str());
  // K: the device's normed key differs from the host's in a bf16 rounding now and then, which can move
  // a code by one step; V (no norm) is exact for every dtype. int8 code x scale is not a bf16 value, so
  // BoundCmp's "differ from bf16(ref)" count is no measure there: the bound alone decides.
  const double kv_frac = i8 ? 1.0 : 0.0;
  expect(kvc[0].ok(i8 ? 1.0 / 16 : 1.0 / 128, i8 ? 1.0 : 0.01) && kvc[1].ok(1e-9, kv_frac), what);
  std::snprintf(what, sizeof what, "%s compressed keys vs reference from the raw keys: %s", tag, cmp[2].text().c_str());
  expect(cmp[2].ok(1.0 / 64, 0.01), what);
  std::snprintf(what, sizeof what, "%s scores vs reference from the device's iq / kc: worst error %.2e of sum |products| (fp32 order)", tag, serr);
  expect(serr <= 1e-6, what);
  std::snprintf(what, sizeof what, "%s selection == reference top-k of the device's keys, every checked row (ids and count)", tag);
  expect(sel_ok, what);
  std::snprintf(what, sizeof what, "%s attn (gated) vs reference over the device's selection: %s", tag, cmp[3].text().c_str());
  expect(cmp[3].ok(1.0 / 64, 0.005), what);
  std::snprintf(what, sizeof what, "%s out vs reference o_proj of the device's attn: %s", tag, cmp[4].text().c_str());
  expect(cmp[4].ok(1.0 / 128, 0.01), what);

  // determinism, and a permuted page table in a pool twice the size (poisoned) == identity
  const Trace t2 = run_plan(d, m, plan, 1000, 0, {}, nullptr, nullptr, nullptr, nullptr);
  std::snprintf(what, sizeof what, "%s run-to-run bit-identical (out, attn, selection)", tag);
  expect(t2.out == t.out && t2.attn == t.attn && t2.sel == t.sel, what);
  free_card(f, d);
  Card dp = make_card(f, 0, m, 1, kv, 2, true, 4242);
  const Trace tp = run_plan(dp, m, plan, 1000, 0, {}, nullptr, nullptr, nullptr, nullptr);
  std::snprintf(what, sizeof what, "%s permuted page table (pool x2, unused pages 0xA5) == identity table, bit for bit", tag);
  expect(tp.out == t.out && tp.attn == t.attn && tp.sel == t.sel, what);
  free_card(f, dp);

  // speculative rows: "4 rows, continue at pos + c" == "c rows" at every later step, c = 1..3
  bool proto = true;
  Card dq = make_card(f, 0, m, 1, kv, 1, false, 0);
  for (int cc = 1; cc <= 3; ++cc) {
    std::vector<Step> pa, pb;
    int pos = 0;
    for (int i = 0; i < 160; ++i) {  // past 512 tokens... and one speculative step in the middle
      const int rows = 4;
      const int acc = i == 80 ? cc : 4;
      pa.push_back({pos, rows, 0});
      pb.push_back({pos, i == 80 ? cc : rows, 0});
      pos += acc;
    }
    const Trace ta = run_plan(dq, m, pa, 5000, 0, {}, nullptr, nullptr, nullptr, nullptr);
    const Trace tb = run_plan(dq, m, pb, 5000, 0, {}, nullptr, nullptr, nullptr, nullptr);
    bool same = true;
    for (size_t i = 0; i < pa.size(); ++i) {
      const size_t n = static_cast<size_t>(std::min(pa[i].rows, pb[i].rows)) * m.s.hidden;
      same &= std::equal(ta.out[i].begin(), ta.out[i].begin() + n, tb.out[i].begin());
    }
    if (!same) std::printf("  protocol differs at c = %d\n", cc);
    proto &= same;
  }
  free_card(f, dq);
  std::snprintf(what, sizeof what, "%s 4 speculative rows then continue at pos + c == c rows (c = 1..3), every later step bit for bit", tag);
  expect(proto, what);
}

// Two cards (heads split, o_proj k split, pair_reduce) vs one card, steps back to back. Every card of
// the fabric: with four, a kv head on two cards, 6 of its q heads each (the four-card wire).
void test_cards(xenithium::Fabric &f, const Model &m, k::QsaKv kv, const char *tag) {
  const int C = static_cast<int>(f.size());
  const std::string who = std::string(tag) + (C == 2 ? " two cards" : " four cards");
  const auto plan = make_plan(2300, 91);
  Card one = make_card(f, 0, m, 1, kv, 1, false, 0);
  std::vector<Card> d;
  for (int c = 0; c < C; ++c) d.push_back(make_card(f, c, m, C, kv, 1, false, 0));
  for (int c = 0; c < C; ++c) {
    if (C == 2) {
      d[c].wire.peer_inbox = d[1 - c].wire.inbox;
    } else {
      d[c].wire.quad.card = c;
      for (int o = 0; o < C; ++o) d[c].wire.quad.inbox[o] = d[o].wire.inbox;
    }
  }
  const int H = m.s.hidden, qks = k::dense_auto_ksplit(d[0].w.qkv.n, H);
  const Trace t1 = run_plan(one, m, plan, 3000, qks, {}, nullptr, nullptr, nullptr, nullptr);
  const size_t S = plan.size();
  std::vector<std::vector<bf16 *>> outs(C), attns(C), xs(C);
  std::vector<std::vector<int32_t *>> poss(C), seln(C), sels(C);
  for (int c = 0; c < C; ++c)
    for (size_t i = 0; i < S; ++i) {
      outs[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * H * 2)));
      attns[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * d[c].s.o_k() * 2)));
      xs[c].push_back(static_cast<bf16 *>(f.alloc(c, 4ull * H * 2)));
      poss[c].push_back(static_cast<int32_t *>(f.alloc(c, 64)));
      sels[c].push_back(static_cast<int32_t *>(f.alloc(c, 4ull * 512 * 4)));
      seln[c].push_back(static_cast<int32_t *>(f.alloc(c, 64)));
      const auto xb = bits(make_x(H, 3000 + static_cast<uint32_t>(i)));
      f.compute(c).memcpy(xs[c].back(), xb.data(), xb.size() * 2).wait();
      f.compute(c).memcpy(poss[c].back(), &plan[i].pos, 4).wait();
    }
  for (size_t i = 0; i < S; ++i)
    for (int c = 0; c < C; ++c) {
      k::QsaScratch sc = d[c].sc;
      sc.attn = attns[c][i];
      sc.sel = sels[c][i];
      sc.sel_n = seln[c][i];
      k::QsaStep step{poss[c][i], plan[i].bound ? kCap : plan[i].pos + plan[i].rows};
      k::qsa_decode(f.compute(c), d[c].s, plan[i].rows, xs[c][i], d[c].w, d[c].c, d[c].rope, step, sc, &d[c].wire, static_cast<int>(i & 1),
                    outs[c][i], qks);
    }
  for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  bool same = true, aeq = true, seq = true;
  BoundCmp tp;
  for (size_t i = 0; i < S; ++i) {
    const int R = plan[i].rows;
    const auto a = rd16(f.compute(0), outs[0][i], static_cast<size_t>(R) * H);
    for (int c = 1; c < C; ++c) same &= rd16(f.compute(c), outs[c][i], a.size()) == a;
    std::vector<double> ref(static_cast<size_t>(R) * H, 0.0);
    const bool refstep = i % 64 == 0 || i + 1 == S;  // host o_proj reference on a subset (cost)
    for (int c = 0; c < C; ++c) {
      const int ok_ = d[c].s.o_k(), o1 = one.s.o_k();
      const auto at = rd(f.compute(c), reinterpret_cast<const uint16_t *>(attns[c][i]), static_cast<size_t>(R) * ok_);
      for (int r = 0; r < R; ++r) {
        aeq &= std::equal(at.begin() + static_cast<size_t>(r) * ok_, at.begin() + static_cast<size_t>(r + 1) * ok_,
                          t1.attn[i].begin() + static_cast<size_t>(r) * o1 + c * ok_);
        for (int n = 0; n < H && refstep; ++n) {
          double acc = 0;
          for (int j = 0; j < ok_; ++j)
            acc += static_cast<double>(k::bf16_value(at[static_cast<size_t>(r) * ok_ + j])) * d[c].o_deq[static_cast<size_t>(n) * ok_ + j];
          ref[static_cast<size_t>(r) * H + n] += acc;
        }
      }
      const auto sl = rd(f.compute(c), sels[c][i], 4ull * 512);
      const auto sn = rd(f.compute(c), seln[c][i], 4);
      std::vector<int32_t> sv;
      for (int r = 0; r < R; ++r) {
        sv.push_back(sn[r]);
        sv.insert(sv.end(), sl.begin() + r * 512, sl.begin() + r * 512 + sn[r]);
      }
      seq &= sv == t1.sel[i];
    }
    for (int r = 0; r < R && refstep; ++r) tp.add_row(&a[static_cast<size_t>(r) * H], &ref[static_cast<size_t>(r) * H], H);
  }
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
    err |= e;
  }
  char what[400];
  std::snprintf(what, sizeof what, "%s, %zu steps back to back: every card's out bit-identical", who.c_str(), S);
  expect(same, what);
  std::snprintf(what, sizeof what, "%s: each card's attn == the one-card attn of its heads; selection == one card's, bit for bit", who.c_str());
  expect(aeq && seq, what);
  std::snprintf(what, sizeof what, "%s: out vs the sum of the per-card o_proj references (every 64th step): %s; no mailbox timeout", who.c_str(), tp.text().c_str());
  expect(tp.ok(1.0 / 128, 0.01) && !err, what);
  for (int c = 0; c < C; ++c) {
    for (auto *v : {&outs[c], &attns[c], &xs[c]})
      for (auto *p : *v) f.release(p);
    for (auto *v : {&poss[c], &sels[c], &seln[c]})
      for (auto *p : *v) f.release(p);
    free_card(f, d[c]);
  }
  free_card(f, one);
}

// Four cards, the 4-row steps by owner rows (each card scores and selects its row,
// owner_share sends its sel row to the other cards) against every card selecting every row (test_cards' run): every
// card's selection, attn and out bit for bit, on the same plan past 2,048 tokens (selects from step ~512 blocks).
void test_cards_owner(xenithium::Fabric &f, const Model &m, k::QsaKv kv, const char *tag) {
  const int C = static_cast<int>(f.size());
  const auto plan = make_plan(2300, 91);
  const size_t S = plan.size();
  const int H = m.s.hidden;
  std::vector<std::vector<Card>> runs(2);
  for (auto &d : runs) {
    for (int c = 0; c < C; ++c) d.push_back(make_card(f, c, m, C, kv, 1, false, 0));
    for (int c = 0; c < C; ++c) {
      d[c].wire.quad.card = c;
      for (int o = 0; o < C; ++o) d[c].wire.quad.inbox[o] = d[o].wire.inbox;
    }
  }
  const int qks = k::dense_auto_ksplit(runs[0][0].w.qkv.n, H);
  const size_t mw = k::owner_share_words(2 * k::kQsaTopBlocks);
  std::vector<k::OwnerShare> share(C);
  std::vector<uint32_t *> inbox(C);
  for (int c = 0; c < C; ++c) {
    inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(mw * 4));
    k::owner_mail_init(f.compute(c), inbox[c], mw).wait();
  }
  for (int c = 0; c < C; ++c) {
    share[c].mail.card = c;
    for (int o = 0; o < C; ++o) share[c].mail.inbox[o] = inbox[o];
    share[c].error = static_cast<uint32_t *>(runs[1][c].dev(f, c, 64));
    f.compute(c).memset(share[c].error, 0, 4).wait();
  }
  // per run, card and step: out, attn, sel, sel_n
  using Steps16 = std::vector<std::vector<uint16_t>>;
  using Steps32 = std::vector<std::vector<int32_t>>;
  std::vector<std::vector<Steps16>> outs(2, std::vector<Steps16>(C)), attns = outs;
  std::vector<std::vector<Steps32>> sels(2, std::vector<Steps32>(C));
  int owner_steps = 0, owner_sel_steps = 0;
  for (int run = 0; run < 2; ++run)
    for (size_t i = 0; i < S; ++i) {
      const auto xb = bits(make_x(H, 3000 + static_cast<uint32_t>(i)));
      const bool own = run == 1 && plan[i].rows == 4;
      owner_steps += own;
      owner_sel_steps += own && (plan[i].pos + 4) / k::kQsaCompress > k::kQsaTopBlocks;
      for (int c = 0; c < C; ++c) {
        Card &d = runs[run][c];
        d.q->memcpy(d.pos, &plan[i].pos, 4);
        d.q->memcpy(d.x, xb.data(), xb.size() * 2);
        k::QsaScratch sc = d.sc;
        if (own) {
          const k::RowRange r = k::owner_rows(c, 4);
          sc.sel_lo = r.lo, sc.sel_hi = r.hi, sc.sel_share = &share[c], sc.sel_parity = static_cast<int>(i & 1);
        }
        k::QsaStep step{d.pos, plan[i].bound ? kCap : plan[i].pos + plan[i].rows};
        k::qsa_decode(*d.q, d.s, plan[i].rows, d.x, d.w, d.c, d.rope, step, sc, &d.wire, static_cast<int>(i & 1), d.out, qks);
      }
      for (int c = 0; c < C; ++c) {
        Card &d = runs[run][c];
        const int R = plan[i].rows;
        outs[run][c].push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), static_cast<size_t>(R) * H));
        attns[run][c].push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.sc.attn), static_cast<size_t>(R) * d.s.o_k()));
        const auto sl = rd(*d.q, d.sc.sel, 4ull * 512);
        const auto sn = rd(*d.q, d.sc.sel_n, 4);
        std::vector<int32_t> sv;
        const bool every = (plan[i].pos + R) / k::kQsaCompress <= k::kQsaTopBlocks;  // no select read: attn takes every block
        for (int r = 0; r < R && !every; ++r) {
          sv.push_back(sn[r]);
          sv.insert(sv.end(), sl.begin() + r * 512, sl.begin() + r * 512 + sn[r]);
        }
        sels[run][c].push_back(sv);
      }
    }
  bool same = true;
  for (int c = 0; c < C; ++c) same &= outs[0][c] == outs[1][c] && attns[0][c] == attns[1][c] && sels[0][c] == sels[1][c];
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0, w = 0;
    f.compute(c).memcpy(&e, share[c].error, 4).wait();
    f.compute(c).memcpy(&w, runs[1][c].wire.error, 4).wait();
    err |= e | w;
  }
  char what[400];
  std::snprintf(what, sizeof what,
                "%s four cards, %zu steps (%d of 4 rows by owner rows, %d of them past 512 blocks): each card's selection, attn and out == every card "
                "selecting every row, bit for bit; no mailbox timeout",
                tag, S, owner_steps, owner_sel_steps);
  expect(same && !err && owner_sel_steps > 0, what);
  for (int c = 0; c < C; ++c) f.release(inbox[c]);
  for (auto &d : runs)
    for (auto &x : d) free_card(f, x);
}

// qsa_rows (prefill) against qsa_decode on 4-row slices. Integer data makes every
// projection exact in both paths (x integers in [-4, 4]; qkv / o int8 codes with a power-of-two
// row scale; idx bf16 integers in [-3, 3]: all sums < 2^24), so the middle (prep, score, select,
// attn, merge) sees identical inputs and must match bit for bit: the KV pools, compressed keys and
// ring after the chunk, the o_proj input of every row, and the next decode step's output. out
// (o_proj over arbitrary bf16 attn values, another k order) gets the step bound. Chunks: at 6
// (inside the first blocks: the ring), and at 2,000 across the 512-block budget (batches with and
// without score / select).
Model make_int_model(int H, uint32_t seed) {
  Model m;
  m.s = k::QsaShape{H, 24, 2, 1, 0, 1e-6f, k::QsaKv::i8};
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> c127(-127, 127), c3(-3, 3);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto codes = [&](std::vector<float> &v, int n, int kk) {  // row max |code| = 127 -> scale 2^-10 exactly
    v.resize(static_cast<size_t>(n) * kk);
    for (int r = 0; r < n; ++r) {
      for (int j = 0; j < kk; ++j) v[static_cast<size_t>(r) * kk + j] = static_cast<float>(c127(rng)) / 1024.f;
      v[static_cast<size_t>(r) * kk + static_cast<size_t>(rng() % kk)] = (rng() & 1 ? 127.f : -127.f) / 1024.f;
    }
  };
  codes(m.qkv, m.s.qkv_n(), H);
  codes(m.o, H, m.s.o_k());
  m.idx.resize(static_cast<size_t>(m.s.idx_n()) * H);
  for (auto &v : m.idx) v = static_cast<float>(c3(rng));
  auto fill = [&](std::vector<float> &v, size_t n) {
    v.resize(n);
    for (auto &x : v) x = bfr(nd(rng) * 0.2f);
  };
  fill(m.qn, 256);
  fill(m.kn, 256);
  fill(m.iqn, 128);
  fill(m.ikn, 128);
  return m;
}

void test_rows(xenithium::Fabric &f, const Model &m, k::QsaKv kv, int p0, int M, const char *tag) {
  Card d = make_card(f, 0, m, 1, kv, 1, false, 0);
  const auto &s = d.s;
  const int H = s.hidden, hq = s.hq(), B = k::kQsaRowsBatch, R4 = M + 4;
  // x for every position (integers)
  std::mt19937 rng(9100u + p0 + M);
  std::uniform_int_distribution<int> xi(-4, 4);
  std::vector<uint16_t> X(static_cast<size_t>(p0 + M + 8) * H);
  for (auto &v : X) v = k::bf16_bits(static_cast<float>(xi(rng)));
  auto dev = [&](size_t bytes) { return d.dev(f, 0, bytes); };
  k::QsaRowsScratch rs;
  rs.y = static_cast<float *>(dev(static_cast<size_t>(R4) * d.w.qkv.n * 4));
  rs.yi = static_cast<float *>(dev(static_cast<size_t>(R4) * d.w.idx.n * 4));
  rs.attn = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * s.o_k() * 2));
  rs.partial = static_cast<float *>(dev(static_cast<size_t>(M) * H * 4));
  rs.q = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * hq * 256 * 2));
  rs.iq = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * 4 * 128 * 2));
  rs.nblk = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 4));
  rs.sel = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 512 * 4));
  rs.sel_n = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 4));
  rs.score = static_cast<uint32_t *>(dev(k::qsa_score_words(kCap) / 4 * B * 4));
  rs.cand_key = static_cast<uint32_t *>(dev(static_cast<size_t>(B) * 16 * 512 * 4));
  rs.cand_id = static_cast<int32_t *>(dev(static_cast<size_t>(B) * 16 * 512 * 4));
  rs.cand_n = static_cast<int32_t *>(dev(static_cast<size_t>(B) * 16 * 4));
  rs.part = static_cast<float *>(dev(k::qsa_part_floats(s) / 4 * B * 4));
  auto *xc = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * H * 2));
  auto *out = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * H * 2));
  const int slices = (M + 3) / 4;
  std::vector<int32_t> words(static_cast<size_t>(slices) * 2, 0);
  for (int sl = 0; sl < slices; ++sl) words[static_cast<size_t>(sl) * 2] = p0 + 4 * sl;
  auto *wd = up(f, 0, words, d.mem);
  const size_t P = d.pool_pages;
  auto poison = [&] {
    d.q->memset(d.c.k, 0xA5, P * k::qsa_page_kv_bytes(s));
    d.q->memset(d.c.v, 0xA5, P * k::qsa_page_v_bytes(s));
    d.q->memset(d.c.ks, 0xA5, P * k::qsa_page_scale_floats(s) * 4);
    d.q->memset(d.c.vs, 0xA5, P * k::qsa_page_vscale_floats(s) * 4);
    d.q->memset(d.c.kc, 0xA5, P * k::qsa_page_kc_elems() * 2);
    d.q->memset(d.c.ring, 0xA5, k::kQsaRing * 128 * 2).wait();
  };
  auto rows_of = [&](int p, int n) { return std::vector<uint16_t>(X.begin() + static_cast<size_t>(p) * H, X.begin() + static_cast<size_t>(p + n) * H); };
  auto step = [&](int pos, int rows) {
    std::vector<uint16_t> xb(4ull * H, 0);
    const auto r = rows_of(pos, rows);
    std::copy(r.begin(), r.end(), xb.begin());
    run_step(d, Step{pos, rows, 0}, xb, nullptr, 0, 0);
  };
  struct Got {
    std::vector<uint8_t> k, v, kc, ring;
    std::vector<float> ks, vs;
    std::vector<uint16_t> attn, next;
    std::vector<float> out;
  };
  auto path = [&](bool rows) {
    poison();
    for (int p = 0; p < p0; p += 4) step(p, std::min(4, p0 - p));  // history
    Got g;
    if (rows) {
      const auto xr = rows_of(p0, M);
      d.q->memcpy(xc, xr.data(), xr.size() * 2).wait();
      k::qsa_rows(*d.q, s, M, xc, d.w, d.c, d.rope, wd, 2, p0, d.sc, rs, nullptr, 0, out);
      d.q->wait_and_throw();
      g.attn = rd(*d.q, reinterpret_cast<const uint16_t *>(rs.attn), static_cast<size_t>(M) * s.o_k());
      g.out = rd16(*d.q, out, static_cast<size_t>(M) * H);
    } else {
      for (int sl = 0; sl < slices; ++sl) {
        const int rr = std::min(4, M - 4 * sl);
        step(p0 + 4 * sl, rr);
        d.q->wait_and_throw();
        const auto a = rd(*d.q, reinterpret_cast<const uint16_t *>(d.sc.attn), static_cast<size_t>(rr) * s.o_k());
        const auto o = rd16(*d.q, d.out, static_cast<size_t>(rr) * H);
        g.attn.insert(g.attn.end(), a.begin(), a.end());
        g.out.insert(g.out.end(), o.begin(), o.end());
      }
    }
    g.k = rd(*d.q, d.c.k, P * k::qsa_page_kv_bytes(s));
    g.v = rd(*d.q, d.c.v, P * k::qsa_page_v_bytes(s));
    g.ks = rd(*d.q, d.c.ks, P * k::qsa_page_scale_floats(s));
    g.vs = rd(*d.q, d.c.vs, P * k::qsa_page_vscale_floats(s));
    g.kc = rd(*d.q, reinterpret_cast<const uint8_t *>(d.c.kc), P * k::qsa_page_kc_elems() * 2);
    g.ring = rd(*d.q, reinterpret_cast<const uint8_t *>(d.c.ring), k::kQsaRing * 128 * 2);
    step(p0 + M, 4);  // the next decode step reads what the chunk left
    d.q->wait_and_throw();
    g.next = rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), 4ull * H);
    return g;
  };
  // The prefill's 8-tile online softmax (the default) has its own numerics; XEN_QSA_PF_TILES=1 is the
  // decode's tiling, whose bits the contract below checks. b8: the default, within a bound.
  const char *pf_env = std::getenv("XEN_QSA_PF_TILES");
  const std::string pf_keep = pf_env ? pf_env : "";
  setenv("XEN_QSA_PF_TILES", "1", 1);
  const Got a = path(false), b = path(true), b2 = path(true);
  unsetenv("XEN_QSA_PF_TILES");
  const Got b8 = path(true);
  if (pf_env) setenv("XEN_QSA_PF_TILES", pf_keep.c_str(), 1);
  BoundCmp oc, ac8, oc8;
  for (int r = 0; r < M; ++r) oc8.add_rowf(&b8.out[static_cast<size_t>(r) * H], &a.out[static_cast<size_t>(r) * H], H);
  {
    const size_t K = static_cast<size_t>(s.o_k());
    std::vector<float> fa(K), fb(K);
    for (int r = 0; r < M; ++r) {
      for (size_t j = 0; j < K; ++j) {
        fa[j] = k::bf16_value(a.attn[static_cast<size_t>(r) * K + j]);
        fb[j] = k::bf16_value(b8.attn[static_cast<size_t>(r) * K + j]);
      }
      ac8.add_rowf(fb.data(), fa.data(), static_cast<int>(K));
    }
  }
  for (int r = 0; r < M; ++r) oc.add_rowf(&b.out[static_cast<size_t>(r) * H], &a.out[static_cast<size_t>(r) * H], H);
  const bool sc8 = k::qsa_kv_scaled(kv);
  char what[400];
  std::snprintf(what, sizeof what, "%s %s KV, qsa_rows %d rows at %d: K / V / scales / compressed keys / ring == the decode path on 4-row slices, bit for bit",
                tag, kv_name(kv), M, p0);
  expect(a.k == b.k && a.v == b.v && (!sc8 || (a.ks == b.ks && a.vs == b.vs)) && a.kc == b.kc && a.ring == b.ring, what);
  std::snprintf(what, sizeof what, "%s qsa_rows %d rows at %d: o_proj input (gated attention) of every row bit-identical", tag, M, p0);
  expect(a.attn == b.attn, what);
  std::snprintf(what, sizeof what, "%s qsa_rows %d rows at %d: out vs the decode path %s; the next decode step bit-identical", tag, M, p0,
                oc.text().c_str());
  expect(oc.ok(1.0 / 128, 0.01) && a.next == b.next, what);
  std::snprintf(what, sizeof what, "%s qsa_rows %d rows at %d: run-to-run bit-identical", tag, M, p0);
  expect(b2.attn == b.attn && b2.out == b.out && b2.next == b.next, what);
  std::snprintf(what, sizeof what, "%s qsa_rows %d rows at %d, 8-tile prefill attention: o_proj input %s, out %s; the next decode step bit-identical",
                tag, M, p0, ac8.text().c_str(), oc8.text().c_str());
  expect(ac8.ok(1.0 / 64, 0.10) && oc8.ok(1.0 / 64, 0.10) && b8.next == b.next, what);
  free_card(f, d);
}

// TP4: qsa_rows on four cards (a kv head on two cards, 6 q heads each; the four-card rows wire)
// against one card's qsa_rows on the same rows: each card's o_proj input == the one-card rows of its heads bit for bit
// (the decode's tiling and the 8-tile prefill attention), every card's out identical, the two cards of a kv head the
// same K / V pools. Integer data (make_int_model): the projections are exact on any split.
void test_rows_cards(xenithium::Fabric &f, k::QsaKv kv, int p0, int M) {
  const Model m = make_int_model(2560, 29);
  const int C = static_cast<int>(f.size()), H = m.s.hidden, B = k::kQsaRowsBatch, R4 = M + 4;
  std::mt19937 rng(7300u + p0 + M);
  std::uniform_int_distribution<int> xi(-4, 4);
  std::vector<uint16_t> X(static_cast<size_t>(p0 + M + 8) * H);
  for (auto &v : X) v = k::bf16_bits(static_cast<float>(xi(rng)));
  struct Run {
    Card d;
    k::QsaRowsScratch rs;
    bf16 *xc = nullptr, *out = nullptr;
    int32_t *wd = nullptr;
    k::PairRowsWire wire;
  };
  auto setup = [&](int c, int cards) {
    Run r;
    r.d = make_card(f, c, m, cards, kv, 1, false, 0);
    const auto &s = r.d.s;
    auto dev = [&](size_t bytes) { return r.d.dev(f, c, bytes); };
    auto &rs = r.rs;
    rs.y = static_cast<float *>(dev(static_cast<size_t>(R4) * r.d.w.qkv.n * 4));
    rs.yi = static_cast<float *>(dev(static_cast<size_t>(R4) * r.d.w.idx.n * 4));
    rs.attn = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * s.o_k() * 2));
    rs.partial = static_cast<float *>(dev(static_cast<size_t>(M) * H * 4));
    rs.q = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * s.hq() * 256 * 2));
    rs.iq = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * 4 * 128 * 2));
    rs.nblk = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 4));
    rs.sel = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 512 * 4));
    rs.sel_n = static_cast<int32_t *>(dev(static_cast<size_t>(R4) * 4));
    rs.score = static_cast<uint32_t *>(dev(k::qsa_score_words(kCap) / 4 * B * 4));
    rs.cand_key = static_cast<uint32_t *>(dev(static_cast<size_t>(B) * 16 * 512 * 4));
    rs.cand_id = static_cast<int32_t *>(dev(static_cast<size_t>(B) * 16 * 512 * 4));
    rs.cand_n = static_cast<int32_t *>(dev(static_cast<size_t>(B) * 16 * 4));
    rs.part = static_cast<float *>(dev(k::qsa_part_floats(s) / 4 * B * 4));
    r.xc = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * H * 2));
    r.out = static_cast<bf16 *>(dev(static_cast<size_t>(R4) * H * 2));
    const int slices = (M + 3) / 4;
    std::vector<int32_t> words(static_cast<size_t>(slices) * 2, 0);
    for (int sl = 0; sl < slices; ++sl) words[static_cast<size_t>(sl) * 2] = p0 + 4 * sl;
    r.wd = up(f, c, words, r.d.mem);
    if (cards > 1) {
      r.wire.quad.inbox[c] = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_rows_wire_words(M, H, cards) * 4));
      r.d.mem.push_back(r.wire.quad.inbox[c]);
      k::pair_rows_wire_init(*r.d.q, M, H, r.wire.quad.inbox[c], cards).wait();
      r.wire.error = r.d.wire.error;
      r.wire.rows_max = M, r.wire.n_max = H;
    }
    return r;
  };
  Run one = setup(0, 1);
  std::vector<Run> d;
  for (int c = 0; c < C; ++c) d.push_back(setup(c, C));
  for (int c = 0; c < C; ++c) {
    d[c].wire.quad.card = c;
    for (int o = 0; o < C; ++o) d[c].wire.quad.inbox[o] = d[o].wire.quad.inbox[o];
  }
  auto history = [&](Run &r, const k::PairWire *w, int &par) {
    for (int p = 0; p < p0; p += 4) {
      const int rows = std::min(4, p0 - p);
      std::vector<uint16_t> xb(4ull * H, 0);
      std::copy(X.begin() + static_cast<size_t>(p) * H, X.begin() + static_cast<size_t>(p + rows) * H, xb.begin());
      run_step(r.d, Step{p, rows, 0}, xb, w, par, 0);
      par ^= 1;
    }
  };
  int rows_par = 0;
  auto chunk = [&](std::vector<Run *> runs, bool tiles1) {
    if (tiles1) setenv("XEN_QSA_PF_TILES", "1", 1);
    else unsetenv("XEN_QSA_PF_TILES");
    const std::vector<uint16_t> xr(X.begin() + static_cast<size_t>(p0) * H, X.begin() + static_cast<size_t>(p0 + M) * H);
    for (Run *r : runs) {
      r->d.q->memcpy(r->xc, xr.data(), xr.size() * 2).wait();
      k::qsa_rows(*r->d.q, r->d.s, M, r->xc, r->d.w, r->d.c, r->d.rope, r->wd, 2, p0, r->d.sc, r->rs, runs.size() > 1 ? &r->wire : nullptr, rows_par, r->out);
    }
    for (Run *r : runs) r->d.q->wait_and_throw();
    if (runs.size() > 1) rows_par ^= 1;
  };
  const char *pf_env = std::getenv("XEN_QSA_PF_TILES");
  const std::string pf_keep = pf_env ? pf_env : "";
  bool attn_eq = true, out_eq = true, kv_eq = true;
  for (const bool tiles1 : {true, false}) {
    int p1 = 0;
    history(one, nullptr, p1);
    std::vector<int> par(C, 0);
    // the four cards' history: each card's decode steps (the four-card decode wire)
    for (int p = 0; p < p0; p += 4) {
      const int rows = std::min(4, p0 - p);
      std::vector<uint16_t> xb(4ull * H, 0);
      std::copy(X.begin() + static_cast<size_t>(p) * H, X.begin() + static_cast<size_t>(p + rows) * H, xb.begin());
      for (int c = 0; c < C; ++c) {
        k::PairWire w = d[c].d.wire;
        w.quad.card = c;
        for (int o = 0; o < C; ++o) w.quad.inbox[o] = d[o].d.wire.inbox;
        const auto &st = Step{p, rows, 0};
        d[c].d.q->memcpy(d[c].d.x, xb.data(), xb.size() * 2).wait();
        d[c].d.q->memcpy(d[c].d.pos, &st.pos, 4).wait();
        k::qsa_decode(*d[c].d.q, d[c].d.s, rows, d[c].d.x, d[c].d.w, d[c].d.c, d[c].d.rope, k::QsaStep{d[c].d.pos, p + rows}, d[c].d.sc, &w, par[c], d[c].d.out, 0);
        par[c] ^= 1;
      }
      for (int c = 0; c < C; ++c) d[c].d.q->wait_and_throw();
    }
    chunk({&one}, tiles1);
    std::vector<Run *> all;
    for (auto &r : d) all.push_back(&r);
    chunk(all, tiles1);
    const int o1 = one.d.s.o_k();
    const auto a1 = rd(*one.d.q, reinterpret_cast<const uint16_t *>(one.rs.attn), static_cast<size_t>(M) * o1);
    const auto out0 = rd16(*d[0].d.q, d[0].out, static_cast<size_t>(M) * H);
    for (int c = 0; c < C; ++c) {
      const int ok_ = d[c].d.s.o_k();
      const auto ac = rd(*d[c].d.q, reinterpret_cast<const uint16_t *>(d[c].rs.attn), static_cast<size_t>(M) * ok_);
      for (int r = 0; r < M; ++r)
        attn_eq &= std::equal(ac.begin() + static_cast<size_t>(r) * ok_, ac.begin() + static_cast<size_t>(r + 1) * ok_, a1.begin() + static_cast<size_t>(r) * o1 + c * ok_);
      out_eq &= rd16(*d[c].d.q, d[c].out, out0.size()) == out0;
    }
    const size_t P = d[0].d.pool_pages;
    for (int c = 0; c + 1 < C; c += 2)
      kv_eq &= rd(*d[c].d.q, d[c].d.c.k, P * k::qsa_page_kv_bytes(d[c].d.s)) == rd(*d[c + 1].d.q, d[c + 1].d.c.k, P * k::qsa_page_kv_bytes(d[c + 1].d.s)) &&
               rd(*d[c].d.q, d[c].d.c.v, P * k::qsa_page_v_bytes(d[c].d.s)) == rd(*d[c + 1].d.q, d[c + 1].d.c.v, P * k::qsa_page_v_bytes(d[c + 1].d.s));
  }
  if (pf_env) setenv("XEN_QSA_PF_TILES", pf_keep.c_str(), 1);
  else unsetenv("XEN_QSA_PF_TILES");
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    d[c].d.q->memcpy(&e, d[c].d.wire.error, 4).wait();
    err |= e;
  }
  char what[400];
  std::snprintf(what, sizeof what,
                "integer data, %s KV, four cards, qsa_rows %d rows at %d (decode tiling and 8-tile): each card's o_proj input == the one-card rows of its "
                "heads, bit for bit; every card's out identical; a kv head's two cards the same K / V; no mailbox timeout",
                kv_name(kv), M, p0);
  expect(attn_eq && out_eq && kv_eq && err == 0, what);
  for (auto &r : d) free_card(f, r.d);
  free_card(f, one.d);
}

// Several sequences in one step (segment records, rows.hpp) against each sequence alone,
// bit for bit: the slots share the pools through their own page tables (disjoint random physical
// pages) and raw-key rings, sit at different depths past the 512-block budget (score + select run),
// skip steps, run 1-4 rows per segment and commit fewer rows than they ran.
void test_segments(xenithium::Fabric &f, const Model &m, k::QsaKv kv, const char *tag) {
  constexpr int NS = 16;  // 16 slots (16-row steps of one-row segments)
  Card d = make_card(f, 0, m, 1, kv, NS, false, 0);
  const int H = d.s.hidden, lp = kCap / k::kQsaPage;
  std::vector<int32_t> table(static_cast<size_t>(NS) * lp);
  std::iota(table.begin(), table.end(), 0);
  std::shuffle(table.begin(), table.end(), std::mt19937(77));
  int32_t *tab = up(f, 0, table, d.mem);
  auto *ring = static_cast<bf16 *>(d.dev(f, 0, static_cast<size_t>(NS) * k::kQsaRing * 128 * 2));
  auto *rec = static_cast<int32_t *>(d.dev(f, 0, k::kMaxSegs * k::kSegRec * 4));
  auto view = [&](int q) {  // slot q alone, on the one-sequence path
    k::QsaCache c = d.c;
    c.pages = tab + static_cast<size_t>(q) * lp;
    c.ring = ring + static_cast<size_t>(q) * k::kQsaRing * 128;
    return c;
  };
  k::QsaCache all = d.c;
  all.pages = tab;
  all.ring = ring;
  all.slot_pages = lp;
  const int hist[NS] = {2100, 2399, 2702, 2113, 2150, 2201, 2230, 2297, 2333, 2410, 2444, 2501, 2555, 2600, 2650, 2690};
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
    int code() const {  // the first segments rr[0] rows each, the rest one row
      return rr.empty() ? rows : k::seg_het(rr[0], static_cast<int>(std::count(rr.begin(), rr.end(), rr[0])), static_cast<int>(seqs.size()));
    }
  };
  // steps of 5-8 rows (the 8-row shape) among them; unequal segments (1-4 rows each)
  const std::vector<SegStep> plan = {{{0, 1, 2}, 1}, {{0, 1}, 2}, {{2}, 4}, {{1, 0}, 2}, {{0, 1, 2}, 0, {3, 3, 1}}, {{0, 1, 2}, 1}, {{2, 0}, 2},
                                     {{1}, 3},       {{0, 1, 2}, 1}, {{0, 2}, 2}, {{2, 0, 1}, 0, {4, 1, 1}}, {{1, 2}, 2}, {{2, 1, 0}, 1}, {{0}, 4},
                                     {{0, 1}, 4},    {{2, 0, 1}, 2}, {{1, 2}, 3}, {{1, 0}, 0, {4, 1}}, {{2, 0}, 4}, {{0, 2, 1}, 0, {2, 2, 1}},
                                     // 9-16 rows (the 16-row shape)
                                     {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1}, {{15, 3, 7, 1, 9, 0, 12, 5}, 2},
                                     {{4, 13, 2, 10}, 4}, {{6, 11, 14}, 4}, {{8, 0, 1, 2, 3, 4, 5, 6, 7, 9}, 0, {4, 4, 1, 1, 1, 1, 1, 1, 1, 1}},
                                     {{2, 9, 15, 0, 11, 13}, 0, {3, 3, 3, 1, 1, 1}}, {{14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15}, 1}, {{1, 0, 2}, 3}};
  auto commit_of = [](int k2, int rows) { return k2 % 2 == 0 ? rows : std::max(1, rows - 1); };
  auto x_of = [&](int q, int k2) { return bits(make_x(H, 31000u + 1000u * static_cast<uint32_t>(q) + static_cast<uint32_t>(k2))); };
  auto history = [&]() {
    for (int q = 0; q < NS; ++q) {
      const k::QsaCache c = view(q);
      for (int p = 0; p < hist[q]; p += 4) {
        const int rows = std::min(4, hist[q] - p);
        d.q->memcpy(d.pos, &p, 4).wait();
        const auto xb = x_of(q, 100000 + p);
        d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
        k::qsa_decode(*d.q, d.s, rows, d.x, d.w, c, d.rope, k::QsaStep{d.pos, p + rows}, d.sc, nullptr, 0, d.out);
      }
    }
    d.q->wait_and_throw();
  };
  // solo
  std::vector<std::vector<std::vector<uint16_t>>> solo(NS);
  history();
  for (int q = 0; q < NS; ++q) {
    const k::QsaCache c = view(q);
    int pos = hist[q], k2 = 0, i = 0;
    for (const auto &st : plan) {
      ++i;
      const auto at = std::find(st.seqs.begin(), st.seqs.end(), q);
      if (at == st.seqs.end()) continue;
      const int rq = st.of(static_cast<size_t>(at - st.seqs.begin()));
      d.q->memcpy(d.pos, &pos, 4).wait();
      const auto xb = x_of(q, k2);
      d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
      k::qsa_decode(*d.q, d.s, rq, d.x, d.w, c, d.rope, k::QsaStep{d.pos, i % 2 ? kCap : pos + rq}, d.sc, nullptr, 0, d.out);
      d.q->wait_and_throw();
      solo[q].push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), static_cast<size_t>(rq) * H));
      pos += commit_of(k2, rq);
      ++k2;
    }
  }
  // batched
  history();
  int pos[NS], own[NS] = {};
  std::copy(hist, hist + NS, pos);
  bool same = true;
  size_t rows_cmp = 0;
  int i = 0;
  for (const auto &st : plan) {
    ++i;
    const int ns = static_cast<int>(st.seqs.size()), R = st.all();
    std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
    std::vector<uint16_t> x(static_cast<size_t>(k::kStepRows32) * H, 0);
    int need = 0;
    for (int j = 0, r0 = 0; j < ns; r0 += st.of(static_cast<size_t>(j)), ++j) {
      const int q = st.seqs[static_cast<size_t>(j)], rq = st.of(static_cast<size_t>(j));
      w[static_cast<size_t>(j) * k::kSegRec + k::kSegPos] = pos[q];
      w[static_cast<size_t>(j) * k::kSegRec + k::kSegSlot] = q;
      const auto xq = x_of(q, own[q]);
      std::copy(xq.begin(), xq.begin() + static_cast<size_t>(rq) * H, x.begin() + static_cast<size_t>(r0) * H);
      need = std::max(need, pos[q] + rq);
    }
    d.q->memcpy(rec, w.data(), w.size() * 4).wait();
    d.q->memcpy(d.x, x.data(), x.size() * 2).wait();
    k::qsa_decode(*d.q, d.s, R, d.x, d.w, all, d.rope, k::QsaStep{rec, i % 2 ? kCap : need, k::kSegRec, st.code()}, d.sc, nullptr, 0, d.out);
    d.q->wait_and_throw();
    const auto out = rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), static_cast<size_t>(R) * H);
    for (int j = 0, r0 = 0; j < ns; r0 += st.of(static_cast<size_t>(j)), ++j) {
      const int q = st.seqs[static_cast<size_t>(j)], rq = st.of(static_cast<size_t>(j));
      same &= std::equal(out.begin() + static_cast<size_t>(r0) * H, out.begin() + static_cast<size_t>(r0 + rq) * H, solo[q][own[q]].begin());
      rows_cmp += rq;
      pos[q] += commit_of(own[q], rq);
      ++own[q];
    }
  }
  char what[512];
  std::snprintf(what, sizeof what, "%s segments (%d slots at %d-%d tokens, own page tables + rings, 1-4 rows each, unequal segments, up to 16 a step, skipped steps, partial commits): out == each "
                "sequence alone, bit for bit (%zu rows)", tag, NS, hist[0], hist[2], rows_cmp);
  expect(same, what);
  // qsa_store writes the cache bits qsa_decode writes. The same batched plan
  // with every other step as qsa_store: K / V / scales / compressed keys / rings == all qsa_decode.
  {
    const size_t P = static_cast<size_t>(NS) * lp;
    auto run = [&](bool store) {
      history();
      int ps[NS], ow[NS] = {};
      std::copy(hist, hist + NS, ps);
      int j2 = 0;
      for (const auto &st : plan) {
        ++j2;
        const int ns = static_cast<int>(st.seqs.size()), R = st.all();
        std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
        std::vector<uint16_t> x(static_cast<size_t>(k::kStepRows32) * H, 0);
        int need = 0;
        for (int j = 0, r0 = 0; j < ns; r0 += st.of(static_cast<size_t>(j)), ++j) {
          const int q = st.seqs[static_cast<size_t>(j)], rq = st.of(static_cast<size_t>(j));
          w[static_cast<size_t>(j) * k::kSegRec + k::kSegPos] = ps[q];
          w[static_cast<size_t>(j) * k::kSegRec + k::kSegSlot] = q;
          const auto xq = x_of(q, ow[q]);
          std::copy(xq.begin(), xq.begin() + static_cast<size_t>(rq) * H, x.begin() + static_cast<size_t>(r0) * H);
          need = std::max(need, ps[q] + rq);
        }
        d.q->memcpy(rec, w.data(), w.size() * 4).wait();
        d.q->memcpy(d.x, x.data(), x.size() * 2).wait();
        const k::QsaStep stp{rec, j2 % 2 ? kCap : need, k::kSegRec, st.code()};
        if (store && j2 % 2) k::qsa_store(*d.q, d.s, R, d.x, d.w, all, d.rope, stp, d.sc);
        else k::qsa_decode(*d.q, d.s, R, d.x, d.w, all, d.rope, stp, d.sc, nullptr, 0, d.out);
        d.q->wait_and_throw();
        for (int j = 0; j < ns; ++j) {
          const int q = st.seqs[static_cast<size_t>(j)];
          ps[q] += commit_of(ow[q], st.of(static_cast<size_t>(j)));
          ++ow[q];
        }
      }
      std::vector<std::vector<uint8_t>> g;
      g.push_back(rd(*d.q, d.c.k, P * k::qsa_page_kv_bytes(d.s)));
      g.push_back(rd(*d.q, d.c.v, P * k::qsa_page_v_bytes(d.s)));
      if (d.c.ks) g.push_back(rd(*d.q, reinterpret_cast<const uint8_t *>(d.c.ks), P * k::qsa_page_scale_floats(d.s) * 4));
      if (d.c.vs) g.push_back(rd(*d.q, reinterpret_cast<const uint8_t *>(d.c.vs), P * k::qsa_page_vscale_floats(d.s) * 4));
      g.push_back(rd(*d.q, reinterpret_cast<const uint8_t *>(d.c.kc), P * k::qsa_page_kc_elems() * 2));
      g.push_back(rd(*d.q, reinterpret_cast<const uint8_t *>(ring), static_cast<size_t>(NS) * k::kQsaRing * 128 * 2));
      return g;
    };
    const auto a = run(false), b = run(true);
    std::snprintf(what, sizeof what, "%s qsa_store on every other step of the segment plan: K / V / scales / compressed keys / rings == qsa_decode's, bit for bit", tag);
    expect(a == b, what);
  }
  free_card(f, d);
}


// ---------------------------------------------------------------- draft trees
// A tree step (records kSegTree, QsaStep:tree) against chain steps, each run from an identical copy of random caches
// (every K / V code, scale, compressed key and raw key random: no history to build), bit for bit on every card:
// rows each tree row j's out, attn, complete blocks and their score keys == row depth(j) of a chain step of j's
// path alone
// move the tree step + qsa_kv_move of a path vs the chain step of the path: K / V / scales of positions p0 - 3..
// p0 + n - 1, their ring entries and the compressed keys of the blocks to p0 + n equal; then the next 1-row
// and 4-row steps' out and attn equal
// mixed tree segments beside chain segments (seg_rows / seg_het layouts, own slots): the chain rows == alone
// Contexts: no selection (< 2,048 tokens), selection (3K, 9K), chunked select (> 16,384 tokens); p0 % 4 = 0..3 (blocks
// straddling p0; depth 7 closes 2 blocks); trees of 8, 9 and 13 (kQsaTreeRows) rows. The kernels need parents only
// (par[j] < j), so the shapes need not be in the GDN core's preorder.
constexpr int kTreeCap = 16640;  // past 16,384 tokens (4,096 blocks: the chunked select)
constexpr int kTreeSlots = 4;

struct TreeSeg {
  int slot = 0, pos = 0, n = 0;
  std::vector<int> par = {};  // empty: a chain of n rows
};
using XRows = std::vector<std::vector<uint16_t>>;  // per step row: x [hidden] (bf16 bits)
struct TreeOut {
  std::vector<std::vector<uint16_t>> out, attn;  // per card: [rows][hidden], [rows][o_k]
  std::vector<std::vector<uint32_t>> keys;  // per card: [rows][key_stride] block score keys
  std::vector<std::vector<int32_t>> nblk;  // per card: [rows] complete blocks
  size_t key_stride = 0;
};

struct TreeRig {
  xenithium::Fabric *f = nullptr;
  std::vector<Card> d;  // per card
  std::vector<k::QsaCache> c;  // per card: the slots' page tables and rings, tkc
  std::vector<int32_t *> rec;  // per card: the step's records
  std::vector<int32_t> table;  // host copy of the page tables, [slot][lp]
  int lp = 0, parity = 0;
  struct Snap {
    int card;
    void *live, *copy;
    size_t bytes;
  };
  std::vector<Snap> snap;  // the random caches every run starts from
};

std::vector<uint16_t> row_x(int H, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> x(H);
  for (auto &v : x) v = k::bf16_bits(nd(rng));
  return x;
}

// Random pools of every physical page and the slots' rings: codes, positive scales, bf16 values (one seed per kv head,
// so a kv head's cards hold the same history).
void fill_random(Card &d, const k::QsaCache &c, size_t pages, uint32_t seed) {
  const auto &s = d.s;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> c127(-127, 127), byte(0, 255);
  std::uniform_real_distribution<float> us(0.004f, 0.02f), uv(-2.f, 2.f);
  auto put = [&](void *dst, const void *src, size_t n) { d.q->memcpy(dst, src, n).wait(); };
  std::vector<uint8_t> kb(pages * k::qsa_page_kv_bytes(s)), vb(pages * k::qsa_page_v_bytes(s));
  if (s.kv == k::QsaKv::bf16) {
    auto *k16 = reinterpret_cast<uint16_t *>(kb.data()), *v16 = reinterpret_cast<uint16_t *>(vb.data());
    for (size_t i = 0; i < kb.size() / 2; ++i) k16[i] = k::bf16_bits(uv(rng)), v16[i] = k::bf16_bits(uv(rng));
  } else {
    for (auto &v : kb) v = static_cast<uint8_t>(static_cast<int8_t>(c127(rng)));
    for (auto &v : vb) v = k::qsa_kv_v4(s.kv) ? static_cast<uint8_t>(byte(rng)) : static_cast<uint8_t>(static_cast<int8_t>(c127(rng)));
    std::vector<float> ks(pages * k::qsa_page_scale_floats(s)), vs(pages * k::qsa_page_vscale_floats(s));
    for (auto &v : ks) v = us(rng);
    if (k::qsa_kv_v4(s.kv)) {
      auto *b = reinterpret_cast<uint16_t *>(vs.data());
      for (size_t i = 0; i < vs.size() * 2; ++i) b[i] = k::bf16_bits(us(rng) * 10.f);
    } else {
      for (auto &v : vs) v = us(rng);
    }
    put(c.ks, ks.data(), ks.size() * 4);
    put(c.vs, vs.data(), vs.size() * 4);
  }
  put(c.k, kb.data(), kb.size());
  put(c.v, vb.data(), vb.size());
  std::vector<uint16_t> kc(pages * k::qsa_page_kc_elems()), ring(static_cast<size_t>(kTreeSlots) * k::kQsaRing * 128);
  for (auto &v : kc) v = k::bf16_bits(uv(rng));
  for (auto &v : ring) v = k::bf16_bits(uv(rng));
  put(c.kc, kc.data(), kc.size() * 2);
  put(c.ring, ring.data(), ring.size() * 2);
}

// the fabric's first `C` cards
TreeRig make_tree_rig(xenithium::Fabric &f, int C, const Model &m, k::QsaKv kv) {
  TreeRig g;
  g.f = &f;
  g.lp = kTreeCap / k::kQsaPage;
  for (int c = 0; c < C; ++c) g.d.push_back(make_card(f, c, m, C, kv, kTreeSlots, false, 0, kTreeCap));
  for (int c = 0; c < C && C > 1; ++c) {
    if (C == 2) {
      g.d[c].wire.peer_inbox = g.d[1 - c].wire.inbox;
    } else {
      g.d[c].wire.quad.card = c;
      for (int o = 0; o < C; ++o) g.d[c].wire.quad.inbox[o] = g.d[o].wire.inbox;
    }
  }
  const size_t P = static_cast<size_t>(kTreeSlots) * g.lp;  // the pool: every slot's pages, shuffled
  g.table.resize(P);
  std::iota(g.table.begin(), g.table.end(), 0);
  std::shuffle(g.table.begin(), g.table.end(), std::mt19937(266));
  for (int c = 0; c < C; ++c) {
    Card &d = g.d[c];
    k::QsaCache cc = d.c;
    cc.pages = up(f, c, g.table, d.mem);
    cc.slot_pages = g.lp;
    cc.ring = static_cast<bf16 *>(d.dev(f, c, static_cast<size_t>(kTreeSlots) * k::kQsaRing * 128 * 2));
    cc.tkc = static_cast<bf16 *>(d.dev(f, c, k::kQsaTreeKcElems * 2));
    g.c.push_back(cc);
    g.rec.push_back(static_cast<int32_t *>(d.dev(f, c, k::kMaxSegs * k::kSegRec * 4)));
    fill_random(d, cc, P, 500u + static_cast<uint32_t>(d.s.kv0()) + 17u * static_cast<uint32_t>(kv));
    auto keep = [&](void *live, size_t bytes) {
      void *copy = d.dev(f, c, bytes);
      d.q->memcpy(copy, live, bytes).wait();
      g.snap.push_back({c, live, copy, bytes});
    };
    keep(cc.k, P * k::qsa_page_kv_bytes(d.s));
    keep(cc.v, P * k::qsa_page_v_bytes(d.s));
    keep(cc.kc, P * k::qsa_page_kc_elems() * 2);
    keep(cc.ring, static_cast<size_t>(kTreeSlots) * k::kQsaRing * 128 * 2);
    if (k::qsa_kv_scaled(kv)) {
      keep(cc.ks, P * k::qsa_page_scale_floats(d.s) * 4);
      keep(cc.vs, P * k::qsa_page_vscale_floats(d.s) * 4);
    }
  }
  return g;
}
void restore(TreeRig &g) {
  for (const auto &sn : g.snap) g.d[sn.card].q->memcpy(sn.live, sn.copy, sn.bytes);
  for (auto &d : g.d) d.q->wait_and_throw();
}

// One step of segments `segs` (seg_rows code `code`) on every card; x = the step's rows.
TreeOut tree_step(TreeRig &g, const std::vector<TreeSeg> &segs, int code, const XRows &x, bool tree) {
  const int C = static_cast<int>(g.d.size()), H = g.d[0].s.hidden, rows = static_cast<int>(x.size());
  std::vector<int32_t> w(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
  int bound = 0;
  for (size_t j = 0; j < segs.size(); ++j) {
    int32_t *r = &w[j * k::kSegRec];
    r[k::kSegPos] = segs[j].pos;
    r[k::kSegSlot] = segs[j].slot;
    if (!segs[j].par.empty()) k::rec_word_set(r, k::kSegTree, k::kSegTreeHi, k::tree_word(segs[j].par.data(), segs[j].n));
    bound = std::max(bound, segs[j].pos + segs[j].n);
  }
  std::vector<uint16_t> xb(static_cast<size_t>(k::kStepRows32) * H, 0);
  for (int r = 0; r < rows; ++r) std::copy(x[r].begin(), x[r].end(), xb.begin() + static_cast<size_t>(r) * H);
  for (int c = 0; c < C; ++c) {
    Card &d = g.d[c];
    d.q->memcpy(g.rec[c], w.data(), w.size() * 4);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2);
    k::qsa_decode(*d.q, d.s, rows, d.x, d.w, g.c[c], d.rope, k::QsaStep{g.rec[c], bound, k::kSegRec, code, tree}, d.sc, C > 1 ? &d.wire : nullptr,
                  g.parity, d.out);
  }
  g.parity ^= 1;
  TreeOut o;
  o.key_stride = k::qsa_score_words(kTreeCap) / 4;
  for (int c = 0; c < C; ++c) {
    Card &d = g.d[c];
    d.q->wait_and_throw();
    o.out.push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.out), static_cast<size_t>(rows) * H));
    o.attn.push_back(rd(*d.q, reinterpret_cast<const uint16_t *>(d.sc.attn), static_cast<size_t>(rows) * d.s.o_k()));
    o.keys.push_back(rd(*d.q, d.sc.score, static_cast<size_t>(rows) * o.key_stride));
    o.nblk.push_back(rd(*d.q, d.sc.nblk, static_cast<size_t>(rows)));
  }
  return o;
}
// row ra of a == row rb of b, every card: out, attn, the complete blocks and their score keys
bool same_row(const TreeRig &g, const TreeOut &a, int ra, const TreeOut &b, int rb) {
  bool ok = true;
  for (size_t c = 0; c < g.d.size(); ++c) {
    const size_t H = static_cast<size_t>(g.d[c].s.hidden), K = static_cast<size_t>(g.d[c].s.o_k()), S = a.key_stride;
    ok &= std::equal(a.out[c].begin() + ra * H, a.out[c].begin() + (ra + 1) * H, b.out[c].begin() + rb * H);
    ok &= std::equal(a.attn[c].begin() + ra * K, a.attn[c].begin() + (ra + 1) * K, b.attn[c].begin() + rb * K);
    const int nb = a.nblk[c][ra];
    ok &= nb == b.nblk[c][rb] && std::equal(a.keys[c].begin() + ra * S, a.keys[c].begin() + ra * S + nb, b.keys[c].begin() + rb * S);
  }
  return ok;
}
// The state a path leaves for the next steps on one card: K / V / scales of every kv head at positions p0 - 3..
// p0 + n - 1, their ring entries, the compressed keys of the blocks from the last committed one to (p0 + n) / 4.
std::vector<uint8_t> path_state(TreeRig &g, int card, int slot, int p0, int n) {
  Card &d = g.d[card];
  const k::QsaCache &c = g.c[card];
  const auto &s = d.s;
  const int hkv = s.hkv(), krow = 256 * s.kv_elem(), vrow = s.v_row_bytes(), vsb = s.v_scale_bytes();
  const bool scaled = k::qsa_kv_scaled(s.kv);
  const int t0 = std::max(0, p0 - 3), t1 = p0 + n;
  auto page_of = [&](int t) { return static_cast<size_t>(g.table[static_cast<size_t>(slot) * g.lp + t / k::kQsaPage]); };
  std::vector<uint8_t> st;
  auto take = [&](const void *base, size_t off, size_t bytes) {
    const auto v = rd(*d.q, static_cast<const uint8_t *>(base) + off, bytes);
    st.insert(st.end(), v.begin(), v.end());
  };
  for (int t = t0; t < t1; ++t)
    for (int h = 0; h < hkv; ++h) {
      const size_t sl = (page_of(t) * hkv + h) * k::kQsaPage + t % k::kQsaPage;
      take(c.k, sl * krow, krow);
      take(c.v, sl * vrow, vrow);
      if (scaled) {
        take(c.ks, sl * 4, 4);
        take(c.vs, sl * vsb, vsb);
      }
    }
  const size_t ring0 = static_cast<size_t>(slot) * k::kQsaRing * 128 * 2;
  for (int t = t0; t < t1; ++t) take(c.ring, ring0 + static_cast<size_t>(t % k::kQsaRing) * 256, 256);
  for (int b = std::max(0, p0 / 4 - 1); b < t1 / 4; ++b) take(c.kc, (page_of(b * 4) * 16 + b % 16) * 256, 256);
  return st;
}

struct TreeTally {
  size_t rows = 0, rows_ok = 0, chain = 0, chain_ok = 0, moves = 0, moves_ok = 0, next = 0, next_ok = 0;
  int shown = 0;
  void fail(const std::string &what) {
    if (shown++ < 12) std::printf("  tree mismatch: %s\n", what.c_str());
  }
};

std::vector<int> tree_path(const std::vector<int> &par, int j) {
  std::vector<int> p;
  for (int x = j; x >= 0; x = par[x]) p.insert(p.begin(), x);
  return p;
}

// One tree step (segs, code) checked against its chains: rows, chain segments, and two accepted paths per tree segment.
void tree_case(TreeRig &g, const std::vector<TreeSeg> &segs, int code, uint32_t seed, TreeTally &t, const std::string &tag) {
  const int H = g.d[0].s.hidden;
  XRows x;
  for (const auto &sg : segs)
    for (int j = 0; j < sg.n; ++j) x.push_back(row_x(H, seed + static_cast<uint32_t>(x.size())));
  restore(g);
  const TreeOut tr = tree_step(g, segs, code, x, true);
  auto xs_of = [&](int rb, const std::vector<int> &rows) {
    XRows v;
    for (const int r : rows) v.push_back(x[static_cast<size_t>(rb + r)]);
    return v;
  };
  char what[256];
  for (size_t si = 0, rb = 0; si < segs.size(); rb += static_cast<size_t>(segs[si].n), ++si) {
    const TreeSeg &sg = segs[si];
    const int r0 = static_cast<int>(rb);
    if (sg.par.empty()) {  // a chain segment of the tree step == the chain alone
      std::vector<int> all(sg.n);
      std::iota(all.begin(), all.end(), 0);
      restore(g);
      const TreeOut ch = tree_step(g, {{sg.slot, sg.pos, sg.n}}, 0, xs_of(r0, all), false);
      for (int j = 0; j < sg.n; ++j) {
        ++t.chain;
        if (same_row(g, tr, r0 + j, ch, j)) ++t.chain_ok;
        else std::snprintf(what, sizeof what, "%s: chain segment %zu row %d", tag.c_str(), si, j), t.fail(what);
      }
      continue;
    }
    for (int j = 0; j < sg.n; ++j) {  // rows: each == the last row of the chain of its path
      const auto path = tree_path(sg.par, j);
      restore(g);
      const TreeOut ch = tree_step(g, {{sg.slot, sg.pos, static_cast<int>(path.size())}}, 0, xs_of(r0, path), false);
      ++t.rows;
      if (same_row(g, tr, r0 + j, ch, static_cast<int>(path.size()) - 1)) ++t.rows_ok;
      else std::snprintf(what, sizeof what, "%s: segment %zu row %d (depth %zu) vs its path's chain", tag.c_str(), si, j, path.size() - 1), t.fail(what);
    }
    // accepted paths: the last row's, and the first deepest row's
    int deep = 0;
    for (int j = 1; j < sg.n; ++j)
      if (tree_path(sg.par, j).size() > tree_path(sg.par, deep).size()) deep = j;
    std::vector<int> ends = {sg.n - 1};
    if (deep != sg.n - 1) ends.push_back(deep);
    for (const int end : ends) {
      const auto path = tree_path(sg.par, end);
      const int n = static_cast<int>(path.size());
      k::QsaMove mv;
      mv.n = n;
      std::copy(path.begin(), path.end(), mv.from);
      const XRows x1 = {row_x(H, seed + 9001)};
      XRows x4;
      for (int r = 0; r < 4; ++r) x4.push_back(row_x(H, seed + 9002 + static_cast<uint32_t>(r)));
      struct After {
        std::vector<std::vector<uint8_t>> state;
        TreeOut n1, n4;
      };
      auto after = [&](bool via_tree) {
        restore(g);
        if (via_tree) {
          tree_step(g, segs, code, x, true);
          for (size_t c = 0; c < g.d.size(); ++c) k::qsa_kv_move(*g.d[c].q, g.d[c].s, &g.c[c], 1, sg.slot, sg.pos, r0, mv);
        } else {
          tree_step(g, {{sg.slot, sg.pos, n}}, 0, xs_of(r0, path), false);
        }
        After a;
        for (size_t c = 0; c < g.d.size(); ++c) {
          g.d[c].q->wait_and_throw();
          a.state.push_back(path_state(g, static_cast<int>(c), sg.slot, sg.pos, n));
        }
        a.n1 = tree_step(g, {{sg.slot, sg.pos + n, 1}}, 0, x1, false);
        a.n4 = tree_step(g, {{sg.slot, sg.pos + n, 4}}, 0, x4, false);
        return a;
      };
      const After a = after(true), b = after(false);
      ++t.moves;
      if (a.state == b.state) ++t.moves_ok;
      else std::snprintf(what, sizeof what, "%s: segment %zu path to row %d: state after the move", tag.c_str(), si, end), t.fail(what);
      bool nx = true;
      for (int r = 0; r < 4; ++r) nx &= same_row(g, a.n4, r, b.n4, r);
      nx &= same_row(g, a.n1, 0, b.n1, 0);
      ++t.next;
      if (nx) ++t.next_ok;
      else std::snprintf(what, sizeof what, "%s: segment %zu path to row %d: the next steps", tag.c_str(), si, end), t.fail(what);
    }
  }
}

const std::vector<std::vector<int>> &tree_shapes() {
  static const std::vector<std::vector<int>> v = {
      {-1, 0, 1, 2, 3, 4, 5, 6},  // a chain as a tree (depth 7: two blocks closed)
      {-1, 0, 1, 2, 0, 4, 1, 6},  // branches at the root and at depth 1
      {-1, 0, 0, 0, 1, 2, 3, 4},  // three children of the root, one grandchild each
      {-1, 0, 1, 2, 3, 4, 5, 6, 1},  // depth 7 and a sibling at depth 2 (9 rows)
      {-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11}};  // two depth-6 branches of the root (13 rows = kQsaTreeRows)
  return v;
}

void tree_report(const TreeTally &t, const std::string &who) {
  char what[512];
  std::snprintf(what, sizeof what, "%s: every tree row == the last row of the chain step of its path: out, attn, block scores, every card (%zu / %zu rows)",
                who.c_str(), t.rows_ok, t.rows);
  expect(t.rows > 0 && t.rows_ok == t.rows, what);
  std::snprintf(what, sizeof what,
                "%s: tree step + qsa_kv_move of a path == the chain step of the path: K / V / scales, ring entries, compressed keys (%zu / %zu "
                "paths); the next 1-row and 4-row steps bit-identical (%zu / %zu)",
                who.c_str(), t.moves_ok, t.moves, t.next_ok, t.next);
  expect(t.moves > 0 && t.moves_ok == t.moves && t.next_ok == t.next, what);
  if (t.chain) {
    std::snprintf(what, sizeof what, "%s: chain segments of tree steps == the chain alone (%zu / %zu rows)", who.c_str(), t.chain_ok, t.chain);
    expect(t.chain_ok == t.chain, what);
  }
}

// Single tree segments at each context x p0 % 4 x shape, then mixed layouts, on the fabric's first C cards. contexts:
// the first positions' bases.
void test_trees(xenithium::Fabric &f, int C, const Model &m, k::QsaKv kv, const std::vector<int> &contexts, const char *tag) {
  TreeRig g = make_tree_rig(f, C, m, kv);
  const std::string who = std::string(tag) + ", " + kv_name(kv) + " KV, " + std::to_string(C) + (C == 1 ? " card" : " cards");
  TreeTally t;
  uint32_t seed = 66000;
  const auto &shapes = tree_shapes();
  for (const int base : contexts)
    for (int mod = 0; mod < 4; ++mod)
      for (size_t sh = 0; sh < shapes.size(); ++sh) {
        const TreeSeg sg{static_cast<int>(sh % kTreeSlots), base + mod, static_cast<int>(shapes[sh].size()), shapes[sh]};
        tree_case(g, {sg}, 0, seed += 100, t, who + " p0 " + std::to_string(sg.pos) + " shape " + std::to_string(sh));
      }
  // mixed steps: trees beside chains, own slots and positions (the first context's base and the last's)
  const int b0 = contexts.front(), b1 = contexts.back();
  const std::vector<int> t4a = {-1, 0, 0, 1}, t4b = {-1, 0, 1, 1};
  struct Mixed {
    std::vector<TreeSeg> segs;
    int code;
  };
  const std::vector<Mixed> mixed = {
      {{{0, b1 + 1, 8, shapes[1]}, {1, b0 + 2, 5}}, 8},  // seg_rows 8, the last short
      {{{2, b0 + 3, 8, shapes[2]}, {0, b1 + 2, 8, shapes[0]}, {1, b1, 1}, {3, b0 + 1, 1}}, k::seg_het(8, 2, 4)},  // two trees + 1-row chains
      {{{1, b1 + 3, 4, t4a}, {2, b0, 4}, {3, b1 + 1, 4, t4b}}, 4},  // 4-row segments
      {{{3, b0 + 1, 9, shapes[3]}, {0, b0 + 3, 1}, {2, b1 + 2, 1}}, k::seg_het(9, 1, 3)},
      {{{0, b1 + 2, 13, shapes[4]}, {1, b1 + 1, 4}}, k::seg_het2(13, 1, 2, 4)}};
  for (size_t i = 0; i < mixed.size(); ++i) tree_case(g, mixed[i].segs, mixed[i].code, seed += 100, t, who + " mixed " + std::to_string(i));
  tree_report(t, who);
  for (auto &d : g.d) free_card(f, d);
}

// The refusals: qsa_store and over-long / tkc-less tree steps, malformed moves (one card).
void test_tree_refusals(xenithium::Fabric &f, const Model &m) {
  Card d = make_card(f, 0, m, 1, k::QsaKv::i8, 1, false, 0);
  bf16 *tkc = static_cast<bf16 *>(d.dev(f, 0, k::kQsaTreeKcElems * 2));
  auto throws = [&](auto &&fn) {
    try {
      fn();
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  k::QsaCache c = d.c;
  c.tkc = tkc;
  const k::QsaStep tree8{d.pos, 64, k::kSegRec, 0, true};
  bool ok = throws([&] { k::qsa_store(*d.q, d.s, 8, d.x, d.w, c, d.rope, tree8, d.sc); });
  ok &= throws([&] { k::qsa_decode(*d.q, d.s, 8, d.x, d.w, d.c, d.rope, tree8, d.sc, nullptr, 0, d.out); });  // no tkc
  ok &= throws([&] { k::qsa_decode(*d.q, d.s, 14, d.x, d.w, c, d.rope, k::QsaStep{d.pos, 64, k::kSegRec, 0, true}, d.sc, nullptr, 0, d.out); });
  ok &= throws([&] { k::qsa_decode(*d.q, d.s, 8, d.x, d.w, c, d.rope, k::QsaStep{d.pos, 64, 0, 0, true}, d.sc, nullptr, 0, d.out); });  // no records
  k::QsaMove bad;
  bad.n = 3;
  bad.from[1] = 2, bad.from[2] = 2;  // not increasing
  ok &= throws([&] { k::qsa_kv_move(*d.q, d.s, &c, 1, 0, 100, 0, bad); });
  bad.from[0] = 1, bad.from[1] = 2, bad.from[2] = 3;  // from[0] != 0
  ok &= throws([&] { k::qsa_kv_move(*d.q, d.s, &c, 1, 0, 100, 0, bad); });
  ok &= throws([&] { k::qsa_kv_move(*d.q, d.s, &d.c, 1, 0, 100, 0, k::QsaMove{2, {0, 1}}); });  // no tkc
  d.q->wait_and_throw();
  expect(ok, "trees: qsa_store refuses a tree step; qsa_decode refuses one without tkc, records or with more than 13 rows a segment; "
             "qsa_kv_move refuses a path not starting at 0 / not increasing, or a cache without tkc");
  free_card(f, d);
}

}  // namespace

// ---------------------------------------------------------------- QsaStep:every
// The plan to 2,400 tokens with every on each step whose bound allows it (rows of at most 512 complete blocks: no score
// and no select launch) against the same plan with them: out and attn of every step, bit for bit. A bound past 2,048
// tokens with every refuses.
void test_every(xenithium::Fabric &f, const Model &m, k::QsaKv kv, const char *tag) {
  const auto plan = make_plan(2400, 77);
  int n_every = 0;
  for (const auto &st : plan) n_every += !st.bound && st.pos + st.rows <= k::kQsaTopBlocks * k::kQsaCompress;
  Card d = make_card(f, 0, m, 1, kv, 1, false, 0);
  const Trace a = run_plan(d, m, plan, 1000, 0, {}, nullptr, nullptr, nullptr, nullptr);
  free_card(f, d);
  Card e = make_card(f, 0, m, 1, kv, 1, false, 0);
  g_every = true;
  const Trace b = run_plan(e, m, plan, 1000, 0, {}, nullptr, nullptr, nullptr, nullptr);
  g_every = false;
  char what[400];
  std::snprintf(what, sizeof what, "%s, %s KV: %d of %zu steps without score / select (QsaStep::every) == the plan with them: out, attn, bit for bit", tag,
                kv_name(kv), n_every, plan.size());
  expect(n_every > 0 && a.out == b.out && a.attn == b.attn, what);
  bool refused = false;
  try {
    k::QsaStep step{e.pos, k::kQsaTopBlocks * k::kQsaCompress + 4};
    step.every = true;
    k::qsa_decode(*e.q, e.s, 4, e.x, e.w, e.c, e.rope, step, e.sc, nullptr, 0, e.out);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  expect(refused, "QsaStep::every with a bound past 2,048 tokens refuses");
  free_card(f, e);
}

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const bool two = loader.size() > 1;
    xenithium::Fabric f(loader, two ? std::vector<size_t>{0, 1} : std::vector<size_t>{0});
    test_select(f);
    const Model model = make_model(2560, 21);
    test_decode(f, model, k::QsaKv::i8, "model dims (h 2560, 24 q / 2 kv heads)");
    test_decode(f, model, k::QsaKv::bf16, "model dims (h 2560, 24 q / 2 kv heads)");
    test_decode(f, model, k::QsaKv::i8v4, "model dims (h 2560, 24 q / 2 kv heads)");  // K int8, V int4
    test_segments(f, model, k::QsaKv::i8, "model dims, int8 KV:");
    test_segments(f, model, k::QsaKv::i8v4, "model dims, int8-v4 KV:");
    test_every(f, model, k::QsaKv::i8v4, "model dims");
    test_every(f, model, k::QsaKv::bf16, "model dims");
    // draft trees on one card (every KV dtype, every context), two cards (int8, one context)
    test_tree_refusals(f, model);
    for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16, k::QsaKv::i8v4}) test_trees(f, 1, model, kv, {1500, 3000, 9000, 16400}, "trees");
    if (f.size() > 1) test_trees(f, 2, model, k::QsaKv::i8, {9000}, "trees");
    if (f.size() > 1) {
      test_cards(f, model, k::QsaKv::i8, "int8 KV");
      test_cards(f, model, k::QsaKv::bf16, "bf16 KV");
      test_cards(f, model, k::QsaKv::i8v4, "int8-v4 KV");
    }
    if (loader.size() >= 4) {  // TP4 (a kv head on two cards, 6 q heads per card)
      xenithium::Fabric f4(loader, {0, 1, 2, 3});
      test_cards(f4, model, k::QsaKv::i8, "int8 KV");
      test_cards(f4, model, k::QsaKv::bf16, "bf16 KV");
      test_cards(f4, model, k::QsaKv::i8v4, "int8-v4 KV");
      test_cards_owner(f4, model, k::QsaKv::i8v4, "int8-v4 KV");
      test_cards_owner(f4, model, k::QsaKv::bf16, "bf16 KV");
      test_rows_cards(f4, k::QsaKv::i8, 2000, 300);
      test_rows_cards(f4, k::QsaKv::i8v4, 2000, 300);
      for (const k::QsaKv kv : {k::QsaKv::i8, k::QsaKv::bf16, k::QsaKv::i8v4}) test_trees(f4, 4, model, kv, {3000, 16400}, "trees");
    } else {
      std::printf("[SKIP] four-card QSA: %zu card(s) visible\n", loader.size());
    }
    {
      const Model im = make_int_model(2560, 23);
      test_rows(f, im, k::QsaKv::i8, 6, 37, "integer data,");
      test_rows(f, im, k::QsaKv::i8, 2000, 300, "integer data,");
      test_rows(f, im, k::QsaKv::bf16, 2000, 300, "integer data,");
      test_rows(f, im, k::QsaKv::i8v4, 2000, 300, "integer data,");
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
