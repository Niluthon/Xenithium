// Gate for the dense gated GQA decode (kernels/attention/gqa.hpp;): the W4 qkv projection, q / k norms +
// rope, the K / V cache rows, attention over the whole context in context splits with a running softmax, the LSE
// merge + sigmoid gate, the W4 o_proj and the two-card sum. Synthetic W4 weights (random codes, signed fp16 scales
// as AutoRound stores them) at a small shape; the reference is teacher-forced at the device's projections:
// q vs norm + rope of the device's projections: 1 ulp flips only (<= 1 % of elements)
// K / V the rows written this step, dequantised, vs the reference from the device's projections
// attn vs softmax(q K / 16) V over every token 0..p, K / V as stored (2 ulps: bf16(o) then the gate)
// out vs the reference o_proj (dequantised W4) of the device's attn, summed over the cards (2^-7)
// proj the qkv GEMM vs the dequantised reference from x (2^-7; w4load / mlp35 gate the GEMM itself)
// over a history past 4,096 tokens (tiles per split 1 -> 2) and on a random 40,000-token cache (10 tiles per
// split, a short last split); and bit-exact identities: run-to-run; an eager pos_bound == the capacity bound (the
// grid's split count differs, the cut does not); "4 rows at P" == "1 row at P" and "2 rows at P + 2"; two
// segments of 2 rows (records) == one 4-row segment; identity page table == a permuted table in a pool twice the
// size (unused pages poisoned); two cards: both cards' out bit-identical.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
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
  bool ok(double bound, double frac) const { return n > 0 && scaled <= bound && off <= frac * n; }
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

// ---- a W4 matrix: codes [n][k] (0..15, w = (q - 8) x scale), fp16 scales [n][k / 128] ----
struct W4Host {
  int n = 0, kk = 0;
  std::vector<uint8_t> code;
  std::vector<float> scale;  // fp16 values
  float w(int r, int c) const { return (static_cast<int>(code[static_cast<size_t>(r) * kk + c]) - 8) * scale[static_cast<size_t>(r) * (kk / 128) + c / 128]; }
};
W4Host make_w4(int n, int kk, float sd, std::mt19937 &rng) {
  W4Host m;
  m.n = n;
  m.kk = kk;
  m.code.resize(static_cast<size_t>(n) * kk);
  m.scale.resize(static_cast<size_t>(n) * (kk / 128));
  std::uniform_int_distribution<int> ud(0, 15);
  std::uniform_real_distribution<float> us(0.5f, 1.5f);
  for (auto &c : m.code) c = static_cast<uint8_t>(ud(rng));
  for (size_t i = 0; i < m.scale.size(); ++i) m.scale[i] = static_cast<float>(sycl::half((i % 2 ? -1.f : 1.f) * us(rng) * sd / 4.6f));  // |q - 8| ~ 4.6 rms
  return m;
}
// rows [r0, r0 + nr) of each listed (matrix, r0, nr), stacked; columns [c0, c0 + nc)
W4Host slice(const std::vector<std::tuple<const W4Host *, int, int>> &parts, int c0, int nc) {
  W4Host s;
  s.kk = nc;
  for (const auto &[m, r0, nr] : parts) {
    for (int r = r0; r < r0 + nr; ++r) {
      s.code.insert(s.code.end(), m->code.begin() + static_cast<size_t>(r) * m->kk + c0, m->code.begin() + static_cast<size_t>(r) * m->kk + c0 + nc);
      s.scale.insert(s.scale.end(), m->scale.begin() + static_cast<size_t>(r) * (m->kk / 128) + c0 / 128,
                     m->scale.begin() + static_cast<size_t>(r) * (m->kk / 128) + (c0 + nc) / 128);
    }
    s.n += nr;
  }
  return s;
}

struct Model {
  k::GqaShape s;  // cards = 1
  W4Host q, kp, vp, o;
  std::vector<float> qn, kn;
};
Model make_model(int H, int QH, int KVH, uint32_t seed) {
  Model m;
  m.s = k::GqaShape{H, QH, KVH, 1, 0, 1e-6f, k::QsaKv::i8};
  std::mt19937 rng(seed);
  const float sw = 1.f / std::sqrt(static_cast<float>(H));
  m.q = make_w4(QH * 512, H, sw, rng);
  m.kp = make_w4(KVH * 256, H, sw, rng);
  m.vp = make_w4(KVH * 256, H, sw, rng);
  m.o = make_w4(H, QH * 256, 1.f / std::sqrt(static_cast<float>(QH * 256)), rng);
  std::normal_distribution<float> nd(0.f, 0.3f);
  m.qn.resize(256);
  m.kn.resize(256);
  for (auto &v : m.qn) v = bfr(nd(rng));
  for (auto &v : m.kn) v = bfr(nd(rng));
  return m;
}

// One card's buffers: weights, cache, scratch.
struct Card {
  sycl::queue *q = nullptr;
  k::GqaShape s;
  k::GqaWeights w;
  W4Host qkv_h, o_h;  // this card's slices (reference)
  k::GqaCache c;
  k::GqaScratch sc;
  k::PairWire wire;
  int32_t *pos = nullptr;
  bf16 *rope = nullptr, *out = nullptr;
  std::vector<void *> mem;
  void *dev(xenithium::Fabric &f, int card, size_t bytes) {
    void *p = f.alloc(card, std::max<size_t>(bytes, 4096));
    mem.push_back(p);
    return p;
  }
};
k::W4Mat upload_w4(xenithium::Fabric &f, int card, Card &cd, const W4Host &h) {
  std::vector<uint32_t> qw(static_cast<size_t>(h.kk / 8) * h.n, 0u);
  for (int r = 0; r < h.n; ++r)
    for (int c = 0; c < h.kk; ++c) qw[static_cast<size_t>(c / 8) * h.n + r] |= static_cast<uint32_t>(h.code[static_cast<size_t>(r) * h.kk + c]) << (4 * (c % 8));
  std::vector<uint8_t> blocks(static_cast<size_t>(h.n) * h.kk / 2);
  k::pack_w4_gptq(qw.data(), h.n, h.kk, blocks.data());
  std::vector<sycl::half> sc(static_cast<size_t>(h.kk / 128) * h.n);
  for (int r = 0; r < h.n; ++r)
    for (int g = 0; g < h.kk / 128; ++g) sc[static_cast<size_t>(g) * h.n + r] = sycl::half(h.scale[static_cast<size_t>(r) * (h.kk / 128) + g]);
  auto *wb = static_cast<uint8_t *>(cd.dev(f, card, blocks.size()));
  auto *sb = static_cast<sycl::half *>(cd.dev(f, card, sc.size() * 2));
  f.compute(card).memcpy(wb, blocks.data(), blocks.size());
  f.compute(card).memcpy(sb, sc.data(), sc.size() * 2).wait();
  k::W4Mat m;
  m.w = wb;
  m.scales = sb;
  m.n = h.n;
  m.k = h.kk;
  return m;
}
void make_cache(xenithium::Fabric &f, int card, Card &cd, int cap, const std::vector<int32_t> &pages, int phys_pages) {
  const auto &s = cd.s;
  cd.c.k = static_cast<uint8_t *>(cd.dev(f, card, phys_pages * k::gqa_page_kv_bytes(s)));
  cd.c.v = static_cast<uint8_t *>(cd.dev(f, card, phys_pages * k::gqa_page_v_bytes(s)));
  cd.c.ks = static_cast<float *>(cd.dev(f, card, phys_pages * k::gqa_page_scale_floats(s) * 4));
  cd.c.vs = static_cast<float *>(cd.dev(f, card, phys_pages * k::gqa_page_vscale_floats(s) * 4));
  auto *pg = static_cast<int32_t *>(cd.dev(f, card, pages.size() * 4));
  f.compute(card).memcpy(pg, pages.data(), pages.size() * 4).wait();
  cd.c.pages = pg;
  cd.c.max_tokens = cap;
}
Card make_card(xenithium::Fabric &f, int card, const Model &m, k::QsaKv kv, int cards, int cap, const std::vector<uint16_t> &rope) {
  Card cd;
  cd.q = &f.compute(card);
  cd.s = m.s;
  cd.s.cards = cards;
  cd.s.card = card;
  cd.s.kv = kv;
  const auto &s = cd.s;
  const int H = s.hidden, hq = s.hq(), hkv = s.hkv();
  cd.qkv_h = slice({{&m.q, card * hq * 512, hq * 512}, {&m.kp, card * hkv * 256, hkv * 256}, {&m.vp, card * hkv * 256, hkv * 256}}, 0, H);
  cd.o_h = slice({{&m.o, 0, H}}, card * s.o_k(), s.o_k());
  cd.w.qkv = upload_w4(f, card, cd, cd.qkv_h);
  cd.w.o = upload_w4(f, card, cd, cd.o_h);
  std::vector<uint16_t> nb(512);
  for (int i = 0; i < 256; ++i) nb[i] = k::bf16_bits(m.qn[i]), nb[256 + i] = k::bf16_bits(m.kn[i]);
  auto *nw = static_cast<bf16 *>(cd.dev(f, card, 1024));
  f.compute(card).memcpy(nw, nb.data(), 1024).wait();
  cd.w.q_norm = nw;
  cd.w.k_norm = nw + 256;
  std::vector<int32_t> pages(cap / 64);
  std::iota(pages.begin(), pages.end(), 0);
  make_cache(f, card, cd, cap, pages, cap / 64);
  cd.sc.y = static_cast<float *>(cd.dev(f, card, 16ull * s.qkv_n() * 4));
  cd.sc.q = static_cast<bf16 *>(cd.dev(f, card, 16ull * hq * 256 * 2));
  cd.sc.part = static_cast<float *>(cd.dev(f, card, k::gqa_part_floats(s) * 4));
  cd.sc.attn = static_cast<bf16 *>(cd.dev(f, card, 16ull * s.o_k() * 2));
  cd.sc.partial = static_cast<float *>(cd.dev(f, card, 16ull * H * 4));
  f.compute(card).memset(cd.sc.attn, 0, 16ull * s.o_k() * 2).wait();
  cd.pos = static_cast<int32_t *>(cd.dev(f, card, 4096));
  cd.rope = static_cast<bf16 *>(cd.dev(f, card, rope.size() * 2));
  f.compute(card).memcpy(cd.rope, rope.data(), rope.size() * 2).wait();
  cd.out = static_cast<bf16 *>(cd.dev(f, card, 16ull * H * 2));
  if (cards > 1) {
    cd.wire.inbox = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_wire_words(H) * 4));
    cd.mem.push_back(cd.wire.inbox);
    cd.wire.error = static_cast<uint32_t *>(cd.dev(f, card, 64));
    f.compute(card).memset(cd.wire.error, 0, 64).wait();
  }
  return cd;
}
void free_card(xenithium::Fabric &f, Card &cd) {
  for (void *p : cd.mem) f.release(p);
  cd.mem.clear();
}

// Positions / segment records into each card's pos buffer, then one step on every card.
struct Run {
  xenithium::Fabric &f;
  std::vector<Card> &cards;
  int parity = 0;
  int attn_rows = 0, attn_grf = 0;  // the attention group (0 = auto, 1 = per row)
  int attn_pf = -1;  // the group kernel's L2 prefetch (-1 = auto)
  int select = 0;  // GqaStep:select (pages)
  std::vector<int32_t *> select_slots{};  // GqaStep:select_slots per card (empty: none)
  bool tree = false;  // GqaStep:tree
  void step(int rows, const std::vector<int32_t> &words, int pos_bound, const std::vector<bf16 *> &x, int stride = 0, int seg_rows = 0) {
    for (size_t c = 0; c < cards.size(); ++c) {
      Card &cd = cards[c];
      cd.q->memcpy(cd.pos, words.data(), words.size() * 4).wait();
      k::GqaStep st{cd.pos, pos_bound, stride, seg_rows};
      st.attn_rows = attn_rows;
      st.attn_grf = attn_grf;
      st.attn_pf = attn_pf;
      st.select = select;
      if (!select_slots.empty()) st.select_slots = select_slots[c];
      st.tree = tree;
      k::gqa_decode(*cd.q, cd.s, rows, x[c], cd.w, cd.c, cd.rope, st, cd.sc,
                    cards.size() > 1 ? &cd.wire : nullptr, parity, cd.out);
    }
    parity ^= 1;
    for (auto &cd : cards) cd.q->wait_and_throw();
  }
};

// K or V of token t, head kvh, dim d as stored (int8 dequantised with its scale)
float kv_at(const k::GqaShape &s, const std::vector<uint8_t> &pool, const std::vector<float> &sc, int t, int kvh, int d, bool isv = false) {
  const size_t slot = (static_cast<size_t>(t / 64) * s.hkv() + kvh) * 64 + t % 64;
  if (isv && s.kv == k::QsaKv::i8v4) {  // a nibble code x its group's bf16 scale
    const int b = pool[slot * 128 + d / 2], c = (d & 1) ? (b >> 4) - ((b & 128) >> 3) : (b & 15) - ((b & 8) << 1);
    uint16_t sb;
    std::memcpy(&sb, reinterpret_cast<const uint8_t *>(sc.data()) + (slot * 8 + d / 32) * 2, 2);
    return static_cast<float>(c) * k::bf16_value(sb);
  }
  if (s.kv != k::QsaKv::bf16) return static_cast<float>(static_cast<int8_t>(pool[slot * 256 + d])) * sc[slot];
  uint16_t b;
  std::memcpy(&b, &pool[(slot * 256 + d) * 2], 2);
  return k::bf16_value(b);
}

struct Gates {
  BoundCmp q, kv, attn, out, proj;
  std::string text(const char *what) const;
};
// Teacher-forced checks of the step just run (identity page tables): rows at pos.. pos + rows - 1; xh the live rows
// of x as values. q from the device's projections; K / V of these rows vs the reference from the device's
// projections; attn vs softmax over tokens 0..p as stored; out vs the per-card o references summed.
// keep: the step's selected pages per kv head (one set for every row); keep_by_row: each row's set (a prefill call's row
// blocks select apart), a null entry or an empty list = every page.
using PageSets = std::vector<std::vector<int32_t>>;  // [kv head] -> pages, ascending
void check_step(std::vector<Card> &cards, const Model &m, int rows, int pos, const std::vector<uint16_t> &rope,
                const std::vector<std::vector<float>> &xh, Gates &g, bool proj, const PageSets *keep = nullptr,
                const std::vector<const PageSets *> *keep_by_row = nullptr) {
  const int H = cards[0].s.hidden;
  std::vector<double> oref(static_cast<size_t>(rows) * H, 0.0);
  for (auto &cd : cards) {
    const auto &s = cd.s;
    const int hq = s.hq(), hkv = s.hkv(), ldy = s.qkv_n(), G = s.grp(), D = 256, cap = cd.c.max_tokens;
    const auto y = rd(*cd.q, cd.sc.y, static_cast<size_t>(rows) * ldy);
    const auto qd = rd16(*cd.q, cd.sc.q, static_cast<size_t>(rows) * hq * D);
    const auto attn = rd16(*cd.q, cd.sc.attn, static_cast<size_t>(rows) * s.o_k());
    const size_t NP = cap / 64;
    const auto kp = rd(*cd.q, cd.c.k, NP * k::gqa_page_kv_bytes(s)), vp = rd(*cd.q, cd.c.v, NP * k::gqa_page_v_bytes(s));
    std::vector<float> ksp(NP * k::gqa_page_scale_floats(s), 1.f), vsp(NP * k::gqa_page_vscale_floats(s), 1.f);
    if (s.kv != k::QsaKv::bf16) ksp = rd(*cd.q, cd.c.ks, ksp.size()), vsp = rd(*cd.q, cd.c.vs, vsp.size());
    for (int r = 0; r < rows; ++r) {
      const int p = pos + r;
      const uint16_t *rr = &rope[static_cast<size_t>(p) * 64];
      std::vector<float> v(D), o(D);
      if (proj) {  // the W4 projection from x
        std::vector<double> ref(ldy);
        for (int n = 0; n < ldy; ++n) {
          double a = 0;
          for (int j = 0; j < H; ++j) a += static_cast<double>(xh[r][j]) * cd.qkv_h.w(n, j);
          ref[n] = a;
        }
        g.proj.add_row(&y[static_cast<size_t>(r) * ldy], ref.data(), ldy);
      }
      for (int h = 0; h < hq; ++h) {
        for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + j]);
        k::qsa_ref_norm_rope(v.data(), m.qn.data(), D, rr, s.eps, o.data());
        g.q.add_rowf(&qd[(static_cast<size_t>(r) * hq + h) * D], o.data(), D);
      }
      for (int kh = 0; kh < hkv; ++kh) {  // this row's K and V as stored
        for (int j = 0; j < D; ++j) v[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + kh * D + j]);
        k::qsa_ref_norm_rope(v.data(), m.kn.data(), D, rr, s.eps, o.data());
        std::vector<float> vv(D);
        for (int j = 0; j < D; ++j) vv[j] = bfr(y[static_cast<size_t>(r) * ldy + s.q_n() + (hkv + kh) * D + j]);
        for (auto *ref : {&o, &vv}) {
          std::vector<float> want(*ref), dev(D);
          if (s.kv == k::QsaKv::i8v4 && ref == &vv) {  // qsa_ref_v4's codes and bf16 group scales
            std::vector<int8_t> cdd(D);
            std::vector<uint16_t> sg(D / 32);
            k::qsa_ref_v4(ref->data(), cdd.data(), sg.data());
            for (int j = 0; j < D; ++j) want[j] = static_cast<float>(cdd[j]) * k::bf16_value(sg[j / 32]);
          } else if (s.kv != k::QsaKv::bf16) {
            std::vector<int8_t> cdd(D);
            float scl;
            k::qsa_ref_i8(ref->data(), D, cdd.data(), &scl);
            for (int j = 0; j < D; ++j) want[j] = static_cast<float>(cdd[j]) * scl;
          }
          for (int j = 0; j < D; ++j) dev[j] = kv_at(s, ref == &o ? kp : vp, ref == &o ? ksp : vsp, p, kh, j, ref != &o);
          g.kv.add_rowf(dev.data(), want.data(), D);
        }
      }
      for (int kh = 0; kh < hkv; ++kh) {
        // the kv head's selected pages (ascending): the softmax over their tokens <= p only
        const PageSets *kept = keep_by_row ? (*keep_by_row)[static_cast<size_t>(r)] : keep;
        std::vector<int> toks;
        if (kept && !(*kept)[static_cast<size_t>(kh)].empty()) {
          for (int pg : (*kept)[static_cast<size_t>(kh)])
            for (int t = pg * 64; t < pg * 64 + 64 && t <= p; ++t) toks.push_back(t);
        } else {
          for (int t = 0; t <= p; ++t) toks.push_back(t);
        }
        const int nt = static_cast<int>(toks.size());
        std::vector<float> K(static_cast<size_t>(nt) * D), V(K.size());
        for (int i = 0; i < nt; ++i)
          for (int j = 0; j < D; ++j) {
            K[static_cast<size_t>(i) * D + j] = kv_at(s, kp, ksp, toks[static_cast<size_t>(i)], kh, j);
            V[static_cast<size_t>(i) * D + j] = kv_at(s, vp, vsp, toks[static_cast<size_t>(i)], kh, j, true);
          }
        for (int hh = 0; hh < G; ++hh) {
          const int h = kh * G + hh;
          std::vector<double> oo(D);
          k::qsa_ref_attend(&qd[(static_cast<size_t>(r) * hq + h) * D], nt, K.data(), V.data(), oo.data());
          std::vector<float> gate(D), ga(D);
          for (int j = 0; j < D; ++j) gate[j] = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * D + D + j]);
          k::qsa_ref_gate(oo.data(), gate.data(), ga.data());
          g.attn.add_rowf(&attn[(static_cast<size_t>(r) * hq + h) * D], ga.data(), D);
        }
      }
      for (int n = 0; n < H; ++n) {
        double acc = 0;
        for (int j = 0; j < s.o_k(); ++j) acc += static_cast<double>(attn[static_cast<size_t>(r) * s.o_k() + j]) * cd.o_h.w(n, j);
        oref[static_cast<size_t>(r) * H + n] += acc;
      }
    }
  }
  const auto o0 = rd16(*cards[0].q, cards[0].out, static_cast<size_t>(rows) * H);
  for (int r = 0; r < rows; ++r) g.out.add_row(&o0[static_cast<size_t>(r) * H], &oref[static_cast<size_t>(r) * H], H);
}

void report(const Gates &g, const std::string &label, bool kv8) {
  expect(g.proj.ok(1.0 / 128, 1.0), label + " qkv W4 projection (fp32, so the bf16 differ count is moot) vs the dequantised reference: " + g.proj.text());
  expect(g.q.ok(1.0 / 128, 0.01), label + " q vs norm + rope of the device's projections: " + g.q.text());
  expect(g.kv.ok(kv8 ? 1.0 / 16 : 1.0 / 128, kv8 ? 1.0 : 0.01), label + " K / V rows as stored vs reference: " + g.kv.text());
  expect(g.attn.ok(1.0 / 64, 0.005), label + " attn (gated) vs softmax over the whole context as stored: " + g.attn.text());
  expect(g.out.ok(1.0 / 128, 0.01), label + " out vs the o_proj reference of the device's attn, summed over the cards: " + g.out.text());
}

std::vector<bf16 *> upload_x(xenithium::Fabric &f, std::vector<Card> &cards, const std::vector<uint16_t> &xb) {
  std::vector<bf16 *> x;
  for (size_t c = 0; c < cards.size(); ++c) {
    auto *d = static_cast<bf16 *>(cards[c].dev(f, static_cast<int>(c), xb.size() * 2));
    cards[c].q->memcpy(d, xb.data(), xb.size() * 2).wait();
    x.push_back(d);
  }
  return x;
}
std::vector<uint16_t> rand_x(int H, std::mt19937 &rng, std::vector<std::vector<float>> *vals = nullptr) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> xb(8ull * H);
  for (auto &v : xb) v = k::bf16_bits(nd(rng));
  if (vals) {
    vals->assign(8, std::vector<float>(H));
    for (int r = 0; r < 8; ++r)
      for (int j = 0; j < H; ++j) (*vals)[r][j] = k::bf16_value(xb[static_cast<size_t>(r) * H + j]);
  }
  return xb;
}

// The history: steps of 1-4 rows (accepts 1-4) to past 4,096 tokens; checks on the first steps, across the
// 4,096-token split change and at the end; then the identities at the end position.
void test_decode(xenithium::Fabric &f, const Model &m, k::QsaKv kv, int ncards, const std::string &name) {
  const std::string label = name + ", " + kv_name(kv) + " KV, " + std::to_string(ncards) + " card(s):";
  const int H = m.s.hidden, cap = 4608, end_pos = 4200;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  for (int c = 0; c < ncards; ++c) cards.push_back(make_card(f, c, m, kv, ncards, cap, rope));
  if (ncards > 1)
    for (int c = 0; c < ncards; ++c) {
      cards[c].wire.peer_inbox = cards[1 - c].wire.inbox;
      k::pair_wire_init(*cards[c].q, H, cards[c].wire.inbox).wait();
    }
  Run run{f, cards};
  std::mt19937 rng(77);
  Gates g;
  int checked = 0, steps = 0, p = 0;
  for (int i = 0; p < end_pos; ++i, ++steps) {
    const int rows = i % 9 == 4 ? 2 : (i % 13 == 6 ? 1 : 4), acc = i % 7 == 3 ? 1 : rows;
    std::vector<std::vector<float>> xh;
    const bool chk = i < 3 || (p >= 4088 && p <= 4100) || p + acc >= end_pos;
    const auto xb = rand_x(H, rng, chk ? &xh : nullptr);
    auto x = upload_x(f, cards, xb);
    run.step(rows, {p}, p + rows, x);
    if (chk) {
      check_step(cards, m, rows, p, rope, xh, g, checked < 2);
      ++checked;
    }
    for (size_t c = 0; c < cards.size(); ++c) {
      f.release(x[c]);
      cards[c].mem.erase(std::find(cards[c].mem.begin(), cards[c].mem.end(), static_cast<void *>(x[c])));
    }
    p += acc;
  }
  std::printf("%s %d steps to position %d (%d splits there), %d checked\n", label.c_str(), steps, p, k::gqa_splits(p), checked);
  report(g, label, kv != k::QsaKv::bf16);

  // ---- identities at P = p (4 rows a, b, c, d) ----
  const int P = p;
  const auto xb = rand_x(H, rng);
  auto x4 = upload_x(f, cards, xb);
  std::vector<uint16_t> xcd(8ull * H, 0);
  std::copy(xb.begin() + 2ll * H, xb.begin() + 4ll * H, xcd.begin());
  auto x2 = upload_x(f, cards, xcd);
  auto out_of = [&](int c, int rows) { return rd(*cards[c].q, reinterpret_cast<const uint16_t *>(cards[c].out), static_cast<size_t>(rows) * H); };
  run.step(4, {P}, P + 4, x4);
  const auto base = out_of(0, 4);
  bool same_cards = true;
  for (int c = 1; c < ncards; ++c) same_cards &= out_of(c, 4) == base;
  run.step(4, {P}, P + 4, x4);
  expect(out_of(0, 4) == base, label + " run-to-run bit-identical");
  run.step(4, {P}, cap, x4);
  expect(out_of(0, 4) == base, label + " pos_bound = capacity (" + std::to_string(std::min(k::kGqaSplits, cap / 64)) + " grid splits) == tight bound");
  run.step(1, {P}, P + 1, x4);
  const bool one = std::equal(base.begin(), base.begin() + H, out_of(0, 1).begin());
  run.step(2, {P + 2}, P + 4, x2);
  const auto two = out_of(0, 2);
  expect(one && std::equal(base.begin() + 2ll * H, base.begin() + 4ll * H, two.begin()), label + " 4 rows at P == 1 row at P and 2 rows at P + 2");
  {
    std::vector<int32_t> rec(2 * k::kSegRec, 0);
    rec[k::kSegPos] = P;
    rec[k::kSegRec + k::kSegPos] = P + 2;
    run.step(4, rec, P + 4, x4, k::kSegRec, 2);
    expect(out_of(0, 4) == base, label + " two segments of 2 rows (records) == one 4-row segment");
    // unequal segments (seg_het: k segments of B rows, then one-row segments)
    std::vector<int32_t> rh(3 * k::kSegRec, 0);
    rh[k::kSegPos] = P, rh[k::kSegRec + k::kSegPos] = P + 2, rh[2 * k::kSegRec + k::kSegPos] = P + 3;
    run.step(4, rh, P + 4, x4, k::kSegRec, k::seg_het(2, 1, 3));
    expect(out_of(0, 4) == base, label + " unequal segments 2 + 1 + 1 rows == one 4-row segment");
    std::vector<int32_t> r3(2 * k::kSegRec, 0);
    r3[k::kSegPos] = P, r3[k::kSegRec + k::kSegPos] = P + 3;
    run.step(4, r3, P + 4, x4, k::kSegRec, k::seg_het(3, 1, 2));
    expect(out_of(0, 4) == base, label + " unequal segments 3 + 1 rows == one 4-row segment");
  }
  if (ncards > 1) expect(same_cards, label + " both cards' out bit-identical");
  {  // permuted page table in a pool twice the size, unused pages poisoned (card 0 alone: one card only)
    if (ncards == 1) {
      Card &cd = cards[0];
      const auto &s = cd.s;
      const int NP = cap / 64, PP = 2 * NP;
      std::vector<int32_t> perm(NP);
      for (int i = 0; i < NP; ++i) perm[i] = (i * 37 + 11) % PP;  // 37 odd, PP even: distinct
      std::vector<int> used(PP, 0);
      for (int v : perm) used[v] = 1;
      const size_t pb = k::gqa_page_kv_bytes(s), ps = k::gqa_page_scale_floats(s), pbv = k::gqa_page_v_bytes(s), psv = k::gqa_page_vscale_floats(s);
      const auto kp = rd(*cd.q, cd.c.k, NP * pb), vp = rd(*cd.q, cd.c.v, NP * pbv);
      const auto ksp = rd(*cd.q, cd.c.ks, NP * ps), vsp = rd(*cd.q, cd.c.vs, NP * psv);
      std::vector<uint8_t> k2(PP * pb, 0x7F), v2(PP * pbv, 0x7F);  // poison: large codes / bf16 3.4e38-ish
      std::vector<float> ks2(PP * ps, 1e30f), vs2(PP * psv, std::nanf(""));  // V scales: NaN (int8-v4 folds them late)
      if (s.kv != k::QsaKv::i8v4) vs2.assign(PP * psv, 1e30f);
      for (int i = 0; i < NP; ++i) {
        std::memcpy(&k2[perm[i] * pb], &kp[i * pb], pb);
        std::memcpy(&v2[perm[i] * pbv], &vp[i * pbv], pbv);
        std::memcpy(&ks2[perm[i] * ps], &ksp[i * ps], ps * 4);
        std::memcpy(&vs2[perm[i] * psv], &vsp[i * psv], psv * 4);
      }
      const k::GqaCache keep = cd.c;
      make_cache(f, 0, cd, cap, perm, PP);
      cd.q->memcpy(cd.c.k, k2.data(), k2.size());
      cd.q->memcpy(cd.c.v, v2.data(), v2.size());
      cd.q->memcpy(cd.c.ks, ks2.data(), ks2.size() * 4);
      cd.q->memcpy(cd.c.vs, vs2.data(), vs2.size() * 4).wait();
      run.step(4, {P}, P + 4, x4);
      expect(out_of(0, 4) == base, label + " permuted page table (pool x 2, poisoned) == identity table");
      cd.c = keep;
    }
  }
  {  // every attention group (2 / 4 / 8 rows, 128 / 256 GRF, two 2-row segments) == groups of
     // one row, bit for bit (attn and out of every row), at P and across the tiles-per-split change at 4,096 tokens
     // (rows at 4,094..4,097 and 4,090..4,097 straddle it: two classes); the per-row kernel of within
     // the fp-order bound of it
    const size_t ok = cards[0].s.o_k();
    auto attn_of = [&](int rows) { return rd16(*cards[0].q, cards[0].sc.attn, static_cast<size_t>(rows) * ok); };
    const int G = cards[0].s.grp();
    for (const int at : {P, 4094, 4090}) {
      for (const int rows : {2, 3, 4, 8}) {
        if (rows == 8 && at != 4090) continue;
        run.attn_rows = 1;
        run.attn_grf = 128;
        run.step(rows, {at}, at + rows, x4);
        const auto ref = attn_of(rows);
        const auto ref_out = out_of(0, rows);
        bool all = true;
        std::string bad;
        auto cmp = [&](const std::string &what) {
          const auto got = attn_of(rows);
          const bool same = got == ref && out_of(0, rows) == ref_out;
          if (!same) bad += " " + what;
          all &= same;
        };
        for (const int rm : {1, 2, 4, 8}) {
          if (rm == 8 && G > 6) continue;
          for (const int grf : {128, 256}) {
            if (rm == 1 && grf == 128) continue;
            run.attn_rows = rm;
            run.attn_grf = grf;
            run.step(rows, {at}, at + rows, x4);
            cmp(std::to_string(rm) + "x" + std::to_string(grf));
          }
        }
        for (const int rm : {2, 4}) {  // the L2 prefetch of the next tile changes no bit
          run.attn_rows = rm;
          run.attn_grf = 256;
          run.attn_pf = 1;
          run.step(rows, {at}, at + rows, x4);
          cmp(std::to_string(rm) + "x256 pf");
        }
        run.attn_pf = -1;
        run.attn_rows = 0;
        run.attn_grf = 0;
        run.step(rows, {at}, at + rows, x4);
        cmp("auto");
        if (rows == 4) {  // two segments of 2 rows (auto groups)
          std::vector<int32_t> rec(2 * k::kSegRec, 0);
          rec[k::kSegPos] = at;
          rec[k::kSegRec + k::kSegPos] = at + 2;
          run.step(4, rec, at + 4, x4, k::kSegRec, 2);
          cmp("segs2");
        }
        if (rows == 4 || rows == 8) {  // unequal segments (B-row segments, then one-row ones), auto + forced groups
          const int b = 2, kk = rows == 4 ? 1 : 3, n = kk + (rows - kk * b);
          std::vector<int32_t> rec(static_cast<size_t>(n) * k::kSegRec, 0);
          for (int sg = 0, r0 = 0; sg < n; r0 += sg < kk ? b : 1, ++sg) rec[static_cast<size_t>(sg) * k::kSegRec + k::kSegPos] = at + r0;
          for (const auto &[rm, grf] : std::vector<std::pair<int, int>>{{0, 0}, {1, 128}, {2, 256}, {4, 256}}) {
            run.attn_rows = rm;
            run.attn_grf = grf;
            run.step(rows, rec, at + rows, x4, k::kSegRec, k::seg_het(b, kk, n));
            cmp("het " + (rm ? std::to_string(rm) + "x" + std::to_string(grf) : std::string("auto")));
          }
          run.attn_rows = 0;
          run.attn_grf = 0;
        }
        expect(all, label + " " + std::to_string(rows) + " rows at " + std::to_string(at) + ": attention groups (1 / 2 / 4 / 8 rows, 128 / 256 GRF, auto) bit-identical" +
                        (bad.empty() ? "" : " (differ:" + bad + ")"));
        if (cards[0].s.kv == k::QsaKv::i8v4) continue;  // the per-row kernel takes int8 | bf16
        run.attn_rows = -1;  // the per-row kernel: fp order only
        run.step(rows, {at}, at + rows, x4);
        run.attn_rows = 0;
        const auto old = attn_of(rows);
        BoundCmp bc;
        for (int r = 0; r < rows; ++r) bc.add_rowf(&ref[static_cast<size_t>(r) * ok], &old[static_cast<size_t>(r) * ok], static_cast<int>(ok));
        expect(bc.ok(1.0 / 128, 0.005), label + " " + std::to_string(rows) + " rows at " + std::to_string(at) + ": per-row kernel (rounds 07-13) vs groups: " + bc.text());
      }
    }
    // The decode attention numerics (GqaShape:dec_qk / dec_p_lo, grp 6): every group (1 row at
    // 128 / 256 GRF, 2 / 4 rows, auto, two 2-row segments) bit-identical within a mode, and the mode within its bound of
    // the exact attention
    struct Mode {
      int qk;
      bool lo;
      const char *name;
      double bound;
    };
    const Mode modes[] = {{0, false, "fast (P hi only)", 1.0 / 64}, {1, true, "int8 (int8 q.k)", 1.0 / 16}, {1, false, "int8-fast (int8 q.k, P hi)", 1.0 / 16},
                          {2, false, "fast on int8 (hi + lo q codes, P hi)", 1.0 / 64}};
    if (G == 6)
      for (const Mode &md : modes) {
        if (md.qk && cards[0].s.kv == k::QsaKv::bf16) continue;
        for (const int at : {P, 4090}) {
          const int rows = at == P ? 4 : 8;
          run.attn_rows = 1;
          run.attn_grf = 128;
          run.step(rows, {at}, at + rows, x4);  // exact
          const auto exact = attn_of(rows);
          for (auto &cd : cards) cd.s.dec_qk = md.qk, cd.s.dec_p_lo = md.lo;
          run.step(rows, {at}, at + rows, x4);
          const auto ref = attn_of(rows);
          const auto ref_out = out_of(0, rows);
          bool all = true;
          std::string bad;
          auto cmp = [&](const std::string &what) {
            const bool same = attn_of(rows) == ref && out_of(0, rows) == ref_out;
            if (!same) bad += " " + what;
            all &= same;
          };
          for (const auto &[rm, grf] : std::vector<std::pair<int, int>>{{1, 256}, {2, 256}, {4, 256}, {0, 0}})
            for (const int pf : {0, 1}) {
              run.attn_rows = rm;
              run.attn_grf = grf;
              run.attn_pf = pf;
              run.step(rows, {at}, at + rows, x4);
              cmp((rm ? std::to_string(rm) + "x" + std::to_string(grf) : std::string("auto")) + (pf ? " pf" : ""));
            }
          run.attn_pf = -1;
          {
            std::vector<int32_t> rec(2 * k::kSegRec, 0);
            rec[k::kSegPos] = at;
            rec[k::kSegRec + k::kSegPos] = at + rows / 2;
            run.step(rows, rec, at + rows, x4, k::kSegRec, rows / 2);
            cmp("segs2");
            // unequal segments (2-row segments, then one-row ones)
            const int kk = rows == 4 ? 1 : 3, n = kk + (rows - 2 * kk);
            std::vector<int32_t> rh(static_cast<size_t>(n) * k::kSegRec, 0);
            for (int sg = 0, r0 = 0; sg < n; r0 += sg < kk ? 2 : 1, ++sg) rh[static_cast<size_t>(sg) * k::kSegRec + k::kSegPos] = at + r0;
            run.step(rows, rh, at + rows, x4, k::kSegRec, k::seg_het(2, kk, n));
            cmp("het");
          }
          run.step(rows, {at}, at + rows, x4);
          cmp("run-to-run");
          for (auto &cd : cards) cd.s.dec_qk = 0, cd.s.dec_p_lo = true;
          run.attn_rows = 0;
          run.attn_grf = 0;
          expect(all, label + " " + md.name + ", " + std::to_string(rows) + " rows at " + std::to_string(at) +
                          ": attention groups (1 / 2 / 4 rows, 128 / 256 GRF, auto, 2 segments) bit-identical" + (bad.empty() ? "" : " (differ:" + bad + ")"));
          BoundCmp bc;
          for (int r = 0; r < rows; ++r) bc.add_rowf(&ref[static_cast<size_t>(r) * ok], &exact[static_cast<size_t>(r) * ok], static_cast<int>(ok));
          expect(bc.ok(md.bound, 1.0), label + " " + md.name + ", " + std::to_string(rows) + " rows at " + std::to_string(at) + " vs the exact attention: " + bc.text());
        }
      }
  }
  if (ncards > 1) {
    uint32_t err[2];
    for (int c = 0; c < 2; ++c) cards[c].q->memcpy(&err[c], cards[c].wire.error, 4).wait();
    expect(!err[0] && !err[1], label + " mailbox: no timeout");
  }
  for (auto &cd : cards) free_card(f, cd);
}

// A random 40,000-token history written straight into the cache, then a 4-row step at its end: 10 tiles per
// split (626 tiles: 63 splits, the last one short).
void test_long(xenithium::Fabric &f, const Model &m, k::QsaKv kv) {
  const std::string label = std::string("long context, ") + kv_name(kv) + " KV, 1 card:";
  const int H = m.s.hidden, cap = 40064, P = 40000;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  cards.push_back(make_card(f, 0, m, kv, 1, cap, rope));
  Card &cd = cards[0];
  const auto &s = cd.s;
  {
    const size_t NP = cap / 64;
    std::vector<uint8_t> kvh(NP * k::gqa_page_kv_bytes(s));
    std::vector<float> scl(NP * k::gqa_page_scale_floats(s));
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ui(-127, 127);
    std::uniform_real_distribution<float> us(0.005f, 0.02f);
    if (kv != k::QsaKv::bf16) {
      for (auto &b : kvh) b = static_cast<uint8_t>(static_cast<int8_t>(ui(rng)));
      for (auto &v : scl) v = us(rng);
    } else {
      for (size_t i = 0; i < kvh.size(); i += 2) {
        const uint16_t b = k::bf16_bits(nd(rng));
        std::memcpy(&kvh[i], &b, 2);
      }
    }
    cd.q->memcpy(cd.c.k, kvh.data(), kvh.size());
    if (kv == k::QsaKv::i8v4) {  // any nibbles, bf16 group scales ~0.1-0.3 (codes +-7: values to ~2)
      std::vector<uint8_t> vh(NP * k::gqa_page_v_bytes(s));
      for (auto &b : vh) b = static_cast<uint8_t>(rng());
      std::vector<uint16_t> vsb(NP * k::gqa_page_vscale_floats(s) * 2);
      for (auto &b : vsb) b = k::bf16_bits(us(rng) * 15.f);
      cd.q->memcpy(cd.c.v, vh.data(), vh.size());
      cd.q->memcpy(cd.c.vs, vsb.data(), vsb.size() * 2);
    } else {
      cd.q->memcpy(cd.c.v, kvh.data(), kvh.size());
      if (kv == k::QsaKv::i8) cd.q->memcpy(cd.c.vs, scl.data(), scl.size() * 4);
    }
    if (kv != k::QsaKv::bf16) cd.q->memcpy(cd.c.ks, scl.data(), scl.size() * 4);
    cd.q->wait();
  }
  Run run{f, cards};
  std::mt19937 rng(9);
  std::vector<std::vector<float>> xh;
  const auto xb = rand_x(H, rng, &xh);
  auto x = upload_x(f, cards, xb);
  run.step(4, {P}, P + 4, x);
  Gates g;
  check_step(cards, m, 4, P, rope, xh, g, true);
  std::printf("%s 4 rows at %d (%d splits)\n", label.c_str(), P, k::gqa_splits(P + 3));
  report(g, label, kv != k::QsaKv::bf16);
  const auto base = rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H);
  run.step(4, {P}, cap, x);
  expect(rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H) == base, label + " pos_bound = capacity == tight bound");
  run.attn_rows = 1;  // groups of one row == the auto group
  run.attn_grf = 128;
  run.step(4, {P}, P + 4, x);
  expect(rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H) == base, label + " attention groups of 1 row == auto group (4 rows)");
  free_card(f, cd);
}


// Page selection under int8-fast (int8 q.k on the K codes, P hi only), on test_select's card after
// its exact checks. The page scores read the bf16 q, so the pages are the exact run's; the attention over them stays
// within the mode's bound of the exact selected attention; a budget the context fits in == the dense int8-fast step;
// every row group and register file gives the same bits.
void test_select_int8_fast(Card &cd, Run &run, const std::vector<bf16 *> &x, int P, int cap, int budget, const std::vector<std::vector<int32_t>> &exact_pages,
                           const std::string &label) {
  const auto &s = cd.s;
  const int hkv = s.hkv(), rows = 4;
  const size_t attn_n = static_cast<size_t>(rows) * s.o_k(), out_n = static_cast<size_t>(rows) * s.hidden;
  auto attn = [&] { return rd16(*cd.q, cd.sc.attn, attn_n); };
  auto out = [&] { return rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), out_n); };
  run.attn_rows = 0;
  run.select = budget;
  run.step(rows, {P}, cap, x);
  const auto exact_attn = attn();
  cd.s.dec_qk = 1, cd.s.dec_p_lo = false;
  run.select = 0;
  run.step(rows, {P}, cap, x);
  const auto dense = out();
  run.select = (P + rows - 1) / 64 + 1;  // every page: the selected launch's dense branch
  run.step(rows, {P}, cap, x);
  expect(out() == dense, label + " int8-fast: budget >= the context's pages == the dense int8-fast step, bit for bit");
  run.select = budget;
  run.step(rows, {P}, cap, x);
  const auto fast_attn = attn();
  const auto fast_out = out();
  const auto pages = rd(*cd.q, cd.sc.sel, static_cast<size_t>(hkv) * k::kGqaSelMax);
  bool same_pages = true;
  for (int kh = 0; kh < hkv; ++kh)
    same_pages = same_pages && std::equal(exact_pages[static_cast<size_t>(kh)].begin(), exact_pages[static_cast<size_t>(kh)].end(),
                                          pages.begin() + static_cast<long>(kh) * k::kGqaSelMax);
  expect(same_pages, label + " int8-fast: the pages selected are the exact numerics' (the scores read the bf16 q)");
  BoundCmp vs_exact;
  for (int r = 0; r < rows; ++r)
    vs_exact.add_rowf(&fast_attn[static_cast<size_t>(r) * s.o_k()], &exact_attn[static_cast<size_t>(r) * s.o_k()], static_cast<int>(s.o_k()));
  expect(vs_exact.ok(1.0 / 16, 1.0), label + " int8-fast: " + std::to_string(rows) + " rows at 40,000, 64 pages vs the exact selected attention: " + vs_exact.text());
  bool groups = true;
  for (const auto &[group_rows, grf] : std::vector<std::pair<int, int>>{{1, 128}, {1, 256}, {2, 256}, {4, 256}}) {
    run.attn_rows = group_rows;
    run.attn_grf = grf;
    run.step(rows, {P}, cap, x);
    groups = groups && attn() == fast_attn && out() == fast_out;
  }
  expect(groups, label + " int8-fast: attention groups 1 x 128 / 1 x 256 / 2 x 256 / 4 x 256 (rows x GRF) bit-identical with the auto group");
  cd.s.dec_qk = 0, cd.s.dec_p_lo = true;
  run.attn_rows = 0;
  run.attn_grf = 0;
}

// Page-selected decode attention. Random KV to 40,000 tokens (626 pages), bounds of the whole context
// (gqa_bounds_range) == the host's min / max of each page's keys as stored; a 4-row step at 40,000 with select = 64
// pages: the step leaves its page's bounds alone and gqa_bounds_page writes the whole page's, the selection = the
// sink, the 16 last pages and 47 pages no unselected page outscores (host page scores of the device's q), attn vs the
// softmax over the selected pages' tokens; a budget the context fits in (the selection kernel's dense branch) == the
// dense step, bit for bit.
void test_select(xenithium::Fabric &f, const Model &m, k::QsaKv kv) {
  const std::string label = std::string("page-selected attention, ") + kv_name(kv) + " KV, 1 card:";
  const int H = m.s.hidden, cap = 40064 + 512, P = 40000, B = 64;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  cards.push_back(make_card(f, 0, m, kv, 1, cap, rope));
  Card &cd = cards[0];
  const auto &s = cd.s;
  const int hkv = s.hkv(), hq = s.hq(), G = s.grp(), D = 256;
  const size_t NP = cap / 64;
  cd.c.kmin = static_cast<bf16 *>(cd.dev(f, 0, NP * k::gqa_page_bound_bytes(s) / 2));
  cd.c.kmax = static_cast<bf16 *>(cd.dev(f, 0, NP * k::gqa_page_bound_bytes(s) / 2));
  cd.sc.sel_key = static_cast<uint32_t *>(cd.dev(f, 0, static_cast<size_t>(k::kMaxSegs) * hkv * k::kGqaSelPagesMax * 4));
  cd.sc.sel = static_cast<int32_t *>(cd.dev(f, 0, static_cast<size_t>(k::kMaxSegs) * hkv * k::kGqaSelMax * 4));
  cd.sc.sel_n = static_cast<int32_t *>(cd.dev(f, 0, static_cast<size_t>(k::kMaxSegs) * hkv * 4));
  {
    std::vector<uint8_t> kvh(NP * k::gqa_page_kv_bytes(s));
    std::vector<float> scl(NP * k::gqa_page_scale_floats(s));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ui(-127, 127);
    std::uniform_real_distribution<float> us(0.005f, 0.02f);
    if (kv == k::QsaKv::i8) {
      for (auto &b : kvh) b = static_cast<uint8_t>(static_cast<int8_t>(ui(rng)));
      for (auto &v : scl) v = us(rng);
    } else {
      for (size_t i = 0; i < kvh.size(); i += 2) {
        const uint16_t b = k::bf16_bits(nd(rng));
        std::memcpy(&kvh[i], &b, 2);
      }
    }
    cd.q->memcpy(cd.c.k, kvh.data(), kvh.size());
    cd.q->memcpy(cd.c.v, kvh.data(), kvh.size());
    if (kv == k::QsaKv::i8) {
      cd.q->memcpy(cd.c.ks, scl.data(), scl.size() * 4);
      cd.q->memcpy(cd.c.vs, scl.data(), scl.size() * 4);
    }
    cd.q->wait();
  }
  k::gqa_bounds_range(*cd.q, s, cd.c, 0, P);
  cd.q->wait_and_throw();
  auto host_bounds = [&](int page, int kh, int t_end, std::vector<float> &mn, std::vector<float> &mx) {
    const auto kp = rd(*cd.q, cd.c.k + static_cast<size_t>(page) * k::gqa_page_kv_bytes(s), k::gqa_page_kv_bytes(s));
    std::vector<float> ksp(k::gqa_page_scale_floats(s), 1.f);
    if (kv == k::QsaKv::i8) ksp = rd(*cd.q, cd.c.ks + static_cast<size_t>(page) * k::gqa_page_scale_floats(s), ksp.size());
    mn.assign(D, INFINITY);
    mx.assign(D, -INFINITY);
    for (int t = page * 64; t < t_end; ++t)
      for (int d = 0; d < D; ++d) {
        const float v = kv_at(s, kp, ksp, t % 64, kh, d);
        mn[d] = std::min(mn[d], v), mx[d] = std::max(mx[d], v);
      }
    for (int d = 0; d < D; ++d) mn[d] = bfr(mn[d]), mx[d] = bfr(mx[d]);
  };
  auto dev_bounds = [&](int page, int kh, std::vector<float> &mn, std::vector<float> &mx) {
    const size_t o = (static_cast<size_t>(page) * hkv + kh) * D;
    mn = rd16(*cd.q, cd.c.kmin + o, D);
    mx = rd16(*cd.q, cd.c.kmax + o, D);
  };
  {
    bool ok = true;
    std::vector<float> a, b, c2, d2;
    for (int page : {0, 1, 313, 624}) {
      for (int kh = 0; kh < hkv; ++kh) {
        host_bounds(page, kh, std::min(P, page * 64 + 64), a, b);
        dev_bounds(page, kh, c2, d2);
        ok = ok && a == c2 && b == d2;
      }
    }
    expect(ok, label + " gqa_bounds_range over 40,000 tokens: each page's key min / max == the host's (bf16), pages 0 / 1 / 313 / 624");
  }
  Run run{f, cards};
  std::mt19937 rng(13);
  std::vector<std::vector<float>> xh;
  const auto xb = rand_x(H, rng, &xh);
  auto x = upload_x(f, cards, xb);
  // the stored bounds of the page the steps below write into (kmin then kmax, every kv head)
  auto step_page_bounds = [&] {
    const size_t o = static_cast<size_t>(P / 64) * hkv * D, n = static_cast<size_t>(hkv) * D;
    auto both = rd16(*cd.q, cd.c.kmin + o, n);
    const auto mx = rd16(*cd.q, cd.c.kmax + o, n);
    both.insert(both.end(), mx.begin(), mx.end());
    return both;
  };
  const auto bounds_before_steps = step_page_bounds();
  run.step(4, {P}, cap, x);  // dense
  const auto dense = rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H);
  run.select = 626;  // np = 626 <= budget, pos_bound past it: the selected launch's dense branch
  run.step(4, {P}, cap, x);
  expect(rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H) == dense, label + " budget >= the context's pages (the selected kernels' dense branch) == the dense step, bit for bit");
  run.select = B;
  run.step(4, {P}, cap, x);
  {
    // a decode step leaves its page's bounds alone; gqa_bounds_page writes the whole page's, as
    // the model does when the sequence leaves the page
    const int page = P / 64;
    std::vector<float> host_min, host_max, dev_min, dev_max;
    const bool untouched = step_page_bounds() == bounds_before_steps;
    bool whole = true;
    k::gqa_bounds_page(*cd.q, s, &cd.c, 1, 0, page);
    cd.q->wait_and_throw();
    for (int kh = 0; kh < hkv; ++kh) {
      host_bounds(page, kh, page * 64 + 64, host_min, host_max);
      dev_bounds(page, kh, dev_min, dev_max);
      whole = whole && host_min == dev_min && host_max == dev_max;
    }
    expect(untouched, label + " a decode step writes no bounds: its page's are as they were before the three steps");
    expect(whole, label + " gqa_bounds_page: the whole page's key min / max == the host's (bf16), after a step that wrote 4 of its keys");
  }
  // The last step's selection, `rows` rows at P, against the host: per kv head B pages, ascending, page 0, the last 16,
  // and no page left out that outscores a kept one (host page scores of the device's q and bounds).
  struct Selection {
    bool ok = true;
    double margin = 1e30;  // smallest kept - largest dropped score
    std::vector<std::vector<int32_t>> pages;
  };
  // counts: the pages each kv head keeps (empty = B for all; a count past the context = every page, count 0)
  const int common_pages = B;
  auto selection_of = [&](int rows, const std::vector<int> &counts = {}) {
    Selection sel;
    sel.pages.resize(static_cast<size_t>(hkv));
    const auto n_dev = rd(*cd.q, cd.sc.sel_n, hkv);
    const auto sel_dev = rd(*cd.q, cd.sc.sel, static_cast<size_t>(hkv) * k::kGqaSelMax);
    const auto qd = rd16(*cd.q, cd.sc.q, static_cast<size_t>(rows) * hq * D);
    const int np = (P + rows - 1) / 64 + 1;
    for (int kh = 0; kh < hkv; ++kh) {
      const int B = counts.empty() ? common_pages : counts[static_cast<size_t>(kh)];  // this head's count
      auto &kk = sel.pages[static_cast<size_t>(kh)];
      if (B >= np) {  // the head keeps every page: the dense walk
        sel.ok = sel.ok && n_dev[static_cast<size_t>(kh)] == 0;
        kk.resize(static_cast<size_t>(np));
        std::iota(kk.begin(), kk.end(), 0);
        continue;
      }
      sel.ok = sel.ok && n_dev[static_cast<size_t>(kh)] == B;
      kk.assign(sel_dev.begin() + static_cast<long>(kh) * k::kGqaSelMax, sel_dev.begin() + static_cast<long>(kh) * k::kGqaSelMax + B);
      sel.ok = sel.ok && std::is_sorted(kk.begin(), kk.end()) && std::adjacent_find(kk.begin(), kk.end()) == kk.end() && kk[0] == 0;
      for (int i = 0; i < k::kGqaSelLocal; ++i) sel.ok = sel.ok && kk[static_cast<size_t>(B - k::kGqaSelLocal + i)] == np - k::kGqaSelLocal + i;
      std::vector<double> score(static_cast<size_t>(np), -1e30);
      std::vector<float> mn, mx;
      for (int pg = 0; pg < np; ++pg) {
        dev_bounds(pg, kh, mn, mx);
        for (int r = 0; r < rows; ++r)
          for (int h = 0; h < G; ++h) {
            const float *qv = &qd[(static_cast<size_t>(r) * hq + kh * G + h) * D];
            score[static_cast<size_t>(pg)] = std::max<double>(score[static_cast<size_t>(pg)], k::gqa_page_score_ref(qv, mn.data(), mx.data()));
          }
      }
      std::vector<char> in(static_cast<size_t>(np), 0);
      for (int pg : kk) in[static_cast<size_t>(pg)] = 1;
      double lo_in = 1e30, hi_out = -1e30;
      for (int pg = k::kGqaSelSink; pg < np - k::kGqaSelLocal; ++pg) {
        if (in[static_cast<size_t>(pg)]) lo_in = std::min(lo_in, score[static_cast<size_t>(pg)]);
        else hi_out = std::max(hi_out, score[static_cast<size_t>(pg)]);
      }
      sel.ok = sel.ok && hi_out <= lo_in + 1e-3 * std::abs(lo_in);  // the device sums its bf16 DPAS products in another order
      sel.margin = std::min(sel.margin, lo_in - hi_out);
    }
    return sel;
  };
  const Selection sel4 = selection_of(4);
  const auto &keep = sel4.pages;
  const bool sel_ok = sel4.ok;
  const double margin = sel4.margin;
  const int np = (P + 3) / 64 + 1;
  char buf[96];
  std::snprintf(buf, sizeof buf, " (smallest kept - largest dropped page score %.4f)", margin);
  expect(sel_ok, label + " selection of 64 of " + std::to_string(np) + " pages per kv head: page 0, the last 16, and the 47 top page scores, ascending" + buf);
  Gates g;
  check_step(cards, m, 4, P, rope, xh, g, false, &keep);
  expect(g.attn.ok(1.0 / 64, 0.005), label + " 4 rows at 40,000, 64 pages: attn vs the softmax over the selected pages' tokens: " + g.attn.text());
  run.step(4, {P}, cap, x);
  const auto again = rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H);
  run.step(4, {P}, cap, x);
  expect(rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H) == again, label + " selected step run-to-run bit-identical");
  {  // a page count per (slot, kv head) from the device table (GqaStep:select_slots)
    std::vector<int> counts(static_cast<size_t>(hkv), 32);
    counts[0] = 96;
    if (hkv > 1) counts[static_cast<size_t>(hkv - 1)] = np + 10;  // more than the context: that head keeps every page
    constexpr int kSlots = 16;
    std::vector<int32_t> table(static_cast<size_t>(kSlots) * hkv);
    for (int sl = 0; sl < kSlots; ++sl)
      for (int kh = 0; kh < hkv; ++kh) table[static_cast<size_t>(sl) * hkv + kh] = counts[static_cast<size_t>(kh)];
    for (auto &card : cards) {
      run.select_slots.push_back(sycl::malloc_device<int32_t>(table.size(), *card.q));
      card.q->memcpy(run.select_slots.back(), table.data(), table.size() * sizeof(int32_t)).wait();
    }
    run.step(4, {P}, cap, x);
    const Selection by_head = selection_of(4, counts);
    Gates gh;
    check_step(cards, m, 4, P, rope, xh, gh, false, &by_head.pages);
    expect(by_head.ok && gh.attn.ok(1.0 / 64, 0.005),
           label + " a page count per (slot, kv head) (96, 32.., the last one past the context: every page): each head's pages by the host's scores, attn vs the softmax over them: " +
               gh.attn.text());
    for (size_t c = 0; c < cards.size(); ++c) sycl::free(run.select_slots[c], *cards[c].q);
    run.select_slots.clear();
    run.step(4, {P}, cap, x);
    expect(rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), 4ull * H) == again, label + " without the per-head counts again: == the common count's step, bit for bit");
  }
  run.attn_rows = 1;
  run.step(4, {P}, cap, x);
  bool rows1 = true;
  {
    Gates g1;
    check_step(cards, m, 4, P, rope, xh, g1, false, &keep);
    rows1 = g1.attn.ok(1.0 / 64, 0.005);
  }
  expect(rows1, label + " the same with attention groups of 1 row");
  if (kv == k::QsaKv::i8) test_select_int8_fast(cd, run, x, P, cap, B, keep, label);
  // the score launch's other row blocks: 1 row, 2 rows, and 6 rows = a block of 4 + a block padded with its last row
  bool blocks_ok = true;
  std::string margins;
  for (const int rows : {1, 2, 6}) {
    run.step(rows, {P}, cap, x);
    const Selection sel = selection_of(rows);
    blocks_ok = blocks_ok && sel.ok;
    std::snprintf(buf, sizeof buf, " %.4f", sel.margin);
    margins += buf;
  }
  expect(blocks_ok, label + " selection of a 1- / 2- / 6-row step by the host's page scores (smallest kept - largest dropped:" + margins + ")");
  free_card(f, cd);
}


// gqa_rows (prefill) over two chunks (300 rows at 0, then 237 at 300: short blocks and batches),
// both attention bodies (the block kernel, the decode kernel per 64-row batch), checked like a decode step: q, K / V
// as stored, attn (gated) vs softmax over every token as stored, out vs the o reference; block == run-to-run; the two
// bodies within the attn bound of each other; then a 4-row decode step after the chunks (reads the K / V gqa_rows wrote).
void test_rows(xenithium::Fabric &f, const Model &m, k::QsaKv kv, int ncards) {
  const std::string label = std::string("gqa_rows, ") + kv_name(kv) + " KV, " + std::to_string(ncards) + " card(s):";
  const int H = m.s.hidden, cap = 1024, RM = 304;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  for (int c = 0; c < ncards; ++c) cards.push_back(make_card(f, c, m, kv, ncards, cap, rope));
  std::vector<k::GqaRowsScratch> rs(ncards);
  std::vector<k::PairRowsWire> rw(ncards);
  std::vector<int32_t *> pos4(ncards);
  std::vector<bf16 *> outs(ncards);
  for (int c = 0; c < ncards; ++c) {
    Card &cd = cards[c];
    const auto &sh = cd.s;
    rs[c].y = static_cast<float *>(cd.dev(f, c, static_cast<size_t>(RM) * sh.qkv_n() * 4));
    rs[c].q = static_cast<bf16 *>(cd.dev(f, c, static_cast<size_t>(RM) * sh.hq() * 256 * 2));
    rs[c].part = static_cast<float *>(cd.dev(f, c, k::gqa_part_floats(sh, k::kGqaRowsBatch) * 4));
    rs[c].attn = static_cast<bf16 *>(cd.dev(f, c, static_cast<size_t>(RM) * sh.o_k() * 2));
    rs[c].partial = static_cast<float *>(cd.dev(f, c, static_cast<size_t>(RM) * H * 4));
    pos4[c] = static_cast<int32_t *>(cd.dev(f, c, RM));
    outs[c] = static_cast<bf16 *>(cd.dev(f, c, static_cast<size_t>(RM) * H * 2));
    f.compute(c).memset(rs[c].q, 0, static_cast<size_t>(RM) * sh.hq() * 256 * 2).wait();
    if (sh.head_dim == 256 && sh.gate)  // the oneDNN attention's scratch (the whole cache's tokens)
      k::gqa_sdpa_scratch_bind(sh, RM, cap, cd.dev(f, c, k::gqa_sdpa_scratch_bytes(sh, RM, cap)), rs[c]);
    if (ncards > 1) {
      rw[c].inbox = static_cast<uint32_t *>(f.alloc_kernel_mailbox(k::pair_rows_wire_words(RM, H) * 4));
      cd.mem.push_back(rw[c].inbox);
      rw[c].error = cd.wire.error;
      rw[c].rows_max = RM;
      rw[c].n_max = H;
    }
  }
  if (ncards > 1)
    for (int c = 0; c < ncards; ++c) {
      rw[c].peer_inbox = rw[1 - c].inbox;
      k::pair_rows_wire_init(*cards[c].q, RM, H, rw[c].inbox).wait();
      cards[c].wire.peer_inbox = cards[1 - c].wire.inbox;
      k::pair_wire_init(*cards[c].q, H, cards[c].wire.inbox).wait();
    }
  std::mt19937 rng(91);
  std::normal_distribution<float> nd(0.f, 1.f);
  int parity = 0;
  auto chunk = [&](int M, int pos, const std::vector<uint16_t> &xb, int attn, int rpg = 0, bool plo = true, int qk = 0) {
    std::vector<bf16 *> x;
    for (int c = 0; c < ncards; ++c) {
      std::vector<int32_t> w(RM / 4);
      for (size_t j = 0; j < w.size(); ++j) w[j] = pos + 4 * static_cast<int>(j);
      cards[c].q->memcpy(pos4[c], w.data(), w.size() * 4).wait();
      auto *d = static_cast<bf16 *>(cards[c].dev(f, c, xb.size() * 2));
      cards[c].q->memcpy(d, xb.data(), xb.size() * 2).wait();
      x.push_back(d);
    }
    for (int c = 0; c < ncards; ++c)
      k::gqa_rows(*cards[c].q, cards[c].s, M, x[c], cards[c].w, cards[c].c, cards[c].rope, pos4[c], pos, rs[c], ncards > 1 ? &rw[c] : nullptr, parity,
                  outs[c], attn, rpg, plo, qk);
    parity ^= 1;
    for (auto &cd : cards) cd.q->wait_and_throw();
  };
  auto rand_rows = [&](int M, std::vector<std::vector<float>> &vals) {
    std::vector<uint16_t> xb(static_cast<size_t>(M) * H);
    for (auto &v : xb) v = k::bf16_bits(nd(rng));
    vals.assign(M, std::vector<float>(H));
    for (int r = 0; r < M; ++r)
      for (int j = 0; j < H; ++j) vals[r][j] = k::bf16_value(xb[static_cast<size_t>(r) * H + j]);
    return xb;
  };
  auto view = [&]() {  // the cards with the rows scratch in place of the decode scratch, for check_step
    std::vector<Card> v = cards;
    for (int c = 0; c < ncards; ++c) {
      v[c].sc.y = rs[c].y;
      v[c].sc.q = rs[c].q;
      v[c].sc.attn = rs[c].attn;
      v[c].out = outs[c];
    }
    return v;
  };
  if (kv == k::QsaKv::i8v4) {  // int8-v4 runs the fa body and the batch path (no staged reference): each gated
    // vs the softmax as stored, run-to-run identity, fa vs batch within the fp-order bound; the K / V rows as stored
    std::vector<std::vector<float>> xa, xb2;
    const auto x1 = rand_rows(300, xa);
    chunk(300, 0, x1, k::kGqaAttnFa);
    const auto x2 = rand_rows(237, xb2);
    const size_t ok = cards[0].s.o_k();
    auto gate = [&](const std::string &what, int attn, bool plo, int qk, double wb, bool full) {
      chunk(237, 300, x2, attn, 0, plo, qk);
      Gates g;
      {
        auto v = view();
        check_step(v, m, 237, 300, rope, xb2, g, full);
      }
      if (full) report(g, label + " " + what + ", 237 rows at 300 (after 300 at 0)", true);
      else expect(g.attn.ok(wb, plo && !qk ? 0.005 : 1.0), label + " " + what + ": attn vs softmax as stored: " + g.attn.text());
      const auto o = rd16(*cards[0].q, rs[0].attn, 237ull * ok);
      chunk(237, 300, x2, attn, 0, plo, qk);
      expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o, label + " " + what + ": run-to-run bit-identical (attn)");
      return o;
    };
    const auto o_bt = gate("decode attention per 64-row batch", k::kGqaAttnBatch, true, 0, 1.0 / 64, true);
    const auto o_fa = gate("fa attention", k::kGqaAttnFa, true, 0, 1.0 / 64, true);
    BoundCmp bc;
    for (int r = 0; r < 237; ++r) bc.add_rowf(&o_bt[static_cast<size_t>(r) * ok], &o_fa[static_cast<size_t>(r) * ok], static_cast<int>(ok));
    expect(bc.ok(1.0 / 64, 0.02), label + " batch vs fa attn (fp32 order): " + bc.text());
    gate("fa attention, P hi only", k::kGqaAttnFa, false, 0, 1.0 / 32, false);
    gate("fa attention, int8 q.k (qk 1)", k::kGqaAttnFa, true, 1, 1.0 / 8, false);
    gate("fa attention, int8 q.k + P hi only (int8-fast)", k::kGqaAttnFa, false, 1, 1.0 / 8, false);
    gate("fa attention, qk 2 + P hi only", k::kGqaAttnFa, false, 2, 1.0 / 32, false);
    bool refused = false;
    try {
      chunk(237, 300, x2, k::kGqaAttnStaged);
    } catch (const std::exception &) {
      refused = true;
    }
    expect(refused, label + " the staged body refuses int8-v4");
    if (ncards > 1) {
      uint32_t err[2];
      for (int c = 0; c < 2; ++c) cards[c].q->memcpy(&err[c], cards[c].wire.error, 4).wait();
      expect(!err[0] && !err[1], label + " mailboxes: no timeout");
    }
    for (auto &cd : cards) free_card(f, cd);
    return;
  }
  std::vector<std::vector<float>> xa, xb2;
  const auto x1 = rand_rows(300, xa);
  chunk(300, 0, x1, k::kGqaAttnStaged);
  const auto x2 = rand_rows(237, xb2);
  chunk(237, 300, x2, k::kGqaAttnStaged);
  const bool kv8 = kv == k::QsaKv::i8;
  const size_t ok = cards[0].s.o_k();
  Gates gs;
  {
    auto v = view();
    check_step(v, m, 237, 300, rope, xb2, gs, true);
  }
  report(gs, label + " staged attention (round 13), 237 rows at 300 (after 300 at 0)", kv8);
  const auto stg = rd16(*cards[0].q, rs[0].attn, 237ull * ok);
  const auto stg_out = rd(*cards[0].q, reinterpret_cast<const uint16_t *>(outs[0]), 237ull * H);
  chunk(237, 300, x2, k::kGqaAttnStaged);
  expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == stg && rd(*cards[0].q, reinterpret_cast<const uint16_t *>(outs[0]), 237ull * H) == stg_out,
         label + " staged attention run-to-run bit-identical (attn, out)");
  bool same_cards = true;
  for (int c = 1; c < ncards; ++c) same_cards &= rd(*cards[c].q, reinterpret_cast<const uint16_t *>(outs[c]), 237ull * H) == stg_out;
  if (ncards > 1) expect(same_cards, label + " both cards' out bit-identical");
  // another body over the same chunk: gated alone, then vs staged. plo false (fa, P hi only): P is rounded to bf16 before
  // P.V, so part of the attention outputs move by a few bf16 ulps: a bound-only gate (worst 1/32, any count)
  auto vs_staged = [&](const std::string &what, int attn, int rpg, bool plo = true, int qk = 0, double wb_q = 0) {
    chunk(237, 300, x2, attn, rpg, plo, qk);
    const double wb = wb_q > 0 ? wb_q : plo ? 1.0 / 64 : 1.0 / 32, f1 = plo && !qk ? 0.005 : 1.0, f2 = plo && !qk ? 0.02 : 1.0;
    Gates g;
    {
      auto v = view();
      check_step(v, m, 237, 300, rope, xb2, g, false);
    }
    expect(g.attn.ok(wb, f1), label + " " + what + ": attn vs softmax as stored: " + g.attn.text());
    const auto o = rd16(*cards[0].q, rs[0].attn, 237ull * ok);
    BoundCmp bc;
    for (int r = 0; r < 237; ++r) bc.add_rowf(&stg[static_cast<size_t>(r) * ok], &o[static_cast<size_t>(r) * ok], static_cast<int>(ok));
    expect(bc.ok(wb, f2), label + " staged vs " + what + " attn (fp32 order): " + bc.text());
    return o;
  };
  const int small = m.s.q_heads / m.s.kv_heads == 6 ? 8 : 4;
  const auto o_small = vs_staged("staged, " + std::to_string(small) + " rows per work-group", k::kGqaAttnStaged, small);
  const auto o_block = vs_staged("round-12 block attention", k::kGqaAttnBlock, 0);
  const auto o_batch = vs_staged("decode attention per 64-row batch", k::kGqaAttnBatch, 0);
  // the fa body (one item per (head, 8 rows), all of o) gives the staged body's bits at every row block
  const bool g6 = m.s.q_heads / m.s.kv_heads == 6;
  const std::vector<int> fa_rpg = g6 ? std::vector<int>{0, 40, 32, 16} : std::vector<int>{0, 16, 8};
  // a call of fewer work-groups than a wave splits each over the context (the tail
  // switch): the identity gates below run unsplit; the split is gated by bound right after
  const int tail0 = k::gqa_attn_fa_tail();
  k::gqa_attn_fa_set_tail(1);  // the last-wave tail only (no split at this size)
  for (const int rpg : fa_rpg) {
    chunk(237, 300, x2, k::kGqaAttnFa, rpg);
    const bool same_attn = rd16(*cards[0].q, rs[0].attn, 237ull * ok) == stg;
    const bool same_out = rd(*cards[0].q, reinterpret_cast<const uint16_t *>(outs[0]), 237ull * H) == stg_out;
    expect(same_attn && same_out, label + " fa attention (round 20), " + std::to_string(rpg) + " rows per work-group (0 = default) == staged (attn, out)");
  }
  {
    k::gqa_attn_fa_set_tail(2);
    const auto o_sp = vs_staged("fa attention, short-call context split (round 60)", k::kGqaAttnFa, 0);
    chunk(237, 300, x2, k::kGqaAttnFa);
    expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o_sp, label + " fa attention, short-call context split, run-to-run bit-identical (attn)");
    k::gqa_attn_fa_set_tail(1);
  }
  const auto o_fa_hi = vs_staged("fa attention, P hi only (numerics switch)", k::kGqaAttnFa, 0, false);
  chunk(237, 300, x2, k::kGqaAttnFa, 0, false);
  expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o_fa_hi, label + " fa attention, P hi only, run-to-run bit-identical (attn)");
  if (rs[0].sdpa_kv) {  // oneDNN's f16 attention (its own numerics: f16 q / K / V / P): bound-only, run-to-run identity
    const auto o_sd = vs_staged("oneDNN attention (f16, round 41)", k::kGqaAttnSdpa, 0, false, 0, 1.0 / 32);
    chunk(237, 300, x2, k::kGqaAttnSdpa);
    expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o_sd, label + " oneDNN attention, run-to-run bit-identical (attn)");
    // a 16-row call (a prompt tail): oneDNN < 3.14.2 had an SDPA correctness bug at query length 16 on Intel GPUs
    std::vector<std::vector<float>> x16v;
    const auto x16 = rand_rows(16, x16v);
    chunk(16, 300, x16, k::kGqaAttnSdpa);
    Gates g16;
    {
      auto v = view();
      check_step(v, m, 16, 300, rope, x16v, g16, false);
    }
    expect(g16.attn.ok(1.0 / 32, 1.0), label + " oneDNN attention, 16 rows at 300: attn vs softmax as stored: " + g16.attn.text());
    chunk(237, 300, x2, k::kGqaAttnStaged);  // the cache back to the 237-row chunk for the checks below
  }
  if (kv8) {  // int8 q.k (numerics switches): bound-only vs the softmax as stored and vs staged, run-to-run identity
    const auto o_q1 = vs_staged("fa attention, int8 q.k (qk 1)", k::kGqaAttnFa, 0, true, 1, 1.0 / 8);
    chunk(237, 300, x2, k::kGqaAttnFa, 0, true, 1);
    expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o_q1, label + " fa attention, int8 q.k, run-to-run bit-identical (attn)");
    vs_staged("fa attention, int8 q.k + P hi only", k::kGqaAttnFa, 0, false, 1, 1.0 / 8);
    const auto o_q2 = vs_staged("fa attention, int8 hi + lo q.k + P hi only (qk 2)", k::kGqaAttnFa, 0, false, 2);
    chunk(237, 300, x2, k::kGqaAttnFa, 0, false, 2);
    expect(rd16(*cards[0].q, rs[0].attn, 237ull * ok) == o_q2, label + " fa attention, qk 2, run-to-run bit-identical (attn)");
    // qk 2: q to ~3e-5 of its amax (hi + lo codes): against P hi only on bf16 q.k, part of the outputs move by 1-2 bf16
    // ulps (P's bf16 rounding flips under the tiny q.k change)
    BoundCmp bc;
    size_t same = 0;
    for (int r = 0; r < 237; ++r) bc.add_rowf(&o_fa_hi[static_cast<size_t>(r) * ok], &o_q2[static_cast<size_t>(r) * ok], static_cast<int>(ok));
    for (size_t e = 0; e < o_q2.size(); ++e) same += o_q2[e] == o_fa_hi[e];
    expect(bc.ok(1.0 / 32, 0.25), label + " qk 2 vs bf16 q.k (both P hi only): " + bc.text() + ", " + std::to_string(same) + " of " + std::to_string(o_q2.size()) + " equal");
    bool threw = false;
    try {
      chunk(237, 300, x2, k::kGqaAttnFa, 0, true, 2);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    expect(threw, label + " qk 2 with P hi + lo is refused");
  } else {
    bool threw = false;
    try {
      chunk(237, 300, x2, k::kGqaAttnFa, 0, true, 1);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    expect(threw, label + " int8 q.k on the bf16 KV is refused");
  }
  {  // a decode step after the chunks reads what gqa_rows wrote
    Run run{f, cards};
    run.parity = 0;
    std::vector<std::vector<float>> xh;
    const auto xd = rand_x(H, rng, &xh);
    auto x = upload_x(f, cards, xd);
    run.step(4, {537}, 541, x);
    Gates g;
    check_step(cards, m, 4, 537, rope, xh, g, false);
    expect(g.attn.ok(1.0 / 64, 0.005) && g.out.ok(1.0 / 128, 0.01), label + " a 4-row decode step after the chunks: attn " + g.attn.text() + "; out " + g.out.text());
  }
  if (ncards == 1) {  // (shared KV pool): every prefill body through a permuted page table in a pool
                      // twice the size, unused pages poisoned == the identity table, bit for bit (attn; out for staged)
    Card &cd = cards[0];
    const int NP = cap / 64, PP = 2 * NP;
    std::vector<int32_t> perm(NP);
    for (int i = 0; i < NP; ++i) perm[i] = (i * 37 + 11) % PP;
    const k::GqaCache keep = cd.c;
    make_cache(f, 0, cd, cap, perm, PP);
    const size_t pb = k::gqa_page_kv_bytes(cd.s), ps = k::gqa_page_scale_floats(cd.s);
    std::vector<uint8_t> poison(PP * pb, 0x7F);
    std::vector<float> spoison(PP * ps, 1e30f);
    cd.q->memcpy(cd.c.k, poison.data(), poison.size());
    cd.q->memcpy(cd.c.v, poison.data(), poison.size());
    cd.q->memcpy(cd.c.ks, spoison.data(), spoison.size() * 4);
    cd.q->memcpy(cd.c.vs, spoison.data(), spoison.size() * 4).wait();
    auto attn_now = [&] { return rd16(*cd.q, rs[0].attn, 237ull * ok); };
    chunk(300, 0, x1, k::kGqaAttnStaged);
    chunk(237, 300, x2, k::kGqaAttnStaged);
    expect(attn_now() == stg && rd(*cd.q, reinterpret_cast<const uint16_t *>(outs[0]), 237ull * H) == stg_out,
           label + " staged attention, permuted page table (pool x 2, poisoned) == identity (attn, out)");
    bool same = true;
    chunk(237, 300, x2, k::kGqaAttnStaged, small);
    same &= attn_now() == o_small;
    chunk(237, 300, x2, k::kGqaAttnBlock);
    same &= attn_now() == o_block;
    chunk(237, 300, x2, k::kGqaAttnBatch);
    same &= attn_now() == o_batch;
    chunk(237, 300, x2, k::kGqaAttnFa);
    same &= attn_now() == stg;
    chunk(237, 300, x2, k::kGqaAttnFa, 0, false);
    same &= attn_now() == o_fa_hi;
    expect(same, label + " staged (" + std::to_string(small) + " rows), block, batch and fa bodies, permuted page table == identity (attn)");
    cd.c = keep;
  }
  k::gqa_attn_fa_set_tail(tail0);
  if (ncards > 1) {
    uint32_t err[2];
    for (int c = 0; c < 2; ++c) cards[c].q->memcpy(&err[c], cards[c].wire.error, 4).wait();
    expect(!err[0] && !err[1], label + " mailboxes: no timeout");
  }
  for (auto &cd : cards) free_card(f, cd);
}

// gqa_rows with page selection (the fa body's SEL instances, gqa_select_rows). A context of random
// keys and values, then calls that cross the page budget and calls past it:
// - a row block still under the budget keeps every page and equals the dense call's rows bit for bit;
// - a block past it selects, per kv head: the budget's pages, ascending, page 0, the last 16 up to its last row, and
// no dropped page that outscores a kept one (the host's page scores of the block's rows' q heads);
// - its rows' attention == the softmax over the selected pages' tokens (the mode's bound), run to run identical;
// - the same for blocks of 16 rows, under the short-call context split, and in the other numerics page selection runs.
// cap: the cache's tokens; the far call sits 1,096 tokens under it (cap 71,680: 1,103-1,105 pages of context, more
// pages than one select work-group has items).
void test_rows_select(xenithium::Fabric &f, const Model &m, k::QsaKv kv, int cap = 4096) {
  const std::string label = std::string("page-selected gqa_rows, ") + kv_name(kv) + " KV, " + std::to_string(cap) + "-token cache, 1 card:";
  const int H = m.s.hidden, RM = 304, B = 32, D = 256;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  cards.push_back(make_card(f, 0, m, kv, 1, cap, rope));
  Card &cd = cards[0];
  const auto &s = cd.s;
  const int hkv = s.hkv(), hq = s.hq(), G = s.grp();
  const size_t NP = cap / 64, ok = s.o_k();
  cd.c.kmin = static_cast<bf16 *>(cd.dev(f, 0, NP * k::gqa_page_bound_bytes(s) / 2));
  cd.c.kmax = static_cast<bf16 *>(cd.dev(f, 0, NP * k::gqa_page_bound_bytes(s) / 2));
  k::GqaRowsScratch rs;
  rs.y = static_cast<float *>(cd.dev(f, 0, static_cast<size_t>(RM) * s.qkv_n() * 4));
  rs.q = static_cast<bf16 *>(cd.dev(f, 0, static_cast<size_t>(RM) * hq * D * 2));
  rs.part = static_cast<float *>(cd.dev(f, 0, k::gqa_part_floats(s, k::kGqaRowsBatch) * 4));
  rs.attn = static_cast<bf16 *>(cd.dev(f, 0, static_cast<size_t>(RM) * ok * 2));
  rs.partial = static_cast<float *>(cd.dev(f, 0, static_cast<size_t>(RM) * H * 4));
  cd.q->memset(rs.q, 0, static_cast<size_t>(RM) * hq * D * 2).wait();
  k::gqa_page_sets_bind(s, RM, 16, cap, cd.dev(f, 0, k::gqa_page_sets_bytes(s, RM, 16, cap)), rs.sel);
  auto *pos4 = static_cast<int32_t *>(cd.dev(f, 0, RM));
  auto *out = static_cast<bf16 *>(cd.dev(f, 0, static_cast<size_t>(RM) * H * 2));
  std::mt19937 rng(193);
  std::normal_distribution<float> nd(0.f, 1.f);
  {  // the context: random keys and values in every page, the bounds of the tokens below the calls
    std::vector<uint8_t> pool(NP * k::gqa_page_kv_bytes(s));
    std::vector<float> scales(NP * k::gqa_page_scale_floats(s));
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.005f, 0.02f);
    if (kv == k::QsaKv::i8) {
      for (auto &b : pool) b = static_cast<uint8_t>(static_cast<int8_t>(code(rng)));
      for (auto &v : scales) v = scale(rng);
      cd.q->memcpy(cd.c.ks, scales.data(), scales.size() * 4);
      cd.q->memcpy(cd.c.vs, scales.data(), scales.size() * 4);
    } else {
      for (size_t i = 0; i < pool.size(); i += 2) {
        const uint16_t b = k::bf16_bits(nd(rng));
        std::memcpy(&pool[i], &b, 2);
      }
    }
    cd.q->memcpy(cd.c.k, pool.data(), pool.size());
    cd.q->memcpy(cd.c.v, pool.data(), pool.size()).wait();
    k::gqa_bounds_range(*cd.q, s, cd.c, 0, cap);
    cd.q->wait_and_throw();
  }
  struct Call {
    int rows, pos;
    std::vector<uint16_t> x;  // bf16 bits [rows][H]
    std::vector<std::vector<float>> xh;  // the same as values
  };
  auto make_call = [&](int rows, int pos) {
    Call call{rows, pos, std::vector<uint16_t>(static_cast<size_t>(rows) * H), {}};
    for (auto &v : call.x) v = k::bf16_bits(nd(rng));
    call.xh.assign(static_cast<size_t>(rows), std::vector<float>(static_cast<size_t>(H)));
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < H; ++j) call.xh[static_cast<size_t>(r)][static_cast<size_t>(j)] = k::bf16_value(call.x[static_cast<size_t>(r) * H + j]);
    return call;
  };
  struct Numerics {
    bool p_lo = true;
    int qk = 0;
  };
  auto run = [&](const Call &call, int select, int rows_per_group = 0, Numerics nm = {}, int body = k::kGqaAttnFa, bool dense_last_block = false) {
    std::vector<int32_t> words(RM / 4);
    for (size_t j = 0; j < words.size(); ++j) words[j] = call.pos + 4 * static_cast<int>(j);
    cd.q->memcpy(pos4, words.data(), words.size() * 4).wait();
    auto *x = static_cast<bf16 *>(cd.dev(f, 0, call.x.size() * 2));
    cd.q->memcpy(x, call.x.data(), call.x.size() * 2).wait();
    k::gqa_rows(*cd.q, s, call.rows, x, cd.w, cd.c, cd.rope, pos4, call.pos, rs, nullptr, 0, out, body, rows_per_group, nm.p_lo, nm.qk,
                k::GqaRowsSelect{.pages = select, .dense_last_block = dense_last_block});
    cd.q->wait_and_throw();
    return rd16(*cd.q, rs.attn, static_cast<size_t>(call.rows) * ok);
  };
  // The page sets of the last call against the host: per (row block, kv head) the count, the order, the fixed pages, and
  // no page left out that outscores a kept one.
  struct Sets {
    bool ok = true;
    double margin = 1e30;  // smallest kept - largest dropped page score
    int selected_blocks = 0;
    std::vector<PageSets> block;  // [row block][kv head]; empty lists = every page
    std::vector<const PageSets *> by_row;
  };
  auto read_sets = [&](const Call &call, int block_rows) {
    Sets sets;
    const int blocks = (call.rows + block_rows - 1) / block_rows;
    const auto counts = rd(*cd.q, rs.sel.counts, static_cast<size_t>(blocks) * hkv);
    const auto pages = rd(*cd.q, rs.sel.pages, static_cast<size_t>(blocks) * hkv * k::kGqaSelMax);
    const auto qd = rd16(*cd.q, rs.q, static_cast<size_t>(call.rows) * hq * D);
    const auto kmin = rd16(*cd.q, cd.c.kmin, NP * hkv * D), kmax = rd16(*cd.q, cd.c.kmax, NP * hkv * D);
    sets.block.assign(static_cast<size_t>(blocks), PageSets(static_cast<size_t>(hkv)));
    for (int b = 0; b < blocks; ++b) {
      const int row0 = b * block_rows, row1 = std::min(call.rows, row0 + block_rows) - 1;
      const int np = (call.pos + row1) / 64 + 1;
      for (int kh = 0; kh < hkv; ++kh) {
        const size_t unit = static_cast<size_t>(b) * hkv + kh;
        if (np <= B) {
          sets.ok = sets.ok && counts[unit] == 0;
          continue;
        }
        sets.selected_blocks += kh == 0;
        auto &kept = sets.block[static_cast<size_t>(b)][static_cast<size_t>(kh)];
        kept.assign(pages.begin() + static_cast<long>(unit) * k::kGqaSelMax, pages.begin() + static_cast<long>(unit) * k::kGqaSelMax + B);
        sets.ok = sets.ok && counts[unit] == B && std::is_sorted(kept.begin(), kept.end()) && std::adjacent_find(kept.begin(), kept.end()) == kept.end() && kept[0] == 0;
        for (int i = 0; i < k::kGqaSelLocal; ++i) sets.ok = sets.ok && kept[static_cast<size_t>(B - k::kGqaSelLocal + i)] == np - k::kGqaSelLocal + i;
        std::vector<char> in(static_cast<size_t>(np), 0);
        for (const int pg : kept)
          if (pg >= 0 && pg < np) in[static_cast<size_t>(pg)] = 1;
        double lo_in = 1e30, hi_out = -1e30;
        for (int pg = k::kGqaSelSink; pg < np - k::kGqaSelLocal; ++pg) {
          double score = -1e30;
          const size_t bounds = (static_cast<size_t>(pg) * hkv + kh) * D;
          for (int r = row0; r <= row1; ++r)
            for (int h = 0; h < G; ++h)
              score = std::max<double>(score, k::gqa_page_score_ref(&qd[(static_cast<size_t>(r) * hq + kh * G + h) * D], &kmin[bounds], &kmax[bounds]));
          if (in[static_cast<size_t>(pg)]) lo_in = std::min(lo_in, score);
          else hi_out = std::max(hi_out, score);
        }
        sets.ok = sets.ok && hi_out <= lo_in + 1e-3 * std::abs(lo_in);  // the device sums its bf16 DPAS products in another order
        sets.margin = std::min(sets.margin, lo_in - hi_out);
      }
    }
    return sets;
  };
  auto rows_of = [&](Sets &sets, const Call &call, int block_rows) {  // by_row after the vector stopped moving
    sets.by_row.clear();
    for (int r = 0; r < call.rows; ++r) sets.by_row.push_back(&sets.block[static_cast<size_t>(r / block_rows)]);
  };
  auto attn_vs_softmax = [&](const Call &call, const Sets &sets) {
    std::vector<Card> view = cards;
    view[0].sc.y = rs.y, view[0].sc.q = rs.q, view[0].sc.attn = rs.attn, view[0].out = out;
    Gates g;
    check_step(view, m, call.rows, call.pos, rope, call.xh, g, false, nullptr, &sets.by_row);
    return g.attn;
  };
  auto margin_text = [](const Sets &sets) {
    char buf[96];
    std::snprintf(buf, sizeof buf, " (smallest kept - largest dropped page score %.4f)", sets.margin);
    return std::string(buf);
  };
  const int tail0 = k::gqa_attn_fa_tail();
  k::gqa_attn_fa_set_tail(1);  // no short-call context split: the identity gates compare whole work-groups
  {  // a call that crosses the budget: 150 rows from 1,948: the blocks of rows 0-79 end in page 31, the rest in page 32
    const Call cross = make_call(150, B * 64 - 100);
    const auto dense = run(cross, 0);
    const auto selected = run(cross, B);
    Sets sets = read_sets(cross, 40);
    rows_of(sets, cross, 40);
    const bool head_same = std::equal(dense.begin(), dense.begin() + 80 * static_cast<long>(ok), selected.begin());
    expect(head_same, label + " a call crossing the budget: the row blocks under it (rows 0-79) == the dense call, bit for bit");
    expect(sets.ok && sets.selected_blocks == 2, label + " the two blocks past it: 32 of 33 pages per kv head (page 0, the last 16, the 15 top scores), ascending" + margin_text(sets));
    const BoundCmp cmp = attn_vs_softmax(cross, sets);
    expect(cmp.ok(1.0 / 64, 0.005), label + " the crossing call's attn vs the softmax over each block's pages: " + cmp.text());
  }
  const Call far = make_call(100, cap - 1096);  // cap 4,096: 47-49 pages of context; every block selects 32
  const std::string far_at = "100 rows at " + std::to_string(far.pos) + " (" + std::to_string(far.pos / 64 + 1) + "+ pages)";
  const auto far_dense = run(far, 0);
  const auto far_selected = run(far, B);
  Sets far_sets = read_sets(far, 40);
  rows_of(far_sets, far, 40);
  {
    expect(far_sets.ok && far_sets.selected_blocks == 3, label + " " + far_at + ": every block's 32 pages by the host's page scores" + margin_text(far_sets));
    const BoundCmp cmp = attn_vs_softmax(far, far_sets);
    expect(cmp.ok(1.0 / 64, 0.005), label + " their attn vs the softmax over each block's pages: " + cmp.text());
    expect(far_selected != far_dense, label + " the selected attention differs from the dense one (the selection engaged)");
    expect(run(far, B) == far_selected, label + " run-to-run bit-identical (attn)");
  }
  {  // the call's last block dense (the call that ends a prompt): its rows == the dense call's, the others == the selected call's
    const auto mixed = run(far, B, 0, {}, k::kGqaAttnFa, true);
    const long cut = 80 * static_cast<long>(ok);  // rows 80-99 are the last block
    expect(std::equal(mixed.begin(), mixed.begin() + cut, far_selected.begin()) && std::equal(mixed.begin() + cut, mixed.end(), far_dense.begin() + cut),
           label + " dense last block: its rows == the dense call, the blocks before it == the selected call, bit for bit");
  }
  {  // blocks of 16 rows
    run(far, B, 16);
    Sets sets = read_sets(far, 16);
    rows_of(sets, far, 16);
    const BoundCmp cmp = attn_vs_softmax(far, sets);
    expect(sets.ok && sets.selected_blocks == 7 && cmp.ok(1.0 / 64, 0.005),
           label + " 16 rows per work-group: 7 blocks' page sets by the host's scores" + margin_text(sets) + ", attn vs the softmax over them: " + cmp.text());
  }
  {  // the short-call context split: a block's units share its selected pages
    k::gqa_attn_fa_set_tail(2);
    const auto split = run(far, B);
    const BoundCmp cmp = attn_vs_softmax(far, far_sets);
    expect(cmp.ok(1.0 / 64, 0.005) && run(far, B) == split, label + " short-call context split over the selected pages: attn vs the softmax " + cmp.text() + ", run-to-run bit-identical");
    k::gqa_attn_fa_set_tail(1);
  }
  {  // the other numerics page selection runs: int8 q.k + P hi only on the int8 KV, P hi only on the bf16 KV
    const bool kv8 = kv == k::QsaKv::i8;
    const Numerics fast = kv8 ? Numerics{false, 1} : Numerics{false, 0};
    run(far, B, 0, fast);
    Sets sets = read_sets(far, 40);
    rows_of(sets, far, 40);
    const BoundCmp cmp = attn_vs_softmax(far, sets);
    expect(sets.block == far_sets.block && cmp.ok(kv8 ? 1.0 / 8 : 1.0 / 32, 1.0),
           label + (kv8 ? " int8 q.k + P hi only (int8-fast)" : " P hi only (fast)") + ": the exact numerics' page sets, attn vs the softmax over them: " + cmp.text());
  }
  auto refused = [&](int body, Numerics nm) {
    try {
      run(far, B, 0, nm, body);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  expect(refused(k::kGqaAttnStaged, {}), label + " the staged body refuses page selection");
  expect(refused(k::kGqaAttnFa, kv == k::QsaKv::i8 ? Numerics{false, 2} : Numerics{true, 1}), label + " a numerics mode without a selected instance is refused");
  k::gqa_attn_fa_set_tail(tail0);
  free_card(f, cd);
}
}  // namespace

// A step whose segment is a draft tree (records' kSegTree) at positions P that
// put the step's rows inside one tile, across a page boundary and across the 4,096-token split change. Gate: every row's
// attn and out == row depth(j) of a chain step over its ancestors on the same cache, bit for bit, for every attention
// group (1 / 2 / 4 / 8 rows, 128 / 256 GRF); then gqa_kv_move of an accepted path == the chain's K / V bytes.
// The same gate runs under a decode attention numerics mode (dec_qk / dec_p_lo: the tree step and
// its chain references all run in the mode; its groups are 1 row at 128 / 256 GRF and 2 / 4 rows at 256).
void test_tree(xenithium::Fabric &f, const Model &m, k::QsaKv kv, int qk = 0, bool lo = true, const char *mode = "") {
  const std::string label = std::string("tree steps, ") + kv_name(kv) + " KV" + (*mode ? std::string(", ") + mode : std::string()) + ":";
  const int H = m.s.hidden, cap = 4608;
  const auto rope = k::qsa_rope_table(cap, 64, 1e7, k::QsaYarn{});
  std::vector<Card> cards;
  cards.push_back(make_card(f, 0, m, kv, 1, cap, rope));
  Card &cd = cards[0];
  cd.s.dec_qk = qk, cd.s.dec_p_lo = lo;
  const bool exact = qk == 0 && lo;
  const auto &s = cd.s;
  Run run{f, cards};
  std::mt19937 rng(234);
  const size_t NP = cap / 64, pb = k::gqa_page_kv_bytes(s), pbv = k::gqa_page_v_bytes(s), ps = k::gqa_page_scale_floats(s), psv = k::gqa_page_vscale_floats(s);
  struct Snap {
    std::vector<uint8_t> k, v;
    std::vector<float> ks, vs;
  };
  auto snap = [&]() {
    Snap sn{rd(*cd.q, cd.c.k, NP * pb), rd(*cd.q, cd.c.v, NP * pbv), rd(*cd.q, cd.c.ks, NP * ps), rd(*cd.q, cd.c.vs, NP * psv)};
    return sn;
  };
  auto restore = [&](const Snap &sn) {
    cd.q->memcpy(cd.c.k, sn.k.data(), sn.k.size());
    cd.q->memcpy(cd.c.v, sn.v.data(), sn.v.size());
    cd.q->memcpy(cd.c.ks, sn.ks.data(), sn.ks.size() * 4);
    cd.q->memcpy(cd.c.vs, sn.vs.data(), sn.vs.size() * 4).wait();
  };
  const size_t ok = s.o_k();
  auto attn_of = [&](int rows) { return rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.sc.attn), static_cast<size_t>(rows) * ok); };
  auto out_of = [&](int rows) { return rd(*cd.q, reinterpret_cast<const uint16_t *>(cd.out), static_cast<size_t>(rows) * H); };
  // (the last three: 16-row trees, depth up to 15, 4-bit parents in two record words)
  const std::vector<std::vector<int>> trees = {{-1, 0, 1, 2, 0, 4, 1, 6}, {-1, 0, 0, 2, 3, 3, 5, 5}, {-1, 0, 0, 0, 0, 0, 0, 0},
                                               {-1, 0, 1, 2, 3, 4, 5, 6}, {-1, 0, 1, 1, 0, 4, 4, 1}, {-1, 0, 0, 1},
                                               {-1, 0, 0, 2, 2, 4, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 10, 11, 8, 13, 14},
                                               {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}};
  struct G {
    int rows, grf;
  };
  const std::vector<G> groups = exact ? std::vector<G>{{0, 0}, {1, 128}, {2, 256}, {4, 256}, {8, 256}, {2, 128}} : std::vector<G>{{0, 0}, {1, 128}, {1, 256}, {2, 256}, {4, 256}};
  int p = 0, bad = 0, checked = 0, bad_mv = 0, moved = 0;
  for (const int P : {57, 1020, 4090}) {
    while (p < P) {  // the context: 4-row steps of random rows
      const int n = std::min(4, P - p);
      auto x = upload_x(f, cards, rand_x(H, rng));
      run.attn_rows = 0, run.attn_grf = 0;
      run.step(n, {p}, p + n, x);
      f.release(x[0]);
      cd.mem.erase(std::find(cd.mem.begin(), cd.mem.end(), static_cast<void *>(x[0])));
      p += n;
    }
    const Snap base = snap();
    for (size_t ti = 0; ti < trees.size(); ++ti) {
      const auto &par = trees[ti];
      const int R = static_cast<int>(par.size());
      uint64_t tw = k::kTreeFlag;
      for (int j = 1; j < R; ++j) tw |= static_cast<uint64_t>(par[static_cast<size_t>(j)]) << (4 * j);
      auto xb = rand_x(H, rng);
      if (R > 8) {  // 16 rows of x
        const auto x2 = rand_x(H, rng);
        xb.insert(xb.end(), x2.begin(), x2.end());
      }
      auto x = upload_x(f, cards, xb);
      std::vector<int32_t> rec(k::kSegRec, 0);
      rec[k::kSegPos] = P;
      k::rec_word_set(rec.data(), k::kSegTree, k::kSegTreeHi, tw);
      // the chain references: row j's ancestors' x rows as a chain at P
      std::vector<std::vector<uint16_t>> ref_attn(static_cast<size_t>(R)), ref_out(static_cast<size_t>(R));
      std::vector<int> depth(static_cast<size_t>(R));
      for (int j = 0; j < R; ++j) {
        std::vector<int> path;
        for (int a = j; a >= 0; a = par[static_cast<size_t>(a)]) path.insert(path.begin(), a);
        depth[static_cast<size_t>(j)] = static_cast<int>(path.size()) - 1;
        std::vector<uint16_t> xc(16ull * H, 0);
        for (size_t i = 0; i < path.size(); ++i) std::copy(xb.begin() + static_cast<ptrdiff_t>(path[i]) * H, xb.begin() + static_cast<ptrdiff_t>(path[i] + 1) * H, xc.begin() + static_cast<ptrdiff_t>(i) * H);
        auto xcd = upload_x(f, cards, xc);
        restore(base);
        run.attn_rows = 1, run.attn_grf = 128, run.tree = false;
        const int n = static_cast<int>(path.size());
        run.step(n, {P}, P + n, xcd);
        const auto a = attn_of(n), o = out_of(n);
        ref_attn[static_cast<size_t>(j)].assign(a.end() - static_cast<ptrdiff_t>(ok), a.end());
        ref_out[static_cast<size_t>(j)].assign(o.end() - H, o.end());
        f.release(xcd[0]);
        cd.mem.erase(std::find(cd.mem.begin(), cd.mem.end(), static_cast<void *>(xcd[0])));
      }
      for (const auto &gg : groups) {
        restore(base);
        run.attn_rows = gg.rows, run.attn_grf = gg.grf, run.tree = true;
        run.step(R, rec, P + R, x, k::kSegRec, R);
        run.tree = false;
        const auto a = attn_of(R), o = out_of(R);
        for (int j = 0; j < R; ++j) {
          const bool same = std::equal(ref_attn[static_cast<size_t>(j)].begin(), ref_attn[static_cast<size_t>(j)].end(), a.begin() + static_cast<ptrdiff_t>(j) * static_cast<ptrdiff_t>(ok)) &&
                            std::equal(ref_out[static_cast<size_t>(j)].begin(), ref_out[static_cast<size_t>(j)].end(), o.begin() + static_cast<ptrdiff_t>(j) * H);
          if (!same && bad < 8) std::printf("  P %d tree %zu group %d / %d: row %d (depth %d) differs from its chain\n", P, ti, gg.rows, gg.grf, j, depth[static_cast<size_t>(j)]);
          bad += !same;
          ++checked;
        }
      }
      // the accepted path to the deepest last row: gqa_kv_move vs the chain's K / V bytes at P.. P + depth
      {
        const int node = R - 1;
        std::vector<int> path;
        for (int a = node; a >= 0; a = par[static_cast<size_t>(a)]) path.insert(path.begin(), a);
        k::GqaMove mv;
        for (size_t i = 0; i < path.size(); ++i) mv.from[mv.n] = path[i], mv.to[mv.n] = static_cast<int>(i), ++mv.n;
        k::gqa_kv_move(*cd.q, s, &cd.c, 1, 0, P, mv).wait();
        const Snap tree = snap();
        std::vector<uint16_t> xc(16ull * H, 0);
        for (size_t i = 0; i < path.size(); ++i) std::copy(xb.begin() + static_cast<ptrdiff_t>(path[i]) * H, xb.begin() + static_cast<ptrdiff_t>(path[i] + 1) * H, xc.begin() + static_cast<ptrdiff_t>(i) * H);
        auto xcd = upload_x(f, cards, xc);
        restore(base);
        run.attn_rows = 0, run.attn_grf = 0;
        run.step(static_cast<int>(path.size()), {P}, P + static_cast<int>(path.size()), xcd);
        const Snap chain = snap();
        f.release(xcd[0]);
        cd.mem.erase(std::find(cd.mem.begin(), cd.mem.end(), static_cast<void *>(xcd[0])));
        for (size_t i = 0; i < path.size(); ++i) {
          const int t = P + static_cast<int>(i);
          for (int h = 0; h < s.hkv(); ++h) {
            const size_t sl = (static_cast<size_t>(t / 64) * s.hkv() + h) * 64 + t % 64;
            const size_t kb = static_cast<size_t>(s.head_dim) * s.kv_elem();
            bool same = std::equal(tree.k.begin() + static_cast<ptrdiff_t>(sl * kb), tree.k.begin() + static_cast<ptrdiff_t>((sl + 1) * kb), chain.k.begin() + static_cast<ptrdiff_t>(sl * kb)) &&
                        std::equal(tree.v.begin() + static_cast<ptrdiff_t>(sl * kb), tree.v.begin() + static_cast<ptrdiff_t>((sl + 1) * kb), chain.v.begin() + static_cast<ptrdiff_t>(sl * kb));
            if (kv != k::QsaKv::bf16) same = same && tree.ks[sl] == chain.ks[sl] && tree.vs[sl] == chain.vs[sl];
            bad_mv += !same;
            ++moved;
          }
        }
      }
      f.release(x[0]);
      cd.mem.erase(std::find(cd.mem.begin(), cd.mem.end(), static_cast<void *>(x[0])));
      restore(base);
    }
  }
  run.attn_rows = 0, run.attn_grf = 0;
  char what[256];
  std::snprintf(what, sizeof what, "%s %d tree rows x attention groups (rows in one tile, across a page, across the 4,096 split change): attn and out == the chain over the row's ancestors, bit for bit (%d differ)",
                label.c_str(), checked, bad);
  expect(bad == 0, what);
  std::snprintf(what, sizeof what, "%s gqa_kv_move of the accepted paths == the chain's K / V bytes (%d token-heads, %d differ)", label.c_str(), moved, bad_mv);
  expect(bad_mv == 0, what);
  free_card(f, cd);
}

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const bool two = loader.size() > 1;
    xenithium::Fabric f(loader, two ? std::vector<size_t>{0, 1} : std::vector<size_t>{0});
    const Model m6 = make_model(512, 12, 2, 31);  // 6 q heads per kv head (Qwen3.8-27B's group)
    const Model m12 = make_model(512, 12, 1, 37);  // 12 (qwen38's group, the 8 + 4 DPAS split)
    for (const auto kv : {k::QsaKv::i8, k::QsaKv::bf16}) {
      test_decode(f, m6, kv, 1, "h 512, 12 q / 2 kv heads");
      if (two) test_decode(f, m6, kv, 2, "h 512, 12 q / 2 kv heads");
    }
    test_decode(f, m12, k::QsaKv::i8, 1, "h 512, 12 q / 1 kv head");
    test_long(f, m6, k::QsaKv::i8);
    test_long(f, m6, k::QsaKv::bf16);
    test_select(f, m6, k::QsaKv::i8);
    test_select(f, m6, k::QsaKv::bf16);
    for (const auto kv : {k::QsaKv::i8, k::QsaKv::bf16}) {
      test_rows(f, m6, kv, 1);
      if (two) test_rows(f, m6, kv, 2);
    }
    test_rows(f, m12, k::QsaKv::i8, 1);  // the staged body's grp-12 shapes
    test_rows_select(f, m6, k::QsaKv::i8);  // page-selected prefill attention
    test_rows_select(f, m6, k::QsaKv::bf16);
    test_rows_select(f, m6, k::QsaKv::i8, 71680);  // a context of more pages than a select work-group's items
    test_tree(f, m6, k::QsaKv::i8);  // DDTree
    test_tree(f, m6, k::QsaKv::bf16);
    test_tree(f, m6, k::QsaKv::i8, 1, false, "int8-fast");  // trees under the decode attention numerics
    test_tree(f, m6, k::QsaKv::i8, 1, true, "int8");
    test_tree(f, m6, k::QsaKv::i8, 2, false, "fast on int8");
    test_tree(f, m6, k::QsaKv::i8, 0, false, "fast");
    test_tree(f, m6, k::QsaKv::bf16, 0, false, "fast");
    // int8-v4 KV (K int8, V int4 + a bf16 scale per 32 dims), grp 6 (the dense 27B)
    test_decode(f, m6, k::QsaKv::i8v4, 1, "h 512, 12 q / 2 kv heads");
    if (two) test_decode(f, m6, k::QsaKv::i8v4, 2, "h 512, 12 q / 2 kv heads");
    test_long(f, m6, k::QsaKv::i8v4);
    test_rows(f, m6, k::QsaKv::i8v4, 1);
    if (two) test_rows(f, m6, k::QsaKv::i8v4, 2);
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
