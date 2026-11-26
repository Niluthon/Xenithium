// Gate for the MoE decode block (kernels/moe/moe.hpp): router, top-k, routed int4 experts, int8
// shared expert, weighted combine, and the two-card all-reduce through the mailbox.
// Synthetic weights at the model's dims (hidden 2560, inter 640, 512 experts, top-10) and at a
// small shape. Experts alias a few distinct blobs through the pointer table (the kernel reads
// whatever the table names), so the host reference only dequantises those. Not exact: the
// device sums fp32 in its own order where the reference sums in double, and device exp differs
// by an ulp, so outputs get the bound of the other step gates (2 bf16 ulps of max(|ref|, row
// rms), <= 1 % of elements off). Routing must equal the reference exactly; two cards must agree
// bit for bit; repeated steps must be bit-identical.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <thread>
#include <type_traits>
#include <memory>
#include <string>
#include <vector>

#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/kernels/moe/moe_cpu.hpp"
#include "xenithium/kernels/moe/moe_rows.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

struct Cmp {
  double scaled = 0;
  size_t off = 0, n = 0;
};
Cmp compare(const std::vector<uint16_t> &a, const std::vector<float> &ref, size_t row) {
  Cmp c;
  c.n = a.size();
  for (size_t r0 = 0; r0 < a.size(); r0 += row) {
    double ss = 0;
    for (size_t i = r0; i < r0 + row; ++i) ss += static_cast<double>(ref[i]) * ref[i];
    const double rms = std::sqrt(ss / static_cast<double>(row));
    for (size_t i = r0; i < r0 + row; ++i) {
      const double d = std::fabs(k::bf16_value(a[i]) - ref[i]);
      c.off += k::bf16_value(a[i]) != ref[i];
      c.scaled = std::max(c.scaled, d / std::max({static_cast<double>(std::fabs(ref[i])), rms, 1e-30}));
    }
  }
  return c;
}
std::string cmp_text(const Cmp &c) {
  char b[128];
  std::snprintf(b, sizeof b, "worst %.2e of max(|ref|, rms), %zu / %zu differ (%.3f %%)", c.scaled, c.off, c.n, 100.0 * c.off / std::max<size_t>(1, c.n));
  return b;
}
bool cmp_ok(const Cmp &c) { return c.scaled <= 1.0 / 128 && c.off * 100 <= c.n; }

// One synthetic expert: GPTQ tensors + its packed blob + the dequantised reference.
struct Expert {
  std::vector<uint8_t> blob;
  k::MoeRefExpert ref;
};
k::MoeBlobLayout layout_of(int H, int I, size_t *stride) {
  k::MoeBlobLayout l;
  l.gu_codes = 0;
  l.dn_codes = static_cast<size_t>(H) * 2 * I / 2;
  l.gu_scales = l.dn_codes + static_cast<size_t>(I) * H / 2;
  l.dn_scales = l.gu_scales + static_cast<size_t>(H / 128) * 2 * I * 2;
  *stride = (l.dn_scales + static_cast<size_t>(I / 128) * H * 2 + 4095) / 4096 * 4096;
  return l;
}
// qweight [k/8][n] random nibbles, scales fp16 [k/128][n]; dequant w[k][n].
void gptq(std::mt19937 &rng, int n, int kk, float smax, std::vector<uint32_t> &qw, std::vector<sycl::half> &sc, std::vector<float> &deq) {
  qw.resize(static_cast<size_t>(kk / 8) * n);
  sc.resize(static_cast<size_t>(kk / 128) * n);
  deq.resize(static_cast<size_t>(kk) * n);
  for (auto &v : qw) v = static_cast<uint32_t>(rng());
  std::uniform_real_distribution<float> us(0.5f * smax, smax);
  for (auto &v : sc) v = sycl::half(us(rng));
  for (int r = 0; r < kk; ++r)
    for (int c = 0; c < n; ++c) {
      const int q = static_cast<int>((qw[static_cast<size_t>(r / 8) * n + c] >> (4 * (r % 8))) & 15u);
      deq[static_cast<size_t>(r) * n + c] = static_cast<float>(q - 8) * static_cast<float>(sc[static_cast<size_t>(r / 128) * n + c]);
    }
}
Expert make_expert(std::mt19937 &rng, int H, int I, const k::MoeBlobLayout &lay, size_t stride) {
  Expert e;
  std::vector<uint32_t> qg, qu, qd;
  std::vector<sycl::half> sg, su, sd;
  gptq(rng, I, H, 0.02f, qg, sg, e.ref.gate);
  gptq(rng, I, H, 0.02f, qu, su, e.ref.up);
  gptq(rng, H, I, 0.02f, qd, sd, e.ref.down);
  e.blob.assign(stride, 0);
  k::pack_w4_gptq_cat(qg.data(), qu.data(), I, 2 * I, H, e.blob.data() + lay.gu_codes);
  k::pack_w4_gptq(qd.data(), H, I, e.blob.data() + lay.dn_codes);
  auto *gs = reinterpret_cast<sycl::half *>(e.blob.data() + lay.gu_scales);
  for (int g = 0; g < H / 128; ++g)
    for (int c = 0; c < I; ++c) {
      gs[static_cast<size_t>(g) * 2 * I + c] = sg[static_cast<size_t>(g) * I + c];
      gs[static_cast<size_t>(g) * 2 * I + I + c] = su[static_cast<size_t>(g) * I + c];
    }
  std::memcpy(e.blob.data() + lay.dn_scales, sd.data(), sd.size() * 2);
  return e;
}

// int8 dense matrix [n][kk]: codes random, per-row scale; packed + dequantised reference.
void make_i8(std::mt19937 &rng, int n, int kk, float smax, std::vector<int8_t> &packed, std::vector<float> &scale, std::vector<float> &deq) {
  std::vector<int8_t> codes(static_cast<size_t>(n) * kk);
  std::uniform_int_distribution<int> uc(-127, 127);
  for (auto &v : codes) v = static_cast<int8_t>(uc(rng));
  std::uniform_real_distribution<float> us(0.5f * smax, smax);
  scale.resize(n);
  for (auto &v : scale) v = us(rng);
  packed.assign(k::dense_packed_elems(n, kk), 0);
  k::pack_dense_i8(codes.data(), n, kk, packed.data());
  deq.resize(codes.size());
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < kk; ++c) deq[static_cast<size_t>(r) * kk + c] = codes[static_cast<size_t>(r) * kk + c] * scale[r];
}

struct Model {
  k::MoeShape s;  // cards / card filled per run
  int distinct = 0;
  size_t stride = 0;
  k::MoeBlobLayout lay;
  std::vector<Expert> ex;  // distinct experts
  std::vector<uint16_t> router_bits;  // [Rn][H] (E rows + shared gate + padding zeros)
  int router_n = 0;
  std::vector<int8_t> sgu_p, sdn_p;
  std::vector<float> sgu_s, sdn_s;
  k::MoeRefWeights ref;
  std::vector<float> sgu_deq;
};
int alias(const Model &m, int e) { return (e * 7 + e / 5) % m.distinct; }  // expert -> distinct blob

Model make_model(int H, int I, int SI, int E, int K, int distinct, uint32_t seed) {
  Model m;
  m.s = k::MoeShape{H, I, E, K, SI, 1, 0};
  m.distinct = distinct;
  m.lay = layout_of(H, I, &m.stride);
  std::mt19937 rng(seed);
  for (int i = 0; i < distinct; ++i) m.ex.push_back(make_expert(rng, H, I, m.lay, m.stride));
  m.router_n = (E + 1 + 15) / 16 * 16;
  m.router_bits.assign(static_cast<size_t>(m.router_n) * H, 0);
  std::normal_distribution<float> nd(0.f, 1.f / std::sqrt(static_cast<float>(H)));
  m.ref.router.resize(static_cast<size_t>(E) * H);
  m.ref.shared_gate.resize(H);
  for (int r = 0; r <= E; ++r)
    for (int c = 0; c < H; ++c) {
      const uint16_t b = k::bf16_bits(nd(rng));
      m.router_bits[static_cast<size_t>(r) * H + c] = b;
      (r < E ? m.ref.router[static_cast<size_t>(r) * H + c] : m.ref.shared_gate[c]) = k::bf16_value(b);
    }
  make_i8(rng, 2 * SI, H, 0.0006f, m.sgu_p, m.sgu_s, m.sgu_deq);
  m.ref.s_gate.assign(m.sgu_deq.begin(), m.sgu_deq.begin() + static_cast<size_t>(SI) * H);
  m.ref.s_up.assign(m.sgu_deq.begin() + static_cast<size_t>(SI) * H, m.sgu_deq.end());
  make_i8(rng, H, SI, 0.0006f, m.sdn_p, m.sdn_s, m.ref.s_down);
  m.ref.experts.resize(E);
  for (int e = 0; e < E; ++e) m.ref.experts[e] = &m.ex[alias(m, e)].ref;
  return m;
}

// The model's buffers on one card.
struct Card {
  sycl::queue *q = nullptr;
  std::vector<void *> mem;
  k::MoeWeights w;
  k::MoeScratch sc;
  k::MoeWire wire;
  bf16 *x = nullptr, *out = nullptr;
};
template <typename T>
T *up(xenithium::Fabric &f, int c, const std::vector<T> &v, std::vector<void *> &mem) {
  auto *p = static_cast<T *>(f.alloc(c, std::max<size_t>(64, v.size() * sizeof(T))));
  f.compute(c).memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  mem.push_back(p);
  return p;
}
Card make_card(xenithium::Fabric &f, int c, const Model &m, int cards) {
  Card d;
  d.q = &f.compute(c);
  const auto &s = m.s;
  std::vector<uint16_t> rp(k::dense_packed_elems(m.router_n, s.hidden));
  k::pack_dense_bf16(m.router_bits.data(), m.router_n, s.hidden, rp.data());
  d.w.router = k::DenseWeights{up(f, c, rp, d.mem), nullptr, k::DenseKind::bf16, m.router_n, s.hidden};
  d.w.shared_gu = k::DenseWeights{up(f, c, m.sgu_p, d.mem), up(f, c, m.sgu_s, d.mem), k::DenseKind::i8, 2 * s.shared_inter, s.hidden};
  d.w.shared_down = k::DenseWeights{up(f, c, m.sdn_p, d.mem), up(f, c, m.sdn_s, d.mem), k::DenseKind::i8, s.hidden, s.shared_inter};
  // Distinct blobs on this card (TP: this card's cut of each), then the table of this card's experts.
  std::vector<uint8_t> all;
  size_t stride = m.stride;
  k::MoeBlobLayout lay = m.lay;
  if (s.tp) {
    lay = k::moe_tp_layout(s.hidden, s.inter, cards, &stride);
    for (const auto &e : m.ex) {
      std::vector<uint8_t> cut(stride, 0);
      k::moe_split_tp(m.lay, s.hidden, s.inter, cards, c, e.blob.data(), lay, cut.data());
      all.insert(all.end(), cut.begin(), cut.end());
    }
  } else {
    for (const auto &e : m.ex) all.insert(all.end(), e.blob.begin(), e.blob.end());
  }
  const uint8_t *blobs = up(f, c, all, d.mem);
  const int per = s.tp ? s.experts : s.experts / cards, first = s.tp ? 0 : c * per;
  std::vector<const uint8_t *> table(per);
  for (int i = 0; i < per; ++i) table[i] = blobs + static_cast<size_t>(alias(m, first + i)) * stride;
  d.w.blobs = up(f, c, table, d.mem);
  d.w.blob = lay;
  k::MoeShape sh = s;
  sh.cards = cards;
  sh.card = c;
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(c, bytes);
    d.mem.push_back(p);
    return p;
  };
  const size_t cap = static_cast<size_t>(sh.rows_cap);
  d.sc.logits = static_cast<float *>(dev(cap * m.router_n * 4));  // the model's largest step (MoeShape:rows_cap)
  d.sc.route = static_cast<k::MoeRoute *>(dev(sizeof(k::MoeRoute)));
  d.sc.h = static_cast<bf16 *>(dev(k::moe_h_elems(sh) * 2));
  d.sc.partial = static_cast<float *>(dev(k::moe_partial_elems(sh) * 4));
  d.sc.xr = static_cast<bf16 *>(dev(cap * s.hidden * 2));  // (rot)
  d.sc.xq = static_cast<int8_t *>(dev(cap * s.hidden));  // (decode a8)
  d.sc.xs = static_cast<float *>(dev(cap * (s.hidden / 128) * 4));
  d.sc.hq = static_cast<int8_t *>(dev(k::moe_h_elems(sh)));
  d.sc.hs = static_cast<float *>(dev(k::moe_hs_elems(sh) * 4));
  d.x = static_cast<bf16 *>(dev(cap * s.hidden * 2));
  d.out = static_cast<bf16 *>(dev(cap * s.hidden * 2));
  return d;
}
void free_card(xenithium::Fabric &f, Card &d) {
  for (void *p : d.mem) f.release(p);
  d.mem.clear();
}

std::vector<float> make_x(int H, uint32_t seed, int rows = k::kStepRows32) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(static_cast<size_t>(rows) * H);  // the 32-row shape's rows
  for (auto &v : x) v = k::bf16_value(k::bf16_bits(nd(rng)));
  return x;
}
std::vector<uint16_t> bits_of(const std::vector<float> &x) {
  std::vector<uint16_t> b(x.size());
  for (size_t i = 0; i < x.size(); ++i) b[i] = k::bf16_bits(x[i]);
  return b;
}
std::vector<uint16_t> read16(sycl::queue &q, const bf16 *p, size_t n) {
  std::vector<uint16_t> v(n);
  q.memcpy(v.data(), p, n * 2).wait();
  return v;
}


// The device's h for every (row, expert) of this card and its shared half, as reference keys.
void read_h(sycl::queue &q, const k::MoeShape &s, int rows, const k::MoeScratch &sc, k::MoeRefH &out) {
  k::MoeRoute rt;
  q.memcpy(&rt, sc.route, sizeof rt).wait();
  const int HW = std::max(s.inter, s.shared_inter);
  std::vector<uint16_t> hb(k::moe_h_elems(s));
  q.memcpy(hb.data(), sc.h, hb.size() * 2).wait();
  // Columns [n0, n1) of the key's vector come from h columns [n0 - shift, n1 - shift) (TP: a card
  // stores its slice of an expert's h from column 0); other cards fill the rest of a TP key.
  const int M = k::row_shape(rows);  // h is laid out by the launch's shape: [M x topk + 1][M][HW]
  auto fill = [&](std::vector<float> &v, int slot, int r, int n0, int n1, int n, int shift) {
    if (v.empty()) v.assign(n, 0.f);
    for (int i = n0; i < n1; ++i) v[i] = k::bf16_value(hb[(static_cast<size_t>(slot) * M + r) * HW + i - shift]);
  };
  const int il = s.inter_local(), e0 = s.tp ? s.card * il : 0;
  for (int u = 0; u < rt.count; ++u)
    for (int r = 0; r < rows; ++r)
      if ((rt.mask[u] >> r) & 1u) fill(out[{r, rt.expert[u]}], u, r, e0, e0 + il, s.inter, e0);
  const int i0 = rt.shared_lo * 16, i1 = rt.shared_hi * 16;
  for (int r = 0; r < rows; ++r) fill(out[{r, -1 - s.card}], M * s.topk, r, i0, i1, s.shared_inter, 0);
}
// h vs the reference's h: every element within 1 bf16 ulp, few off at all.
struct HCmp {
  size_t n = 0, off = 0;
  double worst = 0;  // max bf16 ulp distance
  bool keys = true;
};
HCmp compare_h(const k::MoeRefH &dev, const k::MoeRefH &ref) {
  HCmp c;
  c.keys = dev.size() == ref.size();
  for (const auto &[key, v] : ref) {
    const auto it = dev.find(key);
    if (it == dev.end()) {
      c.keys = false;
      continue;
    }
    for (size_t i = 0; i < v.size(); ++i) {
      ++c.n;
      if (it->second[i] != v[i]) {
        ++c.off;
        auto lin = [](float f) {
          const int b = k::bf16_bits(f);
          return (b & 0x8000) ? -(b & 0x7fff) : b;
        };
        c.worst = std::max(c.worst, static_cast<double>(std::abs(lin(it->second[i]) - lin(v[i]))));
      }
    }
  }
  return c;
}
// A flip of g or u (fp32 sum order, device exp) can move h by more than its own ulp where
// silu's log-slope exceeds 1 (g < -1.3), so: <= 4 ulps, <= 0.5 % of elements off.
bool h_ok(const HCmp &c) { return c.keys && c.worst <= 4 && c.off * 200 <= c.n; }
std::string h_text(const HCmp &c) {
  char b[128];
  std::snprintf(b, sizeof b, "%zu / %zu h differ (%.3f %%), worst %.0f ulp%s", c.off, c.n, 100.0 * c.off / std::max<size_t>(1, c.n), c.worst,
                c.keys ? "" : ", KEYS DIFFER");
  return b;
}

bool routes_equal(sycl::queue &q, const k::MoeRoute *rt, const k::MoeShape &s, int rows, const k::MoeRefOut &o) {
  k::MoeRoute h;
  q.memcpy(&h, rt, sizeof h).wait();
  bool ok = true;
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < s.topk; ++j) {
      ok &= h.ids[r][j] == o.ids[static_cast<size_t>(r) * s.topk + j];
      ok &= std::fabs(h.topw[r][j] - o.topw[static_cast<size_t>(r) * s.topk + j]) <= 1e-5f;
    }
  return ok;
}

// One card, cards = 1: every expert lives here.
void test_one(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d = make_card(f, 0, m, 1);
  k::MoeShape s = m.s;
  s.cards = 1;
  s.card = 0;
  for (int rows : {1, 3, 4, 5, 8}) {
    const auto x = make_x(s.hidden, 500u + rows);
    k::MoeRefOut o;
    k::moe_ref(s, rows, x.data(), m.ref, o);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    std::vector<uint16_t> first;
    for (int ks : {1, 2, 4, 8}) {
      d.q->fill(d.out, bf16(-777.f), 8ull * s.hidden).wait();
      k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
      d.q->wait_and_throw();
      const auto got = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
      char what[200];
      std::snprintf(what, sizeof what, "%s one card rows %d ksplit %d: routing == reference (ids exact, weights 1e-5)", tag, rows, ks);
      if (ks == 1) expect(routes_equal(*d.q, d.sc.route, s, rows, o), what);
      k::MoeRefH dh;
      read_h(*d.q, s, rows, d.sc, dh);
      const HCmp hc = compare_h(dh, o.h);
      std::snprintf(what, sizeof what, "%s one card rows %d ksplit %d: h vs reference %s", tag, rows, ks, h_text(hc).c_str());
      expect(h_ok(hc), what);
      k::MoeRefOut of;
      k::moe_ref(s, rows, x.data(), m.ref, of, &dh);
      const Cmp c = compare(got, of.out, s.hidden), cf = compare(got, o.out, s.hidden);
      std::snprintf(what, sizeof what, "%s one card rows %d ksplit %d: out vs reference from the device's h %s (free-running: %s)", tag, rows,
                    ks, cmp_text(c).c_str(), cmp_text(cf).c_str());
      expect(cmp_ok(c) && cf.scaled <= 1.0 / 128, what);
      if (ks == 1) {
        first = got;
        k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);  // the same splits
        const auto again = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
        std::snprintf(what, sizeof what, "%s one card rows %d: run-to-run bit-identical", tag, rows);
        expect(again == first, what);
        // the ranking's counter (MoeScratch:hits): + 1 per expert the step reads (once, whatever the rows),
        // twice after two steps; the outputs unchanged
        uint32_t *hits = sycl::aligned_alloc_device<uint32_t>(4096, static_cast<size_t>(s.experts), *d.q);
        d.q->memset(hits, 0, static_cast<size_t>(s.experts) * 4).wait();
        k::MoeScratch sh = d.sc;
        sh.hits = hits;
        k::moe_decode(*d.q, s, rows, d.x, d.w, sh, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
        const auto counted = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
        k::moe_decode(*d.q, s, rows, d.x, d.w, sh, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
        std::vector<uint32_t> hv(static_cast<size_t>(s.experts));
        d.q->memcpy(hv.data(), hits, hv.size() * 4).wait();
        sycl::free(hits, *d.q);
        std::vector<uint32_t> want(static_cast<size_t>(s.experts), 0u);
        {
          std::vector<bool> any(static_cast<size_t>(s.experts), false);
          for (size_t j = 0; j < static_cast<size_t>(rows) * s.topk; ++j) any[static_cast<size_t>(o.ids[j])] = true;
          for (int e = 0; e < s.experts; ++e) want[static_cast<size_t>(e)] = any[static_cast<size_t>(e)] ? 2u : 0u;
        }
        std::snprintf(what, sizeof what, "%s one card rows %d: route counter = 1 per unique expert per step (2 steps), outputs unchanged", tag, rows);
        expect(hv == want && counted == first, what);
      }
    }
  }
  // A step of 5-8 rows (the 8-row shape) gives every row the bits of the same row run in
  // a step of its own segment's rows (the 4-row shape): rows [0, 4) alone, then rows [4, rows) alone.
  for (int rows : {8, 6}) {
    const auto x = make_x(s.hidden, 700u + rows);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out);
    const auto all = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
    k::moe_decode(*d.q, s, 4, d.x, d.w, d.sc, nullptr, 0, d.out);
    auto parts = read16(*d.q, d.out, 4ull * s.hidden);
    k::moe_decode(*d.q, s, rows - 4, d.x + 4ull * s.hidden, d.w, d.sc, nullptr, 0, d.out);
    const auto tail = read16(*d.q, d.out, static_cast<size_t>(rows - 4) * s.hidden);
    parts.insert(parts.end(), tail.begin(), tail.end());
    char what[200];
    std::snprintf(what, sizeof what, "%s one card rows %d (8-row shape) == rows 0-3 and 4-%d as their own steps, bit for bit", tag, rows, rows - 1);
    expect(all == parts, what);
  }
  // A step of 9-16 rows (the 16-row shape) == rows 0-7 and rows 8.. as their own steps, bit for bit
  for (int rows : {16, 12, 9}) {
    const auto x = make_x(s.hidden, 900u + rows);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out);
    const auto all = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
    k::moe_decode(*d.q, s, 8, d.x, d.w, d.sc, nullptr, 0, d.out);
    auto parts = read16(*d.q, d.out, 8ull * s.hidden);
    k::moe_decode(*d.q, s, rows - 8, d.x + 8ull * s.hidden, d.w, d.sc, nullptr, 0, d.out);
    const auto tail = read16(*d.q, d.out, static_cast<size_t>(rows - 8) * s.hidden);
    parts.insert(parts.end(), tail.begin(), tail.end());
    char what[200];
    std::snprintf(what, sizeof what, "%s one card rows %d (16-row shape) == rows 0-7 and 8-%d as their own steps, bit for bit", tag, rows, rows - 1);
    expect(all == parts, what);
  }
  // A step of 17-32 rows (the 32-row shape, four 8-row quarters) == rows 0-15 and rows 16.. as
  // their own steps, bit for bit (every k split: the quarters run the 8-row body on the same split)
  for (int rows : {32, 24, 17})
    for (int ks : {0, 1, 8}) {
      const auto x = make_x(s.hidden, 1100u + rows);
      const auto xb = bits_of(x);
      d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
      k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
      const auto all = read16(*d.q, d.out, static_cast<size_t>(rows) * s.hidden);
      k::moe_decode(*d.q, s, 16, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
      auto parts = read16(*d.q, d.out, 16ull * s.hidden);
      k::moe_decode(*d.q, s, rows - 16, d.x + 16ull * s.hidden, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
      const auto tail = read16(*d.q, d.out, static_cast<size_t>(rows - 16) * s.hidden);
      parts.insert(parts.end(), tail.begin(), tail.end());
      char what[200];
      std::snprintf(what, sizeof what, "%s one card rows %d ks %d (32-row shape) == rows 0-15 and 16-%d as their own steps, bit for bit", tag, rows, ks, rows - 1);
      expect(all == parts, what);
    }
  free_card(f, d);
}

// A step of 33-64 rows (the 64-row shape: two mask words per slot, a slot's packed rows in 32-row
// windows, the router as two launches) == rows 0-31 and rows 32.. as their own steps, bit for bit, at every k split.
// `same` rows of the step are copies of row 0, so their experts hold that many rows: 0 = every slot as the routes fall,
// 12 = slots of two blocks, 40 = slots past one 32-row window.
void test_wide(xenithium::Fabric &f, const Model &m0, const char *tag) {
  Model m = m0;
  m.s.rows_cap = k::kMoeRows;
  Card d = make_card(f, 0, m, 1);
  k::MoeShape s = m.s;
  s.cards = 1;
  s.card = 0;
  const size_t H = static_cast<size_t>(s.hidden);
  constexpr int kHalf = k::kStepRows32;  // the step's first launch as its own step
  for (int rows : {64, 48, 33})
    for (int same : {0, 12, 40})
      for (int ks : {0, 1, 8}) {
        if (same >= rows || (same && ks == 1)) continue;
        auto xb = bits_of(make_x(s.hidden, 1300u + rows, k::kMoeRows));
        for (int r = 1; r < same; ++r) std::copy_n(xb.begin(), H, xb.begin() + static_cast<size_t>(r * (rows / same)) * H);  // over both windows
        d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
        k::moe_decode(*d.q, s, rows, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
        const auto all = read16(*d.q, d.out, static_cast<size_t>(rows) * H);
        k::moe_decode(*d.q, s, kHalf, d.x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
        auto parts = read16(*d.q, d.out, static_cast<size_t>(kHalf) * H);
        k::moe_decode(*d.q, s, rows - kHalf, d.x + static_cast<size_t>(kHalf) * H, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
        const auto tail = read16(*d.q, d.out, static_cast<size_t>(rows - kHalf) * H);
        parts.insert(parts.end(), tail.begin(), tail.end());
        size_t bad = 0;
        for (size_t r = 0; r < static_cast<size_t>(rows); ++r)
          bad += !std::equal(all.begin() + static_cast<std::ptrdiff_t>(r * H), all.begin() + static_cast<std::ptrdiff_t>((r + 1) * H),
                             parts.begin() + static_cast<std::ptrdiff_t>(r * H));
        char what[240];
        std::snprintf(what, sizeof what, "%s one card rows %d ks %d, %d equal rows (64-row shape) == rows 0-31 and 32-%d as their own steps, bit for bit (%zu rows differ)",
                      tag, rows, ks, same, rows - 1, bad);
        expect(bad == 0, what);
      }
  {  // a step past the scratch, and a form the 64-row shape does not run, are refused
    auto refused = [&](const k::MoeShape &shape) {
      try {
        k::moe_decode(*d.q, shape, kHalf + 1, d.x, d.w, d.sc, nullptr, 0, d.out);
      } catch (const std::invalid_argument &) {
        return true;
      }
      return false;
    };
    k::MoeShape small = s, rotated = s;
    small.rows_cap = kHalf;
    rotated.rot = true;
    expect(refused(small) && refused(rotated), std::string(tag) + " one card: 33 rows refused on a 32-row scratch and with rotated experts");
  }
  free_card(f, d);
}

// Decode W4A8 (MoeShape:a8, tp, one card). (1) The x and h quantisers == a CPU reference bit for
// bit (under rot on bf16(H_128 x) / bf16(H_64 h), from the device's own h). (2) The output vs the
// reference fed the device's dequantised int8 h, within the step bound. (3) Within int8 rounding of the
// W4A16 output. (4) Run to run, and the 8-row shape == its two 4-row halves, bit for bit. (5) No rot: on x
// whose groups quantise losslessly, h == the W4A16 h bit for bit (int32 sums are exact).
void test_a8(xenithium::Fabric &f, const Model &m, const char *tag) {
  Card d = make_card(f, 0, m, 1);
  k::MoeShape s = m.s;
  s.cards = 1;
  s.card = 0;
  k::MoeShape s8 = s;  // the separate rot_quant launches (writes sc.hq / sc.hs, read back below)
  s8.a8 = true;
  s8.a8_fuse = false;
  k::MoeShape s8f = s8;  // x in the router's launch, h on its own launch
  s8f.a8_fuse = true;
  s8f.a8_h_in_gate_up = false;
  k::MoeShape s8d = s8f;  // h inside down (a8_h_in_down)
  s8d.a8_h_in_down = true;
  k::MoeShape s8g = s8f;  // the shipped path: h in gate_up's epilogue
  s8g.a8_h_in_gate_up = true;
  const int H = s.hidden, I = s.inter_local(), G = H / 128, NH = I / 64, HW = std::max(s.inter, s.shared_inter);
  auto rot_bf = [](float *p, int n) {  // bf16(H_n p), the device's butterflies and rounding
    for (int hh = 1; hh < n; hh <<= 1)
      for (int i = 0; i < n; i += hh << 1)
        for (int j = i; j < i + hh; ++j) {
          const float u = p[j], w = p[j + hh];
          p[j] = u + w, p[j + hh] = u - w;
        }
    const float sc = n == 64 ? 0.125f : 0.08838834764831845f;
    for (int i = 0; i < n; ++i) p[i] = k::bf16_value(k::bf16_bits(p[i] * sc));
  };
  auto quant = [](const float *v, int n, int8_t *q, float *sv) {  // row_quant.hpp's rule
    float amax = 0.f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(v[i]));
    *sv = amax > 0.f ? static_cast<float>(static_cast<double>(amax) / 127.0) : 0.f;
    const float inv = amax > 0.f ? static_cast<float>(127.0 / static_cast<double>(amax)) : 0.f;
    for (int i = 0; i < n; ++i) q[i] = static_cast<int8_t>(std::clamp(std::nearbyint(v[i] * inv), -127.f, 127.f));
  };
  auto run = [&](const k::MoeShape &sh, int rows, int ks, const bf16 *x) {
    d.q->fill(d.out, bf16(-777.f), 32ull * H).wait();
    k::moe_decode(*d.q, sh, rows, x, d.w, d.sc, nullptr, 0, d.out, ks, ks == 8 ? 4 : ks);
    d.q->wait_and_throw();
    return read16(*d.q, d.out, static_cast<size_t>(rows) * H);
  };
  char what[240];
  for (int rows : {1, 4, 5, 8}) {
    const int M = k::row_shape(rows);
    const auto x = make_x(H, 900u + rows);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    for (int ks : {1, 8}) {
      const auto o16 = run(s, rows, ks, d.x);
      const auto o8g = run(s8g, rows, ks, d.x);
      std::vector<int8_t> hqg(k::moe_h_elems(s));
      std::vector<float> hsg(k::moe_hs_elems(s));
      d.q->memcpy(hqg.data(), d.sc.hq, hqg.size()).wait();
      d.q->memcpy(hsg.data(), d.sc.hs, hsg.size() * 4).wait();
      const auto o8d = run(s8d, rows, ks, d.x);
      const auto o8f = run(s8f, rows, ks, d.x);
      std::vector<int8_t> xqf(static_cast<size_t>(k::row_shape(rows)) * H);
      d.q->memcpy(xqf.data(), d.sc.xq, xqf.size()).wait();
      const auto o8 = run(s8, rows, ks, d.x);
      // (1) quantisers
      std::vector<int8_t> xq(static_cast<size_t>(M) * H), hq(k::moe_h_elems(s));
      std::vector<float> xs(static_cast<size_t>(M) * G), hs(k::moe_hs_elems(s));
      std::vector<uint16_t> hb(k::moe_h_elems(s));
      d.q->memcpy(xq.data(), d.sc.xq, xq.size()).wait();
      d.q->memcpy(xs.data(), d.sc.xs, xs.size() * 4).wait();
      d.q->memcpy(hq.data(), d.sc.hq, hq.size()).wait();
      d.q->memcpy(hs.data(), d.sc.hs, hs.size() * 4).wait();
      d.q->memcpy(hb.data(), d.sc.h, hb.size() * 2).wait();
      k::MoeRoute rt;
      d.q->memcpy(&rt, d.sc.route, sizeof rt).wait();
      size_t qbad = 0, qn = 0;
      for (int r = 0; r < M; ++r)
        for (int g = 0; g < G; ++g) {
          float v[128], sv;
          int8_t q[128];
          for (int i = 0; i < 128; ++i) v[i] = x[static_cast<size_t>(r) * H + g * 128 + i];
          if (s.rot) rot_bf(v, 128);
          quant(v, 128, q, &sv);
          qbad += xs[static_cast<size_t>(r) * G + g] != sv;
          for (int i = 0; i < 128; ++i) qbad += xq[static_cast<size_t>(r) * H + g * 128 + i] != q[i];
          qn += 129;
        }
      k::MoeRefH force;
      read_h(*d.q, s, rows, d.sc, force);  // keys + the shared halves (bf16 h, the device's)
      for (int u = 0; u < rt.count; ++u)
        for (int r = 0; r < M; ++r) {
          const size_t row = static_cast<size_t>(u) * M + r;
          std::vector<float> deq(static_cast<size_t>(I));
          for (int g = 0; g < NH; ++g) {
            float v[64], sv;
            int8_t q[64];
            for (int i = 0; i < 64; ++i) v[i] = k::bf16_value(hb[row * HW + g * 64 + i]);
            if (s.rot) rot_bf(v, 64);
            quant(v, 64, q, &sv);
            const float dsv = hs[row * NH + g];
            qbad += dsv != sv;
            for (int i = 0; i < 64; ++i) {
              qbad += hq[row * HW + g * 64 + i] != q[i];
              deq[static_cast<size_t>(g) * 64 + i] = static_cast<float>(hq[row * HW + g * 64 + i]) * dsv;
            }
            qn += 65;
          }
          if (r < rows && ((rt.mask[u] >> r) & 1u)) force[{r, rt.expert[u]}] = deq;
        }
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: x and h int8 quantisers == the CPU reference (%zu of %zu differ)", tag, rows, ks, qbad, qn);
      expect(qbad == 0, what);
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: x in the router's launch, and h inside down, == separate launches, out and xq bit for bit", tag, rows, ks);
      expect(o8f == o8 && o8d == o8 && xqf == xq, what);
      bool hsame = true;  // gate_up's epilogue quantiser == the launch's, on every routed slot's routed rows
      for (int u = 0; u < rt.count; ++u)
        for (int r = 0; r < rows; ++r) {  // a slot stores its routed rows only
          if (!((rt.mask[u] >> r) & 1u)) continue;
          const size_t row = static_cast<size_t>(u) * M + r;
          for (int i = 0; i < I; ++i) hsame &= hqg[row * HW + i] == hq[row * HW + i];
          for (int g = 0; g < NH; ++g) hsame &= hsg[row * NH + g] == hs[row * NH + g];
        }
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: h quantised in gate_up's epilogue == the launch (hq, hs, out) bit for bit", tag, rows, ks);
      expect(hsame && o8g == o8, what);
      // (2) out vs the reference from the device's dequantised h
      k::MoeRefOut of;
      k::moe_ref(s, rows, x.data(), m.ref, of, &force);
      const Cmp c = compare(o8, of.out, H);
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: out vs reference from the device's int8 h %s", tag, rows, ks, cmp_text(c).c_str());
      expect(cmp_ok(c), what);
      // (3) vs W4A16
      double se = 0, sr = 0;
      for (size_t i = 0; i < o8.size(); ++i) {
        const double dv = k::bf16_value(o8[i]) - k::bf16_value(o16[i]);
        se += dv * dv;
        sr += static_cast<double>(k::bf16_value(o16[i])) * k::bf16_value(o16[i]);
      }
      const double rel = std::sqrt(se / std::max(sr, 1e-30));
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: out vs W4A16 relative rms %.2e (int8 rounding: < 2e-2)", tag, rows, ks, rel);
      expect(rel < 2e-2, what);
      // (4) run to run
      const auto again = run(s8, rows, ks, d.x);
      std::snprintf(what, sizeof what, "%s a8 rows %d ks %d: run-to-run bit-identical", tag, rows, ks);
      expect(again == o8, what);
    }
  }
  {  // (4) the 8-row shape == rows 0-3 and 4-7 as their own steps
    const auto x = make_x(H, 1700u);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    const auto all = run(s8, 8, 0, d.x);
    auto parts = run(s8, 4, 0, d.x);
    const auto tail = run(s8, 4, 0, d.x + 4ull * H);
    parts.insert(parts.end(), tail.begin(), tail.end());
    std::snprintf(what, sizeof what, "%s a8 rows 8 (8-row shape) == rows 0-3 and 4-7 as their own steps, bit for bit", tag);
    expect(all == parts, what);
  }
  {  // the 32-row shape == rows 0-15 and 16-31 as their own steps, every W4A8 form (the 32-row
     // shape's h quantised by its own launch where the 16-row shape's gate_up epilogue / down quantise it: the same bits)
    const auto x = make_x(H, 1800u);
    const auto xb = bits_of(x);
    d.q->memcpy(d.x, xb.data(), xb.size() * 2).wait();
    bool same = true;
    for (int form = 0; form < 4; ++form) {
      const k::MoeShape &sh = form == 0 ? s8 : form == 1 ? s8f : form == 2 ? s8d : s8g;
      const auto all = run(sh, 32, 0, d.x);
      auto parts = run(sh, 16, 0, d.x);
      const auto tail = run(sh, 16, 0, d.x + 16ull * H);
      parts.insert(parts.end(), tail.begin(), tail.end());
      same &= all == parts;
    }
    std::snprintf(what, sizeof what, "%s a8 rows 32 (32-row shape) == rows 0-15 and 16-31 as their own steps, bit for bit (4 W4A8 forms)", tag);
    expect(same, what);
  }
  if (!s.rot) {  // (5) lossless x: per (row, group) integers in [-127, 127], one at +-127, times 2^e
    std::mt19937 rng(77u);
    std::uniform_int_distribution<int> qi(-127, 127), ei(-9, -5);
    std::vector<float> xl(8ull * H);
    for (int r = 0; r < 8; ++r)
      for (int g = 0; g < G; ++g) {
        const float sc = std::ldexp(1.f, ei(rng));
        for (int i = 0; i < 128; ++i) xl[static_cast<size_t>(r) * H + g * 128 + i] = static_cast<float>(i == 0 ? (qi(rng) < 0 ? -127 : 127) : qi(rng)) * sc;
      }
    const auto xlb = bits_of(xl);
    d.q->memcpy(d.x, xlb.data(), xlb.size() * 2).wait();
    bool same = true;
    for (int rows : {1, 4, 8})
      for (int ks : {1, 8}) {
        run(s, rows, ks, d.x);
        k::MoeRefH h16, h8;
        read_h(*d.q, s, rows, d.sc, h16);
        run(s8, rows, ks, d.x);
        read_h(*d.q, s, rows, d.sc, h8);
        same &= h16 == h8;
      }
    std::snprintf(what, sizeof what, "%s a8 on losslessly quantised x: h == the W4A16 h bit for bit (rows 1 / 4 / 8, ks 1 / 8)", tag);
    expect(same, what);
  }
  free_card(f, d);
}

// Two cards, EP2 + the mailbox all-reduce: steps submitted back to back without a host wait,
// cycling through inputs; parity alternates per call.
// Every card of the fabric (2, or 4: the four-card wire, MoeWire:quad).
void test_cards(xenithium::Fabric &f, const Model &m, const char *tag) {
  const int C = static_cast<int>(f.size());
  const std::string who = std::string(tag) + (C == 2 ? " two cards" : " four cards");
  std::vector<Card> d;
  std::vector<k::MoeShape> s(C, m.s);
  for (int c = 0; c < C; ++c) {
    d.push_back(make_card(f, c, m, C));
    s[c].cards = C;
    s[c].card = c;
    d[c].wire.inbox = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_wire_words(s[c]) * 4));
    d[c].wire.error = static_cast<uint32_t *>(f.alloc(c, 4096));
  }
  for (int c = 0; c < C; ++c) {
    if (C == k::kWireCardsMax) {
      d[c].wire.quad.card = c;
      for (int o = 0; o < C; ++o) d[c].wire.quad.inbox[o] = d[o].wire.inbox;
    } else {
      d[c].wire.peer_inbox = d[1 - c].wire.inbox;
    }
    k::moe_wire_init(f.compute(c), s[c], d[c].wire.inbox).wait();
    f.compute(c).memset(d[c].wire.error, 0, 4).wait();
  }
  const int inputs = 3, steps = 24, rows = 4;
  std::vector<std::vector<float>> xs;
  std::vector<k::MoeRefOut> refs(inputs);
  for (int i = 0; i < inputs; ++i) {
    xs.push_back(make_x(m.s.hidden, 900u + i));
    k::moe_ref(s[0], rows, xs[i].data(), m.ref, refs[i]);
  }
  // All inputs resident: x buffers per input on each card, outputs per step.
  std::vector<std::vector<bf16 *>> dx(C), dout(C);
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < inputs; ++i) {
      dx[c].push_back(static_cast<bf16 *>(f.alloc(c, static_cast<size_t>(k::kMoeRows) * m.s.hidden * 2)));
      const auto b = bits_of(xs[i]);
      f.compute(c).memcpy(dx[c].back(), b.data(), b.size() * 2).wait();
    }
    for (int st = 0; st < steps; ++st) dout[c].push_back(static_cast<bf16 *>(f.alloc(c, static_cast<size_t>(k::kMoeRows) * m.s.hidden * 2)));
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int st = 0; st < steps; ++st)
    for (int c = 0; c < C; ++c) k::moe_decode(f.compute(c), s[c], rows, dx[c][st % inputs], d[c].w, d[c].sc, &d[c].wire, st, dout[c][st]);
  for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  bool same = true, det = true, ok = true, hok = true;
  Cmp worst;
  HCmp hworst;
  std::vector<uint16_t> per_input[inputs];
  for (int st = 0; st < steps; ++st) {
    const auto a = read16(f.compute(0), dout[0][st], static_cast<size_t>(rows) * m.s.hidden);
    for (int c = 1; c < C; ++c) same &= read16(f.compute(c), dout[c][st], a.size()) == a;
    auto &pi = per_input[st % inputs];
    if (pi.empty()) {
      pi = a;
      // h of this step from every card (the scratch is overwritten by later steps: rerun it).
      k::MoeRefH dh;
      for (int c = 0; c < C; ++c) {
        k::moe_decode(f.compute(c), s[c], rows, dx[c][st % inputs], d[c].w, d[c].sc, &d[c].wire, steps + st, dout[c][st]);
      }
      for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
      for (int c = 0; c < C; ++c) read_h(f.compute(c), s[c], rows, d[c].sc, dh);
      const HCmp hc = compare_h(dh, refs[st % inputs].h);
      hok &= h_ok(hc);
      if (hc.off >= hworst.off) hworst = hc;
      k::MoeRefOut of;
      k::moe_ref(s[0], rows, xs[st % inputs].data(), m.ref, of, &dh);
      const Cmp c = compare(a, of.out, m.s.hidden);
      ok &= cmp_ok(c);
      if (c.scaled >= worst.scaled) worst = c;
    } else {
      det &= pi == a;
    }
  }
  // Steps of both shapes back to back (4, 8, 5, 8...: the mailbox is laid out by the
  // capacity, so a card a step ahead in the other shape never overwrites an unread word); every step's
  // rows 0-3 (input 0) equal the 4-row step's, every card alike. The same with 16-, 32- and 20-row steps.
  bool mixed = true;
  for (int st = 0; st < steps; ++st) {
    const int j = st % 8;
    const int r = j == 3 ? 32 : j == 5 ? 16 : j == 6 ? 20 : j == 2 ? 5 : j % 4 == 0 ? 4 : 8;
    for (int c = 0; c < C; ++c) k::moe_decode(f.compute(c), s[c], r, dx[c][0], d[c].w, d[c].sc, &d[c].wire, 2 * steps + st, dout[c][st]);
  }
  for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  for (int st = 0; st < steps; ++st) {
    for (int c = 0; c < C; ++c) mixed &= read16(f.compute(c), dout[c][st], static_cast<size_t>(rows) * m.s.hidden) == per_input[0];
  }
  std::vector<uint32_t> err(C);
  for (int c = 0; c < C; ++c) f.compute(c).memcpy(&err[c], d[c].wire.error, 4).wait();
  char what[220];
  std::snprintf(what, sizeof what, "%s: steps of 4 / 5 / 8 / 16 / 20 / 32 rows back to back: rows 0-3 == the 4-row step's, every card", who.c_str());
  expect(mixed, what);
  // On a 64-row scratch (MoeShape:rows_cap), steps of 33 / 48 / 64 rows on every card (the cards' wire;
  // four cards: the quarter-piece down in tile pairs) == rows 0-31 and rows 32.. as their own steps, every card alike.
  if (m.s.rows_cap > k::kStepRows32) {
    constexpr int kFirst = k::kStepRows32;
    const size_t H = static_cast<size_t>(m.s.hidden);
    const auto xw = bits_of(make_x(m.s.hidden, 1500u, k::kMoeRows));
    std::vector<bf16 *> wide(C);
    for (int c = 0; c < C; ++c) {
      wide[c] = static_cast<bf16 *>(f.alloc(c, xw.size() * 2));
      f.compute(c).memcpy(wide[c], xw.data(), xw.size() * 2).wait();
    }
    auto step = [&](int r, const size_t row0, int parity) {  // rows [row0, row0 + r) as one step; card 0's output, every card checked equal
      for (int c = 0; c < C; ++c) k::moe_decode(f.compute(c), s[c], r, wide[c] + row0 * H, d[c].w, d[c].sc, &d[c].wire, parity, dout[c][0]);
      for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
      auto out = read16(f.compute(0), dout[0][0], static_cast<size_t>(r) * H);
      for (int c = 1; c < C; ++c)
        if (read16(f.compute(c), dout[c][0], out.size()) != out) out.clear();  // a card differs: nothing can match below
      return out;
    };
    bool wide_ok = true;
    int parity = 0;
    for (const int r : {64, 48, 33}) {
      const auto all = step(r, 0, parity++);
      auto parts = step(kFirst, 0, parity++);
      const auto tail = step(r - kFirst, kFirst, parity++);
      parts.insert(parts.end(), tail.begin(), tail.end());
      wide_ok &= !all.empty() && all == parts;
    }
    for (int c = 0; c < C; ++c) f.release(wide[c]);
    std::snprintf(what, sizeof what, "%s: steps of 64 / 48 / 33 rows (64-row shape) == rows 0-31 and 32.. as their own steps, every card", who.c_str());
    expect(wide_ok, what);
    if (C == k::kWireCardsMax) {  // the owner sum's mailbox holds kPairRows / 4 rows per card: a longer step is refused
      k::MoeWire w = d[0].wire;
      k::OwnerMail mail;
      for (int c = 0; c < C; ++c) mail.inbox[c] = d[c].wire.inbox;  // never reached: the check comes first
      w.owner = mail, w.owner.card = 0;
      bool refused = false;
      try {
        k::moe_decode(f.compute(0), s[0], k::kPairRows + 1, dx[0][0], d[0].w, d[0].sc, &w, 0, dout[0][0]);
      } catch (const std::invalid_argument &) {
        refused = true;
      }
      expect(refused, who + ": an owner sum of " + std::to_string(k::kPairRows + 1) + " rows is refused (its mailbox holds " + std::to_string(k::kPairRows) + ")");
    }
  }
  std::snprintf(what, sizeof what, "%s: h of every card vs reference, worst input %s", who.c_str(), h_text(hworst).c_str());
  expect(hok, what);
  std::snprintf(what, sizeof what, "%s (EP2 + mailbox all-reduce), %d steps back to back (%.1f ms): out vs reference from the device's h, worst input %s",
                who.c_str(), steps, ms, cmp_text(worst).c_str());
  expect(ok, what);
  std::snprintf(what, sizeof what, "%s: every card bit-identical every step", who.c_str());
  expect(same, what);
  std::snprintf(what, sizeof what, "%s: repeated inputs bit-identical across steps", who.c_str());
  expect(det, what);
  std::snprintf(what, sizeof what, "%s: no mailbox timeout, inboxes empty after the run", who.c_str());
  bool clean = std::all_of(err.begin(), err.end(), [](uint32_t e) { return e == 0; });
  for (int c = 0; c < C; ++c) {
    std::vector<uint32_t> box(k::moe_wire_words(s[c]));
    f.compute(c).memcpy(box.data(), d[c].wire.inbox, box.size() * 4).wait();
    for (uint32_t v : box) clean &= v == k::kMoeMailEmpty;
  }
  expect(clean, what);
  // Two cards == one card's grouping? Not bitwise (EP regroups the sum); the reference with
  // cards = 1 is within the bound of the two-card result.
  {
    k::MoeShape s1 = m.s;
    k::MoeRefOut o1;
    k::moe_ref(s1, rows, xs[0].data(), m.ref, o1);
    const Cmp c = compare(per_input[0], o1.out, m.s.hidden);
    std::snprintf(what, sizeof what, "%s vs the one-card reference (free-running, magnitude bound only): %s", who.c_str(), cmp_text(c).c_str());
    expect(c.scaled <= 1.0 / 128, what);
  }
  // A decode step's rows by owner: the combine as an owner sum (MoeWire:owner). A card's own rows of
  // out == the four-card wire's, bit for bit; its other rows are not written (a sentinel stays).
  if (C == k::kWireCardsMax) {
    const size_t sum_words = k::owner_sum_words(m.s.hidden);
    k::OwnerMail mail;
    for (int c = 0; c < C; ++c) {
      mail.inbox[c] = static_cast<uint32_t *>(f.alloc_mailbox(c, sum_words * 4));
      k::owner_mail_init(f.compute(c), mail.inbox[c], sum_words).wait();
    }
    const bf16 sentinel = bf16(-777.f);
    const uint16_t sentinel_bits = bits_of(std::vector<float>{-777.f})[0];
    bool own_ok = true, others_untouched = true;
    int parity = 0;
    for (const int r : {4, 5, 8, 16, 20, 32, 8}) {
      const size_t elems = static_cast<size_t>(r) * m.s.hidden;
      for (int c = 0; c < C; ++c) k::moe_decode(f.compute(c), s[c], r, dx[c][0], d[c].w, d[c].sc, &d[c].wire, parity, dout[c][0]);
      for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
      const auto want = read16(f.compute(0), dout[0][0], elems);
      for (int c = 0; c < C; ++c) {
        k::MoeWire w = d[c].wire;
        w.owner = mail, w.owner.card = c;
        f.compute(c).fill(dout[c][1], sentinel, elems);
        k::moe_decode(f.compute(c), s[c], r, dx[c][0], d[c].w, d[c].sc, &w, parity, dout[c][1]);
      }
      for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
      for (int c = 0; c < C; ++c) {
        const auto got = read16(f.compute(c), dout[c][1], elems);
        for (size_t e = 0; e < elems; ++e) {
          if (k::row_owner(static_cast<int>(e / m.s.hidden), r) == c) own_ok &= got[e] == want[e];
          else others_untouched &= got[e] == sentinel_bits;
        }
      }
      parity ^= 1;
    }
    for (int c = 0; c < C; ++c) f.compute(c).memcpy(&err[c], d[c].wire.error, 4).wait();
    bool no_timeout = std::all_of(err.begin(), err.end(), [](uint32_t e) { return e == 0; });
    for (int c = 0; c < C; ++c) {
      std::vector<uint32_t> box(sum_words);
      f.compute(c).memcpy(box.data(), mail.inbox[c], box.size() * 4).wait();
      for (uint32_t v : box) no_timeout &= v == k::kMoeMailEmpty;
      f.release(mail.inbox[c]);
    }
    std::snprintf(what, sizeof what, "%s: the combine as an owner sum, 4 / 5 / 8 / 16 / 20 / 32 rows: each card's own rows == the four-card wire's", who.c_str());
    expect(own_ok, what);
    std::snprintf(what, sizeof what, "%s: the owner sum writes no other card's rows; no mailbox timeout, inboxes empty after", who.c_str());
    expect(others_untouched && no_timeout, what);
  }
  for (int c = 0; c < C; ++c) {
    for (auto *p : dx[c]) f.release(p);
    for (auto *p : dout[c]) f.release(p);
    f.release(d[c].wire.inbox);
    f.release(d[c].wire.error);
    free_card(f, d[c]);
  }
}

// The CPU path for cold experts (MoeCpuLink + MoeCpuPool), two cards, expert TP. Every third expert is cold
// (its table entry points at a pinned host copy of the card's blob). Gates: with the path on, both cards bit-identical,
// within 2e-3 (rel L2 per row) of the cards alone (the pool sums in another order), the pool served requests and no wait
// timed out; min_cold past any layer's cold count = no request and the cards-alone bits; a recorded graph replays the
// eager bits; a stalled pool (200 ms, bound 10 ms) leaves outputs within the bound, raises the timeouts words and turns
// the on words off. Half of the cold experts form the pool's set (flag 2); each row of a 4-row step equals the
// row alone bit for bit (path off, min_cold 1, min_cold 2 per row): batch invariance.
void test_cpu(xenithium::Fabric &f, const Model &m0, const char *tag) {
  namespace sx = sycl::ext::oneapi::experimental;
  Model m = m0;
  m.s.tp = true;
  const int H = m.s.hidden, rows = 4, inputs = 3;
  const k::MoeCpuIsa isa = k::moe_cpu_detect("auto");
  if (isa == k::MoeCpuIsa::none) {
    std::printf("[SKIP] %s CPU path: this CPU has neither AVX-512 BF16 nor AVX2 + FMA\n", tag);
    return;
  }
  Card d[2] = {make_card(f, 0, m, 2), make_card(f, 1, m, 2)};
  k::MoeShape s[2] = {m.s, m.s};
  std::vector<void *> hostmem;
  k::MoeCpuLink link[2];
  std::unique_ptr<k::MoeCpuPool> pool[2];
  size_t stride = 0;
  const k::MoeBlobLayout lay = k::moe_tp_layout(H, m.s.inter, 2, &stride);
  for (int c = 0; c < 2; ++c) {
    s[c].cards = 2, s[c].card = c;
    d[c].wire.inbox = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_wire_words(s[c]) * 4));
    d[c].wire.error = static_cast<uint32_t *>(f.alloc(c, 4096));
    d[c].mem.push_back(d[c].wire.inbox), d[c].mem.push_back(d[c].wire.error);
    // pinned host copies of this card's distinct blobs; every third expert's entry points there
    auto *host = static_cast<uint8_t *>(f.alloc_host(static_cast<size_t>(m.distinct) * stride));
    hostmem.push_back(host);
    for (int i = 0; i < m.distinct; ++i) k::moe_split_tp(m.lay, H, m.s.inter, 2, c, m.ex[i].blob.data(), lay, host + static_cast<size_t>(i) * stride);
    std::vector<const uint8_t *> table(m.s.experts);
    f.compute(c).memcpy(table.data(), d[c].w.blobs, table.size() * 8).wait();
    std::vector<uint8_t> cold(std::max(m.s.experts, 4096), 0);
    // half of the cold experts (e % 6 == 0) form the pool's set (flag 2), the rest stay cold on the card
    for (int e = 0; e < m.s.experts; e += 3) cold[e] = e % 6 == 0 ? 2 : 1, table[e] = host + static_cast<size_t>(alias(m, e)) * stride;
    f.compute(c).memcpy(const_cast<const uint8_t **>(d[c].w.blobs), table.data(), table.size() * 8).wait();  // the table is ours
    d[c].w.cold = up(f, c, cold, d[c].mem);
    auto *mail = static_cast<k::MoeCpuMail *>(f.alloc_host(std::max<size_t>(4096, sizeof(k::MoeCpuMail))));
    std::memset(mail, 0, 4096);
    auto *xh = static_cast<bf16 *>(f.alloc_host(static_cast<size_t>(k::kMoeRows) * H * 2));
    auto *yh = static_cast<float *>(f.alloc_host(static_cast<size_t>(k::kMoeCpuMax) * k::kMoeRows * H * 4));
    hostmem.push_back(mail), hostmem.push_back(xh), hostmem.push_back(yh);
    auto *state = static_cast<uint32_t *>(f.alloc(c, 4096));
    d[c].mem.push_back(state);
    k::moe_cpu_state_init(f.compute(c), state).wait();
    link[c].mail = mail, link[c].x = xh, link[c].y = yh, link[c].state = state, link[c].min_cold = 1;
  }
  auto start_pools = [&](int stall_us) {
    for (int c = 0; c < 2; ++c) {
      pool[c].reset();
      k::MoeCpuShape cs;
      cs.hidden = H, cs.inter_c = m.s.inter / 2, cs.card = c, cs.cards = 2;
      cs.gu_codes = lay.gu_codes, cs.dn_codes = lay.dn_codes, cs.gu_scales = lay.gu_scales, cs.dn_scales = lay.dn_scales;
      k::MoeCpuOptions o;
      o.isa = isa, o.threads = 4, o.first_core = -1, o.stall_us = stall_us;
      pool[c] = std::make_unique<k::MoeCpuPool>(cs, o, link[c].mail, reinterpret_cast<const uint16_t *>(link[c].x), const_cast<float *>(link[c].y));
    }
  };
  start_pools(0);
  for (int c = 0; c < 2; ++c) {
    d[c].wire.peer_inbox = d[1 - c].wire.inbox;
    k::moe_wire_init(f.compute(c), s[c], d[c].wire.inbox).wait();
    f.compute(c).memset(d[c].wire.error, 0, 4).wait();
  }
  std::vector<bf16 *> dx[2];
  for (int c = 0; c < 2; ++c)
    for (int i = 0; i < inputs; ++i) {
      dx[c].push_back(static_cast<bf16 *>(f.alloc(c, static_cast<size_t>(k::kMoeRows) * H * 2)));
      d[c].mem.push_back(dx[c].back());
      const auto b = bits_of(make_x(H, 1700u + i));
      f.compute(c).memcpy(dx[c].back(), b.data(), b.size() * 2).wait();
    }
  int parity = 0;
  auto run_x = [&](bf16 *const *x, int n, bool cpu) {
    for (int c = 0; c < 2; ++c) k::moe_decode(f.compute(c), s[c], n, x[c], d[c].w, d[c].sc, &d[c].wire, parity, d[c].out, 0, 0, cpu ? &link[c] : nullptr);
    ++parity;
    for (int c = 0; c < 2; ++c) f.compute(c).wait_and_throw();
    return std::make_pair(read16(f.compute(0), d[0].out, static_cast<size_t>(n) * H), read16(f.compute(1), d[1].out, static_cast<size_t>(n) * H));
  };
  auto run = [&](int i, bool cpu) {
    bf16 *x[2] = {dx[0][i], dx[1][i]};
    return run_x(x, rows, cpu);
  };
  auto rel = [&](const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    double worst = 0;
    for (size_t r0 = 0; r0 < a.size(); r0 += H) {
      double num = 0, den = 0;
      for (int c = 0; c < H; ++c) {
        const double va = k::bf16_value(a[r0 + c]), vb = k::bf16_value(b[r0 + c]);
        num += (va - vb) * (va - vb), den += vb * vb;
      }
      worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
    }
    return worst;
  };
  bool both = true, close = true;
  double worst = 0;
  std::vector<std::vector<uint16_t>> alone(inputs), with(inputs);
  for (int i = 0; i < inputs; ++i) {
    alone[i] = run(i, false).first;
    const auto pr = run(i, true);
    with[i] = pr.first;
    both &= pr.first == pr.second;
    const double w = rel(pr.first, alone[i]);
    worst = std::max(worst, w);
    close &= w <= 2e-3;
  }
  const uint64_t served = pool[0]->served() + pool[1]->served();
  char what[260];
  std::snprintf(what, sizeof what, "%s CPU path (%s): both cards bit-identical", tag, k::moe_cpu_isa_name(isa));
  expect(both, what);
  std::snprintf(what, sizeof what, "%s CPU path: within 2e-3 of the cards alone (worst rel L2 per row %.2e); %llu pool requests, timeouts %u / %u", tag, worst,
                static_cast<unsigned long long>(served), link[0].mail->timeouts, link[1].mail->timeouts);
  expect(close && served > 0 && link[0].mail->timeouts == 0 && link[1].mail->timeouts == 0, what);
  // min_cold past any layer's count: no request, the cards-alone bits
  {
    for (auto &l : link) l.min_cold = 1000;
    const uint64_t s0 = pool[0]->served() + pool[1]->served();
    const auto pr = run(0, true);
    std::snprintf(what, sizeof what, "%s CPU path, min_cold past the cold count: no request, == the cards alone bit for bit", tag);
    expect(pr.first == alone[0] && pool[0]->served() + pool[1]->served() == s0, what);
    for (auto &l : link) l.min_cold = 1;
  }
  // Batch invariance: each row of a 4-row step == the same row run alone (1 row), bit for bit, with the path
  // off (control), on with min_cold 1 and on with min_cold 2 (rows below the count keep their set experts on the card)
  {
    bf16 *one[2] = {};
    for (int c = 0; c < 2; ++c) {
      one[c] = static_cast<bf16 *>(f.alloc(c, static_cast<size_t>(k::kMoeRows) * H * 2));
      d[c].mem.push_back(one[c]);
    }
    for (int mc : {0, 1, 2}) {
      for (auto &l : link) l.min_cold = std::max(1, mc);
      const uint64_t s0 = pool[0]->served() + pool[1]->served();
      bool inv = true;
      for (int i = 0; i < inputs; ++i) {
        const auto all = run(i, mc > 0).first;
        for (int r = 0; r < rows; ++r) {
          for (int c = 0; c < 2; ++c) f.compute(c).memcpy(one[c], dx[c][i] + static_cast<size_t>(r) * H, static_cast<size_t>(H) * 2).wait();
          const auto pr = run_x(one, 1, mc > 0);
          inv &= pr.first == pr.second && std::equal(pr.first.begin(), pr.first.end(), all.begin() + static_cast<std::ptrdiff_t>(r) * H);
        }
      }
      std::snprintf(what, sizeof what, "%s CPU path %s: each row of a 4-row step == the row alone, bit for bit (%llu pool requests)", tag,
                    mc == 0 ? "off (control)" : mc == 1 ? "on, min_cold 1" : "on, min_cold 2 per row",
                    static_cast<unsigned long long>(pool[0]->served() + pool[1]->served() - s0));
      expect(inv && (mc == 0 || pool[0]->served() + pool[1]->served() > s0), what);
    }
    // a 16-row step (the 16-row shape, two cards, the pool's experts) likewise,
    // and a 32-row step (the 32-row shape: 320 pool experts at most)
    for (int sr : {16, 32})
    for (int mc : {0, 1}) {
      for (auto &l : link) l.min_cold = 1;
      const uint64_t s0 = pool[0]->served() + pool[1]->served();
      bool inv = true;
      for (int i = 0; i < inputs; ++i) {
        bf16 *x[2] = {dx[0][i], dx[1][i]};
        const auto all = run_x(x, sr, mc > 0);
        inv &= all.first == all.second;
        for (int r = 0; r < sr; ++r) {
          for (int c = 0; c < 2; ++c) f.compute(c).memcpy(one[c], dx[c][i] + static_cast<size_t>(r) * H, static_cast<size_t>(H) * 2).wait();
          const auto pr = run_x(one, 1, mc > 0);
          inv &= pr.first == pr.second && std::equal(pr.first.begin(), pr.first.end(), all.first.begin() + static_cast<std::ptrdiff_t>(r) * H);
        }
      }
      std::snprintf(what, sizeof what, "%s CPU path %s: each row of a %d-row step == the row alone, bit for bit (%llu pool requests)", tag,
                    mc == 0 ? "off (control)" : "on, min_cold 1", sr, static_cast<unsigned long long>(pool[0]->served() + pool[1]->served() - s0));
      expect(inv && (mc == 0 || pool[0]->served() + pool[1]->served() > s0), what);
    }
    for (auto &l : link) l.min_cold = 1;
  }
  // a recorded graph replays the eager bits (the request number differs per replay)
  {
    std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
    for (int c = 0; c < 2; ++c) g.emplace_back(f.compute(c).get_context(), f.compute(c).get_device());
    for (int c = 0; c < 2; ++c) g[c].begin_recording(f.compute(c));
    for (int c = 0; c < 2; ++c) k::moe_decode(f.compute(c), s[c], rows, dx[c][1], d[c].w, d[c].sc, &d[c].wire, parity, d[c].out, 0, 0, &link[c]);
    for (int c = 0; c < 2; ++c) g[c].end_recording();
    ++parity;
    std::vector<sx::command_graph<sx::graph_state::executable>> ex;
    for (int c = 0; c < 2; ++c) ex.push_back(g[c].finalize());
    bool same = true;
    for (int rep = 0; rep < 3; ++rep) {
      for (int c = 0; c < 2; ++c) f.compute(c).ext_oneapi_graph(ex[c]);
      for (int c = 0; c < 2; ++c) f.compute(c).wait_and_throw();
      same &= read16(f.compute(0), d[0].out, static_cast<size_t>(rows) * H) == with[1];
      // the recorded wire parity alternates with nothing else in between: replays reuse the recorded one (two cards, in order)
    }
    std::snprintf(what, sizeof what, "%s CPU path: 3 graph replays == the eager step bit for bit", tag);
    expect(same, what);
  }
  // the timeout fallback
  {
    start_pools(200000);
    for (auto &l : link) l.wait_cycles = 2800000ull * 10;
    const auto pr = run(2, true);
    uint32_t st[2][16];
    for (int c = 0; c < 2; ++c) f.compute(c).memcpy(st[c], link[c].state, sizeof st[c]).wait();
    const double w = rel(pr.first, alone[2]);
    std::snprintf(what, sizeof what, "%s CPU path, pool stalled 200 ms (bound 10 ms): within 2e-3 of the cards alone (%.2e), timeouts %u / %u, on words %u / %u",
                  tag, w, link[0].mail->timeouts, link[1].mail->timeouts, st[0][1], st[1][1]);
    expect(w <= 2e-3 && pr.first == pr.second && link[0].mail->timeouts && link[1].mail->timeouts && st[0][1] == 0 && st[1][1] == 0, what);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the stalled pools answer before they stop
  }
  for (auto &p : pool) p.reset();
  for (void *p : hostmem) f.release(p);
  for (int c = 0; c < 2; ++c) free_card(f, d[c]);
}

// moe_rows (prefill, expert TP) against moe_decode on 4-row slices of the same rows (the
// decode block is gated against the reference above). Routing must be equal (ids exact, weights
// 1e-5); outputs differ by fp32 summation order only (the shared term is split by rows, not
// columns), so: within the step bound of the decode-path output. Also: both cards bit-identical,
// run-to-run bit-identical (the expert list order comes from atomics), no mailbox error.
// moe_rows rounds each card's partial y_e to bf16, so the decode path here runs with
// MoeShape:down_bf16 (the same rounding point; the default decode path keeps fp32 y_e).
// moe_rows' scratch on card c for M rows (released with the card's other buffers).
k::MoeRowsScratch rows_scratch(xenithium::Fabric &f, int c, const k::MoeShape &s, int M, int router_n, std::vector<void *> &mem) {
  auto dev = [&](size_t bytes) {
    void *p = f.alloc(c, std::max<size_t>(bytes, 64));
    mem.push_back(p);
    return p;
  };
  const auto z = k::moe_rows_sizes(s, M, router_n);
  k::MoeRowsScratch rs;
  rs.logits = static_cast<float *>(dev(z.logits * 4));
  rs.ids = static_cast<int32_t *>(dev(z.ids * 4));
  rs.topw = static_cast<float *>(dev(z.ids * 4));
  rs.sg = static_cast<float *>(dev(z.sg * 4));
  rs.slot = static_cast<int32_t *>(dev(z.ids * 4));
  rs.row = static_cast<int32_t *>(dev(z.row * 4));
  rs.seg = static_cast<int32_t *>(dev(z.seg * 4));
  rs.seg_n = static_cast<int32_t *>(dev(64));
  rs.h = static_cast<bf16 *>(dev(z.h * 2));
  rs.y = static_cast<k::moe_bf16 *>(dev(z.y * 2));
  rs.sgu = static_cast<float *>(dev(z.sgu * 4));
  rs.sh = static_cast<bf16 *>(dev(z.sh * 2));
  rs.sy = static_cast<float *>(dev(z.sy * 4));
  rs.xq = static_cast<int8_t *>(dev(z.xq));
  rs.xs = static_cast<float *>(dev(z.xs * 4));
  rs.hq = static_cast<int8_t *>(dev(z.hq));
  rs.hs = static_cast<float *>(dev(z.hs * 4));
  rs.xr = static_cast<bf16 *>(dev(z.xr * 2));
  return rs;
}

// The block list that switches moe_rows' W4A16 gate_up / down to the work-group-per-block
// kernels (MoeRowsScratch:blk), on card c.
void rows_blocks(xenithium::Fabric &f, int c, const k::MoeShape &s, int M, int router_n, std::vector<void *> &mem, k::MoeRowsScratch &rs) {
  const auto z = k::moe_rows_sizes(s, M, router_n);
  rs.blk = static_cast<int32_t *>(f.alloc(c, std::max<size_t>(z.blk * 4, 64)));
  mem.push_back(rs.blk);
  rs.blk_n = static_cast<int32_t *>(f.alloc(c, 64));
  mem.push_back(rs.blk_n);
}

// The routing by row owner (moe_rows_route on each card's own rows of a split call, then shared) against what the
// last W4A8 moe_rows call on the whole call left in rs: ids, top-k weights, shared gates, int8 x and its scales, bit for
// bit. The routing steps run on one card here: their bits do not depend on the card.
bool route_by_owner_same(xenithium::Fabric &f, const k::MoeShape &s, int M, const bf16 *x, const k::MoeWeights &w, const k::MoeRowsScratch &rs) {
  auto &q = f.compute(0);
  const auto z = k::moe_rows_sizes(s, M, w.router.n);
  const size_t A = static_cast<size_t>(M) * s.topk, xs_n = static_cast<size_t>(M) * (s.hidden / k::kMoeXGroup);
  auto read = [&](const auto *p, size_t n) {
    std::vector<std::remove_const_t<std::remove_pointer_t<decltype(p)>>> v(n);
    q.memcpy(v.data(), p, n * sizeof(v[0])).wait();
    return v;
  };
  const auto ids = read(rs.ids, A);
  const auto xq = read(rs.xq, static_cast<size_t>(M) * s.hidden);
  const auto topw = read(rs.topw, A), sg = read(rs.sg, static_cast<size_t>(M)), xs = read(rs.xs, xs_n);  // (all float)
  q.memset(rs.ids, 0xff, z.ids * 4);
  q.memset(rs.topw, 0xff, z.ids * 4);
  q.memset(rs.sg, 0xff, z.sg * 4);
  q.memset(rs.xq, 0x55, z.xq);
  q.memset(rs.xs, 0xff, z.xs * 4).wait();
  for (int c = 0; c < k::kWireCardsMax; ++c) {
    int lo = 0, hi = 0;
    k::pair_split_owner_rows(c, M, &lo, &hi, k::kWireCardsMax);
    k::moe_rows_route(q, s, lo, hi, x, w, rs, true);
  }
  q.wait_and_throw();
  return read(rs.ids, A) == ids && read(rs.topw, A) == topw && read(rs.sg, static_cast<size_t>(M)) == sg &&
         read(rs.xq, static_cast<size_t>(M) * s.hidden) == xq && read(rs.xs, xs_n) == xs;
}

// moe_rows on four cards (32-k quarter down pieces, the four-card MoeRowsWire) against
// moe_decode on four cards over 4-row slices of the same rows (the decode's grouping of the sum differs: the bound of
// test_rows), every card's rows identical, a second run identical; W4A8 on the block kernels; W4A8 without the block
// list and rot refused at 160 columns per card.
void test_rows_quad(xenithium::Fabric &f, const Model &m0, const char *tag, int M) {
  Model m = m0;
  m.s.tp = true;
  m.s.down_bf16 = true;
  const int C = k::kWireCardsMax, H = m.s.hidden;
  std::vector<Card> d;
  std::vector<k::MoeShape> s(C, m.s);
  std::vector<k::MoeRowsScratch> rs(C);
  std::vector<k::MoeRowsWire> rw(C);
  std::vector<bf16 *> x(C), orow(C), odec(C);
  std::vector<std::vector<void *>> mem(C);
  std::mt19937 rng(5151u + M);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> xv(static_cast<size_t>(M + 4) * H);
  for (auto &v : xv) v = k::bf16_value(k::bf16_bits(nd(rng)));
  const auto xb = bits_of(xv);
  k::WireQuad dq, rq;
  for (int c = 0; c < C; ++c) {
    d.push_back(make_card(f, c, m, C));
    s[c].cards = C, s[c].card = c;
    auto dev = [&](size_t bytes) {
      void *p = f.alloc(c, std::max<size_t>(bytes, 64));
      mem[c].push_back(p);
      return p;
    };
    dq.inbox[c] = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_wire_words(s[c]) * 4));
    mem[c].push_back(dq.inbox[c]);
    k::moe_wire_init(f.compute(c), s[c], dq.inbox[c]).wait();
    d[c].wire.error = static_cast<uint32_t *>(dev(4096));
    f.compute(c).memset(d[c].wire.error, 0, 4).wait();
    rq.inbox[c] = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_rows_wire_words(s[c], M) * 4));
    mem[c].push_back(rq.inbox[c]);
    k::moe_rows_wire_init(f.compute(c), s[c], M, rq.inbox[c]).wait();
    rs[c] = rows_scratch(f, c, s[c], M, m.router_n, mem[c]);
    x[c] = static_cast<bf16 *>(dev(xb.size() * 2));
    f.compute(c).memcpy(x[c], xb.data(), xb.size() * 2).wait();
    orow[c] = static_cast<bf16 *>(dev(xb.size() * 2));
    odec[c] = static_cast<bf16 *>(dev(xb.size() * 2));
  }
  for (int c = 0; c < C; ++c) {
    d[c].wire.quad = dq, d[c].wire.quad.card = c;
    rw[c].error = d[c].wire.error, rw[c].rows_max = M, rw[c].quad = rq, rw[c].quad.card = c;
  }
  for (int sl = 0; sl * 4 < M; ++sl) {
    const int rows = std::min(4, M - 4 * sl);
    for (int c = 0; c < C; ++c)
      k::moe_decode(f.compute(c), s[c], rows, x[c] + static_cast<size_t>(sl) * 4 * H, d[c].w, d[c].sc, &d[c].wire, sl, odec[c] + static_cast<size_t>(sl) * 4 * H);
  }
  for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
  auto run_rows = [&](int parity) {
    for (int c = 0; c < C; ++c) k::moe_rows(f.compute(c), s[c], M, x[c], d[c].w, rs[c], &rw[c], parity, orow[c]);
    for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
    return read16(f.compute(0), orow[0], static_cast<size_t>(M) * H);
  };
  const auto a = run_rows(0);
  // the work-group-per-block gate_up / down give the same bits on every card
  bool blk_same = k::moe_rows_blk_shape(s[0]);
  if (blk_same) {
    for (int c = 0; c < C; ++c) rows_blocks(f, c, s[c], M, m.router_n, mem[c], rs[c]);
    const auto ab = run_rows(1);
    for (int c = 0; c < C; ++c) blk_same &= read16(f.compute(c), orow[c], a.size()) == a;
    blk_same &= ab == a;
    for (int c = 0; c < C; ++c) rs[c].blk = nullptr, rs[c].blk_n = nullptr;
  }
  expect(blk_same, std::string(tag) + ", moe_rows " + std::to_string(M) + " rows: the work-group-per-block gate_up / down == the segment kernels, bit for bit");
  const int next_parity = k::moe_rows_blk_shape(s[0]) ? 0 : 1;  // the wire's parity alternates per call
  bool same = true, dec_same = true;
  for (int c = 1; c < C; ++c) same &= read16(f.compute(c), orow[c], a.size()) == a;
  const auto dec16 = read16(f.compute(0), odec[0], static_cast<size_t>(M) * H);
  for (int c = 1; c < C; ++c) dec_same &= read16(f.compute(c), odec[c], dec16.size()) == dec16;
  std::vector<float> dec(dec16.size());
  for (size_t i = 0; i < dec.size(); ++i) dec[i] = k::bf16_value(dec16[i]);
  const Cmp cmp = compare(a, dec, H);
  const bool again = run_rows(next_parity) == a;
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
    err |= e;
  }
  // MoeShape:w4_bias (the block gate_up's widen offset outside the DPAS operand, a numerics choice) against the exact
  // blocks: its one-ulp h flips (~0.2 % of h) reach every output column of their row through down and the sums, so the
  // bound is the four-card rows' two ulps with up to 5 % of elements moved; every card the same rows
  bool bias_ok = k::moe_rows_blk_shape(s[0]);
  std::string bias_text = "no block shape";
  if (bias_ok) {
    for (int c = 0; c < C; ++c) rows_blocks(f, c, s[c], M, m.router_n, mem[c], rs[c]), s[c].w4_bias = true;
    const auto aw = run_rows(next_parity ^ 1);
    for (int c = 1; c < C; ++c) bias_ok &= read16(f.compute(c), orow[c], aw.size()) == aw;
    std::vector<float> af(a.size());
    for (size_t i = 0; i < af.size(); ++i) af[i] = k::bf16_value(a[i]);
    const Cmp wc = compare(aw, af, H);
    bias_ok &= wc.scaled <= 2.0 / 128 && wc.off * 20 <= wc.n;
    bias_text = cmp_text(wc);
    for (int c = 0; c < C; ++c) s[c].w4_bias = false, rs[c].blk = nullptr, rs[c].blk_n = nullptr;
  }
  expect(bias_ok, std::string(tag) + " four cards, rows " + std::to_string(M) + ": the block gate_up with w4_bias vs the exact rows (<= 2 ulps, <= 5 %) " + bias_text +
                      "; every card the same rows");
  // W4A8 on the block kernels (int8 x per 128-k group, int8 h per segment of the card's k): within int8 rounding of
  // the W4A16 rows, every card the same rows, a rerun identical
  bool a8_ok = k::moe_rows_blk8_shape(s[0]);
  bool route_ok = !a8_ok;  // (nothing to compare without the W4A8 blocks)
  double a8_rel = -1;
  if (a8_ok) {
    for (int c = 0; c < C; ++c) rows_blocks(f, c, s[c], M, m.router_n, mem[c], rs[c]);
    auto run8 = [&](int parity) {
      for (int c = 0; c < C; ++c) k::moe_rows(f.compute(c), s[c], M, x[c], d[c].w, rs[c], &rw[c], parity, orow[c], true);
      for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
      return read16(f.compute(0), orow[0], static_cast<size_t>(M) * H);
    };
    const auto q8 = run8(next_parity);  // (the w4_bias run above took next_parity ^ 1)
    for (int c = 1; c < C; ++c) a8_ok &= read16(f.compute(c), orow[c], q8.size()) == q8;
    a8_ok &= run8(next_parity ^ 1) == q8;
    double se = 0, sr = 0;
    for (size_t i = 0; i < q8.size(); ++i) {
      const double dv = k::bf16_value(q8[i]) - k::bf16_value(a[i]);
      se += dv * dv;
      sr += static_cast<double>(k::bf16_value(a[i])) * k::bf16_value(a[i]);
    }
    a8_rel = std::sqrt(se / std::max(sr, 1e-30));
    a8_ok &= a8_rel < 2e-2;
    route_ok = route_by_owner_same(f, s[0], M, x[0], d[0].w, rs[0]);
    for (int c = 0; c < C; ++c) rs[c].blk = nullptr, rs[c].blk_n = nullptr;
  }
  expect(route_ok, std::string(tag) + " four cards, rows " + std::to_string(M) +
                       ": moe_rows_route over each card's own rows == the whole call's routing and int8 x, bit for bit");
  {
    char w8[220];
    std::snprintf(w8, sizeof w8, "%s four cards, rows %d: W4A8 blocks vs W4A16 relative rms %.2e (int8 rounding: < 2e-2); every card the same rows; a rerun identical", tag,
                  M, a8_rel);
    expect(a8_ok, w8);
  }
  auto refused = [&](bool a8, bool rot) {
    k::MoeShape sr = s[0];
    sr.rot = rot;
    try {
      k::moe_rows(f.compute(0), sr, M, x[0], d[0].w, rs[0], &rw[0], 0, orow[0], a8);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  // The two paths group a row's sum differently (decode: each card's experts + its shared slice, then the four-card sum;
  // rows: the experts' four-card sum, then the shared term) and each card's y_e is rounded to bf16 (down_bf16) before
  // it: four rounded partials per expert where two cards have two. So four cards get twice the ulps of test_rows' bound
  // (2 bf16 ulps of max(|ref|, rms), <= 1 % of elements); the decode path itself holds the two-ulp bound against the
  // host reference (test_cards).
  const bool close = cmp.scaled <= 2.0 / 128 && cmp.off * 100 <= cmp.n;
  char what[240];
  std::snprintf(what, sizeof what, "%s four cards, rows %d: moe_rows vs moe_decode's 4-row slices (four cards, <= 4 ulps) %s; every card the same rows, decode too; a rerun identical; no timeout",
                tag, M, cmp_text(cmp).c_str());
  expect(close && same && dec_same && again && err == 0, what);
  std::snprintf(what, sizeof what, "%s four cards: moe_rows refuses W4A8 without the block list and rot at 160 columns per card", tag);
  expect(refused(true, false) && refused(false, true), what);
  for (int c = 0; c < C; ++c) {
    for (void *p : mem[c]) f.release(p);
    free_card(f, d[c]);
  }
}

void test_rows(xenithium::Fabric &f, const Model &m0, const char *tag, int M) {
  Model m = m0;
  m.s.tp = true;
  m.s.down_bf16 = true;
  const int C = f.size() > 1 ? 2 : 1, H = m.s.hidden, K = m.s.topk;
  Card d[2];
  k::MoeShape s[2] = {m.s, m.s};
  k::MoeRowsScratch rs[2];
  k::MoeRowsWire rw[2];
  bf16 *x[2] = {}, *orow[2] = {}, *odec[2] = {};
  std::vector<void *> mem[2];
  std::mt19937 rng(4242u + M);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> xv(static_cast<size_t>(M + 4) * H);
  for (auto &v : xv) v = k::bf16_value(k::bf16_bits(nd(rng)));
  const auto xb = bits_of(xv);
  for (int c = 0; c < C; ++c) {
    d[c] = make_card(f, c, m, C);
    s[c].cards = C;
    s[c].card = c;
    auto dev = [&](size_t bytes) {
      void *p = f.alloc(c, std::max<size_t>(bytes, 64));
      mem[c].push_back(p);
      return p;
    };
    if (C > 1) {
      d[c].wire.inbox = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_wire_words(s[c]) * 4));
      mem[c].push_back(d[c].wire.inbox);
      d[c].wire.error = static_cast<uint32_t *>(dev(4096));
      k::moe_wire_init(f.compute(c), s[c], d[c].wire.inbox).wait();
      f.compute(c).memset(d[c].wire.error, 0, 4).wait();
      rw[c].rows_max = M;
      rw[c].inbox = static_cast<uint32_t *>(f.alloc_mailbox(c, k::moe_rows_wire_words(s[c], M) * 4));
      mem[c].push_back(rw[c].inbox);
      rw[c].error = d[c].wire.error;
      k::moe_rows_wire_init(f.compute(c), s[c], M, rw[c].inbox).wait();
    }
    rs[c] = rows_scratch(f, c, s[c], M, m.router_n, mem[c]);
    x[c] = static_cast<bf16 *>(dev(xb.size() * 2));
    f.compute(c).memcpy(x[c], xb.data(), xb.size() * 2).wait();
    orow[c] = static_cast<bf16 *>(dev(xb.size() * 2));
    odec[c] = static_cast<bf16 *>(dev(xb.size() * 2));
  }
  for (int c = 0; c < C; ++c) {
    d[c].wire.peer_inbox = C > 1 ? d[1 - c].wire.inbox : nullptr;
    rw[c].peer_inbox = C > 1 ? rw[1 - c].inbox : nullptr;
  }
  // decode slices (waits per slice to read the routing record)
  std::vector<int32_t> dids(static_cast<size_t>(M) * K);
  std::vector<float> dtw(dids.size());
  for (int sl = 0; sl * 4 < M; ++sl) {
    const int rows = std::min(4, M - 4 * sl);
    for (int c = 0; c < C; ++c)
      k::moe_decode(f.compute(c), s[c], rows, x[c] + static_cast<size_t>(sl) * 4 * H, d[c].w, d[c].sc, C > 1 ? &d[c].wire : nullptr, sl,
                    odec[c] + static_cast<size_t>(sl) * 4 * H);
    for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
    k::MoeRoute rt;
    f.compute(0).memcpy(&rt, d[0].sc.route, sizeof rt).wait();
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < K; ++j) {
        dids[static_cast<size_t>(4 * sl + r) * K + j] = rt.ids[r][j];
        dtw[static_cast<size_t>(4 * sl + r) * K + j] = rt.topw[r][j];
      }
  }
  auto run_rows = [&](int parity, int m = 0, int r0 = 0, bool a8 = false, bool fuse = true) {
    if (m == 0) m = M;
    for (int c = 0; c < C; ++c)
      k::moe_rows(f.compute(c), s[c], m, x[c] + static_cast<size_t>(r0) * H, d[c].w, rs[c], C > 1 ? &rw[c] : nullptr, parity, orow[c] + static_cast<size_t>(r0) * H,
                  a8, fuse);
    for (int c = 0; c < C; ++c) f.compute(c).wait_and_throw();
    return read16(f.compute(0), orow[0], static_cast<size_t>(M) * H);
  };
  const auto a = run_rows(0);
  std::vector<int32_t> rids(dids.size());
  std::vector<float> rtw(dids.size());
  f.compute(0).memcpy(rids.data(), rs[0].ids, rids.size() * 4).wait();
  f.compute(0).memcpy(rtw.data(), rs[0].topw, rtw.size() * 4).wait();
  bool route = rids == dids;
  for (size_t i = 0; i < rtw.size(); ++i) route &= std::fabs(rtw[i] - dtw[i]) <= 1e-5f;
  const auto dec16 = read16(f.compute(0), odec[0], static_cast<size_t>(M) * H);
  std::vector<float> dec(dec16.size());
  for (size_t i = 0; i < dec.size(); ++i) dec[i] = k::bf16_value(dec16[i]);
  const Cmp cmp = compare(a, dec, H);
  bool same = true;
  if (C > 1) same = read16(f.compute(1), orow[1], a.size()) == a;
  const auto again = run_rows(1);
  // the work-group-per-block gate_up / down == the segment kernels, bit for bit (two calls:
  // the wire's parity keeps alternating)
  if (k::moe_rows_blk_shape(s[0])) {
    for (int c = 0; c < C; ++c) rows_blocks(f, c, s[c], M, m.router_n, mem[c], rs[c]);
    bool blk_same = run_rows(0) == a;
    if (C > 1) blk_same &= read16(f.compute(1), orow[1], a.size()) == a;
    blk_same &= run_rows(1) == a;
    for (int c = 0; c < C; ++c) rs[c].blk = nullptr, rs[c].blk_n = nullptr;
    expect(blk_same, std::string(tag) + " (" + std::to_string(M) + " rows): the work-group-per-block gate_up / down == the segment kernels, bit for bit");
  }
  // The fused launches (x quantiser in the router's launch, swiglu in topk's) == the
  // launches, bit for bit, W4A16 and W4A8 (outputs poisoned before each run).
  bool fuse_same = true;
  auto poison = [&]() {  // outputs and the scratch the fused jobs write (a skipped job must show)
    for (int c = 0; c < C; ++c) {
      const auto z = k::moe_rows_sizes(s[c], M, m.router_n);
      f.compute(c).memset(orow[c], 0xff, static_cast<size_t>(M) * H * 2);
      f.compute(c).memset(rs[c].sh, 0xff, z.sh * 2);
      f.compute(c).memset(rs[c].xq, 0x7f, z.xq);
      f.compute(c).memset(rs[c].xs, 0xff, z.xs * 4).wait();
    }
  };
  for (const bool a8 : {false, true}) {
    poison();
    const auto fu = run_rows(0, 0, 0, a8, true);
    poison();
    const auto nf = run_rows(1, 0, 0, a8, false);
    fuse_same &= fu == nf && (a8 || fu == a);
  }
  {
    char w_[220];
    std::snprintf(w_, sizeof w_, "%s rows %d: fused launches (quantiser + swiglu, round 43) == unfused, bit for bit (W4A16, W4A8)", tag, M);
    expect(fuse_same, w_);
  }
  // The multi-segment cold experts staged in VRAM == read in place, bit for
  // bit (W4A16, W4A8). Every expert flagged cold; the stage poisoned first; the staged count must be
  // the number of experts with >= 2 segments (capped), so a skipped staging cannot pass.
  {
    bool stage_same = true, stage_count = true;
    int staged0 = 0;
    for (const int cap : {3, m.s.experts}) {
      k::MoeRowsScratch st[2] = {rs[0], rs[1]};
      std::vector<void *> smem[2];
      for (int c = 0; c < C; ++c) {
        size_t sstride = 0;
        k::moe_tp_layout(s[c].hidden, s[c].inter, C, &sstride);
        auto sdev = [&](size_t bytes) {
          void *p = f.alloc(c, std::max<size_t>(bytes, 4096));
          smem[c].push_back(p);
          return p;
        };
        st[c].stage_cap = cap;
        st[c].stage_min_rows = 0;
        st[c].stage_stride = sstride;
        st[c].stage = static_cast<uint8_t *>(sdev(sstride * static_cast<size_t>(cap)));
        st[c].tab2 = static_cast<const uint8_t **>(sdev(static_cast<size_t>(s[c].experts) * 8));
        st[c].stage_list = static_cast<int32_t *>(sdev(static_cast<size_t>(cap + 1) * 4));
        f.compute(c).memset(st[c].stage, 0x77, sstride * static_cast<size_t>(cap)).wait();
        auto *cold = static_cast<uint8_t *>(sdev(static_cast<size_t>(s[c].experts)));
        f.compute(c).memset(cold, 1, static_cast<size_t>(s[c].experts)).wait();
        d[c].w.cold = cold;
      }
      for (const bool a8 : {false, true}) {
        poison();
        const auto ref = run_rows(0, 0, 0, a8, true);
        for (int c = 0; c < C; ++c) std::swap(rs[c], st[c]);
        poison();
        const auto got = run_rows(1, 0, 0, a8, true);
        for (int c = 0; c < C; ++c) std::swap(rs[c], st[c]);
        stage_same &= got == ref;
      }
      // expected staged count from the routing
      std::vector<int32_t> ids(static_cast<size_t>(M) * K);
      f.compute(0).memcpy(ids.data(), rs[0].ids, ids.size() * 4).wait();
      std::vector<int> cnt(static_cast<size_t>(m.s.experts), 0);
      for (int32_t e : ids) ++cnt[static_cast<size_t>(e)];
      int want = 0;
      for (int v : cnt) want += (v + k::kMoeSegRows - 1) / k::kMoeSegRows >= 2;
      want = M > k::kMoeSegRows ? std::min(want, cap) : 0;
      int32_t got_n = 0;
      if (M > k::kMoeSegRows) f.compute(0).memcpy(&got_n, st[0].stage_list, 4).wait();
      stage_count &= got_n == want;
      if (cap == m.s.experts) staged0 = got_n;
      for (int c = 0; c < C; ++c) {
        d[c].w.cold = nullptr;
        for (void *p : smem[c]) f.release(p);
      }
    }
    char w_[220];
    std::snprintf(w_, sizeof w_, "%s rows %d: cold experts staged in VRAM == read in place, bit for bit (W4A16, W4A8; %d staged, caps 3 / all)", tag, M, staged0);
    expect(stage_same && stage_count, w_);
  }
  // The same rows in calls of other sizes (M split at a quarter / at 4-row slices) give the
  // same bits: no row's arithmetic depends on m or on its place in the call.
  bool split_same = true;
  for (int cut : {M / 4, 4 * ((M + 7) / 8)}) {
    if (cut <= 0 || cut >= M) continue;
    run_rows(0, cut, 0);
    const auto b = run_rows(1, M - cut, cut);
    split_same &= b == a;
  }
  uint32_t err = 0;
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    if (C > 1) f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
    err |= e;
  }
  char what[300];
  std::snprintf(what, sizeof what, "%s rows %d (%d card%s): routing == the decode path's (ids exact, weights 1e-5)", tag, M, C, C > 1 ? "s" : "");
  expect(route, what);
  std::snprintf(what, sizeof what, "%s rows %d: out vs the decode path on 4-row slices %s", tag, M, cmp_text(cmp).c_str());
  expect(cmp_ok(cmp), what);
  std::snprintf(what, sizeof what, "%s rows %d: both cards bit-identical, run-to-run bit-identical, no mailbox error", tag, M);
  expect(same && again == a && err == 0, what);
  std::snprintf(what, sizeof what, "%s rows %d: split into two calls (at M / 4, at about M / 2) == one call, bit for bit", tag, M);
  if (M >= 8) expect(split_same, what);

  // Split mode (combine leaves each card's fp32 sum, pair_split_rows finishes rows over the
  // two cards with the shared term) == the mailbox combine, bit for bit, on both cards; three calls on
  // the same buffers with fresh sequence numbers (a stale line from an earlier call would show).
  if (C > 1) {
    k::PairSplit sp[2];
    for (int c = 0; c < 2; ++c) {
      rs[c].part = static_cast<float *>(f.alloc(c, static_cast<size_t>(M) * H * 4));
      mem[c].push_back(rs[c].part);
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
    uint32_t seq = 0;
    // Call 1 sends from inside moe_rows on a side queue (the overlap), call 2 the whole split
    // on the side queues, calls 0 / 3 on the compute queues; call 4 as call 1 with the wait for
    // the peer's rows deferred (pair_split_finish without it, then pair_split_wait per card).
    sycl::property_list sprops{sycl::property::queue::in_order()};
    sycl::queue sideq[2] = {sycl::queue(f.card_context(0), f.device(0), sprops), sycl::queue(f.card_context(1), f.device(1), sprops)};
    sycl::queue *side[2] = {&sideq[0], &sideq[1]};
    for (int call = 0; call < 5; ++call) {
      for (int c = 0; c < 2; ++c) f.compute(c).memset(orow[c], 0, static_cast<size_t>(M) * H * 2).wait();
      sycl::queue *qs[2] = {&f.compute(0), &f.compute(1)};
      const float *part[2];
      bf16 *outp[2];
      k::PairSplitAdd ad[2];
      k::PairSplitSend send[2];
      ++seq;
      for (int c = 0; c < 2; ++c) {
        k::MoeRowsWire w = rw[c];
        w.split = true;
        if (call == 1 || call == 4) {
          send[c] = {side[c], &sp[c], seq};
          w.send = &send[c];
        }
        k::moe_rows(f.compute(c), s[c], M, x[c], d[c].w, rs[c], &w, call & 1, orow[c]);
        part[c] = rs[c].part;
        outp[c] = orow[c];
        ad[c] = k::PairSplitAdd{rs[c].sy, rs[c].sg};
      }
      if (call == 1) {
        k::pair_split_finish(qs, side, M, H, part, outp, sp, seq, ad);
      } else if (call == 4) {
        k::pair_split_finish(qs, side, M, H, part, outp, sp, seq, ad, k::SplitRows::deferred);
        for (int c = 0; c < 2; ++c) k::pair_split_wait(*qs[c], sp[c], seq);
      } else {
        k::pair_split_rows(qs, M, H, part, outp, sp, seq, ad, call == 2 ? side : nullptr);
      }
      for (int c = 0; c < 2; ++c) {
        f.compute(c).wait_and_throw();
        sideq[c].wait_and_throw();
      }
      split_ok &= read16(f.compute(0), orow[0], a.size()) == a && read16(f.compute(1), orow[1], a.size()) == a;
    }
    uint32_t e2 = 0;
    for (int c = 0; c < 2; ++c) {
      uint32_t e = 0;
      f.compute(c).memcpy(&e, d[c].wire.error, 4).wait();
      e2 |= e;
    }
    std::snprintf(what, sizeof what, "%s rows %d: split two-card sum (pair_split_rows + shared term) == the mailbox combine on both cards, 5 calls (compute queue, send overlapped on a side queue, side queue, send + deferred wait), no timeout", tag, M);
    expect(split_ok && e2 == 0, what);
    for (int c = 0; c < 2; ++c) rs[c].part = nullptr;
  }

  // W4A8 (a8). (1) The quantiser == a CPU reference, bit for bit. (2) The output within
  // int8 rounding of the W4A16 output. (3) Both cards, run to run and calls of any size bit-identical.
  // (4) On x whose groups quantise losslessly (integers, absmax 127 x 2^e), h == the W4A16 path's h
  // bit for bit (int32 sums are exact, so only the activation rounding separates the two paths).
  {
    const int I = s[0].inter_local(), G = H / 128;
    const auto q8 = run_rows(0, 0, 0, true);
    std::vector<int8_t> xq(static_cast<size_t>(M) * H);
    std::vector<float> xs(static_cast<size_t>(M) * G);
    f.compute(0).memcpy(xq.data(), rs[0].xq, xq.size()).wait();
    f.compute(0).memcpy(xs.data(), rs[0].xs, xs.size() * 4).wait();
    size_t qbad = 0;
    // (rot): the device quantises bf16(H_128 x), the routed experts' input
    std::vector<float> xqin(xv);
    if (s[0].rot)
      for (int r = 0; r < M; ++r)
        for (int g = 0; g < G; ++g) {
          float *p = xqin.data() + static_cast<size_t>(r) * H + g * 128;
          for (int hh = 1; hh < 128; hh <<= 1)
            for (int i = 0; i < 128; i += hh << 1)
              for (int j = i; j < i + hh; ++j) {
                const float u = p[j], w = p[j + hh];
                p[j] = u + w, p[j + hh] = u - w;
              }
          for (int i = 0; i < 128; ++i) p[i] = k::bf16_value(k::bf16_bits(p[i] * 0.08838834764831845f));
        }
    for (int r = 0; r < M; ++r)
      for (int g = 0; g < G; ++g) {
        float amax = 0.f;
        for (int i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(xqin[static_cast<size_t>(r) * H + g * 128 + i]));
        // IEEE quotients via double (exact after rounding to float: 53 >= 2 x 24 + 2); a float division
        // by a constant may be compiled as a reciprocal multiply under the host fp model
        const float sv = amax > 0.f ? static_cast<float>(static_cast<double>(amax) / 127.0) : 0.f;
        const float inv = amax > 0.f ? static_cast<float>(127.0 / static_cast<double>(amax)) : 0.f;
        qbad += xs[static_cast<size_t>(r) * G + g] != sv;
        for (int i = 0; i < 128; ++i) {
          const size_t at = static_cast<size_t>(r) * H + g * 128 + i;
          const float t = std::clamp(std::nearbyint(xqin[at] * inv), -127.f, 127.f);
          qbad += xq[at] != static_cast<int8_t>(t);
        }
      }
    double se = 0, sr = 0;
    for (size_t i = 0; i < q8.size(); ++i) {
      const double dv = k::bf16_value(q8[i]) - k::bf16_value(a[i]);
      se += dv * dv;
      sr += static_cast<double>(k::bf16_value(a[i])) * k::bf16_value(a[i]);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    bool same8 = true;
    if (C > 1) same8 = read16(f.compute(1), orow[1], q8.size()) == q8;
    const auto again8 = run_rows(1, 0, 0, true);
    bool split8 = true;
    for (int cut : {M / 4, 4 * ((M + 7) / 8)}) {
      if (cut <= 0 || cut >= M) continue;
      run_rows(0, cut, 0, true);
      split8 &= run_rows(1, M - cut, cut, true) == q8;
    }
    // lossless x: per (row, group) integers in [-127, 127] with one at +-127, times 2^e
    std::vector<float> xl(static_cast<size_t>(M + 4) * H, 0.f);
    std::uniform_int_distribution<int> qi(-127, 127), ei(-9, -5);
    for (int r = 0; r < M; ++r)
      for (int g = 0; g < G; ++g) {
        const float sc = std::ldexp(1.f, ei(rng));
        for (int i = 0; i < 128; ++i) xl[static_cast<size_t>(r) * H + g * 128 + i] = static_cast<float>(i == 0 ? (qi(rng) < 0 ? -127 : 127) : qi(rng)) * sc;
      }
    const auto xlb = bits_of(xl);
    for (int c = 0; c < C; ++c) f.compute(c).memcpy(x[c], xlb.data(), xlb.size() * 2).wait();
    auto h_by_assign = [&]() {  // card 0's h per (row, k) through the slot map (list order is by atomics)
      std::vector<int32_t> sl(static_cast<size_t>(M) * K);
      std::vector<uint16_t> hv(static_cast<size_t>(M) * K * I), out(hv.size());
      f.compute(0).memcpy(sl.data(), rs[0].slot, sl.size() * 4).wait();
      f.compute(0).memcpy(hv.data(), rs[0].h, hv.size() * 2).wait();
      for (size_t a_ = 0; a_ < sl.size(); ++a_)
        std::copy_n(hv.begin() + static_cast<size_t>(sl[a_]) * I, I, out.begin() + a_ * I);
      return out;
    };
    run_rows(0, 0, 0, false);
    const auto h16 = h_by_assign();
    run_rows(1, 0, 0, true);
    const auto h8 = h_by_assign();
    for (int c = 0; c < C; ++c) f.compute(c).memcpy(x[c], xb.data(), xb.size() * 2).wait();
    std::snprintf(what, sizeof what, "%s rows %d, W4A8: the int8 quantiser == the CPU reference (%zu of %zu differ)", tag, M, qbad, xq.size() + xs.size());
    expect(qbad == 0, what);
    std::snprintf(what, sizeof what, "%s rows %d, W4A8: out vs W4A16 relative rms %.2e (int8 rounding: < 2e-2)", tag, M, rel);
    expect(rel < 2e-2, what);
    std::snprintf(what, sizeof what, "%s rows %d, W4A8: both cards, run to run and calls split at M / 4 and M / 2 bit-identical", tag, M);
    expect(same8 && again8 == q8 && (M < 8 || split8), what);
    if (!s[0].rot) {  // rot: the quantiser sees bf16(H x), which no choice of bf16 x makes lossless
      std::snprintf(what, sizeof what, "%s rows %d, W4A8 on losslessly quantised x: h == the W4A16 h bit for bit", tag, M);
      expect(h8 == h16, what);
    }
  }
  for (int c = 0; c < C; ++c) {
    for (void *p : mem[c]) f.release(p);
    free_card(f, d[c]);
  }
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    const bool two = loader.size() > 1;
    xenithium::Fabric f(loader, two ? std::vector<size_t>{0, 1} : std::vector<size_t>{0});
    {
      const Model small = make_model(512, 256, 256, 32, 4, 6, 7u);
      test_one(f, small, "small (h 512, inter 256, 32 experts, top-4)");
      test_wide(f, small, "small");
      if (two) test_cards(f, small, "small");
      Model rounded = small;  // the numerics switch: y_e rounded to bf16
      rounded.s.down_bf16 = true;
      test_one(f, rounded, "small, down outputs bf16");
      Model tp = small;  // expert TP (one card = the half-block down path on the whole k)
      tp.s.tp = true;
      test_one(f, tp, "small, TP");
      test_wide(f, tp, "small, TP");
      if (two) test_cards(f, tp, "small, TP");
      for (int M : {1, 37, 300}) test_rows(f, small, "small, moe_rows", M);
      // the routed experts in a block-Hadamard basis (x: 128-blocks, h: 64-blocks)
      Model rot = small;
      rot.s.rot = true;
      test_one(f, rot, "small, rotated");
      Model rot_tp = tp;
      rot_tp.s.rot = true;
      test_one(f, rot_tp, "small, TP, rotated");
      if (two) test_cards(f, rot_tp, "small, TP, rotated");
      for (int M : {1, 37, 300}) test_rows(f, rot, "small, moe_rows, rotated", M);
      // decode W4A8 (tp), plain and rotated
      test_a8(f, tp, "small, TP");
      test_a8(f, rot_tp, "small, TP, rotated");
    }
    {
      const Model real = make_model(2560, 640, 640, 512, 10, 12, 11u);
      test_one(f, real, "model dims (h 2560, inter 640, 512 experts, top-10)");
      test_wide(f, real, "model dims");
      if (two) test_cards(f, real, "model dims");
      if (two) {
        Model tp = real;
        tp.s.tp = true;
        test_cards(f, tp, "model dims, TP");
        test_cpu(f, real, "model dims, TP");
      }
      test_rows(f, real, "model dims, moe_rows", 64);
      Model rreal = real;
      rreal.s.rot = true;
      if (two) {
        Model rtp = rreal;
        rtp.s.tp = true;
        test_cards(f, rtp, "model dims, TP, rotated");
      }
      test_rows(f, rreal, "model dims, moe_rows, rotated", 64);
      Model a8p = real;  // decode W4A8 at the model's dims (tp, one card)
      a8p.s.tp = true;
      test_a8(f, a8p, "model dims, TP");
      a8p.s.rot = true;
      test_a8(f, a8p, "model dims, TP, rotated");
    }
    if (loader.size() >= static_cast<size_t>(k::kWireCardsMax)) {  // TP4 (160 columns per card, the four-card wires)
      xenithium::Fabric f4(loader, {0, 1, 2, 3});
      Model tp4 = make_model(2560, 640, 640, 512, 10, 12, 11u);
      tp4.s.tp = true;
      test_cards(f4, tp4, "model dims, TP4");
      Model tp4_wide = tp4;  // the 64-row scratch
      tp4_wide.s.rows_cap = k::kMoeRows;
      test_cards(f4, tp4_wide, "model dims, TP4, 64-row scratch");
      for (int M : {1, 37, 300}) test_rows_quad(f4, tp4, "model dims, TP4", M);
    } else {
      std::printf("[SKIP] four-card MoE: %zu card(s) visible\n", loader.size());
    }
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
