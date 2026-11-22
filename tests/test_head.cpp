// Gate for the step's ends: hyper-connection mixer pieces, embedding gather, greedy pick
// (one card and across two cards through the mailbox wire).
// The mixer is not exact: device exp differs from the host's by an fp32 ulp (sigmoid of the
// injection and of the gates), the norm reduction order differs, and fp32 GEMM sums are rounded
// to bf16 where the reference sums in double. So it gets the "same rounding points" bound:
// every bf16 output within 2 ulps of the host reference that rounds where the device does
// (one flip upstream can carry one more downstream), at most 1 % of elements off at all, and
// run-to-run bit identity. The pick is exact: tokens must equal an independent host argmax.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/comm/wire.hpp"  // owner_rows (a step's rows by owner)
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/head/logprob.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/rt/device.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

// ulp distance of two bf16 bit patterns (same sign assumed close; -0/+0 equal).
int ulps(uint16_t a, uint16_t b) {
  auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
  return std::abs(lin(a) - lin(b));
}
struct Cmp {
  int max_ulp = 0;
  double scaled = 0;  // max |a - b| / max(|b|, rms of b's row)
  size_t off = 0, n = 0;
};
// A plain ulp count misleads on near-cancelling sums (tens of ulps on a value near zero), so
// the bound is 2 bf16 ulps of max(|ref|, the row's rms): |dev - ref| <= 2^-7 * that, with at
// most 1 % of elements off at all. `row` = elements per row for the rms.
Cmp compare(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, size_t row = 0) {
  Cmp c;
  c.n = a.size();
  if (!row) row = a.size();
  for (size_t r0 = 0; r0 < a.size(); r0 += row) {
    double ss = 0;
    for (size_t i = r0; i < std::min(a.size(), r0 + row); ++i) ss += std::pow(k::bf16_value(b[i]), 2);
    const double rms = std::sqrt(ss / static_cast<double>(row));
    for (size_t i = r0; i < std::min(a.size(), r0 + row); ++i) {
      const int u = ulps(a[i], b[i]);
      c.max_ulp = std::max(c.max_ulp, u);
      c.off += u != 0;
      const double y = k::bf16_value(b[i]);
      c.scaled = std::max(c.scaled, std::fabs(k::bf16_value(a[i]) - y) / std::max({std::fabs(y), rms, 1e-30}));
    }
  }
  return c;
}
std::string cmp_text(const Cmp &c) {
  char b[128];
  std::snprintf(b, sizeof b, "max %d ulp, worst %.2e of max(|ref|, rms), %zu / %zu off (%.3f %%)", c.max_ulp, c.scaled, c.off, c.n,
                100.0 * c.off / std::max<size_t>(1, c.n));
  return b;
}
bool cmp_ok(const Cmp &c) { return c.scaled <= 1.0 / 128 && c.off * 100 <= c.n; }

template <typename T>
std::vector<uint16_t> to_bits(const std::vector<T> &v) {
  std::vector<uint16_t> o(v.size());
  std::memcpy(o.data(), v.data(), v.size() * 2);
  return o;
}

// ---- HC mixer ----------------------------------------------------------------------------
// Two mixers in a row, as in a layer: mixer 1 (plain mix, or combine with an external pending
// block) produces its block input and its own injection logits from the merged down|inject
// GEMM; mixer 2 combines mixer 1's output as the pending block with those logits. Each mixer is
// checked against the host reference fed the device's own inputs (teacher-forced), on the
// unfused path and the fused (ESIMD) one; the standalone combine against the reference too.
struct HcDev {
  bf16 *res[2], *blk, *w1, *w2, *xn, *s, *out1, *out2;
  float *lora[2], *gate;
};
void test_hc(xenithium::Device &dev, const k::HcShape &s, int rows, bool combine, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int W = s.width(), R = s.lowrank, S = s.streams, H = s.hidden, RI = R + S, RP = (RI + 15) / 16 * 16;
  auto rnd_bits = [&](size_t n, float sd) {
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = k::bf16_bits(nd(rng) * sd);
    return v;
  };
  std::vector<uint16_t> res(4ull * W, 0), block = rnd_bits(4ull * H, 0.5f), nw1 = rnd_bits(W, 0.3f), nw2 = rnd_bits(W, 0.3f);
  const auto live = rnd_bits(static_cast<size_t>(rows) * W, 1.f);
  std::copy(live.begin(), live.end(), res.begin());
  // Per mixer: down|inject (RI rows, padded to RP with zero codes, scale 1) and up, int8.
  struct Mat {
    std::vector<int8_t> codes, packed;
    std::vector<float> scale, vals;
    int n, kk;
  };
  auto make = [&](int n, int npad, int kk, float sd) {
    Mat m;
    m.n = n;
    m.kk = kk;
    std::vector<uint16_t> w = rnd_bits(static_cast<size_t>(n) * kk, sd);
    m.codes.assign(static_cast<size_t>(npad) * kk, 0);
    m.scale.assign(npad, 1.f);
    k::quantize_i8_rows(w.data(), n, kk, m.codes.data(), m.scale.data(), 1);
    m.vals.assign(m.codes.begin(), m.codes.begin() + static_cast<size_t>(n) * kk);
    m.packed.resize(m.codes.size());
    k::pack_dense_i8(m.codes.data(), npad, kk, m.packed.data());
    return m;
  };
  Mat d1 = make(RI, RP, W, 0.02f), u1 = make(W, W, R, 0.02f), d2 = make(RI, RP, W, 0.02f), u2 = make(W, W, R, 0.02f);
  // External injection logits for mixer 1's combine, at columns [R, R + S) of an ld = RP row.
  std::vector<float> ext(4ull * RP, 0.f), ext_ref(static_cast<size_t>(rows) * S);
  for (int r = 0; r < rows; ++r)
    for (int st = 0; st < S; ++st) ext_ref[r * S + st] = ext[static_cast<size_t>(r) * RP + R + st] = nd(rng) * 3.f;

  auto &q = dev.compute();
  using MK = xenithium::l0::MemKind;
  auto up_mat = [&](const Mat &m, int npad) {
    auto *p = dev.alloc<int8_t>(m.packed.size(), MK::Device);
    auto *sc = dev.alloc<float>(m.scale.size(), MK::Device);
    q.memcpy(p, m.packed.data(), m.packed.size());
    q.memcpy(sc, m.scale.data(), m.scale.size() * 4).wait();
    return k::DenseWeights{p, sc, k::DenseKind::i8, npad, m.kk};
  };
  const k::DenseWeights wd1 = up_mat(d1, RP), wu1 = up_mat(u1, W), wd2 = up_mat(d2, RP), wu2 = up_mat(u2, W);
  HcDev g{};
  for (int i = 0; i < 2; ++i) {
    g.res[i] = dev.alloc<bf16>(res.size(), MK::Device);
    g.lora[i] = dev.alloc<float>(4ull * RP, MK::Device);
  }
  g.blk = dev.alloc<bf16>(block.size(), MK::Device);
  g.w1 = dev.alloc<bf16>(W, MK::Device);
  g.w2 = dev.alloc<bf16>(W, MK::Device);
  g.xn = dev.alloc<bf16>(4ull * W, MK::Device);
  g.s = dev.alloc<bf16>(4ull * R, MK::Device);
  g.gate = dev.alloc<float>(4ull * W, MK::Device);
  g.out1 = dev.alloc<bf16>(4ull * H, MK::Device);
  g.out2 = dev.alloc<bf16>(4ull * H, MK::Device);
  q.memcpy(g.blk, block.data(), block.size() * 2);
  q.memcpy(g.w1, nw1.data(), W * 2);
  q.memcpy(g.w2, nw2.data(), W * 2);
  q.wait();
  const k::HcWeights hw1{reinterpret_cast<const bf16 *>(g.w1), wd1, wu1}, hw2{reinterpret_cast<const bf16 *>(g.w2), wd2, wu2};
  const k::HcScratch sc{g.xn, g.s, g.gate};
  // Fresh state: residual in res[0], the external injection logits as the "previous" down output.
  auto fresh = [&] {
    k::HcState st;
    st.res[0] = g.res[0];
    st.res[1] = g.res[1];
    st.lora[0] = g.lora[0];
    st.lora[1] = g.lora[1];
    st.lora_ld = RP;
    q.memcpy(g.res[0], res.data(), res.size() * 2);
    q.memcpy(g.res[1], res.data(), res.size() * 2);
    q.memcpy(g.lora[0], ext.data(), ext.size() * 4);
    q.memset(g.lora[1], 0, 4ull * RP * 4).wait();
    return st;
  };

  char tag[128];
  std::snprintf(tag, sizeof tag, "hc S=%d H=%d R=%d rows=%d %s", S, H, R, rows, combine ? "combine+mix" : "mix");
  auto rd = [&](const bf16 *p, size_t n) {
    std::vector<uint16_t> v(n);
    q.memcpy(v.data(), p, n * 2).wait();
    return v;
  };
  std::vector<uint16_t> out2_of[2];
  const k::HcPath paths[2] = {k::HcPath::unfused, k::HcPath::three};
  for (int pi = 0; pi < 2; ++pi) {
    if (pi && !k::hc_fast_supported(s)) break;
    const char *path = pi == 0 ? "unfused" : "fused";
    std::vector<uint16_t> outs[2];
    for (int run = 0; run < 2; ++run) {
      k::HcState st = fresh();
      q.memset(g.xn, 0, 4ull * W * 2);
      q.memset(g.s, 0, 4ull * R * 2);
      q.fill(g.out1, bf16(0.f), 4ull * H);
      q.fill(g.out2, bf16(-777.f), 4ull * H).wait();
      k::hc_mixer(q, s, rows, st, combine ? g.blk : nullptr, hw1, sc, g.out1, paths[pi]);
      q.wait_and_throw();
      // Mixer 1 outputs as the device has them (teacher-forcing mixer 2's reference).
      const auto res1 = rd(st.residual(), static_cast<size_t>(rows) * W), out1 = rd(g.out1, static_cast<size_t>(rows) * H);
      std::vector<float> lora1(4ull * RP);
      q.memcpy(lora1.data(), st.down_out(), lora1.size() * 4).wait();
      k::hc_mixer(q, s, rows, st, g.out1, hw2, sc, g.out2, paths[pi]);
      q.wait_and_throw();
      const auto res2 = rd(st.residual(), static_cast<size_t>(rows) * W), out2 = rd(g.out2, static_cast<size_t>(rows) * H);
      outs[run] = out2;
      if (run) continue;
      // References.
      std::vector<uint16_t> r_res1(res.begin(), res.begin() + static_cast<size_t>(rows) * W), r_out1(static_cast<size_t>(rows) * H);
      std::vector<float> r_lora1(static_cast<size_t>(rows) * RI);
      k::hc_mix_ref(s, rows, r_res1.data(), combine ? block.data() : nullptr, ext_ref.data(), nw1.data(), {d1.vals.data(), d1.scale.data(), RI, W},
                    {u1.vals.data(), u1.scale.data(), W, R}, nullptr, r_lora1.data(), r_out1.data());
      std::vector<uint16_t> r_res2 = res1, r_out2(static_cast<size_t>(rows) * H);
      std::vector<float> inj2(static_cast<size_t>(rows) * S);
      for (int r = 0; r < rows; ++r)
        for (int st = 0; st < S; ++st) inj2[r * S + st] = lora1[static_cast<size_t>(r) * RP + R + st];
      k::hc_mix_ref(s, rows, r_res2.data(), out1.data(), inj2.data(), nw2.data(), {d2.vals.data(), d2.scale.data(), RI, W},
                    {u2.vals.data(), u2.scale.data(), W, R}, nullptr, nullptr, r_out2.data());
      // Mixer 1's injection logits (fp32 GEMM sums vs double then bf16): compare as bf16.
      std::vector<uint16_t> li(static_cast<size_t>(rows) * S), lr(li.size());
      for (int r = 0; r < rows; ++r)
        for (int st = 0; st < S; ++st) {
          li[r * S + st] = k::bf16_bits(lora1[static_cast<size_t>(r) * RP + R + st]);
          lr[r * S + st] = k::bf16_bits(r_lora1[static_cast<size_t>(r) * RI + R + st]);
        }
      const std::string p = std::string(tag) + " " + path;
      if (combine) {
        const Cmp c = compare(res1, r_res1, H);
        expect(cmp_ok(c), p + ": mixer 1 combined residual " + cmp_text(c));
      }
      const Cmp c1 = compare(out1, r_out1, H), ci = compare(li, lr), cr = compare(res2, r_res2, H), c2 = compare(out2, r_out2, H);
      expect(cmp_ok(c1), p + ": mixer 1 out " + cmp_text(c1));
      expect(ci.max_ulp <= 2, p + ": mixer 1 injection logits " + cmp_text(ci));
      expect(cmp_ok(cr), p + ": mixer 2 combine with mixer 1's injection " + cmp_text(cr));
      expect(cmp_ok(c2), p + ": mixer 2 out " + cmp_text(c2));
    }
    expect(outs[0] == outs[1], std::string(tag) + " " + path + ": run-to-run bit-identical");
    out2_of[pi] = outs[0];
  }
  if (!out2_of[1].empty()) {
    const Cmp c = compare(out2_of[1], out2_of[0], H);
    expect(cmp_ok(c), std::string(tag) + ": fused vs unfused " + cmp_text(c));
  }
  // Standalone combine of mixer 1's pending state (block = its output, its injection logits).
  {
    k::HcState st = fresh();
    k::hc_mixer(q, s, rows, st, nullptr, hw1, sc, g.out1, k::HcPath::unfused);
    q.wait_and_throw();
    const auto res1 = rd(st.residual(), static_cast<size_t>(rows) * W), out1 = rd(g.out1, static_cast<size_t>(rows) * H);
    std::vector<float> lora1(4ull * RP);
    q.memcpy(lora1.data(), st.down_out(), lora1.size() * 4).wait();
    k::hc_combine(q, s, rows, st, g.out1).wait_and_throw();
    const auto got = rd(st.residual(), static_cast<size_t>(rows) * W);
    std::vector<uint16_t> ref = res1;
    for (int r = 0; r < rows; ++r)
      for (int st = 0; st < S; ++st) {
        const float l = k::bf16_value(k::bf16_bits(lora1[static_cast<size_t>(r) * RP + R + st]));
        const float m = 2.f / (1.f + std::exp(-l / S));
        for (int i = 0; i < H; ++i) {
          const size_t at = (static_cast<size_t>(r) * S + st) * H + i;
          ref[at] = k::bf16_bits(std::fma(k::bf16_value(out1[static_cast<size_t>(r) * H + i]), m, k::bf16_value(res1[at])));
        }
      }
    const Cmp c = compare(got, ref, H);
    expect(cmp_ok(c), std::string(tag) + ": standalone combine " + cmp_text(c));
  }
  for (const auto *w : {&wd1, &wu1, &wd2, &wu2}) {
    dev.release(const_cast<void *>(w->w));
    dev.release(const_cast<float *>(w->scale));
  }
  for (void *p : {(void *)g.res[0], (void *)g.res[1], (void *)g.blk, (void *)g.w1, (void *)g.w2, (void *)g.xn, (void *)g.s, (void *)g.gate,
                  (void *)g.out1, (void *)g.out2, (void *)g.lora[0], (void *)g.lora[1]})
    dev.release(p);
}

// hc_mixer_rows (prefill) against hc_mixer (fused path, gated above) on 4-row slices of
// the same m rows: two chained mixers (combine an external block + its injection logits, then
// combine mixer 1's output). The GEMMs sum k in another order (pf_dense split-K), so the bound of
// the mixer gates; run-to-run bit-identical.
void test_hc_rows(xenithium::Device &dev, const k::HcShape &s, int m, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int W = s.width(), R = s.lowrank, S = s.streams, H = s.hidden, RI = R + S, RP = (RI + 15) / 16 * 16, M4 = (m + 7) / 8 * 8;  // rows rounded up to the 8-row shape
  auto rnd_bits = [&](size_t n, float sd) {
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = k::bf16_bits(nd(rng) * sd);
    return v;
  };
  const auto res0 = rnd_bits(static_cast<size_t>(M4) * W, 1.f), block = rnd_bits(static_cast<size_t>(M4) * H, 0.5f), nw1 = rnd_bits(W, 0.3f),
             nw2 = rnd_bits(W, 0.3f);
  std::vector<float> ext(static_cast<size_t>(M4) * RP, 0.f);
  for (int r = 0; r < m; ++r)
    for (int st = 0; st < S; ++st) ext[static_cast<size_t>(r) * RP + R + st] = nd(rng) * 3.f;
  auto &q = dev.compute();
  using MK = xenithium::l0::MemKind;
  std::vector<void *> mem;
  auto alloc = [&](size_t bytes) {
    void *p = dev.alloc<uint8_t>(std::max<size_t>(bytes, 64), MK::Device);
    mem.push_back(p);
    return p;
  };
  auto mat = [&](int n, int npad, int kk) {
    std::vector<uint16_t> w = rnd_bits(static_cast<size_t>(n) * kk, 0.02f);
    std::vector<int8_t> codes(static_cast<size_t>(npad) * kk, 0), packed(codes.size());
    std::vector<float> scale(npad, 1.f);
    k::quantize_i8_rows(w.data(), n, kk, codes.data(), scale.data(), 1);
    k::pack_dense_i8(codes.data(), npad, kk, packed.data());
    auto *p = static_cast<int8_t *>(alloc(packed.size()));
    auto *sc = static_cast<float *>(alloc(scale.size() * 4));
    q.memcpy(p, packed.data(), packed.size());
    q.memcpy(sc, scale.data(), scale.size() * 4).wait();
    return k::DenseWeights{p, sc, k::DenseKind::i8, npad, kk};
  };
  const k::DenseWeights wd1 = mat(RI, RP, W), wu1 = mat(W, W, R), wd2 = mat(RI, RP, W), wu2 = mat(W, W, R);
  auto upb = [&](const std::vector<uint16_t> &v) {
    auto *p = static_cast<bf16 *>(alloc(v.size() * 2));
    q.memcpy(p, v.data(), v.size() * 2).wait();
    return p;
  };
  const k::HcWeights hw1{upb(nw1), wd1, wu1}, hw2{upb(nw2), wd2, wu2};
  bf16 *res[2] = {static_cast<bf16 *>(alloc(res0.size() * 2)), static_cast<bf16 *>(alloc(res0.size() * 2))};
  float *lora[2] = {static_cast<float *>(alloc(ext.size() * 4)), static_cast<float *>(alloc(ext.size() * 4))};
  bf16 *blk = upb(block), *out1 = static_cast<bf16 *>(alloc(block.size() * 2)), *out2 = static_cast<bf16 *>(alloc(block.size() * 2));
  constexpr size_t kR16 = k::kStepRows32;  // slices of up to 32 rows (the 16- and 32-row shapes)
  const k::HcScratch sc{static_cast<bf16 *>(alloc(kR16 * W * 2)), static_cast<bf16 *>(alloc(kR16 * R * 2)), static_cast<float *>(alloc(kR16 * W * 4))};
  k::HcRowsScratch rsc;
  rsc.xn = static_cast<bf16 *>(alloc(static_cast<size_t>(M4) * W * 2));
  rsc.s = static_cast<bf16 *>(alloc(static_cast<size_t>(M4) * R * 2));
  rsc.gate = static_cast<float *>(alloc(static_cast<size_t>(M4) * W * 4));
  rsc.part = static_cast<float *>(alloc(static_cast<size_t>(k::kHcDownKs) * M4 * RP * 4));
  auto fresh = [&] {
    k::HcState st;
    for (int i = 0; i < 2; ++i) {
      st.res[i] = res[i];
      st.lora[i] = lora[i];
      q.memcpy(res[i], res0.data(), res0.size() * 2);
    }
    st.lora_ld = RP;
    q.memcpy(lora[0], ext.data(), ext.size() * 4);
    q.memset(lora[1], 0, ext.size() * 4);
    q.fill(out1, bf16(-777.f), block.size());
    q.fill(out2, bf16(-777.f), block.size()).wait();  // a path that writes nothing cannot pass
    return st;
  };
  auto rd = [&](const bf16 *p, size_t n) {
    std::vector<uint16_t> v(n);
    q.memcpy(v.data(), p, n * 2).wait();
    return v;
  };
  struct Got {
    std::vector<uint16_t> out1, out2, res;
  };
  // 0: hc_mixer on 4-row slices, 1: hc_mixer_rows, 2: hc_mixer_rows_part in row parts, 3: hc_mixer on
  // 8-row slices (the 8-row shape), 4: on 16-row slices (the 16-row shape), 5: on 32-row slices (the 32-row shape)
  // A step's rows by owner (4..32 rows): 6: both mixers as hc_mixer_at on the four owners' ranges;
  // 7: mixer 1 that way, then the pending combine as hc_combine_at per range; 8: the same with hc_combine on every row
  auto path = [&](int mode) {
    k::HcState st = fresh();
    if (mode >= 6) {
      // The four cards' ranges on this one state: every range runs a launch before any runs the next (a range's down
      // GEMM writes its whole row shape, past its last row), each on a copy of the indices, which flip once.
      auto each_owner = [&](auto &&launch) {
        k::HcState after = st;
        for (int c = 0; c < k::kWireCardsMax; ++c) {
          after = st;
          launch(k::owner_rows(c, m), after);
        }
        st = after;
      };
      each_owner([&](k::RowRange own, k::HcState &v) { k::hc_mixer_at(q, s, own, v, blk, hw1, sc, out1); });
      if (mode == 6) each_owner([&](k::RowRange own, k::HcState &v) { k::hc_mixer_at(q, s, own, v, out1, hw2, sc, out2); });
      if (mode == 7) each_owner([&](k::RowRange own, k::HcState &v) { k::hc_combine_at(q, s, own, v, out1); });
      if (mode == 8) k::hc_combine(q, s, m, st, out1);
    } else if (mode == 2) {
      // the second half first, as card 1 runs it, then the first; mixer 2 in three parts
      const int h = m / 2;
      auto parts = [&](const bf16 *in, const k::HcWeights &hw, bf16 *o, std::vector<std::pair<int, int>> ranges) {
        for (const auto &[r0, r1] : ranges)
          if (r1 > r0) k::hc_mixer_rows_part(q, s, r0, r1, st, in, hw, rsc, o);
        k::hc_mixer_rows_flip(st, hw);
      };
      parts(blk, hw1, out1, {{h, m}, {0, h}});
      parts(out1, hw2, out2, {{0, m / 3}, {(2 * m) / 3, m}, {m / 3, (2 * m) / 3}});
    } else if (mode == 1) {
      k::hc_mixer_rows(q, s, m, st, blk, hw1, rsc, out1);
      k::hc_mixer_rows(q, s, m, st, out1, hw2, rsc, out2);
    } else {
      const int SL = mode == 5 ? 32 : mode == 4 ? 16 : mode == 3 ? 8 : 4;
      for (int sl = 0; sl * SL < m; ++sl) {
        k::HcState v = st;
        for (int i = 0; i < 2; ++i) {
          v.res[i] = st.res[i] + static_cast<size_t>(sl) * SL * W;
          v.lora[i] = st.lora[i] + static_cast<size_t>(sl) * SL * RP;
        }
        const size_t o = static_cast<size_t>(sl) * SL * H;
        const int rr = std::min(SL, m - SL * sl);
        k::hc_mixer(q, s, rr, v, blk + o, hw1, sc, out1 + o, k::HcPath::three);
        k::hc_mixer(q, s, rr, v, out1 + o, hw2, sc, out2 + o, k::HcPath::three);
        if (sl * SL + SL >= m) st = v, st.res[0] = res[0], st.res[1] = res[1], st.lora[0] = lora[0], st.lora[1] = lora[1];
      }
    }
    q.wait_and_throw();
    return Got{rd(out1, static_cast<size_t>(m) * H), rd(out2, static_cast<size_t>(m) * H), rd(st.residual(), static_cast<size_t>(m) * W)};
  };
  const Got a = path(0), b = path(1), b2 = path(1), pt = path(2), a8 = path(3), a16 = path(4), a32 = path(5);
  rsc.split_up_mix = true;  // the two-launch up + mix (pf_dense + hc_gate_mix16) at every row count
  const Got sp = path(1), spt = path(2);
  rsc.split_up_mix = false;
  char tag[128];
  std::snprintf(tag, sizeof tag, "hc_mixer_rows S=%d H=%d R=%d, %d rows (two chained mixers)", S, H, R, m);
  expect(sp.out1 == b.out1 && sp.out2 == b.out2 && sp.res == b.res && spt.out2 == b.out2,
         std::string(tag) + ": up + gate mix in one launch (hc_up_mix_rows above " + std::to_string(k::kHcUpMixRowsMin) +
             " rows per call) == pf_dense + hc_gate_mix16, bit for bit");
  const Cmp c1 = compare(b.out1, a.out1, H), c2 = compare(b.out2, a.out2, H), cr = compare(b.res, a.res, H);
  expect(cmp_ok(c1), std::string(tag) + ": mixer 1 out vs hc_mixer on 4-row slices " + cmp_text(c1));
  expect(cmp_ok(c2), std::string(tag) + ": mixer 2 out vs slices " + cmp_text(c2));
  expect(cmp_ok(cr), std::string(tag) + ": residual after both vs slices " + cmp_text(cr));
  expect(b.out2 == b2.out2 && b.res == b2.res, std::string(tag) + ": run-to-run bit-identical");
  expect(pt.out1 == b.out1 && pt.out2 == b.out2 && pt.res == b.res, std::string(tag) + ": in row parts (any order) == one call, bit for bit");
  expect(a8.out1 == a.out1 && a8.out2 == a.out2 && a8.res == a.res, std::string(tag) + ": hc_mixer on 8-row slices (8-row shape) == on 4-row slices, bit for bit");
  expect(a16.out1 == a.out1 && a16.out2 == a.out2 && a16.res == a.res, std::string(tag) + ": hc_mixer on 16-row slices (16-row shape) == on 4-row slices, bit for bit");
  expect(a32.out1 == a.out1 && a32.out2 == a.out2 && a32.res == a.res, std::string(tag) + ": hc_mixer on 32-row slices (32-row shape) == on 4-row slices, bit for bit");
  if (m >= k::kOwnerMinRows && m <= k::kPairRows) {
    const Got own = path(6), combined_own = path(7), combined = path(8);
    expect(own.out1 == a.out1 && own.out2 == a.out2 && own.res == a.res,
           std::string(tag) + ": hc_mixer_at on the four owners' row ranges == hc_mixer on 4-row slices, bit for bit");
    expect(combined_own.out1 == a.out1 && combined_own.res == combined.res,
           std::string(tag) + ": hc_combine_at on the owners' ranges == hc_combine on every row, bit for bit");
  }
  for (void *p : mem) dev.release(p);
}

// ---- embedding gather from pinned host ---------------------------------------------------------
void test_embed(xenithium::Device &dev) {
  const int vocab = 3000, H = 2560, S = 4, rows = 4;
  std::mt19937 rng(5);
  using MK = xenithium::l0::MemKind;
  auto *table = dev.alloc<uint16_t>(static_cast<size_t>(vocab) * H, MK::Host);
  for (size_t i = 0; i < static_cast<size_t>(vocab) * H; ++i) table[i] = static_cast<uint16_t>(rng());
  const int32_t ids[rows] = {0, 2999, 17, 17};
  auto *d_ids = dev.alloc<int32_t>(rows, MK::Device);
  auto *d_res = dev.alloc<bf16>(static_cast<size_t>(rows) * S * H, MK::Device);
  auto &q = dev.compute();
  q.memcpy(d_ids, ids, sizeof ids).wait();
  k::embed_gather(q, reinterpret_cast<const bf16 *>(table), d_ids, rows, H, S, d_res).wait_and_throw();
  std::vector<uint16_t> got(static_cast<size_t>(rows) * S * H);
  q.memcpy(got.data(), d_res, got.size() * 2).wait();
  bool ok = true;
  for (int r = 0; r < rows; ++r)
    for (int s = 0; s < S; ++s) ok &= !std::memcmp(&got[(static_cast<size_t>(r) * S + s) * H], table + static_cast<size_t>(ids[r]) * H, H * 2);
  expect(ok, "embed_gather from pinned host: every stream equals the table row (4 rows, repeated id)");
  dev.release(table);
  dev.release(d_ids);
  dev.release(d_res);
}

// ---- plain RMS norm -----------------------------------------------------------
// The (1 + w) norm of the dense qwen35 family: no add, a bf16 add (a block's output) and an fp32 add
// (the all-reduced partial sum). The add is exact (bit for bit on the residual); the norm output gets
// the 2-ulp-of-max(|ref|, rms) bound and run-to-run identity. Rows past `rows` must stay untouched.
void test_rms_norm(xenithium::Device &dev, int H, int rows, unsigned seed) {
  using MK = xenithium::l0::MemKind;
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int cap = 8;
  std::vector<uint16_t> res0(static_cast<size_t>(cap) * H), addb(res0.size()), w(H);
  std::vector<float> addf(res0.size());
  for (auto &v : res0) v = k::bf16_bits(nd(rng) * 3.f);
  for (auto &v : addb) v = k::bf16_bits(nd(rng));
  for (auto &v : addf) v = nd(rng) * 0.7f;
  for (auto &v : w) v = k::bf16_bits(nd(rng) * 0.3f);
  auto *d_res = dev.alloc<bf16>(res0.size(), MK::Device);
  auto *d_addb = dev.alloc<bf16>(addb.size(), MK::Device);
  auto *d_addf = dev.alloc<float>(addf.size(), MK::Device);
  auto *d_w = dev.alloc<bf16>(w.size(), MK::Device);
  auto *d_xn = dev.alloc<bf16>(res0.size(), MK::Device);
  auto &q = dev.compute();
  q.memcpy(d_addb, addb.data(), addb.size() * 2);
  q.memcpy(d_addf, addf.data(), addf.size() * 4);
  q.memcpy(d_w, w.data(), w.size() * 2).wait();
  for (int mode = 0; mode < 3; ++mode) {
    std::vector<uint16_t> res_ref = res0, xn_ref(res0.size(), 0xFFFF), got_res(res0.size()), got_xn(res0.size()), again(res0.size());
    k::rms_norm_ref(rows, H, res_ref.data(), mode == 1 ? addb.data() : nullptr, mode == 2 ? addf.data() : nullptr, w.data(), 1e-6f, xn_ref.data());
    for (int rep = 0; rep < 2; ++rep) {
      q.memcpy(d_res, res0.data(), res0.size() * 2);
      q.memset(d_xn, 0xFF, res0.size() * 2).wait();
      k::rms_norm(q, rows, H, d_res, mode == 1 ? d_addb : nullptr, mode == 2 ? d_addf : nullptr, d_w, 1e-6f, d_xn).wait_and_throw();
      q.memcpy(rep ? again.data() : got_xn.data(), d_xn, res0.size() * 2);
      q.memcpy(got_res.data(), d_res, res0.size() * 2).wait();
    }
    const size_t live = static_cast<size_t>(rows) * H;
    const Cmp c = compare(std::vector<uint16_t>(got_xn.begin(), got_xn.begin() + live), std::vector<uint16_t>(xn_ref.begin(), xn_ref.begin() + live), H);
    const bool res_exact = got_res == res_ref;  // the add bit for bit, dead rows untouched
    const bool dead = std::equal(got_xn.begin() + live, got_xn.end(), xn_ref.begin() + live);
    const char *what = mode == 0 ? "no add" : mode == 1 ? "bf16 add" : "fp32 add";
    char line[200];
    std::snprintf(line, sizeof line, "rms_norm H=%d rows=%d %s: residual %s, dead rows %s, xn %s", H, rows, what, res_exact ? "exact" : "DIFFERS",
                  dead ? "untouched" : "WRITTEN", cmp_text(c).c_str());
    expect(res_exact && dead && cmp_ok(c), line);
    expect(got_xn == again, "rms_norm: run-to-run bit identical");
  }
  // the plain norms (bf16 add): 1 = bf16(w x bf16(y)) (arch llama), 2 = bf16(w x y) (arch flint), vs a host rule on the
  // same fp32 rr (the device's y = v x rr, rr from the fp32 sum; compared within the cmp bound like mode 0)
  for (int plain = 1; plain <= 2; ++plain) {
    std::vector<uint16_t> res_ref = res0, xn_ref(res0.size(), 0xFFFF), got_xn(res0.size()), got_res(res0.size());
    for (int r = 0; r < rows; ++r) {
      uint16_t *x = res_ref.data() + static_cast<size_t>(r) * H;
      double ss = 0;
      for (int i = 0; i < H; ++i) {
        const float v = k::bf16_value(x[i] = k::bf16_bits(k::bf16_value(x[i]) + k::bf16_value(addb[static_cast<size_t>(r) * H + i])));
        ss += static_cast<double>(v) * v;
      }
      const float rr = 1.f / std::sqrt(static_cast<float>(ss / H) + 1e-6f);
      for (int i = 0; i < H; ++i) {
        const float y = k::bf16_value(x[i]) * rr;
        xn_ref[static_cast<size_t>(r) * H + i] =
            k::bf16_bits((plain == 1 ? k::bf16_value(k::bf16_bits(y)) : y) * k::bf16_value(w[static_cast<size_t>(i)]));
      }
    }
    q.memcpy(d_res, res0.data(), res0.size() * 2);
    q.memset(d_xn, 0xFF, res0.size() * 2).wait();
    k::rms_norm(q, rows, H, d_res, d_addb, nullptr, d_w, 1e-6f, d_xn, plain).wait_and_throw();
    q.memcpy(got_xn.data(), d_xn, res0.size() * 2);
    q.memcpy(got_res.data(), d_res, res0.size() * 2).wait();
    const size_t live = static_cast<size_t>(rows) * H;
    const Cmp c = compare(std::vector<uint16_t>(got_xn.begin(), got_xn.begin() + live), std::vector<uint16_t>(xn_ref.begin(), xn_ref.begin() + live), H);
    char line[200];
    std::snprintf(line, sizeof line, "rms_norm plain %d H=%d rows=%d bf16 add: residual %s, xn %s", plain, H, rows, got_res == res_ref ? "exact" : "DIFFERS",
                  cmp_text(c).c_str());
    expect(got_res == res_ref && cmp_ok(c), line);
  }
  for (void *p : {(void *)d_res, (void *)d_addb, (void *)d_addf, (void *)d_w, (void *)d_xn}) dev.release(p);
}

// ---- greedy pick -----------------------------------------------------------------------------
// Logits with the traps planted: a cross-chunk tie (two equal maxima, the lower index must win),
// -0 vs +0 as the max (lowest index), values that round to the same bf16 but differ in fp32
// (bf16 tie: lowest index wins although fp32 says otherwise), NaN (never wins).
std::vector<float> make_logits(int rows, int n, unsigned seed, int64_t col_base, std::vector<int> *planted = nullptr) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 3.f);
  std::vector<float> v(static_cast<size_t>(rows) * n);
  for (auto &x : v) x = nd(rng);
  for (int r = 0; r < rows; ++r) {
    float *row = v.data() + static_cast<size_t>(r) * n;
    const int a = static_cast<int>(rng() % n), b = static_cast<int>(rng() % n);
    switch ((r + seed) % 4) {
      case 0:  // exact tie
        row[a] = row[b] = 50.f;
        break;
      case 1:  // bf16 tie: 50.0 and 50.1 both round to 50.0 in bf16 (spacing 0.25)
        row[a] = 50.0f;
        row[b] = 50.1f;
        break;
      case 2:  // everything negative, -0 and +0 at the top
        for (int i = 0; i < n; ++i) row[i] = -std::fabs(row[i]) - 1.f;
        row[a] = -0.f;
        row[b] = 0.f;
        break;
      default:  // NaN beside a clear winner
        row[a] = NAN;
        row[b] = 60.f;
    }
    if (planted) planted->push_back(std::min(a, b));
  }
  (void)col_base;
  return v;
}

// logprob_stats: a row split in two halves (the two cards' vocabulary parts, here on one
// card) reduced to records, merged as the server merges them, against log-softmax over the whole
// bf16-rounded row on the host: rank and the top entries exact (ids in order, ties to the lower id),
// logprobs within 2e-5. Values from a coarse grid, so the top has bf16 ties across the halves.
void test_logprob(xenithium::Device &dev) {
  auto &q = dev.compute();
  using MK = xenithium::l0::MemKind;
  for (const int n : {124160, 37, 5}) {
    for (const int rows : {1, 4}) {
      for (const int top : {0, 1, 5, 20}) {
        std::mt19937 rng(static_cast<unsigned>(n * 31 + rows * 7 + top));
        std::uniform_int_distribution<int> grid(-400, 60);  // value = grid / 8: many equal values at the top
        std::vector<float> half[2];
        for (auto &h : half) {
          h.resize(static_cast<size_t>(rows) * n);
          // n = 37: spread x8 (few values near the top), otherwise a dense grid (many ties at the top)
          for (auto &v : h) v = static_cast<float>(grid(rng)) * (n == 37 ? 1.f : 0.125f) + (rng() % 3 == 0 ? 1e-3f : 0.f);  // + below a bf16 ulp sometimes
        }
        std::vector<int32_t> tgt(rows);
        for (int r = 0; r < rows; ++r) tgt[r] = static_cast<int32_t>(rng() % (2u * n));
        if (rows == 4) tgt[3] = -1;  // a row without a target
        k::LpArgs a;
        for (int r = 0; r < rows; ++r) {
          if (tgt[r] < 0) {
            a.xt[r] = INFINITY;
            continue;
          }
          const int c = tgt[r] / n;
          a.xt[r] = k::bf16_value(k::bf16_bits(half[c][static_cast<size_t>(r) * n + (tgt[r] - c * n)]));
        }
        std::vector<float> rec[2];
        for (int c = 0; c < 2; ++c) {
          auto *d_l = dev.alloc<float>(half[c].size(), MK::Device);
          auto *d_r = dev.alloc<float>(static_cast<size_t>(rows) * k::kLpRec, MK::Device);
          q.memcpy(d_l, half[c].data(), half[c].size() * 4).wait();
          k::logprob_stats(q, d_l, n, rows, n, c * n, top, a, d_r).wait_and_throw();
          rec[c].resize(static_cast<size_t>(rows) * k::kLpRec);
          q.memcpy(rec[c].data(), d_r, rec[c].size() * 4).wait();
          dev.release(d_l);
          dev.release(d_r);
        }
        bool ok = true;
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
          // host reference over the joined row
          std::vector<std::pair<float, int>> all;
          double m = -INFINITY;
          for (int c = 0; c < 2; ++c)
            for (int i = 0; i < n; ++i) {
              const float v = k::bf16_value(k::bf16_bits(half[c][static_cast<size_t>(r) * n + i]));
              all.push_back({v, c * n + i});
              m = std::max(m, static_cast<double>(v));
            }
          double z = 0;
          for (const auto &e : all) z += std::exp(static_cast<double>(e.first) - m);
          const double lse = m + std::log(z);
          std::stable_sort(all.begin(), all.end(), [](const auto &x, const auto &y) { return x.first > y.first; });
          // merge the records
          double M = -INFINITY, Z = 0;
          int64_t above = 0;
          std::vector<std::pair<float, int>> got;
          for (int c = 0; c < 2; ++c) {
            const float *e = rec[c].data() + static_cast<size_t>(r) * k::kLpRec;
            M = std::max(M, static_cast<double>(e[0]));
          }
          for (int c = 0; c < 2; ++c) {
            const float *e = rec[c].data() + static_cast<size_t>(r) * k::kLpRec;
            Z += e[1] * std::exp(static_cast<double>(e[0]) - M);
            above += static_cast<int64_t>(e[2]);
            ok &= static_cast<int>(e[3]) == std::min(top, n);
            for (int i = 0; i < static_cast<int>(e[3]); ++i) got.push_back({e[5 + 2 * i], static_cast<int>(e[4 + 2 * i])});
          }
          std::sort(got.begin(), got.end(), [](const auto &x, const auto &y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
          const double glse = M + std::log(Z);
          worst = std::max(worst, std::isfinite(glse - lse) ? std::fabs(glse - lse) : HUGE_VAL);
          for (int i = 0; i < top && i < 2 * n; ++i) ok &= got.at(static_cast<size_t>(i)).second == all[static_cast<size_t>(i)].second && got[static_cast<size_t>(i)].first == all[static_cast<size_t>(i)].first;
          if (tgt[r] >= 0) {
            int64_t want = 0;
            for (const auto &e : all) want += e.first > a.xt[r];
            ok &= above == want;
          } else {
            ok &= above == 0;
          }
        }
        ok &= worst <= 2e-5;
        if (n == 124160 && rows == 4 && (top == 5 || top == 0)) {  // timing: one card's half, the head's 4 rows
          auto *d_l = dev.alloc<float>(half[0].size(), MK::Device);
          auto *d_r = dev.alloc<float>(static_cast<size_t>(rows) * k::kLpRec, MK::Device);
          q.memcpy(d_l, half[0].data(), half[0].size() * 4).wait();
          std::vector<double> us;
          for (int it = 0; it < 30; ++it) {
            const auto t0 = std::chrono::steady_clock::now();
            k::logprob_stats(q, d_l, n, rows, n, 0, top, a, d_r).wait();
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
          }
          std::sort(us.begin(), us.end());
          std::printf("logprob stats timing: 4 rows x %d columns, top %d: %.1f us median (host wall incl. submit + wait)\n", n, top, us[us.size() / 2]);
          dev.release(d_l);
          dev.release(d_r);
        }
        char what[160];
        std::snprintf(what, sizeof what, "logprob stats n=%d x 2 rows=%d top=%d: rank, top ids / values == host log-softmax (lse off by %.1e)", n, rows, top, worst);
        expect(ok, what);
      }
    }
  }
}

// logprob_stats_rows (bf16 logits, targets' logits from a device array, any row count) +
// logprob_targets == logprob_stats on the fp32 logits in 4-row slices, record for record (the entry
// lists compared as sets: their order is by atomics), 64 rows of one card's half, top 0 / 5 / 20.
void test_logprob_rows(xenithium::Device &dev) {
  auto &q = dev.compute();
  using MK = xenithium::l0::MemKind;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const int n = 124160, rows = 64, col_base = 124160;  // card 1's half
  std::mt19937 rng(4401);
  std::uniform_int_distribution<int> grid(-400, 60);
  std::vector<float> lf(static_cast<size_t>(rows) * n);
  for (auto &v : lf) v = static_cast<float>(grid(rng)) * 0.125f + (rng() % 3 == 0 ? 1e-3f : 0.f);
  std::vector<bf16> lb(lf.size());
  for (size_t i = 0; i < lf.size(); ++i) lb[i] = bf16(lf[i]);
  std::vector<int32_t> tgt(rows);
  for (int r = 0; r < rows; ++r) tgt[r] = r % 7 == 3 ? -1 : static_cast<int32_t>(rng() % (2u * n));  // both halves, some none
  auto *d_f = dev.alloc<float>(lf.size(), MK::Device);
  auto *d_b = dev.alloc<bf16>(lb.size(), MK::Device);
  auto *d_t = dev.alloc<int32_t>(rows, MK::Device);
  auto *d_x = dev.alloc<float>(rows, MK::Device);
  auto *d_r = dev.alloc<float>(static_cast<size_t>(rows) * k::kLpRec, MK::Device);
  q.memcpy(d_f, lf.data(), lf.size() * 4);
  q.memcpy(d_b, lb.data(), lb.size() * 2);
  q.memcpy(d_t, tgt.data(), rows * 4).wait();
  // the gather: owned targets' bf16 logits, 0 elsewhere
  k::logprob_targets(q, d_b, n, rows, n, col_base, d_t, d_x).wait_and_throw();
  std::vector<float> got_x(rows);
  q.memcpy(got_x.data(), d_x, rows * 4).wait();
  bool gather_ok = true;
  std::vector<float> xt(rows);
  for (int r = 0; r < rows; ++r) {
    const int32_t t = tgt[r];
    const bool own = t >= col_base && t < col_base + n;
    const float want = own ? k::bf16_value(k::bf16_bits(lf[static_cast<size_t>(r) * n + (t - col_base)])) : 0.f;
    gather_ok &= got_x[r] == want;
    // the other card's targets: any finite value stands in for the logit it would send
    xt[r] = t < 0 ? INFINITY : own ? want : static_cast<float>(grid(rng)) * 0.125f;
  }
  expect(gather_ok, "logprob_targets: 64 rows, the owned targets' bf16 logits, 0 for the other card's / none");
  q.memcpy(d_x, xt.data(), rows * 4).wait();
  auto entries = [](const float *e) {
    std::vector<std::pair<float, float>> v;
    for (int i = 0; i < static_cast<int>(e[3]); ++i) v.push_back({e[4 + 2 * i], e[5 + 2 * i]});
    std::sort(v.begin(), v.end());
    return v;
  };
  for (const int top : {0, 5, 20}) {
    std::vector<float> a(static_cast<size_t>(rows) * k::kLpRec), b(a.size());
    for (int s = 0; s < rows; s += 4) {
      k::LpArgs la;
      for (int r = 0; r < 4; ++r) la.xt[r] = xt[s + r];
      k::logprob_stats(q, d_f + static_cast<size_t>(s) * n, n, 4, n, col_base, top, la, d_r + static_cast<size_t>(s) * k::kLpRec);
    }
    q.memcpy(a.data(), d_r, a.size() * 4).wait();
    q.fill(d_r, -777.f, a.size());
    k::logprob_stats_rows(q, d_b, n, rows, n, col_base, top, d_x, d_r).wait_and_throw();
    q.memcpy(b.data(), d_r, b.size() * 4).wait();
    bool ok = true;
    for (int r = 0; r < rows; ++r) {
      const float *x = a.data() + static_cast<size_t>(r) * k::kLpRec, *y = b.data() + static_cast<size_t>(r) * k::kLpRec;
      ok &= std::memcmp(x, y, 4 * sizeof(float)) == 0 && entries(x) == entries(y);
    }
    char what[160];
    std::snprintf(what, sizeof what, "logprob_stats_rows: 64 rows of bf16 logits, top %d == logprob_stats on fp32 4-row slices, record for record", top);
    expect(ok, what);
  }
  for (void *p : {(void *)d_f, (void *)d_b, (void *)d_t, (void *)d_x, (void *)d_r}) dev.release(p);
}

void test_pick_one(xenithium::Device &dev) {
  auto &q = dev.compute();
  using MK = xenithium::l0::MemKind;
  for (const int n : {124160, 4097, 4096, 16, 1}) {
    for (const int rows : {1, 4, 8, 16}) {  // 16 = kPickMaxRows
      const auto lg = make_logits(rows, n, 11u + n + rows, 0);
      auto *d_l = dev.alloc<float>(lg.size(), MK::Device);
      auto *d_t = dev.alloc<int32_t>(rows, MK::Device);
      void *scratch = dev.alloc<uint8_t>(k::pick_scratch_bytes(n), MK::Device);
      q.memcpy(d_l, lg.data(), lg.size() * 4).wait();
      k::greedy_pick(q, d_l, n, rows, n, 1000, scratch, nullptr, d_t).wait_and_throw();
      std::vector<int32_t> t(rows);
      q.memcpy(t.data(), d_t, rows * 4).wait();
      bool ok = true;
      for (int r = 0; r < rows; ++r) ok &= t[r] == k::pick_ref(lg.data() + static_cast<size_t>(r) * n, n, 1000);
      char what[96];
      std::snprintf(what, sizeof what, "greedy pick one card n=%d rows=%d == host argmax (ties, bf16 ties, -0, NaN)", n, rows);
      expect(ok, what);
      dev.release(d_l);
      dev.release(d_t);
      dev.release(scratch);
    }
  }
}

// C = 2 or 4 cards (four: the two-hop pick, PickWire:peer2_inbox), the vocabulary in C parts.
void test_pick_cards(const xenithium::l0::Loader &loader, const int rows, const int C) {  // rows 4 and 16
  std::vector<size_t> ids;
  for (int c = 0; c < C; ++c) ids.push_back(static_cast<size_t>(c));
  xenithium::Fabric fab(loader, ids);
  const int n = 248320 / C, steps = 40;
  const size_t words = C == 4 ? k::kPickWireWords4 : k::kPickWireWords;
  std::vector<k::PickWire> w(C);
  std::vector<void *> scratch(C);
  std::vector<int32_t *> d_t(C);
  std::vector<float *> d_l(C);
  for (int c = 0; c < C; ++c) {
    w[c].inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(words * 4));
    w[c].step = static_cast<int32_t *>(fab.alloc(c, 4096));
    w[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    scratch[c] = fab.alloc(c, k::pick_scratch_bytes(n));
    d_t[c] = static_cast<int32_t *>(fab.alloc(c, static_cast<size_t>(steps) * rows * 4));
    d_l[c] = static_cast<float *>(fab.alloc(c, static_cast<size_t>(steps) * rows * n * 4));
  }
  for (int c = 0; c < C; ++c) {
    w[c].peer_inbox = w[c ^ 1].inbox;
    if (C == 4) w[c].peer2_inbox = w[c ^ 2].inbox;
    std::vector<uint32_t> empty(words, k::kMailEmpty);
    uint32_t zero[2] = {0, 0};
    fab.compute(c).memcpy(w[c].inbox, empty.data(), empty.size() * 4);
    fab.compute(c).memcpy(w[c].step, zero, 4);
    fab.compute(c).memcpy(w[c].error, zero, 4).wait();
  }
  // Per step: every card's logits; every 3rd step plants a cross-card exact tie at the max (card 0's last column, the
  // others' first: card 0's lower global index must win), every 3rd+1 makes the last card the clear winner.
  std::vector<std::vector<float>> lg(C);
  std::vector<int64_t> want(static_cast<size_t>(steps) * rows);
  for (int st = 0; st < steps; ++st) {
    std::vector<std::vector<float>> part(C);
    for (int c = 0; c < C; ++c) part[c] = make_logits(rows, n, 100u + 800u * c + st, c * n);
    for (int r = 0; r < rows; ++r) {
      std::vector<float> full;
      for (int c = 0; c < C; ++c) {
        float *rc = part[c].data() + static_cast<size_t>(r) * n;
        if (st % 3 == 0) rc[c == 0 ? n - 1 : 0] = 80.f;
        else if (st % 3 == 1 && c == C - 1) rc[n / 2] = 90.f;
        full.insert(full.end(), rc, rc + n);
      }
      want[static_cast<size_t>(st) * rows + r] = k::pick_ref(full.data(), static_cast<int64_t>(C) * n, 0);
    }
    for (int c = 0; c < C; ++c) lg[c].insert(lg[c].end(), part[c].begin(), part[c].end());
  }
  for (int c = 0; c < C; ++c) fab.compute(c).memcpy(d_l[c], lg[c].data(), lg[c].size() * 4).wait();
  // Submit every step on every card back to back, no host wait in between: the parity
  // double-buffering is what keeps a card that runs ahead from overwriting an unread key.
  const auto t0 = std::chrono::steady_clock::now();
  for (int st = 0; st < steps; ++st)
    for (int c = 0; c < C; ++c)
      k::greedy_pick(fab.compute(c), d_l[c] + static_cast<size_t>(st) * rows * n, n, rows, n, c * n, scratch[c], &w[c],
                     d_t[c] + static_cast<size_t>(st) * rows);
  for (int c = 0; c < C; ++c) fab.compute(c).wait_and_throw();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  bool ok = true, same = true;
  uint32_t err = 0;
  std::vector<std::vector<int32_t>> t(C);
  for (int c = 0; c < C; ++c) {
    uint32_t e = 0;
    t[c].resize(static_cast<size_t>(steps) * rows);
    fab.compute(c).memcpy(t[c].data(), d_t[c], t[c].size() * 4);
    fab.compute(c).memcpy(&e, w[c].error, 4).wait();
    err |= e;
  }
  int bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    ok &= t[0][i] == want[i];
    for (int c = 1; c < C; ++c) same &= t[0][i] == t[c][i];
    if (t[0][i] != want[i] && bad++ < 6)
      std::printf("  step %zu row %zu: want %lld card0 %d\n", i / rows, i % rows, static_cast<long long>(want[i]), t[0][i]);
  }
  if (bad) std::printf("  %d mismatching rows; error words %u\n", bad, err);
  char what[200];
  std::snprintf(what, sizeof what, "greedy pick across %d cards: %d steps x %d rows == host argmax over every part (%.2f ms total)", C, steps, rows, ms);
  expect(ok, what);
  std::snprintf(what, sizeof what, "greedy pick across %d cards: every card holds the same tokens; no mailbox timeout", C);
  expect(same && err == 0, what);
  std::vector<uint32_t> box(words);
  bool clean = true;
  for (int c = 0; c < C; ++c) {
    fab.compute(c).memcpy(box.data(), w[c].inbox, box.size() * 4).wait();
    for (uint32_t v : box) clean &= v == k::kMailEmpty;
  }
  std::snprintf(what, sizeof what, "greedy pick across %d cards: inboxes back to empty", C);
  expect(clean, what);
  for (int c = 0; c < C; ++c) {
    for (void *p : {(void *)w[c].inbox, (void *)w[c].step, (void *)w[c].error, scratch[c], (void *)d_t[c], (void *)d_l[c]}) fab.release(p);
  }
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    auto dev = xenithium::Device::open(loader, 0);
    // pick key: order checks on the host (the device uses the same function).
    expect(k::pick_key(1.f, 5) > k::pick_key(1.f, 6) && k::pick_key(-0.f, 3) == k::pick_key(0.f, 3) &&
               k::pick_key(-1.f, 0) < k::pick_key(-0.5f, 9) && k::pick_key(NAN, 0) < k::pick_key(-1e30f, 9) &&
               (k::pick_key(-3e38f, 0xFFFFFFFE) >> 32) != 0xFFFFFFFFu && static_cast<uint32_t>(k::pick_key(5.f, 0)) != 0xFFFFFFFFu,
           "pick key: value order, ties to lowest index, -0 == +0, NaN lowest, never the sentinel");
    const k::HcShape real{4, 2560, 320, 1e-6f}, small{4, 256, 64, 1e-6f};
    for (int rows : {1, 3, 4}) {
      test_hc(dev, real, rows, false, 10u + rows);
      test_hc(dev, real, rows, true, 20u + rows);
    }
    test_hc(dev, small, 4, true, 30u);
    test_hc(dev, k::HcShape{4, 512, 80, 1e-6f}, 2, true, 40u);  // fused path with a k tail only (no full group)
    // 16 / 17: the one-launch up + mix threshold; 5 / 16 / 17 / 30: a step's rows by owner (hc_mixer_at: 1-8 rows a card)
    for (int m : {1, 5, 16, 17, 30, 37, 300, 700}) test_hc_rows(dev, real, m, 50u + m);
    test_embed(dev);
    for (int rows : {1, 4, 8}) test_rms_norm(dev, 5120, rows, 60u + rows);
    test_rms_norm(dev, 256, 3, 70u);
    test_pick_one(dev);
    test_logprob(dev);
    test_logprob_rows(dev);
    if (loader.size() > 1) test_pick_cards(loader, 4, 2), test_pick_cards(loader, 16, 2);
    if (loader.size() >= 4) test_pick_cards(loader, 4, 4), test_pick_cards(loader, 16, 4);  // TP4
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
