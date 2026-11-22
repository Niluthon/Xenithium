// Gate for the DFlash 2 drafter's own kernels (dflash.hpp) and the plain Qwen3 RMS norm, against the
// host references that round where the device does. Exact gates: the top-16 keys (one card and across two cards through
// the mailbox) and the selector's drafts. Bound gates (2 bf16 ulps of max(|ref|, row rms), <= 1 % of elements off): the
// norm, q / k prep, attention; the convolution: <= 1 ulp, <= 0.1 % off (an fp32 fma contraction can move one rounding).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xenithium/kernels/drafter/dflash.hpp"
#include "xenithium/kernels/drafter/dspine.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/rt/device.hpp"
#include "xenithium/rt/fabric.hpp"

namespace {
namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;
using MK = xenithium::l0::MemKind;

int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}
int ulps(uint16_t a, uint16_t b) {
  auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
  return std::abs(lin(a) - lin(b));
}
struct Cmp {
  int max_ulp = 0;
  double scaled = 0;
  size_t off = 0, n = 0;
};
Cmp compare(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, size_t row) {
  Cmp c;
  c.n = a.size();
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
std::string text(const Cmp &c) {
  char b[160];
  std::snprintf(b, sizeof b, "max %d ulp, worst %.2e of max(|ref|, rms), %zu / %zu off (%.3f %%)", c.max_ulp, c.scaled, c.off, c.n, 100.0 * c.off / std::max<size_t>(1, c.n));
  return b;
}
bool ok2(const Cmp &c) { return c.scaled <= 1.0 / 128 && c.off * 100 <= c.n; }

std::vector<uint16_t> rnd_bf16(size_t n, unsigned seed, float sd) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, sd);
  std::vector<uint16_t> v(n);
  for (auto &x : v) x = k::bf16_bits(nd(rng));
  return v;
}
std::vector<float> rnd_f32(size_t n, unsigned seed, float sd) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, sd);
  std::vector<float> v(n);
  for (auto &x : v) x = nd(rng);
  return v;
}

template <typename T>
T *up(xenithium::Device &dev, const std::vector<T> &h) {
  T *d = dev.alloc<T>(std::max<size_t>(h.size(), 1024), MK::Device);
  dev.compute().memcpy(d, h.data(), h.size() * sizeof(T)).wait();
  return d;
}
template <typename T>
std::vector<T> down(xenithium::Device &dev, const T *d, size_t n) {
  std::vector<T> h(n);
  dev.compute().memcpy(h.data(), d, n * sizeof(T)).wait();
  return h;
}

// ---- plain RMS norm ----
void test_norm_plain(xenithium::Device &dev, int rows) {
  const int H = 5120;
  auto res0 = rnd_bf16(static_cast<size_t>(8) * H, 11u + rows, 2.f), add = rnd_bf16(res0.size(), 12u + rows, 1.f), w = rnd_bf16(H, 13u, 0.8f);
  auto *d_res = up(dev, res0), *d_add = up(dev, add), *d_w = up(dev, w);
  auto *d_xn = dev.alloc<uint16_t>(res0.size(), MK::Device);
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<uint16_t> rr = res0, xr(res0.size());
    k::rms_norm_plain_ref(rows, H, rr.data(), mode ? add.data() : nullptr, w.data(), 1e-6f, xr.data());
    dev.compute().memcpy(d_res, res0.data(), res0.size() * 2).wait();
    k::rms_norm_plain(dev.compute(), rows, H, reinterpret_cast<bf16 *>(d_res), mode ? reinterpret_cast<const bf16 *>(d_add) : nullptr, reinterpret_cast<const bf16 *>(d_w),
                      1e-6f, reinterpret_cast<bf16 *>(d_xn))
        .wait_and_throw();
    const size_t live = static_cast<size_t>(rows) * H;
    auto gx = down(dev, d_xn, live), gr = down(dev, d_res, res0.size());
    const Cmp c = compare(gx, std::vector<uint16_t>(xr.begin(), xr.begin() + static_cast<std::ptrdiff_t>(live)), H);
    expect(ok2(c) && gr == rr, "rms_norm_plain rows " + std::to_string(rows) + (mode ? " + bf16 add" : "") + ": residual " + (gr == rr ? "exact" : "DIFFERS") + ", xn " + text(c));
  }
  for (void *p : {(void *)d_res, (void *)d_add, (void *)d_w, (void *)d_xn}) dev.release(p);
}

// ---- conv ----
void test_conv(xenithium::Device &dev, int rows, int seg) {
  const int H = 5120, group = 16, G = H / group, ld = 4 * G;
  auto x = rnd_bf16(static_cast<size_t>(rows) * H, 21u + rows, 1.f), base = rnd_bf16(4 * static_cast<size_t>(H), 22u, 0.5f);
  for (int c = 0; c < H; ++c) base[c] = k::bf16_bits(1.f + 0.05f * k::bf16_value(base[c]));  // tap 0 near 1, like the checkpoint
  auto dyn = rnd_f32(static_cast<size_t>(rows) * ld, 23u + rows, 0.1f);
  auto *d_x = up(dev, x), *d_b = up(dev, base);
  auto *d_d = up(dev, dyn);
  auto *d_o = dev.alloc<uint16_t>(static_cast<size_t>(rows) * H, MK::Device);
  for (int which = 0; which < 2; ++which) {
    std::vector<uint16_t> ref(static_cast<size_t>(rows) * H);
    k::df_conv_ref(rows, seg, H, group, x.data(), dyn.data(), ld, which, base.data() + static_cast<size_t>(which) * 2 * H, ref.data());
    k::df_conv(dev.compute(), rows, seg, H, group, reinterpret_cast<const bf16 *>(d_x), d_d, ld, which, reinterpret_cast<const bf16 *>(d_b) + which * 2 * H,
               reinterpret_cast<bf16 *>(d_o))
        .wait_and_throw();
    const auto got = down(dev, d_o, ref.size());
    const Cmp c = compare(got, ref, H);
    expect(c.max_ulp <= 1 && c.off * 1000 <= c.n, "df_conv rows " + std::to_string(rows) + " segments of " + std::to_string(seg) + " side " + std::to_string(which) + ": " + text(c));
  }
  for (void *p : {(void *)d_x, (void *)d_b, (void *)d_d, (void *)d_o}) dev.release(p);
}

// ---- prep: host mirror of the rounding points ----
float r16(double v) { return k::bf16_value(k::bf16_bits(static_cast<float>(v))); }
void prep_head_ref(const float *y, const uint16_t *w, bool rope, const uint16_t *cs, const uint16_t *sn, uint16_t *out) {
  const int D = k::kDfDim;
  float o[k::kDfDim];
  double ss = 0;
  for (int d = 0; d < D; ++d) ss += static_cast<double>(r16(y[d])) * r16(y[d]);
  const float rr = 1.f / std::sqrt(static_cast<float>(ss / D) + 1e-6f);
  for (int d = 0; d < D; ++d) o[d] = w ? r16(static_cast<double>(k::bf16_value(w[d])) * r16(static_cast<double>(r16(y[d])) * rr)) : r16(y[d]);
  for (int d = 0; d < D; ++d) {
    if (!rope) {
      out[d] = k::bf16_bits(o[d]);
      continue;
    }
    const int i = d % (D / 2);
    const float c = k::bf16_value(cs[i]), s = k::bf16_value(sn[i]);
    const float rot = d < D / 2 ? -o[d + D / 2] : o[d - D / 2];
    out[d] = k::bf16_bits(r16(static_cast<double>(r16(static_cast<double>(o[d]) * c)) + r16(static_cast<double>(rot) * s)));
  }
}

struct Ring {
  std::vector<uint16_t> k, v;
};

void test_prep_attn(xenithium::Device &dev, const std::vector<int> &starts, int sr, unsigned seed, bool causal = false, int window = 2048) {
  const int hq = 16, hkv = 4, D = k::kDfDim, R = window + 512, slots = 2, P = window > 2048 ? 16384 : 8192;
  const int ns = static_cast<int>(starts.size()), rows = ns * sr;
  std::mt19937 rng(seed);
  // rope tables for P positions (any finite values serve the gate)
  auto cs = rnd_bf16(static_cast<size_t>(P) * (D / 2), seed + 1, 0.7f), sn = rnd_bf16(cs.size(), seed + 2, 0.7f);
  auto qn = rnd_bf16(D, seed + 3, 0.5f), kn = rnd_bf16(D, seed + 4, 0.5f);
  for (auto &x : qn) x = k::bf16_bits(1.f + k::bf16_value(x));
  for (auto &x : kn) x = k::bf16_bits(1.f + k::bf16_value(x));
  // ring: every slot fully written with random K / V
  Ring ring{rnd_bf16(static_cast<size_t>(slots) * R * hkv * D, seed + 5, 1.f), rnd_bf16(static_cast<size_t>(slots) * R * hkv * D, seed + 6, 1.f)};
  auto yq = rnd_f32(static_cast<size_t>(rows) * hq * D, seed + 7, 1.f), ykv = rnd_f32(static_cast<size_t>(rows) * 2 * hkv * D, seed + 8, 1.f);
  auto ykc = rnd_f32(static_cast<size_t>(rows) * 2 * hkv * D, seed + 9, 1.f);
  std::vector<int32_t> rec(static_cast<size_t>(k::kMaxSegs) * k::kSegRec, 0);
  for (int i = 0; i < ns; ++i) {
    rec[static_cast<size_t>(i) * k::kSegRec + k::kDfRecStart] = starts[static_cast<size_t>(i)];
    rec[static_cast<size_t>(i) * k::kSegRec + k::kDfRecCtx] = starts[static_cast<size_t>(i)] - sr;  // the verify's rows before the block
    rec[static_cast<size_t>(i) * k::kSegRec + k::kDfRecSlot] = i % slots;
  }
  auto *d_cs = up(dev, cs), *d_sn = up(dev, sn), *d_qn = up(dev, qn), *d_kn = up(dev, kn), *d_rk = up(dev, ring.k), *d_rv = up(dev, ring.v);
  auto *d_yq = up(dev, yq), *d_ykv = up(dev, ykv), *d_ykc = up(dev, ykc);
  auto *d_rec = up(dev, rec);
  auto *d_qo = dev.alloc<uint16_t>(static_cast<size_t>(rows) * hq * D, MK::Device);
  auto *d_kb = dev.alloc<uint16_t>(static_cast<size_t>(rows) * hkv * D, MK::Device);
  auto *d_vb = dev.alloc<uint16_t>(static_cast<size_t>(rows) * hkv * D, MK::Device);
  auto *d_part = dev.alloc<float>(k::df_attn_part_floats(rows, sr, hq, window), MK::Device);
  auto *d_out = dev.alloc<uint16_t>(static_cast<size_t>(rows) * hq * D, MK::Device);
  auto &q = dev.compute();
  const k::DfRope rope{reinterpret_cast<const bf16 *>(d_cs), reinterpret_cast<const bf16 *>(d_sn), P};
  const k::DfRing dr{reinterpret_cast<bf16 *>(d_rk), reinterpret_cast<bf16 *>(d_rv), R, hkv};
  // context rows first (the verify's rows into the ring), then the block
  k::df_prep(q, rows, sr, d_rec, hq, hkv, nullptr, d_ykc, reinterpret_cast<const bf16 *>(d_qn), reinterpret_cast<const bf16 *>(d_kn), 1e-6f, rope, nullptr, nullptr,
             nullptr, dr);
  k::df_prep(q, rows, sr, d_rec, hq, hkv, d_yq, d_ykv, reinterpret_cast<const bf16 *>(d_qn), reinterpret_cast<const bf16 *>(d_kn), 1e-6f, rope,
             reinterpret_cast<bf16 *>(d_qo), reinterpret_cast<bf16 *>(d_kb), reinterpret_cast<bf16 *>(d_vb), dr);
  k::df_attn(q, rows, sr, d_rec, hq, hkv, reinterpret_cast<const bf16 *>(d_qo), reinterpret_cast<const bf16 *>(d_kb), reinterpret_cast<const bf16 *>(d_vb), dr, window,
             d_part, reinterpret_cast<bf16 *>(d_out), causal)
      .wait_and_throw();
  // host: prep of both kinds, ring update, attention per segment
  std::vector<uint16_t> qo(static_cast<size_t>(rows) * hq * D), kb(static_cast<size_t>(rows) * hkv * D), vb(kb.size());
  for (int r = 0; r < rows; ++r) {
    const int seg = r / sr, j = r % sr, slot = seg % slots;
    const int pb = starts[static_cast<size_t>(seg)] + j, pc = starts[static_cast<size_t>(seg)] - sr + j;
    for (int h = 0; h < hq; ++h)
      prep_head_ref(&yq[(static_cast<size_t>(r) * hq + h) * D], qn.data(), true, &cs[static_cast<size_t>(pb) * (D / 2)], &sn[static_cast<size_t>(pb) * (D / 2)],
                    &qo[(static_cast<size_t>(r) * hq + h) * D]);
    for (int h = 0; h < hkv; ++h) {
      prep_head_ref(&ykv[static_cast<size_t>(r) * 2 * hkv * D + h * D], kn.data(), true, &cs[static_cast<size_t>(pb) * (D / 2)], &sn[static_cast<size_t>(pb) * (D / 2)],
                    &kb[(static_cast<size_t>(r) * hkv + h) * D]);
      prep_head_ref(&ykv[static_cast<size_t>(r) * 2 * hkv * D + (hkv + h) * D], nullptr, false, nullptr, nullptr, &vb[(static_cast<size_t>(r) * hkv + h) * D]);
      const size_t at = ((static_cast<size_t>(slot) * R + static_cast<size_t>(pc % R)) * hkv + h) * D;
      prep_head_ref(&ykc[static_cast<size_t>(r) * 2 * hkv * D + h * D], kn.data(), true, &cs[static_cast<size_t>(pc) * (D / 2)], &sn[static_cast<size_t>(pc) * (D / 2)],
                    &ring.k[at]);
      prep_head_ref(&ykc[static_cast<size_t>(r) * 2 * hkv * D + (hkv + h) * D], nullptr, false, nullptr, nullptr, &ring.v[at]);
    }
  }
  const auto gq = down(dev, d_qo, qo.size()), gk = down(dev, d_kb, kb.size()), gv = down(dev, d_vb, vb.size());
  const auto grk = down(dev, d_rk, ring.k.size()), grv = down(dev, d_rv, ring.v.size());
  std::string tag = std::to_string(ns) + " x " + std::to_string(sr) + " rows at";
  for (int s0 : starts) tag += " " + std::to_string(s0);
  const Cmp cq = compare(gq, qo, D), ck = compare(gk, kb, D), cv = compare(gv, vb, D), crk = compare(grk, ring.k, D), crv = compare(grv, ring.v, D);
  expect(ok2(cq) && ok2(ck) && cv.off == 0, "df_prep block " + tag + ": q " + text(cq) + "; k " + text(ck) + "; v " + text(cv));
  expect(ok2(crk) && crv.off == 0, "df_prep context rows into the ring: k " + text(crk) + "; v " + text(crv) + " (untouched slots exact)");
  std::vector<uint16_t> ref(static_cast<size_t>(rows) * hq * D);
  for (int s = 0; s < ns; ++s) {
    const size_t so = static_cast<size_t>(s % slots) * R * hkv * D;
    k::df_attn_ref(sr, starts[static_cast<size_t>(s)], hq, hkv, &gq[static_cast<size_t>(s) * sr * hq * D], &gk[static_cast<size_t>(s) * sr * hkv * D],
                   &gv[static_cast<size_t>(s) * sr * hkv * D], &grk[so], &grv[so], R, window, &ref[static_cast<size_t>(s) * sr * hq * D], causal);
  }
  const auto got = down(dev, d_out, ref.size());
  const Cmp ca = compare(got, ref, D);
  expect(ok2(ca), "df_attn " + tag + " (window " + std::to_string(window) + ", ring " + std::to_string(R) + (causal ? ", causal block" : "") + "): " + text(ca));
  for (void *p : {(void *)d_cs, (void *)d_sn, (void *)d_qn, (void *)d_kn, (void *)d_rk, (void *)d_rv, (void *)d_yq, (void *)d_ykv, (void *)d_ykc, (void *)d_rec,
                  (void *)d_qo, (void *)d_kb, (void *)d_vb, (void *)d_part, (void *)d_out})
    dev.release(p);
}

// ---- DSpine: the bridge kernels and the selector against their host references ----
void test_dspine(xenithium::Device &dev, int segs, int sr, int sub_k) {
  const int H = 5120, C = k::kDsCode, Rk = k::kDsRank, rows = segs * sr, V = 4096, n = sub_k > 0 ? sub_k + 128 : V;
  const std::string tag = std::to_string(segs) + " x " + std::to_string(sr) + " rows";
  auto &q = dev.compute();
  // bridge in (+ add) and out
  auto res0 = rnd_bf16(static_cast<size_t>(rows) * H, 31u + sr, 2.f), add = rnd_bf16(res0.size(), 32u + sr, 1.f);
  auto *d_res = up(dev, res0), *d_add = up(dev, add);
  auto *d_rms = dev.alloc<float>(64, MK::Device);
  auto *d_xr = dev.alloc<uint16_t>(res0.size(), MK::Device);
  k::ds_bridge_in(q, rows, H, reinterpret_cast<bf16 *>(d_res), reinterpret_cast<const bf16 *>(d_add), 1e-6f, d_rms, reinterpret_cast<bf16 *>(d_xr)).wait_and_throw();
  std::vector<uint16_t> rr = res0, xr(res0.size());
  std::vector<float> rms(static_cast<size_t>(rows));
  k::ds_bridge_in_ref(rows, H, rr.data(), add.data(), 1e-6f, rms.data(), xr.data());
  const auto g_res = down(dev, d_res, rr.size()), g_xr = down(dev, d_xr, xr.size());
  const auto g_rms = down(dev, d_rms, static_cast<size_t>(rows));
  double rerr = 0;
  for (int r = 0; r < rows; ++r) rerr = std::max(rerr, std::fabs(g_rms[static_cast<size_t>(r)] / rms[static_cast<size_t>(r)] - 1.0));
  const Cmp cx = compare(g_xr, xr, H);
  expect(g_res == rr && rerr < 1e-5 && cx.max_ulp <= 1 && cx.off * 100 <= cx.n,
         "ds_bridge_in " + tag + ": raw " + (g_res == rr ? "exact" : "DIFFERS") + ", rms rel " + std::to_string(rerr) + ", xr " + text(cx));
  auto delta = rnd_f32(static_cast<size_t>(rows) * H, 33u + sr, 0.05f);
  auto *d_delta = up(dev, delta);
  k::ds_bridge_out(q, rows, sr, H, reinterpret_cast<bf16 *>(d_res), d_rms, d_delta).wait_and_throw();
  std::vector<uint16_t> ro = g_res;
  k::ds_bridge_out_ref(rows, sr, H, ro.data(), g_rms.data(), delta.data());
  const auto g_ro = down(dev, d_res, ro.size());
  expect(g_ro == ro, "ds_bridge_out " + tag + ": " + (g_ro == ro ? "exact" : "DIFFERS") + " (anchor rows keep raw)");
  // codes (an anchor from a codebook)
  auto cb = rnd_bf16(static_cast<size_t>(V) * C, 34u, 0.03f);
  auto pj = rnd_f32(static_cast<size_t>(rows) * C, 35u + sr, 1.f);
  std::vector<int32_t> anc(static_cast<size_t>(segs));
  for (int i = 0; i < segs; ++i) anc[static_cast<size_t>(i)] = 17 + 1000 * i;
  auto *d_cb = up(dev, cb);
  auto *d_pj = up(dev, pj);
  auto *d_anc = up(dev, anc);
  auto *d_code = dev.alloc<uint16_t>(static_cast<size_t>(rows) * C, MK::Device);
  k::ds_codes(q, rows, sr, d_pj, d_anc, reinterpret_cast<const bf16 *>(d_cb), 1e-6f, reinterpret_cast<bf16 *>(d_code)).wait_and_throw();
  std::vector<uint16_t> code(static_cast<size_t>(rows) * C);
  k::ds_codes_ref(rows, sr, pj.data(), anc.data(), cb.data(), 1e-6f, code.data());
  const Cmp cc = compare(down(dev, d_code, code.size()), code, C);
  expect(cc.max_ulp <= 1 && cc.off * 100 <= cc.n, "ds_codes " + tag + ": " + text(cc));
  // message
  auto packed = rnd_f32(static_cast<size_t>(rows) * 3 * Rk, 36u + sr, 1.5f);
  auto bias = rnd_bf16(Rk, 37u, 0.5f);
  auto *d_pk = up(dev, packed);
  auto *d_bias = up(dev, bias);
  auto *d_msg = dev.alloc<uint16_t>(static_cast<size_t>(rows) * Rk, MK::Device);
  k::ds_message(q, rows, sr, d_pk, reinterpret_cast<const bf16 *>(d_bias), reinterpret_cast<bf16 *>(d_msg)).wait_and_throw();
  std::vector<uint16_t> msg(static_cast<size_t>(rows) * Rk);
  k::ds_message_ref(rows, sr, packed.data(), bias.data(), msg.data());
  const Cmp cm = compare(down(dev, d_msg, msg.size()), msg, Rk);
  expect(cm.max_ulp <= 1 && cm.off * 1000 <= cm.n, "ds_message " + tag + ": " + text(cm));
  // select: tables over n candidate indices, keys with distinct indices per row
  auto tsv = rnd_bf16(static_cast<size_t>(n) * 2 * Rk, 38u, 1.f), tg = rnd_bf16(static_cast<size_t>(n) * Rk, 39u, 0.05f);
  std::vector<uint64_t> keys(static_cast<size_t>(rows) * k::kDfTopK);
  std::mt19937 rng(40u + sr);
  for (int r = 0; r < rows; ++r) {
    std::vector<int> ids(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<uint64_t> kr;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int b = 0; b < k::kDfTopK; ++b) kr.push_back(k::df_key(nd(rng), static_cast<uint32_t>(ids[static_cast<size_t>(b)])));
    std::sort(kr.rbegin(), kr.rend());
    std::copy(kr.begin(), kr.end(), keys.begin() + static_cast<std::ptrdiff_t>(r) * k::kDfTopK);
  }
  auto *d_keys = up(dev, keys);
  auto *d_tsv = up(dev, tsv), *d_tg = up(dev, tg);
  auto *d_dr = dev.alloc<int32_t>(256, MK::Device);
  for (int noisy = 0; noisy < 2; ++noisy) {
    std::vector<k::DraftNoiseRec> nz(static_cast<size_t>(segs));
    for (int i = 0; i < segs; ++i) nz[static_cast<size_t>(i)] = noisy ? k::DraftNoiseRec{0x1234u + static_cast<uint64_t>(i), 77u, 1.4f} : k::DraftNoiseRec{};
    auto *d_nz = up(dev, nz);
    k::ds_select(q, segs, sr, d_keys, d_pk, reinterpret_cast<const bf16 *>(d_tsv), reinterpret_cast<const bf16 *>(d_tg), reinterpret_cast<const bf16 *>(d_bias),
                 k::DsTail{reinterpret_cast<const bf16 *>(d_res), H, 1e-6f, d_rms, reinterpret_cast<const bf16 *>(d_msg)}, sub_k, 3840, d_dr, d_nz)
        .wait_and_throw();
    const auto got = down(dev, d_dr, static_cast<size_t>(segs) * (sr - 1));
    std::vector<int32_t> ref(got.size());
    for (int i = 0; i < segs; ++i)
      k::ds_select_ref(sr, keys.data() + static_cast<size_t>(i) * sr * k::kDfTopK, packed.data() + static_cast<size_t>(i) * sr * 3 * Rk, tsv.data(), tg.data(),
                       bias.data(), g_ro.data() + static_cast<size_t>(i) * sr * H, H, 1e-6f, g_rms.data() + static_cast<size_t>(i) * sr, msg.data() + static_cast<size_t>(i) * sr * Rk,
                       sub_k, 3840, ref.data() + static_cast<size_t>(i) * (sr - 1), &nz[static_cast<size_t>(i)]);
    expect(got == ref, "ds_select " + tag + (sub_k ? " (draft vocabulary)" : "") + (noisy ? ", coupled noise" : ", greedy") + ": drafts " + (got == ref ? "==" : "!=") +
                           " the host walk");
    dev.release(d_nz);
  }
  {  // DSpine's best-first tree == its host reference; the first node == ds_select's first greedy draft
    auto *d_tr = dev.alloc<int32_t>(static_cast<size_t>(segs) * k::kDfTreeInts, MK::Device);
    k::ds_select(q, segs, sr, d_keys, d_pk, reinterpret_cast<const bf16 *>(d_tsv), reinterpret_cast<const bf16 *>(d_tg), reinterpret_cast<const bf16 *>(d_bias),
                 k::DsTail{reinterpret_cast<const bf16 *>(d_res), H, 1e-6f, d_rms, reinterpret_cast<const bf16 *>(d_msg)}, sub_k, 3840, d_dr, nullptr)
        .wait_and_throw();
    const auto chain = down(dev, d_dr, static_cast<size_t>(segs) * (sr - 1));
    int bad = 0, branched = 0, cut_short = 0, first_ok = 0, trees = 0;
    for (const float it : {1.0f, 0.5f})
    for (const float cut : {0.f, 0.3f})
      for (int budget = 1; budget < k::kTreeRows; ++budget) {
        k::ds_tree(q, segs, sr, d_keys, d_pk, reinterpret_cast<const bf16 *>(d_tsv), reinterpret_cast<const bf16 *>(d_tg), reinterpret_cast<const bf16 *>(d_bias),
                   k::DsTail{reinterpret_cast<const bf16 *>(d_res), H, 1e-6f, d_rms, reinterpret_cast<const bf16 *>(d_msg)}, sub_k, 3840, budget, cut, d_tr, it)
            .wait_and_throw();
        const auto got = down(dev, d_tr, static_cast<size_t>(segs) * k::kDfTreeInts);
        for (int i = 0; i < segs; ++i) {
          int32_t want[k::kDfTreeInts];
          k::ds_tree_ref(sr, keys.data() + static_cast<size_t>(i) * sr * k::kDfTopK, packed.data() + static_cast<size_t>(i) * sr * 3 * Rk, tsv.data(), tg.data(), bias.data(),
                         g_ro.data() + static_cast<size_t>(i) * sr * H, H, 1e-6f, g_rms.data() + static_cast<size_t>(i) * sr, msg.data() + static_cast<size_t>(i) * sr * Rk,
                         sub_k, 3840, budget, cut, want, it);
          const int32_t *g = got.data() + static_cast<size_t>(i) * k::kDfTreeInts;
          const bool df = !std::equal(want, want + k::kDfTreeInts, g);
          if (df && bad < 4) {
            std::printf("  ds_tree seg %d budget %d cut %.2f:\n   got ", i, budget, cut);
            for (int x = 0; x < k::kDfTreeInts; ++x) std::printf(" %d", g[x]);
            std::printf("\n   want");
            for (int x = 0; x < k::kDfTreeInts; ++x) std::printf(" %d", want[x]);
            std::printf("\n");
          }
          bad += df;
          ++trees;
          const int nn = g[0];
          cut_short += nn < budget;
          bool br = false;
          for (int x = 1; x < nn; ++x) br |= g[1 + k::kTreeRows + x] != x;
          branched += br;
          first_ok += nn == 0 || g[1] == chain[static_cast<size_t>(i) * (sr - 1)];
        }
      }
    expect(bad == 0 && branched > 0 && first_ok == trees && (sr < 8 || cut_short > 0),
           "ds_tree " + tag + (sub_k ? " (draft vocabulary)" : "") + ", budgets 1..15, cut 0 / 0.3, temperature 1 / 2: " + std::to_string(trees) + " trees == the host reference (" +
               std::to_string(bad) + " differ; " + std::to_string(branched) + " branch, " + std::to_string(cut_short) + " cut short), the first node == ds_select's first draft");
    dev.release(d_tr);
  }
  for (void *pp : {(void *)d_res, (void *)d_add, (void *)d_rms, (void *)d_xr, (void *)d_delta, (void *)d_cb, (void *)d_pj, (void *)d_anc, (void *)d_code, (void *)d_pk,
                   (void *)d_bias, (void *)d_msg, (void *)d_keys, (void *)d_tsv, (void *)d_tg, (void *)d_dr})
    dev.release(pp);
}

// ---- top-k ----
std::vector<float> logits(int rows, int n, unsigned seed) {
  auto v = rnd_f32(static_cast<size_t>(rows) * n, seed, 3.f);
  for (int r = 0; r < rows; ++r) {  // plant exact ties at the top (the lower index must win) across chunks
    float *x = v.data() + static_cast<size_t>(r) * n;
    x[17] = x[n - 3] = x[5000] = 40.f;
    x[9000] = 39.75f;
  }
  return v;
}

void test_topk_one(xenithium::Device &dev, int rows, int n) {
  auto lg = logits(rows, n, 31u + rows);
  auto *d_l = up(dev, lg);
  void *scratch = dev.alloc<uint8_t>(k::df_topk_scratch_bytes(n), MK::Device);
  auto *d_k = dev.alloc<uint64_t>(static_cast<size_t>(rows) * k::kDfTopK, MK::Device);
  k::df_topk(dev.compute(), d_l, n, rows, n, 1000, scratch, nullptr, d_k).wait_and_throw();
  const auto got = down(dev, d_k, static_cast<size_t>(rows) * k::kDfTopK);
  bool same = true;
  for (int r = 0; r < rows; ++r) {
    uint64_t want[k::kDfTopK];
    k::df_topk_ref(lg.data() + static_cast<size_t>(r) * n, n, 1000, want);
    same &= std::equal(want, want + k::kDfTopK, got.begin() + static_cast<std::ptrdiff_t>(r) * k::kDfTopK);
  }
  expect(same && k::df_key_index(got[0]) == 1017 && k::df_key_value(got[0]) == 40.f,
         "df_topk one card, " + std::to_string(rows) + " rows x " + std::to_string(n) + ": the 16 keys == the host's, best first, ties to the lower index");
  for (void *p : {(void *)d_l, scratch, (void *)d_k}) dev.release(p);
}

// C = 2 or 4 cards (four = two hops); card c holds columns [c n, (c + 1) n)
void test_topk_cards(const xenithium::l0::Loader &loader, int C) {
  std::vector<size_t> devs;
  for (int c = 0; c < C; ++c) devs.push_back(static_cast<size_t>(c));
  xenithium::Fabric fab(loader, devs);
  const int n = 248320 / C, rows = 8, steps = 6;
  const size_t words = C == 4 ? k::kDfTopkWireWords4 : k::kDfTopkWireWords;
  k::DfTopkWire w[4];
  void *scratch[4];
  uint64_t *d_k[4];
  float *d_l[4];
  for (int c = 0; c < C; ++c) {
    w[c].inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(words * 4));
    w[c].step = static_cast<int32_t *>(fab.alloc(c, 4096));
    w[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    scratch[c] = fab.alloc(c, k::df_topk_scratch_bytes(n));
    d_k[c] = static_cast<uint64_t *>(fab.alloc(c, static_cast<size_t>(steps) * rows * k::kDfTopK * 8));
    d_l[c] = static_cast<float *>(fab.alloc(c, static_cast<size_t>(steps) * rows * n * 4));
  }
  for (int c = 0; c < C; ++c) {
    w[c].peer_inbox = w[c ^ 1].inbox;
    if (C == 4) w[c].peer2_inbox = w[c ^ 2].inbox;
    k::df_topk_wire_init(fab.compute(c), w[c].inbox, words);
    fab.compute(c).memset(w[c].step, 0, 4);
    fab.compute(c).memset(w[c].error, 0, 4).wait();
  }
  std::vector<std::vector<float>> lg(static_cast<size_t>(C));
  std::vector<uint64_t> want(static_cast<size_t>(steps) * rows * k::kDfTopK);
  for (int st = 0; st < steps; ++st) {
    std::vector<std::vector<float>> part;
    part.push_back(logits(rows, n, 200u + st));
    for (int c = 1; c < C; ++c) part.push_back(rnd_f32(static_cast<size_t>(rows) * n, 700u + 31u * c + st, 3.f));
    for (int r = 0; r < rows; ++r) {
      if (st % 2)  // cross-card ties: the lower global indexes first (every card past 0 holds the same high logit)
        for (int c = 1; c < C; ++c) part[static_cast<size_t>(c)][static_cast<size_t>(r) * n + 3 + c] = part[static_cast<size_t>(c)][static_cast<size_t>(r) * n + 7777 * c] = 40.f;
      std::vector<float> full;
      for (int c = 0; c < C; ++c) full.insert(full.end(), part[static_cast<size_t>(c)].begin() + static_cast<std::ptrdiff_t>(r) * n, part[static_cast<size_t>(c)].begin() + static_cast<std::ptrdiff_t>(r + 1) * n);
      k::df_topk_ref(full.data(), C * n, 0, &want[(static_cast<size_t>(st) * rows + r) * k::kDfTopK]);
    }
    for (int c = 0; c < C; ++c) lg[static_cast<size_t>(c)].insert(lg[static_cast<size_t>(c)].end(), part[static_cast<size_t>(c)].begin(), part[static_cast<size_t>(c)].end());
  }
  for (int c = 0; c < C; ++c) fab.compute(c).memcpy(d_l[c], lg[static_cast<size_t>(c)].data(), lg[static_cast<size_t>(c)].size() * 4).wait();
  for (int st = 0; st < steps; ++st)  // back to back, no host wait: the parity keeps the mailbox slots apart
    for (int c = 0; c < C; ++c)
      k::df_topk(fab.compute(c), d_l[c] + static_cast<size_t>(st) * rows * n, n, rows, n, c * n, scratch[c], &w[c], d_k[c] + static_cast<size_t>(st) * rows * k::kDfTopK);
  std::vector<uint64_t> got[4];
  uint32_t err[4] = {0, 0, 0, 0};
  bool all = true, noerr = true;
  for (int c = 0; c < C; ++c) {
    fab.compute(c).wait_and_throw();
    got[c].resize(want.size());
    fab.compute(c).memcpy(got[c].data(), d_k[c], want.size() * 8).wait();
    fab.compute(c).memcpy(&err[c], w[c].error, 4).wait();
    all &= got[c] == want;
    noerr &= err[c] == 0;
  }
  const std::string tag = "df_topk across " + std::to_string(C) + " cards";
  expect(all, tag + ", 6 steps x 8 rows: every card == the host's 16 best of all columns (ties included)");
  expect(noerr, tag + ": no mailbox timeout");
  bool clean = true;
  std::vector<uint32_t> box(words);
  for (int c = 0; c < C; ++c) {
    fab.compute(c).memcpy(box.data(), w[c].inbox, box.size() * 4).wait();
    for (uint32_t v : box) clean &= v == k::kMailEmpty;
  }
  expect(clean, tag + ": inboxes back to empty");
  for (int c = 0; c < C; ++c)
    for (void *p : {(void *)w[c].inbox, (void *)w[c].step, (void *)w[c].error, scratch[c], (void *)d_k[c], (void *)d_l[c]}) fab.release(p);
}

// ---- selector ----
// ---- the deferred context's stash and record ----
void test_stash(xenithium::Device &dev) {
  const int cols = 12800, segs = 5, sr = 3, B = 16;
  auto src = rnd_bf16(static_cast<size_t>(segs) * sr * cols, 51u, 1.f);
  std::vector<uint16_t> dst0 = rnd_bf16(static_cast<size_t>(8) * B * cols, 52u, 1.f);
  auto *d_src = up(dev, src), *d_dst = up(dev, dst0);
  const int64_t rows[k::kMaxSegs] = {3 * B + 2, 0 * B + 13, 7 * B + 0, 1 * B + 5, 5 * B + 9, 0, 0, 0};
  k::df_ctx_stash(dev.compute(), segs, sr, cols, reinterpret_cast<const k::df_bf16 *>(d_src), reinterpret_cast<k::df_bf16 *>(d_dst), rows).wait();
  auto got = down(dev, d_dst, dst0.size());
  std::vector<uint16_t> want = dst0;
  for (int i = 0; i < segs; ++i)
    for (int r = 0; r < sr; ++r)
      std::copy_n(src.begin() + static_cast<size_t>(i * sr + r) * cols, cols, want.begin() + static_cast<size_t>(rows[i] + r) * cols);
  expect(got == want, "df_ctx_stash: 5 segments x 3 rows into their slots' rows, every other row untouched");
  auto *d_rec = dev.alloc<int32_t>(1024, MK::Device);
  dev.compute().memset(d_rec, 0x7f, 4096).wait();
  k::df_rec_set(dev.compute(), d_rec, 123456, 3).wait();
  auto rec = down(dev, d_rec, k::kSegRec + 1);
  bool ok = rec[k::kSegRec] == 0x7f7f7f7f;
  for (int w = 0; w < k::kSegRec; ++w) ok &= rec[static_cast<size_t>(w)] == (w == k::kDfRecCtx ? 123456 : w == k::kDfRecSlot ? 3 : 0);
  expect(ok, "df_rec_set: a context-only record, the words past it untouched");
}

void test_select(xenithium::Device &dev, int segs, int sr, int sub_k) {
  const int V = 4096, rank = 256, rows = segs * sr, sub_s0 = 3968;  // sub: virtual [0, sub_k) + 128 specials at sub_s0
  auto pred = rnd_bf16(static_cast<size_t>(V) * rank, 41u, 1.f), succ = rnd_bf16(pred.size(), 42u, 1.f);
  auto hp = rnd_f32(static_cast<size_t>(rows) * rank, 43u + rows, 0.5f);
  const int n = sub_k > 0 ? sub_k + 128 : V;
  auto lg = rnd_f32(static_cast<size_t>(rows) * n, 44u + rows, 1.f);
  std::vector<uint64_t> keys(static_cast<size_t>(rows) * k::kDfTopK);
  for (int r = 0; r < rows; ++r) k::df_topk_ref(lg.data() + static_cast<size_t>(r) * n, n, 0, &keys[static_cast<size_t>(r) * k::kDfTopK]);
  std::vector<int32_t> anchors(static_cast<size_t>(segs));
  for (int s = 0; s < segs; ++s) anchors[static_cast<size_t>(s)] = 100 + 37 * s;
  auto *d_p = up(dev, pred), *d_s = up(dev, succ);
  auto *d_h = up(dev, hp);
  auto *d_k = up(dev, keys);
  auto *d_a = up(dev, anchors);
  auto *d_o = dev.alloc<int32_t>(static_cast<size_t>(segs) * (sr - 1), MK::Device);
  k::df_select(dev.compute(), segs, sr, d_k, d_h, rank, reinterpret_cast<const bf16 *>(d_p), reinterpret_cast<const bf16 *>(d_s), d_a, sub_k, sub_s0, d_o).wait_and_throw();
  const auto got = down(dev, d_o, static_cast<size_t>(segs) * (sr - 1));
  std::vector<int32_t> want(got.size());
  for (int s = 0; s < segs; ++s)
    k::df_select_ref(sr, &keys[static_cast<size_t>(s) * sr * k::kDfTopK], &hp[static_cast<size_t>(s) * sr * rank], rank, pred.data(), succ.data(), anchors[static_cast<size_t>(s)],
                     sub_k, sub_s0, &want[static_cast<size_t>(s) * (sr - 1)]);
  // the selector must move the pick off the plain argmax somewhere (else it is untested)
  int moved = 0;
  for (int s = 0; s < segs; ++s)
    for (int j = 1; j < sr; ++j) {
      const auto v = static_cast<int32_t>(k::df_key_index(keys[(static_cast<size_t>(s) * sr + j) * k::kDfTopK]));
      moved += (sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v) != got[static_cast<size_t>(s) * (sr - 1) + j - 1];
    }
  expect(got == want && moved > 0, "df_select " + std::to_string(segs) + " x " + std::to_string(sr) + " rows" + (sub_k ? " (draft vocabulary)" : "") +
                                       ": drafts == the host walk (" + std::to_string(moved) + " picks off the plain argmax)");
  // coupled drafts - even segments carry a noise record, odd ones none (argmax: the bits above). T 20: the
  // random selector terms here are ~10x a real checkpoint's, a small T would move no pick (the test would not test it)
  std::vector<k::DraftNoiseRec> nz(static_cast<size_t>(segs));
  for (int s = 0; s < segs; s += 2) nz[static_cast<size_t>(s)] = k::DraftNoiseRec{0x5eedull + static_cast<uint64_t>(s), 1000u + 7u * static_cast<uint64_t>(s), 1.f / 20.f};
  auto *d_n = dev.alloc<k::DraftNoiseRec>(nz.size(), MK::Device);
  dev.compute().memcpy(d_n, nz.data(), nz.size() * sizeof(k::DraftNoiseRec)).wait();
  k::df_select(dev.compute(), segs, sr, d_k, d_h, rank, reinterpret_cast<const bf16 *>(d_p), reinterpret_cast<const bf16 *>(d_s), d_a, sub_k, sub_s0, d_o, d_n).wait_and_throw();
  const auto got_n = down(dev, d_o, static_cast<size_t>(segs) * (sr - 1));
  std::vector<int32_t> want_n(got.size());
  for (int s = 0; s < segs; ++s)
    k::df_select_ref(sr, &keys[static_cast<size_t>(s) * sr * k::kDfTopK], &hp[static_cast<size_t>(s) * sr * rank], rank, pred.data(), succ.data(), anchors[static_cast<size_t>(s)],
                     sub_k, sub_s0, &want_n[static_cast<size_t>(s) * (sr - 1)], &nz[static_cast<size_t>(s)]);
  int off = 0;
  bool argmax_kept = true;
  for (int s = 0; s < segs; ++s)
    for (int j = 0; j < sr - 1; ++j) {
      const size_t i = static_cast<size_t>(s) * (sr - 1) + j;
      if (s % 2) argmax_kept &= got_n[i] == got[i];
      else off += got_n[i] != got[i];
    }
  expect(got_n == want_n && argmax_kept && off > 0, "df_select " + std::to_string(segs) + " x " + std::to_string(sr) + " rows, coupled noise on even segments: drafts == the host walk (" +
                                                        std::to_string(off) + " picks moved), odd segments keep the argmax picks");
  dev.release(d_n);
  for (void *p : {(void *)d_p, (void *)d_s, (void *)d_h, (void *)d_k, (void *)d_a, (void *)d_o}) dev.release(p);
}

// DDTree: the best-first tree == the host reference (nodes, tokens, parents) per segment, budgets 1..7 and a
// cut; its first node == the selector's first draft; some trees branch, some cut stops early
void test_tree(xenithium::Device &dev, int segs, int sr, int sub_k) {
  const int V = 4096, rank = 256, rows = segs * sr, sub_s0 = 3968;
  auto pred = rnd_bf16(static_cast<size_t>(V) * rank, 61u, 0.3f), succ = rnd_bf16(pred.size(), 62u, 0.3f);
  auto hp = rnd_f32(static_cast<size_t>(rows) * rank, 63u + rows, 0.5f);
  const int n = sub_k > 0 ? sub_k + 128 : V;
  auto lg = rnd_f32(static_cast<size_t>(rows) * n, 64u + rows, 2.f);
  std::vector<uint64_t> keys(static_cast<size_t>(rows) * k::kDfTopK);
  for (int r = 0; r < rows; ++r) k::df_topk_ref(lg.data() + static_cast<size_t>(r) * n, n, 0, &keys[static_cast<size_t>(r) * k::kDfTopK]);
  std::vector<int32_t> anchors(static_cast<size_t>(segs));
  for (int s = 0; s < segs; ++s) anchors[static_cast<size_t>(s)] = 200 + 41 * s;
  auto *d_p = up(dev, pred), *d_s = up(dev, succ);
  auto *d_h = up(dev, hp);
  auto *d_k = up(dev, keys);
  auto *d_a = up(dev, anchors);
  auto *d_o = dev.alloc<int32_t>(static_cast<size_t>(segs) * k::kDfTreeInts, MK::Device);
  auto *d_c = dev.alloc<int32_t>(static_cast<size_t>(segs) * (sr - 1), MK::Device);
  k::df_select(dev.compute(), segs, sr, d_k, d_h, rank, reinterpret_cast<const bf16 *>(d_p), reinterpret_cast<const bf16 *>(d_s), d_a, sub_k, sub_s0, d_c).wait_and_throw();
  const auto chain = down(dev, d_c, static_cast<size_t>(segs) * (sr - 1));
  int bad = 0, branched = 0, cut_short = 0, first_ok = 0, trees = 0;
  for (const float it : {1.0f, 0.5f})  // the tree's temperature (1 / 2)
  for (const float cut : {0.f, 0.3f})
    for (int budget = 1; budget < k::kTreeRows; ++budget) {
      k::df_tree(dev.compute(), segs, sr, d_k, d_h, rank, reinterpret_cast<const bf16 *>(d_p), reinterpret_cast<const bf16 *>(d_s), d_a, sub_k, sub_s0, budget, cut, d_o, it).wait_and_throw();
      const auto got = down(dev, d_o, static_cast<size_t>(segs) * k::kDfTreeInts);
      for (int s = 0; s < segs; ++s) {
        int32_t want[k::kDfTreeInts];
        k::df_tree_ref(sr, &keys[static_cast<size_t>(s) * sr * k::kDfTopK], &hp[static_cast<size_t>(s) * sr * rank], rank, pred.data(), succ.data(), anchors[static_cast<size_t>(s)],
                       sub_k, sub_s0, budget, cut, want, it);
        const int32_t *g = got.data() + static_cast<size_t>(s) * k::kDfTreeInts;
        const bool df = !std::equal(want, want + k::kDfTreeInts, g);
        if (df && bad < 4) {
          std::printf("  tree seg %d budget %d cut %.2f:\n   got ", s, budget, cut);
          for (int i = 0; i < k::kDfTreeInts; ++i) std::printf(" %d", g[i]);
          std::printf("\n   want");
          for (int i = 0; i < k::kDfTreeInts; ++i) std::printf(" %d", want[i]);
          std::printf("\n");
        }
        bad += df;
        ++trees;
        const int nn = g[0];
        cut_short += nn < budget;
        bool br = false;
        for (int i = 1; i < nn; ++i) br |= g[1 + k::kTreeRows + i] != i;  // node i + 1's parent is not node i
        branched += br;
        first_ok += nn == 0 || g[1] == chain[static_cast<size_t>(s) * (sr - 1)];
      }
    }
  expect(bad == 0 && branched > 0 && cut_short > 0 && first_ok == trees, "df_tree " + std::to_string(segs) + " x " + std::to_string(sr) + " rows" + (sub_k ? " (draft vocabulary)" : "") +
                                                                  ", budgets 1..15, cut 0 / 0.3, temperature 1 / 2: " + std::to_string(trees) + " trees == the host reference (" + std::to_string(bad) +
                                                                  " differ; " + std::to_string(branched) + " branch, " + std::to_string(cut_short) +
                                                                  " cut short), the first node == the selector's first draft");
  // the preorder of every tree: parents before children, leaves before their parent's other children
  bool pre_ok = true;
  for (int b = 0; b < 64; ++b) {
    std::mt19937 g(static_cast<uint32_t>(b));
    const int nn = 1 + static_cast<int>(g() % k::kTreeRows);
    int jp[k::kTreeRows] = {-1}, node[k::kTreeRows], par[k::kTreeRows];
    for (int i = 1; i < nn; ++i) jp[i] = static_cast<int>(g() % static_cast<uint32_t>(i));
    k::tree_preorder(jp, nn, node, par);
    bool seen[k::kTreeRows] = {};
    for (int r = 0; r < nn; ++r) {
      seen[node[r]] = true;
      pre_ok &= r == 0 ? par[r] == -1 && node[r] == 0 : par[r] >= 0 && par[r] < r && node[par[r]] == jp[node[r]];
    }
    for (int i = 0; i < nn; ++i) pre_ok &= seen[i];
    // DFS preorder: row r's parent is row r - 1 or an ancestor of it
    for (int r = 1; r < nn; ++r) {
      int a = r - 1;
      while (a >= 0 && a != par[r]) a = par[a];
      pre_ok &= a == par[r];
    }
  }
  expect(pre_ok, "tree_preorder: 64 random trees -> depth-first rows, parents before children, every node once");
  for (void *p : {(void *)d_p, (void *)d_s, (void *)d_h, (void *)d_k, (void *)d_a, (void *)d_o, (void *)d_c}) dev.release(p);
}

}  // namespace

int main() {
  try {
    const auto loader = xenithium::l0::Loader::open();
    auto dev = xenithium::Device::open(loader, 0);
    expect(k::df_key(1.f, 5) > k::df_key(1.f, 6) && k::df_key_value(k::df_key(-2.5f, 3)) == -2.5f && k::df_key_index(k::df_key(7.f, 123456)) == 123456,
           "df_key: value order, ties to the lower index, value and index round-trip");
    for (int rows : {1, 4, 8}) test_norm_plain(dev, rows);
    test_conv(dev, 8, 8);
    test_conv(dev, 8, 4);
    test_conv(dev, 3, 0);
    test_prep_attn(dev, {5000}, 8, 1u);  // one 8-row block, the window wraps the ring
    test_prep_attn(dev, {100, 3000}, 4, 2u);  // two segments: one clipped at position 0
    test_prep_attn(dev, {9, 2600, 4097, 70}, 2, 3u);
    test_prep_attn(dev, {2}, 2, 4u);  // a context of 2 tokens
    test_prep_attn(dev, {5000}, 8, 5u, true);  // DSpine: causal inside the block
    test_prep_attn(dev, {9000, 300}, 4, 6u, true, 8192);  // DSpine's window 8,192 (one segment wraps the ring)
    test_prep_attn(dev, {3000}, 16, 7u, true);  // DSpine's 16-row block: two query tiles per kv head
    test_dspine(dev, 1, 8, 0);
    test_dspine(dev, 2, 4, 96);
    test_dspine(dev, 1, 16, 96);  // DSpine's whole 16-row block, trees of up to 15 nodes
    test_topk_one(dev, 8, 124160);
    test_topk_one(dev, 3, 49216);
    test_select(dev, 1, 8, 0);
    test_select(dev, 2, 4, 0);
    test_select(dev, 4, 2, 3840);
    test_tree(dev, 1, 8, 0);  // DDTree
    test_tree(dev, 2, 8, 3840);
    test_tree(dev, 3, 5, 0);
    test_stash(dev);
    if (loader.size() > 1) test_topk_cards(loader, 2);
    if (loader.size() > 3) test_topk_cards(loader, 4);
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
