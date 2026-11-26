// Kernel gate: every kernel bit-exact against its CPU reference.
// Carried kernels: the int8 DPAS tile, the int8 W8A8 M=4 GEMM (every split-K), the int4
// W4A16 M=4 GEMM (every split-K) and its gate|up packer.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <string>
#include <vector>

#include "xenithium/rt/device.hpp"
#include "xenithium/kernels/tile/dpas_int8.hpp"
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/gemm/int8_gemm.hpp"
#include "xenithium/kernels/attention/kv_copy.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

namespace {
namespace esimd = sycl::ext::intel::esimd;
namespace kernels = xenithium::kernels;

int g_failures = 0;

void expect(bool ok, const char *what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

template <typename T>
std::vector<T> bytes(size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::vector<T> out(count);
  for (auto &v : out) v = static_cast<T>(dist(rng));
  return out;
}

template <int RC>
bool tile_matches(xenithium::Device &dev) {
  constexpr int kN = xenithium::dpas::kExecWidth;
  constexpr int kK = xenithium::dpas::kKPerCall;
  const auto a = bytes<int8_t>(RC * kK, 5 + RC);
  const auto w = bytes<int8_t>(kN * kK, 17 + RC);
  std::vector<int8_t> w_vnni(xenithium::dpas::kBlockBytes);
  std::vector<int32_t> got(RC * kN, 0), want(RC * kN, 0);
  kernels::pack_weights_vnni(w.data(), kN, kK, w_vnni.data());
  kernels::gemm_int8_ref(a.data(), w.data(), RC, kN, kK, want.data());

  auto *d_a = dev.alloc<int8_t>(a.size(), xenithium::l0::MemKind::Device);
  auto *d_w = dev.alloc<int8_t>(w_vnni.size(), xenithium::l0::MemKind::Device);
  auto *d_c = dev.alloc<int32_t>(got.size(), xenithium::l0::MemKind::Device);
  dev.copy_queue().copy(d_a, a.data(), a.size(), true);
  dev.copy_queue().copy(d_w, w_vnni.data(), w_vnni.size(), true);

  dev.compute()
      .submit([&](sycl::handler &h) {
        h.single_task([=]() [[intel::sycl_explicit_simd]] {
          const esimd::simd<int8_t, RC * kK> a_tile(d_a);
          const esimd::simd<int8_t, xenithium::dpas::kBlockBytes> b_tile(d_w);
          esimd::simd<int32_t, RC * kN> acc(0);
          acc = xenithium::dpas::accumulate<RC>(acc, b_tile, a_tile);
          acc.copy_to(d_c);
        });
      })
      .wait_and_throw();
  dev.copy_queue().copy(got.data(), d_c, got.size() * sizeof(int32_t), true);

  dev.release(d_a);
  dev.release(d_w);
  dev.release(d_c);
  return std::equal(got.begin(), got.end(), want.begin());
}

}  // namespace

void test_pack_cat() {
  const int k = 256, n_a = 64;
  const size_t words = static_cast<size_t>(k / 8) * n_a;
  std::mt19937 rng(7);
  std::vector<uint32_t> a(words), b(words), cat(words * 2);
  for (auto &v : a) v = static_cast<uint32_t>(rng());
  for (auto &v : b) v = static_cast<uint32_t>(rng());
  for (int r = 0; r < k / 8; ++r) {
    std::copy_n(a.begin() + static_cast<size_t>(r) * n_a, n_a, cat.begin() + static_cast<size_t>(r) * 2 * n_a);
    std::copy_n(b.begin() + static_cast<size_t>(r) * n_a, n_a, cat.begin() + static_cast<size_t>(r) * 2 * n_a + n_a);
  }
  std::vector<uint8_t> want(words * 2 * 4), got(want.size());
  kernels::pack_w4_gptq(cat.data(), 2 * n_a, k, want.data());
  kernels::pack_w4_gptq_cat(a.data(), b.data(), n_a, 2 * n_a, k, got.data());
  expect(want == got, "pack_w4_gptq_cat == pack_w4_gptq of the concatenation");
}

// The i2 dense GEMM (the MTP draft head). Integer x in [-4, 4], codes (q - 1.5) and
// power-of-two group scales in [2^-4, 1]: every group sum and its scaled sum is exact in fp32
// (|sum| <= 4 x 1.5 x k < 2^24 at 2^-5 resolution), so the device must equal the double reference
// bit for bit at every split and row shape. Also: quantize_i2_groups picks the nearest level.
void test_dense_i2(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Shape { int n, k; };
  for (const Shape s : {Shape{1280, 2560}, Shape{16, 128}, Shape{2048, 1024}, Shape{124160, 2560}}) {  // + the draft head's per-card slice
    std::mt19937 rng(1700 + s.n + s.k);
    std::vector<float> xf(8 * static_cast<size_t>(s.k));
    std::vector<bf16> x(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) {
      xf[i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
      x[i] = bf16(xf[i]);
    }
    const int G = s.k / 128;
    std::vector<uint8_t> codes(static_cast<size_t>(s.n) * s.k), packed(kernels::dense_i2_bytes(s.n, s.k));
    std::vector<float> scale(static_cast<size_t>(G) * s.n);
    for (auto &c : codes) c = static_cast<uint8_t>(rng() % 4);
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 5));
    kernels::pack_dense_i2(codes.data(), s.n, s.k, packed.data());
    std::vector<double> want(8 * static_cast<size_t>(s.n));
    kernels::dense_ref_i2(xf.data(), codes.data(), scale.data(), 8, s.n, s.k, want.data());
    auto &q = dev.compute();
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dw = dev.alloc<uint8_t>(packed.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(want.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dw, packed.data(), packed.size());
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    kernels::DenseWeights w;
    w.kind = kernels::DenseKind::i2;
    w.w = dw;
    w.scale = ds;
    w.n = s.n;
    w.k = s.k;
    std::vector<float> got(want.size());
    for (int ks : {0, 1, 2, 4, 8, 16}) {
      if (ks > G) continue;
      for (const int m : {1, 2, 3, 4, 8}) {  // 1-3 rows too (the drafter's passes 2-3 run 1 row)
        if (s.n > 4096 && ks != 0) continue;  // the head slice: the auto split only
        q.fill(dy, -12345.f, got.size()).wait();
        kernels::dense_gemm(q, m, dx, w, dy, ks).wait_and_throw();
        q.memcpy(got.data(), dy, got.size() * 4).wait();
        bool exact = true;
        for (size_t i = 0; i < static_cast<size_t>(m) * s.n && exact; ++i) exact = static_cast<double>(got[i]) == want[i];
        char what[112];
        std::snprintf(what, sizeof what, "dense int2 M=%d N=%d K=%d ksplit=%d%s bit-exact", m, s.n, s.k, ks ? ks : kernels::dense_auto_ksplit(s.n, s.k),
                      ks ? "" : " (auto)");
        expect(exact, what);
      }
    }
    dev.release(dx);
    dev.release(dw);
    dev.release(ds);
    dev.release(dy);
  }
  // quantiser: every code is the nearest level of its group's chosen scale
  {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 0.02f);
    const int n = 32, k = 256;
    std::vector<uint16_t> wb(static_cast<size_t>(n) * k);
    for (auto &v : wb) v = kernels::bf16_bits(nd(rng));
    std::vector<uint8_t> codes(wb.size());
    std::vector<float> sc(static_cast<size_t>(k / 128) * n);
    kernels::quantize_i2_groups(wb.data(), n, k, codes.data(), sc.data());
    bool nearest = true;
    double e2 = 0, r2 = 0;
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < k; ++c) {
        const float v = kernels::bf16_value(wb[static_cast<size_t>(r) * k + c]), s0 = sc[static_cast<size_t>(c / 128) * n + r];
        const int q0 = codes[static_cast<size_t>(r) * k + c];
        const double d0 = std::fabs(v - (q0 - 1.5) * s0);
        for (int qq = 0; qq < 4; ++qq) nearest = nearest && d0 <= std::fabs(v - (qq - 1.5) * s0) + 1e-12;
        e2 += d0 * d0;
        r2 += static_cast<double>(v) * v;
      }
    const double rel = std::sqrt(e2 / r2);
    char what[128];
    std::snprintf(what, sizeof what, "quantize_i2_groups: nearest level everywhere, Gaussian relative error %.3f (Lloyd-Max 4-level: 0.34)", rel);
    expect(nearest && rel < 0.37, what);
  }
}

// Dense decode GEMM (int8 W8A16 and bf16): integer activations in [-4, 4] with integer
// weights (int8 codes / small-integer bf16) and power-of-two scales keep every fp32
// intermediate exact (|sum| <= 4 * 127 * 10240 < 2^24), so the device must equal the
// double reference bit for bit at every split; shapes include k % 128 tails (320 = 2 groups
// + 4 calls, 48 = 3 calls only) and single-tile n. Output poisoned before each launch.
void test_dense(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Shape { int n, k; };
  for (const Shape s : {Shape{1280, 2560}, Shape{2560, 6144}, Shape{320, 10240}, Shape{10240, 320}, Shape{16, 48}, Shape{48, 2560}}) {
    std::mt19937 rng(900 + s.n + s.k);
    std::vector<float> xf(8 * static_cast<size_t>(s.k));  // 8 rows: the 8-row shape; M=4 reads the first 4
    std::vector<bf16> x(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) {
      xf[i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
      x[i] = bf16(xf[i]);
    }
    std::vector<int8_t> codes(static_cast<size_t>(s.n) * s.k), packed_i8(codes.size());
    std::vector<uint16_t> wb(codes.size()), packed_bf(codes.size());
    std::vector<float> scale(s.n);
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : wb) v = kernels::bf16_bits(static_cast<float>(static_cast<int>(rng() % 33) - 16));
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 12));
    kernels::pack_dense_i8(codes.data(), s.n, s.k, packed_i8.data());
    kernels::pack_dense_bf16(wb.data(), s.n, s.k, packed_bf.data());
    std::vector<double> want_i8(8 * static_cast<size_t>(s.n)), want_bf(want_i8.size());
    kernels::dense_ref_i8(xf.data(), codes.data(), scale.data(), 8, s.n, s.k, want_i8.data());
    kernels::dense_ref_bf16(xf.data(), wb.data(), 8, s.n, s.k, want_bf.data());
    auto &q = dev.compute();
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dwi = dev.alloc<int8_t>(packed_i8.size(), xenithium::l0::MemKind::Device);
    auto *dwb = dev.alloc<uint16_t>(packed_bf.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(want_i8.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dwi, packed_i8.data(), packed_i8.size());
    q.memcpy(dwb, packed_bf.data(), packed_bf.size() * 2);
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    std::vector<float> got(want_i8.size());
    for (const bool i8 : {true, false}) {
      kernels::DenseWeights w;
      w.kind = i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
      w.w = i8 ? static_cast<const void *>(dwi) : static_cast<const void *>(dwb);
      w.scale = i8 ? ds : nullptr;
      w.n = s.n;
      w.k = s.k;
      const auto &want = i8 ? want_i8 : want_bf;
      for (int ks : {0, 1, 2, 4, 8, 16}) {
        if (ks > std::max(1, s.k / 128)) continue;
        for (const int m : {4, 8}) {
          q.fill(dy, -12345.f, got.size()).wait();
          if (m == 4) kernels::dense_gemm_m4(q, dx, w, dy, ks).wait_and_throw();
          else kernels::dense_gemm(q, 8, dx, w, dy, ks).wait_and_throw();
          q.memcpy(got.data(), dy, got.size() * 4).wait();
          bool exact = true;
          for (size_t i = 0; i < static_cast<size_t>(m) * s.n && exact; ++i) exact = static_cast<double>(got[i]) == want[i];
          char what[112];
          std::snprintf(what, sizeof what, "dense %s M=%d N=%d K=%d ksplit=%d%s bit-exact", i8 ? "int8" : "bf16", m, s.n, s.k,
                        ks ? ks : kernels::dense_auto_ksplit(s.n, s.k), ks ? "" : " (auto)");
          expect(exact, what);
        }
      }
    }
    // On real-valued inputs, the 8-row shape gives each row the bits
    // of the 4-row shape on that row: rows 0-3 and rows 4-7 each as a 4-row launch.
    {
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<bf16> xr(x.size());
      for (auto &v : xr) v = bf16(nd(rng));
      q.memcpy(dx, xr.data(), xr.size() * sizeof(bf16)).wait();
      std::vector<float> a8(got.size()), a4(got.size());
      for (const bool i8 : {true, false}) {
        kernels::DenseWeights w;
        w.kind = i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
        w.w = i8 ? static_cast<const void *>(dwi) : static_cast<const void *>(dwb);
        w.scale = i8 ? ds : nullptr;
        w.n = s.n;
        w.k = s.k;
        bool same = true;
        for (int ks : {0, 1, 4}) {
          if (ks > std::max(1, s.k / 128)) continue;
          kernels::dense_gemm(q, 7, dx, w, dy, ks).wait_and_throw();  // 7 live rows run the 8-row shape
          q.memcpy(a8.data(), dy, a8.size() * 4).wait();
          kernels::dense_gemm(q, 4, dx, w, dy, ks).wait_and_throw();
          q.memcpy(a4.data(), dy, 4ull * s.n * 4).wait();
          kernels::dense_gemm(q, 3, dx + 4ull * s.k, w, dy, ks).wait_and_throw();
          q.memcpy(a4.data() + 4ull * s.n, dy, 4ull * s.n * 4).wait();
          same &= std::memcmp(a8.data(), a4.data(), 7ull * s.n * 4) == 0;  // the live rows (dead rows are not written)
          // 1-3 live rows compute only those rows: == the 4-row launch's rows, bit for bit
          std::vector<float> r4(4ull * s.n), rl(4ull * s.n);
          kernels::dense_gemm(q, 4, dx, w, dy, ks).wait_and_throw();
          q.memcpy(r4.data(), dy, r4.size() * 4).wait();
          for (int live = 1; live <= 3; ++live) {
            kernels::dense_gemm(q, live, dx, w, dy, ks).wait_and_throw();
            q.memcpy(rl.data(), dy, static_cast<size_t>(live) * s.n * 4).wait();
            same &= std::memcmp(rl.data(), r4.data(), static_cast<size_t>(live) * s.n * 4) == 0;
          }
        }
        char what[112];
        std::snprintf(what, sizeof what, "dense %s N=%d K=%d: 8-row shape == two 4-row launches, 1-3 rows == 4, bit for bit", i8 ? "int8" : "bf16", s.n, s.k);
        expect(same, what);
      }
    }
    dev.release(dx);
    dev.release(dwi);
    dev.release(dwb);
    dev.release(ds);
    dev.release(dy);
  }
}

// dense_gemm at 33-64 rows (two launches: rows 0-31, then the rest at its own shape) == rows 0-31 and
// rows 32.. as the caller's own two launches, bit for bit; int8 and bf16 weights, each k split.
void test_dense_wide(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  constexpr int kFirst = kernels::kStepRows32, kRows = kernels::kMaxShape;
  struct Shape { int n, k; };
  for (const Shape s : {Shape{512, 2560}, Shape{2560, 640}, Shape{48, 2560}}) {
    std::mt19937 rng(1900 + s.n + s.k);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> x(static_cast<size_t>(kRows) * s.k);
    for (auto &v : x) v = bf16(nd(rng));
    std::vector<int8_t> codes(static_cast<size_t>(s.n) * s.k), packed_i8(codes.size());
    std::vector<uint16_t> wb(codes.size()), packed_bf(codes.size());
    std::vector<float> scale(s.n);
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : wb) v = kernels::bf16_bits(nd(rng));
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 12));
    kernels::pack_dense_i8(codes.data(), s.n, s.k, packed_i8.data());
    kernels::pack_dense_bf16(wb.data(), s.n, s.k, packed_bf.data());
    auto &q = dev.compute();
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dwi = dev.alloc<int8_t>(packed_i8.size(), xenithium::l0::MemKind::Device);
    auto *dwb = dev.alloc<uint16_t>(packed_bf.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(static_cast<size_t>(kRows) * s.n, xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dwi, packed_i8.data(), packed_i8.size());
    q.memcpy(dwb, packed_bf.data(), packed_bf.size() * 2);
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    for (const bool i8 : {true, false}) {
      kernels::DenseWeights w;
      w.kind = i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
      w.w = i8 ? static_cast<const void *>(dwi) : static_cast<const void *>(dwb);
      w.scale = i8 ? ds : nullptr;
      w.n = s.n;
      w.k = s.k;
      bool same = true;
      for (const int rows : {kRows, 48, kFirst + 1})
        for (const int ks : {0, 1, 4}) {
          if (ks > std::max(1, s.k / 128)) continue;
          const size_t n_all = static_cast<size_t>(rows) * s.n, n_first = static_cast<size_t>(kFirst) * s.n;
          std::vector<float> all(n_all), parts(n_all);
          kernels::dense_gemm(q, rows, dx, w, dy, ks).wait_and_throw();
          q.memcpy(all.data(), dy, n_all * 4).wait();
          q.fill(dy, -12345.f, n_all).wait();
          kernels::dense_gemm(q, kFirst, dx, w, dy, ks).wait_and_throw();
          q.memcpy(parts.data(), dy, n_first * 4).wait();
          kernels::dense_gemm(q, rows - kFirst, dx + static_cast<size_t>(kFirst) * s.k, w, dy, ks).wait_and_throw();
          q.memcpy(parts.data() + n_first, dy, (n_all - n_first) * 4).wait();
          same &= std::memcmp(all.data(), parts.data(), n_all * 4) == 0;
        }
      char what[128];
      std::snprintf(what, sizeof what, "dense %s N=%d K=%d: 64 / 48 / 33 rows == rows 0-31 and 32.. as two launches, bit for bit", i8 ? "int8" : "bf16", s.n, s.k);
      expect(same, what);
    }
    bool refused = false;
    try {
      kernels::DenseWeights w;
      w.kind = kernels::DenseKind::bf16, w.w = dwb, w.n = s.n, w.k = s.k;
      kernels::dense_gemm(q, kRows + 1, dx, w, dy, 0);
    } catch (const std::invalid_argument &) {
      refused = true;
    }
    expect(refused, "dense_gemm refuses 65 rows");
    dev.release(dx);
    dev.release(dwi);
    dev.release(dwb);
    dev.release(ds);
    dev.release(dy);
  }
}

// (the two-stage lm_head): head_select_tiles == the host top-T of the tile maxima (real-valued logits,
// plus a row of equal tiles: ties to the lowest tile; a ragged last super tile); dense_gemm_tiles writes, on the listed tiles
// of each row only, the bits dense_gemm gives there (any row count), and leaves every other element alone.
void test_head_two_stage(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  auto &q = dev.compute();
  // a card's head slice, a draft-vocabulary slice (ragged super tile), small; TP1's whole vocabulary (two parts of 7,760 tiles)
  for (const int n : {124160, 49216, 4096, 248320}) {
    const int NT = n / 16, P = kernels::head_select_parts(n);
    expect(P == (n == 248320 ? 2 : 1), "head_select_parts: two parts for the whole vocabulary on one card, one for a TP2 slice");
    std::mt19937 rng(2800 + n);
    std::normal_distribution<float> nd(0.f, 3.f);
    const int R = 8;
    std::vector<float> lg(static_cast<size_t>(R) * n);
    for (auto &v : lg) v = nd(rng);
    for (int c = 0; c < n; ++c) lg[static_cast<size_t>(5) * n + c] = 1.f;  // row 5: every tile ties
    for (int t = 0; t < 40 && t < NT; ++t) lg[static_cast<size_t>(6) * n + static_cast<size_t>(NT - 1 - t) * 16 + 3] = 50.f + static_cast<float>(t % 3);  // row 6: top in the last tiles
    auto *dl = dev.alloc<float>(lg.size(), xenithium::l0::MemKind::Device);
    void *scr = dev.alloc<uint8_t>(kernels::head_select_scratch_bytes(n), xenithium::l0::MemKind::Device);
    auto *dt = dev.alloc<int32_t>(static_cast<size_t>(R) * kernels::kHeadTilesMax * P, xenithium::l0::MemKind::Device);
    q.memcpy(dl, lg.data(), lg.size() * 4).wait();
    for (const int T : {1, 8, 16, 32}) {
      if (16 * T > NT) continue;
      for (const int rows : {1, 3, 8}) {
        kernels::head_select_tiles(q, dl, n, rows, n, T, scr, dt).wait_and_throw();
        std::vector<int32_t> got(static_cast<size_t>(rows) * P * T), want(got.size());
        q.memcpy(got.data(), dt, got.size() * 4).wait();
        for (int r = 0; r < rows; ++r) kernels::head_select_ref(lg.data() + static_cast<size_t>(r) * n, n, T, want.data() + static_cast<size_t>(r) * P * T);
        char what[112];
        std::snprintf(what, sizeof what, "head_select_tiles n=%d T=%d rows=%d == host top-T of the tile maxima", n, T, rows);
        expect(got == want, what);
      }
    }
    // head_cand_tiles (keys on the bf16-rounded maxima) == head_cand_ref, values == the logits there; row 7
    // holds maxima that differ in fp32 but tie after rounding (17 + tiny steps), and signed zeros, so its order is the
    // rounded one (head_select_tiles' fp32 order would differ there)
    {
      std::vector<float> lc = lg;
      for (int t = 0; t < NT; ++t)
        for (int j = 0; j < 16; ++j)
          lc[static_cast<size_t>(7) * n + static_cast<size_t>(t) * 16 + j] =
              j == 5 ? 17.f + 0.001f * static_cast<float>((t * 7919) % 97) : (t % 3 == 0 ? -0.f : (t % 3 == 1 ? 0.f : -3.f));
      q.memcpy(dl, lc.data(), lc.size() * 4).wait();
      auto *dv = dev.alloc<float>(static_cast<size_t>(R) * kernels::kHeadTilesMax * P * 16, xenithium::l0::MemKind::Device);
      bool rounded_differs = false;
      for (const int T : {1, 8, 20, 32}) {
        if (16 * T > NT) continue;
        for (const int rows : {1, 2, 5, 8}) {
          kernels::head_cand_tiles(q, dl, n, rows, n, T, scr, dt, dv).wait_and_throw();
          const int PT = P * T;
          std::vector<int32_t> got(static_cast<size_t>(rows) * PT), want(got.size()), fp(static_cast<size_t>(PT));
          std::vector<float> vals(static_cast<size_t>(rows) * PT * 16);
          q.memcpy(got.data(), dt, got.size() * 4).wait();
          q.memcpy(vals.data(), dv, vals.size() * 4).wait();
          bool vals_ok = true;
          for (int r = 0; r < rows; ++r) {
            const float *row = lc.data() + static_cast<size_t>(r) * n;
            kernels::head_cand_ref(row, n, T, want.data() + static_cast<size_t>(r) * PT);
            for (int i = 0; i < PT; ++i)
              for (int j = 0; j < 16; ++j)
                vals_ok &= std::memcmp(&vals[(static_cast<size_t>(r) * PT + i) * 16 + j], &row[static_cast<size_t>(got[static_cast<size_t>(r) * PT + i]) * 16 + j], 4) == 0;
            if (r == 7) {
              kernels::head_select_ref(row, n, T, fp.data());
              rounded_differs |= !std::equal(fp.begin(), fp.end(), want.begin() + static_cast<std::ptrdiff_t>(r) * PT);
            }
          }
          char what[128];
          std::snprintf(what, sizeof what, "head_cand_tiles n=%d T=%d rows=%d == head_cand_ref, values == the logits", n, T, rows);
          expect(got == want && vals_ok, what);
        }
      }
      if (16 * 8 <= NT) expect(rounded_differs, "head_cand_tiles: the rounding-tie row orders differently from the fp32 keys (the test sees the rounding)");
      dev.release(dv);
    }
    dev.release(dl);
    dev.release(scr);
    dev.release(dt);
  }
  // dense_gemm_tiles on an lm_head-like shape (ksplit 1) and a small one
  for (const int n : {124160, 8192, 248320}) {  // 248,320: TP1's whole head (two parts of 32 tiles per row)
    const int k = 512;
    if (kernels::dense_auto_ksplit(n, k) != 1) continue;
    std::mt19937 rng(2900 + n);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> x(8ull * k);
    for (auto &v : x) v = bf16(nd(rng));
    std::vector<int8_t> codes(static_cast<size_t>(n) * k), packed(kernels::dense_packed_elems(n, k));
    std::vector<float> scale(static_cast<size_t>(n));
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : scale) v = 0.001f * (1.f + static_cast<float>(rng() % 1000) / 1000.f);
    kernels::pack_dense_i8(codes.data(), n, k, packed.data());
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dw = dev.alloc<int8_t>(packed.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(8ull * n, xenithium::l0::MemKind::Device);
    auto *dt = dev.alloc<int32_t>(8ull * 64, xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * 2);
    q.memcpy(dw, packed.data(), packed.size());
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    kernels::DenseWeights w;
    w.kind = kernels::DenseKind::i8, w.w = dw, w.scale = ds, w.n = n, w.k = k;
    for (const int rows : {1, 2, 4, 5, 8}) {
      std::vector<float> full(static_cast<size_t>(rows) * n), got(full.size());
      kernels::dense_gemm(q, rows, dx, w, dy, 0).wait_and_throw();
      q.memcpy(full.data(), dy, full.size() * 4).wait();
      const int per = 32 * kernels::head_select_parts(n);
      std::vector<int32_t> list(static_cast<size_t>(rows) * per);
      for (auto &t : list) t = static_cast<int32_t>(rng() % static_cast<unsigned>(n / 16));
      q.memcpy(dt, list.data(), list.size() * 4).wait();
      q.fill(dy, -7.f, full.size()).wait();
      kernels::dense_gemm_tiles(q, rows, dx, w, dy, dt, per).wait_and_throw();
      q.memcpy(got.data(), dy, got.size() * 4).wait();
      bool ok = true;
      std::vector<char> in(full.size(), 0);
      for (int r = 0; r < rows; ++r)
        for (int i = 0; i < per; ++i)
          for (int j = 0; j < 16; ++j) in[static_cast<size_t>(r) * n + list[static_cast<size_t>(r) * per + i] * 16 + j] = 1;
      for (size_t i = 0; i < full.size() && ok; ++i) ok = in[i] ? std::memcmp(&got[i], &full[i], 4) == 0 : got[i] == -7.f;
      char what[112];
      std::snprintf(what, sizeof what, "dense_gemm_tiles int8 N=%d K=%d rows=%d: listed tiles == dense_gemm bit for bit, rest untouched", n, k, rows);
      expect(ok, what);
    }
    dev.release(dx);
    dev.release(dw);
    dev.release(ds);
    dev.release(dy);
    dev.release(dt);
  }
}

// Prefill GEMM: the same packed weights and the same integer data as test_dense, so
// every fp32 intermediate is exact and the device must equal the double reference bit for bit.
// Row counts cover one row, ragged row blocks (13, 300) and several work-groups' row ranges
// (1031 on the small shapes); lda > k checks the row stride. Split-K partials summed in part
// order on the host (then scaled) must equal the reference too.
void test_pf_dense(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Shape { int n, k, m; };
  const Shape shapes[] = {{1280, 2560, 300}, {2560, 6144, 13}, {320, 10240, 64}, {10240, 320, 256},
                          {16, 48, 1031},    {48, 2560, 1031}, {2560, 640, 1},   {5120, 320, 300}};
  for (const Shape s : shapes) {
    const int lda = s.k + 8;
    std::mt19937 rng(1700 + s.n + s.k + s.m);
    std::vector<float> xf(static_cast<size_t>(s.m) * s.k);
    std::vector<bf16> x(static_cast<size_t>(s.m) * lda, bf16(0.f));
    for (int r = 0; r < s.m; ++r)
      for (int c = 0; c < s.k; ++c) {
        const float v = static_cast<float>(static_cast<int>(rng() % 9) - 4);
        xf[static_cast<size_t>(r) * s.k + c] = v;
        x[static_cast<size_t>(r) * lda + c] = bf16(v);
      }
    std::vector<int8_t> codes(static_cast<size_t>(s.n) * s.k), packed_i8(codes.size());
    std::vector<uint16_t> wb(codes.size()), packed_bf(codes.size());
    std::vector<float> scale(s.n);
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : wb) v = kernels::bf16_bits(static_cast<float>(static_cast<int>(rng() % 33) - 16));
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 12));
    kernels::pack_dense_i8(codes.data(), s.n, s.k, packed_i8.data());
    kernels::pack_dense_bf16(wb.data(), s.n, s.k, packed_bf.data());
    std::vector<double> want_i8(static_cast<size_t>(s.m) * s.n), want_bf(want_i8.size());
    kernels::dense_ref_i8(xf.data(), codes.data(), scale.data(), s.m, s.n, s.k, want_i8.data());
    kernels::dense_ref_bf16(xf.data(), wb.data(), s.m, s.n, s.k, want_bf.data());
    auto &q = dev.compute();
    const int ks_max = 4;
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dwi = dev.alloc<int8_t>(packed_i8.size(), xenithium::l0::MemKind::Device);
    auto *dwb = dev.alloc<uint16_t>(packed_bf.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(want_i8.size() * ks_max, xenithium::l0::MemKind::Device);
    auto *db = dev.alloc<bf16>(want_i8.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dwi, packed_i8.data(), packed_i8.size());
    q.memcpy(dwb, packed_bf.data(), packed_bf.size() * 2);
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    std::vector<float> got(want_i8.size() * ks_max);
    std::vector<bf16> gotb(want_i8.size());
    for (const bool i8 : {true, false}) {
      kernels::DenseWeights w;
      w.kind = i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
      w.w = i8 ? static_cast<const void *>(dwi) : static_cast<const void *>(dwb);
      w.scale = i8 ? ds : nullptr;
      w.n = s.n;
      w.k = s.k;
      const auto &want = i8 ? want_i8 : want_bf;
      q.fill(dy, -12345.f, want.size()).wait();
      q.fill(db, bf16(-12345.f), want.size()).wait();
      kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy, db, 0}).wait_and_throw();
      q.memcpy(got.data(), dy, want.size() * 4).wait();
      q.memcpy(gotb.data(), db, want.size() * sizeof(bf16)).wait();
      bool exact = true, exact_b = true;
      for (size_t i = 0; i < want.size(); ++i) {
        exact = exact && static_cast<double>(got[i]) == want[i];
        exact_b = exact_b && static_cast<float>(gotb[i]) == static_cast<float>(bf16(static_cast<float>(want[i])));
      }
      char what[128];
      std::snprintf(what, sizeof what, "pf_dense %s M=%d N=%d K=%d fp32 and bf16 out bit-exact", i8 ? "int8" : "bf16", s.m,
                    s.n, s.k);
      expect(exact && exact_b, what);
      const int groups = s.k / 128 + (s.k % 128 ? 1 : 0);
      for (int ks : {2, 3, 4}) {
        if (ks > groups) continue;
        q.fill(dy, -12345.f, want.size() * ks).wait();
        kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, ks).wait_and_throw();
        q.memcpy(got.data(), dy, want.size() * ks * 4).wait();
        bool ok = true;
        for (int r = 0; r < s.m && ok; ++r)
          for (int c = 0; c < s.n && ok; ++c) {
            const size_t at = static_cast<size_t>(r) * s.n + c;
            float sum = 0.f;
            for (int p = 0; p < ks; ++p) sum += got[p * want.size() + at];
            if (i8) sum *= scale[c];
            ok = static_cast<double>(sum) == want[at];
          }
        std::snprintf(what, sizeof what, "pf_dense_splitk %s M=%d N=%d K=%d ks=%d bit-exact", i8 ? "int8" : "bf16", s.m, s.n,
                      s.k, ks);
        expect(ok, what);
      }
      // real-valued data (fp32 rounding matters): every tiles-per-work-group choice gives the
      // same bits, and row 0 alone (m = 1) equals row 0 inside the full call (the many-row path's rows
      // do not depend on the call around them)
      {
        std::vector<bf16> xr(x.size());
        std::uniform_real_distribution<float> uni(-1.f, 1.f);
        for (auto &v : xr) v = bf16(uni(rng));
        q.memcpy(dx, xr.data(), xr.size() * sizeof(bf16)).wait();
        std::vector<float> ref(want.size()), alt(want.size());
        kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, 4).wait_and_throw();
        q.memcpy(ref.data(), dy, ref.size() * 4).wait();
        bool same = true;
        for (const int nt : {2, 1, -1, 0}) {  // -1: the direct kernel (m <= 32 only), 0: auto
          if (nt == -1 && s.m > 32) continue;
          q.fill(dy, -12345.f, want.size()).wait();
          kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, nt).wait_and_throw();
          q.memcpy(alt.data(), dy, alt.size() * 4).wait();
          same = same && std::memcmp(ref.data(), alt.data(), ref.size() * 4) == 0;
        }
        // split-K partials: the direct kernel == the SLM kernel (m <= 32)
        const int groups_k = s.k / 128 + (s.k % 128 ? 1 : 0);
        if (s.m <= 32 && groups_k >= 3) {
          std::vector<float> pa(want.size() * 3), pb(want.size() * 3);
          q.fill(dy, -12345.f, want.size() * 3).wait();
          kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, 3, 4).wait_and_throw();
          q.memcpy(pa.data(), dy, pa.size() * 4).wait();
          q.fill(dy, -12345.f, want.size() * 3).wait();
          kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, 3, -1).wait_and_throw();
          q.memcpy(pb.data(), dy, pb.size() * 4).wait();
          same = same && std::memcmp(pa.data(), pb.data(), pa.size() * 4) == 0;
        }
        q.fill(dy, -12345.f, want.size()).wait();
        kernels::pf_dense(q, 1, w, dx, lda, kernels::PfOut{dy, nullptr, 0}).wait_and_throw();
        q.memcpy(alt.data(), dy, static_cast<size_t>(s.n) * 4).wait();
        const bool row0 = std::memcmp(ref.data(), alt.data(), static_cast<size_t>(s.n) * 4) == 0;
        std::snprintf(what, sizeof what, "pf_dense %s M=%d N=%d K=%d: 4 / 2 / 1 tiles, direct, auto bit-identical, row 0 alone == in the call",
                      i8 ? "int8" : "bf16", s.m, s.n, s.k);
        expect(same && row0, what);
        q.memcpy(dx, x.data(), x.size() * sizeof(bf16)).wait();
      }
    }
    dev.release(dx);
    dev.release(dwi);
    dev.release(dwb);
    dev.release(ds);
    dev.release(dy);
    dev.release(db);
  }
}

// pf_dense_pair == the two pf_dense calls, bit for bit, on real-valued data. The pairs the
// model runs: MoE router (bf16, 33 tiles) + shared gate|up (int8), QSA qkv (int8, direct at m <= 32) +
// idx (int8); m from one row to two row groups (where the pair falls back to two launches). The outputs
// are poisoned first; pairable must say one launch where the model expects it.
void test_pf_dense_pair(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Mat { int n, k; bool i8; };
  struct Case { const char *what; Mat a, b; };
  const Case cases[] = {{"router bf16 + shared gate|up i8", {528, 2560, false}, {1280, 2560, true}},
                        {"qsa qkv i8 + idx i8", {6656, 2560, true}, {640, 2560, true}},
                        {"tail k: i8 + bf16", {256, 2576, true}, {48, 2576, false}}};
  auto &q = dev.compute();
  for (const Case &cs : cases) {
    std::mt19937 rng(4242 + cs.a.n + cs.b.n);
    const Mat *ms[2] = {&cs.a, &cs.b};
    kernels::DenseWeights w[2];
    void *dw[2];
    float *dsc[2];
    for (int i = 0; i < 2; ++i) {
      const Mat &M = *ms[i];
      const size_t ne = static_cast<size_t>(M.n) * M.k;
      std::vector<int8_t> codes(ne), pi8(ne);
      std::vector<uint16_t> wb(ne), pbf(ne);
      std::vector<float> scale(M.n);
      std::uniform_real_distribution<float> uw(-0.05f, 0.05f);
      for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
      for (auto &v : wb) v = kernels::bf16_bits(uw(rng));
      for (auto &v : scale) v = 0.001f * static_cast<float>(1 + rng() % 97);
      dsc[i] = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
      q.memcpy(dsc[i], scale.data(), scale.size() * 4).wait();
      if (M.i8) {
        kernels::pack_dense_i8(codes.data(), M.n, M.k, pi8.data());
        dw[i] = dev.alloc<int8_t>(ne, xenithium::l0::MemKind::Device);
        q.memcpy(dw[i], pi8.data(), ne).wait();
      } else {
        kernels::pack_dense_bf16(wb.data(), M.n, M.k, pbf.data());
        dw[i] = dev.alloc<uint16_t>(ne, xenithium::l0::MemKind::Device);
        q.memcpy(dw[i], pbf.data(), ne * 2).wait();
      }
      w[i].kind = M.i8 ? kernels::DenseKind::i8 : kernels::DenseKind::bf16;
      w[i].w = dw[i];
      w[i].scale = M.i8 ? dsc[i] : nullptr;
      w[i].n = M.n;
      w[i].k = M.k;
    }
    const int mmax = 300, lda = cs.a.k + 8;
    std::vector<bf16> x(static_cast<size_t>(mmax) * lda);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (auto &v : x) v = bf16(uni(rng));
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16)).wait();
    float *ref[2], *got[2];
    bf16 *refb[2], *gotb[2];
    for (int i = 0; i < 2; ++i) {
      const size_t ne = static_cast<size_t>(mmax) * ms[i]->n;
      ref[i] = dev.alloc<float>(ne, xenithium::l0::MemKind::Device);
      got[i] = dev.alloc<float>(ne, xenithium::l0::MemKind::Device);
      refb[i] = dev.alloc<bf16>(ne, xenithium::l0::MemKind::Device);
      gotb[i] = dev.alloc<bf16>(ne, xenithium::l0::MemKind::Device);
    }
    for (const int m : {1, 4, 13, 20, 32, 33, 64, 256, 300}) {
      for (int i = 0; i < 2; ++i) {
        const size_t ne = static_cast<size_t>(m) * ms[i]->n;
        q.fill(got[i], -777.f, ne);
        q.fill(gotb[i], bf16(-777.f), ne);
        kernels::pf_dense(q, m, w[i], dx, lda, kernels::PfOut{ref[i], refb[i], 0});
      }
      kernels::pf_dense_pair(q, m, w[0], dx, lda, kernels::PfOut{got[0], gotb[0], 0}, w[1], dx, lda, kernels::PfOut{got[1], gotb[1], 0}).wait_and_throw();
      bool same = true;
      for (int i = 0; i < 2; ++i) {
        const size_t ne = static_cast<size_t>(m) * ms[i]->n;
        std::vector<float> a(ne), b(ne);
        std::vector<bf16> ab(ne), bb(ne);
        q.memcpy(a.data(), ref[i], ne * 4);
        q.memcpy(b.data(), got[i], ne * 4);
        q.memcpy(ab.data(), refb[i], ne * 2);
        q.memcpy(bb.data(), gotb[i], ne * 2).wait();
        same = same && std::memcmp(a.data(), b.data(), ne * 4) == 0 && std::memcmp(ab.data(), bb.data(), ne * 2) == 0;
      }
      const bool paired = kernels::pf_dense_pairable(m, w[0], w[1]);
      // where the model needs one launch: the short calls (router + shared at m <= 256; qkv + idx at m <= 16)
      const bool want_pair = cs.a.k == 2560 && (cs.a.i8 ? m <= 16 : m <= 256);
      char what[160];
      std::snprintf(what, sizeof what, "pf_dense_pair %s M=%d: == two pf_dense bit for bit (fp32 + bf16 out), %s", cs.what, m,
                    paired ? "one launch" : "two launches");
      expect(same && (cs.a.k != 2560 || paired == want_pair), what);
      // The W4A8 row quantiser as the pair's third job (or its own launch when the GEMMs do
      // not pair): the GEMM outputs unchanged, the codes and scales == a CPU reference bit for bit.
      if (cs.a.k % 128 == 0) {
        const int QC = cs.a.k, QG = QC / 128;
        std::vector<bf16> xq_src(static_cast<size_t>(m) * QC);
        for (size_t i = 0; i < xq_src.size(); ++i) xq_src[i] = x[(i * 7919u) % x.size()];
        for (int i = 0; i < 128 && m > 2; ++i) xq_src[2 * static_cast<size_t>(QC) + i] = bf16(0.f);  // an all-zero group
        auto *dqs = dev.alloc<bf16>(xq_src.size(), xenithium::l0::MemKind::Device);
        auto *dqd = dev.alloc<int8_t>(xq_src.size(), xenithium::l0::MemKind::Device);
        auto *dql = dev.alloc<float>(static_cast<size_t>(m) * QG, xenithium::l0::MemKind::Device);
        q.memcpy(dqs, xq_src.data(), xq_src.size() * sizeof(bf16)).wait();
        bool qok = true;
        for (const bool pr : {true, false}) {
          for (int i = 0; i < 2; ++i) q.fill(got[i], -777.f, static_cast<size_t>(m) * ms[i]->n);
          q.fill(dqd, int8_t(99), xq_src.size());
          q.fill(dql, -777.f, static_cast<size_t>(m) * QG);
          const kernels::PfRowQuant rq{dqs, QC, dqd, dql};
          kernels::pf_dense_pair(q, m, w[0], dx, lda, kernels::PfOut{got[0], nullptr, 0}, w[1], dx, lda, kernels::PfOut{got[1], nullptr, 0}, pr, &rq).wait_and_throw();
          for (int i = 0; i < 2; ++i) {
            const size_t ne = static_cast<size_t>(m) * ms[i]->n;
            std::vector<float> a(ne), b(ne);
            q.memcpy(a.data(), ref[i], ne * 4);
            q.memcpy(b.data(), got[i], ne * 4).wait();
            qok = qok && std::memcmp(a.data(), b.data(), ne * 4) == 0;
          }
          std::vector<int8_t> cq(xq_src.size());
          std::vector<float> cs_(static_cast<size_t>(m) * QG);
          q.memcpy(cq.data(), dqd, cq.size());
          q.memcpy(cs_.data(), dql, cs_.size() * 4).wait();
          for (int r = 0; r < m; ++r)
            for (int g = 0; g < QG; ++g) {
              float amax = 0.f;
              for (int i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(static_cast<float>(xq_src[static_cast<size_t>(r) * QC + g * 128 + i])));
              const float sv = amax > 0.f ? static_cast<float>(static_cast<double>(amax) / 127.0) : 0.f;
              const float inv = amax > 0.f ? static_cast<float>(127.0 / static_cast<double>(amax)) : 0.f;
              qok = qok && cs_[static_cast<size_t>(r) * QG + g] == sv;
              for (int i = 0; i < 128; ++i) {
                const size_t at = static_cast<size_t>(r) * QC + g * 128 + i;
                const float t = std::clamp(std::nearbyint(static_cast<float>(xq_src[at]) * inv), -127.f, 127.f);
                qok = qok && cq[at] == static_cast<int8_t>(t);
              }
            }
        }
        dev.release(dqs);
        dev.release(dqd);
        dev.release(dql);
        std::snprintf(what, sizeof what, "pf_dense_pair %s M=%d + row quantiser: GEMMs unchanged, codes / scales == CPU reference (paired and not)", cs.what, m);
        expect(qok, what);
      }
    }
    for (int i = 0; i < 2; ++i) {
      dev.release(dw[i]);
      dev.release(dsc[i]);
      dev.release(ref[i]);
      dev.release(got[i]);
      dev.release(refb[i]);
      dev.release(gotb[i]);
    }
    dev.release(dx);
  }
}

// W8A8 with one activation scale per row (PfA8:row). Lossless data (one +-127 per row, scales 2^-5, power-of-two
// weight scales): out == float(int sum) x 2^-5 x w_scale bit for bit on every body (4 / 2 / 1 tiles, 2 groups per
// barrier, the direct 16-row blocks), codes and scales exact. Real data (real-valued activations and column scales, so
// the epilogue's two multiplies round differently in another order): every body the same bits (the direct one split over
// k included: M <= 16 on few tiles), row 0 alone == in the call, the per-row quantiser == the CPU rule, the pair == two
// calls; split-K refuses the mode.
void test_pf_dense_a8_row(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Shape { int n, k, m; };
  const Shape shapes[] = {{8192, 2560, 512}, {2560, 3072, 300}, {6144, 2560, 1031}, {1280, 2560, 13}, {2560, 640, 30},
                          {16, 64, 7},       {48, 96, 20},      {10240, 320, 256},  {5120, 320, 300}, {2560, 3072, 1},
                          {2560, 3072, 20},  {2560, 3072, 32},  {336, 10240, 17},  // the split direct body (m <= 32, few tiles)
                          {2560, 1536, 9}};  // four cards' GDN / QSA out projection at <= 16 rows (the split direct body)
  auto &q = dev.compute();
  char what[200];
  for (const Shape s : shapes) {
    const int lda = s.k + 8, lds = kernels::pf_a8_ld(s.m);
    std::mt19937 rng(11900 + s.n + s.k + s.m);
    std::vector<int8_t> aq(static_cast<size_t>(s.m) * s.k);
    std::vector<float> as(lds, std::ldexp(1.f, -5));
    std::vector<bf16> x(static_cast<size_t>(s.m) * lda, bf16(0.f));
    for (int r = 0; r < s.m; ++r) {
      const bool zero = r == 2 && s.m > 2;  // an all-zero row: q = 0, s = 0
      for (int c = 0; c < s.k; ++c) {
        int v = zero ? 0 : static_cast<int>(rng() % 255) - 127;
        if (!zero && c == (r * 37) % s.k) v = r % 2 ? 127 : -127;  // the row's amax, exactly 127 x 2^-5
        aq[static_cast<size_t>(r) * s.k + c] = static_cast<int8_t>(v);
        x[static_cast<size_t>(r) * lda + c] = bf16(std::ldexp(static_cast<float>(v), -5));
      }
      if (zero) as[r] = 0.f;
    }
    std::vector<int8_t> codes(static_cast<size_t>(s.n) * s.k), packed(codes.size());
    std::vector<float> scale(s.n);
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 12));
    kernels::pack_dense_i8(codes.data(), s.n, s.k, packed.data());
    std::vector<float> want(static_cast<size_t>(s.m) * s.n);
    for (int r = 0; r < s.m; ++r)
      for (int c = 0; c < s.n; ++c) {
        int64_t sum = 0;
        for (int i = 0; i < s.k; ++i) sum += static_cast<int64_t>(aq[static_cast<size_t>(r) * s.k + i]) * codes[static_cast<size_t>(c) * s.k + i];
        float v = static_cast<float>(sum) * as[r];
        v *= scale[c];
        want[static_cast<size_t>(r) * s.n + c] = v;
      }
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dw = dev.alloc<int8_t>(packed.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(want.size(), xenithium::l0::MemKind::Device);
    auto *db = dev.alloc<bf16>(want.size(), xenithium::l0::MemKind::Device);
    auto *dq = dev.alloc<int8_t>(aq.size(), xenithium::l0::MemKind::Device);
    auto *dqs = dev.alloc<float>(static_cast<size_t>(lds), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dw, packed.data(), packed.size());
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    kernels::DenseWeights w;
    w.kind = kernels::DenseKind::i8;
    w.w = dw;
    w.scale = ds;
    w.n = s.n;
    w.k = s.k;
    kernels::PfA8 a8{dq, dqs};
    a8.row = true;
    std::vector<float> got(want.size());
    std::vector<bf16> gotb(want.size());
    bool all_exact = true;
    for (const int nt : {0, 4, 2, 1, -1}) {
      if (nt == -1 && s.m > 32) continue;
      q.fill(dy, -12345.f, want.size());
      q.fill(db, bf16(-12345.f), want.size());
      q.fill(dq, int8_t(99), aq.size());
      q.fill(dqs, -777.f, static_cast<size_t>(lds)).wait();
      kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, db, 0}, nt, &a8).wait_and_throw();
      q.memcpy(got.data(), dy, want.size() * 4);
      q.memcpy(gotb.data(), db, want.size() * sizeof(bf16));
      std::vector<int8_t> cq(aq.size());
      std::vector<float> cs(static_cast<size_t>(lds));
      q.memcpy(cq.data(), dq, cq.size());
      q.memcpy(cs.data(), dqs, cs.size() * 4).wait();
      bool exact = std::memcmp(got.data(), want.data(), want.size() * 4) == 0;
      for (size_t i = 0; i < want.size() && exact; ++i) exact = static_cast<float>(gotb[i]) == static_cast<float>(bf16(want[i]));
      bool codes_ok = cq == aq;
      for (int r = 0; r < s.m; ++r) codes_ok = codes_ok && cs[r] == as[r];
      all_exact = all_exact && exact && codes_ok;
    }
    std::snprintf(what, sizeof what, "pf_dense W8A8 per row M=%d N=%d K=%d: auto / 4 / 2 / 1 tiles / direct == reference bit for bit, codes and scales exact", s.m,
                  s.n, s.k);
    expect(all_exact, what);
    {  // real data: every body the same bits; row 0 alone == in the call; the per-row quantiser == the CPU rule
      std::vector<bf16> xr(x.size());
      std::uniform_real_distribution<float> uni(-1.f, 1.f);
      for (auto &v : xr) v = bf16(uni(rng) * (rng() % 97 == 0 ? 9.f : 1.f));
      if (s.m > 2)
        for (int i = 0; i < s.k; ++i) xr[2 * static_cast<size_t>(lda) + i] = bf16(0.f);  // an all-zero row
      q.memcpy(dx, xr.data(), xr.size() * sizeof(bf16)).wait();
      std::vector<float> rscale(s.n);
      for (auto &v : rscale) v = 1e-3f * (1.5f + uni(rng));
      q.memcpy(ds, rscale.data(), rscale.size() * 4).wait();
      std::vector<float> ref(want.size()), alt(want.size());
      kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, 4, &a8).wait_and_throw();
      q.memcpy(ref.data(), dy, ref.size() * 4).wait();
      bool same = true;
      for (const int nt : {2, 1, -1, 0}) {
        if (nt == -1 && s.m > 32) continue;
        q.fill(dy, -12345.f, want.size()).wait();
        kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, nt, &a8).wait_and_throw();
        q.memcpy(alt.data(), dy, alt.size() * 4).wait();
        same = same && std::memcmp(ref.data(), alt.data(), ref.size() * 4) == 0;
      }
      q.fill(dy, -12345.f, want.size()).wait();
      kernels::pf_dense(q, 1, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, &a8).wait_and_throw();
      q.memcpy(alt.data(), dy, static_cast<size_t>(s.n) * 4).wait();
      const bool row0 = std::memcmp(ref.data(), alt.data(), static_cast<size_t>(s.n) * 4) == 0;
      kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, &a8).wait_and_throw();
      std::vector<int8_t> cq(aq.size());
      std::vector<float> cs(static_cast<size_t>(lds));
      q.memcpy(cq.data(), dq, cq.size());
      q.memcpy(cs.data(), dqs, cs.size() * 4).wait();
      bool qok = true;
      for (int r = 0; r < s.m; ++r) {
        float amax = 0.f;
        for (int i = 0; i < s.k; ++i) amax = std::max(amax, std::fabs(static_cast<float>(xr[static_cast<size_t>(r) * lda + i])));
        const float sv = amax > 0.f ? static_cast<float>(static_cast<double>(amax) / 127.0) : 0.f;
        const float inv = amax > 0.f ? static_cast<float>(127.0 / static_cast<double>(amax)) : 0.f;
        qok = qok && cs[r] == sv;
        for (int i = 0; i < s.k; ++i) {
          const float t = std::clamp(std::nearbyint(static_cast<float>(xr[static_cast<size_t>(r) * lda + i]) * inv), -127.f, 127.f);
          qok = qok && cq[static_cast<size_t>(r) * s.k + i] == static_cast<int8_t>(t);
        }
      }
      // the pair falls back to two launches for per-row matrices: == two calls
      auto *dy1 = dev.alloc<float>(want.size(), xenithium::l0::MemKind::Device);
      auto *dq1 = dev.alloc<int8_t>(aq.size(), xenithium::l0::MemKind::Device);
      auto *dqs1 = dev.alloc<float>(static_cast<size_t>(lds), xenithium::l0::MemKind::Device);
      kernels::PfA8 a8b{dq1, dqs1};
      a8b.row = true;
      std::vector<float> g0(want.size()), g1(want.size());
      q.fill(dy1, -12345.f, want.size()).wait();
      kernels::pf_dense_pair(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, w, dx, lda, kernels::PfOut{dy1, nullptr, 0}, true, nullptr, &a8, &a8b)
          .wait_and_throw();
      q.memcpy(g0.data(), dy, g0.size() * 4);
      q.memcpy(g1.data(), dy1, g1.size() * 4).wait();
      const bool pair_ok = std::memcmp(ref.data(), g0.data(), ref.size() * 4) == 0 && std::memcmp(ref.data(), g1.data(), ref.size() * 4) == 0;
      bool split_refused = false;
      try {
        kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, 1, 0, &a8);
      } catch (const std::invalid_argument &) {
        split_refused = true;
      }
      std::snprintf(what, sizeof what,
                    "pf_dense W8A8 per row M=%d N=%d K=%d: bodies bit-identical, row 0 alone == in the call, quantiser == CPU, pair == calls, split-K refused",
                    s.m, s.n, s.k);
      expect(same && row0 && qok && pair_ok && split_refused, what);
      dev.release(dy1);
      dev.release(dq1);
      dev.release(dqs1);
    }
    dev.release(dx);
    dev.release(dw);
    dev.release(ds);
    dev.release(dy);
    dev.release(db);
    dev.release(dq);
    dev.release(dqs);
  }
}

// The W8A8 prefill GEMM. Lossless data (activations n x 2^-5 with n in [-127, 127], one +-127 per
// group, so the quantiser's codes are exact and every scale is 2^-5; int8 codes; power-of-two weight
// scales): the device must equal dense_ref_a8 bit for bit on every body (SLM 4 / 2 / 1 tiles, the direct
// 16-row blocks, split-K parts summed in order), and the scratch must hold the exact codes and scales. Real
// data: every body gives the same bits, row 0 alone == row 0 in the call, the quantiser (with a short tail
// group) == the CPU rule, the pair == two calls, bf16 weights ignore the switch.
void test_pf_dense_a8(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  struct Shape { int n, k, m; };
  const Shape shapes[] = {{336, 10240, 300}, {10240, 320, 256}, {8192, 2560, 13}, {8192, 2560, 30}, {2560, 3072, 30},
                          {1280, 2560, 1031}, {2560, 640, 1},   {5120, 320, 300}, {16, 64, 7},      {48, 96, 20}};
  auto &q = dev.compute();
  for (const Shape s : shapes) {
    const int lda = s.k + 8, gpr = (s.k + 127) / 128, lds = kernels::pf_a8_ld(s.m);
    std::mt19937 rng(2900 + s.n + s.k + s.m);
    std::vector<int8_t> aq(static_cast<size_t>(s.m) * s.k);
    std::vector<float> as(kernels::pf_a8_scales(s.m, s.k), std::ldexp(1.f, -5));  // group-major [gpr][lds]
    std::vector<bf16> x(static_cast<size_t>(s.m) * lda, bf16(0.f));
    for (int r = 0; r < s.m; ++r)
      for (int g = 0; g < gpr; ++g) {
        const bool zero = r == 2 && g == 0 && s.m > 2;  // one all-zero group: q = 0, s = 0
        for (int c = g * 128; c < std::min(s.k, (g + 1) * 128); ++c) {
          int v = zero ? 0 : static_cast<int>(rng() % 255) - 127;
          if (!zero && c == g * 128) v = (r + g) % 2 ? 127 : -127;  // the group's amax, exactly 127 x 2^-5
          aq[static_cast<size_t>(r) * s.k + c] = static_cast<int8_t>(v);
          x[static_cast<size_t>(r) * lda + c] = bf16(std::ldexp(static_cast<float>(v), -5));
        }
        if (zero) as[static_cast<size_t>(g) * lds + r] = 0.f;
      }
    std::vector<int8_t> codes(static_cast<size_t>(s.n) * s.k), packed(codes.size());
    std::vector<float> scale(s.n);
    for (auto &c : codes) c = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    for (auto &v : scale) v = std::ldexp(1.f, -static_cast<int>(rng() % 12));
    kernels::pack_dense_i8(codes.data(), s.n, s.k, packed.data());
    std::vector<double> want(static_cast<size_t>(s.m) * s.n);
    kernels::dense_ref_a8(aq.data(), as.data(), lds, codes.data(), scale.data(), s.m, s.n, s.k, want.data());
    const int ks_max = 16;
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dw = dev.alloc<int8_t>(packed.size(), xenithium::l0::MemKind::Device);
    auto *ds = dev.alloc<float>(scale.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(want.size() * ks_max, xenithium::l0::MemKind::Device);
    auto *db = dev.alloc<bf16>(want.size(), xenithium::l0::MemKind::Device);
    auto *dq = dev.alloc<int8_t>(aq.size(), xenithium::l0::MemKind::Device);
    auto *dqs = dev.alloc<float>(as.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dw, packed.data(), packed.size());
    q.memcpy(ds, scale.data(), scale.size() * 4).wait();
    kernels::DenseWeights w;
    w.kind = kernels::DenseKind::i8;
    w.w = dw;
    w.scale = ds;
    w.n = s.n;
    w.k = s.k;
    const kernels::PfA8 a8{dq, dqs};
    auto scales_eq = [&](const std::vector<float> &cs) {  // the pad columns (rows >= m) are never written
      for (int g = 0; g < gpr; ++g)
        for (int r = 0; r < s.m; ++r)
          if (cs[static_cast<size_t>(g) * lds + r] != as[static_cast<size_t>(g) * lds + r]) return false;
      return true;
    };
    std::vector<float> got(want.size() * ks_max);
    std::vector<bf16> gotb(want.size());
    char what[160];
    {
      q.fill(dy, -12345.f, want.size());
      q.fill(db, bf16(-12345.f), want.size());
      q.fill(dq, int8_t(99), aq.size());
      q.fill(dqs, -777.f, as.size()).wait();
      kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy, db, 0}, &a8).wait_and_throw();
      q.memcpy(got.data(), dy, want.size() * 4);
      q.memcpy(gotb.data(), db, want.size() * sizeof(bf16));
      std::vector<int8_t> cq(aq.size());
      std::vector<float> cs(as.size());
      q.memcpy(cq.data(), dq, cq.size());
      q.memcpy(cs.data(), dqs, cs.size() * 4).wait();
      bool exact = true, exact_b = true;
      for (size_t i = 0; i < want.size(); ++i) {
        exact = exact && static_cast<double>(got[i]) == want[i];
        exact_b = exact_b && static_cast<float>(gotb[i]) == static_cast<float>(bf16(static_cast<float>(want[i])));
      }
      const bool codes_ok = cq == aq && scales_eq(cs);
      std::snprintf(what, sizeof what, "pf_dense W8A8 M=%d N=%d K=%d: fp32 and bf16 out == reference bit for bit, codes and scales exact", s.m, s.n, s.k);
      expect(exact && exact_b && codes_ok, what);
    }
    for (int ks : {2, 3, 4, 16}) {
      if (ks > gpr) continue;
      q.fill(dy, -12345.f, want.size() * ks).wait();
      kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, ks, 0, &a8).wait_and_throw();
      q.memcpy(got.data(), dy, want.size() * ks * 4).wait();
      bool ok = true;
      for (int r = 0; r < s.m && ok; ++r)
        for (int c = 0; c < s.n && ok; ++c) {
          const size_t at = static_cast<size_t>(r) * s.n + c;
          float sum = 0.f;
          for (int p = 0; p < ks; ++p) sum += got[p * want.size() + at];
          ok = static_cast<double>(sum * scale[c]) == want[at];
        }
      std::snprintf(what, sizeof what, "pf_dense_splitk W8A8 M=%d N=%d K=%d ks=%d bit-exact", s.m, s.n, s.k, ks);
      expect(ok, what);
    }
    {
      // real data: every body the same bits; row 0 alone == in the call; the quantiser == the CPU rule
      std::vector<bf16> xr(x.size());
      std::uniform_real_distribution<float> uni(-1.f, 1.f);
      for (auto &v : xr) v = bf16(uni(rng));
      for (int i = 0; i < std::min(s.k, 128) && s.m > 2; ++i) xr[2 * static_cast<size_t>(lda) + i] = bf16(0.f);  // an all-zero group
      q.memcpy(dx, xr.data(), xr.size() * sizeof(bf16)).wait();
      std::vector<float> ref(want.size()), alt(want.size());
      kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, 4, &a8).wait_and_throw();
      q.memcpy(ref.data(), dy, ref.size() * 4).wait();
      bool same = true;
      for (const int nt : {2, 1, -1, 0}) {
        if (nt == -1 && s.m > 32) continue;
        q.fill(dy, -12345.f, want.size()).wait();
        kernels::pf_dense_tiles(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, nt, &a8).wait_and_throw();
        q.memcpy(alt.data(), dy, alt.size() * 4).wait();
        same = same && std::memcmp(ref.data(), alt.data(), ref.size() * 4) == 0;
      }
      if (s.m <= 32 && gpr >= 3) {  // split-K partials: the direct body == the SLM body
        std::vector<float> pa(want.size() * 3), pb(want.size() * 3);
        q.fill(dy, -12345.f, want.size() * 3).wait();
        kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, 3, 4, &a8).wait_and_throw();
        q.memcpy(pa.data(), dy, pa.size() * 4).wait();
        q.fill(dy, -12345.f, want.size() * 3).wait();
        kernels::pf_dense_splitk(q, s.m, w, dx, lda, dy, 0, 3, -1, &a8).wait_and_throw();
        q.memcpy(pb.data(), dy, pb.size() * 4).wait();
        same = same && std::memcmp(pa.data(), pb.data(), pa.size() * 4) == 0;
      }
      q.fill(dy, -12345.f, want.size()).wait();
      kernels::pf_dense(q, 1, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, &a8).wait_and_throw();
      q.memcpy(alt.data(), dy, static_cast<size_t>(s.n) * 4).wait();
      const bool row0 = std::memcmp(ref.data(), alt.data(), static_cast<size_t>(s.n) * 4) == 0;
      // the quantiser's codes and scales vs the CPU rule (IEEE divisions), the short last group included
      q.fill(dq, int8_t(99), aq.size());
      q.fill(dqs, -777.f, as.size()).wait();
      kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, &a8).wait_and_throw();
      std::vector<int8_t> cq(aq.size());
      std::vector<float> cs(as.size());
      q.memcpy(cq.data(), dq, cq.size());
      q.memcpy(cs.data(), dqs, cs.size() * 4).wait();
      bool qok = true;
      for (int r = 0; r < s.m; ++r)
        for (int g = 0; g < gpr; ++g) {
          const int k1 = std::min(s.k, (g + 1) * 128);
          float amax = 0.f;
          for (int i = g * 128; i < k1; ++i) amax = std::max(amax, std::fabs(static_cast<float>(xr[static_cast<size_t>(r) * lda + i])));
          const float sv = amax > 0.f ? static_cast<float>(static_cast<double>(amax) / 127.0) : 0.f;
          const float inv = amax > 0.f ? static_cast<float>(127.0 / static_cast<double>(amax)) : 0.f;
          qok = qok && cs[static_cast<size_t>(g) * lds + r] == sv;
          for (int i = g * 128; i < k1; ++i) {
            const float t = std::clamp(std::nearbyint(static_cast<float>(xr[static_cast<size_t>(r) * lda + i]) * inv), -127.f, 127.f);
            qok = qok && cq[static_cast<size_t>(r) * s.k + i] == static_cast<int8_t>(t);
          }
        }
      std::snprintf(what, sizeof what, "pf_dense W8A8 M=%d N=%d K=%d: 4 / 2 / 1 tiles, direct, auto bit-identical; row 0 alone == in the call; quantiser == CPU",
                    s.m, s.n, s.k);
      expect(same && row0 && qok, what);
      // the pair with W8A8: == the two calls bit for bit, paired where pf_dense_pairable says so
      if (s.m <= 1031) {
        std::vector<float> r0(want.size()), r1(want.size()), g0(want.size()), g1(want.size());
        auto *dy1 = dev.alloc<float>(want.size(), xenithium::l0::MemKind::Device);
        auto *dq1 = dev.alloc<int8_t>(aq.size(), xenithium::l0::MemKind::Device);
        auto *dqs1 = dev.alloc<float>(as.size(), xenithium::l0::MemKind::Device);
        const kernels::PfA8 a8b{dq1, dqs1};
        kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, &a8);
        kernels::pf_dense(q, s.m, w, dx, lda, kernels::PfOut{dy1, nullptr, 0}, &a8b).wait_and_throw();
        q.memcpy(r0.data(), dy, r0.size() * 4);
        q.memcpy(r1.data(), dy1, r1.size() * 4).wait();
        q.fill(dy, -12345.f, want.size());
        q.fill(dy1, -12345.f, want.size()).wait();
        kernels::pf_dense_pair(q, s.m, w, dx, lda, kernels::PfOut{dy, nullptr, 0}, w, dx, lda, kernels::PfOut{dy1, nullptr, 0}, true, nullptr, &a8, &a8b)
            .wait_and_throw();
        q.memcpy(g0.data(), dy, g0.size() * 4);
        q.memcpy(g1.data(), dy1, g1.size() * 4).wait();
        const bool pair_ok = std::memcmp(r0.data(), g0.data(), r0.size() * 4) == 0 && std::memcmp(r1.data(), g1.data(), r1.size() * 4) == 0;
        const bool paired = kernels::pf_dense_pairable(s.m, w, w, true, true);
        std::snprintf(what, sizeof what, "pf_dense_pair W8A8 M=%d N=%d K=%d: == two calls bit for bit (%s)", s.m, s.n, s.k, paired ? "paired" : "two launches");
        expect(pair_ok, what);
        dev.release(dy1);
        dev.release(dq1);
        dev.release(dqs1);
      }
    }
    dev.release(dx);
    dev.release(dw);
    dev.release(ds);
    dev.release(dy);
    dev.release(db);
    dev.release(dq);
    dev.release(dqs);
  }
  // bf16 weights ignore the switch: pf_dense with a8 == without, bit for bit
  {
    const int n = 528, k = 2560, m = 40, lda = k;
    std::mt19937 rng(77);
    std::vector<uint16_t> wb(static_cast<size_t>(n) * k), packed(wb.size());
    for (auto &v : wb) v = kernels::bf16_bits(static_cast<float>(static_cast<int>(rng() % 33) - 16) * 0.25f);
    kernels::pack_dense_bf16(wb.data(), n, k, packed.data());
    std::vector<bf16> x(static_cast<size_t>(m) * lda);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (auto &v : x) v = bf16(uni(rng));
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dw = dev.alloc<uint16_t>(packed.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<float>(static_cast<size_t>(m) * n * 2, xenithium::l0::MemKind::Device);
    auto *dq = dev.alloc<int8_t>(static_cast<size_t>(m) * k, xenithium::l0::MemKind::Device);
    auto *dqs = dev.alloc<float>(static_cast<size_t>(m) * (k / 128), xenithium::l0::MemKind::Device);
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16));
    q.memcpy(dw, packed.data(), packed.size() * 2).wait();
    kernels::DenseWeights w;
    w.kind = kernels::DenseKind::bf16;
    w.w = dw;
    w.n = n;
    w.k = k;
    const kernels::PfA8 a8{dq, dqs};
    kernels::pf_dense(q, m, w, dx, lda, kernels::PfOut{dy, nullptr, 0});
    kernels::pf_dense(q, m, w, dx, lda, kernels::PfOut{dy + static_cast<size_t>(m) * n, nullptr, 0}, &a8).wait_and_throw();
    std::vector<float> a(static_cast<size_t>(m) * n), b(a.size());
    q.memcpy(a.data(), dy, a.size() * 4);
    q.memcpy(b.data(), dy + a.size(), b.size() * 4).wait();
    expect(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "pf_dense bf16 weights: the W8A8 switch changes nothing");
    dev.release(dx);
    dev.release(dw);
    dev.release(dy);
    dev.release(dq);
    dev.release(dqs);
  }
}

// w4a16_gemm_pf (many rows). Exact inputs (integer activations, power-of-two scales) -> bit-exact
// vs w4a16_ref at m = 1.. 300, every shape the same bits, rows past m untouched, a row's bits independent of m and of
// its place (a 36-row call at row offset 100 == rows 100..135 of the 300-row call), a strided A / C (lda > k, ldc > n).
void test_w4_pf(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  for (const auto nk : {std::pair<int, int>{48, 256}, std::pair<int, int>{640, 1280}}) {
    const int n = nk.first, k = nk.second, cap = 300, lda = k + 64, ldc = n + 16;
    std::mt19937 rng(1100 + n);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    std::vector<float> scf(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {
      const float m = std::ldexp(1.f, -static_cast<int>(rng() % 4));
      sc[i] = sycl::half(rng() % 2 ? -m : m);  // negative scales as AutoRound's
      scf[i] = static_cast<float>(sc[i]);
    }
    std::vector<float> af(static_cast<size_t>(cap) * k);
    std::vector<bf16> a(static_cast<size_t>(cap) * lda, bf16(0.f));
    for (int r = 0; r < cap; ++r)
      for (int i = 0; i < k; ++i) {
        af[static_cast<size_t>(r) * k + i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
        a[static_cast<size_t>(r) * lda + i] = bf16(af[static_cast<size_t>(r) * k + i]);
      }
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    std::vector<double> want(static_cast<size_t>(cap) * n);
    kernels::w4a16_ref(af.data(), qw.data(), scf.data(), cap, n, k, want.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(static_cast<size_t>(cap + 8) * ldc, MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    std::vector<float> full;
    for (const bool dq : {false, true})  // the dequantised mode (power-of-two scales: bf16(w x s) exact)
    for (int m : {1, 7, 36, 129, 300})
      for (int shape : {0, 1, 2, 3, 4, 7, 11, 12, 13, 14, 15}) {  // 7, 11 (R x CG work-groups); 12-15 (dq)
        if (m != 300 && shape > 0) continue;
        if (dq != (shape >= 12) && shape > 0) continue;
        kernels::w4a16_pf_set_dequant(dq);
        q.fill(wc, -12345.f, static_cast<size_t>(cap + 8) * ldc).wait();
        kernels::w4a16_gemm_pf(q, m, wa, lda, ww, ws, wc, ldc, n, k, shape).wait_and_throw();
        kernels::w4a16_pf_set_dequant(false);
        std::vector<float> got(static_cast<size_t>(cap + 8) * ldc);
        q.memcpy(got.data(), wc, got.size() * 4).wait();
        bool exact = true, dead = true;
        for (int r = 0; r < cap + 8; ++r)
          for (int i = 0; i < ldc; ++i) {
            const float g = got[static_cast<size_t>(r) * ldc + i];
            if (r < m && i < n) exact &= static_cast<double>(g) == want[static_cast<size_t>(r) * n + i];
            else dead &= g == -12345.f;
          }
        if (m == 300 && shape == 0 && !dq) full = got;
        char what[160];
        std::snprintf(what, sizeof what, "w4a16_gemm_pf%s m=%d N=%d K=%d shape %d bit-exact vs w4a16_ref, padding untouched", dq ? " dequantised" : "", m, n, k, shape);
        expect(exact && dead, what);
      }
    {  // a 36-row call on rows 100..135 (A and C at a row offset) == those rows of the 300-row call
      q.fill(wc, -12345.f, static_cast<size_t>(cap + 8) * ldc).wait();
      kernels::w4a16_gemm_pf(q, 36, wa + static_cast<size_t>(100) * lda, lda, ww, ws, wc + static_cast<size_t>(100) * ldc, ldc, n, k).wait_and_throw();
      std::vector<float> got(static_cast<size_t>(cap + 8) * ldc);
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      bool same = true;
      for (int r = 100; r < 136; ++r)
        for (int i = 0; i < n; ++i) same &= got[static_cast<size_t>(r) * ldc + i] == full[static_cast<size_t>(r) * ldc + i];
      expect(same, "w4a16_gemm_pf: a row's bits do not depend on m or its place in the call");
    }
    {  // calls of <= w4a16_pf_set_short_rows rows on the decode GEMM (contiguous A / C)
      std::vector<bf16> ac(static_cast<size_t>(8) * k);
      for (int r = 0; r < 8; ++r)
        for (int i = 0; i < k; ++i) ac[static_cast<size_t>(r) * k + i] = a[static_cast<size_t>(r) * lda + i];
      auto *ca = dev.alloc<bf16>(ac.size(), MK::Device);
      auto *cc = dev.alloc<float>(static_cast<size_t>(8) * n, MK::Device);
      auto *cd = dev.alloc<float>(static_cast<size_t>(8) * n, MK::Device);
      q.memcpy(ca, ac.data(), ac.size() * sizeof(bf16)).wait();
      kernels::w4a16_pf_set_short_rows(8);
      expect(kernels::w4a16_pf_short_rows() == 8, "w4a16_pf_short_rows: the decode GEMMs registered (libxenithium_kernels)");
      for (int m : {1, 5, 8}) {
        kernels::w4a16_gemm_pf(q, m, ca, k, ww, ws, cc, n, n, k).wait_and_throw();
        kernels::w4a16_gemm_rows(q, m, ca, ww, ws, cd, n, k).wait_and_throw();
        std::vector<float> g1(static_cast<size_t>(m) * n), g2(g1.size());
        q.memcpy(g1.data(), cc, g1.size() * 4);
        q.memcpy(g2.data(), cd, g2.size() * 4).wait();
        bool same = g1 == g2, exact = true;
        for (int r = 0; r < m; ++r)
          for (int i = 0; i < n; ++i) exact &= static_cast<double>(g1[static_cast<size_t>(r) * n + i]) == want[static_cast<size_t>(r) * n + i];
        char what[160];
        std::snprintf(what, sizeof what, "w4a16_gemm_pf short m=%d N=%d K=%d == w4a16_gemm_rows bit for bit, exact vs w4a16_ref", m, n, k);
        expect(same && exact, what);
      }
      kernels::w4a16_pf_set_short_rows(0);
      dev.release(ca);
      dev.release(cc);
      dev.release(cd);
    }
    dev.release(wa);
    dev.release(ww);
    dev.release(ws);
    dev.release(wc);
  }
  {  // full-entropy activations: within a bound of the double reference, run-to-run identical, all shapes identical
    const int n = 512, k = 2048, m = 200;
    std::mt19937 rng(1199);
    std::normal_distribution<float> nd(0.f, 1.f);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    std::vector<float> scf(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {
      sc[i] = sycl::half(nd(rng) * 0.01f);
      scf[i] = static_cast<float>(sc[i]);
    }
    std::vector<bf16> a(static_cast<size_t>(m) * k);
    std::vector<float> af(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = bf16(nd(rng));
      af[i] = static_cast<float>(a[i]);
    }
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    std::vector<double> want(static_cast<size_t>(m) * n);
    kernels::w4a16_ref(af.data(), qw.data(), scf.data(), m, n, k, want.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(want.size(), MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    std::vector<float> first;
    bool same = true;
    double worst = 0, rms = 0;
    for (double v : want) rms += v * v;
    rms = std::sqrt(rms / want.size());
    for (int run = 0; run < 6; ++run) {
      kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k, run == 5 ? 7 : run % 5).wait_and_throw();
      std::vector<float> got(want.size());
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      if (run == 0) {
        first = got;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::isfinite(got[i]) ? std::fabs(got[i] - want[i]) / rms : HUGE_VAL);
      } else {
        same &= got == first;
      }
    }
    char what[160];
    std::snprintf(what, sizeof what, "w4a16_gemm_pf full entropy m=%d: worst |err| / rms %.2e <= 1e-5, every shape and run bit-identical", m, worst);
    expect(worst <= 1e-5 && same, what);
    {  // dequantised: vs a double reference on bf16(float(q - 8) x scale) weights; every dq shape and run, and a
       // 37-row call at row 100 == those rows of the whole call
      std::vector<double> wdq(want.size(), 0.0);
      for (int c = 0; c < n; ++c)
        for (int i = 0; i < k; ++i) {
          const uint32_t word = qw[static_cast<size_t>(i / 8) * n + c];
          const int code = static_cast<int>((word >> (4 * (i % 8))) & 15) - 8;
          const float w = static_cast<float>(bf16(static_cast<float>(code) * scf[static_cast<size_t>(i / 128) * n + c]));
          for (int r = 0; r < m; ++r) wdq[static_cast<size_t>(r) * n + c] += static_cast<double>(af[static_cast<size_t>(r) * k + i]) * w;
        }
      kernels::w4a16_pf_set_dequant(true);
      std::vector<float> dfirst;
      bool dsame = true;
      double dworst = 0, dexact = 0;
      for (const int shape : {-1, 12, 13, 14, 15}) {
        kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k, shape).wait_and_throw();
        std::vector<float> got(want.size());
        q.memcpy(got.data(), wc, got.size() * 4).wait();
        if (dfirst.empty()) {
          dfirst = got;
          for (size_t i = 0; i < got.size(); ++i) {
            dworst = std::max(dworst, std::fabs(got[i] - wdq[i]) / rms);
            dexact = std::max(dexact, std::fabs(got[i] - want[i]) / rms);
          }
        } else {
          dsame &= got == dfirst;
        }
      }
      kernels::w4a16_gemm_pf(q, 37, wa + static_cast<size_t>(100) * k, k, ww, ws, wc, n, n, k).wait_and_throw();
      std::vector<float> part(static_cast<size_t>(37) * n);
      q.memcpy(part.data(), wc, part.size() * 4).wait();
      kernels::w4a16_pf_set_dequant(false);
      for (int r = 0; r < 37; ++r)
        for (int c = 0; c < n; ++c) dsame &= part[static_cast<size_t>(r) * n + c] == dfirst[static_cast<size_t>(100 + r) * n + c];
      std::snprintf(what, sizeof what, "w4a16_gemm_pf dequantised m=%d: worst |err| / rms %.2e vs bf16 w x s <= 1e-5 (%.2e vs exact), shapes / rows identical", m,
                    dworst, dexact);
      expect(dworst <= 1e-5 && dexact < 2e-2 && dsame, what);
    }
    dev.release(wa);
    dev.release(ww);
    dev.release(ws);
    dev.release(wc);
  }
  // w4a16_gemm_pf_swiglu == w4a16_gemm_pf + swiglu bit for bit (full entropy; n_half 624 leaves a
  // half-filled last column block), every shape, rows past m untouched
  for (const int nh : {640, 624, 656}) {
    const int k = 1280, n = 2 * nh, mmax = 300;
    std::mt19937 rng(1313 + nh);
    std::normal_distribution<float> nd(0.f, 1.f);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    for (auto &v : sc) v = sycl::half(nd(rng) * (nh == 656 ? 0.5f : 0.02f));  // 656: |g| up to ~100 (the exp's whole range)
    std::vector<bf16> a(static_cast<size_t>(mmax) * k);
    for (auto &v : a) v = bf16(nd(rng));
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(static_cast<size_t>(mmax) * n, MK::Device);
    auto *wh = dev.alloc<bf16>(static_cast<size_t>(mmax + 1) * nh, MK::Device);
    auto *wf = dev.alloc<bf16>(static_cast<size_t>(mmax + 1) * nh, MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    for (const bool dq : {false, true})  // the dequantised mode too
    for (const int m : {37, 300})
      for (const int shape : {-1, 0, 2, 4, 7, 12, 13}) {
        if (shape >= 12 && !dq) continue;
        if (dq && shape >= 0 && shape < 12) continue;
        kernels::w4a16_pf_set_dequant(dq);
        q.fill(wh, bf16(-3.f), static_cast<size_t>(mmax + 1) * nh);
        q.fill(wf, bf16(-3.f), static_cast<size_t>(mmax + 1) * nh).wait();
        kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k, shape);
        kernels::swiglu(q, m, nh, wc, wh);
        kernels::w4a16_gemm_pf_swiglu(q, m, wa, k, ww, ws, wf, nh, nh, k, shape).wait_and_throw();
        std::vector<uint16_t> h0(static_cast<size_t>(mmax + 1) * nh), h1(h0.size());
        q.memcpy(h0.data(), wh, h0.size() * 2);
        q.memcpy(h1.data(), wf, h1.size() * 2).wait();
        bool dead = true;
        for (size_t i = static_cast<size_t>(m) * nh; i < h1.size(); ++i) dead &= h1[i] == h0[i];
        std::vector<float> gu(static_cast<size_t>(m) * n);
        q.memcpy(gu.data(), wc, gu.size() * 4).wait();
        // off_in: with |g| < 60. Past it |silu(g)| < 1e-24 and the fused and the separate kernels' exp paths part.
        size_t off = 0, off_in = 0;
        float g0 = 0;
        for (size_t i = 0; i < static_cast<size_t>(m) * nh; ++i)
          if (h1[i] != h0[i]) {
            const float g = gu[(i / nh) * n + i % nh];
            if (off++ == 0) g0 = g;
            off_in += std::fabs(g) < 60.f;
          }
        char what[256];
        std::snprintf(what, sizeof what,
                      "w4a16_gemm_pf_swiglu%s m=%d n_half=%d K=%d shape %d == w4a16_gemm_pf + swiglu bit for bit where |g| < 60 (%zu off there; %zu in all, first g %.6g), rows past m untouched",
                      dq ? " dequantised" : "", m, nh, k, shape, off_in, off, g0);
        expect(off_in == 0 && dead, what);
        kernels::w4a16_pf_set_dequant(false);
      }
    for (void *pp : {(void *)wa, (void *)ww, (void *)ws, (void *)wc, (void *)wh, (void *)wf}) dev.release(pp);
  }
}

// The norm writes the W4A8 codes. rms_norm_q8 == rms_norm on xn bit for bit, in one call
// or in two row parts; a W4A8 GEMM handed the codes (w4a16_pf_a8_input) == the same GEMM quantising xn itself, bit for bit,
// plain and GLU, at row offsets (multiples of 4 read the codes, others fall back), after a clear (its own quantiser again).
void test_w4a8_norm_codes(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  const int m = 300, H = 1280, nh = 640, n = 2 * nh, ld8 = (m + 4 + 31) / 32 * 32, ldc = n;
  kernels::w4a16_pf_a8_reserve(q, 512, 2048);
  std::mt19937 rng(3101);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> res(static_cast<size_t>(m) * H), add(res.size()), w(H);
  for (auto &v : res) v = bf16(nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f));
  for (auto &v : add) v = bf16(nd(rng) * 0.5f);
  for (auto &v : w) v = bf16(nd(rng) * 0.1f);
  for (int i = 0; i < H; ++i) res[static_cast<size_t>(7) * H + i] = bf16(0.f), add[static_cast<size_t>(7) * H + i] = bf16(0.f);  // a zero row
  const kernels::W4Problem p{n, H};
  std::vector<uint32_t> qw(static_cast<size_t>(H / 8) * n);
  for (auto &v : qw) v = static_cast<uint32_t>(rng());
  std::vector<sycl::half> sc(p.scale_count());
  for (auto &v : sc) v = sycl::half(0.01f * (1 + rng() % 50) * (rng() % 2 ? -1.f : 1.f));
  std::vector<uint8_t> packed(p.packed_bytes());
  kernels::pack_w4_gptq(qw.data(), n, H, packed.data());
  auto *dres = dev.alloc<bf16>(res.size(), MK::Device);
  auto *dadd = dev.alloc<bf16>(add.size(), MK::Device);
  auto *dw = dev.alloc<bf16>(w.size(), MK::Device);
  auto *xn = dev.alloc<bf16>(res.size(), MK::Device);
  auto *xn2 = dev.alloc<bf16>(res.size(), MK::Device);
  auto *xq = dev.alloc<int8_t>(static_cast<size_t>(m + 4) * H, MK::Device);
  auto *xs = dev.alloc<float>(static_cast<size_t>(H / 128) * ld8, MK::Device);
  auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
  auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
  auto *c1 = dev.alloc<float>(static_cast<size_t>(m) * ldc, MK::Device);
  auto *c2 = dev.alloc<float>(static_cast<size_t>(m) * ldc, MK::Device);
  auto *h1 = dev.alloc<bf16>(static_cast<size_t>(m) * nh, MK::Device);
  auto *h2 = dev.alloc<bf16>(static_cast<size_t>(m) * nh, MK::Device);
  q.memset(xs, 0, static_cast<size_t>(H / 128) * ld8 * 4);
  q.memcpy(dw, w.data(), w.size() * sizeof(bf16));
  q.memcpy(ww, packed.data(), packed.size());
  q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
  auto get = [&](auto *d, size_t cnt) {
    std::vector<std::remove_pointer_t<decltype(d)>> v(cnt);
    q.memcpy(v.data(), d, cnt * sizeof(v[0])).wait();
    return v;
  };
  auto same_bytes = [](const auto &a, const auto &b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0; };
  // the norms: plain vs q8 (one call), then q8 in two parts (rows 0..148, 148..300) == one call's codes
  q.memcpy(dres, res.data(), res.size() * sizeof(bf16));
  q.memcpy(dadd, add.data(), add.size() * sizeof(bf16)).wait();
  kernels::rms_norm(q, m, H, dres, dadd, nullptr, dw, 1e-6f, xn).wait_and_throw();
  const auto res1 = get(dres, res.size());
  q.memcpy(dres, res.data(), res.size() * sizeof(bf16));
  kernels::rms_norm_q8(q, m, H, dres, dadd, nullptr, dw, 1e-6f, xn2, kernels::RmsNormQ8{xq, xs, ld8}).wait_and_throw();
  const auto res2 = get(dres, res.size());
  const auto x1 = get(xn, res.size()), x2 = get(xn2, res.size());
  const auto q1 = get(xq, static_cast<size_t>(m) * H);
  const auto s1 = get(xs, static_cast<size_t>(H / 128) * ld8);
  q.memcpy(dres, res.data(), res.size() * sizeof(bf16));
  q.memset(xq, 0x55, static_cast<size_t>(m) * H).wait();
  for (const auto rr : {std::pair<int, int>{0, 148}, std::pair<int, int>{148, m}})
    kernels::rms_norm_q8(q, rr.second - rr.first, H, dres + static_cast<size_t>(rr.first) * H, dadd + static_cast<size_t>(rr.first) * H, nullptr, dw, 1e-6f,
                         xn2 + static_cast<size_t>(rr.first) * H, kernels::RmsNormQ8{xq + static_cast<size_t>(rr.first) * H, xs + rr.first, ld8});
  q.wait_and_throw();
  const auto q2 = get(xq, static_cast<size_t>(m) * H);
  const auto s2 = get(xs, static_cast<size_t>(H / 128) * ld8);
  bool zero_ok = true;
  for (int g = 0; g < H / 128; ++g) zero_ok &= s1[static_cast<size_t>(g) * ld8 + 7] == 0.f;
  expect(same_bytes(x1, x2) && same_bytes(res1, res2) && same_bytes(q1, q2) && same_bytes(s1, s2) && zero_ok,
         "rms_norm_q8 == rms_norm (xn, residual) bit for bit; two row parts == one call (codes, scales); a zero row scales 0");
  // the GEMMs: own quantiser vs the norm's codes, plain + GLU, at row offsets
  kernels::w4a16_pf_set_a8(true);
  kernels::w4a16_pf_a8_input_clear(q);
  bool all = true;
  char what[200];
  for (const auto off : {std::pair<int, int>{0, m}, std::pair<int, int>{100, 36}, std::pair<int, int>{2, 37}, std::pair<int, int>{148, 152}}) {
    const int r0 = off.first, mm = off.second;
    const bf16 *a = xn + static_cast<size_t>(r0) * H;
    kernels::w4a16_pf_a8_input_clear(q);
    kernels::w4a16_gemm_pf(q, mm, a, H, ww, ws, c1, ldc, n, H);
    kernels::w4a16_gemm_pf_swiglu(q, mm, a, H, ww, ws, h1, nh, nh, H);
    kernels::w4a16_pf_a8_input(q, xn, H, m, H, xq, xs, ld8);
    kernels::w4a16_gemm_pf(q, mm, a, H, ww, ws, c2, ldc, n, H);
    kernels::w4a16_gemm_pf_swiglu(q, mm, a, H, ww, ws, h2, nh, nh, H);
    q.wait_and_throw();
    const bool ok = same_bytes(get(c1, static_cast<size_t>(mm) * ldc), get(c2, static_cast<size_t>(mm) * ldc)) &&
                    same_bytes(get(h1, static_cast<size_t>(mm) * nh), get(h2, static_cast<size_t>(mm) * nh));
    std::snprintf(what, sizeof what, "W4A8 pf with the norm's codes == its own quantiser, plain + GLU, rows %d..%d (%s)", r0, r0 + mm,
                  r0 % 4 ? "fallback" : "codes read");
    expect(ok, what);
    all &= ok;
  }
  {  // the codes are what the GEMM reads: zeroed codes on a registered input give a zero result (then a clear restores it)
    kernels::w4a16_pf_a8_input(q, xn, H, m, H, xq, xs, ld8);
    q.memset(xq, 0, static_cast<size_t>(m) * H).wait();
    kernels::w4a16_gemm_pf(q, 36, xn + static_cast<size_t>(100) * H, H, ww, ws, c2, ldc, n, H).wait_and_throw();
    const auto z = get(c2, static_cast<size_t>(36) * ldc);
    bool zeros = true;
    for (float v : z) zeros &= v == 0.f;
    kernels::w4a16_pf_a8_input_clear(q);
    kernels::w4a16_gemm_pf(q, 36, xn + static_cast<size_t>(100) * H, H, ww, ws, c2, ldc, n, H).wait_and_throw();
    kernels::w4a16_gemm_pf(q, 36, xn + static_cast<size_t>(100) * H, H, ww, ws, c1, ldc, n, H).wait_and_throw();
    expect(zeros && same_bytes(get(c1, static_cast<size_t>(36) * ldc), get(c2, static_cast<size_t>(36) * ldc)) && all,
           "W4A8 pf: a registered input reads the registered codes; after w4a16_pf_a8_input_clear it quantises again");
  }
  kernels::w4a16_pf_set_a8(false);
  for (void *pp : {(void *)dres, (void *)dadd, (void *)dw, (void *)xn, (void *)xn2, (void *)xq, (void *)xs, (void *)ww, (void *)ws, (void *)c1, (void *)c2,
                   (void *)h1, (void *)h2})
    dev.release(pp);
}

// the W4A8 prefill GEMM (the nib blocks reordered into s4 DPAS operands while staged).
// - W4A8: lossless activations (integers with a +-127 in every (row, group): the quantiser's scale is 1, codes = values)
// and power-of-two weight scales -> bit-exact vs w4a8_ref at m 1..300 on every shape, padding untouched, a row's bits
// independent of m; full entropy: within a bound of the bf16-activation result (the int8 activation error), run-to-run
// and shape identical; the GLU form == plain + swiglu where |g| < 60; a matrix in the keep set runs W4A16.
void test_w4a8(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  // W4A8, lossless
  kernels::w4a16_pf_a8_reserve(q, 512, 2048);
  for (const auto nk : {std::pair<int, int>{48, 256}, std::pair<int, int>{48, 384}, std::pair<int, int>{640, 1280}}) {  // k 384 = an odd group count (a short last stage)
    const int n = nk.first, k = nk.second, cap = 300, lda = k + 64, ldc = n + 16, G = k / 128;
    std::mt19937 rng(2603 + n);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    std::vector<float> scf(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {
      const float m = std::ldexp(1.f, -static_cast<int>(rng() % 4));
      sc[i] = sycl::half(rng() % 2 ? -m : m);
      scf[i] = static_cast<float>(sc[i]);
    }
    std::vector<int8_t> aq(static_cast<size_t>(cap) * k);
    std::vector<bf16> a(static_cast<size_t>(cap) * lda, bf16(0.f));
    for (int r = 0; r < cap; ++r)
      for (int i = 0; i < k; ++i) {
        int v = static_cast<int>(rng() % 255) - 127;
        if (i % 128 == (r * 7) % 128) v = rng() % 2 ? 127 : -127;  // the group's amax: scale 1
        if (std::abs(v) == 127 && i % 128 != (r * 7) % 128) v = 100;
        aq[static_cast<size_t>(r) * k + i] = static_cast<int8_t>(v);
        a[static_cast<size_t>(r) * lda + i] = bf16(static_cast<float>(v));
      }
    std::vector<float> as(static_cast<size_t>(G) * cap, 1.f);
    std::vector<double> want(static_cast<size_t>(cap) * n);
    kernels::w4a8_ref(aq.data(), as.data(), cap, qw.data(), scf.data(), cap, n, k, want.data());
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(static_cast<size_t>(cap + 8) * ldc, MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    kernels::w4a16_pf_set_a8(true);
    kernels::w4a16_pf_set_a8_body(1);  // shape -1 = the SLM-free body (the exact cases); shapes 0-3 = the pipelined SLM body
    std::vector<float> full;
    for (int m : {1, 7, 36, 129, 300})
      for (int shape : {-1, 0, 1, 2, 3}) {
        if (m != 300 && shape >= 0) continue;
        kernels::w4a16_pf_set_a8_shape(shape);
        q.fill(wc, -12345.f, static_cast<size_t>(cap + 8) * ldc).wait();
        kernels::w4a16_gemm_pf(q, m, wa, lda, ww, ws, wc, ldc, n, k).wait_and_throw();
        std::vector<float> got(static_cast<size_t>(cap + 8) * ldc);
        q.memcpy(got.data(), wc, got.size() * 4).wait();
        bool exact = true, dead = true;
        for (int r = 0; r < cap + 8; ++r)
          for (int i = 0; i < ldc; ++i) {
            const float g = got[static_cast<size_t>(r) * ldc + i];
            if (r < m && i < n) exact &= static_cast<double>(g) == want[static_cast<size_t>(r) * n + i];
            else dead &= g == -12345.f;
          }
        if (m == 300 && shape == 0) full = got;
        char what[160];
        std::snprintf(what, sizeof what, "W4A8 pf m=%d N=%d K=%d shape %d bit-exact vs w4a8_ref (lossless activations), padding untouched", m, n, k, shape);
        expect(exact && dead, what);
      }
    {
      kernels::w4a16_pf_set_a8_shape(-1);
      q.fill(wc, -12345.f, static_cast<size_t>(cap + 8) * ldc).wait();
      kernels::w4a16_gemm_pf(q, 36, wa + static_cast<size_t>(100) * lda, lda, ww, ws, wc + static_cast<size_t>(100) * ldc, ldc, n, k).wait_and_throw();
      std::vector<float> got(static_cast<size_t>(cap + 8) * ldc);
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      bool same = true;
      for (int r = 100; r < 136; ++r)
        for (int i = 0; i < n; ++i) same &= got[static_cast<size_t>(r) * ldc + i] == full[static_cast<size_t>(r) * ldc + i];
      expect(same, "W4A8 pf: a row's bits do not depend on m or its place in the call");
    }
    kernels::w4a16_pf_set_a8_body(0);
    kernels::w4a16_pf_set_a8(false);
    for (void *pp : {(void *)wa, (void *)ww, (void *)ws, (void *)wc}) dev.release(pp);
  }
  {  // W4A8 full entropy: vs the W4A16 result of the same call; shapes / runs identical; GLU == plain + swiglu
    const int nh = 640, n = 2 * nh, k = 1280, m = 300;
    std::mt19937 rng(2604);
    std::normal_distribution<float> nd(0.f, 1.f);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    for (auto &v : sc) v = sycl::half(nd(rng) * 0.02f);
    std::vector<bf16> a(static_cast<size_t>(m) * k);
    for (auto &v : a) v = bf16(nd(rng));
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(static_cast<size_t>(m) * n, MK::Device);
    auto *wh = dev.alloc<bf16>(static_cast<size_t>(m) * nh, MK::Device);
    auto *wf = dev.alloc<bf16>(static_cast<size_t>(m) * nh, MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    std::vector<float> ref(static_cast<size_t>(m) * n), first;
    kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k).wait_and_throw();
    q.memcpy(ref.data(), wc, ref.size() * 4).wait();
    kernels::w4a16_pf_set_a8(true);
    {  // a matrix in the keep set stays W4A16 (both entry points)
      kernels::w4a16_pf_a8_keep(ww);
      kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k).wait_and_throw();
      std::vector<float> got(ref.size());
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      kernels::w4a16_pf_set_a8(false);
      kernels::w4a16_gemm_pf_swiglu(q, m, wa, k, ww, ws, wh, nh, nh, k).wait_and_throw();
      kernels::w4a16_pf_set_a8(true);
      kernels::w4a16_gemm_pf_swiglu(q, m, wa, k, ww, ws, wf, nh, nh, k).wait_and_throw();
      std::vector<uint16_t> h0(static_cast<size_t>(m) * nh), h1(h0.size());
      q.memcpy(h0.data(), wh, h0.size() * 2);
      q.memcpy(h1.data(), wf, h1.size() * 2).wait();
      kernels::w4a16_pf_a8_keep_clear();
      expect(got == ref && h0 == h1, "W4A8 keep set: a kept matrix runs W4A16 under W4A8 (pf and pf_swiglu bit-identical to W4A8 off)");
    }
    bool same = true;
    double num = 0, den = 0;
    for (int shape : {-1, 0, 1, 2, 3, 5, 0}) {  // 5: the 24 x 4 HALF body
      kernels::w4a16_pf_set_a8_shape(shape);
      kernels::w4a16_gemm_pf(q, m, wa, k, ww, ws, wc, n, n, k).wait_and_throw();
      std::vector<float> got(ref.size());
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      if (first.empty()) {
        first = got;
        for (size_t i = 0; i < got.size(); ++i) num += (got[i] - ref[i]) * (got[i] - ref[i]), den += ref[i] * ref[i];
      } else {
        same &= got == first;
      }
    }
    const double rel = std::sqrt(num / den);
    char what[200];
    std::snprintf(what, sizeof what, "W4A8 pf full entropy m=%d: rel. rms vs W4A16 %.2e <= 1.5e-2 (the int8 activation error), every shape and run bit-identical", m, rel);
    expect(rel <= 1.5e-2 && same, what);
    kernels::w4a16_pf_set_a8_shape(-1);
    {  // the SLM-free body (body 1) == the pipelined SLM body, GLU too, on both of its tile shapes (<= 64 rows, above);
       // body 3 (SLM-free up to 128 rows, the SLM body above: both sides of its threshold)
      bool eq = true;
      for (int mm : {37, 64, 65, 300}) {
        std::vector<std::vector<uint16_t>> hb(3, std::vector<uint16_t>(static_cast<size_t>(mm) * nh));
        std::vector<std::vector<float>> cb(3, std::vector<float>(static_cast<size_t>(mm) * n));
        const int bodies[3] = {0, 1, 3};
        for (int bi = 0; bi < 3; ++bi) {
          kernels::w4a16_pf_set_a8_body(bodies[bi]);
          kernels::w4a16_gemm_pf(q, mm, wa, k, ww, ws, wc, n, n, k);
          kernels::w4a16_gemm_pf_swiglu(q, mm, wa, k, ww, ws, wf, nh, nh, k).wait_and_throw();
          q.memcpy(cb[bi].data(), wc, cb[bi].size() * 4);
          q.memcpy(hb[bi].data(), wf, hb[bi].size() * 2).wait();
        }
        eq &= cb[0] == cb[1] && hb[0] == hb[1] && cb[0] == cb[2] && hb[0] == hb[2];
      }
      kernels::w4a16_pf_set_a8_body(0);
      expect(eq, "W4A8 pf: the SLM-free body and body 3 (SLM-free to 128 rows) == the round-33 body bit for bit (plain + GLU, m 37 / 64 / 65 / 300)");
    }
    {  // the 24 x 4 HALF body (shape 5) == 32 x 2 (shape 0), GLU too, a partial row group (37) and several (300)
      bool eq = true;
      for (int mm : {37, 300}) {
        std::vector<std::vector<uint16_t>> hb(2, std::vector<uint16_t>(static_cast<size_t>(mm) * nh));
        std::vector<std::vector<float>> cb(2, std::vector<float>(static_cast<size_t>(mm) * n));
        for (int i = 0; i < 2; ++i) {
          kernels::w4a16_pf_set_a8_shape(i == 0 ? 0 : 5);
          kernels::w4a16_gemm_pf(q, mm, wa, k, ww, ws, wc, n, n, k);
          kernels::w4a16_gemm_pf_swiglu(q, mm, wa, k, ww, ws, wf, nh, nh, k).wait_and_throw();
          q.memcpy(cb[i].data(), wc, cb[i].size() * 4);
          q.memcpy(hb[i].data(), wf, hb[i].size() * 2).wait();
        }
        eq &= cb[0] == cb[1] && hb[0] == hb[1];
      }
      kernels::w4a16_pf_set_a8_shape(-1);
      expect(eq, "W4A8 pf: the 24 x 4 HALF body == 32 x 2 bit for bit (plain + GLU, m 37 / 300)");
    }
    for (int mm : {37, 300}) {
      kernels::w4a16_gemm_pf(q, mm, wa, k, ww, ws, wc, n, n, k);
      kernels::swiglu(q, mm, nh, wc, wh);
      kernels::w4a16_gemm_pf_swiglu(q, mm, wa, k, ww, ws, wf, nh, nh, k).wait_and_throw();
      std::vector<uint16_t> h0(static_cast<size_t>(mm) * nh), h1(h0.size());
      std::vector<float> gu(static_cast<size_t>(mm) * n);
      q.memcpy(h0.data(), wh, h0.size() * 2);
      q.memcpy(h1.data(), wf, h1.size() * 2);
      q.memcpy(gu.data(), wc, gu.size() * 4).wait();
      size_t off_in = 0;
      for (size_t i = 0; i < h0.size(); ++i) off_in += h1[i] != h0[i] && std::fabs(gu[(i / nh) * n + i % nh]) < 60.f;
      std::snprintf(what, sizeof what, "W4A8 pf_swiglu m=%d == W4A8 pf + swiglu bit for bit where |g| < 60 (%zu off)", mm, off_in);
      expect(off_in == 0, what);
    }
    kernels::w4a16_pf_set_a8(false);
    for (void *pp : {(void *)wa, (void *)ww, (void *)ws, (void *)wc, (void *)wh, (void *)wf}) dev.release(pp);
  }
}

// Affine int4 g64 (arch flint): w4a16_gemm_rows on a matrix registered with w4_set_affine == w4aff_ref bit for bit on
// exact inputs (integer activations, power-of-two scales and biases of both signs, every code) at 1 / 2 / 4 / 5 / 8 rows and
// k splits auto / 1 / 2 / 4, rows past the shape untouched; on real-valued bf16 data within 2e-6 of max |ref| per row and
// run-to-run identical; w4a16_gemm_rows_swiglu == w4a16_gemm_rows + swiglu bit for bit; a cleared registration reads the
// symmetric g128 path again. (The many-row GEMM: test_w4aff_pf.)
void test_w4aff(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  const int n = 640, k = 1280, cap = 8, G = k / kernels::kW4AffGroup;
  std::mt19937 rng(6464);
  std::vector<uint32_t> ct(static_cast<size_t>(n) * (k / 8));
  for (auto &v : ct) v = static_cast<uint32_t>(rng());
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2);
  kernels::pack_w4_ct_rows(ct.data(), n, k, 0, n, packed.data());
  auto bf_bits = [](float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return static_cast<uint16_t>(u >> 16);
  };
  auto bf_round = [&](float v) {  // to the nearest bf16
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  auto *wa = dev.alloc<bf16>(static_cast<size_t>(cap) * k, MK::Device);
  auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
  auto *ws = dev.alloc<uint16_t>(static_cast<size_t>(G) * n, MK::Device);
  auto *wb = dev.alloc<uint16_t>(static_cast<size_t>(G) * n, MK::Device);
  auto *wc = dev.alloc<float>(static_cast<size_t>(cap) * n, MK::Device);
  auto *wh = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  auto *wf = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  auto *tab = dev.alloc<float>(kernels::kSiluTabSize, MK::Device);
  q.memcpy(ww, packed.data(), packed.size());
  kernels::swiglu_table(q, tab).wait_and_throw();
  kernels::w4_set_affine(ww, wb);
  const auto *wsh = reinterpret_cast<const sycl::half *>(ws);
  // exact: false = real-valued data (bound + run-to-run identity)
  auto run_case = [&](bool exact_in, bool &exact, bool &dead, double &worst, bool &stable) {
    std::vector<float> scf(static_cast<size_t>(n) * G), bsf(scf.size());
    std::vector<uint16_t> sc(static_cast<size_t>(G) * n), bs(sc.size());
    for (int c = 0; c < n; ++c)
      for (int g = 0; g < G; ++g) {
        float s, b;
        if (exact_in) {
          s = std::ldexp(1.f, -static_cast<int>(rng() % 4));
          b = (rng() & 1 ? -1.f : 1.f) * std::ldexp(1.f, -static_cast<int>(rng() % 5));
        } else {
          s = bf_round(std::ldexp(static_cast<float>(rng() % 1000 + 1) / 1000.f, -10));
          b = bf_round(-8.f * s * static_cast<float>(rng() % 1000) / 1000.f - 2.f * s);
        }
        scf[static_cast<size_t>(c) * G + g] = s, bsf[static_cast<size_t>(c) * G + g] = b;
        sc[static_cast<size_t>(g) * n + c] = bf_bits(s), bs[static_cast<size_t>(g) * n + c] = bf_bits(b);
      }
    std::vector<bf16> a(static_cast<size_t>(cap) * k);
    std::vector<float> af(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      af[i] = exact_in ? static_cast<float>(static_cast<int>(rng() % 9) - 4) : bf_round(std::normal_distribution<float>(0.f, 1.f)(rng));
      a[i] = bf16(af[i]);
    }
    std::vector<double> want(static_cast<size_t>(cap) * n);
    kernels::w4aff_ref(af.data(), ct.data(), scf.data(), bsf.data(), cap, n, k, want.data());
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ws, sc.data(), sc.size() * 2);
    q.memcpy(wb, bs.data(), bs.size() * 2).wait();
    for (const int ks : {0, 1, 2, 4})
      for (int rows : {1, 2, 4, 5, 8}) {
        std::vector<float> got(want.size()), again(want.size());
        for (int rep = 0; rep < 2; ++rep) {
          q.fill(wc, -12345.f, want.size()).wait();
          kernels::w4a16_gemm_rows(q, rows, wa, ww, wsh, wc, n, k, ks).wait_and_throw();
          q.memcpy(rep ? again.data() : got.data(), wc, got.size() * 4).wait();
        }
        stable &= got == again;
        const int shape = kernels::row_shape(rows);
        for (int r = 0; r < rows; ++r) {
          double mx = 0;
          for (int c = 0; c < n; ++c) mx = std::max(mx, std::abs(want[static_cast<size_t>(r) * n + c]));
          for (int c = 0; c < n; ++c) {
            const size_t i = static_cast<size_t>(r) * n + c;
            if (exact_in) exact &= static_cast<double>(got[i]) == want[i];
            else worst = std::max(worst, std::isfinite(got[i]) ? std::abs(got[i] - want[i]) / std::max(mx, 1e-30) : HUGE_VAL);
          }
        }
        for (size_t i = static_cast<size_t>(shape) * n; i < got.size(); ++i) dead &= got[i] == -12345.f;
      }
  };
  bool exact = true, dead = true, stable = true;
  double worst = 0;
  run_case(true, exact, dead, worst, stable);
  expect(exact && dead && stable,
         "w4a16_gemm_rows MLX affine g64 N=640 K=1280: rows 1/2/4/5/8 x k splits auto/1/2/4 bit-exact vs w4aff_ref, rows past the shape untouched");
  bool ex2 = true;
  run_case(false, ex2, dead, worst, stable);
  std::printf("  w4aff real-valued data: worst |got - ref| / max |ref| per row = %.3g\n", worst);
  expect(worst < 2e-6 && stable && dead, "w4a16_gemm_rows MLX affine g64 on real-valued data: within 2e-6 of max |ref| per row, run-to-run identical");
  bool same = true;
  for (const int ks : {0, 2})
    for (int rows : {1, 3, 8}) {
      q.fill(wh, bf16(-3.f), static_cast<size_t>(cap) * n / 2);
      q.fill(wf, bf16(-3.f), static_cast<size_t>(cap) * n / 2).wait();
      kernels::w4a16_gemm_rows(q, rows, wa, ww, wsh, wc, n, k, ks);
      kernels::swiglu(q, rows, n / 2, wc, wh);
      kernels::w4a16_gemm_rows_swiglu(q, rows, wa, ww, wsh, tab, wf, n / 2, k, ks).wait_and_throw();
      std::vector<uint16_t> h0(static_cast<size_t>(cap) * n / 2), h1(h0.size());
      q.memcpy(h0.data(), wh, h0.size() * 2);
      q.memcpy(h1.data(), wf, h1.size() * 2).wait();
      for (size_t i = 0; i < static_cast<size_t>(rows) * n / 2; ++i) same &= h0[i] == h1[i];
    }
  expect(same, "w4a16_gemm_rows_swiglu MLX affine == w4a16_gemm_rows + swiglu bit for bit (rows 1/3/8, k splits auto/2)");
  kernels::w4_clear_affine(ww);
  expect(kernels::w4_affine(ww) == nullptr, "w4_clear_affine: the matrix reads as symmetric g128 again");
  dev.release(wa);
  dev.release(ww);
  dev.release(ws);
  dev.release(wb);
  dev.release(wc);
  dev.release(wh);
  dev.release(wf);
  dev.release(tab);
}

// The many-row GEMMs on an affine g64 matrix (w4a16_gemm_pf / _swiglu on a w4_set_affine registration and
// w4aff_gemm_pf with explicit biases). Exact inputs (integer activations, power-of-two scales and biases of both signs:
// every sum and every bf16(q s + b) exact) at m 1 / 37 / 64 / 300, every exact-mode shape and every dequantised shape:
// bit-exact vs w4aff_ref; real-valued data: the exact mode within 2e-6 of max |ref| per row, the dequantised mode within
// 1e-5 of a double reference on bf16(q s + b) weights; a row's bits do not depend on m or its place in the call; every
// shape the same bits; the swiglu GEMM == the plain GEMM + swiglu bit for bit; explicit biases == the registration.
void test_w4aff_pf(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  const int n = 640, k = 1280, cap = 300, G = k / kernels::kW4AffGroup;
  std::mt19937 rng(6465);
  std::vector<uint32_t> ct(static_cast<size_t>(n) * (k / 8));
  for (auto &v : ct) v = static_cast<uint32_t>(rng());
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2);
  kernels::pack_w4_ct_rows(ct.data(), n, k, 0, n, packed.data());
  auto bf_bits = [](float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return static_cast<uint16_t>(u >> 16);
  };
  auto bf_round = [&](float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  auto *wa = dev.alloc<bf16>(static_cast<size_t>(cap + 100) * k, MK::Device);
  auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
  auto *ws = dev.alloc<uint16_t>(static_cast<size_t>(G) * n, MK::Device);
  auto *wb = dev.alloc<uint16_t>(static_cast<size_t>(G) * n, MK::Device);
  auto *wc = dev.alloc<float>(static_cast<size_t>(cap + 100) * n, MK::Device);
  auto *wh = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  auto *wf = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  q.memcpy(ww, packed.data(), packed.size());
  kernels::w4a16_pf_aff_reserve(q, cap + 100, k);
  kernels::w4_set_affine(ww, wb);
  const auto *wsh = reinterpret_cast<const sycl::half *>(ws);
  for (const bool exact_in : {true, false}) {
    std::vector<float> scf(static_cast<size_t>(n) * G), bsf(scf.size());
    std::vector<uint16_t> sc(static_cast<size_t>(G) * n), bs(sc.size());
    for (int c = 0; c < n; ++c)
      for (int g = 0; g < G; ++g) {
        float sv, bv;
        if (exact_in) {
          sv = std::ldexp(1.f, -static_cast<int>(rng() % 4));
          bv = (rng() & 1 ? -1.f : 1.f) * std::ldexp(1.f, -static_cast<int>(rng() % 5));
        } else {
          sv = bf_round(std::ldexp(static_cast<float>(rng() % 1000 + 1) / 1000.f, -10));
          bv = bf_round(-8.f * sv * static_cast<float>(rng() % 1000) / 1000.f - 2.f * sv);
        }
        scf[static_cast<size_t>(c) * G + g] = sv, bsf[static_cast<size_t>(c) * G + g] = bv;
        sc[static_cast<size_t>(g) * n + c] = bf_bits(sv), bs[static_cast<size_t>(g) * n + c] = bf_bits(bv);
      }
    std::vector<bf16> a(static_cast<size_t>(cap + 100) * k);
    std::vector<float> af(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      af[i] = exact_in ? static_cast<float>(static_cast<int>(rng() % 9) - 4) : bf_round(std::normal_distribution<float>(0.f, 1.f)(rng));
      a[i] = bf16(af[i]);
    }
    std::vector<double> want(static_cast<size_t>(cap) * n), wdq(want.size());
    kernels::w4aff_ref(af.data(), ct.data(), scf.data(), bsf.data(), cap, n, k, want.data());
    {  // the dequantised mode's reference: double sums over bf16(q s + b)
      std::vector<float> wdeq(static_cast<size_t>(n) * k);
      for (int c = 0; c < n; ++c)
        for (int kk = 0; kk < k; ++kk) {
          const size_t gi = static_cast<size_t>(c) * G + kk / kernels::kW4AffGroup;
          wdeq[static_cast<size_t>(c) * k + kk] = bf_round(static_cast<float>(kernels::ct_code(ct.data(), k, c, kk)) * scf[gi] + bsf[gi]);
        }
      for (int r = 0; r < cap; ++r)
        for (int c = 0; c < n; ++c) {
          double d = 0;
          for (int kk = 0; kk < k; ++kk) d += static_cast<double>(af[static_cast<size_t>(r) * k + kk]) * wdeq[static_cast<size_t>(c) * k + kk];
          wdq[static_cast<size_t>(r) * n + c] = d;
        }
    }
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ws, sc.data(), sc.size() * 2);
    q.memcpy(wb, bs.data(), bs.size() * 2).wait();
    for (const bool dq : {false, true}) {
      kernels::w4a16_pf_set_dequant(dq);
      const std::vector<int> shapes = dq ? std::vector<int>{-1, 12, 13, 14, 15} : std::vector<int>{-1, 0, 2, 4, 7};
      bool exact = true, same = true, dead = true;
      double worst = 0;
      std::vector<float> first;
      for (const int m : {1, 37, 64, 300})
        for (const int shape : shapes) {
          q.fill(wc, -12345.f, static_cast<size_t>(cap + 100) * n).wait();
          kernels::w4a16_gemm_pf(q, m, wa, k, ww, wsh, wc, n, n, k, shape).wait_and_throw();
          std::vector<float> got(static_cast<size_t>(cap + 100) * n);
          q.memcpy(got.data(), wc, got.size() * 4).wait();
          for (size_t i = static_cast<size_t>(m) * n; i < got.size(); ++i) dead &= got[i] == -12345.f;
          if (m == 300) {
            if (first.empty()) first.assign(got.begin(), got.begin() + static_cast<size_t>(m) * n);
            else if (!std::equal(first.begin(), first.end(), got.begin())) {
              size_t nd = 0;
              for (size_t i = 0; i < first.size(); ++i) nd += first[i] != got[i];
              std::printf("  w4aff_gemm_pf%s shape %d at 300 rows: %zu of %zu differ from shape %d\n", dq ? " dq" : "", shape, nd, first.size(), shapes[0]);
              same = false;
            }
          }
          const auto &ref = dq ? wdq : want;
          for (int r = 0; r < m; ++r) {
            double mx = 0;
            for (int c = 0; c < n; ++c) mx = std::max(mx, std::abs(ref[static_cast<size_t>(r) * n + c]));
            for (int c = 0; c < n; ++c) {
              const size_t i = static_cast<size_t>(r) * n + c;
              if (exact_in) exact &= static_cast<double>(got[i]) == want[i];
              else worst = std::max(worst, std::isfinite(got[i]) ? std::abs(got[i] - ref[i]) / std::max(mx, 1e-30) : HUGE_VAL);
            }
          }
        }
      // a row's bits do not depend on m or its place: rows 100..135 alone == rows 100..135 of the 300-row call
      q.fill(wc, -1.f, static_cast<size_t>(cap + 100) * n).wait();
      kernels::w4a16_gemm_pf(q, 36, wa + static_cast<size_t>(100) * k, k, ww, wsh, wc, n, n, k).wait_and_throw();
      std::vector<float> part(static_cast<size_t>(36) * n);
      q.memcpy(part.data(), wc, part.size() * 4).wait();
      bool place = std::equal(part.begin(), part.end(), first.begin() + static_cast<size_t>(100) * n);
      // explicit biases == the registration
      kernels::w4aff_gemm_pf(q, 300, wa, k, ww, ws, wb, wc, n, n, k).wait_and_throw();
      std::vector<float> ex(static_cast<size_t>(300) * n);
      q.memcpy(ex.data(), wc, ex.size() * 4).wait();
      place &= ex == first;
      // swiglu == plain + swiglu (h rows only)
      bool glu = true;  // (real-valued data: the epilogue's exp_cw and swiglu's exp agree where |g| < 60; exact inputs reach thousands)
      for (const int m : {37, 300}) {
        if (exact_in) break;
        kernels::w4a16_gemm_pf(q, m, wa, k, ww, wsh, wc, n, n, k);
        kernels::swiglu(q, m, n / 2, wc, wh);
        kernels::w4a16_gemm_pf_swiglu(q, m, wa, k, ww, wsh, wf, n / 2, n / 2, k).wait_and_throw();
        std::vector<uint16_t> h0(static_cast<size_t>(m) * n / 2), h1(h0.size());
        q.memcpy(h0.data(), wh, h0.size() * 2);
        q.memcpy(h1.data(), wf, h1.size() * 2).wait();
        glu &= h0 == h1;
      }
      char what[320];
      if (exact_in)
        std::snprintf(what, sizeof what, "w4aff_gemm_pf%s exact inputs N=640 K=1280 m 1/37/64/300 x every shape: bit-exact vs w4aff_ref, rows past m untouched",
                      dq ? " dequantised" : "");
      else
        std::snprintf(what, sizeof what, "w4aff_gemm_pf%s real-valued: worst |err| / max |ref| %.2e <= %.0e, rows past m untouched", dq ? " dequantised" : "", worst,
                      dq ? 1e-5 : 2e-6);
      expect((exact_in ? exact : worst <= (dq ? 1e-5 : 2e-6)) && dead, what);
      std::snprintf(what, sizeof what,
                    "w4aff_gemm_pf%s%s: every shape the same bits (%d), a row's bits independent of m / place + explicit biases == registration (%d), swiglu == plain + swiglu (%d)",
                    dq ? " dequantised" : "", exact_in ? " (exact inputs)" : "", same ? 1 : 0, place ? 1 : 0, glu ? 1 : 0);
      expect(same && place && glu, what);
    }
    kernels::w4a16_pf_set_dequant(false);
  }
  kernels::w4_clear_affine(ww);
  for (void *pp : {(void *)wa, (void *)ww, (void *)ws, (void *)wb, (void *)wc, (void *)wh, (void *)wf}) dev.release(pp);
}

// int4 g32 with zero points (arch llama): w4a16_gemm_rows on a registered matrix == w4zp_ref bit for bit on exact inputs
// (integer activations, power-of-two scales, every code and zero point 0..15) at 1 / 2 / 4 / 5 / 8 rows and k splits
// auto / 1 / 2 / 4, rows past the shape untouched; w4a16_gemm_rows_swiglu == w4a16_gemm_rows + swiglu bit for bit; the
// many-row GEMM refuses the matrix; a cleared registration reads the symmetric g128 path again.
void test_w4zp(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  const int n = 640, k = 1280, cap = 8, G = k / kernels::kW4ZpGroup;
  std::mt19937 rng(4242);
  std::vector<uint32_t> ct(static_cast<size_t>(n) * (k / 8));  // compressed-tensors words [n][k/8], raw q
  for (auto &v : ct) v = static_cast<uint32_t>(rng());
  std::vector<uint8_t> z(static_cast<size_t>(n) * G);
  for (auto &v : z) v = static_cast<uint8_t>(rng() % 16);
  std::vector<float> scf(static_cast<size_t>(n) * G);  // [n][G]
  std::vector<sycl::half> sc(static_cast<size_t>(G) * n);  // card: [G][n]
  for (int c = 0; c < n; ++c)
    for (int g = 0; g < G; ++g) {
      const float v = std::ldexp(1.f, -static_cast<int>(rng() % 4));
      scf[static_cast<size_t>(c) * G + g] = v;
      sc[static_cast<size_t>(g) * n + c] = sycl::half(v);
    }
  std::vector<bf16> a(static_cast<size_t>(cap) * k);
  std::vector<float> af(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    af[i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
    a[i] = bf16(af[i]);
  }
  std::vector<uint8_t> packed(static_cast<size_t>(n) * k / 2), zp(static_cast<size_t>(G / 2) * n);
  kernels::pack_w4_ct_rows(ct.data(), n, k, 0, n, packed.data());
  kernels::pack_w4_zeros(z.data(), n, k, zp.data());
  std::vector<double> want(static_cast<size_t>(cap) * n);
  kernels::w4zp_ref(af.data(), ct.data(), scf.data(), z.data(), cap, n, k, want.data());
  auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
  auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
  auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
  auto *wz = dev.alloc<uint8_t>(zp.size(), MK::Device);
  auto *wc = dev.alloc<float>(want.size(), MK::Device);
  auto *wh = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  auto *wf = dev.alloc<bf16>(static_cast<size_t>(cap) * n / 2, MK::Device);
  auto *tab = dev.alloc<float>(kernels::kSiluTabSize, MK::Device);
  q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
  q.memcpy(ww, packed.data(), packed.size());
  q.memcpy(wz, zp.data(), zp.size());
  q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half));
  kernels::swiglu_table(q, tab).wait_and_throw();
  kernels::w4_set_zero_points(ww, wz);
  bool exact = true, dead = true;
  for (const int ks : {0, 1, 2, 4})
    for (int rows : {1, 2, 4, 5, 8}) {
      q.fill(wc, -12345.f, want.size()).wait();
      kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k, ks).wait_and_throw();
      std::vector<float> got(want.size());
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      const int shape = kernels::row_shape(rows);
      for (size_t i = 0; i < got.size(); ++i) {
        if (i < static_cast<size_t>(rows) * n) exact &= static_cast<double>(got[i]) == want[i];
        else if (i >= static_cast<size_t>(shape) * n) dead &= got[i] == -12345.f;
      }
    }
  expect(exact && dead, "w4a16_gemm_rows zero points g32 N=640 K=1280: rows 1/2/4/5/8 x k splits auto/1/2/4 bit-exact vs w4zp_ref, rows past the shape untouched");
  bool same = true;
  for (const int ks : {0, 2})
    for (int rows : {1, 3, 8}) {
      q.fill(wh, bf16(-3.f), static_cast<size_t>(cap) * n / 2);
      q.fill(wf, bf16(-3.f), static_cast<size_t>(cap) * n / 2).wait();
      kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k, ks);
      kernels::swiglu(q, rows, n / 2, wc, wh);
      kernels::w4a16_gemm_rows_swiglu(q, rows, wa, ww, ws, tab, wf, n / 2, k, ks).wait_and_throw();
      std::vector<uint16_t> h0(static_cast<size_t>(cap) * n / 2), h1(h0.size());
      q.memcpy(h0.data(), wh, h0.size() * 2);
      q.memcpy(h1.data(), wf, h1.size() * 2).wait();
      for (size_t i = 0; i < static_cast<size_t>(rows) * n / 2; ++i) same &= h0[i] == h1[i];
    }
  expect(same, "w4a16_gemm_rows_swiglu zero points == w4a16_gemm_rows + swiglu bit for bit (rows 1/3/8, k splits auto/2)");
  {  // the many-row GEMM (prefill): the dequantised mode with zero points; exact inputs (power-of-two scales, small
     // integer activations: bf16((q - z) s) exact, fp32 sums exact) -> bit-exact vs w4zp_ref at any shape and row count
    const int M = 513;
    std::vector<bf16> am(static_cast<size_t>(M) * k);
    std::vector<float> amf(am.size());
    for (size_t i = 0; i < am.size(); ++i) {
      amf[i] = std::ldexp(static_cast<float>(static_cast<int>(rng() % 9) - 4), -6);  // x 2^-6: still exact, gates mostly |g| < 60
      am[i] = bf16(amf[i]);
    }
    std::vector<double> wantm(static_cast<size_t>(M) * n);
    kernels::w4zp_ref(amf.data(), ct.data(), scf.data(), z.data(), M, n, k, wantm.data());
    auto *dam = dev.alloc<bf16>(am.size(), MK::Device);
    auto *dcm = dev.alloc<float>(wantm.size(), MK::Device);
    auto *dh0 = dev.alloc<bf16>(static_cast<size_t>(M) * n / 2, MK::Device);
    auto *dh1 = dev.alloc<bf16>(static_cast<size_t>(M) * n / 2, MK::Device);
    q.memcpy(dam, am.data(), am.size() * sizeof(bf16)).wait();
    bool pf_exact = true, pf_glu = true;
    size_t glu_in = 0;  // compared elements (|g| < 60: the established GLU criterion, exp_cw vs the library exp)
    for (const int shape : {-1, 12, 13, 14, 15})
      for (const int m : {1, 7, 64, 200, 513}) {
        q.fill(dcm, -12345.f, wantm.size()).wait();
        kernels::w4a16_gemm_pf(q, m, dam, k, ww, ws, dcm, n, n, k, shape).wait_and_throw();
        std::vector<float> got(static_cast<size_t>(m) * n);
        q.memcpy(got.data(), dcm, got.size() * 4).wait();
        for (size_t i = 0; i < got.size(); ++i) pf_exact &= static_cast<double>(got[i]) == wantm[i];
        kernels::swiglu(q, m, n / 2, dcm, dh0);
        kernels::w4a16_gemm_pf_swiglu(q, m, dam, k, ww, ws, dh1, n / 2, n / 2, k, shape).wait_and_throw();
        std::vector<uint16_t> h0(static_cast<size_t>(m) * n / 2), h1(h0.size());
        q.memcpy(h0.data(), dh0, h0.size() * 2);
        q.memcpy(h1.data(), dh1, h1.size() * 2).wait();
        for (int r = 0; r < m; ++r)
          for (int c = 0; c < n / 2; ++c) {
            const size_t i = static_cast<size_t>(r) * (n / 2) + c;
            if (std::fabs(got[static_cast<size_t>(r) * n + c]) >= 60.f) continue;
            ++glu_in;
            pf_glu &= h0[i] == h1[i];
          }
      }
    expect(pf_exact, "w4a16_gemm_pf zero points g32 (dequantised mode) N=640 K=1280: rows 1/7/64/200/513 x shapes auto/12-15 bit-exact vs w4zp_ref");
    const std::string glu_what = "w4a16_gemm_pf_swiglu zero points == w4a16_gemm_pf + swiglu bit for bit where |g| < 60 (" + std::to_string(glu_in) + " elements; same rows x shapes)";
    expect(pf_glu && glu_in > 0, glu_what.c_str());
    dev.release(dam);
    dev.release(dcm);
    dev.release(dh0);
    dev.release(dh1);
  }
  kernels::w4_clear_zero_points(ww);
  expect(kernels::w4_zero_points(ww) == nullptr, "w4_clear_zero_points: the matrix reads as symmetric g128 again");
  dev.release(wa);
  dev.release(ww);
  dev.release(ws);
  dev.release(wz);
  dev.release(wc);
  dev.release(wh);
  dev.release(wf);
  dev.release(tab);
}

// w4a16_gemm_rows at 1-8 live rows (exact inputs as the M = 4 gate: integer activations
// power-of-two scales -> bit-exact vs w4a16_ref, rows past the live ones untouched) and swiglu (bounded:
// 2 bf16 ulps of max(|ref|, row rms), run-to-run identity; g = 0 and u = 0 columns exact zeros).
void test_w4_rows_swiglu(xenithium::Device &dev) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  using MK = xenithium::l0::MemKind;
  auto &q = dev.compute();
  {
    const int n = 640, k = 1280, cap = 8;
    std::mt19937 rng(404);
    const kernels::W4Problem p{n, k};
    std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
    for (auto &v : qw) v = static_cast<uint32_t>(rng());
    std::vector<sycl::half> sc(p.scale_count());
    std::vector<float> scf(sc.size());
    for (size_t i = 0; i < sc.size(); ++i) {
      sc[i] = sycl::half(std::ldexp(1.f, -static_cast<int>(rng() % 4)));
      scf[i] = static_cast<float>(sc[i]);
    }
    std::vector<bf16> a(static_cast<size_t>(cap) * k);
    std::vector<float> af(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      af[i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
      a[i] = bf16(af[i]);
    }
    std::vector<uint8_t> packed(p.packed_bytes());
    kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
    std::vector<double> want(static_cast<size_t>(cap) * n);
    kernels::w4a16_ref(af.data(), qw.data(), scf.data(), cap, n, k, want.data());
    auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
    auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
    auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
    auto *wc = dev.alloc<float>(want.size(), MK::Device);
    q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
    q.memcpy(ww, packed.data(), packed.size());
    q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
    for (int rows : {1, 4, 5, 8}) {
      q.fill(wc, -12345.f, want.size()).wait();
      kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k).wait_and_throw();
      std::vector<float> got(want.size());
      q.memcpy(got.data(), wc, got.size() * 4).wait();
      const int shape = kernels::row_shape(rows);
      bool exact = true, dead = true;
      for (size_t i = 0; i < got.size(); ++i) {
        if (i < static_cast<size_t>(rows) * n) exact &= static_cast<double>(got[i]) == want[i];
        else if (i >= static_cast<size_t>(shape) * n) dead &= got[i] == -12345.f;
      }
      char what[96];
      std::snprintf(what, sizeof what, "w4a16_gemm_rows rows=%d N=%d K=%d bit-exact, rows past the shape untouched", rows, n, k);
      expect(exact && dead, what);
    }
    // 5-8 rows in one M = 8 pass == two M = 4 launches, bit for bit, on inexact
    // (random bf16) activations and real-valued scales, at every k split
    {
      std::vector<bf16> ar(a.size());
      std::vector<sycl::half> sr(sc.size());
      std::normal_distribution<float> nd(0.f, 1.f);
      for (auto &v : ar) v = bf16(nd(rng));
      for (auto &v : sr) v = sycl::half(0.01f * (1.f + std::fabs(nd(rng))));
      q.memcpy(wa, ar.data(), ar.size() * sizeof(bf16));
      q.memcpy(ws, sr.data(), sr.size() * sizeof(sycl::half)).wait();
      bool same = true;
      for (int ks : {0, 1, 2, 4, 8})
        for (int rows : {5, 6, 7, 8}) {
          std::vector<float> one(want.size()), two(want.size());
          kernels::w4a16_rows8_mode(1);
          q.fill(wc, 0.f, want.size()).wait();
          kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k, ks).wait_and_throw();
          q.memcpy(one.data(), wc, one.size() * 4).wait();
          kernels::w4a16_rows8_mode(0);
          q.fill(wc, 0.f, want.size()).wait();
          kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k, ks).wait_and_throw();
          q.memcpy(two.data(), wc, two.size() * 4).wait();
          for (size_t i = 0; i < static_cast<size_t>(rows) * n; ++i) same &= std::memcmp(&one[i], &two[i], 4) == 0;
        }
      kernels::w4a16_rows8_mode(1);
      expect(same, "w4a16_gemm_rows 5-8 rows: one M = 8 pass == two M = 4 launches bit for bit (random inputs, k splits 1-8 and auto)");
    }
    dev.release(wa);
    dev.release(ww);
    dev.release(ws);
    dev.release(wc);
  }
  {
    const int rows = 6, n = 8704;
    std::mt19937 rng(405);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> gu(static_cast<size_t>(rows) * 2 * n);
    for (auto &v : gu) v = nd(rng);
    for (int r = 0; r < rows; ++r) {
      gu[static_cast<size_t>(r) * 2 * n + 7] = 0.f;  // g = 0 -> h = 0
      gu[static_cast<size_t>(r) * 2 * n + n + 9] = 0.f;  // u = 0 -> h = 0 (or -0)
    }
    std::vector<uint16_t> ref(static_cast<size_t>(rows) * n);
    kernels::swiglu_ref(rows, n, gu.data(), ref.data());
    auto *d_gu = dev.alloc<float>(gu.size(), MK::Device);
    auto *d_h = dev.alloc<bf16>(ref.size(), MK::Device);
    q.memcpy(d_gu, gu.data(), gu.size() * 4).wait();
    std::vector<uint16_t> got(ref.size()), again(ref.size());
    kernels::swiglu(q, rows, n, d_gu, d_h).wait_and_throw();
    q.memcpy(got.data(), d_h, got.size() * 2).wait();
    kernels::swiglu(q, rows, n, d_gu, d_h).wait_and_throw();
    q.memcpy(again.data(), d_h, again.size() * 2).wait();
    auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
    int mx = 0;
    size_t off = 0, beyond = 0;
    bool zeros = true;
    for (int r = 0; r < rows; ++r) {
      double ss = 0;
      for (int i = 0; i < n; ++i) ss += std::pow(kernels::bf16_value(ref[static_cast<size_t>(r) * n + i]), 2);
      const double rms = std::sqrt(ss / n);
      for (int i = 0; i < n; ++i) {
        const size_t at = static_cast<size_t>(r) * n + i;
        const int u = std::abs(lin(got[at]) - lin(ref[at]));
        mx = std::max(mx, u);
        off += u != 0;
        const double y = kernels::bf16_value(ref[at]);
        beyond += std::fabs(kernels::bf16_value(got[at]) - y) / std::max(std::fabs(y), rms) > 1.0 / 128;
      }
      zeros &= (got[static_cast<size_t>(r) * n + 7] & 0x7fff) == 0 && (got[static_cast<size_t>(r) * n + 9] & 0x7fff) == 0;
    }
    char what[160];
    std::snprintf(what, sizeof what, "swiglu rows=%d n=%d vs host: max %d ulp, %zu / %zu off, %zu beyond 2 ulps of max(|ref|, rms); zero columns exact", rows, n, mx,
                  off, ref.size(), beyond);
    expect(beyond == 0 && off * 100 <= ref.size() && zeros, what);
    expect(got == again, "swiglu: run-to-run bit identical");
    dev.release(d_gu);
    dev.release(d_h);
  }
  // w4a16_gemm_rows_swiglu == w4a16_gemm_rows + swiglu bit for bit, every live row count,
  // k split (auto, 1, 2, 4, 8, 16), full-entropy data with |g| past the exp's whole range (scales x 0.6), n_half with a
  // short last tile row (1,040 = 65 tiles) and the model's 8,704 x 5,120; rows past the live ones untouched. Plus the
  // silu table itself: bf16(tab[g] u) == swiglu(g, u) on every bf16 g at 32 u.
  {
    for (const int nh : {1040, 8704}) {
      const int k = nh == 8704 ? 5120 : 1280, n = 2 * nh, cap = 8;
      std::mt19937 rng(2323 + nh);
      std::normal_distribution<float> nd(0.f, 1.f);
      const kernels::W4Problem p{n, k};
      std::vector<uint32_t> qw(static_cast<size_t>(k / 8) * n);
      for (auto &v : qw) v = static_cast<uint32_t>(rng());
      std::vector<sycl::half> sc(p.scale_count());
      for (auto &v : sc) v = sycl::half(nd(rng) * (nh == 1040 ? 0.6f : 0.02f));
      std::vector<bf16> a(static_cast<size_t>(cap) * k);
      for (auto &v : a) v = bf16(nd(rng));
      std::vector<uint8_t> packed(p.packed_bytes());
      kernels::pack_w4_gptq(qw.data(), n, k, packed.data());
      auto *wa = dev.alloc<bf16>(a.size(), MK::Device);
      auto *ww = dev.alloc<uint8_t>(packed.size(), MK::Device);
      auto *ws = dev.alloc<sycl::half>(sc.size(), MK::Device);
      auto *wc = dev.alloc<float>(static_cast<size_t>(cap) * n, MK::Device);
      auto *wh = dev.alloc<bf16>(static_cast<size_t>(cap) * nh, MK::Device);
      auto *wf = dev.alloc<bf16>(static_cast<size_t>(cap) * nh, MK::Device);
      auto *tab = dev.alloc<float>(kernels::kSiluTabSize, MK::Device);
      q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
      q.memcpy(ww, packed.data(), packed.size());
      q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half));
      kernels::swiglu_table(q, tab).wait_and_throw();
      bool same = true, dead = true;
      size_t big = 0;
      for (const int ks : {0, 1, 2, 4, 8, 16})
        for (int rows = 1; rows <= cap; ++rows) {
          q.fill(wh, bf16(-3.f), static_cast<size_t>(cap) * nh);
          q.fill(wf, bf16(-3.f), static_cast<size_t>(cap) * nh).wait();
          kernels::w4a16_gemm_rows(q, rows, wa, ww, ws, wc, n, k, ks);
          kernels::swiglu(q, rows, nh, wc, wh);
          kernels::w4a16_gemm_rows_swiglu(q, rows, wa, ww, ws, tab, wf, nh, k, ks).wait_and_throw();
          std::vector<uint16_t> h0(static_cast<size_t>(cap) * nh), h1(h0.size());
          std::vector<float> gu(static_cast<size_t>(rows) * n);
          q.memcpy(h0.data(), wh, h0.size() * 2);
          q.memcpy(h1.data(), wf, h1.size() * 2);
          q.memcpy(gu.data(), wc, gu.size() * 4).wait();
          for (size_t i = 0; i < h0.size(); ++i) {
            if (i < static_cast<size_t>(rows) * nh) same &= h0[i] == h1[i];
            else dead &= h1[i] == kernels::bf16_bits(-3.f);
          }
          for (int r = 0; r < rows; ++r)
            for (int c = 0; c < nh; ++c) big += std::fabs(gu[static_cast<size_t>(r) * n + c]) > 80.f;
        }
      char what[200];
      std::snprintf(what, sizeof what,
                    "w4a16_gemm_rows_swiglu n_half=%d K=%d rows 1-8, k splits auto/1/2/4/8/16 == w4a16_gemm_rows + swiglu bit for bit (%zu g past |80|), rows "
                    "past the live ones untouched",
                    nh, k, big);
      expect(same && dead && (nh != 1040 || big > 0), what);
      if (nh == 1040) {  // the table: every bf16 g x 32 u values through swiglu and through bf16(tab[g] u)
        const int G = 65536, U = 32;
        std::vector<float> gu2(static_cast<size_t>(U) * 2 * G);
        std::vector<float> uvals(U);
        for (int j = 0; j < U; ++j) uvals[static_cast<size_t>(j)] = kernels::bf16_value(kernels::bf16_bits(nd(rng) * 4.f));
        for (int r = 0; r < U; ++r)
          for (int g = 0; g < G; ++g) {
            gu2[static_cast<size_t>(r) * 2 * G + g] = kernels::bf16_value(static_cast<uint16_t>(g));
            gu2[static_cast<size_t>(r) * 2 * G + G + g] = uvals[static_cast<size_t>(r)];
          }
        auto *dg = dev.alloc<float>(gu2.size(), MK::Device);
        auto *dh = dev.alloc<bf16>(static_cast<size_t>(U) * G, MK::Device);
        q.memcpy(dg, gu2.data(), gu2.size() * 4).wait();
        kernels::swiglu(q, U, G, dg, dh).wait_and_throw();
        std::vector<uint16_t> hs(static_cast<size_t>(U) * G);
        std::vector<float> tv(G);
        q.memcpy(hs.data(), dh, hs.size() * 2);
        q.memcpy(tv.data(), tab, tv.size() * 4).wait();
        size_t diff = 0;
        {  // the product on the card (its fp32 multiply flushes subnormals, as the GEMM epilogue's does)
          auto *dt = dev.alloc<uint16_t>(static_cast<size_t>(U) * G, MK::Device);
          auto *du = dev.alloc<float>(U, MK::Device);
          q.memcpy(du, uvals.data(), U * 4).wait();
          q.parallel_for(sycl::range<1>(static_cast<size_t>(U) * G), [=](sycl::id<1> id) {
             const size_t i = id[0];
             dt[i] = sycl::bit_cast<uint16_t>(bf16(tab[i % G] * du[i / G]));
           }).wait();
          std::vector<uint16_t> tg(static_cast<size_t>(U) * G);
          q.memcpy(tg.data(), dt, tg.size() * 2).wait();
          for (int r = 0; r < U; ++r)
            for (int g = 0; g < G; ++g) {
              const uint16_t gb = static_cast<uint16_t>(g);
              if ((gb & 0x7f80) == 0x7f80 && (gb & 0x7f)) continue;  // NaN g
              const uint16_t want = hs[static_cast<size_t>(r) * G + g], got = tg[static_cast<size_t>(r) * G + g];
              const bool nan_both = (want & 0x7f80) == 0x7f80 && (want & 0x7f) && (got & 0x7f80) == 0x7f80 && (got & 0x7f);
              diff += want != got && !nan_both;
            }
          dev.release(dt);
          dev.release(du);
        }
        (void)tv;
        std::snprintf(what, sizeof what, "swiglu_table: bf16(tab[g] u) == swiglu(g, u) for every bf16 g x %d u (%zu differ)", U, diff);
        expect(diff == 0, what);
        dev.release(dg);
        dev.release(dh);
      }
      for (void *pp : {(void *)wa, (void *)ww, (void *)ws, (void *)wc, (void *)wh, (void *)wf, (void *)tab}) dev.release(pp);
    }
  }
}

// kv_copy_pages (the prefix cache's fork): pages src[i] -> dst[i] of every buffer, byte for byte;
// every other page untouched. Buffers of a K page (32 KiB) and a scale page (512 B), full-entropy bytes.
void test_kv_copy(xenithium::Device &dev) {
  const int pages = 40, n = 13;
  const size_t pb[2] = {32768, 512};
  std::mt19937 rng(22);
  std::vector<int32_t> perm(pages);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int32_t> pairs(2 * n);
  for (int i = 0; i < n; ++i) pairs[static_cast<size_t>(i)] = perm[static_cast<size_t>(i)], pairs[static_cast<size_t>(n + i)] = perm[static_cast<size_t>(n + i)];
  auto *d_pairs = dev.alloc<int32_t>(pairs.size(), xenithium::l0::MemKind::Device);
  dev.copy_queue().copy(d_pairs, pairs.data(), pairs.size() * 4, true);
  std::vector<std::vector<uint8_t>> host(2);
  std::vector<uint8_t *> d(2);
  std::vector<kernels::KvPageBuf> bufs;
  for (int b = 0; b < 2; ++b) {
    host[static_cast<size_t>(b)].resize(pages * pb[b]);
    for (auto &x : host[static_cast<size_t>(b)]) x = static_cast<uint8_t>(rng());
    d[static_cast<size_t>(b)] = dev.alloc<uint8_t>(host[static_cast<size_t>(b)].size(), xenithium::l0::MemKind::Device);
    dev.copy_queue().copy(d[static_cast<size_t>(b)], host[static_cast<size_t>(b)].data(), host[static_cast<size_t>(b)].size(), true);
    bufs.push_back({d[static_cast<size_t>(b)], pb[b]});
  }
  kernels::kv_copy_pages(dev.compute(), bufs, d_pairs, d_pairs + n, n).wait_and_throw();
  bool ok = true;
  for (int b = 0; b < 2; ++b) {
    std::vector<uint8_t> got(host[static_cast<size_t>(b)].size()), want = host[static_cast<size_t>(b)];
    for (int i = 0; i < n; ++i)
      std::memcpy(want.data() + static_cast<size_t>(pairs[static_cast<size_t>(n + i)]) * pb[b], host[static_cast<size_t>(b)].data() + static_cast<size_t>(pairs[static_cast<size_t>(i)]) * pb[b], pb[b]);
    dev.copy_queue().copy(got.data(), d[static_cast<size_t>(b)], got.size(), true);
    ok &= got == want;
    dev.release(d[static_cast<size_t>(b)]);
  }
  dev.release(d_pairs);
  expect(ok, "kv_copy_pages: 13 shuffled page pairs of a 32 KiB and a 512 B buffer copied byte for byte, the other pages untouched");
}

int run_pass() {
  try {
    test_pack_cat();
    const xenithium::l0::Loader loader = xenithium::l0::Loader::open();
    expect(loader.size() > 0, "level zero device present");
    if (loader.size() == 0) return 1;
    xenithium::Device dev = xenithium::Device::open(loader, 0);
    test_kv_copy(dev);

    expect(tile_matches<1>(dev), "dpas tile RC=1 bit-exact");
    expect(tile_matches<2>(dev), "dpas tile RC=2 bit-exact");
    expect(tile_matches<4>(dev), "dpas tile RC=4 bit-exact");
    expect(tile_matches<8>(dev), "dpas tile RC=8 bit-exact");

    // Full M=4 GEMM, small enough for the CPU reference to be cheap.
    constexpr int kN = 256, kK = 512;
    const auto a = bytes<int8_t>(kernels::kGemmM4Rows * kK, 31);
    const auto w = bytes<int8_t>(kN * kK, 32);
    std::vector<int8_t> w_vnni(static_cast<size_t>(kN) * kK);
    std::vector<int32_t> got(kernels::kGemmM4Rows * kN), want(kernels::kGemmM4Rows * kN);
    kernels::pack_weights_vnni(w.data(), kN, kK, w_vnni.data());
    kernels::gemm_int8_ref(a.data(), w.data(), kernels::kGemmM4Rows, kN, kK, want.data());

    auto *d_a = dev.alloc<int8_t>(a.size(), xenithium::l0::MemKind::Device);
    auto *d_w = dev.alloc<int8_t>(w_vnni.size(), xenithium::l0::MemKind::Device);
    auto *d_c = dev.alloc<int32_t>(got.size(), xenithium::l0::MemKind::Device);
    dev.copy_queue().copy(d_a, a.data(), a.size(), true);
    dev.copy_queue().copy(d_w, w_vnni.data(), w_vnni.size(), true);
    kernels::gemm_int8_m4(dev.compute(), d_a, d_w, d_c, kN, kK).wait_and_throw();
    dev.copy_queue().copy(got.data(), d_c, got.size() * sizeof(int32_t), true);
    expect(std::equal(got.begin(), got.end(), want.begin()), "gemm M=4 N=256 K=512 bit-exact");

    // A degenerate row must not be special-cased away.
    std::fill(w_vnni.begin(), w_vnni.end(), 0);
    dev.copy_queue().copy(d_w, w_vnni.data(), w_vnni.size(), true);
    kernels::gemm_int8_m4(dev.compute(), d_a, d_w, d_c, kN, kK).wait_and_throw();
    dev.copy_queue().copy(got.data(), d_c, got.size() * sizeof(int32_t), true);
    expect(std::all_of(got.begin(), got.end(), [](int32_t v) { return v == 0; }),
           "zero weights give zero output");

    dev.release(d_a);
    dev.release(d_w);
    dev.release(d_c);

    // Split-K: every ksplit on decode shapes, including uneven parts (K=640 is 20
    // chunks, K=96 is 3) and KS=2 (fewer work-items than output rows). The output
    // is poisoned before each launch so a skipped row cannot pass on stale data.
    struct Shape { int n, k; };
    for (const Shape s : {Shape{640, 2560}, Shape{2560, 640}, Shape{48, 96}, Shape{160, 4096}}) {
      const auto sa = bytes<int8_t>(kernels::kGemmM4Rows * static_cast<size_t>(s.k), 41 + s.n);
      const auto sw = bytes<int8_t>(static_cast<size_t>(s.n) * s.k, 43 + s.k);
      std::vector<int8_t> sv(sw.size());
      std::vector<int32_t> sgot(kernels::kGemmM4Rows * static_cast<size_t>(s.n)), swant(sgot.size());
      kernels::pack_weights_vnni(sw.data(), s.n, s.k, sv.data());
      kernels::gemm_int8_ref(sa.data(), sw.data(), kernels::kGemmM4Rows, s.n, s.k, swant.data());
      auto *ea = dev.alloc<int8_t>(sa.size(), xenithium::l0::MemKind::Device);
      auto *ew = dev.alloc<int8_t>(sv.size(), xenithium::l0::MemKind::Device);
      auto *ec = dev.alloc<int32_t>(sgot.size(), xenithium::l0::MemKind::Device);
      dev.copy_queue().copy(ea, sa.data(), sa.size(), true);
      dev.copy_queue().copy(ew, sv.data(), sv.size(), true);
      for (int ks : {0, 1, 2, 4, 8, 16}) {
        dev.copy_queue().fill(ec, 0xA5, sgot.size() * sizeof(int32_t), true);
        kernels::gemm_int8_m4(dev.compute(), ea, ew, ec, s.n, s.k, ks).wait_and_throw();
        dev.copy_queue().copy(sgot.data(), ec, sgot.size() * sizeof(int32_t), true);
        char what[96];
        std::snprintf(what, sizeof what, "gemm M=4 N=%d K=%d ksplit=%d%s bit-exact", s.n, s.k,
                      ks ? ks : kernels::gemm_int8_auto_ksplit(s.n, s.k), ks ? "" : " (auto)");
        expect(sgot == swant, what);
      }
      dev.release(ea);
      dev.release(ew);
      dev.release(ec);
    }

    // W4A16: integer activations in [-4, 4] and power-of-two scales keep every fp32
    // intermediate exact, so the device must equal the reference bit for bit at
    // every ksplit; full-entropy nibbles (0..15) so q - 8 covers [-8, 7]. Shapes:
    // fused gate+up (1280 x 2560), down (2560 x 640, 5 groups: uneven parts), and a
    // single-tile / single-group edge.
    for (const Shape s : {Shape{1280, 2560}, Shape{2560, 640}, Shape{16, 128}}) {
      using bf16 = sycl::ext::oneapi::bfloat16;
      constexpr int m = kernels::kGemmM4Rows;
      const kernels::W4Problem p{s.n, s.k};
      std::mt19937 rng(77 + s.n);
      std::vector<uint32_t> qw(static_cast<size_t>(s.k / 8) * s.n);
      for (auto &v : qw) v = static_cast<uint32_t>(rng());
      std::vector<sycl::half> sc(p.scale_count());
      std::vector<float> scf(sc.size());
      for (size_t i = 0; i < sc.size(); ++i) {
        sc[i] = sycl::half(std::ldexp(1.f, -static_cast<int>(rng() % 4)));
        scf[i] = static_cast<float>(sc[i]);
      }
      std::vector<bf16> a(static_cast<size_t>(m) * s.k);
      std::vector<float> af(a.size());
      for (size_t i = 0; i < a.size(); ++i) {
        af[i] = static_cast<float>(static_cast<int>(rng() % 9) - 4);
        a[i] = bf16(af[i]);
      }
      std::vector<uint8_t> packed(p.packed_bytes());
      kernels::pack_w4_gptq(qw.data(), s.n, s.k, packed.data());
      std::vector<double> want(static_cast<size_t>(m) * s.n);
      kernels::w4a16_ref(af.data(), qw.data(), scf.data(), m, s.n, s.k, want.data());

      auto &q = dev.compute();
      auto *wa = dev.alloc<bf16>(a.size(), xenithium::l0::MemKind::Device);
      auto *ww = dev.alloc<uint8_t>(packed.size(), xenithium::l0::MemKind::Device);
      auto *ws = dev.alloc<sycl::half>(sc.size(), xenithium::l0::MemKind::Device);
      auto *wc = dev.alloc<float>(want.size(), xenithium::l0::MemKind::Device);
      q.memcpy(wa, a.data(), a.size() * sizeof(bf16));
      q.memcpy(ww, packed.data(), packed.size());
      q.memcpy(ws, sc.data(), sc.size() * sizeof(sycl::half)).wait();
      std::vector<float> got(want.size());
      for (int ks : {0, 1, 2, 4, 8, 16}) {
        q.fill(wc, -12345.f, got.size()).wait();
        kernels::w4a16_gemm_m4(q, wa, ww, ws, wc, s.n, s.k, ks).wait_and_throw();
        q.memcpy(got.data(), wc, got.size() * sizeof(float)).wait();
        bool exact = true;
        for (size_t i = 0; i < got.size() && exact; ++i) exact = static_cast<double>(got[i]) == want[i];
        char what[96];
        std::snprintf(what, sizeof what, "w4a16 M=4 N=%d K=%d ksplit=%d%s bit-exact", s.n, s.k,
                      ks ? ks : kernels::w4a16_auto_ksplit(s.n, s.k), ks ? " (requested)" : " (auto)");
        expect(exact, what);
      }
      dev.release(wa);
      dev.release(ww);
      dev.release(ws);
      dev.release(wc);
    }

    test_dense(dev);
    test_dense_wide(dev);
    test_dense_i2(dev);
    test_head_two_stage(dev);
    test_w4_rows_swiglu(dev);
    test_w4zp(dev);
    test_w4aff(dev);
    test_w4aff_pf(dev);
    test_w4_pf(dev);
    test_w4a8(dev);
    test_w4a8_norm_codes(dev);
    test_pf_dense(dev);
    test_pf_dense_pair(dev);
    test_pf_dense_a8(dev);
    test_pf_dense_a8_row(dev);

    if (loader.size() > 1) {
      // Same int8 GEMM on every other card: catches a card-specific fault early.
      for (size_t d = 1; d < loader.size(); ++d) {
        xenithium::Device dn = xenithium::Device::open(loader, d);
        char what[64];
        std::snprintf(what, sizeof what, "dpas tile RC=4 bit-exact on device %zu", d);
        expect(tile_matches<4>(dn), what);
      }
    }
  } catch (const std::exception &e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}

int main() {
  const int rc = run_pass();
  std::printf("%s: %d failure(s)\n", rc ? "FAILED" : "OK", g_failures);
  return rc;
}
