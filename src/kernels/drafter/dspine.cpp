// Xenithium · the DSpine drafter's own kernels (see dspine.hpp for the semantics).
#include "xenithium/kernels/drafter/dspine.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/util/gumbel.hpp"

namespace xenithium::kernels {
namespace {

using bf16 = ds_bf16;

inline float bfr(float v) { return static_cast<float>(bf16(v)); }
inline float hbf(float v) { return bf16_value(bf16_bits(v)); }  // host: round to bf16

class DsBridgeInK;
class DsBridgeInFastK;
class DsCodesK;
class DsMessageK;
class DsBridgeOutK;
class DsSelectK;
class DsTreeK;
class DsTreeScoreK;
class DsScaleRowsK;

constexpr int kRowItems = 256;
constexpr float kSqrtCode = 32.0f;  // sqrt(kDsCode), exact

inline float sigmoidf(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float sigmoid_host(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// candidate key -> (bf16 logit as float, candidate index)
inline float key_logit(uint64_t kl) {
  const uint32_t o = static_cast<uint32_t>(kl >> 32);
  const uint16_t bb = (o & 0x8000) ? static_cast<uint16_t>(o & 0x7fff) : static_cast<uint16_t>(~o & 0xffff);
  return sycl::bit_cast<float>(static_cast<uint32_t>(bb) << 16);
}
inline int32_t key_index(uint64_t kl) { return static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(kl)); }

// ds_tree's candidate-list table, one per device (kMaxSegs segments x the most lists a 16-row block needs), allocated on
// the first call and kept for the process
struct DsTreeTab {
  int32_t *cand = nullptr;
  float *lp = nullptr;
};
DsTreeTab ds_tree_tab(sycl::queue &q) {
  static std::mutex mu;
  static std::unordered_map<sycl::device, DsTreeTab> tabs;
  std::lock_guard<std::mutex> lk(mu);
  DsTreeTab &t = tabs[q.get_device()];
  if (!t.cand) {
    const size_t n = static_cast<size_t>(kMaxSegs) * (1 + (kTreeRows - 2) * kDfTopK) * kDfTopK;
    t.cand = sycl::malloc_device<int32_t>(n, q);
    t.lp = sycl::malloc_device<float>(n, q);
    if (!t.cand || !t.lp) throw std::runtime_error("ds_tree: table allocation");
  }
  return t;
}

}  // namespace

// ---- the bridge -----------------------------------------------------------------------------------
sycl::event ds_bridge_in(sycl::queue &q, int rows, int hidden, bf16 *res, const bf16 *add, float eps, float *rms, bf16 *xr) {
  if (rows < 1 || hidden < 1) throw std::invalid_argument("ds_bridge_in: shape");
  const int H = hidden;
  static const bool fast = [] { const char *e = std::getenv("XEN_DS_BRIDGE_FAST"); return !(e && e[0] == '0'); }();
  constexpr int kF = 1024, kV = 8;
  if (fast && H <= kF * kV) {
    return q.parallel_for<DsBridgeInFastK>(sycl::nd_range<1>(static_cast<size_t>(rows) * kF, kF), [=](sycl::nd_item<1> it) {
      const int r = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      bf16 *x = res + static_cast<size_t>(r) * H;
      float v[kV];
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < kV; ++j) {
        const int h = lid + j * kF;
        v[j] = 0.f;
        if (h < H) {
          float t = static_cast<float>(x[h]);
          if (add) {
            t = bfr(t + static_cast<float>(add[static_cast<size_t>(r) * H + h]));
            x[h] = bf16(t);
          }
          v[j] = t;
          acc += t * t;
        }
      }
      acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
      const float s = sycl::sqrt(sycl::fmax(acc / static_cast<float>(H), FLT_MIN));
      if (lid == 0) rms[r] = s;
      const float den = s + eps;
#pragma unroll
      for (int j = 0; j < kV; ++j) {
        const int h = lid + j * kF;
        if (h < H) xr[static_cast<size_t>(r) * H + h] = bf16(v[j] / den);
      }
    });
  }
  return q.parallel_for<DsBridgeInK>(sycl::nd_range<1>(static_cast<size_t>(rows) * kRowItems, kRowItems), [=](sycl::nd_item<1> it) {
    const int r = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
    bf16 *x = res + static_cast<size_t>(r) * H;
    float acc = 0.f;
    for (int h = lid; h < H; h += kRowItems) {
      float v = static_cast<float>(x[h]);
      if (add) {
        v = bfr(v + static_cast<float>(add[static_cast<size_t>(r) * H + h]));
        x[h] = bf16(v);
      }
      acc += v * v;
    }
    acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
    const float s = sycl::sqrt(sycl::fmax(acc / static_cast<float>(H), FLT_MIN));
    if (lid == 0) rms[r] = s;
    const float den = s + eps;
    for (int h = lid; h < H; h += kRowItems) xr[static_cast<size_t>(r) * H + h] = bf16(static_cast<float>(x[h]) / den);
  });
}

void ds_bridge_in_ref(int rows, int hidden, uint16_t *res, const uint16_t *add, float eps, float *rms, uint16_t *xr) {
  for (int r = 0; r < rows; ++r) {
    uint16_t *x = res + static_cast<size_t>(r) * hidden;
    double acc = 0;
    for (int h = 0; h < hidden; ++h) {
      if (add) x[h] = bf16_bits(bf16_value(x[h]) + bf16_value(add[static_cast<size_t>(r) * hidden + h]));
      const double v = bf16_value(x[h]);
      acc += v * v;
    }
    const float s = std::sqrt(std::max(static_cast<float>(acc / hidden), FLT_MIN));
    rms[r] = s;
    for (int h = 0; h < hidden; ++h) xr[static_cast<size_t>(r) * hidden + h] = bf16_bits(bf16_value(x[h]) / (s + eps));
  }
}

sycl::event ds_codes(sycl::queue &q, int rows, int seg_rows, const float *p, const int32_t *anchors, const bf16 *codebook, float eps, bf16 *code) {
  if (rows < 1 || seg_rows < 1 || rows % seg_rows) throw std::invalid_argument("ds_codes: shape");
  constexpr int C = kDsCode, per = C / kRowItems;
  return q.parallel_for<DsCodesK>(sycl::nd_range<1>(static_cast<size_t>(rows) * kRowItems, kRowItems), [=](sycl::nd_item<1> it) {
    const int r = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
    bf16 *o = code + static_cast<size_t>(r) * C;
    if (r % seg_rows == 0) {  // the anchor: the codebook's row (uniform per work-group)
      const bf16 *cb = codebook + static_cast<size_t>(anchors[r / seg_rows]) * C;
      for (int i = 0; i < per; ++i) {
        const int c = lid * per + i;
        o[c] = bf16(kSqrtCode * static_cast<float>(cb[c]));
      }
      return;
    }
    const float *pr = p + static_cast<size_t>(r) * C;
    float v[per];
    float acc = 0.f;
    for (int i = 0; i < per; ++i) {
      v[i] = bfr(pr[lid * per + i]);
      acc += v[i] * v[i];
    }
    acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
    const float den = sycl::sqrt(acc) + eps;
    for (int i = 0; i < per; ++i) o[lid * per + i] = bf16(kSqrtCode * v[i] / den);
  });
}

void ds_codes_ref(int rows, int seg_rows, const float *p, const int32_t *anchors, const uint16_t *codebook, float eps, uint16_t *code) {
  constexpr int C = kDsCode;
  for (int r = 0; r < rows; ++r) {
    uint16_t *o = code + static_cast<size_t>(r) * C;
    if (r % seg_rows == 0) {
      const uint16_t *cb = codebook + static_cast<size_t>(anchors[r / seg_rows]) * C;
      for (int c = 0; c < C; ++c) o[c] = bf16_bits(kSqrtCode * bf16_value(cb[c]));
      continue;
    }
    double acc = 0;
    for (int c = 0; c < C; ++c) {
      const double v = hbf(p[static_cast<size_t>(r) * C + c]);
      acc += v * v;
    }
    const float den = static_cast<float>(std::sqrt(acc)) + eps;
    for (int c = 0; c < C; ++c) o[c] = bf16_bits(kSqrtCode * hbf(p[static_cast<size_t>(r) * C + c]) / den);
  }
}

sycl::event ds_message(sycl::queue &q, int rows, int seg_rows, const float *packed, const bf16 *bias, bf16 *msg) {
  if (rows < 1 || seg_rows < 1 || rows % seg_rows) throw std::invalid_argument("ds_message: shape");
  constexpr int R = kDsRank;
  return q.parallel_for<DsMessageK>(sycl::range<1>(static_cast<size_t>(rows) * R), [=](sycl::id<1> id) {
    const int r = static_cast<int>(id[0]) / R, i = static_cast<int>(id[0]) % R;
    float m = 0.f;
    if (r % seg_rows) {
      const float *cur = packed + static_cast<size_t>(r) * 3 * R, *prv = cur - 3 * R;
      const float g = sigmoidf(bfr(cur[i]) + bfr(prv[R + i]) + static_cast<float>(bias[i]));
      m = g * bfr(prv[2 * R + i]);
    }
    msg[static_cast<size_t>(r) * R + i] = bf16(m);
  });
}

void ds_message_ref(int rows, int seg_rows, const float *packed, const uint16_t *bias, uint16_t *msg) {
  constexpr int R = kDsRank;
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < R; ++i) {
      float m = 0.f;
      if (r % seg_rows) {
        const float *cur = packed + static_cast<size_t>(r) * 3 * R, *prv = cur - 3 * R;
        const float g = sigmoid_host(hbf(cur[i]) + hbf(prv[R + i]) + bf16_value(bias[i]));
        m = g * hbf(prv[2 * R + i]);
      }
      msg[static_cast<size_t>(r) * R + i] = bf16_bits(m);
    }
}

sycl::event ds_bridge_out(sycl::queue &q, int rows, int seg_rows, int hidden, bf16 *res, const float *rms, const float *delta) {
  if (rows < 1 || seg_rows < 1 || rows % seg_rows) throw std::invalid_argument("ds_bridge_out: shape");
  const int H = hidden;
  return q.parallel_for<DsBridgeOutK>(sycl::range<1>(static_cast<size_t>(rows) * H), [=](sycl::id<1> id) {
    const size_t e = id[0];
    const int r = static_cast<int>(e / static_cast<size_t>(H));
    if (r % seg_rows == 0) return;  // the anchor keeps raw
    res[e] = bf16(static_cast<float>(res[e]) + rms[r] * bfr(delta[e]));
  });
}

void ds_bridge_out_ref(int rows, int seg_rows, int hidden, uint16_t *res, const float *rms, const float *delta) {
  for (int r = 0; r < rows; ++r) {
    if (r % seg_rows == 0) continue;
    for (int h = 0; h < hidden; ++h) {
      const size_t e = static_cast<size_t>(r) * hidden + h;
      res[e] = bf16_bits(bf16_value(res[e]) + rms[r] * hbf(delta[e]));
    }
  }
}

// ---- the drafts -----------------------------------------------------------------------------------
sycl::event ds_select(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *packed, const bf16 *tsv, const bf16 *tg, const bf16 *bias,
                      const DsTail &tail, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise) {
  constexpr int kParts = 16, kItems = kDfTopK * kParts, R = kDsRank, per = R / kParts;
  if (segs < 1 || seg_rows < 2 || segs * seg_rows > kMaxRows || !tail.out || !tail.rms || !tail.msg || tail.hidden < 1) throw std::invalid_argument("ds_select: shape");
  const bf16 *outp = tail.out, *msgp = tail.msg;
  const float *rmsp = tail.rms, feps = tail.eps;
  const int H = tail.hidden;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(kItems), h);
    sycl::local_accessor<float, 1> score(sycl::range<1>(kDfTopK), h);
    sycl::local_accessor<int32_t, 1> pred(sycl::range<1>(1), h);  // the last pick's candidate index (-1: the anchor)
    h.parallel_for<DsSelectK>(sycl::nd_range<1>(static_cast<size_t>(segs) * kItems, kItems), [=](sycl::nd_item<1> it) {
      const int s = static_cast<int>(it.get_group(0)), lid = static_cast<int>(it.get_local_id(0));
      const int b = lid / kParts, p = lid % kParts;
      const DraftNoiseRec nz = noise ? noise[s] : DraftNoiseRec{};
      const float *row0 = packed + static_cast<size_t>(s) * seg_rows * 3 * R;
      if (lid == 0) pred[0] = -1;
      sycl::group_barrier(it.get_group());
      for (int j = 1; j < seg_rows; ++j) {
        const int row = s * seg_rows + j;
        // the logits' scale: rms_out / rms (the final norm's rms of out over the bridge's pre-write rms)
        float ss = 0.f;
        for (int e = lid; e < H; e += kItems) {
          const float x = static_cast<float>(outp[static_cast<size_t>(row) * H + e]);
          ss += x * x;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float ratio = sycl::sqrt(ss / static_cast<float>(H) + feps) / rmsp[row];
        const uint64_t key = keys[static_cast<size_t>(row) * kDfTopK + b];
        const int32_t v = key_index(key);
        const int32_t pv = pred[0];
        const float *qr = packed + static_cast<size_t>(row) * 3 * R;
        const bf16 *T = tg + static_cast<size_t>(v) * R, *M = msgp + static_cast<size_t>(row) * R;
        float acc = 0.f;
        if (pv < 0) {
          for (int i = p * per; i < (p + 1) * per; ++i)
            acc += (sigmoidf(bfr(qr[i]) + bfr(row0[R + i]) + static_cast<float>(bias[i])) * bfr(row0[2 * R + i]) - static_cast<float>(M[i])) * static_cast<float>(T[i]);
        } else {
          const bf16 *S = tsv + static_cast<size_t>(pv) * 2 * R;
          for (int i = p * per; i < (p + 1) * per; ++i)
            acc += (sigmoidf(bfr(qr[i]) + static_cast<float>(S[i]) + static_cast<float>(bias[i])) * static_cast<float>(S[R + i]) - static_cast<float>(M[i])) *
                   static_cast<float>(T[i]);
        }
        part[lid] = acc;
        sycl::group_barrier(it.get_group());
        if (lid < kDfTopK) {
          float t = 0.f;
          for (int i = 0; i < kParts; ++i) t += part[lid * kParts + i];
          const uint64_t kl = keys[static_cast<size_t>(row) * kDfTopK + lid];
          float sc = ratio * key_logit(kl) + t;
          if (nz.inv_t > 0.0f) {
            const int32_t vl = key_index(kl);
            const int32_t tl = sub_k > 0 && vl >= sub_k ? vl - sub_k + sub_s0 : vl;
            sc = sc * nz.inv_t + util::gumbel(nz.key, nz.pos + static_cast<uint64_t>(j - 1), tl);
          }
          score[lid] = sc;
        }
        sycl::group_barrier(it.get_group());
        if (lid == 0) {
          int best = 0;
          for (int i = 1; i < kDfTopK; ++i)
            if (score[i] > score[best]) best = i;
          const int32_t vb = key_index(keys[static_cast<size_t>(row) * kDfTopK + best]);
          pred[0] = vb;
          drafts[s * (seg_rows - 1) + j - 1] = sub_k > 0 && vb >= sub_k ? vb - sub_k + sub_s0 : vb;
        }
        sycl::group_barrier(it.get_group());
      }
    });
  });
}

void ds_select_ref(int seg_rows, const uint64_t *keys, const float *packed, const uint16_t *tsv, const uint16_t *tg, const uint16_t *bias, const uint16_t *out,
                   int hidden, float eps, const float *rms, const uint16_t *msg, int sub_k, int sub_s0, int32_t *drafts, const DraftNoiseRec *noise) {
  constexpr int R = kDsRank;
  auto tok_of = [&](int32_t v) { return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v; };
  int32_t pv = -1;
  for (int j = 1; j < seg_rows; ++j) {
    const uint64_t *kr = keys + static_cast<size_t>(j) * kDfTopK;
    const float *qr = packed + static_cast<size_t>(j) * 3 * R;
    double ss = 0;
    for (int e = 0; e < hidden; ++e) ss += std::pow(static_cast<double>(bf16_value(out[static_cast<size_t>(j) * hidden + e])), 2);
    const double ratio = std::sqrt(ss / hidden + eps) / rms[j];
    double best = -1e300;
    int32_t bv = -1;
    for (int b = 0; b < kDfTopK; ++b) {
      const int32_t v = static_cast<int32_t>(df_key_index(kr[b]));
      double a = 0;
      for (int i = 0; i < R; ++i) {
        const float g = pv < 0 ? hbf(packed[R + i]) : bf16_value(tsv[static_cast<size_t>(pv) * 2 * R + i]);
        const float vv = pv < 0 ? hbf(packed[2 * R + i]) : bf16_value(tsv[static_cast<size_t>(pv) * 2 * R + R + i]);
        a += (static_cast<double>(sigmoid_host(hbf(qr[i]) + g + bf16_value(bias[i]))) * vv - bf16_value(msg[static_cast<size_t>(j) * R + i])) *
             bf16_value(tg[static_cast<size_t>(v) * R + i]);
      }
      double sc = ratio * df_key_value(kr[b]) + a;
      if (noise && noise->inv_t > 0.0f)
        sc = static_cast<float>(static_cast<float>(sc) * noise->inv_t) + util::gumbel(noise->key, noise->pos + static_cast<uint64_t>(j - 1), tok_of(v));
      if (sc > best) best = sc, bv = v;
    }
    pv = bv;
    drafts[j - 1] = tok_of(bv);
  }
}

// ---- the best-first draft tree with DSpine's scores ----
// Two launches. DsTreeScoreK: every candidate list a node can need, one work-group each - block row 1 given the anchor
// (slot 0) and block row j >= 2 given each of row j - 1's 16 candidates (slots 1 + (j - 2) x 16 + b) - each scored,
// log-softmaxed and ranked by the same expressions as the serial walk's node expansion (the same work-group shape, so
// the same reductions and bits). DsTreePickK: one item per segment walks the best-first frontier over that table.
sycl::event ds_tree(sycl::queue &q, int segs, int seg_rows, const uint64_t *keys, const float *packed, const bf16 *tsv, const bf16 *tg, const bf16 *bias,
                    const DsTail &tail, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp) {
  constexpr int kParts = 16, kItems = kDfTopK * kParts, R = kDsRank, per = R / kParts, kN = kTreeRows;
  if (segs < 1 || segs > kMaxSegs || seg_rows < 2 || seg_rows > kN || segs * seg_rows > kMaxRows || !tail.out || !tail.rms || !tail.msg || tail.hidden < 1 ||
      budget < 1 || budget >= kN)
    throw std::invalid_argument("ds_tree: shape");
  const bf16 *outp = tail.out, *msgp = tail.msg;
  const float *rmsp = tail.rms, feps = tail.eps;
  const int H = tail.hidden, D = seg_rows - 1;
  const int slots = 1 + (D - 1) * kDfTopK;  // the candidate lists a segment's nodes can need
  const float lcut = cut > 0.f ? std::log(cut) : -std::numeric_limits<float>::infinity();
  const float tscale = inv_temp;
  DsTreeTab tab = ds_tree_tab(q);  // [segs][slots][16]: (the candidate's key slot, its log-probability), best first
  int32_t *tc = tab.cand;
  float *tl = tab.lp;
  const sycl::event sc = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(kItems), h);
    sycl::local_accessor<float, 1> score(sycl::range<1>(kDfTopK), h);
    h.parallel_for<DsTreeScoreK>(sycl::nd_range<1>(static_cast<size_t>(segs) * slots * kItems, kItems), [=](sycl::nd_item<1> it) {
      const int g = static_cast<int>(it.get_group(0)), s = g / slots, u = g % slots, lid = static_cast<int>(it.get_local_id(0));
      const int b = lid / kParts, p = lid % kParts;
      const float *row0 = packed + static_cast<size_t>(s) * seg_rows * 3 * R;
      const int j = u == 0 ? 1 : 2 + (u - 1) / kDfTopK;  // the block row whose candidates this list ranks
      const int row = s * seg_rows + j;
      const int32_t pv = u == 0 ? -1 : key_index(keys[static_cast<size_t>(row - 1) * kDfTopK + (u - 1) % kDfTopK]);
      float ss = 0.f;
      for (int e = lid; e < H; e += kItems) {
        const float x = static_cast<float>(outp[static_cast<size_t>(row) * H + e]);
        ss += x * x;
      }
      ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
      const float ratio = sycl::sqrt(ss / static_cast<float>(H) + feps) / rmsp[row];
      const int32_t v = key_index(keys[static_cast<size_t>(row) * kDfTopK + b]);
      const float *qr = packed + static_cast<size_t>(row) * 3 * R;
      const bf16 *T = tg + static_cast<size_t>(v) * R, *M = msgp + static_cast<size_t>(row) * R;
      float acc = 0.f;
      if (pv < 0) {
        for (int i = p * per; i < (p + 1) * per; ++i)
          acc += (sigmoidf(bfr(qr[i]) + bfr(row0[R + i]) + static_cast<float>(bias[i])) * bfr(row0[2 * R + i]) - static_cast<float>(M[i])) * static_cast<float>(T[i]);
      } else {
        const bf16 *S = tsv + static_cast<size_t>(pv) * 2 * R;
        for (int i = p * per; i < (p + 1) * per; ++i)
          acc += (sigmoidf(bfr(qr[i]) + static_cast<float>(S[i]) + static_cast<float>(bias[i])) * static_cast<float>(S[R + i]) - static_cast<float>(M[i])) *
                 static_cast<float>(T[i]);
      }
      part[lid] = acc;
      sycl::group_barrier(it.get_group());
      if (lid < kDfTopK) {
        float t = 0.f;
        for (int i = 0; i < kParts; ++i) t += part[lid * kParts + i];
        score[lid] = (ratio * key_logit(keys[static_cast<size_t>(row) * kDfTopK + lid]) + t) * tscale;  // the tree's temperature
      }
      sycl::group_barrier(it.get_group());
      if (lid < kDfTopK) {
        float mx = score[0];
        for (int i = 1; i < kDfTopK; ++i) mx = sycl::fmax(mx, score[i]);
        float se = 0.f;
        for (int i = 0; i < kDfTopK; ++i) se += sycl::exp(score[i] - mx);
        const float me = score[lid];
        int rk = 0;
        for (int i = 0; i < kDfTopK; ++i) rk += score[i] > me || (score[i] == me && i < lid);
        const size_t o = (static_cast<size_t>(s) * slots + u) * kDfTopK + rk;
        tc[o] = lid;
        tl[o] = me - mx - sycl::log(se);
      }
    });
  });
  return q.submit([&](sycl::handler &h) {
    h.depends_on(sc);
    h.parallel_for<DsTreeK>(sycl::range<1>(static_cast<size_t>(segs)), [=](sycl::id<1> sid) {
      const int s = static_cast<int>(sid[0]);
      auto tok_of = [&](int32_t v) { return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v; };
      int32_t nslot[kN], npar[kN], ndep[kN], nnext[kN], ncand[kN];
      float nw[kN];
      nslot[0] = 0, npar[0] = -1, ndep[0] = 0, nnext[0] = 0, ncand[0] = -1, nw[0] = 0.f;
      int n = 1;
      // node i's list: the root's slot 0; a node at depth d >= 1 (key slot b of block row d) slot 1 + (d - 1) x 16 + b
      auto list_of = [&](int i) { return (static_cast<size_t>(s) * slots + (ndep[i] == 0 ? 0 : 1 + (ndep[i] - 1) * kDfTopK + nslot[i])) * kDfTopK; };
      while (n - 1 < budget) {  // the frontier's best joins the tree
        int bi = -1;
        float bw = 0.f;
        for (int i = 0; i < n; ++i)
          if (ndep[i] < D && nnext[i] < kDfTopK) {
            const float w = nw[i] + tl[list_of(i) + nnext[i]];
            if (bi < 0 || w > bw) bi = i, bw = w;
          }
        if (bi < 0 || bw < lcut) break;
        const int b = tc[list_of(bi) + nnext[bi]];
        const int d = ndep[bi] + 1;
        nslot[n] = b, npar[n] = bi, ndep[n] = d, nnext[n] = 0, nw[n] = bw;
        ncand[n] = key_index(keys[static_cast<size_t>(s * seg_rows + d) * kDfTopK + b]);
        ++nnext[bi];
        ++n;
      }
      int32_t *o = tree + static_cast<size_t>(s) * kDfTreeInts;
      o[0] = n - 1;
      for (int i = 0; i < kN; ++i) {
        o[1 + i] = i < n - 1 ? tok_of(ncand[i + 1]) : 0;
        o[1 + kN + i] = i < n - 1 ? npar[i + 1] : 0;
      }
    });
  });
}

void ds_tree_ref(int seg_rows, const uint64_t *keys, const float *packed, const uint16_t *tsv, const uint16_t *tg, const uint16_t *bias, const uint16_t *out, int hidden,
                 float eps, const float *rms, const uint16_t *msg, int sub_k, int sub_s0, int budget, float cut, int32_t *tree, float inv_temp) {
  constexpr int R = kDsRank, kN = kTreeRows;
  const int D = seg_rows - 1;
  auto tok_of = [&](int32_t v) { return sub_k > 0 && v >= sub_k ? v - sub_k + sub_s0 : v; };
  int32_t ncand[kN] = {-1}, npar[kN] = {-1}, ndep[kN] = {0}, nnext[kN] = {0};
  float nw[kN] = {0.f}, clp[kN][kDfTopK] = {};
  int32_t ccand[kN][kDfTopK] = {};
  auto cands = [&](int node) {  // ds_select_ref's score given the node's pick
    const int j = ndep[node] + 1;
    const uint64_t *kr = keys + static_cast<size_t>(j) * kDfTopK;
    const float *qr = packed + static_cast<size_t>(j) * 3 * R;
    double ss = 0;
    for (int e = 0; e < hidden; ++e) ss += std::pow(static_cast<double>(bf16_value(out[static_cast<size_t>(j) * hidden + e])), 2);
    const double ratio = std::sqrt(ss / hidden + eps) / rms[j];
    const int32_t pv = ncand[node];
    float sc[kDfTopK];
    for (int b = 0; b < kDfTopK; ++b) {
      const int32_t v = static_cast<int32_t>(df_key_index(kr[b]));
      double a = 0;
      for (int i = 0; i < R; ++i) {
        const float g = pv < 0 ? hbf(packed[R + i]) : bf16_value(tsv[static_cast<size_t>(pv) * 2 * R + i]);
        const float vv = pv < 0 ? hbf(packed[2 * R + i]) : bf16_value(tsv[static_cast<size_t>(pv) * 2 * R + R + i]);
        a += (static_cast<double>(sigmoid_host(hbf(qr[i]) + g + bf16_value(bias[i]))) * vv - bf16_value(msg[static_cast<size_t>(j) * R + i])) *
             bf16_value(tg[static_cast<size_t>(v) * R + i]);
      }
      sc[b] = static_cast<float>(ratio * df_key_value(kr[b]) + a) * inv_temp;
    }
    float mx = sc[0];
    for (int i = 1; i < kDfTopK; ++i) mx = std::fmax(mx, sc[i]);
    float se = 0.f;
    for (int i = 0; i < kDfTopK; ++i) se += std::exp(sc[i] - mx);
    for (int b = 0; b < kDfTopK; ++b) {
      int rk = 0;
      for (int i = 0; i < kDfTopK; ++i) rk += sc[i] > sc[b] || (sc[i] == sc[b] && i < b);
      ccand[node][rk] = static_cast<int32_t>(df_key_index(kr[b]));
      clp[node][rk] = sc[b] - mx - std::log(se);
    }
  };
  const float lcut = cut > 0.f ? std::log(cut) : -std::numeric_limits<float>::infinity();
  int n = 1;
  if (D >= 1) cands(0);
  while (n - 1 < budget) {
    int bi = -1;
    float bw = 0.f;
    for (int i = 0; i < n; ++i)
      if (ndep[i] < D && nnext[i] < kDfTopK) {
        const float w = nw[i] + clp[i][nnext[i]];
        if (bi < 0 || w > bw) bi = i, bw = w;
      }
    if (bi < 0 || bw < lcut) break;
    ncand[n] = ccand[bi][nnext[bi]], npar[n] = bi, ndep[n] = ndep[bi] + 1, nnext[n] = 0, nw[n] = bw;
    ++nnext[bi];
    if (ndep[n] < D) cands(n);
    ++n;
  }
  std::fill(tree, tree + kDfTreeInts, 0);
  tree[0] = n - 1;
  for (int i = 1; i < n; ++i) tree[i] = tok_of(ncand[i]), tree[kN + i] = npar[i];
}

sycl::event ds_scale_rows(sycl::queue &q, int rows, int n, const bf16 *src, const bf16 *w, float scale, bf16 *dst) {
  if (rows < 1 || n < 1) throw std::invalid_argument("ds_scale_rows: shape");
  return q.parallel_for<DsScaleRowsK>(sycl::range<1>(static_cast<size_t>(rows) * n), [=](sycl::id<1> id) {
    const size_t e = id[0];
    const float f = w ? static_cast<float>(w[e % static_cast<size_t>(n)]) : scale;
    dst[e] = bf16(static_cast<float>(src[e]) * f);
  });
}

}  // namespace xenithium::kernels
