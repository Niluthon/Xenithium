// Xenithium · Gated DeltaNet decode block.
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "linear_attention/gdn_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/rows.hpp"

#include <cstdlib>

namespace xenithium::kernels {
namespace {

using namespace gdn_detail;

// The core for this call: 4-row segments here (128 GRF), 8-row segments or an 8-row log in the 256-GRF library.
template <bool SWISH>
sycl::event core_launch(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                        bf16 *core_out) {
  const int sr = st.seg ? seg_size(rows, st.seg_rows) : rows;
  // the 8-row cores from the 128-GRF library (gdn_core_v2.cpp): two work-groups per Xe core, so a card's value
  // heads run in one wave; same code and bits as the 256-GRF instances. XEN_GDN_CORE128=0: the 256-GRF ones. The
  // 16-row core stays in the XL library: it spills at 128 GRF and its 114 KB of SLM fit one work-group per Xe core anyway
  // (preloading its a / b x rows 4 at a time removes the spill: still 107 -> 166 µs).
  static const bool c128 = [] { const char *e = std::getenv("XEN_GDN_CORE128"); return !(e && e[0] == '0'); }();
  if (c128 && sr <= kGdnMaxRows && st.accept_max <= kGdnMaxRows) {
    if (st.tree) return core_launch8_tree_v2<SWISH>(q, s, rows, x, w, st, parity, ystep, core_out);
    if (sr > kGdnRows || st.accept_max > kGdnRows) return core_launch8_v2<SWISH>(q, s, rows, x, w, st, parity, ystep, core_out);
  }
  // segments or replays past 8 rows (a 16-row tree, the step after one)
  if (sr > kGdnMaxRows || st.accept_max > kGdnMaxRows) return core_launch16_tree<SWISH>(q, s, rows, x, w, st, parity, ystep, core_out);
  if (st.tree) return core_launch8_tree<SWISH>(q, s, rows, x, w, st, parity, ystep, core_out);
  if (sr > kGdnRows || st.accept_max > kGdnRows) return core_launch8<SWISH>(q, s, rows, x, w, st, parity, ystep, core_out);
  return core_launch_km<SWISH, kGdnRows>(q, s, rows, x, w, st, parity, ystep, core_out);
}

}  // namespace

size_t gdn_S_elems(const GdnShape &s) { return static_cast<size_t>(s.vh()) * s.v_dim * s.k_dim; }
size_t gdn_win_elems(const GdnShape &s) { return 3ull * s.conv_ch(); }
size_t gdn_tree_S_elems(const GdnShape &s) { return static_cast<size_t>(kMaxRows) * gdn_S_elems(s); }

sycl::event gdn_state_reset(sycl::queue &q, const GdnShape &s, const GdnState &st) {
  q.memset(st.S, 0, gdn_S_elems(s) * sizeof(float));
  q.memset(st.win[0], 0, gdn_win_elems(s) * sizeof(bf16));
  return q.memset(st.win[1], 0, gdn_win_elems(s) * sizeof(bf16));
}

GdnEvents gdn_decode(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                     const GdnScratch &sc, const PairWire *wire, int wire_parity, bf16 *out, int qkvz_ks, int out_ks) {
  // A step carries up to 8 rows (rows.hpp), but one sequence's segment at most kGdnRows: the core is
  // per (value head, segment) and stages 2 x kGdnRows rows.
  // A segment holds up to GdnState:log_rows rows (4, or 8 with an 8-row log: the 256-GRF core)
  if (st.log_rows != kGdnRows && st.log_rows != kGdnMaxRows && st.log_rows != kGdnWideRows) throw std::invalid_argument("gdn_decode: log_rows 4, 8 or 16");
  // a 16-row log's segments / replays past 8 rows run the 16-row tree core: records only
  if (st.log_rows == kGdnWideRows && (!st.seg || !st.tree_S)) throw std::invalid_argument("gdn_decode: a 16-row log runs on records with tree_S");
  if (rows < 1 || rows > kMaxRows || (!st.seg && rows > st.log_rows) || st.accept_max > st.log_rows)
    throw std::invalid_argument("gdn_decode: rows 1..log_rows per sequence, 1..32 per step");
  if (s.k_dim != kD || s.v_dim != kD || s.k_heads % s.cards || s.v_heads % s.cards || s.v_heads % s.k_heads || s.hidden % 64)
    throw std::invalid_argument("gdn_decode: head dims must be 128, heads must split over the cards, hidden % 64 == 0");
  if (w.qkvz.n < s.qkvz_n() || w.qkvz.k != s.hidden || w.out.n != s.hidden || w.out.k != s.val_w() || !w.a || !w.b || !w.conv || !w.a_log ||
      !w.dt_bias || !w.norm)
    throw std::invalid_argument("gdn_decode: weight shapes do not match the shape");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gdn_decode: a wire is needed exactly with more than one card");
  if (st.seg && (!sc.y || !seg_rows_ok(rows, st.seg_rows, kSegRec) || seg_size(rows, st.seg_rows) > st.log_rows || seg_count(rows, st.seg_rows) > kMaxSegs))
    throw std::invalid_argument("gdn_decode: several sequences need the step scratch y and 1..rows rows per segment");
  if (st.tree && (!st.seg || !st.tree_S || st.log_rows < kGdnMaxRows || seg_size(rows, st.seg_rows) > kTreeRows))
    throw std::invalid_argument("gdn_decode: a tree step needs records, tree_S, an 8- or 16-row log and segments of <= 16 rows");
  parity &= 1;
  GdnEvents e;
  // each matrix W4 or int8 on its own (qwen38 --gdn-w4 int4-qkvz: int4 qkvz, int8 out)
  if ((w.qkvz4.w && (!w.qkvz4.valid() || w.qkvz4.n != w.qkvz.n || w.qkvz4.k != w.qkvz.k)) || (w.out4.w && (!w.out4.valid() || w.out4.n != w.out.n || w.out4.k != w.out.k)))
    throw std::invalid_argument("gdn_decode: the W4 matrices must carry the same (n, k) as the stride fields");
  float *yproj = st.seg ? sc.y : st.y[parity];
  e.qkvz = w.qkvz4.w ? w4a16_gemm_rows(q, rows, x, w.qkvz4.w, w.qkvz4.scales, yproj, w.qkvz4.n, w.qkvz4.k, qkvz_ks)
                     : dense_gemm(q, rows, x, w.qkvz, yproj, qkvz_ks);
  e.core = s.swish ? core_launch<true>(q, s, rows, x, w, st, parity, sc.y, sc.core) : core_launch<false>(q, s, rows, x, w, st, parity, sc.y, sc.core);
  e.out = w.out4.w ? w4a16_gemm_rows(q, rows, sc.core, w.out4.w, w.out4.scales, sc.partial, w.out4.n, w.out4.k, out_ks)
                   : dense_gemm(q, rows, sc.core, w.out, sc.partial, out_ks > 0 ? out_ks : kOutKSplit);
  e.reduce = pair_reduce(q, rows, s.hidden, sc.partial, wire, wire_parity, out);
  return e;
}

// ---- host reference ------------------------------------------------------------------------------
namespace {
float bfv(double v) { return bf16_value(bf16_bits(static_cast<float>(v))); }
}  // namespace

void GdnRefState::reset(const GdnShape &s) {
  S.assign(gdn_S_elems(s), 0.0);
  win.assign(gdn_win_elems(s), 0.f);
}

void gdn_ref_proj(const GdnShape &s, const GdnRefWeights &w, const float *x, float *qkvz, float *ab) {
  const int H = s.hidden;
  for (int n = 0; n < s.qkvz_n(); ++n) {
    double a = 0;
    for (int k = 0; k < H; ++k) a += static_cast<double>(x[k]) * w.qkvz[static_cast<size_t>(n) * H + k];
    qkvz[n] = bfv(a);
  }
  for (int n = 0; n < s.ab_n(); ++n) {
    double a = 0;
    for (int k = 0; k < H; ++k) a += static_cast<double>(x[k]) * w.ab[static_cast<size_t>(n) * H + k];
    ab[n] = bfv(a);
  }
}

void gdn_ref_core(const GdnShape &s, const GdnRefWeights &w, GdnRefState &st, const float *qkvz, const float *ab, float *o, float *y) {
  const int CC = s.conv_ch(), KW = s.key_w(), D = s.k_dim, vh = s.vh();
  // conv (fp32 taps oldest first, as the device accumulates)
  std::vector<float> c(CC);
  for (int ch = 0; ch < CC; ++ch) {
    const float *wt = &w.conv[static_cast<size_t>(ch) * 4];
    float acc = wt[0] * st.win[ch];
    acc = wt[1] * st.win[CC + ch] + acc;
    acc = wt[2] * st.win[2 * CC + ch] + acc;
    acc = wt[3] * qkvz[ch] + acc;
    c[ch] = bfv(static_cast<double>(acc) / (1.0 + std::exp(-static_cast<double>(acc))));
    st.win[ch] = st.win[CC + ch];
    st.win[CC + ch] = st.win[2 * CC + ch];
    st.win[2 * CC + ch] = qkvz[ch];
  }
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  std::vector<double> qn(D), kn(D), ov(D);
  for (int hv = 0; hv < vh; ++hv) {
    const int kh = hv / s.rep();
    double sq = 0, sk = 0;
    for (int d = 0; d < D; ++d) {
      sq += static_cast<double>(c[kh * D + d]) * c[kh * D + d];
      sk += static_cast<double>(c[KW + kh * D + d]) * c[KW + kh * D + d];
    }
    for (int d = 0; d < D; ++d) {
      qn[d] = c[kh * D + d] / std::sqrt(sq + 1e-6) * scale;
      kn[d] = c[KW + kh * D + d] / std::sqrt(sk + 1e-6);
    }
    const double xx = static_cast<double>(ab[hv]) + w.dt_bias[hv];
    const double sp = xx > 20 ? xx : std::log1p(std::exp(xx));
    const double decay = std::exp(-std::exp(static_cast<double>(w.a_log[hv])) * sp);
    const double beta = 1.0 / (1.0 + std::exp(-static_cast<double>(ab[vh + hv])));
    double *Sh = &st.S[static_cast<size_t>(hv) * D * D];
    double ss = 0;
    for (int v = 0; v < D; ++v) {
      double *row = Sh + static_cast<size_t>(v) * D;
      double kv = 0;
      for (int k = 0; k < D; ++k) {
        row[k] *= decay;
        kv += row[k] * kn[k];
      }
      const double d = (c[2 * KW + hv * D + v] - kv) * beta;
      double oo = 0;
      for (int k = 0; k < D; ++k) {
        row[k] += d * kn[k];
        oo += row[k] * qn[k];
      }
      ov[v] = bfv(oo);
      o[hv * D + v] = static_cast<float>(ov[v]);
      ss += ov[v] * ov[v];
    }
    const double rr = 1.0 / std::sqrt(ss / D + s.eps);
    for (int v = 0; v < D; ++v) {
      const double z = qkvz[CC + hv * D + v];
      const double sg = 1.0 / (1.0 + std::exp(-z));
      y[hv * D + v] = bfv(ov[v] * rr * w.norm[v] * (s.swish ? z * sg : sg));
    }
  }
}

void gdn_ref_out(const GdnShape &s, const GdnRefWeights &w, const float *y, double *part) {
  const int VW = s.val_w();
  for (int n = 0; n < s.hidden; ++n) {
    double a = 0;
    for (int k = 0; k < VW; ++k) a += static_cast<double>(y[k]) * w.out[static_cast<size_t>(n) * VW + k];
    part[n] = a;
  }
}

}  // namespace xenithium:kernels
