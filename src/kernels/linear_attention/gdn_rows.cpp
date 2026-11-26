// Xenithium · Gated DeltaNet over many rows.
#include "xenithium/kernels/linear_attention/gdn_rows.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"

#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = gdn_bf16;

constexpr int kD = kGdnDim;  // 128
constexpr int kVR = 4;  // value rows of S per work-item
constexpr int kCh = 16;  // conv channels per work-item
constexpr int kLog = kGdnRows;  // at most 4 replayed rows
constexpr int kCvRows = 16;  // conv rows per work-item
constexpr int kAbJ = 4;  // a | b heads per sub-group (1 if it does not divide the heads)
constexpr int kAbRowsMany = 4;  // a | b rows per sub-group from kAbManyRows rows on (1 below)
constexpr int kAbManyRows = 256;
constexpr int kAbBigRows = 1024;  // from here 8 rows x 8 heads per sub-group
constexpr int kAbRowsBig = 8;
constexpr int kAbJBig = 8;
constexpr int kVCh = 64;  // v conv channels per work-item
constexpr int kGateRun = 16;  // (row, head) gate / ln-decay indices per work-item

template <int R, int J>
class GrAb;
class GrWin;
class GrWinAt;
class GrCp;
template <int MODE>
class GrState;
class GrNorm;
class GrNormRc;

// GdnRowsState's buffers in a kernel: its pointers, or (rec set) the slot's buffers of bufs at the record's parity,
// read on the device.
struct StateAt {
  float *S, *S0;
  const bf16 *win_in;
  bf16 *win_out, *win[2];
  const float *log_y, *log_ab;
  float *y[2], *ab[2];
  size_t sS, sW, sY, sA;
  const int32_t *rec;
  explicit StateAt(const GdnRowsState &st)
      : S(st.S), S0(st.bufs.S), win_in(st.win_in), win_out(st.win_out), win{st.bufs.win[0], st.bufs.win[1]}, log_y(st.log_y), log_ab(st.log_ab),
        y{st.bufs.y[0], st.bufs.y[1]}, ab{st.bufs.ab[0], st.bufs.ab[1]}, sS(st.bufs.slot_S), sW(st.bufs.slot_win), sY(st.bufs.slot_y),
        sA(st.bufs.slot_ab), rec(st.rec) {}
  size_t slot() const { return static_cast<size_t>(rec[kSegSlot]); }
  int par() const { return rec[kSegParity] & 1; }
  int pos(int host) const { return rec ? rec[kSegPos] : host; }
  float *s() const { return rec ? S0 + slot() * sS : S; }
  const bf16 *wi() const { return rec ? win[par() ^ 1] + slot() * sW : win_in; }
  bf16 *wo() const { return rec ? win[par()] + slot() * sW : win_out; }
  const float *ly() const { return rec ? y[par() ^ 1] + slot() * sY : log_y; }
  const float *lab() const { return rec ? ab[par() ^ 1] + slot() * sA : log_ab; }
};

template <int N>
inline esimd::simd<float, N> round_bf16(esimd::simd<float, N> v) SYCL_ESIMD_FUNCTION {
  return esimd::simd<float, N>(esimd::simd<bf16, N>(v));
}
template <int N>
inline esimd::simd<float, N> sigmoid_v(esimd::simd<float, N> x) SYCL_ESIMD_FUNCTION {
  return esimd::inv(esimd::simd<float, N>(1.f) + esimd::exp(-x));
}
inline float rsqrt_ieee(float v) SYCL_ESIMD_FUNCTION {
  return esimd::div_ieee(esimd::simd<float, 1>(1.f), esimd::sqrt_ieee(esimd::simd<float, 1>(v)))[0];
}

// a | b dots for R rows x J heads per sub-group (see gdn_rows, step 2). J divides the heads per card.
template <int R, int J>
sycl::event launch_ab(sycl::queue &q, const GdnShape &s, int m, const bf16 *x, const GdnWeights &w, float *ab, int ldab) {
  const int H = s.hidden, vh = s.vh();
  const bf16 *wa = w.a, *wb = w.b;
  const int rb = (m + R - 1) / R, jb = s.ab_n() / J;
  return q.parallel_for<GrAb<R, J>>(sycl::nd_range<1>(static_cast<size_t>(rb) * jb * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int g = static_cast<int>(it.get_group(0)), r0 = (g / jb) * R, j0 = (g % jb) * J, lane = static_cast<int>(it.get_local_id(0));
    const bf16 *wr[J];
    const bf16 *xr[R];
#pragma unroll
    for (int j = 0; j < J; ++j) wr[j] = (j0 + j < vh ? wa + static_cast<size_t>(j0 + j) * H : wb + static_cast<size_t>(j0 + j - vh) * H);
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = x + static_cast<size_t>(r0 + r < m ? r0 + r : m - 1) * H;
    float acc[R][J];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int j = 0; j < J; ++j) acc[r][j] = 0.f;
    for (int k = lane; k < H; k += 16) {
      float xv[R], wv[J];
#pragma unroll
      for (int r = 0; r < R; ++r) xv[r] = static_cast<float>(xr[r][k]);
#pragma unroll
      for (int j = 0; j < J; ++j) wv[j] = static_cast<float>(wr[j][k]);
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) acc[r][j] = xv[r] * wv[j] + acc[r][j];
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int j = 0; j < J; ++j) {
        const float v = sycl::reduce_over_group(it.get_sub_group(), acc[r][j], sycl::plus<float>());
        if (lane == 0 && r0 + r < m) ab[static_cast<size_t>(r0 + r) * ldab + j0 + j] = static_cast<float>(bf16(v));
      }
  });
}


// Row core rows: all rows, or the chunked scan's lead / trail (gdn_scan_split).
constexpr int kRcAll = 0, kRcLead = 1, kRcTrail = 2;

// 5 state on the row core over rows [t_lo, t_hi) of the call's T = accept + m: one work-group per
// value head, item i owns value rows [4 i, 4 i + 4) of S; rows in order, no barrier (each item owns its
// S rows). S is read and written only when the range is not empty.
template <int MODE>
sycl::event launch_row_core(sycl::queue &q, const GdnShape &s, int m, int pos, const GdnRowsState &st, const GdnRowsScratch &sc) {
  constexpr int kItems = kD / kVR;
  const int vh = s.vh(), rep = s.rep(), KW = s.key_w(), VW = s.val_w(), CC = s.conv_ch();
  const StateAt sa(st);
  const int32_t *accept = st.accept;
  const float *cv = sc.cv, *qn = sc.qn, *kn = sc.kn, *gate = sc.gate;
  bf16 *o = sc.o;
  return q.parallel_for<GrState<MODE>>(sycl::nd_range<1>(static_cast<size_t>(vh) * kItems, kItems), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    const int hv = static_cast<int>(it.get_group(0)), i = static_cast<int>(it.get_local_id(0)), khh = hv / rep;
    const int a0 = accept[0], A = a0 < 0 ? 0 : (a0 > kLog ? kLog : a0), T = A + m;
    int t_lo = 0, t_hi = T;
    if constexpr (MODE != kRcAll) {
      int lead, full;
      gdn_scan_split(sa.pos(pos) - A, T, lead, full);
      if constexpr (MODE == kRcLead) {
        t_hi = lead;
      } else {
        t_lo = lead + full * kGdnChunk;
      }
    }
    if (t_lo >= t_hi) return;
    float *Sp = sa.s() + (static_cast<size_t>(hv) * kD + i * kVR) * kD;
    esimd::simd<float, kVR * kD> Sb;
#pragma unroll
    for (int j = 0; j < kVR; ++j) Sb.template select<kD, 1>(j * kD) = esimd::block_load<float, kD>(Sp + j * kD);
    const size_t vcol = static_cast<size_t>(2 * KW + hv * kD + i * kVR);
    // software pipeline: the next row's k, q, v and gates load while this row computes
    esimd::simd<float, kD> kr = esimd::block_load<float, kD>(kn + static_cast<size_t>(t_lo) * KW + khh * kD);
    esimd::simd<float, kD> qr = esimd::block_load<float, kD>(qn + static_cast<size_t>(t_lo) * KW + khh * kD);
    esimd::simd<float, kVR> vv = esimd::block_load<float, kVR>(cv + static_cast<size_t>(t_lo) * CC + vcol);
    float dc = gate[static_cast<size_t>(t_lo) * 2 * vh + hv], bt = gate[static_cast<size_t>(t_lo) * 2 * vh + vh + hv];
    for (int t = t_lo; t < t_hi; ++t) {
      const int tn = t + 1 < t_hi ? t + 1 : t;
      const esimd::simd<float, kD> kr2 = esimd::block_load<float, kD>(kn + static_cast<size_t>(tn) * KW + khh * kD);
      const esimd::simd<float, kD> qr2 = esimd::block_load<float, kD>(qn + static_cast<size_t>(tn) * KW + khh * kD);
      const esimd::simd<float, kVR> vv2 = esimd::block_load<float, kVR>(cv + static_cast<size_t>(tn) * CC + vcol);
      const float dc2 = gate[static_cast<size_t>(tn) * 2 * vh + hv], bt2 = gate[static_cast<size_t>(tn) * 2 * vh + vh + hv];
      Sb *= dc;
#pragma unroll
      for (int j = 0; j < kVR; ++j) {
        esimd::simd<float, kD> rw = Sb.template select<kD, 1>(j * kD);
        const float d = (vv[j] - esimd::reduce<float>(rw * kr, std::plus<>())) * bt;
        Sb.template select<kD, 1>(j * kD) = kr * d + rw;
      }
      if (t >= A) {
        esimd::simd<float, kVR> ov;
#pragma unroll
        for (int j = 0; j < kVR; ++j) {
          esimd::simd<float, kD> rw = Sb.template select<kD, 1>(j * kD);
          ov[j] = esimd::reduce<float>(rw * qr, std::plus<>());
        }
        esimd::block_store<bf16, kVR>(o + static_cast<size_t>(t - A) * VW + hv * kD + i * kVR, esimd::simd<bf16, kVR>(ov));
      }
      kr = kr2;
      qr = qr2;
      vv = vv2;
      dc = dc2;
      bt = bt2;
    }
#pragma unroll
    for (int j = 0; j < kVR; ++j) esimd::block_store<float, kD>(Sp + j * kD, Sb.template select<kD, 1>(j * kD));
  });
}
}  // namespace

GdnRowsSizes gdn_rows_sizes(const GdnShape &s, int qkvz_n, int rows_max) {
  const size_t R = static_cast<size_t>(rows_max), T = R + kLog;
  GdnRowsSizes z;
  z.yq = R * qkvz_n;
  z.ab = R * s.ab_n();
  z.cv = T * s.conv_ch();
  z.qk = T * s.key_w();
  z.gate = T * 2 * s.vh();
  z.lgate = T * s.vh();
  z.o = R * s.val_w();
  z.partial = R * s.hidden;
  return z;
}

const char *gdn_stage_name(GdnStage s) {
  static const char *n[] = {"projections", "prep", "scan pre", "state", "norm", "out + sum"};
  static_assert(std::size(n) == static_cast<size_t>(GdnStage::kCount), "a name per stage");
  const int i = static_cast<int>(s);
  return i >= 0 && i < static_cast<int>(GdnStage::kCount) ? n[i] : "?";
}

// One 16-lane sub-group per (R rows, J of the a | b dots' heads). Each dot keeps the one-dot form's order (lane l sums
// k = l, l + 16... then the sub-group reduction), so R and J change no bits; they cut the x / weight re-reads (8 rows x
// 8 heads from 1,024 rows).
sycl::event gdn_rows_ab(sycl::queue &q, const GdnShape &s, int m, const bf16 *x, const GdnWeights &w, float *ab, int ld) {
  if (m < 1 || ld < s.ab_n() || !x || !w.a || !w.b || !ab) throw std::invalid_argument("gdn_rows_ab: rows / stride / buffers");
  const int vh = s.vh();
  if (vh % kAbJ) return launch_ab<1, 1>(q, s, m, x, w, ab, ld);
  if (m >= kAbBigRows && vh % kAbJBig == 0) return launch_ab<kAbRowsBig, kAbJBig>(q, s, m, x, w, ab, ld);
  if (m >= kAbManyRows) return launch_ab<kAbRowsMany, kAbJ>(q, s, m, x, w, ab, ld);
  return launch_ab<1, kAbJ>(q, s, m, x, w, ab, ld);
}

GdnEvents gdn_rows(sycl::queue &q, const GdnShape &s, int m, const bf16 *x, const GdnWeights &w, const GdnRowsState &st, const GdnRowsScratch &sc,
                   const PairRowsWire *wire, int wire_parity, bf16 *out) {
  if (m < 1) throw std::invalid_argument("gdn_rows: rows");
  if (s.k_dim != kD || s.v_dim != kD || s.v_heads % s.k_heads || s.hidden % 64 || s.conv_ch() % kCh || s.val_w() % kVCh)
    throw std::invalid_argument("gdn_rows: head dims must be 128");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gdn_rows: a wire is needed exactly with more than one card");
  // qwen35: qkvz and out as W4 matrices, w4a16_gemm_pf. qwen38 --gdn-w4: each
  // matrix on its own (int4 qkvz with int8 out is legal); sc.a8 applies to the int8 ones, a W4 matrix runs W4A16.
  const bool w4q = w.qkvz4.w != nullptr;
  const int H = s.hidden, vh = s.vh(), kh = s.kh(), KW = s.key_w(), VW = s.val_w(), CC = s.conv_ch(), ldy = w.qkvz.n,
            ldab = s.ab_n();
  const int Tmax = m + kLog;
  const float scale = 1.f / std::sqrt(static_cast<float>(s.k_dim)), eps = s.eps;
  const int32_t *accept = st.accept;
  GdnEvents ev;
  auto traced = [&](GdnStage stage, const sycl::event &e) {  // prefill profiling (sc.trace)
    if (sc.trace) sc.trace->add(static_cast<int>(stage), e);
    return e;
  };

  if (sc.x8.ready && (w4q || !sc.x8.row || sc.qkvz_ready)) throw std::invalid_argument("gdn_rows: owner-quantised x is for the int8 qkvz on W8A8 per row");
  if (sc.ab_in && sc.ab_ld < ldab) throw std::invalid_argument("gdn_rows: the owners' a | b rows' stride");
  // 1 qkvz (raw fp32)
  if (!sc.qkvz_ready)
    ev.qkvz = traced(GdnStage::projections, w4q ? w4a16_gemm_pf(q, m, x, H, w.qkvz4.w, w.qkvz4.scales, sc.yq, ldy, w.qkvz4.n, H)
                                                : pf_dense(q, m, w.qkvz, x, H, PfOut{sc.yq, nullptr, ldy}, sc.x8.ready ? &sc.x8 : sc.a8.on()));

  // 2 a | b dots (gdn_rows_ab), unless the rows' owners made them
  if (!sc.ab_in) traced(GdnStage::projections, gdn_rows_ab(q, s, m, x, w, sc.ab, ldab));

  // 3+4 conv, prep and gate in one launch. Three item kinds:
  // q / k one item per (16 rows, one 128-channel head): the conv slides over the rows in registers,
  // then the l2 norm of the bf16 conv output -> qn / kn (the q / k conv outputs never reach
  // memory); the norm's reduce runs over all 128 channels.
  // v 16 channels x 16 rows -> cv (the v columns only: nothing reads cv's q / k columns).
  // gate per (row, value head): decay = exp(-exp(A_log) softplus(a + dt_bias)), beta = sigmoid(b).
  // One row per item on purpose: the same expression over 16 rows as 16 lanes rounds exp /
  // log differently.
  // ln a (with the chunked scan only) per (row, value head): -exp(A_log) softplus(...)
  // Conv input t of the sequence seq[] (window, replayed log rows, then new rows): output t reads
  // seq[t.. t + 3]; c = bf16(silu(taps, oldest first)).
  {
    const StateAt sa(st);
    const float *yq = sc.yq, *ab = sc.ab_in ? sc.ab_in : sc.ab;
    const int ldab_new = sc.ab_in ? sc.ab_ld : ldab;  // the new rows' a | b (the replayed log keeps ldab)
    const bf16 *conv = w.conv, *alog = w.a_log, *dtb = w.dt_bias;
    float *cv = sc.cv, *qn = sc.qn, *kn = sc.kn, *gate = sc.gate, *lgate = sc.lgate;
    const int tblocks = (Tmax + kCvRows - 1) / kCvRows;
    // a v item takes kVCh channels and a gate / ln-decay item kGateRun consecutive
    // (row, head) indices in a scalar loop. The same expressions per element whatever the grouping, so the same bits.
    const int vgroups = VW / kVCh, nqk = tblocks * 2 * kh, nv = tblocks * vgroups;
    const int ngate = (Tmax * vh + kGateRun - 1) / kGateRun;
    const int nlg = st.pos >= 0 && gdn_scan_max_chunks(m) > 0 ? ngate : 0;  // ln decay items
    const sycl::event conv_prep = q.parallel_for<GrCp>(sycl::range<1>(static_cast<size_t>(nqk + nv + ngate + nlg)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      const int a0 = accept[0], A = a0 < 0 ? 0 : (a0 > kLog ? kLog : a0), T = A + m;
      const bf16 *win = sa.wi();
      const float *logy = sa.ly(), *lab = sa.lab();
      int idx = static_cast<int>(id[0]);
      if (idx < nqk) {
        const int t0 = (idx / (2 * kh)) * kCvRows, hh = idx % (2 * kh), c0 = hh * kD;
        if (t0 >= T) return;
        const int t1 = t0 + kCvRows < T ? t0 + kCvRows : T;
        auto seq = [=](int j) SYCL_ESIMD_FUNCTION -> esimd::simd<float, kD> {
          if (j < 3) return esimd::block_load<bf16, kD>(win + static_cast<size_t>(j) * CC + c0);
          const int t = j - 3;
          const float *src = t < A ? logy + static_cast<size_t>(t) * ldy + c0 : yq + static_cast<size_t>(t - A) * ldy + c0;
          return round_bf16<kD>(esimd::block_load<float, kD>(src));
        };
        esimd::simd<float, kD * kGdnTaps> wt = esimd::block_load<bf16, kD * kGdnTaps>(conv + static_cast<size_t>(c0) * kGdnTaps);
        esimd::simd<float, kD> s0 = seq(t0), s1 = seq(t0 + 1), s2 = seq(t0 + 2);
        const bool isk = hh >= kh;
        float *dst = (isk ? kn : qn) + (isk ? hh - kh : hh) * kD;
        for (int t = t0; t < t1; ++t) {
          const esimd::simd<float, kD> s3 = seq(t + 3);
          esimd::simd<float, kD> c = wt.template select<kD, kGdnTaps>(0) * s0;
          c = wt.template select<kD, kGdnTaps>(1) * s1 + c;
          c = wt.template select<kD, kGdnTaps>(2) * s2 + c;
          c = wt.template select<kD, kGdnTaps>(3) * s3 + c;
          const esimd::simd<float, kD> v = round_bf16<kD>(c * sigmoid_v<kD>(c));
          const float r = rsqrt_ieee(esimd::reduce<float>(v * v, std::plus<>()) + 1e-6f);
          esimd::block_store<float, kD>(dst + static_cast<size_t>(t) * KW, isk ? esimd::simd<float, kD>(v * r) : esimd::simd<float, kD>(v * (r * scale)));
          s0 = s1;
          s1 = s2;
          s2 = s3;
        }
        return;
      }
      idx -= nqk;
      if (idx < nv) {
        const int t0 = (idx / vgroups) * kCvRows, c0 = 2 * KW + (idx % vgroups) * kVCh;
        if (t0 >= T) return;
        const int t1 = t0 + kCvRows < T ? t0 + kCvRows : T;
        auto seq = [=](int j) SYCL_ESIMD_FUNCTION -> esimd::simd<float, kVCh> {
          if (j < 3) return esimd::block_load<bf16, kVCh>(win + static_cast<size_t>(j) * CC + c0);
          const int t = j - 3;
          const float *src = t < A ? logy + static_cast<size_t>(t) * ldy + c0 : yq + static_cast<size_t>(t - A) * ldy + c0;
          return round_bf16<kVCh>(esimd::block_load<float, kVCh>(src));
        };
        esimd::simd<float, kVCh * kGdnTaps> wt = esimd::block_load<bf16, kVCh * kGdnTaps>(conv + static_cast<size_t>(c0) * kGdnTaps);
        esimd::simd<float, kVCh> s0 = seq(t0), s1 = seq(t0 + 1), s2 = seq(t0 + 2);
        for (int t = t0; t < t1; ++t) {
          const esimd::simd<float, kVCh> s3 = seq(t + 3);
          esimd::simd<float, kVCh> c = wt.template select<kVCh, kGdnTaps>(0) * s0;
          c = wt.template select<kVCh, kGdnTaps>(1) * s1 + c;
          c = wt.template select<kVCh, kGdnTaps>(2) * s2 + c;
          c = wt.template select<kVCh, kGdnTaps>(3) * s3 + c;
          esimd::block_store<float, kVCh>(cv + static_cast<size_t>(t) * CC + c0, round_bf16<kVCh>(c * sigmoid_v<kVCh>(c)));
          s0 = s1;
          s1 = s2;
          s2 = s3;
        }
        return;
      }
      idx -= nv;
      if (idx < ngate) {
#pragma unroll 1
        for (int e = idx * kGateRun; e < (idx + 1) * kGateRun; ++e) {
          const int t = e / vh, hv = e % vh;
          if (t >= T) return;
          const float *src = t < A ? lab + static_cast<size_t>(t) * ldab : ab + static_cast<size_t>(t - A) * ldab_new;
          const float av = src[hv], bv = src[vh + hv];
          const esimd::simd<float, 1> xx(av + static_cast<float>(dtb[hv]));
          esimd::simd<float, 1> sp = esimd::log(esimd::simd<float, 1>(1.f) + esimd::exp(xx));
          sp.merge(xx, xx > 20.f);
          const float ea = esimd::exp(esimd::simd<float, 1>(static_cast<float>(alog[hv])))[0];
          const esimd::simd<float, 1> decay = esimd::exp(sp * (-ea));
          const esimd::simd<float, 1> beta = sigmoid_v<1>(esimd::simd<float, 1>(bv));
          gate[static_cast<size_t>(t) * 2 * vh + hv] = decay[0];
          gate[static_cast<size_t>(t) * 2 * vh + vh + hv] = beta[0];
        }
        return;
      }
      // the chunked scan's g: ln decay = -exp(A_log) softplus(a + dt_bias) on items of its own
      // (only with st.pos >= 0). Storing it from the gate items above would change the decay's bits: fast
      // math evaluates exp(sp * (-ea)) differently once sp * (-ea) is kept.
      idx -= ngate;
#pragma unroll 1
      for (int e = idx * kGateRun; e < (idx + 1) * kGateRun; ++e) {
        const int t = e / vh, hv = e % vh;
        if (t >= T) return;
        const float *src = t < A ? lab + static_cast<size_t>(t) * ldab : ab + static_cast<size_t>(t - A) * ldab_new;
        const esimd::simd<float, 1> xx(src[hv] + static_cast<float>(dtb[hv]));
        esimd::simd<float, 1> sp = esimd::log(esimd::simd<float, 1>(1.f) + esimd::exp(xx));
        sp.merge(xx, xx > 20.f);
        const float ea = esimd::exp(esimd::simd<float, 1>(static_cast<float>(alog[hv])))[0];
        lgate[static_cast<size_t>(t) * vh + hv] = (sp * (-ea))[0];
      }
    });
    traced(GdnStage::prep, conv_prep);
    // The committed window after the last row: seq[T.. T + 2], all channels. After the launch above,
    // which still reads win_in (win_out may equal it).
    const int groups = CC / kCh;
    ev.core = q.parallel_for<GrWin>(sycl::range<1>(static_cast<size_t>(groups)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      const int c0 = static_cast<int>(id[0]) * kCh;
      const int a0 = accept[0], A = a0 < 0 ? 0 : (a0 > kLog ? kLog : a0), T = A + m;
      const bf16 *win = sa.wi();
      bf16 *wout = sa.wo();
      const float *logy = sa.ly();
      auto seq = [=](int j) SYCL_ESIMD_FUNCTION -> esimd::simd<float, kCh> {
        if (j < 3) return esimd::block_load<bf16, kCh>(win + static_cast<size_t>(j) * CC + c0);
        const int t = j - 3;
        const float *src = t < A ? logy + static_cast<size_t>(t) * ldy + c0 : yq + static_cast<size_t>(t - A) * ldy + c0;
        return round_bf16<kCh>(esimd::block_load<float, kCh>(src));
      };
      const esimd::simd<float, kCh> w0 = seq(T), w1 = seq(T + 1), w2 = seq(T + 2);
      esimd::block_store<bf16, kCh>(wout + c0, esimd::simd<bf16, kCh>(w0));
      esimd::block_store<bf16, kCh>(wout + CC + c0, esimd::simd<bf16, kCh>(w1));
      esimd::block_store<bf16, kCh>(wout + 2 * CC + c0, esimd::simd<bf16, kCh>(w2));
    });
    traced(GdnStage::prep, ev.core);
    if (st.snap_pos >= 0) {  // the window at snap_pos, the same expression (rows before it)
      bf16 *sw = st.snap_win;
      const int dpos = st.snap_pos - st.pos;
      const sycl::event win_at = q.parallel_for<GrWinAt>(sycl::range<1>(static_cast<size_t>(groups)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const int c0 = static_cast<int>(id[0]) * kCh;
        const int a0 = accept[0], A = a0 < 0 ? 0 : (a0 > kLog ? kLog : a0), T = A + dpos;
        const bf16 *win = sa.wi();
        const float *logy = sa.ly();
        auto seq = [=](int j) SYCL_ESIMD_FUNCTION -> esimd::simd<float, kCh> {
          if (j < 3) return esimd::block_load<bf16, kCh>(win + static_cast<size_t>(j) * CC + c0);
          const int t = j - 3;
          const float *src = t < A ? logy + static_cast<size_t>(t) * ldy + c0 : yq + static_cast<size_t>(t - A) * ldy + c0;
          return round_bf16<kCh>(esimd::block_load<float, kCh>(src));
        };
        const esimd::simd<float, kCh> w0 = seq(T), w1 = seq(T + 1), w2 = seq(T + 2);
        esimd::block_store<bf16, kCh>(sw + c0, esimd::simd<bf16, kCh>(w0));
        esimd::block_store<bf16, kCh>(sw + CC + c0, esimd::simd<bf16, kCh>(w1));
        esimd::block_store<bf16, kCh>(sw + 2 * CC + c0, esimd::simd<bf16, kCh>(w2));
      });
      traced(GdnStage::prep, win_at);
    }
  }

  // 5 state. With st.pos >= 0 the full 64-row chunks at absolute multiples of 64 run on the
  // chunked scan (gdn_scan.hpp); the rows before the first boundary (lead) and after the last full
  // chunk (trail) stay on the row core, so small calls keep the row core's numerics and cost, and a
  // prompt cut at multiples of 64 gives the same bits as one call. The split is computed on the device
  // (it depends on the accept word).
  // On the scan the gated norm (step 6) of the scan's rows runs inside the scan (GdnScanNorm,
  // the same expression), so their o rows never reach memory; step 6 then norms only the row core's lead /
  // trail rows (< 64 each, found on the device).
  const bool fold = st.pos >= 0 && gdn_scan_max_chunks(m) > 0;
  if (st.snap_pos >= 0 && (!fold || st.snap_pos % kGdnChunk || st.snap_pos <= st.pos || st.snap_pos >= st.pos + m || !st.snap_S || !st.snap_win))
    throw std::invalid_argument("gdn_rows: a snapshot at a multiple of 64 inside a call on the chunked scan");
  if (st.snap_pos >= 0 && st.rec) throw std::invalid_argument("gdn_rows: a snapshot takes the host's position and state (no rec)");
  if (fold) {
    traced(GdnStage::state, launch_row_core<kRcLead>(q, s, m, st.pos, st, sc));
    const GdnScanNorm nm{sc.yq + CC, ldy, w.norm, eps, sc.y, s.swish};
    sycl::event pre;  // the scan's precompute, its own stage
    if (st.snap_pos >= 0) {  // the scan in two parts around snap_pos, S copied out between them
      traced(GdnStage::state, gdn_scan(q, s, m, st.pos, accept, sc.qn, sc.kn, sc.cv, sc.gate, sc.lgate, st.S, sc.o, sc.scan, nm, 1, st.snap_pos, &pre));
      traced(GdnStage::scan_pre, pre);
      traced(GdnStage::state, q.memcpy(st.snap_S, st.S, gdn_S_elems(s) * 4));
      traced(GdnStage::state, gdn_scan(q, s, m, st.pos, accept, sc.qn, sc.kn, sc.cv, sc.gate, sc.lgate, st.S, sc.o, sc.scan, nm, 2, st.snap_pos));
    } else {
      traced(GdnStage::state, gdn_scan(q, s, m, st.pos, accept, sc.qn, sc.kn, sc.cv, sc.gate, sc.lgate, st.rec ? st.bufs.S : st.S, sc.o, sc.scan, nm, 0, -1, &pre, st.rec,
                                                      st.bufs.slot_S));
      traced(GdnStage::scan_pre, pre);
    }
    traced(GdnStage::state, launch_row_core<kRcTrail>(q, s, m, st.pos, st, sc));
  } else {
    traced(GdnStage::state, launch_row_core<kRcAll>(q, s, m, 0, st, sc));
  }

  // 6 gated norm per (row, value head)
  {
    const bf16 *o = sc.o, *nw = w.norm;
    const float *yq = sc.yq;
    bf16 *y = sc.y;
    const bool swish = s.swish;
    auto norm = [=](int r, int hv) SYCL_ESIMD_FUNCTION {
      const esimd::simd<float, kD> ov = esimd::block_load<bf16, kD>(o + static_cast<size_t>(r) * VW + hv * kD);
      const float rr = rsqrt_ieee(esimd::reduce<float>(ov * ov, std::plus<>()) / static_cast<float>(kD) + eps);
      const esimd::simd<float, kD> wn = esimd::block_load<bf16, kD>(nw);
      const esimd::simd<float, kD> z = round_bf16<kD>(esimd::block_load<float, kD>(yq + static_cast<size_t>(r) * ldy + CC + hv * kD));
      esimd::simd<float, kD> gate = sigmoid_v<kD>(z);
      if (swish) gate = z * gate;  // qwen35: silu(z) = z sigmoid(z), the decode core's expression
      const esimd::simd<float, kD> yv = ov * rr * wn * gate;
      esimd::block_store<bf16, kD>(y + static_cast<size_t>(r) * VW + hv * kD, esimd::simd<bf16, kD>(yv));
    };
    if (!fold) {
      traced(GdnStage::norm, q.parallel_for<GrNorm>(sycl::range<1>(static_cast<size_t>(m) * vh), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        norm(static_cast<int>(id[0] / vh), static_cast<int>(id[0] % vh));
      }));
    } else {
      // rows t of T = accept + m: k < 64 -> lead row t = k; k >= 64 -> trail row lead + 64 full + k - 64
      const int32_t *acc = accept, *rec = st.rec;
      const int pos = st.pos;
      const sycl::event norm_rc = q.parallel_for<GrNormRc>(sycl::range<1>(static_cast<size_t>(2 * kGdnChunk) * vh), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        const int k = static_cast<int>(id[0] / vh), hv = static_cast<int>(id[0] % vh);
        const int a0 = acc[0], A = a0 < 0 ? 0 : (a0 > kLog ? kLog : a0), T = A + m;
        int lead, full;
        gdn_scan_split((rec ? rec[kSegPos] : pos) - A, T, lead, full);
        const int t = k < kGdnChunk ? k : lead + full * kGdnChunk + (k - kGdnChunk);
        if ((k < kGdnChunk && t >= lead) || t >= T || t < A) return;
        norm(t - A, hv);
      });
      traced(GdnStage::norm, norm_rc);
    }
  }

  // 7 out projection + the two-card sum
  if (sc.out_later) return ev;  // the caller runs gdn_rows_out over the rows of several calls
  const GdnEvents t = gdn_rows_out(q, s, m, w, sc, wire, wire_parity, out);
  ev.out = t.out, ev.send = t.send, ev.reduce = t.reduce;
  return ev;
}

GdnEvents gdn_rows_out(sycl::queue &q, const GdnShape &s, int m, const GdnWeights &w, const GdnRowsScratch &sc, const PairRowsWire *wire, int wire_parity,
                       bf16 *out) {
  if (m < 1) throw std::invalid_argument("gdn_rows_out: rows");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("gdn_rows_out: a wire is needed exactly with more than one card");
  if (wire && wire->split && s.cards != 2 && s.cards != kWireCardsMax) throw std::invalid_argument("gdn_rows_out: the split sum is two- or four-card");
  if (wire && wire->split && wire->send && wire->send->sp && wire->send->sp->cards != s.cards)
    throw std::invalid_argument("gdn_rows_out: the split's cards differ from the shape's");
  const bool w4o = w.out4.w != nullptr;
  const int H = s.hidden, VW = s.val_w();
  GdnEvents ev;
  auto traced = [&](const sycl::event &e) {  // prefill profiling (sc.trace): every launch here is the out stage
    if (sc.trace) sc.trace->add(static_cast<int>(GdnStage::out), e);
    return e;
  };
  auto out_gemm = [&](int lo, int rows) {
    const bf16 *ya = sc.y + static_cast<size_t>(lo) * VW;
    float *pp = sc.partial + static_cast<size_t>(lo) * H;
    return traced(w4o ? w4a16_gemm_pf(q, rows, ya, VW, w.out4.w, w.out4.scales, pp, H, H, VW) : pf_dense(q, rows, w.out, ya, VW, PfOut{pp, nullptr, H}, sc.a8.on()));
  };
  if (wire && wire->split && wire->send) {
    // the peer's rows first, their copy beside the own rows
    // (four cards = every other card's rows, [0, lo) and [hi, m); two: the peer's, the same launch)
    int lo = 0, hi = 0;
    pair_split_owner_rows(s.card, m, &lo, &hi, wire->send->sp->cards);
    if (lo > 0) out_gemm(0, lo);
    if (hi < m) out_gemm(hi, m - hi);
    ev.send = traced(pair_split_send(q, wire->send->side, s.card, m, H, sc.partial, *wire->send->sp, wire->send->seq));
    if (hi > lo) ev.out = out_gemm(lo, hi - lo);
  } else {
    ev.out = out_gemm(0, m);
  }
  if (!wire || !wire->split) ev.reduce = traced(pair_reduce_rows(q, m, H, sc.partial, wire, wire_parity, out));
  return ev;
}

}  // namespace xenithium:kernels
