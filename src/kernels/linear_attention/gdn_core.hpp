// Xenithium · the GDN decode core (internal: gdn.cpp instantiates the 4-row segments, gdn_core8.cpp the 8-row ones in the
// 256-GRF library). See gdn.hpp for the semantics and the row protocol.
#pragma once

#include <cmath>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/rows.hpp"

namespace xenithium::kernels::gdn_detail {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = gdn_bf16;

constexpr int kD = kGdnDim;  // 128
// Value rows of S per work-item (32 items per head).
constexpr int kVR = 4;
constexpr int kOutKSplit = 8;  // out_proj [hidden][val_w]: its k split

// LIB: the device library the instance is compiled into (0: its own file's, 1: the 128-GRF gdn_core_v2.cpp). A kernel
// name in two device images is ambiguous: the runtime runs one of them
template <int VR, bool SWISH, int NR, int NP, int ABI, int KM, bool TREE, int LIB>
class GdnCoreK;

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

// One work-group per value head of this card; item i owns value rows [i VR, i VR + VR) of S and
// conv channels [i VR, i VR + VR) of the head's q, k and v groups. Latency-bound (3 MB of state
// over 24 work-groups), so every global load is issued up front, before anything depends on
// `accept`: S rows, both steps' projection rows (staged in SLM, picked by accept afterwards),
// windows, conv taps, z, the previous a / b, the scalars; then the a / b dots.
// 1 conv: replayed rows (the previous step's projections) then this step's rows, through
// the channel window; bf16(silu) into SLM; the window after the replayed rows is the
// new committed one (written to win[parity] by its owner: the v group by its head,
// q/k by the first head of each key head).
// 2 gates: a / b of replayed rows from the previous step's log, of new rows from the SLM
// partials (summed in item order, bf16; item 0 logs them); g, beta per row.
// 3 state: per row: decay, delta update, output for new rows; the S after the replayed rows
// is written back (the committed state).
// 4 norm: gated RMSNorm of each new row's o (128 values from SLM), y -> core.
// The compile-time shape:
// NR rows of this step the code is unrolled for (1 or kM): a 1-row step loads and dots one x row, one z row and stages
// one current row instead of four clamped copies.
// NP previous-step rows staged (1 or kM): the host's bound on every accept word (GdnState:accept_max).
// ABI a / b dot iterations known at compile time (hidden / 512; 0 = the runtime loop): every weight and x load first,
// then the FMAs in the loop's order (one memory round trip instead of three).
// The previous step's a / b words come in one 8-lane gather instead of 8 scalar loads.
// commit (GdnState:commit): the step's rows are final (a 1-row step is never speculative), so S and the window are
// committed after the last row and the next step replays nothing (accept 0): one state update per row instead of
// two at 1-row decode.
// TREE (DDTree; records only): the segment's record may carry a draft tree (kSegTree) and the previous step's
// accepted rows as a path mask (kSegPath). The replayed rows are staged from the mask's rows; a current row's conv taps
// are its ancestors' inputs (then the window after the replay), and its state update starts from its parent's state:
// the row before it, or (a branch) the parent's state saved to GdnState:tree_S when its first child started. So row
// j's out / y are what a chain of its ancestors computes, bit for bit (the same expressions on the same values).
// The 4-tap conv is one explicit multiply and three fused multiply-adds (taps oldest first): under the project's fast
// floating point the compiler would otherwise choose the contraction per instance, and the tree instance's chain rows
// would come out of another form than the chain cores'.
template <int VR>
inline esimd::simd<float, VR> conv4(esimd::simd<float, VR * kGdnTaps> wt, const esimd::simd<float, VR> &a, const esimd::simd<float, VR> &b,
                                    const esimd::simd<float, VR> &c, const esimd::simd<float, VR> &u) SYCL_ESIMD_FUNCTION {
  const esimd::simd<float, VR> w0 = wt.template select<VR, kGdnTaps>(0), w1 = wt.template select<VR, kGdnTaps>(1), w2 = wt.template select<VR, kGdnTaps>(2),
                               w3 = wt.template select<VR, kGdnTaps>(3);
  esimd::simd<float, VR> r = w0 * a;
  r = esimd::simd<float, VR>(__builtin_elementwise_fma(w1.data(), b.data(), r.data()));
  r = esimd::simd<float, VR>(__builtin_elementwise_fma(w2.data(), c.data(), r.data()));
  return esimd::simd<float, VR>(__builtin_elementwise_fma(w3.data(), u.data(), r.data()));
}

template <int VR, bool SWISH, int NR, int NP, int ABI, int KM, bool TREE = false, int LIB = 0>
sycl::event core(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                 bf16 *core_out) {
  constexpr int kItems = kD / VR;
  constexpr int kM = KM, kTMax = 2 * KM;  // rows per segment this instance stages (4, or 8)
  constexpr int kAbW = 16;  // k elements per a / b load
  constexpr int kStageF = 2 * kM * 3 * kD, kConvF = kTMax * 3 * kD, kOF = kM * kD, kAbF = kItems * 2 * kM;
  // TREE: + one saved state of the head (aux bit 48: a tree's only saved row) after the rest. Not
  // in the 16-row instance (its stage and conv rows fill the shared local memory: every saved state goes to tree_S)
  // not in the 128-GRF library's instances either: its 64 KB would hold the 8-row tree core to one work-group per
  // Xe core (two waves); its one saved state goes to tree_S (L2) like the rest
  constexpr bool kSaveSlm = TREE && KM <= 8 && LIB == 0;
  constexpr int kSaveF = kSaveSlm ? kD * kD : 0;
  constexpr int kSlm = (kStageF + kConvF + kOF + kAbF + kSaveF) * static_cast<int>(sizeof(float));
  constexpr int kConvBase = kStageF, kOBase = kStageF + kConvF, kAbBase = kStageF + kConvF + kOF, kSaveBase = kAbBase + kAbF;  // floats
  const int vh = s.vh(), rep = s.rep(), KW = s.key_w(), VW = s.val_w(), CC = s.conv_ch(), H = s.hidden, ldy = w.qkvz.n, ldab = s.ab_n();
  const float scale = 1.f / std::sqrt(static_cast<float>(s.k_dim)), eps = s.eps;
  // One sequence: host parity, slot 0, the projections are the log (y[parity]). Several: per work-group
  // (value head, segment) the segment's record gives accept / parity / slot; the step's projections
  // come from ystep and the core writes the segment's rows into its slot's log.
  const bool segm = st.seg != nullptr;
  const int sr = segm ? seg_size(rows, st.seg_rows) : rows, nseg = segm ? seg_count(rows, st.seg_rows) : 1;
  float *y0 = st.y[0], *y1 = st.y[1], *ab0 = st.ab[0], *ab1 = st.ab[1];
  bf16 *wn0 = st.win[0], *wn1 = st.win[1];
  const float *yst = segm ? ystep : st.y[parity];
  const size_t sS = st.slot_S, sW = st.slot_win, sY = st.slot_y, sA = st.slot_ab;
  float *S = st.S;
  const int32_t *accept = st.accept, *rec = st.seg;
  const int hpar = parity;
  const bool commit_new = st.commit;
  const int seg_code = st.seg_rows;  // seg_het = unequal segments (records give each one's rows)
  const bf16 *conv = w.conv, *alog = w.a_log, *dtb = w.dt_bias, *nw = w.norm, *wa = w.a, *wb = w.b;
  float *treeS = st.tree_S;
  return q.parallel_for<GdnCoreK<VR, SWISH, NR, NP, ABI, KM, TREE, LIB>>(
      sycl::nd_range<2>({static_cast<size_t>(nseg), static_cast<size_t>(vh) * kItems}, {1, kItems}), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlm>();
        // a (segment, value head) grid: the ids come from the launch, not from a division by the run-time head count in every item
        const int sg = static_cast<int>(it.get_group(0)), hv = static_cast<int>(it.get_group(1)), i = static_cast<int>(it.get_local_id(1)), kh = hv / rep;
        auto slm_off = [](int f) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>(f * sizeof(float)); };
        // the segment: rows [r0, r0 + srows) of the step; its slot's state and logs
        int r0 = sg * sr, srows = rows - r0 < sr ? rows - r0 : sr;
        if (segm && seg_code < 0) seg_span(seg_code, rows, sg, r0, srows);  // unequal segments
        const int32_t *sgr = rec + sg * kSegRec;
        const int acc0 = segm ? sgr[kSegAccept] : accept[0];
        const int par = segm ? (sgr[kSegParity] & 1) : hpar;
        const size_t slot = segm ? static_cast<size_t>(sgr[kSegSlot]) : 0;
        // the tree word and the previous step's accepted-row mask (0 / 0: a chain)
        const uint64_t tw = TREE ? rec_word(sgr, kSegTree, kSegTreeHi) : 0ull, pmask = TREE ? rec_word(sgr, kSegPath, kSegPathHi) : 0ull;
        auto prow = [&](int r) SYCL_ESIMD_FUNCTION { return TREE ? path_row(pmask, r) : r; };
        const float *ycur = yst + static_cast<size_t>(r0) * ldy;
        const float *yprev = (par ? y0 : y1) + slot * sY;
        float *ylog = (par ? y1 : y0) + slot * sY;  // this step's rows, for the sequence's next step
        const float *abprev = (par ? ab0 : ab1) + slot * sA;
        float *abcur = (par ? ab1 : ab0) + slot * sA;
        const bf16 *winR = (par ? wn0 : wn1) + slot * sW;
        bf16 *winW = (par ? wn1 : wn0) + slot * sW;
        // a current row j of the segment (rows past it repeat its last row: finite, never kept)
        auto crow = [&](int j) SYCL_ESIMD_FUNCTION { return static_cast<size_t>(j < srows ? j : srows - 1); };

        // ---- 0: every global load ----
        float *Sp = S + slot * sS + (static_cast<size_t>(hv) * kD + i * VR) * kD;
        esimd::simd<float, VR * kD> Sb;
#pragma unroll
        for (int j = 0; j < VR; ++j) Sb.template select<kD, 1>(j * kD) = esimd::block_load<float, kD>(Sp + j * kD);
        int ch0[3];
        esimd::simd<float, VR * kGdnTaps> wt[3];
        esimd::simd<float, VR> win[3][3];
#pragma unroll
        for (int G = 0; G < 3; ++G) {
          ch0[G] = (G == 0 ? kh * kD : (G == 1 ? KW + kh * kD : 2 * KW + hv * kD)) + i * VR;
          wt[G] = esimd::block_load<bf16, VR * kGdnTaps>(conv + static_cast<size_t>(ch0[G]) * kGdnTaps);
#pragma unroll
          for (int j = 0; j < 3; ++j) win[G][j] = esimd::block_load<bf16, VR>(winR + j * CC + ch0[G]);
          // staged rows: [0, NP) of the previous step's [0, 4), [4, 4 + NR) this step; raw fp32
#pragma unroll
          for (int r = 0; r < kM + NR; ++r) {
            if (r >= NP && r < kM) continue;  // past the host's accept bound: never read
            const float *src = r < kM ? yprev + static_cast<size_t>(prow(r)) * ldy + ch0[G] : ycur + crow(r - kM) * ldy + ch0[G];
            esimd::slm_block_store<float, VR>(slm_off((r * 3 + G) * kD + i * VR), esimd::block_load<float, VR>(src));
          }
        }
        esimd::simd<float, VR> zr[kM];
#pragma unroll
        for (int r = 0; r < NR; ++r) zr[r] = esimd::block_load<float, VR>(ycur + crow(r) * ldy + CC + hv * kD + i * VR);
        const esimd::simd<float, VR> wn = esimd::block_load<bf16, VR>(nw + i * VR);
        esimd::simd<float, kM> pa, pb;
        {
          esimd::simd<uint32_t, 2 * kM> off;
#pragma unroll
          for (int r = 0; r < kM; ++r) {
            off[r] = static_cast<uint32_t>((prow(r) * ldab + hv) * sizeof(float));
            off[kM + r] = static_cast<uint32_t>((prow(r) * ldab + vh + hv) * sizeof(float));
          }
          esimd::simd<float, 2 * kM> pab = esimd::gather<float, 2 * kM>(abprev, off);
          pa = pab.template select<kM, 1>(0);
          pb = pab.template select<kM, 1>(kM);
        }
        const float al = static_cast<float>(alog[hv]), db = static_cast<float>(dtb[hv]);
        {
          esimd::simd<float, kAbW> acc[2 * kM];
#pragma unroll
          for (int j = 0; j < 2 * kM; ++j) acc[j] = 0.f;
          const bf16 *pwa = wa + static_cast<size_t>(hv) * H, *pwb = wb + static_cast<size_t>(hv) * H;
          if constexpr (ABI > 0) {  // every load first, then the FMAs in the loop's order
            esimd::simd<bf16, ABI * kAbW> la, lb, lx[NR];
            if constexpr (KM == 8) {  // a / b rows first, then the x rows (the 8-row core at 128 GRF 30 % faster)
#pragma unroll
              for (int j = 0; j < ABI; ++j) {
                const int k0 = i * kAbW + j * kItems * kAbW;
                la.template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(pwa + k0);
                lb.template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(pwb + k0);
              }
#pragma unroll
              for (int j = 0; j < ABI; ++j) {
                const int k0 = i * kAbW + j * kItems * kAbW;
#pragma unroll
                for (int r = 0; r < NR; ++r) lx[r].template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(x + (r0 + crow(r)) * H + k0);
              }
            } else {
#pragma unroll
              for (int j = 0; j < ABI; ++j) {
                const int k0 = i * kAbW + j * kItems * kAbW;
                la.template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(pwa + k0);
                lb.template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(pwb + k0);
#pragma unroll
                for (int r = 0; r < NR; ++r) lx[r].template select<kAbW, 1>(j * kAbW) = esimd::block_load<bf16, kAbW>(x + (r0 + crow(r)) * H + k0);
              }
            }
#pragma unroll
            for (int j = 0; j < ABI; ++j) {
              const esimd::simd<float, kAbW> va = la.template select<kAbW, 1>(j * kAbW), vb = lb.template select<kAbW, 1>(j * kAbW);
#pragma unroll
              for (int r = 0; r < NR; ++r) {
                const esimd::simd<float, kAbW> xv = lx[r].template select<kAbW, 1>(j * kAbW);
                acc[r] = xv * va + acc[r];
                acc[kM + r] = xv * vb + acc[kM + r];
              }
            }
          } else {
#pragma unroll 4
            for (int k0 = i * kAbW; k0 < H; k0 += kItems * kAbW) {
              const esimd::simd<float, kAbW> va = esimd::block_load<bf16, kAbW>(pwa + k0), vb = esimd::block_load<bf16, kAbW>(pwb + k0);
#pragma unroll
              for (int r = 0; r < NR; ++r) {
                const esimd::simd<float, kAbW> xv = esimd::block_load<bf16, kAbW>(x + (r0 + crow(r)) * H + k0);
                acc[r] = xv * va + acc[r];
                acc[kM + r] = xv * vb + acc[kM + r];
              }
            }
          }
          esimd::simd<float, 2 * kM> part;
#pragma unroll
          for (int j = 0; j < 2 * kM; ++j) part[j] = esimd::reduce<float>(acc[j], std::plus<>());
          esimd::slm_block_store<float, 2 * kM>(slm_off(kAbBase + i * 2 * kM), part);
        }
        const int A = acc0 < 0 ? 0 : (acc0 > kM ? kM : acc0);
        const int T = A + srows;
        const int last = commit_new ? T - 1 : A - 1;  // the row after which S and the window are committed (-1: before any)

        // TREE: each current row's parent, grandparent and great-grandparent (-1 / -2 / -3: the window entries)
        int te1[kM], te2[kM], te3[kM];
        if constexpr (TREE) {
#pragma unroll
          for (int j = 0; j < kM; ++j) {
            te1[j] = tree_parent(tw, j);
            te2[j] = te1[j] >= 0 ? tree_parent(tw, te1[j]) : te1[j] - 1;
            te3[j] = te2[j] >= 0 ? tree_parent(tw, te2[j]) : te2[j] - 1;
          }
        }
        // ---- 1: conv (own channels only: no barrier needed after staging) ----
#pragma unroll
        for (int G = 0; G < 3; ++G) {
          const bool owner = G == 2 || hv % rep == 0;
          esimd::simd<float, VR> w0 = win[G][0], w1 = win[G][1], w2 = win[G][2];
          auto commit = [&]() SYCL_ESIMD_FUNCTION {
            esimd::block_store<bf16, VR>(winW + ch0[G], esimd::simd<bf16, VR>(w0));
            esimd::block_store<bf16, VR>(winW + CC + ch0[G], esimd::simd<bf16, VR>(w1));
            esimd::block_store<bf16, VR>(winW + 2 * CC + ch0[G], esimd::simd<bf16, VR>(w2));
          };
          if (last < 0 && owner) commit();
          const bool tconv = TREE && (tw & kTreeFlag);  // a chain record runs the chain loop over every row
          for (int t = 0; t < (tconv ? A : T); ++t) {
            const int sr = t < A ? t : kM + t - A;
            const esimd::simd<float, VR> u = round_bf16<VR>(esimd::slm_block_load<float, VR>(slm_off((sr * 3 + G) * kD + i * VR)));
            esimd::simd<float, VR * kGdnTaps> wtt = wt[G];
            const esimd::simd<float, VR> c = conv4<VR>(wtt, w0, w1, w2, u);
            const esimd::simd<float, VR> o = round_bf16<VR>(c * sigmoid_v<VR>(c));
            esimd::slm_block_store<float, VR>(slm_off(kConvBase + (t * 3 + G) * kD + i * VR), o);
            w0 = w1;
            w1 = w2;
            w2 = u;
            if (t == last && owner) commit();
          }
          if (tconv) {
            // the current rows: taps = the inputs of the row's parent, grandparent, great-grandparent (their staged rows
            // in SLM, rounded on use as above), continuing into the window after the replayed rows (e -1 / -2 / -3 = its
            // newest / middle / oldest entry). Rows are independent here (no register chain).
            const esimd::simd<float, VR> c0 = w0, c1 = w1, c2 = w2;
            auto tap = [&](int e) SYCL_ESIMD_FUNCTION {
              esimd::simd<float, VR> v;
              if (e >= 0) v = round_bf16<VR>(esimd::slm_block_load<float, VR>(slm_off(((kM + e) * 3 + G) * kD + i * VR)));
              else v = e == -1 ? c2 : e == -2 ? c1 : c0;
              return v;
            };
#pragma unroll
            for (int j = 0; j < kM; ++j) {
              if (j >= srows) break;
              const esimd::simd<float, VR> u = round_bf16<VR>(esimd::slm_block_load<float, VR>(slm_off(((kM + j) * 3 + G) * kD + i * VR)));
              const int e1 = te1[j], e2 = te2[j], e3 = te3[j];
              const esimd::simd<float, VR> x1 = tap(e1), x2 = tap(e2), x3 = tap(e3);
              esimd::simd<float, VR * kGdnTaps> wtt = wt[G];
              const esimd::simd<float, VR> c = conv4<VR>(wtt, x3, x2, x1, u);
              const esimd::simd<float, VR> o = round_bf16<VR>(c * sigmoid_v<VR>(c));
              esimd::slm_block_store<float, VR>(slm_off(kConvBase + ((A + j) * 3 + G) * kD + i * VR), o);
              if (A + j == last && owner) {  // (a committing 1-row step: its window after row 0)
                w0 = x2, w1 = x1, w2 = u;
                commit();
              }
            }
          }
        }
        esimd::barrier();
        // ---- 1b: the step's k and q rows normalised once: item i < T takes k row i, item T + j the
        // q row of new row j; each writes its row back into the conv stage. This keeps the two 128-wide reductions and two
        // IEEE rsqrt per row off every item's dependent state chain; the same
        // expressions on the same values give the same bits.
        // A 1-row chain (T 1: a plain decode row) keeps its normalisation inline: the extra barrier costs more there. ----
        const bool pre = T > 1;
        if (pre) {
          const int nq = T - A;
          auto norm_row = [&](int u) SYCL_ESIMD_FUNCTION {
            const bool isk = u < T;
            const int t = isk ? u : A + (u - T);
            const uint32_t off = slm_off(kConvBase + (t * 3 + (isk ? 1 : 0)) * kD);
            const esimd::simd<float, kD> r = esimd::slm_block_load<float, kD>(off);
            const float f = rsqrt_ieee(esimd::reduce<float>(r * r, std::plus<>()) + 1e-6f);
            esimd::slm_block_store<float, kD>(off, isk ? esimd::simd<float, kD>(r * f) : esimd::simd<float, kD>(r * (f * scale)));
          };
          if constexpr (2 * kTMax <= kItems) {  // one row per item
            if (i < T + nq) norm_row(i);
          } else {  // the 16-row instance's up to 64 rows, two per item
            for (int u = i; u < T + nq; u += kItems) norm_row(u);
          }
          esimd::barrier();
        }

        // ---- 2: gates ----
        esimd::simd<float, 2 * kM> abn(0.f);  // new rows: a0..a3 | b0..b3 (bf16 values)
#pragma unroll
        for (int p = 0; p < kItems; ++p) abn += esimd::slm_block_load<float, 2 * kM>(slm_off(kAbBase + p * 2 * kM));
        abn = round_bf16<2 * kM>(abn);
        if (i == 0) {
#pragma unroll
          for (int r = 0; r < kM; ++r) {
            abcur[static_cast<size_t>(r) * ldab + hv] = abn[r];
            abcur[static_cast<size_t>(r) * ldab + vh + hv] = abn[kM + r];
          }
        }
        esimd::simd<float, kTMax> av(0.f), bv(0.f);
#pragma unroll
        for (int t = 0; t < kTMax; ++t) {
          const int tp = t < kM ? t : 0;
          const int tn = t - A < 0 ? 0 : (t - A > kM - 1 ? kM - 1 : t - A);
          av[t] = t < A ? static_cast<float>(pa[tp]) : static_cast<float>(abn[tn]);
          bv[t] = t < A ? static_cast<float>(pb[tp]) : static_cast<float>(abn[kM + tn]);
        }
        const esimd::simd<float, kTMax> xx = av + db;
        esimd::simd<float, kTMax> sp = esimd::log(esimd::simd<float, kTMax>(1.f) + esimd::exp(xx));
        sp.merge(xx, xx > 20.f);
        const float ea = esimd::exp(esimd::simd<float, 1>(al))[0];
        const esimd::simd<float, kTMax> decay = esimd::exp(sp * (-ea));
        const esimd::simd<float, kTMax> beta = sigmoid_v<kTMax>(bv);

        // ---- 3: state ----
        // TREE: row j's saved state (a branch row: a later row is its child) in tree_S, per (step row, head)
        auto tree_slot = [&](int j) SYCL_ESIMD_FUNCTION { return treeS + ((static_cast<size_t>(r0 + j) * vh + hv) * kD + i * VR) * kD; };
        const uint64_t aux = TREE ? rec_word(sgr, kSegTreeAux, kSegTreeAux2) : 0ull;  // side / save / restore rows (rows.hpp)
        esimd::simd<float, TREE ? VR * kD : 16> Shold;  // the holder's state around a side leaf
        const bool save_reg = kSaveSlm && ((aux >> 48) & 1u);  // a tree's only saved state: in SLM (kSaveBase)
        for (int t = 0; t < T; ++t) {
          bool side = false;  // a side leaf (its update in temporaries: the holder's state stays)
          if constexpr (TREE) {
            if (t >= A) {
              const int j = t - A;
              if ((aux >> (32 + j)) & 1u) {  // the parent's saved state back
                if (save_reg) {
#pragma unroll
                  for (int jj = 0; jj < VR; ++jj) Sb.template select<kD, 1>(jj * kD) = esimd::slm_block_load<float, kD>(slm_off(kSaveBase + (i * VR + jj) * kD));
                } else {
                  const float *src = tree_slot(tree_parent(tw, j));
#pragma unroll
                  for (int jj = 0; jj < VR; ++jj) Sb.template select<kD, 1>(jj * kD) = esimd::block_load<float, kD>(src + jj * kD);
                }
              }
              side = (aux >> j) & 1u;
            }
          }
          const esimd::simd<float, kD> kr = esimd::slm_block_load<float, kD>(slm_off(kConvBase + (t * 3 + 1) * kD));
          const esimd::simd<float, VR> vv = esimd::slm_block_load<float, VR>(slm_off(kConvBase + (t * 3 + 2) * kD + i * VR));
          esimd::simd<float, kD> kn = kr;  // normalised in 1b
          if (!pre) kn = kr * rsqrt_ieee(esimd::reduce<float>(kr * kr, std::plus<>()) + 1e-6f);
          const float dc = decay[t], bt = beta[t];
          // a side leaf runs the in-place code below (the same instructions: the loop is not unrolled, so its bits are the
          // chain's; a separate expression of the same math rounded differently under fast math) and puts the holder back
          if constexpr (TREE) {
            if (side) Shold = Sb;
          }
          Sb *= dc;
#pragma unroll
          for (int j = 0; j < VR; ++j) {
            esimd::simd<float, kD> row = Sb.template select<kD, 1>(j * kD);
            const float d = (vv[j] - esimd::reduce<float>(row * kn, std::plus<>())) * bt;
            Sb.template select<kD, 1>(j * kD) = kn * d + row;
          }
          if (t >= A) {
            const esimd::simd<float, kD> qr = esimd::slm_block_load<float, kD>(slm_off(kConvBase + (t * 3 + 0) * kD));
            esimd::simd<float, kD> qn = qr;  // normalised and scaled in 1b
            if (!pre) qn = qr * (rsqrt_ieee(esimd::reduce<float>(qr * qr, std::plus<>()) + 1e-6f) * scale);
            esimd::simd<float, VR> ov;
#pragma unroll
            for (int j = 0; j < VR; ++j) {
              esimd::simd<float, kD> row = Sb.template select<kD, 1>(j * kD);
              ov[j] = esimd::reduce<float>(row * qn, std::plus<>());
            }
            esimd::slm_block_store<float, VR>(slm_off(kOBase + (t - A) * kD + i * VR), round_bf16<VR>(ov));
          }
          if (t == last) {
#pragma unroll
            for (int j = 0; j < VR; ++j) {
              // the committed S bypasses L1 and L2 (uncached; LSC hint stores take <= 64 elements):
              // nothing reads it before the next step, and the 1.5 MB per layer does not drain through L2 into the next
              // GEMM's weight stream. Same bits.
#pragma unroll
              for (int h = 0; h < kD; h += 64)
                esimd::block_store<float, 64>(Sp + j * kD + h, Sb.template select<64, 1>(j * kD + h),
                                              esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::uncached>, esimd::cache_hint_L2<esimd::cache_hint::uncached>});
            }
          }
          if constexpr (TREE) {
            if (side) Sb = Shold;
          }
          if constexpr (TREE) {
            if (t >= A && ((aux >> (16 + t - A)) & 1u)) {  // a later in-place row's parent: its state for it
              const int j = t - A;
              if (save_reg) {
#pragma unroll
                for (int jj = 0; jj < VR; ++jj) esimd::slm_block_store<float, kD>(slm_off(kSaveBase + (i * VR + jj) * kD), Sb.template select<kD, 1>(jj * kD));
              } else {
                float *dst = tree_slot(j);
#pragma unroll
                for (int jj = 0; jj < VR; ++jj) esimd::block_store<float, kD>(dst + jj * kD, Sb.template select<kD, 1>(jj * kD));
              }
            }
          }
        }
        esimd::barrier();

        // ---- 4: gated norm ----
#pragma unroll
        for (int r = 0; r < NR; ++r) {
          if (r >= srows) break;
          const esimd::simd<float, kD> o = esimd::slm_block_load<float, kD>(slm_off(kOBase + r * kD));
          const float rr = rsqrt_ieee(esimd::reduce<float>(o * o, std::plus<>()) / static_cast<float>(kD) + eps);
          const esimd::simd<float, VR> om = esimd::slm_block_load<float, VR>(slm_off(kOBase + r * kD + i * VR));
          const esimd::simd<float, VR> zb = round_bf16<VR>(zr[r]);
          esimd::simd<float, VR> gate = sigmoid_v<VR>(zb);
          if constexpr (SWISH) gate = zb * gate;  // silu(z) = z sigmoid(z)
          const esimd::simd<float, VR> y = om * rr * wn * gate;
          esimd::block_store<bf16, VR>(core_out + static_cast<size_t>(r0 + r) * VW + hv * kD + i * VR, esimd::simd<bf16, VR>(y));
        }
        // ---- 5: several sequences: this step's q | k | v rows into the slot's log (from the SLM stage;
        // last, so no store waits on a load) ----
        if (segm) {
#pragma unroll
          for (int G = 0; G < 3; ++G) {
            if (!(G == 2 || hv % rep == 0)) continue;  // q / k: the first value head of each key head keeps them
            for (int r = 0; r < srows; ++r)
              esimd::block_store<float, VR>(ylog + static_cast<size_t>(r) * ldy + ch0[G],
                                            esimd::slm_block_load<float, VR>(slm_off(((kM + r) * 3 + G) * kD + i * VR)));
          }
        }
      });
}

// The core's compile-time shape for this call: NR / NP 1 or kM, ABI = hidden / 512 when it is 8 or 10.
template <bool SWISH, int NR, int NP, int KM, bool TREE = false, int VR = kVR, int LIB = 0>
sycl::event core_abi(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                     bf16 *core_out) {
  constexpr int kPerIter = (kD / VR) * 16;  // a / b elements per iteration of the work-group
  if constexpr (VR == kVR) {
    if (s.hidden == 10 * kPerIter) return core<VR, SWISH, NR, NP, 10, KM, TREE, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
    if (s.hidden == 8 * kPerIter) return core<VR, SWISH, NR, NP, 8, KM, TREE, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
  }
  return core<VR, SWISH, NR, NP, 0, KM, TREE, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
}
template <bool SWISH, int KM, int VR = kVR, int LIB = 0>
sycl::event core_launch_km(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                           bf16 *core_out) {
  const int sr = st.seg ? seg_size(rows, st.seg_rows) : rows;
  const bool r1 = sr == 1, p1 = st.accept_max <= 1;
  if (r1 && p1) return core_abi<SWISH, 1, 1, KM, false, VR, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
  if (r1) return core_abi<SWISH, 1, KM, KM, false, VR, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
  if (p1) return core_abi<SWISH, KM, 1, KM, false, VR, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
  return core_abi<SWISH, KM, KM, KM, false, VR, LIB>(q, s, rows, x, w, st, parity, ystep, core_out);
}

// the 8-row shapes from the 128-GRF library (gdn_core_v2.cpp; XEN_GDN_CORE128=0: off). VR is a template
// parameter; a VR 2 build (64 items) miscompiled (wrong outputs, intact state), so it is not built
#define XEN_GDN_V2_DECL(name)                                                                                                         \
  template <bool SWISH>                                                                                                               \
  sycl::event name(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, \
                   const float *ystep, bf16 *core_out);
XEN_GDN_V2_DECL(core_launch8_v2)
XEN_GDN_V2_DECL(core_launch8_tree_v2)
#undef XEN_GDN_V2_DECL
// The 8-row instances (gdn_core8.cpp, 256 GRF).
template <bool SWISH>
sycl::event core_launch8(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity, const float *ystep,
                         bf16 *core_out);
// The tree instances (8-row segments and log, every row and replay bound; gdn_core8.cpp)
template <bool SWISH>
sycl::event core_launch8_tree(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                              const float *ystep, bf16 *core_out);
// The 16-row tree instance (segments or replays past 8 rows: a 16-row tree, the step after one;
// records, a 16-row log; chains run its chain loop; gdn_core16.cpp)
template <bool SWISH>
sycl::event core_launch16_tree(sycl::queue &q, const GdnShape &s, int rows, const bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                               const float *ystep, bf16 *core_out);

}  // namespace xenithium:kernels:gdn_detail
