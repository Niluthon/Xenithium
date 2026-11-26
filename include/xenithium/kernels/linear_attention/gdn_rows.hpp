// Xenithium · Gated DeltaNet over many rows (prefill): gdn.hpp's semantics and rounding
// points for m consecutive rows of one sequence, one card's heads.
// 1 qkvz pf_dense -> raw fp32 [m][qkvz.n]
// 2 ab per (row, head): the a / b dots over hidden (bf16 weights), bf16 values; a sub-group
// takes 4 heads x 4 rows (x 1 row under 256 rows), each dot in the one-dot order
// 3+4 conv one launch: per (16 rows, q / k head) the conv slides over the rows in
// + prep registers, then l2-normalised q (x k_dim^-0.5) and k -> qn / kn, fp32; per (16 rows,
// + gate 16 v channels) the conv -> cv; per (row, value head) decay = exp(-exp(A_log)
// softplus(a + dt_bias)), beta = sigmoid(b). The conv reads the window, replayed rows
// (the previous decode step's pending log, `accept` of them), then the m new ones;
// c = bf16(silu(taps)). Then the final window -> win_out
// 5 state one work-group per value head, item i owns value rows [4 i, 4 i + 4) of S: rows in
// order, S *= decay; d = (v - S k) beta; S += d k^T; o = bf16(S q) for new rows. No
// item reads another's S rows, so the loop has no barrier. S written once at the end.
// With GdnRowsState:pos >= 0, full chunks of 64 rows at absolute positions
// that are multiples of 64 run on the chunked DPAS scan (gdn_scan.hpp); the rows before
// the first boundary and after the last full chunk stay on this row core.
// 6 norm per (row, value head): gated RMSNorm -> y bf16 [m][val_w]
// (on the chunked scan the scan's rows are normed inside the scan, gdn_scan.hpp;
// this step then covers only the row core's lead / trail rows)
// 7 out pf_dense (int8 [hidden][val_w]) -> fp32 partials, then the two-card sum
// Afterwards the state is committed through the last row: the next decode step must replay
// nothing (accept 0) and read the window from win_out.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/linear_attention/gdn_scan.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/kernels/stage_trace.hpp"

namespace xenithium::kernels {

// The stages of a GDN prefill call in launch order (GdnRowsScratch:trace): projections = qkvz + the a | b dots, prep =
// the conv / prep / gate launch and the windows, scan pre = the chunked scan's per-chunk precompute (gdn_scan.hpp: T^-1,
// Uv, Wk, Qg, Kt, M), state = the recurrent core (row core and the chunked scan's serial part, with the scan's folded
// norm), norm = the row core rows' gated norm, out = the out projection, the split's send and the sum.
enum class GdnStage : int { projections, prep, scan_pre, state, norm, out, kCount };
const char *gdn_stage_name(GdnStage s);
using GdnTrace = StageTrace<static_cast<int>(GdnStage::kCount)>;

struct GdnRowsScratch {
  float *yq = nullptr;  // [rows_max][qkvz.n]
  float *ab = nullptr;  // [rows_max][ab_n]
  float *cv = nullptr;  // [rows_max + 4][conv_ch] conv outputs (bf16 values); only the v columns are written
  float *qn = nullptr;  // [rows_max + 4][key_w]
  float *kn = nullptr;  // [rows_max + 4][key_w]
  float *gate = nullptr;  // [rows_max + 4][2 vh]: decay | beta
  float *lgate = nullptr;  // [rows_max + 4][vh]: ln decay (the chunked scan's g)
  gdn_bf16 *o = nullptr;  // [rows_max][val_w]
  gdn_bf16 *y = nullptr;  // [rows_max][val_w]
  float *partial = nullptr;  // [rows_max][hidden]
  GdnScanScratch scan;  // gdn_scan_sizes(shape, rows_max)
  PfA8 a8;  // set = the qkvz and out GEMMs run W8A8 (pf_dense.hpp)
  bool qkvz_ready = false;  // yq already holds x's qkvz projection (the caller ran it in row parts)
  bool out_later = false;  // gdn_rows stops with the gated rows in y; the caller runs gdn_rows_out
  // The projections' input prepared by the rows' owners (four cards' rows by owner): x8 = x quantised per row for the
  // W8A8 qkvz (PfA8::ready, row: q / s at this call's rows; pf_a8_quantize), ab_in = the a | b dots of this card's heads
  // (gdn_rows_ab's [a | b] per row, the call's rows at row stride ab_ld). With both set gdn_rows reads x for neither.
  PfA8 x8{};
  const float *ab_in = nullptr;
  int ab_ld = 0;
  GdnTrace *trace = nullptr;  // prefill profiling: every launch's event by stage (stage_trace.hpp); null = none
};
struct GdnRowsSizes {
  size_t yq, ab, cv, qk, gate, lgate, o, partial;
};
GdnRowsSizes gdn_rows_sizes(const GdnShape &s, int qkvz_n, int rows_max);

// Step 2 of gdn_rows alone: the a | b dots of shape s's heads (s.card's) over m rows of x into ab (row stride ld >=
// s.ab_n(), [a | b] per row), the bits gdn_rows writes into its scratch. The rows' owner runs it with every card's
// weights for the GdnRowsScratch::ab_in of the others.
sycl::event gdn_rows_ab(sycl::queue &q, const GdnShape &s, int m, const gdn_bf16 *x, const GdnWeights &w, float *ab, int ld);

// x bf16 [m][hidden]; out bf16 [m][hidden]. The state's committed S and window win_in, plus the
// pending log (log_y / log_ab rows [0, *accept)) of the previous decode step, are the history.
struct GdnRowsState {
  float *S = nullptr;
  const gdn_bf16 *win_in = nullptr;
  gdn_bf16 *win_out = nullptr;  // may equal win_in
  const float *log_y = nullptr, *log_ab = nullptr;
  const int32_t *accept = nullptr;  // device word (0..4)
  // The absolute position of the first new row. >= 0: full 64-row chunks at absolute
  // multiples of 64 run on the chunked scan (GdnRowsScratch:scan); < 0: every row on the row core.
  int pos = -1;
  // Null, or the call's first slice record on the device (rows.hpp): the kernels then read the position (kSegPos, for
  // pos >= 0 the chunk split), the slot (kSegSlot) and the GDN parity (kSegParity) from it and take the state from
  // bufs as a step's sequence does (S and its window, pending log of that slot; win_in = win[parity ^ 1], win_out =
  // win[parity], log_y / log_ab = y / ab[parity ^ 1]): S, win_in, win_out, log_y and log_ab are then unused. The same
  // bits; one recorded launch serves every position and slot. Not with snap_pos.
  const int32_t *rec = nullptr;
  GdnState bufs;
  // With snap_pos (an absolute multiple of 64 inside the call, on the chunked scan: pos >= 0)
  // the state at snap_pos also goes to snap_S (S, gdn_S_elems floats) and snap_win (the window there, gdn_win_elems), the
  // same bits a call ending at snap_pos leaves in S / win_out; the call's own outputs are unchanged.
  int snap_pos = -1;
  float *snap_S = nullptr;
  gdn_bf16 *snap_win = nullptr;
};
GdnEvents gdn_rows(sycl::queue &q, const GdnShape &s, int m, const gdn_bf16 *x, const GdnWeights &w, const GdnRowsState &st, const GdnRowsScratch &sc,
                   const PairRowsWire *wire, int wire_parity, gdn_bf16 *out);
// gdn_rows' last stage alone: the out projection of sc.y's m rows into sc.partial and the
// two-card sum (or, wire->split, the partial left for the caller). One prefill call that carries the rows of several
// sequences runs gdn_rows per sequence with sc.out_later (its own state, its rows of yq / y at their offset, the other
// scratch shared: the calls run in order on the queue) and this once over all rows. A row's bits are gdn_rows'.
GdnEvents gdn_rows_out(sycl::queue &q, const GdnShape &s, int m, const GdnWeights &w, const GdnRowsScratch &sc, const PairRowsWire *wire, int wire_parity,
                       gdn_bf16 *out);

}  // namespace xenithium:kernels
