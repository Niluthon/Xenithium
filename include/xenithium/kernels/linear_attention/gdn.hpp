// Xenithium · Gated DeltaNet (linear attention) decode block, one card's heads.
// Semantics (qwen4_exp `linear_attention` layers, from config.json and the checkpoint), per token, per value head:
// [q|k|v|z] = bf16(x. Wqkvz^T) q, k: key heads x 128; v, z: value heads x 128
// [a|b] = bf16(x. Wab^T) one of each per value head
// conv c = bf16(silu(sum_i w[ch][i] u[t-3+i])) per channel of q|k|v, taps oldest first,
// over the last 3 inputs (the window, zero at the start) and this one
// q, k l2-normalised: q / sqrt(sum q^2 + 1e-6); q *= k_dim^-0.5
// g = -exp(A_log) softplus(a + dt_bias) (softplus(x) = x above 20); beta = sigmoid(b)
// S fp32 [V][K] per value head (value head h reads key head h / (v_heads / k_heads)):
// S *= exp(g); d = (v - S k) beta; S += d k^T; o = bf16(S q)
// y bf16(o rsqrt(mean(o^2) + eps) w_norm gate(z)) gate = sigmoid (qwen38, output_gate_type
// sigmoid) or silu = z sigmoid(z) (qwen35, "swish"; GdnShape:swish)
// out y. Wout^T; over C cards each card holds 1/C of the heads and of out_proj's k:
// fp32 partials, two-card sum (wire.hpp), one final bf16.
// Rows. A step carries R <= 4 consecutive tokens of ONE sequence (the token + drafts). Rows are
// speculative until the next step says how many were accepted (`*accept, a device word): the
// step's projections ARE its pending log (y[parity], ab[parity]), and the next step first
// replays the accepted rows from that log into the committed state, then runs its own rows
// on top in registers. Committed state is written once per step, nothing is snapshotted, and
// "R rows, then accept c" equals "c rows, then the next step" bit for bit (the same code
// replays the same values). Plain decode: R = 1 and accept = 1 every step after the first.
// parity: alternate per step (all layers of a step use the same one); win and y/ab are
// double-buffered by it, so the heads that share a key head's conv window never
// read a window another work-group is rewriting.
// Launches: qkvz GEMM (int8), core (a / b dots + conv + gates + recurrence + gated norm, one
// work-group per value head), out GEMM (int8, fp32 partials), pair_reduce. a / b are two dots
// per head over 0.25 MB of weights: the core reads them raw instead of through a GEMM launch.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/comm/wire.hpp"

namespace xenithium::kernels {

using gdn_bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr int kGdnRows = 4;  // rows per sequence in a step (a step: up to kMaxRows in segments of <= 4)
// (the DFlash 2 drafter's 8-row verify): a state whose pending logs hold 8 rows (GdnState:log_rows)
// takes segments of up to 8 rows (the 8-row core instances, compiled at 256 GRF).
inline constexpr int kGdnMaxRows = 8;
// One sequence's 16-row draft tree (DSpine's whole block): a state whose
// pending logs hold 16 rows takes segments of up to 16 rows and replays of up to 16 (the 16-row tree core, records only).
inline constexpr int kGdnWideRows = 16;
inline constexpr int kGdnDim = 128;  // k_dim == v_dim (kernel constant)
inline constexpr int kGdnTaps = 4;  // conv taps (window of 3)

struct GdnShape {
  int hidden = 0, k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;  // the whole model
  int cards = 1, card = 0;
  float eps = 1e-6f;
  bool swish = false;  // output gate silu(z) instead of sigmoid(z) (qwen35)
  int kh() const { return k_heads / cards; }  // this card's heads
  int vh() const { return v_heads / cards; }
  int rep() const { return v_heads / k_heads; }
  int key_w() const { return kh() * k_dim; }
  int val_w() const { return vh() * v_dim; }
  int conv_ch() const { return 2 * key_w() + val_w(); }  // q | k | v
  int qkvz_n() const { return conv_ch() + val_w(); }  // q | k | v | z
  int ab_n() const { return 2 * vh(); }  // a | b
};

struct GdnWeights {
  DenseWeights qkvz;  // int8 [q|k|v|z][hidden] of this card's heads (W4: only n / k are read, as the strides)
  const gdn_bf16 *a = nullptr, *b = nullptr;  // in_proj_a / in_proj_b rows of this card, bf16 [vh][hidden]
  DenseWeights out;  // int8 [hidden][val_w]: this card's k slice of out_proj (W4: n / k only)
  // qwen35: the same two matrices as W4 (gate on qkvz4.w); the decode launches w4a16_gemm_rows for them.
  W4Mat qkvz4, out4;
  const gdn_bf16 *conv = nullptr;  // [conv_ch][4]
  const gdn_bf16 *a_log = nullptr, *dt_bias = nullptr;  // [vh]
  const gdn_bf16 *norm = nullptr;  // [v_dim]
};

// One sequence's state for one layer on this card (device). Zeroed = empty history.
// Several sequences (rows.hpp): with `seg` set, the step's rows are segments of
// seg_rows rows, each a sequence on its own slot; every array below then holds one copy per slot
// (slot k at k x its slot_* stride, in elements), and each segment's record gives its accept
// count, its parity and its slot. The step's projections then go to GdnScratch:y and the core
// keeps each segment's rows in its slot's pending log (y / ab by the sequence's parity), so a
// sequence may skip steps: its log waits for its next step. Without `seg`: one sequence, the
// host parity, `accept, and y[parity] is both the projections and the log.
struct GdnState {
  float *S = nullptr;  // committed [vh][v_dim][k_dim]
  gdn_bf16 *win[2] = {};  // committed conv window [3][conv_ch], oldest first, by parity
  float *y[2] = {};  // [log_rows][qkvz.n]: the step's projections = its pending rows
  float *ab[2] = {};  // [log_rows][ab_n]: the step's a | b (bf16 values), its pending rows
  const int32_t *accept = nullptr;  // rows of the previous step to commit (0..log_rows)
  int log_rows = kGdnRows;  // rows the pending logs hold = the largest segment (4, kGdnMaxRows or kGdnWideRows)
  const int32_t *seg = nullptr;  // segment records (kSegRec ints each); null = one sequence
  int seg_rows = 0;  // rows per segment (0 = all rows)
  size_t slot_S = 0, slot_win = 0, slot_y = 0, slot_ab = 0;  // per-slot strides (elements)
  // the host's bound on every accept word of the step (the core stages that many of the previous
  // step's rows; 4 = no bound), and whether this step's rows are final (a 1-row step: S and the window are committed after
  // the last row, so the sequence's next step must use accept 0).
  int accept_max = 4;
  bool commit = false;
  // Draft trees (DDTree): the segments' records may carry a tree (kSegTree) or a path mask of the
  // previous step's accepted rows (kSegPath): then tree is set (the tree core: 8-row log, records) and tree_S holds
  // gdn_tree_S_elems floats of scratch (the states of the rows whose subtrees are not done; any layer may reuse it).
  // Row j of a tree computes what a chain of its ancestors computes, bit for bit (the same ops on its parent's state).
  bool tree = false;
  float *tree_S = nullptr;
};
size_t gdn_S_elems(const GdnShape &s);
size_t gdn_tree_S_elems(const GdnShape &s);  // GdnState:tree_S (kMaxRows saved states of this card's heads)
size_t gdn_win_elems(const GdnShape &s);
// Zero S and both windows (a fresh sequence; set *accept = 0 for its first step).
sycl::event gdn_state_reset(sycl::queue &q, const GdnShape &s, const GdnState &st);

struct GdnScratch {
  gdn_bf16 *core = nullptr;  // [8][val_w] (the row shape's rows)
  float *partial = nullptr;  // [8][hidden]
  float *y = nullptr;  // [8][qkvz.n]: the step's projections (several sequences only)
};

struct GdnEvents {
  sycl::event qkvz, core, out, reduce;
  sycl::event send;  // the split's peer copy (gdn_rows with a send)
};

// x bf16 [shape][hidden] (row_shape(rows) rows readable); out bf16 [shape][hidden] (rows < `rows`
// written). More than 4 rows only as segments (seg set) of <= 4 rows each.
// wire: null with one card. wire_parity alternates per call on the wire's inbox. ksplits: 0 =
// the defaults (qkvz: dense_auto_ksplit; fix it when comparing different card counts bit for bit).
GdnEvents gdn_decode(sycl::queue &q, const GdnShape &s, int rows, const gdn_bf16 *x, const GdnWeights &w, const GdnState &st, int parity,
                     const GdnScratch &sc, const PairWire *wire, int wire_parity, gdn_bf16 *out, int qkvz_ks = 0, int out_ks = 0);

// ---- host reference (one card's heads; the whole model with cards = 1) -------------------------
struct GdnRefWeights {
  std::vector<float> qkvz;  // [qkvz_n][hidden] dequantised
  std::vector<float> ab;  // [ab_n][hidden]
  std::vector<float> out;  // [hidden][val_w] dequantised
  std::vector<float> conv;  // [conv_ch][4]
  std::vector<float> a_log, dt_bias, norm;
};
struct GdnRefState {
  std::vector<double> S;  // [vh][v_dim][k_dim]
  std::vector<float> win;  // [3][conv_ch]
  void reset(const GdnShape &s);
};
// x [hidden] -> bf16 values of [q|k|v|z] and [a|b].
void gdn_ref_proj(const GdnShape &s, const GdnRefWeights &w, const float *x, float *qkvz, float *ab);
// One token through conv, gates, recurrence and gated norm from bf16 projection values;
// advances st. o, y: bf16 values [val_w].
void gdn_ref_core(const GdnShape &s, const GdnRefWeights &w, GdnRefState &st, const float *qkvz, const float *ab, float *o, float *y);
// Run the recurrence and norm from an already rounded convolution row [conv_ch] and z [val_w].
void gdn_ref_recur(const GdnShape &s, const GdnRefWeights &w, GdnRefState &st, const float *c, const float *z,
                   const float *ab, float *o, float *y);
// This card's out_proj partial [hidden] from y [val_w].
void gdn_ref_out(const GdnShape &s, const GdnRefWeights &w, const float *y, double *part);

}  // namespace xenithium:kernels
