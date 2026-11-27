// Xenithium · PLE (per-layer n-gram embedding) decode block, every card on its own copy.
//
// Semantics (qwen4_exp PLE layer, from config.json and the checkpoint),
// per token t at position p, the residual's S streams of H after the pending combine:
//   e         = the token's n-gram rows (heads x head_dim = E), dequantised bf16   (host: ple_table)
//   [key|v]   = bf16(e . [Wkey | Wvalue]^T)        key: S x H, v: H (one value for every stream)
//   per s:    kn = bf16(rms(key_s) (1 + nk)), qn = bf16(rms(res_s) (1 + nq))
//             d = bf16(bf16(sum bf16(kn qn)) / sqrt(H)); g = bf16(sigmoid(sign(d) bf16(sqrt(max(|d|, 1e-6)))))
//             gated_s = bf16(g v); x_s = bf16(rms(gated_s) (1 + ncw))     (the conv input)
//   conv      c = bf16(sum_k w[ch][k] x[p - 9 + 3k]) per channel (kernel 4, dilation 3; zero before 0)
//   res'      = bf16(res + bf16(gated + bf16(silu(c))))
// (kn, qn, x use the Gemma (1 + w) affine; the host reference is ple_ref.hpp.)
//
// Rows are consecutive tokens of one sequence at pos .. pos + rows - 1 (pos a device word). The
// conv inputs live in a ring by position % kPleRing (32), so speculative rows need no accept word: a rejected
// row's input is rewritten when the real token at that position comes (as the QSA caches). Every
// row reads positions <= its own, all written by the gate launch before the conv launch reads.
//
// Launches: kv GEMM (int8, dense_gemm_m4), gate (one work-group per (row, stream)), conv (one item
// per (row, channel)); the step's rows come from the host (pinned) first. The block is replicated
// on every card: the cards compute bit-identical residuals.
#pragma once

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/embed/ple_ref.hpp"

namespace xenithium::kernels {

using ple_bf16 = sycl::ext::oneapi::bfloat16;

struct PleShape {
  int hidden = 2560, streams = 4, emb = 2560;
  float eps = 1e-6f;
  int width() const { return streams * hidden; }  // residual channels
  int kv_n() const { return width() + hidden; }    // key rows then value rows
};

struct PleWeights {
  DenseWeights kv;                                     // [S*H + H][E] (key_proj | value_proj)
  const ple_bf16 *nk = nullptr, *nq = nullptr, *ncw = nullptr;  // [S*H] each
  const ple_bf16 *conv = nullptr;                      // [S*H][4]
};

// One sequence's conv history on one card: [16][S*H] by position % 16. Needs no reset: positions
// before 0 read as zero and every later position is written before it is read. Several sequences
// one ring per slot, slot_elems apart.
struct PleState {
  ple_bf16 *ring = nullptr;
  size_t slot_elems = 0;
};
size_t ple_ring_elems(const PleShape &s);

struct PleScratch {
  ple_bf16 *emb = nullptr;  // [8][E] the step's rows (device; the row shape's rows are read)
  float *kv = nullptr;      // [8][kv_n]
  ple_bf16 *gated = nullptr;  // [8][S*H]
};

struct PleEvents {
  sycl::event up, kv, gate, conv;
};

// emb_host: the step's rows (pinned host bf16 [rows][E]) copied into sc.emb first; null = sc.emb
// already holds them. res: the residual [rows][S*H], updated in place for rows < `rows` (1..8). pos: a bare
// position word, or segment records (rows.hpp) rec_stride ints apart, seg_rows rows each: row r then
// takes its segment's position and slot (its ring).
// tree: some segment's records carry a draft tree (rows.hpp kSegTree): its row j takes the taps of
// its virtual position p0 + depth(j), those at t >= p0 from its ancestor's row slot p0 + anc(j, t - p0); every row's
// conv input is written at its own slot (p0 + j). false = chains only.
PleEvents ple_decode(sycl::queue &q, const PleShape &s, int rows, const ple_bf16 *emb_host, const PleWeights &w, const PleState &st,
                     const int32_t *pos, const PleScratch &sc, ple_bf16 *res, int kv_ks = 0, int rec_stride = 0, int seg_rows = 0,
                     bool tree = false);
// After the accept of a tree segment's path (from[0] = 0 < from[1] < ..., n accepted rows, slot
// `slot`, first position p0), row from[x]'s conv input moves to position p0 + x (ring slot (p0 + x) % 32) for each x
// with from[x] != x. A path of rows 0 .. n - 1 launches nothing (the returned event is empty then).
sycl::event ple_move(sycl::queue &q, const PleShape &s, const PleState &st, int slot, int p0, const int *from, int n);

// The step's PLE rows arrive while the step runs. The host submits the step with *want (a pinned
// word) set to its sequence number, reads the rows from the drive into emb_host, then stores *word = that number.
// ple_wait_rows (before ple_decode, which then gets emb_host = null) spins on *word (uncached loads, a system fence),
// then copies the rows to emb_dev. After wait_cycles GPU timestamp cycles it gives up: *error = 1 (pinned; the host
// checks it), the rows copied as they are.
struct PleWait {
  const uint32_t *want = nullptr;  // pinned: the step's number (written before the submission)
  const uint32_t *word = nullptr;  // pinned: the published number (written after the rows)
  uint32_t *error = nullptr;       // pinned
  uint64_t wait_cycles = 2800000000ull;  // >= 1 s at 2.8 GHz
};
// rows x emb bf16 elements, emb a multiple of 128.
sycl::event ple_wait_rows(sycl::queue &q, const PleWait &w, int rows, int emb, const ple_bf16 *emb_host, ple_bf16 *emb_dev);

// ---- many rows (prefill) -------------------------------------------------------------------
// m consecutive rows at host position pos0 (any m): the kv GEMM on pf_dense, the gate as decode but
// into a chunk buffer of conv inputs (the ring is too short for a chunk, and the gate would overwrite
// ring entries the chunk's first rows still read), the conv reading the chunk buffer or (before pos0)
// the ring, then the chunk's last 16 positions go into the ring for the next steps.
struct PleRowsScratch {
  ple_bf16 *emb = nullptr;    // [rows_max][E] (device)
  float *kv = nullptr;        // [rows_max][kv.n]
  ple_bf16 *gated = nullptr;  // [rows_max][S*H]
  ple_bf16 *x = nullptr;      // [rows_max][S*H] conv inputs
  PfA8 a8;                    // set = the kv GEMM runs W8A8 (pf_dense.hpp)
};
// rec: null (st.ring is the sequence's ring), or the call's first slice record on the device (rows.hpp): the kernels
// then read the position (kSegPos, instead of pos0) and the slot (kSegSlot: its ring, st.slot_elems apart from st.ring)
// from it, so one recorded launch serves every position and slot. The same bits.
PleEvents ple_rows(sycl::queue &q, const PleShape &s, int m, int64_t pos0, const ple_bf16 *emb_host, const PleWeights &w, const PleState &st,
                   const PleRowsScratch &sc, ple_bf16 *res, const int32_t *rec = nullptr);
// ple_rows in two parts on the chunk's rows [lo, hi) (rows by owner: a card runs its own rows; the scratch and res rows
// at their places in the chunk, the same bits as ple_rows' rows). ple_rows_gate: the rows' host rows up, the kv GEMM, the
// gate (conv inputs into sc.x). ple_rows_conv: the conv of rows [lo, hi), which reads the kPleDil * (kPleTaps - 1) = 9
// conv inputs before each row (sc.x rows of the chunk, or the ring before pos0): rows another card gated must be in this
// card's sc.x first. ring_tail: then the chunk's last kPleRing rows of sc.x (all present) into the ring (ev.kv: that
// launch). m: the chunk's rows.
PleEvents ple_rows_gate(sycl::queue &q, const PleShape &s, int lo, int hi, const ple_bf16 *emb_host, const PleWeights &w, const PleRowsScratch &sc,
                        const ple_bf16 *res);
PleEvents ple_rows_conv(sycl::queue &q, const PleShape &s, int lo, int hi, int m, int64_t pos0, const PleWeights &w, const PleState &st,
                        const PleRowsScratch &sc, ple_bf16 *res, bool ring_tail, const int32_t *rec = nullptr);

}  // namespace xenithium::kernels
