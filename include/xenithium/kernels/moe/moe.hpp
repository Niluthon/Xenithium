// Xenithium · MoE decode block: router, top-k, routed experts (int4 GPTQ), shared expert
// (int8 dense), weighted combine and the two-card all-reduce, for one decode step of <= 8 rows
// (two row shapes, rows.hpp: 1-4 rows run the 4-row code, 5-8 the 8-row code;).
// Semantics (Qwen3-Next sparse MoE, re-derived from the model definition):
// logits = bf16(x. Wr^T) router, bf16 weights, [experts] per row
// top-k by logit (ties -> lowest expert id); weights = softmax over the k selected logits
// (equal to softmax over all then renormalised over the top k)
// expert e: g, u = bf16(x. Wg^T), bf16(x. Wu^T); h = bf16(silu(g) * u); y_e = h. Wd^T
// shared: the same MLP with int8 weights, scaled by sigmoid(bf16(x. w_sg))
// out = bf16(sum_k w_k y_k + sigmoid(sg) y_shared)
// Rounding points of this engine: g, u and h are bf16; y_e, the weighted sum and the shared
// term stay fp32 until the single final bf16.
// Expert parallel over C cards (C = 1 or 2): card c owns experts [c E/C, (c+1) E/C) and half of
// the shared expert's intermediate columns. Both cards hold the same x, run the same router
// and the same top-k (replicated weights, fixed summation orders), compute their own part
// p_c = sum over own routed experts (ascending id) w y + sigmoid(sg) y_shared,c
// and exchange it: out = bf16(p_0 + p_1) on both cards (fp32 addition commutes, so the cards
// agree bit for bit).
// Five launches, every grid fixed by (rows, shape) so a step can be recorded once:
// 1 router dense_gemm_m4 over [router | shared gate] (one bf16 matrix, n = E + 1 padded)
// 2 route one work-group: per row top-k + weights; then this card's unique-expert list
// (ascending expert id) with row masks and weights -> MoeRoute
// 3 gate_up per (unique slot, 16-column tile): gate and up on separate work-items (x KS),
// h = bf16(silu(bf16 g) bf16 u); the card's shared-expert part is slot `slots, its work-groups first
// in the grid; a routed slot computes its own rows only, packed into 8-row blocks,
// and stores those rows (h and the partial keep the [slot][row] layout)
// 4 down per (slot, 16-column output tile): y = h. Wd^T (fp32) -> partial
// 5 combine per (row, 64 columns): p_c in fixed order; with two cards the words go into the
// peer's mailbox and the peer's are awaited (time-bounded); out = bf16(p_0 + p_1)
// Expert blobs are read through a per-card pointer table: a hot expert points into VRAM, a
// cold one into pinned host memory (read in place over PCIe).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <map>
#include <utility>
#include <vector>

#include "xenithium/kernels/comm/wire.hpp"  // WireQuad
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/moe/moe_cpu.hpp"

namespace xenithium::kernels {

using moe_bf16 = sycl::ext::oneapi::bfloat16;

// Rows a step can carry (the capacity of the route record and the pair / quad mailboxes; a launch's shape is
// row_shape(rows)). The affine MoE keeps its 16-row layout (kStepRows16). The scratch is sized by MoeShape:rows_cap,
// not by this.
inline constexpr int kMoeRows = kMaxShape;
inline constexpr int kMoeMaxTopk = 16;
inline constexpr int kMoeMaxExperts = 1024;
inline constexpr int kMoeMaxSlots = kMoeRows * kMoeMaxTopk;  // unique experts a card can get

struct MoeShape {
  int hidden = 0, inter = 0, experts = 0, topk = 0, shared_inter = 0;
  int cards = 1, card = 0;
  // Numerics switch: round each routed expert's down output y_e to bf16 before the weighted sum. Off = fp32 y_e (the
  // default). Under tp each card rounds its own partial of y_e (its half of the down k), exactly as moe_rows does.
  bool down_bf16 = false;
  // Card balance. tp: no expert ownership; every card computes its 1/cards of every
  // routed expert's intermediate columns (gate_up columns, down k), from per-card half blobs
  // (moe_tp_layout / moe_split_tp). Balanced by construction; each card takes a fixed 1/cards of
  // the shared expert. Off = EP: card c owns experts [c E/C, (c+1) E/C).
  bool tp = false;
  // The routed experts' weights are baked in a block-Hadamard basis (W_gu H_128 along hidden,
  // W_dn H_64 along inter). Their input is then bf16(H_128 x) (x itself stays for the router and the
  // shared expert) and their h is bf16(H_64 h) before the down projection (or its int8 quant).
  bool rot = false;
  // Decode W4A8 (moe_decode only; moe_rows takes its own a8 argument). The routed experts'
  // input x (bf16(H x) under rot) is quantised to int8 per (row, 128-group) and h (bf16(H h) under rot)
  // per (slot row, 64-group), with row_quant.hpp's rule, into MoeScratch xq / xs / hq / hs; gate_up and
  // down then run s8 x s4 DPAS on the same int4 bytes: acc += (float(int32 sum) * s_a) * s_w. The
  // router and the shared expert keep bf16. Needs tp (64-k half blocks = the h groups).
  bool a8 = false;
  // With a8, x's rotate + quantise runs in the router's launch (extra work-groups); false = its
  // own rot_quant launch. The same bits (test-moe).
  bool a8_fuse = true;
  // h's rotate + quantise inside down's half-block load (every down work-group quantises the halves it reads) instead
  // of its own launch. The same bits; the redundant work scales with the unique experts, so off.
  bool a8_h_in_down = false;
  // h's rotate + quantise in gate_up's epilogue (a work-group per 64-column group: 4 tiles x gate|up x the k split),
  // no h launch and no routed bf16 h. Takes precedence over a8_h_in_down.
  bool a8_h_in_gate_up = true;
  // moe_rows' W4A16 work-group-per-block gate_up (prefill) widens the int4 codes as 128 + q in two bitwise ops and starts
  // each group's chain from -136 x the row's sum of x over the group, instead of widening q - 8 exactly. Fewer
  // instructions; not the same bits as the segment kernels (fp32 sums of larger terms): a numerics choice.
  bool w4_bias = false;
  // The most rows a step of this model carries, 1..kMoeRows. It sizes h and the partials (moe_h_elems,
  // moe_partial_elems: [slot][row] blocks, so 64 rows take four times the bytes of 32); moe_decode refuses more rows.
  // Above kStepRows32: bf16 x only (no rot, no a8), no CPU path.
  int rows_cap = kStepRows32;
  int per_card() const { return tp ? experts : experts / cards; }  // blob table entries per layer
  int inter_local() const { return tp ? inter / cards : inter; }  // routed h columns on this card
  int slots() const { return rows_cap * topk; }  // routed slots at the capacity (a launch's static grid: shape x topk)
};

// Byte offsets of the sections inside one expert blob (the qwen38 ExpertLayout):
// down codes W4 [inter/128][hidden/16][1024]
// gate|up scales fp16 [hidden/128][2 inter]
// down scales fp16 [inter/128][hidden]
struct MoeBlobLayout {
  size_t gu_codes = 0, dn_codes = 0, gu_scales = 0, dn_scales = 0;
};

struct MoeWeights {
  DenseWeights router;  // bf16 [E + 1 -> padded][hidden]; row E = the shared gate
  const uint8_t *const *blobs = nullptr;  // device array [per_card]: this card's experts' blobs
  MoeBlobLayout blob;
  DenseWeights shared_gu;  // int8 [2 shared_inter][hidden]: gate rows, then up rows
  DenseWeights shared_down;  // int8 [hidden][shared_inter]
  const uint8_t *cold = nullptr;  // device [per_card]: != 0 = the blob is cold (pinned host); moe_rows staging; 2 = the
                                        // decode CPU path's set (MoeCpuLink)
};

// Written by the route launch (device memory). Slots [0, count) are this card's unique experts.
struct MoeRoute {
  int32_t count;
  int32_t expert[kMoeMaxSlots];  // global expert id
  const uint8_t *blob[kMoeMaxSlots];  // its blob (from the card's table: VRAM or pinned host)
  uint32_t mask[kMoeMaxSlots];  // bit r: row r routes here
  uint32_t mask_hi[kMoeMaxSlots];  // bit r: row 32 + r routes here (written and read by the 64-row shape only)
  float weight[kMoeRows][kMoeMaxSlots];  // per row, per slot; 0 where the row does not route here (the 64-row shape
                                              // writes a row's own slots only: the combine reads no others)
  float shared_gate[kMoeRows];  // sigmoid(bf16 logit)
  int32_t shared_lo, shared_hi;  // this card's shared-expert columns (a fixed 1/cards), in 16-column tiles
  int32_t ids[kMoeRows][kMoeMaxTopk];  // every row's top-k (all cards), descending
  float topw[kMoeRows][kMoeMaxTopk];
  // Each row's slots in ascending order (row_n of them; moe_decode's route writes them, its combine reads only these
  // partials).
  int16_t row_slot[kMoeRows][kMoeMaxTopk];
  int32_t row_n[kMoeRows];
  // MoeCpuLink: the experts this step hands to the host pool, and the rows of each the pool computes. An expert whose
  // every row is the pool's has a null blob[] entry (gate_up and down skip it); the wait launch writes the pool's rows
  // of each into its partial.
  int32_t cpu_n;
  uint32_t cpu_req;
  int32_t cpu_slot[kMoeCpuMax];
  const uint8_t *cpu_blob[kMoeCpuMax];
  uint32_t cpu_mask[kMoeCpuMax];
};

// The decode MoE's optional CPU path for cold experts (moe_cpu.hpp: the pool, the mailbox, the numerics). Needs
// MoeShape:tp, no rot, no a8, inter_local <= 512, and MoeWeights:cold (0 hot, 1 cold on the card, 2 cold and the
// pool's). All null = off (the default).
struct MoeCpuLink {
  MoeCpuMail *mail = nullptr;  // host USM: the request and the done word
  moe_bf16 *x = nullptr;  // host USM [8][hidden]: the step's rows, written by the route launch
  const float *y = nullptr;  // host USM [kMoeCpuMax][8][hidden]: the pool's y (MoeCpuPool's y buffer)
  uint32_t *state = nullptr;  // device [16]: [0] request counter, [1] 1 = on (0 after a timeout), [2] timeouts
  int min_rows = 1;  // steps with fewer rows keep every expert on the card (> 1 makes a row's bits depend on the batch)
  int min_cold = 2;  // per row: a row routing to fewer of the layer's pool experts keeps them on the card (one cold
                               // read hides behind the hot ones; a row's own count, so batch-invariant)
  uint64_t wait_cycles = 1400000000ull;  // the wait launch's bound (GPU timestamp cycles, ~2.8 GHz: 500 ms)
};

struct MoeScratch {
  float *logits = nullptr;  // [8][router.n]
  MoeRoute *route = nullptr;  // one record
  uint32_t *hits = nullptr;  // optional: device [experts] += 1 per expert a decode step reads (the ranking)
  moe_bf16 *h = nullptr;  // [shape x topk + 1][shape][max(inter, shared_inter)], sized at the capacity
  float *partial = nullptr;  // [shape x topk + 1][shape][hidden], sized at the capacity
  float *acc = nullptr;  // [kMoeRows][hidden]: the combine's rows as fp32, for the owner sum by copies (MoeWire::copy); null = none
  moe_bf16 *xr = nullptr;  // [8][hidden]: the rotated x (MoeShape:rot only)
  // MoeShape:a8 only: int8 x [8][hidden] + scales [8][hidden / 128]; int8 h laid out as h
  // (moe_h_elems bytes) + scales [shape x topk x shape][inter_local / 64] (moe_hs_elems).
  int8_t *xq = nullptr;
  float *xs = nullptr;
  int8_t *hq = nullptr;
  float *hs = nullptr;
  // XEN_DEC_PROFILE, decode only: cycle stamps (comm/wire.hpp stamp) after the router (word 0), the
  // route (1) and the experts (2), so a profile splits the layer into router | route | experts | combine + all-reduce
  uint64_t *stamps = nullptr;
};
// dst[r][c] = bf16(H_n src[r][c..]) per n-block of each row (normalised Walsh-Hadamard, n 64 or 128);
// rows of `cols` values at stride ld (src == dst allowed).
sycl::event moe_hadamard_rows(sycl::queue &q, int rows, int cols, int ld, const moe_bf16 *src, moe_bf16 *dst, int n);
// Int8 codes + scales per n-block (n 64 or 128) of bf16 rows at stride ld, rotated first when rot
// (rot_quant.hpp: bf16(H_n v) then row_quant.hpp's rule); dst at stride ld, scl [rows][cols / n].
sycl::event moe_rot_quant_rows(sycl::queue &q, int rows, int cols, int ld, const moe_bf16 *src, int8_t *dst, float *scl, int n, bool rot);
size_t moe_h_elems(const MoeShape &s);
size_t moe_hs_elems(const MoeShape &s);  // MoeScratch:hs floats
size_t moe_partial_elems(const MoeShape &s);

// One card's end of the all-reduce: inbox on this card (peer-resident), peer_inbox on the other
// card, each 2 (parity) x 16 (kMoeRows, whatever the shape) x hidden words initialised to kMoeMailEmpty (moe_wire_init). error:
// a device word, set to 1 when a wait timed out (the output then holds NaN for those words).
inline constexpr uint32_t kMoeMailEmpty = 0xFFFFFFFFu;
struct MoeWire {
  uint32_t *inbox = nullptr;
  uint32_t *peer_inbox = nullptr;
  uint32_t *error = nullptr;
  uint64_t wait_cycles = 2800000000ull;  // >= 1 s at 2.8 GHz
  // TP4: four cards (comm/wire.hpp WireQuad: two hops, every card ((p0 + p1) + (p2 + p3)); the inbox
  // then holds a slot region per hop: moe_wire_words with MoeShape:cards 4). inbox / peer_inbox unused.
  WireQuad quad{};
  // On = the combine as an owner sum (comm/wire.hpp, rows by owner; it rules over quad): out gets
  // this card's own rows only. Each inbox: owner_sum_words(hidden), owner_mail_init. rows >= kOwnerMinRows.
  OwnerMail owner{};
  // On (with owner): the owner sum of steps of kOwnerCopyMinRows rows and more goes by copies (comm/wire.hpp): the
  // combine writes every row's fp32 sum to MoeScratch::acc and kernels::pair_reduce's copy path sums the owner's rows.
  OwnerSumCopy copy{};
};
size_t moe_wire_words(const MoeShape &s);
sycl::event moe_wire_init(sycl::queue &q, const MoeShape &s, uint32_t *inbox);

struct MoeEvents {
  sycl::event router, route, gate_up, down, combine;
  sycl::event cpu_wait;  // the CPU path's wait + copy (MoeCpuLink)
  sycl::event topk, shared_gu, shared_down;  // moe_rows only (profile)
  sycl::event send;  // moe_rows split with a send: the peer copy
};

// x bf16 [shape][hidden] (row_shape(rows) rows readable; only rows < `rows` are routed); out bf16
// [shape][hidden] (rows < `rows` written). parity: alternate per call on both cards (the layer index & 1).
// gu_ksplit / dn_ksplit: work-items per (gate or up / down) tile sharing k (1, 2, 4, 8 for
// gate_up; 1, 2, 4 for down; 0 = the default). Every split sums its parts in part
// order, so a split is deterministic (but not bit-equal to another split).
// cpu (optional): the CPU path (MoeCpuLink); the step then also runs the wait launch before the combine.
MoeEvents moe_decode(sycl::queue &q, const MoeShape &s, int rows, const moe_bf16 *x, const MoeWeights &w, const MoeScratch &sc,
                     const MoeWire *wire, int parity, moe_bf16 *out, int gu_ksplit = 0, int dn_ksplit = 0, const MoeCpuLink *cpu = nullptr);
// The device state words a MoeCpuLink needs, initialised (counter 0, on).
sycl::event moe_cpu_state_init(sycl::queue &q, uint32_t *state);

// Per-card blob of an expert under MoeShape:tp: card c's 1/cards of the intermediate columns.
// gate|up codes W4 [hidden/128][2 inter_c/16][1024] (this card's gate tiles, then its up tiles)
// down codes W4 [inter_c/P][hidden/16][P x 8] (P-k pieces of the full 128-k blocks, P = moe_tp_piece)
// gate|up scales fp16 [hidden/128][2 inter_c]
// down scales fp16 [moe_tp_scale_rows][hidden] (the 128-k groups the card's pieces touch)
// inter_c = inter / cards must be a multiple of 32. stride: bytes rounded up to 1 KiB (every block stays 1024-aligned).
MoeBlobLayout moe_tp_layout(int hidden, int inter, int cards, size_t *stride);
// The down codes' k piece: 64-k half blocks while inter_c is a multiple of 64 (TP1 / TP2), 32-k quarter blocks
// otherwise (TP4's 640 / 4 = 160 columns per card).
inline constexpr int moe_tp_piece(int inter_c) { return inter_c % 64 == 0 ? 64 : 32; }
// 128-k groups a card's pieces touch (the most over the cards): ceil(nh / 2) for halves.
inline int moe_tp_scale_rows(int inter, int cards) {
  const int ic = inter / cards, pk = moe_tp_piece(ic), np = ic / pk, pg = 128 / pk;
  int rows = 0;
  for (int c = 0; c < cards; ++c) rows = std::max(rows, (c * np + np - 1) / pg - c * np / pg + 1);
  return rows;
}
// Card c's per-card blob from a full blob (`full` in `full_lay, the qwen38 / pack_w4_gptq layout).
void moe_split_tp(const MoeBlobLayout &full_lay, int hidden, int inter, int cards, int card, const uint8_t *full, const MoeBlobLayout &tp_lay,
                  uint8_t *out);

// Launch 2 alone (tests): logits fp32 [4][ld] (column E = the shared gate logit).
sycl::event moe_route(sycl::queue &q, const MoeShape &s, int rows, const float *logits, int ld, const uint8_t *const *blobs, MoeRoute *rt);

// ---- host reference ----------------------------------------------------------------------------
// One expert as the checkpoint stores it, dequantised: w[k][n] = (q - 8) * scale (float).
struct MoeRefExpert {
  std::vector<float> gate, up;  // [hidden][inter] (k-major, as GPTQ)
  std::vector<float> down;  // [inter][hidden]
};
struct MoeRefWeights {
  std::vector<float> router;  // [E][hidden] (bf16 values)
  std::vector<float> shared_gate;  // [hidden]
  std::vector<float> s_gate, s_up;  // [shared_inter][hidden] (dequantised int8)
  std::vector<float> s_down;  // [hidden][shared_inter]
  // expert(e) -> dequantised expert (fetched lazily; only routed experts are needed)
  std::vector<const MoeRefExpert *> experts;  // [E], null where not provided
};
// h vectors keyed (row, expert) for routed experts and (row, -1 - card) for a card's shared
// half (the shared key holds all shared_inter columns, zeros outside that card's half).
using MoeRefH = std::map<std::pair<int, int>, std::vector<float>>;
struct MoeRefOut {
  std::vector<int32_t> ids;  // [rows][topk]
  std::vector<float> topw;  // [rows][topk]
  std::vector<float> logits;  // [rows][E] bf16-rounded
  std::vector<float> out;  // [rows][hidden] (bf16 values)
  MoeRefH h;  // the reference's own h (bf16 values)
};
// Routing only (needs router + shared gate): fills ids / topw / logits.
void moe_route_ref(const MoeShape &s, int rows, const float *x, const MoeRefWeights &w, MoeRefOut &o);
// The whole block with the same EP grouping as the device (cards from s.cards); the experts
// named by the routing must be present in w.experts. force (optional): use these h vectors
// instead of computing them (teacher forcing at the h rounding point: one bf16 flip of h moves
// a whole output row by a fraction of an ulp, so the down half is checked from the device's h).
void moe_ref(const MoeShape &s, int rows, const float *x, const MoeRefWeights &w, MoeRefOut &o, const MoeRefH *force = nullptr);

}  // namespace xenithium:kernels
