// Xenithium · qwen35 whole-model decode on the cards: one sequence, 1-4 rows per step. Also serves arch llama (Nanbeige)
// and arch flint (Recursive Flint INT4: Config::flint - sliding / full GQA with sinks, the MoE layers through
// arch::flint::MoeSet, affine weights, the 8-bit embedding, the many-row prefill).
//
// A step at position p with rows r = 0..R-1 (tokens at p + r), on both cards (TP2) or on one (TP1: the fabric holds one
// card, every matrix whole, every "two-card sum" below the card's own fp32 partial rounded once, no wire):
//   res = embed(ids)
//   per layer l:  xn = rms_norm(res += mlp_out[l-1], input_layernorm)         (no add at layer 0)
//                 mix = GDN (48 layers) or gated GQA (16) of xn               (both cards end with the same bf16 mix)
//                 xn = rms_norm(res += mix, post_attention_layernorm)
//                 mlp_out = Mlp(xn)
//   xn = rms_norm(res += mlp_out[63], norm); logits = lm_head(xn); greedy pick across the cards
// The residual adds are bf16(res + block output) (rmsnorm.hpp), the bf16 decoder layer's own rounding points.
// Speculative rows: step() leaves the rows pending, accept(n) keeps the first n (GDN commits them at the next step,
// the KV rows past them are rewritten when their positions come again). Teacher-forced runs accept every row.
//
// Prefill: prefill(ids, n) runs a prompt LAYER-MAJOR in chunks of ModelOptions:prefill_chunk
// rows: layer l on every row of the chunk before layer l + 1 (causal without tricks: every state is per layer and the
// rows still run in position order inside a layer). The chunk's residual / norm / mixer / MLP rows live in chunk
// buffers (Card::pf_*); a 4-row slice is a view at row 4 s. With ModelOptions::prefill_slices the norms run over
// the whole chunk (one work-group per row: the same bits), the mixers and the MLP run their DECODE kernels on 4-row
// slices with per-slice position / accept words and the GDN parity advancing per slice, so a prefill computes exactly
// what 4-row step()s compute (bit-identical nll). Without it the many-row kernels run instead of the slice loops:
// the MLP (Mlp:run_rows), the GDN (gdn_rows over the chunk, W4 projections, swish gate; it commits every row, so the
// next step replays none: pf_committed_) and the attention (gqa_rows: projections and prep over the chunk, the decode
// attention + merge per 64-row batch).
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <array>
#include <optional>
#include <span>
#include <vector>

#include "xenithium/arch/common/head_cands.hpp"
#include "xenithium/arch/qwen35/arch.hpp"
#include "xenithium/arch/qwen35/attn.hpp"
#include "xenithium/arch/qwen35/dflash.hpp"
#include "xenithium/arch/qwen35/gdn.hpp"
#include "xenithium/arch/qwen35/head.hpp"
#include "xenithium/arch/qwen35/mlp.hpp"
#include "xenithium/arch/qwen35/mtp.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/kv_lend.hpp"
#include "xenithium/model/kv_pages.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"

namespace xenithium::arch {
class VisionStore;
}

namespace xenithium::arch::flint {
class MoeSet;
}

namespace xenithium::arch::qwen35 {

// A prefill call's two-card sums split over the cards (pair_split_rows) from this many rows (the generic
// kPairSplitMinRows is 64): same bits as the rows wire. Calls of 9-15 rows keep the rows wire, 1-8 rows run as decode
// GEMMs.
inline constexpr int kPfSplitMinRows = 16;

struct ModelOptions {
  int64_t context = 8192;                  // tokens of KV per attention layer (rounded up to 64)
  kernels::QsaKv kv = kernels::QsaKv::i8;  // KV dtype
  int threads = 16;                        // loader threads
  bool profile = false;                    // keep card 0's per-module times of the last step
  int prefill_chunk = 2048;  // rows per prefill chunk (multiple of 4; sizes the chunk buffers)
  // Prefill stage: false = the many-row kernels where they exist (MLP: w4a16_gemm_pf over the chunk);
  // true = every component on its decode kernels per 4-row slice (bit-identical to 4-row steps).
  bool prefill_slices = false;
  bool prefill_attn_slices = false;  // true = attention on decode slices even in the many-row path
  // the many-row attention body (kernels:kGqaAttnBatch 0 / kGqaAttnBlock 1 / kGqaAttnStaged 2 /:
  // kGqaAttnFa 3, the default, same bits as 2) and the staged / fa body's rows per work-group (0 = its default)
  int prefill_attn = kernels::kGqaAttnFa;
  int prefill_attn_rows = 0;
  // the fa body's P as hi + lo bf16 (true: every attention path's numerics) or bf16 hi only (false: half the
  // P.V DPAS, a numerics switch)
  bool prefill_attn_p_lo = true;
  // The fa body's q.k (kernels:gqa_attn_fa qk): 0 = bf16 q (exact), 1 = int8 q per (row, head) on s8 DPAS, 2 = q as
  // hi + lo int8 codes (only with prefill_attn_p_lo false). 1 / 2 need the int8 KV.
  int prefill_attn_qk = 0;
  // the many-row W4 GEMMs (every prefill projection and the MLP) dequantised: bf16((q - 8) x scale) weights, the
  // whole k accumulated on DPAS, 32 x 4 tiles (kernels::w4a16_pf_set_dequant; process-wide). false = the exact per-group
  // fp32 sums (decode's arithmetic). A numerics switch.
  bool prefill_w4_dequant = false;
  // (opt-in, defaults off):
  //   head_w4: the lm_head from the checkpoint's lm_head.qweight / scales / qzeros (int4 g128 symmetric, GPTQ orientation:
  //            `xenithium-bake head4-calib`) on the W4 GEMM, the drafters' draft-vocabulary rows too (a numerics change);
  //   prefill_w4_a8: the many-row W4 GEMMs on int8 activations per (row, 128-k group), s8 x s4 DPAS (the int4 blocks
  //            reordered into s4 operands while staged); runtime-switchable through set_prefill_options; a numerics
  //            switch.
  //   prefill_a8_keep: the W4 matrices that stay W4A16 under prefill_w4_a8, items joined by ',' or '+':
  //            FAMILY[@L0[-L1]], FAMILY = qkvz | out | gdn (both) | qkv | o | attn (both) | gu | down | mlp (both) | all;
  //            no range = every layer. Empty = every prefill W4 GEMM on W4A8.
  bool head_w4 = false;
  // head_rescore: with head_w4, also load the int8 rows (Head's kHeadI8Key: +636 MB per card) and run the two-stage head:
  //   the int4 rows score the vocabulary, each row's top head_rescore 16-column tiles per card get the int8 rows' logits
  //   (the int8 head's bits there; picks == the int8 head's whenever its pick lies in those tiles). 0 = the plain int4
  //   head. StepOptions::head switches int8 / int4 / rescore at run time.
  // draft_head: the drafters' draft-vocabulary rows: -1 = as the target (head_w4), 0 = int8 rows, 1 = the int4
  //   lm_head.qweight rows (a draft pick only steers acceptance, never the output).
  int head_rescore = 0;
  int draft_head = -1;
  bool prefill_w4_a8 = false;
  std::string prefill_a8_keep;
  // prefill_a8_body: the W4A8 GEMM body, -1 = auto (a8_body_auto: by the card count), 0 = slm (the pipelined SLM body),
  // 1 = direct (the SLM-free body on every W4A8 call); same outputs bit for bit (kernels::w4a16_pf_set_a8_body,
  // process-wide), runtime-switchable through set_prefill_options.
  int prefill_a8_body = -1;
  // Prefill calls of <= prefill_short rows run their W4 GEMMs on the decode kernels (W4A16, decode's k split;
  // kernels::w4a16_pf_set_short_rows, process-wide); 0 = off. Runtime-switchable through set_prefill_options.
  int prefill_short = 0;
  // A prefix-cache fork shares the source's KV pages below the snapshot (reference counts in model::KvPages,
  // copy-on-write before a slot writes a shared page) instead of copying them; false = the copy.
  bool kv_share = true;
  // The many-row two-card sums on the bf16 wire (kernels::PairSplit::bf16: the peer's rows sent as bf16, bf16(own) +
  // bf16(peer) summed in fp32, as the decode's XEN_WIRE bf16); false = fp32 partials. Runtime-switchable.
  bool prefill_wire_bf16 = false;
  // YaRN per request. Past the trained positions the rope table holds one sub-table per tier (trained x 2^k positions
  // at that length's factor: 1, 2, ...; the last tier = the whole context at its factor), and a sequence starts at its
  // tier's first row through the rope delta (set_rope_tier: the M-RoPE delta word, plain rope too), so a short request
  // on a 1M server gets factor 1, the bits of a server at the trained context. false = one factor for every request.
  // Arch flint never scales (yarn_factor 1).
  bool yarn_tiers = true;
  int rope_tier_reset = -1;  // the tier reset() starts a sequence in (-1 = the last: the whole context's factor)
  bool prefill_split = true;  // the many-row two-card sums split over the cards (pair_split_rows) from 64 rows; false = the mailbox
  // (all runtime-switchable through set_prefill_options, same bits on or off)
  // From 512 rows: the split's producers (GDN out, attention o, MLP down) send the peer's rows first on a side queue ...
  bool prefill_send = true;
  // ... and the consumer (the norm, then the mixer's qkvz / qkv or the MLP's gate|up GEMM) runs the card's own rows before
  // it waits for the peer's
  bool prefill_defer = true;
  bool prefill_fused_swiglu = true;  // MLP gate|up with swiglu in its epilogue (w4a16_gemm_pf_swiglu)
  bool prefill_embed_chunk = true;   // one embedding gather over the chunk (false: per 4-row slice through the head)
  int prefill_chunk_run = 0;         // rows per chunk at run time (multiple of 4, <= prefill_chunk; 0 = prefill_chunk)
  bool gdn_scan = true;  // prefill GDN: full 64-row chunks at absolute multiples of 64 on the chunked scan (gdn_rows.hpp)
  // The checkpoint's MTP drafter (mtp.hpp): loaded when set; its matrices int8 per row (1) or bf16 (2), its KV.
  // Needs the many-row prefill (prefill_slices false): a prefill also writes the drafter KV of the prompt rows.
  bool mtp = false;
  int mtp_dense_bytes = 1;
  kernels::QsaKv mtp_kv = kernels::QsaKv::i8;
  int mtp_vocab = 0;  // the draft vocabulary (MtpOptions::vocab): 0 = the target's lm_head
  // A DFlash 2 block drafter (dflash.hpp) from this directory instead of the MTP layer (mtp is then ignored): a verify
  // carries up to 8 rows of one sequence (the GDN's 8-row log), the drafter reads the target's features. Its matrices
  // int8 per row (1) or bf16 (2); the draft vocabulary is mtp_vocab.
  std::string dflash_dir;
  int dflash_dense_bytes = 1;
  bool dflash_profile = false;  // diagnostic: DFlashOptions:profile (stage wall times; waits inside the pass)
  int dflash_ctx_defer = 64;  // DFlashOptions:ctx_defer (deferred context rows per slot; 0 = off)
  double dflash_conf_cut = 0.02;  // DFlashOptions:conf_cut (draft confidence chain cut; 0 = off)
  int dflash_tree = 0;  // DFlashOptions:tree (best-first tree nodes per pass; 0 = the chain)
  double dflash_tree_cut = 0;  // DFlashOptions:tree_cut
  int dflash_tree_seqs = 1;  // DFlashOptions:tree_seqs
  double dflash_tree_temp = 0;  // DFlashOptions:tree_temp (0 = by drafter)
  // seg_rows 16 = segments (one sequence's verify) of up to 16 rows: a drafter trained on a block of 16 or more (DSpine
  // b16), its passes up to 16 block rows (DFlashOptions:wide), the GDN's pending logs at 16 rows per slot (more VRAM per
  // slot); 0 / 8 = 8 with a DFlash drafter. dflash_tree_wide: a 16-row tree's nodes (DFlashOptions:tree_wide). When a
  // sequence drafts 16 rows is the step model's (its depth table, StepModelAdapter::wide_from). step_rows: a step's rows
  // in all, 8 or 16 (0 = the arch's: 16 for qwen35, 8 for llama / flint; slots <= step_rows).
  int seg_rows = 0;
  int dflash_tree_wide = 15;
  int step_rows = 0;
  // With dflash_dir, N > 0 loads the MTP layer too (the mtp_* options) and a draft call for N or more sequences drafts
  // with MTP, fewer with DFlash; the other drafter only keeps its state current (MTP: its KV-only pass over the step's
  // rows; DFlash: the context rows, deferred by dflash_ctx_defer). 0 = DFlash alone.
  int mtp_from = 0;
  // MtpOptions::kv_defer (the MTP drafter's KV-only passes deferred, rows per slot; 0 = off)
  int mtp_kv_defer = 64;
  // Segments of up to 8 rows without a drafter (the GDN's 8-row log; DFlash sets it anyway): teacher-forced scoring in
  // 8-row steps, the same bits as 1-row steps.
  bool rows8 = false;
  // Sequence slots, each with its own GDN states, KV (context tokens each) and drafter KV. The single-sequence calls act
  // on use_slot()'s slot; verify_batch / accept_batch / draft_batch run several in one step.
  int slots = 1;
  int profile_card = 0;  // the card whose per-module times profile() keeps
  // The KV layout. 0 = fixed: every slot owns `context` tokens of pages from the load (an identity page table). N > 0 =
  // one shared pool of N tokens (rounded up to 64 per page; at least one context) for every slot's attention layers and
  // drafter: a sequence takes pages as it grows (kv_reserve; the model's own calls take what they write) and returns
  // them at reset() / kv_release. -1 = auto: the pool takes the VRAM the tighter card has left after everything else ran
  // once (kv_fit, after a warm-up; until then one context of pages; needs `loader`).
  int64_t kv_pool_tokens = 0;
  const l0::Loader *loader = nullptr;
  bool kv_pool_scramble = false;  // pages handed out in a fixed shuffled order (a check: results must not depend on it)
  // KV lending. The shared pool's pages no sequence holds keep the oneDNN layouts of the wide prefill matrices
  // (kernels::onednn_pf_keep: gate | up, the GDN q|k|v|z and the attention q|k|v of every layer whose width takes the
  // oneDNN route), so a prefill call reads them instead of repacking each matrix per call. The stored weights stay the
  // one copy: a layout is a cache in memory the KV does not use. The pool's lend limit sits kv_lend_keep tokens above
  // the highest held page; a reservation or copy-on-write that needs pages past it gets them back first (the layouts
  // there are forgotten: the per-call repack again, the same bits) and lend_idle lends them again once the pages are
  // free. Needs the shared, unscrambled pool and the oneDNN route (kernels::w4a16_pf_onednn() > 0).
  bool kv_lend = false;
  int64_t kv_lend_keep = 32768;
  // Pinned host memory for the prefix cache's snapshots (GiB over both cards; 0 = no snapshots). One snapshot holds a
  // slot's GDN states (+ the DFlash rings; arch flint: the sliding layers' ring windows): snap_bytes.
  double snap_gib = 0;
  // Page-selected decode attention (opt-in). N > 0 keeps every attention page's key bounds (KV pool + 1/32 at int8 KV)
  // and lets a decode step whose context passes N pages attend, per sequence and kv head, to N of them: the first, the
  // last 16 and the best page scores of the rest (kernels::GqaStep::select). StepOptions::attn_select overrides it per
  // step. 0 = dense attention, no bounds.
  int attn_select = 0;
  // Page-selected prefill attention (opt-in, lossy). N > 0 keeps the key bounds too and lets a
  // prefill call whose context passes N pages attend, per attention row block and kv head, to N of them, chosen as a
  // decode step's (kernels::gqa_rows select). The call's K / V and every later layer are then built on the selected
  // attention: the prompt's KV differs from a dense prefill's. A call that ends under N pages is the dense call, bit for
  // bit, and the attention row block of the prompt's last rows (a call with no next token) attends to every page. The
  // fa attention body on the int8 or bf16 KV; set_prefill_options switches the page count on a model loaded with N > 0
  // (the bounds and the page-set scratch are the load's). 0 = dense prefill.
  int prefill_select = 0;
  bool page_bounds() const { return attn_select > 0 || prefill_select > 0; }  // the caches keep every page's key bounds
  // A sequence measures its own page counts. On its first select_self selected steps every
  // head keeps attn_select pages and the model reads the page scores the select launch computes anyway; from them
  // alone (SelectShare, select_curve_add: no attention is read back) it shares the pages out among the (attention
  // layer, kv head) units, and the sequence keeps those counts from then on: past kSelectFloor pages a kv head keeps
  // kSelectFloor + round(weight x (N - kSelectFloor)) pages (select_head_pages; <= 4096), heads whose attention is
  // spread more, sharp ones fewer, at the same mean. Nothing to calibrate, any checkpoint. 0 = off: N pages for
  // every head. The selected prefill (prefill_select) keeps N for every head.
  int select_self = 0;
};

// Run-time switches of the decode step (set_step_options).
struct StepOptions {
  // The decode two-card sums on the bf16 wire (kernels::PairWire::bf16: bf16 partials summed; half the bytes). The
  // drafter and the many-row prefill sums stay fp32.
  bool wire_bf16 = true;
  // The int8 wire for those sums (kernels::PairWire::i8; rules over wire_bf16): 8-bit codes per 32-column group, 56 % of
  // the bf16 wire's bytes. A numerics change, opt-in (--wire int8). TP2, the dense 27B only.
  bool wire_i8 = false;
  // The decode W4 GEMMs' auto k split target (kernels:w4a16_set_min_items), in work-items (default 1,024); another
  // target is another fp32 order of the group sums.
  int w4_items = kernels::kW4MinWorkItems;
  // The decode MLP gate|up GEMM's own k-split target; 0 = by the cards: w4_items' split on two (TP2), 16,384 items on
  // one (TP1: the whole [2 x 17,408] x 5,120 matrix is 2,176 tiles, so 1,024 gives no split). The other GEMMs keep
  // w4_items.
  int w4_items_gu = 0;
  // Swiglu in the decode gate|up GEMM's epilogue (kernels::w4a16_gemm_rows_swiglu; the same bits as gate|up -> swiglu).
  // false = the separate swiglu launch.
  bool fused_swiglu = true;
  // Pages a decode step's attention keeps per sequence and kv head (needs ModelOptions::attn_select > 0: the bounds);
  // -1 = ModelOptions::attn_select, 0 = dense.
  int attn_select = -1;
  // The lm_head for every pick and logits row from now on: -1 = as loaded (ModelOptions), 0 = int8 rows, 1 = int4, 2 =
  // the two-stage head (int4 + int8 rescore); head_tiles its tiles per card (-1 = ModelOptions::head_rescore or 16).
  int head = -1;
  int head_tiles = -1;
  // Arch flint: decode steps replayed from recorded graphs (one per card per key: rows, rows per segment, the
  // attention / MLP / MoE wire parities at the step's start); the attention grid bound is the capacity (its splits cap
  // at 64 from 4,096 tokens; idle work-groups return), so a graph replays at any position and slot assignment. The
  // same bits as eager. Profiled steps run eager. Arch flint only (its kernels' grids do not read the bound otherwise).
  bool graphs = true;
  // The decode attention's numerics (kernels::GqaShape::dec_qk / dec_p_lo; arch qwen35 only, dense attention): attn_qk
  // 0 = bf16 q.k, 1 = int8 q x the int8 K codes on s8 DPAS, 2 = int8 q as hi + lo codes (with attn_p_lo false);
  // attn_p_lo false = P.V on P as one bf16. The defaults are the exact decode numerics.
  int attn_qk = 0;
  bool attn_p_lo = true;
  // The decode step's int4 GEMMs of the target (MLP gate | up and down, GDN q|k|v|z and out, attention q|k|v and o) on
  // int8 activations (kernels::w4a8_rows_set; the drafters' and lm_head's matrices stay W4A16). A numerics change,
  // opt-in (--decode-w4 a8). The dense 27B only (symmetric g128 matrices).
  bool w4_a8 = false;
  // the decode attention group's L2 prefetch (-1 = kernels::gqa_attn_group's pick, 0 off, 1 on) and register file (0 =
  // its pick, 128 / 256); the same bits either way
  int attn_pf = -1;
  int attn_grf = 0;
};

// Card 0's GPU time per module in the last step (us, summed over layers), and the step's GPU span.
struct StepProfile {
  double embed = 0, norms = 0, gdn = 0, attn = 0, mlp = 0, head = 0;  // launch sums
  double moe = 0, moe_parts[5] = {};  // arch flint: router, route, gate_up, down, combine (+ the exchange)
  double gdn_parts[4] = {};   // qkvz, core, out, reduce
  double attn_parts[6] = {};  // qkv, prep, attn, merge, o, reduce
  double mlp_parts[4] = {};   // gate|up, swiglu, down, reduce
  double span = 0;            // first launch start to the pick's end
};

// Prefill wall time per stage (ms, summed over the call's chunks; both cards synced after every stage: the profiled
// run is slower than an unprofiled one, the split is what it is). Kept when ModelOptions::profile is set.
struct PfProfile {
  double embed = 0, norms = 0, gdn = 0, attn = 0, mlp = 0, head = 0, total = 0;
  double attn_parts[5] = {};  // card 0 GPU ms of the many-row attention: qkv, prep, attn (+ merge), o, reduce
  double mlp_parts[4] = {};   // gate|up, swiglu, down, reduce
  double gdn_parts[3] = {};   // qkvz, out, reduce (the rest of the GDN between them)
  double moe_parts[5] = {};   // arch flint: card 0 GPU ms of the many-row MoE: router, route, gate_up, down, combine
  double split = 0;           // wall ms of the split two-card sums (pair_split_rows), the producers synced first
};

namespace trained_qsa { class Device; }

class Model {
 public:
  static constexpr int kRows = kernels::kGdnRows;  // rows per sequence in a step (a step: up to kStepRows32 in all)
  using bf16 = sycl::ext::oneapi::bfloat16;

  Model(Fabric &fab, const std::string &dir, const ModelOptions &opt = {});
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  ~Model();

  const Config &config() const { return cfg_; }
  int64_t position() const { return pos_; }
  // ---- sequence slots ----
  int slots() const { return static_cast<int>(seqs_.size()); }
  // The cards the model runs on (the fabric's size): 2 = TP2 (every matrix split by head / column, two-card sums on the
  // mailbox wire), 1 = TP1 (every matrix whole on one card, no wire: each sum is the one card's fp32 partial rounded).
  int cards() const { return static_cast<int>(fab_.size()); }
  // The decode gate|up GEMM's k-split target in effect (StepOptions::w4_items_gu; 0 = by the cards), 0 = the auto split.
  int gate_up_items() const { return sopt_.w4_items_gu > 0 ? sopt_.w4_items_gu : cards() == 1 ? kGateUpItemsTp1 : 0; }
  static constexpr int kGateUpItemsTp1 = 16384;
  void use_slot(int slot);  // the slot the single-sequence calls act on (reset, step, accept, prefill, draft*)
  int slot() const { return slot_; }
  int64_t position(int slot) const;
  // Several sequences in one step: segment i = `rows` tokens of the sequence on slot `slot`, at its own position;
  // every segment has the same row count (1..8), or (unequal segments) the first ones B rows and the rest
  // one row each; up to step_rows() rows in all, each slot at most once. Rows are laid
  // out segment after segment (picks / logits / the drafts' `next`). Every row computes what a step of its own
  // sequence alone computes, bit for bit; a slot left out keeps its state. accept_batch: one count per segment.
  struct SeqRows {
    int slot = 0;
    const int32_t *tokens = nullptr;
    int rows = 1;
    const int *parent = nullptr;  // the rows are a draft tree (step_tree's parents); null = a chain
  };
  void verify_batch(const std::vector<SeqRows> &segs);
  void accept_batch(const std::vector<int> &n);
  // accept_batch after a verify whose segments may be trees: paths[i] = segment i's walked rows (accept_tree's path,
  // n[i] of them), null = a chain (its first n[i] rows)
  void accept_batch(const std::vector<int> &n, const std::vector<const int *> &paths);
  // Drafts after verify_batch + accept_batch: next [segments x rows] as Model::draft's per segment; returns `depth`
  // drafts per segment (fewer near the capacity). depth 0: the drafter's pass 1 writes its KV only (no MLP / head).
  std::vector<std::vector<int32_t>> draft_batch(const int32_t *next, int depth);
  // A depth per segment (0 = none: the drafters' KV / context of its rows only).
  std::vector<std::vector<int32_t>> draft_batch(const int32_t *next, const std::vector<int> &depth);
  int64_t capacity() const { return cap_; }  // one request's most tokens (XEN_CONTEXT)
  // Image input (arch qwen35 with mrope_section; not the prefill slice scaffolds). Token ids >= the vocabulary name
  // image rows of the store (VisionStore::prepare: encoded on first use; the embeddings read its pool) and take the
  // interleaved M-RoPE: 3-D positions for image rows (per-row rope rows of the prefill call), every later row of the
  // sequence at position + its delta (segment records' kSegRope, snapshots keep it). Before the first request.
  void set_vision(VisionStore *store);
  int64_t rope_delta(int slot) const { return seqs_.at(static_cast<size_t>(slot)).rope_delta; }
  // the static YaRN factor of the rotary table (1 = none; qsa_yarn_for). Arch flint keeps 1 at any context: its rotary
  // layers attend within a 513-token window and its full-attention layers have no rotary
  double yarn_factor() const { return yarn_factor_; }
  // YaRN tiers (ModelOptions::yarn_tiers). rope_tiers() = 1: one factor (yarn_factor). Tier t holds positions
  // < rope_tier_limit(t) at rope_tier_factor(t), ascending; the last = capacity() at yarn_factor(). set_rope_tier: the
  // current slot's sequence (at position 0, before its first prefill) runs in tier t; snapshots keep it.
  int rope_tiers() const { return static_cast<int>(tier_limit_.size()); }
  int64_t rope_tier_limit(int t) const { return tier_limit_.at(static_cast<size_t>(t)); }
  double rope_tier_factor(int t) const { return tier_factor_.at(static_cast<size_t>(t)); }
  void set_rope_tier(int t);
  // ---- the KV pages (ModelOptions:kv_pool_tokens) ----
  // kv_shared(): one pool shared by the slots. kv_reserve(slot, tokens): positions [0, tokens) of the slot's sequence
  // get pages (all or nothing; false = the pool cannot, nothing changed; true at once when they already have them; the
  // fixed layout: true up to the capacity). kv_release: the slot's pages back to the pool (reset() does it too). Every
  // model call that writes KV (prefill, steps, drafts) reserves what it writes itself and throws when the pool is out:
  // a serving engine reserves ahead (StepModel::kv_reserve) to choose what happens instead.
  bool kv_shared() const { return kvp_.shared(); }
  bool kv_reserve(int slot, int64_t tokens);
  void kv_release(int slot);
  int64_t kv_pool_tokens() const { return kvp_.pool_tokens(); }
  bool trained_qsa() const { return trained_qsa3_ != nullptr; }  // a trained page selection bank is loaded
  bool step_trained_sparse() const { return qsa_step_sparse_; }  // the last verify step ran the trained sparse walk
  static constexpr int trained_qsa_short() { return kTrainedQsaShort; }  // decode positions below it run dense
  int64_t kv_free_tokens() const { return kvp_.free_tokens(); }
  int64_t kv_used_tokens() const { return kvp_.used_tokens(); }
  size_t kv_bytes(int c) const { return kv_bytes_.at(static_cast<size_t>(c)); }  // KV of the target + drafter on card c
  // Auto pool (kv_pool_tokens -1): the load allocates one context of pages; kv_fit then re-allocates the pool to what it
  // holds now + the tighter card's free VRAM beyond reserve_gib (capped at slots x context). Call it after a warm-up: the
  // runtime takes VRAM on first use of the kernels, and an allocation counts in sysman's free VRAM only once it is
  // resident. Every slot must be empty (no pages held). Returns the pool's tokens.
  int64_t kv_fit(double reserve_gib);
  // KV lending (ModelOptions::kv_lend). lend_idle (the engine's idle work): the layouts whose pages are free again are
  // lent again, then up to max_fill layouts not repacked yet are (returns what it did; 0 = nothing left).
  struct LendStats {
    bool on = false;
    int items = 0;            // the wide matrices (per card)
    int lent = 0;             // of them with a lent region now
    int filled = 0;           // of those, repacked (the tighter card)
    int64_t limit_tokens = 0; // the pool's lend limit
    double lent_gib = 0;      // per card
    uint64_t raises = 0, given_back = 0, relends = 0;
  };
  LendStats lend_stats();
  int lend_idle(int max_fill);
  const std::string &kv_lend_note() const { return lend_note_; }  // "off", "off (why)" or the state at the last layout
  double vram_free_before_kv() const { return vram_free_before_kv_; }  // GiB, the tighter card (ModelOptions::loader set; else -1)
  double load_seconds() const { return load_s_; }
  size_t card_bytes(int c) const;  // dense weights loaded on card c

  // ---- the prefix cache (ModelOptions:snap_gib) ----
  // A snapshot is what a slot's sequence state holds outside its KV pages, at its position: every GDN layer's committed S
  // and conv window (and the DFlash drafter's context rings), in pinned host memory per card. snap_save takes it right
  // after a prefill call of the slot (the state is committed there; copies submitted on the compute queues, no wait);
  // snap_restore puts it back on the same slot, whose pages below the position must be untouched since; snap_fork puts it
  // on another slot together with a copy of the source slot's KV pages below the position (target layers + MTP drafter;
  // dst must hold pages for them). After either the slot continues with a prefill call at the snapshot's position.
  size_t snap_bytes(int c) const;  // one snapshot on card c
  int snap_capacity() const { return snap_cap_; }
  int snap_save(int slot);         // an id, -1 = every snapshot in use
  // Pins the host buffers of up to n more snapshot ids ahead of their first use; the engine's idle work calls it after
  // the load. Returns the ids pinned (0: all are).
  int snap_prepin(int n);
  void snap_restore(int slot, int id);
  void snap_fork(int dst, int src, int id);
  void snap_free(int id);
  void snap_sync();  // every snapshot's host copy done (they run beside the calls; probes time them)
  // A snapshot from inside a prefill call. snap_arm(slot, pos): the slot's next prefill call (prefill /
  // prefill_begin; its chunk must hold pos strictly inside, pos a multiple of 64, the call on the many-row path with the
  // chunked GDN scan) also captures the state at pos into a device buffer of its own: the GDN S and window there (the scan
  // in two parts around pos), the DFlash context of the rows below pos (the chunk's context pass in two parts) - the bits a
  // call ending at pos leaves. snap_take() after that call makes it a snapshot as snap_save does: an id, -1 = every
  // snapshot in use (the capture is kept for another try), -2 = nothing captured (not armed, or pos not inside).
  bool snap_inside() const { return snap_cap_ > 0 && !opt_.prefill_slices && opt_.gdn_scan; }
  void snap_arm(int slot, int64_t pos);
  int snap_take();
  int64_t snap_pos(int id) const { return snaps_.at(static_cast<size_t>(id)).pos; }
  const void *snap_host(int id, int c) const { return snaps_.at(static_cast<size_t>(id)).host.at(static_cast<size_t>(c)); }  // checks (after snap_sync)
  // the slot's live state (its last sequence where it stopped) can take a prefill call: not DFlash (its rings miss the last
  // verify's rows), no prefill call pending on it, the last step accepted (at most 4 pending GDN rows)
  bool live_ok(int slot);
  void kv_trim(int slot, int64_t tokens);   // keep the pages of positions [0, tokens)
  int64_t kv_held_tokens(int slot) const { return static_cast<int64_t>(kvp_.held_pages(slot)) * k_page_tokens; }
  int kv_shared_pages(int slot) const { return kvp_.shared_pages(slot); }  // (checks)
  uint64_t kv_cow_count() const { return kv_cow_pages_; }
  static constexpr int k_page_tokens = model::KvPages::kPageTokens;

  void reset();                             // a fresh sequence at position 0
  void note_prompt(int slot,int64_t total,int64_t restored);
  void step(const int32_t *ids, int rows);  // rows tokens at position(); waits for the cards
  void accept(int n);                       // keep the last step's first n rows (1..rows)
  // A step whose rows are a draft tree (the current slot; DFlash's 8-row verify): parent[r] = row r's parent row
  // (parent[0] = -1, parent[r] < r; DFlash::tree_parents). Row r sits at
  // position() + depth(r) and sees only its ancestors: its pick is the chain's over its path, bit for bit. Then
  // accept_tree(path, n): the accepted rows path[0 .. n) (path[0] = 0, each the previous one's child) become positions
  // position() .. + n - 1 (their K / V moved there, the GDN commits along the path at the next step) and the step's
  // rows are re-laid in path order (row i = path[i]; rows past n repeat the last) for draft(): pass it the picks re-laid
  // the same way, rows = the tree's rows, keep = n. Greedy, records path, no page selection (kernels::GqaStep::tree).
  void step_tree(const int32_t *ids, const int *parent, int rows);
  void accept_tree(const int *path, int n);
  // the parent rows of the slot's last drafts when DFlash drafted a tree ([0] = -1; draft r is row r; empty = a chain);
  // the step_tree parents of [anchor, drafts...]
  std::vector<int> draft_parents(int slot) const {
    return slot >= 0 && static_cast<size_t>(slot) < draft_par_.size() ? draft_par_[static_cast<size_t>(slot)] : std::vector<int>{};
  }
  // n prompt tokens at position(), layer-major in chunks (see above); every row is accepted, position() += n. The next
  // step() follows as after 4-row steps over the same tokens. on_rows(p0, rows): after a chunk's layers the head runs
  // per 4-row slice and on_rows is called with the slice's first position and rows (logits() / picks() then read that
  // slice); without on_rows the head runs on the prompt's last slice only.
  // next (with a drafter): the token after the n (-1 = unknown: the drafter KV of the last row waits for
  // draft_after_prefill).
  void prefill(const int32_t *ids, int n, const std::function<void(int64_t, int)> &on_rows = {}, int32_t next = -1);
  int prefill_chunk() const { return opt_.prefill_chunk_run > 0 ? opt_.prefill_chunk_run : opt_.prefill_chunk; }
  // The prefix cache's alignment (StepModel::snap_align): 64 where a row's bits do not depend on its prefill call's size
  // (no fa-body context splits, no library GEMM route); else the prefill chunk, so a cached continuation runs the very
  // calls a cold read runs (the tail splits and the library-vs-own GEMM choice follow the call's rows).
  int call_align() const;
  // One prefill call in parts (): prefill_begin(ids, n <= prefill_chunk, next) on the current
  // slot, then prefill_advance(layers) runs that many more layers (at least 1) and returns true once all ran (picks /
  // logits / draft_after_prefill then as after prefill()). Same bits as one call. Between parts steps of OTHER slots may
  // run; nothing else may. prefill_abort() drops a pending call (the slot needs reset() before reuse).
  void prefill_begin(const int32_t *ids, int n, int32_t next = -1);
  bool prefill_advance(int layers);
  void prefill_abort();
  // One prefill call for several sequences (prompts that wait together: a short call's time is
  // mostly its launches and two-card sums, which do not grow with the rows). The segments' rows sit in one call at
  // row offsets padded to kPfMergeAlign; embed, norms, every GEMM, the MLP and the two-card sums run once over all
  // rows, the GDN core and the attention once per segment on its own state and KV, one head over the segments' last
  // rows. A segment's rows, KV, GDN state and logits are those of its own prefill_begin call, bit for bit (the
  // many-row kernels' rows depend neither on the call's rows nor on a row's place in it): so the call takes the route
  // every segment would take alone - prefill_mergeable(slot, ids, n): more than the short-call rows (ModelOptions::
  // prefill_short), at most kPfMergeSegMax (under the 512-row send), text only, the many-row path, no armed snapshot,
  // and all segments padded within prefill_merge_rows() (under the oneDNN route's row threshold). After the call:
  // use_slot(slot) and picks / logits read head row prefill_last_row(); draft_after_prefill reads the slot's kept
  // row, so the slots' first drafts may run in any order after every slot's picks / logits were read.
  // prefill_advance / prefill_abort as for one sequence (an abort breaks every segment's slot).
  struct PrefillSeg {
    int slot = 0;
    const int32_t *ids = nullptr;
    int n = 0;
    int32_t next = -1;
  };
  static constexpr int kPfMergeAlign = 16, kPfMergeMax = 16, kPfMergeSegMax = 511;
  int prefill_merge_rows() const;  // 0: this load's prefill calls take one sequence
  int prefill_merge_max() const;   // segments per call
  bool prefill_mergeable(int slot, const int32_t *ids, int n) const;
  void prefill_begin(const PrefillSeg *segs, int count);
  // Switch the run-time prefill options (prefill_attn, prefill_attn_rows, prefill_attn_p_lo, prefill_attn_qk,
  // prefill_select, prefill_w4_dequant, prefill_split, prefill_send, prefill_defer, prefill_fused_swiglu,
  // prefill_embed_chunk, prefill_chunk_run, prefill_attn_slices, prefill_w4_a8, prefill_a8_keep) for the next
  // prefill; the allocation-sized ones (prefill_chunk, prefill_slices, context, kv) stay. Throws on a bad value.
  void set_prefill_options(const ModelOptions &o);
  void check_attn_qk(const ModelOptions &o) const;  // prefill_attn_qk vs the KV dtype / attention body / p_lo
  void check_prefill_select(const ModelOptions &o) const;  // prefill_select vs the arch, the KV dtype, the attention body and its numerics
  void check_decode_attn(const StepOptions &o) const;  // attn_qk / attn_p_lo vs the arch, the KV dtype, page selection
  // prefill_a8_body -1 (auto) resolved by the card count: TP2 = kernel body 3 = slm above 128 rows (its consumers' GEMMs
  // run under the split sums' side-queue sends) and direct on calls of <= 128 rows (no sends there), TP1 = kA8BodyTp1
  int a8_body_for(int body) const;
  // prefill_a8_keep -> the kernels' keep set (every card's matrices); throws on a bad item.
  void apply_a8_keep(const std::string &spec);
  const ModelOptions &options() const { return opt_; }
  std::vector<int32_t> picks(int card, int rows);
  // Rows' logits over the whole vocabulary, rounded to bf16 values: out [rows][vocab] (card 0's half first).
  void logits(int rows, std::vector<float> &out);
  // The step's rows' log-softmax records on the cards (kernels::logprob_stats per card, rows <= 8): rec
  // [cards][rows][kLpRec], xt [rows] the targets' bf16-rounded logits (+inf: no target). Waits.
  void logprob_stats(int rows, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt);
  // Each row's sampling candidates instead of the whole row: both cards' top T 16-column tiles by the bf16-rounded
  // maximum (arch::HeadCands), ids / vals [rows][n], n = cards x T x 16 returned; vals rounded as logits().
  int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals);
  // The residual after the step on card c (bf16 bits [rows][hidden]): both cards must agree.
  std::vector<uint16_t> residual(int card, int rows);
  std::vector<uint16_t> final_hidden(int card, int rows);
  int prefill_last_row() const { return pf_last_row_; }  // final_hidden row of the last prefill's last token // the head's input rows (after the final norm; checks)
  const StepProfile &profile() const { return prof_; }
  const PfProfile &pf_profile() const { return pf_prof_; }
  void set_profile(bool on) { opt_.profile = on && fab_.profiling(); }  // event times need profiling queues
  void set_profile_card(int c) { opt_.profile_card = c; }
  void set_step_options(const StepOptions &o) {
    if (o.attn_select > 0 && opt_.attn_select <= 0) throw std::invalid_argument("qwen35 model: page selection needs ModelOptions::attn_select (the key bounds)");
    if (o.attn_select > 0 && (o.attn_select < 32 || o.attn_select > kernels::kGqaSelMax)) throw std::invalid_argument("qwen35 model: attn_select 32..4096 pages");
    check_decode_attn(o);
    if (o.w4_a8 && !a8_rows_) throw std::invalid_argument("qwen35 model: W4A8 decode runs the dense 27B's symmetric int4 matrices (not llama / flint)");
    apply_head(o);  // (throws before anything changes)
    sopt_ = o;
    kernels::w4a8_rows_set(o.w4_a8);  // (process-wide: the registered matrices are this model's)
  }
  const Head &head() const { return *head_; }
  int attn_select() const { return sopt_.attn_select < 0 ? opt_.attn_select : sopt_.attn_select; }
  // Whether a step whose rows end below pos_bound attends to selected pages: kernels::gqa_decode's rule (selection
  // starts once the step's position bound passes the page budget; below it the launch is the dense one). A tree step
  // attends densely, so trees are drafted and verified only where this is false.
  bool selects_pages(int64_t pos_bound) const {
    const int pages = attn_select();
    return pages > 0 && pos_bound > static_cast<int64_t>(pages) * kernels::kQsaPage;
  }  // 
  const StepOptions &step_options() const { return sopt_; }
  // Host ms of the last step's submission (verify_batch entry to the last launch).
  double submit_ms() const { return submit_ms_; }
  int step_graphs() const { return graphs_recorded(); }  // recorded step graphs (arch flint)
  uint32_t wire_errors();
  // Diagnostic (the probe's select35): of the given whole pages of a slot, how many (page, attention layer, card, kv
  // head) units hold key bounds other than the min / max of the page's keys as stored (bf16). 0 with page selection for
  // every page the slot's position has left. Reads the pages back: waits for the cards.
  int64_t page_bound_errors(int slot, std::span<const int> pages);
  // Diagnostics (the probe's select35 --dump): fn(layer, card, q, q heads, position) after every
  // attention layer of a 1-row step: the row's q [q heads of the card][head_dim] as the attention read it. The step
  // waits for each attention layer, so such steps are slow. An empty fn = off.
  using AttnQFn = std::function<void(int layer, int card, const bf16 *q, int hq, int64_t pos)>;
  void on_attn_q(AttnQFn fn) { attn_q_fn_ = std::move(fn); }
  // The stored keys of a slot's first `pages` pages of attention layer `layer` on `card` (int8 KV only): the codes
  // k [pages][kv heads of the card][64][head_dim] and their scales ks [pages][kv heads][64]. Returns the card's kv
  // heads. Reads the pages back: waits for the cards.
  int read_keys(int slot, int layer, int card, int pages, std::vector<int8_t> &k, std::vector<float> &ks);
  // (checks): bytes of card c's memory at src into the host (waits)
  void read_device(int card, const void *src, void *dst, size_t bytes) { fab_.compute(static_cast<size_t>(card)).memcpy(dst, src, bytes).wait(); }

  // ---- the drafter: the MTP layer (ModelOptions:mtp) or DFlash 2 (ModelOptions:dflash_dir) ----
  bool has_drafter() const { return mtp_ != nullptr || dflash_ != nullptr; }
  Mtp *drafter() { return mtp_.get(); }
  DFlash *dflash() { return dflash_.get(); }
  const DFlash *dflash() const { return dflash_.get(); }
  // rows one sequence's segment may carry in a step: 4 (kRows), 8 with DFlash (its 8-row verify)
  int seg_rows_max() const { return rows_max_; }
  // rows one step carries in all: 16 (kernels::kStepRows32) for qwen35, 8 (kStepRows8) for llama / flint (their
  // zero-point / affine W4 GEMMs and the MoE run 8-row steps); slots up to it
  int step_rows() const { return step_rows_; }
  // the block rows one drafter pass holds (16; a 32-row step's 32)
  int pass_rows() const { return std::max(step_rows_, kernels::kStepRows16); }
  // the rows the step buffers hold (head, MLP, mixers, drafters): pass_rows() or the load's prefill_short, whichever is
  // more (16 unless 32-row steps or 17-32-row short prefill calls)
  int rows_cap() const { return rows_cap_; }
  // One MTP layer reused recursively: pass 1 plus further chained passes, up to this many drafts.
  // DFlash / DSpine still cap at the checkpoint's block - 1 (7 or 15).
  static constexpr int kMtpDepthMax = 6;
  static_assert(kMtpDepthMax >= 6 && kMtpDepthMax <= 15, "qwen35 MTP depth 6 fits a segment and kDepthMax 15");
  // the most drafts one draft call returns: MTP kMtpDepthMax, DFlash the checkpoint's block - 1
  int draft_depth_max() const;
  // both drafters loaded (ModelOptions:mtp_from); which one drafts a call for `seqs` sequences (true: MTP), and
  // the most drafts it returns then
  bool dual_drafters() const { return mtp_ != nullptr && dflash_ != nullptr; }
  int mtp_from() const { return dual_drafters() ? opt_.mtp_from : 0; }
  bool mtp_drafts(int seqs) const { return mtp_ != nullptr && (dflash_ == nullptr || seqs >= opt_.mtp_from); }
  int draft_depth_max(int seqs) const;
  // Coupled drafts for a sampled request on both drafters (Mtp::set_noise, DFlash::set_noise): the slot's next drafts
  // pick argmax of logit / T + the target's Gumbel noise per output position; inv_t 0 = argmax.
  void set_draft_noise(int slot, uint64_t key, uint64_t pos, float inv_t);
  // Drafts after a verified step: call after step(ids, rows) + accept(keep). next[r] = the token after row r of that step
  // (its picks: the accepted drafts, then the correction or bonus token; rows past keep: any token). Pass 1 runs over all
  // `rows` rows (their drafter KV: rows past keep are rewritten by later passes), then depth - 1 passes chained from row
  // keep - 1 (each on the previous pass's hidden and pick, one position further). Returns `depth` drafts (fewer near the
  // capacity). Waits.
  std::vector<int32_t> draft(const int32_t *next, int rows, int keep, int depth);
  // Drafts after a prefill: pass 1 on the prompt's last row with `first` (the first output token) as its next token
  // (its drafter KV written there), then the chain. Waits. seqs (both drafters): the sequences of the step the
  // drafts are for (which drafter drafts).
  std::vector<int32_t> draft_after_prefill(int32_t first, int depth, int seqs = 1);
  // Card 0 GPU ms of the last draft call (profile on): pass 1, the chained passes.
  double draft_ms() const { return draft_ms_; }

 private:
  static void llama_prefill(ModelOptions &o);  // arch llama: the prefill paths built at head_dim 128 (model.cpp)
  static void flint_options(ModelOptions &o);  // arch flint: the prefill paths its kernels take (model.cpp)
  // the decoder norms (kernels::rms_norm plain): 0 = (1 + w), 1 = arch llama's bf16(w x bf16(y)), 2 = arch flint's bf16(w x y)
  int plain_norm() const { return cfg_.flint ? 2 : !cfg_.norm_offset ? 1 : 0; }
  struct DSeg {  // a segment the drafter runs: its slot, the position of its pass-1 row 0, its kept rows
    int slot = 0;
    int64_t p0 = 0;
    int keep = 1;
  };
  // `code` = the verify's seg_rows word (rows per segment, or kernels::seg_het: unequal segments); depth per
  // segment (one entry = every segment). DFlash runs each contiguous run of segments with the same rows and drafting or
  // not as its own passes; MTP's passes go over every segment (to the deepest), each keeping its own drafts.
  std::vector<std::vector<int32_t>> dflash_drafts(const std::vector<DSeg> &segs, int code, const int32_t *anchor, const std::vector<int> &depth);
  std::vector<std::vector<int32_t>> run_drafts(const std::vector<const bf16 *> &h, int code, const std::vector<DSeg> &segs, const int32_t *next,
                                               const std::vector<int> &depth);
  std::vector<std::vector<int32_t>> mtp_run(const std::vector<const bf16 *> &h, int code, const std::vector<DSeg> &segs, const int32_t *next, int depth);
  struct Card;
  void run_layers(int rows, int seg_rows, int pos_bound);
  // the step's launches (embedding excluded) on both cards; ev: the profile's event lists (null: none)
  struct StepEvents;
  void submit_layers(int rows, int seg_rows, int pos_bound, StepEvents *ev);
  // arch flint: step graphs (StepOptions::graphs)
  struct Graphs;
  std::unique_ptr<Graphs> graphs_;
  bool graph_step() const;
  void run_graph(int rows, int seg_rows, int pb = -1);  // pb = the step's attention bound (bucketed); -1 = cap
  int graphs_recorded() const;
  // one sequence's host state; pos_ .. pf_last_row_ below mirror the current slot's (save_seq / load_seq)
  struct Seq {
    int64_t pos = 0;
    int last_rows = 0, pending_accept = 0, parity = 0, pf_last_row = 0;
    int pf_hx = -1;  // the row of Card::pf_hx that holds its last prefilled row's final-normed hidden (-1: the head's)
    bool pf_committed = false;
    bool broken = false;  // a prefill call of it was aborted (its live state is undefined until reset / restore)
    int64_t rope_delta = 0;  // the M-RoPE delta after the processed rows (set_vision; 0 = text only so far)
    // the next step's tree words (step_tree) and the last accepted tree path (the next record's kSegPath)
    uint64_t tree = 0, tree_aux = 0, path = 0;  // (64-bit words: trees of up to 16 rows)
  };
  void save_seq();
  void load_seq();
  void reset_slot(int s);
  std::vector<Seq> seqs_;
  int slot_ = 0;
  std::vector<int> batch_slots_;  // the last verify's segments (slots, row order)
  int batch_rows_ = 0;
  int batch_code_ = 0, batch_total_ = 0;  // the last verify's seg_rows word and rows
  std::vector<float *> lp_dev_, lp_raw_;  // logprob records per card (device), the targets' logits (pinned)
  float *lg_host_ = nullptr;      // pinned [cards][Head:kRows][vocab / cards], the host copy of a step's logits
  int32_t *rec_host_ = nullptr;   // pinned: the step's segment records [kSegs16][kSegRec], then the drafter passes'
  void pf_chunk(const int32_t *ids, int m, bool head_all, const std::function<void(int64_t, int)> &on_rows, bool last, int32_t next_tok);
  void pf_start(const int32_t *ids, int m, bool head_all, const std::function<void(int64_t, int)> &on_rows, bool last, int32_t next_tok);
  bool pf_step(int layers);
  struct PfRun {  // the started chunk (kept across the parts of a call)
    std::vector<int32_t> ids;
    int m = 0, l = 0, slot = 0;  // slot: the call's (another slot may be current between its parts)
    bool head_all = false, last = false, active = false;
    int32_t next = -1;
    int64_t snap = -1;  // the position this chunk captures a snapshot at (snap_arm), -1 none
    int64_t snap_rope = 0;  // the M-RoPE delta at snap
    bool rope_rows = false;  // the chunk's rows carry their own rope rows (an image, or a delta: pf_rope_host_)
    std::function<void(int64_t, int)> on_rows;
    struct Seg {  // one sequence's rows of a call that carries several (segs empty: the call is slot's alone)
      int slot = 0, row0 = 0, m = 0;
      bool rope = false;  // its rows carry their own rope rows (the slot's rope delta is not 0)
      int64_t pos = 0;
      int32_t next = -1;
      int gp = 0, acc = 0;  // its GDN parity for the call and the accept word of its first rows
    };
    std::vector<Seg> segs;
  };
  PfRun pf_;
  void pf_tail_many(double split_ms);  // the end of a call of several sequences (pf_step)
  // rms_norm over the chunk's rows on card c: with a pending split sum (pend) the card's own rows of `add` first, then
  // proj(lo, hi) of them (the consumer's first GEMM in row parts), the wait for the peer's rows, then the rest likewise
  //. Without pend proj is not called (the consumer projects every row itself).
  // codes_only: xn's only readers are W4A8 GEMMs (the norm writes its codes, not its bf16 values; pf_codes_only)
  // mode (with pend; the sum finished as SplitRows::owner, nothing copied back): owner = the card norms its own rows only
  // and shares their codes (the peer's arrive the same way; needs codes_only), so the residual's peer rows go stale on
  // this card; restore = the card norms its own rows, shares their residual, then norms the peer's rows from theirs.
  enum class PfNorm : uint8_t { both, owner, restore };
  void pf_norm(int c, int m, const bf16 *add, const bf16 *w, uint32_t pend, const std::function<void(int, int)> &proj = {}, bool codes_only = false,
               PfNorm mode = PfNorm::both);
  bool pf_codes_only(int m) const;
  void pf_wait_pending();  // both cards wait for a pending split sum's peer rows

  Fabric &fab_;
  Config cfg_;
  ModelOptions opt_;
  int64_t pos_ = 0, cap_ = 0;
  double yarn_factor_ = 1.0;
  std::vector<int64_t> tier_limit_, tier_base_;  // per tier its positions and first row in the rope table
  std::vector<double> tier_factor_;
  bool mrope_cfg_ = false;  // the checkpoint's interleaved M-RoPE (image input); the tiers use the delta without it
  int last_rows_ = 0, pending_accept_ = 0, parity_ = 0, mix_parity_ = 0, pf_parity_ = 0;
  int gdn_accept_max_ = 4;      // the step's largest accept word (GdnState:accept_max)
  bool gdn_commit_ = false;     // this step's GDN rows are final (GdnState:commit)
  bool tree_attn_ = false, tree_gdn_ = false;  // the step holds a tree (attention) / a tree or a path (GDN)
  AttnQFn attn_q_fn_;  // on_attn_q (diagnostic)
  // ModelOptions::select_self: a slot's measurement and the device tables of its counts
  struct SelfSelect {
    int steps = 0;                // selected steps measured so far
    std::vector<double> lost;     // [units][SelectShare::kFractions], summed over those steps
    std::vector<float> weights;   // [units], empty until select_self steps were measured
  };
  std::vector<SelfSelect> self_;  // per slot
  std::vector<std::vector<int32_t>> self_counts_;  // per card: the host copy of Card::sel_slots
  std::vector<uint32_t> self_keys_;                // a measured unit's score keys (host)
  int self_pages_ = 0;                             // the common count the tables were written for
  int attn_rows() const;                           // attention layers
  void self_reset(int slot);                       // a new sequence in the slot: measure again
  void self_write(int slot);                       // the slot's counts -> the host copies -> the cards
  void self_sync(int pages);                       // the tables follow the common count
  void self_measure(const std::vector<SeqRows> &segs, std::span<const int64_t> pos_before, int pages);
 public:
  // the weights slot's sequence measured, [attention layers][kv heads]; empty while it is still measuring
  const std::vector<float> &select_self_weights(int slot) const { return self_.at(static_cast<size_t>(slot)).weights; }
 private:
  bool last_tree_ = false;  // the last step held a tree segment (any slot)
  std::vector<uint8_t> batch_tree_;  // per segment of the last step, its rows were a tree
  std::vector<std::vector<int>> draft_par_;  // per slot, draft_parents()
  void note_tree(int slot, size_t drafts, int seg = 0);  // the last DFlash pass's tree (its segment seg) for the slot's drafts
  void accept_counts(const std::vector<int> &n, const std::vector<const int *> *paths=nullptr);  // paths already validated
  void accept_paths(const std::vector<int> &n, const std::vector<const int *> &paths);  // accept_batch(n, paths) before close_pages
  // One card's KV caches of the attention layers (the accept's launches run over all of them), and the first of those
  // layers: they share attn_.shape_of(card, layer0).
  struct AttnCaches {
    std::vector<kernels::GqaCache> caches;
    int layer0 = -1;
  };
  AttnCaches attn_caches(size_t card) const;
  // Page selection: a page's key bounds are written once, when its sequence's position leaves it
  // (kernels::gqa_bounds_page; a decode step writes none). step_positions(): the position of each segment of the last
  // step, taken before its accept; close_pages(those): after the accept's counts and a tree's K / V move.
  std::vector<int64_t> step_positions();
  void close_pages(std::span<const int64_t> pos_before);
  bool pf_committed_ = false;
  uint32_t pf_seq_ = 0;   // pair_split_rows sequence numbers
  uint32_t pend_seq_ = 0; // a split sum whose peer rows the consumer has not waited for yet (0: none)
  double load_s_ = 0;
  std::unique_ptr<load::DenseStore> dense_;
  std::unique_ptr<trained_qsa::Device> trained_qsa3_;  // optional trained index bank
  int trained_qsa_prefill_min_ = 0;
  std::array<uint64_t,2> trained_prefill_reported_{};
  std::vector<int> trained_qsa_layer_;  // model layer -> bank index; -1 is stock dispatch
  static constexpr int kTrainedQsaShort=8192;
  std::vector<int64_t> qsa_covered_;  // contiguous logical token count; -1 means disabled/incomplete
  void qsa_invalidate_snapshots(int slot,int64_t retained=-1);
  uint32_t qsa_stage_rows_=0;
  bool qsa_step_sparse_=false;      // some segment of the step runs the trained sparse walk
  uint32_t qsa_dense_rows_=0;       // rows of the step kept on the dense walk (plan count 0)
  uint64_t qsa_sparse_slots_=0;     // slots whose segment is sparse this step (route log)
  std::unique_ptr<Head> head_;
  std::unique_ptr<HeadCands> cands_;  // the sampling candidates (logits_cands)
  std::unique_ptr<Mlp> mlp_;
  std::unique_ptr<flint::MoeSet> moe_;  // arch flint: the MoE layers (Config::moe)
  int ring_rows_ = 0;                   // arch flint: the rows a sliding layer's KV ring holds beyond its window
  GdnSet gdn_;
  AttnSet attn_;
  std::vector<std::unique_ptr<Card>> cards_;
  StepProfile prof_;
  PfProfile pf_prof_;
  StepOptions sopt_;
  void apply_head(const StepOptions &o);  // StepOptions:head / head_tiles -> Head:set_mode
  double submit_ms_ = 0;
  std::chrono::steady_clock::time_point t_submit_;
  // the drafter
  std::unique_ptr<Mtp> mtp_;
  std::unique_ptr<DFlash> dflash_;  // 
  int rows_max_ = kRows;            // a segment's rows at most (GdnState:log_rows)
  int step_rows_ = kernels::kStepRows8;  // a step's rows at most (step_rows())
  int rows_cap_ = kernels::kStepRows16;  // rows_cap()
  std::vector<int> feat_of_;        // per target layer l, the DFlash feature its output is (-1: none)
  int pf_last_row_ = 0;                // the head xn row of the last prefill's last token
  std::vector<int32_t *> mtp_pos_;     // [card] device: the passes' segment records [pass][kSegs16][kSegRec]
  int32_t *mtp_drafts_ = nullptr;      // card 0 device: [pass][kStepRows32] drafts
  double draft_ms_ = 0;
  // the KV page table (host side) and its device copy per card [slots][cap / 64]
  model::KvPages kvp_;
  std::vector<int32_t *> kv_table_;
  std::vector<size_t> kv_bytes_;
  double vram_free_before_kv_ = -1;
  void kv_need(int slot, int64_t tokens);  // kv_reserve or throw, then the table upload
  void kv_upload();                        // the host table's changed range to every card (waits)
  // every paged KV buffer of card c; a page copy [n src][n dst] on every card (waits); the copy-on-write of a slot's
  // shared pages it is about to write at pos (and pos - 1)
  std::vector<kernels::KvPageBuf> kv_page_bufs(int c) const;
  void kv_copy_pairs(const std::vector<int32_t> &pairs);
  void kv_cow(int slot, int64_t pos);
  uint64_t kv_cow_pages_ = 0;  // copy-on-write events (checks)
  void kv_alloc(int64_t pool_pages);       // (re)allocate the pool's K / V for every attention layer + the drafter
  size_t kv_page_bytes(int c) const;
  size_t ring_bytes_of(int c) const;
  double vram_free_gib();                  // the tighter card's free VRAM (sysman; -1 without ModelOptions::loader)
  std::vector<std::vector<void *>> kv_mem_;  // [card] the KV buffers (freed by kv_alloc)
  // KV lending (ModelOptions::kv_lend)
  struct LendMat {
    const uint8_t *w = nullptr;
    int n = 0, k = 0;
  };
  std::vector<std::vector<LendMat>> lend_mat_;    // [card][item]
  std::vector<std::vector<uint8_t *>> lend_buf_;  // [card][buffer]: the pool's K / V buffers, the same order on every card
  model::KvLendRegions lend_;
  std::vector<char> lend_item_;  // [item] lent now
  int lend_keep_pages_ = 1;
  LendStats lend_st_;
  std::string lend_note_ = "off";
  // the target's decode int4 matrices registered for W4A8 (StepOptions::w4_a8 switches it)
  std::vector<const uint8_t *> a8_mats_;
  bool a8_rows_ = false;
  void a8_rows_init();
  void lend_forget();                  // every kept layout forgotten (before the pool's buffers are freed)
  void lend_layout();                  // after kv_alloc: the regions of the new pool, lent at its limit
  void lend_apply(int target);         // lend exactly the regions usable at `target`; the pool's limit follows
  bool lend_raise(int short_pages);    // pages past the limit back to the KV
  // the snapshot store (host buffers allocated on first use, kept for reuse)
  struct SnapRec {
    int64_t pos = 0;
    int qsa_owner=-1;  // exact complete groups still resident in this logical slot
    int64_t rope_delta = 0;
    int parity = 0;
    bool used = false;
    std::vector<void *> host;  // [card]
    std::vector<sycl::event> saved, read;  // [card]: its device-to-host copy done / its last restore's reads done
  };
  // snapshots move through two device staging buffers per card: state <-> staging on the compute queue, staging <-> host
  // as one copy on snap_q_ (snap_copy)
  std::vector<std::optional<sycl::queue>> snap_q_;  // [card]
  std::vector<std::array<void *, 2>> snap_stage_;  // [card]
  std::vector<std::array<sycl::event, 2>> snap_stage_ev_;  // [card]: each staging buffer's last user done
  int snap_stage_next_ = 0;
  std::vector<SnapRec> snaps_;
  // set_vision: the image store, the M-RoPE sections, the 1-D rope table on the host (the rope rows of a call are
  // composed from it), a chunk's rope rows and its device ids
  VisionStore *vis_ = nullptr;
  model::Mrope mrope_;
  std::vector<uint16_t> rope_host_;
  std::vector<uint16_t> pf_rope_host_;
  std::vector<int32_t> vis_ids_;
  int64_t capt_rope_ = 0;
  bool capt_qsa_ = false;  // the captured call appended trained-QSA keys from its start (covered==pos)
  void pf_rope_rows(const int32_t *ids, int m);  // pf_start: pf_.rope_rows / snap_rope, pf_rope_host_, the slot's delta
  int snap_cap_ = 0;
  std::vector<int32_t *> fork_pages_;  // [card] device: a fork's page pairs [src slot_pages][dst slot_pages]
  // the in-call capture (snap_arm / snap_take): a device buffer per card in snap_bytes' layout, its last host copy, the
  // armed position and the capture waiting for snap_take
  std::vector<void *> cap_buf_;
  std::vector<sycl::event> cap_ev_;
  int cap_arm_slot_ = -1;
  int64_t cap_arm_pos_ = -1;
  struct Capture {
    bool ready = false;
    int slot = -1, parity = 0;
    int64_t pos = -1;
  } capt_;
  size_t cap_gdn_off(int c, int l) const;  // layer l's S in the capture buffer (its window right after)
  void snap_copy(int slot, SnapRec &r, bool save);
  // arch flint: the sliding layers' ring windows in a snapshot (model.cpp)
  static constexpr int kRingSnapPages = 10;  // >= (window 513 - 1) / 64 + 2
  size_t ring_snap_bytes(int c) const;
  sycl::event ring_snap_copy(int c, sycl::queue &q, int slot, int64_t pos, char *buf, bool save, sycl::event dep);
};

}  // namespace xenithium::arch::qwen35
