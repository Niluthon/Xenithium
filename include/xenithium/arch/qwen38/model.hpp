// Xenithium · qwen38 whole model on two cards: the load, the per-sequence state and one decode
// step through every layer.
//
// A step (rows = 1..4 consecutive tokens of one sequence at positions pos .. pos + rows - 1; or, with
// step_batch, up to 4 sequences' segments of equal rows, each on its own slot):
//
//   embed                                       ids -> residual, every stream (Head)
//   per layer l:
//     [l == PLE layer: materialise the pending combine, then the PLE block adds to the residual]
//     attention mixer (combine the previous MoE output) -> x_a
//     QSA (full-attention layers) or GDN (linear) on x_a -> y_a         two-card sum inside
//     MLP mixer (combine y_a) -> x_m
//     MoE on x_m -> y_m                                                  two-card sum inside
//   final mixer (combine the last y_m) -> lm_head -> greedy pick         (Head)
//
// Every row of a step() is a real token (teacher forcing / plain decode): the next step accepts
// all of them. Speculative rows go through verify() + commit(n); the MTP drafter's
// passes (drafter.cpp) run on the same kernels between target steps.
//
// Placement: the non-expert weights (the arch's dense plan), the KV caches at the asked context,
// the states and scratch are allocated first; the hot-expert count is what the live free VRAM
// leaves after them and the reserve. Cold experts stay in pinned host memory, read in place.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "xenithium/arch/common/head_cands.hpp"
#include "xenithium/arch/qwen38/arch.hpp"
#include "xenithium/arch/qwen38/expert_store.hpp"
#include "xenithium/arch/qwen38/gdn.hpp"
#include "xenithium/arch/qwen38/hc.hpp"
#include "xenithium/arch/common/dflash.hpp"
#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/arch/qwen38/moe.hpp"
#include "xenithium/model/route_counts.hpp"
#include "xenithium/model/route_record.hpp"
#include "xenithium/arch/qwen38/ple.hpp"
#include "xenithium/arch/qwen38/ple_table.hpp"
#include "xenithium/arch/qwen38/qsa.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/kv_lend.hpp"
#include "xenithium/model/placement.hpp"
#include "xenithium/model/kv_pages.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch {
class VisionStore;
}

namespace xenithium::arch::qwen38 {

// Rows per group of the many-row head for prompt logprobs (Model:prefill_logprobs): a cap; the group takes as many rows
// as moe.y holds (105 at 512-row chunks).
inline constexpr int kPfLpRows = 512;
// The load: expert layers the reader runs ahead of the packing (host memory permitting), dense groups read ahead of
// their packing, threads packing the experts (memory-bound beside the reads)
inline constexpr int kExpertReadAhead = 8;
inline constexpr int kDenseReadAhead = 16;
inline constexpr int kExpertPackThreads = 8;
// The VRAM a first-use graph recording keeps free on the tighter card (graph_room)
inline constexpr double kGraphRoomMiB = 64;
// Below the room a new graph first costs kGraphRoomEvict others, so the driver's pool gains a graph's memory of slack
// per recording, until the population is kGraphKeepNum / kGraphKeepDen of what it was when the room ran out (at least
// kGraphKeepMin); from there it is held: one out per new graph.
inline constexpr int kGraphRoomEvict = 2;
inline constexpr int kGraphKeepNum = 3, kGraphKeepDen = 4, kGraphKeepMin = 16;
// kMaxCards: the most cards the model runs on (TP4); per-card arrays hold this many entries. kPfSplitMinRows: a prefill
// call's two-card sums (GDN, QSA, MoE) split over the cards from this many rows (the generic kPairSplitMinRows is 64),
// with the same bits; calls of fewer rows stay on the rows wire.
inline constexpr int kMaxCards = 4;
inline constexpr int kPfSplitMinRows = 24;
// A prefill call's rows by owner (prefill.cpp, "rows by owner"). From kPfOwnerMinRows rows each row's residual streams
// live on one card; below, every card holds every row (the copies' fixed cost exceeds the mixer rows saved). The call's
// rows then run in lanes (runs of rows that go through each mixer and each GDN / QSA as a call of their own): one under
// kPfTwoLaneRows rows, two under kPfFourLaneRows, else kPfLanesMax; a lane ends at a position that is a multiple of
// kPfLaneAlign (where a call may end and keep every bit: the GDN scan's 64-row chunks), so a call that starts between
// two of them (a live resume) may get fewer lanes. The MoE runs as one call over every lane under kPfMoeTwoCallRows
// rows, else as two.
inline constexpr int kPfOwnerMinRows = 128;  // (= kernels::kPairSplitSendMinRows: the modules send their partials from there)
// The layer-ahead fetch (ModelOptions::prefill_fetch) from this many rows in a call: a shorter call's layer computes for
// less time than its copies take, and the copies hold up its sums' sends on the one copy engine (2,048-row calls: 1,806
// -> 1,635 tok/s with the fetch, measured).
inline constexpr int kPfFetchMinRows = 4096;
// The blobs per layer a 4,096-row call's fetch moves (in proportion to the call's rows, up to prefill_fetch): what the
// host link carries per card while both cards compute a layer (a measured sweep).
inline constexpr int kPfFetchPer4K = 96;
// The fetch's buffers per card, layer l's in buffer l % kPfFetchBufs: with three, layer l + 2's fetch starts behind
// layer l's MoE (its buffer's last reader was layer l - 1's), two layers of compute before its own MoE.
inline constexpr int kPfFetchBufs = 3;
// Lane thresholds by a sweep of one call from position 0 (W4A8 experts, the MoE and GDN inputs by owner): one lane
// up to 639 rows (384 / 448 / 576 rows 78 / 91 / 94 -> 68 / 75 / 89 ms against two lanes; 640 rows: 95 against 87),
// two up to 2,047 (1,536 rows 172 -> 161 ms against four; 1,024 rows on four 153-155 against 114-116).
inline constexpr int kPfTwoLaneRows = 640;
inline constexpr int kPfFourLaneRows = 2048;
inline constexpr int kPfMoeTwoCallRows = 2560;
inline constexpr int kPfLaneAlign = 64;
inline constexpr int kPfLanesMax = 4;  // (a card's split buffers are wired per lane)
// Prompt-call graphs (ModelOptions::prefill_graphs). Calls of fewer than kPfGraphMaxRows rows only: from there the
// split sums' copies run on the side queue beside the kernels (sends, lanes), and a graph replays its branches one
// after another (measured: the copies in series cost what the replay saves at 128-256 rows, more above). A call
// shape is recorded the kPfGraphRecordAt-th time it runs (one-off shapes never pay a recording); at most kPfGraphsMax
// prompt-call graphs are kept (least recently replayed out first), each with its own split flag lines.
inline constexpr int kPfGraphMaxRows = 128;
inline constexpr int kPfGraphRecordAt = 2;
inline constexpr int kPfGraphsMax = 32;
inline constexpr int kPfGraphSeenMax = 4096;  // shapes counted but not recorded yet (the count restarts past it)
// A decode step's rows by owner (model.cpp, "a decode step's rows by owner"): at four cards, steps of at least this
// many rows run each row's residual mixers on one card. Below it every card runs every row.
inline constexpr int kDecOwnerMinRows = 6;
// The step bound (tokens) from which a step by owner selects the QSA indexer's blocks by owner rows.
inline constexpr int kDecSelOwnerFrom = 8192;
// The first QSA bound bucket. A step whose rows all sit below 2,048 tokens (512 complete blocks, the selection's size)
// selects every block, so its graph carries no score and no select launch (k::QsaStep::every).
inline constexpr int kDecEveryBound = 2048;

struct ModelOptions {
  int64_t context = 8192;             // KV capacity of each sequence slot (rounded up to 64)
  // YaRN per request past the trained positions (as qwen35's): the rope table holds one sub-table per tier (trained x
  // 2^k positions at that length's factor; the last tier = the whole context at its factor, from row 0) and a sequence's
  // segment records carry its tier's first table row (kSegRope, set_rope_tier), so a short request on a 1M server gets
  // the bits of a server at the trained context. false = one factor for every request.
  bool yarn_tiers = true;
  // Sequence slots: each has its own KV, GDN state, PLE ring and host context, allocated
  // before the placement (the experts get what is left). A step may run up to 4 of them at once
  // (step_batch); the drafter keeps a KV per slot.
  int slots = 1;
  // The most rows a decode step carries, 0 (= 32) | 32 | 64. 64 needs four cards (a step above 32 rows runs by owner:
  // 16 rows per card) and the W4A16 experts without rotation, and takes ~0.4 GiB more per card (the MoE's per-slot
  // partials are laid out by this, the drafter's step buffers too).
  int step_rows_cap = 0;
  kernels::QsaKv kv = kernels::QsaKv::i8;  // target KV (int8 + one scale per token and kv head)
  // How the two cards of a kv head (four cards: 2 kv heads) store its KV (kernels/attention/qsa.hpp): heads = both all of
  // its tokens; tokens = each every other token (a quarter of the K / V per card; the partner's q heads replicated, one
  // pair exchange per attention layer, a numerics change). The MTP layer's KV follows. Four cards only.
  kernels::QsaKvSplit kv_split = kernels::QsaKvSplit::heads;
  int64_t hot = -1;                   // hot experts per card; -1 = measured from free VRAM
  double reserve_gib = 0.25;  // VRAM left free per card after the experts (every allocation precedes them)
  int threads = 16;                   // load threads (dense + expert packing)
  int ple_threads = 32;               // PLE table row readers
  bool verify_experts = false;        // hash every expert blob at load and verify after
  std::vector<double> hot_rank;       // routing counts [layer][expert] for the hot placement; empty = even split
  // The hot-expert ranking recorded while serving (model/route_record.hpp). Read at load when it exists (the placement,
  // unless hot_rank is given), counted on the cards from the decode steps' routes and written every rank_every_s
  // seconds of decoding and at destruction. Empty = no recording.
  std::string rank_file;
  double rank_every_s = 60;
  // The decode MoE's CPU path for cold experts (kernels/moe/moe_cpu.hpp): a host thread pool per card computes part of
  // each decode layer's cold experts from pinned RAM while the card does the rest. A numerics change (KL-gated) and
  // host-CPU dependent: off by default. mode "off" | "auto" (AVX-512 BF16, else AVX2 + FMA) | "avx512" | "avx2"; a CPU
  // without the ISA leaves it off (load_stats().moe_cpu says why).
  struct MoeCpu {
    std::string mode = "off";
    int threads = 8;              // per card, the poller included
    std::vector<int> cores;       // first core per card (empty = 0 and threads: one block of cores per card; -1 = unpinned)
    // Batch invariance: the pool's experts are a fixed pct % of each layer's cold experts (interleaved in ranking
    // order, chosen at load), and a ROW hands them over when it routes to >= min_cold of them in a layer, so a
    // sequence's tokens do not depend on its batch-mates (min_rows > 1 brings that dependence back).
    int pct = 70;                 // percent of each layer's cold experts in the pool's fixed set
    int min_rows = 1;             // steps with fewer rows keep their cold experts on the cards
    int min_cold = 2;             // per row and layer: a row routing to fewer of the set's experts keeps them on the cards
    int timeout_ms = 500;         // the card's wait bound per layer (then it computes them itself and the path goes off)
    int spin_us = 20000;          // the pool spins this long after a request before it naps
  } moe_cpu;
  // Numerics switches, for KL against the shipped path:
  std::vector<std::string> bf16_tensors;  // int8 plan matrices whose name contains one of these load as bf16
  // Int8 plan matrices whose name contains one of these load as int4 g128 by RTN from the BF16 checkpoint
  // (DenseSpec::Store::w4r); consumers with a W4 path only (the GDN in_proj_qkv | z and out_proj: gdn_decode / gdn_rows
  // take W4Mat). A numerics switch (KL-gated), off by default.
  std::vector<std::string> w4_tensors;
  bool w4_mse = false;  // w4_tensors' scales by the MSE clip search (kernels::quantize_w4_row), not plain RTN
  // GDN projections from the directory's int4 GPTQ tensors (`*.qweight` / `.scales` / `.qzeros`, written by
  // `xenithium-bake gdn4-calib`) instead of int8 from BF16: bit 1 = in_proj_qkv | z, bit 2 = out_proj (0 = none). A
  // numerics switch, off by default (server --gdn-w4).
  int gdn_w4 = 0;
  bool moe_down_bf16 = false;             // routed experts' down outputs rounded to bf16 (MoeShape::down_bf16)
  // Step graphs: each step shape is recorded once per card as a SYCL graph and replayed
  // (one submission per card instead of ~800 launches). Keys: rows, the parities, and the QSA grid
  // bound bucket. Profiled steps always run eagerly (they need per-launch events).
  bool graphs = true;
  // Prompt-call graphs (prefill.cpp, "prompt-call graphs"): a prompt call of fewer than kPfGraphMaxRows rows replays
  // its layers from a recorded graph per card once its shape has run kPfGraphRecordAt times. The same bits as the
  // launches; the call's host submission (~1.5-2.6 ms of a 16-96-row call on two or four cards) goes. Off by default:
  // a shape costs 10-60 ms to record and pays back only after ~20 calls of it.
  bool prefill_graphs = false;
  int moe_gu_ksplit = 0;                // MoE gate_up k split (0 = the kernel default 8; probe --moe-gu-ks)
  int moe_dn_ksplit = 0;                // MoE down k split (0 = the kernel default 2); 1 = a null variant:
                                          // same precision, another fp32 summation order
  // Card balance of the MoE: expert TP, each card 1/2 of every routed expert (a load-time
  // layout: per-card half blobs; the default). false = EP2, each card owns half the
  // experts (needed for moe_down_bf16).
  bool moe_tp = true;
  // MTP drafter: the drafter checkpoint directory (runtime/mtp-int4-g32); empty = none.
  // Loaded before the hot-expert placement (its 512 experts sit in VRAM, TP-split like the target's).
  std::string mtp_dir;
  // The drafter's lm_head on a draft vocabulary: token ids [0, K) + the 128 ids from the first EOS id rounded down to
  // 64 (the special tokens), the target's lm_head rows, split evenly over the cards; a draft outside it is never
  // proposed (the verify stays exact). 0 = the whole vocabulary.
  int mtp_vocab = 0;
  // The draft vocabulary head's rows as int4 g128 on the W4 GEMM: half the bytes of each draft pass's head. From the
  // target directory's GPTQ `lm_head.qweight` / `.scales` / `.qzeros` (xenithium-bake head4-calib) when present, else
  // round-to-nearest from the BF16 lm_head at load (DenseSpec::Store::w4r). Drafts only (the verify stays exact):
  // greedy output unchanged, acceptance may move. Needs mtp_vocab > 0. A probe switch (probe --mtp-head-w4); the server
  // keeps int8 rows.
  bool mtp_head_w4 = false;
  // The drafter layer's KV dtype.
  kernels::QsaKv mtp_kv = kernels::QsaKv::bf16;
  // The DFlash 2 block drafter (arch/common/dflash.hpp, shared with qwen35) instead of the MTP layer: a
  // `DFlash2DraftModel` checkpoint for this target; empty = none. Loaded before the placement (its weights, rings and
  // scratch come out of the hot experts' VRAM). Its draft vocabulary: mtp_vocab.
  std::string dflash_dir;
  int dflash_dense_bytes = 1;     // its matrices: 1 = int8 per row, 2 = bf16
  int dflash_ctx_defer = 64;      // DFlashOptions::ctx_defer: a context-only pass's rows wait in a per-slot backlog
  double dflash_conf_cut = 0.02;  // DFlashOptions::conf_cut: one sequence keeps drafts while their confidence product holds
  bool dflash_profile = false;    // DFlashOptions::profile (diagnostic: stage waits, no pass graphs)
  // Draft trees: DFlashOptions::tree (best-first tree nodes per pass, 1..7: a tree of the drafter's 8-row block; 0 =
  // chains), tree_cut, tree_seqs (sequences of a pass that draft trees), tree_temp (0 = by drafter). The verify then
  // runs tree rows (QSA, PLE, GDN: rows.hpp kSegTree) and the accept moves the walked path's state (commit_batch with
  // paths). Refused with the MTP layer beside it (its pass 1 writes KV over the verify's rows).
  int dflash_tree = 0;
  double dflash_tree_cut = 0;
  int dflash_tree_seqs = 1;
  double dflash_tree_temp = 1.5;
  // The target state a feature is (the capture point): 0 = the completed output of layer id (the multi-stream residual
  // right after layer id + 1's attention mixer combined it), 1 = one layer earlier (after layer id's attention mixer).
  // The checkpoint's features are completed, zero-based layer outputs: 0.
  int dflash_feat_shift = 0;
  // Both drafters: with dflash_dir AND mtp_dir set, the DFlash drafter drafts steps of fewer than mtp_from sequences
  // and the MTP layer steps of mtp_from or more; each step also feeds the other drafter (DFlash drafting: MTP's pass 1
  // writes its KV over the verified rows; MTP drafting: the verified rows go into DFlash's context, deferred), so
  // either can draft the next step. 0 = one drafter (both set: refused).
  int mtp_from = 0;
  // Prefill: rows per prefill chunk (a multiple of 4; sizes the chunk buffers, allocated
  // before the placement: ~65 KB per row per card).
  int prefill_chunk = 512;
  // Prefill MoE numerics: true = W4A8 (x and h quantised to int8 per row and group, s8 x s4 DPAS); false = W4A16 (bf16
  // activations, the numerics the bit-exact gates check). The served model runs W4A16 only: the default is false (W4A8
  // stays as an A/B switch for the probes and the calibration tool).
  bool prefill_moe_a8 = false;
  // W4A16 prefill experts with the int4 widen's offset outside the DPAS operand (MoeShape:w4_bias): fewer instructions
  // in gate_up, not the same bits (a numerics change, opt-in). No effect with prefill_moe_a8.
  bool prefill_moe_bias = false;
  // Prefill dense numerics: true = W8A8, every many-row int8 dense GEMM (HC down / up, GDN qkvz /
  // out, QSA qkv / o, the shared expert, PLE kv, the drafter's prefill pass; not the lm_head: prompt logprobs
  // must equal the 4-row head bit for bit) takes
  // its activations as int8 per (row, 128-k group) on s8 x s8 DPAS (the experts' quantiser rule); false =
  // bf16 activations (W8A16, the decode path's numerics and the bits). A numerics change: the
  // KL gates decide the default. Decode stays W8A16 (its GEMMs read weights, not DPAS).
  bool prefill_dense_a8 = false;
  // With prefill_dense_a8: one activation scale per row over the whole k instead of per 128-k group (PfA8::row: one
  // int32 sum, no fold, 4-tile bodies), and the HC mixer's GEMMs stay W8A16. Server `--prefill-dense w8a8-row`.
  bool prefill_dense_a8_row = false;
  // The prefill's two-card sums on the bf16 wire (kernels::PairSplit::bf16 / PairRowsWire::bf16: the peer's rows sent
  // as bf16, half the link bytes; both cards sum bf16(own) + bf16(peer) in fp32). The GDN and QSA sums of every prefill
  // call (the drafter's prefill pass included); the MoE sums stay fp32 (its short calls' MoeRowsWire has no bf16 form,
  // and a row's bits must not depend on the call size). A numerics change (KL-gated); false = fp32 partials. Server
  // `--prefill-wire` (qwen38's auto = fp32).
  bool prefill_wire_bf16 = false;
  // The int8 dense matrices' prefill calls of >= this many rows on oneDNN (kernels::pf_dense_set_onednn; the engine's
  // bits wherever oneDNN runs them under W8A16). Process-wide, set at load; -1 = leave the process setting (the probes:
  // XEN_PF_DENSE_ONEDNN), 0 = the engine's bodies. When on, its repack slots come out of the hot-expert budget
  // (Model::load_stats().onednn_gib).
  int prefill_dense_onednn = -1;
  // Decode MoE numerics (MoeShape:a8): true = W4A8 like the prefill path (x and h int8 per row and group, s8 x s4 DPAS;
  // the numerics AutoRound tunes for); false = W4A16 (bf16 activations, the decode path). Applies with moe_tp only (EP2
  // decodes W4A16).
  // -1 = by the checkpoint, Config::act_bits: W4A8 for a model whose experts were tuned for int8 activations (act_bits
  // 8: the AutoRound W4A8 bake), W4A16 otherwise (the GPTQ W4A16 export). 1 / 0 force W4A8 / W4A16. The default is 0,
  // W4A16 (-1 / 1 stay as A/B switches).
  int decode_moe_a8 = 0;
  // Prefill GDN recurrence: true = full 64-row chunks at absolute multiples of 64 on the
  // chunked bf16-DPAS scan (S enters as a bf16 operand; KL gate), the other rows on the
  // row core; false = every row on the row core (the numerics). Calls cut at multiples of 64
  // give the same bits as one call either way.
  bool gdn_scan = true;
  // Prefill cold-expert staging: VRAM for this many expert blobs per card (out of the hot
  // budget). A prefill call of >= 256 rows copies each cold expert routed to >= 2 of its 16-row
  // segments there once (most segments first) instead of reading it over PCIe once per segment; the
  // same bytes, so the same bits. 0 = off.
  int prefill_cold_stage = 16;
  // The layer-ahead fetch (prefill.cpp pf_fetch_some): VRAM for this many cold expert blobs per layer, kPfFetchBufs
  // times per card. A prefill call of kPfFetchMinRows rows and more on two cards copies each layer's cold experts (the
  // first this many by id; the rest stay as they were) into a buffer on the copy engine while the layers before it
  // compute, and the MoE reads them there: the host link's bytes overlap the compute instead of stalling the MoE. 0 =
  // off. Taken from the hot experts (only when some are cold, with expert TP, without the MoE CPU path); the same bits.
  int prefill_fetch = 0;
  // The PLE n-gram table from another checkpoint directory (the bake's int8 rows, or a BF16
  // source's table); empty = the model directory.
  std::string ple_dir;
  // Calibration mode (xenithium-bake): no expert store and no placement. Each card holds one
  // layer's routed experts (TP half blobs) in a region the caller fills (calib_set_blobs); every
  // layer's table points there, so only calib_layer() on the layer just loaded is meaningful. Needs
  // moe_tp. The checkpoint may be a BF16 source (fused BF16 experts are never read).
  bool calib = false;
  // The routed experts' blobs are in a block-Hadamard basis (MoeShape:rot). A baked model
  // says so in its config (quantization_config.xenithium_rotation); this forces it (calibration).
  bool moe_rotate = false;
  // As bf16_tensors, for the drafter's dense plan (the drafter calibration's BF16 targets).
  std::vector<std::string> mtp_bf16_tensors;
  // KV pages. 0 = every slot owns `context` tokens of QSA KV from the load (identity page tables). N > 0 = one pool of
  // N tokens (rounded up to 64-token pages; at least one context, at most slots x context) shared by every slot's QSA
  // layers and the drafter's: a sequence takes pages as it grows (kv_reserve; the model's own calls reserve what they
  // write) and returns them at begin_sequence / kv_release. The pool is allocated before the hot-expert placement, so
  // VRAM not reserved for idle contexts goes to hot experts.
  int64_t kv_pool_tokens = 0;
  bool kv_pool_scramble = false;  // pages handed out in a fixed shuffled order (a gate: results must not depend on it)
  // The server's `--kv-pool-tokens auto`: kv_pool_tokens is the least the pool holds, and the VRAM still left once every
  // routed expert has its place (minus reserve_gib) goes to the pool too, up to slots x context. Experts come first: a
  // pool GiB is ~800 experts read over PCIe, so a model whose experts do not all fit gets exactly kv_pool_tokens.
  bool kv_pool_fill = false;
  // KV lending: the pool's pages no sequence holds keep cold experts in VRAM (model/kv_lend.hpp). Pages >= the lend
  // limit are not handed to KV; the limit starts at kv_lend_keep tokens, rises (evicting the lowest-ranked lent
  // experts: back to their host copies, no copy) when a reservation needs pages past it (to the need + kv_lend_keep),
  // and lowers again in lend_idle. Needs the shared pool (kv_pool_tokens), expert TP and no scramble; outputs are
  // bit-identical to off. With the MoE CPU path a lent expert of its set runs on the card (flag 0) until it is given
  // back (its flag 2 again).
  bool kv_lend = false;
  int64_t kv_lend_keep = 32768;
  // The live expert exchange (model/placement.hpp): the hot set follows the decode traffic while serving. Every
  // exchange_period decode steps a round reads the cards' expert reads, and up to exchange_spares cold experts whose
  // decayed score beats the weakest hot one of their layer change places with it (each layer keeps its hot count): the
  // cold blob is copied into a spare VRAM blob on the copy queue, then the tables switch at a step boundary and the hot
  // blob goes back to the host slot the cold one left (its VRAM slot is the next spare). Needs expert TP and cold
  // experts, not the MoE CPU path; outputs are bit-identical to off (the same blobs, other addresses).
  bool exchange = false;
  int exchange_spares = 32;     // spare VRAM blobs per card: the most exchanges per round
  int exchange_period = 4;      // decode steps between rounds
  int exchange_budget = 16;     // exchanges per layer per round
  double exchange_decay = 0.98;  // score decay per round
  // Pinned host memory for the prefix cache's snapshots (GiB over both cards; 0 = none). One snapshot = a slot's state
  // outside its KV pages: snap_bytes.
  double snap_gib = 0;
};

// Routing counts file (XRC1, model/route_counts.hpp).
using ::xenithium::model::read_route_counts;
using ::xenithium::model::write_route_counts;

struct ModelLoadStats {
  double dense_s = 0, experts_s = 0, total_s = 0;
  int64_t hot[kMaxCards] = {}, cold[kMaxCards] = {};
  double free_after_dense_gib = 0;    // tighter card, after dense + states + caches
  double onednn_gib = 0;  // VRAM per card kept for the oneDNN prefill route (out of the hot experts)
  int64_t hot_asked = -1;             // a pinned --hot past the VRAM left (minus the reserve): the count asked; -1 = not cut
  std::string moe_cpu;  // the CPU path's state line ("off", "on: avx512-bf16, 8 threads ...", or why not)
  std::string kv_lend;  // "off", "off (why)" or the lend state at load
  std::string exchange;  // the live expert exchange: "off", "off (why)" or its setup
  int prefill_fetch = 0;  // the layer-ahead fetch's blobs per layer (0: off)
  int64_t lent = 0;  // cold experts lent into the KV pool at load (per card)
  int64_t blob_mismatches = -1;       // -1 = not verified
  int bf16_overrides = 0;             // plan matrices switched int8 -> bf16 by ModelOptions::bf16_tensors
  int w4_overrides = 0;               // plan matrices switched int8 -> int4 RTN (ModelOptions:w4_tensors)
  double mtp_s = 0;                   // drafter load (dense + experts read, requant, pack, upload)
  int64_t mtp_scales_inexact = -1;    // drafter g128 scales not exact in fp16 (-1 = no drafter)
  double mtp_requant_err = 0;         // drafter experts: rms(g128 - g32) / rms(g32), all weights
  uint64_t mtp_vram = 0;              // drafter bytes per card (dense + experts + KV)
  int mtp_bf16_overrides = 0;         // drafter plan matrices switched int8 -> bf16 (ModelOptions::mtp_bf16_tensors)
};

// Wall-clock pieces of the last step (host side), ms.
struct StepTimes {
  double ple_read = 0;  // ids + NVMe rows + dequant
  double submit = 0;    // host submission of every launch
  double wait = 0;      // waiting for both cards after the submission
  bool replayed = false;  // the step ran as a recorded graph
  double record = 0;    // recording + finalizing a new graph (first use of a shape), ms
};

// Card 0's kernel time of the last step per module (ms): each block = its first launch's start
// to its last launch's end; gaps = the GPU span minus the sum (launch gaps, host-bound waits).
struct StepProfile {
  enum Part { embed, hc, ple, qsa, gdn, moe, head, kParts };
  double ms[kParts] = {};
  double span = 0;
  static const char *name(int p);
  // The MoE launches on each card, each launch's own start -> end summed over the layers.
  enum MoeLaunch { router, route, gate_up, down, combine, kMoeLaunches };
  double moe_launch[kMaxCards][kMoeLaunches] = {};
  // The combine split per layer: the faster card's combine (its own work + the wire, little
  // waiting) summed, and the difference to the slower one (waiting for the peer) summed.
  double combine_min = 0, combine_skew = 0;
  std::vector<double> combine_ms[kMaxCards];  // [card][layer]
  static const char *moe_name(int m);
};

class Model {
 public:
  // loader: the one the fabric was opened from (live free VRAM per card for the placement).
  Model(const l0::Loader &loader, Fabric &fab, const std::string &model_dir, const ModelOptions &opt);
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  ~Model();

  const Config &config() const { return cfg_; }
  const ModelLoadStats &load_stats() const { return ls_; }
  int64_t capacity() const { return cap_; }  // one request's most tokens (XEN_CONTEXT)
  // the static YaRN factor of the rotary table (1 = none: the context fits the trained positions; qsa_yarn_for)
  double yarn_factor() const { return yarn_factor_; }
  // YaRN tiers (ModelOptions::yarn_tiers). rope_tiers() = 1: one factor (yarn_factor). Tier t holds positions
  // < rope_tier_limit(t) at rope_tier_factor(t), ascending; the last = capacity() at yarn_factor(). set_rope_tier: the
  // slot's fresh sequence (position 0, before its first prefill) runs in tier t; snapshots keep it.
  int rope_tiers() const { return static_cast<int>(tier_limit_.size()); }
  int64_t rope_tier_limit(int t) const { return tier_limit_.at(static_cast<size_t>(t)); }
  double rope_tier_factor(int t) const { return tier_factor_.at(static_cast<size_t>(t)); }
  void set_rope_tier(int slot, int t);
  int slots() const { return static_cast<int>(seqs_.size()); }
  int cards() const { return nc_; }  // 1 = TP1, 2 = TP2, 4 = TP4
  // The slot the single-sequence calls below act on (begin_sequence, step, verify, commit,
  // prefill, the drafter). Default 0.
  void use_slot(int slot);
  int slot() const { return slot_; }
  int64_t position() const { return seqs_[static_cast<size_t>(slot_)].pos; }
  int64_t position(int slot) const { return seqs_.at(static_cast<size_t>(slot)).pos; }

  // A fresh sequence at position 0 on the current slot: zeroes its GDN states; the caches need no
  // reset (every position is written before any row reads it). With a shared KV pool its pages go back.
  void begin_sequence();
  // ---- the KV pages (ModelOptions:kv_pool_tokens) ----
  // kv_reserve(slot, tokens): positions [0, tokens) of the slot's sequence hold pages (all or nothing; false = the pool
  // cannot now, nothing changed; the fixed layout: true up to the capacity). kv_release: the slot's pages back. Every
  // call that writes KV (verify / step, prefill, drafter passes) reserves what it writes and throws when the pool is
  // out: a serving engine reserves ahead (core::StepModel::kv_reserve) to choose what happens instead.
  bool kv_shared() const { return kvp_.shared(); }
  bool kv_reserve(int slot, int64_t tokens);
  void kv_release(int slot);
  int64_t kv_pool_tokens() const { return kvp_.pool_tokens(); }
  int64_t kv_free_tokens() const { return kvp_.free_tokens(); }
  int64_t kv_used_tokens() const { return kvp_.used_tokens(); }
  int64_t kv_held_tokens(int slot) const { return static_cast<int64_t>(kvp_.held_pages(slot)) * model::KvPages::kPageTokens; }
  size_t kv_bytes(int c) const { return kv_bytes_[c]; }  // the QSA + drafter KV pool on card c
  // The VRAM per card an auto pool (ModelOptions::kv_pool_fill) leaves free besides the reserve and the routed experts'
  // blob bytes, at a max context of `context` tokens: kPoolFillSlackGiB for what the experts' allocations take beyond
  // their blobs (per-layer regions, tables, alignment), so the fill never leaves a few cold; and the step graphs the
  // server records ahead for every QSA bound bucket after the warm-up (their count grows with the context).
  static constexpr double kPoolFillSlackGiB = 0.0625;
  static constexpr double kBucketGraphGiBPerMiTokens = 0.64;
  static constexpr double pool_fill_keep_gib(int64_t context) {
    return kPoolFillSlackGiB + kBucketGraphGiBPerMiTokens * static_cast<double>(context) / 1048576.0;
  }
  // KV lending (ModelOptions::kv_lend): lend_idle refills up to max_blobs lent experts when the held pages allow a
  // lower limit (copies on the compute queues; waits); returns the count (0: nothing to do or lending off).
  // lend_pending: slots it would fill now. The engine calls it while it has nothing to run.
  struct LendStats {
    bool on = false;
    int64_t lent = 0, slots = 0;           // lent now (per card), lend slots in the pool
    int64_t limit_tokens = 0;              // the lend limit (KV takes pages below it)
    uint64_t raises = 0, evicted = 0, filled = 0;
    double raise_ms = 0, fill_ms = 0;      // host time in raises (incl. the drain) / idle refills
  };
  int lend_idle(int max_blobs);
  int lend_pending() const;
  // the slot holds a page at or above the lend keep (it keeps lent experts out); a reservation would give experts back
  bool kv_blocks_lend(int slot) const;
  bool kv_reserve_lends(int slot, int64_t tokens) const;
  LendStats lend_stats() const;
  // The live expert exchange (ModelOptions::exchange): rounds that moved experts, experts moved, rounds put off (a split
  // prompt call in flight or the copies not done at a step boundary), host time in the step tail.
  struct ExchangeStats {
    bool on = false;
    int64_t rounds = 0, swaps = 0, deferred = 0;
    double host_ms = 0;
  };
  ExchangeStats exchange_stats() const { return x_st_; }
  void kv_trim(int slot, int64_t tokens);  // keep the pages of positions [0, tokens) only
  // ---- the prefix cache (ModelOptions:snap_gib) ----
  // A snapshot is what a slot's sequence holds outside its KV pages, at its position right after a prefill call: every
  // GDN layer's S and both conv windows, every QSA layer's raw indexer-key ring, the PLE position ring, the drafter's
  // indexer ring (device, both cards), and the host's GDN parity and PLE n-gram context. snap_save copies it into pinned
  // host memory (on the compute queue, in order with the calls around it: no wait); snap_restore puts it back on the
  // same slot, whose pages below the position must be untouched since; snap_fork copies the source slot's KV pages
  // below the position (target + drafter) into dst's (dst holds pages for them) and restores the snapshot there. After
  // either the slot continues with a prefill call at the snapshot's position. live_ok: the slot's own state can take a
  // prefill call (after a prefill, or after decode steps whose last accept count the next call replays).
  size_t snap_bytes(int c) const;
  int snap_capacity() const { return snap_cap_; }
  int snap_save(int slot);  // an id, -1 = every snapshot in use (or no store)
  // Pins the host buffers of up to n more snapshot ids ahead of their first use (the engine's idle work after the
  // load); returns the ids pinned (0: all are)
  int snap_prepin(int n);
  void snap_restore(int slot, int id);
  void snap_fork(int dst, int src, int id);
  void snap_free(int id);
  int64_t snap_pos(int id) const { return snaps_.at(static_cast<size_t>(id)).pos; }
  void snap_sync();  // every snapshot copy done (probes time them)
  const void *snap_host(int id, int c) const { return snaps_.at(static_cast<size_t>(id)).host[c]; }
  std::vector<size_t> snap_piece_bytes(int c) const;  // probes: the snapshot's pieces in order (GDN S / win0 / win1 per layer, rings)
  bool live_ok(int slot) const;
  // Runs `rows` (1..4) tokens at the current position and waits for both cards. Throws when the
  // capacity is exceeded.
  void step(const int32_t *tokens, int rows);
  // Speculative decode. verify() runs `rows` tokens at the current position like step()
  // but keeps nothing yet; commit(n) then keeps its first n rows (1..rows; the rest were rejected
  // drafts) and advances the position by n. step() == verify() + commit(rows). Rejected rows need
  // no undo: GDN replays only the next step's accept count from its pending log, QSA / PLE caches
  // are position-addressed (the real token rewrites them), and the PLE n-gram context advances by
  // the committed tokens only.
  void verify(const int32_t *tokens, int rows);
  void commit(int n);
  // Several sequences in one step: segment i = `rows` tokens of the sequence on slot `slot`, at its own position; every
  // segment has the same row count, up to 8 rows in all, each slot at most once. Or unequal segments: the first ones B
  // rows each, the rest one row each (kernels::seg_het; the engine's drafting sequences first, then the plain ones).
  // Rows are laid out segment after segment: tokens() / logits() of the step read them in that order. A sequence's rows
  // compute exactly what a step of that sequence alone computes, bit for bit, and a slot left out of a step keeps its
  // state. commit_batch(n) keeps n[i] rows of segment i (1..rows).
  // parent != null = the segment's rows are a draft tree (parent[0] = -1, parent[r] < r, rows in depth-first
  // preorder, <= 8 rows; ModelOptions::dflash_tree): row r computes what a chain of its ancestors computes, bit for
  // bit. commit_batch(n, paths) then keeps the walked rows of each tree segment (paths[i][0] = 0, each the next one's
  // parent; null = rows 0 .. n - 1): their state moves to positions p0 .. p0 + n - 1.
  struct SeqRows {
    int slot = 0;
    const int32_t *tokens = nullptr;
    int rows = 1;
    const int *parent = nullptr;
  };
  void verify_batch(const std::vector<SeqRows> &segs);
  void commit_batch(const std::vector<int> &n);
  void commit_batch(const std::vector<int> &n, const std::vector<const int *> &paths);
  // The parent rows of the slot's last DFlash drafts when they are a tree ([0] = -1, then each draft's parent row;
  // StepModel::draft_parents); empty = a chain.
  std::vector<int> draft_parents(int slot) const {
    return slot >= 0 && static_cast<size_t>(slot) < draft_par_.size() ? draft_par_[static_cast<size_t>(slot)] : std::vector<int>{};
  }
  bool drafts_trees() const { return dflash_ != nullptr && opt_.dflash_tree > 0; }
  void step_batch(const std::vector<SeqRows> &segs);
  // Prefill: n tokens (any count) at the current slot's position, committed. Runs in chunks of
  // ModelOptions::prefill_chunk rows, each LAYER-MAJOR: layer l runs on every row of the chunk before layer l + 1.
  // Afterwards the model is where step()s over the same tokens in 4-row steps would leave it. The head runs on the last
  // slice only (tokens() / logits() then read its rows), or, with on_rows, on every slice: on_rows(row0, rows) is
  // called after each slice's head (row0 counted from the first token of this call), and tokens() / logits() /
  // multi_hidden() read that slice.
  // With a drafter loaded, its pass 1 covers every row (row r's next token: the prompt's following token; for the
  // call's last row next_after, or the target's pick when next_after < 0), so the drafter's KV covers the prompt: one
  // many-row pass per call (set_mtp_prefill); with slices, the head runs on every slice and on_rows is called before
  // each slice's pass.
  void prefill(const int32_t *tokens, int n, const std::function<void(int, int)> &on_rows = {}, int32_t next_after = -1);
  int prefill_chunk() const { return opt_.prefill_chunk; }
  // One prefill call in parts: prefill_begin(tokens, n <= prefill_chunk, next_after) on the
  // current slot, then prefill_advance(layers) runs that many more layers (at least 1) and, once all
  // layers ran, the head (and the drafter's pass 1): it returns true and the call is complete, exactly
  // as prefill(tokens, n, {}, next_after) leaves it (bit for bit). Every part waits for both cards. Between the parts decode steps of
  // OTHER slots may run; nothing else may (no other prefill, no step on this slot). prefill_abort()
  // drops a pending call (the slot's state is then undefined: begin_sequence before reuse).
  void prefill_begin(const int32_t *tokens, int n, int32_t next_after = -1);
  bool prefill_advance(int layers);
  bool prefill_pending() const { return pf_run_ != nullptr; }
  void prefill_abort();
  // Drafts after a prefill: pass 1's last row with `first` as its next token (rerun when
  // that differs from the pick it ran with), then passes 2..depth. Waits.
  // seqs: the decoding sequences of the step the request joins (with both drafters it picks the drafter, mtp_drafts).
  std::vector<int32_t> draft_after_prefill(int32_t first, int depth, int seqs = 1);
  // Diagnostic: every module's output rows of each prefill call, read on card 0 after
  // the module (the queues drained, so slow), as bf16 bits; names "L<l>.<point>". Null = off.
  // points (the name after "L<l>.", e.g. "xm", "moe.ids") limit the trace to those; empty =
  // every point. "moe.ids" (expert TP only) is the MoE's routing, int32 [rows][topk] stored as uint16
  // pairs (width 2 x topk).
  struct PfTrace {
    int rows = 64;  // rows kept per point (the call's first rows)
    std::vector<std::string> points;
    std::vector<std::string> names;
    std::vector<std::vector<uint16_t>> data;
    std::vector<int> widths;
  };
  void trace_prefill(PfTrace *t) { pf_trace_ = t; }
  // How a prefill call runs the drafter's pass 1. rows (the default with expert
  // TP): one many-row pass over the call's rows after the target's layers (pf_dense, hc_mixer_rows,
  // qsa_rows, moe_rows on the drafter's weights); the final mixer + lm_head + pick on the last slice
  // only. slices: the pass after each 4-row slice's head (always with EP2). off: no pass
  // (diagnostic: the drafter KV then has a hole, its drafts are wrong until the next prompt).
  enum class MtpPrefill { rows, slices, off };
  void set_mtp_prefill(MtpPrefill p) { mtp_pf_ = p; }
  // Diagnostic: pass 1's pick of every prefill row (appended per call; rows mode then runs the final
  // mixer + pick on every slice). Null = off.
  void trace_mtp_prefill(std::vector<int32_t> *picks) { mtp_pf_picks_ = picks; }
  // The MTP drafter (ModelOptions::mtp_dir). Pass 1 runs over the last verify / step's rows (every
  // segment of a verify_batch, before its commit_batch;): row r takes the target's state at
  // its position and next[r] (host), the token after that position; it writes its slot's drafter KV
  // there and picks its draft (the token after next[r]). Rows whose tokens get rejected leave KV the
  // real pass 1 rewrites later (position-addressed). Every committed position needs its pass 1 for
  // later drafts to see a complete KV.
  bool has_drafter() const { return mtp_ != nullptr; }
  // Both drafters loaded (ModelOptions::mtp_from); which one drafts a step of `seqs` sequences (true: the MTP layer).
  bool dual_drafters() const { return mtp_ != nullptr && dflash_ != nullptr; }
  int mtp_from() const { return dual_drafters() ? opt_.mtp_from : 0; }
  bool mtp_drafts(int seqs) const { return mtp_ != nullptr && (dflash_ == nullptr || seqs >= opt_.mtp_from); }
  // The DFlash 2 drafter (ModelOptions::dflash_dir); null = none.
  arch::DFlash *dflash() { return dflash_.get(); }
  const arch::DFlash *dflash() const { return dflash_.get(); }
  bool has_dflash() const { return dflash_ != nullptr; }
  // DFlash drafts after the last verify_batch + commit_batch: segment i (the verify's order) takes its verified rows as
  // the drafter's context and drafts from anchor[i] (the token after its keep[i] kept rows: the target's pick at its
  // last kept row), at most depth tokens per segment (the checkpoint's block - 1). depth 0 = context only (deferred, no
  // drafts). Waits; draft_ms() = its wall time.
  std::vector<std::vector<int32_t>> dflash_after_commit(const std::vector<int> &keep, const int32_t *anchor, int depth);
  // A depth per segment (0 = context only), after a verify of equal or unequal segments.
  std::vector<std::vector<int32_t>> dflash_after_commit(const std::vector<int> &keep, const int32_t *anchor, const std::vector<int> &depth);
  double draft_ms() const { return draft_ms_; }
  // Rows one sequence's segment of a verify may carry (kGdnRows; 8 with a DFlash drafter).
  int seg_rows_max() const { return seg_rows_max_; }
  int step_rows_cap() const { return step_cap_; }  // rows a step may carry (ModelOptions::step_rows_cap)
  // kv_only: only the cache writes of pass 1 (front, attention mixer, the QSA projections + prep; bit for bit the KV
  // the whole pass writes), for a step after which nothing drafts: no picks (draft_chain / draft_picks must not follow
  // it).
  void draft_pass1(const int32_t *next, int rows, bool kv_only = false);
  // Drafts after row `row` of the last pass 1 (counted over all its segments): d1 = its pick there,
  // then passes 2..depth (one row each on that row's slot, on the previous pass's state and draft).
  // Waits; returns depth tokens (fewer near the capacity).
  std::vector<int32_t> draft_chain(int row, int depth);
  // The chains of several verify rows (one per segment) at once: pass k runs one row per
  // chain; drafts == draft_chain's for each row. One row = draft_chain.
  std::vector<std::vector<int32_t>> draft_chains(const std::vector<int> &rows, int depth);
  // Pass 1's picks of the last draft_pass1 (waits).
  std::vector<int32_t> draft_picks(int rows);
  // The last drafter pass's logits (waits): ids[v] = the token id of drafter column v (the draft vocabulary's virtual
  // order, or the whole vocabulary), vals [rows][ids.size()] fp32 as the pick read them. For probe captures.
  void draft_logits(int rows, std::vector<int32_t> &ids, std::vector<float> &vals);
  // Coupled drafting for a sampled request on `slot` (core::Sampler::draft_noise): the next drafts pick argmax(logit /
  // T + util::gumbel(key, pos + pass, token)) with inv_t = 1 / T; inv_t 0 = argmax (greedy). Only with the draft
  // vocabulary head (ModelOptions::mtp_vocab); the target's-lm_head path keeps argmax drafts.
  void set_draft_noise(int slot, uint64_t key, uint64_t pos, float inv_t);
  const StepTimes &times() const { return t_; }
  // Step graphs recorded so far and the host time spent recording + finalizing them (ms).
  int graphs_recorded() const { return graphs_n_; }
  double graph_record_ms() const { return graphs_ms_; }
  // The drafter's pass graphs: recorded so far, ms recording + finalizing.
  int mtp_graphs_recorded() const;
  double mtp_graph_record_ms() const;
  // Prompt-call graphs (ModelOptions::prefill_graphs): recorded so far, ms recording + finalizing, calls (or parts)
  // replayed; set_prefill_graphs switches them from the next call on.
  int pf_graphs_recorded() const { return pf_graphs_n_; }
  double pf_graph_record_ms() const { return pf_graphs_ms_; }
  int64_t pf_graphs_replayed() const { return pf_graphs_replayed_; }
  void set_prefill_graphs(bool on) { opt_.prefill_graphs = on; }
  // Records, without running them, the step and drafter pass graphs of every QSA bound bucket up to max_bound (default:
  // the capacity) for every key recorded so far (call after a warm-up has recorded the first bucket's). Host state
  // unchanged; returns the graphs recorded. The graphs hold VRAM outside the placement budget.
  int record_bucket_graphs(int64_t max_bound = -1);
  std::vector<int64_t> bound_buckets() const;  // every bucket a need up to the capacity maps to, ascending
  // Card 0's GPU span of the last step: embed start -> pick end (ms).
  double gpu_span_ms() const;
  // Collect card 0's per-module events in the following steps (profile() reads the last one).
  void set_profiling(bool on) { profiling_ = on; }
  // Replay step graphs (ModelOptions::graphs) from the next step on; off = eager submission.
  void set_graphs(bool on) { opt_.graphs = on; }
  // The GDN trunk's GPTQ calibration inputs. While set, every EAGER decode step (set_graphs(false) first: replayed
  // graphs never call it) calls fn after each linear-attention layer with that layer's input rows x bf16 [rows][hidden]
  // (card 0's copy; the cards agree) and out_proj's input core bf16 [rows][value width] (both cards' halves joined,
  // card 0 first); fn returns false to stop. It waits on both cards per layer: calibration only.
  using GdnCapture = std::function<bool(int layer, int rows, const uint16_t *x, const uint16_t *core)>;
  void set_gdn_capture(GdnCapture fn) { gdn_cap_ = std::move(fn); }
  // ModelOptions::gdn_scan from the next prefill call on (a probe switch for gates that cut prompts
  // at positions that are not multiples of 64).
  void set_gdn_scan(bool on) { opt_.gdn_scan = on; }
  // Image input. Prompt ids >= the vocabulary name image rows of the store (arch::VisionStore): prefill calls map them
  // for the device (encoding images on first use) and the embedding gathers read those rows; the PLE n-grams see the
  // image token there. nullptr = text only. Throws when the store has no image rows on one of the model's cards.
  void set_vision(VisionStore *v);
  bool moe_tp() const { return ms_.shape.tp; }
  StepProfile profile() const;
  // The decode step's module times under graph replay. XEN_DEC_PROFILE set (1): every recorded step graph carries a
  // one-item stamp kernel per module per card (GPU timestamps, calibrated against the host clock), read after each
  // replayed step and summed (the stamps add ~200 launches per card).
  struct DecProfile {
    int steps = 0;
    double ms[kMaxCards][StepProfile::kParts] = {};  // [card][part] per step
    double span[kMaxCards] = {};             // first -> last stamp per step
    double moe[kMaxCards][4] = {};           // [card] MoE: router | route | experts (gate_up, down) | combine + all-reduce
    double qsa[kMaxCards][static_cast<int>(kernels::QsaStage::kCount)] = {};  // [card][stage] QSA (kernels::qsa_stage_name)
    int launches[kMaxCards][StepProfile::kParts] = {};  // [card][part] in the last replayed graph (its nodes; the stamps excluded)
    double ghz = 0;                          // the GPU timestamp clock (card 0, mean over the steps)
  };
  bool dec_profiling() const { return dsprof_; }
  DecProfile dec_profile() const {
    DecProfile d = dp_;
    d.ghz = d.steps ? dp_.ghz / d.steps : 0;
    return d;
  }
  void dec_profile_reset() { dp_ = DecProfile{}; }

  // The greedy tokens of the last step (both cards agree by construction; card 0's copy).
  std::vector<int32_t> tokens(int rows);
  // The last step's logits rounded to bf16 (the model's logits dtype), [rows][vocab] as float, both cards' vocabulary
  // halves joined.
  void logits(int rows, std::vector<float> &out);
  // Each row's sampling candidates instead of the whole row: both cards' top T 16-column tiles by the bf16-rounded
  // maximum (arch::HeadCands), ids / vals [rows][n], n = 2 x T x 16 returned; vals as logits().
  int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals);
  // Log-softmax statistics of the head's current rows on the cards (kernels/head/logprob.hpp):
  // rec = [card][rows][kLpRec] records, xt[r] = the bf16 logit of targets[r] (+inf where < 0: no
  // target). The head's rows: the last step / verify, or a prefill slice inside on_rows. Waits.
  void logprob_stats(int rows, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt);
  // Prefill (as prefill(tokens, n, {}, next_after)) with the logprob records of EVERY row from a many-row head: the
  // final mixer over each chunk, lm_head per group of up to Card::Pf::lp_rows rows on pf_dense, logprob_stats_rows; one
  // host round trip per group (the targets' logits cross the cards). targets[r] for row r of the n tokens (< 0: none);
  // rec = [card][n][kLpRec], xt[r] = the target's bf16 logit (+inf: none). The last slice's head still runs as in
  // prefill (picks, prefill logits). Needs expert TP (the moe_rows scratch holds the logits group): pf_logprob_rows().
  void prefill_logprobs(const int32_t *tokens, int n, int32_t next_after, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt);
  bool pf_logprob_rows() const;
  // The last step's multi-stream state after the final mixer's combine (the pre-final-mixer hidden
  // the MTP drafter takes), card 0's copy: bf16 bits [rows][hc * hidden].
  void multi_hidden(int rows, std::vector<uint16_t> &out);
  // Do the two cards hold bit-identical residuals and final-mixer outputs now?
  bool cards_agree();
  // Mailbox timeouts on any wire (pick, pair sums, MoE), OR over both cards; 0 = none.
  uint32_t wire_errors();

  // The last step's routing (card 0's records; both cards route identically): ids
  // [layer][rows][topk], global expert ids, descending weight.
  void route_ids(int rows, std::vector<int32_t> &ids);
  // Is expert (layer, e) in VRAM on its owner card?
  bool expert_hot(int layer, int expert) const;
  // ModelOptions::rank_file: the recorded ranking (null without one), learning on / off (off: the decode steps still
  // count on the cards, and the counts are dropped when learning resumes: the server's warm-up), and a save now (the
  // counts since the last one folded in; between steps).
  const model::RouteRecord *ranking() const { return rec_.get(); }
  // The CPU path (ModelOptions::moe_cpu): on, and the pools' counters (requests, experts, timeouts seen).
  bool moe_cpu_on() const { return cpu_on_; }
  struct MoeCpuStats {
    uint64_t requests = 0, experts = 0;
    bool timed_out = false;  // a card's wait timed out: the path is off since
    double busy_us = 0;
  };
  MoeCpuStats moe_cpu_stats() const;
  void set_learning(bool on);
  void save_ranking();

  // ---- calibration (xenithium-bake) ------------------------------------------------------
  // The multi-stream state between two layers of a prefill chunk of m rows (card 0's copy; both cards
  // hold the same): the residual with layer l's MoE output still pending, the MLP mixer's injection
  // logits for it, and the MoE output. The next layer's first mixer combines them.
  struct CalibState {
    int m = 0;
    std::vector<uint16_t> res;  // bf16 [m][hc * hidden]
    std::vector<float> inj;     // [m][hc]  (the down output's columns [lowrank, lowrank + hc))
    std::vector<uint16_t> ym;   // bf16 [m][hidden]
  };
  // What a layer's MoE saw (card 0): its input, the residual after the MLP mixer's combine (the
  // attention block already added), the injection logits for the MoE output, the routing, and the
  // shared expert's two factors (the MoE output = bf16(routed + sg * sy)).
  struct CalibCapture {
    std::vector<uint16_t> xm;   // bf16 [m][hidden]
    std::vector<uint16_t> res;  // bf16 [m][hc * hidden]
    std::vector<float> inj;     // [m][hc]
    std::vector<int32_t> ids;   // [m][topk], descending weight
    std::vector<float> topw;    // [m][topk]
    std::vector<float> sg;      // [m] sigmoid of the shared gate
    std::vector<float> sy;      // [m][hidden] the shared expert's output (fp32)
  };
  // Replaces the layer's MoE output: given the capture, writes bf16 [m][hidden]; uploaded to both cards.
  using CalibMoeHook = std::function<void(int layer, const CalibCapture &cap, std::vector<uint16_t> &ym)>;
  // One sample (a fresh sequence on slot 0, m <= prefill_chunk tokens) through ONE layer: the state
  // before it from `in` (layer 0: the embedding; in must be null), the state after it into `out`
  // (null = not read back), the MoE's inputs into `cap` (null = not captured), and `hook` (null = the
  // engine's MoE output stands). Bit for bit what the same layer computes inside prefill() of the same
  // tokens when `in` holds that prefill's state (gate: xenithium-bake selftest). Waits.
  void calib_layer(const int32_t *tokens, int m, int layer, const CalibState *in, CalibState *out, CalibCapture *cap,
                   const CalibMoeHook *hook = nullptr);
  // One sample (a fresh sequence on slot 0, m <= prefill_chunk tokens) through every layer as a prefill call runs them
  // (any mode; normal mode = the served experts), the state after the last layer into `out` (calib_layer's `out` of
  // layer layers - 1). For the drafter's calibration on the served target (xenithium-bake mtp-tune --served).
  void calib_final_state(const int32_t *tokens, int m, CalibState &out);
  // Calibration mode: one layer's per-card TP blob region (moe_tp_layout stride, [experts] blobs).
  size_t calib_blob_stride() const { return calib_stride_; }
  void calib_set_blobs(int card, const uint8_t *host, size_t bytes);  // bytes <= experts x stride; waits
  void calib_zero_blobs();                                           // every routed output then reads 0
  // The MTP drafter's layer on one sample (needs a drafter and expert TP, any mode): `in` =
  // the target's state after its last layer (calib_layer's `out` of layer layers - 1; its first m rows
  // are used), next[r] = the token after row r. Runs the drafter's many-row pass 1 (the one a prefill
  // call runs) up to its MoE, with `out` / `cap` / `hook` as calib_layer's (the state after the
  // drafter's MoE: its residual after the MLP mixer, the injection logits, the MoE output). Bit for
  // bit the drafter rows of prefill(tokens, m, {}, next[m - 1]) when `in` holds that prefill's state
  // (gate: xenithium-bake selftest-calib --mtp). Writes the drafter KV of slot 0. Waits.
  void calib_mtp(const int32_t *tokens, int m, const CalibState &in, const int32_t *next, CalibState *out, CalibCapture *cap,
                 const CalibMoeHook *hook = nullptr, std::vector<uint16_t> *head_in = nullptr);
  // head_in: also the drafter's final mixer over the m rows, bf16 [m][hidden] (card 0): the
  // input of its lm_head / draft head, for the draft head's calibration.
  // The drafter's routed experts (TP half blobs, calib_blob_stride() layout, [experts] blobs): replace
  // the loaded ones. Waits.
  void calib_set_mtp_blobs(int card, const uint8_t *host, size_t bytes);
  // For the bake's KLD gate: the model's head on a CalibState after the last layer (calib_layer's `out` of layer
  // layers - 1; its first m rows): the final mixer over the rows, lm_head per group of rows as in a prompt-logprobs
  // prefill (logits rounded to bf16), and sink(row0, rows, logits) per group with the bf16 logits [rows][vocab] (both
  // cards' halves joined). Waits.
  using LogitSink = std::function<void(int row0, int rows, const uint16_t *logits)>;
  void calib_logits(const int32_t *tokens, int m, const CalibState &in, const LogitSink &sink);

 private:
  struct Card;
  struct Mtp;
  kernels::MoeShape mtp_moe_shape(int c) const;  // the target's MoE shape in the drafter's basis
  const std::vector<kernels::DenseWeights> *mtp_head();  // the int2 draft head per card, or null (the target's lm_head)
  // The drafter's lm_head + pick for its passes (the draft vocabulary head when loaded, else lm_pick).
  void mtp_lm_pick(int rows, const std::vector<const Head::bf16 *> &in, const std::vector<int32_t *> &tok, const std::vector<float *> &logits,
                   int pass, int seg_rows, int slot_fixed);
  void mtp_noise_upload();  // Mtp::noise_host -> each card's noise table (before a pass that picks)
 public:
  int mtp_draft_vocab() const;  // the draft vocabulary's size (0 = the whole vocabulary)
 private:
  std::vector<kernels::DenseWeights> mtp_head_w_;
  // seg_key = the step's seg_rows code (rows.hpp: rows per segment, or seg_het for unequal segments) x 16 + replay (a
  // segment replays an accepted tree path: the GDN tree core) x 8 + tree (a segment's rows are a draft tree: QSA /
  // PLE / GDN tree instances) x 4 + wide (an accept word above 4, the GDN's 8-row core) x 2 + p1 (every accept word <=
  // 1: the GDN core stages one previous row); seg_key_code / seg_key_wide / seg_key_p1 / seg_key_tree / seg_key_replay
  // take it apart.
  static constexpr int seg_key(int code, bool wide, bool p1 = false, bool tree = false, bool replay = false) {
    return code * 16 + (replay ? 8 : 0) + (tree ? 4 : 0) + (wide ? 2 : 0) + (p1 ? 1 : 0);
  }
  static constexpr int seg_key_code(int key) { return key >> 4; }  // floor: negative codes come back too
  static constexpr bool seg_key_wide(int key) { return (key & 2) != 0; }
  static constexpr bool seg_key_p1(int key) { return (key & 1) != 0; }
  static constexpr bool seg_key_tree(int key) { return (key & 4) != 0; }
  static constexpr bool seg_key_replay(int key) { return (key & 8) != 0; }
  void submit_step(int rows, int seg_key, int bound);
  // A decode step's rows by owner (model.cpp, "a decode step's rows by owner")
  bool dec_by_owner(int rows) const { return nc_ == kMaxCards && rows >= kDecOwnerMinRows; }
  void dec_owner_init();  // the owner sums' and the shared buffers' mailboxes (four cards)
  struct DecMixer;        // one mixer of a step on one card: the block it combines, its weights, its output
  kernels::HcEvents dec_mixer(int c, int rows, const DecMixer &m);  // .up = the card's last launch of it
  sycl::event dec_ple_combine(int c, int rows, const sycl::ext::oneapi::bfloat16 *block);  // the pending block, before the PLE block
  void dec_feature(int c, int rows, int feature);  // a drafter feature (the streams' mean) into the step's features
  void dec_final_mix(int rows);                    // the final mixer; every card ends with every row
  struct PfRun;  // a prefill call between its parts (prefill.cpp)
  void pf_start(const int32_t *tokens, int m, int row_base, const std::function<void(int, int)> &on_rows, int32_t next_after);
  void pf_layers(int count);  // the next `count` layers of the pending call
  // Prompt-call graphs (prefill.cpp): the next `count` layers from a recorded graph when the call has one (recorded
  // here on the call's kPfGraphRecordAt-th run), else pf_layers. A recorded graph serves every call with the same key:
  // everything the layers' launches take from the host.
  struct PfGraphKey {
    int rows = 0, layer0 = 0, layer_end = 0;  // the call's rows; the layers it (or this part of it) runs
    int64_t qsa_bound = 0;                    // the QSA grid bound (bound_bucket of the call's end)
    int pair_parity = 0, kv_parity = 0, moe_parity = 0;  // the prefill wires'
    int hc_cur = 0, hc_lcur = 0, hc_lora_ld = 0;          // the call's HC state indices (every card's alike)
    auto operator<=>(const PfGraphKey &) const = default;
  };
  struct PfGraphs;
  std::unique_ptr<PfGraphs> pf_graphs_;
  void pf_layers_run(int count);
  [[nodiscard]] std::optional<PfGraphKey> pf_graph_key(int count) const;
  bool pf_record_graph(const PfGraphKey &key, int count);  // false: the recording failed twice (the layers ran eagerly)
  void pf_replay_graph(const PfGraphKey &key);
  void pf_flag_set(int set);  // every card's split flag lines: set 0 = the launches', 1 + i = prompt-call graph flag set i
  int pf_graphs_n_ = 0;
  double pf_graphs_ms_ = 0;
  int64_t pf_graphs_replayed_ = 0;
  // The layer-ahead fetch (ModelOptions::prefill_fetch, prefill.cpp): the lists of cold experts per layer to the cards
  // (as they stand at the call's start), and layer l's fetch into buffer l % 2 on every card (half 0 / 1 / 2 = all).
  void pf_fetch_lists(int m);
  void pf_fetch_some(size_t bytes, int must, int last);
  size_t pf_fetch_gap(int layer, int gap) const;
  int pf_fetch_cap_ = 0;
  std::vector<int16_t> pf_fetch_order_;  // [layer][expert]: the layer's experts by ranking, most routed first
  // The same for a call whose rows live on their owner card (prefill.cpp, "rows by owner")
  enum class PfMixer : uint8_t { attention, mlp };
  struct PfShared {  // one chunk buffer on every card: rows of row_bytes each
    void *bufs[kMaxCards] = {};
    size_t row_bytes = 0;
  };
  template <class BufOf>
  PfShared pf_shared(BufOf buf_of, size_t row_bytes);  // buf_of(Card &) = that card's buffer
  int pf_lane_count(int m) const;                // the lanes of a call of m rows; 0 = not by owner
  void pf_layers_owner(int count);
  void pf_owner_ple(int layer);                  // the PLE layer's block, on every row on every card
  void pf_ple_rows();                            // the pending call's PLE rows on the host, once (as the PLE layer is next)
  void pf_owner_state();                         // every card gets every row of the chunk's state (the PLE block, the call's end)
  void pf_lane_mixers(int layer, PfMixer which); // per lane: its pending sum, the mixer on each card's own rows, the output shared
  void pf_lane_attention(int lane, int layer);   // GDN or QSA over the lane's rows; its sum stays pending
  void pf_lanes_moe(int lane0, int count, int layer);  // one MoE call over `count` lanes' rows; each lane's sum stays pending
  void pf_lane_sum(int lane);                    // the lane's pending sum, each card's own rows
  // The MoE input by owner (PfRun::moe_routed): card c's routing and int8 rows [lo, hi) of the MLP mixer's output, and the
  // buffers its share copies instead of the bf16 rows.
  sycl::event pf_moe_route_own(int c, int lo, int hi, int layer);
  std::array<PfShared, 4> pf_moe_route_sets();
  // The GDN layers' attention input by owner (PfRun::gdn_owner): card c's rows [lo, hi) of the attention mixer's output
  // quantised per row and dotted with every card's a | b; the buffers its share copies instead of the bf16 rows; the
  // setup (every card's a | b weights on each card, the buffers).
  sycl::event pf_gdn_owner_inputs(int c, int lo, int hi, int layer);
  std::array<PfShared, 3> pf_gdn_owner_sets();
  void pf_gdn_owner_init();
  // each card's own rows of the lane to the other cards; written: per card the event of its last kernel that writes them
  // (empty = the copies follow everything on the card's queue so far)
  void pf_lane_share(int lane, std::span<const PfShared> sets, std::span<const sycl::event> written = {});
  void pf_lane_wait(int lane);                   // each card waits for the other owners' rows of the lane's last share
  template <class PartOf, class OutOf>
  void pf_split_sum(PartOf part_of, OutOf out_of, bool add, int kind, const sycl::event *sent = nullptr);  // prefill.cpp
  void pf_split_wait();  // the pending split sum's peer rows (both cards), if any
  // the chunk's mixer on card c; when `block` is the pending split sum's output, the card's
  // own rows first, then the wait, then the peer's (bit-identical to one call)
  void pf_mix_rows(int c, const sycl::ext::oneapi::bfloat16 *block, const kernels::HcWeights &w, sycl::ext::oneapi::bfloat16 *out);
  void pf_finish();           // the head (+ drafter), then the slot's state advances
  void pf_mark(int phase);
  void pf_tr(const std::string &name, const sycl::ext::oneapi::bfloat16 *dev, int width);
  void run_graph(int rows, int seg_rows, int64_t need);
  void record_step_graph(int rows, int seg_rows, int64_t bound);  // from the current host state, not run
  // Graphs hold VRAM outside the placement. Before a graph is recorded on first use, graph_room() evicts the least
  // recently replayed step / drafter graphs (queues drained first) when the tighter card has less than kGraphRoomMiB
  // free: kGraphRoomEvict of them while the population is above its keep mark (graph_keep_, set when the room first
  // runs out), one at the mark, none below it (a destroyed graph's memory stays in the driver's pool for the next
  // recording). all = evict every graph (a recording failed: the mark drops too). Returns the graphs evicted.
  int graph_room(bool all = false);
  int graphs_alive() const;  // step + MTP layer graphs held
  int graph_keep_ = -1;      // the population held below the room; < 0: the room has not run out yet
  double vram_free_mib() const;  // the tighter card (sysman); < 0 unknown
  const l0::Loader *loader_ = nullptr;
  uint64_t graph_tick_ = 0;
  int64_t graphs_evicted_ = 0;
  int graphs_eager_ = 0;  // steps run eagerly after a recording failed twice
  // The step's PLE rows are read while it runs (kernels::ple_wait_rows): pinned words [0] the step's number (before the
  // submission), [16] the number published after the rows, [32 + 16 c] card c's timeout flag
  uint32_t *ple_words_ = nullptr;
  uint32_t ple_seq_ = 0;
  int64_t bound_bucket(int64_t need) const;  // the QSA grid bound a graph is recorded for (>= need)
  void load_mtp(const std::string &dir);
  // One drafter pass of `rows` rows at position pos: hidden[c] the input states (rows of hc x hidden),
  // ids[c] device token ids, picks into the pass's token slot.
  // segs: (slot, position of the segment's first row), seg_rows rows each, rows = segs * seg_rows.
  void mtp_pass(int pass, int rows, const std::vector<const sycl::ext::oneapi::bfloat16 *> &hidden, const std::vector<const int32_t *> &ids,
                const std::vector<std::pair<int, int64_t>> &segs, int seg_rows, bool kv_only = false);
  void mtp_submit(int pass, int rows, const std::vector<const sycl::ext::oneapi::bfloat16 *> &hidden, const std::vector<const int32_t *> &ids, int bound, int seg_rows,
                  bool kv_only = false);
  // Records a pass graph from the current host state at `bound` without running it (state unchanged).
  void mtp_record_graph(int pass, int rows, const std::vector<const sycl::ext::oneapi::bfloat16 *> &hidden, const std::vector<const int32_t *> &ids,
                        int seg_rows, int64_t bound, bool kv_only = false);
  void mtp_record_buckets(const std::vector<int64_t> &buckets);  // every recorded key at every bucket
  // The drafter's pass 1 over a prefill chunk's m rows (drafter.cpp): cs = the chunk's HC
  // state per card after the target's last layer (its combine still pending on Card::Pf::ym); next[r]
  // = row r's next token. Reuses the chunk buffers (the target is done with them).
  // layer_only (calib_mtp): stop after the MoE (no final mixer / pick) and hand back the
  // drafter's chunk state per card.
  void mtp_prefill_rows(int m, int64_t pos0, std::vector<kernels::HcState> &cs, const std::vector<int32_t> &next,
                        std::vector<kernels::HcState> *layer_only = nullptr);
  // Calibration (prefill.cpp): a CalibState into a chunk's HC state + pf.ym (both cards; prev_layer = the layer whose
  // MLP mixer left the injection logits), and back out of card 0.
  void calib_put_state(std::vector<kernels::HcState> &cs, const CalibState &in, int m, int prev_layer);
  void calib_get_state(const kernels::HcState &st, int m, CalibState &out);
  // The capture points: the MoE's input xm + the residual + injection logits (card 0, after the MLP
  // mixer); then the routing + shared term and the hook's output (after the MoE).
  void calib_capture_mix(int m, const kernels::HcState &st, const sycl::ext::oneapi::bfloat16 *xm);
  void calib_capture_moe(int layer, int m);

  Fabric &fab_;
  // The cards the model runs on (fab_.size()): 1 = TP1, every matrix whole and every expert whole on one card, no wire;
  // 2 = TP2 + the two-card sums; 4 = TP4, every matrix and expert in quarters, the four-card sums. Per-card arrays
  // below keep kMaxCards entries; only [0, nc_) are used.
  int nc_ = 2;
  template <class F>
  auto per_card(F f) const {  // {f(0), .., f(nc_ - 1)}
    std::vector<decltype(f(0))> v;
    for (int c = 0; c < nc_; ++c) v.push_back(f(c));
    return v;
  }
  template <class W>
  W *wire_of(W &w) const { return nc_ > 1 ? &w : nullptr; }  // the cards' wire, or none at TP1
  Config cfg_;
  ModelOptions opt_;
  ModelLoadStats ls_;
  StepTimes t_;
  model::Checkpoint ck_;
  std::unique_ptr<load::DenseStore> dense_;
  std::unique_ptr<ExpertStore> experts_;
  std::unique_ptr<Head> head_;
  GdnCapture gdn_cap_;  // (set_gdn_capture)
  std::unique_ptr<HeadCands> cands_;  // the sampling candidates (logits_cands)
  std::unique_ptr<PleTable> ple_tab_;
  HcSet hs_;
  MoeSet ms_;
  GdnSet gs_;
  QsaSet qs_;
  PleSet ps_;
  std::vector<std::unique_ptr<Card>> cards_;
  uint16_t *ple_pinned_ = nullptr;  // the step's PLE rows, bf16 [4][E]
  PfTrace *pf_trace_ = nullptr;
  MtpPrefill mtp_pf_ = MtpPrefill::rows;
  std::vector<int32_t> *mtp_pf_picks_ = nullptr;
  // prefill_logprobs' request while its prefill runs (pf_finish reads it).
  struct PfLp {
    const int32_t *targets = nullptr;
    int n = 0, top = 0;
    float *rec = nullptr, *xt = nullptr;  // host: [card][n][kLpRec], [n]
  };
  const PfLp *pf_lp_ = nullptr;
  void pf_head_rows();  // prefill.cpp: the many-row head of the pending call's chunk
  uint16_t *pf_ple_ = nullptr;      // a prefill chunk's PLE rows, bf16 [prefill_chunk][E] (pinned)
  std::vector<int32_t> pf_words_;   // host source of the chunk's slice records (rows.hpp)
  std::unique_ptr<PfRun> pf_run_;   // the pending prefill call
  int64_t cap_ = 0;
  double yarn_factor_ = 1.0;
  std::vector<int64_t> tier_limit_, tier_base_;  // per tier its positions and first row in the rope table
  std::vector<double> tier_factor_;
  // The KV page table (host) and its device copy per card (read by every QSA layer and the drafter).
  model::KvPages kvp_;
  int32_t *kv_table_[kMaxCards] = {};
  size_t kv_bytes_[kMaxCards] = {};
  void kv_need(int slot, int64_t tokens);  // kv_reserve or throw
  void kv_upload();                        // the host table's changed range to both cards (waits)
  // The pool's buffers (model.cpp), allocated once, after everything but the routed experts so an auto pool can be sized
  // from the VRAM they leave. kv_page_bytes: one 64-token page of every QSA layer (+ the MTP drafter's) on card c.
  // kv_pool_pages_for: the pool's size in pages (ModelOptions::kv_pool_fill grows it; vram_free = free VRAM per card,
  // empty = unknown; expert_bytes = every routed expert blob of a card). kv_pool_alloc: sized pages, table uploaded.
  size_t kv_page_bytes(int c) const;
  int kv_pool_pages_for(const std::vector<double> &vram_free, double expert_bytes) const;
  void kv_pool_alloc(int pool_pages);
  // KV lending (model.cpp). lend_cand_[k] = (layer, blob index) lent in slot k of lend_ (k < lend_n_); tab_host_ /
  // cold_host_ mirror each card's blob table and cold flags.
  void lend_setup(const ModelOptions &opt, const std::vector<double> *rank);
  bool lend_raise(int short_pages);      // evict until short_pages more pages fit below the limit; false: cannot
  void lend_upload();                    // the mirrors to both cards (compute queues, waits)
  void lend_copy(int from, int to);      // H2D of slots [from, to) + their mirror entries (no upload)
  model::KvLend lend_;
  std::vector<std::pair<int, int>> lend_cand_;
  std::vector<uint8_t *> lend_buf_[kMaxCards];
  std::vector<const uint8_t *> tab_host_[kMaxCards];
  std::vector<uint8_t> cold_host_[kMaxCards], cold_orig_[kMaxCards];
  int lend_n_ = 0, lend_keep_pages_ = 0, lend_refill_run_ = 0;
  uint64_t lend_ev_seen_ = 0;  // evictions already logged
  LendStats lend_st_;
  // The live expert exchange (model.cpp, exchange_*). A round's exchanges go through three step boundaries: 1 the cold
  // blobs to spares (copy queue), 2 the cold experts' entries to the spares + the hot blobs to the host slots the cold
  // ones left (copy queue; a lent slot gets the hot blob too, on the compute queue), 3 the hot experts' entries to their
  // host slot (or lent slot) and their VRAM slots to the spares. lend_pos_ = lend_cand_'s index of a blob, -1 none.
  struct XSwap {
    int layer, hot, cold;
    uint8_t *spare[kMaxCards], *host[kMaxCards], *vram[kMaxCards];
  };
  struct XUpd {  // one table entry (pinned; read by the update launch)
    const uint8_t *ptr;
    uint32_t idx;
    uint32_t flag;
  };
  void exchange_setup(const ModelOptions &opt, const std::vector<double> *rank);
  void exchange_tick();
  void exchange_apply(int c, const XUpd *u, int n);  // the entries (and cold flags) on card c's compute queue
  std::unique_ptr<model::Placement> xpl_;
  std::vector<XSwap> xr_;
  int xr_stage_ = 0;
  int64_t x_step_ = 0;
  std::vector<uint8_t *> x_spare_[kMaxCards];
  std::vector<uint8_t *> x_spare_mem_[kMaxCards];  // the spares' own allocations (released at exit)
  XUpd *x_upd_[kMaxCards] = {};  // pinned [2][spares]: stage 2's and stage 3's entries
  std::vector<uint32_t> x_seen_;
  std::vector<int> lend_pos_;
  ExchangeStats x_st_;
  // The snapshot store (host buffers allocated on first use and kept), a device staging buffer per card, the fork's
  // page pairs per card.
  struct SnapRec {
    int64_t pos = 0;
    int64_t rope_delta = 0;  // the sequence's YaRN tier (Seq::rope_delta)
    int gdn_par = 0;
    kernels::PleContext ple;
    bool used = false;
    void *host[kMaxCards] = {};
  };
  std::vector<SnapRec> snaps_;
  int snap_cap_ = 0;
  void *snap_stage_[kMaxCards] = {};
  int32_t *fork_pages_[kMaxCards] = {};
  std::vector<std::pair<void *, size_t>> snap_pieces(int c, int slot) const;  // the slot's state buffers on card c
  void snap_copy(int slot, SnapRec &r, bool save);
  int pf_run_slot() const;  // prefill.cpp: the pending split call's slot, -1 = none
  // One sequence slot's host state.
  struct Seq {
    int64_t pos = 0;
    int64_t rope_delta = 0;           // its YaRN tier's first rope table row (records' kSegRope; 0 = the last tier)
    int last_rows = 0;                // rows committed by its previous step (the GDN accept count)
    int gdn_par = 0;                  // its GDN parity (window / pending log); flips per step and chunk
    kernels::PleContext ple_ctx;      // its n-gram context (advances at commit)
    bool pending = false;             // a verify waits for its commit
    int64_t run_pos = 0;              // position of the last verify's first row
    int run_rows = 0;                 // its rows
    int32_t run_tokens[Head::kRows] = {};
    int run_parent[Head::kRows] = {};  // the last verify's tree (run_tree), parent rows
    bool run_tree = false;
    uint64_t path = 0;                 // the last commit's tree path word (rows.hpp kSegPath; 0 = a chain): the GDN replays it
    bool pf_committed = false;        // the last call on the slot was a prefill call (a snapshot may follow)
    bool broken = false;              // a prefill call of the slot was aborted (no live resume)
  };
  std::vector<Seq> seqs_;
  int slot_ = 0;
  std::vector<int> batch_slots_;    // the last verify's segments (slots, in row order)
  // calibration
  size_t calib_stride_ = 0;
  uint8_t *calib_region_[kMaxCards] = {};
  bool pf_skip_ple_ = false;                 // pf_start: skip the host PLE rows (a calib layer that is not the PLE layer)
  // image input (set_vision). The mapped ids' host buffers: a chunk's (pf_start), the drafter's next tokens
  // (mtp_prefill_rows), the slice drafter's; each is rewritten only after the copies from it ran (pf_finish waits)
  VisionStore *vis_ = nullptr;
  std::vector<int32_t> vis_ids_, vis_next_, vis_tmp_;
  const int32_t *vis_map(const int32_t *tokens, int m, std::vector<int32_t> &buf);  // tokens, or buf with image ids mapped
  void embed_rows(sycl::queue &q, int c, const int32_t *ids, int rows, int streams, sycl::ext::oneapi::bfloat16 *out);  // embed_gather (+ image rows)
  CalibCapture *calib_cap_ = nullptr;        // pf_layers: capture the MoE's inputs
  const CalibMoeHook *calib_hook_ = nullptr; // pf_layers: replace the MoE output
  int calib_layer_ = -1;
  float *calib_tmp_[kMaxCards] = {};  // [prefill_chunk + 4][hc] injection columns
  int batch_rows_ = 0;              // rows per segment of the last verify (the largest segment's with unequal ones)
  int batch_code_ = 0;  // its seg_rows code (rows.hpp; negative = unequal segments)
  std::vector<int> run_slots_;      // the last verify's slots, kept past its commit (the drafter's pass 1)
  std::unique_ptr<Mtp> mtp_;
  // the DFlash 2 drafter; df_feat_at_[l] = the feature captured after layer l's attention mixer (-1 none); its prefill
  // context pass borrows pf.xa / pf.xm / pf.pw and has its own fp32 rows
  std::unique_ptr<arch::DFlash> dflash_;
  std::vector<std::vector<int>> draft_par_;  // per slot, its last drafts' tree (draft_parents)
  void note_tree(int slot, size_t drafts, int seg);  // from the DFlash pass's tree of segment seg (a chain: empty)
  std::vector<int> df_feat_at_;
  float *df_part_[kMaxCards] = {}, *df_ykv_[kMaxCards] = {};
  double draft_ms_ = 0;
  int seg_rows_max_ = kernels::kGdnRows;
  int step_cap_ = kernels::kStepRows32;  // ModelOptions::step_rows_cap
  // rows a DFlash pass holds (its block rows, and its context rows): the step's capacity (one 64-row pass for a 64-row
  // step's drafts)
  int pass_rows() const { return step_cap_; }

  // step_seqs: the sequences of the step the drafts are for (-1 = the segments'; trees: tree_step_seqs)
  std::vector<std::vector<int32_t>> dflash_drafts(const std::vector<arch::DFlash::Seg> &segs, int ctx_rows, int depth, int step_seqs = -1);
  int tree_step_seqs(int step_seqs) const;
  int step_parity_ = 0, pair_parity_ = 0, moe_parity_ = 0;
  int pf_moe_parity_ = 0;  // the prefill MoE wire's (moe_rows)
  int pf_pair_parity_ = 0; // the prefill pair wire's (gdn_rows, qsa)
  // The KV split by tokens (ModelOptions::kv_split): the pairs' mailboxes of the attention merge. Decode: a region per QSA
  // layer and one for the MTP layer (qsa_pair_region), each used once a step, so the module sums between two uses of
  // a region keep the partner past its last read (parity 0 throughout, graph-safe). Prefill: one mailbox, its parity
  // flipped per qsa_rows call (pf_kv_parity_; the target's layers and the MTP layer's, lanes by row offset).
  size_t kv_pair_region_words_ = 0;  // words of one decode region
  std::vector<int> qsa_ord_;         // [layer]: the QSA layer's index (its decode region), -1 for the others
  int pf_kv_parity_ = 0;
  // A copy of `sc` with the pair's mailbox at decode region `region` (qs_.layers.size() = the MTP layer's).
  kernels::QsaScratch qsa_pair_region(const kernels::QsaScratch &sc, int region) const;
  uint32_t pf_split_seq_ = 0;  // the last sequence number on the split sums' flag lines
  uint64_t *pf_stamps_[kMaxCards] = {};  // XEN_PF_PROFILE=3 stamp words per card, allocated on first use
  // XEN_DEC_PROFILE: decode stamp words per card, the marks of the step being recorded (part, word), totals
  static constexpr int kDecStampWords = 1024;
  // Marks below 0 (dec_marks_; -1 = the step's start) stamp inside a module: the MoE layer's router | route | experts
  // (kMarkMoe - j, j < kMoeStamped) and the QSA layer's stages before o_proj (kMarkQsa - stage, stage < kQsaStamped).
  static constexpr int kMoeStamped = 3;
  static constexpr int kMarkMoe = -2, kMarkMoeLast = kMarkMoe - (kMoeStamped - 1);
  static constexpr int kMarkQsa = kMarkMoeLast - 1, kMarkQsaLast = kMarkQsa - (kernels::kQsaStamped - 1);
  bool dsprof_ = false;
  uint64_t *dec_stamps_[kMaxCards] = {};
  std::vector<std::pair<int, int>> dec_marks_[kMaxCards];
  std::vector<int64_t> dec_nodes_[kMaxCards];  // the recording graph's node count at each mark (-1: not counted)
  std::vector<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable>> *dec_rec_ = nullptr;
  DecProfile dp_;
  sycl::event dec_cal_[kMaxCards];
  // A replayed step's stamps are read after the step ends (dec_finish, from verify_batch), not inside run_graph: the
  // step waits for the PLE rows the host publishes after the submission, so a read there would deadlock until the PLE
  // wait times out. dec_pend_ holds the step's marks, launches and calibration events until then.
  struct DecPending {
    bool on = false;
    std::vector<std::pair<int, int>> marks[kMaxCards];
    int launches[kMaxCards][StepProfile::kParts] = {};
    sycl::event cal[kMaxCards][2];
  };
  DecPending dec_pend_;
  void dec_stamp(int c, int part);
  [[nodiscard]] uint64_t *dec_inner_stamps(int c, int mark0, int n);
  void dec_read(const std::vector<std::pair<int, int>> (&marks)[kMaxCards], const sycl::event (&cal)[kMaxCards][2]);
  void dec_finish();  // the pending step's stamps into dp_ (the step has ended)
  bool profiling_ = false;
  struct Graphs;
  std::unique_ptr<Graphs> graphs_;
  // the recorded ranking: card 0's per-(layer, expert) decode reads (the route launch adds them; fixed address, so
  // recorded graphs keep counting), the last values taken, and the record they go to
  std::unique_ptr<model::RouteRecord> rec_;
  // the CPU path per card (the pool reads the link's host buffers: destroyed before them)
  bool cpu_on_ = false, cpu_timeout_seen_ = false;
  kernels::MoeCpuLink cpu_link_[kMaxCards];
  std::unique_ptr<kernels::MoeCpuPool> cpu_pool_[kMaxCards];
  void setup_moe_cpu(const ModelOptions &opt);
  uint32_t *hits_ = nullptr;
  std::vector<uint32_t> hits_seen_;
  bool learn_ = true;
  void take_hits(bool keep);
  int graphs_n_ = 0;
  double graphs_ms_ = 0;
  int32_t seg_words_[kernels::kMaxSegs * kernels::kSegRec] = {};  // this step's segment records (host build)
  float *lg_host_ = nullptr;    // pinned [cards][Head:kRows][vocab / cards], the host copy of a step's logits
  int32_t *seg_pin_ = nullptr;  // their pinned copy, the source of the step's first command
  std::vector<kernels::MoeEvents> moe_ev_[kMaxCards];  // [card][layer], last step (when profiling)
  struct Mark {
    StepProfile::Part part;
    sycl::event first, last;
  };
  std::vector<Mark> marks_;  // card 0, last step
};

}  // namespace xenithium::arch::qwen38
