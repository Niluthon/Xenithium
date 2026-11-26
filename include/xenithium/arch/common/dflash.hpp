// Xenithium · the DFlash 2 block drafter on the cards: an alternative to a target's MTP layer (ModelOptions::dflash_dir;
// server XEN_DRAFTER_MODEL), `DFlash2DraftModel` checkpoints (for Qwen3.8-27B, arch qwen35, and Qwen3.8 Flash-Next,
// arch qwen38; bf16, or GPTQ int4 Linears: dflash_dense_plan). Shared by the archs: the target hands over its sizes,
// embedding tables and lm_head rows (DFlashTarget) and captures its own features (capture() copies a single-stream
// residual's columns; an arch with several residual streams fills step_features / prefill_features with its own kernel).
//
// Semantics (config.json + the checkpoint):
//   features  F_p = [h_p(l_0) | ... | h_p(l_4)]: the target's residual after decoder layers dflash_config.target_layer_ids
//             (the 27B: 5, 19, 33, 47, 61; Flash-Next: 5, 15, 25, 35, 45, the mean of its 4 hyper-connection streams) at
//             every verified position p (bf16; 5 x hidden)
//   context   T_p = rms_norm(fc . F_p, hidden_norm) (fc [hidden][5 hidden]); per drafter layer l: K_l(p) = rope_p(k_norm(
//             k_proj_l T_p)), V_l(p) = v_proj_l T_p, kept per sequence for the sliding window (2,048 positions)
//   a block   of B rows at positions s .. s + B - 1: row 0 = the last verified token (the anchor), rows 1.. = mask_token_id;
//             x = embed(ids) (the target's embedding), then 5 Qwen3 decoder layers (plain RMSNorm, q / k norms, rope theta
//             theta from config.json over all 128 dims (27B 1e7, Flash-Next 1e4), 32 q / 8 kv heads, SwiGLU (17,408 / 7,680)), each sublayer wrapped in the grouped dynamic two-tap
//             convolution (prepare on its input, finish on its output; kernels::df_conv); attention is NON-causal over the
//             block and reads the context keys at positions (s + j - 2,048, s) for row j
//   drafts    h = rms_norm(x, norm); logits = the target's lm_head (h); per row j >= 1 the 16 best tokens, then the candidate
//             selector walks from the anchor: draft_j = argmax_b logit_j(b) + <P[draft_{j-1}] * proj(h_j), S[b]> (greedy:
//             the drafts are deterministic, so the engine's acceptance rule p(draft) keeps sampling exact)
// The block's own K / V are never kept: the next pass's context rows are the verified rows, from the target's features.
//
// Placement (two cards): q (+ k | v) by head rows, o and down along k (+ the fp32 pair all-reduce), gate | up by rows, fc along
// k (each card holds 2.5 of the 5 features' columns), the convolution projections and the selector's hidden projection
// replicated, the codebooks (2 x 127 MB bf16) on card 0 (the walk runs there). Matrices int8 per row (dense_bytes 1) or bf16.
// Passes (pass()): the pinned host words (block ids, segment records, anchors) then one recorded SYCL graph per card per
// shape (segments, context rows, block rows, first context row, wire parity), replayed together; the stages go out
// stage-major over the cards and no host-blocking call sits between a card's mailbox sum and its peer's (a device-to-host
// copy in the card loop would hold card 1's submission until card 0's 1 s mailbox wait times out).
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

#include "xenithium/arch/common/dflash_plan.hpp"
#include "xenithium/arch/common/lm_mat.hpp"
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/drafter/dflash.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/rt/fabric.hpp"

namespace xenithium::arch {

struct DFlashConfig {
  // DSpine (`DFlashDraftModel` or `DSpineDraftModel` + the code bridge, kernels/drafter/dspine.hpp): no convolution,
  // causal inside the block, the bridge after every layer, ds_select's scores.
  enum class Kind : uint8_t { dflash2, dspine };
  Kind kind = Kind::dflash2;
  int hidden = 0, layers = 0, q_heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
  int block = 0, conv_taps = 0, conv_group = 0, sel_rank = 0, sel_topk = 0, window = 0, target_layers_total = 0;
  int trained_block = 0;  // the checkpoint's block_size; block = the rows a pass may draft here (<= trained_block)
  double bridge_eps = 0;  // DSpine: selected_prefix_code_bridge.eps
  int64_t vocab = 0, mask_id = -1, max_positions = 0;
  double eps = 0, theta = 0;
  bool causal = false;
  std::vector<int> target_layers;
  // Throws naming the key or the broken invariant (the kernels' constants: head dim 128, two taps, top-16, ...).
  // max_block: the rows a pass may draft here (8: the 27B's verify per sequence; 16 with DFlashOptions::wide)
  static DFlashConfig load(const std::string &dir, int max_block = 8);
  // config.json names a block drafter (drafter_archs.hpp)
  static bool is_dflash(const std::string &dir);
};

// The target model's side: its sizes, and per card the embedding table the block rows gather from (device or pinned
// host memory) and its lm_head rows (vocabulary-parallel: card c holds ids [c n, (c + 1) n)). The tables must stay
// valid while the drafter lives (recorded pass graphs keep the pointers).
struct DFlashTarget {
  int hidden = 0, layers = 0;
  int64_t vocab = 0;
  std::vector<int64_t> eos;
  std::vector<const sycl::ext::oneapi::bfloat16 *> embed;
  std::vector<LmMat> lm;
  // A target whose embedding is not a bf16 table (arch flint's 8-bit affine one) hands its own gather instead: (queue,
  // card, ids (pinned host), rows, out bf16 [rows][hidden]); submits only (recorded into the pass graphs). Set = embed
  // unused.
  std::function<sycl::event(sycl::queue &, int, const int32_t *, int, sycl::ext::oneapi::bfloat16 *)> gather;
};

struct DFlashOptions {
  // drafter matrices: 1 = int8 per row, 2 = bf16, 4 = int4 by round-to-nearest at load (q, o, gate | up and down as W4
  // g128 on the W4A16 decode GEMM, the rest int8 per row: fc and k | v also run the many-row context passes, the
  // convolution projections and the selector are small). Drafts only: the verify stays exact.
  int dense_bytes = 1;
  int threads = 16;
  int vocab = 0;        // the draft vocabulary (MtpOptions::vocab): 0 = the target's lm_head
  bool head_w4 = false; // the draft vocabulary's rows from the target's int4 lm_head.qweight
  int slots = 1;
  int rows = 0;  // block rows a pass holds (0 = kRows; the target's step rows, 16 unless 32-row steps)
  int prefill_rows = 2048;  // most prompt rows one prefill chunk captures features for
  bool profile = false;     // pass(): wait on both cards after every stage and add its wall time to stage_ms() (diagnostic)
  // A context-only pass (block_rows 0: no drafts, e.g. at depth 0 with many streams) only stashes its feature rows, up
  // to this many per slot; settle() runs them as one many-row context pass (prefill_ctx's kernels) before anything reads
  // the slot's ring (a drafting pass, a snapshot, a prefill). 0 = every pass runs.
  int ctx_defer = 64;
  // A pass returns a segment's drafts while the product of their confidences (each one's softmax probability among its
  // block row's top-16 logits) stays >= this; the verify then carries fewer rows when the drafter is unsure (exact
  // either way: the verify accepts only the target's own picks). Single-sequence passes only. 0 = every draft.
  double conf_cut = 0.02;
  // > 0 = a single-sequence greedy pass drafts a best-first tree of up to this many nodes (1..7; selector edges) instead
  // of the chain (and its conf_cut); tree_cut > 0: only nodes whose path probability at temperature 1 is >= it join
  // (DFlash 2's df_tree; DSpine's ds_tree stops below it). pass() then returns the tree's tokens in row order (preorder,
  // kernels::tree_preorder) and tree_parents() their parent rows. For DFlash 2 checkpoints (kernels::df_tree) and DSpine
  // ones (kernels::ds_tree).
  int tree = 0;
  double tree_cut = 0;
  // A pass of a step of up to this many sequences (pass()'s step_seqs) drafts a tree per segment; 1 = one sequence only.
  // A sampled segment's tree is the greedy one (no noise: the walk follows the sampler's own draws).
  int tree_seqs = 1;
  // The tree's temperature (each node's 16 candidate scores / tree_temp before their log-softmax: the weights that
  // decide breadth vs depth). 0 = by drafter: DSpine 1.5, DFlash 2 1.
  double tree_temp = 0;
  // > 0 = passes of up to 16 block rows (DSpine's whole trained block; a 16-row verify of one sequence) and their trees
  // of up to tree_wide nodes (a pass of more than 8 block rows; 1..15). The model's 16-row GDN log goes with it
  // (ModelOptions::seg_rows 16).
  int wide = 0;
  int tree_wide = 15;
};

class DFlash {
 public:
  using bf16 = sycl::ext::oneapi::bfloat16;
  static constexpr int kRows = kernels::kMaxRows;

  // capacity: the target's tokens per slot (rope positions); target_dir: the target checkpoint (the draft-vocabulary rows).
  DFlash(Fabric &fab, const std::string &dir, const std::string &target_dir, const DFlashTarget &target, int64_t capacity, const DFlashOptions &opt);
  DFlash(const DFlash &) = delete;
  DFlash &operator=(const DFlash &) = delete;
  ~DFlash();

  const DFlashConfig &config() const { return cfg_; }
  int max_depth() const { return cfg_.block - 1; }  // drafts per pass at most (the checkpoint's block - 1)
  // ---- the target's features ----
  // j of target decoder layer l (-1: not a feature layer)
  int feature_of(int layer) const;
  int feat_cols() const { return fcols_; }  // per card: global columns [c F, (c + 1) F) of [5 hidden]
  // Card c's columns of feature j from res bf16 [rows][hidden] into dst [rows][feat_cols()] (submits on q).
  void capture(int c, sycl::queue &q, int j, const bf16 *res, int rows, bf16 *dst) const;
  // The same from a residual of `streams` hyper-connection streams ([rows][streams][hidden], arch qwen38): the feature
  // is the streams' mean (the checkpoint's training capture).
  void capture_mean(int c, sycl::queue &q, int j, const bf16 *res, int streams, int rows, bf16 *dst) const;
  bf16 *step_features(int c) const { return cards_[static_cast<size_t>(c)].feat_step; }     // [kRows][F]: the last verify's rows
  bf16 *prefill_features(int c) const { return cards_[static_cast<size_t>(c)].feat_pf; }    // [prefill_rows][F]: the chunk's rows
  // A verify of unequal segments leaves step_features in its row order; save_step_features(rows) copies them aside once,
  // stage_step_features(map, n) then writes saved row map[i] to row i (i < n <= kRows) for the next pass (submitted on
  // the cards' queues, before the pass).
  void save_step_features(int rows);
  void stage_step_features(const int (&map)[kRows], int n);
  // ---- passes ----
  // A segment: its slot, the position of its first context row (the verify's rows) and of the block (s), the anchor token.
  struct Seg {
    int slot = 0;
    int64_t ctx_p0 = 0, start = 0;
    int32_t anchor = 0;
  };
  // ctx_rows (0..8 per segment, from step_features: row i x ctx_rows + r = segment i's context row r) go to the context;
  // then (block_rows >= 2) a block per segment and its block_rows - 1 drafts, per segment (block_rows 0: context only).
  // ctx_row0: the first segment's first context row in step_features (a step's segments drafted over several passes).
  // segs x block_rows <= kRows. Waits; draft_ms() = its wall time. step_seqs: the sequences of the step the pass drafts
  // for (trees up to DFlashOptions::tree_seqs); kChainsOnly: the step verifies no trees, the pass drafts chains.
  static constexpr int kChainsOnly = std::numeric_limits<int>::max();
  std::vector<std::vector<int32_t>> pass(const std::vector<Seg> &segs, int ctx_rows, int block_rows, int ctx_row0 = 0, int step_seqs = 1);
  // After a tree pass (DFlashOptions::tree), the verify rows' parent rows of segment seg: [0] = -1 (the anchor's row),
  // [r] = draft r's parent row (drafts in pass() order are rows 1..); empty after a chain pass. A prefix of the drafts
  // is a tree.
  const std::vector<int> &tree_parents(int seg = 0) const {
    static const std::vector<int> none;
    return seg >= 0 && static_cast<size_t>(seg) < tree_par_.size() ? tree_par_[static_cast<size_t>(seg)] : none;
  }
  void set_tree(int budget, double cut, int seqs = 1) {
    opt_.tree = budget;
    opt_.tree_cut = cut;
    opt_.tree_seqs = seqs;
  }
  int tree_seqs() const { return opt_.tree_seqs; }
  int tree_budget() const { return opt_.tree; }
  // Coupled drafts for a sampled request. The slot's later passes pick each draft j (0-based) as argmax over the
  // selector's candidates of score x inv_t + util::gumbel(key, pos + j, token), the noise of the target's draw at that
  // output position (core::Sampler::draft_noise); inv_t 0 = the argmax pick (greedy requests, bit for bit).
  void set_noise(int slot, uint64_t key, uint64_t pos, float inv_t);
  // A prefill chunk's m rows at pos0 .. pos0 + m - 1 of the slot into the context, from prefill_features. The caller lends
  // its chunk buffers: t / tn bf16 [m][hidden], part fp32 [m][hidden], ykv fp32 [m][>= 2 hkv x 128], the rows wire + parity.
  struct Lend {
    bf16 *t = nullptr, *tn = nullptr;
    float *part = nullptr, *ykv = nullptr;
    const kernels::PairRowsWire *wire = nullptr;
  };
  // row0: the m rows start at prefill_features row row0 (a chunk's context in two parts).
  void prefill_ctx(int m, int64_t pos0, int slot, const std::vector<Lend> &lend, int parity, int row0 = 0);
  // The slot's deferred context rows at positions < pos into its ring (one many-row pass), the rest dropped (rewritten
  // later or never read): pos = INT64_MAX before reading the ring, 0 before replacing it.
  void settle(int slot, int64_t pos);
  int deferred_rows(int slot) const { return pend_.empty() ? 0 : pend_[static_cast<size_t>(slot)].n; }
  int flushes() const { return flushes_; }  // settle() passes run so far
  double draft_ms() const { return ms_; }
  // profile: summed wall ms per stage over the passes so far (stage_names() order), passes counted; reset with clear_stages()
  static constexpr int kStages = 10;
  static const char *const *stage_names();
  const double *stage_ms() const { return stage_ms_; }
  int stage_passes() const { return stage_passes_; }
  int graphs() const { return graphs_n_; }                                // pass graphs recorded so far
  int graphs_held() const { return static_cast<int>(graphs_.size()); }  // of them still held (at most XEN_DRAFT_GRAPHS)
  int graphs_evicted() const { return graphs_evicted_; }                // the least recently used, dropped at the cap
  double graphs_ms() const { return graphs_ms_; }                     // their recording time in all
  void clear_stages() {
    for (double &x : stage_ms_) x = 0;
    stage_passes_ = 0;
  }
  size_t card_bytes(int c) const;  // weights + rings + scratch on card c
  uint32_t wire_error();
  int draft_vocab() const { return head_w_.empty() ? 0 : sub_k_ + 128; }
  int draft_vocab_s0() const { return sub_s0_; }  // the specials' first token id (index sub_k -> this)
  // checks (probe): card 0's last pass outputs (waits)
  std::vector<uint16_t> read_hidden(int rows);   // after the final norm [rows][hidden]
  std::vector<uint64_t> read_keys(int rows);     // top-16 keys [rows][16]
  std::vector<float> read_hp(int rows);          // the selector's hidden projection [rows][rank]
  // the slot's ring K / V of layer l on card c ([ring][hkv / card][128] bf16)
  std::vector<uint16_t> read_ring(int c, int layer, int slot, bool v);
  int ring_tokens() const { return ring_; }
  // For the prefix cache: one slot's context rings (every layer's K and V) on card c as one byte range: ring_slot_bytes;
  // ring_copy submits the copies of the slot's rings into host (save) or back from it (restore) on q.
  size_t ring_slot_bytes(int c) const;
  // The slot's ring buffers on card c, every layer's K then V (qwen38's snapshots gather pieces).
  std::vector<std::pair<void *, size_t>> ring_pieces(int c, int slot) const;
  void ring_copy(int c, sycl::queue &q, int slot, void *host, bool save) const;
  const std::string &dir() const { return dir_; }

 private:
  struct Layer {
    const bf16 *in_ln = nullptr, *post_ln = nullptr, *qn = nullptr, *kn = nullptr, *abase = nullptr, *mbase = nullptr;
    kernels::DenseWeights kv, aproj, mproj;
    LmMat q, o, gate_up, down;  // int8 / bf16 dense tiles, or int4 (DFlashOptions::dense_bytes 4)
    kernels::DenseWeights bread, bwrite;  // DSpine: the bridge's read R_l [code][hidden] and write W_l [hidden][rank] (bf16)
    kernels::DfRing ring;
  };
  struct Card {
    kernels::DenseWeights fc, sel;
    const bf16 *hnorm = nullptr, *fnorm = nullptr;
    std::vector<Layer> layers;
    bf16 *feat_step = nullptr, *feat_pf = nullptr;
    bf16 *feat_save = nullptr;  // the verify's feature rows, kept while they are re-laid
    float *part = nullptr, *dyn = nullptr, *yq = nullptr, *ykv_b = nullptr, *ykv_c = nullptr, *apart = nullptr, *gu = nullptr, *logits = nullptr, *hp = nullptr;
    bf16 *tnx = nullptr;
    float *ykv_cb = nullptr;
    float *silu = nullptr;  // kernels::swiglu_table (int4 gate | up: swiglu in the GEMM's epilogue)
    bf16 *t = nullptr, *tn = nullptr, *res = nullptr, *xn = nullptr, *xc = nullptr, *mix = nullptr, *mlp = nullptr, *hout = nullptr;
    bf16 *qb = nullptr, *kb = nullptr, *vb = nullptr, *attn = nullptr, *h = nullptr;
    bf16 *cos = nullptr, *sin = nullptr;
    void *topk_scratch = nullptr;
    uint64_t *keys = nullptr;
    int32_t *rec = nullptr, *pf_rec = nullptr, *anchors = nullptr, *drafts = nullptr;
    kernels::DraftNoiseRec *noise = nullptr;  // card 0's selector: the segments' noise records
    const bf16 *pred_cb = nullptr, *succ_cb = nullptr;  // card 0
    // DSpine: the bridge's packed [G_q; G_p; A] (bf16 [3 rank][code]) and gate bias; its activations; card 0: the
    // vocabulary tables TG [n][rank] / TSV [n][2 rank] over the draft head's candidate indices
    kernels::DenseWeights bpack;
    const bf16 *bbias = nullptr, *tg = nullptr, *tsv = nullptr;
    float *brms = nullptr, *bproj = nullptr, *bpacked = nullptr;
    bf16 *bcode = nullptr, *bmsg = nullptr;
    kernels::PairWire wire;
    // four cards: the pass's sums of kOwnerSumRows rows and more as an owner sum + a share by copies (dflash.cpp, submit)
    kernels::PairWire own_wire;
    kernels::OwnerShare own_share;
    kernels::DfTopkWire tw;
    // the deferred context: feature rows [slots][ctx_defer][F], the flush's own scratch and wire
    bf16 *backlog = nullptr, *fl_t = nullptr, *fl_tn = nullptr;
    float *fl_part = nullptr, *fl_ykv = nullptr;
    int32_t *fl_rec = nullptr;  // [slots][kSegRec]
    kernels::PairRowsWire fl_wire;
    std::vector<void *> mem;
  };
  struct Pend {
    int64_t p0 = 0;  // the position of the slot's first deferred row
    int n = 0;
  };
  std::vector<Pend> pend_;
  int fl_par_ = 0, flushes_ = 0;
  void defer(const std::vector<Seg> &segs, int ctx_rows, int ctx_row0);
  void flush(int slot, int m);
  int confident(const std::vector<int32_t> &drafts, int row0) const;
  Fabric &fab_;
  std::string dir_;
  DFlashConfig cfg_;
  DFlashTarget tgt_;
  DFlashOptions opt_;
  int64_t cap_ = 0;
  int ring_ = 0, fcols_ = 0, ipc_ = 0, hq_ = 0, hkv_ = 0, par_ = 0;
  std::unique_ptr<load::DenseStore> dense_, head_store_;
  std::vector<Card> cards_;
  std::vector<size_t> bytes_;
  bf16 *codebook_ = nullptr;  // DSpine: the bridge's codebook bf16 [vocab][code], pinned host (anchor rows)
  void load_dspine(const std::string &dir, const std::string &target_dir);
  int rows_ = kRows;  // DFlashOptions::rows
  int32_t *host_ = nullptr;  // pinned: block ids [kRows], records [kMaxSegs][kSegRec], anchors [kMaxSegs], drafts [kMaxSegs][kRows]
  std::vector<LmMat> head_w_;  // the draft vocabulary's lm_head rows per card (empty: the target's lm_head)
  int head_n_ = 0, sub_k_ = 0, sub_s0_ = 0;
  std::vector<kernels::DraftNoiseRec> noise_;  // per slot (set_noise)
  double ms_ = 0;
  // recorded pass graphs, per (segments, context rows, block rows, first context row, start parity)
  using GraphKey = std::tuple<int, int, int, int, int, int>;  // + the tree budget (0 = the chain)
  std::vector<std::vector<int>> tree_par_;  // per segment of the last pass
  struct Graph {
    std::vector<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>> exec;
    int end_par = 0;
    uint64_t used = 0;  // the last replay (graph_tick_): the least recently used is evicted at the cap
  };
  // at most XEN_DRAFT_GRAPHS (default 96) entries; each holds ~10 MiB of device memory per card
  std::map<GraphKey, Graph> graphs_;
  uint64_t graph_tick_ = 0;
  int graphs_n_ = 0, graphs_evicted_ = 0;
  double graphs_ms_ = 0;
  void submit(int ns, int ctx_rows, int block_rows, int ctx_row0, int tree = 0);
  double stage_ms_[kStages] = {};
  int stage_passes_ = 0;
  int wire_parity() { return (par_ ^= 1) ^ 1; }
};

// The drafter's tensors with their placement (see above). dense_bytes 1 = int8 per row, 2 = bf16, 4 = int4 by RTN (q, o,
// gate | up, down; the rest int8). A GPTQ `.qweight` Linear is packed as w4 (when the path would RTN) or dequantised
// to the dense precision (fc and the other dense-only mats); the store is keyed as the canonical `.weight` name.
inline std::vector<model::DenseSpec> dflash_dense_plan(const model::Checkpoint &ck, const DFlashConfig &cfg, int dense_bytes) {
  return dflash_dense_plan(ck, cfg.layers, dense_bytes, cfg.kind == DFlashConfig::Kind::dspine);
}

}  // namespace xenithium::arch
