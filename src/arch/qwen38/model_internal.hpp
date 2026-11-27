// Xenithium · qwen38 Model internals shared by model.cpp (the target) and drafter.cpp (the MTP
// drafter). Not a public header.
#pragma once

#include <stdexcept>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/kernels/moe/moe_rows.hpp"
#include "xenithium/kernels/linear_attention/gdn_rows.hpp"
#include "xenithium/kernels/embed/ple.hpp"
#include "xenithium/kernels/attention/qsa.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch::qwen38 {

// A graph's finalize failed (device memory); the host state is the recording's start again.
struct GraphRecordError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// One card's per-step buffers and the sequence slots' states (every slot's copy in one allocation).
struct Model::Card {
  using bf16 = sycl::ext::oneapi::bfloat16;
  bf16 *xa = nullptr, *ya = nullptr, *xm = nullptr, *ym = nullptr;  // [4][hidden]: mixer outputs, block outputs
  int32_t *seg = nullptr;                                           // the step's segment records [kMaxSegs][kSegRec]
  bf16 *rope = nullptr;                                             // [capacity][64]
  kernels::PairWire pw;                                             // GDN / QSA two-card sums
  kernels::MoeScratch msc;                                          // msc.route: [layer] records
  kernels::MoeWire mw;
  const uint8_t **tab = nullptr;                                    // [layer][per_card] expert blobs
  uint8_t *cold = nullptr;                                          // [layer][per_card] 1 = cold (staging)
  kernels::GdnScratch gsc;
  std::vector<kernels::GdnState> gst;                               // [layer] (linear layers)
  float *tree_S = nullptr;  // the GDN tree core's saved states (kernels::gdn_tree_S_elems; DFlash trees)
  kernels::QsaScratch qsc;
  std::vector<kernels::QsaCache> qc;                                // [layer] (full layers)
  float *lp = nullptr;                                              // logprob records [kLpMaxRows][kLpRec]
  kernels::PleState ple_st;
  kernels::PleScratch ple_sc;
  // A decode step's rows by owner (four cards only): the owner sums' wires (pw / mw with their owner mailboxes on) and
  // one mailbox per buffer the owners share.
  struct Own {
    kernels::PairWire pw;       // the GDN / QSA sum
    kernels::MoeWire mw;        // the MoE combine's sum
    kernels::OwnerShare xa;     // the attention mixer's output
    kernels::OwnerShare xm;     // the MLP mixer's output
    kernels::OwnerShare out;    // the final mixer's output
    kernels::OwnerShare feat;   // a drafter feature: the residual streams' mean
    kernels::OwnerShare res;    // the residual streams (before the PLE block, after the final mixer)
    kernels::OwnerShare sel;  // the QSA layers' selected blocks of each owner's rows
  } own;
  // Prefill chunk buffers (prefill.cpp): rows = ModelOptions:prefill_chunk + 4 (a short last
  // slice's 4-row kernels read and write past the chunk's rows into finite padding).
  struct Pf {
    bf16 *res[2] = {nullptr, nullptr};   // [rows][hc * hidden]: the chunk's multi-stream state
    float *lora[2] = {nullptr, nullptr}; // [rows][head lora_ld]
    bf16 *xa = nullptr, *ya = nullptr, *xm = nullptr, *ym = nullptr;  // [rows][hidden]
    int32_t *ids = nullptr;              // [rows] token ids
    int32_t *words = nullptr;            // [slices][kSegRec]: a segment record per 4-row slice (rows.hpp)
    int32_t *lane_words = nullptr;  // the same per lane, each lane's slices from its own first row
    kernels::HcRowsScratch hc;           // the many-row HC mixers' scratch
    kernels::MoeRowsScratch moe;         // the many-row MoE (expert TP only)
    kernels::MoeRowsWire mw;             // its chunk-sized two-card exchange
    kernels::GdnRowsScratch gdn;         // the many-row GDN
    kernels::PairRowsWire pw;            // chunk-sized two-card sums (GDN, QSA)
    kernels::PairSplit split;            // the same sums split over the cards, from 64 rows
    kernels::PairSplit lane[kPfLanesMax];  // the split per lane (its own recv and flag lines; lane 0 = split)
    // the QSA selection shared by owner rows, a flag line of its own per lane (on the lane's line, a peer's next write
    // could land before this card read the out projection's flag value: a wait on == seq)
    kernels::PairSplit sel[kPfLanesMax];
    // The split sums' flag lines (pf.split's): [0] the launches' (the lines pf.split names at load), [1 + i] prompt-call
    // graph flag set i (Model::pf_flag_set). A recorded graph replays its sequence numbers, so it owns its lines: a
    // line then only ever holds that graph's numbers (or 0), and its first wait never matches the line's old value.
    uint32_t *flag_lines[1 + kPfGraphsMax] = {};
    std::optional<sycl::queue> side;     // the split's copies, beside the compute queue
    // The layer-ahead fetch (ModelOptions::prefill_fetch): the side queue copies a layer's cold experts into a VRAM
    // buffer (layer l: buffer l % kPfFetchBufs) while the layers before it compute; the MoE reads ftab (its blob table:
    // the buffer's slot or the expert's own blob) and fflag (the cold flags of what was not fetched, for the staging).
    uint8_t *fbuf[kPfFetchBufs] = {};
    const uint8_t **ftab[kPfFetchBufs] = {};  // [per_card]
    uint8_t *fflag[kPfFetchBufs] = {};        // [per_card]
    // host (pinned): the card's blob table and cold flags as the call sees them, fetched entries already pointing at
    // their buffer slot; per layer the host blobs to fetch, in slot order
    const uint8_t **htab = nullptr;  // [layers][per_card]
    uint8_t *hflag = nullptr;        // [layers][per_card]
    std::vector<std::vector<const uint8_t *>> hsrc;  // [layers]
    kernels::QsaRowsScratch qsa;         // QSA's chunk projections (stage A)
    kernels::PleRowsScratch ple;         // the many-row PLE block
    // The GDN layers' attention input by owner (prefill.cpp pf_gdn_owner_inputs; null = off): each row's x quantised per
    // row for the W8A8 qkvz ([rows][hidden] int8, [rows] scales) and the a | b dots of every card's heads ([rows][card]
    // [ab_n]), at the rows' places in the chunk; and every card's a | b weights stacked per layer ([card][a | b][hidden]
    // bf16, the dots' column order).
    int8_t *gx_q = nullptr;
    float *gx_s = nullptr;
    float *gx_ab = nullptr;
    std::vector<const sycl::ext::oneapi::bfloat16 *> gx_w;
    kernels::PfA8 a8;                    // W8A8 scratch for every many-row int8 dense GEMM (ModelOptions:prefill_dense_a8;
                                         // q [rows][hc width] int8, s [rows][width / 128]; the rows-scratch structs above point at it)
    // Prompt logprobs, many-row head: a group of lp_rows rows at a time; the group's bf16 logits live in moe.y (dead
    // after the layers), its targets / their logits / records here.
    int lp_rows = 0;                     // 0 = no many-row head (no moe_rows scratch: EP2)
    int32_t *lp_tgt = nullptr;           // [lp_rows]
    float *lp_xt = nullptr;              // [lp_rows]
    float *lp_rec = nullptr;             // [lp_rows][kLpRec]
  } pf;
  std::vector<void *> mem;
};

// The drafter's attention shape: the target's QSA shape with the drafter's own KV dtype.
inline kernels::QsaShape mtp_shape(const QsaSet &qs, kernels::QsaKv kv, int c) {
  auto s = qs.shape_of(c);
  s.kv = kv;
  return s;
}

// The MTP drafter on both cards (drafter.cpp). One decoder layer (full attention on the target's
// QSA kernels + MoE), its front and its own final mixer; the target's embedding and lm_head. Its
// per-step activations reuse the target card's xa / ya / xm / ym and scratch (one in-order queue
// per card: a pass never overlaps a target step).
struct Model::Mtp {
  using bf16 = sycl::ext::oneapi::bfloat16;
  static constexpr int kPasses = 3;    // MTP depth this state is sized for
  static constexpr int kTokStride = kernels::kMaxRows;  // int32 per pass in tok
  model::Checkpoint ck;
  std::unique_ptr<load::DenseStore> dense;
  struct Card {
    // weights
    const bf16 *norm_e = nullptr, *norm_h = nullptr;  // [hidden], [hc x hidden]
    kernels::DenseWeights fe, fh;                      // int8 [hidden][hidden]
    kernels::HcWeights attn, mlp, fin;
    kernels::QsaWeights qw;
    kernels::MoeWeights mw;                            // blobs -> tab
    // state
    kernels::HcState st;                               // the pass's multi-stream state (double-buffered)
    kernels::QsaCache qc;                              // the drafter layer's KV (position-addressed; a page range per slot)
    kernels::MoeRoute *route = nullptr;                // the drafter's routing record
    // per-pass buffers
    bf16 *e = nullptr, *en = nullptr;                  // [8][hidden] (a step's rows)
    bf16 *hn = nullptr;                                // [8][hc][hidden]
    float *fe_y = nullptr, *fh_y = nullptr;            // [8][hidden], [8][hc][hidden]
    bf16 *out = nullptr;                               // [8][hidden] final mixer output (lm_head input)
    int32_t *ids = nullptr;                            // [8] pass 1's input tokens
    bf16 *gh = nullptr;                                // [8][hc][hidden] draft_chains: the pass's input rows, gathered
    int32_t *gid = nullptr;                            // [8] draft_chains: their tokens, gathered
    int32_t *tok = nullptr;                            // [kPasses][kTokStride] each pass's picks
    float *logits = nullptr;                           // the passes' lm_head output (own: the target's stay readable)
    int32_t *seg = nullptr;                            // [kPasses][kMaxSegs * kSegRec] segment records (rows.hpp)
    kernels::DraftNoiseRec *noise = nullptr;  // [slots] coupled drafting per slot (draft_gumbel)
    uint8_t *experts = nullptr;                        // [experts][stride] (VRAM)
    const uint8_t **tab = nullptr;                     // [experts] blob pointers (device)
    kernels::DenseWeights dhead;                       // the int2 draft head (this card's vocab slice); w null = the target's lm_head
    kernels::DenseWeights vhead;                       // the draft vocabulary's rows of the target's lm_head (this card's half)
    kernels::W4Mat vhead4;                             // the same rows as int4 g128 RTN (ModelOptions:mtp_head_w4)
  };
  std::unique_ptr<load::DenseStore> vdense;            // the draft vocabulary head's store (from the target checkpoint)
  int sub_k = 0, sub_s0 = 0, sub_n = 0;                // ids [0, sub_k) + [sub_s0, sub_s0 + 128); sub_n rows per card
  bool rot = false;                                    // the drafter's experts in the block-Hadamard basis (its config)
  std::vector<Card> cards;
  std::vector<void *> mem;                             // device allocations (released by ~Model)
  // The passes' copy sources are pinned (a pageable copy is staged by the runtime on the host): seg_host
  // [kPasses][kMaxSegs * kSegRec], ids_host [kMaxRows], noise_pin [slots]; each is rewritten only after the step that
  // read it has finished (the token read between steps).
  int32_t *seg_host = nullptr;
  int32_t *ids_host = nullptr;
  kernels::DraftNoiseRec *noise_pin = nullptr;
  std::vector<kernels::DraftNoiseRec> noise_host;  // [slots] the host values of cards[].noise
  bool noise_dirty = true;                             // noise_host changed since the last upload
  int last_rows = 0;                                   // rows of the last pass 1
  // The last pass 1's segments (several slots): (slot, position of its first row), in row order, laid out by
  // p1_seg_rows: rows per segment, or a kernels::seg_het word for unequal segments. draft_chain follows its row's
  // segment (p1_row).
  std::vector<std::pair<int, int64_t>> p1_segs;
  int p1_seg_rows = 0;
  // The slot and position of pass 1's row `row`.
  std::pair<int, int64_t> p1_row(int row) const {
    int s = 0, j = row;
    if (p1_seg_rows < 0) {
      const int b = kernels::seg_het_b(p1_seg_rows), kb = kernels::seg_het_k(p1_seg_rows) * b, b2 = kernels::seg_het_b2(p1_seg_rows);
      s = row < kb ? row / b : kernels::seg_het_k(p1_seg_rows) + (row - kb) / b2;
      j = row < kb ? row - s * b : (row - kb) - (s - kernels::seg_het_k(p1_seg_rows)) * b2;
    } else {
      s = row / p1_seg_rows;
      j = row % p1_seg_rows;
    }
    const auto &[slot, pos0] = p1_segs.at(static_cast<size_t>(s));
    return {slot, pos0 + j};
  }
  // Pass graphs (as the target's step graphs): one recorded graph per card per key. The key holds
  // every host value a pass bakes into its launches: pass slot, rows, the input pointers (card 0's;
  // card 1's follow in lockstep), the wire parities, the QSA grid bound bucket and the drafter's HC
  // state indices at the start. The end parities / indices are stored and set on replay.
  using Exec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  using GraphKey = std::tuple<int, int, int, const void *, const void *, int, int, int64_t, int, int, int, int>;  // + seg_rows, kv_only
  struct Hc {
    int cur = 0, lcur = 0, lora_ld = 0;
  };
  struct Graph {
    std::vector<Exec> exec;  // [card]
    int end_pair = 0, end_moe = 0;
    std::vector<Hc> end_hc;  // [card]
    // What the pass was recorded from (re-recorded at other bound buckets).
    std::vector<Hc> start_hc;  // [card]
    std::vector<const sycl::ext::oneapi::bfloat16 *> hidden;
    std::vector<const int32_t *> ids;
    uint64_t used = 0;  // Model::graph_tick_ at the last replay (the room's LRU order)
  };
  std::map<GraphKey, Graph> graphs;
  int graphs_n = 0;
  double graphs_ms = 0;
};

// The state of one prefill call between its parts (split calls, prefill_begin). Nothing
// here is touched by a decode step: the chunk's rows live in Card::Pf, its wires (pf_pair_parity_,
// pf_moe_parity_) are its own, and the slot is not decoding. The head's HC indices are taken
// relative to their value at the start, so steps between the parts keep their own sequence.
struct Model::PfRun {
  std::vector<int32_t> tokens;
  int m = 0, row_base = 0, slices = 0;
  int32_t next_after = -1;
  std::function<void(int, int)> on_rows;
  int slot = 0;
  int64_t pos0 = 0;
  int gp0 = 0;
  std::vector<kernels::HcState> cs;  // the chunk's HC state per card
  int start_cur[kMaxCards] = {}, start_lcur[kMaxCards] = {};
  int l = 0;                   // the next layer
  bool ple_rows_due = false;   // the chunk's PLE rows wait for Model::pf_ple_rows (made as the PLE layer is next)
  // While the call's layers record into a graph: the QSA grid bound every batch takes (the key's bucket, so the
  // launches do not depend on the position); unset = each batch's own (kernels::qsa_rows).
  std::optional<int64_t> qsa_bound;
  // the layer-ahead fetch (pf_fetch_some): on for this call; per card the next layer to fetch, its next blob, the last
  // copy queued, and per buffer the copy that completes its layer
  bool fetch = false;
  int fq_l[kMaxCards] = {};
  size_t fq_j[kMaxCards] = {};
  sycl::event fq_ev[kMaxCards];
  sycl::event fetched[kMaxCards][kPfFetchBufs];
  // Model::pf_layers_owner: the call's rows by owner, in lanes. Empty: every card holds every row.
  struct Lane {
    int r0 = 0, m = 0;         // the call's rows [r0, r0 + m)
    int w0 = 0;                // its first slice record in Pf::lane_words
    int gp0 = 0;               // the slot's GDN parity as the lane's rows start (each lane commits as a call does)
    int pend_kind = -1;        // the module sum the lane's next mixer combines: -1 none, 0 GDN, 1 QSA, 2 MoE
    uint32_t pend_seq = 0;     // that sum's sequence number on the lane's flag lines
    uint32_t share_seq = 0;    // the lane's last share's
  };
  std::vector<Lane> lanes;
  int moe_lanes = 1;           // the lanes one MoE call spans (pf_lanes_moe)
  // The MoE input by owner (W4A8 blocks): each row's owner routes it and quantises it to int8 after its MLP mixer, and
  // shares the routing and the int8 row instead of the bf16 one (pf_lane_mixers, kernels::moe_rows_route).
  bool moe_routed = false;
  // The GDN layers' attention input by owner: each row's owner quantises it for the W8A8 qkvz and dots it with every
  // card's a | b, and shares those instead of the bf16 row (pf_gdn_owner_inputs).
  bool gdn_owner = false;
  // XEN_PF_PROFILE (see pf_mark)
  std::vector<std::pair<int, sycl::event>> dmarks;
  std::chrono::steady_clock::time_point t_sub0, tick;
  double sub_ms = 0;
  double ph[7] = {};
  std::vector<kernels::MoeEvents> moe_ev, moe_ev1;  // card 0, card 1
  // XEN_PF_PROFILE=1 / 2 on profiling queues: card 0's GDN and QSA launches by stage (pf_finish)
  kernels::GdnTrace gdn_trace;
  kernels::QsaTrace qsa_trace;
  // XEN_PF_PROFILE: each split two-card sum's events per card; kind 0 gdn, 1 qsa, 2 moe;
  // prev[c]: the card's kernel before it (the partial's producer).
  struct SplitRec {
    int kind = 0;
    kernels::PairSplitEvents ev;
    sycl::event prev[kMaxCards];
  };
  std::vector<SplitRec> split_ev;
  // A split sum whose peer rows are still arriving (pair_split_finish without its wait): the consumer mixer runs the
  // card's own rows first, then pf_split_wait. 0 = none pending.
  uint32_t pend_seq = 0;
  int pend_rec = -1;  // its split_ev record (profile), or -1
  // XEN_PF_PROFILE=3: word indices into Model:pf_stamps_[card]. smarks: card 0's phase
  // stamps (phase, word); cal: each card's first and last stamp (events, words) for cycles -> ns;
  // split_span: per split sum the 4 wait-span words per card; pend_span: the deferred sum's.
  std::vector<std::pair<int, int>> smarks;
  int stamp_n[kMaxCards] = {};
  sycl::event cal[kMaxCards][2];
  int cal_at[kMaxCards][2] = {};
  struct SplitSpan {
    int kind = 0;
    int at[kMaxCards] = {};
  };
  std::vector<SplitSpan> split_span;
  uint64_t *pend_span[kMaxCards] = {};
};

// The recorded prompt-call graphs (prefill.cpp, "prompt-call graphs").
struct Model::PfGraphs {
  using Exec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  struct Hc {
    int cur = 0, lcur = 0, lora_ld = 0;
  };
  struct Entry {
    std::vector<Exec> exec;       // [card]
    std::vector<Hc> end_hc;       // [card]: the call's HC state indices after the recorded layers
    int end_pair = 0, end_kv = 0, end_moe = 0;  // the prefill wires' parities after them
    int flag_set = 0;             // its split flag lines (Card::Pf::flag_lines index; 0 = none: no split sum)
    uint64_t used = 0;            // Model::graph_tick_ at the last replay (the LRU order, with the step graphs')
  };
  std::map<PfGraphKey, Entry> map;
  std::map<PfGraphKey, int> seen;  // the calls of each key not recorded yet
  std::vector<int> free_flag_sets = [] {
    std::vector<int> v;
    for (int i = kPfGraphsMax; i >= 1; --i) v.push_back(i);
    return v;
  }();
  // drops entry `it`; its flag set goes back to the free list
  void erase(std::map<PfGraphKey, Entry>::iterator it) {
    if (it->second.flag_set > 0) free_flag_sets.push_back(it->second.flag_set);
    map.erase(it);
  }
};

}  // namespace xenithium::arch::qwen38
