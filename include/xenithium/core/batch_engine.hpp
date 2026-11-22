// The serving engine's step loop over any model that can run several sequences per decode
// step. Model-agnostic: the model is reached only through StepModel; each arch
// provides an adapter (qwen38: src/arch/qwen38/step_model.cpp).
//
// One thread owns the model. Per loop:
//   1. take new submits and cancels (a cancel frees its slot at once);
//   2. drop requests past the queue deadline, admit waiting requests into free slots (the
//      admission policy: fifo or sjf, core/admission.hpp);
//   3. one prefill call for the admitted request with the fewest prompt tokens left (shortest
//      remaining first, so prompts finish one after another instead of all at the end).
//      While other requests decode, the call is sized to the stall budget (prefill_stall_ms): the
//      decoding streams wait at most about that long per loop. A request whose prompt completes
//      gets its first token from that call and joins this loop's step (4). A short prompt's single call
//      predicted longer than prefill_part_ms while others decode runs in parts of whole layers, one part per loop;
//   4. one shared decode step over every request that has a token to feed (one row each); with a
//      drafter (StepModel::draft_depth(1) > 0) a speculative step instead: each request's token and
//      its drafts verified in one step, the accepted prefix plus one more token emitted (several
//      requests in one step);
//   5. retire finished requests and hand the step's events to the sink (one call).
// With StepModel::kv_pool_tokens > 0 the slots share a pool of KV pages. Admission reserves a
// prompt's pages (+ max_rows of headroom) before the request takes a slot (it waits while the pool cannot); before each
// decode step every decoding request reserves the positions the step and its drafts write, oldest admitted first. When
// the pool is out the youngest other request is preempted: its pages go back at once, and after the step it leaves its
// slot and waits (ahead of new requests) to be prefilled again over its prompt + every token it emitted, then continues
// (no output is repeated; its first token after the recompute comes from that prefill).
// The prefix cache (StepModel::snap_capacity() > 0, BatchConfig::prefix_cache). A finished request
// leaves its sequence on its slot (parked: its KV pages, its live state, and the model's snapshots of its state taken after
// its prefill calls at multiples of snap_align(), thinned per slot and held in a store of snap_capacity()). Admission looks
// for the longest prefix of the new prompt that a slot holds a usable state for (any slot's snapshot; a parked slot's live
// state when it is made of prefill rows at the alignment, or with prefix_live; with a drafter the token after the prefix
// must match too) and continues it: in place on a parked slot (when that keeps at least half of what the slot could
// continue), else as a copy of the source's KV pages into another slot (snap_fork; the source may be running). The
// prompt's rest is prefilled in calls that keep a cold prefill's numerics (pf_call_rows), and a request's prefix_mark cuts
// a call near there (pc_cut: its snapshot is where the conversation's next turn starts to differ). Parked sequences give
// their pages up first (least recently used) when admission or a decode step needs them, before any running request is
// preempted. Prompt-logprob requests start cold. Alone, every output is bit-identical to the cache off.
// Greedy requests use the model's own picks (no logits leave the device); sampling,
// penalties, logit_bias, logprobs and constraints read the step's logits on the host
// (core:Sampler), one host thread per row.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xenithium/core/admission.hpp"
#include "xenithium/core/engine.hpp"
#include "xenithium/core/sampler.hpp"

namespace xenithium::core {

class StepModel {
 public:
  virtual ~StepModel() = default;
  virtual int slots() const = 0;            // sequences the model holds state for
  virtual int64_t capacity() const = 0;     // tokens per slot (prompt + output)
  virtual int64_t vocab() const = 0;        // logits row length
  virtual int max_rows() const = 0;         // rows per decode step (one per sequence here)
  // Rows a step of `sequences` sequences may carry, <= max_rows(): a model whose largest step shape pays only from some
  // sequence count on keeps the smaller budget below it (the drafts per sequence and the unequal layouts follow it).
  virtual int step_rows_at(int sequences) const { (void)sequences; return max_rows(); }
  virtual int prefill_chunk() const = 0;    // most prompt tokens per prefill call
  virtual void begin(int slot) = 0;         // a fresh sequence on the slot (position 0)
  // n prompt tokens (1..prefill_chunk) on the slot after what it holds; then prefill_pick /
  // prefill_logits read the row of the last of them. next: the prompt token after these n (-1 =
  // they end the prompt); a drafter needs it for the call's last row.
  virtual void prefill(int slot, const int32_t *tokens, int n, int32_t next) = 0;
  // The same with every row's logits (prompt logprobs): on_rows(row0, rows, logits [rows][vocab])
  // for consecutive groups of rows, row0 counted from tokens[0]. Only when prompt_logprobs().
  using RowsFn = std::function<void(int row0, int rows, const float *logits)>;
  virtual bool prompt_logprobs() const { return false; }
  virtual void prefill_rows(int slot, const int32_t *tokens, int n, int32_t next, const RowsFn &on_rows) {
    (void)slot, (void)tokens, (void)n, (void)next, (void)on_rows;
  }
  // Logprobs computed on the model's side: only small per-row results leave it instead of
  // full logits rows. When device_logprobs():
  //   row_logprobs: the raw logprob of targets[r] (< 0: no target, out[r] untouched) with rank and the
  //     top_n best, for the rows of the last step / verify (rows in picks() order) or, inside
  //     prefill_each's callback, of that group of prompt rows;
  //   prefill_logprobs: the same for the row of the last prompt token of the last prefill call;
  //   prefill_each: prefill with a callback after each group of rows (row0 counted from tokens[0]).
  using EachFn = std::function<void(int row0, int rows)>;
  virtual bool device_logprobs() const { return false; }
  virtual void row_logprobs(int rows, const int32_t *targets, int top_n, TokenLogprob *out) { (void)rows, (void)targets, (void)top_n, (void)out; }
  virtual void prefill_logprobs(int32_t token, int top_n, TokenLogprob &out) { (void)token, (void)top_n, (void)out; }
  virtual void prefill_each(int slot, const int32_t *tokens, int n, int32_t next, const EachFn &on_rows) {
    (void)slot, (void)tokens, (void)n, (void)next, (void)on_rows;
  }
  // Prefill with every row's logprobs from one many-row head pass per chunk (instead of
  // prefill_each + row_logprobs per 4-row group): out[r] for row r of the n tokens scores targets[r]
  // (< 0: none, out[r] untouched). Returns false (nothing run) when the model cannot; the engine then
  // takes prefill_each.
  virtual bool prefill_row_logprobs(int slot, const int32_t *tokens, int n, int32_t next, const int32_t *targets, int top_n, TokenLogprob *out) {
    (void)slot, (void)tokens, (void)n, (void)next, (void)targets, (void)top_n, (void)out;
    return false;
  }
  // Speculative decoding (several sequences). A model with a drafter proposes up
  // to draft_depth(n) tokens per sequence in a step of n sequences (0: plain one-row segments, the
  // drafter still follows them); draft_depth(1) > 0 means a drafter is loaded, and the engine then
  // runs every decode step through verify + accept.
  //   first_drafts: `depth` drafts after a prompt (0..), given its first output token; `sequences` = the decoding
  // sequences of the step it joins (a model with two drafters picks one by it).
  //   verify: one segment per sequence (the fed token + its drafts), every segment the same rows;
  //     picks / logits then read the rows segment after segment.
  //   accept: per segment in verify's order, keep its first n rows; next[r] (r < n) is the token
  //     that follows row r (the accepted drafts, then the correction or bonus token); returns each
  //     segment's next `depth` drafts (fewer near the capacity; none for depth 0).
  // A model whose drafter drafts a tree (draft_parents) verifies the tree in its segment: SpecSeg::parent[r] = row r's
  // parent row (parent[0] = -1, parent[r] < r; null = a chain). Row r's picks then follow the tokens of its ancestors
  // only; accept takes the walked rows (SpecAccept::path[i] = the row of next[i]: 0, then each one a child of the one
  // before; null = rows 0 .. n - 1).
  struct SpecSeg {
    int slot = 0;
    const int32_t *tokens = nullptr;
    int rows = 1;
    const int *parent = nullptr;
  };
  struct SpecAccept {
    int n = 1;
    const int32_t *next = nullptr;
    int depth = 0;
    const int *path = nullptr;
  };
  // The parent rows of the slot's last drafts (first_drafts / accept) when they are a tree: [0] = -1 (the fed
  // token's row), [r] = draft r's parent row (drafts are rows 1..; any prefix is a tree); empty = a chain.
  virtual std::vector<int> draft_parents(int slot) const { (void)slot; return {}; }
  // Whether the next verify, of these sequences with these rows each, may carry draft trees. A
  // model drafts trees only where it verifies them, but a sequence it did not know of at the draft can join the step
  // (one that just finished its prefill); when the step then cannot take trees the model answers false and the engine
  // verifies each tree's first path as a chain (exact: a tree's leading rows whose parents are the rows before them
  // are a chain). qwen35: a tree step attends densely, so not once a sequence of the step passes the page budget.
  virtual bool verifies_trees(std::span<const int> slots, std::span<const int> rows) const { (void)slots, (void)rows; return true; }
  virtual int draft_depth(int sequences) const { (void)sequences; return 0; }
  // A lone drafting sequence drafts wide_depth() (> draft_depth(1): a
  // 16-row tree and verify) after a verify that accepted wide_from() or more of its drafts, else draft_depth(1)'s pick;
  // 0 / 0 = never. The engine reserves KV for the wider step too.
  virtual int wide_depth() const { return 0; }
  virtual int wide_from() const { return 0; }
  // A speculative step's relative cost at `sequences` x `rows` rows each (any unit; only
  // ratios at one sequence count are used, to scale the engine's measured time of another depth). 0 = unknown: the
  // adaptive depth then only chooses among the shapes it has measured.
  virtual double spec_cost(int sequences, int rows) const { (void)sequences; (void)rows; return 0; }
  // True when the last verify recorded a step graph (a first use of its shape / bound); the adaptive depth then
  // leaves that step out of its measured times.
  virtual bool step_recorded() const { return false; }
  // Unequal segments. true = verify takes segments of B rows (the first ones) and then one-row
  // segments in one step (rows <= max_rows()), and accept() takes a depth per segment; the engine then gives each
  // request its own depth (BatchConfig::unequal_segments). spec_cost_rows: the relative cost of a step of `sequences`
  // segments and `rows` rows in all, `drafting` = some segment drafts after it (0 = unknown).
  virtual bool unequal_segments() const { return false; }
  // The deepest drafts pick_layout gives a request in a step of `sequences` (default: the
  // drafter's one-sequence depth). A model whose drafter at that count costs more per extra draft than the engine's
  // measured accept time shows (qwen35's MTP chain: one chained pass per further draft) caps it.
  virtual int layout_depth(int sequences) const { (void)sequences; return draft_depth(1); }
  virtual double spec_cost_rows(int sequences, int rows, bool drafting) const { (void)sequences, (void)rows, (void)drafting; return 0; }
  // Two classes of segments. unequal_classes: verify also takes the first segments B rows each and
  // then B2 rows each, 1 <= B2 <= B (not only one row). deep_depth: in a step of `sequences` whose one depth d leaves rows
  // free (sequences x (1 + d) < max_rows()), some requests may draft up to this many (> d; 0 = one depth for all).
  // draft_pass_ms: the model's cost of the accept's drafting pass at that count when the deepest request drafts `depth`
  // (the engine's measured accept time per (sequences, depth) replaces it; 0 = unknown: no second class).
  virtual bool unequal_classes() const { return false; }
  virtual int deep_depth(int sequences) const { (void)sequences; return 0; }
  virtual double draft_pass_ms(int sequences, int depth) const { (void)sequences, (void)depth; return 0; }
  // KV positions one decode step of a sequence at draft depth d writes past its position (the verify rows and the drafter
  // passes after it, up to the next drafts; d 0: the one row). Default: the MTP drafter's.
  virtual int64_t kv_ahead(int depth) const { return depth > 0 ? 2 + 2 * static_cast<int64_t>(depth) : 1; }
  virtual std::vector<int32_t> first_drafts(int slot, int32_t first, int depth, int sequences) { (void)slot, (void)first, (void)depth, (void)sequences; return {}; }
  // A sampled request's drafter noise for its next drafts (Sampler::draft_noise; inv_t 0 = argmax drafts),
  // set before first_drafts / accept on every drafting request. A model without coupled drafting ignores it.
  virtual void draft_sampling(int slot, uint64_t key, uint64_t pos, float inv_t) { (void)slot, (void)key, (void)pos, (void)inv_t; }
  virtual void verify(const std::vector<SpecSeg> &segs) { (void)segs; }
  virtual std::vector<std::vector<int32_t>> accept(const std::vector<SpecAccept> &a) { return std::vector<std::vector<int32_t>>(a.size()); }
  // A prefill call in parts: prefill_begin as prefill (n <= prefill_chunk), then
  // prefill_advance(layers) per part until it returns true; prefill_pick / prefill_logits / the
  // drafts are then as after prefill(). Between parts the engine may run decode steps of OTHER slots
  // (only those). prefill_abort drops a pending call (the slot needs begin() before reuse).
  // prefill_layers() = the layers one call spans (0: no split calls; the engine then calls prefill).
  virtual int prefill_layers() const { return 0; }
  // true = while others decode, prompt calls stay whole prefill_chunk calls (not the stall
  // budget's) and every call runs in parts of whole layers sized to prefill_part_ms, long prompts too. For a model whose
  // call has a large fixed cost (the dense 27B), small budget calls waste it.
  virtual bool prefill_long_parts() const { return false; }
  virtual void prefill_begin(int slot, const int32_t *tokens, int n, int32_t next) { (void)slot, (void)tokens, (void)n, (void)next; }
  virtual bool prefill_advance(int layers) { (void)layers; return true; }
  virtual void prefill_abort() {}
  // One prefill call for several requests (EngineConfig::prefill_merge). A model whose call
  // can carry several sequences answers prefill_merge_max() > 1; prefill_mergeable(slot, tokens, n) = this slot's
  // call of n rows may share a call (it must be the call the slot would run alone: the model keeps a sequence's rows
  // bit for bit); the members' rows, each padded to a multiple of prefill_merge_pad(), stay within
  // prefill_merge_rows(). prefill_begin_many starts the call (prefill_advance as for one; prefill_abort breaks every
  // member's slot). After the call prefill_result(slot) selects the member whose row prefill_pick / prefill_logits /
  // prefill_logprobs read: every member's are read before the first first_drafts (which reuses the head).
  struct PrefillSeg {
    int slot = 0;
    const int32_t *tokens = nullptr;
    int n = 0;
    int32_t next = -1;
  };
  virtual int prefill_merge_max() const { return 1; }
  virtual int prefill_merge_rows() const { return 0; }
  virtual int prefill_merge_pad() const { return 1; }
  virtual bool prefill_mergeable(int slot, const int32_t *tokens, int n) const { (void)slot, (void)tokens, (void)n; return false; }
  virtual void prefill_begin_many(const std::vector<PrefillSeg> &segs) { (void)segs; }
  virtual void prefill_result(int slot) { (void)slot; }
  virtual int32_t prefill_pick() = 0;
  virtual void prefill_logits(std::vector<float> &row) = 0;
  // One decode step: row i = tokens[i] on slot slots[i] (each slot at most once).
  virtual void step(const std::vector<int> &slots, const int32_t *tokens) = 0;
  virtual void picks(int rows, int32_t *out) = 0;                 // greedy picks of the last step
  virtual void logits(int rows, std::vector<float> &out) = 0;     // [rows][vocab]
  // The rows' sampling candidates instead of logits(): ids / vals [rows][n] (n returned),
  // each row's k best (value descending, lower id first, values rounded as logits()) for every k <= T. cand_tiles() = the
  // largest T the model serves (0 = none: logits() only). The engine uses it when every host request of the step reads
  // only its top_k (Sampler::cands_ok, no mask, no logprobs).
  virtual int cand_tiles() const { return 0; }
  virtual int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) {
    (void)rows, (void)T, (void)ids, (void)vals;
    return 0;
  }
  // Shared KV pages. kv_pool_tokens() > 0: the slots share a pool of that many tokens of KV, and a
  // sequence holds pages only for the positions it reserved: kv_reserve(slot, tokens) gives positions [0, tokens) of
  // the slot's sequence pages (all or nothing; false = the pool cannot now, nothing changed); kv_release(slot) returns
  // them (begin() does too). 0 (the default): every slot owns capacity() tokens and kv_reserve always succeeds.
  virtual int64_t kv_pool_tokens() const { return 0; }
  virtual int64_t kv_free_tokens() const { return 0; }
  virtual bool kv_reserve(int slot, int64_t tokens) { (void)slot, (void)tokens; return true; }
  virtual void kv_release(int slot) { (void)slot; }
  // The prefix cache. snap_capacity() > 0: the model keeps snapshots of a slot's sequence state (what
  // is not in its KV pages: the recurrent states, a drafter's rings) and a request may continue another's sequence:
  //   snap_save(slot): the slot's state at its position, right after a prefill call, into the store: an id, or -1 when
  //     the store is full (the engine frees one and asks again);
  //   snap_restore(slot, id): the slot back at the snapshot's position. Its KV pages below that position must be the ones
  //     the snapshot was taken over (the same slot, no write below it since);
  //   snap_fork(dst, src, id): slot dst gets a copy of slot src's KV pages below the snapshot's position (dst reserved
  //     them) and the snapshot's state (src may be decoding: it writes only at or past its own position);
  //   snap_free(id);
  //   snap_live(slot) / resume(slot): the slot's own live state (its last sequence at its position) can take a prefill
  //     call next / resume it there instead of begin();
  //   kv_trim(slot, tokens): keep the pages of positions [0, tokens) only; kv_held_tokens: positions the slot holds pages for;
  //   prefix_strict(): a state at position p continues only a sequence whose token p is the one the slot saw there (a
  //     drafter wrote its KV row p - 1 from it); snap_align(): see below.
  virtual int snap_capacity() const { return 0; }
  virtual int snap_save(int slot) { (void)slot; return -1; }
  virtual void snap_restore(int slot, int id) { (void)slot, (void)id; }
  virtual void snap_fork(int dst, int src, int id) { (void)dst, (void)src, (void)id; }
  virtual void snap_free(int id) { (void)id; }
  virtual bool snap_live(int slot) const { (void)slot; return false; }
  virtual void resume(int slot) { (void)slot; }
  virtual void kv_trim(int slot, int64_t tokens) { (void)slot, (void)tokens; }
  virtual int64_t kv_held_tokens(int slot) const { (void)slot; return 0; }
  virtual int kv_page_tokens() const { return 64; }
  // Work the model does while the engine has nothing to run (qwen38: KV lending's refills), at most `budget`
  // units per call (a few ms); returns the units done (0: none left). The engine calls it until 0 or a request arrives.
  virtual int idle_work(int budget) { (void)budget; return 0; }
  // KV lending (qwen38). kv_blocks_lend(slot): the slot holds KV pages above the lend keep (they keep lent
  // experts out of VRAM); kv_reserve_lends(slot, tokens): that reservation would give lent experts back.
  virtual bool kv_blocks_lend(int slot) const { (void)slot; return false; }
  virtual bool kv_reserve_lends(int slot, int64_t tokens) const { (void)slot, (void)tokens; return false; }
  // snapshots only at multiples of this, and prefill calls that start and end at its multiples with at least this many
  // rows compute every row as a cold prefill does (qwen35: 64): the engine sizes a
  // continued prompt's calls so (pf_call_rows, pc_cut)
  virtual int snap_align() const { return 1; }
  // YaRN per request. Tier t holds positions < rope_tier_limit(t), ascending (the last = the
  // capacity); a fresh sequence runs in the tier set_rope_tier names (after begin, before its first prefill). One tier =
  // one factor for every request.
  virtual int rope_tiers() const { return 1; }
  virtual int64_t rope_tier_limit(int t) const { (void)t; return capacity(); }
  virtual void set_rope_tier(int slot, int t) { (void)slot, (void)t; }
  virtual bool prefix_strict() const { return draft_depth(1) > 0; }
  // Snapshots from inside a prefill call. snap_inside(): the model can; snap_arm(slot,
  // pos) before the slot's next prefill call (pos a multiple of snap_align strictly inside it): the call also captures the
  // state at pos; snap_take() right after the call stores it (an id; -1 = the store is full: free one and ask again; -2 =
  // nothing captured). The engine then keeps a request's prefix_mark cut inside one call instead of ending a call there.
  virtual bool snap_inside() const { return false; }
  virtual void snap_arm(int slot, int64_t pos) { (void)slot, (void)pos; }
  virtual int snap_take() { return -2; }
};

struct BatchConfig {
  ModelInfo model;              // name, max_model_len (0 = the model's capacity), defaults
  std::vector<int32_t> eos_ids; // generation stops at these (unless ignore_eos)
  AdmissionConfig admission;
  double log_interval_s = 10.0; // stderr status line while busy; 0 = off
  // While any request decodes, a prefill call is sized so that it takes about this long
  // (ms), from a cost model (fixed + per row) fitted to the engine's own recent prefill calls.
  // 0 = off: full prefill_chunk calls always.
  double prefill_stall_ms = 0;
  // While any request decodes, a prompt prefilled in one call (it fits the call: a short
  // prompt) whose call is predicted longer than this (ms) runs in parts of whole layers with a decode
  // step between them (bit-identical to one call; its first token comes those steps later). Longer
  // prompts keep whole stall-budget calls. 0 = off. Calls with prompt logprobs never split.
  double prefill_part_ms = 0;
  // The waiting prompts one prefill call may carry (StepModel::prefill_begin_many; 0 / 1 = one
  // request per call). A member's rows are those of its own call, so replies do not change.
  int prefill_merge = 0;
  int sample_threads = 4;       // host threads sampling the rows of a step (1 = serial)
  // Prompt ids >= the vocabulary name image rows (the model's image input); the host sampler's penalties see
  // this token there (the input id of an image position in the model's format; -1: such ids are left out of the
  // penalties).
  int32_t media_token = -1;
  // The same per id (videos name the video token); empty = media_token for every id >= the vocabulary.
  std::function<int32_t(int32_t)> media_token_of;
  // The prefix cache, when the model has one (StepModel::snap_capacity() > 0). A finished request
  // leaves its sequence on its slot (parked: its KV pages, its live state and the snapshots taken after its prefill
  // calls); a new request continues the longest prefix of its prompt some slot holds a usable state for (in place on a
  // parked slot, or a copy of the KV into another slot), and prefills only the rest. false = every request from position 0.
  bool prefix_cache = true;
  int prefix_min_tokens = 256;   // a shorter common prefix is not reused (a slot's history is not given up for it)
  // A request's SamplingParams::prefix_mark (the server: where a chat's generation prompt starts) cuts a prefill call near
  // there (pc_cut: a snapshot the conversation's next turn continues from) when the mark is at least this far into the
  // prompt. The cut costs an extra call; below the mark the prompt is cheap to redo.
  int prefix_mark_min = 1024;
  // With a model that snapshots inside a prefill call (StepModel:snap_inside), the cut's
  // snapshot is taken inside the call that spans it instead of ending a call there (the same state bits, no extra call).
  // false = the cut calls (an A/B switch for probes; no server setting).
  bool prefix_inside = true;
  // In-flight prefix sharing: a waiting request whose prompt starts like the prompt a running request is still
  // prefilling waits for that prefill to pass their common part (its snapshot is taken there) and forks from it,
  // instead of prefilling the same tokens again in another slot (BatchEngine::pc_leader). Only for a common part of
  // prefix_min_tokens or more that the running request has not prefilled yet. false = every request placed at once (an
  // A/B switch for tests and probes; no server setting).
  bool prefix_share = true;
  // A parked sequence's live state after its decode steps is a valid continuation, but its output rows were computed by
  // decode steps where the run without the cache prefills them: other last bits, so a reply can differ from the cold
  // run's. false = a request continues from the snapshot at its previous prompt's mark instead and prefills
  // the reply again (results identical to the run without the cache); true = no reply re-prefill.
  bool prefix_live = false;
  // KV lending vs the prefix cache (qwen38, StepModel::kv_blocks_lend). lend_park_s: a parked sequence holding pages above
  // the lend keep is dropped after this many seconds parked (its experts are lent again while idle; 0 = never).
  // lend_pressure: when a reservation would give lent experts back, first drop parked sequences holding pages above the
  // keep, least recently used first: "kv" = never (KV wins, the experts go), "lru" = any such parked sequence, "idle" =
  // those parked at least lend_grace_s.
  double lend_park_s = 300;
  std::string lend_pressure = "kv";
  double lend_grace_s = 60;
  // The draft depth of a speculative step chosen from the running requests' acceptance
  // (per request: an average of each draft position's acceptance given the ones before it, over its recent steps) and the
  // engine's own step times per (sequences, rows per sequence): the depth in [0, StepModel::draft_depth(n)] with the most
  // tokens per ms, held unless another is better by depth_margin; every depth_explore-th step drafts the full depth (the
  // positions past a lower depth keep being measured), at most one change per depth_hold steps at a sequence count.
  // false = always draft_depth(n).
  bool adaptive_depth = true;
  double depth_margin = 0.10;
  int depth_hold = 32;  // spec steps at one sequence count between two depth changes
  int depth_explore = 16;
  // With a model that takes unequal segments (StepModel::unequal_segments) and adaptive_depth,
  // a depth per request: the k requests with the most expected tokens draft d tokens each, the rest none, the (k, d) with
  // the most expected tokens per ms (also where one depth for all is 0). false = one depth for all.
  bool unequal_segments = true;
  // Lookup drafting: with a drafter loaded, a step's drafts may come from the
  // request's own tokens instead: the latest earlier occurrence of its last `lookup` tokens (prompt + output) proposes the
  // tokens that followed it (up to the step's depth). Used for a step when, for every decoding request, the lookup's recent
  // yield (each proposal scored against the tokens that came, also on steps it did not draft) beats the drafter's; the
  // model's drafter pass of that step is skipped (its context deferred). Lossless: drafts only decide how many tokens a
  // verify keeps. 0 = off; else the match length (2..8).
  int lookup = 0;
};

// A scratch request of the warm-up: its output is dropped.
struct WarmRequest {
  std::vector<int32_t> prompt;
  SamplingParams params;
};

class BatchEngine final : public Engine {
 public:
  static constexpr int kIdleBudget = 64;  // StepModel::idle_work units per call (qwen38: 64 lent experts)
  BatchEngine(StepModel &m, BatchConfig cfg);
  ~BatchEngine() override;

  void start(EventSink *sink) override;
  void stop() override;
  void submit(uint64_t id, std::vector<int32_t> prompt, const SamplingParams &params) override;
  void cancel(uint64_t id) override;
  Stats stats() const override;
  const ModelInfo &model() const override { return cfg_.model; }
  // Warm-up before serving: the first use of a shape pays one-time costs
  // (the first prefill call, step / drafter graph recording). Runs the waves through the step
  // loop one after another (a wave's requests submitted together, the next wave once all finished),
  // output dropped, then clears what they leave behind: the stats and the prefill cost model's
  // samples. Call before start(), not while serving. Returns the wall time in ms; throws if a
  // request ended in an error.
  double warm_up(const std::vector<std::vector<WarmRequest>> &waves);
  // Re-read the model's KV pool into stats() (kv_pool_tokens, kv_used_tokens) after the model re-sized it outside a step
  // (qwen35's kv_fit after the warm-up); /v1/models reports it. Call before start(), not while serving.
  void kv_refresh() { kv_stats(); }
  // Rows of the next prefill call while requests decode (the stall budget through the cost model).
  int budget_rows() const;
  std::pair<double, double> pf_fit() const;  // prefill call ms = a + b rows, from the recent calls

 private:
  // A prompt token's logprob target: an image id (>= the vocabulary) scores the image token (the input id of an image
  // position in the model's format); -1 (no target) without BatchConfig::media_token
  int32_t media_lp(int32_t t) const { return t < m_.vocab() ? t : cfg_.media_token_of ? cfg_.media_token_of(t) : cfg_.media_token; }
  using clock = std::chrono::steady_clock;
  struct Submit {
    uint64_t id = 0;
    std::vector<int32_t> prompt;
    SamplingParams params;
    clock::time_point t_submit;
    // in-flight prefix sharing: it waits for request hold_id's prefill to reach hold_len (pc_leader)
    bool held = false;
    uint64_t hold_id = 0;
    int64_t hold_len = -1;
  };
  struct Req {
    uint64_t id = 0;
    std::vector<int32_t> prompt;
    SamplingParams params;
    std::unique_ptr<Sampler> sampler;  // host path only
    std::vector<Sampler::Row> prep;  // its verify rows, prepared in parallel (spec_step)
    std::vector<Sampler::LpStats> prep_lp;  // their logprob statistics (host rows with logprobs)
    std::unique_ptr<TokenConstraint> constraint;
    int slot = -1;
    size_t prefilled = 0;              // prompt tokens on the model
    int64_t pos = 0;                   // tokens on the slot
    int32_t next = -1;                 // the last emitted token (fed by the next decode step)
    int32_t generated = 0;
    bool host = false;                 // needs logits on the host
    std::vector<TokenLogprob> prompt_lps;  // prompt logprobs, sent with the first event
    std::vector<int32_t> drafts;           // speculative: the drafts for the next step
    std::vector<int> tree;  // their parent rows when they are a tree (StepModel::draft_parents)
    clock::time_point t_submit, t_admit;
    // (shared KV pool): the emitted tokens, the submitted prompt's length; a preempted request (its pages taken
    // for another) waits in resume_ and recomputes its KV: prompt = the submitted prompt + every emitted token
    std::vector<int32_t> out;
    size_t prompt_len = 0, prompt_counted = 0;  // prompt_counted: prompt tokens already in Stats::prompt_tokens
    bool resumed = false;  // re-prefilling after a preemption (no prompt logprobs, no TTFT sample)
    int64_t snap_armed = -1;  // the cut position its current prefill call captures a snapshot at (snap_arm)
    int64_t share_cut = -1;   // a snapshot is taken here: waiting requests continue from it (pc_leader; set per loop)
    bool preempt = false;  // marked in this loop's KV check: moves to resume_ after the step
    int tier = 0;  // its YaRN tier (StepModel::rope_tiers); past the tier's positions it resumes in the next
    bool aborted = false;  // its split prefill call was dropped (the slot's live state is undefined)
    bool pc_dropped = false;  // finished, its pages taken in this loop's KV check: nothing to park
    // draft position j's acceptance given j - 1 accepted (-1: unseen); 15 positions (kDepthMax)
    float acc_p[15] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};
    int acc_n[15] = {};  // its observations
    // lookup drafting (BatchConfig::lookup): the request's tokens (prompt, then each emitted one), the index of its
    // `lookup`-token windows (hash -> the position after the latest occurrence whose next token is known), the proposals
    // not yet scored (position, tokens), each drafter's recent yield (tokens per step) and whether this step's drafts are
    // the lookup's
    std::vector<int32_t> lk_hist;
    std::unordered_map<uint64_t, int32_t> lk_idx;
    size_t lk_indexed = 0;  // windows ending before this position are in lk_idx
    std::deque<std::pair<size_t, std::vector<int32_t>>> lk_pend;
    // Initial estimates in tokens per step. The optional lookup trial can sample lookup before it outbids the drafter.
    float lk_yield = 1.f, df_yield = 4.f;
    bool lk_used = false;
    // XEN_STEPSTAT: this request's speculative accounting (steps, drafts offered, drafts accepted, steps the lookup
    // drafted instead of the drafter), printed once when it finishes -- one line per request, so a bench that sends a
    // kind per request gets the numbers per kind.
    uint64_t st_steps = 0, st_draft = 0, st_acc = 0, st_lk = 0;
    bool st_done = false;
    bool lay_draft = false;  // drafting in pick_layout's last layout (kept unless another leads by kLayStick)
    bool lay_deep = false;  // in pick_deep's deeper class (kept likewise)
  };
  // A row to sample on the host: in (logits row), out (token, its logprob).
  struct Pick {
    Req *r = nullptr;
    const float *row = nullptr;
    const int32_t *cid = nullptr;  // the row's candidates instead of row (cn > 0)
    const float *cval = nullptr;
    int cn = 0;
    int32_t tok = -1;
    TokenLogprob lp;
    bool has_lp = false;
  };

  void loop();
  bool is_stop(const Req &r, int32_t tok) const;
  // Emits token `tok` for r (appends to ev); returns the finish reason (none = continues).
  Finish emit(Req &r, int32_t tok, TokenLogprob *lp, Event &ev);
  // Logprobs on the model's side (device_logprobs) for the rows of the last step / verify:
  // row i belongs to reqs[i] (null: no logprobs wanted) and emitted toks[i]. Fills lps[i] (one model call).
  void device_lps(const std::vector<const Req *> &reqs, const int32_t *toks, std::vector<TokenLogprob> &lps);
  // the T of StepModel:logits_cands for a step's requests (the largest top_k of its host requests),
  // or 0 when one of them reads more than its top_k (a mask, a modifier, top_k off or above cand_tiles(), logprobs
  // without device_logprobs() or above kDevLpTop best).
  int cands_for(const std::vector<const Req *> &reqs) const;
  static constexpr int kDevLpTop = 20;  // the most top_logprobs the model's records carry (kernels::kLpMaxTop)
  void retire(size_t i, Finish f);
  // pages for the next decode step of the requests in cand (indexes in active_), oldest admitted first; when
  // the pool is out, the youngest other request is preempted (marked; pages released now). A request no preemption can
  // serve ends with length (appended to done). Returns the requests that decode this step.
  std::vector<size_t> kv_ensure(const std::vector<size_t> &cand, std::vector<std::pair<size_t, Finish>> &done);
  void preempt_now(size_t i);  // mark + release the pages (+ abort its split prefill)
  void kv_stats();
  void sample_rows(std::vector<Pick> &picks);   // host picks of a step, rows in parallel
  void parallel(size_t n, const std::function<void(size_t)> &fn);
  void record_prefill(int rows, double ms);
  // One speculative step over the decoding requests dec (indexes in active_,: all of them
  // in one verify); appends their events, and (i, reason) to done.
  void spec_step(const std::vector<size_t> &dec, std::vector<std::pair<size_t, Finish>> &done, uint64_t &gen_add);
  // Adaptive draft depth (BatchConfig::adaptive_depth).
  // kDepthMax 15: a whole 16-row segment of one sequence (XEN_SEG_ROWS 16, XEN_DRAFT_DEPTHS); the
  // model's depth table (7 by default) caps every pick.
  static constexpr int kDepthMax = 15, kSeqMax = 64;  // kSeqMax: 64 sequences (qwen38 on four cards)
  static constexpr float kLkAlpha = 0.3f;  // lookup drafting: the weight of a step in each drafter's recent yield
  static_assert(sizeof(Req::acc_p) / sizeof(float) == kDepthMax && sizeof(Req::acc_n) / sizeof(int) == kDepthMax, "acc_p / acc_n per draft position");
  int pick_depth(const std::vector<size_t> &going, int depth_max);
  void depth_observe(Req &r, int drafted, int accepted);
  // Unequal segments on (BatchConfig::unequal_segments, the model's, adaptive depth).
  bool unequal_on() const { return cfg_.unequal_segments && cfg_.adaptive_depth && m_.unequal_segments(); }
  // the next step's depth per going request (their order); expected tokens of a step of r at depth d
  std::vector<int> pick_layout(const std::vector<size_t> &going);
  double expected_tokens(const Req &r, int d) const;
  static constexpr int kHetRows = 64;  // a layout's rows at most (qwen38's 64-row step)
  // A request drafting in the layout keeps its place unless another's expected tokens lead by this factor: a request
  // that starts drafting reads every context row deferred while it ran plain.
  static constexpr double kLayStick = 1.15;
  // A layout change needs this much more expected tokens per ms (below depth_margin's 10 %; depth_hold still
  // spaces the changes).
  static constexpr double kLayMargin = 0.03;
  // EMAs of a step's verify ms (the step without the accept) per (sequences, rows) and of its accept (commit + the
  // drafter's pass) per (sequences, drafts after it); counts < 0: warm-up steps to skip (constructor)
  double ver_ms_[kSeqMax + 1][kHetRows + 1] = {}, pass_ms_[kSeqMax + 1][2] = {};
  int ver_n_[kSeqMax + 1][kHetRows + 1], pass_n_[kSeqMax + 1][2];
  int cur_k_[kSeqMax + 1], cur_d_[kSeqMax + 1] = {};  // the held layout (cur_k_ -1: none yet, the constructor)
  uint64_t lay_since_[kSeqMax + 1] = {};
  // Two classes where the one depth for all is > 0 and leaves rows free (StepModel::deep_depth):
  // k requests draft dhi, the rest the one depth d; depth[i] of going[i] is raised in place. The (k, dhi) with the most
  // expected tokens per ms: ms = the verify of the layout's rows (ver_est) + the accept at its deepest draft (deep_ms_,
  // else the model's draft_pass_ms difference to a measured depth). Held as pick_layout's (kLayMargin, depth_hold).
  void pick_deep(const std::vector<size_t> &going, int d, std::vector<int> &depth);
  double ver_est(int N, int rows) const;  // the verify of N segments and `rows` rows: measured, else a measured neighbour scaled by the model's cost, else 0
  double deep_ms_[kSeqMax + 1][kDepthMax + 1] = {};  // EMA of the accept's ms per (sequences, deepest draft after it)
  int deep_n_[kSeqMax + 1][kDepthMax + 1] = {};
  int cur_k2_[kSeqMax + 1] = {}, cur_d2_[kSeqMax + 1] = {};  // the held deeper class (0 requests: none)
  uint64_t deep_since_[kSeqMax + 1] = {};
  bool explore_now_ = false;  // this step's layout is pick_layout's drafting explore (its accept not timed)
  bool lay_steady_ = true;    // this step's drafting requests drafted after the last step too (its pass timed)
  // lookup drafting: index r.lk_hist's new windows, score the pending proposals, and propose up to k tokens after it
  void lk_index(Req &r);
  void lk_score(Req &r);
  std::vector<int32_t> lk_propose(Req &r, int k);
  double spec_ms_[kSeqMax + 1][kDepthMax + 2] = {};  // EMA of a spec step's ms per (sequences, rows per sequence)
  static constexpr int kWarmSteps = 2;
  int step_n_[kSeqMax + 1][kDepthMax + 2];  // timed steps per shape; < 0: warm-up steps still to skip (constructor)
  int cur_depth_[kSeqMax + 1];  // -1: none yet (the constructor)
  uint64_t depth_since_[kSeqMax + 1] = {};  // spec_steps_ at the last change
  float glob_p_[kDepthMax] = {-1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f};  // every request's, the prior of a new one
  uint64_t spec_steps_ = 0;
  // The prefix cache. place: a free slot for a prompt (positions [0, need) of pages), continuing the longest
  // cached prefix when use_cache: returns the slot and sets start (prompt tokens already on it), or -1 (the pages are not
  // free even after evicting parked sequences: nothing changed).
  int place(const std::vector<int32_t> &prompt, int64_t need, bool use_cache, int64_t &start, int tier);
  // the smallest YaRN tier holding n tokens plus room to grow: the rest of max_tokens up to a quarter of the
  // tier, and at least a page and the step's writes ahead (rem: the tokens it may still generate)
  int tier_for(int64_t n, int64_t rem) const;
  struct Snap {
    int64_t pos = 0;
    int id = -1;
    bool mark = false;  // at a request's cut (pc_cut) or its prompt's end (kept over the plain call-end ones)
  };
  struct SlotCache {
    std::vector<int32_t> hist;  // parked: the tokens behind the slot's state, and the one after them (pc_park)
    int64_t live = -1;          // parked: the position its live state continues from (-1: none)
    std::vector<Snap> snaps;    // ascending positions; valid while the slot holds its pages
    uint64_t tick = 0;          // last use
    std::chrono::steady_clock::time_point parked{};  // when it was parked (lend_park_s / lend_grace_s)
    int tier = -1;  // the YaRN tier of its sequence (a prefix is reused in that tier only)
  };
  const Req *busy(int slot) const;  // the active request on the slot (null: free)
  bool is_free(int slot) const;
  int64_t pc_tip(int slot) const;   // the furthest position the slot's cache continues from (-1: nothing)
  void pc_park(Req &r, Finish f);   // a retired request's slot keeps its sequence
  void pc_evict(int slot);          // a free slot's cache and pages dropped
  void pc_drop(int slot, int64_t above);  // snapshots past a position dropped (the slot writes there next)
  void pc_free_snaps(int slot);     // every snapshot of the slot
  void pc_snapshot(int slot, int64_t pos, bool mark, bool inside = false);  // inside: the call's capture (snap_take)
  bool pc_inside(const Req &r) const;  // r's cut goes inside a call (StepModel:snap_inside)
  int pf_call_rows(const Req &r) const;  // the next prefill call's rows when nothing decodes
  int64_t pc_cut(const Req &r) const;  // where r's prefill is cut for the next turn's snapshot (-1: nowhere)
  struct PcCand {  // a cached state a prompt can continue from
    int src = -1;   // its slot (-1: none)
    int64_t R = 0;  // the position
    int snap = -1;  // index in the source's snapshots; -1 = its live state
    int rank = -1;  // 2 a parked slot's live state, 1 its snapshot, 0 a running slot's snapshot
  };
  PcCand pc_find(const std::vector<int32_t> &T, int tier) const;
  Req *pc_leader(const std::vector<int32_t> &T, int tier, int64_t &cut) const;  // the running request T waits for (null: none)
  bool pc_free_one(int current, bool mark);    // one snapshot out of the store (the least used slot's least useful); false: none
  size_t pc_victim(const SlotCache &c) const;
  bool pc_evict_lru(int keep_a, int keep_b);  // the least recently used parked slot holding pages; false: none
  // KV lending: parked sequences above the lend keep dropped (BatchConfig::lend_park_s / lend_pressure)
  void lend_park_timeout();
  void lend_pressure(int slot, int64_t tokens, int keep_a, int keep_b);
  double lend_park_wait_s() const;  // seconds until the next park timeout (< 0: none pending)
  void pc_clear();                  // every cache (the warm-up's end)
  void pc_stats();

  StepModel &m_;
  BatchConfig cfg_;
  EventSink *sink_ = nullptr;
  std::thread thread_;

  std::mutex mu_;  // the inbox and stop_
  std::condition_variable cv_;
  std::vector<Submit> submits_;
  std::vector<uint64_t> cancels_;
  bool stop_ = false;

  mutable std::mutex stats_mu_;
  Stats stats_;

  // engine thread only
  std::deque<Submit> waiting_;
  std::vector<std::unique_ptr<Req>> active_;
  std::deque<std::unique_ptr<Req>> resume_;  // preempted requests, readmitted before waiting_
  bool kv_shared_ = false;                   // the model's slots share a KV pool
  std::vector<int> free_slots_;
  bool pc_on_ = false;                       // the prefix cache
  int tiers_ = 1;  // the model's YaRN tiers
  std::vector<SlotCache> pc_;                // per slot
  uint64_t pc_tick_ = 0;
  int pc_count_ = 0;                         // snapshots held
  std::vector<Event> events_;
  // Events handed to the sink before the iteration ends - a step's tokens before
  // its drafter pass, a prompt's first token before its first drafts and the other requests' step. Only events that do
  // not finish their request: a finish still goes out at the iteration's end, after retire and the stats. A prompt's
  // first token does not keep its request out of the step that follows it.
  void publish_early(std::vector<Event> &evs);
  std::vector<float> logits_;
  std::vector<int32_t> cand_ids_;  // the step's candidate rows (StepModel:logits_cands)
  std::vector<float> cand_vals_;
  std::deque<std::pair<int, double>> pf_samples_;  // recent prefill calls: (rows, ms)
  bool pf_skipped_ = false;  // the last call's sample was left out of the fit (record_prefill)
  struct PfPending {  // a prefill call split into parts
    bool active = false;
    uint64_t id = 0;   // its request
    int n = 0, layers = 0, parts = 0;
    size_t depth = 0;
    double ms = 0;     // the parts' time so far
    std::pair<double, double> fit;
    clock::time_point t0;
    // a call for several requests (StepModel::prefill_begin_many); n = its rows with the padding
    struct Member {
      uint64_t id = 0;
      int n = 0;
      size_t depth = 0;
      bool read = false;       // its first token was read (pf_group_read)
      std::vector<float> row{};  // a sampled request's logits
      int32_t tok = -1;
      bool has_lp = false;
      TokenLogprob lp{};
    };
    std::vector<Member> group;  // empty: the call is id's alone
    bool finished = false;      // the model's call ended; the members complete in the loop's prefill stage
  };
  PfPending pf_pend_;
  void pf_group_read();
  bool pf_group_leave(uint64_t id);
  // sampling workers
  struct Pool;
  std::unique_ptr<Pool> pool_;
  // status line
  clock::time_point log_at_;
  clock::time_point last_step_end_;  // the last decode step's end (XEN_PF_LOG)
  std::vector<double> step_ms_;
  uint64_t log_prefill_tokens_ = 0;
  double log_prefill_s_ = 0;
  uint64_t log_prefill_calls_ = 0;
};

}  // namespace xenithium::core
