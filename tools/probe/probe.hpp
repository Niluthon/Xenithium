// xenithium-probe internals: the subcommands (one.cpp per op family / model, dispatched by main.cpp) and the
// run-wide flags main.cpp's parser sets (defined in flags.cpp). Every probe file includes this instead of declaring
// its own externs.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// ---- flags (flags.cpp) ---------------------------------------------------------------------------------------------
// model35: --dflash DIR = the DFlash 2 drafter for --spec / --seq-depths (instead of the MTP layer),
// --dflash-bytes 1|2 its matrices (int8 per row / bf16), --dflash-check N = the host-reference gate on N prompt tokens.
extern std::string g_dflash;
// --head-w4 0|1 (ModelOptions:head_w4), --pf-a8 0|1 (prefill_w4_a8), --pf-a8-keep SPEC, for the qwen35
// model commands; --head-x-out F (head35x: the lm_head's inputs at generation time, XHI1)
extern int g_head_w4, g_pf_a8;
// --head-rescore T (ModelOptions:head_rescore: the two-stage head, T tiles per card; needs --head-w4 1),
// --draft-head -1|0|1 (ModelOptions:draft_head: the drafters' rows as the target / int8 / int4)
extern int g_head_rescore, g_draft_head;
extern std::string g_pf_a8_keep;  // --pf-a8-keep SPEC (ModelOptions:prefill_a8_keep)
extern std::string g_head_x_out;
extern int g_dflash_bytes;
extern int g_dflash_check;
extern bool g_draft_conf;  // --draft-conf FILE (audit C2): per-step draft confidences of --spec (DFlash)
extern std::string g_draft_conf_path;
extern int g_dflash_tree;  // --dflash-tree N
extern double g_dflash_tree_cut;  // --dflash-tree-cut X
extern int g_dflash_tree_seqs;  // --dflash-tree-seqs N
extern double g_dflash_tree_temp;  // --dflash-tree-temp T
extern int g_dflash_wide;  // --dflash-wide N
extern int g_dflash_tree_wide;  // --dflash-tree-wide N
extern double g_dflash_cut;  // --dflash-cut X: DFlashOptions:conf_cut (0 = every draft; -1 = the default)
extern bool g_kv_v4;  // --kv-v4 0|1: --kv-bytes 1 runs int8 K + int4 V (qwen38 / qsa
extern int g_mtp_defer;  // --mtp-defer N: ModelOptions:mtp_kv_defer (0 = every KV-only pass runs; -1 = the default)
extern bool g_pf_checks;  // --pf-checks 0|1: model35 --rows pf timing + continuation check
extern int g_pf_merge_at;  // --pf-merge-at P (model35)
extern std::string g_pf_merge;  // --pf-merge N,N.. (model35): slot i's prompt call of N rows, merged against alone
extern std::string g_pf_calls;  // --pf-calls R,R..: model35: prefill calls of R rows at --at (9-63-row calls, e.g. a 16-token verify)
extern bool g_lend_check;  // --lend-check 1: model35 with --kv-lend 1 --kv-pool N: the kept layouts' check
extern bool g_pf_ab_prof;  // --pf-ab-prof 0|1: model35 --pf-ab: one profiled prefill per case after the timing
extern bool g_snap_prepin;  // --snap-prepin 0|1: prefix35: every snapshot id pinned through the adapter's idle work first
extern std::string g_lp_ab;  // --lp-ab "name:key=v....": the --ppl section once per case, one load
extern int g_mtp_from;  // --mtp-from N: with --dflash, the MTP layer too, drafting for N+ sequences (0 = DFlash alone)
extern int g_tp;  // --tp 1|2|4: on one card (TP1, device 0: pick it with ZE_AFFINITY_MASK), two (TP2) or four (TP4: qwen38 only)
// The devices of --tp N: {0}, {0, 1} or {0, 1, 2, 3}; `four` = the tool takes TP4.
std::vector<size_t> tp_devices(const char *tool, bool four);
extern std::string g_dec_attn;  // --dec-attn exact|fast|int8|int8-fast (gqa35)
extern int g_attn_select;  // --attn-select N (model35)
extern bool g_sel_kv;  // --sel-kv 0|1 (select35)
extern bool g_sel_pf;  // --sel-pf 0|1 (select35)
extern bool g_sel_raw;  // --sel-raw 0|1 (select35)
extern std::string g_sel_dump;  // --sel-dump DIR (select35)
extern int g_sel_self;  // --sel-self K (select35)
extern int g_pf_select;  // --pf-select N (model35)
extern bool g_queue_prof;  // --queue-prof 0|1: model35 compute queues with event profiling (default 1)
extern int g_dflash_defer;  // --dflash-defer N: DFlashOptions:ctx_defer (0 = every context pass runs; -1 = the default)
extern bool g_dflash_prof;  // --dflash-prof 1: the drafter's stage wall times after --spec (waits per stage: diagnostic)
extern bool g_step_reset;  // --step-reset 1: model35 --step-ab prefills the shape's slots to --at again before every shape
// --pf-moe-a8 0|1: ModelOptions:prefill_moe_a8 for every model command.
extern bool g_pf_moe_a8;  // (user): W4A16 only by default
// --pf-dense-a8 0|1|2 (2 = per row): ModelOptions:prefill_dense_a8 (+ _row) for every model command.
extern int g_pf_dense_a8;  // 0 W8A16, 1 W8A8 per group, 2 W8A8 per row with the HC on W8A16
// --pf-wire fp32|bf16: qwen38 ModelOptions:prefill_wire_bf16 for every qwen38 model command.
extern bool g_q38_pf_wire_bf16;
extern int g_q38_moe_dn_ks;  // --moe-dn-ks N for every qwen38 model command
// --dec-moe-a8 0|1: ModelOptions:decode_moe_a8 for every model command (default -1 = by the
// checkpoint's act_bits).
extern int g_dec_moe_a8;  // (user): W4A16 only by default
extern int g_q38_mtp_head_w4;  // --mtp-head-w4 1, the draft vocabulary rows as int4 RTN (qwen38 mtp / serve)
extern int g_q38_mtp_vocab;  // --mtp-vocab K for the qwen38 commands (mtp, serve, prefix): the draft vocabulary
// --kv-pool N / --kv-scramble 0|1: qwen38 ModelOptions:kv_pool_tokens / kv_pool_scramble for
// every qwen38 model command (0 = the fixed per-slot layout).
extern int64_t g_kv_pool;
extern bool g_kv_scramble;
// --kv-lend 0|1 / --kv-lend-keep N: qwen38 ModelOptions:kv_lend / kv_lend_keep for every qwen38 model command.
extern bool g_kv_lend;
extern int64_t g_kv_lend_keep;
extern bool g_kv_lend_idle;  // --kv-lend-idle 1: probe model refills (Model:lend_idle) before each ppl chunk
// --moe-gu-ks N: ModelOptions:moe_gu_ksplit for probe model (0 = the kernel default).
extern int g_moe_gu_ks;
extern std::string g_w4_tensors;
extern int g_w4_mse;  // model --w4-mse 1
extern int g_gdn_w4;  // model --gdn-w4 0..3 (ModelOptions:gdn_w4): bit 1 qkv | z, bit 2 out_proj from the dir's int4 GPTQ tensors
extern std::string g_gdn_cap;  // model --gdn-cap FILE: the GDN trunk's calibration inputs (XGC1)
extern int g_gdn_cap_rows, g_gdn_cap_every;  // model --w4 PAT.., int8 plan matrices loaded as int4 RTN (ModelOptions:w4_tensors)
// --gdn-scan 0|1: ModelOptions:gdn_scan for every model command.
extern bool g_gdn_scan;
// --pf-cold-stage N: ModelOptions:prefill_cold_stage for every model command.
extern int g_pf_cold_stage;
extern std::string g_moe_cpu;  // --moe-cpu
extern int g_moe_cpu_pct, g_moe_cpu_threads, g_moe_cpu_min_cold;
// --pf-route-trace F: probe model writes every prefill call's MoE inputs and
// routing to F (XLA1, model.cpp) and exits; xenithium-lookahead scores routing predictors on it.
extern std::string g_pf_route_trace;
// --lp-time N: probe serve times only a P-token prefill plain vs with prompt
// logprobs on the many-row head, N alternating runs each, prints the medians and a hash of the records
// (bit-compare variants by the hash), then exits.
extern int g_lp_time;
extern int g_step_rows;  // --step-rows 4|8: rows a decode step may carry (probe serve)
extern std::string g_spec_cap;  // --spec-cap F (probe mtp): target + drafter pass-1 top logits per position
extern int g_two_deep;  // --two-deep 0|1: 2 requests at MTP-3 each (8 rows) instead of MTP-1 each
// --bucket-gate N: probe mtp runs only the bound-bucket gate (mtp.cpp, part 5)
// with N generated tokens across the first QSA bound bucket's end (65,536 tokens), then exits. --bucket-pre 0 skips
// Model:record_bucket_graphs (the control: the long runs must then record graphs).
extern int g_bucket_gate;
// --p1-kv-gate A,B: probe mtp section 3b, output tokens [A, B) as plain steps
extern int g_p1_kv_a, g_p1_kv_b;
extern bool g_bucket_pre;

// ---- subcommands ---------------------------------------------------------------------------------------------------
int cmd_moe_run(const std::string &model, const std::string &layers, int iters, int threads);  // moe.cpp
int cmd_gdn_run(const std::string &model, const std::string &layers, int iters, int threads);  // gdn.cpp
int cmd_w4load(const std::string &model, const std::string &layers, int iters, int threads);  // qwen35.cpp
int cmd_w4load_llama(const std::string &model, const std::string &layers, int iters, int threads, int tp);  // llama.cpp
int cmd_w4load_flint(const std::string &model, const std::string &layers, int iters, int threads, int tp);  // flint.cpp
int cmd_gqa_flint(int iters, int context);
int cmd_moe_flint(const std::string &model, int layer, int iters);  // flint.cpp
int cmd_w4pf_llama(const std::string &model, const std::string &layers, const std::string &rows, int iters, int threads, int tp);  // llama.cpp
int cmd_head35(const std::string &model, int iters, int threads);  // qwen35.cpp
int cmd_mlp35(const std::string &model, const std::string &layers, int rows, int iters, int threads);  // qwen35.cpp
int cmd_gdn35(const std::string &model, const std::string &layers, int iters, int threads);  // qwen35_gdn.cpp
int cmd_head35x(const std::string &model, const std::string &ppl, int generate, int threads);  // qwen35_headx.cpp
int cmd_model35(const std::string &model, const std::string &ppl, const std::string &rows, int generate, int64_t at, int64_t context, int kv_bytes,
                int threads, const std::string &lp_out, const std::string &lp_ref, int pf_chunk, bool pf_slices, bool pf_attn_slices, int pf_attn, bool pf_p_lo, bool pf_w4dq, bool pf_split,
                const std::string &pf_ab, int pf_reps, const std::vector<int> &pf_lens, int spec, int depth, int mtp_kv_bytes,
                int mtp_dense_bytes, int mtp_vocab, const std::string &mtp_dump, int seqs, int seq_tokens, const std::string &seq_depths, const std::string &step_ab,
                const std::string &step_shapes, int step_reps, int profile_card, const std::string &step_opts, long long kv_pool, bool kv_scramble);  // qwen35_model.cpp
int cmd_gqa35(const std::string &model, const std::string &layers, int iters, int threads, int kv_bytes, int64_t context);  // qwen35_attn.cpp
int cmd_qsa_run(const std::string &model, const std::string &layers, int iters, int threads, int kv_bytes, int64_t context, bool verify);  // qsa.cpp
int cmd_ple_run(const std::string &model, int iters, int threads);  // ple.cpp
int cmd_model_run(const std::string &model, const std::string &ppl, const std::string &rows, int generate, int64_t at, int64_t context, int kv_bytes,
                  int64_t hot, int threads, bool verify, const std::string &routes_out, const std::string &hot_rank, double reserve_gib,
                  const std::string &lp_out, const std::string &lp_ref, const std::string &bf16, bool moe_down_bf16, int moe_dn_ks, bool graphs, bool moe_tp,
                  const std::string &hidden_out, const std::string &gen_out, int pf_chunk, bool pf_time_only,
                  const std::string &pf_sweep);  // model.cpp
int cmd_mtp_run(const std::string &model, const std::string &mtp, const std::string &ppl, int prompt_len, int generate, int depth,
                const std::string &drafts_ref, int64_t context, bool graphs, const std::string &hot_rank, double reserve_gib, int threads,
                int mtp_kv_bytes, int pf_prompt_len, bool pf_only, int pf_chunk);  // mtp.cpp
int cmd_seqs_run(const std::string &model, const std::string &ppl, int nseq, int prompt_len, int generate, int64_t context, bool graphs,
                 const std::string &hot_rank, double reserve_gib, int threads, int kv_bytes, int64_t hot = -1, int pf_chunk = 0);  // seqs.cpp
int cmd_prefix35(const std::string &model, const std::string &ppl, int64_t prompt_len, int generate, int64_t context, int threads, int turns);
int cmd_prefix_run(const std::string &model, const std::string &ppl, const std::string &mtp, int64_t prompt_len, int generate, int turns,
                   const std::string &hot_rank, int threads);  // prefix.cpp
int cmd_select35(const std::string &model, const std::string &ppl, const std::string &lens, const std::string &docs, const std::string &budgets,
                 int generate, int reps, int threads, int pf_chunk);  // qwen35_select.cpp
int cmd_serve_run(const std::string &model, const std::string &ppl, int nseq, int prompt_len, int generate, int64_t context, int pf_chunk,
                  bool graphs, const std::string &hot_rank, double reserve_gib, int threads, int kv_bytes, double stall_ms,
                  const std::string &mtp);  // serve.cpp
int cmd_rank(int argc, char **argv);
// peers.cpp: the TP wire's peer path per ordered pair (a peer store vs a host-staged copy)
int cmd_peers(int mib, int iters);
// wiremin.cpp (scratch): the two-card pair_reduce mismatch, per call
int cmd_wiremin(int argc, char **argv);
// The DFlash host-reference gate's view of a target (qwen35_dflash.cpp: qwen35's; dflash38.cpp: qwen38's).
namespace xenithium::arch {
class DFlash;
}
struct DfCheckTarget {
  virtual ~DfCheckTarget() = default;
  virtual xenithium::arch::DFlash *dflash() = 0;
  virtual int cards() = 0;
  virtual int prefill_chunk() = 0;
  virtual int slot() = 0;
  virtual int mtp_vocab() = 0;
  virtual std::vector<int64_t> eos() = 0;
  virtual std::string embed_name() = 0;  // the target checkpoint's embedding tensor
  virtual void reset() = 0;
  virtual void prefill(const int32_t *ids, int n) = 0;
  virtual std::vector<int32_t> picks(int rows) = 0;  // the last prefill's / step's greedy picks of its first `rows` rows
  virtual std::vector<int32_t> draft_after_prefill(int32_t first, int depth) = 0;
  virtual void step(const int32_t *rows, int n) = 0;  // one sequence's verify
  virtual void accept(int keep) = 0;
  // after accept(keep): the drafter's context from the verify's kept rows, drafts from picks[keep - 1]
  virtual std::vector<int32_t> draft(const int32_t *picks, int rows, int keep, int depth) = 0;
  virtual void read_device(int c, const void *src, void *dst, size_t bytes) = 0;
};
void dflash_check(DfCheckTarget &m, const std::string &target_dir, const std::vector<int32_t> &prompt, int n, int dense_bytes,
                  const std::function<void(bool, const std::string &)> &check);
// dflash38.cpp: the DFlash 2 drafter on qwen38 (spec greedy == plain greedy, acceptance, timing)
int cmd_dflash38(const std::string &model, const std::string &dflash, const std::string &ppl, int prompt_len, int prompts, int generate, int depth, int64_t context,
                 bool graphs, const std::string &hot_rank, double reserve_gib, int threads, int shift);  // rank.cpp: a hot-expert ranking (XRC1) for a MoE checkpoint
