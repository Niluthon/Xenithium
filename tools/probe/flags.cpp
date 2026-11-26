// xenithium-probe's run-wide flags (probe.hpp), set by main.cpp's option parser.
#include <vector>
#include <string>
#include <stdexcept>
#include "probe.hpp"

// model35: --dflash DIR = the DFlash 2 drafter for --spec / --seq-depths (instead of the MTP layer),
// --dflash-bytes 1|2|4 its matrices (int8 per row / bf16 / int4 by RTN for q, o, gate | up, down), --dflash-check N = the host-reference gate on N prompt tokens.
std::string g_dflash;
// --head-w4 0|1 (ModelOptions:head_w4), --pf-a8 0|1 (prefill_w4_a8), --pf-a8-keep SPEC, for the qwen35
// model commands; --head-x-out F (head35x: the lm_head's inputs at generation time, XHI1)
int g_head_w4 = 0, g_pf_a8 = 0;
// --head-rescore T (ModelOptions:head_rescore: the two-stage head, T tiles per card; needs --head-w4 1),
// --draft-head -1|0|1 (ModelOptions:draft_head: the drafters' rows as the target / int8 / int4)
int g_head_rescore = 0, g_draft_head = -1;
std::string g_pf_a8_keep;  // --pf-a8-keep SPEC (ModelOptions:prefill_a8_keep)
std::string g_head_x_out;
int g_dflash_bytes = 1;
int g_dflash_check = 0;
bool g_draft_conf = false;  // --draft-conf FILE (audit C2): per-step draft confidences of --spec (DFlash)
std::string g_draft_conf_path;
int g_dflash_tree = 0;  // --dflash-tree N: DFlashOptions:tree
double g_dflash_tree_cut = 0;  // --dflash-tree-cut X: DFlashOptions:tree_cut
int g_dflash_tree_seqs = 1;  // --dflash-tree-seqs N: DFlashOptions:tree_seqs
double g_dflash_tree_temp = 0;  // --dflash-tree-temp T: DFlashOptions:tree_temp
int g_dflash_wide = 0;  // --dflash-wide N: ModelOptions:dflash_wide
int g_dflash_tree_wide = 15;  // --dflash-tree-wide N: ModelOptions:dflash_tree_wide
double g_dflash_cut = -1;  // --dflash-cut X: DFlashOptions:conf_cut (0 = every draft; -1 = the default)
bool g_kv_v4 = false;  // --kv-v4 0|1: qwen38 / qsa: --kv-bytes 1 runs int8 K + int4 V
bool g_kv_tokens = false;  // --kv-split heads|tokens: qwen38 TP4 ModelOptions::kv_split
int g_mtp_defer = -1;  // --mtp-defer N: ModelOptions:mtp_kv_defer (0 = every KV-only pass runs; -1 = the default)
bool g_pf_checks = true;  // --pf-checks 0|1: model35 --rows pf timing + continuation check
int g_pf_merge_at = -1;  // --pf-merge-at P: the slots' history for --pf-merge (default --at; 0 = new prompts at position 0)
std::string g_pf_merge;  // --pf-merge N,N..: model35: one prefill call for several slots against each slot's own call
std::string g_pf_calls;  // --pf-calls R,R..: model35: prefill calls of R rows at --at (9-63-row calls, e.g. a 16-token verify)
bool g_lend_check = false;  // --lend-check 1: model35 with --kv-lend 1 --kv-pool N: the kept layouts lent / given back / lent again give the same logits
bool g_pf_ab_prof = false;  // --pf-ab-prof 0|1: model35 --pf-ab: one profiled prefill per case after the timing
bool g_snap_prepin = false;  // --snap-prepin 0|1: prefix35: every snapshot id pinned through the adapter's idle work first
std::string g_lp_ab;  // --lp-ab "name:key=v....": the --ppl section once per case, one load
int g_mtp_from = 0;  // --mtp-from N: with --dflash, the MTP layer too, drafting for N+ sequences (0 = DFlash alone)
int g_tp = 2;  // --tp 1|2|4: model35 on one card (TP1, device 0: pick it with ZE_AFFINITY_MASK) or two (TP2)
std::vector<size_t> tp_devices(const char *tool, bool four) {
  if (g_tp == 1) return {0};
  if (g_tp == 2) return {0, 1};
  if (g_tp == 4 && four) return {0, 1, 2, 3};
  throw std::runtime_error(std::string(tool) + ": --tp 1 or 2" + (four ? " or 4" : ""));
}
std::string g_dec_attn = "exact";  // --dec-attn exact|fast|int8|int8-fast: gqa35's decode attention numerics
int g_attn_select = 0;  // --attn-select N: model35 with page selection at N pages (--spec then audits the key bounds
bool g_sel_kv = false;  // --sel-kv 0|1: select35's study of a KV built on selected attention
int g_sel_self = 0;  // --sel-self K (select35): ModelOptions:select_self, and the model's weights against the host's
std::string g_sel_dump;  // --sel-dump DIR: select35 dumps the dense steps' q and the keys for the host scripts
bool g_sel_raw = false;  // --sel-raw 0|1: select35's --ppl prompts are the file's text as it is, not needle prompts
bool g_sel_pf = false;  // --sel-pf 0|1: select35 measures the page-selected prefill
int g_pf_select = 0;  // --pf-select N: load model35 for page-selected prefill at N pages (ModelOptions:prefill_select
bool g_queue_prof = true;  // --queue-prof 0|1: model35 compute queues with event profiling (0: the oneAPI 2026.2 runtime's
                           // ~15 us per submit gone, walls as served; the profiled lines read 0)
int g_dflash_defer = -1;  // --dflash-defer N: DFlashOptions:ctx_defer (0 = every context pass runs; -1 = the default)
bool g_dflash_prof = false;  // --dflash-prof 1: the drafter's stage wall times after --spec (waits per stage: diagnostic)
bool g_step_reset = false;  // --step-reset 1: model35 --step-ab prefills the shape's slots to --at again before every shape (shapes at one position)
// --pf-moe-a8 0|1: ModelOptions:prefill_moe_a8 for every model command.
bool g_pf_moe_a8 = false;  // (user): W4A16 only by default
// --pf-moe-fast 0|1: ModelOptions:prefill_moe_bias for every model command.
bool g_pf_moe_fast = false;
// --pf-dense-a8 0|1|2 (2 = per row): ModelOptions:prefill_dense_a8 (+ _row) for every model command.
int g_pf_dense_a8 = 0;  // 0 W8A16, 1 W8A8 per group, 2 W8A8 per row with the HC on W8A16
// --pf-wire fp32|bf16: qwen38 ModelOptions:prefill_wire_bf16 for every qwen38 model command.
bool g_q38_pf_wire_bf16 = false;
int g_q38_moe_dn_ks = 0;
// --dec-moe-a8 0|1: ModelOptions:decode_moe_a8 for every model command (default -1 = by the
// checkpoint's act_bits).
int g_dec_moe_a8 = 0;  // (user): W4A16 only by default
int g_q38_mtp_head_w4 = 0;  // --mtp-head-w4 1, the draft vocabulary rows as int4 RTN (qwen38 mtp / serve)
int g_q38_mtp_vocab = 0;  // --mtp-vocab K for the qwen38 commands (mtp, serve, prefix): the draft vocabulary
// --kv-pool N / --kv-scramble 0|1: qwen38 ModelOptions:kv_pool_tokens / kv_pool_scramble for
// every qwen38 model command (0 = the fixed per-slot layout).
int64_t g_kv_pool = 0;
bool g_kv_scramble = false;
// --kv-lend 0|1 / --kv-lend-keep N: qwen38 ModelOptions:kv_lend / kv_lend_keep for every qwen38 model command.
bool g_kv_lend = false;
int64_t g_kv_lend_keep = 32768;
bool g_kv_lend_idle = false;  // --kv-lend-idle 1: probe model refills (Model:lend_idle) before each ppl chunk
// --moe-gu-ks N: ModelOptions:moe_gu_ksplit for probe model (0 = the kernel default).
int g_moe_gu_ks = 0;
std::string g_w4_tensors;
int g_w4_mse = 0;  // model --w4-mse 1
int g_gdn_w4 = 0;  // model --gdn-w4 0..3 (ModelOptions:gdn_w4): bit 1 qkv | z, bit 2 out_proj from the dir's int4 GPTQ tensors
std::string g_gdn_cap;  // model --gdn-cap FILE: the GDN trunk's calibration inputs (XGC1)
int g_gdn_cap_rows = 4096, g_gdn_cap_every = 4;  // model --w4 PAT.., int8 plan matrices loaded as int4 RTN (ModelOptions:w4_tensors)
// --gdn-scan 0|1: ModelOptions:gdn_scan for every model command.
bool g_gdn_scan = true;
// --pf-cold-stage N: ModelOptions:prefill_cold_stage for every model command.
int g_pf_cold_stage = 16;
// --pf-fetch N: ModelOptions:prefill_fetch for every model command.
int g_pf_fetch = 0;
// --pf-graphs 0|1, --pf-graph-gate N, --pf-graph-sizes N,N (probe.hpp)
bool g_pf_graphs = false;
int g_pf_graph_gate = 0;
std::string g_pf_graph_sizes = "16,32,64,96,127";
// --moe-cpu off|auto|avx512|avx2 [off], --moe-cpu-pct P [70], --moe-cpu-threads N [8]: qwen38
// ModelOptions:moe_cpu for probe model (the decode MoE's CPU path for cold experts; rank takes its own flags).
std::string g_moe_cpu = "off";
int g_moe_cpu_pct = 70, g_moe_cpu_threads = 8, g_moe_cpu_min_cold = 2;  // --moe-cpu-min-cold N
// --pf-route-trace F: probe model writes every prefill call's MoE inputs and
// routing to F (XLA1, model.cpp) and exits; xenithium-lookahead scores routing predictors on it.
std::string g_pf_route_trace;
// --lp-time N: probe serve times only a P-token prefill plain vs with prompt
// logprobs on the many-row head, N alternating runs each, prints the medians and a hash of the records
// (bit-compare variants by the hash), then exits.
int g_lp_time = 0;
int g_step_rows = 8;  // --step-rows 4|8: rows a decode step may carry (probe serve)
std::string g_spec_cap;  // --spec-cap F: mtp, per teacher-forced position the target's and the drafter's top logits (XSC1)
int g_two_deep = 0;  // --two-deep 0|1: 2 requests at MTP-3 each (8 rows) instead of MTP-1 each
// --bucket-gate N: probe mtp runs only the bound-bucket gate (mtp.cpp, part 5)
// with N generated tokens across the first QSA bound bucket's end (65,536 tokens), then exits. --bucket-pre 0 skips
// Model:record_bucket_graphs (the control: the long runs must then record graphs).
int g_bucket_gate = 0;
// --p1-kv-gate A,B: probe mtp section 3b, output tokens [A, B) as plain steps
int g_p1_kv_a = -1, g_p1_kv_b = -1;
bool g_bucket_pre = true;
