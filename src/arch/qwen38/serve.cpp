// Xenithium · Qwen3.8 Flash-Next's serving backend: TP2 + EP2 (its experts spread over both cards), TP4, or TP1 on one
// card (every expert whole); the hot-expert placement, the MTP drafter checkpoint (--mtp, TP2), step graphs recorded
// ahead for every QSA bound bucket.
#include "xenithium/arch/qwen38/serve.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>

#include "xenithium/arch/common/vision.hpp"
#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/arch/qwen38/step_model.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/util/env.hpp"
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/util/host_mem.hpp"

namespace xenithium::arch::qwen38 {

namespace {

constexpr int kDenseOneDnnRows = 1024;  // rows per call from which the int8 trunk runs on oneDNN

// --vision 1 (serve::vision_plan): the tower's VRAM kept free through the reserve while the hot experts fill the cards;
// the tower loads after the model.
serve::VisionPlan vision_plan(const serve::Options &o, const model::ModelConfig &cfg) { return serve::vision_plan(o, cfg, o.tp); }

ModelOptions model_options(const serve::Options &o, const model::ModelConfig &cfg) {
  ModelOptions mo;
  mo.slots = o.slots;
  mo.step_rows_cap = o.step_rows > kernels::kStepRows32 ? kernels::kMaxRows : 0;  // --step-rows 64 (four cards)
  mo.context = o.context;
  mo.kv = o.kv == "bf16" ? kernels::QsaKv::bf16 : o.kv == "int8-v4" ? kernels::QsaKv::i8v4 : kernels::QsaKv::i8;
  // The MTP layer's KV in the target's dtype; its KV per token as a target QSA layer's (bf16: 1,088 B per card).
  mo.mtp_kv = mo.kv;
  mo.hot = o.hot > 0 ? o.hot : -1;  // 0 = measured from the free VRAM
  mo.reserve_gib = o.reserve_gib + vision_plan(o, cfg).max_gib();  // the tower's VRAM stays free for it
  mo.graphs = o.graphs;
  mo.prefill_chunk = o.prefill_chunk > 0 ? o.prefill_chunk : 4096;
  mo.prefill_moe_a8 = o.prefill_moe == "w4a8";
  mo.prefill_dense_a8 = o.prefill_dense == "w8a8" || o.prefill_dense == "w8a8-row";
  mo.prefill_dense_a8_row = o.prefill_dense == "w8a8-row";
  mo.prefill_wire_bf16 = o.prefill_wire == "bf16";  // auto = fp32 (a numerics change: opt-in)
  // --prefill-gemm onednn = the int8 trunk's prefill calls of >= kDenseOneDnnRows rows on oneDNN (same bits), only with
  // prefill chunks of >= 2 x kDenseOneDnnRows rows (below, the per-call repack outweighs it)
  mo.prefill_dense_onednn = o.prefill_gemm == "onednn" && mo.prefill_chunk >= 2 * kDenseOneDnnRows ? kDenseOneDnnRows : 0;
  mo.decode_moe_a8 = o.decode_moe == "auto" ? -1 : o.decode_moe == "w4a8" ? 1 : 0;
  mo.prefill_cold_stage = o.prefill_cold_stage;
  mo.kv_lend = o.kv_lend == "on", mo.kv_lend_keep = o.kv_lend_keep;
  mo.gdn_w4 = o.gdn_w4 == "int4" ? 3 : o.gdn_w4 == "int4-qkvz" ? 1 : 0;
  mo.threads = o.threads;
  mo.mtp_dir = o.mtp;
  // --drafter-model DIR = the DFlash 2 block drafter instead of the MTP layer (with --drafter-defer / --drafter-cut);
  // its depth is the block's 7 (adaptive below). With --mtp too and --drafter-mtp-from N > 0, both drafters: DFlash
  // drafts steps of fewer than N sequences, the MTP layer steps of N or more
  if (o.drafter != 0 && !o.drafter_model.empty()) {
    if (!arch::DFlashConfig::is_dflash(o.drafter_model))
      throw serve::OptionError("--drafter-model " + o.drafter_model + ": config.json does not name DFlash2DraftModel");
    const auto dc = arch::DFlashConfig::load(o.drafter_model);
    const auto tc = Config::parse(cfg);
    if (dc.hidden != tc.hidden || dc.target_layers_total != tc.layers)  // e.g. the 27B's drafter from an env shared with qwen35
      throw serve::OptionError("--drafter-model " + o.drafter_model + ": a drafter for another target (hidden " + std::to_string(dc.hidden) + ", " +
                               std::to_string(dc.target_layers_total) + " target layers; this model: " + std::to_string(tc.hidden) + ", " +
                               std::to_string(tc.layers) + ")");
    mo.dflash_dir = o.drafter_model;
    if (o.drafter_mtp_from > 0 && !o.mtp.empty()) mo.mtp_from = o.drafter_mtp_from;
    else mo.mtp_dir.clear();
    mo.dflash_ctx_defer = o.drafter_defer;
    mo.dflash_conf_cut = o.drafter_cut;
    // draft trees (XEN_DRAFTER_TREE / _SEQS); chains beside the MTP layer (its pass 1 writes KV over the verify's rows)
    mo.dflash_tree = mo.mtp_dir.empty() ? o.drafter_tree : 0;
    mo.dflash_tree_seqs = o.drafter_tree_seqs;
  }
  // a KV pool shared by the slots; auto = at least min(slots, 4) contexts, each counted at most at the trained
  // positions, and at least one whole context (past the trained positions the max context runs on YaRN)
  mo.kv_pool_tokens = o.kv_pool_tokens >= 0 ? o.kv_pool_tokens : serve::shared_pool_auto(o.slots, o.context, Config::parse(cfg).max_positions);
  mo.kv_pool_fill = o.kv_pool_tokens < 0;  // auto: also the VRAM every routed expert leaves (ModelOptions::kv_pool_fill)
  mo.snap_gib = o.prefix_cache ? o.prefix_cache_gib : 0;  // the prefix cache's snapshot store
  mo.mtp_vocab = o.draft_vocab;  // the drafter's lm_head on the draft vocabulary (0 = all)
  mo.rank_file = o.hot_rank;  // read when it exists (the placement), recorded while serving
  mo.moe_cpu.mode = o.moe_cpu;  // the decode MoE's CPU path for cold experts
  mo.moe_cpu.threads = o.moe_cpu_threads;
  mo.moe_cpu.pct = o.moe_cpu_pct;
  mo.moe_cpu.min_rows = o.moe_cpu_min_rows;
  mo.moe_cpu.min_cold = o.moe_cpu_min_cold;
  mo.moe_cpu.timeout_ms = o.moe_cpu_timeout_ms;
  if (!o.moe_cpu_cores.empty()) {
    std::stringstream ss(o.moe_cpu_cores);
    for (std::string t; std::getline(ss, t, ',');)
      if (!t.empty()) mo.moe_cpu.cores.push_back(std::atoi(t.c_str()));
  }
  return mo;
}

// --plan: the memory a launch will have, before it. An ESTIMATE of the load's own arithmetic, per card:
//   room    = the card - the weights - the runtime (scratch, states, graphs, staging) - the reserve
//   KV pool = a number: that many tokens; auto: the room every routed expert leaves (room - all experts), at least
//             min(slots, 4) contexts of at most the trained length (serve::shared_pool_auto), at most slots x max context
//   hot     = what the room holds after the pool, at most every expert (a pinned --hot is cut to it)
// Experts come before the pool: while some stay cold, every pool GiB is ~800 experts read over PCIe, so the auto pool
// grows past its least only once every expert has its place (TP4 here). Weights and KV bytes per token from the arch
// budget (the KV is exact: the QSA pages' K, V, scales and compressed keys); kRuntimeGiB (the rest the budget does not
// model: GDN states per slot, prefill scratch per chunk row, graphs, staging) and kMtpGiB = the --mtp drafter's weights
// per card are fitted to loads.
std::string plan(const serve::Options &o, const model::ModelConfig &cfg, std::string_view) {
  constexpr double G = 1073741824.0, kMtpGiB = 0.42;
  constexpr double kDFlashGiB = 0.37;  // the DFlash drafter (no KV pages in the pool)
  const int chunk_rows = o.prefill_chunk > 0 ? o.prefill_chunk : 4096;
  // + the 64-row step's scratch (--step-rows 64: the MoE's per-slot partials at 64 rows)
  const double kRuntimeGiB = 1.153 + 0.0525 * o.slots + 0.000533 * std::max(0, chunk_rows - 512) + (o.step_rows > 32 ? 0.42 : 0.0);
  serve::Options oo = o;
  oo.hot_rank.clear();  // the ranking does not move the counts
  const ModelOptions mo = model_options(oo, cfg);
  const auto c = Config::parse(cfg);
  const auto ck = model::Checkpoint::open(o.model_dir);
  const auto arch = model::open_arch(cfg);
  model::DeployParams p;
  const int cards = o.tp;  // TP1, TP2 or TP4
  p.cards = cards, p.context = o.context, p.seqs = o.slots, p.kv_bytes = o.kv == "bf16" ? 2 : 1, p.kv_v4 = o.kv == "int8-v4", p.drafter = !mo.mtp_dir.empty();
  const model::Budget b = arch->budget(ck, p);
  serve::MemPlan m;
  m.arch = "qwen38", m.exact = false, m.cards = cards, m.card_gib = p.vram_gib;
  for (const auto &r : b.rows)
    if (r.name.rfind("KV cache", 0) != 0) m.weights_gib += r.resident_bytes / G;
  if (!mo.mtp_dir.empty()) m.weights_gib += kMtpGiB;
  if (!mo.dflash_dir.empty()) m.weights_gib += kDFlashGiB;
  m.scratch_gib = kRuntimeGiB + o.reserve_gib + vision_plan(o, cfg).max_gib();
  if (mo.prefill_dense_onednn > 0)  // the oneDNN route's repack slots at the largest int8 matrix (the PLE kv)
    m.scratch_gib += static_cast<double>(kernels::onednn_dense_scratch_bytes(static_cast<size_t>(c.hc + 1) * c.hidden * c.ple_dim)) / G;
  size_t stride = ExpertLayout::of(c).stride;
  kernels::moe_tp_layout(c.hidden, c.moe_inter, cards, &stride);  // TP2: each card holds its half of every expert; TP1 whole
  m.moe_layers = c.layers, m.experts_total = static_cast<int64_t>(c.layers) * c.experts;
  // the KV pool in 64-token pages (Model::kv_pool_pages_for)
  constexpr int64_t kPage = 64;
  const double page = b.kv_bytes_per_token * kPage;
  const int64_t all_pages = static_cast<int64_t>(o.slots) * ((o.context + kPage - 1) / kPage);
  int64_t pages = mo.kv_pool_tokens > 0 ? std::min((mo.kv_pool_tokens + kPage - 1) / kPage, all_pages) : all_pages;
  if (mo.kv_pool_fill) {
    const double experts = static_cast<double>(m.experts_total) * static_cast<double>(stride);
    const double keep = Model::pool_fill_keep_gib((o.context + kPage - 1) / kPage * kPage);
    const double left = (m.card_gib - m.weights_gib - m.scratch_gib - keep) * G - experts;
    const int64_t least = pages;
    pages = std::clamp<int64_t>(left > 0 ? static_cast<int64_t>(left / page) : 0, least, all_pages);
    if (pages > least) m.scratch_gib += keep;  // the bound-bucket graphs then take it (it shows in the VRAM total)
  }
  m.kv_tokens = pages * kPage;
  m.kv_gib = static_cast<double>(pages) * page / G;
  const double room = (m.card_gib - m.weights_gib - m.kv_gib - m.scratch_gib) * G;
  const int64_t fit = std::clamp<int64_t>(static_cast<int64_t>(room / static_cast<double>(stride)), 0, m.experts_total);
  m.hot_experts = mo.hot >= 0 ? std::min<int64_t>(mo.hot, fit) : fit;  // the load cuts a pinned count to the fit
  m.hot_gib = static_cast<double>(m.hot_experts) * stride / G;
  m.context = o.context, m.slots = o.slots;
  m.cold_gib = static_cast<double>(m.experts_total - m.hot_experts) * stride * cards / G;
  m.embed_gib = static_cast<double>(c.vocab) * c.hidden * 2 / G;
  m.prefix_gib = mo.snap_gib;
  const util::HostMem hm = util::host_mem();
  m.host_total_gib = hm.total / G, m.host_available_gib = hm.available / G;
  m.host_other_gib = util::kLoadHostOverheadGib, m.host_keep_gib = util::kHostKeepGib;
  return serve::plan_json(m);
}

class Backend final : public serve::Backend {
 public:
  Backend(const serve::Options &o, const model::ModelConfig &cfg) : o_(o) {
    if (o_.drafter != 0 && !o_.drafter_model.empty()) o_.mtp_depth = 7;  // DFlash drafts its block - 1 (the adapter caps it)
    if (o_.mtp_depth < 1 || o_.mtp_depth > (o_.drafter_model.empty() || o_.drafter == 0 ? StepModelAdapter::kDepth : 7))
      throw serve::OptionError("--mtp-depth takes 1.." + std::to_string(StepModelAdapter::kDepth) + " (the MTP depth of one decoding sequence)");
    loader_ = std::make_unique<l0::Loader>(l0::Loader::open());
    const auto cards = serve::cards_for(*loader_, o_.tp, o_.card, "qwen38", true, true);  // TP2 + EP2, TP1 or TP4
    const ModelOptions mo = model_options(o_, cfg);
    // queue profiling only for XEN_PF_PROFILE (its phase splits read event times)
    fab_ = std::make_unique<Fabric>(*loader_, cards, util::env_flag("XEN_PF_PROFILE"));
    model_ = std::make_unique<Model>(*loader_, *fab_, o_.model_dir, mo);
    const Model &m = *model_;
    const auto &ls = m.load_stats();
    std::fprintf(stderr, "server: model loaded in %.1f s (%s); %d slots x %lld tokens, KV %s; hot experts %lld of %lld (%s); graphs %s%s%s\n",
                 ls.total_s, m.cards() == 1 ? ("TP1 on card " + std::to_string(o_.card)).c_str() : m.cards() == 4 ? "TP4" : "TP2", m.slots(), static_cast<long long>(m.capacity()), o_.kv.c_str(), static_cast<long long>(ls.hot[0]),
                 static_cast<long long>(ls.hot[0] + ls.cold[0]), m.cards() == 1 ? "whole on the card" : m.cards() == 4 ? "each split over the four cards" : "each split over both cards", o_.graphs ? "on" : "off",
                 m.has_dflash()  ? ("; DFlash 2 drafter " + o_.drafter_model + " (block " + std::to_string(model_->dflash()->config().block) + ")" +
                                     (m.dual_drafters() ? " + the MTP layer for steps of " + std::to_string(m.mtp_from()) + "+ sequences" : std::string())).c_str()
                 : m.has_drafter() ? ("; MTP drafter (depth " + std::to_string(o_.mtp_depth) + ")").c_str()
                                   : "",
                 m.has_dflash() && m.dflash()->draft_vocab() > 0 ? (" (draft vocabulary " + std::to_string(m.dflash()->draft_vocab()) + ")").c_str()
                 : m.mtp_draft_vocab() > 0                       ? (" (draft vocabulary " + std::to_string(m.mtp_draft_vocab()) + ")").c_str()
                                                                 : "");
    if (ls.hot_asked >= 0)
      std::fprintf(stderr, "server: --hot %lld cut to %lld: that is what the VRAM left after the KV pool, drafter and states holds minus --reserve %.2f GiB "
                           "(more would make the driver evict and page: every step many times slower)\n",
                   static_cast<long long>(ls.hot_asked), static_cast<long long>(ls.hot[0]), o_.reserve_gib);
    if (ls.onednn_gib > 0)
      std::fprintf(stderr, "server: prefill trunk GEMMs on oneDNN from %d rows per call (--prefill-gemm onednn; %.3f GiB per card of repack slots)\n",
                   mo.prefill_dense_onednn, ls.onednn_gib);
    serve::log_rope(m.capacity(), m.config().max_positions, m.yarn_factor());
    serve::log_ranking(m.ranking(), o_.hot_rank);
    std::fprintf(stderr, "server: MoE CPU path %s\n", ls.moe_cpu.c_str());  // --moe-cpu
    std::fprintf(stderr, "server: KV lending %s\n", ls.kv_lend.c_str());  // --kv-lend
    adapter_ = std::make_unique<StepModelAdapter>(*model_);
    adapter_->set_depth(o_.mtp_depth);
    // 16 (the default), 32 or 8 (--step-rows / XEN_STEP_ROWS)
    adapter_->set_step_rows(o_.step_rows_or(16));
    std::fprintf(stderr, "server: steps of up to %d rows\n", adapter_->max_rows());
    if (const char *ds = std::getenv("XEN_STEPSTAT")) {  // diagnostic: what the depth path is handed
      (void)ds;
      std::fprintf(stderr, "DEPTHCFG depth_=%d seg_rows_max=%d draft_depth(1)=%d draft_depth(2)=%d layout_depth(1)=%d wide_depth=%d wide_from=%d max_rows=%d\n",
                   adapter_->depth(), adapter_->seg_rows_max(), adapter_->draft_depth(1), adapter_->draft_depth(2),
                   adapter_->layout_depth(1), adapter_->wide_depth(), adapter_->wide_from(), adapter_->max_rows());
    }
    std::fprintf(stderr, "server: KV %s, %lld tokens (%.2f / %.2f GiB on the cards); one request up to %lld tokens (max_model_len)\n",
                 m.kv_shared() ? "pool shared by the slots" : "fixed per slot", static_cast<long long>(m.kv_pool_tokens()), m.kv_bytes(0) / 1073741824.0,
                 m.kv_bytes(m.cards() - 1) / 1073741824.0, static_cast<long long>(m.capacity()));
    serve::log_prefix_cache(m.snap_capacity(), static_cast<double>(m.snap_bytes(0)), m.cards() > 1 ? static_cast<double>(m.snap_bytes(1)) : 0.0, o_, false,
                            std::max(0, m.cards() - 2));  // TP4
    if (const serve::VisionPlan vp = vision_plan(o_, cfg); vp.on) {  // the tower in the VRAM the reserve kept
      vis_ = serve::load_vision(*fab_, o_, vp, m.config().vocab);
      model_->set_vision(vis_.get());
      hook_ = serve::make_vision_hook(*vis_, o_);
    }
  }

  core::StepModel &step_model() override { return *adapter_; }
  server::MediaHook *media() override { return hook_.get(); }
  int64_t vocab() const override { return model_->config().vocab; }
  // four cards: parts of 90 ms with whole-chunk prompt calls (step_model.hpp prefill_long_parts); one / two cards keep
  // 12 ms with the stall budget's calls
  double prefill_part_ms() const override { return model_->cards() == 4 ? 90 : 12; }

  void start(core::BatchEngine &be, const text::Tokenizer &tok, bool warmup) override {
    if (!warmup) return;
    Model &m = *model_;
    const int g0 = m.graphs_recorded(), d0 = m.mtp_graphs_recorded();
    m.set_learning(false);  // synthetic warm-up traffic does not enter the recorded ranking
    const double ms = be.warm_up(serve::warm_plan(tok, adapter_->prefill_chunk(), adapter_->capacity(), be.model().prompt_logprobs, m.slots()));
    m.set_learning(true);
    std::fprintf(stderr, "server: warm-up took %.0f ms (step graphs recorded %d, drafter graphs %d)\n", ms, m.graphs_recorded() - g0, m.mtp_graphs_recorded() - d0);
    // The same graphs at every longer QSA bound bucket, recorded without running, so the first request past 65,536
    // tokens (131,072, ...) does not record them. Bucket by bucket, ascending, while the tighter card keeps
    // kGraphVramFloorGiB free: the graphs hold VRAM outside the placement budget. Buckets past the floor record on
    // first use.
    constexpr double kGraphVramFloorGiB = 0.125;
    const int g1 = m.graphs_recorded(), d1 = m.mtp_graphs_recorded();
    const double v0 = serve::vram_free_gib(*loader_, *fab_);
    const auto tb = std::chrono::steady_clock::now();
    int nb = 0;
    int64_t upto = 0;
    bool floor_hit = false;
    for (const int64_t b : m.bound_buckets()) {
      const double f = serve::vram_free_gib(*loader_, *fab_);
      if (f >= 0 && f < kGraphVramFloorGiB) {
        floor_hit = true;
        break;
      }
      nb += m.record_bucket_graphs(b);
      upto = b;
    }
    std::fprintf(stderr, "server: bound-bucket graphs recorded ahead: %d (step %d, drafter %d) in %.0f ms, up to %lld of %lld tokens%s; VRAM free %.3f -> %.3f GiB\n", nb,
                 m.graphs_recorded() - g1, m.mtp_graphs_recorded() - d1, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count(),
                 static_cast<long long>(upto), static_cast<long long>(m.capacity()), floor_hit ? " (stopped at the VRAM floor)" : "", v0,
                 serve::vram_free_gib(*loader_, *fab_));
  }

 private:
  serve::Options o_;
  std::unique_ptr<l0::Loader> loader_;
  std::unique_ptr<Fabric> fab_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<StepModelAdapter> adapter_;
  std::unique_ptr<arch::VisionStore> vis_;  // --vision 1: destroyed before the fabric
  std::unique_ptr<server::MediaHook> hook_;
};

}  // namespace

serve::BackendEntry serve_entry() {
  return {"qwen38", {"qwen38"},
          [](const serve::Options &o, const model::ModelConfig &cfg, std::string_view) -> std::unique_ptr<serve::Backend> { return std::make_unique<Backend>(o, cfg); },
          plan};
}

}  // namespace xenithium::arch::qwen38
