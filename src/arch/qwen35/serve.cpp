// Xenithium · the qwen35 runtime's serving backend: the dense Qwen3.8-27B (its own MTP drafter or a DFlash 2 block
// drafter), and the archs that run on the same runtime: llama (Nanbeige4.3, dense) and flint (Recursive Flint INT4,
// MoE). TP2 or TP1. Arch-specific resolution of the shared settings lives here (resolve()).
#include "xenithium/arch/qwen35/serve.hpp"
#include "xenithium/kernels/attention/gqa.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

#include "xenithium/arch/common/drafter_archs.hpp"
#include "xenithium/arch/common/vision.hpp"
#include "xenithium/arch/flint/arch.hpp"
#include "xenithium/arch/llama/arch.hpp"
#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/arch/qwen35/step_model.hpp"
#include "xenithium/arch/qwen35/trained_qsa_device.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/util/host_mem.hpp"

namespace xenithium::arch::qwen35 {

namespace {

// The settings as this arch serves them: `auto` values resolved, options the arch has no use for switched off.
serve::Options resolve(serve::Options o, std::string_view arch_id) {
  const bool llama = arch_id == "llama", flint = arch_id == "flint";
  if (o.kv == "int8-v4" && (llama || flint))  // the 4-bit V cache: qwen38's QSA kernels + the 27B's
    throw serve::OptionError("--kv int8-v4: K int8 + V int4 is built for qwen38's and the dense 27B's attention; arch " + std::string(arch_id) + " takes int8|bf16");
  if (o.kv_split != "heads")  // qwen38's QSA kernels on four cards
    throw serve::OptionError("--kv-split tokens: the KV split by tokens is built for qwen38 on four cards; arch " + std::string(arch_id) + " takes heads");
  const bool kv8 = o.kv != "bf16";  // int8 K codes (int8 or int8-v4)
  if (o.kv == "int8-v4" && o.prefill_attn == "onednn")
    throw serve::OptionError("--kv int8-v4: oneDNN's attention (--prefill-attn onednn) takes int8|bf16 KV");
  if (o.kv == "int8-v4" && o.attn_select > 0)
    throw serve::OptionError("--kv int8-v4: page-selected attention (--attn-select) takes int8|bf16 KV");
  const bool pf_attn_auto = o.prefill_attn == "auto";
  if (pf_attn_auto) o.prefill_attn = kv8 ? "int8-fast" : "fast";
  // the page-selected prefill attention is built in the exact numerics and in what auto serves
  // (int8-fast on the int8 KV, fast on the bf16 KV)
  if (o.prefill_select > 0 && o.prefill_attn != "exact" && o.prefill_attn != (kv8 ? "int8-fast" : "fast"))
    throw serve::OptionError("--prefill-select: page-selected prefill runs with --prefill-attn auto, exact or " + std::string(kv8 ? "int8-fast" : "fast") +
                             " on this KV (not " + o.prefill_attn + ")");
  // Draft trees (--drafter-tree with a block drafter on the dense 27B) verify on dense attention over an int8, int8-v4,
  // or bf16 KV (tree-masked group attention; int8-v4 reads int4 V and its bf16 scales). Tree steps run under every
  // decode attention numerics, and with page selection too for the steps under its page budget: the model drafts chains
  // once a sequence of the step passes it, Model::selects_pages.
  const bool trees = !llama && !flint && o.drafter != 0 && !o.drafter_model.empty() && o.drafter_tree > 0;
  if (trees && o.attn_select > 0)
    std::fprintf(stderr, "server: draft trees for steps under the page budget (--attn-select %d pages); a sequence past it, and the ones decoding beside it, draft chains\n",
                 o.attn_select);
  // auto = int8-fast where it is built - the dense 27B's heads on the int8 K codes, dense or page-selected - else exact
  // (bf16 KV, llama / flint)
  if (o.decode_attn == "auto") o.decode_attn = !llama && !flint && kv8 ? "int8-fast" : "exact";
  if (o.lm_head == "auto") {  // by the directory's head formats: int8 rows need lm_head.weight, int4 rows lm_head.qweight
    try {
      const auto ck = model::Checkpoint::open(o.model_dir);
      const bool w4 = ck.find("lm_head.qweight") != nullptr, bf = ck.find("lm_head.weight") != nullptr;
      o.lm_head = w4 && bf ? "int4-rescore" : w4 ? "int4" : "int8";
    } catch (const std::exception &) {
      o.lm_head = "int8";  // an unreadable directory fails the load with its own message
    }
  }
  if (!llama && !flint) return o;
  const char *name = llama ? "llama" : "flint";
  if (o.prefill_attn == "onednn")
    throw serve::OptionError(std::string("--prefill-attn onednn: oneDNN's attention is built for head_dim 256 with the gate (qwen35); arch ") + name +
                             " takes auto|exact|fast|int8|int8-fast");
  if (llama && o.drafter != 0 && !o.drafter_model.empty())  // a launcher env shared with qwen35 may carry one: ignored, said once
    std::fprintf(stderr, "server: arch %s serves without a drafter (--drafter-model %s ignored)\n", name, o.drafter_model.c_str());
  if (llama) o.drafter_model.clear();
  if (o.attn_select > 0)
    throw serve::OptionError(std::string("--attn-select: page-selected attention is built for head_dim 256 (qwen35); arch ") + name + " runs dense attention");
  if (o.prefill_select > 0)
    throw serve::OptionError(std::string("--prefill-select: page-selected prefill is built for head_dim 256 (qwen35); arch ") + name + " runs dense attention");
  if (o.prefill_gemm == "onednn-all")  // (the routing and lending of the dense 27B's symmetric g128 matrices only)
    throw serve::OptionError(std::string("--prefill-gemm onednn-all: every int4 prefill GEMM on oneDNN is built for the dense 27B (qwen35); arch ") + name +
                             " takes onednn|esimd");
  if (o.wire == "int8")  // (built for the dense 27B's sums only)
    throw serve::OptionError(std::string("--wire int8: the int8 wire is built for the dense 27B (qwen35); arch ") + name + " takes bf16 or fp32");
  if (o.decode_w4 != "exact")  // (the dense 27B's symmetric g128 matrices only)
    throw serve::OptionError(std::string("--decode-w4 ") + o.decode_w4 + ": the W4A8 decode GEMMs are built for the dense 27B (qwen35); arch " + name + " runs exact");
  if (o.decode_attn != "exact")  // (grp 6 at head_dim 256 only)
    throw serve::OptionError(std::string("--decode-attn ") + o.decode_attn + ": the decode attention numerics are built for qwen35's heads; arch " + name + " runs exact");
  // llama has no drafter; flint a DFlash 2 checkpoint (--drafter-model, Flint-DFlash2) and no MTP layer: its drafter
  // drafts chains (8-row steps; the tree verify needs a KV move the rings lack) over the whole lm_head (no draft
  // vocabulary), alone at every stream count (no MTP from N streams)
  if (o.drafter_model.empty()) o.drafter = 0;
  if (flint && o.drafter != 0) {
    if (o.seg_rows == 16) o.seg_rows = 8;  // 16 = the default; flint's steps hold 8 rows, so it runs 8 (the load line says)
    o.drafter_tree = 0, o.drafter_tree_seqs = 1, o.drafter_wide = 0, o.drafter_mtp_from = 0, o.draft_vocab = 0;
    o.drafter_int4 = 0;  // (its drafter's matrices stay int8 per row: int4 is built for the 27B's drafters only)
  }
  if (flint) {
    o.lm_head = "mlx-int4";  // the checkpoint's own int4 matrix (no switch)
    // the many-row prefill. W4A8 is not offered (a W4A16 checkpoint): a8 (the qwen35 default) = dequant; auto attention
    // = fast (P hi only); int8 / int8-fast stay offered (Model::flint_options)
    if (o.prefill_w4 == "a8") o.prefill_w4 = "dequant";
    if (pf_attn_auto) o.prefill_attn = "fast";
  } else {
    // the int4 g32 zero-point matrices run the dequantised W4A16 prefill GEMM (the model pins it, and maps the prefill
    // attention to the bodies built at head_dim 128: Model::llama_prefill)
    o.prefill_w4 = "dequant";
    // auto = exact on this model
    if (pf_attn_auto) o.prefill_attn = "exact";
  }
  return o;
}

ModelOptions model_options(const serve::Options &o, bool flint) {
  ModelOptions mo;
  mo.slots = o.slots;
  mo.step_rows = o.step_rows_or(32);  // 32 = the default (the model runs llama / flint at 8)
  mo.context = o.context;
  mo.kv_pool_tokens = o.kv_pool_tokens;
  mo.kv_lend = o.kv_lend == "on", mo.kv_lend_keep = o.kv_lend_keep;  // the wide prefill matrices' oneDNN layouts
  mo.kv = o.kv == "bf16" ? kernels::QsaKv::bf16 : o.kv == "int8-v4" ? kernels::QsaKv::i8v4 : kernels::QsaKv::i8;
  if (o.prefill_chunk > 0) mo.prefill_chunk = o.prefill_chunk;
  mo.threads = o.threads;
  mo.mtp = o.drafter != 0;
  mo.mtp_vocab = o.draft_vocab;
  mo.mtp_kv_defer = (o.drafter_defer + 3) / 4 * 4;  // MTP's deferred KV rows (a multiple of 4)
  // onednn = oneDNN's fused GQA attention (kernels::gqa_attn_sdpa; its scratch is reserved at load)
  if (o.prefill_attn == "onednn") mo.prefill_attn = kernels::kGqaAttnSdpa;
  mo.prefill_attn_p_lo = o.prefill_attn == "exact" || o.prefill_attn == "int8" || o.prefill_attn == "onednn";
  // int8 q.k (int8 / int8-fast: one code per element; fast on the int8 KV: hi + lo codes, P hi only)
  mo.prefill_attn_qk = o.prefill_attn.rfind("int8", 0) == 0 ? 1 : o.prefill_attn == "fast" && o.kv != "bf16" ? 2 : 0;
  mo.prefill_w4_dequant = o.prefill_w4 == "dequant";
  mo.prefill_w4_a8 = o.prefill_w4 == "a8";  // (on the nib blocks, staged as s4 operands)
  mo.prefill_a8_keep = o.prefill_a8_keep;
  // the W4A8 GEMMs' ESIMD body; auto = slm at TP2 (its consumer GEMMs run under the split sums' sends) with direct on
  // calls of <= 128 rows, direct at TP1 (Model::a8_body_for)
  mo.prefill_a8_body = o.prefill_a8_body == "slm" ? 0 : o.prefill_a8_body == "direct" ? 1 : -1;
  // the W4A8 prefill GEMMs on oneDNN by their output width per card, repacked per call into a 2-slot scratch (no
  // stored copy) or read from the layouts KV lending keeps; calls under 1,024 rows per card stay ESIMD.
  // onednn = the wide ones (gate|up, GDN qkvz, attn qkv: n >= kOneDnnWideMinN); onednn-all = every W4A8 prefill GEMM
  // (also down, GDN out and attn o: oneDNN's rounding on them, a numerics change)
  constexpr int kOneDnnWideMinN = 6144, kOneDnnAllMinN = 1;
  const int onednn_min_n = o.prefill_w4 != "a8" ? 0 : o.prefill_gemm == "onednn" ? kOneDnnWideMinN : o.prefill_gemm == "onednn-all" ? kOneDnnAllMinN : 0;
  kernels::w4a16_pf_set_onednn(onednn_min_n);
  // the route's row threshold (default 1,024; lower = more calls on oneDNN's rounding, a numerics change, meant for the
  // layouts kept by KV lending: --kv-lend)
  kernels::w4a16_pf_set_onednn_rows(o.prefill_gemm_rows);
  // the prefill attention's last wave split over the context (the call's W mod C work-groups as merged units)
  kernels::gqa_attn_fa_set_tail(o.attn_tail);  // 0 / 1 / 2 (2 = short calls split too)
  // a prefill call of <= prefill_short rows (1-16) runs its int4 GEMMs on the decode kernels (W4A16, decode's k split)
  mo.prefill_short = o.prefill_short;
  // the prefill two-card sums (split and rows wires, the drafters' too) on bf16 partials
  mo.prefill_wire_bf16 = o.prefill_wire != "fp32";  // auto = bf16
  mo.yarn_tiers = o.yarn_tiers == "on";
  mo.head_w4 = !flint && o.lm_head != "int8";  // lm_head.qweight (arch flint: its lm_head as loaded)
  mo.head_rescore = !flint && o.lm_head == "int4-rescore" ? o.lm_head_tiles : 0;
  mo.draft_head = o.draft_head == "auto" ? -1 : o.draft_head == "int4" ? 1 : 0;
  mo.snap_gib = o.prefix_cache ? o.prefix_cache_gib : 0;  // (arch flint: its ring windows in the snapshot)
  mo.attn_select = o.attn_select;
  if (o.attn_select > 0 && !flint && o.select_heads == "auto") mo.select_self = kSelectSelfSteps;
  mo.prefill_select = o.prefill_select;
  if (o.drafter != 0 && !o.drafter_model.empty()) {
    if (!DFlashConfig::is_dflash(o.drafter_model)) throw serve::OptionError("--drafter-model " + o.drafter_model + ": config.json names no block drafter (" + block_drafter_arch_names() + ")");
    mo.dflash_dir = o.drafter_model;
    mo.dflash_ctx_defer = o.drafter_defer;
    mo.dflash_conf_cut = o.drafter_cut;
    mo.dflash_tree = o.drafter_tree;  // DDTree
    mo.dflash_tree_seqs = o.drafter_tree_seqs;
    mo.seg_rows = o.seg_rows, mo.dflash_tree_wide = o.drafter_tree_wide;  // 16 needs DSpine b16 (the model checks the block)
    mo.mtp_from = o.drafter_mtp_from;  // both drafters
    mo.dflash_dense_bytes = o.drafter_int4 != 0 ? 4 : 1;  // q, o, gate | up, down as int4 (RTN at load)
  }
  // Depth 4..6 needs a 5- to 7-row verify: raise the GDN pending logs to 8 rows. DFlash / DSpine already do.
  if (o.drafter != 0 && o.drafter_model.empty() && o.mtp_depth > 3) mo.rows8 = true;
  std::vector<int> want;
  if (o.drafter != 0 && o.drafter_model.empty() && serve::parse_draft_depths(o.draft_depths, want))
    for (int x : want)
      if (x > 3) mo.rows8 = true;
  return mo;
}

// --plan: the KV pool a launch will have, before it. An ESTIMATE: the load sizes an auto pool from the free VRAM it
// measures after the warm-up (Model::kv_fit), so the plan models that number: per card
//   room = the card - the weights (the arch budget) - what the runtime holds besides the KV,
//   pool = (room - the reserve - arch flint's rings) / the bytes of one KV token, at most slots x max context.
// The KV bytes per token are the load's own arithmetic (the arch budget; +1 int8 layer with the MTP drafter). The
// runtime's part is an empirical fit (the constants below) over the slots, the max context, TP2 / TP1, the KV format,
// the drafter and its matrices' format, the segment rows, the prefill chunk, image input, the prefix cache, the lm_head
// format and the prefill route.
// Not modelled: another process's VRAM, a card that is not idle. A request the load would refuse (one max context of KV
// does not fit) throws with the context that does.
std::string plan_of(const serve::Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) {
  constexpr double G = 1073741824.0;
  const serve::Options o = resolve(opt, arch_id);
  const bool llama = arch_id == "llama", flint = arch_id == "flint", dense27 = !llama && !flint;
  const int C = o.tp == 1 ? 1 : 2;
  if (!dense27 && o.slots > 8) throw serve::OptionError("slots 1..8 (arch " + std::string(arch_id) + ")");  // (the model's check)
  const double half = 2.0 / C;  // what splits over the cards by head or row: a TP2 card holds half of it
  const auto ck = model::Checkpoint::open(o.model_dir);
  const auto arch = model::open_arch(cfg);
  const Config c = llama ? arch::llama::parse_config(cfg) : flint ? arch::flint::runtime_config(cfg) : Config::parse(cfg);
  model::DeployParams p;
  p.cards = C, p.context = o.context, p.seqs = o.slots, p.kv_bytes = o.kv == "bf16" ? 2 : 1, p.kv_v4 = o.kv == "int8-v4";
  const model::Budget b = arch->budget(ck, p);
  // the drafters: the MTP layer (the dense 27B's own; with a block drafter only when --drafter-mtp-from > 0), DFlash 2 / DSpine
  int block = 0;  // 1 DFlash 2, 2 DSpine
  if (o.drafter != 0 && !o.drafter_model.empty() && !llama) {
    if (!DFlashConfig::is_dflash(o.drafter_model)) throw serve::OptionError("--drafter-model " + o.drafter_model + ": config.json names no block drafter (" + block_drafter_arch_names() + ")");
    const DFlashConfig dc = DFlashConfig::load(o.drafter_model, o.seg_rows >= 16 ? 16 : 8);
    if (dc.hidden != c.hidden || dc.vocab != c.vocab || dc.target_layers_total != c.layers)  // (the load's check: DFlash's constructor)
      throw serve::OptionError("another model's drafter (its hidden size, vocabulary or layer count differ)");
    block = dc.kind == DFlashConfig::Kind::dspine ? 2 : 1;
  }
  const bool mtp = dense27 && o.drafter != 0 && (block == 0 || o.drafter_mtp_from > 0);
  // one KV token on one card
  double kvb = b.kv_bytes_per_token;
  if (mtp) {  // the MTP layer's KV pages: one int8 layer
    model::DeployParams p8 = p, p8d = p;
    p8.kv_bytes = p8d.kv_bytes = 1, p8.kv_v4 = p8d.kv_v4 = false, p8d.drafter = true;
    kvb += arch->budget(ck, p8d).kv_bytes_per_token - arch->budget(ck, p8).kv_bytes_per_token;
  }
  const bool page_bounds = o.attn_select > 0 || o.prefill_select > 0;
  if (page_bounds) kvb += 32.0 * c.full_layers() * half;  // the pages' key bounds
  serve::MemPlan m;
  m.arch = std::string(arch_id), m.exact = false, m.cards = C, m.card_gib = p.vram_gib;
  for (const auto &r : b.rows)
    if (r.name.rfind("KV", 0) != 0) m.weights_gib += r.resident_bytes / G;
  m.weights_gib += b.expert_bytes * static_cast<double>(b.experts_per_card) / G;  // arch flint: every routed expert is resident
  const int chunk = o.prefill_chunk > 0 ? o.prefill_chunk : 2048;
  const double ctx_m = static_cast<double>(o.context) / 1e6, kchunk = (chunk - 2048) / 1024.0;
  double first_use = 0;  // VRAM the warm-up takes after the load line (kept free by a start without one)
  double staged = 0;     // taken after the load line too: the prefix cache's staging, the image tower
  if (dense27) {
    if (o.prefix_cache) staged += 0.186 * half;
    m.scratch_gib = 1.141 + (C == 1 ? 0.479 : 0.0) + 0.317 * kchunk;
    m.state_gib = 0.0741 * o.slots * half + 0.248 * ctx_m + 0.198 * ctx_m * o.slots / 16;
    if (mtp) m.weights_gib += 0.491 * half;
    // the MTP layer's int8 draft rows of lm_head (--draft-vocab; 0.234 GiB per TP2 card at the default 98,304); a block
    // drafter keeps its own
    if (mtp && !block) m.weights_gib += (static_cast<double>(o.draft_vocab) - 98304.0) * c.hidden / G / C;
    if (block) {
      m.weights_gib += (block == 2 ? 1.375 : 1.236) * half + (o.drafter_int4 == 0 ? 0.354 * half : 0.0) - (C == 1 ? 0.598 : 0.0);
      m.state_gib += (block == 2 ? 0.0978 + (o.seg_rows >= 16 ? 0.0236 : 0.0) : 0.0402) * o.slots * half;
      if (o.prefix_cache) staged += (block == 2 ? 0.276 : 0.104) * half;
    }
    if (o.kv == "int8-v4") m.scratch_gib -= 0.056;
    if (page_bounds) m.scratch_gib -= 0.052;
    if (o.lm_head == "int4") m.weights_gib -= 0.512 * half;  // the budget's row is the int8 head
    if (o.lm_head == "int4-rescore") m.weights_gib += 0.078 * half;
    if (o.prefill_attn == "onednn") m.scratch_gib += 0.95 * ctx_m;  // its scratch grows with the context (0.50 GiB at 524,288, 0.27 at 262,144)
    if (o.prefill_w4 == "exact") m.scratch_gib -= 0.182;  // (dequant on this arch: counted as a8)
    else if (o.prefill_w4 == "a8" && o.prefill_gemm == "esimd") m.scratch_gib -= 0.142;
    first_use = 0.24;
    // The trained page selection (a bank in the environment, read as the load reads it): every slot's block summaries over
    // the max context, the scorer's weights and scratch, allocated at load before the KV (the load's own arithmetic).
    if (const char *bank = std::getenv("XEN_TRAINED_QSA_DELTA_DIR"); bank && *bank) {
      size_t layers = 0;
      try {
        layers = model::Checkpoint::open(bank, false).tensors().size() / 5;  // five tensors per trained layer
      } catch (const std::exception &e) {
        throw serve::OptionError(std::string("XEN_TRAINED_QSA_DELTA_DIR: ") + e.what());
      }
      const char *pfx = std::getenv("XEN_TRAINED_QSA_PREFILL_XMX");
      m.state_gib += static_cast<double>(trained_qsa::Device::bytes_per_rank(layers, c.hidden, o.slots, static_cast<int>(o.context), chunk,
                                                                             std::min(o.step_rows_or(32), kernels::kStepRows32),
                                                                             pfx && std::string_view(pfx) == "1")) / G;
    }
  } else if (llama) {
    m.scratch_gib = 0.938 + (C == 1 ? 0.026 : 0.0) + 0.230 * kchunk;
    m.state_gib = 0.199 * ctx_m;
    first_use = 0.10;
  } else {
    m.scratch_gib = 0.759 + (C == 1 ? 0.363 : 0.0) + 0.165 * kchunk;
    m.state_gib = 0.452 * ctx_m;
    if (block) m.weights_gib += 0.757, m.state_gib += 0.0223 * o.slots * half;
    first_use = 0.14;
  }
  const serve::VisionPlan vp = serve::vision_plan(o, cfg, C);
  if (vp.on) staged += vp.max_gib();
  m.scratch_gib += staged;
  // arch flint: the sliding layers' rings, apart from the pool's pages (AttnSet::ring_pages at Model's ring rows)
  double rings = 0;
  if (flint && c.window > 0) {
    int sliding = 0;
    for (const bool sl : c.sliding) sliding += sl;
    const int ring_pages = (c.window + std::max(8, chunk) - 2) / 64 + 2;
    // (the budget's KV bytes are the full-attention layers'; a sliding layer's token takes as much)
    rings = static_cast<double>(sliding) * ring_pages * 64 * (b.kv_bytes_per_token / std::max(1, c.layers - sliding)) * o.slots;
  }
  // the reserve as the server takes it (backend start()): a start without a warm-up keeps the first-use VRAM free too
  const double reserve = o.reserve_gib + (o.warmup ? 0.0 : 1.5);
  if (!o.warmup) m.scratch_gib -= first_use;
  const double room = (m.card_gib - m.weights_gib - m.state_gib - m.scratch_gib) * G;  // for the KV and the reserve
  const int64_t kPage = 64, slot_pages = o.context / kPage;
  const double page = kvb * kPage;
  // the load's own check on its first pool (Model: 1.5 GiB kept for the runtime's first use; what the warm-up and the
  // staged parts take later is still free then)
  const double later = (o.warmup ? first_use : 0.0) + staged + (dense27 && block ? 0.09 : 0.0);  // (a block drafter's load line: 0.09 more)
  const int64_t fit0 = static_cast<int64_t>(std::max(0.0, room + (later - 1.5) * G - rings) / page);
  const int64_t fit = static_cast<int64_t>(std::max(0.0, room - reserve * G - rings) / page);
  int64_t pool = 0;
  if (o.kv_pool_tokens == 0) pool = slot_pages * o.slots;  // fixed per slot
  else if (o.kv_pool_tokens > 0) pool = (o.kv_pool_tokens + kPage - 1) / kPage;
  else pool = std::min<int64_t>(fit, slot_pages * o.slots);  // auto: the VRAM left, at most every slot at the max context
  auto refuse = [&](int64_t have, const char *what) {
    char e[400];
    std::snprintf(e, sizeof e, "the KV of %s needs %.2f GiB per card, about %.2f GiB will be free: a max context of about %lld tokens fits one request (lower XEN_CONTEXT%s)",
                  what, (static_cast<double>(o.kv_pool_tokens < 0 ? slot_pages : pool) * page + rings) / G, (static_cast<double>(have) * page + rings) / G,
                  static_cast<long long>(have * kPage / 1024 * 1024), o.kv_pool_tokens > 0 ? " or XEN_KV_POOL_TOKENS; auto takes what fits" : o.kv_pool_tokens == 0 ? ", or share the KV pool: XEN_KV_POOL_TOKENS auto" : "");
    throw serve::OptionError(e);
  };
  if (o.kv_pool_tokens >= 0 && pool > fit0) refuse(fit0, o.kv_pool_tokens == 0 ? "every slot at the max context" : "the pool");
  if (o.kv_pool_tokens > 0 && pool < slot_pages) throw serve::OptionError("the KV pool (" + std::to_string(pool * kPage) + " tokens) holds less than one max context of " + std::to_string(o.context) + " tokens");
  if (o.kv_pool_tokens < 0 && std::min(fit, fit0) < slot_pages) refuse(std::min(fit, fit0), "one max context");
  m.kv_tokens = pool * kPage, m.context = o.context, m.slots = o.slots;
  m.kv_gib = (static_cast<double>(pool) * page + rings) / G;
  m.scratch_gib += o.reserve_gib;
  m.embed_gib = static_cast<double>(c.vocab) * c.hidden * 2 / G;
  m.prefix_gib = o.prefix_cache ? o.prefix_cache_gib : 0;
  const util::HostMem hm = util::host_mem();
  m.host_total_gib = hm.total / G, m.host_available_gib = hm.available / G;
  m.host_other_gib = util::kLoadHostOverheadGib, m.host_keep_gib = util::kHostKeepGib;
  return serve::plan_json(m);
}

std::string plan(const serve::Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) {
  try {
    return plan_of(opt, cfg, arch_id);
  } catch (const serve::OptionError &) {
    throw;
  } catch (const std::exception &e) {  // a checkpoint or drafter directory the load would refuse too
    throw serve::OptionError(e.what());
  }
}

class Backend final : public serve::Backend {
 public:
  Backend(const serve::Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) : o_(resolve(opt, arch_id)), arch_(arch_id) {
    const bool flint = arch_id == "flint";
    if (o_.vision && arch_id != "qwen35")  // llama / flint checkpoints carry no vision tower
      throw serve::OptionError("--vision 1: arch " + std::string(arch_id) + " has no image input");
    const serve::VisionPlan vp = serve::vision_plan(o_, cfg, o_.tp == 1 ? 1 : 2);
    ModelOptions mo = model_options(o_, flint);
    loader_ = std::make_unique<l0::Loader>(l0::Loader::open());
    fab_ = std::make_unique<Fabric>(*loader_, serve::cards_for(*loader_, o_.tp, o_.card, arch_id, true), false);  // no queue profiling
    mo.loader = loader_.get();
    model_ = std::make_unique<Model>(*fab_, o_.model_dir, mo);
    StepOptions so;
    so.wire_bf16 = o_.wire == "bf16";
    so.wire_i8 = o_.wire == "int8";  // opt-in: a numerics change
    so.w4_items = o_.w4_items;
    so.graphs = o_.graphs;  // arch flint: decode step graphs (--graphs, XEN_GRAPHS)
    // the decode attention numerics, as the prefill names (fast on the int8 KV = hi + lo q codes)
    so.attn_qk = o_.decode_attn.rfind("int8", 0) == 0 ? 1 : o_.decode_attn == "fast" && o_.kv != "bf16" ? 2 : 0;
    so.attn_p_lo = o_.decode_attn == "exact" || o_.decode_attn == "int8";
    so.w4_a8 = o_.decode_w4 == "a8";  // opt-in: a numerics change
    try {
      model_->set_step_options(so);
    } catch (const std::invalid_argument &e) {
      if (!so.w4_a8) throw;
      throw serve::OptionError(std::string("--decode-w4 a8: ") + e.what());
    }
    log_load();
    if (vp.on) {  // the tower after the model (an auto KV pool is sized from the VRAM left after it, start())
      vis_ = serve::load_vision(*fab_, o_, vp, model_->config().vocab);
      model_->set_vision(vis_.get());
      hook_ = serve::make_vision_hook(*vis_, o_);
    }
    adapter_ = std::make_unique<StepModelAdapter>(*model_);
    adapter_->set_wide_from(o_.drafter != 0 && !o_.drafter_model.empty() ? o_.drafter_wide : 0);
    if (model_->drafter()) {
      if (o_.mtp_depth < 1 || o_.mtp_depth > StepModelAdapter::kDepth)
        throw serve::OptionError("--mtp-depth takes 1.." + std::to_string(StepModelAdapter::kDepth) +
                                 " (the MTP depth of one decoding sequence)");
      if (o_.draft_depths.empty() && o_.mtp_depth != StepModelAdapter::kMtpTableDepth) {
        if (model_->dflash()) adapter_->set_mtp_depth_from(o_.mtp_depth, model_->mtp_from());
        else adapter_->set_mtp_depth(o_.mtp_depth);
      }
    }
    std::vector<int> dd;  // XEN_DRAFT_DEPTHS over the built-in table (validated with the options)
    if (serve::parse_draft_depths(o_.draft_depths, dd) && !dd.empty()) adapter_->set_depths(dd);
    if (model_->has_drafter()) {  // the step rows and the depth table in effect
      std::string t;
      const int ns = std::min(8, model_->step_rows());
      for (int n = 1; n <= ns; ++n) t += (n > 1 ? "," : "") + std::to_string(adapter_->draft_depth(n));
      std::fprintf(stderr, "server: steps of up to %d rows, one sequence's up to %d; drafts per sequence at 1..%d sequences %s%s\n", model_->step_rows(),
                   model_->seg_rows_max(), ns, t.c_str(), o_.draft_depths.empty() ? " (the measured table)" : " (XEN_DRAFT_DEPTHS)");
    }
  }

  core::StepModel &step_model() override { return *adapter_; }
  server::MediaHook *media() override { return hook_.get(); }
  std::vector<server::Device> devices() const override { return serve::devices_of(*loader_, *fab_); }
  std::string drafter() const override {  // the checkpoint's MTP layer, a block drafter, or both (--drafter-mtp-from)
    const bool block = o_.drafter != 0 && !o_.drafter_model.empty();
    return serve::drafter_name(block ? o_.drafter_model : std::string(), o_.drafter != 0 && (!block || o_.drafter_mtp_from > 0));
  }
  int64_t vocab() const override { return model_->config().vocab; }
  double prefill_part_ms() const override { return 25; }

  void start(core::BatchEngine &be, const text::Tokenizer &tok, bool warmup) override {
    if (warmup) {
      const double ms = be.warm_up(serve::warm_plan(tok, adapter_->prefill_chunk(), adapter_->capacity(), be.model().prompt_logprobs));
      std::fprintf(stderr, "server: warm-up took %.0f ms; VRAM free %.2f GiB\n", ms, serve::vram_free_gib(*loader_, *fab_));
    }
    if (o_.kv_pool_tokens < 0) {  // the auto pool takes the VRAM left once the kernels ran (the warm-up)
      // without a warm-up the runtime's first-use VRAM is not taken yet: keep it free too
      const double res = o_.reserve_gib + (warmup ? 0.0 : 1.5);
      const int64_t t = model_->kv_fit(res);
      be.kv_refresh();  // stats() and /v1/models report the re-sized pool from the first request on
      std::fprintf(stderr, "server: KV pool sized to the VRAM left: %lld tokens (%s), %.2f GiB reserve; VRAM free %.2f GiB\n", static_cast<long long>(t),
                   gib_per_card().c_str(), res, serve::vram_free_gib(*loader_, *fab_));
    }
    std::fprintf(stderr, "server: KV lending %s\n", model_->kv_lend_note().c_str());  // --kv-lend
  }

 private:
  std::string gib_per_card() const {  // the KV bytes: "a / b GiB per card" (TP2) or "a GiB" (TP1)
    char b[96];
    if (model_->cards() > 1) std::snprintf(b, sizeof b, "%.2f / %.2f GiB per card", model_->kv_bytes(0) / 1073741824.0, model_->kv_bytes(1) / 1073741824.0);
    else std::snprintf(b, sizeof b, "%.2f GiB", model_->kv_bytes(0) / 1073741824.0);
    return b;
  }

  void log_load() {
    Model &m = *model_;
    const std::string head_desc = o_.lm_head + (o_.lm_head == "int4-rescore" ? " (" + std::to_string(o_.lm_head_tiles) + " tiles)" : std::string()) + ", draft rows " + o_.draft_head;
    const std::string tp_desc = o_.tp == 1 ? "TP1 on card " + std::to_string(o_.card) : "TP2, " + o_.wire + " wire";
    std::string drafter = "no drafter";
    if (m.dflash())
      drafter = "DFlash 2 drafter " + m.dflash()->dir() + " (block " + std::to_string(m.dflash()->config().block) + ", draft vocabulary " +
                std::to_string(m.dflash()->draft_vocab()) + (o_.drafter_int4 != 0 ? ", int4 matrices" : "") + ")" +
                (m.seg_rows_max() > kernels::kStepRows8 && o_.drafter_wide > 0
                     ? ", one stream drafts 16 rows (" + (o_.drafter_tree > 0 ? "trees of " + std::to_string(o_.drafter_tree_wide) + " nodes" : std::string("chains")) +
                           ") after " + std::to_string(o_.drafter_wide) + "+ accepted drafts"
                     : std::string()) +
                (m.dual_drafters() ? " + the MTP layer (depth " + std::to_string(o_.mtp_depth) + ") for steps of " +
                                         std::to_string(m.mtp_from()) + "+ sequences"
                                   : std::string());
    else if (m.has_drafter())
      drafter = "MTP drafter (depth " + std::to_string(o_.mtp_depth) + ", draft vocabulary " +
                 std::to_string(m.drafter()->draft_vocab()) + ")";
    const char *what = arch_ == "llama" ? "llama (dense, qwen35 runtime)" : arch_ == "flint" ? "flint (MoE, qwen35 runtime, step graphs)" : "qwen35";
    std::fprintf(stderr, "server: %s model loaded in %.1f s (%s); %d slots x %lld tokens, KV %s, prefill chunk %d (attention %s, W4 %s), lm_head %s, W4 split %d items; %s\n",
                 what, m.load_seconds(), tp_desc.c_str(), m.slots(), static_cast<long long>(m.capacity()), o_.kv.c_str(), m.prefill_chunk(), o_.prefill_attn.c_str(),
                 o_.prefill_w4.c_str(), head_desc.c_str(), o_.w4_items, drafter.c_str());
    serve::log_rope(m.capacity(), m.config().max_positions, m.yarn_factor(),
                    arch_ == "flint" ? "its rotary layers attend within a window, its full-attention layers have no rotary" : nullptr, m.rope_tiers() > 1);
    if (m.rope_tiers() > 1) {
      std::vector<std::pair<int64_t, double>> tiers;
      for (int i = 0; i < m.rope_tiers(); ++i) tiers.emplace_back(m.rope_tier_limit(i), m.rope_tier_factor(i));
      serve::log_rope_tiers(tiers);
    }
    if (o_.decode_attn != "exact")
      std::fprintf(stderr, "server: decode attention numerics %s (%s)\n", o_.decode_attn.c_str(),
                   o_.decode_attn == "int8" ? "int8 q.k on the int8 K codes, P hi + lo"
                   : o_.decode_attn == "int8-fast" ? "int8 q.k on the int8 K codes, P hi only"
                   : o_.kv != "bf16" ? "q.k on int8 hi + lo q codes, P hi only" : "P hi only");
    if (o_.decode_w4 == "a8")
      std::fprintf(stderr, "server: decode int4 GEMMs on int8 activations (--decode-w4 a8: MLP, GDN and attention projections; the drafters and lm_head stay W4A16)\n");
    if (o_.attn_select > 0)
      std::fprintf(stderr, "server: page-selected decode attention: %d pages (%d tokens) per sequence and kv head past that context%s\n", o_.attn_select, o_.attn_select * 64,
                   m.options().select_self > 0 ? "; each sequence measures its heads' counts on its first selected steps" : "");
    if (o_.prefill_select > 0)
      std::fprintf(stderr, "server: page-selected prefill attention: %d pages (%d tokens) per attention row block and kv head past that context; the prompt's KV "
                           "past it is built on the selection\n", o_.prefill_select, o_.prefill_select * 64);
    std::fprintf(stderr, "server: KV %s: %lld tokens (%s); one request up to %lld tokens (max_model_len); VRAM free %.2f GiB before the KV, %.2f after the load\n",
                 m.kv_shared() ? "pool shared by the slots" : "fixed per slot", static_cast<long long>(m.kv_pool_tokens()), gib_per_card().c_str(),
                 static_cast<long long>(m.capacity()), m.vram_free_before_kv(), serve::vram_free_gib(*loader_, *fab_));
    serve::log_prefix_cache(m.snap_capacity(), static_cast<double>(m.snap_bytes(0)), m.cards() > 1 ? static_cast<double>(m.snap_bytes(1)) : 0.0, o_);
  }

  serve::Options o_;
  std::string arch_;
  std::unique_ptr<l0::Loader> loader_;
  std::unique_ptr<Fabric> fab_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<StepModelAdapter> adapter_;
  std::unique_ptr<arch::VisionStore> vis_;  // --vision 1: destroyed before the model and the fabric
  std::unique_ptr<server::MediaHook> hook_;
};

}  // namespace

serve::BackendEntry serve_entry() {
  return {"qwen35", {"qwen35", "llama", "flint"},
          [](const serve::Options &o, const model::ModelConfig &cfg, std::string_view arch_id) -> std::unique_ptr<serve::Backend> {
            return std::make_unique<Backend>(o, cfg, arch_id);
          },
          plan};
}

}  // namespace xenithium::arch::qwen35
