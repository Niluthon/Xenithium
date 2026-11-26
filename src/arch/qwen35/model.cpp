#include <array>
#include "xenithium/arch/qwen35/model.hpp"
#include "xenithium/arch/qwen35/trained_qsa_device.hpp"

#include "xenithium/arch/common/vision.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <sycl/ext/intel/esimd.hpp>

#include "xenithium/arch/flint/arch.hpp"
#include "xenithium/arch/flint/moe.hpp"
#include "xenithium/arch/llama/arch.hpp"
#include "xenithium/kernels/linear_attention/gdn.hpp"
#include "xenithium/kernels/linear_attention/gdn_rows.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/kernels/linear_attention/gdn_scan.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/attention/kv_copy.hpp"
#include "xenithium/kernels/attention/trained_qsa_pages.hpp"
#include "xenithium/kernels/drafter/dflash.hpp"  // df_rows_gather
#include "xenithium/kernels/head/logprob.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/env.hpp"
#include "xenithium/util/host_mem.hpp"

namespace xenithium::arch::qwen35 {

namespace k = kernels;

struct Model::Card {
  std::vector<const bf16 *> in_norm, post_norm;  // [layer]
  std::vector<k::GdnState> gdn;                  // [layer] (GDN layers)
  std::vector<k::GqaCache> kv;                   // [layer] (attention layers)
  k::GdnScratch gsc;
  k::GqaScratch asc;
  k::PairWire wire;  // the mixers' all-reduce (GDN and attention share it, parity per call)
  bf16 *mix = nullptr, *rope = nullptr;
  bf16 *pf_rope = nullptr;  // set_vision: a chunk's rope rows [prefill_chunk][rot]
  bf16 *pf_hx = nullptr;  // [kPfMergeMax][hidden] a merged call's segments' last rows after the final norm
  int32_t *accept = nullptr, *pos = nullptr;
  int32_t *seg = nullptr;  // the step's segment records [kSegs16][kSegRec] (device)
  // select_self: the score keys of every attention layer's last selected step, [attention layer][kSegs16]
  // [kv head][kGqaSelPagesMax] (a region per layer, so they can be read after the step), and the pages each (slot,
  // kv head) keeps, [attention layer][slot][kv head] (GqaStep::select_slots)
  uint32_t *sel_keys = nullptr;
  int32_t *sel_slots = nullptr;
  float *tree_S = nullptr;  // the GDN tree core's saved states (k::gdn_tree_S_elems; with DFlash)
  bf16 *tree_rows = nullptr;  // [kTreeRows][hidden] scratch (accept_tree's re-laid head rows)
  // prefill chunk buffers: [prefill_chunk + 4][hidden] bf16 each (4 rows of padding: the decode kernels
  // of a short last slice read and write 4 rows), and one position / accept word per 4-row slice
  bf16 *pf_res = nullptr, *pf_xn = nullptr, *pf_mix = nullptr, *pf_mlp = nullptr;
  int8_t *pf_xq = nullptr;  // pf_xn's W4A8 codes (rms_norm_q8) + scales, rows pf_xq_ld
  float *pf_xs = nullptr;
  float *pf_xs_rows = nullptr;  // the same scales row-major [rows][H / 128] (the oneDNN W4A8 GEMMs' layout)
  int pf_xq_ld = 0;
  int32_t *pf_pos = nullptr, *pf_acc = nullptr;
  k::GdnRowsScratch pf_gdn;  // many-row GDN
  k::GqaRowsScratch pf_gqa;  // many-row attention
  uint8_t *pf_arena = nullptr;  // pf_gdn's / pf_gqa's (not q) / the MLP's h chunk scratch, shared
  k::PairRowsWire pf_wire;   // the many-row mixers' two-card sum
  k::PairSplit pf_split;     // the split sum: recv + flags on this card
  std::optional<sycl::queue> side;  // the split's copies beside the compute queue (beta)
  int32_t *pf_ids = nullptr;        // the chunk's token ids for one embedding gather (pinned host)
  int32_t *ring_table = nullptr;    // arch flint: the sliding layers' ring pages, [slots][ring pages] identity
  std::vector<void *> mem;
};

namespace {
constexpr int kRecInts = k::kSegs16 * k::kSegRec;  // one step's segment records
constexpr int kMtpPasses = Model::kMtpDepthMax + 1;  // pass 1 + up to kMtpDepthMax chained passes

// slot s of a per-slot state / cache as a one-sequence view (prefill, the slices scaffold)
k::GdnState gdn_view(const k::GdnState &st, int s) {
  k::GdnState v = st;
  const size_t z = static_cast<size_t>(s);
  v.S += z * st.slot_S;
  for (int p = 0; p < 2; ++p) {
    v.win[p] += z * st.slot_win;
    v.y[p] += z * st.slot_y;
    v.ab[p] += z * st.slot_ab;
  }
  v.seg = nullptr;
  v.seg_rows = 0;
  v.slot_S = v.slot_win = v.slot_y = v.slot_ab = 0;
  return v;
}
k::GqaCache kv_view(const k::GqaCache &c, int s) {
  k::GqaCache v = c;
  v.pages += static_cast<size_t>(s) * c.slot_pages;
  v.slot_pages = 0;
  return v;
}

double ev_us(const sycl::event &e) {
  const auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(b - a) * 1e-3;
}
}  // namespace

// Arch llama's many-row prefill: the attention bodies built at head_dim 128 are the decode group kernel per 64-row batch
// (prefill_attn 0) and the fa body (3; the block / staged bodies 1 / 2 run it instead), int8 q.k as hi + lo codes only
// (qk 1 -> 2); the W4 GEMMs take the zero-point matrices in their dequantised mode (w4a16_rows.cpp: W4A8 does not take
// zero points). Applied over whatever the caller asked.
// Arch flint: the many-row prefill on the affine W4 GEMMs (w4aff_gemm_pf), the many-row MoE
// (kernels::moe_aff_rows) and the fa attention body with the window, the KV ring and the sinks (prefill_attn 0 = the
// decode attention per 64-row batch); int8 q.k as hi + lo codes with P hi only (qk 1 -> 2), no W4A8 (the checkpoint is
// affine W4A16). Applied over whatever the caller asked; prefill_slices stays the caller's (the bit-identical scaffold).
void Model::flint_options(ModelOptions &o) {
  if (o.prefill_attn != k::kGqaAttnBatch) o.prefill_attn = k::kGqaAttnFa;
  if (o.prefill_attn_qk == 1) o.prefill_attn_qk = 2;  // int8 q.k as hi + lo codes only (as llama)
  if (o.prefill_attn_qk == 2 && o.kv == k::QsaKv::bf16) o.prefill_attn_qk = 0;
  if (o.prefill_attn_qk == 2) o.prefill_attn_p_lo = false;  // qk 2 runs with P hi only (gqa_attn_fa)
  o.prefill_w4_a8 = false;
  k::w4a16_pf_set_a8(false);
}

void Model::llama_prefill(ModelOptions &o) {
  if (o.prefill_attn != k::kGqaAttnBatch) o.prefill_attn = k::kGqaAttnFa;
  if (o.prefill_attn_qk == 1) o.prefill_attn_qk = 2;
  if (o.prefill_attn_qk == 2) o.prefill_attn_p_lo = false;  // qk 2 runs with P hi only (gqa_attn_fa)
  o.prefill_w4_a8 = false;
  k::w4a16_pf_set_a8(false);
}

// The W4A8 body that auto picks: on one card (no two-card sums, so no sends hide the consumer GEMMs) the direct body;
// at TP2 kernel body 3 (the SLM-free body on calls of <= 128 rows, where no sends run, the SLM body above).
constexpr int kA8BodyTp1 = 1, kA8BodyTp2 = 3;
// The smallest attention row block the fa body runs at 6 q heads per kv head (prefill_attn_rows 16; the default is
// 40): the page-set scratch of a page-selected prefill holds a chunk in blocks of this size.
constexpr int kPfSelectBlockRows = 16;
int Model::a8_body_for(int body) const {
  if (body >= 0) return body;
  return fab_.size() == 1 ? kA8BodyTp1 : kA8BodyTp2;
}

Model::Model(Fabric &fab, const std::string &dir, const ModelOptions &opt) : fab_(fab), opt_(opt) {
  check_attn_qk(opt);  // 
  k::w4a16_pf_set_dequant(opt_.prefill_w4_dequant);  // the many-row W4 GEMMs' numerics (process-wide)
  k::w4a16_pf_set_a8(opt_.prefill_w4_a8);  // W4A8 prefill (process-wide)
  k::w4a16_pf_set_a8_body(a8_body_for(opt_.prefill_a8_body));  // (fab_ is set: the card count is known)
  k::w4a16_pf_set_short_rows(opt_.prefill_short);
  const auto t0 = std::chrono::steady_clock::now();
  const auto mc = model::ModelConfig::load(dir);
  const auto arch = model::open_arch(mc);
  // arch llama (Nanbeige4.3): the same dense decoder with Llama semantics (Config flags: every layer full attention,
  // head_dim 128 without gate or q / k norms, plain RMSNorm, int4 g32 with zero points)
  const bool llama = arch->id() == "llama", flint = arch->id() == "flint";
  if (arch->id() != "qwen35" && !llama && !flint) throw std::runtime_error("qwen35 model: " + dir + " is arch " + std::string(arch->id()));
  cfg_ = llama ? arch::llama::parse_config(mc) : flint ? arch::flint::runtime_config(mc) : Config::parse(mc);
  // arch flint takes a DFlash 2 drafter (Flint-DFlash2); neither arch has an MTP layer
  if (llama && (opt.mtp || !opt.dflash_dir.empty())) throw std::invalid_argument("qwen35 model: arch llama has no drafter");
  if (flint && (opt.dflash_dir.empty() ? opt.mtp : opt.mtp_from > 0))
    throw std::invalid_argument("qwen35 model: arch flint has no MTP layer (its drafter is a DFlash 2 checkpoint, mtp_from 0)");
  if (flint && opt.dflash_tree > 0)  // the tree verify needs the GDN tree core's scratch and a KV move the rings lack
    throw std::invalid_argument("qwen35 model: arch flint drafts chains (draft trees: dflash_tree 0)");
  if (llama) llama_prefill(opt_);
  if (flint) flint_options(opt_);
  const int C = static_cast<int>(fab.size());
  if (C < 1 || C > 2) throw std::runtime_error("qwen35 model: runs on one card (TP1) or two (TP2)");
  cap_ = (std::max<int64_t>(opt.context, 64) + 63) / 64 * 64;
  if (cap_ > k::kQsaMaxTokens) throw std::runtime_error("qwen35 model: context above the attention kernels' 1,048,576 tokens");
  // 16-row steps for qwen35; llama / flint keep 8 (zero-point / affine W4, the flint MoE). 32-row steps on request
  // (XEN_STEP_ROWS 32, the server's default); the slots stay <= 16 (kSegs16).
  step_rows_ = llama || flint ? k::kStepRows8 : k::kStepRows16;
  const int step_max = llama || flint ? k::kStepRows8 : k::kStepRows32;
  if (opt.step_rows != 0) {  // XEN_STEP_ROWS
    if (opt.step_rows != k::kStepRows8 && opt.step_rows != k::kStepRows16 && opt.step_rows != k::kStepRows32)
      throw std::invalid_argument("qwen35 model: step rows 8, 16 or 32 (arch " + std::string(arch->id()) + ")");
    step_rows_ = std::min(opt.step_rows, step_max);  // llama / flint run 8 at any setting (the server's default is 32)
  }
  if (opt.slots < 1 || opt.slots > std::min(step_rows_, k::kSegs16))
    throw std::invalid_argument("qwen35 model: slots 1.." + std::to_string(std::min(step_rows_, k::kSegs16)) + " (arch " + std::string(arch->id()) + ")");
  if (opt.prefill_short < 0 || opt.prefill_short > k::kStepRows32) throw std::invalid_argument("qwen35 model: prefill_short 0..32 rows");
  rows_cap_ = std::max(pass_rows(), opt.prefill_short);  // the step buffers' rows
  const size_t NS = static_cast<size_t>(opt.slots);
  std::vector<trained_qsa::Layer> qsa_bank;
  std::string qsa_layers;
  const char *trained_dir=std::getenv("XEN_TRAINED_QSA_DELTA_DIR");
  if(trained_dir && *trained_dir) {
    // The selector's buffers depend on the slots and the context, not on the pool: an auto pool (kv_pool_tokens < 0)
    // is sized after them by kv_fit, from the VRAM they leave.
    if (llama || flint || opt_.kv!=k::QsaKv::i8v4 || opt_.attn_select!=0 ||
        opt_.kv_pool_tokens==0 || opt_.dflash_dir.empty() || cap_<=2048 || cap_>trained_qsa::kMaxContext)
      throw std::invalid_argument("trained QSA requires Swift, a shared int8-v4 pool (a token count or auto), context 2049..262144, a drafter and no other selector");
    const auto bank=model::Checkpoint::open(trained_dir,false);
    trained_qsa_layer_.assign(cfg_.layers,-1);
    for (int l=0;l<cfg_.layers;++l) {
      const auto prefix=cfg_.prefix+"layers."+std::to_string(l)+".self_attn._qsa_indexer.";
      if (std::none_of(bank.tensors().begin(),bank.tensors().end(),[&](const auto &t) { return t.name.starts_with(prefix); })) continue;
      trained_qsa_layer_[l]=static_cast<int>(qsa_bank.size());
      qsa_bank.push_back(trained_qsa::Layer::load(bank,cfg_,l,trained_qsa::NormStorage::delta));
      qsa_layers+=(qsa_layers.empty()?"":",")+std::to_string(l);
    }
    if (qsa_bank.empty() || bank.tensors().size()!=5*qsa_bank.size())
      throw std::invalid_argument("trained QSA: no matching bank or unexpected tensors in the bank");
  }
  const char *pf_xmx=std::getenv("XEN_TRAINED_QSA_PREFILL_XMX");
  const char *pf_min=std::getenv("XEN_TRAINED_QSA_PREFILL_MIN_POS");
  const char *pf_four=std::getenv("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG");
  if(pf_four && std::string_view(pf_four)!="0" && std::string_view(pf_four)!="1")
    throw std::invalid_argument("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG: expected 0 or 1");
  const bool allow_four=pf_four && std::string_view(pf_four)=="1";
  const char *pf_merged=std::getenv("XEN_TRAINED_QSA_PREFILL_MERGED");
  if(pf_merged && std::string_view(pf_merged)!="0" && std::string_view(pf_merged)!="1")
    throw std::invalid_argument("XEN_TRAINED_QSA_PREFILL_MERGED: expected 0 or 1");
  if(pf_merged && std::string_view(pf_merged)=="1" && !allow_four)
    throw std::invalid_argument("merged trained prefill requires the four-slot XMX prefill opt-in");
  if(allow_four && (opt_.slots!=4 || !pf_xmx || std::string_view(pf_xmx)!="1"))
    throw std::invalid_argument("four-slot prefill screen requires four configured slots and XMX prefill");
  if (pf_xmx && std::string_view(pf_xmx)!="0" && std::string_view(pf_xmx)!="1")
    throw std::invalid_argument("XEN_TRAINED_QSA_PREFILL_XMX: expected 0 or 1");
  if (pf_xmx && std::string_view(pf_xmx)=="1") {
    const std::string_view min=pf_min ? pf_min : "32768";
    const auto [end, ec]=std::from_chars(min.data(),min.data()+min.size(),trained_qsa_prefill_min_);
    if (ec!=std::errc{} || end!=min.data()+min.size() || trained_qsa_prefill_min_<2112 ||
        trained_qsa_prefill_min_>=cap_ || trained_qsa_prefill_min_%k::kQsaPage)
      throw std::invalid_argument("XEN_TRAINED_QSA_PREFILL_MIN_POS: expected page-aligned position in [2112, context)");
    if (qsa_bank.size()!=16 || (opt_.slots!=1 && !allow_four) || opt_.prefill_slices || opt_.prefill_attn_slices ||
        opt_.prefill_attn!=k::kGqaAttnFa || opt_.prefill_attn_qk!=1 || opt_.prefill_attn_p_lo)
      throw std::invalid_argument("trained XMX prefill requires a qualified slot layout, the all-layer bank and native K8V4 int8-fast FA");
  } else if (pf_min && *pf_min) {
    throw std::invalid_argument("XEN_TRAINED_QSA_PREFILL_MIN_POS requires XEN_TRAINED_QSA_PREFILL_XMX=1");
  }
  const model::Checkpoint ck = model::Checkpoint::open(dir);
  const bool head_only_int4 = !ck.find("lm_head.weight") && ck.find("lm_head.qweight");
  if (flint && (opt.head_w4 || opt.head_rescore > 0)) throw std::invalid_argument("qwen35 model: arch flint's lm_head is the checkpoint's MLX int4 (no lm_head switch)");
  if (head_only_int4 && !opt.head_w4)
    throw std::runtime_error("qwen35 model: " + dir + " has an int4-only lm_head (no bf16 rows); serve it with --lm-head int4");
  if (head_only_int4 && opt.head_rescore > 0)
    throw std::runtime_error("qwen35 model: int4-rescore needs the bf16 lm_head rows; " + dir + " has an int4-only head (use --lm-head int4)");
  dense_ = std::make_unique<load::DenseStore>(fab);
  auto plan = arch->dense_plan(ck, model::DeployParams{});
  if (opt.head_rescore < 0 || opt.head_rescore > k::kHeadTilesMax) throw std::invalid_argument("qwen35 model: head_rescore 0..32 tiles");
  if (opt.head_rescore > 0 && !opt.head_w4) throw std::invalid_argument("qwen35 model: head_rescore needs head_w4 (the int4 rows score)");
  const bool draft_w4 = opt.draft_head < 0 ? opt.head_w4 : opt.draft_head == 1;
  if ((opt.head_w4 || draft_w4) && !ck.find("lm_head.qweight"))
    throw std::runtime_error("qwen35 model: an int4 lm_head but " + dir + " has no lm_head.qweight (xenithium-bake head4-calib)");
  if (opt.head_w4) {  // the int4 lm_head (stored under the bf16 head's name)
    std::vector<model::DenseSpec> extra;
    for (auto &sp : plan)
      if (sp.key == "lm_head.weight" || sp.name == "lm_head.weight") {
        if (opt.head_rescore > 0) {  // the int8 rows too, under their own key (the two-stage head's rescore)
          model::DenseSpec i8 = sp;
          i8.key = kHeadI8Key;
          i8.store = model::DenseSpec::Store::i8;
          extra.push_back(i8);
        }
        sp.name = "lm_head.qweight";
        sp.key = "lm_head.weight";
        sp.store = model::DenseSpec::Store::w4;
        sp.w4_layout = model::DenseSpec::W4Layout::gptq;
      }
    plan.insert(plan.end(), extra.begin(), extra.end());
  }
  dense_->load(ck, plan, opt.threads);
  if (!opt.lora_dir.empty()) {  // the PEFT LoRA adapter (load/lora.hpp): the per-(card, layer) A / B deltas
    load::LoraGeom lg;
    lg.cards = C;
    lg.layers = cfg_.layers;
    lg.hidden = cfg_.hidden;
    lg.q_heads = cfg_.q_heads;
    lg.kv_heads = cfg_.kv_heads;
    lg.head_dim = cfg_.head_dim;
    lg.attn_output_gate = cfg_.attn_output_gate;
    lg.inter = cfg_.inter;
    lg.full_attention = cfg_.full_attention;
    lg.prefix = cfg_.prefix;
    lora_ = std::make_unique<load::LoraAdapter>(fab, opt.lora_dir, lg, opt.threads);
  }
  head_ = std::make_unique<Head>(fab, *dense_, cfg_, opt.head_rescore > 0, rows_cap_);
  cands_ = std::make_unique<HeadCands>(fab, head_->vocab_per_card(), rows_cap_);  // device scratch before any VRAM fit
  if (opt.head_rescore > 0) head_->set_mode(HeadMode::rescore, opt.head_rescore);
  mlp_ = std::make_unique<Mlp>(fab, cfg_, rows_cap_);
  // (its many-row scratch after the cards' stage arenas, h inside them)
  if (cfg_.flint) moe_ = std::make_unique<arch::flint::MoeSet>(fab, ck, *dense_, cfg_, opt.threads);
  if (cfg_.flint && !opt_.prefill_slices) {  // the many-row MoE's scratch and the affine GEMMs' row sums
    moe_->reserve_rows(opt.prefill_chunk);
    const int kmax = std::max({cfg_.hidden, cfg_.inter / C, cfg_.q_heads * cfg_.head_dim / C});
    for (int c = 0; c < C; ++c) k::w4a16_pf_aff_reserve(fab.compute(static_cast<size_t>(c)), opt.prefill_chunk, kmax);
  }
  if (opt_.prefill_w4_a8)  // the W4A8 prefill's activation scratch (set_prefill_options reserves it late)
    for (int c = 0; c < C; ++c) k::w4a16_pf_a8_reserve(fab.compute(static_cast<size_t>(c)), opt.prefill_chunk, std::max(cfg_.hidden, cfg_.inter));
  gdn_ = GdnSet::from(*dense_, cfg_, C);
  if (opt.kv == k::QsaKv::i8v4 && (cfg_.flint || cfg_.llama() || cfg_.head_dim != 256 || cfg_.q_heads / cfg_.kv_heads != 6))
    throw std::invalid_argument("qwen35 model: int8-v4 KV (4-bit V) is built for the dense 27B's attention (6 q heads per kv head, head_dim 256)");
  attn_ = AttnSet::from(*dense_, cfg_, C, opt.kv);
  apply_a8_keep(opt_.prefill_a8_keep);  // 
  const int H = cfg_.hidden, L = cfg_.layers;
  // with DFlash a verify carries up to 8 rows of one sequence: the GDN's pending logs hold 8 rows
  rows_max_ = opt.dflash_dir.empty() && !opt.rows8 ? kRows : k::kGdnMaxRows;  // rows8
  // 16-row verifies of one sequence (seg_rows 16), a drafter trained on a block of 16 or more
  if (opt.seg_rows != 0 && opt.seg_rows != k::kGdnMaxRows && opt.seg_rows != k::kGdnWideRows) throw std::invalid_argument("qwen35 model: segment rows 8 or 16");
  if (opt.seg_rows == k::kGdnWideRows) {
    if (opt.dflash_dir.empty() || DFlashConfig::load(opt.dflash_dir, k::kStepRows16).block < k::kGdnWideRows)
      throw std::invalid_argument("qwen35 model: 16-row segments (--seg-rows 16, the default) need a drafter trained on a 16-token block (DSpine b16); an 8-block drafter (DFlash 2) runs --seg-rows 8 (XEN_SEG_ROWS)");
    if (step_rows_ < k::kGdnWideRows) throw std::invalid_argument("qwen35 model: 16-row segments need 16-row steps");
    rows_max_ = k::kGdnWideRows;
  }
  // Past the trained positions: static YaRN at the smallest factor (steps of 0.01) reaching the context, for every
  // request (the same table for the MTP drafter). Arch flint keeps the plain table: its rotary layers attend within a
  // 513-token window (relative distances stay inside the trained ones) and its full-attention layers have no rotary.
  const k::QsaYarn yarn = flint ? k::QsaYarn{} : k::qsa_yarn_for(cap_, cfg_.max_positions);
  yarn_factor_ = yarn.factor;
  auto rope = k::qsa_rope_table(cap_, attn_.shape.rot, cfg_.rope_theta, yarn);
  // the tiers below the context's factor follow its table (the last tier starts at row 0: a sequence there has the bits
  // of one factor for all); a sequence's rows read row P + delta, the delta from its tier (the M-RoPE delta word: with
  // plain rope too, arch llama; a text row's composed row is the table's row P + delta)
  const bool mrope_cfg = cfg_.mrope_section.size() == 3 && cfg_.mrope_interleaved;
  if (opt_.yarn_tiers && yarn.factor > 1.0 && !opt_.prefill_slices && !opt_.prefill_attn_slices && cfg_.max_positions > 0) {
    for (int64_t Lt = cfg_.max_positions; Lt < cap_; Lt *= 2) {
      const k::QsaYarn yt = k::qsa_yarn_for(Lt, cfg_.max_positions);
      const auto t = k::qsa_rope_table(Lt, attn_.shape.rot, cfg_.rope_theta, yt);
      tier_base_.push_back(static_cast<int64_t>(rope.size() / static_cast<size_t>(attn_.shape.rot)));
      tier_limit_.push_back(Lt);
      tier_factor_.push_back(yt.factor);
      rope.insert(rope.end(), t.begin(), t.end());
    }
    if (static_cast<int64_t>(rope.size() / static_cast<size_t>(attn_.shape.rot)) > int64_t{2147483647}) throw std::invalid_argument("qwen35 model: YaRN tiers past int32 rows");
  }
  tier_base_.push_back(0);
  tier_limit_.push_back(cap_);
  tier_factor_.push_back(yarn.factor);
  if (mrope_cfg || tier_limit_.size() > 1) rope_host_ = rope;  // M-RoPE rows (set_vision) and the tiers' rows
  mrope_cfg_ = mrope_cfg;
  for (int c = 0; c < C; ++c) {
    auto cd = std::make_unique<Card>();
    auto &q = fab.compute(static_cast<size_t>(c));
    auto dev = [&](size_t bytes) {
      void *p = fab.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096));
      cd->mem.push_back(p);
      return p;
    };
    const std::string P = cfg_.prefix;
    for (int l = 0; l < L; ++l) {
      const std::string Lp = P + "layers." + std::to_string(l) + ".";
      cd->in_norm.push_back(reinterpret_cast<const bf16 *>(dense_->at(c, Lp + "input_layernorm.weight").ptr));
      cd->post_norm.push_back(reinterpret_cast<const bf16 *>(dense_->at(c, Lp + "post_attention_layernorm.weight").ptr));
    }
    const auto gs = gdn_.shape_of(c);
    const auto as = attn_.shape_of(c);
    cd->gdn.resize(static_cast<size_t>(L));
    cd->kv.resize(static_cast<size_t>(L));
    cd->accept = static_cast<int32_t *>(dev(64));
    cd->pos = static_cast<int32_t *>(dev(64));
    cd->seg = static_cast<int32_t *>(dev(kRecInts * 4));
    q.memset(cd->seg, 0, kRecInts * 4);
    // per slot: slot s at s x the slot_* strides; decode reads the step's segment records
    for (int l : gdn_.layers) {
      k::GdnState &st = cd->gdn[l];
      st.slot_S = k::gdn_S_elems(gs);
      st.slot_win = k::gdn_win_elems(gs);
      st.slot_y = static_cast<size_t>(rows_max_) * gdn_.w[c][l].qkvz.n;
      st.slot_ab = static_cast<size_t>(rows_max_) * gs.ab_n();
      st.log_rows = rows_max_;
      st.S = static_cast<float *>(dev(NS * st.slot_S * 4));
      for (int p = 0; p < 2; ++p) {
        st.win[p] = static_cast<k::gdn_bf16 *>(dev(NS * st.slot_win * 2));
        st.y[p] = static_cast<float *>(dev(NS * st.slot_y * 4));
        st.ab[p] = static_cast<float *>(dev(NS * st.slot_ab * 4));
      }
      st.accept = cd->accept;
      st.seg = cd->seg;
    }
    if (rows_max_ >= k::kGdnMaxRows && !gdn_.layers.empty()) {  // the tree steps' scratch (8- / 16-row verifies)
      cd->tree_S = static_cast<float *>(dev(k::gdn_tree_S_elems(gs) * 4));
      cd->tree_rows = static_cast<bf16 *>(dev(static_cast<size_t>(rows_cap_) * H * 2));  // every row of a step
    }
    if (!gdn_.layers.empty())  // (arch llama: no GDN layer)
      cd->gsc.y = static_cast<float *>(dev(static_cast<size_t>(rows_cap_) * gdn_.w[c][gdn_.layers.front()].qkvz.n * 4));
    // the attention layers' KV: after the drafter, sized by the page pool
    cd->gsc.core = static_cast<k::gdn_bf16 *>(dev(std::max<size_t>(1, static_cast<size_t>(rows_cap_) * gs.val_w() * 2)));
    cd->gsc.partial = static_cast<float *>(dev(static_cast<size_t>(rows_cap_) * H * 4));
    cd->asc.y = static_cast<float *>(dev(static_cast<size_t>(rows_cap_) * as.qkv_n() * 4));
    cd->asc.q = static_cast<bf16 *>(dev(static_cast<size_t>(rows_cap_) * as.hq() * k::kQsaDim * 2));
    cd->asc.part = static_cast<float *>(dev(k::gqa_part_floats(as, rows_cap_) * 4));
    cd->asc.attn = static_cast<bf16 *>(dev(static_cast<size_t>(rows_cap_) * as.o_k() * 2));
    cd->asc.partial = static_cast<float *>(dev(static_cast<size_t>(rows_cap_) * H * 4));
    if (opt.attn_select > 0) {  // the page selection's scratch
      cd->asc.sel_key = static_cast<uint32_t *>(dev(static_cast<size_t>(k::kSegs16) * as.hkv() * k::kGqaSelPagesMax * 4));
      cd->asc.sel = static_cast<int32_t *>(dev(static_cast<size_t>(k::kSegs16) * as.hkv() * k::kGqaSelMax * 4));
      cd->asc.sel_n = static_cast<int32_t *>(dev(static_cast<size_t>(k::kSegs16) * as.hkv() * 4));
      if (opt.select_self > 0) {
        const size_t attn_layers = static_cast<size_t>(std::count(cfg_.full_attention.begin(), cfg_.full_attention.end(), true));
        cd->sel_keys = static_cast<uint32_t *>(dev(attn_layers * k::kSegs16 * as.hkv() * k::kGqaSelPagesMax * 4));
        cd->sel_slots = static_cast<int32_t *>(dev(attn_layers * NS * as.hkv() * 4));
      }
    }
    cd->mix = static_cast<bf16 *>(dev(static_cast<size_t>(rows_cap_) * H * 2));
    cd->rope = static_cast<bf16 *>(dev(rope.size() * 2));
    q.memcpy(cd->rope, rope.data(), rope.size() * 2);
    // dead rows of GEMM inputs stay finite
    if (gs.val_w() > 0) q.memset(cd->gsc.core, 0, static_cast<size_t>(rows_cap_) * gs.val_w() * 2);
    q.memset(cd->asc.attn, 0, static_cast<size_t>(rows_cap_) * as.o_k() * 2);
    q.memset(cd->mix, 0, static_cast<size_t>(rows_cap_) * H * 2);
    {
      if (opt.prefill_chunk < 4 || opt.prefill_chunk % 4) throw std::invalid_argument("qwen35 model: prefill_chunk is a multiple of 4");
      const size_t rows = static_cast<size_t>(opt.prefill_chunk) + 4, b = rows * H * 2;
      for (bf16 **pp : {&cd->pf_res, &cd->pf_xn, &cd->pf_mix, &cd->pf_mlp}) {
        *pp = static_cast<bf16 *>(dev(b));
        q.memset(*pp, 0, b);
      }
      {  // the norms write pf_xn's W4A8 codes when W4A8 is on (any time: set_prefill_options)
        cd->pf_xq_ld = static_cast<int>((rows + 31) / 32 * 32);
        cd->pf_xq = static_cast<int8_t *>(dev(rows * H));
        const size_t ns = static_cast<size_t>(H / 128) * cd->pf_xq_ld * 4;
        cd->pf_xs = static_cast<float *>(dev(ns));
        q.memset(cd->pf_xs, 0, ns);
        cd->pf_xs_rows = static_cast<float *>(dev(std::max<size_t>(4096, static_cast<size_t>(rows) * (H / 128) * 4)));
      }
      cd->pf_pos = static_cast<int32_t *>(dev(static_cast<size_t>(opt.prefill_chunk / 4) * 4));
      cd->pf_acc = static_cast<int32_t *>(dev(static_cast<size_t>(opt.prefill_chunk / 4) * 4));
      if (!opt_.prefill_slices) {
        const int R = opt.prefill_chunk;
        // One arena per card for the stages' chunk scratch. A layer runs
        // either the GDN or the attention, then the MLP, all on the card's in-order compute queue, and none of these
        // buffers is read outside its stage (the side queue reads only the partials and the split's own buffers), so the
        // GDN set, the attention set (but q: its rows past a short call must stay finite) and the MLP's h share one
        // allocation: the largest set's bytes instead of their sum. The GDN and attention partials ([R][hidden] fp32
        // both) are one buffer too: consecutive GDN layers rewrite one partial, an attention layer between them
        // is the same order.
        auto al = [](size_t b) { return (b + 4095) / 4096 * 4096; };
        k::GdnRowsSizes z{};
        k::GdnScanSizes zs{};
        size_t gdn_b = 0;
        if (!gdn_.layers.empty()) {
          z = k::gdn_rows_sizes(gs, gdn_.w[c][gdn_.layers.front()].qkvz.n, R);
          zs = k::gdn_scan_sizes(gs, R);
          gdn_b = al(z.yq * 4) + al(z.ab * 4) + al(z.cv * 4) + 2 * al(z.qk * 4) + al(z.gate * 4) + al(z.lgate * 4) + 4 * al(zs.wk * 2) +
                  al(zs.mt * 2) + al(zs.uv * 4) + al(zs.gc * 4) + 2 * al(z.o * 2);
        }
        const int qkvn = attn_.w[c][attn_.layers.front()].qkv.n;
        const size_t ay = static_cast<size_t>(R) * qkvn * 4, apart = k::gqa_part_floats(as, k::kGqaRowsBatch) * 4,
                     aattn = static_cast<size_t>(R) * as.o_k() * 2;
        const size_t attn_b = al(ay) + al(apart) + al(aattn), mlp_b = al(static_cast<size_t>(R) * (cfg_.inter / C) * 2);
        auto *arena = static_cast<uint8_t *>(dev(std::max({gdn_b, attn_b, mlp_b})));
        cd->pf_arena = arena;
        size_t off = 0;
        auto carve = [&](size_t b) {
          void *p = arena + off;
          off += al(b);
          return p;
        };
        auto *partial = static_cast<float *>(dev(static_cast<size_t>(R) * H * 4));
        if (!gdn_.layers.empty()) {  // the many-row GDN's scratch
          auto &g = cd->pf_gdn;
          g.yq = static_cast<float *>(carve(z.yq * 4));
          g.ab = static_cast<float *>(carve(z.ab * 4));
          g.cv = static_cast<float *>(carve(z.cv * 4));
          g.qn = static_cast<float *>(carve(z.qk * 4));
          g.kn = static_cast<float *>(carve(z.qk * 4));
          g.gate = static_cast<float *>(carve(z.gate * 4));
          g.lgate = static_cast<float *>(carve(z.lgate * 4));
          g.scan.wk = static_cast<bf16 *>(carve(zs.wk * 2));
          g.scan.qg = static_cast<bf16 *>(carve(zs.wk * 2));
          g.scan.kt = static_cast<bf16 *>(carve(zs.wk * 2));
          g.scan.kt_lo = static_cast<bf16 *>(carve(zs.wk * 2));
          g.scan.mt = static_cast<bf16 *>(carve(zs.mt * 2));
          g.scan.uv = static_cast<float *>(carve(zs.uv * 4));
          g.scan.gc = static_cast<float *>(carve(zs.gc * 4));
          g.o = static_cast<bf16 *>(carve(z.o * 2));
          g.y = static_cast<bf16 *>(carve(z.o * 2));
          g.partial = partial;
        }
        auto &a = cd->pf_gqa;
        off = 0;
        a.y = static_cast<float *>(carve(ay));
        a.part = static_cast<float *>(carve(apart));
        a.attn = static_cast<bf16 *>(carve(aattn));
        a.q = static_cast<bf16 *>(dev(static_cast<size_t>(R) * as.hq() * k::kQsaDim * 2));
        q.memset(a.q, 0, static_cast<size_t>(R) * as.hq() * k::kQsaDim * 2);  // rows past a short chunk stay finite
        a.partial = partial;
        if (opt_.prefill_attn == k::kGqaAttnSdpa)  // oneDNN's attention, one kv head's context in f16 + q / o
          k::gqa_sdpa_scratch_bind(as, R, static_cast<int>(cap_), dev(k::gqa_sdpa_scratch_bytes(as, R, static_cast<int>(cap_))), a);
        if (opt_.prefill_select > 0)  // the row blocks' page sets, sized for the smallest row block
          k::gqa_page_sets_bind(as, R, kPfSelectBlockRows, static_cast<int>(cap_), dev(k::gqa_page_sets_bytes(as, R, kPfSelectBlockRows, static_cast<int>(cap_))), a.sel);
        if (C > 1) {  // the two-card sums: split sum (recv, flags, side queue) and the chunk mailbox
          cd->pf_split.recv = static_cast<float *>(dev(k::pair_split_recv_elems(R, H) * 4));
          cd->pf_split.stage = static_cast<bf16 *>(dev(k::pair_split_recv_elems(R, H) * 2));  // the bf16 wire's stage
          cd->pf_split.bf16 = opt_.prefill_wire_bf16;
          cd->pf_split.flags = static_cast<uint32_t *>(fab.alloc_mailbox(static_cast<size_t>(c), k::kPairSplitFlagBytes));
          cd->mem.push_back(cd->pf_split.flags);
          q.memset(cd->pf_split.flags, 0, k::kPairSplitFlagBytes);
          cd->pf_split.rows_max = R;
          cd->pf_split.n_max = H;
          {
            auto *tab = static_cast<uint32_t *>(dev(k::kPairSplitSeqTabBytes));
            k::pair_split_seq_tab_init(q, tab);
            cd->pf_split.seq_tab = tab;
          }
          cd->side.emplace(fab.card_context(static_cast<size_t>(c)), fab.device(static_cast<size_t>(c)),
                           Fabric::queue_props(fab.profiling()));
          cd->pf_wire.rows_max = R;
          cd->pf_wire.bf16 = opt_.prefill_wire_bf16;
          cd->pf_wire.n_max = H;
          cd->pf_wire.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(static_cast<size_t>(c), k::pair_rows_wire_words(R, H) * 4));
          cd->mem.push_back(cd->pf_wire.inbox);
        }
        cd->pf_ids = static_cast<int32_t *>(fab.alloc_host(std::max<size_t>(static_cast<size_t>(R) * 4, 4096)));
        cd->mem.push_back(cd->pf_ids);
      }
    }
    if (C > 1) {  // TP1: no wire (every kernel takes a null one: the sum is the card's own partial, rounded)
      cd->wire.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(static_cast<size_t>(c), k::pair_wire_words(H) * 4));
      cd->mem.push_back(cd->wire.inbox);
    }
    cd->wire.error = static_cast<uint32_t *>(dev(64));
    q.memset(cd->wire.error, 0, 64);
    q.wait();
    cards_.push_back(std::move(cd));
  }
  for (int c = 0; c < C && C > 1; ++c) {
    cards_[c]->wire.peer_inbox = cards_[1 - c]->wire.inbox;
    k::pair_wire_init(fab.compute(static_cast<size_t>(c)), H, cards_[c]->wire.inbox).wait();
    if (cards_[c]->pf_wire.inbox) {
      cards_[c]->pf_wire.peer_inbox = cards_[1 - c]->pf_wire.inbox;
      cards_[c]->pf_wire.error = cards_[c]->wire.error;
      cards_[c]->pf_split.peer_recv = cards_[1 - c]->pf_split.recv;
      cards_[c]->pf_split.peer_flags = cards_[1 - c]->pf_split.flags;
      cards_[c]->pf_split.error = cards_[c]->wire.error;
      k::pair_rows_wire_init(fab.compute(static_cast<size_t>(c)), opt.prefill_chunk, H, cards_[c]->pf_wire.inbox).wait();
    }
  }
  if (!opt_.prefill_slices) {  // h in the cards' stage arenas, the reduce on the mixers' wire
    std::vector<bf16 *> h;
    std::vector<const k::PairRowsWire *> w;
    for (auto &cd : cards_) {
      h.push_back(reinterpret_cast<bf16 *>(cd->pf_arena));
      if (C > 1) w.push_back(&cd->pf_wire);
    }
    mlp_->reserve_rows(opt.prefill_chunk, h, w);
  }
  if (lora_)  // the adapter's per-card scratch (mid + delta), sized to the largest prefill chunk
    lora_->reserve_scratch(fab, std::max(rows_max_, opt.prefill_chunk));
  if (!opt.dflash_dir.empty()) {  // the DFlash 2 drafter (own store, context rings, scratch; the target's embedding, lm_head)
    if (opt.prefill_slices) throw std::invalid_argument("qwen35 model: the drafter needs the many-row prefill (prefill_slices false)");
    DFlashOptions dop;
    dop.dense_bytes = opt.dflash_dense_bytes;
    dop.threads = opt.threads;
    dop.vocab = opt.mtp_vocab;
    dop.head_w4 = draft_w4;  // the draft rows' own choice
    dop.slots = opt.slots;
    dop.rows = rows_cap_;
    dop.prefill_rows = opt.prefill_chunk;
    dop.profile = opt.dflash_profile;
    dop.ctx_defer = opt.dflash_ctx_defer;
    dop.conf_cut = opt.dflash_conf_cut;
    dop.tree = opt.dflash_tree, dop.tree_cut = opt.dflash_tree_cut;
    dop.tree_seqs = opt.dflash_tree_seqs, dop.tree_temp = opt.dflash_tree_temp;
    dop.wide = rows_max_ >= k::kGdnWideRows ? 1 : 0, dop.tree_wide = opt.dflash_tree_wide;
    DFlashTarget tg;  // the shared drafter takes the target's sizes, embedding tables and lm_head rows
    tg.hidden = cfg_.hidden, tg.layers = cfg_.layers, tg.vocab = cfg_.vocab, tg.eos = cfg_.eos;
    for (int c = 0; c < static_cast<int>(fab.size()); ++c) tg.embed.push_back(head_->embed_table(c)), tg.lm.push_back(head_->lm_weights(c));
    if (cfg_.flint)  // the 8-bit affine table (no bf16 rows): the drafter's block rows through the head's own gather
      tg.gather = [h = head_.get()](sycl::queue &q, int c, const int32_t *ids, int rows, bf16 *out) { return h->gather_on(q, c, ids, rows, out); };
    dflash_ = std::make_unique<DFlash>(fab, opt.dflash_dir, dir, tg, cap_, dop);
    feat_of_.assign(static_cast<size_t>(L), -1);
    for (int l = 0; l < L; ++l) feat_of_[static_cast<size_t>(l)] = dflash_->feature_of(l);
  }
  // the MTP drafter (its own store, KV and scratch; the target's rope, embedding and lm_head); beside
  // DFlash with mtp_from > 0 (both drafters)
  if (opt.mtp_from < 0) throw std::invalid_argument("qwen35 model: mtp_from >= 0");
  if (opt.attn_select != 0 && (opt.attn_select < 32 || opt.attn_select > k::kGqaSelMax))
    throw std::invalid_argument("qwen35 model: attn_select 0 or 32..4096 pages");
  check_prefill_select(opt);
  if (opt.dflash_dir.empty() ? opt.mtp : opt.mtp_from > 0) {
    if (opt.prefill_slices) throw std::invalid_argument("qwen35 model: the drafter needs the many-row prefill (prefill_slices false)");
    std::vector<const bf16 *> rp(static_cast<size_t>(C));
    for (int c = 0; c < C; ++c) rp[static_cast<size_t>(c)] = cards_[static_cast<size_t>(c)]->rope;
    MtpOptions mo;
    mo.dense_bytes = opt.mtp_dense_bytes;
    mo.kv = opt.mtp_kv;
    mo.threads = opt.threads;
    mo.vocab = opt.mtp_vocab;
    mo.head_w4 = draft_w4;  // the draft rows' own choice
    mo.slots = opt.slots;
    mo.rows = rows_cap_;
    mo.kv_defer = opt.mtp_kv_defer;  // 
    mo.prefill_rows = opt.prefill_chunk;  // TP1: the fc's concatenated input rows
    mtp_ = std::make_unique<Mtp>(fab, dir, cfg_, cap_, rp, mo);
    const size_t mtp_pos_bytes = static_cast<size_t>(kMtpPasses) * kRecInts * 4;
    const size_t mtp_draft_bytes = static_cast<size_t>(kMtpPasses) * k::kStepRows32 * 4;
    for (int c = 0; c < C; ++c) {
      auto *p = static_cast<int32_t *>(fab.alloc(static_cast<size_t>(c), std::max<size_t>(mtp_pos_bytes, 4096)));
      cards_[static_cast<size_t>(c)]->mem.push_back(p);
      mtp_pos_.push_back(p);
    }
    mtp_drafts_ = static_cast<int32_t *>(fab.alloc(0, std::max<size_t>(mtp_draft_bytes, 4096)));
    cards_[0]->mem.push_back(mtp_drafts_);
  }
  // arch flint: a step writes up to 8 rows of a sequence into a ring, a many-row prefill chunk prefill_chunk rows (its
  // attention reads the window before the chunk after the chunk's rows are written)
  ring_rows_ = cfg_.flint && !opt_.prefill_slices ? std::max(rows_cap_, opt.prefill_chunk) : rows_cap_;
  {  // the KV pages (fixed per-slot layout, or one pool shared by the slots), the target's and the drafter's
    const int slot_pages = static_cast<int>(cap_ / k::kQsaPage);
    if (opt.loader) vram_free_before_kv_ = vram_free_gib();
    int64_t pool = 0;
    if (opt.kv_pool_tokens == 0) pool = static_cast<int64_t>(slot_pages) * opt.slots;
    else if (opt.kv_pool_tokens > 0) pool = (opt.kv_pool_tokens + k::kQsaPage - 1) / k::kQsaPage;
    else pool = slot_pages;  // auto: one context until kv_fit sizes the pool from the VRAM left
    if (opt.kv_pool_tokens != 0 && pool < slot_pages)
      throw std::runtime_error("qwen35 model: the KV pool (" + std::to_string(pool * k::kQsaPage) + " tokens) holds less than one context of " +
                               std::to_string(cap_) + " tokens");
    // A pool past the free VRAM is accepted by the driver and then pages (steps many times slower, or page faults):
    // refuse it, and name the longest max context whose single request fits (the runtime's first-use VRAM, kFirstUseGib
    // per card, kept free).
    if (vram_free_before_kv_ >= 0) {
      constexpr double kFirstUseGib = 1.5;
      const double room = (vram_free_before_kv_ - kFirstUseGib) * 1073741824.0;
      for (int c = 0; c < static_cast<int>(cards_.size()); ++c) {
        const double page = static_cast<double>(kv_page_bytes(c)), rings = static_cast<double>(ring_bytes_of(c));
        if (static_cast<double>(pool) * page + rings <= room) continue;
        const int64_t fit_ctx = room > rings ? static_cast<int64_t>((room - rings) / page) * k::kQsaPage : 0;
        char m[400];
        std::snprintf(m, sizeof m,
                      "qwen35 model: the KV of %lld tokens needs %.2f GiB per card, %.2f GiB of VRAM is free (%.1f kept for the runtime): "
                      "a max context of at most %lld tokens fits one request here (lower --context / XEN_CONTEXT%s)",
                      static_cast<long long>(pool * k::kQsaPage), (static_cast<double>(pool) * page + rings) / 1073741824.0, vram_free_before_kv_,
                      kFirstUseGib, static_cast<long long>(fit_ctx), opt.kv_pool_tokens > 0 ? " or --kv-pool-tokens" : "");
        throw std::runtime_error(m);
      }
    }
    if (!qsa_bank.empty()) {
      const size_t qsa_bytes=trained_qsa::Device::bytes_per_rank(qsa_bank.size(),cfg_.hidden,opt_.slots,static_cast<int>(cap_),opt_.prefill_chunk,step_rows_,trained_qsa_prefill_min_>0);
      if (!opt_.loader) throw std::runtime_error("trained QSA: VRAM preflight unavailable");
      if (opt_.loader) for (int c=0;c<C;++c) {
        const auto v=l0::vram_state(opt_.loader->info(fab_.rank_index(c)));
        const size_t kv_bytes=static_cast<size_t>(std::min<int64_t>(pool,static_cast<int64_t>(slot_pages)*opt_.slots))*kv_page_bytes(c)+ring_bytes_of(c);
        constexpr size_t startup_margin=1610612736;  // same 1.5 GiB first-use VRAM as the stock pool guard
        // Cache-safe QSA uses direct host snapshots plus one in-call capture
        // buffer; budget that buffer instead of silently spending the reserve.
        const size_t snap=opt.snap_gib>0 && !opt_.prefill_slices?
          std::max<size_t>(snap_bytes(c),4096)+std::max<size_t>(2*size_t(slot_pages)*4,4096):0;
        std::fprintf(stderr,"trained QSA budget: rank=%d free=%llu kv_bytes=%zu selector_bytes=%zu snapshot_device_bytes=%zu runtime_margin=%zu layers=%s\n",
                     c,static_cast<unsigned long long>(v.free),kv_bytes,qsa_bytes,snap,startup_margin,qsa_layers.c_str());
        if (!v.ok || v.free<kv_bytes+qsa_bytes+snap+startup_margin)
          throw std::runtime_error("trained QSA: refusing allocation; rank="+std::to_string(c)+" selector_bytes="+std::to_string(qsa_bytes)+" kv_bytes="+std::to_string(kv_bytes));
      }
      trained_qsa3_=std::make_unique<trained_qsa::Device>(fab_,std::span<const trained_qsa::Layer>(qsa_bank),opt_.slots,static_cast<int>(cap_),opt_.prefill_chunk,step_rows_,trained_qsa_prefill_min_>0);
      qsa_covered_.assign(opt_.slots,-1);
    }
    kv_alloc(pool);
  }
  rec_host_ = static_cast<int32_t *>(fab.alloc_host((1 + kMtpPasses) * kRecInts * 4));
  std::memset(rec_host_, 0, (1 + kMtpPasses) * kRecInts * 4);
  // the prefix cache's snapshot store (host buffers on first use) and the fork's page pairs per card
  if (opt.snap_gib > 0 && !opt_.prefill_slices) {  // (arch flint: the sliding layers' ring windows too)
    double one = 0;
    for (int c = 0; c < C; ++c) one += static_cast<double>(snap_bytes(c));
    snap_cap_ = util::snap_capacity(opt.snap_gib, one);  // cut to the host RAM left (pinned on first use)
    const size_t pairs = 2 * static_cast<size_t>(cap_ / k::kQsaPage) * 4;
    for (int c = 0; c < C; ++c) {
      auto *p = static_cast<int32_t *>(fab.alloc(static_cast<size_t>(c), std::max<size_t>(pairs, 4096)));
      cards_[static_cast<size_t>(c)]->mem.push_back(p);
      fork_pages_.push_back(p);
      std::array<void *, 2> st{};
      if(!trained_qsa3_) for (auto &b : st) {
        b = fab.alloc(static_cast<size_t>(c), std::max<size_t>(snap_bytes(c), 4096));
        cards_[static_cast<size_t>(c)]->mem.push_back(b);
      }
      snap_stage_.push_back(st);
      snap_stage_ev_.emplace_back();
      // the in-call capture's own buffer (restores / forks may use the staging buffers between the parts of a call)
      void *cb = fab.alloc(static_cast<size_t>(c), std::max<size_t>(snap_bytes(c), 4096));
      cards_[static_cast<size_t>(c)]->mem.push_back(cb);
      cap_buf_.push_back(cb);
      cap_ev_.emplace_back();
      snap_q_.emplace_back(std::in_place, fab.card_context(static_cast<size_t>(c)), fab.device(static_cast<size_t>(c)), sycl::property_list{sycl::property::queue::in_order()});
    }
  }
  if (tier_limit_.size() > 1) {  // the rows' delta in every rope read (as image input), the chunk's rope rows
    for (size_t c = 0; c < cards_.size(); ++c) {
      void *p = fab_.alloc(c, static_cast<size_t>(prefill_chunk()) * attn_.shape.rot * 2 + 4096);
      cards_[c]->mem.push_back(p);
      cards_[c]->pf_rope = static_cast<bf16 *>(p);
    }
    attn_.shape.mrope = true;
    if (mtp_) mtp_->set_mrope(true);
  }
  seqs_.assign(NS, Seq{});
  if (opt.select_self < 0 || opt.select_self > 64) throw std::invalid_argument("qwen35 model: select_self 0 (off) or 1..64 steps");
  if (opt.select_self > 0 && opt.attn_select > 0) {
    self_.assign(NS, SelfSelect{});
    for (int c = 0; c < C; ++c) self_counts_.emplace_back(static_cast<size_t>(attn_rows()) * NS * attn_.shape_of(c, 0).hkv());
    self_sync(opt.attn_select);
  }
  for (int s = 0; s < opt.slots; ++s) reset_slot(s);
  slot_ = 0;
  a8_rows_init();
  load_seq();
  load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Model::~Model() {
  for (size_t c = 0; c < fab_.size(); ++c) fab_.compute(c).wait();
  if (a8_rows_) {  // the registry holds this model's matrix pointers
    k::w4a8_rows_set(false);
    for (const uint8_t *w : a8_mats_) k::w4a8_rows_unregister(w);
  }
  graphs_.reset();  // the recorded graphs reference the buffers below
  lend_forget();  // the kernels' kept layouts point into the KV buffers
  for (auto &q : snap_q_)
    if (q) q->wait();
  for (auto &r : snaps_)
    for (void *p : r.host)
      if (p) fab_.release(p);
  mtp_.reset();
  dflash_.reset();
  moe_.reset();
  if (rec_host_) fab_.release(rec_host_);
  if (lg_host_) fab_.release(lg_host_);
  for (float *p : lp_raw_) fab_.release(p);
  for (float *p : lp_dev_) fab_.release(p);
  for (auto &cd : cards_)
    if (cd->side) cd->side->wait();
  for (auto &cd : cards_)
    for (void *p : cd->mem) fab_.release(p);
  for (auto &m : kv_mem_)
    for (void *p : m) fab_.release(p);
}

size_t Model::card_bytes(int c) const {
  size_t b = 0;
  for (const auto &kv : dense_->stats().card_bytes[static_cast<size_t>(c)]) b += kv.second;
  return b;
}

int Model::attn_rows() const { return static_cast<int>(std::count(cfg_.full_attention.begin(), cfg_.full_attention.end(), true)); }

// A slot's counts in the host copies: the common count for every head until its sequence has measured its weights.
void Model::self_reset(int slot) {
  if (self_.empty()) return;
  self_[static_cast<size_t>(slot)] = SelfSelect{};
  self_write(slot);
}

void Model::self_write(int slot) {
  const SelfSelect &st = self_[static_cast<size_t>(slot)];
  const int rows = attn_rows(), lo = k::kGqaSelSink + k::kGqaSelLocal + 1;
  for (size_t c = 0; c < cards_.size(); ++c) {
    const int hkv = attn_.shape_of(static_cast<int>(c), 0).hkv();
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < hkv; ++h) {
        const size_t unit = static_cast<size_t>(r) * cfg_.kv_heads + c * static_cast<size_t>(hkv) + static_cast<size_t>(h);
        const int pages = st.weights.empty() ? self_pages_ : select_head_pages(st.weights[unit], self_pages_);
        self_counts_[c][(static_cast<size_t>(r) * slots() + static_cast<size_t>(slot)) * hkv + h] = std::clamp(pages, lo, k::kGqaSelMax);
      }
    // (in order on the card's queue: before the next step's select launches; waited: the host copy may change again)
    fab_.compute(c).memcpy(cards_[c]->sel_slots, self_counts_[c].data(), self_counts_[c].size() * sizeof(int32_t)).wait();
  }
}

void Model::self_sync(int pages) {
  if (self_.empty() || pages <= 0 || pages == self_pages_) return;
  self_pages_ = pages;
  for (int s = 0; s < slots(); ++s) self_write(s);
}

// After a selected step: every segment whose sequence is still measuring and was selected in this step (its context
// passes the common count, so every head's pages were scored) adds the step's estimated lost-mass curves; after
// select_self steps the sequence's weights are fixed and its counts go to the cards.
void Model::self_measure(const std::vector<SeqRows> &segs, std::span<const int64_t> pos_before, int pages) {
  if (self_.empty() || pages <= kSelectFloor) return;  // (up to the floor every head keeps the common count)
  constexpr int F = SelectShare::kFractions;
  const int rows = attn_rows(), units = rows * cfg_.kv_heads;
  const float scale = 1.f / std::sqrt(static_cast<float>(attn_.shape.head_dim));  // the attention's own logit scale
  std::vector<float> scores;
  for (size_t i = 0; i < segs.size(); ++i) {
    SelfSelect &st = self_[static_cast<size_t>(segs[i].slot)];
    const int64_t last_pos = pos_before[i] + segs[i].rows - 1;
    const int np = static_cast<int>(last_pos / k::kQsaPage) + 1;
    if (!st.weights.empty() || np <= pages) continue;
    st.lost.resize(static_cast<size_t>(units) * F, 0.0);
    for (size_t c = 0; c < cards_.size(); ++c) {
      const int hkv = attn_.shape_of(static_cast<int>(c), 0).hkv();
      const size_t block = static_cast<size_t>(hkv) * k::kGqaSelPagesMax;  // one unit's keys of one layer
      self_keys_.resize(static_cast<size_t>(rows) * block);
      auto &q = fab_.compute(c);
      for (int r = 0; r < rows; ++r)
        q.memcpy(self_keys_.data() + static_cast<size_t>(r) * block, cards_[c]->sel_keys + (static_cast<size_t>(r) * k::kSegs16 + i) * block, block * sizeof(uint32_t));
      q.wait();
      for (int r = 0; r < rows; ++r)
        for (int h = 0; h < hkv; ++h) {
          const uint32_t *keys = self_keys_.data() + static_cast<size_t>(r) * block + static_cast<size_t>(h) * k::kGqaSelPagesMax;
          scores.resize(static_cast<size_t>(np));
          std::transform(keys, keys + np, scores.begin(), select_key_score);
          const size_t unit = static_cast<size_t>(r) * cfg_.kv_heads + c * static_cast<size_t>(hkv) + static_cast<size_t>(h);
          select_curve_add(scores, last_pos + 1, k::kQsaPage, k::kGqaSelSink, k::kGqaSelLocal, scale, std::span<double>(st.lost).subspan(unit * F, F));
        }
    }
    if (++st.steps < opt_.select_self) continue;
    st.weights = SelectShare::weights(st.lost, units);
    st.lost = {};
    self_write(segs[i].slot);
  }
}

namespace {
// Optional dense/sparse crossover for decode: below it, every row runs the stock dense kernel.
int64_t qsa_decode_min_pos() {
  static const int64_t v=[] { const char *s=std::getenv("XEN_TRAINED_QSA_DECODE_MIN_POS"); return s && *s?std::max<int64_t>(0,std::atoll(s)):0; }();
  return v;
}
bool qsa_resident_snapshots() {
  static const bool enabled=[] {
    const char *s=std::getenv("XEN_QSA_RESIDENT_SNAPSHOTS");return s && std::string_view(s)=="1";
  }();
  return enabled;
}
}
void Model::qsa_invalidate_snapshots(int slot,int64_t retained) {
  if(!qsa_resident_snapshots()) return;
  for(auto &r:snaps_) if(r.used && r.qsa_owner==slot && r.pos>retained) r.qsa_owner=-1;
}
void Model::reset_slot(int s) {
  if(trained_qsa3_) {qsa_covered_[s]=-1;qsa_invalidate_snapshots(s);}
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    const auto gs = gdn_.shape_of(static_cast<int>(c));
    for (int l : gdn_.layers) k::gdn_state_reset(q, gs, gdn_view(cards_[c]->gdn[l], s));
    q.wait();
  }
  seqs_[static_cast<size_t>(s)] = Seq{};
  self_reset(s);
  if (mtp_) mtp_->drop(s);  // deferred drafter KV rows of the old sequence
  if (capt_.slot == s) capt_.ready = false;  // an in-call capture of the old sequence
  if (cap_arm_slot_ == s) cap_arm_slot_ = -1;
  kvp_.release(s);  // a fresh sequence holds no pages of a shared pool (the fixed layout keeps its own)
}

bool Model::kv_reserve(int slot, int64_t tokens) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if (tokens > cap_) return false;
  if (!kvp_.reserve(slot, tokens)) {
    // pages past the lend limit come back from the lent layouts first
    if (!lend_st_.on || !lend_raise(kvp_.pages_short(slot, tokens)) || !kvp_.reserve(slot, tokens)) return false;
  }
  kv_upload();
  return true;
}

// ---- KV lending (ModelOptions::kv_lend) ----

void Model::lend_forget() {
  if (lend_mat_.empty()) return;
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    q.wait();
    for (size_t i = 0; i < lend_item_.size(); ++i)
      if (lend_item_[i]) k::onednn_pf_keep(q, lend_mat_[c][i].w, 0, 0, nullptr);
  }
  std::fill(lend_item_.begin(), lend_item_.end(), 0);
  lend_st_.lent = 0;
}

// The target's decode int4 matrices by pointer (kernels::w4a8_rows_register) and the cards' scratch for A's codes;
// StepOptions::w4_a8 switches the kernels over. The dense 27B's plain symmetric g128 matrices.
void Model::a8_rows_init() {
  if (cfg_.llama() || cfg_.flint || moe_ || !mlp_) return;
  const size_t C = cards_.size();
  int max_k = 0;
  bool ok = true;
  std::vector<const uint8_t *> mats;
  auto add = [&](const uint8_t *w, int n, int kk) {
    if (!w) return;
    if (n % k::kW4Width || kk % k::kW4Group || k::w4_zero_points(w) || k::w4_affine(w)) ok = false;
    mats.push_back(w), max_k = std::max(max_k, kk);
  };
  for (size_t c = 0; c < C; ++c)
    for (int l = 0; l < cfg_.layers; ++l) {
      if (std::find(attn_.layers.begin(), attn_.layers.end(), l) != attn_.layers.end()) {
        const k::GqaWeights &w = attn_.w[c][static_cast<size_t>(l)];
        if (w.dense()) ok = false;
        else add(w.qkv.w, w.qkv.n, w.qkv.k), add(w.o.w, w.o.n, w.o.k);
      } else if (std::find(gdn_.layers.begin(), gdn_.layers.end(), l) != gdn_.layers.end()) {
        const k::GdnWeights &w = gdn_.w[c][static_cast<size_t>(l)];
        add(w.qkvz4.w, w.qkvz4.n, w.qkvz4.k), add(w.out4.w, w.out4.n, w.out4.k);
      }
      const MlpWeights mw = mlp_->weights(*dense_, static_cast<int>(c), l);
      add(mw.gate_up.w, mw.gate_up.n, mw.gate_up.k), add(mw.down.w, mw.down.n, mw.down.k);
    }
  if (!ok || mats.empty()) return;
  for (const uint8_t *w : mats) k::w4a8_rows_register(w);
  for (size_t c = 0; c < C; ++c) k::w4a8_rows_reserve(fab_.compute(c), max_k);
  a8_mats_ = std::move(mats), a8_rows_ = true;
}

void Model::lend_layout() {
  lend_st_ = LendStats{};
  lend_ = model::KvLendRegions{};
  lend_mat_.clear(), lend_buf_.clear(), lend_item_.clear();
  lend_note_ = "off";
  if (!opt_.kv_lend) return;
  auto refuse = [&](const std::string &why) { lend_note_ = "off (" + why + ")"; };
  if (!kvp_.shared()) return refuse("needs the shared KV pool");
  if (opt_.kv_pool_scramble) return refuse("the pool is scrambled");
  const int min_n = k::w4a16_pf_onednn();
  if (min_n <= 0 || !opt_.prefill_w4_a8) return refuse("the prefill GEMMs are not on the oneDNN route");
  const size_t C = cards_.size();
  // the wide matrices per card, in the layers' order (what a prefill call passes to the W4A8 GEMM: pointer, n, k)
  lend_mat_.resize(C);
  for (size_t c = 0; c < C; ++c)
    for (int l = 0; l < cfg_.layers; ++l) {
      auto add = [&](const uint8_t *w, int n, int kk) {
        if (w && n >= min_n) lend_mat_[c].push_back(LendMat{w, n, kk});
      };
      if (std::find(attn_.layers.begin(), attn_.layers.end(), l) != attn_.layers.end()) {
        const k::GqaWeights &w = attn_.w[c][static_cast<size_t>(l)];
        if (!w.dense()) add(w.qkv.w, w.qkv.n, w.qkv.k);
      } else if (std::find(gdn_.layers.begin(), gdn_.layers.end(), l) != gdn_.layers.end()) {
        const k::GdnWeights &w = gdn_.w[c][static_cast<size_t>(l)];
        add(w.qkvz4.w, w.qkvz4.n, cfg_.hidden);
      }
      if (mlp_ && !moe_) {
        const MlpWeights mw = mlp_->weights(*dense_, static_cast<int>(c), l);
        add(mw.gate_up.w, mw.gate_up.n, cfg_.hidden);
      }
    }
  // the narrow matrices (the mixers' out / o and the MLP's down, n = hidden) after every wide one: they take the oneDNN
  // route only under a lower width threshold, and the wide ones keep their place in the lending order
  for (size_t c = 0; c < C; ++c)
    for (int l = 0; l < cfg_.layers; ++l) {
      auto add = [&](const uint8_t *w, int n, int kk) {
        if (w && n >= min_n) lend_mat_[c].push_back(LendMat{w, n, kk});
      };
      if (std::find(attn_.layers.begin(), attn_.layers.end(), l) != attn_.layers.end()) {
        const k::GqaWeights &w = attn_.w[c][static_cast<size_t>(l)];
        if (!w.dense()) add(w.o.w, w.o.n, w.o.k);
      } else if (std::find(gdn_.layers.begin(), gdn_.layers.end(), l) != gdn_.layers.end()) {
        const k::GdnWeights &w = gdn_.w[c][static_cast<size_t>(l)];
        add(w.out4.w, w.out4.n, w.out4.k);
      }
      if (mlp_ && !moe_) {
        const MlpWeights mw = mlp_->weights(*dense_, static_cast<int>(c), l);
        add(mw.down.w, mw.down.n, mw.down.k);
      }
    }
  for (size_t c = 1; c < C; ++c)
    if (lend_mat_[c].size() != lend_mat_[0].size()) return lend_mat_.clear(), refuse("the cards' wide matrices differ");
  if (lend_mat_[0].empty()) return lend_mat_.clear(), refuse("no matrix on the oneDNN route");
  // the pool's K / V buffers per card (the sliding layers' rings are apart: not lent)
  std::vector<size_t> pb;
  lend_buf_.resize(C);
  for (size_t c = 0; c < C; ++c) {
    const auto as = attn_.shape_of(static_cast<int>(c));
    std::vector<size_t> bytes;
    for (int l : attn_.layers) {
      if (attn_.ring_pages(l, ring_rows_) > 0) continue;
      const k::GqaCache &kc = cards_[c]->kv[l];
      lend_buf_[c].push_back(kc.k), bytes.push_back(k::gqa_page_kv_bytes(as));
      lend_buf_[c].push_back(kc.v), bytes.push_back(k::gqa_page_v_bytes(as));
    }
    for (const uint8_t *p : lend_buf_[c])
      if (reinterpret_cast<uintptr_t>(p) % 4096) throw std::logic_error("qwen35 model: a KV pool buffer is not 4 KiB aligned");
    if (c == 0) pb = bytes;
    else if (bytes != pb) return lend_mat_.clear(), lend_buf_.clear(), refuse("the cards' pools differ");
  }
  const size_t N = lend_mat_[0].size();
  std::vector<size_t> sizes(N, 0);
  for (size_t c = 0; c < C; ++c)
    for (size_t i = 0; i < N; ++i) sizes[i] = std::max(sizes[i], static_cast<size_t>(lend_mat_[c][i].n) * lend_mat_[c][i].k / 2);
  // the per-call fallback's scratch exists before a pool is sized from the VRAM left (kv_fit), used yet or not
  for (size_t c = 0; c < C; ++c) k::onednn_pf_reserve(fab_.compute(c), *std::max_element(sizes.begin(), sizes.end()));
  lend_ = model::KvLendRegions(pb, kvp_.pool_pages(), sizes, 4096);
  lend_item_.assign(N, 0);
  lend_keep_pages_ = std::max(1, model::KvPages::pages_for(opt_.kv_lend_keep));
  lend_st_.on = true;
  lend_st_.items = static_cast<int>(N);
  lend_apply(std::max(lend_keep_pages_, kvp_.high_water() + lend_keep_pages_));
  char b[240];
  std::snprintf(b, sizeof b, "on: %d of %d wide prefill matrices keep their oneDNN layout in the KV pool's free pages (%.2f GiB per card), KV limit %lld tokens, keep %lld",
                lend_st_.lent, lend_st_.items, lend_st_.lent_gib, static_cast<long long>(lend_st_.limit_tokens), static_cast<long long>(opt_.kv_lend_keep));
  lend_note_ = b;
}

// Lends exactly the regions the KV does not touch at lend limit `target` (>= the highest held page: the caller's), and
// sets the pool's limit to the highest that keeps them. A region taken back: its layout is forgotten after the card's
// queue drained (a prefill GEMM in flight may read it), before any KV kernel can write there.
void Model::lend_apply(int target) {
  const int PP = kvp_.pool_pages();
  target = std::clamp(target, 1, PP);
  const size_t N = lend_item_.size();
  bool drop = false;
  for (size_t i = 0; i < N; ++i) drop |= lend_item_[i] && !lend_.usable(i, target);
  if (drop)
    for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  size_t bytes = 0;
  int lent = 0;
  for (size_t i = 0; i < N; ++i) {
    const bool want = lend_.usable(i, target);
    if (want != static_cast<bool>(lend_item_[i])) {
      const auto &r = lend_.regions()[i];
      for (size_t c = 0; c < cards_.size(); ++c) {
        const LendMat &m = lend_mat_[c][i];
        k::onednn_pf_keep(fab_.compute(c), m.w, m.n, m.k, want ? lend_buf_[c][static_cast<size_t>(r.buf)] + r.off : nullptr);
      }
      lend_item_[i] = want;
    }
    if (want) ++lent, bytes += lend_.regions()[i].bytes;
  }
  const int limit = lend_.limit_for(target, PP);
  if (!kvp_.set_limit(limit)) throw std::logic_error("qwen35 model: the lend limit below a held page");
  lend_st_.lent = lent;
  lend_st_.lent_gib = static_cast<double>(bytes) / 1073741824.0;
  lend_st_.limit_tokens = static_cast<int64_t>(kvp_.limit()) * model::KvPages::kPageTokens;
}

bool Model::lend_raise(int short_pages) {
  if (short_pages <= 0) return true;
  const int PP = kvp_.pool_pages();
  if (kvp_.limit() >= PP) return false;
  // pages [limit, want) are all free (no KV past the limit): they cover the shortfall, and a keep beyond it
  const int want = std::min(PP, kvp_.limit() + (short_pages - kvp_.free_below_limit()) + lend_keep_pages_);
  const int before = lend_st_.lent;
  lend_apply(want);
  lend_st_.raises++;
  lend_st_.given_back += static_cast<uint64_t>(before - lend_st_.lent);
  std::fprintf(stderr, "qwen35: KV lending: KV limit -> %lld tokens, %d kept layouts given back (%d stay)\n", static_cast<long long>(lend_st_.limit_tokens),
               before - lend_st_.lent, lend_st_.lent);
  return kvp_.free_below_limit() >= short_pages;
}

int Model::lend_idle(int max_fill) {
  if (!lend_st_.on || pf_.active) return 0;
  const int target = std::max(lend_keep_pages_, kvp_.high_water() + lend_keep_pages_);
  if (target < kvp_.limit() && lend_.lent(target) > lend_st_.lent) {
    const int before = lend_st_.lent;
    lend_apply(target);
    lend_st_.relends += static_cast<uint64_t>(lend_st_.lent - before);
    std::fprintf(stderr, "qwen35: KV lending: %d layouts lent again while idle (%d of %d, KV limit %lld tokens)\n", lend_st_.lent - before, lend_st_.lent,
                 lend_st_.items, static_cast<long long>(lend_st_.limit_tokens));
    return lend_st_.lent - before;
  }
  int done = 0;
  for (size_t c = 0; c < cards_.size(); ++c) done = std::max(done, k::onednn_pf_keep_fill(fab_.compute(c), std::max(0, max_fill)));
  if (done)
    for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  return done;
}

Model::LendStats Model::lend_stats() {
  LendStats s = lend_st_;
  if (!s.on) return s;
  s.filled = s.lent;
  for (size_t c = 0; c < cards_.size(); ++c) {
    int kept = 0, filled = 0;
    k::onednn_pf_keep_count(fab_.compute(c), &kept, &filled);
    s.filled = std::min(s.filled, filled);
  }
  return s;
}

void Model::kv_release(int slot) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if(trained_qsa3_) {qsa_covered_[slot]=-1;qsa_invalidate_snapshots(slot);}
  if (mtp_) mtp_->drop(slot);  // 
  kvp_.release(slot);
}

void Model::kv_need(int slot, int64_t tokens) {
  if (!kv_reserve(slot, std::min(tokens, cap_)))
    throw std::runtime_error("qwen35 model: the KV pool is out of pages (slot " + std::to_string(slot) + " needs " + std::to_string(tokens) + " tokens, " +
                             std::to_string(kvp_.free_tokens()) + " free)");
}

// Changed entries of the host table to every card. Rare (a page per 64 tokens per sequence) and small, from pageable
// memory: wait, so the host table may change again right after.
void Model::kv_upload() {
  if (!kvp_.dirty()) return;
  const size_t lo = kvp_.dirty_lo(), n = kvp_.dirty_hi() - lo;
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).memcpy(kv_table_[c] + lo, kvp_.table().data() + lo, n * 4);
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  kvp_.clear_dirty();
}

size_t Model::ring_bytes_of(int c) const {  // arch flint: the sliding layers' rings on card c (all slots)
  const auto as = attn_.shape_of(c);
  size_t b = 0;
  for (int l : attn_.layers)
    if (const int rp = attn_.ring_pages(l, ring_rows_); rp > 0)
      b += static_cast<size_t>(rp) * static_cast<size_t>(opt_.slots) *
           k::gqa_page_total_bytes(as);
  return b;
}

size_t Model::kv_page_bytes(int c) const {  // one physical page of every attention layer + the drafter on card c
  const auto as = attn_.shape_of(c);
  const size_t one = k::gqa_page_total_bytes(as) +
                     (opt_.page_bounds() ? k::gqa_page_bound_bytes(as) : 0);  // the key bounds
  size_t paged = 0;
  for (int l : attn_.layers) paged += attn_.ring_pages(l, ring_rows_) == 0;  // arch flint: the sliding layers' rings are apart
  return one * paged + (mtp_ ? mtp_->kv_page_bytes(c) : 0);
}

double Model::vram_free_gib() {
  if (!opt_.loader) return -1;
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  double lo = 1e30;
  for (size_t c = 0; c < cards_.size(); ++c) {
    const auto v = l0::vram_state(opt_.loader->info(fab_.rank_index(c)));
    if (v.ok) lo = std::min(lo, static_cast<double>(v.free));
  }
  return lo < 1e30 ? lo / 1073741824.0 : -1;
}

// (Re)allocates the K / V (+ scales) of every attention layer and the drafter for `pool` physical pages per card; the
// page table (slots x capacity / 64 ints per card) once. Every slot holds no page afterwards (or every page: fixed).
void Model::kv_alloc(int64_t pool) {
  for(auto &r:snaps_) r.qsa_owner=-1;
  const int C = static_cast<int>(cards_.size()), slot_pages = static_cast<int>(cap_ / k::kQsaPage);
  pool = std::min<int64_t>(pool, static_cast<int64_t>(slot_pages) * opt_.slots);  // more could never be used
  // kernel offsets are int element indices: pages x kv heads x 64 tokens x 256 dims stays below 2^31
  if (static_cast<double>(pool) * attn_.shape_of(0).hkv() * k::kQsaPage * k::kQsaDim >= 2147483648.0)
    throw std::runtime_error("qwen35 model: the KV pool exceeds the kernels' int offsets (2^31 elements per layer)");
  for (int c = 0; c < C; ++c) fab_.compute(static_cast<size_t>(c)).wait();
  graphs_.reset();  // recorded step graphs point at the KV buffers freed below (kv_fit after a warm-up)
  lend_forget();  // and so do the kept layouts
  if (opt_.kv_pool_tokens == 0) kvp_ = model::KvPages(opt_.slots, slot_pages, static_cast<int>(pool), false);
  else kvp_ = model::KvPages(opt_.slots, slot_pages, static_cast<int>(pool), true, opt_.kv_pool_scramble);
  const size_t P = static_cast<size_t>(pool);
  kv_bytes_.assign(static_cast<size_t>(C), 0);
  kv_mem_.resize(static_cast<size_t>(C));
  for (int c = 0; c < C; ++c) {
    Card &cd = *cards_[static_cast<size_t>(c)];
    for (void *p : kv_mem_[static_cast<size_t>(c)]) fab_.release(p);
    kv_mem_[static_cast<size_t>(c)].clear();
    auto dev = [&](size_t bytes, bool kv) {
      bytes = std::max<size_t>(bytes, 4096);
      void *p = fab_.alloc(static_cast<size_t>(c), bytes);
      (kv ? kv_mem_[static_cast<size_t>(c)] : cd.mem).push_back(p);
      return p;
    };
    const auto as = attn_.shape_of(c);
    if (kv_table_.size() <= static_cast<size_t>(c)) kv_table_.push_back(static_cast<int32_t *>(dev(kvp_.table().size() * 4, false)));
    int32_t *pg = kv_table_[static_cast<size_t>(c)];
    size_t ring_bytes = 0;
    for (int l : attn_.layers) {
      k::GqaCache &kc = cd.kv[l];
      kc = k::GqaCache{};
      if (const int rp = attn_.ring_pages(l, ring_rows_); rp > 0) {
        // arch flint, a sliding layer: a fixed ring of rp pages per slot (slot s owns physical pages [s rp, s rp + rp):
        // an identity table shared by the sliding layers, allocated with the first), outside the shared pool
        const size_t RP = static_cast<size_t>(rp) * static_cast<size_t>(opt_.slots);
        if (cd.ring_table == nullptr) {
          cd.ring_table = static_cast<int32_t *>(dev(RP * 4, false));
          std::vector<int32_t> id(RP);
          for (size_t i = 0; i < RP; ++i) id[i] = static_cast<int32_t>(i);
          fab_.compute(static_cast<size_t>(c)).memcpy(cd.ring_table, id.data(), RP * 4).wait();
        }
        kc.k = static_cast<uint8_t *>(dev(RP * k::gqa_page_kv_bytes(as), true));
        kc.v = static_cast<uint8_t *>(dev(RP * k::gqa_page_v_bytes(as), true));
        if (k::qsa_kv_scaled(as.kv)) {
          kc.ks = static_cast<float *>(dev(RP * k::gqa_page_scale_floats(as) * 4, true));
          kc.vs = static_cast<float *>(dev(RP * k::gqa_page_vscale_floats(as) * 4, true));
        }
        kc.pages = cd.ring_table;
        kc.max_tokens = static_cast<int>(cap_);
        kc.slot_pages = rp;
        kc.ring_pages = rp;
        ring_bytes += RP * k::gqa_page_total_bytes(as);
        continue;
      }
      kc.k = static_cast<uint8_t *>(dev(P * k::gqa_page_kv_bytes(as), true));
      kc.v = static_cast<uint8_t *>(dev(P * k::gqa_page_v_bytes(as), true));
      if (k::qsa_kv_scaled(as.kv)) {
        kc.ks = static_cast<float *>(dev(P * k::gqa_page_scale_floats(as) * 4, true));
        kc.vs = static_cast<float *>(dev(P * k::gqa_page_vscale_floats(as) * 4, true));
      }
      if (opt_.page_bounds()) {  // every page's key bounds
        kc.kmin = static_cast<bf16 *>(dev(P * k::gqa_page_bound_bytes(as) / 2, true));
        kc.kmax = static_cast<bf16 *>(dev(P * k::gqa_page_bound_bytes(as) / 2, true));
      }
      kc.pages = pg;
      kc.max_tokens = static_cast<int>(cap_);
      kc.slot_pages = slot_pages;
    }
    kv_bytes_[static_cast<size_t>(c)] = P * kv_page_bytes(c) + ring_bytes;
  }
  if (mtp_) {
    std::vector<const int32_t *> t(kv_table_.begin(), kv_table_.end());
    mtp_->attach_kv(static_cast<int>(pool), t, slot_pages);
  }
  kv_upload();
  lend_layout();
}

int64_t Model::kv_fit(double reserve_gib) {
  if (!kvp_.shared()) throw std::logic_error("qwen35 model: kv_fit sizes a shared pool (kv_pool_tokens != 0)");
  if (!opt_.loader) throw std::invalid_argument("qwen35 model: kv_fit needs ModelOptions::loader (the free-VRAM query)");
  if (pf_.active) throw std::logic_error("qwen35 model: kv_fit with a prefill call pending");
  for (int s = 0; s < slots(); ++s)
    if (kvp_.held_pages(s) > 0) throw std::logic_error("qwen35 model: kv_fit with sequences holding pages (reset every slot first)");
  const double free = vram_free_gib();
  if (free < 0) throw std::runtime_error("qwen35 model: sysman free VRAM unavailable; pass kv_pool_tokens");
  // the tighter card: what the pool holds now + what is free beyond the reserve
  int64_t fit = INT64_MAX;
  for (size_t c = 0; c < cards_.size(); ++c) {
    const double have = static_cast<double>(kv_bytes_[c] - ring_bytes_of(static_cast<int>(c))) + (free - reserve_gib) * 1073741824.0;
    fit = std::min<int64_t>(fit, have > 0 ? static_cast<int64_t>(have / static_cast<double>(kv_page_bytes(static_cast<int>(c)))) : 0);
  }
  const int64_t slot_pages = cap_ / k::kQsaPage;
  if (fit < slot_pages)
    throw std::runtime_error("qwen35 model: the VRAM left (" + std::to_string(free) + " GiB free) holds less than one context of KV");
  kv_alloc(fit);
  for (int s = 0; s < slots(); ++s) seqs_[static_cast<size_t>(s)] = Seq{}, self_reset(s);
  std::fill(qsa_covered_.begin(), qsa_covered_.end(), -1);  // the trained history described the freed KV
  load_seq();
  return kvp_.pool_tokens();
}

// ----: the prefix cache's snapshots ----

size_t Model::snap_bytes(int c) const {
  const Card &cd = *cards_.at(static_cast<size_t>(c));
  size_t b = 0;
  for (int l : gdn_.layers) b += cd.gdn[static_cast<size_t>(l)].slot_S * 4 + cd.gdn[static_cast<size_t>(l)].slot_win * 2;
  if (dflash_) b += dflash_->ring_slot_bytes(c);
  return b + ring_snap_bytes(c);
}

// arch flint: a snapshot at pos holds each sliding layer's ring pages of the positions a row at pos or after
// reads below pos: the pages (p / 64) of p in [pos - window + 1, pos), at most kRingSnapPages, as [k][v][k scales][v
// scales] x kRingSnapPages pages per layer (page j = page number pos0 + j, pos0 = max(0, (pos - 1) / 64 + 1 -
// kRingSnapPages)). The full layers' KV is in the slot's pool pages (kept by the prefix cache) and nothing else holds
// sequence state.
size_t Model::ring_snap_bytes(int c) const {
  const auto as = attn_.shape_of(c);
  size_t b = 0;
  for (int l : attn_.layers)
    if (cards_.at(static_cast<size_t>(c))->kv[static_cast<size_t>(l)].ring_pages > 0)
      b += kRingSnapPages * k::gqa_page_total_bytes(as);
  return b;
}

// The slot's ring windows at pos <-> buf (ring_snap_bytes' layout) on q; at most two contiguous runs of ring pages per
// buffer. Returns the last copy's event (none submitted: a default event).
sycl::event Model::ring_snap_copy(int c, sycl::queue &q, int slot, int64_t pos, char *buf, bool save, sycl::event dep) {
  const auto as = attn_.shape_of(c);
  const bool sc = k::qsa_kv_scaled(as.kv);
  const size_t kvb = k::gqa_page_kv_bytes(as), sfb = sc ? k::gqa_page_scale_floats(as) * 4 : 0;
  const size_t vb = k::gqa_page_v_bytes(as), vsfb = sc ? k::gqa_page_vscale_floats(as) * 4 : 0;
  const int64_t pn1 = pos > 0 ? (pos - 1) / k::kQsaPage + 1 : 0;  // page numbers [pn0, pn1)
  const int64_t pn0 = std::max<int64_t>(0, pn1 - kRingSnapPages);
  sycl::event last;
  bool first = true;
  auto cp = [&](void *dst, const void *src, size_t n) {
    last = first ? q.memcpy(dst, src, n, dep) : q.memcpy(dst, src, n);
    first = false;
  };
  for (int l : attn_.layers) {
    const k::GqaCache &kc = cards_[static_cast<size_t>(c)]->kv[static_cast<size_t>(l)];
    const int rp = kc.ring_pages;
    if (rp <= 0) continue;
    struct B {
      char *base;
      size_t pb;
    };
    const B bufs[4] = {{static_cast<char *>(static_cast<void *>(kc.k)), kvb}, {static_cast<char *>(static_cast<void *>(kc.v)), vb},
                       {reinterpret_cast<char *>(kc.ks), sfb}, {reinterpret_cast<char *>(kc.vs), vsfb}};
    // each part at its own offset ([k][v][k scales][v scales] x kRingSnapPages)
    size_t off = 0;
    for (int bi = 0; bi < (sc ? 4 : 2); ++bi) {
      const B &bb = bufs[bi];
      char *h = buf + off;
      off += kRingSnapPages * bb.pb;
      for (int64_t pn = pn0; pn < pn1;) {  // a run of consecutive ring indices
        const int64_t ri = pn % rp;
        const int64_t run = std::min<int64_t>(pn1 - pn, rp - ri);
        char *dev = bb.base + (static_cast<size_t>(slot) * rp + static_cast<size_t>(ri)) * bb.pb;
        char *hp = h + static_cast<size_t>(pn - pn0) * bb.pb;
        if (save) cp(hp, dev, static_cast<size_t>(run) * bb.pb);
        else cp(dev, hp, static_cast<size_t>(run) * bb.pb);
        pn += run;
      }
    }
    buf += kRingSnapPages * (kvb + vb + sfb + vsfb);  // = gqa_page_total_bytes (ring_snap_bytes)
  }
  return last;
}

// The slot's committed GDN states (S, the window of the snapshot's parity) and DFlash rings into the snapshot's host
// buffers (save) or back (restore), in snap_bytes' layout, through one of two device staging buffers per card: the
// gather / scatter between the state and the staging buffer runs on the compute queue (in order with the calls around
// it), the host copy of the whole buffer on snap_q_ (one large copy). A save's host copy runs beside the calls that
// follow; a restore waits for its own.
void Model::snap_copy(int slot, SnapRec &r, bool save) {
  // a save reads the slot's whole context, a restore replaces it
  if (dflash_) dflash_->settle(slot, save ? std::numeric_limits<int64_t>::max() : 0);
  if(trained_qsa3_) {
    // One scheduler thread; in-order device copies preserve the exact snapshot
    // layout without two full-sized device staging buffers. A host snapshot
    // can be reused only after its last reader AND writer have completed.
    r.saved.resize(cards_.size()); r.read.resize(cards_.size());
    for(size_t c=0;c<cards_.size();++c) {
      auto &q=fab_.compute(c);
      q.ext_oneapi_submit_barrier({r.saved[c],r.read[c]});
      auto *host=static_cast<char *>(r.host[c]);
      for(int l:gdn_.layers) {
        const auto &st=cards_[c]->gdn[l]; const auto g=gdn_view(st,slot);
        const size_t sb=st.slot_S*4,wb=st.slot_win*2;
        if(save) { q.memcpy(host,g.S,sb); q.memcpy(host+sb,g.win[r.parity],wb); }
        else { q.memcpy(g.S,host,sb); q.memcpy(g.win[r.parity],host+sb,wb); }
        host+=sb+wb;
      }
      if(dflash_) dflash_->ring_copy(static_cast<int>(c),q,slot,host,save);
      if(save) r.saved[c]=q.ext_oneapi_submit_barrier();
      else r.read[c]=q.ext_oneapi_submit_barrier();
    }
    return;  // trained QSA already excludes Flint's ring-cache variant
  }
  const size_t k = static_cast<size_t>(snap_stage_next_);
  snap_stage_next_ ^= 1;
  r.saved.resize(cards_.size());
  r.read.resize(cards_.size());
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    char *stage = static_cast<char *>(snap_stage_[c][k]);
    const size_t bytes = snap_bytes(static_cast<int>(c));
    sycl::event dep = snap_stage_ev_[c][k];  // the staging buffer's last user is done
    if (!save) {
      r.read[c] = snap_q_[c]->memcpy(stage, r.host[c], bytes, {dep, r.saved[c]});
      dep = r.read[c];
    }
    sycl::event last;
    auto cp = [&](void *dst, const void *src, size_t n) {
      last = q.memcpy(dst, src, n, dep);
      dep = sycl::event{};
    };
    char *h = stage;
    for (int l : gdn_.layers) {
      const k::GdnState &st = cards_[c]->gdn[static_cast<size_t>(l)];
      const k::GdnState g = gdn_view(st, slot);
      const size_t sb = st.slot_S * 4, wb = st.slot_win * 2;
      if (save) cp(h, g.S, sb), cp(h + sb, g.win[r.parity], wb);
      else cp(g.S, h, sb), cp(g.win[r.parity], h + sb, wb);
      h += sb + wb;
    }
    if (dflash_) {
      dflash_->ring_copy(static_cast<int>(c), q, slot, h, save);
      last = q.ext_oneapi_submit_barrier();
      h += dflash_->ring_slot_bytes(static_cast<int>(c));
    }
    if (cfg_.flint) {  // the sliding layers' ring windows
      const sycl::event e = ring_snap_copy(static_cast<int>(c), q, slot, r.pos, h, save, dep);
      dep = sycl::event{};
      last = q.ext_oneapi_submit_barrier({e});
    }
    if (save) {
      r.saved[c] = snap_q_[c]->memcpy(r.host[c], stage, bytes, {last, r.read[c]});
      snap_stage_ev_[c][k] = r.saved[c];
    } else {
      snap_stage_ev_[c][k] = last;
    }
  }
}

int Model::snap_save(int slot) {
  if (snap_cap_ <= 0) return -1;
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if (pf_.active && pf_.slot == slot) throw std::logic_error("qwen35 model: a snapshot with the slot's prefill call pending");
  save_seq();
  const Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (!sq.pf_committed) throw std::logic_error("qwen35 model: a snapshot is taken right after a prefill call (a committed state)");
  int id = -1;
  for (size_t i = 0; i < snaps_.size() && id < 0; ++i)
    if (!snaps_[i].used) id = static_cast<int>(i);
  if (id < 0) {
    if (static_cast<int>(snaps_.size()) >= snap_cap_) return -1;
    snaps_.emplace_back();
    id = static_cast<int>(snaps_.size()) - 1;
  }
  SnapRec &r = snaps_[static_cast<size_t>(id)];
  if (r.host.empty())
    for (size_t c = 0; c < cards_.size(); ++c) r.host.push_back(fab_.alloc_host(std::max<size_t>(snap_bytes(static_cast<int>(c)), 4096)));
  r.pos = sq.pos;
  r.qsa_owner=-1;
  if(qsa_resident_snapshots() && trained_qsa3_ && sq.pos>0 && sq.pos%k::kQsaPage==0 &&
     sq.rope_delta==0 && qsa_covered_[slot]==sq.pos) {
    r.qsa_owner=slot;
    std::fprintf(stderr,"QSA_RESIDENT_SAVE id=%d slot=%d pos=%lld\n",id,slot,static_cast<long long>(sq.pos));
  }
  r.parity = sq.parity;
  r.rope_delta = sq.rope_delta;
  r.used = true;
  snap_copy(slot, r, true);
  return id;
}

int Model::snap_prepin(int n) {
  int done = 0;
  while (done < n && static_cast<int>(snaps_.size()) < snap_cap_) {
    snaps_.emplace_back();  // unused: snap_save / snap_take pick it before growing the store
    SnapRec &r = snaps_.back();
    for (size_t c = 0; c < cards_.size(); ++c) r.host.push_back(fab_.alloc_host(std::max<size_t>(snap_bytes(static_cast<int>(c)), 4096)));
    ++done;
  }
  if (done > 0 && static_cast<int>(snaps_.size()) == snap_cap_)
    std::fprintf(stderr, "qwen35 model: prefix cache: all %d snapshot ids' host buffers pinned ahead of use\n", snap_cap_);
  return done;
}

void Model::snap_sync() {
  if(trained_qsa3_) for(size_t c=0;c<cards_.size();++c) fab_.compute(c).wait_and_throw();
  for (auto &q : snap_q_)
    if (q) q->wait_and_throw();
}

void Model::snap_free(int id) {
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen35 model: no such snapshot");
  snaps_[static_cast<size_t>(id)].used = false;  // the host buffers stay for the next one (a later copy queues behind the last)
}

void Model::snap_restore(int slot, int id) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen35 model: no such snapshot");
  if (pf_.active && pf_.slot == slot) throw std::logic_error("qwen35 model: a restore with the slot's prefill call pending");
  SnapRec &r = snaps_[static_cast<size_t>(id)];
  if (kvp_.held_pages(slot) * static_cast<int64_t>(k::kQsaPage) < r.pos)
    throw std::logic_error("qwen35 model: the slot holds no pages for the snapshot's positions");
  save_seq();
  snap_copy(slot, r, false);
  if(trained_qsa3_) {
    // Target KV alone is not proof. Only the original slot's complete groups,
    // never overwritten since the aligned snapshot was recorded, qualify.
    const bool retained=qsa_resident_snapshots() && r.qsa_owner==slot && r.pos>0 &&
        r.pos%k::kQsaPage==0 && r.rope_delta==0;
    qsa_invalidate_snapshots(slot,retained?r.pos:-1);
    qsa_covered_[slot]=retained?r.pos:-1;
    if(qsa_resident_snapshots())
      std::fprintf(stderr,"QSA_RESIDENT_RESTORE id=%d slot=%d pos=%lld retained=%d\n",id,slot,static_cast<long long>(r.pos),int(retained));
  }
  if (mtp_) mtp_->drop(slot);  // deferred drafter KV rows are of positions the restore replaces
  Seq &sq = seqs_[static_cast<size_t>(slot)];
  sq = Seq{};
  self_reset(slot);  // (a restored sequence measures again: the snapshot holds no weights)
  sq.pos = r.pos;
  sq.rope_delta = r.rope_delta;
  sq.last_rows = 4;  // as after a prefill call: committed, the next call / step flips the parity
  sq.parity = r.parity;
  sq.pf_committed = true;
  batch_slots_.erase(std::remove(batch_slots_.begin(), batch_slots_.end(), slot), batch_slots_.end());
  load_seq();
}

std::vector<k::KvPageBuf> Model::kv_page_bufs(int c) const {  // every paged KV buffer of card c (a page copy's set)
  const auto as = attn_.shape_of(c);
  std::vector<k::KvPageBuf> bufs;
  for (int l : attn_.layers) {
    const k::GqaCache &kc = cards_[static_cast<size_t>(c)]->kv[static_cast<size_t>(l)];
    if (kc.ring_pages > 0) continue;  // arch flint: a sliding layer's ring is the slot's own (the snapshot restores it)
    bufs.push_back({kc.k, k::gqa_page_kv_bytes(as)});
    bufs.push_back({kc.v, k::gqa_page_v_bytes(as)});
    if (k::qsa_kv_scaled(as.kv)) {
      bufs.push_back({kc.ks, k::gqa_page_scale_floats(as) * 4});
      bufs.push_back({kc.vs, k::gqa_page_vscale_floats(as) * 4});
    }
    if (kc.kmin) {  // the key bounds
      bufs.push_back({kc.kmin, k::gqa_page_bound_bytes(as) / 2});
      bufs.push_back({kc.kmax, k::gqa_page_bound_bytes(as) / 2});
    }
  }
  if (mtp_)
    for (const auto &b : mtp_->kv_page_bufs(c)) bufs.push_back(b);
  return bufs;
}

void Model::kv_copy_pairs(const std::vector<int32_t> &pairs) {  // [n src pages][n dst pages], every card, waits
  const int n = static_cast<int>(pairs.size() / 2);
  if (n < 1) return;
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    q.memcpy(fork_pages_[c], pairs.data(), pairs.size() * 4);
    k::kv_copy_pages(q, kv_page_bufs(static_cast<int>(c)), fork_pages_[c], fork_pages_[c] + n, n);
  }
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait_and_throw();  // the pairs live in host memory here
}

// Before a slot writes at position pos (its row pos - 1 too: the drafter KV of the row before a call), its shared
// pages there get private copies (copy-on-write; a fork's pages are below a page boundary, so only the last one may
// come here, once).
void Model::kv_cow(int slot, int64_t pos) {
  if (!kvp_.shared()) return;
  if (lend_st_.on && kvp_.limit() < kvp_.pool_pages()) {  // the copies' pages from below the lend limit
    int need = 0;
    for (int i = static_cast<int>(std::max<int64_t>(pos - 1, 0) / k_page_tokens); i < kvp_.held_pages(slot); ++i) need += kvp_.page_shared(slot, i);
    if (need > kvp_.free_below_limit()) lend_raise(need);
  }
  const auto moved = kvp_.unshare(slot, static_cast<int>(std::max<int64_t>(pos - 1, 0) / k_page_tokens));
  if (moved.empty()) return;
  std::vector<int32_t> pairs(2 * moved.size());
  for (size_t i = 0; i < moved.size(); ++i) pairs[i] = moved[i].first, pairs[moved.size() + i] = moved[i].second;
  kv_copy_pairs(pairs);
  kv_upload();
  ++kv_cow_pages_;
}

void Model::snap_fork(int dst, int src, int id) {
  if (dst == src || dst < 0 || src < 0 || dst >= slots() || src >= slots()) throw std::invalid_argument("qwen35 model: a fork takes two slots");
  if (id < 0 || id >= static_cast<int>(snaps_.size()) || !snaps_[static_cast<size_t>(id)].used) throw std::invalid_argument("qwen35 model: no such snapshot");
  const auto &snap=snaps_[static_cast<size_t>(id)];
  static const bool copy_qsa=[] {
    const char *s=std::getenv("XEN_QSA_RESIDENT_FORK_COPY");return s && std::string_view(s)=="1";
  }();
  const bool qsa_fork=copy_qsa && qsa_resident_snapshots() && trained_qsa3_ && snap.qsa_owner==src &&
      snap.pos>0 && snap.pos%k::kQsaPage==0 && snap.rope_delta==0;
  const int n = model::KvPages::pages_for(snap.pos);
  if (kvp_.held_pages(src) < n || kvp_.held_pages(dst) < n) throw std::logic_error("qwen35 model: a fork needs the snapshot's pages on both slots");
  if (mtp_) mtp_->settle(*head_, src, std::numeric_limits<int64_t>::max());  // the source's deferred drafter KV rows written
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait_and_throw();
  // the pages below the snapshot (a page boundary: neither slot writes there again; kv_cow copies a page either one is
  // about to write) are shared, not copied (ModelOptions::kv_share)
  if (opt_.kv_share && kvp_.share(dst, src, n)) {
    kv_upload();
  } else {
    std::vector<int32_t> pairs(2 * static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      pairs[static_cast<size_t>(i)] = kvp_.page(src, i);
      pairs[static_cast<size_t>(n + i)] = kvp_.page(dst, i);
    }
    kv_copy_pairs(pairs);
  }
  snap_restore(dst, id);
  if(qsa_fork) {
    // Native prefix matching and the ordinary target KV fork succeeded. Only
    // then publish copied, immutable, completed-summary groups at dst.
    const auto bytes=trained_qsa3_->copy_aligned_prefix(dst,src,static_cast<int>(snap.pos));
    qsa_covered_[dst]=snap.pos;
    std::fprintf(stderr,"QSA_RESIDENT_FORK id=%d src=%d dst=%d pos=%lld bytes_per_rank=%zu\n",id,src,dst,
      static_cast<long long>(snap.pos),bytes);
  }
}

bool Model::live_ok(int slot) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if (dflash_ || opt_.prefill_slices || (pf_.active && pf_.slot == slot)) return false;
  save_seq();
  const Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (sq.broken) return false;
  if (sq.pf_committed) return true;
  if (sq.last_rows > 0 && sq.pending_accept == 0) return false;  // a step not accepted
  return sq.pending_accept <= k::kGdnRows;                       // the many-row GDN replays up to 4 pending rows
}

void Model::kv_trim(int slot, int64_t tokens) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  if (mtp_) mtp_->settle(*head_, slot, tokens);  // deferred drafter KV rows below the kept pages, the rest dropped
  kvp_.trim(slot, tokens);
  if(trained_qsa3_) {
    // Trimming at or above the covered position leaves every appended group (and the tail) intact: a live
    // resume continues tracking. Below it, only an aligned snapshot restore can re-establish coverage.
    auto &cv=qsa_covered_[slot];
    if(!(qsa_resident_snapshots() && cv>=0 && tokens>=cv)) cv=-1;
    qsa_invalidate_snapshots(slot,tokens);
  }
  kv_upload();
}

void Model::reset() {
  reset_slot(slot_);
  batch_slots_.erase(std::remove(batch_slots_.begin(), batch_slots_.end(), slot_), batch_slots_.end());
  load_seq();
  if (opt_.rope_tier_reset >= 0 && tier_limit_.size() > 1) set_rope_tier(std::min(opt_.rope_tier_reset, rope_tiers() - 1));
}

void Model::note_prompt(int slot,int64_t total,int64_t restored) {
  if(!trained_qsa3_) return;
  // A fresh long prompt starts the index at token 0. A resumed/restored
  // request has no proof of coverage, even if it reuses the same slot ID.
  const bool fresh=restored==0 && position(slot)==0 && total>kTrainedQsaShort;
  const bool retained=qsa_resident_snapshots() && restored>0 && restored%k::kQsaPage==0 &&
      restored==position(slot) && qsa_covered_.at(slot)==restored;
  qsa_covered_.at(slot)=fresh?0:retained?restored:-1;
  static const bool log=util::env_flag("XEN_QSA_TRIAL_ROUTE_LOG");
  if(log) std::fprintf(stderr,"QSA_TRIAL_PROMPT slot=%d tokens=%lld restored=%lld tracking=%d\n",slot,
                       static_cast<long long>(total),static_cast<long long>(restored),int(fresh || retained));
}

void Model::set_rope_tier(int t) {
  if (t < 0 || t >= rope_tiers()) throw std::invalid_argument("qwen35 model: no YaRN tier " + std::to_string(t));
  if (pos_ != 0) throw std::logic_error("qwen35 model: a sequence's YaRN tier is set at position 0");
  if(trained_qsa3_) {qsa_covered_[slot_]=-1;qsa_invalidate_snapshots(slot_);}
  seqs_[static_cast<size_t>(slot_)].rope_delta = tier_base_[static_cast<size_t>(t)];
}

void Model::save_seq() {
  Seq &sq = seqs_[static_cast<size_t>(slot_)];
  sq.pos = pos_;
  sq.last_rows = last_rows_;
  sq.pending_accept = pending_accept_;
  sq.parity = parity_;
  sq.pf_committed = pf_committed_;
  sq.pf_last_row = pf_last_row_;
}

void Model::load_seq() {
  const Seq &sq = seqs_[static_cast<size_t>(slot_)];
  pos_ = sq.pos;
  last_rows_ = sq.last_rows;
  pending_accept_ = sq.pending_accept;
  parity_ = sq.parity;
  pf_committed_ = sq.pf_committed;
  pf_last_row_ = sq.pf_last_row;
}

void Model::use_slot(int slot) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  save_seq();
  slot_ = slot;
  load_seq();
}

int64_t Model::position(int slot) const {
  if (slot == slot_) return pos_;
  return seqs_.at(static_cast<size_t>(slot)).pos;
}

struct Model::StepEvents {
  std::vector<sycl::event> norms;
  std::vector<k::GdnEvents> gev;
  std::vector<k::GqaEvents> aev;
  std::vector<Mlp::Events> mev;
  std::vector<kernels::MoeAffEvents> moev;
};

// One recorded graph per card per key (rows, rows per segment, the attention / MLP / MoE parities at the start); the
// entry keeps the parities after the step (set on replay).
struct Model::Graphs {
  using Exec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  struct Entry {
    std::vector<Exec> exec;  // [card]
    int end_mix = 0, end_mlp = 0, end_moe = 0;
  };
  std::map<std::array<int64_t, 13>, Entry> map;  // see run_graph's key
  std::map<std::array<int64_t, 13>, uint64_t> used;  // LRU stamps
  uint64_t tick = 0;
  int evicted = 0;
  double record_ms = 0;
};

bool Model::graph_step() const {
  if (!sopt_.graphs || opt_.profile) return false;
  if (cfg_.flint) return true;
  // arch qwen35 decode step graphs (opt-in); never with a host hook or a diagnostic that reads / checks on the host
  static const bool on = [] {
    const char *e = std::getenv("XEN_STEP_GRAPHS");
    if (!e || e[0] != '1') return false;
    for (const char *v : {"XEN_QSA_SCORE_CHECK", "XEN_QSA_PACK_CHECK"})
      if (const char *x = std::getenv(v); x && *x) {
        std::fprintf(stderr, "qwen35: XEN_STEP_GRAPHS ignored: %s is set (the step runs eagerly)\n", v);
        return false;
      }
    std::fprintf(stderr, "qwen35: decode step graphs on (XEN_STEP_GRAPHS=1)\n");
    return true;
  }();
  if (!on || attn_q_fn_) return false;
  // Page-selected attention (--attn-select) takes its route from the exact context bound, which the graph key holds
  // only as a class; replays measured different greedy text (under and past the budget), so those steps run eagerly.
  if (attn_select() > 0) {
    static bool told = false;
    if (!told) told = true, std::fprintf(stderr, "qwen35: XEN_STEP_GRAPHS ignored with page-selected attention (--attn-select): the steps run eagerly\n");
    return false;
  }
  return true;
}

int Model::graphs_recorded() const { return graphs_ ? static_cast<int>(graphs_->map.size()) : 0; }

void Model::run_graph(int rows, int seg_rows, int pb) {
  namespace sx = sycl::ext::oneapi::experimental;
  if (!graphs_) graphs_ = std::make_unique<Graphs>();
  const size_t C = cards_.size();
  const int mlp_p = mlp_->parity(), moe_p = moe_ ? moe_->parity() : 0;
  // flint keys and keeps its graphs as before (rows, segment rows, parities; no cap); the rest key every host value
  // that steers a launch, with the attention bound as its kernel-variant class
  const bool fl = cfg_.flint;
  int64_t bound = static_cast<int64_t>(cap_);
  if (pb >= 0 && !fl) {
    // the attention kernel's variant thresholds (gqa.cpp gqa_attn_group / pf_seg): record at the class's top value
    constexpr int64_t kCls[] = {32768, 65536, 131072, 196608};
    bound = static_cast<int64_t>(cap_);
    for (const int64_t t : kCls)
      if (pb < t) { bound = std::min<int64_t>(t - 1, static_cast<int64_t>(cap_)); break; }
  }
  // the GDN replay bound as the core's template classes (a larger bound is exact): 1, 4, 8, 16, at most rows_max_
  const int acc_true = gdn_accept_max_;
  const int acc = cfg_.flint ? acc_true : std::min(rows_max_, acc_true <= 1 ? 1 : acc_true <= 4 ? 4 : acc_true <= 8 ? 8 : 16 > acc_true ? 16 : acc_true);
  const std::array<int64_t, 13> key =
      fl ? std::array<int64_t, 13>{rows, seg_rows, mix_parity_, mlp_p, moe_p}
         : std::array<int64_t, 13>{rows, seg_rows, mix_parity_, mlp_p, moe_p, tree_attn_, tree_gdn_, qsa_step_sparse_,
                                   static_cast<int64_t>(qsa_stage_rows_), static_cast<int64_t>(qsa_dense_rows_), acc, gdn_commit_, bound};
  auto it = graphs_->map.find(key);
  if (it == graphs_->map.end()) {
    static const size_t max = [] { const char *e = std::getenv("XEN_STEP_GRAPHS_MAX"); return e && *e ? size_t(std::max(4, std::atoi(e))) : size_t(32); }();
    const size_t cap = fl ? std::numeric_limits<size_t>::max() : max;
    static const double room = [] { const char *e = std::getenv("XEN_GRAPH_ROOM_MIB"); return e && *e ? std::atof(e) : 128.0; }();
    auto evict_lru = [&](size_t n) {
      for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();  // no evicted graph in flight
      for (size_t i = 0; i < n && !graphs_->used.empty(); ++i) {
        auto lru = graphs_->used.begin();
        for (auto u = graphs_->used.begin(); u != graphs_->used.end(); ++u)
          if (u->second < lru->second) lru = u;
        graphs_->map.erase(lru->first);
        graphs_->used.erase(lru);
        ++graphs_->evicted;
      }
    };
    if (!fl && !graphs_->map.empty() && opt_.loader) {
      double lo = 1e30;
      for (size_t c = 0; c < C; ++c) {
        const auto v = l0::vram_state(opt_.loader->info(fab_.rank_index(c)));
        if (v.ok) lo = std::min(lo, static_cast<double>(v.free) / 1048576.0);
      }
      if (lo < room) {
        static int warned = 0;
        if (warned++ == 0) std::fprintf(stderr, "qwen35: step graphs: %.0f MiB free on the tighter card (< %.0f, XEN_GRAPH_ROOM_MIB): each new graph evicts 4\n", lo, room);
        evict_lru(4);
      }
    }
    if (graphs_->map.size() >= cap) {
      for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();  // no evicted graph in flight
      while (graphs_->map.size() >= cap) {
        auto lru = graphs_->used.begin();
        for (auto u = graphs_->used.begin(); u != graphs_->used.end(); ++u)
          if (u->second < lru->second) lru = u;
        graphs_->map.erase(lru->first);
        graphs_->used.erase(lru);
        if (graphs_->evicted++ == 0) std::fprintf(stderr, "qwen35: step graphs at the cap of %zu (XEN_STEP_GRAPHS_MAX): least recently used evicted from now on\n", cap);
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
    for (size_t c = 0; c < C; ++c) g.emplace_back(fab_.compute(c).get_context(), fab_.compute(c).get_device());
    for (size_t c = 0; c < C; ++c) g[c].begin_recording(fab_.compute(c));
    bool failed = false;
    try {
      gdn_accept_max_ = acc;
      submit_layers(rows, seg_rows, static_cast<int>(bound), nullptr);
      gdn_accept_max_ = acc_true;
    } catch (const std::exception &x) {
      gdn_accept_max_ = acc_true;
      for (size_t c = 0; c < C; ++c) g[c].end_recording();
      std::fprintf(stderr, "qwen35: step graph recording failed (%s): all %zu evicted, this step runs eagerly\n", x.what(), graphs_->map.size());
      failed = true;
    }
    if (failed) {  // back to the start state, then the eager step (memory freed for the next recording)
      mix_parity_ = static_cast<int>(key[2]);
      mlp_->set_parity(mlp_p);
      if (moe_) moe_->set_parity(moe_p);
      evict_lru(graphs_->map.size());
      submit_layers(rows, seg_rows, pb >= 0 ? pb : static_cast<int>(cap_), nullptr);
      return;
    }
    for (size_t c = 0; c < C; ++c) g[c].end_recording();
    Graphs::Entry e;
    e.end_mix = mix_parity_;
    e.end_mlp = mlp_->parity();
    e.end_moe = moe_ ? moe_->parity() : 0;
    mix_parity_ = static_cast<int>(key[2]);  // the recording ran no kernel: back to the start
    mlp_->set_parity(mlp_p);
    if (moe_) moe_->set_parity(moe_p);
    try {
      for (size_t c = 0; c < C; ++c) e.exec.push_back(g[c].finalize());
    } catch (const std::exception &x) {  // the host state is already the start's (restored above)
      std::fprintf(stderr, "qwen35: step graph finalize failed (%s): all %zu evicted, this step runs eagerly\n", x.what(), graphs_->map.size());
      e.exec.clear();
      evict_lru(graphs_->map.size());
      submit_layers(rows, seg_rows, pb >= 0 ? pb : static_cast<int>(cap_), nullptr);
      return;
    }
    it = graphs_->map.emplace(key, std::move(e)).first;
    graphs_->record_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  graphs_->used[key] = ++graphs_->tick;
  // both cards' graphs in flight together: the mailbox waits inside them pair up across the cards
  for (size_t c = 0; c < C; ++c) fab_.compute(c).ext_oneapi_graph(it->second.exec[c]);
  mix_parity_ = it->second.end_mix;
  mlp_->set_parity(it->second.end_mlp);
  if (moe_) moe_->set_parity(it->second.end_moe);
}

void Model::run_layers(int rows, int seg_rows, int pb) {
  prof_ = StepProfile{};
  if (graph_step()) {
    run_graph(rows, seg_rows, pb);
    submit_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_submit_).count();
    head_->wait();
    return;
  }
  const bool pr = opt_.profile;
  StepEvents ev;
  submit_layers(rows, seg_rows, pb, pr ? &ev : nullptr);
  submit_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_submit_).count();
  head_->wait();
  if (!pr) return;
  for (const auto &e : ev.norms) prof_.norms += ev_us(e);
  for (const auto &e : ev.gev) {
    const double p[4] = {ev_us(e.qkvz), ev_us(e.core), ev_us(e.out), ev_us(e.reduce)};
    for (int i = 0; i < 4; ++i) prof_.gdn_parts[i] += p[i], prof_.gdn += p[i];
  }
  for (const auto &e : ev.aev) {
    const double p[6] = {ev_us(e.qkv), ev_us(e.prep), ev_us(e.attn), ev_us(e.merge), ev_us(e.o), ev_us(e.reduce)};
    for (int i = 0; i < 6; ++i) prof_.attn_parts[i] += p[i], prof_.attn += p[i];
  }
  for (const auto &e : ev.mev) {
    const double p[4] = {ev_us(e.gu), e.silu == e.gu ? 0.0 : ev_us(e.silu), ev_us(e.down), ev_us(e.reduce)};  // fused = one event
    for (int i = 0; i < 4; ++i) prof_.mlp_parts[i] += p[i], prof_.mlp += p[i];
  }
  for (const auto &e : ev.moev) {
    const double p[5] = {ev_us(e.router), ev_us(e.route), ev_us(e.gate_up), ev_us(e.down), ev_us(e.combine)};
    for (int i = 0; i < 5; ++i) prof_.moe_parts[i] += p[i], prof_.moe += p[i];
  }
  const auto &he = head_->events(opt_.profile_card);
  prof_.embed = ev_us(he.embed);
  prof_.norms += ev_us(he.norm);
  prof_.head = ev_us(he.lm) + ev_us(he.chunks) + ev_us(he.pick);
  if (head_->mode() == HeadMode::rescore) prof_.head += ev_us(he.selmax) + ev_us(he.sel) + ev_us(he.rescore);  // 
  const auto a = he.embed.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto b = he.pick.get_profiling_info<sycl::info::event_profiling::command_end>();
  prof_.span = static_cast<double>(b - a) * 1e-3;
}

void Model::submit_layers(int rows, int seg_rows, int pb, StepEvents *sev) {
  const size_t C = cards_.size();
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.rms_eps);
  std::vector<const bf16 *> xn(C), mlp_out(C);  // mlp_out: the last layer's MLP (or MoE) output per card
  for (size_t c = 0; c < C; ++c) {
    xn[c] = head_->card(static_cast<int>(c)).xn;
    mlp_out[c] = mlp_->card(static_cast<int>(c)).out;
  }
  std::vector<MlpWeights> mw(C);
  const bool pr = sev != nullptr;
  const int sel_pages = tree_attn_ ? 0 : attn_select();  // a tree step is under the page budget (verify_batch): the dense launch
  // gate|up's own k split (StepOptions::w4_items_gu; TP1: 8), 0 = the auto split of w4_items
  const int gu_ks = gate_up_items() > 0 ? k::w4a16_ksplit_for(2 * mlp_->ipc(), H, gate_up_items()) : 0;
  const bool q_hook = attn_q_fn_ && rows == 1;  // on_attn_q
  std::vector<sycl::event> q_ready(C);
  size_t attn_row = 0;  // the attention layers before layer l
  for (int l = 0; l < cfg_.layers; attn_row += cfg_.full_attention[l] ? 1 : 0, ++l) {
    const bool full = cfg_.full_attention[l];
    for (size_t c = 0; c < C; ++c) {
      auto &q = fab_.compute(c);
      Card &cd = *cards_[c];
      bf16 *res = head_->card(static_cast<int>(c)).res, *x = head_->card(static_cast<int>(c)).xn;
      const k::PairWire *wire = C > 1 ? &cd.wire : nullptr;  // TP1: no two-card sum
      const bool pc = pr && static_cast<int>(c) == opt_.profile_card;
      {
        auto e = k::rms_norm(q, rows, H, res, l > 0 ? mlp_out[c] : nullptr, nullptr, cd.in_norm[l], eps, x, plain_norm());
        if (pc) sev->norms.push_back(e);
      }
      // res now holds layer l - 1's output; DFlash reads some of them (its features of the step's rows)
      if (dflash_ && l > 0 && feat_of_[static_cast<size_t>(l - 1)] >= 0)
        dflash_->capture(static_cast<int>(c), q, feat_of_[static_cast<size_t>(l - 1)], res, rows, dflash_->step_features(static_cast<int>(c)));
      if (full) {
        k::GqaStep gs{cd.seg, pb, k::kSegRec, seg_rows};
        gs.select = sel_pages;
        k::GqaScratch asc = cd.asc;
        if (sel_pages > 0 && cd.sel_slots) {  // select_self: counts per slot, this layer's own score keys
          const size_t hkv = static_cast<size_t>(attn_.shape_of(static_cast<int>(c), l).hkv());
          gs.select_slots = cd.sel_slots + attn_row * static_cast<size_t>(slots()) * hkv;
          asc.sel_key = cd.sel_keys + attn_row * k::kSegs16 * hkv * k::kGqaSelPagesMax;
        }
        gs.attn_pf = sopt_.attn_pf, gs.attn_grf = sopt_.attn_grf;
        gs.tree = tree_attn_;
        k::GqaShape ash = attn_.shape_of(static_cast<int>(c), l);
        ash.dec_qk = sopt_.attn_qk, ash.dec_p_lo = sopt_.attn_p_lo;  // the decode attention numerics
        if (trained_qsa3_ && trained_qsa_layer_[l]>=0 && qsa_stage_rows_)
          trained_qsa3_->stage(static_cast<int>(c),x,rows,ash.hkv(),gs,trained_qsa_layer_[l],l,qsa_stage_rows_,qsa_step_sparse_,qsa_dense_rows_);
        auto ev = k::gqa_decode(q, ash, rows, x, attn_.w[c][l], cd.kv[l], cd.rope, gs, asc, wire, mix_parity_, cd.mix, 0, 0,
                               lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c)));
        if (pc) sev->aev.push_back(ev);
        if (q_hook) q_ready[c] = ev.attn;
      } else {
        k::GdnState st = cd.gdn[l];  // the segment records carry each sequence's accept and parity
        st.seg_rows = seg_rows;
        st.accept_max = gdn_accept_max_;
        st.commit = gdn_commit_;
        st.tree = tree_gdn_, st.tree_S = cd.tree_S;
        auto ev = k::gdn_decode(q, gdn_.shape_of(static_cast<int>(c)), rows, x, gdn_.w[c][l], st, parity_, cd.gsc, wire, mix_parity_, cd.mix);
        if (pc) sev->gev.push_back(ev);
      }
      {
        auto e = k::rms_norm(q, rows, H, res, cd.mix, nullptr, cd.post_norm[l], eps, x, plain_norm());
        if (pc) sev->norms.push_back(e);
      }
    }
    if (q_hook && full) {  // after both cards' launches: each card's q of the row, before the next layer overwrites it
      std::vector<bf16> host;
      for (size_t c = 0; c < C; ++c) {
        const k::GqaShape shape = attn_.shape_of(static_cast<int>(c), l);
        host.resize(static_cast<size_t>(shape.hq()) * shape.head_dim);
        q_ready[c].wait();
        fab_.compute(c).memcpy(host.data(), cards_[c]->asc.q, host.size() * sizeof(bf16)).wait();
        attn_q_fn_(l, static_cast<int>(c), host.data(), shape.hq(), k::seg_row(rec_host_, k::kSegRec, seg_rows, 0).pos);
      }
    }
    mix_parity_ ^= 1;
    if (moe_ && moe_->has(l)) {  // arch flint: the MoE block
      moe_->run(l, rows, xn);
      for (size_t c = 0; c < C; ++c) mlp_out[c] = moe_->out(static_cast<int>(c));
      if (pr) sev->moev.push_back(moe_->events(opt_.profile_card));
      continue;
    }
    for (size_t c = 0; c < C; ++c) mw[c] = mlp_->weights(*dense_, static_cast<int>(c), l);
    mlp_->run(rows, xn, mw, gu_ks, 0, lora_.get(), l);
    for (size_t c = 0; c < C; ++c) mlp_out[c] = mlp_->card(static_cast<int>(c)).out;
    if (pr) sev->mev.push_back(mlp_->events(opt_.profile_card));
  }
  head_->final_norm(rows, mlp_out);
  head_->lm_pick(rows);
}

void Model::step(const int32_t *ids, int rows) {
  if (rows < 1 || rows > rows_max_) throw std::invalid_argument("qwen35 model: rows 1..4 (1..8 with DFlash, 16 with its 16-row trees)");
  verify_batch({SeqRows{slot_, ids, rows}});
}

void Model::accept(int n) {
  if (batch_slots_.size() != 1 || batch_slots_[0] != slot_) throw std::logic_error("qwen35 model: accept(n) after a step of the current slot only");
  accept_batch({n});
}

// ---- draft trees ----
void Model::note_tree(int slot, size_t drafts, int seg) {
  if (draft_par_.size() != static_cast<size_t>(slots())) draft_par_.resize(static_cast<size_t>(slots()));
  auto &p = draft_par_[static_cast<size_t>(slot)];
  p.clear();
  if (!dflash_ || drafts == 0) return;
  const auto &t = dflash_->tree_parents(seg);
  bool chain = true;
  for (size_t r = 1; r <= drafts && r < t.size(); ++r) chain &= t[r] == static_cast<int>(r) - 1;
  if (t.size() >= drafts + 1 && !chain) p.assign(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(drafts + 1));
}

void Model::step_tree(const int32_t *ids, const int *parent, int rows) {
  if (parent[0] != -1) throw std::invalid_argument("qwen35 model: a tree's row 0 is its root");
  verify_batch({SeqRows{slot_, ids, rows, parent}});
}

void Model::accept_tree(const int *path, int n) {
  if (!last_tree_ || batch_slots_.size() != 1 || batch_slots_[0] != slot_) throw std::logic_error("qwen35 model: accept_tree() after step_tree() of the current slot");
  accept_batch({n}, {path});
}

// The accept of a step whose segments may be trees. Each tree segment's walked rows become its positions p0 .. p0 + n
// - 1 (their K / V moved there; the GDN commits along the path at the next step), and the step's rows are re-laid per
// segment in path order (row r0 + i = r0 + path[i], rows past n repeat the last) for draft_batch.
void Model::accept_batch(const std::vector<int> &n, const std::vector<const int *> &paths) {
  const std::vector<int64_t> pos_before = step_positions();
  accept_paths(n, paths);
  close_pages(pos_before);
}

void Model::accept_paths(const std::vector<int> &n, const std::vector<const int *> &paths) {
  if (paths.size() != n.size() || n.size() != batch_slots_.size()) throw std::logic_error("qwen35 model: accept_batch takes one count and path per segment of the last step");
  const size_t ns = n.size();
  std::vector<int64_t> p0(ns);
  std::vector<int> rows(ns);
  save_seq();
  for (size_t i = 0; i < ns; ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    p0[i] = sq.pos, rows[i] = sq.last_rows;
    if (!paths[i]) continue;
    if (i >= batch_tree_.size() || !batch_tree_[i]) throw std::logic_error("qwen35 model: a path for a segment whose rows were not a tree");
    const int *path = paths[i];
    if (n[i] < 1 || n[i] > rows[i] || path[0] != 0) throw std::invalid_argument("qwen35 model: accept_tree takes 1..rows rows from the root");
    for (int r = 1; r < n[i]; ++r)
      if (path[r] <= path[r - 1] || path[r] >= rows[i]) throw std::invalid_argument("qwen35 model: accept_tree's rows descend the tree");
  }
  load_seq();
  // a tree segment accepted without its path: only its root (row 0 is the chain's row 0 either way)
  for (size_t i = 0; i < ns; ++i)
    if (!paths[i] && i < batch_tree_.size() && batch_tree_[i] && n[i] > 1) throw std::logic_error("qwen35 model: accept_tree() after step_tree()");
  accept_counts(n, &paths);  // position() += n (pending_accept = n: the next step's GDN replays n rows, along the path)
  last_tree_ = false;
  save_seq();
  bool any = false;
  for (size_t i = 0; i < ns; ++i) {
    if (!paths[i]) continue;
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    sq.path = k::tree_path_word(paths[i], n[i]);
    any |= sq.path != 0;
  }
  load_seq();
  if (!any) return;  // every path is rows 0 .. n - 1: every row already sits where a chain puts it
  const int R = batch_total_, H = cfg_.hidden;
  int map[k::kMaxRows] = {};  // (the kernels' row capacity: df_rows_gather's argument)
  for (int r = 0; r < k::kMaxRows; ++r) map[r] = std::min(r, R - 1);
  for (size_t i = 0, r0 = 0; i < ns; r0 += static_cast<size_t>(rows[i]), ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    if (!paths[i] || sq.path == 0) continue;
    for (int r = 0; r < rows[i]; ++r) map[r0 + static_cast<size_t>(r)] = static_cast<int>(r0) + paths[i][std::min(r, n[i] - 1)];
  }
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    Card &cd = *cards_[c];
    const AttnCaches attn = attn_caches(c);
    const std::vector<k::GqaCache> &cs = attn.caches;
    if (!cs.empty()) {
      const k::GqaShape ash = attn_.shape_of(static_cast<int>(c), attn.layer0);
      for (size_t i = 0; i < ns; ++i) {
        if (!paths[i] || seqs_[static_cast<size_t>(batch_slots_[i])].path == 0) continue;
        k::GqaMove mv;
        for (int x = 0; x < n[i]; ++x) mv.from[x] = paths[i][x], mv.to[x] = x;
        mv.n = n[i];
        for (size_t j = 0; j < cs.size(); j += k::kGqaMoveCaches)
          k::gqa_kv_move(q, ash, cs.data() + j, static_cast<int>(std::min<size_t>(k::kGqaMoveCaches, cs.size() - j)), batch_slots_[i], static_cast<int>(p0[i]), mv);
      }
    }
    // the head's rows (MTP's pass 1 reads them) in path order
    bf16 *xn = head_->card(static_cast<int>(c)).xn;
    k::df_rows_gather(q, R, H, xn, cd.tree_rows, map);
    q.memcpy(xn, cd.tree_rows, static_cast<size_t>(R) * H * 2);
  }
  if (dflash_) {  // the verify's feature rows in path order
    dflash_->save_step_features(R);
    dflash_->stage_step_features(map, R);
  }
}

void Model::verify_batch(const std::vector<SeqRows> &segs) {
  const int ns = static_cast<int>(segs.size());
  if (ns < 1 || ns > slots()) throw std::invalid_argument("qwen35 model: 1..slots segments per step");
  const int sr = segs[0].rows;
  // unequal segments, the engine's layouts: k_wide segments of sr rows, then one-row segments (kernels::seg_het); every
  // other layout has equal rows. Or two classes, the rest b2 rows each (1 <= b2 < sr; kernels::seg_het2)
  int k_wide = 0, rows = 0;
  while (k_wide < ns && segs[static_cast<size_t>(k_wide)].rows == sr) ++k_wide;
  for (const auto &g : segs) rows += g.rows;
  const int b2 = k_wide < ns ? segs[static_cast<size_t>(k_wide)].rows : 1;
  const int code = k_wide == ns ? sr : k::seg_het2(sr, k_wide, ns, b2);
  if (sr < 1 || sr > rows_max_ || rows > step_rows_)
    throw std::invalid_argument("qwen35 model: 1..4 rows per segment (8 with DFlash, 16 with its 16-row trees), 1.." + std::to_string(step_rows_) + " in all");
  save_seq();
  for (int i = 0; i < ns; ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    if (g.rows != (i < k_wide ? sr : b2) || b2 < 1 || b2 > sr)
      throw std::invalid_argument("qwen35 model: the segments of a step have the same rows, or B rows each and then B2 <= B rows each");
    if (g.slot < 0 || g.slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
    for (int j = 0; j < i; ++j)
      if (segs[static_cast<size_t>(j)].slot == g.slot) throw std::invalid_argument("qwen35 model: a slot appears twice in one step");
    const Seq &sq = seqs_[static_cast<size_t>(g.slot)];
    if (sq.last_rows > 0 && sq.pending_accept == 0 && !sq.pf_committed) throw std::logic_error("qwen35 model: accept() the previous step first");
    if (sq.pos + g.rows > cap_) throw std::runtime_error("qwen35 model: the step runs past the KV capacity");
  }
  // segments whose rows are a draft tree (their records carry the tree words)
  std::vector<uint8_t> is_tree(static_cast<size_t>(ns), 0);
  for (int i = 0; i < ns; ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    if (!g.parent) continue;
    if (g.rows > std::min(k::kTreeRows, rows_max_) || !cards_[0]->tree_S)
      throw std::invalid_argument("qwen35 model: a tree step holds 1..8 rows per segment (DFlash's verify; 16 with the 16-row trees)");
    if (g.parent[0] != -1) throw std::invalid_argument("qwen35 model: a tree's row 0 is its root");
    for (int r = 1; r < g.rows; ++r)
      if (g.parent[r] < 0 || g.parent[r] >= r) throw std::invalid_argument("qwen35 model: a tree row's parent comes before it");
    is_tree[static_cast<size_t>(i)] = 1;
  }
  for (int i = 0; i < ns; ++i)
    if (is_tree[static_cast<size_t>(i)]) {
      const auto &g = segs[static_cast<size_t>(i)];
      Seq &sq = seqs_[static_cast<size_t>(g.slot)];
      sq.tree = k::tree_word(g.parent, g.rows);
      sq.tree_aux = k::tree_aux_word(g.parent, g.rows);
    }
  for (const auto &g : segs) kv_need(g.slot, seqs_[static_cast<size_t>(g.slot)].pos + g.rows);  // the rows' pages
  std::vector<int64_t> pos_before(static_cast<size_t>(ns));
  for (int i = 0; i < ns; ++i) pos_before[static_cast<size_t>(i)] = seqs_[static_cast<size_t>(segs[static_cast<size_t>(i)].slot)].pos;
  for (const auto &g : segs) kv_cow(g.slot, seqs_[static_cast<size_t>(g.slot)].pos);  // copy-on-write of shared pages
  int32_t tokens[k::kStepRows32], rec[kRecInts] = {};
  int64_t need = 0;
  int amax = 0;
  tree_attn_ = tree_gdn_ = false;
  last_tree_ = false;
  qsa_stage_rows_=0;
  qsa_step_sparse_=false;
  qsa_dense_rows_=0;
  qsa_sparse_slots_=0;
  uint32_t qsa_ok_rows=0;  // rows of segments eligible for the trained sparse walk
  for (int i = 0, r0 = 0; i < ns; r0 += segs[static_cast<size_t>(i)].rows, ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    Seq &sq = seqs_[static_cast<size_t>(g.slot)];
    if(trained_qsa3_) {
      auto &covered=qsa_covered_[g.slot];
      if(covered!=sq.pos || sq.rope_delta!=0) covered=-1;
      if(covered>=0) for(int r=r0;r<r0+g.rows;++r) qsa_stage_rows_|=uint32_t(1)<<r;
      if(covered>=std::max<int64_t>(kTrainedQsaShort,qsa_decode_min_pos()))
        for(int r=r0;r<r0+g.rows;++r) qsa_ok_rows|=uint32_t(1)<<r;
    }
    for (int r = 0; r < g.rows; ++r) tokens[r0 + r] = g.tokens[r];
    const int a = sq.pf_committed ? 0 : (sq.last_rows > 0 ? sq.pending_accept : 0);
    if (sq.last_rows > 0) sq.parity ^= 1;  // the GDN's pending log of the previous step is the other parity
    int32_t *w = rec + i * k::kSegRec;
    w[k::kSegPos] = static_cast<int32_t>(sq.pos);
    w[k::kSegAccept] = a;
    amax = std::max(amax, a);
    w[k::kSegParity] = sq.parity;
    w[k::kSegRope] = static_cast<int32_t>(sq.rope_delta);  // GqaShape::mrope
    w[k::kSegSlot] = g.slot;
    // a tree (step_tree) and the last accepted tree's path (the GDN's replay of it)
    k::rec_word_set(w, k::kSegTree, k::kSegTreeHi, sq.tree);
    k::rec_word_set(w, k::kSegTreeAux, k::kSegTreeAux2, sq.tree_aux);
    k::rec_word_set(w, k::kSegPath, k::kSegPathHi, a > 0 ? sq.path : 0);
    tree_attn_ = tree_attn_ || sq.tree != 0;
    tree_gdn_ = tree_gdn_ || sq.tree != 0 || (a > 0 && sq.path != 0);
    sq.tree = sq.tree_aux = sq.path = 0;
    need = std::max(need, sq.pos + g.rows);
    sq.last_rows = g.rows;
    sq.pending_accept = 0;
    sq.pf_committed = false;
  }
  t_submit_ = std::chrono::steady_clock::now();
  for (auto &cd : cards_) cd->wire.bf16 = sopt_.wire_bf16, cd->wire.i8 = sopt_.wire_i8;
  mlp_->set_wire_bf16(sopt_.wire_bf16);
  mlp_->set_wire_i8(sopt_.wire_i8);
  if (moe_) moe_->set_wire_bf16(sopt_.wire_bf16);
  mlp_->set_fused(sopt_.fused_swiglu);
  head_->set_ids(tokens, rows);  // waits for the cards: the pinned records are free
  std::memcpy(rec_host_, rec, sizeof(rec));
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).memcpy(cards_[c]->seg, rec_host_, static_cast<size_t>(ns) * k::kSegRec * 4);
  head_->embed(rows);
  // The GDN core stages only as many of the previous step's rows as any segment accepts, and a 1-row step's rows are
  // final (never speculative: accept_batch takes 1..rows), so the core commits S and the window after them and the next
  // step replays nothing (pf_committed, as after a prefill).
  gdn_accept_max_ = amax;
  gdn_commit_ = sr == 1 && k_wide == ns;  // (an unequal step has sr >= 2: nothing committed, its 1-row segments replay)
  k::w4a16_set_min_items(sopt_.w4_items);
  const int64_t pos_bound = std::min<int64_t>(cap_, need);
  if (tree_attn_ && selects_pages(pos_bound))  // the engine asks first (StepModelAdapter::verifies_trees)
    throw std::invalid_argument("qwen35 model: a tree step attends densely; this step passes the page budget of the selected attention");
  const int sel_pages = tree_attn_ ? 0 : attn_select();  // (submit_layers' rule: a tree step attends densely)
  self_sync(sel_pages);
  // Per-segment choice: an eligible segment (covered, long, at most 16 tree rows) takes the trained sparse walk;
  // every other row of the step gets plan count 0, the dense walk bit for bit. Trees need the tree opt-in.
  static const bool tree_qsa=[] {
    const char *p=std::getenv("XEN_TRAINED_QSA_TREE"); return p && p[0]=='1' && p[1]=='\0';
  }();
  if(trained_qsa3_ && (!tree_attn_ || tree_qsa))
    for(int i=0,r0=0;i<ns;r0+=segs[static_cast<size_t>(i)].rows,++i) {
      const auto &g=segs[static_cast<size_t>(i)];
      const uint32_t mask=(g.rows>=32?~uint32_t(0):((uint32_t(1)<<g.rows)-1))<<r0;
      if((qsa_ok_rows&mask)==mask && (!tree_attn_ || g.rows<=k::kTreeRows)) {
        qsa_step_sparse_=true;
        if(g.slot<64) qsa_sparse_slots_|=uint64_t(1)<<g.slot;
      } else qsa_dense_rows_|=mask;
    }
  if(!qsa_step_sparse_) qsa_dense_rows_=0;
  static const bool qsa_log=util::env_flag("XEN_QSA_TRIAL_ROUTE_LOG");
  if(qsa_log && trained_qsa3_) for(const auto &g:segs) {
    const auto pos=seqs_[g.slot].pos, covered=qsa_covered_[g.slot];
    std::fprintf(stderr,"QSA_TRIAL_ROUTE slot=%d pos=%lld rows=%d covered=%lld route=%s reason=%s\n",g.slot,
      static_cast<long long>(pos),g.rows,static_cast<long long>(covered),((qsa_sparse_slots_>>g.slot)&1)?"sparse":"dense",
      ((qsa_sparse_slots_>>g.slot)&1)?(tree_attn_?"covered-tree":"covered"):pos<kTrainedQsaShort?"short":covered!=pos?"incomplete":tree_attn_?"tree":"mixed");
  }
  run_layers(rows, code, static_cast<int>(pos_bound));
  if (sel_pages > 0 && selects_pages(pos_bound)) self_measure(segs, pos_before, sel_pages);
  gdn_accept_max_ = rows_max_;
  gdn_commit_ = false;
  tree_attn_ = tree_gdn_ = false;
  if (sr == 1)
    for (const auto &g : segs) seqs_[static_cast<size_t>(g.slot)].pf_committed = true;
  batch_slots_.clear();
  for (const auto &g : segs) batch_slots_.push_back(g.slot);
  batch_rows_ = sr;
  batch_code_ = code;
  batch_total_ = rows;
  batch_tree_ = is_tree;
  last_tree_ = std::any_of(is_tree.begin(), is_tree.end(), [](uint8_t t) { return t != 0; });
  load_seq();
}

void Model::accept_batch(const std::vector<int> &n) {
  // a tree segment kept past its root needs its path (accept_batch(n, paths))
  for (size_t i = 0; last_tree_ && i < n.size() && i < batch_tree_.size(); ++i)
    if (batch_tree_[i] && n[i] > 1) throw std::logic_error("qwen35 model: accept_tree() after step_tree()");
  const std::vector<int64_t> pos_before = step_positions();
  accept_counts(n);
  close_pages(pos_before);
}

Model::AttnCaches Model::attn_caches(size_t card) const {
  AttnCaches a;
  for (int l = 0; l < cfg_.layers; ++l) {
    if (!cfg_.full_attention[l]) continue;
    a.caches.push_back(cards_[card]->kv[static_cast<size_t>(l)]);
    if (a.layer0 < 0) a.layer0 = l;
  }
  return a;
}

std::vector<int64_t> Model::step_positions() {
  save_seq();  // the current slot's position into seqs_
  std::vector<int64_t> pos(batch_slots_.size());
  for (size_t i = 0; i < pos.size(); ++i) pos[i] = seqs_[static_cast<size_t>(batch_slots_[i])].pos;
  return pos;
}

int Model::read_keys(int slot, int layer, int card, int pages, std::vector<int8_t> &keys, std::vector<float> &scales) {
  if (layer < 0 || layer >= cfg_.layers || !cfg_.full_attention[layer]) throw std::invalid_argument("qwen35 model: read_keys of a layer without attention");
  const k::GqaShape shape = attn_.shape_of(card, layer);
  if (shape.kv != k::QsaKv::i8) throw std::logic_error("qwen35 model: read_keys reads the int8 KV");
  auto &q = fab_.compute(static_cast<size_t>(card));
  q.wait_and_throw();
  const k::GqaCache &kc = cards_[static_cast<size_t>(card)]->kv[layer];
  const size_t hkv = static_cast<size_t>(shape.hkv()), page_k = hkv * k::kQsaPage * static_cast<size_t>(shape.head_dim), page_s = hkv * k::kQsaPage;
  std::vector<int32_t> table(static_cast<size_t>(pages));
  q.memcpy(table.data(), kc.pages + static_cast<size_t>(slot) * kc.slot_pages, table.size() * sizeof(int32_t)).wait();
  keys.resize(table.size() * page_k);
  scales.resize(table.size() * page_s);
  for (size_t p = 0; p < table.size(); ++p) {
    const size_t at = static_cast<size_t>(table[p]);
    q.memcpy(keys.data() + p * page_k, reinterpret_cast<const int8_t *>(kc.k) + at * page_k, page_k);
    q.memcpy(scales.data() + p * page_s, kc.ks + at * page_s, page_s * sizeof(float));
  }
  q.wait_and_throw();
  return static_cast<int>(hkv);
}

int64_t Model::page_bound_errors(int slot, std::span<const int> pages) {
  if (!opt_.page_bounds()) throw std::logic_error("qwen35 model: page_bound_errors needs page selection (the caches' bounds)");
  int64_t errors = 0;
  for (size_t c = 0; c < cards_.size(); ++c) {
    auto &q = fab_.compute(c);
    q.wait_and_throw();
    const AttnCaches attn = attn_caches(c);
    if (attn.caches.empty()) continue;
    const k::GqaShape shape = attn_.shape_of(static_cast<int>(c), attn.layer0);
    const size_t hkv = static_cast<size_t>(shape.hkv()), dims = static_cast<size_t>(shape.head_dim), tokens = k::kQsaPage;
    const bool scaled = k::qsa_kv_scaled(shape.kv);
    std::vector<uint8_t> keys(tokens * dims * static_cast<size_t>(shape.kv_elem()));
    std::vector<float> scales(tokens, 1.f);
    std::vector<bf16> stored_min(dims), stored_max(dims);
    for (const k::GqaCache &kc : attn.caches) {
      for (const int page : pages) {
        int32_t table_entry = 0;
        q.memcpy(&table_entry, kc.pages + static_cast<size_t>(slot) * kc.slot_pages + page, sizeof(table_entry)).wait();
        for (size_t kvh = 0; kvh < hkv; ++kvh) {
          const size_t unit = static_cast<size_t>(table_entry) * hkv + kvh;  // the kernels' (physical page, kv head)
          q.memcpy(keys.data(), kc.k + unit * keys.size(), keys.size());
          if (scaled) q.memcpy(scales.data(), kc.ks + unit * tokens, tokens * sizeof(float));
          q.memcpy(stored_min.data(), kc.kmin + unit * dims, dims * sizeof(bf16));
          q.memcpy(stored_max.data(), kc.kmax + unit * dims, dims * sizeof(bf16)).wait();
          bool same = true;
          for (size_t d = 0; d < dims && same; ++d) {
            float lo = INFINITY, hi = -INFINITY;
            for (size_t t = 0; t < tokens; ++t) {
              const size_t e = t * dims + d;
              const float v = scaled ? static_cast<float>(reinterpret_cast<const int8_t *>(keys.data())[e]) * scales[t]
                                     : static_cast<float>(reinterpret_cast<const bf16 *>(keys.data())[e]);
              lo = std::min(lo, v), hi = std::max(hi, v);
            }
            const bf16 want_min(lo), want_max(hi);
            same = std::memcmp(&want_min, &stored_min[d], sizeof(bf16)) == 0 && std::memcmp(&want_max, &stored_max[d], sizeof(bf16)) == 0;
          }
          errors += !same;
        }
      }
    }
  }
  return errors;
}

void Model::close_pages(std::span<const int64_t> pos_before) {
  if (!opt_.page_bounds()) return;  // the caches keep no bounds
  save_seq();
  for (size_t i = 0; i < batch_slots_.size() && i < pos_before.size(); ++i) {
    const int slot = batch_slots_[i];
    const int64_t page = pos_before[i] / k::kQsaPage;
    if (seqs_[static_cast<size_t>(slot)].pos / k::kQsaPage == page) continue;  // still writing the page
    for (size_t c = 0; c < cards_.size(); ++c) {
      const AttnCaches attn = attn_caches(c);
      if (attn.caches.empty()) continue;
      const k::GqaShape shape = attn_.shape_of(static_cast<int>(c), attn.layer0);
      for (size_t j = 0; j < attn.caches.size(); j += k::kGqaMoveCaches)
        k::gqa_bounds_page(fab_.compute(c), shape, attn.caches.data() + j, static_cast<int>(std::min<size_t>(k::kGqaMoveCaches, attn.caches.size() - j)), slot,
                           static_cast<int>(page));
    }
  }
}

void Model::accept_counts(const std::vector<int> &n, const std::vector<const int *> *paths) {
  if (batch_slots_.empty() || n.size() != batch_slots_.size()) throw std::logic_error("qwen35 model: accept_batch takes one count per segment of the last step");
  save_seq();
  for (size_t i = 0; i < n.size(); ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    if (sq.pending_accept != 0 || n[i] < 1 || n[i] > sq.last_rows) throw std::logic_error("qwen35 model: accept 1..rows once per step");
  }
  int trained_row0=0;
  for (size_t i = 0; i < n.size(); ++i) {
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    if (trained_qsa3_) {
      auto &covered=qsa_covered_[batch_slots_[i]];
      if(covered==sq.pos && (qsa_stage_rows_&(uint32_t(1)<<trained_row0))) {
        const int *path=paths?(*paths)[i]:nullptr;
        trained_qsa3_->accept(batch_slots_[i],static_cast<int>(sq.pos),trained_row0,n[i],last_tree_,
                              path?k::tree_path_word(path,n[i]):0);
        covered+=n[i];
      } else covered=-1;
    }
    trained_row0+=sq.last_rows;
    sq.pending_accept = n[i];
    sq.pos += n[i];
  }
  load_seq();
}

void Model::apply_a8_keep(const std::string &spec) {
  std::vector<const uint8_t *> keep;
  const size_t C = fab_.size();  // cards_ is filled later in the constructor
  const int L = cfg_.layers;
  size_t p = 0;
  while (p <= spec.size()) {
    const size_t e = std::min(spec.find_first_of(",+", p), spec.size());
    const std::string item = spec.substr(p, e - p);
    p = e + 1;
    if (item.empty()) continue;
    const size_t at = item.find('@');
    const std::string fam = item.substr(0, at);
    int l0 = 0, l1 = L - 1;
    if (at != std::string::npos) {
      const std::string r = item.substr(at + 1);
      const size_t dash = r.find('-');
      try {
        l0 = std::stoi(r.substr(0, dash));
        l1 = dash == std::string::npos ? l0 : std::stoi(r.substr(dash + 1));
      } catch (const std::exception &) {
        l0 = -1;
      }
      if (l0 < 0 || l1 < l0 || l1 >= L) throw std::invalid_argument("qwen35 model: prefill_a8_keep: bad layer range in " + item);
    }
    const bool qkvz = fam == "qkvz" || fam == "gdn" || fam == "all", out = fam == "out" || fam == "gdn" || fam == "all";
    const bool qkv = fam == "qkv" || fam == "attn" || fam == "all", o = fam == "o" || fam == "attn" || fam == "all";
    const bool gu = fam == "gu" || fam == "mlp" || fam == "all", down = fam == "down" || fam == "mlp" || fam == "all";
    if (!(qkvz || out || qkv || o || gu || down))
      throw std::invalid_argument("qwen35 model: prefill_a8_keep: unknown family " + fam + " (qkvz out gdn qkv o attn gu down mlp all)");
    for (int l = l0; l <= l1; ++l)
      for (size_t c = 0; c < C; ++c) {
        if (cfg_.full_attention[l]) {
          if (qkv) keep.push_back(attn_.w[c][l].qkv.w);
          if (o) keep.push_back(attn_.w[c][l].o.w);
        } else {
          if (qkvz) keep.push_back(gdn_.w[c][l].qkvz4.w);
          if (out) keep.push_back(gdn_.w[c][l].out4.w);
        }
        const MlpWeights mw = mlp_->weights(*dense_, static_cast<int>(c), l);
        if (gu) keep.push_back(mw.gate_up.w);
        if (down) keep.push_back(mw.down.w);
      }
  }
  k::w4a16_pf_a8_keep_clear();
  for (const uint8_t *w : keep)
    if (w) k::w4a16_pf_a8_keep(w);
  opt_.prefill_a8_keep = spec;
}

void Model::check_attn_qk(const ModelOptions &o) const {  // 
  if (o.prefill_attn < k::kGqaAttnBatch || o.prefill_attn > k::kGqaAttnSdpa) throw std::invalid_argument("qwen35 model: prefill_attn 0..4");
  if (o.prefill_attn == k::kGqaAttnSdpa && (opt_.kv != k::QsaKv::i8 && opt_.kv != k::QsaKv::bf16))
    throw std::invalid_argument("qwen35 model: the oneDNN attention (prefill_attn 4) takes int8 or bf16 KV");
  if (o.prefill_attn_qk < 0 || o.prefill_attn_qk > 2) throw std::invalid_argument("qwen35 model: prefill_attn_qk 0..2");
  if (o.prefill_attn_qk && (opt_.kv == k::QsaKv::bf16 || o.prefill_attn != k::kGqaAttnFa))
    throw std::invalid_argument("qwen35 model: int8 q.k (prefill_attn_qk) needs the int8 KV and the fa attention body");
  if (o.prefill_attn_qk == 2 && o.prefill_attn_p_lo) throw std::invalid_argument("qwen35 model: prefill_attn_qk 2 runs with prefill_attn_p_lo false");
}

void Model::check_prefill_select(const ModelOptions &o) const {
  if (o.prefill_select == 0) return;
  if (o.prefill_select < 32 || o.prefill_select > k::kGqaSelMax) throw std::invalid_argument("qwen35 model: prefill_select 0 or 32..4096 pages");
  if (cfg_.flint || cfg_.llama())
    throw std::invalid_argument("qwen35 model: page-selected prefill (prefill_select) is built for qwen35 (6 q heads per kv head, head_dim 256)");
  if (o.prefill_slices || o.prefill_attn_slices || o.prefill_attn != k::kGqaAttnFa)
    throw std::invalid_argument("qwen35 model: page-selected prefill (prefill_select) runs the many-row fa attention body (prefill_attn 3, no slices)");
  const bool exact = o.prefill_attn_qk == 0 && o.prefill_attn_p_lo, hi_only = o.prefill_attn_qk == 0 && !o.prefill_attn_p_lo;
  const bool int8_fast = o.prefill_attn_qk == 1 && !o.prefill_attn_p_lo;
  if (!(exact || (o.kv != k::QsaKv::bf16 ? int8_fast : hi_only)))
    throw std::invalid_argument("qwen35 model: page-selected prefill (prefill_select) runs the exact attention numerics, int8 q.k with P hi only on the int8 / int8-v4 KV "
                                "(int8-fast) or P hi only on the bf16 KV (fast)");
}

void Model::check_decode_attn(const StepOptions &o) const {
  if (o.attn_qk == 0 && o.attn_p_lo) return;
  if (cfg_.flint || cfg_.llama()) throw std::invalid_argument("qwen35 model: the decode attention numerics (attn_qk / attn_p_lo) are built for qwen35 (grp 6, head_dim 256)");
  if (o.attn_qk < 0 || o.attn_qk > 2) throw std::invalid_argument("qwen35 model: attn_qk 0..2");
  if (o.attn_qk && opt_.kv == k::QsaKv::bf16) throw std::invalid_argument("qwen35 model: int8 decode q.k (attn_qk) needs the int8 KV");
  if (o.attn_qk == 2 && o.attn_p_lo) throw std::invalid_argument("qwen35 model: attn_qk 2 runs with attn_p_lo false");
  const bool int8_fast = o.attn_qk == 1 && !o.attn_p_lo;  // besides exact, the one mode the page-selected kernels are built for
  if ((o.attn_select < 0 ? opt_.attn_select : o.attn_select) > 0 && !int8_fast)
    throw std::invalid_argument("qwen35 model: page selection runs the exact or the int8-fast decode attention numerics");
}

void Model::set_prefill_options(const ModelOptions &o) {
  check_attn_qk(o);
  {
    ModelOptions as_loaded = o;  // the KV dtype and the slices are the load's
    as_loaded.kv = opt_.kv, as_loaded.prefill_slices = opt_.prefill_slices;
    check_prefill_select(as_loaded);
  }
  if (o.prefill_select > 0 && !cards_.empty() && !cards_[0]->pf_gqa.sel.valid())
    throw std::invalid_argument("qwen35 model: prefill_select needs its key bounds and page-set scratch: load with prefill_select > 0");
  if (o.prefill_attn == k::kGqaAttnSdpa && !cards_.empty() && !cards_[0]->pf_gqa.sdpa_kv)
    throw std::invalid_argument("qwen35 model: prefill_attn 4 (oneDNN) needs its scratch: load with prefill_attn 4");
  if (o.prefill_chunk_run < 0 || o.prefill_chunk_run % 4 || o.prefill_chunk_run > opt_.prefill_chunk)
    throw std::invalid_argument("qwen35 model: prefill_chunk_run is a multiple of 4 up to prefill_chunk");
  if (o.prefill_short < 0 || o.prefill_short > rows_cap_)
    throw std::invalid_argument("qwen35 model: prefill_short 0.." + std::to_string(rows_cap_) + " rows (the load's step buffers: load with it or 32-row steps)");
  opt_.prefill_attn = o.prefill_attn;
  opt_.prefill_attn_rows = o.prefill_attn_rows;
  opt_.prefill_attn_p_lo = o.prefill_attn_p_lo;
  opt_.prefill_attn_qk = o.prefill_attn_qk;
  opt_.prefill_select = o.prefill_select;
  opt_.prefill_w4_dequant = o.prefill_w4_dequant;
  k::w4a16_pf_set_dequant(o.prefill_w4_dequant);
  if (o.prefill_w4_a8)  // the scratch is reserved at load when W4A8 was on then
    for (size_t c = 0; c < cards_.size(); ++c) k::w4a16_pf_a8_reserve(fab_.compute(c), opt_.prefill_chunk, std::max(cfg_.hidden, cfg_.inter));
  apply_a8_keep(o.prefill_a8_keep);  // (throws before anything else changes)
  opt_.prefill_w4_a8 = o.prefill_w4_a8;
  k::w4a16_pf_set_a8(o.prefill_w4_a8);
  opt_.prefill_a8_body = o.prefill_a8_body;
  k::w4a16_pf_set_a8_body(a8_body_for(o.prefill_a8_body));
  opt_.prefill_short = o.prefill_short;
  k::w4a16_pf_set_short_rows(o.prefill_short);
  opt_.prefill_attn_slices = o.prefill_attn_slices;
  opt_.prefill_split = o.prefill_split;
  opt_.prefill_wire_bf16 = o.prefill_wire_bf16;
  opt_.rope_tier_reset = o.rope_tier_reset;  // the next reset()'s YaRN tier
  for (auto &cd : cards_) cd->pf_split.bf16 = cd->pf_wire.bf16 = o.prefill_wire_bf16;
  opt_.prefill_send = o.prefill_send;
  opt_.prefill_defer = o.prefill_defer;
  opt_.prefill_fused_swiglu = o.prefill_fused_swiglu;
  opt_.prefill_embed_chunk = o.prefill_embed_chunk;
  opt_.prefill_chunk_run = o.prefill_chunk_run;
  if (cfg_.llama()) llama_prefill(opt_);
  if (cfg_.flint) flint_options(opt_);
}

void Model::prefill(const int32_t *ids, int n, const std::function<void(int64_t, int)> &on_rows, int32_t next) {
  if (n < 1) throw std::invalid_argument("qwen35 model: prefill of no tokens");
  if (last_rows_ > 0 && pending_accept_ == 0 && !pf_committed_) throw std::logic_error("qwen35 model: accept() the previous step first");
  if (pos_ + n > cap_) throw std::runtime_error("qwen35 model: the prompt runs past the KV capacity");
  // the many-row GDN replays at most 4 pending rows (an 8-row DFlash verify's accept may be more)
  if (!opt_.prefill_slices && !pf_committed_ && last_rows_ > 0 && pending_accept_ > k::kGdnRows)
    throw std::logic_error("qwen35 model: a prefill after a step that kept more than 4 rows (step one row first)");
  if (!pf_committed_ && last_rows_ > 0 && seqs_[static_cast<size_t>(slot_)].path != 0)  // the many-row GDN replays a chain
    throw std::logic_error("qwen35 model: a prefill after a tree step that kept a branch (step one row first)");
  kv_need(slot_, pos_ + n);  // 
  kv_cow(slot_, pos_);  // copy-on-write of shared pages
  pf_prof_ = PfProfile{};
  const int chunk = prefill_chunk();
  const auto t0 = std::chrono::steady_clock::now();
  for (int r0 = 0; r0 < n; r0 += chunk) {
    const int m = std::min(chunk, n - r0);
    pf_chunk(ids + r0, m, static_cast<bool>(on_rows), on_rows, r0 + m == n, r0 + m < n ? ids[r0 + m] : next);
  }
  pf_prof_.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// A norm whose xn feeds only W4A8 GEMMs may skip the bf16 xn: W4A8 on, every matrix on it (no keep list), every call
// over the chunk's rows or a two-card half of them past the short-call rows (those read bf16 on the decode GEMMs), and
// the halves' first row a multiple of 4 (a W4A8 call finds a producer's codes only from such a row).
bool Model::pf_codes_only(int m) const {
  return k::w4a16_pf_a8() && opt_.prefill_a8_keep.empty() && m / 2 > std::max(opt_.prefill_short, 0) + 4 && (cards() == 1 || (m / 2) % 4 == 0);
}

int Model::call_align() const { return k::gqa_attn_fa_tail() == 0 && k::w4a16_pf_onednn() == 0 ? 64 : prefill_chunk(); }

void Model::pf_norm(int c, int m, const bf16 *add, const bf16 *w, uint32_t pend, const std::function<void(int, int)> &proj, bool codes_only, PfNorm mode) {
  Card &cd = *cards_[static_cast<size_t>(c)];
  auto &q = fab_.compute(static_cast<size_t>(c));
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.rms_eps);
  // with W4A8 on, the norm also writes xn's codes and hands them to the W4A8 GEMMs that read xn
  const bool q8 = k::w4a16_pf_a8();
  codes_only = codes_only && q8;
  if (mode != PfNorm::both && (!pend || !add || cards() != 2 || (mode == PfNorm::owner && !codes_only)))
    throw std::logic_error("qwen35 model: pf_norm on the owner's rows without a pending two-card sum, or owner without codes only");
  const size_t H_ = static_cast<size_t>(H);
  auto norm = [&](int r0, int r1, const bf16 *ad) {
    if (q8)
      return k::rms_norm_q8(q, r1 - r0, H, cd.pf_res + r0 * H_, ad ? ad + r0 * H_ : nullptr, nullptr, w, eps, cd.pf_xn + r0 * H_,
                            k::RmsNormQ8{cd.pf_xq + r0 * H_, cd.pf_xs + r0, cd.pf_xq_ld, cd.pf_xs_rows + r0 * (H_ / 128), codes_only}, plain_norm());
    return k::rms_norm(q, r1 - r0, H, cd.pf_res + r0 * H_, ad ? ad + r0 * H_ : nullptr, nullptr, w, eps, cd.pf_xn + r0 * H_, plain_norm());
  };
  if (q8) k::w4a16_pf_a8_input(q, cd.pf_xn, H, m, H, cd.pf_xq, cd.pf_xs, cd.pf_xq_ld, cd.pf_xs_rows, codes_only);
  else k::w4a16_pf_a8_input_clear(q);
  if (!pend || !add) {
    norm(0, m, add);
    return;
  }
  // add's own rows are this card's sum (in order on q); the peer's arrive by its back copy (pair_split_wait), or with
  // the sum kept by its owner (owner / restore) the peer shares its normed rows' codes or its residual rows
  int lo = 0, hi = 0;
  k::pair_split_owner_rows(c, m, &lo, &hi);
  sycl::event own;
  if (hi > lo) own = norm(lo, hi, add);
  if (mode != PfNorm::both) {
    Card &c0 = *cards_[0], &c1 = *cards_[1];
    void *const xq[2] = {c0.pf_xq, c1.pf_xq}, *const xs[2] = {c0.pf_xs_rows, c1.pf_xs_rows}, *const res[2] = {c0.pf_res, c1.pf_res};
    const k::SplitShare codes[2] = {{xq, H_}, {xs, H_ / 128 * 4}}, rows[1] = {{res, H_ * 2}};
    const std::span<const k::SplitShare> sets = mode == PfNorm::owner ? std::span<const k::SplitShare>(codes) : std::span<const k::SplitShare>(rows);
    k::pair_split_share(q, cd.side ? &*cd.side : nullptr, c, m, sets, cd.pf_split, pend, hi > lo ? &own : nullptr);
  }
  if (proj && hi > lo) proj(lo, hi);
  k::pair_split_wait(q, cd.pf_split, pend);
  auto part = [&](int r0, int r1) {
    if (r1 <= r0) return;
    if (mode == PfNorm::owner)  // the peer's codes and row-major scales arrived; the group-major scales from them
      k::rms_q8_scales_cols(q, r1 - r0, H / 128, cd.pf_xs_rows + r0 * (H_ / 128), cd.pf_xs + r0, cd.pf_xq_ld);
    else
      norm(r0, r1, mode == PfNorm::restore ? nullptr : add);  // restore: the peer's residual rows arrived with its add
  };
  part(0, lo);
  part(hi, m);
  if (proj) {
    if (lo > 0) proj(0, lo);
    if (hi < m) proj(hi, m);
  }
}

void Model::pf_wait_pending() {
  if (!pend_seq_) return;
  for (size_t c = 0; c < cards_.size(); ++c) k::pair_split_wait(fab_.compute(c), cards_[c]->pf_split, pend_seq_);
  pend_seq_ = 0;
}

// One chunk of m rows at pos_, layer-major. Slice s = rows [4 s, 4 s + r_s) at position pos_ + 4 s; its GDN parity and
// accept are what the s-th of a run of 4-row steps would use (accept 4 = the previous slice, every row kept).
void Model::pf_chunk(const int32_t *ids, int m, bool head_all, const std::function<void(int64_t, int)> &on_rows, bool last, int32_t next_tok) {
  pf_start(ids, m, head_all, on_rows, last, next_tok);
  pf_step(cfg_.layers);
}

void Model::pf_start(const int32_t *ids, int m, bool head_all, const std::function<void(int64_t, int)> &on_rows, bool last, int32_t next_tok) {
  pf_.ids.assign(ids, ids + m);
  pf_.m = m;
  pf_.head_all = head_all;
  pf_.on_rows = on_rows;
  pf_.last = last;
  pf_.next = next_tok;
  pf_.l = 0;
  pf_.slot = slot_;
  pf_.active = true;
  pf_.segs.clear();
  seqs_[static_cast<size_t>(slot_)].pf_hx = -1;
  // an armed snapshot strictly inside this chunk (consumed by the call either way)
  pf_.snap = -1;
  if (cap_arm_slot_ == slot_) {
    const int64_t a = cap_arm_pos_;
    if (snap_inside() && a % call_align() == 0 && a > pos_ && a < pos_ + m) pf_.snap = a;
    cap_arm_slot_ = -1;
    cap_arm_pos_ = -1;
  }
  pf_.rope_rows = false;
  pf_.snap_rope = seqs_[static_cast<size_t>(slot_)].rope_delta;
  if (vis_ || tier_limit_.size() > 1) pf_rope_rows(ids, m);  // the rows' M-RoPE, or a tier's delta
  if (vis_) {  // the ids for the device (image ids -> pool rows, encoded first)
    vis_ids_.resize(static_cast<size_t>(m));
    vis_->prepare(ids, m, vis_ids_.data());
    pf_.ids.assign(vis_ids_.begin(), vis_ids_.end());
    if (next_tok >= cfg_.vocab) vis_->prepare(&pf_.next, 1, &pf_.next);
  }
}

void Model::pf_rope_rows(const int32_t *ids, int m) {
  Seq &sq = seqs_[static_cast<size_t>(slot_)];
  int64_t d = sq.rope_delta;
  bool any = d != 0;
  for (int r = 0; r < m && !any; ++r) any = ids[r] >= cfg_.vocab;
  if (!any) return;  // text from a zero delta: the rope by position (the bits of a text-only server)
  const int rot = attn_.shape.rot;
  pf_.rope_rows = true;
  pf_rope_host_.resize(static_cast<size_t>(m) * rot);
  model::MropeImage im;
  for (int r = 0; r < m; ++r) {
    const int64_t P = pos_ + r;
    if (P == pf_.snap) pf_.snap_rope = d;
    const bool img = ids[r] >= cfg_.vocab && vis_->image(ids[r], im);
    int64_t thw[3];
    model::Mrope::row(P, d, img ? &im : nullptr, img ? static_cast<int>(ids[r] - im.base) : 0, thw);
    mrope_.compose(rope_host_.data(), rot, thw, pf_rope_host_.data() + static_cast<size_t>(r) * rot);
  }
  sq.rope_delta = d;
}

void Model::set_vision(VisionStore *store) {
  if (trained_qsa3_ && store) throw std::invalid_argument("trained QSA layer3 is text-only");
  if (!store) return;
  if (!mrope_cfg_) throw std::invalid_argument("qwen35 model: image input needs the interleaved M-RoPE config (mrope_section)");
  if (opt_.prefill_slices || opt_.prefill_attn_slices) throw std::invalid_argument("qwen35 model: image input runs the many-row prefill only");
  if (store->vocab() != cfg_.vocab) throw std::invalid_argument("qwen35 model: the image store's vocabulary differs");
  if (store->cards() < static_cast<int>(cards_.size()))
    throw std::invalid_argument("qwen35 model: the image store holds rows on " + std::to_string(store->cards()) + " cards, the model gathers on " +
                                std::to_string(cards_.size()));
  vis_ = store;
  for (int i = 0; i < 3; ++i) mrope_.sec[i] = cfg_.mrope_section[static_cast<size_t>(i)];
  std::vector<const bf16 *> pools;
  for (size_t c = 0; c < cards_.size(); ++c) {
    pools.push_back(static_cast<const bf16 *>(store->pool(static_cast<int>(c))));
    Card &cd = *cards_[c];
    if (cd.pf_rope) continue;  // the YaRN tiers allocated it
    void *p = fab_.alloc(c, static_cast<size_t>(prefill_chunk()) * attn_.shape.rot * 2 + 4096);
    cd.mem.push_back(p);
    cd.pf_rope = static_cast<bf16 *>(p);
  }
  head_->set_ext(pools, static_cast<int32_t>(cfg_.vocab));
  attn_.shape.mrope = true;
  if (mtp_) mtp_->set_mrope(true);
}

size_t Model::cap_gdn_off(int c, int l) const {
  const Card &cd = *cards_.at(static_cast<size_t>(c));
  size_t b = 0;
  for (int x : gdn_.layers) {
    if (x == l) return b;
    b += cd.gdn[static_cast<size_t>(x)].slot_S * 4 + cd.gdn[static_cast<size_t>(x)].slot_win * 2;
  }
  throw std::logic_error("qwen35 model: not a GDN layer");
}

void Model::snap_arm(int slot, int64_t pos) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen35 model: no such slot");
  cap_arm_slot_ = slot;
  cap_arm_pos_ = pos;
}

int Model::snap_take() {
  if (!capt_.ready) return -2;
  int id = -1;
  for (size_t i = 0; i < snaps_.size() && id < 0; ++i)
    if (!snaps_[i].used) id = static_cast<int>(i);
  if (id < 0) {
    if (static_cast<int>(snaps_.size()) >= snap_cap_) return -1;
    snaps_.emplace_back();
    id = static_cast<int>(snaps_.size()) - 1;
  }
  SnapRec &r = snaps_[static_cast<size_t>(id)];
  if (r.host.empty())
    for (size_t c = 0; c < cards_.size(); ++c) r.host.push_back(fab_.alloc_host(std::max<size_t>(snap_bytes(static_cast<int>(c)), 4096)));
  r.pos = capt_.pos;
  // In-chunk capture: the call started covered (keys appended from its first row), so every complete
  // group below the page-aligned capture position is resident in the capturing slot.
  r.qsa_owner=(qsa_resident_snapshots() && trained_qsa3_ && capt_qsa_ && capt_.pos>0 &&
               capt_.pos%k::kQsaPage==0 && capt_rope_==0)?capt_.slot:-1;
  if(r.qsa_owner>=0) std::fprintf(stderr,"QSA_RESIDENT_TAKE id=%d slot=%d pos=%lld\n",id,r.qsa_owner,static_cast<long long>(r.pos));
  r.parity = capt_.parity;
  r.rope_delta = capt_rope_;
  r.used = true;
  r.saved.resize(cards_.size());
  r.read.resize(cards_.size());
  for (size_t c = 0; c < cards_.size(); ++c) {
    // the capture's writes ran on the compute queue, drained at the call's end; a restore of this id may still read
    r.saved[c] = snap_q_[c]->memcpy(r.host[c], cap_buf_[c], snap_bytes(static_cast<int>(c)), {r.read[c],r.saved[c]});
    cap_ev_[c] = r.saved[c];
  }
  capt_.ready = false;
  return id;
}

// Layers [pf_.l, pf_.l + layers) of the started chunk (a call in parts; the embed with the first, the head and
// the drafter KV with the last). Between parts every queue is drained and no split sum is pending, so decode steps of
// OTHER slots may run (they share no buffer with the chunk's). Returns true when the chunk is done.
bool Model::pf_step(int layers) {
  if (!pf_.active) throw std::logic_error("qwen35 model: no prefill chunk started");
  if (slot_ != pf_.slot) use_slot(pf_.slot);  // the engine may have switched slots (begin of another) between parts
  const int32_t *ids = pf_.ids.data();
  const int m = pf_.m;
  const bool head_all = pf_.head_all, last = pf_.last;
  const int32_t next_tok = pf_.next;
  const bool ends_prompt = last && next_tok < 0;  // no prompt token follows this call: its last rows produce the first reply token
  const auto on_rows = pf_.on_rows;
  const int l0 = pf_.l, l1 = std::min(cfg_.layers, l0 + std::max(layers, 1));
  const size_t C = cards_.size();
  const int H = cfg_.hidden, S = (m + 3) / 4;
  const size_t sl = static_cast<size_t>(4) * H;  // elements per slice
  std::vector<int32_t> pos(static_cast<size_t>(S)), acc(static_cast<size_t>(S));
  std::vector<int> par(static_cast<size_t>(S)), rows(static_cast<size_t>(S));
  int p = parity_;
  for (int s = 0; s < S; ++s) {
    rows[s] = std::min(4, m - 4 * s);
    pos[s] = static_cast<int32_t>(pos_ + 4 * s);
    const bool prev = s > 0 || last_rows_ > 0;
    acc[s] = s > 0 ? 4 : (pf_committed_ ? 0 : (last_rows_ > 0 ? pending_accept_ : 0));
    if (prev) p ^= 1;
    par[s] = p;
  }
  const bool multi = !pf_.segs.empty();
  if(trained_qsa3_) {
    if(multi) {
      for(const auto &sg:pf_.segs) if(sg.rope || qsa_covered_[sg.slot]!=sg.pos) qsa_covered_[sg.slot]=-1;
    } else if(pf_.rope_rows || qsa_covered_[slot_]!=pos_) qsa_covered_[slot_]=-1;
  }
  if (multi) {  // every segment's slices; the padding rows between segments are 4-row slices at position 0
    std::fill(rows.begin(), rows.end(), 4);
    std::fill(pos.begin(), pos.end(), 0);
    std::fill(acc.begin(), acc.end(), 0);
    for (const auto &sg : pf_.segs)
      for (int j = 0; 4 * j < sg.m; ++j) {
        const size_t s = static_cast<size_t>(sg.row0 / 4 + j);
        rows[s] = std::min(4, sg.m - 4 * j);
        pos[s] = static_cast<int32_t>(sg.pos + 4 * j);
        acc[s] = j ? 4 : sg.acc;
      }
  }
  const bool rows_mode = !opt_.prefill_slices;
  const bool pr = opt_.profile;
  auto tick = std::chrono::steady_clock::now();
  auto mark = [&](double &bucket) {  // profiled: sync both cards, charge the wall time since the last mark
    if (!pr) return;
    for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
    const auto now = std::chrono::steady_clock::now();
    bucket += std::chrono::duration<double, std::milli>(now - tick).count();
    tick = now;
  };
  const int gp = par[0];  // many-row GDN: the chunk acts as one step of slice 0's parity (reads [gp ^ 1], writes win[gp])
  const int64_t snap = rows_mode ? pf_.snap : -1;  // the in-call capture's position
  if (l0 == 0 && snap >= 0) {
    capt_.ready = false;
    for (size_t c = 0; c < C; ++c) fab_.compute(c).ext_oneapi_submit_barrier({cap_ev_[c]});  // its last host copy read the buffer
  }
  if (l0 == 0) {
  if (rows_mode && opt_.prefill_embed_chunk) {
    // one gather over the chunk (the previous chunk ended with both cards waited: the host ids are free)
    for (size_t c = 0; c < C; ++c) {
      std::memcpy(cards_[c]->pf_ids, ids, static_cast<size_t>(m) * 4);
      head_->gather(static_cast<int>(c), cards_[c]->pf_ids, m, cards_[c]->pf_res);
    }
  } else {
    for (int s = 0; s < S; ++s) {  // embed per slice into the chunk residual
      head_->set_ids(ids + 4 * s, rows[s]);  // waits for the cards
      head_->embed(rows[s]);
      for (size_t c = 0; c < C; ++c)
        fab_.compute(c).memcpy(cards_[c]->pf_res + s * sl, head_->card(static_cast<int>(c)).res, static_cast<size_t>(rows[s]) * H * 2);
    }
  }
  for (size_t c = 0; c < C; ++c) {
    auto &q = fab_.compute(c);
    q.memcpy(cards_[c]->pf_pos, pos.data(), static_cast<size_t>(S) * 4);
    q.memcpy(cards_[c]->pf_acc, acc.data(), static_cast<size_t>(S) * 4);
    if (pf_.rope_rows) q.memcpy(cards_[c]->pf_rope, pf_rope_host_.data(), pf_rope_host_.size() * 2);  // (host rows kept to the chunk's end)
  }
  mark(pf_prof_.embed);
  }
  std::vector<MlpWeights> mw(C);
  std::vector<const bf16 *> xs(C);
  k::GqaEvents ae;
  k::GdnEvents ge;
  const bool split = rows_mode && opt_.prefill_split && C == 2 && m >= kPfSplitMinRows;
  // the producers send the peer's rows on the side queue; the consumer (norm + its first GEMM) takes the peer's
  // rows late. Both from kSendMinRows.
  constexpr int kSendMinRows = 512;
  const bool send = split && opt_.prefill_send && m >= kSendMinRows && !multi && !has_lora();  // (a segment is under kSendMinRows: its own call sends nothing; a LoRA delta lands after a send)
  const bool defer = send && opt_.prefill_defer;
  double split_ms = 0;
  // the split two-card sum of a many-row producer's fp32 partials (both cards' partial on their queues); sent: the
  // producers already sent the peer's rows (pair_split_send). With defer the sum's peer rows stay pending (pend_seq_);
  // owner: its consumer norms each card's own rows (PfNorm owner / restore) and nothing is copied back.
  auto split_sum = [&](auto part_of, bf16 *Card::*out, bool sent, bool owner = false) {
    sycl::queue *qq[2] = {&fab_.compute(0), &fab_.compute(1)};
    sycl::queue *side[2] = {&*cards_[0]->side, &*cards_[1]->side};
    const float *part[2] = {part_of(0), part_of(1)};
    bf16 *o[2] = {cards_[0].get()->*out, cards_[1].get()->*out};
    k::PairSplit sp[2] = {cards_[0]->pf_split, cards_[1]->pf_split};
    pf_seq_ = k::pair_split_next_seq(pf_seq_);
    if (pr)
      for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();  // the producers done: time the sum alone
    const auto t_split = std::chrono::steady_clock::now();
    if (sent) {
      k::pair_split_finish(qq, side, m, H, part, o, sp, pf_seq_, nullptr, owner ? k::SplitRows::owner : defer ? k::SplitRows::deferred : k::SplitRows::all);
      if (defer) pend_seq_ = pf_seq_;
    } else {
      k::pair_split_rows(qq, m, H, part, o, sp, pf_seq_);
    }
    if (pr) {
      pf_wait_pending();
      for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
      split_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_split).count();
    }
  };
  // the send record a split producer gets: the sequence number the next split_sum will use
  std::vector<k::PairSplitSend> sends(C);
  auto send_of = [&](size_t c) -> const k::PairSplitSend * {
    if (!send) return nullptr;
    sends[c] = k::PairSplitSend{&*cards_[c]->side, &cards_[c]->pf_split, k::pair_split_next_seq(pf_seq_)};
    return &sends[c];
  };
  // Reuse the unmodified coverage guard above: restored, partial, media and multi-segment
  // inputs must not read stale selector history. Only a covered, native text chunk is eligible.
  // With the stock page-selected prefill on, it chooses the pages; the trained bank only indexes the chunk (decode reads it).
  const bool sparse_pf=trained_qsa_prefill_min_ && !opt_.prefill_select && rows_mode && !multi && !pf_.rope_rows &&
                       pos_>=trained_qsa_prefill_min_ && qsa_covered_[slot_]==pos_;
  static const bool pf_merged=[] {
    const char *s=std::getenv("XEN_TRAINED_QSA_PREFILL_MERGED");return s && std::string_view(s)=="1";
  }();
  const auto use_merged_plan=[&](const auto &sg) {
    return pf_merged && rows_mode && trained_qsa_prefill_min_ && !opt_.prefill_attn_slices &&
           !opt_.prefill_select && !pf_.rope_rows && !sg.rope && sg.pos>=trained_qsa_prefill_min_ && qsa_covered_[sg.slot]==sg.pos;
  };
  static const bool pf4_route=[] {
    const char *s=std::getenv("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG");return s && std::string_view(s)=="1";
  }();
  if(pf4_route && l0==0) {
    const auto coverage=multi?-1:qsa_covered_[slot_];
    const char *reason=multi?"merged":pf_.rope_rows?"rope":!rows_mode?"not-rows":
                       pos_<trained_qsa_prefill_min_?"below-threshold":coverage!=pos_?"no-coverage":"eligible";
    std::fprintf(stderr,"PREFILL_4SLOT slot=%d pos=%lld rows=%d covered=%lld route=%s reason=%s\n",multi?-1:slot_,
      static_cast<long long>(pos_),m,static_cast<long long>(coverage),sparse_pf?"trained":"dense",reason);
  }
  // trained QSA layers project the bf16 xn of every row (the selector index): their input norm stays on both cards' rows, not codes-only
  auto qsa_xn = [&](int l) { return trained_qsa3_ && trained_qsa_layer_[l] >= 0; };
  // The norms on their owner's rows (PfNorm): a norm whose xn feeds only W4A8 GEMMs (attention and dense MLP inputs) runs
  // on each card's own rows and the cards swap the codes (half the norm work, half the bytes of the bf16 sum's back
  // copy); the residual's peer rows then go stale, and the next norm that needs them (a GDN input, a DFlash feature
  // layer, the call's last norms) restores them from their owners. Not under profiling (its waits need the back copy).
  const bool own_ok = defer && !pr && C == 2;
  auto in_owner = [&](int l) {  // layer l's input norm on the owner's rows (l > 0: a pending MLP sum)
    return own_ok && l > l0 && cfg_.full_attention[l] && rows_mode && !opt_.prefill_attn_slices && pf_codes_only(m) &&
           !(dflash_ && feat_of_[static_cast<size_t>(l - 1)] >= 0) && !qsa_xn(l);
  };
  auto post_owner = [&](int l) {  // layer l's MLP input norm on the owner's rows: the next layer's input norm in this call restores
    return own_ok && rows_mode && (!cfg_.full_attention[l] || !opt_.prefill_attn_slices) && !(moe_ && moe_->has(l)) && mlp_ && pf_codes_only(m) && l + 1 < l1;
  };
  auto mode_of = [&](bool owner, bool stale) { return owner ? PfNorm::owner : stale ? PfNorm::restore : PfNorm::both; };
  bool stale = false;  // the residual's peer rows are stale on each card (the last norm ran on the owner's rows only)
  for (int l = l0; l < l1; ++l) {
    const bool full = cfg_.full_attention[l];
    std::array<k::GqaTrainedPrefill,2> pf_plan{};
    // with the last MLP sum pending, the mixer's first GEMM runs on the card's own rows before the wait
    const uint32_t pend_in = l > 0 ? pend_seq_ : 0;
    const PfNorm in_mode = pend_in ? mode_of(in_owner(l), stale) : PfNorm::both;
    if (!pend_in && stale) throw std::logic_error("qwen35 model: a stale residual without a pending sum to restore it");
    const bool pre_in = pend_in != 0 && rows_mode && (!full || !opt_.prefill_attn_slices);
    for (size_t c = 0; c < C; ++c) {
      Card &cd = *cards_[c];
      auto &q = fab_.compute(c);
      std::function<void(int, int)> proj;
      if (pre_in && full) {
        const k::GqaWeights &w = attn_.w[c][l];
        const k::LoraLayer *ll = lora_layer(static_cast<int>(c), l);
        k::LoraScratch *ls = lora_scratch(static_cast<int>(c));
        proj = [&](int lo, int hi) {
          k::w4a16_gemm_pf(q, hi - lo, cd.pf_xn + static_cast<size_t>(lo) * H, H, w.qkv.w, w.qkv.scales, cd.pf_gqa.y + static_cast<size_t>(lo) * w.qkv.n, w.qkv.n,
                           w.qkv.n, w.qkv.k);
          if (ll) {  // the q | k | v deltas into the fp32 qkv (before the norms / rope)
            if (ll->q.present()) k::lora_apply(q, hi - lo, cd.pf_xn + static_cast<size_t>(lo) * H, ll->q, cd.pf_gqa.y + static_cast<size_t>(lo) * w.qkv.n, w.qkv.n, *ls, 0);
            if (ll->k.present()) k::lora_apply(q, hi - lo, cd.pf_xn + static_cast<size_t>(lo) * H, ll->k, cd.pf_gqa.y + static_cast<size_t>(lo) * w.qkv.n, w.qkv.n, *ls, 0);
            if (ll->v.present()) k::lora_apply(q, hi - lo, cd.pf_xn + static_cast<size_t>(lo) * H, ll->v, cd.pf_gqa.y + static_cast<size_t>(lo) * w.qkv.n, w.qkv.n, *ls, 0);
          }
        };
      } else if (pre_in) {
        const k::GdnWeights &w = gdn_.w[c][l];
        proj = [&](int lo, int hi) {
          k::w4a16_gemm_pf(q, hi - lo, cd.pf_xn + static_cast<size_t>(lo) * H, H, w.qkvz4.w, w.qkvz4.scales, cd.pf_gdn.yq + static_cast<size_t>(lo) * w.qkvz.n,
                           w.qkvz.n, w.qkvz4.n, H);
        };
      }
      // an attention layer's xn feeds only its q|k|v GEMM (the GDN's a / b dots read the bf16 xn)
      pf_norm(static_cast<int>(c), m, l > 0 ? cd.pf_mlp : nullptr, cd.in_norm[l], pend_in, proj, full && rows_mode && !opt_.prefill_attn_slices && pf_codes_only(m) && !qsa_xn(l),
              in_mode);
      if (trained_qsa3_ && trained_qsa_layer_[l]>=0) {
        if(multi) {
          for(const auto &sg:pf_.segs) if(qsa_covered_[sg.slot]>=0 && !(full && use_merged_plan(sg)))
            trained_qsa3_->prefill(static_cast<int>(c),cd.pf_xn+size_t(sg.row0)*H,sg.m,sg.slot,static_cast<int>(sg.pos),trained_qsa_layer_[l]);
        } else if(qsa_covered_[slot_]>=0) {
          const int group=sparse_pf && full ? k::gqa_attn_fa_rows(attn_.shape_of(static_cast<int>(c),l),opt_.prefill_attn_rows):0;
          pf_plan[c]=trained_qsa3_->prefill(static_cast<int>(c),cd.pf_xn,m,slot_,static_cast<int>(pos_),trained_qsa_layer_[l],group);
        }
      }
    }
    stale = in_mode == PfNorm::owner;
    const PfNorm post_mode = mode_of(post_owner(l), stale);
    pend_seq_ = 0;
    mark(pf_prof_.norms);
    // pf_res holds layer l - 1's output of every row: the DFlash features of the chunk
    if (dflash_ && rows_mode && l > 0 && feat_of_[static_cast<size_t>(l - 1)] >= 0)
      for (size_t c = 0; c < C; ++c)
        dflash_->capture(static_cast<int>(c), fab_.compute(c), feat_of_[static_cast<size_t>(l - 1)], cards_[c]->pf_res, m, dflash_->prefill_features(static_cast<int>(c)));
    if (full && rows_mode && !opt_.prefill_attn_slices) {
      for (size_t c = 0; c < C; ++c) {
        Card &cd = *cards_[c];
        k::PairRowsWire pw = cd.pf_wire;
        pw.split = split;
        pw.send = send_of(c);
        k::GqaRowsScratch sc = cd.pf_gqa;
        sc.qkv_ready = pre_in;
        k::GqaShape ash = attn_.shape_of(static_cast<int>(c), l);
        if (pf_.rope_rows) ash.rope_rows = cd.pf_rope;  // the chunk's M-RoPE rows
        if (multi) {  // q|k|v and o over all rows, the K / V writes and the attention per segment
          auto &q = fab_.compute(c);
          const k::GqaWeights &w = attn_.w[c][l];
          k::w4a16_gemm_pf(q, m, cd.pf_xn, H, w.qkv.w, w.qkv.scales, cd.pf_gqa.y, w.qkv.n, w.qkv.n, w.qkv.k);
          if (auto *ll = lora_layer(static_cast<int>(c), l)) {  // the q | k | v deltas into the fp32 qkv (before the norms / rope)
            auto *ls = lora_scratch(static_cast<int>(c));
            if (ll->q.present()) k::lora_apply(q, m, cd.pf_xn, ll->q, cd.pf_gqa.y, w.qkv.n, *ls, 0);
            if (ll->k.present()) k::lora_apply(q, m, cd.pf_xn, ll->k, cd.pf_gqa.y, w.qkv.n, *ls, 0);
            if (ll->v.present()) k::lora_apply(q, m, cd.pf_xn, ll->v, cd.pf_gqa.y, w.qkv.n, *ls, 0);
          }
          for (const auto &sg : pf_.segs) {
            const size_t r0 = static_cast<size_t>(sg.row0);
            k::GqaRowsScratch ss = cd.pf_gqa;
            ss.qkv_ready = true, ss.o_later = true;
            ss.y += r0 * static_cast<size_t>(w.qkv.n);
            ss.attn += r0 * static_cast<size_t>(ash.o_k());
            k::GqaShape as = ash;
            as.rope_rows = sg.rope ? cd.pf_rope + r0 * static_cast<size_t>(ash.rot) : nullptr;  // (a zero delta: the rope by position)
            k::GqaTrainedPrefill segment_plan{};
            if(trained_qsa3_ && trained_qsa_layer_[l]>=0 && use_merged_plan(sg)) {
              segment_plan=trained_qsa3_->prefill(static_cast<int>(c),cd.pf_xn+r0*static_cast<size_t>(H),sg.m,
                  sg.slot,static_cast<int>(sg.pos),trained_qsa_layer_[l],k::gqa_attn_fa_rows(as,opt_.prefill_attn_rows));
            }
            // In-order queue: consume this segment's masks before the same
            // workspace is populated for a different slot. Keep batched QKV/O.
            k::gqa_rows(q, as, sg.m, cd.pf_xn + r0 * static_cast<size_t>(H), w, kv_view(cd.kv[l], sg.slot), cd.rope, cd.pf_pos + r0 / 4, static_cast<int>(sg.pos), ss,
                        C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix + r0 * static_cast<size_t>(H), opt_.prefill_attn, opt_.prefill_attn_rows, opt_.prefill_attn_p_lo,
                        opt_.prefill_attn_qk, k::GqaRowsSelect{.pages = opt_.prefill_select, .dense_last_block = sg.next < 0},
                        segment_plan.pages?&segment_plan:nullptr, lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c)));
            if(pf_merged && l==3 && c==0)
              std::fprintf(stderr,"QSA_MERGED_PREFILL slot=%d pos=%lld rows=%d route=%s\n",sg.slot,
                static_cast<long long>(sg.pos),sg.m,segment_plan.pages?"trained":"dense");
          }
          const auto e = k::gqa_rows_o(q, ash, m, w, cd.pf_gqa, C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix, lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c)));
          if (c == 0) ae = e;
          continue;
        }
        const auto e = k::gqa_rows(fab_.compute(c), ash, m, cd.pf_xn, attn_.w[c][l], kv_view(cd.kv[l], slot_), cd.rope, cd.pf_pos,
                                   static_cast<int>(pos_), sc, C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix, opt_.prefill_attn, opt_.prefill_attn_rows,
                                   opt_.prefill_attn_p_lo, opt_.prefill_attn_qk,
                                   k::GqaRowsSelect{.pages = opt_.prefill_select, .dense_last_block = ends_prompt},
                                   pf_plan[c].pages ? &pf_plan[c] : nullptr, lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c)));
        if (pf_plan[c].pages) {
          const uint64_t bit=uint64_t(1)<<trained_qsa_layer_[l];
          if (!(trained_prefill_reported_[c]&bit)) {
            trained_prefill_reported_[c]|=bit;
            std::fprintf(stderr,"trained QSA prefill submitted: rank=%zu layer=%d slot=%d pos=%lld rows=%d group=%d scorer=bf16-xmx8x16 fa=k8v4-i8-fast-masked-union\n",
                         c,l,slot_,static_cast<long long>(pos_),m,pf_plan[c].rows_per_group);
          }
        }
        if (c == 0) ae = e;
      }
      if (split) split_sum([&](int c) { return static_cast<const float *>(cards_[c]->pf_gqa.partial); }, &Card::pf_mix, send, post_mode != PfNorm::both);
      if (pr) {  // after both cards submitted: the reduce waits on the peer
        for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
        const sycl::event ev[5] = {ae.qkv, ae.prep, ae.attn, ae.o, ae.reduce};
        for (int i = 0; i < (split ? 4 : 5); ++i) pf_prof_.attn_parts[i] += ev_us(ev[i]) / 1e3;
      }
      pf_parity_ ^= 1;
    }
    for (int s = 0; s < S && !(full && rows_mode && !opt_.prefill_attn_slices); ++s) {
      for (size_t c = 0; c < C; ++c) {
        auto &q = fab_.compute(c);
        Card &cd = *cards_[c];
        const bf16 *x = cd.pf_xn + s * sl;
        bf16 *out = cd.pf_mix + s * sl;
        if (full) {
          const int pb = static_cast<int>(std::min<int64_t>(cap_, pos[s] + rows[s]));
          k::gqa_decode(q, attn_.shape_of(static_cast<int>(c), l), rows[s], x, attn_.w[c][l], kv_view(cd.kv[l], slot_), cd.rope, k::GqaStep{cd.pf_pos + s, pb}, cd.asc,
                        C > 1 ? &cd.wire : nullptr, mix_parity_, out, 0, 0, lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c)));
        } else if (!rows_mode) {
          k::GdnState st = gdn_view(cd.gdn[l], slot_);
          st.accept = cd.pf_acc + s;
          k::gdn_decode(q, gdn_.shape_of(static_cast<int>(c)), rows[s], x, gdn_.w[c][l], st, par[s], cd.gsc, C > 1 ? &cd.wire : nullptr, mix_parity_, out);
        }
      }
      if (full || !rows_mode) mix_parity_ ^= 1;
      if (!full && rows_mode) break;  // the GDN runs once over the chunk below
    }
    if (!full && rows_mode) {
      for (size_t c = 0; c < C; ++c) {
        Card &cd = *cards_[c];
        if (multi) {  // qkvz and out over all rows, the core per segment on its own state
          auto &q = fab_.compute(c);
          const k::GdnWeights &w = gdn_.w[c][l];
          const auto gs = gdn_.shape_of(static_cast<int>(c));
          k::PairRowsWire pw = cd.pf_wire;
          pw.split = split;
          k::w4a16_gemm_pf(q, m, cd.pf_xn, H, w.qkvz4.w, w.qkvz4.scales, cd.pf_gdn.yq, w.qkvz.n, w.qkvz4.n, H);
          for (const auto &sg : pf_.segs) {
            const size_t r0 = static_cast<size_t>(sg.row0);
            const k::GdnState g = gdn_view(cd.gdn[l], sg.slot);
            k::GdnRowsState rs;
            rs.S = g.S;
            rs.win_in = g.win[sg.gp ^ 1];
            rs.win_out = g.win[sg.gp];
            rs.log_y = g.y[sg.gp ^ 1];
            rs.log_ab = g.ab[sg.gp ^ 1];
            rs.accept = cd.pf_acc + r0 / 4;
            rs.pos = opt_.gdn_scan ? static_cast<int>(sg.pos) : -1;
            k::GdnRowsScratch ss = cd.pf_gdn;
            ss.qkvz_ready = true, ss.out_later = true;
            ss.yq += r0 * static_cast<size_t>(w.qkvz.n);
            ss.y += r0 * static_cast<size_t>(gs.val_w());
            k::gdn_rows(q, gs, sg.m, cd.pf_xn + r0 * static_cast<size_t>(H), w, rs, ss, C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix + r0 * static_cast<size_t>(H));
          }
          const auto e = k::gdn_rows_out(q, gs, m, w, cd.pf_gdn, C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix);
          if (c == 0) ge = e;
          continue;
        }
        const k::GdnState g = gdn_view(cd.gdn[l], slot_);
        k::GdnRowsState rs;
        rs.S = g.S;
        rs.win_in = g.win[gp ^ 1];
        rs.win_out = g.win[gp];  // the next step (parity gp ^ 1) reads win[gp]
        rs.log_y = g.y[gp ^ 1];
        rs.log_ab = g.ab[gp ^ 1];
        rs.accept = cd.pf_acc;  // slice 0's word: the previous step's rows still pending (0 after a prefill)
        rs.pos = opt_.gdn_scan ? static_cast<int>(pos_) : -1;
        if (snap >= 0) {  // the state at snap into the capture buffer
          char *b = static_cast<char *>(cap_buf_[c]) + cap_gdn_off(static_cast<int>(c), l);
          rs.snap_pos = static_cast<int>(snap);
          rs.snap_S = reinterpret_cast<float *>(b);
          rs.snap_win = reinterpret_cast<k::gdn_bf16 *>(b + cd.gdn[l].slot_S * 4);  // (a view's strides are 0)
        }
        k::PairRowsWire pw = cd.pf_wire;
        pw.split = split;
        pw.send = send_of(c);
        k::GdnRowsScratch sc = cd.pf_gdn;
        sc.qkvz_ready = pre_in;
        const auto e = k::gdn_rows(fab_.compute(c), gdn_.shape_of(static_cast<int>(c)), m, cd.pf_xn, gdn_.w[c][l], rs, sc, C > 1 ? &pw : nullptr, pf_parity_, cd.pf_mix);
        if (c == 0) ge = e;
      }
      if (split) split_sum([&](int c) { return static_cast<const float *>(cards_[c]->pf_gdn.partial); }, &Card::pf_mix, send, post_mode != PfNorm::both);
      if (pr) {
        for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
        const sycl::event ev[3] = {ge.qkvz, ge.out, ge.reduce};
        for (int i = 0; i < (split ? 2 : 3); ++i) pf_prof_.gdn_parts[i] += ev_us(ev[i]) / 1e3;
      }
      pf_parity_ ^= 1;
    }
    mark(full ? pf_prof_.attn : pf_prof_.gdn);
    // with the mixer sum pending, the MLP's gate|up runs on the card's own rows before the wait
    const uint32_t pend_mid = pend_seq_;
    const bool moe_l = moe_ && moe_->has(l);  // arch flint: the MoE block
    const bool pre_mid = pend_mid != 0 && !opt_.prefill_slices && opt_.prefill_fused_swiglu && !moe_l;
    for (size_t c = 0; c < C; ++c) {
      if (!moe_l) mw[c] = mlp_->weights(*dense_, static_cast<int>(c), l);
      std::function<void(int, int)> proj;
      if (pre_mid) proj = [&, c](int lo, int hi) { mlp_->gate_up_rows(static_cast<int>(c), lo, hi - lo, cards_[c]->pf_xn, mw[c], lora_layer(static_cast<int>(c), l), lora_scratch(static_cast<int>(c))); };
      // the dense MLP's xn feeds only its gate|up GEMM
      pf_norm(static_cast<int>(c), m, cards_[c]->pf_mix, cards_[c]->post_norm[l], pend_mid, proj, !moe_l && mlp_ && !opt_.prefill_slices && pf_codes_only(m),
              pend_mid ? post_mode : PfNorm::both);
    }
    if (!pend_mid && post_mode != PfNorm::both) throw std::logic_error("qwen35 model: an owner's-rows norm without its pending sum");
    stale = post_mode == PfNorm::owner;
    // the next layer's input norm (this call) takes the MLP sum on the owner's rows when it runs there or restores
    const bool next_own = l + 1 < l1 && (stale || in_owner(l + 1));
    pend_seq_ = 0;
    mark(pf_prof_.norms);
    if (!opt_.prefill_slices && moe_l) {  // arch flint: the many-row MoE block
      std::vector<bf16 *> outs(C);
      std::vector<k::PairRowsWire> pwv(C);
      std::vector<const k::PairRowsWire *> pw(C, nullptr);
      for (size_t c = 0; c < C; ++c) {
        xs[c] = cards_[c]->pf_xn, outs[c] = cards_[c]->pf_mlp;
        pwv[c] = cards_[c]->pf_wire;
        pwv[c].split = split;
        pwv[c].send = send_of(c);
        if (C > 1) pw[c] = &pwv[c];
      }
      moe_->run_rows(l, m, xs, pw, pf_parity_, outs);
      if (pr) {
        for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
        const auto &e = moe_->events(0);
        const sycl::event ev[5] = {e.router, e.route, e.gate_up, e.down, e.combine};
        for (int i = 0; i < 5; ++i) pf_prof_.moe_parts[i] += ev_us(ev[i]) / 1e3;
      }
      pf_parity_ ^= 1;
      if (split) split_sum([&](int c) { return moe_->rows_part(c); }, &Card::pf_mlp, send, next_own);
    } else if (!opt_.prefill_slices) {
      std::vector<bf16 *> outs(C);
      for (size_t c = 0; c < C; ++c) xs[c] = cards_[c]->pf_xn, outs[c] = cards_[c]->pf_mlp;
      const k::PairSplitSend *snd[2] = {nullptr, nullptr};
      if (send)
        for (size_t c = 0; c < C; ++c) snd[c] = send_of(c);
      mlp_->run_rows(m, xs, mw, outs, !split, opt_.prefill_fused_swiglu, send ? snd : nullptr, pre_mid, pf_parity_, lora_.get(), l);
      if (!split) pf_parity_ ^= 1;  // the reduce ran on the mixers' wire
      if (pr) {
        mlp_->wait();
        const auto &e = mlp_->events(0);
        const sycl::event ev[4] = {e.gu, e.silu, e.down, e.reduce};
        for (int i = 0; i < (split ? 3 : 4); ++i) pf_prof_.mlp_parts[i] += ev_us(ev[i]) / 1e3;
        if (opt_.prefill_fused_swiglu) pf_prof_.mlp_parts[1] = 0;  // gate|up carries it
      }
      if (split) split_sum([&](int c) { return mlp_->rows_part(c); }, &Card::pf_mlp, send, next_own);
    } else {
      for (int s = 0; s < S; ++s) {
        for (size_t c = 0; c < C; ++c) xs[c] = cards_[c]->pf_xn + s * sl;
        if (moe_l) moe_->run(l, rows[s], xs);
        else mlp_->run(rows[s], xs, mw, 0, 0, lora_.get(), l);
        for (size_t c = 0; c < C; ++c)
          fab_.compute(c).memcpy(cards_[c]->pf_mlp + s * sl, moe_l ? moe_->out(static_cast<int>(c)) : mlp_->card(static_cast<int>(c)).out,
                                 static_cast<size_t>(rows[s]) * H * 2);
      }
    }
    mark(pf_prof_.mlp);
  }
  pf_.l = l1;
  if (l1 < cfg_.layers) {  // a part: drain (the host words above, the split sums) before other slots' steps
    pf_wait_pending();
    for (size_t c = 0; c < C; ++c) {
      fab_.compute(c).wait_and_throw();
      if (cards_[c]->side) cards_[c]->side->wait_and_throw();
    }
    pf_prof_.split += split_ms;
    return false;
  }
  pf_.active = false;
  pf_wait_pending();  // the head (and the next chunk's first norm) read every row of the last MLP sum
  if (multi) {
    pf_tail_many(split_ms);
    return true;
  }
  // the head: every slice (scoring) or the prompt's last slice
  std::vector<const bf16 *> add(C);
  for (int s = head_all ? 0 : (last ? S - 1 : S); s < S; ++s) {
    for (size_t c = 0; c < C; ++c) {
      fab_.compute(c).memcpy(head_->card(static_cast<int>(c)).res, cards_[c]->pf_res + s * sl, static_cast<size_t>(rows[s]) * H * 2);
      add[c] = cards_[c]->pf_mlp + s * sl;
    }
    head_->final_norm(rows[s], add);
    head_->lm_pick(rows[s]);
    head_->wait();
    if (on_rows) on_rows(pos[s], rows[s]);
  }
  if (last) pf_last_row_ = rows[static_cast<size_t>(S - 1)] - 1;
  if (mtp_ && rows_mode) {  // the drafter KV of the chunk's rows whose next token is known
    const int mk = next_tok >= 0 ? m : m - 1;
    if (mk > 0) {
      std::vector<int32_t> nx(static_cast<size_t>(mk));
      for (int r = 0; r < mk; ++r) nx[static_cast<size_t>(r)] = r + 1 < m ? ids[r + 1] : next_tok;
      std::vector<const bf16 *> h(C);
      std::vector<const int32_t *> p4(C);
      std::vector<Mtp::Lend> lend(C);
      const float eps = static_cast<float>(cfg_.rms_eps);
      for (size_t c = 0; c < C; ++c) {
        Card &cd = *cards_[c];
        // the target's final-normed hidden of every row (the head above read its own copies of the residual)
        k::w4a16_pf_a8_input_clear(fab_.compute(c));  // pf_xn rewritten without codes
        k::rms_norm(fab_.compute(c), m, H, cd.pf_res, cd.pf_mlp, nullptr, head_->norm_weight(static_cast<int>(c)), eps, cd.pf_xn, plain_norm());
        h[c] = cd.pf_xn;
        p4[c] = cd.pf_pos;
        lend[c] = Mtp::Lend{cd.pf_mix, cd.pf_mlp, cd.pf_gqa.partial, cd.pf_gqa, C > 1 ? &cd.pf_wire : nullptr};
      }
      mtp_->settle(*head_, slot_, pos_);  // the slot's deferred rows below the chunk first
      std::vector<const bf16 *> rr(C);
      for (size_t c = 0; c < C; ++c) rr[c] = cards_[c]->pf_rope;
      mtp_->prefill_kv(*head_, mk, h, nx.data(), p4, static_cast<int>(pos_), lend, pf_parity_, slot_, pf_.rope_rows ? &rr : nullptr);
      pf_parity_ ^= 1;
    }
    mark(pf_prof_.head);
  }
  if (dflash_ && rows_mode) {  // every row of the chunk into the drafter's context (its features need no next token)
    std::vector<DFlash::Lend> lend(C);
    for (size_t c = 0; c < C; ++c) {
      Card &cd = *cards_[c];
      lend[c] = DFlash::Lend{cd.pf_mix, cd.pf_xn, cd.pf_gqa.partial, cd.pf_gqa.y, C > 1 ? &cd.pf_wire : nullptr};
    }
    if (snap >= 0) {  // the context below snap, its rings into the capture, then the rest
      const int m1 = static_cast<int>(snap - pos_);
      dflash_->prefill_ctx(m1, pos_, slot_, lend, pf_parity_);
      pf_parity_ ^= 1;
      for (size_t c = 0; c < C; ++c) {
        size_t oc = 0;
        for (int l : gdn_.layers) oc += cards_[c]->gdn[static_cast<size_t>(l)].slot_S * 4 + cards_[c]->gdn[static_cast<size_t>(l)].slot_win * 2;
        dflash_->ring_copy(static_cast<int>(c), fab_.compute(c), slot_, static_cast<char *>(cap_buf_[c]) + oc, true);
      }
      dflash_->prefill_ctx(m - m1, snap, slot_, lend, pf_parity_, m1);
    } else {
      dflash_->prefill_ctx(m, pos_, slot_, lend, pf_parity_);
    }
    pf_parity_ ^= 1;
    mark(pf_prof_.head);
  }
  for (size_t c = 0; c < C; ++c) {
    fab_.compute(c).wait_and_throw();  // pos / acc host arrays live until here
    if (cards_[c]->side) cards_[c]->side->wait_and_throw();
  }
  mark(pf_prof_.head);
  pf_prof_.split += split_ms;
  if (snap >= 0 && cfg_.flint) {  // arch flint: the ring windows at snap (the rings still hold them: ring_rows_ >= the chunk)
    for (size_t c = 0; c < C; ++c) {
      // after the drafter's rings (snap_bytes' layout; flint has no GDN layer)
      char *rb = static_cast<char *>(cap_buf_[c]) + (snap_bytes(static_cast<int>(c)) - ring_snap_bytes(static_cast<int>(c)));
      ring_snap_copy(static_cast<int>(c), fab_.compute(c), slot_, snap, rb, true, {});
      fab_.compute(c).wait_and_throw();
    }
  }
  if (snap >= 0) capt_ = Capture{true, slot_, gp, snap}, capt_rope_ = pf_.snap_rope,  // snap_take() makes it a snapshot (the parity of this call)
                 capt_qsa_ = trained_qsa3_ && qsa_covered_[slot_]==pos_;
  if(trained_qsa3_ && qsa_covered_[slot_]>=0) qsa_covered_[slot_]+=m;
  pos_ += m;
  if (rows_mode) {  // the GDN committed every row: the next step replays none, parity gp ^ 1
    parity_ = gp;
    last_rows_ = 4;
    pending_accept_ = 0;
    pf_committed_ = true;
  } else {
    parity_ = par[static_cast<size_t>(S - 1)];
    last_rows_ = rows[static_cast<size_t>(S - 1)];
    pending_accept_ = last_rows_;
  }
  return true;
}

void Model::prefill_begin(const int32_t *ids, int n, int32_t next) {
  if (n < 1 || n > prefill_chunk()) throw std::invalid_argument("qwen35 model: prefill_begin takes 1..prefill_chunk tokens");
  if (pf_.active) throw std::logic_error("qwen35 model: a prefill call is pending");
  if (last_rows_ > 0 && pending_accept_ == 0 && !pf_committed_) throw std::logic_error("qwen35 model: accept() the previous step first");
  if (pos_ + n > cap_) throw std::runtime_error("qwen35 model: the prompt runs past the KV capacity");
  // the many-row GDN replays at most 4 pending rows (an 8-row DFlash verify's accept may be more)
  if (!opt_.prefill_slices && !pf_committed_ && last_rows_ > 0 && pending_accept_ > k::kGdnRows)
    throw std::logic_error("qwen35 model: a prefill after a step that kept more than 4 rows (step one row first)");
  if (!pf_committed_ && last_rows_ > 0 && seqs_[static_cast<size_t>(slot_)].path != 0)  // the many-row GDN replays a chain
    throw std::logic_error("qwen35 model: a prefill after a tree step that kept a branch (step one row first)");
  kv_need(slot_, pos_ + n);  // 
  kv_cow(slot_, pos_);  // copy-on-write of shared pages
  pf_prof_ = PfProfile{};
  pf_start(ids, n, false, {}, true, next);
}

// What a call of several sequences may carry (model.hpp).
int Model::prefill_merge_rows() const {
  if (opt_.prefill_slices || opt_.prefill_attn_slices || !opt_.prefill_embed_chunk || moe_ || cfg_.flint || gdn_.layers.empty()) return 0;
  int cap = prefill_chunk();
  if (k::w4a16_pf_onednn() > 0) cap = std::min(cap, k::w4a16_pf_onednn_rows() - 1);  // every row on the ESIMD bodies, as each segment's own call
  return cap;
}

int Model::prefill_merge_max() const { return prefill_merge_rows() > 0 ? std::min(kPfMergeMax, head_->rows()) : 1; }

bool Model::prefill_mergeable(int slot, const int32_t *ids, int n) const {
  if (prefill_merge_rows() == 0 || slot < 0 || slot >= slots() || !ids) return false;
  // its own call's route: the many-row bodies (not the short call's decode GEMMs), the split sums, no send
  if (n <= std::max(opt_.prefill_short, k::w4a16_pf_short_rows()) || n < kPfSplitMinRows || n > kPfMergeSegMax) return false;
  const Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (sq.broken || cap_arm_slot_ == slot || position(slot) + n > cap_) return false;  // (a rope delta, a YaRN tier's or after an image: the segment's own rope rows)
  for (int r = 0; r < n; ++r)
    if (ids[r] < 0 || ids[r] >= cfg_.vocab) return false;  // text rows only (an image's rows carry their own rope)
  return true;
}

void Model::prefill_begin(const PrefillSeg *in, int count) {
  if (!in || count < 1) throw std::invalid_argument("qwen35 model: prefill_begin takes at least one sequence");
  if (count == 1) {
    use_slot(in[0].slot);
    prefill_begin(in[0].ids, in[0].n, in[0].next);
    return;
  }
  if (pf_.active) throw std::logic_error("qwen35 model: a prefill call is pending");
  if (count > prefill_merge_max()) throw std::invalid_argument("qwen35 model: a prefill call carries at most " + std::to_string(prefill_merge_max()) + " sequences");
  std::vector<PfRun::Seg> segs;
  int total = 0;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < i; ++j)
      if (in[j].slot == in[i].slot) throw std::invalid_argument("qwen35 model: a slot once per prefill call");
    if (!prefill_mergeable(in[i].slot, in[i].ids, in[i].n))
      throw std::invalid_argument("qwen35 model: slot " + std::to_string(in[i].slot) + "'s " + std::to_string(in[i].n) + " rows cannot share a prefill call (prefill_mergeable)");
    total = (total + kPfMergeAlign - 1) / kPfMergeAlign * kPfMergeAlign;
    PfRun::Seg sg;
    sg.slot = in[i].slot, sg.row0 = total, sg.m = in[i].n, sg.next = in[i].next;
    segs.push_back(sg);
    total += in[i].n;
  }
  if (total > prefill_merge_rows())
    throw std::invalid_argument("qwen35 model: the sequences' " + std::to_string(total) + " rows pass prefill_merge_rows() = " + std::to_string(prefill_merge_rows()));
  for (auto &sg : segs) {  // each slot as its own prefill_begin would take it
    use_slot(sg.slot);
    if (last_rows_ > 0 && pending_accept_ == 0 && !pf_committed_) throw std::logic_error("qwen35 model: accept() the previous step first");
    if (!pf_committed_ && last_rows_ > 0 && pending_accept_ > k::kGdnRows)
      throw std::logic_error("qwen35 model: a prefill after a step that kept more than 4 rows (step one row first)");
    if (!pf_committed_ && last_rows_ > 0 && seqs_[static_cast<size_t>(slot_)].path != 0)
      throw std::logic_error("qwen35 model: a prefill after a tree step that kept a branch (step one row first)");
    kv_need(slot_, pos_ + sg.m);
    kv_cow(slot_, pos_);
    sg.pos = pos_;
    sg.acc = pf_committed_ ? 0 : (last_rows_ > 0 ? pending_accept_ : 0);
    sg.gp = last_rows_ > 0 ? parity_ ^ 1 : parity_;  // pf_step's slice 0
    seqs_[static_cast<size_t>(slot_)].pf_hx = -1;
    sg.rope = seqs_[static_cast<size_t>(slot_)].rope_delta != 0;
  }
  // a segment of a slot with a rope delta (a YaRN tier's base, or text after an image) carries its own rope rows, at its
  // rows of the call: the rows pf_rope_rows gives its own call
  bool rope_any = false;
  for (const auto &sg : segs) rope_any |= sg.rope;
  if (rope_any) {
    const int rot = attn_.shape.rot;
    pf_rope_host_.resize(static_cast<size_t>(total) * rot);
    for (const auto &sg : segs) {
      if (!sg.rope) continue;
      int64_t d = seqs_[static_cast<size_t>(sg.slot)].rope_delta;  // (text rows leave it as it is)
      for (int r = 0; r < sg.m; ++r) {
        int64_t thw[3];
        model::Mrope::row(sg.pos + r, d, nullptr, 0, thw);
        mrope_.compose(rope_host_.data(), rot, thw, pf_rope_host_.data() + static_cast<size_t>(sg.row0 + r) * rot);
      }
    }
  }
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = *cards_[c];
    if (cd.pf_hx) continue;
    void *p = fab_.alloc(c, static_cast<size_t>(kPfMergeMax) * cfg_.hidden * 2 + 4096);
    cd.mem.push_back(p);
    cd.pf_hx = static_cast<bf16 *>(p);
  }
  pf_prof_ = PfProfile{};
  pf_.ids.assign(static_cast<size_t>(total), 0);  // (the padding rows: token 0; no row of a sequence reads them)
  for (int i = 0; i < count; ++i) std::copy(in[i].ids, in[i].ids + in[i].n, pf_.ids.begin() + segs[static_cast<size_t>(i)].row0);
  pf_.m = total;
  pf_.head_all = false;
  pf_.on_rows = {};
  pf_.last = true;
  pf_.next = -1;
  pf_.l = 0;
  pf_.slot = segs[0].slot;
  pf_.active = true;
  pf_.snap = -1;
  pf_.rope_rows = rope_any;
  pf_.snap_rope = 0;
  pf_.segs = std::move(segs);
  use_slot(pf_.slot);
}

// The end of a call of several sequences: one head over the segments' last rows (kept per slot for the first
// drafts), each segment's drafter rows, each slot's position and GDN parity.
void Model::pf_tail_many(double split_ms) {
  const size_t C = cards_.size();
  const int H = cfg_.hidden, m = pf_.m, K = static_cast<int>(pf_.segs.size());
  const size_t row = static_cast<size_t>(H) * 2;
  std::vector<const bf16 *> add(C);
  for (size_t c = 0; c < C; ++c) {
    Card &cd = *cards_[c];
    auto &q = fab_.compute(c);
    for (int i = 0; i < K; ++i) {
      const size_t r = static_cast<size_t>(pf_.segs[static_cast<size_t>(i)].row0 + pf_.segs[static_cast<size_t>(i)].m - 1);
      q.memcpy(head_->card(static_cast<int>(c)).res + static_cast<size_t>(i) * H, cd.pf_res + r * H, row);
      q.memcpy(cd.pf_mix + static_cast<size_t>(i) * H, cd.pf_mlp + r * H, row);  // (pf_mix is free after the last layer)
    }
    add[c] = cd.pf_mix;
  }
  head_->final_norm(K, add);
  head_->lm_pick(K);
  head_->wait();
  for (size_t c = 0; c < C; ++c) fab_.compute(c).memcpy(cards_[c]->pf_hx, head_->card(static_cast<int>(c)).xn, static_cast<size_t>(K) * row);
  if (mtp_) {  // the drafter KV of every segment's rows whose next token is known (pf_step's, per segment)
    const float eps = static_cast<float>(cfg_.rms_eps);
    std::vector<const bf16 *> h(C);
    std::vector<const int32_t *> p4(C);
    std::vector<Mtp::Lend> lend(C);
    for (size_t c = 0; c < C; ++c) {
      Card &cd = *cards_[c];
      k::w4a16_pf_a8_input_clear(fab_.compute(c));  // pf_xn rewritten without codes
      k::rms_norm(fab_.compute(c), m, H, cd.pf_res, cd.pf_mlp, nullptr, head_->norm_weight(static_cast<int>(c)), eps, cd.pf_xn, plain_norm());
      lend[c] = Mtp::Lend{cd.pf_mix, cd.pf_mlp, cd.pf_gqa.partial, cd.pf_gqa, C > 1 ? &cd.pf_wire : nullptr};
    }
    for (const auto &sg : pf_.segs) {
      const int mk = sg.next >= 0 ? sg.m : sg.m - 1;
      if (mk < 1) continue;
      std::vector<int32_t> nx(static_cast<size_t>(mk));
      for (int r = 0; r < mk; ++r) nx[static_cast<size_t>(r)] = r + 1 < sg.m ? pf_.ids[static_cast<size_t>(sg.row0 + r + 1)] : sg.next;
      for (size_t c = 0; c < C; ++c) {
        h[c] = cards_[c]->pf_xn + static_cast<size_t>(sg.row0) * H;
        p4[c] = cards_[c]->pf_pos + sg.row0 / 4;
      }
      mtp_->settle(*head_, sg.slot, sg.pos);  // the slot's deferred rows below its segment first
      std::vector<const bf16 *> rr(C);
      for (size_t c = 0; c < C; ++c) rr[c] = cards_[c]->pf_rope + static_cast<size_t>(sg.row0) * attn_.shape.rot;
      mtp_->prefill_kv(*head_, mk, h, nx.data(), p4, static_cast<int>(sg.pos), lend, pf_parity_, sg.slot, sg.rope ? &rr : nullptr);
      pf_parity_ ^= 1;
    }
  }
  if (dflash_) {  // every segment's rows into its slot's drafter context
    std::vector<DFlash::Lend> lend(C);
    for (size_t c = 0; c < C; ++c) {
      Card &cd = *cards_[c];
      lend[c] = DFlash::Lend{cd.pf_mix, cd.pf_xn, cd.pf_gqa.partial, cd.pf_gqa.y, C > 1 ? &cd.pf_wire : nullptr};
    }
    for (const auto &sg : pf_.segs) {
      dflash_->prefill_ctx(sg.m, sg.pos, sg.slot, lend, pf_parity_, sg.row0);
      pf_parity_ ^= 1;
    }
  }
  for (size_t c = 0; c < C; ++c) {
    fab_.compute(c).wait_and_throw();  // pos / acc host arrays live until here
    if (cards_[c]->side) cards_[c]->side->wait_and_throw();
  }
  pf_prof_.split += split_ms;
  for (int i = 0; i < K; ++i) {  // the GDN committed every row: the next step replays none
    const auto &sg = pf_.segs[static_cast<size_t>(i)];
    use_slot(sg.slot);
    if(trained_qsa3_ && qsa_covered_[slot_]>=0) qsa_covered_[slot_]+=sg.m;
    pos_ += sg.m;
    parity_ = sg.gp;
    last_rows_ = 4;
    pending_accept_ = 0;
    pf_committed_ = true;
    pf_last_row_ = i;
    seqs_[static_cast<size_t>(slot_)].pf_hx = i;
  }
}

bool Model::prefill_advance(int layers) { return pf_step(layers); }

void Model::prefill_abort() {
  if (!pf_.active) return;
  if(trained_qsa3_) {
    qsa_covered_[pf_.slot]=-1;qsa_invalidate_snapshots(pf_.slot);
    for(const auto &sg:pf_.segs){qsa_covered_[sg.slot]=-1;qsa_invalidate_snapshots(sg.slot); }
  }
  pf_wait_pending();
  for (size_t c = 0; c < cards_.size(); ++c) {
    fab_.compute(c).wait();
    if (cards_[c]->side) cards_[c]->side->wait();
  }
  pf_.active = false;  // the slot's state is undefined: reset() before reuse
  save_seq();
  seqs_[static_cast<size_t>(pf_.slot)].broken = true;  // never resumed live (a snapshot restore or reset clears it)
  for (const auto &sg : pf_.segs) seqs_[static_cast<size_t>(sg.slot)].broken = true;  // every sequence of the call
  pf_.segs.clear();
}

// Coupled drafts for a sampled request, both drafters (core::StepModel::draft_sampling).
void Model::set_draft_noise(int slot, uint64_t key, uint64_t pos, float inv_t) {
  if (mtp_) mtp_->set_noise(slot, key, pos, inv_t);
  if (dflash_) dflash_->set_noise(slot, key, pos, inv_t);
}

int Model::draft_depth_max() const { return dflash_ ? dflash_->max_depth() : (mtp_ ? kMtpPasses - 1 : 0); }
int Model::draft_depth_max(int seqs) const { return mtp_drafts(seqs) ? kMtpPasses - 1 : draft_depth_max(); }

namespace {
// Rows of a step of ns segments by its seg_rows word, and segment i's first row / rows
int code_rows(int code, int ns) { return code < 0 ? k::seg_het_rows(code) : code * ns; }
void code_span(int code, int ns, int i, int &r0, int &n) { k::seg_span(code, code_rows(code, ns), i, r0, n); }
int depth_of(const std::vector<int> &depth, size_t i) { return depth.size() == 1 ? depth[0] : depth[i]; }
}  // namespace

// DFlash: segment i's context rows are the last verify's rows of it (features of that step; 0 after a prefill: the
// prompt is in the context already), its block starts after its kept rows with anchor[i] (the token after them). A
// contiguous run of segments with the same rows and drafting or not is one equal shape (its context rows are adjacent
// in the verify's order: ctx_row0); each run gets its own passes, its block from its deepest segment.
std::vector<std::vector<int32_t>> Model::dflash_drafts(const std::vector<DSeg> &segs, int code, const int32_t *anchor, const std::vector<int> &depth) {
  const int ns = static_cast<int>(segs.size());
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(ns));
  draft_ms_ = 0;
  // trees only while every sequence of the step stays under the page budget through its widest verify (a tree step
  // attends densely); else the passes draft chains
  int64_t widest_bound = 0;
  for (const auto &g : segs) widest_bound = std::max<int64_t>(widest_bound, g.p0 + g.keep + std::min(k::kTreeRows, rows_max_));
  const int tree_seqs = selects_pages(std::min<int64_t>(cap_, widest_bound)) ? DFlash::kChainsOnly : ns;
  for (const auto &g : segs) note_tree(g.slot, 0);  // a chain unless a tree pass below
  bool any = false;
  for (size_t i = 0; i < segs.size(); ++i) any |= depth_of(depth, i) > 0;
  if (code < 0 && any) {
    // unequal segments with drafts (qwen38's form): the drafting segments pass together, the others for their
    // context only; each pass's feature rows re-laid as [segment][the group's largest rows] (rows past a segment's own
    // repeat its last: their context positions lie past what it verified, nothing reads them and the next verify
    // rewrites them), so one pass holds segments of unequal rows
    const int total = code_rows(code, ns);
    std::vector<int> rows(static_cast<size_t>(ns)), row0(static_cast<size_t>(ns)), drafting, plain;
    int dmax = 0;
    for (int i = 0; i < ns; ++i) {
      code_span(code, ns, i, row0[static_cast<size_t>(i)], rows[static_cast<size_t>(i)]);
      const int di = depth_of(depth, static_cast<size_t>(i));
      if (di > 0) drafting.push_back(i), dmax = std::max(dmax, di);
      else plain.push_back(i);
    }
    dflash_->save_step_features(total);
    std::function<void(const std::vector<int> &, int)> run = [&](const std::vector<int> &grp, int d) {
      if (grp.empty()) return;
      int R = 0, block = d > 0 ? std::min(d, dflash_->max_depth()) + 1 : 0;
      for (int i : grp) {
        const DSeg &g = segs[static_cast<size_t>(i)];
        R = std::max(R, rows[static_cast<size_t>(i)]);
        if (block > 0) block = static_cast<int>(std::min<int64_t>(block, cap_ - (g.p0 + g.keep)));
      }
      if (grp.size() > 1) {  // a segment whose padded rows would pass the capacity passes alone
        std::vector<int> fit, tight;
        for (int i : grp) (cap_ - segs[static_cast<size_t>(i)].p0 >= R ? fit : tight).push_back(i);
        if (!tight.empty()) {
          run(fit, d);
          for (int i : tight) run({i}, d);
          return;
        }
      }
      if (block < 2) block = 0;
      const int g = std::max(1, std::min(block > 0 ? pass_rows() / block : ns, pass_rows() / R));
      for (size_t i0 = 0; i0 < grp.size(); i0 += static_cast<size_t>(g)) {
        const size_t i1 = std::min(grp.size(), i0 + static_cast<size_t>(g));
        int map[DFlash::kRows] = {};
        std::vector<DFlash::Seg> part;
        for (size_t a = i0; a < i1; ++a) {
          const int sg = grp[a];
          const DSeg &gs = segs[static_cast<size_t>(sg)];
          part.push_back(DFlash::Seg{gs.slot, gs.p0, gs.p0 + gs.keep, anchor[sg]});
          for (int r = 0; r < R; ++r)
            map[(a - i0) * static_cast<size_t>(R) + static_cast<size_t>(r)] = row0[static_cast<size_t>(sg)] + std::min(r, rows[static_cast<size_t>(sg)] - 1);
        }
        dflash_->stage_step_features(map, static_cast<int>(i1 - i0) * R);
        auto res = dflash_->pass(part, R, block, 0, tree_seqs);
        draft_ms_ += dflash_->draft_ms();
        for (size_t a = i0; a < i1; ++a) {
          auto &v = out[static_cast<size_t>(grp[a])] = std::move(res[a - i0]);
          const int di = depth_of(depth, static_cast<size_t>(grp[a]));
          if (static_cast<int>(v.size()) > di) v.resize(static_cast<size_t>(std::max(0, di)));
          note_tree(part[a - i0].slot, v.size(), static_cast<int>(a - i0));
        }
      }
    };
    run(drafting, dmax);
    run(plain, 0);
    return out;
  }
  for (int a = 0; a < ns;) {
    int r0 = 0, nr = 0;
    if (code > 0 || code < 0) code_span(code, ns, a, r0, nr);
    const int ctx_rows = code == 0 ? 0 : nr;
    const bool dr = depth_of(depth, static_cast<size_t>(a)) > 0;
    int b = a + 1, dmax = depth_of(depth, static_cast<size_t>(a));
    while (b < ns) {
      int r1 = 0, n1 = 0;
      if (code != 0) code_span(code, ns, b, r1, n1);
      if ((code == 0 ? 0 : n1) != ctx_rows || (depth_of(depth, static_cast<size_t>(b)) > 0) != dr) break;
      dmax = std::max(dmax, depth_of(depth, static_cast<size_t>(b)));
      ++b;
    }
    int block = dmax > 0 ? std::min(dmax, dflash_->max_depth()) + 1 : 0;
    std::vector<DFlash::Seg> ds;
    for (int i = a; i < b; ++i) {
      const DSeg &g = segs[static_cast<size_t>(i)];
      const int64_t start = g.p0 + g.keep;
      if (block > 0) block = static_cast<int>(std::min<int64_t>(block, cap_ - start));  // the block's positions < the capacity
      ds.push_back(DFlash::Seg{g.slot, g.p0, start, anchor[i]});
    }
    if (block < 2) block = 0;
    if (ctx_rows > 0 || block > 0) {
      // a pass holds 16 block rows: the next step may draft deeper than this step's rows allow (fewer sequences next),
      // so the segments go in groups of 16 / block, each with its own context rows
      const int n = b - a, g = block > 0 ? std::max(1, pass_rows() / block) : n;
      for (int i0 = 0; i0 < n; i0 += g) {
        const int i1 = std::min(n, i0 + g);
        auto part = dflash_->pass(std::vector<DFlash::Seg>(ds.begin() + i0, ds.begin() + i1), ctx_rows, block, r0 + i0 * ctx_rows, tree_seqs);
        draft_ms_ += dflash_->draft_ms();
        for (int i = i0; i < i1; ++i) {
          auto &v = part[static_cast<size_t>(i - i0)];
          const int di = depth_of(depth, static_cast<size_t>(a + i));
          if (static_cast<int>(v.size()) > di) v.resize(static_cast<size_t>(std::max(0, di)));
          note_tree(ds[static_cast<size_t>(i)].slot, v.size(), i - i0);  // per segment (a prefix of a tree is a tree)
          out[static_cast<size_t>(a + i)] = std::move(v);
        }
      }
    }
    a = b;
  }
  return out;
}

std::vector<std::vector<int32_t>> Model::run_drafts(const std::vector<const bf16 *> &h, int code, const std::vector<DSeg> &segs, const int32_t *next,
                                                    const std::vector<int> &depth) {
  const int ns = static_cast<int>(segs.size());
  int dmax = 0;
  for (size_t i = 0; i < segs.size(); ++i) dmax = std::max(dmax, depth_of(depth, i));
  std::vector<int32_t> anc;  // the token after a segment's kept rows
  for (int i = 0; i < ns; ++i) {
    int r0 = 0, nr = 0;
    code_span(code, ns, i, r0, nr);
    anc.push_back(next[r0 + segs[static_cast<size_t>(i)].keep - 1]);
  }
  if (dflash_ && !mtp_drafts(ns)) {  // the verify's rows are the context; the anchor is the token after a segment's kept rows
    double mtp_ms = 0;
    if (mtp_) {  // MTP's KV of the verify's rows (pass 1 KV only, before the DFlash pass reuses the head's rows)
      mtp_run(h, code, segs, next, 0);
      mtp_ms = draft_ms_;
    }
    // an equal verify: one pass for all at the deepest (finishing requests ask for none); unequal segments: each run of
    // segments its own
    auto out = dflash_drafts(segs, code, anc.data(), code > 0 ? std::vector<int>{dmax} : depth);
    for (size_t i = 0; i < out.size(); ++i)
      if (static_cast<int>(out[i].size()) > depth_of(depth, i)) out[i].resize(static_cast<size_t>(std::max(0, depth_of(depth, i))));
    draft_ms_ += mtp_ms;
    return out;
  }
  auto out = mtp_run(h, code, segs, next, dmax);
  for (size_t i = 0; i < out.size(); ++i) {  // each segment its own drafts (the passes ran to the deepest)
    const int di = depth_of(depth, i);
    if (static_cast<int>(out[i].size()) > di) out[i].resize(static_cast<size_t>(std::max(0, di)));
  }
  if (dflash_) {  // MTP drafts; the DFlash context takes the verify's rows (a depth-0 pass: deferred)
    const double mtp_ms = draft_ms_;
    dflash_drafts(segs, code, anc.data(), {0});
    draft_ms_ += mtp_ms;
  }
  return out;
}

std::vector<std::vector<int32_t>> Model::mtp_run(const std::vector<const bf16 *> &h, int code, const std::vector<DSeg> &segs, const int32_t *next,
                                                 int depth) {
  if (!mtp_) throw std::logic_error("qwen35 model: no drafter loaded (ModelOptions::mtp)");
  const int ns = static_cast<int>(segs.size());
  if (ns < 1 || code == 0 || depth < 0) throw std::invalid_argument("qwen35 model: draft segments / rows / depth");
  const int rows = code_rows(code, ns);
  if (rows > Mtp::kRows) throw std::invalid_argument("qwen35 model: draft rows");
  const size_t C = cards_.size();
  // segment i's rows of pass 1 (unequal segments)
  int r0s[k::kSegs16], nrs[k::kSegs16];
  for (int i = 0; i < ns; ++i) code_span(code, ns, i, r0s[i], nrs[i]);
  // pass 1 at p0 .. p0 + rows_i - 1 per segment; pass k (k >= 1) at p0 + keep - 1 + k: stop at the capacity
  int64_t b1 = 0;
  for (int i = 0; i < ns; ++i) {
    const DSeg &g = segs[static_cast<size_t>(i)];
    if (g.keep < 1 || g.keep > nrs[i]) throw std::invalid_argument("qwen35 model: draft keep");
    if (g.p0 + nrs[i] > cap_) return std::vector<std::vector<int32_t>>(static_cast<size_t>(ns));
    depth = static_cast<int>(std::min<int64_t>(depth, cap_ - (g.p0 + g.keep - 1)));
    b1 = std::max(b1, g.p0 + nrs[i]);
  }
  depth = std::min(depth, kMtpPasses);
  const bool kv_only = depth < 1;
  const int passes = kv_only ? 1 : depth;
  for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();  // the pinned records / ids of the last call are free
  // the pages of every drafter KV row the passes write (pass 1 at p0 .. p0 + rows_i - 1, pass k at p0 + keep - 1 + k)
  for (int i = 0; i < ns; ++i) {
    const DSeg &g = segs[static_cast<size_t>(i)];
    kv_need(g.slot, std::max<int64_t>(g.p0 + nrs[i], g.p0 + g.keep + passes - 1));
  }
  if (kv_only && mtp_->defers()) {  // a KV-only pass waits in the slots' backlogs
    const auto t0 = std::chrono::steady_clock::now();
    int sl[k::kSegs16];
    int64_t p0[k::kSegs16];
    for (int i = 0; i < ns; ++i) sl[i] = segs[static_cast<size_t>(i)].slot, p0[i] = segs[static_cast<size_t>(i)].p0;
    int32_t *ids = mtp_->pinned_ids();
    std::memcpy(ids, next, static_cast<size_t>(rows) * 4);
    int32_t rd[k::kSegs16];
    for (int i = 0; i < ns; ++i) rd[i] = static_cast<int32_t>(seqs_[static_cast<size_t>(sl[i])].rope_delta);
    // the card of the fc's hidden half (TP1: the one card); unequal segments: the B-row run, then the B2-row run
    const int kw = code < 0 ? k::seg_het_k(code) : ns, bw = code < 0 ? k::seg_het_b(code) : code, b2 = code < 0 ? k::seg_het_b2(code) : 1;
    mtp_->defer(*head_, kw, bw, sl, p0, h[C - 1], ids, rd);
    if (kw < ns) mtp_->defer(*head_, ns - kw, b2, sl + kw, p0 + kw, h[C - 1] + static_cast<size_t>(kw) * bw * cfg_.hidden, ids + kw * bw, rd + kw);
    draft_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return std::vector<std::vector<int32_t>>(static_cast<size_t>(ns));
  }
  for (const auto &g : segs) mtp_->settle(*head_, g.slot, g.p0);  // the deferred rows below the passes' rows
  int32_t *rh = rec_host_ + kRecInts;  // [pass][kSegs16][kSegRec]
  std::fill(rh, rh + kMtpPasses * kRecInts, 0);
  for (int k = 0; k < passes; ++k)
    for (int i = 0; i < ns; ++i) {
      const DSeg &g = segs[static_cast<size_t>(i)];
      int32_t *w = rh + k * kRecInts + i * k::kSegRec;
      w[k::kSegPos] = static_cast<int32_t>(k == 0 ? g.p0 : g.p0 + g.keep - 1 + k);
      w[k::kSegRope] = static_cast<int32_t>(seqs_[static_cast<size_t>(g.slot)].rope_delta);
      w[k::kSegSlot] = g.slot;
    }
  int32_t *ids = mtp_->pinned_ids();
  std::memcpy(ids, next, static_cast<size_t>(rows) * 4);
  for (size_t c = 0; c < C; ++c) fab_.compute(c).memcpy(mtp_pos_[c], rh, static_cast<size_t>(passes) * kRecInts * 4);
  mtp_->noise_upload();  // the slots' coupled-draft noise (set_draft_noise)
  std::vector<const int32_t *> rec(C), id(C);
  std::vector<const bf16 *> hh(C);
  auto &q0 = fab_.compute(0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < passes; ++k) {
    const int which = k & 1, prev = which ^ 1, r = k == 0 ? rows : ns;
    int64_t bound = b1;
    if (k == 1) {  // every segment's chain starts from its last kept row of pass 1
      std::vector<int> src;
      for (int i = 0; i < ns; ++i) src.push_back(r0s[i] + segs[static_cast<size_t>(i)].keep - 1);
      mtp_->gather_chain(0, src);
    }
    if (k > 0) {
      bound = 0;
      for (const auto &g : segs) bound = std::max(bound, g.p0 + g.keep + k);
    }
    for (size_t c = 0; c < C; ++c) {
      rec[c] = mtp_pos_[c] + k * kRecInts;
      if (k == 0) {
        hh[c] = h[c];
        id[c] = ids;
      } else if (k == 1) {
        hh[c] = mtp_->chain_hidden(static_cast<int>(c));
        id[c] = mtp_->chain_tokens(static_cast<int>(c));
      } else {
        hh[c] = mtp_->hidden(static_cast<int>(c), prev);
        id[c] = mtp_->tokens(static_cast<int>(c), prev);
      }
    }
    mtp_->pass(*head_, r, hh, id, rec, k == 0 ? code : 1, static_cast<int>(std::min<int64_t>(cap_, bound)), which, kv_only, k);
    if (!kv_only) q0.memcpy(mtp_drafts_ + k * k::kStepRows32, mtp_->tokens(0, which), static_cast<size_t>(r) * 4);
  }
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(ns));
  if (!kv_only) {
    int32_t d[kMtpPasses * k::kStepRows32];
    q0.memcpy(d, mtp_drafts_, static_cast<size_t>(passes) * k::kStepRows32 * 4).wait();
    for (int i = 0; i < ns; ++i) {
      out[static_cast<size_t>(i)].push_back(d[r0s[i] + segs[static_cast<size_t>(i)].keep - 1]);
      for (int k = 1; k < passes; ++k) out[static_cast<size_t>(i)].push_back(d[k * k::kStepRows32 + i]);
    }
  }
  for (size_t c = 0; c < C; ++c) fab_.compute(c).wait_and_throw();
  draft_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

std::vector<int32_t> Model::draft(const int32_t *next, int rows, int keep, int depth) {
  if (batch_slots_.size() != 1 || batch_slots_[0] != slot_ || last_rows_ != rows || pending_accept_ != keep)
    throw std::logic_error("qwen35 model: draft() follows step(rows) + accept(keep) of the current slot");
  if (depth < 1 && !dflash_) throw std::invalid_argument("qwen35 model: draft depth >= 1");
  std::vector<const bf16 *> h(cards_.size());
  for (size_t c = 0; c < cards_.size(); ++c) h[c] = head_->card(static_cast<int>(c)).xn;
  return run_drafts(h, rows, {DSeg{slot_, pos_ - keep, keep}}, next, {depth})[0];
}

std::vector<int32_t> Model::draft_after_prefill(int32_t first, int depth, int seqs) {
  if (!pf_committed_) throw std::logic_error("qwen35 model: draft_after_prefill() follows a prefill");
  std::vector<const bf16 *> h(cards_.size());
  // (after a call of several sequences the head's rows belong to all of them; each slot's row was kept)
  const int hx = seqs_[static_cast<size_t>(slot_)].pf_hx;
  for (size_t c = 0; c < cards_.size(); ++c)
    h[c] = hx >= 0 ? cards_[c]->pf_hx + static_cast<size_t>(hx) * cfg_.hidden : head_->card(static_cast<int>(c)).xn + static_cast<size_t>(pf_last_row_) * cfg_.hidden;
  if (dflash_ && !mtp_drafts(seqs)) {  // the prompt is in the context
    double mtp_ms = 0;
    if (mtp_) {  // MTP's KV of the prompt's last row (with `first`), before the DFlash pass reuses the head
      mtp_run(h, 1, {DSeg{slot_, pos_ - 1, 1}}, &first, 0);
      mtp_ms = draft_ms_;
    }
    auto d = dflash_drafts({DSeg{slot_, pos_, 0}}, 0, &first, {depth})[0];
    draft_ms_ += mtp_ms;
    return d;
  }
  if (depth < 1) throw std::invalid_argument("qwen35 model: draft depth >= 1");
  return mtp_run(h, 1, {DSeg{slot_, pos_ - 1, 1}}, &first, depth)[0];  // both: the prompt is in the DFlash context too
}

std::vector<std::vector<int32_t>> Model::draft_batch(const int32_t *next, int depth) { return draft_batch(next, std::vector<int>{depth}); }

std::vector<std::vector<int32_t>> Model::draft_batch(const int32_t *next, const std::vector<int> &depth) {
  if (batch_slots_.empty()) throw std::logic_error("qwen35 model: draft_batch follows verify_batch + accept_batch");
  if (depth.size() != 1 && depth.size() != batch_slots_.size()) throw std::invalid_argument("qwen35 model: draft_batch takes one depth or one per segment");
  save_seq();
  std::vector<DSeg> segs;
  const int ns = static_cast<int>(batch_slots_.size());
  for (int i = 0; i < ns; ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(batch_slots_[static_cast<size_t>(i)])];
    int r0 = 0, nr = 0;
    code_span(batch_code_, ns, i, r0, nr);
    if (sq.pending_accept < 1 || sq.last_rows != nr) throw std::logic_error("qwen35 model: draft_batch follows accept_batch");
    segs.push_back(DSeg{batch_slots_[static_cast<size_t>(i)], sq.pos - sq.pending_accept, sq.pending_accept});
  }
  std::vector<const bf16 *> h(cards_.size());
  for (size_t c = 0; c < cards_.size(); ++c) h[c] = head_->card(static_cast<int>(c)).xn;
  return run_drafts(h, batch_code_, segs, next, depth);
}

std::vector<int32_t> Model::picks(int card, int rows) { return head_->tokens(card, rows); }

void Model::apply_head(const StepOptions &o) {  // 
  if (o.head > 2) throw std::invalid_argument("qwen35 model: StepOptions::head -1..2");
  // -1 = as loaded (arch flint: the checkpoint's own affine int4 rows, no int8 rows)
  const HeadMode m = o.head < 0 ? (opt_.head_w4 ? (opt_.head_rescore > 0 ? HeadMode::rescore : HeadMode::int4) : head_->can(HeadMode::int8) ? HeadMode::int8 : HeadMode::int4)
                     : o.head == 0 ? HeadMode::int8 : o.head == 1 ? HeadMode::int4 : HeadMode::rescore;
  head_->set_mode(m, o.head_tiles > 0 ? o.head_tiles : opt_.head_rescore > 0 ? opt_.head_rescore : kHeadTilesDefault);
}

void Model::logits(int rows, std::vector<float> &out) {
  // both cards' rows copy at once into pinned memory, then one vectorised rounding pass per card's part
  const int vpc = head_->vocab_per_card();
  const size_t C = cards_.size(), V = static_cast<size_t>(vpc) * C, per = static_cast<size_t>(Head::kRows) * vpc;
  if (rows < 0 || rows > head_->rows()) throw std::invalid_argument("model logits: rows 0..rows_cap");
  out.resize(static_cast<size_t>(rows) * V);
  if (!lg_host_) lg_host_ = static_cast<float *>(fab_.alloc_host(C * per * 4));
  std::vector<sycl::event> ev(C);
  for (size_t c = 0; c < C; ++c)
    ev[c] = fab_.compute(c).memcpy(lg_host_ + c * per, head_->card(static_cast<int>(c)).logits, static_cast<size_t>(rows) * vpc * 4);
  for (size_t c = 0; c < C; ++c) {
    ev[c].wait();
    for (int r = 0; r < rows; ++r)
      k::bf16_round_floats(lg_host_ + c * per + static_cast<size_t>(r) * vpc, out.data() + r * V + c * vpc, static_cast<size_t>(vpc));
  }
}

// Log-softmax statistics of the step's logits rows on the cards (kernels::logprob_stats, the qwen38 form): per card
// its vocabulary part's record, the targets' logits (bf16-rounded, as the host copy reports them) from their owning
// cards; the host merges (StepModelAdapter::row_logprobs). Waits.
void Model::logprob_stats(int rows, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt) {
  if (rows < 1 || rows > k::kLpMaxRows || rows > head_->rows()) throw std::invalid_argument("qwen35 model: logprob rows 1..32");
  const int V = head_->vocab_per_card();
  const size_t C = cards_.size();
  if (lp_dev_.empty())
    for (size_t c = 0; c < C; ++c) {
      lp_dev_.push_back(static_cast<float *>(fab_.alloc(c, static_cast<size_t>(k::kLpMaxRows) * k::kLpRec * 4)));
      lp_raw_.push_back(static_cast<float *>(fab_.alloc_host(static_cast<size_t>(k::kLpMaxRows) * 4)));
    }
  for (size_t c = 0; c < C; ++c) fab_.compute(c).wait_and_throw();  // the head's rows written
  std::vector<sycl::event> ev;
  for (int r = 0; r < rows; ++r) {
    const int32_t t = targets[r];
    if (t < 0) continue;
    if (t >= static_cast<int32_t>(C) * V) throw std::invalid_argument("qwen35 model: logprob target outside the vocabulary");
    const size_t c = static_cast<size_t>(t / V);
    ev.push_back(fab_.compute(c).memcpy(lp_raw_[0] + r, head_->card(static_cast<int>(c)).logits + static_cast<size_t>(r) * V + (t - static_cast<int32_t>(c) * V), 4));
  }
  for (auto &e : ev) e.wait_and_throw();
  k::LpArgs a;
  xt.assign(static_cast<size_t>(rows), std::numeric_limits<float>::infinity());
  for (int r = 0; r < rows; ++r)
    if (targets[r] >= 0) xt[static_cast<size_t>(r)] = a.xt[r] = k::bf16_value(k::bf16_bits(lp_raw_[0][r]));
    else a.xt[r] = std::numeric_limits<float>::infinity();
  rec.resize(C * static_cast<size_t>(rows) * k::kLpRec);
  std::vector<sycl::event> cp(C);
  // every card's stats queued before the copies to pageable memory (such a copy blocks until its queue drains)
  for (size_t c = 0; c < C; ++c) k::logprob_stats(fab_.compute(c), head_->card(static_cast<int>(c)).logits, V, rows, V, static_cast<int>(c) * V, top, a, lp_dev_[c]);
  for (size_t c = 0; c < C; ++c)
    cp[c] = fab_.compute(c).memcpy(rec.data() + c * static_cast<size_t>(rows) * k::kLpRec, lp_dev_[c], static_cast<size_t>(rows) * k::kLpRec * 4);
  for (auto &e : cp) e.wait_and_throw();
}

int Model::logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) {
  if (rows < 1 || rows > head_->rows()) throw std::invalid_argument("model logits_cands: rows 1..rows_cap");
  std::vector<const float *> lg;
  for (size_t c = 0; c < cards_.size(); ++c) lg.push_back(head_->card(static_cast<int>(c)).logits);
  return cands_->run(lg, rows, T, ids, vals);
}

std::vector<uint16_t> Model::residual(int card, int rows) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * cfg_.hidden);
  fab_.compute(static_cast<size_t>(card)).memcpy(v.data(), head_->card(card).res, v.size() * 2).wait();
  return v;
}

std::vector<uint16_t> Model::final_hidden(int card, int rows) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * cfg_.hidden);
  fab_.compute(static_cast<size_t>(card)).memcpy(v.data(), head_->card(card).xn, v.size() * 2).wait();
  return v;
}

uint32_t Model::wire_errors() {
  uint32_t e = 0;
  for (size_t c = 0; c < cards_.size(); ++c) {
    uint32_t w = 0;
    fab_.compute(c).memcpy(&w, cards_[c]->wire.error, 4).wait();
    e |= w | mlp_->wire_error(static_cast<int>(c)) | head_->wire_error(static_cast<int>(c)) | (moe_ ? moe_->wire_error(static_cast<int>(c)) : 0u);
  }
  if (mtp_) e |= mtp_->wire_error();
  if (dflash_) e |= dflash_->wire_error();
  return e;
}

}  // namespace xenithium::arch::qwen35
