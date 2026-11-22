#include "xenithium/arch/common/dflash.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "xenithium/kernels/drafter/dspine.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"

namespace xenithium::arch {

namespace k = kernels;
using model::ModelConfig;

namespace {

int as_int(int64_t v, const char *what) {
  if (v < 0 || v > 1 << 30) throw std::runtime_error(std::string("dflash config.json: ") + what + " out of range");
  return static_cast<int>(v);
}
void need(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error("dflash config.json: " + what);
}

// pinned host words (DFlash::host_): block ids, records, anchors, drafts
constexpr int kIdsOff = 0, kRecOff = k::kMaxRows, kAnchorOff = kRecOff + k::kMaxSegs * k::kSegRec, kDraftOff = kAnchorOff + k::kMaxSegs;
constexpr int kPfRecOff = kDraftOff + k::kMaxSegs * k::kMaxRows;
constexpr int kKeysOff = (kPfRecOff + k::kSegRec + 1) / 2 * 2;  // the block rows' top-16 keys (uint64, 8-byte aligned)
// the segments' draft-noise records (k::DraftNoiseRec, 8 words each, 8-byte aligned)
constexpr int kNoiseOff = kKeysOff + 2 * k::kMaxRows * k::kDfTopK;
constexpr int kNoiseWords = static_cast<int>(sizeof(k::DraftNoiseRec) / 4);
constexpr int kHostWords = kNoiseOff + k::kMaxSegs * kNoiseWords;

}  // namespace

bool DFlashConfig::is_dflash(const std::string &dir) {
  try {
    const auto mc = ModelConfig::load(dir);
    for (const auto &a : mc.architectures())
      if (a == "DFlash2DraftModel" || a == "DFlashDraftModel") return true;
  } catch (const std::exception &) {
  }
  return false;
}

DFlashConfig DFlashConfig::load(const std::string &dir, int max_block) {
  const auto mc = ModelConfig::load(dir);
  bool arch = false, ds = false;
  for (const auto &a : mc.architectures()) arch |= a == "DFlash2DraftModel", ds |= a == "DFlashDraftModel";
  need(arch || ds, "architectures does not name DFlash2DraftModel or DFlashDraftModel (" + dir + ")");
  const json::Value &t = mc.root();
  DFlashConfig c;
  c.kind = ds && !arch ? Kind::dspine : Kind::dflash2;
  c.hidden = as_int(ModelConfig::need_int(t, "hidden_size"), "hidden_size");
  c.layers = as_int(ModelConfig::need_int(t, "num_hidden_layers"), "num_hidden_layers");
  c.q_heads = as_int(ModelConfig::need_int(t, "num_attention_heads"), "num_attention_heads");
  c.kv_heads = as_int(ModelConfig::need_int(t, "num_key_value_heads"), "num_key_value_heads");
  c.head_dim = as_int(ModelConfig::need_int(t, "head_dim"), "head_dim");
  c.inter = as_int(ModelConfig::need_int(t, "intermediate_size"), "intermediate_size");
  c.vocab = ModelConfig::need_int(t, "vocab_size");
  c.max_positions = ModelConfig::need_int(t, "max_position_embeddings");
  c.eps = ModelConfig::need_num(t, "rms_norm_eps");
  c.window = as_int(ModelConfig::need_int(t, "sliding_window"), "sliding_window");
  c.target_layers_total = as_int(ModelConfig::need_int(t, "num_target_layers"), "num_target_layers");
  c.causal = ModelConfig::need_bool(t, "is_causal");
  const json::Value &rope = ModelConfig::need(t, "rope_parameters");
  c.theta = ModelConfig::need_num(rope, "rope_theta");
  need(ModelConfig::str_or(rope, "rope_type", "default") == "default", "rope_type other than default is not implemented");
  need(ModelConfig::num_or(t, "partial_rotary_factor", 1.0) == 1.0, "partial rotary is not implemented");
  if (const json::Value *b = t.get("attention_bias")) need(!(b->is_bool() && b->b), "attention_bias true is not implemented");
  need(ModelConfig::str_or(t, "hidden_act", "silu") == "silu", "hidden_act other than silu");
  for (const auto &s : ModelConfig::need_strs(t, "layer_types")) need(s == "sliding_attention", "layer type '" + s + "' (sliding_attention expected)");
  const json::Value &d = ModelConfig::need(t, "dflash_config");
  if (c.kind == Kind::dspine) {  // DSpine's config contract
    c.block = as_int(ModelConfig::need_int(t, "block_size"), "block_size");
    const json::Value &b = ModelConfig::need(d, "selected_prefix_code_bridge");
    need(ModelConfig::need_int(b, "code_size") == k::kDsCode && ModelConfig::need_int(b, "rank") == k::kDsRank, "selected_prefix_code_bridge code_size 1024, rank 512 (kernel constants)");
    c.bridge_eps = ModelConfig::need_num(b, "eps");
    need(ModelConfig::need_bool(d, "draft_block_causal"), "draft_block_causal true");
    need(ModelConfig::str_or(d, "selected_prefix_decode_mode", "") == "serial", "selected_prefix_decode_mode serial");
    need(ModelConfig::str_or(d, "selected_prefix_feature_write_mode", "") == "independent", "selected_prefix_feature_write_mode independent");
    need(!ModelConfig::need_bool(d, "selected_prefix_terminal"), "selected_prefix_terminal false");
    const auto fl = ModelConfig::need_ints(d, "selected_prefix_feature_layers");
    need(static_cast<int>(fl.size()) == c.layers, "selected_prefix_feature_layers = every layer");
    for (size_t i = 0; i < fl.size(); ++i) need(fl[i] == static_cast<int64_t>(i) + 1, "selected_prefix_feature_layers = 1..layers");
    const json::Value &sp = ModelConfig::need(t, "code_bridge_second_pass");
    need(ModelConfig::str_or(sp, "mode", "") == "serial_table" && ModelConfig::need_int(sp, "first_exact") == c.layers - 1 && ModelConfig::need_int(sp, "head_topk") == k::kDfTopK,
         "code_bridge_second_pass: serial_table, first_exact = layers - 1, head_topk 16");
    need(c.causal, "is_causal true");
    c.sel_topk = k::kDfTopK;
  } else {
    c.block = as_int(ModelConfig::need_int(d, "block_size"), "block_size");
    c.conv_group = as_int(ModelConfig::need_int(d, "conv_group_size"), "conv_group_size");
    c.conv_taps = as_int(ModelConfig::need_int(d, "conv_kernel_size"), "conv_kernel_size");
    c.sel_rank = as_int(ModelConfig::need_int(d, "selector_rank"), "selector_rank");
    c.sel_topk = as_int(ModelConfig::need_int(d, "selector_top_k"), "selector_top_k");
    need(!c.causal, "is_causal true (a causal block) is not implemented");
    need(c.conv_taps == 2, "conv_kernel_size 2 (kernel constant)");
    need(c.sel_rank % 16 == 0, "selector_rank a multiple of 16");
    need(c.conv_group > 0 && c.hidden % c.conv_group == 0, "hidden a multiple of conv_group_size");
  }
  c.mask_id = ModelConfig::need_int(d, "mask_token_id");
  for (const int64_t v : ModelConfig::need_ints(d, "target_layer_ids")) c.target_layers.push_back(as_int(v, "target_layer_ids"));
  if (const json::Value *s = d.get("input_embedding_scale")) need(!s->is_number() || s->num == 1.0, "input_embedding_scale != 1 is not implemented");
  if (const json::Value *s = t.get("output_multiplier")) need(!s->is_number() || s->num == 1.0, "output_multiplier != 1 is not implemented");
  if (const json::Value *s = t.get("final_logit_softcapping")) need(s->is_null() || (s->is_number() && s->num <= 0), "final_logit_softcapping is not implemented");
  need(c.head_dim == k::kDfDim, "head_dim 128 (kernel constant)");
  need(c.sel_topk == k::kDfTopK, "selector_top_k 16 (kernel constant)");
  need(c.hidden % 128 == 0, "hidden a multiple of 128");
  need(c.q_heads % c.kv_heads == 0 && c.q_heads / c.kv_heads <= k::kDfMaxGroup, "q heads per kv head <= 8");
  // a pass drafts at most a verify's rows per sequence; DSpine's block is causal, so its first rows are exactly what the
  // trained block gives
  c.trained_block = c.block;
  // the 27B verifies <= 8 rows per sequence (qwen35 kGdnMaxRows); 16 with the 16-row trees (kGdnWideRows)
  c.block = std::min(c.block, max_block);
  need(c.block >= 2 && c.block <= k::kStepRows16, "block_size 2..16 (a verify segment's rows)");
  need(!c.target_layers.empty() && c.mask_id >= 0 && c.mask_id < c.vocab, "target_layer_ids / mask_token_id");
  need(c.window >= 2, "sliding_window");
  return c;
}

std::vector<model::DenseSpec> dflash_dense_plan(const model::Checkpoint &ck, const DFlashConfig &cfg, int dense_bytes) {
  using S = model::DenseSpec;
  if (dense_bytes != 1 && dense_bytes != 2 && dense_bytes != 4) throw std::invalid_argument("dflash plan: dense_bytes 1 (int8), 2 (bf16) or 4 (int4 by RTN)");
  const S::Store mat = dense_bytes == 2 ? S::Store::bf16 : S::Store::i8;
  const S::Store big = dense_bytes == 4 ? S::Store::w4r : mat;  // q, o, gate | up, down
  std::vector<S> out;
  auto add = [&](std::string name, S::Split split, S::Store store, int layer, std::vector<std::string> append = {}) {
    if (!ck.find(name)) throw std::runtime_error("dflash plan: missing " + name);
    for (const auto &a : append)
      if (!ck.find(a)) throw std::runtime_error("dflash plan: missing " + a);
    S sp;
    sp.name = std::move(name);
    sp.split = split;
    sp.store = store;
    sp.append = std::move(append);
    sp.comp = model::Component::mtp;
    sp.layer = layer;
    out.push_back(std::move(sp));
  };
  const bool ds = cfg.kind == DFlashConfig::Kind::dspine;
  add("fc.weight", S::Split::cols, mat, 0);
  add("hidden_norm.weight", S::Split::replicate, S::Store::raw, 0);
  add("norm.weight", S::Split::replicate, S::Store::raw, cfg.layers - 1);
  if (!ds) add("candidate_selector.hidden_projection.weight", S::Split::replicate, mat, cfg.layers - 1);
  for (int l = 0; l < cfg.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    add(L + "input_layernorm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "post_attention_layernorm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "self_attn.q_proj.weight", S::Split::rows, big, l);
    add(L + "self_attn.k_proj.weight", S::Split::rows, mat, l, {L + "self_attn.v_proj.weight"});
    add(L + "self_attn.o_proj.weight", S::Split::cols, big, l);
    add(L + "self_attn.q_norm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "self_attn.k_norm.weight", S::Split::replicate, S::Store::raw, l);
    add(L + "mlp.gate_proj.weight", S::Split::rows, big, l, {L + "mlp.up_proj.weight"});
    add(L + "mlp.down_proj.weight", S::Split::cols, big, l);
    if (ds) continue;  // DSpine: no convolution (its bridge loads in DFlash::load_dspine)
    add(L + "attention_conv.kernel_projection.weight", S::Split::replicate, mat, l);
    add(L + "attention_conv.base_kernel", S::Split::replicate, S::Store::raw, l);
    add(L + "mlp_conv.kernel_projection.weight", S::Split::replicate, mat, l);
    add(L + "mlp_conv.base_kernel", S::Split::replicate, S::Store::raw, l);
  }
  return out;
}

DFlash::DFlash(Fabric &fab, const std::string &dir, const std::string &target_dir, const DFlashTarget &tcfg, int64_t capacity, const DFlashOptions &opt)
    : fab_(fab), dir_(dir), cfg_(DFlashConfig::load(dir, opt.wide > 0 ? k::kStepRows16 : 8)), tgt_(tcfg), opt_(opt) {
  const int C = static_cast<int>(fab.size());
  if (C != 1 && C != 2 && C != 4) throw std::runtime_error("dflash: runs on one card (TP1), two (TP2) or four (TP4)");
  if (opt.slots < 1 || opt.slots > k::kMaxSegs) throw std::invalid_argument("dflash: slots 1..64");
  if ((!tcfg.gather && static_cast<int>(tcfg.embed.size()) != C) || (opt.vocab == 0 && static_cast<int>(tcfg.lm.size()) != C))
    throw std::invalid_argument("dflash: the target's embedding table (or gather) and lm_head rows per card");
  if (cfg_.hidden != tcfg.hidden || cfg_.vocab != tcfg.vocab || cfg_.target_layers_total != tcfg.layers)
    throw std::runtime_error("dflash: the drafter's hidden / vocabulary / target layers do not match the target");
  for (int l : cfg_.target_layers)
    if (l < 0 || l + 1 >= tcfg.layers) throw std::runtime_error("dflash: a target layer id past the target's layers - 1");
  const int H = cfg_.hidden, L = cfg_.layers, D = k::kDfDim;
  if ((cfg_.q_heads % C) || (cfg_.kv_heads % C) || (cfg_.inter % C) || (static_cast<int64_t>(cfg_.target_layers.size()) * H) % (C * 128))
    throw std::runtime_error("dflash: heads / intermediate / features must split over the cards");
  cap_ = capacity;
  hq_ = cfg_.q_heads / C;
  hkv_ = cfg_.kv_heads / C;
  ipc_ = cfg_.inter / C;
  fcols_ = static_cast<int>(cfg_.target_layers.size()) * H / C;
  ring_ = (cfg_.window + 2 * k::kMaxRows + 127) / 128 * 128;  // the window + a block and a verify's rows (no slot read is rewritten in a pass)
  const model::Checkpoint ck = model::Checkpoint::open(dir);
  dense_ = std::make_unique<load::DenseStore>(fab);
  dense_->load(ck, dflash_dense_plan(ck, cfg_, opt.dense_bytes), opt.threads);
  if (opt.vocab > 0) {  // the draft vocabulary: ids [0, K) + 128 rows at the first eos rounded down to 64 (MtpOptions::vocab)
    if (opt.vocab % 32 || tcfg.eos.empty()) throw std::invalid_argument("dflash: draft vocabulary K a multiple of 32, and an eos id");
    if (opt.vocab % (16 * C)) throw std::invalid_argument("dflash: draft vocabulary K a multiple of 64 at four cards (16 rows per card)");
    const int64_t e0 = *std::min_element(tcfg.eos.begin(), tcfg.eos.end()), s0 = e0 / 64 * 64;
    if (opt.vocab > s0 || s0 + 128 > tcfg.vocab) throw std::invalid_argument("dflash: draft vocabulary past the special tokens");
    sub_k_ = opt.vocab;
    sub_s0_ = static_cast<int>(s0);
    const model::Checkpoint tck = model::Checkpoint::open(target_dir);
    model::DenseSpec sp;
    sp.name = opt.head_w4 ? "lm_head.qweight" : "lm_head.weight";  // the target's int4 head rows
    sp.key = "lm_head.weight";
    sp.split = model::DenseSpec::Split::rows;
    sp.store = opt.head_w4 ? model::DenseSpec::Store::w4 : opt.dense_bytes == 2 ? model::DenseSpec::Store::bf16 : model::DenseSpec::Store::i8;
    sp.w4_layout = model::DenseSpec::W4Layout::gptq;
    if (tck.find("lm_head.biases")) {  // an affine int4 head (arch flint): its own rows, as the target reads them
      sp.name = "lm_head.weight";
      sp.store = model::DenseSpec::Store::w4;
      sp.w4_layout = model::DenseSpec::W4Layout::mlx;
      sp.w4_group = k::kW4AffGroup;
    }
    sp.row_ranges = {{0, opt.vocab}, {s0, s0 + 128}};
    sp.comp = model::Component::mtp;
    sp.layer = 0;
    if (!tck.find(sp.name)) throw std::runtime_error("dflash: the target has no " + sp.name + " for the draft vocabulary");
    head_store_ = std::make_unique<load::DenseStore>(fab);
    head_store_->load(tck, {sp}, opt.threads);
    head_n_ = (sub_k_ + 128) / C;
  }
  // rope (Qwen3 default): inv_freq_i = theta^(-2i / 128) in fp32, angle = fp32(pos x inv_freq_i), bf16 cos / sin
  std::vector<uint16_t> cs(static_cast<size_t>(cap_) * (D / 2)), sn(cs.size());
  {
    std::vector<float> inv(D / 2);
    for (int i = 0; i < D / 2; ++i) inv[static_cast<size_t>(i)] = static_cast<float>(1.0 / std::pow(cfg_.theta, static_cast<double>(2 * i) / D));
    for (int64_t p = 0; p < cap_; ++p)
      for (int i = 0; i < D / 2; ++i) {
        const float a = static_cast<float>(p) * inv[static_cast<size_t>(i)];
        cs[static_cast<size_t>(p) * (D / 2) + i] = k::bf16_bits(static_cast<float>(std::cos(static_cast<double>(a))));
        sn[static_cast<size_t>(p) * (D / 2) + i] = k::bf16_bits(static_cast<float>(std::sin(static_cast<double>(a))));
      }
  }
  // the selector's codebooks (card 0 only): bf16 [vocab][rank] each, read as stored
  std::vector<std::pair<const model::TensorInfo *, const bf16 **>> books;
  cards_.resize(static_cast<size_t>(C));
  bytes_.assign(static_cast<size_t>(C), 0);
  if (opt.rows < 0 || opt.rows > kRows) throw std::invalid_argument("dflash: rows 0..kRows");
  rows_ = opt.rows > 0 ? opt.rows : kRows;
  const size_t R = static_cast<size_t>(rows_), NS = static_cast<size_t>(opt.slots);
  const bool copy_wire = C == k::kWireCardsMax && opt.copy_wire;
  const bool ds = cfg_.kind == DFlashConfig::Kind::dspine;
  const int kvn = 2 * hkv_ * D, proj_n = ds ? 0 : 2 * cfg_.conv_taps * (H / cfg_.conv_group);
  for (int c = 0; c < C; ++c) {
    Card &cd = cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(static_cast<size_t>(c));
    auto at = [&](const std::string &n) -> const load::DeviceTensor & { return dense_->at(c, n); };
    auto vec = [&](const std::string &n) { return reinterpret_cast<const bf16 *>(at(n).ptr); };
    auto dev = [&](size_t b) {
      b = std::max<size_t>(b, 4096);
      void *p = fab.alloc(static_cast<size_t>(c), b);
      cd.mem.push_back(p);
      bytes_[static_cast<size_t>(c)] += b;
      return p;
    };
    cd.fc = at("fc.weight").dense();
    if (!ds) cd.sel = at("candidate_selector.hidden_projection.weight").dense();
    cd.hnorm = vec("hidden_norm.weight");
    cd.fnorm = vec("norm.weight");
    if (cd.fc.n != H || cd.fc.k != fcols_ || (!ds && (cd.sel.n != cfg_.sel_rank || cd.sel.k != H))) throw std::runtime_error("dflash: fc / selector slice shapes");
    for (int l = 0; l < L; ++l) {
      const std::string P = "layers." + std::to_string(l) + ".";
      Layer ly;
      ly.in_ln = vec(P + "input_layernorm.weight");
      ly.post_ln = vec(P + "post_attention_layernorm.weight");
      ly.qn = vec(P + "self_attn.q_norm.weight");
      ly.kn = vec(P + "self_attn.k_norm.weight");
      if (!ds) {
        ly.abase = vec(P + "attention_conv.base_kernel");
        ly.mbase = vec(P + "mlp_conv.base_kernel");
        ly.aproj = at(P + "attention_conv.kernel_projection.weight").dense();
        ly.mproj = at(P + "mlp_conv.kernel_projection.weight").dense();
      }
      ly.q = lm_mat(at(P + "self_attn.q_proj.weight"));
      ly.kv = at(P + "self_attn.k_proj.weight").dense();
      ly.o = lm_mat(at(P + "self_attn.o_proj.weight"));
      ly.gate_up = lm_mat(at(P + "mlp.gate_proj.weight"));
      ly.down = lm_mat(at(P + "mlp.down_proj.weight"));
      if (ly.q.n != hq_ * D || ly.q.k != H || ly.kv.n != kvn || ly.kv.k != H || ly.o.n != H || ly.o.k != hq_ * D || ly.aproj.n != proj_n ||
          ly.mproj.n != proj_n || ly.gate_up.n != 2 * ipc_ || ly.down.n != H || ly.down.k != ipc_)
        throw std::runtime_error("dflash: unexpected layer slice shapes (layer " + std::to_string(l) + ")");
      const size_t rb = NS * static_cast<size_t>(ring_) * hkv_ * D * 2;
      ly.ring.k = static_cast<bf16 *>(dev(rb));
      ly.ring.v = static_cast<bf16 *>(dev(rb));
      q.memset(ly.ring.k, 0, rb);  // a key the window never wrote stays finite (0 x V)
      q.memset(ly.ring.v, 0, rb);
      ly.ring.ring = ring_;
      ly.ring.hkv = hkv_;
      cd.layers.push_back(ly);
    }
    if (sub_k_ > 0) {
      head_w_.push_back(lm_mat(head_store_->at(c, "lm_head.weight")));
      if (head_w_.back().n != head_n_ || head_w_.back().k != H) throw std::runtime_error("dflash: draft head slice shape");
    }
    const int vpc = sub_k_ > 0 ? head_n_ : tcfg.lm[static_cast<size_t>(c)].n;
    cd.feat_step = static_cast<bf16 *>(dev(R * fcols_ * 2));
    cd.feat_save = static_cast<bf16 *>(dev(R * fcols_ * 2));
    cd.feat_pf = static_cast<bf16 *>(dev(static_cast<size_t>(opt.prefill_rows) * fcols_ * 2));
    for (bf16 **pp : {&cd.t, &cd.tn, &cd.res, &cd.xn, &cd.xc, &cd.mix, &cd.mlp, &cd.hout}) *pp = static_cast<bf16 *>(dev(R * H * 2));
    cd.part = static_cast<float *>(dev(R * H * 4));
    cd.dyn = static_cast<float *>(dev(R * std::max(proj_n, 16) * 4));
    cd.yq = static_cast<float *>(dev(R * hq_ * D * 4));
    cd.ykv_b = static_cast<float *>(dev(R * kvn * 4));
    cd.ykv_c = static_cast<float *>(dev(R * kvn * 4));
    cd.qb = static_cast<bf16 *>(dev(R * hq_ * D * 2));
    cd.kb = static_cast<bf16 *>(dev(R * hkv_ * D * 2));
    cd.vb = static_cast<bf16 *>(dev(R * hkv_ * D * 2));
    cd.apart = static_cast<float *>(dev(k::df_attn_part_floats(rows_, 1, hq_, cfg_.window) * 4));
    cd.attn = static_cast<bf16 *>(dev(R * hq_ * D * 2));
    cd.gu = static_cast<float *>(dev(R * 2 * ipc_ * 4));
    cd.silu = static_cast<float *>(dev(static_cast<size_t>(k::kSiluTabSize) * 4));
    k::swiglu_table(q, cd.silu);
    cd.h = static_cast<bf16 *>(dev(R * ipc_ * 2));
    cd.logits = static_cast<float *>(dev(R * vpc * 4));
    cd.hp = static_cast<float *>(dev(R * std::max(cfg_.sel_rank, 16) * 4));
    cd.topk_scratch = dev(k::df_topk_scratch_bytes(vpc));
    cd.keys = static_cast<uint64_t *>(dev(R * k::kDfTopK * 8));
    cd.rec = static_cast<int32_t *>(dev(k::kMaxSegs * k::kSegRec * 4));
    cd.pf_rec = static_cast<int32_t *>(dev(k::kSegRec * 4));
    cd.anchors = static_cast<int32_t *>(dev(k::kMaxSegs * 4));
    cd.noise = static_cast<k::DraftNoiseRec *>(dev(k::kMaxSegs * sizeof(k::DraftNoiseRec)));
    q.memset(cd.noise, 0, k::kMaxSegs * sizeof(k::DraftNoiseRec));
    cd.drafts = static_cast<int32_t *>(dev(k::kMaxSegs * k::kMaxRows * 4));
    cd.cos = static_cast<bf16 *>(dev(cs.size() * 2));
    cd.sin = static_cast<bf16 *>(dev(sn.size() * 2));
    q.memcpy(cd.cos, cs.data(), cs.size() * 2);
    q.memcpy(cd.sin, sn.data(), sn.size() * 2);
    // GEMM inputs past the live rows stay finite: zero every activation once
    for (bf16 *p : {cd.t, cd.tn, cd.res, cd.xn, cd.xc, cd.mix, cd.mlp, cd.hout}) q.memset(p, 0, R * H * 2);
    q.memset(cd.feat_step, 0, R * fcols_ * 2);
    q.memset(cd.attn, 0, R * hq_ * D * 2);
    q.memset(cd.h, 0, R * ipc_ * 2);
    q.memset(cd.rec, 0, k::kMaxSegs * k::kSegRec * 4);
    cd.wire.error = static_cast<uint32_t *>(dev(64));
    cd.nanw = static_cast<unsigned *>(dev(64));
    q.memset(cd.nanw, 0, 64);
    q.memset(cd.wire.error, 0, 64);
    if (C > 1) {  // TP1: no wire (the sums are the card's own partials, the top-k its own list)
      const size_t tw_words = C == 4 ? k::kDfTopkWireWords4 : k::kDfTopkWireWords;  // two hops at four cards
      cd.wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(k::pair_wire_words(H, C) * sizeof(uint32_t)));
      cd.mem.push_back(cd.wire.inbox);
      const size_t topk_bytes = tw_words * sizeof(uint32_t);
      cd.tw.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(topk_bytes));
      cd.mem.push_back(cd.tw.inbox);
      k::df_topk_wire_init(q, cd.tw.inbox, tw_words);
    }
    cd.tw.step = static_cast<int32_t *>(dev(64));
    q.memset(cd.tw.step, 0, 64);
    cd.tw.error = cd.wire.error + 1;  // its own word (wire_error: bit 0 the sums, bit 1 the top-k merge)
    if (opt.ctx_defer > 0) {  // the deferred context's rows, scratch and many-row wire
      const size_t B = static_cast<size_t>(opt.ctx_defer);
      cd.backlog = static_cast<bf16 *>(dev(NS * B * fcols_ * 2));
      q.memset(cd.backlog, 0, NS * B * fcols_ * 2);
      cd.fl_t = static_cast<bf16 *>(dev(B * H * 2));
      cd.fl_tn = static_cast<bf16 *>(dev(B * H * 2));
      cd.fl_part = static_cast<float *>(dev(B * H * 4));
      cd.fl_ykv = static_cast<float *>(dev(B * kvn * 4));
      cd.fl_rec = static_cast<int32_t *>(dev(NS * k::kSegRec * 4));
      if (C > 1) {
        cd.fl_wire.rows_max = opt.ctx_defer;
        cd.fl_wire.n_max = H;
        cd.fl_wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(k::pair_rows_wire_words(opt.ctx_defer, H, C) * sizeof(uint32_t)));
        cd.mem.push_back(cd.fl_wire.inbox);
        cd.fl_wire.error = cd.wire.error;
      }
    }
    if (c == 0 && !ds) {
      for (const char *nm : {"candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"}) {
        const model::TensorInfo &ti = ck.at(nm);
        if (ti.dtype != model::DType::bf16 || ti.shape.size() != 2 || ti.shape[0] != tcfg.vocab || ti.shape[1] != cfg_.sel_rank)
          throw std::runtime_error(std::string("dflash: ") + nm + " is not bf16 [vocab][selector_rank]");
        void *dst = dev(ti.bytes);
        void *stage = fab.alloc_host(ti.bytes);
        const int fd = ::open(ck.shards()[ti.shard].path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("dflash: cannot open " + ck.shards()[ti.shard].path);
        size_t got = 0;
        while (got < ti.bytes) {
          const ssize_t r = ::pread(fd, static_cast<char *>(stage) + got, ti.bytes - got, static_cast<off_t>(ti.offset + got));
          if (r <= 0) break;
          got += static_cast<size_t>(r);
        }
        ::close(fd);
        if (got != ti.bytes) throw std::runtime_error(std::string("dflash: short read of ") + nm);
        q.memcpy(dst, stage, ti.bytes).wait();
        fab.release(stage);
        (std::string(nm).find("predecessor") != std::string::npos ? cd.pred_cb : cd.succ_cb) = static_cast<const bf16 *>(dst);
      }
    }
    q.wait();
  }
  for (int c = 0; c < C && C == 2; ++c) {
    Card &cd = cards_[static_cast<size_t>(c)];
    cd.wire.peer_inbox = cards_[static_cast<size_t>(1 - c)].wire.inbox;
    cd.tw.peer_inbox = cards_[static_cast<size_t>(1 - c)].tw.inbox;
    k::pair_wire_init(fab.compute(static_cast<size_t>(c)), H, cd.wire.inbox).wait();
    if (cd.fl_wire.inbox) {
      cd.fl_wire.peer_inbox = cards_[static_cast<size_t>(1 - c)].fl_wire.inbox;
      k::pair_rows_wire_init(fab.compute(static_cast<size_t>(c)), opt.ctx_defer, H, cd.fl_wire.inbox).wait();
    }
  }
  if (C == 4) {  // the four-card sums (recursive doubling, every inbox by the host), the top-k in two hops
    k::WireQuad dq, rq;
    if (copy_wire) {
      fab.alloc_step_copy(step_copy_, rows_, H, false, cards_[0].mem);
      dq.copy = &step_copy_;
      if (opt.ctx_defer > 0) {
        fab.alloc_step_copy(flush_copy_, opt.ctx_defer, H, false, cards_[0].mem);
        rq.copy = &flush_copy_;
      }
    }
    for (int c = 0; c < C; ++c) dq.inbox[c] = cards_[static_cast<size_t>(c)].wire.inbox, rq.inbox[c] = cards_[static_cast<size_t>(c)].fl_wire.inbox;
    for (int c = 0; c < C; ++c) {
      Card &cd = cards_[static_cast<size_t>(c)];
      cd.wire.quad = dq;
      cd.wire.quad.card = c;
      cd.tw.peer_inbox = cards_[static_cast<size_t>(c ^ 1)].tw.inbox;
      cd.tw.peer2_inbox = cards_[static_cast<size_t>(c ^ 2)].tw.inbox;
      k::pair_wire_init(fab.compute(static_cast<size_t>(c)), H, cd.wire.inbox, C).wait();
      if (cd.fl_wire.inbox) {
        cd.fl_wire.quad = rq;
        cd.fl_wire.quad.card = c;
        k::pair_rows_wire_init(fab.compute(static_cast<size_t>(c)), opt.ctx_defer, H, cd.fl_wire.inbox, C).wait();
      }
    }
  }
  host_ = static_cast<int32_t *>(fab.alloc_host(kHostWords * 4));
  std::memset(host_, 0, kHostWords * 4);
  if (ds) load_dspine(dir, target_dir);
}

namespace {
// a bf16 tensor's elements [off, off + n) from its shard (pread)
void read_bf16(const model::Checkpoint &ck, const std::string &nm, size_t off, size_t n, void *dst) {
  const model::TensorInfo &ti = ck.at(nm);
  if (ti.dtype != model::DType::bf16 || (off + n) * 2 > ti.bytes) throw std::runtime_error("dflash: " + nm + " is not bf16 of the expected size");
  const int fd = ::open(ck.shards()[ti.shard].path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("dflash: cannot open " + ck.shards()[ti.shard].path);
  size_t got = 0;
  const size_t want = n * 2;
  while (got < want) {
    const ssize_t r = ::pread(fd, static_cast<char *>(dst) + got, want - got, static_cast<off_t>(ti.offset + off * 2 + got));
    if (r <= 0) break;
    got += static_cast<size_t>(r);
  }
  ::close(fd);
  if (got != want) throw std::runtime_error("dflash: short read of " + nm);
}
}  // namespace

// DSpine's code bridge (replicated per card, bf16), its codebook (pinned host: only anchor rows are read per pass) and
// card 0's vocabulary tables over the draft head's candidate indices (built here on the card: TG = bf16(bf16(lm_head
// row x norm) W_last), TSV = bf16(bf16(32 codebook row) [G_p; A]^T)).
void DFlash::load_dspine(const std::string &dir, const std::string &target_dir) {
  const int H = cfg_.hidden, L = cfg_.layers, Cd = k::kDsCode, Rk = k::kDsRank;
  const size_t R = static_cast<size_t>(rows_), C = cards_.size();
  const model::Checkpoint ck = model::Checkpoint::open(dir);
  auto shape_is = [&](const std::string &nm, std::vector<int64_t> s) {
    if (ck.at(nm).shape != s) throw std::runtime_error("dflash (DSpine): unexpected shape of " + nm);
  };
  shape_is("selected_prefix_bridge.read_weight", {Cd, H});
  shape_is("selected_prefix_bridge.early_read_weights", {L - 1, Cd, H});
  shape_is("selected_prefix_bridge.feature_write_weights", {L, H, Rk});
  for (const char *nm : {"selected_prefix_bridge.gate.weight", "selected_prefix_bridge.predecessor_gate.weight", "selected_prefix_bridge.prev.weight"}) shape_is(nm, {Rk, Cd});
  shape_is("selected_prefix_bridge.gate.bias", {Rk});
  shape_is("selected_prefix_bridge.codebook", {cfg_.vocab, Cd});
  {  // the terminal projection is disabled in the checkpoints this serves: a nonzero one is refused
    std::vector<uint16_t> o(static_cast<size_t>(Cd) * Rk);
    read_bf16(ck, "selected_prefix_bridge.out.weight", 0, o.size(), o.data());
    for (uint16_t v : o)
      if (v & 0x7fff) throw std::runtime_error("dflash (DSpine): a nonzero terminal projection (selected_prefix_bridge.out.weight) is not implemented");
  }
  // bf16 [n][k] row-major -> packed tiles on every card
  auto upload = [&](const std::vector<uint16_t> &w, int n, int kk) {
    std::vector<uint16_t> pk(k::dense_packed_elems(n, kk));
    k::pack_dense_bf16(w.data(), n, kk, pk.data());
    std::vector<k::DenseWeights> out;
    for (size_t c = 0; c < C; ++c) {
      void *d = fab_.alloc(c, pk.size() * 2);
      cards_[c].mem.push_back(d);
      bytes_[c] += pk.size() * 2;
      fab_.compute(c).memcpy(d, pk.data(), pk.size() * 2).wait();
      k::DenseWeights dw;
      dw.w = d;
      dw.kind = k::DenseKind::bf16;
      dw.n = n;
      dw.k = kk;
      out.push_back(dw);
    }
    return out;
  };
  std::vector<uint16_t> buf;
  for (int l = 0; l < L; ++l) {
    buf.resize(static_cast<size_t>(Cd) * H);
    if (l < L - 1) read_bf16(ck, "selected_prefix_bridge.early_read_weights", static_cast<size_t>(l) * Cd * H, buf.size(), buf.data());
    else read_bf16(ck, "selected_prefix_bridge.read_weight", 0, buf.size(), buf.data());
    const auto rw = upload(buf, Cd, H);
    buf.resize(static_cast<size_t>(H) * Rk);
    read_bf16(ck, "selected_prefix_bridge.feature_write_weights", static_cast<size_t>(l) * H * Rk, buf.size(), buf.data());
    const auto ww = upload(buf, H, Rk);
    for (size_t c = 0; c < C; ++c) cards_[c].layers[static_cast<size_t>(l)].bread = rw[c], cards_[c].layers[static_cast<size_t>(l)].bwrite = ww[c];
  }
  std::vector<uint16_t> wl_t(static_cast<size_t>(Rk) * H);  // W_last^T [rank][hidden] (the TG build's weight)
  for (int h = 0; h < H; ++h)
    for (int r = 0; r < Rk; ++r) wl_t[static_cast<size_t>(r) * H + h] = buf[static_cast<size_t>(h) * Rk + r];
  buf.resize(static_cast<size_t>(3) * Rk * Cd);
  read_bf16(ck, "selected_prefix_bridge.gate.weight", 0, static_cast<size_t>(Rk) * Cd, buf.data());
  read_bf16(ck, "selected_prefix_bridge.predecessor_gate.weight", 0, static_cast<size_t>(Rk) * Cd, buf.data() + static_cast<size_t>(Rk) * Cd);
  read_bf16(ck, "selected_prefix_bridge.prev.weight", 0, static_cast<size_t>(Rk) * Cd, buf.data() + static_cast<size_t>(2) * Rk * Cd);
  const auto pw = upload(buf, 3 * Rk, Cd);
  const std::vector<uint16_t> p2(buf.begin() + static_cast<std::ptrdiff_t>(Rk) * Cd, buf.end());  // [G_p; A] (the TSV build's weight)
  std::vector<uint16_t> bias(static_cast<size_t>(Rk));
  read_bf16(ck, "selected_prefix_bridge.gate.bias", 0, bias.size(), bias.data());
  for (size_t c = 0; c < C; ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    auto dev = [&](size_t b) {
      b = std::max<size_t>(b, 4096);
      void *p = fab_.alloc(c, b);
      cd.mem.push_back(p);
      bytes_[c] += b;
      q.memset(p, 0, b);
      return p;
    };
    cd.bpack = pw[c];
    auto *bb = static_cast<bf16 *>(dev(bias.size() * 2));
    q.memcpy(bb, bias.data(), bias.size() * 2);
    cd.bbias = bb;
    cd.brms = static_cast<float *>(dev(R * 4));
    cd.bproj = static_cast<float *>(dev(R * Cd * 4));
    cd.bcode = static_cast<bf16 *>(dev(R * Cd * 2));
    cd.bpacked = static_cast<float *>(dev(R * 3 * Rk * 4));
    cd.bmsg = static_cast<bf16 *>(dev(R * Rk * 2));
    q.wait();
  }
  // the codebook, pinned host
  const size_t cb_elems = static_cast<size_t>(cfg_.vocab) * Cd;
  codebook_ = static_cast<bf16 *>(fab_.alloc_host(cb_elems * 2));
  read_bf16(ck, "selected_prefix_bridge.codebook", 0, cb_elems, codebook_);
  // card 0's tables over the candidate indices: v < sub_k -> token v, else v - sub_k + sub_s0 (the draft vocabulary); all
  // of the vocabulary without one
  const int64_t n = sub_k_ > 0 ? sub_k_ + 128 : cfg_.vocab;
  auto tok_of = [&](int64_t v) { return sub_k_ > 0 && v >= sub_k_ ? v - sub_k_ + sub_s0_ : v; };
  Card &c0 = cards_[0];
  auto &q = fab_.compute(0);
  auto alloc0 = [&](size_t b) {
    void *p = fab_.alloc(0, std::max<size_t>(b, 4096));
    c0.mem.push_back(p);
    bytes_[0] += b;
    return p;
  };
  auto *tg = static_cast<bf16 *>(alloc0(static_cast<size_t>(n) * Rk * 2));
  auto *tsv = static_cast<bf16 *>(alloc0(static_cast<size_t>(n) * 2 * Rk * 2));
  c0.tg = tg;
  c0.tsv = tsv;
  // the build's weights on card 0 (freed after)
  auto pack0 = [&](const std::vector<uint16_t> &w, int nn, int kk, void *&mem) {
    std::vector<uint16_t> pk(k::dense_packed_elems(nn, kk));
    k::pack_dense_bf16(w.data(), nn, kk, pk.data());
    mem = fab_.alloc(0, pk.size() * 2);
    q.memcpy(mem, pk.data(), pk.size() * 2).wait();
    k::DenseWeights dw;
    dw.w = mem;
    dw.kind = k::DenseKind::bf16;
    dw.n = nn;
    dw.k = kk;
    return dw;
  };
  void *m_wl = nullptr, *m_p2 = nullptr;
  const k::DenseWeights w_tg = pack0(wl_t, Rk, H, m_wl), w_tsv = pack0(p2, 2 * Rk, Cd, m_p2);
  const model::Checkpoint tck = model::Checkpoint::open(target_dir);
  const char *head_nm = tck.find("lm_head.weight") ? "lm_head.weight" : nullptr;
  if (!head_nm || tck.at(head_nm).dtype != model::DType::bf16 || tck.at(head_nm).shape != std::vector<int64_t>{cfg_.vocab, H})
    throw std::runtime_error("dflash (DSpine): the target's lm_head.weight must be bf16 [vocab][hidden] (the tables' rows)");
  constexpr int kChunk = 4096;
  auto *stage = static_cast<uint16_t *>(fab_.alloc_host(static_cast<size_t>(kChunk) * H * 2));
  void *a = fab_.alloc(0, static_cast<size_t>(kChunk) * H * 2), *a2 = fab_.alloc(0, static_cast<size_t>(kChunk) * H * 2);
  for (int64_t r0 = 0; r0 < n; r0 += kChunk) {
    const int m = static_cast<int>(std::min<int64_t>(kChunk, n - r0));
    // head rows of the chunk's tokens (contiguous runs: at most two, the draft vocabulary's specials start a new one)
    for (int i = 0; i < m;) {
      const int64_t t0 = tok_of(r0 + i);
      int run = 1;
      while (i + run < m && tok_of(r0 + i + run) == t0 + run) ++run;
      read_bf16(tck, head_nm, static_cast<size_t>(t0) * H, static_cast<size_t>(run) * H, stage + static_cast<size_t>(i) * H);
      i += run;
    }
    q.memcpy(a, stage, static_cast<size_t>(m) * H * 2).wait();  // the stage is rewritten below
    k::ds_scale_rows(q, m, H, static_cast<const bf16 *>(a), c0.fnorm, 1.f, static_cast<bf16 *>(a2));
    k::pf_dense(q, m, w_tg, static_cast<const bf16 *>(a2), H, k::PfOut{nullptr, tg + static_cast<size_t>(r0) * Rk, Rk});
    // codebook rows of the same tokens (pinned host: gather on the host into the stage)
    for (int i = 0; i < m; ++i) std::memcpy(stage + static_cast<size_t>(i) * Cd, codebook_ + static_cast<size_t>(tok_of(r0 + i)) * Cd, static_cast<size_t>(Cd) * 2);
    q.wait();  // the head rows' GEMM has read a
    q.memcpy(a, stage, static_cast<size_t>(m) * Cd * 2);
    k::ds_scale_rows(q, m, Cd, static_cast<const bf16 *>(a), nullptr, 32.f, static_cast<bf16 *>(a2));
    k::pf_dense(q, m, w_tsv, static_cast<const bf16 *>(a2), Cd, k::PfOut{nullptr, tsv + static_cast<size_t>(r0) * 2 * Rk, 2 * Rk});
    q.wait();
  }
  for (void *p : {a, a2, m_wl, m_p2}) fab_.release(p);
  fab_.release(stage);
}

DFlash::~DFlash() {
  for (size_t c = 0; c < fab_.size(); ++c) fab_.compute(c).wait();
  graphs_.clear();
  for (auto &cd : cards_)
    for (void *p : cd.mem) fab_.release(p);
  if (host_) fab_.release(host_);
  if (codebook_) fab_.release(codebook_);
}

size_t DFlash::card_bytes(int c) const {
  size_t b = bytes_[static_cast<size_t>(c)];
  for (const auto &kv : dense_->stats().card_bytes[static_cast<size_t>(c)]) b += kv.second;
  if (head_store_)
    for (const auto &kv : head_store_->stats().card_bytes[static_cast<size_t>(c)]) b += kv.second;
  return b;
}

int DFlash::feature_of(int layer) const {
  for (size_t j = 0; j < cfg_.target_layers.size(); ++j)
    if (cfg_.target_layers[j] == layer) return static_cast<int>(j);
  return -1;
}

void DFlash::capture(int c, sycl::queue &q, int j, const bf16 *res, int rows, bf16 *dst) const {
  const int H = cfg_.hidden;
  const int lo = std::max(j * H, c * fcols_), hi = std::min((j + 1) * H, (c + 1) * fcols_);
  if (hi > lo) k::df_copy_cols(q, rows, res, H, lo - j * H, dst, fcols_, lo - c * fcols_, hi - lo);
}

void DFlash::capture_mean(int c, sycl::queue &q, int j, const bf16 *res, int streams, int rows, bf16 *dst) const {
  const int H = cfg_.hidden;
  const int lo = std::max(j * H, c * fcols_), hi = std::min((j + 1) * H, (c + 1) * fcols_);
  if (hi > lo) k::df_mean_cols(q, rows, res, streams, H, lo - j * H, dst, fcols_, lo - c * fcols_, hi - lo);
}

const char *const *DFlash::stage_names() {
  static const char *const n[kStages] = {"context fc + norm", "context k|v + prep", "norm + conv proj + conv", "q, k|v", "prep + attention",
                                         "o + sum + conv", "mlp norm + conv proj + conv", "gate|up + swiglu", "down + sum + conv", "head + top-k + select"};
  return n;
}

// Every GPU operation of a pass (pass() fills the pinned words first). Recorded into graphs, or run directly with
// DFlashOptions::profile (each stage waited and timed).
namespace {
// The confidence cut a context pass (or a pass with nothing to trim) uses: the shipped default.
constexpr float kConfCutDefault = 0.02f;
}  // namespace

void DFlash::submit(int ns, int ctx_rows, int block_rows, int ctx_row0, int tree) {
  const int H = cfg_.hidden, rc = ns * ctx_rows, rb = ns * block_rows;
  if (const char *e = std::getenv("XEN_DRAFT_TRACE"); e && *e == '1')
    std::fprintf(stderr, "dflash submit: ns %d ctx_rows %d block_rows %d ctx_row0 %d tree %d -> cut %.4g%s\n", ns, ctx_rows, block_rows, ctx_row0, tree,
                 static_cast<double>(ctx_rows > 0 ? kConfCutDefault : opt_.conf_cut), ctx_rows > 0 ? " (context: default)" : " (verify: configured)");
  // The block pass's layout limits (DFlash 2: block 8, a verify segment's rows <= 16). `pass` refuses the first two, so
  // a fault under a deep check shows up as a CCS timeout with no numbers; print them once instead.
  if (block_rows > cfg_.block || rb > ns * k::kStepRows16 || ctx_rows > k::kStepRows16) {
    static bool said = false;
    if (!said) {
      said = true;
      std::fprintf(stderr,
                   "dflash: block pass over its layout: ns %d, block_rows %d, ctx_rows %d, ctx_row0 %d, cfg_.block %d, "
                   "kStepRows16 %d, rb %d\n",
                   ns, block_rows, ctx_rows, ctx_row0, cfg_.block, k::kStepRows16, rb);
    }
  }
  const size_t C = cards_.size();
  const float eps = static_cast<float>(cfg_.eps);
  auto wire_of = [&](Card &cd) -> const k::PairWire * { return C > 1 ? &cd.wire : nullptr; };  // TP1: no two-card sum
  int32_t *ids = host_ + kIdsOff, *rec = host_ + kRecOff, *anc = host_ + kAnchorOff;
  // every pair_reduce on the wire alternates its parity: draw one only for a sum that runs
  const int pc = rc > 0 ? wire_parity() : 0;  // the fc sum (context rows)
  auto t_mark = std::chrono::steady_clock::now();
  auto mark = [&](int stage) {  // profile only: every stage waited on both cards
    if (!opt_.profile) return;
    for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
    const auto now = std::chrono::steady_clock::now();
    stage_ms_[stage] += std::chrono::duration<double, std::milli>(now - t_mark).count();
    t_mark = now;
  };
  if (opt_.profile) {
    for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();
    t_mark = std::chrono::steady_clock::now();
  }
  // Keep invalid token IDs out of the embedding gather. The verifier logs the source rows separately.
  {
    static unsigned long bad_ids = 0;
    unsigned bad_here = 0;
    for (int r = 0; r < ns * block_rows; ++r) {
      if (ids[r] < 0 || static_cast<int64_t>(ids[r]) >= tgt_.vocab) {
        if (std::getenv("XEN_DF_STAGE")) ++bad_here;
        ids[r] = static_cast<int32_t>(cfg_.mask_id);
      }
    }
    if (bad_here) {
      bad_ids += bad_here;
      std::fprintf(stderr, "qwen38 adapter: %u block ids outside [0, %lld) mapped to mask_id %lld (total %lu)\n",
                   bad_here, static_cast<long long>(tgt_.vocab), static_cast<long long>(cfg_.mask_id), bad_ids);
    }
  }
  const bool watch_input = rb > 0 && std::getenv("XEN_DF_STAGE");
  for (size_t c = 0; c < C; ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    if (watch_input) q.memset(cd.nanw, 0, sizeof(unsigned));
    q.memcpy(cd.rec, rec, static_cast<size_t>(ns) * k::kSegRec * 4);
    // DSpine: every card's bridge reads the anchors (before the fc sum: a host copy behind a queued mailbox wait can hold
    // the submitting thread until the peer's 1 s wait gives up, eager / profile passes)
    if (rb > 0 && cfg_.kind == DFlashConfig::Kind::dspine) q.memcpy(cd.anchors, anc, static_cast<size_t>(ns) * 4);
    if (rc > 0) {
      const bf16 *features = cd.feat_step + static_cast<size_t>(ctx_row0) * fcols_;
      if (watch_input) k::df_nanwatch(q, rc * fcols_, features, cd.nanw, 1u << 13);
      k::dense_gemm(q, rc, features, cd.fc, cd.part);
      k::pair_reduce(q, rc, H, cd.part, wire_of(cd), pc, cd.t);
      if (watch_input) k::df_nanwatch(q, rc * H, cd.t, cd.nanw, 1u << 14);
      k::rms_norm_plain(q, rc, H, cd.t, nullptr, cd.hnorm, eps, cd.tn);
    }
    if (rb > 0) {
      if (tgt_.gather) tgt_.gather(q, static_cast<int>(c), ids, rb, cd.res);  // the target's own gather (arch flint)
      else k::embed_gather(q, tgt_.embed[c], ids, rb, H, 1, cd.res);
      if (watch_input) {
        k::df_nanwatch(q, H, cd.res, cd.nanw, 1u << 0);
        if (block_rows > 1) k::df_nanwatch(q, (block_rows - 1) * H, cd.res + H, cd.nanw, 1u << 9);
      }
    }
  }
  mark(0);
  const k::DfRope rope{nullptr, nullptr, static_cast<int>(cap_)};
  // gate | up + swiglu: the dense GEMM and the swiglu launch, or (int4 matrices) the W4 GEMM with swiglu in its epilogue
  auto gate_up = [&](sycl::queue &q, int rows, const bf16 *x, const Layer &ly, Card &cd) {
    if (ly.gate_up.is_w4()) {
      k::w4a16_gemm_rows_swiglu(q, rows, x, ly.gate_up.w4.w, ly.gate_up.w4.scales, cd.silu, cd.h, ipc_, H);
      return;
    }
    k::dense_gemm(q, rows, x, ly.gate_up.d, cd.gu);
    k::swiglu(q, rows, ipc_, cd.gu, cd.h);
  };
  // Stage-major: each stage is submitted for both cards before the next (a card's sum never waits on the peer's host
  // submission of the whole layer); profile marks sit between the stages.
  for (int l = 0; l < cfg_.layers; ++l) {
    const int pa = rb > 0 ? wire_parity() : 0, pm = rb > 0 ? wire_parity() : 0;
    auto each = [&](int stage, const std::function<void(Card &, const Layer &, sycl::queue &)> &fn) {
      for (size_t c = 0; c < C; ++c) fn(cards_[c], cards_[c].layers[static_cast<size_t>(l)], fab_.compute(c));
      mark(stage);
    };
    const int pn = cards_[0].layers[static_cast<size_t>(l)].aproj.n;
    if (rc > 0)  // the verified rows' context K / V
      each(1, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        k::dense_gemm(q, rc, cd.tn, ly.kv, cd.ykv_c);
        k::df_prep(q, rc, ctx_rows, cd.rec, hq_, hkv_, nullptr, cd.ykv_c, ly.qn, ly.kn, eps, k::DfRope{cd.cos, cd.sin, rope.positions}, nullptr, nullptr, nullptr, ly.ring);
      });
    if (rb == 0) continue;
    if (cfg_.kind == DFlashConfig::Kind::dspine) {  // DSpine's layer (no convolution, a causal block) + its bridge
      const float beps = static_cast<float>(cfg_.bridge_eps);
      each(2, [&](Card &cd, const Layer &ly, sycl::queue &q) { k::rms_norm_plain(q, rb, H, cd.res, nullptr, ly.in_ln, eps, cd.xn); });
      each(3, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        lm_gemm(q, rb, cd.xn, ly.q, cd.yq);
        k::dense_gemm(q, rb, cd.xn, ly.kv, cd.ykv_b);
      });
      each(4, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        k::df_prep(q, rb, block_rows, cd.rec, hq_, hkv_, cd.yq, cd.ykv_b, ly.qn, ly.kn, eps, k::DfRope{cd.cos, cd.sin, rope.positions}, cd.qb, cd.kb, cd.vb, ly.ring);
        k::df_attn(q, rb, block_rows, cd.rec, hq_, hkv_, cd.qb, cd.kb, cd.vb, ly.ring, cfg_.window, cd.apart, cd.attn, true);
      });
      each(5, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        lm_gemm(q, rb, cd.attn, ly.o, cd.part);
        k::pair_reduce(q, rb, H, cd.part, wire_of(cd), pa, cd.mix);
      });
      each(6, [&](Card &cd, const Layer &ly, sycl::queue &q) { k::rms_norm_plain(q, rb, H, cd.res, cd.mix, ly.post_ln, eps, cd.xn); });
      each(7, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        gate_up(q, rb, cd.xn, ly, cd);
      });
      each(8, [&](Card &cd, const Layer &ly, sycl::queue &q) {
        lm_gemm(q, rb, cd.h, ly.down, cd.part);
        k::pair_reduce(q, rb, H, cd.part, wire_of(cd), pm, cd.mlp);
        // the bridge: raw = res + mlp -> codes -> the predecessor message -> res = raw + rms delta (rows past the anchor)
        k::ds_bridge_in(q, rb, H, cd.res, cd.mlp, beps, cd.brms, cd.xc);
        k::dense_gemm(q, rb, cd.xc, ly.bread, cd.bproj);
        k::ds_codes(q, rb, block_rows, cd.bproj, cd.anchors, codebook_, beps, cd.bcode);
        k::dense_gemm(q, rb, cd.bcode, cd.bpack, cd.bpacked);
        k::ds_message(q, rb, block_rows, cd.bpacked, cd.bbias, cd.bmsg);
        k::dense_gemm(q, rb, cd.bmsg, ly.bwrite, cd.part);
        k::ds_bridge_out(q, rb, block_rows, H, cd.res, cd.brms, cd.part);
      });
      continue;
    }
    // attention sublayer: norm, prepare conv, q | k | v, prep, attention, o + sum, finish conv
    each(2, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      k::rms_norm_plain(q, rb, H, cd.res, l > 0 ? cd.mlp : nullptr, ly.in_ln, eps, cd.xn);
      k::dense_gemm(q, rb, cd.xn, ly.aproj, cd.dyn);
      k::df_conv(q, rb, block_rows, H, cfg_.conv_group, cd.xn, cd.dyn, pn, 0, ly.abase, cd.xc);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].xn, cards_[wc].nanw, 1u << 1);
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].xc, cards_[wc].nanw, 1u << 3);
    }
    each(3, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      lm_gemm(q, rb, cd.xc, ly.q, cd.yq);
      k::dense_gemm(q, rb, cd.xc, ly.kv, cd.ykv_b);
    });
    each(4, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      k::df_prep(q, rb, block_rows, cd.rec, hq_, hkv_, cd.yq, cd.ykv_b, ly.qn, ly.kn, eps, k::DfRope{cd.cos, cd.sin, rope.positions}, cd.qb, cd.kb, cd.vb, ly.ring);
      if (std::getenv("XEN_DF_STAGE") && l == 0) {
        k::df_nanwatch(q, rb * hq_ * k::kDfDim, cd.qb, cd.nanw, 1u << 10);
        k::df_nanwatch(q, rb * hkv_ * k::kDfDim, cd.kb, cd.nanw, 1u << 11);
        k::df_nanwatch(q, rb * hkv_ * k::kDfDim, cd.vb, cd.nanw, 1u << 12);
      }
      k::df_attn(q, rb, block_rows, cd.rec, hq_, hkv_, cd.qb, cd.kb, cd.vb, ly.ring, cfg_.window, cd.apart, cd.attn);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * hq_ * k::kDfDim, cards_[wc].attn, cards_[wc].nanw, 1u << 4);
    }
    each(5, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      lm_gemm(q, rb, cd.attn, ly.o, cd.part);
      k::pair_reduce(q, rb, H, cd.part, wire_of(cd), pa, cd.mix);
      k::df_conv(q, rb, block_rows, H, cfg_.conv_group, cd.mix, cd.dyn, pn, 1, ly.abase + 2 * H, cd.xc);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].mix, cards_[wc].nanw, 1u << 5);
    }
    // MLP sublayer
    each(6, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      k::rms_norm_plain(q, rb, H, cd.res, cd.xc, ly.post_ln, eps, cd.xn);
      k::dense_gemm(q, rb, cd.xn, ly.mproj, cd.dyn);
      k::df_conv(q, rb, block_rows, H, cfg_.conv_group, cd.xn, cd.dyn, pn, 0, ly.mbase, cd.xc);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].xn, cards_[wc].nanw, 1u << 6);
    }
    each(7, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      gate_up(q, rb, cd.xc, ly, cd);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].h, cards_[wc].nanw, 1u << 7);
    }
    each(8, [&](Card &cd, const Layer &ly, sycl::queue &q) {
      lm_gemm(q, rb, cd.h, ly.down, cd.part);
      k::pair_reduce(q, rb, H, cd.part, wire_of(cd), pm, cd.mix);
      k::df_conv(q, rb, block_rows, H, cfg_.conv_group, cd.mix, cd.dyn, pn, 1, ly.mbase + 2 * H, cd.mlp);
    });
    if (std::getenv("XEN_DF_STAGE") && rb > 0 && l == 0) {
      for (size_t wc = 0; wc < C; ++wc) k::df_nanwatch(fab_.compute(wc), rb * H, cards_[wc].mlp, cards_[wc].nanw, 1u << 8);
    }
  }
  if (rb > 0) {
    for (size_t c = 0; c < C; ++c) {
      Card &cd = cards_[c];
      auto &q = fab_.compute(c);
      k::rms_norm_plain(q, rb, H, cd.res, cfg_.kind == DFlashConfig::Kind::dspine ? nullptr : cd.mlp, cd.fnorm, eps, cd.hout);
      const LmMat &hw = head_w_.empty() ? tgt_.lm[c] : head_w_[c];
      lm_gemm(q, rb, cd.hout, hw, cd.logits);
      k::df_topk(q, cd.logits, hw.n, rb, hw.n, static_cast<int>(c) * hw.n, cd.topk_scratch, C > 1 ? &cd.tw : nullptr, cd.keys);
    }
    // card 0 selects only after both cards' merges are queued: a device-to-host copy can block the submitting thread
    // until its queue drains, and card 0's merge waits on card 1's list (a 1 s mailbox timeout per pass)
    Card &cd = cards_[0];
    auto &q = fab_.compute(0);
    q.memcpy(cd.noise, host_ + kNoiseOff, static_cast<size_t>(ns) * sizeof(k::DraftNoiseRec));  // read at replay
    if (cfg_.kind == DFlashConfig::Kind::dspine) {
      const k::DsTail tail{cd.res, H, eps, cd.brms, cd.bmsg};
      if (tree > 0)  // DSpine's best-first tree
        k::ds_tree(q, ns, block_rows, cd.keys, cd.bpacked, cd.tsv, cd.tg, cd.bbias, tail, sub_k_, sub_s0_, tree, static_cast<float>(opt_.tree_cut), cd.drafts,
                   static_cast<float>(1.0 / (opt_.tree_temp > 0 ? opt_.tree_temp : 1.5)));  // DSpine's auto 1.5
      else
        k::ds_select(q, ns, block_rows, cd.keys, cd.bpacked, cd.tsv, cd.tg, cd.bbias, tail, sub_k_, sub_s0_, cd.drafts, cd.noise);
    } else {
      k::dense_gemm(q, rb, cd.hout, cd.sel, cd.hp);
      q.memcpy(cd.anchors, anc, static_cast<size_t>(ns) * 4);
      if (tree > 0)  // the best-first tree (its words in the drafts' buffers)
        k::df_tree(q, ns, block_rows, cd.keys, cd.hp, cfg_.sel_rank, cd.pred_cb, cd.succ_cb, cd.anchors, sub_k_, sub_s0_, tree, static_cast<float>(opt_.tree_cut), cd.drafts,
                   static_cast<float>(1.0 / (opt_.tree_temp > 0 ? opt_.tree_temp : 1.0)));  // DFlash 2's auto 1
      else
        k::df_select(q, ns, block_rows, cd.keys, cd.hp, cfg_.sel_rank, cd.pred_cb, cd.succ_cb, cd.anchors, sub_k_, sub_s0_, cd.drafts, cd.noise);
    }
    q.memcpy(host_ + kDraftOff, cd.drafts, static_cast<size_t>(tree > 0 ? ns * k::kDfTreeInts : ns * (block_rows - 1)) * 4);
    q.memcpy(host_ + kKeysOff, cd.keys, static_cast<size_t>(rb) * k::kDfTopK * 8);  // the drafts' confidences
  }
  mark(9);
}

std::vector<std::vector<int32_t>> DFlash::pass(const std::vector<Seg> &segs, int ctx_rows, int block_rows, int ctx_row0, int step_seqs) {
  const int ns = static_cast<int>(segs.size());
  if (ns < 1 || ns > k::kMaxSegs || ctx_rows < 0 || block_rows < 0 || block_rows == 1 || ctx_row0 < 0 || ctx_row0 + ns * ctx_rows > rows_ || ns * block_rows > rows_ ||
      block_rows > cfg_.block || (ctx_rows == 0 && block_rows == 0))
    throw std::invalid_argument("dflash: pass segments / rows");
  for (const auto &g : segs)
    if (g.slot < 0 || g.slot >= opt_.slots || g.ctx_p0 < 0 || g.ctx_p0 + ctx_rows > cap_ || (block_rows > 0 && (g.start < 1 || g.start + block_rows > cap_)))
      throw std::invalid_argument("dflash: pass slot / positions");
  const size_t C = cards_.size();
  const int rb = ns * block_rows;
  if (opt_.ctx_defer > 0) {  // the deferred context
    if (block_rows == 0) {   // context only: stash the rows, no pass
      const auto t0 = std::chrono::steady_clock::now();
      defer(segs, ctx_rows, ctx_row0);
      ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      return std::vector<std::vector<int32_t>>(static_cast<size_t>(ns));
    }
    // the block reads the ring below its start: the deferred rows first (this pass's own context rows rewrite from ctx_p0)
    for (const auto &g : segs) settle(g.slot, ctx_rows > 0 ? g.ctx_p0 : g.start);
  }
  for (size_t c = 0; c < C; ++c) fab_.compute(c).wait();  // the pinned words of the last pass are free
  const auto t0 = std::chrono::steady_clock::now();
  // A tree for one greedy sequence (DFlash 2's selector or DSpine's), or a tree per segment for steps of up to tree_seqs
  // sequences. Sampled segments draft the same tree (their noise unused: the engine walks the tree with the sampler's
  // draws). The tree holds at most the step's depth (block_rows - 1 drafts): a best-first tree of that budget, not the
  // preorder prefix of a larger one (several sequences draft shallower blocks).
  const int tree = opt_.tree > 0 && std::max(ns, step_seqs) <= std::max(1, opt_.tree_seqs) && block_rows >= 2
                       ? std::min({block_rows > k::kStepRows8 ? opt_.tree_wide : opt_.tree, k::kTreeRows - 1, block_rows - 1})
                       : 0;
  tree_par_.assign(static_cast<size_t>(ns), {});
  int32_t *ids = host_ + kIdsOff, *rec = host_ + kRecOff, *anc = host_ + kAnchorOff;
  std::memset(rec, 0, k::kMaxSegs * k::kSegRec * 4);
  auto *nz = reinterpret_cast<k::DraftNoiseRec *>(host_ + kNoiseOff);
  for (int i = 0; i < ns; ++i) {
    const Seg &g = segs[static_cast<size_t>(i)];
    nz[i] = static_cast<size_t>(g.slot) < noise_.size() ? noise_[static_cast<size_t>(g.slot)] : k::DraftNoiseRec{};
    int32_t *w = rec + i * k::kSegRec;
    w[k::kDfRecStart] = static_cast<int32_t>(g.start);
    w[k::kDfRecCtx] = static_cast<int32_t>(g.ctx_p0);
    w[k::kDfRecSlot] = g.slot;
    anc[i] = g.anchor;
    for (int r = 0; r < block_rows; ++r) ids[i * block_rows + r] = r == 0 ? g.anchor : static_cast<int32_t>(cfg_.mask_id);
  }
  if (opt_.profile) {  // stage waits inside: no graph
    submit(ns, ctx_rows, block_rows, ctx_row0, tree);
    ++stage_passes_;
  } else {
    // One recorded graph per card per (segments, context rows, block rows, first context row, start parity); every
    // per-pass input is in the pinned words / device records.
    const GraphKey key{ns, ctx_rows, block_rows, ctx_row0, par_, tree};
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
      namespace sx = sycl::ext::oneapi::experimental;
      const auto tr = std::chrono::steady_clock::now();
      std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
      for (size_t c = 0; c < C; ++c) g.emplace_back(fab_.compute(c).get_context(), fab_.compute(c).get_device());
      const int p0 = par_;
      for (size_t c = 0; c < C; ++c) g[c].begin_recording(fab_.compute(c));
      try {
        submit(ns, ctx_rows, block_rows, ctx_row0, tree);
      } catch (...) {
        for (size_t c = 0; c < C; ++c) g[c].end_recording();
        par_ = p0;
        throw;
      }
      for (size_t c = 0; c < C; ++c) g[c].end_recording();
      Graph e;
      e.end_par = par_;
      par_ = p0;
      for (size_t c = 0; c < C; ++c) e.exec.push_back(g[c].finalize());
      it = graphs_.emplace(key, std::move(e)).first;
      graphs_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
    }
    // both cards' graphs in flight together: the mailbox waits inside pair up across the cards
    for (size_t c = 0; c < C; ++c) fab_.compute(c).ext_oneapi_graph(it->second.exec[c]);
    par_ = it->second.end_par;
  }
  for (size_t c = 0; c < C; ++c) fab_.compute(c).wait_and_throw();
  if (const char *d = std::getenv("XEN_CHAIN_TRACE"); d && *d == '1' && rb > 0 && block_rows >= 2) {
    // the same device buffer into a second landing zone: distinguishes an empty device buffer from a copy that
    // never reached host_
    static std::vector<int32_t> probe;
    const size_t cnt = static_cast<size_t>(ns) * static_cast<size_t>(block_rows - 1);
    probe.assign(cnt, -12345);
    // the tail inputs: hp (selector hidden) and logits, is this pass writing them at all?
    {
      static std::vector<float> hpp, lgp;
      static int in_prints = 0;
      const char *limit = std::getenv("XEN_DF_TRACE_PASSES");
      if (in_prints < (limit ? std::max(0, std::atoi(limit)) : 6)) {
        ++in_prints;
        const size_t hpn = static_cast<size_t>(ns) * static_cast<size_t>(block_rows) * static_cast<size_t>(cfg_.sel_rank);
        hpp.assign(hpn, -99.f);
        lgp.assign(32, -99.f);
        fab_.compute(0).memcpy(hpp.data(), cards_[0].hp, hpn * sizeof(float)).wait();
        fab_.compute(0).memcpy(lgp.data(), cards_[0].logits, lgp.size() * sizeof(float)).wait();
        size_t hz = 0, hnan = 0;
        for (float v : hpp) { if (v == 0.f) ++hz; if (!(v == v)) ++hnan; }
        {
          if (std::getenv("XEN_DF_STAGE") && !tgt_.gather) {
            for (const int32_t token : {ids[0], ids[1]}) {
              const auto *row = tgt_.embed[0] + static_cast<size_t>(token) * cfg_.hidden;
              int nonfinite = 0;
              for (int col = 0; col < cfg_.hidden; ++col)
                nonfinite += !std::isfinite(static_cast<float>(row[col]));
              uint16_t bits[4];
              std::memcpy(bits, row, sizeof(bits));
              std::fprintf(stderr, "DFTABLE token %d nonfinite %d bits %04x %04x %04x %04x\n",
                           token, nonfinite, bits[0], bits[1], bits[2], bits[3]);
            }
          }
          if (std::getenv("XEN_DF_STAGE")) {
            const Seg &seg = segs.front();
            const size_t width = static_cast<size_t>(hkv_) * k::kDfDim;
            for (size_t c = 0; c < C; ++c) {
              const auto &ring = cards_[c].layers.front().ring;
              std::vector<uint16_t> vals(static_cast<size_t>(ring_) * width);
              for (const auto *ptr : {ring.k, ring.v}) {
                fab_.compute(c).memcpy(vals.data(), ptr + static_cast<size_t>(seg.slot) * vals.size(),
                                      vals.size() * sizeof(uint16_t)).wait();
                int first = -1, last = -1, bad = 0;
                for (int64_t pos = std::max<int64_t>(0, seg.start - cfg_.window); pos < seg.start; ++pos) {
                  const auto at = vals.begin() + static_cast<size_t>(pos % ring_) * width;
                  if (std::any_of(at, at + width, [](uint16_t v) { return (v & 0x7f80u) == 0x7f80u; })) {
                    if (first < 0) first = static_cast<int>(pos);
                    last = static_cast<int>(pos);
                    ++bad;
                  }
                }
                std::fprintf(stderr, "DFRING card %zu %s start %lld bad_rows %d first %d last %d\n", c,
                             ptr == ring.k ? "key" : "value", static_cast<long long>(seg.start), bad, first, last);
              }
            }
          }
          unsigned nw[8] = {0,0,0,0,0,0,0,0};
          for (size_t c = 0; c < C; ++c) fab_.compute(c).memcpy(&nw[c], cards_[c].nanw, sizeof(unsigned)).wait();
          std::fprintf(stderr, "DFNANW card bits %08x %08x %08x %08x\n", nw[0], nw[1], nw[2], nw[3]);
          if (rb > 0 && ns > 0) std::fprintf(stderr, "DFIDS block anchor %d mask %lld rows %d (vocab %lld)\n", ids[0], static_cast<long long>(cfg_.mask_id), block_rows, static_cast<long long>(tgt_.vocab));
        }
        {
          static std::vector<uint16_t> fb, rb_, hb;
          fb.assign(8, 0), rb_.assign(8, 0), hb.assign(8, 0);
          fab_.compute(0).memcpy(fb.data(), cards_[0].feat_step, 8 * sizeof(uint16_t)).wait();
          fab_.compute(0).memcpy(rb_.data(), cards_[0].res, 8 * sizeof(uint16_t)).wait();
          fab_.compute(0).memcpy(hb.data(), cards_[0].hout, 8 * sizeof(uint16_t)).wait();
          std::fprintf(stderr, "DFSRC feat_step %04x %04x %04x %04x | res %04x %04x %04x %04x | hout %04x %04x %04x %04x\n",
                       fb[0], fb[1], fb[2], fb[3], rb_[0], rb_[1], rb_[2], rb_[3], hb[0], hb[1], hb[2], hb[3]);
        }
        std::fprintf(stderr, "DFIN hp %.4g %.4g %.4g %.4g rank %d n %zu zeros %zu nan %zu | logits %.4g %.4g %.4g %.4g\n",
                     static_cast<double>(hpp[0]), static_cast<double>(hpp[1]), static_cast<double>(hpp[2]), static_cast<double>(hpp[3]),
                     cfg_.sel_rank, hpn, hz, hnan, static_cast<double>(lgp[0]), static_cast<double>(lgp[1]),
                     static_cast<double>(lgp[2]), static_cast<double>(lgp[3]));
      }
    }
    fab_.compute(0).memcpy(probe.data(), cards_[0].drafts, cnt * sizeof(int32_t)).wait();
    std::fprintf(stderr, "DFPROBE device %d %d %d | host_ %d %d %d | ns %d block_rows %d\n", probe[0], cnt > 1 ? probe[1] : -1,
                 cnt > 2 ? probe[2] : -1, host_[kDraftOff], host_[kDraftOff + 1], host_[kDraftOff + 2], ns, block_rows);
  }
  if (const char *d = std::getenv("XEN_CHAIN_TRACE"); d && *d == '1' && rb > 0)
    std::fprintf(stderr, "DFIDS host_drafts %d %d %d %d\n", host_[kDraftOff], host_[kDraftOff + 1], host_[kDraftOff + 2], host_[kDraftOff + 3]);
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(ns));
  if (const char *e = std::getenv("XEN_DRAFT_TRACE"); e && *e == '1' && rb > 0)
    std::fprintf(stderr, "dflash verify shape: ns %d block_rows %d tree %d rb %d | kTreeRows %d kDfTopK %d kStepRows16 %d\n", ns, block_rows, tree, rb,
                 k::kTreeRows, k::kDfTopK, k::kStepRows16);
  if (rb > 0 && tree > 0) {  // the tree's nodes (join order) as verify rows (preorder), per segment
    for (int s = 0; s < ns; ++s) {
      const int32_t *w = host_ + kDraftOff + s * k::kDfTreeInts;
      const int n = std::clamp(w[0], 0, k::kTreeRows - 1);
      int jpar[k::kTreeRows] = {-1}, node[k::kTreeRows] = {}, par[k::kTreeRows] = {};
      for (int i = 1; i <= n; ++i) jpar[i] = w[k::kTreeRows + i];
      k::tree_preorder(jpar, n + 1, node, par);
      tree_par_[static_cast<size_t>(s)].assign(par, par + n + 1);
      for (int r = 1; r <= n; ++r) out[static_cast<size_t>(s)].push_back(w[node[r]]);
    }
  } else if (rb > 0)
    for (int i = 0; i < ns; ++i) {
      auto &o = out[static_cast<size_t>(i)];
      o.assign(host_ + kDraftOff + i * (block_rows - 1), host_ + kDraftOff + (i + 1) * (block_rows - 1));
      // one sequence only: a step of several shares its rows (the shortest segment sets them for all)
      const double cut = ctx_rows > 0 ? kConfCutDefault : opt_.conf_cut;  // a context pass keeps the default cut
      const auto invalid = std::find_if(o.begin(), o.end(), [&](int32_t token) { return token < 0 || token >= tgt_.vocab; });
      if (invalid != o.end()) {
        std::fprintf(stderr, "dflash: out-of-vocabulary draft %d at index %td; discarding the suffix\n", *invalid, invalid - o.begin());
        o.erase(invalid, o.end());
      }
      const size_t pre = o.size();
      if (cut > 0 && ns == 1) o.resize(static_cast<size_t>(confident(o, i * block_rows, static_cast<float>(cut))));
      if (const char *d = std::getenv("XEN_CHAIN_TRACE"); d && *d == '1')
        std::fprintf(stderr, "CHAIN slot %d requested_drafts %zu block_rows %d returned %zu cut %.3f first_id %d ctx_rows %d ids %d %d %d\n",
                     segs[static_cast<size_t>(i)].slot, pre, block_rows, o.size(), cut, pre > 0 ? o[0] : -999, ctx_rows,
                     pre > 0 ? o[0] : -999, pre > 1 ? o[1] : -999, pre > 2 ? o[2] : -999);
    }
  ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

void DFlash::prefill_ctx(int m, int64_t pos0, int slot, const std::vector<Lend> &lend, int parity, int row0) {
  if (m < 1 || row0 < 0 || row0 + m > opt_.prefill_rows || pos0 < 0 || pos0 + m > cap_ || slot < 0 || slot >= opt_.slots)
    throw std::invalid_argument("dflash: prefill rows");
  settle(slot, pos0);  // the slot's deferred rows below the chunk first
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.eps);
  // Stage-major over the cards: every card's fc and sum are queued before any card's later work. A sum waits on the
  // device for the peer card's partial, so a host wait after card 0's sum but before card 1's fc (pf_dense's oneDNN
  // route checks a new shape with a read-back and frees a scratch: both wait for the queue) would stall card 0 until the
  // sum's bounded wait gives up and leave a wrong context sum.
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    // the record from kernel arguments: no pinned words a later chunk could rewrite before the copy runs (the 27B waits
    // after every chunk; qwen38's next chunk may be submitted before this one ran)
    k::df_rec_set(q, cd.pf_rec, pos0, slot);
    k::pf_dense(q, m, cd.fc, cd.feat_pf + static_cast<size_t>(row0) * fcols_, fcols_, k::PfOut{lend[c].part, nullptr, H});
  }
  for (size_t c = 0; c < cards_.size(); ++c) k::pair_reduce_rows(fab_.compute(c), m, H, lend[c].part, lend[c].wire, parity, lend[c].t);
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    const Lend &b = lend[c];
    auto &q = fab_.compute(c);
    k::rms_norm_plain(q, m, H, b.t, nullptr, cd.hnorm, eps, b.tn);
    const k::DfRope rp{cd.cos, cd.sin, static_cast<int>(cap_)};
    for (const Layer &ly : cd.layers) {
      k::pf_dense(q, m, ly.kv, b.tn, H, k::PfOut{b.ykv, nullptr, ly.kv.n});
      k::df_prep(q, m, 0, cd.pf_rec, hq_, hkv_, nullptr, b.ykv, ly.qn, ly.kn, eps, rp, nullptr, nullptr, nullptr, ly.ring);
    }
  }
}

void DFlash::save_step_features(int rows) {
  if (rows < 1 || rows > rows_) throw std::invalid_argument("dflash: save_step_features rows");
  for (size_t c = 0; c < cards_.size(); ++c)
    fab_.compute(c).memcpy(cards_[c].feat_save, cards_[c].feat_step, static_cast<size_t>(rows) * fcols_ * 2);
}

void DFlash::stage_step_features(const int (&map)[kRows], int n) {
  for (size_t c = 0; c < cards_.size(); ++c) k::df_rows_gather(fab_.compute(c), n, fcols_, cards_[c].feat_save, cards_[c].feat_step, map);
}

void DFlash::set_noise(int slot, uint64_t key, uint64_t pos, float inv_t) {
  if (slot < 0 || slot >= opt_.slots) return;
  if (noise_.size() != static_cast<size_t>(opt_.slots)) noise_.assign(static_cast<size_t>(opt_.slots), k::DraftNoiseRec{});
  auto &r = noise_[static_cast<size_t>(slot)];
  r.key = key, r.pos = pos, r.inv_t = inv_t;
}

// How many of a segment's drafts to keep: the chain of their confidences (each draft's softmax probability among its
// block row's top-16 logits; draft j comes from block row j + 1) stays >= conf_cut.
int DFlash::confident(const std::vector<int32_t> &drafts, int row0, float cut) const {
  const auto *keys = reinterpret_cast<const uint64_t *>(host_ + kKeysOff);
  const bool trace = std::getenv("XEN_CONF_TRACE") != nullptr;  // diagnostic: which draft fails to match a key row
  double p = 1;
  for (size_t j = 0; j < drafts.size(); ++j) {
    const uint64_t *kr = keys + (static_cast<size_t>(row0) + j + 1) * k::kDfTopK;
    double mx = -1e30, se = 0, sel = -1e30;
    for (int b = 0; b < k::kDfTopK; ++b) mx = std::max(mx, static_cast<double>(k::df_key_value(kr[b])));
    for (int b = 0; b < k::kDfTopK; ++b) {
      const double v = k::df_key_value(kr[b]);
      se += std::exp(v - mx);
      int64_t t = k::df_key_index(kr[b]);
      if (sub_k_ > 0 && t >= sub_k_) t = t - sub_k_ + sub_s0_;
      if (t == drafts[j]) sel = v;
    }
    const double pr = sel > -1e29 ? std::exp(sel - mx) / se : 0.0;
    if (j == 0) {
      if (const char *d = std::getenv("XEN_CHAIN_TRACE"); d && *d == '1') {
        int64_t braw = -1, brem = -1, r0 = -1, r1 = -1, r2 = -1;
        double bv = -1e30;
        for (int b = 0; b < k::kDfTopK; ++b) {
          const double v = k::df_key_value(kr[b]);
          int64_t t = k::df_key_index(kr[b]);
          const int64_t tr = (sub_k_ > 0 && t >= sub_k_) ? t - sub_k_ + sub_s0_ : t;
          if (b == 0) r0 = t;
          if (b == 1) r1 = t;
          if (b == 2) r2 = t;
          if (v > bv) bv = v, braw = t, brem = tr;
        }
        std::fprintf(stderr,
                     "CHAINID row0 %d draft %d argmax_raw %lld argmax_remap %lld key0..2 %lld %lld %lld sub_k %d sub_s0 %d matched %d\n",
                     row0, drafts[j], static_cast<long long>(braw), static_cast<long long>(brem), static_cast<long long>(r0),
                     static_cast<long long>(r1), static_cast<long long>(r2), sub_k_, sub_s0_, sel > -1e29 ? 1 : 0);
      }
    }
    p *= pr;
    if (trace)
      std::fprintf(stderr, "conf j=%zu draft=%d matched=%d prob=%.3e p=%.3e cut=%.3g\n", j, drafts[j],
                   sel > -1e29 ? 1 : 0, pr, p, static_cast<double>(cut));
    // A draft that does not appear in its block row's top-16 has an unknown, not zero, probability: stopping the
    // chain there is right, but returning 0 throws away the guesses already accepted and degenerates the step to
    // plain decode (1 tok/step). Keep at least the first guess, which is what a cut of 0.02 buys on the reference
    // rig (2.25 tok/step on prose).
    if (p < cut) return std::max(1, static_cast<int>(j));
  }
  return static_cast<int>(drafts.size());
}

// ---- the deferred context ----
// A context-only pass's rows (the verified rows of a step without drafts) are stashed per slot; the ring sees them at
// the next settle(), as one many-row pass (fc, sum, norm, five kv projections, ring writes: prefill_ctx's kernels).
// Rows of a later step that start inside the stashed rows replace the rows from there on (a step rewrites the
// positions its last verify rejected), as the passes would have in order.
void DFlash::defer(const std::vector<Seg> &segs, int ctx_rows, int ctx_row0) {
  const int ns = static_cast<int>(segs.size()), B = opt_.ctx_defer;
  if (ctx_rows < 1) return;
  if (ctx_rows > B) throw std::invalid_argument("dflash: context rows past the deferred backlog");
  if (pend_.empty()) pend_.resize(static_cast<size_t>(opt_.slots));
  int64_t dst[k::kMaxSegs] = {};
  for (int i = 0; i < ns; ++i) {
    const Seg &g = segs[static_cast<size_t>(i)];
    Pend &p = pend_[static_cast<size_t>(g.slot)];
    if (p.n > 0 && (g.ctx_p0 < p.p0 || g.ctx_p0 > p.p0 + p.n)) settle(g.slot, g.ctx_p0);  // not a continuation
    if (p.n == 0) p.p0 = g.ctx_p0;
    int keep = static_cast<int>(g.ctx_p0 - p.p0);
    if (keep + ctx_rows > B) {  // full: the rows before these into the ring now
      settle(g.slot, g.ctx_p0);
      p.p0 = g.ctx_p0;
      keep = 0;
    }
    dst[i] = static_cast<int64_t>(g.slot) * B + keep;
    p.n = keep + ctx_rows;
  }
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    k::df_ctx_stash(fab_.compute(c), ns, ctx_rows, fcols_, cd.feat_step + static_cast<size_t>(ctx_row0) * fcols_, cd.backlog, dst);
  }
}

void DFlash::settle(int slot, int64_t pos) {
  if (pend_.empty() || slot < 0 || slot >= opt_.slots) return;
  Pend &p = pend_[static_cast<size_t>(slot)];
  if (p.n == 0) return;
  const int64_t keep = std::clamp<int64_t>(pos - p.p0, 0, p.n);
  if (keep > 0) flush(slot, static_cast<int>(keep));
  p.n = 0;
}

void DFlash::flush(int slot, int m) {
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.eps);
  const int par = fl_par_;
  fl_par_ ^= 1;
  const int64_t p0 = pend_[static_cast<size_t>(slot)].p0;
  // stage-major over the cards, as prefill_ctx (no host wait between a card's sum and its peer's fc)
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    k::df_rec_set(q, cd.fl_rec + static_cast<size_t>(slot) * k::kSegRec, p0, slot);
    const bf16 *feat = cd.backlog + static_cast<size_t>(slot) * opt_.ctx_defer * fcols_;
    k::pf_dense(q, m, cd.fc, feat, fcols_, k::PfOut{cd.fl_part, nullptr, H});
  }
  for (size_t c = 0; c < cards_.size(); ++c)
    k::pair_reduce_rows(fab_.compute(c), m, H, cards_[c].fl_part, cards_.size() > 1 ? &cards_[c].fl_wire : nullptr, par, cards_[c].fl_t);
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    int32_t *rec = cd.fl_rec + static_cast<size_t>(slot) * k::kSegRec;
    k::rms_norm_plain(q, m, H, cd.fl_t, nullptr, cd.hnorm, eps, cd.fl_tn);
    const k::DfRope rp{cd.cos, cd.sin, static_cast<int>(cap_)};
    for (const Layer &ly : cd.layers) {
      k::pf_dense(q, m, ly.kv, cd.fl_tn, H, k::PfOut{cd.fl_ykv, nullptr, ly.kv.n});
      k::df_prep(q, m, 0, rec, hq_, hkv_, nullptr, cd.fl_ykv, ly.qn, ly.kn, eps, rp, nullptr, nullptr, nullptr, ly.ring);
    }
  }
  ++flushes_;
}

uint32_t DFlash::wire_error() {
  uint32_t e = 0;
  for (size_t c = 0; c < cards_.size(); ++c) {
    uint32_t v = 0;
    uint32_t v2[2] = {0, 0};
    fab_.compute(c).memcpy(v2, cards_[c].wire.error, 8).wait();
    v = (v2[0] ? 1u : 0u) | (v2[1] ? 2u : 0u);
    e |= v;
  }
  return e;
}

std::vector<uint16_t> DFlash::read_hidden(int rows) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * cfg_.hidden);
  fab_.compute(0).memcpy(v.data(), cards_[0].hout, v.size() * 2).wait();
  return v;
}

std::vector<uint64_t> DFlash::read_keys(int rows) {
  std::vector<uint64_t> v(static_cast<size_t>(rows) * k::kDfTopK);
  fab_.compute(0).memcpy(v.data(), cards_[0].keys, v.size() * 8).wait();
  return v;
}

std::vector<float> DFlash::read_hp(int rows) {
  std::vector<float> v(static_cast<size_t>(rows) * cfg_.sel_rank);
  fab_.compute(0).memcpy(v.data(), cards_[0].hp, v.size() * 4).wait();
  return v;
}

size_t DFlash::ring_slot_bytes(int c) const {
  return cards_[static_cast<size_t>(c)].layers.size() * 2 * static_cast<size_t>(ring_) * hkv_ * k::kDfDim * 2;
}

std::vector<std::pair<void *, size_t>> DFlash::ring_pieces(int c, int slot) const {
  const size_t n = static_cast<size_t>(ring_) * hkv_ * k::kDfDim;
  std::vector<std::pair<void *, size_t>> v;
  for (const auto &ly : cards_[static_cast<size_t>(c)].layers)
    for (bf16 *r : {ly.ring.k, ly.ring.v}) v.push_back({r + static_cast<size_t>(slot) * n, n * 2});
  return v;
}

void DFlash::ring_copy(int c, sycl::queue &q, int slot, void *host, bool save) const {
  const size_t n = static_cast<size_t>(ring_) * hkv_ * k::kDfDim;  // elements of one slot's ring of one layer
  auto *h = static_cast<bf16 *>(host);
  for (const auto &ly : cards_[static_cast<size_t>(c)].layers)
    for (bf16 *r : {ly.ring.k, ly.ring.v}) {
      bf16 *d = r + static_cast<size_t>(slot) * n;
      if (save) q.memcpy(h, d, n * 2);
      else q.memcpy(d, h, n * 2);
      h += n;
    }
}

std::vector<uint16_t> DFlash::read_ring(int c, int layer, int slot, bool v) {
  settle(slot, std::numeric_limits<int64_t>::max());  // the deferred rows first
  const auto &ly = cards_[static_cast<size_t>(c)].layers.at(static_cast<size_t>(layer));
  const size_t n = static_cast<size_t>(ring_) * hkv_ * k::kDfDim;
  std::vector<uint16_t> out(n);
  fab_.compute(static_cast<size_t>(c)).memcpy(out.data(), (v ? ly.ring.v : ly.ring.k) + static_cast<size_t>(slot) * n, n * 2).wait();
  return out;
}

}  // namespace xenithium::arch
