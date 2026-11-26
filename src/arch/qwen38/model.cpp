#include "xenithium/arch/qwen38/model.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <filesystem>
#include <future>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include "xenithium/util/env.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/kernels/residual/hc.hpp"
#include "xenithium/kernels/head/logprob.hpp"
#include "xenithium/kernels/pack.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/util/host_mem.hpp"
#include "model_internal.hpp"
#include "xenithium/kernels/gemm/onednn_pf.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"

namespace xenithium::arch::qwen38 {

namespace k = xenithium::kernels;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
double since_s(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
double since_ms(std::chrono::steady_clock::time_point t0) { return since_s(t0) * 1e3; }

constexpr double kGiB = 1073741824.0;

// Free VRAM per card (sysman, bytes); empty when any card does not report it.
std::vector<double> vram_free_per_card(const l0::Loader &loader, const Fabric &fab, int cards) {
  std::vector<double> free;
  for (size_t c = 0; c < static_cast<size_t>(cards); ++c) {
    const auto v = l0::vram_state(loader.info(fab.rank_index(c)));
    if (!v.ok) return {};
    free.push_back(static_cast<double>(v.free));
  }
  return free;
}

// The oneDNN prefill route's repack slots, sized at the largest trunk matrix (the PLE kv; the lm_head never
// takes the route: its many-row calls are < 1,024 rows), capped there; the GiB per card it keeps (0 = route off).
double onednn_cap_gib(const load::DenseStore &dense, int cards) {
  if (k::pf_dense_onednn() <= 0) return 0;
  size_t mb = 0;
  for (int c = 0; c < cards; ++c) {
    size_t b = 0;
    int64_t r = 0;
    dense.max_i8(c, &b, &r, {"lm_head"});
    mb = std::max(mb, b);
  }
  k::pf_dense_set_onednn_cap(mb);
  return static_cast<double>(k::onednn_dense_scratch_bytes(mb)) / kGiB;
}

// One 64-token page of one QSA layer: K, V, their scales and the indexer's compressed keys.
size_t qsa_page_bytes(const k::QsaShape &s) {
  return k::qsa_page_kv_bytes(s) + k::qsa_page_v_bytes(s) + (k::qsa_page_scale_floats(s) + k::qsa_page_vscale_floats(s)) * 4 + k::qsa_page_kc_elems() * 2;
}
}  // namespace

struct Model::Graphs {
  using Exec = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
  struct Entry {
    std::vector<Exec> exec;  // [card]
    int end_step = 0, end_pair = 0, end_moe = 0;  // parities after the recorded step
    // The head's HC state indices after the recorded step, per card: the mixers flip them on the
    // host at submission, so a replay must advance them too, or host reads of the residual
    // (multi_hidden, cards_agree, the drafter's input) see the other buffer.
    struct Hc {
      int cur = 0, lcur = 0, lora_ld = 0;
    };
    std::vector<Hc> start_hc, end_hc;  // [card]
    std::vector<std::pair<int, int>> smarks[kMaxCards];  // the stamps it carries (XEN_DEC_PROFILE), [card] (part, word)
    uint64_t used = 0;  // Model::graph_tick_ at the last replay (graph_room's LRU order)
    int launches[kMaxCards][StepProfile::kParts] = {};  // its launches per module (stamps excluded)
  };
  std::map<std::tuple<int, int, int, int, int, int64_t>, Entry> map;  // rows, seg_rows, parities, bound
};

Model::Model(const l0::Loader &loader, Fabric &fab, const std::string &dir, const ModelOptions &opt)
    : loader_(&loader), fab_(fab), opt_(opt), ck_(model::Checkpoint::open(dir)) {
  const auto t_all = std::chrono::steady_clock::now();
  if (fab.size() != 1 && fab.size() != 2 && fab.size() != static_cast<size_t>(kMaxCards))
    throw std::runtime_error("qwen38 model: runs on one card (TP1), two (TP2 + EP2) or four (TP4)");
  nc_ = static_cast<int>(fab.size());  // TP1, TP2 or TP4
  if (nc_ == 1 && !opt.moe_tp) throw std::invalid_argument("qwen38 model: TP1 needs the expert TP layout (every expert whole on the card)");
  if (nc_ == 4 && !opt.moe_tp) throw std::invalid_argument("qwen38 model: TP4 needs the expert TP layout (every expert in quarters; there is no EP4)");
  if (nc_ != 2 && opt.calib) throw std::invalid_argument("qwen38 model: calibration runs at TP2");
  if (opt.kv_split == k::QsaKvSplit::tokens && nc_ != kMaxCards)
    throw std::invalid_argument("qwen38 model: the KV split by tokens needs four cards (TP4: the two cards of a kv head split its tokens)");
  const auto mc = model::ModelConfig::load(dir);
  const auto arch = model::open_arch(mc);
  cfg_ = Config::parse(mc);
  cap_ = std::max<int64_t>(64, (opt.context + 63) / 64 * 64);
  // with the DFlash drafter a sequence's segment carries up to 8 rows (the 7 drafts' verify): the GDN
  // pending logs hold 8 rows, its 8-row core runs the segments of more than 4 rows or accepts
  seg_rows_max_ = opt.dflash_dir.empty() || opt.calib ? k::kGdnRows : k::kGdnMaxRows;
  if (cap_ > k::kQsaMaxTokens) throw std::runtime_error("qwen38 model: context above the select limit (1,048,576)");
  // 17-64 slots on four cards (opt-in: more streams in one step)
  if (opt.slots < 1 || opt.slots > (nc_ == k::kWireCardsMax ? k::kMaxSegs : k::kSegs16))
    throw std::invalid_argument("qwen38 model: slots 1.." + std::to_string(nc_ == k::kWireCardsMax ? k::kMaxSegs : k::kSegs16) + " (" + std::to_string(nc_) + " cards)");
  if (opt.step_rows_cap != 0 && opt.step_rows_cap != k::kStepRows32 && opt.step_rows_cap != k::kMaxRows)
    throw std::invalid_argument("qwen38 model: step_rows_cap 0, 32 or 64");
  step_cap_ = opt.step_rows_cap > 0 ? opt.step_rows_cap : k::kStepRows32;
  if (step_cap_ > k::kStepRows32 && nc_ != k::kWireCardsMax)
    throw std::invalid_argument("qwen38 model: steps above 32 rows need four cards (their rows run by owner)");
  const size_t NS = static_cast<size_t>(opt.slots);
  seqs_.resize(NS);
  {  // the KV pages, fixed per slot (identity) or one pool shared by the slots
    const int slot_pages = static_cast<int>(cap_ / k::kQsaPage);
    const int64_t all = static_cast<int64_t>(slot_pages) * opt.slots;
    if (opt.kv_pool_tokens == 0) {
      kvp_ = model::KvPages(opt.slots, slot_pages, static_cast<int>(all), false);
    } else {
      int64_t pool = (opt.kv_pool_tokens + k::kQsaPage - 1) / k::kQsaPage;
      if (pool < slot_pages)
        throw std::runtime_error("qwen38 model: the KV pool (" + std::to_string(pool * k::kQsaPage) + " tokens) holds less than one context of " +
                                 std::to_string(cap_) + " tokens");
      pool = std::min(pool, all);  // more could never be used
      kvp_ = model::KvPages(opt.slots, slot_pages, static_cast<int>(pool), true, opt.kv_pool_scramble);
    }
  }

  // ---- non-expert weights ----
  model::DeployParams p;
  p.cards = nc_;
  p.context = cap_;
  p.kv_shard = opt.kv_split == k::QsaKvSplit::tokens ? model::DeployParams::KvShard::tokens : model::DeployParams::KvShard::heads;
  p.kv_bytes = opt.kv == k::QsaKv::bf16 ? 2 : 1;
  p.kv_v4 = k::qsa_kv_v4(opt.kv);
  auto plan = arch->dense_plan(ck_, p);
  for (auto &sp : plan)
    if (sp.store == model::DenseSpec::Store::i8)
      for (const auto &pat : opt.bf16_tensors)
        if (!pat.empty() && sp.name.find(pat) != std::string::npos) {
          sp.store = model::DenseSpec::Store::bf16;
          ++ls_.bf16_overrides;
          break;
        }
  if (opt.gdn_w4 & 3)  // GDN projections from the directory's int4 GPTQ tensors (xenithium-bake gdn4-calib)
    for (auto &sp : plan) {
      if (sp.store != model::DenseSpec::Store::i8 ||
          !(((opt.gdn_w4 & 1) && sp.name.find(".linear_attn.in_proj_qkv.weight") != std::string::npos) ||
            ((opt.gdn_w4 & 2) && sp.name.find(".linear_attn.out_proj.weight") != std::string::npos)))
        continue;
      auto packed = [](const std::string &n) { return n.substr(0, n.size() - 6) + "qweight"; };  // "...weight" -> "...qweight"
      if (!ck_.find(packed(sp.name)))
        throw std::runtime_error("qwen38 model: gdn_w4: " + packed(sp.name) + " missing (make the directory with xenithium-bake gdn4-calib)");
      sp.key = sp.name;
      sp.name = packed(sp.name);
      sp.store = model::DenseSpec::Store::w4;
      sp.w4_layout = model::DenseSpec::W4Layout::gptq;
      for (auto &a : sp.append) a = packed(a);
      ++ls_.w4_overrides;
    }
  for (auto &sp : plan)  // int4 RTN for the matrices whose consumer has a W4 path
    if (sp.store == model::DenseSpec::Store::i8)
      for (const auto &pat : opt.w4_tensors)
        if (!pat.empty() && sp.name.find(pat) != std::string::npos) {
          if (sp.name.find(".linear_attn.in_proj_qkv.") == std::string::npos && sp.name.find(".linear_attn.out_proj.") == std::string::npos)
            throw std::invalid_argument("qwen38 model: w4_tensors: " + sp.name + " has no W4 consumer (GDN in_proj_qkv / out_proj only)");
          sp.store = model::DenseSpec::Store::w4r;
          sp.w4_mse = opt.w4_mse;
          ++ls_.w4_overrides;
          break;
        }
  // The routed experts' raw reads need no placement, so their reader starts now; its first read waits for the dense
  // reader's last (two readers at once slow the drive) and then runs while the dense weights pack and the expert
  // memory is allocated: up to kExpertReadAhead layers (~1.3 GB each), at most a quarter of the available host
  // memory. The dense reads run ahead of their packing (kDenseReadAhead groups) so the drive stays busy from the
  // first read to the last.
  // (the baked .tp2 blobs are TP2's halves: TP1 / TP4 pack from the checkpoint)
  const bool early_reads = !opt.calib && !(opt.moe_tp && fab.size() == 2 && std::filesystem::exists(dir + "/native/experts-L00.tp2"));
  std::promise<void> dense_read;  // set by the dense reader after its last read
  dense_ = std::make_unique<load::DenseStore>(fab);
  if (early_reads) {
    const util::HostMem hm = util::host_mem();
    const double layer_bytes = static_cast<double>(cfg_.experts) * static_cast<double>(ExpertLayout::of(cfg_).stride);
    const int ahead = hm.ok ? std::clamp(static_cast<int>(static_cast<double>(hm.available) / 4 / layer_bytes), 1, kExpertReadAhead) : 2;
    experts_ = std::make_unique<ExpertStore>(fab, cfg_, opt.moe_tp);
    experts_->start_reads(ck_, model::BulkOptions{}, ahead, dense_read.get_future().share());
    dense_->set_reads(kDenseReadAhead, [&dense_read] { dense_read.set_value(); });
  }
  dense_->load(ck_, plan, opt.threads);
  ls_.dense_s = dense_->stats().total_s;
  head_ = std::make_unique<Head>(fab, *dense_, cfg_);
  cands_ = std::make_unique<HeadCands>(fab, head_->vocab_per_card(), Head::kRows);  // before any VRAM fit
  hs_ = HcSet::from(*dense_, cfg_, nc_);
  ms_ = MoeSet::from(*dense_, cfg_, nc_, opt.moe_tp);
  ms_.shape.down_bf16 = opt.moe_down_bf16;
  ms_.shape.rot = cfg_.w_rotate || opt.moe_rotate;  // experts baked in a block-Hadamard basis
  ms_.shape.w4_bias = opt.prefill_moe_bias;  // read only by the prefill W4A16 block gate_up
  // decode W4A8 (the prefill path's numerics; tp only); by default as the checkpoint was tuned
  ms_.shape.a8 = (opt.decode_moe_a8 < 0 ? cfg_.act_bits == 8 : opt.decode_moe_a8 != 0) && opt.moe_tp;
  ms_.shape.rows_cap = step_cap_;  // the MoE's scratch by the largest step
  if (step_cap_ > k::kStepRows32 && (ms_.shape.rot || ms_.shape.a8 || !opt.moe_tp || opt.moe_cpu.mode != "off"))
    throw std::invalid_argument("qwen38 model: steps above 32 rows run the W4A16 experts tensor-parallel, without rotation and without the CPU path");
  gs_ = GdnSet::from(*dense_, cfg_, nc_);
  qs_ = QsaSet::from(*dense_, cfg_, nc_, opt.kv, opt.kv_split);
  qsa_ord_.assign(static_cast<size_t>(cfg_.layers), -1);
  for (size_t i = 0; i < qs_.layers.size(); ++i) qsa_ord_[static_cast<size_t>(qs_.layers[i])] = static_cast<int>(i);
  ps_ = PleSet::from(*dense_, cfg_, nc_);
  if (head_->lora_ld() < hs_.down_ld) throw std::runtime_error("qwen38 model: head lora buffer narrower than a layer mixer's down output");
  if (opt.ple_dir.empty()) ple_tab_ = std::make_unique<PleTable>(ck_, cfg_, opt.ple_threads);
  else ple_tab_ = std::make_unique<PleTable>(model::Checkpoint::open(opt.ple_dir, false), cfg_, opt.ple_threads);  // 
  ple_pinned_ = static_cast<uint16_t *>(fab.alloc_host(static_cast<size_t>(Head::kRows) * ps_.shape.emb * 2));
  ple_words_ = static_cast<uint32_t *>(fab.alloc_host(4096));
  std::memset(ple_words_, 0, 4096);
  std::memset(ple_pinned_, 0, static_cast<size_t>(Head::kRows) * ps_.shape.emb * 2);
  seg_pin_ = static_cast<int32_t *>(fab.alloc_host(std::max<size_t>(sizeof(seg_words_), 4096)));
  std::memset(seg_pin_, 0, sizeof(seg_words_));
  if (opt.prefill_chunk < 4 || opt.prefill_chunk % 4) throw std::invalid_argument("qwen38 model: prefill_chunk must be a positive multiple of 4");
  if (opt.prefill_dense_onednn >= 0) k::pf_dense_set_onednn(opt.prefill_dense_onednn);  // (process-wide)
  pf_ple_ = static_cast<uint16_t *>(fab.alloc_host(static_cast<size_t>(opt.prefill_chunk + 4) * ps_.shape.emb * 2));
  std::memset(pf_ple_, 0, static_cast<size_t>(opt.prefill_chunk + 4) * ps_.shape.emb * 2);

  // ---- states, caches, scratch, wires ----
  const int H = cfg_.hidden, L = cfg_.layers;
  // past the trained positions: static YaRN at the smallest factor (steps of 0.01) reaching the context; whole factors
  // give the tables of the integer rule (ceil(context / trained)) bit for bit
  const k::QsaYarn yarn = k::qsa_yarn_for(cap_, cfg_.max_positions);
  yarn_factor_ = yarn.factor;
  auto rope_h = k::qsa_rope_table(cap_, k::kQsaRot, cfg_.rope_theta, yarn);
  // YaRN tiers (ModelOptions::yarn_tiers): the tiers below the context's factor follow its table (the last tier starts
  // at row 0: a sequence there has the bits of one factor for all); a sequence's rows read row P + its tier's first row
  if (opt.yarn_tiers && yarn.factor > 1.0 && cfg_.max_positions > 0) {
    for (int64_t Lt = cfg_.max_positions; Lt < cap_; Lt *= 2) {
      const k::QsaYarn yt = k::qsa_yarn_for(Lt, cfg_.max_positions);
      const auto t = k::qsa_rope_table(Lt, k::kQsaRot, cfg_.rope_theta, yt);
      tier_base_.push_back(static_cast<int64_t>(rope_h.size() / k::kQsaRopeRow));
      tier_limit_.push_back(Lt);
      tier_factor_.push_back(yt.factor);
      rope_h.insert(rope_h.end(), t.begin(), t.end());
    }
    if (static_cast<int64_t>(rope_h.size() / k::kQsaRopeRow) > int64_t{2147483647}) throw std::invalid_argument("qwen38 model: YaRN tiers past int32 rows");
  }
  tier_base_.push_back(0);
  tier_limit_.push_back(cap_);
  tier_factor_.push_back(yarn.factor);
  for (int c = 0; c < nc_; ++c) {
    auto card = std::make_unique<Card>();
    Card &K = *card;
    auto &q = fab.compute(c);
    auto dev = [&](size_t bytes) {
      bytes = std::max<size_t>(bytes, 64);
      void *ptr = fab.alloc(c, bytes);
      q.memset(ptr, 0, bytes);
      K.mem.push_back(ptr);
      return ptr;
    };
    // Decode buffers hold a step's rows at the largest row shape (rows.hpp: 8;), zeroed so
    // rows past a step's live rows stay finite.
    const size_t R8 = static_cast<size_t>(Head::kRows);  // a step's rows (Head::kRows)
    const size_t act = R8 * H * 2;
    K.xa = static_cast<bf16 *>(dev(act));
    K.ya = static_cast<bf16 *>(dev(act));
    K.xm = static_cast<bf16 *>(dev(act));
    K.ym = static_cast<bf16 *>(dev(act));
    K.seg = static_cast<int32_t *>(dev(sizeof(seg_words_)));
    K.rope = static_cast<bf16 *>(dev(rope_h.size() * 2));
    q.memcpy(K.rope, rope_h.data(), rope_h.size() * 2);
    // pair wire (n = hidden)
    if (nc_ > 1) {  // TP1 has no wire
      K.pw.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_wire_words(H, nc_) * 4));
      K.mem.push_back(K.pw.inbox);
    }
    K.pw.error = static_cast<uint32_t *>(dev(64));
    K.lp = static_cast<float *>(dev(static_cast<size_t>(k::kLpMaxRows) * k::kLpRec * 4));
    // MoE
    const auto msh = ms_.shape_of(c);
    K.msc.logits = static_cast<float *>(dev(R8 * ms_.w[c][0].router.n * 4));
    // One routing record per layer: the step's routing stays readable after it (no extra launch).
    K.msc.route = static_cast<k::MoeRoute *>(dev(sizeof(k::MoeRoute) * static_cast<size_t>(L)));
    if (c == 0 && (!opt.rank_file.empty() || opt.exchange)) {  // the decode reads per (layer, expert), card 0 (both route alike)
      hits_ = static_cast<uint32_t *>(dev(static_cast<size_t>(L) * cfg_.experts * 4));
      hits_seen_.assign(static_cast<size_t>(L) * cfg_.experts, 0u);
    }
    K.msc.h = static_cast<bf16 *>(dev(k::moe_h_elems(msh) * 2));
    K.msc.partial = static_cast<float *>(dev(k::moe_partial_elems(msh) * 4));
    K.msc.xr = static_cast<bf16 *>(dev(R8 * cfg_.hidden * 2));  // (rot; kMaxRows rows)
    if (ms_.shape.a8) {  // the decode W4A8 activations
      K.msc.xq = static_cast<int8_t *>(dev(R8 * cfg_.hidden));
      K.msc.xs = static_cast<float *>(dev(R8 * (cfg_.hidden / 128) * 4));
      K.msc.hq = static_cast<int8_t *>(dev(k::moe_h_elems(msh)));
      K.msc.hs = static_cast<float *>(dev(k::moe_hs_elems(msh) * 4));
    }
    if (nc_ > 1) {
      K.mw.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::moe_wire_words(msh) * 4));
      K.mem.push_back(K.mw.inbox);
    }
    K.mw.error = static_cast<uint32_t *>(dev(64));
    // GDN: per linear layer, one copy per slot (state, both windows, both pending logs)
    const auto gsh = gs_.shape_of(c);
    K.gsc.core = static_cast<bf16 *>(dev(R8 * gsh.val_w() * 2));
    K.gsc.partial = static_cast<float *>(dev(R8 * H * 4));
    K.gsc.y = static_cast<float *>(dev(R8 * gs_.w[c][gs_.layers.front()].qkvz.n * 4));
    K.gst.resize(static_cast<size_t>(L));
    for (int l : gs_.layers) {
      k::GdnState &st = K.gst[static_cast<size_t>(l)];
      st.slot_S = k::gdn_S_elems(gsh);
      st.slot_win = k::gdn_win_elems(gsh);
      st.log_rows = seg_rows_max_;  // 8 with the DFlash drafter
      st.slot_y = static_cast<size_t>(seg_rows_max_) * gs_.w[c][l].qkvz.n;  // a sequence's rows of a step (<= 4; 8 with DFlash)
      st.slot_ab = static_cast<size_t>(seg_rows_max_) * gsh.ab_n();
      st.S = static_cast<float *>(dev(NS * st.slot_S * 4));
      for (int pr = 0; pr < 2; ++pr) {
        st.win[pr] = static_cast<bf16 *>(dev(NS * st.slot_win * 2));
        st.y[pr] = static_cast<float *>(dev(NS * st.slot_y * 4));
        st.ab[pr] = static_cast<float *>(dev(NS * st.slot_ab * 4));
      }
      st.seg = K.seg;
    }
    if (!opt.dflash_dir.empty() && opt.dflash_tree > 0 && !opt.calib)  // the tree verifies' GDN scratch
      K.tree_S = static_cast<float *>(dev(k::gdn_tree_S_elems(gsh) * 4));
    // QSA: per full layer, one pool of pages: every slot's pages by the shared table (the identity
    // table of when the pool is fixed: slot s owns pages [s P, s P + P)); the pages themselves come in kv_pool_alloc
    const auto qsh = qs_.shape_of(c);
    const size_t P = static_cast<size_t>(cap_) / k::kQsaPage;
    kv_table_[c] = static_cast<int32_t *>(dev(P * NS * 4));
    K.qc.resize(static_cast<size_t>(L));
    for (int l : qs_.layers) {
      k::QsaCache &cc = K.qc[static_cast<size_t>(l)];
      cc.ring = static_cast<bf16 *>(dev(NS * k::kQsaRing * k::kQsaIdxDim * 2));
      if (!opt.dflash_dir.empty() && opt.dflash_tree > 0 && !opt.calib)  // a tree step's per-row compressed keys
        cc.tkc = static_cast<bf16 *>(dev(static_cast<size_t>(k::kMaxRows) * k::kQsaIdxDim * 2));
      cc.pages = kv_table_[c];
      cc.max_tokens = static_cast<int>(cap_);
      cc.slot_pages = static_cast<int>(P);
    }
    const auto &qw = qs_.w[c][qs_.layers.front()];
    auto &sc = K.qsc;
    sc.y = static_cast<float *>(dev(R8 * qw.qkv.n * 4));
    sc.yi = static_cast<float *>(dev(R8 * qw.idx.n * 4));
    sc.q = static_cast<bf16 *>(dev(R8 * qsh.attn_hq() * k::kQsaDim * 2));
    sc.iq = static_cast<bf16 *>(dev(R8 * k::kQsaIdxHeads * k::kQsaIdxDim * 2));
    sc.nblk = static_cast<int32_t *>(dev(R8 * 4));
    sc.score = static_cast<uint32_t *>(dev(k::qsa_score_words(static_cast<int>(cap_), Head::kRows) * 4));
    sc.cand_key = static_cast<uint32_t *>(dev(R8 * 16 * 512 * 4));
    sc.cand_id = static_cast<int32_t *>(dev(R8 * 16 * 512 * 4));
    sc.cand_n = static_cast<int32_t *>(dev(R8 * 16 * 4));
    sc.sel = static_cast<int32_t *>(dev(R8 * 512 * 4));
    sc.sel_n = static_cast<int32_t *>(dev(R8 * 4));
    sc.part = static_cast<float *>(dev(k::qsa_part_floats(qsh, Head::kRows) * 4));
    sc.attn = static_cast<bf16 *>(dev(R8 * qsh.o_k() * 2));
    sc.partial = static_cast<float *>(dev(R8 * H * 4));
    if (qsh.kv_ways() > 1) {  // the KV split by tokens: the pair's merge states and its mailbox (a region per QSA layer + the MTP layer's)
      const int regions = static_cast<int>(qs_.layers.size()) + 1;
      kv_pair_region_words_ = k::qsa_pair_words(static_cast<int>(R8), qsh.hq());
      sc.pend = static_cast<float *>(dev(k::qsa_pair_floats(static_cast<int>(R8), qsh.hq()) * 4));
      sc.pair.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, kv_pair_region_words_ * regions * 4));
      K.mem.push_back(sc.pair.inbox);
      for (int g = 0; g < regions; ++g) k::qsa_pair_init(q, sc.pair.inbox + kv_pair_region_words_ * g, static_cast<int>(R8), qsh.hq());
      sc.pair.error = K.pw.error;
      sc.pair.rows_max = static_cast<int>(R8);
    }
    // PLE
    const auto &psh = ps_.shape;
    K.ple_st.slot_elems = k::ple_ring_elems(psh);
    K.ple_st.ring = static_cast<bf16 *>(dev(NS * K.ple_st.slot_elems * 2));
    K.ple_sc.emb = static_cast<bf16 *>(dev(R8 * psh.emb * 2));
    K.ple_sc.kv = static_cast<float *>(dev(R8 * psh.kv_n() * 4));
    K.ple_sc.gated = static_cast<bf16 *>(dev(R8 * psh.width() * 2));
    // Prefill chunk buffers
    {
      const size_t rows = static_cast<size_t>(opt.prefill_chunk) + 4;
      const size_t W = static_cast<size_t>(hs_.shape.width());
      for (int i = 0; i < 2; ++i) {
        K.pf.res[i] = static_cast<bf16 *>(dev(rows * W * 2));
        K.pf.lora[i] = static_cast<float *>(dev(rows * static_cast<size_t>(head_->lora_ld()) * 4));
      }
      K.pf.xa = static_cast<bf16 *>(dev(rows * H * 2));
      K.pf.ya = static_cast<bf16 *>(dev(rows * H * 2));
      K.pf.xm = static_cast<bf16 *>(dev(rows * H * 2));
      K.pf.ym = static_cast<bf16 *>(dev(rows * H * 2));
      K.pf.ids = static_cast<int32_t *>(dev(rows * 4));
      calib_tmp_[c] = static_cast<float *>(dev(rows * static_cast<size_t>(hs_.shape.streams) * 4));  // 
      K.pf.words = static_cast<int32_t *>(dev((rows / 4 + 1) * k::kSegRec * 4));
      K.pf.lane_words = static_cast<int32_t *>(dev((rows / 4 + 1 + kPfLanesMax) * k::kSegRec * 4));  // (a lane's last slice may be short)
      K.pf.hc.xn = static_cast<bf16 *>(dev(rows * W * 2));
      K.pf.hc.s = static_cast<bf16 *>(dev(rows * static_cast<size_t>(hs_.shape.lowrank) * 2));
      K.pf.hc.gate = static_cast<float *>(dev(rows * W * 4));
      // The HC down's split partials; with the MTP drafter also mtp_prefill_rows' front GEMM output (rows x hidden fp32,
      // its fe_y), so at least hidden per row then.
      const size_t part_ld = std::max<size_t>(static_cast<size_t>(k::kHcDownKs) * static_cast<size_t>(head_->lora_ld()),
                                              opt.mtp_dir.empty() ? 0 : static_cast<size_t>(H));
      K.pf.hc.part = static_cast<float *>(dev(rows * part_ld * 4));
      if (opt.prefill_dense_a8) {  // one W8A8 scratch per card, sized for the widest input (the HC down's S x hidden)
        K.pf.a8.q = static_cast<int8_t *>(dev(rows * W));
        K.pf.a8.s = static_cast<float *>(dev(k::pf_a8_scales(static_cast<int>(rows), W) * 4));
        K.pf.a8.row = opt.prefill_dense_a8_row;
        if (!opt.prefill_dense_a8_row) K.pf.hc.a8 = K.pf.a8;  // per row leaves the HC mixer on W8A16
        K.pf.gdn.a8 = K.pf.a8;
        K.pf.qsa.a8 = K.pf.a8;
        K.pf.ple.a8 = K.pf.a8;
        K.pf.moe.da8 = K.pf.a8;
      }
      if (ms_.shape.tp) {
        const auto msh = ms_.shape_of(c);
        const int R = static_cast<int>(rows);
        const auto z = k::moe_rows_sizes(msh, R, ms_.w[c][0].router.n);
        auto &mr = K.pf.moe;
        mr.logits = static_cast<float *>(dev(z.logits * 4));
        mr.ids = static_cast<int32_t *>(dev(z.ids * 4));
        mr.topw = static_cast<float *>(dev(z.ids * 4));
        mr.sg = static_cast<float *>(dev(z.sg * 4));
        mr.slot = static_cast<int32_t *>(dev(z.ids * 4));
        mr.row = static_cast<int32_t *>(dev(z.row * 4));
        mr.seg = static_cast<int32_t *>(dev(z.seg * 4));
        mr.seg_n = static_cast<int32_t *>(dev(64));
        // the W4A16 routed experts by 32-row block (moe_rows_gu.cpp)
        mr.blk = static_cast<int32_t *>(dev(z.blk * 4));
        mr.blk_n = static_cast<int32_t *>(dev(64));
        mr.h = static_cast<bf16 *>(dev(z.h * 2));
        mr.y = static_cast<k::moe_bf16 *>(dev(z.y * 2));
        mr.sgu = static_cast<float *>(dev(z.sgu * 4));
        mr.sh = static_cast<bf16 *>(dev(z.sh * 2));
        mr.sy = static_cast<float *>(dev(z.sy * 4));
        mr.xr = static_cast<bf16 *>(dev(z.xr * 2));  // the target's or the drafter's rotated experts
        if (opt.prefill_moe_a8) {
          mr.xq = static_cast<int8_t *>(dev(z.xq));
          mr.xs = static_cast<float *>(dev(z.xs * 4));
          mr.hq = static_cast<int8_t *>(dev(z.hq));
          mr.hs = static_cast<float *>(dev(z.hs * 4));
        }
        mr.part = static_cast<float *>(dev(static_cast<size_t>(R) * H * 4));  // split mode
        // Staging for the multi-segment cold experts of a call, out of the hot budget.
        if (opt.prefill_cold_stage > 0) {
          size_t st = ExpertLayout::of(cfg_).stride;
          k::moe_tp_layout(cfg_.hidden, cfg_.moe_inter, nc_, &st);
          mr.stage_cap = opt.prefill_cold_stage;
          mr.stage_stride = st;
          mr.stage = static_cast<uint8_t *>(dev(st * static_cast<size_t>(mr.stage_cap)));
          mr.tab2 = static_cast<const uint8_t **>(dev(static_cast<size_t>(cfg_.experts) * 8));
          mr.stage_list = static_cast<int32_t *>(dev(static_cast<size_t>(mr.stage_cap + 1) * 4));
        }
        // the many-row head's logits group in y (bf16 [lp_rows][vocab per card]); as many
        // rows as y holds (105 at 512-row chunks: 5 lm_head passes per chunk instead of 8)
        K.pf.lp_rows = static_cast<int>(std::min<size_t>(kPfLpRows, z.y / static_cast<size_t>(head_->vocab_per_card())));
        if (K.pf.lp_rows > 0) {
          K.pf.lp_tgt = static_cast<int32_t *>(dev(static_cast<size_t>(K.pf.lp_rows) * 4));
          K.pf.lp_xt = static_cast<float *>(dev(static_cast<size_t>(K.pf.lp_rows) * 4));
          K.pf.lp_rec = static_cast<float *>(dev(static_cast<size_t>(K.pf.lp_rows) * k::kLpRec * 4));
        }
        K.pf.mw.rows_max = R;
        if (nc_ > 1) {
          K.pf.mw.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::moe_rows_wire_words(msh, R) * 4));
          K.mem.push_back(K.pf.mw.inbox);
        }
        K.pf.mw.error = K.mw.error;
      }
      {
        const auto gsh = gs_.shape_of(c);
        const int R = static_cast<int>(rows);
        const auto z = k::gdn_rows_sizes(gsh, gs_.w[c][gs_.layers.front()].qkvz.n, R);
        auto &g = K.pf.gdn;
        g.yq = static_cast<float *>(dev(z.yq * 4));
        g.ab = static_cast<float *>(dev(z.ab * 4));
        g.cv = static_cast<float *>(dev(z.cv * 4));
        g.qn = static_cast<float *>(dev(z.qk * 4));
        g.kn = static_cast<float *>(dev(z.qk * 4));
        g.gate = static_cast<float *>(dev(z.gate * 4));
        g.lgate = static_cast<float *>(dev(z.lgate * 4));
        {  // the chunked scan's per (chunk, value head) blocks
          const auto zs = k::gdn_scan_sizes(gsh, R);
          g.scan.wk = static_cast<bf16 *>(dev(zs.wk * 2));
          g.scan.qg = static_cast<bf16 *>(dev(zs.wk * 2));
          g.scan.kt = static_cast<bf16 *>(dev(zs.wk * 2));
          g.scan.kt_lo = static_cast<bf16 *>(dev(zs.wk * 2));
          g.scan.mt = static_cast<bf16 *>(dev(zs.mt * 2));
          g.scan.uv = static_cast<float *>(dev(zs.uv * 4));
          g.scan.gc = static_cast<float *>(dev(zs.gc * 4));
        }
        g.o = static_cast<bf16 *>(dev(z.o * 2));
        g.y = static_cast<bf16 *>(dev(z.o * 2));
        g.partial = static_cast<float *>(dev(z.partial * 4));
        K.pf.pw.rows_max = R;
        K.pf.pw.n_max = H;
        if (nc_ > 1) {
          K.pf.pw.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::pair_rows_wire_words(R, H, nc_) * 4));
          K.mem.push_back(K.pf.pw.inbox);
        }
        K.pf.pw.error = K.pw.error;
        K.pf.pw.bf16 = opt.prefill_wire_bf16;
        if (nc_ > 1) {  // TP1 has no split sum; TP4: a quarter per card
          // the split sum: the other cards' parts of a chunk's partial, and the flag lines
          K.pf.split.rows_max = R;
          K.pf.split.n_max = H;
          K.pf.split.cards = nc_;
          K.pf.split.card = c;
          K.pf.split.recv = static_cast<float *>(dev(k::pair_split_recv_elems(R, H, nc_) * 4));
          K.pf.split.flags = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::kPairSplitFlagBytes));
          K.mem.push_back(K.pf.split.flags);
          q.memset(K.pf.split.flags, 0, k::kPairSplitFlagBytes);
          K.pf.flag_lines[0] = K.pf.split.flags;
          if (opt.prefill_graphs) {  // the prompt-call graphs' flag sets, one aligned slice each
            constexpr size_t kSetStride = 4096;
            static_assert(k::kPairSplitFlagBytes <= kSetStride);
            auto *sets = static_cast<std::byte *>(fab.alloc_mailbox(c, kSetStride * kPfGraphsMax));
            K.mem.push_back(sets);
            q.memset(sets, 0, kSetStride * kPfGraphsMax);
            for (int i = 0; i < kPfGraphsMax; ++i) K.pf.flag_lines[1 + i] = reinterpret_cast<uint32_t *>(sets + kSetStride * static_cast<size_t>(i));
          }
          K.pf.split.error = K.pw.error;
          if (opt.prefill_wire_bf16) {  // the bf16 wire's stage (the peer's rows converted before the copy)
            K.pf.split.bf16 = true;
            K.pf.split.stage = static_cast<bf16 *>(dev(k::pair_split_stage_elems(R, H, nc_) * 2));
          }
          {
            auto *tab = static_cast<uint32_t *>(dev(k::kPairSplitSeqTabBytes));
            k::pair_split_seq_tab_init(q, tab);
            K.pf.split.seq_tab = tab;
          }
        }
        K.pf.side.emplace(fab.card_context(c), fab.device(c), Fabric::queue_props(fab.profiling()));
      }
      {
        const auto &qw0 = qs_.w[c][qs_.layers.front()];
        auto &r = K.pf.qsa;
        r.y = static_cast<float *>(dev((rows + 4) * qw0.qkv.n * 4));
        r.yi = static_cast<float *>(dev((rows + 4) * qw0.idx.n * 4));
        r.attn = static_cast<bf16 *>(dev((rows + 4) * static_cast<size_t>(qs_.shape_of(c).o_k()) * 2));
        r.partial = static_cast<float *>(dev(rows * H * 4));
        const size_t rr = rows + 4, B = k::kQsaRowsBatch;
        r.q = static_cast<bf16 *>(dev(rr * qs_.shape_of(c).attn_hq() * k::kQsaDim * 2));
        r.iq = static_cast<bf16 *>(dev(rr * k::kQsaIdxHeads * k::kQsaIdxDim * 2));
        r.nblk = static_cast<int32_t *>(dev(rr * 4));
        r.sel = static_cast<int32_t *>(dev(rr * k::kQsaTopBlocks * 4));
        r.sel_n = static_cast<int32_t *>(dev(rr * 4));
        r.score = static_cast<uint32_t *>(dev(k::qsa_score_words(static_cast<int>(cap_)) / k::kQsaRows * B * 4));
        r.cand_key = static_cast<uint32_t *>(dev(B * 16 * k::kQsaTopBlocks * 4));
        r.cand_id = static_cast<int32_t *>(dev(B * 16 * k::kQsaTopBlocks * 4));
        r.cand_n = static_cast<int32_t *>(dev(B * 16 * 4));
        r.part = static_cast<float *>(dev(k::qsa_part_floats(qs_.shape_of(c)) / k::kQsaRows * B * 4));
        if (qsh.kv_ways() > 1) {  // the split's merge states and mailbox for a call's rows (lanes by row offset)
          const int RR = static_cast<int>(rr);
          r.pend = static_cast<float *>(dev(k::qsa_pair_floats(RR, qsh.hq()) * 4));
          r.pair.inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::qsa_pair_words(RR, qsh.hq()) * 4));
          K.mem.push_back(r.pair.inbox);
          k::qsa_pair_init(q, r.pair.inbox, RR, qsh.hq());
          r.pair.error = K.pw.error;
          r.pair.rows_max = RR;
        }
      }
      {
        const auto &psh = ps_.shape;
        auto &pr = K.pf.ple;
        pr.emb = static_cast<bf16 *>(dev(rows * psh.emb * 2));
        pr.kv = static_cast<float *>(dev(rows * static_cast<size_t>(ps_.w[c].kv.n) * 4));
        pr.gated = static_cast<bf16 *>(dev(rows * psh.width() * 2));
        pr.x = static_cast<bf16 *>(dev(rows * psh.width() * 2));
      }
    }
    q.wait();
    cards_.push_back(std::move(card));
  }
  for (int c = 0; c < nc_ && nc_ == 2; ++c) {  // the two-card wires (TP2)
    Card &K = *cards_[static_cast<size_t>(c)];
    K.pw.peer_inbox = cards_[static_cast<size_t>(1 - c)]->pw.inbox;
    K.mw.peer_inbox = cards_[static_cast<size_t>(1 - c)]->mw.inbox;
    k::pair_wire_init(fab.compute(c), H, K.pw.inbox).wait();
    k::moe_wire_init(fab.compute(c), ms_.shape_of(c), K.mw.inbox).wait();
    K.pf.pw.peer_inbox = cards_[static_cast<size_t>(1 - c)]->pf.pw.inbox;
    K.pf.split.peer_recv = cards_[static_cast<size_t>(1 - c)]->pf.split.recv;
    K.pf.split.peer_flags = cards_[static_cast<size_t>(1 - c)]->pf.split.flags;
    k::pair_rows_wire_init(fab.compute(c), K.pf.pw.rows_max, K.pf.pw.n_max, K.pf.pw.inbox).wait();
    if (K.pf.mw.inbox) {
      K.pf.mw.peer_inbox = cards_[static_cast<size_t>(1 - c)]->pf.mw.inbox;
      k::moe_rows_wire_init(fab.compute(c), ms_.shape_of(c), K.pf.mw.rows_max, K.pf.mw.inbox).wait();
    }
  }
  if (nc_ == kMaxCards) {  // the split sums' four cards: every card's recv and flags
    for (int c = 0; c < nc_; ++c)
      for (int d = 0; d < nc_; ++d) {
        cards_[static_cast<size_t>(c)]->pf.split.recv_of[d] = cards_[static_cast<size_t>(d)]->pf.split.recv;
        cards_[static_cast<size_t>(c)]->pf.split.flags_of[d] = cards_[static_cast<size_t>(d)]->pf.split.flags;
      }
  }
  // The split sums' buffers per prefill lane. Lane 0 is pf.split; each later lane has its own recv and
  // flag lines (and bf16 stage), because its copies are in flight while the lane before it computes.
  for (int c = 0; c < nc_ && nc_ > 1; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(c);
    auto zeroed = [&](size_t bytes) {
      void *ptr = fab.alloc(static_cast<size_t>(c), bytes);
      q.memset(ptr, 0, bytes);
      K.mem.push_back(ptr);
      return ptr;
    };
    K.pf.lane[0] = K.pf.split;
    for (int j = 1; j < kPfLanesMax; ++j) {
      k::PairSplit &s = K.pf.lane[j];
      s = K.pf.split;
      s.recv = static_cast<float *>(zeroed(k::pair_split_recv_elems(s.rows_max, s.n_max, nc_) * sizeof(float)));
      s.flags = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::kPairSplitFlagBytes));
      K.mem.push_back(s.flags);
      q.memset(s.flags, 0, k::kPairSplitFlagBytes);
      if (s.bf16) s.stage = static_cast<bf16 *>(zeroed(k::pair_split_stage_elems(s.rows_max, s.n_max, nc_) * sizeof(bf16)));
    }
    for (int j = 0; j < kPfLanesMax && nc_ == kMaxCards; ++j) {  // the selection's flag lines (copies only: no recv)
      k::PairSplit &s = K.pf.sel[j];
      s = K.pf.lane[j];
      s.flags = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::kPairSplitFlagBytes));
      K.mem.push_back(s.flags);
      q.memset(s.flags, 0, k::kPairSplitFlagBytes);
    }
    q.wait();
  }
  for (int c = 0; c < nc_ && nc_ > 1; ++c)
    for (int j = 1; j < kPfLanesMax; ++j) {
      k::PairSplit &s = cards_[static_cast<size_t>(c)]->pf.lane[j];
      if (nc_ == 2) {
        s.peer_recv = cards_[static_cast<size_t>(1 - c)]->pf.lane[j].recv;
        s.peer_flags = cards_[static_cast<size_t>(1 - c)]->pf.lane[j].flags;
      }
      for (int d = 0; d < nc_ && nc_ == kMaxCards; ++d) {
        s.recv_of[d] = cards_[static_cast<size_t>(d)]->pf.lane[j].recv;
        s.flags_of[d] = cards_[static_cast<size_t>(d)]->pf.lane[j].flags;
      }
    }
  for (int c = 0; c < nc_ && nc_ == kMaxCards; ++c)
    for (int j = 0; j < kPfLanesMax; ++j)
      for (int d = 0; d < nc_; ++d) cards_[static_cast<size_t>(c)]->pf.sel[j].flags_of[d] = cards_[static_cast<size_t>(d)]->pf.sel[j].flags;
  if (nc_ == kMaxCards) {  // the four-card wires (TP4): every card's inbox in each, by the host
    k::WireQuad dq, mq, rq, mrq;  // decode pair / decode MoE / rows pair / rows MoE inboxes
    for (int c = 0; c < nc_; ++c) {
      const Card &K = *cards_[static_cast<size_t>(c)];
      dq.inbox[c] = K.pw.inbox, mq.inbox[c] = K.mw.inbox, rq.inbox[c] = K.pf.pw.inbox, mrq.inbox[c] = K.pf.mw.inbox;
    }
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      K.pw.quad = dq, K.mw.quad = mq, K.pf.pw.quad = rq;
      K.pw.quad.card = K.mw.quad.card = K.pf.pw.quad.card = c;
      if (K.qsc.pair.inbox) {  // the KV split by tokens: the pair's mailboxes (card ^ 1)
        K.qsc.pair.peer_inbox = cards_[static_cast<size_t>(c ^ 1)]->qsc.pair.inbox;
        K.pf.qsa.pair.peer_inbox = cards_[static_cast<size_t>(c ^ 1)]->pf.qsa.pair.inbox;
      }
      k::pair_wire_init(fab.compute(c), H, K.pw.inbox, nc_).wait();
      k::moe_wire_init(fab.compute(c), ms_.shape_of(c), K.mw.inbox).wait();
      k::pair_rows_wire_init(fab.compute(c), K.pf.pw.rows_max, K.pf.pw.n_max, K.pf.pw.inbox, nc_).wait();
      if (K.pf.mw.inbox) {
        K.pf.mw.quad = mrq;
        K.pf.mw.quad.card = c;
        k::moe_rows_wire_init(fab.compute(c), ms_.shape_of(c), K.pf.mw.rows_max, K.pf.mw.inbox).wait();
      }
    }
    dec_owner_init();  // after the four-card wires: the owner sums' wires start as copies of them
  }
  pf_gdn_owner_init();
  // ---- the MTP drafter: before the placement (its weights, experts and KV come out of the VRAM
  // the target's hot experts would get) ----
  // both drafters only with the step count that hands the steps to the MTP layer
  if (!opt.mtp_dir.empty() && !opt.dflash_dir.empty() && !opt.calib && opt.mtp_from < 1)
    throw std::invalid_argument("qwen38 model: the MTP layer beside the DFlash drafter needs mtp_from >= 1 (the steps of that many sequences it drafts)");
  if (!opt.mtp_dir.empty()) load_mtp(opt.mtp_dir);
  // ---- the DFlash 2 drafter (arch/common/dflash.hpp), before the placement as the MTP drafter ----
  if (!opt.dflash_dir.empty() && !opt.calib) {
    if (!opt.moe_tp) throw std::invalid_argument("qwen38 model: the DFlash drafter needs expert TP (its prompt context comes from the many-row prefill)");
    if (opt.dflash_feat_shift < 0 || opt.dflash_feat_shift > 1) throw std::invalid_argument("qwen38 model: dflash_feat_shift 0 or 1");
    arch::DFlashOptions dop;
    dop.dense_bytes = opt.dflash_dense_bytes;
    dop.threads = opt.threads;
    dop.vocab = opt.mtp_vocab;
    dop.slots = opt.slots;
    dop.rows = step_cap_;  // the step's rows (its features are the drafter's context); a pass holds kPassRows of them
    dop.prefill_rows = opt.prefill_chunk;
    dop.profile = opt.dflash_profile;
    dop.ctx_defer = opt.dflash_ctx_defer;
    dop.conf_cut = opt.dflash_conf_cut;
    // draft trees (DDTree): trees of the 8-row block, verified on the tree rows of QSA / PLE / GDN; not beside the MTP
    // layer (its pass 1 writes KV over the verify's rows before the accept)
    if (opt.dflash_tree < 0 || opt.dflash_tree > 7) throw std::invalid_argument("qwen38 model: dflash_tree 0..7 (nodes of the drafter's 8-row block)");
    if (opt.dflash_tree > 0 && !opt.mtp_dir.empty()) throw std::invalid_argument("qwen38 model: DFlash draft trees beside the MTP layer (drafters: one with trees)");
    dop.tree = opt.dflash_tree, dop.tree_cut = opt.dflash_tree_cut;
    dop.tree_seqs = std::max(1, opt.dflash_tree_seqs), dop.tree_temp = opt.dflash_tree_temp;
    arch::DFlashTarget tg;  // the target's sizes, its pinned embedding table, its int8 lm_head rows per card
    tg.hidden = cfg_.hidden, tg.layers = L, tg.vocab = cfg_.vocab, tg.eos = cfg_.eos;
    for (int c = 0; c < nc_; ++c) {
      arch::LmMat lm;
      lm.d = head_->lm_weights(c);
      lm.n = lm.d.n, lm.k = lm.d.k;
      tg.embed.push_back(head_->embed_table(c));
      tg.lm.push_back(lm);
    }
    dflash_ = std::make_unique<arch::DFlash>(fab, opt.dflash_dir, dir, tg, cap_, dop);
    // a feature of target layer t is captured after the attention mixer of layer t + 1 - shift (the mixer combines the
    // pending MoE block of the layer before it into the multi-stream residual: the completed output of that layer)
    df_feat_at_.assign(static_cast<size_t>(L), -1);
    for (int l = 0; l < L; ++l) {
      const int j = dflash_->feature_of(l);
      if (j < 0) continue;
      const int at = l + 1 - opt.dflash_feat_shift;
      if (at < 1 || at >= L) throw std::runtime_error("qwen38 model: a DFlash feature layer has no attention mixer after it");
      df_feat_at_[static_cast<size_t>(at)] = j;
    }
    const auto &dc = dflash_->config();
    const size_t R = static_cast<size_t>(opt.prefill_chunk), kvn = 2 * static_cast<size_t>(dc.kv_heads / nc_) * dc.head_dim;
    for (int c = 0; c < nc_; ++c) {  // the prefill context pass's fp32 rows (its bf16 rows and wire are the chunk's)
      Card &K = *cards_[static_cast<size_t>(c)];
      df_part_[c] = static_cast<float *>(fab.alloc(c, std::max<size_t>(R * cfg_.hidden * 4, 4096)));
      df_ykv_[c] = static_cast<float *>(fab.alloc(c, std::max<size_t>(R * kvn * 4, 4096)));
      K.mem.push_back(df_part_[c]);
      K.mem.push_back(df_ykv_[c]);
    }
  }
  // ----: the prefix cache's device buffers (a staging buffer of one snapshot, the fork's page pairs) before
  // the placement; the pinned host buffers come on first use ----
  if (opt.snap_gib > 0 && !opt.calib) {
    double one = 0;
    for (int c = 0; c < nc_; ++c) one += static_cast<double>(snap_bytes(c));
    snap_cap_ = util::snap_capacity(opt.snap_gib, one);  // cut to the host RAM left (pinned on first use)
    for (int c = 0; c < nc_; ++c) {
      snap_stage_[c] = fab.alloc(c, std::max<size_t>(snap_bytes(c), 4096));
      cards_[static_cast<size_t>(c)]->mem.push_back(snap_stage_[c]);
      fork_pages_[c] = static_cast<int32_t *>(fab.alloc(c, std::max<size_t>(2 * static_cast<size_t>(cap_ / k::kQsaPage) * 4, 4096)));
      cards_[static_cast<size_t>(c)]->mem.push_back(fork_pages_[c]);
    }
  }

  // ---- the KV pool's pages: after everything but the routed experts, so an auto pool (kv_pool_fill) takes the VRAM
  // every expert leaves ----
  // Blobs per card: EP = the card's experts; TP = its 1/2 of every expert (half the bytes each).
  const int64_t per_total = static_cast<int64_t>(L) * cfg_.experts / (opt.moe_tp ? 1 : 2);
  size_t stride = ExpertLayout::of(cfg_).stride;
  if (opt.moe_tp) k::moe_tp_layout(cfg_.hidden, cfg_.moe_inter, nc_, &stride);
  {
    const std::vector<double> vram_free = opt.calib ? std::vector<double>{} : vram_free_per_card(loader, fab, nc_);
    if (!vram_free.empty()) ls_.onednn_gib = onednn_cap_gib(*dense_, nc_);
    kv_pool_alloc(kv_pool_pages_for(vram_free, static_cast<double>(per_total) * static_cast<double>(stride)));
  }

  // ---- calibration mode: one layer's TP blobs per card, filled by the caller ----
  if (opt.calib) {
    if (!opt.moe_tp) throw std::invalid_argument("qwen38 model: calib needs moe_tp");
    // a drafter may load (calib_mtp); its own experts stay where load_mtp put them
    size_t st = ExpertLayout::of(cfg_).stride;
    k::moe_tp_layout(cfg_.hidden, cfg_.moe_inter, nc_, &st);
    calib_stride_ = st;
    const int per = cfg_.experts;
    for (int c = 0; c < nc_; ++c) {
      Card &K = *cards_[static_cast<size_t>(c)];
      auto &q = fab.compute(c);
      calib_region_[c] = static_cast<uint8_t *>(fab.alloc(c, st * static_cast<size_t>(per)));
      K.mem.push_back(calib_region_[c]);
      q.memset(calib_region_[c], 0, st * static_cast<size_t>(per));
      std::vector<const uint8_t *> t(static_cast<size_t>(L) * per);
      for (int l = 0; l < L; ++l)
        for (int i = 0; i < per; ++i) t[static_cast<size_t>(l) * per + i] = calib_region_[c] + static_cast<size_t>(i) * st;
      K.tab = static_cast<const uint8_t **>(fab.alloc(c, t.size() * 8));
      K.mem.push_back(K.tab);
      q.memcpy(K.tab, t.data(), t.size() * 8).wait();
      if (K.pf.moe.stage_cap > 0) {  // every blob is in VRAM: no cold flags
        K.cold = static_cast<uint8_t *>(fab.alloc(c, std::max<size_t>(t.size(), 4096)));
        K.mem.push_back(K.cold);
        q.memset(K.cold, 0, std::max<size_t>(t.size(), 4096)).wait();
      }
      ls_.hot[c] = per;
    }
    ls_.total_s = since_s(t_all);
    use_slot(0);
    begin_sequence();
    return;
  }

  // ---- routed experts: whatever VRAM is left ----
  ExpertStoreOptions so;
  so.pack_threads = opt.threads;
  // the packing is memory-bound beside the drive's DMA: more threads pack no faster and slow the reads
  so.pack_threads = std::min(opt.threads, kExpertPackThreads);
  so.hash = opt.verify_experts;
  // A baked model's prepacked TP blobs (<model>/native/experts-LNN.tp2), when they match
  // this load (TP over 2 cards); otherwise the GPTQ tensors are read and packed.
  if (opt.moe_tp && nc_ == 2 && std::filesystem::exists(dir + "/native/experts-L00.tp2")) so.native_dir = dir + "/native";  // TP2's halves only
  if (!opt.hot_rank.empty()) so.rank = &opt.hot_rank;
  if (!opt.rank_file.empty()) {  // the ranking recorded by earlier runs places the hot set
    rec_ = std::make_unique<model::RouteRecord>(L, cfg_.experts, opt.rank_file, opt.rank_every_s);
    if (!rec_->ok()) {
      std::fprintf(stderr, "qwen38: hot-expert ranking %s: %s; not recorded this run\n", opt.rank_file.c_str(), rec_->status().c_str());
      rec_.reset();
      hits_ = nullptr;  // (stays allocated, never read: the graphs are recorded after this, without it)
    } else if (rec_->loaded() && !so.rank) {
      so.rank = &rec_->scores();
    }
  }
  {
    // The VRAM left after everything else, minus the reserve: the auto count, and the cap of a pinned one. A pinned count
    // past it does not fail an allocation: the driver evicts and pages, and every step runs far slower.
    const std::vector<double> vram_free = vram_free_per_card(loader, fab, nc_);
    const bool known = !vram_free.empty();
    const double free_now = known ? *std::min_element(vram_free.begin(), vram_free.end()) : 0.0;
    if (!known && opt.hot < 0) throw std::runtime_error("qwen38 model: sysman free VRAM unavailable; pass a hot count");
    int64_t fit = per_total;
    double room = 0;
    if (known) {
      ls_.free_after_dense_gib = free_now / 1073741824.0;
      room = free_now - (opt.reserve_gib + ls_.onednn_gib) * 1073741824.0;
      fit = std::clamp<int64_t>(static_cast<int64_t>(room / static_cast<double>(stride)), 0, per_total);
    }
    // Some experts stay cold: the live exchange's spare blobs and the prefill's layer-ahead fetch buffers, out of the hot
    // experts (allocated now, so the count below sees them)
    if ((opt.hot >= 0 ? std::min(opt.hot, fit) : fit) < per_total && opt.moe_tp && opt.moe_cpu.mode == "off") {
      const int per = cfg_.experts;
      size_t extra = 0;
      for (int c = 0; c < nc_; ++c) {
        Card &K = *cards_[static_cast<size_t>(c)];
        if (opt.exchange)
          for (int i = 0; i < opt.exchange_spares; ++i) {
            auto *p = static_cast<uint8_t *>(fab.alloc(c, stride));
            x_spare_mem_[c].push_back(p);
            x_spare_[c].push_back(p);
          }
        if (opt.prefill_fetch > 0) {
          auto dev = [&](size_t b) {
            void *p = fab.alloc(c, std::max<size_t>(b, 4096));
            K.mem.push_back(p);
            return p;
          };
          for (int b = 0; b < kPfFetchBufs; ++b) {
            K.pf.fbuf[b] = static_cast<uint8_t *>(dev(stride * static_cast<size_t>(opt.prefill_fetch)));
            K.pf.ftab[b] = static_cast<const uint8_t **>(dev(static_cast<size_t>(per) * 8));
            K.pf.fflag[b] = static_cast<uint8_t *>(dev(static_cast<size_t>(per)));
          }
          K.pf.htab = static_cast<const uint8_t **>(fab.alloc_host(static_cast<size_t>(L) * per * 8));
          K.pf.hflag = static_cast<uint8_t *>(fab.alloc_host(static_cast<size_t>(L) * per));
          K.mem.push_back(K.pf.htab);
          K.mem.push_back(K.pf.hflag);
          K.pf.hsrc.assign(static_cast<size_t>(L), {});
          pf_fetch_cap_ = opt.prefill_fetch;
          ls_.prefill_fetch = pf_fetch_cap_;
          if (c == 0) {  // the fetch order: per layer by ranking (no ranking: by id)
            pf_fetch_order_.resize(static_cast<size_t>(L) * per);
            const bool ranked = so.rank && so.rank->size() == static_cast<size_t>(L) * cfg_.experts;
            for (int l = 0; l < L; ++l) {
              int16_t *o = pf_fetch_order_.data() + static_cast<size_t>(l) * per;
              for (int i = 0; i < per; ++i) o[i] = static_cast<int16_t>(i);
              if (ranked) {
                const double *rk = so.rank->data() + static_cast<size_t>(l) * cfg_.experts;
                std::stable_sort(o, o + per, [&](int16_t a, int16_t b) { return rk[a] > rk[b]; });
              }
            }
          }
        }
        if (c == 0) extra = (x_spare_[0].size() + (pf_fetch_cap_ > 0 ? kPfFetchBufs * static_cast<size_t>(pf_fetch_cap_) : 0)) * stride;
      }
      if (known) fit = std::clamp<int64_t>(static_cast<int64_t>((room - static_cast<double>(extra)) / static_cast<double>(stride)), 0, per_total);
    }
    so.hot_per_card = opt.hot >= 0 ? std::min(opt.hot, fit) : fit;
    if (opt.hot >= 0 && so.hot_per_card < std::min(opt.hot, per_total)) ls_.hot_asked = opt.hot;
  }
  if (!experts_) experts_ = std::make_unique<ExpertStore>(fab, cfg_, opt.moe_tp);
  {  // the cold experts are pinned (never swapped): a plan past the host's RAM gets the process OOM-killed mid-load
    const util::HostMem hm = util::host_mem();
    const double need = static_cast<double>(per_total - so.hot_per_card) * static_cast<double>(stride) * nc_;  // a blob per card
    // (the early reader's buffers are part of the load overhead, not of what the process keeps)
    const double avail = static_cast<double>(hm.available) + static_cast<double>(experts_->reader_bytes()) -
                         (util::kHostKeepGib + util::kLoadHostOverheadGib) * 1073741824.0;
    if (hm.ok && need > avail) {
      char m[400];
      std::snprintf(m, sizeof m,
                    "qwen38 model: the cold experts need %.1f GiB of pinned host memory (%lld of %lld experts hot), but %.1f GiB are available (%.1f GiB "
                    "kept for the process and the system): raise --hot, or free host RAM",
                    need / 1073741824.0, static_cast<long long>(so.hot_per_card), static_cast<long long>(per_total), hm.available / 1073741824.0,
                    util::kHostKeepGib + util::kLoadHostOverheadGib);
      throw std::runtime_error(m);
    }
  }
  experts_->load(ck_, so);
  ls_.experts_s = experts_->stats().total_s;
  if (snap_cap_ > 0)  // the snapshots come on first use: cut them to the host RAM the cold experts left
  {
    double one = 0;
    for (int c = 0; c < nc_; ++c) one += static_cast<double>(snap_bytes(c));
    snap_cap_ = std::min(snap_cap_, util::snap_capacity(opt.snap_gib, one));
  }
  if (opt.verify_experts) ls_.blob_mismatches = experts_->verify();
  const int per = experts_->per_card();
  for (int c = 0; c < nc_; ++c) {
    ls_.hot[c] = experts_->hot_total(c);
    ls_.cold[c] = per_total - ls_.hot[c];
    std::vector<const uint8_t *> t(static_cast<size_t>(L) * per);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < per; ++i) t[static_cast<size_t>(l) * per + i] = opt.moe_tp ? experts_->blob(l, i, nullptr, c) : experts_->blob(l, c * per + i);
    Card &K = *cards_[static_cast<size_t>(c)];
    K.tab = static_cast<const uint8_t **>(fab.alloc(c, t.size() * 8));
    K.mem.push_back(K.tab);
    fab.compute(c).memcpy(K.tab, t.data(), t.size() * 8).wait();
    if (K.pf.moe.stage_cap > 0 || opt.moe_cpu.mode != "off" || pf_fetch_cap_ > 0) {  // cold flags for the prefill staging, the fetch and the CPU path
      std::vector<uint8_t> cf(static_cast<size_t>(L) * per, 0);
      for (int l = 0; l < L; ++l)
        for (int i = 0; i < per; ++i) {
          bool hot = true;
          if (opt.moe_tp) experts_->blob(l, i, &hot, c);
          cf[static_cast<size_t>(l) * per + i] = hot ? 0 : 1;
        }
      if (opt.moe_cpu.mode != "off" && opt.moe_tp) {
        // The CPU path's fixed set (flag 2), per layer pct % of its cold experts interleaved in ranking order
        // (most routed first; no ranking: by id), so the set carries about pct % of the cold reads. A row hands its
        // set experts to the pool by its own count (MoeCpuLink::min_cold): batch-invariant, unlike a per-step pick.
        const int pct = std::clamp(opt.moe_cpu.pct, 0, 100);
        for (int l = 0; l < L; ++l) {
          std::vector<int> cl;
          for (int i = 0; i < per; ++i)
            if (cf[static_cast<size_t>(l) * per + i]) cl.push_back(i);
          if (so.rank && so.rank->size() == static_cast<size_t>(L) * cfg_.experts) {
            const double *rk = so.rank->data() + static_cast<size_t>(l) * cfg_.experts;
            std::stable_sort(cl.begin(), cl.end(), [&](int a, int b) { return rk[a] > rk[b]; });
          }
          for (size_t j = 0; j < cl.size(); ++j)
            if ((static_cast<int64_t>(j) + 1) * pct / 100 > static_cast<int64_t>(j) * pct / 100) cf[static_cast<size_t>(l) * per + cl[j]] = 2;
        }
      }
      K.cold = static_cast<uint8_t *>(fab.alloc(c, std::max<size_t>(cf.size(), 4096)));
      K.mem.push_back(K.cold);
      fab.compute(c).memcpy(K.cold, cf.data(), cf.size()).wait();
    }
  }
  setup_moe_cpu(opt);
  lend_setup(opt, so.rank);
  exchange_setup(opt, so.rank);
  dsprof_ = util::env_flag("XEN_DEC_PROFILE");
  if (dsprof_)
    for (int c = 0; c < nc_; ++c) {
      dec_stamps_[c] = static_cast<uint64_t *>(fab.alloc(static_cast<size_t>(c), kDecStampWords * 8));
      cards_[static_cast<size_t>(c)]->mem.push_back(dec_stamps_[c]);
      fab.compute(static_cast<size_t>(c)).memset(dec_stamps_[c], 0, kDecStampWords * 8).wait();
    }
  ls_.total_s = since_s(t_all);
  for (int s = opt.slots - 1; s >= 0; --s) {
    use_slot(s);
    begin_sequence();
  }
}

Model::~Model() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  if (xpl_) {
    for (int c = 0; c < nc_; ++c) fab_.copy(static_cast<size_t>(c)).wait();
    std::fprintf(stderr, "qwen38: expert exchange: %lld experts moved in %lld rounds (%lld put off), %.1f ms of host time\n",
                 static_cast<long long>(x_st_.swaps), static_cast<long long>(x_st_.rounds), static_cast<long long>(x_st_.deferred), x_st_.host_ms);
  }
  for (int c = 0; c < kMaxCards; ++c) {
    for (void *p : x_spare_mem_[c]) fab_.release(p);
    if (x_upd_[c]) fab_.release(x_upd_[c]);
  }
  if (cpu_on_) {
    const auto st = moe_cpu_stats();
    std::fprintf(stderr, "qwen38: MoE CPU path: %llu requests, %llu experts (%.1f us per request)%s\n", static_cast<unsigned long long>(st.requests),
                 static_cast<unsigned long long>(st.experts), st.requests ? st.busy_us / static_cast<double>(st.requests) : 0.0,
                 st.timed_out ? "; switched off after a timeout" : "");
  }
  for (auto &p : cpu_pool_) p.reset();  // the pools read the host buffers K.mem frees below
  if (pf_graphs_n_ > 0)
    std::fprintf(stderr, "qwen38: prompt-call graphs: %d recorded (%.0f ms recording + finalizing), %lld calls replayed\n", pf_graphs_n_, pf_graphs_ms_,
                 static_cast<long long>(pf_graphs_replayed_));
  try {
    save_ranking();  // the counts since the last save
  } catch (const std::exception &e) {
    std::fprintf(stderr, "qwen38: hot-expert ranking at exit: %s\n", e.what());
  }
  if (mtp_) {
    for (void *ptr : mtp_->mem) fab_.release(ptr);
    mtp_.reset();
  }
  dflash_.reset();  // (its graphs and buffers; it reads the head's tables, kept below)
  for (auto &card : cards_)
    for (void *ptr : card->mem) fab_.release(ptr);
  if (ple_pinned_) fab_.release(ple_pinned_);
  if (ple_words_) fab_.release(ple_words_);
  if (seg_pin_) fab_.release(seg_pin_);
  if (lg_host_) fab_.release(lg_host_);
  if (pf_ple_) fab_.release(pf_ple_);
  for (auto &r : snaps_)
    for (void *h : r.host)
      if (h) fab_.release(h);
}

void Model::use_slot(int slot) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  slot_ = slot;
}

void Model::set_rope_tier(int slot, int t) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  if (t < 0 || t >= rope_tiers()) throw std::invalid_argument("qwen38 model: no YaRN tier " + std::to_string(t));
  Seq &sq = seqs_[static_cast<size_t>(slot)];
  if (sq.pos != 0) throw std::logic_error("qwen38 model: a sequence's YaRN tier is set at position 0");
  sq.rope_delta = tier_base_[static_cast<size_t>(t)];
}

void Model::begin_sequence() {
  Seq &sq = seqs_[static_cast<size_t>(slot_)];
  if (sq.pending) throw std::logic_error("qwen38 model: begin_sequence while a verify waits for its commit");
  const size_t s = static_cast<size_t>(slot_);
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    const Card &K = *cards_[static_cast<size_t>(c)];
    for (int l : gs_.layers) {
      const k::GdnState &st = K.gst[static_cast<size_t>(l)];
      q.memset(st.S + s * st.slot_S, 0, st.slot_S * sizeof(float));
      for (int pr = 0; pr < 2; ++pr) q.memset(st.win[pr] + s * st.slot_win, 0, st.slot_win * sizeof(bf16));
    }
    q.wait();
  }
  sq = Seq{};
  sq.ple_ctx.reset(ple_tab_->layout());
  if (dflash_) dflash_->settle(slot_, 0);  // a fresh sequence drops the slot's deferred drafter rows
  kvp_.release(slot_);  // a fresh sequence holds no pages of a shared pool (the fixed layout keeps its own)
}

// The GDN layers' attention input by owner (prefill.cpp pf_gdn_owner_inputs): where a prompt's rows live on their
// owners (four cards, expert TP: pf_lane_count) and the qkvz runs W8A8 per row (a row's int8 form does not depend on the call), each card gets
// every card's a | b weights stacked per layer (bf16 [card][a | b][hidden]: ~0.5 MB a layer) and the chunk-row buffers
// the owners' shares land in (~13 MB a card at 4,096-row chunks). Otherwise off (null buffers).
void Model::pf_gdn_owner_init() {
  if (nc_ != 4 || !ms_.shape.tp || !opt_.prefill_dense_a8 || !opt_.prefill_dense_a8_row || opt_.calib) return;
  for (int l : gs_.layers)
    for (int c = 0; c < nc_; ++c)
      if (gs_.w[c][l].qkvz4.w || gs_.w[c][l].qkvz.kind != k::DenseKind::i8) return;  // an int4 qkvz reads bf16 x
  const int H = cfg_.hidden, abn = gs_.shape_of(0).ab_n(), vh = gs_.shape_of(0).vh();
  const size_t rows = static_cast<size_t>(opt_.prefill_chunk) + 4, head = static_cast<size_t>(vh) * H;
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    auto &q = fab_.compute(c);
    auto zeroed = [&](size_t bytes) {
      void *p = fab_.alloc(static_cast<size_t>(c), bytes);
      q.memset(p, 0, bytes);
      K.mem.push_back(p);
      return p;
    };
    K.pf.gx_q = static_cast<int8_t *>(zeroed(rows * H));
    K.pf.gx_s = static_cast<float *>(zeroed(static_cast<size_t>(k::pf_a8_ld(static_cast<int>(rows))) * sizeof(float)));
    K.pf.gx_ab = static_cast<float *>(zeroed(rows * nc_ * abn * sizeof(float)));
    K.pf.gx_w.assign(static_cast<size_t>(cfg_.layers), nullptr);
    for (int l : gs_.layers) {
      auto *w = static_cast<bf16 *>(fab_.alloc(static_cast<size_t>(c), nc_ * 2 * head * sizeof(bf16)));
      K.mem.push_back(w);
      for (int s = 0; s < nc_; ++s) {  // card s's a | b, copied on card s's queue
        const k::GdnWeights &from = gs_.w[s][l];
        auto &cq = fab_.compute(s);
        cq.memcpy(w + (2 * s) * head, from.a, head * sizeof(bf16));
        cq.memcpy(w + (2 * s + 1) * head, from.b, head * sizeof(bf16));
      }
      K.pf.gx_w[static_cast<size_t>(l)] = w;
    }
  }
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait_and_throw();
}

bool Model::kv_reserve(int slot, int64_t tokens) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  if (tokens > cap_) return false;
  if (!kvp_.reserve(slot, tokens)) {
    // pages past the lend limit come back from the lent experts first
    if (!lend_st_.on || !lend_raise(kvp_.pages_short(slot, tokens)) || !kvp_.reserve(slot, tokens)) return false;
  }
  kv_upload();
  return true;
}

// ---- KV lending ----

void Model::lend_setup(const ModelOptions &opt, const std::vector<double> *rank) {
  ls_.kv_lend = "off";
  if (!opt.kv_lend) return;
  auto refuse = [&](const std::string &why) { ls_.kv_lend = "off (" + why + ")"; };
  if (!kvp_.shared()) return refuse("needs the shared KV pool");
  if (opt.kv_pool_scramble) return refuse("the pool is scrambled");
  if (!opt.moe_tp || !experts_) return refuse("needs the expert TP layout");
  const int L = cfg_.layers, per = experts_->per_card(), PP = kvp_.pool_pages();
  // the pool's buffers per card, in the same order on both cards
  std::vector<size_t> pb;
  for (int c = 0; c < nc_; ++c) {
    std::vector<size_t> bytes;
    auto add = [&](void *p, size_t b) {
      if (!p || b == 0) return;
      if (reinterpret_cast<uintptr_t>(p) % 1024) throw std::logic_error("qwen38 model: a KV pool buffer is not 1 KiB aligned");
      lend_buf_[c].push_back(static_cast<uint8_t *>(p));
      bytes.push_back(b);
    };
    auto add_cache = [&](const k::QsaCache &cc, const k::QsaShape &sh) {
      add(cc.k, k::qsa_page_kv_bytes(sh));
      add(cc.v, k::qsa_page_v_bytes(sh));
      add(cc.ks, k::qsa_page_scale_floats(sh) * 4);
      add(cc.vs, k::qsa_page_vscale_floats(sh) * 4);
      add(cc.kc, k::qsa_page_kc_elems() * 2);
    };
    const Card &K = *cards_[static_cast<size_t>(c)];
    for (int l : qs_.layers) add_cache(K.qc[static_cast<size_t>(l)], qs_.shape_of(c));
    if (mtp_) {
      auto sh = qs_.shape_of(c);
      sh.kv = opt_.mtp_kv;
      add_cache(mtp_->cards[static_cast<size_t>(c)].qc, sh);
    }
    if (c == 0) pb = bytes;
    else if (bytes != pb) return refuse("the cards' pools differ");
  }
  // the candidates: cold experts, most routed first (no ranking: layer-interleaved by id)
  std::vector<std::pair<int, int>> cand;
  for (int l = 0; l < L; ++l)
    for (int i = 0; i < per; ++i) {
      bool hot = true;
      experts_->blob(l, i, &hot, 0);
      if (!hot) cand.emplace_back(l, i);
    }
  if (rank && rank->size() == static_cast<size_t>(L) * cfg_.experts)
    std::stable_sort(cand.begin(), cand.end(), [&](const auto &a, const auto &b) {
      return (*rank)[static_cast<size_t>(a.first) * cfg_.experts + a.second] > (*rank)[static_cast<size_t>(b.first) * cfg_.experts + b.second];
    });
  else
    std::stable_sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) { return a.second != b.second ? a.second < b.second : a.first < b.first; });
  if (cand.empty()) return refuse("every expert is in VRAM");
  lend_ = model::KvLend(pb, PP, experts_->stride(), 1024, cand.size());
  cand.resize(std::min(cand.size(), lend_.size()));
  lend_cand_ = std::move(cand);
  for (int c = 0; c < nc_; ++c) {
    tab_host_[c].resize(static_cast<size_t>(L) * per);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < per; ++i) tab_host_[c][static_cast<size_t>(l) * per + i] = experts_->blob(l, i, nullptr, c);
    if (cards_[static_cast<size_t>(c)]->cold) {  // the flags as uploaded (1 cold, 2 the CPU path's set): an eviction restores them
      cold_host_[c].resize(static_cast<size_t>(L) * per);
      fab_.compute(c).memcpy(cold_host_[c].data(), cards_[static_cast<size_t>(c)]->cold, cold_host_[c].size()).wait();
      cold_orig_[c] = cold_host_[c];
    }
  }
  lend_keep_pages_ = std::max(1, model::KvPages::pages_for(opt.kv_lend_keep));
  const int n = std::min(lend_.usable(lend_keep_pages_), static_cast<int>(lend_cand_.size()));
  lend_st_.on = true;
  lend_st_.slots = static_cast<int64_t>(lend_.size());
  if (!kvp_.set_limit(std::max(lend_keep_pages_, lend_.limit_for(n, PP)))) throw std::logic_error("qwen38 model: the lend limit at load");
  lend_copy(0, n);
  lend_upload();
  lend_n_ = n;
  lend_st_.filled = 0;
  ls_.lent = n;
  char b[200];
  std::snprintf(b, sizeof b, "on: %d cold experts lent into the KV pool's free pages (%zu slots), KV limit %lld tokens, keep %lld", n, lend_.size(),
                static_cast<long long>(kvp_.limit()) * model::KvPages::kPageTokens, static_cast<long long>(opt.kv_lend_keep));
  ls_.kv_lend = b;
}

void Model::lend_copy(int from, int to) {
  const int per = experts_->per_card();
  const size_t stride = experts_->stride();
  for (int k = from; k < to; ++k) {
    const auto &sl = lend_.slots()[static_cast<size_t>(k)];
    const auto [l, i] = lend_cand_[static_cast<size_t>(k)];
    const size_t e = static_cast<size_t>(l) * per + i;
    for (int c = 0; c < nc_; ++c) {
      uint8_t *dst = lend_buf_[c][static_cast<size_t>(sl.buf)] + sl.off;
      fab_.compute(c).memcpy(dst, experts_->blob(l, i, nullptr, c), stride);  // pinned host copy -> the free pages
      tab_host_[c][e] = dst;
      if (!cold_host_[c].empty()) cold_host_[c][e] = 0;
    }
  }
  lend_st_.filled += static_cast<uint64_t>(std::max(0, to - from));
}

void Model::lend_upload() {
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    auto &q = fab_.compute(c);
    q.memcpy(K.tab, tab_host_[c].data(), tab_host_[c].size() * sizeof(const uint8_t *));
    if (K.cold && !cold_host_[c].empty()) q.memcpy(K.cold, cold_host_[c].data(), cold_host_[c].size());
  }
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();  // the mirrors are pageable and change again
}

bool Model::lend_raise(int short_pages) {
  if (short_pages <= 0) return true;
  const auto t0 = std::chrono::steady_clock::now();
  const int PP = kvp_.pool_pages();
  if (kvp_.limit() >= PP) return false;
  // pages [limit, want) are all free (no KV past the limit): they cover the shortfall
  const int want = std::min(PP, kvp_.limit() + (short_pages - kvp_.free_below_limit()) + lend_keep_pages_);
  const int keep = std::min(lend_n_, lend_.usable(want));
  // every kernel that may read a blob being evicted runs before the KV writes into its slot: drain both cards
  for (int c = 0; c < nc_; ++c) {
    fab_.compute(c).wait();
    if (cards_[static_cast<size_t>(c)]->pf.side) cards_[static_cast<size_t>(c)]->pf.side->wait();
  }
  const int per = experts_->per_card();
  for (int k = keep; k < lend_n_; ++k) {
    const auto [l, i] = lend_cand_[static_cast<size_t>(k)];
    const size_t e = static_cast<size_t>(l) * per + i;
    for (int c = 0; c < nc_; ++c) {
      tab_host_[c][e] = experts_->blob(l, i, nullptr, c);  // the host copy (pinned, kept)
      if (!cold_host_[c].empty()) cold_host_[c][e] = cold_orig_[c][e];
    }
  }
  lend_st_.evicted += static_cast<uint64_t>(lend_n_ - keep);
  lend_n_ = keep;
  lend_upload();
  kvp_.set_limit(std::max(want, lend_.limit_for(keep, PP)));
  lend_st_.raises++;
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  lend_st_.raise_ms += ms;
  std::fprintf(stderr, "qwen38: KV lending: KV limit -> %lld tokens, %d lent experts given back (%d stay), %.1f ms\n",
               static_cast<long long>(kvp_.limit()) * model::KvPages::kPageTokens, static_cast<int>(lend_st_.evicted - lend_ev_seen_), lend_n_, ms);
  lend_ev_seen_ = lend_st_.evicted;
  return kvp_.free_below_limit() >= short_pages;
}

int Model::lend_pending() const {
  if (!lend_st_.on) return 0;
  const int target = std::max(lend_keep_pages_, kvp_.high_water() + lend_keep_pages_);
  return std::max(0, std::min(lend_.usable(target), static_cast<int>(lend_cand_.size())) - lend_n_);
}

int Model::lend_idle(int max_blobs) {
  const int n = std::min(lend_pending(), std::max(0, max_blobs));
  if (n <= 0) return 0;
  const auto t0 = std::chrono::steady_clock::now();
  // the limit first (above the held pages: no KV there now; kernels still in flight on freed pages run before the copies
  // on the same in-order queues), then the copies, then the table
  if (!kvp_.set_limit(std::max(lend_keep_pages_, lend_.limit_for(lend_n_ + n, kvp_.pool_pages())))) return 0;
  lend_copy(lend_n_, lend_n_ + n);
  lend_n_ += n;
  lend_upload();
  lend_st_.fill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  lend_refill_run_ += n;
  if (lend_pending() == 0) {
    std::fprintf(stderr, "qwen38: KV lending: %d experts lent again while idle (%d lent, KV limit %lld tokens)\n", lend_refill_run_, lend_n_,
                 static_cast<long long>(kvp_.limit()) * model::KvPages::kPageTokens);
    lend_refill_run_ = 0;
  }
  return n;
}

bool Model::kv_blocks_lend(int slot) const {
  if (!lend_st_.on || !kvp_.shared()) return false;
  for (int i = 0; i < kvp_.held_pages(slot); ++i)
    if (kvp_.page(slot, i) >= lend_keep_pages_) return true;
  return false;
}

bool Model::kv_reserve_lends(int slot, int64_t tokens) const {
  if (!lend_st_.on || tokens > cap_ || kvp_.limit() >= kvp_.pool_pages()) return false;
  return kvp_.pages_short(slot, tokens) > kvp_.free_below_limit();
}

Model::LendStats Model::lend_stats() const {
  LendStats s = lend_st_;
  s.lent = lend_n_;
  s.limit_tokens = static_cast<int64_t>(kvp_.limit()) * model::KvPages::kPageTokens;
  return s;
}

void Model::kv_release(int slot) {
  if (slot < 0 || slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
  kvp_.release(slot);
}

void Model::kv_need(int slot, int64_t tokens) {
  if (!kv_reserve(slot, std::min(tokens, cap_)))
    throw std::runtime_error("qwen38 model: the KV pool is out of pages (slot " + std::to_string(slot) + " needs " + std::to_string(tokens) + " tokens, " +
                             std::to_string(kvp_.free_tokens()) + " free)");
}

// The host table's changed entries to both cards: rare (a page per 64 tokens per sequence) and small, from pageable
// memory, so wait (the host table may change again right after). In order behind any step in flight.
void Model::kv_upload() {
  if (!kvp_.dirty()) return;
  const size_t lo = kvp_.dirty_lo(), n = kvp_.dirty_hi() - lo;
  for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(kv_table_[c] + lo, kvp_.table().data() + lo, n * 4);
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
  kvp_.clear_dirty();
}

size_t Model::kv_page_bytes(int c) const {
  return qs_.layers.size() * qsa_page_bytes(qs_.shape_of(c)) + (mtp_ ? qsa_page_bytes(mtp_shape(qs_, opt_.mtp_kv, c)) : 0);
}

// The auto pool's rule: at least the pool asked for (kv_pool_tokens); with kv_pool_fill, the pages the tighter card holds
// in its free VRAM once every routed expert blob, the reserve, the oneDNN repack slots and pool_fill_keep_gib are set
// aside; at most slots x context.
int Model::kv_pool_pages_for(const std::vector<double> &vram_free, double expert_bytes) const {
  const int asked = kvp_.pool_pages();
  if (!opt_.kv_pool_fill || !kvp_.shared() || vram_free.empty()) return asked;
  const int64_t all = static_cast<int64_t>(opt_.slots) * (cap_ / k::kQsaPage);
  int64_t fit = all;
  for (int c = 0; c < nc_; ++c) {
    const double left = vram_free[static_cast<size_t>(c)] - (opt_.reserve_gib + ls_.onednn_gib + pool_fill_keep_gib(cap_)) * kGiB - expert_bytes;
    fit = std::min<int64_t>(fit, left > 0 ? static_cast<int64_t>(left / static_cast<double>(kv_page_bytes(c))) : 0);
  }
  return static_cast<int>(std::clamp<int64_t>(fit, asked, all));
}

// The pool's K / V / scales / compressed keys for every QSA layer and the MTP drafter's layer (zeroed), and the page
// table, for pool_pages pages per card (the fixed layout: its identity pages, unchanged).
void Model::kv_pool_alloc(int pool_pages) {
  const int slot_pages = static_cast<int>(cap_ / k::kQsaPage);
  if (kvp_.shared() && pool_pages != kvp_.pool_pages()) kvp_ = model::KvPages(opt_.slots, slot_pages, pool_pages, true, opt_.kv_pool_scramble);
  const size_t PP = static_cast<size_t>(kvp_.pool_pages());
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    auto dev = [&](size_t bytes, std::vector<void *> &mem) {
      bytes = std::max<size_t>(bytes, 64);
      void *ptr = fab_.alloc(c, bytes);
      q.memset(ptr, 0, bytes);
      mem.push_back(ptr);
      return ptr;
    };
    auto pages = [&](k::QsaCache &cc, const k::QsaShape &s, std::vector<void *> &mem) {
      // the KV split by tokens with draft trees: the tree step area's rows past the pool's (QsaCache::tree_row0)
      const size_t area = s.kv_ways() > 1 && cc.tkc ? static_cast<size_t>(k::kQsaTreeAreaRows) * s.hkv() : 0;
      cc.k = static_cast<uint8_t *>(dev(PP * k::qsa_page_kv_bytes(s) + area * k::kQsaDim * s.kv_elem(), mem));
      cc.v = static_cast<uint8_t *>(dev(PP * k::qsa_page_v_bytes(s) + area * s.v_row_bytes(), mem));
      cc.ks = static_cast<float *>(dev((PP * k::qsa_page_scale_floats(s) + area) * 4, mem));
      cc.vs = static_cast<float *>(dev((PP * k::qsa_page_vscale_floats(s) + area * s.v_scale_bytes() / 4) * 4, mem));
      cc.kc = static_cast<bf16 *>(dev(PP * k::qsa_page_kc_elems() * 2, mem));
      cc.tree_row0 = area ? static_cast<int>(PP * s.hkv() * s.page_tokens()) : -1;
    };
    Card &K = *cards_[static_cast<size_t>(c)];
    for (int l : qs_.layers) pages(K.qc[static_cast<size_t>(l)], qs_.shape_of(c), K.mem);
    if (mtp_) {
      const auto ms = mtp_shape(qs_, opt_.mtp_kv, c);
      pages(mtp_->cards[static_cast<size_t>(c)].qc, ms, mtp_->mem);
      if (c == 0) ls_.mtp_vram += PP * qsa_page_bytes(ms);
    }
    kv_bytes_[c] = PP * kv_page_bytes(c);
    q.memcpy(kv_table_[c], kvp_.table().data(), kvp_.table().size() * 4);
    q.wait();
  }
  kvp_.clear_dirty();
}

k::QsaScratch Model::qsa_pair_region(const k::QsaScratch &sc, int region) const {
  k::QsaScratch r = sc;
  if (r.pair.inbox) {
    r.pair.inbox += kv_pair_region_words_ * static_cast<size_t>(region);
    r.pair.peer_inbox += kv_pair_region_words_ * static_cast<size_t>(region);
    r.pair_parity = 0;
  }
  return r;
}

void Model::step(const int32_t *tokens, int rows) {
  verify(tokens, rows);
  commit(rows);
}

void Model::verify(const int32_t *tokens, int rows) { verify_batch({SeqRows{slot_, tokens, rows}}); }

void Model::commit(int n) {
  if (batch_slots_.size() != 1 || batch_slots_[0] != slot_) throw std::logic_error("qwen38 model: commit(n) after a verify of the current slot only");
  commit_batch({n});
}

void Model::step_batch(const std::vector<SeqRows> &segs) {
  verify_batch(segs);
  std::vector<int> n;
  for (const auto &g : segs) n.push_back(g.rows);
  commit_batch(n);
}

void Model::commit_batch(const std::vector<int> &n) {
  if (batch_slots_.empty() || n.size() != batch_slots_.size()) throw std::logic_error("qwen38 model: commit_batch takes one count per segment of the last verify");
  for (size_t i = 0; i < n.size(); ++i) {
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    if (!sq.pending) throw std::logic_error("qwen38 model: commit without a verify");
    if (n[i] < 1 || n[i] > sq.run_rows) throw std::invalid_argument("qwen38 model: commit takes 1..rows of the verify");
    if (sq.run_tree && n[i] > 1) throw std::logic_error("qwen38 model: a tree segment kept past its root needs its path (commit_batch(n, paths))");
  }
  for (size_t i = 0; i < n.size(); ++i) {
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    sq.path = 0, sq.run_tree = false;  // (commit_batch(n, paths) sets the path after this)
    for (int j = 0; j < n[i]; ++j) sq.ple_ctx.push(ple_tab_->layout(), sq.run_tokens[j]);
    sq.pos += n[i];
    sq.last_rows = batch_rows_ == 1 && batch_code_ == 1 ? 0 : n[i];  // a 1-row step committed its GDN rows
    sq.gdn_par ^= 1;  // the step wrote its window / pending log at gdn_par: the next step reads them
    sq.pending = false;
    sq.pf_committed = false;  // 
  }
  batch_slots_.clear();
}

// The accept of a step whose segments may be trees. Each tree segment's walked rows become its
// positions p0 .. p0 + n - 1: the QSA K / V / scales / raw index keys / closed blocks' compressed keys and the PLE conv
// inputs of row paths[i][x] move to position p0 + x (kernels::qsa_kv_move, ple_move), the GDN replays the path at the
// next step (kSegPath), and the verify's feature rows are re-laid in path order for the DFlash pass (row r0 + x =
// r0 + path[min(x, n - 1)]). A path of rows 0 .. n - 1 (the tree's leading chain) moves nothing.
void Model::commit_batch(const std::vector<int> &n, const std::vector<const int *> &paths) {
  if (batch_slots_.empty() || n.size() != batch_slots_.size() || paths.size() != n.size())
    throw std::logic_error("qwen38 model: commit_batch takes one count and path per segment of the last verify");
  const size_t ns = n.size();
  std::vector<int64_t> p0(ns);
  std::vector<int> rows(ns), row0(ns);
  int total = 0;
  for (size_t i = 0; i < ns; ++i) {
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    p0[i] = sq.pos, rows[i] = sq.run_rows, row0[i] = total, total += sq.run_rows;
    const int *path = paths[i];
    if (!path) continue;
    if (!sq.pending || !sq.run_tree) throw std::logic_error("qwen38 model: a path for a segment whose rows were not a tree");
    if (n[i] < 1 || n[i] > sq.run_rows || path[0] != 0) throw std::invalid_argument("qwen38 model: commit_batch's path takes 1..rows rows from the root");
    for (int r = 1; r < n[i]; ++r)
      if (path[r] <= path[r - 1] || path[r] >= sq.run_rows || sq.run_parent[path[r]] != path[r - 1])
        throw std::invalid_argument("qwen38 model: commit_batch's path descends the tree");
  }
  std::vector<uint64_t> words(ns, 0);
  for (size_t i = 0; i < ns; ++i) {
    Seq &sq = seqs_[static_cast<size_t>(batch_slots_[i])];
    if (!paths[i]) continue;
    words[i] = k::tree_path_word(paths[i], n[i]);
    int32_t t[Head::kRows];
    for (int x = 0; x < n[i]; ++x) t[x] = sq.run_tokens[paths[i][x]];
    std::copy_n(t, n[i], sq.run_tokens);  // the committed tokens in path order (the n-gram context)
    sq.run_tree = false;
  }
  commit_batch(n);
  bool any = false;
  for (size_t i = 0; i < ns; ++i) {
    seqs_[static_cast<size_t>(run_slots_[i])].path = words[i];
    any |= words[i] != 0;
  }
  if (!any) return;
  int map[arch::DFlash::kRows] = {};
  for (int r = 0; r < arch::DFlash::kRows; ++r) map[r] = std::min(r, total - 1);
  for (size_t i = 0; i < ns; ++i)
    if (words[i])
      for (int r = 0; r < rows[i]; ++r) map[row0[i] + r] = row0[i] + paths[i][std::min(r, n[i] - 1)];
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    Card &K = *cards_[static_cast<size_t>(c)];
    std::vector<k::QsaCache> qc;
    for (int l : qs_.layers) qc.push_back(K.qc[static_cast<size_t>(l)]);
    for (size_t i = 0; i < ns; ++i) {
      if (!words[i]) continue;
      k::QsaMove mv;
      mv.n = n[i];
      for (int x = 0; x < n[i]; ++x) mv.from[x] = paths[i][x];
      k::qsa_kv_move(q, qs_.shape_of(c), qc.data(), static_cast<int>(qc.size()), run_slots_[i], static_cast<int>(p0[i]), row0[i], mv);
      k::ple_move(q, ps_.shape, K.ple_st, run_slots_[i], static_cast<int>(p0[i]), paths[i], n[i]);
    }
  }
  if (dflash_) {  // the verify's feature rows in path order
    dflash_->save_step_features(total);
    dflash_->stage_step_features(map, total);
  }
}

void Model::verify_batch(const std::vector<SeqRows> &segs) {
  const int ns = static_cast<int>(segs.size());
  if (ns < 1 || ns > k::kMaxSegs) throw std::invalid_argument("qwen38 model: 1..64 segments per step");
  const int sr = segs[0].rows;
  // A step: up to kMaxRows rows (the 8-row shape), a sequence's segment up to 4 (GDN's core; 8 with DFlash).
  // Unequal segments: the first k of sr rows, the rest of one row (kernels::seg_het).
  int k_wide = 0, rows = 0;
  while (k_wide < ns && segs[static_cast<size_t>(k_wide)].rows == sr) ++k_wide;
  for (const auto &g : segs) rows += g.rows;
  const int code = k_wide == ns ? sr : k::seg_het(sr, k_wide, ns);
  if (sr < 1 || sr > seg_rows_max_ || rows > step_cap_)
    throw std::invalid_argument("qwen38 model: 1.." + std::to_string(seg_rows_max_) + " rows per segment, 1.." + std::to_string(step_cap_) + " in all");
  for (int i = 0; i < ns; ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    if (g.rows != (i < k_wide ? sr : 1))
      throw std::invalid_argument("qwen38 model: the segments of a step have the same rows, or B rows each and then one row each");
    if (g.slot < 0 || g.slot >= slots()) throw std::invalid_argument("qwen38 model: no such slot");
    for (int j = 0; j < i; ++j)
      if (segs[static_cast<size_t>(j)].slot == g.slot) throw std::invalid_argument("qwen38 model: a slot appears twice in one step");
    const Seq &sq = seqs_[static_cast<size_t>(g.slot)];
    if (sq.pending) throw std::logic_error("qwen38 model: verify before the last one was committed");
    if (sq.pos + g.rows > cap_) throw std::runtime_error("qwen38 model: the sequence outgrew its KV capacity");
  }
  // segments whose rows are a draft tree (rows.hpp: preorder, each row's parent the row before it or
  // one of that row's ancestors)
  bool any_tree = false;
  for (const auto &g : segs) {
    if (!g.parent) continue;
    if (!drafts_trees() || !cards_[0]->tree_S) throw std::invalid_argument("qwen38 model: tree rows need the DFlash drafter's trees (ModelOptions::dflash_tree)");
    if (g.rows > k::kGdnMaxRows || g.parent[0] != -1) throw std::invalid_argument("qwen38 model: a tree segment holds 1..8 rows from its root");
    for (int r = 1; r < g.rows; ++r) {
      bool up = false;  // the parent is row r - 1 or one of its ancestors
      for (int x = r - 1; x >= 0 && !up; x = g.parent[x]) up = g.parent[r] == x;
      if (g.parent[r] < 0 || g.parent[r] >= r || !up) throw std::invalid_argument("qwen38 model: a tree's rows go in depth-first preorder");
    }
    any_tree = true;
  }
  for (const auto &g : segs) kv_need(g.slot, seqs_[static_cast<size_t>(g.slot)].pos + g.rows);  // the rows' KV pages
  // The PLE row ids: each segment's from its own n-gram context, which advances only at commit (rejected rows must
  // not enter it). The rows themselves are read from the drive after the step is submitted, while its first layers
  // run (the PLE layer's kernels::ple_wait_rows waits for them).
  int32_t tokens[Head::kRows];
  const int heads = ple_tab_->heads();
  std::vector<int64_t> ple_ids(static_cast<size_t>(rows) * heads);
  for (int i = 0, r0 = 0; i < ns; r0 += segs[static_cast<size_t>(i)].rows, ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    int64_t toks[Head::kRows];
    for (int r = 0; r < g.rows; ++r) {
      toks[r] = g.tokens[r];
      tokens[r0 + r] = g.tokens[r];
    }
    if (!g.parent) {
      kernels::PleContext ctx = seqs_[static_cast<size_t>(g.slot)].ple_ctx;
      ple_tab_->ids(toks, g.rows, ctx, ple_ids.data() + static_cast<size_t>(r0) * heads);
      continue;
    }
    // a tree row's n-gram ids from its path (the root .. the row), its last row's
    std::vector<int64_t> path_ids(static_cast<size_t>(Head::kRows) * heads);
    for (int r = 0; r < g.rows; ++r) {
      int64_t pt[Head::kRows];
      int n = 0;
      for (int x = r; x >= 0; x = g.parent[x]) ++n;
      for (int x = r, i = n - 1; x >= 0; x = g.parent[x], --i) pt[i] = g.tokens[x];
      kernels::PleContext ctx = seqs_[static_cast<size_t>(g.slot)].ple_ctx;
      ple_tab_->ids(pt, n, ctx, path_ids.data());
      std::copy_n(path_ids.data() + static_cast<size_t>(n - 1) * heads, heads, ple_ids.data() + static_cast<size_t>(r0 + r) * heads);
    }
  }
  const uint32_t ple_n = ++ple_seq_;
  __atomic_store_n(&ple_words_[0], ple_n, __ATOMIC_RELEASE);  // the step's number, before the submission
  const auto t1 = std::chrono::steady_clock::now();
  // Segment records (rows.hpp): position, accept, GDN parity, slot.
  int64_t need = 0;
  int amax = 0;  // the largest accept word (more than 4: the GDN's 8-row core stages the previous rows)
  bool replay = false;  // a segment replays an accepted tree path (the GDN tree core)
  std::fill(std::begin(seg_words_), std::end(seg_words_), 0);
  for (int i = 0; i < ns; ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(segs[static_cast<size_t>(i)].slot)];
    int32_t *w = seg_words_ + i * k::kSegRec;
    w[k::kSegPos] = static_cast<int32_t>(sq.pos);
    w[k::kSegRope] = static_cast<int32_t>(sq.rope_delta);  // its YaRN tier's table rows (qsa.hpp)
    w[k::kSegAccept] = sq.last_rows;
    amax = std::max(amax, sq.last_rows);
    w[k::kSegParity] = sq.gdn_par;
    w[k::kSegSlot] = segs[static_cast<size_t>(i)].slot;
    need = std::max(need, sq.pos + segs[static_cast<size_t>(i)].rows);
    // the tree's words (rows.hpp) and the last accepted path (the GDN replays it)
    const auto &g = segs[static_cast<size_t>(i)];
    if (g.parent) {
      k::rec_word_set(w, k::kSegTree, k::kSegTreeHi, k::tree_word(g.parent, g.rows));
      k::rec_word_set(w, k::kSegTreeAux, k::kSegTreeAux2, k::tree_aux_word(g.parent, g.rows));
    }
    if (sq.last_rows > 0 && sq.path != 0) {
      k::rec_word_set(w, k::kSegPath, k::kSegPathHi, sq.path);
      replay = true;
    }
  }
  // The records go to pinned memory; the step's first command (inside the graph) copies them
  // to the cards.
  std::memcpy(seg_pin_, seg_words_, sizeof(seg_words_));
  head_->set_ids(tokens, rows);
  t_.replayed = opt_.graphs && !profiling_;
  t_.record = 0;
  // the step's shape key carries the GDN accept bound (wide: the 8-row core; only with 8-row logs) and p1, every accept
  // word <= 1 (the core stages one previous row instead of 4: bit-identical)
  // with draft trees on, every multi-row step runs the tree instances (a chain segment in them computes
  // the chain's bits: test-qsa / test-ple / test-gdn), so a step's graph key is the chain's and the warm-up records
  // what the trees replay (else each row count's tree / replay variant records on first use while serving)
  // (a lone sequence drafts chains, tree_step_seqs: its steps keep the chain kernels)
  const bool tree_kind = drafts_trees() && sr >= 2 && tree_step_seqs(ns) != arch::DFlash::kChainsOnly;
  const int sk = seg_key(code, amax > k::kGdnRows, amax <= 1, any_tree || tree_kind, replay || tree_kind);
  if (t_.replayed)
    run_graph(rows, sk, need);
  else
    submit_step(rows, sk, static_cast<int>(need));
  const auto t2 = std::chrono::steady_clock::now();
  {  // the rows while the step runs; published even when the read fails (the step then ends, the error follows)
    std::exception_ptr err;
    try {
      std::vector<uint8_t> codes(ple_ids.size() * ple_tab_->row_bytes());
      ple_tab_->read(ple_ids.data(), rows, codes.data());
      ple_tab_->dequant(codes.data(), rows, ple_pinned_);
    } catch (...) {
      err = std::current_exception();
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    __atomic_store_n(&ple_words_[16], ple_n, __ATOMIC_RELEASE);
    t_.ple_read = since_ms(t2);
    if (err) {
      head_->wait();
      std::rethrow_exception(err);
    }
  }
  head_->wait();
  for (int c = 0; c < nc_; ++c)
    if (__atomic_load_n(&ple_words_[32 + 16 * c], __ATOMIC_ACQUIRE) != 0u)
      throw std::runtime_error("qwen38 model: card " + std::to_string(c) + " timed out waiting for the step's PLE rows");
  dec_finish();  // XEN_DEC_PROFILE: the replayed step's stamps
  t_.submit = std::chrono::duration<double, std::milli>(t2 - t1).count();
  t_.wait = since_ms(t2);
  if (rec_ && rec_->due()) save_ranking();  // once a period, the step done (a ~0.1 MB copy + the file)
  exchange_tick();  // the live expert exchange's next stage (between steps: nothing reads a blob now)
  if (cpu_on_ && !cpu_timeout_seen_)  // a card gave up waiting for the pool (two host words per step)
    for (int c = 0; c < nc_; ++c)
      if (__atomic_load_n(&cpu_link_[c].mail->timeouts, __ATOMIC_RELAXED) != 0u && !cpu_timeout_seen_) {
        cpu_timeout_seen_ = true;
        std::fprintf(stderr, "qwen38: card %d's wait for the MoE CPU pool timed out (> %d ms); it computed those experts itself and the CPU path is off "
                             "from here (cold experts on the cards)\n", c, opt_.moe_cpu.timeout_ms);
      }
  batch_slots_.clear();
  for (int i = 0; i < ns; ++i) {
    const auto &g = segs[static_cast<size_t>(i)];
    Seq &sq = seqs_[static_cast<size_t>(g.slot)];
    sq.run_pos = sq.pos;
    sq.run_rows = g.rows;
    for (int r = 0; r < g.rows; ++r) sq.run_tokens[r] = g.tokens[r];
    sq.run_tree = g.parent != nullptr;
    for (int r = 0; r < g.rows; ++r) sq.run_parent[r] = g.parent ? g.parent[r] : r - 1;
    sq.pending = true;
    batch_slots_.push_back(g.slot);
  }
  run_slots_ = batch_slots_;
  batch_rows_ = sr;
  batch_code_ = code;
}

// ---- a decode step's rows by owner -----------------------------------------------------------------------------------
// The residual mixers are the same work on every card. Below kDecOwnerMinRows rows (or on fewer than four cards) each
// card runs both mixers of every layer on every row of the step, and every module's sum goes to every card. From
// kDecOwnerMinRows rows at four cards a row belongs to one card (kernels::owner_rows): only the owner keeps the row's
// residual streams and injection logits current and runs its mixers (dec_mixer). A mixer's output is the next module's
// input on every card, so the owner shares it (kernels::owner_share); a module's sum is read by the owner's next mixer
// only, so the partials go to the owner in one hop (the owner sum: Card::own's wires). Every card needs every row's
// streams in three places, and there the owners share them: before the PLE block (its conv ring is kept per card:
// dec_ple_combine), for a drafter feature (the streams' mean: dec_feature) and at the step's end (dec_final_mix:
// lm_head, the MTP layer and the probes read every row on every card). Each row runs the same kernels on the same
// values in the same order: the same bits.
// The mode is a function of the step's rows alone, so a step graph's key does not change, and a recorded graph
// replays every launch: below kernels::kOwnerCopyMinRows rows each is a mailbox kernel (its words are their own flags),
// from there the sums and the mixers' shares go by peer copies whose flags count in device words (the copy path,
// comm/wire.hpp: -0.7 % of a 24-row step, -1.2 % of a 32-row one, -3.1 % of a 64-row one).
namespace {
// The residual streams' mailbox carries two messages a step: before the PLE block and after the final mixer.
constexpr int kDecResParityPle = 0, kDecResParityEnd = 1;
static_assert(kDecOwnerMinRows >= k::kOwnerMinRows, "a step by owner needs a row for every card");
}  // namespace
struct Model::DecMixer {
  const bf16 *block = nullptr;              // the previous module's sum the mixer combines (null: none pending)
  const k::HcWeights *weights = nullptr;
  bf16 *out = nullptr;                      // [Head::kRows][hidden]: the next module's input
  const k::OwnerShare *share = nullptr;     // by owner: the mailbox `out` is shared through
  int parity = 0;                           // and its parity (the layer's wire parity)
};

void Model::dec_owner_init() {
  const int H = cfg_.hidden, W = hs_.shape.width();
  // one use's mailbox of `words` on every card, emptied
  auto mailboxes = [&](size_t words) {
    k::OwnerMail mail;
    for (int c = 0; c < nc_; ++c) {
      auto *inbox = static_cast<uint32_t *>(fab_.alloc_mailbox(static_cast<size_t>(c), words * sizeof(uint32_t)));
      cards_[static_cast<size_t>(c)]->mem.push_back(inbox);
      k::owner_mail_init(fab_.compute(c), inbox, words).wait();
      mail.inbox[static_cast<size_t>(c)] = inbox;
    }
    return mail;
  };
  const k::OwnerMail attn_sum = mailboxes(k::owner_sum_words(H)), moe_sum = mailboxes(k::owner_sum_words(H));
  const k::OwnerMail xa = mailboxes(k::owner_share_words(H)), xm = mailboxes(k::owner_share_words(H)), out = mailboxes(k::owner_share_words(H)),
                     feat = mailboxes(k::owner_share_words(H)), res = mailboxes(k::owner_share_words(W)),
                     sel = mailboxes(k::owner_share_words(2 * k::kQsaTopBlocks));  // sel rows as bf16 pairs
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    auto on_card = [&](k::OwnerMail mail) {
      mail.card = c;
      return mail;
    };
    auto share = [&](const k::OwnerMail &mail) { return k::OwnerShare{on_card(mail), K.pw.error, K.pw.wait_cycles}; };
    K.own.pw = K.pw;
    K.own.pw.owner = on_card(attn_sum);
    K.own.mw = K.mw;
    K.own.mw.owner = on_card(moe_sum);
    K.own.xa = share(xa), K.own.xm = share(xm), K.own.out = share(out), K.own.feat = share(feat), K.own.res = share(res), K.own.sel = share(sel);
  }
  // The copy path (kernels::OwnerFlags; steps of kOwnerCopyMinRows rows and more): per use every card's flag lines
  // (mailbox memory) and its count line, zeroed.
  auto zeroed = [&](int c, size_t bytes, bool mailbox = false) {
    bytes = std::max<size_t>(bytes, 4096);
    void *p = mailbox ? fab_.alloc_mailbox(static_cast<size_t>(c), bytes) : fab_.alloc(static_cast<size_t>(c), bytes);
    cards_[static_cast<size_t>(c)]->mem.push_back(p);
    fab_.compute(c).memset(p, 0, bytes).wait();
    return p;
  };
  auto flags = [&]() {
    std::array<k::OwnerFlags, k::kWireCardsMax> fl{};
    std::array<uint32_t *, k::kWireCardsMax> lines{};
    for (int c = 0; c < nc_; ++c) lines[static_cast<size_t>(c)] = static_cast<uint32_t *>(zeroed(c, k::kOwnerCopyLineBytes, true));
    for (int c = 0; c < nc_; ++c) {
      fl[static_cast<size_t>(c)].lines = lines;
      fl[static_cast<size_t>(c)].count = static_cast<uint32_t *>(zeroed(c, k::kOwnerCopyCountBytes));
    }
    return fl;
  };
  // The mixer outputs' shares. The buffers are named per call (dec_mixer). Writing into a peer's buffer is safe by the
  // step's order: a card's next share into xa (xm) follows its attention (MoE) owner sum of the layer, which needs every
  // peer's partials, and a peer sends those after its kernels that read xa (xm).
  for (k::OwnerShare Card::Own::*use : {&Card::Own::xa, &Card::Own::xm}) {
    const auto fl = flags();
    for (int c = 0; c < nc_; ++c) (cards_[static_cast<size_t>(c)]->own.*use).copy.flags = fl[static_cast<size_t>(c)];
  }
  // The owner sums: every card's receive buffer of the use; the MoE's also takes its rows' fp32 sums (MoeScratch::acc).
  auto sum_copy = [&]() {
    std::array<k::OwnerSumCopy, k::kWireCardsMax> cp{};
    const auto fl = flags();
    std::array<float *, k::kWireCardsMax> recv{};
    for (int c = 0; c < nc_; ++c) recv[static_cast<size_t>(c)] = static_cast<float *>(zeroed(c, k::owner_sum_copy_elems(H) * sizeof(float)));
    for (int c = 0; c < nc_; ++c) cp[static_cast<size_t>(c)].recv = recv, cp[static_cast<size_t>(c)].flags = fl[static_cast<size_t>(c)];
    return cp;
  };
  const auto attn_copy = sum_copy(), moe_copy = sum_copy();
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    K.own.pw.copy = attn_copy[static_cast<size_t>(c)];
    K.own.mw.copy = moe_copy[static_cast<size_t>(c)];
    K.msc.acc = static_cast<float *>(zeroed(c, static_cast<size_t>(k::kMoeRows) * H * sizeof(float)));
  }
}

k::HcEvents Model::dec_mixer(int c, int rows, const DecMixer &m) {
  auto &q = fab_.compute(c);
  if (!dec_by_owner(rows)) return k::hc_mixer(q, hs_.shape, rows, head_->state(c), m.block, *m.weights, head_->scratch(c), m.out);
  k::HcEvents e = k::hc_mixer_at(q, hs_.shape, k::owner_rows(c, rows), head_->state(c), m.block, *m.weights, head_->scratch(c), m.out);
  k::OwnerShare share = *m.share;
  if (share.copy.on()) {  // the copy path's buffers: every card's xa, or every card's xm
    const bool xa = m.out == cards_[static_cast<size_t>(c)]->xa;
    for (int d = 0; d < nc_; ++d) share.copy.buf[static_cast<size_t>(d)] = xa ? cards_[static_cast<size_t>(d)]->xa : cards_[static_cast<size_t>(d)]->xm;
  }
  e.up = k::owner_share(q, rows, cfg_.hidden, m.out, share, m.parity);
  return e;
}

sycl::event Model::dec_ple_combine(int c, int rows, const bf16 *block) {
  auto &q = fab_.compute(c);
  auto &st = head_->state(c);
  if (!dec_by_owner(rows)) return k::hc_combine(q, hs_.shape, rows, st, block);
  k::hc_combine_at(q, hs_.shape, k::owner_rows(c, rows), st, block);
  return k::owner_share(q, rows, hs_.shape.width(), st.residual(), cards_[static_cast<size_t>(c)]->own.res, kDecResParityPle);
}

void Model::dec_feature(int c, int rows, int feature) {
  auto &q = fab_.compute(c);
  Card &K = *cards_[static_cast<size_t>(c)];
  const bf16 *res = head_->residual(c);
  if (!dec_by_owner(rows)) {
    dflash_->capture_mean(c, q, feature, res, hs_.shape.streams, rows, dflash_->step_features(c));
    return;
  }
  // The card's own rows of the whole mean into ya (free until this layer's GDN / QSA sum), shared, then the card's
  // columns of every row (the prefill's rows by owner do the same: pf_lane_mixers). Features alternate the parity.
  const int H = cfg_.hidden;
  const k::RowRange own = k::owner_rows(c, rows);
  k::df_mean_cols(q, own.rows(), res + static_cast<size_t>(own.lo) * hs_.shape.width(), hs_.shape.streams, H, 0, K.ya + static_cast<size_t>(own.lo) * H, H, 0, H);
  k::owner_share(q, rows, H, K.ya, K.own.feat, feature);
  dflash_->capture(c, q, feature, K.ya, rows, dflash_->step_features(c));
}

void Model::dec_final_mix(int rows) {
  const auto blocks = per_card([&](int c) { return cards_[static_cast<size_t>(c)]->ym; });
  if (!dec_by_owner(rows)) {
    head_->final_mix(rows, blocks);
    return;
  }
  head_->final_mix_own(rows, blocks);
  for (int c = 0; c < nc_; ++c) {
    auto &q = fab_.compute(c);
    const Card &K = *cards_[static_cast<size_t>(c)];
    k::owner_share(q, rows, cfg_.hidden, head_->card(c).out, K.own.out, 0);
    k::owner_share(q, rows, hs_.shape.width(), head_->residual(c), K.own.res, kDecResParityEnd);
  }
}

void Model::submit_step(int rows, int seg_key, int bound) {
  const int L = cfg_.layers, per = ms_.shape.per_card();
  const int seg_rows = seg_key_code(seg_key);  // the key's rows per segment (or seg_het)
  // and its GDN accept bound; 1 with p1
  const int accept_max = seg_key_wide(seg_key) ? k::kGdnMaxRows : seg_key_p1(seg_key) ? 1 : k::kGdnRows;
  // a step of 1-row segments is final (commit takes 1..rows: 1), so the GDN
  // core commits S and the window after its row and the sequence's next step replays nothing (commit_batch: accept 0)
  const bool gdn_commit = seg_rows == 1;
  // tree rows (QSA / PLE / GDN tree instances); a replay of an accepted path (the GDN tree core)
  const bool tree = seg_key_tree(seg_key), gdn_tree = tree || seg_key_replay(seg_key);
  const bool by_owner = dec_by_owner(rows);  // the owner sums' wires (the mixers: dec_mixer)
  marks_.clear();
  for (auto &m : dec_marks_) m.clear();
  for (auto &m : dec_nodes_) m.clear();
  for (auto &v : moe_ev_) v.assign(profiling_ ? static_cast<size_t>(L) : 0, {});
  for (int c = 0; c < nc_; ++c) fab_.compute(c).memcpy(cards_[static_cast<size_t>(c)]->seg, seg_pin_, sizeof(seg_words_));  // recorded in the graph
  auto mark = [&](int c, StepProfile::Part part, const sycl::event &first, const sycl::event &last) {
    if (profiling_ && c == 0) marks_.push_back({part, first, last});
  };
  for (int c = 0; c < nc_; ++c) dec_stamp(c, -1);  // the step's start (after the records' copy)
  head_->embed(rows);
  mark(0, StepProfile::embed, head_->events(0).embed, head_->events(0).embed);
  for (int c = 0; c < nc_; ++c) dec_stamp(c, StepProfile::embed);
  for (int l = 0; l < L; ++l) {
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      auto &st = head_->state(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      const bf16 *blk = l > 0 ? K.ym : nullptr;
      if (l == ps_.layer) {
        if (blk) {  // the PLE block adds to the multi-stream state itself: combine first
          const auto e = dec_ple_combine(c, rows, blk);
          mark(c, StepProfile::hc, e, e);
          dec_stamp(c, StepProfile::hc);
          blk = nullptr;
        }
        // the rows are published by the host while the first layers run (verify_batch)
        k::PleWait pw;
        pw.want = ple_words_, pw.word = ple_words_ + 16, pw.error = ple_words_ + 32 + 16 * c;
        const auto w = k::ple_wait_rows(q, pw, rows, ps_.shape.emb, reinterpret_cast<const bf16 *>(ple_pinned_), K.ple_sc.emb);
        auto e = k::ple_decode(q, ps_.shape, rows, nullptr, ps_.w[c], K.ple_st, K.seg, K.ple_sc, st.residual(), 0, k::kSegRec, seg_rows, tree);
        e.up = w;
        mark(c, StepProfile::ple, e.up, e.conv);
        dec_stamp(c, StepProfile::ple);
      }
      const auto he = dec_mixer(c, rows, {blk, &hs_.attn[c][l], K.xa, &K.own.xa, pair_parity_});
      mark(c, StepProfile::hc, he.norm, he.up);
      dec_stamp(c, StepProfile::hc);
      if (dflash_ && df_feat_at_[static_cast<size_t>(l)] >= 0)  // a drafter feature (the mixer combined the block)
        dec_feature(c, rows, df_feat_at_[static_cast<size_t>(l)]);
      const k::PairWire *pair_wire = by_owner ? &K.own.pw : wire_of(K.pw);
      if (cfg_.full_attention[l]) {
        k::QsaStep qst{K.seg, bound, k::kSegRec, seg_rows};
        qst.tree = tree;
        qst.every = bound <= kDecEveryBound;  // the first bound bucket selects every block
        k::QsaScratch qsc = qsa_pair_region(K.qsc, qsa_ord_[static_cast<size_t>(l)]);
        qsc.stamps = dec_inner_stamps(c, kMarkQsa, k::kQsaStamped);
        // from kDecSelOwnerFrom tokens of step bound each owner scores and selects its own rows and shares their
        // selection (the same bits as every card selecting every row)
        if (by_owner && bound >= kDecSelOwnerFrom) {
          const k::RowRange own = k::owner_rows(c, rows);
          qsc.sel_lo = own.lo, qsc.sel_hi = own.hi;
          qsc.sel_share = &K.own.sel;
          qsc.sel_parity = pair_parity_;
        }
        const auto e = k::qsa_decode(q, qs_.shape_of(c), rows, K.xa, qs_.w[c][l], K.qc[static_cast<size_t>(l)], K.rope, qst, qsc, pair_wire, pair_parity_, K.ya);
        mark(c, StepProfile::qsa, e.qkv, e.reduce);
        dec_stamp(c, StepProfile::qsa);
      } else {
        k::GdnState g = K.gst[static_cast<size_t>(l)];
        g.seg_rows = seg_rows;
        g.accept_max = accept_max;
        g.commit = gdn_commit;
        g.tree = gdn_tree, g.tree_S = K.tree_S;
        const auto e = k::gdn_decode(q, gs_.shape_of(c), rows, K.xa, gs_.w[c][l], g, step_parity_, K.gsc, pair_wire, pair_parity_, K.ya);
        mark(c, StepProfile::gdn, e.qkvz, e.reduce);
        dec_stamp(c, StepProfile::gdn);
      }
    }
    if (gdn_cap_ && !cfg_.full_attention[l]) {  // calibration inputs (both cards submitted: safe to wait)
      const size_t H = static_cast<size_t>(cfg_.hidden), VWc = static_cast<size_t>(gs_.shape_of(0).val_w()), R = static_cast<size_t>(rows);
      std::vector<uint16_t> x(R * H), core(R * static_cast<size_t>(nc_) * VWc), part(R * VWc);
      fab_.compute(0).memcpy(x.data(), cards_[0]->xa, x.size() * 2).wait();
      for (int c = 0; c < nc_; ++c) {
        fab_.compute(c).memcpy(part.data(), cards_[static_cast<size_t>(c)]->gsc.core, part.size() * 2).wait();
        for (size_t r = 0; r < R; ++r) std::memcpy(&core[r * static_cast<size_t>(nc_) * VWc + static_cast<size_t>(c) * VWc], &part[r * VWc], VWc * 2);
      }
      if (!gdn_cap_(l, rows, x.data(), core.data())) gdn_cap_ = nullptr;
    }
    pair_parity_ ^= 1;
    for (int c = 0; c < nc_; ++c) {
      auto &q = fab_.compute(c);
      Card &K = *cards_[static_cast<size_t>(c)];
      const auto he = dec_mixer(c, rows, {K.ya, &hs_.mlp[c][l], K.xm, &K.own.xm, moe_parity_});
      mark(c, StepProfile::hc, he.norm, he.up);
      dec_stamp(c, StepProfile::hc);
      k::MoeWeights w = ms_.w[c][l];
      w.blobs = K.tab + static_cast<size_t>(l) * per;
      k::MoeScratch sc = K.msc;
      sc.route = K.msc.route + l;
      if (c == 0 && hits_) sc.hits = hits_ + static_cast<size_t>(l) * cfg_.experts;  // the ranking's reads
      if (cpu_on_) w.cold = K.cold + static_cast<size_t>(l) * per;  // the CPU path picks among the cold
      sc.stamps = dec_inner_stamps(c, kMarkMoe, kMoeStamped);  // the layer's split
      const auto me = k::moe_decode(q, ms_.shape_of(c), rows, K.xm, w, sc, by_owner ? &K.own.mw : wire_of(K.mw), moe_parity_, K.ym, opt_.moe_gu_ksplit, opt_.moe_dn_ksplit,
                                    cpu_on_ ? &cpu_link_[c] : nullptr);
      mark(c, StepProfile::moe, me.router, me.combine);
      dec_stamp(c, StepProfile::moe);
      if (profiling_) moe_ev_[c][static_cast<size_t>(l)] = me;
    }
    moe_parity_ ^= 1;
  }
  dec_final_mix(rows);
  mark(0, StepProfile::hc, head_->events(0).norm, head_->events(0).mix);
  for (int c = 0; c < nc_; ++c) dec_stamp(c, StepProfile::hc);
  head_->lm_pick(rows);
  mark(0, StepProfile::head, head_->events(0).lm, head_->events(0).pick);
  for (int c = 0; c < nc_; ++c) dec_stamp(c, StepProfile::head);
  step_parity_ ^= 1;
}

// ---- step graphs ----------------------------------------------------------------------------------
// One recorded graph per card per key. Everything a step changes lives in device words (the
// segment records: positions, accept counts, GDN parities, slots; token ids, the PLE rows' pinned
// buffer, the pick / wire counters), so a graph replays at any position and slot assignment. Host
// values baked in at record time are the key: rows, rows per segment, the three parities at the
// step's start, and the QSA grid bound (score work-groups, select chunks). The bound is a bucket >=
// every segment's pos + rows: the kernels read the true positions from the device and skip the
// rest, and the select is exact either way (direct or chunked), so a bucket changes no result.
// One bucket up to 65,536 tokens (the direct select's range): the extra score work-groups return
// at once, so the powers of two below it would only cost graph recordings (and their VRAM). Past
// 65,536 the bucket is a 64K multiple (chunked select: a larger bound means more work-groups).

int64_t Model::bound_bucket(int64_t need) const {
  // A first bucket at the selection's size (kDecEveryBound = 2,048 tokens = 512 blocks). A step whose rows all sit below
  // it selects every block, so its graph carries no score and no select launch (k::QsaStep::every): two launches of a
  // QSA layer's eight. The same bits (the attention never reads a selection there). Cost: every recorded shape gets
  // this bucket's graph too.
  if (need <= kDecEveryBound) return std::min<int64_t>(kDecEveryBound, cap_);
  const int64_t b = need <= 65536 ? 65536 : (need + 65535) / 65536 * 65536;  // one select chunk (16,384 blocks) per bucket past 64K
  return std::min(b, cap_);
}

double Model::vram_free_mib() const {
  double f = -1;
  for (int c = 0; c < nc_ && loader_; ++c) {
    const auto v = l0::vram_state(loader_->info(fab_.rank_index(static_cast<size_t>(c))));  // TP1 on card 1
    if (!v.ok) return -1;
    const double m = static_cast<double>(v.free) / 1048576.0;
    f = f < 0 ? m : std::min(f, m);
  }
  return f;
}

int Model::graphs_alive() const {
  return static_cast<int>((graphs_ ? graphs_->map.size() : 0) + (mtp_ ? mtp_->graphs.size() : 0) + (pf_graphs_ ? pf_graphs_->map.size() : 0));
}

int Model::graph_room(bool all) {
  constexpr double kRoom = kGraphRoomMiB;
  const double f0 = vram_free_mib();
  if (!all && (f0 < 0 || f0 >= kRoom)) return 0;
  // A destroyed graph's memory goes back to the driver's pool, which the next recording reuses, not to what sysman
  // reports free: once below the room, the card stays there. Evicting kGraphRoomEvict per new graph for good would
  // shrink the population by one per first use until every step records its graph again (a server that has seen
  // enough step shapes: decode at half speed). So the population shrinks only to its keep mark - the pool then holds
  // the difference as slack for graphs larger than the ones they replace - and is held there. A failed recording
  // (all) means the mark was too high for the pool: every graph goes and the mark drops.
  const int alive = graphs_alive();
  const bool first = graph_keep_ < 0;
  if (first && !all) graph_keep_ = std::max(kGraphKeepMin, alive * kGraphKeepNum / kGraphKeepDen);
  if (all && graph_keep_ > 0) graph_keep_ = std::max(kGraphKeepMin, std::min(graph_keep_, alive) * kGraphKeepNum / kGraphKeepDen);
  const int want = all ? alive : alive > graph_keep_ ? std::min(kGraphRoomEvict, alive - graph_keep_ + 1) : alive == graph_keep_ ? 1 : 0;
  if (want == 0) return 0;
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();  // no evicted graph may be in flight
  int n = 0;
  while (n < want) {
    uint64_t best = UINT64_MAX;
    int which = -1;
    decltype(graphs_->map.begin()) sit;
    if (graphs_)
      for (auto i = graphs_->map.begin(); i != graphs_->map.end(); ++i)
        if (i->second.used < best) best = i->second.used, which = 0, sit = i;
    decltype(mtp_->graphs.begin()) mit;
    if (mtp_)
      for (auto i = mtp_->graphs.begin(); i != mtp_->graphs.end(); ++i)
        if (i->second.used < best) best = i->second.used, which = 1, mit = i;
    decltype(pf_graphs_->map.begin()) pit;
    if (pf_graphs_)
      for (auto i = pf_graphs_->map.begin(); i != pf_graphs_->map.end(); ++i)
        if (i->second.used < best) best = i->second.used, which = 2, pit = i;
    if (which < 0) break;
    if (which == 0) graphs_->map.erase(sit);
    else if (which == 1) mtp_->graphs.erase(mit);
    else pf_graphs_->erase(pit);
    ++n;
  }
  if (first && !all)
    std::fprintf(stderr, "qwen38: graphs: below %.0f MiB of free VRAM on the tighter card (%.0f) with %d graphs: new graphs now replace the least recently used, %d kept\n",
                 kRoom, f0, alive, graph_keep_);
  if (all) std::fprintf(stderr, "qwen38: graphs: a recording failed (%.0f MiB free): all %d evicted, %d kept from here\n", f0, n, graph_keep_);
  graphs_evicted_ += n;
  return n;
}

std::vector<int64_t> Model::bound_buckets() const {
  std::vector<int64_t> v;
  for (int64_t need = 1; need <= cap_;) {
    const int64_t b = bound_bucket(need);
    v.push_back(b);
    need = b + 1;
  }
  return v;
}

// Records the step graph of the current host state (rows, rows per segment, the three parities,
// the head's HC indices) at `bound` without running it; the host state is left as it was.
void Model::record_step_graph(int rows, int seg_rows, int64_t bound) {
  namespace sx = sycl::ext::oneapi::experimental;
  if (!graphs_) graphs_ = std::make_unique<Graphs>();
  const auto key = std::make_tuple(rows, seg_rows, step_parity_, pair_parity_, moe_parity_, bound);
  {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<sx::command_graph<sx::graph_state::modifiable>> g;
    for (int c = 0; c < nc_; ++c) g.emplace_back(fab_.compute(c).get_context(), fab_.compute(c).get_device());
    const int sp = step_parity_, pp = pair_parity_, mp = moe_parity_;
    std::vector<Graphs::Entry::Hc> start_hc;
    for (int c = 0; c < nc_; ++c) {
      const auto &st = head_->state(c);
      start_hc.push_back({st.cur, st.lcur, st.lora_ld});
    }
    for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].begin_recording(fab_.compute(c));
    if (dsprof_) dec_rec_ = &g;  // the stamps count the graph's nodes (launches per module)
    try {
      submit_step(rows, seg_rows, static_cast<int>(bound));
      dec_rec_ = nullptr;
    } catch (...) {
      dec_rec_ = nullptr;
      for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].end_recording();
      throw;
    }
    for (int c = 0; c < nc_; ++c) g[static_cast<size_t>(c)].end_recording();
    Graphs::Entry e;
    for (int c = 0; c < nc_; ++c) e.smarks[c] = dec_marks_[c];
    for (int c = 0; c < nc_ && dsprof_; ++c) {  // launches per module (between counted marks), the stamps excluded
      size_t prev = 0;
      for (size_t i = 1; i < dec_marks_[c].size() && i < dec_nodes_[c].size(); ++i) {
        if (dec_nodes_[c][i] < 0) continue;
        const int p = dec_marks_[c][i].first;
        if (p >= 0) e.launches[c][p] += static_cast<int>(dec_nodes_[c][i] - dec_nodes_[c][prev] - static_cast<int64_t>(i - prev));
        prev = i;
      }
    }
    e.end_step = step_parity_;
    e.end_pair = pair_parity_;
    e.end_moe = moe_parity_;
    step_parity_ = sp;
    pair_parity_ = pp;
    moe_parity_ = mp;
    for (int c = 0; c < nc_; ++c) {
      auto &st = head_->state(c);
      e.start_hc.push_back(start_hc[static_cast<size_t>(c)]);
      e.end_hc.push_back({st.cur, st.lcur, st.lora_ld});
      st.cur = start_hc[static_cast<size_t>(c)].cur;
      st.lcur = start_hc[static_cast<size_t>(c)].lcur;
      st.lora_ld = start_hc[static_cast<size_t>(c)].lora_ld;
    }
    try {
      for (int c = 0; c < nc_; ++c) e.exec.push_back(g[static_cast<size_t>(c)].finalize());
    } catch (const std::exception &x) {  // the host state is the start's again: the caller may retry or run eagerly
      throw GraphRecordError(std::string("qwen38 model: step graph finalize: ") + x.what());
    }
    graphs_->map.emplace(key, std::move(e));
    t_.record = since_ms(t0);
    ++graphs_n_;
    graphs_ms_ += t_.record;
  }
}

void Model::run_graph(int rows, int seg_rows, int64_t need) {
  if (!graphs_) graphs_ = std::make_unique<Graphs>();
  const int64_t bound = bound_bucket(need);
  const auto key = std::make_tuple(rows, seg_rows, step_parity_, pair_parity_, moe_parity_, bound);
  auto it = graphs_->map.find(key);
  if (it == graphs_->map.end()) {
    // room first; a failed recording evicts every graph and retries, then runs eagerly
    graph_room();
    try {
      record_step_graph(rows, seg_rows, bound);
    } catch (const GraphRecordError &x) {
      graph_room(true);
      try {
        record_step_graph(rows, seg_rows, bound);
      } catch (const GraphRecordError &y) {
        if (graphs_eager_++ == 0) std::fprintf(stderr, "qwen38: %s; this step runs without a graph\n", y.what());
        submit_step(rows, seg_rows, static_cast<int>(bound));
        return;
      }
    }
    it = graphs_->map.find(key);
  }
  it->second.used = ++graph_tick_;
  for (int c = 0; c < nc_; ++c) {
    const auto &st = head_->state(c);
    const auto &h = it->second.start_hc[static_cast<size_t>(c)];
    if (st.cur != h.cur || st.lcur != h.lcur)  // lora_ld is not read at a step's start (no pending combine)
      throw std::logic_error("qwen38 model: HC state indices differ from the recorded graph's start (card " + std::to_string(c) + ": cur/lcur/ld " +
                             std::to_string(st.cur) + "/" + std::to_string(st.lcur) + "/" + std::to_string(st.lora_ld) + " vs recorded " + std::to_string(h.cur) +
                             "/" + std::to_string(h.lcur) + "/" + std::to_string(h.lora_ld) + "; rows " + std::to_string(rows) + ", step parity " +
                             std::to_string(step_parity_) + ")");
  }
  // Both cards' graphs are in flight together: the mailbox waits inside them pair up across cards.
  if (dsprof_)
    for (int c = 0; c < nc_; ++c) dec_cal_[c] = k::stamp(fab_.compute(c), dec_stamps_[c] + kDecStampWords - 2);
  for (int c = 0; c < nc_; ++c) fab_.compute(c).ext_oneapi_graph(it->second.exec[static_cast<size_t>(c)]);
  if (dsprof_) {  // this step's stamps, read by dec_finish once the step has ended
    dec_pend_.on = true;
    for (int c = 0; c < nc_; ++c) {
      dec_pend_.cal[c][0] = dec_cal_[c];
      dec_pend_.cal[c][1] = k::stamp(fab_.compute(c), dec_stamps_[c] + kDecStampWords - 1);
      dec_pend_.marks[c] = it->second.smarks[c];
      for (int p = 0; p < StepProfile::kParts; ++p) dec_pend_.launches[c][p] = it->second.launches[c][p];
    }
  }
  step_parity_ = it->second.end_step;
  pair_parity_ = it->second.end_pair;
  moe_parity_ = it->second.end_moe;
  for (int c = 0; c < nc_; ++c) {
    auto &st = head_->state(c);
    const auto &h = it->second.end_hc[static_cast<size_t>(c)];
    st.cur = h.cur;
    st.lcur = h.lcur;
    st.lora_ld = h.lora_ld;
  }
}

// The graphs of every longer QSA bound bucket, recorded ahead. For each
// step graph key recorded so far (rows, rows per segment, parities), the same key at every other
// bucket up to the capacity, recorded from that key's start state (parities and the head's HC
// indices) without running; then the drafter's pass graphs the same way. The host state is
// restored after each, so it can run between any two steps.
int Model::record_bucket_graphs(int64_t max_bound) {
  const int n0 = graphs_n_ + mtp_graphs_recorded();
  if (!opt_.graphs) return 0;
  auto buckets = bound_buckets();
  if (max_bound >= 0) buckets.erase(std::remove_if(buckets.begin(), buckets.end(), [&](int64_t b) { return b > max_bound; }), buckets.end());
  if (graphs_) {
    struct Base {
      int rows, seg_rows, sp, pp, mp;
      std::vector<Graphs::Entry::Hc> hc;
    };
    std::map<std::tuple<int, int, int, int, int>, Base> bases;
    for (const auto &[key, e] : graphs_->map) {
      const auto &[rows, sr, sp, pp, mp, bound] = key;
      (void)bound;
      bases.try_emplace(std::make_tuple(rows, sr, sp, pp, mp), Base{rows, sr, sp, pp, mp, e.start_hc});
    }
    const int sp0 = step_parity_, pp0 = pair_parity_, mp0 = moe_parity_;
    std::vector<Graphs::Entry::Hc> hc0;
    for (int c = 0; c < nc_; ++c) hc0.push_back({head_->state(c).cur, head_->state(c).lcur, head_->state(c).lora_ld});
    auto set = [&](int sp, int pp, int mp, const std::vector<Graphs::Entry::Hc> &hc) {
      step_parity_ = sp;
      pair_parity_ = pp;
      moe_parity_ = mp;
      for (int c = 0; c < nc_; ++c) {
        auto &st = head_->state(c);
        st.cur = hc[static_cast<size_t>(c)].cur;
        st.lcur = hc[static_cast<size_t>(c)].lcur;
        st.lora_ld = hc[static_cast<size_t>(c)].lora_ld;
      }
    };
    for (const auto &[bk, b] : bases)
      for (const int64_t bound : buckets) {
        if (graphs_->map.count(std::make_tuple(b.rows, b.seg_rows, b.sp, b.pp, b.mp, bound))) continue;
        set(b.sp, b.pp, b.mp, b.hc);
        record_step_graph(b.rows, b.seg_rows, bound);
      }
    set(sp0, pp0, mp0, hc0);
  }
  if (mtp_) mtp_record_buckets(buckets);
  t_.record = 0;
  return graphs_n_ + mtp_graphs_recorded() - n0;
}

void Model::dec_stamp(int c, int part) {
  if (!dsprof_ || static_cast<int>(dec_marks_[c].size()) >= kDecStampWords - 2) return;  // the last 2: the calibration
  const int at = static_cast<int>(dec_marks_[c].size());
  k::stamp(fab_.compute(static_cast<size_t>(c)), dec_stamps_[c] + at);
  dec_marks_[c].push_back({part, at});
  if (dec_rec_) {  // the graph's nodes so far (this stamp included); -1 for the MoE split's marks (pushed before theirs ran)
    const auto n = static_cast<int64_t>((*dec_rec_)[static_cast<size_t>(c)].get_nodes().size());
    while (dec_nodes_[c].size() + 1 < dec_marks_[c].size()) dec_nodes_[c].push_back(-1);
    dec_nodes_[c].push_back(n);
  }
}

// The stamp words a module writes inside itself (n marks mark0, mark0 - 1, ...; their nodes are not counted), or null
// without XEN_DEC_PROFILE or when the step's words would not hold them and the module's own mark after them.
uint64_t *Model::dec_inner_stamps(int c, int mark0, int n) {
  const int at = static_cast<int>(dec_marks_[c].size());
  if (!dsprof_ || at + n + 1 > kDecStampWords - 2) return nullptr;  // the last 2 words: the calibration
  for (int j = 0; j < n; ++j) dec_marks_[c].push_back({mark0 - j, at + j});
  return dec_stamps_[c] + at;
}

void Model::dec_finish() {
  if (!dec_pend_.on) return;
  dec_pend_.on = false;
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait();
  dec_read(dec_pend_.marks, dec_pend_.cal);
  for (int c = 0; c < nc_; ++c)
    for (int p = 0; p < StepProfile::kParts; ++p) dp_.launches[c][p] = dec_pend_.launches[c][p];
  if (dp_.steps >= 512) {  // a server prints its totals every 512 replayed steps (the probe resets per pass, before this)
    const DecProfile d = dec_profile();
    std::string line = "decode stamps (512 replayed steps, ms per step, card 0 / 1):";
    char b[96];
    for (int p = 0; p < StepProfile::kParts; ++p) {
      std::snprintf(b, sizeof b, " %s %.3f / %.3f;", StepProfile::name(p), d.ms[0][p] / d.steps, d.ms[1][p] / d.steps);
      line += b;
    }
    std::snprintf(b, sizeof b, " span %.3f / %.3f", d.span[0] / d.steps, d.span[1] / d.steps);
    std::fprintf(stderr, "%s%s\n", line.c_str(), b);
    line = "decode stamps, QSA stages (ms per step, card 0 / 1):";
    for (int st = 0; st < static_cast<int>(k::QsaStage::kCount); ++st) {
      std::snprintf(b, sizeof b, " %s %.3f / %.3f;", k::qsa_stage_name(static_cast<k::QsaStage>(st)), d.qsa[0][st] / d.steps, d.qsa[1][st] / d.steps);
      line += b;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    dec_profile_reset();
  }
}

void Model::dec_read(const std::vector<std::pair<int, int>> (&marks)[kMaxCards], const sycl::event (&cal)[kMaxCards][2]) {
  // the stamps count GPU core cycles (the clock moves with the frequency): each step is calibrated by two stamps around
  // its graph, timed by their events (profiling queue) or, without profiling, at a nominal 2.8 GHz
  for (int c = 0; c < nc_; ++c) {
    if (marks[c].size() < 2) return;
    std::vector<uint64_t> st(marks[c].size());
    uint64_t cw[2];
    fab_.compute(static_cast<size_t>(c)).memcpy(st.data(), dec_stamps_[c], st.size() * 8).wait();
    fab_.compute(static_cast<size_t>(c)).memcpy(cw, dec_stamps_[c] + kDecStampWords - 2, 16).wait();
    double ns_per = 1.0 / 2.8;
    if (fab_.profiling()) {
      auto end_of = [](const sycl::event &e) { return e.get_profiling_info<sycl::info::event_profiling::command_end>(); };
      const double ns = static_cast<double>(end_of(cal[c][1]) - end_of(cal[c][0])), ticks = static_cast<double>(cw[1] - cw[0]);
      if (ns > 0 && ticks > 0) ns_per = ns / ticks;
    }
    auto ms = [&](size_t a, size_t b) { return static_cast<double>(static_cast<int64_t>(st[b] - st[a])) * ns_per / 1e6; };
    for (size_t i = 1; i < marks[c].size(); ++i) {
      const int p = marks[c][i].first, prev = marks[c][i - 1].first;
      const double v = ms(i - 1, i);
      if (p >= 0) dp_.ms[c][p] += v;
      if (p <= kMarkMoe && p >= kMarkMoeLast) {  // the MoE layer's split: router | route | experts
        dp_.ms[c][StepProfile::moe] += v;
        dp_.moe[c][kMarkMoe - p] += v;
      } else if (p == StepProfile::moe && prev == kMarkMoeLast) {
        dp_.moe[c][kMoeStamped] += v;  // combine + all-reduce
      } else if (p <= kMarkQsa && p >= kMarkQsaLast) {  // the QSA layer's stages before o_proj
        dp_.ms[c][StepProfile::qsa] += v;
        dp_.qsa[c][kMarkQsa - p] += v;
      } else if (p == StepProfile::qsa && prev == kMarkQsaLast) {
        dp_.qsa[c][static_cast<int>(k::QsaStage::out)] += v;
      }
    }
    dp_.span[c] += ms(0, marks[c].size() - 1);
    if (c == 0) dp_.ghz += 1.0 / ns_per;  // summed per step, the mean in dec_profile()
  }
  ++dp_.steps;
}

const char *StepProfile::moe_name(int m) {
  static const char *n[] = {"router", "route", "gate_up", "down", "combine + all-reduce"};
  return m >= 0 && m < kMoeLaunches ? n[m] : "?";
}

const char *StepProfile::name(int p) {
  static const char *n[] = {"embed", "hyper-connection mixers", "PLE block", "QSA attention", "GDN", "MoE", "lm_head + pick"};
  return p >= 0 && p < kParts ? n[p] : "?";
}

StepProfile Model::profile() const {
  StepProfile p;
  auto ns = [](const sycl::event &e, bool end) {
    return end ? e.get_profiling_info<sycl::info::event_profiling::command_end>() : e.get_profiling_info<sycl::info::event_profiling::command_start>();
  };
  for (const Mark &m : marks_) p.ms[m.part] += static_cast<double>(ns(m.last, true) - ns(m.first, false)) / 1e6;
  for (int c = 0; c < nc_; ++c)
    for (const auto &e : moe_ev_[c]) {
      const sycl::event *ev[] = {&e.router, &e.route, &e.gate_up, &e.down, &e.combine};
      for (int j = 0; j < StepProfile::kMoeLaunches; ++j) p.moe_launch[c][j] += static_cast<double>(ns(*ev[j], true) - ns(*ev[j], false)) / 1e6;
    }
  for (size_t l = 0; l < moe_ev_[0].size() && l < moe_ev_[1].size(); ++l) {
    double d[kMaxCards] = {};
    for (int c = 0; c < nc_; ++c) {
      const auto &e = moe_ev_[c][l].combine;
      d[c] = static_cast<double>(ns(e, true) - ns(e, false)) / 1e6;
    }
    p.combine_ms[0].push_back(d[0]);
    p.combine_ms[1].push_back(d[1]);
    p.combine_min += std::min(d[0], d[1]);
    p.combine_skew += std::fabs(d[0] - d[1]);
  }
  p.span = marks_.empty() ? 0 : gpu_span_ms();
  return p;
}

double Model::gpu_span_ms() const {
  const auto &e = head_->events(0);
  return static_cast<double>(e.pick.get_profiling_info<sycl::info::event_profiling::command_end>() -
                             e.embed.get_profiling_info<sycl::info::event_profiling::command_start>()) /
         1e6;
}

std::vector<int32_t> Model::tokens(int rows) { return head_->tokens(0, rows); }

void Model::logits(int rows, std::vector<float> &out) {
  // pinned host buffers and a vectorised rounding pass
  const int V = head_->vocab_per_card();
  if (rows < 0 || rows > Head::kRows) throw std::invalid_argument("model logits: rows 0..kRows");
  const size_t n = static_cast<size_t>(rows) * V, per = static_cast<size_t>(Head::kRows) * V;
  if (!lg_host_) lg_host_ = static_cast<float *>(fab_.alloc_host(static_cast<size_t>(nc_) * per * 4));
  sycl::event ev[kMaxCards];
  for (int c = 0; c < nc_; ++c) ev[c] = fab_.compute(c).memcpy(lg_host_ + c * per, head_->card(c).logits, n * 4);
  out.resize(static_cast<size_t>(rows) * nc_ * V);
  for (int c = 0; c < nc_; ++c) {
    ev[c].wait();
    for (int r = 0; r < rows; ++r)
      k::bf16_round_floats(lg_host_ + c * per + static_cast<size_t>(r) * V, out.data() + static_cast<size_t>(r) * nc_ * V + static_cast<size_t>(c) * V,
                           static_cast<size_t>(V));
  }
}

int Model::logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) {
  if (rows < 1 || rows > Head::kRows) throw std::invalid_argument("model logits_cands: rows 1..kRows");
  return cands_->run(per_card([&](int c) -> const float * { return head_->card(c).logits; }), rows, T, ids, vals);
}

void Model::logprob_stats(int rows, const int32_t *targets, int top, std::vector<float> &rec, std::vector<float> &xt) {
  if (rows < 1 || rows > k::kLpMaxRows) throw std::invalid_argument("qwen38 model: logprob rows 1..8");
  const int V = head_->vocab_per_card();
  head_->wait();
  // the targets' logits from their owning cards, rounded to bf16 as the logits are reported
  float raw[k::kLpMaxRows];
  for (int r = 0; r < rows; ++r) {
    const int32_t t = targets[r];
    if (t < 0) continue;
    if (t >= nc_ * V) throw std::invalid_argument("qwen38 model: logprob target outside the vocabulary");
    const int c = t / V;
    fab_.compute(c).memcpy(&raw[r], head_->card(c).logits + static_cast<size_t>(r) * V + (t - c * V), 4);
  }
  for (int c = 0; c < nc_; ++c) fab_.compute(c).wait_and_throw();
  k::LpArgs a;
  xt.assign(static_cast<size_t>(rows), std::numeric_limits<float>::infinity());
  for (int r = 0; r < rows; ++r)
    if (targets[r] >= 0) xt[static_cast<size_t>(r)] = a.xt[r] = k::bf16_value(k::bf16_bits(raw[r]));
    else a.xt[r] = std::numeric_limits<float>::infinity();
  rec.resize(static_cast<size_t>(nc_) * rows * k::kLpRec);
  sycl::event ev[kMaxCards];
  // both cards' stats queued before the copies to pageable memory (such a copy blocks until its queue drains)
  for (int c = 0; c < nc_; ++c) k::logprob_stats(fab_.compute(c), head_->card(c).logits, V, rows, V, c * V, top, a, cards_[static_cast<size_t>(c)]->lp);
  for (int c = 0; c < nc_; ++c)
    ev[c] = fab_.compute(c).memcpy(rec.data() + static_cast<size_t>(c) * rows * k::kLpRec, cards_[static_cast<size_t>(c)]->lp, static_cast<size_t>(rows) * k::kLpRec * 4);
  for (int c = 0; c < nc_; ++c) ev[c].wait_and_throw();
}

void Model::multi_hidden(int rows, std::vector<uint16_t> &out) {
  out.resize(static_cast<size_t>(rows) * hs_.shape.width());
  fab_.compute(0).memcpy(out.data(), head_->residual(0), out.size() * 2).wait();
}

bool Model::cards_agree() {
  const size_t nr = static_cast<size_t>(Head::kRows) * hs_.shape.width(), no = static_cast<size_t>(Head::kRows) * cfg_.hidden;
  std::vector<uint16_t> r[kMaxCards], o[kMaxCards];
  for (int c = 0; c < nc_; ++c) {
    r[c].resize(nr);
    o[c].resize(no);
    fab_.compute(c).memcpy(r[c].data(), head_->residual(c), nr * 2);
    fab_.compute(c).memcpy(o[c].data(), head_->card(c).out, no * 2).wait();
  }
  bool same = true;  // TP1: one copy
  for (int c = 1; c < nc_; ++c) same &= r[c] == r[0] && o[c] == o[0];
  return same;
}

// ---- the decode MoE's CPU path for cold experts (ModelOptions::moe_cpu, kernels/moe/moe_cpu.hpp) ---------------------
void Model::setup_moe_cpu(const ModelOptions &opt) {
  const auto &o = opt.moe_cpu;
  if (o.mode == "off" || o.mode.empty()) {
    ls_.moe_cpu = "off";
    return;
  }
  auto refuse = [&](const std::string &why) { ls_.moe_cpu = "off (" + why + ")"; };
  if (o.mode != "auto" && o.mode != "avx512" && o.mode != "avx2") throw std::invalid_argument("qwen38 model: moe_cpu mode must be off, auto, avx512 or avx2");
  if (!opt.moe_tp) return refuse("needs the expert TP layout");
  if (ms_.shape.rot || ms_.shape.a8) return refuse("needs the W4A16 decode experts without rotation");
  // the CPU kernels read 64-k half blocks; TP4's 160 columns per card come in 32-k quarters (cold
  // experts then stream over PCIe as at moe_cpu off)
  if (k::moe_tp_piece(cfg_.moe_inter / nc_) != 64) return refuse("its kernels read 64-column halves; TP4's experts are in 32-column quarters");
  if (!experts_ || ls_.cold[0] == 0) return refuse("every expert is in VRAM");
  if (k::kMoeCpuRows * cfg_.topk > k::kMoeCpuMax) return refuse("top-k past the pool's request size");
  const k::MoeCpuIsa isa = k::moe_cpu_detect(o.mode);
  if (isa == k::MoeCpuIsa::none)
    return refuse(o.mode == "auto" ? "this CPU has neither AVX-512 BF16 nor AVX2 + FMA" : "this CPU has no " + o.mode);
  const int threads = std::clamp(o.threads, 1, 64);
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const auto lay = k::moe_tp_layout(cfg_.hidden, cfg_.moe_inter, nc_, nullptr);
  const uint64_t cycles = static_cast<uint64_t>(std::max(1, o.timeout_ms)) * 2800000ull;  // GPU timestamps, ~2.8 GHz
  std::string cores;
  for (int c = 0; c < nc_; ++c) {
    Card &K = *cards_[static_cast<size_t>(c)];
    auto *mail = static_cast<k::MoeCpuMail *>(fab_.alloc_host(std::max<size_t>(4096, sizeof(k::MoeCpuMail))));
    std::memset(mail, 0, 4096);
    auto *xh = static_cast<bf16 *>(fab_.alloc_host(static_cast<size_t>(k::kMoeCpuRows) * cfg_.hidden * 2));
    auto *yh = static_cast<float *>(fab_.alloc_host(static_cast<size_t>(k::kMoeCpuMax) * k::kMoeCpuRows * cfg_.hidden * 4));
    auto *state = static_cast<uint32_t *>(fab_.alloc(static_cast<size_t>(c), 4096));
    K.mem.push_back(mail), K.mem.push_back(xh), K.mem.push_back(yh), K.mem.push_back(state);
    k::moe_cpu_state_init(fab_.compute(c), state).wait();
    k::MoeCpuShape cs;
    cs.hidden = cfg_.hidden, cs.inter_c = cfg_.moe_inter / nc_, cs.card = c, cs.cards = nc_;  // TP1 = whole experts
    cs.gu_ks = opt.moe_gu_ksplit > 0 ? opt.moe_gu_ksplit : 8;
    cs.dn_ks = opt.moe_dn_ksplit > 0 ? opt.moe_dn_ksplit : 2;
    cs.gu_codes = lay.gu_codes, cs.dn_codes = lay.dn_codes, cs.gu_scales = lay.gu_scales, cs.dn_scales = lay.dn_scales;
    k::MoeCpuOptions po;
    po.isa = isa, po.threads = threads, po.pct = o.pct, po.min_rows = o.min_rows, po.timeout_ms = o.timeout_ms, po.spin_us = o.spin_us;
    po.first_core = static_cast<size_t>(c) < o.cores.size() ? o.cores[static_cast<size_t>(c)] : c * threads;
    if (po.first_core >= 0 && po.first_core + threads > static_cast<int>(hw)) po.first_core = -1;  // past the CPU's cores: unpinned
    cpu_pool_[c] = std::make_unique<k::MoeCpuPool>(cs, po, mail, reinterpret_cast<const uint16_t *>(xh), yh);
    k::MoeCpuLink &lk = cpu_link_[c];
    lk.mail = mail, lk.x = xh, lk.y = yh, lk.state = state, lk.min_rows = std::max(1, o.min_rows), lk.wait_cycles = cycles;
    lk.min_cold = std::max(1, o.min_cold);
    cores += (c ? ", " : "") + (po.first_core >= 0 ? "cores " + std::to_string(po.first_core) + "-" + std::to_string(po.first_core + threads - 1) : std::string("unpinned"));
  }
  cpu_on_ = true;
  char b[256];
  std::snprintf(b, sizeof b, "on: %s, %d threads per card (%s), a fixed %d %% of each layer's cold experts, a row's from %d of them, from %d rows, wait bound %d ms",
                k::moe_cpu_isa_name(isa), threads, cores.c_str(), std::clamp(o.pct, 0, 100), std::max(1, o.min_cold), std::max(1, o.min_rows), o.timeout_ms);
  ls_.moe_cpu = b;
}

Model::MoeCpuStats Model::moe_cpu_stats() const {
  MoeCpuStats st;
  for (int c = 0; c < nc_; ++c) {
    if (!cpu_pool_[c]) continue;
    st.requests += cpu_pool_[c]->served();
    st.experts += cpu_pool_[c]->experts();
    st.busy_us += cpu_pool_[c]->busy_us();
    st.timed_out = st.timed_out || __atomic_load_n(&cpu_link_[c].mail->timeouts, __ATOMIC_RELAXED) != 0u;
  }
  return st;
}

void Model::route_ids(int rows, std::vector<int32_t> &ids) {
  const int L = cfg_.layers, K = cfg_.topk;
  std::vector<k::MoeRoute> rec(static_cast<size_t>(L));
  fab_.compute(0).memcpy(rec.data(), cards_[0]->msc.route, rec.size() * sizeof(k::MoeRoute)).wait();
  ids.resize(static_cast<size_t>(L) * rows * K);
  for (int l = 0; l < L; ++l)
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < K; ++j) ids[(static_cast<size_t>(l) * rows + r) * K + j] = rec[static_cast<size_t>(l)].ids[r][j];
}

// ---- the live expert exchange (ModelOptions::exchange) ------------------------------------------------------------
// Every stage runs at a step boundary: the step before is done on both cards and none runs, so no kernel reads a blob
// or a table entry while it changes (a split prompt call in flight would: the stage waits for its end). The copies go
// on the copy queues and are not waited for: a stage that finds them running tries again at the next boundary.

void Model::exchange_setup(const ModelOptions &opt, const std::vector<double> *rank) {
  ls_.exchange = "off";
  auto refuse = [&](const std::string &why) {
    ls_.exchange = "off (" + why + ")";
    for (int c = 0; c < nc_; ++c) {
      for (void *p : x_spare_mem_[c]) fab_.release(p);
      x_spare_mem_[c].clear();
      x_spare_[c].clear();
    }
  };
  if (!opt.exchange) return refuse("--expert-exchange off");
  if (!opt.moe_tp || !experts_) return refuse("needs the expert TP layout");
  if (cpu_on_ || opt.moe_cpu.mode != "off") return refuse("the MoE CPU path is on");
  if (ls_.cold[0] == 0) return refuse("every expert is in VRAM");
  if (!hits_ || x_spare_[0].empty()) return refuse("no spare VRAM blobs");
  const int L = cfg_.layers, E = cfg_.experts, per = experts_->per_card();
  if (per != E) return refuse("needs every expert on every card");
  std::vector<uint8_t> hot(static_cast<size_t>(L) * E, 0);
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) {
      bool h = false;
      experts_->blob(l, e, &h, 0);
      hot[static_cast<size_t>(l) * E + e] = h ? 1 : 0;
    }
  model::PlacementOptions po;
  po.budget = opt.exchange_budget;
  po.max_swaps = static_cast<int>(x_spare_[0].size());
  po.decay = opt.exchange_decay;
  const bool prior = rank && rank->size() == hot.size();
  xpl_ = std::make_unique<model::Placement>(L, E, std::move(hot), prior ? rank : nullptr, po);
  lend_pos_.assign(static_cast<size_t>(L) * per, -1);
  for (size_t k = 0; k < lend_cand_.size(); ++k)
    lend_pos_[static_cast<size_t>(lend_cand_[k].first) * per + lend_cand_[k].second] = static_cast<int>(k);
  x_seen_.assign(static_cast<size_t>(L) * E, 0u);
  fab_.compute(0).memcpy(x_seen_.data(), hits_, x_seen_.size() * 4).wait();
  for (int c = 0; c < nc_; ++c) x_upd_[c] = static_cast<XUpd *>(fab_.alloc_host(2 * x_spare_[c].size() * sizeof(XUpd)));
  x_st_.on = true;
  char b[200];
  std::snprintf(b, sizeof b, "on: %zu spare blobs per card, a round every %d decode steps (up to %d per layer, %zu in all)", x_spare_[0].size(),
                opt.exchange_period, opt.exchange_budget, x_spare_[0].size());
  ls_.exchange = b;
}

void Model::exchange_apply(int c, const XUpd *u, int n) {
  if (n <= 0) return;
  Card &K = *cards_[static_cast<size_t>(c)];
  const uint8_t **tab = K.tab;
  uint8_t *cold = K.cold;
  fab_.compute(static_cast<size_t>(c)).parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
    const XUpd x = u[i];
    tab[x.idx] = x.ptr;
    if (cold) cold[x.idx] = static_cast<uint8_t>(x.flag);
  });
}

void Model::exchange_tick() {
  if (!xpl_) return;
  const auto t0 = std::chrono::steady_clock::now();
  ++x_step_;
  const int per = experts_->per_card();
  const size_t S = experts_->stride(), spares = x_spare_mem_[0].size();
  auto copies_done = [&] {
    for (int c = 0; c < nc_; ++c)
      if (!fab_.copy(static_cast<size_t>(c)).done()) return false;
    return true;
  };
  auto lent_at = [&](int k, int c) -> uint8_t * {
    const auto &sl = lend_.slots()[static_cast<size_t>(k)];
    return lend_buf_[c][static_cast<size_t>(sl.buf)] + sl.off;
  };
  if (xr_stage_ == 0) {
    if (x_step_ % std::max(1, opt_.exchange_period) != 0) return;
    if (pf_run_slot() >= 0) {
      ++x_st_.deferred;
      return;
    }
    // the reads since the last round (card 0's counters; the steps before are done)
    std::vector<uint32_t> now(x_seen_.size());
    fab_.compute(0).memcpy(now.data(), hits_, now.size() * 4).wait();
    if (learn_) {
      std::vector<double> delta(now.size());
      for (size_t i = 0; i < now.size(); ++i) delta[i] = static_cast<double>(now[i] - x_seen_[i]);  // uint32 wrap-safe
      xpl_->add_counts(delta);
    }
    x_seen_ = std::move(now);
    // a lent expert is in VRAM already: it does not enter
    std::vector<uint8_t> lent;
    if (lend_n_ > 0) {
      lent.assign(x_seen_.size(), 0);
      for (int k = 0; k < lend_n_; ++k) lent[static_cast<size_t>(lend_cand_[static_cast<size_t>(k)].first) * per + lend_cand_[static_cast<size_t>(k)].second] = 1;
    }
    auto sw = xpl_->round(lent.empty() ? nullptr : &lent);
    while (sw.size() > x_spare_[0].size()) xpl_->undo(sw.back()), sw.pop_back();
    if (!sw.empty()) {
      std::vector<std::vector<l0::Queue::CopyOp>> ops(static_cast<size_t>(nc_));
      xr_.clear();
      for (const auto &w : sw) {
        XSwap x{w.layer, w.hot, w.cold, {}, {}, {}};
        for (int c = 0; c < nc_; ++c) {
          x.spare[c] = x_spare_[c].back();
          x_spare_[c].pop_back();
          x.host[c] = experts_->where(w.layer, w.cold, c);
          x.vram[c] = experts_->where(w.layer, w.hot, c);
          ops[static_cast<size_t>(c)].push_back({x.spare[c], x.host[c], S});
        }
        xr_.push_back(x);
      }
      for (int c = 0; c < nc_; ++c) fab_.copy(static_cast<size_t>(c)).copy_batch(ops[static_cast<size_t>(c)], false);
      xr_stage_ = 1;
    }
  } else if (xr_stage_ == 1) {
    if (pf_run_slot() >= 0 || !copies_done()) {
      ++x_st_.deferred;
      return;
    }
    // the cold experts enter: their entries to the spares
    for (int c = 0; c < nc_; ++c) {
      XUpd *u = x_upd_[c];
      int m = 0;
      for (const XSwap &x : xr_) {
        const size_t e = static_cast<size_t>(x.layer) * per + x.cold;
        u[m++] = {x.spare[c], static_cast<uint32_t>(e), 0u};
        experts_->relocate(c, x.layer, x.cold, x.spare[c], true);
        if (!tab_host_[c].empty()) tab_host_[c][e] = x.spare[c];
        if (!cold_host_[c].empty()) cold_host_[c][e] = 0, cold_orig_[c][e] = 0;
      }
      exchange_apply(c, u, m);
    }
    // the hot experts leave: their blobs to the host slots the cold ones left (and to a lent slot the cold one held:
    // it takes the cold one's place in the lending order; on the compute queue, behind the entries that left it)
    std::vector<std::vector<l0::Queue::CopyOp>> back(static_cast<size_t>(nc_));
    for (const XSwap &x : xr_) {
      const size_t ec = static_cast<size_t>(x.layer) * per + x.cold, eh = static_cast<size_t>(x.layer) * per + x.hot;
      const int k = lend_pos_[ec];
      if (k >= 0) {
        lend_cand_[static_cast<size_t>(k)] = {x.layer, x.hot};
        lend_pos_[eh] = k;
        lend_pos_[ec] = -1;
      }
      for (int c = 0; c < nc_; ++c) {
        if (k >= 0 && k < lend_n_) fab_.compute(static_cast<size_t>(c)).memcpy(lent_at(k, c), x.vram[c], S);
        back[static_cast<size_t>(c)].push_back({x.host[c], x.vram[c], S});
      }
    }
    for (int c = 0; c < nc_; ++c) fab_.copy(static_cast<size_t>(c)).copy_batch(back[static_cast<size_t>(c)], false);
    xr_stage_ = 2;
  } else {
    if (pf_run_slot() >= 0 || !copies_done()) {
      ++x_st_.deferred;
      return;
    }
    // the hot experts' entries to their host slots (or their lent slot), their VRAM slots to the spares
    for (int c = 0; c < nc_; ++c) {
      XUpd *u = x_upd_[c] + spares;
      int m = 0;
      for (const XSwap &x : xr_) {
        const size_t e = static_cast<size_t>(x.layer) * per + x.hot;
        const int k = lend_pos_[e];
        const bool lent = k >= 0 && k < lend_n_;
        uint8_t *at = lent ? lent_at(k, c) : x.host[c];
        u[m++] = {at, static_cast<uint32_t>(e), lent ? 0u : 1u};
        experts_->relocate(c, x.layer, x.hot, x.host[c], false);
        if (!tab_host_[c].empty()) tab_host_[c][e] = at;
        if (!cold_host_[c].empty()) cold_host_[c][e] = lent ? 0 : 1, cold_orig_[c][e] = 1;
        x_spare_[c].push_back(x.vram[c]);
      }
      exchange_apply(c, u, m);
    }
    ++x_st_.rounds;
    x_st_.swaps += static_cast<int64_t>(xr_.size());
    xr_.clear();
    xr_stage_ = 0;
  }
  x_st_.host_ms += since_ms(t0);
}

// ---- the hot-expert ranking recorded while serving ---------------------------------------------------------------
void Model::take_hits(bool keep) {
  if (!hits_) return;
  const size_t n = hits_seen_.size();
  std::vector<uint32_t> now(n);
  fab_.compute(0).memcpy(now.data(), hits_, n * 4).wait();  // in order behind the steps already submitted
  if (keep && learn_) {
    std::vector<double> delta(n);
    for (size_t i = 0; i < n; ++i) delta[i] = static_cast<double>(now[i] - hits_seen_[i]);  // uint32 wrap-safe
    rec_->add_counts(delta);
  }
  hits_seen_ = std::move(now);
}

void Model::set_learning(bool on) {
  if (on && !learn_) take_hits(false);  // what the cards counted meanwhile (the warm-up) is dropped
  learn_ = on;
}

void Model::save_ranking() {
  if (!rec_) return;
  take_hits(true);
  if (!rec_->save()) std::fprintf(stderr, "qwen38: hot-expert ranking %s: %s\n", rec_->path().c_str(), rec_->status().c_str());
}

bool Model::expert_hot(int layer, int expert) const {
  if (!experts_) return true;  // calibration mode: the layer's blobs are in VRAM
  bool hot = false;
  experts_->blob(layer, expert, &hot);
  return hot;
}

void Model::calib_set_blobs(int card, const uint8_t *host, size_t bytes) {
  if (!opt_.calib) throw std::logic_error("qwen38 model: calib_set_blobs outside calibration mode");
  if (card < 0 || card > 1 || bytes > calib_stride_ * static_cast<size_t>(cfg_.experts)) throw std::invalid_argument("qwen38 model: calib_set_blobs");
  fab_.compute(card).memcpy(calib_region_[card], host, bytes).wait();
}

void Model::calib_zero_blobs() {
  if (!opt_.calib) throw std::logic_error("qwen38 model: calib_zero_blobs outside calibration mode");
  for (int c = 0; c < nc_; ++c) fab_.compute(c).memset(calib_region_[c], 0, calib_stride_ * static_cast<size_t>(cfg_.experts)).wait();
}

// ---- DFlash drafts -------------------------------------------------------------------------------
// The slot's drafts are a tree when the pass drafted one for its segment `seg` (a chain: empty).
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

// The step_seqs a pass drafts for (DFlash::pass). A lone sequence drafts the chain with its confidence cut (conf_cut,
// single-sequence passes only). Steps of two or more sequences draft without the cut, where the tree is the same
// rows' better guesses.
int Model::tree_step_seqs(int step_seqs) const { return step_seqs <= 1 && opt_.dflash_conf_cut > 0 ? arch::DFlash::kChainsOnly : step_seqs; }

std::vector<std::vector<int32_t>> Model::dflash_drafts(const std::vector<arch::DFlash::Seg> &segs, int ctx_rows, int depth, int step_seqs) {
  const int ns = static_cast<int>(segs.size());
  if (step_seqs < 1) step_seqs = ns;
  for (const auto &g : segs) note_tree(g.slot, 0, 0);  // a chain unless a tree pass below
  int block = depth > 0 ? std::min(depth, dflash_->max_depth()) + 1 : 0;
  for (const auto &g : segs)
    if (block > 0) block = static_cast<int>(std::min<int64_t>(block, cap_ - g.start));  // the block's positions < the capacity
  if (block < 2) block = 0;
  draft_ms_ = 0;
  if (ctx_rows == 0 && block == 0) return std::vector<std::vector<int32_t>>(static_cast<size_t>(ns));
  // a pass holds 8 block rows: the segments go in groups of 8 / block, each with its own context rows
  const int g = block > 0 ? std::max(1, pass_rows() / block) : std::max(1, ctx_rows > 0 ? pass_rows() / ctx_rows : ns);  // a pass holds kPassRows rows
  std::vector<std::vector<int32_t>> out;
  for (int i0 = 0; i0 < ns; i0 += g) {
    const int i1 = std::min(ns, i0 + g);
    auto part = dflash_->pass(std::vector<arch::DFlash::Seg>(segs.begin() + i0, segs.begin() + i1), ctx_rows, block, i0 * ctx_rows, tree_step_seqs(step_seqs));
    draft_ms_ += dflash_->draft_ms();
    for (int i = i0; i < i1; ++i) {
      auto &v = part[static_cast<size_t>(i - i0)];
      if (static_cast<int>(v.size()) > depth) v.resize(static_cast<size_t>(std::max(0, depth)));
      note_tree(segs[static_cast<size_t>(i)].slot, v.size(), i - i0);
    }
    for (auto &v : part) out.push_back(std::move(v));
  }
  return out;
}

std::vector<std::vector<int32_t>> Model::dflash_after_commit(const std::vector<int> &keep, const int32_t *anchor, int depth) {
  return dflash_after_commit(keep, anchor, std::vector<int>(keep.size(), depth));
}

// A depth per segment. The drafting segments (depth > 0) pass together (a block of their largest
// depth + 1 each, as many per pass as fit 8 rows), the others pass for their context only (deferred). After a verify of
// unequal segments, or when the two groups interleave, each pass's feature rows are re-laid as [segment][the group's
// largest rows]; rows past a segment's own repeat its last one: their context positions lie past what the segment
// verified, nothing reads them (a block reads the ring below its start) and the next verify rewrites them.
std::vector<std::vector<int32_t>> Model::dflash_after_commit(const std::vector<int> &keep, const int32_t *anchor, const std::vector<int> &depth) {
  if (!dflash_) throw std::logic_error("qwen38 model: no DFlash drafter loaded (ModelOptions::dflash_dir)");
  const int ns = static_cast<int>(run_slots_.size());
  if (static_cast<int>(keep.size()) != ns || static_cast<int>(depth.size()) != ns || ns == 0)
    throw std::logic_error("qwen38 model: dflash_after_commit follows verify_batch + commit_batch");
  std::vector<arch::DFlash::Seg> segs;
  std::vector<int> rows(static_cast<size_t>(ns)), row0(static_cast<size_t>(ns));
  int total = 0;
  bool same = true;  // the drafting segments' depths alike
  for (int i = 0; i < ns; ++i) {
    const Seq &sq = seqs_[static_cast<size_t>(run_slots_[static_cast<size_t>(i)])];
    if (sq.pending || keep[static_cast<size_t>(i)] < 1 || keep[static_cast<size_t>(i)] > sq.run_rows)
      throw std::logic_error("qwen38 model: dflash_after_commit keep / commit");
    segs.push_back({run_slots_[static_cast<size_t>(i)], sq.run_pos, sq.run_pos + keep[static_cast<size_t>(i)], anchor[i]});
    rows[static_cast<size_t>(i)] = sq.run_rows, row0[static_cast<size_t>(i)] = total, total += sq.run_rows;
  }
  std::vector<int> drafting, plain;
  int dmax = 0;
  for (int i = 0; i < ns; ++i) {
    if (depth[static_cast<size_t>(i)] > 0) drafting.push_back(i), dmax = std::max(dmax, depth[static_cast<size_t>(i)]);
    else plain.push_back(i);
  }
  for (int i : drafting) same &= depth[static_cast<size_t>(i)] == dmax;
  // after an equal-rows verify with one depth: the passes over all segments at that depth (a segment keeps
  // at most its own depth's drafts), unless the plain segments' own context pass saves passes
  const int per = std::max(1, pass_rows() / (dmax > 0 ? std::min(dmax, dflash_->max_depth()) + 1 : 1));
  const int nd = static_cast<int>(drafting.size());
  const bool split_wins = nd > 0 && !plain.empty() && (nd + per - 1) / per + 1 < (ns + per - 1) / per;
  if (batch_code_ >= 0 && same && !split_wins) {
    auto out = dflash_drafts(segs, batch_rows_, dmax);
    for (int i = 0; i < ns; ++i)
      if (static_cast<int>(out[static_cast<size_t>(i)].size()) > depth[static_cast<size_t>(i)]) {
        out[static_cast<size_t>(i)].resize(static_cast<size_t>(depth[static_cast<size_t>(i)]));
        auto &tp = draft_par_[static_cast<size_t>(segs[static_cast<size_t>(i)].slot)];  // a prefix of a tree is one
        if (tp.size() > out[static_cast<size_t>(i)].size() + 1) tp.resize(out[static_cast<size_t>(i)].size() + 1);
      }
    return out;
  }
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(ns));
  draft_ms_ = 0;
  for (const auto &g : segs) note_tree(g.slot, 0, 0);  // a chain unless a tree pass below
  dflash_->save_step_features(total);
  auto run = [&](const std::vector<int> &grp, int d) {
    if (grp.empty()) return;
    int R = 0, block = d > 0 ? std::min(d, dflash_->max_depth()) + 1 : 0;
    for (int i : grp) {
      R = std::max(R, rows[static_cast<size_t>(i)]);
      if (block > 0) block = static_cast<int>(std::min<int64_t>(block, cap_ - segs[static_cast<size_t>(i)].start));
    }
    if (block < 2) block = 0;
    const int g = std::max(1, std::min(block > 0 ? pass_rows() / block : ns, pass_rows() / R));
    for (size_t i0 = 0; i0 < grp.size(); i0 += static_cast<size_t>(g)) {
      const size_t i1 = std::min(grp.size(), i0 + static_cast<size_t>(g));
      int map[arch::DFlash::kRows] = {};
      std::vector<arch::DFlash::Seg> part;
      for (size_t a = i0; a < i1; ++a) {
        const int s = grp[a];
        part.push_back(segs[static_cast<size_t>(s)]);
        for (int r = 0; r < R; ++r) map[(a - i0) * static_cast<size_t>(R) + static_cast<size_t>(r)] = row0[static_cast<size_t>(s)] + std::min(r, rows[static_cast<size_t>(s)] - 1);
      }
      dflash_->stage_step_features(map, static_cast<int>(i1 - i0) * R);
      auto res = dflash_->pass(part, R, block, 0, tree_step_seqs(ns));
      draft_ms_ += dflash_->draft_ms();
      for (size_t a = i0; a < i1; ++a) {
        auto &v = out[static_cast<size_t>(grp[a])] = std::move(res[a - i0]);
        if (static_cast<int>(v.size()) > depth[static_cast<size_t>(grp[a])]) v.resize(static_cast<size_t>(depth[static_cast<size_t>(grp[a])]));
        note_tree(part[a - i0].slot, v.size(), static_cast<int>(a - i0));
      }
    }
  };
  run(drafting, dmax);
  run(plain, 0);
  return out;
}

uint32_t Model::wire_errors() {
  uint32_t e = 0;
  if (dflash_) e |= dflash_->wire_error() << 4;
  for (int c = 0; c < nc_; ++c) {
    e |= head_->wire_error(c);
    const Card &K = *cards_[static_cast<size_t>(c)];
    uint32_t w[2] = {0, 0};
    fab_.compute(c).memcpy(&w[0], K.pw.error, 4);
    fab_.compute(c).memcpy(&w[1], K.mw.error, 4).wait();
    e |= w[0] | w[1];
  }
  return e;
}

}  // namespace xenithium::arch::qwen38
