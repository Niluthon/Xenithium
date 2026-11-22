#include "xenithium/arch/qwen35/mtp.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "xenithium/kernels/drafter/dflash.hpp"
#include "xenithium/kernels/head/head.hpp"
#include "xenithium/kernels/gemm/mlp.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"
#include "xenithium/model/checkpoint.hpp"

namespace xenithium::arch::qwen35 {

namespace k = kernels;

namespace {
// pinned host ids: a prefill call's rows + a pass's rows, sized from MtpOptions::prefill_rows; this is the least
constexpr int kPinnedIdsMin = 4096;
}

Mtp::Mtp(Fabric &fab, const std::string &dir, const Config &cfg, int64_t capacity, const std::vector<const bf16 *> &rope, const MtpOptions &opt)
    : fab_(fab), cfg_(cfg), opt_(opt) {
  const int C = static_cast<int>(fab.size());
  if (C < 1 || C > 2) throw std::runtime_error("qwen35 mtp: runs on one card (TP1) or two (TP2)");
  if (cfg.mtp_layers != 1 || cfg.mtp_dedicated_embeddings) throw std::runtime_error("qwen35 mtp: one layer with the target's embeddings expected");
  if (capacity % k::kQsaPage || capacity > k::kQsaMaxTokens) throw std::runtime_error("qwen35 mtp: capacity");
  if (opt.slots < 1 || opt.slots > k::kSegs16) throw std::invalid_argument("qwen35 mtp: slots 1..16");  // 16-row steps
  if (opt.rows < 0 || opt.rows > kRows) throw std::invalid_argument("qwen35 mtp: rows 0..kRows");
  rows_ = opt.rows > 0 ? opt.rows : kRows;
  if (rope.size() != static_cast<size_t>(C)) throw std::invalid_argument("qwen35 mtp: one rope table per card");
  cap_ = capacity;
  const model::Checkpoint ck = model::Checkpoint::open(dir);
  dense_ = std::make_unique<load::DenseStore>(fab);
  std::vector<std::pair<int64_t, int64_t>> head_rows;
  if (opt.vocab > 0) {  // the draft vocabulary: ids [0, K) + 128 rows at the first eos rounded down to 64
    if (opt.vocab % 32 || cfg.eos.empty()) throw std::invalid_argument("qwen35 mtp: draft vocabulary K a multiple of 32, and an eos id");
    const int64_t e0 = *std::min_element(cfg.eos.begin(), cfg.eos.end()), s0 = e0 / 64 * 64;
    if (opt.vocab > s0 || s0 + 128 > cfg.vocab) throw std::invalid_argument("qwen35 mtp: draft vocabulary past the special tokens");
    sub_k_ = opt.vocab;
    sub_s0_ = static_cast<int>(s0);
    head_rows = {{0, opt.vocab}, {s0, s0 + 128}};
  }
  dense_->load(ck, mtp_dense_plan(ck, opt.dense_bytes, head_rows, opt.head_w4), opt.threads);
  const int H = cfg.hidden;
  ipc_ = cfg.inter / C;
  const std::string M = "mtp.", L = "mtp.layers.0.";
  cards_.resize(static_cast<size_t>(C));
  bytes_.assign(static_cast<size_t>(C), 0);
  for (int c = 0; c < C; ++c) {
    Card &cd = cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(static_cast<size_t>(c));
    auto at = [&](const std::string &n) -> const load::DeviceTensor & { return dense_->at(c, n); };
    auto vec = [&](const std::string &n) { return reinterpret_cast<const bf16 *>(at(n).ptr); };
    cd.norm_e = vec(M + "pre_fc_norm_embedding.weight");
    cd.norm_h = vec(M + "pre_fc_norm_hidden.weight");
    cd.norm_out = vec(M + "norm.weight");
    cd.in_ln = vec(L + "input_layernorm.weight");
    cd.post_ln = vec(L + "post_attention_layernorm.weight");
    cd.fc = at(M + "fc.weight").dense();
    cd.gate_up = at(L + "mlp.gate_proj.weight").dense();
    cd.down = at(L + "mlp.down_proj.weight").dense();
    k::GqaShape s{cfg.hidden, cfg.q_heads, cfg.kv_heads, C, c, static_cast<float>(cfg.rms_eps), opt.kv};
    shape_.push_back(s);
    cd.attn.qkv_d = at(L + "self_attn.q_proj.weight").dense();
    cd.attn.o_d = at(L + "self_attn.o_proj.weight").dense();
    cd.attn.q_norm = vec(L + "self_attn.q_norm.weight");
    cd.attn.k_norm = vec(L + "self_attn.k_norm.weight");
    if (cd.fc.n != H || cd.fc.k != 2 * H / C || cd.gate_up.n != 2 * ipc_ || cd.gate_up.k != H || cd.down.n != H || cd.down.k != ipc_ ||
        cd.attn.qkv_d.n != s.qkv_n() || cd.attn.qkv_d.k != H || cd.attn.o_d.n != H || cd.attn.o_d.k != s.o_k())
      throw std::runtime_error("qwen35 mtp: unexpected drafter slice shapes");
    cd.rope = rope[static_cast<size_t>(c)];
    if (sub_k_ > 0) {
      head_w_.push_back(lm_mat(at(kMtpHeadKey)));
      head_n_ = (sub_k_ + 128) / C;
      if (head_w_.back().n != head_n_ || head_w_.back().k != H) throw std::runtime_error("qwen35 mtp: draft head slice shape");
    }
    auto dev = [&](size_t b) {
      b = std::max<size_t>(b, 4096);
      void *p = fab.alloc(static_cast<size_t>(c), b);
      cd.mem.push_back(p);
      bytes_[static_cast<size_t>(c)] += b;
      return p;
    };
    // the KV: attach_kv, once the target has sized its page pool
    cd.gh = static_cast<bf16 *>(dev(static_cast<size_t>(k::kSegs16) * H * 2));
    cd.gid = static_cast<int32_t *>(dev(static_cast<size_t>(k::kSegs16) * 4));
    q.memset(cd.gh, 0, static_cast<size_t>(k::kSegs16) * H * 2);
    q.memset(cd.gid, 0, static_cast<size_t>(k::kSegs16) * 4);
    const size_t R = static_cast<size_t>(rows_), rowb = R * H * 2;
    cd.asc.y = static_cast<float *>(dev(R * s.qkv_n() * 4));
    cd.asc.q = static_cast<bf16 *>(dev(R * s.hq() * k::kQsaDim * 2));
    cd.asc.part = static_cast<float *>(dev(k::gqa_part_floats(s, rows_) * 4));
    cd.asc.attn = static_cast<bf16 *>(dev(R * s.o_k() * 2));
    cd.asc.partial = static_cast<float *>(dev(R * H * 4));
    for (bf16 **pp : {&cd.e, &cd.xin, &cd.res, &cd.xn, &cd.mix, &cd.mlp, &cd.hout[0], &cd.hout[1]}) *pp = static_cast<bf16 *>(dev(rowb));
    cd.part = static_cast<float *>(dev(R * H * 4));
    cd.gu = static_cast<float *>(dev(R * 2 * ipc_ * 4));
    cd.h = static_cast<bf16 *>(dev(R * ipc_ * 2));
    const int vpc = static_cast<int>(cfg.vocab / C);
    for (int w = 0; w < 2; ++w) {
      cd.logits[w] = static_cast<float *>(dev(R * vpc * 4));
      cd.tok[w] = static_cast<int32_t *>(dev(k::kPickMaxRows * 4));
    }
    // GEMM inputs past the live rows stay finite: zero every activation once
    for (bf16 *p : {cd.e, cd.xin, cd.res, cd.xn, cd.mix, cd.mlp, cd.hout[0], cd.hout[1]}) q.memset(p, 0, rowb);
    q.memset(cd.asc.attn, 0, R * s.o_k() * 2);
    q.memset(cd.h, 0, R * ipc_ * 2);
    if (C == 1) {  // TP1: fc's input [norm(e) | norm(h)] as one [rows][2 hidden] matrix (a pass, a prefill chunk, a flush)
      cat_rows_ = std::max({static_cast<int>(R), opt.prefill_rows, opt.kv_defer});
      const size_t cb = static_cast<size_t>(cat_rows_) * 2 * H * 2;
      cd.cat = static_cast<bf16 *>(dev(cb));
      q.memset(cd.cat, 0, cb);
    } else {
      cd.wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(k::pair_wire_words(H) * 4));
      cd.mem.push_back(cd.wire.inbox);
    }
    cd.wire.error = static_cast<uint32_t *>(dev(64));
    q.memset(cd.wire.error, 0, 64);
    cd.noise = static_cast<k::DraftNoiseRec *>(dev(static_cast<size_t>(opt.slots) * sizeof(k::DraftNoiseRec)));
    q.memset(cd.noise, 0, static_cast<size_t>(opt.slots) * sizeof(k::DraftNoiseRec));
    if (opt.kv_defer > 0) {  // the deferred KV rows, the flush's scratch and many-row wire
      if (opt.kv_defer < rows_ || opt.kv_defer % 4) throw std::invalid_argument("qwen35 mtp: kv_defer 0 or a multiple of 4 >= 8");
      const size_t B = static_cast<size_t>(opt.kv_defer), NS = static_cast<size_t>(opt.slots);
      cd.bl_h = static_cast<bf16 *>(dev(NS * B * H * 2));
      cd.bl_ids = static_cast<int32_t *>(dev(NS * B * 4));
      q.memset(cd.bl_h, 0, NS * B * H * 2);
      q.memset(cd.bl_ids, 0, NS * B * 4);
      cd.fl_a = static_cast<bf16 *>(dev(B * H * 2));
      cd.fl_b = static_cast<bf16 *>(dev(B * H * 2));
      cd.fl_part = static_cast<float *>(dev(B * H * 4));
      cd.fl_pos4 = static_cast<int32_t *>(dev(B));
      cd.fl_sc.y = static_cast<float *>(dev(B * s.qkv_n() * 4));
      cd.fl_sc.q = static_cast<bf16 *>(dev(B * s.hq() * k::kQsaDim * 2));
      if (C > 1) {
        cd.fl_wire.rows_max = opt.kv_defer;
        cd.fl_wire.n_max = H;
        cd.fl_wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(k::pair_rows_wire_words(opt.kv_defer, H) * 4));
        cd.mem.push_back(cd.fl_wire.inbox);
        cd.fl_wire.error = cd.wire.error;
      }
    }
    q.wait();
  }
  for (int c = 0; c < C && C > 1; ++c) {
    Card &cd = cards_[static_cast<size_t>(c)];
    cd.wire.peer_inbox = cards_[static_cast<size_t>(1 - c)].wire.inbox;
    k::pair_wire_init(fab.compute(static_cast<size_t>(c)), H, cd.wire.inbox).wait();
    if (cd.fl_wire.inbox) {
      cd.fl_wire.peer_inbox = cards_[static_cast<size_t>(1 - c)].fl_wire.inbox;
      k::pair_rows_wire_init(fab.compute(static_cast<size_t>(c)), opt.kv_defer, H, cd.fl_wire.inbox).wait();
    }
  }
  if (opt.kv_defer > 0) pend_.resize(static_cast<size_t>(opt.slots));
  pinned_n_ = std::max({kPinnedIdsMin, opt.prefill_rows + rows_, opt.kv_defer + rows_});
  ids_host_ = static_cast<int32_t *>(fab.alloc_host(static_cast<size_t>(pinned_n_) * 4));
  std::memset(ids_host_, 0, static_cast<size_t>(pinned_n_) * 4);
}

Mtp::~Mtp() {
  for (size_t c = 0; c < fab_.size(); ++c) fab_.compute(c).wait();
  for (auto &cd : cards_) {
    for (void *p : cd.mem) fab_.release(p);
    for (const auto &kb : cd.kv_mem) fab_.release(kb.first);
  }
  if (ids_host_) fab_.release(ids_host_);
}

size_t Mtp::card_bytes(int c) const {
  size_t b = bytes_[static_cast<size_t>(c)];
  for (const auto &kv : dense_->stats().card_bytes[static_cast<size_t>(c)]) b += kv.second;
  return b;
}

size_t Mtp::kv_page_bytes(int c) const {
  const auto &s = shape_[static_cast<size_t>(c)];
  return k::gqa_page_total_bytes(s);
}

std::vector<k::KvPageBuf> Mtp::kv_page_bufs(int c) const {
  const auto &s = shape_[static_cast<size_t>(c)];
  const auto &kv = cards_[static_cast<size_t>(c)].kv;
  std::vector<k::KvPageBuf> b{{kv.k, k::gqa_page_kv_bytes(s)}, {kv.v, k::gqa_page_v_bytes(s)}};
  if (k::qsa_kv_scaled(s.kv)) {
    b.push_back({kv.ks, k::gqa_page_scale_floats(s) * 4});
    b.push_back({kv.vs, k::gqa_page_vscale_floats(s) * 4});
  }
  return b;
}

void Mtp::attach_kv(int pool_pages, const std::vector<const int32_t *> &table, int slot_pages) {
  if (table.size() != cards_.size() || pool_pages < 1 || slot_pages * static_cast<int64_t>(k::kQsaPage) != cap_)
    throw std::invalid_argument("qwen35 mtp: attach_kv takes one table per card and the target's slot pages");
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    const auto &s = shape_[c];
    fab_.compute(c).wait();
    for (const auto &[p, b] : cd.kv_mem) {  // a re-attach (the target's pool re-sized): the old KV goes
      fab_.release(p);
      bytes_[c] -= b;
    }
    cd.kv_mem.clear();
    cd.kv = kernels::GqaCache{};
    auto dev = [&](size_t b) {
      b = std::max<size_t>(b, 4096);
      void *p = fab_.alloc(c, b);
      cd.kv_mem.push_back({p, b});
      bytes_[c] += b;
      return p;
    };
    const size_t P = static_cast<size_t>(pool_pages);
    cd.kv.k = static_cast<uint8_t *>(dev(P * k::gqa_page_kv_bytes(s)));
    cd.kv.v = static_cast<uint8_t *>(dev(P * k::gqa_page_v_bytes(s)));
    if (k::qsa_kv_scaled(s.kv)) {
      cd.kv.ks = static_cast<float *>(dev(P * k::gqa_page_scale_floats(s) * 4));
      cd.kv.vs = static_cast<float *>(dev(P * k::gqa_page_vscale_floats(s) * 4));
    }
    cd.kv.pages = table[c];  // the target's table: the drafter's tokens sit on the same pages as the target's
    cd.kv.max_tokens = static_cast<int>(cap_);
    cd.kv.slot_pages = slot_pages;
  }
}

void Mtp::gather_chain(int which, const std::vector<int> &src) {
  const int H = cfg_.hidden;
  if (src.empty() || src.size() > static_cast<size_t>(k::kSegs16)) throw std::invalid_argument("qwen35 mtp: chain rows 1..16");
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    for (size_t i = 0; i < src.size(); ++i) {
      if (src[i] < 0 || src[i] >= rows_) throw std::invalid_argument("qwen35 mtp: chain source row");
      q.memcpy(cd.gh + i * H, cd.hout[which] + static_cast<size_t>(src[i]) * H, static_cast<size_t>(H) * 2);
      q.memcpy(cd.gid + i, cd.tok[which] + src[i], 4);
    }
  }
}

void Mtp::set_noise(int slot, uint64_t key, uint64_t pos, float inv_t) {
  if (slot < 0 || slot >= opt_.slots) return;
  if (noise_host_.size() != static_cast<size_t>(opt_.slots)) noise_host_.assign(static_cast<size_t>(opt_.slots), k::DraftNoiseRec{});
  auto &r = noise_host_[static_cast<size_t>(slot)];
  r.key = key, r.pos = pos, r.inv_t = inv_t;
}

void Mtp::noise_upload() {
  bool any = false;
  for (const auto &r : noise_host_) any = any || r.inv_t > 0.0f;
  if (any)
    for (size_t c = 0; c < cards_.size(); ++c)
      fab_.compute(c).memcpy(cards_[c].noise, noise_host_.data(), noise_host_.size() * sizeof(k::DraftNoiseRec));
  noise_any_ = any;
}

void Mtp::pass(Head &head, int rows, const std::vector<const bf16 *> &h, const std::vector<const int32_t *> &ids,
               const std::vector<const int32_t *> &rec, int seg_rows, int pos_bound, int which, bool kv_only, int draft) {
  if (rows < 1 || rows > rows_ || which < 0 || which > 1) throw std::invalid_argument("qwen35 mtp: pass rows 1..MtpOptions::rows, output 0 / 1");
  if (pos_bound > cap_) throw std::invalid_argument("qwen35 mtp: pass past the drafter KV");
  const size_t C = cards_.size();
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.rms_eps);
  const int p_fc = parity_, p_attn = parity_ ^ 1, p_mlp = parity_;
  parity_ ^= 1;  // three reduces per pass: fc (p), attention (p ^ 1), MLP (p); the next pass starts at p ^ 1
  for (size_t c = 0; c < C; ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    const k::PairWire *wire = C > 1 ? &cd.wire : nullptr;
    if (C == 1) {  // TP1: both halves on the card, fc over the whole k (one fp32 sum, rounded once as the pair's)
      head.gather_on(q, 0, ids[0], rows, cd.e);
      k::rms_norm(q, rows, H, cd.e, nullptr, nullptr, cd.norm_e, eps, cd.xin);
      k::rms_norm(q, rows, H, const_cast<bf16 *>(h[c]), nullptr, nullptr, cd.norm_h, eps, cd.xn);
      concat(q, rows, cd.xin, cd.xn);
      k::dense_gemm(q, rows, cd.cat, cd.fc, cd.part);
    } else {
      if (c == 0) {  // the embedding half of fc
        head.gather_on(q, 0, ids[0], rows, cd.e);
        k::rms_norm(q, rows, H, cd.e, nullptr, nullptr, cd.norm_e, eps, cd.xin);
      } else {  // the hidden half (rms_norm does not write res without an add)
        k::rms_norm(q, rows, H, const_cast<bf16 *>(h[c]), nullptr, nullptr, cd.norm_h, eps, cd.xin);
      }
      k::dense_gemm(q, rows, cd.xin, cd.fc, cd.part);
    }
    k::pair_reduce(q, rows, H, cd.part, wire, p_fc, cd.res);
    k::rms_norm(q, rows, H, cd.res, nullptr, nullptr, cd.in_ln, eps, cd.xn);
    // a KV-only pass stops after the K / V rows (no attention, o or two-card sum: its output is unused)
    k::GqaStep st{rec[c], pos_bound, k::kSegRec, seg_rows};
    st.kv_only = kv_only;
    k::gqa_decode(q, shape_[c], rows, cd.xn, cd.attn, cd.kv, cd.rope, st, cd.asc, wire, p_attn, kv_only ? nullptr : cd.mix);
    if (kv_only) continue;
    k::rms_norm(q, rows, H, cd.res, cd.mix, nullptr, cd.post_ln, eps, cd.xn);
    k::dense_gemm(q, rows, cd.xn, cd.gate_up, cd.gu);
    k::swiglu(q, rows, ipc_, cd.gu, cd.h);
    k::dense_gemm(q, rows, cd.h, cd.down, cd.part);
    k::pair_reduce(q, rows, H, cd.part, wire, p_mlp, cd.mlp);
    k::rms_norm(q, rows, H, cd.res, cd.mlp, nullptr, cd.norm_out, eps, cd.hout[which]);
  }
  if (kv_only) return;  // one sum (fc p): the next pass starts at p ^ 1
  std::vector<const bf16 *> x(C);
  std::vector<float *> lg(C);
  std::vector<int32_t *> tk(C);
  for (size_t c = 0; c < C; ++c) x[c] = cards_[c].hout[which], lg[c] = cards_[c].logits[which], tk[c] = cards_[c].tok[which];
  // coupled drafts (a sampled request's rows: logit / T + the target's Gumbel noise) before the pick
  auto noisy = [&](size_t c, sycl::queue &q, float *l, int n, int sub_k, int sub_s0) {
    if (noise_any_)
      k::draft_gumbel(q, l, n, rows, n, static_cast<int>(c) * n, sub_k, sub_s0, rec[c], k::kSegRec, seg_rows, -1, cards_[c].noise, draft);
  };
  if (head_w_.empty()) {
    head.lm_pick_to(rows, x, lg, tk, [&](size_t c, sycl::queue &q, float *l, int n) { noisy(c, q, l, n, 0, 0); });
    return;
  }
  // the draft vocabulary: its own rows, the pick over them (virtual indices), then virtual -> token id
  const int n = head_n_, kk = sub_k_, s0 = sub_s0_;
  for (size_t c = 0; c < C; ++c) lm_gemm(fab_.compute(c), rows, x[c], head_w_[c], lg[c]);
  for (size_t c = 0; c < C; ++c) {
    auto &q = fab_.compute(c);
    noisy(c, q, lg[c], n, kk, s0);
    k::greedy_pick(q, lg[c], n, rows, n, static_cast<int>(c) * n, head.card(static_cast<int>(c)).scratch, C > 1 ? &head.card(static_cast<int>(c)).wire : nullptr,
                   tk[c]);
    int32_t *t = tk[c];
    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> r) {
      const int32_t v = t[r];
      t[r] = v < kk ? v : v - kk + s0;
    });
  }
}

void Mtp::prefill_kv(Head &head, int m, const std::vector<const bf16 *> &h, const int32_t *ids, const std::vector<const int32_t *> &pos4, int pos0,
                     const std::vector<Lend> &lend, int parity, int slot, const std::vector<const bf16 *> *rope_rows) {
  if (slot < 0 || slot >= opt_.slots) throw std::invalid_argument("qwen35 mtp: no such slot");
  if (m < 1 || m > pinned_n_ - rows_) throw std::invalid_argument("qwen35 mtp: prefill rows past MtpOptions::prefill_rows");
  if (pos0 + m > cap_) throw std::invalid_argument("qwen35 mtp: prefill past the drafter KV");
  const size_t C = cards_.size();
  const int H = cfg_.hidden;
  const float eps = static_cast<float>(cfg_.rms_eps);
  fab_.compute(0).wait();  // the pinned ids of an earlier gather are free
  std::memcpy(ids_host_, ids, static_cast<size_t>(m) * 4);
  for (size_t c = 0; c < C; ++c) {
    Card &cd = cards_[c];
    const Lend &b = lend[c];
    auto &q = fab_.compute(c);
    if (C == 1) {  // TP1: [norm(e) | norm(h)] into the card's concat rows, fc over the whole k
      if (m > cat_rows_) throw std::invalid_argument("qwen35 mtp: prefill rows past MtpOptions::prefill_rows");
      head.gather_on(q, 0, ids_host_, m, b.a);
      k::rms_norm(q, m, H, b.a, nullptr, nullptr, cd.norm_e, eps, b.b);
      k::rms_norm(q, m, H, const_cast<bf16 *>(h[c]), nullptr, nullptr, cd.norm_h, eps, b.a);
      concat(q, m, b.b, b.a);
      k::pf_dense(q, m, cd.fc, cd.cat, 2 * H, k::PfOut{b.part, nullptr, H});
    } else {
      if (c == 0) {
        head.gather_on(q, 0, ids_host_, m, b.a);
        k::rms_norm(q, m, H, b.a, nullptr, nullptr, cd.norm_e, eps, b.b);
      } else {
        k::rms_norm(q, m, H, const_cast<bf16 *>(h[c]), nullptr, nullptr, cd.norm_h, eps, b.b);
      }
      k::pf_dense(q, m, cd.fc, b.b, H, k::PfOut{b.part, nullptr, H});
    }
    k::pair_reduce_rows(q, m, H, b.part, b.wire, parity, b.a);
    k::rms_norm(q, m, H, b.a, nullptr, nullptr, cd.in_ln, eps, b.b);
    k::GqaCache kv = cd.kv;  // the slot's pages, as one sequence
    kv.pages += static_cast<size_t>(slot) * kv.slot_pages;
    kv.slot_pages = 0;
    k::GqaShape sh = shape_[c];
    if (rope_rows) sh.rope_rows = (*rope_rows)[c];  // the target's rope rows of the call (M-RoPE)
    k::gqa_kv_rows(q, sh, m, b.b, cd.attn, kv, cd.rope, pos4[c], pos0, b.sc);
  }
}

// ---- deferred KV-only passes (the same design as DFlash's deferred context) ----
namespace {
class MtpIdsStashK;
class MtpPos4K;
}  // namespace

void Mtp::defer(Head &head, int ns, int seg_rows, const int *slot, const int64_t *p0, const bf16 *h1, const int32_t *ids, const int32_t *rope) {
  const int B = opt_.kv_defer;
  if (B <= 0) throw std::logic_error("qwen35 mtp: defer without kv_defer");
  if (ns < 1 || ns > k::kSegs16 || seg_rows < 1 || seg_rows > B) throw std::invalid_argument("qwen35 mtp: deferred rows");
  int64_t dst[k::kMaxSegs] = {};  // (df_ctx_stash's row table)
  for (int i = 0; i < ns; ++i) {
    const int s = slot[i];
    if (s < 0 || s >= opt_.slots || p0[i] < 0 || p0[i] + seg_rows > cap_) throw std::invalid_argument("qwen35 mtp: deferred rows' slot / position");
    Pend &p = pend_[static_cast<size_t>(s)];
    if (p.n > 0 && (p0[i] < p.p0 || p0[i] > p.p0 + p.n)) settle(head, s, p0[i]);  // not a continuation
    if (p.n == 0) p.p0 = p0[i];
    p.rope = rope ? rope[i] : 0;  // constant over a sequence's decode rows
    int keep = static_cast<int>(p0[i] - p.p0);
    if (keep + seg_rows > B) {  // full: the rows before these first
      settle(head, s, p0[i]);
      p.p0 = p0[i];
      keep = 0;
    }
    dst[i] = static_cast<int64_t>(s) * B + keep;
    p.n = keep + seg_rows;
  }
  // card 1 keeps the hidden rows (its fc half), card 0 the tokens (its embedding half); TP1: card 0 both
  const size_t hc = cards_.size() - 1;
  k::df_ctx_stash(fab_.compute(hc), ns, seg_rows, cfg_.hidden, h1, cards_[hc].bl_h, dst);
  struct Rows {
    int64_t r[k::kSegs16];
  } dr;
  for (int i = 0; i < k::kSegs16; ++i) dr.r[i] = dst[i];
  int32_t *bl = cards_[0].bl_ids;
  const int sr = seg_rows;
  fab_.compute(0).parallel_for<MtpIdsStashK>(sycl::range<1>(static_cast<size_t>(ns * sr)), [=](sycl::id<1> id) {
    const int row = static_cast<int>(id[0]);
    bl[dr.r[row / sr] + row % sr] = ids[row];
  });
}

void Mtp::settle(Head &head, int slot, int64_t pos) {
  if (pend_.empty() || slot < 0 || slot >= opt_.slots) return;
  Pend &p = pend_[static_cast<size_t>(slot)];
  if (p.n == 0) return;
  const int64_t keep = std::clamp<int64_t>(pos - p.p0, 0, p.n);
  if (keep > 0) flush(head, slot, static_cast<int>(keep));
  p.n = 0;
}

void Mtp::drop(int slot) {
  if (!pend_.empty() && slot >= 0 && slot < opt_.slots) pend_[static_cast<size_t>(slot)].n = 0;
}

// prefill_kv's kernels over the slot's first m deferred rows at p0 .. p0 + m - 1, on the flush's own buffers
void Mtp::flush(Head &head, int slot, int m) {
  const int H = cfg_.hidden, B = opt_.kv_defer;
  const float eps = static_cast<float>(cfg_.rms_eps);
  const int par = fl_par_;
  fl_par_ ^= 1;
  const int64_t p0 = pend_[static_cast<size_t>(slot)].p0;
  for (size_t c = 0; c < cards_.size(); ++c) {
    Card &cd = cards_[c];
    auto &q = fab_.compute(c);
    int32_t *w = cd.fl_pos4;
    const int words = (m + 3) / 4, base = static_cast<int>(p0);
    q.parallel_for<MtpPos4K>(sycl::range<1>(static_cast<size_t>(words)), [=](sycl::id<1> id) { w[id[0]] = base + 4 * static_cast<int>(id[0]); });
    if (cards_.size() == 1) {  // TP1: both halves here, fc over the whole k
      head.gather_on(q, 0, cd.bl_ids + static_cast<size_t>(slot) * B, m, cd.fl_a);
      k::rms_norm(q, m, H, cd.fl_a, nullptr, nullptr, cd.norm_e, eps, cd.fl_b);
      k::rms_norm(q, m, H, cd.bl_h + static_cast<size_t>(slot) * B * H, nullptr, nullptr, cd.norm_h, eps, cd.fl_a);
      concat(q, m, cd.fl_b, cd.fl_a);
      k::pf_dense(q, m, cd.fc, cd.cat, 2 * H, k::PfOut{cd.fl_part, nullptr, H});
    } else {
      if (c == 0) {
        head.gather_on(q, 0, cd.bl_ids + static_cast<size_t>(slot) * B, m, cd.fl_a);
        k::rms_norm(q, m, H, cd.fl_a, nullptr, nullptr, cd.norm_e, eps, cd.fl_b);
      } else {
        k::rms_norm(q, m, H, cd.bl_h + static_cast<size_t>(slot) * B * H, nullptr, nullptr, cd.norm_h, eps, cd.fl_b);
      }
      k::pf_dense(q, m, cd.fc, cd.fl_b, H, k::PfOut{cd.fl_part, nullptr, H});
    }
    k::pair_reduce_rows(q, m, H, cd.fl_part, cards_.size() > 1 ? &cd.fl_wire : nullptr, par, cd.fl_a);
    k::rms_norm(q, m, H, cd.fl_a, nullptr, nullptr, cd.in_ln, eps, cd.fl_b);
    k::GqaCache kv = cd.kv;  // the slot's pages, as one sequence
    kv.pages += static_cast<size_t>(slot) * kv.slot_pages;
    kv.slot_pages = 0;
    k::GqaShape sh = shape_[c];
    sh.rope_delta = pend_[static_cast<size_t>(slot)].rope;  // the rows' M-RoPE delta (sh.mrope)
    k::gqa_kv_rows(q, sh, m, cd.fl_b, cd.attn, kv, cd.rope, w, base, cd.fl_sc);
  }
  ++flushes_;
}

std::vector<int32_t> Mtp::read_tokens(int which, int rows) {
  std::vector<int32_t> t(static_cast<size_t>(rows));
  fab_.compute(0).memcpy(t.data(), cards_[0].tok[which], static_cast<size_t>(rows) * 4).wait();
  return t;
}

std::vector<uint16_t> Mtp::read_hidden(int which, int rows) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * cfg_.hidden);
  fab_.compute(0).memcpy(v.data(), cards_[0].hout[which], v.size() * 2).wait();
  return v;
}

// TP1: rows [rows][hidden] of e (the embedding half) and hn (the hidden half) side by side into card 0's cat rows. One
// kernel, 8 bf16 per item.
namespace {
class MtpConcatK;
}
void Mtp::concat(sycl::queue &q, int rows, const bf16 *e, const bf16 *hn) {
  const size_t H8 = static_cast<size_t>(cfg_.hidden) / 8;
  using V = sycl::vec<uint16_t, 8>;
  const V *ve = reinterpret_cast<const V *>(e), *vh = reinterpret_cast<const V *>(hn);
  V *vc = reinterpret_cast<V *>(cards_[0].cat);
  q.parallel_for<MtpConcatK>(sycl::range<1>(static_cast<size_t>(rows) * 2 * H8), [=](sycl::id<1> id) {
    const size_t i = id[0], r = i / (2 * H8), c = i % (2 * H8);
    vc[i] = c < H8 ? ve[r * H8 + c] : vh[r * H8 + c - H8];
  });
}

uint32_t Mtp::wire_error() {
  uint32_t e = 0;
  for (size_t c = 0; c < cards_.size(); ++c) {
    uint32_t w = 0;
    fab_.compute(c).memcpy(&w, cards_[c].wire.error, 4).wait();
    e |= w;
  }
  return e;
}

}  // namespace xenithium::arch::qwen35
