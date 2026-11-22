#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/util/env.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "xenithium/kernels/comm/wire.hpp"

namespace xenithium::arch::qwen38 {

namespace {
const std::string kP = "model.language_model.";
}

Head::Head(Fabric &fab, const load::DenseStore &dense, const Config &cfg) : fab_(fab) {
  hc_ = kernels::HcShape{cfg.hc, cfg.hidden, cfg.hc_lowrank, static_cast<float>(cfg.rms_eps)};
  const int C = static_cast<int>(fab.size());
  if (C != 1 && C != 2 && C != 4) throw std::runtime_error("qwen38 head: the pick wire runs on 1, 2 or 4 cards");  // 4 cards: two hops
  if (cfg.vocab % (16 * C)) throw std::runtime_error("qwen38 head: vocab does not split into 16-row tiles per card");
  vpc_ = static_cast<int>(cfg.vocab / C);
  const int W = hc_.width(), R = hc_.lowrank, H = hc_.hidden;
  lora_ld_ = std::max(R, (R + hc_.streams + 15) / 16 * 16);  // room for a layer mixer's down|inject output
  cards_.resize(static_cast<size_t>(C));
  ev_.resize(static_cast<size_t>(C));
  host_pick_ = C > 1 && (util::env_flag("XEN_Q38_HOST_PICK") || (C == 4 && util::env_flag("XEN_Q38_COPY_WIRE")));
  for (int c = 0; c < C; ++c) {
    auto get = [&](const std::string &n) -> const load::DeviceTensor & { return dense.at(c, n); };
    embed_.push_back(reinterpret_cast<const bf16 *>(get(kP + "embed_tokens.weight").ptr));
    kernels::HcWeights m;
    m.norm = reinterpret_cast<const bf16 *>(get(kP + "hyper_connection_mixer.hc_norm.weight").ptr);
    m.down = get(kP + "hyper_connection_mixer.input_mix_weight_down.weight").dense();
    m.up = get(kP + "hyper_connection_mixer.input_mix_weight_up.weight").dense();
    mix_.push_back(m);
    lm_.push_back(get("lm_head.weight").dense());
    if (lm_.back().n != vpc_ || m.down.n != R || m.up.n != W) throw std::runtime_error("qwen38 head: unexpected weight shapes");

    Card &k = cards_[static_cast<size_t>(c)];
    auto dev = [&](size_t bytes) { return fab.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096)); };
    for (int i = 0; i < 2; ++i) {
      k.st.res[i] = static_cast<bf16 *>(dev(static_cast<size_t>(kRows) * W * 2));
      k.st.lora[i] = static_cast<float *>(dev(static_cast<size_t>(kRows) * lora_ld_ * 4));
    }
    k.xn = static_cast<bf16 *>(dev(static_cast<size_t>(kRows) * W * 2));
    k.s = static_cast<bf16 *>(dev(static_cast<size_t>(kRows) * R * 2));
    k.gate = static_cast<float *>(dev(static_cast<size_t>(kRows) * W * 4));
    k.out = static_cast<bf16 *>(dev(static_cast<size_t>(kRows) * H * 2));
    k.logits = static_cast<float *>(dev(static_cast<size_t>(kRows) * vpc_ * 4));
    k.tokens = static_cast<int32_t *>(dev(kernels::kPickMaxRows * 4));
    k.scratch = dev(kernels::pick_scratch_bytes(vpc_));
    if (host_pick_) k.local_keys = static_cast<uint64_t *>(fab.alloc_host(kernels::kPickMaxRows * sizeof(uint64_t)));
    if (C > 1) {
      k.wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(4096));
      k.wire.step = static_cast<int32_t *>(dev(4));
      k.wire.error = static_cast<uint32_t *>(dev(4));
    }
  }
  ids_host_ = static_cast<int32_t *>(fab.alloc_host(4096));
  std::memset(ids_host_, 0, 4096);
  // Dead rows of the GEMM inputs must hold finite values: zero them once (rows < kRows reuse
  // whatever the last full step left, which is finite).
  for (int c = 0; c < C; ++c) {
    Card &k = cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(static_cast<size_t>(c));
    for (int i = 0; i < 2; ++i) {
      q.memset(k.st.res[i], 0, static_cast<size_t>(kRows) * W * 2);
      q.memset(k.st.lora[i], 0, static_cast<size_t>(kRows) * lora_ld_ * 4);
    }
    q.memset(k.xn, 0, static_cast<size_t>(kRows) * W * 2);
    q.memset(k.s, 0, static_cast<size_t>(kRows) * R * 2);
    q.memset(k.out, 0, static_cast<size_t>(kRows) * H * 2);
    if (C > 1) {
      k.wire.peer_inbox = cards_[static_cast<size_t>(c ^ 1)].wire.inbox;
      if (C == 4) k.wire.peer2_inbox = cards_[static_cast<size_t>(c ^ 2)].wire.inbox;  // the second hop
      static_assert(kernels::kPickWireWords4 * 4 <= 4096, "the pick inbox holds both hops' regions");
      std::vector<uint32_t> empty(1024, kernels::kMailEmpty);
      q.memcpy(k.wire.inbox, empty.data(), 4096);
      q.memset(k.wire.step, 0, 4);
      q.memset(k.wire.error, 0, 4);
    }
    q.wait();
  }
}

Head::~Head() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  for (Card &k : cards_)
    for (void *p : {(void *)k.st.res[0], (void *)k.st.res[1], (void *)k.st.lora[0], (void *)k.st.lora[1], (void *)k.xn, (void *)k.s, (void *)k.gate, (void *)k.out, (void *)k.logits,
                    (void *)k.tokens, (void *)k.local_keys, k.scratch, (void *)k.wire.inbox, (void *)k.wire.step, (void *)k.wire.error})
      if (p) fab_.release(p);
  if (ids_host_) fab_.release(ids_host_);
}

void Head::set_ids(const int32_t *ids, int rows) {
  if (rows < 1 || rows > kRows) throw std::invalid_argument("qwen38 head: rows");
  // A host write into the pinned buffer the embed reads.
  // The waits keep it safe when an earlier embed is still queued; after a step's wait they return at once.
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  std::memcpy(ids_host_, ids, static_cast<size_t>(rows) * 4);
}

void Head::embed(int rows) {
  for (size_t c = 0; c < cards_.size(); ++c)
    ev_[c].embed = kernels::embed_gather(fab_.compute(c), embed_[c], ids_host_, rows, hc_.hidden, hc_.streams, cards_[c].st.residual());
}

void Head::final_mix(int rows, const std::vector<bf16 *> &block, kernels::HcPath path) {
  const size_t C = cards_.size();
  for (size_t c = 0; c < C; ++c) {
    const bf16 *blk = c < block.size() ? block[c] : nullptr;
    const auto e = kernels::hc_mixer(fab_.compute(c), hc_, rows, cards_[c].st, blk, mix_[c], scratch(static_cast<int>(c)), cards_[c].out, path);
    ev_[c].norm = e.norm;
    ev_[c].down = e.down;
    ev_[c].mix = e.up;
  }
}

void Head::final_mix_own(int rows, const std::vector<bf16 *> &block) {
  const size_t C = cards_.size();
  if (static_cast<int>(C) != kernels::kWireCardsMax || rows < kernels::kOwnerMinRows || block.size() != C)
    throw std::invalid_argument("qwen38 head: rows by owner need four cards, 4+ rows and every card's block");
  for (size_t c = 0; c < C; ++c) {
    const auto e = kernels::hc_mixer_at(fab_.compute(c), hc_, kernels::owner_rows(static_cast<int>(c), rows), cards_[c].st, block[c], mix_[c],
                                        scratch(static_cast<int>(c)), cards_[c].out);
    ev_[c].norm = e.norm;
    ev_[c].down = e.down;
    ev_[c].mix = e.up;
  }
}

void Head::lm_pick(int rows) {
  const size_t C = cards_.size();
  for (size_t c = 0; c < C; ++c) ev_[c].lm = kernels::dense_gemm(fab_.compute(c), rows, cards_[c].out, lm_[c], cards_[c].logits);
  for (size_t c = 0; c < C; ++c)
    ev_[c].pick = kernels::greedy_pick(fab_.compute(c), cards_[c].logits, vpc_, rows, vpc_, static_cast<int>(c) * vpc_, cards_[c].scratch,
                                       C > 1 && !host_pick_ ? &cards_[c].wire : nullptr, cards_[c].tokens, &ev_[c].chunks,
                                       cards_[c].local_keys);
}

void Head::lm_pick(int rows, const std::vector<const bf16 *> &in, const std::vector<int32_t *> &tokens, const std::vector<float *> &logits,
                   const std::vector<kernels::DenseWeights> *w) {
  const size_t C = cards_.size();
  if (in.size() != C || tokens.size() != C || logits.size() != C) throw std::invalid_argument("qwen38 head: lm_pick needs one input, token and logits buffer per card");
  if (w && (w->size() != C || (*w)[0].n != lm_[0].n || (*w)[0].k != lm_[0].k)) throw std::invalid_argument("qwen38 head: lm_pick weights must match lm_head's slice");
  for (size_t c = 0; c < C; ++c) kernels::dense_gemm(fab_.compute(c), rows, in[c], w ? (*w)[c] : lm_[c], logits[c]);
  for (size_t c = 0; c < C; ++c)  // events not kept: events(c) stays the target step's (gpu_span_ms)
    kernels::greedy_pick(fab_.compute(c), logits[c], vpc_, rows, vpc_, static_cast<int>(c) * vpc_, cards_[c].scratch,
                         C > 1 ? &cards_[c].wire : nullptr, tokens[c]);
}

void Head::wait() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait_and_throw();
}

std::vector<int32_t> Head::tokens(int card, int rows) {
  std::vector<int32_t> t(static_cast<size_t>(rows));
  if (host_pick_) {
    std::vector<uint64_t> best(static_cast<size_t>(rows));
    wait();
    for (const auto &entry : cards_)
      for (size_t r = 0; r < best.size(); ++r) best[r] = std::max(best[r], entry.local_keys[r]);
    for (size_t r = 0; r < best.size(); ++r) t[r] = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(best[r]));
    return t;
  }
  fab_.compute(static_cast<size_t>(card)).memcpy(t.data(), cards_[static_cast<size_t>(card)].tokens, t.size() * sizeof(int32_t)).wait_and_throw();
  const int vocab = static_cast<int>(cards_.size()) * vpc_;
  std::vector<float> logits;
  for (int r = 0; r < rows; ++r) {
    int32_t &token = t[static_cast<size_t>(r)];
    if (token >= 0 && token < vocab) continue;
    std::fprintf(stderr, "qwen38 head: invalid device pick %d (row %d of %d, card %d); rescoring logits\n", token, r, rows, card);
    logits.resize(static_cast<size_t>(vpc_));
    uint64_t best = 0;
    for (size_t c = 0; c < cards_.size(); ++c) {
      const float *row = cards_[c].logits + static_cast<size_t>(r) * vpc_;
      fab_.compute(c).memcpy(logits.data(), row, logits.size() * sizeof(float)).wait_and_throw();
      const int base = static_cast<int>(c) * vpc_;
      for (int j = 0; j < vpc_; ++j)
        best = std::max(best, kernels::pick_key(logits[static_cast<size_t>(j)], base + j));
    }
    token = static_cast<int32_t>(0xFFFFFFFEu - static_cast<uint32_t>(best));
  }
  return t;
}

uint32_t Head::wire_error(int c) {
  uint32_t e = 0;
  if (cards_[static_cast<size_t>(c)].wire.error)
    fab_.compute(static_cast<size_t>(c)).memcpy(&e, cards_[static_cast<size_t>(c)].wire.error, 4).wait();
  return e;
}

}  // namespace xenithium::arch::qwen38
