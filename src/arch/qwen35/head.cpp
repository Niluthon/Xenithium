#include "xenithium/arch/qwen35/head.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "xenithium/kernels/embed/embed_aff.hpp"
#include "xenithium/kernels/residual/rmsnorm.hpp"

namespace xenithium::arch::qwen35 {

Head::Head(Fabric &fab, const load::DenseStore &dense, const Config &cfg, bool int8_copy, int rows) : fab_(fab), rows_(rows) {
  if (rows < 1 || rows > kRows) throw std::invalid_argument("qwen35 head: rows 1..kRows");
  const int C = static_cast<int>(fab.size());
  if (C > 2) throw std::runtime_error("qwen35 head: the pick wire pairs exactly two cards");
  if (cfg.vocab % (16 * C)) throw std::runtime_error("qwen35 head: vocab does not split into 16-row tiles per card");
  hidden_ = cfg.hidden;
  vpc_ = static_cast<int>(cfg.vocab / C);
  eps_ = static_cast<float>(cfg.rms_eps);
  plain_ = cfg.flint ? 2 : !cfg.norm_offset ? 1 : 0;  // arch llama: bf16(w x bf16(norm(x))); arch flint: bf16(w x norm(x))
  const std::string P = cfg.prefix;
  cards_.resize(static_cast<size_t>(C));
  ev_.resize(static_cast<size_t>(C));
  for (int c = 0; c < C; ++c) {
    auto get = [&](const std::string &n) -> const load::DeviceTensor & { return dense.at(c, n); };
    if (cfg.flint) {  // the 8-bit table + its group scales / biases (replicated raw)
      embed8_ = true;
      vocab_ = cfg.vocab;
      e8_.push_back(static_cast<const uint8_t *>(get(P + "embed_tokens.weight").ptr));
      e8s_.push_back(reinterpret_cast<const uint16_t *>(get(P + "embed_tokens.scales").ptr));
      e8b_.push_back(reinterpret_cast<const uint16_t *>(get(P + "embed_tokens.biases").ptr));
      embed_.push_back(nullptr);
    } else {
      embed_.push_back(reinterpret_cast<const bf16 *>(get(P + "embed_tokens.weight").ptr));
    }
    norm_.push_back(reinterpret_cast<const bf16 *>(get(P + "norm.weight").ptr));
    lm_.push_back(lm_mat(get("lm_head.weight")));
    if (lm_.back().n != vpc_ || lm_.back().k != hidden_) throw std::runtime_error("qwen35 head: unexpected lm_head slice shape");
    if (int8_copy) {  // the int8 rows beside the int4 head (the two-stage head's rescore)
      if (!lm_.back().is_w4()) throw std::runtime_error("qwen35 head: an int8 copy beside an int8 head");
      lm8_.push_back(get(kHeadI8Key).dense());
      if (lm8_.back().kind != kernels::DenseKind::i8 || lm8_.back().n != vpc_ || lm8_.back().k != hidden_)
        throw std::runtime_error("qwen35 head: unexpected int8 lm_head copy");
      if (kernels::dense_auto_ksplit(vpc_, hidden_) != 1) throw std::runtime_error("qwen35 head: the int8 head splits k: a tile rescore would not give its bits");
    }

    Card &k = cards_[static_cast<size_t>(c)];
    auto dev = [&](size_t bytes) { return fab.alloc(static_cast<size_t>(c), std::max<size_t>(bytes, 4096)); };
    k.res = static_cast<bf16 *>(dev(static_cast<size_t>(rows_) * hidden_ * 2));
    k.xn = static_cast<bf16 *>(dev(static_cast<size_t>(rows_) * hidden_ * 2));
    k.logits = static_cast<float *>(dev(static_cast<size_t>(rows_) * vpc_ * 4));
    k.tokens = static_cast<int32_t *>(dev(kernels::kPickMaxRows * 4));
    k.scratch = dev(kernels::pick_scratch_bytes(vpc_));
    if (int8_copy) {
      k.sel_scratch = dev(kernels::head_select_scratch_bytes(vpc_));
      k.tiles = static_cast<int32_t *>(dev(static_cast<size_t>(rows_) * kernels::kHeadTilesMax * kernels::head_select_parts(vpc_) * 4));  // TP1: 2 parts
    }
    if (C > 1) {
      k.wire.inbox = static_cast<uint32_t *>(fab.alloc_kernel_mailbox(4096));
      k.wire.step = static_cast<int32_t *>(dev(4));
      k.wire.error = static_cast<uint32_t *>(dev(4));
    }
  }
  mode_ = lm_[0].is_w4() ? (lm8_.empty() ? HeadMode::int4 : HeadMode::rescore) : HeadMode::int8;
  ids_host_ = static_cast<int32_t *>(fab.alloc_host(4096));
  std::memset(ids_host_, 0, 4096);
  // Dead rows of the GEMM input must hold finite values: zero them once.
  for (int c = 0; c < C; ++c) {
    Card &k = cards_[static_cast<size_t>(c)];
    auto &q = fab.compute(static_cast<size_t>(c));
    q.memset(k.res, 0, static_cast<size_t>(rows_) * hidden_ * 2);
    q.memset(k.xn, 0, static_cast<size_t>(rows_) * hidden_ * 2);
    if (C > 1) {
      k.wire.peer_inbox = cards_[static_cast<size_t>((c + 1) % C)].wire.inbox;
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
    for (void *p : {(void *)k.res, (void *)k.xn, (void *)k.logits, (void *)k.tokens, k.scratch, (void *)k.wire.inbox, (void *)k.wire.step,
                    (void *)k.wire.error, k.sel_scratch, (void *)k.tiles})
      if (p) fab_.release(p);
  if (ids_host_) fab_.release(ids_host_);
}

void Head::set_ids(const int32_t *ids, int rows) {
  if (rows < 1 || rows > rows_) throw std::invalid_argument("qwen35 head: rows");
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait();
  std::memcpy(ids_host_, ids, static_cast<size_t>(rows) * 4);
}

void Head::embed(int rows) {
  for (size_t c = 0; c < cards_.size(); ++c) ev_[c].embed = gather_on(fab_.compute(c), static_cast<int>(c), ids_host_, rows, cards_[c].res);
}

sycl::event Head::gather(int c, const int32_t *ids, int rows, bf16 *out) { return gather_on(fab_.compute(static_cast<size_t>(c)), c, ids, rows, out); }

sycl::event Head::gather_on(sycl::queue &q, int c, const int32_t *ids, int rows, bf16 *out) const {
  const size_t k = static_cast<size_t>(c);
  if (embed8_) return kernels::embed_aff8(q, rows, ids, e8_[k], e8s_[k], e8b_[k], vocab_, hidden_, out, hidden_);
  if (!ext_.empty()) return kernels::embed_gather_ext(q, embed_[k], ids, rows, hidden_, 1, out, ext_[k], ext_base_);  // the image rows (set_ext)
  return kernels::embed_gather(q, embed_[k], ids, rows, hidden_, 1, out);
}

void Head::final_norm(int rows) {
  for (size_t c = 0; c < cards_.size(); ++c)
    ev_[c].norm = kernels::rms_norm(fab_.compute(c), rows, hidden_, cards_[c].res, nullptr, nullptr, norm_[c], eps_, cards_[c].xn, plain_);
}

void Head::final_norm(int rows, const std::vector<const bf16 *> &add) {
  if (add.size() != cards_.size()) throw std::invalid_argument("qwen35 head: one add per card");
  for (size_t c = 0; c < cards_.size(); ++c)
    ev_[c].norm = kernels::rms_norm(fab_.compute(c), rows, hidden_, cards_[c].res, add[c], nullptr, norm_[c], eps_, cards_[c].xn, plain_);
}

bool Head::can(HeadMode m) const {
  const bool w4 = lm_[0].is_w4();
  return m == HeadMode::int8 ? !w4 || !lm8_.empty() : m == HeadMode::int4 ? w4 : w4 && !lm8_.empty();
}

void Head::set_mode(HeadMode m, int tiles) {
  if (!can(m)) throw std::invalid_argument("qwen35 head: that lm_head mode needs rows that are not loaded");
  if (tiles < 1 || tiles > kernels::kHeadTilesMax || 16 * tiles > vpc_ / 16) throw std::invalid_argument("qwen35 head: tiles 1..32");
  mode_ = m;
  tiles_ = tiles;
}

void Head::lm_run(size_t c, int rows, const bf16 *x, float *logits) {
  auto &q = fab_.compute(c);
  Events &e = ev_[c];
  if (mode_ == HeadMode::int8 && !lm8_.empty()) e.lm = kernels::dense_gemm(q, rows, x, lm8_[c], logits);
  else e.lm = lm_gemm(q, rows, x, lm_[c], logits);
  e.selmax = e.sel = e.rescore = e.lm;
  if (mode_ != HeadMode::rescore) return;
  Card &k = cards_[c];
  e.sel = kernels::head_select_tiles(q, logits, vpc_, rows, vpc_, tiles_, k.sel_scratch, k.tiles, &e.selmax);
  e.rescore = kernels::dense_gemm_tiles(q, rows, x, lm8_[c], logits, k.tiles, tiles_ * kernels::head_select_parts(vpc_));
}

void Head::lm_pick(int rows) {
  const size_t C = cards_.size();
  for (size_t c = 0; c < C; ++c) lm_run(c, rows, cards_[c].xn, cards_[c].logits);
  for (size_t c = 0; c < C; ++c)
    ev_[c].pick = kernels::greedy_pick(fab_.compute(c), cards_[c].logits, vpc_, rows, vpc_, static_cast<int>(c) * vpc_, cards_[c].scratch,
                                       C > 1 ? &cards_[c].wire : nullptr, cards_[c].tokens, &ev_[c].chunks);
}

void Head::lm_pick_to(int rows, const std::vector<const bf16 *> &x, const std::vector<float *> &logits, const std::vector<int32_t *> &tokens,
                      const PrePick &before_pick) {
  const size_t C = cards_.size();
  if (x.size() != C || logits.size() != C || tokens.size() != C) throw std::invalid_argument("qwen35 head: lm_pick_to wants one buffer per card");
  for (size_t c = 0; c < C; ++c) lm_run(c, rows, x[c], logits[c]);
  if (before_pick)
    for (size_t c = 0; c < C; ++c) before_pick(c, fab_.compute(c), logits[c], vpc_);
  for (size_t c = 0; c < C; ++c)
    ev_[c].pick = kernels::greedy_pick(fab_.compute(c), logits[c], vpc_, rows, vpc_, static_cast<int>(c) * vpc_, cards_[c].scratch,
                                       C > 1 ? &cards_[c].wire : nullptr, tokens[c], &ev_[c].chunks);
}

void Head::wait() {
  for (size_t c = 0; c < cards_.size(); ++c) fab_.compute(c).wait_and_throw();
}

std::vector<int32_t> Head::tokens(int card, int rows) {
  std::vector<int32_t> t(static_cast<size_t>(rows));
  fab_.compute(static_cast<size_t>(card)).memcpy(t.data(), cards_[static_cast<size_t>(card)].tokens, t.size() * 4).wait();
  return t;
}

uint32_t Head::wire_error(int c) {
  uint32_t e = 0;
  if (cards_[static_cast<size_t>(c)].wire.error) fab_.compute(static_cast<size_t>(c)).memcpy(&e, cards_[static_cast<size_t>(c)].wire.error, 4).wait();
  return e;
}

}  // namespace xenithium::arch::qwen35
