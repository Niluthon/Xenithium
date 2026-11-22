#include "xenithium/arch/qwen38/step_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "xenithium/core/sampler.hpp"
#include "xenithium/kernels/head/logprob.hpp"
#include "xenithium/util/env.hpp"

namespace xenithium::arch::qwen38 {

void StepModelAdapter::begin(int slot) {
  m_.use_slot(slot);
  m_.begin_sequence();
}

void StepModelAdapter::prefill(int slot, const int32_t *tokens, int n, int32_t next) {
  m_.use_slot(slot);
  m_.prefill(tokens, n, {}, next);
  // the head ran on the call's last 4-row slice (counted from the call's first token)
  last_ = (n - 1) % kernels::kDecodeRows;
}

void StepModelAdapter::prefill_begin(int slot, const int32_t *tokens, int n, int32_t next) {
  m_.use_slot(slot);
  m_.prefill_begin(tokens, n, next);
  last_ = (n - 1) % kernels::kDecodeRows;
}

void StepModelAdapter::prefill_rows(int slot, const int32_t *tokens, int n, int32_t next, const RowsFn &on_rows) {
  m_.use_slot(slot);
  m_.prefill(
      tokens, n,
      [&](int row0, int rows) {
        m_.logits(rows, rows_);
        on_rows(row0, rows, rows_.data());
      },
      next);
  last_ = (n - 1) % kernels::kDecodeRows;
}

void StepModelAdapter::prefill_each(int slot, const int32_t *tokens, int n, int32_t next, const EachFn &on_rows) {
  m_.use_slot(slot);
  m_.prefill(tokens, n, [&](int row0, int rows) { on_rows(row0, rows); }, next);
  last_ = (n - 1) % kernels::kDecodeRows;
}

void StepModelAdapter::merge_row(const std::vector<float> &rec, int n, int r, float xt, int top_n, core::TokenLogprob &out) const {
  core::Sampler::LpPart parts[kMaxCards];
  const int C = m_.cards();  // 1 at TP1
  for (int c = 0; c < C; ++c) {
    const float *x = rec.data() + (static_cast<size_t>(c) * n + r) * kernels::kLpRec;
    auto &p = parts[c];
    p.max = x[0];
    p.sum = x[1];
    p.above = static_cast<int64_t>(x[2]);
    const int k = static_cast<int>(x[3]);
    for (int i = 0; i < k; ++i) p.top.push_back({static_cast<int32_t>(x[4 + 2 * i]), x[5 + 2 * i]});
  }
  core::Sampler::merge_logprobs(parts, C, xt, top_n, out);
}

void StepModelAdapter::row_logprobs(int rows, const int32_t *targets, int top_n, core::TokenLogprob *out) {
  if (rows < 1 || rows > kernels::kLpMaxRows) throw std::invalid_argument("qwen38 adapter: logprobs for 1..8 rows (one head pass)");
  m_.logprob_stats(rows, targets, std::clamp(top_n, 0, kernels::kLpMaxTop), lp_rec_, lp_xt_);
  for (int r = 0; r < rows; ++r)
    if (targets[r] >= 0) merge_row(lp_rec_, rows, r, lp_xt_[static_cast<size_t>(r)], top_n, out[r]);
}

bool StepModelAdapter::prefill_row_logprobs(int slot, const int32_t *tokens, int n, int32_t next, const int32_t *targets, int top_n, core::TokenLogprob *out) {
  if (!rows_lp_ || !m_.pf_logprob_rows()) return false;
  m_.use_slot(slot);
  m_.prefill_logprobs(tokens, n, next, targets, std::clamp(top_n, 0, kernels::kLpMaxTop), lp_rec_, lp_xt_);
  last_ = (n - 1) % kernels::kDecodeRows;
  for (int r = 0; r < n; ++r)
    if (targets[r] >= 0) merge_row(lp_rec_, n, r, lp_xt_[static_cast<size_t>(r)], top_n, out[r]);
  return true;
}

void StepModelAdapter::prefill_logprobs(int32_t token, int top_n, core::TokenLogprob &out) {
  int32_t t[Head::kRows];
  std::fill(std::begin(t), std::end(t), -1);
  t[last_] = token;
  core::TokenLogprob lp[Head::kRows];
  row_logprobs(last_ + 1, t, top_n, lp);
  out = std::move(lp[last_]);
}

int32_t StepModelAdapter::prefill_pick() { return m_.tokens(last_ + 1)[static_cast<size_t>(last_)]; }

void StepModelAdapter::prefill_logits(std::vector<float> &row) {
  std::vector<float> all;
  m_.logits(last_ + 1, all);
  const auto v = static_cast<size_t>(m_.config().vocab);
  row.assign(all.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(last_) * v),
             all.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(last_ + 1) * v));
}

std::vector<int32_t> StepModelAdapter::first_drafts(int slot, int32_t first, int depth, int sequences) {
  m_.use_slot(slot);
  const int seqs = std::max(1, sequences);
  if (!m_.mtp_drafts(seqs)) {  // the prompt is in the DFlash drafter's context; depth 0 drafts nothing
    // (with the MTP layer beside it, its KV row of the prompt's last token is redone when needed)
    if (depth < 1 && !m_.has_drafter()) return {};
    auto d = m_.draft_after_prefill(first, std::clamp(depth, 0, seg_rows_max() - 1), seqs);
    d.resize(std::min(d.size(), static_cast<size_t>(std::max(depth, 0))));
    return d;
  }
  // XEN_PF_LOG (diagnostic): the first drafts' own time. The prefill's drafter pass is
  // drained first (card 0's pick read), so the time is the drafts alone; the read is the only change.
  static const bool log = util::env_flag("XEN_PF_LOG");
  std::chrono::steady_clock::time_point t0;
  int g0 = 0;
  double r0 = 0;
  bool rerun = false;
  if (log) {
    rerun = prefill_pick() != first;
    g0 = m_.mtp_graphs_recorded();
    r0 = m_.mtp_graph_record_ms();
    t0 = std::chrono::steady_clock::now();
  }
  // the prompt's last pass-1 row reruns with the emitted token when it differs from the pick (its
  // KV must hold the real token) even when no draft is wanted
  auto d = m_.draft_after_prefill(first, std::clamp(depth, 1, kDepth), seqs);
  if (log)
    std::fprintf(stderr, "first-drafts took %.1f ms (depth %d, pass-1 rerun %d; drafter graphs recorded %d in %.1f ms)\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), depth, rerun ? 1 : 0, m_.mtp_graphs_recorded() - g0,
                 m_.mtp_graph_record_ms() - r0);
  d.resize(std::min(d.size(), static_cast<size_t>(std::max(depth, 0))));
  return d;
}

void StepModelAdapter::verify(const std::vector<SpecSeg> &segs) {
  std::vector<Model::SeqRows> s;
  vr_.clear();
  for (const auto &g : segs) {
    s.push_back({g.slot, g.tokens, g.rows, g.parent});  // a tree's parent rows (DFlash trees)
    vr_.push_back(g.rows);
  }
  m_.verify_batch(s);
  vsegs_ = static_cast<int>(segs.size());
}

int StepModelAdapter::pass1_next(const std::vector<SpecAccept> &a, int32_t *nx) const {
  int row = 0;
  for (int k = 0; k < vsegs_; ++k) {
    const auto &x = a[static_cast<size_t>(k)];
    for (int r = 0; r < vr_[static_cast<size_t>(k)]; ++r) nx[row++] = x.next[std::min(r, x.n - 1)];
  }
  return row;
}

std::vector<std::vector<int32_t>> StepModelAdapter::accept(const std::vector<SpecAccept> &a) {
  if (static_cast<int>(a.size()) != vsegs_) throw std::invalid_argument("qwen38 adapter: one accept per verified segment");
  return m_.mtp_drafts(vsegs_) ? accept_mtp(a) : accept_dflash(a);
}

// Commit, then one DFlash pass: the verified rows into the context, the drafts from each segment's token after its kept
// rows (depth 0 everywhere: a context-only pass, deferred). With the MTP layer loaded too, its pass 1 writes its KV
// over every verified row first (KV only, before the commit: it reads the verify's last state).
std::vector<std::vector<int32_t>> StepModelAdapter::accept_dflash(const std::vector<SpecAccept> &a) {
  if (m_.has_drafter()) {
    int32_t nx[Head::kRows];
    m_.draft_pass1(nx, pass1_next(a, nx), true);
  }
  std::vector<int> keep, depth;
  std::vector<int32_t> anchor;
  for (int k = 0; k < vsegs_; ++k) {
    const auto &x = a[static_cast<size_t>(k)];
    keep.push_back(x.n);
    const int32_t anc = x.next[x.n - 1];
    anchor.push_back(anc);
    // A sentinel token (any id below 0: no sampled token this row) must never reach the drafter: its block row 0 embeds
    // the anchor, and an out-of-range id faults the GPU (CAT error -> DEVICE_LOST). Such a segment drafts nothing; the
    // pass stays a context-only pass and the request keeps decoding.
    const bool bad_anchor = anc < 0;
    static int bad_anchor_logged = 0;
    if (bad_anchor && bad_anchor_logged < 8) {
      ++bad_anchor_logged;
      std::fprintf(stderr, "qwen38 adapter: sentinel anchor %d (n %d, depth %d): no drafts for this segment\n", anc, x.n, x.depth);
    }
    static const bool pass_dump = [] {
      const char *setting = std::getenv("XEN_PASSDUMP");
      return setting && *setting == '1';
    }();
    if (pass_dump) {
      std::fprintf(stderr, "ACCEPTGEO seg %d n %d depth %d anchor %d next", k, x.n, x.depth, x.next[x.n - 1]);
      for (int r = 0; r < x.n && r < 8; ++r) std::fprintf(stderr, " %d", x.next[r]);
      std::fprintf(stderr, "\n");
    }
    depth.push_back(bad_anchor ? 0 : std::clamp(x.depth, 0, seg_rows_max() - 1));  // each segment's own (unequal segments)
  }
  // the walked rows of the tree segments (SpecAccept::path)
  std::vector<const int *> paths;
  bool any = false;
  for (const auto &x : a) paths.push_back(x.path), any |= x.path != nullptr;
  if (any) m_.commit_batch(keep, paths);
  else m_.commit_batch(keep);
  auto out = m_.dflash_after_commit(keep, anchor.data(), depth);
  for (int k = 0; k < vsegs_; ++k) {
    auto &v = out[static_cast<size_t>(k)];
    v.resize(std::min(v.size(), static_cast<size_t>(std::max(0, a[static_cast<size_t>(k)].depth))));
  }
  return out;
}

// Pass 1 over every verified row of every segment (rows past n only write KV a later pass 1 rewrites), then commit,
// then each segment's chain from its last kept row. Segments of unequal rows too (the verify's layout), and with the
// DFlash drafter loaded too, its context takes the verified rows (a depth-0 pass: deferred).
std::vector<std::vector<int32_t>> StepModelAdapter::accept_mtp(const std::vector<SpecAccept> &a) {
  int32_t nx[Head::kRows];
  const int rows = pass1_next(a, nx);
  std::vector<int> keep, row0;
  for (int k = 0, r0 = 0; k < vsegs_; r0 += vr_[static_cast<size_t>(k)], ++k) {
    keep.push_back(a[static_cast<size_t>(k)].n);
    row0.push_back(r0);
  }
  // When no segment drafts after this step, pass 1 only writes the drafter KV (the same bits; later drafts need every
  // position), no picks.
  bool any_depth = false;
  for (int k = 0; k < vsegs_; ++k) any_depth |= a[static_cast<size_t>(k)].depth > 0;
  m_.draft_pass1(nx, rows, kv_only_ && !any_depth);
  m_.commit_batch(keep);
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(vsegs_));
  std::vector<int32_t> picks;  // pass 1's picks, read once for the depth-1 segments
  std::vector<int> chain_rows, chain_segs;  // depth > 1: every segment's chain at once
  int chain_depth = 0;
  for (int k = 0; k < vsegs_; ++k) {
    const auto &x = a[static_cast<size_t>(k)];
    const int row = row0[static_cast<size_t>(k)] + x.n - 1;
    if (x.depth == 1) {
      if (picks.empty()) picks = m_.draft_picks(rows);
      out[static_cast<size_t>(k)] = {picks[static_cast<size_t>(row)]};
    } else if (x.depth > 1) {
      chain_rows.push_back(row);
      chain_segs.push_back(k);
      chain_depth = std::max(chain_depth, std::min(x.depth, kDepth));
    }
  }
  if (!chain_rows.empty()) {
    auto d = m_.draft_chains(chain_rows, chain_depth);
    for (size_t i = 0; i < chain_segs.size(); ++i) {
      auto &v = d[i];
      v.resize(std::min(v.size(), static_cast<size_t>(std::min(a[static_cast<size_t>(chain_segs[i])].depth, kDepth))));
      out[static_cast<size_t>(chain_segs[i])] = std::move(v);
    }
  }
  if (m_.has_dflash()) {
    std::vector<int32_t> anchor;
    for (int k = 0; k < vsegs_; ++k) anchor.push_back(a[static_cast<size_t>(k)].next[a[static_cast<size_t>(k)].n - 1]);
    m_.dflash_after_commit(keep, anchor.data(), std::vector<int>(static_cast<size_t>(vsegs_), 0));
  }
  return out;
}

void StepModelAdapter::step(const std::vector<int> &slots, const int32_t *tokens) {
  std::vector<Model::SeqRows> segs;
  segs.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) segs.push_back({slots[i], tokens + i, 1});
  m_.step_batch(segs);
}

void StepModelAdapter::picks(int rows, int32_t *out) {
  const auto t = m_.tokens(rows);
  std::copy(t.begin(), t.begin() + rows, out);
}

}  // namespace xenithium::arch::qwen38
