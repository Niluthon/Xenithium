// xenithium-probe dflash: the DFlash 2 drafter on qwen38, one load for every run. Written for this engine.
// [--depth D] [--context N] [--graphs 0|1] [--hot-rank F] [--reserve G] [--mtp-vocab K]
// [--dflash-shift 0|1] [--dflash-cut X] [--dflash-defer N] [--dflash-bytes 1|2]
// Prompts: P pieces of N tokens of the token files (the files' chunks concatenated, cut in order); --prompt-len -1: the
// files' first P chunks, each one prompt as stored (chat-formatted prompts). Per prompt:
// 1. plain greedy: Model:prefill, then G one-row steps (the reference tokens, ms per token);
// 2. DFlash greedy: Model:prefill (the drafter's context from the chunk's features), draft_after_prefill, then steps
// of [token, drafts] verified in one step, the longest prefix equal to the target's picks kept (host), commit_batch,
// dflash_after_commit from the token after the kept rows. Gate: the tokens == plain greedy's, every prompt.
// Reports tokens per verify step, the accepted-rows histogram, per-position acceptance, ms per token plain vs spec, the
// verify and the draft ms per step.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/route_counts.hpp"
#include "xenithium/rt/device.hpp"

#include "probe.hpp"
namespace q38 = xenithium::arch::qwen38;


namespace {

// A token file: int32 n, then n chunks of int32 len + len token ids (the layout every probe reads).
bool read_chunks(const std::string &path, std::vector<std::vector<int32_t>> &out) {
  std::ifstream f(path, std::ios::binary);
  int32_t n = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n < 1) return false;
  for (int32_t c = 0; c < n; ++c) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len < 2) return false;
    std::vector<int32_t> v(static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(len) * 4)) return false;
    out.push_back(std::move(v));
  }
  return true;
}

double ms_since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }

// qwen38's side of the DFlash host-reference gate (--dflash-check N; qwen35_dflash.cpp): slot 0, the many-row prefill,
// one sequence's verify of up to 8 rows, commit, dflash_after_commit.
struct Q38Target final : DfCheckTarget {
  q38::Model &m;
  xenithium::Fabric &fab;
  Q38Target(q38::Model &mm, xenithium::Fabric &f) : m(mm), fab(f) {}
  xenithium::arch::DFlash *dflash() override { return m.dflash(); }
  int cards() override { return 2; }
  int chunk = 512;
  int vocab = 0;
  int prefill_chunk() override { return chunk; }
  int slot() override { return 0; }
  int mtp_vocab() override { return vocab; }
  std::vector<int64_t> eos() override { return m.config().eos; }
  std::string embed_name() override { return "model.language_model.embed_tokens.weight"; }
  void reset() override {
    m.use_slot(0);
    m.begin_sequence();
  }
  void prefill(const int32_t *ids, int n) override { m.prefill(ids, n); }
  std::vector<int32_t> picks(int rows) override { return m.tokens(rows); }
  std::vector<int32_t> draft_after_prefill(int32_t first, int depth) override { return m.draft_after_prefill(first, depth); }
  void step(const int32_t *rows, int n) override { m.verify(rows, n); }
  void accept(int keep) override { m.commit(keep); }
  std::vector<int32_t> draft(const int32_t *pk, int rows, int keep, int depth) override {
    (void)rows;
    const int32_t anchor = pk[keep - 1];
    return m.dflash_after_commit({keep}, &anchor, depth)[0];
  }
  void read_device(int c, const void *src, void *dst, size_t bytes) override { fab.compute(static_cast<size_t>(c)).memcpy(dst, src, bytes).wait(); }
};

}  // namespace

int cmd_dflash38(const std::string &model, const std::string &dflash, const std::string &ppl, int prompt_len, int prompts, int generate, int depth, int64_t context,
                 bool graphs, const std::string &hot_rank, double reserve_gib, int threads, int shift) {
  if (model.empty() || dflash.empty() || ppl.empty() || depth < 1 || depth > 7) {
    std::fprintf(stderr, "dflash: --model DIR --dflash DIR --ppl F are required; --depth 1..7\n");
    return 2;
  }
  std::vector<std::vector<int32_t>> chunks;
  {
    std::stringstream ss(ppl);
    for (std::string p; std::getline(ss, p, ',');)
      if (!read_chunks(p, chunks)) {
        std::fprintf(stderr, "dflash: cannot read token file %s\n", p.c_str());
        return 2;
      }
  }
  // Prompts: --prompt-len N > 0 (default 1,024) cuts the files' text (chunks joined) into pieces of N tokens; -1 = each
  // chunk is one prompt as stored (chat-formatted prompt files).
  std::vector<int32_t> text;
  for (const auto &c : chunks) text.insert(text.end(), c.begin(), c.end());
  if (prompt_len == 0) prompt_len = 1024;
  if (prompts <= 0) prompts = 4;
  std::vector<std::vector<int32_t>> prompt_list;
  if (prompt_len < 0) {
    for (size_t c = 0; c < chunks.size() && static_cast<int>(prompt_list.size()) < prompts; ++c) prompt_list.push_back(chunks[c]);
  } else {
    for (int pi = 0; pi < prompts && static_cast<size_t>(pi + 1) * prompt_len <= text.size(); ++pi)
      prompt_list.emplace_back(text.begin() + static_cast<ptrdiff_t>(pi) * prompt_len, text.begin() + static_cast<ptrdiff_t>(pi + 1) * prompt_len);
  }
  prompts = static_cast<int>(prompt_list.size());
  if (prompts < 1) {
    std::fprintf(stderr, "dflash: the token files hold fewer than --prompt-len tokens\n");
    return 2;
  }
  size_t longest = 0;
  for (const auto &pr : prompt_list) longest = std::max(longest, pr.size());
  q38::ModelOptions mo;
  mo.kv_split = g_kv_tokens ? xenithium::kernels::QsaKvSplit::tokens : xenithium::kernels::QsaKvSplit::heads;
  mo.prefill_moe_a8 = g_pf_moe_a8;
  mo.prefill_moe_bias = g_pf_moe_fast;
  mo.prefill_dense_a8 = g_pf_dense_a8 != 0;
  mo.prefill_dense_a8_row = g_pf_dense_a8 == 2;
  mo.prefill_wire_bf16 = g_q38_pf_wire_bf16;
  mo.moe_dn_ksplit = g_q38_moe_dn_ks;
  mo.kv_pool_tokens = g_kv_pool;
  mo.kv_lend = g_kv_lend, mo.kv_lend_keep = g_kv_lend_keep;
  mo.context = std::max<int64_t>(context, static_cast<int64_t>(longest) + generate + 64);
  mo.graphs = graphs;
  mo.threads = threads;
  mo.dflash_dir = dflash;
  mo.mtp_vocab = g_q38_mtp_vocab;
  if (g_dflash_bytes > 0) mo.dflash_dense_bytes = g_dflash_bytes;
  if (g_dflash_defer >= 0) mo.dflash_ctx_defer = g_dflash_defer;
  if (g_dflash_cut >= 0) mo.dflash_conf_cut = g_dflash_cut;
  mo.dflash_tree = g_dflash_tree, mo.dflash_tree_cut = g_dflash_tree_cut;
  mo.dflash_tree_seqs = g_dflash_tree_seqs;
  if (g_dflash_tree_temp > 0) mo.dflash_tree_temp = g_dflash_tree_temp;
  mo.dflash_feat_shift = shift;
  if (reserve_gib >= 0) mo.reserve_gib = reserve_gib;
  if (!hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(model));
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, tp_devices("dflash38", true));  // --tp 1 (device 0)
  q38::Model m(loader, fab, model, mo);
  const auto &ls = m.load_stats();
  auto *df = m.dflash();
  std::printf("dflash: loaded in %.1f s; drafter %s: %.3f / %.3f GiB on cards 0 / 1, block %d, draft vocabulary %d; context %lld; hot experts %lld / %lld per card, "
              "placed by %s; feature shift %d\n",
              ls.total_s, dflash.c_str(), df->card_bytes(0) / 1073741824.0, (fab.size() > 1 ? df->card_bytes(1) : 0) / 1073741824.0, df->config().block, df->draft_vocab(),
              static_cast<long long>(m.capacity()), static_cast<long long>(ls.hot[0]), static_cast<long long>(ls.hot[0] + ls.cold[0]),
              hot_rank.empty() ? "even split" : hot_rank.c_str(), shift);
  // drafts at the asked depth (up to the block's 7); the verify carries what a segment holds (seg_rows_max - 1), the rest
  // are scored against plain greedy's tokens (the accept a deeper verify would have had)
  const int ddepth = std::min(depth, df->max_depth());
  depth = std::min(depth, m.seg_rows_max() - 1);
  std::vector<int> hyp(static_cast<size_t>(ddepth) + 2, 0);
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  if (g_dflash_check > 0) {  // the drafter's kernels on this target vs the checkpoint's host reference (two cycles)
    Q38Target tg(m, fab);
    tg.chunk = mo.prefill_chunk;
    tg.vocab = mo.mtp_vocab;
    dflash_check(tg, model, std::vector<int32_t>(text.begin(), text.begin() + std::min<size_t>(text.size(), 4096)), g_dflash_check, mo.dflash_dense_bytes, check);
  }
  std::vector<int> hist(static_cast<size_t>(depth) + 2, 0);
  std::vector<int64_t> pos_drafted(static_cast<size_t>(depth), 0), pos_accepted(static_cast<size_t>(depth), 0);
  int64_t steps_all = 0, toks_all = 0, trees_all = 0, branch_all = 0, rec_n = 0;
  double rec_ms = 0;
  struct DecSum {
    int steps = 0;
    double ms[q38::StepProfile::kParts] = {}, span = 0, moe[4] = {};
  } dsum[2];
  double plain_all = 0, spec_all = 0, verify_all = 0, draft_all = 0, first_all = 0;
  bool same_all = true;
  for (int pi = 0; pi < prompts; ++pi) {
    const int32_t *prompt = prompt_list[static_cast<size_t>(pi)].data();
    const int plen = static_cast<int>(prompt_list[static_cast<size_t>(pi)].size());
    const int lr = (plen - 1) % 4;
    m.use_slot(0);
    m.begin_sequence();
    m.prefill(prompt, plen);
    const int32_t first = m.tokens(lr + 1)[static_cast<size_t>(lr)];
    std::vector<int32_t> ref{first};
    auto t0 = std::chrono::steady_clock::now();
    int32_t t = first;
    while (static_cast<int>(ref.size()) < generate) {
      m.step(&t, 1);
      t = m.tokens(1)[0];
      ref.push_back(t);
    }
    const double plain_ms = ms_since(t0);
    m.begin_sequence();
    m.prefill(prompt, plen);
    const int32_t first2 = m.tokens(lr + 1)[static_cast<size_t>(lr)];
    std::vector<int32_t> out{first2};
    t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> drafts = m.draft_after_prefill(first2, ddepth);
    first_all += m.draft_ms();
    double vms = 0, dms = 0;
    int64_t steps = 0;
    t = first2;
    while (static_cast<int>(out.size()) < generate) {
      {
        int a7 = 0;
        while (a7 < static_cast<int>(drafts.size()) && out.size() + a7 < ref.size() && drafts[static_cast<size_t>(a7)] == ref[out.size() + a7]) ++a7;
        if (out.size() + drafts.size() <= ref.size()) ++hyp[static_cast<size_t>(a7 + 1)];
      }
      if (static_cast<int>(drafts.size()) > depth) drafts.resize(static_cast<size_t>(depth));
      int32_t rows_t[8] = {t};
      const int nd = static_cast<int>(drafts.size());
      for (int k = 0; k < nd; ++k) rows_t[k + 1] = drafts[static_cast<size_t>(k)];
      const int R = 1 + nd;
      // The drafts may be a tree (Model:draft_parents: rows 1.. in preorder); the walk
      // follows the target's pick at each row into the child holding it
      std::vector<int> par = m.draft_parents(0);
      if (static_cast<int>(par.size()) > R) par.resize(static_cast<size_t>(R));
      const bool tree = static_cast<int>(par.size()) == R && R > 1;
      if (m.dec_profiling()) m.dec_profile_reset();
      const auto tv = std::chrono::steady_clock::now();
      if (tree) m.verify_batch({q38::Model::SeqRows{0, rows_t, R, par.data()}});
      else m.verify(rows_t, R);
      const auto pk = m.tokens(R);
      vms += ms_since(tv);
      if (m.times().record > 0) rec_ms += m.times().record, ++rec_n;  // A step graph recorded on first use
      if (m.dec_profiling() && R == 8) {  // 8-row verify parts, chains vs trees
        const auto dp = m.dec_profile();
        if (dp.steps == 1) {
          DecSum &ds = dsum[tree ? 1 : 0];
          ++ds.steps;
          for (int k = 0; k < q38::StepProfile::kParts; ++k) ds.ms[k] += dp.ms[0][k];
          ds.span += dp.span[0];
          for (int k = 0; k < 4; ++k) ds.moe[k] += dp.moe[0][k];
        }
      }
      int a = 0;
      std::vector<int> path{0};
      if (tree) {
        for (int node = 0;;) {
          int nx = -1;
          for (int x = node + 1; x < R && nx < 0; ++x)
            if (par[static_cast<size_t>(x)] == node && rows_t[x] == pk[static_cast<size_t>(node)]) nx = x;
          if (nx < 0) break;
          path.push_back(nx), node = nx;
        }
        a = static_cast<int>(path.size()) - 1;
      } else {
        while (a < nd && rows_t[a + 1] == pk[static_cast<size_t>(a)]) ++a;
        for (int k = 1; k <= a; ++k) path.push_back(k);
      }
      for (int k = 0; k < nd; ++k) {
        ++pos_drafted[static_cast<size_t>(k)];
        if (k < a) ++pos_accepted[static_cast<size_t>(k)];
      }
      if (tree) m.commit_batch({a + 1}, {path.data()});
      else m.commit(a + 1);
      for (int k = 1; k <= a; ++k) out.push_back(rows_t[path[static_cast<size_t>(k)]]);
      t = pk[static_cast<size_t>(path.back())];
      trees_all += tree, branch_all += tree && path.back() != a;
      out.push_back(t);
      ++hist[static_cast<size_t>(a + 1)];
      ++steps;
      const int32_t anchor = t;
      drafts = m.dflash_after_commit({a + 1}, &anchor, ddepth)[0];
      dms += m.draft_ms();
    }
    const double spec_ms = ms_since(t0);
    out.resize(static_cast<size_t>(generate));
    const bool same = out == ref && first == first2;
    same_all &= same;
    const int64_t gen = generate - 1;
    std::printf("  prompt %d (%d tokens): %lld verify steps for %lld tokens = %.2f per step; plain %.2f ms per token, spec %.2f (verify %.2f + drafts %.2f ms per "
                "step) -> %.2fx; %s\n",
                pi, plen, static_cast<long long>(steps), static_cast<long long>(gen), static_cast<double>(gen) / std::max<int64_t>(steps, 1), plain_ms / gen,
                spec_ms / gen, vms / std::max<int64_t>(steps, 1), dms / std::max<int64_t>(steps, 1), plain_ms / spec_ms, same ? "== plain greedy" : "DIFFERS from plain greedy");
    if (!same) {
      size_t i = 0;
      while (i < out.size() && i < ref.size() && out[i] == ref[i]) ++i;
      std::printf("    first difference at token %zu of %zu\n", i, ref.size());
    }
    steps_all += steps, toks_all += gen, plain_all += plain_ms, spec_all += spec_ms, verify_all += vms, draft_all += dms;
  }
  std::printf("dflash: %d prompts, %lld tokens in %lld verify steps = %.3f tokens per step (accepted rows per step:", prompts, static_cast<long long>(toks_all),
              static_cast<long long>(steps_all), static_cast<double>(toks_all) / std::max<int64_t>(steps_all, 1));
  for (size_t n = 1; n < hist.size(); ++n) std::printf(" %zu:%d", n, hist[n]);
  std::printf("); per draft position accepted:");
  for (int k = 0; k < depth; ++k)
    std::printf(" %.1f %%", 100.0 * static_cast<double>(pos_accepted[static_cast<size_t>(k)]) / std::max<int64_t>(1, pos_drafted[static_cast<size_t>(k)]));
  std::printf("\ndflash: plain %.2f ms per token (%.1f tok/s), spec %.2f (%.1f tok/s, verify %.2f + drafts %.2f ms per step; first drafts %.2f ms) = %.2fx\n",
              plain_all / toks_all, 1e3 * toks_all / plain_all, spec_all / toks_all, 1e3 * toks_all / spec_all, verify_all / std::max<int64_t>(steps_all, 1),
              draft_all / std::max<int64_t>(steps_all, 1), first_all / prompts, plain_all / spec_all);
  if (dsum[0].steps + dsum[1].steps > 0) {  // XEN_DEC_PROFILE, card 0, 8-row verifies
    std::printf("dflash: 8-row verify stamps (XEN_DEC_PROFILE, card 0, ms per step): %-10s %9s %9s\n", "part", "chain", "tree");
    for (int k = 0; k < q38::StepProfile::kParts; ++k)
      std::printf("  %-26s %9.3f %9.3f\n", q38::StepProfile::name(k), dsum[0].ms[k] / std::max(1, dsum[0].steps), dsum[1].ms[k] / std::max(1, dsum[1].steps));
    std::printf("  %-26s %9.3f %9.3f\n", "span", dsum[0].span / std::max(1, dsum[0].steps), dsum[1].span / std::max(1, dsum[1].steps));
    std::printf("  %-26s %9d %9d\n", "steps", dsum[0].steps, dsum[1].steps);
  }
  // The step graphs recorded on first use (a shape / bound / parity the run had not seen) inside the
  // spec timing; steady state = without them
  std::printf("dflash: %lld step graphs recorded during the speculative runs (%.1f ms); spec without them %.2f ms per token (%.1f tok/s)\n",
              static_cast<long long>(rec_n), rec_ms, (spec_all - rec_ms) / toks_all, 1e3 * toks_all / (spec_all - rec_ms));
  if (g_dflash_tree > 0)
    std::printf("dflash: trees (--dflash-tree %d): %lld of %lld verify steps verified a tree, %lld kept a branch (a path off the tree's leading chain)\n",
                g_dflash_tree, static_cast<long long>(trees_all), static_cast<long long>(steps_all), static_cast<long long>(branch_all));
  if (ddepth > depth) {
    int64_t n = 0, tk = 0;
    std::printf("dflash: depth %d drafts scored vs plain greedy (the accept of a %d-row verify):", ddepth, ddepth + 1);
    for (size_t k = 1; k < hyp.size(); ++k) std::printf(" %zu:%d", k, hyp[k]), n += hyp[k], tk += static_cast<int64_t>(k) * hyp[k];
    std::printf(" = %.3f tokens per step\n", static_cast<double>(tk) / std::max<int64_t>(n, 1));
  }
  check(same_all, "dflash: speculative greedy == plain greedy, token for token, every prompt");
  std::printf("dflash: pass graphs %d recorded in %.1f ms; deferred-context flushes %d\n", df->graphs(), df->graphs_ms(), df->flushes());
  check(m.wire_errors() == 0, "dflash: no mailbox timeout");
  std::printf("dflash: %s\n", rc ? "FAIL" : "PASS");
  return rc;
}
