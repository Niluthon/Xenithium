// xenithium-probe mtp: the MTP drafter on the cards, one load for every run.
// xenithium-probe mtp --model DIR --mtp DIR --ppl F [--prompt-len N] [--generate N] [--depth D]
// [--drafts-ref F] [--context N] [--graphs 0|1] [--hot-rank F] [--reserve G]
// [--mtp-kv-bytes 1|2] (the drafter layer's KV: int8 or bf16 = the default; the target's
// stays int8) [--pf-prompt-len N [--pf-only 1] [--pf-chunk N]]
// --ppl F: a one-chunk token file of greedy text (probe model --gen-out: the prompt + the target's
// plain greedy tokens); --prompt-len N: where the greedy part starts (default: the whole chunk is
// the prompt, and --generate continues past it).
// 1. Plain greedy decode (1 row per step) of --generate tokens after the prompt: the baseline
// tokens and ms per token. With greedy text in --ppl, its tokens must equal the file's.
// 2. Teacher-forced drafter gate on the whole chunk: 1-row target steps, after each one drafter
// pass 1 (next token = the text's) and the chain of passes 2..D, as a decode that rejects every
// draft would run them. Scored as xenithium-mtp-ref scores its drafts (pass k at p accepted
// while it equals the target's pick at p + k, on the greedy region), and compared draft for
// draft with --drafts-ref (mtp-ref --drafts-out records: chunk, position, d1, d2, d3).
// 3. MTP greedy decode of --generate tokens: prompt in 4-row steps with pass 1 after each; then
// steps of [token, D drafts] verified in one target step, the longest prefix equal to the
// target's picks accepted (host), pass 1 over the step's rows and the chain from the last
// accepted row. Gate: the tokens == plain greedy's. Reports tokens per step, acceptance per
// draft position, ms per token against plain decode.
// 4. (--pf-prompt-len N) The drafter in Model:prefill on the file's first N tokens: off /
// slices / rows timed (prefill + first drafts); MTP greedy after a slices or rows prefill ==
// plain greedy after the prefill; pass-1 prompt picks rows vs slices. --pf-only 1 skips 1-3.
// "ttft parts" first: prefill off / rows until the first token, then the first drafts
// alone (greedy, and with a forced pass-1 rerun), drafter graphs recorded per run.
#include <algorithm>
#include <chrono>
#include <utility>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/core/sampler.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/device.hpp"
#include "kv.hpp"

#include "probe.hpp"  // the run-wide flags
namespace q38 = xenithium::arch::qwen38;

namespace {

void Sampler_top(const float *x, int64_t n, size_t k, std::vector<int32_t> &out) {
  xenithium::core::Sampler::top_ids(x, n, k, nullptr, out);
}

bool read_tokens(const std::string &path, std::vector<int32_t> &out) {
  std::ifstream f(path, std::ios::binary);
  int32_t n = 0, len = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n < 1 || !f.read(reinterpret_cast<char *>(&len), 4) || len < 2) return false;
  out.resize(static_cast<size_t>(len));
  return static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(len) * 4));
}

double med(std::vector<double> v, double p = 0.5) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int cmd_mtp_run(const std::string &model, const std::string &mtp, const std::string &ppl, int prompt_len, int generate, int depth,
                const std::string &drafts_ref, int64_t context, bool graphs, const std::string &hot_rank, double reserve_gib, int threads, int mtp_kv_bytes,
                int pf_prompt_len, bool pf_only, int pf_chunk) {
  if (model.empty() || mtp.empty() || ppl.empty() || depth < 1 || depth > 3) {
    std::fprintf(stderr, "mtp: --model DIR --mtp DIR --ppl F are required; --depth 1..3\n");
    return 2;
  }
  std::vector<int32_t> text;
  if (!read_tokens(ppl, text)) {
    std::fprintf(stderr, "mtp: cannot read token file %s\n", ppl.c_str());
    return 2;
  }
  const int len = static_cast<int>(text.size());
  if (prompt_len <= 0 || prompt_len > len) prompt_len = len;
  const std::vector<int32_t> prompt(text.begin(), text.begin() + prompt_len);
  q38::ModelOptions mo;
  mo.kv_split = g_kv_tokens ? xenithium::kernels::QsaKvSplit::tokens : xenithium::kernels::QsaKvSplit::heads;
  mo.prefill_moe_a8 = g_pf_moe_a8;
  mo.prefill_moe_bias = g_pf_moe_fast;
  mo.prefill_dense_a8 = g_pf_dense_a8 != 0;
  mo.prefill_dense_a8_row = g_pf_dense_a8 == 2;
  mo.prefill_wire_bf16 = g_q38_pf_wire_bf16;
  mo.moe_dn_ksplit = g_q38_moe_dn_ks;
  mo.decode_moe_a8 = g_dec_moe_a8;
  mo.kv_pool_tokens = g_kv_pool;
  mo.kv_lend = g_kv_lend, mo.kv_lend_keep = g_kv_lend_keep;
  mo.kv_pool_scramble = g_kv_scramble;
  mo.prefill_cold_stage = g_pf_cold_stage;
  mo.prefill_fetch = g_pf_fetch;
  mo.gdn_scan = g_gdn_scan;
  if (pf_prompt_len > len) pf_prompt_len = len;
  mo.context = std::max<int64_t>({context, static_cast<int64_t>(len) + 64, static_cast<int64_t>(prompt_len) + generate + 64,
                                  static_cast<int64_t>(pf_prompt_len) + generate + 64});
  if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;
  mo.graphs = graphs;
  mo.threads = threads;
  mo.mtp_dir = mtp;
  mo.mtp_vocab = g_q38_mtp_vocab;
  mo.mtp_head_w4 = g_q38_mtp_head_w4 != 0;
  mo.mtp_kv = probe_kv(mtp_kv_bytes);
  if (reserve_gib >= 0) mo.reserve_gib = reserve_gib;
  if (!hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(model));
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, tp_devices("mtp", true));  // --tp 1 (device 0)
  q38::Model m(loader, fab, model, mo);
  const auto &ls = m.load_stats();
  std::printf("mtp: loaded in %.1f s (drafter %.1f s: %.3f GiB per card incl. its KV; experts g32 -> g128 rms error %.4f of the weights, "
              "%lld scales not exact in fp16; KV %s); context %lld; hot experts %lld / %lld per card (card 0), placed by %s\n",
              ls.total_s, ls.mtp_s, static_cast<double>(ls.mtp_vram) / 1073741824.0, ls.mtp_requant_err, static_cast<long long>(ls.mtp_scales_inexact), probe_kv_name(mo.mtp_kv),
              static_cast<long long>(m.capacity()), static_cast<long long>(ls.hot[0]), static_cast<long long>(ls.hot[0] + ls.cold[0]),
              hot_rank.empty() ? "even split" : hot_rank.c_str());
  std::printf("mtp: draft vocabulary %d (0 = the whole vocabulary)\n", m.mtp_draft_vocab());
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  // Prompt in 4-row steps; with_drafter: pass 1 after each step (next = the following prompt token,
  // the target's pick after the last). Returns the target's token after the prompt.
  auto prefill = [&](bool with_drafter) {
    m.begin_sequence();
    int32_t pick = 0;
    for (int p0 = 0; p0 < prompt_len; p0 += 4) {
      const int rows = std::min(4, prompt_len - p0);
      m.step(&prompt[static_cast<size_t>(p0)], rows);
      const auto g = m.tokens(rows);
      pick = g[static_cast<size_t>(rows) - 1];
      if (with_drafter) {
        int32_t next[4];
        for (int r = 0; r < rows; ++r) next[r] = p0 + r + 1 < prompt_len ? prompt[static_cast<size_t>(p0 + r + 1)] : g[static_cast<size_t>(r)];
        m.draft_pass1(next, rows);
      }
    }
    return pick;
  };

  // ---- 5. the bound buckets' graphs recorded ahead: --bucket-gate N ----
  // One prompt of P = 65,536 - N / 2 tokens (the token file cycled) through Model:prefill (drafter
  // rows), then N tokens of plain greedy and N of MTP greedy: the decode crosses from the first QSA
  // bound bucket (65,536) into the next (needs --context > 65,536 + N / 2). Eager first (the
  // reference tokens); then with graphs: a short request (64-token prompt, plain + MTP: the first
  // bucket's step and pass graphs, as the server's warm-up), then Model:record_bucket_graphs (unless
  // --bucket-pre 0), then the long runs again. Gates: tokens == eager, and no graph recorded during
  // the long runs' decode (the control with --bucket-pre 0 must record some).
  if (g_bucket_gate > 0) {
    const int N = g_bucket_gate, P = 65536 - N / 2;
    if (N < 8 || N > 4096 || P + N + 8 > m.capacity()) {
      std::fprintf(stderr, "mtp: --bucket-gate N needs 8 <= N <= 4,096 and --context above 65,536 + N / 2 + 8\n");
      return 2;
    }
    std::vector<int32_t> ptext(static_cast<size_t>(P));
    for (int i = 0; i < P; ++i) ptext[static_cast<size_t>(i)] = text[static_cast<size_t>(i % len)];
    using PfMode = q38::Model::MtpPrefill;
    m.set_mtp_prefill(PfMode::rows);
    auto graphs_now = [&] { return m.graphs_recorded() + m.mtp_graphs_recorded(); };
    // returns the tokens; *rec = graphs recorded after the prefill (decode only)
    auto run = [&](int plen, int n, bool with_mtp, int *rec) {
      m.begin_sequence();
      m.prefill(ptext.data(), plen);
      const int last = (plen - 1) % 4;
      int32_t t = m.tokens(last + 1)[static_cast<size_t>(last)];
      const int g0 = graphs_now();
      std::vector<int32_t> out;
      if (!with_mtp) {
        while (static_cast<int>(out.size()) < n) {
          out.push_back(t);
          m.step(&t, 1);
          t = m.tokens(1)[0];
        }
      } else {
        std::vector<int32_t> drafts = m.draft_after_prefill(t, depth);
        while (static_cast<int>(out.size()) < n && m.position() + depth + 1 <= m.capacity()) {
          int32_t rows_t[4] = {t, 0, 0, 0};
          const int nd = static_cast<int>(drafts.size());
          for (int k = 0; k < nd; ++k) rows_t[k + 1] = drafts[static_cast<size_t>(k)];
          const int rows = 1 + nd;
          m.verify(rows_t, rows);
          const auto g = m.tokens(rows);
          int a = 0;
          while (a < nd && rows_t[a + 1] == g[static_cast<size_t>(a)]) ++a;
          m.draft_pass1(g.data(), rows);
          m.commit(a + 1);
          for (int k = 0; k <= a; ++k) out.push_back(rows_t[k]);
          t = g[static_cast<size_t>(a)];
          drafts = m.draft_chain(a, depth);
        }
        out.resize(std::min<size_t>(out.size(), static_cast<size_t>(n)));
      }
      if (rec) *rec = graphs_now() - g0;
      return out;
    };
    auto same_n = [](const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
      size_t i = 0;
      while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
      return i;
    };
    m.set_graphs(false);
    const auto ref_plain = run(P, N, false, nullptr), ref_mtp = run(P, N, true, nullptr);
    check(same_n(ref_plain, ref_mtp) == ref_plain.size(), "bucket gate: eager MTP greedy == eager plain greedy (" + std::to_string(same_n(ref_plain, ref_mtp)) + " / " +
                                                            std::to_string(ref_plain.size()) + ")");
    m.set_graphs(true);
    int rs = 0, rm = 0;
    run(64, 48, false, &rs);
    run(64, 48, true, &rm);
    std::printf("bucket gate: short request (64-token prompt, 48 tokens): graphs recorded plain %d, MTP %d; so far target %d, drafter %d\n", rs, rm,
                m.graphs_recorded(), m.mtp_graphs_recorded());
    if (g_bucket_pre) {
      auto vram_free = [&] {
        double lo = 1e30;
        for (size_t c = 0; c < fab.size(); ++c) {
          const auto vs = xenithium::l0::vram_state(loader.info(fab.rank_index(c)));
          if (vs.ok) lo = std::min(lo, static_cast<double>(vs.free));
        }
        return lo / 1073741824.0;
      };
      const double v0 = vram_free();
      const int t0n = m.graphs_recorded(), d0n = m.mtp_graphs_recorded();
      const double t0ms = m.graph_record_ms(), d0ms = m.mtp_graph_record_ms();
      const auto t0 = std::chrono::steady_clock::now();
      const int n = m.record_bucket_graphs();
      const double wall = ms_since(t0);
      std::printf("bucket gate: record_bucket_graphs: %d graphs (target %d in %.1f ms, drafter %d in %.1f ms), %.1f ms wall; capacity %lld; "
                  "VRAM free (tighter card) %.3f -> %.3f GiB\n",
                  n, m.graphs_recorded() - t0n, m.graph_record_ms() - t0ms, m.mtp_graphs_recorded() - d0n, m.mtp_graph_record_ms() - d0ms, wall,
                  static_cast<long long>(m.capacity()), v0, vram_free());
      check(n > 0 && n == m.graphs_recorded() - t0n + m.mtp_graphs_recorded() - d0n, "bucket gate: graphs recorded ahead (" + std::to_string(n) + ")");
      check(m.record_bucket_graphs() == 0, "bucket gate: a second record_bucket_graphs records nothing");
    }
    int gp = 0, gm = 0;
    const auto out_plain = run(P, N, false, &gp);
    const auto out_mtp = run(P, N, true, &gm);
    std::printf("bucket gate: prompt %d, %d tokens to position %lld; graphs recorded during decode: plain %d, MTP %d (pre-record %s)\n", P, N,
                static_cast<long long>(m.position()), gp, gm, g_bucket_pre ? "on" : "off");
    check(same_n(out_plain, ref_plain) == ref_plain.size(), "bucket gate: plain greedy with graphs == eager (" + std::to_string(same_n(out_plain, ref_plain)) + " / " +
                                                             std::to_string(ref_plain.size()) + ")");
    check(same_n(out_mtp, ref_mtp) == ref_mtp.size(), "bucket gate: MTP greedy with graphs == eager (" + std::to_string(same_n(out_mtp, ref_mtp)) + " / " +
                                                       std::to_string(ref_mtp.size()) + ")");
    if (g_bucket_pre)
      check(gp == 0 && gm == 0, "bucket gate: no graph recorded during the long runs' decode");
    else
      check(gp > 0 || gm > 0, "bucket gate control: without the pre-record the long runs record graphs");
    check(m.wire_errors() == 0, "bucket gate: no mailbox timeout");
    std::printf("step graphs: target %d recorded (%.1f ms recording + finalizing), drafter passes %d (%.1f ms)\n", m.graphs_recorded(), m.graph_record_ms(),
                m.mtp_graphs_recorded(), m.mtp_graph_record_ms());
    return rc;
  }

  // ---- 1. plain greedy ----
  std::vector<int32_t> plain;
  double plain_ms = 0;
  std::vector<double> plain_step;
  if (generate > 0 && !pf_only) {
    int32_t t = prefill(false);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < generate; ++i) {
      plain.push_back(t);
      const auto ts = std::chrono::steady_clock::now();
      m.step(&t, 1);
      t = m.tokens(1)[0];
      plain_step.push_back(ms_since(ts));
    }
    plain_ms = ms_since(t0);
    std::printf("plain greedy: %d tokens after a %d-token prompt in %.1f ms: %.2f ms per token (step median %.2f, p90 %.2f)\n", generate, prompt_len,
                plain_ms, plain_ms / generate, med(plain_step), med(plain_step, 0.9));
    if (prompt_len < len) {
      int same = 0;
      while (same < generate && prompt_len + same < len && plain[static_cast<size_t>(same)] == text[static_cast<size_t>(prompt_len + same)]) ++same;
      const int cmp = std::min(generate, len - prompt_len);
      check(same == cmp, "plain greedy tokens == the file's greedy text (" + std::to_string(same) + " / " + std::to_string(cmp) + " equal before the first difference)");
    }
  }

  // ---- 2. teacher-forced drafter gate ----
  if (!pf_only) {
    m.begin_sequence();
    std::vector<int32_t> pick(static_cast<size_t>(len)), d(static_cast<size_t>(len) * 3, -1);
    const auto t0 = std::chrono::steady_clock::now();
    // --spec-cap F (XSC1): int32 magic, K; then per position p: target's top-K (id int32, logit f32) of step p and the
    // drafter pass 1's top-K after it (it drafts token p + 2; the target's distribution for it is step p + 1's)
    std::ofstream cap;
    constexpr int kCapK = 64;
    std::vector<float> cap_lg, cap_dv;
    std::vector<int32_t> cap_ids;
    auto cap_top = [&](const float *x, const int32_t *ids, int64_t n) {
      std::vector<int32_t> top;
      Sampler_top(x, n, kCapK, top);
      for (int32_t v : top) {
        const int32_t id = ids ? ids[v] : v;
        cap.write(reinterpret_cast<const char *>(&id), 4);
        cap.write(reinterpret_cast<const char *>(&x[v]), 4);
      }
    };
    if (!g_spec_cap.empty()) {
      cap.open(g_spec_cap, std::ios::binary);
      const int32_t hdr[2] = {0x31435358, kCapK};
      cap.write(reinterpret_cast<const char *>(hdr), 8);
    }
    for (int p = 0; p < len; ++p) {
      m.step(&text[static_cast<size_t>(p)], 1);
      pick[static_cast<size_t>(p)] = m.tokens(1)[0];
      if (p + 1 >= len) break;
      const int32_t next = text[static_cast<size_t>(p) + 1];
      if (cap.is_open()) {
        m.logits(1, cap_lg);
        cap.write(reinterpret_cast<const char *>(&p), 4);
        cap_top(cap_lg.data(), nullptr, static_cast<int64_t>(cap_lg.size()));
      }
      m.draft_pass1(&next, 1);
      if (cap.is_open()) {
        m.draft_logits(1, cap_ids, cap_dv);
        cap_top(cap_dv.data(), cap_ids.data(), static_cast<int64_t>(cap_ids.size()));
      }
      const auto dr = m.draft_chain(0, depth);
      for (size_t k = 0; k < dr.size(); ++k) d[static_cast<size_t>(p) * 3 + k] = dr[k];
    }
    const double tf_ms = ms_since(t0);
    int greedy_from = len - 1;
    while (greedy_from > 0 && pick[static_cast<size_t>(greedy_from) - 1] == text[static_cast<size_t>(greedy_from)]) --greedy_from;
    int64_t n1 = 0, a1 = 0, chains = 0, acc = 0, at[4] = {0, 0, 0, 0}, pos_acc[3] = {0, 0, 0};
    for (int p = 0; p + 1 < len; ++p) {
      ++n1;
      a1 += d[static_cast<size_t>(p) * 3] == pick[static_cast<size_t>(p) + 1];
      if (p + 1 >= greedy_from && p + depth < len) {
        ++chains;
        int a = 0;
        while (a < depth && d[static_cast<size_t>(p) * 3 + a] == pick[static_cast<size_t>(p + a) + 1]) ++a;
        acc += a;
        ++at[a];
        for (int k = 0; k < a; ++k) ++pos_acc[k];
      }
    }
    std::printf("teacher-forced drafter (%d positions, %.1f s, greedy from %d): pass-1 agreement %.2f %% (%lld / %lld); MTP-%d on greedy text: %.3f "
                "tokens per target step over %lld chains (0: %lld, 1: %lld, 2: %lld, 3: %lld), per draft position %.2f %.2f %.2f\n",
                len, tf_ms / 1e3, greedy_from, n1 ? 100.0 * a1 / n1 : 0.0, static_cast<long long>(a1), static_cast<long long>(n1), depth,
                chains ? 1.0 + static_cast<double>(acc) / chains : 0.0, static_cast<long long>(chains), static_cast<long long>(at[0]), static_cast<long long>(at[1]),
                static_cast<long long>(at[2]), static_cast<long long>(at[3]), chains ? static_cast<double>(pos_acc[0]) / chains : 0.0,
                chains ? static_cast<double>(pos_acc[1]) / chains : 0.0, chains ? static_cast<double>(pos_acc[2]) / chains : 0.0);
    check(m.wire_errors() == 0, "teacher-forced drafter: no mailbox timeout");
    if (!drafts_ref.empty()) {
      std::ifstream f(drafts_ref, std::ios::binary);
      int32_t rec[5];
      int64_t n = 0, same[3] = {0, 0, 0}, nk[3] = {0, 0, 0}, chain_same = 0;
      while (f.read(reinterpret_cast<char *>(rec), sizeof rec)) {
        if (rec[0] != 0 || rec[1] < 0 || rec[1] + 1 >= len) continue;
        ++n;
        bool all = true;
        for (int k = 0; k < depth; ++k) {
          if (rec[2 + k] < 0) continue;
          ++nk[k];
          const bool eq = rec[2 + k] == d[static_cast<size_t>(rec[1]) * 3 + k];
          same[k] += eq;
          all &= eq;
        }
        chain_same += all;
      }
      std::printf("vs reference drafts %s: %lld positions; equal draft for draft: pass 1 %.2f %%, pass 2 %.2f %%, pass 3 %.2f %%; whole chain %.2f %%\n",
                  drafts_ref.c_str(), static_cast<long long>(n), nk[0] ? 100.0 * same[0] / nk[0] : 0.0, nk[1] ? 100.0 * same[1] / nk[1] : 0.0,
                  nk[2] ? 100.0 * same[2] / nk[2] : 0.0, n ? 100.0 * chain_same / n : 0.0);
    }
  }

  // ---- 3. MTP greedy decode ----
  if (generate > 0 && !pf_only) {
    int32_t t = prefill(true);
    std::vector<int32_t> drafts = m.draft_chain(m.position() > 0 ? std::min(3, (prompt_len - 1) % 4) : 0, depth);
    std::vector<int32_t> out;
    std::vector<double> verify_ms, draft_ms, step_ms;
    int64_t steps = 0, hist[4] = {0, 0, 0, 0}, pos_acc[3] = {0, 0, 0};
    int last_rows = 1;
    const auto t0 = std::chrono::steady_clock::now();
    while (static_cast<int>(out.size()) < generate && m.position() + depth + 1 <= m.capacity()) {
      const auto ts = std::chrono::steady_clock::now();
      int32_t rows_t[4] = {t, 0, 0, 0};
      const int nd = static_cast<int>(drafts.size());
      for (int k = 0; k < nd; ++k) rows_t[k + 1] = drafts[static_cast<size_t>(k)];
      const int rows = 1 + nd;
      last_rows = rows;
      m.verify(rows_t, rows);
      const auto g = m.tokens(rows);
      int a = 0;
      while (a < nd && rows_t[a + 1] == g[static_cast<size_t>(a)]) ++a;
      verify_ms.push_back(ms_since(ts));
      const auto td = std::chrono::steady_clock::now();
      m.draft_pass1(g.data(), rows);  // row r's next token is the target's pick (== the text where accepted)
      m.commit(a + 1);
      for (int k = 0; k <= a; ++k) out.push_back(rows_t[k]);
      for (int k = 0; k < a; ++k) ++pos_acc[k];
      ++hist[a];
      ++steps;
      t = g[static_cast<size_t>(a)];
      drafts = m.draft_chain(a, depth);
      draft_ms.push_back(ms_since(td));
      step_ms.push_back(ms_since(ts));
    }
    const double total = ms_since(t0);
    out.resize(std::min<size_t>(out.size(), static_cast<size_t>(generate)));
    const double tps = static_cast<double>(out.size()) / static_cast<double>(steps);
    std::printf("MTP-%d greedy: %zu tokens in %lld steps = %.3f tokens per step (accepted drafts 0: %lld, 1: %lld, 2: %lld, 3: %lld; per draft position "
                "%.2f %.2f %.2f)\n",
                depth, out.size(), static_cast<long long>(steps), tps, static_cast<long long>(hist[0]), static_cast<long long>(hist[1]),
                static_cast<long long>(hist[2]), static_cast<long long>(hist[3]), static_cast<double>(pos_acc[0]) / steps, static_cast<double>(pos_acc[1]) / steps,
                static_cast<double>(pos_acc[2]) / steps);
    std::printf("  ms per step median: %.2f (verify %.2f + drafter %.2f; p90 step %.2f); %.2f ms per token over the run (plain greedy %.2f: x%.2f)\n",
                med(step_ms), med(verify_ms), med(draft_ms), med(step_ms, 0.9), total / static_cast<double>(out.size()),
                generate > 0 ? plain_ms / generate : 0.0, generate > 0 && total > 0 ? (plain_ms / generate) / (total / static_cast<double>(out.size())) : 0.0);
    // The drafter alone at the end position (host wall incl. the wait; junk KV past the text is
    // harmless): pass 1 over the last verify's rows, then the chain of passes 2..D.
    {
      std::vector<double> p1, ch, kv;
      int32_t nx[4] = {t, t, t, t};
      for (int i = 0; i < 40; ++i) {
        auto ts = std::chrono::steady_clock::now();
        m.draft_pass1(nx, last_rows, true);  // the KV-only pass (steps after which nothing drafts)
        m.draft_picks(1);  // (only the wait: a KV-only pass leaves no picks)
        kv.push_back(ms_since(ts));
        ts = std::chrono::steady_clock::now();
        m.draft_pass1(nx, last_rows);
        m.draft_picks(1);
        p1.push_back(ms_since(ts));
        ts = std::chrono::steady_clock::now();
        m.draft_chain(0, depth);
        ch.push_back(ms_since(ts));
      }
      std::printf("  drafter alone (median of 40, wall incl. wait): pass 1 over %d rows %.3f ms (KV only %.3f); passes 2..%d %.3f ms (%.3f per pass)\n", last_rows,
                  med(p1), med(kv), depth, med(ch), depth > 1 ? med(ch) / (depth - 1) : 0.0);
    }
    int same = 0;
    while (same < static_cast<int>(out.size()) && same < static_cast<int>(plain.size()) && out[static_cast<size_t>(same)] == plain[static_cast<size_t>(same)]) ++same;
    check(same == static_cast<int>(std::min(out.size(), plain.size())),
          "MTP greedy tokens == plain greedy tokens (" + std::to_string(same) + " / " + std::to_string(std::min(out.size(), plain.size())) + " equal)");
    check(m.wire_errors() == 0, "MTP decode: no mailbox timeout");
  }

  // ---- 3b. --p1-kv-gate A,B: the KV-only pass 1 ----
  // MTP greedy decode as section 3, but output tokens [A, B) run as plain 1-row steps (as the engine
  // at 3-4 decoding requests) with the whole pass 1 (run 0) or the KV-only one (run 2; the last of
  // them drafts, so it runs the whole pass). Run 1 writes those positions' drafter KV from wrong
  // tokens first, so a KV-only pass that skipped a write would leave it there (run 0's KV of the same
  // text would otherwise hide it). Gate: every draft after them equal, run 2 == run 0; tokens per MTP
  // step before / after the plain steps.
  if (g_p1_kv_a >= 0 && g_p1_kv_b > g_p1_kv_a && generate > 0 && !pf_only) {
    const int ha = g_p1_kv_a, hb = g_p1_kv_b;
    std::vector<int32_t> drafts0;
    for (int run = 0; run < 3; ++run) {
      int32_t t = prefill(true);
      std::vector<int32_t> drafts = m.draft_chain(m.position() > 0 ? std::min(3, (prompt_len - 1) % 4) : 0, depth);
      std::vector<int32_t> out, all_drafts;
      int64_t st[2] = {0, 0}, tk[2] = {0, 0};  // MTP steps before / after the plain steps
      while (static_cast<int>(out.size()) < generate && m.position() + depth + 1 <= m.capacity()) {
        const int o = static_cast<int>(out.size());
        if (o >= ha && o < hb) {
          m.verify(&t, 1);
          const auto g = m.tokens(1);
          const bool last = o + 1 == hb;
          const int32_t bad = (g[0] + 7919) % m.config().vocab;
          if (run == 1 && !last) m.draft_pass1(&bad, 1);  // the wrong KV the KV-only run must overwrite
          else m.draft_pass1(g.data(), 1, run == 2 && !last);
          m.commit(1);
          out.push_back(t);
          t = g[0];
          drafts = last ? m.draft_chain(0, depth) : std::vector<int32_t>{};
          continue;
        }
        int32_t rows_t[4] = {t, 0, 0, 0};
        const int nd = static_cast<int>(drafts.size());
        for (int k = 0; k < nd; ++k) rows_t[k + 1] = drafts[static_cast<size_t>(k)];
        const int rows = 1 + nd;
        m.verify(rows_t, rows);
        const auto g = m.tokens(rows);
        int a = 0;
        while (a < nd && rows_t[a + 1] == g[static_cast<size_t>(a)]) ++a;
        m.draft_pass1(g.data(), rows);
        m.commit(a + 1);
        for (int k = 0; k <= a; ++k) out.push_back(rows_t[k]);
        const int w = o < ha ? 0 : 1;
        ++st[w];
        tk[w] += a + 1;
        t = g[static_cast<size_t>(a)];
        drafts = m.draft_chain(a, depth);
        all_drafts.insert(all_drafts.end(), drafts.begin(), drafts.end());
      }
      int same = 0;
      while (same < static_cast<int>(out.size()) && same < static_cast<int>(plain.size()) && out[static_cast<size_t>(same)] == plain[static_cast<size_t>(same)]) ++same;
      std::printf("p1-kv-gate run %d (%s pass 1 on plain steps [%d, %d)): tokens per MTP step before %.3f (%lld steps), after %.3f (%lld); == plain %d / %zu\n", run,
                  run == 0 ? "whole" : run == 1 ? "wrong-token" : "KV-only", ha, hb, st[0] ? static_cast<double>(tk[0]) / static_cast<double>(st[0]) : 0.0, static_cast<long long>(st[0]),
                  st[1] ? static_cast<double>(tk[1]) / static_cast<double>(st[1]) : 0.0, static_cast<long long>(st[1]), same, std::min(out.size(), plain.size()));
      check(same == static_cast<int>(std::min(out.size(), plain.size())), "p1-kv-gate run " + std::to_string(run) + ": MTP greedy tokens == plain greedy tokens");
      if (run == 0) drafts0 = all_drafts;
      else if (run == 1)
        check(all_drafts != drafts0, "p1-kv-gate: the wrong-token KV changes later drafts (the gate can see a missing write)");
      else if (run == 2)
        check(all_drafts == drafts0, "KV-only pass 1 on the plain steps: every later draft == the whole pass's (" + std::to_string(all_drafts.size()) + " drafts)");
    }
  }

  // ---- 4. the drafter in prefill ----
  // The first pf_prompt_len tokens through Model:prefill three ways on one load: drafter off (the
  // target alone: its time and the plain greedy reference after it), slices (pass 1 after
  // each 4-row slice's head) and rows (one many-row pass per chunk). Gates: MTP greedy after either
  // drafted prefill == plain greedy after the prefill; pass 1's prompt picks rows vs slices (fp32
  // summation order only); tokens per step from either. Timed: prefill + first drafts, median of 3.
  if (pf_prompt_len > 0 && generate > 0) {
    using PfMode = q38::Model::MtpPrefill;
    const std::vector<int32_t> pp(text.begin(), text.begin() + pf_prompt_len);
    const int last = (pf_prompt_len - 1) % 4;
    auto run_pf = [&](PfMode mode, std::vector<int32_t> *picks, std::vector<int32_t> *drafts) {
      m.set_mtp_prefill(mode);
      m.trace_mtp_prefill(picks);
      m.begin_sequence();
      const auto t0 = std::chrono::steady_clock::now();
      m.prefill(pp.data(), pf_prompt_len);
      const int32_t first = m.tokens(last + 1)[static_cast<size_t>(last)];
      if (mode != PfMode::off) {
        auto d = m.draft_after_prefill(first, depth);
        if (drafts) *drafts = d;
      }
      const double ms = ms_since(t0);
      m.trace_mtp_prefill(nullptr);
      return std::make_pair(first, ms);
    };
    // The MTP TTFT parts on this load. A run = prefill (rows or off) until the
    // first token is read (card 0's queue drained: the drafter pass included), then the first drafts
    // alone: greedy (emitted == pick, no pass-1 rerun) or forced rerun (emitted != pick, as sampled).
    // Drafter graphs recorded per run show whether a request re-records (keys: parities, indices);
    // this block runs first, so run 0 of the rows mode pays the first request's recording.
    {
      struct Part {
        double pf = 0, dr = 0;
        int graphs = 0;
        double rec_ms = 0;
      };
      auto part = [&](PfMode mode, bool rerun) {
        Part p;
        m.set_mtp_prefill(mode);
        m.begin_sequence();
        const int g0 = m.mtp_graphs_recorded();
        const double r0 = m.mtp_graph_record_ms();
        const auto t0 = std::chrono::steady_clock::now();
        m.prefill(pp.data(), pf_prompt_len);
        const int32_t first = m.tokens(last + 1)[static_cast<size_t>(last)];
        p.pf = ms_since(t0);
        if (mode != PfMode::off) {
          const auto t1 = std::chrono::steady_clock::now();
          m.draft_after_prefill(rerun ? (first + 1) % m.config().vocab : first, depth);
          p.dr = ms_since(t1);
        }
        p.graphs = m.mtp_graphs_recorded() - g0;
        p.rec_ms = m.mtp_graph_record_ms() - r0;
        return p;
      };
      const char *what[3] = {"off", "rows, greedy drafts", "rows, pass-1 rerun"};
      for (int i = 0; i < 3; ++i) {
        const PfMode mode = i == 0 ? PfMode::off : PfMode::rows;
        std::vector<double> pf, dr;
        std::string g;
        for (int r = 0; r < 6; ++r) {
          const Part p = part(mode, i == 2);
          g += (r ? " " : "") + std::to_string(p.graphs);
          if (r == 0) {
            std::printf("ttft parts, %s, run 0: prefill %.1f ms, first drafts %.1f ms, drafter graphs recorded %d (%.1f ms)\n", what[i], p.pf, p.dr, p.graphs,
                        p.rec_ms);
            continue;
          }
          pf.push_back(p.pf);
          dr.push_back(p.dr);
        }
        std::printf("ttft parts, %s, median of 5: prefill %.1f ms (min %.1f), first drafts %.1f ms (min %.1f); drafter graphs recorded per run: %s\n", what[i],
                    med(pf), med(pf, 0), med(dr), med(dr, 0), g.c_str());
      }
      m.set_mtp_prefill(PfMode::rows);
    }
    const char *names[3] = {"off", "slices", "rows"};
    const PfMode modes[3] = {PfMode::off, PfMode::slices, PfMode::rows};
    double pf_ms[3];
    for (int i = 0; i < 3; ++i) {
      std::vector<double> t;
      for (int r = 0; r < 4; ++r) t.push_back(run_pf(modes[i], nullptr, nullptr).second);  // run 0 records graphs: dropped
      t.erase(t.begin());
      pf_ms[i] = med(t);
    }
    std::printf("prefill of %d tokens (chunk %d) + first drafts, median of 3: drafter off %.1f ms, slices %.1f ms (+%.1f), rows %.1f ms (+%.1f); %.0f / %.0f / %.0f tok/s\n",
                pf_prompt_len, m.prefill_chunk(), pf_ms[0], pf_ms[1], pf_ms[1] - pf_ms[0], pf_ms[2], pf_ms[2] - pf_ms[0], pf_prompt_len / (pf_ms[0] / 1e3),
                pf_prompt_len / (pf_ms[1] / 1e3), pf_prompt_len / (pf_ms[2] / 1e3));
    // plain greedy after the prefill (the target alone)
    auto [t_plain, _] = run_pf(PfMode::off, nullptr, nullptr);
    std::vector<int32_t> ref;
    for (int32_t t = t_plain; static_cast<int>(ref.size()) < generate;) {
      ref.push_back(t);
      m.step(&t, 1);
      t = m.tokens(1)[0];
    }
    // the target's logits after the prefill (host sampling of the first token) must be the target's,
    // not the drafter's pass (the passes have their own logits buffer)
    {
      std::vector<float> lo[3];
      for (int i = 0; i < 3; ++i) {
        run_pf(modes[i], nullptr, nullptr);
        m.logits(last + 1, lo[i]);
      }
      check(lo[1] == lo[0] && lo[2] == lo[0], "target logits after a drafted prefill (slices, rows) == after the target alone (bit for bit)");
    }
    std::vector<int32_t> pk[3];
    // i = 3 = rows untraced (the server's path: the drafter's MLP mixer + MoE on the last slice only); its first
    // drafts and every step after == the traced rows run's (which runs every row)
    std::vector<int32_t> first_d[4];
    int64_t steps_of[4] = {0, 0, 0, 0};
    for (int i = 1; i < 4; ++i) {
      std::vector<int32_t> drafts;
      auto [t, ms] = run_pf(modes[std::min(i, 2)], i < 3 ? &pk[i] : nullptr, &drafts);
      first_d[i] = drafts;
      (void)ms;
      std::vector<int32_t> out;
      int64_t steps = 0, pos_acc[3] = {0, 0, 0};
      while (static_cast<int>(out.size()) < generate && m.position() + depth + 1 <= m.capacity()) {
        int32_t rows_t[4] = {t, 0, 0, 0};
        const int nd = static_cast<int>(drafts.size());
        for (int k = 0; k < nd; ++k) rows_t[k + 1] = drafts[static_cast<size_t>(k)];
        const int rows = 1 + nd;
        m.verify(rows_t, rows);
        const auto g = m.tokens(rows);
        int a = 0;
        while (a < nd && rows_t[a + 1] == g[static_cast<size_t>(a)]) ++a;
        m.draft_pass1(g.data(), rows);
        m.commit(a + 1);
        for (int k = 0; k <= a; ++k) out.push_back(rows_t[k]);
        for (int k = 0; k < a; ++k) ++pos_acc[k];
        ++steps;
        t = g[static_cast<size_t>(a)];
        drafts = m.draft_chain(a, depth);
      }
      out.resize(std::min<size_t>(out.size(), static_cast<size_t>(generate)));
      int same = 0;
      while (same < static_cast<int>(out.size()) && out[static_cast<size_t>(same)] == ref[static_cast<size_t>(same)]) ++same;
      steps_of[i] = steps;
      const char *nm = i < 3 ? names[i] : "rows (untraced)";
      std::printf("MTP-%d greedy after a %s prefill: %zu tokens in %lld steps = %.3f tokens per step (per draft position %.2f %.2f %.2f)\n", depth, nm,
                  out.size(), static_cast<long long>(steps), static_cast<double>(out.size()) / static_cast<double>(steps), static_cast<double>(pos_acc[0]) / steps,
                  static_cast<double>(pos_acc[1]) / steps, static_cast<double>(pos_acc[2]) / steps);
      check(same == static_cast<int>(out.size()), std::string("MTP greedy after a ") + nm + " prefill == plain greedy after the prefill (" + std::to_string(same) +
                                                     " / " + std::to_string(out.size()) + " equal)");
    }
    check(first_d[3] == first_d[2] && steps_of[3] == steps_of[2], "rows prefill untraced (KV-only before the last slice) == traced: first drafts and steps (round 110)");
    int eq = 0;
    const size_t n = std::min(pk[1].size(), pk[2].size());
    for (size_t i = 0; i < n; ++i) eq += pk[1][i] == pk[2][i];
    const double agree = n ? 100.0 * eq / static_cast<double>(n) : 0.0;
    std::printf("pass-1 picks over the prompt, rows vs slices: %d / %zu equal (%.2f %%)\n", eq, n, agree);
    check(n == static_cast<size_t>(pf_prompt_len) && pk[1].size() == pk[2].size(), "pass-1 picks: one per prompt row in both modes");
    check(agree >= 95.0, "pass-1 picks rows vs slices >= 95 % equal (fp32 summation order only)");
    check(m.wire_errors() == 0, "drafter in prefill: no mailbox timeout");
    m.set_mtp_prefill(PfMode::rows);
  }
  std::printf("step graphs: target %d recorded (%.1f ms recording + finalizing), drafter passes %d (%.1f ms)\n", m.graphs_recorded(), m.graph_record_ms(),
              m.mtp_graphs_recorded(), m.mtp_graph_record_ms());
  return rc;
}
