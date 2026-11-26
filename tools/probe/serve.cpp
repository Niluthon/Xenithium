// xenithium-probe serve: the serving engine's step loop on the real model, one load.
// xenithium-probe serve --model DIR --ppl F [--seqs N] [--prompt-len P] [--generate G] [--context C]
// [--pf-chunk N] [--hot-rank F] [--reserve G] [--graphs 0|1] [--kv-bytes 1|2] [--stall-ms MS]
// [--lp-time N] (only the prompt-logprob timing + records hash, then exit)
// Prompt i (K = 2 N + 1 of them) = P + 97 i tokens of the token file's text from offset 700 i, so the
// prompts differ in text and length and the longer ones need several prefill chunks.
// 1 reference each prompt alone, straight on the model: prefill, then G greedy 1-row steps.
// 2 engine all K prompts submitted at once to core:BatchEngine over N slots (the rest wait and
// reuse freed slots; prefill chunks interleave with shared decode steps): every
// request's tokens must equal its reference, token for token.
// 3 seeded two requests with the same seed at T 0.7 on two slots: identical tokens (the host
// sampler sees bit-identical logits rows), logprobs on every token.
// 4 timing n = 1..N concurrent requests (prompt P, 4 G tokens): per-request decode tok/s, the
// aggregate, and TTFT; against the model-level step time.
// 5 prompt logprobs prompt 0 with prompt_logprobs 5 through the engine against the same
// tokens teacher-forced in 4-row decode steps: one entry per token, top-1 agreement and
// the target logprob difference (prefill vs decode numerics); the prefill's time with and
// without every row's head. Logprobs on the cards (prompt and greedy output) ==
// the host's from full logits rows; timed against the host path.
// The engine runs with the stall budget (--stall-ms, default 100 as the server): while requests
// decode, prefill calls are sized below a full chunk, at other boundaries than the reference's.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/arch/qwen38/step_model.hpp"
#include "xenithium/core/batch_engine.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/device.hpp"
#include "xenithium/util/env.hpp"
#include "kv.hpp"

#include "probe.hpp"  // the run-wide flags
namespace q38 = xenithium::arch::qwen38;
namespace core = xenithium::core;

namespace {

bool read_tokens(const std::string &path, std::vector<int32_t> &out) {
  std::ifstream f(path, std::ios::binary);
  int32_t n = 0;
  if (!f || !f.read(reinterpret_cast<char *>(&n), 4) || n < 1) return false;
  for (int32_t c = 0; c < n; ++c) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len < 1) return false;
    const size_t at = out.size();
    out.resize(at + static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(out.data() + at), static_cast<std::streamsize>(len) * 4)) return false;
  }
  return true;
}

using clk = std::chrono::steady_clock;

struct Sink final : core::EventSink {
  std::mutex mu;
  std::condition_variable cv;
  std::map<uint64_t, std::vector<int32_t>> toks;
  std::map<uint64_t, int> lps;
  std::map<uint64_t, clk::time_point> first, last;
  std::map<uint64_t, std::vector<std::pair<clk::time_point, int>>> at;  // (event time, tokens)
  std::map<uint64_t, core::Finish> fin;
  void on_events(const core::Event *e, size_t n) override {
    const auto now = clk::now();
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < n; ++i) {
      auto &t = toks[e[i].id];
      if (e[i].n && t.empty()) first[e[i].id] = now;
      if (e[i].n) last[e[i].id] = now;
      if (e[i].n) at[e[i].id].push_back({now, e[i].n});
      t.insert(t.end(), e[i].tokens, e[i].tokens + e[i].n);
      lps[e[i].id] += static_cast<int>(e[i].logprobs.size());
      if (e[i].finish != core::Finish::none) fin[e[i].id] = e[i].finish;
    }
    cv.notify_all();
  }
  void wait(size_t n) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return fin.size() >= n; });
  }
};

// Forwards to the model and records the prefill calls per sequence: with the stall budget
// the engine splits prompts at other boundaries than one call, and prefill numerics depend on the
// boundaries, so the reference replays the engine's own splits.
struct Recorder final : core::StepModel {
  core::StepModel &m;
  std::map<int, std::vector<int>> cur;  // slot -> the calls of its current sequence
  std::map<size_t, std::vector<int>> by_len;  // prompt length -> its calls
  explicit Recorder(core::StepModel &x) : m(x) {}
  int slots() const override { return m.slots(); }
  int64_t capacity() const override { return m.capacity(); }
  int64_t vocab() const override { return m.vocab(); }
  int max_rows() const override { return m.max_rows(); }
  int prefill_chunk() const override { return m.prefill_chunk(); }
  int snap_align() const override { return m.snap_align(); }  // the engine sizes calls by it (198)
  void begin(int s) override {
    close(s);
    m.begin(s);
  }
  void prefill(int s, const int32_t *t, int n, int32_t next) override {
    cur[s].push_back(n);
    m.prefill(s, t, n, next);
  }
  int split_calls = 0, parts = 0;
  int prefill_layers() const override { return m.prefill_layers(); }
  void prefill_begin(int s, const int32_t *t, int n, int32_t next) override {
    cur[s].push_back(n);
    ++split_calls;
    m.prefill_begin(s, t, n, next);
  }
  bool prefill_advance(int layers) override {
    ++parts;
    return m.prefill_advance(layers);
  }
  void prefill_abort() override { m.prefill_abort(); }
  bool prompt_logprobs() const override { return m.prompt_logprobs(); }
  void prefill_rows(int s, const int32_t *t, int n, int32_t next, const RowsFn &f) override {
    cur[s].push_back(n);
    m.prefill_rows(s, t, n, next, f);
  }
  bool device_logprobs() const override { return m.device_logprobs(); }
  void row_logprobs(int rows, const int32_t *t, int top, core::TokenLogprob *out) override { m.row_logprobs(rows, t, top, out); }
  void prefill_logprobs(int32_t tok, int top, core::TokenLogprob &out) override { m.prefill_logprobs(tok, top, out); }
  void prefill_each(int s, const int32_t *t, int n, int32_t next, const EachFn &f) override {
    cur[s].push_back(n);
    m.prefill_each(s, t, n, next, f);
  }
  bool prefill_row_logprobs(int s, const int32_t *t, int n, int32_t next, const int32_t *tg, int top, core::TokenLogprob *out) override {
    if (!m.prefill_row_logprobs(s, t, n, next, tg, top, out)) return false;
    cur[s].push_back(n);
    return true;
  }
  int draft_depth(int n) const override { return m.draft_depth(n); }
  double spec_cost(int n, int rows) const override { return m.spec_cost(n, rows); }
  std::vector<int32_t> first_drafts(int s, int32_t first, int d, int n) override { return m.first_drafts(s, first, d, n); }
  void draft_sampling(int s, uint64_t key, uint64_t pos, float inv_t) override { m.draft_sampling(s, key, pos, inv_t); }
  void verify(const std::vector<SpecSeg> &segs) override { m.verify(segs); }
  std::vector<std::vector<int32_t>> accept(const std::vector<SpecAccept> &a) override { return m.accept(a); }
  int32_t prefill_pick() override { return m.prefill_pick(); }
  void prefill_logits(std::vector<float> &row) override { m.prefill_logits(row); }
  void step(const std::vector<int> &sl, const int32_t *t) override { m.step(sl, t); }
  void picks(int rows, int32_t *out) override { m.picks(rows, out); }
  void logits(int rows, std::vector<float> &out) override { m.logits(rows, out); }
  void close(int s) {
    auto &c = cur[s];
    if (c.empty()) return;
    size_t len = 0;
    for (int n : c) len += static_cast<size_t>(n);
    by_len[len] = c;
    c.clear();
  }
  void close_all() {
    for (auto &[s, c] : cur) close(s);
  }
};

double med(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int cmd_serve_run(const std::string &model, const std::string &ppl, int nseq, int prompt_len, int generate, int64_t context, int pf_chunk,
                  bool graphs, const std::string &hot_rank, double reserve_gib, int threads, int kv_bytes, double stall_ms,
                  const std::string &mtp) {
  if (model.empty() || ppl.empty()) {
    std::fprintf(stderr, "serve: --model DIR and --ppl F are required\n");
    return 2;
  }
  std::vector<int32_t> text;
  if (!read_tokens(ppl, text) || text.size() < 64) {
    std::fprintf(stderr, "serve: cannot read token file %s\n", ppl.c_str());
    return 2;
  }
  const int N = std::clamp(nseq, 1, q38::Head::kRows),  // 1..16 slots
             P = prompt_len > 0 ? prompt_len : 512, G = generate > 0 ? generate : 32, K = 2 * N + 1;
  auto prompt_of = [&](int i, int len) {
    std::vector<int32_t> p(static_cast<size_t>(len));
    for (int k = 0; k < len; ++k) p[static_cast<size_t>(k)] = text[static_cast<size_t>((700LL * i + k) % static_cast<int64_t>(text.size()))];
    return p;
  };
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
  mo.slots = N;
  mo.context = std::max<int64_t>(context, P + 97LL * K + 4LL * G + 64);
  mo.kv = probe_kv(kv_bytes);
  mo.graphs = graphs;
  mo.threads = threads;
  if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;
  mo.mtp_dir = mtp;  // MTP through the engine (one slot)
  mo.mtp_vocab = g_q38_mtp_vocab;
  mo.mtp_head_w4 = g_q38_mtp_head_w4 != 0;
  if (reserve_gib >= 0) mo.reserve_gib = reserve_gib;
  if (!hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(model));
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, tp_devices("serve", true));  // --tp 1 (device 0)
  q38::Model m(loader, fab, model, mo);
  q38::StepModelAdapter ad(m);
  ad.set_step_rows(g_step_rows);
  ad.set_two_deep(g_two_deep != 0);
  if (const char *e = std::getenv("XEN_MTP_DEPTH"); e && *e) ad.set_depth(std::atoi(e));  // A/B
  std::printf("serve: loaded in %.1f s; %d slots x context %lld, prefill chunk %d; hot experts %lld (card 0); graphs %s; step rows %d\n", m.load_stats().total_s,
              m.slots(), static_cast<long long>(m.capacity()), m.prefill_chunk(), static_cast<long long>(m.load_stats().hot[0]), graphs ? "on" : "off", ad.max_rows());
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };

  // --lp-time N: prompt 0 (P tokens) prefilled plain and with prompt logprobs
  // (top 5) on the many-row head, alternating, N runs each; medians, and an FNV hash of every row's
  // logprob / rank / top ids / values so variants can be compared bit for bit. Then exit.
  if (g_lp_time > 0) {
    const auto p = prompt_of(0, P);
    std::vector<int32_t> tg(static_cast<size_t>(P));
    for (int j = 0; j < P; ++j) tg[static_cast<size_t>(j)] = j + 1 < P ? p[static_cast<size_t>(j + 1)] : -1;
    std::vector<double> tp, tp2, tl;
    uint64_t hash = 0;
    for (int k = 0; k < g_lp_time + 1; ++k) {  // run 0 warms up both paths
      auto plain_run = [&] {
        ad.begin(0);
        const auto a = clk::now();
        ad.prefill(0, p.data(), P, -1);
        ad.prefill_pick();
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
      };
      const double plain = plain_run();  // after the previous iteration's logprob call
      const double plain2 = plain_run();  // after a plain call
      std::vector<core::TokenLogprob> all(static_cast<size_t>(P));
      ad.begin(0);
      auto a = clk::now();
      if (!ad.prefill_row_logprobs(0, p.data(), P, -1, tg.data(), 5, all.data())) {
        std::printf("lp-time: no many-row head (EP2)\n");
        return 1;
      }
      ad.prefill_pick();
      const double lp = std::chrono::duration<double, std::milli>(clk::now() - a).count();
      if (k == 0) continue;
      tp.push_back(plain);
      tp2.push_back(plain2);
      tl.push_back(lp);
      uint64_t h = 1469598103934665603ull;
      auto mix = [&](const void *d, size_t n) {
        for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t *>(d)[i]) * 1099511628211ull;
      };
      for (const auto &e : all) {
        mix(&e.logprob, 4);
        mix(&e.rank, 4);
        for (const auto &t : e.top) {
          mix(&t.first, 4);
          mix(&t.second, 4);
        }
      }
      if (k == 1) hash = h;
      if (h != hash) std::printf("lp-time: records differ between runs (run %d)\n", k);
    }
    std::printf("lp-time: %d-token prefill plain %.2f ms (after a plain call; %.2f after a logprob call), with prompt logprobs (top 5, many-row head) %.2f ms, +%.2f ms (median of %d); records hash %016llx\n",
                P, med(tp2), med(tp), med(tl), med(tl) - med(tp2), g_lp_time, static_cast<unsigned long long>(hash));
    return 0;
  }
  // XEN_PF_BISECT=1 (diagnostic): the first module whose output rows differ between one
  // prefill call of P rows and a first call of 64 / 192 rows (same prompt, same start state).
  if (xenithium::util::env_flag("XEN_PF_BISECT")) {
    auto bf = [](uint16_t b) {
      uint32_t u = static_cast<uint32_t>(b) << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    std::vector<int> pieces{64, 192};  // XEN_PF_BISECT_PIECES=a,b... (e.g. 256)
    if (const char *e = std::getenv("XEN_PF_BISECT_PIECES"); e && *e) {
      pieces.clear();
      for (const char *s = e; *s;) {
        pieces.push_back(std::atoi(s));
        while (*s && *s != ',') ++s;
        if (*s) ++s;
      }
    }
    // XEN_PF_BISECT_AT=A: both calls start at position A (the prompt's first A tokens prefilled before in
    // calls of 512 rows)
    const char *at_env = std::getenv("XEN_PF_BISECT_AT");
    const int A = at_env && *at_env ? std::atoi(at_env) : 0;
    const auto pa = prompt_of(0, A + P);
    auto lead = [&] {
      ad.begin(0);
      for (int s = 0; s < A; s += 512) {
        const int n = std::min(512, A - s);
        ad.prefill(0, pa.data() + s, n, pa[static_cast<size_t>(s + n)]);
      }
    };
    // XEN_PF_BISECT_SECOND=1: the call AFTER a first call of `piece` rows (rows [A + piece, A + P)) vs the same rows
    // inside one call of P rows at A
    const bool second = xenithium::util::env_flag("XEN_PF_BISECT_SECOND");
    for (int piece : pieces) {
      if (piece >= P) continue;
      q38::Model::PfTrace whole, split;
      const int n = second ? P - piece : piece, off = second ? piece : 0;
      whole.rows = P;
      split.rows = n;
      lead();
      m.trace_prefill(&whole);
      ad.prefill(0, pa.data() + A, P, -1);
      m.trace_prefill(nullptr);
      lead();
      if (second) {
        ad.prefill(0, pa.data() + A, piece, pa[static_cast<size_t>(A + piece)]);
        m.trace_prefill(&split);
        ad.prefill(0, pa.data() + A + piece, n, -1);
      } else {
        m.trace_prefill(&split);
        ad.prefill(0, pa.data() + A, piece, pa[static_cast<size_t>(A + piece)]);
      }
      m.trace_prefill(nullptr);
      std::printf("bisect: one call of %d at %d vs %s call of %d rows (rows %d..%d), %zu / %zu trace points\n", P, A, second ? "a second" : "a first", n, A + off,
                  A + off + n, whole.names.size(), split.names.size());
      int shown = 0;
      for (size_t i = 0; i < std::min(whole.names.size(), split.names.size()) && shown < 12; ++i) {
        const auto &a = whole.data[i], &b = split.data[i];
        const int w = whole.widths[i];
        int first = -1, nrows = 0;
        double maxd = 0;
        for (int r = 0; r < n; ++r) {
          bool d = false;
          for (int j = 0; j < w; ++j) {
            const size_t ka = static_cast<size_t>(r + off) * w + j, kb = static_cast<size_t>(r) * w + j;
            if (a[ka] != b[kb]) {
              d = true;
              maxd = std::max(maxd, static_cast<double>(std::fabs(bf(a[ka]) - bf(b[kb]))));
            }
          }
          if (d) {
            ++nrows;
            if (first < 0) first = r;
          }
        }
        if (nrows) {
          std::printf("  %-18s %3d of %d rows differ, the first %d; max |d| %.3g\n", whole.names[i].c_str(), nrows, n, first, maxd);
          ++shown;
        }
      }
      if (!shown) std::printf("  no trace point differs\n");
    }
    return 0;
  }

  // ---- 1 reference: each prompt alone, the whole prompt in one prefill call ----
  std::vector<std::vector<int32_t>> ref(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i) {
    const auto p = prompt_of(i, P + 97 * i);
    ad.begin(0);
    ad.prefill(0, p.data(), static_cast<int>(p.size()), -1);
    int32_t t = ad.prefill_pick();
    auto &r = ref[static_cast<size_t>(i)];
    r.push_back(t);
    for (int g = 1; g < G; ++g) {
      ad.step({0}, &t);
      ad.picks(1, &t);
      r.push_back(t);
    }
  }
  std::printf("reference: %d prompts (%d..%d tokens) x %d greedy tokens; prompt 0:", K, P, P + 97 * (K - 1), G);
  for (int g = 0; g < std::min(G, 12); ++g) std::printf(" %d", ref[0][static_cast<size_t>(g)]);
  std::printf("\n");

  core::BatchConfig bc;
  bc.eos_ids = {};  // run every request to max_tokens: the comparison needs whole sequences
  bc.log_interval_s = 0;
  bc.prefill_stall_ms = stall_ms;
  if (const char *e = std::getenv("XEN_ADAPTIVE_DEPTH"); e && *e) bc.adaptive_depth = *e == '1';  // A/B
  if (const char *e = std::getenv("XEN_LOOKUP"); e && *e) bc.lookup = std::atoi(e);  // A/B

  // ---- 2 engine: all K at once on N slots; the reference replays each prompt's prefill calls ----
  {
    Recorder rec(ad);
    core::BatchEngine e(rec, bc);
    Sink s;
    e.start(&s);
    for (int i = 0; i < K; ++i) {
      core::SamplingParams sp;
      sp.temperature = 0;
      sp.max_tokens = G;
      e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, P + 97 * i), sp);
    }
    s.wait(static_cast<size_t>(K));
    e.stop();
    rec.close_all();
    int eq = 0, eq_whole = 0, split = 0;
    for (int i = 0; i < K; ++i) {
      const auto p = prompt_of(i, P + 97 * i);
      const auto &calls = rec.by_len[p.size()];
      split += calls.size() > static_cast<size_t>((p.size() + 511) / 512);
      std::vector<int32_t> r;
      ad.begin(0);
      size_t at = 0;
      for (int n : calls) ad.prefill(0, p.data() + at, n, at + static_cast<size_t>(n) < p.size() ? p[at + static_cast<size_t>(n)] : -1), at += static_cast<size_t>(n);
      int32_t t = ad.prefill_pick();
      r.push_back(t);
      for (int g = 1; g < G; ++g) {
        ad.step({0}, &t);
        ad.picks(1, &t);
        r.push_back(t);
      }
      const auto &got = s.toks[static_cast<uint64_t>(i + 1)];
      const bool ok = at == p.size() && got == r;
      eq += ok;
      eq_whole += got == ref[static_cast<size_t>(i)];
      if (got != ref[static_cast<size_t>(i)]) {  // the calls of a prompt that differs from its one-call reference
        std::string cs;
        for (int n : calls) cs += " " + std::to_string(n);
        size_t d = 0;
        while (d < std::min(got.size(), ref[static_cast<size_t>(i)].size()) && got[d] == ref[static_cast<size_t>(i)][d]) ++d;
        std::printf("  prompt %d (%zu tokens) vs one call: first difference at token %zu; its calls:%s\n", i, p.size(), d, cs.c_str());
      }
      if (!ok) {
        size_t d = 0;
        while (d < std::min(got.size(), r.size()) && got[d] == r[d]) ++d;
        std::printf("  prompt %d: first difference at token %zu of %zu (got %zu tokens)\n", i, d, r.size(), got.size());
      }
    }
    std::printf("engine: %d of %d prompts were prefilled in more calls than one per 512-row chunk (stall budget %.0f ms); %d / %d equal the one-call reference\n", split, K,
                stall_ms, eq_whole, K);
    // prefill rows no longer depend on the call boundaries, so the plain one-call
    // reference must hold too (needed the Recorder's replay)
    check(eq_whole == K, "engine: budgeted prefill calls == the one-call reference, token for token (" + std::to_string(eq_whole) + " / " + std::to_string(K) +
                             ", " + std::to_string(split) + " prompts split)");
    if (ad.draft_depth(1) > 0) {
      const auto st = e.stats();
      std::printf("engine MTP (depth %d / %d / %d for 1 / 2 / 3+ sequences): %llu speculative segments, %.3f tokens per segment (accepted drafts per position %.2f %.2f %.2f)\n", ad.draft_depth(1), ad.draft_depth(2), ad.draft_depth(3),
                  static_cast<unsigned long long>(st.spec_drafts), st.spec_drafts ? 1.0 + static_cast<double>(st.spec_accepted_tokens) / st.spec_drafts : 0.0,
                  st.spec_drafts ? static_cast<double>(st.spec_accepted_at[0]) / st.spec_drafts : 0.0, st.spec_drafts ? static_cast<double>(st.spec_accepted_at[1]) / st.spec_drafts : 0.0,
                  st.spec_drafts ? static_cast<double>(st.spec_accepted_at[2]) / st.spec_drafts : 0.0);
    }
    check(eq == K, "engine: " + std::to_string(K) + " requests on " + std::to_string(N) + (ad.draft_depth(1) > 0 ? " slots with MTP (speculative steps)" : " slots") +
                       " (queued, budgeted prefill calls between shared decode steps) == alone with the same prefill calls, token for token (" + std::to_string(eq) +
                       " / " + std::to_string(K) + ")");
  }
  // ---- 2b: short prompts (40..336 tokens, each one call) on N slots with prefill calls in
  // parts (prefill_part_ms 8: a prompt admitted beside a decoding request splits), against each
  // prompt's one-call reference ----
  {
    auto len_of = [](int i) { return 40 + 37 * i; };
    std::vector<std::vector<int32_t>> ref2(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) {
      const auto p = prompt_of(i, len_of(i));
      ad.begin(0);
      ad.prefill(0, p.data(), static_cast<int>(p.size()), -1);
      int32_t t = ad.prefill_pick();
      auto &r = ref2[static_cast<size_t>(i)];
      r.push_back(t);
      for (int g = 1; g < G; ++g) {
        ad.step({0}, &t);
        ad.picks(1, &t);
        r.push_back(t);
      }
    }
    core::BatchConfig b2 = bc;
    b2.prefill_part_ms = 8;
    Recorder rec(ad);
    core::BatchEngine e(rec, b2);
    Sink s;
    e.start(&s);
    for (int i = 0; i < K; ++i) {
      core::SamplingParams sp;
      sp.temperature = 0;
      sp.max_tokens = G;
      e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, len_of(i)), sp);
    }
    s.wait(static_cast<size_t>(K));
    e.stop();
    int eq = 0;
    for (int i = 0; i < K; ++i) eq += s.toks[static_cast<uint64_t>(i + 1)] == ref2[static_cast<size_t>(i)];
    std::printf("engine, split calls: %d prefill calls ran in %d parts in all; %d / %d requests equal the one-call reference\n", rec.split_calls, rec.parts, eq, K);
    check(eq == K && rec.split_calls > 0, "engine: prefill calls in parts (decode steps between them) == the one-call reference, token for token (" + std::to_string(eq) + " / " +
                                              std::to_string(K) + ", " + std::to_string(rec.split_calls) + " calls split)");
  }
  // ---- 2c: one call in parts on slot 0 with slot 1's decode steps between them: slot 0's
  // logits == one call's bit for bit, slot 1's tokens == its steps without the call ----
  if (N >= 2) {
    const auto p = prompt_of(0, P), q = prompt_of(1, P);
    std::vector<float> w, g;
    ad.begin(0);
    ad.prefill(0, p.data(), P, -1);
    ad.prefill_logits(w);
    const int32_t w_pick = ad.prefill_pick();
    const int L = ad.prefill_layers(), per = 6;
    std::vector<int32_t> t1_ref, t1;
    auto start1 = [&]() {
      ad.begin(1);
      ad.prefill(1, q.data(), P, -1);
      return ad.prefill_pick();
    };
    int32_t t = start1();
    for (int k = 0; k <= L / per; ++k) {
      ad.step({1}, &t);
      ad.picks(1, &t);
      t1_ref.push_back(t);
    }
    t = start1();
    ad.begin(0);
    ad.prefill_begin(0, p.data(), P, -1);
    int parts = 0;
    while (!ad.prefill_advance(per)) {
      ++parts;
      ad.step({1}, &t);
      ad.picks(1, &t);
      t1.push_back(t);
    }
    ad.prefill_logits(g);
    const int32_t g_pick = ad.prefill_pick();
    const bool same1 = std::equal(t1.begin(), t1.end(), t1_ref.begin());
    std::printf("split call: %d decode steps of slot 1 between its parts; slot 0 logits %s, pick %d vs %d; slot 1 tokens %s\n", parts, w == g ? "equal" : "DIFFER", g_pick, w_pick,
                same1 ? "equal" : "DIFFER");
    check(parts > 0 && w == g && g_pick == w_pick && same1, "split call: " + std::to_string(P) + " tokens in parts of " + std::to_string(per) +
                                                               " layers with another slot's decode steps between == one call (last row bit for bit), the other slot unchanged");
  }
  // ---- 3 seeded sampling on two slots (whole prefill calls: the same numerics on both) ----
  if (N >= 2) {
    core::BatchConfig b0 = bc;
    b0.prefill_stall_ms = 0;
    core::BatchEngine e(ad, b0);
    Sink s;
    e.start(&s);
    core::SamplingParams sp;
    sp.temperature = 0.7f;
    sp.top_p = 0.95f;
    sp.top_k = 20;
    sp.has_seed = true;
    sp.seed = 4242;
    sp.max_tokens = G;
    sp.logprobs = 1;
    e.submit(1, prompt_of(1, P), sp);
    e.submit(2, prompt_of(1, P), sp);
    s.wait(2);
    e.stop();
    check(s.toks[1] == s.toks[2] && static_cast<int>(s.toks[1].size()) == G && s.lps[1] == G,
          "seeded: T 0.7 top_k 20 top_p 0.95, same seed on two slots -> identical tokens, a logprob per token");
  }
  // ---- 3b MTP: greedy and sampled speed through the engine, 1-4 concurrent requests (the
  // steady window: every request decoding; 3-4 as well, MTP-1 each in an 8-row step) ----
  if (ad.draft_depth(1) > 0) {
    for (int n = 1; n <= std::min(4, N); ++n)
      for (float temp : {0.0f, 0.7f}) {
        core::BatchEngine e(ad, bc);
        Sink s;
        e.start(&s);
        core::SamplingParams sp;
        sp.temperature = temp;
        sp.top_k = 20;
        sp.top_p = 0.95f;
        sp.has_seed = true;
        sp.seed = 7;
        sp.max_tokens = 4 * G;
        for (int i = 0; i < n; ++i) e.submit(static_cast<uint64_t>(1 + i), prompt_of(i, P), sp);
        s.wait(static_cast<size_t>(n));
        e.stop();
        const auto st = e.stats();
        clk::time_point a{}, b = clk::time_point::max();
        for (int i = 0; i < n; ++i) a = std::max(a, s.first[static_cast<uint64_t>(1 + i)]), b = std::min(b, s.last[static_cast<uint64_t>(1 + i)]);
        size_t inside = 0;
        for (int i = 0; i < n; ++i)
          for (const auto &[t, k] : s.at[static_cast<uint64_t>(1 + i)])
            if (t > a && t <= b) inside += static_cast<size_t>(k);
        const double secs = std::chrono::duration<double>(b - a).count();
        std::printf("MTP through the engine, %d request%s, T %.1f: %.1f tok/s in all while every request decodes; %.3f tokens per segment (per draft position %.2f %.2f %.2f)\n",
                    n, n > 1 ? "s" : "", temp, secs > 0 ? inside / secs : 0.0, st.spec_drafts ? 1.0 + static_cast<double>(st.spec_accepted_tokens) / st.spec_drafts : 0.0,
                    st.spec_drafts ? static_cast<double>(st.spec_accepted_at[0]) / st.spec_drafts : 0.0, st.spec_drafts ? static_cast<double>(st.spec_accepted_at[1]) / st.spec_drafts : 0.0,
                    st.spec_drafts ? static_cast<double>(st.spec_accepted_at[2]) / st.spec_drafts : 0.0);
      }
  }
  // ---- 4 timing ----
  {
    const int GT = 4 * G;
    std::printf("timing: n concurrent requests, prompt %d, %d tokens each (graphs %s)\n", P, GT, graphs ? "on" : "off");
    std::printf("  %-4s %10s %14s %12s %16s %16s %14s\n", "n", "TTFT ms", "tok/s per req", "tok/s all", "steady ms/step", "model ms/step", "engine extra");
    for (int n = 1; n <= N; ++n) {
      core::BatchEngine e(ad, bc);
      Sink s;
      e.start(&s);
      const auto t0 = clk::now();
      for (int i = 0; i < n; ++i) {
        core::SamplingParams sp;
        sp.temperature = 0;
        sp.max_tokens = GT;
        e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, P), sp);
      }
      s.wait(static_cast<size_t>(n));
      e.stop();
      std::vector<double> ttft, per;
      clk::time_point all_first = clk::time_point::max(), all_last{};
      for (int i = 0; i < n; ++i) {
        const auto id = static_cast<uint64_t>(i + 1);
        ttft.push_back(std::chrono::duration<double, std::milli>(s.first[id] - t0).count());
        per.push_back((GT - 1) / std::chrono::duration<double>(s.last[id] - s.first[id]).count());
        all_first = std::min(all_first, s.first[id]);
        all_last = std::max(all_last, s.last[id]);
      }
      const double all = n * (GT - 1) / std::chrono::duration<double>(all_last - all_first).count();
      // steady state: every request decoding (after the last first token), until the first finishes
      clk::time_point s0{}, s1 = clk::time_point::max();
      for (int i = 0; i < n; ++i) {
        s0 = std::max(s0, s.first[static_cast<uint64_t>(i + 1)]);
        s1 = std::min(s1, s.last[static_cast<uint64_t>(i + 1)]);
      }
      int steps = 0;
      for (const auto &[t, k] : s.at[1])
        if (t > s0 && t <= s1) ++steps;
      const double steady = steps ? std::chrono::duration<double, std::milli>(s1 - s0).count() / steps : 0;
      // the same n sequences stepped straight on the model (no engine), as probe seqs times it
      std::vector<double> ms;
      {
        std::vector<int> sl;
        std::vector<int32_t> tk(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
          const auto p = prompt_of(i, P);
          ad.begin(i);
          ad.prefill(i, p.data(), P, -1);
          tk[static_cast<size_t>(i)] = ad.prefill_pick();
          sl.push_back(i);
        }
        for (int g = 0; g < GT; ++g) {
          const auto a = clk::now();
          ad.step(sl, tk.data());
          ad.picks(n, tk.data());
          ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - a).count());
        }
      }
      std::printf("  %-4d %10.1f %14.1f %12.1f %16.2f %16.2f %14.2f\n", n, med(ttft), med(per), all, steady, med(ms), steady - med(ms));
    }
  }
  // ---- 5 prompt logprobs ----
  {
    const auto p = prompt_of(0, P);
    // one request through the engine: prompt logprobs 5 (+ greedy logprobs 5 on G output tokens)
    struct PlSink final : core::EventSink {
      std::mutex mu;
      std::condition_variable cv;
      std::vector<core::TokenLogprob> plp, lp;
      std::vector<int32_t> toks;
      bool done = false;
      clk::time_point first{}, last{};
      void on_events(const core::Event *ev, size_t n) override {
        std::lock_guard<std::mutex> lk(mu);
        for (size_t i = 0; i < n; ++i) {
          if (!ev[i].prompt_logprobs.empty()) plp = ev[i].prompt_logprobs;
          lp.insert(lp.end(), ev[i].logprobs.begin(), ev[i].logprobs.end());
          if (toks.empty()) first = clk::now();
          toks.insert(toks.end(), ev[i].tokens, ev[i].tokens + ev[i].n);
          last = clk::now();
          done |= ev[i].finish != core::Finish::none;
        }
        cv.notify_all();
      }
    };
    struct Run {
      std::vector<core::TokenLogprob> plp, lp;
      std::vector<int32_t> toks;
      double ms = 0, decode_tps = 0;
    };
    auto run = [&](bool device, int prompt_lp, int lp, int max_tokens, bool rows_lp = true) {
      ad.set_device_logprobs(device);
      ad.set_rows_logprobs(rows_lp);
      core::BatchEngine e(ad, bc);
      PlSink s;
      e.start(&s);
      core::SamplingParams sp;
      sp.temperature = 0;
      sp.max_tokens = max_tokens;
      sp.prompt_logprobs = prompt_lp;
      sp.logprobs = lp;
      const auto t0 = clk::now();
      e.submit(1, p, sp);
      {
        std::unique_lock<std::mutex> lk(s.mu);
        s.cv.wait(lk, [&] { return s.done; });
      }
      Run r;
      r.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      e.stop();
      r.plp = s.plp;
      r.lp = s.lp;
      r.toks = s.toks;
      const double dt = std::chrono::duration<double>(s.last - s.first).count();
      r.decode_tps = dt > 0 ? (static_cast<double>(s.toks.size()) - 1) / dt : 0;
      return r;
    };
    // device vs host logprobs: the same rows (bf16 logits), so ranks and top ids must agree and the
    // values within fp32 summation order of the normaliser
    auto cmp = [](const std::vector<core::TokenLogprob> &a, const std::vector<core::TokenLogprob> &b, int &bad_rank, int &bad_top, double &max_d) {
      bad_rank = bad_top = 0;
      max_d = 0;
      for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (std::isnan(a[i].logprob) && std::isnan(b[i].logprob)) continue;
        max_d = std::max(max_d, std::fabs(static_cast<double>(a[i].logprob) - b[i].logprob));
        bad_rank += a[i].rank != b[i].rank;
        bool t = a[i].top.size() == b[i].top.size();
        for (size_t k = 0; t && k < a[i].top.size(); ++k) {
          t &= a[i].top[k].first == b[i].top[k].first;
          max_d = std::max(max_d, std::fabs(static_cast<double>(a[i].top[k].second) - b[i].top[k].second));
        }
        bad_top += !t;
      }
      return a.size() == b.size();
    };
    const Run hp = run(false, 5, -1, 1), dp = run(true, 5, -1, 1);
    int br = 0, bt = 0;
    double md = 0;
    const bool same_n = cmp(hp.plp, dp.plp, br, bt, md);
    std::printf("prompt logprobs (top 5), %d tokens: engine request %.1f ms host, %.1f ms device; device vs host: rank differs on %d, top ids on %d, max |logprob difference| %.2e\n",
                P, hp.ms, dp.ms, br, bt, md);
    check(same_n && hp.plp.size() == p.size() && br == 0 && bt == 0 && md <= 1e-4, "prompt logprobs on the cards == the host's from full logits (rank, top 5 ids, values within 1e-4)");
    // The many-row head (the default above) against the head per 4-row slice.
    // bit for bit held to andone row of 512 differs by one fp32 ulp of its logprob (5.96e-07;
    // the rounds between have no binaries to bisect). The contract is now ranks and top ids equal, values within 2e-6
    // The exact count is printed.
    {
      const Run ep = run(true, 5, -1, 1, false);
      int diff = 0, first = -1, far = 0;
      double mx = 0;
      for (size_t i = 0; i < std::min(ep.plp.size(), dp.plp.size()); ++i) {
        const auto &a = dp.plp[i], &b = ep.plp[i];
        const bool nan2 = std::isnan(a.logprob) && std::isnan(b.logprob);
        bool same = nan2 || (a.logprob == b.logprob && a.rank == b.rank && a.top == b.top);
        if (!same) {
          ++diff;
          if (first < 0) first = static_cast<int>(i);
          if (!nan2) mx = std::max(mx, std::fabs(static_cast<double>(a.logprob) - b.logprob));
          // Within 4e-6 = 2 fp32 ulps at the logsumexp's magnitude (16-32 on this vocabulary), which every
          // logprob subtracts: at TP1 the two heads' logsumexp differ by one ulp on a row (2.15e-6; TP2 <= 6e-7)
          auto close = [](float x, float y) { return std::fabs(static_cast<double>(x) - y) <= 4e-6; };
          bool near = !nan2 && a.rank == b.rank && a.top.size() == b.top.size() && close(a.logprob, b.logprob);
          for (size_t t = 0; near && t < a.top.size(); ++t)
            near = a.top[t].first == b.top[t].first && close(a.top[t].second, b.top[t].second);
          if (!near) std::printf("  prompt logprobs: row %zu differs past the bound: %.9g vs %.9g\n", i, static_cast<double>(a.logprob), static_cast<double>(b.logprob));
          far += near ? 0 : 1;
        }
      }
      std::printf("prompt logprobs (top 5): many-row head %.1f ms vs a head per 4-row slice %.1f ms (engine request); rows differing %d (first %d), max |logprob difference| %.2e\n",
                  dp.ms, ep.ms, diff, first, mx);
      check(ep.plp.size() == dp.plp.size() && far == 0,
            "prompt logprobs: many-row head == a head per 4-row slice (rank, top 5 ids equal; logprob and top values within 4e-6)");
      check(ep.toks == dp.toks, "prompt logprobs: the first token is the same either way");
    }
    const auto &plp = dp.plp;
    // the same tokens in 4-row decode steps: row j's logits score token j + 1
    ad.begin(0);
    std::vector<core::TokenLogprob> dec(p.size());
    std::vector<float> lg;
    const int V = static_cast<int>(ad.vocab());
    for (int r0 = 0; r0 < P; r0 += 4) {
      const int rows = std::min(4, P - r0);
      m.use_slot(0);
      m.step(p.data() + r0, rows);
      m.logits(rows, lg);
      for (int j = 0; j < rows; ++j)
        if (r0 + j + 1 < P)
          core::Sampler::raw_logprobs(lg.data() + static_cast<size_t>(j) * V, V, p[static_cast<size_t>(r0 + j + 1)], 5, dec[static_cast<size_t>(r0 + j + 1)]);
    }
    int agree = 0, finite = 0;
    double sum_d = 0, max_d = 0;
    for (int i = 1; i < P && static_cast<size_t>(i) < plp.size(); ++i) {
      const auto &a = plp[static_cast<size_t>(i)], &b = dec[static_cast<size_t>(i)];
      finite += std::isfinite(a.logprob);
      agree += !a.top.empty() && !b.top.empty() && a.top[0].first == b.top[0].first;
      const double d = std::fabs(static_cast<double>(a.logprob) - b.logprob);
      sum_d += d;
      max_d = std::max(max_d, d);
    }
    const bool shape = plp.size() == p.size() && std::isnan(plp[0].logprob) && finite == P - 1;
    std::printf("prompt logprobs: %d tokens, top-1 agreement with 4-row decode steps %.2f %%, |target logprob difference| mean %.4f max %.4f\n", P,
                100.0 * agree / (P - 1), sum_d / (P - 1), max_d);
    // Prefill rounds each card's expert outputs to bf16 (moe_rows y), the decode path keeps
    // fp32: 89.82 -> 89.24 % here, so the bar is 88 % (a structural check, not a numerics gate; ppl / KL
    // vs the reference judge the numerics).
    check(shape && agree >= (P - 1) * 88 / 100, "prompt logprobs: one per prompt token ([0] none), top-1 == the decode path's on >= 88 % of rows");
    // greedy logprobs on the output: device vs host, and the decode rate against none
    const Run hg = run(false, -1, 5, G * 4), dg = run(true, -1, 5, G * 4), ng = run(true, -1, -1, G * 4);
    const bool same_g = cmp(hg.lp, dg.lp, br, bt, md);
    std::printf("greedy logprobs (top 5), %zu tokens: device vs host rank differs on %d, top ids on %d, max |difference| %.2e; decode %.1f tok/s host, %.1f device, %.1f without logprobs\n",
                dg.toks.size(), br, bt, md, hg.decode_tps, dg.decode_tps, ng.decode_tps);
    check(same_g && hg.toks == dg.toks && dg.toks == ng.toks && dg.lp.size() == dg.toks.size() && br == 0 && bt == 0 && md <= 1e-4,
          "greedy logprobs on the cards == the host's (same tokens, rank, top 5 ids, values within 1e-4)");
    ad.set_device_logprobs(true);
    // prefill time: plain, every row's head + logits on the host, every row's head + logprobs on the cards
    auto time_pf = [&](int mode) {
      std::vector<double> v;
      std::vector<int32_t> t(4);
      std::vector<core::TokenLogprob> out(4);
      for (int k = 0; k < 3; ++k) {
        ad.begin(0);
        const auto a = clk::now();
        if (mode == 1) ad.prefill_rows(0, p.data(), P, -1, [&](int, int, const float *) {});
        else if (mode == 2)
          ad.prefill_each(0, p.data(), P, -1, [&](int row0, int rows) {
            for (int j = 0; j < rows; ++j) t[static_cast<size_t>(j)] = row0 + j + 1 < P ? p[static_cast<size_t>(row0 + j + 1)] : -1;
            ad.row_logprobs(rows, t.data(), 5, out.data());
          });
        else if (mode == 3) {  // the many-row head
          std::vector<int32_t> tg(static_cast<size_t>(P));
          std::vector<core::TokenLogprob> all(static_cast<size_t>(P));
          for (int j = 0; j < P; ++j) tg[static_cast<size_t>(j)] = j + 1 < P ? p[static_cast<size_t>(j + 1)] : -1;
          if (!ad.prefill_row_logprobs(0, p.data(), P, -1, tg.data(), 5, all.data())) return -1.0;
        }
        else ad.prefill(0, p.data(), P, -1);
        ad.prefill_pick();
        v.push_back(std::chrono::duration<double, std::milli>(clk::now() - a).count());
      }
      return med(v);
    };
    const double plain = time_pf(0), rows = time_pf(1), each = time_pf(2), many = time_pf(3);
    std::printf("prompt logprobs: %d-token prefill %.1f ms plain, %.1f ms with every row's head + logits copied, %.1f ms with every row's head + logprobs (top 5) on the cards, %.1f ms with the many-row head + logprobs on the cards\n", P,
                plain, rows, each, many);
  }
  // ---- 6 split prefill: the same prompt prefilled in one call vs in calls of 64 / 192 rows, and of
  // 20 / 7 (calls of <= 32 rows take pf_dense's direct kernel) ----
  {
    // per-row logits: the first row that differs tells a row-count-dependent kernel (rows < 64
    // differ) from state carried across calls (only rows after the first boundary differ)
    const auto p = prompt_of(0, P);
    auto rows_of = [&](int piece, std::vector<float> &all) {
      const size_t V = static_cast<size_t>(ad.vocab());
      all.assign(static_cast<size_t>(P) * V, 0.0f);
      ad.begin(0);
      for (int r0 = 0; r0 < P; r0 += piece)
        ad.prefill_rows(0, p.data() + r0, std::min(piece, P - r0), r0 + piece < P ? p[static_cast<size_t>(r0 + piece)] : -1, [&](int row0, int rows, const float *lg) {
          std::copy(lg, lg + static_cast<size_t>(rows) * V, all.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(r0 + row0) * V));
        });
    };
    std::vector<float> w1, wrow, s;
    rows_of(P, w1);
    const size_t V = static_cast<size_t>(ad.vocab());
    for (int piece : {64, 192, 20, 7}) {
      // the chunked GDN scan keeps calls cut at multiples of 64 bit-identical; calls of 20 / 7
      // (which only exist here, to reach the small-call kernels) are compared with the GDN recurrence on
      // the row core for both the one-call reference and the pieces
      const bool aligned = piece % 64 == 0;
      if (!aligned) {
        m.set_gdn_scan(false);
        if (wrow.empty()) rows_of(P, wrow);
      }
      rows_of(piece, s);
      m.set_gdn_scan(g_gdn_scan);
      const std::vector<float> &w = aligned ? w1 : wrow;
      int first = -1, ndiff = 0, agree = 0;
      for (int r = 0; r < P; ++r) {
        const float *a = w.data() + static_cast<size_t>(r) * V, *b = s.data() + static_cast<size_t>(r) * V;
        const bool d = !std::equal(a, a + V, b);
        ndiff += d;
        if (d && first < 0) first = r;
        agree += std::max_element(a, a + V) - a == std::max_element(b, b + V) - b;
      }
      std::printf("split prefill: %d tokens in calls of %d vs one call: %d rows differ, the first at row %d; argmax equal on %.2f %% of rows\n", P, piece, ndiff,
                  first, 100.0 * agree / P);
      check(ndiff == 0, "split prefill: " + std::to_string(P) + " tokens in calls of " + std::to_string(piece) + " == one call, every row's logits bit for bit" +
                            (aligned ? "" : " (GDN on the row core)"));
    }
  }
  // ---- 7 split prefill at the end of a longer prompt ----
  {
    const auto p = prompt_of(K - 1, P + 97 * (K - 1));
    const int n = static_cast<int>(p.size());
    auto run = [&](int piece, std::vector<float> &last, std::vector<int32_t> &toks) {
      ad.begin(0);
      // : as the engine sizes calls, no call is left with fewer than 64 rows after it (calls of
      // fewer rows run the GDN row core, not the chunked scan: other bits)
      for (int r0 = 0; r0 < n;) {
        int c = std::min(piece, n - r0);
        if (n - r0 - c > 0 && n - r0 - c < 64) c = n - r0;
        ad.prefill(0, p.data() + r0, c, r0 + c < n ? p[static_cast<size_t>(r0 + c)] : -1);
        r0 += c;
      }
      ad.prefill_logits(last);
      int32_t t = ad.prefill_pick();
      toks.assign(1, t);
      for (int g = 1; g < 16; ++g) {
        ad.step({0}, &t);
        ad.picks(1, &t);
        toks.push_back(t);
      }
    };
    std::vector<float> whole, part;
    std::vector<int32_t> tw, tp;
    run(n, whole, tw);
    for (int piece : {64, 192, 256}) {
      run(piece, part, tp);
      double maxd = 0;
      size_t diff = 0;
      for (size_t i = 0; i < whole.size(); ++i) {
        const double d = std::fabs(static_cast<double>(whole[i]) - part[i]);
        maxd = std::max(maxd, d);
        diff += d != 0;
      }
      std::printf("split prefill: %d tokens in calls of %d vs one call: last-row logits differ in %zu of %zu (max |d| %.4g); 16 greedy tokens %s\n", n, piece,
                  diff, whole.size(), maxd, tp == tw ? "equal" : "differ");
      check(diff == 0 && tp == tw, "split prefill: " + std::to_string(n) + " tokens in calls of " + std::to_string(piece) + " == one call (last row bit for bit, 16 greedy tokens)");
    }
  }
  check(m.wire_errors() == 0, "no mailbox timeout");
  return rc;
}
