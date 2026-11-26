// xenithium-probe seqs: several sequences in one decode step, one load for every run.
// [--hot-rank F] [--reserve G] [--graphs 0|1] [--kv-bytes 1|2] [--pf-chunk N]
// Slot s gets its own prompt: P + 37 s tokens of the token file's text (all chunks joined, cycled)
// from offset 1000 s, so the slots sit at different positions and hold different text. Every phase
// starts each slot fresh and prefills its prompt (Model:prefill on that slot).
// 1 solo each slot alone, G greedy 1-row steps: the reference tokens and logits per step.
// 2 batched all N slots in every step (N segments of 1 row): each slot's tokens and logits must
// equal its solo run's, bit for bit.
// 3 subsets slot s skips every step i with (i + s) % 3 == 2: a slot left out keeps its state (its
// k-th own step still equals solo step k).
// 4 drafts 2 rows per segment (token + one draft, teacher-forced on the slot's text): a slot
// commits 2 or 1 rows by its own pattern (a rejected draft is a junk token); batched
// vs the same verify / commit sequence of the slot alone, bit for bit (GDN replay of
// an accept count per slot, position-addressed caches rewritten per slot).
// 4b unequal segments of B rows, then one-row segments (layouts and drafting slots change per step),
// teacher-forced, commits per slot pattern: == each slot alone, bit for bit.
// 4c 16 rows (B1): 8 segments x 2 rows and 4 x 4 (the 16-row shape), teacher-forced, commits per slot
// pattern: == each slot alone, bit for bit.
// 5 timing n slots of r rows per step, r = 1, 2, 4 and n x r <= 16 (Head:kRows was 8):
// ms per step and aggregate tokens per second, graph replay as configured and unequal
// layouts); then card 0's
// kernel time per module (eager, profiled) for the 4- and 8-row shapes.
// 5b contexts XEN_PROBE_SEQS_CTX=C,C... with XEN_PROBE_SEQS_TIME=NxR...: the shapes at C tokens of
// context per sequence instead (slot 0 reads the prompt once per C; a shape's sequences are slots 1..N,
// forked from it, each with a tail of its own: --seqs N + 1; time_at_contexts).
// XEN_PROBE_PROFILING=0: queues without event profiling, as the server (the prompt reads at the
// served rate; the eager profile is skipped).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/kv_pages.hpp"
#include "xenithium/rt/device.hpp"
#include "kv.hpp"

#include "probe.hpp"  // the run-wide flags
namespace q38 = xenithium::arch::qwen38;

namespace {

bool read_all_tokens(const std::string &path, std::vector<int32_t> &out) {
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

double med(std::vector<double> v, double p = 0.5) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

// FNV-1a over a logits row's bits (bf16-rounded values: exact equality = equal hashes).
uint64_t row_hash(const float *l, int64_t n) {
  uint64_t h = 1469598103934665603ull;
  for (int64_t i = 0; i < n; ++i) {
    uint32_t u;
    std::memcpy(&u, &l[i], 4);
    h = (h ^ u) * 1099511628211ull;
  }
  return h;
}

struct Rec {
  std::vector<int32_t> tok;  // per own step and row
  std::vector<uint64_t> hash;  // per own step and row
};

// With XEN_DEC_PROFILE the replayed steps' module stamps since the last reset (ms per step, the
// slowest card per module), printed under a timing line (the stamps cost ~0.1-0.2 ms per step: take walls from a run
// without the env). And the QSA layer's stages.
void print_step_stamps(q38::Model &m, const char *what) {
  if (!m.dec_profiling()) return;
  const auto d = m.dec_profile();
  m.dec_profile_reset();
  if (d.steps == 0) return;
  std::printf("    stamps %-14s", what);
  double sum = 0;
  for (int p = 0; p < q38::StepProfile::kParts; ++p) {
    double v = 0;
    for (int c = 0; c < m.cards(); ++c) v = std::max(v, d.ms[c][p] / d.steps);
    sum += v;
    std::printf(" %s %.3f", q38::StepProfile::name(p), v);
  }
  double span = 0;
  for (int c = 0; c < m.cards(); ++c) span = std::max(span, d.span[c] / d.steps);
  std::printf(" | span %.3f sum %.3f | moe router %.3f route %.3f experts %.3f combine %.3f | qsa", span, sum, d.moe[0][0] / d.steps,
              d.moe[0][1] / d.steps, d.moe[0][2] / d.steps, d.moe[0][3] / d.steps);
  for (int st = 0; st < static_cast<int>(xenithium::kernels::QsaStage::kCount); ++st) {  // The slowest card per stage
    double v = 0;
    for (int c = 0; c < m.cards(); ++c) v = std::max(v, d.qsa[c][st] / d.steps);
    std::printf(" %s %.3f", xenithium::kernels::qsa_stage_name(static_cast<xenithium::kernels::QsaStage>(st)), v);
  }
  std::printf(" | launches");
  for (int p = 0; p < q38::StepProfile::kParts; ++p) std::printf(" %d", d.launches[0][p]);
  std::printf("\n");
}

// "NxR..." -> the (N sequences, R rows) shapes of the list; other entries (the unequal layouts "uB-K-N") are left out
std::vector<std::pair<int, int>> plain_shapes(const std::string &list) {
  std::vector<std::pair<int, int>> out;
  for (size_t at = 0; at < list.size();) {
    const size_t end = std::min(list.find(',', at), list.size());
    int n = 0, r = 0;
    char extra = 0;
    if (std::sscanf(list.substr(at, end - at).c_str(), "%dx%d%c", &n, &r, &extra) == 2 && n > 0 && r > 0) out.emplace_back(n, r);
    at = end + 1;
  }
  return out;
}

// "8192,32768..." -> token counts (entries that are not positive numbers are left out)
std::vector<int64_t> token_counts(const std::string &list) {
  std::vector<int64_t> out;
  for (size_t at = 0; at < list.size();) {
    const size_t end = std::min(list.find(',', at), list.size());
    const long long v = std::atoll(list.substr(at, end - at).c_str());
    if (v > 0) out.push_back(v);
    at = end + 1;
  }
  return out;
}

// The timing shapes at agent contexts. For each context C slot 0 reads C
// tokens of the text once and is snapshotted, and holds that prompt for the rest of the run. Before each shape, each of
// its sequences (slots 1..N) starts fresh, forks the snapshot (the KV pages below C copied, the state restored) and reads
// a tail of its own text (P + 37 s tokens): every shape starts at the same context, the sequences hold different recent
// text, and the long prompt is read once per C. Returns false when a context or a fork could not be placed.
struct ContextRun {
  q38::Model &m;
  const std::vector<int32_t> &text;
  int slots = 0, P = 0, G = 0;
};
constexpr int64_t kPage = xenithium::model::KvPages::kPageTokens;
constexpr int kContextRows = 4;  // rows per sequence in the context shapes
constexpr double kContextSnapGib = 1;  // the prompt's snapshot (~0.16 GiB at four cards) with room
// the tokens a slot needs at context C: the prompt, the longest tail (slot slots - 1) and 2 G + 2 steps of 4 rows
int64_t context_need(int64_t C, int slots, int P, int G) { return C + P + 37LL * (slots - 1) + (2LL * G + 2) * kContextRows + kPage; }
double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

bool time_at_contexts(const ContextRun &x, const std::vector<int64_t> &contexts, const std::vector<std::pair<int, int>> &shapes) {
  q38::Model &m = x.m;
  auto token = [&](int s, int64_t i) { return x.text[static_cast<size_t>((1000 * static_cast<int64_t>(s) + i) % static_cast<int64_t>(x.text.size()))]; };
  auto tail_len = [&](int s) { return x.P + 37 * s; };
  const int steps = 2 * x.G + 2;
  bool ok = true;
  std::printf("timing at contexts (slot 0 holds the prompt; a shape's N sequences are slots 1..N, each forked from it with a tail of P + 37 s "
              "tokens), median / p90 of %d steps\n  %-9s %-18s %9s %9s %12s %12s\n",
              steps, "context", "step", "med ms", "p90 ms", "tok/s (all)", "tok/s / seq");
  for (const int64_t asked : contexts) {
    const int64_t C = asked / kPage * kPage;  // a snapshot sits at a page boundary
    const int64_t need = context_need(C, x.slots, x.P, x.G);
    if (C < kPage || need > m.capacity()) {
      std::printf("  context %lld: skipped (needs %lld tokens per slot, the capacity is %lld: --context)\n", static_cast<long long>(asked), static_cast<long long>(need),
                  static_cast<long long>(m.capacity()));
      ok = false;
      continue;
    }
    std::vector<int32_t> prompt(static_cast<size_t>(C));
    for (int64_t i = 0; i < C; ++i) prompt[static_cast<size_t>(i)] = token(0, i);
    m.use_slot(0);
    m.begin_sequence();
    if (!m.kv_reserve(0, C)) {
      std::printf("  context %lld: skipped (the KV pool cannot hold the prompt: --kv-pool)\n", static_cast<long long>(C));
      ok = false;
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    m.prefill(prompt.data(), static_cast<int>(C));
    (void)m.tokens(1);  // waits for the call's last chunk
    const double read_ms = ms_since(t0);
    const int id = m.snap_save(0);
    if (id < 0) {
      std::printf("  context %lld: no snapshot store\n", static_cast<long long>(C));
      return false;
    }
    m.snap_sync();
    std::printf("  context %lld: slot 0 read the prompt in %.0f ms (%.0f tok/s)\n", static_cast<long long>(C), read_ms, static_cast<double>(C) * 1e3 / read_ms);
    for (const auto &[n, r] : shapes) {
      if (n >= x.slots || r > kContextRows || n * r > m.step_rows_cap()) {
        std::printf("  %-9lld %d x %d: skipped (needs %d slots, at most %d rows per sequence and %d rows per step; --seqs, XEN_PROBE_STEP_ROWS_CAP)\n",
                    static_cast<long long>(C), n, r, n + 1, kContextRows, n * r);
        continue;
      }
      std::vector<int32_t> next(static_cast<size_t>(n) + 1);
      bool placed = true;
      for (int s = 1; s <= n && placed; ++s) {
        m.use_slot(s);
        m.begin_sequence();
        placed = m.kv_reserve(s, need);
        if (!placed) break;
        m.snap_fork(s, 0, id);
        std::vector<int32_t> tail(static_cast<size_t>(tail_len(s)));
        for (size_t i = 0; i < tail.size(); ++i) tail[i] = token(s, C + static_cast<int64_t>(i));
        m.prefill(tail.data(), static_cast<int>(tail.size()));
        const int last = static_cast<int>((tail.size() - 1) % 4);
        next[static_cast<size_t>(s)] = m.tokens(last + 1)[static_cast<size_t>(last)];
      }
      m.use_slot(0);
      if (!placed) {
        std::printf("  %-9lld %d x %d: skipped (the KV pool cannot hold %d forks: --kv-pool)\n", static_cast<long long>(C), n, r, n);
        ok = false;
        continue;
      }
      if (m.dec_profiling()) m.dec_profile_reset();
      std::vector<double> ms;
      int32_t t[q38::Head::kRows][kContextRows];
      for (int i = 0; i < steps; ++i) {
        std::vector<q38::Model::SeqRows> segs;
        for (int s = 1; s <= n; ++s) {  // 1 row = the greedy pick; r rows = the sequence's text after its tail (teacher-forced)
          int32_t *row = t[s - 1];
          if (r == 1) row[0] = next[static_cast<size_t>(s)];
          else
            for (int j = 0; j < r; ++j) row[j] = token(s, C + tail_len(s) + static_cast<int64_t>(i) * r + j);
          segs.push_back({s, row, r});
        }
        const auto ts = std::chrono::steady_clock::now();
        m.step_batch(segs);
        const double w = ms_since(ts);
        if (m.times().record == 0) ms.push_back(w);
        const auto tk = m.tokens(n * r);
        for (int s = 1; s <= n; ++s) next[static_cast<size_t>(s)] = tk[static_cast<size_t>((s - 1) * r + r - 1)];
      }
      const double md = med(ms);
      char what[64];
      std::snprintf(what, sizeof what, "%d seq%s x %d row%s", n, n > 1 ? "s" : "", r, r > 1 ? "s" : "");
      std::printf("  %-9lld %-18s %9.2f %9.2f %12.1f %12.1f\n", static_cast<long long>(C), what, md, med(ms, 0.9), n * r * 1e3 / md, r * 1e3 / md);
      char tag[96];
      std::snprintf(tag, sizeof tag, "%s @%lldK", what, static_cast<long long>(C / 1024));
      print_step_stamps(m, tag);
    }
    m.snap_free(id);
  }
  return ok;
}

}  // namespace

int cmd_seqs_run(const std::string &model, const std::string &ppl, int nseq, int prompt_len, int generate, int64_t context, bool graphs,
                 const std::string &hot_rank, double reserve_gib, int threads, int kv_bytes, int64_t hot, int pf_chunk) {
  if (model.empty() || ppl.empty()) {
    std::fprintf(stderr, "seqs: --model DIR and --ppl F are required\n");
    return 2;
  }
  if (nseq < 1 || nseq > xenithium::kernels::kMaxSegs) {  // 1..16 (the model's slots 9-16 run the 16-row shape)
    std::fprintf(stderr, "seqs: --seqs takes 1..64 (17-64: four cards)\n");
    return 2;
  }
  std::vector<int32_t> text;
  if (!read_all_tokens(ppl, text) || text.size() < 64) {
    std::fprintf(stderr, "seqs: cannot read token file %s\n", ppl.c_str());
    return 2;
  }
  const int P = prompt_len > 0 ? prompt_len : 512, G = generate > 0 ? generate : 32;
  auto slot_text = [&](int s, int64_t i) { return text[static_cast<size_t>((1000 * static_cast<int64_t>(s) + i) % static_cast<int64_t>(text.size()))]; };
  auto prompt_of = [&](int s) {
    std::vector<int32_t> p(static_cast<size_t>(P + 37 * s));
    for (size_t i = 0; i < p.size(); ++i) p[i] = slot_text(s, static_cast<int64_t>(i));
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
  if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;  // --pf-chunk (the served TP4 setup reads 4,096-row chunks)
  mo.slots = nseq;
  // XEN_PROBE_STEP_ROWS_CAP=64, the model's steps of up to 64 rows (ModelOptions:step_rows_cap)
  if (const char *cap = std::getenv("XEN_PROBE_STEP_ROWS_CAP")) mo.step_rows_cap = std::atoi(cap);
  mo.context = std::max<int64_t>(context, P + 37 * nseq + 8LL * G + 64);
  // XEN_PROBE_SEQS_CTX = "C,C..." times the XEN_PROBE_SEQS_TIME shapes at C tokens of context per
  // sequence (time_at_contexts): every slot's capacity holds the largest, and one snapshot (the prompt) is kept
  const char *ctx_env = std::getenv("XEN_PROBE_SEQS_CTX");
  const std::vector<int64_t> contexts = ctx_env ? token_counts(ctx_env) : std::vector<int64_t>{};
  for (const int64_t c : contexts) mo.context = std::max(mo.context, context_need(c, nseq, P, G));
  if (!contexts.empty()) mo.snap_gib = std::max(mo.snap_gib, kContextSnapGib);
  mo.kv = probe_kv(kv_bytes);
  mo.graphs = graphs;
  mo.threads = threads;
  if (reserve_gib >= 0) mo.reserve_gib = reserve_gib;
  mo.hot = hot;  // --hot N (cold experts exist) and the MoE CPU path's flags, for the batch-invariance gate
  mo.moe_cpu.mode = g_moe_cpu, mo.moe_cpu.pct = g_moe_cpu_pct, mo.moe_cpu.threads = g_moe_cpu_threads;
  mo.moe_cpu.min_cold = g_moe_cpu_min_cold;
  if (!hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(model));
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  // XEN_PROBE_PROFILING=0 runs the queues without event profiling, as the server does (probe model's
  // round-202 switch; a profiling queue costs ~17 us of host time per launch): the prompt reads of the contexts mode then
  // run at the served rate. The eager per-kernel profile and XEN_PF_PROFILE need it on (the default): off skips the
  // eager profile.
  const char *queue_prof_env = std::getenv("XEN_PROBE_PROFILING");
  xenithium::Fabric fab(loader, tp_devices("seqs", true), !(queue_prof_env && std::string_view(queue_prof_env) == "0"));  // --tp 1 (device 0)
  q38::Model m(loader, fab, model, mo);
  const auto &ls = m.load_stats();
  std::string hot_cards;  // Every card
  for (size_t c = 0; c < fab.size(); ++c) hot_cards += (c ? ", " : "") + std::to_string(ls.hot[c]) + " (card " + std::to_string(c) + ")";
  std::printf("seqs: loaded in %.1f s; %d slots x context %lld, KV %s; hot experts %s; graphs %s\n", ls.total_s, m.slots(),
              static_cast<long long>(m.capacity()), probe_kv_name(mo.kv), hot_cards.c_str(), graphs ? "on" : "off");
  std::printf("seqs: MoE CPU path %s\n", ls.moe_cpu.c_str());
  const int64_t V = m.config().vocab;
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  std::vector<float> lg;
  auto start_all = [&](std::vector<int32_t> &next) {
    next.assign(static_cast<size_t>(nseq), 0);
    for (int s = 0; s < nseq; ++s) {
      const auto pr = prompt_of(s);
      m.use_slot(s);
      m.begin_sequence();
      m.prefill(pr.data(), static_cast<int>(pr.size()));
      const int last = static_cast<int>((pr.size() - 1) % 4);
      next[static_cast<size_t>(s)] = m.tokens(last + 1)[static_cast<size_t>(last)];
    }
    m.use_slot(0);
  };
  // the last step's rows -> rec of each segment (segment i = slots[i], rows r each)
  auto keep = [&](const std::vector<int> &slots, int r, std::vector<Rec> &rec) {
    const int rows = static_cast<int>(slots.size()) * r;
    m.logits(rows, lg);
    const auto t = m.tokens(rows);
    for (size_t i = 0; i < slots.size(); ++i)
      for (int j = 0; j < r; ++j) {
        const size_t row = i * static_cast<size_t>(r) + static_cast<size_t>(j);
        rec[static_cast<size_t>(slots[i])].tok.push_back(t[row]);
        rec[static_cast<size_t>(slots[i])].hash.push_back(row_hash(lg.data() + row * static_cast<size_t>(V), V));
      }
  };
  // The same for unequal segments (segment i: rows rr[i])
  auto keep_rr = [&](const std::vector<int> &slots, const std::vector<int> &rr, std::vector<Rec> &rec) {
    int rows = 0;
    for (int r : rr) rows += r;
    m.logits(rows, lg);
    const auto t = m.tokens(rows);
    size_t row = 0;
    for (size_t i = 0; i < slots.size(); ++i)
      for (int j = 0; j < rr[i]; ++j, ++row) {
        rec[static_cast<size_t>(slots[i])].tok.push_back(t[row]);
        rec[static_cast<size_t>(slots[i])].hash.push_back(row_hash(lg.data() + row * static_cast<size_t>(V), V));
      }
  };
  auto same = [&](const Rec &a, const Rec &b, size_t &n) {
    n = std::min(a.hash.size(), b.hash.size());
    size_t eq = 0;
    while (eq < n && a.hash[eq] == b.hash[eq] && a.tok[eq] == b.tok[eq]) ++eq;
    return eq;
  };

  // ---- 1 solo ----
  std::vector<Rec> solo(static_cast<size_t>(nseq));
  std::vector<int32_t> next;
  // XEN_PROBE_SEQS_TIME = "NxR..." (N sequences x R rows; uB-K-N unequal layouts): those timing
  // shapes only, no checks
  const char *time_only = std::getenv("XEN_PROBE_SEQS_TIME");
  const std::string shapes = time_only ? "," + std::string(time_only) + "," : std::string();
  auto timed = [&](const std::string &key) { return !time_only || shapes.find("," + key + ",") != std::string::npos; };
  if (!time_only) {
  start_all(next);
  for (int s = 0; s < nseq; ++s) {
    int32_t t = next[static_cast<size_t>(s)];
    for (int i = 0; i < G; ++i) {
      m.step_batch({{s, &t, 1}});
      keep({s}, 1, solo);
      t = solo[static_cast<size_t>(s)].tok.back();
    }
  }
  std::printf("solo: %d slots x %d greedy steps; slot 0 tokens:", nseq, G);
  for (int i = 0; i < std::min(G, 16); ++i) std::printf(" %d", solo[0].tok[static_cast<size_t>(i)]);
  std::printf("\n");

  // ---- 2 batched ----
  {
    std::vector<Rec> b(static_cast<size_t>(nseq));
    start_all(next);
    std::vector<int> all;
    for (int s = 0; s < nseq; ++s) all.push_back(s);
    for (int i = 0; i < G; ++i) {
      std::vector<q38::Model::SeqRows> segs;
      for (int s = 0; s < nseq; ++s) segs.push_back({s, &next[static_cast<size_t>(s)], 1});
      m.step_batch(segs);
      keep(all, 1, b);
      for (int s = 0; s < nseq; ++s) next[static_cast<size_t>(s)] = b[static_cast<size_t>(s)].tok.back();
    }
    for (int s = 0; s < nseq; ++s) {
      size_t n = 0;
      const size_t eq = same(solo[static_cast<size_t>(s)], b[static_cast<size_t>(s)], n);
      check(eq == n && n == static_cast<size_t>(G), "batched (" + std::to_string(nseq) + " x 1 row): slot " + std::to_string(s) + " tokens + logits == solo, bit for bit (" +
                                                        std::to_string(eq) + " / " + std::to_string(n) + " steps)");
    }
  }
  // ---- 3 subsets: slot s skips step i when (i + s) % 3 == 2 ----
  if (nseq > 1) {
    std::vector<Rec> b(static_cast<size_t>(nseq));
    start_all(next);
    std::vector<int> own(static_cast<size_t>(nseq), 0);
    for (int i = 0; i < 2 * G; ++i) {
      std::vector<q38::Model::SeqRows> segs;
      std::vector<int> in;
      for (int s = 0; s < nseq; ++s)
        if ((i + s) % 3 != 2 && own[static_cast<size_t>(s)] < G) {
          segs.push_back({s, &next[static_cast<size_t>(s)], 1});
          in.push_back(s);
        }
      if (segs.empty()) break;
      m.step_batch(segs);
      keep(in, 1, b);
      for (int s : in) {
        next[static_cast<size_t>(s)] = b[static_cast<size_t>(s)].tok.back();
        ++own[static_cast<size_t>(s)];
      }
    }
    for (int s = 0; s < nseq; ++s) {
      size_t n = 0;
      const size_t eq = same(solo[static_cast<size_t>(s)], b[static_cast<size_t>(s)], n);
      check(eq == n && n == static_cast<size_t>(G), "subsets (slot s skips every third step): slot " + std::to_string(s) + " == solo (" + std::to_string(eq) +
                                                        " / " + std::to_string(n) + " own steps)");
    }
  }
  // ---- 4 drafts: 2 rows per segment, commits 2 or 1 by slot pattern ----
  if (nseq >= 2) {
    const int nd = 2;  // 2 segments of 2 rows = the 4-row step
    const int steps = G;
    auto commit_of = [](int s, int i) { return ((i * (s + 2)) % 3) == 0 ? 1 : 2; };  // per slot pattern
    constexpr int32_t kJunk = 13;
    auto rows_of = [&](int s, int64_t p, int c, int32_t *t) {  // teacher-forced: the slot's text after its prompt
      const int64_t base = P + 37 * s;
      t[0] = slot_text(s, base + p);
      t[1] = c == 2 ? slot_text(s, base + p + 1) : kJunk;
    };
    std::vector<Rec> ref(static_cast<size_t>(nd)), b(static_cast<size_t>(nd));
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<int32_t> nx;
      start_all(nx);
      std::vector<int64_t> p(static_cast<size_t>(nd), 0);
      if (pass == 0) {
        for (int s = 0; s < nd; ++s)
          for (int i = 0; i < steps; ++i) {
            const int c = commit_of(s, i);
            int32_t t[2];
            rows_of(s, p[static_cast<size_t>(s)], c, t);
            m.verify_batch({{s, t, 2}});
            keep({s}, 2, ref);
            m.commit_batch({c});
            p[static_cast<size_t>(s)] += c;
          }
      } else {
        for (int i = 0; i < steps; ++i) {
          int32_t t[2][2];
          std::vector<q38::Model::SeqRows> segs;
          std::vector<int> in, cs;
          for (int s = 0; s < nd; ++s) {
            const int c = commit_of(s, i);
            rows_of(s, p[static_cast<size_t>(s)], c, t[s]);
            segs.push_back({s, t[s], 2});
            in.push_back(s);
            cs.push_back(c);
          }
          m.verify_batch(segs);
          keep(in, 2, b);
          m.commit_batch(cs);
          for (int s = 0; s < nd; ++s) p[static_cast<size_t>(s)] += cs[static_cast<size_t>(s)];
        }
      }
    }
    for (int s = 0; s < nd; ++s) {
      size_t n = 0;
      const size_t eq = same(ref[static_cast<size_t>(s)], b[static_cast<size_t>(s)], n);
      check(eq == n && n == static_cast<size_t>(2 * steps), "drafts (2 segments x 2 rows, commits 2 / 1 per slot pattern): slot " + std::to_string(s) +
                                                                " rows == alone, bit for bit (" + std::to_string(eq) + " / " + std::to_string(n) + " rows)");
    }
  }
  // ---- 4c: the 16-row shape, n segments x r rows (8 x 2, 4 x 4), teacher-forced, commits per slot
  // pattern; == each slot alone, bit for bit ----
  // And the 32-row shape (16 x 2, 8 x 4) where the step holds 32 rows
  // And the 64-row shape (16 x 3, 16 x 4) where the model carries it (ModelOptions:step_rows_cap)
  for (const auto &[nd, r] : std::vector<std::pair<int, int>>{{8, 2}, {4, 4}, {16, 2}, {8, 4}, {16, 3}, {16, 4}}) {
    if (nseq < nd || nd * r > m.step_rows_cap()) continue;
    const int steps = G;
    auto commit_of = [r](int s, int i) { return 1 + ((i + s) % r); };
    constexpr int32_t kJunk = 13;
    std::vector<Rec> ref(static_cast<size_t>(nd)), b(static_cast<size_t>(nd));
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<int32_t> nx;
      start_all(nx);
      std::vector<int64_t> p(static_cast<size_t>(nd), 0);
      auto rows_of = [&](int s, int c, int32_t *t) {
        const int64_t base = P + 37 * s;
        for (int j = 0; j < r; ++j) t[j] = j < c ? slot_text(s, base + p[static_cast<size_t>(s)] + j) : kJunk;
      };
      for (int i = 0; i < steps; ++i) {
        int32_t t[16][4];
        std::vector<q38::Model::SeqRows> segs;
        std::vector<int> in, cs;
        for (int s = 0; s < nd; ++s) {
          cs.push_back(commit_of(s, i));
          rows_of(s, cs.back(), t[s]);
          segs.push_back({s, t[s], r});
          in.push_back(s);
        }
        if (pass == 0) {
          for (int s = 0; s < nd; ++s) {
            m.verify_batch({segs[static_cast<size_t>(s)]});
            keep({s}, r, ref);
            m.commit_batch({cs[static_cast<size_t>(s)]});
          }
        } else {
          m.verify_batch(segs);
          keep(in, r, b);
          m.commit_batch(cs);
        }
        for (int s = 0; s < nd; ++s) p[static_cast<size_t>(s)] += cs[static_cast<size_t>(s)];
      }
    }
    size_t eq_all = 0, n_all = 0;
    for (int s = 0; s < nd; ++s) {
      size_t n = 0;
      eq_all += same(ref[static_cast<size_t>(s)], b[static_cast<size_t>(s)], n);
      n_all += n;
    }
    check(eq_all == n_all && n_all == static_cast<size_t>(nd * r * steps), std::to_string(nd * r) + " rows (" + std::to_string(nd) + " segments x " + std::to_string(r) +
                                                                           " rows, partial commits): rows == each slot alone, bit for bit (" +
                                                                           std::to_string(eq_all) + " / " + std::to_string(n_all) + " rows)");
  }
  check(m.wire_errors() == 0 && m.cards_agree(), "no mailbox timeout; both cards bit-identical at the end");

  // ---- 4b unequal segments: the first k segments B rows, the rest one row, layouts and the
  // drafting slots changing per step; teacher-forced rows, commits by slot pattern; == each slot alone, bit for bit ----
  if (nseq >= 3) {
    struct Lay {
      int b, k, n;
    };
    std::vector<Lay> lays = {{2, 2, 3}, {3, 2, 3}, {4, 1, 3}, {2, 3, 5}, {3, 1, 4}, {2, 1, 3}, {4, 1, 4}, {2, 2, 6}, {3, 2, 4}, {2, 3, 4}};
    if (nseq >= 8)  // 16-row steps (9-16 rows)
      for (const Lay L : {Lay{2, 6, 10}, Lay{3, 4, 8}, Lay{4, 3, 7}, Lay{2, 4, 12}, Lay{4, 2, 6}, Lay{2, 1, 9}}) lays.push_back(L);
    const int ns = std::min(nseq, nseq >= 8 ? 16 : 6), steps = G;
    auto commit_of = [](int s, int i, int r) { return r == 1 ? 1 : 1 + ((i + s) % r); };
    constexpr int32_t kJunk = 13;
    // step i: layout lays[i % size], segment j = slot (i + j) % ns, the first k of B rows (n segments, n <= ns)
    auto plan = [&](int i, std::vector<int> &slots, std::vector<int> &rr) {
      const Lay &L = lays[static_cast<size_t>(i) % lays.size()];
      slots.clear(), rr.clear();
      for (int j = 0; j < std::min(L.n, ns); ++j) slots.push_back((i + j) % ns), rr.push_back(j < L.k ? L.b : 1);
    };
    std::vector<Rec> ref(static_cast<size_t>(ns)), b(static_cast<size_t>(ns));
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<int32_t> nx;
      start_all(nx);
      std::vector<int64_t> p(static_cast<size_t>(ns), 0);
      std::vector<int> own(static_cast<size_t>(ns), 0);
      auto rows_of = [&](int s, int r, int c, int32_t *t) {  // r rows: the first c the slot's text, then junk (rejected)
        const int64_t base = P + 37 * s;
        for (int j = 0; j < r; ++j) t[j] = j < c ? slot_text(s, base + p[static_cast<size_t>(s)] + j) : kJunk;
      };
      for (int i = 0; i < steps; ++i) {
        std::vector<int> slots, rr;
        plan(i, slots, rr);
        int32_t t[16][8];
        std::vector<int> cs;
        for (size_t j = 0; j < slots.size(); ++j) {
          const int s = slots[j];
          cs.push_back(commit_of(s, own[static_cast<size_t>(s)], rr[j]));
          rows_of(s, rr[j], cs.back(), t[j]);
        }
        if (pass == 0) {  // each slot alone, the same rows and commits
          for (size_t j = 0; j < slots.size(); ++j) {
            m.verify_batch({{slots[j], t[j], rr[j]}});
            keep({slots[j]}, rr[j], ref);
            m.commit_batch({cs[j]});
          }
        } else {
          std::vector<q38::Model::SeqRows> segs;
          for (size_t j = 0; j < slots.size(); ++j) segs.push_back({slots[j], t[j], rr[j]});
          m.verify_batch(segs);
          keep_rr(slots, rr, b);
          m.commit_batch(cs);
        }
        for (size_t j = 0; j < slots.size(); ++j) p[static_cast<size_t>(slots[j])] += cs[j], ++own[static_cast<size_t>(slots[j])];
      }
    }
    size_t eq_all = 0, n_all = 0;
    for (int s = 0; s < ns; ++s) {
      size_t n = 0;
      eq_all += same(ref[static_cast<size_t>(s)], b[static_cast<size_t>(s)], n);
      n_all += n;
    }
    check(eq_all == n_all && n_all > 0, "unequal segments (B rows then 1-row segments, 3-6 slots, layouts and drafting slots changing, partial commits): rows == each slot alone, "
                                        "bit for bit (" + std::to_string(eq_all) + " / " + std::to_string(n_all) + " rows)");
  }

  }  // !time_only

  // ---- 5 timing ----
  // XEN_PROBE_SEQS_CHECKS_ONLY=1 ends after the checks (the timing below re-reads every prompt per
  // shape: ~15 minutes at 4K prompts that a gate run does not read)
  if (const char *co = std::getenv("XEN_PROBE_SEQS_CHECKS_ONLY"); co && *co == '1' && !time_only) {
    std::printf("timing skipped (XEN_PROBE_SEQS_CHECKS_ONLY=1)\n");
    check(m.wire_errors() == 0, "checks: no mailbox timeout");
    return rc;
  }
  // A step's rows: 1 row = the greedy pick; r > 1 rows = the slot's text after its
  // prompt, r tokens per step, every row kept (teacher-forced: routing as on real text, not r copies).
  auto text_rows = [&](int s, int i, int r, int32_t *t, int32_t pick) {
    if (r == 1) {
      t[0] = pick;
      return;
    }
    for (int j = 0; j < r; ++j) t[j] = slot_text(s, P + 37 * s + static_cast<int64_t>(i) * r + j);
  };
  auto stamp_line = [&](const char *what) { print_step_stamps(m, what); };
  if (!contexts.empty()) {  // The shapes at agent contexts instead of after the short prompts
    const auto shapes = plain_shapes(time_only ? time_only : "");
    check(!shapes.empty() && time_at_contexts({m, text, nseq, P, G}, contexts, shapes),
          "timing at contexts: XEN_PROBE_SEQS_TIME names NxR shapes, and every context and fork found room");
  } else {
    std::printf("timing (graphs %s): ms per step after the prompts (P = %d + 37 s), median / p90 of %d steps\n", graphs ? "on" : "off", P, 2 * G);
    std::printf("  %-22s %9s %9s %12s %12s\n", "step", "med ms", "p90 ms", "tok/s (all)", "tok/s / seq");
    for (int r : {1, 2, 3, 4}) {
      if (r == 3 && !time_only) continue;  // 3-row segments on request (XEN_PROBE_SEQS_TIME)
      for (int n = 1; n * r <= m.step_rows_cap() && n <= nseq; ++n) {
        if (!timed(std::to_string(n) + "x" + std::to_string(r))) continue;
        start_all(next);
        if (m.dec_profiling()) m.dec_profile_reset();
        std::vector<double> ms;
        int32_t t[q38::Head::kRows][4];
        for (int i = 0; i < 2 * G + 2; ++i) {
          std::vector<q38::Model::SeqRows> segs;
          for (int s = 0; s < n; ++s) {
            text_rows(s, i, r, t[s], next[static_cast<size_t>(s)]);
            segs.push_back({s, t[s], r});
          }
          const auto ts = std::chrono::steady_clock::now();
          m.step_batch(segs);
          const double w = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
          if (m.times().record == 0) ms.push_back(w);
          const auto tk = m.tokens(n * r);
          for (int s = 0; s < n; ++s) next[static_cast<size_t>(s)] = tk[static_cast<size_t>(s * r + r - 1)];
        }
        const double md = med(ms);
        char what[64];
        std::snprintf(what, sizeof what, "%d seq%s x %d row%s", n, n > 1 ? "s" : "", r, r > 1 ? "s" : "");
        std::printf("  %-22s %9.2f %9.2f %12.1f %12.1f\n", what, md, med(ms, 0.9), n * r * 1e3 / md, r * 1e3 / md);
        stamp_line(what);
      }
    }
    // Unequal segments (B rows for the first k, one row for the rest), teacher-forced rows, all kept
    if (nseq >= 3) {
      struct Lay {
        int b, k, n;
      };
      for (const Lay L : std::vector<Lay>{{3, 2, 3}, {4, 1, 3}, {2, 2, 4}, {3, 2, 4}, {4, 1, 4}, {2, 3, 5}, {2, 1, 5}, {2, 2, 6}, {2, 1, 7},
                                          {2, 4, 12}, {2, 6, 10}, {2, 2, 14}, {4, 2, 8},  // 16-row layouts
                                          {2, 8, 16}, {2, 16, 16}, {3, 8, 16}, {4, 6, 14}}) {  // 24 / 32 rows
        if (L.n > nseq || !timed("u" + std::to_string(L.b) + "-" + std::to_string(L.k) + "-" + std::to_string(L.n))) continue;
        start_all(next);
        if (m.dec_profiling()) m.dec_profile_reset();
        std::vector<double> ms;
        int32_t t[16][4];
        for (int i = 0; i < 2 * G + 2; ++i) {
          std::vector<q38::Model::SeqRows> segs;
          for (int s = 0; s < L.n; ++s) {
            const int r = s < L.k ? L.b : 1;
            text_rows(s, i, r, t[s], next[static_cast<size_t>(s)]);
            segs.push_back({s, t[s], r});
          }
          const auto ts = std::chrono::steady_clock::now();
          m.step_batch(segs);
          const double w = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
          if (m.times().record == 0) ms.push_back(w);
          const int rows = L.k * L.b + L.n - L.k;
          const auto tk = m.tokens(rows);
          for (int s = 0, r0 = 0; s < L.n; r0 += s < L.k ? L.b : 1, ++s) next[static_cast<size_t>(s)] = tk[static_cast<size_t>(r0 + (s < L.k ? L.b : 1) - 1)];
        }
        const double md = med(ms);
        char what[64];
        std::snprintf(what, sizeof what, "%d seqs, %d x %d + %d x 1", L.n, L.k, L.b, L.n - L.k);
        const int rows = L.k * L.b + L.n - L.k;
        std::printf("  %-22s %9.2f %9.2f %12.1f %12.1f\n", what, md, med(ms, 0.9), rows * 1e3 / md, rows * 1e3 / md / L.n);
        stamp_line(what);
      }
    }
    // Card 0's kernel time per module (eager, profiled; medians of G steps).
    // XEN_PROBE_SEQS_PROFILE=0 skips it (a timing A/B that reads the replayed stamps saves ~3 minutes per load).
    const char *eager_env = std::getenv("XEN_PROBE_SEQS_PROFILE");
    const bool eager_profile = !(eager_env && std::string_view(eager_env) == "0") && fab.profiling();
    if (eager_profile) {
      std::printf("profile (card 0 kernel ms per step, eager, median of %d steps):\n  %-12s", G, "step");
      for (int p = 0; p < q38::StepProfile::kParts; ++p) std::printf(" %7s", q38::StepProfile::name(p));
      std::printf(" %7s %7s\n", "span", "sum");
      for (const auto &[n, r] : std::vector<std::pair<int, int>>{{1, 4}, {2, 2}, {4, 1}, {2, 4}, {4, 2}, {3, 2}, {6, 1}, {8, 1}, {8, 2}, {16, 1}, {4, 4}}) {
        if (n > nseq) continue;
        start_all(next);
        m.set_profiling(true);
        std::vector<std::vector<double>> part(q38::StepProfile::kParts), ml(q38::StepProfile::kMoeLaunches);
        std::vector<double> span;
        int32_t t[q38::Head::kRows][4];
        for (int i = 0; i < G + 2; ++i) {
          std::vector<q38::Model::SeqRows> segs;
          for (int s = 0; s < n; ++s) {
            text_rows(s, i, r, t[s], next[static_cast<size_t>(s)]);
            segs.push_back({s, t[s], r});
          }
          m.step_batch(segs);
          const auto pr = m.profile();
          if (i >= 2) {
            for (int p = 0; p < q38::StepProfile::kParts; ++p) part[static_cast<size_t>(p)].push_back(pr.ms[p]);
            for (int j = 0; j < q38::StepProfile::kMoeLaunches; ++j) ml[static_cast<size_t>(j)].push_back(pr.moe_launch[0][j]);
            span.push_back(pr.span);
          }
          const auto tk = m.tokens(n * r);
          for (int s = 0; s < n; ++s) next[static_cast<size_t>(s)] = tk[static_cast<size_t>(s * r + r - 1)];
        }
        m.set_profiling(false);
        char what[32];
        std::snprintf(what, sizeof what, "%d x %d", n, r);
        std::printf("  %-12s", what);
        double sum = 0;
        for (int p = 0; p < q38::StepProfile::kParts; ++p) {
          const double v = med(part[static_cast<size_t>(p)]);
          sum += v;
          std::printf(" %7.3f", v);
        }
        std::printf(" %7.3f %7.3f   MoE launches:", med(span), sum);  // Card 0's MoE per launch
        for (int j = 0; j < q38::StepProfile::kMoeLaunches; ++j) std::printf(" %s %.3f", q38::StepProfile::moe_name(j), med(ml[static_cast<size_t>(j)]));
        std::printf("\n");
      }
    }
  }
  std::printf("step graphs: %d recorded, %.1f ms recording in total\n", m.graphs_recorded(), m.graph_record_ms());
  check(m.wire_errors() == 0, "timing: no mailbox timeout");
  if (m.moe_cpu_on()) {  // The CPU path really served experts (the invariance checks above covered it)
    const auto st = m.moe_cpu_stats();
    std::printf("seqs: MoE CPU path served %llu requests, %llu experts%s\n", static_cast<unsigned long long>(st.requests),
                static_cast<unsigned long long>(st.experts), st.timed_out ? " (a wait timed out)" : "");
    check(!st.timed_out && st.requests > 0, "MoE CPU path: requests served, no timeout");
  }
  return rc;
}
