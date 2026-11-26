// xenithium-probe prefix: the qwen38 prefix cache, one load for every part.
// prefix --model DIR --ppl F [--mtp DIR] [--prompt-len P] [--generate G] [--seqs TURNS] [--hot-rank F]
// 0. call boundaries: a prompt's last k rows in a call of their own vs inside the previous calls (which cuts keep the
// bits: the engine only resumes at cuts that do, StepModel:snap_align);
// 1. model level: snap_save / restore after other text over the slot / fork into another slot / the engine's order
// (restore after the slot decoded past the snapshot) / live resume, each vs the cold prefill (bit for bit, or KL);
// snapshot, restore and fork costs;
// 2. engine level: an agent-like session (each turn = the previous prompt + reply + 1,500 new tokens), cache off, on, on
// with live resumes: TTFT per turn and the replies (greedy).
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

#include "probe.hpp"  // the run-wide flags
namespace q38 = xenithium::arch::qwen38;
namespace core = xenithium::core;
using clk = std::chrono::steady_clock;

namespace {

bool read_all(const std::string &path, std::vector<int32_t> &out) {
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

double kl(const std::vector<float> &p, const std::vector<float> &q) {
  if (p.size() != q.size() || p.empty()) return INFINITY;
  const double mp = *std::max_element(p.begin(), p.end()), mq = *std::max_element(q.begin(), q.end());
  double zp = 0, zq = 0;
  for (size_t i = 0; i < p.size(); ++i) zp += std::exp(p[i] - mp), zq += std::exp(q[i] - mq);
  double s = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double lp = p[i] - mp - std::log(zp), lq = q[i] - mq - std::log(zq);
    s += std::exp(lp) * (lp - lq);
  }
  return s;
}

size_t argmax(const std::vector<float> &v) { return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

struct Sink final : core::EventSink {
  std::mutex mu;
  std::condition_variable cv;
  std::map<uint64_t, std::vector<int32_t>> toks;
  std::map<uint64_t, clk::time_point> first;
  std::map<uint64_t, bool> fin;
  void on_events(const core::Event *e, size_t n) override {
    std::lock_guard<std::mutex> lk(mu);
    const auto now = clk::now();
    for (size_t i = 0; i < n; ++i) {
      auto &t = toks[e[i].id];
      if (e[i].n && !first.count(e[i].id)) first[e[i].id] = now;
      t.insert(t.end(), e[i].tokens, e[i].tokens + e[i].n);
      if (e[i].finish != core::Finish::none) fin[e[i].id] = true;
    }
    cv.notify_all();
  }
  bool wait(uint64_t id, double s) {
    std::unique_lock<std::mutex> lk(mu);
    return cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return fin.count(id) > 0; });
  }
};

}  // namespace

int cmd_prefix_run(const std::string &model, const std::string &ppl, const std::string &mtp, int64_t prompt_len, int generate, int turns,
                   const std::string &hot_rank, int threads) {
  int failures = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    failures += !ok;
  };
  std::vector<int32_t> text;
  if (model.empty() || ppl.empty() || !read_all(ppl, text)) {
    std::fprintf(stderr, "prefix: --model DIR --ppl F (a token file)\n");
    return 2;
  }
  const int64_t P = prompt_len > 0 ? prompt_len : 8192;
  const int G = generate > 0 ? generate : 32;
  if (static_cast<int64_t>(text.size()) < P + 12000) {
    std::fprintf(stderr, "prefix: the text has %zu tokens, needs --prompt-len + 12,000\n", text.size());
    return 2;
  }
  q38::ModelOptions mo;
  mo.kv_split = g_kv_tokens ? xenithium::kernels::QsaKvSplit::tokens : xenithium::kernels::QsaKvSplit::heads;
  mo.prefill_moe_a8 = g_pf_moe_a8;
  mo.prefill_moe_bias = g_pf_moe_fast;
  mo.prefill_dense_a8 = g_pf_dense_a8 != 0;
  mo.prefill_dense_a8_row = g_pf_dense_a8 == 2;
  mo.prefill_wire_bf16 = g_q38_pf_wire_bf16;
  mo.moe_dn_ksplit = g_q38_moe_dn_ks;
  mo.decode_moe_a8 = g_dec_moe_a8;
  mo.prefill_cold_stage = g_pf_cold_stage;
  mo.prefill_fetch = g_pf_fetch;
  mo.gdn_scan = g_gdn_scan;
  mo.threads = threads;
  mo.slots = 3;
  const int64_t need = P + 6000 + static_cast<int64_t>(turns) * 2200;
  mo.context = (need + 63) / 64 * 64;
  mo.kv_pool_tokens = 3 * mo.context;
  mo.kv_pool_scramble = true;
  mo.snap_gib = 4;
  mo.mtp_dir = mtp;
  mo.mtp_vocab = g_q38_mtp_vocab;
  if (!g_dflash.empty()) {  // --dflash DIR = the DFlash 2 drafter (its rings in the snapshots) instead of --mtp
    mo.dflash_dir = g_dflash;
    if (g_mtp_from > 0) mo.mtp_from = g_mtp_from;  // --mtp-from N keeps --mtp too (both drafters)
    else mo.mtp_dir.clear();
  }
  if (!hot_rank.empty()) {
    const auto c = q38::Config::parse(xenithium::model::ModelConfig::load(model));
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, tp_devices("prefix", true));  // --tp 1 (device 0)
  q38::Model m(loader, fab, model, mo);
  q38::StepModelAdapter ad(m);
  const int chunk = m.prefill_chunk();
  const int depth = m.has_drafter() || m.has_dflash() ? ad.draft_depth(1) : 0;
  std::printf("prefix: %s, context %lld, %d slots, pool %lld tokens (shuffled pages), chunk %d; drafter %s; snapshot %.2f MiB (%.2f + %.2f per card), store %d\n",
              model.c_str(), static_cast<long long>(m.capacity()), m.slots(), static_cast<long long>(m.kv_pool_tokens()), chunk, m.has_dflash() ? "DFlash" : m.has_drafter() ? "MTP" : "none",
              (m.snap_bytes(0) + (fab.size() > 1 ? m.snap_bytes(1) : 0)) / 1048576.0, m.snap_bytes(0) / 1048576.0, fab.size() > 1 ? m.snap_bytes(1) / 1048576.0 : 0.0, m.snap_capacity());
  auto wait_all = [&] {
    for (size_t c = 0; c < fab.size(); ++c) fab.compute(c).wait();
  };
  auto ms_since = [](clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };
  if (g_snap_prepin) {  // The adapter's idle work pins every snapshot id before the first save
    const auto tp = clk::now();
    int ids = 0;
    for (int n; (n = m.snap_prepin(1)) > 0;) ids += n;
    std::printf("prefix: %d snapshot ids pinned ahead of use (the adapter's idle work) in %.1f ms\n", ids, ms_since(tp));
  }
  // prefill calls over [from, to) of t as the engine sizes them: chunk rows from `from, a call also ends at `cut`; each
  // call is given the token after it when t has one
  auto calls = [&](int slot, const std::vector<int32_t> &t, int64_t from, int64_t to, int64_t cut = -1) {
    if (!m.kv_reserve(slot, to + 64)) throw std::runtime_error("prefix: pool full");
    for (int64_t p = from; p < to;) {
      // the engine's sizing (BatchEngine:pf_call_rows): chunk rows from p, a call ends at the cut, and no call is left
      // with fewer than 64 rows (snap_align) after it
      const int64_t stop = cut > p && cut < to ? cut : to;
      int64_t n = std::min<int64_t>(chunk, stop - p);
      if (const int64_t rest = stop - p - n; rest > 0 && rest < 64) n = (stop - p - 64) / 64 * 64;
      const int64_t e = p + std::max<int64_t>(n, 1);
      ad.prefill(slot, t.data() + p, static_cast<int>(e - p), e < static_cast<int64_t>(t.size()) ? t[static_cast<size_t>(e)] : -1);
      p = e;
    }
  };
  struct Out {
    std::vector<float> logits;
    std::vector<int32_t> drafts, toks;
  };
  auto finish = [&](int slot) {
    Out o;
    ad.prefill_logits(o.logits);
    int32_t t = ad.prefill_pick();
    if (depth > 0) o.drafts = ad.first_drafts(slot, t, depth, 1);
    o.toks.push_back(t);
    int64_t pos = 0;
    (void)pos;
    for (int g = 1; g < G; ++g) {
      m.kv_reserve(slot, m.kv_held_tokens(slot) + 64);
      ad.step({slot}, &t);
      ad.picks(1, &t);
      o.toks.push_back(t);
    }
    return o;
  };
  std::vector<int32_t> prompt(text.begin(), text.begin() + P);
  std::vector<int32_t> junk(text.begin() + P + 4000, text.begin() + P + 7000);
  auto same = [](const Out &a, const Out &b) { return a.logits == b.logits && a.drafts == b.drafts && a.toks == b.toks; };
  auto agree = [](const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return n;
  };

  if (std::getenv("XEN_PREFIX_REPEAT")) {  // diagnostics: the same 4,396-token prefill, again and again on slots 0 / 1 / 2
    std::vector<int32_t> pt(text.begin(), text.begin() + 4396);
    std::vector<float> l0;
    for (int rep = 0; rep < 9; ++rep) {
      const int slot = rep % 3;
      const char *pat = std::getenv("XEN_PREFIX_REPEAT");  // e.g. "010110": 1 = a cut at XEN_PREFIX_CUT (3840)
      const char *ce = std::getenv("XEN_PREFIX_CUT");
      const int64_t cut = rep < static_cast<int>(std::strlen(pat)) && pat[rep] == '1' ? (ce ? std::atoll(ce) : 3840) : -1;
      ad.begin(slot);
      q38::Model::PfTrace tr;  // XEN_PREFIX_TRACE: the last call's first 16 rows at every trace point, vs rep 0's
      tr.rows = 16;
      static q38::Model::PfTrace tr0;
      if (std::getenv("XEN_PREFIX_TRACE")) {
        calls(slot, pt, 0, 4096, cut);
        m.trace_prefill(&tr);
        calls(slot, pt, 4096, 4396);
        m.trace_prefill(nullptr);
        if (rep == 0) tr0 = tr;
        int shown = 0;
        for (size_t i = 0; i < tr.names.size() && i < tr0.names.size() && shown < 6; ++i)
          if (tr.data[i] != tr0.data[i]) {
            size_t d = 0;
            for (size_t j = 0; j < tr.data[i].size(); ++j) d += tr.data[i][j] != tr0.data[i][j];
            std::printf("  trace: %s differs (%zu of %zu values)\n", tr.names[i].c_str(), d, tr.data[i].size());
            ++shown;
          }
      } else {
        calls(slot, pt, 0, 4396, cut);
      }
      std::vector<float> l;
      ad.prefill_logits(l);
      if (rep == 0) l0 = l;
      std::printf("repeat %d (slot %d, cut %lld): %s (KL %.3g)\n", rep, slot, static_cast<long long>(cut), l == l0 ? "same" : "differs", kl(l0, l));
    }
    return 0;
  }
  if (std::getenv("XEN_PREFIX_DIAG")) {  // diagnostics: the state at 4,096 after calls with and without a cut, piece by piece
    std::vector<int32_t> pt(text.begin(), text.begin() + 4096 + 64);
    const int64_t cut = std::atoll(std::getenv("XEN_PREFIX_DIAG"));
    ad.begin(0);
    calls(0, pt, 0, 4096);
    const int a = m.snap_save(0);
    ad.begin(1);
    calls(1, pt, 0, 4096, cut);
    const int b = m.snap_save(1);
    m.snap_sync();
    for (int c = 0; c < static_cast<int>(fab.size()); ++c) {
      const auto pcs = m.snap_piece_bytes(c);
      const auto *x = static_cast<const unsigned char *>(m.snap_host(a, c)), *y = static_cast<const unsigned char *>(m.snap_host(b, c));
      size_t off = 0;
      for (size_t i = 0; i < pcs.size(); ++i) {
        size_t d = 0;
        for (size_t j = 0; j < pcs[i]; ++j) d += x[off + j] != y[off + j];
        if (d) std::printf("diag: card %d piece %zu (%zu bytes): %zu bytes differ\n", c, i, pcs[i], d);
        off += pcs[i];
      }
    }
    std::printf("diag: done (%zu pieces per card)\n", m.snap_piece_bytes(0).size());
    std::vector<int32_t> tail(text.begin() + 4096, text.begin() + 4396);
    auto last = [&](int slot) {
      m.kv_reserve(slot, 4396 + 64);
      ad.prefill(slot, tail.data(), 300, -1);
      std::vector<float> l;
      ad.prefill_logits(l);
      return l;
    };
    const auto la = last(0);
    const auto lb = last(1);
    m.kv_trim(1, 4096);
    m.snap_restore(1, a);  // slot 1: the cut run's KV pages, the uncut run's state
    const auto lc = last(1);
    m.kv_trim(0, 4096);
    m.snap_restore(0, b);  // slot 0: the uncut run's KV pages, the cut run's state
    const auto ld = last(0);
    std::printf("diag: last call logits: cut vs uncut %s (KL %.3g); cut KV + uncut state %s; uncut KV + cut state %s\n", la == lb ? "same" : "differ", kl(la, lb),
                lc == la ? "== uncut" : lc == lb ? "== cut" : "neither", ld == la ? "== uncut" : ld == lb ? "== cut" : "neither");
    return 0;
  }
  // ---- 0. call boundaries ----
  {
    const int64_t Pt = 4096 + 300;  // 4,396 tokens: calls of 512 rows from 0, the last 300
    std::vector<int32_t> pt(text.begin(), text.begin() + Pt);
    ad.begin(0);
    calls(0, pt, 0, Pt);
    std::vector<float> l0;
    ad.prefill_logits(l0);
    std::string line;
    for (int64_t cut : {int64_t{4096 + 128}, int64_t{4096 + 256}, int64_t{4096 - 64}, int64_t{4096 - 256}, int64_t{3584 + 64}, int64_t{4096 + 64}, int64_t{4096 + 37},
                        int64_t{4096 + 4}, int64_t{1024}, int64_t{2048 + 192}}) {
      ad.begin(0);
      calls(0, pt, 0, Pt, cut);
      std::vector<float> l1;
      ad.prefill_logits(l1);
      const bool eq = l0 == l1;
      char b[128];
      std::snprintf(b, sizeof b, " %lld: %s (KL %.2g)", static_cast<long long>(cut), eq ? "same" : "differs", kl(l0, l1));
      line += b;
      // the engine's sizing keeps every call at 64+ rows around a cut at a multiple of 64 (pf_call_rows): such a cut, and
      // the call boundaries it shifts, must keep the bits
      if (cut % 64 == 0 && cut >= 64 && Pt - cut >= 64) check(eq, "a cut at " + std::to_string(cut) + " (a multiple of 64; every call 64+ rows) keeps the bits");
    }
    std::printf("call boundary: a %lld-token prompt in %d-row calls, one extra cut; last-row logits vs without it:%s\n", static_cast<long long>(Pt), chunk, line.c_str());
  }
  // ---- 1. model level ----
  ad.begin(0);
  auto t0 = clk::now();
  calls(0, prompt, 0, P);
  wait_all();
  const double cold_ms = ms_since(t0);
  const Out ref = finish(0);
  std::printf("cold: %lld-token prompt prefilled in %.0f ms (%.0f tok/s); greedy %d tokens, %zu first drafts\n", static_cast<long long>(P), cold_ms, P / cold_ms * 1e3, G,
              ref.drafts.size());

  const int64_t R = (P / 2) / chunk * chunk;
  ad.begin(1);
  calls(1, prompt, 0, R);
  wait_all();
  t0 = clk::now();
  const int id1 = m.snap_save(1);
  wait_all();
  const double save_ms0 = ms_since(t0);
  double save_ms = 0;
  {
    const int idx = m.snap_save(1);
    m.snap_sync();
    m.snap_free(idx);
    wait_all();
    t0 = clk::now();
    const int idy = m.snap_save(1);
    m.snap_sync();
    save_ms = ms_since(t0);
    m.snap_free(idy);
  }
  check(id1 >= 0 && m.snap_pos(id1) == R, "snap_save at " + std::to_string(R) + ": id " + std::to_string(id1));
  calls(1, junk, 0, static_cast<int64_t>(junk.size()));
  m.kv_trim(1, R);
  t0 = clk::now();
  m.snap_restore(1, id1);
  wait_all();
  const double restore_ms = ms_since(t0);
  t0 = clk::now();
  calls(1, prompt, R, P);
  wait_all();
  const double tail_ms = ms_since(t0);
  const Out a1 = finish(1);
  check(same(a1, ref), "restore at " + std::to_string(R) + " after 3,000 other tokens over the slot: logits, " + std::to_string(a1.drafts.size()) + " first drafts and " +
                           std::to_string(G) + " greedy tokens == the cold prefill, bit for bit");
  std::printf("  snap_save %.2f ms with its host copy (first save of an id %.2f, pinned allocation), snap_restore %.2f ms (%.2f MiB each way); the tail [%lld, %lld) in %.0f ms vs %.0f cold\n",
              save_ms, save_ms0, restore_ms, (m.snap_bytes(0) + (fab.size() > 1 ? m.snap_bytes(1) : 0)) / 1048576.0, static_cast<long long>(R), static_cast<long long>(P), tail_ms, cold_ms);
  // fork: slot 1 holds [0, R) under the snapshot (and its own tail past R)
  ad.begin(2);
  m.kv_reserve(2, R + 64);
  wait_all();
  t0 = clk::now();
  m.snap_fork(2, 1, id1);
  const double fork_ms = ms_since(t0);
  calls(2, prompt, R, P);
  const Out a2 = finish(2);
  check(same(a2, ref), "fork of slot 1's snapshot at " + std::to_string(R) + " into slot 2 (KV pages copied): == the cold prefill, bit for bit");
  std::printf("  snap_fork %.2f ms (%lld positions: target + drafter pages, the state)\n", fork_ms, static_cast<long long>(R));
  // a resume point at a multiple of 64 inside a chunk (the call before it ends there)
  {
    const int64_t R3 = R + chunk / 2 + 64;
    ad.begin(1);
    calls(1, prompt, 0, R3);
    const int id3 = m.snap_save(1);
    calls(1, junk, 0, 1000);
    m.kv_trim(1, R3);
    m.snap_restore(1, id3);
    calls(1, prompt, R3, P);
    const Out a4 = finish(1);
    std::printf("  restore at %lld (inside a chunk): last-row KL %.3g vs cold, greedy tokens equal for %zu of %d\n", static_cast<long long>(R3), kl(ref.logits, a4.logits),
                agree(ref.toks, a4.toks), G);
    // the same call boundaries cold
    ad.begin(2);
    calls(2, prompt, 0, P, R3);
    const Out c4 = finish(2);
    check(same(a4, c4), "restore at " + std::to_string(R3) + " (inside a chunk): == a cold run with a call cut there, bit for bit");
    check(same(a4, ref) || !(R3 % 64 == 0), "restore at " + std::to_string(R3) + " (a multiple of 64 inside a chunk): == the cold prefill, bit for bit");
    m.snap_free(id3);
  }
  // the engine's order: the snapshot after a call that ends at Rm, the prompt's last call, decode steps; then restore at
  // Rm and a different continuation from there, vs its cold prefill
  {
    const int64_t Rm = P / 2 / 64 * 64 + 64;
    std::vector<int32_t> p2(prompt.begin(), prompt.begin() + Rm + 4);
    std::vector<int32_t> p3(prompt.begin(), prompt.begin() + Rm);
    p3.insert(p3.end(), text.begin() + P + 9000, text.begin() + P + 10500);
    ad.begin(2);
    calls(2, p3, 0, static_cast<int64_t>(p3.size()), Rm);
    const Out c3 = finish(2);
    ad.begin(1);
    calls(1, p2, 0, Rm);
    const int id4 = m.snap_save(1);
    calls(1, p2, Rm, static_cast<int64_t>(p2.size()));
    (void)finish(1);
    m.kv_trim(1, Rm);
    m.snap_restore(1, id4);
    calls(1, p3, Rm, static_cast<int64_t>(p3.size()));
    const Out a5 = finish(1);
    std::printf("  restore at %lld after the slot's own last call and %d decode steps: last-row KL %.3g vs cold, greedy tokens equal for %zu of %d\n",
                static_cast<long long>(Rm), G - 1, kl(c3.logits, a5.logits), agree(c3.toks, a5.toks), G);
    check(same(a5, c3), "restore at " + std::to_string(Rm) + " after the slot decoded past it (the engine's order): == the cold prefill, bit for bit");
    m.snap_free(id4);
  }
  // live: slot 0 continues after its G tokens with 1,000 new tokens; vs a cold prefill of the whole sequence
  {
    std::vector<int32_t> more(text.begin() + P + 8000, text.begin() + P + 9000);
    std::vector<int32_t> cont{ref.toks.back()};
    cont.insert(cont.end(), more.begin(), more.end());
    check(m.live_ok(0), "slot 0's live state after its greedy steps can take a prefill call");
    const int64_t at = P + G - 1;
    m.kv_reserve(0, at + static_cast<int64_t>(cont.size()) + 64);
    t0 = clk::now();
    for (int64_t p = 0; p < static_cast<int64_t>(cont.size());) {
      const int64_t e = std::min<int64_t>(static_cast<int64_t>(cont.size()), p + chunk);
      ad.prefill(0, cont.data() + p, static_cast<int>(e - p), e < static_cast<int64_t>(cont.size()) ? cont[static_cast<size_t>(e)] : -1);
      p = e;
    }
    wait_all();
    const double live_ms = ms_since(t0);
    std::vector<float> l0;
    ad.prefill_logits(l0);
    std::vector<int32_t> whole = prompt;
    whole.insert(whole.end(), ref.toks.begin(), ref.toks.end());
    whole.insert(whole.end(), more.begin(), more.end());
    ad.begin(1);
    t0 = clk::now();
    calls(1, whole, 0, static_cast<int64_t>(whole.size()));
    wait_all();
    const double whole_ms = ms_since(t0);
    std::vector<float> l1;
    ad.prefill_logits(l1);
    const double k = kl(l1, l0);
    std::printf("  live resume at %lld + 1,001 tokens: %.0f ms vs %.0f ms for the whole %zu-token sequence cold; last-row KL %.3g, top-1 %s\n",
                static_cast<long long>(at), live_ms, whole_ms, whole.size(), k, argmax(l0) == argmax(l1) ? "same" : "differs");
    // The bound was 0.01, below the noise it measures: the live slot's 31 rows come from decode steps, the
    // cold run's from prefill, and those two paths differ by a mean row KL of 0.034 (TP1 0.0339, TP2 0.0331; long-ppl,
    // 8,192 rows). TP2 read 0.0024-0.0098 here, TP1 0.0152 (its live sessions replay their replies like TP2's). A broken
    // live state reads > 1.
    check(std::isfinite(k) && k < 0.05 && argmax(l0) == argmax(l1), "live resume vs the whole sequence cold: KL < 0.05 and the same top-1");
  }
  for (int64_t r : {int64_t{4096}, int64_t{8192}, int64_t{16384}}) {
    if (r + 64 > m.capacity()) continue;
    ad.begin(1);
    calls(1, text, 0, r);
    const int id = m.snap_save(1);
    ad.begin(2);
    m.kv_reserve(2, r + 64);
    wait_all();
    t0 = clk::now();
    m.snap_fork(2, 1, id);
    std::printf("  snap_fork at %lld positions: %.2f ms\n", static_cast<long long>(r), ms_since(t0));
    m.snap_free(id);
  }
  m.snap_free(id1);

  // ---- 2. engine level ----
  for (int s = 0; s < m.slots(); ++s) ad.begin(s);
  if (turns > 0) {
    const std::vector<int32_t> gen{text[0], text[1], text[2], text[3]};
    struct TurnRes {
      double ttft_ms = 0;
      size_t prompt = 0;
      std::vector<int32_t> out;
    };
    auto session = [&](bool cache, bool live, std::vector<TurnRes> &res, core::Stats &st, const std::vector<std::vector<int32_t>> *forced) {
      core::BatchConfig bc;
      bc.prefix_live = live;
      bc.eos_ids = {};
      bc.log_interval_s = 0;
      bc.prefix_cache = cache;
      bc.prefill_part_ms = 25;
      core::BatchEngine e(ad, bc);
      Sink sink;
      e.start(&sink);
      std::vector<int32_t> pr = prompt;
      pr.insert(pr.end(), gen.begin(), gen.end());
      for (int k = 0; k < turns; ++k) {
        core::SamplingParams p;
        p.temperature = 0;
        p.ignore_eos = true;
        p.max_tokens = G;
        p.prefix_mark = static_cast<int32_t>(pr.size() - gen.size());
        const uint64_t id = static_cast<uint64_t>(k + 1);
        const auto ts = clk::now();
        e.submit(id, pr, p);
        sink.wait(id, 600);
        TurnRes r;
        {
          std::lock_guard<std::mutex> lk(sink.mu);
          r.ttft_ms = std::chrono::duration<double, std::milli>(sink.first[id] - ts).count();
          r.out = sink.toks[id];
        }
        r.prompt = pr.size();
        res.push_back(r);
        const auto &reply = forced ? (*forced)[static_cast<size_t>(k)] : r.out;
        pr.insert(pr.end(), reply.begin(), reply.end());
        const size_t at = static_cast<size_t>(P) + 1000 + static_cast<size_t>(k) * 1500;
        pr.insert(pr.end(), text.begin() + static_cast<std::ptrdiff_t>(at), text.begin() + static_cast<std::ptrdiff_t>(at + 1500));
        pr.insert(pr.end(), gen.begin(), gen.end());
      }
      e.stop();
      st = e.stats();
    };
    std::vector<TurnRes> off, on, lv, off2;
    core::Stats st_off, st_on, st_lv, st_off2;
    session(false, false, off, st_off, nullptr);
    std::vector<std::vector<int32_t>> replies;
    for (const auto &r : off) replies.push_back(r.out);
    session(true, false, on, st_on, &replies);
    session(true, true, lv, st_lv, &replies);
    session(false, false, off2, st_off2, &replies);
    {
      std::string eq;
      for (size_t k = 0; k < off.size() && k < off2.size(); ++k) eq += " " + std::to_string(agree(off[k].out, off2[k].out));
      std::printf("  without the cache, a second run: reply tokens equal per turn:%s (of %d)\n", eq.c_str(), G);
    }
    std::printf("agent session, %d turns (greedy, %d tokens per reply; turn k + 1 = turn k's prompt + reply + 1,500 new tokens + a 4-token generation prompt)\n", turns, G);
    std::printf("  turn   prompt tokens   TTFT off ms   TTFT on ms   speed-up   reply tokens equal   on + live: TTFT ms   reply tokens equal\n");
    double sum_off = 0, sum_on = 0, sum_lv = 0;
    bool first_same = true;
    for (int k = 0; k < turns; ++k) {
      const auto &a = off[static_cast<size_t>(k)], &b = on[static_cast<size_t>(k)], &c = lv[static_cast<size_t>(k)];
      std::printf("  %4d   %13zu   %11.0f   %10.0f   %7.1fx   %8zu of %zu   %18.0f   %8zu of %zu\n", k + 1, a.prompt, a.ttft_ms, b.ttft_ms, a.ttft_ms / b.ttft_ms,
                  agree(a.out, b.out), a.out.size(), c.ttft_ms, agree(a.out, c.out), a.out.size());
      if (k > 0) sum_off += a.ttft_ms, sum_on += b.ttft_ms, sum_lv += c.ttft_ms;
      first_same &= !a.out.empty() && !b.out.empty() && a.out[0] == b.out[0];
    }
    auto pcs = [](const core::Stats &st) {
      char b[200];
      std::snprintf(b, sizeof b, "%llu of %llu prompt tokens hit (%llu live, %llu restores, %llu forks)", static_cast<unsigned long long>(st.prefix_hits),
                    static_cast<unsigned long long>(st.prefix_queries), static_cast<unsigned long long>(st.prefix_live),
                    static_cast<unsigned long long>(st.prefix_restores), static_cast<unsigned long long>(st.prefix_forks));
      return std::string(b);
    };
    std::printf("  turns 2..%d: TTFT %.0f ms off, %.0f ms on (%s), %.0f ms on + live (%s)\n", turns, sum_off, sum_on, pcs(st_on).c_str(), sum_lv, pcs(st_lv).c_str());
    check(off.size() == on.size() && !on.empty() && on[0].out == off[0].out, "turn 1 (nothing cached): the same reply with the cache on");
    check(first_same, "every turn's first reply token is the same with the cache on");
    bool all_same = off.size() == on.size();
    for (size_t k = 0; all_same && k < off.size(); ++k) all_same = off[k].out == on[k].out;
    check(all_same, "every turn's whole reply is the same with the cache on (resumes at multiples of 64)");
    check(st_on.prefix_hits > 0 && st_off.prefix_hits == 0 && sum_on < sum_off, "turns 2.. continue a cached prefix and reach their first token sooner");
  }
  check(m.wire_errors() == 0, "no mailbox timeout");
  std::printf("prefix: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
