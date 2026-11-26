// xenithium-probe model: the whole model on both cards, one load for every run.
// --ppl FILE teacher-forced perplexity on a token file (int32 n_chunks, then per chunk int32
// length + ids; tools/bench/ppl_tokens.py in the draft repo writes it from the gate's
// corpus). Each chunk is a fresh sequence from position 0; a step feeds R consecutive
// tokens and row j's logits score token p0 + j + 1. As the reference gate (echo +
// max_tokens 1), the greedy token after the chunk's last token is scored too: n scored
// tokens for an n-token chunk. Logits are rounded to bf16 (the reference's dtype),
// log-softmax in double. Runs once per row count in --rows (default 1,4); "pf" = the
// prefill path (Model:prefill, layer-major chunks of --pf-chunk rows, head on every
// 4-row slice): its per-chunk nll must equal rows 4's bit for bit.
// --generate N plain greedy decode, 1 row per step, after chunk 0 of the token file as the prompt
// (or a fixed prompt): ms per step (host wall and card 0 GPU span), the tokens. Then
// N steps of 4 rows (the verify shape of a token + 3 drafts) on the corpus text.
// --at P history before the timed steps: the corpus tokens (cycled) in 4-row steps up to
// position P (default: the prompt only). The context grows to fit P + the steps.
// --routes-out F routing counts of every step run (ppl and generate) -> F (XRC1, see model.hpp):
// the calibration input of --hot-rank.
// --hot-rank F place the hot experts by these routing counts (global hot per card) instead of
// the even split.
// --reserve G VRAM (GiB per card) left free after the experts (default ModelOptions, 0.25).
// --lp-out DIR per scored row the top 32 log-probs + the target's (eval/logprobs.hpp, XLP1) ->
// DIR/<token file stem>.rows<R>.xlp, for `xenithium-kl compare.
// --lp-ref F.. reference dumps (`xenithium-kl dump`), one per --ppl file in order: KL(ref || this)
// per position bucket, exact on this side (full logits), lower bound on the ref's
// truncated tail; top-1 agreement. Rows must line up (same token file).
// --bf16 P,P numerics switch: int8 dense-plan matrices whose tensor name contains one of
// these substrings load as bf16 (e.g. input_mix_weight_down, ple.key_proj).
// --moe-down-bf16 1 numerics switch: routed experts' down outputs rounded to bf16 (reference); under
// expert TP each card rounds its partial, the prefill path's rounding point.
// --moe-dn-ks N MoE down k split (default 2); 1 = a null variant (same precision, another fp32
// summation order): the KL floor of any bit-level change.
// --graphs 0|1 replay each step shape as recorded SYCL graphs (default 1); profiled steps
// (the first --generate pass) always run eagerly. --generate then times a second
// pass of 2N 1-row and 2N 4-row steps without profiling; with --graphs 1 the steps
// alternate graph replay and eager submission (paired A/B on one load).
// --hidden-out D the drafter's inputs from the first --rows pass of each --ppl file -> D/<stem>.xmh:
// uint32 'XMH1', int32 width (hc x hidden), int32 chunks; per chunk int32 len, then
// per position int32 token, int32 the target's greedy pick, bf16 multi-stream
// hidden [width] (the state after the final mixer's combine).
// --gen-out F the prompt + the plain greedy tokens of --generate as a one-chunk token file (the
// --ppl format; needs no --at history): scored with --hidden-out it gives the
// drafter's inputs of a real greedy decode (1 row == 4 rows bit for bit).
// --moe-tp 0|1 expert TP (default 1): each card computes half the columns of every routed
// expert; 0 = EP2, each card owns half the experts.
// Judge a switch by --lp-ref against a dump of the shipped path.
// --pf-route-trace F every chunk of every --ppl file through prefill calls of
// --pf-chunk rows (a fresh sequence per chunk), card 0's MoE-side rows per call -> F,
// then exit. XLA1, little endian: uint32 'XLA1', int32 layers, experts, topk, hidden,
// uint64 a routed expert's blob bytes per card (TP); then per call int32 file index,
// chunk index, first row's position, rows m; per layer bf16 xa [m][hidden] (the
// attention mixer's output), bf16 xm [m][hidden] (the MLP mixer's output = the
// router's input), int32 ids [m][topk] (the MoE's routing). For xenithium-lookahead.
// The timed steps also report the MoE split: each launch's time on both cards and the cold
// experts per step (unique routed experts not in VRAM: per card per layer, and the critical path
// = sum over layers of the larger card's count, since the combine waits for the slower card).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xenithium/arch/qwen38/model.hpp"
#include "xenithium/eval/logprobs.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/device.hpp"
#include "kv.hpp"

#include "probe.hpp"  // the run-wide flags
namespace q38 = xenithium::arch::qwen38;
// pf_graphs.cpp: --pf-graph-gate
int pf_graph_gate(q38::Model &m, const std::vector<int32_t> &tok, int calls, const std::vector<int> &sizes);

namespace {

bool read_chunks(const std::string &path, std::vector<std::vector<int32_t>> &chunks) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  int32_t n = 0;
  if (!f.read(reinterpret_cast<char *>(&n), 4) || n <= 0 || n > 100000) return false;
  chunks.resize(static_cast<size_t>(n));
  for (auto &c : chunks) {
    int32_t len = 0;
    if (!f.read(reinterpret_cast<char *>(&len), 4) || len <= 1 || len > (1 << 24)) return false;
    c.resize(static_cast<size_t>(len));
    if (!f.read(reinterpret_cast<char *>(c.data()), static_cast<std::streamsize>(len) * 4)) return false;
  }
  return true;
}

double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

}  // namespace

// A --rows entry as printed: 1..4, or "pf" (0) for the prefill path.
std::string rlabel(int R) { return R == 0 ? std::string("pf") : std::to_string(R); }

int cmd_model_run(const std::string &model, const std::string &ppl, const std::string &rows_arg, int generate, int64_t at, int64_t context,
                  int kv_bytes, int64_t hot, int threads, bool verify, const std::string &routes_out, const std::string &hot_rank, double reserve_gib,
                  const std::string &lp_out, const std::string &lp_ref, const std::string &bf16, bool moe_down_bf16, int moe_dn_ks, bool graphs, bool moe_tp,
                  const std::string &hidden_out, const std::string &gen_out, int pf_chunk, bool pf_time_only,
                  const std::string &pf_sweep) {
  namespace ev = xenithium::eval;
  if (model.empty()) {
    std::fprintf(stderr, "model: --model DIR is required\n");
    return 2;
  }
  // --ppl takes comma-separated token files, each scored and reported on its own.
  std::vector<std::string> files;
  std::vector<std::vector<std::vector<int32_t>>> sets;
  {
    std::stringstream ss(ppl);
    std::string f;
    while (std::getline(ss, f, ','))
      if (!f.empty()) {
        sets.emplace_back();
        if (!read_chunks(f, sets.back())) {
          std::fprintf(stderr, "model: cannot read token file %s\n", f.c_str());
          return 2;
        }
        files.push_back(f.substr(f.find_last_of('/') + 1));
      }
  }
  std::vector<ev::LpSet> refs;  // --lp-ref, one per token file
  {
    std::stringstream ss(lp_ref);
    std::string f;
    while (std::getline(ss, f, ','))
      if (!f.empty()) {
        refs.emplace_back();
        std::string err;
        if (!ev::read_lp(f, refs.back(), &err)) {
          std::fprintf(stderr, "model: %s\n", err.c_str());
          return 2;
        }
      }
    if (!refs.empty() && refs.size() != sets.size()) {
      std::fprintf(stderr, "model: --lp-ref takes one dump per --ppl file (%zu vs %zu)\n", refs.size(), sets.size());
      return 2;
    }
    for (size_t i = 0; i < refs.size(); ++i) {
      bool ok = refs[i].chunks.size() == sets[i].size();
      for (size_t c = 0; ok && c < sets[i].size(); ++c) {
        ok &= refs[i].chunks[c].size() == sets[i][c].size();
        for (size_t j = 0; ok && j + 1 < sets[i][c].size(); ++j) ok &= refs[i].chunks[c][j].target == sets[i][c][j + 1];
      }
      if (!ok) {
        std::fprintf(stderr, "model: --lp-ref dump %zu does not line up with %s (chunks, rows or targets differ)\n", i, files[i].c_str());
        return 2;
      }
    }
  }
  const std::vector<std::vector<int32_t>> chunks = sets.empty() ? std::vector<std::vector<int32_t>>{} : sets[0];
  std::vector<int> row_list;
  {
    std::stringstream ss(rows_arg);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) row_list.push_back(t == "pf" ? 0 : std::stoi(t));
    for (int r : row_list)
      if (r < 0 || r > 4) {
        std::fprintf(stderr, "model: --rows takes 1..4 or pf\n");
        return 2;
      }
  }
  q38::ModelOptions mo;
  mo.kv_split = g_kv_tokens ? xenithium::kernels::QsaKvSplit::tokens : xenithium::kernels::QsaKvSplit::heads;
  mo.prefill_moe_a8 = g_pf_moe_a8;
  mo.prefill_moe_bias = g_pf_moe_fast;
  mo.prefill_dense_a8 = g_pf_dense_a8 != 0;
  mo.prefill_dense_a8_row = g_pf_dense_a8 == 2;
  mo.prefill_wire_bf16 = g_q38_pf_wire_bf16;
  mo.decode_moe_a8 = g_dec_moe_a8;
  mo.kv_pool_tokens = g_kv_pool;
  mo.kv_lend = g_kv_lend, mo.kv_lend_keep = g_kv_lend_keep;
  mo.kv_pool_scramble = g_kv_scramble;
  mo.prefill_cold_stage = g_pf_cold_stage;
  mo.prefill_fetch = g_pf_fetch;
  mo.prefill_graphs = g_pf_graphs || g_pf_graph_gate != 0;
  if (g_pf_graph_gate > 0) mo.slots = std::max(mo.slots, 2);  // the gate's script runs two slots (< 0: the timing alone)
  mo.gdn_scan = g_gdn_scan;
  mo.moe_cpu.mode = g_moe_cpu, mo.moe_cpu.pct = g_moe_cpu_pct, mo.moe_cpu.threads = g_moe_cpu_threads;
  mo.moe_cpu.min_cold = g_moe_cpu_min_cold;
  int64_t longest = 0;
  for (const auto &st : sets)
    for (const auto &c : st) longest = std::max<int64_t>(longest, static_cast<int64_t>(c.size()));
  mo.context = std::max<int64_t>({context, longest, generate > 0 ? std::max<int64_t>(at, longest) + 16LL * generate + 64 : int64_t{0}});
  mo.kv = probe_kv(kv_bytes);
  mo.hot = hot;
  mo.threads = threads;
  mo.verify_experts = verify;
  if (reserve_gib >= 0) mo.reserve_gib = reserve_gib;
  {
    std::stringstream ss(bf16);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) mo.bf16_tensors.push_back(t);
  }
  {
    std::stringstream ss(g_w4_tensors);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) mo.w4_tensors.push_back(t);
  }
  mo.w4_mse = g_w4_mse != 0;
  mo.gdn_w4 = g_gdn_w4;
  mo.moe_down_bf16 = moe_down_bf16;
  mo.moe_dn_ksplit = moe_dn_ks;
  mo.moe_gu_ksplit = g_moe_gu_ks;
  mo.graphs = graphs;
  mo.moe_tp = moe_tp;
  if (pf_chunk > 0) mo.prefill_chunk = pf_chunk;
  if (!hot_rank.empty()) {
    const auto mc = xenithium::model::ModelConfig::load(model);
    const auto c = q38::Config::parse(mc);
    mo.hot_rank = q38::read_route_counts(hot_rank, c.layers, c.experts);
  }
  const auto loader = xenithium::l0::Loader::open();
  // XEN_PROBE_PROFILING=0 runs the queues without event profiling, as the server does (a profiling queue costs
  // ~17 us of host time per launch against ~1.9: short prefill calls become host-bound); the eager per-kernel profile
  // and XEN_PF_PROFILE need it on (the default)
  const char *pq = std::getenv("XEN_PROBE_PROFILING");
  // --tp 1 = TP1 on device 0 (pick the card with ZE_AFFINITY_MASK)
  xenithium::Fabric fab(loader, tp_devices("model", true), !(pq && std::string(pq) == "0"));
  q38::Model m(loader, fab, model, mo);
  const auto &ls = m.load_stats();
  const auto &cfg = m.config();
  // ----: --gdn-cap FILE, the GDN trunk's calibration inputs from the eager decode steps of the scoring
  // runs (use --rows 4). One of every --gdn-cap-every steps, up to --gdn-cap-rows rows per layer. FILE = XGC1: uint32
  // 'XGC1', int32 hidden, int32 value width, int32 linear layers, then records (int32 layer, int32 rows, bf16 x [rows][hidden],
  // bf16 core [rows][value width]); `xenithium-bake gdn4-calib` reads it. ----
  std::ofstream gdn_cap_f;
  std::vector<int64_t> gdn_cap_n(static_cast<size_t>(cfg.layers), 0);
  int64_t gdn_cap_steps = 0;
  int gdn_first = -1, gdn_layers = 0;
  for (int l = 0; l < cfg.layers; ++l)
    if (!cfg.full_attention[l]) gdn_first = gdn_first < 0 ? l : gdn_first, ++gdn_layers;
  if (!g_gdn_cap.empty()) {
    gdn_cap_f.open(g_gdn_cap, std::ios::binary);
    if (!gdn_cap_f) throw std::runtime_error("model: cannot write " + g_gdn_cap);
    const uint32_t magic = 0x31434758u;  // 'XGC1'
    const int32_t hdr[3] = {cfg.hidden, cfg.lin_v_heads * cfg.lin_v_dim, gdn_layers};
    gdn_cap_f.write(reinterpret_cast<const char *>(&magic), 4).write(reinterpret_cast<const char *>(hdr), 12);
    m.set_graphs(false);
    m.set_gdn_capture([&](int l, int rows, const uint16_t *x, const uint16_t *core) {
      if (l == gdn_first) ++gdn_cap_steps;
      if ((gdn_cap_steps - 1) % std::max(1, g_gdn_cap_every) != 0 || gdn_cap_n[static_cast<size_t>(l)] >= g_gdn_cap_rows) return true;
      const int32_t rec[2] = {l, rows};
      gdn_cap_f.write(reinterpret_cast<const char *>(rec), 8);
      gdn_cap_f.write(reinterpret_cast<const char *>(x), static_cast<std::streamsize>(rows) * cfg.hidden * 2);
      gdn_cap_f.write(reinterpret_cast<const char *>(core), static_cast<std::streamsize>(rows) * cfg.lin_v_heads * cfg.lin_v_dim * 2);
      gdn_cap_n[static_cast<size_t>(l)] += rows;
      for (int j = 0; j < cfg.layers; ++j)
        if (!cfg.full_attention[j] && gdn_cap_n[static_cast<size_t>(j)] < g_gdn_cap_rows) return true;
      std::printf("model: --gdn-cap: %d rows of each of %d GDN layers -> %s (after %lld steps)\n", g_gdn_cap_rows, gdn_layers, g_gdn_cap.c_str(),
                  static_cast<long long>(gdn_cap_steps));
      gdn_cap_f.close();
      return false;
    });
  }
  std::string hot_cards;  // Every card of the fabric (was cards 0 and 1)
  for (size_t c = 0; c < fab.size(); ++c)
    hot_cards += (c ? ", " : "") + std::to_string(ls.hot[c]) + " / " + std::to_string(ls.hot[c] + ls.cold[c]) + " (card " + std::to_string(c) + ")";
  std::printf("model: loaded in %.1f s (dense %.1f s, experts %.1f s); context %lld, KV %s; free after dense + states %.2f GiB per card -> "
              "hot experts %s, placed by %s; cold read in place over PCIe\n",
              ls.total_s, ls.dense_s, ls.experts_s, static_cast<long long>(m.capacity()), probe_kv_name(mo.kv), ls.free_after_dense_gib, hot_cards.c_str(),
              hot_rank.empty() ? "even split per layer" : hot_rank.c_str());
  std::printf("model: MoE CPU path %s\n", ls.moe_cpu.c_str());
  std::printf("model: KV lending %s\n", ls.kv_lend.c_str());
  if (!mo.bf16_tensors.empty() || mo.moe_down_bf16 || moe_dn_ks > 0)
    std::printf("model: numerics switches: %d dense matrices int8 -> bf16 (--bf16 %s); MoE down outputs %s, down k split %s\n", ls.bf16_overrides,
                bf16.empty() ? "-" : bf16.c_str(), moe_down_bf16 ? "bf16 (reference)" : "fp32 (shipped)",
                moe_dn_ks > 0 ? std::to_string(moe_dn_ks).c_str() : "default");
  if (!mo.w4_tensors.empty())
    std::printf("model: numerics switch (engine round 122): %d dense matrices int8 -> int4 g128 RTN (--w4 %s)\n", ls.w4_overrides, g_w4_tensors.c_str());
  std::printf("model: MoE card balance: %s\n", g_tp == 1 ? "TP1 (every expert whole on the one card)"
                                                : g_tp == 4 ? "TP4 (each card 1/4 of every expert, 32-column quarters)"
                                                : moe_tp    ? "expert TP (each card 1/2 of every expert)"
                                                            : "EP2 (each card owns half the experts)");
  int rc = 0;
  // ---- routing: counts for --routes-out, cold experts per step ----
  const int L = cfg.layers, E = cfg.experts, TK = cfg.topk, per = E / 2;
  const bool want_routes = !routes_out.empty() || generate > 0;
  std::vector<double> counts(static_cast<size_t>(L) * E, 0.0);
  std::vector<int32_t> ids;
  struct Cold {
    double crit = 0, unique = 0, routed = 0, routes = 0;  // per step
    double imb = 0;  // sum over layers of |unique experts on card 0 - on card 1| (EP imbalance)
    std::vector<int> diff;  // [layer] unique experts on card 0 - on card 1
  };
  auto read_routes = [&](int rows) {
    Cold c;
    if (!want_routes) return c;
    m.route_ids(rows, ids);
    std::vector<char> seen(static_cast<size_t>(E));
    for (int l = 0; l < L; ++l) {
      std::fill(seen.begin(), seen.end(), 0);
      int cold[2] = {0, 0}, owned[2] = {0, 0};
      for (int j = 0; j < rows * TK; ++j) {
        const int e = ids[static_cast<size_t>(l) * rows * TK + j];
        counts[static_cast<size_t>(l) * E + e] += 1;
        const bool h = m.expert_hot(l, e);
        c.routes += !h;
        if (seen[static_cast<size_t>(e)]) continue;
        seen[static_cast<size_t>(e)] = 1;
        c.routed += 1;
        ++owned[e / per];
        if (!h) {
          if (m.moe_tp()) {  // TP: each card reads its half of every cold expert
            ++cold[0];
            ++cold[1];
          } else {
            ++cold[e / per];
          }
        }
      }
      c.unique += m.moe_tp() ? cold[0] : cold[0] + cold[1];
      c.crit += std::max(cold[0], cold[1]);
      c.imb += std::abs(owned[0] - owned[1]);
      c.diff.push_back(owned[0] - owned[1]);
    }
    return c;
  };
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  if (ls.blob_mismatches >= 0) check(ls.blob_mismatches == 0, "expert blobs verified against their pack-time hashes (" + std::to_string(ls.blob_mismatches) + " differ)");

  // ---- --pf-graph-gate N: the prompt-call graphs against the launches (pf_graphs.cpp), then exit ----
  if (g_pf_graph_gate != 0) {
    if (chunks.empty()) {
      std::fprintf(stderr, "model: --pf-graph-gate needs --ppl (its tokens)\n");
      return 2;
    }
    std::vector<int32_t> tok;
    for (const auto &c : chunks) tok.insert(tok.end(), c.begin(), c.end());
    std::vector<int> sizes;
    std::stringstream ss(g_pf_graph_sizes);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty()) sizes.push_back(std::stoi(t));
    return rc | pf_graph_gate(m, tok, g_pf_graph_gate, sizes);
  }

  // ---- --pf-route-trace: capture for xenithium-lookahead, then exit ----
  if (!g_pf_route_trace.empty()) {
    if (!moe_tp) {
      std::fprintf(stderr, "model: --pf-route-trace needs expert TP (the many-row MoE's routing)\n");
      return 2;
    }
    size_t stride = 0;
    xenithium::kernels::moe_tp_layout(cfg.hidden, cfg.moe_inter, 2, &stride);
    std::FILE *f = std::fopen(g_pf_route_trace.c_str(), "wb");
    if (!f) throw std::runtime_error("model: cannot write " + g_pf_route_trace);
    const uint32_t magic = 0x31414C58u;  // 'XLA1'
    const int32_t hdr[4] = {L, E, TK, cfg.hidden};
    const uint64_t st64 = stride;
    bool ok = std::fwrite(&magic, 4, 1, f) == 1 && std::fwrite(hdr, 4, 4, f) == 4 && std::fwrite(&st64, 8, 1, f) == 1;
    q38::Model::PfTrace tr;
    tr.rows = m.prefill_chunk();
    tr.points = {"xa", "xm", "moe.ids"};
    int64_t calls = 0;
    double bytes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t si = 0; si < sets.size() && ok; ++si)
      for (size_t ci = 0; ci < sets[si].size() && ok; ++ci) {
        const auto &tok = sets[si][ci];
        const int len = static_cast<int>(tok.size());
        m.begin_sequence();
        for (int r0 = 0; r0 < len && ok; r0 += m.prefill_chunk()) {
          const int n = std::min(m.prefill_chunk(), len - r0);
          tr.names.clear();
          tr.data.clear();
          tr.widths.clear();
          m.trace_prefill(&tr);
          m.prefill(tok.data() + r0, n, {}, r0 + n < len ? tok[static_cast<size_t>(r0 + n)] : -1);
          m.trace_prefill(nullptr);
          if (tr.names.size() != static_cast<size_t>(3 * L)) throw std::runtime_error("model: route trace points per call");
          const int32_t rec[4] = {static_cast<int32_t>(si), static_cast<int32_t>(ci), r0, n};
          ok = std::fwrite(rec, 4, 4, f) == 4;
          for (int l = 0; l < L && ok; ++l)
            for (int p = 0; p < 3 && ok; ++p) {
              const size_t i = static_cast<size_t>(3 * l + p);
              static const char *pts[3] = {".xa", ".xm", ".moe.ids"};
              if (tr.names[i] != "L" + std::to_string(l) + pts[p]) throw std::runtime_error("model: route trace order: " + tr.names[i]);
              const auto &d = tr.data[i];
              ok = std::fwrite(d.data(), 2, d.size(), f) == d.size();
              bytes += static_cast<double>(d.size()) * 2;
            }
          ++calls;
        }
      }
    ok = std::fclose(f) == 0 && ok;
    check(ok, "route trace: " + std::to_string(calls) + " prefill calls -> " + g_pf_route_trace);
    std::printf("route trace: %.1f MB in %.1f s\n", bytes / 1e6, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return rc;
  }
  const int64_t V = cfg.vocab;
  std::vector<float> lg;
  // ---- teacher-forced perplexity (skipped by --pf-time-only 1) ----
  for (size_t si = 0; si < sets.size() && !pf_time_only; ++si) {
  const auto &chunks = sets[si];
  std::vector<std::vector<double>> nll_by_rows;  // [row count][chunk]
  for (int R : row_list) {
    double nll_all = 0;
    int64_t n_all = 0, hits_all = 0, pick_diff = 0, disagree = 0;
    std::vector<double> chunk_ppl, chunk_nll;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<double> step_ms;
    constexpr int kLpK = 32;
    ev::LpSet dump;
    dump.k = kLpK;
    const bool want_dump = !lp_out.empty(), want_kl = si < refs.size();
    ev::KlStats kls({128, 512, 2048, 8192, 32768});
    const bool want_hidden = !hidden_out.empty() && R == row_list.front();
    std::vector<uint8_t> hid_buf;  // the.xmh body after its header
    std::vector<uint16_t> mh;
    auto put32 = [&](int32_t v) {
      const auto *b = reinterpret_cast<const uint8_t *>(&v);
      hid_buf.insert(hid_buf.end(), b, b + 4);
    };
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      if (want_dump) dump.chunks.emplace_back();
      const auto &tok = chunks[ci];
      const int len = static_cast<int>(tok.size());
      m.begin_sequence();
      if (g_kv_lend_idle) {  // The engine's idle refill between sequences (a gate: refilled blobs give the same bits)
        int t = 0;
        for (int n; (n = m.lend_idle(64)) > 0;) t += n;
        if (t) std::printf("  KV lending: %d experts refilled before %s chunk %zu\n", t, files[si].c_str(), ci);
      }
      if (want_hidden) put32(len);
      double nll = 0;
      int n = 0, hits = 0;
      // One run's rows [p0, p0 + rows) after the model ran them (a step, or a prefill slice).
      auto score = [&](int p0, int rows) {
        m.logits(rows, lg);
        const auto picks = m.tokens(rows);
        if (want_hidden) {
          m.multi_hidden(rows, mh);
          const size_t w = mh.size() / static_cast<size_t>(rows);
          for (int j = 0; j < rows; ++j) {
            put32(tok[static_cast<size_t>(p0 + j)]);
            put32(picks[static_cast<size_t>(j)]);
            const auto *b = reinterpret_cast<const uint8_t *>(mh.data() + static_cast<size_t>(j) * w);
            hid_buf.insert(hid_buf.end(), b, b + w * 2);
          }
        }
        for (int j = 0; j < rows; ++j) {
          const float *l = lg.data() + static_cast<size_t>(j) * V;
          int64_t best = 0;
          for (int64_t i = 1; i < V; ++i)
            if (l[i] > l[best]) best = i;  // first maximum = lowest index on ties
          pick_diff += picks[static_cast<size_t>(j)] != best;
          const int t = p0 + j + 1;
          const int64_t target = t < len ? tok[static_cast<size_t>(t)] : best;
          double se = 0;
          const double mx = l[best];
          for (int64_t i = 0; i < V; ++i) se += std::exp(static_cast<double>(l[i]) - mx);
          const double lse = mx + std::log(se);
          nll -= static_cast<double>(l[target]) - lse;
          ++n;
          if (want_dump) dump.chunks.back().push_back(ev::top_logprobs(l, V, kLpK, static_cast<int32_t>(target)));
          if (want_kl) {
            const auto &ref = refs[si].chunks[ci][static_cast<size_t>(p0 + j)];
            auto test_lp = [&](int32_t id, bool &exact) {
              exact = true;
              return static_cast<double>(l[id]) - lse;
            };
            kls.add(p0 + j, ev::kl_coarse(ref, test_lp, static_cast<int32_t>(best)), ref.target_lp, static_cast<double>(l[target]) - lse);
          }
          hits += t < len && best == tok[static_cast<size_t>(t)];
        }
      };
      if (R == 0) {  // prefill: the whole chunk, the head on every 4-row slice
        const auto ts = std::chrono::steady_clock::now();
        m.prefill(tok.data(), len, score);
        step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      } else {
        for (int p0 = 0; p0 < len; p0 += R) {
          const int rows = std::min(R, len - p0);
          const auto ts = std::chrono::steady_clock::now();
          m.step(&tok[static_cast<size_t>(p0)], rows);
          step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
          if (!routes_out.empty()) read_routes(rows);
          score(p0, rows);
        }
      }
      disagree += !m.cards_agree();
      chunk_ppl.push_back(std::exp(nll / n));
      chunk_nll.push_back(nll);
      std::printf("  %s rows %s chunk %zu: %d tokens scored, ppl %.4f, greedy next-token hits %d/%d\n", files[si].c_str(), rlabel(R).c_str(), ci, n, chunk_ppl.back(), hits, n);
      nll_all += nll;
      n_all += n;
      hits_all += hits;
    }
    const double ppl_v = std::exp(nll_all / static_cast<double>(n_all));
    const uint32_t we = m.wire_errors();
    std::printf("ppl %s (rows %s): %.4f over %lld tokens (%zu chunks, %.1f s, step median %.2f ms), greedy hits %lld/%lld; device pick != host argmax "
                "in %lld rows; cards disagree after %lld chunks; wire errors %u\n",
                files[si].c_str(), rlabel(R).c_str(), ppl_v, static_cast<long long>(n_all), chunks.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                pct(step_ms, 0.5), static_cast<long long>(hits_all), static_cast<long long>(n_all), static_cast<long long>(pick_diff),
                static_cast<long long>(disagree), we);
    char line[256];
    std::snprintf(line, sizeof line, "%s rows %s: ppl %.4f finite, device picks == host argmax, every card bit-identical, no mailbox timeout", files[si].c_str(), rlabel(R).c_str(), ppl_v);
    check(std::isfinite(ppl_v) && pick_diff == 0 && disagree == 0 && we == 0, line);
    nll_by_rows.push_back(chunk_nll);
    if (want_hidden) {
      const std::string base = files[si].substr(files[si].find_last_of('/') + 1);
      const std::string path = hidden_out + "/" + base.substr(0, base.find_last_of('.')) + ".xmh";
      std::FILE *f = std::fopen(path.c_str(), "wb");
      if (!f) throw std::runtime_error("model: cannot write " + path);
      const uint32_t magic = 0x31484D58u;  // 'XMH1'
      const int32_t width = cfg.hc * cfg.hidden, nch = static_cast<int32_t>(chunks.size());
      bool ok = std::fwrite(&magic, 4, 1, f) == 1 && std::fwrite(&width, 4, 1, f) == 1 && std::fwrite(&nch, 4, 1, f) == 1 &&
                std::fwrite(hid_buf.data(), 1, hid_buf.size(), f) == hid_buf.size();
      ok = std::fclose(f) == 0 && ok;
      if (!ok) throw std::runtime_error("model: short write " + path);
      std::printf("hidden: %s (%.1f MB)\n", path.c_str(), static_cast<double>(hid_buf.size() + 12) / 1e6);
    }
    if (want_dump) {
      const std::string stem = files[si].substr(0, files[si].find_last_of('.'));
      const std::string path = lp_out + "/" + stem + ".rows" + rlabel(R) + ".xlp";
      std::string err;
      check(ev::write_lp(path, dump, &err), "log-probs (top " + std::to_string(kLpK) + ") -> " + path + (err.empty() ? "" : ": " + err));
    }
    if (want_kl) {
      std::printf("KL(reference || xenithium) %s rows %s, ref top %d + tail bucket (a lower bound on the full-vocab KL); "
                  "ppl columns use each engine's own last-row target\n%s",
                  files[si].c_str(), rlabel(R).c_str(), refs[si].k, kls.table().c_str());
    }
  }
  for (size_t i = 1; i < nll_by_rows.size(); ++i)
    std::printf("%s rows %s vs rows %s: per-chunk nll %s\n", files[si].c_str(), rlabel(row_list[0]).c_str(), rlabel(row_list[i]).c_str(),
                nll_by_rows[i] == nll_by_rows[0] ? "bit-identical" : "DIFFERENT");
  }

  if (m.graphs_recorded())
    std::printf("step graphs: %d recorded so far, %.1f ms recording + finalizing in total (%.1f ms each)\n", m.graphs_recorded(), m.graph_record_ms(),
                m.graph_record_ms() / m.graphs_recorded());
  // ---- prefill (--rows with pf): timing with the head on the last slice only, then the
  // continuation gate: greedy decode after a prefill == greedy decode after 4-row steps over the same
  // prompt (the state a prefill leaves, under graph replay too) ----
  if (std::find(row_list.begin(), row_list.end(), 0) != row_list.end()) {
    for (size_t si = 0; si < sets.size(); ++si) {
      const auto &tok = sets[si][0];
      const int len = static_cast<int>(tok.size());
      std::vector<double> ms;
      for (int rep = 0; rep < 3 && pf_sweep.empty(); ++rep) {  // --pf-sweep: the sweep alone
        m.begin_sequence();
        const auto ts = std::chrono::steady_clock::now();
        m.prefill(tok.data(), len);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      }
      std::sort(ms.begin(), ms.end());
      if (!ms.empty())
        std::printf("prefill %s chunk 0: %d tokens, chunk %d rows: %.1f ms (median of 3, head on the last slice) = %.0f tokens/s\n", files[si].c_str(), len,
                    m.prefill_chunk(), ms[1], len / (ms[1] / 1e3));
      // --pf-sweep: one prefill call of N rows from an empty sequence (what admitting an
      // N-token prompt costs the decoding streams), median of 5, against a 4-row decode step there
      if (!pf_sweep.empty() && si == 0) {
        std::vector<int> ns;
        for (size_t p0 = 0; p0 < pf_sweep.size();) {
          const size_t e = pf_sweep.find(',', p0);
          ns.push_back(std::atoi(pf_sweep.substr(p0, e - p0).c_str()));
          p0 = e == std::string::npos ? pf_sweep.size() : e + 1;
        }
        for (int n : ns) {
          if (n < 1 || n > len) continue;
          std::vector<double> v;
          for (int rep = 0; rep < 5; ++rep) {
            m.begin_sequence();
            const auto ts = std::chrono::steady_clock::now();
            m.prefill(tok.data(), n);
            v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
          }
          std::sort(v.begin(), v.end());
          std::vector<double> st;
          for (int rep = 0; rep < 5; ++rep) {
            const auto ts = std::chrono::steady_clock::now();
            m.step(&tok[static_cast<size_t>(n % 4)], 4);
            st.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
          }
          std::sort(st.begin(), st.end());
          std::printf("pf-sweep: %d rows in one prefill call %.2f ms (median of 5); a 4-row decode step after it %.1f ms\n", n, v[2], st[2]);
        }
      }
      if (pf_time_only) continue;  // timing only: no continuation / handoff gates (they run the prompt in 4-row steps)
      const int gen = 16;
      std::vector<int32_t> a, b;
      for (int path = 0; path < 2; ++path) {
        m.begin_sequence();
        if (path == 0) {
          m.prefill(tok.data(), len);
        } else {
          for (int p0 = 0; p0 < len; p0 += 4) m.step(&tok[static_cast<size_t>(p0)], std::min(4, len - p0));
        }
        const int last = (len - 1) % 4;
        int32_t next = m.tokens(last + 1)[static_cast<size_t>(last)];
        auto &out = path == 0 ? a : b;
        for (int i = 0; i < gen && m.position() < m.capacity(); ++i) {
          out.push_back(next);
          m.step(&next, 1);
          next = m.tokens(1)[0];
        }
      }
      {
        size_t same = 0;
        while (same < a.size() && same < b.size() && a[same] == b[same]) ++same;
        std::printf("  continuation: %zu / %zu leading tokens equal\n    after prefill: ", same, a.size());
        for (int32_t t : a) std::printf("%d ", t);
        std::printf("\n    after steps:   ");
        for (int32_t t : b) std::printf("%d ", t);
        std::printf("\n");
      }
      // Handoff gate that no near-tie can flip: the first half by prefill (or by steps), then the second
      // half teacher-forced in 4-row steps; the second half's ppl must agree (within rounding noise).
      {
        const int half = (len / 2) / 4 * 4;
        double ppl2[2] = {0, 0};
        int scored = 0;
        std::vector<float> lgv;
        for (int path = 0; path < 2 && half >= 4 && len - half >= 8; ++path) {
          m.begin_sequence();
          if (path == 0) {
            m.prefill(tok.data(), half);
          } else {
            for (int p0 = 0; p0 < half; p0 += 4) m.step(&tok[static_cast<size_t>(p0)], 4);
          }
          double nll = 0;
          int n = 0;
          for (int p0 = half; p0 + 1 < len; p0 += 4) {
            const int rows = std::min(4, len - 1 - p0);
            m.step(&tok[static_cast<size_t>(p0)], rows);
            m.logits(rows, lgv);
            for (int j = 0; j < rows; ++j) {
              const float *l = lgv.data() + static_cast<size_t>(j) * V;
              double mx = l[0];
              for (int64_t i = 1; i < V; ++i) mx = std::max<double>(mx, l[i]);
              double se = 0;
              for (int64_t i = 0; i < V; ++i) se += std::exp(static_cast<double>(l[i]) - mx);
              nll -= static_cast<double>(l[tok[static_cast<size_t>(p0 + j + 1)]]) - (mx + std::log(se));
              ++n;
            }
          }
          ppl2[path] = std::exp(nll / std::max(1, n));
          scored = n;
        }
        if (ppl2[0] > 0) {
          const double rel = std::fabs(ppl2[0] - ppl2[1]) / ppl2[1];
          std::printf("  handoff: second-half ppl after a prefill %.4f vs after steps %.4f (%.2f %%, %d tokens)\n", ppl2[0], ppl2[1], rel * 100, scored);
          // Gated on >= 1,000 scored tokens: a rounding-order change alone moves a ~70-token ppl by a few %.
          if (scored >= 1000)
            check(rel < 0.02, files[si] + ": decode after a prefill continues the sequence (second-half ppl within 2 % of the steps path)");
        }
      }
      // Token identity holds only while prefill is bit-identical to steps (stage 1); with the many-row
      // kernels a near-tie can flip a greedy token, so the handoff ppl above is the gate.
      check(!a.empty() && m.wire_errors() == 0, files[si] + ": greedy decode after prefill runs (" + std::to_string(a.size()) + " tokens, step graphs " +
                                                    (graphs ? "on" : "off") + ", no mailbox timeout)");
    }
  }
  // ---- greedy decode timing ----
  if (generate > 0) {
    std::vector<int32_t> prompt = chunks.empty() ? std::vector<int32_t>{9707, 11, 1879, 0} : chunks[0];
    std::vector<int32_t> corpus;
    for (const auto &c : chunks) corpus.insert(corpus.end(), c.begin(), c.end());
    if (corpus.empty()) corpus = prompt;
    size_t cur = 0;
    auto corpus_rows = [&](int32_t *t, int n) {
      for (int r = 0; r < n; ++r) t[r] = corpus[cur++ % corpus.size()];
    };
    m.begin_sequence();
    const auto t_fill = std::chrono::steady_clock::now();
    if (at > static_cast<int64_t>(prompt.size())) {
      int32_t t4[4];
      while (m.position() + 4 <= at - 1) {
        corpus_rows(t4, 4);
        m.step(t4, 4);
      }
      while (m.position() < at - 1) {
        corpus_rows(t4, 1);
        m.step(t4, 1);
      }
      m.step(&prompt.back(), 1);
      std::printf("generate: history filled to position %lld in %.1f s (corpus text, 4-row steps)\n", static_cast<long long>(m.position()),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fill).count());
    } else {
      for (size_t p = 0; p < prompt.size(); ++p) m.step(&prompt[p], 1);
    }
    const int64_t pos_start = m.position();
    int32_t next = m.tokens(1)[0];
    std::vector<int32_t> out;
    std::vector<double> wall, span, ple, submit, wait;
    using P = q38::StepProfile;
    std::vector<std::vector<double>> prof1(P::kParts), prof4(P::kParts);
    std::vector<std::vector<double>> moe1(2 * P::kMoeLaunches), moe4(2 * P::kMoeLaunches);
    std::vector<Cold> cold1, cold4;
    std::vector<double> cmin1, cskew1, cmin4, cskew4;
    std::vector<double> lx1, ly1, lx4, ly4;  // per layer: experts card 0 - card 1, combine ms card 1 - card 0
    auto keep_moe = [&](const P &pr, std::vector<std::vector<double>> &v) {
      for (int c = 0; c < 2; ++c)
        for (int j = 0; j < P::kMoeLaunches; ++j) v[static_cast<size_t>(c * P::kMoeLaunches + j)].push_back(pr.moe_launch[c][j]);
    };
    m.set_profiling(true);
    for (int i = 0; i < generate && m.position() < m.capacity(); ++i) {
      out.push_back(next);
      const auto ts = std::chrono::steady_clock::now();
      m.step(&next, 1);
      wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      span.push_back(m.gpu_span_ms());
      ple.push_back(m.times().ple_read);
      submit.push_back(m.times().submit);
      wait.push_back(m.times().wait);
      const auto pr = m.profile();
      for (int k = 0; k < P::kParts; ++k) prof1[k].push_back(pr.ms[k]);
      keep_moe(pr, moe1);
      cmin1.push_back(pr.combine_min);
      cskew1.push_back(pr.combine_skew);
      cold1.push_back(read_routes(1));
      for (size_t l = 0; l < cold1.back().diff.size() && l < pr.combine_ms[0].size(); ++l) {
        lx1.push_back(cold1.back().diff[l]);
        ly1.push_back(pr.combine_ms[1][l] - pr.combine_ms[0][l]);
      }
      next = m.tokens(1)[0];
    }
    if (!gen_out.empty()) {
      if (at > static_cast<int64_t>(prompt.size())) throw std::runtime_error("model: --gen-out needs no --at history");
      std::vector<int32_t> all = prompt;
      all.insert(all.end(), out.begin(), out.end());
      std::FILE *f = std::fopen(gen_out.c_str(), "wb");
      if (!f) throw std::runtime_error("model: cannot write " + gen_out);
      const int32_t one = 1, len = static_cast<int32_t>(all.size());
      bool ok = std::fwrite(&one, 4, 1, f) == 1 && std::fwrite(&len, 4, 1, f) == 1 && std::fwrite(all.data(), 4, all.size(), f) == all.size();
      ok = std::fclose(f) == 0 && ok;
      if (!ok) throw std::runtime_error("model: short write " + gen_out);
      std::printf("gen-out: %s (%zu prompt + %zu greedy tokens)\n", gen_out.c_str(), prompt.size(), out.size());
    }
    // The verify shape: 4 rows per step (the corpus text continues; all rows accepted).
    std::vector<double> wall4, span4;
    for (int i = 0; i < generate && m.position() + 4 <= m.capacity(); ++i) {
      int32_t t4[4];
      corpus_rows(t4, 4);
      const auto ts = std::chrono::steady_clock::now();
      m.step(t4, 4);
      wall4.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      span4.push_back(m.gpu_span_ms());
      const auto pr = m.profile();
      for (int k = 0; k < P::kParts; ++k) prof4[k].push_back(pr.ms[k]);
      keep_moe(pr, moe4);
      cmin4.push_back(pr.combine_min);
      cskew4.push_back(pr.combine_skew);
      cold4.push_back(read_routes(4));
      for (size_t l = 0; l < cold4.back().diff.size() && l < pr.combine_ms[0].size(); ++l) {
        lx4.push_back(cold4.back().diff[l]);
        ly4.push_back(pr.combine_ms[1][l] - pr.combine_ms[0][l]);
      }
    }
    m.set_profiling(false);
    std::printf("generate: %zu tokens from position %lld, 1 row per step; then %zu steps of 4 rows to position %lld\n", out.size(),
                static_cast<long long>(pos_start), wall4.size(), static_cast<long long>(m.position()));
    // The same shapes again without profiling, as a paired A/B on the same load (same cards, same
    // positions, same text): with --graphs 1 the steps alternate graph replay and eager submission.
    const bool alternate = graphs;
    struct Pass {
      std::vector<double> wall, submit, wait, loop;
    } pr1[2], pr4[2];  // [0] eager, [1] graph
    int recorded = 0;
    double record_ms = 0;
    q38::Model::DecProfile dps[2];  // The replayed steps' module stamps per pass
    for (int pass = 0; pass < 2; ++pass) {
      const int R = pass == 0 ? 1 : 4;
      if (m.dec_profiling()) m.dec_profile_reset();
      for (int i = 0; i < 2 * generate && m.position() + R <= m.capacity(); ++i) {
        const bool g = alternate ? (i & 1) == 0 : graphs;
        m.set_graphs(g);
        int32_t t4[4];
        if (R == 1) t4[0] = next; else corpus_rows(t4, 4);
        const auto ts = std::chrono::steady_clock::now();
        m.step(t4, R);
        const double w = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        if (m.times().record > 0) {  // a first use of a shape: not a steady-state step
          ++recorded;
          record_ms += m.times().record;
        } else {
          Pass &pp = (R == 1 ? pr1 : pr4)[g ? 1 : 0];
          pp.wall.push_back(w);
          pp.submit.push_back(m.times().submit);
          pp.wait.push_back(m.times().wait);
        }
        if (R == 1) next = m.tokens(1)[0];
        if (m.times().record <= 0) (R == 1 ? pr1 : pr4)[g ? 1 : 0].loop.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      }
      if (m.dec_profiling()) dps[pass] = m.dec_profile();
    }
    m.set_graphs(graphs);
    if (m.dec_profiling() && dps[0].steps > 0) {  // Module times under graph replay, both cards
      std::printf("decode stamps (XEN_DEC_PROFILE, replayed steps; GPU clock %.3f GHz; ms per step, mean; stamps include their own launches)\n", dps[0].ghz);
      std::printf("  %-26s %9s %9s %9s %9s\n", "module", "1r card0", "1r card1", "4r card0", "4r card1");
      auto per = [](const q38::Model::DecProfile &d, double v) { return d.steps ? v / d.steps : 0.0; };
      double s[4] = {};
      for (int k = 0; k < P::kParts; ++k) {
        const double v[4] = {per(dps[0], dps[0].ms[0][k]), per(dps[0], dps[0].ms[1][k]), per(dps[1], dps[1].ms[0][k]), per(dps[1], dps[1].ms[1][k])};
        for (int j = 0; j < 4; ++j) s[j] += v[j];
        std::printf("  %-26s %9.3f %9.3f %9.3f %9.3f\n", P::name(k), v[0], v[1], v[2], v[3]);
      }
      std::printf("  %-26s %9.3f %9.3f %9.3f %9.3f\n", "span (first -> last stamp)", per(dps[0], dps[0].span[0]), per(dps[0], dps[0].span[1]), per(dps[1], dps[1].span[0]),
                  per(dps[1], dps[1].span[1]));
      const char *mn[4] = {"  MoE: router", "  MoE: route", "  MoE: experts", "  MoE: combine + all-reduce"};
      for (int k = 0; k < 4; ++k)
        std::printf("  %-26s %9.3f %9.3f %9.3f %9.3f\n", mn[k], per(dps[0], dps[0].moe[0][k]), per(dps[0], dps[0].moe[1][k]), per(dps[1], dps[1].moe[0][k]),
                    per(dps[1], dps[1].moe[1][k]));
      std::printf("  %-26s %9d %9s %9d\n", "steps", dps[0].steps, "", dps[1].steps);
      std::printf("  launches per step (graph nodes, stamps excluded)\n");
      int tot[4] = {};
      for (int k = 0; k < P::kParts; ++k) {
        const int v[4] = {dps[0].launches[0][k], dps[0].launches[1][k], dps[1].launches[0][k], dps[1].launches[1][k]};
        for (int j = 0; j < 4; ++j) tot[j] += v[j];
        std::printf("  %-26s %9d %9d %9d %9d\n", P::name(k), v[0], v[1], v[2], v[3]);
      }
      std::printf("  %-26s %9d %9d %9d %9d\n", "total", tot[0], tot[1], tot[2], tot[3]);
    }
    std::printf("replay pass (no profiling, %s): %d graphs recorded on the way (%.1f ms, those steps excluded), to position %lld\n",
                graphs ? "graph and eager steps alternating" : "eager only",
                recorded, record_ms, static_cast<long long>(m.position()));
    std::printf("  %-26s %9s %9s %9s %9s %6s\n", "ms per step", "1r med", "1r p90", "4r med", "4r p90", "steps");
    for (int g = 1; g >= 0; --g) {
      const Pass &a1 = pr1[g], &a4 = pr4[g];
      if (a1.wall.empty() || a4.wall.empty()) continue;
      const char *tag = g ? "graph" : "eager";
      std::printf("  %-6s %-19s %9.2f %9.2f %9.2f %9.2f %6zu\n", tag, "wall (host)", pct(a1.wall, 0.5), pct(a1.wall, 0.9), pct(a4.wall, 0.5), pct(a4.wall, 0.9), a1.wall.size());
      std::printf("  %-6s %-19s %9.2f %9.2f %9.2f %9.2f\n", tag, "  submit", pct(a1.submit, 0.5), pct(a1.submit, 0.9), pct(a4.submit, 0.5), pct(a4.submit, 0.9));
      std::printf("  %-6s %-19s %9.2f %9.2f %9.2f %9.2f\n", tag, "  wait", pct(a1.wait, 0.5), pct(a1.wait, 0.9), pct(a4.wait, 0.5), pct(a4.wait, 0.9));
      std::printf("  %-6s %-19s %9.2f %9.2f %9.2f %9.2f\n", tag, "loop (+ token read)", pct(a1.loop, 0.5), pct(a1.loop, 0.9), pct(a4.loop, 0.5), pct(a4.loop, 0.9));
    }
    std::printf("  %-26s %9s %9s\n", "ms per step", "median", "p90");
    std::printf("  %-26s %9.2f %9.2f\n", "wall (host)", pct(wall, 0.5), pct(wall, 0.9));
    std::printf("  %-26s %9.2f %9.2f\n", "  PLE table read", pct(ple, 0.5), pct(ple, 0.9));
    std::printf("  %-26s %9.2f %9.2f\n", "  submit", pct(submit, 0.5), pct(submit, 0.9));
    std::printf("  %-26s %9.2f %9.2f\n", "  wait", pct(wait, 0.5), pct(wait, 0.9));
    std::printf("  %-26s %9.2f %9.2f\n", "GPU span card 0", pct(span, 0.5), pct(span, 0.9));
    if (!wall4.empty()) {
      std::printf("  %-26s %9.2f %9.2f\n", "4 rows: wall (host)", pct(wall4, 0.5), pct(wall4, 0.9));
      std::printf("  %-26s %9.2f %9.2f\n", "4 rows: GPU span card 0", pct(span4, 0.5), pct(span4, 0.9));
    }
    std::printf("  card 0 kernel time per module (ms, median; block = first launch start -> last launch end)\n");
    std::printf("  %-26s %9s %9s\n", "module", "1 row", "4 rows");
    double s1 = 0, s4 = 0;
    for (int k = 0; k < P::kParts; ++k) {
      const double a = pct(prof1[k], 0.5), b = prof4[k].empty() ? 0 : pct(prof4[k], 0.5);
      s1 += a;
      s4 += b;
      std::printf("  %-26s %9.3f %9.3f\n", P::name(k), a, b);
    }
    std::printf("  %-26s %9.3f %9.3f\n", "sum of modules", s1, s4);
    std::printf("  %-26s %9.3f %9.3f\n", "gaps (span - sum)", pct(span, 0.5) - s1, span4.empty() ? 0 : pct(span4, 0.5) - s4);
    std::printf("  MoE launches (ms per step, median; each launch start -> end, summed over %d layers)\n", L);
    std::printf("  %-26s %9s %9s %9s %9s\n", "launch", "1r card0", "1r card1", "4r card0", "4r card1");
    for (int j = 0; j < P::kMoeLaunches; ++j) {
      auto med = [&](std::vector<std::vector<double>> &v, int c) {
        auto &x = v[static_cast<size_t>(c * P::kMoeLaunches + j)];
        return x.empty() ? 0.0 : pct(x, 0.5);
      };
      std::printf("  %-26s %9.3f %9.3f %9.3f %9.3f\n", P::moe_name(j), med(moe1, 0), med(moe1, 1), med(moe4, 0), med(moe4, 1));
    }
    std::printf("  %-26s %9.3f %9s %9.3f\n", "combine, faster card", pct(cmin1, 0.5), "", cmin4.empty() ? 0 : pct(cmin4, 0.5));
    std::printf("  %-26s %9.3f %9s %9.3f\n", "combine, |card0 - card1|", pct(cskew1, 0.5), "", cskew4.empty() ? 0 : pct(cskew4, 0.5));
    auto fit = [](const std::vector<double> &x, const std::vector<double> &y, double &a, double &b) {
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      const double n = static_cast<double>(x.size());
      for (size_t i = 0; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
      }
      const double den = n * sxx - sx * sx;
      if (den <= 0) return false;
      b = (n * sxy - sx * sy) / den;
      a = (sy - b * sx) / n;
      return true;
    };
    auto imb_line = [&](const char *what, const std::vector<Cold> &v, const std::vector<double> &skew) {
      if (v.empty()) return;
      std::vector<double> x;
      for (const auto &c : v) x.push_back(c.imb);
      double a = 0, b = 0;
      if (fit(x, skew, a, b))
        std::printf("  %s: EP imbalance sum_l |experts card 0 - card 1| = %.0f per step (median); combine |card0 - card1| ms = %.3f + %.4f x imbalance\n",
                    what, pct(x, 0.5), a, b);
    };
    imb_line("1 row", cold1, cskew1);
    imb_line("4 rows", cold4, cskew4);
    auto layer_line = [&](const char *what, const std::vector<double> &x, const std::vector<double> &y) {
      double a = 0, b = 0;
      if (!fit(x, y, a, b)) return;
      double my = 0;
      for (double v : y) my += v;
      my /= static_cast<double>(y.size());
      double ss = 0, sr = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        ss += (y[i] - my) * (y[i] - my);
        sr += (y[i] - a - b * x[i]) * (y[i] - a - b * x[i]);
      }
      std::printf("  %s, per layer (%zu layer-steps): combine ms card 1 - card 0 = %.4f + %.4f x (experts card 0 - card 1), R^2 %.2f\n", what, x.size(), a,
                  b, ss > 0 ? 1 - sr / ss : 0.0);
    };
    layer_line("1 row", lx1, ly1);
    layer_line("4 rows", lx4, ly4);
    auto cold_line = [&](const char *what, const std::vector<Cold> &v, const std::vector<std::vector<double>> &prof) {
      if (v.empty()) return;
      std::vector<double> crit, uniq, routed, routes;
      for (const auto &c : v) {
        crit.push_back(c.crit);
        uniq.push_back(c.unique);
        routed.push_back(c.routed);
        routes.push_back(c.routes);
      }
      std::printf("  %s: unique experts routed %.0f per step (median); cold: unique %.0f (both cards), critical path %.0f "
                  "(p90 %.0f, max %.0f), cold routes %.0f\n",
                  what, pct(routed, 0.5), pct(uniq, 0.5), pct(crit, 0.5), pct(crit, 0.9), pct(crit, 1.0), pct(routes, 0.5));
      // Least squares: MoE ms per step = a + b * critical-path cold experts.
      const auto &y = prof[P::moe];
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      const double n = static_cast<double>(crit.size());
      for (size_t i = 0; i < crit.size(); ++i) {
        sx += crit[i];
        sy += y[i];
        sxx += crit[i] * crit[i];
        sxy += crit[i] * y[i];
      }
      const double den = n * sxx - sx * sx;
      if (den > 0) {
        const double b = (n * sxy - sx * sy) / den, a = (sy - b * sx) / n;
        std::printf("  %s: MoE ms per step = %.3f + %.4f x critical-path cold experts (least squares over %zu steps)\n", what, a, b, crit.size());
      }
    };
    cold_line("1 row", cold1, prof1);
    cold_line("4 rows", cold4, prof4);
    std::printf("  tokens:");
    for (size_t i = 0; i < out.size() && i < 64; ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    check(m.wire_errors() == 0, "generate: no mailbox timeout");
  }
  {
    double lo = 1e30;
    for (size_t c = 0; c < fab.size(); ++c) {
      const auto vs = xenithium::l0::vram_state(loader.info(fab.rank_index(c)));
      if (vs.ok) lo = std::min(lo, static_cast<double>(vs.free));
    }
    if (lo < 1e30) std::printf("VRAM free at the end (tighter card, sysman): %.3f GiB\n", lo / 1073741824.0);
    if (const auto st = m.lend_stats(); st.on)
      std::printf("KV lending at the end: %lld lent of %lld slots, limit %lld tokens; %llu raises (%.1f ms), %llu evicted, %llu refilled (%.1f ms)\n",
                  static_cast<long long>(st.lent), static_cast<long long>(st.slots), static_cast<long long>(st.limit_tokens),
                  static_cast<unsigned long long>(st.raises), st.raise_ms, static_cast<unsigned long long>(st.evicted),
                  static_cast<unsigned long long>(st.filled), st.fill_ms);
    std::printf("step graphs at the end: %d recorded, %.1f ms recording + finalizing in total\n", m.graphs_recorded(), m.graph_record_ms());
  }
  if (!routes_out.empty()) {
    double tot = 0;
    for (double v : counts) tot += v;
    q38::write_route_counts(routes_out, L, E, counts);
    std::printf("routes: %.0f routes (%.0f row-steps x %d layers x top-%d) -> %s\n", tot, tot / (L * TK), L, TK, routes_out.c_str());
  }
  return rc;
}
