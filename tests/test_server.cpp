// Host-only gate for the serving layer: the host sampler, the batch engine's step
// loop over a CPU fake model (slot isolation, chunked prefill, stops, cancels, queueing), and the
// HTTP server end to end over a real socket with the stub engine and with the batch engine.
// A small byte-level tokenizer.json is written to a temp dir, so the gate needs no checkpoint.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "xenithium/util/image.hpp"
#include "xenithium/core/batch_engine.hpp"
#include "xenithium/core/sampler.hpp"
#include "xenithium/core/stub_engine.hpp"
#include "xenithium/server/server.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/json_constraint.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/http.hpp"
#include "xenithium/util/json.hpp"
#include "xenithium/util/env.hpp"

#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
namespace core = xenithium::core;
namespace text = xenithium::text;
namespace json = xenithium::json;

int g_checks = 0, g_failures = 0;
void expect(bool ok, const std::string &what) {
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
}

// A diagnostic switch is on only when set to a non-empty value other than "0".
void test_env_flags() {
  namespace util = xenithium::util;
  const char *n = "XEN_TEST_ENV_FLAG";
  unsetenv(n);
  expect(!util::env_flag(n), "env flag: unset = off");
  setenv(n, "", 1);
  expect(!util::env_flag(n), "env flag: empty = off (round 46's XEN_PF_PROFILE=${PROF:-})");
  setenv(n, "0", 1);
  expect(!util::env_flag(n), "env flag: \"0\" = off");
  setenv(n, "2", 1);
  expect(util::env_flag(n) && std::strcmp(util::env_value(n), "2") == 0, "env flag: \"2\" = on, value kept");
  unsetenv(n);
}

// ---------------------------------------------------------------- a tiny tokenizer.json

std::string byte_char(int b) {  // the GPT-2 byte-level mapping, as UTF-8
  static std::vector<int> map;
  if (map.empty()) {
    map.assign(256, 0);
    int n = 0;
    for (int c = 0; c < 256; ++c) {
      const bool direct = (c >= 33 && c <= 126) || (c >= 161 && c <= 172) || (c >= 174 && c <= 255);
      map[c] = direct ? c : 256 + n++;
    }
  }
  const int cp = map[b];
  std::string s;
  if (cp < 0x80) s += static_cast<char>(cp);
  else {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

const char *kAdded[] = {"<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>",
                        "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"};  // the image and video placeholders
constexpr int kVocab = 256 + 11;

std::string write_tokenizer_dir() {
  const auto dir = std::filesystem::temp_directory_path() / ("xen-test-server-" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  std::string v = "{";
  for (int b = 0; b < 256; ++b) {
    if (b) v += ",";
    json::write_string(byte_char(b), v);
    v += ":" + std::to_string(b);
  }
  v += "}";
  std::string added = "[";
  for (int i = 0; i < static_cast<int>(std::size(kAdded)); ++i) {
    if (i) added += ",";
    added += "{\"id\":" + std::to_string(256 + i) + ",\"content\":";
    json::write_string(kAdded[i], added);
    added += std::string(",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":") +
             (i < 2 || i >= 6 ? "true" : "false") + "}";
  }
  added += "]";
  const std::string re =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
  std::string j = "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":" + added +
                  ",\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" + re +
                  "\"},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]}"
                  ",\"post_processor\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}"
                  ",\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}"
                  ",\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\","
                  "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":" + v + ",\"merges\":[]}}";
  std::ofstream(dir / "tokenizer.json") << j;
  return dir.string();
}

// ---------------------------------------------------------------- sampler

void test_sampler() {
  const int V = 50;
  std::vector<float> lg(V);
  for (int i = 0; i < V; ++i) lg[i] = static_cast<float>(std::sin(i * 1.7) * 3);
  const int32_t argmax = static_cast<int32_t>(std::max_element(lg.begin(), lg.end()) - lg.begin());
  {
    core::SamplingParams p;
    p.temperature = 0;
    p.logit_bias[7] = 100;
    core::Sampler s(p, {});
    expect(s.sample(lg.data(), V, nullptr, 0) == 7, "sampler: greedy over logit_bias picks the biased token");
  }
  {
    core::SamplingParams p;
    p.temperature = 1;
    p.top_k = 1;
    core::Sampler s(p, {});
    bool all = true;
    for (int i = 0; i < 50; ++i) all &= s.sample(lg.data(), V, nullptr, 0) == argmax;
    expect(all, "sampler: top_k 1 == argmax");
  }
  {
    core::SamplingParams p;
    p.temperature = 1;
    p.top_p = 1e-6f;
    core::Sampler s(p, {});
    bool all = true;
    for (int i = 0; i < 50; ++i) all &= s.sample(lg.data(), V, nullptr, 0) == argmax;
    expect(all, "sampler: tiny top_p keeps only the top token");
  }
  {
    core::SamplingParams p;
    p.temperature = 1;
    p.min_p = 0.999f;
    core::Sampler s(p, {});
    expect(s.sample(lg.data(), V, nullptr, 0) == argmax, "sampler: min_p near 1 keeps only the top token");
  }
  {
    core::SamplingParams p;
    p.temperature = 0.8f;
    p.has_seed = true;
    p.seed = 1234;
    core::Sampler a(p, {}), b(p, {});
    bool same = true;
    std::set<int32_t> seen;
    for (int i = 0; i < 200; ++i) {
      const int32_t x = a.sample(lg.data(), V, nullptr, 0), y = b.sample(lg.data(), V, nullptr, 0);
      a.accept(x), b.accept(y);  // the next output position
      same &= x == y;
      seen.insert(x);
    }
    expect(same && seen.size() > 3, "sampler: a seed reproduces the draws, and T 0.8 draws several tokens");
    core::Sampler c(p, {});
    const int32_t x0 = c.sample(lg.data(), V, nullptr, 0);
    bool fixed = true;
    for (int i = 0; i < 20; ++i) fixed &= c.sample(lg.data(), V, nullptr, 0) == x0;
    expect(fixed, "sampler: a draw is a function of the output position (no accept: the same token)");
  }
  {  // draft-depth invariance: a seeded request's tokens do not depend on the drafts its steps carried (alone or under
     // load), and a draft is accepted with probability p(draft) with the output following p
    core::SamplingParams p;
    p.temperature = 0.7f;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.has_seed = true;
    p.seed = 77;
    const int P = 20000;
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> rows(static_cast<size_t>(8) * V);
    for (float &v : rows) v = nd(g);
    core::Sampler plain(p, {}), spec(p, {}), argd(p, {});
    bool same = true;
    int acc = 0;
    double pd = 0;  // sum of p(draft) over positions
    std::map<int32_t, int> cnt;
    std::uniform_int_distribution<int> pick_row(0, 7), pick_tok(0, V - 1);
    for (int i = 0; i < P; ++i) {
      const float *row = rows.data() + static_cast<size_t>(i % 8) * V;
      const int32_t a = plain.sample(row, V, nullptr, 0);
      const int32_t draft = i % 3 == 0 ? -1 : pick_tok(g);  // no draft, or any token
      const int32_t b = spec.speculate(row, V, draft);
      same &= a == b;
      plain.accept(a), spec.accept(b);
      std::vector<int32_t> ids;
      std::vector<double> w;
      argd.distribution(row, V, ids, w);
      double z = 0;
      for (double v : w) z += v;
      const int32_t d = ids.front();  // an argmax drafter's token
      const int32_t t = argd.speculate(row, V, d);
      argd.accept(t);
      acc += t == d;
      pd += w.front() / z;
      if (i % 8 == 0) ++cnt[t];
    }
    expect(same, "sampler: speculate() == sample() at every position whatever the draft (seeded text independent of depth)");
    expect(std::fabs(acc / double(P) - pd / P) < 0.01, "sampler: argmax drafts accepted at p(draft) (" + std::to_string(acc / double(P)) + " vs " + std::to_string(pd / P) + ")");
    std::vector<int32_t> ids;
    std::vector<double> w;
    argd.distribution(rows.data(), V, ids, w);
    double z = 0, maxdev = 0;
    for (double v : w) z += v;
    for (size_t k = 0; k < ids.size(); ++k) maxdev = std::max(maxdev, std::fabs(cnt[ids[k]] / double(P / 8) - w[k] / z));
    expect(maxdev < 0.03, "sampler: speculative output follows p (row 0, max dev " + std::to_string(maxdev) + ")");
  }
  {  // the empirical distribution follows softmax(x / T) restricted to top_k
    core::SamplingParams p;
    p.temperature = 1.5f;
    p.top_k = 5;
    p.has_seed = true;
    p.seed = 9;
    core::Sampler s(p, {});
    std::vector<int32_t> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    double z = 0;
    for (int k = 0; k < 5; ++k) z += std::exp(lg[idx[k]] / 1.5);
    std::map<int32_t, int> cnt;
    const int N = 40000;
    for (int i = 0; i < N; ++i) {
      const int32_t t = s.sample(lg.data(), V, nullptr, 0);
      ++cnt[t];
      s.accept(t);
    }
    double maxdev = 0;
    bool outside = false;
    for (auto &[t, c] : cnt) {
      const auto it = std::find(idx.begin(), idx.begin() + 5, t);
      if (it == idx.begin() + 5) outside = true;
      else maxdev = std::max(maxdev, std::fabs(c / double(N) - std::exp(lg[t] / 1.5) / z));
    }
    expect(!outside && maxdev < 0.01, "sampler: top_k 5 at T 1.5 follows the renormalised softmax (max dev " + std::to_string(maxdev) + ")");
  }
  {  // penalties
    core::SamplingParams p;
    p.temperature = 0;
    p.frequency_penalty = 100;
    core::Sampler s(p, {});
    s.accept(argmax);
    expect(s.sample(lg.data(), V, nullptr, 0) != argmax, "sampler: frequency penalty moves the pick off a seen token");
    core::SamplingParams q;
    q.temperature = 0;
    q.repetition_penalty = 1000;
    core::Sampler r(q, {argmax});
    expect(r.sample(lg.data(), V, nullptr, 0) != argmax, "sampler: repetition penalty counts prompt tokens");
  }
  {  // raw logprobs: normalised, sorted, unaffected by processing
    core::TokenLogprob lp;
    core::Sampler::raw_logprobs(lg.data(), V, 3, 5, lp);
    double z = 0;
    for (float x : lg) z += std::exp(x);
    expect(std::fabs(lp.logprob - (lg[3] - std::log(z))) < 1e-5, "sampler: raw logprob == log-softmax");
    bool sorted = lp.top.size() == 5 && lp.top[0].first == argmax;
    for (size_t i = 1; i < lp.top.size(); ++i) sorted &= lp.top[i - 1].second >= lp.top[i].second;
    expect(sorted, "sampler: top logprobs sorted, best first");
  }
}

// The sampling processing over the whole row, written plainly: the kept tokens (best first) and
// their unnormalised weights. The fast sampler must keep exactly these with the same weights.
void reference_distribution(const std::vector<float> &raw, const core::SamplingParams &p, const std::vector<int32_t> &prompt,
                            const std::vector<int32_t> &out, const std::vector<uint64_t> *allowed, std::vector<int32_t> &ids,
                            std::vector<double> &w) {
  const int64_t V = static_cast<int64_t>(raw.size());
  std::vector<float> x = raw;
  for (const auto &[t, b] : p.logit_bias) x[t] += b;
  if (p.repetition_penalty != 1.0f) {
    std::set<int32_t> seen(prompt.begin(), prompt.end());
    seen.insert(out.begin(), out.end());
    for (int32_t t : seen) x[t] = x[t] > 0 ? x[t] / p.repetition_penalty : x[t] * p.repetition_penalty;
  }
  std::map<int32_t, int> cnt;
  for (int32_t t : out) ++cnt[t];
  if (p.frequency_penalty != 0 || p.presence_penalty != 0)
    for (const auto &[t, c] : cnt) x[t] -= p.frequency_penalty * static_cast<float>(c) + p.presence_penalty;
  if (allowed)
    for (int64_t i = 0; i < V; ++i)
      if (!(((*allowed)[i >> 6] >> (i & 63)) & 1)) x[i] = -INFINITY;
  const float inv_t = 1.0f / p.temperature;
  for (auto &v : x) v *= inv_t;
  std::vector<int32_t> idx(V);
  for (int64_t i = 0; i < V; ++i) idx[i] = static_cast<int32_t>(i);
  std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
  const float m = x[idx[0]];
  const float floor_x = p.min_p > 0 ? m + std::log(p.min_p) : -INFINITY;
  ids.clear();
  for (int32_t i : idx)
    if (x[i] >= floor_x && x[i] > -INFINITY) ids.push_back(i);
  if (p.top_k > 0 && ids.size() > static_cast<size_t>(p.top_k)) ids.resize(static_cast<size_t>(p.top_k));
  w.clear();
  for (int32_t i : ids) w.push_back(std::exp(static_cast<double>(x[i]) - m));
  if (p.top_p < 1.0f) {
    double z = 0;
    for (double v : w) z += v;
    double before = 0;
    size_t keep = 0;
    while (keep < ids.size() && (keep == 0 || before < p.top_p * z)) before += w[keep++];
    ids.resize(keep);
    w.resize(keep);
  }
}

void test_sampler_fast() {
  std::mt19937_64 rng(5);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  int cases = 0, bad = 0;
  for (int c = 0; c < 400; ++c) {
    const int V = 2000 + static_cast<int>(rng() % 3000);
    std::vector<float> lg(static_cast<size_t>(V));
    for (auto &v : lg) v = std::round(nd(rng) * 4) / 4;  // quarter steps: many ties
    core::SamplingParams p;
    p.temperature = 0.3f + static_cast<float>(rng() % 100) / 50.0f;
    const int ks[] = {1, 2, 5, 20, 64, 256};
    p.top_k = ks[rng() % 6];
    if (rng() % 3 == 0) p.top_p = 0.5f + static_cast<float>(rng() % 50) / 100.0f;
    if (rng() % 3 == 0) p.min_p = static_cast<float>(rng() % 30) / 100.0f;
    std::vector<int32_t> prompt, out;
    if (rng() % 2) {
      p.repetition_penalty = rng() % 2 ? 1.3f : 0.7f;  // also < 1: raises seen tokens
      for (int i = 0; i < 40; ++i) prompt.push_back(static_cast<int32_t>(rng() % V));
    }
    if (rng() % 2) {
      p.frequency_penalty = rng() % 2 ? 0.8f : -1.5f;
      p.presence_penalty = rng() % 2 ? 0.5f : -2.0f;
    }
    if (rng() % 2)
      for (int i = 0; i < 6; ++i) p.logit_bias[static_cast<int32_t>(rng() % V)] = static_cast<float>(static_cast<int>(rng() % 41) - 20);
    std::vector<uint64_t> mask;
    if (rng() % 3 == 0) {
      mask.assign(static_cast<size_t>((V + 63) / 64), 0);
      for (int i = 0; i < V; ++i)
        if (rng() % 4 == 0) mask[static_cast<size_t>(i >> 6)] |= 1ull << (i & 63);
    }
    core::Sampler s(p, prompt);
    for (int i = 0; i < 30; ++i) out.push_back(static_cast<int32_t>(rng() % V));
    for (int32_t t : out) s.accept(t);
    std::vector<int32_t> a, b;
    std::vector<double> wa, wb;
    s.distribution(lg.data(), V, a, wa, mask.empty() ? nullptr : mask.data());
    reference_distribution(lg, p, prompt, out, mask.empty() ? nullptr : &mask, b, wb);
    bool ok = a == b && wa.size() == wb.size();
    for (size_t i = 0; ok && i < wa.size(); ++i) ok = std::fabs(wa[i] - wb[i]) <= 1e-12 * std::max(1.0, wb[i]);
    ++cases;
    bad += !ok;
  }
  expect(bad == 0, "sampler: candidate path == the full-row reference (" + std::to_string(cases) + " random cases with ties, penalties both ways, "
                   "bias, min_p, top_p, masks; " + std::to_string(bad) + " differ)");
  // (qwen38-27b): top_ids skips blocks of 64 below the threshold: == a sort on sizes around the block, ties, -inf runs
  {
    int bad3 = 0, n3 = 0;
    for (int c = 0; c < 300; ++c) {
      const int V = 1 + static_cast<int>(rng() % 700);
      std::vector<float> lg(static_cast<size_t>(V));
      for (auto &v : lg) v = rng() % 5 == 0 ? -INFINITY : std::round(nd(rng) * 2) / 2;
      if (rng() % 4 == 0)  // ascending values: every block holds a new best
        for (int i = 0; i < V; ++i) lg[static_cast<size_t>(i)] = static_cast<float>(i % 97);
      const size_t n = 1 + rng() % 40;
      std::vector<uint8_t> skip(static_cast<size_t>(V));
      for (auto &s : skip) s = rng() % 6 == 0;
      const bool use_skip = rng() % 2;
      std::vector<int32_t> got, ref;
      core::Sampler::top_ids(lg.data(), V, n, use_skip ? skip.data() : nullptr, got);
      for (int i = 0; i < V; ++i)
        if (!use_skip || !skip[static_cast<size_t>(i)]) ref.push_back(i);
      std::stable_sort(ref.begin(), ref.end(), [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
      if (ref.size() > n) ref.resize(n);
      bad3 += got != ref;
      ++n3;
    }
    expect(bad3 == 0, "sampler: top_ids == a sorted reference (" + std::to_string(n3) + " cases, sizes 1-700, ties, -inf, skips; " + std::to_string(bad3) + " differ)");
  }
  // (qwen38-27b): verify rows prepared at once (prepare_row with the drafts before each row taken as accepted) then
  // drawn in order == speculate row by row with accept between, on the same seed: every setting path (candidates, full
  // row, greedy with penalties), penalties both ways, repeated drafts
  {
    int bad4 = 0, n4 = 0, deep = 0;
    for (int c = 0; c < 1000; ++c) {
      const int V = 500 + static_cast<int>(rng() % 1500);
      const int sr = 2 + static_cast<int>(rng() % 7);
      std::vector<float> lg(static_cast<size_t>(V) * sr);
      for (auto &v : lg) v = nd(rng);
      std::vector<int32_t> t(static_cast<size_t>(sr));  // t[0] the fed token, t[1..] drafts
      for (int j = 0; j < sr; ++j) {
        t[static_cast<size_t>(j)] = static_cast<int32_t>(rng() % (j > 0 && rng() % 3 == 0 ? 8 : V));  // small ids repeat
        if (j > 0) lg[static_cast<size_t>(j - 1) * V + t[static_cast<size_t>(j)]] += 12.0f;  // row j - 1 favours draft j
      }
      core::SamplingParams p;
      p.has_seed = true;
      p.seed = rng();
      const int mode = static_cast<int>(rng() % 3);
      p.temperature = mode == 2 ? 0.0f : 0.4f + static_cast<float>(rng() % 100) / 100.0f;
      p.top_k = mode == 0 ? static_cast<int32_t>(1 + rng() % 64) : 0;
      if (rng() % 2) p.top_p = 0.6f + static_cast<float>(rng() % 40) / 100.0f;
      if (rng() % 2) p.repetition_penalty = rng() % 2 ? 1.3f : 0.8f;
      if (rng() % 2) p.presence_penalty = rng() % 2 ? 1.5f : -1.0f;
      if (rng() % 2) p.frequency_penalty = rng() % 2 ? 0.7f : -0.4f;
      if (mode == 2 && p.repetition_penalty == 1.0f && p.presence_penalty == 0.0f) p.frequency_penalty = 0.5f;  // not plain greedy
      std::vector<int32_t> prompt;
      for (int i = 0; i < 30; ++i) prompt.push_back(static_cast<int32_t>(rng() % 16));
      core::Sampler a(p, prompt), b(p, prompt);
      for (int i = 0; i < 10; ++i) {
        const auto tk = static_cast<int32_t>(rng() % 16);
        a.accept(tk), b.accept(tk);
      }
      std::vector<int32_t> ta, tb;
      for (int j = 0; j < sr; ++j) {
        const int32_t draft = j + 1 < sr ? t[static_cast<size_t>(j + 1)] : -1;
        const int32_t tok = a.speculate(lg.data() + static_cast<size_t>(j) * V, V, draft);
        a.accept(tok);
        ta.push_back(tok);
        if (tok != draft) break;
      }
      std::vector<core::Sampler::Row> rows(static_cast<size_t>(sr));
      for (int j = sr - 1; j >= 0; --j) b.prepare_row(lg.data() + static_cast<size_t>(j) * V, V, t.data() + 1, j, rows[static_cast<size_t>(j)]);
      for (int j = 0; j < sr; ++j) {
        const int32_t draft = j + 1 < sr ? t[static_cast<size_t>(j + 1)] : -1;
        const int32_t tok = b.speculate_row(rows[static_cast<size_t>(j)], lg.data() + static_cast<size_t>(j) * V, V, draft);
        b.accept(tok);
        tb.push_back(tok);
        if (tok != draft) break;
      }
      bad4 += ta != tb;
      deep += ta.size() >= 3;
      ++n4;
    }
    expect(bad4 == 0 && deep > 200, "sampler: verify rows prepared at once == row by row (" + std::to_string(n4) + " chains, " + std::to_string(deep) +
                                       " of 3+ rows; " + std::to_string(bad4) + " differ)");
  }
  // the full path (top_k off) keeps the same set as the reference
  {
    int bad2 = 0;
    for (int c = 0; c < 50; ++c) {
      const int V = 3000;
      std::vector<float> lg(V);
      for (auto &v : lg) v = nd(rng);
      core::SamplingParams p;
      p.temperature = 0.8f;
      p.top_k = 0;
      p.top_p = 0.9f;
      core::Sampler s(p, {});
      std::vector<int32_t> a, b;
      std::vector<double> wa, wb;
      s.distribution(lg.data(), V, a, wa);
      reference_distribution(lg, p, {}, {}, nullptr, b, wb);
      bad2 += a != b;
    }
    expect(bad2 == 0, "sampler: full path (top_k off, top_p 0.9) == the reference");
  }
  // speed on a real vocabulary: the checkpoint defaults (T 1, top_k 20, top_p 0.95)
  {
    const int V = 248320;
    std::vector<float> lg(V);
    for (auto &v : lg) v = nd(rng);
    core::SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.presence_penalty = 0.5f;
    core::Sampler s(p, {});
    for (int i = 0; i < 200; ++i) s.accept(static_cast<int32_t>(rng() % V));
    const auto t0 = std::chrono::steady_clock::now();
    const int N = 200;
    for (int i = 0; i < N; ++i) s.sample(lg.data(), V, nullptr, 0);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
    core::TokenLogprob lp;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) core::Sampler::raw_logprobs(lg.data(), V, 5, 5, lp);
    const double us2 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / N;
    std::printf("sampler speed, vocab 248,320: sample %.0f us per row (T 1, top_k 20, top_p 0.95, 200 penalised tokens); raw_logprobs top 5 %.0f us\n", us, us2);
    expect(us < 1000, "sampler: one 248K row samples in under 1 ms (" + std::to_string(static_cast<int>(us)) + " us)");
  }
}

// ---------------------------------------------------------------- a fake model for BatchEngine

// Every slot holds the tokens it was fed; the pick of a row is a hash of the slot's whole history,
// so any mix-up of slots, positions or fed tokens changes the output. Logits put 8 on the pick.
class FakeModel final : public core::StepModel {
 public:
  FakeModel(int slots, int64_t cap, int chunk, int vocab) : hist_(static_cast<size_t>(slots)), begun_(static_cast<size_t>(slots)), cap_(cap), chunk_(chunk), vocab_(vocab) {}
  int slots() const override { return static_cast<int>(hist_.size()); }
  int64_t capacity() const override { return cap_; }
  int64_t vocab() const override { return vocab_; }
  int max_rows() const override { return rows_max; }
  int rows_max = 4;  // 8 for a DFlash-like block (depth 7)
  // StepModel:step_rows_at: steps of fewer than rows_small_below sequences carry at most rows_small
  // rows (0 = rows_max for every count); a verify over its budget is an error
  int rows_small = 0, rows_small_below = 0;
  int step_rows_at(int n) const override { return rows_small > 0 && n < rows_small_below ? rows_small : rows_max; }
  int max_rows_small_steps = 0, max_rows_large_steps = 0;  // the most rows of a verify below / from rows_small_below sequences
  int prefill_chunk() const override { return chunk_; }
  void begin(int s) override {
    hist_.at(static_cast<size_t>(s)).clear();
    begun_[static_cast<size_t>(s)] = true;
    tier_.resize(hist_.size(), -1);
    tier_[static_cast<size_t>(s)] = -1;  // the last tier until set_rope_tier
    kv_release(s);
    rows_of(s).clear();
    drows_of(s).clear();
    chk_.resize(hist_.size(), -1);
    chk_[static_cast<size_t>(s)] = -1;
    clean(s);
  }
  // The prefix cache. snap_cap > 0: snapshots = (slot, the slot's history); the slot's "KV" = rows_[s][p], the
  // token position p's KV row was written from, and drows_[s][p], the token after it a drafter wrote its row p from (-1:
  // not yet). A restore checks the slot's rows below the snapshot are the snapshot's tokens (no overwrite since); a fork
  // copies the rows; the first prefill call after either (or a live resume) checks, with a drafter, that the row before it
  // was written from the token the call starts with.
  int snap_cap = 0;
  bool live_ok = true;
  int restores = 0, forks = 0, resumes = 0, saves = 0;
  struct FSnap {
    int slot = -1;
    std::vector<int32_t> hist;
    bool used = false;
    int tier = -1;  // the slot's YaRN tier when saved
  };
  // YaRN tiers. tier_limits non-empty = rope_tiers of them (ascending, the last = capacity);
  // a write at or past a slot's tier limit, a tier set on a used sequence or a snapshot of another tier = an error.
  std::vector<int64_t> tier_limits;
  std::vector<int> tier_;
  int tier_misses = 0, tier_sets = 0;
  int rope_tiers() const override { return tier_limits.empty() ? 1 : static_cast<int>(tier_limits.size()); }
  int64_t rope_tier_limit(int t) const override { return tier_limits.empty() ? cap_ : tier_limits.at(static_cast<size_t>(t)); }
  int &tier_of(int s) {
    tier_.resize(hist_.size(), -1);
    int &t = tier_[static_cast<size_t>(s)];
    if (t < 0) t = rope_tiers() - 1;
    return t;
  }
  void set_rope_tier(int s, int t) override {
    if (tier_limits.empty() || t < 0 || t >= rope_tiers() || !begun_.at(static_cast<size_t>(s)) || !hist_.at(static_cast<size_t>(s)).empty()) ++errors, ++tier_misses;
    tier_of(s) = t;
    ++tier_sets;
  }
  void tier_check(int s, size_t positions) {
    if (!tier_limits.empty() && static_cast<int64_t>(positions) > tier_limits.at(static_cast<size_t>(tier_of(s)))) ++errors, ++tier_misses;
  }
  std::vector<FSnap> fsnaps;
  int align = 1;  // snap_align: a snapshot elsewhere is a model misuse
  int snap_capacity() const override { return snap_cap; }
  int snap_align() const override { return align; }
  int snap_save(int s) override {
    if (hist_.at(static_cast<size_t>(s)).size() % static_cast<size_t>(align) != 0) ++snap_errors, ++errors;
    int used = 0, id = -1;
    for (size_t i = 0; i < fsnaps.size(); ++i) {
      used += fsnaps[i].used;
      if (!fsnaps[i].used && id < 0) id = static_cast<int>(i);
    }
    if (used >= snap_cap) return -1;
    if (id < 0) fsnaps.emplace_back(), id = static_cast<int>(fsnaps.size()) - 1;
    fsnaps[static_cast<size_t>(id)] = {s, hist_.at(static_cast<size_t>(s)), true, tier_of(s)};
    ++saves;
    return id;
  }
  void snap_free(int id) override {
    if (id < 0 || id >= static_cast<int>(fsnaps.size()) || !fsnaps[static_cast<size_t>(id)].used) ++errors;
    else fsnaps[static_cast<size_t>(id)].used = false;
  }
  bool rows_match(int s, const std::vector<int32_t> &h) {
    const auto &r = rows_of(s);
    if (r.size() < h.size()) return false;
    for (size_t p = 0; p < h.size(); ++p)
      if (r[p] != h[p]) return false;
    held_pages_.resize(hist_.size(), 0);
    return pool_pages <= 0 || static_cast<size_t>(held_pages_[static_cast<size_t>(s)]) * kPage >= h.size();
  }
  void snap_restore(int s, int id) override {
    const FSnap &f = fsnaps.at(static_cast<size_t>(id));
    if (!f.used || f.slot != s || !rows_match(s, f.hist) || (pend_ && pend_slot_ == s)) ++errors, ++snap_errors;
    if (!tier_limits.empty() && tier_of(s) != f.tier) ++errors, ++tier_misses;
    hist_.at(static_cast<size_t>(s)) = f.hist;
    begun_[static_cast<size_t>(s)] = true;
    chk_.resize(hist_.size(), -1);
    chk_[static_cast<size_t>(s)] = static_cast<int64_t>(f.hist.size());
    clean(s);
    ++restores;
  }
  void snap_fork(int dst, int src, int id) override {
    const FSnap &f = fsnaps.at(static_cast<size_t>(id));
    held_pages_.resize(hist_.size(), 0);
    const bool dst_pages = pool_pages <= 0 || static_cast<size_t>(held_pages_[static_cast<size_t>(dst)]) * kPage >= f.hist.size();
    if (!f.used || f.slot != src || dst == src || !rows_match(src, f.hist) || !dst_pages) ++errors, ++snap_errors;
    if (!tier_limits.empty() && tier_of(dst) != f.tier) ++errors, ++tier_misses;  // the engine set dst's tier at begin
    rows_of(dst).assign(rows_of(src).begin(), rows_of(src).begin() + static_cast<std::ptrdiff_t>(std::min(rows_of(src).size(), f.hist.size())));
    drows_of(dst).assign(drows_of(src).begin(), drows_of(src).begin() + static_cast<std::ptrdiff_t>(std::min(drows_of(src).size(), f.hist.size())));
    hist_.at(static_cast<size_t>(dst)) = f.hist;
    begun_[static_cast<size_t>(dst)] = true;
    clean(dst);
    chk_.resize(hist_.size(), -1);
    chk_[static_cast<size_t>(dst)] = static_cast<int64_t>(f.hist.size());
    ++forks;
  }
  bool snap_live(int s) const override { return live_ok && !dirty(s); }
  void resume(int s) override {
    if (!live_ok || dirty(s)) ++errors, ++snap_errors;
    begun_[static_cast<size_t>(s)] = true;
    chk_.resize(hist_.size(), -1);
    chk_[static_cast<size_t>(s)] = static_cast<int64_t>(hist_.at(static_cast<size_t>(s)).size());
    ++resumes;
  }
  void kv_trim(int s, int64_t tokens) override {
    if (pool_pages <= 0) return;
    held_pages_.resize(hist_.size(), 0);
    int &h = held_pages_.at(static_cast<size_t>(s));
    h = std::min(h, static_cast<int>((tokens + kPage - 1) / kPage));
  }
  int64_t kv_held_tokens(int s) const override {
    return pool_pages <= 0 || static_cast<size_t>(s) >= held_pages_.size() ? 0 : static_cast<int64_t>(held_pages_[static_cast<size_t>(s)]) * kPage;
  }
  int kv_page_tokens() const override { return kPage; }
  int snap_errors = 0, strict_errors = 0;
  std::vector<std::vector<int32_t>> rows_, drows_;
  std::vector<int64_t> chk_;
  std::vector<int32_t> &rows_of(int s) {
    rows_.resize(hist_.size());
    return rows_[static_cast<size_t>(s)];
  }
  std::vector<int32_t> &drows_of(int s) {
    drows_.resize(hist_.size());
    return drows_[static_cast<size_t>(s)];
  }
  void write_row(int s, size_t p, int32_t tok) {
    auto &r = rows_of(s);
    if (r.size() <= p) r.resize(p + 1, -1);
    r[p] = tok;
  }
  void write_drow(int s, size_t p, int32_t next) {
    auto &r = drows_of(s);
    if (r.size() <= p) r.resize(p + 1, -1);
    r[p] = next;
  }
  // the first call after a resume: with a drafter, its row before the call's position came from the call's first token
  void check_resume(int s, int32_t first) {
    chk_.resize(hist_.size(), -1);
    int64_t &c = chk_[static_cast<size_t>(s)];
    if (c < 0) return;
    if (depth > 0 && c > 0) {
      const auto &d = drows_of(s);
      if (static_cast<size_t>(c - 1) >= d.size() || d[static_cast<size_t>(c - 1)] != first) ++errors, ++strict_errors;
    }
    c = -1;
  }
  // A shared KV pool: pool_pages > 0 = pages of kPage tokens shared by the slots; every write past a slot's
  // reserved pages counts as an error (prefill, step, verify; the drafts write up to `depth` more positions).
  static constexpr int kPage = 8;
  int pool_pages = 0;
  int64_t kv_pool_tokens() const override { return static_cast<int64_t>(pool_pages) * kPage; }
  int64_t kv_free_tokens() const override {
    int held = 0;
    for (int h : held_pages_) held += h;
    return static_cast<int64_t>(pool_pages - held) * kPage;
  }
  bool kv_reserve(int s, int64_t tokens) override {
    if (pool_pages <= 0) return true;
    held_pages_.resize(hist_.size(), 0);
    const int want = static_cast<int>((tokens + kPage - 1) / kPage);
    int &h = held_pages_.at(static_cast<size_t>(s));
    if (tokens > cap_) return false;
    if (want <= h) return true;
    if ((want - h) * kPage > kv_free_tokens()) return false;
    h = want;
    max_held = std::max(max_held, pool_pages * kPage - static_cast<int>(kv_free_tokens()));
    return true;
  }
  void kv_release(int s) override {
    if (pool_pages <= 0) return;
    held_pages_.resize(hist_.size(), 0);
    held_pages_.at(static_cast<size_t>(s)) = 0;
  }
  int max_held = 0, kv_misses = 0;
  // KV lending (qwen38-like): lend_keep > 0 = a slot holding more than lend_keep tokens keeps lent experts
  // out; a reservation lends experts back when the pool's held tokens would pass lend_limit; idle_work counts calls.
  int64_t lend_keep = 0, lend_limit = 0;
  int idle_calls = 0;
  bool kv_blocks_lend(int s) const override { return lend_keep > 0 && kv_held_tokens(s) > lend_keep; }
  bool kv_reserve_lends(int s, int64_t tokens) const override {
    if (lend_keep <= 0 || pool_pages <= 0) return false;
    const int want = static_cast<int>((tokens + kPage - 1) / kPage), h = static_cast<size_t>(s) < held_pages_.size() ? held_pages_[static_cast<size_t>(s)] : 0;
    return want > h && pool_pages * kPage - kv_free_tokens() + (want - h) * kPage > lend_limit;
  }
  int idle_work(int budget) override {
    (void)budget;
    ++idle_calls;
    return 0;
  }
  void kv_check(int s, size_t positions) {
    if (pool_pages <= 0) return;
    held_pages_.resize(hist_.size(), 0);
    if (positions > static_cast<size_t>(held_pages_.at(static_cast<size_t>(s))) * kPage) ++errors, ++kv_misses;
  }
  std::vector<int> held_pages_;
  void prefill(int s, const int32_t *t, int n, int32_t next) override {
    if (!begun_.at(static_cast<size_t>(s)) || n < 1 || n > chunk_ || (pend_ && pend_slot_ != s)) ++errors;
    auto &h = hist_[static_cast<size_t>(s)];
    check_resume(s, t[0]);
    for (int j = 0; j < n; ++j) {
      write_row(s, h.size() + static_cast<size_t>(j), t[j]);
      write_drow(s, h.size() + static_cast<size_t>(j), j + 1 < n ? t[j + 1] : next);
    }
    h.insert(h.end(), t, t + n);
    if (static_cast<int64_t>(h.size()) > cap_) ++errors;
    kv_check(s, h.size());
    tier_check(s, h.size());
    pf_pick_ = pick_of(h);
    ++prefills;
    if (!many_quiet_ && (pf_us_fixed > 0 || pf_us_row > 0)) std::this_thread::sleep_for(std::chrono::microseconds(pf_us_fixed + pf_us_row * n));
    if (const int once = pf_us_once.exchange(0); once > 0) std::this_thread::sleep_for(std::chrono::microseconds(once));
    pf_log.push_back({s, n, steps});
  }
  bool prompt_logprobs() const override { return true; }
  void prefill_rows(int s, const int32_t *t, int n, int32_t next, const RowsFn &on_rows) override {
    // the logits of every row: row j = the pick after the history through t[j]
    auto &h = hist_.at(static_cast<size_t>(s));
    std::vector<float> lg;
    for (int r0 = 0; r0 < n; r0 += 4) {
      const int rows = std::min(4, n - r0);
      lg.assign(static_cast<size_t>(rows) * static_cast<size_t>(vocab_), 0.0f);
      for (int j = 0; j < rows; ++j) {
        h.push_back(t[r0 + j]);
        lg[static_cast<size_t>(j) * static_cast<size_t>(vocab_) + static_cast<size_t>(pick_of(h))] = 8.0f;
      }
      h.resize(h.size() - static_cast<size_t>(rows));
      h.insert(h.end(), t + r0, t + r0 + rows);
      on_rows(r0, rows, lg.data());
    }
    h.resize(h.size() - static_cast<size_t>(n));
    prefill(s, t, n, next);
  }
  // Split prefill calls: split_layers > 0 enables them; the call commits (as prefill) when
  // its last layer ran. A step on the pending slot or another prefill while one is pending is an error.
  int split_layers = 0, split_begins = 0, split_parts = 0, split_aborts = 0, steps_between_parts = 0;
  int prefill_layers() const override { return split_layers; }
  void prefill_begin(int s, const int32_t *t, int n, int32_t next) override {
    if (pend_ || split_layers <= 0) ++errors;
    pend_ = true;
    pend_slot_ = s;
    pend_tok_.assign(t, t + n);
    pend_next_ = next;
    pend_done_ = 0;
    pend_steps_ = steps;
    ++split_begins;
  }
  bool prefill_advance(int layers) override {
    if (!pend_ || layers < 1) ++errors;
    ++split_parts;
    if (split_parts > 1) steps_between_parts += steps - pend_steps_;
    if (!many_.empty() && pend_done_ > 0 && layers >= split_layers) ++merge_rushed;  // run to its end mid-call
    pend_steps_ = steps;
    pend_done_ += layers;
    if (part_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(part_us));
    if (pend_done_ < split_layers) return false;
    pend_ = false;
    if (!many_.empty()) {  // every member commits as its own call; one head holds all their rows
      int total = 0;
      many_quiet_ = true;
      many_pick_.clear();
      for (const auto &mb : many_) {
        prefill(mb.slot, mb.tok.data(), static_cast<int>(mb.tok.size()), mb.next);
        many_pick_[mb.slot] = pf_pick_;
        total = (total + merge_pad - 1) / merge_pad * merge_pad + static_cast<int>(mb.tok.size());
      }
      many_quiet_ = false;
      many_.clear();
      if (pf_us_fixed > 0 || pf_us_row > 0) std::this_thread::sleep_for(std::chrono::microseconds(pf_us_fixed + pf_us_row * total));
      return true;
    }
    prefill(pend_slot_, pend_tok_.data(), static_cast<int>(pend_tok_.size()), pend_next_);
    return true;
  }
  void prefill_abort() override {
    if (!pend_) ++errors;
    pend_ = false;
    ++split_aborts;
    dirty_.resize(hist_.size(), false);
    dirty_[static_cast<size_t>(pend_slot_)] = true;  // like the real model, the slot's live state is undefined now
    for (const auto &mb : many_) dirty_[static_cast<size_t>(mb.slot)] = true;
    many_.clear();
  }
  // A call for several slots (merge_max > 1, with split_layers > 0). As the real model: the call
  // has one head - prefill_result(slot) selects a member's row, and the first first_drafts or step after the call
  // reuses the head, so a member read after that counts as an error (merge_stale).
  int merge_max = 1, merge_rows = 1023, merge_pad = 16, merge_min = 33, merge_seg_max = 511;
  int merge_calls = 0, merge_members = 0, merge_stale = 0, merge_rushed = 0, part_us = 0;
  struct Many {
    int slot = 0;
    std::vector<int32_t> tok;
    int32_t next = -1;
  };
  std::vector<Many> many_;
  std::map<int, int32_t> many_pick_;
  bool many_quiet_ = false;
  std::vector<std::vector<int>> merge_log;  // the slots of each merged call
  int prefill_merge_max() const override { return merge_max; }
  int prefill_merge_rows() const override { return merge_max > 1 ? merge_rows : 0; }
  int prefill_merge_pad() const override { return merge_pad; }
  bool prefill_mergeable(int s, const int32_t *t, int n) const override {
    return merge_max > 1 && t && s >= 0 && s < static_cast<int>(hist_.size()) && n >= merge_min && n <= merge_seg_max;
  }
  void prefill_begin_many(const std::vector<PrefillSeg> &segs) override {
    if (pend_ || split_layers <= 0 || segs.size() < 2 || static_cast<int>(segs.size()) > merge_max) ++errors;
    int total = 0;
    std::set<int> u;
    merge_log.emplace_back();
    for (const auto &g : segs) {
      if (!prefill_mergeable(g.slot, g.tokens, g.n) || !u.insert(g.slot).second) ++errors;
      total = (total + merge_pad - 1) / merge_pad * merge_pad + g.n;
      many_.push_back(Many{g.slot, std::vector<int32_t>(g.tokens, g.tokens + g.n), g.next});
      merge_log.back().push_back(g.slot);
    }
    if (total > merge_rows) ++errors;
    pend_ = true;
    pend_slot_ = segs.empty() ? -1 : segs[0].slot;
    pend_done_ = 0;
    pend_steps_ = steps;
    ++merge_calls;
    merge_members += static_cast<int>(segs.size());
  }
  void prefill_result(int s) override {
    const auto it = many_pick_.find(s);
    if (it == many_pick_.end()) ++errors, ++merge_stale;
    else pf_pick_ = it->second;
  }
  std::vector<bool> dirty_;
  bool dirty(int s) const { return static_cast<size_t>(s) < dirty_.size() && dirty_[static_cast<size_t>(s)]; }
  void clean(int s) {
    dirty_.resize(hist_.size(), false);
    dirty_[static_cast<size_t>(s)] = false;
  }
  bool pend_ = false;
  int pend_slot_ = -1, pend_done_ = 0, pend_steps_ = 0;
  int32_t pend_next_ = -1;
  std::vector<int32_t> pend_tok_;
  int32_t prefill_pick() override { return pf_pick_; }
  void prefill_logits(std::vector<float> &row) override {
    ++logits_calls;
    row_of(pf_pick_, row);
  }
  // Logprobs on the model's side, from the same logits rows as the host path: the rows
  // the "head" holds (the last step / verify, or the prefill_each group in its callback).
  bool dev_lp = false;
  int logits_calls = 0, lp_calls = 0, each_calls = 0;
  bool device_logprobs() const override { return dev_lp; }
  void row_logprobs(int rows, const int32_t *t, int top, core::TokenLogprob *out) override {
    ++lp_calls;
    if (rows < 1 || rows > 4 || rows > static_cast<int>(held_.size())) ++errors;
    std::vector<float> row;
    for (int r = 0; r < rows && r < static_cast<int>(held_.size()); ++r) {
      if (t[r] < 0) continue;
      row_of(held_[static_cast<size_t>(r)], row);
      core::Sampler::raw_logprobs(row.data(), vocab_, t[r], top, out[r]);
    }
  }
  void prefill_logprobs(int32_t token, int top, core::TokenLogprob &out) override {
    std::vector<float> row;
    row_of(pf_pick_, row);
    core::Sampler::raw_logprobs(row.data(), vocab_, token, top, out);
  }
  void prefill_each(int s, const int32_t *t, int n, int32_t next, const EachFn &on_rows) override {
    ++each_calls;
    auto &h = hist_.at(static_cast<size_t>(s));
    for (int r0 = 0; r0 < n; r0 += 4) {
      const int rows = std::min(4, n - r0);
      held_.clear();
      for (int j = 0; j < rows; ++j) {
        h.push_back(t[r0 + j]);
        held_.push_back(pick_of(h));
      }
      on_rows(r0, rows);
    }
    h.resize(h.size() - static_cast<size_t>(n));
    prefill(s, t, n, next);
  }
  // every prompt row's logprobs in one call (the many-row head), from the same rows
  bool rows_lp = false;
  int rows_lp_calls = 0;
  bool prefill_row_logprobs(int s, const int32_t *t, int n, int32_t next, const int32_t *tg, int top, core::TokenLogprob *out) override {
    if (!rows_lp) return false;
    ++rows_lp_calls;
    auto &h = hist_.at(static_cast<size_t>(s));
    std::vector<float> row;
    for (int j = 0; j < n; ++j) {
      h.push_back(t[j]);
      if (tg[j] < 0) continue;
      row_of(pick_of(h), row);
      core::Sampler::raw_logprobs(row.data(), vocab_, tg[j], top, out[j]);
    }
    h.resize(h.size() - static_cast<size_t>(n));
    prefill(s, t, n, next);
    return true;
  }
  void step(const std::vector<int> &slots, const int32_t *t) override {
    std::set<int> u(slots.begin(), slots.end());
    if (slots.empty() || slots.size() > 4 || u.size() != slots.size() || (pend_ && u.count(pend_slot_))) ++errors;
    for (const auto &mb : many_) errors += u.count(mb.slot) > 0;  // a step on a member of the pending call
    many_pick_.clear();  // the head is the step's now
    picks_.clear();
    for (size_t i = 0; i < slots.size(); ++i) {
      auto &h = hist_.at(static_cast<size_t>(slots[i]));
      write_row(slots[i], h.size(), t[i]);
      h.push_back(t[i]);
      if (static_cast<int64_t>(h.size()) > cap_) ++errors;
      kv_check(slots[i], h.size());
      tier_check(slots[i], h.size());
      picks_.push_back(pick_of(h));
    }
    held_ = picks_;
    max_batch = std::max(max_batch, static_cast<int>(slots.size()));
    ++steps;
    if (step_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(step_us));
  }
  void picks(int rows, int32_t *out) override {
    for (int i = 0; i < rows; ++i) out[i] = picks_.at(static_cast<size_t>(i));
  }
  void logits(int rows, std::vector<float> &out) override {
    ++logits_calls;
    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(vocab_), 0.0f);
    if (rich) {
      std::vector<float> row;
      for (int i = 0; i < rows; ++i) {
        row_of(picks_[static_cast<size_t>(i)], row);
        std::copy(row.begin(), row.end(), out.begin() + static_cast<std::ptrdiff_t>(i) * vocab_);
      }
      return;
    }
    for (int i = 0; i < rows; ++i) out[static_cast<size_t>(i) * static_cast<size_t>(vocab_) + static_cast<size_t>(picks_[static_cast<size_t>(i)])] = 8.0f;
  }
  // candidate rows. cand_T > 0 serves StepModel:logits_cands like the cards do: the vocabulary as
  // two "cards" (halves), each row's top T 16-column tiles per half by the tile maximum (ties: the lower tile), ascending,
  // all 16 values of each (an independent host version of kernels:head_cand_tiles on rows whose values are bf16-exact).
  // rich: values in steps of 0.25 by a hash of (pick, id) - many ties, every value bf16-exact - the pick at 3.0;
  // rows of picks divisible by 13 all -inf (the kept set is empty: the sampler's fallback)
  bool rich = false;
  int cand_T = 0, cand_calls = 0;
  int cand_tiles() const override { return cand_T; }
  int logits_cands(int rows, int T, std::vector<int32_t> &ids, std::vector<float> &vals) override {
    ++cand_calls;
    if (T < 1 || T > cand_T || vocab_ % 32) ++errors;
    const int half = vocab_ / 2, nt = half / 16, n = 2 * T * 16;
    ids.assign(static_cast<size_t>(rows) * n, 0);
    vals.assign(ids.size(), 0.0f);
    std::vector<float> row;
    for (int r = 0; r < rows; ++r) {
      row_of(picks_.at(static_cast<size_t>(r)), row);
      size_t o = static_cast<size_t>(r) * n;
      for (int c = 0; c < 2; ++c) {
        std::vector<std::pair<float, int>> tm;
        for (int t = 0; t < nt; ++t) {
          float m = -std::numeric_limits<float>::infinity();
          for (int j = 0; j < 16; ++j) m = std::max(m, row[static_cast<size_t>(c * half + t * 16 + j)]);
          tm.push_back({m, t});
        }
        std::stable_sort(tm.begin(), tm.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        std::vector<int> sel;
        for (int i = 0; i < T; ++i) sel.push_back(tm[static_cast<size_t>(i)].second);
        std::sort(sel.begin(), sel.end());
        for (int t : sel)
          for (int j = 0; j < 16; ++j, ++o) {
            ids[o] = c * half + t * 16 + j;
            vals[o] = row[static_cast<size_t>(ids[o])];
          }
      }
    }
    return n;
  }
  // A drafter (several sequences): up to `depth` drafts per sequence (4 rows in
  // all, like the qwen38 adapter), the model's own picks except where a hash of the position says
  // wrong (so steps accept 0..depth drafts).
  int depth = 0;
  int draft_depth(int n) const override {
    if (plain_from > 0 && n >= plain_from) return 0;  // plain from n sequences (the 27B's depth table)
    return depth <= 0 ? 0 : n <= 1 ? depth : std::min(depth, step_rows_at(n) / n - 1);
  }
  int plain_from = 0, lay_depth = -1, max_het_b = 0;  // StepModel:layout_depth (-1 = the default)
  int layout_depth(int n) const override { return lay_depth >= 0 ? lay_depth : core::StepModel::layout_depth(n); }
  std::vector<int32_t> first_drafts(int s, int32_t first, int d, int n) override {
    (void)n;
    if (on_first_drafts) on_first_drafts();
    many_pick_.clear();  // the drafter reuses the head
    const auto &h = hist_.at(static_cast<size_t>(s));
    if (!h.empty()) write_drow(s, h.size() - 1, first);
    return drafts_of(s, h, first, d);
  }
  // tree = the drafts are a tree (draft_parents): along the chain drafts_for gives, each node first
  // gets a leaf sibling (the right token where the chain's draft is wrong, else a wrong one), then the chain node
  bool tree = false;
  int tree_verifies = 0, tree_side_accepts = 0, multi_tree_verifies = 0;  // tree verifies of several segments
  // StepModel:wide_depth / wide_from (a lone sequence's wider block after a good verify)
  int wide_d = 0, wide_f = 0, wide_verifies = 0;
  int wide_depth() const override { return wide_d; }
  int wide_from() const override { return wide_f; }
  std::vector<std::vector<int>> tree_par_;
  std::vector<int> draft_parents(int s) const override {
    return static_cast<size_t>(s) < tree_par_.size() ? tree_par_[static_cast<size_t>(s)] : std::vector<int>{};
  }
  // StepModel:verifies_trees: no_tree_every > 0 = every that-many-th step that holds a tree cannot
  // verify trees; the engine then sends the trees' first paths as chains (tree_refusals asks answered no,
  // trees_as_chains segments whose drafts were a tree and came as a chain, a tree after a refusal = an error)
  int no_tree_every = 0;
  mutable int tree_asks = 0, tree_refusals = 0;
  mutable bool refused_ = false;
  int trees_as_chains = 0;
  bool verifies_trees(std::span<const int> slots, std::span<const int> rows) const override {
    if (slots.size() != rows.size() || slots.empty()) return true;
    refused_ = no_tree_every > 0 && ++tree_asks % no_tree_every == 0;
    tree_refusals += refused_;
    return !refused_;
  }
  // unequal segments (the qwen38 layouts: the first segments sr rows each, then one-row segments)
  bool uneq = false;
  int het_verifies = 0;
  bool unequal_segments() const override { return uneq; }
  // two classes (the first segments sr rows each, then b2 < sr rows each); deep: some requests
  // draft `deep` more than the one depth where the step's rows are free
  bool classes = false;
  int deep = 0, het2_verifies = 0;
  bool unequal_classes() const override { return classes; }
  int deep_depth(int n) const override { return deep > 0 && draft_depth(n) > 0 ? draft_depth(n) + deep : 0; }
  double draft_pass_ms(int n, int d) const override { return deep > 0 && d > 0 ? 0.02 + 0.01 * d + 0 * n : 0; }
  double spec_cost_rows(int n, int rows, bool drafting) const override {
    return uneq && verify_us > 0 ? verify_us + static_cast<double>(verify_us_row) * rows + (drafting ? 0 : -1) + 0 * n : 0;
  }
  void verify(const std::vector<SpecSeg> &segs) override {
    many_pick_.clear();  // the head is the step's now
    const int sr = segs.empty() ? 0 : segs[0].rows;
    int rows = 0, one = 0;
    for (const auto &g : segs) rows += g.rows, one += g.rows == 1;
    if (segs.empty() || sr < 1 || rows > step_rows_at(static_cast<int>(segs.size()))) ++errors;
    int &most = static_cast<int>(segs.size()) < rows_small_below ? max_rows_small_steps : max_rows_large_steps;
    most = std::max(most, rows);
    het_verifies += rows != sr * static_cast<int>(segs.size());
    if (rows != sr * static_cast<int>(segs.size())) max_het_b = std::max(max_het_b, sr);
    picks_.clear();
    pending_.clear();
    pending_par_.clear();
    int b2 = 0;  // the rows of the segments after the sr-row ones (0: none yet)
    bool any_tree = false;
    for (const auto &g : segs) {
      if (uneq && g.rows < sr && (g.rows == 1 || classes) && (b2 == 0 || g.rows == b2)) b2 = g.rows;
      else if (g.rows != sr || b2 > 0) ++errors;
      if (g.parent) {  // row r's pick follows its ancestors' tokens only (any segment of a step)
        any_tree = true;
        if (refused_) ++errors;  // a tree in a step the model said it cannot verify trees in
        const auto &h = hist_.at(static_cast<size_t>(g.slot));
        kv_check(g.slot, h.size() + static_cast<size_t>(g.rows));
        std::vector<int> par(g.parent, g.parent + g.rows);
        if (par[0] != -1) ++errors;
        for (int r = 0; r < g.rows; ++r) {
          if (r > 0 && (par[static_cast<size_t>(r)] < 0 || par[static_cast<size_t>(r)] >= r)) ++errors;
          std::vector<int> anc;
          for (int x = r; x >= 0 && anc.size() <= static_cast<size_t>(g.rows); x = x == 0 ? -1 : par[static_cast<size_t>(x)]) anc.insert(anc.begin(), x);
          auto hh = h;
          for (int x : anc) hh.push_back(g.tokens[x]);
          picks_.push_back(pick_of(hh));
        }
        pending_.push_back({g.slot, std::vector<int32_t>(g.tokens, g.tokens + g.rows)});
        pending_par_.push_back(par);
        continue;
      }
      auto h = hist_.at(static_cast<size_t>(g.slot));
      trees_as_chains += refused_ && g.rows > 1 && static_cast<size_t>(g.slot) < tree_par_.size() && !tree_par_[static_cast<size_t>(g.slot)].empty();
      kv_check(g.slot, h.size() + static_cast<size_t>(g.rows));
      tier_check(g.slot, h.size() + static_cast<size_t>(g.rows));
      for (int k = 0; k < g.rows; ++k) {
        write_row(g.slot, h.size(), g.tokens[k]);
        h.push_back(g.tokens[k]);
        picks_.push_back(pick_of(h));
      }
      pending_.push_back({g.slot, std::vector<int32_t>(g.tokens, g.tokens + g.rows)});
      pending_par_.push_back({});
    }
    het2_verifies += b2 > 1;
    refused_ = false;
    if (any_tree) ++tree_verifies, multi_tree_verifies += segs.size() > 1;
    max_batch = std::max(max_batch, static_cast<int>(segs.size()));
    held_ = picks_;
    max_spec_rows = std::max(max_spec_rows, rows);
    (void)one;
    two_by_two += segs.size() == 2 && sr == 2;
    full_verifies += rows == rows_max;
    wide_verifies += wide_d > 0 && segs.size() == 1 && rows > 8;
    ++verifies;
    one_row_verifies += sr == 1;
    if (verify_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(verify_us + verify_us_row * rows));
    ++steps;
  }
  std::vector<std::vector<int32_t>> accept(const std::vector<SpecAccept> &a) override {
    if (on_accept) on_accept();
    std::vector<std::vector<int32_t>> out;
    if (a.size() != pending_.size()) ++errors;
    for (size_t k = 0; k < a.size() && k < pending_.size(); ++k) {
      const auto &[slot, toks] = pending_[k];
      const int n = a[k].n;
      if (n < 1 || n > static_cast<int>(toks.size())) ++errors;
      auto &h = hist_.at(static_cast<size_t>(slot));
      for (int r = 0; r < static_cast<int>(toks.size()); ++r) write_drow(slot, h.size() + static_cast<size_t>(r), a[k].next[std::min(r, n - 1)]);
      if (a[k].path) {  // the walked rows of a tree
        const auto &par = pending_par_.at(k);
        if (par.empty() || a[k].path[0] != 0) ++errors;
        for (int i = 0; i < n && !par.empty(); ++i) {
          const int r = a[k].path[i];
          if (r < 0 || r >= static_cast<int>(toks.size()) || (i > 0 && par[static_cast<size_t>(r)] != a[k].path[i - 1])) {
            ++errors;
            break;
          }
          tree_side_accepts += i > 0 && r != i;
          write_row(slot, h.size(), toks[static_cast<size_t>(r)]);
          h.push_back(toks[static_cast<size_t>(r)]);
        }
      } else {
        if (!pending_par_.empty() && !pending_par_.at(k).empty()) ++errors;  // a tree accepted as a chain
        h.insert(h.end(), toks.begin(), toks.begin() + n);
      }
      if (static_cast<int64_t>(h.size()) > cap_) ++errors;
      if (a[k].depth > 0) kv_check(slot, std::min<size_t>(static_cast<size_t>(cap_), h.size() + static_cast<size_t>(a[k].depth)));
      out.push_back(a[k].depth > 0 ? drafts_of(slot, h, a[k].next[n - 1], a[k].depth) : std::vector<int32_t>{});
      if (a[k].depth <= 0 && static_cast<size_t>(slot) < tree_par_.size()) tree_par_[static_cast<size_t>(slot)].clear();
    }
    pending_.clear();
    pending_par_.clear();
    ++accepts;
    return out;
  }
  int max_spec_rows = 0, two_by_two = 0, full_verifies = 0;
  // drafts never right; a verify sleeps verify_us + verify_us_row per row; spec_cost the same shape
  bool all_wrong = false;
  int copy_period = 0;  // > 0: the pick repeats the token copy_period back (output that copies its prompt: lookup drafting)
  int verify_us = 0, verify_us_row = 0, one_row_verifies = 0, verifies = 0;
  double spec_cost(int n, int rows) const override { return verify_us > 0 ? verify_us + static_cast<double>(verify_us_row) * n * rows : 0; }
  int accepts = 0;
  // called at the start of accept (the drafter pass after a verify) / first_drafts (after a prompt's call)
  std::function<void()> on_accept, on_first_drafts;
  int errors = 0, steps = 0, prefills = 0, max_batch = 0;
  int pf_us_fixed = 0, pf_us_row = 0, step_us = 0;  // simulated time per prefill call / row / step
  std::atomic<int> pf_us_once{0};  // added to the next prefill call only (a first use)
  struct PfCall {
    int slot, n, steps_before;
  };
  std::vector<PfCall> pf_log;
  int32_t pick_after(const std::vector<int32_t> &h) const { return pick_of(h); }
  int32_t eos_every = 0;  // > 0: a history whose length is a multiple of this picks token 0 (the EOS in the tests)

 private:
  int32_t pick_of(const std::vector<int32_t> &h) const {
    if (eos_every > 0 && h.size() % static_cast<size_t>(eos_every) == 0) return 0;
    if (copy_period > 0 && h.size() >= static_cast<size_t>(copy_period)) return h[h.size() - static_cast<size_t>(copy_period)];
    uint64_t x = 1469598103934665603ull;
    for (int32_t v : h) x = (x ^ static_cast<uint64_t>(v)) * 1099511628211ull;
    return static_cast<int32_t>(1 + x % static_cast<uint64_t>(vocab_ - 1));
  }
  std::vector<int32_t> held_;  // the picks of the rows the "head" holds (row_logprobs)
  void row_of(int32_t p, std::vector<float> &row) const {
    row.assign(static_cast<size_t>(vocab_), 0.0f);
    if (rich) {
      for (int i = 0; i < vocab_; ++i) {
        uint64_t x = (static_cast<uint64_t>(p) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(i) * 0xC2B2AE3D27D4EB4Full);
        x ^= x >> 29;
        row[static_cast<size_t>(i)] = p % 13 == 0 ? -std::numeric_limits<float>::infinity() : static_cast<float>(static_cast<int>(x % 21) - 14) * 0.25f;
      }
      if (p % 13 != 0) row[static_cast<size_t>(p)] = 3.0f;
      return;
    }
    row[static_cast<size_t>(p)] = 8.0f;
  }
  std::vector<std::vector<int32_t>> hist_;
  std::vector<bool> begun_;
  int64_t cap_;
  int chunk_, vocab_;
  int32_t pf_pick_ = 0;
  std::vector<int32_t> picks_;
  std::vector<std::pair<int, std::vector<int32_t>>> pending_;  // the last verify's segments (slot, tokens)
  std::vector<std::vector<int>> pending_par_;  // their tree parents (empty: a chain)
  std::vector<int32_t> drafts_of(int s, const std::vector<int32_t> &h, int32_t next, int n) {
    auto d = drafts_for(h, next, n);
    if (tree_par_.size() <= static_cast<size_t>(s)) tree_par_.resize(static_cast<size_t>(s) + 1);
    auto &par = tree_par_[static_cast<size_t>(s)];
    par.clear();
    if (!tree || d.empty()) return d;
    std::vector<int32_t> t;
    par.push_back(-1);
    auto hh = h;
    hh.push_back(next);
    int cur = 0;
    for (size_t k = 0; k < d.size() && static_cast<int>(t.size()) < n; ++k) {
      const int32_t right = pick_of(hh);
      if (static_cast<int>(t.size()) + 2 <= n) {  // the leaf sibling
        int32_t alt = d[k] != right ? right : 1 + (right % (vocab_ - 1));
        if (alt == d[k]) alt = 1 + (alt % (vocab_ - 1));
        t.push_back(alt);
        par.push_back(cur);
      }
      t.push_back(d[k]);
      par.push_back(cur);
      cur = static_cast<int>(t.size());
      hh.push_back(d[k]);
    }
    return t;
  }
  std::vector<int32_t> drafts_for(std::vector<int32_t> h, int32_t next, int n) const {
    std::vector<int32_t> d;
    h.push_back(next);
    for (int k = 0; k < n; ++k) {
      int32_t x = pick_of(h);
      if (all_wrong || (h.size() * 7 + static_cast<size_t>(k)) % 5 == 0) x = 1 + (x % (vocab_ - 1));  // a wrong draft
      d.push_back(x);
      h.push_back(x);
    }
    return d;
  }
};

struct Collect final : core::EventSink {
  std::mutex mu;
  std::condition_variable cv;
  std::map<uint64_t, std::vector<int32_t>> toks;
  std::map<uint64_t, core::Finish> fin;
  std::map<uint64_t, bool> stop_tok;
  std::map<uint64_t, int> lps;
  std::map<uint64_t, std::vector<int32_t>> cached;  // each event's cached_tokens report (>= 0)
  void on_events(const core::Event *e, size_t n) override {
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < n; ++i) {
      if (e[i].cached_tokens >= 0) cached[e[i].id].push_back(e[i].cached_tokens);
      auto &t = toks[e[i].id];
      t.insert(t.end(), e[i].tokens, e[i].tokens + e[i].n);
      lps[e[i].id] += static_cast<int>(e[i].logprobs.size());
      if (e[i].finish != core::Finish::none) {
        fin[e[i].id] = e[i].finish;
        stop_tok[e[i].id] = e[i].stop_token;
      }
    }
    cv.notify_all();
  }
  bool wait(size_t n, double s = 20) {
    std::unique_lock<std::mutex> lk(mu);
    return cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return fin.size() >= n; });
  }
};

std::vector<int32_t> prompt_of(int i, int len) {
  std::vector<int32_t> p(static_cast<size_t>(len));
  for (int k = 0; k < len; ++k) p[static_cast<size_t>(k)] = 1 + (i * 131 + k * 7) % 200;
  return p;
}

// Candidate rows instead of full logits rows. The same seeded sampled requests on a
// model that serves logits_cands and on one that does not give the same tokens (the kept sets and draws are the same),
// in plain steps and in speculative steps (drafter depth 3), with top_k 1..32, top_p, min_p; a request that reads more
// (top_k off, logprobs, a penalty) makes its steps fetch full rows; rows with nothing finite fall back to the full row.
void test_cands() {
  const int V = 4096;
  struct Cfg {
    float t, top_p, min_p;
    int top_k, logprobs = -1;
    float presence = 0;
  };
  const std::vector<Cfg> cfgs = {{0.7f, 1.0f, 0.0f, 20}, {1.0f, 0.95f, 0.0f, 32}, {1.3f, 0.9f, 0.05f, 5}, {0.5f, 1.0f, 0.0f, 1},
                                 {1.0f, 0.8f, 0.0f, 0},  {0.8f, 1.0f, 0.0f, 20, 3}, {0.9f, 1.0f, 0.0f, 20, -1, 0.5f}};
  for (int depth : {0, 3}) {
    std::vector<std::map<uint64_t, std::vector<int32_t>>> out(2);
    std::vector<int> calls(2), full(2);
    for (int v = 0; v < 2; ++v) {
      FakeModel fm(1, 4096, 16, V);
      fm.rich = true;
      fm.depth = depth;
      fm.cand_T = v ? 32 : 0;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.adaptive_depth = false;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      for (size_t i = 0; i < cfgs.size(); ++i)
        for (int s = 0; s < 3; ++s) {
          core::SamplingParams p;
          p.temperature = cfgs[i].t;
          p.top_p = cfgs[i].top_p;
          p.min_p = cfgs[i].min_p;
          p.top_k = cfgs[i].top_k;
          p.logprobs = cfgs[i].logprobs;
          p.presence_penalty = cfgs[i].presence;
          p.has_seed = true;
          p.seed = 77 + 1000 * i + s;
          p.max_tokens = 60;
          const uint64_t id = 1 + i * 10 + static_cast<uint64_t>(s);
          e.submit(id, prompt_of(static_cast<int>(i * 3 + s), 9 + s), p);
          c.wait(static_cast<size_t>(i * 3 + s + 1), 30);
        }
      e.stop();
      out[static_cast<size_t>(v)] = c.toks;
      calls[static_cast<size_t>(v)] = fm.cand_calls;
      full[static_cast<size_t>(v)] = fm.logits_calls;
      expect(fm.errors == 0, "cands: no model misuse (depth " + std::to_string(depth) + (v ? ", candidates)" : ", full rows)"));
    }
    expect(out[0].size() == 21 && out[0] == out[1],
           "cands: 21 seeded sampled requests (top_k 1-32, top_p, min_p, nothing-finite rows) == the full-row path, depth " + std::to_string(depth));
    expect(calls[0] == 0 && calls[1] > 0 && full[1] < full[0],
           "cands: the candidate path served the eligible steps (" + std::to_string(calls[1]) + " calls; full-row fetches " + std::to_string(full[0]) + " -> " +
               std::to_string(full[1]) + "), depth " + std::to_string(depth));
  }
}

// A drafter whose drafts are trees (StepModel:draft_parents). One greedy
// request verifies its tree (SpecSeg:parent) and accepts the walked rows (SpecAccept:path), side leaves included; the
// output == plain decoding and == the same drafts as chains; sampled requests and the lookup's drafts verify chains
// (greedy requests verify trees in steps of several too). The engine's walk never misuses the
// model (parents, paths).
void test_tree_spec() {
  const int V = kVocab;
  // sampled: 0 = greedy requests, 1 = the odd ones sampled, 2 = all sampled
  int wide = 0, wide_v = 0, wide_max_rows = 0;  // > 0 = the model's wide_depth 15 from `wide` accepted drafts
  int no_tree_every = 0, refusals = 0, as_chains = 0;  // FakeModel:no_tree_every and what it counted
  auto run = [&](int depth, bool tree, int reqs, int sampled, std::map<uint64_t, std::vector<int32_t>> &out, int &tv, int &side, int &errors,
                 uint64_t &acc_tok, int &verifies, int *multi = nullptr) {
    FakeModel fm(4, 4096, 16, V);
    fm.depth = depth;
    fm.rows_max = 16;
    fm.tree = tree;
    fm.wide_d = wide > 0 ? 15 : 0, fm.wide_f = wide;
    fm.no_tree_every = no_tree_every;
    fm.eos_every = 97;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.adaptive_depth = false;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < reqs; ++i) {
      core::SamplingParams p;
      p.temperature = sampled == 2 || (sampled == 1 && i % 2) ? 0.8f : 0.0f;
      if (sampled == 2 && i % 2) p.presence_penalty = 1.5f;  // a tree row's kept set follows its path's tokens
      p.has_seed = true;
      p.seed = 11 + i;
      p.max_tokens = 150;
      p.logprobs = i == 0 ? 2 : -1;
      e.submit(static_cast<uint64_t>(1 + i), prompt_of(30 + i, 12 + 5 * i), p);
    }
    c.wait(static_cast<size_t>(reqs), 60);
    const core::Stats st = e.stats();
    e.stop();
    out = c.toks;
    tv = fm.tree_verifies, side = fm.tree_side_accepts, errors = fm.errors, acc_tok = st.spec_accepted_tokens, verifies = fm.verifies;
    if (multi) *multi = fm.multi_tree_verifies;
    wide_v = fm.wide_verifies, wide_max_rows = fm.max_spec_rows;
    refusals = fm.tree_refusals, as_chains = fm.trees_as_chains;
  };
  std::map<uint64_t, std::vector<int32_t>> plain, chain, tree;
  int tv = 0, side = 0, err = 0, vc = 0, vt = 0, vp = 0;
  uint64_t ac = 0, at = 0, ap = 0;
  run(0, false, 1, false, plain, tv, side, err, ap, vp);
  run(7, false, 1, false, chain, tv, side, err, ac, vc);
  const int err_c = err;
  run(7, true, 1, false, tree, tv, side, err, at, vt);
  expect(err == 0 && err_c == 0 && plain.size() == 1 && tree == plain && chain == plain && tv > 0 && side > 0 && vt < vc,
         "batch engine, draft trees, one greedy request: " + std::to_string(tv) + " tree verifies (" + std::to_string(side) + " side rows kept), " +
             std::to_string(vt) + " verifies vs " + std::to_string(vc) + " as chains, output == plain and == chains");
  {  // a lone request drafts 15 (a 16-row tree) after a verify that kept 3+ drafts; greedy, then
     // sampled with a seed; output == plain, the wide steps happen, no model misuse (16 rows at most)
    std::map<uint64_t, std::vector<int32_t>> wt, ws, ps;
    int ew = 0, es = 0, ep = 0, wv = 0, wr = 0, tvw = 0;
    wide = 3;
    run(7, true, 1, 0, wt, tvw, side, ew, at, vt);
    wv = wide_v, wr = wide_max_rows;
    run(7, true, 1, 2, ws, tv, side, es, at, vt);
    const int wvs = wide_v;
    wide = 0;
    run(0, false, 1, 2, ps, tv, side, ep, at, vp);
    expect(ew == 0 && es == 0 && ep == 0 && wt == plain && ws == ps && wv > 0 && wvs > 0 && wr == 16,
           "batch engine, 16-row trees after 3+ accepted drafts (StepModel::wide_depth 15), one request: " + std::to_string(wv) + " greedy / " +
               std::to_string(wvs) + " sampled 16-row verifies (" + std::to_string(tvw) + " tree verifies), output == plain (greedy) and == the same seeded request without drafts");
  }
  {  // a depth table of 15 for one sequence (XEN_DRAFT_DEPTHS 15.. with XEN_SEG_ROWS 16): every step up to 16
     // rows (an event of up to 16 tokens), trees and chains, greedy and seeded sampled: output == plain
    std::map<uint64_t, std::vector<int32_t>> dt, dc, ds, ps;
    int e1 = 0, e2 = 0, e3 = 0, e4 = 0, r1 = 0, r2 = 0, tvd = 0;
    run(15, true, 1, 0, dt, tvd, side, e1, at, vt);
    r1 = wide_max_rows;
    run(15, false, 1, 0, dc, tv, side, e2, at, vt);
    r2 = wide_max_rows;
    run(15, true, 1, 2, ds, tv, side, e3, at, vt);
    run(0, false, 1, 2, ps, tv, side, e4, at, vp);
    expect(e1 == 0 && e2 == 0 && e3 == 0 && e4 == 0 && dt == plain && dc == plain && ds == ps && r1 == 16 && r2 == 16,
           "batch engine, depth 15 for one sequence (16-row segments): trees (" + std::to_string(tvd) + " tree verifies) and chains reach 16 rows, output == plain (greedy) and == the seeded request without drafts");
  }
  {  // a model that cannot verify trees in every third step holding one (StepModel:verifies_trees):
     // the engine sends the trees' first paths as chains there. One greedy request, then 3 (the odd one sampled).
    for (const int reqs : {1, 3}) {
      std::map<uint64_t, std::vector<int32_t>> all_trees, some_chains;
      int tva = 0, tvs = 0, errs = 0, va = 0, vs = 0;
      uint64_t acc = 0;
      run(7, true, reqs, 1, all_trees, tva, side, err, acc, va);
      const int err_all = err;
      no_tree_every = 3;
      run(7, true, reqs, 1, some_chains, tvs, side, errs, acc, vs);
      no_tree_every = 0;
      expect(err_all == 0 && errs == 0 && static_cast<int>(some_chains.size()) == reqs && some_chains == all_trees && refusals > 0 && as_chains > 0 && tva > 0 && tvs > 0,
             "batch engine, draft trees with steps that cannot verify them, " + std::to_string(reqs) + " request(s): " + std::to_string(refusals) +
                 " steps refused (" + std::to_string(as_chains) + " trees sent as their first path), " + std::to_string(tvs) +
                 " tree verifies in the others, output == every step verifying trees, no tree in a refused step");
    }
  }
  // several requests verify their trees in one step (the greedy ones; sampled requests and the
  // lookup's drafts verify chains), output == the same drafts as chains, no model misuse
  int tv4 = 0, mt = 0;
  run(7, true, 3, true, tree, tv4, side, err, at, vt, &mt);
  std::map<uint64_t, std::vector<int32_t>> tree3 = tree;
  int err3 = err;
  run(7, false, 3, true, chain, tv, side, err, ac, vc);
  expect(err3 == 0 && err == 0 && tree3.size() == 3 && tree3 == chain && mt > 0,
         "batch engine, draft trees, 3 requests (sampled ones among them): " + std::to_string(tv4) + " tree verifies, " + std::to_string(mt) +
             " of them with several segments, output == chain drafts, no model misuse");
  // sampled requests verify their trees too (the walk takes the sampler's draws), output == the
  // same requests without drafts; one sampled request alone, then 3 all sampled (several-segment tree verifies)
  for (int reqs : {1, 3}) {
    std::map<uint64_t, std::vector<int32_t>> ps, ts;
    int tvs = 0, sides = 0, ms = 0, errp = 0;
    run(0, false, reqs, 2, ps, tv, side, errp, ap, vp);
    run(7, true, reqs, 2, ts, tvs, sides, err, at, vt, &ms);
    expect(errp == 0 && err == 0 && static_cast<int>(ts.size()) == reqs && ts == ps && tvs > 0 && sides > 0 && (reqs == 1 || ms > 0),
           "batch engine, draft trees, " + std::to_string(reqs) + " sampled request(s): " + std::to_string(tvs) + " tree verifies (" + std::to_string(sides) +
               " side rows kept, " + std::to_string(ms) + " with several segments), output == plain sampling");
  }
  run(7, true, 3, false, tree, tv4, side, err, at, vt, &mt);
  tree3 = tree, err3 = err;
  const int side3 = side;
  run(0, false, 3, false, plain, tv, side, err, ap, vp);
  expect(err3 == 0 && err == 0 && tree3.size() == 3 && tree3 == plain && mt > 0 && side3 > 0,
         "batch engine, draft trees, 3 greedy requests: " + std::to_string(mt) + " several-segment tree verifies (" + std::to_string(side3) +
             " side rows kept), output == plain");
}

// Events published before the drafter work they do not need. A speculative
// step's going tokens reach the sink after its verify and before its accept (the drafter pass), a prompt's first token
// before its first drafts (and the other requests' step); a finishing event arrives after the engine counted the finish.
// The tokens equal each request run alone, with drafts (equal and unequal segments) and without (plain steps).
void test_early_events() {
  const int N = 6;
  const int lens[N] = {3, 17, 40, 95, 9, 300};
  for (int cfg = 0; cfg < 3; ++cfg) {  // 0 plain steps, 1 drafter depth 3, 2 the same on unequal segments
    const std::string tag = cfg == 0 ? "plain" : cfg == 1 ? "drafts" : "drafts, unequal segments";
    auto run = [&](std::map<uint64_t, std::vector<int32_t>> &out, int &early_steps, int &late_steps, int &fd_bad, int &fin_bad, int &errors) {
      FakeModel fm(4, 4096, 16, kVocab);
      fm.eos_every = 23;
      fm.depth = cfg == 0 ? 0 : 3;
      fm.uneq = cfg == 2;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      struct Sink final : core::EventSink {
        std::mutex mu;
        std::condition_variable cv;
        std::map<uint64_t, std::vector<int32_t>> toks;
        std::set<uint64_t> seen;
        size_t finished = 0;
        int early = 0, late = 0, fin_bad = 0;
        FakeModel *fm = nullptr;
        core::BatchEngine *eng = nullptr;
        void on_events(const core::Event *ev, size_t n) override {
          std::lock_guard<std::mutex> lk(mu);
          for (size_t i = 0; i < n; ++i) {
            const bool first = seen.insert(ev[i].id).second;
            auto &t = toks[ev[i].id];
            t.insert(t.end(), ev[i].tokens, ev[i].tokens + ev[i].n);
            if (ev[i].finish != core::Finish::none) {
              ++finished;
              fin_bad += eng->stats().requests_finished < finished;  // counted before its finish is handed over
            } else if (!first && fm->depth > 0) {
              (fm->verifies == fm->accepts + 1 ? early : late)++;  // a step's tokens: between its verify and its accept
            }
          }
          cv.notify_all();
        }
      } sink;
      sink.fm = &fm;
      sink.eng = &e;
      int fds = 0;
      fm.on_first_drafts = [&] {  // every first-drafts call follows its request's first token at the sink
        std::lock_guard<std::mutex> lk(sink.mu);
        fd_bad += sink.seen.size() < static_cast<size_t>(++fds);
      };
      e.start(&sink);
      for (int i = 0; i < N; ++i) {
        core::SamplingParams p;
        p.temperature = 0;
        p.max_tokens = 30 + 7 * i;
        e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, lens[i]), p);
      }
      {
        std::unique_lock<std::mutex> lk(sink.mu);
        sink.cv.wait_for(lk, std::chrono::seconds(30), [&] { return sink.finished >= static_cast<size_t>(N); });
      }
      e.stop();
      out = sink.toks;
      early_steps = sink.early, late_steps = sink.late, fin_bad = sink.fin_bad, errors = fm.errors;
    };
    std::map<uint64_t, std::vector<int32_t>> together;
    int early = 0, late = 0, fd_bad = 0, fin_bad = 0, errors = 0;
    run(together, early, late, fd_bad, fin_bad, errors);
    // the solo reference: request k alone gives the same tokens (the engine's outputs do not depend on the batch)
    bool same = together.size() == static_cast<size_t>(N);
    for (int i = 0; i < N && same; ++i) {
      FakeModel fm(1, 4096, 16, kVocab);
      fm.eos_every = 23;
      fm.depth = cfg == 0 ? 0 : 3;
      fm.uneq = cfg == 2;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 30 + 7 * i;
      e.submit(1, prompt_of(i, lens[i]), p);
      c.wait(1);
      e.stop();
      same &= c.toks[1] == together[static_cast<uint64_t>(i + 1)];
    }
    expect(errors == 0 && same, "early events (" + tag + "): 6 requests on 4 slots == each alone, no model misuse");
    expect(fin_bad == 0, "early events (" + tag + "): every finish arrives after the engine counted it");
    if (cfg > 0) {
      expect(early > 0 && late == 0, "early events (" + tag + "): every going step's tokens before its drafter pass (" + std::to_string(early) + " early, " +
                                         std::to_string(late) + " late)");
      expect(fd_bad == 0, "early events (" + tag + "): every first token before its first drafts");
    }
  }
}

// A request whose prompt completes while another decodes joins the step of the same iteration:
// between its first token and the event of its first step the model ran exactly one step.
void test_join_step() {
  for (int cfg = 0; cfg < 2; ++cfg) {  // 0 plain steps, 1 drafter depth 3
    const std::string tag = cfg == 0 ? "plain" : "drafts";
    FakeModel fm(2, 4096, 16, kVocab);
    fm.depth = cfg == 0 ? 0 : 3;
    fm.step_us = 300, fm.verify_us = 300;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    struct Sink final : core::EventSink {
      std::mutex mu;
      std::condition_variable cv;
      FakeModel *fm = nullptr;
      std::map<uint64_t, std::vector<int>> at;  // per request: the model's step count at each of its events
      std::map<uint64_t, std::vector<int32_t>> toks;
      size_t finished = 0;
      void on_events(const core::Event *ev, size_t n) override {
        std::lock_guard<std::mutex> lk(mu);
        for (size_t i = 0; i < n; ++i) {
          at[ev[i].id].push_back(fm->steps);
          auto &t = toks[ev[i].id];
          t.insert(t.end(), ev[i].tokens, ev[i].tokens + ev[i].n);
          finished += ev[i].finish != core::Finish::none;
        }
        cv.notify_all();
      }
    } sink;
    sink.fm = &fm;
    e.start(&sink);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 2000;
    e.submit(1, prompt_of(0, 5), p);
    {
      std::unique_lock<std::mutex> lk(sink.mu);
      sink.cv.wait_for(lk, std::chrono::seconds(10), [&] { return sink.at[1].size() >= 3; });  // request 1 is decoding
    }
    p.max_tokens = 40;
    e.submit(2, prompt_of(1, 9), p);
    {
      std::unique_lock<std::mutex> lk(sink.mu);
      sink.cv.wait_for(lk, std::chrono::seconds(30), [&] { return sink.finished >= 2; });
    }
    e.stop();
    const auto &b = sink.at[2];
    const bool beside = b.size() >= 2 && sink.at[1].size() >= 3 && sink.at[1].back() > b[1];  // request 1 still decoding then
    expect(fm.errors == 0 && sink.finished == 2 && beside && b[1] == b[0] + 1,
           "join (" + tag + "): a prompt completing beside a decoding request joins the same iteration's step (" +
               (b.size() >= 2 ? std::to_string(b[1] - b[0]) : std::string("?")) + " step(s) between its first token and its first step's event)");
    FakeModel solo(1, 4096, 16, kVocab);
    solo.depth = fm.depth;
    core::BatchEngine e1(solo, bc);
    Collect c;
    e1.start(&c);
    e1.submit(1, prompt_of(1, 9), p);
    c.wait(1);
    e1.stop();
    expect(c.toks[1] == sink.toks[2], "join (" + tag + "): its tokens == the request alone");
  }
}

// One prefill call for several waiting prompts (BatchConfig:prefill_merge).
void test_prefill_merge() {
  const int V = kVocab;
  auto solo_of = [&](int seed, int len, int max_tokens, int depth) {
    FakeModel fm(1, 8192, 1024, V);
    fm.depth = depth;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = max_tokens;
    e.submit(1, prompt_of(seed, len), p);
    c.wait(1);
    e.stop();
    return c.toks[1];
  };
  for (int depth : {0, 3}) {
    const std::string tag = depth ? "merge (drafts)" : "merge (plain)";
    // 1. an idle engine: six prompts arrive together (four run at a time); 20 rows is under the model's merge class,
    // 700 over it
    {
      const int lens[6] = {40, 100, 64, 200, 20, 700};
      FakeModel fm(8, 8192, 1024, V);
      fm.depth = depth;
      fm.split_layers = 8;
      fm.merge_max = 8;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_merge = 8;
      core::BatchEngine e(fm, bc);
      Collect c;
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 12;
      for (int i = 0; i < 6; ++i) e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, lens[i]), p);
      e.start(&c);
      c.wait(6);
      e.stop();
      bool same = true;
      for (int i = 0; i < 6; ++i) same = same && c.toks[static_cast<uint64_t>(i + 1)] == solo_of(i, lens[i], 12, depth);
      expect(fm.errors == 0 && fm.merge_stale == 0 && fm.merge_calls == 1 && fm.merge_members == 4 && fm.split_aborts == 0,
             tag + ": six prompts at once: the four of 40 / 100 / 64 / 200 rows share one call, 20 and 700 rows run alone after them (" + std::to_string(fm.merge_calls) + " call(s), " +
                 std::to_string(fm.merge_members) + " members, " + std::to_string(fm.errors) + " errors)");
      expect(same, tag + ": every request's tokens == the request alone");
    }
    // 2. the rows cap: four 300-row prompts do not fit 1,023 rows: three share a call, the fourth runs alone
    {
      FakeModel fm(4, 8192, 1024, V);
      fm.depth = depth;
      fm.split_layers = 8;
      fm.merge_max = 8;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_merge = 8;
      core::BatchEngine e(fm, bc);
      Collect c;
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 8;
      for (int i = 0; i < 4; ++i) e.submit(static_cast<uint64_t>(i + 1), prompt_of(10 + i, 300), p);
      e.start(&c);
      c.wait(4);
      e.stop();
      bool same = true;
      for (int i = 0; i < 4; ++i) same = same && c.toks[static_cast<uint64_t>(i + 1)] == solo_of(10 + i, 300, 8, depth);
      expect(fm.errors == 0 && fm.merge_calls == 1 && fm.merge_members == 3 && same, tag + ": four 300-row prompts: three in one call (304 + 304 + 300 rows), one alone, tokens == alone");
    }
    // 3. the setting: 2 members per call; off (the default) = no merged call
    for (int setting : {2, 0}) {
      FakeModel fm(4, 8192, 1024, V);
      fm.depth = depth;
      fm.split_layers = 8;
      fm.merge_max = 8;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_merge = setting;
      core::BatchEngine e(fm, bc);
      Collect c;
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 8;
      for (int i = 0; i < 4; ++i) e.submit(static_cast<uint64_t>(i + 1), prompt_of(20 + i, 50 + 10 * i), p);
      e.start(&c);
      c.wait(4);
      e.stop();
      expect(fm.errors == 0 && fm.merge_calls == (setting ? 2 : 0) && fm.merge_members == (setting ? 4 : 0),
             tag + ": prefill_merge " + std::to_string(setting) + ": " + (setting ? "two calls of two prompts" : "every prompt its own call"));
    }
    // 5. sampled members (the host samples from the member's logits row) and a greedy one with logprobs
    {
      FakeModel fm(4, 8192, 1024, V);
      fm.depth = depth;
      fm.split_layers = 8;
      fm.merge_max = 8;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_merge = 8;
      auto params_of = [](int i) {
        core::SamplingParams p;
        p.max_tokens = 10;
        if (i < 3) p.temperature = 0.9f, p.has_seed = true, p.seed = 100 + static_cast<uint64_t>(i);
        else p.temperature = 0, p.logprobs = 2;
        return p;
      };
      // the reference: the same four requests (seeded) on an
      // engine that gives every prompt its own call
      std::map<uint64_t, std::vector<int32_t>> alone;
      std::map<uint64_t, int> alone_lps;
      {
        FakeModel f1(4, 8192, 1024, V);
        f1.depth = depth;
        f1.split_layers = 8;
        core::BatchConfig b1 = bc;
        b1.prefill_merge = 0;
        core::BatchEngine e1(f1, b1);
        Collect c1;
        for (int i = 0; i < 4; ++i) e1.submit(static_cast<uint64_t>(i + 1), prompt_of(40 + i, 70 + 11 * i), params_of(i));
        e1.start(&c1);
        c1.wait(4);
        e1.stop();
        alone = c1.toks, alone_lps = c1.lps;
      }
      core::BatchEngine e(fm, bc);
      Collect c;
      for (int i = 0; i < 4; ++i) e.submit(static_cast<uint64_t>(i + 1), prompt_of(40 + i, 70 + 11 * i), params_of(i));
      e.start(&c);
      c.wait(4);
      e.stop();
      expect(fm.errors == 0 && fm.merge_stale == 0 && fm.merge_calls == 1 && fm.merge_members == 4 && c.toks == alone && c.lps == alone_lps && c.lps[4] > 0,
             tag + ": three sampled members and a greedy one with logprobs in one call: tokens and logprob counts == the same requests in their own calls (" + std::to_string(fm.errors) + " errors, " +
                 std::to_string(fm.merge_stale) + " stale, " + std::to_string(fm.merge_calls) + " call(s) of " + std::to_string(fm.merge_members) + ", tokens " +
                 (c.toks == alone ? "equal" : "differ") + ", logprobs " + std::to_string(c.lps[4]) + " vs " + std::to_string(alone_lps[4]) + ")");
    }
    // 4. beside a decoding request the merged call runs in parts with decode steps between them; a member cancelled
    // mid-call: the call runs to its end, the other members complete
    {
      FakeModel fm(5, 8192, 1024, V);
      fm.depth = depth;
      fm.split_layers = 64;
      fm.merge_max = 8;
      fm.step_us = 200, fm.verify_us = 200;
      fm.part_us = 1500;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_merge = 8;
      bc.prefill_part_ms = 1e-6;  // every call beside decoding in parts of one layer
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 4000;
      e.submit(1, prompt_of(0, 6), p);
      for (int i = 0; i < 2000 && fm.steps < 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      p.max_tokens = 10;
      for (int i = 0; i < 3; ++i) e.submit(static_cast<uint64_t>(i + 2), prompt_of(30 + i, 60 + 20 * i), p);
      for (int i = 0; i < 5000 && fm.merge_calls < 1; ++i) std::this_thread::sleep_for(std::chrono::microseconds(200));
      std::this_thread::sleep_for(std::chrono::milliseconds(6));  // a few parts in
      e.cancel(3);
      {
        std::unique_lock<std::mutex> lk(c.mu);
        c.cv.wait_for(lk, std::chrono::seconds(20), [&] { return c.fin.count(2) && c.fin.count(4); });
      }
      e.cancel(1);
      e.stop();
      const bool same = c.toks[2] == solo_of(30, 60, 10, depth) && c.toks[4] == solo_of(32, 100, 10, depth);
      expect(fm.errors == 0 && fm.merge_stale == 0 && fm.merge_calls == 1 && fm.merge_members == 3 && fm.steps_between_parts > 0,
             tag + ": three prompts beside a decoding request share one call in parts (" + std::to_string(fm.steps_between_parts) + " decode steps between its parts, " +
                 std::to_string(fm.errors) + " errors)");
      expect(fm.merge_rushed == 1 && fm.split_aborts == 0 && same,
             tag + ": a member cancelled mid-call: the call runs to its end (no abort), the two others' tokens == alone (" + std::to_string(fm.merge_rushed) + " run(s) to the end)");
    }
  }
}

void test_batch_engine() {
  const int V = kVocab;
  const int N = 7;
  const int lens[N] = {1, 3, 16, 17, 40, 95, 300};
  // solo: each request alone on a 1-slot engine
  std::vector<std::vector<int32_t>> solo(N);
  for (int i = 0; i < N; ++i) {
    FakeModel fm(1, 4096, 16, V);
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 20 + i;
    e.submit(1, prompt_of(i, lens[i]), p);
    c.wait(1);
    e.stop();
    solo[static_cast<size_t>(i)] = c.toks[1];
    expect(fm.errors == 0 && c.fin[1] == core::Finish::length && c.toks[1].size() == static_cast<size_t>(20 + i),
           "batch engine: solo request " + std::to_string(i) + " runs to max_tokens");
  }
  // batched: all at once on 4 slots (3 wait, slots are reused)
  {
    FakeModel fm(4, 4096, 16, V);
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < N; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 20 + i;
      e.submit(static_cast<uint64_t>(100 + i), prompt_of(i, lens[i]), p);
    }
    expect(c.wait(N), "batch engine: 7 requests on 4 slots all finish");
    e.stop();
    bool eq = true;
    for (int i = 0; i < N; ++i) eq &= c.toks[static_cast<uint64_t>(100 + i)] == solo[static_cast<size_t>(i)];
    expect(eq, "batch engine: every request batched (4 slots, chunked prefill between decode steps) == alone");
    expect(fm.errors == 0, "batch engine: no slot misuse (rows <= 4, each slot once per step, prefill on begun slots, chunk <= 16)");
    expect(fm.max_batch == 4, "batch engine: decode steps batch 4 sequences");
    const auto st = e.stats();
    expect(st.requests_finished == N && st.running == 0 && st.waiting == 0, "batch engine: stats settle (finished 7, nothing running)");
  }
  // stops: EOS (stop token flagged), ignore_eos, stop_token_ids, cancels, capacity
  {
    FakeModel fm(2, 64, 16, V);
    fm.eos_every = 12;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 30;
    e.submit(1, prompt_of(1, 5), p);  // history hits 12 after 7 generated tokens -> EOS
    c.wait(1);
    expect(c.fin[1] == core::Finish::stop && c.stop_tok[1] && c.toks[1].back() == 0 && c.toks[1].size() == 8,
           "batch engine: EOS stops with the stop token flagged (8 tokens incl. EOS)");
    core::SamplingParams q = p;
    q.ignore_eos = true;
    q.max_tokens = 20;
    e.submit(2, prompt_of(1, 5), q);
    c.wait(2);
    expect(c.fin[2] == core::Finish::length && c.toks[2].size() == 20, "batch engine: ignore_eos runs past the EOS to max_tokens");
    core::SamplingParams r = q;
    r.stop_token_ids = {c.toks[2][3]};
    e.submit(3, prompt_of(1, 5), r);
    c.wait(3);
    expect(c.fin[3] == core::Finish::stop && c.toks[3].size() == 4, "batch engine: stop_token_ids stop at that token");
    core::SamplingParams big = q;
    big.max_tokens = 1000;
    e.submit(4, prompt_of(2, 50), big);  // capacity 64: 50 prompt -> at most 15 generated
    c.wait(4);
    expect(c.fin[4] == core::Finish::length && c.toks[4].size() == 15, "batch engine: the slot's capacity ends the request (length)");
    // a long request holds both slots... cancel it running, and one waiting
    FakeModel fm2(1, 100000, 16, V);
    core::BatchEngine e2(fm2, bc);
    Collect c2;
    e2.start(&c2);
    core::SamplingParams lng = q;
    lng.max_tokens = 1 << 20;
    e2.submit(10, prompt_of(3, 8), lng);
    e2.submit(11, prompt_of(4, 8), lng);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    e2.cancel(11);
    e2.cancel(10);
    expect(c2.wait(2), "batch engine: cancels end a running and a waiting request");
    expect(c2.fin[10] == core::Finish::cancelled && c2.fin[11] == core::Finish::cancelled && c2.toks[11].empty(),
           "batch engine: both cancelled; the waiting one produced nothing");
    core::SamplingParams s2 = p;
    s2.max_tokens = 5;
    e2.submit(12, prompt_of(5, 8), s2);
    expect(c2.wait(3) && c2.toks[12].size() == 5, "batch engine: the slot is free again after the cancel");
    e2.stop();
    e.stop();
    expect(fm.errors == 0 && fm2.errors == 0, "batch engine: no slot misuse in the stop / cancel runs");
  }
  // host path: sampling with a seed is reproducible; logprobs come with every token
  {
    FakeModel fm(2, 4096, 16, V);
    core::BatchConfig bc;
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 1.0f;
    p.has_seed = true;
    p.seed = 77;
    p.max_tokens = 24;
    p.logprobs = 2;
    e.submit(1, prompt_of(6, 9), p);
    e.submit(2, prompt_of(6, 9), p);
    c.wait(2);
    e.stop();
    expect(c.toks[1] == c.toks[2] && c.toks[1].size() == 24, "batch engine: seeded sampling gives the same tokens on two slots");
    expect(c.lps[1] == 24, "batch engine: logprobs for every generated token");
  }
  // speculative steps with a drafter == plain decoding, token for token
  {
    int bad = 0, steps_spec = 0, tokens_spec = 0;
    for (int i = 0; i < N; ++i) {
      FakeModel fm(1, 4096, 16, V);
      fm.depth = 3;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 20 + i;
      e.submit(1, prompt_of(i, lens[i]), p);
      c.wait(1);
      e.stop();
      bad += c.toks[1] != solo[static_cast<size_t>(i)] || fm.errors != 0;
      steps_spec += fm.accepts;
      tokens_spec += static_cast<int>(c.toks[1].size()) - 1;
    }
    expect(bad == 0, "batch engine: MTP-3 speculative steps (a drafter with wrong drafts) == plain greedy, token for token (" + std::to_string(N - bad) + " / " + std::to_string(N) + ")");
    const double tps = steps_spec ? static_cast<double>(tokens_spec) / steps_spec : 0;
    expect(tps > 1.3 && tps < 4.0, "batch engine: speculative steps emit " + std::to_string(tps) + " tokens per step (1 < x <= 4)");
    // EOS inside an accepted run ends the request there; the stats count the drafts
    FakeModel fm(1, 4096, 16, V);
    fm.depth = 3;
    fm.eos_every = 12;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 30;
    e.submit(1, prompt_of(1, 5), p);
    c.wait(1);
    e.stop();
    const auto st = e.stats();
    expect(c.fin[1] == core::Finish::stop && c.stop_tok[1] && c.toks[1].size() == 8 && st.spec_drafts > 0 && st.spec_draft_tokens >= st.spec_accepted_tokens,
           "batch engine: EOS inside a speculative run stops there (8 tokens incl. EOS), spec counters set");
    // sampled requests verify drafts by p(draft): seeded runs reproduce, and the max_tokens cap holds
    FakeModel fs(1, 4096, 16, V);
    fs.depth = 3;
    core::BatchEngine es(fs, bc);
    Collect cs;
    es.start(&cs);
    core::SamplingParams q;
    q.temperature = 1.0f;
    q.has_seed = true;
    q.seed = 5;
    q.max_tokens = 40;
    q.ignore_eos = true;
    es.submit(1, prompt_of(2, 9), q);
    es.submit(2, prompt_of(2, 9), q);
    cs.wait(2);
    es.stop();
    expect(cs.toks[1] == cs.toks[2] && cs.toks[1].size() == 40 && fs.errors == 0, "batch engine: sampled speculative decoding, same seed -> same 40 tokens");
    // the seeded text does not depend on the draft depth its steps got (depth 3 alone, 0 at 5-8 streams, the
    // adaptive pick): the same request with no drafter writes the same tokens
    for (int d : {0, 1}) {
      FakeModel fp(1, 4096, 16, V);
      fp.depth = d;
      core::BatchEngine ep(fp, bc);
      Collect cp;
      ep.start(&cp);
      ep.submit(1, prompt_of(2, 9), q);
      cp.wait(1);
      ep.stop();
      expect(cp.toks[1] == cs.toks[1] && fp.errors == 0,
             "batch engine: a seeded sampled request at draft depth " + std::to_string(d) + " == at depth 3, token for token");
    }
  }
  {
    // adaptive draft depth. Drafts that are never right and a verify whose time grows with its rows: the
    // engine drafts nothing on most steps (1-row verifies; every 16th step still drafts the full depth), output unchanged;
    // with adaptive_depth off every step verifies 4 rows.
    int one[2] = {0, 0}, all[2] = {0, 0};
    bool same = true;
    std::vector<int32_t> out[2];
    for (int ad = 0; ad < 2; ++ad) {
      FakeModel fm(1, 4096, 16, V);
      fm.depth = 3;
      fm.all_wrong = true;
      fm.verify_us = 1500;
      fm.verify_us_row = 600;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.adaptive_depth = ad == 1;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 160;
      p.ignore_eos = true;
      e.submit(1, prompt_of(3, 7), p);
      c.wait(1);
      e.stop();
      same &= c.toks[1].size() == 160 && fm.errors == 0;
      out[ad] = c.toks[1];
      one[ad] = fm.one_row_verifies;
      all[ad] = fm.verifies;
    }
    expect(same && out[0] == out[1] && one[0] == 0 && one[1] * 2 > all[1], "batch engine: adaptive depth with drafts never right: " + std::to_string(one[1]) + " / " + std::to_string(all[1]) +
                                                           " verifies of 1 row (fixed depth: " + std::to_string(one[0]) + " / " + std::to_string(all[0]) + "), 160 tokens");
  }
  {
    // lookup drafting: output that copies its prompt (period 40) and a drafter that is never right:
    // with BatchConfig:lookup the drafts come from the request's own tokens once its scored proposals lead, steps keep
    // up to 8 tokens and the output is the same as without; the drafter's passes of those steps are skipped (depth 0).
    std::vector<int32_t> out[2];
    int verifies[2] = {0, 0};
    uint64_t lk_steps[2] = {0, 0}, lk_acc[2] = {0, 0};
    bool ok = true;
    for (int lk = 0; lk < 2; ++lk) {
      FakeModel fm(1, 4096, 64, V);
      fm.depth = 7;
      fm.rows_max = 8;
      fm.all_wrong = true;
      fm.copy_period = 40;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.adaptive_depth = false;
      bc.lookup = lk ? 4 : 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 200;
      p.ignore_eos = true;
      e.submit(1, prompt_of(5, 40), p);
      c.wait(1);
      const core::Stats st = e.stats();
      e.stop();
      ok &= c.toks[1].size() == 200 && fm.errors == 0;
      out[lk] = c.toks[1];
      verifies[lk] = fm.verifies;
      lk_steps[lk] = st.lookup_steps;
      lk_acc[lk] = st.lookup_accepted_tokens;
    }
    expect(ok && out[0] == out[1] && lk_steps[0] == 0 && lk_steps[1] > 0 && verifies[1] * 3 < verifies[0],
           "batch engine: lookup drafting on copied output: " + std::to_string(verifies[1]) + " verifies (off: " + std::to_string(verifies[0]) +
               "), " + std::to_string(lk_steps[1]) + " lookup steps, " + std::to_string(lk_acc[1]) + " lookup tokens accepted; output identical");
  }
  {
    // adaptive depth up to a whole 8-row verify (depth 7, the DFlash drafter's block). Drafts right except where
    // the position hash says wrong, a verify nearly free per row: the adaptive pick keeps depth 7 (8-row verifies), as the
    // fixed depth does. Output == fixed depth.
    int full[2] = {0, 0}, all[2] = {0, 0};
    bool ok = true;
    std::vector<int32_t> out[2];
    for (int ad = 0; ad < 2; ++ad) {
      FakeModel fm(1, 4096, 16, V);
      fm.depth = 7;
      fm.rows_max = 8;
      fm.verify_us = 1500;
      fm.verify_us_row = 20;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.adaptive_depth = ad == 1;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 200;
      p.ignore_eos = true;
      e.submit(1, prompt_of(3, 7), p);
      c.wait(1);
      e.stop();
      ok &= c.toks[1].size() == 200 && fm.errors == 0;
      out[ad] = c.toks[1];
      full[ad] = fm.full_verifies;
      all[ad] = fm.verifies;
    }
    expect(ok && out[0] == out[1] && full[0] * 10 > all[0] * 9 && full[1] * 10 > all[1] * 9,
           "batch engine: adaptive depth keeps depth 7 (8-row verifies: fixed " + std::to_string(full[0]) + " / " + std::to_string(all[0]) + ", adaptive " +
               std::to_string(full[1]) + " / " + std::to_string(all[1]) + "), output == fixed");
  }
  // speculative steps over several slots (MTP-3 alone, MTP-1 each for two, plain rows for
  // three or four, the drafter following every step) == each request alone, token for token
  for (int slots : {2, 4}) {
    FakeModel fm(slots, 4096, 16, V);
    fm.depth = 3;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < N; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 20 + i;
      e.submit(static_cast<uint64_t>(100 + i), prompt_of(i, lens[i]), p);
    }
    const bool all = c.wait(N);
    e.stop();
    bool eq = all;
    for (int i = 0; i < N; ++i) eq &= c.toks[static_cast<uint64_t>(100 + i)] == solo[static_cast<size_t>(i)];
    expect(eq && fm.errors == 0 && fm.max_spec_rows <= 4 && fm.two_by_two > 0,
           "batch engine: MTP over " + std::to_string(slots) + " slots (7 requests; " + std::to_string(fm.two_by_two) +
               " steps of 2 sequences x 2 rows, at most 4 rows a step) == each alone, token for token");
  }
  // Unequal segments (a depth per request) over 4 / 6 / 8 slots, greedy and sampled, drafts
  // wrong where the position hash says, 7 + 1 requests of other lengths: == each request alone, token for token (greedy);
  // sampled: two requests on one seed alike; the steps run unequal layouts.
  // 12 slots x 16 rows, plain from 5 sequences, the model's layout depth 1 (the 27B's MTP):
  // unequal layouts of 2-row segments only.
  // Two classes (12 slots x 32 rows; some requests draft 1 / 3 more than the one depth where the
  // step's rows are free): the steps run segments of B rows and then B2 > 1 rows.
  for (int var = 0; var < 6; ++var) {
    const int slots = var < 3 ? 4 + 2 * var : 12, deep = var < 4 ? 0 : var == 4 ? 1 : 3;
    FakeModel fm(slots, 4096, 16, V);
    fm.depth = 7;
    fm.rows_max = deep > 0 ? 32 : slots == 12 ? 16 : 8;
    if (slots == 12 && deep == 0) fm.plain_from = 5, fm.lay_depth = 1;
    fm.uneq = true;
    fm.classes = deep > 0, fm.deep = deep;
    fm.verify_us = 1500;
    fm.verify_us_row = deep > 0 ? 0 : 120;  // (two classes: free rows, so the deeper class wins wherever it fits)
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.depth_explore = 5;
    if (deep > 0) bc.depth_hold = 4;  // (the requests are short: the deeper class may start early)
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < N; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 60 + 7 * i;
      e.submit(static_cast<uint64_t>(100 + i), prompt_of(i, lens[i]), p);
    }
    core::SamplingParams q;
    q.temperature = 1.0f;
    q.has_seed = true;
    q.seed = 5;
    q.max_tokens = 50;
    q.ignore_eos = true;
    e.submit(1, prompt_of(3, 9), q);
    e.submit(2, prompt_of(3, 9), q);
    const bool all = c.wait(N + 2);
    e.stop();
    bool eq = all;
    std::vector<std::vector<int32_t>> solo_n;
    for (int i = 0; i < N; ++i) {  // each alone: the plain engine, one slot
      FakeModel f1(1, 4096, 16, V);
      core::BatchConfig b1;
      b1.eos_ids = {0};
      b1.log_interval_s = 0;
      core::BatchEngine e1(f1, b1);
      Collect c1;
      e1.start(&c1);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 60 + 7 * i;
      e1.submit(1, prompt_of(i, lens[i]), p);
      c1.wait(1);
      e1.stop();
      eq &= c.toks[static_cast<uint64_t>(100 + i)] == c1.toks[1];
    }
    // the engine runs the held drafts' segment size with the most tokens per ms (4 slots: equal layouts win)
    expect(eq && fm.errors == 0 && (fm.het_verifies > 0 || slots == 4) && (fm.het2_verifies > 0) == (deep > 0) && c.toks[1] == c.toks[2] && c.toks[1].size() == 50,
           "batch engine: unequal segments over " + std::to_string(slots) + " slots" + (deep > 0 ? ", two classes (+" + std::to_string(deep) + " drafts)" : std::string()) +
               " (" + std::to_string(fm.het_verifies) + " / " + std::to_string(fm.verifies) + " verifies unequal, " + std::to_string(fm.het2_verifies) +
               " of two classes, at most " + std::to_string(fm.max_spec_rows) + " rows, unequal B <= " + std::to_string(fm.max_het_b) +
               ") == each alone, token for token; sampled on one seed alike");
  }
  // StepModel:step_rows_at: 8 slots, 16-row steps from 5 sequences and 8-row steps below, depth 7,
  // one depth for all and unequal segments: no verify over its sequence count's budget, the 16 rows used from 5
  // sequences, every request == alone, token for token
  for (const bool uneq : {false, true}) {
    FakeModel fm(8, 4096, 16, V);
    fm.depth = 7;
    fm.rows_max = 16, fm.rows_small = 8, fm.rows_small_below = 5;
    fm.uneq = uneq;
    fm.verify_us = 1500;
    fm.verify_us_row = 60;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.depth_explore = 5;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    const int reqs = 8;
    for (int i = 0; i < reqs; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 30 + 15 * i;  // the requests end one by one: every sequence count from 8 down to 1
      e.submit(static_cast<uint64_t>(100 + i), prompt_of(i % N, lens[i % N]), p);
    }
    const bool all = c.wait(reqs);
    e.stop();
    bool eq = all;
    for (int i = 0; i < reqs; ++i) {
      FakeModel f1(1, 4096, 16, V);
      core::BatchConfig b1;
      b1.eos_ids = {0};
      b1.log_interval_s = 0;
      core::BatchEngine e1(f1, b1);
      Collect c1;
      e1.start(&c1);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 30 + 15 * i;
      e1.submit(1, prompt_of(i % N, lens[i % N]), p);
      c1.wait(1);
      e1.stop();
      eq &= c.toks[static_cast<uint64_t>(100 + i)] == c1.toks[1];
    }
    expect(eq && fm.errors == 0 && fm.max_rows_small_steps <= 8 && fm.max_rows_large_steps > 8,
           std::string("batch engine: rows per step by sequence count (8 below 5 sequences, 16 from 5; ") + (uneq ? "unequal segments" : "one depth") +
               ": at most " + std::to_string(fm.max_rows_small_steps) + " / " + std::to_string(fm.max_rows_large_steps) +
               " rows) == each alone, token for token");
  }
  {
    // unequal segments with lookup drafting on (copying output, the drafter right where the position hash
    // says, 4 slots; the lookup proposes at the layout's largest depth, for all going requests, as it does at the one
    // depth without unequal segments), the output == each alone
    FakeModel fm(4, 4096, 64, V);
    fm.depth = 7;
    fm.rows_max = 8;
    fm.uneq = true;
    fm.copy_period = 40;
    fm.verify_us = 1500;
    fm.verify_us_row = 120;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 200;
    p.ignore_eos = true;
    for (int i = 0; i < 4; ++i) e.submit(static_cast<uint64_t>(10 + i), prompt_of(5 + i, 40), p);
    c.wait(4);
    const core::Stats st = e.stats();
    e.stop();
    bool eq = fm.errors == 0;
    for (int i = 0; i < 4; ++i) {
      FakeModel f1(1, 4096, 64, V);
      f1.copy_period = 40;
      core::BatchConfig b1;
      b1.eos_ids = {0};
      b1.log_interval_s = 0;
      core::BatchEngine e1(f1, b1);
      Collect c1;
      e1.start(&c1);
      e1.submit(1, prompt_of(5 + i, 40), p);
      c1.wait(1);
      e1.stop();
      eq &= c.toks[static_cast<uint64_t>(10 + i)] == c1.toks[1] && c1.toks[1].size() == 200;
    }
    expect(eq, "batch engine: unequal segments + lookup drafting over 4 slots (" + std::to_string(st.lookup_steps) + " lookup segments, " +
                   std::to_string(fm.het_verifies) + " / " + std::to_string(fm.verifies) + " verifies unequal) == each alone, token for token");
  }
  {
    // sampled, two slots: the same seed on both -> the same tokens (each segment verified by p(draft))
    FakeModel fm(2, 4096, 16, V);
    fm.depth = 3;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams q;
    q.temperature = 1.0f;
    q.has_seed = true;
    q.seed = 11;
    q.max_tokens = 40;
    q.ignore_eos = true;
    e.submit(1, prompt_of(3, 9), q);
    e.submit(2, prompt_of(3, 9), q);
    c.wait(2);
    e.stop();
    expect(c.toks[1] == c.toks[2] && c.toks[1].size() == 40 && fm.errors == 0 && fm.two_by_two > 0,
           "batch engine: sampled MTP on two slots, same seed -> the same 40 tokens");
  }
  // the prefill budget while requests decode, shortest prompt first, prompt logprobs
  {
    const auto long_p = prompt_of(9, 600), mid_p = prompt_of(10, 300);
    std::vector<int32_t> solo_long, solo_mid;
    for (int which = 0; which < 2; ++which) {
      FakeModel fm(1, 8192, 256, V);
      core::BatchConfig bc;
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 10;
      e.submit(1, which ? mid_p : long_p, p);
      c.wait(1);
      e.stop();
      (which ? solo_mid : solo_long) = c.toks[1];
    }
    FakeModel fm(3, 8192, 256, V);
    fm.pf_us_fixed = 2000;  // 2 ms + 0.05 ms per row: a 6 ms stall budget fits 80 rows -> 64
    fm.pf_us_row = 50;
    core::BatchConfig bc;
    bc.log_interval_s = 0;
    bc.prefill_stall_ms = 6;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams dp;
    dp.temperature = 0;
    dp.max_tokens = 1 << 20;
    e.submit(1, prompt_of(8, 5), dp);  // decodes all along
    while (true) {
      {
        std::lock_guard<std::mutex> lk(c.mu);
        if (!c.toks[1].empty()) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 10;
    e.submit(2, long_p, p);
    e.submit(3, mid_p, p);
    c.wait(2);
    e.cancel(1);
    c.wait(3);
    e.stop();
    bool budget = true;
    int long_calls = 0, mid_done_at = -1, long_after_mid = 0;
    for (size_t i = 0; i < fm.pf_log.size(); ++i) {
      const auto &pc = fm.pf_log[i];
      if (pc.slot == 0) continue;
      if (pc.n > 64) std::printf("  pf call slot %d n %d steps %d\n", pc.slot, pc.n, pc.steps_before);
      budget &= pc.n <= 64;
      if (pc.n == 0) budget = false;
    }
    // slots: request 2 and 3 got slots 1 and 2 in admission order
    int pf_long = 0, pf_mid = 0, last_mid = -1, first_long_after = -1;
    for (size_t i = 0; i < fm.pf_log.size(); ++i) {
      const auto &pc = fm.pf_log[i];
      if (pc.slot == 1) pf_long += pc.n;
      if (pc.slot == 2) pf_mid += pc.n, last_mid = static_cast<int>(i);
    }
    for (size_t i = 0; i < fm.pf_log.size(); ++i)
      if (fm.pf_log[i].slot == 1 && static_cast<int>(i) < last_mid) ++long_calls;
    (void)mid_done_at, (void)long_after_mid, (void)first_long_after;
    expect(budget && pf_long == 600 && pf_mid == 300, "batch engine: while a request decodes, prefill calls take <= 64 rows (stall budget)");
    expect(long_calls <= 1, "batch engine: the shorter prompt prefills first (the long one ran " + std::to_string(long_calls) + " calls before it finished)");
    expect(c.toks[2] == solo_long && c.toks[3] == solo_mid, "batch engine: budgeted prefill + decode interleaving == each alone");
    expect(fm.errors == 0, "batch engine: no slot misuse with the prefill budget");
  }
  // prefill calls in parts while a request decodes: decode steps between the parts, the
  // same tokens as whole calls, a cancel mid-call aborts it
  {
    const auto long_p = prompt_of(9, 300), mid_p = prompt_of(10, 40);
    std::vector<int32_t> solo_long, solo_mid;
    for (int which = 0; which < 2; ++which) {
      FakeModel fm(1, 8192, 256, V);
      core::BatchConfig bc;
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 10;
      e.submit(1, which ? mid_p : long_p, p);
      c.wait(1);
      e.stop();
      (which ? solo_mid : solo_long) = c.toks[1];
    }
    FakeModel fm(4, 8192, 256, V);
    fm.split_layers = 8;
    fm.step_us = 2000;
    fm.pf_us_fixed = 20000;  // a call costs 20 ms + 0.1 ms per row: 3+ parts of 10 ms
    fm.pf_us_row = 100;
    core::BatchConfig bc;
    bc.log_interval_s = 0;
    bc.prefill_part_ms = 10;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams dp;
    dp.temperature = 0;
    dp.max_tokens = 1 << 20;
    e.submit(1, prompt_of(8, 5), dp);  // decodes all along
    while (true) {
      {
        std::lock_guard<std::mutex> lk(c.mu);
        if (!c.toks[1].empty()) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 10;
    e.submit(2, long_p, p);
    e.submit(3, mid_p, p);
    c.wait(2);
    const int begins = fm.split_begins;
    e.cancel(1);
    c.wait(3);
    // a cancel while its call is in parts: a long prompt beside a decoding request, cancelled at once
    e.submit(4, prompt_of(11, 5), dp);
    while (true) {
      {
        std::lock_guard<std::mutex> lk(c.mu);
        if (!c.toks[4].empty()) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const int b0 = fm.split_begins;
    e.submit(5, prompt_of(12, 200), p);
    while (fm.split_begins == b0) std::this_thread::sleep_for(std::chrono::microseconds(100));
    e.cancel(5);
    c.wait(4);
    e.cancel(4);
    c.wait(5);
    e.stop();
    std::printf("  split calls %d, parts %d, decode steps between parts %d, aborts %d\n", fm.split_begins, fm.split_parts, fm.steps_between_parts, fm.split_aborts);
    expect(begins > 0 && fm.steps_between_parts > 0, "batch engine: while a request decodes, prefill calls run in parts with decode steps between them");
    expect(c.toks[2] == solo_long && c.toks[3] == solo_mid, "batch engine: prefill calls in parts == each request alone");
    expect(fm.split_aborts == 1 && c.fin.count(5) == 1, "batch engine: a cancel during a call in parts aborts the call");
    expect(fm.errors == 0, "batch engine: no slot misuse with prefill calls in parts");
  }
  {
    FakeModel fm(2, 4096, 16, V);
    core::BatchConfig bc;
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    struct PlSink final : core::EventSink {
      std::mutex mu;
      std::condition_variable cv;
      std::vector<core::TokenLogprob> plp;
      int done = 0;
      void on_events(const core::Event *e, size_t n) override {
        std::lock_guard<std::mutex> lk(mu);
        for (size_t i = 0; i < n; ++i) {
          if (!e[i].prompt_logprobs.empty()) plp = e[i].prompt_logprobs;
          done += e[i].finish != core::Finish::none;
        }
        cv.notify_all();
      }
    } sink;
    e.start(&sink);
    core::SamplingParams p;
    p.temperature = 0;
    p.max_tokens = 3;
    p.prompt_logprobs = 2;
    const auto pr = prompt_of(11, 37);
    e.submit(1, pr, p);
    {
      std::unique_lock<std::mutex> lk(sink.mu);
      sink.cv.wait_for(lk, std::chrono::seconds(10), [&] { return sink.done == 1; });
    }
    e.stop();
    // row j's logits put 8 on the fake's pick after pr[0..j]: token j + 1 scores hit or miss
    const double z = std::log(std::exp(8.0) + (V - 1));
    bool ok = sink.plp.size() == pr.size() && std::isnan(sink.plp[0].logprob);
    for (size_t j = 1; ok && j < pr.size(); ++j) {
      std::vector<int32_t> h(pr.begin(), pr.begin() + static_cast<std::ptrdiff_t>(j));
      const bool hit = fm.pick_after(h) == pr[j];
      const double want = hit ? 8.0 - z : -z;
      ok &= std::fabs(sink.plp[j].logprob - want) < 1e-4 && sink.plp[j].top.size() == 2 && sink.plp[j].top[0].first == fm.pick_after(h) &&
            sink.plp[j].rank == (hit ? 1 : 2);
    }
    expect(ok, "batch engine: prompt logprobs, one per prompt token ([0] NaN), each the row before's log-softmax with rank and top 2");
  }
  // logprobs on the model's side (device_logprobs) == the host path, for prompt logprobs and
  // greedy logprobs, plain steps on 3 slots and speculative steps (2 x 2 rows and MTP-3); with them no
  // logits rows are copied for greedy requests, and prompt logprobs take prefill_each
  {
    struct LpSink final : core::EventSink {
      std::mutex mu;
      std::condition_variable cv;
      std::map<uint64_t, std::vector<int32_t>> toks;
      std::map<uint64_t, std::vector<core::TokenLogprob>> lp, plp;
      int done = 0;
      void on_events(const core::Event *e, size_t n) override {
        std::lock_guard<std::mutex> lk(mu);
        for (size_t i = 0; i < n; ++i) {
          auto &t = toks[e[i].id];
          t.insert(t.end(), e[i].tokens, e[i].tokens + e[i].n);
          auto &l = lp[e[i].id];
          l.insert(l.end(), e[i].logprobs.begin(), e[i].logprobs.end());
          if (!e[i].prompt_logprobs.empty()) plp[e[i].id] = e[i].prompt_logprobs;
          done += e[i].finish != core::Finish::none;
        }
        cv.notify_all();
      }
    };
    auto same_lp = [](const core::TokenLogprob &a, const core::TokenLogprob &b) {
      if (std::isnan(a.logprob) || std::isnan(b.logprob)) return std::isnan(a.logprob) && std::isnan(b.logprob);
      return a.logprob == b.logprob && a.rank == b.rank && a.top == b.top;
    };
    for (const int depth : {0, 3}) {
      LpSink sinks[3];
      int logits_calls[3] = {}, lp_calls[3] = {}, each_calls[3] = {}, errs[3] = {}, rows_calls[3] = {};
      // dl 0: host logprobs; 1: device, a head per 4-row group (prefill_each); 2: device, prompt rows in
      // one call (prefill_row_logprobs)
      for (int dl = 0; dl < 3; ++dl) {
        FakeModel fm(3, 4096, 16, V);
        fm.dev_lp = dl >= 1;
        fm.rows_lp = dl == 2;
        fm.depth = depth;
        core::BatchConfig bc;
        bc.log_interval_s = 0;
        core::BatchEngine e(fm, bc);
        e.start(&sinks[dl]);
        for (int i = 0; i < 5; ++i) {
          core::SamplingParams p;
          p.temperature = 0;
          p.max_tokens = 9 + i;
          p.logprobs = i % 3;  // 0, 1, 2 alternatives
          p.prompt_logprobs = i % 2 ? 2 : -1;
          e.submit(static_cast<uint64_t>(1 + i), prompt_of(30 + i, 5 + 7 * i), p);
        }
        {
          std::unique_lock<std::mutex> lk(sinks[dl].mu);
          sinks[dl].cv.wait_for(lk, std::chrono::seconds(10), [&] { return sinks[dl].done == 5; });
        }
        e.stop();
        logits_calls[dl] = fm.logits_calls;
        lp_calls[dl] = fm.lp_calls;
        each_calls[dl] = fm.each_calls;
        rows_calls[dl] = fm.rows_lp_calls;
        errs[dl] = fm.errors;
      }
      bool ok = sinks[0].done == 5 && sinks[1].done == 5 && sinks[2].done == 5 && sinks[0].toks == sinks[1].toks && sinks[0].toks == sinks[2].toks;
      for (int o = 1; o <= 2; ++o)
        for (uint64_t id = 1; ok && id <= 5; ++id) {
          const auto &a = sinks[0].lp[id], &b = sinks[o].lp[id];
          ok &= a.size() == sinks[0].toks[id].size() && a.size() == b.size();
          for (size_t j = 0; ok && j < a.size(); ++j) ok &= same_lp(a[j], b[j]);
          const auto &pa = sinks[0].plp[id], &pb = sinks[o].plp[id];
          ok &= pa.size() == pb.size();
          for (size_t j = 0; ok && j < pa.size(); ++j) ok &= same_lp(pa[j], pb[j]);
        }
      const std::string tag = depth ? " (speculative steps, MTP-3 / 2 x 2)" : " (plain steps, 3 slots)";
      expect(ok, "batch engine: device logprobs (per 4-row group and all prompt rows in one call) == host logprobs, token for token, prompt and output" + tag);
      expect(logits_calls[1] == 0 && lp_calls[1] > 0 && each_calls[1] >= 2 && logits_calls[0] > 0 && lp_calls[0] == 0 && errs[0] == 0 && errs[1] == 0 &&
                 errs[2] == 0 && rows_calls[1] == 0 && each_calls[2] == 0 && rows_calls[2] >= 2 && logits_calls[2] == 0,
             "batch engine: with device logprobs no logits rows are copied for greedy requests (" + std::to_string(logits_calls[1]) + " vs " +
                 std::to_string(logits_calls[0]) + " copies), prompt logprobs by prefill_each, or by prefill_row_logprobs when the model has it" + tag);
    }
  }
}

// ---------------------------------------------------------------- structured output

// Walks documents through compiled schemas with a tokenizer: valid ones must be offered token by
// token and end complete; invalid ones must be refused at the token that breaks them; along every
// valid walk no state may offer nothing.
void constraint_cases(const text::Tokenizer &tok, int64_t vocab, int32_t eos, const std::string &label, bool timing) {
  const auto cv = text::ConstraintVocab::build(tok, vocab, {eos});
  struct Case {
    const char *schema;  // JSON schema text, or a JSON list for guided_choice (raw)
    bool raw;
    std::vector<const char *> good, bad;
  };
  const std::vector<Case> cases = {
      {R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"ok":{"type":"boolean"}},"required":["name","age"]})", false,
       {R"({"name":"Ann","age":31})", R"({"name": "B\"o\\b \u00e9", "age": -7, "ok": true})", "{\n  \"name\": \"x\",\n  \"age\": 0,\n  \"ok\": false\n}"},
       {R"({"age":31,"name":"Ann"})", R"({"name":"Ann"})", R"({"name":"Ann","age":3.5})", R"({"name":"Ann","age":31,"x":1})", R"({"name":"Ann","age":01})",
        R"({"name":"Ann","age":31,})", R"({"name":Ann})", "\n{\"name\":\"Ann\",\"age\":31}"}},
      {R"({"type":"array","items":{"type":"number"},"minItems":1,"maxItems":3})", false,
       {"[1]", "[1.5e3, -0.25, 7]", "[ 0 , 1E+2 ]"}, {"[]", "[1,2,3,4]", "[1,]", "[.5]", "[1e]", " [1]", "[1] "}},
      {R"({"enum":["red","green",3,null,true]})", false, {R"("red")", R"("green")", "3", "null", "true"}, {R"("blue")", "4", "false", R"("re")", " 3", "3 ", "\n\"red\""}},
      {R"({"type":["string","null"],"maxLength":3})", false, {R"("abc")", "null", R"("")", R"("\n\u0041")"}, {R"("abcd")", "nul", "1"}},
      {R"({"type":"string","minLength":2})", false, {R"("ab")", R"("éé")"}, {R"("a")", R"("")"}},
      {R"({"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}]})", false,
       {R"({"a":1})", R"({"b":"x"})"}, {R"({"c":1})", R"({"a":"x"})", "{}"}},
      {R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"next":{"anyOf":[{"$ref":"#/$defs/node"},{"type":"null"}]}},"required":["v","next"]}},"$ref":"#/$defs/node"})", false,
       {R"({"v":1,"next":{"v":2,"next":null}})"}, {R"({"v":1,"next":{"v":2}})"}},
      {R"({"type":"object"})", false, {"{}", R"({"k":[1,{"x":null}],"s":"t","n":-1.5e-3,"b":false})"}, {"[]", R"({"k":})", R"({"k":1}x)", "\n\n{}", " {}", "{}\n", "{} "}},
      {R"({"type":"object","properties":{"id":{"type":"integer"}},"additionalProperties":{"type":"string"}})", false,
       {R"({"id":1})", R"({"id":1,"x":"y"})", R"({"x":"y"})"}, {R"({"id":1,"x":2})"}},
      {R"({"type":"object","properties":{"t":{"const":"fixed"},"l":{"type":"array","items":{"type":"boolean"}}},"required":["t","l"]})", false,
       {R"({"t":"fixed","l":[true,false,true]})", R"({"t":"fixed","l":[]})"}, {R"({"t":"other","l":[]})", R"({"t":"fixed","l":[null]})"}},
      {R"(["yes","no","not sure","n"])", true, {"yes", "no", "not sure", "n"}, {"maybe", " yes", "yes!", "nope", "yes ", "\nno"}},
  };
  int good = 0, good_total = 0, bad = 0, bad_total = 0, computed = 0;
  std::string first_fail;
  for (const auto &c : cases) {
    json::Value s;
    json::parse(c.schema, s);
    std::string err;
    const auto spec = text::compile_json_constraint(s, c.raw, cv, -1, err);
    if (!spec) {
      first_fail = std::string("compile ") + c.schema + ": " + err;
      continue;
    }
    for (const char *d : c.good) {
      auto ids = tok.encode(d);
      const auto r = text::walk_constraint(*spec, ids);
      ++good_total;
      // every state on the walk offers something (no dead end) and EOS at the end
      bool live = true;
      {
        auto k = spec->start();
        for (int32_t t : ids) {
          const uint64_t *m = k->allowed();
          bool any = false;
          for (int64_t w = 0; m && w < (vocab + 63) / 64 && !any; ++w) any = m[w] != 0;
          live &= m && any;
          k->accept(t);
        }
        const uint64_t *m = k->allowed();
        live &= m && ((m[eos >> 6] >> (eos & 63)) & 1);
      }
      if (r.first_rejected < 0 && r.complete && live) ++good;
      else if (first_fail.empty())
        first_fail = std::string("valid ") + d + " under " + c.schema + ": rejected at token " + std::to_string(r.first_rejected) + (r.complete ? "" : ", not complete") +
                     (live ? "" : ", a dead state or no EOS at the end");
      computed += r.masks_computed;
    }
    for (const char *d : c.bad) {
      const auto r = text::walk_constraint(*spec, tok.encode(d));
      ++bad_total;
      if (r.first_rejected >= 0 || !r.complete) ++bad;
      else if (first_fail.empty()) first_fail = std::string("invalid ") + d + " under " + c.schema + " was accepted";
    }
  }
  expect(good == good_total, label + ": every valid document is offered token by token, ends complete, no dead state (" + std::to_string(good) + " / " +
                                 std::to_string(good_total) + ")" + (first_fail.empty() ? "" : "; " + first_fail));
  expect(bad == bad_total, label + ": every invalid document is refused (" + std::to_string(bad) + " / " + std::to_string(bad_total) + ")");
  {
    json::Value s;
    json::parse(R"({"type":"string","pattern":"a+"})", s);
    std::string err;
    expect(!text::compile_json_constraint(s, false, cv, -1, err) && err.find("pattern") != std::string::npos, label + ": pattern is refused with a reason");
  }
  {  // whitespace runs are capped (a greedy model cannot stall on spaces)
    json::Value s;
    json::parse(R"({"type":"object"})", s);
    std::string err;
    const auto spec = text::compile_json_constraint(s, false, cv, -1, err);
    expect(text::walk_constraint(*spec, tok.encode("{" + std::string(32, ' ') + "}")).first_rejected < 0 &&
               text::walk_constraint(*spec, tok.encode("{" + std::string(40, ' ') + "}")).first_rejected >= 0,
           label + ": 32 whitespace bytes in a run pass, 40 are refused");
  }
  {  // the constraint starts after the reasoning end token
    json::Value s;
    json::parse(R"(["a","b"])", s);
    std::string err;
    const int32_t think_end = tok.token_id("</think>");
    const auto spec = text::compile_json_constraint(s, true, cv, think_end, err);
    std::vector<int32_t> ids = tok.encode("free text ");
    ids.push_back(think_end);
    const auto tail = tok.encode("b");
    ids.insert(ids.end(), tail.begin(), tail.end());
    const auto r = text::walk_constraint(*spec, ids);
    std::vector<int32_t> bad_ids = ids;
    bad_ids.back() = tok.encode("c")[0];
    expect(think_end >= 0 && r.first_rejected < 0 && r.complete && text::walk_constraint(*spec, bad_ids).first_rejected >= 0,
           label + ": a thinking chat is free until </think>, then constrained");
  }
  {  // no whitespace before the root value after </think>:
     // the first mask after </think> offers "{" but not "\n\n" / "\n" / " "
    json::Value s;
    json::parse(R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})", s);
    std::string err;
    const int32_t think_end = tok.token_id("</think>");
    const auto spec = text::compile_json_constraint(s, false, cv, think_end, err);
    std::vector<int32_t> ids = tok.encode("let me think");
    ids.push_back(think_end);
    auto k = spec->start();
    for (int32_t t : ids) k->accept(t);
    const uint64_t *m = k->allowed();
    auto offered = [&](const std::string &text) {
      const auto t = tok.encode(text);
      return m && t.size() == 1 && ((m[t[0] >> 6] >> (t[0] & 63)) & 1);
    };
    std::vector<int32_t> good = ids, bad = ids;
    for (int32_t t : tok.encode(R"({"city": "Paris"})")) good.push_back(t);
    for (int32_t t : tok.encode("\n\n{\"city\": \"Paris\"}")) bad.push_back(t);
    const auto rg = text::walk_constraint(*spec, good), rb = text::walk_constraint(*spec, bad);
    expect(offered("{") && !offered("\n\n") && !offered("\n") && !offered(" ") && rg.first_rejected < 0 && rg.complete && rb.first_rejected >= 0,
           label + ": after </think> the JSON starts at once (\"{\" offered, no whitespace before the root value)");
  }
  if (timing) {
    // mask cost on this vocabulary: a fresh schema, every state computed once, then cached
    json::Value s;
    json::parse(R"({"type":"object","properties":{"title":{"type":"string"},"year":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},"required":["title","year","tags"]})", s);
    std::string err;
    const auto spec = text::compile_json_constraint(s, false, cv, -1, err);
    const auto ids = tok.encode(R"({"title": "The quick brown fox jumps over the lazy dog", "year": 1999, "tags": ["animals", "classic", "pangram"]})");
    const auto t0 = std::chrono::steady_clock::now();
    const auto r1 = text::walk_constraint(*spec, ids);
    const double cold = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    const auto r2 = text::walk_constraint(*spec, ids);
    const double warm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    std::printf("%s: %zu-token document: %d states computed in %.1f ms (%.2f ms each); again, cached: %.2f ms (%d computed)\n", label.c_str(), ids.size(),
                r1.masks_computed, cold, r1.masks_computed ? cold / r1.masks_computed : 0.0, warm, r2.masks_computed);
  }
  (void)computed;
}

void test_constraint(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  constraint_cases(tok, kVocab, tok.token_id("<|im_end|>"), "structured output (byte vocab)", false);
  const char *home = std::getenv("HOME");
  const std::string real = home ? std::string(home) + "/models/qwen38-w4a16" : "";
  text::Tokenizer qt;
  if (!real.empty() && std::filesystem::exists(real + "/tokenizer.json") && qt.load(real, err))
    constraint_cases(qt, 248320, qt.token_id("<|im_end|>"), "structured output (qwen38 vocab)", true);
  else
    std::printf("structured output: no ~/models/qwen38-w4a16 tokenizer, real-vocab cases skipped\n");
}

// ---------------------------------------------------------------- server end to end

struct Running {
  std::unique_ptr<xenithium::server::Server> srv;
  std::thread th;
  std::string base;
  ~Running() {
    if (srv) srv->stop();
    if (th.joinable()) th.join();
  }
};

void start_server(Running &r, core::Engine &e, const text::Tokenizer &tok, const text::ChatFormat *chat, int max_queue = 0,
                  std::vector<int32_t> eos = {}, int prepare_threads = 2, xenithium::server::MediaHook *media = nullptr, int64_t vocab = 0,
                  const xenithium::util::FetchOptions *fetch = nullptr, const std::function<void(xenithium::server::Config &)> &more = {}) {
  xenithium::server::Config cfg;
  if (fetch) cfg.fetch = *fetch;
  if (more) more(cfg);  // what GET /v1/models reports beyond the engine: cards, build, launch
  cfg.prepare_threads = prepare_threads;
  cfg.media = media;
  cfg.vocab = vocab;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  cfg.max_queue = max_queue;
  cfg.eos_ids = std::move(eos);
  r.srv = std::make_unique<xenithium::server::Server>(cfg, e, tok, chat);
  std::string err;
  if (!r.srv->listen(err)) {
    std::printf("listen: %s\n", err.c_str());
    std::exit(1);
  }
  e.start(r.srv->sink());
  r.base = "http://127.0.0.1:" + std::to_string(r.srv->port());
  r.th = std::thread([&r] { r.srv->run(); });
}

json::Value post(const std::string &url, const std::string &body, int *status = nullptr) {
  const auto res = xenithium::http::request("POST", url, body, 30);
  if (status) *status = res.status;
  json::Value v;
  json::parse(res.body, v);
  return v;
}

// SSE body -> the data payloads (without [DONE]) and whether [DONE] came.
std::vector<json::Value> sse(const std::string &body, bool &done) {
  std::vector<json::Value> out;
  done = false;
  size_t p = 0;
  while ((p = body.find("data: ", p)) != std::string::npos) {
    const size_t e = body.find("\n\n", p);
    const std::string d = body.substr(p + 6, e - p - 6);
    p = e;
    if (d == "[DONE]") {
      done = true;
      continue;
    }
    json::Value v;
    if (json::parse(d, v)) out.push_back(v);
  }
  return out;
}

// A raw keep-alive connection: send bytes, read until the server closes (or the timeout).
int raw_connect(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port));
  a.sin_addr.s_addr = htonl(0x7f000001);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0) {
    ::close(fd);
    return -1;
  }
  timeval tv{20, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  return fd;
}
std::string raw_post(const std::string &path, const std::string &body, bool close) {
  return "POST " + path + " HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
         (close ? "\r\nConnection: close" : "") + "\r\n\r\n" + body;
}
void raw_send(int fd, const std::string &s) {
  for (size_t off = 0; off < s.size();) {
    const ssize_t w = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
    if (w <= 0) return;
    off += static_cast<size_t>(w);
  }
}
std::string raw_read_all(int fd) {
  std::string out;
  char b[65536];
  while (true) {
    const ssize_t r = ::recv(fd, b, sizeof b, 0);
    if (r <= 0) break;
    out.append(b, static_cast<size_t>(r));
  }
  return out;
}
std::vector<int> statuses(const std::string &s) {
  std::vector<int> v;
  for (size_t p = 0; (p = s.find("HTTP/1.1 ", p)) != std::string::npos; p += 9) v.push_back(std::atoi(s.c_str() + p + 9));
  return v;
}

// Generation requests are parsed, rendered and tokenized on prepare threads.
// A request whose preparation is slow (a multi-MB prompt) must not stall the loop: /health answers
// while it is prepared (with prepare_threads 0 it waits behind it); pipelined requests on one
// connection keep their order; a client that leaves mid-preparation is dropped cleanly.
void test_prepare(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  // ~3 MB of varied words: a slow tokenize on the test vocabulary; over max_model_len, so the answer
  // is a 400 after the full preparation (no engine time)
  std::string big;
  {
    std::mt19937 rng(7);
    while (big.size() < 3u << 20) {
      const int len = 2 + static_cast<int>(rng() % 9);
      for (int i = 0; i < len; ++i) big += static_cast<char>('a' + rng() % 26);
      big += rng() % 13 == 0 ? ". " : " ";
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  const size_t ntok = tok.encode(big).size();
  const double tok_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::string big_body = "{\"prompt\":\"" + big + "\",\"max_tokens\":4}";
  const std::string small_body = "{\"prompt\":\"hi\",\"max_tokens\":5,\"temperature\":0}";
  double health_ms[2] = {}, big_ms[2] = {};
  int big_status[2] = {};
  for (const int threads : {0, 2}) {
    core::StubConfig sc;
    sc.model.name = "test-model";
    sc.model.max_model_len = 4096;
    sc.script = tok.encode("Let me think.</think>The answer is 4.");
    sc.eos_ids = {tok.token_id("<|im_end|>")};
    sc.step_ms = 1;
    sc.slots = 2;
    core::StubEngine stub(sc);
    Running r;
    start_server(r, stub, tok, chat.get(), 0, {}, threads);
    const int k = threads ? 1 : 0;
    std::thread th([&] {
      const auto a = std::chrono::steady_clock::now();
      big_status[k] = xenithium::http::request("POST", r.base + "/v1/completions", big_body, 60).status;
      big_ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(tok_ms / 4) + 30));  // the body is in, preparation under way
    const auto a = std::chrono::steady_clock::now();
    const int hs = xenithium::http::request("GET", r.base + "/health", "", 60).status;
    health_ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    th.join();
    if (hs != 200) health_ms[k] = 1e9;
    if (threads) {
      // pipelined: a slow request, then a fast one, then one with Connection: close, in one write
      const int fd = raw_connect(r.srv->port());
      raw_send(fd, raw_post("/v1/completions", big_body, false) + raw_post("/v1/completions", small_body, false) +
                       raw_post("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}],\"max_tokens\":8}", true));
      const auto got = raw_read_all(fd);
      ::close(fd);
      const auto st = statuses(got);
      expect(st == std::vector<int>({400, 200, 200}) && got.find("maximum context length") < got.find("The answer") ,
             "server: pipelined requests answered in order across the prepare threads (400 slow, 200, 200 + close; got " + std::to_string(st.size()) + ")");
      // clients that leave while their request is prepared: dropped, the server keeps serving
      for (int i = 0; i < 4; ++i) {
        const int f = raw_connect(r.srv->port());
        raw_send(f, raw_post("/v1/completions", big_body, false));
        ::close(f);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(tok_ms) + 50));
      int st2 = 0;
      const auto v = post(r.base + "/v1/completions", small_body, &st2);
      expect(st2 == 200 && r.srv->live_generations() == 0, "server: clients gone mid-preparation are dropped; the next request answers 200");
    }
  }
  std::printf("server: a %zu-token prompt (%.1f MB) tokenizes in %.0f ms; /health during its preparation %.1f ms with prepare threads, %.1f ms on the loop thread\n",
              ntok, big.size() / 1048576.0, tok_ms, health_ms[1], health_ms[0]);
  expect(big_status[0] == 400 && big_status[1] == 400, "server: the multi-MB prompt is prepared and refused (over max_model_len) in both modes");
  expect(health_ms[1] < 50 && health_ms[1] < tok_ms / 4 && health_ms[0] > tok_ms / 3,
         "server: /health is not stalled by a slow preparation with prepare threads (" + std::to_string(health_ms[1]) + " ms vs " + std::to_string(health_ms[0]) +
             " ms on the loop thread)");
}

// the server marks a chat request for the prefix cache at its last message header (the generation
// prompt's <|im_start|>); a completions request has no mark. An engine that records what it is given, over the stub.
struct RecEngine final : core::Engine {
  explicit RecEngine(core::StubEngine &s) : stub(s) {}
  core::StubEngine &stub;
  std::mutex mu;
  std::vector<std::pair<std::vector<int32_t>, int32_t>> got;  // (prompt, prefix_mark)
  void start(core::EventSink *sink) override { stub.start(sink); }
  void stop() override { stub.stop(); }
  void submit(uint64_t id, std::vector<int32_t> prompt, const core::SamplingParams &p) override {
    {
      std::lock_guard<std::mutex> lk(mu);
      got.push_back({prompt, p.prefix_mark});
    }
    stub.submit(id, std::move(prompt), p);
  }
  void cancel(uint64_t id) override { stub.cancel(id); }
  core::Stats stats() const override { return stub.stats(); }
  const core::ModelInfo &model() const override { return stub.model(); }
};

// A local http server for media URL cases: GET path -> body (404 otherwise), one connection at a time.
class FileServer {
 public:
  explicit FileServer(std::map<std::string, std::string> files) : files_(std::move(files)) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ::bind(fd_, reinterpret_cast<sockaddr *>(&a), sizeof a);
    ::listen(fd_, 16);
    socklen_t len = sizeof a;
    getsockname(fd_, reinterpret_cast<sockaddr *>(&a), &len);
    port_ = ntohs(a.sin_port);
    th_ = std::thread([this] {
      while (!stop_) {
        pollfd p{fd_, POLLIN, 0};
        if (::poll(&p, 1, 50) <= 0) continue;
        const int c = ::accept(fd_, nullptr, nullptr);
        if (c < 0) continue;
        std::string req;
        char buf[4096];
        while (req.find("\r\n\r\n") == std::string::npos) {
          const ssize_t n = ::recv(c, buf, sizeof buf, 0);
          if (n <= 0) break;
          req.append(buf, static_cast<size_t>(n));
        }
        const size_t sp = req.find(' ');
        const std::string path = sp == std::string::npos ? "" : req.substr(sp + 1, req.find(' ', sp + 1) - sp - 1);
        const auto it = files_.find(path);
        const std::string out = it == files_.end() ? std::string("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                                                   : "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(it->second.size()) + "\r\n\r\n" + it->second;
        for (size_t off = 0; off < out.size();) {
          const ssize_t n = ::send(c, out.data() + off, out.size() - off, MSG_NOSIGNAL);
          if (n <= 0) break;
          off += static_cast<size_t>(n);
        }
        ::close(c);
      }
    });
  }
  ~FileServer() {
    stop_ = true;
    th_.join();
    ::close(fd_);
  }
  int port() const { return port_; }

 private:
  std::map<std::string, std::string> files_;
  int fd_ = -1, port_ = 0;
  std::atomic<bool> stop_{false};
  std::thread th_;
};

// Image parts. A fake MediaHook hands out id ranges (the real one: arch:VisionStore through serve:
// make_vision_hook); the server must put each image's ids where the template's image placeholder is, in order, take
// image_url (object or string) and input_image parts, refuse what the hook cannot take or a server without image input.
struct FakeMedia final : xenithium::server::MediaHook {
  int32_t pad = -1;
  int64_t next = 300000;
  std::vector<std::pair<int, int>> sizes;  // (w, h) of the decoded images, in order
  bool add_image(const xenithium::util::Image &img, Tokens &out, int &status, std::string &err) override {
    if (img.w > 4000) {
      status = 400, err = "too big";
      return false;
    }
    sizes.push_back({img.w, img.h});
    out.base = next, out.count = 5 + static_cast<int>(sizes.size());  // 6, 7...
    out.lease = std::make_shared<int>(0);
    next += 1000;
    return true;
  }
  int32_t image_token() const override { return pad; }
  int64_t max_pixels() const override { return 1 << 20; }
  // a video = 2 frames of 3 ids each, with their timestamps
  int32_t vpad = -1, vs = -1, ve = -1;
  size_t video_bytes = 0;
  bool add_video(const std::string &bytes, std::vector<Tokens> &frames, std::vector<std::string> &stamps, int &status, std::string &err) override {
    if (bytes.size() < 4) {
      status = 400, err = "not a video";
      return false;
    }
    video_bytes = bytes.size();
    for (int f = 0; f < 2; ++f) {
      Tokens t;
      t.base = next, t.count = 3;
      t.lease = std::make_shared<int>(0);
      next += 1000;
      frames.push_back(std::move(t));
      stamps.push_back(f == 0 ? "<0.5 seconds>" : "<1.5 seconds>");
    }
    return true;
  }
  int32_t video_token() const override { return vpad; }
  int32_t vision_start() const override { return vs; }
  int32_t vision_end() const override { return ve; }
};

void test_images(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 8192;
  sc.script = tok.encode("A red square.");
  sc.eos_ids = {tok.token_id("<|im_end|>")};
  sc.step_ms = 1;
  sc.slots = 2;
  const std::string png = [] {
    std::ifstream f(std::string(XENITHIUM_SOURCE_DIR) + "/tests/data/vision/c_rgb.png", std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }();
  const std::string data = "data:image/png;base64," + xenithium::util::base64_encode(png);
  const std::string img_obj = "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + data + "\"}}";
  const std::string img_in = "{\"type\":\"input_image\",\"image_url\":\"" + data + "\"}";
  auto body = [](const std::string &content, const std::string &extra = "") {
    return "{\"messages\":[{\"role\":\"user\",\"content\":[" + content + "]}],\"max_tokens\":4" + extra + "}";
  };
  {  // no image input: 400 naming the switch
    core::StubEngine stub(sc);
    Running r;
    start_server(r, stub, tok, chat.get());
    int st = 0;
    const auto v = post(r.base + "/v1/chat/completions", body(img_obj + ",{\"type\":\"text\",\"text\":\"hi\"}"), &st);
    const json::Value *e = v.get("error");
    const std::string *msg = e ? e->get_string("message") : nullptr;
    expect(st == 400 && msg && msg->find("XEN_VISION") != std::string::npos, "server: image parts without image input get 400 naming XEN_VISION (" + std::to_string(st) + ")");
  }
  core::StubEngine stub(sc);
  RecEngine rec(stub);
  FakeMedia media;
  media.pad = tok.token_id("<|image_pad|>");
  const int32_t vs = tok.token_id("<|vision_start|>"), ve = tok.token_id("<|vision_end|>");
  Running r;
  start_server(r, rec, tok, chat.get(), 0, {}, 2, &media, kVocab);
  int st = 0;
  const auto v2 = post(r.base + "/v1/chat/completions", body("{\"type\":\"text\",\"text\":\"Compare\"}," + img_obj + "," + img_in), &st);
  if (st != 200) std::printf("  two images: %s\n", json::dump(v2).c_str());
  std::vector<int32_t> got;
  {
    std::lock_guard<std::mutex> lk(rec.mu);
    if (!rec.got.empty()) got = rec.got.back().first;
  }
  // expect <|vision_start|> 300000..300005 <|vision_end|> <|vision_start|> 301000..301006 <|vision_end|>, no pad left
  bool ok = st == 200 && media.pad >= 0 && media.sizes.size() == 2 && media.sizes[0].first > 0 && std::count(got.begin(), got.end(), media.pad) == 0;
  int found = 0;
  for (size_t i = 0; ok && i < got.size(); ++i) {
    if (got[i] != vs) continue;
    const int64_t base = found == 0 ? 300000 : 301000;
    const int n = found == 0 ? 6 : 7;
    for (int k = 0; k < n; ++k) ok = ok && i + 1 + k < got.size() && got[i + 1 + k] == base + k;
    ok = ok && i + 1 + n < got.size() && got[i + 1 + n] == ve;
    ++found;
  }
  expect(ok && found == 2, "server: two image parts (image_url object, input_image string) -> each image's ids between its vision markers, in order (" +
                               std::to_string(st) + ", " + std::to_string(found) + " found, " + std::to_string(media.sizes.size()) + " decoded)");
  int s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  post(r.base + "/v1/chat/completions", body("{\"type\":\"image_url\",\"image_url\":{\"url\":\"file:///tmp/a.png\"}}"), &s1);
  post(r.base + "/v1/chat/completions", body("{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}"), &s2);
  post(r.base + "/v1/chat/completions",
       "{\"messages\":[{\"role\":\"system\",\"content\":[" + img_obj + "]},{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}", &s3);
  const auto lp = post(r.base + "/v1/chat/completions", body(img_obj, ",\"prompt_logprobs\":0"), &s4);
  const json::Value *pl = lp.get("prompt_logprobs");
  bool pad_listed = false;
  if (pl && pl->is_array())
    for (const auto &e : pl->arr)
      if (e.is_object())
        for (const auto &kv : e.obj)
          if (kv.first == std::to_string(media.pad)) pad_listed = true;
  expect(s1 == 400 && s2 == 400 && s3 == 400, "server: a file: image URL, undecodable data and a system-message image get 400 (" + std::to_string(s1) + " / " +
                                                  std::to_string(s2) + " / " + std::to_string(s3) + ")");
  expect(s4 == 200 && pad_listed, "server: prompt logprobs with an image list the image token at its positions (" + std::to_string(s4) + ")");

  {  // /v1/responses with input_text + input_image parts
    int rs = 0;
    const size_t n0 = media.sizes.size();
    post(r.base + "/v1/responses", "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"What is it?\"}," + img_in + "]}],\"max_output_tokens\":4}", &rs);
    std::vector<int32_t> g;
    {
      std::lock_guard<std::mutex> lk(rec.mu);
      if (!rec.got.empty()) g = rec.got.back().first;
    }
    bool has = false;
    for (size_t i = 0; i + 1 < g.size(); ++i) has |= g[i] == vs && g[i + 1] >= 300000;
    expect(rs == 200 && media.sizes.size() == n0 + 1 && has, "server: /v1/responses takes input_image parts (" + std::to_string(rs) + ")");
  }
  {  // a video_url part -> per frame: timestamp tokens, <|vision_start|>, its ids, <|vision_end|>, inside the template's markers
    media.vpad = tok.token_id("<|video_pad|>"), media.vs = vs, media.ve = ve;
    int st2 = 0;
    const std::string vid = "data:video/mp4;base64," + xenithium::util::base64_encode("fake video bytes");
    const auto v = post(r.base + "/v1/chat/completions", body("{\"type\":\"text\",\"text\":\"What happens?\"},{\"type\":\"video_url\",\"video_url\":{\"url\":\"" + vid + "\"}}"), &st2);
    if (st2 != 200) std::printf("  video: %s\n", json::dump(v).c_str());
    std::vector<int32_t> g;
    {
      std::lock_guard<std::mutex> lk(rec.mu);
      if (!rec.got.empty()) g = rec.got.back().first;
    }
    std::vector<int32_t> want = {vs};
    const int64_t b0 = media.next - 2000;
    for (int f = 0; f < 2; ++f) {
      for (int32_t t : tok.encode(f == 0 ? "<0.5 seconds>" : "<1.5 seconds>")) want.push_back(t);
      want.push_back(vs);
      for (int i = 0; i < 3; ++i) want.push_back(static_cast<int32_t>(b0 + f * 1000 + i));
      want.push_back(ve);
    }
    want.push_back(ve);
    const bool has = std::search(g.begin(), g.end(), want.begin(), want.end()) != g.end();
    expect(st2 == 200 && has && media.video_bytes == 16 && std::count(g.begin(), g.end(), media.vpad) == 0,
           "server: a video_url part -> the frames' timestamps and ids between the template's vision markers (" + std::to_string(st2) + ")");
    int st3 = 0;
    post(r.base + "/v1/chat/completions", body("{\"type\":\"video_url\",\"video_url\":{\"url\":\"data:video/mp4;base64,AAAA\"}}"), &st3);
    expect(st3 == 400, "server: a video the hook refuses gets 400 (" + std::to_string(st3) + ")");
  }
  // http(s) image URLs, fetched by the server (a local file server; the fetcher's own gate is test-fetch)
  FileServer files({{"/c.png", png}});
  const std::string url = "http://127.0.0.1:" + std::to_string(files.port());
  const size_t before = media.sizes.size();
  int u1 = 0, u2 = 0, u3 = 0;
  post(r.base + "/v1/chat/completions", body("{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + url + "/c.png\"}}"), &u1);
  const bool same = media.sizes.size() == before + 1 && media.sizes.back() == media.sizes.front();
  const auto e2 = post(r.base + "/v1/chat/completions", body("{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + url + "/missing.png\"}}"), &u2);
  const json::Value *ee = e2.get("error");
  const std::string *em = ee ? ee->get_string("message") : nullptr;
  expect(u1 == 200 && same, "server: an http image URL is fetched and decoded like its data: URL (" + std::to_string(u1) + ")");
  expect(u2 == 400 && em && em->find("HTTP 404") != std::string::npos, "server: a missing http image gets 400 with the status (" + std::to_string(u2) + ")");
  {
    core::StubEngine stub2(sc);
    FakeMedia media2;
    media2.pad = media.pad;
    xenithium::util::FetchOptions off;
    off.enabled = false;
    Running r2;
    start_server(r2, stub2, tok, chat.get(), 0, {}, 2, &media2, kVocab, &off);
    const auto e3 = post(r2.base + "/v1/chat/completions", body("{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + url + "/c.png\"}}"), &u3);
    const json::Value *e3e = e3.get("error");
    const std::string *e3m = e3e ? e3e->get_string("message") : nullptr;
    expect(u3 == 400 && e3m && e3m->find("XEN_MEDIA_URLS") != std::string::npos, "server: --media-urls 0 refuses http URLs naming the switch (" + std::to_string(u3) + ")");
  }
}


// POST /v1/responses (stateless, over the chat path). The input -> messages translation is checked on the
// rendered prompt (RecEngine), the output items and the streamed events on the stub's scripted reply.
std::vector<std::pair<std::string, json::Value>> sse_events(const std::string &body) {
  std::vector<std::pair<std::string, json::Value>> out;
  size_t p = 0;
  while ((p = body.find("event: ", p)) != std::string::npos) {
    const size_t nl = body.find('\n', p);
    const std::string type = body.substr(p + 7, nl - p - 7);
    const size_t d = body.find("data: ", nl);
    const size_t e = body.find("\n\n", d);
    json::Value v;
    json::parse(body.substr(d + 6, e - d - 6), v);
    out.emplace_back(type, v);
    p = e;
  }
  return out;
}

void test_responses(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  const std::string script = "Let me think.</think>The answer is 4.";
  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 4096;
  sc.script = tok.encode(script);
  sc.script.push_back(tok.token_id("<|im_end|>"));
  sc.eos_ids = {tok.token_id("<|im_end|>")};
  sc.step_ms = 1;
  sc.tokens_per_step = 2;
  sc.slots = 2;
  core::StubEngine stub(sc);
  RecEngine rec(stub);
  Running r;
  start_server(r, rec, tok, chat.get());
  const int n_script = static_cast<int>(tok.encode(script).size());
  auto last_prompt = [&] {
    std::lock_guard<std::mutex> lk(rec.mu);
    std::string s;
    if (!rec.got.empty())
      for (int32_t t : rec.got.back().first) tok.decode_piece(t, s, false);
    return s;
  };
  {  // string input + instructions, non-streaming
    int st = 0;
    const auto v = post(r.base + "/v1/responses", "{\"input\":\"2+2?\",\"instructions\":\"Be brief.\",\"max_output_tokens\":64,\"temperature\":0}", &st);
    const std::string prompt = last_prompt();
    const json::Value *out = v.get_array("output");
    const std::string *id = v.get_string("id"), *obj = v.get_string("object"), *status = v.get_string("status");
    std::string reasoning, text, types;
    if (out)
      for (const auto &it : out->arr) {
        types += (types.empty() ? "" : ",") + (it.get_string("type") ? *it.get_string("type") : std::string("?"));
        if (const json::Value *c = it.get_array("content"); c && !c->arr.empty()) {
          if (const std::string *t = c->arr[0].get_string("text")) (*it.get_string("type") == "reasoning" ? reasoning : text) = *t;
        }
      }
    const json::Value *u = v.get_object("usage");
    expect(st == 200 && id && id->rfind("resp_", 0) == 0 && obj && *obj == "response" && status && *status == "completed" &&
               types == "reasoning,message" && reasoning == "Let me think." && text == "The answer is 4." && u && u->get("input_tokens") &&
               u->get("output_tokens") && u->get("output_tokens")->num == n_script + 1 && u->get_object("output_tokens_details") &&
               u->get_object("output_tokens_details")->get("reasoning_tokens")->num > 0,
           "server: /v1/responses: reasoning item + message, usage with reasoning tokens (" + std::to_string(st) + ", " + types + ", \"" + reasoning +
               "\" / \"" + text + "\")");
    expect(prompt.rfind("<|im_start|>system\n", 0) == 0 && prompt.find("Be brief.<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n") != std::string::npos,
           "server: /v1/responses: instructions -> system message, input string -> user message (" + prompt + ")");
    const std::string *ins = v.get_string("instructions");
    expect(ins && *ins == "Be brief." && v.get("max_output_tokens") && v.get("max_output_tokens")->num == 64 && v.get("tools") &&
               v.get("tools")->is_array() && v.get_string("tool_choice") && *v.get_string("tool_choice") == "auto",
           "server: /v1/responses: the response echoes instructions, max_output_tokens, tools, tool_choice");
  }
  {  // the token limit: incomplete
    int st = 0;
    const auto v = post(r.base + "/v1/responses", "{\"input\":\"x\",\"max_output_tokens\":3}", &st);
    const std::string *status = v.get_string("status");
    const json::Value *inc = v.get_object("incomplete_details");
    expect(st == 200 && status && *status == "incomplete" && inc && inc->get_string("reason") && *inc->get_string("reason") == "max_output_tokens",
           "server: /v1/responses: max_output_tokens reached -> status incomplete, reason max_output_tokens");
  }
  {  // items: messages with input_text parts, a function call and its output, an assistant message; function tools
    int st = 0;
    const std::string body =
        "{\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"Weather in Paris?\"}]},"
        "{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"call_1\",\"output\":\"sunny, 21 C\"},"
        "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"It is sunny.\"}]},"
        "{\"role\":\"user\",\"content\":\"thanks\"}],"
        "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}],"
        "\"max_output_tokens\":4}";
    post(r.base + "/v1/responses", body, &st);
    const std::string prompt = last_prompt();
    const bool tools = prompt.find("\"name\": \"get_weather\"") != std::string::npos;
    const bool call = prompt.find("<function=get_weather>") != std::string::npos && prompt.find("Paris") != std::string::npos;
    const bool result = prompt.find("<tool_response>\nsunny, 21 C\n</tool_response>") != std::string::npos;
    const bool order = prompt.find("Weather in Paris?") < prompt.find("<function=get_weather>") &&
                       prompt.find("sunny, 21 C") < prompt.find("It is sunny.") && prompt.find("It is sunny.") < prompt.find("thanks");
    expect(st == 200 && tools && call && result && order,
           "server: /v1/responses: input_text parts, function_call / function_call_output items and an assistant message render as the chat turns, flat tools as chat tools (" +
               std::to_string(st) + std::string(tools ? "" : " no-tools") + (call ? "" : " no-call") + (result ? "" : " no-result") + (order ? "" : " order") + ")");
  }
  {  // refusals
    int s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
    post(r.base + "/v1/responses", "{\"input\":\"x\",\"previous_response_id\":\"resp_1\"}", &s1);
    post(r.base + "/v1/responses", "{\"input\":\"x\",\"tools\":[{\"type\":\"web_search\"}]}", &s2);
    post(r.base + "/v1/responses", "{\"input\":[{\"type\":\"item_reference\",\"id\":\"x\"}]}", &s3);
    post(r.base + "/v1/responses", "{\"instructions\":\"x\"}", &s4);
    const auto g = xenithium::http::request("GET", r.base + "/v1/responses/resp_123", "", 10);
    s5 = g.status;
    expect(s1 == 400 && s2 == 400 && s3 == 400 && s4 == 400 && s5 == 404,
           "server: /v1/responses refuses previous_response_id, non-function tools, item references, a missing input (400) and retrieval (404) (" +
               std::to_string(s1) + " " + std::to_string(s2) + " " + std::to_string(s3) + " " + std::to_string(s4) + " " + std::to_string(s5) + ")");
  }
  {  // streaming: the event sequence
    const auto res = xenithium::http::request("POST", r.base + "/v1/responses",
                                              "{\"input\":\"x\",\"stream\":true,\"max_output_tokens\":64}", 30);
    const auto ev = sse_events(res.body);
    std::string seq, reasoning, text;
    bool numbered = true;
    std::string final_status, final_text;
    for (size_t i = 0; i < ev.size(); ++i) {
      const auto &[type, v] = ev[i];
      const json::Value *sn = v.get("sequence_number");
      numbered = numbered && sn && sn->num == static_cast<double>(i) && v.get_string("type") && *v.get_string("type") == type;
      if (type == "response.reasoning_text.delta" || type == "response.output_text.delta") {
        (type == "response.output_text.delta" ? text : reasoning) += v.get_string("delta") ? *v.get_string("delta") : "";
        if (seq.size() < 7 || seq.substr(seq.size() - 6) != "delta,") seq += "delta,";
        continue;
      }
      seq += type.substr(type.find('.') + 1) + ",";
      if (type == "response.completed")
        if (const json::Value *resp = v.get_object("response")) {
          final_status = resp->get_string("status") ? *resp->get_string("status") : "";
          if (const json::Value *o = resp->get_array("output"); o && o->arr.size() == 2)
            if (const json::Value *c = o->arr[1].get_array("content"); c && !c->arr.empty() && c->arr[0].get_string("text")) final_text = *c->arr[0].get_string("text");
        }
    }
    const std::string want = "created,in_progress,output_item.added,reasoning_part.added,delta,reasoning_text.done,reasoning_part.done,output_item.done,"
                             "output_item.added,content_part.added,delta,output_text.done,content_part.done,output_item.done,completed,";
    expect(res.status == 200 && numbered && seq == want && reasoning == "Let me think." && text == "The answer is 4." && final_status == "completed" &&
               final_text == text && res.body.find("[DONE]") == std::string::npos,
           "server: /v1/responses stream: the reference's events in order, numbered, deltas == the final output (" + seq + " \"" + reasoning + "\" / \"" +
               text + "\" / " + final_status + " / \"" + final_text + "\" " + (numbered ? "numbered" : "unnumbered") + ")");
  }
  std::printf("  responses: %d script tokens\n", n_script);
}

void test_responses_tools(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  const std::string script = "ok</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 4096;
  sc.script = tok.encode(script);
  sc.script.push_back(tok.token_id("<|im_end|>"));
  sc.eos_ids = {tok.token_id("<|im_end|>")};
  sc.step_ms = 1;
  sc.tokens_per_step = 3;
  sc.slots = 2;
  core::StubEngine stub(sc);
  Running r;
  start_server(r, stub, tok, chat.get());
  const std::string tools = "\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}]";
  int st = 0;
  const auto v = post(r.base + "/v1/responses", "{\"input\":\"Weather in Paris?\"," + tools + "}", &st);
  const json::Value *out = v.get_array("output");
  const json::Value *fc = nullptr;
  if (out)
    for (const auto &it : out->arr)
      if (it.get_string("type") && *it.get_string("type") == "function_call") fc = &it;
  const std::string *args = fc ? fc->get_string("arguments") : nullptr;
  expect(st == 200 && fc && fc->get_string("name") && *fc->get_string("name") == "get_weather" && args && *args == "{\"city\": \"Paris\"}" &&
             fc->get_string("call_id") && fc->get_string("id") && fc->get_string("id")->rfind("fc_", 0) == 0,
         "server: /v1/responses: a tool call becomes a function_call item (name, arguments, call_id)");
  const auto res = xenithium::http::request("POST", r.base + "/v1/responses", "{\"input\":\"Weather in Paris?\",\"stream\":true," + tools + "}", 30);
  const auto ev = sse_events(res.body);
  std::string streamed, done_args, added_name;
  for (const auto &[type, e] : ev) {
    if (type == "response.function_call_arguments.delta" && e.get_string("delta")) streamed += *e.get_string("delta");
    if (type == "response.function_call_arguments.done" && e.get_string("arguments")) done_args = *e.get_string("arguments");
    if (type == "response.output_item.added")
      if (const json::Value *it = e.get_object("item"); it && it->get_string("name")) added_name = *it->get_string("name");
  }
  expect(res.status == 200 && added_name == "get_weather" && streamed == "{\"city\": \"Paris\"}" && done_args == streamed && !ev.empty() &&
             ev.back().first == "response.completed",
         "server: /v1/responses stream: function_call item added, argument deltas, arguments done, completed");
}

// usage.prompt_tokens_details.cached_tokens is the request's own prefix-cache share (the stub's cache: the longest
// prefix shared with an earlier prompt, up to the last token), on every endpoint, streamed or not
void test_cached_usage(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 4096;
  sc.script = tok.encode("The answer is 4.");
  sc.step_ms = 1;
  sc.slots = 2;
  sc.prefix_cache = true;
  core::StubEngine stub(sc);
  Running r;
  start_server(r, stub, tok, chat.get());
  auto details = [](const json::Value &u, const char *key) -> int {
    const json::Value *d = u.get_object(key);
    const json::Value *c = d ? d->get("cached_tokens") : nullptr;
    return c && c->is_number() ? static_cast<int>(c->num) : -1;
  };
  const std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about the prefix cache and its tokens.\"}],\"max_tokens\":6,\"temperature\":0";
  int prompt = -1, cold = -2, warm = -2;
  for (int k = 0; k < 2; ++k) {
    const auto v = post(r.base + "/v1/chat/completions", body + "}");
    if (const json::Value *u = v.get_object("usage")) {
      prompt = u->get("prompt_tokens") ? static_cast<int>(u->get("prompt_tokens")->num) : -1;
      (k == 0 ? cold : warm) = details(*u, "prompt_tokens_details");
    }
  }
  expect(prompt > 1 && cold == 0 && warm == prompt - 1, "server: chat usage reports the request's cached prompt tokens (cold " + std::to_string(cold) +
                                                            ", again " + std::to_string(warm) + " of " + std::to_string(prompt) + ")");
  {
    const auto res = xenithium::http::request("POST", r.base + "/v1/chat/completions", body + ",\"stream\":true,\"stream_options\":{\"include_usage\":true}}", 30);
    bool done = false;
    int got = -2;
    for (const auto &e : sse(res.body, done))
      if (const json::Value *u = e.get_object("usage")) got = details(*u, "prompt_tokens_details");
    expect(done && got == prompt - 1, "server: the streamed usage chunk carries cached_tokens (" + std::to_string(got) + ")");
  }
  {
    int st = 0;
    const auto a = post(r.base + "/v1/completions", "{\"prompt\":[5,6,7,8,9,10,11,12,13,14,15,16],\"max_tokens\":3}", &st);
    const auto b = post(r.base + "/v1/completions", "{\"prompt\":[5,6,7,8,9,10,11,12,40,41],\"max_tokens\":3}", &st);
    const json::Value *ua = a.get_object("usage"), *ub = b.get_object("usage");
    const int ca = ua ? details(*ua, "prompt_tokens_details") : -2, cb = ub ? details(*ub, "prompt_tokens_details") : -2;
    expect(st == 200 && ca == 0 && cb == 8, "server: completions usage: a new prompt 0 cached, one sharing 8 tokens 8 (" + std::to_string(ca) + ", " + std::to_string(cb) + ")");
  }
  {
    const std::string rb = "{\"input\":\"Tell me about the cached responses.\",\"max_output_tokens\":6,\"temperature\":0}";
    post(r.base + "/v1/responses", rb);
    const auto v = post(r.base + "/v1/responses", rb);
    const json::Value *u = v.get_object("usage");
    const int in = u && u->get("input_tokens") ? static_cast<int>(u->get("input_tokens")->num) : -1;
    const int c = u ? details(*u, "input_tokens_details") : -2;
    expect(in > 1 && c == in - 1, "server: /v1/responses usage reports input_tokens_details.cached_tokens (" + std::to_string(c) + " of " + std::to_string(in) + ")");
  }
}

void test_prefix_mark(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) return;
  const auto chat = text::open_chat_format("qwen38");
  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 4096;
  sc.script = tok.encode("The answer is 4.");
  sc.eos_ids = {tok.token_id("<|im_end|>")};
  sc.step_ms = 1;
  sc.slots = 2;
  core::StubEngine stub(sc);
  RecEngine rec(stub);
  Running r;
  start_server(r, rec, tok, chat.get(), 0, {}, 2);
  int st1 = 0, st2 = 0;
  post(r.base + "/v1/chat/completions",
       "{\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"2+2?\"},{\"role\":\"assistant\",\"content\":\"4\"},"
       "{\"role\":\"user\",\"content\":\"and 3+3?\"}],\"max_tokens\":4}",
       &st1);
  post(r.base + "/v1/completions", "{\"prompt\":\"<|im_start|>user hi\",\"max_tokens\":4}", &st2);
  std::vector<std::pair<std::vector<int32_t>, int32_t>> got;
  {
    std::lock_guard<std::mutex> lk(rec.mu);
    got = rec.got;
  }
  const int32_t ims = tok.token_id("<|im_start|>");
  bool ok = st1 == 200 && st2 == 200 && got.size() == 2 && ims >= 0;
  int32_t want = -1;
  if (ok)
    for (size_t i = 0; i < got[0].first.size(); ++i)
      if (got[0].first[i] == ims) want = static_cast<int32_t>(i);
  ok = ok && want > 0 && got[0].second == want && got[1].second == -1;
  expect(ok, "server: a chat request's prefix mark is its last <|im_start|> (" + std::to_string(got.empty() ? -2 : got[0].second) + " of " + std::to_string(want) +
                 "; status " + std::to_string(st1) + " / " + std::to_string(st2) + ", " + std::to_string(got.size()) + " submits, token " + std::to_string(ims) +
                 "), a completion has none");
}

void test_server(const std::string &tokdir) {
  text::Tokenizer tok;
  std::string err;
  if (!tok.load(tokdir, err)) {
    expect(false, "server: the test tokenizer loads (" + err + ")");
    return;
  }
  expect(tok.vocab_size() == kVocab, "server: test tokenizer vocab " + std::to_string(tok.vocab_size()));
  const auto chat = text::open_chat_format("qwen38");
  expect(chat != nullptr, "server: qwen38 has a chat format");

  core::StubConfig sc;
  sc.model.name = "test-model";
  sc.model.max_model_len = 4096;
  sc.script = tok.encode("Let me think.</think>The answer is 4.");
  sc.eos_ids = {tok.token_id("<|im_end|>")};
  sc.step_ms = 1;
  sc.tokens_per_step = 3;
  sc.slots = 2;
  core::StubEngine stub(sc);
  Running r;
  start_server(r, stub, tok, chat.get(), 0, {}, 2, nullptr, 0, nullptr, [](xenithium::server::Config &cfg) {
    cfg.devices = {{.name = "Card A", .memory_bytes = 1ull << 30, .memory_gb = 1, .pci = "0000:03:00.0", .driver = "1.17.39395+16"}, {.name = "Card A", .memory_bytes = 1ull << 30}};
    cfg.tensor_parallel = 2;
    cfg.build = "abc1234", cfg.oneapi = "2025.3.0", cfg.drafter = "dspine+mtp";
    cfg.machine = {.os = "Ubuntu 26.04.1 LTS", .kernel = "7.0.0-43-generic", .cpu = "AMD EPYC 7282", .cpu_threads = 16, .memory_gib = 46};
    cfg.launch_args = {"--model", "/m/ckpt", "--kv", "int8-v4", "--report-launch"};
    cfg.launch_env = {{"XEN_QSA_GATHER", "1"}};
  });

  {
    const auto h = xenithium::http::request("GET", r.base + "/health", "", 10);
    expect(h.status == 200, "server: /health 200");
    json::Value m;
    json::parse(xenithium::http::request("GET", r.base + "/v1/models", "", 10).body, m);
    const json::Value *d = m.get_array("data");
    expect(d && d->arr.size() == 1 && d->arr[0].get_string("id") && *d->arr[0].get_string("id") == "test-model", "server: /v1/models lists the served name");
    if (d && d->arr.size() == 1) {  // max_model_len = one request; the KV of all slots (the stub: fixed per slot) beside it
      const json::Value &e = d->arr[0];
      const json::Value *len = e.get("max_model_len"), *sl = e.get("slots"), *pool = e.get("kv_pool_tokens");
      expect(len && len->num == 4096 && sl && sl->num == 2 && pool && pool->num == 2 * 4096 && e.get_string("kv_layout") &&
                 *e.get_string("kv_layout") == "per_slot",
             "server: /v1/models reports max_model_len per request plus slots, kv_layout and kv_pool_tokens");
      const json::Value *tp = e.get("tensor_parallel"), *devs = e.get_array("devices");
      expect(tp && tp->num == 2 && devs && devs->arr.size() == 2 && devs->arr[0].get_string("name") && *devs->arr[0].get_string("name") == "Card A" &&
                 devs->arr[0].get("memory_bytes") && devs->arr[0].get("memory_bytes")->num == 1073741824.0 && devs->arr[0].get_string("pci") &&
                 *devs->arr[0].get_string("pci") == "0000:03:00.0" && devs->arr[0].get("memory_gb") && devs->arr[0].get("memory_gb")->num == 1 &&
                 !devs->arr[1].get("pci") && !devs->arr[1].get("memory_gb"),
             "server: /v1/models reports the TP width and the cards in rank order (no pci or memory_gb field where none is known)");
      const json::Value *launch = e.get_object("launch");
      const json::Value *args = launch ? launch->get_array("args") : nullptr, *env = launch ? launch->get_object("env") : nullptr;
      auto text = [&](const char *key) { return e.get_string(key) ? *e.get_string(key) : std::string(); };
      expect(text("build") == "abc1234" && text("oneapi") == "2025.3.0" && text("drafter") == "dspine+mtp" && devs && devs->arr[0].get_string("driver") &&
                 *devs->arr[0].get_string("driver") == "1.17.39395+16" && args && args->arr.size() == 5 && args->arr[3].str == "int8-v4" && env &&
                 env->get_string("XEN_QSA_GATHER") && *env->get_string("XEN_QSA_GATHER") == "1" && e.get_object("machine") &&
                 e.get_object("machine")->get_string("kernel") && *e.get_object("machine")->get_string("kernel") == "7.0.0-43-generic" &&
                 e.get_object("machine")->get("memory_gib") && e.get_object("machine")->get("memory_gib")->num == 46,
             "server: /v1/models reports its build, oneAPI, drafter, machine, the cards' driver and its launch (arguments, XEN_ variables)");
    }
    expect(xenithium::server::secret_name("XEN_BENCH_API_KEY") && xenithium::server::secret_name("--api-key") && xenithium::server::secret_name("HF_TOKEN") &&
               !xenithium::server::secret_name("XEN_KV_POOL_TOKENS") && !xenithium::server::secret_name("--kv-pool-tokens") && !xenithium::server::secret_name("XEN_KV_LEND_KEEP"),
           "server: secret-looking names (key, token) are hidden from the launch report, counts of tokens are not");
    {
      namespace sv = xenithium::server;
      expect(sv::secret_name("XEN_AUTHORIZATION") && sv::secret_name("XEN_JWT") && sv::secret_name("XEN_OAUTH_CLIENT") &&
                 sv::secret_name("XEN_DB_PASSWORD") && sv::secret_name("XEN_SESSION_COOKIE") && !sv::secret_name("XEN_PREFILL_SHARE"),
             "server: AUTH, JWT, PASSW, COOKIE inside a name hide it (XEN_AUTHORIZATION), ordinary names stay");
      expect(sv::hide_url_credentials("http://user:pa55@proxy:3128") == "http://user:(hidden)@proxy:3128" &&
                 sv::hide_url_credentials("a=https://u:p@x/y,b=http://plain/z") == "a=https://u:(hidden)@x/y,b=http://plain/z" &&
                 sv::hide_url_credentials("http://user@host/x") == "http://user@host/x" && sv::hide_url_credentials("/models/q@1") == "/models/q@1",
             "server: a URL's password is hidden, a user name alone and a path with @ are kept");
      expect(sv::secret_value("hf_abcdefghijklmnopqrstuvwxyz") && sv::secret_value("Bearer abc") &&
                 sv::secret_value("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig") && !sv::secret_value("hf_short") &&
                 !sv::secret_value("/models/Qwen/Qwen3.8-27B") && !sv::secret_value("sk-x"),
             "server: token-looking values (hf_, Bearer, a JWT) are secret whatever their name, paths and short words are not");
      sv::Config c;
      sv::fill_launch(c,
                      {"--model", "/m", "--api-key", "k1", "--x-token=k2", "--served-model-name", "hf_abcdefghijklmnopqrstuvwxyz",
                       "--name=sk-abcdefghijklmnopqrstuv", "--proxy", "http://u:p@h:1"},
                      {"XEN_B=1", "XEN_AUTHORIZATION=abc", "XEN_HTTP_PROXY=http://u:p@h:1", "HOME=/home/x", "XEN_A=ghp_abcdefghijklmnopqrstu"});
      const std::vector<std::string> want_args = {"--model", "/m", "--api-key", "(hidden)", "--x-token=(hidden)", "--served-model-name",
                                                  "(hidden)", "--name=(hidden)", "--proxy", "http://u:(hidden)@h:1"};
      const std::vector<std::pair<std::string, std::string>> want_env = {
          {"XEN_A", "(hidden)"}, {"XEN_AUTHORIZATION", "(hidden)"}, {"XEN_B", "1"}, {"XEN_HTTP_PROXY", "http://u:(hidden)@h:1"}};
      bool env_ok = c.launch_env.size() == want_env.size();
      for (size_t k = 0; env_ok && k < want_env.size(); ++k) env_ok = c.launch_env[k].first == want_env[k].first && c.launch_env[k].second == want_env[k].second;
      expect(c.launch_args == want_args && env_ok,
             "server: the launch report hides a secret flag's value (both forms), a token passed under an ordinary flag, a URL's "
             "password, and keeps only XEN_ variables, sorted");
    }
    expect(xenithium::http::request("GET", r.base + "/nope", "", 10).status == 404, "server: unknown path 404");
  }
  const std::string script_text = "Let me think.</think>The answer is 4.";
  {
    int st = 0;
    const auto v = post(r.base + "/v1/completions", "{\"prompt\":\"hi\",\"max_tokens\":64,\"temperature\":0}", &st);
    const json::Value *c = v.get_array("choices");
    const std::string *t = c && !c->arr.empty() ? c->arr[0].get_string("text") : nullptr;
    expect(st == 200 && t && t->rfind(script_text, 0) == 0, "server: /v1/completions 200, the text starts with the stub's script");
    const json::Value *u = v.get_object("usage");
    expect(u && u->get("completion_tokens") && u->get("completion_tokens")->num == 64 && u->get("prompt_tokens")->num == 2,
           "server: usage counts 2 prompt + 64 completion tokens");
  }
  {
    const auto res = xenithium::http::request("POST", r.base + "/v1/completions", "{\"prompt\":[1,2,3],\"max_tokens\":30,\"stream\":true,\"stop\":[\"think\"]}", 30);
    bool done = false;
    const auto ev = sse(res.body, done);
    std::string all;
    std::string finish;
    int empty = 0;
    for (const auto &e : ev)
      if (const json::Value *c = e.get_array("choices"); c && !c->arr.empty()) {
        const std::string *t = c->arr[0].get_string("text");
        if (t) all += *t;
        if (const std::string *f = c->arr[0].get_string("finish_reason")) finish = *f;
        else empty += t && t->empty();
      }
    expect(res.status == 200 && done && all == "Let me " && finish == "stop", "server: streamed completion with a stop string ends before it (got \"" + all + "\", " + finish + ")");
    // tokens whose text is held back (a possible stop-string prefix) still get a chunk,
    // with empty text
    expect(empty > 0, "server: streamed completion sends a chunk per step even when its text is held back (" + std::to_string(empty) + " empty-text chunks)");
  }
  {
    int st = 0;
    const auto v = post(r.base + "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"2+2?\"}],\"max_tokens\":" +
                                                             std::to_string(tok.encode(script_text).size()) + "}", &st);
    const json::Value *c = v.get_array("choices");
    const json::Value *msg = c && !c->arr.empty() ? c->arr[0].get_object("message") : nullptr;
    const std::string *reasoning = msg ? msg->get_string("reasoning") : nullptr;
    const std::string *content = msg ? msg->get_string("content") : nullptr;
    expect(st == 200 && reasoning && content, "server: chat completion 200 with reasoning and content");
    if (reasoning && content)
      expect(*reasoning == "Let me think." && *content == "The answer is 4.",
             "server: chat output split at </think> (reasoning \"" + *reasoning + "\", content \"" + *content + "\")");
  }
  {
    const auto res = xenithium::http::request("POST", r.base + "/v1/chat/completions",
                                              "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":40,\"stream\":true,"
                                              "\"stream_options\":{\"include_usage\":true}}", 30);
    bool done = false;
    const auto ev = sse(res.body, done);
    std::string reasoning, content;
    bool usage = false;
    for (const auto &e : ev) {
      if (e.get_object("usage")) usage = true;
      if (const json::Value *c = e.get_array("choices"); c && !c->arr.empty())
        if (const json::Value *d = c->arr[0].get_object("delta")) {
          if (const std::string *t = d->get_string("reasoning")) reasoning += *t;
          if (const std::string *t = d->get_string("content")) content += *t;
        }
    }
    expect(done && usage && reasoning == "Let me think." && content.rfind("The answer", 0) == 0,
           "server: streamed chat deltas: reasoning then content, usage chunk, [DONE] (reasoning \"" + reasoning + "\", content \"" + content + "\")");
  }
  {
    int st = 0;
    post(r.base + "/v1/completions", "{\"prompt\":\"x\",\"guided_regex\":\"a+\"}", &st);
    expect(st == 400, "server: guided_regex answers 400");
    post(r.base + "/v1/completions", "{\"prompt\":\"x\",\"guided_json\":{\"type\":\"string\",\"pattern\":\"a\"}}", &st);
    expect(st == 400, "server: a schema with pattern answers 400");
    post(r.base + "/v1/completions", "{\"prompt\":\"x\",\"max_tokens\":0}", &st);
    expect(st == 400, "server: max_tokens 0 answers 400");
    post(r.base + "/v1/completions", "{\"prompt\":\"x\",\"model\":\"other\"}", &st);
    expect(st == 404, "server: an unknown model answers 404");
    post(r.base + "/v1/completions", "{\"prompt\":\"x\",\"max_tokens\":5000}", &st);
    expect(st == 400, "server: prompt + max_tokens over max_model_len answers 400");
  }
  {
    int st = 0;
    const auto v = post(r.base + "/v1/completions", "{\"prompt\":\"hi\",\"max_tokens\":6,\"n\":3,\"temperature\":0}", &st);
    const json::Value *c = v.get_array("choices");
    expect(st == 200 && c && c->arr.size() == 3, "server: n 3 gives 3 choices");
  }
  {
    const auto res = xenithium::http::request("GET", r.base + "/metrics", "", 10);
    expect(res.status == 200 && res.body.find("vllm:generation_tokens_total") != std::string::npos &&
               res.body.find("vllm:time_to_first_token_seconds_bucket") != std::string::npos,
           "server: /metrics carries the vllm:* counters and histograms");
  }
  r.srv->stop();
  r.th.join();
  stub.stop();

  // The batch engine over the fake model behind the server: greedy completions, 6 at once.
  {
    FakeModel fm(4, 4096, 16, kVocab);
    fm.step_us = 2000;  // requests arrive over HTTP slower than a free fake step: let them overlap
    core::BatchConfig bc;
    bc.model.name = "fake";
    bc.eos_ids = {-1};
    bc.log_interval_s = 0;
    core::BatchEngine be(fm, bc);
    Running r2;
    start_server(r2, be, tok, chat.get());
    std::vector<std::thread> th;
    std::vector<int> ntok(6, 0), status(6, 0);
    for (int i = 0; i < 6; ++i)
      th.emplace_back([&, i] {
        const auto p = prompt_of(i, 10 + 30 * i);
        std::string arr = "[";
        for (size_t k = 0; k < p.size(); ++k) arr += (k ? "," : "") + std::to_string(p[k]);
        arr += "]";
        const auto v = post(r2.base + "/v1/completions", "{\"prompt\":" + arr + ",\"max_tokens\":" + std::to_string(12 + i) + ",\"temperature\":0}", &status[i]);
        if (const json::Value *u = v.get_object("usage"))
          if (const json::Value *ct = u->get("completion_tokens")) ntok[i] = static_cast<int>(ct->num);
      });
    for (auto &t : th) t.join();
    bool ok = true;
    for (int i = 0; i < 6; ++i) ok &= status[i] == 200 && ntok[i] == 12 + i;
    expect(ok, "server + batch engine: 6 concurrent completions on 4 slots, each its max_tokens");
    expect(fm.errors == 0 && fm.max_batch >= 2, "server + batch engine: no slot misuse, steps batched (max " + std::to_string(fm.max_batch) + ")");
    r2.srv->stop();
    r2.th.join();
    be.stop();
  }
  // Structured output end to end: the fake model's logits favour nothing allowed, so a greedy
  // constrained pick is the lowest allowed id at each step; the answers must still be valid.
  {
    FakeModel fm(2, 4096, 16, kVocab);
    core::BatchConfig bc;
    bc.model.name = "fake";
    const int32_t eos = tok.token_id("<|im_end|>");
    bc.eos_ids = {eos};
    bc.log_interval_s = 0;
    core::BatchEngine be(fm, bc);
    Running r3;
    start_server(r3, be, tok, chat.get(), 0, {eos});
    int st = 0;
    auto v = post(r3.base + "/v1/completions", "{\"prompt\":\"pick\",\"max_tokens\":20,\"temperature\":0,\"guided_choice\":[\"yes\",\"no\"]}", &st);
    const json::Value *c = v.get_array("choices");
    const std::string *t = c && !c->arr.empty() ? c->arr[0].get_string("text") : nullptr;
    const std::string *f = c && !c->arr.empty() ? c->arr[0].get_string("finish_reason") : nullptr;
    expect(st == 200 && t && f && (*t == "yes" || *t == "no") && *f == "stop",
           "server + batch engine: guided_choice answers one of the choices, then stops (" + (t ? *t : std::string("?")) + ")");
    v = post(r3.base + "/v1/completions",
             "{\"prompt\":\"x\",\"max_tokens\":200,\"temperature\":0.9,\"seed\":3,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":"
             "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"},\"n\":{\"enum\":[1,2,3]}},\"required\":[\"ok\",\"n\"]}}}}",
             &st);
    c = v.get_array("choices");
    t = c && !c->arr.empty() ? c->arr[0].get_string("text") : nullptr;
    json::Value doc;
    const bool parsed = t && json::parse(*t, doc) && doc.is_object() && doc.get("ok") && doc.get("ok")->is_bool() && doc.get("n") && doc.get("n")->is_number();
    expect(st == 200 && parsed, "server + batch engine: a sampled json_schema answer parses and has its required typed fields (" + (t ? *t : std::string("?")) + ")");
    // tool_choice required, one call (parallel_tool_calls false): the answer is the call
    const std::string tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}},"
                              "{\"type\":\"function\",\"function\":{\"name\":\"get_date\"}}]";
    v = post(r3.base + "/v1/chat/completions",
             "{\"messages\":[{\"role\":\"user\",\"content\":\"time?\"}],\"max_tokens\":1500,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},"
             "\"tools\":" + tools + ",\"tool_choice\":\"required\",\"parallel_tool_calls\":false}",
             &st);
    c = v.get_array("choices");
    const json::Value *msg = c && !c->arr.empty() ? c->arr[0].get_object("message") : nullptr;
    const json::Value *calls = msg ? msg->get_array("tool_calls") : nullptr;
    f = c && !c->arr.empty() ? c->arr[0].get_string("finish_reason") : nullptr;
    const json::Value *fn = calls && calls->arr.size() == 1 ? calls->arr[0].get_object("function") : nullptr;
    expect(st == 200 && fn && fn->get_string("name") && *fn->get_string("arguments") == "{}" && f && *f == "tool_calls" && msg->get_string("content") &&
               msg->get_string("content")->empty(),
           "server + batch engine: tool_choice required -> one tool call from the guided JSON, finish tool_calls, content \"\" (" + json::dump(v).substr(0, 600) + ")");
    v = post(r3.base + "/v1/chat/completions",
             "{\"messages\":[{\"role\":\"user\",\"content\":\"time?\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},"
             "\"tools\":" + tools + ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_date\"}}}",
             &st);
    c = v.get_array("choices");
    msg = c && !c->arr.empty() ? c->arr[0].get_object("message") : nullptr;
    calls = msg ? msg->get_array("tool_calls") : nullptr;
    fn = calls && calls->arr.size() == 1 ? calls->arr[0].get_object("function") : nullptr;
    const std::string *args = fn ? fn->get_string("arguments") : nullptr;
    expect(st == 200 && fn && *fn->get_string("name") == "get_date" && args && !args->empty() && args->front() == '{',
           "server + batch engine: a named tool_choice -> that function, its arguments the guided JSON text, starting at \"{\" (" + (args ? *args : std::string("?")) + ")");
    r3.srv->stop();
    r3.th.join();
    be.stop();
  }
}

// The warm-up runs scratch waves through the step loop before serving, then
// leaves no trace: stats cleared, every slot free, and the requests after it == each alone.
void test_warm_up() {
  const int V = kVocab;
  const int N = 6;
  const int lens[N] = {2, 9, 16, 33, 70, 140};
  for (int depth : {0, 3}) {
    const std::string tag = depth ? " (drafter)" : "";
    std::vector<std::vector<int32_t>> solo(N);
    for (int i = 0; i < N; ++i) {
      FakeModel fm(1, 4096, 16, V);
      fm.depth = depth;
      core::BatchConfig bc;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 15 + i;
      e.submit(1, prompt_of(i, lens[i]), p);
      c.wait(1);
      e.stop();
      solo[static_cast<size_t>(i)] = c.toks[1];
    }
    FakeModel fm(4, 4096, 16, V);
    fm.depth = depth;
    fm.split_layers = 4;
    fm.dev_lp = true;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefill_stall_ms = 100;
    bc.prefill_part_ms = 1e-6;  // every single-call prompt beside decoding runs in parts
    core::BatchEngine e(fm, bc);
    auto wr = [&](int i, int len, int max_tokens) {
      core::WarmRequest r;
      r.prompt = prompt_of(50 + i, len);
      r.params.temperature = 0;
      r.params.max_tokens = max_tokens;
      r.params.ignore_eos = true;
      return r;
    };
    std::vector<std::vector<core::WarmRequest>> waves(3);
    waves[0].push_back(wr(0, 40, 16));
    waves[1].push_back(wr(0, 40, 16));
    waves[2].push_back(wr(1, 5, 28));
    auto s = wr(2, 30, 20);
    s.params.temperature = 0.7f;
    s.params.has_seed = true;
    s.params.logprobs = 2;
    waves[2].push_back(s);
    auto l = wr(3, 12, 16);
    l.params.logprobs = 1;
    l.params.prompt_logprobs = 1;
    waves[2].push_back(l);
    waves[2].push_back(wr(4, 7, 12));
    const double ms = e.warm_up(waves);
    const auto st = e.stats();
    expect(ms >= 0 && fm.errors == 0 && fm.prefills >= 6 && fm.max_batch == 4 && fm.split_begins > 0,
           "warm-up" + tag + ": its waves ran (6 requests, a 4-sequence step, a prefill call in parts)");
    expect(st.requests_finished == 0 && st.prompt_tokens == 0 && st.generation_tokens == 0 && st.steps == 0 && st.ttft.count == 0 &&
               st.finished_by[static_cast<int>(core::Finish::length)] == 0 && st.running == 0 && st.waiting == 0 && st.slots == 4,
           "warm-up" + tag + ": the stats are cleared (slots kept)");
    expect(e.pf_fit() == core::BatchEngine(fm, bc).pf_fit(), "warm-up" + tag + ": the prefill cost model is back at its defaults");
    Collect c;
    e.start(&c);
    for (int i = 0; i < N; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.max_tokens = 15 + i;
      e.submit(static_cast<uint64_t>(1 + i), prompt_of(i, lens[i]), p);
    }
    expect(c.wait(N), "warm-up" + tag + ": the requests after it all finish");
    e.stop();
    bool eq = true;
    for (int i = 0; i < N; ++i) eq &= c.toks[static_cast<uint64_t>(1 + i)] == solo[static_cast<size_t>(i)];
    expect(eq && fm.errors == 0, "warm-up" + tag + ": 6 requests on 4 slots after it == each alone");
    expect(e.stats().requests_finished == N, "warm-up" + tag + ": the stats count only the requests after it");
    // a warm-up request that cannot run is an error, not a silent skip
    FakeModel fe(1, 64, 16, V);
    core::BatchEngine ee(fe, bc);
    bool threw = false;
    try {
      ee.warm_up({{wr(0, 100, 4)}});
    } catch (const std::runtime_error &) {
      threw = true;
    }
    expect(threw, "warm-up" + tag + ": a prompt longer than the context throws");
  }
}

}  // namespace

// YaRN tiers (FakeModel tier limits 128 / 256 / capacity 1,024). A request starts in the smallest
// tier holding its prompt plus room to grow; one that grows past its tier resumes (re-prefills) in the next: every
// output == the same requests without tiers, no write past a slot's tier, a prefix reused only within its tier.
void test_yarn_tiers() {
  struct Q {
    int prompt, max_tokens, retiers;  // retiers: tier crossings expected
  };
  // 20+40: tier 0; 30 -> 230: crosses 128; 150+50: tier 1; 300+100: the last; 20 -> 620: crosses 128 and 256
  const std::vector<Q> qs = {{20, 40, 0}, {30, 200, 1}, {150, 50, 0}, {300, 100, 0}, {20, 600, 2}};
  auto run = [&](bool tiers, int slots, int pool_pages, int depth, bool cache, int rounds, std::map<uint64_t, std::vector<int32_t>> &toks, core::Stats &st,
                 FakeModel *&keep) {
    auto *fm = new FakeModel(slots, 1024, 16, kVocab);
    if (tiers) fm->tier_limits = {128, 256, 1024};
    fm->pool_pages = pool_pages;
    fm->depth = depth;
    fm->snap_cap = cache ? 16 : 0;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefix_cache = cache;
    bc.prefix_min_tokens = 16;
    bc.prefix_mark_min = 16;
    bool ok = true;
    {
      core::BatchEngine e(*fm, bc);
      Collect c;
      e.start(&c);
      uint64_t id = 0;
      for (int r = 0; r < rounds && ok; ++r) {
        for (size_t i = 0; i < qs.size(); ++i) {
          core::SamplingParams p;
          p.temperature = 0.0f;
          p.ignore_eos = true;
          p.max_tokens = qs[i].max_tokens;
          p.prefix_mark = qs[i].prompt - 3;  // a 3-token "generation prompt"
          // Repeats 's prompts: a cached prefix of the same request (same tier) is reused
          e.submit(++id, prompt_of(static_cast<int>(i), qs[i].prompt), p);
        }
        ok = c.wait(static_cast<size_t>(id), 60);
      }
      e.stop();
      toks = c.toks;
      st = e.stats();
      for (uint64_t k = 1; k <= id; ++k) ok &= c.fin[k] == core::Finish::length && c.toks[k].size() == static_cast<size_t>(qs[(k - 1) % qs.size()].max_tokens);
    }
    keep = fm;
    return ok;
  };
  int want_retiers = 0;
  for (const auto &q : qs) want_retiers += q.retiers;
  struct Case {
    const char *name;
    int slots, pool_pages, depth;
    bool cache;
    int rounds;
  };
  for (const Case &k : {Case{"1 slot, fixed layout", 1, 0, 0, false, 1}, Case{"3 slots, fixed layout", 3, 0, 0, false, 1},
                        Case{"3 slots, drafter", 3, 0, 3, false, 1}, Case{"3 slots, shared pool 1,600 tokens", 3, 200, 0, false, 1}}) {
    const std::string tag = std::string("yarn tiers (") + k.name + "): ";
    std::map<uint64_t, std::vector<int32_t>> ref, got;
    core::Stats st0, st;
    FakeModel *f0 = nullptr, *f1 = nullptr;
    const bool ok0 = run(false, k.slots, k.pool_pages, k.depth, k.cache, k.rounds, ref, st0, f0);
    const bool ok1 = run(true, k.slots, k.pool_pages, k.depth, k.cache, k.rounds, got, st, f1);
    expect(ok0 && ok1, tag + "every request runs to max_tokens");
    expect(got == ref, tag + "every output == the same requests without tiers");
    expect(f0->errors == 0 && f1->errors == 0 && f1->tier_misses == 0,
           tag + "no write past a slot's tier, no tier set on a used sequence, no snapshot of another tier (" + std::to_string(f1->tier_misses) + " misses, " +
               std::to_string(f1->errors) + " errors)");
    const int rt = static_cast<int>(st.yarn_retiers);
    expect(rt == want_retiers * k.rounds && st0.yarn_retiers == 0 && f1->tier_sets >= static_cast<int>(qs.size()) * k.rounds,
           tag + std::to_string(rt) + " tier crossings (want " + std::to_string(want_retiers * k.rounds) + "), " + std::to_string(f1->tier_sets) + " tiers set");
    delete f0;
    delete f1;
  }
  // the prefix cache: A (40 tokens, tier 0) twice, then B = A's prompt grown to 200 tokens (tier 1): the second A reuses
  // the first's prefix; B may not (another tier's rows) and runs from position 0; without tiers B reuses it too
  for (const bool tiers : {false, true}) {
    FakeModel fm(2, 1024, 16, kVocab);
    if (tiers) fm.tier_limits = {128, 256, 1024};
    fm.snap_cap = 16;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefix_min_tokens = 16;
    bc.prefix_mark_min = 16;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    const auto a = prompt_of(1, 40);
    auto b = a;
    const auto more = prompt_of(2, 160);
    b.insert(b.end(), more.begin(), more.end());
    std::vector<uint64_t> hits;
    uint64_t id = 0;
    bool ok = true;
    for (const std::vector<int32_t> *pr : {&a, &a, static_cast<const std::vector<int32_t> *>(&b)}) {
      core::SamplingParams p;
      p.temperature = 0.0f;
      p.ignore_eos = true;
      p.max_tokens = 20;
      p.prefix_mark = 37;  // the shared part's end
      e.submit(++id, *pr, p);
      ok &= c.wait(static_cast<size_t>(id), 30);
      hits.push_back(e.stats().prefix_hits);
    }
    e.stop();
    const std::string tag = std::string("yarn tiers (prefix cache, ") + (tiers ? "tiers" : "one factor") + "): ";
    const bool a2 = hits[1] > hits[0], b2 = hits[2] > hits[1];
    expect(ok && fm.errors == 0 && fm.tier_misses == 0 && a2 && b2 == !tiers,
           tag + "the repeated prompt reuses its prefix (" + (a2 ? "yes" : "no") + "), the longer prompt in tier 1 " + (b2 ? "reuses it" : "does not") +
               " (want " + (tiers ? "not" : "reuse") + "), no tier misuse");
  }
}

// a KV pool shared by the slots (FakeModel pages of 8 tokens). A pool too small for every running
// request preempts the youngest; it recomputes its prompt + output and continues: every output == the request alone.
// KV lending vs the prefix cache (BatchConfig:lend_park_s / lend_pressure): a parked sequence above the lend keep is
// dropped after the park timeout while idle; under reservation pressure (lru / idle) before the experts go back.
void test_kv_lend_park() {
  const int V = kVocab;
  struct R {
    int64_t held_a = 0;
    core::Stats st;
    int errors = 0;
    bool ok = false;
  };
  auto run = [&](double park_s, const std::string &pressure, double grace_s, bool big_b, double idle_wait_s) {
    R r;
    FakeModel fm(2, 1024, 16, V);
    fm.pool_pages = 128;  // 1,024 tokens
    fm.snap_cap = 8;
    fm.lend_keep = 64, fm.lend_limit = 400;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefix_min_tokens = 16;
    bc.lend_park_s = park_s;
    bc.lend_pressure = pressure;
    bc.lend_grace_s = grace_s;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.ignore_eos = true;
    p.max_tokens = 20;
    e.submit(1, prompt_of(1, 300), p);  // A: 320 tokens, parked above the keep
    r.ok = c.wait(1, 30);
    std::this_thread::sleep_for(std::chrono::duration<double>(idle_wait_s));
    if (big_b) {
      e.submit(2, prompt_of(2, 200), p);  // B: 220 more tokens: past the lend limit with A parked
      r.ok = r.ok && c.wait(2, 30);
    }
    r.st = e.stats();
    e.stop();
    r.held_a = fm.kv_held_tokens(0) + fm.kv_held_tokens(1);  // A parked (and B parked after it)
    r.errors = fm.errors;
    return r;
  };
  const R t1 = run(0.2, "kv", 60, false, 0.6), t0 = run(0, "kv", 60, false, 0.6);
  expect(t1.ok && t1.held_a == 0 && t1.st.lend_park_drops == 1 && t1.errors == 0, "kv lend: a parked sequence above the keep is dropped after the park timeout while idle");
  expect(t0.ok && t0.held_a > 0 && t0.st.lend_park_drops == 0, "kv lend: park timeout 0 keeps it");
  const R kv = run(0, "kv", 60, true, 0), lru = run(0, "lru", 60, true, 0), idle = run(0, "idle", 60, true, 0), idle0 = run(0, "idle", 0.1, true, 0.3);
  expect(kv.ok && kv.st.lend_pressure_drops == 0 && kv.held_a > 0, "kv lend pressure kv: the parked sequence stays (the experts go)");
  expect(lru.ok && lru.st.lend_pressure_drops == 1 && lru.held_a < kv.held_a && lru.errors == 0, "kv lend pressure lru: the parked sequence is dropped first");
  expect(idle.ok && idle.st.lend_pressure_drops == 0 && idle0.ok && idle0.st.lend_pressure_drops == 1,
         "kv lend pressure idle: only a sequence parked past the grace is dropped");
}

void test_kv_pool() {
  const int V = kVocab;
  const int N = 8;
  const int lens[N] = {5, 40, 17, 90, 3, 61, 120, 33};
  auto run = [&](int slots, int pool_pages, int depth, bool sampled, std::map<uint64_t, std::vector<int32_t>> &toks, core::Stats &st, int &errors,
                 int &max_held) {
    FakeModel fm(slots, 256, 16, V);
    fm.pool_pages = pool_pages;
    fm.depth = depth;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < N; ++i) {
      core::SamplingParams p;
      p.temperature = sampled ? 0.8f : 0.0f;
      p.has_seed = sampled;
      p.seed = 7 + static_cast<uint64_t>(i);
      p.ignore_eos = true;
      p.max_tokens = 60 + 11 * i;
      if (i == 2) p.logprobs = 1;
      e.submit(static_cast<uint64_t>(i + 1), prompt_of(i, lens[i]), p);
    }
    const bool ok = c.wait(N, 60);
    e.stop();
    toks = c.toks;
    st = e.stats();
    errors = fm.errors;
    max_held = fm.max_held;
    bool all_len = ok;
    for (int i = 0; i < N; ++i) all_len &= c.fin[static_cast<uint64_t>(i + 1)] == core::Finish::length && c.toks[static_cast<uint64_t>(i + 1)].size() == static_cast<size_t>(60 + 11 * i);
    return all_len;
  };
  for (const int depth : {0, 3}) {
    for (const bool sampled : {false, true}) {
      if (depth && sampled) continue;  // speculative sampling depends on the drafts, which depend on the batch: not "== alone"
      const std::string tag = std::string("kv pool (") + (depth ? "drafter, " : "plain, ") + (sampled ? "sampled" : "greedy") + "): ";
      std::map<uint64_t, std::vector<int32_t>> ref, got;
      core::Stats st0, st;
      int e0 = 0, e1 = 0, h0 = 0, h1 = 0;
      const bool ok0 = run(1, 0, depth, sampled, ref, st0, e0, h0);  // one slot, fixed layout: every request alone
      const bool ok1 = run(4, 40, depth, sampled, got, st, e1, h1);  // 4 slots on 320 tokens of pool (fixed would need 1,024)
      expect(ok0 && ok1, tag + "every request runs to max_tokens");
      expect(got == ref, tag + "every output == the request alone on one slot");
      expect(e0 == 0 && e1 == 0, tag + "no write past a slot's reserved pages, no slot misuse");
      expect(st.preemptions > 0 && st.kv_pool_tokens == 320 && st.kv_used_tokens == 0 && h1 <= 320,
             tag + "preemptions " + std::to_string(st.preemptions) + ", pool 320 tokens, peak held " + std::to_string(h1) + ", nothing held at the end");
      expect(st.prompt_tokens == st0.prompt_tokens && st.ttft.count == static_cast<uint64_t>(N) && st.running == 0 && st.waiting == 0,
             tag + "stats: prompt tokens as without preemption, one TTFT sample per request, nothing left");
    }
  }
  {  // a big pool: no preemption, the same outputs
    std::map<uint64_t, std::vector<int32_t>> ref, got;
    core::Stats st0, st;
    int e0 = 0, e1 = 0, h0 = 0, h1 = 0;
    run(1, 0, 3, false, ref, st0, e0, h0);
    run(4, 4 * 32, 3, false, got, st, e1, h1);
    expect(got == ref && st.preemptions == 0 && e1 == 0, "kv pool: a pool of slots x capacity never preempts, same outputs");
  }
  {  // cancels of running and preempted requests
    FakeModel fm(4, 256, 16, V);
    fm.pool_pages = 40;
    fm.step_us = 200;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    for (int i = 0; i < 6; ++i) {
      core::SamplingParams p;
      p.temperature = 0;
      p.ignore_eos = true;
      p.max_tokens = 1000;
      e.submit(static_cast<uint64_t>(50 + i), prompt_of(i, 30), p);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    for (int i = 0; i < 6; ++i) e.cancel(static_cast<uint64_t>(50 + i));
    const bool ok = c.wait(6, 20);
    e.stop();
    int ended = 0;
    for (int i = 0; i < 6; ++i) {
      const auto f = c.fin[static_cast<uint64_t>(50 + i)];
      ended += f == core::Finish::cancelled || f == core::Finish::length;
    }
    const auto st = e.stats();
    expect(ok && ended == 6 && fm.errors == 0 && st.running == 0 && st.waiting == 0 && st.kv_used_tokens == 0,
           "kv pool: cancels end running and preempted requests (" + std::to_string(st.preemptions) + " preemptions), nothing held after");
  }
}

// the prefix cache. Sessions of turns (a turn's prompt = the previous prompt + its output + new
// tokens; or with the last tokens before the output re-rendered, like a chat template's generation prompt; or an edit that
// keeps two thirds) run through an engine with the cache and through one without: every output must be the same (the
// fake's pick hashes the whole history), no snapshot may be restored over overwritten rows or into the wrong slot, and
// with a drafter the row before a resumed position must come from the prompt's token there.
struct PcRun {
  std::map<uint64_t, std::vector<int32_t>> toks;
  core::Stats st;
  int errors = 0, snap_errors = 0, strict_errors = 0, restores = 0, forks = 0, resumes = 0, saves = 0;
  bool ok = true;
  // per-request prefix-cache reports: each request reported once, 0 <= cached < its prompt; their sum
  bool cached_once = true;
  uint64_t cached_sum = 0;
};

PcRun pc_run(bool cache, int slots, int pool_pages, int depth, int snap_cap, bool sampled, int turns = 6, int extra = 0, int sessions = 3, int align = 1, bool live = false) {
  PcRun res;
  FakeModel fm(slots, 1024, 16, kVocab);
  fm.align = align;
  fm.pool_pages = pool_pages;
  fm.depth = depth;
  fm.snap_cap = snap_cap;
  core::BatchConfig bc;
  bc.eos_ids = {0};
  bc.log_interval_s = 0;
  bc.prefix_cache = cache;
  bc.prefix_min_tokens = 16;
  bc.prefix_mark_min = 16;
  bc.prefix_live = live;
  core::BatchEngine e(fm, bc);
  Collect c;
  e.start(&c);
  const auto sys = prompt_of(99, 70);  // every session starts with it (a shared system prompt)
  std::vector<std::vector<int32_t>> prompt(static_cast<size_t>(sessions));
  for (int s = 0; s < sessions; ++s) {
    prompt[static_cast<size_t>(s)] = sys;
    const auto own = prompt_of(s, 20 + 15 * s);
    prompt[static_cast<size_t>(s)].insert(prompt[static_cast<size_t>(s)].end(), own.begin(), own.end());
  }
  uint64_t id = 0;
  size_t want = 0;
  std::map<uint64_t, size_t> plen;
  for (int t = 0; t < turns && res.ok; ++t) {
    std::vector<uint64_t> ids(static_cast<size_t>(sessions));
    for (int s = 0; s < sessions; ++s) {
      core::SamplingParams p;
      p.temperature = sampled ? 0.8f : 0.0f;
      p.has_seed = sampled;
      p.seed = 100 + static_cast<uint64_t>(s * 10 + t);
      p.ignore_eos = true;
      p.max_tokens = 12 + 3 * s + t + extra;
      p.prefix_mark = static_cast<int32_t>(prompt[static_cast<size_t>(s)].size()) - 3;  // a 3-token "generation prompt"
      ids[static_cast<size_t>(s)] = ++id;
      plen[id] = prompt[static_cast<size_t>(s)].size();
      e.submit(id, prompt[static_cast<size_t>(s)], p);
    }
    want += static_cast<size_t>(sessions);
    if (!c.wait(want, 30)) {
      const core::Stats st = e.stats();
      std::fprintf(stderr, "pc_run(cache %d, slots %d, pool %d, depth %d, live %d): turn %d timed out: %u running, %u waiting, kv used %llu\n", cache, slots, pool_pages,
                   depth, live, t, st.running, st.waiting, static_cast<unsigned long long>(st.kv_used_tokens));
      res.ok = false;
      break;
    }
    for (int s = 0; s < sessions; ++s) {
      std::vector<int32_t> out;
      {
        std::lock_guard<std::mutex> lk(c.mu);
        out = c.toks[ids[static_cast<size_t>(s)]];
      }
      auto &P = prompt[static_cast<size_t>(s)];
      std::vector<int32_t> next;
      switch ((s + t) % 3) {
        case 0:  // the whole history
          next = P;
          next.insert(next.end(), out.begin(), out.end());
          break;
        case 1:  // the last 2 prompt tokens and the first output token rendered differently
          next.assign(P.begin(), P.end() - 2);
          next.push_back(7);
          next.insert(next.end(), out.begin() + 1, out.end());
          break;
        default:  // an edit: the last third of the prompt replaced
          next.assign(P.begin(), P.begin() + static_cast<std::ptrdiff_t>(P.size() * 2 / 3));
          break;
      }
      const auto add = prompt_of(1000 + s * 17 + t, 10 + 7 * t);
      next.insert(next.end(), add.begin(), add.end());
      P = std::move(next);
    }
  }
  e.stop();
  res.toks = c.toks;
  res.st = e.stats();
  for (const auto &[rid, len] : plen) {
    const auto it = c.cached.find(rid);
    if (it == c.cached.end() || it->second.size() != 1 || static_cast<size_t>(it->second[0]) >= len) res.cached_once = false;
    else res.cached_sum += static_cast<uint64_t>(it->second[0]);
  }
  res.errors = fm.errors;
  res.snap_errors = fm.snap_errors;
  res.strict_errors = fm.strict_errors;
  res.restores = fm.restores;
  res.forks = fm.forks;
  res.resumes = fm.resumes;
  res.saves = fm.saves;
  return res;
}

void test_prefix_cache() {
  struct Case {
    const char *name;
    int slots, pool_pages, depth, snap_cap;
    bool sampled;
    int turns = 6, extra = 0;  // turns per session (3 sessions), output tokens added to each request's
    bool preempts = false;
    int align = 1;  // the model's snap_align
    bool live = false;  // BatchConfig:prefix_live
  };
  const Case cases[] = {
      {"2 slots, fixed KV, plain", 2, 0, 0, 64, false},
      {"2 slots, fixed KV, plain, live resumes", 2, 0, 0, 64, false, 6, 0, false, 1, true},
      {"2 slots, fixed KV, drafter", 2, 0, 3, 64, false},
      {"2 slots, fixed KV, drafter, live resumes", 2, 0, 3, 64, false, 6, 0, false, 1, true},
      {"3 slots, tight pool, drafter, live resumes", 3, 64, 3, 64, false, 6, 0, false, 1, true},
      {"3 slots, sampled", 3, 0, 0, 64, true},
      {"3 slots, tight pool (parked sequences evicted)", 3, 64, 0, 64, false},
      {"3 slots, tight pool, drafter", 3, 64, 3, 64, false},
      {"3 slots, long outputs on a tight pool (preemptions)", 3, 100, 0, 64, false, 3, 150, true},
      {"3 slots, long outputs on a tight pool, drafter", 3, 100, 3, 64, false, 3, 150, true},
      {"4 slots, store of 3 snapshots", 4, 0, 0, 3, false},
      {"1 slot", 1, 0, 0, 64, false},
      {"3 slots, snapshots at multiples of 8", 3, 0, 0, 64, false, 6, 0, false, 8},
      {"3 slots, snapshots at multiples of 8, drafter", 3, 0, 3, 64, false, 6, 0, false, 8},
  };
  int restores = 0, forks = 0, resumes = 0;
  for (const Case &k : cases) {
    const std::string tag = std::string("prefix cache (") + k.name + "): ";
    const PcRun ref = pc_run(false, k.slots, k.pool_pages, k.depth, k.snap_cap, k.sampled, k.turns, k.extra, 3, k.align, k.live);
    const PcRun got = pc_run(true, k.slots, k.pool_pages, k.depth, k.snap_cap, k.sampled, k.turns, k.extra, 3, k.align, k.live);
    const size_t n = 3 * static_cast<size_t>(k.turns);
    expect(ref.ok && got.ok && ref.toks.size() == n && got.toks.size() == n, tag + "every turn of every session ends");
    if (k.preempts) expect(got.st.preemptions > 0, tag + "the pool preempted running requests (" + std::to_string(got.st.preemptions) + ")");
    expect(got.toks == ref.toks, tag + "every output == the engine without the cache");
    expect(ref.errors == 0 && got.errors == 0, tag + "no model misuse (" + std::to_string(got.snap_errors) + " snapshot, " +
                                                    std::to_string(got.strict_errors) + " drafter-row errors)");
    expect(got.st.prefix_hits > 0 && got.st.prefix_hits < got.st.prefix_queries && ref.st.prefix_hits == 0,
           tag + std::to_string(got.st.prefix_hits) + " of " + std::to_string(got.st.prefix_queries) + " prompt tokens from the cache (" +
               std::to_string(got.resumes) + " live, " + std::to_string(got.restores) + " restores, " + std::to_string(got.forks) + " forks, " +
               std::to_string(got.saves) + " snapshots taken)");
    expect(got.st.prompt_tokens == ref.st.prompt_tokens && got.st.running == 0 && got.st.waiting == 0 && got.st.ttft.count == ref.st.ttft.count,
           tag + "stats: the same prompt tokens and TTFT samples, nothing left running (" + std::to_string(got.st.preemptions) + " preemptions, " +
               std::to_string(ref.st.preemptions) + " without the cache)");
    // place() counts a preempted request's every placement, the reports its first token's: equal without preemptions
    expect(ref.cached_once && got.cached_once && ref.cached_sum == 0 &&
               (got.st.preemptions == 0 ? got.cached_sum == got.st.prefix_hits : got.cached_sum <= got.st.prefix_hits) && got.cached_sum > 0,
           tag + "each request reports its cached prompt tokens once (" + std::to_string(got.cached_sum) + " of the " +
               std::to_string(got.st.prefix_hits) + " hit tokens)");
    restores += got.restores, forks += got.forks, resumes += got.resumes;
  }
  expect(restores > 0 && forks > 0 && resumes > 0, "prefix cache: the matrix ran live resumes (" + std::to_string(resumes) + "), restores (" +
                                                       std::to_string(restores) + ") and forks (" + std::to_string(forks) + ")");
  {  // a prompt-logprobs request never continues a cached prefix (every prompt row's logits are needed)
    FakeModel fm(2, 1024, 16, kVocab);
    fm.snap_cap = 16;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefix_min_tokens = 16;
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams p;
    p.temperature = 0;
    p.ignore_eos = true;
    p.max_tokens = 4;
    const auto pr = prompt_of(5, 100);
    e.submit(1, pr, p);
    c.wait(1);
    const uint64_t h1 = e.stats().prefix_hits;
    p.prompt_logprobs = 1;
    e.submit(2, pr, p);
    c.wait(2);
    p.prompt_logprobs = -1;
    e.submit(3, pr, p);
    c.wait(3);
    e.stop();
    const auto st = e.stats();
    expect(h1 == 0 && st.prefix_hits > 0 && st.prefix_hits < 100 && c.toks[1] == c.toks[2] && c.toks[1] == c.toks[3] && fm.errors == 0,
           "prefix cache: a prompt-logprobs request starts cold (" + std::to_string(st.prefix_hits) + " hit tokens from the third request only)");
  }
  {  // a request cancelled inside a prefill call in parts: its slot's live state is never resumed; the same prompt again
     // continues from the snapshots of its earlier calls (or cold) and gives the uncancelled output
    auto make = [&](FakeModel &fm, core::BatchConfig &bc) {
      fm.split_layers = 4;
      fm.snap_cap = 16;
      fm.pf_us_fixed = 3000;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefill_part_ms = 1e-6;  // every single-call prompt beside decoding runs in parts
      bc.prefix_min_tokens = 16;
    };
    core::SamplingParams p;
    p.temperature = 0;
    p.ignore_eos = true;
    p.max_tokens = 6;
    const auto pr = prompt_of(8, 120);
    std::vector<int32_t> ref;
    {
      FakeModel fm(2, 1024, 256, kVocab);
      core::BatchConfig bc;
      make(fm, bc);
      bc.prefix_cache = false;
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      e.submit(1, pr, p);
      c.wait(1);
      e.stop();
      ref = c.toks[1];
    }
    FakeModel fm(2, 1024, 256, kVocab);
    core::BatchConfig bc;
    make(fm, bc);
    core::BatchEngine e(fm, bc);
    Collect c;
    e.start(&c);
    core::SamplingParams lp = p;
    lp.max_tokens = 100000;
    fm.step_us = 300;
    e.submit(9, prompt_of(9, 30), lp);  // decodes beside: the next prompt runs in parts
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    e.submit(1, pr, p);
    for (int k = 0; k < 400 && fm.split_parts < 2; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    e.cancel(1);
    c.wait(1);
    e.submit(2, pr, p);
    for (int k = 0; k < 2000; ++k) {
      {
        std::lock_guard<std::mutex> lk(c.mu);
        if (c.fin.count(2)) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    e.cancel(9);
    c.wait(3);
    e.stop();
    expect(fm.split_aborts >= 1 && c.fin[1] == core::Finish::cancelled && c.toks[2] == ref && fm.errors == 0 && fm.resumes == 0,
           "prefix cache: a request cancelled inside a split call (" + std::to_string(fm.split_aborts) +
               " aborts) leaves no live state to resume; the same prompt again gives the uncancelled output");
  }
  // In-flight prefix sharing (BatchConfig:prefix_share): requests that arrive together on one long common prefix. The
  // first is prefilled; the others wait for its prefill to pass the common part and fork from the snapshot of the call
  // that ends there, instead of each prefilling it again. Every output == the engine without the cache.
  {
    struct Share {
      const char *name;
      int slots, depth, pool_pages, align;
      bool beside;  // a request decodes all along: the prompts' calls run in parts
    };
    const Share cases[] = {
        {"4 slots", 4, 0, 0, 8, false},
        {"4 slots, drafter", 4, 3, 0, 8, false},
        {"5 slots, a request decoding beside (calls in parts)", 5, 0, 0, 8, true},
        {"5 slots, drafter, a request decoding beside", 5, 3, 0, 8, true},
        {"4 slots, shared pool", 4, 0, 400, 8, false},
        {"4 slots, snapshots anywhere", 4, 0, 0, 1, false},
    };
    const auto doc = prompt_of(5, 600);
    const int N = 4;
    std::vector<std::vector<int32_t>> prompts;
    for (int i = 0; i < N; ++i) {
      auto pr = doc;
      const auto q = prompt_of(50 + i, 12 + 3 * i);
      pr.insert(pr.end(), q.begin(), q.end());
      prompts.push_back(std::move(pr));
    }
    struct Out {
      std::map<uint64_t, std::vector<int32_t>> toks;
      core::Stats st;
      int errors = 0, forks = 0, rows = 0;
      bool ok = false;
    };
    for (const Share &k : cases) {
      auto run = [&](bool cache, bool share) {
        Out o;
        FakeModel fm(k.slots, 4096, 64, kVocab);
        fm.align = k.align;
        fm.depth = k.depth;
        fm.pool_pages = k.pool_pages;
        fm.snap_cap = 64;
        fm.pf_us_fixed = 5000;  // the first request is still prefilling when the others arrive
        if (k.beside) fm.split_layers = 4, fm.step_us = 300, fm.rows_max = 8;  // 5 sequences in a step
        core::BatchConfig bc;
        bc.eos_ids = {0};
        bc.log_interval_s = 0;
        bc.prefix_cache = cache;
        bc.prefix_share = share;
        bc.prefix_min_tokens = 16;
        bc.prefix_mark_min = 16;
        if (k.beside) bc.prefill_part_ms = 1e-6;
        core::BatchEngine e(fm, bc);
        Collect c;
        e.start(&c);
        core::SamplingParams p;
        p.temperature = 0;
        p.ignore_eos = true;
        p.max_tokens = 6;
        size_t want = static_cast<size_t>(N);
        if (k.beside) {
          core::SamplingParams lp = p;
          lp.max_tokens = 1 << 20;
          e.submit(99, prompt_of(9, 30), lp);
          for (int w = 0; w < 2000; ++w) {
            {
              std::lock_guard<std::mutex> lk(c.mu);
              if (!c.toks[99].empty()) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        const int rows0 = [&] { int r = 0; for (const auto &l : fm.pf_log) r += l.n; return r; }();
        for (int i = 0; i < N; ++i) {
          p.prefix_mark = static_cast<int32_t>(prompts[static_cast<size_t>(i)].size()) - 3;
          e.submit(static_cast<uint64_t>(i + 1), prompts[static_cast<size_t>(i)], p);
        }
        o.ok = c.wait(want, 30);
        if (k.beside) {
          e.cancel(99);
          o.ok = c.wait(want + 1, 30) && o.ok;
        }
        e.stop();
        o.toks = c.toks;
        o.toks.erase(99);
        o.st = e.stats();
        o.errors = fm.errors;
        o.forks = fm.forks;
        for (const auto &l : fm.pf_log) o.rows += l.n;
        o.rows -= rows0;
        return o;
      };
      const Out ref = run(false, false), off = run(true, false), on = run(true, true);
      const std::string tag = std::string("prefix sharing (") + k.name + "): ";
      int total = 0;
      for (const auto &pr : prompts) total += static_cast<int>(pr.size());
      // the common part every follower can continue: the document, cut at the alignment (a drafter needs the next token too)
      const int cut = (600 - (k.depth > 0 ? 1 : 0)) / k.align * k.align;
      expect(ref.ok && off.ok && on.ok && on.toks == ref.toks && off.toks == ref.toks && on.toks.size() == static_cast<size_t>(N), tag + "every output == the engine without the cache");
      expect(ref.errors == 0 && off.errors == 0 && on.errors == 0, tag + "no model misuse");
      expect(on.st.prefix_holds == static_cast<uint64_t>(N - 1) && on.forks == N - 1 && on.st.prefix_hits == static_cast<uint64_t>((N - 1) * cut) && on.rows == total - (N - 1) * cut,
             tag + std::to_string(on.st.prefix_holds) + " requests waited and forked at " + std::to_string(cut) + ": " + std::to_string(on.rows) + " rows prefilled of " +
                 std::to_string(total) + " (" + std::to_string(off.rows) + " with sharing off, " + std::to_string(ref.rows) + " without the cache)");
      expect(off.st.prefix_holds == 0 && off.rows > on.rows && ref.rows == total, tag + "with sharing off nothing waits and more rows are prefilled");
      expect(on.st.running == 0 && on.st.waiting == 0 && on.st.prompt_tokens == ref.st.prompt_tokens && on.st.ttft.count == ref.st.ttft.count,
             tag + "stats: the same prompt tokens and TTFT samples, nothing left waiting");
    }
  }
  {  // the leader is cancelled inside its prefill: the request waiting for it is released and gives its own output;
     // and a waiting request keeps a free slot (an unrelated request behind it takes the other one, not its)
    const auto doc = prompt_of(5, 600);
    auto with = [&](int i) {
      auto pr = doc;
      const auto q = prompt_of(50 + i, 12 + 3 * i);
      pr.insert(pr.end(), q.begin(), q.end());
      return pr;
    };
    core::SamplingParams p;
    p.temperature = 0;
    p.ignore_eos = true;
    p.max_tokens = 6;
    auto make = [&](FakeModel &fm, core::BatchConfig &bc, bool cache) {
      fm.align = 8;
      fm.snap_cap = 64;
      fm.pf_us_fixed = 5000;
      bc.eos_ids = {0};
      bc.log_interval_s = 0;
      bc.prefix_cache = cache;
      bc.prefix_min_tokens = 16;
    };
    std::map<uint64_t, std::vector<int32_t>> ref;
    {
      FakeModel fm(3, 4096, 64, kVocab);
      core::BatchConfig bc;
      make(fm, bc, false);
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      e.submit(2, with(1), p);
      e.submit(3, prompt_of(77, 200), p);
      e.submit(4, with(2), p);
      c.wait(3);
      e.stop();
      ref = c.toks;
    }
    {
      FakeModel fm(3, 4096, 64, kVocab);
      core::BatchConfig bc;
      make(fm, bc, true);
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      e.submit(1, with(0), p);
      e.submit(2, with(1), p);
      for (int k = 0; k < 2000 && e.stats().prefix_holds < 1; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      const uint64_t holds = e.stats().prefix_holds;
      e.cancel(1);
      const bool ok = c.wait(2);
      e.stop();
      expect(ok && holds == 1 && c.fin[1] == core::Finish::cancelled && c.toks[2] == ref[2] && fm.errors == 0,
             "prefix sharing: the request a waiting one waits for is cancelled inside its prefill: the waiting request runs and gives its own output");
    }
    {
      FakeModel fm(2, 4096, 64, kVocab);
      core::BatchConfig bc;
      make(fm, bc, true);
      core::BatchEngine e(fm, bc);
      Collect c;
      e.start(&c);
      e.submit(2, with(1), p);
      e.submit(4, with(2), p);  // waits for request 2's prefill
      e.submit(3, prompt_of(77, 200), p);  // unrelated: the one free slot is request 4's, so it waits for a slot
      const bool ok = c.wait(3);
      e.stop();
      const auto st = e.stats();
      expect(ok && st.prefix_holds == 1 && fm.forks == 1 && c.toks[2] == ref[2] && c.toks[3] == ref[3] && c.toks[4] == ref[4] && fm.errors == 0 &&
                 st.prefix_hits == 600,
             "prefix sharing: a waiting request keeps a free slot (2 slots, an unrelated request behind it): it forks at 600, every output as without the cache");
    }
  }
  {  // the warm-up leaves nothing parked: no snapshot held, no page held (the pool may be re-sized after it)
    FakeModel fm(3, 1024, 16, kVocab);
    fm.snap_cap = 16;
    fm.pool_pages = 128;
    core::BatchConfig bc;
    bc.eos_ids = {0};
    bc.log_interval_s = 0;
    bc.prefix_min_tokens = 16;
    core::BatchEngine e(fm, bc);
    std::vector<std::vector<core::WarmRequest>> waves(2);
    core::WarmRequest w;
    w.prompt = prompt_of(3, 64);
    w.params.temperature = 0;
    w.params.max_tokens = 8;
    w.params.ignore_eos = true;
    waves[0].push_back(w);
    waves[1].push_back(w);  // the same again: continues the first one's state
    e.warm_up(waves);
    int held = 0;
    for (const auto &f : fm.fsnaps) held += f.used;
    const auto st = e.stats();
    expect(fm.errors == 0 && fm.resumes + fm.restores > 0 && held == 0 && fm.kv_free_tokens() == 128 * FakeModel::kPage && st.prefix_snapshots == 0 &&
               st.prefix_hits == 0,
           "prefix cache: the warm-up's second wave continues the first; after it no snapshot or page is held and the stats are cleared");
  }
}

// The prefill cost fit (BatchEngine:record_prefill): one call far over its prediction (a first use) stays out of the fit,
// so the prompts after it are not cut into many more parts than their cost asks for.
void test_prefill_fit() {
  FakeModel fm(3, 8192, 256, kVocab);
  fm.split_layers = 32;
  fm.step_us = 1000;
  fm.pf_us_fixed = 20000;  // a call costs 20 ms + 0.1 ms per row: a 100-row prompt in 3-4 parts of 10 ms
  fm.pf_us_row = 100;
  core::BatchConfig bc;
  bc.eos_ids = {0};
  bc.log_interval_s = 0;
  bc.prefix_cache = false;
  bc.prefill_part_ms = 10;
  core::BatchEngine e(fm, bc);
  Collect c;
  e.start(&c);
  core::SamplingParams dp;
  dp.temperature = 0;
  dp.ignore_eos = true;
  dp.max_tokens = 1 << 20;
  e.submit(1, prompt_of(8, 5), dp);  // decodes all along
  for (int w = 0; w < 5000; ++w) {
    {
      std::lock_guard<std::mutex> lk(c.mu);
      if (!c.toks[1].empty()) break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  core::SamplingParams p;
  p.temperature = 0;
  p.ignore_eos = true;
  p.max_tokens = 2;
  std::vector<int> parts;
  size_t done = 0;
  for (int i = 0; i < 9; ++i) {
    if (i == 4) fm.pf_us_once = 600000;  // this call: 0.6 s more (20x its cost)
    const int p0 = fm.split_parts;
    e.submit(static_cast<uint64_t>(10 + i), prompt_of(20 + i, 96 + i), p);
    c.wait(++done, 30);
    parts.push_back(fm.split_parts - p0);
  }
  e.cancel(1);
  c.wait(done + 1, 30);
  e.stop();
  std::string list;
  for (int v : parts) list += (list.empty() ? "" : " ") + std::to_string(v);
  const int before = *std::max_element(parts.begin() + 1, parts.begin() + 4);
  const int after = *std::max_element(parts.begin() + 5, parts.end());
  expect(fm.errors == 0 && before >= 2 && before <= 8 && after <= before + 2,
         "prefill fit: a call 20x over its prediction does not change the parts of the prompts after it (parts per ~100-row prompt: " + list + ")");
}

int main() {
  const std::string dir = write_tokenizer_dir();
  test_env_flags();
  test_sampler();
  test_sampler_fast();
  test_batch_engine();
  test_cands();
  test_tree_spec();
  test_early_events();
  test_join_step();
  test_prefill_merge();
  test_kv_pool();
  test_kv_lend_park();
  test_yarn_tiers();
  test_prefix_cache();
  test_prefill_fit();
  test_warm_up();
  test_constraint(dir);
  test_server(dir);
  test_prefix_mark(dir);
  test_images(dir);
  test_responses(dir);
  test_responses_tools(dir);
  test_cached_usage(dir);
  test_prepare(dir);
  std::filesystem::remove_all(dir);
  std::printf("server: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
