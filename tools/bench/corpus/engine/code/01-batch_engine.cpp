#include "xenithium/core/batch_engine.hpp"
#include "xenithium/util/env.hpp"
#include "xenithium/util/host_mem.hpp"

#include <algorithm>
#include <numeric>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace xenithium::core {

namespace {
using dsec = std::chrono::duration<double>;
constexpr int kTreePathMax = 64;  // a tree row's path tokens (a segment holds <= max_rows rows)

// The leading rows of a draft tree that are a chain: row r's parent is row r - 1 (parents[0] = -1 is the fed token's
// row). At most `rows`, at least 1.
int first_path_rows(const std::vector<int> &parents, int rows) {
  int n = 1;
  while (n < rows && static_cast<size_t>(n) < parents.size() && parents[static_cast<size_t>(n)] == n - 1) ++n;
  return n;
}

void merge(Histogram &to, const Histogram &from) {
  for (int i = 0; i <= Histogram::kBounds; ++i) to.bucket[i] += from.bucket[i];
  to.sum += from.sum;
  to.count += from.count;
}

// Prefill cost per call before the engine has its own samples: a fixed part and a part per row (ms).
constexpr double kPfFixedMs = 29.0, kPfRowMs = 0.32;
constexpr int kPfSamples = 32;
constexpr int kPfQuantum = 64;  // budget rows are a multiple of this
}  // namespace

// Sampling workers: run fn(i) for i < n on the engine thread plus the workers; returns when all ran.
// An index is claimed together with the job's generation (one atomic word), so a worker that wakes
// late can never run a finished job's function on the next job's indices.
struct BatchEngine::Pool {
  std::vector<std::thread> th;
  std::mutex mu;
  std::condition_variable cv, done_cv;
  const std::function<void(size_t)> *fn = nullptr;
  size_t n = 0;
  std::atomic<uint64_t> ticket{0};  // generation << 32 | next index
  size_t done = 0;
  uint64_t gen = 0;
  bool quit = false;

  explicit Pool(int workers) {
    for (int w = 0; w < workers; ++w)
      th.emplace_back([this] {
        uint64_t seen = 0;
        while (true) {
          const std::function<void(size_t)> *f;
          size_t cnt;
          uint64_t g;
          {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return quit || gen != seen; });
            if (quit) return;
            seen = g = gen;
            f = fn;
            cnt = n;
          }
          if (f) run(f, cnt, g);
        }
      });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu);
      quit = true;
    }
    cv.notify_all();
    for (auto &t : th) t.join();
  }
  void run(const std::function<void(size_t)> *f, size_t cnt, uint64_t g) {
    size_t mine = 0;
    while (true) {
      uint64_t cur = ticket.load();
      if ((cur >> 32) != (g & 0xffffffffu) || (cur & 0xffffffffu) >= cnt) break;
      if (!ticket.compare_exchange_weak(cur, cur + 1)) continue;
      (*f)(static_cast<size_t>(cur & 0xffffffffu));
      ++mine;
    }
    if (mine) {
      std::lock_guard<std::mutex> lk(mu);
      done += mine;
      if (done == n) done_cv.notify_all();
    }
  }
  void operator()(size_t cnt, const std::function<void(size_t)> &f) {
    uint64_t g;
    {
      std::lock_guard<std::mutex> lk(mu);
      fn = &f;
      n = cnt;
      done = 0;
      g = ++gen;
      ticket = (g & 0xffffffffu) << 32;
    }
    cv.notify_all();
    run(&f, cnt, g);
    std::unique_lock<std::mutex> lk(mu);
    done_cv.wait(lk, [&] { return done == n; });
    fn = nullptr;
  }
};

BatchEngine::BatchEngine(StepModel &m, BatchConfig cfg) : m_(m), cfg_(std::move(cfg)) {
  if (cfg_.model.max_model_len <= 0 || cfg_.model.max_model_len > m_.capacity())
    cfg_.model.max_model_len = static_cast<int32_t>(m_.capacity());
  cfg_.model.prompt_logprobs = m_.prompt_logprobs();
  for (auto &row : step_n_)  // each shape's first steps record its graphs, untimed
    for (int &n : row) n = -kWarmSteps;
  for (auto &a : ver_n_)
    for (int &n : a) n = -kWarmSteps;
  for (auto &a : pass_n_)
    for (int &n : a) n = -kWarmSteps;
  for (auto &a : deep_n_)
    for (int &n : a) n = -kWarmSteps;
  for (int &k : cur_k_) k = -1;  // no held layout / depth yet at any sequence count
  for (int &d : cur_depth_) d = -1;
  const int slots = std::min(m_.slots(), m_.max_rows());
  stats_.slots = static_cast<uint32_t>(slots);
  for (int s = slots - 1; s >= 0; --s) free_slots_.push_back(s);
  kv_shared_ = m_.kv_pool_tokens() > 0;
  kv_stats();
  pc_.assign(static_cast<size_t>(m_.slots()), SlotCache{});
  pc_on_ = cfg_.prefix_cache && m_.snap_capacity() > 0;  // 
  tiers_ = std::max(1, m_.rope_tiers());
  if (cfg_.sample_threads > 1) pool_ = std::make_unique<Pool>(std::min(cfg_.sample_threads, m_.max_rows()) - 1);
}

BatchEngine::~BatchEngine() {
  stop();
  // the prefix cache's snapshots go back to the model (a later engine over the same model would
  // otherwise find its snapshot pool held by this one's)
  if (pc_on_)
    for (int s = 0; s < static_cast<int>(pc_.size()); ++s) pc_free_snaps(s);
}

void BatchEngine::start(EventSink *sink) {
  sink_ = sink;
  kv_shared_ = m_.kv_pool_tokens() > 0;  // the pool may have been re-sized since the constructor (kv_fit)
  kv_stats();
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = false;
  }
  thread_ = std::thread([this] { loop(); });
}

void BatchEngine::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void BatchEngine::submit(uint64_t id, std::vector<int32_t> prompt, const SamplingParams &params) {
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.waiting++;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    submits_.push_back({id, std::move(prompt), params, clock::now()});
  }
  cv_.notify_one();
}

void BatchEngine::cancel(uint64_t id) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    cancels_.push_back(id);
  }
  cv_.notify_one();
}

Stats BatchEngine::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

double BatchEngine::warm_up(const std::vector<std::vector<WarmRequest>> &waves) {
  struct Sink final : EventSink {
    std::mutex mu;
    std::condition_variable cv;
    size_t done = 0;
    std::string error;
    void on_events(const Event *e, size_t n) override {
      std::lock_guard<std::mutex> lk(mu);
      for (size_t i = 0; i < n; ++i)
        if (e[i].finish != Finish::none) {
          ++done;
          if (e[i].finish != Finish::length && e[i].finish != Finish::stop && error.empty())
            error = std::string("request ended with ") + finish_name(e[i].finish) + (e[i].error.empty() ? "" : ": " + e[i].error);
        }
      cv.notify_all();
    }
  } sink;
  const auto t0 = clock::now();
  start(&sink);
  uint64_t id = 0;
  size_t want = 0;
  for (const auto &wave : waves) {
    for (const auto &r : wave) submit(++id, r.prompt, r.params);
    want += wave.size();
    std::unique_lock<std::mutex> lk(sink.mu);
    sink.cv.wait(lk, [&] { return sink.done >= want; });
  }
  stop();
  sink_ = nullptr;
  pc_clear();  // the scratch requests' sequences are not kept (the pool may be re-sized next: kv_fit)
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    const uint32_t slots = stats_.slots;
    stats_ = Stats{};
    stats_.slots = slots;
  }
  kv_stats();
  pf_samples_.clear();
  pf_skipped_ = false;
  step_ms_.clear();
  log_prefill_tokens_ = 0;
  log_prefill_s_ = 0;
  log_prefill_calls_ = 0;
  if (!sink.error.empty()) throw std::runtime_error("warm-up: " + sink.error);
  return dsec(clock::now() - t0).count() * 1e3;
}

void BatchEngine::parallel(size_t n, const std::function<void(size_t)> &fn) {
  if (n == 0) return;
  if (!pool_ || n == 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  (*pool_)(n, fn);
}

void BatchEngine::record_prefill(int rows, double ms) {
  // A call over 3x its prediction is a first use (a shape's first launch, a page fault), not the cost: one such sample
  // stays out of the fit (as spec_step's step times). The next one is taken whatever it is, so a cost that did change
  // is learnt from its second call.
  if (!pf_samples_.empty() && !pf_skipped_) {
    const auto [a, b] = pf_fit();
    if (ms > 3 * (a + b * rows)) {
      pf_skipped_ = true;
      return;
    }
  }
  pf_skipped_ = false;
  pf_samples_.push_back({rows, ms});
  if (pf_samples_.size() > kPfSamples) pf_samples_.pop_front();
}

int BatchEngine::budget_rows() const {
  const int chunk = m_.prefill_chunk();
  if (cfg_.prefill_stall_ms <= 0) return chunk;
  const auto [a, b] = pf_fit();
  const double rows = (cfg_.prefill_stall_ms - a) / b;
  int r = static_cast<int>(rows) / kPfQuantum * kPfQuantum;
  return std::clamp(r, std::min(kPfQuantum, chunk), chunk);
}

std::pair<double, double> BatchEngine::pf_fit() const {
  // fit ms = a + b rows to the recent calls (least squares); one distinct size scales the defaults
  double a = kPfFixedMs, b = kPfRowMs;
  if (!pf_samples_.empty()) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int lo = std::numeric_limits<int>::max(), hi = 0;
    for (const auto &[r, t] : pf_samples_) {
      sx += r, sy += t, sxx += static_cast<double>(r) * r, sxy += r * t;
      lo = std::min(lo, r), hi = std::max(hi, r);
    }
    const double n = static_cast<double>(pf_samples_.size());
    const double var = sxx - sx * sx / n;
    if (hi - lo >= kPfQuantum && var > 0) {
      b = (sxy - sx * sy / n) / var;
      a = (sy - b * sx) / n;
    }
    if (!(hi - lo >= kPfQuantum && var > 0) || b <= 0.01 || a < 0) {
      // one size seen: the default per-row cost, scaled to the observed calls when they are large
      // enough to say something about it; the fixed part is what is left of the mean call
      const double mr = sx / n, mt = sy / n;
      b = kPfRowMs * (mr >= 2 * kPfQuantum ? mt / (kPfFixedMs + kPfRowMs * mr) : 1.0);
      a = std::max(0.0, mt - b * mr);
    }
  }
  return {a, b};
}

bool BatchEngine::is_stop(const Req &r, int32_t tok) const {
  const auto &st = r.params.stop_token_ids;
  if (std::find(st.begin(), st.end(), tok) != st.end()) return true;
  if (r.params.ignore_eos) return false;
  return std::find(cfg_.eos_ids.begin(), cfg_.eos_ids.end(), tok) != cfg_.eos_ids.end();
}

Finish BatchEngine::emit(Req &r, int32_t tok, TokenLogprob *lp, Event &ev) {
  if (lp) ev.logprobs.push_back(std::move(*lp));
  if (r.sampler) r.sampler->accept(tok);
  if (r.constraint) r.constraint->accept(tok);
  if (ev.n >= Event::kMaxTokens) throw std::logic_error("batch engine: a step kept more tokens than Event::kMaxTokens");
  ev.tokens[ev.n++] = tok;
  if (cfg_.lookup > 0) r.lk_hist.push_back(tok);
  if (kv_shared_ || pc_on_ || tiers_ > 1) r.out.push_back(tok);  // a preempted request re-prefills them; the parked history
  ++r.generated;
  r.next = tok;
  if (is_stop(r, tok)) {
    ev.stop_token = true;
    return Finish::stop;
  }
  if (r.generated >= r.params.max_tokens) return Finish::length;
  if (r.pos >= m_.capacity()) return Finish::length;  // no position left to feed it
  return Finish::none;
}

void BatchEngine::publish_early(std::vector<Event> &evs) {
  if (evs.empty()) return;
  if (sink_) sink_->on_events(evs.data(), evs.size());
}

void BatchEngine::device_lps(const std::vector<const Req *> &reqs, const int32_t *toks, std::vector<TokenLogprob> &lps) {
  std::vector<int32_t> t(reqs.size(), -1);
  int top = -1;
  for (size_t i = 0; i < reqs.size(); ++i)
    if (reqs[i]) {
      t[i] = toks[i];
      top = std::max(top, reqs[i]->params.logprobs);
    }
  lps.assign(reqs.size(), TokenLogprob{});
  if (top < 0) return;
  m_.row_logprobs(static_cast<int>(reqs.size()), t.data(), top, lps.data());
  for (size_t i = 0; i < reqs.size(); ++i)  // each request its own count (a prefix of the best first list)
    if (reqs[i] && lps[i].top.size() > static_cast<size_t>(reqs[i]->params.logprobs)) lps[i].top.resize(static_cast<size_t>(reqs[i]->params.logprobs));
}

void BatchEngine::sample_rows(std::vector<Pick> &picks) {
  parallel(picks.size(), [&](size_t i) {
    Pick &p = picks[i];
    Req &r = *p.r;
    if (p.cn > 0) {  // candidates (no mask, no logprobs); -1 = nothing finite kept: the full row decides below
      p.tok = r.sampler->sample_cands(p.cid, p.cval, p.cn);
      return;
    }
    const uint64_t *allowed = r.constraint ? r.constraint->allowed() : nullptr;
    if (r.sampler) p.tok = r.sampler->sample(p.row, m_.vocab(), nullptr, 0, allowed);
    if (r.params.logprobs >= 0) {
      Sampler::raw_logprobs(p.row, m_.vocab(), p.tok, r.params.logprobs, p.lp);
      p.has_lp = true;
    }
  });
}

int BatchEngine::cands_for(const std::vector<const Req *> &reqs) const {
  const int tmax = m_.cand_tiles();
  if (tmax <= 0) return 0;
  int T = 0;
  for (const Req *r : reqs) {
    if (!r->host) continue;
    // logprobs of a candidate row come from the model's records (device_logprobs)
    const bool lp_ok = r->params.logprobs < 0 || (m_.device_logprobs() && r->params.logprobs <= kDevLpTop);
    if (!r->sampler || r->constraint || !lp_ok || !r->sampler->cands_ok(tmax)) return 0;
    T = std::max(T, r->params.top_k);
  }
  return T;
}

void BatchEngine::spec_step(const std::vector<size_t> &dec_in, std::vector<std::pair<size_t, Finish>> &done, uint64_t &gen_add) {
  const int N = static_cast<int>(dec_in.size());
  const auto t_step = clock::now();  // the step's time per (sequences, rows per sequence)
  std::vector<size_t> dec = dec_in;
  std::vector<int> rr(static_cast<size_t>(N)), off(static_cast<size_t>(N));  // rows and first row of each segment
  int sr = 1;
  const bool uneq = unequal_on();
  if (!uneq) {
    // rows per segment: the fed token + the drafts the step's shape allows, as many as every request
    // has (segments are the same size) and as fit in each one's capacity
    sr = 1 + (N == 1 ? std::max(m_.draft_depth(1), m_.wide_depth()) : m_.draft_depth(N));  // a held wide block
    for (size_t i : dec) {
      const Req &r = *active_[i];
      sr = std::min<int>(sr, 1 + static_cast<int>(r.drafts.size()));
      sr = static_cast<int>(std::min<int64_t>(sr, std::max<int64_t>(1, m_.capacity() - r.pos)));
    }
    sr = std::max(1, std::min(sr, m_.step_rows_at(N) / N));
    std::fill(rr.begin(), rr.end(), sr);
  } else {
    // Unequal segments. A request carries its drafts (the layout pick_layout chose at the last accept: B - 1 drafts
    // or none); the drafting ones first with B rows each, the rest one row (StepModel::unequal_segments); a request
    // with fewer drafts than B - 1 (its capacity, a fresh prompt's first drafts) runs one row, and the lowest-yield
    // drafting requests drop theirs while the rows exceed step_rows_at(N).
    // Two classes (StepModel::unequal_classes; pick_deep's layouts): the rest B2 rows each, 1 <= B2 < B.
    std::vector<int> d(static_cast<size_t>(N)), room(static_cast<size_t>(N));
    const int R = m_.step_rows_at(N);
    int B = 1;
    for (int k = 0; k < N; ++k) {
      const Req &r = *active_[dec[static_cast<size_t>(k)]];
      room[static_cast<size_t>(k)] = static_cast<int>(std::min<int64_t>(R, std::max<int64_t>(1, m_.capacity() - r.pos)));  // rows its capacity takes
      d[static_cast<size_t>(k)] = static_cast<int>(std::min<int64_t>(static_cast<int64_t>(r.drafts.size()), std::max<int64_t>(0, m_.capacity() - r.pos - 1)));
      B = std::max(B, 1 + d[static_cast<size_t>(k)]);
    }
    B = std::min(B, R);
    // rows per request at segment sizes (b, b2): b for those holding b - 1 drafts or more (a request's drafts are a chain:
    // it may run fewer of them), b2 for the rest (one row; two classes: a request holding fewer than b2 - 1 drafts pads
    // its rows with its fed token, drafts like any other that are right by chance only); while the rows exceed
    // R, the b-row request with the fewest expected tokens runs b2 rows. -1: b2 rows do not fit (the step's rows
    // or a request's capacity).
    auto lay = [&](int b, int b2, std::vector<int> &out) {
      int total = 0;
      for (int k = 0; k < N; ++k) {
        total += out[static_cast<size_t>(k)] = d[static_cast<size_t>(k)] >= b - 1 && b > 1 ? b : b2;
        if (out[static_cast<size_t>(k)] > room[static_cast<size_t>(k)]) return -1;
      }
      while (total > R) {
        int w = -1;
        double wv = 0;
        for (int k = 0; k < N; ++k)
          if (out[static_cast<size_t>(k)] > b2) {
            const double v = expected_tokens(*active_[dec[static_cast<size_t>(k)]], b - 1);
            if (w < 0 || v < wv) w = k, wv = v;
          }
        if (w < 0) return b2 > 1 ? -1 : total;
        total -= out[static_cast<size_t>(w)] - b2;
        out[static_cast<size_t>(w)] = b2;
      }
      return total;
    };
    lay(B, 1, rr);
    // B = the most drafts any request holds demotes the requests holding fewer to one row (a request that
    // joined at another sequence count, 1 x 8 + 1 x 1 at 2 streams). With the model's cost, (B, B2) are the segment
    // sizes among the held draft counts with the most expected tokens per ms of the verify (the drafter pass follows
    // either way).
    if (B > 2 && m_.spec_cost_rows(N, N, true) > 0) {
      auto value = [&](const std::vector<int> &x, int total) {
        double tok = 0;
        for (int k = 0; k < N; ++k) tok += expected_tokens(*active_[dec[static_cast<size_t>(k)]], std::min(d[static_cast<size_t>(k)], x[static_cast<size_t>(k)] - 1));
        const double ms = m_.spec_cost_rows(N, total, true);
        return ms > 0 ? tok / ms : 0.0;
      };
      int tb = 0;
      for (int k = 0; k < N; ++k) tb += rr[static_cast<size_t>(k)];
      double best = value(rr, tb);
      std::vector<int> x(static_cast<size_t>(N));
      const bool classes = m_.unequal_classes();
      for (int k = -1; k < N; ++k) {  // b: B, then each held count below it
        const int b = k < 0 ? B : 1 + d[static_cast<size_t>(k)];
        if (b < 2 || (k >= 0 && b >= B)) continue;
        for (int b2 = 1; b2 < (classes ? b : 2); ++b2) {  // b2: one row, then (two classes) every size below b
          if (k < 0 && b2 == 1) continue;  // (B, 1): rr
          const int t = lay(b, b2, x);
          if (t < 0) continue;
          const double v = value(x, t);
          if (v > best * (1 + 1e-9)) best = v, rr = x;
        }
      }
    }
    // the widest segments first (stable: the others keep their order)
    const int rmax = *std::max_element(rr.begin(), rr.end());
    std::vector<int> ord(static_cast<size_t>(N));
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_partition(ord.begin(), ord.end(), [&](int k) { return rr[static_cast<size_t>(k)] == rmax && rmax > 1; });
    std::vector<size_t> dec2;
    std::vector<int> rr2;
    for (int k : ord) dec2.push_back(dec[static_cast<size_t>(k)]), rr2.push_back(rr[static_cast<size_t>(k)]);
    dec = std::move(dec2), rr = std::move(rr2);
    sr = rr[0];
  }
  int rows = 0;
  for (int k = 0; k < N; ++k) off[static_cast<size_t>(k)] = rows, rows += rr[static_cast<size_t>(k)];
  // a segment's rows that carry its request's drafts (a padded segment of the second class has fewer than its rows)
  std::vector<int> rv(static_cast<size_t>(N));
  for (int k = 0; k < N; ++k) rv[static_cast<size_t>(k)] = std::min(rr[static_cast<size_t>(k)], 1 + static_cast<int>(active_[dec[static_cast<size_t>(k)]]->drafts.size()));
  // where the model cannot verify trees in this step (StepModel::verifies_trees), a request's tree
  // goes as its first path: the rows past it are padding, as a padded segment's
  bool held_tree = false;
  for (int k = 0; k < N; ++k) held_tree |= !active_[dec[static_cast<size_t>(k)]]->tree.empty();
  bool step_trees = true;
  if (held_tree) {
    std::vector<int> step_slots(static_cast<size_t>(N));
    for (int k = 0; k < N; ++k) step_slots[static_cast<size_t>(k)] = active_[dec[static_cast<size_t>(k)]]->slot;
    step_trees = m_.verifies_trees(step_slots, rr);
  }
  if (!step_trees)
    for (int k = 0; k < N; ++k) {
      const std::vector<int> &parents = active_[dec[static_cast<size_t>(k)]]->tree;
      if (!parents.empty()) rv[static_cast<size_t>(k)] = first_path_rows(parents, rv[static_cast<size_t>(k)]);
    }
  std::vector<int32_t> toks(static_cast<size_t>(rows));
  std::vector<StepModel::SpecSeg> segs;
  bool host = false;
  // a request whose drafts are a tree verifies it as one, in a step of any number of sequences (the model drafts
  // trees only where it verifies them): a greedy request, and a sampled one too (the walk takes the sampler's draw
  // at each row: the draws a chain would make along the same tokens); constrained requests and the lookup's drafts
  // verify as a chain (exact either way, the rows of a chain follow the tokens before them)
  std::vector<uint8_t> is_tree(static_cast<size_t>(N), 0);
  std::vector<int> paths(static_cast<size_t>(rows), 0);
  for (int k = 0; k < N; ++k) {
    const Req &r = *active_[dec[static_cast<size_t>(k)]];
    int32_t *t = toks.data() + off[static_cast<size_t>(k)];
    t[0] = r.next;
    for (int j = 1; j < rr[static_cast<size_t>(k)]; ++j) t[j] = j < rv[static_cast<size_t>(k)] ? r.drafts[static_cast<size_t>(j - 1)] : t[0];
    const bool tree = step_trees && rr[static_cast<size_t>(k)] > 1 && !r.constraint && !r.lk_used &&
                      r.tree.size() >= static_cast<size_t>(rr[static_cast<size_t>(k)]);  // (a prefix of the tree's rows)
    is_tree[static_cast<size_t>(k)] = tree;
    segs.push_back({r.slot, t, rr[static_cast<size_t>(k)], tree ? r.tree.data() : nullptr});
    host |= r.host;
  }
  m_.verify(segs);
  std::vector<int32_t> dev(static_cast<size_t>(rows));
  m_.picks(rows, dev.data());
  // when every host request reads only its top_k, each row's candidates (both cards' top T tiles) instead of the
  // whole rows; cn = candidates per row, 0 = full rows in logits_
  int cn = 0;
  if (host) {
    std::vector<const Req *> hr;
    for (int k = 0; k < N; ++k) hr.push_back(active_[dec[static_cast<size_t>(k)]].get());
    if (const int T = cands_for(hr); T > 0) cn = m_.logits_cands(rows, T, cand_ids_, cand_vals_);
    else m_.logits(rows, logits_);
  }
  // device logprobs: every row of a greedy request that wants them (its token there is the pick)
  std::vector<TokenLogprob> dlp;
  bool any_dlp = false;
  {
    std::vector<const Req *> lr(static_cast<size_t>(rows), nullptr);
    for (int k = 0; k < N; ++k) {
      const Req &r = *active_[dec[static_cast<size_t>(k)]];
      if (!r.host && r.params.logprobs >= 0)
        for (int j = 0; j < rr[static_cast<size_t>(k)]; ++j) lr[static_cast<size_t>(off[static_cast<size_t>(k)] + j)] = &r, any_dlp = true;
    }
    if (any_dlp) device_lps(lr, dev.data(), dlp);
  }
  const auto V = static_cast<size_t>(m_.vocab());
  // every verify row of a sampled request is prepared at once (its kept set, as if the drafts before it were
  // accepted: the only case in which the chain below reads it) instead of one row after the other; the draws stay
  // in order on its RNG (the same tokens). Constrained requests keep the row-by-row path (their mask follows the
  // grammar state). The token-independent half of a host row's logprobs too.
  std::vector<uint8_t> prepared(static_cast<size_t>(N), 0), lp_ready(static_cast<size_t>(N), 0);
  auto prepare_all = [&] {
    std::vector<std::pair<int, int>> jobs;
    for (int k = 0; k < N; ++k) {
      Req &r = *active_[dec[static_cast<size_t>(k)]];
      prepared[static_cast<size_t>(k)] = lp_ready[static_cast<size_t>(k)] = 0;
      if (!r.host) continue;
      prepared[static_cast<size_t>(k)] = r.sampler && !r.constraint;
      lp_ready[static_cast<size_t>(k)] = r.params.logprobs >= 0;
      if (!prepared[static_cast<size_t>(k)] && !lp_ready[static_cast<size_t>(k)]) continue;
      const int rk = rr[static_cast<size_t>(k)];
      if (r.prep.size() < static_cast<size_t>(rk)) r.prep.resize(static_cast<size_t>(rk));
      if (r.prep_lp.size() < static_cast<size_t>(rk)) r.prep_lp.resize(static_cast<size_t>(rk));
      for (int j = 0; j < rk; ++j) jobs.push_back({k, j});
    }
    parallel(jobs.size(), [&](size_t q) {
      const auto [k, j] = jobs[q];
      Req &r = *active_[dec[static_cast<size_t>(k)]];
      const size_t row_i = static_cast<size_t>(off[static_cast<size_t>(k)]) + static_cast<size_t>(j);
      if (cn > 0) {  // every host request here is a plain top_k sampler (cands_for)
        r.sampler->prepare_cands(cand_ids_.data() + row_i * cn, cand_vals_.data() + row_i * cn, cn, r.prep[static_cast<size_t>(j)]);
        return;
      }
      const float *row = logits_.data() + row_i * V;
      const int32_t *t = toks.data() + off[static_cast<size_t>(k)];
      if (prepared[static_cast<size_t>(k)] && is_tree[static_cast<size_t>(k)]) {  // a tree row follows its path's tokens
        int32_t path[kTreePathMax];
        int d = 0;
        for (int x = j; x > 0 && d < kTreePathMax; x = r.tree[static_cast<size_t>(x)]) path[d++] = t[x];
        std::reverse(path, path + d);
        r.sampler->prepare_row(row, m_.vocab(), path, d, r.prep[static_cast<size_t>(j)]);
      } else if (prepared[static_cast<size_t>(k)]) {
        r.sampler->prepare_row(row, m_.vocab(), t + 1, j, r.prep[static_cast<size_t>(j)]);
      }
      if (lp_ready[static_cast<size_t>(k)]) Sampler::raw_logprob_stats(row, m_.vocab(), r.params.logprobs, r.prep_lp[static_cast<size_t>(j)]);
    });
  };
  if (host && (rows > N || cn > 0)) prepare_all();
  if (cn > 0) {  // a kept set left empty (every candidate -inf) is decided on the full row: fetch it and prepare again
    bool empty = false;
    for (int k = 0; k < N && !empty; ++k) {
      const Req &r = *active_[dec[static_cast<size_t>(k)]];
      for (int j = 0; r.host && j < rr[static_cast<size_t>(k)]; ++j) empty |= r.prep[static_cast<size_t>(j)].idx.empty() && r.prep[static_cast<size_t>(j)].pick < 0;
    }
    if (empty) {
      cn = 0;
      m_.logits(rows, logits_);
      if (rows > N) prepare_all();
      else std::fill(prepared.begin(), prepared.end(), 0);
    }
  }
  std::vector<int32_t> next(static_cast<size_t>(rows));
  std::vector<StepModel::SpecAccept> acc(static_cast<size_t>(N));
  std::vector<Finish> fin(static_cast<size_t>(N), Finish::none);
  std::vector<Event> evs(static_cast<size_t>(N));
  // a candidate row's logprobs, scored once its token is drawn: (event index, row, token)
  struct LpLater {
    size_t at;
    size_t row;
    int32_t tok;
  };
  std::vector<std::vector<LpLater>> lp_later(static_cast<size_t>(N));
  // the requests' rows: independent of each other (own sampler / constraint), so in parallel
  parallel(static_cast<size_t>(N), [&](size_t k) {
    Req &r = *active_[dec[k]];
    const int rk = rr[k];
    const int32_t *t = toks.data() + off[k];
    int32_t *nx = next.data() + off[k];
    Event &e = evs[k];
    e.id = r.id;
    int keep = 0;
    Finish f = Finish::none;
    if (is_tree[k]) {  // from the root, the child whose token is the row's pick (greedy: the device picks)
      const int *par = r.tree.data();
      int *path = paths.data() + off[k];
      for (int j = 0;;) {
        const size_t row_i = static_cast<size_t>(off[k]) + static_cast<size_t>(j);
        int32_t tok = dev[row_i];
        if (r.sampler) {  // the sampler's draw at this row (prepared along the row's path)
          const float *srow = r.host && cn == 0 ? logits_.data() + row_i * V : nullptr;
          tok = prepared[k] ? r.sampler->speculate_row(r.prep[static_cast<size_t>(j)], srow, m_.vocab(), -1) : r.sampler->speculate(srow, m_.vocab(), -1);
        }
        TokenLogprob lp;
        if (r.params.logprobs >= 0) {  // a greedy request's logprobs: the device's, or the host's from the row (as a chain's)
          const float *row = r.host && cn == 0 ? logits_.data() + row_i * V : nullptr;
          if (r.host && cn > 0) lp_later[k].push_back({e.logprobs.size(), row_i, tok});
          else if (r.host && lp_ready[k]) Sampler::raw_logprobs_from(row, m_.vocab(), tok, r.prep_lp[static_cast<size_t>(j)], lp);
          else if (r.host) Sampler::raw_logprobs(row, m_.vocab(), tok, r.params.logprobs, lp);
          else lp = dlp[row_i];
        }
        path[keep] = j;
        nx[keep++] = tok;
        r.pos += 1;
        f = emit(r, tok, r.params.logprobs >= 0 ? &lp : nullptr, e);
        int c = -1;
        for (int x = j + 1; x < rk && c < 0; ++x)
          if (par[x] == j && t[x] == tok) c = x;
        if (f != Finish::none || c < 0) break;
        j = c;
      }
      fin[k] = f;
      acc[k].n = keep;
      acc[k].next = nx;
      acc[k].path = path;
      return;
    }
    for (int j = 0; j < rk; ++j) {
      // row j's distribution gives the token after t[j]; the draft there is t[j + 1]
      const size_t row_i = static_cast<size_t>(off[k]) + static_cast<size_t>(j);
      const float *row = r.host && cn == 0 ? logits_.data() + row_i * V : nullptr;  // candidates = prepared rows
      const int32_t draft = j + 1 < rk ? t[j + 1] : -1;
      int32_t tok = dev[row_i];
      if (r.sampler)
        tok = prepared[k] ? r.sampler->speculate_row(r.prep[static_cast<size_t>(j)], row, m_.vocab(), draft)
                          : r.sampler->speculate(row, m_.vocab(), draft, r.constraint ? r.constraint->allowed() : nullptr);
      TokenLogprob lp;
      if (r.params.logprobs >= 0 && r.host && cn > 0) {
        lp_later[k].push_back({e.logprobs.size(), row_i, tok});  // from the model's records after the walk
      } else if (r.params.logprobs >= 0) {
        if (r.host && lp_ready[k]) Sampler::raw_logprobs_from(row, m_.vocab(), tok, r.prep_lp[static_cast<size_t>(j)], lp);
        else if (r.host) Sampler::raw_logprobs(row, m_.vocab(), tok, r.params.logprobs, lp);
        else lp = dlp[row_i];
      }
      nx[keep++] = tok;
      r.pos += 1;
      f = emit(r, tok, r.params.logprobs >= 0 ? &lp : nullptr, e);
      if (f != Finish::none || tok != draft) break;  // finished, or the draft was not this token
    }
    fin[k] = f;
    acc[k].n = keep;
    acc[k].next = nx;
  });
  if (std::any_of(lp_later.begin(), lp_later.end(), [](const auto &v) { return !v.empty(); })) {
    std::vector<const Req *> lr(static_cast<size_t>(rows), nullptr);
    std::vector<int32_t> lt(static_cast<size_t>(rows), -1);
    for (int k = 0; k < N; ++k)
      for (const LpLater &x : lp_later[static_cast<size_t>(k)]) lr[x.row] = active_[dec[static_cast<size_t>(k)]].get(), lt[x.row] = x.tok;
    std::vector<TokenLogprob> lps;
    device_lps(lr, lt.data(), lps);
    for (int k = 0; k < N; ++k)
      for (const LpLater &x : lp_later[static_cast<size_t>(k)]) evs[static_cast<size_t>(k)].logprobs[x.at] = std::move(lps[x.row]);
  }
  // the drafts a segment's rows confirmed (a chain: the leading drafts equal to the tokens emitted at their rows;
  // a tree: the walked rows past the root)
  auto confirmed = [&](int k) {
    const auto &a = acc[static_cast<size_t>(k)];
    if (is_tree[static_cast<size_t>(k)]) return a.n - 1;
    const int rk = rr[static_cast<size_t>(k)];
    const int32_t *t = toks.data() + off[static_cast<size_t>(k)];
    int accepted = 0;
    while (accepted < a.n && accepted + 1 < rk && a.next[accepted] == t[accepted + 1]) ++accepted;
    return accepted;
  };
  std::vector<size_t> going_i;
  for (int k = 0; k < N; ++k)
    if (fin[static_cast<size_t>(k)] == Finish::none) going_i.push_back(dec[static_cast<size_t>(k)]);
  const int going = static_cast<int>(going_i.size());
  // the acceptance of this step first (it informs the depth of the next)
  int wide_acc = -1;  // the lone going request's accepted drafts this step
  for (int k = 0; k < N; ++k) {
    const int rk = rv[static_cast<size_t>(k)];  // the rows of its drafts (padded rows are not the drafter's)
    if (rk < 2) continue;
    auto &a = acc[static_cast<size_t>(k)];
    if (fin[static_cast<size_t>(k)] != Finish::none) continue;  // a stop cuts the row walk short: not an acceptance
    (void)a;
    const int accepted = confirmed(k);
    if (!active_[dec[static_cast<size_t>(k)]]->lk_used) wide_acc = std::min(accepted, rk - 1);
    Req &r = *active_[dec[static_cast<size_t>(k)]];
    if (r.lk_used) continue;  // the lookup's drafts: the drafter's acceptance is not observed
    depth_observe(r, rk - 1, std::min(accepted, rk - 1));
    if (cfg_.lookup > 0) r.df_yield += (static_cast<float>(std::min(accepted, rk - 1) + 1) - r.df_yield) * kLkAlpha;
  }
  // one depth for all (pick_depth); where that is no drafts and the model takes unequal segments, each going
  // request its own depth (pick_layout: some draft, the rest run one row)
  std::vector<int> req_depth;
  int depth = going > 0 ? pick_depth(going_i, m_.draft_depth(going)) : 0;
  // one going request that accepted wide_from() or more drafts drafts the wide block next
  if (going == 1 && depth > 0 && m_.wide_from() > 0 && m_.wide_depth() > depth && wide_acc >= m_.wide_from()) depth = m_.wide_depth();
  explore_now_ = false;
  lay_steady_ = true;  // every request drafting after this step drafted after the last one too
  if (uneq && going > 0 && depth == 0) {
    req_depth = pick_layout(going_i);
    for (size_t j = 0; j < going_i.size(); ++j) {
      Req &r = *active_[going_i[j]];
      lay_steady_ &= req_depth[j] == 0 || r.lay_draft;
      if (!explore_now_) r.lay_draft = req_depth[j] > 0;
    }
  } else if (uneq) {
    req_depth.assign(static_cast<size_t>(going), depth);
    // where that depth leaves rows free, some requests draft deeper (two classes)
    if (depth > 0 && going > 1 && m_.unequal_classes()) pick_deep(going_i, depth, req_depth);
  }
  if (uneq) depth = 0;  // the per-request depths rule (req_depth)
  // lookup drafting (BatchConfig::lookup): this step's proposals scored, the next ones made; the lookup drafts the next
  // step when it leads for every going request (one drafter per step: a verify's segments have the same rows; with
  // unequal segments at the largest picked depth, for all)
  int lk_depth = depth;
  for (int d : req_depth) lk_depth = std::max(lk_depth, d);
  bool lk_step = cfg_.lookup > 0 && lk_depth > 0 && going > 0;
  std::vector<std::vector<int32_t>> lk_props(static_cast<size_t>(N));
  if (cfg_.lookup > 0)
    for (int k = 0; k < N; ++k) {
      Req &r = *active_[dec[static_cast<size_t>(k)]];
      if (fin[static_cast<size_t>(k)] != Finish::none) continue;
      lk_score(r);
      if (lk_depth > 0) lk_props[static_cast<size_t>(k)] = lk_propose(r, lk_depth);
      lk_step = lk_step && !lk_props[static_cast<size_t>(k)].empty() && r.lk_yield > r.df_yield;
    }
  size_t gi = 0;  // the going requests in order (req_depth)
  for (int k = 0; k < N; ++k) {
    auto &a = acc[static_cast<size_t>(k)];
    const bool goes = fin[static_cast<size_t>(k)] == Finish::none;
    a.depth = goes && !lk_step ? (uneq ? req_depth[gi] : depth) : 0;
    gi += goes;
    gen_add += static_cast<uint64_t>(a.n);
    const int rk = rv[static_cast<size_t>(k)];
    if (rk > 1) {
      const int accepted = confirmed(k);  // drafts equal to the token emitted at their row
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.spec_drafts++;
      stats_.spec_draft_tokens += static_cast<uint64_t>(rk - 1);
      const int got = std::min(accepted, rk - 1);
      stats_.spec_accepted_tokens += static_cast<uint64_t>(got);
      for (int j = 0; j < got && j < Stats::kSpecPositions; ++j) stats_.spec_accepted_at[j]++;
    }
  }
  for (int k = 0; k < N; ++k) {  // coupled drafts for sampled requests (argmax for the rest)
    const Req &r = *active_[dec[static_cast<size_t>(k)]];
    const auto dn = r.sampler ? r.sampler->draft_noise() : Sampler::DraftNoise{};
    m_.draft_sampling(r.slot, dn.key, dn.pos, dn.inv_t);
  }
  {  // the going requests' tokens go to the client here, before the drafter pass below
    std::vector<Event> now;
    for (int k = 0; k < N; ++k)
      if (fin[static_cast<size_t>(k)] == Finish::none) now.push_back(std::move(evs[static_cast<size_t>(k)]));
    publish_early(now);
  }
  const auto t_acc0 = clock::now();
  auto drafts = m_.accept(acc);
  const auto t_acc1 = clock::now();
  bool drafts_after = false;  // the accept ran a drafting pass
  for (const auto &a : acc) drafts_after |= a.depth > 0;
  for (int k = 0; k < N; ++k) {
    const size_t i = dec[static_cast<size_t>(k)];
    Req &r = *active_[i];
    const int rk = rv[static_cast<size_t>(k)];
    if (rk > 1 && r.lk_used) {  // the step's drafts were the lookup's: its accepted tokens
      const int accepted = confirmed(k);
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.lookup_steps++;
      stats_.lookup_accepted_tokens += static_cast<uint64_t>(std::min(accepted, rk - 1));
    }
    r.lk_used = lk_step && fin[static_cast<size_t>(k)] == Finish::none;
    if (r.lk_used) r.drafts = std::move(lk_props[static_cast<size_t>(k)]);
    else r.drafts = k < static_cast<int>(drafts.size()) ? std::move(drafts[static_cast<size_t>(k)]) : std::vector<int32_t>{};
    r.tree = r.lk_used ? std::vector<int>{} : m_.draft_parents(r.slot);
    if (fin[static_cast<size_t>(k)] != Finish::none) {  // a finishing event at the iteration's end (the going ones went above)
      Event &e = evs[static_cast<size_t>(k)];
      e.finish = fin[static_cast<size_t>(k)];
      events_.push_back(std::move(e));
      done.push_back({i, fin[static_cast<size_t>(k)]});
    }
  }
  const bool recorded = m_.step_recorded();  // a step that recorded a graph is not timed
  if (uneq && N <= kSeqMax && rows <= kHetRows && !recorded) {  // the verify per (sequences, rows), the accept per drafts after
    const double pass = std::chrono::duration<double, std::milli>(t_acc1 - t_acc0).count();
    const double ver = std::chrono::duration<double, std::milli>(clock::now() - t_step).count() - pass;
    auto ema = [](double ms, double &e, int &n) {
      if (n < 0) {
        ++n;
      } else if (n == 0 || ms < 3 * e) {
        e = n == 0 ? ms : e + (ms - e) * (n < 8 ? 1.0 / (n + 1) : 0.125);
        ++n;
      }
    };
    ema(ver, ver_ms_[N][rows], ver_n_[N][rows]);
    // a drafting pass is timed only when its requests drafted the last step too: one that starts drafting also reads
    // the context rows deferred while it ran plain, once per stint.
    // The accept per deepest draft after it (pick_deep's cost) is timed on steps whose requests all drafted before
    // and after (no deferred rows to catch up in the pass); a step of two classes stays out of pick_layout's average
    int dmax = 0, dmin = kDepthMax + 1;
    bool all = true;
    for (int k = 0; k < N; ++k) {
      const int dk = acc[static_cast<size_t>(k)].depth;
      dmax = std::max(dmax, dk), all &= dk > 0 && rv[static_cast<size_t>(k)] > 1;
      if (dk > 0) dmin = std::min(dmin, dk);
    }
    if (!explore_now_ && (!drafts_after || lay_steady_) && dmax <= dmin) ema(pass, pass_ms_[N][drafts_after], pass_n_[N][drafts_after]);
    if (all && dmax <= kDepthMax) ema(pass, deep_ms_[N][dmax], deep_n_[N][dmax]);
  }
  if (rows == N * sr && N <= kSeqMax && sr <= kDepthMax + 1 && !recorded) {  // this shape's step time (verify, rows, pass 1, chains)
    // The first steps of a shape record its graphs: kWarmSteps are not timed, and a step over 3x the
    // average (a recording on a new segment layout, a page fault) is dropped.
    const double ms = std::chrono::duration<double, std::milli>(clock::now() - t_step).count();
    double &e = spec_ms_[N][sr];
    int &n = step_n_[N][sr];
    if (n < 0) {
      ++n;
    } else if (n == 0 || ms < 3 * e) {
      e = n == 0 ? ms : e + (ms - e) * (n < 8 ? 1.0 / (n + 1) : 0.125);
      ++n;
    }
  }
  ++spec_steps_;
}

// Position j's acceptance given j - 1 accepted, per request (recent steps) and over every request
// (the prior of a request's unseen positions).
void BatchEngine::depth_observe(Req &r, int drafted, int accepted) {
  // Running means that start from a prior worth kPriorN observations (the engine's mean for a request, 0.6 for the
  // engine), then follow the last ~20 (requests) / ~50 (engine) observations: one early rejection must not zero a
  // position (a request at depth 0 sees its positions only on the exploring steps).
  constexpr float kPriorN = 4.f;
  for (int j = 0; j < drafted && j <= accepted && j < kDepthMax; ++j) {
    const float x = j < accepted ? 1.f : 0.f;
    float &g = glob_p_[j], &p = r.acc_p[j];
    if (g < 0) g = 0.6f;
    if (p < 0) p = g, r.acc_n[j] = 0;
    const float a = std::max(0.05f, 1.f / (kPriorN + static_cast<float>(++r.acc_n[j])));
    p += (x - p) * a;
    g += (x - g) * 0.02f;
  }
}

// Lookup drafting (BatchConfig::lookup). A window = the `lookup` tokens ending at a position; lk_idx maps its hash to the
// position after its latest occurrence whose next token is known (windows ending at the last token wait for it), so the
// last window's entry is always an earlier occurrence. Hash collisions only cost a wrong proposal (drafts are verified).
namespace {
uint64_t lk_hash(const int32_t *t, int n) {
  uint64_t h = 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < n; ++i) h = (h ^ static_cast<uint32_t>(t[i])) * 0x100000001b3ull + (h >> 29);
  return h;
}
}  // namespace

void BatchEngine::lk_index(Req &r) {
  const size_t n = static_cast<size_t>(cfg_.lookup), L = r.lk_hist.size();
  if (r.lk_indexed < n) r.lk_indexed = n;
  for (; r.lk_indexed < L; ++r.lk_indexed)  // the window ending at lk_indexed - 1, followed by lk_hist[lk_indexed]
    r.lk_idx[lk_hash(r.lk_hist.data() + r.lk_indexed - n, static_cast<int>(n))] = static_cast<int32_t>(r.lk_indexed);
}

// A proposal made at position p yields 1 + its tokens equal to the ones that came at p, p + 1, ... (what a step it drafted
// keeps); scored once the tokens decide it, also when the drafter drafted that step.
void BatchEngine::lk_score(Req &r) {
  const size_t L = r.lk_hist.size();
  while (!r.lk_pend.empty()) {
    const auto &[p, prop] = r.lk_pend.front();
    size_t a = 0;
    while (a < prop.size() && p + a < L && prop[a] == r.lk_hist[p + a]) ++a;
    if (a < prop.size() && p + a >= L) break;  // undecided: the tokens after p are not all there yet
    r.lk_yield += (static_cast<float>(a + 1) - r.lk_yield) * kLkAlpha;
    r.lk_pend.pop_front();
  }
}

std::vector<int32_t> BatchEngine::lk_propose(Req &r, int k) {
  const size_t n = static_cast<size_t>(cfg_.lookup), L = r.lk_hist.size();
  std::vector<int32_t> prop;
  if (L <= n || k <= 0) return prop;
  lk_index(r);
  const auto it = r.lk_idx.find(lk_hash(r.lk_hist.data() + L - n, static_cast<int>(n)));
  if (it == r.lk_idx.end()) return prop;
  const size_t j = static_cast<size_t>(it->second);
  if (j < n || !std::equal(r.lk_hist.begin() + static_cast<std::ptrdiff_t>(j - n), r.lk_hist.begin() + static_cast<std::ptrdiff_t>(j),
                           r.lk_hist.begin() + static_cast<std::ptrdiff_t>(L - n)))
    return prop;  // a hash collision
  for (size_t q = j; q < L && prop.size() < static_cast<size_t>(k); ++q) prop.push_back(r.lk_hist[q]);
  if (r.lk_pend.size() < 64) r.lk_pend.emplace_back(L, prop);
  return prop;
}

// The step's depth: the one in [0, depth_max] with the most expected tokens per ms over the going requests. Expected tokens
// of a request at depth d: 1 + p1 + p1 p2 + ... (its own averages, the engine's for positions it has not seen, 0.6 before
// any); ms per step: this engine's measured time of the shape, else a measured neighbour scaled by the model's relative
// cost (StepModel::spec_cost), else no change. Held unless another depth wins by depth_margin; every depth_explore-th step
// drafts depth_max.
int BatchEngine::pick_depth(const std::vector<size_t> &going, int depth_max) {
  const int N = static_cast<int>(going.size());
  if (!cfg_.adaptive_depth || depth_max <= 0 || N > kSeqMax) return depth_max;
  depth_max = std::min(depth_max, kDepthMax);
  int &cur = cur_depth_[N];
  if (cur < 0 || cur > depth_max) cur = depth_max;
  if (cfg_.depth_explore > 0 && spec_steps_ % static_cast<uint64_t>(cfg_.depth_explore) == 0) return depth_max;
  auto ms_of = [&](int d) -> double {
    const int rows = 1 + d;
    if (step_n_[N][rows] >= 3) return spec_ms_[N][rows];
    // a measured neighbour at this N, scaled by the model's relative cost
    for (int dd = 0; dd <= kDepthMax; ++dd) {
      const int r2 = 1 + dd;
      if (step_n_[N][r2] < 3) continue;
      const double a = m_.spec_cost(N, rows), b = m_.spec_cost(N, r2);
      if (a > 0 && b > 0) return spec_ms_[N][r2] * a / b;
    }
    return 0;
  };
  double best_v = 0, cur_v = 0;
  int best = cur;
  for (int d = 0; d <= depth_max; ++d) {
    const double ms = ms_of(d);
    if (ms <= 0) continue;
    double tok = 0;
    for (size_t i : going) {
      const Req &r = *active_[i];
      double e = 1, run = 1;
      for (int j = 0; j < d; ++j) {
        const float p = r.acc_p[j] >= 0 ? r.acc_p[j] : glob_p_[j] >= 0 ? glob_p_[j] : 0.6f;
        run *= p;
        e += run;
      }
      tok += e;
    }
    const double v = tok / ms;
    if (d == cur) cur_v = v;
    if (v > best_v) best_v = v, best = d;
  }
  if (best != cur && spec_steps_ - depth_since_[N] >= static_cast<uint64_t>(cfg_.depth_hold) &&
      (cur_v <= 0 || best_v > cur_v * (1 + cfg_.depth_margin))) {
    depth_since_[N] = spec_steps_;
    static const bool trace = [] { const char *e = std::getenv("XEN_DEPTH_TRACE"); return e && *e == '1'; }();
    if (trace) std::fprintf(stderr, "engine: draft depth at %d sequences %d -> %d (%.4f -> %.4f tokens per ms)\n", N, cur, best, cur_v, best_v);
    cur = best;
  }
  return cur;
}

double BatchEngine::expected_tokens(const Req &r, int d) const {
  double e = 1, run = 1;
  for (int j = 0; j < d && j < kDepthMax; ++j) {
    const float p = r.acc_p[j] >= 0 ? r.acc_p[j] : glob_p_[j] >= 0 ? glob_p_[j] : 0.6f;
    run *= p;
    e += run;
  }
  return e;
}

double BatchEngine::ver_est(int N, int rows) const {
  if (N < 1 || N > kSeqMax) return 0;
  const int R = std::min(m_.step_rows_at(N), kHetRows);
  if (rows < N || rows > R) return 0;
  if (ver_n_[N][rows] >= 3) return ver_ms_[N][rows];
  double best = 0;
  int bd = 1 << 30;
  for (int r2 = N; r2 <= R; ++r2) {
    const double a = m_.spec_cost_rows(N, rows, false), b = m_.spec_cost_rows(N, r2, false);
    if (ver_n_[N][r2] < 3 || a <= 0 || b <= 0) continue;
    if (std::abs(r2 - rows) < bd) bd = std::abs(r2 - rows), best = ver_ms_[N][r2] * a / b;
  }
  return best;
}

// Two classes. The one depth for all (pick_depth, d > 0) leaves max_rows - N (1 + d) rows free; k of the N going
// requests draft dhi > d, the k with the most expected tokens gained (the product of their acceptance through
// position dhi). Every (k, dhi) with N (1 + d) + k (dhi - d) <= max_rows and dhi <= StepModel::deep_depth is valued
// by expected tokens per ms: the verify of its rows (ver_est) + the accept at its deepest draft (measured per
// (sequences, depth), else a measured depth plus the model's difference, else the model's). Held unless another
// wins by kLayMargin, one change per depth_hold steps; pick_depth's exploring step keeps the one depth (its times
// stay measured), and midway between two of those the deeper class goes to the requests in turn (every request's
// position dhi keeps being observed).
void BatchEngine::pick_deep(const std::vector<size_t> &going, int d, std::vector<int> &depth) {
  const int N = static_cast<int>(going.size()), R = std::min(m_.step_rows_at(N), kHetRows);
  auto none = [&] {
    for (size_t i : going) active_[i]->lay_deep = false;
  };
  if (N < 2 || N > kSeqMax || d < 1) return none();
  const int dcap = std::min(kDepthMax, m_.deep_depth(N)), base = N * (1 + d);
  if (dcap <= d || base >= R || m_.draft_pass_ms(N, d) <= 0) return none();
  if (cfg_.depth_explore > 0 && spec_steps_ % static_cast<uint64_t>(cfg_.depth_explore) == 0) return;  // (the classes keep their requests)
  auto pass_of = [&](int x) -> double {
    if (deep_n_[N][x] >= 3) return deep_ms_[N][x];
    for (int off = 1; off <= kDepthMax; ++off)
      for (int y : {x - off, x + off})
        if (y >= 1 && y <= kDepthMax && deep_n_[N][y] >= 3) return std::max(0.0, deep_ms_[N][y] + m_.draft_pass_ms(N, x) - m_.draft_pass_ms(N, y));
    return m_.draft_pass_ms(N, x);
  };
  auto ms_of = [&](int rows, int x) -> double {
    double v = ver_est(N, rows);
    if (v <= 0) v = m_.spec_cost_rows(N, rows, false);
    return v > 0 ? v + pass_of(x) : 0.0;
  };
  // the requests by the tokens depth dhi adds to the one depth (best first)
  std::vector<int> ord(static_cast<size_t>(N));
  auto gain = [&](int i, int dhi) {
    const Req &r = *active_[going[static_cast<size_t>(i)]];
    return expected_tokens(r, dhi) - expected_tokens(r, d);
  };
  auto rank = [&](int dhi) {
    std::iota(ord.begin(), ord.end(), 0);
    auto key = [&](int i) { return gain(i, dhi) * (active_[going[static_cast<size_t>(i)]]->lay_deep ? kLayStick : 1.0); };
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return key(a) > key(b); });
  };
  double tok0 = 0;
  for (size_t i : going) tok0 += expected_tokens(*active_[i], d);
  auto value = [&](int k, int dhi) -> double {
    if (k < 0 || k > N || (k > 0 && (dhi <= d || dhi > dcap)) || base + k * (dhi - d) > R) return -1;
    const double ms = ms_of(k > 0 ? base + k * (dhi - d) : base, k > 0 ? dhi : d);
    if (ms <= 0) return -1;
    double tok = tok0;
    if (k > 0) {
      rank(dhi);
      for (int i = 0; i < k; ++i) tok += gain(ord[static_cast<size_t>(i)], dhi);
    }
    return tok / ms;
  };
  int &ck = cur_k2_[N], &cd = cur_d2_[N];
  if (ck > 0 && value(ck, cd) < 0) ck = cd = 0;  // the held class does not fit this depth
  double best_v = value(0, 0), cur_v = value(ck, cd);
  int bk = 0, bdp = 0;
  for (int dhi = d + 1; dhi <= dcap; ++dhi)
    for (int k = 1; k <= N && base + k * (dhi - d) <= R; ++k) {
      const double v = value(k, dhi);
      if (v > best_v) best_v = v, bk = k, bdp = dhi;
    }
  if ((bk != ck || (bk > 0 && bdp != cd)) && spec_steps_ - deep_since_[N] >= static_cast<uint64_t>(cfg_.depth_hold) &&
      (cur_v <= 0 || best_v > cur_v * (1 + kLayMargin))) {
    deep_since_[N] = spec_steps_;
    static const bool trace = [] { const char *e = std::getenv("XEN_DEPTH_TRACE"); return e && *e == '1'; }();
    if (trace)
      std::fprintf(stderr, "engine: deeper class at %d sequences (depth %d) %d x depth %d -> %d x depth %d (%.4f -> %.4f tokens per ms)\n", N, d, ck, cd, bk, bdp, cur_v,
                   best_v);
    ck = bk, cd = bdp;
  }
  // in turn (the class's requests are not changed by it); without a held class every 4th of those steps runs the widest
  // one at d + 1, so its accept's time and position d + 1 stay measured
  const uint64_t ex = static_cast<uint64_t>(cfg_.depth_explore);
  if (ex >= 2 && spec_steps_ % ex == ex / 2 && (ck > 0 ? ck < N : (spec_steps_ / ex) % 4 == 0)) {
    const int tk = ck > 0 ? ck : std::min(N - 1, R - base), td = ck > 0 ? cd : d + 1;
    const int first = static_cast<int>((spec_steps_ / ex) % static_cast<uint64_t>(N));
    for (int i = 0; i < tk; ++i) depth[static_cast<size_t>((first + i) % N)] = td;
    return;
  }
  if (ck <= 0) return none();
  rank(cd);
  for (int i = 0; i < N; ++i) {
    const int j = ord[static_cast<size_t>(i)];
    active_[going[static_cast<size_t>(j)]]->lay_deep = i < ck;
    if (i < ck) depth[static_cast<size_t>(j)] = cd;
  }
}

// The next step's layout with unequal segments. k of the N going requests draft d tokens each
// (the k with the most expected tokens at d), the rest none; the (k, d), N + k d <= max_rows, with the most expected
// tokens per ms. ms = the verify of N + k d rows + the accept (a drafting pass or the context-only one), each this
// engine's measured time at N (the step's time split at the accept), else scaled from a measured neighbour by
// StepModel::spec_cost_rows, else the model's cost. The layout is held unless another wins by depth_margin, at most one
// change per depth_hold steps at a sequence count; every depth_explore-th step drafts for the requests in turn (each
// request's positions keep being observed), or under a drafting layout runs plain. The drafting requests themselves
// follow the expected tokens every step (the same cost). Called where pick_depth's one depth is 0.
std::vector<int> BatchEngine::pick_layout(const std::vector<size_t> &going) {
  const int N = static_cast<int>(going.size()), R = std::min(m_.step_rows_at(N), kHetRows);
  std::vector<int> out(static_cast<size_t>(N), 0);
  const int dcap = std::min({kDepthMax, m_.layout_depth(N), R - N});  // the model's cap at N
  if (N < 1 || N > kSeqMax || dcap <= 0) return out;
  explore_now_ = false;
  // under a drafting layout every depth_explore-th step runs plain: the plain verify's and the context accept's times
  // come from steady steps, not only from those around an admission. Under a plain layout every 4th of those drafts
  // for some requests in turn: such a pass also reads every context row deferred since the request last drafted,
  // so it is not timed and runs a quarter as often.
  const uint64_t ex = static_cast<uint64_t>(cfg_.depth_explore) * (cur_k_[N] > 0 ? 1 : 4);
  if (cfg_.depth_explore > 0 && spec_steps_ % ex == 0) {
    if (cur_k_[N] > 0) return out;
    explore_now_ = true;
    const int d = std::min(dcap, std::max(1, R / N - 1)), k = std::min(N, (R - N) / d);
    const int first = static_cast<int>((spec_steps_ / ex) % static_cast<uint64_t>(N));
    for (int i = 0; i < k; ++i) out[static_cast<size_t>((first + i) % N)] = d;
    return out;
  }
  // ms of a layout = its verify (rows) + its accept (drafts after it or not): measured, else a measured neighbour scaled
  // by the model's relative verify cost / the other accept plus the model's difference, else the model's whole cost
  auto ver_of = [&](int rows) -> double { return ver_est(N, rows); };
  auto pass_of = [&](bool dr) -> double {
    if (pass_n_[N][dr] >= 3) return pass_ms_[N][dr];
    const double md = m_.spec_cost_rows(N, N, dr) - m_.spec_cost_rows(N, N, !dr);
    return pass_n_[N][!dr] >= 3 ? std::max(0.0, pass_ms_[N][!dr] + md) : -1;
  };
  auto ms_of = [&](int rows, bool dr) -> double {
    const double v = ver_of(rows), p = pass_of(dr);
    if (v > 0 && p >= 0) return v + p;
    return m_.spec_cost_rows(N, rows, dr);
  };
  // the requests by expected tokens at depth d (best first)
  std::vector<int> ord(static_cast<size_t>(N));
  auto rank = [&](int d) {
    std::iota(ord.begin(), ord.end(), 0);
    auto key = [&](int i) {
      const Req &r = *active_[going[static_cast<size_t>(i)]];
      return expected_tokens(r, d) * (r.lay_draft ? kLayStick : 1.0);
    };
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return key(a) > key(b); });
  };
  auto value = [&](int k, int d) -> double {
    if (k < 0 || k > N || (k > 0 && (d < 1 || d > dcap)) || N + k * d > R) return -1;
    const double ms = ms_of(N + k * d, k > 0);
    if (ms <= 0) return -1;
    double tok = N - k;
    if (k > 0) {
      rank(d);
      for (int i = 0; i < k; ++i) tok += expected_tokens(*active_[going[static_cast<size_t>(ord[static_cast<size_t>(i)])]], d);
    }
    return tok / ms;
  };
  int &ck = cur_k_[N], &cd = cur_d_[N];
  if (ck < 0 || value(ck, cd) < 0) {  // start from one depth for all (as without unequal segments), else no drafts
    const int d0 = std::min(dcap, R / N - 1);
    ck = d0 > 0 ? N : 0, cd = d0 > 0 ? d0 : 0;
  }
  double best_v = value(0, 0), cur_v = value(ck, cd);
  int bk = 0, bdp = 0;
  for (int d = 1; d <= dcap; ++d)
    for (int k = 1; k <= N && N + k * d <= R; ++k) {
      const double v = value(k, d);
      if (v > best_v) best_v = v, bk = k, bdp = d;
    }
  if ((bk != ck || (bk > 0 && bdp != cd)) && spec_steps_ - lay_since_[N] >= static_cast<uint64_t>(cfg_.depth_hold) &&
      (cur_v <= 0 || best_v > cur_v * (1 + kLayMargin))) {
    lay_since_[N] = spec_steps_;
    static const bool trace = [] { const char *e = std::getenv("XEN_DEPTH_TRACE"); return e && *e == '1'; }();
    if (trace) std::fprintf(stderr, "engine: layout at %d sequences %d x depth %d -> %d x depth %d (%.4f -> %.4f tokens per ms)\n", N, ck, cd, bk, bdp, cur_v, best_v);
    ck = bk, cd = bdp;
  }
  if (ck > 0) {
    rank(cd);
    for (int i = 0; i < ck; ++i) out[static_cast<size_t>(ord[static_cast<size_t>(i)])] = cd;
  }
  return out;
}

void BatchEngine::retire(size_t i, Finish f) {
  Req &r = *active_[i];
  if (r.slot >= 0) {
    if (pc_on_) pc_park(r, f);           // the slot keeps its sequence (pages, live state, snapshots)
    else if (kv_shared_) m_.kv_release(r.slot);  // the pages back to the pool
    free_slots_.push_back(r.slot);
  }
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.finished_by[static_cast<int>(f)]++;
    stats_.e2e.observe(dsec(clock::now() - r.t_submit).count());
    if (f != Finish::cancelled) stats_.requests_finished++;
    stats_.running = static_cast<uint32_t>(active_.size() - 1);
  }
  active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(i));
}

void BatchEngine::kv_stats() {
  const int64_t pool = m_.kv_pool_tokens();
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.kv_pool_tokens = pool;
  stats_.kv_used_tokens = pool > 0 ? pool - m_.kv_free_tokens() : 0;
}

// Every member's first token of a finished call for several requests, read before anything reuses the head
// (the first member's first drafts, a decode step): the logits row of a sampled request, the device's pick, the
// logprobs of a greedy one.
void BatchEngine::pf_group_read() {
  for (auto &mb : pf_pend_.group) {
    if (mb.read) continue;
    const auto it = std::find_if(active_.begin(), active_.end(), [&](const auto &q) { return q->id == mb.id; });
    if (it == active_.end()) throw std::logic_error("batch engine: a merged prefill call lost a request");
    Req &r = **it;
    m_.prefill_result(r.slot);
    if (r.host) m_.prefill_logits(mb.row);
    mb.tok = m_.prefill_pick();
    if (!r.host && r.params.logprobs >= 0) {
      m_.prefill_logprobs(mb.tok, r.params.logprobs, mb.lp);
      mb.has_lp = true;
    }
    mb.read = true;
  }
}

// A request leaves (cancelled, preempted) while the call it shares is in parts: the call runs to its end now (an abort
// would break every member's slot), the others' first tokens are read and they complete in the loop's prefill stage.
// True when id was a member.
bool BatchEngine::pf_group_leave(uint64_t id) {
  if (!pf_pend_.active || pf_pend_.group.empty()) return false;
  const auto it = std::find_if(pf_pend_.group.begin(), pf_pend_.group.end(), [&](const auto &mb) { return mb.id == id; });
  if (it == pf_pend_.group.end()) return false;
  if (!pf_pend_.finished) {
    const auto ta = clock::now();
    while (!m_.prefill_advance(std::max(1, m_.prefill_layers()))) {
    }
    pf_pend_.ms += std::chrono::duration<double, std::milli>(clock::now() - ta).count();
    ++pf_pend_.parts;
    pf_pend_.finished = true;
  }
  pf_pend_.group.erase(it);
  pf_group_read();
  if (pf_pend_.group.empty()) pf_pend_.active = false;
  return true;
}

void BatchEngine::preempt_now(size_t i) {
  Req &r = *active_[i];
  if (pf_group_leave(r.id)) {
  } else if (pf_pend_.active && pf_pend_.id == r.id) {  // its split prefill call ends here
    m_.prefill_abort();
    pf_pend_.active = false;
  }
  m_.kv_release(r.slot);
  if (pc_on_) pc_free_snaps(r.slot);  // its pages are gone, and every snapshot over them
  r.preempt = true;
}

std::vector<size_t> BatchEngine::kv_ensure(const std::vector<size_t> &cand0, std::vector<std::pair<size_t, Finish>> &done) {
  std::vector<size_t> cand = cand0;
  auto in_done = [&](size_t i) { return std::any_of(done.begin(), done.end(), [&](const auto &d) { return d.first == i; }); };
  if (tiers_ > 1) {  // a step past its YaRN tier's positions: it resumes (re-prefills) in the tier that holds it
    const int64_t ahead = m_.kv_ahead(std::max(m_.draft_depth(1), m_.wide_depth()));  // + the wide step
    bool any = false;
    for (size_t i : cand) {
      Req &r = *active_[i];
      if (r.preempt || in_done(i) || r.tier + 1 >= tiers_ || r.pos + ahead <= m_.rope_tier_limit(r.tier)) continue;
      preempt_now(i);
      any = true;
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.yarn_retiers++;
    }
    if (any) {
      std::vector<size_t> keep;
      for (size_t i : cand)
        if (!active_[i]->preempt) keep.push_back(i);
      cand.swap(keep);
    }
  }
  if (!kv_shared_) return cand;
  // requests that ended in this loop's prefill: their pages are free now (the prefix cache: they keep them until
  // retire parks them, unless a reservation below needs them)
  if (!pc_on_)
    for (const auto &d : done)
      if (active_[d.first]->slot >= 0) m_.kv_release(active_[d.first]->slot);
  // pages for a step come from parked sequences first, then from the requests that just ended
  auto free_cached = [&]() -> bool {
    if (!pc_on_) return false;
    if (pc_evict_lru(-1, -1)) return true;
    for (const auto &d : done) {
      Req &q = *active_[d.first];
      if (q.slot >= 0 && !q.pc_dropped && m_.kv_held_tokens(q.slot) > 0) {
        m_.kv_release(q.slot);
        pc_free_snaps(q.slot);
        q.pc_dropped = true;
        return true;
      }
    }
    return false;
  };
  // the positions a step writes: its rows (1 + drafts) and the drafter passes after it (up to the next drafts)
  const int depth = std::max(m_.draft_depth(1), m_.wide_depth());  // the wide step's rows too
  const int64_t ahead = m_.kv_ahead(depth);
  std::vector<size_t> order = cand;  // oldest admitted first: the youngest are the ones preempted
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return active_[a]->t_admit < active_[b]->t_admit; });
  for (size_t i : order) {
    Req &r = *active_[i];
    if (r.preempt) continue;
    const int64_t need = std::min<int64_t>(m_.capacity(), r.pos + ahead);
    lend_pressure(r.slot, need, -1, -1);
    while (!m_.kv_reserve(r.slot, need)) {
      if (free_cached()) continue;  // 
      // the youngest request still holding pages, this one included (an older request never yields to a younger)
      size_t victim = active_.size();
      int holders = 0;
      for (size_t k = 0; k < active_.size(); ++k) {
        const Req &v = *active_[k];
        if (v.preempt || in_done(k)) continue;
        ++holders;
        if (victim == active_.size() || v.t_admit > active_[victim]->t_admit) victim = k;
      }
      if (holders <= 1) {  // alone and still short: a pool smaller than this request's need (the model forbids it)
        Event e;
        e.id = r.id;
        e.finish = Finish::length;
        events_.push_back(std::move(e));
        done.push_back({i, Finish::length});
        break;
      }
      preempt_now(victim);
      if (victim == i) break;
    }
  }
  std::vector<size_t> dec;
  for (size_t i : cand)
    if (!active_[i]->preempt && !in_done(i)) dec.push_back(i);
  return dec;
}

// ----: the prefix cache ----
//
// Every slot's cache entry (pc_[slot]) describes what a new request may continue from it:
//   - while a request runs on the slot: its snapshots (taken after its prefill calls, positions <= what it prefilled);
//     the tokens behind them are the request's own (prompt, then output);
//   - once it finished (parked): hist = the tokens behind the slot's state plus the one after them (the drafter's KV row
//     before a position was written from the token at that position), live = the position its live state stands at,
//     and the snapshots. The slot keeps its KV pages until it is evicted (a request needs them) or taken.
// A state at position R serves a prompt T when T[0, R) == hist[0, R), R < |T| (one token is left to prefill: its row
// gives the first output) and, for a model with a drafter (prefix_strict), T[R] == hist[R] too.

namespace {
// the length of the common prefix of a and the concatenation b1 ++ b2
int64_t lcp2(const std::vector<int32_t> &a, const int32_t *b1, size_t n1, const std::vector<int32_t> &b2) {
  const size_t n = std::min(a.size(), n1 + b2.size());
  size_t i = 0;
  for (; i < n; ++i)
    if (a[i] != (i < n1 ? b1[i] : b2[i - n1])) break;
  return static_cast<int64_t>(i);
}
}  // namespace

const BatchEngine::Req *BatchEngine::busy(int slot) const {
  for (const auto &r : active_)
    if (r->slot == slot) return r.get();
  return nullptr;
}

bool BatchEngine::is_free(int slot) const { return std::find(free_slots_.begin(), free_slots_.end(), slot) != free_slots_.end(); }

// Where a request's prefill call is cut so a snapshot exists for the conversation's next turn: its prefix_mark rounded
// down to snap_align, and at least snap_align rows before the prompt's end (on the dense 27B the rows of a call under 64
// rows differ in the last bits from the same rows inside a longer call; 64+ rows at multiples of 64 are bit-identical),
// so both sides of the cut keep a cold prefill's numerics. -1: no cut.
int64_t BatchEngine::pc_cut(const Req &r) const {
  if (!pc_on_ || r.params.prefix_mark < 0 || r.params.prefix_mark < cfg_.prefix_mark_min) return -1;
  const int64_t al = std::max(1, m_.snap_align()), TN = static_cast<int64_t>(r.prompt.size());
  const int64_t a = std::min<int64_t>(r.params.prefix_mark / al * al, (TN - al) / al * al);
  return a > 0 && TN - al >= 0 ? a : -1;
}

// The rows of r's next prefill call when nothing decodes. Up to prefill_chunk rows, ending at the request's cut (pc_cut)
// and never leaving fewer than snap_align rows to a later call, except the one a cold prefill has too: from position 0
// that is the schedule (calls at multiples of the chunk, the last one possibly short); from a cached prefix R (a
// multiple of snap_align) every boundary is a multiple of snap_align and every call has snap_align+ rows or is the cold
// run's own short last call, so the rows keep a cold prefill's numerics.
bool BatchEngine::pc_inside(const Req &r) const {
  return cfg_.prefix_inside && m_.snap_inside() && !(r.params.prompt_logprobs >= 0 && !r.resumed);
}

int BatchEngine::pf_call_rows(const Req &r) const {
  const int64_t p = static_cast<int64_t>(r.prefilled), TN = static_cast<int64_t>(r.prompt.size());
  const int64_t ch = std::max(1, m_.prefill_chunk()), al = std::max(1, m_.snap_align());
  int64_t t = TN;  // the next boundary every run has
  // a cut the model snapshots inside a call ends no call
  if (const int64_t a = pc_cut(r); a > p && !pc_inside(r)) t = std::min(t, a);
  if (const int64_t h = r.share_cut; h > p && !pc_inside(r)) t = std::min(t, h);  // a waiting request continues from there (pc_leader)
  if (const int64_t c_last = (TN - 1) / ch * ch; TN - c_last < al && c_last > p) t = std::min(t, c_last);
  int64_t n = std::min(ch, t - p);
  if (const int64_t rest = t - p - n; rest > 0 && rest < al) n = (t - p - al) / al * al;
  return static_cast<int>(std::max<int64_t>(n, 1));
}

int64_t BatchEngine::pc_tip(int slot) const {
  const SlotCache &c = pc_[static_cast<size_t>(slot)];
  int64_t t = c.live;
  if (!c.snaps.empty()) t = std::max(t, c.snaps.back().pos);
  return t;
}

void BatchEngine::pc_stats() {
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.prefix_snapshots = static_cast<uint32_t>(pc_count_);
}

void BatchEngine::pc_free_snaps(int slot) {
  SlotCache &c = pc_[static_cast<size_t>(slot)];
  for (const Snap &s : c.snaps) m_.snap_free(s.id);
  pc_count_ -= static_cast<int>(c.snaps.size());
  c.snaps.clear();
}

void BatchEngine::pc_drop(int slot, int64_t above) {
  SlotCache &c = pc_[static_cast<size_t>(slot)];
  while (!c.snaps.empty() && c.snaps.back().pos > above) {
    m_.snap_free(c.snaps.back().id);
    c.snaps.pop_back();
    --pc_count_;
  }
}

void BatchEngine::pc_evict(int slot) {
  pc_free_snaps(slot);
  SlotCache &c = pc_[static_cast<size_t>(slot)];
  c.hist.clear();
  c.hist.shrink_to_fit();
  c.live = -1;
  if (kv_shared_) m_.kv_release(slot);
}

void BatchEngine::pc_clear() {
  for (int s = 0; s < static_cast<int>(pc_.size()); ++s) {
    if (busy(s)) continue;
    pc_evict(s);
  }
  pc_stats();
}

void BatchEngine::pc_park(Req &r, Finish f) {
  SlotCache &c = pc_[static_cast<size_t>(r.slot)];
  c.tick = ++pc_tick_;
  c.parked = clock::now();
  if (r.pc_dropped || f == Finish::error) {  // its pages went to a decode step this loop (or its state is not trusted)
    pc_evict(r.slot);
    return;
  }
  c.hist.clear();
  if (r.prefilled == r.prompt.size()) {  // the prompt, then every emitted token (the last one not fed yet)
    c.hist.assign(r.prompt.begin(), r.prompt.begin() + static_cast<std::ptrdiff_t>(r.prompt_len));
    c.hist.insert(c.hist.end(), r.out.begin(), r.out.end());
  } else {  // inside its prompt: the tokens fed and the next one
    c.hist.assign(r.prompt.begin(), r.prompt.begin() + static_cast<std::ptrdiff_t>(std::min(r.prompt.size(), r.prefilled + 1)));
  }
  // a live state whose rows past the prompt came from decode steps continues only with prefix_live: the cold run would
  // prefill those tokens (other numerics); one made of prefill rows only, at the alignment, always
  const bool exact = r.pos <= static_cast<int64_t>(r.prompt.size()) && r.pos % std::max(1, m_.snap_align()) == 0;
  c.live = !r.aborted && r.pos >= cfg_.prefix_min_tokens && (cfg_.prefix_live || exact) && m_.snap_live(r.slot) ? r.pos : -1;
  if (c.live < 0 && c.snaps.empty()) pc_evict(r.slot);  // nothing to continue from: the pages go back
}

// The snapshot of slot c the store gives up first: not the newest one and not the newest two marks; of the rest, the one
// whose removal leaves the smallest gap relative to its distance from the newest (so snapshots thin out geometrically
// away from where the sequence is: an edit near its end recomputes little, one near its start up to about its distance).
size_t BatchEngine::pc_victim(const SlotCache &c) const {
  const size_t n = c.snaps.size();
  if (n <= 1) return 0;
  std::vector<bool> keep(n, false);
  keep[n - 1] = true;
  for (size_t i = n, marks = 0; i-- > 0 && marks < 2;)
    if (c.snaps[i].mark) keep[i] = true, ++marks;
  const double tip = static_cast<double>(c.snaps[n - 1].pos);
  size_t best = n;
  double cost = 1e300;
  for (size_t i = 0; i + 1 < n; ++i) {
    if (keep[i]) continue;
    const double lo = i > 0 ? static_cast<double>(c.snaps[i - 1].pos) : 0.0, hi = static_cast<double>(c.snaps[i + 1].pos);
    const double v = (hi - lo) / (tip - lo + 1.0);
    if (v < cost) cost = v, best = i;
  }
  return best < n ? best : 0;  // every one kept: the oldest
}

bool BatchEngine::pc_free_one(int current, bool mark) {
  // a full store gives up a plain (unmarked) snapshot first: of the least recently used slot (the asking slot last), not
  // a slot's newest unless no other is left. A plain snapshot never displaces a marked one (the next turns' resume points)
  {
    int vs = -1;
    size_t vk = 0;
    int vlevel = 9;
    for (int s = 0; s < static_cast<int>(pc_.size()); ++s) {
      const SlotCache &c = pc_[static_cast<size_t>(s)];
      for (size_t k = 0; k < c.snaps.size(); ++k) {
        if (c.snaps[k].mark) continue;
        const int level = (k + 1 == c.snaps.size() ? 2 : 0) + (s == current ? 1 : 0);
        const bool better = vs < 0 || level < vlevel || (level == vlevel && c.tick < pc_[static_cast<size_t>(vs)].tick);
        if (better) vs = s, vk = k, vlevel = level;
      }
    }
    if (vs >= 0) {
      SlotCache &c = pc_[static_cast<size_t>(vs)];
      m_.snap_free(c.snaps[vk].id);
      c.snaps.erase(c.snaps.begin() + static_cast<std::ptrdiff_t>(vk));
      --pc_count_;
      if (c.snaps.empty() && c.live < 0 && !busy(vs)) pc_evict(vs);
      return true;
    }
  }
  if (!mark) return false;
  int victim = -1;
  for (int s = 0; s < static_cast<int>(pc_.size()); ++s) {
    const SlotCache &c = pc_[static_cast<size_t>(s)];
    if (c.snaps.empty()) continue;
    // the least recently used slot; the slot asking only when no other holds a snapshot
    const bool better = victim < 0 || (s != current && (victim == current || c.tick < pc_[static_cast<size_t>(victim)].tick));
    if (better) victim = s;
  }
  if (victim < 0) return false;
  SlotCache &c = pc_[static_cast<size_t>(victim)];
  const size_t k = pc_victim(c);
  m_.snap_free(c.snaps[k].id);
  c.snaps.erase(c.snaps.begin() + static_cast<std::ptrdiff_t>(k));
  --pc_count_;
  if (c.snaps.empty() && c.live < 0 && !busy(victim)) pc_evict(victim);
  return true;
}

void BatchEngine::pc_snapshot(int slot, int64_t pos, bool mark, bool inside) {
  constexpr size_t kSlotSnaps = 8;  // per slot: what pc_victim thins down to
  SlotCache &c = pc_[static_cast<size_t>(slot)];
  if (!c.snaps.empty() && c.snaps.back().pos >= pos) pc_drop(slot, pos - 1);  // the same position again: the newer state
  auto save = [&] { return inside ? m_.snap_take() : m_.snap_save(slot); };
  int id = save();
  if (id == -2) return;  // nothing captured inside the call
  while (id < 0) {
    if (!pc_free_one(slot, mark)) return;  // nothing it may displace
    id = save();
  }
  c.snaps.push_back({pos, id, mark});
  ++pc_count_;
  while (c.snaps.size() > kSlotSnaps) {
    const size_t k = pc_victim(c);
    m_.snap_free(c.snaps[k].id);
    c.snaps.erase(c.snaps.begin() + static_cast<std::ptrdiff_t>(k));
    --pc_count_;
  }
}

bool BatchEngine::pc_evict_lru(int keep_a, int keep_b) {
  int victim = -1;
  for (int s : free_slots_) {
    if (s == keep_a || s == keep_b || m_.kv_held_tokens(s) <= 0) continue;
    if (victim < 0 || pc_[static_cast<size_t>(s)].tick < pc_[static_cast<size_t>(victim)].tick) victim = s;
  }
  if (victim < 0) return false;
  pc_evict(victim);
  return true;
}

// ---- KV lending vs the prefix cache (BatchConfig::lend_park_s / lend_pressure) ----

void BatchEngine::lend_park_timeout() {
  if (!pc_on_ || !kv_shared_ || cfg_.lend_park_s <= 0) return;
  const auto now = clock::now();
  for (int s : free_slots_) {
    const SlotCache &c = pc_[static_cast<size_t>(s)];
    if (m_.kv_held_tokens(s) <= 0 || !m_.kv_blocks_lend(s) || dsec(now - c.parked).count() < cfg_.lend_park_s) continue;
    std::fprintf(stderr, "engine: KV lending: a parked sequence (slot %d, %lld tokens) dropped after %.0f s parked: its pages go back to the lent experts\n",
                 s, static_cast<long long>(m_.kv_held_tokens(s)), dsec(now - c.parked).count());
    pc_evict(s);
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.lend_park_drops++;
  }
  pc_stats();
}

double BatchEngine::lend_park_wait_s() const {
  if (!pc_on_ || !kv_shared_ || cfg_.lend_park_s <= 0) return -1;
  const auto now = clock::now();
  double w = -1;
  for (int s : free_slots_) {
    const SlotCache &c = pc_[static_cast<size_t>(s)];
    if (m_.kv_held_tokens(s) <= 0 || !m_.kv_blocks_lend(s)) continue;
    const double left = std::max(0.0, cfg_.lend_park_s - dsec(now - c.parked).count());
    w = w < 0 ? left : std::min(w, left);
  }
  return w;
}

void BatchEngine::lend_pressure(int slot, int64_t tokens, int keep_a, int keep_b) {
  if (!pc_on_ || !kv_shared_ || cfg_.lend_pressure == "kv") return;
  const bool idle_only = cfg_.lend_pressure == "idle";
  const auto now = clock::now();
  while (m_.kv_reserve_lends(slot, tokens)) {
    int victim = -1;
    for (int s : free_slots_) {
      if (s == slot || s == keep_a || s == keep_b || m_.kv_held_tokens(s) <= 0 || !m_.kv_blocks_lend(s)) continue;
      const SlotCache &c = pc_[static_cast<size_t>(s)];
      if (idle_only && dsec(now - c.parked).count() < cfg_.lend_grace_s) continue;
      if (victim < 0 || c.tick < pc_[static_cast<size_t>(victim)].tick) victim = s;
    }
    if (victim < 0) return;
    pc_evict(victim);
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.lend_pressure_drops++;
  }
}

int BatchEngine::tier_for(int64_t n, int64_t rem) const {
  const int64_t margin = std::max<int64_t>(64, 2 * m_.kv_ahead(std::max(m_.draft_depth(1), m_.wide_depth())));
  for (int t = 0; t + 1 < tiers_; ++t) {
    const int64_t L = m_.rope_tier_limit(t);
    if (n + std::max(margin, std::min(rem, L / 4)) <= L) return t;
  }
  return tiers_ - 1;
}

// The longest usable cached prefix of T: a parked slot's live state or any slot's snapshot (src -1: none).
BatchEngine::PcCand BatchEngine::pc_find(const std::vector<int32_t> &T, int tier) const {
  PcCand best;
  const bool strict = m_.prefix_strict();
  const int64_t TN = static_cast<int64_t>(T.size());
  const int64_t al = std::max(1, m_.snap_align()), chunk = std::max(1, m_.prefill_chunk());
  for (int s = 0; s < static_cast<int>(pc_.size()); ++s) {
    const SlotCache &c = pc_[static_cast<size_t>(s)];
    const Req *b = busy(s);
    if (tiers_ > 1 && c.tier != tier) continue;  // another YaRN tier's rows
    if (b ? c.snaps.empty() : (c.hist.empty() || (c.live < 0 && c.snaps.empty()))) continue;
    const int64_t L = b ? lcp2(T, b->prompt.data(), b->prompt_len, b->out) : lcp2(T, c.hist.data(), c.hist.size(), {});
    // the calls from R keep a cold prefill's numerics (pf_call_rows) unless fewer than snap_align rows are left and
    // the cold run's last call does not start at R
    auto usable = [&](int64_t R) {
      return R >= cfg_.prefix_min_tokens && R < TN && (strict ? R < L : R <= L) && (TN - R >= al || R == (TN - 1) / chunk * chunk);
    };
    // ties: a parked slot's live state (nothing to copy), then its snapshot, then a running slot's snapshot
    auto better = [&](int64_t R, int rank) { return best.src < 0 || R > best.R || (R == best.R && rank > best.rank); };
    if (!b && c.live >= 0 && usable(c.live) && better(c.live, 2)) best = {s, c.live, -1, 2};
    for (size_t k = c.snaps.size(); k-- > 0;)
      if (usable(c.snaps[k].pos)) {
        if (better(c.snaps[k].pos, b ? 0 : 1)) best = {s, c.snaps[k].pos, static_cast<int>(k), b ? 0 : 1};
        break;
      }
  }
  return best;
}

// In-flight prefix sharing (BatchConfig::prefix_share). A waiting request whose prompt starts like the prompt a running
// request is still prefilling would be placed before that prefill has left a snapshot, and prefill the common part
// again (N requests on one document: N prefills of it). pc_leader names the running request to wait for and the
// position `cut` (a multiple of snap_align both prompts can be cut at): the leader's snapshot is taken there
// (Req::share_cut: inside its call when the model can, StepModel::snap_inside, so the leader's calls and rows are the
// ones it runs alone; else its call ends there), then the waiting request is placed as any other and forks from it.
// A request arriving while the leader's call over `cut` is already running continues from the call end before it. It
// waits only while that
// saves it at least prefix_min_tokens the leader has not prefilled yet, and only when no cached state would take it
// further than the leader is now (prefill calls run one at a time, so the wait is the time of rows it would have
// prefilled itself). Null: place it now.
BatchEngine::Req *BatchEngine::pc_leader(const std::vector<int32_t> &T, int tier, int64_t &cut) const {
  cut = -1;
  if (!pc_on_ || !cfg_.prefix_share) return nullptr;
  const int64_t TN = static_cast<int64_t>(T.size()), al = std::max(1, m_.snap_align());
  const int64_t gain = std::max(1, cfg_.prefix_min_tokens), strict = m_.prefix_strict() ? 1 : 0;
  Req *lead = nullptr;
  for (const auto &up : active_) {
    Req &b = *up;
    const int64_t TB = static_cast<int64_t>(b.prompt.size()), pb = static_cast<int64_t>(b.prefilled);
    if (pb >= TB || b.aborted || (tiers_ > 1 && b.tier != tier)) continue;
    if (std::min(TN, TB) - al - pb < gain) continue;  // nothing to gain whatever they share
    const int64_t L = lcp2(T, b.prompt.data(), b.prompt_len, b.out);
    const int64_t c = std::min({L - strict, TN - al, TB - al}) / al * al;
    if (c < cfg_.prefix_min_tokens || c - pb < gain) continue;
    if (c > cut) lead = &b, cut = c;
  }
  if (lead && pc_find(T, tier).R > static_cast<int64_t>(lead->prefilled)) lead = nullptr;
  if (!lead) cut = -1;
  return lead;
}

int BatchEngine::place(const std::vector<int32_t> &T, int64_t need, bool use_cache, int64_t &start, int tier) {
  start = 0;
  if (free_slots_.empty()) return -1;
  if (!pc_on_) {  // the last free slot, a fresh sequence
    const int slot = free_slots_.back();
    if (kv_shared_ && !m_.kv_reserve(slot, need)) return -1;
    free_slots_.pop_back();
    m_.begin(slot);  // a fresh sequence (it drops the slot's pages) ...
    if (tiers_ > 1) m_.set_rope_tier(slot, tier);
    if (kv_shared_ && !m_.kv_reserve(slot, need)) throw std::logic_error("batch engine: pages taken between reserve and begin");  // ... taken again
    return slot;
  }
  // 1. the longest usable cached prefix: a parked slot's live state or any slot's snapshot
  PcCand best = use_cache ? pc_find(T, tier) : PcCand{};
  const int64_t TN = static_cast<int64_t>(T.size());
  // 2. where it runs: in place on a parked source when that keeps at least half of what the slot could continue (or no
  //    other slot is free); else a copy into another slot (an empty one first, else the least recently used parked one)
  auto pick_free = [&](int not_this) {
    int pick = -1;
    for (int s : free_slots_) {
      if (s == not_this) continue;
      const SlotCache &c = pc_[static_cast<size_t>(s)];
      const bool empty = c.hist.empty() && c.snaps.empty() && c.live < 0;
      if (pick < 0) {
        pick = s;
        continue;
      }
      const SlotCache &p = pc_[static_cast<size_t>(pick)];
      const bool pempty = p.hist.empty() && p.snaps.empty() && p.live < 0;
      if (empty != pempty) {
        if (empty) pick = s;
      } else if (!empty && c.tick < p.tick) {
        pick = s;
      }
    }
    return pick;
  };
  bool in_place = false, fork = false;
  int dst = -1;
  if (best.src >= 0) {
    const bool parked = is_free(best.src);
    if (parked && (2 * best.R >= pc_tip(best.src) || pick_free(best.src) < 0)) in_place = true;
    else if (best.snap >= 0 && pick_free(best.src) >= 0) fork = true;
    else if (parked) in_place = true;
    if (in_place) dst = best.src;
    else if (fork) dst = pick_free(best.src);
  }
  if (dst < 0) {  // no usable prefix: a fresh sequence on an empty slot, else on the least recently used parked one
    best = PcCand{};
    dst = pick_free(-1);
  }
  // 3. pages: positions [0, need) on dst, evicting other parked sequences while the pool is short (the source's too, when
  //    it is not dst); first check that it can work at all, so nothing is evicted for a request that waits anyway. A fork
  //    keeps its source's pages: when that is what the pool lacks, the request runs in place on a parked source or from
  //    position 0 instead (with nothing running, waiting would never end)
  auto fits = [&](bool ip, int d, int keep) {
    if (!kv_shared_) return true;
    const int64_t P = std::max(1, m_.kv_page_tokens());
    auto up = [&](int64_t t) { return (t + P - 1) / P * P; };
    int64_t avail = m_.kv_free_tokens();
    for (int s : free_slots_) {
      if (s == keep) continue;
      const int64_t h = m_.kv_held_tokens(s);
      avail += s == d && ip ? std::max<int64_t>(0, h - up(best.R)) : h;
    }
    return avail >= up(need) - (ip ? std::min(up(best.R), m_.kv_held_tokens(d)) : 0);
  };
  if (!fits(in_place, dst, fork ? best.src : -1)) {
    if (fork && is_free(best.src) && fits(true, best.src, -1)) {
      fork = false, in_place = true, dst = best.src;
    } else if (best.src >= 0) {
      best = PcCand{};
      fork = in_place = false;
      dst = pick_free(-1);
      if (!fits(false, dst, -1)) return -1;
    } else {
      return -1;
    }
  }
  const int keep_src = fork ? best.src : -1;
  free_slots_.erase(std::find(free_slots_.begin(), free_slots_.end(), dst));
  SlotCache &dc = pc_[static_cast<size_t>(dst)];
  if (in_place) {
    if (kv_shared_) m_.kv_trim(dst, best.R);
    pc_drop(dst, best.R);  // it writes from R on: later snapshots are over pages it overwrites
  } else {
    pc_evict(dst);
    m_.begin(dst);
    if (tiers_ > 1) m_.set_rope_tier(dst, tier);  // a fork's snapshot brings the same tier
  }
  dc.tier = tier;
  dc.hist.clear();
  dc.hist.shrink_to_fit();
  dc.live = -1;
  dc.tick = ++pc_tick_;
  lend_pressure(dst, need, dst, keep_src);
  while (kv_shared_ && !m_.kv_reserve(dst, need))
    if (!pc_evict_lru(dst, keep_src)) {  // counted free above, so not expected: the slot goes back empty, the request waits
      pc_evict(dst);
      free_slots_.push_back(dst);
      return -1;
    }
  if (best.src >= 0) {
    if (in_place && best.snap < 0) m_.resume(dst);
    else if (in_place) m_.snap_restore(dst, pc_[static_cast<size_t>(dst)].snaps[static_cast<size_t>(best.snap)].id);
    else m_.snap_fork(dst, best.src, pc_[static_cast<size_t>(best.src)].snaps[static_cast<size_t>(best.snap)].id);
    pc_[static_cast<size_t>(best.src)].tick = pc_tick_;
    start = best.R;
  }
  {
    std::lock_guard<std::mutex> lk(stats_mu_);
    stats_.prefix_queries += static_cast<uint64_t>(TN);
    stats_.prefix_hits += static_cast<uint64_t>(start);
    if (best.src >= 0) (in_place ? (best.snap < 0 ? stats_.prefix_live : stats_.prefix_restores) : stats_.prefix_forks)++;
  }
  static const bool pc_log = util::env_flag("XEN_PF_LOG");
  if (pc_log && best.src >= 0)
    std::fprintf(stderr, "prefix: %lld of %lld prompt tokens from slot %d (%s) into slot %d\n", static_cast<long long>(start),
                 static_cast<long long>(TN), best.src, in_place ? (best.snap < 0 ? "live state" : "snapshot, in place") : "snapshot, copied", dst);
  return dst;
}

void BatchEngine::loop() {
  std::vector<Submit> submits;
  std::vector<uint64_t> cancels;
  log_at_ = clock::now();
  while (true) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (active_.empty() && waiting_.empty() && resume_.empty()) {
        // the model's idle work (KV lending's refills) until it is done or a request arrives;
        // parked sequences past lend_park_s are dropped first (their pages become lendable), and the wait wakes for the next
        bool more = true;
        while (more) {
          lk.unlock();
          lend_park_timeout();
          int n = 0;
          do {
            n = m_.idle_work(kIdleBudget);
            std::lock_guard<std::mutex> l2(mu_);
            if (stop_ || !submits_.empty() || !cancels_.empty()) break;
          } while (n > 0);
          const double w = lend_park_wait_s();
          lk.lock();
          auto wake = [&] { return stop_ || !submits_.empty() || !cancels_.empty(); };
          if (w < 0) {
            cv_.wait(lk, wake);
            more = false;
          } else {
            more = !cv_.wait_for(lk, std::chrono::duration<double>(w + 0.05), wake);
          }
        }
      }
      if (stop_) return;
      submits.swap(submits_);
      cancels.swap(cancels_);
    }
    events_.clear();
    for (auto &s : submits) waiting_.push_back(std::move(s));
    submits.clear();

    // 1. cancels: waiting ones never run; running ones free their slot now.
    for (uint64_t id : cancels) {
      const auto w = std::find_if(waiting_.begin(), waiting_.end(), [&](const Submit &s) { return s.id == id; });
      bool found = false;
      if (w != waiting_.end()) {
        waiting_.erase(w);
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.waiting--;
        stats_.finished_by[static_cast<int>(Finish::cancelled)]++;
        found = true;
      } else {
        for (size_t i = 0; i < active_.size(); ++i)
          if (active_[i]->id == id) {
            if (pf_group_leave(id)) {
            } else if (pf_pend_.active && pf_pend_.id == id) {  // its split prefill call ends here
              m_.prefill_abort();
              pf_pend_.active = false;
              active_[i]->aborted = true;  // its live state is undefined (its snapshots stay valid)
            }
            retire(i, Finish::cancelled);
            found = true;
            break;
          }
        // a preempted request waiting to resume
        for (auto it = resume_.begin(); !found && it != resume_.end(); ++it)
          if ((*it)->id == id) {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.waiting--;
            stats_.finished_by[static_cast<int>(Finish::cancelled)]++;
            stats_.e2e.observe(dsec(clock::now() - (*it)->t_submit).count());
            resume_.erase(it);
            found = true;
            break;
          }
      }
      if (found) {
        Event e;
        e.id = id;
        e.finish = Finish::cancelled;
        events_.push_back(std::move(e));
      }
    }
    cancels.clear();

    // 2. queue deadline, then admission into free slots
    {
      const auto now = clock::now();
      const auto late = admission_expired(waiting_, cfg_.admission, [&](const Submit &s) { return dsec(now - s.t_submit).count(); });
      for (auto it = late.rbegin(); it != late.rend(); ++it) {
        Event e;
        e.id = waiting_[*it].id;
        e.finish = Finish::expired;
        e.error = "queue deadline exceeded";
        events_.push_back(std::move(e));
        waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(*it));
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.waiting--;
        stats_.finished_by[static_cast<int>(Finish::expired)]++;
      }
    }
    // a shared KV pool admits a request only when its prompt's pages (+ headroom for its first token and
    // drafts) are free; preempted requests resume first, oldest preemption first. place() picks the slot and
    // continues a cached prefix of the prompt (prefilled = pos = the tokens already on the slot).
    auto pf_need = [&](size_t prompt) { return std::min<int64_t>(m_.capacity(), static_cast<int64_t>(prompt) + m_.max_rows()); };
    bool kv_blocked = false;
    while (!resume_.empty() && !free_slots_.empty()) {
      Req &r = *resume_.front();
      int64_t start = 0;
      const bool lps = !r.resumed && r.params.prompt_logprobs >= 0;  // preempted inside its prompt: every row's logprobs again
      const int tier = tier_for(static_cast<int64_t>(r.prompt.size()), r.params.max_tokens - r.generated);
      const int slot = place(r.prompt, pf_need(r.prompt.size()), !lps, start, tier);
      if (slot < 0) {
        kv_blocked = true;
        break;
      }
      r.slot = slot;
      r.tier = tier;
      r.prefilled = static_cast<size_t>(start);
      r.pos = start;
      active_.push_back(std::move(resume_.front()));
      resume_.pop_front();
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.waiting--;
      stats_.running = static_cast<uint32_t>(active_.size());
    }
    // In-flight prefix sharing (pc_leader): the requests still waiting for a running request's prefix keep waiting (a
    // free slot stays theirs), the others are released; each leader's next cut is the nearest one waited for.
    size_t nheld = 0;
    for (auto &a : active_) a->share_cut = -1;
    for (Submit &w : waiting_) {
      if (!w.held) continue;
      Req *b = nullptr;
      for (auto &a : active_)
        if (a->id == w.hold_id) b = a.get();
      if (b && !b->aborted && static_cast<int64_t>(b->prefilled) < w.hold_len) {
        b->share_cut = b->share_cut < 0 ? w.hold_len : std::min(b->share_cut, w.hold_len);
        ++nheld;
      } else {
        w.held = false;
      }
    }
    // the waiting requests admission picks from (queue order): the ones not held
    struct Elig {
      const std::deque<Submit> &q;
      std::vector<size_t> idx;
      const Submit &operator[](size_t i) const { return q[idx[i]]; }
      size_t size() const { return idx.size(); }
    } elig{waiting_, {}};
    for (size_t i = 0; i < waiting_.size(); ++i)
      if (!waiting_[i].held) elig.idx.push_back(i);
    while (!kv_blocked && resume_.empty() && elig.size() > 0 && free_slots_.size() > nheld) {
      const auto now = clock::now();
      const size_t ke = admission_pick(
          elig, cfg_.admission, [](const Submit &s) { return s.params.max_tokens; },
          [&](const Submit &s) { return dsec(now - s.t_submit).count(); });
      const size_t k = elig.idx[ke];
      int slot = -1, tier = 0;
      int64_t start = 0;
      {
        Submit &w = waiting_[k];
        const bool bad_w = w.prompt.empty() || static_cast<int64_t>(w.prompt.size()) >= m_.capacity() || w.params.max_tokens <= 0 ||
                           (w.params.prompt_logprobs >= 0 && !m_.prompt_logprobs());
        if (!bad_w) {
          // prompt logprobs need every prompt row's logits: no cached prefix
          tier = tier_for(static_cast<int64_t>(w.prompt.size()), w.params.max_tokens);
          int64_t cut = -1;
          if (Req *b = w.params.prompt_logprobs < 0 ? pc_leader(w.prompt, tier, cut) : nullptr) {
            w.held = true;  // it waits for b's prefill to pass `cut`; the requests behind it go on
            w.hold_id = b->id;
            w.hold_len = cut;
            b->share_cut = b->share_cut < 0 ? cut : std::min(b->share_cut, cut);
            ++nheld;
            elig.idx.erase(elig.idx.begin() + static_cast<std::ptrdiff_t>(ke));
            static const bool hold_log = util::env_flag("XEN_PF_LOG");
            if (hold_log)
              std::fprintf(stderr, "prefix: request %llu waits for request %llu's prefill to reach %lld of its %zu prompt tokens (at %zu)\n",
                           static_cast<unsigned long long>(w.id), static_cast<unsigned long long>(b->id), static_cast<long long>(cut), w.prompt.size(), b->prefilled);
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.prefix_holds++;
            continue;
          }
          slot = place(w.prompt, pf_need(w.prompt.size()), w.params.prompt_logprobs < 0, start, tier);
          if (slot < 0) break;  // waits for pages (the pick stays first)
        }
      }
      Submit s = std::move(waiting_[k]);
      waiting_.erase(waiting_.begin() + static_cast<std::ptrdiff_t>(k));
      elig.idx.erase(elig.idx.begin() + static_cast<std::ptrdiff_t>(ke));
      for (size_t &x : elig.idx) x -= x > k;
      {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.waiting--;
        stats_.queue_time.observe(dsec(now - s.t_submit).count());
      }
      if (slot < 0) {  // a request that cannot run
        Event e;
        e.id = s.id;
        e.finish = s.params.max_tokens <= 0 ? Finish::length : Finish::error;
        if (e.finish == Finish::error)
          e.error = s.params.prompt_logprobs >= 0 && !m_.prompt_logprobs() ? "this engine cannot return prompt logprobs"
                                                                             : "the prompt does not fit the model's context";
        events_.push_back(std::move(e));
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.finished_by[static_cast<int>(e.finish)]++;
        continue;
      }
      auto r = std::make_unique<Req>();
      r->id = s.id;
      r->prompt = std::move(s.prompt);
      r->params = s.params;
      // greedy requests with logprobs stay on the device's pick when the model computes logprobs itself
      r->host = !r->params.greedy() || (r->params.logprobs >= 0 && !m_.device_logprobs());
      if (!r->params.greedy()) {
        const int64_t V = m_.vocab();
        if (std::any_of(r->prompt.begin(), r->prompt.end(), [V](int32_t t) { return t >= V; })) {  // image ids
          std::vector<int32_t> ids;
          ids.reserve(r->prompt.size());
          for (const int32_t t : r->prompt)
            if (t < V) ids.push_back(t);
            else if (const int32_t mt = media_lp(t); mt >= 0) ids.push_back(mt);
          r->sampler = std::make_unique<Sampler>(r->params, ids);
        } else {
          r->sampler = std::make_unique<Sampler>(r->params, r->prompt);
        }
      }
      if (r->params.constraint) r->constraint = r->params.constraint->start();
      if (r->params.prompt_logprobs >= 0) {
        r->prompt_lps.resize(r->prompt.size());
        r->prompt_lps[0].logprob = std::numeric_limits<float>::quiet_NaN();
      }
      r->slot = slot;
      r->tier = tier;
      r->prefilled = static_cast<size_t>(start);
      r->pos = start;
      r->prompt_len = r->prompt.size();
      if (cfg_.lookup > 0) {  // the lookup's history starts from the prompt (a resumed request's holds its output too)
        r->lk_hist = r->prompt;
        r->lk_idx.clear();
        r->lk_indexed = 0;
        r->lk_pend.clear();
        r->lk_used = false;
      }
      r->t_submit = s.t_submit;
      r->t_admit = now;
      active_.push_back(std::move(r));
      std::lock_guard<std::mutex> lk(stats_mu_);
      stats_.running = static_cast<uint32_t>(active_.size());
    }

    Histogram ttft_add, prefill_add;
    uint64_t prompt_add = 0, gen_add = 0;
    std::vector<std::pair<size_t, Finish>> done;  // (active index, reason), applied after the step

    // 3. one prefill call: the request with the fewest prompt tokens left (ties: the earliest
    //    admitted); sized to the stall budget while other requests decode
    {
      size_t best = active_.size();
      bool decoding = false;
      for (size_t i = 0; i < active_.size(); ++i) {
        const Req &r = *active_[i];
        if (r.prefilled == r.prompt.size()) {
          decoding |= r.next >= 0;
          continue;
        }
        if (best == active_.size()) best = i;
        else {
          const Req &b = *active_[best];
          const size_t lr = r.prompt.size() - r.prefilled, lb = b.prompt.size() - b.prefilled;
          if (lr < lb || (lr == lb && r.t_admit < b.t_admit)) best = i;
        }
      }
      // The call's tail: the prompt's position advances; the last prompt row gives the first token.
      // ms: the model's call alone (the parts' sum of a split call). The first token's sampling and the first drafts
      // below are not prefill rows, so they stay out of the fit.
      // pre: a member of a call for several requests - its first token was read with the other members'
      // (pf_group_read), before the first member's first drafts reuse the head; such a call's time is not a sample of the fit.
      auto complete = [&](size_t i, int n, size_t depth, bool rows_lp, std::pair<double, double> fit, clock::time_point t0, double ms, int parts,
                          PfPending::Member *pre = nullptr) {
        const auto t_tail = clock::now();
        Req &r = *active_[i];
        r.prefilled += static_cast<size_t>(n);
        r.pos += n;
        // the cut's snapshot captured inside this call (before the call end's: positions ascend)
        if (r.snap_armed > 0) {
          if (pc_on_ && r.snap_armed >= cfg_.prefix_min_tokens) pc_snapshot(r.slot, r.snap_armed, r.snap_armed == pc_cut(r), true);
          r.snap_armed = -1;
        }
        // a snapshot of the sequence state after every call that ends at a multiple of the model's snap_align
        // (the prefix cache keeps the useful ones); a mark when the call ends at the request's cut (pc_cut) or ends its
        // prompt
        if (const int64_t e = static_cast<int64_t>(r.prefilled), al = std::max(1, m_.snap_align());
            pc_on_ && e >= cfg_.prefix_min_tokens && e % al == 0)
          pc_snapshot(r.slot, e, r.prefilled == r.prompt.size() || e == pc_cut(r));
        // prompt tokens counted once (a recompute after a preemption is not new prompt input)
        if (const size_t hi = std::min(r.prefilled, r.prompt_len); hi > r.prompt_counted) {
          prompt_add += static_cast<uint64_t>(hi - r.prompt_counted);
          r.prompt_counted = hi;
        }
        if (r.prefilled == r.prompt.size()) {  // the prompt's last row gives the first token
          Event e;
          e.id = r.id;
          std::vector<float> row;
          if (pre) row = std::move(pre->row);
          else if (r.host) m_.prefill_logits(row);
          std::vector<Pick> pk(1);
          pk[0].r = &r;
          pk[0].row = r.host ? row.data() : nullptr;
          pk[0].tok = pre ? pre->tok : m_.prefill_pick();  // greedy requests keep the device's pick (a sampler overrides it)
          if (r.host) sample_rows(pk);
          else if (pre && pre->has_lp) {
            pk[0].lp = std::move(pre->lp);
            pk[0].has_lp = true;
          } else if (!pre && r.params.logprobs >= 0) {
            m_.prefill_logprobs(pk[0].tok, r.params.logprobs, pk[0].lp);
            pk[0].has_lp = true;
          }
          if (!r.prompt_lps.empty()) e.prompt_logprobs = std::move(r.prompt_lps);
          const Finish f = emit(r, pk[0].tok, pk[0].has_lp ? &pk[0].lp : nullptr, e);
          // a first token that does not finish its request goes to the client here (before its first
          // drafts and the other requests' step); TTFT is taken when it is handed over
          const auto t1 = clock::now();
          if (!r.resumed) {
            ttft_add.observe(dsec(t1 - r.t_submit).count());
            prefill_add.observe(dsec(t1 - r.t_admit).count());
          }
          if (f == Finish::none) {
            std::vector<Event> now(1);
            now[0] = std::move(e);
            publish_early(now);
          }
          if (f == Finish::none && m_.draft_depth(1) > 0) {
            // the drafts for the step it joins: as many as a step of the decoding requests allows
            int dec_n = 0;
            for (const auto &q : active_) dec_n += q->prefilled == q->prompt.size();
            const auto dn = r.sampler ? r.sampler->draft_noise() : Sampler::DraftNoise{};
            m_.draft_sampling(r.slot, dn.key, dn.pos, dn.inv_t);
            r.drafts = m_.first_drafts(r.slot, pk[0].tok, m_.draft_depth(std::max(1, dec_n)), std::max(1, dec_n));
            r.tree = m_.draft_parents(r.slot);
          }
          r.resumed = false;
          ++gen_add;
          if (f != Finish::none) {
            e.finish = f;
            events_.push_back(std::move(e));
            done.push_back({i, f});
          }
        }
        // XEN_PF_LOG=1 (diagnostic): every prefill call, its prediction, the decode gap and the call's tail (the
        // snapshots, the first token and the first drafts)
        static const bool pf_log = util::env_flag("XEN_PF_LOG");
        if (pf_log)
          std::fprintf(stderr, "pf-call at %.1f rows %d depth %zu decoding %d predicted %.1f took %.1f ms in %d parts; since the last decode step %.1f ms; tail %.1f ms\n",
                       std::chrono::duration<double, std::milli>(t0.time_since_epoch()).count(), n, depth, decoding ? 1 : 0, fit.first + fit.second * n, ms, parts,
                       dsec(clock::now() - last_step_end_).count() * 1e3, dsec(clock::now() - t_tail).count() * 1e3);
        if (!rows_lp && !pre) record_prefill(n, ms);
        log_prefill_tokens_ += static_cast<uint64_t>(n);
        log_prefill_s_ += ms * 1e-3;
        ++log_prefill_calls_;
      };
      // the members of a call for several requests complete in the call's order: every first token was read
      // first (pf_group_read), then each member's snapshots, first token and first drafts
      auto finish_group = [&]() {
        pf_group_read();
        bool first = true;
        for (auto &mb : pf_pend_.group) {
          size_t i = active_.size();
          for (size_t k = 0; k < active_.size(); ++k)
            if (active_[k]->id == mb.id) i = k;
          if (i == active_.size()) throw std::logic_error("batch engine: a merged prefill call lost a request");
          complete(i, mb.n, mb.depth, false, pf_pend_.fit, pf_pend_.t0, first ? pf_pend_.ms : 0.0, pf_pend_.parts, &mb);
          first = false;
        }
        pf_pend_.group.clear();
      };
      if (pf_pend_.active && !pf_pend_.group.empty()) {
        if (!pf_pend_.finished) {  // the next part (all the rest once nothing decodes)
          const auto ta = clock::now();
          pf_pend_.finished = m_.prefill_advance(decoding ? pf_pend_.layers : m_.prefill_layers());
          pf_pend_.ms += dsec(clock::now() - ta).count() * 1e3;
          ++pf_pend_.parts;
        }
        if (pf_pend_.finished) {
          pf_pend_.active = false;
          finish_group();
        }
      } else if (pf_pend_.active) {
        // the next part of a split call (all the rest once nothing decodes)
        size_t i = active_.size();
        for (size_t k = 0; k < active_.size(); ++k)
          if (active_[k]->id == pf_pend_.id) i = k;
        if (i == active_.size()) throw std::logic_error("batch engine: a split prefill call lost its request");
        const auto ta = clock::now();
        const bool fin = m_.prefill_advance(decoding ? pf_pend_.layers : m_.prefill_layers());
        pf_pend_.ms += dsec(clock::now() - ta).count() * 1e3;
        static const bool part_log = util::env_flag("XEN_PF_LOG");  // each part of a split call
        if (part_log)
          std::fprintf(stderr, "pf-part at %.1f took %.1f ms%s\n", std::chrono::duration<double, std::milli>(ta.time_since_epoch()).count(),
                       dsec(clock::now() - ta).count() * 1e3, fin ? " (last)" : "");
        ++pf_pend_.parts;
        if (fin) {
          pf_pend_.active = false;
          complete(i, pf_pend_.n, pf_pend_.depth, false, pf_pend_.fit, pf_pend_.t0, pf_pend_.ms, pf_pend_.parts);
        }
      } else if (best < active_.size()) {
        const size_t i = best;
        Req &r = *active_[i];
        const bool rows_lp = r.params.prompt_logprobs >= 0 && !r.resumed;  // a resumed prompt sent them already
        const int L = m_.prefill_layers();
        // whole calls in parts for every prompt (StepModel:prefill_long_parts)
        const bool long_parts = decoding && !rows_lp && cfg_.prefill_part_ms > 0 && L > 1 && m_.prefill_long_parts();
        // the next call of a request: its rows (the call it runs alone: who else waits changes nothing)
        auto call_rows = [&](const Req &q, bool lp_rows) {
        const bool lp_long = decoding && !lp_rows && cfg_.prefill_part_ms > 0 && L > 1 && m_.prefill_long_parts();
        int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(m_.prefill_chunk()), q.prompt.size() - q.prefilled));
        if (decoding && !lp_long) n = std::min(n, budget_rows());
        if (decoding && !lp_long) {
          // a budgeted call keeps snap_align's rule too (a multiple of it, no tail under it left behind): the model's
          // calls of fewer rows run another GDN path (the chunked scan needs 64-row chunks), which gives other bits than
          // one call
          const int64_t al = std::max(1, m_.snap_align()), rem = static_cast<int64_t>(q.prompt.size() - q.prefilled);
          int64_t nn = n;
          if (al > 1 && nn < rem) {
            nn = std::max(al, nn / al * al);
            if (nn >= rem) nn = rem;
            else if (rem - nn < al) nn = rem - al >= al ? (rem - al) / al * al : rem;
          }
          n = static_cast<int>(nn);
        }
        if (!decoding) {
          n = pf_call_rows(q);
        } else {
          const int64_t p = static_cast<int64_t>(q.prefilled);
          // a call ends where a waiting request continues from (pc_leader), and at the request's cut, unless the model
          // snapshots inside a call (below)
          if (const int64_t h = q.share_cut; h > p && h < p + n && !pc_inside(q)) n = static_cast<int>(h - p);
          if (const int64_t a = pc_cut(q); a > p && a < p + n && !pc_inside(q)) n = static_cast<int>(a - p);
        }
        return n;
        };
        const int n = call_rows(r, rows_lp);
        // the position a request's call of nq rows captures a snapshot at (-1: none)
        auto arm_pos = [&](const Req &q, int nq) -> int64_t {
          if (!pc_inside(q)) return -1;
          const int64_t lo = static_cast<int64_t>(q.prefilled), hi = lo + nq, a = pc_cut(q), h = q.share_cut;
          return h > lo && h < hi ? h : a > lo && a < hi ? a : -1;
        };
        // a cut inside this call: the model captures its snapshot there
        // One position per call: where a waiting request continues from (pc_leader) before the request's own cut. The
        // call itself is the one the request runs alone, so its rows do not depend on who waits for it.
        // EngineConfig::prefill_merge: the other waiting prompts whose whole remaining prompt is the one
        // call they would run alone, with nothing captured inside it, share this call when the model says each may
        // (StepModel::prefill_mergeable); the members' rows are their own calls', so a reply does not depend on who waited.
        std::vector<std::pair<size_t, int>> grp;  // (active index, rows), the best request first
        int grp_rows = 0;                         // the call's rows with the padding
        if (const int mmax = std::min(cfg_.prefill_merge, m_.prefill_merge_max()); mmax > 1 && !rows_lp) {
          auto whole = [&](const Req &q, int nq) {
            return static_cast<size_t>(nq) == q.prompt.size() - q.prefilled && arm_pos(q, nq) < 0 && m_.prefill_mergeable(q.slot, q.prompt.data() + q.prefilled, nq);
          };
          if (whole(r, n)) {
            std::vector<size_t> cand;
            for (size_t k = 0; k < active_.size(); ++k)
              if (k != i && active_[k]->prefilled < active_[k]->prompt.size() && !(active_[k]->params.prompt_logprobs >= 0 && !active_[k]->resumed)) cand.push_back(k);
            std::sort(cand.begin(), cand.end(), [&](size_t a, size_t b) {
              const Req &x = *active_[a], &y = *active_[b];
              const size_t lx = x.prompt.size() - x.prefilled, ly = y.prompt.size() - y.prefilled;
              return lx != ly ? lx < ly : x.t_admit != y.t_admit ? x.t_admit < y.t_admit : a < b;
            });
            const int pad = std::max(1, m_.prefill_merge_pad()), cap = m_.prefill_merge_rows();
            int total = n;
            grp.push_back({i, n});
            for (size_t k : cand) {
              if (static_cast<int>(grp.size()) >= mmax) break;
              const Req &q = *active_[k];
              const int nq = call_rows(q, false);
              const int at = (total + pad - 1) / pad * pad;
              if (at + nq > cap || !whole(q, nq)) continue;
              grp.push_back({k, nq});
              total = at + nq;
            }
            if (grp.size() < 2 || total > cap) grp.clear();
            grp_rows = total;
          }
        }
        r.snap_armed = -1;
        if (const int64_t arm = arm_pos(r, n); arm > 0) {
          m_.snap_arm(r.slot, arm);
          r.snap_armed = arm;
        }
        const auto t0 = clock::now();
        const auto fit = pf_fit();
        const size_t depth = r.prefilled;
        const size_t end = r.prefilled + static_cast<size_t>(n);
        const int32_t next = end < r.prompt.size() ? r.prompt[end] : -1;
        // while others decode, a prompt prefilled in this one call (a short prompt) whose call
        // is predicted longer than prefill_part_ms runs in parts of whole layers, a decode step between
        // them (bit-identical to one call). Longer prompts keep their stall-budget calls whole.
        int parts = 1;
        if (decoding && !rows_lp && cfg_.prefill_part_ms > 0 && L > 1 && (long_parts || (r.prefilled == 0 && end == r.prompt.size())))
          parts = std::clamp(static_cast<int>(std::ceil((fit.first + fit.second * n) / cfg_.prefill_part_ms)), 1, L);
        if (!grp.empty()) {  // one call for the group, in parts while others decode
          const int total = grp_rows;
          int gparts = 1;
          if (decoding && cfg_.prefill_part_ms > 0 && L > 1)
            gparts = std::clamp(static_cast<int>(std::ceil((fit.first + fit.second * total) / cfg_.prefill_part_ms)), 1, L);
          pf_pend_ = PfPending{};
          pf_pend_.active = true;
          pf_pend_.id = r.id;
          pf_pend_.n = total;
          pf_pend_.layers = (L + gparts - 1) / gparts;
          pf_pend_.fit = fit;
          pf_pend_.t0 = t0;
          std::vector<StepModel::PrefillSeg> segs;
          for (const auto &[k, nq] : grp) {
            Req &q = *active_[k];
            q.snap_armed = -1;
            pf_pend_.group.push_back(PfPending::Member{q.id, nq, q.prefilled});
            segs.push_back(StepModel::PrefillSeg{q.slot, q.prompt.data() + q.prefilled, nq, -1});
          }
          m_.prefill_begin_many(segs);
          pf_pend_.finished = m_.prefill_advance(std::max(1, pf_pend_.layers));
          pf_pend_.ms = dsec(clock::now() - t0).count() * 1e3;
          pf_pend_.parts = 1;
          if (pf_pend_.finished) {
            pf_pend_.active = false;
            finish_group();
          }
        } else if (parts > 1) {
          pf_pend_ = PfPending{};
          pf_pend_.active = true;
          pf_pend_.id = r.id;
          pf_pend_.n = n;
          pf_pend_.depth = depth;
          pf_pend_.layers = (L + parts - 1) / parts;
          pf_pend_.fit = fit;
          pf_pend_.t0 = t0;
          m_.prefill_begin(r.slot, r.prompt.data() + r.prefilled, n, next);
          const bool fin = m_.prefill_advance(pf_pend_.layers);
          pf_pend_.ms = dsec(clock::now() - t0).count() * 1e3;
          if (util::env_flag("XEN_PF_LOG"))  // each part of a split call
            std::fprintf(stderr, "pf-part at %.1f took %.1f ms (first)\n", std::chrono::duration<double, std::milli>(t0.time_since_epoch()).count(), pf_pend_.ms);
          pf_pend_.parts = 1;
          if (fin) {
            pf_pend_.active = false;
            complete(i, n, depth, false, fit, t0, pf_pend_.ms, 1);
          }
        } else {
        if (rows_lp && m_.device_logprobs()) {
          const size_t base = r.prefilled;
          const int k = r.params.prompt_logprobs;
          // row j scores prompt token base + j + 1 (the last prompt row scores the first output: none)
          std::vector<int32_t> tg(static_cast<size_t>(n), -1);
          for (int j = 0; j < n; ++j) {
            const size_t p = base + static_cast<size_t>(j) + 1;
            if (p < r.prompt.size()) tg[static_cast<size_t>(j)] = media_lp(r.prompt[p]);
          }
          std::vector<TokenLogprob> all(static_cast<size_t>(n));
          // one many-row head per chunk when the model has it, else a head per 4-row group
          if (m_.prefill_row_logprobs(r.slot, r.prompt.data() + r.prefilled, n, next, tg.data(), k, all.data())) {
            for (int j = 0; j < n; ++j)
              if (tg[static_cast<size_t>(j)] >= 0) r.prompt_lps[base + static_cast<size_t>(j) + 1] = std::move(all[static_cast<size_t>(j)]);
          } else {
          std::vector<int32_t> t;
          std::vector<TokenLogprob> lps;
          m_.prefill_each(r.slot, r.prompt.data() + r.prefilled, n, next, [&](int row0, int rows) {
            t.assign(tg.begin() + row0, tg.begin() + row0 + rows);
            lps.assign(static_cast<size_t>(rows), TokenLogprob{});
            m_.row_logprobs(rows, t.data(), k, lps.data());
            for (int j = 0; j < rows; ++j)
              if (t[static_cast<size_t>(j)] >= 0) r.prompt_lps[base + static_cast<size_t>(row0 + j) + 1] = std::move(lps[static_cast<size_t>(j)]);
          });
          }
        } else if (rows_lp) {
          const size_t base = r.prefilled;
          const int k = r.params.prompt_logprobs;
          m_.prefill_rows(r.slot, r.prompt.data() + r.prefilled, n, next, [&](int row0, int rows, const float *lg) {
            // row j's logits score prompt token base + j + 1 (the last prompt row scores the first output)
            parallel(static_cast<size_t>(rows), [&](size_t j) {
              const size_t p = base + static_cast<size_t>(row0) + j + 1;
              if (p < r.prompt.size() && media_lp(r.prompt[p]) >= 0)
                Sampler::raw_logprobs(lg + j * static_cast<size_t>(m_.vocab()), m_.vocab(), media_lp(r.prompt[p]), k, r.prompt_lps[p]);
            });
          });
        } else {
          m_.prefill(r.slot, r.prompt.data() + r.prefilled, n, next);
        }
        complete(i, n, depth, rows_lp, fit, t0, dsec(clock::now() - t0).count() * 1e3, 1);
        }
      }
    }

    // 4. one decode step over every request with a token to feed
    {
      std::vector<size_t> dec;
      for (size_t i = 0; i < active_.size(); ++i) {
        const Req &r = *active_[i];
        const bool finished = std::any_of(done.begin(), done.end(), [&](const auto &d) { return d.first == i; });
        // a request whose prompt completed above joins this step: its first token went out
        // early and its first drafts were made for this step.
        if (!finished && r.prefilled == r.prompt.size() && r.next >= 0 &&
            !std::any_of(events_.begin(), events_.end(), [&](const Event &e) { return e.id == r.id; }))
          dec.push_back(i);
      }
      if (!dec.empty()) dec = kv_ensure(dec, done);  // the step's pages (a shared pool may preempt)
      if (!dec.empty() && m_.draft_depth(1) > 0) {
        const auto t0 = clock::now();
        spec_step(dec, done, gen_add);
        step_ms_.push_back(dsec(clock::now() - t0).count() * 1e3);
        last_step_end_ = clock::now();
      } else if (!dec.empty()) {
        std::vector<int> slots;
        std::vector<int32_t> toks;
        bool host = false;
        for (size_t i : dec) {
          slots.push_back(active_[i]->slot);
          toks.push_back(active_[i]->next);
          host |= active_[i]->host;
        }
        const auto t0 = clock::now();
        m_.step(slots, toks.data());
        const int rows = static_cast<int>(dec.size());
        std::vector<int32_t> dev(static_cast<size_t>(rows));
        m_.picks(rows, dev.data());
        static const bool step_log = util::env_flag("XEN_PF_LOG");  // the steps beside the calls
        if (step_log)
          std::fprintf(stderr, "step at %.1f rows %d took %.1f ms\n", std::chrono::duration<double, std::milli>(t0.time_since_epoch()).count(), rows,
                       dsec(clock::now() - t0).count() * 1e3);
        int cn = 0;  // candidate rows (cands_for), 0 = full rows
        if (host) {
          std::vector<const Req *> hr;
          for (size_t i : dec) hr.push_back(active_[i].get());
          if (const int T = cands_for(hr); T > 0) cn = m_.logits_cands(rows, T, cand_ids_, cand_vals_);
          else m_.logits(rows, logits_);
        }
        std::vector<Pick> pk(static_cast<size_t>(rows));
        std::vector<Pick> hp;  // the host rows
        for (int k = 0; k < rows; ++k) {
          Pick &p = pk[static_cast<size_t>(k)];
          p.r = active_[dec[static_cast<size_t>(k)]].get();
          p.tok = dev[static_cast<size_t>(k)];
          if (p.r->host && cn > 0) {
            p.cid = cand_ids_.data() + static_cast<size_t>(k) * cn;
            p.cval = cand_vals_.data() + static_cast<size_t>(k) * cn;
            p.cn = cn;
          } else if (p.r->host) {
            p.row = logits_.data() + static_cast<size_t>(k) * static_cast<size_t>(m_.vocab());
          }
        }
        for (auto &p : pk)
          if (p.r->host) hp.push_back(std::move(p));
        sample_rows(hp);
        if (cn > 0 && std::any_of(hp.begin(), hp.end(), [](const Pick &p) { return p.tok < 0; })) {
          // a kept set left empty (every candidate -inf, nothing drawn): those rows on the full row (sample())
          m_.logits(rows, logits_);
          std::vector<Pick> again;
          std::vector<size_t> at;
          for (size_t h = 0, k = 0; k < pk.size(); ++k) {
            if (!pk[k].r->host) continue;
            if (hp[h].tok < 0) {
              Pick p = std::move(hp[h]);
              p.cn = 0;
              p.row = logits_.data() + k * static_cast<size_t>(m_.vocab());
              again.push_back(std::move(p));
              at.push_back(h);
            }
            ++h;
          }
          sample_rows(again);
          for (size_t i = 0; i < at.size(); ++i) hp[at[i]] = std::move(again[i]);
        }
        for (size_t k = 0, h = 0; k < pk.size(); ++k)
          if (pk[k].r->host) pk[k] = std::move(hp[h++]);
        {  // device logprobs of the greedy rows that want them, and of the sampled candidate rows' draws
          std::vector<const Req *> lr(pk.size(), nullptr);
          std::vector<int32_t> lt(dev);
          bool any = false;
          for (size_t k = 0; k < pk.size(); ++k)
            if (pk[k].r->params.logprobs >= 0 && (!pk[k].r->host || (pk[k].cn > 0 && !pk[k].has_lp))) lr[k] = pk[k].r, lt[k] = pk[k].tok, any = true;
          if (any) {
            std::vector<TokenLogprob> lps;
            device_lps(lr, lt.data(), lps);
            for (size_t k = 0; k < pk.size(); ++k)
              if (lr[k]) {
                pk[k].lp = std::move(lps[k]);
                pk[k].has_lp = true;
              }
          }
        }
        step_ms_.push_back(dsec(clock::now() - t0).count() * 1e3);
        last_step_end_ = clock::now();
        std::vector<Event> now;  // the going requests' tokens at once (a finish at the iteration's end)
        for (int k = 0; k < rows; ++k) {
          Pick &p = pk[static_cast<size_t>(k)];
          Req &r = *p.r;
          r.pos += 1;
          Event e;
          e.id = r.id;
          const Finish f = emit(r, p.tok, p.has_lp ? &p.lp : nullptr, e);
          ++gen_add;
          e.finish = f;
          if (f == Finish::none) {
            now.push_back(std::move(e));
          } else {
            events_.push_back(std::move(e));
            done.push_back({dec[static_cast<size_t>(k)], f});
          }
        }
        publish_early(now);
      }
    }

    // 5. retire (highest index first so the others stay valid), stats, hand off; the preempted requests leave
    //    their slots for resume_ (the same pass, so every index stays valid)
    {
      std::vector<std::pair<size_t, int>> out;  // (index, 0 retire / 1 preempted)
      for (const auto &d : done) out.push_back({d.first, 0});
      for (size_t i = 0; i < active_.size(); ++i)
        if (active_[i]->preempt) out.push_back({i, 1});
      std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
      std::vector<std::unique_ptr<Req>> pre;
      for (const auto &[i, kind] : out) {
        if (kind == 0) {
          const auto f = std::find_if(done.begin(), done.end(), [&](const auto &d) { return d.first == i; })->second;
          retire(i, f);
          continue;
        }
        auto r = std::move(active_[i]);
        active_.erase(active_.begin() + static_cast<std::ptrdiff_t>(i));
        free_slots_.push_back(r->slot);  // its pages went back in kv_ensure
        r->slot = -1;
        r->preempt = false;
        if (r->generated > 0) {  // recompute the prompt + every emitted token; the next token then comes from that prefill
          r->prompt.resize(r->prompt_len);
          r->prompt.insert(r->prompt.end(), r->out.begin(), r->out.end());
          r->resumed = true;
        }
        r->prompt_lps.clear();
        if (!r->resumed && r->params.prompt_logprobs >= 0) {  // preempted inside its prompt: they are computed again
          r->prompt_lps.resize(r->prompt.size());
          r->prompt_lps[0].logprob = std::numeric_limits<float>::quiet_NaN();
        }
        r->prefilled = 0;
        r->pos = 0;
        r->drafts.clear();
        r->tree.clear();
        pre.push_back(std::move(r));
      }
      // oldest admitted resumes first
      std::sort(pre.begin(), pre.end(), [](const auto &a, const auto &b) { return a->t_admit < b->t_admit; });
      for (auto &r : pre) resume_.push_back(std::move(r));
      if (!pre.empty()) {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.preemptions += pre.size();
        stats_.waiting += static_cast<uint32_t>(pre.size());
        stats_.running = static_cast<uint32_t>(active_.size());
      }
    }
    if (kv_shared_) kv_stats();
    if (pc_on_) pc_stats();
    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      ++stats_.steps;
      stats_.prompt_tokens += prompt_add;
      stats_.generation_tokens += gen_add;
      merge(stats_.ttft, ttft_add);
      merge(stats_.prefill_time, prefill_add);
    }
    if (!events_.empty() && sink_) sink_->on_events(events_.data(), events_.size());

    if (cfg_.log_interval_s > 0) {
      const auto now = clock::now();
      if (dsec(now - log_at_).count() >= cfg_.log_interval_s && (!step_ms_.empty() || log_prefill_tokens_)) {
        std::vector<double> v = step_ms_;
        std::sort(v.begin(), v.end());
        char kvs[96] = "", pcs[128] = "";
        const auto st = stats();
        if (kv_shared_)
          std::snprintf(kvs, sizeof kvs, "; KV pool %lld / %lld tokens used, %llu preemptions", static_cast<long long>(st.kv_used_tokens),
                        static_cast<long long>(st.kv_pool_tokens), static_cast<unsigned long long>(st.preemptions));
        if (pc_on_)  // 
          std::snprintf(pcs, sizeof pcs, "; prefix cache %llu / %llu prompt tokens hit, %u snapshots", static_cast<unsigned long long>(st.prefix_hits),
                        static_cast<unsigned long long>(st.prefix_queries), st.prefix_snapshots);
        // host RAM: a growing RSS shows up here long before the kernel's OOM killer ends the process (exit 137, no log)
        char hms[96] = "";
        const util::HostMem hm = util::host_mem();
        if (hm.ok)
          std::snprintf(hms, sizeof hms, "; host RSS %.1f GiB, %.1f / %.1f GiB free", static_cast<double>(util::self_rss()) / 1073741824.0,
                        static_cast<double>(hm.available) / 1073741824.0, static_cast<double>(hm.total) / 1073741824.0);
        std::fprintf(stderr, "engine: %zu running, %zu waiting; %zu decode steps median %.2f ms p90 %.2f; prefill %llu tokens in %llu calls at %.0f tok/s%s%s%s\n",
                     active_.size(), waiting_.size() + resume_.size(), v.size(), v.empty() ? 0.0 : v[v.size() / 2],
                     v.empty() ? 0.0 : v[std::min(v.size() - 1, v.size() * 9 / 10)],
                     static_cast<unsigned long long>(log_prefill_tokens_), static_cast<unsigned long long>(log_prefill_calls_),
                     log_prefill_s_ > 0 ? static_cast<double>(log_prefill_tokens_) / log_prefill_s_ : 0.0, kvs, pcs, hms);
        step_ms_.clear();
        log_prefill_tokens_ = 0;
        log_prefill_s_ = 0;
        log_prefill_calls_ = 0;
        log_at_ = now;
      }
    }
  }
}

}  // namespace xenithium::core
