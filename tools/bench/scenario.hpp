// xenithium-bench: a scenario is one benchmark case family (decode, prefill...). It declares its parameters and
// measures against a Session. Adding one = a file in scenarios/ and a line in registry.cpp.
#pragma once

#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "format.hpp"
#include "params.hpp"
#include "session.hpp"
#include "types.hpp"

namespace xenithium::bench {

class Scenario {
 public:
  virtual ~Scenario() = default;
  virtual std::string_view id() const = 0;  // "prefill": the name on the command line and in run files
  virtual std::string_view title() const = 0;  // the report's heading
  virtual std::string_view about() const = 0;  // what it measures and why, a sentence or two (`list`)
  virtual std::span<const ParamSpec> params() const = 0;
  // Throws std:invalid_argument for a parameter value the scenario cannot use (make_plan calls it, so a bad value is
  // refused before anything is sent). `params` holds every declared parameter.
  virtual void check(const Params & /*params*/) const {}
  // Measures. `params` holds every declared parameter (defaults merged with the suite's and the user's values). Fills
  // rows, requests / errors / first_error; the runner sets id, title and seconds.
  virtual ScenarioResult run(Session &session, const Params &params) const = 0;
};

// Every scenario, in report order.
std::vector<std::unique_ptr<Scenario>> all_scenarios();

// One step of a suite: a scenario with the parameters this suite changes.
struct SuiteStep {
  std::string scenario;
  Params params;
};
struct Suite {
  std::string id;
  std::string about;
  int minutes;  // about how long it takes on two Arc Pro B70 with the dense 27B (the tool prints it before starting)
  std::vector<SuiteStep> steps;
};
const std::vector<Suite> &suites();

// ---- shared by the scenarios (scenarios/common.cpp) ----

// Metric names used by more than one scenario and read back by the report's headline.
namespace metric {
inline constexpr std::string_view kDecode = "decode";  // tok/s per request after its first update
inline constexpr std::string_view kTtft = "TTFT";  // ms
inline constexpr std::string_view kTokensPerUpdate = "tok/update";
inline constexpr std::string_view kUpdateGap = "update gap";  // ms between streamed updates
inline constexpr std::string_view kPrefill = "prefill";
inline constexpr std::string_view kPromptTokens = "prompt";  // tokens, as the server counted them
inline constexpr std::string_view kTotal = "total";  // tok/s of all requests of a round together
inline constexpr std::string_view kPower = "power";  // W of all cards over a row's requests
inline constexpr std::string_view kEfficiency = "efficiency";  // tokens per joule (= tok/s per watt)
inline constexpr std::string_view kQualityScore = "score";  // % of the quality questions answered correctly
}  // namespace metric

// Counts a sample into the result's requests / errors; true when it is usable.
bool tally(ScenarioResult &result, const Sample &sample);
// The decode figures of one finished request: decode tok/s, TTFT, tokens per update, update gap.
void add_decode_metrics(Row &row, const Sample &sample);
// The prompt-processing figures of one request: prompt tokens, TTFT, prefill tok/s.
void add_prefill_metrics(Row &row, const Sample &sample);

// When a request's first token arrived.
Clock::time_point first_token_time(const Sample &sample);

// A generating stream's updates against a window of time [from, to]: another request's prompt processing.
struct StreamInWindow {
  double longest_gap_ms = 0;  // between consecutive updates that overlap the window
  int updates_before = 0, updates_inside = 0;
  double before_s = 0;  // the stream's first update -> the window's start
  bool outlasted = false;  // the stream was still generating at the window's end
};
StreamInWindow split_at(const Sample &stream, Clock::time_point from, Clock::time_point to);

// What the cards draw from its construction until add_to: the row's power and, over the tokens the endpoint handled
// meanwhile, tokens per joule. Adds nothing when the run does not sample power.
class PowerSpan {
 public:
  explicit PowerSpan(const Session &session);
  void add_to(Row &row, int64_t tokens) const;

 private:
  PowerMeter *meter_;
  double from_s_ = 0;
};
// Of the asked depths, those a request of `reply_tokens` fits the endpoint's context with; the rest are named in
// `skipped` (a note for the report).
std::vector<int64_t> depths_that_fit(const Session &session, std::span<const int64_t> depths, int reply_tokens, std::string &skipped);

// Whether `text` holds `needle, ignoring the case of ASCII letters.
bool contains_text(std::string_view text, std::string_view needle);
// The start of a reply on one line, at most `chars` characters: for a note that shows what came back.
std::string head_of(std::string_view text, size_t chars);
// A request's fate under a request-level refusal: an endpoint without a feature answers it with a client error
// (4xx), which makes a scenario skipped rather than failed.
inline bool refused_by_endpoint(const Sample &sample) { return !sample.ok && sample.http_status >= 400 && sample.http_status < 500; }

// Seconds since `from.
double seconds_since(Clock::time_point from);
// The gap to the next event of a Poisson process of `per_second` events per second: exponentially distributed, drawn
// by the inverse of the distribution so that a seed gives the same gaps with every standard library.
double exponential_gap_s(double per_second, std::mt19937_64 &rng);

// The decode rate of a mix of kinds of output. `weights` are shares of generated tokens per kind, `tps` each kind's
// rate. By default (harmonic) the mix's rate is the weighted harmonic mean over the kinds that have a weight and a rate
// - each kind's time per token, weighted, inverted: the rate at which that mix of tokens comes out (an arithmetic mean
// of the rates reads high). `arithmetic` gives that weighted mean of the rates instead, the public mix's figure. `covered`
// is the weight those kinds hold: 1 when every weighted kind ran; {0, 0} when none did.
struct MixRate {
  double tps = 0, covered = 0;
};
MixRate mix_rate(std::span<const std::pair<std::string, double>> weights, const std::map<std::string, double, std::less<>> &tps,
                 MixAverage average = MixAverage::harmonic);

}  // namespace xenithium:bench
