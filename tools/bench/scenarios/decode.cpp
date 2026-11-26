// Scenario "decode": one request at a time over the short prompts of every category, sampled as chat clients sample
// (the run's temperature / top-p / top-k, a seed per request). Decode speed depends on the kind of text - code drafts
// well, prose does not - so it is reported per category and as figures for mixes of the categories (weight_mixes): the
// mix the run asks for heads the decode figure, and the benchmark and agentic mixes are always reported beside it
// (headline_mixes), so the report can switch between them. The benchmark mix is the weighted mean of the categories'
// rates, as the usual public benchmarks quote it; the agentic mix the rate at which a coding agent's mix of tokens comes
// out (mix_rate, a weighted harmonic mean). The prompts are the run's (Session::prompts): per category its open tasks and its
// tasks on given input, every reply ending where the model ends it.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "scenario.hpp"
#include "stats.hpp"

namespace xenithium::bench {
namespace {

constexpr ParamSpec kParams[] = {
    {"categories", "", "categories to run, comma separated; empty = all"},
    {"passes", "2", "times every prompt is sent (seven prompts per category: 14 samples each)"},
    {"warmup", "3", "requests sent first and not counted (the server's first-use costs)"},
    {"weights", "benchmark", "the headline's mix of categories: benchmark (the usual public benchmarks' code-led shares, a plain weighted mean), agentic (a coding agent's measured output, the rate it comes out at), even, or category:share,... (shares of generated tokens); benchmark and agentic are reported either way"},
};

// A mix by its name, or "category:share..." (normalised to sum 1). Throws std:invalid_argument naming the problem.
struct Mix {
  std::string name;  // the mix row's label: "benchmark mix", "agentic mix", "even mix", "custom mix"
  std::vector<std::pair<std::string, double>> weights;
  MixAverage average = MixAverage::harmonic;
};

Mix named_mix(const WeightMix &mix) {
  Mix out{std::string(mix.id) + " mix", {}, mix.average};
  for (const CategoryWeight &w : mix.weights) out.weights.emplace_back(std::string(w.category), w.weight);
  return out;
}

Mix mix_of(const Params &params) {
  const std::string spec = params.text("weights");
  for (const WeightMix &mix : weight_mixes())
    if (mix.id == spec) return named_mix(mix);
  Mix out{"custom mix", {}};
  double sum = 0;
  for (const std::string &item : params.list("weights")) {
    const size_t colon = item.find(':');
    const std::string category = item.substr(0, colon);
    const bool known = std::ranges::any_of(category_weights(), [&](const CategoryWeight &w) { return w.category == category; });
    char *end = nullptr;
    const double share = colon == std::string::npos ? -1 : std::strtod(item.c_str() + colon + 1, &end);
    if (!known || colon == std::string::npos || end == item.c_str() + colon + 1 || *end != '\0' || !std::isfinite(share) || share < 0)
      throw std::invalid_argument("decode.weights: \"" + item + "\" is not category:share with a known category and a share >= 0 (or a mix: benchmark, agentic, even)");
    out.weights.emplace_back(category, share);
    sum += share;
  }
  if (!(sum > 0)) throw std::invalid_argument("decode.weights: the shares sum to 0");
  for (auto &w : out.weights) w.second /= sum;
  return out;
}

class Decode final : public Scenario {
 public:
  std::string_view id() const override { return "decode"; }
  std::string_view title() const override { return "Single-stream decode by category (sampled)"; }
  std::string_view about() const override {
    return "One request at a time over short prompts of eight kinds of output (chat, code, reasoning, maths, prose, JSON, edits, summaries), each kind both written from scratch and "
           "made from given input, replies ending where the model ends them: decode tok/s, TTFT and tokens per streamed update per category, and the rate of a mix of them: the "
           "usual public benchmarks' code-led weighting by default, and a coding agent's measured output beside it.";
  }
  std::span<const ParamSpec> params() const override { return kParams; }
  void check(const Params &params) const override { (void)mix_of(params); }

  ScenarioResult run(Session &session, const Params &params) const override {
    ScenarioResult result;
    const Mix mix = mix_of(params);
    const std::vector<std::string> only = params.list("categories");
    const auto wanted = [&](std::string_view category) { return only.empty() || std::ranges::find(only, category) != only.end(); };

    const auto prompts = session.prompts();
    for (int64_t i = 0; i < params.integer("warmup"); ++i) (void)session.client().chat(session.sampled(prompts.front()));

    std::map<std::string, Row, std::less<>> by_category;
    const PowerSpan power(session);  // over every pass: one stream's power and tokens per joule
    for (int64_t pass = 0; pass < params.integer("passes"); ++pass) {
      for (const ShortPrompt &prompt : prompts) {
        if (!wanted(prompt.category)) continue;
        const Sample sample = session.client().chat(session.sampled(prompt));
        if (!tally(result, sample)) continue;
        Row &row = by_category[std::string(prompt.category)];
        if (row.labels.empty()) row.labels = {{"category", std::string(prompt.category)}};
        add_decode_metrics(row, sample);
      }
      session.progress("decode: pass " + std::to_string(pass + 1) + " of " + params.text("passes") + " done");
    }

    // rows in the usual order of the categories, then the mix's rate over the weighted categories that ran
    std::map<std::string, double, std::less<>> tps;
    for (const CategoryWeight &usual : category_weights())
      if (const auto it = by_category.find(usual.category); it != by_category.end()) {
        tps[it->first] = summarize(it->second.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples).median;
        result.rows.push_back(std::move(it->second));
      }
    // the asked mix first (it heads the decode figure and carries the power), then benchmark and agentic where not asked
    std::vector<Mix> mixes = {mix};
    for (const WeightMix &m : headline_mixes())
      if (std::string(m.id) + " mix" != mix.name) mixes.push_back(named_mix(m));
    for (const Mix &m : mixes)
      if (const MixRate rate = mix_rate(m.weights, tps, m.average); rate.tps > 0) {
        Row all = Row::with_labels({{"category", m.name}});
        all.metric(std::string(metric::kDecode), "tok/s", Better::higher).samples.push_back(rate.tps);
        if (&m == &mixes.front()) power.add_to(all, result.completion_tokens);
        if (rate.covered < 1 - 1e-9) all.notes.push_back("the " + m.name + " over the categories that ran: " + format_number(std::round(rate.covered * 1000) / 10) + " % of its weight");
        all.notes.push_back(m.average == MixAverage::arithmetic ? "the " + m.name + " is the weighted mean of the categories' median rates"
                                                                : "the " + m.name + " is the rate its mix of tokens comes out at (a weighted harmonic mean)");
        result.rows.push_back(std::move(all));
      }
    result.charts = {
        {ChartKind::bars, "Decode rate by kind of output", "category", {std::string(metric::kDecode)}, false},
        {ChartKind::bars, "Tokens per streamed update", "category", {std::string(metric::kTokensPerUpdate)}, false},
    };
    return result;
  }
};

}  // namespace

std::unique_ptr<Scenario> make_decode() { return std::make_unique<Decode>(); }

}  // namespace xenithium:bench
