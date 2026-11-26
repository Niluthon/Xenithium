#include "runner.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace xenithium::bench {
namespace {

const Scenario *find(const std::vector<std::unique_ptr<Scenario>> &all, std::string_view id) {
  const auto it = std::ranges::find_if(all, [&](const auto &s) { return s->id() == id; });
  return it == all.end() ? nullptr : it->get();
}

std::vector<std::string> split_commas(const std::string &text) {
  std::vector<std::string> items;
  for (size_t at = 0; at < text.size();) {
    const size_t comma = std::min(text.find(',', at), text.size());
    if (comma > at) items.push_back(text.substr(at, comma - at));
    at = comma + 1;
  }
  return items;
}

// "scenario.key=value", split. Throws std:invalid_argument for another form.
struct Setting {
  std::string scenario, key, value;
};
Setting setting_of(std::string_view item) {
  const size_t dot = item.find('.'), eq = item.find('=');
  if (dot == std::string::npos || eq == std::string::npos || dot > eq) throw std::invalid_argument("--set takes scenario.key=value, got \"" + std::string(item) + "\"");
  return {std::string(item.substr(0, dot)), std::string(item.substr(dot + 1, eq - dot - 1)), std::string(item.substr(eq + 1))};
}

// "scenario.key=value" onto the step of that scenario.
void apply_override(RunPlan &plan, const std::string &item) {
  const Setting s = setting_of(item);
  bool applied = false;
  for (SuiteStep &step : plan.steps)
    if (step.scenario == s.scenario) step.params.set(s.key, s.value), applied = true;
  if (!applied) throw std::invalid_argument("--set " + item + ": the run has no scenario \"" + s.scenario + "\"");
}

// Seconds the cards are watched before the first scenario for the run's idle power.
constexpr double kIdleWindowS = 1.0;
// Points of the power-over-time trace kept in the run file: enough for a chart as wide as a page.
constexpr size_t kTracePoints = 720;

// What the cards draw before anything is asked of them.
double watts_when_idle(PowerMeter &meter) {
  const double from_s = meter.mark();
  std::this_thread::sleep_for(std::chrono::duration<double>(kIdleWindowS));
  return meter.use(from_s, meter.mark()).watts();
}

ScenarioResult run_step(const Scenario &scenario, const SuiteStep &step, Session &session) {
  ScenarioResult result;
  const Clock::time_point started = Clock::now();
  const double power_from_s = session.power() ? session.power()->mark() : 0;
  const auto before = session.counters();
  try {
    result = scenario.run(session, Params::merged(scenario.params(), step.params));
  } catch (const std::exception &e) {
    ++result.errors;
    if (result.first_error.empty()) result.first_error = e.what();
  }
  result.id = std::string(scenario.id());
  result.title = std::string(scenario.title());
  result.about = std::string(scenario.about());
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  if (PowerMeter *meter = session.power())
    if (const PowerUse use = meter->use(power_from_s, meter->mark()); use.joules > 0) result.power = use;
  if (const auto after = session.counters(); before && after)
    if (std::vector<std::string> lines = describe_delta(*before, *after); !lines.empty()) {
      if (result.rows.empty()) result.rows.emplace_back();
      for (std::string &line : lines) result.rows.back().notes.push_back(std::move(line));
    }
  return result;
}

}  // namespace

RunPlan make_plan(const std::string &suite, const std::string &scenarios, const std::vector<std::string> &overrides) {
  RunPlan plan;
  const auto all = all_scenarios();
  if (!scenarios.empty()) {
    plan.suite = "custom";
    for (const std::string &id : split_commas(scenarios)) plan.steps.push_back({id, {}});
  } else {
    const auto it = std::ranges::find_if(suites(), [&](const Suite &s) { return s.id == suite; });
    if (it == suites().end()) throw std::invalid_argument("no suite \"" + suite + "\" (see `xenithium-bench list`)");
    plan.suite = it->id;
    plan.steps = it->steps;
  }
  for (const std::string &item : overrides) apply_override(plan, item);
  for (const SuiteStep &step : plan.steps) {
    const Scenario *scenario = find(all, step.scenario);
    if (!scenario) throw std::invalid_argument("no scenario \"" + step.scenario + "\" (see `xenithium-bench list`)");
    scenario->check(Params::merged(scenario->params(), step.params));  // merged throws on a parameter the scenario does not have
  }
  return plan;
}

Run execute(const RunPlan &plan, Session &session, RunInfo info, const std::function<void(const ScenarioResult &)> &on_scenario) {
  Run run{std::move(info), {}};
  PowerMeter *meter = session.power();
  const double idle_watts = meter ? watts_when_idle(*meter) : 0;
  const double power_origin_s = meter ? meter->mark() : 0;
  const Clock::time_point started = Clock::now();
  const auto all = all_scenarios();
  for (const SuiteStep &step : plan.steps) {
    const Scenario *scenario = find(all, step.scenario);
    if (!scenario) continue;  // make_plan refused unknown ids
    const double started_s = std::chrono::duration<double>(Clock::now() - started).count();
    run.scenarios.push_back(run_step(*scenario, step, session));
    run.scenarios.back().started_s = started_s;
    if (on_scenario) on_scenario(run.scenarios.back());
  }
  run.info.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  if (meter) run.info.power = RunPower{meter->cards(), idle_watts, meter->trace(power_origin_s, kTracePoints)};
  return run;
}

}  // namespace xenithium:bench
