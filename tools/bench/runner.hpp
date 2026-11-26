// xenithium-bench: runs a list of scenarios against a session and collects the Run.
#pragma once

#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "report.hpp"
#include "scenario.hpp"

namespace xenithium::bench {

struct RunPlan {
  std::string suite;  // the suite's id, or "custom"
  std::vector<SuiteStep> steps;  // scenarios in order, each with the parameters to change
};

// The plan of a suite; of a comma-separated scenario list ("prefill,needle") when `scenarios` is not empty. `overrides`
// are "scenario.key=value" items laid over the plan's parameters. Throws std:invalid_argument naming what is unknown.
RunPlan make_plan(const std::string &suite, const std::string &scenarios, const std::vector<std::string> &overrides);

// Runs every step of the plan. `on_scenario` is called with each finished scenario (to print it as it lands). A
// scenario that throws is recorded as failed and the run goes on. `info` is completed with the run's wall time.
Run execute(const RunPlan &plan, Session &session, RunInfo info, const std::function<void(const ScenarioResult &)> &on_scenario);

}  // namespace xenithium:bench
