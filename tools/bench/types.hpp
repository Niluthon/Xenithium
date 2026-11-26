// xenithium-bench: the values a scenario reports.
// A scenario returns rows; a row is one measured case (its labels say which: depth 32768, level 8, category code)
// with named metrics, each holding every sample taken. Reports, the run file and `compare` are written against these
// types only, so a new scenario needs no change outside its own file and the registry.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xenithium::bench {

// Which way a metric improves: `compare` colours a change by it; `none` is context (prompt tokens, a count).
enum class Better { higher, lower, none };

// The unit of a yes / no metric: its samples are 1 or 0 and a table shows how many were yes ("2/3").
inline constexpr std::string_view kFlagUnit = "yes/no";

struct Metric {
  std::string name;  // "decode", "TTFT"
  std::string unit;  // "tok/s", "ms", "tokens"; "" for a bare number; kFlagUnit for a yes / no
  Better better = Better::none;
  std::vector<double> samples;  // one per measured request (or per round for whole-round figures)
};

using Label = std::pair<std::string, std::string>;  // ("depth", "32768"), in display order

struct Row {
  std::vector<Label> labels;
  std::vector<Metric> metrics;
  // Rows of one `table` are shown as a table of their own under this caption. Empty = the scenario's main table.
  // For a scenario that measures two things of different shape (a figure per load level, and a matrix per category).
  std::string table;
  std::vector<std::string> notes;  // free text shown under the table (a failed request's error, a wrapped corpus)

  // A yes / no metric (found, identical, valid): more yes is better.
  Metric &flag(std::string name) { return metric(std::move(name), std::string(kFlagUnit), Better::higher); }

  static Row with_labels(std::vector<Label> labels) {
    Row row;
    row.labels = std::move(labels);
    return row;
  }
  // A row without a case: a remark about the whole scenario (depths that were skipped).
  static Row note_only(std::string note) {
    Row row;
    row.notes.push_back(std::move(note));
    return row;
  }

  Metric &metric(std::string name, std::string unit, Better better) {
    for (Metric &m : metrics)
      if (m.name == name) return m;
    metrics.push_back(Metric{std::move(name), std::move(unit), better, {}});
    return metrics.back();
  }
};

// How a scenario wants its rows drawn in the HTML report. The report has one renderer per kind and knows no
// scenario: a new scenario gets its charts by declaring them.
enum class ChartKind {
  bars,  // one bar per row and metric: the metric's median, labelled with its value
  line,  // the metrics' medians over the rows' x label read as a number (a depth, a stream count)
  range,  // per row the metric's min, p10, median, p90 and max: where a latency's samples lie
  dots,  // one dot per sample of a yes / no metric, filled when yes
};
struct ChartSpec {
  ChartKind kind = ChartKind::bars;
  std::string title;  // "Throughput by streams"
  std::string x_label;  // the row label along the axis ("streams"); rows without it are left out
  std::vector<std::string> metrics;  // the metrics drawn, each a series (they share one axis, so one unit)
  bool log_x = false;  // line charts: x doubles from tick to tick (depths 2K.. 250K)
};

// What the GPUs drew over a span of the run (power.hpp); present when the run sampled power.
struct PowerUse {
  double seconds = 0;
  double joules = 0;  // all cards together
  std::vector<double> watts_per_card;  // mean over the span
  double mean_mhz = 0, min_mhz = 0;  // the GPUs' actual clock over the span's samples (0 = not readable)

  double watts() const { return seconds > 0 ? joules / seconds : 0; }
};

struct PowerCard {
  std::string name;  // "card0"
  double cap_watts = 0;  // the driver's sustained power limit (power1_cap); 0 = not readable
  double max_mhz = 0;  // the clock's hardware maximum (xe: rp0_freq); 0 = not readable
  bool reads_temp = false;  // its GPU package temperature is readable (hwmon "pkg")
  bool reads_throttle = false;  // its throttle reasons are readable (xe: freq0/throttle/reasons)
};
// A point of the power-over-time trace kept in the run file.
struct PowerPoint {
  double at_s = 0;  // since the run's first scenario began
  std::vector<double> watts;  // per card, mean since the previous point
  double mhz = 0;  // actual GPU clock, mean over the cards
  double temp_c = 0;  // the hottest card's GPU package since the previous point (0 = not readable)
  std::string throttle = {};  // what limited a card's clock since the previous point ("pl1 thermal"); empty = nothing (or not readable)
};
// The run's power record.
struct RunPower {
  std::vector<PowerCard> cards;
  double idle_watts = 0;  // all cards, over a second before the first scenario
  std::vector<PowerPoint> trace;
};

// A reply that failed a check with one right answer (a wrong answer). Kept apart from the notes: a report shows each
// one where the eye lands (a flagged table in its scenario, a link from the checks strip), not as a line of fine print.
struct Miss {
  std::string item;  // the check's id ("format-reverse")
  std::string group;  // the row it counts in ("format")
  std::string asked;  // the question, without the passage or document it was asked on
  std::string expected;
  std::string got;  // the answer given; empty when there was none (then `why`)
  std::string why;  // why there is no answer ("no ANSWER line"); empty when there is one
};

struct ScenarioResult {
  std::string id;  // the scenario's id ("prefill")
  std::string title;  // one line for the report
  std::string about;  // what it measures, a sentence or two
  std::vector<Row> rows;
  std::vector<ChartSpec> charts;
  int requests = 0;  // requests sent
  int errors = 0;  // of those, failed
  std::string first_error;
  int64_t completion_tokens = 0;  // generated by the answered requests
  double started_s = 0;  // since the run's first scenario began
  double seconds = 0;  // wall time of the scenario
  std::optional<PowerUse> power;  // over the scenario
  std::vector<Miss> misses;  // the replies its checks found wrong, in the order asked
};

}  // namespace xenithium:bench
