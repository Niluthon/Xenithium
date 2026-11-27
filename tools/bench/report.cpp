#include "report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>

#include "scenario.hpp"
#include "stats.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::bench {
namespace {

namespace json = xenithium::json;

// A change under this is not reported as one (the engine's keep bar for a lever is 0.5 %).
constexpr double kNoticeablePct = 0.5;
// From this many samples on, a "lower is better" metric (a TTFT, a gap) also shows its worst sample.
constexpr int kShowWorstFrom = 4;
constexpr std::string_view kIndent = "  ";

constexpr std::pair<ChartKind, std::string_view> kChartKindNames[] = {
    {ChartKind::bars, "bars"}, {ChartKind::line, "line"}, {ChartKind::range, "range"}, {ChartKind::dots, "dots"}};
std::string_view chart_kind_name(ChartKind kind) {
  for (const auto &[k, name] : kChartKindNames)
    if (k == kind) return name;
  return "bars";
}
ChartKind chart_kind_of(std::string_view name) {
  for (const auto &[k, n] : kChartKindNames)
    if (n == name) return k;
  return ChartKind::bars;
}

std::string_view better_name(Better b) { return b == Better::higher ? "higher" : b == Better::lower ? "lower" : "none"; }
Better better_of(std::string_view name) { return name == "higher" ? Better::higher : name == "lower" ? Better::lower : Better::none; }

std::string_view thinking_or_default(const std::string &t) { return t.empty() ? "default" : std::string_view(t); }

// What a metric shows in a table cell.
std::string cell_of(const Metric &m) {
  const Summary s = summarize(m.samples);
  if (s.n == 0) return "-";
  if (m.unit == kFlagUnit) {  // how many of the samples were yes
    const auto yes = std::ranges::count_if(m.samples, [](double v) { return v > 0.5; });
    return std::to_string(yes) + "/" + std::to_string(s.n);
  }
  if (m.better == Better::lower && s.n >= kShowWorstFrom) return format_number(s.median) + " / " + format_number(s.max);
  return format_number(s.median);
}

std::string column_title(const Metric &m, bool with_worst) {
  std::string title = m.name;
  if (!m.unit.empty() && m.unit != kFlagUnit) title += " " + m.unit;
  if (with_worst) title += " (median / max)";
  return title;
}

std::string padded(std::string text, size_t width) {
  if (text.size() < width) text.append(width - text.size(), ' ');
  return text;
}

// A table: a header row and text cells, columns as wide as their widest cell.
std::string table(const std::vector<std::vector<std::string>> &rows) {
  std::vector<size_t> widths;
  for (const auto &row : rows)
    for (size_t c = 0; c < row.size(); ++c) {
      if (widths.size() <= c) widths.push_back(0);
      widths[c] = std::max(widths[c], row[c].size());
    }
  std::string out;
  for (const auto &row : rows) {
    std::string line(kIndent);
    for (size_t c = 0; c < row.size(); ++c) line += padded(row[c], widths[c]) + (c + 1 < row.size() ? "  " : "");
    while (!line.empty() && line.back() == ' ') line.pop_back();
    out += line + "\n";
  }
  return out;
}

// The label names and metric columns of a table's rows, in first-seen order (rows may lack some).
struct Columns {
  std::vector<std::string> labels;
  std::vector<const Metric *> metrics;  // one representative per name
  std::vector<bool> with_worst;
};
Columns columns_of(std::span<const Row *const> rows) {
  Columns cols;
  for (const Row *row : rows) {
    for (const Label &label : row->labels)
      if (std::ranges::find(cols.labels, label.first) == cols.labels.end()) cols.labels.push_back(label.first);
    for (const Metric &m : row->metrics) {
      const auto it = std::ranges::find_if(cols.metrics, [&](const Metric *have) { return have->name == m.name; });
      const bool worst = m.better == Better::lower && static_cast<int>(m.samples.size()) >= kShowWorstFrom;
      if (it == cols.metrics.end()) cols.metrics.push_back(&m), cols.with_worst.push_back(worst);
      else if (worst) cols.with_worst[static_cast<size_t>(it - cols.metrics.begin())] = true;
    }
  }
  return cols;
}

const std::string *label_value(const Row &row, std::string_view name) {
  for (const Label &label : row.labels)
    if (label.first == name) return &label.second;
  return nullptr;
}
const Metric *metric_named(const Row &row, std::string_view name) {
  for (const Metric &m : row.metrics)
    if (m.name == name) return &m;
  return nullptr;
}
const ScenarioResult *scenario_named(const Run &run, std::string_view id) {
  for (const ScenarioResult &s : run.scenarios)
    if (s.id == id) return &s;
  return nullptr;
}
double median_of(const Row &row, std::string_view metric_name) {
  const Metric *m = metric_named(row, metric_name);
  return m ? summarize(m->samples).median : 0;
}

json::Value labels_json(const std::vector<Label> &labels) {
  json::Value obj = json::Value::object();
  for (const Label &label : labels) obj.set(label.first, json::Value::string(label.second));
  return obj;
}

json::Value metric_json(const Metric &m) {
  const Summary s = summarize(m.samples);
  json::Value obj = json::Value::object();
  obj.set("name", json::Value::string(m.name));
  obj.set("unit", json::Value::string(m.unit));
  obj.set("better", json::Value::string(better_name(m.better)));
  obj.set("n", json::Value::integer(s.n));
  obj.set("median", json::Value::real(s.median));
  obj.set("mean", json::Value::real(s.mean));
  obj.set("min", json::Value::real(s.min));
  obj.set("max", json::Value::real(s.max));
  obj.set("p10", json::Value::real(s.p10));
  obj.set("p90", json::Value::real(s.p90));
  json::Value samples = json::Value::array();
  for (const double v : m.samples) samples.push(json::Value::real(v));
  obj.set("samples", std::move(samples));
  return obj;
}

json::Value reals_json(std::span<const double> values) {
  json::Value arr = json::Value::array();
  for (const double v : values) arr.push(json::Value::real(v));
  return arr;
}

json::Value power_use_json(const PowerUse &use) {
  json::Value obj = json::Value::object();
  obj.set("seconds", json::Value::real(use.seconds));
  obj.set("joules", json::Value::real(use.joules));
  obj.set("watts", json::Value::real(use.watts()));
  obj.set("watts_per_card", reals_json(use.watts_per_card));
  obj.set("mean_mhz", json::Value::real(use.mean_mhz));
  obj.set("min_mhz", json::Value::real(use.min_mhz));
  return obj;
}

json::Value chart_json(const ChartSpec &chart) {
  json::Value obj = json::Value::object();
  obj.set("kind", json::Value::string(chart_kind_name(chart.kind)));
  obj.set("title", json::Value::string(chart.title));
  obj.set("x_label", json::Value::string(chart.x_label));
  json::Value metrics = json::Value::array();
  for (const std::string &name : chart.metrics) metrics.push(json::Value::string(name));
  obj.set("metrics", std::move(metrics));
  obj.set("log_x", json::Value::boolean(chart.log_x));
  return obj;
}

json::Value run_power_json(const RunPower &power) {
  json::Value obj = json::Value::object();
  json::Value cards = json::Value::array();
  for (const PowerCard &card : power.cards) {
    json::Value c = json::Value::object();
    c.set("name", json::Value::string(card.name));
    c.set("cap_watts", json::Value::real(card.cap_watts));
    if (card.max_mhz > 0) c.set("max_mhz", json::Value::real(card.max_mhz));
    if (card.reads_temp) c.set("reads_temp", json::Value::boolean(true));
    if (card.reads_throttle) c.set("reads_throttle", json::Value::boolean(true));
    cards.push(std::move(c));
  }
  obj.set("cards", std::move(cards));
  obj.set("idle_watts", json::Value::real(power.idle_watts));
  // one row per point: [seconds, clock MHz, watts of card 0, watts of card 1...]
  json::Value trace = json::Value::array();
  for (const PowerPoint &point : power.trace) {
    json::Value row = json::Value::array();
    row.push(json::Value::real(point.at_s));
    row.push(json::Value::real(point.mhz));
    for (const double watts : point.watts) row.push(json::Value::real(watts));
    trace.push(std::move(row));
  }
  obj.set("trace", std::move(trace));
  // per trace row: the hottest GPU package and the throttle reasons (apart from the rows, which older readers parse)
  json::Value temps = json::Value::array(), throttles = json::Value::array();
  for (const PowerPoint &point : power.trace) temps.push(json::Value::real(point.temp_c)), throttles.push(json::Value::string(point.throttle));
  obj.set("trace_temp_c", std::move(temps));
  obj.set("trace_throttle", std::move(throttles));
  return obj;
}

json::Value miss_json(const Miss &m) {
  json::Value obj = json::Value::object();
  obj.set("item", json::Value::string(m.item));
  obj.set("group", json::Value::string(m.group));
  obj.set("asked", json::Value::string(m.asked));
  obj.set("expected", json::Value::string(m.expected));
  obj.set("got", json::Value::string(m.got));
  if (!m.why.empty()) obj.set("why", json::Value::string(m.why));
  return obj;
}

json::Value misses_json(const std::vector<Miss> &misses) {
  json::Value arr = json::Value::array();
  for (const Miss &m : misses) arr.push(miss_json(m));
  return arr;
}

json::Value scenario_json(const ScenarioResult &s) {
  json::Value obj = json::Value::object();
  obj.set("id", json::Value::string(s.id));
  obj.set("title", json::Value::string(s.title));
  obj.set("about", json::Value::string(s.about));
  obj.set("started_s", json::Value::real(s.started_s));
  obj.set("seconds", json::Value::real(s.seconds));
  obj.set("requests", json::Value::integer(s.requests));
  obj.set("errors", json::Value::integer(s.errors));
  obj.set("first_error", json::Value::string(s.first_error));
  obj.set("completion_tokens", json::Value::integer(s.completion_tokens));
  if (s.power) obj.set("power", power_use_json(*s.power));
  json::Value charts = json::Value::array();
  for (const ChartSpec &chart : s.charts) charts.push(chart_json(chart));
  obj.set("charts", std::move(charts));
  json::Value rows = json::Value::array();
  for (const Row &row : s.rows) {
    json::Value r = json::Value::object();
    r.set("labels", labels_json(row.labels));
    if (!row.table.empty()) r.set("table", json::Value::string(row.table));
    json::Value metrics = json::Value::array();
    for (const Metric &m : row.metrics) metrics.push(metric_json(m));
    r.set("metrics", std::move(metrics));
    json::Value notes = json::Value::array();
    for (const std::string &note : row.notes) notes.push(json::Value::string(note));
    r.set("notes", std::move(notes));
    rows.push(std::move(r));
  }
  obj.set("rows", std::move(rows));
  if (!s.misses.empty()) obj.set("misses", misses_json(s.misses));
  return obj;
}

json::Value depths_json(const std::vector<Headline::Depth> &depths, std::string_view tps_key) {
  json::Value arr = json::Value::array();
  for (const Headline::Depth &d : depths) {
    json::Value obj = json::Value::object();
    obj.set("depth", json::Value::string(d.depth));
    obj.set(tps_key, json::Value::real(d.tps));
    obj.set("ttft_ms", json::Value::real(d.ttft_ms));
    arr.push(std::move(obj));
  }
  return arr;
}

json::Value pairs_json(const std::vector<std::pair<std::string, double>> &pairs, std::string_view key, std::string_view value) {
  json::Value arr = json::Value::array();
  for (const auto &[k, v] : pairs) {
    json::Value obj = json::Value::object();
    obj.set(key, json::Value::string(k));
    obj.set(value, json::Value::real(v));
    arr.push(std::move(obj));
  }
  return arr;
}

json::Value headline_json(const Headline &h) {
  json::Value obj = json::Value::object();
  obj.set("decode_tps", json::Value::real(h.decode_tps));
  if (!h.decode_mix.empty()) obj.set("decode_mix", json::Value::string(h.decode_mix));
  obj.set("decode_by_category", pairs_json(h.decode_by_category, "category", "decode_tps"));
  obj.set("prefill", depths_json(h.prefill, "prefill_tps"));
  obj.set("deep_decode", depths_json(h.deep_decode, "decode_tps"));
  obj.set("concurrency", pairs_json(h.concurrency, "streams", "total_tps"));
  obj.set("peak_total_tps", json::Value::real(h.peak_total_tps));
  obj.set("peak_streams", json::Value::string(h.peak_streams));
  obj.set("best_tokens_per_joule", json::Value::real(h.best_tokens_per_joule));
  if (h.quality_pct >= 0) obj.set("quality_pct", json::Value::real(h.quality_pct));
  if (!h.quality_misses.empty()) obj.set("quality_misses", misses_json(h.quality_misses));
  return obj;
}

std::string text_at(const json::Value &obj, std::string_view key) {
  const std::string *s = obj.get_string(key);
  return s ? *s : std::string();
}
double number_at(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  return v && v->is_number() ? v->num : 0;
}
json::Value devices_json(const std::vector<Device> &devices) {
  json::Value arr = json::Value::array();
  for (const Device &d : devices) {
    json::Value obj = json::Value::object();
    obj.set("name", json::Value::string(d.name));
    if (d.memory_bytes) obj.set("memory_bytes", json::Value::integer(static_cast<int64_t>(d.memory_bytes)));
    if (d.memory_gb) obj.set("memory_gb", json::Value::integer(d.memory_gb));
    if (!d.pci.empty()) obj.set("pci", json::Value::string(d.pci));
    if (!d.driver.empty()) obj.set("driver", json::Value::string(d.driver));
    arr.push(std::move(obj));
  }
  return arr;
}
std::vector<Device> devices_from(const json::Value *arr) {
  std::vector<Device> out;
  if (!arr) return out;
  for (const json::Value &d : arr->arr)
    out.push_back({.name = text_at(d, "name"), .memory_bytes = static_cast<uint64_t>(number_at(d, "memory_bytes")), .memory_gb = static_cast<int>(number_at(d, "memory_gb")),
                   .pci = text_at(d, "pci"), .driver = text_at(d, "driver")});
  return out;
}
json::Value machine_json(const util::Machine &m) {
  json::Value obj = json::Value::object();
  obj.set("os", json::Value::string(m.os));
  obj.set("kernel", json::Value::string(m.kernel));
  obj.set("cpu", json::Value::string(m.cpu));
  obj.set("cpu_threads", json::Value::integer(m.cpu_threads));
  obj.set("memory_gib", json::Value::integer(m.memory_gib));
  return obj;
}
json::Value strings_json(const std::vector<std::string> &strings) {
  json::Value arr = json::Value::array();
  for (const std::string &s : strings) arr.push(json::Value::string(s));
  return arr;
}
std::vector<std::string> strings_from(const json::Value *arr) {
  std::vector<std::string> out;
  if (arr)
    for (const json::Value &v : arr->arr)
      if (v.is_string()) out.push_back(v.str);
  return out;
}

bool flag_at(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  return v && v->is_bool() && v->b;
}

std::vector<Label> labels_from(const json::Value *obj) {
  std::vector<Label> labels;
  if (obj)
    for (const auto &member : obj->obj) labels.emplace_back(member.first, member.second.str);
  return labels;
}

std::vector<double> reals_from(const json::Value *arr) {
  std::vector<double> values;
  if (arr)
    for (const json::Value &v : arr->arr) values.push_back(v.num);
  return values;
}

PowerUse power_use_from(const json::Value &obj) {
  return {.seconds = number_at(obj, "seconds"),
          .joules = number_at(obj, "joules"),
          .watts_per_card = reals_from(obj.get_array("watts_per_card")),
          .mean_mhz = number_at(obj, "mean_mhz"),
          .min_mhz = number_at(obj, "min_mhz")};
}

ChartSpec chart_from(const json::Value &obj) {
  ChartSpec chart{chart_kind_of(text_at(obj, "kind")), text_at(obj, "title"), text_at(obj, "x_label"), {}, flag_at(obj, "log_x")};
  if (const json::Value *metrics = obj.get_array("metrics"))
    for (const json::Value &name : metrics->arr) chart.metrics.push_back(name.str);
  return chart;
}

RunPower run_power_from(const json::Value &obj) {
  RunPower power;
  if (const json::Value *cards = obj.get_array("cards"))
    for (const json::Value &card : cards->arr)
      power.cards.push_back({.name = text_at(card, "name"), .cap_watts = number_at(card, "cap_watts"), .max_mhz = number_at(card, "max_mhz"),
                             .reads_temp = flag_at(card, "reads_temp"), .reads_throttle = flag_at(card, "reads_throttle")});
  power.idle_watts = number_at(obj, "idle_watts");
  constexpr size_t kTraceHead = 2;  // seconds and clock come before the cards' watts
  // trace_temp_c and trace_throttle hold one entry per trace row: taken by the row's own index, so a row skipped as
  // malformed does not shift the later ones onto the wrong points
  const json::Value *temps = obj.get_array("trace_temp_c"), *throttles = obj.get_array("trace_throttle");
  if (const json::Value *trace = obj.get_array("trace"))
    for (size_t r = 0; r < trace->arr.size(); ++r) {
      const json::Value &row = trace->arr[r];
      if (row.arr.size() < kTraceHead) continue;
      PowerPoint point{.at_s = row.arr[0].num, .watts = {}, .mhz = row.arr[1].num};
      for (size_t i = kTraceHead; i < row.arr.size(); ++i) point.watts.push_back(row.arr[i].num);
      if (temps && r < temps->arr.size() && temps->arr[r].is_number()) point.temp_c = temps->arr[r].num;
      if (throttles && r < throttles->arr.size() && throttles->arr[r].is_string()) point.throttle = throttles->arr[r].str;
      power.trace.push_back(std::move(point));
    }
  return power;
}

// Runs from before the misses kept a quality miss as a note: `id: expected "X", got "Y"` (Y "(no ANSWER line...)"
// when there was none). Those notes become misses (without the question or the category, which they did not keep).
void misses_from_notes(Row &row, std::vector<Miss> &misses) {
  constexpr std::string_view kExpected = ": expected \"", kGot = "\", got \"";
  std::erase_if(row.notes, [&](const std::string &note) {
    const size_t e = note.find(kExpected), g = note.rfind(kGot);
    if (e == std::string::npos || g == std::string::npos || g < e || note.size() < 2 || note.back() != '"') return false;
    Miss m;
    m.item = note.substr(0, e);
    m.expected = note.substr(e + kExpected.size(), g - e - kExpected.size());
    std::string got = note.substr(g + kGot.size(), note.size() - g - kGot.size() - 1);
    if (got.size() >= 2 && got.front() == '(' && got.back() == ')') m.why = got.substr(1, got.size() - 2);
    else m.got = std::move(got);
    misses.push_back(std::move(m));
    return true;
  });
}

ScenarioResult scenario_from(const json::Value &obj) {
  ScenarioResult s;
  s.id = text_at(obj, "id"), s.title = text_at(obj, "title"), s.about = text_at(obj, "about"), s.first_error = text_at(obj, "first_error");
  s.seconds = number_at(obj, "seconds"), s.started_s = number_at(obj, "started_s");
  s.completion_tokens = static_cast<int64_t>(number_at(obj, "completion_tokens"));
  if (const json::Value *power = obj.get_object("power")) s.power = power_use_from(*power);
  if (const json::Value *charts = obj.get_array("charts"))
    for (const json::Value &chart : charts->arr) s.charts.push_back(chart_from(chart));
  s.requests = static_cast<int>(number_at(obj, "requests")), s.errors = static_cast<int>(number_at(obj, "errors"));
  if (const json::Value *rows = obj.get_array("rows"))
    for (const json::Value &r : rows->arr) {
      Row row;
      row.labels = labels_from(r.get_object("labels"));
      row.table = text_at(r, "table");
      if (const json::Value *metrics = r.get_array("metrics"))
        for (const json::Value &m : metrics->arr) {
          Metric metric{text_at(m, "name"), text_at(m, "unit"), better_of(text_at(m, "better")), {}};
          if (const json::Value *samples = m.get_array("samples"))
            for (const json::Value &v : samples->arr) metric.samples.push_back(v.num);
          row.metrics.push_back(std::move(metric));
        }
      if (const json::Value *notes = r.get_array("notes"))
        for (const json::Value &n : notes->arr) row.notes.push_back(n.str);
      s.rows.push_back(std::move(row));
    }
  if (const json::Value *misses = obj.get_array("misses"))
    for (const json::Value &m : misses->arr)
      s.misses.push_back({text_at(m, "item"), text_at(m, "group"), text_at(m, "asked"), text_at(m, "expected"), text_at(m, "got"), text_at(m, "why")});
  else if (s.id == "quality")
    for (Row &row : s.rows) misses_from_notes(row, s.misses);
  return s;
}

std::string row_key(const Row &row) {
  std::string key;
  for (const Label &label : row.labels) key += label.first + "=" + label.second + ";";
  return key;
}

// The change of `other` against `base` in percent and whether it is worth a mark.
struct Change {
  double pct = 0;
  bool marked = false;
};
Change change_of(const Metric &base, const Metric &other) {
  const Summary a = summarize(base.samples), b = summarize(other.samples);
  Change c;
  if (a.n == 0 || b.n == 0 || a.median == 0) return c;
  c.pct = 100.0 * (b.median / a.median - 1);
  c.marked = std::abs(c.pct) > std::max({kNoticeablePct, spread_pct(a), spread_pct(b)});
  return c;
}

std::string change_text(const Metric &m, const Change &c) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%+.1f %%", c.pct);
  std::string text = buf;
  if (!c.marked || m.better == Better::none) return text;
  const bool improved = (c.pct > 0) == (m.better == Better::higher);
  return text + (improved ? "  better" : "  WORSE");
}

}  // namespace

Headline headline(const Run &run) {
  Headline h;
  if (const ScenarioResult *decode = scenario_named(run, "decode"))
    for (const Row &row : decode->rows) {
      const std::string *category = label_value(row, "category");
      if (!category) continue;
      const double tps = median_of(row, metric::kDecode);
      if (*category == "weighted" || category->ends_with(" mix")) h.decode_tps = tps, h.decode_mix = *category;
      else h.decode_by_category.emplace_back(*category, tps);
    }
  if (const ScenarioResult *prefill = scenario_named(run, "prefill")) {
    const std::string *first_kind = prefill->rows.empty() ? nullptr : label_value(prefill->rows.front(), "kind");
    for (const Row &row : prefill->rows) {
      const std::string *kind = label_value(row, "kind"), *depth = label_value(row, "depth");
      if (depth && kind && first_kind && *kind == *first_kind) h.prefill.push_back({*depth, median_of(row, metric::kPrefill), median_of(row, metric::kTtft)});
    }
  }
  if (const ScenarioResult *deep = scenario_named(run, "deep-decode"))
    for (const Row &row : deep->rows)
      if (const std::string *depth = label_value(row, "depth")) h.deep_decode.push_back({*depth, median_of(row, metric::kDecode), median_of(row, "hit TTFT")});
  if (const ScenarioResult *conc = scenario_named(run, "concurrency"))
    for (const Row &row : conc->rows) {
      const std::string *streams = label_value(row, "streams");
      if (!streams) continue;
      const double total = median_of(row, metric::kTotal);
      h.concurrency.emplace_back(*streams, total);
      if (total > h.peak_total_tps) h.peak_total_tps = total, h.peak_streams = *streams;
      h.best_tokens_per_joule = std::max(h.best_tokens_per_joule, median_of(row, metric::kEfficiency));
    }
  if (const ScenarioResult *quality = scenario_named(run, "quality")) {
    for (const Row &row : quality->rows)
      if (const Metric *score = metric_named(row, metric::kQualityScore); score && !score->samples.empty()) h.quality_pct = score->samples.front();
    h.quality_misses = quality->misses;
  }
  return h;
}

std::string misses_heading(const ScenarioResult &scenario) {
  size_t asked = 0;
  for (const Row &row : scenario.rows)
    for (const Metric &m : row.metrics)
      if (m.unit == kFlagUnit) asked += m.samples.size();
  const size_t n = scenario.misses.size();
  return std::to_string(n) + (n == 1 ? " wrong answer" : " wrong answers") + (asked >= n && asked > 0 ? " of " + std::to_string(asked) + " asked" : std::string());
}

std::string given_text(const Miss &miss) { return miss.why.empty() ? "\"" + miss.got + "\"" : "(" + miss.why + ")"; }

std::vector<std::string> unique_notes(const ScenarioResult &scenario) {
  std::vector<std::string> notes;
  for (const Row &row : scenario.rows)
    for (const std::string &note : row.notes)
      if (std::ranges::find(notes, note) == notes.end()) notes.push_back(note);
  return notes;
}

std::string gpus_text(const std::vector<Device> &devices) {
  auto name_of = [](std::string name) {
    for (const std::string_view mark : {"(R)", "(TM)"})
      for (size_t at; (at = name.find(mark)) != std::string::npos;) name.erase(at, mark.size());
    return name;
  };
  std::string out;
  for (size_t d = 0; d < devices.size();) {
    size_t same = d + 1;
    while (same < devices.size() && devices[same].name == devices[d].name && devices[same].memory_bytes == devices[d].memory_bytes &&
           devices[same].memory_gb == devices[d].memory_gb)
      ++same;
    std::string part = std::to_string(same - d) + "x " + name_of(devices[d].name);
    if (devices[d].memory_gb) part += ", " + std::to_string(devices[d].memory_gb) + " GB";  // the size on the box
    else if (devices[d].memory_bytes) {  // only the driver's figure: in GiB, as it is
      char gib[32];
      std::snprintf(gib, sizeof gib, ", %.1f GiB", static_cast<double>(devices[d].memory_bytes) / 1073741824.0);
      part += gib;
    }
    out += (out.empty() ? "" : " + ") + part;
    d = same;
  }
  return out;
}

std::string devices_text(const std::vector<Device> &devices, int tensor_parallel, std::string_view missing) {
  std::string out = gpus_text(devices);
  if (tensor_parallel > 0) out += (out.empty() ? "TP " : "; TP ") + std::to_string(tensor_parallel);
  return out.empty() ? std::string(missing) : out;
}

std::string to_json(const Run &run) {
  const RunInfo &i = run.info;
  json::Value root = json::Value::object();
  root.set("schema", json::Value::integer(kRunSchema));
  root.set("tool", json::Value::string("xenithium-bench"));
  json::Value env = json::Value::object();
  env.set("name", json::Value::string(i.name));
  env.set("timestamp", json::Value::string(i.timestamp));
  env.set("host", json::Value::string(i.host));
  env.set("endpoint", json::Value::string(i.endpoint));
  if (!i.engine.empty()) env.set("engine", json::Value::string(i.engine));
  if (!i.engine_build.empty()) env.set("engine_build", json::Value::string(i.engine_build));
  if (!i.engine_oneapi.empty()) env.set("engine_oneapi", json::Value::string(i.engine_oneapi));
  if (!i.engine_drafter.empty()) env.set("engine_drafter", json::Value::string(i.engine_drafter));
  if (!i.server_args.empty() || !i.server_env.empty()) {
    json::Value server = json::Value::object();
    server.set("args", strings_json(i.server_args));
    server.set("env", labels_json(i.server_env));
    env.set("server", std::move(server));
  }
  if (!i.command.empty()) env.set("command", strings_json(i.command));
  env.set("model", json::Value::string(i.model));
  env.set("context_tokens", json::Value::integer(i.context_tokens));
  env.set("kv_pool_tokens", json::Value::integer(i.kv_pool_tokens));
  if (i.tensor_parallel > 0) env.set("tensor_parallel", json::Value::integer(i.tensor_parallel));
  if (!i.devices.empty()) env.set("devices", devices_json(i.devices));
  if (!i.machine.kernel.empty()) env.set("machine", machine_json(i.machine));
  if (!i.bench_machine.kernel.empty()) env.set("bench_machine", machine_json(i.bench_machine));
  env.set("seconds", json::Value::real(i.seconds));
  env.set("notes", labels_json(i.notes));
  json::Value caps = json::Value::object();
  caps.set("ignore_eos", json::Value::boolean(i.capabilities.ignore_eos));
  caps.set("top_k", json::Value::boolean(i.capabilities.top_k));
  caps.set("thinking_switch", json::Value::boolean(i.capabilities.thinking_switch));
  caps.set("stream_usage", json::Value::boolean(i.capabilities.stream_usage));
  caps.set("metrics", json::Value::boolean(i.capabilities.metrics));
  env.set("capabilities", std::move(caps));
  root.set("env", std::move(env));
  if (i.power) root.set("power", run_power_json(*i.power));
  json::Value config = json::Value::object();
  config.set("suite", json::Value::string(i.suite));
  config.set("seed", json::Value::integer(static_cast<int64_t>(i.seed)));
  config.set("temperature", json::Value::real(i.sampling.temperature));
  config.set("top_p", json::Value::real(i.sampling.top_p));
  config.set("top_k", json::Value::integer(i.sampling.top_k));
  config.set("thinking", json::Value::string(thinking_or_default(i.thinking)));
  config.set("mode", json::Value::string(i.mode));
  config.set("mode_version", json::Value::integer(i.mode_version));
  config.set("reply_length", json::Value::string(i.reply_length));
  config.set("corpus", json::Value::string(i.corpus));
  config.set("corpus_version", json::Value::integer(i.corpus_version));
  config.set("corpus_hash", json::Value::string(i.corpus_hash));
  root.set("config", std::move(config));
  root.set("summary", headline_json(headline(run)));
  json::Value scenarios = json::Value::array();
  for (const ScenarioResult &s : run.scenarios) scenarios.push(scenario_json(s));
  root.set("scenarios", std::move(scenarios));
  return json::dump(root, json::Style::python);
}

std::optional<Run> run_from_json(std::string_view text, std::string &err) {
  json::Value root;
  if (!json::parse(text, root, &err)) return std::nullopt;
  if (text_at(root, "tool") != "xenithium-bench") return err = "not a xenithium-bench run file", std::nullopt;
  if (static_cast<int>(number_at(root, "schema")) > kRunSchema) return err = "the run file is of a newer schema than this tool reads", std::nullopt;
  Run run;
  RunInfo &i = run.info;
  if (const json::Value *env = root.get_object("env")) {
    i.name = text_at(*env, "name"), i.timestamp = text_at(*env, "timestamp"), i.host = text_at(*env, "host");
    i.endpoint = text_at(*env, "endpoint"), i.model = text_at(*env, "model"), i.engine = text_at(*env, "engine");
    i.engine_build = text_at(*env, "engine_build"), i.engine_oneapi = text_at(*env, "engine_oneapi"), i.engine_drafter = text_at(*env, "engine_drafter");
    if (const json::Value *server = env->get_object("server")) {
      i.server_args = strings_from(server->get_array("args"));
      i.server_env = labels_from(server->get_object("env"));
    }
    i.command = strings_from(env->get_array("command"));
    i.context_tokens = static_cast<int64_t>(number_at(*env, "context_tokens"));
    i.kv_pool_tokens = static_cast<int64_t>(number_at(*env, "kv_pool_tokens"));
    i.tensor_parallel = static_cast<int>(number_at(*env, "tensor_parallel"));  // runs from before: 0, no devices
    i.devices = devices_from(env->get_array("devices"));
    if (const json::Value *m = env->get_object("machine")) i.machine = machine_from(*m);
    if (const json::Value *m = env->get_object("bench_machine")) i.bench_machine = machine_from(*m);
    i.seconds = number_at(*env, "seconds");
    i.notes = labels_from(env->get_object("notes"));
    if (const json::Value *caps = env->get_object("capabilities"))
      i.capabilities = {.ignore_eos = flag_at(*caps, "ignore_eos"), .top_k = flag_at(*caps, "top_k"), .thinking_switch = flag_at(*caps, "thinking_switch"),
                        .stream_usage = flag_at(*caps, "stream_usage"), .metrics = flag_at(*caps, "metrics"), .max_model_len = i.context_tokens, .kv_pool_tokens = i.kv_pool_tokens};
  }
  if (const json::Value *config = root.get_object("config")) {
    i.suite = text_at(*config, "suite"), i.thinking = text_at(*config, "thinking"), i.corpus_hash = text_at(*config, "corpus_hash");
    // runs from before the option always asked for fixed lengths
    if (const std::string length = text_at(*config, "reply_length"); !length.empty()) i.reply_length = length;
    i.seed = static_cast<uint64_t>(number_at(*config, "seed"));
    i.corpus_version = static_cast<int>(number_at(*config, "corpus_version"));
    i.corpus = text_at(*config, "corpus");
    if (i.corpus.empty()) i.corpus = "hard";  // runs from before the second set read the engine's own code
    i.mode = text_at(*config, "mode");
    i.mode_version = std::max(1, static_cast<int>(number_at(*config, "mode_version")));
    if (i.mode.empty()) {  // runs from before the modes: the set named "easy" then is the medium corpus now
      if (i.corpus == "easy") i.corpus = "medium";
      i.mode = i.corpus == "medium" ? "medium" : "hard";
    }
    i.sampling = {.temperature = number_at(*config, "temperature"), .top_p = number_at(*config, "top_p"), .top_k = static_cast<int>(number_at(*config, "top_k")), .seed = std::nullopt};
  }
  if (const json::Value *power = root.get_object("power")) i.power = run_power_from(*power);
  if (const json::Value *scenarios = root.get_array("scenarios"))
    for (const json::Value &s : scenarios->arr) run.scenarios.push_back(scenario_from(s));
  return run;
}

std::string power_line(const ScenarioResult &scenario) {
  constexpr double kJoulesPerKilojoule = 1000;
  const PowerUse &use = *scenario.power;
  std::string line = "power: " + format_number(use.watts()) + " W mean";
  if (use.watts_per_card.size() > 1) {
    line += " (";
    for (size_t card = 0; card < use.watts_per_card.size(); ++card) line += (card ? ", card " : "card ") + std::to_string(card) + " " + format_number(use.watts_per_card[card]);
    line += ")";
  }
  line += ", " + format_number(use.joules / kJoulesPerKilojoule) + " kJ";
  if (use.mean_mhz > 0) line += "; GPU clock mean " + format_number(use.mean_mhz) + " MHz, lowest " + format_number(use.min_mhz);
  return line;
}

namespace {

TableView table_of_rows(std::string caption, std::span<const Row *const> rows) {
  const Columns cols = columns_of(rows);
  TableView view{std::move(caption), cols.labels, {}, cols.labels.size()};
  for (size_t c = 0; c < cols.metrics.size(); ++c) view.header.push_back(column_title(*cols.metrics[c], cols.with_worst[c]));
  for (const Row *row : rows) {
    std::vector<std::string> line;
    for (const std::string &label : cols.labels) {
      const std::string *value = label_value(*row, label);
      line.push_back(value ? *value : "");
    }
    for (const Metric *col : cols.metrics) {
      const Metric *m = metric_named(*row, col->name);
      line.push_back(m ? cell_of(*m) : "-");
    }
    view.rows.push_back(std::move(line));
  }
  return view;
}

}  // namespace

std::vector<TableView> tables_of(const ScenarioResult &scenario) {
  std::vector<std::pair<std::string, std::vector<const Row *>>> groups;  // (Row:table, its rows), first-seen order
  for (const Row &row : scenario.rows) {
    if (row.labels.empty() && row.metrics.empty()) continue;  // a row that only carries notes
    auto group = std::ranges::find(groups, row.table, [](const auto &g) { return g.first; });
    if (group == groups.end()) group = groups.insert(groups.end(), {row.table, {}});
    group->second.push_back(&row);
  }
  std::vector<TableView> tables;
  for (const auto &[caption, rows] : groups) tables.push_back(table_of_rows(caption, rows));
  return tables;
}

std::string render_scenario(const ScenarioResult &scenario) {
  char head[160];
  std::snprintf(head, sizeof(head), "   [%s: %.0f s, %d requests%s]", scenario.id.c_str(), scenario.seconds, scenario.requests,
                scenario.errors ? (", " + std::to_string(scenario.errors) + " FAILED").c_str() : "");
  std::string out = "== " + scenario.title + head + "\n";
  for (const TableView &view : tables_of(scenario)) {
    if (!view.caption.empty()) out += std::string(kIndent) + "-- " + view.caption + "\n";
    std::vector<std::vector<std::string>> cells = {view.header};
    cells.insert(cells.end(), view.rows.begin(), view.rows.end());
    out += table(cells);
  }
  if (!scenario.misses.empty()) {
    // a line each, marked, before the notes: a wrong answer is a result, not a remark
    out += std::string(kIndent) + "\xE2\x9C\x97 " + misses_heading(scenario) + ":\n";
    for (const Miss &m : scenario.misses)
      out += std::string(kIndent) + "  \xE2\x9C\x97 " + m.item + (m.group.empty() ? "" : " (" + m.group + ")") + ": expected \"" + m.expected + "\", got " + given_text(m) + "\n";
  }
  for (const std::string &note : unique_notes(scenario)) out += std::string(kIndent) + "note: " + note + "\n";
  if (scenario.power) out += std::string(kIndent) + power_line(scenario) + "\n";
  if (scenario.errors) out += std::string(kIndent) + "first error: " + scenario.first_error + "\n";
  return out;
}

std::string render_run(const Run &run) {
  const RunInfo &i = run.info;
  std::string out = "xenithium-bench run " + i.timestamp + "-" + i.name + "\n";
  out += std::string(kIndent) + "endpoint " + i.endpoint + ", model " + i.model + (i.context_tokens ? ", context " + with_thousands(i.context_tokens) + " tokens" : "") +
         (i.kv_pool_tokens ? ", KV pool " + with_thousands(i.kv_pool_tokens) + " tokens" : "") + "\n";
  if (const std::string on = devices_text(i.devices, i.tensor_parallel, ""); !on.empty()) out += std::string(kIndent) + "runs on " + on + "\n";
  char line[520];
  std::snprintf(line, sizeof(line), "mode %s v%d, suite %s, seed %llu, sampling T %.2f / top-p %.2f / top-k %d, thinking %.*s, replies %s, corpus %s v%d %s, %.0f s", i.mode.c_str(),
                i.mode_version, i.suite.c_str(), static_cast<unsigned long long>(i.seed), i.sampling.temperature, i.sampling.top_p, i.sampling.top_k,
                static_cast<int>(thinking_or_default(i.thinking).size()), thinking_or_default(i.thinking).data(), i.reply_length.c_str(), i.corpus.c_str(), i.corpus_version, i.corpus_hash.c_str(),
                i.seconds);
  out += std::string(kIndent) + line + "\n";
  const Headline h = headline(run);
  if (h.decode_tps > 0) {
    const std::string mix = h.decode_mix.empty() || h.decode_mix == "weighted" ? "weighted over the categories" : "the " + h.decode_mix + " of the categories";
    out += std::string(kIndent) + "headline: decode " + format_number(h.decode_tps) + " tok/s (" + mix + ")\n";
  }
  for (const ScenarioResult &s : run.scenarios) out += "\n" + render_scenario(s);
  return out;
}

std::string render_compare(const Run &base, const Run &other) {
  std::string out = "base:  " + base.info.timestamp + "-" + base.info.name + "  (" + base.info.model + ")\nother: " + other.info.timestamp + "-" + other.info.name + "  (" +
                    other.info.model + ")\n";
  if (base.info.corpus_hash != other.info.corpus_hash || base.info.corpus_version != other.info.corpus_version)
    out += "WARNING: the runs used different corpora or prompt sets (" + base.info.corpus + " version " + std::to_string(base.info.corpus_version) + " against " +
           other.info.corpus + " version " + std::to_string(other.info.corpus_version) + "): their prompts differ.\n";
  if (base.info.mode != other.info.mode || base.info.mode_version != other.info.mode_version)
    out += "WARNING: the runs measured in different modes (" + base.info.mode + " v" + std::to_string(base.info.mode_version) + " against " + other.info.mode + " v" +
           std::to_string(other.info.mode_version) + "): their prompts and method differ, so their numbers do not compare.\n";
  if (base.info.model != other.info.model) out += "note: the runs are of different models.\n";
  if (base.info.reply_length != other.info.reply_length)
    out += "WARNING: the replies were of " + base.info.reply_length + " length against " + other.info.reply_length +
           " length: fixed replies run on past the model's own end, so their decode rates do not compare with natural ones.\n";
  for (const ScenarioResult &b : base.scenarios) {
    const ScenarioResult *o = scenario_named(other, b.id);
    if (!o) continue;
    std::vector<std::vector<std::string>> cells = {{"row", "metric", "base", "other", "change"}};
    for (const Row &brow : b.rows) {
      const auto match = std::ranges::find_if(o->rows, [&](const Row &r) { return row_key(r) == row_key(brow); });
      if (match == o->rows.end() || brow.metrics.empty()) continue;
      std::string name;
      for (const Label &label : brow.labels) name += (name.empty() ? "" : ", ") + label.first + " " + label.second;
      for (const Metric &bm : brow.metrics) {
        const Metric *om = metric_named(*match, bm.name);
        if (!om) continue;
        cells.push_back({name, column_title(bm, false), cell_of(bm), cell_of(*om), bm.unit == kFlagUnit ? "" : change_text(bm, change_of(bm, *om))});
        name.clear();  // the row's name once, on its first metric
      }
    }
    if (cells.size() > 1) out += "\n== " + b.title + "\n" + table(cells);
  }
  return out;
}

}  // namespace xenithium:bench
