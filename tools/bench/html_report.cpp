#include "html_report.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <tuple>
#include <pwd.h>
#include <unistd.h>

#include "format.hpp"
#include "html_assets.hpp"
#include "scenario.hpp"
#include "stats.hpp"
#include "svg_charts.hpp"

namespace xenithium::bench {
namespace {

using svg::duration_text;

bool wordy(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }
bool pathy(char c) { return wordy(c) || c == '-' || c == '.' || c == '/'; }
bool pathy_not_slash(char c) { return c != '/' && pathy(c); }

// Replaces each occurrence of `name` in `out` by `as` unless a neighbour continues it: before(c) / after(c) true = the
// character before / after belongs to a longer word or path, and that occurrence stays.
void mask(std::string &out, const std::string &name, std::string_view as, bool (*before)(char), bool (*after)(char)) {
  for (size_t pos = out.find(name); pos != std::string::npos; pos = out.find(name, pos)) {
    const size_t end = pos + name.size();
    if ((pos > 0 && before(out[pos - 1])) || (end < out.size() && after(out[end]))) {
      pos = end;
      continue;
    }
    out.replace(pos, name.size(), as);
    pos += as.size();
  }
}

// The report page leaves the machine, and results.json / report.txt stay local: so the page masks the two identities
// a value can carry: the login name and the home directory's path. The home directory where a path starts with it
// (not inside another path: /x/root/y with the home /root) becomes $HOME; a login name standing as its own word (the
// host name of a machine named for its user) becomes $USER. The strings written are escaped as before.
std::string escaped(std::string_view text) {
  std::string out(text);
  static const std::string home = [] {
    const char *h = std::getenv("HOME");
    return h && *h ? std::string(h) : std::string();
  }();
  if (home.size() > 1) mask(out, home, "$HOME", pathy, pathy_not_slash);
  static const std::string user = [] {
    const passwd *p = getpwuid(geteuid());
    return p ? std::string(p->pw_name) : std::string();
  }();
  if (user.size() > 3) mask(out, user, "$USER", wordy, wordy);  // a shorter name would mask its way through ordinary words
  return svg::escaped(out);
}

// Stands in for the file's own size until the page is whole.
constexpr std::string_view kSizeToken = "@SIZE@";
constexpr double kBytesPerKilobyte = 1024;
constexpr double kSecondsPerMinute = 60, kMsPerSecond = 1000;
constexpr std::string_view kDot = " \xC2\xB7 ";  // a middle dot between two facts

// ---- small text helpers ----

// A latency for prose: "55 ms", "7.4 s", "1 min 2 s".
std::string latency_text(double ms) {
  if (ms < kMsPerSecond) return format_number(ms) + " ms";
  if (ms < kSecondsPerMinute * kMsPerSecond) return format_number(ms / kMsPerSecond) + " s";
  return duration_text(ms / kMsPerSecond);
}

// "20261116-200152" -> "2026-11-16 20:01"; anything else as it is.
std::string timestamp_text(const std::string &stamp) {
  constexpr size_t kStampLength = 15;
  if (stamp.size() != kStampLength || stamp[8] != '-') return stamp;
  return stamp.substr(0, 4) + "-" + stamp.substr(4, 2) + "-" + stamp.substr(6, 2) + " " + stamp.substr(9, 2) + ":" + stamp.substr(11, 2);
}

const std::string *label_of(const Row &row, std::string_view name) {
  for (const Label &label : row.labels)
    if (label.first == name) return &label.second;
  return nullptr;
}
const Metric *metric_of(const Row &row, std::string_view name) {
  for (const Metric &m : row.metrics)
    if (m.name == name) return &m;
  return nullptr;
}

// "32,000" -> 32000, "0.5" -> 0.5; nullopt for a label that is not a number ("1 alone").
std::optional<double> number_of_label(const std::string &label) {
  std::string digits;
  for (const char c : label)
    if (c != ',') digits.push_back(c);
  char *end = nullptr;
  const double value = std::strtod(digits.c_str(), &end);
  return !digits.empty() && end == digits.c_str() + digits.size() ? std::optional(value) : std::nullopt;
}

// ---- charts from a scenario's rows ----

// A row label this short is its own axis tick ("0.5", "16"); a longer one is shortened ("250,000" -> "250K").
constexpr size_t kTickLabelChars = 4;

struct Chart {
  std::string svg;
  std::string unit;
};

std::optional<Chart> bars_of(const std::vector<const Row *> &rows, const ChartSpec &spec) {
  std::vector<std::string> categories;
  for (const Row *row : rows) categories.push_back(*label_of(*row, spec.x_label));
  std::vector<svg::BarSeries> series;
  std::string unit;
  for (const std::string &name : spec.metrics) {
    svg::BarSeries s{name, {}};
    bool any = false;
    for (const Row *row : rows) {
      const Metric *m = metric_of(*row, name);
      const bool has = m && !m->samples.empty();
      s.values.push_back(has ? summarize(m->samples).median : std::nan(""));
      if (has && unit.empty()) unit = m->unit;
      any |= has;
    }
    if (any) series.push_back(std::move(s));
  }
  if (series.empty()) return std::nullopt;
  return Chart{svg::bar_chart(categories, series, {unit, spec.x_label}), unit};
}

std::optional<Chart> line_of(const std::vector<const Row *> &rows, const ChartSpec &spec) {
  std::vector<svg::LineSeries> series;
  std::string unit;
  for (const std::string &name : spec.metrics) {
    svg::LineSeries s{name, {}};
    for (const Row *row : rows) {
      const Metric *m = metric_of(*row, name);
      const auto x = number_of_label(*label_of(*row, spec.x_label));
      if (!m || m->samples.empty() || !x) continue;
      const std::string &label = *label_of(*row, spec.x_label);
      s.points.push_back({*x, summarize(m->samples).median, label.size() <= kTickLabelChars ? label : short_number(*x)});
      if (unit.empty()) unit = m->unit;
    }
    std::ranges::sort(s.points, {}, &svg::LinePoint::x);
    if (!s.points.empty()) series.push_back(std::move(s));
  }
  if (series.empty()) return std::nullopt;
  return Chart{svg::line_chart(series, spec.log_x, {unit, spec.x_label}), unit};
}

std::optional<Chart> range_of(const std::vector<const Row *> &rows, const ChartSpec &spec) {
  std::vector<svg::RangeRow> ranges;
  std::string unit;
  for (const Row *row : rows)
    if (const Metric *m = spec.metrics.empty() ? nullptr : metric_of(*row, spec.metrics.front()); m && !m->samples.empty()) {
      ranges.push_back({*label_of(*row, spec.x_label), summarize(m->samples)});
      unit = m->unit;
    }
  if (ranges.empty()) return std::nullopt;
  return Chart{svg::range_chart(ranges, {unit, spec.x_label}), unit};
}

std::optional<Chart> dots_of(const std::vector<const Row *> &rows, const ChartSpec &spec, const std::vector<Miss> &misses) {
  std::vector<svg::DotRow> dots;
  for (const Row *row : rows)
    if (const Metric *m = spec.metrics.empty() ? nullptr : metric_of(*row, spec.metrics.front()); m && !m->samples.empty()) {
      svg::DotRow dot{*label_of(*row, spec.x_label), {}, {}};
      for (const double sample : m->samples) dot.yes.push_back(sample > 0.5);
      for (const Miss &miss : misses)
        if (miss.group == dot.label) dot.wrong.push_back(miss.item);
      dots.push_back(std::move(dot));
    }
  if (dots.empty()) return std::nullopt;
  return Chart{svg::dot_chart(dots), {}};
}

// The chart a scenario declared, or nothing when its rows do not hold the metrics (power not sampled, a skipped case).
std::optional<Chart> chart_of(const ScenarioResult &scenario, const ChartSpec &spec) {
  std::vector<const Row *> rows;
  for (const Row &row : scenario.rows)
    if (label_of(row, spec.x_label)) rows.push_back(&row);
  if (rows.empty()) return std::nullopt;
  switch (spec.kind) {
    case ChartKind::bars: return bars_of(rows, spec);
    case ChartKind::line: return line_of(rows, spec);
    case ChartKind::range: return range_of(rows, spec);
    case ChartKind::dots: return dots_of(rows, spec, scenario.misses);
  }
  return std::nullopt;
}

// ---- the page's parts ----

struct Figure {
  std::string label, value, unit, detail;
  std::string href;  // the scenario the figure comes from
  std::string alert;  // failures behind the figure ("1 wrong answer"): a line of its own in the failure colour
};

// The anchor of a scenario's misses panel.
std::string misses_anchor(const std::string &scenario_id) { return scenario_id + "-misses"; }

std::vector<Figure> figures_of(const Run &run) {
  const Headline h = headline(run);
  std::vector<Figure> figures;
  if (h.decode_tps > 0) {
    const std::string mix = h.decode_mix.empty() || h.decode_mix == "weighted" ? "weighted over " : "the " + h.decode_mix + " of ";
    figures.push_back({"Decode", format_number(h.decode_tps), "tok/s", "one stream, " + mix + std::to_string(h.decode_by_category.size()) + " kinds of output", "#decode", {}});
  }
  if (h.peak_total_tps > 0) figures.push_back({"Throughput", format_number(h.peak_total_tps), "tok/s", "all streams together, at " + h.peak_streams + " streams", "#concurrency", {}});
  if (!h.prefill.empty() && h.prefill.back().tps > 0)
    figures.push_back({"Prompt processing", format_number(h.prefill.back().tps), "tok/s",
                       "a " + h.prefill.back().depth + "-token prompt, first token after " + latency_text(h.prefill.back().ttft_ms), "#prefill", {}});
  if (!h.deep_decode.empty() && h.deep_decode.back().tps > 0)
    figures.push_back({"Decode at depth", format_number(h.deep_decode.back().tps), "tok/s", "with " + h.deep_decode.back().depth + " tokens of context", "#deep-decode", {}});
  if (h.best_tokens_per_joule > 0)
    figures.push_back({"Efficiency", format_number(h.best_tokens_per_joule), "tok/J", "tokens per joule of board power, at the best stream count", "#concurrency", {}});
  if (h.quality_pct >= 0) {
    const size_t wrong = h.quality_misses.size();
    figures.push_back({"Quality", format_number(h.quality_pct), "%", "of the checkable questions answered correctly", wrong ? "#" + misses_anchor("quality") : "#quality",
                       wrong ? "\xE2\x9C\x97 " + std::to_string(wrong) + (wrong == 1 ? " wrong answer" : " wrong answers") : std::string()});
  }
  return figures;
}

std::string figures_html(const std::vector<Figure> &figures) {
  if (figures.empty()) return {};
  std::string html = "<div class=\"figs\">";
  for (const Figure &f : figures)
    html += std::string("<a class=\"fig") + (f.alert.empty() ? "" : " bad") + "\" href=\"" + escaped(f.href) + "\"><div class=\"k\">" + escaped(f.label) + "</div><div class=\"n\">" + escaped(f.value) + "<small>" + escaped(f.unit) +
            "</small></div><div class=\"d\">" + escaped(f.detail) + "</div>" + (f.alert.empty() ? std::string() : "<div class=\"w\">" + escaped(f.alert) + "</div>") + "</a>";
  return html + "</div>";
}

int failed_requests(const Run &run) {
  int failed = 0;
  for (const ScenarioResult &s : run.scenarios) failed += s.errors;
  return failed;
}
int all_requests(const Run &run) {
  int requests = 0;
  for (const ScenarioResult &s : run.scenarios) requests += s.requests;
  return requests;
}

// Every yes / no metric of the run as "scenario: metric 3/3" (a link to the scenario), after the count of failed requests.
std::string checks_html(const Run &run) {
  std::string items;
  for (const ScenarioResult &scenario : run.scenarios) {
    std::vector<std::pair<std::string, std::pair<int, int>>> flags;  // metric -> (yes, asked), in first-seen order
    for (const Row &row : scenario.rows)
      for (const Metric &m : row.metrics) {
        if (m.unit != kFlagUnit) continue;
        auto it = std::ranges::find(flags, m.name, [](const auto &flag) { return flag.first; });
        if (it == flags.end()) it = flags.insert(flags.end(), {m.name, {0, 0}});
        it->second.first += static_cast<int>(std::ranges::count_if(m.samples, [](double v) { return v > 0.5; }));
        it->second.second += static_cast<int>(m.samples.size());
      }
    const std::string target = scenario.misses.empty() ? scenario.id : misses_anchor(scenario.id);
    for (const auto &[name, count] : flags)
      items += std::string("<li") + (count.first == count.second ? "" : " class=\"bad\"") + "><a href=\"#" + escaped(count.first == count.second ? scenario.id : target) + "\"><span>" +
               escaped(scenario.id) + "</span> " + escaped(name) + " " + std::to_string(count.first) + "/" + std::to_string(count.second) + "</a></li>";
  }
  const int failed = failed_requests(run);
  std::string html = "<ul class=\"checks\"><li";
  html += failed ? " class=\"bad\">" : ">";
  html += std::to_string(failed) + " of " + with_thousands(all_requests(run)) + " requests failed</li>";
  return html + items + "</ul>";
}

// What the timeline's tooltip says about a scenario besides its start and length.
std::vector<svg::TipLine> span_details(const ScenarioResult &s) {
  std::vector<svg::TipLine> lines = {{"requests", with_thousands(s.requests)}};
  if (s.errors) lines.push_back({"failed", std::to_string(s.errors)});
  if (s.completion_tokens > 0) lines.push_back({"tokens generated", with_thousands(s.completion_tokens)});
  if (s.power) lines.push_back({"power", format_number(s.power->watts()) + " W mean"});
  return lines;
}

std::string timeline_html(const Run &run) {
  std::vector<svg::TimelineSpan> spans;
  for (const ScenarioResult &s : run.scenarios) spans.push_back({s.id, s.started_s, s.started_s + s.seconds, span_details(s), "#" + s.id});
  svg::TimelinePower power;
  std::string legend;
  if (run.info.power && !run.info.power->trace.empty()) {
    const RunPower &p = *run.info.power;
    power.watts.resize(p.cards.size());
    for (const PowerPoint &point : p.trace) {
      power.at_s.push_back(point.at_s);
      power.mhz.push_back(point.mhz);
      for (size_t card = 0; card < p.cards.size(); ++card) power.watts[card].push_back(card < point.watts.size() ? point.watts[card] : 0);
    }
    for (const PowerCard &card : p.cards) power.cap_watts += card.cap_watts, power.card_names.push_back(card.name);
    power.idle_watts = p.idle_watts;
    legend = "<p class=\"legend\">";
    // each card's swatch in its band's shade (the bands cycle through the series shades)
    for (size_t card = 0; card < p.cards.size(); ++card)
      legend += "<i class=\"area" + std::to_string(card % svg::kSeriesColors) + "\"></i>" + escaped(p.cards[card].name);
    legend += " &nbsp;board power from the driver's energy counters, stacked: the whole card, whatever runs on it; the host's CPU is not counted.";
    if (power.cap_watts > 0 || power.idle_watts > 0) {
      legend += " Dashed:";
      if (power.cap_watts > 0) legend += " the power limit of " + format_number(power.cap_watts) + " W";
      if (power.cap_watts > 0 && power.idle_watts > 0) legend += " and";
      if (power.idle_watts > 0) legend += " the " + format_number(power.idle_watts) + " W drawn before the run";
      legend += ".";
    }
    legend += " <i class=\"clk\"></i>the GPU clock.</p>";
  }
  return "<section id=\"run\"><h2><span class=\"ttl\">The run</span> <code>" + duration_text(run.info.seconds) + "</code></h2><figure class=\"wide\">" +
         svg::timeline(spans, run.info.seconds, power) + "</figure>" + legend +
         "<p class=\"hint\" hidden>Point at a scenario for its numbers and click it to go there; point at the power trace for the readings at that moment.</p></section>";
}

// The diagram of one request, with the medians of the decode scenario's first category.
std::string anatomy_html(const Run &run) {
  const auto decode = std::ranges::find(run.scenarios, std::string("decode"), &ScenarioResult::id);
  if (decode == run.scenarios.end() || decode->rows.empty()) return {};
  const Row &row = decode->rows.front();
  const Metric *ttft = metric_of(row, metric::kTtft), *gap = metric_of(row, metric::kUpdateGap), *per = metric_of(row, metric::kTokensPerUpdate), *rate = metric_of(row, metric::kDecode);
  if (!ttft || !gap || !per || !rate || row.labels.empty()) return {};
  const svg::RequestAnatomy request{summarize(ttft->samples).median, summarize(gap->samples).median, summarize(per->samples).median, summarize(rate->samples).median};
  return "<section id=\"request\"><h2><span class=\"ttl\">A request, as measured</span></h2><p class=\"about\">Every number in this report comes from time stamps on a streamed "
         "reply. Shown: the median " +
         escaped(row.labels.front().second) + " reply of the decode scenario (" + std::to_string(ttft->samples.size()) +
         " requests). The drawing is not to scale.</p><figure class=\"wide\">" + svg::request_anatomy(request) + "</figure></section>";
}

std::string table_html(const TableView &view) {
  const auto cell = [&](std::string_view tag, size_t column, const std::string &text) {
    return "<" + std::string(tag) + (column < view.label_columns ? " class=\"l\">" : ">") + escaped(text) + "</" + std::string(tag) + ">";
  };
  std::string html = "<div class=\"tbl\"><div class=\"tbar\"><span>" + escaped(view.caption.empty() ? "the median of each row's samples" : view.caption) +
                     "</span></div><div class=\"tscroll\"><table><thead><tr>";
  for (size_t c = 0; c < view.header.size(); ++c) html += cell("th", c, view.header[c]);
  html += "</tr></thead><tbody>";
  for (const auto &row : view.rows) {
    html += "<tr>";
    for (size_t c = 0; c < row.size(); ++c) html += cell("td", c, row[c]);
    html += "</tr>";
  }
  return html + "</tbody></table></div></div>";
}

// A scenario's wrong answers, first in its section and in the failure colour: per miss the check, its category, the
// question, and the expected answer beside the one given.
std::string misses_html(const ScenarioResult &scenario) {
  if (scenario.misses.empty()) return {};
  std::string html = "<div class=\"miss\" id=\"" + escaped(misses_anchor(scenario.id)) + "\"><div class=\"mbar\"><b>\xE2\x9C\x97 " + escaped(misses_heading(scenario)) + "</b></div><ol>";
  for (const Miss &m : scenario.misses) {
    html += "<li><div class=\"mq\"><code>" + escaped(m.item) + "</code>" + (m.group.empty() ? std::string() : "<span>" + escaped(m.group) + "</span>") + "</div>";
    // a question of several lines holds a program: kept as laid out
    if (!m.asked.empty()) html += m.asked.find('\n') == std::string::npos ? "<p>" + escaped(m.asked) + "</p>" : "<pre>" + escaped(m.asked) + "</pre>";
    html += "<dl><dt>expected</dt><dd class=\"want\">" + escaped(m.expected) + "</dd><dt>answer given</dt>";
    html += m.why.empty() ? "<dd class=\"got\">" + escaped(m.got) + "</dd>" : "<dd class=\"none\">" + escaped(m.why) + "</dd>";
    html += "</dl></li>";
  }
  return html + "</ol></div>";
}

std::string scenario_html(const ScenarioResult &scenario) {
  std::string html = "<section class=\"sc\" id=\"" + escaped(scenario.id) + "\"><h2><span class=\"ttl\">" + escaped(scenario.title) + "</span> <code>" + escaped(scenario.id) + "</code></h2>";
  if (!scenario.about.empty()) html += "<p class=\"about\">" + escaped(scenario.about) + "</p>";
  html += "<p class=\"meta\">" + duration_text(scenario.seconds) + std::string(kDot) + with_thousands(scenario.requests) + " requests";
  if (scenario.errors) html += std::string(kDot) + "<b>" + std::to_string(scenario.errors) + " failed</b>";
  if (scenario.power) html += std::string(kDot) + escaped(power_line(scenario));
  html += "</p>" + misses_html(scenario);

  std::string charts;
  for (const ChartSpec &spec : scenario.charts)
    if (const auto chart = chart_of(scenario, spec))
      charts += "<figure><figcaption>" + escaped(spec.title) + (chart->unit.empty() ? "" : " <span>\xC2\xB7 " + escaped(chart->unit) + "</span>") + "</figcaption>" + chart->svg + "</figure>";
  if (!charts.empty()) html += "<div class=\"charts\">" + charts + "</div>";
  for (const TableView &view : tables_of(scenario)) html += table_html(view);

  std::string notes;
  if (scenario.errors) notes += "<li class=\"err\">first error: " + escaped(scenario.first_error) + "</li>";
  for (const std::string &note : unique_notes(scenario)) notes += "<li>" + escaped(note) + "</li>";
  if (!notes.empty()) html += "<ul class=\"notes\">" + notes + "</ul>";
  return html + "</section>";
}

std::string term(std::string_view name, const std::string &value) { return "<dt>" + escaped(name) + "</dt><dd>" + escaped(value) + "</dd>"; }

std::string replies_text(const RunInfo &i) {
  if (i.reply_length != "fixed") return i.reply_length + ": the sampled scenarios' replies end where the model ends them";
  return std::string("fixed: the sampled scenarios' replies run to their length") + (i.capabilities.ignore_eos ? " (ignore_eos)" : ", asked for but not taken by the endpoint");
}

// The page's last section: what the facts rows under the title leave out (where and when, the suite, the replies, which
// API extensions the endpoint took or not, the power limits, the cards' PCI addresses), then every note and every XEN_
// variable the run recorded, as they were recorded.
std::string setup_html(const RunInfo &i) {
  const Capabilities &caps = i.capabilities;
  std::string extensions;
  for (const auto &[name, taken] : {std::pair{"ignore_eos", caps.ignore_eos}, {"top_k", caps.top_k}, {"thinking switch", caps.thinking_switch},
                                    {"usage in the stream", caps.stream_usage}, {"/metrics", caps.metrics}})
    extensions += std::string(extensions.empty() ? "" : ", ") + name + (taken ? " yes" : " no");

  std::string html = "<section id=\"setup\"><h2><span class=\"ttl\">Everything the run recorded</span></h2><dl class=\"setup\">";
  html += term("run", i.timestamp + "-" + i.name) + term("endpoint", i.endpoint) + term("served model", i.model) + term("bench ran on", i.host);
  html += term("suite", i.suite) + term("mode", i.mode + ", version " + std::to_string(i.mode_version)) + term("replies", replies_text(i));
  html += term("API extensions", extensions);
  std::string pci;
  for (const Device &d : i.devices)
    if (!d.pci.empty()) pci += (pci.empty() ? "" : ", ") + d.pci;
  if (!pci.empty()) html += term("PCI", pci);
  if (const util::Machine &b = i.bench_machine; !b.kernel.empty())
    html += term("bench machine", b.os + ", kernel " + b.kernel + ", " + b.cpu + " (" + std::to_string(b.cpu_threads) + " threads), " + std::to_string(b.memory_gib) + " GiB RAM");
  if (i.power) {
    std::string cards;
    for (const PowerCard &card : i.power->cards)
      cards += (cards.empty() ? "" : ", ") + card.name + (card.cap_watts > 0 ? " (limit " + format_number(card.cap_watts) + " W)" : "");
    html += term("power", cards + "; idle " + format_number(i.power->idle_watts) + " W before the run");
  }
  for (const Label &note : i.notes) html += term(note.first, note.second);
  for (const Label &var : i.server_env) html += term(var.first, var.second);
  return html + "</dl></section>";
}

std::string nav_html(const Run &run, bool with_anatomy) {
  std::string html = "<nav aria-label=\"sections\"><a class=\"home\" href=\"#top\" title=\"to the top\">\xE2\x97\x86</a><div class=\"links\"><a href=\"#run\">run</a>";
  if (with_anatomy) html += "<a href=\"#request\">request</a>";
  for (const ScenarioResult &s : run.scenarios) {
    std::string mark;
    if (s.errors) mark = " class=\"err\" title=\"" + std::to_string(s.errors) + " failed\"";
    else if (!s.misses.empty()) mark = " class=\"err\" title=\"" + escaped(misses_heading(s)) + "\"";
    html += "<a href=\"#" + escaped(s.id) + "\"" + mark + ">" + escaped(s.id) + "</a>";
  }
  return html + "<a href=\"#setup\">recorded</a></div></nav>";
}

// ---- the run's facts: labelled rows of chips under the title (what ran, on what, with which settings) ----

// The server's settings, from the best source the run has:
//   1. the launch the server reported (xenithium-server --report-launch): its XEN_ variables and --flag value pairs, exact;
//   2. the launcher's notes (a Bench-tab run carries its launch.env as XEN_* notes);
//   3. fallback, for other engines and runs from before: the server_cmd note's --flag value pairs and XEN_X=value
//      words (split at spaces, quotes dropped; `bash -c '... exec xenithium-server --model ...'` reads the same).
// The launcher's own notes (engine_rev, settings_file) come from the notes either way.
struct Settings {
  const std::vector<Label> *notes = nullptr;
  std::vector<Label> env;  // XEN_ variables: reported, else parsed from server_cmd
  std::vector<Label> flags;  // --flag, value ("1" for a flag without a value): reported, else parsed from server_cmd
  bool reported = false;  // env and flags are the server's own report

  std::string note(std::string_view key) const {
    for (const auto &[k, v] : *notes)
      if (k == key) return v;
    return {};
  }
  // The XEN_ key's value, else its flag's; "" = not known.
  std::string get(std::string_view key, std::string_view flag) const {
    for (const auto &[k, v] : env)
      if (k == key) return v;
    for (const auto &[f, v] : flags)
      if (f == flag) return v;
    return note(key);
  }
};

Settings settings_of(const RunInfo &i) {
  Settings s{&i.notes, {}, {}, !i.server_args.empty() || !i.server_env.empty()};
  std::vector<std::string> words;
  if (s.reported) {
    words = i.server_args;
    s.env = i.server_env;
  } else {
    words.emplace_back();
    for (const char c : s.note("server_cmd")) {
      if (c == '\'' || c == '"') continue;
      if (std::isspace(static_cast<unsigned char>(c))) {
        if (!words.back().empty()) words.emplace_back();
      } else
        words.back() += c;
    }
  }
  for (size_t k = 0; k < words.size(); ++k)
    if (const size_t eq = words[k].find('='); !s.reported && words[k].rfind("XEN_", 0) == 0 && eq != std::string::npos)
      s.env.push_back({words[k].substr(0, eq), words[k].substr(eq + 1)});
    else if (words[k].size() > 2 && words[k].rfind("--", 0) == 0) {
      const bool valued = k + 1 < words.size() && !words[k + 1].empty() && words[k + 1].rfind("--", 0) != 0;
      s.flags.push_back({words[k], valued ? words[k + 1] : "1"});
    }
  return s;
}

std::string base_name(const std::string &path) {
  std::string p = path;
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  const size_t slash = p.rfind('/');
  return slash == std::string::npos ? p : p.substr(slash + 1);
}

// What the server's counters said over the whole run: each scenario's notes carry its share ("server: 4.43 tokens per
// speculative step over 7916 steps", "server: its prefix cache served 0 of 8339 prompt tokens"; counters.cpp).
struct ServerTotals {
  double spec_tokens = 0, spec_steps = 0;
  double cache_hit = 0, cache_asked = 0;
};

ServerTotals server_totals(const Run &run) {
  ServerTotals t;
  for (const ScenarioResult &scenario : run.scenarios)
    for (const std::string &note : unique_notes(scenario)) {
      double a = 0, b = 0;
      if (std::sscanf(note.c_str(), "server: %lf tokens per speculative step over %lf steps", &a, &b) == 2) t.spec_tokens += a * b, t.spec_steps += b;
      else if (std::sscanf(note.c_str(), "server: its prefix cache served %lf of %lf prompt tokens", &a, &b) == 2) t.cache_hit += a, t.cache_asked += b;
    }
  return t;
}

// ---- run this yourself: the three commands that repeat the run (build, server, benchmark) ----

// A word as a POSIX shell reads it back: as it is when it holds only plain characters, else in single quotes.
std::string shell_word(const std::string &w) {
  const bool plain = !w.empty() && std::all_of(w.begin(), w.end(), [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || std::strchr("_-./:=,@%+", c); });
  if (plain) return w;
  std::string out = "'";
  for (const char c : w) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return out + "'";
}

// A command over several lines: the program, then each --flag with its value on a line of its own.
std::string command_lines(const std::vector<std::string> &env, const std::string &program, const std::vector<std::string> &args) {
  std::string out;
  for (const std::string &e : env) out += e + " \\\n";
  out += program;
  for (size_t a = 0; a < args.size(); ++a) {
    const bool flag = args[a].rfind("--", 0) == 0;
    out += flag ? " \\\n  " : " ";
    out += shell_word(args[a]);
  }
  return out;
}

// The benchmark's command: as typed (recorded since the run file kept it), else rebuilt from the run's settings.
// Flags of the recorded command that only make sense on the machine that ran it: --out (its folder) and --api-key (its
// key, hidden anyway); `dropped` says which were there.
std::vector<std::string> portable_args(const std::vector<std::string> &args, std::vector<std::string> &dropped) {
  std::vector<std::string> out;
  for (size_t a = 0; a < args.size(); ++a) {
    bool drop = false;
    for (const char *flag : {"--out", "--api-key"})
      if (args[a] == flag || args[a].rfind(std::string(flag) + "=", 0) == 0) {
        drop = true;
        if (std::find(dropped.begin(), dropped.end(), flag) == dropped.end()) dropped.emplace_back(flag);
        if (args[a] == flag && a + 1 < args.size()) ++a;  // its value
      }
    if (!drop) out.push_back(args[a]);
  }
  return out;
}

std::string bench_command(const RunInfo &i, bool &rebuilt, std::vector<std::string> &dropped) {
  std::vector<std::string> args = portable_args(i.command, dropped);
  rebuilt = i.command.empty();
  if (rebuilt) {
    char number[32];
    auto add = [&](const char *flag, const std::string &value) { args.push_back(flag), args.push_back(value); };
    add("--url", i.endpoint);
    if (i.suite != "custom") add("--suite", i.suite);
    add("--mode", i.mode);
    add("--seed", std::to_string(i.seed));
    std::snprintf(number, sizeof number, "%g", i.sampling.temperature), add("--temperature", number);
    std::snprintf(number, sizeof number, "%g", i.sampling.top_p), add("--top-p", number);
    add("--top-k", std::to_string(i.sampling.top_k));
    if (!i.thinking.empty() && i.thinking != "default") add("--thinking", i.thinking);
    add("--reply-length", i.reply_length);
    add("--corpus", i.corpus);
  }
  return command_lines({}, "build/xenithium-bench run", args);
}

// Where the engine's code is published (the clone step of "Run this yourself").
constexpr std::string_view kRepoUrl = "https://github.com/Niluthon/xenithium";

// "Run this yourself": the run repeated from a fresh machine, for someone who has never run the engine. Numbered steps,
// each a short why, its commands (one copy button each) and what to look for: what is needed, the code at the run's
// revision, the dependencies, the build, the model files, the server, the benchmark, the result.
std::string rerun_html(const Run &run, const Settings &s) {
  const RunInfo &i = run.info;
  const std::string rev = !i.engine_build.empty() ? i.engine_build : s.note("engine_rev");
  const bool xenithium = i.engine == "xenithium" || (i.engine.empty() && (!rev.empty() || !i.devices.empty() || s.reported));
  const std::string engine = i.engine.empty() ? std::string("the engine") : i.engine;
  std::string steps;
  auto step = [&](const std::string &title, const std::string &about, const std::string &cmd, const std::string &after) {
    steps += "<li><p><b class=\"step\">" + title + ".</b>" + (about.empty() ? "" : " " + about) + "</p>";
    if (!cmd.empty()) steps += "<pre class=\"cmd\">" + escaped(cmd) + "</pre>";
    if (!after.empty()) steps += "<p class=\"hint\">" + after + "</p>";
    steps += "</li>";
  };
  auto code = [](const std::string &text) { return "<code>" + escaped(text) + "</code>"; };

  // what is needed: the machine, the cards
  const std::string gpus = gpus_text(i.devices);
  std::string intro = "Repeat this run on your own machine, from nothing. Run the commands in order, each in a terminal; every box has a copy "
                      "button. You need Linux with the <code>xe</code> GPU driver (tested on Ubuntu 26.04), <code>git</code>, and ";
  intro += gpus.empty() ? std::string("Intel Arc Pro B-series cards (this run did not record which).")
                        : "the same cards: <b>" + escaped(gpus) + "</b>" + (i.tensor_parallel > 1 ? " (TP " + std::to_string(i.tensor_parallel) + ")" : "") +
                              ". Cards with less memory may not fit its settings.";

  // 1. the code at the run's revision
  const bool dirty = rev.size() > 6 && rev.compare(rev.size() - 6, 6, "-dirty") == 0;
  const std::string commit = dirty ? rev.substr(0, rev.size() - 6) : rev;
  std::string clone = "git clone " + std::string(kRepoUrl) + "\ncd xenithium";
  if (!commit.empty() && commit != "unknown") clone += "\ngit checkout " + shell_word(commit);
  std::string clone_after;
  if (commit.empty() || commit == "unknown") clone_after = "This run did not record its revision: you get the newest code, which may give other numbers.";
  else {
    clone_after = "The checkout puts the code back to the revision this run was built from (" + code(commit) +
                  "). If git says it does not know it, the run was built from code that is not published: skip the checkout and use the newest code.";
    if (dirty) clone_after += " <b>Its build also had changes that were never committed, so even this revision may not match it exactly.</b>";
  }
  step("Get the code", xenithium ? "The engine's code, at the revision this run used." : "The benchmark tool's code.", clone, clone_after);

  // 2. dependencies, and a look at the cards
  step("Install what it needs", "Once per machine. It asks for your password: it installs packages with apt.",
       "scripts/install-deps.sh\nsource scripts/oneapi-env.sh\nsycl-ls",
       "<code>install-deps.sh</code> installs Intel's oneAPI compiler, oneDNN and the GPU driver. <code>sycl-ls</code> should then list each of "
       "your cards as a <code>level_zero:gpu</code>; if none shows up, the driver is not working yet. A new terminal needs "
       "<code>source scripts/oneapi-env.sh</code> again before the commands below.");

  // 3. the build
  step("Build it", xenithium ? "The server and the benchmark tool; this takes several minutes." : "The benchmark tool.",
       xenithium ? "scripts/build-host.sh xenithium-server xenithium-bench" : "scripts/build-host.sh xenithium-bench",
       "The programs land in <code>build/</code>.");

  // 4. the model files the server read
  std::vector<std::pair<std::string, std::string>> files;  // (what, path)
  for (const auto &[what, key, flag] : {std::tuple{"the model", "XEN_MODEL", "--model"}, {"the drafter", "XEN_DRAFTER_MODEL", "--drafter-model"},
                                        {"the MTP drafter", "XEN_MTP", "--mtp"}, {"the trained-QSA sidecar", "XEN_TRAINED_QSA_DELTA_DIR", ""}})
    if (const std::string path = s.get(key, flag); !path.empty()) files.push_back({what, path});
  if (files.empty() && !i.model.empty()) {
    step("Get the model", "This run served " + code(i.model) + ".", "",
         "The run did not record which checkpoint that is. <code>README.md</code> (Supported models) lists the checkpoints the engine runs and where "
         "to download them.");
  } else if (!files.empty()) {
    std::string list, check = "ls -d";
    for (const auto &[what, path] : files) {
      list += "<li>" + std::string(what) + ": " + code(base_name(path)) + " (this run: " + code(path) + ")</li>";
      check += " " + shell_word(path);
    }
    steps += "<li><p><b class=\"step\">Get the model files.</b> This run read these:</p><ul class=\"files\">" + list +
             "</ul><p class=\"hint\">The <a href=\"" + std::string(kRepoUrl) + "#supported-models\">Supported models</a> table in <code>README.md</code> "
             "links each checkpoint the engine runs on Hugging Face. A name that is not there was probably converted or renamed on the machine that ran "
             "this: ask whoever shared this page how they made it. Put each one at the path above (the server command below reads those), or change the "
             "paths in that command to where you put them. This shows whether they are in place:</p><pre class=\"cmd\">" + escaped(check) + "</pre></li>";
  }

  // 5. the server
  const std::string oneapi = "source scripts/oneapi-env.sh\n";  // the server needs oneAPI's libraries in its terminal
  const std::string wait = "Leave it running in this terminal. Loading takes a few minutes; it is ready when it prints " +
                           code("server: serving on http://...") + ". Stop it with Ctrl+C when you are done.";
  std::vector<std::string> launcher_env;  // a Bench-tab run (settings_file names its launch.env): XEN_* notes start-server.sh reads
  if (const std::string file = s.note("settings_file"); !file.empty() && file[0] != '(')
    for (const auto &[k, v] : i.notes)
      if (k.rfind("XEN_", 0) == 0) launcher_env.push_back(k + "=" + shell_word(v));
  if (s.reported) {
    std::vector<std::string> env;
    std::vector<std::string> hidden;  // a key or token the server would not show: left out, the reader sets their own
    for (const auto &[k, v] : i.server_env)
      if (v == "(hidden)") hidden.push_back(k);
      else env.push_back(k + "=" + shell_word(v));
    std::string about = wait;
    if (!hidden.empty()) {
      std::string names;
      for (const std::string &h : hidden) names += (names.empty() ? "" : ", ") + code(h);
      about += " Left out: " + names + ", which held a key or token (hidden by the server): set your own if your setup needs it.";
    }
    // any (hidden) left in the command: a flag's value (--key V, --key=V) or part of one (a URL's password)
    auto has_hidden = [](const std::string &text) { return text.find("(hidden)") != std::string::npos; };
    const bool hidden_left = std::any_of(i.server_args.begin(), i.server_args.end(), has_hidden) ||
                             std::any_of(env.begin(), env.end(), has_hidden);
    if (hidden_left) about += " Where the command shows <code>(hidden)</code>, it held a key, token or password: put your own there.";
    step("Start the server", "Exactly as this run's was started: the server reported its own command line.",
         oneapi + command_lines(env, "build/xenithium-server", i.server_args), about);
  } else if (!launcher_env.empty()) {
    step("Start the server", "With the launcher settings this run used.", command_lines(launcher_env, "scripts/start-server.sh", {}), wait);
  } else if (const std::string cmd = s.note("server_cmd"); !cmd.empty()) {
    step("Start the server", "With the command line recorded with this run.", oneapi + cmd, wait);
  } else {
    step("Start the server", "<b>This run did not record how its server was started</b>, so your numbers may differ.",
         xenithium ? oneapi + (files.empty() ? std::string("build/xenithium-server --model /path/to/the/model") : "build/xenithium-server --model " + shell_word(files[0].second)) : "",
         (xenithium ? wait + " " : "Start " + escaped(engine) + " as you normally do, serving an OpenAI-compatible API. ") +
             "Next time, start xenithium-server with <code>--report-launch</code> and the benchmark records the exact command.");
  }

  // 6. the benchmark
  bool rebuilt = false;
  std::vector<std::string> dropped;
  const std::string bench = bench_command(i, rebuilt, dropped);
  std::string bench_after = "Run it in a second terminal, in the <code>xenithium</code> folder, while the server is running";
  if (i.seconds > 0) bench_after += ": this run took " + duration_text(i.seconds);
  bench_after += ". It talks to <code>http://127.0.0.1:8000/v1</code> unless <code>--url</code> says otherwise: change it if your server listens elsewhere.";
  if (rebuilt) bench_after += " This run did not record its command, so this one is rebuilt from its settings.";
  if (std::find(dropped.begin(), dropped.end(), "--out") != dropped.end())
    bench_after += " Its <code>--out</code> folder was on the machine that ran it, so it is left out: your results go to <code>~/.xenithium/bench/runs</code>.";
  if (std::find(dropped.begin(), dropped.end(), "--api-key") != dropped.end())
    bench_after += " It used an API key: add <code>--api-key YOUR_KEY</code> if your server needs one.";
  step("Run the benchmark", "With the same settings as this run.", bench, bench_after);

  // 7. the result
  step("Look at your result", "When the benchmark finishes it prints <code>run DIR</code>: open <code>DIR/report.html</code> in a browser, a page like this one.",
       "build/xenithium-bench compare THIS_RUN YOUR_RUN",
       "To see your numbers against this run, metric by metric: save this run's <code>results.json</code> (ask whoever shared this page) and give "
       "its folder as <code>THIS_RUN</code>, your run's as <code>YOUR_RUN</code>.");

  return "<details class=\"launch rerun\"><summary>Run this yourself: step by step, from a fresh machine</summary><p class=\"intro\">" + intro + "</p><ol>" +
         steps + "</ol></details>";
}

// One labelled row; a row with no known fact is left out.
void fact_row(std::string &html, std::string_view label, const std::vector<std::string> &facts) {
  std::string items;
  for (const std::string &f : facts)
    if (!f.empty()) items += "<li>" + escaped(f) + "</li>";
  if (!items.empty()) html += "<dt>" + escaped(label) + "</dt><dd><ul class=\"chips\">" + items + "</ul></dd>";
}

std::string facts_html(const Run &run) {
  const RunInfo &i = run.info;
  const Settings s = settings_of(i);
  const ServerTotals totals = server_totals(run);
  const std::string dot(kDot);
  auto num = [](double v) { return with_thousands(static_cast<int64_t>(std::llround(v))); };
  std::string rows;

  // engine: its name and build (the server's own report, else the launcher's note), how it was started, and the API
  // extensions the endpoint took (what the benchmark could ask for: ignore_eos = replies run to a fixed length)
  const std::string rev = !i.engine_build.empty() ? i.engine_build : s.note("engine_rev"), settings_file = s.note("settings_file");
  std::string extensions;
  for (const auto &[name, taken] : {std::pair{"ignore_eos", i.capabilities.ignore_eos}, {"top_k", i.capabilities.top_k}, {"thinking switch", i.capabilities.thinking_switch},
                                    {"usage in stream", i.capabilities.stream_usage}, {"/metrics", i.capabilities.metrics}})
    if (taken) extensions += (extensions.empty() ? "API: " : dot) + name;
  const std::string engine = i.engine.empty() ? (rev.empty() ? "" : "xenithium") : i.engine;
  fact_row(rows, "engine",
           {engine.empty() ? "" : engine + (rev.empty() ? "" : " " + rev), i.engine_oneapi.empty() ? "" : "oneAPI " + i.engine_oneapi,
            settings_file.empty() ? "" : settings_file[0] == '(' ? "started outside the launcher" : "launcher settings " + settings_file, extensions});

  // model: the served name's checkpoint, the context, the chat template and image input where set
  const std::string model_dir = s.get("XEN_MODEL", "--model"), tmpl = s.get("XEN_CHAT_TEMPLATE", "--chat-template"), vision = s.get("XEN_VISION", "--vision");
  fact_row(rows, "model",
           {model_dir.empty() ? i.model : base_name(model_dir), i.context_tokens > 0 ? with_thousands(i.context_tokens) + "-token context" : "",
            tmpl.empty() || tmpl == "auto" ? "" : "chat template " + tmpl, vision == "1" ? "image input on" : ""});

  // KV cache: its type, the pool, the prefix cache
  const std::string kv = s.get("XEN_KV", "--kv"), pool_set = s.get("XEN_KV_POOL_TOKENS", "--kv-pool-tokens");
  const std::string prefix = s.get("XEN_PREFIX_CACHE", "--prefix-cache"), prefix_gib = s.get("XEN_PREFIX_CACHE_GIB", "--prefix-cache-gib");
  const std::string qsa = s.get("XEN_TRAINED_QSA_DELTA_DIR", "");  // the trained sparse-attention sidecar (docs/trained-k8v4-qsa.md)
  fact_row(rows, "KV cache",
           {kv.empty() ? "" : kv + " KV", i.kv_pool_tokens > 0 ? with_thousands(i.kv_pool_tokens) + "-token pool" + (pool_set == "auto" ? " (auto)" : "") : "",
            prefix.empty() ? "" : prefix == "0" ? "prefix cache off" : "prefix cache on" + (prefix_gib.empty() ? "" : dot + prefix_gib + " GiB host"),
            qsa.empty() ? "" : "trained QSA " + base_name(qsa)});

  // parallel: the TP width (and the wire between the cards, or the one card), the slots
  const std::string tp_set = s.get("XEN_TP", "--tp"), slots = s.get("XEN_SLOTS", "--slots");
  const int tp = i.tensor_parallel > 0 ? i.tensor_parallel : std::atoi(tp_set.c_str());
  const std::string wire = s.get("XEN_WIRE", "--wire"), card = s.get("XEN_CARD", "--card");
  fact_row(rows, "parallel",
           {tp > 0 ? "TP " + std::to_string(tp) + (tp > 1 && !wire.empty() ? dot + "wire " + wire : "") : "", tp == 1 && !card.empty() ? "card " + card : "",
            slots.empty() ? "" : slots + " slots"});

  // speculation: the drafter (the kind the server reports, else what the settings say), its checkpoint, and what it gave
  const std::string drafter = s.get("XEN_DRAFTER", "--drafter"), drafter_model = s.get("XEN_DRAFTER_MODEL", "--drafter-model");
  const std::string tree = s.get("XEN_DRAFTER_TREE", "--drafter-tree"), mtp = s.get("XEN_MTP_DEPTH", "--mtp-depth");
  const std::string &kind = i.engine_drafter;  // "none", "mtp", "dflash2", "dspine", "...+mtp"
  const bool off = kind == "none" || (kind.empty() && drafter == "0");
  const bool block = kind.empty() ? !off && !drafter_model.empty() : kind.rfind("dflash2", 0) == 0 || kind.rfind("dspine", 0) == 0;
  std::string drafting;
  if (off) drafting = "drafter off";
  else if (kind.rfind("dspine", 0) == 0) drafting = "DSpine";
  else if (kind.rfind("dflash2", 0) == 0) drafting = "DFlash 2";
  else if (kind == "mtp" || (kind.empty() && !drafter.empty() && !block)) drafting = "MTP" + (mtp.empty() ? std::string() : " depth " + mtp);
  if (kind.ends_with("+mtp")) drafting += " + MTP";
  char per_step[96] = "";
  if (totals.spec_steps > 0) std::snprintf(per_step, sizeof per_step, "%.2f tokens per step this run", totals.spec_tokens / totals.spec_steps);
  fact_row(rows, "speculation",
           {drafting, block && !drafter_model.empty() ? "drafter " + base_name(drafter_model) : "", !off && block && !tree.empty() && tree != "0" ? "tree " + tree : "", per_step});

  // hardware: the cards, their power limit and draw, the clock, the machine
  std::string cap, draw, clock;
  if (i.power) {
    const RunPower &p = *i.power;
    if (!p.cards.empty() && p.cards.front().cap_watts > 0) {
      const bool same = std::all_of(p.cards.begin(), p.cards.end(), [&](const PowerCard &c) { return c.cap_watts == p.cards.front().cap_watts; });
      cap = format_number(p.cards.front().cap_watts) + " W cap" + (same && p.cards.size() > 1 ? " each" : "");
    }
    double sum = 0, peak = 0, mhz_sum = 0, mhz_low = 0;
    int n = 0, n_mhz = 0;
    for (const PowerPoint &pt : p.trace) {
      double w = 0;
      for (const double card_w : pt.watts) w += card_w;
      sum += w, peak = std::max(peak, w), ++n;
      if (pt.mhz > 0) mhz_sum += pt.mhz, mhz_low = n_mhz ? std::min(mhz_low, pt.mhz) : pt.mhz, ++n_mhz;
    }
    if (n) draw = num(sum / n) + " W mean" + dot + num(peak) + " W peak" + (p.cards.size() > 1 ? " (all cards)" : "") + dot + "idle " + num(p.idle_watts) + " W";
    if (n_mhz) clock = "clock " + num(mhz_sum / n_mhz) + " MHz mean" + dot + num(mhz_low) + " lowest";
  }
  // the cards' heat and what held their clock back (the power sampler's sensors; none when it could not read them). The
  // driver's reasons in two kinds: the power limit (pl1, pl2, pl4, vr_tdc: a card at its cap, expected under load) and
  // heat (thermal, prochot, vr_thermalert, ratl and anything else: the cooling cannot keep up), each with its share of
  // the run's points.
  std::string heat, power_limited, hot_limited;
  if (i.power) {
    const RunPower &p = *i.power;
    const bool temps = std::any_of(p.cards.begin(), p.cards.end(), [](const PowerCard &c) { return c.reads_temp; });
    const bool limits = std::any_of(p.cards.begin(), p.cards.end(), [](const PowerCard &c) { return c.reads_throttle; });
    double sum = 0, peak = 0;
    int n = 0, by_power = 0, by_heat = 0;
    std::string power_reasons, heat_reasons;
    auto add = [](std::string &list, const std::string &word) {
      if ((", " + list + ",").find(" " + word + ",") == std::string::npos) list += (list.empty() ? "" : ", ") + word;
    };
    for (const PowerPoint &pt : p.trace) {
      if (pt.temp_c > 0) sum += pt.temp_c, peak = std::max(peak, pt.temp_c), ++n;
      bool power_point = false, heat_point = false;
      std::string word;
      for (size_t c = 0; c <= pt.throttle.size(); ++c)
        if (c < pt.throttle.size() && pt.throttle[c] != ' ') word += pt.throttle[c];
        else {
          if (word.empty()) continue;
          const bool power_word = word == "pl1" || word == "pl2" || word == "pl4" || word == "vr_tdc";
          add(power_word ? power_reasons : heat_reasons, word);
          (power_word ? power_point : heat_point) = true;
          word.clear();
        }
      by_power += power_point, by_heat += heat_point;
    }
    if (temps && n) heat = "GPU " + num(sum / n) + " \xC2\xB0" "C mean" + dot + num(peak) + " \xC2\xB0" "C peak";
    if (limits && !p.trace.empty()) {
      auto share = [&](int k) {
        char text[32];
        std::snprintf(text, sizeof text, "%.0f%%", 100.0 * k / static_cast<double>(p.trace.size()));
        return std::string(text);
      };
      if (by_power) power_limited = "at its power limit " + share(by_power) + " of the run (" + power_reasons + ")";
      hot_limited = by_heat ? "throttled by heat " + share(by_heat) + " of the run (" + heat_reasons + ")" : by_power ? "never throttled by heat" : "never throttled";
    }
    if (!p.cards.empty() && p.cards.front().max_mhz > 0 && !clock.empty()) clock += dot + "max " + num(p.cards.front().max_mhz);
  }
  const std::string driver = i.devices.empty() ? std::string() : i.devices.front().driver;
  fact_row(rows, "hardware", {gpus_text(i.devices), driver.empty() ? "" : "driver " + driver, cap, draw, clock, heat, power_limited, hot_limited});

  // machine: the server's (its own report; else the bench's when it measured the cards here, so the same machine)
  const util::Machine &m = !i.machine.kernel.empty() ? i.machine : i.power ? i.bench_machine : util::Machine{};
  fact_row(rows, "machine",
           {m.os, m.kernel.empty() ? "" : "kernel " + m.kernel, m.cpu.empty() ? "" : m.cpu + (m.cpu_threads > 0 ? " (" + std::to_string(m.cpu_threads) + " threads)" : ""),
            m.memory_gib > 0 ? std::to_string(m.memory_gib) + " GiB RAM" : "", i.host.empty() ? "" : "bench ran on " + i.host});

  // sampling: what the benchmark's requests asked for
  char sampling[96];
  std::snprintf(sampling, sizeof sampling, "temperature %g%stop_p %g%stop_k %d", i.sampling.temperature, dot.c_str(), i.sampling.top_p, dot.c_str(), i.sampling.top_k);
  fact_row(rows, "sampling", {sampling, "thinking " + (i.thinking.empty() ? std::string("default") : i.thinking), "seed " + std::to_string(i.seed)});

  // during the run: the server's prefix cache, the corpus (the requests and replies are chips above)
  std::string cache;
  if (totals.cache_asked > 0) {
    char pct[32];
    std::snprintf(pct, sizeof pct, " (%.0f%%)", 100 * totals.cache_hit / totals.cache_asked);
    cache = num(totals.cache_hit) + " of " + num(totals.cache_asked) + " prompt tokens from the server's cache" + pct;
  }
  fact_row(rows, "during run", {cache, "corpus " + i.corpus + " v" + std::to_string(i.corpus_version) + dot + i.corpus_hash});

  return "<dl class=\"facts\">" + rows + "</dl>" + rerun_html(run, s);
}

// The masthead: the brand (and the buttons the script adds), the model, and the run's facts as chips.
std::string header_html(const Run &run) {
  const RunInfo &i = run.info;
  const int failed = failed_requests(run);
  std::string chips = "<li>" + escaped(timestamp_text(i.timestamp)) + "</li>";
  chips += "<li>mode <b>" + escaped(i.mode) + "</b></li><li>suite <b>" + escaped(i.suite) + "</b></li><li><b>" +
                      duration_text(i.seconds) + "</b></li>";
  chips += "<li><b>" + with_thousands(all_requests(run)) + "</b> requests</li>";
  if (failed) chips += "<li class=\"bad\"><b>" + std::to_string(failed) + "</b> failed</li>";
  chips += "<li>replies <b>" + escaped(i.reply_length) + "</b></li><li>" + escaped(i.endpoint) + "</li>";
  return "<header id=\"top\"><div class=\"top\"><div class=\"brand\"><i>\xE2\x97\x86</i>xenithium <b>bench</b></div><div class=\"tools\"></div></div><h1>" + escaped(i.model) +
         "</h1><ul class=\"chips\">" + chips + "</ul>" + facts_html(run) + "</header>";
}

// Writes the page's own size where kSizeToken stands. The number's width can change the size, so settle it.
std::string with_own_size(std::string page) {
  const size_t at = page.find(kSizeToken);
  if (at == std::string::npos) return page;
  std::string text = "0";
  for (int pass = 0; pass < 3; ++pass) {
    const double bytes = static_cast<double>(page.size() - kSizeToken.size() + text.size());
    const std::string settled = format_number(bytes / kBytesPerKilobyte);
    if (settled == text) break;
    text = settled;
  }
  return page.replace(at, kSizeToken.size(), text);
}

}  // namespace

std::string render_html(const Run &run) {
  const RunInfo &i = run.info;
  const std::string anatomy = anatomy_html(run);
  std::string page = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>xenithium bench" +
                     std::string(kDot) + escaped(i.model) + std::string(kDot) + escaped(i.timestamp) + "</title><style>";
  page.append(page_style()).append("</style></head><body>");
  page += header_html(run) + nav_html(run, !anatomy.empty()) + "<main>" + figures_html(figures_of(run)) + checks_html(run);
  page += "<p class=\"hint\" hidden>Point at a chart for its numbers \xC2\xB7 click a legend entry to show or hide its series \xC2\xB7 click a column head to sort a table</p>";
  page += timeline_html(run) + anatomy;
  for (const ScenarioResult &scenario : run.scenarios) page += scenario_html(scenario);
  page += setup_html(i) + "</main><footer><span>\xE2\x97\x86 xenithium-bench" + std::string(kDot) + "results.json holds every sample behind these charts</span><span>one file, ";
  page.append(kSizeToken).append(" KB").append(kDot).append("no fonts, no network: it opens offline</span></footer><script>").append(page_script()).append("</script></body></html>\n");
  return with_own_size(std::move(page));
}

}  // namespace xenithium:bench
