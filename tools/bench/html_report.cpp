#include "html_report.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
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

// One way to weigh the decode figure (a mix row of the decode scenario): its name, value and the figure's line.
struct MixOption {
  std::string name;  // "benchmark" (the row "benchmark mix")
  std::string value, detail;
};

struct Figure {
  std::string label, value, unit, detail;
  std::string href;  // the scenario the figure comes from
  std::string alert;  // failures behind the figure ("1 wrong answer"): a line of its own in the failure colour
  std::vector<MixOption> mixes = {};  // the decode figure's weightings the reader can switch between (two or more)
};

// "benchmark mix" -> "benchmark"
std::string mix_name(const std::string &row_label) { return row_label.ends_with(" mix") ? row_label.substr(0, row_label.size() - 4) : row_label; }

// The decode figure's line for one mix: "one stream, the benchmark mix of 8 kinds of output".
std::string mix_detail(const std::string &row_label, size_t kinds) {
  const std::string mix = row_label.empty() || row_label == "weighted" ? "weighted over " : "the " + row_label + " of ";
  return "one stream, " + mix + std::to_string(kinds) + " kinds of output";
}

// The anchor of a scenario's misses panel.
std::string misses_anchor(const std::string &scenario_id) { return scenario_id + "-misses"; }

std::vector<Figure> figures_of(const Run &run) {
  const Headline h = headline(run);
  std::vector<Figure> figures;
  if (h.decode_tps > 0) {
    const size_t kinds = h.decode_by_category.size();
    Figure decode{"Decode", format_number(h.decode_tps), "tok/s", mix_detail(h.decode_mix, kinds), "#decode", {}};
    if (h.decode_mixes.size() > 1) {
      decode.href = "#decode-mix";
      for (const auto &[label, tps] : h.decode_mixes) decode.mixes.push_back({mix_name(label), format_number(tps), mix_detail(label, kinds)});
    }
    figures.push_back(std::move(decode));
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

// A switch between the decode figure's weightings: one button per mix, the first pressed. The page's script shows it
// (hidden without the script: every value is then in the decode table) and keeps the reader's pick for the next report.
std::string mix_switch_html(const std::vector<MixOption> &mixes) {
  std::string html = "<div class=\"mixsw\" role=\"group\" aria-label=\"weigh the decode figure\" hidden>";
  for (size_t k = 0; k < mixes.size(); ++k)
    html += "<button type=\"button\" data-mix=\"" + escaped(mixes[k].name) + "\" data-v=\"" + escaped(mixes[k].value) + "\" data-d=\"" + escaped(mixes[k].detail) + "\" aria-pressed=\"" +
            (k == 0 ? "true" : "false") + "\">" + escaped(mixes[k].name) + "</button>";
  return html + "</div>";
}

std::string figures_html(const std::vector<Figure> &figures) {
  if (figures.empty()) return {};
  std::string html = "<div class=\"figs\">";
  for (const Figure &f : figures)
    if (!f.mixes.empty())  // a card, not a link: it holds the switch (a button may not sit inside a link)
      html += "<div class=\"fig mixfig\"><div class=\"k\">" + escaped(f.label) + "</div><div class=\"n\"><span class=\"mv\">" + escaped(f.value) + "</span><small>" + escaped(f.unit) +
              "</small></div><div class=\"d\"><span class=\"md\">" + escaped(f.detail) + "</span> \xC2\xB7 <a href=\"" + escaped(f.href) + "\">how it is weighed</a></div>" +
              mix_switch_html(f.mixes) + "</div>";
    else
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

// Under the decode figure's switch: what each weighting is - its shares per category, how it averages, its figure in
// this run - and when to read which. Only for a run that reports two or more mixes (runs before the switch had one).
std::string mix_panel_html(const ScenarioResult &decode) {
  std::vector<MixOption> options;
  std::vector<const WeightMix *> defs;
  for (const Row &row : decode.rows) {
    const std::string *label = label_of(row, "category");
    const Metric *rate = metric_of(row, metric::kDecode);
    if (!label || !label->ends_with(" mix") || !rate || rate->samples.empty()) continue;
    options.push_back({mix_name(*label), format_number(summarize(rate->samples).median), {}});
    const auto def = std::ranges::find(weight_mixes(), std::string_view(options.back().name), &WeightMix::id);
    defs.push_back(def == weight_mixes().end() ? nullptr : &*def);
  }
  if (options.size() < 2) return {};
  std::string html = "<div class=\"mixinfo\" id=\"decode-mix\"><div class=\"mbar\"><b>How the decode figure is weighed</b>" + mix_switch_html(options) +
                     "</div><div class=\"tscroll\"><table><thead><tr><th class=\"l\">mix</th>";
  for (const CategoryWeight &c : category_weights()) html += "<th>" + escaped(c.category) + "</th>";
  html += "<th class=\"l\">average</th><th>this run</th></tr></thead><tbody>";
  for (size_t k = 0; k < options.size(); ++k) {
    html += "<tr data-mix=\"" + escaped(options[k].name) + "\"" + (k == 0 ? " class=\"on\"" : "") + "><td class=\"l\">" + escaped(options[k].name) + "</td>";
    for (const CategoryWeight &c : category_weights()) {
      double share = -1;
      if (defs[k])
        for (const CategoryWeight &w : defs[k]->weights)
          if (w.category == c.category) share = w.weight;
      html += "<td>" + (share < 0 ? std::string("-") : share == 0 ? std::string("0") : format_number(share * 100) + " %") + "</td>";
    }
    const std::string average = !defs[k] ? "as asked" : defs[k]->average == MixAverage::arithmetic ? "plain mean of the rates" : "rate of the mix (harmonic)";
    html += "<td class=\"l\">" + average + "</td><td>" + escaped(options[k].value) + " tok/s</td></tr>";
  }
  html += "</tbody></table></div><ul>"
          "<li>Both figures come from the same per-category medians in the table below; only the shares and the way they are averaged differ.</li>"
          "<li><b>benchmark</b> weighs the categories the way the usual public benchmarks do (code first, chat and maths left out) and takes a plain "
          "weighted mean of their rates. Set it beside other benchmarks' numbers. It reads higher: a fast kind of output (JSON, code) lifts a plain mean "
          "more than a slow one pulls it down.</li>"
          "<li><b>agentic</b> weighs them by what a coding agent actually generated in measured sessions (most of it reasoning) and averages time per "
          "token: the tokens per second an agent gets over its work. A slow kind counts by the time it takes, so it reads lower where reasoning or chat "
          "decode slowly.</li>"
          "<li>Half the tokens at 300 tok/s and half at 100: a plain mean reads 200, the rate they come out at is 150.</li></ul></div>";
  return html;
}

std::string scenario_html(const ScenarioResult &scenario) {
  std::string html = "<section class=\"sc\" id=\"" + escaped(scenario.id) + "\"><h2><span class=\"ttl\">" + escaped(scenario.title) + "</span> <code>" + escaped(scenario.id) + "</code></h2>";
  if (!scenario.about.empty()) html += "<p class=\"about\">" + escaped(scenario.about) + "</p>";
  html += "<p class=\"meta\">" + duration_text(scenario.seconds) + std::string(kDot) + with_thousands(scenario.requests) + " requests";
  if (scenario.errors) html += std::string(kDot) + "<b>" + std::to_string(scenario.errors) + " failed</b>";
  if (scenario.power) html += std::string(kDot) + escaped(power_line(scenario));
  html += "</p>" + misses_html(scenario);
  if (scenario.id == "decode") html += mix_panel_html(scenario);

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

std::string setup_html(const RunInfo &i) {
  char sampling[96];
  std::snprintf(sampling, sizeof(sampling), "temperature %.2f, top-p %.2f, top-k %d, seed %llu + request number", i.sampling.temperature, i.sampling.top_p, i.sampling.top_k,
                static_cast<unsigned long long>(i.seed));
  const Capabilities &caps = i.capabilities;
  std::string extensions;
  for (const auto &[name, taken] : {std::pair{"ignore_eos", caps.ignore_eos}, {"top_k", caps.top_k}, {"thinking switch", caps.thinking_switch},
                                    {"usage in the stream", caps.stream_usage}, {"/metrics", caps.metrics}})
    extensions += std::string(extensions.empty() ? "" : ", ") + name + (taken ? " yes" : " no");

  std::string html = "<section id=\"setup\"><h2><span class=\"ttl\">Setup</span></h2><dl class=\"setup\">";
  html += term("endpoint", i.endpoint) + term("model", i.model);
  html += term("context", i.context_tokens > 0 ? with_thousands(i.context_tokens) + " tokens per request" : "not reported by the endpoint");
  if (i.kv_pool_tokens > 0) html += term("KV pool", with_thousands(i.kv_pool_tokens) + " tokens shared by all requests");
  html += term("method", i.method > 0 || i.mode.empty() ? "version " + std::to_string(i.method) : "mode " + i.mode + ", version " + std::to_string(i.mode_version) + " (a run from before the one method)") +
          term("suite", i.suite) + term("sampling", sampling) + term("thinking", i.thinking.empty() ? "default" : i.thinking) + term("replies", replies_text(i));
  html += term("corpus", (i.corpus.empty() ? std::string() : i.corpus + ", ") + "version " + std::to_string(i.corpus_version) + ", hash " + i.corpus_hash);
  html += term("extensions", extensions);
  if (i.power) {
    std::string cards;
    for (const PowerCard &card : i.power->cards)
      cards += (cards.empty() ? "" : ", ") + card.name + (card.cap_watts > 0 ? " (limit " + format_number(card.cap_watts) + " W)" : "");
    html += term("power", cards + "; idle " + format_number(i.power->idle_watts) + " W before the run");
  }
  html += term("measured from", i.host) + term("run", i.timestamp + "-" + i.name) + "</dl>";
  if (!i.notes.empty()) {
    html += "<details><summary>" + std::to_string(i.notes.size()) + " notes recorded with the run</summary><dl class=\"setup\">";
    for (const Label &note : i.notes) html += term(note.first, note.second);
    html += "</dl></details>";
  }
  return html + "</section>";
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
  return html + "<a href=\"#setup\">setup</a></div></nav>";
}

// The masthead: the brand (and the buttons the script adds), the model, and the run's facts as chips.
std::string header_html(const Run &run) {
  const RunInfo &i = run.info;
  const int failed = failed_requests(run);
  std::string chips = "<li>" + escaped(timestamp_text(i.timestamp)) + "</li>" + (i.mode.empty() ? std::string() : "<li>mode <b>" + escaped(i.mode) + "</b></li>") + "<li>suite <b>" + escaped(i.suite) + "</b></li><li><b>" +
                      duration_text(i.seconds) + "</b></li>";
  chips += "<li><b>" + with_thousands(all_requests(run)) + "</b> requests</li>";
  if (failed) chips += "<li class=\"bad\"><b>" + std::to_string(failed) + "</b> failed</li>";
  chips += "<li>replies <b>" + escaped(i.reply_length) + "</b></li><li>" + escaped(i.endpoint) + "</li>";
  return "<header id=\"top\"><div class=\"top\"><div class=\"brand\"><i>\xE2\x97\x86</i>xenithium <b>bench</b></div><div class=\"tools\"></div></div><h1>" + escaped(i.model) +
         "</h1><ul class=\"chips\">" + chips + "</ul></header>";
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
