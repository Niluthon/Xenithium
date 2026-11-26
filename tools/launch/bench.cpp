// The launcher's Bench tab: the runs scan, the two results.json readers and the `xenithium-bench run`
// launcher.
#include "bench.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string_view>
#include <system_error>

#include "settings.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::launch {

namespace {

// One row per mode: its tag, its menu line, and the `xenithium-bench run` arguments.
struct ModeRow {
  BenchMode mode;
  const char *name;
  const char *title;
  std::vector<std::string> args;
};
const std::vector<ModeRow> &mode_rows() {
  static const std::vector<ModeRow> rows = {
      {BenchMode::Standard, "standard", "Standard (decode, concurrency, prefill to 128K, decode at depth, agent session, quality; ~13 min)", {"--suite", "standard"}},
      {BenchMode::Quick, "quick", "Quick look (every headline number once; ~3 min)", {"--suite", "quick"}},
      {BenchMode::Full, "full", "Full (every scenario, to the server's context; ~62 min)", {"--suite", "full"}},
      {BenchMode::Long, "long", "Long contexts (prefill to 250K, decode at depth, the needle, a long agent session; ~13 min)", {"--suite", "long"}},
      {BenchMode::Load, "load", "Load (concurrency 1-16, arrivals at a rate, a late long prompt, a burst, mixed depths; ~12 min)", {"--suite", "load"}},
      {BenchMode::Agents, "agents", "Agents (1, 2, 4 agents on documents sized from the KV pool, a prompt read beside 3 generating; ~25 min)", {"--suite", "agents"}},
      {BenchMode::Check, "check", "Check the output (answer quality, the needle, repeatability, JSON schema, image input, tool calls; ~7 min)", {"--suite", "check"}},
      {BenchMode::Apps, "apps", "Real-world apps (chat users vs a latency target, RAG under load, editor autocomplete, tool calls, a long reply; ~7 min)", {"--suite", "apps"}},
      {BenchMode::Decode, "decode", "Decode only (single stream, by category)", {"--scenario", "decode"}},
      {BenchMode::Prefill, "prefill", "Prefill only (the prompt depth sweep, 2K to 250K)", {"--scenario", "prefill"}},
      {BenchMode::Concurrency, "concurrency", "Concurrency sweep (1, 2, 4, 8, 16 requests at once)", {"--scenario", "concurrency"}},
      {BenchMode::Streams, "streams", "Concurrency at stream counts you enter (e.g. 1, 8, 24, 48 requests at once)", {"--scenario", "concurrency"}},
  };
  return rows;
}
const ModeRow &mode_row(BenchMode m) {
  for (const ModeRow &row : mode_rows())
    if (row.mode == m) return row;
  return mode_rows().front();
}

}  // namespace

const char *bench_mode_name(BenchMode m) { return mode_row(m).name; }

std::string bench_method_text(const BenchRun &r) {
  if (r.method > 0) return "v" + std::to_string(r.method) + " (open tasks and tasks on given input, natural replies, benchmark and agentic weighting), corpus " + r.corpus;
  struct Gloss {
    std::string_view mode, corpus, about;
  };
  static constexpr Gloss kGlosses[] = {{"easy", "generic", "tasks that repeat or reshape their input, natural-length replies"},
                                       {"medium", "generic", "conventional code and prose"},
                                       {"hard", "engine", "the engine's own code and docs"}};
  for (const Gloss &g : kGlosses)
    if (g.mode == r.mode)
      return "older, mode " + r.mode + " (" + std::string(g.about) + (r.corpus == g.corpus ? ")" : "; corpus " + r.corpus + ")") + ": does not compare with newer runs";
  return "older, mode " + r.mode + ", corpus " + r.corpus;
}
const char *bench_mode_title(BenchMode m) { return mode_row(m).title; }
std::vector<std::string> bench_mode_args(BenchMode m) { return mode_row(m).args; }

std::string bench_stream_levels(std::string_view text, std::string &err) {
  std::set<int> levels;
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == ',' || text[i] == ' ') {
      ++i;
      continue;
    }
    size_t end = i;
    while (end < text.size() && text[end] != ',' && text[end] != ' ') ++end;
    const std::string_view item = text.substr(i, end - i);
    int n = 0;
    const auto [ptr, ec] = std::from_chars(item.data(), item.data() + item.size(), n);
    if (ec != std::errc() || ptr != item.data() + item.size() || n < 1 || n > kMaxBenchStreams) {
      err = "\"" + std::string(item) + "\" is not a stream count from 1 to " + std::to_string(kMaxBenchStreams);
      return "";
    }
    levels.insert(n);
    i = end;
  }
  if (levels.empty()) err = "enter one or more stream counts, e.g. 1, 8, 24, 48";
  else if (levels.size() > static_cast<size_t>(kMaxBenchLevels)) err = "at most " + std::to_string(kMaxBenchLevels) + " stream counts per run";
  if (!err.empty()) return "";
  std::string out;
  for (const int n : levels) out += (out.empty() ? "" : ",") + std::to_string(n);
  return out;
}

std::vector<std::pair<std::string, std::string>> launch_notes(std::string_view env_text) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto &[k, v] : EnvFile::parse(env_text).values())
    if (!v.empty()) out.push_back({k, v});
  return out;
}

BenchSession::~BenchSession() {
  if (child_ > 0) kill(child_, SIGKILL);
}

int BenchSession::total_runs() const {
  int n = 0;
  for (const auto &m : models_) n += static_cast<int>(m.runs.size());
  return n;
}

namespace {

namespace json = xenithium::json;

// "32,000" -> 32000 (a depth or stream count label of a xenithium-bench summary).
int label_number(const std::string &label) {
  int value = 0;
  for (const char c : label)
    if (c >= '0' && c <= '9') value = value * 10 + (c - '0');
  return value;
}

double number_at(const json::Value &obj, std::string_view key) {
  const json::Value *v = obj.get(key);
  return v && v->is_number() ? v->num : 0;
}

// env.model / endpoint / host / timestamp / notes, and config.corpus.
void read_env(const json::Value &v, BenchRun &r) {
  if (const auto *env = v.get("env")) {
    if (const auto *s = env->get_string("model")) r.model = *s;
    if (const auto *s = env->get_string("endpoint")) r.endpoint = *s;
    if (const auto *s = env->get_string("host")) r.host = *s;
    if (const auto *s = env->get_string("timestamp")) r.timestamp = *s;
    if (const auto *notes = env->get_object("notes"))
      for (const auto &m : notes->obj)
        if (m.second.is_string()) r.notes.push_back({m.first, m.second.str});
  }
  if (r.model.empty()) r.model = "(unknown)";
  if (const auto *config = v.get_object("config")) {
    if (const auto *s = config->get_string("corpus")) r.corpus = *s;
    if (const auto *s = config->get_string("mode")) r.mode = *s;
    if (const auto *m = config->get("method"); m && m->is_number()) r.method = static_cast<int>(m->num);
  }
  if (r.method == 0) {  // a run from before the one method
    if (r.corpus.empty()) r.corpus = "hard";  // runs from before the second set read the engine's own code
    if (r.mode.empty()) r.mode = r.corpus == "hard" ? "hard" : "medium";  // before the modes: the corpus said which
  }
  // the sets' names before the modes were merged
  if (r.corpus == "hard") r.corpus = "engine";
  else if (r.corpus == "medium" || r.corpus == "easy") r.corpus = "generic";
}

// The run's `summary.
void read_summary(const json::Value &v, BenchRun &r) {
  const json::Value *summary = v.get_object("summary");
  if (!summary) return;
  r.decode_tps = number_at(*summary, "decode_tps");
  if (const std::string *mix = summary->get_string("decode_mix")) r.decode_mix = *mix;
  if (const json::Value *mixes = summary->get_array("decode_mixes"))
    for (const json::Value &m : mixes->arr)
      if (const std::string *name = m.get_string("mix")) r.decode_mixes.push_back({*name, number_at(m, "decode_tps")});
  r.tokens_per_joule = number_at(*summary, "best_tokens_per_joule");
  if (const json::Value *quality = summary->get("quality_pct"); quality && quality->is_number()) r.quality_pct = quality->num;
  if (const json::Value *misses = summary->get_array("quality_misses"))
    for (const json::Value &m : misses->arr) {
      const std::string *item = m.get_string("item"), *expected = m.get_string("expected"), *got = m.get_string("got"), *why = m.get_string("why");
      r.quality_misses.push_back({item ? *item : "?", expected ? *expected : "", why && !why->empty() ? "(" + *why + ")" : "\"" + (got ? *got : std::string()) + "\""});
    }
  auto each = [&](std::string_view key, auto &&take) {
    if (const json::Value *arr = summary->get_array(key))
      for (const json::Value &item : arr->arr) take(item);
  };
  each("decode_by_category", [&](const json::Value &c) {
    if (const std::string *name = c.get_string("category")) r.per_cat.push_back({*name, number_at(c, "decode_tps")});
  });
  each("prefill", [&](const json::Value &p) {
    if (const std::string *depth = p.get_string("depth")) r.prefill.push_back({label_number(*depth), number_at(p, "prefill_tps")});
  });
  each("deep_decode", [&](const json::Value &d) {
    if (const std::string *depth = d.get_string("depth")) r.deep_decode.push_back({label_number(*depth), number_at(d, "decode_tps")});
  });
  each("concurrency", [&](const json::Value &c) {
    if (const std::string *streams = c.get_string("streams")) r.concurrency.push_back({label_number(*streams), number_at(c, "total_tps")});
  });
}

// Every readable run of one runs directory, appended to its model's group.
void scan_dir(const std::string &runs_dir, std::vector<BenchModel> &out) {
  DIR *d = opendir(runs_dir.c_str());
  if (!d) return;
  struct dirent *e;
  while ((e = readdir(d))) {
    if (e->d_name[0] == '.') continue;
    const std::string dir = join_path(runs_dir, e->d_name);
    std::string text;
    if (!read_text(join_path(dir, "results.json"), text)) continue;  // an empty / aborted run
    json::Value v;
    if (!json::parse(text, v)) continue;

    const std::string *tool = v.get_string("tool");
    if (!tool || *tool != "xenithium-bench") continue;  // not a run of the benchmark tool
    BenchRun r;
    r.dir = e->d_name;
    r.path = dir;
    read_env(v, r);
    read_summary(v, r);

    auto group = std::find_if(out.begin(), out.end(), [&](const BenchModel &m) { return m.name == r.model; });
    if (group == out.end()) {
      out.push_back(BenchModel{r.model, {}});
      group = out.end() - 1;
    }
    group->runs.push_back(std::move(r));
  }
  closedir(d);
}

}  // namespace

void BenchSession::scan(const std::string &runs_dir) {
  std::vector<BenchModel> out;
  scan_dir(runs_dir, out);
  for (auto &m : out)
    std::sort(m.runs.begin(), m.runs.end(), [](const BenchRun &a, const BenchRun &b) { return a.dir > b.dir; });
  std::sort(out.begin(), out.end(), [](const BenchModel &a, const BenchModel &b) {
    return a.runs.empty() || b.runs.empty() ? a.name < b.name : a.runs[0].dir > b.runs[0].dir;
  });
  models_ = std::move(out);
}

bool BenchSession::start(const std::string &bin, const std::string &endpoint, const std::string &model,
                         const std::string &tag, const std::string &log, const std::string &cwd,
                         const std::vector<std::string> &extra) {
  if (run_.active) {
    run_.error = "a benchmark is already running";
    return false;
  }
  std::vector<std::string> argv = {bin, "run", "--url", endpoint, "--model", model, "--name", tag};
  argv.insert(argv.end(), extra.begin(), extra.end());
  const int lf = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (lf < 0) {
    run_.error = "cannot open " + log + ": " + strerror(errno);
    return false;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    run_.error = std::string("fork: ") + strerror(errno);
    close(lf);
    return false;
  }
  if (pid == 0) {
    setsid();  // its own session: survives the launcher, no SIGHUP from the terminal
    dup2(lf, 1);
    dup2(lf, 2);
    const int dn = open("/dev/null", O_RDONLY);
    if (dn >= 0) dup2(dn, 0);
    if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
    std::vector<char *> av;
    for (auto &a : argv) av.push_back(const_cast<char *>(a.c_str()));
    av.push_back(nullptr);
    execvp(av[0], av.data());
    _exit(127);
  }
  close(lf);
  child_ = pid;
  run_.active = true;
  run_.pid = pid;
  run_.name = tag;
  run_.log = log;
  run_.started = Clock::now();
  run_.error.clear();
  return true;
}

void BenchSession::cancel() {
  if (child_ > 0) kill(child_, SIGTERM);
  child_ = -1;
  run_.active = false;
}

bool BenchSession::poll() {
  if (!run_.active || child_ < 0) return false;
  int st = 0;
  const pid_t r = waitpid(child_, &st, WNOHANG);
  if (r == child_) {
    child_ = -1;
    run_.active = false;
    return true;
  }
  return false;
}

}  // namespace xenithium:launch
