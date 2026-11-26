// The launcher's Bench tab: scan the runs directory, read each run's results.json, group the runs by the served
// model and show each run's headline numbers (the weighted decode, the prefill, decode-at-depth and concurrency
// tables). It also launches a new `xenithium-bench run` (tools/bench) against the running server and reports its
// progress.
// A run record is `<runs dir>/<ts>-<name>/results.json` as xenithium-bench writes it: `env` (model, endpoint, host,
// notes) and `summary` (decode_tps, decode_by_category[], prefill[], deep_decode[], concurrency[]). A results.json
// of any other tool is skipped.
#pragma once

#include <chrono>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xenithium::launch {

struct BenchRun {
  std::string dir;  // the run dir name (starts with the timestamp)
  std::string path;  // absolute path to the run dir
  std::string model;  // env.model (the grouping key)
  std::string endpoint;  // env.endpoint
  std::string host;  // env.host
  std::string timestamp;  // env.timestamp
  int method = 0;  // config.method: the measuring method's version; 0 = a run from before it (it had a mode)
  std::string mode;  // such a run's config.mode: "easy", "medium", "hard" (runs from before the modes: from their corpus)
  std::string corpus;  // config.corpus: the text the long prompts were cut from ("engine", "generic", or a directory)
  double decode_tps = 0;  // the decode figure of the categories' mix (summary.decode_tps)
  std::string decode_mix;  // summary.decode_mix: "benchmark mix"...; empty for older runs (a weighted mean)
  std::vector<std::pair<std::string, double>> decode_mixes;  // summary.decode_mixes: every mix's figure, the headline's first
  std::vector<std::pair<std::string, double>> per_cat;  // (category, median decode_tps), weight > 0
  std::vector<std::pair<int, double>> prefill;  // (target_depth, pp_tps)
  std::vector<std::pair<int, double>> deep_decode;  // (context depth, decode_tps at it)
  std::vector<std::pair<int, double>> concurrency;  // (level, aggregate_tps)
  double tokens_per_joule = 0;  // at the most efficient stream count; 0 = the run did not sample power
  double quality_pct = -1;  // share of the quality questions answered correctly; -1 = not asked
  struct Miss {
    std::string item, expected, got;  // got: the answer given, or why there was none in parentheses
  };
  std::vector<Miss> quality_misses;  // summary.quality_misses: its wrong answers
  std::vector<std::pair<std::string, std::string>> notes;  // env.notes (engine, binary...)
};

struct BenchModel {
  std::string name;
  std::vector<BenchRun> runs;  // newest first
};

// What a run measures: a suite of xenithium-bench (`xenithium-bench list` describes each) or one of its scenarios.
// Standard = the numbers to quote for a build (decode, concurrency, prefill to 128K, decode at depth, an agent
// session, structured output); Quick = every headline number once; Full = every scenario to the server's context;
// Long = the long-context set (prefill to 250K, decode at depth, the needle, a long agent session); Load = many
// requests at once (concurrency, arrivals at a rate, a late long prompt, a burst on one document, mixed depths);
// Agents = coding agents on long contexts (1 / 2 / 4 agents on documents sized from the KV pool, a prompt read
// beside generating agents, a long session); Check = is the output still right (answer quality, the needle,
// repeatability, schema output, image input, tool calls); Apps = real-world applications (chat users against a latency
// target, answers from retrieved passages, editor autocomplete, an agent's tool calls, a long reply); Decode / Prefill /
// Concurrency = that scenario alone; Streams = the concurrency scenario at stream counts the user enters.
enum class BenchMode { Standard, Quick, Full, Long, Load, Agents, Check, Apps, Decode, Prefill, Concurrency, Streams };
inline constexpr BenchMode kBenchModes[] = {BenchMode::Standard, BenchMode::Quick, BenchMode::Full,   BenchMode::Long,    BenchMode::Load,        BenchMode::Agents,
                                            BenchMode::Check,    BenchMode::Apps,  BenchMode::Decode, BenchMode::Prefill, BenchMode::Concurrency, BenchMode::Streams};
// A run's measuring method for its report: its version and corpus, or for a run from before the one method its mode
// ("older, mode hard (the engine's own code and docs): does not compare with newer runs").
std::string bench_method_text(const BenchRun &r);
const char *bench_mode_name(BenchMode m);  // "standard", "quick"... (the run tag's suffix, the bench_mode note)
const char *bench_mode_title(BenchMode m);  // the menu line: what it runs and about how long it takes
// The `xenithium-bench run` arguments of the mode: its suite or its scenario.
std::vector<std::string> bench_mode_args(BenchMode m);
// The Streams mode's stream counts from what the user typed ("8, 1 24"): whole numbers from 1 to kMaxBenchStreams,
// separated by commas or spaces, at most kMaxBenchLevels of them; returned sorted without repeats ("1,8,24") for
// `--set concurrency.levels=. Empty with `err` set when the text is not that.
inline constexpr int kMaxBenchStreams = 256, kMaxBenchLevels = 12;
std::string bench_stream_levels(std::string_view text, std::string &err);
// The launch settings of a launch.env (the file the launcher started the server with) as run notes: every KEY=value
// line with a non-empty value, unquoted, in key order.
std::vector<std::pair<std::string, std::string>> launch_notes(std::string_view env_text);

class BenchSession {
 public:
  using Clock = std::chrono::steady_clock;

  struct RunState {
    bool active = false;
    std::string name;  // the run tag being written
    std::string log;  // the log file
    int pid = 0;
    Clock::time_point started{};
    std::string error;
  };

  BenchSession() = default;
  ~BenchSession();
  BenchSession(const BenchSession &) = delete;
  BenchSession &operator=(const BenchSession &) = delete;

  // Scan the runs dir (and the earlier tool's, when given) and rebuild the model groups. Runs are ordered newest
  // first within a model; models by their newest run. Dirs without a readable results.json are skipped.
  void scan(const std::string &runs_dir);
  const std::vector<BenchModel> &models() const { return models_; }
  int total_runs() const;

  // Launch `xenithium-bench run --url E --model M --name TAG [extra...]. `bin` is the xenithium-bench binary, `log`
  // where its output goes, `cwd` the working dir, `extra` the mode's arguments, --out and the --note KEY=VALUE
  // pairs. Returns false (and sets state.error) when one is already running or it could not be started.
  bool start(const std::string &bin, const std::string &endpoint, const std::string &model, const std::string &tag,
             const std::string &log, const std::string &cwd, const std::vector<std::string> &extra = {});
  void cancel();
  // Reap a finished child; true when a run just ended (the caller re-scans).
  bool poll();
  const RunState &run_state() const { return run_; }
  bool busy() const { return run_.active; }

 private:
  std::vector<BenchModel> models_;
  RunState run_;
  int child_ = -1;
};

}  // namespace xenithium:launch
