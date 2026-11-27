// xenithium-bench: a run as data - what was measured against which endpoint - its file form (results.json) and its
// renderings (the console tables, a comparison of two runs, the headline numbers other tools read).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "client.hpp"
#include "format.hpp"
#include "types.hpp"

namespace xenithium::bench {

inline constexpr int kRunSchema = 1;  // of results.json; bumped when a reader of the old form would misread the new

struct RunInfo {
  std::string name;  // the run's tag (its directory is <timestamp>-<name>)
  std::string timestamp;  // "20261116-200500" local time: sorts by time
  std::string host;  // where the tool ran
  std::string endpoint;
  std::string engine = {};  // the endpoint's owned_by ("xenithium", "vllm"); empty = not reported (runs from before)
  // What the server runs and how it was started, as it reports it (xenithium-server's GET /v1/models): its build's git
  // revision, the oneAPI compiler, the drafter ("none", "mtp", "dflash2", "dspine", "...+mtp"); with --report-launch its
  // arguments after the program and the XEN_ variables it saw. Empty = not reported.
  std::string engine_build = {}, engine_oneapi = {}, engine_drafter = {};
  std::vector<std::string> server_args = {};
  std::vector<Label> server_env = {};
  std::vector<std::string> command = {};  // this run's xenithium-bench arguments after `run` (--api-key hidden)
  std::string model;
  int64_t context_tokens = 0;
  int64_t kv_pool_tokens = 0;  // the endpoint's shared KV pool, where it reports one
  int tensor_parallel = 0;  // the endpoint's TP width (0 = not known)
  std::vector<Device> devices = {};  // the cards it runs on, in rank order (empty = not known)
  // The server's machine as it reports it, and the machine the benchmark ran on (an empty kernel = not known)
  util::Machine machine = {}, bench_machine = {};
  std::string suite;  // a suite's id, or "custom" for a hand-picked scenario list
  uint64_t seed = 0;
  Sampling sampling;
  std::string thinking;  // "default", "on" or "off"
  std::string mode = "hard";  // the run's mode (mode.hpp): "easy", "medium" or "hard"
  int mode_version = 1;
  std::string reply_length = "fixed";  // of the sampled scenarios' replies: "fixed" or "natural" (ReplyLength)
  std::string corpus;  // the embedded set ("hard", "medium") or the directory the prompts were cut from
  int corpus_version = 0;
  std::string corpus_hash;
  Capabilities capabilities;
  std::vector<Label> notes;  // free key = value facts about the endpoint (the launcher passes the server's settings)
  double seconds = 0;
  std::optional<RunPower> power = std::nullopt;  // when the run sampled the cards' power
};

struct Run {
  RunInfo info;
  std::vector<ScenarioResult> scenarios;
};

// The numbers a run is known by; empty where the run did not measure them.
struct Headline {
  double decode_tps = 0;  // the decode figure of the categories' mix
  std::string decode_mix;  // its row's label: "agent mix" ("weighted" in runs before the mixes)
  std::vector<std::pair<std::string, double>> decode_by_category;
  struct Depth {
    std::string depth;  // "32,000"
    double tps = 0, ttft_ms = 0;
  };
  std::vector<Depth> prefill;  // prefill tok/s and TTFT per depth (the first corpus kind)
  std::vector<Depth> deep_decode;  // decode tok/s per depth (ttft_ms = the prefix-cache hit's)
  std::vector<std::pair<std::string, double>> concurrency;  // (streams, tok/s of all)
  double peak_total_tps = 0;  // the best concurrency level's tok/s of all streams
  std::string peak_streams;  // ... and its stream count
  double best_tokens_per_joule = 0;  // the most efficient concurrency level (0 = power not sampled)
  double quality_pct = -1;  // share of the quality questions answered correctly (-1 = not asked)
  std::vector<Miss> quality_misses;  // its wrong answers
};
Headline headline(const Run &run);

std::string to_json(const Run &run);

// A command line with what must not leave the machine hidden (util/secrets.hpp, the server's rules): the value of a
// secret-named NAME=V, --flag=V or --flag V, a token-looking word, a URL's password. The spacing is kept.
std::string hide_command_secrets(std::string_view command);
// A note's value as it may be recorded and shown: "(hidden)" for a secret name or value, else hide_command_secrets (a
// server_cmd note is a command line the user wrote).
std::string hide_note(std::string_view key, std::string_view value);
std::optional<Run> run_from_json(std::string_view text, std::string &err);

// "1x Intel Arc Pro B65 Graphics, 32 GB": the cards, equal neighbours counted once ("(R)" and "(TM)" dropped), the
// nominal size where known (else the driver's GiB); empty when none are known.
std::string gpus_text(const std::vector<Device> &devices);
// The cards and the TP width ("1x Intel Arc Pro B65 Graphics, 32 GB; TP 1"), or `missing` when neither is known.
std::string devices_text(const std::vector<Device> &devices, int tensor_parallel, std::string_view missing);

// A scenario's rows as text cells: the label columns, then one column per metric (a latency with several samples
// shows "median / max", a yes / no metric "yes/total"). Rows that only carry notes are left out.
struct TableView {
  std::string caption;  // Row:table of its rows; empty for the scenario's main table
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  size_t label_columns = 0;  // the leading columns that are labels, not numbers
};
// The scenario's tables in first-seen order of Row:table: usually one.
std::vector<TableView> tables_of(const ScenarioResult &scenario);
// "power: 418 W mean (card 0 211, card 1 207), 12.4 kJ; GPU clock mean 2,310 MHz, lowest 1,650" for a scenario with
// a power record.
std::string power_line(const ScenarioResult &scenario);

// "1 wrong answer of 34 asked": the heading of a scenario's misses (the asked count is its yes / no samples).
std::string misses_heading(const ScenarioResult &scenario);
// A miss's answer as a report shows it: the answer given in quotes, or why there was none in parentheses.
std::string given_text(const Miss &miss);
// The notes of a scenario's rows in order, each text once (rows of one case often say the same thing).
std::vector<std::string> unique_notes(const ScenarioResult &scenario);

// One scenario's table with its notes, as printed when it finishes.
std::string render_scenario(const ScenarioResult &scenario);
// The whole run: a header, the headline and every scenario.
std::string render_run(const Run &run);
// `other` against `base, row by row and metric by metric, with the change in percent. A change is marked when it is
// larger than kNoticeablePct and than what the samples of either run scatter by.
std::string render_compare(const Run &base, const Run &other);

}  // namespace xenithium:bench
