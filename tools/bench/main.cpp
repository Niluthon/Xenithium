// xenithium-bench: the engine's benchmark, against any OpenAI-compatible endpoint (docs/bench.md). This file is the
// command line: the options (kUsage below is their reference), the run's setup, and where a run is written. The
// scenarios, the client and the reports are the xenithium_bench library beside it.
#include <unistd.h>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "client.hpp"
#include "corpus.hpp"
#include "html_report.hpp"
#include "mode.hpp"
#include "power.hpp"
#include "report.hpp"
#include "runner.hpp"
#include "session.hpp"

namespace {

namespace bench = xenithium::bench;
namespace fs = std::filesystem;

constexpr double kDefaultTimeoutS = 1800;  // one request: a 250K-token prompt on a slow endpoint still fits
constexpr double kSetupTimeoutS = 60;  // the model list and the capability probe
constexpr const char *kDefaultUrl = "http://127.0.0.1:8000/v1";
constexpr int kExitFailedRequests = 1, kExitSetup = 2;

constexpr const char *kUsage =
    "xenithium-bench: benchmark an OpenAI-compatible endpoint (docs/bench.md)\n"
    "\n"
    "  xenithium-bench run     [options]       run a suite or scenarios; prints each table as it lands\n"
    "  xenithium-bench list                    the suites and scenarios, with every parameter and its default\n"
    "  xenithium-bench report  RUN             a finished run's tables (RUN = its directory or results.json)\n"
    "  xenithium-bench html    RUN [FILE]      the run as one HTML page (default: report.html beside results.json)\n"
    "  xenithium-bench compare BASE OTHER      OTHER against BASE, metric by metric\n"
    "\n"
    "Options of run:\n"
    "  --url URL             the API base (default http://127.0.0.1:8000/v1; http://host:port gets /v1; https works)\n"
    "  --model ID            the model to bench (default: the endpoint's first)\n"
    "  --api-key KEY         sent as a bearer token (default $XEN_BENCH_API_KEY, else $OPENAI_API_KEY)\n"
    "  --suite ID            quick, standard (default), full, long, load, agents, check, apps (`list` says what each runs)\n"
    "  --scenario a,b        these scenarios instead of a suite\n"
    "  --set s.key=value     one parameter of scenario s; repeatable\n"
    "  --name TAG            the run is <time>-TAG (default: the suite's name)\n"
    "  --note KEY=VALUE      recorded in the run file; repeatable\n"
    "  --out DIR             where runs go (default $XEN_BENCH_RUNS, else ~/.xenithium/bench/runs)\n"
    "  --seed N              request k samples with seed N + k (default 1)\n"
    "  --temperature T --top-p P --top-k K    sampling (default 0.7 / 0.95 / 20; T 0 = greedy)\n"
    "  --thinking default|on|off              the model's thinking switch, where the endpoint has one\n"
    "  --mode easy|medium|hard  how easy the measured text is for a draft model (default hard):\n"
    "                        hard = the engine's own code and docs, open tasks, replies run to their length, the\n"
    "                        agent mix (the published numbers); medium = the same method over conventional\n"
    "                        template-made code and prose; easy = tasks that repeat or reshape their input, replies\n"
    "                        that end where the model ends them, a code-led weighted mean (the public-benchmark style)\n"
    "  --reply-length fixed|natural           the sampled scenarios' replies (default: the mode's, fixed for hard and\n"
    "                        medium, natural for easy): fixed = run to max_tokens (ignore_eos: equal lengths on every\n"
    "                        build), natural = end where the model ends them (what a chat client sees)\n"
    "  --context TOKENS      the longest prompt to send (default: what the endpoint reports)\n"
    "  --kv-pool TOKENS      tokens of context the endpoint holds for all requests at once (default: what it\n"
    "                        reports); the scenarios sized by a share of it use this\n"
    "  --gpu NAME            a card the endpoint runs on, once per card (default: what it reports; xenithium-server\n"
    "                        lists its cards), shown in the report's hardware row\n"
    "  --tp N                the endpoint's tensor-parallel width (default: what it reports)\n"
    "  --timeout SECONDS     per request (default 1800)\n"
    "  --corpus hard|medium|DIR the text long prompts are cut from (default: the mode's): hard = this engine's own code\n"
    "                        and docs, medium = conventional template-made code and prose a drafter guesses well,\n"
    "                        DIR = another directory of DIR/<kind>/<files>\n"
    "  --power auto|on|off   sample the GPUs' board power and clock from this machine's hwmon files (auto: when\n"
    "                        the URL is this machine's loopback address)\n"
    "  --strict              OpenAI fields only: no ignore_eos / top_k / thinking switch / usage-in-stream probing\n"
    "\n"
    "Exit code: 0 = every request answered, 1 = some failed, 2 = a usage or setup error.\n";

struct Options {
  std::string url = kDefaultUrl, model, api_key, suite = "standard", scenarios, name, out_dir, thinking = "default", power = "auto";
  std::string mode = std::string(bench::kDefaultMode);
  std::string corpus, reply_length;  // empty = the mode's
  std::vector<std::string> overrides;
  std::vector<bench::Label> notes;
  uint64_t seed = 1;
  bench::Sampling sampling;
  int64_t context_tokens = 0;
  int64_t kv_pool_tokens = 0;
  std::vector<std::string> command;  // the run's arguments after `run` (--api-key's value hidden)
  std::vector<std::string> gpus;  // --gpu, one per card
  int tensor_parallel = 0;
  double timeout_s = kDefaultTimeoutS;
  bool strict = false;
};

// The command line is wrong: say what, and where the reference is.
[[noreturn]] void usage_error(const std::string &why) {
  std::fprintf(stderr, "xenithium-bench: %s\n`xenithium-bench help` lists the commands and options.\n", why.c_str());
  std::exit(kExitSetup);
}
// The command line is fine but the run cannot start (no endpoint, an unreadable file).
[[noreturn]] void setup_error(const std::string &why) {
  std::fprintf(stderr, "xenithium-bench: %s\n", why.c_str());
  std::exit(kExitSetup);
}

std::string env_or(const char *name, std::string fallback) {
  const char *v = std::getenv(name);
  return v && *v ? std::string(v) : fallback;
}

std::string runs_dir(const Options &o) {
  if (!o.out_dir.empty()) return o.out_dir;
  return env_or("XEN_BENCH_RUNS", env_or("HOME", ".") + "/.xenithium/bench/runs");
}

std::string timestamp_now() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  ::localtime_r(&now, &local);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &local);
  return buf;
}

std::string host_name() {
  // XEN_BENCH_ID names the machine in the report instead of the hostname: a published report should
  // carry a lab name, not a machine identity (the hostname is otherwise the local network's name for the box).
  if (const char *id = std::getenv("XEN_BENCH_ID"); id && *id) return id;
  char name[256] = {};
  return ::gethostname(name, sizeof(name) - 1) == 0 ? std::string(name) : std::string("unknown");
}

bench::Thinking thinking_of(const std::string &word) {
  if (word == "default") return bench::Thinking::server_default;
  if (word == "on") return bench::Thinking::on;
  if (word == "off") return bench::Thinking::off;
  usage_error("--thinking takes default, on or off");
}

const bench::Mode &mode_of(const std::string &id) {
  if (const bench::Mode *mode = bench::find_mode(id)) return *mode;
  usage_error("--mode takes easy, medium or hard");
}

bench::ReplyLength reply_length_of(const std::string &word) {
  if (word == "fixed") return bench::ReplyLength::fixed;
  if (word == "natural") return bench::ReplyLength::natural;
  usage_error("--reply-length takes fixed or natural");
}

// The option's value as a number; a value that is not one is a usage error naming the option.
template <typename Number>
Number number_of(const std::string &flag, const std::string &value) {
  Number number{};
  const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), number);
  if (ec != std::errc() || end != value.data() + value.size()) usage_error(flag + " takes a number, got \"" + value + "\"");
  return number;
}

Options parse_run_options(int argc, char **argv) {
  Options o;
  for (int i = 2; i < argc; ++i) {  // the command as typed, for the run file (the key's value and a note's secrets hidden)
    const std::string arg = argv[i], before = i > 2 ? argv[i - 1] : "";
    const size_t eq = arg.find('=');
    if (before == "--api-key") o.command.emplace_back("(hidden)");
    else if (before == "--note" && eq != std::string::npos) o.command.push_back(arg.substr(0, eq + 1) + bench::hide_note(arg.substr(0, eq), arg.substr(eq + 1)));
    else o.command.push_back(arg);
  }
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--strict") {
      o.strict = true;
      continue;
    }
    if (i + 1 >= argc) usage_error(flag + " needs a value");
    const std::string value = argv[++i];
    if (flag == "--url") o.url = value;
    else if (flag == "--model") o.model = value;
    else if (flag == "--api-key") o.api_key = value;
    else if (flag == "--suite") o.suite = value;
    else if (flag == "--scenario") o.scenarios = value;
    else if (flag == "--set") o.overrides.push_back(value);
    else if (flag == "--name") o.name = value;
    else if (flag == "--out") o.out_dir = value;
    else if (flag == "--mode") o.mode = value;
    else if (flag == "--corpus") o.corpus = value;
    else if (flag == "--thinking") o.thinking = value;
    else if (flag == "--reply-length") o.reply_length = value;
    else if (flag == "--power") o.power = value;
    else if (flag == "--seed") o.seed = number_of<uint64_t>(flag, value);
    else if (flag == "--temperature") o.sampling.temperature = number_of<double>(flag, value);
    else if (flag == "--top-p") o.sampling.top_p = number_of<double>(flag, value);
    else if (flag == "--top-k") o.sampling.top_k = number_of<int>(flag, value);
    else if (flag == "--context") o.context_tokens = number_of<int64_t>(flag, value);
    else if (flag == "--kv-pool") o.kv_pool_tokens = number_of<int64_t>(flag, value);
    else if (flag == "--gpu") o.gpus.push_back(value);
    else if (flag == "--tp") o.tensor_parallel = number_of<int>(flag, value);
    else if (flag == "--timeout") o.timeout_s = number_of<double>(flag, value);
    else if (flag == "--note") {
      const size_t eq = value.find('=');
      if (eq == std::string::npos) usage_error("--note takes KEY=VALUE");
      o.notes.emplace_back(value.substr(0, eq), bench::hide_note(value.substr(0, eq), value.substr(eq + 1)));  // never a key in the run file
    } else usage_error("unknown option " + flag);
  }
  if (o.api_key.empty()) o.api_key = env_or("XEN_BENCH_API_KEY", env_or("OPENAI_API_KEY", ""));
  const bench::Mode &mode = mode_of(o.mode);  // refused before anything is sent, like a wrong reply length
  if (o.corpus.empty()) o.corpus = mode.corpus;
  if (o.reply_length.empty()) o.reply_length = mode.reply_length == bench::ReplyLength::natural ? "natural" : "fixed";
  (void)reply_length_of(o.reply_length);
  return o;
}

// --corpus: an embedded set's name, else a directory.
bench::Corpus load_corpus(const Options &o) {
  if (bench::find_corpus_set(o.corpus)) return bench::Corpus::from_embedded(o.corpus);
  std::string err;
  auto corpus = bench::Corpus::from_dir(o.corpus, err);
  if (!corpus) setup_error(err);
  return std::move(*corpus);
}

bool write_file(const fs::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
  if (out.good()) return true;
  std::fprintf(stderr, "xenithium-bench: cannot write %s\n", path.c_str());
  return false;
}

std::optional<bench::Run> load_run(const std::string &where, std::string &err) {
  fs::path path(where);
  if (fs::is_directory(path)) path /= "results.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) return err = "cannot read " + path.string(), std::nullopt;
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return bench::run_from_json(text, err);
}

bool is_loopback(const std::string &host) { return host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]"; }

// The cards' power meter when the run is to sample power: asked for, or by default when the endpoint is on this
// machine (the counters of another machine's cards would say nothing about it).
std::unique_ptr<bench::PowerMeter> power_meter_for(const Options &o, const bench::Endpoint &endpoint) {
  if (o.power != "auto" && o.power != "on" && o.power != "off") usage_error("--power takes auto, on or off");
  if (o.power == "off" || (o.power == "auto" && !is_loopback(endpoint.host))) return nullptr;
  auto meter = bench::PowerMeter::open();
  if (!meter && o.power == "on") setup_error("--power on: no GPU under /sys/class/drm has a hwmon energy counter");
  return meter;
}

// A run's HTML page beside its results.json (or at `file`).
int cmd_html(const std::string &where, const std::string &file) {
  std::string err;
  const auto run = load_run(where, err);
  if (!run) setup_error(err);
  fs::path out = file;
  if (file.empty()) out = (fs::is_directory(where) ? fs::path(where) : fs::path(where).parent_path()) / "report.html";
  if (!write_file(out, bench::render_html(*run))) return kExitSetup;
  std::printf("%s\n", out.c_str());
  return 0;
}

int cmd_list() {
  std::printf("Modes (xenithium-bench run --mode ID):\n");
  for (const bench::Mode &mode : bench::modes())
    std::printf("  %-9.*s %.*s\n", static_cast<int>(mode.id.size()), mode.id.data(), static_cast<int>(mode.about.size()), mode.about.data());
  std::printf("\nSuites (xenithium-bench run --suite ID):\n");
  for (const bench::Suite &s : bench::suites()) {
    std::printf("  %-9s ~%d min  %s\n            ", s.id.c_str(), s.minutes, s.about.c_str());
    for (const bench::SuiteStep &step : s.steps) std::printf("%s ", step.scenario.c_str());
    std::printf("\n");
  }
  std::printf("\nScenarios (--scenario ID[,ID]; parameters with --set ID.key=value):\n");
  for (const auto &scenario : bench::all_scenarios()) {
    std::printf("  %.*s\n      %.*s\n", static_cast<int>(scenario->id().size()), scenario->id().data(), static_cast<int>(scenario->about().size()), scenario->about().data());
    for (const bench::ParamSpec &p : scenario->params())
      std::printf("      %-14.*s %-36.*s %.*s\n", static_cast<int>(p.key.size()), p.key.data(), static_cast<int>(p.fallback.size()), p.fallback.data(),
                  static_cast<int>(p.about.size()), p.about.data());
  }
  return 0;
}

int cmd_report(const std::string &where) {
  std::string err;
  const auto run = load_run(where, err);
  if (!run) setup_error(err);
  std::fputs(bench::render_run(*run).c_str(), stdout);
  return 0;
}

int cmd_compare(const std::string &base_at, const std::string &other_at) {
  std::string err;
  const auto base = load_run(base_at, err);
  if (!base) setup_error(err);
  const auto other = load_run(other_at, err);
  if (!other) setup_error(err);
  std::fputs(bench::render_compare(*base, *other).c_str(), stdout);
  return 0;
}

// The model to bench and its context length. Without --model the endpoint must list one; with --model an endpoint
// that has no usable model list is still benched (some gateways list nothing), its context then unknown.
bench::ModelInfo model_to_bench(const bench::Endpoint &endpoint, const Options &o) {
  std::string err;
  if (const auto served = bench::served_model(endpoint, o.model, kSetupTimeoutS, err)) return *served;
  if (o.model.empty()) setup_error("the endpoint " + endpoint.url() + ": " + err);
  std::fprintf(stderr, "xenithium-bench: %s; going on with --model %s, its context unknown\n", err.c_str(), o.model.c_str());
  bench::ModelInfo info;
  info.id = o.model;
  return info;
}

// One tiny request as the run will send them. A refusal here (a wrong key, a wrong model name) would fail every
// request of the run, so it stops the run instead.
void require_first_reply(const bench::Client &client) {
  bench::ChatRequest tiny;
  tiny.messages = {{"user", "Say OK."}};
  tiny.max_tokens = 4;
  if (const bench::Sample reply = client.chat(tiny); !reply.ok) setup_error("the endpoint refused a first request: " + reply.error);
}

int cmd_run(const Options &o) {
  std::string err;
  auto endpoint = bench::parse_endpoint(o.url, err);
  if (!endpoint) usage_error(err);
  endpoint->api_key = o.api_key;

  const bench::Mode &mode = mode_of(o.mode);
  bench::RunPlan plan;
  try {
    plan = bench::make_plan(o.suite, o.scenarios, o.overrides, mode.defaults);
  } catch (const std::invalid_argument &e) {
    usage_error(e.what());
  }

  const bench::ModelInfo served = model_to_bench(*endpoint, o);
  const std::string &model = served.id;
  bench::Capabilities caps = o.strict ? bench::Capabilities{} : bench::probe_capabilities(*endpoint, model, kSetupTimeoutS);
  caps.max_model_len = served.max_model_len;
  caps.kv_pool_tokens = o.kv_pool_tokens > 0 ? o.kv_pool_tokens : served.kv_pool_tokens;
  const int64_t context_tokens = o.context_tokens > 0 ? o.context_tokens : caps.max_model_len;
  // What the endpoint runs on: --gpu / --tp over what it reports (--gpu without --tp: one card each)
  std::vector<bench::Device> devices = served.devices;
  if (!o.gpus.empty()) {
    devices.clear();
    for (const std::string &name : o.gpus) devices.push_back({.name = name});
  }
  const int tensor_parallel = o.tensor_parallel > 0 ? o.tensor_parallel : !o.gpus.empty() ? static_cast<int>(o.gpus.size()) : served.tensor_parallel;

  const bench::Corpus corpus = load_corpus(o);
  const bench::CorpusSet *corpus_set = bench::find_corpus_set(o.corpus);
  const bench::Client client(*endpoint, model, caps, o.timeout_s);
  require_first_reply(client);
  const std::unique_ptr<bench::PowerMeter> power = power_meter_for(o, *endpoint);
  bench::Session session(client, corpus,
                         {.seed = o.seed,
                          .sampling = o.sampling,
                          .thinking = thinking_of(o.thinking),
                          .context_tokens = context_tokens,
                          .kv_pool_tokens = caps.kv_pool_tokens,
                          .progress = [](std::string_view line) { std::printf("  .. %.*s\n", static_cast<int>(line.size()), line.data()), std::fflush(stdout); },
                          .power = power.get(),
                          .reply_length = reply_length_of(o.reply_length),
                          .prompts = bench::short_prompts(mode.prompts),
                          .fresh_prompts = mode.fresh_prompts});

  bench::RunInfo info{.name = o.name.empty() ? plan.suite : o.name,
                      .timestamp = timestamp_now(),
                      .host = host_name(),
                      .endpoint = endpoint->url(),
                      .engine = served.owned_by,
                      .engine_build = served.build,
                      .engine_oneapi = served.oneapi,
                      .engine_drafter = served.drafter,
                      .server_args = served.launch_args,
                      .server_env = served.launch_env,
                      .command = o.command,
                      .model = model,
                      .context_tokens = context_tokens,
                      .kv_pool_tokens = caps.kv_pool_tokens,
                      .tensor_parallel = tensor_parallel,
                      .devices = devices,
                      .machine = served.machine,
                      .bench_machine = xenithium::util::this_machine(),
                      .suite = plan.suite,
                      .seed = o.seed,
                      .sampling = o.sampling,
                      .thinking = o.thinking,
                      .mode = o.mode,
                      .mode_version = mode.version,
                      .reply_length = o.reply_length,
                      .corpus = o.corpus,
                      .corpus_version = corpus_set ? corpus_set->version : bench::kCorpusVersion,
                      .corpus_hash = corpus.hash(),
                      .capabilities = caps,
                      .notes = o.notes};
  const fs::path dir = fs::path(runs_dir(o)) / (info.timestamp + "-" + info.name);
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) setup_error("cannot create " + dir.string() + ": " + ec.message());

  std::printf("xenithium-bench: %s, model %s, context %s, KV pool %s; suite %s (%zu scenarios) -> %s\n", info.endpoint.c_str(), model.c_str(),
              context_tokens > 0 ? std::to_string(context_tokens).c_str() : "not reported (depths are not limited: pass --context)",
              caps.kv_pool_tokens > 0 ? std::to_string(caps.kv_pool_tokens).c_str() : "not reported", plan.suite.c_str(), plan.steps.size(), dir.c_str());
  std::printf("  extensions: ignore_eos %s, top_k %s, thinking switch %s, usage in the stream %s, /metrics %s\n", caps.ignore_eos ? "yes" : "no", caps.top_k ? "yes" : "no",
              caps.thinking_switch ? "yes" : "no", caps.stream_usage ? "yes" : "no (token counts are estimates)", caps.metrics ? "yes" : "no");
  std::printf("  runs on: %s\n", bench::devices_text(info.devices, info.tensor_parallel, "not reported (pass --gpu NAME, --tp N)").c_str());
  if (power) std::printf("  power: sampling %zu card(s) from hwmon (board energy, GPU clock)\n", power->cards().size());
  std::printf("  mode %s: %.*s; long prompts from the %s corpus\n", o.mode.c_str(), static_cast<int>(mode.about.size()), mode.about.data(), o.corpus.c_str());
  std::printf("  replies of the sampled scenarios: %s length%s\n", o.reply_length.c_str(),
              o.reply_length == "fixed" && !caps.ignore_eos ? " asked, but the endpoint refuses ignore_eos: they end where the model ends them" : "");
  std::fflush(stdout);

  const bench::Run run = bench::execute(plan, session, std::move(info), [](const bench::ScenarioResult &s) {
    std::printf("\n%s", bench::render_scenario(s).c_str());
    std::fflush(stdout);
  });
  write_file(dir / "results.json", bench::to_json(run));
  write_file(dir / "report.txt", bench::render_run(run));
  write_file(dir / "report.html", bench::render_html(run));
  int failed = 0;
  for (const bench::ScenarioResult &s : run.scenarios) failed += s.errors;
  std::printf("\nxenithium-bench: done in %.0f s, %d failed request(s); run %s\n", run.info.seconds, failed, dir.c_str());
  return failed ? kExitFailedRequests : 0;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string command = argc > 1 ? argv[1] : "";
  if (command == "help" || command == "--help" || command == "-h") return std::fputs(kUsage, stdout), 0;
  if (command == "list") return cmd_list();
  if (command == "report" && argc == 3) return cmd_report(argv[2]);
  if (command == "compare" && argc == 4) return cmd_compare(argv[2], argv[3]);
  if (command == "html" && (argc == 3 || argc == 4)) return cmd_html(argv[2], argc == 4 ? argv[3] : "");
  if (command == "run") return cmd_run(parse_run_options(argc, argv));
  usage_error(command.empty() ? "no command" : "unknown command or wrong number of arguments: " + command);
}
