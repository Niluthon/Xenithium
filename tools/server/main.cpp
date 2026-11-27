// xenithium-server: the OpenAI-compatible HTTP server over the serving engine.
// xenithium-server --model DIR [options] (options: tools/server/options.cpp, docs/server-env.md; --help)
// xenithium-server --stub --model DIR... GPU-less: a scripted engine behind the real server
// Arch-agnostic: the checkpoint's config.json picks the arch (model:open_arch), the arch picks its chat format
// (text:open_chat_format) and its serving backend (serve:open_backend: loads the model, owns it, warms it up). The
// model dir supplies the weights, tokenizer.json, the chat template and generation_config.json (sampling defaults and
// EOS ids).
#include <execinfo.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "options.hpp"
#include "xenithium/core/batch_engine.hpp"
#include "xenithium/core/stub_engine.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/serve/backend.hpp"
#include "xenithium/server/server.hpp"
#include "xenithium/text/chat.hpp"
#include "xenithium/text/tokenizer.hpp"
#include "xenithium/util/host_mem.hpp"
#include "xenithium/util/json.hpp"
#include "xenithium/util/tls.hpp"

namespace core = xenithium::core;
namespace serve = xenithium::serve;
namespace text = xenithium::text;
namespace json = xenithium::json;

extern char **environ;

namespace xenithium {
const char *build_rev();  // `git describe --always --dirty` at build time (cmake/BuildRev.cmake)
}  // namespace xenithium

namespace {

xenithium::server::Server *g_server = nullptr;

using xenithium::server::secret_name;

// The oneAPI compiler this server was built with ("2025.3.0" from __INTEL_LLVM_COMPILER = 20250300), else the
// compiler's own version string.
std::string oneapi_version() {
#ifdef __INTEL_LLVM_COMPILER
  char v[32];
  std::snprintf(v, sizeof v, "%d.%d.%d", __INTEL_LLVM_COMPILER / 10000, __INTEL_LLVM_COMPILER / 100 % 100, __INTEL_LLVM_COMPILER % 100);
  return v;
#else
  return std::string("not oneAPI: ") + __VERSION__;
#endif
}

// The XEN_ variables as the server was started with them, taken first thing in main: the server sets some of its own
// later (XEN_EMU_CARD), which are not part of how it was started.
std::vector<std::string> g_start_env;


void on_signal(int) {
  if (g_server) g_server->stop();
}

// A crash (segfault, abort, bus error, illegal instruction, FP trap) says so in the log with a backtrace before the
// process dies, instead of only an exit status in the launcher. Async-signal-safe calls only (write, backtrace*), then
// the default action re-raised (core dump / status 128 + N as before). SIGKILL cannot be caught: the host-RAM watchdog
// below warns before the kernel's OOM killer sends it.
void on_fatal(int sig) {
  char msg[160];
  const char *name = sig == SIGSEGV ? "SIGSEGV (segmentation fault)" : sig == SIGABRT ? "SIGABRT (abort)" : sig == SIGBUS ? "SIGBUS (bus error)"
                     : sig == SIGILL ? "SIGILL (illegal instruction)" : sig == SIGFPE ? "SIGFPE (arithmetic trap)" : "fatal signal";
  const int n = std::snprintf(msg, sizeof msg, "\nserver: crashed on %s (signal %d, exit status %d); backtrace:\n", name, sig, 128 + sig);
  if (n > 0) (void)!write(STDERR_FILENO, msg, static_cast<size_t>(n));
  void *frames[64];
  const int depth = backtrace(frames, 64);
  backtrace_symbols_fd(frames, depth, STDERR_FILENO);
  std::signal(sig, SIG_DFL);
  std::raise(sig);
}

void install_crash_handlers() {
  void *warm[1];
  (void)backtrace(warm, 1);  // loads libgcc now: the first backtrace may allocate, not allowed inside the handler
  for (int sig : {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE}) std::signal(sig, on_fatal);
}

// Host RAM watchdog: every 5 s; when MemAvailable falls under util:host_mem_low's floor it logs this process's RSS and
// what is left, so a server the kernel later OOM-kills (SIGKILL: exit 137 and no message of its own) leaves the reason in
// its log. It warns once, not repeatedly: a recovery is logged once and the next fall below the floor warns again.
class MemWatch {
 public:
  MemWatch() : th_([this] { run(); }) {}
  ~MemWatch() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    th_.join();
  }

 private:
  void run() {
    bool warned = false;
    std::unique_lock<std::mutex> lk(mu_);
    while (!cv_.wait_for(lk, std::chrono::seconds(5), [this] { return stop_; })) {
      const xenithium::util::HostMem m = xenithium::util::host_mem();
      if (!m.ok) continue;
      const double gib = 1073741824.0, rss = static_cast<double>(xenithium::util::self_rss()) / gib;
      const double avail = static_cast<double>(m.available) / gib, total = static_cast<double>(m.total) / gib;
      if (xenithium::util::host_mem_low(m)) {
        if (!warned) {
          std::fprintf(stderr,
                       "server: WARNING host RAM low: this server holds %.1f GiB (RSS), %.1f of %.1f GiB left on the host; at 0 the kernel's OOM "
                       "killer ends the server (exit status 137, nothing more in this log)\n",
                       rss, avail, total);
          warned = true;
        }
      } else if (warned) {
        std::fprintf(stderr, "server: host RAM back to %.1f of %.1f GiB free (this server holds %.1f GiB)\n", avail, total, rss);
        warned = false;
      }
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread th_;
};

// EOS ids from generation_config.json (an int or a list), else the tokenizer's <|im_end|> / <|endoftext|>.
std::vector<int32_t> eos_ids(const std::string &dir, const text::Tokenizer &tok) {
  std::vector<int32_t> out;
  std::ifstream f(dir + "/generation_config.json");
  if (f) {
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value g;
    if (json::parse(ss.str(), g) && g.is_object())
      if (const json::Value *e = g.get("eos_token_id")) {
        if (e->is_int()) out.push_back(static_cast<int32_t>(e->num));
        else if (e->is_array())
          for (const auto &x : e->arr)
            if (x.is_int()) out.push_back(static_cast<int32_t>(x.num));
      }
  }
  if (out.empty())
    for (const char *t : {"<|im_end|>", "<|endoftext|>"})
      if (int32_t id = tok.token_id(t); id >= 0) out.push_back(id);
  return out;
}

core::BatchConfig batch_config(const serve::Options &o, const core::ModelInfo &info, const std::vector<int32_t> &eos, const core::AdmissionConfig &adm,
                               double arch_part_ms) {
  core::BatchConfig bc;
  bc.model = info;
  bc.eos_ids = eos;
  bc.admission = adm;
  bc.log_interval_s = o.log_interval_s;
  bc.prefill_stall_ms = o.prefill_stall_ms;
  bc.prefill_part_ms = o.prefill_part_ms >= 0 ? o.prefill_part_ms : arch_part_ms;
  bc.prefill_merge = o.prefill_merge;  // (a model whose calls take one sequence runs as with 1)
  bc.sample_threads = std::max(1, o.sample_threads);
  bc.prefix_cache = o.prefix_cache != 0;  // (a model without snapshots runs without it)
  bc.prefix_mark_min = o.prefix_mark_min;
  bc.prefix_live = o.prefix_live != 0;
  bc.lend_park_s = o.kv_lend_park_s;  // KV lending vs the prefix cache (qwen38; other archs report no lent pages)
  bc.lend_pressure = o.kv_lend_pressure;
  bc.lend_grace_s = o.kv_lend_grace_s;
  bc.adaptive_depth = o.adaptive_depth != 0;
  bc.unequal_segments = o.unequal_segments != 0;
  bc.lookup = o.lookup;
  return bc;
}

}  // namespace

int server_main(int argc, char **argv) {
  serve::Options o;
  if (const int rc = xenithium::server_cli::parse(argc, argv, o); rc >= 0) return rc;
  // --emu-card reaches the device layer and the kernels (one in a shared library) through the environment, before any
  // device is opened or plan computed (rt/card.hpp); the flag wins over an inherited XEN_EMU_CARD
  setenv("XEN_EMU_CARD", o.emu_card.c_str(), 1);
  core::AdmissionConfig adm;
  if (!core::parse_queue_policy(o.queue_policy, adm.policy)) {
    std::fprintf(stderr, "--queue-policy takes fifo or sjf\n");
    return 2;
  }
  adm.aging_s = o.queue_aging_s;
  adm.timeout_s = o.queue_timeout_s;
  if (o.plan) {  // the load's memory plan only: no tokenizer, no device, no model load
    try {
      const auto pcfg = xenithium::model::ModelConfig::load(o.model_dir);
      std::printf("%s\n", serve::plan_backend(o, pcfg, xenithium::model::open_arch(pcfg)->id()).c_str());
      return 0;
    } catch (const serve::OptionError &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
  }

  text::Tokenizer tok;
  std::string err;
  if (!tok.load(o.model_dir, err)) {
    std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
    return 1;
  }
  const auto mcfg = xenithium::model::ModelConfig::load(o.model_dir);
  const auto arch = xenithium::model::open_arch(mcfg);
  std::string chat_note;
  const auto chat = text::open_chat_format(arch->id(), o.model_dir, o.chat_template, &chat_note);
  if (!chat && o.chat_template != "auto" && o.chat_template != "builtin") {
    std::fprintf(stderr, "--chat-template %s: %s\n", o.chat_template.c_str(), chat_note.c_str());
    return 1;
  }
  std::fprintf(stderr, "server: chat: %s\n", chat_note.c_str());
  core::ModelInfo info;
  info.name = o.served_name;
  if (!core::load_generation_defaults(o.model_dir, info.default_sampling, err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  {  // The launch's sampling defaults over generation_config.json (the same ranges as a request's)
    auto &sp = info.default_sampling;
    auto num = [&](const std::string &v, const char *flag, double lo, double hi, bool lo_open, float &out) {
      if (v.empty()) return true;
      char *end = nullptr;
      const double d = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end != '\0' || !(lo_open ? d > lo : d >= lo) || d > hi) {
        std::fprintf(stderr, "%s %s: a number in %s%g, %g]\n", flag, v.c_str(), lo_open ? "(" : "[", lo, hi);
        return false;
      }
      out = static_cast<float>(d);
      return true;
    };
    bool ok = num(o.temperature, "--temperature", 0, 100, false, sp.temperature) && num(o.top_p, "--top-p", 0, 1, true, sp.top_p) &&
              num(o.min_p, "--min-p", 0, 1, false, sp.min_p) && num(o.presence_penalty, "--presence-penalty", -2, 2, false, sp.presence_penalty) &&
              num(o.repetition_penalty, "--repetition-penalty", 0, 100, true, sp.repetition_penalty);
    if (ok && !o.top_k.empty()) {
      char *end = nullptr;
      const long k = std::strtol(o.top_k.c_str(), &end, 10);
      if (end == o.top_k.c_str() || *end != '\0' || k < -1) {
        std::fprintf(stderr, "--top-k %s: -1 or 0 (off) or a positive count\n", o.top_k.c_str());
        ok = false;
      } else {
        sp.top_k = k <= 0 ? 0 : static_cast<int32_t>(std::min<long>(k, 1L << 30));
      }
    }
    if (!ok) return 1;
    std::fprintf(stderr, "server: sampling defaults (a request's own values win): temperature %g, top_p %g, top_k %d, min_p %g, presence_penalty %g, repetition_penalty %g%s\n",
                 sp.temperature, sp.top_p, sp.top_k, sp.min_p, sp.presence_penalty, sp.repetition_penalty,
                 o.temperature.empty() && o.top_p.empty() && o.top_k.empty() && o.min_p.empty() && o.presence_penalty.empty() && o.repetition_penalty.empty()
                     ? " (generation_config.json)"
                     : " (launch settings over generation_config.json)");
  }
  const auto eos = eos_ids(o.model_dir, tok);

  std::unique_ptr<serve::Backend> backend;  // owns the model the engine steps (declared first: destroyed last)
  std::unique_ptr<core::Engine> engine;
  int64_t vocab = 0;
  if (o.stub) {
    core::StubConfig sc;
    info.max_model_len = static_cast<int32_t>(o.context);
    sc.model = info;
    sc.script = tok.encode("The quick brown fox jumps over the lazy dog. ");
    sc.eos_ids = eos;
    sc.step_ms = o.stub_step_ms;
    sc.tokens_per_step = o.stub_tokens_per_step;
    sc.slots = o.slots;
    sc.admission = adm;
    engine = std::make_unique<core::StubEngine>(sc);
  } else {
    try {
      backend = serve::open_backend(o, mcfg, arch->id());
    } catch (const serve::OptionError &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
    vocab = backend->vocab();
    core::BatchConfig bc = batch_config(o, info, eos, adm, backend->prefill_part_ms());
    if (xenithium::server::MediaHook *mh = backend->media()) {  // Image ids -> the image token
      bc.media_token = mh->image_token();
      bc.media_token_of = [mh](int32_t id) { return mh->pad_token(id); };
    }
    auto be = std::make_unique<core::BatchEngine>(backend->step_model(), bc);
    backend->start(*be, tok, o.warmup);
    engine = std::move(be);
  }

  xenithium::server::Config scfg;
  scfg.host = o.host;
  scfg.port = o.port;
  scfg.idle_timeout_s = o.idle_timeout_s;
  scfg.cors_origin = o.cors_origin;
  scfg.max_queue = o.max_queue;
  scfg.eos_ids = eos;
  scfg.vocab = vocab;  // token masks cover the logits row (structured output); 0 with the stub
  scfg.media = backend ? backend->media() : nullptr;  // Image input (--vision 1)
  if (backend) {  // What the model runs on (GET /v1/models): its cards, their count the TP width
    scfg.devices = backend->devices();
    scfg.tensor_parallel = static_cast<int>(scfg.devices.size());
  }
  scfg.build = xenithium::build_rev();  // what it runs
  scfg.oneapi = oneapi_version();
  scfg.machine = xenithium::util::this_machine();
  if (backend) scfg.drafter = backend->drafter();
  if (o.report_launch) fill_launch(scfg, std::vector<std::string>(argv + 1, argv + argc), g_start_env);  // and how it was started (opt-in: it shows local paths)
  scfg.fetch.enabled = o.media_urls != 0;  // Http(s) media URLs
  scfg.fetch.allowed_domains = xenithium::util::split_domains(o.media_domains);
  scfg.fetch.timeout_s = o.media_timeout_s;
  if (scfg.media) {
    std::string why;
    const bool https = xenithium::util::tls::available(&why);
    if (!scfg.fetch.enabled) std::fprintf(stderr, "server: media URLs: data: only (--media-urls 0)\n");
    else
      std::fprintf(stderr, "server: media URLs: data:, http://%s (timeout %.0f s, %s)\n", https ? ", https://" : "", scfg.fetch.timeout_s,
                   o.media_domains.empty() ? "any host" : ("hosts " + o.media_domains).c_str());
    if (scfg.fetch.enabled && !https) std::fprintf(stderr, "server: %s\n", why.c_str());
  }
  xenithium::server::Server server(scfg, *engine, tok, chat.get());
  if (!server.listen(err)) {
    std::fprintf(stderr, "server: %s\n", err.c_str());
    return 1;
  }
  g_server = &server;
  MemWatch mem_watch;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGPIPE, SIG_IGN);
  engine->start(server.sink());
  std::fprintf(stderr, "server: %s on http://%s:%d (model %s, max_model_len %d)%s\n", o.stub ? "stub engine" : "serving", o.host.c_str(), server.port(),
               o.served_name.c_str(), engine->model().max_model_len, chat ? "" : "; no chat format for this arch");
  server.run();
  g_server = nullptr;  // qwen38-27b: a second SIGINT during shutdown hit the destroyed server (segfault after "stopped")
  engine->stop();
  std::fprintf(stderr, "server: stopped\n");
  return 0;
}

// An exception that escapes (a bad option found while the model loads, e.g. a --prefill-a8-keep item or --lm-head int4
// on a directory without lm_head.qweight) ends the server with its message and exit 1, without an abort's core dump. No
// unwinding: catching it in main hung in the destructors of the half-built devices and threads.
int main(int argc, char **argv) {
  for (char **e = environ; e && *e; ++e)
    if (std::string_view(*e).starts_with("XEN_")) g_start_env.emplace_back(*e);
  install_crash_handlers();
  std::set_terminate([] {
    if (const std::exception_ptr e = std::current_exception()) {
      try {
        std::rethrow_exception(e);
      } catch (const std::exception &x) {
        std::fprintf(stderr, "server: %s\n", x.what());
      } catch (...) {
        std::fprintf(stderr, "server: unknown exception\n");
      }
    }
    std::_Exit(1);
  });
  return server_main(argc, argv);
}
