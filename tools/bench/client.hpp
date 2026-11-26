// xenithium-bench: the client side of an OpenAI-compatible endpoint.
// One streamed /chat/completions request = one connection; Client:chat time-stamps every event as its bytes arrive
// and returns a Sample. Nothing here is specific to this engine: the extensions a request may carry (ignore_eos,
// top_k, chat_template_kwargs.enable_thinking, stream_options.include_usage) are listed in Capabilities, probed once
// per run (probe_capabilities) and left out where the endpoint refuses them.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/json.hpp"
#include "xenithium/util/machine.hpp"

namespace xenithium::bench {

using Clock = std::chrono::steady_clock;

struct Endpoint {
  bool tls = false;
  std::string host;
  std::string port;
  std::string base;  // the API's path prefix, "/v1" unless the URL names another
  std::string api_key;  // sent as "Authorization: Bearer" when not empty

  std::string url() const { return std::string(tls ? "https://" : "http://") + host + ":" + port + base; }
};
// "http(s)://host[:port][/base]"; a URL without a path gets "/v1". nullopt + err for anything else.
std::optional<Endpoint> parse_endpoint(std::string_view url, std::string &err);

// A function call of an assistant turn: what the model streamed, or what the client sends back in the history.
struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments;  // a JSON object as text, as the model wrote it
};

struct Message {
  std::string role;  // "system", "user", "assistant" or "tool"
  std::string content;
  std::vector<std::string> image_urls = {};  // images shown before the text (data: URLs); empty = a plain text turn
  std::vector<ToolCall> tool_calls = {};  // an assistant turn that called functions
  std::string tool_call_id = {};  // a "tool" turn: the id of the call it answers
};

struct Sampling {
  double temperature = 0.7;  // what chat clients send; 0 = greedy
  double top_p = 0.95;
  int top_k = 20;  // an extension: sent only where the endpoint takes it
  std::optional<int64_t> seed;
};

// Whether the model may think before it answers (chat_template_kwargs.enable_thinking, an extension).
enum class Thinking { server_default, off, on };

struct ChatRequest {
  std::vector<Message> messages;
  int max_tokens = 256;
  Sampling sampling;
  bool ignore_eos = false;  // an extension: generate max_tokens whatever the model emits (a fixed reply length)
  Thinking thinking = Thinking::server_default;
  std::string response_format;  // a JSON object ("{\"type\":\"json_schema\"...}") or empty
  std::string tools;  // a JSON array of function tools, or empty
  std::string tool_choice;  // "auto", "none", "required" or a JSON object; empty = not sent
};

// How long the client reads a reply. abandon_after_ms > 0 closes the connection that long after the submit if the
// reply is still streaming: a user who moved on (an editor's completion overtaken by the next keystroke).
struct ReadLimits {
  double abandon_after_ms = 0;
};

// What the endpoint accepts beyond the OpenAI fields, and what it reports about itself.
struct Capabilities {
  bool ignore_eos = false;
  bool top_k = false;
  bool thinking_switch = false;
  bool stream_usage = false;  // a usage chunk closes the stream (prompt / completion tokens as the server counts them)
  bool metrics = false;  // a Prometheus /metrics next to the API
  int64_t max_model_len = 0;  // the longest request it takes (0 = not reported)
  int64_t kv_pool_tokens = 0;  // tokens of context all its requests share at once (this engine's KV pool; 0 = not reported)
};

// One finished (or failed) request.
struct Sample {
  bool ok = false;
  std::string error;  // why not ok: a transport error, or the HTTP status with the head of its body
  int http_status = 0;
  Clock::time_point submitted{};
  double ttft_ms = 0;  // submit (before the connect) -> the first event that carries text
  double wall_ms = 0;  // submit -> the last event
  int prompt_tokens = 0;
  int completion_tokens = 0;  // from the usage chunk; without one, the number of text events (see usage_reported)
  int cached_tokens = -1;  // usage.prompt_tokens_details.cached_tokens; -1 = not reported
  bool cached_unknown = false;  // the cache served part of the prompt but how much is unknown (no prefill rate)
  bool usage_reported = false;
  std::string finish_reason;
  std::string text;  // the answer
  std::string reasoning;  // the thinking, where the endpoint streams it apart
  std::vector<ToolCall> tool_calls;  // the functions the reply called, put together from the stream's pieces
  bool abandoned = false;  // the client closed the stream early (ReadLimits); the counts are of what came
  std::vector<double> update_ms;  // ms since submit of every event that carried tokens (text, reasoning or a call)

  // Tokens per streamed update: > 1 when the server sends several tokens at once (a speculative step).
  double tokens_per_update() const;
  // Decode rate after the first update: the tokens of the later updates over the time from the first to the last.
  // 0 when the reply came in fewer than two updates.
  double decode_tps() const;
  // Prompt tokens the server had to process (those not served from its prefix cache) per second of TTFT.
  double prefill_tps() const;
  // The median time between two updates, ms: the decode step as a client sees it. 0 below two updates.
  double update_gap_ms() const;
};

class Client {
 public:
  Client(Endpoint endpoint, std::string model, Capabilities caps, double timeout_s);

  // Sends the request streamed and blocks until its stream ends (or `limits` abandons it). Thread-safe: every call
  // has its own connection.
  Sample chat(const ChatRequest &request, const ReadLimits &limits = {}) const;
  // GET `path` (absolute, e.g. "/metrics", or relative to the API base, e.g. "models"): the body of a 2xx.
  std::optional<std::string> get(std::string_view path, std::string *err = nullptr) const;
  // The request body chat sends for `request` under this client's capabilities.
  std::string body_of(const ChatRequest &request) const;

  const Endpoint &endpoint() const { return endpoint_; }
  const std::string &model() const { return model_; }
  const Capabilities &capabilities() const { return caps_; }

 private:
  Endpoint endpoint_;
  std::string model_;
  Capabilities caps_;
  double timeout_s_;
};

// A card the endpoint's model runs on (xenithium-server lists them in GET models; --gpu names them otherwise).
struct Device {
  std::string name;
  uint64_t memory_bytes = 0;  // as the driver reports it; 0 = not known
  int memory_gb = 0;  // the nominal size (a 32 GB card reports about 30.3 GiB); 0 = not known
  std::string pci = {};  // PCI address; empty = not known
  std::string driver = {};  // the GPU driver's version; empty = not known
};

// A model the endpoint lists (GET models): its id, and its max_model_len and kv_pool_tokens where reported (0 = not),
// and the cards it runs on and its tensor-parallel width where reported (empty / 0 = not).
struct ModelInfo {
  std::string id;
  int64_t max_model_len = 0;
  int64_t kv_pool_tokens = 0;
  int tensor_parallel = 0;
  std::vector<Device> devices;
  std::string owned_by;  // the engine as the endpoint names it ("xenithium", "vllm", "llamacpp"); empty = not reported
  // What the server runs (xenithium-server: build, oneapi, drafter) and, when it was started with --report-launch, how
  // (launch.args, launch.env); empty = not reported.
  std::string build, oneapi, drafter;
  util::Machine machine;  // the server's machine (xenithium-server: machine); an empty kernel = not reported
  std::vector<std::string> launch_args;
  std::vector<std::pair<std::string, std::string>> launch_env;
};
// A machine as GET models (machine) and the run file (env.machine, env.bench_machine) write it: os, kernel, cpu,
// cpu_threads, memory_gib; missing fields stay empty / 0.
util::Machine machine_from(const json::Value &obj);
// The model to bench: the listed entry named `wanted, or the list's first entry when `wanted` is empty. nullopt + err
// when the endpoint does not answer, refuses the key, lists nothing, or does not list `wanted.
std::optional<ModelInfo> served_model(const Endpoint &endpoint, std::string_view wanted, double timeout_s, std::string &err);

// Finds what the endpoint accepts: one tiny request with every extension, then one per extension if that is refused.
Capabilities probe_capabilities(const Endpoint &endpoint, const std::string &model, double timeout_s);

}  // namespace xenithium:bench
