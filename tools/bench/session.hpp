// xenithium-bench: what a scenario works with during a run - the endpoint's client, the corpus, the run's sampling and
// seed, prompts of a given token length, and requests sent together.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "client.hpp"
#include "corpus.hpp"
#include "counters.hpp"
#include "power.hpp"

namespace xenithium::bench {

// How long the replies of the sampled scenarios are. `natural` (the method's): the reply ends where the model ends it,
// as a chat client sees it. `fixed`: the endpoint is asked to ignore the end-of-sequence token (where it takes
// ignore_eos), so every reply runs to its max_tokens - the same number of tokens on every build, but the tokens after
// the model's own end are text no user reads, and a draft model predicts them badly or, in a loop, very well.
enum class ReplyLength { fixed, natural };

struct SessionOptions {
  uint64_t seed = 1;  // request k of the run samples with seed + k: a run repeats its draws on another build
  Sampling sampling;  // what the sampled scenarios send
  Thinking thinking = Thinking::server_default;
  int64_t context_tokens = 0;  // the longest request the endpoint takes; depths past it are skipped
  int64_t kv_pool_tokens = 0;  // tokens of context the endpoint holds for all requests at once (0 = not known)
  std::function<void(std::string_view)> progress;  // one line per finished step of a scenario (may be empty)
  PowerMeter *power = nullptr;  // the cards' power meter when the run samples power (not owned)
  ReplyLength reply_length = ReplyLength::natural;
  std::span<const ShortPrompt> prompts = measured_prompts();  // the short prompts of decode, concurrency and arrivals
  bool fresh_prompts = true;  // a unique first line on every short prompt: no reply starts from the prefix cache
};

class Session {
 public:
  Session(const Client &client, const Corpus &corpus, SessionOptions options);

  const Client &client() const { return client_; }
  const Corpus &corpus() const { return corpus_; }
  int64_t context_tokens() const { return options_.context_tokens; }
  int64_t kv_pool_tokens() const { return options_.kv_pool_tokens; }
  uint64_t seed() const { return options_.seed; }
  Thinking thinking() const { return options_.thinking; }
  ReplyLength reply_length() const { return options_.reply_length; }
  std::span<const ShortPrompt> prompts() const { return options_.prompts; }
  PowerMeter *power() const { return options_.power; }
  void progress(std::string_view line) const;

  // A single-turn request with the run's sampling and thinking mode, the run's reply length (ReplyLength) and the
  // run's next seed. Not thread-safe (the seed counter): build requests before sending them in parallel.
  ChatRequest sampled(std::string user_text, int max_tokens);
  // The same for one of the short prompts: its earlier turns, then its text at its reply length - after a unique first
  // line when the run asks for fresh prompts.
  ChatRequest sampled(const ShortPrompt &prompt);
  // At temperature 0 and without a seed, a fixed reply length (ignore_eos where the endpoint has it).
  ChatRequest greedy(std::string user_text, int max_tokens) const;

  // Characters per token of a corpus kind on this endpoint: one request's usage.prompt_tokens over a known text,
  // measured on first use (call it before sending requests in parallel). Without usage reports, a typical value.
  double chars_per_token(std::string_view kind);

  struct DocumentPrompt {
    std::string text;
    bool wrapped = false;  // the corpus was shorter than asked and repeats
  };
  // A user turn of about `tokens` tokens: [a unique first line] + the document + the question. The unique line makes
  // the prompt new to a prefix cache; pass unique = false where the cache is what is measured.
  DocumentPrompt document_prompt(std::string_view kind, int64_t tokens, std::string_view question, bool unique, size_t first_file = 0);
  // A line no earlier request of any run began with.
  std::string unique_line();

  // One request with nothing else in flight. Where the endpoint's usage does not say how many prompt tokens came from
  // its prefix cache but its /metrics counts them, the sample's cached_tokens is that counter's growth over the request;
  // a growth past the prompt (another client's hits) leaves it unknown (-1).
  Sample chat_alone(const ChatRequest &request) const;

  // Sends the requests at once (request i after start_after_s[i] seconds, 0 when the span is shorter) and waits for all.
  std::vector<Sample> run_together(std::span<const ChatRequest> requests, std::span<const double> start_after_s = {}) const;

  // Sends the requests `workers` at a time: each worker takes the next unsent request as soon as its last one ends
  // (a closed loop, the load of `workers` users who each wait for their answer). Samples in request order.
  std::vector<Sample> run_closed_loop(std::span<const ChatRequest> requests, int workers) const;

  // The server's counters now; nullopt where the endpoint has no /metrics.
  std::optional<ServerCounters> counters() const;

 private:
  const Client &client_;
  const Corpus &corpus_;
  SessionOptions options_;
  uint64_t requests_seeded_ = 0;
  uint64_t unique_lines_ = 0;
  uint64_t run_stamp_;  // the wall clock at the session's start: makes unique_line differ between runs
  std::map<std::string, double, std::less<>> chars_per_token_;
};

}  // namespace xenithium:bench
