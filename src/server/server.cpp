#include "xenithium/server/server.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "xenithium/server/responses.hpp"
#include "xenithium/text/json_constraint.hpp"
#include "xenithium/util/host_mem.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::server {

using json::Value;

namespace {

constexpr const char* kVersion = "0.2.0";
constexpr size_t kMaxChoices = 128;  // OpenAI's `n` bound

std::string random_hex(size_t n) {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string s(n, '0');
  uint64_t bits = 0;
  for (size_t i = 0; i < n; ++i) {
    if (i % 16 == 0) bits = rng();
    s[i] = hex[bits & 15];
    bits >>= 4;
  }
  return s;
}

void append_int(std::string& out, long long v) {
  char b[24];
  const int n = std::snprintf(b, sizeof(b), "%lld", v);
  out.append(b, static_cast<size_t>(n));
}

// A request error in OpenAI's shape.
struct ApiError {
  int status;
  std::string message;
  std::string type;
  std::string param;
  ApiError(int s = 400, std::string m = {}, std::string t = "BadRequestError", std::string p = {})
      : status(s), message(std::move(m)), type(std::move(t)), param(std::move(p)) {}
};

std::string error_body(const ApiError& e) {
  std::string s = "{\"error\":{\"message\":";
  json::write_string(e.message, s);
  s += ",\"type\":";
  json::write_string(e.type, s);
  s += ",\"param\":";
  if (e.param.empty()) s += "null";
  else json::write_string(e.param, s);
  s += ",\"code\":";
  append_int(s, e.status);
  s += "}}";
  return s;
}

// ---- typed field readers: absent/null leaves `out` unchanged.

bool read_number(const Value& req, const char* key, double& out, ApiError& err) {
  const Value* v = req.get(key);
  if (!v || v->is_null()) return true;
  if (!v->is_number()) {
    err = {400, std::string(key) + " must be a number", "BadRequestError", key};
    return false;
  }
  out = v->num;
  return true;
}

bool read_int(const Value& req, const char* key, long long& out, bool& present, ApiError& err) {
  present = false;
  const Value* v = req.get(key);
  if (!v || v->is_null()) return true;
  if (!v->is_number() || v->num != std::floor(v->num) || std::fabs(v->num) > 9.0e15) {
    err = {400, std::string(key) + " must be an integer", "BadRequestError", key};
    return false;
  }
  out = static_cast<long long>(v->num);
  present = true;
  return true;
}

bool read_bool(const Value& req, const char* key, bool& out, ApiError& err) {
  const Value* v = req.get(key);
  if (!v || v->is_null()) return true;
  if (!v->is_bool()) {
    err = {400, std::string(key) + " must be a boolean", "BadRequestError", key};
    return false;
  }
  out = v->b;
  return true;
}

// OpenAI `logit_bias`: {"<token id>": bias}, bias in [-100, 100].
bool read_logit_bias(const Value& r, int32_t vocab, std::unordered_map<int32_t, float>& out, ApiError& err) {
  const Value* v = r.get("logit_bias");
  if (!v || v->is_null()) return true;
  if (!v->is_object()) {
    err = {400, "logit_bias must be an object of token id to bias", "BadRequestError", "logit_bias"};
    return false;
  }
  for (const json::Member& m : v->obj) {
    char* end = nullptr;
    errno = 0;
    const long id = std::strtol(m.first.c_str(), &end, 10);
    if (m.first.empty() || *end != '\0' || errno != 0 || id < 0 || id >= static_cast<long>(vocab)) {
      err = {400,
             "logit_bias keys must be token ids in [0, " + std::to_string(vocab) + "): \"" + m.first + "\"",
             "BadRequestError",
             "logit_bias"};
      return false;
    }
    if (!m.second.is_number()) {
      err = {400, "logit_bias values must be numbers", "BadRequestError", "logit_bias"};
      return false;
    }
    if (m.second.num < -100.0 || m.second.num > 100.0) {
      err = {400, "logit_bias values must be in [-100, 100]", "BadRequestError", "logit_bias"};
      return false;
    }
    out[static_cast<int32_t>(id)] = static_cast<float>(m.second.num);
  }
  return true;
}

// `response_format` (OpenAI) and the guided_json / guided_choice extensions become a JSON schema
// the decoded text must follow.
bool parse_response_format(const Value& r, Value& schema, bool& have, bool& raw, ApiError& err) {
  have = false;
  raw = false;
  if (const Value* g = r.get("guided_json"); g && !g->is_null()) {
    if (g->is_object()) {
      schema = *g;
      have = true;
      return true;
    }
    if (g->is_string()) {
      Value parsed;
      std::string perr;
      if (!json::parse(g->str, parsed, &perr) || !parsed.is_object()) {
        err = {400, "guided_json must be a JSON schema object or its text: " + perr, "BadRequestError", "guided_json"};
        return false;
      }
      schema = parsed;
      have = true;
      return true;
    }
    err = {400, "guided_json must be a JSON schema object or its text", "BadRequestError", "guided_json"};
    return false;
  }
  if (const Value* g = r.get_array("guided_choice"); g) {
    for (const Value& c : g->arr)
      if (!c.is_string()) {
        err = {400, "guided_choice must be a list of strings", "BadRequestError", "guided_choice"};
        return false;
      }
    Value s = Value::object();
    s.set("type", Value::string("string"));
    s.set("enum", *g);
    schema = s;
    have = true;
    raw = true;  // guided_choice emits the choice text itself
    return true;
  }
  if (const Value* so = r.get_object("structured_outputs")) {
    if (const Value* j = so->get("json"); j && !j->is_null()) {
      Value pick = Value::object();
      pick.set("guided_json", *j);
      return parse_response_format(pick, schema, have, raw, err);
    }
    if (const Value* ch = so->get("choice"); ch && !ch->is_null()) {
      Value pick = Value::object();
      pick.set("guided_choice", *ch);
      return parse_response_format(pick, schema, have, raw, err);
    }
    err = {400, "structured_outputs supports json and choice only", "BadRequestError", "structured_outputs"};
    return false;
  }
  for (const char* k : {"guided_regex", "guided_grammar"})
    if (const Value* v = r.get(k); v && !v->is_null()) {
      err = {400, std::string(k) + " is not supported; use response_format or guided_json", "BadRequestError", k};
      return false;
    }
  const Value* rf = r.get("response_format");
  if (!rf || rf->is_null()) return true;
  if (!rf->is_object()) {
    err = {400, "response_format must be an object", "BadRequestError", "response_format"};
    return false;
  }
  const std::string* t = rf->get_string("type");
  if (!t) {
    err = {400, "response_format.type is required", "BadRequestError", "response_format"};
    return false;
  }
  if (*t == "text") return true;
  if (*t == "json_object") {
    Value s = Value::object();
    s.set("type", Value::string("object"));
    schema = s;
    have = true;
    return true;
  }
  if (*t == "json_schema") {
    const Value* js = rf->get_object("json_schema");
    if (!js) {
      err = {400, "response_format.json_schema is required when type is json_schema", "BadRequestError",
             "response_format"};
      return false;
    }
    const Value* s = js->get_object("schema");
    if (!s) {
      err = {400, "response_format.json_schema.schema is required", "BadRequestError", "response_format"};
      return false;
    }
    schema = *s;
    have = true;
    return true;
  }
  err = {400, "response_format.type must be text, json_object or json_schema", "BadRequestError", "response_format"};
  return false;
}

// tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name"...}}.
bool parse_tool_choice(const Value& r, std::string& mode, std::string& name, ApiError& err) {
  mode = "auto";
  name.clear();
  const Value* v = r.get("tool_choice");
  if (!v || v->is_null()) return true;
  if (v->is_string()) {
    if (v->str == "none" || v->str == "auto" || v->str == "required") {
      mode = v->str;
      return true;
    }
    err = {400, "tool_choice must be \"none\", \"auto\", \"required\" or a function object", "BadRequestError",
           "tool_choice"};
    return false;
  }
  if (v->is_object()) {
    const Value* f = v->get_object("function");
    const std::string* n = f ? f->get_string("name") : nullptr;
    if (!n || n->empty()) {
      err = {400, "tool_choice.function.name is required", "BadRequestError", "tool_choice"};
      return false;
    }
    mode = "function";
    name = *n;
    return true;
  }
  err = {400, "tool_choice must be \"none\", \"auto\", \"required\" or a function object", "BadRequestError",
         "tool_choice"};
  return false;
}

}  // namespace

// =====================================================================================

struct Server::Impl final : core::EventSink {
  struct Gen;

  struct Conn {
    int fd = -1;
    std::string in;
    std::string out;
    size_t out_off = 0;
    http::RequestParser parser;
    bool continue_sent = false;
    bool epollout = false;
    bool close_after_flush = false;
    bool closed = false;
    bool dirty = false;
    bool preparing = false;  // a request is on a prepare thread: input waits, the Conn outlives a close
    Gen* gen = nullptr;  // the request this connection is waiting on
    std::chrono::steady_clock::time_point last_activity = std::chrono::steady_clock::now();
    explicit Conn(http::Limits lim) : parser(lim) {}
  };

  struct ToolCall {
    std::string id, name, arguments;
  };

  // One choice (`n` of them per request): its own engine request and output state.
  struct Choice {
    uint64_t engine_id = 0;
    bool done = false;
    bool echo_sent = false;
    int32_t completion_tokens = 0;
    int32_t cached_tokens = -1;  // prompt tokens the engine continued from its prefix cache (-1: not known yet)
    const char* finish = "stop";
    text::Detokenizer detok;
    text::StopMatcher stop;
    std::unique_ptr<text::OutputParser> parser;
    // Accumulators.
    std::string text, reasoning, content;
    std::vector<ToolCall> calls;
    std::vector<std::string> stream_call_ids;
    // echo + logprobs.
    std::string echo_text;
    std::vector<int32_t> prompt_ids;
    size_t text_chars = 0;  // code points of the completion text so far (text_offset)
    bool token_ids = false;  // return_tokens_as_token_ids: logprob tokens as "token_id:N"
    std::string lp_tokens, lp_values, lp_top, lp_offsets;  // completions, comma-joined JSON items
    std::string lp_content;  // chat: comma-joined content entries
    explicit Choice(const text::Tokenizer& tok) : detok(tok, true) {}
  };

  struct Gen {
    uint64_t id = 0;
    Conn* conn = nullptr;  // nullptr once the client is gone
    bool chat = false;
    bool stream = false;
    bool include_usage = false;
    bool keep_alive = true;
    bool echo = false;  // completions echo
    int32_t prompt_tokens = 0;
    int top_n = -1;
    std::string rid;
    long long created = 0;
    std::string chunk_head;  // `data: {"id"...,"choices":[{"index":` up to the index
    std::vector<std::unique_ptr<Choice>> choices;
    Value tools;  // chat: the request's tools (the parsers read their schemas)
    size_t remaining = 0;
    bool expired = false;  // a choice waited past the queue deadline: the answer is a 503
    // prompt_logprobs (an extension parameter): top alternatives per prompt
    // token (-1 = not asked), and its JSON list once the engine sent it (first choice's first event)
    int prompt_lp = -1;
    std::string prompt_lp_json;
    // tool_choice named (1) / required (2): the output JSON becomes the tool calls
    int forced = 0;
    std::string forced_name;
    bool parallel_calls = true;  // parallel_tool_calls: false keeps only the first call
    std::vector<int32_t> prompt_ids;
    std::vector<std::shared_ptr<void>> media;  // the prompt's images, held while the request lives
    // A /v1/responses request (served as chat, written back as Responses items / events)
    bool responses = false;
    responses::Meta meta;
    responses::Stream rstream;
    std::string pending_lp;  // streaming: logprob entries not yet attached to an output_text delta
  };

  // A generation request between the loop and a prepare thread: the body in,
  // the parsed request (or the error to answer) out.
  struct Prepared {
    Conn* conn = nullptr;
    bool chat = false, ka = true;
    std::string body;
    bool ok = false;
    ApiError err;
    double prep_s = 0;
    int n = 1;
    bool stream = false, include_usage = false;
    std::vector<int32_t> prompt;
    std::string prompt_text;
    bool thinking = false, tools_on = false, no_tool_parse = false, parallel_calls = true;
    int forced = 0;
    std::string forced_name;
    Value tools_val;
    core::SamplingParams sp;
    std::vector<std::string> stops;
    bool echo = false, token_ids = false;
    int explicit_prompt_lp = -1;
    std::vector<std::shared_ptr<void>> media;  // image leases (MediaHook), moved into the Gen
    bool responses = false;  // POST /v1/responses
    Value resp_request;  // its original request (echoed in the response object)
  };

  Config cfg;
  core::Engine& engine;
  const text::Tokenizer& tok;
  const text::ChatFormat* chat_fmt;
  int listen_fd = -1;
  int epfd = -1;
  int wake_fd = -1;
  int bound_port = 0;
  std::atomic<bool> stopping{false};
  std::atomic<size_t> live{0};

  // Engine -> server handoff.
  std::mutex ev_mu;
  std::vector<core::Event> ev_pending;
  bool ev_signaled = false;

  std::unordered_set<Conn*> conns;
  std::vector<Conn*> dirty;
  std::vector<Conn*> graveyard;
  std::unordered_map<uint64_t, std::unique_ptr<Gen>> gens;  // by group id
  std::unordered_map<uint64_t, std::pair<Gen*, size_t>> slots;  // engine id -> group, choice
  uint64_t next_id = 0;
  std::atomic<uint64_t> rejected{0};  // 429s: the queue was full

  std::string fingerprint;

  // Prepare threads: jobs in under prep_mu, results back under ev_mu with the engine's wake-up.
  std::vector<std::thread> prep_threads;
  std::mutex prep_mu;
  std::condition_variable prep_cv;
  std::deque<std::unique_ptr<Prepared>> prep_jobs;
  bool prep_stop = false;
  std::vector<std::unique_ptr<Prepared>> prep_done;  // under ev_mu

  // Sentinels for epoll_event.data.ptr.
  char listen_tag = 0, wake_tag = 0;

  Impl(Config c, core::Engine& e, const text::Tokenizer& t, const text::ChatFormat* f)
      : cfg(std::move(c)), engine(e), tok(t), chat_fmt(f) {
    fingerprint = "fp_" + random_hex(16);
    think_end = tok.token_id("</think>");
    im_start = tok.token_id("<|im_start|>");
    if (im_start < 0) im_start = tok.token_id("<|assistant|>");  // role-marker templates (flint): the generation prompt's header
  }

  // Structured output: the vocabulary trie is built on first use; compiled grammars are
  // shared by schema text (and the token they start after), up to 64.
  std::shared_ptr<const text::ConstraintVocab> cvocab;
  std::unordered_map<std::string, std::shared_ptr<const core::ConstraintSpec>> specs;
  int32_t think_end = -1;
  int32_t im_start = -1;  // a chat prompt's last message header (<|im_start|>, flint: <|assistant|>) = the prefix cache's mark
  core::Histogram prepare_hist;  // loop thread only

  std::mutex spec_mu;  // cvocab and specs: the prepare threads share them
  std::shared_ptr<const core::ConstraintSpec> constraint_for(const Value& schema, bool raw, int32_t start_after, std::string& err) {
    std::lock_guard<std::mutex> lk(spec_mu);
    if (!cvocab) {
      const int64_t v = cfg.vocab > 0 ? cfg.vocab : tok.vocab_size();
      cvocab = text::ConstraintVocab::build(tok, v, cfg.eos_ids);
    }
    std::string key = (raw ? "c" : "s") + std::to_string(start_after) + ":" + json::dump(schema);
    if (const auto it = specs.find(key); it != specs.end()) return it->second;
    auto spec = text::compile_json_constraint(schema, raw, cvocab, start_after, err);
    if (!spec) return nullptr;
    if (specs.size() >= 64) specs.clear();
    specs.emplace(std::move(key), spec);
    return spec;
  }

  ~Impl() override {
    stop_prepare();
    for (Conn* c : conns) {
      if (c->fd >= 0) ::close(c->fd);
      delete c;
    }
    for (Conn* c : graveyard) delete c;
    if (listen_fd >= 0) ::close(listen_fd);
    if (wake_fd >= 0) ::close(wake_fd);
    if (epfd >= 0) ::close(epfd);
  }

  // ------------------------------------------------------------------ setup

  bool listen(std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(cfg.port);
    if (int rc = getaddrinfo(cfg.host.empty() ? nullptr : cfg.host.c_str(), port.c_str(), &hints, &res); rc != 0) {
      err = std::string("getaddrinfo: ") + gai_strerror(rc);
      return false;
    }
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
      const int fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, ai->ai_protocol);
      if (fd < 0) continue;
      const int one = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
      if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, cfg.backlog) == 0) {
        listen_fd = fd;
        break;
      }
      err = std::string("bind/listen: ") + std::strerror(errno);
      ::close(fd);
    }
    freeaddrinfo(res);
    if (listen_fd < 0) {
      if (err.empty()) err = "no usable address";
      return false;
    }
    sockaddr_storage ss{};
    socklen_t len = sizeof(ss);
    getsockname(listen_fd, reinterpret_cast<sockaddr*>(&ss), &len);
    bound_port = ss.ss_family == AF_INET6 ? ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port)
                                          : ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);

    epfd = epoll_create1(EPOLL_CLOEXEC);
    wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (epfd < 0 || wake_fd < 0) {
      err = std::string("epoll/eventfd: ") + std::strerror(errno);
      return false;
    }
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.ptr = &listen_tag;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);
    ev.data.ptr = &wake_tag;
    epoll_ctl(epfd, EPOLL_CTL_ADD, wake_fd, &ev);
    return true;
  }

  void wake() {
    const uint64_t one = 1;
    [[maybe_unused]] ssize_t r = ::write(wake_fd, &one, sizeof(one));
  }

  // Engine thread.
  void on_events(const core::Event* evs, size_t n) override {
    bool signal = false;
    {
      std::lock_guard<std::mutex> lk(ev_mu);
      ev_pending.insert(ev_pending.end(), evs, evs + n);
      if (!ev_signaled) {
        ev_signaled = true;
        signal = true;
      }
    }
    if (signal) wake();
  }

  // ------------------------------------------------------------------ loop

  void run() {
    for (int i = 0; i < cfg.prepare_threads; ++i) prep_threads.emplace_back([this] { prepare_loop(); });
    std::vector<epoll_event> events(256);
    std::vector<core::Event> batch;
    while (!stopping.load(std::memory_order_relaxed)) {
      const int n = epoll_wait(epfd, events.data(), static_cast<int>(events.size()), 1000);
      if (n < 0) {
        if (errno == EINTR) continue;
        std::perror("epoll_wait");
        break;
      }
      for (int i = 0; i < n; ++i) {
        void* tag = events[i].data.ptr;
        const uint32_t e = events[i].events;
        if (tag == &listen_tag) {
          accept_all();
        } else if (tag == &wake_tag) {
          uint64_t v;
          [[maybe_unused]] ssize_t r = ::read(wake_fd, &v, sizeof(v));  // clear before taking the batch
          std::vector<std::unique_ptr<Prepared>> ready;
          {
            std::lock_guard<std::mutex> lk(ev_mu);
            batch.swap(ev_pending);
            ready.swap(prep_done);
            ev_signaled = false;
          }
          for (const core::Event& ev : batch) on_engine_event(ev);
          batch.clear();
          for (auto& p : ready) prepared(*p);
        } else {
          Conn* c = static_cast<Conn*>(tag);
          if (c->closed) continue;
          if (e & EPOLLIN) on_readable(c);
          if (!c->closed && (e & (EPOLLHUP | EPOLLERR))) close_conn(c);
          if (!c->closed && (e & EPOLLOUT)) mark_dirty(c);
        }
      }
      for (Conn* c : dirty) {
        c->dirty = false;
        if (!c->closed) flush(c);
      }
      dirty.clear();
      if (cfg.idle_timeout_s > 0) sweep_idle();
      for (Conn* c : graveyard) delete c;
      graveyard.clear();
      live.store(gens.size(), std::memory_order_relaxed);
    }
    for (Conn* c : std::vector<Conn*>(conns.begin(), conns.end())) close_conn(c);
    stop_prepare();
    for (Conn* c : graveyard) delete c;
    graveyard.clear();
  }

  // Joins the prepare threads; connections still waiting on one are closed by now and deleted here.
  void stop_prepare() {
    {
      std::lock_guard<std::mutex> lk(prep_mu);
      prep_stop = true;
    }
    prep_cv.notify_all();
    for (auto& t : prep_threads) t.join();
    prep_threads.clear();
    std::vector<std::unique_ptr<Prepared>> left;
    for (auto& p : prep_jobs) left.push_back(std::move(p));
    prep_jobs.clear();
    {
      std::lock_guard<std::mutex> lk(ev_mu);
      for (auto& p : prep_done) left.push_back(std::move(p));
      prep_done.clear();
    }
    for (auto& p : left)
      if (p->conn && p->conn->closed) delete p->conn;
  }

  // Prepare thread.
  void prepare_loop() {
    while (true) {
      std::unique_ptr<Prepared> p;
      {
        std::unique_lock<std::mutex> lk(prep_mu);
        prep_cv.wait(lk, [&] { return prep_stop || !prep_jobs.empty(); });
        if (prep_stop) return;
        p = std::move(prep_jobs.front());
        prep_jobs.pop_front();
      }
      p->ok = prepare(*p);
      bool signal = false;
      {
        std::lock_guard<std::mutex> lk(ev_mu);
        prep_done.push_back(std::move(p));
        if (!ev_signaled) signal = ev_signaled = true;
      }
      if (signal) wake();
    }
  }

  void start_generation(Conn* c, http::Request& req, bool chat, bool responses = false) {
    auto p = std::make_unique<Prepared>();
    p->conn = c;
    p->chat = chat;
    p->responses = responses;
    p->ka = req.keep_alive;
    p->body = std::move(req.body);
    if (prep_threads.empty()) {  // prepare_threads 0 (or before run): inline on the loop thread
      p->ok = prepare(*p);
      if (p->ok) commit(c, *p);
      else send_error(c, p->err, p->ka);
      return;
    }
    c->preparing = true;
    {
      std::lock_guard<std::mutex> lk(prep_mu);
      prep_jobs.push_back(std::move(p));
    }
    prep_cv.notify_one();
  }

  // Loop thread: a prepare thread finished a request.
  void prepared(Prepared& p) {
    Conn* c = p.conn;
    c->preparing = false;
    if (c->closed) {  // the client left meanwhile; close_conn left the Conn to us
      delete c;
      return;
    }
    if (p.ok) commit(c, p);
    else send_error(c, p.err, p.ka);
    if (!p.ka && !c->gen) c->close_after_flush = true;
    mark_dirty(c);
    process_input(c);  // a pipelined request may be waiting
  }

  void accept_all() {
    while (true) {
      const int fd = ::accept4(listen_fd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EINTR) continue;
        return;  // EAGAIN, or EMFILE and friends: retried on the next readiness
      }
      const int one = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      Conn* c = new Conn(cfg.limits);
      c->fd = fd;
      epoll_event ev{};
      ev.events = EPOLLIN | EPOLLRDHUP;
      ev.data.ptr = c;
      if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) != 0) {
        ::close(fd);
        delete c;
        continue;
      }
      conns.insert(c);
    }
  }

  // Close keep-alive connections that have been silent for `idle_timeout_s.
  void sweep_idle() {
    const auto now = std::chrono::steady_clock::now();
    std::vector<Conn*> dead;
    for (Conn* c : conns) {
      if (c->gen || c->preparing || !c->out.empty()) continue;  // a running, preparing or unflushed response keeps it
      const auto idle = std::chrono::duration_cast<std::chrono::seconds>(now - c->last_activity).count();
      if (idle >= cfg.idle_timeout_s) dead.push_back(c);
    }
    for (Conn* c : dead) close_conn(c);
  }

  void on_readable(Conn* c) {
    c->last_activity = std::chrono::steady_clock::now();
    bool eof = false;
    size_t budget = 4u << 20;  // per wake-up, so one uploader cannot starve the rest
    while (budget > 0) {
      const size_t old = c->in.size();
      constexpr size_t kChunk = 64 * 1024;
      c->in.resize(old + kChunk);
      const ssize_t r = ::recv(c->fd, c->in.data() + old, kChunk, 0);
      if (r > 0) {
        c->in.resize(old + static_cast<size_t>(r));
        budget -= std::min(budget, static_cast<size_t>(r));
        if (static_cast<size_t>(r) < kChunk) break;
        continue;
      }
      c->in.resize(old);
      if (r == 0) eof = true;
      else if (errno == EINTR) continue;
      else if (errno != EAGAIN && errno != EWOULDBLOCK) eof = true;
      break;
    }
    // Input queued behind a running request is bounded too.
    if (c->in.size() > cfg.limits.max_body_bytes + cfg.limits.max_header_bytes) {
      close_conn(c);
      return;
    }
    if (eof) {
      // A peer that closes (or half-closes) is treated as gone: its request is
      // cancelled so its KV is freed now.
      close_conn(c);
      return;
    }
    process_input(c);
  }

  void process_input(Conn* c) {
    while (!c->closed && !c->gen && !c->preparing && !c->close_after_flush && !c->in.empty()) {
      const auto st = c->parser.feed(c->in);
      if (st == http::RequestParser::Status::need_more) {
        if (c->parser.wants_continue() && !c->continue_sent) {
          c->out += "HTTP/1.1 100 Continue\r\n\r\n";
          c->continue_sent = true;
          mark_dirty(c);
        }
        return;
      }
      if (st == http::RequestParser::Status::error) {
        ApiError e{c->parser.error_status(), c->parser.error_message()};
        send_json(c, e.status, error_body(e), false);
        c->close_after_flush = true;
        return;
      }
      http::Request req = std::move(c->parser.request());
      c->in.erase(0, c->parser.consumed());
      c->parser.reset();
      c->continue_sent = false;
      const bool ka = req.keep_alive;
      handle(c, req);
      // an answer written right away (no generation running) to a Connection: close request
      // closes once it is flushed; a generation closes in response_done
      if (!ka && !c->gen && !c->preparing) c->close_after_flush = true;
    }
  }

  void mark_dirty(Conn* c) {
    if (!c->dirty) {
      c->dirty = true;
      dirty.push_back(c);
    }
  }

  void set_epollout(Conn* c, bool on) {
    if (c->epollout == on) return;
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP | (on ? EPOLLOUT : 0u);
    ev.data.ptr = c;
    epoll_ctl(epfd, EPOLL_CTL_MOD, c->fd, &ev);
    c->epollout = on;
  }

  void flush(Conn* c) {
    while (c->out_off < c->out.size()) {
      const ssize_t w = ::send(c->fd, c->out.data() + c->out_off, c->out.size() - c->out_off, MSG_NOSIGNAL);
      if (w > 0) {
        c->out_off += static_cast<size_t>(w);
        continue;
      }
      if (w < 0 && errno == EINTR) continue;
      if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (c->out.size() - c->out_off > cfg.max_output_buffer) {
          close_conn(c);
          return;
        }
        set_epollout(c, true);
        return;
      }
      close_conn(c);
      return;
    }
    c->out.clear();
    c->out_off = 0;
    set_epollout(c, false);
    if (c->close_after_flush && !c->gen && !c->preparing) close_conn(c);
  }

  void close_conn(Conn* c) {
    if (c->closed) return;
    c->closed = true;
    if (Gen* g = c->gen) {
      g->conn = nullptr;
      for (auto& ch : g->choices)
        if (!ch->done) engine.cancel(ch->engine_id);
      c->gen = nullptr;
    }
    epoll_ctl(epfd, EPOLL_CTL_DEL, c->fd, nullptr);
    ::close(c->fd);
    c->fd = -1;
    conns.erase(c);
    if (!c->preparing) graveyard.push_back(c);  // else deleted when its prepare thread is done
  }

  // Called when the connection's current response is complete.
  void response_done(Conn* c, bool keep_alive) {
    c->gen = nullptr;
    c->last_activity = std::chrono::steady_clock::now();
    if (!keep_alive) c->close_after_flush = true;
    mark_dirty(c);
    process_input(c);  // a pipelined request may be waiting
  }

  std::string cors() const {
    if (cfg.cors_origin.empty()) return {};
    return "Access-Control-Allow-Origin: " + cfg.cors_origin + "\r\n";
  }

  std::string cors_preflight() const {
    if (cfg.cors_origin.empty()) return {};
    return "Access-Control-Allow-Origin: " + cfg.cors_origin +
           "\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
           "Access-Control-Allow-Headers: Authorization, Content-Type, OpenAI-Beta\r\n"
           "Access-Control-Max-Age: 86400\r\n";
  }

  void send_json(Conn* c, int status, const std::string& body, bool keep_alive, bool head_only = false) {
    http::write_head(c->out, status, "application/json", body.size(), keep_alive, cors());
    if (!head_only) c->out += body;
    mark_dirty(c);
  }

  void send_error(Conn* c, const ApiError& e, bool keep_alive) { send_json(c, e.status, error_body(e), keep_alive); }

  // ------------------------------------------------------------------ routes

  void handle(Conn* c, http::Request& req) {
    const bool ka = req.keep_alive;
    const std::string& p = req.path;
    const bool get = req.method == "GET";
    const bool head = req.method == "HEAD";
    const bool post = req.method == "POST";
    if (req.method == "OPTIONS") {  // CORS preflight: 204, no body
      http::write_head(c->out, 204, "", 0, ka, cors_preflight());
      mark_dirty(c);
      return;
    }
    if (p == "/v1/chat/completions" || p == "/v1/completions") {
      if (!post) return send_error(c, {405, "Method Not Allowed", "MethodNotAllowed"}, ka);
      return start_generation(c, req, p == "/v1/chat/completions");
    }
    if (p == "/v1/responses") {  // stateless Responses API over the chat path
      if (!post) return send_error(c, {405, "Method Not Allowed", "MethodNotAllowed"}, ka);
      return start_generation(c, req, true, true);
    }
    if (p.rfind("/v1/responses/", 0) == 0) {  // retrieve / cancel / delete: nothing is stored
      const std::string id = p.substr(std::strlen("/v1/responses/"));
      return send_error(c, {404, "Response with id '" + id.substr(0, id.find('/')) + "' not found.", "NotFoundError", "response_id"}, ka);
    }
    if (!get && !head) {
      if (p == "/health" || p == "/ping" || p == "/v1/models" || p == "/metrics" || p == "/version" ||
          p.rfind("/v1/models/", 0) == 0)
        return send_error(c, {405, "Method Not Allowed", "MethodNotAllowed"}, ka);
      return send_error(c, {404, "Not Found", "NotFoundError"}, ka);
    }
    if (p == "/health" || p == "/ping") {
      http::write_head(c->out, 200, "", 0, ka, cors());
      mark_dirty(c);
      return;
    }
    if (p == "/version") return send_json(c, 200, std::string("{\"version\":\"") + kVersion + "\"}", ka, head);
    if (p == "/v1/models") return send_json(c, 200, models_body(), ka, head);
    if (p.rfind("/v1/models/", 0) == 0) {  // retrieve one model
      const std::string id = p.substr(std::strlen("/v1/models/"));
      if (id != engine.model().name)
        return send_error(c, {404, "The model `" + id + "` does not exist.", "NotFoundError", "model"}, ka);
      return send_json(c, 200, model_body(), ka, head);
    }
    if (p == "/metrics") {
      const std::string body = metrics_body();
      http::write_head(c->out, 200, "text/plain; version=0.0.4; charset=utf-8", body.size(), ka, cors());
      if (!head) c->out += body;
      mark_dirty(c);
      return;
    }
    send_error(c, {404, "Not Found", "NotFoundError"}, ka);
  }

  std::string model_body() const {
    const auto& m = engine.model();
    std::string s = "{\"id\":";
    json::write_string(m.name, s);
    s += ",\"object\":\"model\",\"created\":";
    append_int(s, static_cast<long long>(std::time(nullptr)));
    s += ",\"owned_by\":\"xenithium\",\"root\":";
    json::write_string(m.name, s);
    s += ",\"parent\":null,\"max_model_len\":";
    append_int(s, m.max_model_len);  // the longest ONE request (prompt + output): the context per slot
    // The KV the requests running at once share, read live (qwen35's auto pool is re-sized after the warm-up). A shared
    // pool: its tokens (a request takes pages as it grows; slots x max_model_len may exceed it). Fixed per slot: every
    // slot owns max_model_len tokens. slots 0 (a stub without a limit): no slot or pool fields.
    const core::Stats st = engine.stats();
    if (st.slots > 0) {
      const bool pool = st.kv_pool_tokens > 0;
      s += ",\"slots\":";
      append_int(s, st.slots);
      s += ",\"kv_layout\":";
      s += pool ? "\"pool\"" : "\"per_slot\"";
      s += ",\"kv_pool_tokens\":";
      append_int(s, pool ? static_cast<long long>(st.kv_pool_tokens) : static_cast<long long>(st.slots) * m.max_model_len);
    }
    // What the model runs on (benchmarks record it): the tensor-parallel width and the cards in rank order.
    if (cfg.tensor_parallel > 0) {
      s += ",\"tensor_parallel\":";
      append_int(s, cfg.tensor_parallel);
    }
    if (!cfg.devices.empty()) {
      s += ",\"devices\":[";
      for (size_t d = 0; d < cfg.devices.size(); ++d) {
        s += d ? ",{\"name\":" : "{\"name\":";
        json::write_string(cfg.devices[d].name, s);
        s += ",\"memory_bytes\":";
        append_int(s, static_cast<long long>(cfg.devices[d].memory_bytes));
        if (cfg.devices[d].memory_gb > 0) {
          s += ",\"memory_gb\":";
          append_int(s, cfg.devices[d].memory_gb);
        }
        if (!cfg.devices[d].pci.empty()) {
          s += ",\"pci\":";
          json::write_string(cfg.devices[d].pci, s);
        }
        if (!cfg.devices[d].driver.empty()) {
          s += ",\"driver\":";
          json::write_string(cfg.devices[d].driver, s);
        }
        s += "}";
      }
      s += "]";
    }
    // What it runs and how it was started (benchmarks show it, so others can run the same).
    for (const auto &[key, value] : {std::pair{",\"build\":", &cfg.build}, {",\"oneapi\":", &cfg.oneapi}, {",\"drafter\":", &cfg.drafter}})
      if (!value->empty()) {
        s += key;
        json::write_string(*value, s);
      }
    if (!cfg.machine.kernel.empty()) {
      const util::Machine &mc = cfg.machine;
      s += ",\"machine\":{\"os\":";
      json::write_string(mc.os, s);
      s += ",\"kernel\":";
      json::write_string(mc.kernel, s);
      s += ",\"cpu\":";
      json::write_string(mc.cpu, s);
      s += ",\"cpu_threads\":";
      append_int(s, mc.cpu_threads);
      s += ",\"memory_gib\":";
      append_int(s, mc.memory_gib);
      s += "}";
    }
    if (!cfg.launch_args.empty() || !cfg.launch_env.empty()) {
      s += ",\"launch\":{\"args\":[";
      for (size_t a = 0; a < cfg.launch_args.size(); ++a) {
        if (a) s += ",";
        json::write_string(cfg.launch_args[a], s);
      }
      s += "],\"env\":{";
      for (size_t e = 0; e < cfg.launch_env.size(); ++e) {
        if (e) s += ",";
        json::write_string(cfg.launch_env[e].first, s);
        s += ":";
        json::write_string(cfg.launch_env[e].second, s);
      }
      s += "}}";
    }
    s += "}";
    return s;
  }

  std::string models_body() const {
    std::string s = "{\"object\":\"list\",\"data\":[";
    s += model_body();
    s += "]}";
    return s;
  }

  std::string metrics_body() const {
    const core::Stats st = engine.stats();
    std::string label = "{model_name=";
    json::write_string(engine.model().name, label);
    label += "}";
    std::string s;
    auto metric = [&](const char* name, const char* type, const char* help, double v) {
      s += "# HELP ";
      s += name;
      s += " ";
      s += help;
      s += "\n# TYPE ";
      s += name;
      s += " ";
      s += type;
      s += "\n";
      s += name;
      s += label;
      char b[48];
      std::snprintf(b, sizeof(b), " %.17g\n", v);
      s += b;
    };
    metric("vllm:num_requests_running", "gauge", "Requests in decode.", st.running);
    metric("vllm:num_requests_waiting", "gauge", "Requests waiting to be scheduled.", st.waiting);
    if (st.kv_pool_tokens > 0) {  // the shared KV pool
      metric("vllm:kv_cache_usage_perc", "gauge", "KV pool usage (1 = full).", static_cast<double>(st.kv_used_tokens) / static_cast<double>(st.kv_pool_tokens));
      metric("vllm:num_preemptions_total", "counter", "Requests preempted to free KV pages.", static_cast<double>(st.preemptions));
      metric("xenithium:kv_pool_tokens", "gauge", "Tokens of KV in the shared pool.", static_cast<double>(st.kv_pool_tokens));
      metric("xenithium:kv_used_tokens", "gauge", "Pool tokens held by running requests (and parked prefix-cache sequences).", static_cast<double>(st.kv_used_tokens));
    }
    // the prefix cache (tokens: the prompts looked up, the prompt tokens continued from cached states)
    metric("vllm:prefix_cache_queries_total", "counter", "Prompt tokens of the requests looked up in the prefix cache.", static_cast<double>(st.prefix_queries));
    metric("vllm:prefix_cache_hits_total", "counter", "Prompt tokens continued from the prefix cache instead of prefilled.", static_cast<double>(st.prefix_hits));
    metric("xenithium:prefix_cache_snapshots", "gauge", "Sequence-state snapshots the prefix cache holds.", static_cast<double>(st.prefix_snapshots));
    metric("xenithium:prefix_cache_live_total", "counter", "Requests continued from a parked slot's live state.", static_cast<double>(st.prefix_live));
    metric("xenithium:prefix_cache_restores_total", "counter", "Requests continued from a snapshot on its own slot.", static_cast<double>(st.prefix_restores));
    metric("xenithium:prefix_cache_forks_total", "counter", "Requests continued from a snapshot with the KV copied to another slot.", static_cast<double>(st.prefix_forks));
    metric("xenithium:prefix_cache_holds_total", "counter", "Requests that waited for a running request to prefill their common prefix.", static_cast<double>(st.prefix_holds));
    metric("xenithium:kv_lend_park_drops_total", "counter", "Parked sequences dropped by the KV-lending park timeout (their pages lent to experts again).",
           static_cast<double>(st.lend_park_drops));
    metric("xenithium:kv_lend_pressure_drops_total", "counter", "Parked sequences dropped so a reservation keeps the lent experts.",
           static_cast<double>(st.lend_pressure_drops));
    metric("vllm:prompt_tokens_total", "counter", "Prefill tokens processed.", static_cast<double>(st.prompt_tokens));
    metric("vllm:generation_tokens_total", "counter", "Tokens generated.", static_cast<double>(st.generation_tokens));
    metric("vllm:spec_decode_num_drafts_total", "counter", "Speculative steps, one per request.",
           static_cast<double>(st.spec_drafts));
    metric("vllm:spec_decode_num_accepted_tokens_total", "counter", "Accepted draft tokens.",
           static_cast<double>(st.spec_accepted_tokens));
    metric("vllm:spec_decode_num_draft_tokens_total", "counter", "Proposed draft tokens.",
           static_cast<double>(st.spec_draft_tokens));
    metric("xenithium:lookup_steps_total", "counter", "Speculative steps drafted by lookup (XEN_LOOKUP), one per request.",
           static_cast<double>(st.lookup_steps));
    metric("xenithium:lookup_accepted_tokens_total", "counter", "Accepted lookup draft tokens.",
           static_cast<double>(st.lookup_accepted_tokens));
    {  // per draft position: [i] / [i - 1] is position i's conditional acceptance
      const char* name = "vllm:spec_decode_num_accepted_tokens_per_pos_total";
      s += "# HELP ";
      s += name;
      s += " Accepted draft tokens by draft position.\n# TYPE ";
      s += name;
      s += " counter\n";
      // depth = draft tokens per speculative step (the stub engine reports none: no lines)
      const int npos = st.spec_drafts ? static_cast<int>(std::min<uint64_t>(core::Stats::kSpecPositions,
                                                                            st.spec_draft_tokens / st.spec_drafts))
                                      : 0;
      for (int i = 0; i < npos; ++i) {
        s += name;
        s += label.substr(0, label.size() - 1);
        s += ",position=\"";
        s += std::to_string(i);
        s += "\"}";
        char b[48];
        std::snprintf(b, sizeof(b), " %.17g\n", static_cast<double>(st.spec_accepted_at[i]));
        s += b;
      }
    }
    metric("xenithium:engine_steps_total", "counter", "Engine steps.", static_cast<double>(st.steps));
    {  // host RAM as /metrics: the engine's status line and the watchdog's warning already print it; a steady climb ends in the kernel's OOM kill (exit 137)
      const util::HostMem hm = util::host_mem();
      metric("process_resident_memory_bytes", "gauge", "Resident memory of the server process (host RAM; driver-pinned buffers are not in it).",
             static_cast<double>(util::self_rss()));
      if (hm.ok) {
        metric("xenithium:host_memory_available_bytes", "gauge", "Host RAM available (MemAvailable).", static_cast<double>(hm.available));
        metric("xenithium:host_memory_total_bytes", "gauge", "Host RAM (MemTotal).", static_cast<double>(hm.total));
      }
    }
    metric("xenithium:requests_finished_total", "counter", "Requests finished.",
           static_cast<double>(st.requests_finished));
    metric("xenithium:server_live_requests", "gauge", "Requests the server is tracking.",
           static_cast<double>(gens.size()));
    metric("xenithium:request_slots", "gauge", "Requests the engine runs at once (0 = unlimited).", st.slots);
    metric("xenithium:requests_rejected_total", "counter", "Requests refused with 429 because the queue was full.",
           static_cast<double>(rejected.load(std::memory_order_relaxed)));
    {  // every request's last event by reason (queue cancels and queue-deadline expiries included)
      const char* name = "xenithium:requests_ended_total";
      s += "# HELP ";
      s += name;
      s += " Requests that ended, by finish reason.\n# TYPE ";
      s += name;
      s += " counter\n";
      for (int i = 1; i < core::Stats::kFinishKinds; ++i) {
        s += name;
        s += label.substr(0, label.size() - 1);
        s += ",finished_reason=\"";
        s += core::finish_name(static_cast<core::Finish>(i));
        s += "\"}";
        char b[48];
        std::snprintf(b, sizeof(b), " %.17g\n", static_cast<double>(st.finished_by[i]));
        s += b;
      }
    }
    auto histogram = [&](const char* name, const char* help, const core::Histogram& h) {
      s += "# HELP ";
      s += name;
      s += " ";
      s += help;
      s += "\n# TYPE ";
      s += name;
      s += " histogram\n";
      uint64_t cum = 0;
      char b[96];
      for (int i = 0; i <= core::Histogram::kBounds; ++i) {
        cum += h.bucket[i];
        s += name;
        s += "_bucket";
        s += label.substr(0, label.size() - 1);
        if (i < core::Histogram::kBounds) std::snprintf(b, sizeof(b), ",le=\"%g\"} %llu\n", core::Histogram::bounds[i],
                                                         static_cast<unsigned long long>(cum));
        else std::snprintf(b, sizeof(b), ",le=\"+Inf\"} %llu\n", static_cast<unsigned long long>(cum));
        s += b;
      }
      s += name;
      s += "_sum";
      s += label;
      std::snprintf(b, sizeof(b), " %.17g\n", h.sum);
      s += b;
      s += name;
      s += "_count";
      s += label;
      std::snprintf(b, sizeof(b), " %llu\n", static_cast<unsigned long long>(h.count));
      s += b;
    };
    histogram("vllm:request_queue_time_seconds", "Time a request waited for a slot.", st.queue_time);
    histogram("vllm:request_prefill_time_seconds", "Time from admission to the first token.", st.prefill_time);
    histogram("vllm:time_to_first_token_seconds", "Time from submission to the first token.", st.ttft);
    histogram("vllm:e2e_request_latency_seconds", "Time from submission to the last event.", st.e2e);
    histogram("xenithium:request_prepare_seconds", "Time per request to prepare it: JSON parse, checks, chat template, tokenizer, grammar (on a prepare thread since round 25).", prepare_hist);
    return s;
  }

  // ------------------------------------------------------------------ generation

  bool parse_sampling(const Value& r, bool chat, size_t prompt_len, core::SamplingParams& sp,
                      std::vector<std::string>& stops, ApiError& err) {
    const auto& model = engine.model();
    sp = model.default_sampling;
    double d;
    d = sp.temperature;
    if (!read_number(r, "temperature", d, err)) return false;
    if (d < 0) return err = {400, "temperature must be non-negative", "BadRequestError", "temperature"}, false;
    sp.temperature = static_cast<float>(d);
    d = sp.top_p;
    if (!read_number(r, "top_p", d, err)) return false;
    if (!(d > 0 && d <= 1)) return err = {400, "top_p must be in (0, 1]", "BadRequestError", "top_p"}, false;
    sp.top_p = static_cast<float>(d);
    d = sp.min_p;
    if (!read_number(r, "min_p", d, err)) return false;
    if (d < 0 || d > 1) return err = {400, "min_p must be in [0, 1]", "BadRequestError", "min_p"}, false;
    sp.min_p = static_cast<float>(d);
    d = sp.repetition_penalty;
    if (!read_number(r, "repetition_penalty", d, err)) return false;
    if (d <= 0) return err = {400, "repetition_penalty must be positive", "BadRequestError", "repetition_penalty"}, false;
    sp.repetition_penalty = static_cast<float>(d);
    d = sp.presence_penalty;
    if (!read_number(r, "presence_penalty", d, err)) return false;
    sp.presence_penalty = static_cast<float>(d);
    d = sp.frequency_penalty;
    if (!read_number(r, "frequency_penalty", d, err)) return false;
    sp.frequency_penalty = static_cast<float>(d);

    long long iv = 0;
    bool present = false;
    if (!read_int(r, "top_k", iv, present, err)) return false;
    if (present) {
      if (iv < -1) return err = {400, "top_k must be -1 (off) or positive", "BadRequestError", "top_k"}, false;
      sp.top_k = iv <= 0 ? 0 : static_cast<int32_t>(std::min<long long>(iv, 1 << 30));
    }
    if (!read_int(r, "seed", iv, present, err)) return false;
    if (present) {
      sp.has_seed = true;
      sp.seed = static_cast<uint64_t>(iv);
    }
    if (!read_bool(r, "ignore_eos", sp.ignore_eos, err)) return false;
    if (!read_logit_bias(r, tok.vocab_size(), sp.logit_bias, err)) return false;

    // max_tokens: chat may say max_completion_tokens; unset in chat = the rest of the context.
    const long long room = static_cast<long long>(model.max_model_len) - static_cast<long long>(prompt_len);
    long long mt = chat ? room : 16;
    bool have = false;
    if (chat) {
      if (!read_int(r, "max_completion_tokens", iv, present, err)) return false;
      if (present) mt = iv, have = true;
    }
    if (!have) {
      if (!read_int(r, "max_tokens", iv, present, err)) return false;
      if (present) mt = iv;
    }
    if (mt < 1) return err = {400, "max_tokens must be at least 1", "BadRequestError", "max_tokens"}, false;
    if (static_cast<long long>(prompt_len) >= model.max_model_len || mt > room) {
      err = {400,
             "This model's maximum context length is " + std::to_string(model.max_model_len) +
                 " tokens. However, you requested " + std::to_string(mt) + " output tokens and your prompt contains " +
                 std::to_string(prompt_len) + " input tokens, for a total of " +
                 std::to_string(mt + static_cast<long long>(prompt_len)) + " tokens.",
             "BadRequestError", "max_tokens"};
      return false;
    }
    sp.max_tokens = static_cast<int32_t>(mt);

    if (const Value* st = r.get("stop"); st && !st->is_null()) {
      if (st->is_string()) {
        stops.push_back(st->str);
      } else if (st->is_array()) {
        for (const Value& s : st->arr) {
          if (!s.is_string()) return err = {400, "stop must be a string or a list of strings", "BadRequestError", "stop"}, false;
          stops.push_back(s.str);
        }
      } else {
        return err = {400, "stop must be a string or a list of strings", "BadRequestError", "stop"}, false;
      }
    }
    if (const Value* st = r.get("stop_token_ids"); st && !st->is_null()) {
      if (!st->is_array()) return err = {400, "stop_token_ids must be a list of integers", "BadRequestError", "stop_token_ids"}, false;
      for (const Value& t : st->arr) {
        if (!t.is_int()) return err = {400, "stop_token_ids must be a list of integers", "BadRequestError", "stop_token_ids"}, false;
        sp.stop_token_ids.push_back(static_cast<int32_t>(t.num));
      }
    }
    // logprobs: completions take a count of top alternatives, chat a bool + top_logprobs.
    if (chat) {
      bool lp = false;
      if (!read_bool(r, "logprobs", lp, err)) return false;
      long long tl = 0;
      bool have_tl = false;
      if (!read_int(r, "top_logprobs", tl, have_tl, err)) return false;
      if (have_tl && (tl < 0 || tl > 20))
        return err = {400, "top_logprobs must be in [0, 20]", "BadRequestError", "top_logprobs"}, false;
      if (have_tl && tl > 0 && !lp)
        return err = {400, "top_logprobs needs logprobs: true", "BadRequestError", "top_logprobs"}, false;
      if (lp) sp.logprobs = have_tl ? static_cast<int32_t>(tl) : 0;
    } else if (const Value* v = r.get("logprobs"); v && !v->is_null()) {
      if (!v->is_int() || v->num < 0 || v->num > 20)
        return err = {400, "logprobs must be an integer in [0, 20]", "BadRequestError", "logprobs"}, false;
      sp.logprobs = static_cast<int32_t>(v->num);
    }
    if (!read_int(r, "prompt_logprobs", iv, present, err)) return false;
    if (present) {
      if (iv < 0 || iv > 20) return err = {400, "prompt_logprobs must be in [0, 20]", "BadRequestError", "prompt_logprobs"}, false;
      sp.prompt_logprobs = static_cast<int32_t>(iv);
    }
    return true;
  }

  // Field checks shared by both endpoints (and the features that are deliberately not
  // implemented, rejected instead of silently ignored).
  bool check_request(const Value& r, bool chat, int& n, ApiError& err) {
    long long lv = 1;
    bool present;
    if (!read_int(r, "n", lv, present, err)) return false;
    if (lv < 1 || static_cast<size_t>(lv) > kMaxChoices)
      return err = {400, "n must be in [1, " + std::to_string(kMaxChoices) + "]", "BadRequestError", "n"}, false;
    n = static_cast<int>(lv);
    if (const Value* v = r.get("best_of"); v && !v->is_null()) {
      long long bo = 1;
      if (!read_int(r, "best_of", bo, present, err)) return false;
      if (present && bo != lv) return err = {400, "best_of must equal n (no beam search)", "BadRequestError", "best_of"}, false;
    }
    if (const Value* v = r.get("echo"); chat && v && v->is_bool() && v->b)
      return err = {400, "echo is a /v1/completions option", "BadRequestError", "echo"}, false;
    if (chat) {
      if (const Value* v = r.get("continue_final_message"); v && v->is_bool() && v->b)
        return err = {400, "continue_final_message is not supported", "BadRequestError", "continue_final_message"}, false;
    }
    return true;
  }

  // Everything about a request that needs neither the connection nor the loop's state: JSON parse,
  // checks, chat template, tokenizer, grammar. Runs on a prepare thread, or
  // inline with prepare_threads 0. False: p.err is the answer.
  bool prepare(Prepared& p) {
    const auto t_prep = std::chrono::steady_clock::now();
    const bool chat = p.chat;
    auto send_error = [&p](const ApiError& e) {
      p.err = e;
      return false;
    };
    Value r;
    std::string perr;
    if (!json::parse(p.body, r, &perr) || !r.is_object())
      return send_error({400, perr.empty() ? "request body must be a JSON object" : "Invalid JSON: " + perr});
    p.body.clear();
    p.body.shrink_to_fit();
    if (p.responses) {  // the chat request the Responses request means
      Value chat_req;
      std::string rerr, rparam;
      if (!responses::to_chat(r, chat_req, rerr, rparam)) return send_error({400, rerr, "BadRequestError", rparam});
      p.resp_request = std::move(r);
      r = std::move(chat_req);
    }

    const auto& model = engine.model();
    if (const std::string* m = r.get_string("model"); m && !m->empty() && *m != model.name)
      return send_error({404, "The model `" + *m + "` does not exist.", "NotFoundError", "model"});

    ApiError err;
    int& n = p.n;
    if (!check_request(r, chat, n, err)) return send_error(err);

    bool& stream = p.stream;
    bool& include_usage = p.include_usage;
    if (!read_bool(r, "stream", stream, err)) return send_error(err);
    if (const Value* so = r.get_object("stream_options")) {
      if (!read_bool(*so, "include_usage", include_usage, err)) return send_error(err);
    }

    // Structured output: the grammar is compiled once per schema and shared between
    // requests; each choice gets its own walk through it.
    Value constraint_schema;
    bool constrained = false, raw_choice = false;
    if (!parse_response_format(r, constraint_schema, constrained, raw_choice, err)) return send_error(err);

    auto& prompt = p.prompt;
    auto& prompt_text = p.prompt_text;
    bool& thinking = p.thinking;
    bool& tools_on = p.tools_on;
    bool& no_tool_parse = p.no_tool_parse;
    bool& parallel_calls = p.parallel_calls;
    int& forced = p.forced;  // tool_choice 1 = a named function, 2 = required: the answer is the call(s) as JSON
    std::string& forced_name = p.forced_name;
    Value& tools_val = p.tools_val;  // the parser keeps a pointer to the schemas: moved into the Gen
    if (chat) {
      if (!chat_fmt) return send_error({400, "this model has no chat format; use /v1/completions"});
      const Value* messages = r.get("messages");
      if (!messages) return send_error({400, "messages is required", "BadRequestError", "messages"});
      // request-level fields override chat_template_kwargs unless null / "auto"; a top-level
      // reasoning_effort also sets enable_thinking = (effort != "none") when the user did not
      // pass enable_thinking. The output parser sees the same merged kwargs.
      text::ChatOptions opt;
      const Value* kw = r.get_object("chat_template_kwargs");
      const Value* et = kw ? kw->get("enable_thinking") : nullptr;
      opt.reasoning_effort = kw ? kw->get("reasoning_effort") : nullptr;
      static const Value kTrue = Value::boolean(true), kFalse = Value::boolean(false);
      if (const Value* top = r.get("reasoning_effort"); top && !top->is_null() && !(top->is_string() && top->str == "auto")) {
        opt.reasoning_effort = top;
        if (!et) et = (top->is_string() && top->str == "none") ? &kFalse : &kTrue;
      }
      opt.enable_thinking = text::flag_of(et);
      opt.preserve_thinking = text::flag_of(kw ? kw->get("preserve_thinking") : nullptr);
      opt.kwargs = kw;
      if (!read_bool(r, "add_generation_prompt", opt.add_generation_prompt, err)) return send_error(err);

      std::string choice_mode, choice_name;
      if (!parse_tool_choice(r, choice_mode, choice_name, err)) return send_error(err);
      const Value* tools = r.get("tools");
      Value filtered;
      if (tools && tools->is_array() && !tools->arr.empty()) {
        if (choice_mode == "none") {
          // the tools are rendered anyway and only the tool parser is skipped; a format that excludes the tools
          // on tool_choice "none" (Flint) renders none
          if (chat_fmt->exclude_tools_on_none()) tools = nullptr;
        } else if (choice_mode == "function") {
          bool found = false;
          filtered = Value::array();
          for (const Value& t : tools->arr) {
            const Value* f = t.get_object("function");
            const std::string* tn = f ? f->get_string("name") : nullptr;
            if (tn && *tn == choice_name) {
              filtered.push(t);
              found = true;
            }
          }
          if (!found)
            return send_error({400,
                               "Invalid value for 'tool_choice': function \"" + choice_name +
                                   "\" is not in the request's tools.",
                               "BadRequestError", "tool_choice"});
          tools = &filtered;
        }
      }
      // The image_url parts in the order the template prints them (messages, then parts), and the video_url
      // parts (their own placeholder token, their own order)
      std::vector<const std::string*> image_urls, video_urls;
      if (messages->is_array())
        for (const Value& msg : messages->arr) {
          const Value* content = msg.is_object() ? msg.get("content") : nullptr;
          if (!content || !content->is_array()) continue;
          for (const Value& part : content->arr) {
            if (!part.is_object()) continue;
            const std::string* type = part.get_string("type");
            const bool video = type ? *type == "video_url" : !part.get("image_url") && part.get("video_url");
            if (!video && (type ? *type != "image_url" && *type != "input_image" : !part.get("image_url"))) continue;
            const Value* iu = part.get(video ? "video_url" : "image_url");
            const std::string* url = iu && iu->is_string() ? &iu->str : iu && iu->is_object() ? iu->get_string("url") : nullptr;
            if (url) (video ? video_urls : image_urls).push_back(url);
          }
        }
      if ((!image_urls.empty() || !video_urls.empty()) && !cfg.media)
        return send_error({400, std::string(image_urls.empty() ? "video" : "image") +
                                    " input is off on this server: start it with --vision on (XEN_VISION=1) on a model with a vision tower",
                           "BadRequestError", "messages"});
      opt.images = cfg.media != nullptr;
      std::string rendered, terr;
      if (!chat_fmt->render(*messages, tools, opt, rendered, terr)) return send_error({400, terr});
      prompt = tok.encode(rendered);
      if (!image_urls.empty() || !video_urls.empty()) {  // each placeholder token -> the image's / the video frames' tokens
        std::vector<server::MediaHook::Tokens> imgs;
        for (size_t k = 0; k < image_urls.size(); ++k) {
          std::string bytes, ierr;
          if (!util::load_media_url(*image_urls[k], cfg.fetch, bytes, ierr)) return send_error({400, "image " + std::to_string(k + 1) + ": " + ierr, "BadRequestError", "messages"});
          util::Image im;
          if (!util::decode_image(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), im, ierr))
            return send_error({400, "image " + std::to_string(k + 1) + ": " + ierr, "BadRequestError", "messages"});
          server::MediaHook::Tokens t;
          int status = 400;
          if (!cfg.media->add_image(im, t, status, ierr))
            return send_error({status, "image " + std::to_string(k + 1) + ": " + ierr, status == 503 ? "ServiceUnavailableError" : "BadRequestError", "messages"});
          imgs.push_back(std::move(t));
        }
        // a video's expansion = per frame: its timestamp's tokens, <|vision_start|>, its ids, <|vision_end|>
        std::vector<std::vector<int32_t>> vids;
        for (size_t k = 0; k < video_urls.size(); ++k) {
          std::string bytes, verr;
          if (!util::load_media_url(*video_urls[k], cfg.fetch, bytes, verr)) return send_error({400, "video " + std::to_string(k + 1) + ": " + verr, "BadRequestError", "messages"});
          std::vector<server::MediaHook::Tokens> frames;
          std::vector<std::string> stamps;
          int status = 400;
          if (!cfg.media->add_video(bytes, frames, stamps, status, verr))
            return send_error({status, "video " + std::to_string(k + 1) + ": " + verr, status == 503 ? "ServiceUnavailableError" : "BadRequestError", "messages"});
          std::vector<int32_t> ids;
          for (size_t f = 0; f < frames.size(); ++f) {
            if (f < stamps.size())
              for (int32_t t : tok.encode(stamps[f])) ids.push_back(t);
            ids.push_back(cfg.media->vision_start());
            for (int i = 0; i < frames[f].count; ++i) ids.push_back(static_cast<int32_t>(frames[f].base + i));
            ids.push_back(cfg.media->vision_end());
            p.media.push_back(std::move(frames[f].lease));
          }
          vids.push_back(std::move(ids));
        }
        const int32_t pad = cfg.media->image_token(), vpad = cfg.media->video_token();
        size_t found = 0, vfound = 0;
        for (int32_t t : prompt) found += t == pad, vfound += vpad >= 0 && t == vpad;
        if (found != imgs.size() || vfound != vids.size())
          return send_error({400, "the rendered prompt has " + std::to_string(found) + " image and " + std::to_string(vfound) + " video placeholders for " +
                                      std::to_string(imgs.size()) + " images and " + std::to_string(vids.size()) + " videos",
                             "BadRequestError", "messages"});
        std::vector<int32_t> expanded;
        size_t k = 0, j = 0;
        for (int32_t t : prompt) {
          if (t == pad) {
            for (int i = 0; i < imgs[k].count; ++i) expanded.push_back(static_cast<int32_t>(imgs[k].base + i));
            p.media.push_back(std::move(imgs[k].lease));
            ++k;
          } else if (vpad >= 0 && t == vpad) {
            expanded.insert(expanded.end(), vids[j].begin(), vids[j].end());
            ++j;
          } else {
            expanded.push_back(t);
          }
        }
        prompt = std::move(expanded);
      }
      thinking = chat_fmt->thinking(et);
      tools_on = tools && tools->is_array() && !tools->arr.empty();
      no_tool_parse = tools_on && choice_mode == "none";
      if (tools_on) tools_val = *tools;
      if (!read_bool(r, "parallel_tool_calls", parallel_calls, err)) return send_error(err);
      // tool_choice named / required: the output is guided by the function's parameters schema (named)
      // or an array of {name, parameters} (required)
      if (tools_on && (choice_mode == "function" || choice_mode == "required")) {
        if (constrained)
          return send_error({400, "tool_choice \"" + choice_mode + "\" cannot be combined with response_format / guided decoding", "BadRequestError", "tool_choice"});
        auto params_of = [](const Value& t) {
          const Value* f = t.get_object("function");
          const Value* ps = f ? f->get("parameters") : nullptr;
          if (ps && ps->is_object()) return *ps;
          Value o = Value::object();
          o.set("type", Value::string("object"));
          o.set("properties", Value::object());
          return o;
        };
        if (choice_mode == "function") {
          forced = 1;
          forced_name = choice_name;
          constraint_schema = params_of(tools->arr[0]);
        } else {
          forced = 2;
          Value any = Value::array(), defs = Value::object();
          for (const Value& t : tools->arr) {
            const Value* f = t.get_object("function");
            const std::string* nm = f ? f->get_string("name") : nullptr;
            if (!nm) continue;
            Value ps = params_of(t);
            if (const Value* d = ps.get_object("$defs")) {
              for (const auto& [k, v] : d->obj) defs.set(k, v);
              ps.erase("$defs");
            }
            Value name_s = Value::object(), en = Value::array();
            name_s.set("type", Value::string("string"));
            en.push(Value::string(*nm));
            name_s.set("enum", en);
            Value props = Value::object();
            props.set("name", name_s);
            props.set("parameters", ps);
            Value req = Value::array();
            req.push(Value::string("name"));
            req.push(Value::string("parameters"));
            Value br = Value::object();
            br.set("properties", props);
            br.set("required", req);
            any.push(br);
          }
          Value items = Value::object();
          items.set("type", Value::string("object"));
          items.set("anyOf", any);
          Value s = Value::object();
          s.set("type", Value::string("array"));
          s.set("minItems", Value::integer(1));
          if (!parallel_calls) s.set("maxItems", Value::integer(1));
          s.set("items", items);
          if (!defs.obj.empty()) s.set("$defs", defs);
          constraint_schema = s;
        }
        constrained = true;
        raw_choice = false;
      }
    } else {
      const Value* pv = r.get("prompt");
      if (pv && pv->is_array() && pv->arr.size() == 1 && pv->arr[0].is_string()) pv = &pv->arr[0];
      if (!pv) return send_error({400, "prompt is required", "BadRequestError", "prompt"});
      if (pv->is_string()) {
        prompt = tok.encode(pv->str);
        prompt_text = pv->str;
      } else if (pv->is_array() && !pv->arr.empty() && pv->arr[0].is_number()) {
        const int32_t vocab = tok.vocab_size();
        for (const Value& t : pv->arr) {
          if (!t.is_int() || t.num < 0 || t.num >= vocab)
            return send_error({400, "prompt token ids must be integers in [0, vocab)", "BadRequestError", "prompt"});
          prompt.push_back(static_cast<int32_t>(t.num));
        }
        for (int32_t t : prompt) tok.decode_piece(t, prompt_text, false);
      } else {
        return send_error({400, "prompt must be a string, one-element list of strings, or a list of token ids",
                              "BadRequestError", "prompt"});
      }
    }
    if (prompt.empty()) return send_error({400, "the prompt is empty", "BadRequestError", "prompt"});

    core::SamplingParams& sp = p.sp;
    std::vector<std::string>& stops = p.stops;
    if (!parse_sampling(r, chat, prompt.size(), sp, stops, err)) return send_error(err);
    // the prefix cache's mark of a chat request = where its last message header (the generation
    // prompt, "<|im_start|>assistant\n...") starts: the conversation's next turn renders everything before it the same
    // way and differs from there (the reply, rendered back without its reasoning, or other text)
    if (chat && im_start >= 0)
      for (size_t i = prompt.size(); i-- > 0;)
        if (prompt[i] == im_start) {
          sp.prefix_mark = static_cast<int32_t>(i);
          break;
        }
    if (constrained) {  // a thinking chat is constrained after its reasoning ends (</think>)
      std::string cerr;
      const Value& s = raw_choice ? *constraint_schema.get("enum") : constraint_schema;
      sp.constraint = constraint_for(s, raw_choice, chat && thinking ? think_end : -1, cerr);
      if (!sp.constraint)
        return send_error({400, "structured output: " + cerr, "BadRequestError", raw_choice ? "guided_choice" : "response_format"});
    }
    bool& echo = p.echo;
    bool& token_ids = p.token_ids;
    if (!chat && !read_bool(r, "echo", echo, err)) return send_error(err);
    if (!read_bool(r, "return_tokens_as_token_ids", token_ids, err)) return send_error(err);
    const int explicit_prompt_lp = p.explicit_prompt_lp = sp.prompt_logprobs;
    if (explicit_prompt_lp >= 0 && stream)
      return send_error({400, "prompt_logprobs is not supported with stream: true", "BadRequestError", "prompt_logprobs"});
    if (echo && sp.logprobs >= 0) sp.prompt_logprobs = std::max(sp.prompt_logprobs, sp.logprobs);
    if (sp.prompt_logprobs >= 0 && !model.prompt_logprobs)
      return send_error({400, "prompt logprobs (prompt_logprobs, or echo with logprobs) are not supported by this engine yet",
                            "BadRequestError", "prompt_logprobs"});
    p.prep_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prep).count();
    return true;
  }

  // The loop thread's part: the queue cap, the Gen, the stream head, submit.
  void commit(Conn* c, Prepared& p) {
    const bool ka = p.ka, chat = p.chat;
    prepare_hist.observe(p.prep_s);
    const int n = p.n;
    const bool stream = p.stream, include_usage = p.include_usage, thinking = p.thinking, tools_on = p.tools_on,
               no_tool_parse = p.no_tool_parse, parallel_calls = p.parallel_calls, echo = p.echo, token_ids = p.token_ids;
    const int forced = p.forced, explicit_prompt_lp = p.explicit_prompt_lp;
    const auto& prompt = p.prompt;
    const auto& model = engine.model();
    const core::SamplingParams& sp = p.sp;
    if (cfg.max_queue > 0) {  // the request would have to queue behind more than the cap
      const core::Stats st = engine.stats();
      const long wait_after = static_cast<long>(st.waiting) + st.running + n - st.slots;
      if (st.slots > 0 && wait_after > cfg.max_queue) {
        rejected.fetch_add(1, std::memory_order_relaxed);
        const std::string body =
            error_body({429, "The queue is full (" + std::to_string(st.waiting) + " waiting, limit " +
                                 std::to_string(cfg.max_queue) + "); retry later.",
                        "TooManyRequestsError"});
        http::write_head(c->out, 429, "application/json", body.size(), ka, cors() + "Retry-After: 1\r\n");
        c->out += body;
        mark_dirty(c);
        return;
      }
    }

    auto g = std::make_unique<Gen>();
    g->id = ++next_id;
    g->conn = c;
    g->chat = chat;
    g->stream = stream;
    g->include_usage = include_usage;
    g->keep_alive = ka;
    g->echo = echo;
    g->prompt_tokens = static_cast<int32_t>(prompt.size());
    g->top_n = sp.logprobs;
    g->prompt_lp = explicit_prompt_lp;
    g->forced = forced;
    g->forced_name = p.forced_name;
    g->parallel_calls = parallel_calls;
    if (explicit_prompt_lp >= 0) {
      g->prompt_ids = prompt;
      if (!p.media.empty())  // image ids are listed as the image token
        for (int32_t& t : g->prompt_ids)
          if (cfg.vocab > 0 && t >= cfg.vocab) t = cfg.media->pad_token(t);  // <|video_pad|> at video frames
    }
    g->media = std::move(p.media);
    g->rid = (chat ? "chatcmpl-" : "cmpl-") + random_hex(32);
    g->created = static_cast<long long>(std::time(nullptr));
    if (p.responses) {
      g->responses = true;
      g->rid = "resp_" + random_hex(32);
      g->meta.id = g->rid;
      g->meta.model = model.name;
      g->meta.created = g->created;
      g->meta.request = std::move(p.resp_request);
      g->meta.temperature = sp.temperature;
      g->meta.top_p = sp.top_p;
      g->meta.max_output_tokens = sp.max_tokens;
      g->meta.top_logprobs = std::max(0, sp.logprobs);
      if (stream) {
        http::write_chunked_head(c->out, 200, "text/event-stream", ka, (cors() + "Cache-Control: no-cache\r\n").c_str());
        std::string sse;
        g->rstream.created(g->meta, sse);
        http::write_chunk(c->out, sse);
        mark_dirty(c);
      }
    }

    if (stream && !g->responses) {
      std::string& h = g->chunk_head;
      h = "data: {\"id\":";
      json::write_string(g->rid, h);
      h += chat ? ",\"object\":\"chat.completion.chunk\",\"created\":" : ",\"object\":\"text_completion\",\"created\":";
      append_int(h, g->created);
      h += ",\"model\":";
      json::write_string(model.name, h);
      h += ",\"system_fingerprint\":";
      json::write_string(fingerprint, h);
      h += ",\"choices\":[{\"index\":";
      http::write_chunked_head(c->out, 200, "text/event-stream", ka, (cors() + "Cache-Control: no-cache\r\n").c_str());
      if (chat) {  // OpenAI opens with a role-only delta for every choice
        std::string sse;
        for (int i = 0; i < n; ++i) {
          sse += h;
          append_int(sse, i);
          sse += ",\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
        }
        http::write_chunk(c->out, sse);
      }
      mark_dirty(c);
    }

    g->tools = std::move(p.tools_val);
    c->gen = g.get();
    Gen* gp = g.get();
    gens.emplace(gp->id, std::move(g));
    for (int i = 0; i < n; ++i) {
      auto ch = std::make_unique<Choice>(tok);
      ch->engine_id = ++next_id;
      ch->stop = text::StopMatcher(p.stops);
      if (chat)
        ch->parser = forced ? chat_fmt->parser(thinking, nullptr)
                     : no_tool_parse ? chat_fmt->parser_no_tool_calls(thinking)
                                     : chat_fmt->parser(thinking, tools_on ? &gp->tools : nullptr);
      ch->echo_text = p.prompt_text;
      ch->token_ids = token_ids;
      if (echo && sp.prompt_logprobs >= 0) ch->prompt_ids = prompt;
      const core::SamplingParams& csp = sp;
      slots.emplace(ch->engine_id, std::make_pair(gp, static_cast<size_t>(i)));
      const uint64_t id = ch->engine_id;
      gp->choices.push_back(std::move(ch));
      engine.submit(id, prompt, csp);
    }
    gp->remaining = gp->choices.size();
  }

  // ---- streaming pieces (per choice)

  void choice_chunk(const Gen& g, size_t idx, std::string& sse, std::string_view delta_json, const char* finish,
                    std::string_view lp_json = "null") {
    sse += g.chunk_head;
    append_int(sse, static_cast<long long>(idx));
    sse += ",\"delta\":";
    sse += delta_json;
    sse += ",\"logprobs\":";
    sse += lp_json;
    sse += ",\"finish_reason\":";
    if (finish) {
      sse += '"';
      sse += finish;
      sse += '"';
    } else {
      sse += "null";
    }
    sse += "}]}\n\n";
  }

  void text_chunk(const Gen& g, size_t idx, std::string& sse, std::string_view text, const char* finish,
                  std::string_view lp_json = "null") {
    sse += g.chunk_head;
    append_int(sse, static_cast<long long>(idx));
    sse += ",\"text\":";
    json::write_string(text, sse);
    sse += ",\"logprobs\":";
    sse += lp_json;
    sse += ",\"finish_reason\":";
    if (finish) {
      sse += '"';
      sse += finish;
      sse += '"';
    } else {
      sse += "null";
    }
    sse += ",\"stop_reason\":null}]}\n\n";
  }

  // The prompt tokens the request's prefix cache served: the first choice's (the prompt is counted once in the
  // usage); 0 where the engine never ran it.
  static int32_t cached_tokens(const Gen& g) {
    return g.choices.empty() ? 0 : std::clamp(g.choices[0]->cached_tokens, 0, g.prompt_tokens);
  }

  void usage_json(const Gen& g, std::string& s) {
    long long completion = 0;
    for (const auto& ch : g.choices) completion += ch->completion_tokens;
    s += "{\"prompt_tokens\":";
    append_int(s, g.prompt_tokens);
    s += ",\"total_tokens\":";
    append_int(s, static_cast<long long>(g.prompt_tokens) + completion);
    s += ",\"completion_tokens\":";
    append_int(s, completion);
    s += ",\"prompt_tokens_details\":{\"cached_tokens\":";
    append_int(s, cached_tokens(g));
    s += "}}";
  }

  // tool_choice named / required: the collected JSON content -> ch.calls (a named call's arguments
  // are the text itself; required parses [{name, parameters}...])
  void forced_calls(const Gen& g, Choice& ch) {
    std::string_view body(ch.content);
    while (!body.empty() && std::isspace(static_cast<unsigned char>(body.front()))) body.remove_prefix(1);
    while (!body.empty() && std::isspace(static_cast<unsigned char>(body.back()))) body.remove_suffix(1);
    if (g.forced == 1) {
      if (!body.empty()) ch.calls.push_back({"chatcmpl-tool-" + random_hex(32), g.forced_name, ch.content});
    } else {
      Value arr;
      if (json::parse(body, arr) && arr.is_array())
        for (const Value& x : arr.arr) {
          const std::string* nm = x.get_string("name");
          const Value* ps = x.get("parameters");
          if (nm && ps) ch.calls.push_back({"chatcmpl-tool-" + random_hex(32), *nm, json::dump(*ps, json::Style::python)});
        }
    }
    if (!g.parallel_calls && ch.calls.size() > 1) ch.calls.resize(1);
    ch.content.clear();
  }

  void stream_deltas(Gen& g, Choice& ch, size_t idx, const std::vector<text::Delta>& deltas, std::string& sse) {
    using text::Delta;
    std::string d;
    for (const Delta& x : deltas) {
      if (!g.parallel_calls && (x.kind == Delta::tool_start || x.kind == Delta::tool_args) && x.index > 0) continue;
      d.clear();
      switch (x.kind) {
        case Delta::reasoning:
          d = "{\"reasoning\":";
          json::write_string(x.text, d);
          d += "}";
          break;
        case Delta::content:
          d = "{\"content\":";
          json::write_string(x.text, d);
          d += "}";
          break;
        case Delta::tool_start: {
          ch.stream_call_ids.push_back("chatcmpl-tool-" + random_hex(32));
          d = "{\"tool_calls\":[{\"index\":";
          append_int(d, x.index);
          d += ",\"id\":";
          json::write_string(ch.stream_call_ids.back(), d);
          d += ",\"type\":\"function\",\"function\":{\"name\":";
          json::write_string(x.text, d);
          d += ",\"arguments\":\"\"}}]}";
          break;
        }
        case Delta::tool_args:
          d = "{\"tool_calls\":[{\"index\":";
          append_int(d, x.index);
          d += ",\"function\":{\"arguments\":";
          json::write_string(x.text, d);
          d += "}}]}";
          break;
      }
      choice_chunk(g, idx, sse, d, nullptr);
    }
  }

  void accumulate(Choice& ch, const std::vector<text::Delta>& deltas) {
    using text::Delta;
    for (const Delta& x : deltas) {
      switch (x.kind) {
        case Delta::reasoning: ch.reasoning += x.text; break;
        case Delta::content: ch.content += x.text; break;
        case Delta::tool_start: ch.calls.push_back({"chatcmpl-tool-" + random_hex(32), x.text, "{}"}); break;
        case Delta::tool_args:
          if (!ch.calls.empty()) ch.calls.back().arguments = x.text;
          break;
      }
    }
  }

  // ---- logprobs

  // A token as text for logprobs: its bytes when they are valid UTF-8, else the
  // "bytes:\\xNN" spelling.
  std::string token_str(int32_t id, bool as_ids = false) const {
    if (as_ids) return "token_id:" + std::to_string(id);
    std::string piece;
    tok.decode_piece(id, piece, false);
    for (size_t i = 0; i < piece.size();) {
      size_t len = 1;
      if (text::utf8_at(piece, i, len) != text::Utf8::complete) {
        std::string b = "bytes:";
        char x[8];
        for (unsigned char ch : piece) {
          std::snprintf(x, sizeof(x), "\\x%02x", ch);
          b += x;
        }
        return b;
      }
      i += len;
    }
    return piece;
  }

  static size_t code_points(std::string_view s) {
    size_t n = 0;
    for (unsigned char ch : s) n += (ch & 0xC0) != 0x80;
    return n;
  }

  static void lp_number(std::string& out, float v, bool chat) {
    if (std::isnan(v)) out += "null";
    else if (std::isinf(v)) out += chat ? "-9999.0" : "-Infinity";
    else out += json::python_float_repr(static_cast<double>(v));
  }

  static void sep(std::string& s) {
    if (!s.empty()) s += ',';
  }

  // One token into the completions arrays (`lp` null: the prompt's first token).
  void add_completion_lp(Choice& ch, int32_t t, const core::TokenLogprob* lp) {
    const std::string ts = token_str(t, ch.token_ids);
    sep(ch.lp_tokens);
    json::write_string(ts, ch.lp_tokens);
    sep(ch.lp_values);
    if (lp) lp_number(ch.lp_values, lp->logprob, false);
    else ch.lp_values += "null";
    sep(ch.lp_top);
    if (!lp) {
      ch.lp_top += "null";
    } else {
      ch.lp_top += '{';
      bool first = true, has_self = false;
      for (const auto& [id, v] : lp->top) {
        if (!first) ch.lp_top += ',';
        first = false;
        has_self |= id == t;
        json::write_string(token_str(id, ch.token_ids), ch.lp_top);
        ch.lp_top += ':';
        lp_number(ch.lp_top, v, false);
      }
      if (!has_self) {  // the chosen token is listed too
        if (!first) ch.lp_top += ',';
        json::write_string(ts, ch.lp_top);
        ch.lp_top += ':';
        lp_number(ch.lp_top, lp->logprob, false);
      }
      ch.lp_top += '}';
    }
    sep(ch.lp_offsets);
    append_int(ch.lp_offsets, static_cast<long long>(ch.text_chars));
    ch.text_chars += code_points(ts);
  }

  void chat_lp_entry(std::string& out, int32_t t, float v, bool as_ids) const {
    const std::string ts = token_str(t, as_ids);
    std::string piece;
    tok.decode_piece(t, piece, false);
    out += "{\"token\":";
    json::write_string(ts, out);
    out += ",\"logprob\":";
    lp_number(out, v, true);
    out += ",\"bytes\":[";
    for (size_t i = 0; i < piece.size(); ++i) {
      if (i) out += ',';
      append_int(out, static_cast<unsigned char>(piece[i]));
    }
    out += ']';
  }

  void add_chat_lp(Choice& ch, int32_t t, const core::TokenLogprob& lp) const {
    sep(ch.lp_content);
    chat_lp_entry(ch.lp_content, t, lp.logprob, ch.token_ids);
    ch.lp_content += ",\"top_logprobs\":[";
    for (size_t i = 0; i < lp.top.size(); ++i) {
      if (i) ch.lp_content += ',';
      chat_lp_entry(ch.lp_content, lp.top[i].first, lp.top[i].second, ch.token_ids);
      ch.lp_content += '}';
    }
    ch.lp_content += "]}";
  }

  // The prompt_logprobs list: null for the first token, then per prompt token an
  // object keyed by token id: the token itself and the top k, each {logprob, rank, decoded_token}.
  std::string prompt_lp_list(const std::vector<int32_t>& ids, const std::vector<core::TokenLogprob>& lps, bool as_ids) const {
    std::string s = "[";
    for (size_t i = 0; i < ids.size() && i < lps.size(); ++i) {
      if (i) s += ',';
      if (i == 0) {
        s += "null";
        continue;
      }
      const core::TokenLogprob& lp = lps[i];
      s += '{';
      auto entry = [&](int32_t id, float v, int32_t rank) {
        s += '"';
        append_int(s, id);
        s += "\":{\"logprob\":";
        lp_number(s, v, false);
        s += ",\"rank\":";
        append_int(s, rank);
        s += ",\"decoded_token\":";
        json::write_string(token_str(id, as_ids), s);
        s += '}';
      };
      bool self_in_top = false;
      for (const auto& [id, v] : lp.top) self_in_top |= id == ids[i];
      if (!self_in_top) {
        entry(ids[i], lp.logprob, lp.rank);
        if (!lp.top.empty()) s += ',';
      }
      for (size_t k = 0; k < lp.top.size(); ++k) {
        if (k) s += ',';
        entry(lp.top[k].first, lp.top[k].second, static_cast<int32_t>(k + 1));
      }
      s += '}';
    }
    s += ']';
    return s;
  }

  std::string completion_lp_json(const Choice& ch, size_t t0, size_t v0, size_t p0, size_t o0) const {
    auto tail = [](const std::string& s, size_t from) {
      std::string_view v(s);
      v.remove_prefix(std::min(from, v.size()));
      if (!v.empty() && v.front() == ',') v.remove_prefix(1);
      return v;
    };
    std::string s = "{\"text_offset\":[";
    s += tail(ch.lp_offsets, o0);
    s += "],\"token_logprobs\":[";
    s += tail(ch.lp_values, v0);
    s += "],\"tokens\":[";
    s += tail(ch.lp_tokens, t0);
    s += "],\"top_logprobs\":[";
    s += tail(ch.lp_top, p0);
    s += "]}";
    return s;
  }

  void message_json(const Choice& ch, std::string& s, bool forced = false) const {
    s += "\"message\":{\"role\":\"assistant\",\"content\":";
    if (ch.content.empty() && !ch.calls.empty() && !forced) s += "null";
    else json::write_string(ch.content, s);
    s += ",\"reasoning\":";
    if (ch.reasoning.empty()) s += "null";
    else json::write_string(ch.reasoning, s);
    s += ",\"refusal\":null,\"tool_calls\":[";
    for (size_t i = 0; i < ch.calls.size(); ++i) {
      if (i) s += ",";
      s += "{\"id\":";
      json::write_string(ch.calls[i].id, s);
      s += ",\"type\":\"function\",\"function\":{\"name\":";
      json::write_string(ch.calls[i].name, s);
      s += ",\"arguments\":";
      json::write_string(ch.calls[i].arguments, s);
      s += "}}";
    }
    s += "]}";
  }

  std::string final_body(const Gen& g) {
    const auto& model = engine.model();
    std::string s = "{\"id\":";
    json::write_string(g.rid, s);
    s += g.chat ? ",\"object\":\"chat.completion\",\"created\":" : ",\"object\":\"text_completion\",\"created\":";
    append_int(s, g.created);
    s += ",\"model\":";
    json::write_string(model.name, s);
    s += ",\"system_fingerprint\":";
    json::write_string(fingerprint, s);
    s += ",\"choices\":[";
    for (size_t i = 0; i < g.choices.size(); ++i) {
      const Choice& ch = *g.choices[i];
      if (i) s += ",";
      s += "{\"index\":";
      append_int(s, static_cast<long long>(i));
      s += ",";
      if (g.chat) {
        message_json(ch, s, g.forced != 0);
        s += ",\"logprobs\":";
        if (g.top_n >= 0) s += "{\"content\":[" + ch.lp_content + "]}";
        else s += "null";
      } else {
        s += "\"text\":";
        json::write_string(g.echo ? ch.echo_text + ch.text : ch.text, s);
        s += ",\"logprobs\":";
        if (g.top_n >= 0) s += completion_lp_json(ch, 0, 0, 0, 0);
        else s += "null";
      }
      s += ",\"finish_reason\":\"";
      s += ch.finish;
      s += "\"";
      if (!g.chat) s += ",\"stop_reason\":null";
      if (!g.chat && g.prompt_lp >= 0) s += ",\"prompt_logprobs\":" + (g.prompt_lp_json.empty() ? std::string("null") : g.prompt_lp_json);
      s += "}";
    }
    s += "]";
    if (g.chat && g.prompt_lp >= 0) s += ",\"prompt_logprobs\":" + (g.prompt_lp_json.empty() ? std::string("null") : g.prompt_lp_json);
    s += ",\"usage\":";
    usage_json(g, s);
    s += "}";
    return s;
  }

  // A finished Responses request's output items, usage and status (one choice).
  const char* responses_result(const Gen& g, json::Value& output, json::Value& use) const {
    const Choice& ch = *g.choices[0];
    std::vector<responses::Call> calls;
    for (const auto& k : ch.calls) calls.push_back({k.id, k.name, k.arguments});
    output = responses::output_items(ch.reasoning, ch.content, ch.lp_content, g.top_n >= 0, calls);
    const long long reasoning = ch.reasoning.empty() ? 0 : static_cast<long long>(tok.encode(ch.reasoning).size());
    use = responses::usage(g.prompt_tokens, cached_tokens(g), ch.completion_tokens, reasoning);
    return std::strcmp(ch.finish, "length") == 0 ? "incomplete" : "completed";
  }

  // The engine's event for one choice: build its chunks / accumulate its text.
  void on_engine_event(const core::Event& ev) {
    auto sit = slots.find(ev.id);
    if (sit == slots.end()) return;
    Gen* gp = sit->second.first;
    const size_t idx = sit->second.second;
    Gen& g = *gp;
    Choice& ch = *g.choices[idx];
    const bool final_event = ev.finish != core::Finish::none;
    if (ch.done) {
      if (final_event) slots.erase(sit);
      return;
    }

    if (ev.cached_tokens >= 0) ch.cached_tokens = ev.cached_tokens;
    if (g.prompt_lp >= 0 && g.prompt_lp_json.empty() && !ev.prompt_logprobs.empty())
      g.prompt_lp_json = prompt_lp_list(g.prompt_ids, ev.prompt_logprobs, ch.token_ids);
    // Logprobs of this event: the echoed prompt's (first event), then each token's.
    const size_t t0 = ch.lp_tokens.size(), v0 = ch.lp_values.size(), p0 = ch.lp_top.size(), o0 = ch.lp_offsets.size();
    const size_t c0 = ch.lp_content.size();
    std::string echo_now;
    if (g.echo && !ch.echo_text.empty() && ch.completion_tokens == 0 && !ch.echo_sent) {
      ch.echo_sent = true;
      echo_now = ch.echo_text;
      if (g.top_n >= 0 && !ev.prompt_logprobs.empty())
        for (size_t i = 0; i < ch.prompt_ids.size() && i < ev.prompt_logprobs.size(); ++i)
          add_completion_lp(ch, ch.prompt_ids[i], i == 0 ? nullptr : &ev.prompt_logprobs[i]);
      else
        ch.text_chars += code_points(ch.echo_text);
    }
    if (g.top_n >= 0)
      for (int i = 0; i < ev.n && static_cast<size_t>(i) < ev.logprobs.size(); ++i) {
        if (g.chat) add_chat_lp(ch, ev.tokens[i], ev.logprobs[static_cast<size_t>(i)]);
        else add_completion_lp(ch, ev.tokens[i], &ev.logprobs[static_cast<size_t>(i)]);
      }

    ch.completion_tokens += ev.n;
    std::string piece;
    for (int i = 0; i < ev.n - (ev.stop_token ? 1 : 0); ++i) ch.detok.push(ev.tokens[i]);
    if (final_event) ch.detok.flush(piece);
    else ch.detok.take(piece);

    std::string out_text;
    const bool stopped = ch.stop.feed(piece, out_text);
    if (final_event && !stopped) ch.stop.flush(out_text);
    const bool finishing = final_event || stopped;

    std::vector<text::Delta> deltas;
    if (ch.parser) {
      ch.parser->feed(out_text, deltas);
      if (finishing) ch.parser->finish(deltas);
    }
    if (g.forced) {  // the content is the call(s): collected, never sent as content
      for (const auto& d : deltas)
        if (d.kind == text::Delta::content) ch.content += d.text;
      deltas.erase(std::remove_if(deltas.begin(), deltas.end(), [](const text::Delta& d) { return d.kind == text::Delta::content; }), deltas.end());
    }

    if (finishing) {
      if (stopped) {
        ch.finish = "stop";
      } else {
        switch (ev.finish) {
          case core::Finish::stop: ch.finish = "stop"; break;
          case core::Finish::length: ch.finish = "length"; break;
          case core::Finish::cancelled: ch.finish = "abort"; break;
          case core::Finish::expired: ch.finish = "error"; g.expired = true; break;
          default: ch.finish = "error"; break;
        }
      }
      if (ch.parser && ch.parser->tool_calls() > 0 && std::strcmp(ch.finish, "stop") == 0) ch.finish = "tool_calls";
      if (g.forced) {
        forced_calls(g, ch);
        // the finish reason: "tool_calls" for required, "stop" for a named function
        if (g.forced == 2 && !ch.calls.empty() && std::strcmp(ch.finish, "stop") == 0) ch.finish = "tool_calls";
        if (g.stream)
          for (size_t i = 0; i < ch.calls.size(); ++i) {
            deltas.push_back({text::Delta::tool_start, ch.calls[i].name, static_cast<int>(i)});
            deltas.push_back({text::Delta::tool_args, ch.calls[i].arguments, static_cast<int>(i)});
          }
      }
    }
    const bool engine_error =
        finishing && !stopped && (ev.finish == core::Finish::error || ev.finish == core::Finish::expired);

    Conn* c = g.conn;
    if (c && g.responses) {  // Responses events / items
      if (g.stream) {
        std::string sse;
        if (g.top_n >= 0 && ch.lp_content.size() != c0) {
          std::string_view v(ch.lp_content);
          v.remove_prefix(c0);
          if (!v.empty() && v.front() == ',') v.remove_prefix(1);
          sep(g.pending_lp);
          g.pending_lp += v;
        }
        for (const text::Delta& x : deltas) {
          if (!g.parallel_calls && (x.kind == text::Delta::tool_start || x.kind == text::Delta::tool_args) && x.index > 0) continue;
          switch (x.kind) {
            case text::Delta::reasoning: g.rstream.reasoning(x.text, sse); break;
            case text::Delta::content:
              g.rstream.content(x.text, g.pending_lp, sse);
              if (!x.text.empty()) g.pending_lp.clear();
              break;
            case text::Delta::tool_start: g.rstream.tool_start(x.text, x.index, sse); break;
            case text::Delta::tool_args: g.rstream.tool_args(x.text, sse); break;
          }
        }
        if (!sse.empty()) {
          http::write_chunk(c->out, sse);
          mark_dirty(c);
        }
      }
      if (!g.forced) accumulate(ch, deltas);
      else
        for (const auto& d : deltas)
          if (d.kind == text::Delta::reasoning) ch.reasoning += d.text;
      if (finishing && !g.forced && ch.parser) chat_fmt->finish_content(ch.content, ch.parser->tool_markup());
      if (finishing && !g.parallel_calls && ch.calls.size() > 1) ch.calls.resize(1);
    } else if (c) {
      if (g.stream) {
        std::string sse;
        const bool new_lp = g.chat ? ch.lp_content.size() != c0 : ch.lp_tokens.size() != t0;
        if (g.chat) {
          stream_deltas(g, ch, idx, deltas, sse);
          if (new_lp) {
            std::string_view v(ch.lp_content);
            v.remove_prefix(c0);
            if (!v.empty() && v.front() == ',') v.remove_prefix(1);
            choice_chunk(g, idx, sse, "{}", nullptr, "{\"content\":[" + std::string(v) + "]}");
          }
        } else if (!out_text.empty() || !echo_now.empty() || new_lp || (ev.n > 0 && !finishing)) {
          // a step's tokens get a chunk even when they add no text (an EOS under
          // ignore_eos, text held back by the stop matcher or an incomplete UTF-8 sequence);
          // the finishing step's chunk below carries it otherwise
          text_chunk(g, idx, sse, echo_now + out_text, nullptr,
                     new_lp ? completion_lp_json(ch, t0, v0, p0, o0) : std::string("null"));
        }
        if (finishing) {
          if (engine_error) {
            sse += "data: ";
            sse += ev.finish == core::Finish::expired
                       ? error_body({503, "The request waited longer than the queue deadline; retry later.",
                                     "ServiceUnavailableError"})
                       : error_body({500, ev.error.empty() ? "engine error" : ev.error, "InternalServerError"});
            sse += "\n\n";
          } else if (g.chat) {
            choice_chunk(g, idx, sse, "{}", ch.finish);
          } else {
            text_chunk(g, idx, sse, "", ch.finish);
          }
        }
        if (!sse.empty()) {
          http::write_chunk(c->out, sse);
          mark_dirty(c);
        }
      } else {
        if (g.chat) {
          if (!g.forced) accumulate(ch, deltas);
          else
            for (const auto& d : deltas)
              if (d.kind == text::Delta::reasoning) ch.reasoning += d.text;
          if (finishing && !g.forced && ch.parser) chat_fmt->finish_content(ch.content, ch.parser->tool_markup());
          if (finishing && !g.parallel_calls && ch.calls.size() > 1) ch.calls.resize(1);
        } else {
          ch.text += out_text;
        }
      }
    }

    if (!finishing) return;

    // This choice is over.
    ch.done = true;
    slots.erase(sit);
    if (!final_event && c) engine.cancel(ch.engine_id);  // a stop string ended it: free the slot now
    if (--g.remaining > 0) return;

    // Last choice: finish the response.
    if (c && g.responses) {
      const bool err = g.expired || std::strcmp(ch.finish, "error") == 0;
      const std::string emsg = g.expired ? "The request waited longer than the queue deadline; retry later."
                                         : ev.error.empty() ? std::string("engine error") : ev.error;
      if (g.stream) {
        std::string sse;
        if (err) {
          g.rstream.error(g.expired ? 503 : 500, emsg, sse);
        } else {
          json::Value output, use;
          const char* status = responses_result(g, output, use);
          g.rstream.completed(g.meta, status, output, use, sse);
        }
        http::write_chunk(c->out, sse);
        http::write_last_chunk(c->out);
        mark_dirty(c);
      } else if (err) {
        send_error(c, {g.expired ? 503 : 500, emsg, g.expired ? "ServiceUnavailableError" : "InternalServerError"}, g.keep_alive);
      } else {
        json::Value output, use;
        const char* status = responses_result(g, output, use);
        send_json(c, 200, responses::response_json(g.meta, status, &output, &use), g.keep_alive);
      }
      c->gen = nullptr;
    } else if (c) {
      if (g.stream) {
        std::string sse;
        if (g.include_usage) {
          sse += g.chunk_head.substr(0, g.chunk_head.size() - std::strlen("[{\"index\":")) + "[]";
          sse += ",\"usage\":";
          usage_json(g, sse);
          sse += "}\n\n";
        }
        sse += "data: [DONE]\n\n";
        http::write_chunk(c->out, sse);
        http::write_last_chunk(c->out);
        mark_dirty(c);
      } else {
        bool any_error = false;
        for (const auto& x : g.choices) any_error |= std::strcmp(x->finish, "error") == 0;
        if (g.expired) {
          const std::string body = error_body({503, "The request waited longer than the queue deadline; retry later.",
                                               "ServiceUnavailableError"});
          http::write_head(c->out, 503, "application/json", body.size(), g.keep_alive, cors() + "Retry-After: 1\r\n");
          c->out += body;
          mark_dirty(c);
        } else if (any_error) send_error(c, {500, "engine error", "InternalServerError"}, g.keep_alive);
        else send_json(c, 200, final_body(g), g.keep_alive);
      }
      c->gen = nullptr;
    }
    const bool ka = g.keep_alive;
    const uint64_t gid = g.id;
    gens.erase(gid);  // frees `g`
    live.store(gens.size(), std::memory_order_relaxed);
    if (c) response_done(c, ka);
  }
};

// =====================================================================================

bool secret_name(std::string_view name) {
  std::string part;
  for (size_t at = 0; at <= name.size(); ++at) {
    if (at < name.size() && name[at] != '_' && name[at] != '-') {
      part += static_cast<char>(std::toupper(static_cast<unsigned char>(name[at])));
      continue;
    }
    for (const char* word : {"KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASS", "AUTH", "CREDENTIALS", "BEARER"})
      if (part == word) return true;
    part.clear();
  }
  return false;
}

Server::Server(Config cfg, core::Engine& engine, const text::Tokenizer& tok, const text::ChatFormat* chat)
    : impl_(std::make_unique<Impl>(std::move(cfg), engine, tok, chat)) {}
Server::~Server() = default;
bool Server::listen(std::string& err) { return impl_->listen(err); }
int Server::port() const { return impl_->bound_port; }
core::EventSink* Server::sink() { return impl_.get(); }
void Server::run() { impl_->run(); }
void Server::stop() {
  impl_->stopping.store(true);
  impl_->wake();
}
size_t Server::live_generations() const { return impl_->live.load(std::memory_order_relaxed); }

}  // namespace xenithium:server
