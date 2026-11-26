#include "client.hpp"

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>

#include "http_stream.hpp"
#include "stats.hpp"
#include "xenithium/util/json.hpp"
#include "xenithium/util/tls.hpp"

namespace xenithium::bench {
namespace {

namespace json = xenithium::json;
using Deadline = Clock::time_point;

constexpr size_t kReadBytes = 64 * 1024;
constexpr size_t kErrorBodyChars = 300;  // of a refused request's body, kept in Sample:error
constexpr int kHttpUnauthorized = 401, kHttpForbidden = 403;

double ms_between(Clock::time_point from, Clock::time_point to) { return std::chrono::duration<double, std::milli>(to - from).count(); }

int poll_timeout_ms(Deadline deadline) {
  // rounded up: a poll that returned before the deadline would report a timeout while time is left
  const auto left = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
  return static_cast<int>(std::clamp<long long>(left, 0, 1 << 30));
}

// A connected, non-blocking TCP socket, with TLS on top when the endpoint is https. Closes itself.
class Connection {
 public:
  Connection() = default;
  ~Connection() {
    if (tls_on_) tls_.shutdown();
    if (fd_ >= 0) ::close(fd_);
  }
  Connection(const Connection &) = delete;
  Connection &operator=(const Connection &) = delete;

  bool open(const Endpoint &to, Deadline deadline, std::string &err) {
    addrinfo hints{}, *found = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (const int rc = ::getaddrinfo(to.host.c_str(), to.port.c_str(), &hints, &found); rc != 0) {
      err = "cannot resolve " + to.host + ": " + ::gai_strerror(rc);
      return false;
    }
    err = "cannot connect to " + to.host + ":" + to.port;
    for (const addrinfo *a = found; a && fd_ < 0; a = a->ai_next) fd_ = connect_one(*a, deadline, err);
    ::freeaddrinfo(found);
    if (fd_ < 0) return false;
    err.clear();
    if (!to.tls) return true;
    tls_on_ = tls_.connect(fd_, to.host, "", deadline, err);
    return tls_on_;
  }

  bool write_all(std::string_view bytes, Deadline deadline, std::string &err) {
    if (tls_on_) return tls_.write_all(bytes.data(), bytes.size(), deadline, err);
    while (!bytes.empty()) {
      const ssize_t n = ::send(fd_, bytes.data(), bytes.size(), MSG_NOSIGNAL);
      if (n > 0) {
        bytes.remove_prefix(static_cast<size_t>(n));
        continue;
      }
      if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return err = std::string("send: ") + std::strerror(errno), false;
      if (!wait(POLLOUT, deadline)) return err = "timed out sending the request", false;
    }
    return true;
  }

  // > 0 bytes read, 0 = the peer closed, -1 = error or timeout (err set).
  long read(char *into, size_t size, Deadline deadline, std::string &err) {
    if (tls_on_) return tls_.read(into, size, deadline, err);
    while (true) {
      const ssize_t n = ::recv(fd_, into, size, 0);
      if (n >= 0) return n;
      if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) return err = std::string("recv: ") + std::strerror(errno), -1;
      if (!wait(POLLIN, deadline)) return err = "timed out waiting for the response", -1;
    }
  }

 private:
  bool wait(short events, Deadline deadline) const {
    pollfd p{fd_, events, 0};
    return ::poll(&p, 1, poll_timeout_ms(deadline)) > 0;
  }

  // The connected non-blocking socket for one address, or -1 (err says why).
  static int connect_one(const addrinfo &a, Deadline deadline, std::string &err) {
    const int fd = ::socket(a.ai_family, a.ai_socktype, a.ai_protocol);
    if (fd < 0) return -1;
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // a request is one small write: no Nagle delay
    bool connected = ::connect(fd, a.ai_addr, a.ai_addrlen) == 0;
    if (!connected && errno == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      int so_error = 0;
      socklen_t len = sizeof(so_error);
      connected = ::poll(&p, 1, poll_timeout_ms(deadline)) > 0 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) == 0 && so_error == 0;
      if (!connected && so_error != 0) err = std::string("connect: ") + std::strerror(so_error);
    }
    if (connected) return fd;
    ::close(fd);
    return -1;
  }

  int fd_ = -1;
  util::tls::Conn tls_;
  bool tls_on_ = false;
};

std::string request_head(const Endpoint &to, std::string_view method, std::string_view path, size_t body_bytes) {
  std::string head;
  head.append(method).append(" ").append(path).append(" HTTP/1.1\r\nHost: ").append(to.host);
  head += "\r\nAccept: text/event-stream, application/json\r\nConnection: close\r\n";
  if (!to.api_key.empty()) head += "Authorization: Bearer " + to.api_key + "\r\n";
  if (body_bytes > 0) head += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body_bytes) + "\r\n";
  return head + "\r\n";
}

// One whole exchange without streaming: the status and the body. nullopt + err on a transport error.
struct Reply {
  int status = 0;
  std::string body;
};
std::optional<Reply> exchange(const Endpoint &to, std::string_view method, std::string_view path, std::string_view body, double timeout_s, std::string &err) {
  const Deadline deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout_s));
  Connection conn;
  if (!conn.open(to, deadline, err) || !conn.write_all(request_head(to, method, path, body.size()), deadline, err) || !conn.write_all(body, deadline, err))
    return std::nullopt;
  HttpResponseDecoder http;
  Reply reply;
  std::array<char, kReadBytes> buf{};
  while (!http.done()) {
    const long n = conn.read(buf.data(), buf.size(), deadline, err);
    if (n < 0) return std::nullopt;
    if (n == 0) break;
    if (!http.feed({buf.data(), static_cast<size_t>(n)}, reply.body)) return err = http.error(), std::nullopt;
  }
  if (!http.head_done()) return err = "the connection closed before a response", std::nullopt;
  reply.status = http.status();
  return reply;
}

std::string api_path(const Endpoint &to, std::string_view path) {
  return !path.empty() && path.front() == '/' ? std::string(path) : to.base + "/" + std::string(path);
}

const std::string *string_at(const json::Value &obj, std::string_view key) {
  const std::string *s = obj.get_string(key);
  return s && !s->empty() ? s : nullptr;
}

// Folds a delta's tool_calls pieces into the sample's calls: the first piece of a call names it (and its id), the later
// ones append to its arguments. Returns whether a piece carried any of the call's text.
bool fold_tool_calls(const json::Value &calls, Sample &s) {
  bool carried = false;
  for (const json::Value &piece : calls.arr) {
    const json::Value *index = piece.get("index");
    const size_t at = index && index->is_number() ? static_cast<size_t>(index->num) : s.tool_calls.size();
    if (s.tool_calls.size() <= at) s.tool_calls.resize(at + 1);
    ToolCall &call = s.tool_calls[at];
    if (const std::string *id = string_at(piece, "id")) call.id = *id;
    if (const json::Value *function = piece.get_object("function")) {
      if (const std::string *name = string_at(*function, "name")) call.name += *name, carried = true;
      if (const std::string *args = string_at(*function, "arguments")) call.arguments += *args, carried = true;
    }
  }
  return carried;
}

// Folds one stream event into the sample. Returns whether the event carried tokens.
bool fold_event(std::string_view data, Sample &s) {
  json::Value ev;
  if (data == "[DONE]" || !json::parse(data, ev)) return false;
  if (const json::Value *e = ev.get("error")) {  // some servers report a failure inside the stream
    const std::string *msg = e->is_object() ? e->get_string("message") : nullptr;
    s.error = "stream error: " + (msg ? *msg : json::dump(*e));
    return false;
  }
  if (const json::Value *u = ev.get_object("usage")) {
    s.usage_reported = true;
    if (const json::Value *v = u->get("prompt_tokens")) s.prompt_tokens = static_cast<int>(v->num);
    if (const json::Value *v = u->get("completion_tokens")) s.completion_tokens = static_cast<int>(v->num);
    if (const json::Value *d = u->get_object("prompt_tokens_details"))
      if (const json::Value *v = d->get("cached_tokens"); v && v->is_number()) s.cached_tokens = static_cast<int>(v->num);
  }
  const json::Value *choices = ev.get_array("choices");
  if (!choices || choices->arr.empty()) return false;
  const json::Value &choice = choices->arr.front();
  if (const std::string *why = string_at(choice, "finish_reason")) s.finish_reason = *why;
  const json::Value *delta = choice.get_object("delta");
  const json::Value &from = delta ? *delta : choice;  // chat streams a delta, /completions a bare text
  const std::string *text = from.get_string(delta ? "content" : "text");
  const std::string *think = from.get_string("reasoning_content");
  if (!think) think = from.get_string("reasoning");
  if (text) s.text += *text;
  if (think) s.reasoning += *think;
  const json::Value *calls = delta ? delta->get_array("tool_calls") : nullptr;
  const bool has_call = calls && fold_tool_calls(*calls, s);
  const bool has_text = (text && !text->empty()) || (think && !think->empty()) || has_call;
  // After the first token an empty chunk is still a step's update (a token that ends inside a UTF-8 character);
  // before it, an empty chunk is the role announcement some servers send at once.
  const bool empty_update = !s.update_ms.empty() && (text || think) && s.finish_reason.empty();
  return has_text || empty_update;
}

}  // namespace

std::optional<Endpoint> parse_endpoint(std::string_view url, std::string &err) {
  Endpoint e;
  constexpr std::string_view kHttp = "http://", kHttps = "https://";
  if (url.substr(0, kHttps.size()) == kHttps) e.tls = true, url.remove_prefix(kHttps.size());
  else if (url.substr(0, kHttp.size()) == kHttp) url.remove_prefix(kHttp.size());
  else return err = "the endpoint must start with http:// or https://", std::nullopt;
  const size_t slash = url.find('/');
  std::string_view authority = url.substr(0, slash);
  std::string_view path = slash == std::string_view::npos ? std::string_view{} : url.substr(slash);
  while (!path.empty() && path.back() == '/') path.remove_suffix(1);
  const size_t colon = authority.rfind(':');
  e.host = std::string(authority.substr(0, colon));
  e.port = colon == std::string_view::npos ? (e.tls ? "443" : "80") : std::string(authority.substr(colon + 1));
  e.base = path.empty() ? "/v1" : std::string(path);
  if (e.host.empty() || e.port.empty()) return err = "the endpoint has no host", std::nullopt;
  return e;
}

double Sample::tokens_per_update() const { return update_ms.empty() ? 0 : static_cast<double>(completion_tokens) / static_cast<double>(update_ms.size()); }

double Sample::decode_tps() const {
  if (update_ms.size() < 2) return 0;
  const double span_s = (update_ms.back() - update_ms.front()) / 1e3;
  const double later_tokens = tokens_per_update() * static_cast<double>(update_ms.size() - 1);
  return span_s > 0 ? later_tokens / span_s : 0;
}

double Sample::prefill_tps() const {
  const int processed = prompt_tokens - std::clamp(cached_tokens, 0, prompt_tokens);
  return ttft_ms > 0 ? processed / (ttft_ms / 1e3) : 0;
}

double Sample::update_gap_ms() const {
  if (update_ms.size() < 2) return 0;
  std::vector<double> gaps(update_ms.size() - 1);
  for (size_t i = 0; i + 1 < update_ms.size(); ++i) gaps[i] = update_ms[i + 1] - update_ms[i];
  return summarize(gaps).median;
}

Client::Client(Endpoint endpoint, std::string model, Capabilities caps, double timeout_s)
    : endpoint_(std::move(endpoint)), model_(std::move(model)), caps_(caps), timeout_s_(timeout_s) {}

namespace {

// A turn with images in the OpenAI content-parts form: the images, then the text.
json::Value content_parts(const Message &m) {
  json::Value parts = json::Value::array();
  for (const std::string &url : m.image_urls) {
    json::Value image_url = json::Value::object();
    image_url.set("url", json::Value::string(url));
    json::Value part = json::Value::object();
    part.set("type", json::Value::string("image_url"));
    part.set("image_url", std::move(image_url));
    parts.push(std::move(part));
  }
  json::Value text = json::Value::object();
  text.set("type", json::Value::string("text"));
  text.set("text", json::Value::string(m.content));
  parts.push(std::move(text));
  return parts;
}

json::Value tool_calls_json(const std::vector<ToolCall> &calls) {
  json::Value list = json::Value::array();
  for (const ToolCall &call : calls) {
    json::Value function = json::Value::object();
    function.set("name", json::Value::string(call.name));
    function.set("arguments", json::Value::string(call.arguments));
    json::Value item = json::Value::object();
    item.set("id", json::Value::string(call.id));
    item.set("type", json::Value::string("function"));
    item.set("function", std::move(function));
    list.push(std::move(item));
  }
  return list;
}

json::Value message_json(const Message &m) {
  json::Value msg = json::Value::object();
  msg.set("role", json::Value::string(m.role));
  msg.set("content", m.image_urls.empty() ? json::Value::string(m.content) : content_parts(m));
  if (!m.tool_calls.empty()) msg.set("tool_calls", tool_calls_json(m.tool_calls));
  if (!m.tool_call_id.empty()) msg.set("tool_call_id", json::Value::string(m.tool_call_id));
  return msg;
}

}  // namespace

std::string Client::body_of(const ChatRequest &r) const {
  json::Value body = json::Value::object();
  body.set("model", json::Value::string(model_));
  json::Value messages = json::Value::array();
  for (const Message &m : r.messages) messages.push(message_json(m));
  body.set("messages", std::move(messages));
  body.set("max_tokens", json::Value::integer(r.max_tokens));
  body.set("temperature", json::Value::real(r.sampling.temperature));
  body.set("top_p", json::Value::real(r.sampling.top_p));
  if (r.sampling.seed) body.set("seed", json::Value::integer(*r.sampling.seed));
  body.set("stream", json::Value::boolean(true));
  if (caps_.stream_usage) {
    json::Value opts = json::Value::object();
    opts.set("include_usage", json::Value::boolean(true));
    body.set("stream_options", std::move(opts));
  }
  if (caps_.top_k && r.sampling.top_k > 0) body.set("top_k", json::Value::integer(r.sampling.top_k));
  if (caps_.ignore_eos && r.ignore_eos) body.set("ignore_eos", json::Value::boolean(true));
  if (caps_.thinking_switch && r.thinking != Thinking::server_default) {
    json::Value kwargs = json::Value::object();
    kwargs.set("enable_thinking", json::Value::boolean(r.thinking == Thinking::on));
    body.set("chat_template_kwargs", std::move(kwargs));
  }
  if (json::Value format; !r.response_format.empty() && json::parse(r.response_format, format)) body.set("response_format", std::move(format));
  if (json::Value tools; !r.tools.empty() && json::parse(r.tools, tools)) body.set("tools", std::move(tools));
  if (!r.tool_choice.empty()) {
    json::Value choice;  // a function object, or one of the words
    body.set("tool_choice", r.tool_choice.front() == '{' && json::parse(r.tool_choice, choice) ? std::move(choice) : json::Value::string(r.tool_choice));
  }
  return json::dump(body);
}

Sample Client::chat(const ChatRequest &request, const ReadLimits &limits) const {
  Sample s;
  s.submitted = Clock::now();
  const auto after_s = [&](double seconds) { return s.submitted + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds)); };
  const Deadline deadline = after_s(timeout_s_);
  // the reply is read until this moment at most: the request's timeout, or the moment the user moves on
  const Deadline read_until = limits.abandon_after_ms > 0 ? std::min(deadline, after_s(limits.abandon_after_ms / 1e3)) : deadline;
  const std::string body = body_of(request);
  Connection conn;
  if (!conn.open(endpoint_, deadline, s.error) || !conn.write_all(request_head(endpoint_, "POST", api_path(endpoint_, "chat/completions"), body.size()), deadline, s.error) ||
      !conn.write_all(body, deadline, s.error))
    return s;

  HttpResponseDecoder http;
  SseSplitter events;
  std::string decoded;  // body bytes of the current read
  std::string refused;  // a non-200 body
  std::array<char, kReadBytes> buf{};
  while (!http.done() && s.error.empty()) {
    const long n = conn.read(buf.data(), buf.size(), read_until, s.error);
    if (n < 0 && read_until < deadline && Clock::now() >= read_until) {  // still streaming when the user moved on
      s.error.clear();
      s.abandoned = true;
      break;
    }
    if (n <= 0) break;
    const Clock::time_point arrived = Clock::now();
    decoded.clear();
    if (!http.feed({buf.data(), static_cast<size_t>(n)}, decoded)) {
      s.error = http.error();
      break;
    }
    if (http.head_done() && http.status() != 200) {
      refused += decoded;
      continue;
    }
    events.feed(decoded, [&](std::string_view data) {
      if (fold_event(data, s)) s.update_ms.push_back(ms_between(s.submitted, arrived));
    });
  }
  s.http_status = http.status();
  if (s.abandoned && (!http.head_done() || s.http_status == 200)) {  // what came before the user moved on
    if (!s.usage_reported) s.completion_tokens = static_cast<int>(s.update_ms.size());
    s.ttft_ms = s.update_ms.empty() ? 0 : s.update_ms.front();
    s.wall_ms = ms_between(s.submitted, Clock::now());
    s.ok = true;
    return s;
  }
  s.abandoned = false;  // refused before the user moved on: an error like any other
  if (s.error.empty() && !http.head_done()) s.error = "the connection closed before a response";
  if (s.error.empty() && s.http_status != 200) s.error = "HTTP " + std::to_string(s.http_status) + ": " + refused.substr(0, kErrorBodyChars);
  if (s.error.empty() && s.update_ms.empty()) s.error = "the stream carried no tokens";
  if (!s.error.empty()) return s;
  if (!s.usage_reported) s.completion_tokens = static_cast<int>(s.update_ms.size());  // one token per event at best
  s.ttft_ms = s.update_ms.front();
  s.wall_ms = s.update_ms.back();
  s.ok = true;
  return s;
}

std::optional<std::string> Client::get(std::string_view path, std::string *err) const {
  std::string why;
  const auto reply = exchange(endpoint_, "GET", api_path(endpoint_, path), {}, timeout_s_, why);
  if (reply && reply->status / 100 == 2) return reply->body;
  if (err) *err = reply ? "HTTP " + std::to_string(reply->status) : why;
  return std::nullopt;
}

util::Machine machine_from(const json::Value &obj) {
  util::Machine m;
  auto text = [&](const char *key) { const std::string *v = obj.get_string(key); return v ? *v : std::string(); };
  auto count = [&](const char *key) { const json::Value *v = obj.get(key); return v && v->is_number() ? static_cast<int>(v->num) : 0; };
  m.os = text("os"), m.kernel = text("kernel"), m.cpu = text("cpu"), m.cpu_threads = count("cpu_threads"), m.memory_gib = count("memory_gib");
  return m;
}

std::optional<ModelInfo> served_model(const Endpoint &endpoint, std::string_view wanted, double timeout_s, std::string &err) {
  const auto reply = exchange(endpoint, "GET", api_path(endpoint, "models"), {}, timeout_s, err);
  if (!reply) return std::nullopt;
  if (reply->status == kHttpUnauthorized || reply->status == kHttpForbidden)
    return err = "GET models: HTTP " + std::to_string(reply->status) + " (the endpoint wants a key: --api-key, or XEN_BENCH_API_KEY)", std::nullopt;
  if (reply->status / 100 != 2) return err = "GET models: HTTP " + std::to_string(reply->status), std::nullopt;
  json::Value list;
  if (!json::parse(reply->body, list)) return err = "GET models: the reply is not JSON", std::nullopt;
  const json::Value *data = list.get_array("data");
  if (!data || data->arr.empty()) return err = "the endpoint lists no model", std::nullopt;
  for (const json::Value &entry : data->arr) {
    const std::string *id = entry.get_string("id");
    if (!id || (!wanted.empty() && *id != wanted)) continue;
    ModelInfo info;
    info.id = *id;
    if (const json::Value *len = entry.get("max_model_len"); len && len->is_number()) info.max_model_len = static_cast<int64_t>(len->num);
    if (const json::Value *pool = entry.get("kv_pool_tokens"); pool && pool->is_number()) info.kv_pool_tokens = static_cast<int64_t>(pool->num);
    if (const std::string *owner = entry.get_string("owned_by")) info.owned_by = *owner;
    if (const std::string *build = entry.get_string("build")) info.build = *build;
    if (const std::string *oneapi = entry.get_string("oneapi")) info.oneapi = *oneapi;
    if (const std::string *drafter = entry.get_string("drafter")) info.drafter = *drafter;
    if (const json::Value *machine = entry.get_object("machine")) info.machine = machine_from(*machine);
    if (const json::Value *launch = entry.get_object("launch")) {
      if (const json::Value *args = launch->get_array("args"))
        for (const json::Value &a : args->arr)
          if (a.is_string()) info.launch_args.push_back(a.str);
      if (const json::Value *env = launch->get_object("env"))
        for (const auto &[k, v] : env->obj)
          if (v.is_string()) info.launch_env.push_back({k, v.str});
    }
    if (const json::Value *tp = entry.get("tensor_parallel"); tp && tp->is_number()) info.tensor_parallel = static_cast<int>(tp->num);
    if (const json::Value *devices = entry.get_array("devices"))
      for (const json::Value &d : devices->arr) {
        const std::string *name = d.get_string("name");
        if (!name) continue;
        Device device;
        device.name = *name;
        if (const json::Value *mem = d.get("memory_bytes"); mem && mem->is_number() && mem->num > 0) device.memory_bytes = static_cast<uint64_t>(mem->num);
        if (const json::Value *gb = d.get("memory_gb"); gb && gb->is_number() && gb->num > 0) device.memory_gb = static_cast<int>(gb->num);
        if (const std::string *pci = d.get_string("pci")) device.pci = *pci;
        if (const std::string *driver = d.get_string("driver")) device.driver = *driver;
        info.devices.push_back(std::move(device));
      }
    return info;
  }
  return err = "the endpoint does not list a model \"" + std::string(wanted) + "\"", std::nullopt;
}

Capabilities probe_capabilities(const Endpoint &endpoint, const std::string &model, double timeout_s) {
  ChatRequest tiny;
  tiny.messages = {{"user", "Say OK."}};
  tiny.max_tokens = 4;
  tiny.ignore_eos = true;
  tiny.thinking = Thinking::off;
  auto accepted = [&](Capabilities with) { return Client(endpoint, model, with, timeout_s).chat(tiny); };

  Capabilities caps;
  const Capabilities all{.ignore_eos = true, .top_k = true, .thinking_switch = true, .stream_usage = true};
  if (const Sample s = accepted(all); s.ok) {
    caps = all;
    caps.stream_usage = s.usage_reported;
  } else {  // refused as a whole: which extension is it?
    caps.ignore_eos = accepted({.ignore_eos = true}).ok;
    caps.top_k = accepted({.top_k = true}).ok;
    caps.thinking_switch = accepted({.thinking_switch = true}).ok;
    caps.stream_usage = accepted({.stream_usage = true}).usage_reported;
  }
  const Client plain(endpoint, model, caps, timeout_s);
  if (const auto metrics = plain.get("/metrics")) caps.metrics = metrics->find("# TYPE") != std::string::npos;
  return caps;
}

}  // namespace xenithium:bench
