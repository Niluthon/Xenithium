// The launcher's screens.
#include "app.hpp"

#include "bench.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <deque>
#include <future>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "chat.hpp"
#include "gpu.hpp"
#include "job.hpp"
#include "proc.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/util/http.hpp"
#include "xenithium/util/json.hpp"

namespace xenithium::launch {

namespace {

// A run's quality score under this is drawn as a warning in its report: the questions are easy on purpose, so a healthy
// model misses a few at most.
constexpr double kQualityWarnBelowPct = 85;

using Clock = std::chrono::steady_clock;

// ---- theme ----------------------------------------------------------------------------------

namespace th {
// Royal blue theme: royal (0x4169E1) fills (active tab, chosen option), royal_lt is the same hue
// lifted for text on a dark ground, the greys, rules and panels are blue-tinted. Orange / green /
// yellow / red keep their meaning (changed / ok / warning / error).
constexpr uint32_t royal = 0x4169E1, royal_lt = 0x7C9BFF, white = 0xFFFFFF, cyan = 0xA9C1FF, green = 0x7EDC9A, yellow = 0xE8C46A,
                   red = 0xFF7A90, orange = 0xFFA45E, purple = 0x93A8FF, grey = 0x8C98C8, border = 0x2E4BA8, bar = 0x111C45,
                   sel = 0x203276, sel_off = 0x18235A, rule = 0x27356F, popup = 0x0C1236, field = 0x1A2660,
                   text = 0xD8E0FF;
const Style plain{};
const Style dim{grey, kDefault, 0};
const Style bold{kDefault, kDefault, kBold};
const Style acc{royal_lt, kDefault, kBold};
const Style acc2{cyan, kDefault, 0};
const Style ok{green, kDefault, kBold};
const Style warn{yellow, kDefault, 0};
const Style err{red, kDefault, kBold};
const Style mod{orange, kDefault, kBold};
const Style code{purple, kDefault, 0};
const Style label{cyan, kDefault, kBold};  // the help panel's line labels (Output, KV context:)
const Style gives{green, kDefault, kBold};  // its "Gives:" labels
const Style takes{orange, kDefault, kBold};  // its "Takes:" labels
const Style frame{border, kDefault, 0};
const Style frame_on{royal_lt, kDefault, 0};
}  // namespace th

const char *kSpin[] = {"\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8", "\xe2\xa0\xbc",
                       "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7", "\xe2\xa0\x87", "\xe2\xa0\x8f"};

std::string spinner() {
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
  return kSpin[(ms / 100) % 10];
}

std::string dur(double s) {
  const long t = static_cast<long>(s);
  char b[32];
  if (t >= 3600) std::snprintf(b, sizeof b, "%ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60);
  else std::snprintf(b, sizeof b, "%ld:%02ld", t / 60, t % 60);
  return b;
}

std::string fmt1(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1f", v);
  return b;
}

std::string trim_text(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

// ---- one-line editor ------------------------------------------------------------------------

struct LineEdit {
  std::string s;
  size_t cur = 0;  // byte offset
  bool path = false;
  std::string root;
  std::vector<std::string> cands;  // path completions after a Tab

  void set(const std::string &v) {
    s = v;
    cur = s.size();
    cands.clear();
  }
  static size_t prev(const std::string &s, size_t i) {
    if (i == 0) return 0;
    --i;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xc0) == 0x80) --i;
    return i;
  }
  static size_t next(const std::string &s, size_t i) {
    if (i >= s.size()) return s.size();
    ++i;
    while (i < s.size() && (static_cast<unsigned char>(s[i]) & 0xc0) == 0x80) ++i;
    return i;
  }
  // True when the key was an editing key.
  bool handle(const Key &k) {
    switch (k.type) {
      case Key::Char: {
        std::string u;
        utf8_append(u, k.ch);
        s.insert(cur, u);
        cur += u.size();
        cands.clear();
        return true;
      }
      case Key::Paste: {
        std::string t;
        for (const char c : k.text)
          if (c != '\n' && c != '\r') t += c;
        s.insert(cur, t);
        cur += t.size();
        return true;
      }
      case Key::Backspace:
        if (cur > 0) {
          const size_t p = prev(s, cur);
          s.erase(p, cur - p);
          cur = p;
        }
        cands.clear();
        return true;
      case Key::Delete:
        if (cur < s.size()) s.erase(cur, next(s, cur) - cur);
        return true;
      case Key::Left: cur = prev(s, cur); return true;
      case Key::Right: cur = next(s, cur); return true;
      case Key::Home: cur = 0; return true;
      case Key::End: cur = s.size(); return true;
      case Key::Ctrl:
        if (k.ch == 'a') cur = 0;
        else if (k.ch == 'e') cur = s.size();
        else if (k.ch == 'u') {
          s.erase(0, cur);
          cur = 0;
        } else if (k.ch == 'k') s.erase(cur);
        else if (k.ch == 'w') {
          size_t p = cur;
          while (p > 0 && s[p - 1] == ' ') --p;
          while (p > 0 && s[p - 1] != ' ' && s[p - 1] != '/') --p;
          s.erase(p, cur - p);
          cur = p;
        } else return false;
        return true;
      case Key::Tab:
        if (!path) return false;
        {
          std::string common;
          cands = complete_path(s, root, common);
          if (!common.empty() && common.size() > s.size()) {
            s = common;
            cur = s.size();
            if (cands.size() == 1) cands.clear();
          }
        }
        return true;
      default: return false;
    }
  }
  void draw(Canvas &c, int y, int x, int w, Style st) const {
    c.fill(y, x, w, 1, st);
    // Scroll so the cursor stays visible.
    const int before = text_width(std::string_view(s).substr(0, cur));
    const int off = std::max(0, before - (w - 2));
    int col = 0;
    for (size_t i = 0; i < s.size();) {
      const size_t at = i;
      const char32_t ch = utf8_next(s, i);
      const int cw = cp_width(ch);
      if (col >= off && col - off + cw <= w) {
        Style cs = at == cur ? st.with(kReverse) : st;
        std::string u;
        utf8_append(u, ch);
        c.put(y, x + col - off, u, cs);
      }
      col += cw;
    }
    if (cur >= s.size() && before - off < w) c.put(y, x + before - off, " ", st.with(kReverse));
  }
};

// ---- popups ---------------------------------------------------------------------------------

struct Item {
  std::string label, detail, value, mark;
  Style mark_st = th::plain, label_st = th::plain;
  bool enabled = true, header = false;
};

struct Popup {
  enum Kind { List, Input } kind = List;
  std::string title, note;
  std::vector<Item> items;
  int sel = 0, scroll = 0, width = 0;
  bool selectable = true, filterable = false;
  std::string filter;
  LineEdit edit;
  std::string prompt, error;
  std::vector<std::pair<std::string, std::string>> hints;
  std::function<void(Popup &)> on_ok;  // Enter on an enabled item / the input
  std::function<bool(Popup &, const Key &)> on_key;  // extra keys; true = consumed
  std::function<void()> on_cancel;
  bool close = false;  // set by callbacks

  std::vector<int> visible() const {
    std::vector<int> v;
    std::string f = filter;
    std::transform(f.begin(), f.end(), f.begin(), ::tolower);
    for (int i = 0; i < static_cast<int>(items.size()); ++i) {
      if (!f.empty()) {
        if (items[i].header) continue;
        std::string l = items[i].label + " " + items[i].detail;
        std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        if (l.find(f) == std::string::npos) continue;
      }
      v.push_back(i);
    }
    return v;
  }
  void fix_sel() {
    const auto v = visible();
    if (v.empty()) return;
    if (std::find(v.begin(), v.end(), sel) == v.end() || items[sel].header || !items[sel].enabled) {
      for (const int i : v)
        if (!items[i].header && items[i].enabled) {
          sel = i;
          return;
        }
      sel = v[0];
    }
  }
  void move(int d) {
    const auto v = visible();
    if (v.empty()) return;
    int pos = static_cast<int>(std::find(v.begin(), v.end(), sel) - v.begin());
    if (pos >= static_cast<int>(v.size())) pos = 0;
    for (size_t n = 0; n < v.size(); ++n) {
      pos = (pos + d + static_cast<int>(v.size())) % static_cast<int>(v.size());
      if (!items[v[pos]].header && items[v[pos]].enabled) break;
      if (!selectable) break;
    }
    sel = v[pos];
  }
};

// ---- the server monitor (a thread) ----------------------------------------------------------

struct GpuLive {
  GpuInfo info;
  double watts = -1;  // board power over the last second (energy counter delta); -1 = not known yet
};

struct Snapshot {
  std::vector<GpuLive> gpus;
  pid_t pid = -1;  // the server the launcher started (or adopted from its pid file)
  bool ours_alive = false;
  int exit_code = INT_MIN;  // INT_MIN = still running / never ran; -1000 = gone, code unknown
  std::string exit_why;  // the end in words (exit_reason: signal, OOM kill from the kernel log), set with exit_code
  int port = 0;
  bool ready = false;
  double ready_after = 0;
  Clock::time_point started{}, ready_at{};
  bool stopping = false, adopted = false;
  Clock::time_point stop_sent{};
  std::vector<ProcInfo> servers;
  std::vector<std::string> containers;
  bool docker_ok = true, docker_known = false;
  std::map<std::string, double> metrics;
  bool metrics_ok = false;
  // Decode rate summed over every request: tokens the server generated for all of them (its
  // vllm:generation_tokens_total) per second, over the last ~6 s of 1 s polls. peak = the best such rate
  // since the server was seen; per-stream = total / requests running.
  double gen_tps = 0, peak_tps = 0;
};

class Monitor {
 public:
  explicit Monitor(bool live) : live_(live) {
    if (live_) th_ = std::thread([this] { run(); });
  }
  ~Monitor() {
    stop_ = true;
    if (th_.joinable()) th_.join();
  }
  Snapshot get() {
    std::lock_guard<std::mutex> g(mu_);
    return s_;
  }
  void watch(pid_t pid, bool child, int port) {
    std::lock_guard<std::mutex> g(mu_);
    s_.pid = pid;
    child_ = child;
    s_.port = port;
    s_.ours_alive = pid > 0;
    s_.exit_code = INT_MIN;
    s_.exit_why.clear();
    killed_here_ = false;
    s_.ready = false;
    s_.stopping = false;
    s_.started = Clock::now();
    s_.adopted = !child;
    if (!child && pid > 0) {
      const double age = proc_age(pid);
      if (age > 0) s_.started -= std::chrono::milliseconds(static_cast<long long>(age * 1000));
    }
    s_.metrics.clear();
    s_.metrics_ok = false;
    s_.gen_tps = s_.peak_tps = 0;
    gens_.clear();
  }
  void mark_stopping() {
    std::lock_guard<std::mutex> g(mu_);
    s_.stopping = true;
    s_.stop_sent = Clock::now();
  }
  void mark_killed() { killed_here_ = true; }  // the launcher sent SIGKILL (exit_reason says so instead of "another process")
  void set_idle_port(int port) {
    std::lock_guard<std::mutex> g(mu_);
    idle_port_ = port;
  }
  void kick_docker() { docker_due_ = Clock::time_point{}; }

 private:
  void run() {
    Clock::time_point procs_due{}, ready_due{}, metrics_due{}, gpu_due{};
    while (!stop_) {
      const auto now = Clock::now();
      if (now >= gpu_due) {
        gpu_due = now + std::chrono::milliseconds(1000);
        sample_gpus(now);
      }
      pid_t pid;
      bool child;
      int port, idle_port;
      {
        std::lock_guard<std::mutex> g(mu_);
        pid = s_.pid;
        child = child_;
        port = s_.port;
        idle_port = idle_port_;
      }
      bool alive = false;
      int code = INT_MIN;
      if (pid > 0) {
        if (child) {
          int st = 0;
          const pid_t r = waitpid(pid, &st, WNOHANG);
          if (r == pid) code = WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1000;
          else alive = r == 0;
          if (r < 0) code = -1000;
        } else {
          alive = pid_alive(pid);
          if (!alive) code = -1000;
        }
      }
      std::vector<ProcInfo> servers;
      const bool do_procs = now >= procs_due;
      if (do_procs) {
        servers = find_servers();
        procs_due = now + std::chrono::milliseconds(1000);
      }
      std::vector<std::string> cont;
      bool dok = true;
      const bool do_docker = now >= docker_due_.load();
      if (do_docker) {
        cont = docker_containers(dok);
        docker_due_ = now + std::chrono::seconds(5);
      }
      // Readiness and metrics of the server on the watched port.
      int wport = alive ? port : 0;
      if (!wport) {
        std::lock_guard<std::mutex> g(mu_);
        const auto &sv = do_procs ? servers : s_.servers;
        if (!sv.empty()) wport = sv[0].port;
      }
      bool ready = false, have_ready = false, have_metrics = false;
      std::map<std::string, double> m;
      if (wport && now >= ready_due) {
        have_ready = true;
        try {
          ready = http::request("GET", "http://127.0.0.1:" + std::to_string(wport) + "/v1/models", "", 0.8).status == 200;
        } catch (...) {
          ready = false;
        }
        ready_due = now + std::chrono::milliseconds(ready ? 1000 : 500);
        if (ready && now >= metrics_due) {
          try {
            const auto r = http::request("GET", "http://127.0.0.1:" + std::to_string(wport) + "/metrics", "", 1.0);
            if (r.status == 200) {
              m = parse_metrics(r.body);
              have_metrics = true;
            }
          } catch (...) {
          }
          metrics_due = now + std::chrono::milliseconds(1000);
        }
      }
      bool explain = false;
      {
        std::lock_guard<std::mutex> g(mu_);
        if (s_.pid == pid) {  // not replaced by a new launch meanwhile
          if (pid > 0) {
            s_.ours_alive = alive;
            if (!alive && code != INT_MIN && s_.exit_code == INT_MIN) {
              s_.exit_code = code;
              explain = code != 0;
            }
          }
        }
        if (do_procs) s_.servers = servers;
        if (do_docker) {
          s_.containers = cont;
          s_.docker_ok = dok;
          s_.docker_known = true;
        }
        if (have_ready) {
          if (ready && !s_.ready) s_.ready_at = Clock::now();
          s_.ready = ready;
          if (!ready) s_.metrics_ok = false;
        }
        if (!wport) s_.ready = false;
        if (have_metrics) {
          const double gen = m.count("vllm:generation_tokens_total") ? m["vllm:generation_tokens_total"] : 0;
          const auto t = Clock::now();
          if (!gens_.empty() && gen < gens_.back().second) gens_.clear();  // the server restarted: its counter began again
          gens_.push_back({t, gen});
          while (gens_.size() > 2 && std::chrono::duration<double>(t - gens_[1].first).count() >= 6.0) gens_.erase(gens_.begin());
          s_.gen_tps = 0;
          if (gens_.size() > 1) {
            const double dt = std::chrono::duration<double>(t - gens_.front().first).count();
            if (dt > 0) s_.gen_tps = (gen - gens_.front().second) / dt;
          }
          s_.peak_tps = std::max(s_.peak_tps, s_.gen_tps);
          s_.metrics = std::move(m);
          s_.metrics_ok = true;
        }
      }
      if (explain) {  // outside the lock: exit_reason may read the kernel log (journalctl, up to a few seconds)
        Clock::time_point started;
        int ecode;
        {
          std::lock_guard<std::mutex> g(mu_);
          started = s_.started;
          ecode = s_.exit_code;
        }
        const long long since = static_cast<long long>(std::time(nullptr)) -
                                static_cast<long long>(std::chrono::duration<double>(Clock::now() - started).count()) - 60;
        std::string why = exit_reason(pid, ecode, since, killed_here_);
        std::lock_guard<std::mutex> g(mu_);
        if (s_.pid == pid) s_.exit_why = std::move(why);
      }
      (void)idle_port;
      for (int i = 0; i < 5 && !stop_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  bool live_;
  std::atomic<bool> stop_{false};
  std::atomic<Clock::time_point> docker_due_{Clock::time_point{}};
  std::thread th_;
  std::mutex mu_;
  // Board power of every xe card from its energy counter (microjoules): watts = delta / seconds.
  void sample_gpus(Clock::time_point now) {
    std::vector<GpuInfo> infos = read_gpus();
    std::lock_guard<std::mutex> g(mu_);
    std::vector<GpuLive> next;
    for (auto &info : infos) {
      GpuLive gl;
      for (const auto &old : s_.gpus)
        if (old.info.card == info.card) gl.watts = old.watts;
      const auto prev = gpu_prev_.find(info.card);
      if (info.have_energy && prev != gpu_prev_.end()) {
        const double w = power_between(prev->second.first, info.energy_uj, std::chrono::duration<double>(now - prev->second.second).count());
        // A board cannot draw more than its power cap (plus a burst margin): a reading past that
        // means the interval or counter was bad — keep the last good value instead of showing a spike.
        const bool capped = info.cap_w >= 50.0;  // a real GPU power limit (gpu-power.sh clamps 50..peak)
        if (w >= 0 && (!capped || w <= info.cap_w + 100.0)) gl.watts = w;
      }
      if (info.have_energy) gpu_prev_[info.card] = {info.energy_uj, now};
      gl.info = std::move(info);
      next.push_back(std::move(gl));
    }
    s_.gpus = std::move(next);
  }

  std::map<int, std::pair<uint64_t, Clock::time_point>> gpu_prev_;
  Snapshot s_;
  bool child_ = false;
  int idle_port_ = 0;
  std::atomic<bool> killed_here_{false};
  std::vector<std::pair<Clock::time_point, double>> gens_;  // (poll time, generation_tokens_total)
};

// ---- log tail -------------------------------------------------------------------------------

struct LogTail {
  std::string path;
  std::streamoff off = 0;
  std::string partial;
  std::deque<std::string> lines;
  size_t total = 0;  // lines ever read (for change detection)

  void reset(const std::string &p) {
    path = p;
    off = 0;
    partial.clear();
    lines.clear();
    ++total;
  }
  void poll() {
    if (path.empty()) return;
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return;
    if (st.st_size < off) {  // truncated: a new run
      off = 0;
      partial.clear();
      lines.clear();
    }
    if (st.st_size == off) return;
    std::ifstream f(path, std::ios::binary);
    f.seekg(off);
    std::string chunk(static_cast<size_t>(std::min<std::streamoff>(st.st_size - off, 4 << 20)), '\0');
    f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    chunk.resize(static_cast<size_t>(f.gcount()));
    off += static_cast<std::streamoff>(chunk.size());
    partial += chunk;
    size_t p;
    while ((p = partial.find('\n')) != std::string::npos) {
      std::string l = partial.substr(0, p);
      if (!l.empty() && l.back() == '\r') l.pop_back();
      // A progress line redrawn with '\r': keep its last state.
      const size_t cr = l.rfind('\r');
      if (cr != std::string::npos) l = l.substr(cr + 1);
      lines.push_back(std::move(l));
      partial.erase(0, p + 1);
      ++total;
    }
    while (lines.size() > 5000) lines.pop_front();
  }
};

Style log_style(const std::string &l) {
  std::string s = l;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s.find("error") != std::string::npos || s.find("fail") != std::string::npos || s.find("abort") != std::string::npos ||
      s.find("terminate") != std::string::npos || s.find("refus") != std::string::npos || s.find("unknown option") != std::string::npos ||
      s.find("needs a value") != std::string::npos || s.find("out of memory") != std::string::npos)
    return th::err;
  if (s.rfind("launcher: the server", 0) == 0 || s.find("server: crashed") != std::string::npos) return th::err;
  if (s.find("\xE2\x9C\x97") != std::string::npos) return th::err;  // a failed check (xenithium-bench marks wrong answers with it)
  if (s.find("warn") != std::string::npos || s.rfind("start-server:", 0) == 0) return th::warn;
  if (s.find("server: serving") != std::string::npos || s.find("server: stub engine") != std::string::npos) return th::ok;
  if (s.rfind("server:", 0) == 0) return th::acc2;
  return th::plain;
}

struct Rect {
  int y = 0, x = 0, h = 0, w = 0;
  bool in(int yy, int xx) const { return yy >= y && yy < y + h && xx >= x && xx < x + w; }
};

}  // namespace

// ---- the app --------------------------------------------------------------------------------

struct App::Impl {
  // The tabs, in the header's order. The Chat tab is hidden for now: the screen, its keys and its tests stay in the
  // tree and the flag alone brings it back. A hidden tab keeps its enum number and loses its header slot - the digits
  // name the visible tabs in order, so hiding Chat leaves Settings 1, Logs 2, Benchmark 3.
  enum Tab { kSettings = 0, kServer = 1, kChat = 2, kBench = 3, kTabs = 4 };
  static constexpr bool kShowChatTab = false;
  static bool tab_shown(int t) { return kShowChatTab || t != kChat; }
  // The (pos+1)-th visible tab number, or -1 when the digit stands past the last tab.
  static int shown_tab_at(int pos) {
    for (int t = 0; t < kTabs; ++t) {
      if (!tab_shown(t)) continue;
      if (pos-- == 0) return t;
    }
    return -1;
  }
  // The neighbor of `from` in the visible cycle (a hidden tab sits between: from Settings, BackTab reaches Benchmark).
  int next_shown_tab(int from, bool back) const {
    int t = from;
    for (int i = 0; i < kTabs; ++i) {
      t = (t + (back ? kTabs - 1 : 1)) % kTabs;
      if (tab_shown(t)) return t;
    }
    return from;
  }
  AppConfig cfg;
  bool models_dir_cli = false;  // --models was passed: it wins over the settings file's XEN_MODELS_DIR
  Catalog cat;
  Values vals, saved, defs;
  std::string target;  // the env file being edited
  // Model mode (server.env opened): the settings are the current model's per-model file (cfg.model_envs_dir/<id>.env,
  // `target`), server.env only names it (XEN_MODEL_ENV); a model switch opens the new model's file. Off for a profile.
  bool model_mode = false;
  // Per-setup settings. In model mode a model with setups (setup_key_of: its card count) keeps the
  // shared settings in `target` and the rest in the current setup's file (`setup_target, models/<id>.tp<N>.env), over
  // the model's file (`model_file_vals, as read); "" = no setups (other archs, a profile).
  std::string setup_target;
  Values model_file_vals;
  std::vector<Values> undo;
  ModelInfo model;
  std::vector<ModelInfo> models;
  bool models_scanned = false;
  int tab_ = 0;

  // Configure tab.
  struct Row {
    bool group = false;
    std::string name;  // group name
    int spec = -1;
    int hidden = 0;  // > 0: the "N settings hidden" line
  };
  std::vector<Row> rows;
  int sel = 0, top = 0, help_scroll = 0;
  std::set<std::string> collapsed{"Numerics & kernels", "Diagnostics", "Other"};
  bool show_na = false;
  bool filtering = false;
  LineEdit filter;
  bool editing = false;
  LineEdit edit;
  std::string edit_err;

  // Preview (the script's dry run) and validation.
  std::string preview_cmd, preview_err;
  Values preview_for;
  std::vector<Issue> issues;
  Values issues_for;
  std::string issues_arch;
  // The memory plan (xenithium-server --plan through start-server.sh SERVE_PLAN=1: the load's own arithmetic, no GPU,
  // ~0.4 s), computed off the UI thread for the launch values; plan_gen moves with each result. Every arch has one
  // since (the dense runtime's is an estimate fitted on measured loads).
  MemPlanView plan;
  std::string plan_err;
  Values plan_for;
  std::future<RunResult> plan_job;
  int plan_gen = 0;
  // The KV pool by value of the selected setting (moves_memory): one plan per value, in parallel off the UI thread,
  // started once the selection has rested (kWhatIfRestMs) and kept until the setting or another value changes.
  struct WhatIfRow {
    std::string value;
    bool ok = false;
    MemPlanView plan;
    std::string err;
  };
  static constexpr int kWhatIfRestMs = 250;
  std::vector<WhatIfRow> whatif;
  std::string whatif_key, whatif_job_key, whatif_want_key;
  Values whatif_base, whatif_job_base, whatif_want_base;
  Clock::time_point whatif_since{};
  std::future<std::vector<WhatIfRow>> whatif_job;

  // Server tab.
  Monitor mon;
  LogTail log;
  int log_scroll = 0;  // display lines up from the bottom; 0 = follow
  bool log_wrap = true;
  std::string lan;

  // Chat tab.
  ChatSession chat;
  LineEdit chat_input;  // always focused on this tab: typing is the point
  int chat_scroll = 0;  // display lines up from the bottom; 0 = follow
  int chat_port = 0;  // test override (0 = the running server's port, or XEN_PORT in tests)
  int chat_rows_visible = 10;

  // Benchmark tab.
  BenchSession bench;
  struct BenchLine {
    bool header = false;
    std::string text;  // the model name (header) or the run's one-line summary
    int run = -1;  // index into bench_runs (run rows); -1 for headers
  };
  std::vector<BenchLine> bench_lines;  // display rows, rebuilt after a scan
  std::vector<const BenchRun *> bench_runs;  // flat run pointers, in display order
  int bench_sel = 0, bench_scroll = 0, bench_rows_visible = 10;
  bool bench_detail = false;  // showing a run's report
  int bench_detail_run = -1;  // index into bench_runs
  LogTail bench_log;  // the run's output (state_dir/bench.log), tailed in the lower panel
  int bench_log_scroll = 0;  // display lines up from the bottom; 0 = follow
  std::string bench_streams_last = "1,8,16,32";  // the Streams mode's last stream counts (the input's start value)
  bool bench_log_wrap = true;

  void rebuild_bench() {
    bench.scan(cfg.bench_runs);
    bench_lines.clear();
    bench_runs.clear();
    int ri = 0;
    for (const auto &m : bench.models()) {
      bench_lines.push_back({true, m.name + "   (" + std::to_string(m.runs.size()) + " run" +
                                    (m.runs.size() == 1 ? "" : "s") + ")", -1});
      for (const auto &r : m.runs) {
        std::string t = r.dir;
        if (r.decode_tps > 0) t += "   " + fmt1(r.decode_tps) + " dec";
        if (!r.prefill.empty() && r.prefill.back().second > 0) t += "   " + fmt1(r.prefill.back().second) + " pp@" + std::to_string(r.prefill.back().first);
        if (!r.deep_decode.empty() && r.deep_decode.back().second > 0)
          t += "   " + fmt1(r.deep_decode.back().second) + " dec@" + std::to_string(r.deep_decode.back().first);
        if (!r.concurrency.empty() && r.concurrency.back().second > 0) t += "   " + fmt1(r.concurrency.back().second) + " conc@" + std::to_string(r.concurrency.back().first);
        if (r.quality_pct >= 0) t += "   " + fmt1(r.quality_pct) + "% quality";
        bench_lines.push_back({false, t, ri});
        bench_runs.push_back(&r);
        ++ri;
      }
    }
    bench_sel = std::clamp(bench_sel, 0, std::max(0, static_cast<int>(bench_lines.size()) - 1));
  }

  // The hot-expert ranking job: scripts/make-hot-rank.sh in its own session (output to state_dir/rank.log, pid in
  // state_dir/rank.pid); adopted when one is already running, polled every frame for its result.
  Job rank_job;
  LogTail rank_log;
  std::string rank_job_model, rank_job_out;  // what the running / last job makes
  int64_t rank_job_t0 = 0;  // wall time it started: the file it leaves must be newer
  bool welcome = false;  // the env file did not exist at start: a first-run hint

  std::vector<Popup> pops;
  std::string toast;
  Style toast_st;
  Clock::time_point toast_until{};
  bool quit = false;
  int ctrl_c = 0;

  // Mouse targets from the last frame.
  Rect r_list, r_detail, r_log, r_bench_log;
  std::vector<std::pair<int, int>> row_at;  // (screen y, row index)
  std::vector<std::pair<Rect, int>> tab_at;
  int list_rows_visible = 10, log_rows_visible = 10;

  explicit Impl(AppConfig c) : cfg(std::move(c)), mon(cfg.live) {
    models_dir_cli = !cfg.models_dir.empty();  // a --models DIR was passed (read from cfg: c is already moved)
    if (cfg.env_path.empty()) cfg.env_path = join_path(cfg.root, "server.env");
    if (cfg.state_dir.empty()) cfg.state_dir = join_path(cfg.root, "build/launch");
    if (cfg.profiles_dir.empty()) cfg.profiles_dir = join_path(cfg.root, "profiles");
    if (cfg.bench_bin.empty()) cfg.bench_bin = join_path(cfg.root, "build/xenithium-bench");
    if (cfg.bench_runs.empty()) {
      const char *runs = std::getenv("XEN_BENCH_RUNS");  // the variable xenithium-bench itself reads
      cfg.bench_runs = runs && *runs ? std::string(runs) : join_path(home_dir(), ".xenithium/bench/runs");
    }
    if (cfg.experts_dir.empty()) cfg.experts_dir = join_path(cfg.root, ".cache/experts");
    if (cfg.model_envs_dir.empty()) cfg.model_envs_dir = join_path(cfg.root, "models");
    welcome = cfg.live && !is_file(cfg.env_path);
    cat = load_catalog(cfg.root);
    defs = defaults_of(cat);
    open_file(cfg.env_path);
    edit.root = cfg.root;
    lan = cfg.live ? lan_ip() : "";
    if (cfg.live) adopt_running();
    if (cfg.live) adopt_rank_job();
    rank_log.reset(join_path(cfg.state_dir, "rank.log"));
    rebuild_rows();
    rebuild_bench();
    bench_log.reset(join_path(cfg.state_dir, "bench.log"));  // tail it even before the first run
    if (!cat.warnings.empty()) say(cat.warnings[0] + (cat.warnings.size() > 1 ? " (+" + std::to_string(cat.warnings.size() - 1) + " more)" : ""), th::warn, 8);
    if (welcome) show_welcome();
  }

  // ---- state helpers

  std::string val(const std::string &k) const {
    const auto it = vals.find(k);
    return it == vals.end() ? std::string() : it->second;
  }
  const Spec &spec(int i) const { return cat.specs[static_cast<size_t>(i)]; }
  int spec_index(const std::string &key) const {
    for (size_t i = 0; i < cat.specs.size(); ++i)
      if (cat.specs[i].key == key) return static_cast<int>(i);
    return -1;
  }
  int dirty_count() const {
    int n = 0;
    for (const auto &s : cat.specs) {
      const auto a = vals.find(s.key), b = saved.find(s.key);
      if ((a == vals.end() ? "" : a->second) != (b == saved.end() ? "" : b->second)) ++n;
    }
    return n;
  }
  bool is_dirty(const std::string &k) const {
    const auto a = vals.find(k), b = saved.find(k);
    return (a == vals.end() ? "" : a->second) != (b == saved.end() ? "" : b->second);
  }
  bool is_modified(const std::string &k) const {
    const auto d = defs.find(k);
    return val(k) != (d == defs.end() ? "" : d->second);
  }
  // A settings file's state beside its name: the user's own file, the shipped preset (a save makes the user's copy), new.
  static std::string file_state(const std::string &user_path) {
    return is_file(user_path) ? "" : is_file(preset_env_path(user_path)) ? " (preset)" : " (new)";
  }
  std::string target_name() const {
    if (model_mode)
      return "models/" + model_env_id(target) + ".env" + file_state(target) +
             (setup_target.empty() ? "" : " + " + model_env_id(setup_target) + ".env" + file_state(setup_target));
    const size_t sl = target.find_last_of('/');
    std::string n = sl == std::string::npos ? target : target.substr(sl + 1);
    if (target.rfind(cfg.profiles_dir + "/", 0) != 0) return n;
    if (n.size() > 4 && n.substr(n.size() - 4) == ".env") n.resize(n.size() - 4);
    return "profile " + n;
  }
  void say(const std::string &m, Style st = th::plain, int secs = 4) {
    toast = m;
    toast_st = st;
    toast_until = Clock::now() + std::chrono::seconds(secs);
  }

  void open_file(const std::string &path) {
    if (is_pointer_file(path)) {
      open_model_mode(path);
      return;
    }
    model_mode = false;
    std::string text;
    const bool exists = read_text(path, text);
    const Values file = EnvFile::parse(exists ? text : "").values();
    vals = defs;
    for (const auto &[k, v] : file)
      if (cat.find(k)) vals[k] = v;
    target = in_presets(path) ? user_env_path(path) : path;  // a preset is read, never written
    undo.clear();
    refresh_model();
    sync_served_name(false);  // a blank / default served name takes the checkpoint's folder name
    saved = vals;
  }

  // server.env: the model it starts and that model's own settings file (read_model_env: as scripts/start-server.sh).
  // Settings server.env itself still holds (a file from before the per-model files) win, show as unsaved and move to the
  // model's file on the next save.
  // A settings file that starts a model with its own settings: one named server.env, or one holding XEN_MODEL_ENV. A
  // profile (profiles/NAME.env) is a whole settings file and is edited as it is.
  bool is_pointer_file(const std::string &path) const { return is_pointer_env(path); }
  void open_model_mode(const std::string &path) {
    cfg.env_path = path;  // the pointer the saves update
    const ModelEnvView mv = read_model_env(cat, cfg.root, cfg.model_envs_dir, path);
    model_mode = true;
    vals = defs;
    for (const auto &[k, v] : mv.file_values) vals[k] = v;
    model_file_vals = mv.file_values;
    vals["XEN_MODEL"] = mv.model;
    target = mv.file.empty() ? model_env_new_path(cfg.model_envs_dir, mv.model) : mv.file;
    undo.clear();
    refresh_model();
    if (const std::string k = setup_key_of(model.arch); !k.empty() && mv.env_values.count(k)) vals[k] = mv.env_values.at(k);  // server.env picks the setup
    attach_setup();
    sync_served_name(false);
    int moved = 0;
    for (const auto &[k, v] : mv.env_values)
      if (k != "XEN_MODEL" && val(k) != v) {
        vals[k] = v;
        if (k != "XEN_MODELS_DIR") ++moved;  // a global: it stays in the pointer file, not moved to the model's file
      }
    saved = vals;  // as start-server.sh would launch them; the next save writes them to the model's file
    if (moved) say(std::to_string(moved) + " setting" + (moved == 1 ? "" : "s") + " from server.env: s moves them to " + target_name(), th::warn, 8);
  }

  // Model mode: the model's own settings file becomes the one edited (the defaults where it has none yet).
  void switch_model(const std::string &model_path, const std::string &file_hint = "") {
    std::string f = file_hint.empty() ? find_model_env(cfg.model_envs_dir, model_path) : file_hint;
    if (f.empty()) f = model_env_new_path(cfg.model_envs_dir, model_path);
    std::string text;
    const bool exists = read_text(is_file(f) ? f : readable_env_path(f), text);  // the user's file, else the shipped preset
    f = user_env_path(f);  // a preset is never written: the saves go to the user's file of the same name
    const std::string keep_models_dir = model_mode ? val("XEN_MODELS_DIR") : "";  // a global in the pointer file: it carries over
    vals = defs;
    model_file_vals.clear();
    if (exists)
      for (const auto &[k, v] : EnvFile::parse(text).values())
        if (cat.find(k)) vals[k] = v, model_file_vals[k] = v;
    if (model_mode && !keep_models_dir.empty()) vals["XEN_MODELS_DIR"] = keep_models_dir;
    vals["XEN_MODEL"] = model_path;
    target = f;
    undo.clear();
    const std::string before = model.arch;
    refresh_model();
    attach_setup();
    sync_served_name(!exists);
    saved = vals;
    rebuild_rows();
    say("model " + model.name + ": " + (exists ? "its settings from " : "no settings file yet (defaults; s creates ") + target_name() + (exists ? "" : ")") +
            (!model.arch.empty() && model.arch != before ? " - " + model.arch : ""),
        th::acc2, 6);
  }
  void request_model_switch(const std::string &model_path) {
    if (same_path(model_path, val("XEN_MODEL"))) return;
    const int n = dirty_count();
    if (!n) {
      switch_model(model_path);
      return;
    }
    confirm("Unsaved changes", {std::to_string(n) + " unsaved change" + (n == 1 ? "" : "s") + " in " + target_name() + "."},
            {{"Save them, then switch", [this, model_path] {
                if (save_to(target)) switch_model(model_path);
              }},
             {"Discard them and switch", [this, model_path] { switch_model(model_path); }},
             {"Cancel", nullptr}},
            th::warn);
  }

  void refresh_model() {
    const std::string m = val("XEN_MODEL");
    if (model.path != m || model.path.empty()) model = probe_model(m);
  }

  // ---- per-setup settings
  std::string setup_key() const { return model_mode ? setup_key_of(model.arch) : ""; }
  // The setup part of vals (every setting but the shared ones): the setup file's value, else the model file's, else
  // the default - the order scripts/start-server.sh applies them in.
  void load_setup(const std::string &path) {
    setup_target = path;
    std::string text;
    const Values file = read_text(readable_env_path(path), text) ? EnvFile::parse(text).values() : Values{};
    for (const auto &s : cat.specs) {
      if (is_shared_setting(s)) continue;
      const auto f = file.find(s.key);
      const auto m = model_file_vals.find(s.key);
      const auto d = defs.find(s.key);
      vals[s.key] = f != file.end() ? f->second : m != model_file_vals.end() ? m->second : d != defs.end() ? d->second : "";
    }
  }
  // After a model file is read: its current setup's file over it (none for a model without setups).
  void attach_setup() {
    setup_target.clear();
    const std::string k = setup_key();
    if (k.empty()) return;
    const std::string n = val(k).empty() ? "2" : val(k);  // start-server.sh's default setup
    load_setup(setup_env_path(target, n));
  }
  int dirty_setup_count() const {
    int n = 0;
    for (const auto &s : cat.specs)
      if (!is_shared_setting(s) && is_dirty(s.key)) ++n;
    return n;
  }
  // The setup key changed: its own settings file is loaded (the setup key itself stays unsaved until s writes it to
  // the model's file, so the next start picks that setup).
  void switch_setup(const std::string &v) {
    const std::string k = setup_key();
    undo.clear();
    vals[k] = v;
    load_setup(setup_env_path(target, v));
    for (const auto &s : cat.specs)
      if (!is_shared_setting(s)) saved[s.key] = vals[s.key];
    rebuild_rows();
    const bool exists = is_file(readable_env_path(setup_target));
    say("setup " + k + "=" + v + ": " + (exists ? "its settings from " : "no settings file yet (the model's; s creates ") + model_env_id(setup_target) + ".env" +
            (exists ? "" : ")"),
        th::acc2, 6);
  }
  void request_setup_switch(const std::string &v) {
    const int n = dirty_setup_count();
    if (!n) {
      switch_setup(v);
      return;
    }
    confirm("Unsaved changes", {std::to_string(n) + " unsaved change" + (n == 1 ? "" : "s") + " in " + model_env_id(setup_target) + ".env."},
            {{"Save them, then switch", [this, v] {
                if (save_to(target)) switch_setup(v);
              }},
             {"Discard them and switch", [this, v] { switch_setup(v); }},
             {"Cancel", nullptr}},
            th::warn);
  }

  // The served name (the model id in /v1/models and in responses) follows the checkpoint's folder name.
  // On a model switch it always takes it; on opening a file it takes it only when the file left it blank
  // or at the default, so an id typed there is kept.
  void sync_served_name(bool on_model_switch) {
    if (model.name.empty()) return;
    const std::string def = defs.count("XEN_SERVED_NAME") ? defs.at("XEN_SERVED_NAME") : "";
    const std::string cur = val("XEN_SERVED_NAME");
    if (on_model_switch || cur.empty() || cur == def) vals["XEN_SERVED_NAME"] = model.name;
  }

  void change(const std::string &key, const std::string &v) {
    if (val(key) == v) return;
    if (key == "XEN_MODEL" && model_mode) {  // each model keeps its own settings file
      request_model_switch(v);
      return;
    }
    if (!setup_key().empty() && key == setup_key()) {  // Each setup keeps its own settings file
      request_setup_switch(v);
      return;
    }
    if (key == "XEN_MODELS_DIR") models.clear(), models_scanned = false;  // the m list rescans the folder next time it opens
    undo.push_back(vals);
    if (undo.size() > 200) undo.erase(undo.begin());
    vals[key] = v;
    if (key != "XEN_MODEL") rebuild_rows();  // a switch can show or hide the settings that depend on it
    if (key == "XEN_MODEL") {
      const std::string before = model.arch;
      refresh_model();
      sync_served_name(true);  // the served name follows the new checkpoint's folder name
      rebuild_rows();
      if (!model.arch.empty() && model.arch != before) {
        int hidden = 0;
        for (const auto &s : cat.specs)
          if (!applies(s, model.arch)) ++hidden;
        say("model " + model.name + " (" + model.arch + "): " + std::to_string(hidden) + " settings of the other arch hidden", th::acc2);
      }
    }
  }

  bool plan_busy() const { return plan_job.valid() && plan_job.wait_for(std::chrono::seconds(0)) != std::future_status::ready; }
  // The plan for the current launch values: a result when the job finished, a new job when the values moved.
  void update_plan() {
    if (plan_job.valid() && !plan_busy()) {
      const RunResult r = plan_job.get();
      MemPlanView p;
      if (r.status == 0 && parse_plan(r.out, p)) {
        plan = p;
        plan_err.clear();
      } else {
        plan = {};
        plan_err = plan_error_of(r);
      }
      ++plan_gen;
    }
    if (model.arch.empty() || val("XEN_STUB") == "1") {
      if (!plan_job.valid()) plan = {}, plan_err.clear(), plan_for.clear();
      return;
    }
    const Values v = launch_values();
    if (v == plan_for || plan_job.valid()) return;
    plan_for = v;
    const std::string dir = plan_dir();
    const std::string f = join_path(dir, "plan.env"), root = cfg.root;
    std::string err;
    if (!write_text_atomic(f, launch_env_text(v), &err)) {
      plan = {}, plan_err = err;
      return;
    }
    plan_job = std::async(std::launch::async, [f, root] { return run_capture({"bash", "scripts/start-server.sh"}, {{"SERVE_ENV", f}, {"SERVE_PLAN", "1"}}, root, 15000); });
  }
  // The plan shown: for this model and the current values (else stale: being recomputed).
  const MemPlanView *shown_plan() const { return plan.ok && plan.arch == model.arch ? &plan : nullptr; }
  bool plan_stale() const { return plan_busy() || plan_for != launch_values(); }
  std::string plan_dir() const {  // the plans' env files (the launcher's state dir; tests keep theirs apart)
    const std::string dir = cfg.live ? cfg.state_dir : join_path(cfg.state_dir, "test");
    mkdir(join_path(cfg.root, "build").c_str(), 0755);
    mkdir(cfg.state_dir.c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    return dir;
  }
  // A failed plan's reason: the last line the script or the server wrote.
  static std::string plan_error_of(const RunResult &r) {
    std::string e = r.err;
    while (!e.empty() && e.back() == '\n') e.pop_back();
    if (const size_t nl = e.rfind('\n'); nl != std::string::npos) e = e.substr(nl + 1);
    if (e.rfind("server: ", 0) == 0) e = e.substr(8);
    return !e.empty() ? e : r.timed_out ? "the memory plan timed out" : "the memory plan failed (status " + std::to_string(r.status) + ")";
  }

  bool whatif_busy() const { return whatif_job.valid() && whatif_job.wait_for(std::chrono::seconds(0)) != std::future_status::ready; }
  // The drafter folders a path setting's table compares.
  std::vector<std::string> whatif_extra(const Spec &s) const {
    std::vector<std::string> x;
    if (s.key == "XEN_DRAFTER_MODEL")
      for (const auto &d : dflash_drafters(models_dir())) x.push_back(d.path);
    if (s.key == "XEN_MTP")
      for (const auto &d : drafters_for(model.path)) x.push_back(d);
    return x;
  }
  // The KV pool by value of the selected setting: a finished job's table, a new job when the selection or a value moved
  // and has rested (now = at once: --render and tests).
  void update_whatif(bool now = false) {
    if (whatif_job.valid() && !whatif_busy()) {
      whatif = whatif_job.get();
      whatif_key = whatif_job_key, whatif_base = whatif_job_base;
    }
    const Spec *s = sel_spec();
    if (!s || !moves_memory(s->key) || model.arch.empty() || val("XEN_STUB") == "1" || editing || !unused_reason(*s, model.arch, vals).empty()) {
      whatif_want_key.clear();
      return;
    }
    if (s->key == whatif_key && vals == whatif_base) return;  // the table shown is this one
    if (s->key != whatif_want_key || vals != whatif_want_base) whatif_want_key = s->key, whatif_want_base = vals, whatif_since = Clock::now();
    if (whatif_job.valid()) return;  // one at a time: the next starts when this one is in
    if (!now && std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - whatif_since).count() < kWhatIfRestMs) return;
    const std::vector<std::string> values = whatif_values(*s, val(s->key), whatif_extra(*s));
    whatif_job_key = s->key, whatif_job_base = vals;
    if (values.empty()) {  // nothing to compare: an empty table for this setting
      whatif.clear(), whatif_key = whatif_job_key, whatif_base = whatif_job_base;
      return;
    }
    const std::string dir = plan_dir(), root = cfg.root;
    std::vector<std::pair<std::string, std::string>> files;  // value, its env file
    for (size_t i = 0; i < values.size(); ++i) {
      Values v = vals;
      v[s->key] = values[i];
      const std::string f = join_path(dir, "plan-value-" + std::to_string(i) + ".env");
      std::string err;
      if (!write_text_atomic(f, launch_env_text(launch_values_of(v)), &err)) return;
      files.push_back({values[i], f});
    }
    whatif_job = std::async(std::launch::async, [files, root] {
      std::vector<std::future<RunResult>> runs;
      for (const auto &[value, f] : files)
        runs.push_back(std::async(std::launch::async, [f = f, root] { return run_capture({"bash", "scripts/start-server.sh"}, {{"SERVE_ENV", f}, {"SERVE_PLAN", "1"}}, root, 15000); }));
      std::vector<WhatIfRow> rows;
      for (size_t i = 0; i < files.size(); ++i) {
        const RunResult r = runs[i].get();
        WhatIfRow row;
        row.value = files[i].first;
        row.ok = r.status == 0 && parse_plan(r.out, row.plan);
        if (!row.ok) row.err = plan_error_of(r);
        rows.push_back(std::move(row));
      }
      return rows;
    });
  }
  // The table for the selected setting when it is current (else nullptr: none, or being computed).
  const std::vector<WhatIfRow> *shown_whatif() const {
    const Spec *s = sel_spec();
    return s && s->key == whatif_key && vals == whatif_base && !whatif.empty() ? &whatif : nullptr;
  }

  const std::vector<Issue> &current_issues() {
    const bool srv = !snap().servers.empty();
    const std::string arch_key = model.arch + "/" + std::to_string(max_context_of(model)) + "/" + std::to_string(max_hot_of(model)) + "/" + std::to_string(trained_positions_of(model)) + "/" + std::to_string(plan_gen) +
                                 (srv ? "/srv" : "");
    if (issues_for != vals || issues_arch != arch_key) {
      issues = validate(cat, vals, model.arch, cfg.root, max_context_of(model), max_hot_of(model), trained_positions_of(model));
      if (const MemPlanView *p = shown_plan(); p && !plan_stale())
        for (auto &i : plan_issues(*p, launch_values(), srv)) issues.push_back(std::move(i));
      issues_for = vals;
      issues_arch = arch_key;
    }
    return issues;
  }
  int error_count() {
    int n = 0;
    for (const auto &i : current_issues())
      if (i.error) ++n;
    return n;
  }

  // The values a launch uses: settings that do not apply to the model go back to their default,
  // so the command line carries only what the model reads.
  Values launch_values() const { return launch_values_of(vals); }
  Values launch_values_of(const Values &from) const {
    Values v = from;
    for (const auto &s : cat.specs)
      if (!unused_reason(s, model.arch, from).empty()) v[s.key] = s.allow_empty ? "" : defs.count(s.key) ? defs.at(s.key) : "";
    return v;
  }
  std::string launch_env_text(const Values &v) const {
    std::string out = "# Written by xenithium-launch from " + tilde(target) + "; read by scripts/start-server.sh (SERVE_ENV).\n"
                      "XEN_MODEL_ENV=\n";  // every value is below: no per-model file under them
    for (const auto &s : cat.specs) {
      const std::string x = v.count(s.key) ? v.at(s.key) : "";
      if (s.diag && x.empty()) continue;
      out += s.key + "=" + env_quote(x) + "\n";
    }
    return out;
  }

  void update_preview() {
    const Values v = launch_values();
    if (v == preview_for) return;
    preview_for = v;
    const std::string dir = cfg.live ? cfg.state_dir : join_path(cfg.state_dir, "test");
    mkdir(join_path(cfg.root, "build").c_str(), 0755);
    mkdir(cfg.state_dir.c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    const std::string f = join_path(dir, "preview.env");
    std::string err;
    if (!write_text_atomic(f, launch_env_text(v), &err)) {
      preview_cmd.clear();
      preview_err = err;
      return;
    }
    const RunResult r = run_capture({"bash", "scripts/start-server.sh"}, {{"SERVE_ENV", f}, {"SERVE_DRY_RUN", "1"}}, cfg.root, 3000);
    preview_cmd = r.out;
    while (!preview_cmd.empty() && (preview_cmd.back() == '\n' || preview_cmd.back() == ' ')) preview_cmd.pop_back();
    const std::string h = home_dir();
    for (size_t p; !h.empty() && (p = preview_cmd.find(h)) != std::string::npos;) preview_cmd.replace(p, h.size(), "~");
    preview_err = r.err;
    while (!preview_err.empty() && preview_err.back() == '\n') preview_err.pop_back();
    if (r.status != 0 && preview_err.empty()) preview_err = r.timed_out ? "the dry run timed out" : "the dry run failed (status " + std::to_string(r.status) + ")";
  }

  // ---- rows

  // A filter matches names (label, key, flag, group); when no setting's name matches, the help
  // text is searched instead ("which setting is about drafts?").
  bool help_search = false;
  bool matches_filter(const Spec &s) const {
    if (filter.s.empty()) return true;
    std::string f = filter.s, hay = s.label + " " + s.key + " " + s.flag + " " + s.group;
    if (help_search) hay += " " + s.help;
    std::transform(f.begin(), f.end(), f.begin(), ::tolower);
    std::transform(hay.begin(), hay.end(), hay.begin(), ::tolower);
    return hay.find(f) != std::string::npos;
  }

  // A row is listed when the model uses the setting, when `a` shows the unused ones, or when a filter is typed and the
  // setting applies to this model's arch but waits on another one (Images by http(s) link while Image input is off): a
  // search finds it dimmed with its reason instead of answering "no setting matches". Settings of another arch stay
  // hidden until `a`.
  bool listed(const Spec &s) const {
    if (show_na || unused_reason(s, model.arch, vals).empty()) return true;
    return !filter.s.empty() && applies(s, model.arch);
  }

  void rebuild_rows() {
    std::string keep;
    if (sel >= 0 && sel < static_cast<int>(rows.size())) keep = rows[sel].group ? "#" + rows[sel].name : (rows[sel].spec >= 0 ? spec(rows[sel].spec).key : "");
    rows.clear();
    int hidden = 0;
    help_search = false;
    if (!filter.s.empty()) {
      bool any = false;
      for (const auto &sp : cat.specs)
        if (listed(sp) && sp.key != "XEN_MODEL" && matches_filter(sp)) any = true;
      help_search = !any;
    }
    for (const auto &g : cat.groups) {
      std::vector<int> members;
      for (size_t i = 0; i < cat.specs.size(); ++i) {
        const Spec &s = cat.specs[i];
        if (s.group != g || s.key == "XEN_MODEL") continue;
        if (!listed(s)) {
          ++hidden;
          continue;
        }
        if (!matches_filter(s)) continue;
        members.push_back(static_cast<int>(i));
      }
      if (members.empty()) continue;
      rows.push_back({true, g, -1, 0});
      if (collapsed.count(g) && filter.s.empty()) continue;
      for (const int i : members) rows.push_back({false, "", i, 0});
    }
    if (hidden > 0 && filter.s.empty()) rows.push_back({false, "", -1, hidden});
    int found = -1, first = -1;
    for (size_t i = 0; i < rows.size(); ++i) {
      const std::string k = rows[i].group ? "#" + rows[i].name : (rows[i].spec >= 0 ? spec(rows[i].spec).key : "");
      if (!keep.empty() && k == keep) found = static_cast<int>(i);
      if (first < 0 && !rows[i].group && rows[i].spec >= 0) first = static_cast<int>(i);
    }
    sel = found >= 0 ? found : first >= 0 ? first : 0;
    clamp_sel();
  }
  void clamp_sel() {
    if (rows.empty()) {
      sel = 0;
      return;
    }
    sel = std::clamp(sel, 0, static_cast<int>(rows.size()) - 1);
  }
  const Spec *sel_spec() const {
    if (sel < 0 || sel >= static_cast<int>(rows.size()) || rows[sel].group || rows[sel].spec < 0) return nullptr;
    return &spec(rows[sel].spec);
  }
  void move_sel(int d) {
    if (rows.empty()) return;
    sel = std::clamp(sel + d, 0, static_cast<int>(rows.size()) - 1);
    help_scroll = 0;
  }

  // ---- popups

  Popup &push(Popup p) {
    p.fix_sel();
    pops.push_back(std::move(p));
    return pops.back();
  }

  void message(const std::string &title, const std::vector<std::string> &lines, Style st = th::plain) {
    Popup p;
    p.title = title;
    p.selectable = false;
    for (const auto &l : lines) {
      Item it;
      it.label = l;
      it.label_st = st;
      p.items.push_back(it);
    }
    p.hints = {{"esc", "close"}};
    p.on_ok = [](Popup &q) { q.close = true; };
    push(std::move(p));
  }

  // Choices / suggestions of a setting.
  void open_choices(const Spec &s) {
    Popup p;
    p.title = s.label;
    p.note = s.key + (s.flag.empty() ? "" : "  " + s.flag);
    std::vector<std::pair<std::string, std::string>> opts;  // value, detail
    if (s.kind == Kind::Bool) opts = {{"1", ""}, {"0", ""}};
    for (const auto &c : s.choices) opts.push_back({c, ""});
    if (s.key == "XEN_MTP") {
      opts.push_back({"", ""});
      for (const auto &d : drafters_for(model.path)) opts.push_back({d, human_bytes(probe_model(d).bytes)});
    }
    if (s.key == "XEN_DRAFTER_MODEL") {
      opts.push_back({"", ""});
      for (const auto &d : dflash_drafters(models_dir())) opts.push_back({d.path, human_bytes(d.bytes)});
    }
    if (is_bank_key(s.key)) {  // the banks found, with their layers
      opts.push_back({"", "off"});
      for (const auto &b : banks()) opts.push_back({b.path, qsa_bank_detail(b)});
    }
    if (s.key == "XEN_VISION_FFMPEG") {  // The name on PATH (the default), off, and the programs found
      opts.push_back({"ffmpeg", check_value(s, "ffmpeg", model.arch, cfg.root, nullptr, 0).empty() ? "on PATH" : "not on PATH"});
      opts.push_back({"", ""});
      for (const auto &f : ffmpeg_candidates()) opts.push_back({f, "found"});
    }
    if (s.key == "XEN_HOT_RANK") {
      opts.push_back({"auto", s.key == "XEN_HOT_RANK" ? (rank_file().exists ? "ready" : "not created yet") : "the server writes it"});
      opts.push_back({"", ""});
    }
    if (s.key == "XEN_HOT_RANK") {
      // Rankings of every model under the experts folder (the shape check marks those of other models).
      std::string c3;
      for (const auto &d : complete_path(cfg.experts_dir + "/", "/", c3)) {
        if (d.empty() || d.back() != '/') continue;
        const std::string f = join_path(cfg.experts_dir, d + "hot-rank.xrc");
        if (!is_file(f)) continue;
        const ExpertFile ef = expert_file(f, "hot-rank.xrc", cfg.root, cfg.experts_dir, model);
        opts.push_back({f, ef.error.empty() ? "this model's shape" : "another model"});
      }
    }
    std::set<std::string> seen;
    const std::string cur = val(s.key), def = defs.count(s.key) ? defs.at(s.key) : "";
    for (const auto &[v, detail] : opts) {
      if (!seen.insert(v).second) continue;
      Item it;
      it.value = v;
      it.label = display_value(s, v);
      if (s.kind == Kind::Path && !v.empty()) it.label = tilde(v);
      if (it.label.empty()) it.label = "(empty)";
      it.detail = detail;
      // a setting that moves memory: the KV pool this value would give (the help panel's table, when it is in)
      if (const std::vector<WhatIfRow> *rows = shown_whatif())
        for (const auto &r : *rows)
          if (r.value == v) {
            const std::string kvt = !r.ok ? "will not load" : (r.plan.exact || r.plan.experts_total > 0 ? "" : "~") + with_commas(r.plan.kv_tokens) + " KV tokens" +
                                                (r.plan.experts_total > 0 ? ", " + with_commas(r.plan.hot_experts) + " hot" : "");
            it.detail += (it.detail.empty() ? "" : "  ") + kvt;
          }
      if (v == def) it.detail += (it.detail.empty() ? "" : "  ") + std::string("default");
      if (v == cur) {
        it.mark = "\xe2\x97\x8f";  // ●
        it.mark_st = th::acc;
        p.sel = static_cast<int>(p.items.size());
      }
      const std::string bad = check_value(s, v, model.arch, cfg.root, nullptr, max_context_of(model));
      if (!bad.empty() && s.kind != Kind::Path) {
        it.enabled = false;
        if (s.key == "XEN_CONTEXT") it.detail = "past max positions";
      }
      p.items.push_back(it);
    }
    const bool custom = s.kind != Kind::Choice && s.kind != Kind::Bool;
    if (custom) {
      Item it;
      it.label = s.kind == Kind::Path ? "Other path\xe2\x80\xa6" : "Custom value\xe2\x80\xa6";
      it.value = "\x01custom";
      it.label_st = th::acc2;
      p.items.push_back(it);
    }
    const std::string key = s.key;
    p.on_ok = [this, key](Popup &q) {
      const Item &it = q.items[q.sel];
      q.close = true;
      if (it.value == "\x01custom") {
        start_edit(*cat.find(key));
        return;
      }
      change(key, it.value);
    };
    p.hints = {{"\xe2\x86\x91\xe2\x86\x93", "move"}, {"\xe2\x8f\x8e", "pick"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  void start_edit(const Spec &s) {
    editing = true;
    edit.set(val(s.key));
    edit.path = s.kind == Kind::Path;
    edit.root = cfg.root;
    edit_err.clear();
  }

  void commit_edit() {
    const Spec *s = sel_spec();
    if (!s) {
      editing = false;
      return;
    }
    std::string v = edit.s;
    while (!v.empty() && v.back() == ' ') v.pop_back();
    while (!v.empty() && v.front() == ' ') v.erase(0, 1);
    if (s->kind == Kind::Path && !v.empty()) {
      v = expand_home(v);
      while (v.size() > 1 && v.back() == '/') v.pop_back();
    }
    bool warn = false;
    const std::string bad = check_value(*s, v, model.arch, cfg.root, &warn, max_context_of(model));
    if (!bad.empty() && !warn) {
      edit_err = bad;
      return;
    }
    editing = false;
    edit_err.clear();
    change(s->key, v);
    if (!bad.empty()) say(s->label + ": " + bad, th::warn);
  }

  void activate() {
    if (sel < 0 || sel >= static_cast<int>(rows.size())) return;
    const Row &r = rows[sel];
    if (r.group) {
      if (collapsed.count(r.name)) collapsed.erase(r.name);
      else collapsed.insert(r.name);
      rebuild_rows();
      return;
    }
    if (r.hidden > 0) {
      show_na = true;
      rebuild_rows();
      return;
    }
    const Spec &s = spec(r.spec);
    if (s.kind == Kind::Bool && s.choices.empty()) {
      change(s.key, val(s.key) == "1" ? "0" : "1");
      return;
    }
    if (s.kind == Kind::Choice || !s.choices.empty() || s.key == "XEN_MTP" || s.key == "XEN_HOT_RANK" || s.key == "XEN_DRAFTER_MODEL" || is_bank_key(s.key)) {
      open_choices(s);
      return;
    }
    start_edit(s);
  }

  // Left / Right on a setting: the previous / next choice or suggestion.
  void cycle(int d) {
    const Spec *s = sel_spec();
    if (!s) {
      if (sel < static_cast<int>(rows.size()) && rows[sel].group) {
        if (d < 0) collapsed.insert(rows[sel].name);
        else collapsed.erase(rows[sel].name);
        rebuild_rows();
      }
      return;
    }
    std::vector<std::string> opts = s->choices;
    if (s->kind == Kind::Bool) opts = {"0", "1"};
    if (s->key == "XEN_MTP") {
      opts = {""};
      for (const auto &x : drafters_for(model.path)) opts.push_back(x);
    }
    if (s->key == "XEN_DRAFTER_MODEL") {
      opts = {""};
      for (const auto &x : dflash_drafters(models_dir())) opts.push_back(x.path);
    }
    if (is_bank_key(s->key)) {
      opts = {""};
      for (const auto &b : banks()) opts.push_back(b.path);
    }
    if (s->key == "XEN_HOT_RANK") opts = {"auto", ""};
    if (opts.empty()) return;
    const std::string cur = val(s->key);
    auto it = std::find(opts.begin(), opts.end(), cur);
    int i = it == opts.end() ? (d > 0 ? -1 : 0) : static_cast<int>(it - opts.begin());
    for (size_t n = 0; n < opts.size(); ++n) {
      i = (i + d + static_cast<int>(opts.size())) % static_cast<int>(opts.size());
      if (check_value(*s, opts[i], model.arch, cfg.root, nullptr, max_context_of(model)).empty() || s->kind == Kind::Path) break;
    }
    change(s->key, opts[i]);
  }

  // The models folder (the m list and the drafter lists scan it): a --models DIR for one run wins; else the
  // settings file's XEN_MODELS_DIR (live: editing it rescans the next time the list opens); empty = the
  // project's models/ folder.
  std::string models_dir() const {
    if (models_dir_cli) return cfg.models_dir;
    std::string d = expand_home(val("XEN_MODELS_DIR"));
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return d.empty() ? join_path(cfg.root, "models") : d;
  }

  // Trained page selection banks: the folders of the models folder and of the model's own parent folder (where its
  // drafters and bank usually sit) that hold indexer tensors.
  static bool is_bank_key(const std::string &k) { return k == "XEN_TRAINED_QSA_DELTA_DIR"; }
  std::vector<QsaBank> banks() const {
    std::vector<std::string> dirs = {models_dir()};
    if (const size_t sl = model.path.find_last_of('/'); sl != std::string::npos && sl > 0) dirs.push_back(model.path.substr(0, sl));
    return qsa_banks(dirs);
  }

  void open_models() {
    if (!models_scanned) {
      models = scan_models(models_dir());
      models_scanned = true;
    }
    Popup p;
    p.title = "Model";
    p.note = tilde(models_dir());
    p.filterable = true;
    p.width = 84;
    const std::string cur = val("XEN_MODEL");
    // one group per arch in the registry's order, then the checkpoints no arch serves
    std::vector<std::pair<std::string, std::vector<const ModelInfo *>>> groups;
    for (const auto &e : model::arch_registry()) groups.push_back({std::string(e.id), {}});
    groups.push_back({"", {}});
    auto entry_of = [](const std::string &id) -> const model::ArchEntry * {
      for (const auto &e : model::arch_registry())
        if (e.id == id) return &e;
      return nullptr;
    };
    for (const auto &m : models) {
      if (m.drafter || m.dflash) continue;
      for (auto &g : groups)
        if (g.first == m.arch || (g.first.empty() && !entry_of(m.arch))) {
          g.second.push_back(&m);
          break;
        }
    }
    for (const auto &[arch, list] : groups) {
      if (list.empty()) continue;
      Item h;
      h.header = true;
      const model::ArchEntry *ae = entry_of(arch);
      h.label = ae ? std::string(ae->id) + "  \xc2\xb7  " + std::string(ae->label) : "not served by this engine";
      p.items.push_back(h);
      for (const ModelInfo *m : list) {
        Item it;
        it.label = m->name;
        it.value = m->path;
        it.detail = human_bytes(m->bytes);
        if (arch == "qwen38") {
          const size_t nd = drafters_for(m->path).size();
          if (nd) it.detail = "MTP \xc3\x97" + std::to_string(nd) + "  " + it.detail;
        }
        // A model_type other than the arch's own (qwen3_5_moe matched by its architectures name):
        // shown, since the engine may refuse the checkpoint at load.
        if (ae && m->model_type != ae->model_type) {
          it.detail = m->model_type + "?  " + it.detail;
        }
        if (arch.empty()) {
          it.detail = (m->model_type.empty() ? "?" : m->model_type) + "  " + it.detail;
          it.label_st = th::dim;
          it.enabled = false;
        }
        if (m->path == cur) {
          it.mark = "\xe2\x97\x8f";
          it.mark_st = th::acc;
          p.sel = static_cast<int>(p.items.size());
        }
        p.items.push_back(it);
      }
    }
    Item gap;
    gap.header = true;
    p.items.push_back(gap);
    Item other;
    other.label = "Other directory\xe2\x80\xa6";
    other.value = "\x01custom";
    other.label_st = th::acc2;
    p.items.push_back(other);
    p.on_ok = [this](Popup &q) {
      const Item &it = q.items[q.sel];
      q.close = true;
      if (it.value == "\x01custom") {
        open_input("Model directory", "a checkpoint directory with config.json", tilde(val("XEN_MODEL")), true, [this](const std::string &v, std::string &err) {
          const std::string path = expand_home(v);
          const ModelInfo m = probe_model(path);
          if (!m.has_config) {
            err = m.error.empty() ? "no config.json" : m.error;
            return false;
          }
          change("XEN_MODEL", m.path);
          if (m.arch.empty()) say(m.name + ": " + m.error, th::warn, 6);
          return true;
        });
        return;
      }
      change("XEN_MODEL", it.value);
    };
    p.hints = {{"type", "filter"}, {"\xe2\x8f\x8e", "pick"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  // i: every problem of the current values, each with its setting; Enter jumps there.
  void open_issues() {
    const auto &iss = current_issues();
    if (iss.empty()) {
      say("no problems: every setting is valid", th::ok);
      return;
    }
    Popup p;
    p.title = "Problems";
    p.note = "\xe2\x9c\x97 blocks a launch, ! is a warning. Enter goes to the setting.";
    p.width = 96;
    for (const auto &is : iss) {
      const Spec *sp = cat.find(is.key);
      Item it;
      it.mark = is.error ? "\xe2\x9c\x97" : "!";
      it.mark_st = is.error ? th::err : th::warn;
      it.label = (sp ? sp->label : is.key) + ": " + is.msg;
      it.value = is.key;
      p.items.push_back(it);
    }
    p.on_ok = [this](Popup &q) {
      q.close = true;
      const std::string key = q.items[q.sel].value;
      if (!filter.s.empty()) {
        filter.set("");
        rebuild_rows();
      }
      const int si = spec_index(key);
      const Spec &sp = spec(si);
      if (!unused_reason(sp, model.arch, vals).empty() && !show_na) {
        show_na = true;
        rebuild_rows();
      }
      collapsed.erase(sp.group);
      rebuild_rows();
      for (size_t i = 0; i < rows.size(); ++i)
        if (!rows[i].group && rows[i].spec == si) sel = static_cast<int>(i);
      tab_ = kSettings;
    };
    p.hints = {{"\xe2\x8f\x8e", "go to it"}, {"esc", "close"}};
    push(std::move(p));
  }

  // First run (no settings file yet): what the launcher is and the three steps to a running server.
  void show_welcome() {
    std::vector<std::string> lines = {
        "This launcher sets up and runs xenithium-server, the engine's OpenAI-compatible API.",
        "",
        "  1  Pick a model: m (every checkpoint under " + tilde(models_dir()) + ").",
        "  2  Check the settings: each one explains itself in the help panel; the defaults are measured",
        "     and safe. Orange = changed, red = invalid. s saves them to " + tilde(cfg.env_path) + ".",
        "  3  Launch: r runs the checks, then starts the server; the Logs tab (2) shows it loading.",
        "",
        "MoE models (qwen38) record which experts your traffic uses and start faster from it the next time.",
        tab_shown(kChat) ? "Chat (3) talks to the running server; Benchmark (4) runs the benchmark against it. ? lists every key."
                         : "Benchmark (3) runs the benchmark against the running server. ? lists every key.",
    };
    message("Welcome to xenithium launch", lines, th::plain);
    pops.back().width = 100;
  }

  void open_input(const std::string &title, const std::string &prompt, const std::string &init, bool path,
                  std::function<bool(const std::string &, std::string &)> ok) {
    Popup p;
    p.kind = Popup::Input;
    p.title = title;
    p.prompt = prompt;
    p.edit.set(init);
    p.edit.path = path;
    p.edit.root = cfg.root;
    p.width = 72;
    p.on_ok = [ok](Popup &q) {
      std::string err;
      if (ok(q.edit.s, err)) q.close = true;
      else q.error = err;
    };
    p.hints = path ? std::vector<std::pair<std::string, std::string>>{{"tab", "complete"}, {"\xe2\x8f\x8e", "ok"}, {"esc", "cancel"}}
                   : std::vector<std::pair<std::string, std::string>>{{"\xe2\x8f\x8e", "ok"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  void confirm(const std::string &title, const std::vector<std::string> &lines, const std::vector<std::pair<std::string, std::function<void()>>> &buttons,
               Style st = th::plain) {
    Popup p;
    p.title = title;
    for (const auto &l : lines) {
      Item it;
      it.label = l;
      it.label_st = st;
      it.header = true;
      p.items.push_back(it);
    }
    if (!lines.empty()) {
      Item gap;
      gap.header = true;
      p.items.push_back(gap);
    }
    std::vector<std::function<void()>> acts;
    for (const auto &[label, fn] : buttons) {
      Item it;
      it.label = label;
      it.value = std::to_string(acts.size());
      p.items.push_back(it);
      acts.push_back(fn);
    }
    p.on_ok = [acts](Popup &q) {
      q.close = true;
      const int i = std::atoi(q.items[q.sel].value.c_str());
      if (i >= 0 && i < static_cast<int>(acts.size()) && acts[i]) acts[i]();
    };
    p.hints = {{"\xe2\x86\x91\xe2\x86\x93", "choose"}, {"\xe2\x8f\x8e", "ok"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  // ---- files

  // server.env names the model's file (XEN_MODEL_ENV); it also holds XEN_MODELS_DIR, the one global setting that
  // does not live per model (the folder the launcher scans for checkpoints to serve).
  bool write_pointer() {
    std::string text;
    read_text(cfg.env_path, text);
    EnvFile e = EnvFile::parse(text.empty() ? "# xenithium server settings, read by scripts/start-server.sh (docs/server-env.md).\n"
                                            "# The settings live per model in models/<id>.env; this file names the one to start."
                                            : text);
    for (const auto &s : cat.specs)
      if (s.key != "XEN_MODELS_DIR") e.erase(s.key);
    if (val("XEN_MODELS_DIR").empty()) e.erase("XEN_MODELS_DIR");
    else e.set("XEN_MODELS_DIR", val("XEN_MODELS_DIR"));
    const bool in_dir = target.rfind(cfg.model_envs_dir + "/", 0) == 0 && cfg.model_envs_dir == join_path(cfg.root, "models");
    e.set("XEN_MODEL_ENV", in_dir ? model_env_id(target) : target);
    std::string err;
    if (!write_text_atomic(cfg.env_path, e.text(), &err)) {
      say(err, th::err, 6);
      return false;
    }
    return true;
  }

  // The text a user's settings file starts from when it does not exist yet: the shipped preset's lines under a header
  // of its own (the preset's leading comment describes the preset).
  static std::string preset_seed(const std::string &user_path, const std::string &header) {
    std::string text;
    if (!read_text(preset_env_path(user_path), text)) return "";
    size_t at = 0;
    while (at < text.size() && text[at] == '#') {
      const size_t nl = text.find('\n', at);
      at = nl == std::string::npos ? text.size() : nl + 1;
    }
    return header + text.substr(at);
  }

  // True for a file of the shipped presets' folder (<model settings folder>/presets/).
  bool in_presets(const std::string &path) const { return path.rfind(join_path(cfg.model_envs_dir, "presets") + "/", 0) == 0; }

  bool save_to(const std::string &path_in) {
    const std::string path = in_presets(path_in) ? user_env_path(path_in) : path_in;  // the presets are never written
    std::string text;
    read_text(path, text);
    const bool model_file = model_mode && path == target;
    if (text.empty())
      text = preset_seed(path, "# " + model.name + " (" + model.arch + "): your settings, started from the preset models/presets/" + model_env_id(path) +
                                   ".env.\n# Read by scripts/start-server.sh instead of the preset; delete this file to return to it.\n");
    EnvFile f = EnvFile::parse(text);
    if (text.empty())
      f = EnvFile::parse(model_file ? "# " + model.name + " (" + model.arch + "): its settings, read by scripts/start-server.sh (docs/server-env.md,\n"
                                     "# \"Per-model settings files\"); edit by hand or with scripts/launch.sh.\n"
                                    : "# xenithium server settings, read by scripts/start-server.sh (docs/server-env.md).\n"
                                      "# Only settings that differ from the defaults are listed; edit by hand or with scripts/launch.sh.\n");
    const bool split = model_file && !setup_target.empty();
    if (split) {  // The shared settings only; the rest go to the setup's file (save_setup)
      Values shared;
      for (const auto &s : cat.specs)
        if (is_shared_setting(s)) shared[s.key] = val(s.key);
      f.apply(shared, defs, cat, true);
    } else {
      f.apply(vals, defs, cat, model_file);
    }
    if (model_file) {
      f.erase("XEN_MODELS_DIR");  // a global: it stays in server.env (write_pointer), not in the model's file
      if (!f.has("XEN_MODEL")) f.set("XEN_MODEL", val("XEN_MODEL"));  // the file names its checkpoint
    }
    if (split && !save_setup()) return false;
    std::string err;
    const size_t sl = path.find_last_of('/');
    if (sl != std::string::npos) mkdir(path.substr(0, sl).c_str(), 0755);
    if (!write_text_atomic(path, f.text(), &err)) {
      say(err, th::err, 6);
      return false;
    }
    const int n = dirty_count();
    if (!model_file) model_mode = false, setup_target.clear();  // saved as a profile: that file is edited from now on
    if (split)
      for (const auto &sp : cat.specs)
        if (is_shared_setting(sp) && f.has(sp.key)) model_file_vals[sp.key] = val(sp.key);
    target = path;
    saved = vals;
    if (model_file) write_pointer();
    say("saved to " + tilde(path) + (split ? " + " + model_env_id(setup_target) + ".env" : "") + (n ? " (" + std::to_string(n) + " changed)" : "") +
            (model_file ? "; server.env starts it" : ""),
        th::ok);
    return true;
  }

  // The setup's file: every non-shared setting that differs from what the model's file and the defaults give.
  bool save_setup() {
    std::string text;
    read_text(setup_target, text);
    if (text.empty())
      text = preset_seed(setup_target, "# " + model.name + " (" + model.arch + "), setup " + setup_key() + "=" + val(setup_key()) +
                                           ": your settings, started from the preset models/presets/" + model_env_id(setup_target) +
                                           ".env.\n# Read by scripts/start-server.sh instead of the preset; delete this file to return to it.\n");
    EnvFile f = EnvFile::parse(text.empty() ? "# " + model.name + " (" + model.arch + "), setup " + setup_key() + "=" + val(setup_key()) +
                                                  ": this setup's settings over models/" + model_env_id(target) +
                                                  ".env, read by scripts/start-server.sh\n# (docs/server-env.md, \"Per-setup settings files\"); edit by hand or with scripts/launch.sh.\n"
                                            : text);
    Values part, inherited = defs;
    for (const auto &s : cat.specs) {
      if (is_shared_setting(s)) {
        f.erase(s.key);  // shared settings live in the model's file
        continue;
      }
      part[s.key] = val(s.key);
      if (const auto m = model_file_vals.find(s.key); m != model_file_vals.end()) inherited[s.key] = m->second;
    }
    f.apply(part, inherited, cat, false);
    std::string err;
    if (const size_t sl = setup_target.find_last_of('/'); sl != std::string::npos) mkdir(setup_target.substr(0, sl).c_str(), 0755);
    if (!write_text_atomic(setup_target, f.text(), &err)) {
      say(err, th::err, 6);
      return false;
    }
    return true;
  }

  std::vector<std::string> profile_files() const {
    std::vector<std::string> out;
    std::string common;
    for (const auto &f : complete_path(cfg.profiles_dir + "/", "/", common))
      if (f.size() > 4 && f.substr(f.size() - 4) == ".env") out.push_back(join_path(cfg.profiles_dir, f));
    return out;
  }

  std::string profile_summary(const std::string &path) const {
    std::string text;
    if (!read_text(path, text)) return "not created yet";
    const Values v = EnvFile::parse(text).values();
    std::string s;
    if (v.count("XEN_MODEL")) {
      const std::string m = v.at("XEN_MODEL");
      s = m.substr(m.find_last_of('/') + 1);
    }
    int n = 0;
    for (const auto &kv : v)
      if (cat.find(kv.first)) ++n;
    s += (s.empty() ? "" : "  ") + std::to_string(n) + " set";
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
      char b[32];
      std::strftime(b, sizeof b, "%b %d %H:%M", std::localtime(&st.st_mtime));
      s += std::string("  ") + b;
    }
    return s;
  }

  void guarded(const std::function<void()> &fn, const std::string &what) {
    const int n = dirty_count();
    if (!n) {
      fn();
      return;
    }
    confirm("Unsaved changes", {std::to_string(n) + " unsaved change" + (n == 1 ? "" : "s") + " in " + target_name() + "."},
            {{"Discard them and " + what, fn}, {"Cancel", nullptr}}, th::warn);
  }

  void open_profiles() {
    Popup p;
    p.title = "Settings files";
    p.note = "server.env starts a model with its own settings (models/<id>.env); profiles live in " + tilde(cfg.profiles_dir);
    p.width = 84;
    auto add = [&](const std::string &path, const std::string &label) {
      Item it;
      it.label = label;
      it.value = path;
      it.detail = profile_summary(path);
      if (path == target) {
        it.mark = "\xe2\x97\x8f";
        it.mark_st = th::acc;
        p.sel = static_cast<int>(p.items.size());
      }
      p.items.push_back(it);
    };
    add(cfg.env_path, "server.env");
    if (model_mode) p.items.back().detail = "the model's own settings: " + target_name();
    const auto menvs = model_env_files(cfg.model_envs_dir);
    if (!menvs.empty()) {
      Item h;
      h.header = true;
      h.label = "per-model settings";
      p.items.push_back(h);
    }
    for (const auto &f : menvs) {
      Item it;
      it.label = model_env_id(f);
      it.value = "\x01model:" + f;
      const std::string m = model_env_model(f);
      it.detail = (m.empty() ? std::string("no XEN_MODEL") : tilde(m)) + "  " + profile_summary(f).substr(profile_summary(f).find("  ") == std::string::npos ? 0 : profile_summary(f).find("  ") + 2);
      if (model_mode && f == target) {
        it.mark = "\xe2\x97\x8f";
        it.mark_st = th::acc;
        p.sel = static_cast<int>(p.items.size());
      }
      p.items.push_back(it);
    }
    const auto files = profile_files();
    if (!files.empty()) {
      Item h;
      h.header = true;
      h.label = "profiles";
      p.items.push_back(h);
    }
    for (const auto &f : files) {
      std::string n = f.substr(f.find_last_of('/') + 1);
      n.resize(n.size() - 4);
      add(f, n);
    }
    p.on_ok = [this](Popup &q) {
      const std::string path = q.items[q.sel].value;
      q.close = true;
      if (path.rfind("\x01model:", 0) == 0) {  // a per-model file: model mode on that model
        const std::string f = path.substr(7);
        const std::string m = model_env_model(f);
        if (m.empty()) {
          say(tilde(f) + " names no XEN_MODEL", th::warn);
          return;
        }
        guarded([this, f, m] {
          model_mode = true;
          switch_model(m, f);
        }, "open it");
        return;
      }
      guarded([this, path] {
        open_file(path);
        rebuild_rows();
        say("editing " + target_name(), th::acc2);
      }, "open it");
    };
    p.on_key = [this](Popup &q, const Key &k) {
      if (k.is('n')) {
        q.close = true;
        new_profile();
        return true;
      }
      if (k.is('x') || k.type == Key::Delete) {
        std::string path = q.items[q.sel].value;
        if (path.rfind("\x01model:", 0) == 0) path = path.substr(7);
        if (path == cfg.env_path || path.empty()) {
          say("server.env is not deleted from here", th::warn);
          return true;
        }
        q.close = true;
        confirm("Delete profile", {tilde(path)}, {{"Delete", [this, path] {
                                                    std::remove(path.c_str());
                                                    if (target == path) {
                                                      if (!model_mode) target = cfg.env_path;  // model mode: the file is new again
                                                      saved.clear();
                                                    }
                                                    say("deleted " + tilde(path), th::ok);
                                                  }},
                                                  {"Cancel", nullptr}},
                th::warn);
        return true;
      }
      return false;
    };
    p.hints = {{"\xe2\x8f\x8e", "open"}, {"n", "save current as new"}, {"x", "delete"}, {"esc", "close"}};
    push(std::move(p));
  }

  void new_profile() {
    std::string suggest = model.name.empty() ? "my-profile" : model.name;
    open_input("Save as profile", "profile name (letters, digits, - _ .)", suggest, false, [this](const std::string &v, std::string &err) {
      std::string n = v;
      while (!n.empty() && n.back() == ' ') n.pop_back();
      if (n.size() > 4 && n.substr(n.size() - 4) == ".env") n.resize(n.size() - 4);
      if (n.empty()) {
        err = "a name is needed";
        return false;
      }
      for (const char c : n)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')) {
          err = "only letters, digits, - _ .";
          return false;
        }
      const std::string path = join_path(cfg.profiles_dir, n + ".env");
      mkdir(cfg.profiles_dir.c_str(), 0755);
      // A new profile holds the current values only (no lines carried from another file).
      std::remove(path.c_str());
      return save_to(path);
    });
  }

  void save_as() {
    Popup p;
    p.title = "Save as";
    p.width = 72;
    auto add = [&](const std::string &label, const std::string &value, const std::string &detail) {
      Item it;
      it.label = label;
      it.value = value;
      it.detail = detail;
      p.items.push_back(it);
    };
    if (model_mode) add(target_name(), target, "this model's settings; server.env starts it");
    else add("server.env", cfg.env_path, "what scripts/start-server.sh reads");
    for (const auto &f : profile_files()) {
      std::string n = f.substr(f.find_last_of('/') + 1);
      n.resize(n.size() - 4);
      add("profile " + n, f, "overwrite");
    }
    add("New profile\xe2\x80\xa6", "\x01new", "");
    p.items.back().label_st = th::acc2;
    p.on_ok = [this](Popup &q) {
      const std::string v = q.items[q.sel].value;
      q.close = true;
      if (v == "\x01new") new_profile();
      else save_to(v);
    };
    p.hints = {{"\xe2\x8f\x8e", "save"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  // ---- launch and stop

  Snapshot snap() { return mon.get(); }

  pid_t running_pid(const Snapshot &s, int *port = nullptr) const {
    if (s.pid > 0 && s.ours_alive) {
      if (port) *port = s.port;
      return s.pid;
    }
    if (!s.servers.empty()) {
      if (port) *port = s.servers[0].port;
      return s.servers[0].pid;
    }
    return -1;
  }

  void adopt_running() {
    std::ifstream f(join_path(cfg.state_dir, "server.pid"));
    long pid = 0;
    int port = 0;
    if (!(f >> pid >> port) || pid <= 0 || !pid_alive(static_cast<pid_t>(pid))) return;
    std::ifstream c("/proc/" + std::to_string(pid) + "/comm");
    std::string comm;
    std::getline(c, comm);
    if (comm != "xenithium-serve") return;
    mon.watch(static_cast<pid_t>(pid), false, port);
    log.reset(join_path(cfg.state_dir, "server.log"));
    tab_ = kServer;
  }

  void preflight() {
    const Snapshot s = snap();
    int rport = 0;
    const pid_t rp = running_pid(s, &rport);
    if (rp > 0) {
      message("A server is running", {"xenithium-server pid " + std::to_string(rp) + " serves on port " + std::to_string(rport) + ".",
                                      "Stop it first: Logs tab, x."},
              th::warn);
      return;
    }
    Popup p;
    p.title = "Launch";
    p.width = 86;
    bool blocked = false;
    auto check = [&](int state, const std::string &label, const std::string &detail = "") {  // 1 ok, 0 fail, 2 warn, 3 skip
      Item it;
      it.header = true;
      it.mark = state == 1 ? "\xe2\x9c\x93" : state == 0 ? "\xe2\x9c\x97" : state == 2 ? "!" : "\xe2\x80\x93";
      it.mark_st = state == 1 ? th::ok : state == 0 ? th::err : state == 2 ? th::warn : th::dim;
      it.label = label;
      it.detail = detail;
      if (state == 0) blocked = true;
      p.items.push_back(it);
    };
    const std::string bin = join_path(cfg.root, "build/xenithium-server");
    check(access(bin.c_str(), X_OK) == 0 ? 1 : 0, "server binary", access(bin.c_str(), X_OK) == 0 ? "build/xenithium-server" : "missing: scripts/build-host.sh xenithium-server");
    const bool stub = val("XEN_STUB") == "1";
    if (model.arch.empty() && !stub) check(0, "model " + model.name, model.error.empty() ? "not served" : model.error);
    else check(1, "model " + model.name, model.arch.empty() ? "stub engine" : model.arch + "  " + human_bytes(model.bytes));
    int errs = 0, warns = 0;
    std::string first;
    for (const auto &i : current_issues()) {
      if (i.error) {
        ++errs;
        if (first.empty()) first = cat.find(i.key)->label + ": " + i.msg;
      } else ++warns;
    }
    if (errs) {
      check(0, std::to_string(errs) + " invalid setting" + (errs > 1 ? "s" : ""), "");
      for (const auto &l : wrap(first, std::min(c_h_w_ - 16, 74))) {  // the first one, in full
        Item it;
        it.header = true;
        it.mark = " ";
        it.label = l;
        it.label_st = th::err;
        p.items.push_back(it);
      }
    }
    else check(warns ? 2 : 1, "settings valid", warns ? std::to_string(warns) + " warning" + (warns > 1 ? "s" : "") : "");
    if (is_moe_arch(model.arch) && !stub) {  // not blocking: the default placement works, only slower
      std::string t;
      const int st = rank_state(t);
      check(st == 1 ? 1 : 2, st == 1 ? "hot-expert ranking" : "no hot-expert ranking", st == 1 ? t : t + "; auto records one while serving");
      if (rank_job.busy()) check(0, "a ranking run holds the GPUs", "wait for it, or stop the make-hot-rank.sh run");
    }
    // The KV context this launch will have (the memory plan; the dense runtime's is an estimate, so a plan that says
    // the load will not fit warns and does not block: the load itself decides). A plan still in flight is waited for
    // (~0.4 s): the list is built once.
    if (!stub && !model.arch.empty()) {
      update_plan();
      if (plan_job.valid()) plan_job.wait();
      update_plan();
      const MemPlanView *mp = shown_plan();
      if (mp && !plan_stale()) {
        const std::string t = mp->exact || mp->experts_total > 0 ? "" : "~";
        const int64_t full = mp->context > 0 ? mp->kv_tokens / mp->context : 0;
        check(full >= 1 ? 1 : 2, "KV context " + t + with_commas(mp->kv_tokens) + " tokens",
              full >= mp->slots ? "every slot at the " + with_commas(mp->context) + " max context"
                                : std::to_string(full) + " x the " + with_commas(mp->context) + " max context, or " + std::to_string(mp->slots) + " x " +
                                      with_commas(mp->kv_tokens / std::max(1, mp->slots) / 64 * 64));
        if (mp->vram_used_gib > mp->card_gib + 0.05)
          check(2, "VRAM", "the plan takes " + fmt_gib(mp->vram_used_gib) + " of " + fmt_gib(mp->card_gib) + " GiB per card");
      } else if (!plan_err.empty() && !plan_stale()) {
        check(2, "memory plan", "");
        for (const auto &l : wrap(plan_err, std::min(c_h_w_ - 16, 74))) {
          Item it;
          it.header = true;
          it.mark = " ";
          it.label = l;
          it.label_st = th::warn;
          p.items.push_back(it);
        }
      }
    }
    const int port = std::atoi(val("XEN_PORT").c_str());
    const bool busy = cfg.live && port_in_use(port);
    check(busy ? 0 : 1, "port " + std::to_string(port), busy ? "in use" : "free");
    if (stub) check(3, "cards", "not needed (stub engine)");
    else if (!s.docker_known) check(2, "cards", "docker not checked yet");
    else if (!s.docker_ok) check(2, "cards", "docker not reachable");
    else if (!s.containers.empty()) {
      std::string names;
      for (const auto &c : s.containers) names += (names.empty() ? "" : " ") + c;
      check(0, "cards held by containers", names);
    } else check(1, "cards free", "no containers running");
    if (dirty_count()) check(2, "unsaved settings", "launched as shown; " + target_name() + " is not changed");
    Item gap;
    gap.header = true;
    p.items.push_back(gap);
    Item go;
    go.label = blocked ? "Launch (blocked)" : "Launch";
    go.value = "go";
    go.enabled = !blocked;
    go.label_st = blocked ? th::dim : th::ok;
    p.items.push_back(go);
    Item cancel;
    cancel.label = "Cancel";
    cancel.value = "cancel";
    p.items.push_back(cancel);
    p.sel = static_cast<int>(p.items.size()) - (blocked ? 1 : 2);
    p.on_ok = [this](Popup &q) {
      q.close = true;
      if (q.items[q.sel].value == "go") launch();
    };
    p.hints = {{"\xe2\x8f\x8e", "ok"}, {"esc", "cancel"}};
    push(std::move(p));
  }

  void launch() {
    if (!cfg.live) {
      say("launch skipped (test mode)");
      return;
    }
    mkdir(join_path(cfg.root, "build").c_str(), 0755);
    mkdir(cfg.state_dir.c_str(), 0755);
    const std::string env = join_path(cfg.state_dir, "launch.env"), logf = join_path(cfg.state_dir, "server.log");
    std::string err;
    if (!write_text_atomic(env, launch_env_text(launch_values()), &err)) {
      say(err, th::err, 6);
      return;
    }
    // the previous launch's output is kept once, as server.prev.log (still a .log for editors and viewers)
    if (is_file(logf)) std::rename(logf.c_str(), join_path(cfg.state_dir, "server.prev.log").c_str());
    const pid_t pid = spawn_server(cfg.root, env, logf, err);
    if (pid < 0) {
      say(err, th::err, 6);
      return;
    }
    const int port = std::atoi(val("XEN_PORT").c_str());
    std::ofstream(join_path(cfg.state_dir, "server.pid")) << pid << " " << port << "\n";
    mon.watch(pid, true, port);
    log.reset(logf);
    log_scroll = 0;
    tab_ = kServer;
    say("launched pid " + std::to_string(pid) + "; log " + tilde(logf), th::ok);
  }

  void stop_server() {
    const Snapshot s = snap();
    int port = 0;
    const pid_t pid = running_pid(s, &port);
    if (pid <= 0) {
      say("no server running", th::dim);
      return;
    }
    const bool ours = s.pid == pid;
    std::string cmd;
    for (const auto &p : s.servers)
      if (p.pid == pid) cmd = p.cmdline;
    if (ours && s.stopping && std::chrono::duration<double>(Clock::now() - s.stop_sent).count() > 5) {
      confirm("Kill the server", {"pid " + std::to_string(pid) + " has not stopped after SIGTERM."},
              {{"Kill now (SIGKILL)", [this, pid] {
                  mon.mark_killed();
                  kill(pid, SIGKILL);
                  say("SIGKILL sent to " + std::to_string(pid), th::warn);
                }},
               {"Keep waiting", nullptr}},
              th::warn);
      return;
    }
    std::vector<std::string> lines = {"xenithium-server pid " + std::to_string(pid) + " on port " + std::to_string(port) +
                                      (ours ? " (started here)" : " (started outside the launcher)")};
    if (!cmd.empty()) lines.push_back(ellipsize(tilde(cmd), 200));
    lines.push_back("SIGTERM: it stops listening, ends its requests and exits.");
    confirm("Stop the server", lines, {{"Stop (SIGTERM)", [this, pid, ours] {
                                          if (kill(pid, SIGTERM) != 0) {
                                            say("kill " + std::to_string(pid) + ": " + std::strerror(errno), th::err);
                                            return;
                                          }
                                          if (ours) mon.mark_stopping();
                                          say("SIGTERM sent to " + std::to_string(pid), th::acc2);
                                        }},
                                       {"Cancel", nullptr}},
            th::plain);
  }

  // ---- MoE expert files

  ExpertFile rank_file() const { return expert_file(val("XEN_HOT_RANK"), "hot-rank.xrc", cfg.root, cfg.experts_dir, model); }
  std::string rank_auto_path() const { return join_path(experts_dir_of(cfg.experts_dir, model), "hot-rank.xrc"); }
  // The ranking's state in a few words, for the model card and the launch checks: 1 ok, 2 warn (off / unusable). Round
  // 169: the server records it while serving, so a file that does not exist yet is the normal first run.
  int rank_state(std::string &text) const {
    const ExpertFile f = rank_file();
    if (f.path.empty()) {
      text = "ranking off (default placement)";
      return 2;
    }
    if (!f.exists) {
      text = "ranking: recorded from this run on";
      return 1;
    }
    if (!f.error.empty()) {
      text = "ranking unusable: " + f.error;
      return 2;
    }
    text = "ranking saved " + age_text(f.mtime);
    return 1;
  }

  void adopt_rank_job() {
    std::ifstream f(join_path(cfg.state_dir, "rank.pid"));
    long pid = 0;
    std::string mpath, out;
    if (!(f >> pid) || pid <= 0) return;
    std::getline(f, mpath);
    std::getline(f, mpath);
    std::getline(f, out);
    if (!pid_alive(static_cast<pid_t>(pid))) {
      std::remove(join_path(cfg.state_dir, "rank.pid").c_str());
      return;
    }
    std::ifstream c("/proc/" + std::to_string(pid) + "/comm");
    std::string comm;
    std::getline(c, comm);
    if (comm != "xenithium-probe" && comm != "bash") return;
    rank_job.adopt(static_cast<pid_t>(pid), join_path(cfg.state_dir, "rank.log"), mpath.substr(mpath.find_last_of('/') + 1), std::max(0.0, proc_age(static_cast<pid_t>(pid))));
    rank_job_model = mpath, rank_job_out = out;
    rank_job_t0 = static_cast<int64_t>(std::time(nullptr)) - static_cast<int64_t>(std::max(0.0, proc_age(static_cast<pid_t>(pid))));
  }

  // Called every frame: a ranking run that just ended reports, then offers the follow-ups.
  void rank_poll() {
    if (!rank_job.poll()) return;
    std::remove(join_path(cfg.state_dir, "rank.pid").c_str());
    const auto &st = rank_job.state();
    struct stat fs{};
    const bool fresh = stat(rank_job_out.c_str(), &fs) == 0 && static_cast<int64_t>(fs.st_mtime) + 1 >= rank_job_t0;
    if (st.cancelled) say("ranking run stopped", th::warn, 6);
    else if (!fresh || (st.exit_code != 0 && st.exit_code != -1000))
      say("ranking run failed" + (st.exit_code >= 0 ? " (exit " + std::to_string(st.exit_code) + ")" : std::string()) + ": its log is " + tilde(join_path(cfg.state_dir, "rank.log")), th::err, 10);
    else {
      say("ranking created: " + tilde(rank_job_out), th::ok, 8);
      rank_after();
    }
  }

  void rank_after() {
    if (model.path != rank_job_model) return;  // the editor moved on to another model
    if (val("XEN_HOT_RANK") == "auto") return;
    confirm("Ranking created",
            {"The new ranking is in " + tilde(rank_job_out) + ".",
             "Hot-expert ranking is set to \"" + display_value(*cat.find("XEN_HOT_RANK"), val("XEN_HOT_RANK")) + "\": switch it to auto to use the new file."},
            {{"Use the new ranking (XEN_HOT_RANK = auto)", [this] { change("XEN_HOT_RANK", "auto"); }}, {"Keep everything as it is", nullptr}}, th::plain);
  }

  // ---- drawing

  void draw_header(Canvas &c, const Snapshot &s) {
    const int W = c.w();
    const Style bar{th::text, th::bar, 0};
    c.fill(0, 0, W, 1, bar);
    int x = c.put(0, 1, " \xe2\x97\x86 xenithium ", Style{th::royal_lt, th::bar, kBold}) + 1;
    x += c.put(0, x, "launch", Style{th::grey, th::bar, 0}) + 3;
    tab_at.clear();
    const char *plain[] = {"Settings", "Logs", "Chat", "Benchmark"};
    int num = 0;
    for (int i = 0; i < kTabs; ++i) {
      if (!tab_shown(i)) continue;
      const std::string label = " " + std::to_string(++num) + " " + plain[i] + " ";
      const Style st = i == tab_ ? Style{th::white, th::royal, kBold} : Style{th::text, th::bar, 0};
      const int w = c.put(0, x, label, st);
      tab_at.push_back({Rect{0, x, 1, w}, i});
      x += w + 1;
    }
    // Right: server state and cards.
    std::string st_txt;
    Style st_st;
    int port = 0;
    const pid_t rp = running_pid(s, &port);
    if (rp > 0 && s.ready) {
      st_txt = "\xe2\x97\x8f ready :" + std::to_string(port);
      st_st = Style{th::green, th::bar, kBold};
    } else if (rp > 0 && s.stopping) {
      st_txt = spinner() + " stopping";
      st_st = Style{th::yellow, th::bar, kBold};
    } else if (rp > 0) {
      st_txt = spinner() + " loading " + dur(std::chrono::duration<double>(Clock::now() - s.started).count());
      if (s.pid != rp) st_txt = spinner() + " starting :" + std::to_string(port);
      st_st = Style{th::yellow, th::bar, kBold};
    } else if (s.pid > 0 && s.exit_code != INT_MIN && s.exit_code != 0) {
      st_txt = "\xe2\x9c\x97 exited " + (s.exit_code == -1000 ? std::string("") : std::to_string(s.exit_code));
      st_st = Style{th::red, th::bar, kBold};
    } else {
      st_txt = "\xe2\x96\xa0 no server";
      st_st = Style{th::grey, th::bar, 0};
    }
    std::string cards;
    Style cards_st;
    if (!s.docker_known) {
      cards = "cards \xe2\x80\xa6";
      cards_st = Style{th::grey, th::bar, 0};
    } else if (!s.docker_ok) {
      cards = "docker ?";
      cards_st = Style{th::grey, th::bar, 0};
    } else if (!s.containers.empty()) {
      cards = "cards: " + s.containers[0] + (s.containers.size() > 1 ? " +" + std::to_string(s.containers.size() - 1) : "");
      cards_st = Style{th::yellow, th::bar, 0};
    } else if (rp > 0) {
      cards = "cards: xenithium";
      cards_st = Style{th::green, th::bar, 0};
    } else {
      cards = "cards free";
      cards_st = Style{th::green, th::bar, 0};
    }
    if (!cfg.live) {
      cards = "test mode";
      cards_st = Style{th::grey, th::bar, 0};
    }
    std::string pw;
    {
      double tot = 0;
      bool any = false;
      for (const auto &g : s.gpus)
        if (g.watts >= 0) {
          tot += g.watts;
          any = true;
        }
      if (any) pw = "\xe2\x9a\xa1 " + std::to_string(static_cast<int>(std::lround(tot))) + " W";
    }
    const int rw = text_width(st_txt) + 3 + text_width(cards) + 2 + (pw.empty() ? 0 : text_width(pw) + 3);
    if (W - rw > x + 1) {
      int rx = W - rw;
      if (!pw.empty()) {
        rx += c.put(0, rx, pw, Style{th::royal_lt, th::bar, kBold});
        rx += c.put(0, rx, " \xe2\x94\x82 ", Style{th::border, th::bar, 0});
      }
      rx += c.put(0, rx, st_txt, st_st);
      rx += c.put(0, rx, " \xe2\x94\x82 ", Style{th::border, th::bar, 0});
      c.put(0, rx, cards, cards_st);
    }
  }

  void draw_hints(Canvas &c, const std::vector<std::pair<std::string, std::string>> &hints, const std::string &right, Style right_st) {
    const int y = c.h() - 1, W = c.w();
    c.fill(y, 0, W, 1, Style{});
    int rx = W - text_width(right) - 1;
    if (!right.empty()) c.put(y, rx, right, right_st);
    if (Clock::now() < toast_until && !toast.empty()) {
      c.put(y, 1, ellipsize(toast, rx - 3), toast_st);
      return;
    }
    int x = 1;
    for (const auto &[k, label] : hints) {
      const int need = text_width(k) + text_width(label) + 3;
      if (x + need > rx - 1) break;
      x += c.put(y, x, k, th::acc) + 1;
      x += c.put(y, x, label, th::dim) + 2;
    }
  }

  // The model card: what is selected (line 1) and whether it is ready to launch (line 2).
  int card_h() const { return model.arch.empty() || val("XEN_STUB") == "1" ? 4 : 5; }  // a served model: a third line with the memory plan

  // The KV context the launch would have, and the pool by value of the selected setting, for the help panel.
  // *lead: the setting moves the pool (or its table is still being computed): the block goes above the description.
  std::vector<std::pair<std::string, Style>> kv_lines(const Spec &s, int w, bool *lead) const {
    std::vector<std::pair<std::string, Style>> out;
    *lead = false;
    auto add = [&](const std::string &t, Style st) {
      const int ind = t.rfind("  ", 0) == 0 ? 2 : 0;
      for (const auto &l : wrap(t.substr(static_cast<size_t>(ind)), std::max(10, w - ind))) out.push_back({std::string(static_cast<size_t>(ind), ' ') + l, st});
    };
    const MemPlanView *p = shown_plan();
    if (!p) return out;  // (the plan lines say why)
    const bool moe = p->experts_total > 0;
    const std::string t = p->exact || moe ? "" : "~";  // the MoE archs' pool is a formula, the dense runtime's an estimate
    add(std::string("KV context with these settings") + (plan_stale() ? " (updating...)" : ""), th::acc);
    add("  " + t + with_commas(p->kv_tokens) + " tokens (" + fmt_gib(p->kv_gib) + " GiB per card) for " + std::to_string(p->slots) + " slot" + (p->slots == 1 ? "" : "s"), th::bold);
    if (const std::string cap = kv_capacity_text(p->kv_tokens, p->context, p->slots); !cap.empty()) add("  = " + cap, th::plain);
    const std::vector<WhatIfRow> *rows = shown_whatif();
    if (!rows) {
      if (!whatif_values(s, val(s.key), whatif_extra(s)).empty()) add("By value of this setting: computing...", th::dim), *lead = true;
      return out;
    }
    bool differ = false;
    for (const auto &r : *rows) differ |= !r.ok || r.plan.kv_tokens != p->kv_tokens || (moe && r.plan.hot_experts != p->hot_experts);
    if (!differ) {
      add("This setting does not move the KV pool" + std::string(moe ? " or the hot experts" : "") + ": every value gives the same.", th::dim);
      return out;
    }
    *lead = true;
    add("By value of this setting", th::acc);
    size_t nw = 0, tw = 0;
    for (const auto &r : *rows) nw = std::max(nw, static_cast<size_t>(text_width(whatif_name(s, r.value)))), tw = std::max(tw, with_commas(r.plan.kv_tokens).size());
    nw = std::min<size_t>(nw, 26);
    for (const auto &r : *rows) {
      const bool cur = r.value == val(s.key);
      std::string name = ellipsize(whatif_name(s, r.value), static_cast<int>(nw));
      name += std::string(nw - std::min(nw, static_cast<size_t>(text_width(name))), ' ');
      std::string line = std::string(cur ? "\xe2\x80\xba " : "  ") + name + "  ";
      if (!r.ok) {
        out.push_back({ellipsize(line + "will not load: " + r.err, w), th::warn});
        continue;
      }
      const std::string tok = with_commas(r.plan.kv_tokens);
      line += std::string(tw - std::min(tw, tok.size()), ' ') + t + tok + " tokens";
      if (moe) line += ", " + with_commas(r.plan.hot_experts) + " hot experts";
      // against the current value: the pool (the MoE archs' pool is a formula of slots and context: their hot experts)
      std::string d = cur ? "now" : kv_delta_text(r.plan.kv_tokens, p->kv_tokens);
      if (!cur && moe && r.plan.kv_tokens == p->kv_tokens) d = kv_delta_text(r.plan.hot_experts, p->hot_experts) + (r.plan.hot_experts == p->hot_experts ? "" : " experts");
      if (!d.empty()) line += "  " + d;
      out.push_back({ellipsize(line, w), cur ? th::bold : th::plain});
    }
    return out;
  }

  // The memory plan as text lines for the help panel (VRAM per card, KV, host RAM), wrapped to w.
  std::vector<std::pair<std::string, Style>> plan_lines(int w) const {
    std::vector<std::pair<std::string, Style>> out;
    auto add = [&](const std::string &t, Style st) {  // a leading " " indents every wrapped line (wrap trims it)
      const int ind = t.rfind("  ", 0) == 0 ? 2 : 0;
      for (const auto &l : wrap(t.substr(static_cast<size_t>(ind)), std::max(10, w - ind))) out.push_back({std::string(static_cast<size_t>(ind), ' ') + l, st});
    };
    const MemPlanView *p = shown_plan();
    if (!p) {
      if (!plan_err.empty()) add("Memory plan: " + plan_err, th::warn);
      else add("Memory plan: computing...", th::dim);
      return out;
    }
    const bool moe = p->experts_total > 0;
    const std::string t = p->exact ? "" : "~";
    add(std::string("Memory plan") + (plan_stale() ? " (updating...)" : "") + (p->exact ? "" : " (estimate: the load measures the free VRAM)"), th::acc);
    add("VRAM per card   " + t + fmt_gib(p->vram_used_gib) + " of " + fmt_gib(p->card_gib) + " GiB", p->vram_used_gib > p->card_gib + 0.05 ? th::err : th::bold);
    add("  weights       " + fmt_gib(p->weights_gib) + " GiB" + (moe ? "" : " (the model and its drafter)"), th::plain);
    add("  KV cache      " + fmt_gib(p->kv_gib) + " GiB: " + (moe ? "" : t) + with_commas(p->kv_tokens) + " tokens shared by " + std::to_string(p->slots) + " slot" +
            (p->slots == 1 ? "" : "s") + ", one request up to " + with_commas(p->context),
        th::plain);
    if (p->state_gib > 0.005) add("  slot states   " + fmt_gib(p->state_gib) + " GiB", th::plain);
    if (moe)
      add("  hot experts   " + fmt_gib(p->hot_gib) + " GiB: " + t + with_commas(p->hot_experts) + " of " + with_commas(p->experts_total) +
              (p->cards == 1 ? ", each whole on the card" : p->cards == 4 ? ", each split over the four cards" : ", each split over both cards"),
          th::plain);
    add("  scratch       " + fmt_gib(p->scratch_gib) + (moe ? " GiB (prefill buffers, staging, runtime)" : " GiB (prefill buffers, runtime, the reserve)"), th::dim);
    const double room = p->host_room_gib();
    add("Host RAM pinned " + fmt_gib(p->pinned_gib) + " GiB: " + (moe ? "cold experts " + fmt_gib(p->cold_gib) + " + " : "") + "embedding " + fmt_gib(p->embed_gib) + "; " +
            fmt_gib(p->host_available_gib) + " of " + fmt_gib(p->host_total_gib) + " GiB free now",
        p->pinned_gib > room ? th::err : th::bold);
    if (val("XEN_PREFIX_CACHE") == "1" && p->prefix_gib > 0)
      add("  prefix cache  up to " + fmt_gib(std::min(p->prefix_gib, std::max(0.0, room - p->pinned_gib))) + " GiB more on first use" +
              (p->pinned_gib + p->prefix_gib > room ? " (cut from " + fmt_gib(p->prefix_gib) + " to the RAM left)" : ""),
          th::dim);
    return out;
  }

  void draw_model_card(Canvas &c, int y, int W) {
    c.box(y, 0, card_h(), W, th::frame, "Model", th::acc, "m change");
    int x = 2;
    if (model.arch.empty()) {
      x += c.put(y + 1, x, model.name.empty() ? "(none)" : model.name, th::err) + 2;
      c.put(y + 1, x, model.error.empty() ? "not served" : model.error, th::warn, W - x - 2);
      c.put(y + 2, 2, val("XEN_STUB") == "1" ? "the stub engine answers without a model (API tests)" : "press m to pick a checkpoint this engine serves", th::dim, W - 4);
      return;
    }
    x += c.put(y + 1, x, model.name, th::bold) + 2;
    auto chip = [&](int row, int &cx, const std::string &t, Style st) {
      if (cx + text_width(t) + 4 > W - 2) return;
      if (cx > 2) cx += c.put(y + row, cx, "\xc2\xb7 ", th::dim);
      cx += c.put(y + row, cx, t, st) + 1;
    };
    chip(1, x, model.arch, th::acc2);
    chip(1, x, model.arch == "qwen38" ? "Flash-Next MoE" : model.arch == "qwen35" ? "dense 27B" : model.arch == "llama" ? "dense 33B, no drafter"
              : model.arch == "flint" ? "MoE 256 x top-6, no drafter" : model.model_type, th::dim);
    chip(1, x, human_bytes(model.bytes), th::dim);
    chip(1, x, tilde(model.path), th::dim);
    // Line 2: readiness.
    int x2 = 2;
    const std::string bin = join_path(cfg.root, "build/xenithium-server");
    if (access(bin.c_str(), X_OK) != 0) chip(2, x2, "\xe2\x9c\x97 server not built (scripts/build-host.sh xenithium-server)", th::err);
    int64_t ctx = 0;
    ctx = std::atoll(val("XEN_CONTEXT").c_str());
    chip(2, x2, val("XEN_SLOTS") + " slots, " + (ctx >= 1024 ? std::to_string(ctx / 1024) + "K" : val("XEN_CONTEXT")) + " max context", th::plain);
    if (const int64_t tr = trained_positions_of(model); tr > 0 && model.arch != "flint" && !yarn_note(model.arch, ctx, tr).empty()) {
      char f[48];
      std::snprintf(f, sizeof f, "YaRN %.2f", yarn_factor_for((std::max<int64_t>(ctx, 64) + 63) / 64 * 64, tr));
      chip(2, x2, f, th::warn);
    }
    if (is_moe_arch(model.arch)) {
      std::string t;
      const int st = rank_state(t);
      chip(2, x2, st == 1 ? "\xe2\x9c\x93 experts " + t : "! " + t, st == 1 ? th::ok : th::warn);
    }
    if (model.arch == "qwen38") {
      const std::string mtp = val("XEN_MTP");
      chip(2, x2, mtp.empty() ? "MTP drafter off" : "MTP drafter on", mtp.empty() ? th::dim : th::ok);
    } else if (model.arch == "qwen35") {
      chip(2, x2, val("XEN_DRAFTER") != "1" ? "drafter off" : val("XEN_DRAFTER_MODEL").empty() ? "drafter on" : "DFlash drafter", val("XEN_DRAFTER") == "1" ? th::ok : th::dim);
    }
    chip(2, x2, val("XEN_PREFIX_CACHE") == "1" ? "prefix cache on" : "prefix cache off", val("XEN_PREFIX_CACHE") == "1" ? th::ok : th::dim);
    chip(2, x2, "listens on " + val("XEN_HOST") + ":" + val("XEN_PORT"), th::dim);
    if (card_h() < 5) return;
    // Line 3: the memory plan - the KV context the launch will have, what the load takes per card and in pinned host RAM.
    int x3 = 2;
    const MemPlanView *p = shown_plan();
    if (!p) {
      chip(3, x3, plan_err.empty() ? "memory plan: computing..." : "memory plan: " + plan_err, plan_err.empty() ? th::dim : th::warn);
      return;
    }
    const bool moe = p->experts_total > 0;
    const std::string t = p->exact ? "" : "~";
    const double room = p->host_room_gib();
    if (moe) {
      chip(3, x3, "VRAM " + t + fmt_gib(p->vram_used_gib) + "/" + fmt_gib(p->card_gib) + " GiB per card", p->vram_used_gib > p->card_gib + 0.05 ? th::err : th::plain);
      chip(3, x3, "KV " + with_commas(p->kv_tokens) + " tokens (" + fmt_gib(p->kv_gib) + " GiB)", th::plain);
      chip(3, x3, "hot " + t + with_commas(p->hot_experts) + "/" + with_commas(p->experts_total) + " experts (" + fmt_gib(p->hot_gib) + " GiB)", th::plain);
    } else {  // the dense runtime: the pool is what the load leaves, so it leads
      chip(3, x3, "KV pool " + t + with_commas(p->kv_tokens) + " tokens (" + fmt_gib(p->kv_gib) + " GiB per card)", th::bold);
      const int64_t full = p->context > 0 ? p->kv_tokens / p->context : 0;
      chip(3, x3, full >= p->slots ? "every slot at the max context"
                                   : std::to_string(full) + " x max context, or " + std::to_string(p->slots) + " x " + with_commas(p->kv_tokens / std::max(1, p->slots) / 64 * 64),
           th::plain);
      chip(3, x3, "VRAM " + t + fmt_gib(p->vram_used_gib) + "/" + fmt_gib(p->card_gib) + " GiB per card", p->vram_used_gib > p->card_gib + 0.05 ? th::err : th::plain);
    }
    chip(3, x3, (p->pinned_gib > room ? "\xe2\x9c\x97 " : "") + std::string("host RAM ") + fmt_gib(p->pinned_gib) + " GiB pinned of " + fmt_gib(p->host_available_gib) + " free",
         p->pinned_gib > room ? th::err : th::plain);
    if (plan_stale()) chip(3, x3, "updating...", th::dim);
  }

  void draw_list(Canvas &c, Rect r) {
    r_list = r;
    const int n = dirty_count();
    std::string title = "Settings";
    std::string right = filter.s.empty() ? (show_na ? "a: hide n/a" : "") : (help_search ? "help text: " : "find: ") + filter.s + "  esc clears";
    c.box(r.y, r.x, r.h, r.w, (pops.empty() && !filtering) ? th::frame_on : th::frame, title, th::acc, right);
    // Title extras: file and unsaved count.
    int tx = r.x + 2 + text_width(title) + 2;
    const int tend = r.x + r.w - 4;
    const std::string un = n ? "\xe2\x97\x8f " + std::to_string(n) + " unsaved " : "";
    tx += c.put(r.y, tx, ellipsize("\xc2\xb7 " + target_name(), std::max(0, tend - tx - text_width(un) - 1)) + " ", th::dim);
    if (n) c.put(r.y, tx, un, th::mod, std::max(0, tend - tx));
    int y0 = r.y + 1, h = r.h - 2;
    if (filtering) {
      c.put(y0, r.x + 2, "/", th::acc);
      filter.draw(c, y0, r.x + 4, r.w - 6, Style{th::text, th::field, 0});
      ++y0;
      --h;
    }
    list_rows_visible = h;
    if (sel < top) top = sel;
    if (sel >= top + h) top = sel - h + 1;
    top = std::max(0, std::min(top, std::max(0, static_cast<int>(rows.size()) - h)));
    row_at.clear();
    const int inner = r.w - 2;
    const int lw = std::clamp(inner * 45 / 100, 14, 26);
    const auto &iss = current_issues();
    for (int i = 0; i < h && top + i < static_cast<int>(rows.size()); ++i) {
      const int ri = top + i, y = y0 + i;
      const Row &row = rows[ri];
      const bool on = ri == sel && pops.empty();
      row_at.push_back({y, ri});
      if (ri == sel) c.fill(y, r.x + 1, inner, 1, Style{kDefault, on ? th::sel : th::sel_off, 0});
      auto S = [&](Style st) { return ri == sel ? st.with_bg(on ? th::sel : th::sel_off) : st; };
      if (row.group) {
        const bool col = collapsed.count(row.name) && filter.s.empty();
        int x = r.x + 2;
        x += c.put(y, x, col ? "\xe2\x96\xb8 " : "\xe2\x96\xbe ", S(th::acc));
        x += c.put(y, x, row.name, S(th::acc));
        int changed = 0;
        for (const auto &s : cat.specs)
          if (s.group == row.name && unused_reason(s, model.arch, vals).empty() && is_modified(s.key)) ++changed;
        if (changed) {
          const std::string t = std::to_string(changed) + " changed";
          c.put(y, r.x + r.w - 2 - text_width(t), t, S(th::mod));
        }
        continue;
      }
      if (row.hidden > 0) {
        c.put(y, r.x + 4, std::to_string(row.hidden) + " settings this model does not use hidden \xc2\xb7 a shows them", S(th::dim), inner - 4);
        continue;
      }
      const Spec &s = spec(row.spec);
      const bool na = !unused_reason(s, model.arch, vals).empty();
      const bool dirty_k = is_dirty(s.key), modk = is_modified(s.key);
      const Issue *issue = nullptr;
      for (const auto &is : iss)
        if (is.key == s.key) issue = &is;
      if (dirty_k) c.put(y, r.x + 1, "\xe2\x97\x8f", S(th::mod));
      if (on) c.put(y, r.x + 2, "\xe2\x80\xba", S(th::acc));
      c.put(y, r.x + 4, ellipsize(s.label, lw), S(na ? th::dim : on ? th::bold : th::plain));
      const int vx = r.x + 4 + lw + 2, vw = r.x + r.w - 1 - vx - 1;
      if (editing && ri == sel) {
        edit.draw(c, y, vx, vw, Style{th::text, th::field, 0});
        continue;
      }
      Style vs = na ? th::dim : issue ? (issue->error ? th::err : th::warn) : modk ? th::mod : th::plain;
      if (!na && !issue && !modk && s.kind == Kind::Bool) vs = val(s.key) == "1" ? th::ok : th::dim;
      std::string shown = display_value(s, val(s.key));
      if (s.kind == Kind::Bool && !na) shown = (val(s.key) == "1" ? "\xe2\x97\x89 on" : "\xe2\x97\x8b off");
      if (!na && s.key == "XEN_HOT_RANK" && val(s.key) == "auto") {  // auto: say what the model has
        std::string t;
        shown = rank_state(t) == 1 ? "auto \xc2\xb7 " + (rank_file().exists ? "saved " + age_text(rank_file().mtime) : std::string("server creates")) : "auto \xc2\xb7 unusable";
        if (!issue && !modk && rank_state(t) != 1) vs = th::warn;
      }
      if (shown.empty()) shown = "(empty)";
      std::string suffix = na ? "  n/a" : issue ? (issue->error ? "  \xe2\x9c\x97" : "  !") : "";
      const bool cyc = on && (s.kind == Kind::Choice || s.kind == Kind::Bool || !s.choices.empty());
      const int sw = text_width(suffix) + (cyc ? 2 : 0);
      const int used = c.put(y, vx, ellipsize(shown, vw - sw), S(vs));
      if (!suffix.empty()) c.put(y, vx + used, suffix, S(na ? th::dim : vs));
      if (cyc) {
        // Hint that Left / Right cycles.
        c.put(y, r.x + r.w - 3, "\xe2\x87\x86", S(th::dim));
      }
    }
    if (rows.empty()) c.put(y0, r.x + 3, filter.s.empty() ? "no settings" : "no setting matches \"" + filter.s + "\"", th::dim);
    // Scroll marks.
    if (top > 0) c.put(r.y, r.x + r.w - 3, "\xe2\x96\xb2", th::dim);
    if (top + h < static_cast<int>(rows.size())) c.put(r.y + r.h - 1, r.x + r.w - 3, "\xe2\x96\xbc", th::dim);
  }

  // The help panel: what the selected setting does, in plain words, with its problems first; the facts (value,
  // default, options, scope, key and flag) stay at the bottom.
  void draw_detail(Canvas &c, Rect r) {
    r_detail = r;
    const Spec *s = sel_spec();
    const int w = r.w - 4, x = r.x + 2;
    if (!s) {
      c.box(r.y, r.x, r.h, r.w, th::frame, "Help", th::acc);
      int y = r.y + 1;
      std::vector<std::pair<std::string, Style>> lines;
      if (sel < static_cast<int>(rows.size()) && rows[sel].group) {
        lines.push_back({rows[sel].name, th::acc});
        lines.push_back({group_brief(rows[sel].name), th::plain});
        lines.push_back({"", th::plain});
        lines.push_back({"\xe2\x8f\x8e or \xe2\x86\x90 \xe2\x86\x92 folds / unfolds the group.", th::dim});

      } else if (sel < static_cast<int>(rows.size()) && rows[sel].hidden) {
        lines.push_back({"Hidden settings", th::acc});
        lines.push_back({"The server does not read these with " + model.arch + " and the current values (another architecture's settings, or one whose switch is off). A launch passes their defaults. a shows them.", th::dim});
      }
      for (const auto &[t, st] : lines)
        for (const auto &l : wrap(t, w))
          if (y < r.y + r.h - 1) c.put(y++, x, l, st);
      if (sel < static_cast<int>(rows.size()) && rows[sel].group && rows[sel].name == "Capacity & memory" && !model.arch.empty() && val("XEN_STUB") != "1") {
        ++y;
        for (const auto &[l, st] : plan_lines(w))  // already wrapped (with its indents)
          if (y < r.y + r.h - 1) c.put(y++, x, l, st);
      }
      return;
    }
    const auto &iss = current_issues();
    c.box(r.y, r.x, r.h, r.w, th::frame, s->label, th::acc, "J/K scroll");
    const int ybot = r.y + r.h - 1;
    // Facts at the bottom.
    using Parts = std::vector<std::pair<std::string, Style>>;
    std::vector<Parts> facts;
    const std::string def = defs.count(s->key) ? defs.at(s->key) : "";
    auto shown = [&](const std::string &v) {
      const std::string d = display_value(*s, v);
      return d.empty() ? std::string("(empty)") : d;
    };
    facts.push_back({{"now      ", th::dim}, {shown(val(s->key)), is_modified(s->key) ? th::mod : th::bold}, {is_dirty(s->key) ? "   (unsaved)" : "", th::mod}});
    if (is_modified(s->key)) facts.push_back({{"default  ", th::dim}, {shown(def), th::plain}, {"   d resets", th::dim}});
    if (s->kind == Kind::Choice || s->kind == Kind::Bool) {
      Parts parts = {{"options  ", th::dim}};
      std::vector<std::string> opts = s->kind == Kind::Bool ? std::vector<std::string>{"0", "1"} : s->choices;
      for (size_t i = 0; i < opts.size(); ++i) {
        std::string t = s->kind == Kind::Bool ? (opts[i] == "1" ? "on" : "off") : (opts[i].empty() ? "off" : opts[i]);
        parts.push_back({" " + t + " ", opts[i] == val(s->key) ? Style{th::white, th::royal, kBold} : th::plain});
        if (i + 1 < opts.size()) parts.push_back({" ", th::plain});
      }
      facts.push_back(parts);
    } else if (s->kind == Kind::Int || s->kind == Kind::Float) {
      std::string rng;
      double hi = s->hi;
      if (s->key == "XEN_SLOTS" && model.arch != "qwen38" && model.arch != "qwen35") hi = 8;  // qwen38 / qwen35 1..16, llama / flint 1..8
      if (s->key == "XEN_CONTEXT" && max_context_of(model) > 0) hi = static_cast<double>(max_context_of(model));
      if (s->key == "XEN_HOT" && max_hot_of(model) > 0) hi = static_cast<double>(max_hot_of(model));
      if (s->lo > -1e299 && hi < 1e299) rng = fmt1(s->lo) + " .. " + fmt1(hi);
      else if (s->lo > -1e299) rng = "\xe2\x89\xa5 " + (s->kind == Kind::Int ? std::to_string(static_cast<long long>(s->lo)) : fmt1(s->lo));
      if (s->kind == Kind::Int && s->lo > -1e299 && hi < 1e299) rng = with_commas(static_cast<int64_t>(s->lo)) + " .. " + with_commas(static_cast<int64_t>(hi));
      if (s->multiple) rng += ", multiple of " + std::to_string(s->multiple);
      if (s->allow_auto) rng += ", or auto";
      if (!s->unit.empty()) rng += "  (" + s->unit + ")";
      facts.push_back({{"range    ", th::dim}, {rng, th::plain}});
      if (s->key == "XEN_CONTEXT" && trained_positions_of(model) > 0) {  // the rotary the load builds for this max context
        const int64_t tr = trained_positions_of(model), ctx = std::atoll(val("XEN_CONTEXT").c_str());
        std::string rot = "trained " + with_commas(tr) + ": ";
        const std::string note = yarn_note(model.arch, ctx, tr);
        if (note.empty()) rot += "no YaRN";
        else if (model.arch == "flint") rot += "no YaRN needed (windowed rotary)";
        else {
          char f[32];
          std::snprintf(f, sizeof f, "%.2f", yarn_factor_for((std::max<int64_t>(ctx, 64) + 63) / 64 * 64, tr));
          rot += std::string("YaRN factor ") + f + " (automatic)";
        }
        facts.push_back({{"rotary   ", th::dim}, {rot, note.empty() ? th::plain : th::warn}});
      }
    } else if (s->kind == Kind::Path) {
      facts.push_back({{"expects  ", th::dim}, {s->path_file ? "a file" : "a directory", th::plain}, {s->path_auto ? ", auto" : "", th::plain}, {s->allow_empty ? ", or empty" : "", th::plain}});
    }
    std::string scope = scope_name(s->scope);
    Style scope_st = th::plain;
    if (const std::string why = unused_reason(*s, model.arch, vals); !why.empty()) {
      scope += "  (" + why + ")";
      scope_st = th::warn;
    }
    facts.push_back({{"applies  ", th::dim}, {scope, scope_st}});
    facts.push_back({{"setting  ", th::dim}, {s->key, th::code}, {"   ", th::plain}, {s->flag, th::dim}, {s->diag ? "   environment switch" : "", th::dim}});
    // Description area: problems, the description, then this model's live state for the expert files.
    struct DescLine {
      std::string text;
      Style st;
      int label = 0;  // bytes at the start drawn as a label (help_lines)
    };
    std::vector<DescLine> desc;
    for (const auto &is : iss)
      if (is.key == s->key)
        for (const auto &l : wrap((is.error ? "\xe2\x9c\x97 " : "! ") + is.msg, w)) desc.push_back({l, is.error ? th::err : th::warn});
    if (editing && !edit_err.empty())
      for (const auto &l : wrap("\xe2\x9c\x97 " + edit_err, w)) desc.push_back({l, th::err});
    if (editing && !edit.cands.empty()) {
      std::string all;
      for (size_t i = 0; i < edit.cands.size() && i < 12; ++i) all += edit.cands[i] + "  ";
      for (const auto &l : wrap(all, w)) desc.push_back({l, th::dim});
    }
    if (!desc.empty()) desc.push_back({"", th::plain});
    // A setting that moves memory: the KV context the launch would have - the pool, what it holds, and the pool by
    // value of this setting - above the text when the setting moves it, so it shows without scrolling.
    const bool mem = moves_memory(s->key) && !model.arch.empty() && val("XEN_STUB") != "1" && unused_reason(*s, model.arch, vals).empty();
    bool kv_lead = false;
    std::vector<std::pair<std::string, Style>> kv;
    if (mem) kv = kv_lines(*s, w, &kv_lead);
    if (kv_lead) {
      for (auto &l : kv) desc.push_back({std::move(l.first), l.second});
      desc.push_back({"", th::plain});
      kv.clear();
    }
    // The description (what it does, the effect, the cost). The long docs row, with the details it adds,
    // stays in docs/server-env.md: / searches it, and it is the text of a setting the launcher has no schema for.
    std::string help = s->brief;
    if (s->key == "XEN_HOT" && is_moe_arch(model.arch)) {
      const std::string hd = hot_description(model, shown_plan());
      if (!hd.empty()) help += "\n\nThis model: " + hd + ".";
    }
    if (help.empty()) help = s->known ? "(no row in docs/server-env.md)" : s->help;
    for (auto &l : help_lines(help, w)) desc.push_back({std::move(l.text), th::plain, l.label});
    // The settings that move memory end with the memory plan (what the load takes per card and in host RAM).
    if (mem) {
      desc.push_back({"", th::plain});
      for (auto &l : kv) desc.push_back({std::move(l.first), l.second});
      if (!kv.empty()) desc.push_back({"", th::plain});
      for (auto &l : plan_lines(w)) desc.push_back({std::move(l.first), l.second});
    }
    if (s->key == "XEN_HOT_RANK" && is_moe_arch(model.arch)) {
      std::string t;
      const int st = rank_state(t);
      const ExpertFile f = rank_file();
      desc.push_back({"", th::plain});
      for (const auto &l : wrap(std::string("This model: ") + (st == 1 ? "\xe2\x9c\x93 " : "! ") + t + (f.path.empty() ? "" : " (" + tilde(f.path) + ")") +
                                    (st == 1 ? "." : "."),
                                w))
        desc.push_back({l, st == 1 ? th::ok : th::warn});
    }
    const int fy = std::max(r.y + 1, ybot - static_cast<int>(facts.size()));  // the facts' first row
    const int room = std::max(0, fy - 1 - (r.y + 1));  // one row for the rule
    help_scroll = std::clamp(help_scroll, 0, std::max(0, static_cast<int>(desc.size()) - room));
    for (int i = 0; i < room && help_scroll + i < static_cast<int>(desc.size()); ++i) {
      const DescLine &d = desc[static_cast<size_t>(help_scroll + i)];
      int cx = x;
      if (d.label > 0) {
        const std::string_view lab = std::string_view(d.text).substr(0, static_cast<size_t>(d.label));
        cx += c.put(r.y + 1 + i, cx, lab, lab.rfind("Gives", 0) == 0 ? th::gives : lab.rfind("Takes", 0) == 0 ? th::takes : th::label, w);
      }
      c.put_md(r.y + 1 + i, cx, std::string_view(d.text).substr(static_cast<size_t>(d.label)), d.st, th::code, w - (cx - x));
    }
    if (static_cast<int>(desc.size()) > room && room > 0) c.put(fy - 2, x + w - 1, "\xe2\x96\xbc", th::dim);
    if (fy - 1 > r.y) {
      std::string rule;
      for (int i = 0; i < w; ++i) rule += "\xe2\x94\x80";
      c.put(fy - 1, x, rule, Style{th::rule, kDefault, 0});
    }
    int y = fy;
    for (const auto &parts : facts) {
      if (y >= ybot) break;
      int cx = x;
      for (const auto &[t, st] : parts) cx += c.put(y, cx, t, st, x + w - cx);
      ++y;
    }
  }

  void draw_preview(Canvas &c, Rect r) {
    c.box(r.y, r.x, r.h, r.w, th::frame, "Command", th::acc, "scripts/start-server.sh \xc2\xb7 r launch");
    const int w = r.w - 4, x = r.x + 2;
    int y = r.y + 1;
    const int ybot = r.y + r.h - 1;
    if (!preview_err.empty()) {
      for (const auto &l : wrap(preview_err, w)) {
        if (y >= ybot - (preview_cmd.empty() ? 0 : 1)) break;
        c.put(y++, x, l, preview_cmd.empty() ? th::err : th::warn);
      }
    }
    // Tokens: flags in cyan, the binary in blue, wrapped at token boundaries.
    std::vector<std::string> toks;
    std::istringstream in(preview_cmd);
    for (std::string t; in >> t;) toks.push_back(t);
    int cx = x;
    for (size_t i = 0; i < toks.size(); ++i) {
      const std::string &t = toks[i];
      const int tw = text_width(t);
      if (cx > x && cx + tw > x + w) {
        ++y;
        cx = x;
      }
      if (y >= ybot) {
        c.put(ybot - 1, x + w - 1, "\xe2\x80\xa6", th::dim);
        break;
      }
      const Style st = i == 0 ? th::acc : (t.rfind("--", 0) == 0 ? th::acc2 : th::plain);
      cx += c.put(y, cx, t, st, x + w - cx) + 1;
    }
  }

  void draw_configure(Canvas &c) {
    const int W = c.w(), H = c.h();
    draw_model_card(c, 1, W);
    // Preview: as many lines as the command needs, 2..4.
    int need = 1;
    {
      int cx = 0;
      std::istringstream in(preview_cmd);
      for (std::string t; in >> t;) {
        const int tw = text_width(t) + 1;
        if (cx > 0 && cx + tw > W - 4) {
          ++need;
          cx = 0;
        }
        cx += tw;
      }
      if (!preview_err.empty()) need += static_cast<int>(wrap(preview_err, W - 4).size());
    }
    const int ph = std::clamp(need, 1, H >= 34 ? 5 : H >= 26 ? 3 : 2) + 2;
    const int body_y = 1 + card_h(), body_h = H - 1 - ph - body_y;
    if (W >= 100) {
      const int lw = std::clamp(W * 47 / 100, 48, 72);
      draw_list(c, Rect{body_y, 0, body_h, lw});
      draw_detail(c, Rect{body_y, lw, body_h, W - lw});
    } else {
      const int lh = std::max(6, body_h * 58 / 100);
      draw_list(c, Rect{body_y, 0, lh, W});
      draw_detail(c, Rect{body_y + lh, 0, body_h - lh, W});
    }
    draw_preview(c, Rect{H - 1 - ph, 0, ph, W});
    std::vector<std::pair<std::string, std::string>> hints;
    if (editing)
      hints = {{"\xe2\x8f\x8e", "set"}, {"esc", "cancel"}, {"tab", "complete path"}, {"^U", "clear"}};
    else if (filtering)
      hints = {{"type", "filter"}, {"\xe2\x8f\x8e", "keep"}, {"esc", "clear"}};
    else {
      hints = {{"\xe2\x86\x91\xe2\x86\x93", "move"}, {"\xe2\x8f\x8e", "change"}, {"\xe2\x86\x90\xe2\x86\x92", "cycle"}, {"d", "default"}, {"/", "find"},
               {"m", "model"}, {"s", "save"}, {"r", "launch"}};
      if (!current_issues().empty()) hints.push_back({"i", "problems"});
      hints.push_back({"?", "all keys"});
      hints.push_back({"q", "quit"});
    }
    // Right: what needs attention before a launch.
    const int errs = error_count(), warns = static_cast<int>(current_issues().size()) - errs, un = dirty_count();
    std::string right;
    if (un) right += "\xe2\x97\x8f " + std::to_string(un) + " unsaved";
    if (errs) right += std::string(right.empty() ? "" : "   ") + "\xe2\x9c\x97 " + std::to_string(errs) + " invalid";
    else if (warns) right += std::string(right.empty() ? "" : "   ") + "! " + std::to_string(warns) + " warning" + (warns > 1 ? "s" : "");
    else if (!un) right = "\xe2\x9c\x93 ready to launch";
    draw_hints(c, hints, right, errs ? th::err : warns || un ? th::mod : th::ok);
  }

  std::vector<std::pair<std::string, Style>> log_display(const LogTail &lt, bool wr, int width) {
    std::vector<std::pair<std::string, Style>> out;
    const size_t from = lt.lines.size() > 3000 ? lt.lines.size() - 3000 : 0;
    for (size_t i = from; i < lt.lines.size(); ++i) {
      const std::string &l = lt.lines[i];
      const Style st = log_style(l);
      if (wr) {
        for (const auto &w : wrap(l, width)) out.push_back({w, st});
      } else {
        out.push_back({ellipsize(l, width), st});
      }
    }
    return out;
  }
  std::vector<std::pair<std::string, Style>> log_display(int width) { return log_display(log, log_wrap, width); }

  void draw_server(Canvas &c) {
    const int W = c.w(), H = c.h();
    log.poll();
    const Snapshot s = snap();
    int port = 0;
    const pid_t rp = running_pid(s, &port);
    const bool ours = rp > 0 && rp == s.pid;
    // Status card.
    const int sh = 6 + (s.gpus.empty() ? 0 : 1);
    c.box(1, 0, sh, W, th::frame, "Server", th::acc, rp > 0 ? "x stop" : "r launch");
    int y = 2, x = 2;
    const std::string host = val("XEN_HOST");
    if (rp > 0) {
      std::string url_host = (host == "0.0.0.0" || host.empty()) ? (lan.empty() ? "127.0.0.1" : lan) : host;
      if (s.ready) {
        x += c.put(y, x, "\xe2\x97\x8f READY", th::ok) + 2;
        x += c.put(y, x, "http://" + url_host + ":" + std::to_string(port) + "/v1", Style{th::cyan, kDefault, kUnder}) + 2;
        if (url_host != "127.0.0.1") c.put(y, x, "(local http://127.0.0.1:" + std::to_string(port) + "/v1)", th::dim, W - x - 2);
      } else if (s.stopping && ours) {
        x += c.put(y, x, spinner() + " STOPPING", Style{th::yellow, kDefault, kBold}) + 2;
        c.put(y, x, dur(std::chrono::duration<double>(Clock::now() - s.stop_sent).count()) + "  (x again after 5 s: SIGKILL)", th::dim);
      } else {
        x += c.put(y, x, spinner() + " LOADING", Style{th::yellow, kDefault, kBold}) + 2;
        if (ours) x += c.put(y, x, dur(std::chrono::duration<double>(Clock::now() - s.started).count()) + "  ", th::bold);
        std::string last;
        for (auto it = log.lines.rbegin(); it != log.lines.rend(); ++it)
          if (!it->empty()) {
            last = *it;
            break;
          }
        if (ours) c.put(y, x, ellipsize(last, W - x - 2), th::dim);
        else c.put(y, x, "started outside the launcher; waiting for /v1/models on :" + std::to_string(port), th::dim, W - x - 2);
      }
      ++y;
      x = 2;
      std::string info = "pid " + std::to_string(rp);
      if (ours) info += "  \xc2\xb7  up " + dur(std::chrono::duration<double>(Clock::now() - s.started).count());
      if (s.ready && ours && !s.adopted) info += "  \xc2\xb7  ready in " + dur(std::chrono::duration<double>(s.ready_at - s.started).count());
      if (ours) {
        // The model the process runs (the editor may have another file open).
        std::string mname = model.name;
        for (const auto &p : s.servers)
          if (p.pid == rp) {
            const size_t at = p.cmdline.find("--model ");
            if (at != std::string::npos) {
              std::string m = p.cmdline.substr(at + 8);
              m = m.substr(0, m.find(' '));
              while (m.size() > 1 && m.back() == '/') m.pop_back();
              mname = m.substr(m.find_last_of('/') + 1);
            }
          }
        info += "  \xc2\xb7  " + mname;
        info += "  \xc2\xb7  log " + tilde(log.path);
      } else {
        for (const auto &p : s.servers)
          if (p.pid == rp) info += "  \xc2\xb7  " + tilde(p.cmdline);
      }
      c.put(y, x, ellipsize(info, W - 4), th::dim);
      ++y;
      if (s.ready && s.metrics_ok) {
        auto m = [&](const char *k) { return s.metrics.count(k) ? s.metrics.at(k) : 0.0; };
        using Parts = std::vector<std::pair<std::string, Style>>;
        Parts live, totals;
        auto add = [&](Parts &parts, const std::string &label, const std::string &v, Style vs = th::bold) {
          if (!parts.empty()) parts.push_back({"   ", th::plain});
          parts.push_back({label + " ", th::dim});
          parts.push_back({v, vs});
        };
        const long running = static_cast<long>(m("vllm:num_requests_running")), slots = static_cast<long>(m("xenithium:request_slots"));
        const long waiting = static_cast<long>(m("vllm:num_requests_waiting"));
        // Line 1, now: requests in flight and the decode throughput of the whole server (every request's
        // tokens together; per stream = that over the requests running).
        add(live, "running", std::to_string(running) + "/" + std::to_string(slots));
        add(live, "waiting", std::to_string(waiting), waiting > 0 ? th::warn : th::bold);
        add(live, "decode", fmt1(s.gen_tps) + " tok/s (all requests)", s.gen_tps > 0.05 ? th::ok : th::bold);
        // Shown at every stream count (at 1 stream it equals the decode rate above); 0.0 when idle.
        add(live, "per stream", fmt1(running > 0 ? s.gen_tps / static_cast<double>(running) : 0.0) + " tok/s", th::dim);
        add(live, "peak", fmt1(s.peak_tps) + " tok/s", th::dim);
        if (s.metrics.count("xenithium:kv_pool_tokens")) add(live, "KV pool", fmt1(100 * m("vllm:kv_cache_usage_perc")) + "% / " + with_commas(static_cast<int64_t>(m("xenithium:kv_pool_tokens"))) + " tokens");
        // Line 2, since the server started: totals over all requests.
        add(totals, "requests done", with_commas(static_cast<int64_t>(m("xenithium:requests_finished_total"))));
        add(totals, "decoded", with_commas(static_cast<int64_t>(m("vllm:generation_tokens_total"))) + " tokens");
        add(totals, "prompts", with_commas(static_cast<int64_t>(m("vllm:prompt_tokens_total"))) + " tokens");
        if (s.metrics.count("xenithium:kv_pool_tokens")) add(totals, "preempted", std::to_string(static_cast<long>(m("vllm:num_preemptions_total"))));
        // prompt tokens continued from the prefix cache instead of prefilled
        if (m("vllm:prefix_cache_queries_total") > 0)
          add(totals, "prefix cache", fmt1(100 * m("vllm:prefix_cache_hits_total") / m("vllm:prefix_cache_queries_total")) + "% of " +
                                          with_commas(static_cast<int64_t>(m("vllm:prefix_cache_queries_total"))) + " prompt tokens");
        for (const Parts *ps : {&live, &totals}) {
          int cx = x;
          for (const auto &[t, st] : *ps) cx += c.put(y, cx, t, st, W - 2 - cx);
          ++y;
        }
      } else if (s.ready) {
        c.put(y, x, "metrics \xe2\x80\xa6", th::dim);
      }
    } else {
      if (s.pid > 0 && s.exit_code != INT_MIN) {
        const bool clean = s.exit_code == 0;
        x += c.put(y, x, clean ? "\xe2\x96\xa0 STOPPED" : "\xe2\x9c\x97 EXITED", clean ? th::dim : th::err) + 2;
        std::string why = s.exit_why;
        if (why.empty()) why = s.exit_code == -1000 ? "the server is gone" : clean ? "exit status 0" : "exit status " + std::to_string(s.exit_code) + "  (finding out why\xe2\x80\xa6)";
        c.put(y, x, why, clean ? th::dim : th::warn, W - x - 2);
      } else {
        x += c.put(y, x, "\xe2\x96\xa0 NO SERVER", th::dim) + 2;
        c.put(y, x, "press r to launch " + (model.name.empty() ? std::string("the model") : model.name) + " with the current settings", th::plain, W - x - 2);
      }
      ++y;
      if (s.docker_known && !s.containers.empty()) {
        std::string names;
        for (const auto &n : s.containers) names += (names.empty() ? "" : ", ") + n;
        c.put(y, 2, "! containers hold the cards: " + names + " (stop them before a GPU launch)", th::warn, W - 4);
      }
    }
    if (!s.gpus.empty()) {  // the last row of the card: power now, limit and temperatures of each GPU
      int gx = 2;
      const int gy = 1 + sh - 2;
      for (const auto &g : s.gpus) {
        if (gx > 2) gx += c.put(gy, gx, "   ", th::plain);
        gx += c.put(gy, gx, "GPU " + std::to_string(g.info.card) + " ", th::dim, W - 2 - gx);
        gx += c.put(gy, gx, g.watts >= 0 ? std::to_string(static_cast<int>(std::lround(g.watts))) + " W" : "\xe2\x80\xa6", g.watts >= 0 ? th::acc2 : th::dim, W - 2 - gx);
        if (g.info.cap_w > 0) gx += c.put(gy, gx, " / " + std::to_string(static_cast<int>(std::lround(g.info.cap_w))) + " W", th::dim, W - 2 - gx);
        if (g.info.temp_pkg >= 0) {
          const double t = g.info.temp_pkg;
          gx += c.put(gy, gx, "  " + std::to_string(static_cast<int>(std::lround(t))) + "\xc2\xb0" "C", t >= 90 ? th::err : t >= 80 ? th::warn : th::dim, W - 2 - gx);
        }
      }
    }
    // Log.
    const Rect r{1 + sh, 0, H - 2 - sh, W};
    r_log = r;
    const std::string ltitle = "Log";
    std::string right = log_scroll ? "\xe2\x86\x91 " + std::to_string(log_scroll) + " lines \xc2\xb7 End follows" : "following";
    if (!log_wrap) right += " \xc2\xb7 no wrap";
    c.box(r.y, r.x, r.h, r.w, th::frame, ltitle + (log.path.empty() ? "" : "  " + tilde(log.path)), th::acc, right);
    const int lh = r.h - 2;
    log_rows_visible = lh;
    if (log.path.empty() && rp <= 0) {
      c.put(r.y + 1, 2, "No log yet: a launch from here writes " + tilde(join_path(cfg.state_dir, "server.log")) + ".", th::dim);
    } else if (log.path.empty()) {
      c.put(r.y + 1, 2, "This server was started outside the launcher: its log went to the terminal that started it.", th::dim);
    } else {
      const auto disp = log_display(W - 4);
      const int total = static_cast<int>(disp.size());
      log_scroll = std::clamp(log_scroll, 0, std::max(0, total - lh));
      const int start = std::max(0, total - lh - log_scroll);
      for (int i = 0; i < lh && start + i < total; ++i) c.put(r.y + 1 + i, 2, disp[start + i].first, disp[start + i].second, W - 4);
    }
    std::vector<std::pair<std::string, std::string>> hints = {{"↑↓", "scroll"}, {"PgUp/PgDn", "page"}, {"End", "follow"}, {"w", "wrap"}};
    if (rp > 0) hints.push_back({"x", "stop"});
    else hints.push_back({"r", "launch"});
    hints.push_back({"tab", tab_shown(kChat) ? "chat" : "benchmark"});
    hints.push_back({"q", "quit"});
    draw_hints(c, hints, "", th::plain);
  }

  // ---- the chat tab

  // The transcript as display lines (rebuilt every frame: the running answer grows).
  std::vector<std::pair<std::string, Style>> chat_lines(const ChatSession::State &st, int width) {
    std::vector<std::pair<std::string, Style>> out;
    for (const auto &m : st.msgs) {
      if (m.role == "user") {
        out.push_back({"▸ you", th::acc});
        for (const auto &l : wrap(m.text, width)) out.push_back({l, th::plain});
      } else {
        const bool running = !m.done && st.status == ChatSession::Status::running;
        const Style name_st = running ? Style{th::warn.fg, kDefault, kBold} : m.done ? th::ok : th::dim;
        out.push_back({"◂ model" + (running ? std::string(" ") + spinner() : ""), name_st});
        if (!m.reasoning.empty())
          for (const auto &l : wrap(m.reasoning, std::max(1, width - 2))) out.push_back({"  " + l, th::dim});
        if (m.text.empty() && running && m.reasoning.empty()) out.push_back({"  …", th::dim});
        for (const auto &l : wrap(m.text, width)) out.push_back({l, th::plain});
      }
      out.push_back({"", th::plain});
    }
    if (st.status == ChatSession::Status::error) {
      out.push_back({"", th::plain});
      out.push_back({"✗ " + st.error, th::err});
    }
    return out;
  }

  void draw_chat(Canvas &c) {
    const int W = c.w(), H = c.h();
    const ChatSession::State st = chat.get();
    const Snapshot s = snap();
    int port = 0;
    const pid_t rp = running_pid(s, &port);
    const bool ready = rp > 0 && s.ready;
    // The box header's right side: what the request is doing.
    std::string right;
    if (st.status == ChatSession::Status::running) {
      const double el = std::chrono::duration<double>(Clock::now() - st.started).count();
      right = spinner() + " answering " + fmt1(el) + "s   esc stops";
    } else if (st.status == ChatSession::Status::done) {
      right = "done in " + fmt1(st.elapsed) + "s";
      if (st.tokens >= 0 && st.elapsed > 0)
        right += "   " + std::to_string(st.tokens) + " tok   " + fmt1(static_cast<double>(st.tokens) / st.elapsed) + " tok/s";
    } else if (st.status == ChatSession::Status::error) {
      right = "error";
    } else if (st.status == ChatSession::Status::cancelled) {
      right = "stopped";
    } else if (!cfg.live) {
      right = "test mode";
    } else if (!ready) {
      right = rp > 0 ? "loading" : "no server";
    } else {
      right = "ready :" + std::to_string(port);
    }
    // Transcript.
    const int iy = 3;  // input area: the line plus the hint row
    const Rect r{1, 0, std::max(4, H - 2 - iy), W};
    c.box(r.y, r.x, r.h, r.w, th::frame, "Chat", th::acc, right.empty() ? "" : right.c_str());
    const int rows = r.h - 2, inner = W - 6;
    chat_rows_visible = std::max(1, rows);
    const auto lines = chat_lines(st, inner);
    chat_scroll = std::clamp(chat_scroll, 0, std::max(0, static_cast<int>(lines.size()) - rows));
    const int start = std::max(0, static_cast<int>(lines.size()) - rows - chat_scroll);
    for (int i = 0; i < rows && start + i < static_cast<int>(lines.size()); ++i)
      c.put(r.y + 1 + i, 3, lines[start + i].first, lines[start + i].second, W - 6);
    if (st.msgs.empty() && st.status == ChatSession::Status::idle) {
      c.put(r.y + 1, 3, "Chat with the running server: type a message, Enter sends it. The whole", th::dim, W - 6);
      c.put(r.y + 2, 3, "conversation is sent every turn. Esc stops a running answer, F4 starts over.", th::dim, W - 6);
    }
    if (chat_scroll > 0) c.put(r.y + r.h - 1, W - 3, "▲", th::dim);
    // The input line: always focused on this tab.
    const int iyy = H - 2;
    c.fill(iyy, 0, W, 1, Style{th::text, th::field, 0});
    int x = 1;
    x += c.put(iyy, x, "▸", th::acc) + 1;
    chat_input.draw(c, iyy, x, W - x - 2, Style{th::text, th::field, 0});
    std::vector<std::pair<std::string, std::string>> hints = {{"type", "message"}, {"⏎", "send"}, {"esc", "stop"},
                                                              {"↑↓", "scroll"}, {"F4", "new chat"}, {"tab", "next tab"},
                                                              {"q", "quit"}};
    draw_hints(c, hints, "", th::plain);
  }

  // ---- the bench tab

  // r: pick what to measure, then run it.
  void bench_menu() {
    if (bench.busy()) {
      say("a benchmark is already running", th::warn);
      return;
    }
    std::vector<std::pair<std::string, std::function<void()>>> picks;
    for (const BenchMode mode : kBenchModes)
      picks.push_back({bench_mode_title(mode), [this, mode] { mode == BenchMode::Streams ? bench_streams() : bench_run(mode); }});
    picks.push_back({"Cancel", nullptr});
    confirm("Run the benchmark",
            {"xenithium-bench against the running server. The run's notes record the server's launch settings.",
             "Depths past the server's max context are skipped and said so in the report."},
            std::move(picks));
  }

  // The run's notes: what it runs, the engine's git revision, and the settings the server runs with: the launcher's
  // launch.env when the launcher started this server, else the server's command line (started outside it).
  std::vector<std::pair<std::string, std::string>> bench_notes(const Snapshot &s, pid_t rp, BenchMode mode) const {
    std::vector<std::pair<std::string, std::string>> notes = {{"bench_mode", bench_mode_name(mode)}};
    const RunResult g = run_capture({"git", "describe", "--always", "--dirty"}, {}, cfg.root, 2000);
    std::string rev = g.status == 0 ? g.out : "";
    while (!rev.empty() && (rev.back() == '\n' || rev.back() == ' ')) rev.pop_back();
    if (!rev.empty()) notes.push_back({"engine_rev", rev});
    const std::string envf = join_path(cfg.state_dir, "launch.env");
    std::string text;
    if (s.pid == rp && read_text(envf, text)) {
      notes.push_back({"settings_file", tilde(envf)});
      for (auto &kv : launch_notes(text)) notes.push_back(std::move(kv));
    } else {
      for (const auto &p : s.servers)
        if (p.pid == rp) notes.push_back({"server_cmd", tilde(p.cmdline)});
      notes.push_back({"settings_file", "(server started outside the launcher: see server_cmd)"});
    }
    return notes;
  }

  // The Streams mode: ask for the stream counts, then run the concurrency scenario at them. The prompt names the
  // server's slot count when the launcher started it (its launch.env): a count past it queues the excess requests.
  void bench_streams() {
    const Snapshot s = snap();
    int port = 0;
    const pid_t rp = running_pid(s, &port);
    if (rp <= 0 || !s.ready) {
      say("no ready server to bench (start one on the Logs tab)", th::warn);
      return;
    }
    std::string slots, env_text;
    if (s.pid == rp && read_text(join_path(cfg.state_dir, "launch.env"), env_text))
      for (const auto &[k, v] : launch_notes(env_text))
        if (k == "XEN_SLOTS") slots = v;
    const std::string prompt = "Requests in flight at once, e.g. 1, 8, 24, 48 (1-" + std::to_string(kMaxBenchStreams) + ")" +
                               (slots.empty() ? std::string("; past the server's slots they queue") : "; past the server's " + slots + " slots they queue");
    open_input("Concurrency at your stream counts", prompt, bench_streams_last, false, [this](const std::string &typed, std::string &err) {
      const std::string levels = bench_stream_levels(typed, err);
      if (levels.empty()) return false;
      bench_streams_last = levels;
      bench_run(BenchMode::Streams, {"--set", "concurrency.levels=" + levels}, {{"bench_streams", levels}});
      return true;
    });
    pops.back().width = 90;
  }

  // Launch `xenithium-bench run` against the running server, on the served name from /v1/models. `more` = further
  // `run` arguments of the mode, `more_notes` = further run notes.
  void bench_run(BenchMode mode, const std::vector<std::string> &more = {}, const std::vector<std::pair<std::string, std::string>> &more_notes = {}) {
    const Snapshot s = snap();
    int port = 0;
    const pid_t rp = running_pid(s, &port);
    if (rp <= 0 || !s.ready) {
      say("no ready server to bench (start one on the Logs tab)", th::warn);
      return;
    }
    if (bench.busy()) {
      say("a benchmark is already running", th::warn);
      return;
    }
    std::string model;
    try {
      const auto r = http::request("GET", "http://127.0.0.1:" + std::to_string(port) + "/v1/models", "", 5.0);
      if (r.status == 200) {
        xenithium::json::Value v;
        if (xenithium::json::parse(r.body, v))
          if (const auto *arr = v.get_array("data"))
            for (const auto &e : arr->arr)
              if (const std::string *id = e.get_string("id")) {
                model = *id;
                break;
              }
      }
    } catch (...) {
    }
    if (model.empty()) {
      say("could not read the served name from /v1/models", th::err);
      return;
    }
    std::string tag;
    for (char c : model) tag += (std::isalnum(static_cast<unsigned char>(c)) ? c : '-');
    while (tag.size() > 1 && tag.back() == '-') tag.pop_back();
    if (tag.empty()) tag = "run";
    tag += std::string("-") + bench_mode_name(mode);
    if (!is_file(cfg.bench_bin)) {
      say("no " + tilde(cfg.bench_bin) + " (scripts/build-host.sh xenithium-bench, or --bench-bin)", th::err);
      return;
    }
    std::vector<std::string> extra = bench_mode_args(mode);
    extra.insert(extra.end(), more.begin(), more.end());
    extra.insert(extra.end(), {"--out", cfg.bench_runs});
    auto notes = bench_notes(s, rp, mode);
    notes.insert(notes.end(), more_notes.begin(), more_notes.end());
    for (const auto &[k, v] : notes) {
      extra.push_back("--note");
      extra.push_back(k + "=" + v);
    }
    const std::string endpoint = "http://127.0.0.1:" + std::to_string(port) + "/v1";
    const std::string log = join_path(cfg.state_dir, "bench.log");
    if (!bench.start(cfg.bench_bin, endpoint, model, tag, log, cfg.root, extra))
      say(bench.run_state().error, th::err);
    else {
      bench_log.reset(log);
      bench_log_scroll = 0;  // follow the new run from its first line
      say("running xenithium-bench: " + tag, th::dim);
    }
  }

  void draw_bench_detail(Canvas &c, int W, int H) {
    if (bench_detail_run < 0 || bench_detail_run >= static_cast<int>(bench_runs.size())) {
      c.put(3, 3, "(no run selected)", th::dim, W - 6);
      return;
    }
    const BenchRun &r = *bench_runs[bench_detail_run];
    const int bottom = H - 3;
    int y = 2;
    auto line = [&](const std::string &t, Style st) {
      if (y >= bottom) return;
      c.put(y++, 2, t, st, W - 4);
    };
    line("run       " + r.dir, th::plain);
    line("model     " + r.model, th::plain);
    line("method    " + bench_method_text(r), th::plain);
    if (!r.endpoint.empty()) line("endpoint  " + r.endpoint, th::dim);
    if (!r.host.empty()) line("host      " + r.host, th::dim);
    line("decode    " + fmt1(r.decode_tps) + " t/s   " +
             (r.decode_mix.empty() || r.decode_mix == "weighted" ? std::string("(weighted mean of the per-category medians)") : "(the " + r.decode_mix + " of the per-category rates)"),
         th::ok);
    if (r.decode_mixes.size() > 1) {  // the other weightings the run reports (the page switches between them)
      std::string t = "          ";
      for (size_t k = 1; k < r.decode_mixes.size(); ++k) t += r.decode_mixes[k].first + " " + fmt1(r.decode_mixes[k].second) + " t/s  ";
      line(t, th::dim);
    }
    if (!r.per_cat.empty()) {
      std::string t = "          ";
      for (const auto &[cat, tps] : r.per_cat) t += cat + " " + fmt1(tps) + "  ";
      line(t, th::dim);
    }
    if (!r.prefill.empty()) {
      std::string t = "prefill   ";
      for (const auto &[depth, pp] : r.prefill) t += std::to_string(depth) + ":" + fmt1(pp) + "  ";
      line(t, th::dim);
    }
    if (!r.deep_decode.empty()) {
      std::string t = "decode at depth ";
      for (const auto &[depth, tps] : r.deep_decode) t += std::to_string(depth) + ":" + fmt1(tps) + "  ";
      line(t, th::dim);
    }
    if (!r.concurrency.empty()) {
      std::string t = "concurrency ";
      for (const auto &[level, agg] : r.concurrency) t += std::to_string(level) + ":" + fmt1(agg) + "  ";
      line(t, th::dim);
    }
    if (r.tokens_per_joule > 0) line("efficiency " + fmt1(r.tokens_per_joule) + " tokens per joule of GPU board power, at the best stream count", th::dim);
    if (r.quality_pct >= 0) line("quality   " + fmt1(r.quality_pct) + " % of the checkable questions answered correctly", r.quality_pct >= kQualityWarnBelowPct ? th::dim : th::warn);
    for (const BenchRun::Miss &m : r.quality_misses)  // each wrong answer on a line of its own, in the failure colour
      line("          \xE2\x9C\x97 " + m.item + ": expected \"" + m.expected + "\", got " + m.got, th::err);
    line("page      " + tilde(join_path(r.path, "report.html")) + "   (charts; open it in a browser)", th::dim);
    line("tables    xenithium-bench report " + tilde(r.path) + "   (every scenario; compare A B sets two runs side by side)", th::dim);
    if (!r.notes.empty()) {
      // The launch settings make ~100 notes: pack them as KEY=value pairs into the width.
      line("notes (" + std::to_string(r.notes.size()) + ")", th::acc);
      std::string t = " ";
      for (const auto &[k, v] : r.notes) {
        const std::string kv = " " + k + "=" + v;
        if (t.size() > 1 && static_cast<int>(t.size() + kv.size()) > W - 4) {
          line(t, th::dim);
          t = " ";
        }
        t += kv;
      }
      if (t.size() > 1) line(t, th::dim);
    }
  }

  void draw_bench(Canvas &c) {
    const int W = c.w(), H = c.h();
    if (bench.poll() && cfg.live) rebuild_bench();
    bench_log.poll();
    std::string right;
    if (bench.busy()) {
      const double el = std::chrono::duration<double>(Clock::now() - bench.run_state().started).count();
      right = spinner() + " running " + bench.run_state().name + " " + fmt1(el) + "s   x stops";
    } else if (!cfg.bench_runs.empty()) {
      right = std::to_string(bench.total_runs()) + " runs   " + tilde(cfg.bench_runs);
    }
    if (bench_detail) {
      const int bh = std::max(4, H - 2);
      c.box(1, 0, bh, W, th::frame, "Benchmark", th::acc, right.empty() ? "" : right.c_str());
      draw_bench_detail(c, W, H);
      return;
    }
    // The tab splits in two: the runs list on top, the run's log below (both inside the frame).
    const int area = std::max(8, H - 3);
    const int list_box = std::clamp((area + 1) / 2, 6, area - 4);
    const int log_box = area - list_box;
    c.box(1, 0, list_box, W, th::frame, "Benchmark", th::acc, right.empty() ? "" : right.c_str());
    const int rows = std::max(1, list_box - 2);
    bench_rows_visible = rows;
    if (bench_lines.empty()) {
      c.put(3, 3, "No benchmark runs found in " + tilde(cfg.bench_runs), th::dim, W - 6);
      c.put(4, 3, "Run one with r (a ready server is needed), or point the launcher at the runs dir.", th::dim, W - 6);
    } else {
      bench_scroll = std::clamp(bench_scroll, 0, std::max(0, static_cast<int>(bench_lines.size()) - rows));
      const int start = bench_scroll;
      for (int i = 0; i < rows && start + i < static_cast<int>(bench_lines.size()); ++i) {
        const BenchLine &l = bench_lines[start + i];
        const bool on = !l.header && (start + i) == bench_sel;
        const Style st = l.header ? th::acc : (on ? Style{th::white, th::royal, kBold} : th::plain);
        c.put(2 + i, 2, l.text, st, W - 4);
      }
    }
    const int ly = 1 + list_box;
    r_bench_log = {ly, 0, log_box, W};
    std::string lright = bench_log_scroll ? "\xe2\x86\x91 " + std::to_string(bench_log_scroll) + " lines \xc2\xb7 End follows" : "following";
    if (!bench_log_wrap) lright += " \xc2\xb7 no wrap";
    c.box(ly, 0, log_box, W, th::frame, "Log  " + tilde(bench_log.path), th::acc, lright);
    const int lh = std::max(1, log_box - 2);
    if (bench_log.lines.empty()) {
      c.put(ly + 1, 2, "No benchmark log yet: r starts a run, its output lands in " + tilde(bench_log.path) + ".", th::dim, W - 4);
    } else {
      const auto disp = log_display(bench_log, bench_log_wrap, W - 4);
      const int total = static_cast<int>(disp.size());
      bench_log_scroll = std::clamp(bench_log_scroll, 0, std::max(0, total - lh));
      const int start = std::max(0, total - lh - bench_log_scroll);
      for (int i = 0; i < lh && start + i < total; ++i) c.put(ly + 1 + i, 2, disp[start + i].first, disp[start + i].second, W - 4);
    }
    std::vector<std::pair<std::string, std::string>> hints = {{"↑↓", "list"}, {"⏎", "report"}, {"r", "run"},
                                                              {"x", "stop"},
                                                              {"j/k", "log"}, {"G", "follow"}, {"w", "wrap"},
                                                              {"tab", "next tab"}};
    draw_hints(c, hints, "", th::plain);
  }

  bool handle_bench(const Key &k) {
    if (bench_detail) {
      if (k.type == Key::Esc) {
        bench_detail = false;
        bench_detail_run = -1;
        return true;
      }
      return handle_global(k);
    }
    switch (k.type) {
      case Key::Up:
      case Key::Down: {
        const int d = (k.type == Key::Up ? -1 : 1);
        int n = bench_sel + d;
        while (n >= 0 && n < static_cast<int>(bench_lines.size()) && bench_lines[n].header) n += d;
        if (n >= 0 && n < static_cast<int>(bench_lines.size())) {
          bench_sel = n;
          if (bench_sel < bench_scroll) bench_scroll = bench_sel;
          else if (bench_sel >= bench_scroll + bench_rows_visible) bench_scroll = bench_sel - bench_rows_visible + 1;
        }
        return true;
      }
      case Key::Enter: {
        if (bench_sel >= 0 && bench_sel < static_cast<int>(bench_lines.size())) {
          const BenchLine &l = bench_lines[bench_sel];
          if (!l.header && l.run >= 0) {
            bench_detail = true;
            bench_detail_run = l.run;
            bench_scroll = 0;
          }
        }
        return true;
      }
      // Esc deliberately does nothing here: an accidental press must not kill a running
      // benchmark (x stops it). In the report view it just goes back to the list.
      case Key::PgUp: bench_scroll = std::max(0, bench_scroll - std::max(1, bench_rows_visible - 1)); return true;
      case Key::PgDn: bench_scroll = std::min(std::max(0, static_cast<int>(bench_lines.size()) - bench_rows_visible),
                                              bench_scroll + std::max(1, bench_rows_visible - 1));
        return true;
      case Key::Home: bench_scroll = 0; return true;
      case Key::End: bench_scroll = std::max(0, static_cast<int>(bench_lines.size()) - bench_rows_visible); return true;
      case Key::Mouse:
        if (k.button != 64 && k.button != 65) break;  // not the wheel: the click goes on to the global targets (the header's tabs)
        if (r_bench_log.in(k.my, k.mx)) {  // over the log panel: scroll the log
          if (k.button == 64) bench_log_scroll += 3;
          else bench_log_scroll = std::max(0, bench_log_scroll - 3);
        } else if (k.button == 64) ++bench_scroll;
        else bench_scroll = std::max(0, bench_scroll - 1);
        return true;
      default: break;
    }
    if (k.is('r') || k.type == Key::F5) {
      bench_menu();
      return true;
    }
    if (k.is('x')) {
      if (bench.busy()) {
        bench.cancel();
        say("stopping the benchmark", th::dim);
      }
      return true;
    }
    // The log panel: j/k scroll it, g jumps to the top, G follows the end, w toggles wrapping.
    if (k.is('j')) bench_log_scroll += 1;
    else if (k.is('k')) bench_log_scroll = std::max(0, bench_log_scroll - 1);
    else if (k.is('g')) bench_log_scroll = 1 << 28;
    else if (k.is('G')) bench_log_scroll = 0;
    else if (k.is('w')) bench_log_wrap = !bench_log_wrap;
    if (handle_global(k)) return true;
    return true;
  }

  void draw_popup(Canvas &c, Popup &p) {
    const int W = c.w(), H = c.h();
    const Style bg{th::text, th::popup, 0};
    auto on_bg = [&](Style s) { return s.with_bg(th::popup); };
    if (p.kind == Popup::Input) {
      const int w = std::min(W - 4, std::max(p.width, 50));
      const int ncand = std::min<int>(static_cast<int>(p.edit.cands.size()), 6);
      const int h = 6 + (p.error.empty() ? 0 : 1) + (ncand ? 1 : 0) + ncand;
      const int y0 = std::max(1, (H - h) / 2), x0 = (W - w) / 2;
      c.fill(y0, x0, w, h, bg);
      std::string hint;
      for (const auto &[k, l] : p.hints) hint += k + " " + l + "  ";
      c.box(y0, x0, h, w, on_bg(th::frame_on), p.title, on_bg(th::acc), hint);
      int y = y0 + 2;
      c.put(y++, x0 + 3, p.prompt, on_bg(th::dim), w - 6);
      p.edit.draw(c, y++, x0 + 3, w - 6, Style{th::text, th::field, 0});
      if (!p.error.empty()) c.put(y++, x0 + 3, "\xe2\x9c\x97 " + p.error, on_bg(th::err), w - 6);
      if (ncand) {
        ++y;
        for (int i = 0; i < ncand; ++i) c.put(y++, x0 + 5, p.edit.cands[i], on_bg(th::dim), w - 8);
      }
      return;
    }
    const auto vis = p.visible();
    int want = 36;
    for (const int i : vis) want = std::max(want, text_width(p.items[i].label) + text_width(p.items[i].detail) + 12);
    want = std::max(want, text_width(p.title) + 10);
    if (!p.note.empty()) want = std::max(want, std::min(text_width(p.note) + 6, 90));
    const int w = std::min(W - 4, p.width ? std::max(p.width, std::min(want, W - 4)) : want);
    const int extra = (p.filterable ? 2 : 0) + (p.note.empty() ? 0 : 2);
    const int rows = std::min(static_cast<int>(vis.size()), std::max(3, H - 6 - extra));
    const int h = rows + 2 + extra + 1;
    const int y0 = std::max(1, (H - h) / 2), x0 = (W - w) / 2;
    c.fill(y0, x0, w, h, bg);
    std::string hint;
    for (const auto &[k, l] : p.hints) hint += k + " " + l + "  ";
    if (!hint.empty()) hint.resize(hint.size() - 2);
    c.box(y0, x0, h, w, on_bg(th::frame_on), p.title, on_bg(th::acc), hint);
    int y = y0 + 1;
    if (!p.note.empty()) {
      c.put(y, x0 + 3, ellipsize(p.note, w - 6), on_bg(th::dim));
      y += 2;
    }
    if (p.filterable) {
      c.put(y, x0 + 3, "\xe2\x8c\x95 ", on_bg(th::acc));
      c.fill(y, x0 + 5, w - 8, 1, Style{th::text, th::field, 0});
      c.put(y, x0 + 6, p.filter.empty() ? "type to filter" : p.filter, p.filter.empty() ? Style{th::grey, th::field, 0} : Style{th::text, th::field, 0}, w - 10);
      y += 2;
    }
    int pos = static_cast<int>(std::find(vis.begin(), vis.end(), p.sel) - vis.begin());
    if (pos >= static_cast<int>(vis.size())) pos = 0;
    if (pos < p.scroll) p.scroll = pos;
    if (pos >= p.scroll + rows) p.scroll = pos - rows + 1;
    p.scroll = std::clamp(p.scroll, 0, std::max(0, static_cast<int>(vis.size()) - rows));
    for (int i = 0; i < rows && p.scroll + i < static_cast<int>(vis.size()); ++i) {
      const Item &it = p.items[vis[p.scroll + i]];
      const bool on = p.selectable && vis[p.scroll + i] == p.sel && !it.header;
      const uint32_t rbg = on ? th::sel : th::popup;
      if (on) c.fill(y + i, x0 + 1, w - 2, 1, Style{kDefault, th::sel, 0});
      int x = x0 + 2;
      if (it.header && it.mark.empty()) {
        c.put(y + i, x + 1, it.label, (it.label_st == th::plain ? th::acc2 : it.label_st).with_bg(rbg), w - 6);
        continue;
      }
      if (on) c.put(y + i, x, "\xe2\x80\xba", th::acc.with_bg(rbg));
      x += 2;
      if (!it.mark.empty()) c.put(y + i, x, it.mark, it.mark_st.with_bg(rbg));
      x += 2;
      // The label keeps its width; a long detail is cut instead (down to 12 columns).
      const int avail = w - (x - x0) - 3;
      const int dfull = text_width(it.detail);
      const int lw = std::min(text_width(it.label), std::max(0, avail - (dfull ? std::min(dfull, 12) + 2 : 0)));
      const std::string detail = ellipsize(it.detail, std::max(0, avail - lw - 2));
      const int dw = text_width(detail);
      Style ls = it.enabled ? it.label_st : th::dim;
      if (on) ls = ls.with(kBold);
      c.put(y + i, x, ellipsize(it.label, lw), ls.with_bg(rbg));
      if (dw) c.put(y + i, x0 + w - 3 - dw, detail, th::dim.with_bg(rbg));
    }
    if (vis.empty()) c.put(y, x0 + 4, "nothing matches", on_bg(th::dim));
    if (p.scroll > 0) c.put(y0, x0 + w - 3, "\xe2\x96\xb2", on_bg(th::dim));
    if (p.scroll + rows < static_cast<int>(vis.size())) c.put(y0 + h - 1, x0 + 2, "\xe2\x96\xbc more", on_bg(th::dim));
  }

  void draw_help(Canvas &) {}

  // Once per ended server: its end in words goes into its log (the Server tab's log pane shows it; the file keeps it for
  // later) and into the status line. A server cannot log a SIGKILL (the OOM killer's) or a crash past its handler itself.
  pid_t exit_noted_ = -1;
  void note_exit(const Snapshot &s) {
    if (s.pid <= 0 || s.exit_code == INT_MIN || s.exit_code == 0 || s.exit_why.empty() || exit_noted_ == s.pid) return;
    exit_noted_ = s.pid;
    const std::string line = "launcher: the server (pid " + std::to_string(s.pid) + ") " + s.exit_why;
    if (!log.path.empty()) std::ofstream(log.path, std::ios::app) << "\n" << line << "\n";
    say("server " + s.exit_why, th::err, 15);
  }

  void draw(Canvas &c) {
    const Snapshot s = snap();
    c.clear();
    if (c.w() < 64 || c.h() < 18) {
      int y = 0;
      for (const auto &l : wrap("The terminal is " + std::to_string(c.w()) + "x" + std::to_string(c.h()) + "; xenithium launch needs 64x18.", c.w()))
        c.put(y++, 0, l, th::warn);
      c.put(y, 0, "q quits", th::dim);
      return;
    }
    rank_poll();
    note_exit(s);
    if (tab_ == kSettings) update_preview();
    if (tab_ == kSettings) update_plan(), update_whatif();
    draw_header(c, s);
    if (tab_ == kSettings) draw_configure(c);
    else if (tab_ == kServer) draw_server(c);
    else if (tab_ == kBench) draw_bench(c);
    else draw_chat(c);
    for (auto &p : pops) draw_popup(c, p);
  }

  // ---- keys

  void show_help() {
    message("Keys", {
                        "Settings (1)",
                        "  \xe2\x86\x91\xe2\x86\x93 / j k      move           PgUp PgDn   page",
                        "  \xe2\x8f\x8e              change: a list of choices, or type a value; fold a group",
                        "  \xe2\x86\x90 \xe2\x86\x92 / h l     previous / next choice; fold a group",
                        "  space          toggle on / off; next choice",
                        "  e              type a custom value (tab completes paths)",
                        "  d / D          this setting / every setting back to its default",
                        "  u              undo                    i   list the problems",
                        "  /              find a setting (name, key, flag or help text)",
                        "  a              show the settings this model does not use",
                        "  m              pick the model          p   server.env and profiles",
                        "  s              save to the open file   S   save as \xe2\x80\xa6",
                        "  r / F5         launch (checks first)   x   stop the server",
                        "  J K            scroll the help panel",
                        "Logs (2)",
                        "  \xe2\x86\x91\xe2\x86\x93 PgUp PgDn    scroll the log      End   follow    w   wrap    x  stop",
                        tab_shown(kChat) ? "Chat (3)" : "Chat (hidden for now)",
                        "  type           the message          \xe2\x8f\x8e    send",
                        "  esc            stop a running answer (or clear the line)",
                        "  \xe2\x86\x91\xe2\x86\x93 PgUp PgDn   scroll              End   follow    F4   new conversation",
                        tab_shown(kChat) ? "Benchmark (4)" : "Benchmark (3)",
                        "  \xe2\x86\x91\xe2\x86\x93 PgUp PgDn   move a run          \xe2\x8f\x8e    its report      esc   back from it",
                        "  r              run xenithium-bench   x   stop it",
                        "  m              mode: easy (conventional code and prose, the default) or hard (the engine's own code)",
                        "  j k g G        the log: scroll \xc2\xb7 top \xc2\xb7 follow         w   wrap",
                        "Everywhere",
                        tab_shown(kChat) ? "  tab / 1-4      switch tabs     ?   this help     q   quit (the server keeps running)"
                                         : "  tab / 1-3      switch tabs     ?   this help     q   quit (the server keeps running)",
                        "",
                        "Values that differ from the default are orange; \xe2\x97\x8f marks unsaved ones; red is invalid.",
                        "A launch runs scripts/start-server.sh with exactly the command shown under Command.",
                    });
    pops.back().title = "xenithium launch \xc2\xb7 keys";
    pops.back().width = 92;
  }

  bool handle_popup(const Key &k) {
    Popup &p = pops.back();
    auto finish = [&]() {
      // Callbacks may push popups: close the one that was on top.
      for (size_t i = 0; i < pops.size(); ++i)
        if (pops[i].close) {
          pops.erase(pops.begin() + static_cast<long>(i));
          break;
        }
    };
    if (k.type == Key::Esc || (k.type == Key::Ctrl && k.ch == 'c')) {
      auto cancel = p.on_cancel;
      pops.pop_back();
      if (cancel) cancel();
      return true;
    }
    if (p.kind == Popup::Input) {
      if (k.type == Key::Enter) {
        if (p.on_ok) {
          auto fn = p.on_ok;
          fn(p);
        }
        finish();
        return true;
      }
      p.edit.handle(k);
      p.error.clear();
      return true;
    }
    if (p.on_key) {
      auto fn = p.on_key;
      if (fn(p, k)) {
        finish();
        return true;
      }
    }
    const int page = std::max(1, c_h_ - 12);
    switch (k.type) {
      case Key::Up: p.move(-1); return true;
      case Key::Down: p.move(1); return true;
      case Key::PgUp:
        for (int i = 0; i < page; ++i) p.move(-1);
        return true;
      case Key::PgDn:
        for (int i = 0; i < page; ++i) p.move(1);
        return true;
      case Key::Home:
        p.sel = 0;
        p.fix_sel();
        return true;
      case Key::End: {
        const auto v = p.visible();
        if (!v.empty()) p.sel = v.back();
        p.fix_sel();
        return true;
      }
      case Key::Enter: {
        if (!p.selectable) {
          pops.pop_back();
          return true;
        }
        const auto v = p.visible();
        if (v.empty() || std::find(v.begin(), v.end(), p.sel) == v.end() || p.items[p.sel].header || !p.items[p.sel].enabled) return true;
        if (p.on_ok) {
          auto fn = p.on_ok;
          fn(p);
        }
        finish();
        return true;
      }
      case Key::Backspace:
        if (p.filterable && !p.filter.empty()) {
          LineEdit e;
          e.set(p.filter);
          e.handle(k);
          p.filter = e.s;
          p.fix_sel();
        }
        return true;
      case Key::Char:
        if (p.filterable) {
          utf8_append(p.filter, k.ch);
          p.fix_sel();
          return true;
        }
        if (k.ch == 'j') p.move(1);
        else if (k.ch == 'k') p.move(-1);
        else if (k.ch == 'q') pops.pop_back();
        return true;
      case Key::Mouse:
        if (k.button == 64) p.move(-1);
        else if (k.button == 65) p.move(1);
        return true;
      default: return true;
    }
  }

  void request_quit() {
    const int n = dirty_count();
    if (!n) {
      quit = true;
      return;
    }
    confirm("Quit", {std::to_string(n) + " unsaved change" + (n == 1 ? "" : "s") + " in " + target_name() + "."},
            {{"Save and quit", [this] {
                if (save_to(target)) quit = true;
              }},
             {"Quit without saving", [this] { quit = true; }},
             {"Cancel", nullptr}},
            th::warn);
  }

  // The keys every tab shares: switching tabs, help, quit. A letter that acts belongs to the tab whose hint row shows it
  // (m on Benchmark picks the corpus, on Settings the model).
  bool handle_global(const Key &k) {
    if (k.type == Key::Tab || k.type == Key::BackTab) {
      tab_ = next_shown_tab(tab_, k.type == Key::BackTab);
      return true;
    }
    for (int t = 0; t < kTabs; ++t)
      if (k.is(static_cast<char32_t>('1' + t))) {
        if (const int v = shown_tab_at(t); v >= 0) {
          tab_ = v;
          return true;
        }
        return false;  // the digit stands past the last shown tab: it does nothing
      }
    if (k.is('?') || k.type == Key::F1) {
      show_help();
      return true;
    }
    if (k.is('q')) {
      request_quit();
      return true;
    }
    if (k.type == Key::Ctrl && k.ch == 'c') {
      if (++ctrl_c >= 2) quit = true;
      else request_quit();
      return true;
    }
    if (k.type == Key::Mouse && k.press && k.button == 0) {
      for (const auto &[r, t] : tab_at)
        if (r.in(k.my, k.mx)) {
          tab_ = t;
          return true;
        }
    }
    return false;
  }

  // r / F5 launch (after the checks) and x stop: the Settings and Logs tabs' server keys.
  bool handle_server_keys(const Key &k) {
    if (k.is('r') || k.type == Key::F5) {
      preflight();
      return true;
    }
    if (k.is('x')) {
      stop_server();
      return true;
    }
    return false;
  }

  bool handle_configure(const Key &k) {
    if (editing) {
      if (k.type == Key::Enter) commit_edit();
      else if (k.type == Key::Esc) {
        editing = false;
        edit_err.clear();
      } else if (k.type == Key::Up || k.type == Key::Down) {
        commit_edit();
        if (!editing) move_sel(k.type == Key::Up ? -1 : 1);
      } else {
        edit.handle(k);
        edit_err.clear();
      }
      return true;
    }
    if (filtering) {
      if (k.type == Key::Esc) {
        filtering = false;
        filter.set("");
        rebuild_rows();
      } else if (k.type == Key::Enter || k.type == Key::Down || k.type == Key::Up) {
        filtering = false;
        if (k.type != Key::Enter) move_sel(k.type == Key::Up ? -1 : 1);
      } else {
        filter.handle(k);
        rebuild_rows();
        // First setting that matches.
        for (size_t i = 0; i < rows.size(); ++i)
          if (!rows[i].group && rows[i].spec >= 0) {
            sel = static_cast<int>(i);
            break;
          }
      }
      return true;
    }
    if (handle_global(k) || handle_server_keys(k)) return true;
    if (k.is('m')) {
      open_models();
      return true;
    }
    const Spec *s = sel_spec();
    switch (k.type) {
      case Key::Up: move_sel(-1); return true;
      case Key::Down: move_sel(1); return true;
      case Key::PgUp: move_sel(-std::max(1, list_rows_visible - 1)); return true;
      case Key::PgDn: move_sel(std::max(1, list_rows_visible - 1)); return true;
      case Key::Home: sel = 0; return true;
      case Key::End: sel = std::max(0, static_cast<int>(rows.size()) - 1); return true;
      case Key::Enter: activate(); return true;
      case Key::Left: cycle(-1); return true;
      case Key::Right: cycle(1); return true;
      case Key::F2:
        if (s) start_edit(*s);
        return true;
      case Key::Esc:
        if (!filter.s.empty()) {
          filter.set("");
          rebuild_rows();
        }
        return true;
      case Key::Backspace:
      case Key::Delete:
        if (s) change(s->key, defs.count(s->key) ? defs.at(s->key) : "");
        return true;
      case Key::Ctrl:
        if (k.ch == 's') save_to(target);
        else if (k.ch == 'f') {
          filtering = true;
        } else if (k.ch == 'z') {
          // handled by main (suspend)
        }
        return true;
      case Key::Mouse: {
        if (k.button == 64 || k.button == 65) {
          const int d = k.button == 64 ? -1 : 1;
          if (r_detail.in(k.my, k.mx)) help_scroll = std::max(0, help_scroll + 2 * d);
          else move_sel(d);
          return true;
        }
        if (k.press && k.button == 0) {
          for (const auto &[y, ri] : row_at)
            if (y == k.my && r_list.in(k.my, k.mx)) {
              if (ri == sel) activate();
              else {
                sel = ri;
                help_scroll = 0;
              }
              return true;
            }
          if (k.my >= 1 && k.my <= 4) open_models();
        }
        return true;
      }
      case Key::Char: break;
      default: return true;
    }
    switch (k.ch) {
      case 'j': move_sel(1); break;
      case 'k': move_sel(-1); break;
      case 'h': cycle(-1); break;
      case 'l': cycle(1); break;
      case 'g': sel = 0; break;
      case 'G': sel = std::max(0, static_cast<int>(rows.size()) - 1); break;
      case ' ':
        if (s && (s->kind == Kind::Bool || s->kind == Kind::Choice)) cycle(1);
        else activate();
        break;
      case 'e':
        if (s) start_edit(*s);
        break;
      case 'd':
        if (s) {
          change(s->key, defs.count(s->key) ? defs.at(s->key) : "");
          say(s->label + " back to its default", th::dim);
        }
        break;
      case 'D':
        confirm("Reset every setting", {"Every setting goes back to the default of scripts/start-server.sh (the model too)."},
                {{"Reset all", [this] {
                    undo.push_back(vals);
                    const std::string m = val("XEN_MODEL");
                    vals = defs;
                    refresh_model();
                    rebuild_rows();
                    say("every setting at its default (u undoes)", th::acc2);
                    (void)m;
                  }},
                 {"Cancel", nullptr}},
                th::warn);
        break;
      case 'u':
        if (!undo.empty()) {
          vals = undo.back();
          undo.pop_back();
          refresh_model();
          rebuild_rows();
          say("undone", th::dim);
        } else say("nothing to undo", th::dim);
        break;
      case '/':
        filtering = true;
        break;
      case 'a':
        show_na = !show_na;
        rebuild_rows();
        say(show_na ? "showing settings of the other arch (n/a)" : "hiding settings of the other arch", th::dim);
        break;
      case 'i': open_issues(); break;
      case 'J': help_scroll += 3; break;
      case 'K': help_scroll = std::max(0, help_scroll - 3); break;
      case 'p': open_profiles(); break;
      case 's': save_to(target); break;
      case 'S': save_as(); break;
      default: break;
    }
    return true;
  }

  bool handle_server(const Key &k) {
    if (handle_global(k) || handle_server_keys(k)) return true;
    const int page = std::max(1, log_rows_visible - 1);
    switch (k.type) {
      case Key::Up: log_scroll += 1; break;
      case Key::Down: log_scroll = std::max(0, log_scroll - 1); break;
      case Key::PgUp: log_scroll += page; break;
      case Key::PgDn: log_scroll = std::max(0, log_scroll - page); break;
      case Key::Home: log_scroll = 1 << 28; break;
      case Key::End: log_scroll = 0; break;
      case Key::Mouse:
        if (k.button == 64) log_scroll += 3;
        else if (k.button == 65) log_scroll = std::max(0, log_scroll - 3);
        break;
      case Key::Char:
        if (k.ch == 'k') log_scroll += 1;
        else if (k.ch == 'j') log_scroll = std::max(0, log_scroll - 1);
        else if (k.ch == 'g') log_scroll = 1 << 28;
        else if (k.ch == 'G') log_scroll = 0;
        else if (k.ch == 'w') log_wrap = !log_wrap;
        break;
      default: break;
    }
    return true;
  }

  void chat_send(const std::string &text) {
    int port = 0;
    if (cfg.live) {
      const Snapshot s = snap();
      const pid_t rp = running_pid(s, &port);
      if (rp <= 0) {
        say("no server running: r launches one", th::warn, 5);
        return;
      }
      if (!s.ready) {
        say("the server is still loading", th::warn, 5);
        return;
      }
    } else {
      port = chat_port ? chat_port : std::max(1, std::atoi(val("XEN_PORT").c_str()));
    }
    if (!chat.send(text, port)) {
      say("wait for the answer first (esc stops)", th::warn);
      return;
    }
    chat_input.set("");
    chat_scroll = 0;  // follow the new answer
  }

  bool handle_chat(const Key &k) {
    switch (k.type) {
      case Key::Enter: {
        const std::string t = trim_text(chat_input.s);
        if (!t.empty()) chat_send(t);
        return true;
      }
      case Key::Esc:
        if (chat.busy()) {
          chat.cancel();
          say("stopping the answer", th::dim);
        } else chat_input.set("");
        return true;
      case Key::Up: ++chat_scroll; return true;
      case Key::Down: chat_scroll = std::max(0, chat_scroll - 1); return true;
      case Key::PgUp: chat_scroll += std::max(1, chat_rows_visible - 1); return true;
      case Key::PgDn: chat_scroll = std::max(0, chat_scroll - std::max(1, chat_rows_visible - 1)); return true;
      case Key::Home: chat_scroll = 1 << 28; return true;
      case Key::End: chat_scroll = 0; return true;
      case Key::F4:
        if (chat.busy()) {
          say("wait for the answer first", th::warn);
          return true;
        }
        chat.clear();
        say("new conversation", th::dim);
        return true;
      case Key::Mouse:
        if (k.button != 64 && k.button != 65) break;  // not the wheel: the click goes on to the global targets (the header's tabs); the input line owns keys, not the mouse
        if (k.button == 64) ++chat_scroll;
        else chat_scroll = std::max(0, chat_scroll - 1);
        return true;
      default: break;
    }
    // Typing owns the rest of the keys: letters go into the message, not to the global bindings.
    if (chat_input.handle(k)) return true;
    if (handle_global(k)) return true;
    return true;
  }

  int c_h_ = 24, c_h_w_ = 120;

  bool handle(const Key &k) {
    if (k.type != Key::Ctrl || k.ch != 'c') ctrl_c = 0;
    if (k.type == Key::Resize) return !quit;
    if (!pops.empty()) handle_popup(k);
    else if (tab_ == kSettings) handle_configure(k);
    else if (tab_ == kServer) handle_server(k);
    else if (tab_ == kBench) handle_bench(k);
    else handle_chat(k);
    return !quit;
  }
};

App::App(AppConfig cfg) : p_(std::make_unique<Impl>(std::move(cfg))) {}
App::~App() = default;

void App::draw(Canvas &c) {
  p_->c_h_ = c.h();
  p_->c_h_w_ = c.w();
  p_->draw(c);
}
bool App::handle(const Key &k) { return p_->handle(k); }

std::string App::exit_note() const {
  const Snapshot s = p_->snap();
  std::string job;
  if (p_->rank_job.busy())
    job = "the ranking run of " + p_->rank_job.state().title + " keeps running: pid " + std::to_string(p_->rank_job.state().pid) + ", log " +
          tilde(p_->rank_log.path) + "\n";
  if (s.pid > 0 && s.ours_alive) {
    const std::string host = p_->val("XEN_HOST");
    const std::string h = (host == "0.0.0.0" || host.empty()) ? (p_->lan.empty() ? "127.0.0.1" : p_->lan) : host;
    return "xenithium-server keeps running: pid " + std::to_string(s.pid) + ", http://" + h + ":" + std::to_string(s.port) + "/v1, log " +
           tilde(p_->log.path) + "\n(stop it with scripts/launch.sh -> x, or kill " + std::to_string(s.pid) + ")" + (job.empty() ? "" : "\n" + job);
  }
  return job;
}

const Values &App::values() const { return p_->vals; }
void App::set_value(const std::string &key, const std::string &v) { p_->change(key, v); }
int App::tab() const { return p_->tab_; }
bool App::dirty() const { return p_->dirty_count() > 0; }
std::string App::selected_key() const {
  const Spec *s = p_->sel_spec();
  return s ? s->key : "";
}
bool App::popup_open() const { return !p_->pops.empty(); }
std::string App::models_dir() const { return p_->models_dir(); }
std::string App::preview() const { return p_->preview_cmd; }

void App::set_chat_port(int port) { p_->chat_port = port; }
void App::show_tab(int t) { p_->tab_ = t; }  // (tests) open a tab by its number, also one the header hides
bool App::tab_click_point(int t, int &x, int &y) const {
  for (const auto &[r, tt] : p_->tab_at)
    if (tt == t) {
      x = r.x + r.w / 2;
      y = r.y;
      return true;
    }
  return false;
}
bool App::chat_busy() const { return p_->chat.busy(); }
std::string App::chat_answer() const {
  const ChatSession::State st = p_->chat.get();
  for (auto it = st.msgs.rbegin(); it != st.msgs.rend(); ++it)
    if (it->role == "assistant") return it->text;
  return "";
}
bool App::chat_failed() const {
  const ChatSession::State st = p_->chat.get();
  return st.status == ChatSession::Status::error || st.status == ChatSession::Status::cancelled;
}
int App::frame_wait_ms() const {
  // a plan or a table in flight, or a table waiting for the selection to rest: short frames, so it shows when it is in
  const bool resting = !p_->whatif_want_key.empty() && (p_->whatif_want_key != p_->whatif_key || p_->vals != p_->whatif_base);
  return p_->chat.busy() || p_->plan_busy() || p_->whatif_busy() || (p_->tab_ == Impl::kSettings && resting) ? 25 : 200;
}
MemPlanView App::memory_plan() const { return p_->plan; }
void App::settle() {
  p_->update_plan();
  if (p_->plan_job.valid()) p_->plan_job.wait();
  p_->update_plan();
  if (p_->tab_ != Impl::kSettings) return;
  p_->update_whatif(true);
  if (p_->whatif_job.valid()) p_->whatif_job.wait();
  p_->update_whatif(true);
}

void App::set_bench_runs_dir(const std::string &dir) {
  p_->cfg.bench_runs = dir;
  p_->rebuild_bench();
}
std::vector<std::string> App::bench_model_names() const {
  std::vector<std::string> out;
  for (const auto &m : p_->bench.models()) out.push_back(m.name);
  return out;
}
int App::bench_run_count() const { return p_->bench.total_runs(); }
std::string App::bench_selected_decode() const {
  if (p_->bench_detail_run < 0 || p_->bench_detail_run >= static_cast<int>(p_->bench_runs.size())) return "";
  const BenchRun *r = p_->bench_runs[p_->bench_detail_run];
  if (!r) return "";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.1f", r->decode_tps);
  return buf;
}

}  // namespace xenithium:launch
