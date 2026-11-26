// Host-only gate for the launcher (tools/launch): the settings catalog against the real
// scripts/start-server.sh and docs/server-env.md, env files, validation, the key decoder, text
// layout, and whole screen flows on an off-screen canvas (no terminal, no processes).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "app.hpp"
#include "bench.hpp"
#include "proc.hpp"
#include "settings.hpp"
#include "term.hpp"

using namespace xenithium::launch;

namespace {
int g_failures = 0;
void expect(bool ok, const std::string &what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

std::string screen(const Canvas &c) {
  std::string s;
  for (int y = 0; y < c.h(); ++y) s += c.row(y) + "\n";
  return s;
}
bool has(const std::string &hay, const std::string &needle) { return hay.find(needle) != std::string::npos; }

void write(const std::string &path, const std::string &text) { std::ofstream(path) << text; }

Key ch(char c) { return Key::chr(static_cast<char32_t>(c)); }

// A fake /v1/chat/completions server: answers every request with the same SSE stream (in tiny
// chunked pieces, so the client's incremental parser is exercised), and records the requests.
struct FakeChat {
  int fd = -1, port = 0;
  std::thread th;
  std::vector<std::string> reqs;

  bool start() {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0 || ::listen(fd, 4) != 0) return false;
    socklen_t n = sizeof a;
    if (::getsockname(fd, reinterpret_cast<sockaddr *>(&a), &n) != 0) return false;
    port = ntohs(a.sin_port);
    th = std::thread([this] {
      for (;;) {
        const int c = ::accept(fd, nullptr, nullptr);
        if (c < 0) return;
        char buf[1 << 16];
        std::string req;
        for (;;) {
          const ssize_t r = ::recv(c, buf, sizeof buf, 0);
          if (r <= 0) break;
          req.append(buf, static_cast<size_t>(r));
          if (req.find("\r\n\r\n") != std::string::npos) {
            // Read the whole body too (Content-Length), then stop.
            const size_t he = req.find("\r\n\r\n") + 4;
            const size_t cl = req.find("Content-Length:");
            size_t need = 0;
            if (cl != std::string::npos) need = static_cast<size_t>(std::atol(req.c_str() + cl + 15));
            if (req.size() >= he + need) break;
          }
        }
        reqs.push_back(req);
        // The SSE stream: reasoning then content deltas, the finish chunk, usage, [DONE].
        std::string body;
        body += "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"hmm \"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
        body += "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
        body += "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"xenithium\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n";
        body += "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"stop\"}]}\n\n";
        body += "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"total_tokens\":9,\"completion_tokens\":2}}\n\n";
        body += "data: [DONE]\n\n";
        std::string out;
        for (size_t off = 0; off < body.size();) {
          const size_t n = std::min<size_t>(7, body.size() - off);  // tiny chunks exercise the client
          char h[16];
          std::snprintf(h, sizeof h, "%zx\r\n", n);
          out += h;
          out += body.substr(off, n);
          out += "\r\n";
          off += n;
        }
        out += "0\r\n\r\n";
        const std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n" + out;
        for (size_t off = 0; off < resp.size();) {
          const ssize_t r = ::send(c, resp.data() + off, resp.size() - off, MSG_NOSIGNAL);
          if (r <= 0) break;
          off += static_cast<size_t>(r);
        }
        ::close(c);
      }
    });
    return true;
  }
  void stop() {
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      fd = -1;
    }
    if (th.joinable()) th.join();
  }
  ~FakeChat() { stop(); }
};

void test_catalog(const std::string &root) {
  const Catalog cat = load_catalog(root);
  std::string w;
  for (const auto &x : cat.warnings) w += x + "; ";
  expect(cat.warnings.empty(), "schema, start-server.sh and docs/server-env.md list the same settings" + (w.empty() ? "" : " (" + w + ")"));
  std::string script;
  read_text(join_path(root, "scripts/start-server.sh"), script);
  const auto vars = parse_script_defaults(script);
  bool all = !vars.empty();
  for (const auto &v : vars) all = all && cat.find(v.key) && cat.find(v.key)->in_script && cat.find(v.key)->def == v.def;
  expect(all, "every start-server.sh variable is in the catalog with its default (" + std::to_string(vars.size()) + ")");
  const Spec *port = cat.find("XEN_PORT");
  expect(port && port->def == "8000" && port->flag == "--port" && !port->help.empty(), "XEN_PORT: default, flag and help from the script and docs");
  const Spec *hot = cat.find("XEN_HOT_RANK");
  expect(hot && hot->def == "auto" && hot->path_auto, "XEN_HOT_RANK keeps its ${X-default} default (auto: the model's own ranking)");
  expect(!cat.find("XEN_ROUTE_STATE"), "XEN_ROUTE_STATE is gone (round 169: the ranking file is recorded while serving)");
  expect(check_value(*hot, "auto", "qwen38", root).empty() && check_value(*hot, "/nonexistent/hot-rank.xrc", "qwen38", root).empty(),
         "XEN_HOT_RANK: auto valid, and a file that does not exist yet is fine (the server records it)");
  const Spec *hotn = cat.find("XEN_HOT");
  expect(hotn && hotn->def.empty() && hotn->flag == "--hot" && hotn->scope == Scope::MoE && hotn->kind == Kind::Int,
         "XEN_HOT: a MoE setting, empty = auto (as many as fit)");
  const Spec *pf = cat.find("XEN_PREFILL_PART_MS");
  expect(pf && pf->def == "-1", "${X:--1} parses as -1");
  {  // The drafter defaults
    const Spec *seg = cat.find("XEN_SEG_ROWS"), *mfp = cat.find("XEN_DRAFTER_MTP_FROM");
    expect(seg && seg->def == "16" && seg->kind == Kind::Choice && std::find(seg->choices.begin(), seg->choices.end(), "8") != seg->choices.end(),
           "XEN_SEG_ROWS: 16 = the default (was 8), 8 stays offered");
    expect(mfp && mfp->def == "9" && mfp->kind == Kind::Int, "XEN_DRAFTER_MTP_FROM: 9 = the default (was 5)");
  }
  {
    const Catalog pc = build_catalog("", "| setting | default | flag | meaning |\n| --- | --- | --- | --- |\n| `XEN_W4_ITEMS` | `512` | `--w4-items` | splits q\\|k\\|v |\n");
    const Spec *w4 = pc.find("XEN_W4_ITEMS");
    expect(w4 && has(w4->help, "q|k|v") && !has(w4->help, "\\|"), "docs cells keep their escaped pipes as text");
  }
  const Spec *pfl = cat.find("XEN_PF_LOG");
  expect(pfl && pfl->diag && !pfl->help.empty(), "diagnostic switches come from the docs' second table");
  // Every known setting has the launcher's own description, and the hiding rules read the current values.
  bool briefs = true;
  for (const auto &x : cat.specs) briefs = briefs && (!x.known || !x.brief.empty());
  expect(briefs, "every setting has a brief description");
  {  // the descriptions' layout code spans closed per paragraph, every choice of a closed list named
    std::string bad_ticks, no_value, no_kv;
    for (const auto &x : cat.specs) {
      if (!x.known) continue;
      size_t p0 = 0;
      while (p0 <= x.brief.size()) {
        const size_t nl = x.brief.find('\n', p0);
        const std::string para = x.brief.substr(p0, nl == std::string::npos ? std::string::npos : nl - p0);
        if (std::count(para.begin(), para.end(), '`') % 2) bad_ticks += x.key + " ";
        if (nl == std::string::npos) break;
        p0 = nl + 1;
      }
      if (x.kind == Kind::Choice)  // (a list of numbers is a range: its special values only)
        for (const auto &c : x.choices)
          if (!c.empty() && !std::isdigit(static_cast<unsigned char>(c[0])) && !has(x.brief, "`" + c + "`")) no_value += x.key + "=" + c + " ";
      // a setting the memory plan reads says what it does to the KV context or the memory (the numbers follow in the panel)
      if (moves_memory(x.key) && x.key != "XEN_MODEL" && !has(x.brief, "KV") && !has(x.brief, "VRAM") && !has(x.brief, "RAM") && !has(x.brief, "memory")) no_kv += x.key + " ";
    }
    expect(bad_ticks.empty(), "descriptions: every code span closes in its paragraph (" + bad_ticks + ")");
    expect(no_value.empty(), "descriptions: every choice of a closed list has its line (" + no_value + ")");
    expect(no_kv.empty(), "descriptions: a setting that moves memory says so (" + no_kv + ")");
    // what a setting gives and takes is said in words: no measured speeds, no "Gain" / "Cost" / "Measured" lines
    std::string speeds;
    for (const auto &x : cat.specs) {
      if (!x.known) continue;
      for (const char *w : {"tok/s", "ms per step", "ms per token", "% faster", "% slower", "% tok", "Measured", "\nGain", "\nCost", "KL 0."})
        if (has(x.brief, w)) speeds += x.key + " ";
    }
    expect(speeds.empty(), "descriptions: what a setting gives and takes, without measured results (" + speeds + ")");
  }
  {  // the KV context before a launch: which settings get a table, its values, the pool in words
    const Spec *kvs = cat.find("XEN_KV"), *sl = cat.find("XEN_SLOTS"), *dm = cat.find("XEN_DRAFTER_MODEL"), *vis = cat.find("XEN_VISION"), *port = cat.find("XEN_PORT");
    expect(moves_memory("XEN_KV") && moves_memory("XEN_SLOTS") && moves_memory("XEN_DRAFTER_INT4") && !moves_memory("XEN_PORT") && !moves_memory("XEN_QUEUE_POLICY"),
           "moves_memory: the settings the memory plan reads");
    expect(kvs && whatif_values(*kvs, "int8") == std::vector<std::string>({"int8", "bf16", "int8-v4"}), "whatif_values: a closed list's choices");
    expect(sl && whatif_values(*sl, "12") == std::vector<std::string>({"1", "2", "4", "8", "12", "16", "32", "64"}), "whatif_values: a number's suggestions with the current value, ascending");
    expect(vis && whatif_values(*vis, "0") == std::vector<std::string>({"1", "0"}) && whatif_name(*vis, "1") == "on" && whatif_name(*vis, "0") == "off", "whatif_values: on / off");
    expect(dm && whatif_values(*dm, "/m/a", {"/m/a", "/m/b"}) == std::vector<std::string>({"", "/m/a", "/m/b"}) && whatif_name(*dm, "/m/a") == "a" && whatif_name(*dm, "") == "(none)",
           "whatif_values: a drafter path's candidates after none, named by their folder");
    expect(port && whatif_values(*port, "8000").empty() && dm && whatif_values(*dm, "/m/a").empty(), "whatif_values: nothing to compare = no table");
    expect(sl && whatif_name(*sl, "16") == "16" && whatif_name(*cat.find("XEN_CONTEXT"), "524288") == "524,288", "whatif_name: large numbers with separators");
    expect(kv_capacity_text(1112000, 524288, 16) == "2 requests at the full 524,288-token max context, or all 16 slots at 69,440 tokens each",
           "kv_capacity_text: full contexts, or every slot at once (" + kv_capacity_text(1112000, 524288, 16) + ")");
    expect(kv_capacity_text(262144, 32768, 8) == "every one of the 8 slots can reach the full 32,768-token max context at once", "kv_capacity_text: a pool of slots x context");
    expect(kv_capacity_text(1000, 32768, 1) == "less than one request at the full 32,768-token max context" && kv_capacity_text(0, 1, 1).empty(), "kv_capacity_text: a pool below one context");
    expect(kv_delta_text(556000, 1112000) == "-50 %" && kv_delta_text(1150000, 1112000) == "+3.4 %" && kv_delta_text(1112000, 1112000) == "same" && kv_delta_text(0, 5).empty(),
           "kv_delta_text: a value's pool against the current one");
  }
  Values dv = defaults_of(cat);
  const Spec *vocab = cat.find("XEN_DRAFT_VOCAB"), *aging = cat.find("XEN_QUEUE_AGING"), *reserve = cat.find("XEN_RESERVE");
  const Spec *prof = cat.find("XEN_PF_PROFILE"), *chunk = cat.find("XEN_PREFILL_CHUNK"), *slots = cat.find("XEN_SLOTS");
  expect(prof && unused_reason(*prof, "qwen38", dv).empty() && !unused_reason(*prof, "qwen35", dv).empty(), "XEN_PF_PROFILE is a qwen38-only switch");
  const Spec *qtiles = cat.find("XEN_QSA_PF_TILES");
  expect(qtiles && qtiles->diag && unused_reason(*qtiles, "qwen38", dv).empty() && !unused_reason(*qtiles, "qwen35", dv).empty(), "XEN_QSA_PF_TILES is a qwen38-only switch");
  const Spec *dprof = cat.find("XEN_DEC_PROFILE");
  expect(dprof && dprof->diag && unused_reason(*dprof, "qwen38", dv).empty() && !unused_reason(*dprof, "qwen35", dv).empty(), "XEN_DEC_PROFILE is a qwen38-only diagnostic");
  expect(vocab && unused_reason(*vocab, "qwen35", dv).empty(), "draft vocabulary shown with the drafter on");
  {  // The KV split by tokens: qwen38 on four cards only
    const Spec *ks = cat.find("XEN_KV_SPLIT");
    Values t4 = dv, t2 = dv;
    t4["XEN_Q38_TP"] = "4", t2["XEN_Q38_TP"] = "2";
    expect(ks && ks->kind == Kind::Choice && dv.at("XEN_KV_SPLIT") == "heads" && unused_reason(*ks, "qwen38", t4).empty() && !unused_reason(*ks, "qwen38", t2).empty() &&
               !unused_reason(*ks, "qwen35", t4).empty() && moves_memory("XEN_KV_SPLIT"),
           "XEN_KV_SPLIT: qwen38 at TP4 only, heads by default, moves the memory plan");
  }
  {  // Unequal segments on the dense 27B with its drafter (not llama / flint)
    const Spec *uq = cat.find("XEN_UNEQUAL_SEGMENTS");
    Values nd = dv;
    nd["XEN_DRAFTER"] = "0";
    expect(uq && unused_reason(*uq, "qwen35", dv).empty() && !unused_reason(*uq, "qwen35", nd).empty() && !unused_reason(*uq, "flint", dv).empty() &&
               !unused_reason(*uq, "llama", dv).empty(),
           "XEN_UNEQUAL_SEGMENTS: qwen35 with its drafter (round 229), not llama / flint");
  }
  {  // XEN_HOT: the MoE archs only; qwen38's reserve applies only with auto hot experts
    expect(hotn && unused_reason(*hotn, "qwen38", dv).empty() &&
               !unused_reason(*hotn, "qwen35", dv).empty() && !unused_reason(*hotn, "llama", dv).empty() && !unused_reason(*hotn, "flint", dv).empty(),
           "XEN_HOT: shown for qwen38, hidden for the dense archs");
    Values v2 = dv;
    v2["XEN_HOT"] = "1000";
    expect(reserve && unused_reason(*reserve, "qwen38", dv).empty() && unused_reason(*reserve, "qwen38", v2).empty(),
           "qwen38 reserve: shown with auto and pinned hot experts (a pinned count is cut to the fit minus the reserve)");
  }
  {  // arch flint: the qwen35 runtime's settings minus drafter / lm_head, plus the step graphs;
     // (flint r07) the prefill numerics (int4, attention) and the prefix cache too, not W4A8's kept GEMMs
    const Spec *gr = cat.find("XEN_GRAPHS"), *lm = cat.find("XEN_LM_HEAD"), *pc = cat.find("XEN_PREFIX_CACHE"), *kv = cat.find("XEN_KV");
    const Spec *tp = cat.find("XEN_TP"), *dr = cat.find("XEN_DRAFTER"), *w4 = cat.find("XEN_PREFILL_W4"), *pa = cat.find("XEN_PREFILL_ATTN");
    const Spec *a8 = cat.find("XEN_PREFILL_A8_KEEP");
    expect(gr && lm && pc && kv && tp && dr && w4 && pa && a8 && unused_reason(*gr, "flint", dv).empty() && !unused_reason(*lm, "flint", dv).empty() &&
               unused_reason(*pc, "flint", dv).empty() && unused_reason(*kv, "flint", dv).empty() && unused_reason(*tp, "flint", dv).empty() &&
               !unused_reason(*dr, "flint", dv).empty() && !unused_reason(*gr, "qwen35", dv).empty() && unused_reason(*w4, "flint", dv).empty() &&
               unused_reason(*pa, "flint", dv).empty() && !unused_reason(*a8, "flint", dv).empty(),
           "flint: graphs, KV, TP, prefill int4 / attention and the prefix cache shown; lm_head, W4A8 keep and the drafter hidden");
    // With a DFlash drafter (Flint-DFlash2) its settings show; trees, 16-row segments, MTP from N
    // streams and the draft vocabulary stay hidden (chains over the whole int4 lm_head, no MTP layer)
    Values df = dv;
    df["XEN_DRAFTER"] = "1", df["XEN_DRAFTER_MODEL"] = "/models/flint-dflash2";
    bool shown = true, hidden = true;
    for (const char *k : {"XEN_DRAFTER", "XEN_DRAFTER_MODEL", "XEN_DRAFTER_DEFER", "XEN_DRAFTER_CUT", "XEN_DRAFT_DEPTHS", "XEN_ADAPTIVE_DEPTH"})
      shown = shown && cat.find(k) && unused_reason(*cat.find(k), "flint", df).empty();
    for (const char *k : {"XEN_DRAFTER_TREE", "XEN_SEG_ROWS", "XEN_DRAFTER_MTP_FROM", "XEN_DRAFTER_INT4", "XEN_DRAFT_VOCAB", "XEN_DRAFTER_WIDE"})
      hidden = hidden && cat.find(k) && !unused_reason(*cat.find(k), "flint", df).empty();
    Values nm = df;
    nm["XEN_DRAFTER_MODEL"] = "";
    expect(shown && hidden && !unused_reason(*cat.find("XEN_ADAPTIVE_DEPTH"), "flint", nm).empty(),
           "flint with a DFlash drafter: its settings shown, trees / 16-row segments / MTP / draft vocabulary hidden; none without the model");
  }
  {  // qwen38 reads it with an MTP drafter
    auto d2 = dv;
    d2["XEN_MTP"] = "";
    const bool off = vocab && !unused_reason(*vocab, "qwen38", d2).empty();
    d2["XEN_MTP"] = "/models/mtp";
    expect(off && unused_reason(*vocab, "qwen38", d2).empty(), "qwen38 draft vocabulary: shown with XEN_MTP set, hidden without");
    const Spec *ad = cat.find("XEN_ADAPTIVE_DEPTH");
    d2["XEN_MTP"] = "";
    const bool ad_off = ad && !unused_reason(*ad, "qwen38", d2).empty();
    d2["XEN_MTP"] = "/models/mtp";
    expect(ad_off && unused_reason(*ad, "qwen38", d2).empty(), "adaptive draft depth: shown with XEN_MTP set, hidden without (round 111)");
    d2["XEN_DRAFTER"] = "0";
    const bool ad35_off = !unused_reason(*ad, "qwen35", d2).empty();
    d2["XEN_DRAFTER"] = "1";
    expect(ad35_off && unused_reason(*ad, "qwen35", d2).empty(), "adaptive draft depth: a qwen35 setting too, shown with the drafter on (round 119)");
    if (const Spec *lk = cat.find("XEN_LOOKUP")) {  // The engine's, with any drafter
      d2["XEN_DRAFTER"] = "0";
      const bool lk_off = !unused_reason(*lk, "qwen35", d2).empty() && !unused_reason(*lk, "llama", d2).empty();
      d2["XEN_DRAFTER"] = "1";
      expect(lk_off && unused_reason(*lk, "qwen35", d2).empty() && unused_reason(*lk, "qwen38", d2).empty(),
             "lookup drafting: shown with a drafter (qwen35 drafter on, qwen38 XEN_MTP), hidden without (round 53)");
    } else {
      expect(false, "lookup drafting: XEN_LOOKUP in the catalog (round 53)");
    }
  }
  {  // MTP depth: the user's cap on one decoding sequence's drafts, qwen38 with XEN_MTP (1-3), the dense 27B with its drafter (1-6)
    const Spec *md = cat.find("XEN_MTP_DEPTH");
    Values with = dv, without = dv, q35 = dv, q35_depths = dv, q35_off = dv;
    with["XEN_MTP"] = "/models/mtp";
    without["XEN_MTP"] = "";
    q35["XEN_DRAFTER"] = "1", q35["XEN_DRAFT_DEPTHS"] = "";
    q35_depths = q35, q35_depths["XEN_DRAFT_DEPTHS"] = "15,7,4,3";
    q35_off["XEN_DRAFTER"] = "0";
    expect(md && md->def == "3" && md->flag == "--mtp-depth" && md->choices.size() == 6 && md->choices[0] == "3" && md->choices[2] == "1" &&
               unused_reason(*md, "qwen38", with).empty() && !unused_reason(*md, "qwen38", without).empty() && unused_reason(*md, "qwen35", q35).empty() &&
               !unused_reason(*md, "qwen35", q35_depths).empty() && !unused_reason(*md, "qwen35", q35_off).empty() && !unused_reason(*md, "llama", q35).empty() &&
               !check_value(*md, "6", "qwen38", "").empty() && check_value(*md, "3", "qwen38", "").empty() && check_value(*md, "6", "qwen35", "").empty(),
           "MTP depth: default 3, flag from the docs; qwen38 1-3 with XEN_MTP, the dense 27B 1-6 with its drafter (not with per-stream depths), not llama");
  }
  {  // Trained page selection: the dense 27B's server-read switches, each shown once its bank and the switch it builds on are set
    const char *const keys[] = {"XEN_TRAINED_QSA_DELTA_DIR", "XEN_TRAINED_QSA_TILED_SCORE", "XEN_TRAINED_QSA_POOL_FP16", "XEN_TRAINED_QSA_TREE",
                                "XEN_QSA_XMX_SCORE", "XEN_QSA_GATHER", "XEN_QSA_RESIDENT_SNAPSHOTS", "XEN_QSA_RESIDENT_FORK_COPY",
                                "XEN_TRAINED_QSA_DECODE_MIN_POS", "XEN_TRAINED_QSA_PREFILL_XMX", "XEN_TRAINED_QSA_PREFILL_MIN_POS",
                                "XEN_TRAINED_QSA_STREAM_PROJECTION", "XEN_QSA_TRIAL_ROUTE_LOG"};
    bool shape = true;
    for (const char *k : keys) {
      const Spec *x = cat.find(k);
      shape = shape && x && x->known && x->diag && x->in_docs && !x->brief.empty() && unused_reason(*x, "llama", dv) == "ignored for llama" &&
              !unused_reason(*x, "qwen38", dv).empty();
    }
    expect(shape, "trained page selection: known, server-read, documented, described; the dense 27B only (not llama / qwen38)");
    const Spec *bank = cat.find("XEN_TRAINED_QSA_DELTA_DIR"), *tile = cat.find("XEN_TRAINED_QSA_TILED_SCORE");
    const Spec *tree = cat.find("XEN_TRAINED_QSA_TREE"), *xmx = cat.find("XEN_QSA_XMX_SCORE"), *snap = cat.find("XEN_QSA_RESIDENT_SNAPSHOTS");
    const Spec *fork = cat.find("XEN_QSA_RESIDENT_FORK_COPY"), *pfx = cat.find("XEN_TRAINED_QSA_PREFILL_XMX"), *pfmin = cat.find("XEN_TRAINED_QSA_PREFILL_MIN_POS");
    const Spec *merged = cat.find("XEN_TRAINED_QSA_PREFILL_MERGED");
    Values nob = dv, b = dv;
    nob["XEN_TRAINED_QSA_DELTA_DIR"] = "";
    b = nob, b["XEN_TRAINED_QSA_DELTA_DIR"] = "/m/bank", b["XEN_PREFIX_CACHE"] = "1", b["XEN_PREFILL_SELECT"] = "0";
    Values tiled = b, fp16 = b, snaps = b, pf = b, pfsel = b;
    tiled["XEN_TRAINED_QSA_TILED_SCORE"] = "1";
    fp16["XEN_TRAINED_QSA_POOL_FP16"] = "1";
    snaps["XEN_QSA_RESIDENT_SNAPSHOTS"] = "1";
    pf["XEN_TRAINED_QSA_PREFILL_XMX"] = "1";
    pfsel["XEN_PREFILL_SELECT"] = "256";
    expect(bank && unused_reason(*bank, "qwen35", nob).empty() && tile && !unused_reason(*tile, "qwen35", nob).empty() && unused_reason(*tile, "qwen35", b).empty() &&
               !cat.find("XEN_TRAINED_QSA_LAYER3_DELTA_DIR") && !cat.find("XEN_TRAINED_QSA_TRACE_MARKER") && !cat.find("XEN_PF_TOPK_REPLAY_DIR") &&
               !cat.find("XEN_BENCH_VERIFY_LOG"),
           "trained page selection: the switches wait for a bank (the bank folder does not); no layer-3 bank, trace or replay setting");
    expect(tree && !unused_reason(*tree, "qwen35", b).empty() && unused_reason(*tree, "qwen35", tiled).empty() && xmx && !unused_reason(*xmx, "qwen35", b).empty() &&
               unused_reason(*xmx, "qwen35", fp16).empty() && fork && !unused_reason(*fork, "qwen35", b).empty() && unused_reason(*fork, "qwen35", snaps).empty() &&
               snap && unused_reason(*snap, "qwen35", b).empty(),
           "trained page selection: trees need tiled scoring, XMX FP16 summaries, the fork copy the snapshots");
    expect(pfx && unused_reason(*pfx, "qwen35", b).empty() && !unused_reason(*pfx, "qwen35", pfsel).empty() && pfmin && !unused_reason(*pfmin, "qwen35", b).empty() &&
               unused_reason(*pfmin, "qwen35", pf).empty() && merged && !unused_reason(*merged, "qwen35", pf).empty() && check_value(*pfmin, "32768", "qwen35", "").empty() &&
               !check_value(*pfmin, "32769", "qwen35", "").empty() && !check_value(*pfmin, "1024", "qwen35", "").empty(),
           "trained page selection: the selected prompt reading off beside XEN_PREFILL_SELECT; its start a multiple of 64 from 2,112; merged calls need the 4-slot switch");
    const Spec *share = cat.find("XEN_PREFILL_SHARE");
    expect(share && share->diag && share->kind == Kind::Float && unused_reason(*share, "llama", dv).empty() && check_value(*share, "0.33", "qwen35", "").empty() &&
               check_value(*share, "", "qwen35", "").empty() && !check_value(*share, "2", "qwen35", "").empty(),
           "XEN_PREFILL_SHARE: a server-read share for every arch, 0..1 or empty");
    {  // Page selection on the int8-v4 KV: the long-prompt one runs, the decode one has no kernel; the bank's start-up needs
      auto errs = [&](const Values &x, const std::string &key) {
        int n = 0;
        for (const auto &i : validate(cat, x, "qwen35", "")) n += i.key == key && i.error && i.msg != "directory not found";  // (made-up paths)
        return n;
      };
      const Spec *as = cat.find("XEN_ATTN_SELECT"), *ps = cat.find("XEN_PREFILL_SELECT");
      Values v4 = dv, i8 = dv;
      v4["XEN_KV"] = "int8-v4", v4["XEN_ATTN_SELECT"] = "512", v4["XEN_PREFILL_SELECT"] = "512";
      i8 = v4, i8["XEN_KV"] = "int8";
      expect(as && ps && !unused_reason(*as, "qwen35", v4).empty() && unused_reason(*as, "qwen35", i8).empty() && unused_reason(*ps, "qwen35", v4).empty() &&
                 errs(v4, "XEN_ATTN_SELECT") == 0 && errs(v4, "XEN_PREFILL_SELECT") == 0,
             "page selection on int8-v4: the long-prompt one runs (512 pages); the decode one is unused (launched as 0), never an error");
      Values ok = dv;
      ok["XEN_KV"] = "int8-v4", ok["XEN_KV_POOL_TOKENS"] = "884416", ok["XEN_CONTEXT"] = "262144", ok["XEN_DRAFTER"] = "1", ok["XEN_DRAFTER_MODEL"] = "/m/dspine",
      ok["XEN_ATTN_SELECT"] = "0", ok["XEN_TRAINED_QSA_DELTA_DIR"] = "/m/bank", ok["XEN_TRAINED_QSA_TREE"] = "1", ok["XEN_TRAINED_QSA_TILED_SCORE"] = "1";
      Values autopool = ok, notile = ok, i8kv = ok, nodraft = ok, sel = ok;
      autopool["XEN_KV_POOL_TOKENS"] = "auto", notile["XEN_TRAINED_QSA_TILED_SCORE"] = "0", i8kv["XEN_KV"] = "int8", nodraft["XEN_DRAFTER_MODEL"] = "";
      sel["XEN_ATTN_SELECT"] = "512";
      int all = 0;
      for (const Values *x : {&ok, &autopool, &notile, &sel})
        for (const auto &i : validate(cat, *x, "qwen35", "")) all += i.error && i.msg != "directory not found" && i.key.find("QSA") != std::string::npos;
      expect(all == 0 && errs(i8kv, "XEN_TRAINED_QSA_DELTA_DIR") == 1 && errs(nodraft, "XEN_TRAINED_QSA_DELTA_DIR") == 1,
             "trained bank: refused without int8-v4 or a block drafter; an auto pool, unused page selection or an unused tree switch never block");
    }
    const auto g = std::find(cat.groups.begin(), cat.groups.end(), std::string("Trained page selection"));
    const auto n = std::find(cat.groups.begin(), cat.groups.end(), std::string("Numerics & kernels"));
    expect(g != cat.groups.end() && n != cat.groups.end() && g == n + 1, "trained page selection: its own group, after the numerics");
  }
  const Spec *dfl = cat.find("XEN_DRAFTER_MODEL");
  expect(dfl && unused_reason(*dfl, "qwen35", dv).empty() && unused_reason(*dfl, "qwen38", dv).empty() && !unused_reason(*dfl, "llama", dv).empty(),
         "DFlash drafter: qwen35 and qwen38 (round 171), shown with the drafter on; not llama");
  {  // On qwen38 the DFlash drafter replaces XEN_MTP (and its depth); its defer / cut need it
    Values q = dv;
    q["XEN_MTP"] = "~/models/qwen38-w4a16/runtime/mtp-int4-g32";
    q["XEN_DRAFTER_MODEL"] = "";
    Values qd = q;
    qd["XEN_DRAFTER_MODEL"] = "~/models/qwen38-flash-next-dflash2";
    const Spec *mtp = cat.find("XEN_MTP"), *cut = cat.find("XEN_DRAFTER_CUT"), *dep = cat.find("XEN_MTP_DEPTH"), *voc = cat.find("XEN_DRAFT_VOCAB");
    Values qd0 = qd;  // XEN_DRAFTER_MTP_FROM 0 = DFlash alone; above 0 both drafters load
    qd0["XEN_DRAFTER_MTP_FROM"] = "0";
    qd["XEN_DRAFTER_MTP_FROM"] = "9";
    const Spec *mfp = cat.find("XEN_DRAFTER_MTP_FROM");
    expect(mtp && cut && dep && voc && mfp && unused_reason(*mtp, "qwen38", q).empty() && !unused_reason(*mtp, "qwen38", qd0).empty() &&
               !unused_reason(*cut, "qwen38", q).empty() && unused_reason(*cut, "qwen38", qd).empty() && !unused_reason(*dep, "qwen38", qd).empty() &&
               unused_reason(*voc, "qwen38", qd).empty(),
           "qwen38 DFlash drafter: replaces XEN_MTP (XEN_DRAFTER_MTP_FROM 0) / XEN_MTP_DEPTH, enables its cut, keeps the draft vocabulary (round 171)");
    expect(unused_reason(*mtp, "qwen38", qd).empty() && unused_reason(*mfp, "qwen38", qd).empty() && !unused_reason(*mfp, "qwen38", q).empty(),
           "qwen38 both drafters: XEN_MTP beside XEN_DRAFTER_MODEL with XEN_DRAFTER_MTP_FROM > 0 (engine round 264); the switch needs both");
  }
  const Spec *pfa = cat.find("XEN_PREFILL_ATTN");
  expect(pfa && pfa->def == "auto" && pfa->flag == "--prefill-attn" && pfa->choices.size() == 6 && pfa->choices[0] == "auto" && pfa->choices[3] == "int8" && pfa->choices[4] == "int8-fast" &&
             pfa->choices[5] == "onednn" &&
             unused_reason(*pfa, "qwen35", dv).empty() && !unused_reason(*pfa, "qwen38", dv).empty(),
         "prefill attention numerics: a qwen35 choice (auto / exact / fast / int8 / int8-fast / onednn, rounds 34 / 36 / 41), default auto, flag from the docs");
  const Spec *pw4 = cat.find("XEN_PREFILL_W4");
  expect(pw4 && pw4->def == "a8" && pw4->flag == "--prefill-w4" && pw4->choices.size() == 3 && unused_reason(*pw4, "qwen35", dv).empty() &&
             !unused_reason(*pw4, "qwen38", dv).empty(),
         "prefill W4 numerics: a qwen35 choice (exact / dequant / a8), default a8 (qwen38-27b round 33), flag from the docs");
  const Spec *pfd = cat.find("XEN_PREFILL_DENSE");  // w8a8-row
  expect(pfd && pfd->def == "w8a8-row" && pfd->flag == "--prefill-dense" && pfd->choices.size() == 3 && pfd->choices[2] == "w8a8-row" &&
             unused_reason(*pfd, "qwen38", dv).empty() && !unused_reason(*pfd, "qwen35", dv).empty(),
         "prefill dense trunk: a qwen38 choice (w8a16 / w8a8 / w8a8-row), default w8a8-row (round 36), flag from the docs");
  const Spec *gw4 = cat.find("XEN_GDN_W4");
  expect(gw4 && gw4->def == "int8" && gw4->flag == "--gdn-w4" && gw4->choices.size() == 3 && gw4->choices[2] == "int4-qkvz" &&
             unused_reason(*gw4, "qwen38", dv).empty() && !unused_reason(*gw4, "qwen35", dv).empty(),
         "GDN projections: a qwen38 choice (int8 / int4 / int4-qkvz), default int8, flag from the docs");
  const Spec *a8k = cat.find("XEN_PREFILL_A8_KEEP");
  {
    Values on = dv, off = dv;  // a8 is the default, so the default shows it and exact hides it
    off["XEN_PREFILL_W4"] = "exact";
    expect(a8k && a8k->def.empty() && a8k->flag == "--prefill-a8-keep" && a8k->kind == Kind::Text && !unused_reason(*a8k, "qwen35", off).empty() &&
               unused_reason(*a8k, "qwen35", on).empty() && !unused_reason(*a8k, "qwen38", on).empty(),
           "W4A8 keep spec: a qwen35 text setting, default empty, flag from the docs, shown only with XEN_PREFILL_W4=a8");
  }
  {  // The W4A8 GEMM body, a qwen35 choice shown only with a8
    const Spec *ab = cat.find("XEN_PREFILL_A8_BODY");
    Values on = dv, off = dv;
    off["XEN_PREFILL_W4"] = "exact";
    expect(ab && ab->def == "auto" && ab->flag == "--prefill-a8-body" && ab->kind == Kind::Choice && ab->choices.size() == 3 &&
               ab->choices[2] == "direct" && unused_reason(*ab, "qwen35", on).empty() && !unused_reason(*ab, "qwen35", off).empty() &&
               !unused_reason(*ab, "qwen38", on).empty() && !unused_reason(*ab, "llama", on).empty() && !unused_reason(*ab, "flint", on).empty(),
           "W4A8 GEMM body: a qwen35 choice auto / slm / direct, default auto, flag from the docs, shown only with XEN_PREFILL_W4=a8");
  }
  {  // TP1 / TP2 and TP1's card (shown only with TP1); the wire is TP2's
    const Spec *tp = cat.find("XEN_TP"), *cd = cat.find("XEN_CARD"), *wr = cat.find("XEN_WIRE");
    Values one = dv;
    one["XEN_TP"] = "1";
    expect(tp && tp->def == "2" && tp->flag == "--tp" && tp->choices.size() == 2 && unused_reason(*tp, "qwen35", dv).empty() &&
               !unused_reason(*tp, "qwen38", dv).empty(),
           "TP: a qwen35 choice (2 / 1), default 2, flag from the docs");
    expect(cd && cd->def == "0" && cd->flag == "--card" && !unused_reason(*cd, "qwen35", dv).empty() && unused_reason(*cd, "qwen35", one).empty(),
           "TP1 card: default 0, flag from the docs, shown only with XEN_TP=1");
    expect(wr && unused_reason(*wr, "qwen35", dv).empty() && !unused_reason(*wr, "qwen35", one).empty(), "two-card wire: hidden with TP1");
    const Spec *pw = cat.find("XEN_PREFILL_WIRE");
    expect(pw && pw->def == "auto" && pw->flag == "--prefill-wire" && pw->choices.size() == 3 && unused_reason(*pw, "qwen35", dv).empty() &&
               !unused_reason(*pw, "qwen35", one).empty() && unused_reason(*pw, "qwen38", dv).empty() && unused_reason(*pw, "qwen38", one).empty(),
           "prefill two-card wire: auto / bf16 / fp32 for qwen35 and qwen38 (round 207), default auto, flag from the docs, hidden with qwen35 TP1");
    const Spec *kl = cat.find("XEN_KV_LEND"), *kk = cat.find("XEN_KV_LEND_KEEP");
    expect(kl && kk && kl->def == "on" && kl->flag == "--kv-lend" && kl->choices.size() == 2 && kk->def == "32768" && kk->flag == "--kv-lend-keep" &&
               unused_reason(*kl, "qwen38", dv).empty() && unused_reason(*kl, "qwen35", dv).empty() && unused_reason(*kk, "qwen35", dv).empty() &&
               !unused_reason(*kl, "llama", dv).empty() && !unused_reason(*kl, "flint", dv).empty(),
           "KV lending: a choice on / off (default on) + its keep (32768) for qwen38 and the dense 27B (qwen38-27b round 89), flags from the docs");
    const Spec *kp = cat.find("XEN_KV_LEND_PARK_S"), *kr = cat.find("XEN_KV_LEND_PRESSURE"), *kg = cat.find("XEN_KV_LEND_GRACE_S");
    expect(kp && kr && kg && kp->def == "300" && kp->flag == "--kv-lend-park-s" && kr->def == "kv" && kr->choices.size() == 3 &&
               kr->flag == "--kv-lend-pressure" && kg->def == "60" && kg->flag == "--kv-lend-grace-s" && unused_reason(*kr, "qwen38", dv).empty() &&
               !unused_reason(*kp, "qwen35", dv).empty(),
           "KV lending vs the prefix cache: park timeout 300 s, pressure kv / lru / idle, grace 60 s (qwen38), flags from the docs");
    {
      const Spec *gr = cat.find("XEN_PREFILL_GEMM_ROWS");
      Values es = dv;
      es["XEN_PREFILL_GEMM"] = "esimd";
      expect(gr && gr->def == "1024" && gr->flag == "--prefill-gemm-rows" && gr->choices.size() == 4 && unused_reason(*gr, "qwen35", dv).empty() &&
                 !unused_reason(*gr, "qwen35", es).empty() && !unused_reason(*gr, "llama", dv).empty() && !unused_reason(*gr, "flint", dv).empty() &&
                 !unused_reason(*gr, "qwen38", dv).empty(),
             "prefill GEMM row threshold (qwen38-27b round 89): 1024 / 512 / 256 / 128 for the dense 27B on oneDNN, default 1024, flag from the docs");
    }
    {
      Values esimd = dv, fixed = dv, off = dv;
      esimd["XEN_PREFILL_GEMM"] = "esimd", fixed["XEN_KV_POOL_TOKENS"] = "0", off["XEN_KV_LEND"] = "off";
      expect(kl && kk && !unused_reason(*kl, "qwen35", esimd).empty() && !unused_reason(*kl, "qwen35", fixed).empty() && unused_reason(*kl, "qwen35", off).empty() &&
                 !unused_reason(*kk, "qwen35", off).empty() && !unused_reason(*kp, "qwen35", dv).empty(),
             "KV lending on the dense 27B: only with the oneDNN prefill GEMMs and the shared pool; the keep only with it on; the prefix-cache settings stay qwen38's");
    }
    const Spec *yt = cat.find("XEN_YARN_TIERS");
    expect(yt && yt->def == "on" && yt->flag == "--yarn-tiers" && yt->choices.size() == 2 && unused_reason(*yt, "qwen35", dv).empty() &&
               unused_reason(*yt, "qwen38", dv).empty() && unused_reason(*yt, "llama", dv).empty() && !unused_reason(*yt, "flint", dv).empty(),
           "YaRN per request: a choice on / off for qwen35, qwen38 and llama (not flint), default on, flag from the docs");
    // qwen38 keeps its own pair: XEN_TP=1 for qwen35 leaves qwen38's TP alone
    const Spec *t8 = cat.find("XEN_Q38_TP"), *c8 = cat.find("XEN_Q38_CARD");
    Values one8 = dv;
    one8["XEN_Q38_TP"] = "1";
    expect(t8 && t8->def == "2" && t8->flag == "--tp" && unused_reason(*t8, "qwen38", one).empty() && !unused_reason(*t8, "qwen35", dv).empty() &&
               !unused_reason(*tp, "qwen38", one).empty(),
           "qwen38 TP: its own qwen38 setting (default 2), apart from qwen35's XEN_TP");
    expect(c8 && c8->def == "0" && c8->flag == "--card" && !unused_reason(*c8, "qwen38", one).empty() && unused_reason(*c8, "qwen38", one8).empty(),
           "qwen38 TP1 card: shown only with XEN_Q38_TP=1, not with qwen35's XEN_TP=1");
  }
  const Spec *lmh = cat.find("XEN_LM_HEAD");
  expect(lmh && lmh->def == "auto" && lmh->flag == "--lm-head" && lmh->choices.size() == 4 && lmh->choices[0] == "auto" && unused_reason(*lmh, "qwen35", dv).empty() &&
             !unused_reason(*lmh, "qwen38", dv).empty(),
         "lm_head format: a qwen35 choice (auto / int8 / int4 / int4-rescore, round 36), default auto, flag from the docs");
  {  // the two-stage head's tiles (shown only with int4-rescore) and the drafters' rows
    const Spec *lht = cat.find("XEN_LM_HEAD_TILES"), *dh = cat.find("XEN_DRAFT_HEAD");
    Values on = dv, off = dv;
    on["XEN_LM_HEAD"] = "int4-rescore";
    off["XEN_LM_HEAD"] = "int8";
    expect(lht && lht->def == "32" && lht->flag == "--lm-head-tiles" && lht->kind == Kind::Int && !unused_reason(*lht, "qwen35", off).empty() &&
               unused_reason(*lht, "qwen35", dv).empty() &&
               unused_reason(*lht, "qwen35", on).empty() && !unused_reason(*lht, "qwen38", on).empty() && !check_value(*lht, "33", "qwen35", "").empty() &&
               check_value(*lht, "16", "qwen35", "").empty(),
           "two-stage head tiles: a qwen35 int 1..32, default 32, flag from the docs, shown with XEN_LM_HEAD=int4-rescore or auto (round 36), hidden with int8");
    expect(dh && dh->def == "auto" && dh->flag == "--draft-head" && dh->choices.size() == 3 && unused_reason(*dh, "qwen35", dv).empty() &&
               !unused_reason(*dh, "qwen38", dv).empty(),
           "draft head rows: a qwen35 choice (auto / int8 / int4), default auto, flag from the docs");
  }
  {  // Image input on the checkpoints with a vision tower (qwen38, the 27B); its size settings only with it on
    const Spec *vi = cat.find("XEN_VISION"), *vp = cat.find("XEN_VISION_MAX_PIXELS"), *vn = cat.find("XEN_VISION_IMAGES");
    Values von = dv, voff = dv;
    von["XEN_VISION"] = "1", voff["XEN_VISION"] = "0";
    expect(vi && vp && vn && vi->def == "0" && vi->flag == "--vision" && vp->def == "1048576" && vp->flag == "--vision-max-pixels" && vn->def == "16" &&
               vn->flag == "--vision-images" && vi->scope == Scope::Vision && scope_name(vi->scope).find("qwen35") != std::string::npos &&
               unused_reason(*vi, "qwen38", voff).empty() && unused_reason(*vi, "qwen35", voff).empty() &&
               !unused_reason(*vi, "llama", voff).empty() && unused_reason(*vp, "qwen35", von).empty() &&
               !unused_reason(*vp, "qwen35", voff).empty() && !check_value(*vp, "1000", "qwen38", "").empty() && check_value(*vn, "64", "qwen38", "").empty(),
           "image input (round 167): qwen38 / qwen35 only (Scope::Vision, named so in the help), defaults 0 / 1048576 / 16 and flags from the docs, size settings shown with it on");
    const Spec *mu = cat.find("XEN_MEDIA_URLS"), *md = cat.find("XEN_MEDIA_DOMAINS"), *mt = cat.find("XEN_MEDIA_TIMEOUT");
    Values uoff = von;
    uoff["XEN_MEDIA_URLS"] = "0";
    expect(mu && md && mt && mu->def == "1" && mu->flag == "--media-urls" && md->def.empty() && md->flag == "--media-domains" && mt->def == "30" &&
               mt->flag == "--media-timeout" && unused_reason(*mu, "qwen35", von).empty() && !unused_reason(*mu, "qwen35", voff).empty() &&
               !unused_reason(*mu, "flint", von).empty() && unused_reason(*md, "qwen38", von).empty() && !unused_reason(*mt, "qwen38", uoff).empty() &&
               check_value(*md, "", "qwen38", "").empty() && !check_value(*mt, "0", "qwen38", "").empty(),
           "media URLs (round 168): qwen38 / qwen35 with image input on, defaults 1 / any host / 30 s and flags from the docs, host list and timeout only with links on");
    const Spec *vf = cat.find("XEN_VISION_FFMPEG"), *vm = cat.find("XEN_VIDEO_MAX_PIXELS"), *vr = cat.find("XEN_VIDEO_FPS");
    Values noff = von;
    noff["XEN_VISION_FFMPEG"] = "";
    expect(vf && vm && vr && vf->def == "ffmpeg" && vf->flag == "--vision-ffmpeg" && vm->def == "25165824" && vm->flag == "--video-max-pixels" &&
               vr->def == "2" && vr->flag == "--video-fps" && unused_reason(*vf, "qwen35", von).empty() && !unused_reason(*vf, "qwen35", voff).empty() &&
               unused_reason(*vr, "qwen38", von).empty() && !unused_reason(*vr, "qwen38", noff).empty() && !unused_reason(*vm, "flint", von).empty() &&
               !check_value(*vr, "0", "qwen38", "").empty() && check_value(*vm, "6291456", "qwen38", "").empty(),
           "video input (round 168): ffmpeg path / pixel budget / fps, defaults ffmpeg / 25165824 / 2 and flags from the docs, shown with image input on and a decoder set");
    // the decoder is a program, not a folder: empty = off, a missing one only warns (the server starts with video off),
    // every program the launcher offers passes
    bool w1 = false, w2 = false;
    const std::string m1 = check_value(*vf, "no-such-ffmpeg-xyz", "qwen38", "/", &w1), m2 = check_value(*vf, "/nonexistent/ffmpeg", "qwen38", "/", &w2);
    bool all_ok = true;
    const auto found = ffmpeg_candidates();
    for (const auto &f : found) all_ok = all_ok && check_value(*vf, f, "qwen38", "/").empty();
    expect(check_value(*vf, "", "qwen38", "/").empty() && w1 && w2 && m1.find("not on PATH") != std::string::npos &&
               m2.find("directory") == std::string::npos && m2.find("video stays off") != std::string::npos && all_ok,
           "video decoder: a program on PATH or a path (no folder check), a missing one a warning; " + std::to_string(found.size()) + " found on this host pass");
  }
  const Spec *pc = cat.find("XEN_PREFIX_CACHE"), *pcg = cat.find("XEN_PREFIX_CACHE_GIB"), *pcm = cat.find("XEN_PREFIX_MARK_MIN"),
             *pcl = cat.find("XEN_PREFIX_LIVE");
  expect(pc && pcg && pcm && pc->def == "1" && pc->flag == "--prefix-cache" && pcg->def == "8" && pcg->flag == "--prefix-cache-gib" && pcm->def == "1024" &&
             pcm->flag == "--prefix-mark-min" && pcl && pcl->def == "0" && pcl->flag == "--prefix-live" && unused_reason(*pc, "qwen35", dv).empty() && unused_reason(*pcg, "qwen35", dv).empty() &&
             unused_reason(*pc, "qwen38", dv).empty() && unused_reason(*pcg, "qwen38", dv).empty(),
         "prefix cache (qwen38-27b round 22; qwen38 since round 104): both archs, defaults 1 / 8 GiB / 1024 / live 0 and flags from the docs");
  dv["XEN_PREFIX_CACHE"] = "0";
  expect(pcg && pcm && pcl && !unused_reason(*pcg, "qwen35", dv).empty() && !unused_reason(*pcm, "qwen35", dv).empty() &&
             !unused_reason(*pcl, "qwen35", dv).empty(),
         "prefix cache memory, mark and live resumes hidden with the cache off");
  dv["XEN_PREFIX_CACHE"] = "1";
  dv["XEN_DRAFTER"] = "0";
  expect(vocab && !unused_reason(*vocab, "qwen35", dv).empty(), "draft vocabulary hidden with the drafter off");
  expect(dfl && !unused_reason(*dfl, "qwen35", dv).empty(), "DFlash drafter hidden with the drafter off");
  dv["XEN_QUEUE_POLICY"] = "sjf";
  expect(aging && unused_reason(*aging, "qwen38", dv).empty(), "SJF aging shown with policy sjf");
  dv["XEN_QUEUE_POLICY"] = "fifo";
  expect(aging && !unused_reason(*aging, "qwen38", dv).empty(), "SJF aging hidden with policy fifo");
  dv["XEN_KV_POOL_TOKENS"] = "100000";
  expect(reserve && !unused_reason(*reserve, "qwen35", dv).empty() && unused_reason(*reserve, "qwen38", dv).empty(), "reserve is unused by qwen35 without an auto pool");
  dv["XEN_STUB"] = "1";
  expect(chunk && slots && !unused_reason(*chunk, "qwen38", dv).empty() && unused_reason(*slots, "qwen38", dv).empty(), "the stub engine ignores prefill settings, keeps slots");
  // Every typed setting's default passes its own check.
  bool defaults_ok = true;
  std::string which;
  for (const auto &s : cat.specs) {
    if (s.kind == Kind::Path) continue;
    if (!check_value(s, expand_home(s.def), "", root).empty()) {
      defaults_ok = false;
      which += s.key + " ";
    }
  }
  expect(defaults_ok, "every default is a valid value " + which);
  // A setting the launcher has no type for still shows up.
  const Catalog c2 = build_catalog(script + "\nXEN_NEW_KNOB=${XEN_NEW_KNOB:-7}\n", "| `XEN_NEW_KNOB` | `7` | `--new-knob` | a new knob |\n");
  const Spec *nk = c2.find("XEN_NEW_KNOB");
  expect(nk && !nk->known && nk->group == "Other" && nk->def == "7" && nk->help == "a new knob", "an unknown setting appears under Other as text");
}

void test_env() {
  const std::string home = home_dir();
  EnvFile f = EnvFile::parse("# comment\nXEN_PORT=8001\nexport XEN_SLOTS=2 # two\nXEN_MODEL=$HOME/models/x\nXEN_SERVED_NAME='my model'\nOTHER=1\n");
  Values v = f.values();
  expect(v["XEN_PORT"] == "8001" && v["XEN_SLOTS"] == "2" && v["OTHER"] == "1", "env: plain, export and trailing-comment lines");
  expect(v["XEN_MODEL"] == home + "/models/x", "env: $HOME expands");
  expect(v["XEN_SERVED_NAME"] == "my model", "env: single quotes");
  f.set("XEN_PORT", "9000");
  f.set("XEN_HOST", "a b");
  f.erase("XEN_SLOTS");
  const std::string t = f.text();
  expect(has(t, "# comment\nXEN_PORT=9000\n") && has(t, "XEN_HOST='a b'\n") && !has(t, "XEN_SLOTS") && has(t, "OTHER=1"),
         "env: set in place, append quoted, erase, keep comments and other lines");
  expect(env_unquote(env_quote("it's")) == "it's", "env: a quote inside a value round-trips");
  Catalog cat = build_catalog("XEN_PORT=${XEN_PORT:-8000}\nXEN_SLOTS=${XEN_SLOTS:-4}\n", "");
  EnvFile g = EnvFile::parse("XEN_PORT=8001\n");
  Values d = defaults_of(cat), now = d;
  now["XEN_SLOTS"] = "2";
  g.apply(now, d, cat);
  expect(g.text() == "XEN_SLOTS=2\n", "env: apply keeps changed settings and drops the ones back at their default");
}

void test_validate(const std::string &root) {
  const Catalog cat = load_catalog(root);
  auto chk = [&](const char *k, const std::string &v, const char *arch) { return check_value(*cat.find(k), v, arch, root); };
  expect(chk("XEN_SLOTS", "64", "qwen38").empty() && chk("XEN_SLOTS", "65", "qwen38").empty() == false && chk("XEN_SLOTS", "16", "qwen35").empty() && !chk("XEN_SLOTS", "17", "qwen35").empty() && !chk("XEN_SLOTS", "9", "llama").empty() && chk("XEN_SLOTS", "8", "flint").empty(),
         "slots: qwen38 1..64 (engine round 276; 17+ needs four cards: a cross-setting check), qwen35 1..16 (qwen38-27b round 71), llama/flint 1..8");
  expect(!chk("XEN_PREFILL_CHUNK", "6", "").empty() && chk("XEN_PREFILL_CHUNK", "512", "").empty(), "prefill chunk: a multiple of 4");
  expect(!chk("XEN_DRAFT_VOCAB", "100", "qwen35").empty() && chk("XEN_DRAFT_VOCAB", "0", "qwen35").empty(), "draft vocabulary: a multiple of 32");
  expect(!chk("XEN_DRAFT_VOCAB", "32", "qwen38").empty() && chk("XEN_DRAFT_VOCAB", "98304", "qwen38").empty(), "draft vocabulary: a multiple of 64 for qwen38 (round 106)");
  expect(chk("XEN_DRAFTER_DEFER", "0", "qwen35").empty() && chk("XEN_DRAFTER_DEFER", "64", "qwen35").empty() && !chk("XEN_DRAFTER_DEFER", "4", "qwen35").empty() &&
             !chk("XEN_DRAFTER_DEFER", "1024", "qwen35").empty(),
         "drafter defer: 0 or 8..512 rows");
  expect(chk("XEN_DRAFTER_CUT", "0", "qwen35").empty() && chk("XEN_DRAFTER_CUT", "0.02", "qwen35").empty() && !chk("XEN_DRAFTER_CUT", "1", "qwen35").empty() &&
             !chk("XEN_DRAFTER_CUT", "-0.1", "qwen35").empty(),
         "drafter cut: a probability below 1");
  {
    Values u = defaults_of(cat);
    u["XEN_DRAFTER"] = "1";
    u["XEN_DRAFTER_MODEL"] = "";
    const bool off = !unused_reason(*cat.find("XEN_DRAFTER_CUT"), "qwen35", u).empty() && !unused_reason(*cat.find("XEN_DRAFTER_MTP_FROM"), "qwen35", u).empty() &&
                     !unused_reason(*cat.find("XEN_DRAFTER_INT4"), "qwen35", u).empty();
    const bool defer_mtp = unused_reason(*cat.find("XEN_DRAFTER_DEFER"), "qwen35", u).empty();  // MTP defers too
    u["XEN_DRAFTER_MODEL"] = "/m/dflash";
    const bool on = unused_reason(*cat.find("XEN_DRAFTER_DEFER"), "qwen35", u).empty() && unused_reason(*cat.find("XEN_DRAFTER_CUT"), "qwen35", u).empty() &&
                    unused_reason(*cat.find("XEN_DRAFTER_MTP_FROM"), "qwen35", u).empty() && unused_reason(*cat.find("XEN_DRAFTER_INT4"), "qwen35", u).empty();
    u["XEN_DRAFTER"] = "0";
    const bool none = !unused_reason(*cat.find("XEN_DRAFTER_DEFER"), "qwen35", u).empty();
    expect(off && on && defer_mtp && none, "drafter cut / MTP-from / int4: shown only with a DFlash drafter model; defer with any drafter");
  }
  expect(chk("XEN_DRAFTER_MTP_FROM", "0", "qwen35").empty() && chk("XEN_DRAFTER_MTP_FROM", "3", "qwen35").empty() && chk("XEN_DRAFTER_MTP_FROM", "9", "qwen35").empty() &&
             !chk("XEN_DRAFTER_MTP_FROM", "17", "qwen35").empty() &&
             !chk("XEN_DRAFTER_MTP_FROM", "-1", "qwen35").empty(),
         "drafter MTP-from: 0..16 sequences");
  {  // (audit C1)
    const Spec *as = cat.find("XEN_ATTN_SELECT");
    expect(as && as->def == "0" && chk("XEN_ATTN_SELECT", "1024", "qwen35").empty() && !chk("XEN_ATTN_SELECT", "5000", "qwen35").empty() &&
               !unused_reason(*as, "qwen38", Values{}).empty(),
           "page selection: 0 (dense) by default, up to 4096 pages, qwen35 only");
    const Spec *ps = cat.find("XEN_PREFILL_SELECT");
    expect(ps && ps->def == "0" && chk("XEN_PREFILL_SELECT", "512", "qwen35").empty() && !chk("XEN_PREFILL_SELECT", "5000", "qwen35").empty() &&
               !unused_reason(*ps, "qwen38", Values{}).empty() && !unused_reason(*ps, "llama", Values{}).empty(),
           "prefill page selection: 0 (dense) by default, up to 4096 pages, qwen35 only");
    const Spec *pm = cat.find("XEN_PREFILL_MERGE");
    expect(pm && pm->def == "16" && chk("XEN_PREFILL_MERGE", "8", "qwen35").empty() && !chk("XEN_PREFILL_MERGE", "0", "qwen35").empty() &&
               !chk("XEN_PREFILL_MERGE", "17", "qwen35").empty() && !unused_reason(*pm, "qwen38", Values{}).empty() && !unused_reason(*pm, "llama", Values{}).empty(),
           "prompts per prefill call: 16 by default, 1 (a call per prompt) to 16, qwen35 only");
    const Spec *sh = cat.find("XEN_SELECT_HEADS");
    expect(sh && sh->def == "auto" && chk("XEN_SELECT_HEADS", "same", "qwen35").empty() && !chk("XEN_SELECT_HEADS", "table", "qwen35").empty() &&
               !unused_reason(*sh, "qwen38", Values{}).empty(),
           "page sharing among heads: auto by default, same allowed, anything else refused, qwen35 only");
  }
  {
    bool warn = false;
    const std::string m = check_value(*cat.find("XEN_DRAFTER_MODEL"), "docs", "qwen35", root, &warn);
    expect(chk("XEN_DRAFTER_MODEL", "", "qwen35").empty() && !m.empty() && warn && !chk("XEN_DRAFTER_MODEL", "no-such-dir", "qwen35").empty(),
           "DFlash drafter: empty = the MTP layer; a directory without a DFlash2DraftModel config warns; a missing one fails");
  }
  expect(chk("XEN_KV_POOL_TOKENS", "auto", "qwen35").empty() && !chk("XEN_KV_POOL_TOKENS", "lots", "qwen35").empty(), "KV pool: auto or a number");
  expect(chk("XEN_KV_POOL_TOKENS", "0", "qwen35").find("removed") != std::string::npos && chk("XEN_KV_POOL_TOKENS", "262144", "qwen35").empty(),
         "KV pool: 0 (the fixed per-slot layout) is refused with a reason");
  expect(!chk("XEN_KV", "fp8", "").empty() && !chk("XEN_PORT", "0", "").empty() && !chk("XEN_PORT", "80x", "").empty(), "choices and integers are checked");
  expect(chk("XEN_KV", "int8-v4", "qwen38").empty() && chk("XEN_KV", "int8-v4", "qwen35").empty() &&
             !chk("XEN_KV", "int8-v4", "flint").empty() && !chk("XEN_KV", "int8-v4", "llama").empty() && chk("XEN_KV", "int8", "qwen35").empty(),
         "KV int8-v4 (int8 K + int4 V, rounds 205 / 226): offered for qwen38 and the 27B, refused with a reason elsewhere");
  expect(chk("XEN_WIRE", "int8", "qwen35").empty() && chk("XEN_WIRE", "bf16", "flint").empty() && !chk("XEN_WIRE", "int8", "flint").empty() &&
             !chk("XEN_WIRE", "int8", "llama").empty() && !chk("XEN_WIRE", "int4", "qwen35").empty(),
         "wire int8 (qwen38-27b round 89): offered for the dense 27B, refused with a reason for llama / flint");
  expect(!chk("XEN_SERVED_NAME", "", "").empty(), "the served name must not be empty");
  const Spec &ctx = *cat.find("XEN_CONTEXT");
  expect(check_value(ctx, "1048576", "qwen35", root, nullptr, 1048576).empty() && check_value(ctx, "1048500", "qwen35", root, nullptr, 1048576).empty() &&
             !check_value(ctx, "1048577", "qwen35", root, nullptr, 1048576).empty() && check_value(ctx, "1048576", "qwen38", root).empty(),
         "max context: up to the kernels' 1,048,576 (rounded up to 64)");
  {  // past the trained positions: a warning naming the automatic YaRN factor (flint: none needed); at or under: nothing
    auto ctx_issue = [&](const char *c, const char *arch, int64_t trained) {
      Values vv = defaults_of(cat);
      vv["XEN_CONTEXT"] = c;
      for (const auto &i : validate(cat, vv, arch, root, 1048576, 0, trained))
        if (i.key == "XEN_CONTEXT") return std::string(i.error ? "E:" : "W:") + i.msg;
      return std::string();
    };
    const std::string q2 = ctx_issue("524288", "qwen35", 262144), q1 = ctx_issue("262144", "qwen35", 262144), q3 = ctx_issue("333000", "qwen38", 262144),
                      fl = ctx_issue("524288", "flint", 262144), nb = ctx_issue("1048576", "llama", 524288);
    expect(q2.rfind("W:", 0) == 0 && q2.find("YaRN factor 2.00") != std::string::npos && q1.empty() && q3.find("YaRN factor 1.28") != std::string::npos &&
               fl.rfind("W:", 0) == 0 && fl.find("no YaRN needed") != std::string::npos && nb.find("YaRN factor 2.00") != std::string::npos,
           "max context past the trained positions: a warning with the automatic YaRN factor (2.00 / 1.28; Nanbeige 524,288 trained); flint none");
    expect(yarn_factor_for(262144, 262144) == 1.0 && yarn_factor_for(333000, 262144) == 1.28 && yarn_factor_for(1048576, 262144) == 4.0,
           "launcher YaRN factor = the load's rule (qsa_yarn_for)");
  }
  Values v = defaults_of(cat);
  v["XEN_KV_POOL_TOKENS"] = "1000";
  v["XEN_CONTEXT"] = "32768";
  v["XEN_PORT"] = "80";
  bool pool = false, port_warn = false;
  for (const auto &i : validate(cat, v, "qwen35", root)) {
    if (i.key == "XEN_KV_POOL_TOKENS" && i.error) pool = true;
    if (i.key == "XEN_PORT" && !i.error) port_warn = true;
  }
  expect(pool, "a KV pool smaller than one context is an error");
  expect(port_warn, "a port below 1024 is a warning");
  {  // The step rows (the server's checks): defaults clean; bad lists, 15 drafts at 8-row segments
     // the default is 16, so the 8-row segments are pinned here), 2 x 16 rows, 16-row segments in 8-row steps, 16 slots in
     // 8-row steps are errors; 15,7,4,3 at 16-row segments is clean
    auto errs = [&](const std::map<std::string, std::string> &set) {
      Values w = defaults_of(cat);
      for (const auto &[k, x] : set) w[k] = x;
      std::string keys;
      for (const auto &i : validate(cat, w, "qwen35", root))
        if (i.error && (i.key == "XEN_STEP_ROWS" || i.key == "XEN_SEG_ROWS" || i.key == "XEN_DRAFT_DEPTHS")) keys += i.key + ";";
      return keys;
    };
    const bool ok = errs({}).empty() && errs({{"XEN_SEG_ROWS", "16"}, {"XEN_DRAFT_DEPTHS", "15,7,4,3"}}).empty() && errs({{"XEN_DRAFT_DEPTHS", "7,3"}}).empty() &&
                    errs({{"XEN_SEG_ROWS", "8"}, {"XEN_DRAFT_DEPTHS", "15"}}) == "XEN_DRAFT_DEPTHS;" && errs({{"XEN_DRAFT_DEPTHS", "7,x"}}) == "XEN_DRAFT_DEPTHS;" &&
                    errs({{"XEN_SEG_ROWS", "16"}, {"XEN_DRAFT_DEPTHS", "15,15"}}).empty() &&
                    errs({{"XEN_SEG_ROWS", "16"}, {"XEN_DRAFT_DEPTHS", "15,15,15"}}) == "XEN_DRAFT_DEPTHS;" &&
                    errs({{"XEN_SEG_ROWS", "16"}, {"XEN_STEP_ROWS", "16"}, {"XEN_DRAFT_DEPTHS", "15,15"}}) == "XEN_DRAFT_DEPTHS;" &&
                    errs({{"XEN_SEG_ROWS", "16"}, {"XEN_STEP_ROWS", "8"}, {"XEN_SLOTS", "4"}}) == "XEN_SEG_ROWS;" &&
                    errs({{"XEN_SEG_ROWS", "8"}, {"XEN_STEP_ROWS", "8"}, {"XEN_SLOTS", "16"}}) == "XEN_STEP_ROWS;";  // 8 pinned - the default 16-row segments would add their own error first
    expect(ok, "step rows: XEN_STEP_ROWS / XEN_SEG_ROWS / XEN_DRAFT_DEPTHS checked as the server checks them");
    // Flint's DFlash drafter in its 8-row steps: 7,3 clean, 7,7 (2 x 8 rows) an error
    auto errs_f = [&](const std::map<std::string, std::string> &set) {
      Values w = defaults_of(cat);
      w["XEN_SLOTS"] = "8";
      for (const auto &[k, x] : set) w[k] = x;
      std::string keys;
      for (const auto &i : validate(cat, w, "flint", root))
        if (i.error && (i.key == "XEN_STEP_ROWS" || i.key == "XEN_SEG_ROWS" || i.key == "XEN_DRAFT_DEPTHS")) keys += i.key + ";";
      return keys;
    };
    expect(errs_f({}).empty() && errs_f({{"XEN_DRAFT_DEPTHS", "7,3,1"}}).empty() && errs_f({{"XEN_DRAFT_DEPTHS", "7,7"}}) == "XEN_DRAFT_DEPTHS;",
           "flint: draft depths checked against its 8-row steps");
  }
  {  // XEN_HOT past the model's total: a warning (the load caps it), not an error
    Values v2 = defaults_of(cat);
    v2["XEN_HOT"] = "999999";
    bool w38 = false;
    for (const auto &i : validate(cat, v2, "qwen38", root, 0, 24576))
      if (i.key == "XEN_HOT" && !i.error) w38 = true;
    v2["XEN_HOT"] = "10000";  // a total over all layers: within the model's count
    bool ok = true;
    for (const auto &i : validate(cat, v2, "qwen38", root, 0, 24576))
      if (i.key == "XEN_HOT") ok = false;
    expect(w38 && ok, "XEN_HOT: a count past the model's total warns, a count within it does not");
  }
  {  // XEN_HOT: the dynamic description (expert count + fit) and the per-expert bytes
    ModelInfo m38; m38.arch = "qwen38"; m38.experts = 512; m38.layers = 48; m38.moe_layers = 48; m38.hidden = 2560; m38.moe_inter = 640; m38.bits = 4;
    ModelInfo m35; m35.arch = "qwen35";
    expect(max_hot_of(m38) == 24576, "max_hot_of: the routed experts over all MoE layers (qwen38 48 x 512)");
    expect(bytes_per_expert_card(m38) == 1228800.0, "bytes_per_expert_card: qwen38 = 3 x 2560 x 640 x 0.5 / 2");
    expect(has(hot_description(m38), "24,576 routed experts in all (512 per layer x 48 MoE layers)") && has(hot_description(m38), "split over both cards"),
           "hot_description: qwen38 states the total expert count and the two-card split");
    MemPlanView pv;
    const bool parsed = parse_plan(
        "{\"arch\":\"qwen38\",\"exact\":true,\"cards\":2,\"card_gib\":31.850,\"vram_used_gib\":31.804,\"weights_gib\":4.961,\"kv_gib\":6.797,"
        "\"state_gib\":0.275,\"scratch_gib\":3.860,\"hot_gib\":15.911,\"hot_experts\":4284,\"experts_total\":24576,\"moe_layers\":48,"
        "\"kv_tokens\":1048576,\"context\":262144,\"slots\":4,\"cold_gib\":58.030,\"embed_gib\":1.182,\"pinned_gib\":59.212,\"prefix_gib\":8.000,"
        "\"host_total_gib\":61.307,\"host_available_gib\":57.819}",
        pv);
    expect(parsed && pv.hot_experts == 4284 && pv.kv_tokens == 1048576 && pv.slots == 4 && pv.pinned_gib > 59.2 && pv.exact,
           "parse_plan: the server's --plan JSON (hot experts, KV tokens, pinned host RAM)");
    expect(has(hot_description(m38, &pv), "With these settings 4,284 experts fit"), "hot_description: the plan's fit when there is one");
    Values pvv = defaults_of(cat);
    auto issue_of = [&](const std::vector<Issue> &is, const char *key, bool err) {
      for (const auto &i : is)
        if (i.key == key && i.error == err) return true;
      return false;
    };
    // 4 slots x 262K on a 61 GiB host: 59.2 GiB pinned > 57.8 free - 4 kept -> an error (the OOM kill the server had)
    expect(issue_of(plan_issues(pv, pvv, false), "XEN_HOT", true), "plan_issues: pinned host RAM past the free RAM is an error");
    expect(issue_of(plan_issues(pv, pvv, true), "XEN_HOT", false) && !issue_of(plan_issues(pv, pvv, true), "XEN_HOT", true),
           "plan_issues: only a warning while a server runs (its memory is counted in the free RAM)");
    MemPlanView pv2 = pv;
    pv2.cold_gib = 47.1, pv2.pinned_gib = 48.3;  // 1 slot: fits, but not with 8 GiB of prefix cache
    pvv["XEN_PREFIX_CACHE"] = "1";
    const auto is2 = plan_issues(pv2, pvv, false);
    expect(!issue_of(is2, "XEN_HOT", true) && issue_of(is2, "XEN_PREFIX_CACHE_GIB", false), "plan_issues: a prefix cache past the RAM left is a warning (the server cuts it)");
    MemPlanView pv3 = pv2;
    pv3.hot_experts = 5754, pv3.experts_total = 12096;  // the plan reports the count the load cuts a pinned one to
    pvv["XEN_HOT"] = "6000";
    expect(issue_of(plan_issues(pv3, pvv, false), "XEN_HOT", false), "plan_issues: a pinned hot count past the fit warns (the load cuts it)");
    pvv["XEN_HOT"] = "5000";
    expect(!issue_of(plan_issues(pv3, pvv, false), "XEN_HOT", false), "plan_issues: a pinned hot count within the fit is fine");
    expect(hot_description(m35).empty(), "hot_description: empty for the dense archs");
    const int64_t fit38 = hot_fit(m38);
    expect(fit38 == 0 || (fit38 > 0 && fit38 <= 24576), "hot_fit: qwen38 within [0, max] (0 when no known card)");
  }
  bool scoped = true, pool38 = false;
  v["XEN_W4_ITEMS"] = "0";  // below its minimum: an error for qwen35 only
  for (const auto &i : validate(cat, v, "qwen38", root)) {
    if (i.key == "XEN_W4_ITEMS") scoped = false;
    if (i.key == "XEN_KV_POOL_TOKENS" && i.error) pool38 = true;
  }
  expect(pool38, "qwen38 too (round 102): a KV pool smaller than one context is an error");
  expect(scoped, "settings of the other arch are not validated");
}

void test_keys() {
  KeyDecoder d;
  d.feed("\x1b[A\x1b[6~\x1bOH\x1b[<0;12;5M\x1b[<65;1;1M\x1b[200~pa\nste\x1b[201~\xc3\xa9q\r\t\x7f\x03\x1b[Z");
  std::vector<Key> got;
  Key k;
  while (d.next(k, false)) got.push_back(k);
  bool ok = got.size() == 13;
  if (ok) {
    ok = got[0].type == Key::Up && got[1].type == Key::PgDn && got[2].type == Key::Home && got[3].type == Key::Mouse && got[3].mx == 11 &&
         got[3].my == 4 && got[3].button == 0 && got[3].press && got[4].button == 65 && got[5].type == Key::Paste && got[5].text == "pa\nste" &&
         got[6].type == Key::Char && got[6].ch == 0xe9 && got[7].is('q') && got[8].type == Key::Enter && got[9].type == Key::Tab &&
         got[10].type == Key::Backspace && got[11].type == Key::Ctrl && got[11].ch == 'c' && got[12].type == Key::BackTab;
  }
  expect(ok, "keys: arrows, CSI ~, SS3, SGR mouse, paste, UTF-8, controls (" + std::to_string(got.size()) + " keys)");
  KeyDecoder e;
  e.feed("\x1b");
  expect(!e.next(k, true) && e.next(k, false) && k.type == Key::Esc, "keys: a lone ESC waits for more bytes, then reads as Esc");
  KeyDecoder f;
  f.feed("\x1b[1");
  bool waited = !f.next(k, true);
  f.feed(";5A");
  expect(waited && f.next(k, true) && k.type == Key::Up, "keys: a sequence cut between reads is joined");
}

void test_text() {
  expect(text_width("a\xe2\x94\x80\xe4\xb8\xad") == 4, "widths: ASCII 1, box drawing 1, CJK 2");
  expect(ellipsize("abcdefgh", 5) == "abcd\xe2\x80\xa6" && ellipsize("abc", 5) == "abc", "ellipsize");
  const auto w = wrap("one two three four", 10);
  expect(w.size() == 2 && w[0] == "one two" && w[1] == "three four", "wrap at word boundaries");
  expect(wrap("abcdefghij", 4).size() == 3, "wrap splits a word longer than the line");
  {  // the help text layout paragraphs, values in one column, labels, bullets, code spans
    expect(help_label("Takes: more") == 6 && help_label("Gives (int8-v4): x") == 16 && help_label("KV context: x") == 11 && help_label("KV context (qwen35 / flint): y") == 28,
           "help_label: a label word, an optional (case), the colon");
    expect(help_label("Given: x") == 0 && help_label("Gives = x") == 0 && help_label("With KV lending: tokens") == 0 && help_label("") == 0,
           "help_label: only the listed words followed by a colon");
    const auto h = help_lines("What it does.\n\n`int8`  half the memory\n`int8-v4`  keys int8, values int4, a quarter less\n\nGives (int8): twice the tokens of bf16\n- one\n- two", 30);
    std::string all;
    for (const auto &l : h) all += l.text + "|";
    expect(h.size() == 10 && h[0].text == "What it does." && h[1].text.empty(), "help_lines: one paragraph per line break, an empty one is a blank line");
    expect(h[2].text == "`int8`     half the memory" && h[3].text == "`int8-v4`  keys int8, values" && h[4].text == "         int4, a quarter less",
           "help_lines: a run of values shares one text column (as drawn: the backticks take no cell), wrapped lines hang under it (" + all + ")");
    expect(h[6].label == 13 && h[6].text == "Gives (int8): twice the tokens" && h[7].text == "  of bf16" && h[7].label == 0, "help_lines: a labelled line hangs by two, the label's bytes marked");
    expect(h[8].text == "\xe2\x80\xa2 one" && h[9].text == "\xe2\x80\xa2 two", "help_lines: bullets");
    const auto pc = help_lines("a step is 3.5-5.9 % faster with rows", 22);
    expect(pc.size() == 2 && pc[0].text == "a step is 3.5-5.9 %" && pc[1].text == "faster with rows", "help_lines: a percent sign stays with its number");
    const auto cs = help_lines("see the file `docs server env` now", 22);
    expect(cs.size() == 2 && cs[0].text == "see the file `docs`" && cs[1].text == "`server env` now", "help_lines: a code span cut by a line break is closed and reopened");
    const auto lv = help_lines("`25165824`  the model's default", 40);
    expect(lv.size() == 1 && lv[0].text == "`25165824`  the model's default", "help_lines: a single value keeps two spaces before its text");
  }
  Canvas c;
  c.resize(6, 1);
  c.put(0, 0, "a\xe4\xb8\xad" "bcdef");
  expect(c.row(0) == "a\xe4\xb8\xad" "bcd", "canvas: a wide char takes two cells, clipped at the edge");
  const auto m = parse_metrics("# HELP x\nvllm:num_requests_running 2\nfoo{a=\"1\"} 1.5\nfoo{a=\"2\"} 2\n");
  expect(m.at("vllm:num_requests_running") == 2 && m.at("foo") == 3.5, "metrics: labels summed");
}

void test_exit_reason() {
  // the kernel log lines of an OOM kill (journalctl -k -o short-iso), one unrelated kill before it
  const std::string log =
      "2026-11-09T00:20:01+0000 host kernel: Out of memory: Killed process 777 (python) total-vm:10kB, anon-rss:1048576kB, file-rss:0kB\n"
      "2026-11-09T00:33:50+0000 host kernel: Out of memory: Killed process 1604264 (xenithium-serve) total-vm:75069628kB, "
      "anon-rss:49599232kB, file-rss:2876kB, shmem-rss:0kB, UID:1000 pgtables:120160kB oom_score_adj:0\n"
      "2026-11-09T00:33:53+0000 host kernel: oom_reaper: reaped process 1604264 (xenithium-serve), now anon-rss:0kB\n";
  double gib = 0;
  std::string when;
  expect(find_oom_kill(log, 1604264, &gib, &when) && when == "00:33:50" && gib > 47.2 && gib < 47.4, "exit: the OOM kill of the pid, its time and anon RSS");
  expect(!find_oom_kill(log, 160426, nullptr, nullptr), "exit: a pid prefix is not a match");
  const std::string dm = "[Mon Nov  9 00:33:50 2026] Out of memory: Killed process 42 (xenithium-serve) total-vm:1kB, anon-rss:2097152kB\n";
  expect(find_oom_kill(dm, 42, &gib, &when) && when == "00:33:50" && gib == 2.0, "exit: dmesg -T format");
  expect(describe_exit(139).find("SIGSEGV") != std::string::npos && describe_exit(137).find("SIGKILL") != std::string::npos &&
             describe_exit(1).find("status 1") != std::string::npos && describe_exit(134).find("SIGABRT") != std::string::npos,
         "exit: statuses in words");
  expect(exit_reason(12345, 139, 0, false) == describe_exit(139), "exit: a crash needs no kernel log");
}

void test_app(const std::string &root) {
  const char *tmp = std::getenv("TMPDIR");
  std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/xen-launch-XXXXXX";
  const char *tdir = mkdtemp(tmpl.data());
  if (!tdir) {
    expect(false, "mkdtemp");
    return;
  }
  const std::string t = tdir;
  // Fake checkpoints, an MTP drafter, a DFlash 2 drafter and DSpine drafters under both names.
  const std::string models = t + "/models";
  mkdir(models.c_str(), 0755);
  for (const auto &[name, type] : std::vector<std::pair<std::string, std::string>>{
           {"moe-a", "qwen4_exp"}, {"dense-b", "qwen3_5"}, {"other-c", "gemma3"}, {"moe-a-mtp", "qwen4_exp"},
           {"moe-a-dflash2", "DFlash2DraftModel"}, {"dense-b-dspine", "DFlashDraftModel"}, {"dense-b-dspine2", "DSpineDraftModel"}}) {
    mkdir((models + "/" + name).c_str(), 0755);
    write(models + "/" + name + "/config.json", "{\"model_type\": \"" + type + "\"}");
  }
  write(models + "/moe-a-mtp/mtp-dense.safetensors", "x");
  const auto ms = scan_models(models);
  int q38 = 0, q35 = 0, none = 0, drafters = 0, dflashes = 0;
  for (const auto &m : ms) {
    if (m.drafter) ++drafters;
    else if (m.dflash) ++dflashes;
    else if (m.arch == "qwen38") ++q38;
    else if (m.arch == "qwen35") ++q35;
    else if (m.arch.empty()) ++none;
  }
  expect(q38 == 1 && q35 == 1 && none == 1 && drafters == 1 && dflashes == 3, "model scan: arch from the registry, MTP and DFlash drafters told apart");
  const auto dr = drafters_for(models + "/moe-a");
  expect(dr.size() == 1 && dr[0] == models + "/moe-a-mtp", "drafters: <target>-mtp* siblings");
  const auto df = dflash_drafters(models);
  bool df2 = false, dsp = false, dsp2 = false;
  for (const auto &d : df)
    df2 |= d.path == models + "/moe-a-dflash2" && d.bytes > 0, dsp |= d.path == models + "/dense-b-dspine" && d.bytes > 0,
        dsp2 |= d.path == models + "/dense-b-dspine2" && d.bytes > 0;
  expect(df.size() == 3 && df2 && dsp && dsp2,
         "dflash drafters: DFlash2DraftModel, DFlashDraftModel and DSpineDraftModel (DSpine) checkpoints under the models directory");
  write(models + "/dense-b/config.json", "{\"model_type\": \"qwen3_5\", \"text_config\": {\"max_position_embeddings\": 262144}}");
  const ModelInfo db = probe_model(models + "/dense-b");
  expect(db.max_positions == 262144 && trained_positions_of(db) == 262144 && max_context_of(db) == 1048576,
         "model: max_position_embeddings read from text_config (trained positions; max context the kernels' 1,048,576)");

  const std::string env = t + "/server.env";
  write(env, "# mine\nXEN_MODEL=" + models + "/moe-a\nXEN_PORT=8123\n");
  AppConfig cfg;
  cfg.root = root;
  cfg.env_path = env;
  cfg.model_envs_dir = t + "/model-envs";
  cfg.models_dir = models;
  cfg.state_dir = t + "/state";
  cfg.profiles_dir = t + "/profiles";
  cfg.experts_dir = t + "/experts";
  cfg.live = false;
  App app(cfg);
  Canvas c;
  c.resize(120, 36);
  auto run = [&](std::initializer_list<Key> ks) {
    for (const Key &k : ks) {
      app.handle(k);
      app.draw(c);
    }
    app.draw(c);
    return screen(c);
  };
  std::string s = run({});
  expect(has(s, "xenithium") && has(s, "Settings") && has(s, "Logs") && has(s, "moe-a") && has(s, "qwen38"), "first frame: header, model card, settings");
  expect(has(s, "recorded from this run on"), "the model card says a MoE model's ranking is recorded from this run on (none yet)");
  expect(has(s, "8,123") || has(s, "8123"), "the file's port is shown");
  expect(has(app.preview(), "--port 8123") && has(app.preview(), "xenithium-server"), "preview is the script's dry run with the file's values");
  expect(!has(s, "W4 k-split target") && has(run({Key::of(Key::End)}), "this model does not use"), "qwen35-only settings hidden for a qwen38 model");  // the KV pool applies to both
  run({Key::of(Key::Home)});
  expect(!app.dirty(), "nothing unsaved after open");
  expect(app.values().at("XEN_SERVED_NAME") == "moe-a", "the served name defaults to the checkpoint's folder name on open");
  expect(has(app.preview(), "--served-model-name moe-a"), "the preview carries the folder name as the served model");

  // A search finds a setting that waits on another one (dimmed, n/a), not one of the other arch.
  s = run({ch('/'), ch('h'), ch('t'), ch('t'), ch('p'), ch('s')});
  expect(app.values().at("XEN_VISION") == "0" && has(s, "Images by http(s) link") && has(s, "n/a"),
         "filter: / https finds Images by http(s) link while Image input is off (dimmed, n/a)");
  run({Key::of(Key::Esc)});
  s = run({ch('/'), ch('k'), ch('-'), ch('s'), ch('p'), ch('l'), ch('i'), ch('t')});
  expect(!has(s, "W4 k-split target"), "filter: a qwen35-only setting stays hidden for a qwen38 model");
  s = run({Key::of(Key::Esc)});
  expect(!has(s, "find: "), "Esc clears the filter");

  // Find Slots, pick 2 from its popup.
  s = run({ch('/'), ch('s'), ch('l'), ch('o'), ch('t'), Key::of(Key::Enter)});
  expect(app.selected_key() == "XEN_SLOTS", "filter: / slots selects Slots");
  s = run({Key::of(Key::Enter)});
  expect(app.popup_open() && has(s, "Slots"), "Enter opens the choices popup");
  // Options 1, 2, 4, 8; the current (4) is selected: Up once picks 2.
  s = run({Key::of(Key::Up), Key::of(Key::Enter)});
  expect(!app.popup_open() && app.values().at("XEN_SLOTS") == "2", "picking 2 sets XEN_SLOTS=2");
  expect(app.dirty() && has(s, "1 unsaved"), "the change is marked unsaved");
  expect(has(app.preview(), "--slots 2"), "the preview follows the change");
  // Clear the filter, then cycle KV dtype with Right.
  s = run({Key::of(Key::Esc), ch('/'), ch('k'), ch('v'), ch(' '), ch('c'), ch('a'), Key::of(Key::Enter), Key::of(Key::Right)});
  expect(app.selected_key() == "XEN_KV" && app.values().at("XEN_KV") == "bf16", "Right cycles a choice (int8 -> bf16)");
  s = run({ch('u')});
  expect(app.values().at("XEN_KV") == "int8", "u undoes");
  // Typed value with validation: a slot count of 65 is refused (qwen38 1..64).
  s = run({Key::of(Key::Esc), ch('/'), ch('s'), ch('l'), ch('o'), ch('t'), Key::of(Key::Enter), ch('e'), Key::ctrl('u'), ch('6'), ch('5'), Key::of(Key::Enter)});
  expect(has(s, "between 1 and 64") && app.values().at("XEN_SLOTS") == "2", "a slot count of 65 is refused with the reason");
  s = run({Key::of(Key::Esc)});
  // Save: server.env names the model's own settings file; the settings (its own XEN_PORT moved over) go there, only the
  // non-default values, server.env's comment stays.
  s = run({ch('s')});
  std::string text;
  const std::string menv = cfg.model_envs_dir + "/moe-a.env";
  read_text(menv, text);
  expect(has(text, "XEN_MODEL=" + models + "/moe-a") && !has(text, "XEN_SLOTS=") && has(text, "XEN_PORT=8123") && !has(text, "XEN_KV="),
         "s writes the shared settings to the model's own file (models/<folder>.env), with its XEN_MODEL");
  {  // The rest to the setup's file (qwen38, XEN_Q38_TP unset = 2)
    std::string st;
    read_text(cfg.model_envs_dir + "/moe-a.tp2.env", st);
    expect(has(st, "XEN_SLOTS=2") && !has(st, "XEN_PORT") && !has(st, "XEN_MODEL") && !has(st, "XEN_KV="),
           "s writes the setup's settings to models/<folder>.tp2.env (only what differs from the model's file and the defaults)");
  }
  read_text(env, text);
  expect(has(text, "# mine") && has(text, "XEN_MODEL_ENV=" + menv) && !has(text, "XEN_PORT=") && !has(text, "XEN_SLOTS="),
         "server.env keeps its comment, names the model's file and holds no settings any more");
  expect(!app.dirty(), "saved: nothing unsaved");
  {  // Per-setup settings. Changing the card count loads that setup's own file; s saves the choice in
     // the model's file; switching back reloads the first setup's file.
    write(cfg.model_envs_dir + "/moe-a.tp4.env", "# setup 4\nXEN_SLOTS=8\nXEN_PORT=9999\n");
    app.set_value("XEN_Q38_TP", "4");
    const bool loaded = app.values().at("XEN_SLOTS") == "8" && app.values().at("XEN_PORT") == "8123" && app.values().at("XEN_Q38_TP") == "4";
    run({});  // a frame: the preview follows the values
    expect(loaded && app.dirty() && has(app.preview(), "--slots 8") && has(app.preview(), "--tp 4"),
           "setup switch: XEN_Q38_TP 4 loads models/<folder>.tp4.env (its shared keys ignored), the choice unsaved");
    run({ch('s')});
    std::string mt, t4;
    read_text(menv, mt);
    read_text(cfg.model_envs_dir + "/moe-a.tp4.env", t4);
    expect(has(mt, "XEN_Q38_TP=4") && !has(mt, "XEN_SLOTS=") && has(t4, "# setup 4") && has(t4, "XEN_SLOTS=8") && !has(t4, "XEN_PORT") && !app.dirty(),
           "setup switch: s writes the choice to the model's file and keeps the setup's file (its comment; shared keys dropped)");
    app.set_value("XEN_SLOTS", "1");
    app.set_value("XEN_Q38_TP", "2");
    expect(app.popup_open(), "setup switch with unsaved setup changes asks first");
    run({Key::of(Key::Down), Key::of(Key::Enter)});  // discard them and switch
    expect(app.values().at("XEN_SLOTS") == "2" && app.values().at("XEN_Q38_TP") == "2", "setup switch back: models/<folder>.tp2.env again (the discarded change gone)");
    run({ch('s')});
    expect(!app.dirty(), "setup switch back: saved");
  }
  // Switch the model to the dense one through the model popup (filter + Enter).
  s = run({Key::of(Key::Esc), ch('m')});
  expect(app.popup_open() && has(s, "dense-b") && has(s, "not served by this engine") && !has(s, "moe-a-mtp") && !has(s, "moe-a-dflash2"),
         "model popup: grouped by arch, drafter checkpoints left out");
  s = run({ch('d'), ch('e'), ch('n'), Key::of(Key::Enter)});
  expect(app.values().at("XEN_MODEL") == models + "/dense-b" && has(s, "qwen35"), "model popup: filter and pick");
  // XEN_MODELS_DIR (the settings file's models folder): the m list and the drafter lists scan it; empty = the
  // project's models/ folder. A --models DIR (the first App above) wins over it.
  {
    const std::string dir2 = t + "/elsewhere";
    mkdir(dir2.c_str(), 0755);
    mkdir((dir2 + "/moe-x").c_str(), 0755);
    write(dir2 + "/moe-x/config.json", "{\"model_type\": \"qwen4_exp\"}");
    write(t + "/server-dir.env", "# x\nXEN_SERVED_NAME=z\nXEN_MODEL=\nXEN_MODELS_DIR=" + dir2 + "\n");
    AppConfig c2;
    c2.root = root;
    c2.env_path = t + "/server-dir.env";
    c2.model_envs_dir = t + "/model-envs2";
    c2.state_dir = t + "/state2";
    c2.profiles_dir = t + "/profiles2";
    c2.experts_dir = t + "/experts2";
    c2.live = false;
    App a2(c2);
    expect(a2.models_dir() == dir2, "models folder: the settings file's XEN_MODELS_DIR beats the project's models/");
    a2.handle(ch('m'));
    a2.draw(c);
    expect(a2.popup_open() && has(screen(c), "moe-x"), "m scans the settings file's XEN_MODELS_DIR");
    a2.handle(Key::of(Key::Esc));
    // Changing the folder in the TUI rescans the m list the next time it opens (no launcher restart).
    const std::string dir3 = t + "/somewhere";
    mkdir(dir3.c_str(), 0755);
    mkdir((dir3 + "/moe-w").c_str(), 0755);
    write(dir3 + "/moe-w/config.json", "{\"model_type\": \"qwen4_exp\"}");
    a2.set_value("XEN_MODELS_DIR", dir3);
    expect(a2.models_dir() == dir3, "changing XEN_MODELS_DIR in the TUI moves the models folder live");
    a2.handle(ch('m'));
    a2.draw(c);
    expect(a2.popup_open() && has(screen(c), "moe-w") && !has(screen(c), "moe-x"), "the m list rescans the new folder (the old one's models are gone)");
    a2.handle(Key::of(Key::Esc));
  }
  {
    write(t + "/server-nodefault.env", "# x\nXEN_SERVED_NAME=z\nXEN_MODEL=\n");
    AppConfig c3;
    c3.root = root;
    c3.env_path = t + "/server-nodefault.env";
    c3.model_envs_dir = t + "/model-envs3";
    c3.state_dir = t + "/state3";
    c3.profiles_dir = t + "/profiles3";
    c3.experts_dir = t + "/experts3";
    c3.live = false;
    App a3(c3);
    expect(a3.models_dir() == root + "/models", "XEN_MODELS_DIR empty = the project's models/ folder");
  }
  expect(app.values().at("XEN_PORT") == "8000" && app.values().at("XEN_SLOTS") == "4", "a model without a settings file starts from the defaults (moe-a's port and slots stay in its file)");
  expect(app.values().at("XEN_SERVED_NAME") == "dense-b", "switching the model retakes the served name from its folder name");
  expect(has(s, "KV pool") && has(s, "Checkpoint drafter"), "qwen35 settings shown after the switch");
  expect(!has(app.preview(), "--hot-rank"), "a qwen38-only setting is left at its default (no --hot-rank) for qwen35");
  // The DFlash drafter is picked, not typed: Enter lists the DFlash 2 checkpoints under the models directory.
  s = run({ch('/'), ch('f'), ch('l'), ch('a'), Key::of(Key::Enter), Key::of(Key::Enter)});
  expect(app.selected_key() == "XEN_DRAFTER_MODEL" && app.popup_open() && has(s, "MTP layer") && has(s, "moe-a-dflash2") && has(s, "dense-b-dspine"),
         "DFlash drafter: Enter lists the DFlash 2 and DSpine checkpoints");
  s = run({Key::of(Key::Down), Key::of(Key::Down), Key::of(Key::Down), Key::of(Key::Enter)});  // MTP layer, dense-b-dspine, dense-b-dspine2, moe-a-dflash2
  expect(!app.popup_open() && app.values().at("XEN_DRAFTER_MODEL") == models + "/moe-a-dflash2", "picking one sets XEN_DRAFTER_MODEL");
  expect(has(app.preview(), "--drafter-model"), "the preview gains --drafter-model");
  s = run({Key::of(Key::Right)});
  expect(app.values().at("XEN_DRAFTER_MODEL").empty() && !has(app.preview(), "--drafter-model"), "Right cycles back to the MTP layer");
  run({Key::of(Key::Esc)});
  app.set_value("XEN_CONTEXT", "2097152");
  s = run({ch('r')});
  const bool blocked = has(s, "invalid setting") && has(s, "1,048,576") && has(s, "Launch (blocked)");
  expect(blocked, "launch blocked: a max context past the kernels' 1,048,576");
  if (!blocked) std::printf("%s", s.c_str());
  s = run({Key::of(Key::Esc), ch('u')});
  // Profiles: save the current values as a new profile.
  s = run({ch('p'), ch('n'), Key::ctrl('u'), ch('t'), ch('e'), ch('s'), ch('t'), Key::of(Key::Enter)});
  read_text(cfg.profiles_dir + "/test.env", text);
  expect(has(text, "dense-b") && !app.popup_open(), "p n saves a profile");
  s = run({ch('p')});
  expect(has(s, "test") && has(s, "server.env") && has(s, "per-model settings") && has(s, "moe-a"), "the profile list shows server.env, the per-model files and the profile");
  s = run({Key::of(Key::Esc)});
  // Back to server.env (model mode), then moe-a through the model popup: its own settings come back.
  s = run({ch('p'), Key::of(Key::Home), Key::of(Key::Enter)});
  s = run({ch('m'), ch('m'), ch('o'), ch('e'), ch('-'), ch('a'), Key::of(Key::Enter)});
  expect(app.values().at("XEN_MODEL") == models + "/moe-a" && app.values().at("XEN_PORT") == "8123" && app.values().at("XEN_SLOTS") == "2" && !app.dirty(),
         "switching back to a model reopens its own settings file");
  app.set_value("XEN_PORT", "8124");
  s = run({ch('m'), ch('d'), ch('e'), ch('n'), Key::of(Key::Enter)});
  expect(app.popup_open() && has(s, "Unsaved changes") && has(s, "Save them, then switch"), "a model switch with unsaved changes asks: save, discard or cancel");
  s = run({Key::of(Key::Down), Key::of(Key::Enter)});
  read_text(menv, text);
  expect(app.values().at("XEN_MODEL") == models + "/dense-b" && has(text, "XEN_PORT=8123"), "discard and switch: the old model's file unchanged");
  s = run({Key::of(Key::Esc)});
  // Logs tab without a server, then the Benchmark tab, then back (the Chat tab is hidden for now: Tab skips it).
  s = run({Key::of(Key::Tab)});
  expect(app.tab() == 1 && has(s, "NO SERVER") && has(s, "Log"), "Logs tab: no server yet");
  s = run({Key::of(Key::Tab)});
  expect(app.tab() == 3 && has(s, "Benchmark"), "Benchmark tab: one Tab from the Logs tab, the hidden Chat tab skipped");
  s = run({Key::of(Key::Tab)});
  expect(app.tab() == 0, "Tab cycles back to Settings");
  // Launch checks (test mode: no process is started).
  s = run({ch('r')});
  expect(app.popup_open() && has(s, "server binary") && has(s, "port 8000"), "r opens the launch checks");
  s = run({Key::of(Key::Esc)});
  // Quit with unsaved changes asks first.
  run({ch('/'), ch('p'), ch('o'), ch('r'), ch('t'), Key::of(Key::Enter), ch('e'), Key::ctrl('u'), ch('9'), ch('0'), ch('0'), ch('1'), Key::of(Key::Enter)});
  expect(app.values().at("XEN_PORT") == "9001", "typed port accepted");
  const bool still = app.handle(ch('q'));
  app.draw(c);
  s = screen(c);
  expect(still && app.popup_open() && has(s, "Quit without saving"), "q with unsaved changes asks");
  // Help overlay, then a narrow terminal still draws.
  run({Key::of(Key::Esc), ch('?')});
  s = screen(c);
  expect(has(s, "keys"), "? shows the keys");
  run({Key::of(Key::Esc)});
  c.resize(70, 22);
  app.draw(c);
  s = screen(c);
  expect(has(s, "Settings") && has(s, "Command"), "70x22 draws the stacked layout");
  c.resize(40, 10);
  app.draw(c);
  expect(has(screen(c), "64x18"), "a tiny terminal gets a message, not garbage");
  std::string cmd = "rm -rf '" + t + "'";
  (void)!std::system(cmd.c_str());
}

// The Chat tab against a fake SSE server: tab keys, sending, the streamed answer, tokens from the
// usage chunk, the history re-sent on the next turn, the refused-connection error, and F4.
void test_chat(const std::string &root) {
  std::string tmpl = "/tmp/xenithium-launch-chat-XXXXXX";  // owned: mkdtemp writes the name into it
  const char *tdir = mkdtemp(&tmpl[0]);
  if (!tdir) {
    expect(false, "mkdtemp");
    return;
  }
  const std::string t = tdir;
  FakeChat fc;
  expect(fc.start(), "fake chat server starts");

  const std::string env = t + "/server.env";
  write(env, "# chat test\nXEN_PORT=8123\n");
  AppConfig cfg;
  cfg.root = root;
  cfg.env_path = env;
  cfg.model_envs_dir = t + "/model-envs";
  cfg.models_dir = t + "/models";
  cfg.state_dir = t + "/state";
  cfg.profiles_dir = t + "/profiles";
  cfg.live = false;
  App app(cfg);
  app.set_chat_port(fc.port);
  Canvas c;
  c.resize(120, 36);
  auto run = [&](std::initializer_list<Key> ks) {
    for (const Key &k : ks) {
      app.handle(k);
      app.draw(c);
    }
    app.draw(c);
    return screen(c);
  };
  auto settle = [&] {  // wait for the worker thread to finish
    for (int i = 0; i < 500 && app.chat_busy(); ++i) {
      usleep(10 * 1000);
      app.draw(c);
    }
  };
  app.show_tab(2);  // the Chat tab is hidden from the tab keys for now: the test opens it by its number
  std::string s = run({});
  expect(app.tab() == 2 && has(s, "starts over"), "chat: the Chat tab renders (opened by number while hidden)");
  // Send a message.
  expect(app.frame_wait_ms() == 200, "chat: idle frames sleep long");
  s = run({ch('h'), ch('e'), ch('l'), ch('l'), ch('o'), Key::of(Key::Enter)});
  expect(app.chat_busy(), "chat: Enter sends and the request runs");
  expect(app.frame_wait_ms() < 100, "chat: streaming frames poll fast (smooth deltas)");
  settle();
  expect(!app.chat_busy() && !app.chat_failed(), "chat: the stream finishes without an error");
  expect(app.frame_wait_ms() == 200, "chat: idle again after the answer");
  expect(app.chat_answer() == "Hello xenithium", "chat: the answer accumulates the content deltas (reasoning kept apart)");
  app.draw(c);
  s = screen(c);
  expect(has(s, "Hello xenithium") && has(s, "you") && has(s, "model") && has(s, "hmm "), "chat: the transcript shows user, model and the dim reasoning");
  expect(has(s, "2 tok"), "chat: completion tokens from the usage chunk");
  expect(fc.reqs.size() == 1 && has(fc.reqs[0], "\"role\":\"user\"") && has(fc.reqs[0], "\"stream\":true") && has(fc.reqs[0], "hello"),
         "chat: the request carries the user turn and stream: true");
  // A second turn re-sends the whole history.
  s = run({ch('a'), ch('g'), ch('a'), ch('i'), ch('n'), Key::of(Key::Enter)});
  settle();
  expect(app.chat_answer() == "Hello xenithium" && !app.chat_failed(), "chat: the second turn answers");
  expect(fc.reqs.size() == 2 && has(fc.reqs[1], "\"role\":\"assistant\"") && has(fc.reqs[1], "Hello xenithium"),
         "chat: the second request re-sends the conversation (assistant turn included)");
  // F4 starts a new conversation (refused while a request runs - none runs now).
  s = run({Key::of(Key::F4)});
  app.draw(c);
  s = screen(c);
  expect(!has(s, "Hello xenithium") && has(s, "starts over"), "chat: F4 clears the conversation");
  // The server gone: the error is shown and the truncated answer is not kept.
  fc.stop();
  s = run({ch('d'), ch('e'), ch('a'), ch('d'), Key::of(Key::Enter)});
  settle();
  expect(!app.chat_busy() && app.chat_failed() && app.chat_answer().empty(), "chat: a refused connection fails and drops the partial answer");
  app.draw(c);
  s = screen(c);
  expect(has(s, "cannot connect") && has(s, "dead"), "chat: the error is shown, the user turn stays");

  {  // the mouse regression on this tab too: a header click leaves Chat (the input line owns keys, not the mouse)
    int x = 0, y = 0;
    expect(app.tab_click_point(0, x, y), "chat: the header has a Settings chip to aim at");
    Key m;
    m.type = Key::Mouse;
    m.press = true;
    m.button = 0;
    m.mx = x;
    m.my = y;
    s = run({m});
    expect(app.tab() == 0, "chat: a left click on the header leaves the Chat tab");
  }
  std::string cmd = "rm -rf '" + t + "'";
  (void)!std::system(cmd.c_str());
}

void test_bench(const std::string &root) {
  const char *tmp = std::getenv("TMPDIR");
  std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/xen-bench-XXXXXX";
  const char *tdir = mkdtemp(tmpl.data());
  if (!tdir) {
    expect(false, "mkdtemp");
    return;
  }
  const std::string t = tdir;
  const std::string runs = t + "/runs";
  mkdir(runs.c_str(), 0755);
  // Two models: alpha with two runs, beta with one. A results.json that xenithium-bench did not write (no "tool") is
  // not a run of this tab: other-1 is skipped.
  for (const auto &d : std::vector<std::string>{"alpha-1", "alpha-2", "beta-1", "other-1"}) mkdir((runs + "/" + d).c_str(), 0755);
  write(runs + "/alpha-1/results.json",
        "{\"schema\":1,\"tool\":\"xenithium-bench\","
        "\"env\":{\"model\":\"alpha\",\"endpoint\":\"http://x:1/v1\",\"host\":\"h\",\"timestamp\":\"20260101-000001\"},"
        "\"summary\":{\"decode_tps\":105.0,\"decode_by_category\":[{\"category\":\"prose\",\"decode_tps\":105.0}],"
        "\"prefill\":[{\"depth\":\"2,000\",\"prefill_tps\":5000.0,\"ttft_ms\":400.0}],"
        "\"concurrency\":[{\"streams\":\"1\",\"total_tps\":90.0}]},\"scenarios\":[]}");
  write(runs + "/other-1/results.json", "{\"env\":{\"model\":\"gamma\"},\"single_stream\":{}}");
  write(runs + "/alpha-2/results.json",
        "{\"schema\":1,\"tool\":\"xenithium-bench\","
        "\"env\":{\"model\":\"alpha\",\"endpoint\":\"http://x:1/v1\",\"host\":\"h\",\"timestamp\":\"20260101-000002\",\"notes\":{\"XEN_SLOTS\":\"16\"}},"
        "\"summary\":{\"decode_tps\":120.0,\"decode_by_category\":[{\"category\":\"code\",\"decode_tps\":130.0}],"
        "\"prefill\":[{\"depth\":\"2,000\",\"prefill_tps\":4300.0,\"ttft_ms\":470.0},{\"depth\":\"128,000\",\"prefill_tps\":3900.0,\"ttft_ms\":33000.0}],"
        "\"deep_decode\":[{\"depth\":\"109,000\",\"decode_tps\":88.0,\"ttft_ms\":60.0}],"
        "\"concurrency\":[{\"streams\":\"8\",\"total_tps\":612.0}],\"best_tokens_per_joule\":1.9,\"quality_pct\":97.1},\"scenarios\":[]}");
  write(runs + "/beta-1/results.json",
        "{\"schema\":1,\"tool\":\"xenithium-bench\","
        "\"env\":{\"model\":\"beta\",\"endpoint\":\"http://x:2/v1\",\"host\":\"h\",\"timestamp\":\"20260201-000001\"},"
        "\"summary\":{\"decode_tps\":200.0,\"decode_by_category\":[{\"category\":\"code\",\"decode_tps\":200.0}]},\"scenarios\":[]}");

  const std::string env = t + "/server.env";
  write(env, "XEN_PORT=8123\n");
  AppConfig cfg;
  cfg.root = root;
  cfg.env_path = env;
  cfg.model_envs_dir = t + "/model-envs";
  cfg.state_dir = t + "/state";
  cfg.bench_runs = runs;
  cfg.live = false;
  App app(cfg);

  expect(app.bench_run_count() == 3, "bench: 3 runs read from the runs dir (a results.json of another tool is skipped)");
  const auto names = app.bench_model_names();
  expect(names.size() == 2 && names[0] == "beta" && names[1] == "alpha",
         "bench: grouped by model, the newest run's model first (beta 2026-02 before alpha 2026-01)");

  Canvas c;
  c.resize(120, 36);
  auto run = [&](std::initializer_list<Key> ks) {
    for (const Key &k : ks) {
      app.handle(k);
      app.draw(c);
    }
    app.draw(c);
    return screen(c);
  };
  std::string s = run({ch('3')});  // Benchmark is the third shown tab while Chat stays hidden
  expect(app.tab() == 3 && has(s, "Benchmark") && has(s, "beta") && has(s, "alpha") && has(s, "3 runs"),
         "bench: key 3 opens the Benchmark tab (the digits name the shown tabs), runs grouped by model with a run count");
  // A mouse regression: a left click on a header chip switches tabs from any tab - the Benchmark tab used to swallow
  // every mouse event for its scroll handling.
  auto click_tab = [&](int t) {
    int x = 0, y = 0;
    if (!app.tab_click_point(t, x, y)) return false;
    Key m;
    m.type = Key::Mouse;
    m.press = true;
    m.button = 0;
    m.mx = x;
    m.my = y;
    app.handle(m);
    app.draw(c);
    return true;
  };
  expect(click_tab(1) && app.tab() == 1, "bench: clicking the Logs chip in the header leaves the Benchmark tab");
  app.draw(c);
  expect(click_tab(3) && app.tab() == 3, "bench: clicking the Benchmark chip returns from Logs");
  expect(has(s, "alpha-2   120.0 dec   3900.0 pp@128000   88.0 dec@109000   612.0 conc@8   97.1% quality") && has(s, "alpha-1   105.0 dec"),
         "bench: a run lists its decode, deepest prefill, decode at depth and widest concurrency");
  // The log panel tails the last run's output from the state dir; esc in the list view is a
  // no-op (x is the only stop key) and w toggles the log's wrapping.
  mkdir((t + "/state").c_str(), 0755);
  write(t + "/state/bench.log", "== Single-stream decode by category\n  42.1 tok/s\n");
  s = run({Key::of(Key::Esc), ch('w')});
  expect(app.tab() == 3 && has(s, "Benchmark") && has(s, "Log") && has(s, "42.1 tok/s") && has(s, "no wrap"),
         "bench: esc in the list view is a no-op; the log panel shows the run's output (w = no wrap)");
  s = run({Key::of(Key::Down), Key::of(Key::Enter)});
  expect(has(s, "decode") && has(s, "200.0"), "bench: the report shows the selected run's decode figure");
  expect(app.bench_selected_decode() == "200.0", "bench: the selected run's decode value");
  s = run({Key::of(Key::Esc)});
  expect(app.tab() == 3 && has(s, "Benchmark"), "bench: esc leaves the report, back to the list");
  // alpha's newest run: its report carries the summary's tables and the notes.
  s = run({Key::of(Key::Down), Key::of(Key::Enter)});  // Down skips alpha's header
  expect(app.bench_selected_decode() == "120.0" && has(s, "code 130.0") && has(s, "2000:4300.0") && has(s, "decode at depth 109000:88.0") && has(s, "8:612.0") &&
             has(s, "xenithium-bench report") && has(s, "XEN_SLOTS=16") && has(s, "efficiency 1.9 tokens per joule") &&
             has(s, "quality   97.1 %") && has(s, "report.html"),
         "bench: a run's report shows its summary (categories, prefill, decode at depth, concurrency), its notes and the report command");
  s = run({Key::of(Key::Esc), Key::of(Key::Down), Key::of(Key::Enter)});
  expect(app.bench_selected_decode() == "105.0", "bench: the next run down is alpha's older one");

  // r opens the mode menu: the suites and three scenarios alone. In test mode no server runs, so a pick only warns.
  s = run({Key::of(Key::Esc), ch('r')});
  bool every_mode_listed = has(s, "Run the benchmark");
  for (const BenchMode mode : kBenchModes) every_mode_listed &= has(s, std::string(bench_mode_title(mode)).substr(0, 24));
  expect(every_mode_listed && has(s, "Standard (decode") && has(s, "Long contexts") && has(s, "Load (concurrency") && has(s, "Agents (1, 2, 4 agents") &&
             has(s, "Real-world apps (chat") && has(s, "Concurrency at stream counts you enter"),
         "bench: r opens the mode menu (standard / quick / full / long / load / agents / check / apps / decode / prefill / concurrency / streams)");
  s = run({Key::of(Key::Down), Key::of(Key::Down), Key::of(Key::Down), Key::of(Key::Enter)});
  expect(!has(s, "Run the benchmark") && has(s, "no ready server"), "bench: a mode picked without a ready server only warns");
  // the Streams mode: the last entry of the menu; without a server it warns before asking for the counts
  run({ch('r')});
  for (size_t i = 0; i + 1 < std::size(kBenchModes); ++i) run({Key::of(Key::Down)});
  s = run({Key::of(Key::Enter)});
  expect(!has(s, "Run the benchmark") && !has(s, "Concurrency at your stream counts") && has(s, "no ready server"),
         "bench: the stream-count mode without a ready server warns instead of asking for counts");
  // one measuring method: no mode menu. A letter acts only on the tab whose hint row shows it (m once jumped to
  // Settings' model list), so m does nothing on Benchmark
  s = run({ch('m')});
  expect(app.tab() == 3 && !app.popup_open() && !has(s, "Benchmark mode") && !has(s, "m mode") && !has(s, "dense-b"),
         "bench: m does nothing on the Benchmark tab (no mode menu, no model list)");
  s = run({Key::of(Key::Down), Key::of(Key::Down), Key::of(Key::Enter), ch('m'), ch('r')});
  expect(app.tab() == 3 && !app.popup_open(), "bench: in a run's report m and r do nothing (no model list, no launch checks)");
  run({Key::of(Key::Esc)});
  s = run({ch('2'), ch('m')});
  expect(app.tab() == 1 && !app.popup_open() && !has(s, "m model"), "logs: m does nothing there (the model list is the Settings tab's key)");
  s = run({ch('1'), ch('m')});
  expect(app.tab() == 0 && app.popup_open(), "settings: m opens the model list");
  run({Key::of(Key::Esc), ch('3')});
  std::string level_err;
  const std::string levels = bench_stream_levels("48, 1 8,,8 24", level_err);
  std::string e0, e1, e2, e3, e4;
  expect(levels == "1,8,24,48" && level_err.empty() && bench_stream_levels("", e0).empty() && !e0.empty() && bench_stream_levels("0", e1).empty() &&
             bench_stream_levels("257", e2).empty() && bench_stream_levels("8x", e3).empty() && e3.find("\"8x\"") != std::string::npos &&
             bench_stream_levels("1,2,3,4,5,6,7,8,9,10,11,12,13", e4).empty() && e4.find("at most 12") != std::string::npos,
         "bench: stream counts are read from commas or spaces, sorted without repeats; empty, 0, past 256, not a number or 13 counts are refused");
  expect(bench_mode_args(BenchMode::Streams) == std::vector<std::string>{"--scenario", "concurrency"} && std::string(bench_mode_name(BenchMode::Streams)) == "streams",
         "bench: the stream-count mode runs the concurrency scenario (its levels come from the input)");
  expect(bench_mode_args(BenchMode::Standard) == std::vector<std::string>{"--suite", "standard"} &&
             bench_mode_args(BenchMode::Long) == std::vector<std::string>{"--suite", "long"} && bench_mode_args(BenchMode::Load) == std::vector<std::string>{"--suite", "load"},
         "bench: a suite mode = --suite <its name>");
  expect(bench_mode_args(BenchMode::Prefill) == std::vector<std::string>{"--scenario", "prefill"} &&
             bench_mode_args(BenchMode::Concurrency) == std::vector<std::string>{"--scenario", "concurrency"},
         "bench: a scenario mode = --scenario <its name>");
  std::set<std::string> mode_names;
  for (const BenchMode mode : kBenchModes) mode_names.insert(bench_mode_name(mode));
  expect(mode_names.size() == std::size(kBenchModes), "bench: every mode has its own tag");
  const auto ln = launch_notes("# Written by xenithium-launch\nXEN_MODEL_ENV=\nXEN_SLOTS=8\nXEN_QSA_PF_TILES=\nXEN_MODEL='/m/a b'\n");
  expect(ln.size() == 2 && ln[0].first == "XEN_MODEL" && ln[0].second == "/m/a b" && ln[1].first == "XEN_SLOTS" && ln[1].second == "8",
         "bench: launch notes = the launch.env's non-empty settings, unquoted, in key order");

  std::string cmd = "rm -rf '" + t + "'";
  (void)!std::system(cmd.c_str());
}

// XRC1 routing counts (model/route_counts.hpp) for the expert-file tests.
void write_xrc(const std::string &path, int layers, int experts, const std::vector<double> &c) {
  std::ofstream f(path, std::ios::binary);
  const uint32_t magic = 0x31435258u;
  f.write(reinterpret_cast<const char *>(&magic), 4);
  f.write(reinterpret_cast<const char *>(&layers), 4);
  f.write(reinterpret_cast<const char *>(&experts), 4);
  f.write(reinterpret_cast<const char *>(c.data()), static_cast<std::streamsize>(c.size() * 8));
}

// The expert files: a MoE model without a ranking, then with one (its coverage), one of another model's shape,
// and the launch check (test mode: nothing starts).
void test_experts(const std::string &root) {
  const char *tmp = std::getenv("TMPDIR");
  std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/xen-launch-experts-XXXXXX";
  const char *tdir = mkdtemp(tmpl.data());
  if (!tdir) {
    expect(false, "mkdtemp");
    return;
  }
  const std::string t = tdir, models = t + "/models", ex = t + "/experts";
  mkdir(models.c_str(), 0755);
  mkdir(ex.c_str(), 0755);
  for (const auto &[name, type] : std::vector<std::pair<std::string, std::string>>{{"moe-q", "qwen4_exp"}, {"moe-n", "qwen4_exp"}}) {
    mkdir((models + "/" + name).c_str(), 0755);
    write(models + "/" + name + "/config.json", "{\"model_type\": \"" + type + "\", \"num_hidden_layers\": 2, \"num_experts\": 4}");
  }
  const ModelInfo mq = probe_model(models + "/moe-q"), mn = probe_model(models + "/moe-n");
  expect(mq.arch == "qwen38" && mn.arch == "qwen38" && mq.layers == 2 && mq.experts == 4 && is_moe_arch("qwen38") && !is_moe_arch("qwen35"),
         "experts: two fake qwen38 checkpoints of 2 layers x 4 experts");
  const ExpertFile none = expert_file("auto", "hot-rank.xrc", root, ex, mq);
  expect(none.is_auto && none.path == ex + "/moe-q/hot-rank.xrc" && !none.exists, "experts: auto resolves to <experts>/<model folder>/hot-rank.xrc");
  // layer 0 routes expert 0, layer 1 expert 3: the ranking holds both with 2 hot, the default placement (the lowest ids,
  // one per layer) only layer 0's.
  mkdir((ex + "/moe-q").c_str(), 0755);
  write_xrc(ex + "/moe-q/hot-rank.xrc", 2, 4, {10, 0, 0, 0, 0, 0, 0, 10});
  const ExpertFile rf = expert_file("auto", "hot-rank.xrc", root, ex, mq);
  double rk = 0, df = 0;
  expect(rf.exists && rf.shape_ok && rf.error.empty() && rank_coverage(rf.path, mq, 2, rk, df) && rk == 1.0 && df == 0.5,
         "experts: a ranking of the model's shape, coverage 100 % ranked vs 50 % default at 2 hot");
  write_xrc(t + "/other.xrc", 3, 4, std::vector<double>(12, 1.0));
  const ExpertFile bad = expert_file(t + "/other.xrc", "hot-rank.xrc", root, ex, mq);
  expect(bad.exists && !bad.shape_ok && has(bad.error, "another model"), "experts: a ranking of another shape is unusable, with the reason");

  const std::string env = t + "/server.env";
  write(env, "XEN_MODEL=" + models + "/moe-n\nXEN_PORT=8124\n");
  AppConfig cfg;
  cfg.root = root;
  cfg.env_path = env;
  cfg.model_envs_dir = t + "/model-envs";
  cfg.models_dir = models;
  cfg.state_dir = t + "/state";
  cfg.profiles_dir = t + "/profiles";
  cfg.experts_dir = ex;
  cfg.live = false;
  App app(cfg);
  Canvas c;
  c.resize(120, 40);
  auto run = [&](std::initializer_list<Key> ks) {
    for (const Key &k : ks) {
      app.handle(k);
      app.draw(c);
    }
    app.draw(c);
    return screen(c);
  };
  std::string s = run({});
  expect(has(s, "recorded from this run on") && has(s, "server creates"), "experts: a MoE model without a ranking yet: the card and the list say the server records it");
  expect(!has(s, "invalid") && !has(s, "no hot-expert ranking"), "experts: a ranking that does not exist yet is not a problem (the server records it)");
  expect(has(app.preview(), "--hot-rank") && has(app.preview(), ".cache/experts/moe-n/hot-rank.xrc"),
         "experts: the script resolves auto to the model's ranking file even before it exists (the server creates it)");
  // The launch checks mention the ranking.
  s = run({ch('1'), ch('r')});
  expect(app.popup_open() && has(s, "hot-expert ranking"), "launch checks: the MoE ranking line");
  run({Key::of(Key::Esc)});
  // A dense model's ranking setting stays out of the command line.
  std::string cmd = "rm -rf '" + t + "'";
  (void)!std::system(cmd.c_str());
}

}  // namespace

// Per-model settings files: scripts/start-server.sh applies the model's file under the settings file and the
// environment; a file of another checkpoint is not applied; XEN_MODEL_ENV= turns it off. Also the launcher's lookup.
void test_model_env(const std::string &root) {
  std::string tmpl = "/tmp/xenithium-launch-menv-XXXXXX";
  if (!mkdtemp(&tmpl[0])) {
    expect(false, "mkdtemp");
    return;
  }
  const std::string t = tmpl, m = t + "/m.env", fake = t + "/fake-model";
  write(m, "# a model's settings\nXEN_MODEL=" + fake + "\nXEN_PREFILL_CHUNK=1024\nXEN_SLOTS=2\n");
  auto dry = [&](const std::string &settings, const std::vector<std::pair<std::string, std::string>> &env = {}) {
    write(t + "/s.env", settings);
    std::vector<std::pair<std::string, std::string>> e = {{"SERVE_ENV", t + "/s.env"}, {"SERVE_DRY_RUN", "1"}};
    e.insert(e.end(), env.begin(), env.end());
    const RunResult r = run_capture({"bash", "scripts/start-server.sh"}, e, root, 5000);
    return r.out + r.err;
  };
  std::string o = dry("XEN_MODEL_ENV=" + m + "\n");
  if (!has(o, "--prefill-chunk 1024")) std::printf("%s\n", o.c_str());
  expect(has(o, "--model " + fake) && has(o, "--prefill-chunk 1024") && has(o, "--slots 2") && has(o, "per-model settings from"),
         "start-server: XEN_MODEL_ENV applies the model's file (its XEN_MODEL too)");
  o = dry("XEN_MODEL_ENV=" + m + "\nXEN_SLOTS=3\n");
  expect(has(o, "--slots 3") && has(o, "--prefill-chunk 1024"), "start-server: a key in the settings file wins over the model's file");
  o = dry("XEN_MODEL_ENV=" + m + "\n", {{"XEN_SLOTS", "1"}});
  expect(has(o, "--slots 1") && has(o, "--prefill-chunk 1024"), "start-server: a key in the environment wins over the model's file");
  o = dry("XEN_MODEL_ENV=" + m + "\nXEN_MODEL=" + t + "/other\n");
  expect(has(o, "--model " + t + "/other") && has(o, "--prefill-chunk 0") && has(o, "is for"), "start-server: a model file of another checkpoint is not applied");
  o = dry("XEN_MODEL_ENV=\nXEN_MODEL=" + fake + "\n");
  expect(has(o, "--prefill-chunk 0") && !has(o, "per-model settings from"), "start-server: XEN_MODEL_ENV= applies none");
  o = dry("XEN_MODEL_ENV=qwen38-flash-next\n", {{"XEN_MODEL_ENV", m}});
  expect(has(o, "--model " + fake), "start-server: the caller's XEN_MODEL_ENV wins over the settings file's");
  // Per-setup files (<id>.tp<N>.env) over the model's file, picked by the setup key (unset = 2)
  write(t + "/m.tp4.env", "XEN_SLOTS=6\nXEN_Q38_TP=1\nXEN_MODEL=" + t + "/other\n");
  write(t + "/m.tp2.env", "XEN_PREFILL_CHUNK=512\n");
  o = dry("XEN_MODEL_ENV=" + m + "\nXEN_Q38_TP=4\n");
  expect(has(o, "--slots 6") && has(o, "--tp 4") && has(o, "--model " + fake) && has(o, "--prefill-chunk 1024") && has(o, "per-setup settings from " + t + "/m.tp4.env"),
         "start-server: the setup's file over the model's file (its setup key and checkpoint ignored)");
  o = dry("XEN_MODEL_ENV=" + m + "\n");
  expect(has(o, "--slots 2") && has(o, "--prefill-chunk 512") && has(o, "--tp 2"), "start-server: the setup key unset = setup 2");
  o = dry("XEN_MODEL_ENV=" + m + "\nXEN_Q38_TP=4\nXEN_SLOTS=3\n");
  expect(has(o, "--slots 3"), "start-server: the settings file wins over the setup's file");
  expect(setup_env_path(t + "/menvs/b.env", "4") == t + "/menvs/b.tp4.env" && is_setup_env("x/b.tp4.env") && is_setup_env("b.tp12.env") &&
             !is_setup_env("b.env") && !is_setup_env("b.tpx.env") && !is_setup_env("b.tp.env") && setup_key_of("qwen38") == "XEN_Q38_TP" &&
             setup_key_of("qwen35") == "XEN_TP" && setup_key_of("").empty(),
         "setup_env_path / is_setup_env / setup_key_of");
  // the launcher's side
  mkdir((t + "/menvs").c_str(), 0755);
  write(t + "/menvs/b.env", "XEN_MODEL=" + fake + "\n");
  write(t + "/menvs/a.env", "XEN_MODEL=" + t + "/x\n");
  write(t + "/menvs/a.tp2.env", "XEN_MODEL=" + fake + "\n");  // a setup file never counts as a model's file
  expect(find_model_env(t + "/menvs", fake) == t + "/menvs/b.env" && find_model_env(t + "/menvs", t + "/y").empty(), "find_model_env: by the file's XEN_MODEL");
  expect(model_env_new_path(t + "/menvs", t + "/x") == t + "/menvs/x.env" && model_env_id(t + "/menvs/b.env") == "b", "model_env_new_path / model_env_id");
  expect(resolve_model_env(root, t + "/menvs", "b") == t + "/menvs/b.env" && resolve_model_env(root, t + "/menvs", "auto").empty(), "resolve_model_env: an id is <dir>/<id>.env");

  // Presets (<dir>/presets/<id>.env, shipped) and the user's files (<dir>/<id>.env): the user's file replaces the preset
  // of the same name, and nothing writes a preset.
  const std::string pd = t + "/menvs/presets";
  mkdir(pd.c_str(), 0755);
  const std::string preset_text = "# the preset's own header\nXEN_MODEL=" + t + "/pm\n# a note on the chunk\nXEN_PREFILL_CHUNK=2048\n";
  write(pd + "/p.env", preset_text);
  write(pd + "/p.tp4.env", "XEN_SLOTS=6\n");
  write(pd + "/b.env", "XEN_MODEL=" + t + "/shadowed\n");  // b.env has a user file: this preset is not listed
  expect(preset_env_path(t + "/menvs/p.env") == pd + "/p.env" && user_env_path(pd + "/p.env") == t + "/menvs/p.env" &&
             user_env_path(t + "/menvs/b.env") == t + "/menvs/b.env" && readable_env_path(t + "/menvs/p.env") == pd + "/p.env" &&
             readable_env_path(t + "/menvs/b.env") == t + "/menvs/b.env" && readable_env_path(t + "/menvs/none.env") == t + "/menvs/none.env",
         "preset_env_path / user_env_path / readable_env_path");
  {
    const auto files = model_env_files(t + "/menvs");
    expect(files.size() == 3 && files[0] == t + "/menvs/a.env" && files[1] == t + "/menvs/b.env" && files[2] == pd + "/p.env",
           "model_env_files: the user's files, then the presets no user file replaces (setup files never)");
  }
  expect(find_model_env(t + "/menvs", t + "/pm") == pd + "/p.env" && find_model_env(t + "/menvs", t + "/shadowed").empty(),
         "find_model_env: a preset is found by its XEN_MODEL; a replaced preset is not");
  write(pd + "/q.env", "XEN_PRESET_EXPLICIT=1\nXEN_MODEL=" + t + "/qm\n");  // needs files beyond the checkpoint: by name only
  expect(find_model_env(t + "/menvs", t + "/qm").empty() && resolve_model_env(root, t + "/menvs", "q") == pd + "/q.env",
         "find_model_env: a preset marked XEN_PRESET_EXPLICIT=1 is never the model's file; its id still names it");
  unlink((pd + "/q.env").c_str());
  {  // Trained page selection banks: folders whose safetensors hold indexer tensors, listed with their layers
    const std::string banks = t + "/banks", full = banks + "/bank-all", l7 = banks + "/bank-7", ck = banks + "/ckpt";
    for (const std::string &d : {banks, full, l7, ck}) mkdir(d.c_str(), 0755);
    auto st = [&](const std::string &path, std::vector<int> layers) {  // a minimal safetensors: header + one f32 per tensor
      std::string h = "{";
      for (size_t i = 0; i < layers.size(); ++i)
        h += (i ? "," : "") + std::string("\"model.language_model.layers.") + std::to_string(layers[i]) + ".self_attn._qsa_indexer.q_norm\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[" +
             std::to_string(4 * i) + "," + std::to_string(4 * i + 4) + "]}";
      h += "}";
      std::string body(8, '\0');
      for (int i = 0; i < 8; ++i) body[i] = static_cast<char>((h.size() >> (8 * i)) & 0xff);
      write(path, body + h + std::string(4 * layers.size(), '\0'));
    };
    st(full + "/qsa-indexers.safetensors", {3, 7, 11, 15, 63});
    st(l7 + "/b.safetensors", {7});
    st(ck + "/model.safetensors", {3});
    write(ck + "/config.json", "{}\n");  // a checkpoint folder is never a bank
    const auto found = qsa_banks({banks, banks});
    QsaBank one;
    expect(found.size() == 2 && found[0].path == l7 && found[1].path == full && found[1].layers == std::vector<int>({3, 7, 11, 15, 63}) &&
               qsa_bank_detail(found[1]).rfind("5 layers (3, 7 ... 63), ", 0) == 0 && qsa_bank_detail(found[0]).rfind("1 layer (7), ", 0) == 0 &&
               !probe_qsa_bank(ck, one) && !probe_qsa_bank(t + "/none", one),
           "qsa_banks: the folders holding indexer tensors (not checkpoints), each once, with their layers");
    const Catalog bc = load_catalog(root);
    const Spec *bank = bc.find("XEN_TRAINED_QSA_DELTA_DIR");
    expect(bank && check_value(*bank, full, "qwen35", root).empty() && check_value(*bank, l7, "qwen35", root).empty() && !check_value(*bank, ck, "qwen35", root).empty(),
           "the bank folder must hold indexer tensors (a checkpoint folder does not)");
  }
  expect(resolve_model_env(root, t + "/menvs", "p") == pd + "/p.env" && resolve_model_env(root, t + "/menvs", t + "/menvs/p.env") == pd + "/p.env" &&
             resolve_model_env(root, t + "/menvs", pd + "/b.env") == t + "/menvs/b.env",
         "resolve_model_env: an id or a path without a user file is its preset; a preset's path with a user file is the user's");
  expect(model_env_new_path(t + "/menvs", t + "/p") == t + "/menvs/p-2.env", "model_env_new_path: a preset of another checkpoint holds its name");
  // start-server.sh: the same rule, for the model's file and for the setup's file
  o = dry("XEN_MODEL_ENV=" + t + "/menvs/p.env\nXEN_Q38_TP=4\n");
  expect(has(o, "--model " + t + "/pm") && has(o, "--prefill-chunk 2048") && has(o, "--slots 6") && has(o, "per-model settings from " + pd + "/p.env") &&
             has(o, "per-setup settings from " + pd + "/p.tp4.env"),
         "start-server: without a user file the preset applies (model and setup)");
  write(t + "/menvs/p.tp4.env", "XEN_SLOTS=3\n");
  o = dry("XEN_MODEL_ENV=" + pd + "/p.env\nXEN_Q38_TP=4\n");
  expect(has(o, "--prefill-chunk 2048") && has(o, "--slots 3") && has(o, "per-setup settings from " + t + "/menvs/p.tp4.env"),
         "start-server: the user's setup file replaces the setup preset (the model's preset still applies)");
  write(t + "/menvs/p.env", "XEN_MODEL=" + t + "/pm\nXEN_PREFILL_CHUNK=512\n");
  o = dry("XEN_MODEL_ENV=" + pd + "/p.env\nXEN_Q38_TP=4\n");
  expect(has(o, "--prefill-chunk 512") && has(o, "per-model settings from " + t + "/menvs/p.env"), "start-server: the user's file replaces the preset, even named by the preset's path");
  o = dry("XEN_MODEL=" + t + "/pm\n", {{"XEN_Q38_TP", "4"}});
  expect(has(o, "--model " + t + "/pm"), "start-server: the auto lookup still starts the checkpoint");
  remove((t + "/menvs/p.env").c_str());
  remove((t + "/menvs/p.tp4.env").c_str());

  // The launcher: a model whose settings are a preset opens with the preset's values; a save writes the user's files
  // (started from the presets' lines) and leaves the presets as they were.
  {
    const std::string pm = t + "/pm";
    mkdir(pm.c_str(), 0755);
    write(pm + "/config.json", "{\"model_type\": \"qwen4_exp\"}");
    const std::string env = t + "/server.env";
    write(env, "XEN_MODEL_ENV=p\nXEN_Q38_TP=4\n");
    AppConfig cfg;
    cfg.root = root;
    cfg.env_path = env;
    cfg.model_envs_dir = t + "/menvs";
    cfg.models_dir = t;
    cfg.state_dir = t + "/state";
    cfg.profiles_dir = t + "/profiles";
    cfg.experts_dir = t + "/experts";
    cfg.live = false;
    App app(cfg);
    Canvas c;
    c.resize(120, 36);
    app.draw(c);
    std::string s = screen(c);
    expect(app.values().at("XEN_PREFILL_CHUNK") == "2048" && app.values().at("XEN_SLOTS") == "6" && has(s, "p.env (preset)"),
           "launcher: a preset's values open (model and setup), the title marks the file as a preset");
    app.set_value("XEN_PORT", "8222");
    app.set_value("XEN_SLOTS", "5");
    app.handle(ch('s'));
    std::string user, user_setup, kept, kept_setup;
    read_text(t + "/menvs/p.env", user);
    read_text(t + "/menvs/p.tp4.env", user_setup);
    read_text(pd + "/p.env", kept);
    read_text(pd + "/p.tp4.env", kept_setup);
    expect(has(user, "XEN_PORT=8222") && has(user, "XEN_PREFILL_CHUNK=2048") && has(user, "# a note on the chunk") && has(user, "XEN_MODEL=" + pm) &&
               has(user, "your settings, started from the preset") && !has(user, "the preset's own header"),
           "launcher: s writes the user's file, started from the preset's lines under a header of its own");
    expect(has(user_setup, "XEN_SLOTS=5") && !has(user_setup, "XEN_SLOTS=6"), "launcher: s writes the user's setup file");
    expect(kept == preset_text && kept_setup == "XEN_SLOTS=6\n", "launcher: the presets are byte for byte as shipped after a save");
    app.draw(c);
    expect(!has(screen(c), "(preset)") && !app.dirty(), "launcher: after the save the user's files are the ones edited");
  }
  std::string cmd = "rm -rf '" + t + "'";
  (void)!std::system(cmd.c_str());
}

int main() {
  const std::string root = XEN_SOURCE_DIR;
  test_catalog(root);
  test_env();
  test_validate(root);
  test_keys();
  test_text();
  test_exit_reason();
  test_app(root);
  test_experts(root);
  test_chat(root);
  test_bench(root);
  test_model_env(root);
  std::printf("%s: %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
  return g_failures ? 1 : 0;
}
