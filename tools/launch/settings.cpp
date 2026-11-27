// The launcher's settings catalog, env files, validation and model scan.
#include "settings.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <fstream>
#include <filesystem>
#include <set>
#include <sstream>

#include "xenithium/arch/common/drafter_archs.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/serve/options.hpp"
#include "xenithium/util/cards.hpp"
#include "xenithium/util/host_mem.hpp"
#include "xenithium/util/video.hpp"

namespace {
// a block drafter directory's config.json names its architecture (DFlashConfig::is_dflash's test): DFlash 2 or DSpine
bool is_dflash_dir(const std::string &dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string s = ss.str();
  const auto names = [&](std::string_view n) { return s.find("\"" + std::string(n) + "\"") != std::string::npos; };
  if (names(xenithium::arch::kDFlash2Arch)) return true;
  for (const auto n : xenithium::arch::kDSpineArchs)
    if (names(n)) return true;
  return false;
}
}  // namespace

namespace xenithium::launch {

namespace {

std::string trim(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

std::string strip_ticks(std::string_view s) {
  std::string t = trim(s);
  if (t.size() >= 2 && t.front() == '`' && t.back() == '`') t = t.substr(1, t.size() - 2);
  return t;
}

bool parse_int(const std::string &v, int64_t &out) {
  if (v.empty()) return false;
  char *end = nullptr;
  errno = 0;
  const long long x = std::strtoll(v.c_str(), &end, 10);
  if (errno || *end) return false;
  out = x;
  return true;
}

bool parse_num(const std::string &v, double &out) {
  if (v.empty()) return false;
  char *end = nullptr;
  out = std::strtod(v.c_str(), &end);
  return !*end && std::isfinite(out);
}

std::string fmt_num(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}

// The launcher's schema: label, group, type and scope of every setting start-server.sh reads,
// plus the server's own diagnostic switches. Defaults, flags and help text are not typed here:
// they come from the script and the docs, which the 5-place rule keeps current.

// The description the TUI shows for each setting, in plain words: what it does, what it gives and what it takes. No
// speeds or other measured results: a "Gives:" line names what improves ("+ decode speed"), a "Takes:" line what it
// costs ("- KV pool", "- VRAM: 0.5 GiB per card"). Sizes, limits and defaults are stated; results of runs are not.
// docs/server-env.md carries the same text. Rewrite one when a setting's behaviour changes.
// Layout (term.hpp help_lines draws it; one paragraph per '\n', an empty one = a blank line):
// what the setting does, in one or two short sentences
// `value` what that value does - one line per value worth knowing (`on` / `off, `empty, `a path`)
// Gives: / Takes: / Output: / Needs: / KV context: / Tip: / Note: - labelled lines
// - item - bullets, e.g. one per model
// Names in backticks are drawn as code: settings (`XEN_KV`), values, files. A setting that moves the KV pool says so
// in a "KV context:" line; the panel adds the computed numbers below the text (app.cpp).
const char *brief_of(std::string_view key) {
  static const std::pair<const char *, const char *> t[] = {
      {"XEN_MODEL",
       "The checkpoint folder to serve: weights, `tokenizer.json`, `generation_config.json`.\n"
       "\n"
       "Its architecture (qwen38, qwen35, llama, flint) decides which settings apply. The others are hidden.\n"
       "\n"
       "Tip: press `m` to pick a checkpoint."},
      {"XEN_MODELS_DIR",
       "The folder the launcher scans for checkpoints: the `m` list and the drafter lists.\n"
       "\n"
       "`empty`  the project's `models/` folder\n"
       "\n"
       "Note: only the launcher reads it. The server never does."},
      {"XEN_SERVED_NAME",
       "The model name clients send and see: `/v1/models` and the \"model\" field of a request.\n"
       "\n"
       "`empty`  the checkpoint's folder name\n"
       "\n"
       "Note: a request that names another model gets HTTP 404."},
      {"XEN_HOST",
       "Who can reach the API.\n"
       "\n"
       "`0.0.0.0`  any machine on your network\n"
       "`127.0.0.1`  this computer only"},
      {"XEN_PORT",
       "Port of the OpenAI-compatible API: `http://HOST:PORT/v1`."},
      {"XEN_SLOTS",
       "How many requests are decoded at the same time. More requests wait in the queue.\n"
       "\n"
       "Gives: + tokens per second in all when several requests run at once.\n"
       "Takes: - speed of each single stream (the streams share every step).\n"
       "KV context: on qwen35 / llama / flint every slot keeps its own state in VRAM, so more slots leave a smaller pool. On qwen38 the pool grows with the slots, up to 4 contexts.\n"
       "\n"
       "Limits: qwen38 and qwen35 up to 16, llama and flint up to 8. qwen38 on four cards takes up to 64; each slot above 16 takes about 0.05 GiB of VRAM per card."},
      {"XEN_CONTEXT",
       "The longest conversation ONE request may reach, prompt plus answer. It is the API's `max_model_len`.\n"
       "\n"
       "A limit, not a reservation: all slots share the KV pool, and a request takes KV only as it grows.\n"
       "\n"
       "Past the trained length (262,144 on qwen35 / qwen38, 524,288 on Nanbeige) the engine turns on YaRN by itself, with the smallest factor that reaches the limit: 524,288 on a 262,144 model = factor 2.00.\n"
       "- qwen35, qwen38 and Nanbeige: each request gets its own factor (`XEN_YARN_TIERS`), so short requests keep the trained numerics.\n"
       "- flint needs no YaRN.\n"
       "\n"
       "Limits: up to 1,048,576 tokens. The load refuses a limit whose KV for one request does not fit the cards, and names the largest that does."},
      {"XEN_KV_POOL_TOKENS",
       "The size of the KV pool: one block of KV memory shared by all slots, so a long conversation can use the room the others leave free.\n"
       "\n"
       "`auto`  sized for the model (below)\n"
       "`a number`  that many tokens, at least one max context\n"
       "\n"
       "What auto gives:\n"
       "- qwen35 / llama / flint: all the VRAM left after loading, at most slots x max context\n"
       "- qwen38, every expert in VRAM (TP4): all the VRAM left after loading, at most slots x max context\n"
       "- qwen38, some experts cold (TP1 / TP2): the experts get the VRAM; the pool is min(slots, 4) contexts of at most the trained length, at least one max context\n"
       "\n"
       "Gives (a larger number): + room for long conversations running side by side.\n"
       "Takes (a larger number): - VRAM. On qwen38 TP1 / TP2 it comes out of the hot experts, so decode and prompt reading slow down.\n"
       "Note: when the pool runs out, the newest stream pauses and recomputes later. Nothing is lost."},
      {"XEN_KV",
       "Precision of the KV cache. It decides how many tokens fit in the pool.\n"
       "\n"
       "`int8`  half the memory of bf16\n"
       "`bf16`  full precision\n"
       "`int8-v4`  keys int8, values int4: about a quarter less memory than int8 (qwen38 and the 27B)\n"
       "\n"
       "Gives (int8): + KV pool: twice the tokens of bf16.\n"
       "Takes (int8): - nothing noticeable: the quality change is within run-to-run noise.\n"
       "Gives (int8-v4): + KV pool on the 27B (about 1.3x the tokens of int8) and + decode speed in long conversations. On qwen38 the freed VRAM goes to hot experts.\n"
       "Takes (int8-v4): - prompt reading speed on the 27B, - a small numerics change, - on the 27B no decode page selection (`XEN_ATTN_SELECT`); the trained page selection and the long-prompt page selection run on it."},
      {"XEN_KV_SPLIT",
       "qwen38 on four cards: how the two cards that share a KV head store its KV cache.\n"
       "\n"
       "`heads`  both cards hold every token of the head (each card stores half of the model's KV)\n"
       "`tokens`  each card holds every other token (each card stores a quarter)\n"
       "\n"
       "Gives (tokens): + KV pool: about 1.75x the tokens per GiB, so longer or more conversations fit.\n"
       "Takes (tokens): - a little decode speed (one more exchange between the two cards per attention layer), - a small numerics change.\n"
       "Needs: four cards (XEN_Q38_TP 4)."},
      {"XEN_HOT",
       "MoE models: how many routed experts live in GPU memory (hot), counted over ALL layers.\n"
       "\n"
       "`10000`  10,000 experts in VRAM, each split over the cards\n"
       "`empty`  auto: as many as fit (`0` means the same)\n"
       "\n"
       "The other experts stay in pinned host RAM and are read over PCIe when needed, which is slower. They must fit the host's free RAM.\n"
       "\n"
       "Gives (more hot experts): + decode speed and + prompt reading speed.\n"
       "Takes (more hot experts): - VRAM, so a smaller KV pool.\n"
       "Note: a count past what fits is cut at load. More would make the driver page memory, and every step would run many times slower.\n"
       "Memory: the plan below shows what a value costs."},
      {"XEN_HOT_RANK",
       "MoE models: the ranking that decides WHICH experts are hot at startup: the most used ones.\n"
       "\n"
       "The server records it while it serves. It counts the experts your requests use, saves the file every 60 s of decoding and at exit, and the next start puts the most used experts in GPU memory.\n"
       "\n"
       "`auto`  this model's file in `.cache/experts/`\n"
       "`a path`  that file (also recorded)\n"
       "`empty`  the default placement, nothing recorded\n"
       "\n"
       "Note: nothing is needed before the first start. That run uses the default placement and creates the file.\n"
       "Tip: `scripts/make-hot-rank.sh` (the Experts tab) seeds it from sample chats. Delete the file to start over."},
      {"XEN_MOE_CPU",
       "qwen38: the CPU computes part of the experts that did not fit in GPU memory (cold experts), while the cards compute the rest.\n"
       "\n"
       "A cold expert normally crosses PCIe to reach its card. The CPU reads it from RAM itself.\n"
       "\n"
       "`off`  the cards compute every expert\n"
       "`auto`  the best instruction set this CPU has; a CPU with neither stays off\n"
       "`avx512`  AVX-512 BF16: AMD Zen 4 / 5, Intel Xeon 4th gen and later\n"
       "`avx2`  AVX2 + FMA: most x86 CPUs since 2013\n"
       "\n"
       "Gives: + decode speed when part of the experts are cold (big contexts, a fresh ranking).\n"
       "Takes: - CPU: one block of cores per card, fully busy while decoding.\n"
       "Output: changes in the last bits (another summation order)."},
      {"XEN_MOE_CPU_THREADS",
       "Threads per card for the CPU path. Each card has its own pool.\n"
       "\n"
       "Tip: 8 + 8 fills a 16-core CPU. Use fewer on a smaller one: 6 on 12 cores."},
      {"XEN_MOE_CPU_CORES",
       "First CPU core of each card's pool, written `A,B`: card 0 from core A, card 1 from core B.\n"
       "\n"
       "`empty`  core 0, then one block of cores per card\n"
       "`-1`  not pinned"},
      {"XEN_MOE_CPU_PCT",
       "How many of each layer's cold experts belong to the CPU, as a fixed set chosen at start-up. The cards read the rest over PCIe at the same time.\n"
       "\n"
       "Gives (higher): + work taken off the PCIe link.\n"
       "Takes (higher): - CPU time per step; at 100 the PCIe link sits idle while the CPU is the bottleneck.\n"
       "Tip: 70 is a good start.\n"
       "Output: a fixed set keeps a prompt's output the same alone and under load."},
      {"XEN_MOE_CPU_MIN_COLD",
       "A sequence hands its CPU experts over only when it uses at least this many of them in a layer.\n"
       "\n"
       "Why: one cold expert per layer is read by the card while it computes the in-memory ones, about as fast as the CPU would be.\n"
       "Output: counted per sequence, so other requests never change a prompt's output."},
      {"XEN_MOE_CPU_MIN_ROWS",
       "Decode steps with fewer rows than this keep all cold experts on the cards.\n"
       "\n"
       "Output: above 1, a prompt's output can change with the load."},
      {"XEN_MOE_CPU_TIMEOUT_MS",
       "How long a card waits for the CPU per layer.\n"
       "\n"
       "If the CPU does not answer in time (an overloaded host), the card computes those experts itself and the CPU path switches off until the next start."},
      {"XEN_RESERVE",
       "GiB of VRAM kept free on each card after loading.\n"
       "\n"
       "Gives: + room for other programs that need GPU memory, and for the driver.\n"
       "Takes: - hot experts on qwen38 (slower decode and prompts), - auto KV pool on qwen35 / llama / flint.\n"
       "\n"
       "Tip: raise it if something else needs GPU memory."},
      {"XEN_MTP",
       "qwen38: the MTP drafter checkpoint for speculative decoding. It guesses several tokens and the model checks them in one step.\n"
       "\n"
       "`empty`  off\n"
       "\n"
       "Gives: + decode speed at 1-2 streams.\n"
       "Takes: - decode speed from about 4 streams (the checks crowd the step), - VRAM: about 0.7 GiB per card, taken from the hot experts.\n"
       "Output: the same.\n"
       "Tip: best for 1-2 streams."},
      {"XEN_DRAFTER",
       "Speculative decoding: a drafter guesses the next tokens and the model checks them in one step.\n"
       "\n"
       "`on`  the checkpoint's own MTP layer (qwen35), or the drafter named in `XEN_DRAFTER_MODEL`\n"
       "`off`  one token per step\n"
       "\n"
       "Gives: + decode speed, most with few streams and long contexts.\n"
       "Takes: - VRAM for the drafter.\n"
       "KV context (qwen35): the MTP layer keeps its own KV for every token, so the pool holds fewer tokens.\n"
       "Output: the same.\n"
       "Note: flint has no MTP layer and needs `XEN_DRAFTER_MODEL`."},
      {"XEN_DRAFTER_MODEL",
       "A separate block drafter (DFlash 2 or DSpine) instead of the MTP drafter. It guesses up to 7 tokens per step, DSpine b16 up to 15.\n"
       "\n"
       "`empty`  the MTP drafter\n"
       "\n"
       "- qwen35 with DFlash 2 (`qwen38-27b-dflash2`): faster than the MTP layer on every kind of text, most on maths.\n"
       "- qwen35 with DSpine (`qwen38-27b-dspine-b16`, 16-token blocks): with `XEN_SEG_ROWS` 16 one stream drafts the whole block. The shipped 27B settings use it. It reads the target's output head: a bf16 head, or an int4-only GPTQ head (`swift15-27b-int4-baked`), dequantised a chunk at a time at load.\n"
       "- flint (`flint-dflash2`, its only drafter): a clear gain at 1-2 streams, none from about 4.\n"
       "- qwen38 (`qwen38-flash-next-dflash2`, replaces `XEN_MTP`): ahead on reasoning-heavy requests, behind MTP on plain chat.\n"
       "\n"
       "Gives: + decode speed (more tokens accepted per step than the MTP drafter).\n"
       "Takes: - VRAM per card, for the drafter and for its context in every slot: the 27B at 16 slots about 2.0 GiB with DFlash 2 and 3.6 GiB with DSpine; flint about 1 GiB; qwen38 about 0.5 GiB.\n"
       "KV context (qwen35 / flint): the drafter's VRAM comes out of the auto KV pool.\n"
       "Output: the same."},
      {"XEN_DRAFTER_DEFER",
       "With a block drafter: on steps where the drafter does not guess (a busy server), its catch-up work is batched per slot instead of done every step.\n"
       "\n"
       "`0`  catch up every step\n"
       "\n"
       "Gives: + decode speed with many streams (shorter steps).\n"
       "Takes: - nothing.\n"
       "Output: the same."},
      {"XEN_DRAFTER_MTP_FROM",
       "With a block drafter: also load the checkpoint's MTP layer, and let it draft once this many streams decode at once. Below that, the block drafter drafts.\n"
       "\n"
       "`0`  the block drafter only; the MTP layer is not loaded\n"
       "`5`  the MTP layer drafts from 5 streams\n"
       "`9`  the default. With DSpine: DSpine through 8 streams, the MTP layer at 9-16\n"
       "\n"
       "Gives: + decode speed at the stream counts where one MTP pass is cheaper than a block-drafter pass (many streams, wide steps).\n"
       "Takes: - VRAM: the MTP layer takes about 0.5 GiB per card. - a little decode speed on the block drafter's steps, where the MTP layer still writes its KV.\n"
       "KV context: the MTP layer also keeps its own KV for every token, so the pool holds fewer tokens.\n"
       "Output: the same.\n"
       "Tip: with 8 slots, `0` behaves like `9` without loading the MTP layer.\n"
       "\n"
       "qwen38: both drafters load when `XEN_MTP` and `XEN_DRAFTER_MODEL` are both set and this is above `0`. It pays with more than 16 slots and `XEN_STEP_ROWS` 64 (for example `13`); with 16 slots DFlash alone is as fast."},
      {"XEN_DRAFTER_INT4",
       "Keeps the block drafter's large matrices as 4-bit instead of 8-bit numbers, rounded when the server loads.\n"
       "\n"
       "`on`  4-bit\n"
       "`off`  8-bit\n"
       "\n"
       "Gives: + decode speed (a cheaper guessing pass), + VRAM: about 0.35 GiB per card more for the auto KV pool.\n"
       "Takes: - nothing noticeable: it guesses as well as with 8 bits.\n"
       "Output: the same."},
      {"XEN_DRAFTER_CUT",
       "With a block drafter, one stream: stop checking guesses once their confidence falls below this, so steps carry fewer rows.\n"
       "\n"
       "`0`  check every guess\n"
       "\n"
       "Gives: + decode speed at one stream.\n"
       "Takes: - nothing.\n"
       "Output: the same."},
      {"XEN_DRAFTER_TREE",
       "With a block drafter: guess a small tree of alternatives (up to this many guesses) instead of one line, and keep the longest branch the model agrees with. Greedy and sampled requests.\n"
       "\n"
       "`0`  one line of guesses (a chain)\n"
       "\n"
       "Gives: + decode speed at few streams (more tokens accepted per step).\n"
       "Takes: - nothing: it runs only at the stream counts of `XEN_DRAFTER_TREE_SEQS`; more streams keep chains.\n"
       "Qwen3.8-27B: any KV cache (int8, bf16 or int8-v4). With `XEN_ATTN_SELECT`, a conversation past the page count drafts chains, and so do the ones generating beside it.\n"
       "Flash-Next: any KV cache and context, from 2 streams (one stream keeps the line with `XEN_DRAFTER_CUT`, which is as fast there). Chains when the MTP layer drafts beside DFlash (`XEN_DRAFTER_MTP_FROM`).\n"
       "Output: the same."},
      {"XEN_DRAFTER_TREE_SEQS",
       "Use the draft trees (`XEN_DRAFTER_TREE`) while up to this many streams decode at once, each stream with its own tree.\n"
       "\n"
       "`1`  trees at one stream only\n"
       "\n"
       "Gives: + decode speed at 2 streams on code and chat.\n"
       "Takes: - nothing at higher stream counts: they keep chains.\n"
       "Output: the same."},
      {"XEN_STEP_ROWS",
       "The most rows one decode step runs over all streams. A row is one token being checked.\n"
       "\n"
       "`0`  the default: the model's. 32 on qwen35. qwen38: 64 on four cards, 16 on one or two\n"
       "`64`  qwen38 on four cards only: 9 and more streams guess deeper in one 64-row step\n"
       "`32`  6-16 streams guess deeper in one wider step\n"
       "`16`  narrow steps\n"
       "`8`  every step on the 8-row kernels; needs `XEN_SLOTS` 8 or fewer\n"
       "\n"
       "Gives (wider): + tokens per second in all from about 6 streams. Few streams run the narrow steps either way.\n"
       "Takes (wider): - VRAM for the step's buffers (`64`: about 0.4 GiB more per card, so a slightly smaller KV pool).\n"
       "qwen38 on four cards: 1-5 streams run the 16-row steps, 6-8 streams the 32-row ones, 9 and more the 64-row ones.\n"
       "Tip: `32` on four cards gives the KV pool that memory back; 9 and more streams then decode slower.\n"
       "Output: the same.\n"
       "Note: llama and flint always run 8."},
      {"XEN_SEG_ROWS",
       "The most rows ONE stream's check holds in a step.\n"
       "\n"
       "`8`  up to 7 guesses per stream: a drafter trained on 8-token blocks (DFlash 2)\n"
       "`16`  the default. Up to 15 guesses per stream: a drafter trained on 16-token blocks (DSpine b16)\n"
       "\n"
       "Gives (16): + decode speed at one stream with a 16-token drafter.\n"
       "Takes (16): - VRAM: 24 MiB per slot and card, so the KV pool gets smaller (0.38 GiB at 16 slots). A 1M max context may no longer fit.\n"
       "Note: llama and flint always run 8.\n"
       "Output: the same.\n"
       "Tip: 16 is what lets `XEN_DRAFT_DEPTHS` and `XEN_DRAFTER_WIDE` draft up to 15 tokens."},
      {"XEN_DRAFT_DEPTHS",
       "Guesses per stream when 1, 2, 3, ... streams decode at once, comma-separated.\n"
       "\n"
       "`empty`  the engine's built-in table. 32-row steps: 7, 7, 4, 3, 3, 3, 3, 3, 2, 2 at 1-10 streams, then 1.\n"
       "`15,7,4,3`  one stream always checks 16 rows\n"
       "\n"
       "Gives (deeper): + tokens per step when the guesses land.\n"
       "Takes (deeper): - time per step, wasted when they miss.\n"
       "Limits: up to 7 guesses with `XEN_SEG_ROWS` 8, 15 with 16. Streams x (guesses + 1) must fit `XEN_STEP_ROWS`. The Qwen3.8-27B's MTP drafter alone (no `XEN_DRAFTER_MODEL`) guesses at most 6; deeper entries run at 6.\n"
       "Note: the adaptive depth picks at or below these. flint runs 8-row steps: `7,3,1,1`.\n"
       "Output: the same."},
      {"XEN_DRAFTER_WIDE",
       "DSpine with `XEN_SEG_ROWS` 16, one stream: after a step that kept this many of its guesses or more, guess the whole 16-token block instead of 8 rows. Easy runs (maths) then take bigger steps.\n"
       "\n"
       "`0`  the depth table only\n"
       "\n"
       "Gives: + decode speed at one stream on predictable text (most on maths).\n"
       "Takes: - nothing on other text: the wide step runs only after a good one.\n"
       "Output: the same.\n"
       "Note: the block is a tree of `XEN_DRAFTER_TREE_WIDE` guesses, or a chain when trees are off."},
      {"XEN_DRAFTER_TREE_WIDE",
       "Guesses in the 16-row tree of `XEN_DRAFTER_WIDE`, up to 15."},
      {"XEN_UNEQUAL_SEGMENTS",
       "Several streams: each stream gets its own number of guesses per step, instead of one count for all. The streams whose guesses land best guess; the rest run one token in the same step.\n"
       "\n"
       "`on`  a count per stream\n"
       "`off`  one count for all\n"
       "\n"
       "Gives: + tokens per second with many streams (most at 9-14 of 16 slots): free rows of a step go to the streams that guess best.\n"
       "Takes: - nothing.\n"
       "Applies: qwen38 with DFlash, and the dense 27B with its drafter.\n"
       "Output: the same."},
      {"XEN_ADAPTIVE_DEPTH",
       "Speculative decoding picks how many tokens to guess per step from how well the recent guesses landed and how long steps take.\n"
       "\n"
       "`on`  adapt\n"
       "`off`  a fixed depth\n"
       "\n"
       "Gives: + decode speed at one stream with sampling.\n"
       "Takes: - nothing.\n"
       "Output: the same."},
      {"XEN_LOOKUP",
       "On steps where it has recently guessed better than the drafter, the guesses come from the request's own text (what followed the last time its latest N tokens appeared) and the drafter is skipped.\n"
       "\n"
       "`0`  off\n"
       "`4`  N = 4 tokens, the default\n"
       "\n"
       "Gives: + decode speed when the reply copies from the prompt (rewriting a file, quoting).\n"
       "Takes: - nothing on other text.\n"
       "Output: the same."},
      {"XEN_MTP_DEPTH",
       "How many tokens the MTP drafter guesses per step for a single stream (1-3; Qwen3.8-27B: 1-6).\n"
       "\n"
       "Gives (more): + tokens per step when the guesses land.\n"
       "Takes (more): - time per step.\n"
       "Note: with adaptive depth on, this is its upper limit.\n"
       "Output: the same."},
      {"XEN_DRAFT_VOCAB",
       "The drafter only proposes the most common token ids (the first N plus the special tokens), which makes its guesses cheaper. The model still checks with its full vocabulary.\n"
       "\n"
       "`0`  the full vocabulary\n"
       "\n"
       "Gives: + decode speed (a cheaper guessing pass).\n"
       "Takes: - guesses for rare tokens (98,304 ids cover nearly all text on qwen35), - VRAM on the 27B with the MTP drafter alone: its own copy of the output rows, 0.23 GiB per card at 98,304.\n"
       "Note: qwen38 needs a multiple of 64.\n"
       "Output: the same."},
      {"XEN_PREFILL_MOE",
       "qwen38: precision of the expert activations while reading the prompt.\n"
       "\n"
       "`w4a16`  the same numerics as generation\n"
       "`w4a8`  int8 activations\n"
       "\n"
       "Gives (w4a8): + prompt reading speed (a little).\n"
       "Takes (w4a8): - exactness: the output changes slightly."},
      {"XEN_PREFILL_DENSE",
       "qwen38: precision of the dense layers' activations while reading the prompt.\n"
       "\n"
       "`w8a8-row`  int8 per row\n"
       "`w8a16`  bf16: the generation path's numerics\n"
       "`w8a8`  int8 per 128-k group (slower than per row)\n"
       "\n"
       "Gives (w8a8-row): + prompt reading speed.\n"
       "Takes (w8a8-row): - exactness: a small numerics change against `w8a16`."},
      {"XEN_DECODE_MOE",
       "qwen38: precision of the expert activations while generating.\n"
       "\n"
       "`w4a16`  bf16 activations\n"
       "`w4a8`  int8 activations: for checkpoints calibrated for it\n"
       "`auto`  by the checkpoint: int8 if it was calibrated for it\n"
       "\n"
       "Note: on a W4A16 checkpoint `w4a8` gives no speed and changes the numerics."},
      {"XEN_GDN_W4",
       "qwen38, opt-in: format of the GDN (linear attention) projections.\n"
       "\n"
       "`int8`  converted at load\n"
       "`int4`  the folder's GPTQ int4 tensors (a gdn4-calib checkpoint)\n"
       "`int4-qkvz`  in_proj int4 only\n"
       "\n"
       "Gives (int4): + decode speed in long contexts.\n"
       "Takes (int4): - quality: further from the full-precision model than int8. `int4-qkvz` sits between the two on both."},
      {"XEN_TP",
       "How many cards the model runs on.\n"
       "\n"
       "`2`  both cards; the cards exchange sums every layer\n"
       "`1`  one card (`XEN_CARD`); the other stays free for a second server\n"
       "\n"
       "Gives (2): + decode speed, + prompt reading speed, + KV pool.\n"
       "Takes (2): - both cards.\n"
       "KV context: with one card the whole model sits on it, so far fewer KV tokens fit."},
      {"XEN_CARD",
       "With one card (`XEN_TP` 1): which card the server runs on, 0 or 1.\n"
       "\n"
       "Tip: two servers on different cards and ports serve two copies of the model.\n"
       "Note: each card count keeps its own settings in `models/<id>.tp<N>.env`. Changing this loads that file; `s` saves into it."},
      {"XEN_EMU_CARD",
       "Runs as another card on these B70s, for testing without it.\n"
       "\n"
       "`off`  the real cards\n"
       "`b60`  Arc Pro B60: 20 Xe cores, 24 GB\n"
       "`b65`  Arc Pro B65: 20 Xe cores, 32 GB\n"
       "\n"
       "What changes: the kernels plan for its cores, and every budget (KV pool, hot experts, context fit) sees its memory.\n"
       "What does not: the speed stays the B70's. A card cannot hand cores, bandwidth or clock to an emulation."},
      {"XEN_Q38_TP",
       "qwen38: how many cards it runs on (kept apart from qwen35's setting).\n"
       "\n"
       "`2`  cards 0 and 1\n"
       "`4`  cards 0-3, every matrix and expert in quarters\n"
       "`1`  one card (`XEN_Q38_CARD`), every expert whole on it\n"
       "\n"
       "Gives (4): + decode speed, + prompt reading speed, + KV pool: every expert fits in VRAM.\n"
       "Takes (4): - all four cards. No CPU expert path (nothing is cold).\n"
       "Gives (1): + a free card for a second server.\n"
       "Takes (1): - decode and prompt reading speed (far fewer experts fit in VRAM), - KV pool.\n"
       "Note: each card count keeps its own settings in `models/<id>.tp<N>.env`. Changing this loads that file; `s` saves into it."},
      {"XEN_Q38_CARD",
       "qwen38 with one card: which card it runs on, 0 or 1."},
      {"XEN_PREFILL_WIRE",
       "Precision of the numbers the cards exchange while reading a prompt.\n"
       "\n"
       "`auto`  bf16 for qwen35, fp32 for qwen38\n"
       "`bf16`  half the traffic\n"
       "`fp32`  exact sums\n"
       "\n"
       "Gives (bf16): + prompt reading speed.\n"
       "Takes (bf16): - exactness: a small numerics change."},
      {"XEN_YARN_TIERS",
       "qwen35, qwen38 and Nanbeige, for a max context past the trained length (262,144 tokens; Nanbeige 524,288): YaRN per request. flint needs no YaRN.\n"
       "\n"
       "`on`  a request gets the factor of the smallest tier that holds its prompt plus room to grow. Tiers: the trained length at 1.00, twice it at 2.00 (and so on, doubling), then the max context.\n"
       "`off`  one factor for every request: the max context's\n"
       "\n"
       "Gives (on): + quality for chat-sized requests: they keep the trained model's numerics, and long ones still reach the max context.\n"
       "Takes (on): - time once per request that grows past its tier: it re-reads its tokens in the next tier."},
      {"XEN_WIRE",
       "Precision of the numbers the cards exchange while generating.\n"
       "\n"
       "`bf16`  half the traffic of fp32\n"
       "`fp32`  exact sums\n"
       "`int8`  8-bit codes per group of 32: about half of bf16's bytes (dense 27B only)\n"
       "\n"
       "Gives (bf16): + decode speed with several streams.\n"
       "Takes (bf16): - exactness: a very small numerics change.\n"
       "Gives (int8): + decode speed, most on wide steps (5-16 streams).\n"
       "Takes (int8): - exactness: a little further from fp32 than bf16.\n"
       "Tip: int8 is your choice. It is off by default."},
      {"XEN_ATTN_SELECT",
       "qwen35, opt-in: in very long conversations, attend only to the most relevant 64-token pages (the first, the last 16 and the best-scoring rest) instead of all of them.\n"
       "\n"
       "`0`  attend to everything\n"
       "`1024`  the mildest: for conversations past 65K tokens\n"
       "`512`  past 32K tokens\n"
       "`128`  the strongest: past 8K tokens\n"
       "\n"
       "Gives: + decode speed in long conversations, more the longer they are and the fewer pages are kept.\n"
       "Takes: - exactness: the output changes once a conversation passes that many pages (fewer pages = further from full attention). - draft trees: such a conversation drafts chains, and so do the ones generating beside it. - a little KV pool: every KV page also stores its key bounds.\n"
       "Note: a conversation within 1.25x of the page count gains nothing; a shorter one runs as with `0`. Runs with `XEN_DECODE_ATTN` auto, exact or int8-fast.\n"
       "Needs: an int8 or bf16 KV cache; it has no int8-v4 kernel and the server refuses it there. On int8-v4 the Qwen3.8-27B's trained page selection (`XEN_TRAINED_QSA_DELTA_DIR`) reads the 512 best four-token blocks instead."},
      {"XEN_SELECT_HEADS",
       "qwen35: with `XEN_ATTN_SELECT` above 128, how the pages are shared among the attention heads.\n"
       "\n"
       "`auto`  each request measures its own heads on its first 8 selected steps: heads whose attention is spread over the text then keep more pages, sharp ones fewer (never under 128); the average stays the setting\n"
       "`same`  every head keeps the setting's count\n"
       "\n"
       "Gives (auto): + quality: output closer to full attention at the same step time.\n"
       "Takes (auto): - nothing.\n"
       "Note: nothing to calibrate and no file; it works for any checkpoint. `XEN_PREFILL_SELECT` keeps the same count for every head."},
      {"XEN_PREFILL_SELECT",
       "qwen35, opt-in: while a very long prompt is read, its rows attend only to the most relevant 64-token pages (the first, the last 16 and the best-scoring rest) instead of all of them. The prompt's last rows always read everything.\n"
       "\n"
       "`0`  read everything\n"
       "`1024`  the mildest: for prompts past 65K tokens\n"
       "`512`  past 32K tokens\n"
       "`128`  the strongest: past 8K tokens\n"
       "\n"
       "Gives: + prompt reading speed on long prompts, more the longer they are and the fewer pages are kept.\n"
       "Takes: - exactness: the stored prompt differs from a fully read one once it passes that many pages, and a cached prefix keeps the way it was read. - a little KV pool: every KV page also stores its key bounds.\n"
       "Note: a prompt within 1.25x of the page count gains nothing; a shorter one is read as with `0`. Runs with `XEN_PREFILL_ATTN` auto or exact, on every KV cache (int8-v4 included). Beside the trained page selection it is the faster way to read long prompts: `256` there, and the trained prompt reading (`XEN_TRAINED_QSA_PREFILL_XMX`) stays off."},
      {"XEN_W4_ITEMS",
       "qwen35: how the int4 matrix work is split across the GPU while generating.\n"
       "\n"
       "`1024`  larger work items\n"
       "`512`  smaller work items\n"
       "\n"
       "Gives (1024): + decode speed (slightly).\n"
       "Output: 512 and 1024 give slightly different bits."},
      {"XEN_DECODE_ATTN",
       "Dense 27B: how attention is computed while generating.\n"
       "\n"
       "`auto`  `int8-fast` where it can run: an int8 KV cache. Otherwise `exact`.\n"
       "`exact`  full-precision arithmetic\n"
       "`fast`  the attention weights as one bf16 number instead of two\n"
       "`int8`  8-bit queries against the 8-bit keys of the KV cache\n"
       "`int8-fast`  both shortcuts\n"
       "\n"
       "Gives (int8-fast): + decode speed in long conversations, more the longer they are. Draft trees gain the same.\n"
       "Takes (int8-fast): - a little decode speed in short conversations, - exactness: the output changes slightly.\n"
       "Needs: `int8` and `int8-fast` need an int8 KV cache. With page selection (`XEN_ATTN_SELECT`) only `auto`, `exact` and `int8-fast` run."},
      {"XEN_DECODE_W4",
       "Dense 27B: numerics of the model's int4 matrices while generating.\n"
       "\n"
       "`exact`  bf16 activations: full-precision arithmetic\n"
       "`a8`  int8 activations, as the prompt path's `a8` (`XEN_PREFILL_W4`)\n"
       "\n"
       "Gives (a8): + decode speed on wide steps (16-32 rows: several streams, or one stream drafting deep).\n"
       "Takes (a8): - quality: clearly further from `exact` than the other shortcuts.\n"
       "Output: changes. A request's reply stays the same whatever else the server decodes.\n"
       "Note: the drafters and the output layer keep bf16 activations."},
      {"XEN_PREFILL_ATTN",
       "qwen35 / flint: how attention is computed while reading the prompt.\n"
       "\n"
       "`auto`  the fastest safe choice for the KV precision: `int8-fast` on an int8 KV, `fast` on bf16 and on flint\n"
       "`exact`  the generation path's arithmetic\n"
       "`fast`  the attention weights as one bf16 number instead of two\n"
       "`int8`  8-bit queries against the 8-bit keys\n"
       "`int8-fast`  both shortcuts\n"
       "`onednn`  oneDNN's f16 attention (qwen35, opt-in)\n"
       "\n"
       "Gives (the shortcuts): + prompt reading speed on long prompts.\n"
       "Takes (the shortcuts): - exactness: a very small numerics change.\n"
       "Gives (onednn): + exact-class numerics at the shortcuts' speed on long prompts.\n"
       "Takes (onednn): - KV pool: its scratch takes about 0.3 GiB of VRAM per card at a 262K context.\n"
       "Needs: `int8` and `int8-fast` need an int8 KV cache."},
      {"XEN_ATTN_TAIL",
       "qwen35 / llama: how the prompt's attention splits its work over the context when a call does not fill whole waves of GPU work-groups.\n"
       "\n"
       "`2`  as 1, and a short call (under ~640 rows on two cards: a short resumed turn, a tool result) splits every work-group\n"
       "`1`  only the work-groups past a long call's last full wave are split\n"
       "`0`  no splits\n"
       "\n"
       "Gives (2): + speed of short prompt calls in long conversations (a faster first token on resumed turns).\n"
       "Takes (2): - bit-exactness: a short call's sums depend on the call's size, so the continuation of a cached prefix is not bit-identical to a cold read.\n"
       "Gives (1): + prompt reading speed on long calls.\n"
       "Takes (1): - nothing: rows keep their bits at every call size below a wave."},
      {"XEN_PREFILL_SHORT",
       "qwen35: a prompt call of at most N rows (a short resumed turn, a tool result's tail) runs its int4 GEMMs on the generation kernels. The prompt kernels have a larger fixed cost per call.\n"
       "\n"
       "`32`  calls of 1-32 rows\n"
       "`16`  calls of 1-16 rows\n"
       "`0`  off: every call on the prompt kernels\n"
       "\n"
       "Gives: + speed of short prompt calls (a faster first token on resumed turns).\n"
       "Takes: - sameness of numerics: those rows take the generation path's (int4 x bf16 instead of int8 activations)."},
      {"XEN_PREFILL_GEMM_ROWS",
       "Dense 27B with the oneDNN prefill library: the rows per call from which the wide matrices use it. A prompt of 512+ tokens runs as two half calls.\n"
       "\n"
       "`1024`  prompts of 2,048+ tokens (the default)\n"
       "`512`  also 1,024-token prompts\n"
       "`256`  also 512-token prompts\n"
       "`128`  also 128-384-token prompts\n"
       "\n"
       "Gives (lower): + prompt reading speed for those shorter prompts, while KV lending keeps the layouts (`XEN_KV_LEND`).\n"
       "Takes (lower): - the same prompts get slower instead when a full KV pool takes the layouts back. - a small numerics change: shorter prompts take the library's rounding.\n"
       "Tip: your choice."},
      {"XEN_PREFILL_GEMM",
       "Which kernels run the prompt's large matrix products.\n"
       "\n"
       "`onednn`  library GEMMs where they win, repacked per call (no extra weight copy)\n"
       "`onednn-all`  qwen35: the library for every int4 matrix of the prompt, the narrow ones too (down, out, o)\n"
       "`esimd`  the engine's own kernels only\n"
       "\n"
       "Gives (onednn): + prompt reading speed.\n"
       "Takes (onednn): - VRAM on qwen38: about 0.12 GiB per card, taken from the hot experts.\n"
       "Gives (onednn-all): + prompt reading speed on the dense 27B: +4.8 % at 2K, +3.1 % at 8K, +2.9 % at 32K over `onednn` (two cards, KV lending on).\n"
       "Takes (onednn-all): - a small numerics change: the narrow matrices take the library's rounding. - KV lending keeps 128 more layouts (about 1.8 GiB per card of free KV pages; given back when the pool fills).\n"
       "- qwen35 (with `XEN_PREFILL_W4` a8): the wide int4 matrices run on oneDNN. Calls under `XEN_PREFILL_GEMM_ROWS` rows stay on esimd.\n"
       "- qwen38: its int8 dense matrices run on oneDNN with prefill chunks of 2,048+ (same output). With smaller chunks it stays off."},
      {"XEN_PREFILL_W4",
       "qwen35 / flint: numerics of the int4 matrices while reading the prompt.\n"
       "\n"
       "`a8`  int8 activations: the fastest\n"
       "`dequant`  bf16 activations, a little faster than exact\n"
       "`exact`  the generation path's arithmetic\n"
       "\n"
       "Gives (a8): + prompt reading speed (the largest single gain for prompts).\n"
       "Takes (a8): - exactness: a small numerics change. - KV pool on the 27B: 0.18 GiB of VRAM per card for its buffers, which `exact` leaves to the auto KV pool.\n"
       "Note: flint takes exact or dequant; a8 runs as dequant there."},
      {"XEN_PREFILL_A8_BODY",
       "qwen35 with a8 prefill: the engine kernel that runs the int4 x int8 GEMMs.\n"
       "\n"
       "`auto`  slm with two cards (direct for calls of up to 128 rows), direct with one\n"
       "`slm`  stages weights in shared memory. Best on two cards, where the GEMMs run while the cards exchange rows.\n"
       "`direct`  no shared memory, weights and activations prefetched. Best on one card.\n"
       "\n"
       "Output: the same bit for bit."},
      {"XEN_PREFILL_A8_KEEP",
       "qwen35 with a8 prefill: the matrices that stay at full activation precision, joined by `+`.\n"
       "\n"
       "`empty`  all int8\n"
       "`gdn+attn`  int8 only in the MLP\n"
       "\n"
       "Gives (keeping more): + exactness: closer to the `exact` prompt path.\n"
       "Takes (keeping more): - prompt reading speed: part of a8's gain.\n"
       "Names: `qkvz`, `out`, `gdn`, `qkv`, `o`, `attn`, `gu`, `down`, `mlp`, `all`. `FAMILY@L0-L1` picks a range of layers."},
      {"XEN_LM_HEAD",
       "qwen35: format of the output layer.\n"
       "\n"
       "`auto`  the best the folder offers: int4-rescore with both heads, int4 when int4-only, else int8\n"
       "`int8`  converted at load\n"
       "`int4`  the 4-bit head alone\n"
       "`int4-rescore`  int4 scores, the top candidates rescored in int8: the same picks as int8\n"
       "\n"
       "Gives (int4): + decode speed, + VRAM: 0.5 GiB per card more for the auto KV pool than `int8`.\n"
       "Takes (int4): - quality: the 4-bit scores pick differently.\n"
       "Gives (int4-rescore): + decode speed at int8's quality.\n"
       "Takes (int4-rescore): - VRAM: it keeps both heads, 0.1 GiB per card more than `int8`."},
      {"XEN_LM_HEAD_TILES",
       "qwen35 with int4-rescore: how many candidate tiles per card get rescored in int8.\n"
       "\n"
       "Gives (more): + quality: closer to the int8 head, at the same speed.\n"
       "Takes (more): - nothing noticeable."},
      {"XEN_DRAFT_HEAD",
       "qwen35: format of the drafter's output rows.\n"
       "\n"
       "`auto`  like the main output layer (`XEN_LM_HEAD`)\n"
       "`int8`  8-bit rows\n"
       "`int4`  4-bit rows\n"
       "\n"
       "Gives (int4): + decode speed at one stream.\n"
       "Takes (int4): - nothing in the output: it only affects guesses.\n"
       "Output: the same."},
      {"XEN_GRAPHS",
       "qwen38 / flint: replays pre-recorded GPU step graphs instead of launching each kernel separately.\n"
       "\n"
       "`on`  replay the graphs\n"
       "`off`  plain launches (for debugging)\n"
       "\n"
       "Gives: + decode speed (less launch overhead per step).\n"
       "Takes: - load time and a little VRAM for the recorded graphs.\n"
       "Output: identical."},
      {"XEN_PREFILL_CHUNK",
       "How many prompt tokens are processed in one go.\n"
       "\n"
       "`0`  the model's default: qwen38 4,096, qwen35 2,048\n"
       "\n"
       "Gives (bigger): + prompt reading speed on long prompts.\n"
       "Takes (bigger): - VRAM for the prompt buffers.\n"
       "KV context (qwen35): the buffers come out of the auto KV pool: against 2,048, a chunk of 4,096 holds a few percent fewer tokens, 8,192 about a tenth fewer, 16,384 over a quarter fewer."},
      {"XEN_PREFILL_STALL_MS",
       "While other streams are generating, a new long prompt is read in pieces of about this many ms. The running streams then pause at most about that long per token.\n"
       "\n"
       "`100`  short pauses\n"
       "`0`  whole chunks: the longest pauses, the fastest first token\n"
       "\n"
       "Gives (lower, above 0): + smoothness of the running streams.\n"
       "Takes (lower, above 0): - time to first token of the new request."},
      {"XEN_PREFILL_PART_MS",
       "Splits a long prompt's processing into parts of about this many ms, with a generation step for the other streams in between, so running streams stay smooth.\n"
       "\n"
       "`-1`  the model's default (12-90 ms)\n"
       "`0`  no parts\n"
       "\n"
       "Gives (shorter parts): + smoothness and speed of the running streams.\n"
       "Takes (shorter parts): - time to first token of the new request, growing with the number of running streams.\n"
       "Output: identical."},
      {"XEN_PREFILL_MERGE",
       "qwen35: short prompts that are waiting together are read in one call instead of one call each. Most of a short call's time does not depend on its length, so a burst of new requests gets its first tokens sooner.\n"
       "\n"
       "`1`  a call per prompt\n"
       "`4`  up to 4 prompts per call\n"
       "`8`  up to 8\n"
       "`16`  up to 16\n"
       "\n"
       "Gives: + tokens per second in all and + mean time to first token when many short requests arrive at once.\n"
       "Takes: - time to first token of the earliest prompts of a small burst: the prompts of a shared call get their first token together.\n"
       "Output: identical. Each prompt's rows are computed as in its own call.\n"
       "Note: only prompts of 33 to 511 tokens share a call, up to 1,023 rows in all."},
      {"XEN_KV_LEND",
       "Puts the KV pool's unused pages to work. They are given back when a conversation grows into them, and refilled while the server is idle.\n"
       "\n"
       "`on`  lend the free pages\n"
       "`off`  the pool's free VRAM stays empty\n"
       "\n"
       "- qwen38: the pages hold extra experts in VRAM.\n"
       "- Dense 27B: the pages hold the prefill's oneDNN weight layouts, so a prompt does not rebuild them per call.\n"
       "\n"
       "Gives: + prompt reading speed; on qwen38 also + decode speed (more hot experts).\n"
       "Takes: - nothing from the KV pool: a conversation that needs the pages gets them back first. The gain shrinks as the pool fills.\n"
       "Output: identical."},
      {"XEN_KV_LEND_PARK_S",
       "qwen38 with KV lending: a finished conversation kept by the prefix cache that blocks lent experts is dropped after this many seconds. The idle server then lends its VRAM to experts again.\n"
       "\n"
       "`0`  keep it until the memory is needed\n"
       "\n"
       "Gives: + hot experts (speed) for the requests that come next.\n"
       "Takes: - that conversation's cache: its next turn reads its whole prompt again."},
      {"XEN_KV_LEND_PRESSURE",
       "qwen38 with KV lending: what gives way when a conversation grows into the lent experts.\n"
       "\n"
       "`kv`  the experts give way\n"
       "`lru`  older cached conversations are dropped first\n"
       "`idle`  only cached conversations idle longer than the grace time (`XEN_KV_LEND_GRACE_S`)"},
      {"XEN_KV_LEND_GRACE_S",
       "qwen38, lending pressure `idle`: how long a cached conversation is safe from being dropped for the experts."},
      {"XEN_EXPERT_EXCHANGE",
       "qwen38 with experts that do not all fit in VRAM (TP1 / TP2): the experts kept in VRAM follow what is being generated now. Every few decode steps "
       "the most-read experts left in host memory change places with the least-read ones in VRAM, copied in the background while the cards generate.\n"
       "\n"
       "`on`  the VRAM set follows the traffic\n"
       "`off`  the set placed at load (from the recorded ranking) stays\n"
       "\n"
       "Gives: + decode speed when the traffic differs from what the ranking was recorded on, more with more streams.\n"
       "Takes: - 32 experts' worth of VRAM per card (the spare slots the copies land in), - a little PCIe traffic beside the cold reads.\n"
       "Output: identical. Off by itself on four cards (every expert in VRAM) and with the MoE CPU path."},
      {"XEN_KV_LEND_KEEP",
       "With KV lending: tokens of KV kept free below the lent pages. Each time a conversation grows past it, more of what the pages hold is given back."},
      {"XEN_PREFILL_FETCH",
       "qwen38 on two cards with experts that do not all fit in VRAM: VRAM for this many experts per layer, three times per card. While a long prompt (4,096 tokens and more in one go) is read, each layer's most-used experts that live in host memory are copied to the card in the background while the layers before it compute, so the prompt's experts wait less on PCIe.\n"
       "\n"
       "`0`  off: the prompt reads those experts over PCIe in place\n"
       "`96`  enough for 4,096-token chunks (`XEN_PREFILL_CHUNK` 0 / 4096)\n"
       "`160`  for 8,192-token chunks\n"
       "\n"
       "Gives: + prompt reading speed on long prompts (32K tokens and more), much more with `XEN_PREFILL_CHUNK=8192` and `160`.\n"
       "Takes: - 3 x this many experts' worth of VRAM per card (160: 480 experts, 0.6 GiB) out of the experts kept in VRAM: - decode speed (most with several streams), - short prompts; 8,192-token chunks take 2.1 GiB more per card on top.\n"
       "Output: identical. Nothing to do on four cards (every expert in VRAM), on one card or with the MoE CPU path."},
      {"XEN_PREFILL_GRAPHS",
       "qwen38: a short prompt (under 128 tokens in one go, e.g. the next turn of a cached conversation) is read by replaying a recorded copy of its "
       "work on each card instead of starting it piece by piece. A prompt length is recorded the second time it comes (in any conversation; "
       "how long the conversation already is counts only in steps: up to 2K, 64K, then every 64K), and up to 32 are kept.\n"
       "\n"
       "`0`  off\n"
       "`1`  on\n"
       "\n"
       "Gives: + 1-2 ms per short prompt on two or four cards (2-5 %; nothing on one card).\n"
       "Takes: - 10-60 ms once per recorded length (the call that records it); it pays back only when the same lengths come again and again "
       "(benchmarks, fixed-size requests).\n"
       "Output: identical."},
      {"XEN_PREFILL_COLD_STAGE",
       "qwen38: VRAM for this many expert blobs per card, used to copy cold experts once per long prompt instead of reading them over PCIe again and again.\n"
       "\n"
       "`0`  off\n"
       "\n"
       "Gives: + prompt reading speed when experts are cold.\n"
       "Takes: - VRAM for the staged blobs.\n"
       "Output: identical."},
      {"XEN_TEMPERATURE",
       "How random the answers of requests without their own `temperature` are. Lower = more focused and repeatable; maths and code often do best low.\n"
       "\n"
       "`empty`  the model's recommended value (Qwen3.8: 1.0)\n"
       "`0`  greedy: always the most likely token\n"
       "`0.3`  focused (maths, code)\n"
       "\n"
       "A request's own value always wins."},
      {"XEN_TOP_P",
       "Requests without their own `top_p` sample only from the most likely tokens that together reach this probability.\n"
       "\n"
       "`empty`  the model's recommended value (Qwen3.8: 0.95)\n"
       "`1`  off"},
      {"XEN_TOP_K",
       "Requests without their own `top_k` sample only from this many most likely tokens.\n"
       "\n"
       "`empty`  the model's recommended value (Qwen3.8: 20)\n"
       "`0`  off"},
      {"XEN_MIN_P",
       "Requests without their own `min_p` drop tokens less likely than this fraction of the most likely one.\n"
       "\n"
       "`empty`  the model's (0: off)"},
      {"XEN_PRESENCE_PENALTY",
       "Requests without their own `presence_penalty` are pushed away from tokens they already wrote (positive: fewer repeats, more new words; negative: the opposite).\n"
       "\n"
       "`empty` / `0`  off\n"
       "\n"
       "Takes: - a little decode speed: penalised requests sample from the whole vocabulary on the CPU."},
      {"XEN_REPETITION_PENALTY",
       "Requests without their own `repetition_penalty` make every token already seen (prompt or answer) less likely by this factor.\n"
       "\n"
       "`empty`  the model's (1: off)\n"
       "`1.05`  mild\n"
       "\n"
       "Takes: - a little decode speed, as the presence penalty."},
      {"XEN_MAX_QUEUE",
       "How many requests may wait for a free slot. One more gets HTTP 429 (busy).\n"
       "\n"
       "`0`  no limit"},
      {"XEN_QUEUE_POLICY",
       "Which waiting request gets the next free slot.\n"
       "\n"
       "`fifo`  first come, first served\n"
       "`sjf`  the one asking for the fewest tokens first, so short jobs finish sooner"},
      {"XEN_QUEUE_AGING",
       "With `sjf`: after this many seconds of waiting, the oldest request goes first anyway, so long jobs are never starved."},
      {"XEN_QUEUE_TIMEOUT",
       "Seconds a request may wait for a slot before it is turned away with HTTP 503.\n"
       "\n"
       "`0`  wait forever"},
      {"XEN_IDLE_TIMEOUT",
       "Seconds before an idle keep-alive connection is closed.\n"
       "\n"
       "`0`  never"},
      {"XEN_SAMPLE_THREADS",
       "CPU threads that pick the next tokens (temperature, penalties, JSON output rules) for the streams of a step.\n"
       "\n"
       "`1`  one stream at a time"},
      {"XEN_PREFIX_CACHE",
       "Remembers finished conversations, so the next turn with the same start (an agent session, a chat history) only reads what is new.\n"
       "\n"
       "`on`  remember\n"
       "`off`  every prompt is read from the start\n"
       "\n"
       "Gives: + time to first token on every later turn of a conversation. Several requests at once on one document read it once and share it.\n"
       "Takes: - host RAM for the snapshots (`XEN_PREFIX_CACHE_GIB`). - bit-exactness: a reply continued from the cache can differ from one that read the same prompt from the start.\n"
       "KV context (27B): its staging buffers take 0.2 GiB of VRAM per card from the auto KV pool, 0.3 with DFlash 2, 0.5 with DSpine."},
      {"XEN_PREFIX_CACHE_GIB",
       "Host RAM for the prefix cache's snapshots, shared by all cards.\n"
       "\n"
       "Gives (more): + conversations remembered, so fewer prompts are read again from the start (it matters with many agents at once).\n"
       "Takes (more): - host RAM. One snapshot takes roughly 60-160 MiB, depending on the model and its drafter."},
      {"XEN_PREFIX_MARK_MIN",
       "Chat prompts at least this long are split where the reply starts, so the next turn can resume exactly there.\n"
       "\n"
       "`0`  every chat prompt\n"
       "\n"
       "Gives: + time to first token of the next turn.\n"
       "Takes: - one extra short step per marked prompt."},
      {"XEN_PREFIX_LIVE",
       "The next turn continues right after the previous reply instead of re-reading it.\n"
       "\n"
       "Gives: + time to first token of the next turn.\n"
       "Takes: - bit-exactness: late tokens may differ slightly from a cache-free run."},
      {"XEN_WARMUP",
       "Runs a few scratch requests after loading, before accepting traffic, so the first real request is not slower.\n"
       "\n"
       "Gives: + time to first token of the first real request.\n"
       "Takes: - the server is ready a moment later. Leaves no trace in the stats.\n"
       "KV context (qwen35 / llama / flint): without it the auto pool keeps 1.5 GiB more per card free for the kernels' first use, so it holds fewer tokens."},
      {"XEN_LOG_INTERVAL",
       "Seconds between the server's status lines in the log while it is busy: step time, prefill speed, KV pool, prefix cache, host RAM.\n"
       "\n"
       "`0`  off\n"
       "\n"
       "Note: the host-RAM warning and the crash report are written either way."},
      {"XEN_CHAT_TEMPLATE",
       "How chat messages are turned into the prompt.\n"
       "\n"
       "`auto`  the checkpoint's own template; the built-in renderer if it has none\n"
       "`builtin`  the engine's renderer\n"
       "`jinja`  the checkpoint's template, or refuse to start without one\n"
       "`a path`  that `.jinja` file\n"
       "\n"
       "Note: the shipped models give the same prompts either way."},
      {"XEN_VISION",
       "qwen38 and the 27B: lets requests include images. PNG / JPEG as `data:` URLs, or http(s) links the server downloads, in `image_url` or `input_image` parts.\n"
       "\n"
       "`on`  load the model's image encoder\n"
       "`off`  text only: image parts are refused\n"
       "\n"
       "Gives: + image (and video) input.\n"
       "Takes: - VRAM: 0.7 GiB per card at the default image size. qwen38 takes it from the hot experts, the 27B from the auto KV pool."},
      {"XEN_VISION_MAX_PIXELS",
       "Larger images are scaled down to at most this many pixels before encoding.\n"
       "\n"
       "`1048576`  ~1,024 prompt tokens per image; reads fine print\n"
       "\n"
       "Gives (larger): + detail the model can read in an image.\n"
       "Takes (larger): - prompt tokens and time to first token per image, - VRAM for the image encoder: 0.7 GiB per card at 1,048,576, 1.5 GiB at 4,194,304 (on the 27B from the auto KV pool)."},
      {"XEN_VISION_IMAGES",
       "How many images of the largest size stay encoded on the cards at once. A repeated image is not encoded again.\n"
       "\n"
       "Gives (more): + time to first token when images repeat across requests.\n"
       "Takes (more): - VRAM: each takes ~5 MiB (qwen38) or ~10 MiB (27B) per card at the default size."},
      {"XEN_MEDIA_URLS",
       "With image input on: requests may link images by `http://` or `https://` URL, and the server downloads them.\n"
       "\n"
       "`on`  links allowed\n"
       "`off`  only `data:` URLs (the image inside the request)\n"
       "\n"
       "Needs: https goes through the system's OpenSSL 3. Without it https links are off."},
      {"XEN_MEDIA_DOMAINS",
       "Host names the server may download images from, comma-separated: `upload.wikimedia.org,example.com`. Checked on every redirect too.\n"
       "\n"
       "`empty`  any host\n"
       "\n"
       "Careful: with any host, anyone who can reach the server can make it fetch any URL."},
      {"XEN_MEDIA_TIMEOUT",
       "Seconds one image download may take: connect, redirects and body. A slower link gets HTTP 400."},
      {"XEN_VISION_FFMPEG",
       "With image input on: requests may also include videos (`video_url` parts: mp4, webm, mov, ... as `data:` or http(s) URLs). This ffmpeg program decodes them.\n"
       "\n"
       "`ffmpeg`  the one on PATH, else the first of /usr/bin, /usr/local/bin, /opt/ffmpeg/bin, /snap/bin, ~/bin, ~/.local/bin, ~/.hermes/tools/*/bin\n"
       "`empty`  videos are refused; so is a program that is not found (the log says why)\n"
       "\n"
       "Needs: `ffprobe` beside it or on PATH."},
      {"XEN_VIDEO_MAX_PIXELS",
       "Pixel budget of one video over all its sampled frames. Longer videos get smaller frames.\n"
       "\n"
       "`25165824`  the model's default: at most ~12,300 prompt tokens per video\n"
       "\n"
       "Note: each frame also stays under the image size cap."},
      {"XEN_VIDEO_FPS",
       "Frames taken per second of video: at least 4 and at most 768 per video. Two frames become one picture of tokens.\n"
       "\n"
       "`2`  the model's default\n"
       "\n"
       "Tip: lower = fewer tokens and a faster first token."},
      {"XEN_REPORT_LAUNCH",
       "Shows how the server was started in GET /v1/models (its arguments and every XEN_ variable), so a benchmark's page can say \"Run this yourself\" with the exact command.\n"
       "\n"
       "Note: the values of key-, token- or password-like names are hidden, but local paths (the model folder) are visible to every client that can reach the server. Its build, the oneAPI version, the drafter and the cards are reported either way."},
      {"XEN_STUB",
       "A fake engine without GPU behind the real server, for testing clients and the API.\n"
       "\n"
       "Note: model settings do not apply."},
      {"XEN_PF_LOG",
       "Diagnostic: one log line per prompt-processing call, part and generation step."},
      {"XEN_PF_PROFILE",
       "qwen38 diagnostic: times every prompt-processing phase.\n"
       "\n"
       "`1`  host time\n"
       "`2`  device time with barriers\n"
       "`3`  GPU clock stamps\n"
       "\n"
       "Takes: - prompt reading speed. Never use it to measure speed."},
      {"XEN_QSA_PF_TILES",
       "qwen38 numerics switch for the prompt's attention.\n"
       "\n"
       "`empty`  the faster 8-tile prompt attention (the default)\n"
       "`1`  the generation step's tiling: bit-equal to it\n"
       "\n"
       "Gives (1): + bit-equality of prompt reading with generation.\n"
       "Takes (1): - a little prompt reading speed."},
      {"XEN_DEC_PROFILE",
       "qwen38 diagnostic: GPU clock stamps around every module of the generation step. The log gets one line of times per module every 512 steps.\n"
       "\n"
       "Takes: - decode speed. Never use it to measure speed."},
      {"XEN_DRAFT_GRAPHS",
       "Block drafter (DFlash 2 / DSpine): the most recorded drafter-pass graphs kept on the cards (a number, at least 4; default `96`, `72` with XEN_STEP_GRAPHS). Each holds about 10 MiB per "
       "card; once the cap is reached the least recently used one is dropped, and the log says so once.\n"
       "\n"
       "Lower: + VRAM headroom on tight cards. Takes: - a recording (~20-40 ms) whenever a dropped step shape comes back.\n"
       "Output: the same."},
      {"XEN_STEP_GRAPHS",
       "The dense 27B and llama (flint always does): records each decode step shape once as a GPU graph per card and replays it, instead of launching the step's ~2,000 kernels one by one. Needs XEN_GRAPHS on (the default).\n"
       "\n"
       "`1`  replay recorded step graphs\n"
       "`0`  plain launches (the default)\n"
       "\n"
       "Gives (1): + decode speed when the host CPU or its link is the bottleneck. On a fast host: the same.\n"
       "Takes (1): - ~42 ms the first time each step shape runs, - about 7.5 MiB per step graph per card (XEN_STEP_GRAPHS_MAX); the drafter's pass graphs (XEN_DRAFT_GRAPHS) drop to 72 so the total stays about the same.\n"
       "Output: the same greedy text, except with a trained bank and the adaptive depth on: the replays time differently, so the depth picks other step shapes at some steps, and the trained selection follows them (XEN_ADAPTIVE_DEPTH 0: the same).\n"
       "Note: not used with page-selected attention (XEN_ATTN_SELECT) or while XEN_QSA_SCORE_CHECK or XEN_QSA_PACK_CHECK is set."},
      {"XEN_STEP_GRAPHS_MAX",
       "The most decode step graphs kept (XEN_STEP_GRAPHS 1); the least recently replayed is dropped at the cap.\n"
       "\n"
       "`empty`  32 (the default)\n"
       "\n"
       "Lower: + VRAM headroom. Takes: - more re-recordings (~42 ms each) when step shapes vary.\n"
       "Limits: at least 4.\n"
       "Output: the same."},
      {"XEN_GRAPH_ROOM_MIB",
       "Decode step graphs (XEN_STEP_GRAPHS 1): below this much free VRAM on the tighter card, each new step graph first drops 4 old ones. A recording that fails drops all of them and runs that step without a graph.\n"
       "\n"
       "`empty`  128 MiB (the default)\n"
       "\n"
       "Higher: + VRAM headroom for everything else. Takes: - fewer step graphs kept on tight cards.\n"
       "Output: the same."},
      {"XEN_PREFILL_SHARE",
       "How the loop divides its time between reading new prompts and generating for the requests that are already answering. By default every loop turn runs one prompt call (or one part of a split call) next to the generation step, so a long prompt arriving slows down every stream that is generating while it is read. With a share `s` between 0 and 1, while any request generates, prompt reading gets about `s` of the loop's wall time and generation the rest; with nothing generating, prompts run at full speed again.\n"
       "\n"
       "`empty`  off: one prompt call or part every loop turn (the default)\n"
       "`0.33`  about a third of the time for prompts, two thirds for generation\n"
       "\n"
       "Gives: + steadier and faster generation for the users already answering while long prompts arrive (interactive serving with several users).\n"
       "Takes: - a long prompt can take longer to read while others generate (it gets at most its share).\n"
       "Note: every model. 0 and 1 also mean off.\n"
       "Output: the same."},
      {"XEN_TRAINED_QSA_DELTA_DIR",
       "Turns trained page selection on for the Qwen3.8-27B: the folder of the trained indexer, a small scorer per global-attention layer (downloaded separately; the int4-baked QSA preset names it). In speculative decoding past 8,192 tokens of context, each layer's scorer ranks the completed four-token blocks of the past for every new token, and attention reads only the 512 best (plus the token's own block) instead of the whole context. Layers missing from the bank, short contexts and prompts keep full attention.\n"
       "\n"
       "`empty`  off: full attention, nothing allocated (the default)\n"
       "\n"
       "Gives: + decode speed at long contexts (tens of thousands of tokens and more), more with longer contexts and more streams.\n"
       "Takes: - VRAM for the scorer weights and every slot's block summaries; - replies are no longer identical to full attention (each token reads part of the past).\n"
       "Needs: the int8-v4 KV cache (its pool a token count or auto), a block drafter (XEN_DRAFTER_MODEL) and a context of 2,049-262,144; the server refuses to start otherwise. The long-context page selection (XEN_ATTN_SELECT) has no int8-v4 kernel and is not used with it. Image input is not supported with it.\n"
       "Output: differs from full attention."},
      {"XEN_TRAINED_QSA_TILED_SCORE",
       "How the scorer computes the block scores of chain verifier steps. Tiled: one kernel scores a tile of up to 16 rows per stream and 32 rows per step together, with the same arithmetic order as the untiled scorer; steps outside the tile use the untiled one.\n"
       "\n"
       "`1`  tiled scorer\n"
       "`0`  untiled scorer (the default)\n"
       "\n"
       "Gives (1): + decode speed (a cheaper scoring pass every step).\n"
       "Takes (1): nothing.\n"
       "Note: draft trees with selection (XEN_TRAINED_QSA_TREE) need it.\n"
       "Output: the same."},
      {"XEN_TRAINED_QSA_POOL_FP16",
       "The format of the block summaries the scorer ranks: one 128-value key per completed four-token block, kept for every slot and trained layer for the whole context. FP16 halves their VRAM; the scores are still computed in FP32 (the summaries are widened first), and the current query and unfinished blocks stay FP32.\n"
       "\n"
       "`1`  FP16 summaries\n"
       "`0`  FP32 summaries (the default)\n"
       "\n"
       "Gives (1): + half the summaries' VRAM (room for a larger KV pool or more slots); the XMX scorer (XEN_QSA_XMX_SCORE) needs it.\n"
       "Takes (1): - rounding: blocks with nearly equal scores can be picked differently than with FP32.\n"
       "Output: can differ from FP32 summaries."},
      {"XEN_TRAINED_QSA_TREE",
       "Lets draft-tree verifier steps (XEN_DRAFTER_TREE) read selectively as well; without it every tree step reads the full context. With it, each eligible tree segment (a stream with complete history, past 8,192 tokens, at most 16 tree rows) takes the selection, and the other rows of the step keep full attention.\n"
       "\n"
       "`1`  tree steps select\n"
       "`0`  tree steps read the full context (the default)\n"
       "\n"
       "Gives (1): + decode speed at long contexts whenever the drafter grows trees (the first streams, XEN_DRAFTER_TREE_SEQS).\n"
       "Takes (1): nothing beyond the selection's own cost.\n"
       "Needs: tiled scoring (XEN_TRAINED_QSA_TILED_SCORE 1); the server refuses to start without it.\n"
       "Output: differs from full attention, like every selected step."},
      {"XEN_QSA_XMX_SCORE",
       "Computes the block scores on the cards' matrix engines (XMX, FP16 products) instead of vector FMA. The query is split into two FP16 halves (high and low part), so the products stay close to FP32 and the 512 picks normally match the vector scorer.\n"
       "\n"
       "`1`  XMX scorer\n"
       "`0`  vector scorer (the default)\n"
       "\n"
       "Gives (1): + decode speed: scoring is the selector's largest cost and becomes several times cheaper.\n"
       "Takes (1): - repeatability: tree and chain steps score with differently rounded methods, so nearly equal blocks can be picked differently and the same greedy request can return a different reply on another run.\n"
       "Needs: FP16 summaries (XEN_TRAINED_QSA_POOL_FP16 1); without them the vector scorer runs.\n"
       "Output: can differ from run to run."},
      {"XEN_QSA_GATHER",
       "How attention reads what the scorer picked. Without it the picks are read as whole 64-token KV pages, with the unpicked blocks masked: a page with one picked block is still read in full. With it the walk gathers just the picked four-token blocks (16 per entry, from any pages).\n"
       "\n"
       "`1`  gather the picked blocks\n"
       "`0`  read whole pages (the default)\n"
       "\n"
       "Gives (1): + decode speed: less KV traffic per step, the more so the more scattered the picks.\n"
       "Takes (1): nothing.\n"
       "Output: the same blocks are read; the attention sums run in another order."},
      {"XEN_QSA_RESIDENT_SNAPSHOTS",
       "The scorer needs each slot's block summaries from token zero. A request resumed from the prefix cache (a follow-up turn) gets its KV back but not the summaries, and reads the full context for the rest of the request. With this, prefix-cache snapshots remember which slot's summaries match them, and a turn that resumes in that slot keeps selecting.\n"
       "\n"
       "`1`  keep the history\n"
       "`0`  resumed turns read the full context (the default)\n"
       "\n"
       "Gives (1): + decode speed on multi-turn chats and agents with long histories.\n"
       "Takes (1): nothing beyond the bookkeeping.\n"
       "Needs: the prefix cache (XEN_PREFIX_CACHE 1). Resuming in another slot also needs XEN_QSA_RESIDENT_FORK_COPY.\n"
       "Output: differs from full attention, like every selected step."},
      {"XEN_QSA_RESIDENT_FORK_COPY",
       "When a prefix-cache snapshot resumes in a different slot than the one that wrote it, also copies its block summaries there, so that request keeps selecting too. Only for snapshots that end on a 64-token page boundary and carry no rotary offset; the others resume with full attention.\n"
       "\n"
       "`1`  copy the summaries\n"
       "`0`  only the writing slot keeps selecting (the default)\n"
       "\n"
       "Gives (1): + decode speed when a follow-up turn lands in another slot.\n"
       "Takes (1): - a copy of the summaries on the card at each such resume.\n"
       "Needs: XEN_QSA_RESIDENT_SNAPSHOTS 1.\n"
       "Output: differs from full attention, like every selected step."},
      {"XEN_TRAINED_QSA_DECODE_MIN_POS",
       "The context position below which every step reads the full context, on top of the fixed 8,192-token threshold (the higher of the two applies).\n"
       "\n"
       "`0`  the 8,192-token threshold alone (the default)\n"
       "\n"
       "Higher: + full-attention output further into the context; - the selection's speed starts later.\n"
       "A value past the context: every step reads the full context with the bank loaded (a comparison run).\n"
       "Output: the same as full attention below the position."},
      {"XEN_TRAINED_QSA_PREFILL_XMX",
       "Reads long prompts selectively too. Past XEN_TRAINED_QSA_PREFILL_MIN_POS, every prompt chunk's queries are scored on XMX (BF16) against the completed blocks, and each query reads only its 512 best blocks instead of the whole prefix.\n"
       "\n"
       "`1`  selected prompt reading\n"
       "`0`  full prompt attention, or the long-prompt page selection (the default)\n"
       "\n"
       "Gives (1): + prompt reading speed on very long prompts.\n"
       "Takes (1): - the prompt is read differently from full attention (BF16 scores, part of the prefix); - VRAM for its workspace.\n"
       "Needs: the full bank (XEN_TRAINED_QSA_DELTA_DIR), one slot (four with XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG), the int8-fast prompt attention and no prompt slicing. It stays off while the long-prompt page selection (XEN_PREFILL_SELECT) is on, which reads long prompts faster.\n"
       "Output: differs from full attention."},
      {"XEN_TRAINED_QSA_PREFILL_MIN_POS",
       "The prompt position from which the selected prompt reading starts; earlier chunks read the full prefix. A multiple of 64, at least 2,112 and below the context.\n"
       "\n"
       "`empty`  32,768 (the default)\n"
       "\n"
       "Lower: + more of the prompt read selectively (faster); - more of it read differently from full attention.\n"
       "Needs: XEN_TRAINED_QSA_PREFILL_XMX 1; the server refuses it otherwise."},
      {"XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG",
       "Allows the selected prompt reading on a server with four slots (XEN_SLOTS 4) instead of one. Prompt calls that carry one prompt read it selectively; calls that merge several prompts read the full prefix unless XEN_TRAINED_QSA_PREFILL_MERGED is on.\n"
       "\n"
       "`1`  allowed with four slots\n"
       "`0`  one slot only (the default)\n"
       "\n"
       "Gives (1): + the selected prompt reading for up to four users.\n"
       "Needs: XEN_SLOTS 4 and XEN_TRAINED_QSA_PREFILL_XMX 1; the server refuses it otherwise.\n"
       "Output: differs from full attention."},
      {"XEN_TRAINED_QSA_PREFILL_MERGED",
       "Gives each prompt of a merged prompt call (several prompts read in one call) its own selection, instead of reading the full prefix for all of them.\n"
       "\n"
       "`1`  merged calls select per prompt\n"
       "`0`  merged calls read the full prefix (the default)\n"
       "\n"
       "Gives (1): + prompt reading speed when several long prompts arrive together.\n"
       "Needs: XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG 1; the server refuses it otherwise.\n"
       "Output: differs from full attention."},
      {"XEN_PF_SUBTILE_SKIP",
       "In the selected prompt reading, skips the parts of each 40-row attention group that picked nothing from a page, instead of computing them and masking the result.\n"
       "\n"
       "`1`  skip them\n"
       "`0`  compute and mask (the default)\n"
       "\n"
       "Gives (1): + prompt reading speed.\n"
       "Takes (1): - a small table in VRAM.\n"
       "Needs: XEN_TRAINED_QSA_PREFILL_XMX 1 with the native 40-row groups.\n"
       "Output: the same."},
      {"XEN_TRAINED_QSA_SHARE_SCRATCH",
       "The selector's work buffers (projections, scores, candidates) each have their own VRAM by default. This lays them over one another, sized for the largest stage at a time.\n"
       "\n"
       "`1`  one shared buffer\n"
       "`0`  separate buffers (the default)\n"
       "\n"
       "Gives (1): + VRAM headroom (room for a larger KV pool or another slot).\n"
       "Takes (1): nothing.\n"
       "Output: the same."},
      {"XEN_TRAINED_QSA_STREAM_PROJECTION",
       "Keeps the scorer's projection weights (three matrices per trained layer) in pinned host memory and copies a layer's matrices to the card when that layer runs, instead of holding all of them on every card.\n"
       "\n"
       "`1`  copy from the host as needed\n"
       "`0`  all weights in VRAM (the default)\n"
       "\n"
       "Gives (1): + VRAM headroom: the weights of every trained layer but one.\n"
       "Takes (1): - decode and prompt speed (a host-to-card copy for each trained layer of every step); - pinned host RAM.\n"
       "Output: the same."},
      {"XEN_REQ_LOG",
       "Diagnostic: one `REQ_DONE` log line per finished request with its slot, prompt and cached tokens, tokens generated, time to first token, decode time and speed, verifier steps and tokens per step, the average number of streams beside it, and why it finished.\n"
       "\n"
       "Takes: nothing measurable (one line per request)."},
      {"XEN_DEPTH_TRACE",
       "Diagnostic for the adaptive draft depth (XEN_ADAPTIVE_DEPTH): a log line each time it changes the depth or the drafter layers used at a stream count, with the tokens per millisecond before and after.\n"
       "\n"
       "Takes: nothing measurable."},
      {"XEN_QSA_TRIAL_ROUTE_LOG",
       "Trained page selection diagnostic: one log line per stream per verifier step with the route its attention took and why: `covered` or `covered-tree` (selected), `short` (under the threshold), `incomplete` (history missing, e.g. resumed from the cache), `tree` (tree steps without XEN_TRAINED_QSA_TREE), `mixed`.\n"
       "\n"
       "Takes: - decode speed (very verbose). Never use it to measure speed."},
      {"XEN_QSA_PACK_CHECK",
       "Trained page selection diagnostic: for the first N steps on card 0, reads back the packed attention entries and checks them against a host reference; the log reports mismatches.\n"
       "\n"
       "`0`  off (the default)\n"
       "\n"
       "Takes: - decode speed during the checked steps (waits on the card)."},
      {"XEN_QSA_SCORE_CHECK",
       "Trained page selection diagnostic: for the first N XMX scoring passes on card 0, also computes the scores with the vector scorer and logs how far the two differ.\n"
       "\n"
       "`0`  off (the default)\n"
       "\n"
       "Needs: XEN_QSA_XMX_SCORE 1 and XEN_TRAINED_QSA_POOL_FP16 1.\n"
       "Takes: - decode speed during the checked steps."},
      {"XEN_PF_DENSE_ONEDNN_LOG",
       "qwen38 diagnostic for the oneDNN prompt GEMMs of the dense trunk (XEN_PREFILL_GEMM onednn): one line per new matrix shape saying whether oneDNN or the engine's own kernel runs it (oneDNN is kept only where it gives the engine's bits)."},
      {"XEN_PF_DENSE_ONEDNN_SLOTS",
       "qwen38, oneDNN prompt GEMMs of the dense trunk: how many repacked weight matrices stay cached on each card.\n"
       "\n"
       "`empty`  4 (the default)\n"
       "\n"
       "More: + fewer repacks during long prompts; - VRAM (one largest int8 matrix per slot)."},
      {"XEN_PF_ONEDNN_SLOTS",
       "The int4 prompt GEMMs on oneDNN (XEN_PREFILL_GEMM onednn): how many repacked int4 weight matrices stay cached on each card.\n"
       "\n"
       "`empty`  2 (the default)\n"
       "\n"
       "More: + fewer repacks per prompt call; - VRAM (one repacked matrix per slot)."},
  };
  for (const auto &[k, v] : t)
    if (key == k) return v;
  return "";
}

std::vector<Spec> schema() {
  std::vector<Spec> v;
  auto add = [&](std::string key, std::string label, std::string group, Kind kind) -> Spec & {
    Spec s;
    s.key = std::move(key);
    s.label = std::move(label);
    s.group = std::move(group);
    s.kind = kind;
    v.push_back(std::move(s));
    return v.back();
  };
  const std::string net = "Model & network", cap = "Capacity & memory", spec = "Speculative decoding", num = "Numerics & kernels",
                    pf = "Prompt processing & cache", q = "Queue & connections", rt = "Runtime", dg = "Diagnostics",
                    smp = "Sampling defaults", tq = "Trained page selection";
  add("XEN_MODEL", "Model", net, Kind::Path);
  {
    auto &s = add("XEN_MODELS_DIR", "Models folder", net, Kind::Path);
    s.allow_empty = true;
  }
  add("XEN_SERVED_NAME", "Served model name", net, Kind::Text);
  {
    auto &s = add("XEN_CHAT_TEMPLATE", "Chat template", net, Kind::Text);
    s.choices = {"auto", "builtin", "jinja"};
    s.names = {{"auto", "auto (checkpoint template)"}, {"builtin", "built-in renderer"}, {"jinja", "checkpoint template only"}};
  }
  {
    auto &s = add("XEN_HOST", "Listen address", net, Kind::Text);
    s.choices = {"0.0.0.0", "127.0.0.1"};
    s.names = {{"0.0.0.0", "0.0.0.0 (LAN)"}, {"127.0.0.1", "127.0.0.1 (this host)"}};
  }
  {
    auto &s = add("XEN_PORT", "Port", net, Kind::Int);
    s.lo = 1;
    s.hi = 65535;
  }
  {
    auto &s = add("XEN_SLOTS", "Slots", cap, Kind::Int);
    s.lo = 1;
    s.hi = 64;  // Qwen38 on four cards 64; the other archs are capped below
    s.choices = {"1", "2", "4", "8", "16", "32", "64"};
    s.unit = "seq";
  }
  {
    auto &s = add("XEN_CONTEXT", "Max context", cap, Kind::Int);
    s.lo = 256;
    s.choices = {"8192", "32768", "65536", "131072", "262144", "524288", "1048576"};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_TP", "Cards (tensor parallel)", cap, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"2", "1"};
    s.names = {{"2", "TP2 (both cards)"}, {"1", "TP1 (one card)"}};
  }
  {
    auto &s = add("XEN_CARD", "TP1 card", cap, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 7;
    s.choices = {"0", "1"};
  }
  {  // every arch: run as an Arc Pro B60 / B65 (util/cards.hpp, rt/card.hpp)
    auto &s = add("XEN_EMU_CARD", "Emulate card", cap, Kind::Choice);
    s.choices = {"off", "b60", "b65"};
    s.names = {{"off", "off (the real cards)"}, {"b60", "Arc Pro B60 (20 cores, 24 GB)"}, {"b65", "Arc Pro B65 (20 cores, 32 GB)"}};
  }
  {  // qwen38's own pair: switching XEN_MODEL between the archs keeps each one's TP
    auto &s = add("XEN_Q38_TP", "Cards (tensor parallel)", cap, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"2", "4", "1"};  // TP1 TP4
    s.names = {{"2", "TP2 (cards 0 and 1)"}, {"4", "TP4 (cards 0-3)"}, {"1", "TP1 (one card)"}};
  }
  {
    auto &s = add("XEN_Q38_CARD", "TP1 card", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.hi = 7;
    s.choices = {"0", "1"};
  }
  {
    auto &s = add("XEN_KV_POOL_TOKENS", "KV pool", cap, Kind::Int);
    s.lo = 64;  // the fixed per-slot layout (0) is no longer a server option
    s.allow_auto = true;
    s.choices = {"auto"};
    s.names = {{"auto", "auto (sized for the model)"}};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_KV", "KV cache dtype", cap, Kind::Choice);
    s.choices = {"int8", "bf16", "int8-v4"};  // int8-v4: qwen38 and the 27B
    s.names = {{"int8", "int8"}, {"bf16", "bf16"}, {"int8-v4", "int8 K + int4 V (qwen38, 27B)"}};
  }
  {
    auto &s = add("XEN_KV_SPLIT", "KV split on four cards", cap, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"heads", "tokens"};
    s.names = {{"heads", "by head (both cards of a head hold all its tokens)"}, {"tokens", "by token (a quarter of the KV per card)"}};
  }
  {
    auto &s = add("XEN_HOT", "Hot experts in VRAM", cap, Kind::Int);
    s.scope = Scope::MoE;
    s.lo = 0;
    s.allow_empty = true;
    s.names = {{"", "auto (as many as fit)"}, {"0", "auto (as many as fit)"}};
  }
  {
    auto &s = add("XEN_RESERVE", "VRAM reserve", cap, Kind::Float);
    s.lo = 0;
    s.hi = 30;
    s.choices = {"0.25", "0.5", "1", "2"};
    s.unit = "GiB/card";
  }
  {
    auto &s = add("XEN_HOT_RANK", "Hot-expert ranking", cap, Kind::Path);
    s.scope = Scope::MoE;
    s.path_file = true;
    s.path_auto = true;
    s.allow_empty = true;
    s.names = {{"auto", "auto (recorded for this model)"}, {"", "none (default placement, not recorded)"}};
  }
  {
    auto &s = add("XEN_MOE_CPU", "CPU computes cold experts", cap, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"off", "auto", "avx512", "avx2"};
    s.names = {{"off", "off (cold experts on the cards)"}, {"auto", "auto (AVX-512 BF16, else AVX2)"}, {"avx512", "AVX-512 BF16"}, {"avx2", "AVX2 + FMA"}};
  }
  {
    auto &s = add("XEN_MOE_CPU_THREADS", "CPU threads per card", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 1;
    s.hi = 64;
    s.choices = {"8", "6", "4"};
    s.unit = "threads";
  }
  {
    auto &s = add("XEN_MOE_CPU_CORES", "CPU first cores", cap, Kind::Text);
    s.scope = Scope::Qwen38;
    s.allow_empty = true;
    s.names = {{"", "auto (0 and threads)"}};
  }
  {
    auto &s = add("XEN_MOE_CPU_PCT", "CPU share of cold experts", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.hi = 100;
    s.choices = {"70", "50", "100"};
    s.unit = "%";
  }
  {
    auto &s = add("XEN_MOE_CPU_MIN_ROWS", "CPU path from rows", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 1;
    s.hi = 8;
    s.choices = {"1", "2", "4"};
    s.unit = "rows";
  }
  {
    auto &s = add("XEN_MOE_CPU_MIN_COLD", "CPU path from cold experts", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 1;
    s.hi = 80;
    s.choices = {"2", "1", "3"};
    s.unit = "per layer";
  }
  {
    auto &s = add("XEN_MOE_CPU_TIMEOUT_MS", "CPU path wait bound", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 1;
    s.hi = 60000;
    s.choices = {"500", "100", "2000"};
    s.unit = "ms";
  }
  {
    auto &s = add("XEN_MTP", "MTP drafter", spec, Kind::Path);
    s.scope = Scope::Qwen38;
    s.allow_empty = true;
    s.names = {{"", "off"}};
  }
  {
    auto &s = add("XEN_DRAFTER", "Checkpoint drafter", spec, Kind::Bool);
    s.scope = Scope::Qwen35;
  }
  {
    auto &s = add("XEN_DRAFTER_MODEL", "DFlash drafter", spec, Kind::Path);
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
    s.names = {{"", "MTP layer"}};
  }
  {
    auto &s = add("XEN_DRAFTER_MTP_FROM", "MTP for sequences from", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 64;  // Qwen38 up to 64 slots
    s.choices = {"0", "5", "9", "13"};  // 9 = the default 13 (qwen38, 32 slots, 64-row steps)
    s.names = {{"0", "DFlash alone"}};
    s.unit = "sequences";
  }
  {
    auto &s = add("XEN_DRAFTER_DEFER", "Drafter deferred rows", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 512;
    s.choices = {"0", "32", "64", "128"};
    s.names = {{"0", "off"}};
    s.unit = "rows/slot";
  }
  {
    auto &s = add("XEN_DRAFTER_CUT", "DFlash draft cut", spec, Kind::Float);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 0.99;
    s.choices = {"0", "0.01", "0.02", "0.05"};
    s.names = {{"0", "off"}};
  }
  {
    auto &s = add("XEN_DRAFTER_TREE", "DFlash draft tree", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 7;
    s.choices = {"0", "3", "5", "7"};
    s.names = {{"0", "off"}};
    s.unit = "nodes";
  }
  {
    auto &s = add("XEN_DRAFTER_TREE_SEQS", "Draft trees up to", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 1;
    s.hi = 8;
    s.choices = {"1", "2", "4"};
    s.unit = "streams";
  }
  {
    auto &s = add("XEN_STEP_ROWS", "Step rows (all sequences)", spec, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"0", "32", "16", "8", "64"};  // 32 0 = the model's 64
    s.names = {{"0", "the model's (qwen35 32; qwen38 16, 64 on four cards)"}, {"32", "32 (8 for llama / flint)"}, {"16", "16 (8 for llama / flint)"}, {"8", "8 (slots <= 8)"},
               {"64", "64 (qwen38, four cards)"}};
  }
  {
    auto &s = add("XEN_SEG_ROWS", "Rows per sequence", spec, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"16", "8"};  // 16 = the default
    s.names = {{"16", "16 (16-block drafter; the default)"}, {"8", "8 (8-block drafter)"}};
  }
  {
    auto &s = add("XEN_DRAFT_DEPTHS", "Drafts per sequence", spec, Kind::Text);
    s.scope = Scope::Qwen35;
    s.choices = {"", "15,7,4,3", "15,7", "7,3"};
    s.names = {{"", "built-in table (7,7,4,3,1,1,1,1)"}};
  }
  {
    auto &s = add("XEN_DRAFTER_WIDE", "16-row drafts after", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 15;
    s.choices = {"0", "5", "7"};
    s.names = {{"0", "off"}};
    s.unit = "accepted";
  }
  {
    auto &s = add("XEN_DRAFTER_INT4", "Drafter matrices int4", spec, Kind::Bool);
    s.scope = Scope::Qwen35;
  }
  {
    auto &s = add("XEN_DRAFTER_TREE_WIDE", "16-row tree nodes", spec, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 1;
    s.hi = 15;
    s.choices = {"11", "15"};
    s.unit = "nodes";
  }
  {
    auto &s = add("XEN_DRAFT_VOCAB", "Draft vocabulary", spec, Kind::Int);  // both archs since
    s.lo = 0;
    s.multiple = 32;
    s.choices = {"0", "32768", "65536", "98304"};
    s.names = {{"0", "full head"}};
    s.unit = "ids";
  }
  {
    auto &s = add("XEN_DRAFT_HEAD", "Draft head rows", spec, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "int8", "int4"};
    s.names = {{"auto", "auto (as the lm_head)"}, {"int8", "int8 rows"}, {"int4", "int4 rows (lm_head.qweight)"}};
  }
  {
    add("XEN_ADAPTIVE_DEPTH", "Adaptive draft depth", spec, Kind::Bool);  // both archs
  }
  {
    auto &s = add("XEN_UNEQUAL_SEGMENTS", "Draft depth per stream", spec, Kind::Bool);  // Qwen38 + DFlash; 229: qwen35
    s.scope = Scope::Qwen38;
  }
  {
    auto &s = add("XEN_LOOKUP", "Lookup drafting", spec, Kind::Choice);  // the engine's, every arch with a drafter
    s.choices = {"4", "0", "3", "6"};
    s.names = {{"0", "off"}, {"4", "4-token match (default)"}, {"3", "3-token match"}, {"6", "6-token match"}};
  }
  {
    auto &s = add("XEN_MTP_DEPTH", "MTP depth", spec, Kind::Choice);  // one decoding sequence; qwen38 1-3, the dense 27B 1-6
    s.choices = {"3", "2", "1", "4", "5", "6"};
    s.names = {{"3", "MTP-3 (default)"}, {"2", "MTP-2"}, {"1", "MTP-1"}, {"4", "MTP-4 (Qwen3.8-27B)"}, {"5", "MTP-5 (Qwen3.8-27B)"}, {"6", "MTP-6 (Qwen3.8-27B)"}};
  }
  {
    auto &s = add("XEN_PREFILL_MOE", "Prefill experts", num, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"w4a16", "w4a8"};
  }
  {
    auto &s = add("XEN_PREFILL_DENSE", "Prefill dense trunk", num, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"w8a16", "w8a8", "w8a8-row"};
  }
  {
    auto &s = add("XEN_DECODE_MOE", "Decode experts", num, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"w4a16", "w4a8", "auto"};
  }
  {
    auto &s = add("XEN_GDN_W4", "GDN projections", num, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"int8", "int4", "int4-qkvz"};
    s.names = {{"int8", "int8 rows"}, {"int4", "int4 GPTQ (gdn4-calib dir)"}, {"int4-qkvz", "int4 GPTQ in_proj, int8 out_proj"}};
  }
  {
    auto &s = add("XEN_WIRE", "Two-card wire", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"bf16", "fp32", "int8"};
  }
  {
    auto &s = add("XEN_PREFILL_WIRE", "Two-card wire (prompts)", num, Kind::Choice);
    s.scope = Scope::Qwen35;  // + qwen38)
    s.choices = {"auto", "bf16", "fp32"};
  }
  {
    auto &s = add("XEN_YARN_TIERS", "YaRN per request", num, Kind::Choice);  // every arch but flint (flint_skips)
    s.choices = {"on", "off"};
  }
  {
    auto &s = add("XEN_W4_ITEMS", "W4 k-split target", num, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 1;
    s.choices = {"512", "1024"};
    s.unit = "items";
  }
  {
    auto &s = add("XEN_ATTN_SELECT", "Long-context page selection", num, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 4096;
    s.choices = {"0", "512", "1024", "2048"};
    s.names = {{"0", "off (dense)"}};
    s.unit = "pages";
  }
  {
    auto &s = add("XEN_SELECT_HEADS", "Page selection: sharing among heads", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "same"};
    s.names = {{"auto", "auto (each request measures its heads)"}, {"same", "same (the count for every head)"}};
  }
  {
    auto &s = add("XEN_PREFILL_SELECT", "Long-prompt page selection", num, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.hi = 4096;
    s.choices = {"0", "512", "1024", "2048"};
    s.names = {{"0", "off (dense)"}};
    s.unit = "pages";
  }
  {
    auto &s = add("XEN_DECODE_W4", "Decode int4 GEMMs", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"exact", "a8"};
    s.names = {{"exact", "exact (bf16 activations)"}, {"a8", "a8 (int8 activations, numerics change)"}};
  }
  {
    auto &s = add("XEN_DECODE_ATTN", "Decode attention", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "exact", "fast", "int8", "int8-fast"};
    s.names = {{"auto", "auto (int8-fast on int8 KV, else exact)"},
               {"exact", "exact (hi + lo P)"},
               {"fast", "fast (bf16 P, numerics change)"},
               {"int8", "int8 q.k (int8 KV, numerics change)"},
               {"int8-fast", "int8 q.k + bf16 P (int8 KV, numerics change)"}};
  }
  {
    auto &s = add("XEN_PREFILL_ATTN", "Prefill attention", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "exact", "fast", "int8", "int8-fast", "onednn"};  // int8 q.k; auto; onednn
    s.names = {{"auto", "auto (int8-fast on int8 KV, fast on bf16 KV)"},
               {"exact", "exact (hi + lo P)"},
               {"fast", "fast (bf16 P, numerics change)"},
               {"int8", "int8 q.k (int8 KV, numerics change)"},
               {"int8-fast", "int8 q.k + bf16 P (int8 KV, numerics change)"},
               {"onednn", "oneDNN f16 attention (qwen35, numerics change)"}};
  }
  {
    auto &s = add("XEN_PREFILL_W4", "Prefill int4 GEMMs", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"exact", "dequant", "a8"};
    s.names = {{"exact", "exact (fp32 group sums)"}, {"dequant", "dequant (bf16 w x s, numerics change)"}, {"a8", "a8 (int8 activations, numerics change)"}};
  }
  {
    auto &s = add("XEN_ATTN_TAIL", "Prefill attention: context splits", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"2", "1", "0"};
    s.names = {{"2", "long calls' last wave + short calls"}, {"1", "long calls' last wave"}, {"0", "off"}};
  }
  {
    auto &s = add("XEN_PREFILL_SHORT", "Short prompt calls on generation kernels", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"32", "16", "8", "4", "1", "0"};  // 32
    s.names = {{"32", "calls of 1-32 rows"}, {"16", "calls of 1-16 rows"}, {"8", "calls of 1-8 rows"}, {"4", "calls of 1-4 rows"}, {"1", "1-row calls"}, {"0", "off"}};
  }
  {
    auto &s = add("XEN_PREFILL_GEMM", "Prefill GEMM library", num, Kind::Choice);
    s.scope = Scope::All;  // qwen35 and qwen38 (applies)
    s.choices = {"onednn", "onednn-all", "esimd"};
    s.names = {{"onednn", "oneDNN where it wins"}, {"onednn-all", "oneDNN for every int4 matrix (qwen35)"}, {"esimd", "engine kernels only"}};
  }
  {
    auto &s = add("XEN_PREFILL_GEMM_ROWS", "Prefill GEMM library: from rows per call", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"1024", "512", "256", "128"};
    s.names = {{"1024", "1,024 (prompts of 2,048+ tokens)"}, {"512", "512 (numerics change; with KV lending)"}, {"256", "256 (numerics change; with KV lending)"},
               {"128", "128 (numerics change; with KV lending)"}};
  }
  {
    auto &s = add("XEN_PREFILL_A8_BODY", "W4A8: GEMM kernel", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "slm", "direct"};
    s.names = {{"auto", "auto (slm on two cards, direct on one)"}, {"slm", "slm (shared-memory staging)"}, {"direct", "direct (no shared memory)"}};
  }
  {
    auto &s = add("XEN_PREFILL_A8_KEEP", "W4A8: GEMMs kept W4A16", num, Kind::Text);
    s.scope = Scope::Qwen35;
    s.choices = {"", "gdn+attn", "gdn", "down"};
    s.names = {{"", "none (every GEMM W4A8)"}, {"gdn+attn", "gdn+attn (only the MLP on W4A8)"}, {"gdn", "gdn"}, {"down", "down"}};
  }
  {
    auto &s = add("XEN_LM_HEAD", "lm_head format", num, Kind::Choice);
    s.scope = Scope::Qwen35;
    s.choices = {"auto", "int8", "int4", "int4-rescore"};  // auto
    s.names = {{"auto", "auto (int4-rescore if the dir has both heads, int4 if int4-only, else int8)"},
               {"int8", "int8 rows (converted at load)"}, {"int4", "int4 g128 GPTQ (lm_head.qweight, numerics change)"},
               {"int4-rescore", "int4 scores + int8 rescore of the top tiles (two-stage)"}};
  }
  {
    auto &s = add("XEN_LM_HEAD_TILES", "Two-stage head: tiles per card", num, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 1;
    s.hi = 32;
    s.choices = {"8", "16", "32"};
    s.unit = "tiles";
  }
  {
    auto &s = add("XEN_GRAPHS", "Step graphs", num, Kind::Bool);
    s.scope = Scope::Qwen38;
  }
  {
    auto &s = add("XEN_STEP_GRAPHS", "Decode step graphs (qwen35 runtime)", num, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_PREFILL_GRAPHS", "Short prompts replay recorded graphs", num, Kind::Bool);
    s.scope = Scope::Qwen38;
  }
  {
    auto &s = add("XEN_PREFILL_CHUNK", "Prefill chunk", pf, Kind::Int);
    s.lo = 0;
    s.multiple = 4;
    s.choices = {"0", "512", "1024", "2048", "4096"};
    s.names = {{"0", "arch default"}};
    s.unit = "rows";
  }
  {
    auto &s = add("XEN_PREFILL_STALL_MS", "Prefill stall budget", pf, Kind::Float);
    s.lo = 0;
    s.choices = {"0", "100", "150"};
    s.names = {{"0", "whole chunks"}};
    s.unit = "ms";
  }
  {
    auto &s = add("XEN_PREFILL_PART_MS", "Prefill part length", pf, Kind::Float);
    s.lo = -1;
    s.choices = {"-1", "0", "6", "12", "25", "90"};
    s.names = {{"-1", "arch default"}, {"0", "whole calls"}};
    s.unit = "ms";
  }
  {
    auto &s = add("XEN_PREFILL_MERGE", "Prompts per prefill call", pf, Kind::Int);
    s.scope = Scope::Qwen35;
    s.lo = 1;
    s.hi = 16;
    s.choices = {"1", "4", "8", "16"};
    s.names = {{"1", "off (a call per prompt)"}};
    s.unit = "prompts";
  }
  {  // Image input
    auto &s = add("XEN_VISION", "Image input", net, Kind::Bool);
    s.scope = Scope::Vision;
  }
  {
    auto &s = add("XEN_VISION_MAX_PIXELS", "Image size cap", net, Kind::Int);
    s.scope = Scope::Vision;
    s.lo = 65536;
    s.hi = 16777216;
    s.choices = {"262144", "1048576", "4194304"};
    s.unit = "pixels";
  }
  {
    auto &s = add("XEN_VISION_IMAGES", "Images kept encoded", net, Kind::Int);
    s.scope = Scope::Vision;
    s.lo = 1;
    s.hi = 256;
    s.choices = {"4", "16", "64"};
  }
  {  // Media URLs
    auto &s = add("XEN_MEDIA_URLS", "Images by http(s) link", net, Kind::Bool);
    s.scope = Scope::Vision;
  }
  {
    auto &s = add("XEN_MEDIA_DOMAINS", "Image link hosts", net, Kind::Text);
    s.scope = Scope::Vision;
    s.allow_empty = true;
    s.names = {{"", "any host"}};
  }
  {  // Video input
    auto &s = add("XEN_VISION_FFMPEG", "Video decoder (ffmpeg)", net, Kind::Path);
    s.scope = Scope::Vision;
    s.allow_empty = true;
    s.names = {{"", "off (no video)"}};
  }
  {
    auto &s = add("XEN_VIDEO_MAX_PIXELS", "Video pixel budget", net, Kind::Int);
    s.scope = Scope::Vision;
    s.lo = 65536;
    s.hi = 268435456;
    s.choices = {"6291456", "25165824", "50331648"};
    s.unit = "pixels";
  }
  {
    auto &s = add("XEN_VIDEO_FPS", "Video frames per second", net, Kind::Float);
    s.scope = Scope::Vision;
    s.lo = 0.01;
    s.hi = 30;
    s.choices = {"1", "2", "4"};
    s.unit = "fps";
  }
  {
    auto &s = add("XEN_MEDIA_TIMEOUT", "Image download timeout", net, Kind::Float);
    s.scope = Scope::Vision;
    s.lo = 0.1;
    s.hi = 600;
    s.choices = {"5", "30", "120"};
    s.unit = "s";
  }
  {
    auto &s = add("XEN_KV_LEND", "KV lending (free KV pages put to use)", cap, Kind::Choice);  // Qwen38-27b the dense 27B too (applies)
    s.scope = Scope::Qwen38;
    s.choices = {"on", "off"};
  }
  {
    auto &s = add("XEN_KV_LEND_KEEP", "KV kept free below lent pages", cap, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.choices = {"8192", "32768", "131072"};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_KV_LEND_PARK_S", "Cached conversation timeout (lending)", cap, Kind::Float);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.hi = 86400;
    s.choices = {"0", "60", "300", "900"};
    s.names = {{"0", "never"}};
    s.unit = "s";
  }
  {
    auto &s = add("XEN_KV_LEND_PRESSURE", "Lending under KV pressure", cap, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"kv", "lru", "idle"};
  }
  {
    auto &s = add("XEN_KV_LEND_GRACE_S", "Cached conversation grace (pressure idle)", cap, Kind::Float);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.hi = 86400;
    s.choices = {"30", "60", "300"};
    s.unit = "s";
  }
  {
    auto &s = add("XEN_EXPERT_EXCHANGE", "Hot experts follow the traffic", cap, Kind::Choice);
    s.scope = Scope::Qwen38;
    s.choices = {"on", "off"};
  }
  {
    auto &s = add("XEN_PREFILL_COLD_STAGE", "Cold-expert staging", pf, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.choices = {"0", "16", "32", "64"};
    s.names = {{"0", "off"}};
    s.unit = "blobs";
  }
  {
    auto &s = add("XEN_PREFILL_FETCH", "Prompt experts fetched ahead", pf, Kind::Int);
    s.scope = Scope::Qwen38;
    s.lo = 0;
    s.hi = 512;
    s.choices = {"0", "96", "160"};
    s.names = {{"0", "off"}};
    s.unit = "blobs per layer";
  }
  {
    add("XEN_PREFIX_CACHE", "Prefix cache", pf, Kind::Bool);
  }
  {
    auto &s = add("XEN_PREFIX_CACHE_GIB", "Prefix cache host memory", pf, Kind::Float);
    s.lo = 0;
    s.hi = 256;
    s.choices = {"2", "4", "8", "16"};
    s.unit = "GiB";
  }
  {
    auto &s = add("XEN_PREFIX_MARK_MIN", "Prefix cache mark from", pf, Kind::Int);
    s.lo = 0;
    s.choices = {"0", "1024", "2048", "8192"};
    s.names = {{"0", "every chat prompt"}};
    s.unit = "tokens";
  }
  {
    add("XEN_PREFIX_LIVE", "Prefix cache live resumes", pf, Kind::Bool);
  }
  {
    auto &s = add("XEN_MAX_QUEUE", "Max queued requests", q, Kind::Int);
    s.lo = 0;
    s.names = {{"0", "unlimited"}};
  }
  {
    auto &s = add("XEN_QUEUE_POLICY", "Queue policy", q, Kind::Choice);
    s.choices = {"fifo", "sjf"};
    s.names = {{"fifo", "fifo (arrival order)"}, {"sjf", "sjf (smallest max_tokens)"}};
  }
  {
    auto &s = add("XEN_QUEUE_AGING", "SJF aging", q, Kind::Float);
    s.lo = 0;
    s.unit = "s";
  }
  {
    auto &s = add("XEN_QUEUE_TIMEOUT", "Queue timeout", q, Kind::Float);
    s.lo = 0;
    s.names = {{"0", "forever"}};
    s.unit = "s";
  }
  {
    auto &s = add("XEN_IDLE_TIMEOUT", "Idle keep-alive", q, Kind::Int);
    s.lo = 0;
    s.names = {{"0", "never"}};
    s.unit = "s";
  }
  // The sampling of requests that send no value of their own (empty = the checkpoint's)
  {
    auto &s = add("XEN_TEMPERATURE", "Temperature", smp, Kind::Float);
    s.allow_empty = true;
    s.lo = 0;
    s.hi = 100;
    s.choices = {"", "0", "0.3", "0.6", "0.7", "1.0"};
    s.names = {{"", "the checkpoint's"}, {"0", "greedy"}};
  }
  {
    auto &s = add("XEN_TOP_P", "Top-p", smp, Kind::Float);
    s.allow_empty = true;
    s.lo = 0.001;
    s.hi = 1;
    s.choices = {"", "0.8", "0.9", "0.95", "1"};
    s.names = {{"", "the checkpoint's"}, {"1", "off"}};
  }
  {
    auto &s = add("XEN_TOP_K", "Top-k", smp, Kind::Int);
    s.allow_empty = true;
    s.lo = -1;
    s.hi = 1 << 30;
    s.choices = {"", "0", "20", "40", "100"};
    s.names = {{"", "the checkpoint's"}, {"0", "off"}, {"-1", "off"}};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_MIN_P", "Min-p", smp, Kind::Float);
    s.allow_empty = true;
    s.lo = 0;
    s.hi = 1;
    s.choices = {"", "0", "0.01", "0.05", "0.1"};
    s.names = {{"", "the checkpoint's"}, {"0", "off"}};
  }
  {
    auto &s = add("XEN_PRESENCE_PENALTY", "Presence penalty", smp, Kind::Float);
    s.allow_empty = true;
    s.lo = -2;
    s.hi = 2;
    s.choices = {"", "0", "0.5", "1.0", "1.5"};
    s.names = {{"", "off"}, {"0", "off"}};
  }
  {
    auto &s = add("XEN_REPETITION_PENALTY", "Repetition penalty", smp, Kind::Float);
    s.allow_empty = true;
    s.lo = 0.01;
    s.hi = 100;
    s.choices = {"", "1", "1.05", "1.1"};
    s.names = {{"", "the checkpoint's"}, {"1", "off"}};
  }
  {
    auto &s = add("XEN_SAMPLE_THREADS", "Sampler threads", rt, Kind::Int);
    s.lo = 1;
    s.hi = 256;
    s.unit = "threads";
  }
  add("XEN_WARMUP", "Warm-up", rt, Kind::Bool);
  {
    auto &s = add("XEN_LOG_INTERVAL", "Status log interval", rt, Kind::Float);
    s.lo = 0;
    s.names = {{"0", "off"}};
    s.unit = "s";
  }
  add("XEN_STUB", "Stub engine (no GPU)", rt, Kind::Bool);
  add("XEN_REPORT_LAUNCH", "Report the launch (benchmarks)", dg, Kind::Bool);  // a start-server.sh setting, so not .diag
  {
    auto &s = add("XEN_PF_LOG", "Prefill call log", dg, Kind::Bool);
    s.diag = true;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_PF_PROFILE", "Prefill profile", dg, Kind::Choice);
    s.diag = true;
    s.scope = Scope::Qwen38;
    s.choices = {"", "1", "2", "3"};
    s.names = {{"", "off"}, {"1", "1 (host wall per phase)"}, {"2", "2 (device time, barriers)"}, {"3", "3 (GT stamps)"}};
  }
  {
    auto &s = add("XEN_QSA_PF_TILES", "Prompt attention tiling", dg, Kind::Choice);
    s.diag = true;
    s.scope = Scope::Qwen38;
    s.choices = {"", "1"};
    s.names = {{"", "8 tiles (default)"}, {"1", "1 (bits of the generation step)"}};
  }
  {
    auto &s = add("XEN_DEC_PROFILE", "Decode profile", dg, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen38;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_DRAFT_GRAPHS", "Drafter pass graphs kept", dg, Kind::Int);
    s.diag = true;
    s.lo = 4;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_STEP_GRAPHS_MAX", "Decode step graphs kept", dg, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 4;
    s.allow_empty = true;
    s.unit = "graphs";
  }
  {
    auto &s = add("XEN_GRAPH_ROOM_MIB", "Decode step graphs: VRAM room", dg, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.allow_empty = true;
    s.unit = "MiB";
  }
  {
    auto &s = add("XEN_PREFILL_SHARE", "Prompt time share while decoding", pf, Kind::Float);
    s.diag = true;
    s.lo = 0;
    s.hi = 1;
    s.allow_empty = true;
    s.choices = {"0.33", "0.5"};
  }
  {
    auto &s = add("XEN_TRAINED_QSA_DELTA_DIR", "Trained bank folder", tq, Kind::Path);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_TILED_SCORE", "Tiled scoring", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_POOL_FP16", "Block summaries in FP16", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_TREE", "Selection in draft trees", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_XMX_SCORE", "Scoring on XMX", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_GATHER", "Read only the selected blocks", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_RESIDENT_SNAPSHOTS", "Keep selection history in the prefix cache", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_RESIDENT_FORK_COPY", "Copy selection history to another slot", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_DECODE_MIN_POS", "Selection starts at position", tq, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.allow_empty = true;
    s.choices = {"0", "16384", "32768"};
    s.names = {{"0", "the 8,192-token threshold"}};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_TRAINED_QSA_PREFILL_XMX", "Selected prompt reading (XMX)", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_PREFILL_MIN_POS", "Selected prompt reading from", tq, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 2112;
    s.multiple = 64;
    s.allow_empty = true;
    s.choices = {"32768", "16384", "65536"};
    s.unit = "tokens";
  }
  {
    auto &s = add("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG", "Selected prompt reading with 4 slots", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_PREFILL_MERGED", "Selected reading in merged prompt calls", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_PF_SUBTILE_SKIP", "Skip unpicked prompt subtiles", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_SHARE_SCRATCH", "Shared selector scratch", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_TRAINED_QSA_STREAM_PROJECTION", "Scorer weights in host memory", tq, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_REQ_LOG", "Request log", dg, Kind::Bool);
    s.diag = true;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_DEPTH_TRACE", "Adaptive depth log", dg, Kind::Bool);
    s.diag = true;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_TRIAL_ROUTE_LOG", "Trained selection route log", dg, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_QSA_PACK_CHECK", "Gather pack check", dg, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.allow_empty = true;
    s.unit = "steps";
  }
  {
    auto &s = add("XEN_QSA_SCORE_CHECK", "XMX score check", dg, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen35;
    s.lo = 0;
    s.allow_empty = true;
    s.unit = "steps";
  }
  {
    auto &s = add("XEN_PF_DENSE_ONEDNN_LOG", "oneDNN dense prompt GEMM log", dg, Kind::Bool);
    s.diag = true;
    s.scope = Scope::Qwen38;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_PF_DENSE_ONEDNN_SLOTS", "oneDNN dense repack slots", dg, Kind::Int);
    s.diag = true;
    s.scope = Scope::Qwen38;
    s.lo = 1;
    s.hi = 64;
    s.allow_empty = true;
  }
  {
    auto &s = add("XEN_PF_ONEDNN_SLOTS", "oneDNN int4 repack slots", dg, Kind::Int);
    s.diag = true;
    s.lo = 1;
    s.hi = 512;
    s.allow_empty = true;
  }
  return v;
}

}  // namespace

const Spec *Catalog::find(std::string_view key) const {
  for (const auto &s : specs)
    if (s.key == key) return &s;
  return nullptr;
}

std::vector<ScriptVar> parse_script_defaults(std::string_view text) {
  // Lines of the form KEY=${KEY:-default} or KEY=${KEY-default}.
  std::vector<ScriptVar> out;
  std::istringstream in{std::string(text)};
  std::string line;
  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.rfind("XEN_", 0) != 0) continue;
    const size_t eq = t.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = t.substr(0, eq);
    const std::string open = "${" + key;
    if (t.compare(eq + 1, open.size(), open) != 0 || t.back() != '}') continue;
    size_t p = eq + 1 + open.size();
    if (p < t.size() && t[p] == ':') ++p;
    if (p >= t.size() || t[p] != '-') continue;
    out.push_back({key, t.substr(p + 1, t.size() - p - 2)});
  }
  return out;
}

std::vector<DocRow> parse_docs(std::string_view text) {
  std::vector<DocRow> out;
  std::istringstream in{std::string(text)};
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("| `XEN_", 0) != 0) continue;
    // Split on the table's '|', not on the "\|" escapes inside a cell.
    std::vector<std::string> cells;
    std::string cur;
    for (size_t i = 1; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == '|') {
        cur += '|';
        ++i;
      } else if (line[i] == '|') {
        cells.push_back(trim(cur));
        cur.clear();
      } else {
        cur += line[i];
      }
    }
    if (cells.size() == 4) out.push_back({strip_ticks(cells[0]), strip_ticks(cells[1]), strip_ticks(cells[2]), cells[3], false});
    else if (cells.size() == 2) out.push_back({strip_ticks(cells[0]), "", "", cells[1], true});
  }
  return out;
}

Catalog build_catalog(std::string_view script, std::string_view docs) {
  Catalog cat;
  cat.specs = schema();
  const auto vars = parse_script_defaults(script);
  const auto rows = parse_docs(docs);
  for (auto &s : cat.specs) {
    s.brief = brief_of(s.key);
    if (s.brief.empty()) cat.warnings.push_back(s.key + ": no brief description in settings.cpp");
    for (const auto &v : vars)
      if (v.key == s.key) {
        s.in_script = true;
        s.def = v.def;
      }
    for (const auto &r : rows)
      if (r.key == s.key) {
        s.in_docs = true;
        s.help = r.meaning;
        s.flag = r.flag;
        if (r.def == "empty") s.def = s.in_script ? s.def : "";
      }
    if (s.diag) {
      if (!s.in_docs) cat.warnings.push_back(s.key + ": not in docs/server-env.md's diagnostic table");
    } else {
      if (!s.in_script) cat.warnings.push_back(s.key + ": not read by scripts/start-server.sh");
      if (!s.in_docs) cat.warnings.push_back(s.key + ": no row in docs/server-env.md");
    }
  }
  // Settings the launcher has no type for: shown as plain text under "Other".
  auto add_unknown = [&](const std::string &key) -> Spec & {
    Spec s;
    s.key = key;
    s.label = key.substr(4);
    std::replace(s.label.begin(), s.label.end(), '_', ' ');
    for (size_t i = 1; i < s.label.size(); ++i) s.label[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s.label[i])));
    s.group = "Other";
    s.known = false;
    cat.specs.push_back(std::move(s));
    cat.warnings.push_back(key + ": not in the launcher's schema (shown as text under Other)");
    return cat.specs.back();
  };
  for (const auto &v : vars)
    if (!cat.find(v.key)) {
      auto &s = add_unknown(v.key);
      s.in_script = true;
      s.def = v.def;
    }
  for (const auto &r : rows) {
    Spec *s = const_cast<Spec *>(cat.find(r.key));
    if (!s) {
      s = &add_unknown(r.key);
      s->diag = r.diag;
    }
    if (!s->known) {
      s->in_docs = true;
      s->help = r.meaning;
      s->flag = r.flag;
    }
  }
  for (const auto &s : cat.specs)
    if (std::find(cat.groups.begin(), cat.groups.end(), s.group) == cat.groups.end()) cat.groups.push_back(s.group);
  // Display order, from what is chosen first to what rarely changes: what and where, how much, how it
  // decodes, how prompts are scheduled, the numerics, who may connect, the process, diagnostics.
  static const char *const order[] = {"Model & network", "Capacity & memory", "Speculative decoding", "Prompt processing & cache", "Numerics & kernels",
                                      "Trained page selection", "Sampling defaults", "Queue & connections", "Runtime", "Diagnostics", "Other"};
  auto rank = [&](const std::string &g) {
    for (size_t i = 0; i < std::size(order); ++i)
      if (g == order[i]) return i;
    return std::size(order);
  };
  std::stable_sort(cat.groups.begin(), cat.groups.end(), [&](const std::string &a, const std::string &b) { return rank(a) < rank(b); });
  return cat;
}

Catalog load_catalog(const std::string &root) {
  std::string script, docs;
  const bool a = read_text(join_path(root, "scripts/start-server.sh"), script);
  const bool b = read_text(join_path(root, "docs/server-env.md"), docs);
  Catalog cat = build_catalog(script, docs);
  if (!a) cat.warnings.insert(cat.warnings.begin(), "scripts/start-server.sh not readable: no defaults");
  if (!b) cat.warnings.insert(cat.warnings.begin(), "docs/server-env.md not readable: no help text");
  return cat;
}

Values defaults_of(const Catalog &cat) {
  Values v;
  for (const auto &s : cat.specs) v[s.key] = expand_home(s.def);
  return v;
}

// ---- env files ------------------------------------------------------------------------------

std::string home_dir() {
  const char *h = std::getenv("HOME");
  return h ? h : "";
}

std::string expand_home(std::string_view s) {
  const std::string h = home_dir();
  std::string out(s);
  if (out == "~" || out.rfind("~/", 0) == 0) out = h + out.substr(1);
  for (const char *pat : {"${HOME}", "$HOME"}) {
    size_t p;
    while ((p = out.find(pat)) != std::string::npos) out.replace(p, std::strlen(pat), h);
  }
  return out;
}

std::string tilde(std::string_view path) {
  const std::string h = home_dir();
  if (!h.empty() && path.size() >= h.size() && path.substr(0, h.size()) == h && (path.size() == h.size() || path[h.size()] == '/'))
    return "~" + std::string(path.substr(h.size()));
  return std::string(path);
}

std::string env_quote(std::string_view v) {
  bool plain = true;
  for (const char c : v)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || std::strchr("_./:@%+,=-", c))) plain = false;
  if (plain) return std::string(v);
  std::string out = "'";
  for (const char c : v) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  return out + "'";
}

std::string env_unquote(std::string_view raw) {
  std::string t = trim(raw);
  // A trailing comment after whitespace (bash: `KEY=v # note`).
  if (!t.empty() && t[0] != '\'' && t[0] != '"') {
    const size_t c = t.find(" #");
    if (c != std::string::npos) t = trim(t.substr(0, c));
  }
  if (t.size() >= 2 && t.front() == '\'' && t.back() == '\'') {
    std::string out = t.substr(1, t.size() - 2);
    size_t p;
    while ((p = out.find("'\\''")) != std::string::npos) out.replace(p, 4, "'");
    return out;
  }
  if (t.size() >= 2 && t.front() == '"' && t.back() == '"') return expand_home(t.substr(1, t.size() - 2));
  return expand_home(t);
}

namespace {
// KEY of a `KEY=...` or `export KEY=...` line, else "".
std::string line_key(const std::string &line, size_t *value_at = nullptr) {
  std::string t = line;
  size_t off = 0;
  while (off < t.size() && std::isspace(static_cast<unsigned char>(t[off]))) ++off;
  if (t.compare(off, 7, "export ") == 0) off += 7;
  size_t e = off;
  while (e < t.size() && (std::isalnum(static_cast<unsigned char>(t[e])) || t[e] == '_')) ++e;
  if (e == off || e >= t.size() || t[e] != '=') return "";
  if (value_at) *value_at = e + 1;
  return t.substr(off, e - off);
}
}  // namespace

EnvFile EnvFile::parse(std::string_view text) {
  EnvFile f;
  std::istringstream in{std::string(text)};
  std::string line;
  while (std::getline(in, line)) f.lines_.push_back(line);
  return f;
}

Values EnvFile::values() const {
  Values v;
  for (const auto &l : lines_) {
    size_t at = 0;
    const std::string k = line_key(l, &at);
    if (!k.empty()) v[k] = env_unquote(std::string_view(l).substr(at));
  }
  return v;
}

bool EnvFile::has(std::string_view key) const {
  for (const auto &l : lines_)
    if (line_key(l) == key) return true;
  return false;
}

void EnvFile::set(const std::string &key, const std::string &value) {
  const std::string line = key + "=" + env_quote(value);
  for (auto &l : lines_)
    if (line_key(l) == key) {
      l = line;
      return;
    }
  lines_.push_back(line);
}

void EnvFile::erase(std::string_view key) {
  lines_.erase(std::remove_if(lines_.begin(), lines_.end(), [&](const std::string &l) { return line_key(l) == key; }), lines_.end());
}

std::string EnvFile::text() const {
  std::string out;
  for (const auto &l : lines_) out += l + "\n";
  return out;
}

void EnvFile::apply(const Values &vals, const Values &defaults, const Catalog &cat, bool keep_present) {
  const Values cur = values();
  for (const auto &s : cat.specs) {
    const auto v = vals.find(s.key);
    if (v == vals.end()) continue;
    const auto c = cur.find(s.key);
    if (c != cur.end() && c->second == v->second) continue;  // unchanged: the line stays as written ($HOME, quotes, comments)
    const auto d = defaults.find(s.key);
    if (d != defaults.end() && d->second == v->second && !(keep_present && c != cur.end())) erase(s.key);
    else set(s.key, v->second);
  }
}

// ---- validation -----------------------------------------------------------------------------

std::string scope_name(Scope s) {
  switch (s) {
    case Scope::Qwen38: return "qwen38 only";
    case Scope::Qwen35: return "qwen35 only";
    case Scope::MoE: return "MoE models only";
    case Scope::Vision: return "qwen38 and qwen35 (the checkpoints with a vision tower)";
    default: return "every model";
  }
}

// arch llama (Nanbeige4.3) runs on the qwen35 runtime: the qwen35 settings apply except the drafter's, the prefill
// int4 switches (its zero-point matrices always run the dequantised W4A16 prefill GEMM), the two-stage head's tiles and the
// page-selected attention (head_dim 256 only). XEN_PREFILL_ATTN applies (auto = exact for llama: tools/server).
bool llama_skips(std::string_view key) {
  static const char *const skip[] = {"XEN_DRAFTER", "XEN_DRAFTER_MODEL", "XEN_DRAFTER_DEFER", "XEN_DRAFTER_CUT", "XEN_DRAFTER_TREE", "XEN_DRAFTER_TREE_SEQS", "XEN_DRAFTER_WIDE",
                                     "XEN_DRAFTER_TREE_WIDE", "XEN_SEG_ROWS", "XEN_DRAFT_DEPTHS", "XEN_DRAFTER_MTP_FROM", "XEN_DRAFTER_INT4",
                                     "XEN_DRAFT_HEAD", "XEN_DRAFT_VOCAB", "XEN_ADAPTIVE_DEPTH", "XEN_LOOKUP", "XEN_PREFILL_W4", "XEN_PREFILL_GEMM", "XEN_PREFILL_GEMM_ROWS",
                                     "XEN_PREFILL_A8_KEEP", "XEN_PREFILL_A8_BODY", "XEN_PREFILL_SHORT", "XEN_ATTN_SELECT", "XEN_SELECT_HEADS", "XEN_PREFILL_SELECT", "XEN_PREFILL_MERGE", "XEN_DECODE_ATTN", "XEN_DECODE_W4", "XEN_LM_HEAD_TILES"};
  for (const char *k : skip)
    if (key == k) return true;
  return false;
}

// arch flint (Recursive Flint INT4 on the qwen35 runtime): llama's skips but the prefill int4 numerics
// exact / dequant; a8 runs as dequant), plus its lm_head (the checkpoint's MLX int4, no switch); the prefill attention and
// the prefix cache apply(the many-row prefill, ring windows in the snapshots); it does use the step
// graphs (XEN_GRAPHS, otherwise qwen38's). The DFlash 2 drafter (Flint-DFlash2: chains, the
// whole lm_head, no MTP layer) and the settings it reads.
bool flint_skips(std::string_view key) {
  if (key == "XEN_PREFILL_W4") return false;
  if (key == "XEN_LM_HEAD" || key == "XEN_YARN_TIERS") return true;  // (no YaRN: rotary only inside its windows)
  for (const char *k : {"XEN_DRAFTER", "XEN_DRAFTER_MODEL", "XEN_DRAFTER_DEFER", "XEN_DRAFTER_CUT", "XEN_DRAFT_DEPTHS", "XEN_ADAPTIVE_DEPTH"})
    if (key == k) return false;
  return llama_skips(key);
}

namespace {

// The trained page selection's settings (the dense 27B with a trained bank) and its diagnostics.
bool trained_qsa_key(std::string_view key) {
  static const char *const keys[] = {"XEN_TRAINED_QSA_DELTA_DIR", "XEN_TRAINED_QSA_TILED_SCORE",
                                     "XEN_TRAINED_QSA_POOL_FP16", "XEN_TRAINED_QSA_TREE", "XEN_QSA_XMX_SCORE", "XEN_QSA_GATHER",
                                     "XEN_QSA_RESIDENT_SNAPSHOTS", "XEN_QSA_RESIDENT_FORK_COPY", "XEN_TRAINED_QSA_DECODE_MIN_POS",
                                     "XEN_TRAINED_QSA_PREFILL_XMX", "XEN_TRAINED_QSA_PREFILL_MIN_POS", "XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG",
                                     "XEN_TRAINED_QSA_PREFILL_MERGED", "XEN_PF_SUBTILE_SKIP", "XEN_TRAINED_QSA_SHARE_SCRATCH",
                                     "XEN_TRAINED_QSA_STREAM_PROJECTION", "XEN_QSA_TRIAL_ROUTE_LOG", "XEN_QSA_PACK_CHECK", "XEN_QSA_SCORE_CHECK"};
  for (const char *k : keys)
    if (key == k) return true;
  return false;
}

// Why a trained page selection setting does nothing with these values ("" = it is used). The server refuses the
// combinations noted "refused"; the launcher shows them as unused with the setting they need.
template <class Get>
std::string trained_qsa_unused(const std::string &key, Get v) {
  if (!trained_qsa_key(key) || key == "XEN_TRAINED_QSA_DELTA_DIR") return "";
  if (v("XEN_TRAINED_QSA_DELTA_DIR").empty()) return "needs a trained bank (XEN_TRAINED_QSA_DELTA_DIR)";
  auto on = [&](const char *k) { return v(k) == "1"; };
  if (key == "XEN_TRAINED_QSA_TREE" && !on("XEN_TRAINED_QSA_TILED_SCORE")) return "needs tiled scoring (XEN_TRAINED_QSA_TILED_SCORE 1; refused without)";
  if (key == "XEN_QSA_XMX_SCORE" && !on("XEN_TRAINED_QSA_POOL_FP16")) return "needs FP16 summaries (XEN_TRAINED_QSA_POOL_FP16 1)";
  if (key == "XEN_QSA_SCORE_CHECK" && (!on("XEN_QSA_XMX_SCORE") || !on("XEN_TRAINED_QSA_POOL_FP16"))) return "needs the XMX scorer (XEN_QSA_XMX_SCORE 1, FP16 summaries)";
  if (key == "XEN_QSA_RESIDENT_SNAPSHOTS" && v("XEN_PREFIX_CACHE") != "1") return "needs the prefix cache (XEN_PREFIX_CACHE 1)";
  if (key == "XEN_QSA_RESIDENT_FORK_COPY" && !on("XEN_QSA_RESIDENT_SNAPSHOTS")) return "needs XEN_QSA_RESIDENT_SNAPSHOTS 1";
  if (key == "XEN_TRAINED_QSA_PREFILL_XMX" && !v("XEN_PREFILL_SELECT").empty() && v("XEN_PREFILL_SELECT") != "0")
    return "off while the long-prompt page selection is on (XEN_PREFILL_SELECT)";
  if ((key == "XEN_TRAINED_QSA_PREFILL_MIN_POS" || key == "XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG" || key == "XEN_PF_SUBTILE_SKIP") &&
      !on("XEN_TRAINED_QSA_PREFILL_XMX"))
    return "needs the selected prompt reading (XEN_TRAINED_QSA_PREFILL_XMX 1; refused without)";
  if (key == "XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG" && v("XEN_SLOTS") != "4") return "needs four slots (XEN_SLOTS 4; refused otherwise)";
  if (key == "XEN_TRAINED_QSA_PREFILL_MERGED" && !on("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG"))
    return "needs XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG 1 (refused without)";
  return "";
}

}  // namespace

bool applies(const Spec &s, std::string_view arch) {
  if (arch == "llama" && llama_skips(s.key)) return false;
  if (arch == "flint" && flint_skips(s.key)) return false;
  if (arch == "flint" && s.key == "XEN_GRAPHS") return true;
  if (s.key == "XEN_PREFILL_GEMM") return arch.empty() || arch == "qwen35" || arch == "qwen38";  // Qwen38 too
  if (s.key == "XEN_PREFILL_WIRE" && arch == "qwen38") return true;
  if (s.key == "XEN_UNEQUAL_SEGMENTS" && arch == "qwen35") return true;
  if ((s.key == "XEN_KV_LEND" || s.key == "XEN_KV_LEND_KEEP") && arch == "qwen35") return true;
  if (arch == "qwen38" && (s.key == "XEN_DRAFTER_MODEL" || s.key == "XEN_DRAFTER_DEFER" || s.key == "XEN_DRAFTER_CUT")) return true;
  if (arch == "qwen38" && s.key == "XEN_DRAFTER_MTP_FROM") return true;  // Both drafters
  if (arch == "qwen38" && (s.key == "XEN_DRAFTER_TREE" || s.key == "XEN_DRAFTER_TREE_SEQS")) return true;  // DFlash trees
  if (arch == "qwen38" && s.key == "XEN_STEP_ROWS") return true;  // Qwen38's 32-row steps
  if (trained_qsa_key(s.key)) return arch.empty() || arch == "qwen35";  // the Qwen3.8-27B's selector, not llama / flint
  if (s.scope == Scope::Vision) return arch.empty() || arch == "qwen38" || arch == "qwen35";
  if (arch.empty() || s.scope == Scope::All) return true;
  return (s.scope == Scope::Qwen38 && arch == "qwen38") || (s.scope == Scope::MoE && arch == "qwen38") ||
         (s.scope == Scope::Qwen35 && (arch == "qwen35" || arch == "llama" || arch == "flint"));
}

std::string unused_reason(const Spec &s, std::string_view arch, const Values &vals) {
  if (arch.empty()) return "";
  auto v = [&](const char *k) {
    const auto it = vals.find(k);
    return it == vals.end() ? std::string() : it->second;
  };
  if (!applies(s, arch)) return "ignored for " + std::string(arch);
  if (v("XEN_STUB") == "1") {  // the stub engine sits behind the real server: only network, capacity numbers and queue
    static const char *const used[] = {"XEN_MODEL", "XEN_MODELS_DIR", "XEN_HOST", "XEN_PORT", "XEN_SERVED_NAME", "XEN_SLOTS", "XEN_CONTEXT", "XEN_MAX_QUEUE",
                                       "XEN_QUEUE_POLICY", "XEN_QUEUE_AGING", "XEN_QUEUE_TIMEOUT", "XEN_IDLE_TIMEOUT", "XEN_STUB"};
    bool uses = false;
    for (const char *k : used)
      if (s.key == k) uses = true;
    if (!uses) return "not used by the stub engine";
  }
  // Decode step graphs on the qwen35 runtime: flint always records them; the dense 27B and llama with XEN_STEP_GRAPHS
  if (s.key == "XEN_STEP_GRAPHS" && arch == "flint") return "flint always replays step graphs (XEN_GRAPHS)";
  if ((s.key == "XEN_STEP_GRAPHS" || s.key == "XEN_STEP_GRAPHS_MAX" || s.key == "XEN_GRAPH_ROOM_MIB") && v("XEN_GRAPHS") == "0")
    return "needs XEN_GRAPHS on";
  if ((s.key == "XEN_STEP_GRAPHS_MAX" || s.key == "XEN_GRAPH_ROOM_MIB") && arch != "flint" && v("XEN_STEP_GRAPHS") != "1")
    return "needs XEN_STEP_GRAPHS 1";
  if (s.key == "XEN_STEP_GRAPHS" && !v("XEN_ATTN_SELECT").empty() && v("XEN_ATTN_SELECT") != "0")
    return "not used with page-selected attention (XEN_ATTN_SELECT)";
  if (arch == "qwen38") {  // A drafter is XEN_MTP or the DFlash drafter (which replaces it); XEN_DRAFTER is qwen35's
    const bool df = !v("XEN_DRAFTER_MODEL").empty();
    if ((s.key == "XEN_DRAFT_VOCAB" || s.key == "XEN_ADAPTIVE_DEPTH" || s.key == "XEN_LOOKUP") && v("XEN_MTP").empty() && !df) return "needs a drafter (XEN_MTP or XEN_DRAFTER_MODEL)";
    // XEN_MTP beside the DFlash drafter loads both (XEN_DRAFTER_MTP_FROM > 0); the depth cap is DFlash's then
    if (s.key == "XEN_MTP" && df && v("XEN_DRAFTER_MTP_FROM") == "0") return "the DFlash drafter (XEN_DRAFTER_MODEL) replaces MTP (XEN_DRAFTER_MTP_FROM 0)";
    if (s.key == "XEN_MTP_DEPTH" && df) return "the DFlash drafter (XEN_DRAFTER_MODEL) sets the depth";
    if (s.key == "XEN_DRAFTER_MTP_FROM") return df && !v("XEN_MTP").empty() ? "" : "needs both drafters (XEN_DRAFTER_MODEL and XEN_MTP)";
    if (s.key == "XEN_MTP_DEPTH" && v("XEN_MTP").empty()) return "needs an MTP drafter (XEN_MTP)";
    if ((s.key == "XEN_DRAFTER_DEFER" || s.key == "XEN_DRAFTER_CUT") && !df) return "needs the DFlash drafter";
    if (s.key == "XEN_UNEQUAL_SEGMENTS" && !df && v("XEN_MTP").empty()) return "needs a drafter (XEN_MTP or XEN_DRAFTER_MODEL)";  // MTP too
    if (s.key == "XEN_UNEQUAL_SEGMENTS" && v("XEN_ADAPTIVE_DEPTH") == "0") return "needs XEN_ADAPTIVE_DEPTH on";
    if (s.key == "XEN_DRAFTER_MODEL" || s.key == "XEN_DRAFTER_DEFER" || s.key == "XEN_DRAFTER_CUT") return "";
  }
  if (arch == "flint") {  // Its only drafter is the DFlash checkpoint (no MTP layer)
    const bool df = v("XEN_DRAFTER") == "1" && !v("XEN_DRAFTER_MODEL").empty();
    if (s.key == "XEN_DRAFTER" && v("XEN_DRAFTER_MODEL").empty()) return "flint has no MTP layer: needs a DFlash drafter (XEN_DRAFTER_MODEL)";
    if ((s.key == "XEN_DRAFTER_DEFER" || s.key == "XEN_DRAFTER_CUT" || s.key == "XEN_DRAFT_DEPTHS" || s.key == "XEN_ADAPTIVE_DEPTH") && !df)
      return "needs the DFlash drafter (XEN_DRAFTER on + XEN_DRAFTER_MODEL)";
  }
  if (s.key == "XEN_MTP_DEPTH" && arch != "qwen38") {  // The dense 27B's checkpoint MTP layer; llama / flint have none
    if (arch != "qwen35") return "no MTP layer";
    if (v("XEN_DRAFTER") != "1") return "needs the drafter on";
    if (!v("XEN_DRAFT_DEPTHS").empty()) return "the per-stream depths (XEN_DRAFT_DEPTHS) set it";
    if (!v("XEN_DRAFTER_MODEL").empty() && v("XEN_DRAFTER_MTP_FROM") == "0") return "the block drafter drafts every step (XEN_DRAFTER_MTP_FROM 0)";
  }
  if (arch == "qwen35") {
    if (const std::string why = trained_qsa_unused(s.key, v); !why.empty()) return why;
    if ((s.key == "XEN_ATTN_SELECT" || s.key == "XEN_SELECT_HEADS") && v("XEN_KV") == "int8-v4")
      return "not used with the int8-v4 KV (launched as 0): the trained page selection selects there";
  }
  if ((s.key == "XEN_ADAPTIVE_DEPTH" || s.key == "XEN_LOOKUP") && arch != "qwen38" && v("XEN_DRAFTER") != "1") return "needs the drafter on";  // qwen35 too
  if (s.key == "XEN_DRAFT_VOCAB" && arch != "qwen38" && v("XEN_DRAFTER") != "1") return "needs the checkpoint drafter on";
  if (s.key == "XEN_UNEQUAL_SEGMENTS" && arch == "qwen35" && v("XEN_DRAFTER") != "1") return "needs the drafter on";
  if (s.key == "XEN_UNEQUAL_SEGMENTS" && arch == "qwen35" && v("XEN_ADAPTIVE_DEPTH") == "0") return "needs XEN_ADAPTIVE_DEPTH on";
  if (s.key == "XEN_DRAFTER_MODEL" && v("XEN_DRAFTER") != "1") return "needs the drafter on";
  if (s.key == "XEN_DRAFTER_DEFER" && v("XEN_DRAFTER") != "1") return "needs the drafter on";  // MTP's KV rows too
  if ((s.key == "XEN_DRAFTER_CUT" || s.key == "XEN_DRAFTER_TREE" || s.key == "XEN_DRAFTER_TREE_SEQS" || s.key == "XEN_DRAFTER_MTP_FROM" || s.key == "XEN_DRAFTER_WIDE" ||
       s.key == "XEN_DRAFTER_TREE_WIDE" || s.key == "XEN_DRAFTER_INT4") && (v("XEN_DRAFTER") != "1" || v("XEN_DRAFTER_MODEL").empty()))
    return "needs the DFlash drafter";
  if ((s.key == "XEN_SEG_ROWS" || s.key == "XEN_DRAFT_DEPTHS") && v("XEN_DRAFTER") != "1") return "needs the drafter on";
  // Qwen38 drafts chains beside the MTP layer (src/arch/qwen38/serve.cpp)
  if ((s.key == "XEN_DRAFTER_TREE" || s.key == "XEN_DRAFTER_TREE_SEQS") && arch == "qwen38" && !v("XEN_MTP").empty() && v("XEN_DRAFTER_MTP_FROM") != "" &&
      v("XEN_DRAFTER_MTP_FROM") != "0")
    return "chains beside the MTP layer (XEN_DRAFTER_MTP_FROM)";
  if ((s.key == "XEN_DRAFTER_WIDE" || s.key == "XEN_DRAFTER_TREE_WIDE") && v("XEN_SEG_ROWS") == "8") return "only with 16-row segments (XEN_SEG_ROWS 16, the default)";
  if (s.key == "XEN_DRAFTER_TREE_WIDE" && v("XEN_DRAFTER_WIDE") == "0") return "only with 16-row drafts (XEN_DRAFTER_WIDE)";
  if (s.key == "XEN_DRAFTER_TREE_WIDE" && v("XEN_DRAFTER_TREE") == "0") return "only with draft trees (XEN_DRAFTER_TREE)";
  if (s.key == "XEN_QUEUE_AGING" && v("XEN_QUEUE_POLICY") != "sjf") return "only with queue policy sjf";
  // Qwen38 TP4's experts are in 32-column quarters; the CPU kernels read 64-column halves (and on the
  // B70s every expert fits in VRAM at TP4: nothing is cold)
  if (s.key.rfind("XEN_MOE_CPU", 0) == 0 && arch == "qwen38" && v("XEN_Q38_TP") == "4") return "not at TP4 (32-column expert quarters)";
  if ((s.key == "XEN_KV_LEND" || s.key == "XEN_KV_LEND_KEEP") && arch == "qwen35") {
    if (v("XEN_PREFILL_W4") != "a8" || v("XEN_PREFILL_GEMM").rfind("onednn", 0) != 0) return "only with the prefill GEMMs on oneDNN (int4 GEMMs a8)";
    if (v("XEN_KV_POOL_TOKENS") == "0") return "needs the shared KV pool";
    if (s.key == "XEN_KV_LEND_KEEP" && v("XEN_KV_LEND") == "off") return "needs KV lending on";
  }
  if (s.key == "XEN_PREFILL_GEMM_ROWS" && (v("XEN_PREFILL_W4") != "a8" || v("XEN_PREFILL_GEMM").rfind("onednn", 0) != 0)) return "only with the prefill GEMMs on oneDNN (int4 GEMMs a8)";
  if (s.key == "XEN_PREFILL_A8_KEEP" && v("XEN_PREFILL_W4") != "a8") return "only with prefill int4 GEMMs a8";
  if (s.key == "XEN_PREFILL_A8_BODY" && v("XEN_PREFILL_W4") != "a8") return "only with prefill int4 GEMMs a8";
  if (s.key == "XEN_PREFILL_GEMM" && arch != "qwen38" && v("XEN_PREFILL_W4") != "a8") return "only with prefill int4 GEMMs a8";
  if (s.key == "XEN_PREFILL_GEMM" && arch == "qwen38" && v("XEN_PREFILL_CHUNK") != "0" && std::atoll(v("XEN_PREFILL_CHUNK").c_str()) < 2048)
    return "qwen38: only with a prefill chunk of 2,048+";
  if (s.key == "XEN_LM_HEAD_TILES" && v("XEN_LM_HEAD") != "int4-rescore" && v("XEN_LM_HEAD") != "auto") return "only with lm_head int4-rescore (or auto on a two-head directory)";
  if (s.key == "XEN_DRAFT_HEAD" && v("XEN_DRAFTER") != "1") return "needs the checkpoint drafter on";
  if ((s.key == "XEN_PREFIX_CACHE_GIB" || s.key == "XEN_PREFIX_MARK_MIN" || s.key == "XEN_PREFIX_LIVE") && v("XEN_PREFIX_CACHE") != "1")
    return "needs the prefix cache on";
  if (s.key == "XEN_RESERVE" && (arch == "qwen35" || arch == "llama" || arch == "flint") && v("XEN_KV_POOL_TOKENS") != "auto")
    return "qwen35 / llama / flint read it only with an auto KV pool";
  if ((s.key == "XEN_VISION_MAX_PIXELS" || s.key == "XEN_VISION_IMAGES" || s.key == "XEN_VISION_FFMPEG" || s.key.rfind("XEN_MEDIA", 0) == 0 ||
       s.key.rfind("XEN_VIDEO", 0) == 0) &&
      v("XEN_VISION") != "1")
    return "needs image input on";
  if (s.key.rfind("XEN_VIDEO", 0) == 0 && v("XEN_VISION_FFMPEG").empty()) return "needs a video decoder (XEN_VISION_FFMPEG)";
  if ((s.key == "XEN_MEDIA_DOMAINS" || s.key == "XEN_MEDIA_TIMEOUT") && v("XEN_MEDIA_URLS") == "0") return "needs images by link on";
  if (s.key == "XEN_CARD" && v("XEN_TP") != "1") return "only with TP1 (TP2 runs on cards 0 and 1)";
  if (s.key == "XEN_KV_SPLIT" && v("XEN_Q38_TP") != "4") return "only with TP4 (at TP1 / TP2 each card holds whole KV heads)";
  if (s.key == "XEN_Q38_CARD" && v("XEN_Q38_TP") != "1") return "only with TP1 (TP2 runs on cards 0 and 1, TP4 on 0-3)";
  if (s.key == "XEN_PREFILL_WIRE" && arch == "qwen38") return v("XEN_Q38_TP") == "1" ? "TP1 has no two-card sums" : "";
  if ((s.key == "XEN_WIRE" || s.key == "XEN_PREFILL_WIRE") && v("XEN_TP") == "1") return "TP1 has no two-card sums";
  return "";
}

namespace {
bool scales_rope(std::string_view arch) { return arch == "qwen35" || arch == "llama" || arch == "flint" || arch == "qwen38"; }
constexpr int64_t kMaxContext = 1048576;  // the attention kernels' limit (kernels:kQsaMaxTokens)
}  // namespace

int64_t max_context_of(const ModelInfo &m) { return scales_rope(m.arch) ? kMaxContext : 0; }

int64_t trained_positions_of(const ModelInfo &m) { return scales_rope(m.arch) ? m.max_positions : 0; }

double yarn_factor_for(int64_t context, int64_t trained) {
  if (trained <= 0 || context <= trained) return 1.0;
  return static_cast<double>((context * 100 + trained - 1) / trained) / 100.0;
}

std::string yarn_note(std::string_view arch, int64_t context, int64_t trained) {
  const int64_t cap = (std::max<int64_t>(context, 64) + 63) / 64 * 64;  // the load rounds the context up to 64
  if (trained <= 0 || cap <= trained) return "";
  if (arch == "flint") return "past the " + with_commas(trained) + " trained positions: no YaRN needed (its rotary layers attend within a window)";
  char f[32];
  std::snprintf(f, sizeof f, "%.2f", yarn_factor_for(cap, trained));
  return "past the " + with_commas(trained) + " trained positions: YaRN factor " + f + " on every request (numerics change, short requests too)";
}

int64_t max_hot_of(const ModelInfo &m) {
  // XEN_HOT counts the routed experts in VRAM over ALL MoE layers (TP2: every hot expert is split over both cards).
  if (is_moe_arch(m.arch) && m.experts > 0 && m.moe_layers > 0) return m.moe_layers * m.experts;
  return 0;
}

double card_vram_gib() {
  // Usable VRAM of the known cards (util/cards.hpp: B70 / B65 30.3, B60 22.3), by PCI device id from sysfs (xpu-smi's
  // memory path is dead on this host); the smallest card when they differ
  double best = -1;
  std::error_code ec;
  for (const auto &e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
    const std::string n = e.path().filename().string();
    if (n.size() < 5 || n.compare(0, 4, "card") != 0) continue;
    std::ifstream f(e.path() / "device" / "device");
    std::string s;
    if (!std::getline(f, s)) continue;
    char *end = nullptr;
    const unsigned long id = std::strtoul(s.c_str(), &end, 16);
    if (!end || end == s.c_str()) continue;
    if (const auto *c = xenithium::util::card_by_pci(static_cast<uint32_t>(id))) best = best < 0 ? c->l0_gib : std::min(best, c->l0_gib);
  }
  return best;
}

double bytes_per_expert_card(const ModelInfo &m) {
  if (m.hidden <= 0 || m.moe_inter <= 0 || m.bits <= 0) return 0;
  return 3.0 * m.hidden * m.moe_inter * (m.bits / 8.0) / 2.0;  // 3 matrices (gate/up/down), TP2 halves each card
}

int64_t hot_fit(const ModelInfo &m) {
  // A rough figure for when no memory plan is available (the server's --plan is the real one): the card minus ~3 GiB.
  if (!is_moe_arch(m.arch)) return 0;
  const double vram = card_vram_gib();
  const double bpe = bytes_per_expert_card(m);
  if (vram <= 0 || bpe <= 0) return 0;
  const int64_t total = static_cast<int64_t>((vram - 3.0) * 1073741824.0 / bpe);
  return total <= 0 ? 0 : std::min<int64_t>(total, max_hot_of(m));
}

std::string hot_description(const ModelInfo &m, const MemPlanView *plan) {
  if (!is_moe_arch(m.arch)) return "";
  std::string s;
  if (m.experts > 0 && m.moe_layers > 0)
    s += with_commas(m.experts * m.moe_layers) + " routed experts in all (" + std::to_string(m.experts) + " per layer x " + std::to_string(m.moe_layers) +
         " MoE layers); " +
         (plan && plan->cards == 1 ? std::string("each hot one is whole on the card")
          : plan && plan->cards == 4 ? std::string("each hot one is split over the four cards (a quarter of its bytes on each)")
                                     : std::string("each hot one is split over both cards (half of its bytes on each)"));
  if (plan && plan->ok) {
    char buf[200];
    std::snprintf(buf, sizeof buf, ". With these settings %s%s experts fit (%s GiB per card)", plan->exact ? "" : "~", with_commas(plan->hot_experts).c_str(),
                  fmt_gib(plan->hot_gib).c_str());
    s += buf;
  } else if (const int64_t fit = hot_fit(m); fit > 0) {
    s += ". Roughly " + with_commas(fit) + " fit an empty card";
  }
  return s + ". 0 / empty = auto (as many as fit)";
}

std::string fmt_gib(double g) {
  char b[32];
  std::snprintf(b, sizeof b, g >= 100 ? "%.0f" : "%.1f", g);
  return b;
}

bool moves_memory(std::string_view key) {
  static const char *const keys[] = {"XEN_MODEL", "XEN_SLOTS", "XEN_CONTEXT", "XEN_KV", "XEN_KV_SPLIT", "XEN_KV_POOL_TOKENS", "XEN_RESERVE", "XEN_TP", "XEN_Q38_TP", "XEN_EMU_CARD",
                                     "XEN_HOT", "XEN_MTP", "XEN_DRAFTER", "XEN_DRAFTER_MODEL", "XEN_DRAFTER_MTP_FROM", "XEN_DRAFTER_INT4",
                                     "XEN_SEG_ROWS", "XEN_DRAFT_VOCAB", "XEN_PREFILL_CHUNK", "XEN_PREFILL_GEMM", "XEN_PREFILL_W4", "XEN_PREFILL_ATTN", "XEN_ATTN_SELECT", "XEN_PREFILL_SELECT",
                                     "XEN_LM_HEAD", "XEN_VISION", "XEN_VISION_MAX_PIXELS", "XEN_VISION_IMAGES", "XEN_WARMUP", "XEN_PREFILL_COLD_STAGE", "XEN_PREFILL_FETCH",
                                     "XEN_PREFIX_CACHE", "XEN_PREFIX_CACHE_GIB"};
  for (const char *k : keys)
    if (key == k) return true;
  return false;
}

std::vector<std::string> whatif_values(const Spec &s, const std::string &current, const std::vector<std::string> &extra) {
  std::vector<std::string> v;
  auto add = [&](const std::string &x) {
    if (std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x);
  };
  if (s.kind == Kind::Bool) v = {"1", "0"};
  else if (s.kind == Kind::Choice) v = s.choices;
  else if (s.kind == Kind::Int || s.kind == Kind::Float) {
    for (const auto &c : s.choices) add(c);
    if (v.empty()) return v;  // a free number: nothing to compare with
    add(current);
    auto numeric = [](const std::string &x) {
      double d;
      return parse_num(x, d);
    };
    std::stable_sort(v.begin(), v.end(), [&](const std::string &a, const std::string &b) {  // words (auto) first, numbers ascending
      const bool na = numeric(a), nb = numeric(b);
      if (na != nb) return !na;
      double x = 0, y = 0;
      return na && parse_num(a, x) && parse_num(b, y) && x < y;
    });
  } else if (s.kind == Kind::Path && !extra.empty()) {
    if (s.allow_empty) add("");
    for (const auto &e : extra) add(e);
    add(current);
  }
  if (v.size() > 8) {  // keep the current value in a cut list
    const bool had = std::find(v.begin(), v.begin() + 8, current) != v.begin() + 8;
    v.resize(8);
    if (!had) v.back() = current;
  }
  return v.size() < 2 ? std::vector<std::string>{} : v;
}

std::string whatif_name(const Spec &s, const std::string &v) {
  if (s.kind == Kind::Bool) return v == "1" ? "on" : "off";
  if (v.empty()) return "(none)";
  if (s.kind == Kind::Path) {
    const size_t sl = v.find_last_of('/');
    return sl == std::string::npos ? v : v.substr(sl + 1);
  }
  double d;
  if (s.kind == Kind::Int && parse_num(v, d) && std::fabs(d) >= 10000) return with_commas(static_cast<int64_t>(d));
  return v;
}

std::string kv_capacity_text(int64_t kv_tokens, int64_t context, int slots) {
  if (kv_tokens <= 0 || context <= 0 || slots <= 0) return "";
  const int64_t full = kv_tokens / context;
  if (full >= slots) return "every one of the " + std::to_string(slots) + " slots can reach the full " + with_commas(context) + "-token max context at once";
  const int64_t each = kv_tokens / slots / 64 * 64;
  std::string t = full <= 0 ? "less than one request at the full " + with_commas(context) + "-token max context"
                            : std::to_string(full) + (full == 1 ? " request" : " requests") + " at the full " + with_commas(context) + "-token max context";
  if (slots > 1) t += ", or all " + std::to_string(slots) + " slots at " + with_commas(each) + " tokens each";
  return t;
}

std::string kv_delta_text(int64_t tokens, int64_t now) {
  if (tokens <= 0 || now <= 0) return "";
  const double pct = 100.0 * (static_cast<double>(tokens) - static_cast<double>(now)) / static_cast<double>(now);
  if (std::fabs(pct) < 0.05) return "same";
  char b[32];
  std::snprintf(b, sizeof b, std::fabs(pct) < 9.95 ? "%+.1f %%" : "%+.0f %%", pct);
  return b;
}

bool parse_plan(const std::string &text, MemPlanView &p) {
  p = {};
  json::Value v;
  if (!json::parse(text, v) || !v.is_object()) return false;
  auto num = [&](const char *k) {
    const json::Value *x = v.get(k);
    return x && x->is_number() ? x->num : 0.0;
  };
  if (const json::Value *a = v.get("arch"); a && a->is_string()) p.arch = a->str;
  if (const json::Value *e = v.get("exact"); e) p.exact = e->is_bool() ? e->b : true;
  p.card_gib = num("card_gib"), p.vram_used_gib = num("vram_used_gib"), p.weights_gib = num("weights_gib"), p.kv_gib = num("kv_gib");
  p.state_gib = num("state_gib"), p.scratch_gib = num("scratch_gib"), p.hot_gib = num("hot_gib");
  p.hot_experts = static_cast<int64_t>(num("hot_experts"));
  p.experts_total = static_cast<int64_t>(num("experts_total")), p.moe_layers = static_cast<int>(num("moe_layers"));
  p.cards = static_cast<int>(num("cards"));
  p.kv_tokens = static_cast<int64_t>(num("kv_tokens")), p.context = static_cast<int64_t>(num("context")), p.slots = static_cast<int>(num("slots"));
  p.cold_gib = num("cold_gib"), p.embed_gib = num("embed_gib"), p.pinned_gib = num("pinned_gib"), p.prefix_gib = num("prefix_gib");
  p.host_total_gib = num("host_total_gib"), p.host_available_gib = num("host_available_gib");
  p.host_other_gib = v.get("host_other_gib") ? num("host_other_gib") : util::kLoadHostOverheadGib;
  p.host_keep_gib = v.get("host_keep_gib") ? num("host_keep_gib") : util::kHostKeepGib;
  p.ok = p.card_gib > 0 && !p.arch.empty();
  return p.ok;
}

std::vector<Issue> plan_issues(const MemPlanView &p, const Values &vals, bool server_running) {
  std::vector<Issue> out;
  if (!p.ok) return out;
  auto get = [&](const std::string &k) {
    const auto it = vals.find(k);
    return it == vals.end() ? std::string() : it->second;
  };
  const bool pinned_hot = !get("XEN_HOT").empty() && get("XEN_HOT") != "0";
  // Host RAM: the cold experts are pinned at load; past MemAvailable the server refuses to load (it used to get
  // OOM-killed, taking other processes with it).
  const double room = p.host_room_gib();
  if (p.host_available_gib > 0 && p.pinned_gib > room) {
    char b[400];
    std::snprintf(b, sizeof b,
                  "the cold experts need %s GiB of pinned host RAM, %s GiB are free (%.1f of them kept for the server process and the system)%s: hold "
                  "more experts in VRAM (fewer slots, a shorter context or a smaller KV pool leave room for them)%s",
                  fmt_gib(p.pinned_gib).c_str(), fmt_gib(p.host_available_gib).c_str(), p.host_other_gib + p.host_keep_gib,
                  server_running ? "; a running server's memory is counted, stop it first" : "",
                  pinned_hot ? ", or raise this count" : "");
    out.push_back({"XEN_HOT", b, !server_running});
  } else if (p.host_available_gib > 0 && get("XEN_PREFIX_CACHE") == "1" && p.prefix_gib > 0 && p.pinned_gib + p.prefix_gib > room) {
    const double left = std::max(0.0, room - p.pinned_gib);
    out.push_back({"XEN_PREFIX_CACHE_GIB",
                   "only ~" + fmt_gib(left) + " GiB of host RAM stay free after the cold experts; the server cuts the prefix cache to that (fewer remembered conversations)",
                   false});
  }
  // VRAM: a pinned hot count past what fits is cut by the load; before, the driver evicted memory and every
  // step ran 30-70x slower). The plan reports the cut count.
  if (pinned_hot) {
    const long long asked = std::atoll(get("XEN_HOT").c_str());
    if (asked > p.hot_experts && p.hot_experts < p.experts_total)
      out.push_back({"XEN_HOT",
                     "only ~" + with_commas(p.hot_experts) + " of these " + with_commas(asked) + " hot experts fit the VRAM left (after the KV pool, drafter, scratch " +
                         "and the reserve): the load cuts the count to that; auto (empty) does the same",
                     false});
  }
  return out;
}

std::string check_value(const Spec &s, const std::string &v, std::string_view arch, const std::string &root, bool *warn_only, int64_t max_context) {
  if (warn_only) *warn_only = false;
  if (v.empty() && (s.allow_empty || s.kind == Kind::Text || !s.known)) {
    if (s.key == "XEN_SERVED_NAME" || s.key == "XEN_HOST") return "must not be empty";
    return "";
  }
  if (!s.known) return "";
  if (v.empty() && s.kind != Kind::Choice) return "required";
  switch (s.kind) {
    case Kind::Int: {
      if (s.allow_auto && v == "auto") return "";
      int64_t x = 0;
      if (!parse_int(v, x)) return s.allow_auto ? "a whole number or auto" : "a whole number";
      if (s.key == "XEN_KV_POOL_TOKENS" && x == 0) return "auto or a token count (0, a fixed context per slot, was removed: the slots share the pool)";
      double hi = s.hi;
      if (s.key == "XEN_SLOTS" && arch != "qwen38" && arch != "qwen35") hi = 8;  // qwen38 / qwen35 1..16, llama / flint 1..8
      else if (s.key == "XEN_SLOTS" && arch == "qwen35") hi = 16;  // 17-64 on qwen38 (four cards: checked with XEN_Q38_TP)
      if (s.key == "XEN_DRAFTER_MTP_FROM" && arch != "qwen38") hi = 16;  // Qwen38 0..64
      if (x < s.lo || x > hi)
        return s.hi >= 1e300 ? "at least " + fmt_num(s.lo) : "between " + fmt_num(s.lo) + " and " + fmt_num(hi);
      if (s.multiple > 0 && x % s.multiple) return "a multiple of " + std::to_string(s.multiple);
      if (s.key == "XEN_DRAFT_VOCAB" && arch == "qwen38" && x % 64) return "a multiple of 64 for qwen38";
      if (s.key == "XEN_DRAFTER_DEFER" && x > 0 && x < 8) return "0 or 8..512 (a step's context rows must fit)";
      if (s.key == "XEN_CONTEXT" && max_context > 0 && (std::max<int64_t>(x, 64) + 63) / 64 * 64 > max_context)
        return "at most " + with_commas(max_context) + " for this model (the attention kernels' limit)";
      return "";
    }
    case Kind::Float: {
      double x = 0;
      if (!parse_num(v, x)) return "a number";
      if (x < s.lo || x > s.hi) return s.hi >= 1e300 ? "at least " + fmt_num(s.lo) : "between " + fmt_num(s.lo) + " and " + fmt_num(s.hi);
      return "";
    }
    case Kind::Choice:
      if (std::find(s.choices.begin(), s.choices.end(), v) == s.choices.end()) {
        std::string all;
        for (const auto &c : s.choices) all += (all.empty() ? "" : ", ") + (c.empty() ? std::string("(empty)") : c);
        return "one of " + all;
      }
      if (s.key == "XEN_KV" && v == "int8-v4" && !arch.empty() && arch != "qwen38" && arch != "qwen35")  // the dense 27B and qwen38
        return "int8-v4 is built for qwen38 and the dense 27B (int8 or bf16 here)";
      if (s.key == "XEN_MTP_DEPTH" && arch == "qwen38" && v > "3") return "1-3 for qwen38 (4-6 are the Qwen3.8-27B's)";
      if (s.key == "XEN_WIRE" && v == "int8" && !arch.empty() && arch != "qwen35")
        return "the int8 wire is built for the dense 27B (bf16 or fp32 here)";
      return "";
    case Kind::Bool:
      if (v == "0" || v == "1") return "";
      if (s.allow_empty && v.empty()) return "";
      return "0 or 1";
    case Kind::Path: {
      if (s.path_auto && v == "auto") return "";  // the model's own file; its state is shown on the model card
      if (s.key == "XEN_VISION_FFMPEG") {  // A program (a name looked up on PATH, or a path), not a folder
        if (v.empty()) return "";
        xenithium::util::VideoTools vt;
        std::string why;
        if (xenithium::util::find_video_tools(v, vt, why)) return "";
        if (warn_only) *warn_only = true;  // the server starts anyway, with video off
        return vt.ffmpeg.empty() ? (v.find('/') == std::string::npos ? "not on PATH: video stays off (pick a found ffmpeg or give its path)"
                                                                      : "no ffmpeg program at this path: video stays off")
                                 : "no ffprobe beside it or on PATH: video stays off";
      }
      const std::string p = v[0] == '/' ? v : join_path(root, v);
      if (s.path_file) {
        // The ranking is written by the server itself: a file that does not exist yet is the normal first run.
        if (s.key == "XEN_HOT_RANK") return "";
        if (!is_file(p)) return "file not found";
        return "";
      }
      if (!is_dir(p)) return "directory not found";
      if (s.key == "XEN_MODEL" && !is_file(join_path(p, "config.json"))) return "no config.json in this directory";
      if (QsaBank b; s.key == "XEN_TRAINED_QSA_DELTA_DIR" && !probe_qsa_bank(p, b))  // the server refuses it
        return "no trained indexer tensors (*.safetensors with _qsa_indexer) in this folder";
      if (s.key == "XEN_DRAFTER_MODEL" && !is_dflash_dir(p)) {
        if (warn_only) *warn_only = true;
        return "config.json names no block drafter (" + xenithium::arch::block_drafter_arch_names() + ")";
      }
      if (s.key == "XEN_MTP" && !is_file(join_path(p, "mtp-dense.safetensors"))) {
        if (warn_only) *warn_only = true;
        return "no mtp-dense.safetensors: not an MTP drafter checkpoint?";
      }
      return "";
    }
    case Kind::Text: return "";
  }
  return "";
}

std::vector<std::string> ffmpeg_candidates() {
  std::vector<std::string> out;
  std::set<std::string> seen;
  auto take = [&](const std::string &p) {
    xenithium::util::VideoTools vt;
    std::string why;
    if (!xenithium::util::find_video_tools(p, vt, why)) return;
    std::error_code ec;
    const std::string real = std::filesystem::canonical(vt.ffmpeg, ec).string();
    if (seen.insert(ec ? vt.ffmpeg : real).second) out.push_back(vt.ffmpeg);
  };
  take(std::string(xenithium::util::kDefaultFfmpeg));  // PATH (else the first install place the server would pick)
  for (const std::string &p : xenithium::util::ffmpeg_install_paths()) take(p);
  return out;
}

std::vector<Issue> validate(const Catalog &cat, const Values &vals, std::string_view arch, const std::string &root, int64_t max_context, int64_t max_hot,
                            int64_t trained) {
  std::vector<Issue> out;
  auto get = [&](const std::string &k) {
    const auto it = vals.find(k);
    return it == vals.end() ? std::string() : it->second;
  };
  for (const auto &s : cat.specs) {
    if (!applies(s, arch)) continue;
    bool warn = false;
    const std::string m = check_value(s, get(s.key), arch, root, &warn, max_context);
    if (!m.empty()) out.push_back({s.key, m, !warn});
  }
  // Cross-setting rules.
  int64_t pool = 0, ctx = 0;
  if (parse_int(get("XEN_KV_POOL_TOKENS"), pool) && parse_int(get("XEN_CONTEXT"), ctx) && pool > 0 && pool < ctx)
    out.push_back({"XEN_KV_POOL_TOKENS", "must hold at least one context (" + with_commas(ctx) + " tokens)", true});
  if (parse_int(get("XEN_CONTEXT"), ctx) && (max_context <= 0 || ctx <= max_context))
    if (const std::string n = yarn_note(arch, ctx, trained); !n.empty()) out.push_back({"XEN_CONTEXT", n, false});
  // The trained bank's start-up needs (tools/server refuses a start without them). Only settings that are in use: a
  // setting shown as unused goes back to its default at launch (decode page selection on int8-v4, a trained switch
  // whose base switch is off), so it never blocks one.
  if (arch == "qwen35" && !get("XEN_TRAINED_QSA_DELTA_DIR").empty()) {
    int64_t cx = 0;
    if (get("XEN_KV") != "int8-v4") out.push_back({"XEN_TRAINED_QSA_DELTA_DIR", "the trained page selection runs on the int8-v4 KV: set XEN_KV int8-v4", true});
    if (get("XEN_DRAFTER") != "1" || get("XEN_DRAFTER_MODEL").empty())
      out.push_back({"XEN_TRAINED_QSA_DELTA_DIR", "the trained page selection needs a block drafter (XEN_DRAFTER on + XEN_DRAFTER_MODEL)", true});
    if (parse_int(get("XEN_CONTEXT"), cx) && (cx <= 2048 || cx > 262144))
      out.push_back({"XEN_TRAINED_QSA_DELTA_DIR", "the trained page selection needs a max context of 2,049-262,144", true});
    if (get("XEN_TRAINED_QSA_PREFILL_XMX") == "1" && get("XEN_SLOTS") != "1" && get("XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG") != "1" &&
        (get("XEN_PREFILL_SELECT").empty() || get("XEN_PREFILL_SELECT") == "0"))
      out.push_back({"XEN_TRAINED_QSA_PREFILL_XMX", "the selected prompt reading needs one slot (four with XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG)", true});
  }
  if (arch == "qwen38") {  // Qwen38's step rows (as tools/server checks them)
    int64_t slots = 0;
    if (get("XEN_STEP_ROWS") == "8" && parse_int(get("XEN_SLOTS"), slots) && slots > 8) out.push_back({"XEN_STEP_ROWS", "8-row steps hold at most 8 streams: XEN_SLOTS 1..8", true});
    if (get("XEN_STEP_ROWS") == "64" && get("XEN_Q38_TP") != "4") out.push_back({"XEN_STEP_ROWS", "64-row steps need four cards: XEN_Q38_TP 4", true});
    if (parse_int(get("XEN_SLOTS"), slots) && slots > 16 && get("XEN_Q38_TP") != "4") out.push_back({"XEN_SLOTS", "more than 16 slots need four cards: XEN_Q38_TP 4", true});
  }
  if (arch != "qwen38" && get("XEN_STEP_ROWS") == "64") out.push_back({"XEN_STEP_ROWS", "64-row steps run on qwen38 only (four cards)", true});
  if (arch == "qwen35" || arch == "flint") {  // The step rows (as tools/server checks them); flint 8
    const int step = get("XEN_STEP_ROWS") == "8" || arch == "flint" ? 8 : get("XEN_STEP_ROWS") == "16" ? 16 : 32,  // 0 / unset = 32 (the server's)
              seg = arch == "flint" ? 8 : get("XEN_SEG_ROWS") == "8" ? 8 : 16;  // Unset = 16
    int64_t slots = 0;
    if (seg > step) out.push_back({"XEN_SEG_ROWS", "16 rows per sequence need XEN_STEP_ROWS 16 or 32", true});
    if (step == 8 && parse_int(get("XEN_SLOTS"), slots) && slots > 8) out.push_back({"XEN_STEP_ROWS", "8-row steps hold at most 8 streams: XEN_SLOTS 1..8", true});
    std::vector<int> dd;
    if (!serve::parse_draft_depths(get("XEN_DRAFT_DEPTHS"), dd)) out.push_back({"XEN_DRAFT_DEPTHS", "up to 16 comma-separated numbers 0..15 (15,7,4,3)", true});
    for (size_t i = 0; i < dd.size(); ++i) {
      if (dd[i] > seg - 1) {
        out.push_back({"XEN_DRAFT_DEPTHS", std::to_string(dd[i]) + " guesses need XEN_SEG_ROWS 16 (at most 7 with 8)", true});
        break;
      }
      if (static_cast<int>(i + 1) * (dd[i] + 1) > step) {
        out.push_back({"XEN_DRAFT_DEPTHS", std::to_string(i + 1) + " streams x " + std::to_string(dd[i] + 1) + " rows exceed XEN_STEP_ROWS " + std::to_string(step), true});
        break;
      }
    }
  }
  int64_t port = 0;
  if (parse_int(get("XEN_PORT"), port) && port > 0 && port < 1024) out.push_back({"XEN_PORT", "below 1024: needs root", false});
  if (is_moe_arch(arch) && max_hot > 0) {
    int64_t h = 0;
    if (parse_int(get("XEN_HOT"), h) && h > max_hot)
      out.push_back({"XEN_HOT", "more than this model's " + with_commas(max_hot) + " routed experts in all; the load caps a higher count", false});
  }
  if (arch.empty()) {
    const std::string m = get("XEN_MODEL");
    if (!m.empty() && is_dir(m) && is_file(join_path(m, "config.json")))
      out.push_back({"XEN_MODEL", "not an architecture this engine serves", get("XEN_STUB") == "1" ? false : true});
  }
  return out;
}

std::string with_commas(int64_t v) {
  std::string s = std::to_string(v < 0 ? -v : v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
  return v < 0 ? "-" + s : s;
}

std::string human_bytes(uint64_t b) {
  char buf[32];
  const double g = static_cast<double>(b) / (1024.0 * 1024 * 1024);
  if (g >= 100) std::snprintf(buf, sizeof buf, "%.0f GiB", g);
  else if (g >= 1) std::snprintf(buf, sizeof buf, "%.1f GiB", g);
  else if (b >= 1024 * 1024) std::snprintf(buf, sizeof buf, "%.0f MiB", static_cast<double>(b) / (1024.0 * 1024));
  else std::snprintf(buf, sizeof buf, "%.0f KiB", static_cast<double>(b) / 1024.0);
  return buf;
}

std::string group_brief(std::string_view group) {
  static const std::pair<const char *, const char *> t[] = {
      {"Model & network", "What is served and where clients reach it: the checkpoint, its name in the API, the chat template, the address and port."},
      {"Capacity & memory", "How much the server holds at once: parallel requests (slots), the longest conversation, the KV cache, and for MoE models how many experts live in GPU memory and which ones."},
      {"Speculative decoding", "A small drafter guesses the next tokens and the model checks several at once: the same output, up to ~2x faster for one or two streams."},
      {"Prompt processing & cache", "How prompts are read (prefill) while others are generating, and the prefix cache that lets a conversation's next turn skip what was already read."},
      {"Numerics & kernels", "Precision and kernel choices. The defaults are measured and safe; change them for A/B tests. Settings marked as changing numerics alter the output slightly."},
      {"Queue & connections", "What happens when every slot is busy: how many requests may wait, in which order, and for how long."},
      {"Sampling defaults", "How requests that do not choose their own sampling get sampled: temperature, top-p / top-k / min-p and the penalties. A request's own values always win; empty = the model's recommended values (generation_config.json)."},
      {"Runtime", "The server process itself: sampler threads, warm-up, status log, and a GPU-less stub engine for API tests."},
      {"Diagnostics", "Extra logging, profiling and reporting for debugging and benchmarks; the logging and profiling slow the server down. Off for normal use."},
      {"Other", "Settings the script or docs know but the launcher has no description for yet; edited as plain text."},
  };
  for (const auto &[g, d] : t)
    if (group == g) return d;
  return "";
}

bool is_moe_arch(std::string_view arch) { return arch == "qwen38"; }

std::string experts_dir_of(const std::string &experts_root, const ModelInfo &m) { return m.name.empty() ? "" : join_path(experts_root, m.name); }

namespace {
// XRC1 header: magic, layers, experts (model/route_counts.hpp).
bool xrc_header(const std::string &path, int &layers, int &experts) {
  std::ifstream f(path, std::ios::binary);
  uint32_t magic = 0;
  int32_t l = 0, e = 0;
  if (!f.read(reinterpret_cast<char *>(&magic), 4) || !f.read(reinterpret_cast<char *>(&l), 4) || !f.read(reinterpret_cast<char *>(&e), 4)) return false;
  if (magic != 0x31435258u || l <= 0 || e <= 0) return false;
  layers = l, experts = e;
  return true;
}
bool xrc_counts(const std::string &path, const ModelInfo &m, std::vector<double> &c) {
  int l = 0, e = 0;
  if (!xrc_header(path, l, e) || l != m.layers || e != m.experts) return false;
  std::ifstream f(path, std::ios::binary);
  f.seekg(12);
  c.resize(static_cast<size_t>(l) * e);
  return static_cast<bool>(f.read(reinterpret_cast<char *>(c.data()), static_cast<std::streamsize>(c.size() * 8)));
}
}  // namespace

ExpertFile expert_file(const std::string &value, const std::string &file, const std::string &root, const std::string &experts_root, const ModelInfo &m) {
  ExpertFile f;
  f.is_auto = value == "auto";
  if (f.is_auto) f.path = m.name.empty() ? "" : join_path(experts_dir_of(experts_root, m), file);
  else if (!value.empty()) f.path = value[0] == '/' ? value : join_path(root, expand_home(value));
  if (f.path.empty()) return f;
  struct stat st{};
  if (stat(f.path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return f;
  f.exists = true;
  f.bytes = static_cast<uint64_t>(st.st_size);
  f.mtime = static_cast<int64_t>(st.st_mtime);
  if (!xrc_header(f.path, f.layers, f.experts)) {
    f.error = "not an XRC1 routing-counts file";
    return f;
  }
  if (m.layers > 0 && m.experts > 0 && (f.layers != m.layers || f.experts != m.experts)) {
    f.error = "made for another model (" + std::to_string(f.layers) + " layers x " + std::to_string(f.experts) + " experts, this one has " +
              std::to_string(m.layers) + " x " + std::to_string(m.experts) + ")";
    return f;
  }
  f.shape_ok = static_cast<uint64_t>(12) + static_cast<uint64_t>(f.layers) * f.experts * 8 <= f.bytes;
  if (!f.shape_ok) f.error = "a short file";
  return f;
}

bool rank_coverage(const std::string &path, const ModelInfo &m, int64_t hot, double &ranked, double &fallback) {
  std::vector<double> c;
  if (hot <= 0 || !is_moe_arch(m.arch) || !xrc_counts(path, m, c)) return false;
  const int L = static_cast<int>(m.layers), E = static_cast<int>(m.experts);
  double total = 0;
  for (const double x : c) total += x;
  if (total <= 0) return false;
  double r = 0, d = 0;
  {  // the global top `hot` (layer, expert) pairs vs hot / layers of the lowest ids per layer
    const int64_t h = std::min<int64_t>(hot, static_cast<int64_t>(L) * E);
    std::vector<double> all = c;
    std::partial_sort(all.begin(), all.begin() + h, all.end(), std::greater<double>());
    for (int64_t i = 0; i < h; ++i) r += all[static_cast<size_t>(i)];
    for (int l = 0; l < L; ++l) {
      const int64_t n = std::min<int64_t>(h / L + (l < h % L ? 1 : 0), E);
      for (int64_t e = 0; e < n; ++e) d += c[static_cast<size_t>(l) * E + static_cast<size_t>(e)];
    }
  }
  ranked = r / total, fallback = d / total;
  return true;
}

std::string age_text(int64_t mtime) {
  const int64_t s = std::max<int64_t>(0, static_cast<int64_t>(std::time(nullptr)) - mtime);
  if (s < 90) return "just now";
  if (s < 90 * 60) return std::to_string(s / 60) + " min ago";
  if (s < 36 * 3600) return std::to_string(s / 3600) + " h ago";
  return std::to_string(s / 86400) + " days ago";
}

std::string display_value(const Spec &s, const std::string &v) {
  for (const auto &[val, name] : s.names)
    if (val == v) return name;
  if (s.kind == Kind::Bool) return v == "1" ? "on" : v == "0" || v.empty() ? "off" : v;
  if (s.kind == Kind::Path) return tilde(v);
  int64_t x = 0;
  if (s.kind == Kind::Int && parse_int(v, x)) return with_commas(x) + (s.unit.empty() ? "" : " " + s.unit);
  if (s.kind == Kind::Float && !v.empty() && !s.unit.empty()) return v + " " + s.unit;
  return v;
}

// ---- files and models -----------------------------------------------------------------------

bool read_text(const std::string &path, std::string &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

bool is_pointer_env(const std::string &path) {
  if (path.size() >= 10 && path.substr(path.size() - 10) == "server.env" && (path.size() == 10 || path[path.size() - 11] == '/')) return true;
  std::string text;
  return read_text(path, text) && EnvFile::parse(text).has("XEN_MODEL_ENV");
}

bool write_text_atomic(const std::string &path, const std::string &text, std::string *err) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      if (err) *err = "cannot write " + tmp + ": " + std::strerror(errno);
      return false;
    }
    f << text;
    if (!f) {
      if (err) *err = "write failed: " + tmp;
      return false;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    if (err) *err = "rename to " + path + ": " + std::strerror(errno);
    return false;
  }
  return true;
}

bool is_dir(const std::string &p) {
  struct stat st{};
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_file(const std::string &p) {
  struct stat st{};
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) return b;
  if (!b.empty() && b[0] == '/') return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}

bool same_path(const std::string &a, const std::string &b) {
  if (a.empty() || b.empty()) return false;
  char ra[PATH_MAX], rb[PATH_MAX];
  const bool ka = realpath(a.c_str(), ra) != nullptr, kb = realpath(b.c_str(), rb) != nullptr;
  if (ka && kb) return std::strcmp(ra, rb) == 0;
  std::string x = a, y = b;  // a path that does not exist (yet): compare the text without trailing slashes
  while (x.size() > 1 && x.back() == '/') x.pop_back();
  while (y.size() > 1 && y.back() == '/') y.pop_back();
  return x == y;
}

std::string model_env_model(const std::string &env_path) {
  std::string text;
  if (!read_text(env_path, text)) return "";
  const Values v = EnvFile::parse(text).values();
  const auto it = v.find("XEN_MODEL");
  return it == v.end() ? "" : it->second;
}

std::string setup_key_of(std::string_view arch) {
  if (arch == "qwen38") return "XEN_Q38_TP";
  if (arch == "qwen35" || arch == "llama" || arch == "flint") return "XEN_TP";
  return "";
}

bool is_shared_setting(const Spec &s) { return s.group == "Model & network" || s.key == "XEN_Q38_TP" || s.key == "XEN_TP"; }

std::string setup_env_path(const std::string &model_env, const std::string &setup_value) {
  if (model_env.size() <= 4 || model_env.substr(model_env.size() - 4) != ".env" || setup_value.empty()) return "";
  return model_env.substr(0, model_env.size() - 4) + ".tp" + setup_value + ".env";
}

bool is_setup_env(const std::string &path) {
  const std::string n = path.substr(path.find_last_of('/') + 1);
  const size_t t = n.rfind(".tp");
  if (t == std::string::npos || n.size() <= 4 || n.substr(n.size() - 4) != ".env" || t + 3 >= n.size() - 4) return false;
  return std::all_of(n.begin() + static_cast<std::ptrdiff_t>(t + 3), n.end() - 4, [](char c) { return c >= '0' && c <= '9'; });
}

std::string preset_env_path(const std::string &path) {
  const size_t sl = path.find_last_of('/');
  return sl == std::string::npos ? join_path("presets", path) : join_path(join_path(path.substr(0, sl), "presets"), path.substr(sl + 1));
}

std::string user_env_path(const std::string &path) {
  const size_t sl = path.find_last_of('/');
  if (sl == std::string::npos) return path;
  const std::string dir = path.substr(0, sl);
  const size_t ps = dir.find_last_of('/');
  if (dir.substr(ps == std::string::npos ? 0 : ps + 1) != "presets") return path;
  return join_path(ps == std::string::npos ? std::string() : dir.substr(0, ps), path.substr(sl + 1));
}

std::string readable_env_path(const std::string &user_path) {
  if (is_file(user_path)) return user_path;
  const std::string preset = preset_env_path(user_path);
  return is_file(preset) ? preset : user_path;
}

namespace {

// The model files (not the setup files) of one folder, sorted.
std::vector<std::string> env_files_in(const std::string &dir) {
  std::vector<std::string> out;
  if (DIR *d = opendir(dir.c_str())) {
    while (const dirent *e = readdir(d)) {
      const std::string n = e->d_name;
      if (is_setup_env(n)) continue;  // a setup's file, not a model's
      if (n.size() > 4 && n[0] != '.' && n.substr(n.size() - 4) == ".env" && is_file(join_path(dir, n))) out.push_back(join_path(dir, n));
    }
    closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

std::vector<std::string> model_env_files(const std::string &dir) {
  std::vector<std::string> out = env_files_in(dir);
  for (const auto &p : env_files_in(join_path(dir, "presets")))
    if (!is_file(user_env_path(p))) out.push_back(p);  // a user file of the same name replaces the preset
  return out;
}

namespace {

// A preset that needs files beyond the checkpoint (XEN_PRESET_EXPLICIT=1, e.g. a separately downloaded bank) is picked only
// by its name, as scripts/start-server.sh does: never as the model's settings file.
bool explicit_preset(const std::string &env_path) {
  std::string text;
  if (!read_text(env_path, text)) return false;
  const Values all = EnvFile::parse(text).values();
  const auto it = all.find("XEN_PRESET_EXPLICIT");
  return it != all.end() && it->second == "1";
}

}  // namespace

std::string find_model_env(const std::string &dir, const std::string &model_path) {
  if (model_path.empty()) return "";
  for (const auto &f : model_env_files(dir))
    if (same_path(model_env_model(f), model_path) && !explicit_preset(f)) return f;
  return "";
}

std::string model_env_id(const std::string &env_path) {
  std::string n = env_path.substr(env_path.find_last_of('/') + 1);
  if (n.size() > 4 && n.substr(n.size() - 4) == ".env") n.resize(n.size() - 4);
  return n;
}

std::string model_env_new_path(const std::string &dir, const std::string &model_path) {
  std::string m = model_path;
  while (m.size() > 1 && m.back() == '/') m.pop_back();
  std::string base = m.substr(m.find_last_of('/') + 1);
  if (base.empty()) base = "model";
  std::string p = join_path(dir, base + ".env");
  // a free name: neither a user file nor a preset of another checkpoint
  auto taken = [&](const std::string &q) {
    const std::string r = readable_env_path(q);
    return is_file(r) && !same_path(model_env_model(r), model_path);
  };
  for (int i = 2; taken(p); ++i) p = join_path(dir, base + "-" + std::to_string(i) + ".env");
  return p;
}

std::string resolve_model_env(const std::string &root, const std::string &dir, const std::string &ref) {
  if (ref.empty() || ref == "auto") return "";
  const std::string r = expand_home(ref);
  if (r.find('/') != std::string::npos || (r.size() > 4 && r.substr(r.size() - 4) == ".env")) return readable_env_path(user_env_path(join_path(root, r)));
  return readable_env_path(join_path(dir, r + ".env"));
}

ModelEnvView read_model_env(const Catalog &cat, const std::string &root, const std::string &dir, const std::string &env_path) {
  ModelEnvView v;
  std::string text;
  if (read_text(env_path, text)) {
    const EnvFile f = EnvFile::parse(text);
    const Values all = f.values();
    for (const auto &[k, x] : all)
      if (cat.find(k)) v.env_values[k] = x;
    if (f.has("XEN_MODEL_ENV")) v.pointer_set = true, v.pointer = all.count("XEN_MODEL_ENV") ? all.at("XEN_MODEL_ENV") : "";
  }
  const Values defs = defaults_of(cat);
  const std::string pointed = resolve_model_env(root, dir, v.pointer);
  const auto em = v.env_values.find("XEN_MODEL");
  if (em != v.env_values.end() && !em->second.empty()) v.model = em->second;
  else if (!pointed.empty() && is_file(pointed) && !model_env_model(pointed).empty()) v.model = model_env_model(pointed);
  else v.model = defs.count("XEN_MODEL") ? defs.at("XEN_MODEL") : "";
  if (v.pointer_set && v.pointer.empty()) return v;  // XEN_MODEL_ENV=: no per-model file
  if (!pointed.empty() && is_file(pointed) && same_path(model_env_model(pointed), v.model)) v.file = pointed;
  else v.file = find_model_env(dir, v.model);
  if (v.file.empty()) v.file = model_env_new_path(dir, v.model);
  v.file_exists = is_file(v.file);
  v.file_source = v.file_exists ? v.file : std::string();
  v.file = user_env_path(v.file);  // a preset is read, never written: saves go to the user's file of the same name
  if (v.file_exists && read_text(v.file_source, text))
    for (const auto &[k, x] : EnvFile::parse(text).values())
      if (cat.find(k)) v.file_values[k] = x;
  // the setup's file, picked as start-server.sh picks it (server.env, then the model's file, then 2)
  const std::string key = setup_key_of(probe_model(v.model).arch);
  if (!key.empty()) {
    const Values defs = defaults_of(cat);
    std::string n = v.env_values.count(key) ? v.env_values.at(key) : v.file_values.count(key) ? v.file_values.at(key) : "";
    if (n.empty()) n = defs.count(key) && !defs.at(key).empty() ? defs.at(key) : "2";
    v.setup_file = setup_env_path(v.file, n);
    if (read_text(readable_env_path(v.setup_file), text))
      for (const auto &[k, x] : EnvFile::parse(text).values())
        if (const Spec *sp = cat.find(k); sp && !is_shared_setting(*sp)) v.setup_values[k] = x;
  }
  return v;
}

namespace {

std::vector<std::string> list_dir(const std::string &dir) {
  std::vector<std::string> out;
  DIR *d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent *e = readdir(d)) {
    const std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    out.push_back(n);
  }
  closedir(d);
  std::sort(out.begin(), out.end(), [](const std::string &a, const std::string &b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                        [](char x, char y) { return std::tolower(static_cast<unsigned char>(x)) < std::tolower(static_cast<unsigned char>(y)); });
  });
  return out;
}

uint64_t tree_bytes(const std::string &dir, int depth) {
  uint64_t sum = 0;
  for (const auto &n : list_dir(dir)) {
    const std::string p = join_path(dir, n);
    struct stat st{};
    if (lstat(p.c_str(), &st) != 0) continue;
    if (S_ISREG(st.st_mode)) sum += static_cast<uint64_t>(st.st_size);
    else if (S_ISLNK(st.st_mode)) {
      if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) sum += static_cast<uint64_t>(st.st_size);
    } else if (S_ISDIR(st.st_mode) && depth > 0) {
      sum += tree_bytes(p, depth - 1);
    }
  }
  return sum;
}

}  // namespace

ModelInfo probe_model(const std::string &dir_in) {
  ModelInfo m;
  std::string dir = dir_in;
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  m.path = dir;
  const size_t sl = dir.find_last_of('/');
  m.name = sl == std::string::npos ? dir : dir.substr(sl + 1);
  if (!is_dir(dir)) {
    m.error = "directory not found";
    return m;
  }
  m.has_config = is_file(join_path(dir, "config.json"));
  m.drafter = is_file(join_path(dir, "mtp-dense.safetensors"));
  m.dflash = is_dflash_dir(dir);
  m.bytes = tree_bytes(dir, 2);
  if (!m.has_config) {
    m.error = "no config.json";
    return m;
  }
  try {
    const auto cfg = model::ModelConfig::load(dir);
    m.model_type = cfg.model_type();
    for (const auto *o : {&cfg.text(), &cfg.root()}) {
      if (const auto *mp = o->get("max_position_embeddings"); mp && mp->is_int() && m.max_positions == 0) m.max_positions = static_cast<int64_t>(mp->num);
      if (m.experts == 0) {
        if (const auto *e = o->get("num_experts"); e && e->is_int()) m.experts = static_cast<int64_t>(e->num);
        if (m.experts == 0)
          if (const auto *e = o->get("n_routed_experts"); e && e->is_int()) m.experts = static_cast<int64_t>(e->num);
      }
      if (m.layers == 0)
        if (const auto *e = o->get("num_hidden_layers"); e && e->is_int()) m.layers = static_cast<int64_t>(e->num);
      if (m.hidden == 0)
        if (const auto *e = o->get("hidden_size"); e && e->is_int()) m.hidden = static_cast<int64_t>(e->num);
      if (m.moe_inter == 0)
        if (const auto *e = o->get("moe_intermediate_size"); e && e->is_int()) m.moe_inter = static_cast<int64_t>(e->num);
    }
    // MoE layers: mlp_layer_types' "sparse" entries, else the layers past
    // first_k_dense_replace, else every layer (qwen38).
    if (m.experts > 0 && m.layers > 0) {
      m.moe_layers = m.layers;
      const auto &t = cfg.text();
      if (const auto *mt = t.get("mlp_layer_types"); mt && mt->is_array()) {
        int64_t n = 0;
        for (const auto &x : mt->arr) n += x.is_string() && x.str == "sparse" ? 1 : 0;
        if (n > 0) m.moe_layers = n;
      } else if (const auto *fk = t.get("first_k_dense_replace"); fk && fk->is_int() && fk->num > 0 && fk->num < m.layers) {
        m.moe_layers = m.layers - static_cast<int64_t>(fk->num);
      }
    }
    // bits: the expert weight precision - config.json's quantization_config.bits, else a separate
    // quantization_config.json.
    if (m.bits <= 0) {
      if (const auto *qc = cfg.root().get("quantization_config"); qc && qc->is_object())
        if (const auto *b = qc->get("bits"); b && b->is_number()) m.bits = b->num;
    }
    if (m.bits <= 0 && is_file(join_path(dir, "quantization_config.json"))) {
      std::string text;
      json::Value v;
      if (read_text(join_path(dir, "quantization_config.json"), text) && json::parse(text, v))
        if (const auto *b = v.get("bits"); b && b->is_number()) m.bits = b->num;
    }
    for (const auto &e : model::arch_registry())
      if (e.matches(cfg)) {
        m.arch = std::string(e.id);
        m.arch_desc = std::string(e.description);
        break;
      }
    if (m.arch.empty()) m.error = "architecture " + (m.model_type.empty() ? std::string("?") : m.model_type) + " is not served";
  } catch (const std::exception &e) {
    m.error = e.what();
  }
  return m;
}

std::vector<ModelInfo> scan_models(const std::string &dir) {
  std::vector<ModelInfo> out;
  for (const auto &n : list_dir(dir)) {
    const std::string p = join_path(dir, n);
    if (!is_dir(p) || !is_file(join_path(p, "config.json"))) continue;
    out.push_back(probe_model(p));
  }
  return out;
}

std::vector<std::string> drafters_for(const std::string &target_in) {
  std::string target = target_in;
  while (target.size() > 1 && target.back() == '/') target.pop_back();
  std::vector<std::string> out;
  const std::string rt = join_path(target, "runtime/mtp-int4-g32");
  if (is_dir(rt)) out.push_back(rt);
  const size_t sl = target.find_last_of('/');
  if (sl == std::string::npos) return out;
  const std::string parent = target.substr(0, sl), base = target.substr(sl + 1);
  for (const auto &n : list_dir(parent)) {
    if (n.rfind(base + "-mtp", 0) != 0) continue;
    const std::string p = join_path(parent, n);
    if (is_dir(p) && is_file(join_path(p, "mtp-dense.safetensors"))) out.push_back(p);
  }
  return out;
}

std::vector<ModelInfo> dflash_drafters(const std::string &dir) {
  std::vector<ModelInfo> out;
  for (const auto &m : scan_models(dir))
    if (m.dflash) out.push_back(m);
  return out;
}

bool probe_qsa_bank(const std::string &dir_in, QsaBank &out) {
  std::string dir = dir_in;
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  out = QsaBank{};
  out.path = dir;
  if (!is_dir(dir) || is_file(join_path(dir, "config.json"))) return false;  // a checkpoint, not a bank
  std::set<int> layers;
  static const std::string tag = ".self_attn._qsa_indexer.";
  for (const auto &n : list_dir(dir)) {
    if (n.size() < 12 || n.compare(n.size() - 12, 12, ".safetensors") != 0) continue;
    // safetensors: an 8-byte little-endian header length, then the JSON header naming every tensor
    std::ifstream f(join_path(dir, n), std::ios::binary);
    unsigned char len[8] = {};
    if (!f.read(reinterpret_cast<char *>(len), 8)) continue;
    uint64_t hl = 0;
    for (int i = 7; i >= 0; --i) hl = hl << 8 | len[i];
    if (hl == 0 || hl > (64u << 20)) continue;
    std::string header(hl, '\0');
    if (!f.read(header.data(), static_cast<std::streamsize>(hl))) continue;
    json::Value v;
    if (!json::parse(header, v) || !v.is_object()) continue;
    for (const auto &m : v.obj) {
      const size_t t = m.first.find(tag), l = m.first.rfind("layers.", t);
      if (t == std::string::npos || l == std::string::npos) continue;
      const std::string num = m.first.substr(l + 7, t - l - 7);
      if (num.empty() || num.size() > 4 || num.find_first_not_of("0123456789") != std::string::npos) continue;
      layers.insert(std::stoi(num));
    }
    out.bytes += static_cast<uint64_t>(std::filesystem::file_size(join_path(dir, n)));
  }
  out.layers.assign(layers.begin(), layers.end());
  return !out.layers.empty();
}

std::vector<QsaBank> qsa_banks(const std::vector<std::string> &dirs) {
  std::vector<QsaBank> out;
  std::set<std::string> seen;
  for (const auto &d : dirs) {
    if (d.empty() || !seen.insert(d).second) continue;
    for (const auto &n : list_dir(d)) {
      QsaBank b;
      if (n[0] != '.' && probe_qsa_bank(join_path(d, n), b)) out.push_back(std::move(b));
    }
  }
  std::sort(out.begin(), out.end(), [](const QsaBank &a, const QsaBank &b) { return a.path < b.path; });
  out.erase(std::unique(out.begin(), out.end(), [](const QsaBank &a, const QsaBank &b) { return a.path == b.path; }), out.end());
  return out;
}

std::string qsa_bank_detail(const QsaBank &b) {
  std::string l;
  const size_t n = b.layers.size();
  for (size_t i = 0; i < n; ++i) {
    if (n > 4 && i == 2) {  // the first two, then the last
      l += " ... " + std::to_string(b.layers.back());
      break;
    }
    l += (i ? ", " : "") + std::to_string(b.layers[i]);
  }
  return std::to_string(n) + (n == 1 ? " layer (" : " layers (") + l + "), " + human_bytes(b.bytes);
}

std::vector<std::string> complete_path(const std::string &typed, const std::string &root, std::string &common) {
  std::string full = expand_home(typed);
  const bool rel = full.empty() || full[0] != '/';
  const size_t sl = full.find_last_of('/');
  const std::string dir_part = sl == std::string::npos ? "" : full.substr(0, sl + 1);
  const std::string stem = sl == std::string::npos ? full : full.substr(sl + 1);
  const std::string dir = rel ? join_path(root, dir_part.empty() ? "." : dir_part) : (dir_part.empty() ? "/" : dir_part);
  std::vector<std::string> cands;
  for (const auto &n : list_dir(dir)) {
    if (n.rfind(stem, 0) != 0) continue;
    if (n[0] == '.' && (stem.empty() || stem[0] != '.')) continue;
    cands.push_back(n + (is_dir(join_path(dir, n)) ? "/" : ""));
  }
  common.clear();
  if (cands.empty()) return cands;
  std::string pre = cands[0];
  for (const auto &c : cands) {
    size_t i = 0;
    while (i < pre.size() && i < c.size() && pre[i] == c[i]) ++i;
    pre.resize(i);
  }
  // What the typed text becomes: keep the user's ~ prefix.
  std::string head = typed.substr(0, typed.size() - stem.size());
  common = head + pre;
  return cands;
}

}  // namespace xenithium:launch
