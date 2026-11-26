// Xenithium · the serving backends: how an arch's runtime is opened behind the generic server.
// xenithium-server is written against this interface only. It parses serve:Options, opens the checkpoint's arch
// (model:open_arch), then asks the backend registry for the runtime that serves that arch. The backend loads its
// model, prints its load lines, owns every object it built (loader, fabric, model, the core:StepModel adapter), and
// runs what has to happen between the batch engine's creation and the first request (the warm-up, a KV pool sized to
// the VRAM left, graphs recorded ahead). Adding a runtime = an src/arch/<id>/serve.cpp with its `serve_entry` + one
// line in src/serve/registry.cpp (explicit, like model:arch_registry: a static library drops self-registering
// objects).
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/core/batch_engine.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/model/route_record.hpp"
#include "xenithium/model/vision.hpp"
#include "xenithium/serve/options.hpp"
#include "xenithium/server/server.hpp"
#include "xenithium/text/tokenizer.hpp"

namespace xenithium {
class Fabric;
namespace l0 {
class Loader;
}
namespace arch {
class VisionStore;
}
}  // namespace xenithium

namespace xenithium::serve {

// A bad or unsupported option, found while the backend validates or opens: the server prints the message and exits 2.
// Thrown only before the model is built (an exception out of a half-built model ends the process instead: see
// tools/server/main.cpp).
struct OptionError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual core::StepModel &step_model() = 0;
  virtual int64_t vocab() const = 0;  // logits row length (structured-output token masks)
  // The arch's prefill part length while others decode (--prefill-part-ms -1), ms.
  virtual double prefill_part_ms() const = 0;
  // Everything between the batch engine's creation and listening: the warm-up when `warmup, and what depends on it.
  virtual void start(core::BatchEngine &engine, const text::Tokenizer &tok, bool warmup) = 0;
  // Image input (--vision 1 on a checkpoint with a vision tower): the hook the server hands decoded images to;
  // nullptr = text only (image parts answer 400).
  virtual server::MediaHook *media() { return nullptr; }
};

struct BackendEntry {
  std::string_view name;  // the runtime ("qwen35" also serves the llama and flint archs)
  std::vector<std::string_view> archs;  // model:Arch ids it serves
  std::unique_ptr<Backend> (*open)(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id);
  // --plan: the load's memory plan as one JSON object (plan_json's fields), from the checkpoint headers and the options
  // alone - no device, no model load. nullptr = the runtime has no plan (the server says so).
  std::string (*plan)(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id) = nullptr;
};

const std::vector<BackendEntry> &backend_registry();
// The backend serving `arch_id, opened (the model loaded); throws OptionError when no runtime serves the arch.
std::unique_ptr<Backend> open_backend(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id);
// --plan: the runtime's memory plan for these options (BackendEntry:plan); throws OptionError when it has none.
std::string plan_backend(const Options &opt, const model::ModelConfig &cfg, std::string_view arch_id);

// The memory plan --plan prints (one JSON object, the launcher's estimate). Sizes in GiB; VRAM figures per card.
struct MemPlan {
  std::string arch;
  bool exact = true;  // false: an estimate (qwen38's auto hot count is measured from the free VRAM at load)
  int cards = 2;
  double card_gib = 0;  // the card's memory the budget fills
  double weights_gib = 0;  // non-expert weights (incl. a resident drafter's)
  double kv_gib = 0;  // the KV cache
  double state_gib = 0;  // recurrent state per slot (GDN)
  double scratch_gib = 0;  // runtime, prefill scratch, staging, spare slots
  double hot_gib = 0;  // the hot experts' shares
  int64_t hot_experts = 0;  // routed experts in VRAM over all layers
  int64_t experts_total = 0;  // routed experts over all MoE layers
  int moe_layers = 0;
  int64_t kv_tokens = 0;  // KV tokens held over all slots
  int64_t context = 0;  // one request's max tokens
  int slots = 0;
  double cold_gib = 0;  // cold experts, pinned host (both cards)
  double embed_gib = 0;  // embedding table, pinned host
  double prefix_gib = 0;  // the prefix cache's pinned snapshots (on first use; the load cuts them to the RAM left)
  double host_other_gib = 0;  // the process's host RAM besides the pinned set (util:kLoadHostOverheadGib)
  double host_keep_gib = 0;  // RAM the load leaves to the system (util:kHostKeepGib)
  double host_total_gib = 0, host_available_gib = 0;
};
std::string plan_json(const MemPlan &p);

// --vision 1: the tower's per-card bytes (weights + scratch at the largest image) plus the image rows every card holds,
// for a load on `cards` cards; on = false with --vision 0. Throws OptionError when the checkpoint has no vision tower.
struct VisionPlan {
  bool on = false;
  model::VisionConfig vc;
  int max_patches = 0;
  int64_t rows = 0;
  std::vector<double> gib;     // per card
  std::vector<size_t> bytes;   // per card (the same, in bytes: what ModelOptions::later_bytes keeps free)
  double max_gib() const { return gib.empty() ? 0.0 : *std::max_element(gib.begin(), gib.end()); }
};
VisionPlan vision_plan(const Options &o, const model::ModelConfig &cfg, int cards);
// The tower + store of a plan on the fabric (after the model loaded), the store's cap set, the load line logged.
std::unique_ptr<arch::VisionStore> load_vision(Fabric &fab, const Options &o, const VisionPlan &vp, int64_t vocab);

// The MediaHook over an arch's VisionStore: the image processor (model:preprocess at the store's max_pixels)
// on the server's prepare thread, then the store's registration (400 for an image the rows can never hold, 503 while
// running requests hold every row).
std::unique_ptr<server::MediaHook> make_vision_hook(arch::VisionStore &store, const Options &o);
// "server: vision..." the load line of an arch that loaded its tower.
void log_vision(const arch::VisionStore &store, const std::vector<double> &gib_per_card, double load_s);

// ---- helpers shared by the backends (src/serve/backend.cpp) ----------------------------------------------------------

// The warm-up: scratch requests run through the engine before the server listens, so the first user
// request does not pay the first prefill call or the step / drafter graph recording. Wave 1: a whole prefill chunk + 16
// greedy tokens (the 1-row step graphs at every parity, the drafter's MTP-3 passes); wave 2: the same again (the
// drafter's pass graphs alternate parities per request); wave 3: four requests at once with staggered lengths (2-4-row
// steps, MTP-1, a short prompt split in parts beside decoding, sampling, logprobs, prompt logprobs), plus one more
// short request per slot past 4 (the 5-8-segment step graphs). Prompts are cut to fit `cap` tokens.
std::vector<std::vector<core::WarmRequest>> warm_plan(const text::Tokenizer &tok, int chunk, int64_t cap, bool prompt_lps, int slots = 4);

// --kv-pool-tokens auto for the MoE archs whose pool is booked before the hot-expert fit (qwen38): min(slots, 4)
// contexts shared by the slots. `trained` > 0 (qwen38, YaRN past its trained positions): each of
// those contexts counts at most `trained` tokens, and the pool holds at least one whole context, so a 1,048,576-token
// max context on 4 slots books 1,048,576 tokens (one such request, or 4 of 262,144), not 4 x 1,048,576.
inline int64_t shared_pool_auto(int slots, int64_t context, int64_t trained = 0) {
  const int64_t ctx = (context + 63) / 64 * 64, base = trained > 0 ? std::min(ctx, (trained + 63) / 64 * 64) : ctx;
  return std::max(ctx, static_cast<int64_t>(std::min(slots, 4)) * base);
}

// One load line when the max context passes the checkpoint's trained positions: the YaRN factor the load applied
// (factor 1 = none; `why_none` says why the arch needs none; tiers: YaRN per request, the factor is the last tier's).
// Silent at or below the trained positions.
void log_rope(int64_t context, int64_t trained, double yarn_factor, const char *why_none = nullptr, bool tiers = false);
// With YaRN per request (--yarn-tiers on, more than one tier): the tiers, (positions, factor) ascending. Silent with one.
void log_rope_tiers(const std::vector<std::pair<int64_t, double>> &tiers);
// The MoE servers' recorded hot-expert ranking: where it came from and where it is saved (--hot-rank FILE;
// rec null = not recorded: no file given, or a file of another model).
void log_ranking(const model::RouteRecord *rec, const std::string &file);

// The tighter card's free VRAM (sysman) in GiB, -1 when unknown.
double vram_free_gib(const l0::Loader &loader, const Fabric &fab);

// The cards of the deployment checked against the visible devices; throws OptionError. tp1_ok: the arch runs TP1 too
// (--tp / --card); else it is TP2 on cards 0 and 1. tp4_ok: --tp 4 = cards 0-3 (qwen38).
std::vector<size_t> cards_for(const l0::Loader &loader, int tp, int card, std::string_view arch, bool tp1_ok, bool tp4_ok = false);

// "server: prefix cache on..." / "off": the one load line every prefix-cache arch prints (bytes per snapshot and card;
// bytes_card0 == 0 with a capacity = KV pages only, no recurrent state). first_use: say the store is allocated on first use.
// more_cards (qwen38 TP4): cards past the second, each holding bytes_card1 as card 1
void log_prefix_cache(int capacity, double bytes_card0, double bytes_card1, const Options &o, bool first_use = false, int more_cards = 0);

}  // namespace xenithium:serve
