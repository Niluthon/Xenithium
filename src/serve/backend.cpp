// Xenithium · the helpers every serving backend shares (serve/backend.hpp).
#include "xenithium/serve/backend.hpp"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>

#include "xenithium/arch/common/drafter_archs.hpp"
#include "xenithium/arch/common/vision.hpp"
#include "xenithium/model/checkpoint.hpp"
#include "xenithium/model/config.hpp"
#include "xenithium/rt/card.hpp"
#include "xenithium/rt/fabric.hpp"
#include "xenithium/rt/l0.hpp"
#include "xenithium/util/video.hpp"

namespace xenithium::serve {

std::vector<std::vector<core::WarmRequest>> warm_plan(const text::Tokenizer &tok, int chunk, int64_t cap, bool prompt_lps, int slots) {
  const auto para = tok.encode(
      "The river runs past the old mill, where the miller kept a ledger of every sack of grain that came in from "
      "the farms. In the winter of the long frost the wheel froze solid, and the village ground its flour by hand "
      "for six weeks. ");
  auto text = [&](int n) {
    std::vector<int32_t> t;
    while (static_cast<int>(t.size()) < n) t.insert(t.end(), para.begin(), para.end());
    t.resize(static_cast<size_t>(std::max(n, 1)));
    return t;
  };
  auto req = [&](int len, int max_tokens) {
    core::WarmRequest r;
    max_tokens = static_cast<int>(std::min<int64_t>(max_tokens, std::max<int64_t>(cap / 2, 1)));
    r.prompt = text(static_cast<int>(std::min<int64_t>(len, cap - max_tokens - 1)));
    r.params.temperature = 0;
    r.params.max_tokens = max_tokens;
    r.params.ignore_eos = true;
    return r;
  };
  std::vector<std::vector<core::WarmRequest>> w(3);
  w[0].push_back(req(chunk, 16));
  w[1].push_back(req(chunk, 16));
  w[2].push_back(req(24, 28));
  auto s = req(std::min(300, chunk), 20);  // sampled, with logprobs
  s.params.temperature = 0.7f;
  s.params.top_p = 0.9f;
  s.params.has_seed = true;
  s.params.seed = 1;
  s.params.logprobs = 2;
  w[2].push_back(s);
  auto l = req(std::min(100, chunk), 16);  // greedy with logprobs and prompt logprobs
  l.params.logprobs = 1;
  if (prompt_lps) l.params.prompt_logprobs = 1;
  w[2].push_back(l);
  w[2].push_back(req(40, 12));
  for (int i = 4; i < slots; ++i) w[2].push_back(req(32 + 4 * i, 10 + i));
  return w;
}

double vram_free_gib(const l0::Loader &loader, const Fabric &fab) {
  double lo = 1e30;
  for (size_t c = 0; c < fab.size(); ++c) {
    const auto vs = l0::vram_state(loader.info(fab.rank_index(c)));
    if (vs.ok) lo = std::min(lo, static_cast<double>(vs.free));
  }
  return lo < 1e30 ? lo / 1073741824.0 : -1.0;
}

std::vector<size_t> cards_for(const l0::Loader &loader, int tp, int card, std::string_view arch, bool tp1_ok, bool tp4_ok) {
  char b[160];
  if (tp == 4) {  // qwen38 at TP4 (devices 0-3)
    if (!tp4_ok) {
      std::snprintf(b, sizeof b, "--tp 4: %.*s runs on one or two cards (TP4: qwen38 only)", static_cast<int>(arch.size()), arch.data());
      throw OptionError(b);
    }
    if (loader.size() < 4) {
      std::snprintf(b, sizeof b, "--tp 4: needs four devices, %zu visible", loader.size());
      throw OptionError(b);
    }
    return {0, 1, 2, 3};
  }
  if (!tp1_ok) {
    if (tp != 2) {
      std::snprintf(b, sizeof b, "--tp 1: %.*s runs on two cards only (TP2)", static_cast<int>(arch.size()), arch.data());  // (TP4: above)
      throw OptionError(b);
    }
    if (loader.size() < 2) {
      std::snprintf(b, sizeof b, "%.*s: needs two devices (TP2), %zu visible", static_cast<int>(arch.size()), arch.data(), loader.size());
      throw OptionError(b);
    }
    return {0, 1};
  }
  if (tp == 1 && static_cast<size_t>(card) >= loader.size()) {
    std::snprintf(b, sizeof b, "--card %d: only %zu device(s)", card, loader.size());
    throw OptionError(b);
  }
  if (tp == 2 && loader.size() < 2) {
    std::snprintf(b, sizeof b, "--tp 2: needs two devices, %zu visible (one card: --tp 1)", loader.size());
    throw OptionError(b);
  }
  return tp == 1 ? std::vector<size_t>{static_cast<size_t>(card)} : std::vector<size_t>{0, 1};
}

std::vector<server::Device> devices_of(const l0::Loader &loader, const Fabric &fab) {
  std::vector<server::Device> out;
  for (size_t c = 0; c < fab.size(); ++c) {
    const l0::DeviceInfo d = loader.info(fab.rank_index(c));
    const util::CardMatch card = util::identify_card(d.device_id, d.name, static_cast<int>(d.eu_total()) / rt::kXvePerCore, d.vram_bytes);
    out.push_back({.name = d.name, .memory_bytes = d.vram_bytes, .memory_gb = util::nominal_vram_gb(card.spec, d.vram_bytes), .pci = d.has_pci ? d.pci_bdf() : std::string(),
                   .driver = loader.driver_version_text(fab.rank_index(c))});
  }
  return out;
}

std::string plan_json(const MemPlan &p) {
  char b[1600];
  const double used = p.weights_gib + p.kv_gib + p.state_gib + p.scratch_gib + p.hot_gib;
  std::snprintf(b, sizeof b,
                "{\"arch\":\"%s\",\"exact\":%s,\"cards\":%d,\"card_gib\":%.3f,\"vram_used_gib\":%.3f,\"weights_gib\":%.3f,\"kv_gib\":%.3f,"
                "\"state_gib\":%.3f,\"scratch_gib\":%.3f,\"hot_gib\":%.3f,\"hot_experts\":%lld,\"experts_total\":%lld,"
                "\"moe_layers\":%d,\"kv_tokens\":%lld,\"context\":%lld,\"slots\":%d,\"cold_gib\":%.3f,\"embed_gib\":%.3f,\"pinned_gib\":%.3f,"
                "\"prefix_gib\":%.3f,\"host_other_gib\":%.3f,\"host_keep_gib\":%.3f,\"host_total_gib\":%.3f,\"host_available_gib\":%.3f}",
                p.arch.c_str(), p.exact ? "true" : "false", p.cards, p.card_gib, used, p.weights_gib, p.kv_gib, p.state_gib, p.scratch_gib, p.hot_gib,
                static_cast<long long>(p.hot_experts), static_cast<long long>(p.experts_total), p.moe_layers,
                static_cast<long long>(p.kv_tokens), static_cast<long long>(p.context), p.slots, p.cold_gib, p.embed_gib, p.cold_gib + p.embed_gib,
                p.prefix_gib, p.host_other_gib, p.host_keep_gib, p.host_total_gib, p.host_available_gib);
  return b;
}

void log_prefix_cache(int capacity, double bytes_card0, double bytes_card1, const Options &o, bool first_use, int more_cards) {
  const char *live = o.prefix_live ? "on" : "off";
  if (capacity > 0 && bytes_card0 == 0)
    std::fprintf(stderr, "server: prefix cache on: KV pages only (no recurrent state to snapshot), mark from %d tokens, live resumes %s\n", o.prefix_mark_min, live);
  else if (capacity > 0) {
    // the model cuts the capacity to the host RAM left after its load (util:snap_capacity): say so when it did
    const double one = bytes_card0 + bytes_card1 * (1 + more_cards), gib = capacity * one / 1073741824.0;
    char cut[96] = "";
    if (gib + one / 1073741824.0 < o.prefix_cache_gib) std::snprintf(cut, sizeof cut, " (--prefix-cache-gib %.2f cut to the host RAM left)", o.prefix_cache_gib);
    char per[96];
    if (more_cards > 0) std::snprintf(per, sizeof per, "%.1f per card on %d cards", bytes_card0 / 1048576.0, 2 + more_cards);
    else std::snprintf(per, sizeof per, "%.1f + %.1f per card", bytes_card0 / 1048576.0, bytes_card1 / 1048576.0);
    std::fprintf(stderr, "server: prefix cache on: up to %d snapshots of %.1f MiB (%s) in %.2f GiB of pinned host memory%s%s, mark from %d tokens, live resumes %s\n",
                 capacity, one / 1048576.0, per, gib, cut, first_use ? " (allocated on first use)" : "", o.prefix_mark_min, live);
  } else
    std::fprintf(stderr, "server: prefix cache off%s\n", o.prefix_cache ? " (--prefix-cache-gib holds no snapshot)" : "");
}

void log_ranking(const model::RouteRecord *rec, const std::string &file) {
  if (file.empty()) {
    std::fprintf(stderr, "server: hot-expert ranking off (--hot-rank empty): the default placement, nothing recorded\n");
    return;
  }
  if (!rec) return;  // the model said why (another model's file)
  std::fprintf(stderr, "server: hot-expert ranking %s: %s; recorded while serving, saved every 60 s of decoding and at exit\n", file.c_str(),
               rec->loaded() ? "the hot set placed from it" : "none yet (the default placement this run)");
}

void log_rope(int64_t context, int64_t trained, double yarn_factor, const char *why_none, bool tiers) {
  if (trained <= 0 || context <= trained) return;
  if (yarn_factor > 1.0)
    std::fprintf(stderr, "server: max context %lld tokens past the %lld trained positions: static YaRN factor %.2f (attention factor %.4f) %s\n",
                 static_cast<long long>(context), static_cast<long long>(trained), yarn_factor, 1.0 + 0.1 * std::log(yarn_factor),
                 tiers ? "for the requests of the last tier" : "on every request");
  else
    std::fprintf(stderr, "server: max context %lld tokens past the %lld trained positions: no YaRN%s%s%s\n", static_cast<long long>(context),
                 static_cast<long long>(trained), why_none ? " (" : "", why_none ? why_none : "", why_none ? ")" : "");
}

void log_rope_tiers(const std::vector<std::pair<int64_t, double>> &tiers) {
  if (tiers.size() < 2) return;
  std::string t;
  for (size_t i = 0; i < tiers.size(); ++i) {
    char b[96];
    std::snprintf(b, sizeof b, "%s%lld at %.2f", i ? ", " : "", static_cast<long long>(tiers[i].first), tiers[i].second);
    t += b;
  }
  std::fprintf(stderr, "server: YaRN per request (--yarn-tiers on): a request runs at the factor of the smallest tier holding its prompt and "
                       "room to grow (tokens up to: %s); one growing past its tier re-reads its tokens in the next\n", t.c_str());
}

namespace {

class VisionHook final : public server::MediaHook {
 public:
  VisionHook(arch::VisionStore &st, const Options &o) : st_(st), video_max_pixels_(o.video_max_pixels), video_fps_(o.video_fps) {
    if (o.vision_ffmpeg.empty()) why_ = "video input is off on this server (--vision-ffmpeg is empty)";
    else if (util::find_video_tools(o.vision_ffmpeg, tools_, why_)) video_ = true;
  }
  bool video_on(std::string &why) const {
    why = why_;
    return video_;
  }

  // The video path: frame sampling by fps, the checkpoint's video processor's sizing over all the frames, the uint8
  // resize per frame, one tower input per temporal patch (two frames; an odd count repeats the last). Frames are capped
  // at the server's image size (frames larger than XEN_VISION_MAX_PIXELS would not fit the tower's scratch): the budget
  // is min(XEN_VIDEO_MAX_PIXELS, frames x the image cap), the video processor's own sizing whenever a frame stays under
  // the image cap.
  bool add_video(const std::string &bytes, std::vector<Tokens> &frames, std::vector<std::string> &stamps, int &status, std::string &err) override {
    status = 400;
    if (!video_) return err = why_, false;
    const model::VisionConfig &vc = st_.config();
    util::VideoInfo vi;
    if (!util::probe_video(bytes, tools_, kVideoTimeoutS, vi, err)) return false;
    int H = vi.height, W = vi.width;
    if (vi.rotation % 180 == 90) std::swap(H, W);
    std::vector<int64_t> idx = model::video_frame_indices(vi.total_frames, vi.fps, video_fps_, vc.video_min_frames, vc.video_max_frames);
    std::vector<util::Image> rs;
    int rh = 0, rw = 0;
    auto size_for = [&](int n) {
      const int64_t tb = static_cast<int64_t>(std::nearbyint(static_cast<double>(n) / vc.temporal)) * vc.temporal;
      const int64_t budget = std::min<int64_t>(video_max_pixels_, std::min<int64_t>(tb, n) * st_.max_pixels);
      return model::video_smart_resize(n, H, W, vc.temporal, vc.patch * vc.merge, vc.video_min_pixels, budget, rh, rw, err);
    };
    for (int pass = 0; pass < 2; ++pass) {
      if (!size_for(static_cast<int>(idx.size()))) return false;
      rs.clear();
      size_t got = 0;
      bool size_bad = false;
      if (!util::decode_video_frames(bytes, tools_, idx, kVideoTimeoutS,
                                     [&](size_t, util::Image &&im) {
                                       if (im.w != W || im.h != H) return !(size_bad = true);
                                       rs.emplace_back();
                                       model::resize_bicubic_u8(im, rw, rh, rs.back());
                                       return true;
                                     },
                                     got, err))
        return false;
      if (size_bad) return err = "the video's frames change size", false;
      if (got == idx.size()) break;
      // the stream ended before the header's frame count: the frames read are kept, sized again for their count
      idx.resize(got);
      const int ph = rh, pw = rw;
      if (!size_for(static_cast<int>(got))) return false;
      if (rh == ph && rw == pw) break;
    }
    const int n = static_cast<int>(rs.size());
    if (n < vc.temporal) return err = "the video has " + std::to_string(n) + " readable frame(s); at least " + std::to_string(vc.temporal) + " are needed", false;
    const int per = (rh / vc.patch) * (rw / vc.patch) / (vc.merge * vc.merge);
    const int patches = (n + vc.temporal - 1) / vc.temporal;
    if (static_cast<int64_t>(per) * patches > st_.rows())
      return err = "the video needs " + std::to_string(static_cast<int64_t>(per) * patches) + " tokens, more than the server's image rows (" +
                   std::to_string(st_.rows()) + "): raise XEN_VISION_IMAGES or lower XEN_VIDEO_MAX_PIXELS / XEN_VIDEO_FPS", false;
    for (const double t : model::video_timestamps(idx, vi.fps, vc.temporal)) stamps.push_back(model::video_timestamp_text(t));
    for (int t = 0; t < patches; ++t) {
      model::VisionInput in;
      model::preprocess_frames(vc, rs[static_cast<size_t>(t * vc.temporal)], rs[static_cast<size_t>(std::min(t * vc.temporal + 1, n - 1))], in);
      auto lease = st_.add(std::move(in), err);
      if (!lease) {
        status = err.rfind("the image needs", 0) == 0 ? 400 : 503;
        return false;
      }
      Tokens tk;
      tk.base = lease->base, tk.count = lease->tokens;
      tk.lease = std::move(lease);
      frames.push_back(std::move(tk));
    }
    return true;
  }
  int32_t video_token() const override { return st_.config().video_token; }
  int32_t vision_start() const override { return st_.config().vision_start; }
  int32_t vision_end() const override { return st_.config().vision_end; }
  int32_t pad_token(int64_t id) const override { return st_.pad_token(id); }
  bool add_image(const util::Image &img, Tokens &out, int &status, std::string &err) override {
    model::VisionInput in;
    status = 400;
    if (!model::preprocess(st_.config(), img, st_.max_pixels, in, err)) return false;
    auto lease = st_.add(std::move(in), err);
    if (!lease) {
      status = err.rfind("the image needs", 0) == 0 ? 400 : 503;
      return false;
    }
    out.base = lease->base, out.count = lease->tokens;
    out.lease = std::move(lease);
    return true;
  }
  int32_t image_token() const override { return st_.image_token(); }
  int64_t max_pixels() const override { return st_.max_pixels; }

 private:
  static constexpr double kVideoTimeoutS = 300;  // probe or decode of one video
  arch::VisionStore &st_;
  int64_t video_max_pixels_ = 0;
  double video_fps_ = 2;
  util::VideoTools tools_;
  bool video_ = false;
  std::string why_;
};

}  // namespace

VisionPlan vision_plan(const Options &o, const model::ModelConfig &cfg, int cards) {
  VisionPlan v;
  if (!o.vision) return v;
  v.vc = model::VisionConfig::parse(cfg);
  if (!v.vc.present) throw OptionError("--vision 1: " + o.model_dir + " has no vision tower (config.json vision_config)");
  v.on = true;
  const int64_t pp = static_cast<int64_t>(v.vc.patch) * v.vc.patch, mm = static_cast<int64_t>(v.vc.merge) * v.vc.merge;
  v.max_patches = static_cast<int>(o.vision_max_pixels / pp / mm * mm);
  v.rows = static_cast<int64_t>(o.vision_images) * (v.max_patches / mm);
  for (int c = 0; c < cards; ++c)  // the tower on the first cards, the image rows on all
    v.gib.push_back(static_cast<double>(arch::VisionTower::bytes_on(v.vc, v.max_patches, cards, c) + arch::VisionStore::pool_bytes(v.vc, v.rows)) / 1073741824.0);
  return v;
}

std::unique_ptr<arch::VisionStore> load_vision(Fabric &fab, const Options &o, const VisionPlan &vp, int64_t vocab) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto ck = model::Checkpoint::open(o.model_dir);
  auto tower = std::make_unique<arch::VisionTower>(fab, ck, vp.vc, vp.max_patches);
  auto store = std::make_unique<arch::VisionStore>(fab, std::move(tower), vp.vc, vocab, vp.rows);
  store->max_pixels = o.vision_max_pixels;
  log_vision(*store, vp.gib, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return store;
}

std::unique_ptr<server::MediaHook> make_vision_hook(arch::VisionStore &store, const Options &o) {
  auto hook = std::make_unique<VisionHook>(store, o);
  std::string why;
  if (hook->video_on(why))
    std::fprintf(stderr, "server: video on: video_url parts decoded by %s (%.3g frames per second, up to %lld pixels per video, frames at most %lld pixels)\n",
                 o.vision_ffmpeg.c_str(), o.video_fps, o.video_max_pixels, static_cast<long long>(store.max_pixels));
  else
    std::fprintf(stderr, "server: %s\n", why.c_str());
  return hook;
}

void log_vision(const arch::VisionStore &store, const std::vector<double> &gib_per_card, double load_s) {
  const model::VisionConfig &vc = store.config();
  const int64_t px = store.max_pixels, max_tokens = px / (static_cast<int64_t>(vc.patch) * vc.patch * vc.merge * vc.merge);
  std::string gib;  // "1.489 / 1.467 / 0.112 / 0.112"
  for (const double g : gib_per_card) {
    char b[32];
    std::snprintf(b, sizeof b, "%s%.3f", gib.empty() ? "" : " / ", g);
    gib += b;
  }
  std::fprintf(stderr,
               "server: vision on: tower %d blocks (hidden %d, %d heads) loaded in %.1f s; images up to %lld pixels (~%lld tokens each), "
               "%lld image rows on each of %d cards (%s GiB per card); image_url / input_image parts take PNG / JPEG\n",
               vc.depth, vc.hidden, vc.heads, load_s, static_cast<long long>(px), static_cast<long long>(max_tokens), static_cast<long long>(store.rows()),
               store.cards(), gib.c_str());
}

std::string drafter_name(const std::string &block_dir, bool mtp) {
  if (block_dir.empty()) return mtp ? "mtp" : "none";
  std::string kind = "dflash2";
  try {
    for (const std::string &a : model::ModelConfig::load(block_dir).architectures())
      if (arch::is_dspine_arch(a)) kind = "dspine";
  } catch (const std::exception &) {  // the load read it already: kept as dflash2 if a second read fails
  }
  return mtp ? kind + "+mtp" : kind;
}

}  // namespace xenithium:serve
