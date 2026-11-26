// Xenithium · every setting of xenithium-server, parsed once (tools/server/options.cpp: the flag table, defaults and
// validation; docs/server-env.md: the meaning of each) and handed to the serving backend of the
// checkpoint's arch (serve/backend.hpp). A backend reads the fields that apply to it; arch-specific resolution of
// shared settings (e.g. flint's prefill numerics for `auto`) happens in its backend, never in the server.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace xenithium::serve {

struct Options {
  // ---- what and where ----
  std::string model_dir;
  std::string chat_template = "auto";  // auto | builtin | jinja | FILE
  std::string host = "0.0.0.0";
  int port = 8000;
  std::string served_name = "xenithium";
  bool stub = false;  // GPU-less scripted engine behind the real server
  bool plan = false;  // --plan: print the load's memory plan as JSON (no GPU, no model load) and exit
  double stub_step_ms = 2;
  int stub_tokens_per_step = 1;

  // ---- sequences, KV, cards ----
  int slots = 4;
  int64_t context = 32768;  // the max context: one request's most tokens (max_model_len); YaRN past the trained positions
  int64_t kv_pool_tokens = -1;  // -1 = auto, N = a pool of N tokens shared by the slots (the server refuses 0)
  std::string kv = "int8";  // int8 | bf16
  std::string kv_split = "heads";  // qwen38 TP4: heads | tokens (the two cards of a kv head split its tokens)
  double reserve_gib = 0.25;
  int tp = 2, card = 0;  // cards (TP1 / TP2); TP1's card
  std::string emu_card = "off";  // off | b60 | b65: run as that card (XEN_EMU_CARD, util/cards.hpp, rt/card.hpp)
  std::string wire = "bf16";
  std::string yarn_tiers = "on";  // qwen35 / qwen38 / llama: YaRN per request past the trained positions, or one factor
  std::string prefill_wire = "auto";  // the prefill two-card sums: bf16 partials or fp32; auto = qwen35 bf16, qwen38 fp32
  int threads = 16;
  bool graphs = true;
  bool prefill_graphs = false;  // qwen38: prompt calls of fewer than 128 rows replay recorded graphs (--prefill-graphs)

  // ---- prefill ----
  int prefill_chunk = 0;  // 0 = the arch's
  double prefill_stall_ms = 100;
  double prefill_part_ms = -1;  // -1 = the arch's
  int prefill_merge = 16;  // qwen35: the waiting prompts one prefill call may carry; 1 = one each
  std::string prefill_attn = "auto";  // auto | exact | fast | int8 | int8-fast (auto kept: backends resolve it)
  std::string prefill_w4 = "a8";  // exact | dequant | a8
  int attn_tail = 2;  // qwen35 / llama prefill attention context splits: 1 = the last wave's work-groups, 2 = + calls under one wave, 0 = none
  int prefill_gemm_rows = 1024;  // qwen35: rows per call from which the wide W4A8 GEMMs take oneDNN (128..4096; below 1024 a numerics change that pays with KV lending)
  int prefill_short = 32;  // qwen35: prefill calls of <= N rows run their int4 GEMMs on the decode kernels (0 off, 1..32)
  std::string prefill_gemm = "onednn";  // onednn | onednn-all | esimd: qwen35 W4A8 prefill GEMMs n >= 6144 (onednn-all: every one) on oneDNN; qwen38 int8 trunk with chunks >= 2048
  std::string prefill_a8_keep;
  std::string prefill_a8_body = "auto";  // auto | slm | direct: qwen35 W4A8 prefill GEMM body
  std::string prefill_moe = "w4a16";  // qwen38
  std::string prefill_dense = "w8a8-row";  // qwen38
  int prefill_cold_stage = 16;  // qwen38
  int prefill_fetch = 0;        // qwen38 (two cards)
  std::string kv_lend = "on";  // qwen38: the KV pool's free pages hold cold experts (on|off)
  int64_t kv_lend_keep = 32768;  // qwen38: KV tokens kept free below the lent pages
  double kv_lend_park_s = 300;  // qwen38: a parked prefix above the keep is dropped after this long (0 = never)
  std::string kv_lend_pressure = "kv";  // qwen38: kv | lru | idle (core:BatchConfig:lend_pressure)
  double kv_lend_grace_s = 60;  // qwen38: with idle, parked at least this long
  std::string expert_exchange = "on";  // qwen38: the hot experts follow the decode traffic while serving (on|off)

  // ---- weights ----
  std::string lm_head = "auto";  // auto | int8 | int4 | int4-rescore
  int lm_head_tiles = 32;
  std::string draft_head = "auto";
  std::string decode_moe = "w4a16";  // qwen38: auto | w4a8 | w4a16
  std::string gdn_w4 = "int8";  // qwen38
  int w4_items = 1024;  // qwen35
  int attn_select = 0;  // qwen35
  int prefill_select = 0;  // qwen35: page-selected prefill attention, pages; 0 = dense
  std::string select_heads = "auto";  // qwen35: auto = each sequence measures its heads' page counts (--attn-select) | same
  std::string decode_attn = "auto";  // qwen35: auto | exact | fast | int8 | int8-fast; auto = int8-fast on the int8 KV with dense attention, else exact
  std::string decode_w4 = "exact";  // qwen35: exact | a8: the decode int4 GEMMs on int8 activations; opt-in

  // ---- MoE placement ----
  long long hot = 0;  // routed experts held in VRAM over ALL layers (each split over the cards); 0 = as many as fit
  std::string hot_rank;
  // qwen38: the decode MoE's CPU path for cold experts; off by default
  std::string moe_cpu = "off";  // off | auto | avx512 | avx2
  int moe_cpu_threads = 8;  // per card
  std::string moe_cpu_cores;  // first core per card, "A,B" (empty = 0 and threads; -1 = unpinned)
  int moe_cpu_pct = 70;
  int moe_cpu_min_rows = 1;
  int moe_cpu_min_cold = 2;
  int moe_cpu_timeout_ms = 500;

  // ---- drafters ----
  std::string mtp;  // qwen38: the MTP drafter checkpoint
  int mtp_depth = 3;  // qwen38
  int drafter = 1;  // qwen35: the checkpoint's own MTP layer
  int draft_vocab = 98304;
  std::string drafter_model;  // qwen35: a DFlash 2 checkpoint
  int drafter_defer = 64;
  double drafter_cut = 0.02;
  int drafter_tree = 7;  // DDTree: qwen35's DFlash 2 tree nodes per pass, one greedy stream (0 = the chain)
  int drafter_tree_seqs = 2;  // trees in steps of up to this many sequences (1 = one stream only)
  // qwen35 step rows:
  // step_rows: a step's rows in all, 8, 16 or 32 (llama / flint run 8 at any setting; 8 needs slots <= 8);
  // seg_rows: one sequence's verify, 8 or 16 (16 = the default: a drafter trained on a 16-token block, DSpine b16;
  // the GDN's pending logs hold 16 rows per slot: more VRAM per slot; 8 = a drafter trained on an 8-token block,
  // DFlash 2; llama / flint run 8 at any setting);
  // draft_depths: drafts per sequence in a step of 1, 2... sequences ("15,7,4,3": comma list, at most 16; missing
  // entries and "" keep the default table 7,7,4,3,1,1,1,1,0..); each <= seg_rows - 1, n x (depth + 1) <= step_rows;
  // drafter_wide: with seg_rows 16, one stream drafts 15 (a 16-row tree of drafter_tree_wide nodes, or a chain without
  // trees) after a verify that accepted this many drafts or more, else the table's pick (0 = the table only).
  // step_rows 0 = the model's default (qwen35 32; qwen38 16, on four cards 64 with the budget by stream count);
  // step_rows_or(d) reads it
  int step_rows = 0;  // llama / flint run 8 at any setting above 8
  int step_rows_or(int def) const { return step_rows > 0 ? step_rows : def; }
  int seg_rows = 16;  // llama / flint run 8 at any setting
  std::string draft_depths;
  int drafter_wide = 7;
  int drafter_tree_wide = 15;
  // With a block drafter: the block drafter drafts steps of fewer sequences than this, the checkpoint's MTP layer
  // steps from this many on (the default 9: the block drafter for 1-8 sequences)
  int drafter_mtp_from = 9;
  // The block drafter's q, o, gate | up and down matrices as int4 (round-to-nearest per 128-k group
  // at load) on the W4A16 decode GEMMs, 0 = int8 per row. Drafts only (the output is the same); arch flint keeps int8.
  int drafter_int4 = 1;
  int adaptive_depth = 1;
  int unequal_segments = 1;  // a draft depth per request (qwen38 with DFlash)
  int lookup = 4;  // lookup drafting: the match length (0 = off)

  // ---- adapters ----
  std::string lora;  // a PEFT LoRA adapter directory (adapter_model.safetensors + adapter_config.json); qwen35 / llama only

  // ---- prefix cache ----
  int prefix_cache = 1;
  double prefix_cache_gib = 8;
  int prefix_mark_min = 1024;
  int prefix_live = 0;

  // ---- image input ----
  int vision = 0;  // load the checkpoint's vision tower (qwen38); 0 = text only
  long long vision_max_pixels = 1048576;  // an image is scaled down to at most this many pixels (~1,024 tokens at 1M)
  int vision_images = 16;  // images of the largest size the image rows hold at once (per card)
  // media URLs (image_url / input_image / video_url): http(s) fetched by the server; data: always
  int media_urls = 1;  // 0 = data: URLs only
  std::string media_domains;  // comma-separated host names a URL may name (empty = any)
  double media_timeout_s = 30;  // one fetch (connect, redirects, body)
  // video input (video_url parts) through an ffmpeg subprocess
  std::string vision_ffmpeg = "ffmpeg";  // ffmpeg executable (path or name on PATH; ffprobe beside it); empty = video off
  long long video_max_pixels = 25165824;  // pixels over all sampled frames of one video (the checkpoints' video processor)
  double video_fps = 2;  // frames sampled per second of video (at most 30)

  // ---- sampling defaults: a request's value where it sends none; empty = the checkpoint's
  // generation_config.json (the engine's defaults for what that file leaves out: temperature 1, top_p 1, top_k off,
  // min_p 0, penalties off). A request's own value always wins.
  std::string temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty;

  // ---- serving loop ----
  bool warmup = true;
  int sample_threads = 4;
  int max_queue = 0;
  std::string queue_policy = "fifo";
  double queue_aging_s = 10, queue_timeout_s = 0;
  int idle_timeout_s = 300;
  std::string cors_origin = "*";
  double log_interval_s = 10;
};

// ServeOptions:draft_depths parsed ("" = none; false on a bad list: not 0..15 integers, more than 16)
inline bool parse_draft_depths(const std::string &s, std::vector<int> &out) {
  out.clear();
  if (s.empty()) return true;
  size_t p = 0;
  while (p <= s.size()) {
    const size_t e = std::min(s.find(',', p), s.size());
    const std::string t = s.substr(p, e - p);
    if (t.empty() || t.size() > 2 || t.find_first_not_of("0123456789") != std::string::npos) return false;
    const int d = std::stoi(t);
    if (d > 15 || out.size() >= 16) return false;
    out.push_back(d);
    p = e + 1;
  }
  return true;
}

}  // namespace xenithium:serve
