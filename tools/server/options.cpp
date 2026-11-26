// xenithium-server's options: the flag table, the shared validation and the help text. Model-specific resolution (an
// arch's `auto` values, the options it ignores) is in its serving backend (src/arch/<id>/serve.cpp).
//
//   xenithium-server --model DIR [--slots N] [--context N] [--port P] [--host H] ...
//   xenithium-server --stub --model DIR ...     GPU-less: a scripted engine behind the real server
//
// The model directory supplies the weights, tokenizer.json, the chat template and generation_config.json (sampling
// defaults and EOS ids). Options, defaults in brackets:
//
// --host H [0.0.0.0] --port P [8000] --served-model-name S [xenithium]
// --slots N [4] sequences decoded at once, one state slot each (qwen38 / qwen35 1..16, llama / flint 1..8)
// --context N [32768] the max context: one request's most tokens (prompt + output; max_model_len), not a per-slot
//   reservation (the slots share the KV pool). Past the checkpoint's trained positions (262,144 on qwen35 / qwen38,
//   524,288 on Nanbeige) the load applies static YaRN at the smallest factor (steps of 0.01) that reaches it, on every
//   request; flint needs none (windowed rotary)
// --kv-pool-tokens auto|N [auto] one KV pool shared by the slots, so a long conversation uses the room the others
//   leave free. qwen35 / llama / flint: auto = the VRAM left after the warm-up minus --reserve, capped at
//   slots x context. qwen38: N = a pool of N tokens for the slots and the drafter, allocated before the hot-expert
//   placement (VRAM not held for idle contexts goes to hot experts); auto = min(slots, 4) contexts of at most the
//   trained positions each, at least one --context. A full pool preempts the youngest request.
// --kv int8|bf16|int8-v4 [int8] the KV cache's precision. int8-v4 (qwen38 and the dense 27B): K int8, V int4 with a
//   bf16 scale per 32 dims; on qwen38 it frees VRAM per card for hot experts at a 524,288-token pool. A numerics
//   change; llama / flint refuse it.
// --kv-split heads|tokens [heads] qwen38 on four cards: how the two cards of a kv head store its KV. heads = both hold
//   all of its tokens; tokens = each holds every other token: a quarter of the KV per card (the pool's tokens per GiB
//   about 1.75x at int8-v4), the partner's q heads computed on both, one exchange per attention layer; a numerics change.
//   Other models and card counts refuse tokens.
// --hot N [0] (MoE: the routed experts kept in VRAM, counted over ALL layers; each hot expert is split over the cards,
//   so N cost N half-experts on each; 0 = as many as fit the VRAM left)
// --plan print the load's memory plan as JSON (VRAM per card, the KV tokens the pool will hold, hot experts, pinned
//   host memory) and exit: no GPU, no model load. Exit 2 with the reason when the load would refuse the settings.
// --hot-rank F qwen38: the hot-expert ranking (XRC1), read at boot when it exists and written every 60 s of decoding
//   and at exit; a file of another model is neither read nor written. Empty = the default placement, nothing recorded.
// --moe-cpu off|auto|avx512|avx2 [off] qwen38: a host thread pool per card computes part of each decode layer's cold
//   experts (pinned RAM, AVX-512 BF16 or AVX2 + FMA) while the cards compute the rest; auto = the best ISA the CPU
//   has, a CPU without either leaves it off. A numerics change (the pool's summation order) and host-CPU dependent.
// --moe-cpu-threads N [8] per card, --moe-cpu-cores A,B [0,N] first core per card (-1 = unpinned),
//   --moe-cpu-pct P [70] of each layer's cold experts form the pool's fixed set, --moe-cpu-min-cold N [2] (per row and
//   layer: a row routing to fewer of the set's experts keeps them on the cards; a row's own count, so a sequence's
//   tokens do not depend on its batch-mates), --moe-cpu-min-rows N [1] (> 1 makes them depend on the batch again),
//   --moe-cpu-timeout-ms MS [500] (a card's wait per layer; on a timeout it computes them itself and the path goes off)
// --reserve GIB [0.25] --graphs 0|1 [1] --prefill-chunk N [0 = the arch's: qwen38 4,096, qwen35 2,048] --threads N [16]
// --mtp DIR [none] the MTP drafter checkpoint (runtime/mtp-int4-g32): speculative steps, MTP-3 for one decoding
//   request and MTP-1 each for two, plain rows for three or four; --mtp-depth N [3] its depth for one sequence, 1..3
// --prefill-stall-ms MS [100] while requests decode, size each prefill call to take about this long (0 = full
//   --prefill-chunk calls) --sample-threads N [4]
// --temperature X --top-p X --top-k N --min-p X --presence-penalty X --repetition-penalty X [the checkpoint's
//   generation_config.json]: the sampling of requests that send no value of their own
// --prefill-moe w4a8|w4a16|w4a16-fast [w4a16] the prefill experts' activations: int8 or bf16 (w4a16-fast: bf16 with the
//   int4 widen's offset outside the DPAS operand, not the same bits)
// --prefill-dense w8a8|w8a8-row|w8a16 [w8a8-row] the prefill dense trunk's activations: int8 on s8 x s8 DPAS or bf16
//   (w8a8-row: int8 per row, the HC mixer on bf16)
// --decode-moe auto|w4a8|w4a16 [w4a16] the decode experts' activations: int8 or bf16; auto = as the checkpoint's
//   experts were tuned (quantization_config.act_bits 8: w4a8)
// --gdn-w4 int8|int4|int4-qkvz [int8] the GDN projections' format: int8 rows (converted from the BF16 checkpoint) or
//   the directory's int4 GPTQ tensors (`xenithium-bake gdn4-calib`; int4-qkvz: in_proj_qkv | z only, out_proj int8):
//   a numerics change
// --prefill-gemm-rows N [1024] qwen35 with --prefill-gemm onednn: the rows per call from which the wide W4A8 GEMMs
//   take oneDNN (128..4096). Below 1024: shorter prompts take oneDNN's rounding (a numerics change), and it pays only
//   while --kv-lend keeps the layouts
// --prefill-cold-stage N [16] VRAM blobs per card for copying a prefill call's multi-segment cold experts once
//   (0 = read every segment over PCIe)
// --prefill-fetch N [0] qwen38 on two cards: VRAM for N cold expert blobs per layer, three times per card; a prompt
//   call of 4,096 rows and more copies each layer's cold experts (up to N) into it on the copy engine while the layers
//   before it compute, instead of its MoE reading them over PCIe (0 = off); identical output
// --prefill-graphs 0|1 [0] qwen38: a prompt call of fewer than 128 rows replays its layers from a recorded graph per
//   card once its shape (rows, position bucket, wire parities; any slot) has run twice; identical output, 1-2 ms less
//   per such call on two or four cards, 20-60 ms to record a shape
// --kv-lend on|off [on] qwen38: the KV pool's pages no sequence holds keep cold experts in VRAM (given back, no copy,
//   when the KV grows into them; refilled while the server is idle); identical output. On the dense 27B the pages keep
//   the wide prefill matrices' oneDNN layouts instead (no repack per call, given back the same way); there
//   --kv-lend-keep applies, the three below do not.
// --kv-lend-keep N [32768] KV tokens kept free below the lent pages (one eviction per N tokens of growth)
// --kv-lend-park-s S [300] a parked prefix (prefix cache) holding pages above the keep is dropped after S seconds
//   parked, so its pages are lent to experts again while idle (0 = never)
// --kv-lend-pressure kv|lru|idle [kv] when a request's KV would give lent experts back: kv = it does (parked prefixes
//   stay); lru = parked prefixes above the keep are dropped first, least recently used first; idle = only those parked
//   at least --kv-lend-grace-s
// --kv-lend-grace-s S [60] see --kv-lend-pressure idle
// --expert-exchange on|off [on] qwen38 with cold experts: the hot set follows the decode traffic while serving
//   (exchanges between VRAM and pinned host memory on the copy engines; the same output)
// --prefill-part-ms MS [-1] while requests decode, a prompt that fits one prefill call runs it in parts of about this
//   long (whole layers), a decode step between them (0 = whole calls; -1 = the arch's: qwen38 12, qwen35 25 with every
//   call in parts)
// --vision 0|1 [0] image input: the checkpoint's vision tower loaded, image_url parts (data: URLs, PNG / JPEG) accepted
//   by /v1/chat/completions --vision-max-pixels N [1048576] an image is scaled down to at most N pixels (1M ~ 1,024
//   tokens) --vision-images N [16] largest images the image rows hold at once
// --media-urls 0|1 [1] media parts may name http:// / https:// URLs, fetched by the server (https through the
//   system's libssl.so.3 when present); 0 = data: URLs only --media-domains LIST [any] comma-separated host names a URL
//   may name (every redirect hop too) --media-timeout S [30] one fetch
// --vision-ffmpeg PATH [ffmpeg] video_url parts decoded by this ffmpeg (ffprobe beside it; the default name is also
//   looked up in util::ffmpeg_install_paths); empty or missing = video
//   parts get 400 --video-max-pixels N [25165824] pixels over all sampled frames of a video --video-fps F [2] frames
//   sampled per second
// --warmup 0|1 [1] scratch requests before listening, so the first request pays no first-use costs
// --prefill-merge N [16] (qwen35) the waiting prompts one prefill call may carry, 1..16: prompts whose whole remaining
//   prompt is one many-row call of 33..511 rows share a call of up to 1,023 rows (each padded to 16), so norms, GEMMs,
//   MLP and the two-card sums run once and the GDN core and the attention per prompt. A prompt's rows are its own
//   call's bit for bit. 1 = a call per prompt.
// --max-queue N [0 = unlimited] --queue-policy fifo|sjf [fifo] --queue-aging S [10]
// --queue-timeout S [0] --idle-timeout S [300] --cors-origin O [*] --log-interval S [10]
// --stub-step-ms MS [2] --stub-tokens-per-step N [1] (stub only)
//
// The dense Qwen3.8-27B (arch qwen35) serves --slots sequences with its checkpoint's own MTP layer:
// --drafter 0|1 [1] (MTP-3 speculative steps), --draft-vocab K [98304] (the drafter's lm_head rows: token ids [0, K) +
// the special tokens; 0 = the whole vocabulary), --drafter-model DIR [none] (a DFlash 2 block drafter checkpoint, e.g.
// ~/models/qwen38-27b-dflash2, instead of the MTP layer: up to 7 drafts per step, an 8-row verify; --draft-vocab
// applies to it too; --drafter-defer N [64] its context rows of steps without drafts deferred per slot, 0 or 8..512;
// --drafter-cut P [0.02] one sequence's drafts kept while their confidence product >= P, 0 = all; --drafter-mtp-from
// N [9] with a DFlash drafter the checkpoint's MTP layer is loaded too and drafts for steps of N or more decoding
// sequences (0..16; 9 = the block drafter through 8 streams, MTP from 9), 0 = DFlash alone; --drafter-defer applies to
// MTP's KV rows too); --drafter-int4 0|1 [1] the block drafter's q, o, gate | up and down matrices as int4 (RTN at
// load, the same acceptance), 0 = int8 per row.
//
// qwen38: --drafter-model DIR (~/models/qwen38-flash-next-dflash2) serves the DFlash 2 drafter instead of --mtp (which
// is then not loaded; --drafter-mtp-from does not apply): 7 drafts per step and an 8-row verify per sequence,
// --drafter-defer / --drafter-cut / --draft-vocab as above; --mtp-depth applies to MTP only.
//
// --tp 1|2 [2]: qwen35 on two cards (TP2: every matrix split by head / column, the sums on the two-card wire) or on one
// (TP1: every matrix whole, no wire; the other card stays free, e.g. for a second server); --card N [0]: TP1's card
// (its Level Zero device index). qwen38: --tp 1|2|4.
//
// --emu-card off|b60|b65 [off]: run as an Arc Pro B60 / B65 on a bigger Battlemage card (the B70): the kernels' launch
// shapes for its Xe cores, its VRAM for every budget; the server exports it as XEN_EMU_CARD (rt/card.hpp).
//
// --attn-select N [0] (opt-in): page-selected decode attention: every KV page's key bounds kept, and a decode step
// whose context passes N pages (N x 64 tokens) attends per sequence and kv head to N pages (the first, the last 16,
// the best page scores of the rest); 0 = dense attention; 32..4096.
// --select-heads auto|same [auto] (qwen35 with --attn-select N past 128) the pages each kv head keeps: auto = every
// sequence measures it (on its first 8 selected steps every head keeps N and the engine reads the page scores it
// computes anyway; heads whose attention is spread then keep more than N, sharp ones fewer, never under 128, the mean
// N; nothing to calibrate), same = N for every head. Not read with --attn-select 0.
// --prefill-select N [0] (qwen35, opt-in) page-selected prefill attention: a prefill call whose context passes N pages
// attends per attention row block and kv head to N pages chosen the same way, so the prompt's KV past that point is
// built on selected attention (another KV than a dense prefill's); 0 = dense prefill; 32..4096. Keeps the key bounds
// too. Runs with --prefill-attn auto, exact, int8-fast (int8 KV) or fast (bf16 KV).
// --decode-attn auto|exact|fast|int8|int8-fast [auto] (qwen35) the decode attention's numerics: fast = P.V on bf16 P
// only (on the int8 KV its q.k on int8 hi + lo q codes), int8 = q.k on int8 q per (row, head) x the int8 K codes,
// int8-fast = both (int8* need --kv int8; numerics changes; dense attention only: not with --attn-select). auto =
// int8-fast on the dense 27B with the int8 KV and dense attention, else exact.
// --decode-w4 exact|a8 [exact] (qwen35) a8 = the decode step's int4 GEMMs of the model (MLP, GDN, attention
// projections) on int8 activations per (row, 128-k group), the prefill W4A8's quantisation (a numerics change, opt-in;
// the drafters and lm_head stay W4A16; the dense 27B only).
// --wire bf16|fp32|int8 [bf16] the decode two-card sums' wire: bf16 partials, fp32 partials, or int8 (the dense 27B,
// TP2: 8-bit codes per 32 columns, 56 % of bf16's bytes, a numerics change).
// --w4-items N [1024] the decode W4 GEMMs' k split target in work-items (512 = another fp32 order of the group sums).
// --prefill-attn auto|exact|fast|int8|int8-fast|onednn [auto = int8-fast on int8 KV, fast on bf16 KV] the prefill
// attention's numerics: exact = the decode path's; fast = the P.V sums as one bf16 number per term (half the P.V work,
// a numerics change; on the int8 KV its q.k runs on int8 hi + lo q codes); int8 / int8-fast: q.k on int8 q per (row,
// head) x the int8 K codes, P exact / P hi only; onednn = oneDNN's fused GQA attention in f16 on the gathered context.
// --prefill-w4 exact|dequant|a8 [a8] the prefill W4 GEMMs' numerics: exact = per-group fp32 sums; dequant =
// bf16(w x scale) weights with whole-k DPAS sums (a numerics change); a8 = int8 activations per (row, 128-k group),
// s8 x s4 DPAS (a numerics change; decode stays W4A16). --prefill-a8-keep SPEC: the W4 GEMMs that stay W4A16 under a8
// (ModelOptions:prefill_a8_keep, e.g. gdn+attn). --prefill-a8-body auto|slm|direct [auto]: the a8 GEMMs' ESIMD body
// (same outputs bit for bit): slm = the pipelined SLM body, direct = the SLM-free body, auto = slm at TP2, direct at TP1.
// --lm-head auto|int8|int4|int4-rescore [auto] the lm_head's format: auto = int4-rescore when the directory holds
// lm_head.weight and lm_head.qweight, int4 for an int4-only head, int8 otherwise; int8 = int8 rows (converted at load);
// int4 = the checkpoint's int4 g128 lm_head.qweight (`xenithium-bake head4-calib`; a numerics change);
// int4-rescore = the int4 rows score, each row's top --lm-head-tiles T [32] 16-column tiles per card are rescored by
// the int8 rows (both loaded, +636 MB per card; picks == int8 on every captured row).
// --chat-template auto|builtin|jinja|FILE [auto] the chat prompt renderer: auto = the checkpoint's own chat template
// (chat_template.jinja / tokenizer_config.json) through the engine's Jinja interpreter, the arch's built-in renderer
// when it ships none or it does not parse; builtin = the built-in renderer; jinja = the checkpoint's template or fail;
// FILE = that template file. The output parser is the arch's either way.
// --draft-head auto|int8|int4 [auto] the drafters' draft-vocabulary rows (auto = as the lm_head: int4 rows for int4 /
// int4-rescore).
//
// --reserve also sizes qwen35's auto KV pool. MoE / qwen38-only options (--hot-rank, --graphs (also arch flint),
// --prefill-moe / -dense, --decode-moe, --prefill-cold-stage, --prefill-fetch, --prefill-graphs, --mtp) do not apply to it.
//
// The prefix cache (qwen35, qwen38): --prefix-cache 0|1 [1] (finished requests leave their sequence on the slot and
// snapshots of its state; a new request continues the longest cached prefix of its prompt), --prefix-cache-gib F [8]
// (pinned host memory for the snapshots, both cards; one is ~147 MiB with the MTP drafter; allocated on first use, at
// most 8 per slot are kept), --prefix-mark-min N [1024] (a chat request's prefill call ends where its generation prompt
// starts when that is at least N tokens in: the snapshot the next turn continues from; costs one short call),
// --prefix-live 0|1 [0] (continue from a finished request's live state, its reply included: no reply re-prefill, but
// the reply's rows are decode rows where the run without the cache prefills them, so results differ in the last bits;
// 0 = from the mark's snapshot, identical results).
//
// --step-rows 8|16|32|64 [0 = the model's], --seg-rows 8|16 [16], --draft-depths D1,D2.. [the built-in table] (qwen35):
// a step's rows in all; one sequence's verify rows (16 = a DSpine b16 drafter, 16-row GDN logs; 8 = an 8-block DFlash 2
// drafter, the 8-row GDN logs; llama / flint run 8); drafts per sequence at 1, 2.. sequences.
// --drafter-wide N [7]: with --seg-rows 16 one stream drafts 15 after N+ accepted drafts (the same output).
// --adaptive-depth 0|1 [1]: the draft depth of each speculative step from the requests' acceptance and the step times;
// 0 = the fixed depth per request count.
// --unequal-segments 0|1 [1] (qwen38 with the DFlash drafter, the dense 27B with its drafter; with --adaptive-depth 1):
// each request its own draft depth - the requests expected to keep the most tokens draft, the rest run one row in the
// same step; 0 = one depth for all. The output is unchanged.
// --lookup 0|2..8 [4] (with a drafter) lookup drafting: a step's drafts may come from the request's own tokens (the
// tokens after the latest earlier occurrence of its last N tokens) when that has recently yielded more tokens per step
// than the drafter, skipping the drafter's pass. The output is unchanged.

#include "options.hpp"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "xenithium/util/cards.hpp"

namespace xenithium::server_cli {

namespace {

struct Flag {
  std::string_view name, arg;  // arg empty = a switch without a value
  std::function<void(const std::string &)> set;
};

std::vector<Flag> flags(serve::Options &o, std::string &kv_pool) {
  auto str = [](std::string &f) { return [&f](const std::string &v) { f = v; }; };
  auto i32 = [](int &f) { return [&f](const std::string &v) { f = std::atoi(v.c_str()); }; };
  auto i64 = [](auto &f) { return [&f](const std::string &v) { f = std::atoll(v.c_str()); }; };
  auto f64 = [](double &f) { return [&f](const std::string &v) { f = std::atof(v.c_str()); }; };
  auto on = [](bool &f) { return [&f](const std::string &v) { f = std::atoi(v.c_str()) != 0; }; };
  return {
      // what and where
      {"--model", "DIR", str(o.model_dir)},
      {"--chat-template", "auto|builtin|jinja|FILE", str(o.chat_template)},
      {"--host", "H", str(o.host)},
      {"--port", "P", i32(o.port)},
      {"--served-model-name", "S", str(o.served_name)},
      {"--stub", "", [&o](const std::string &) { o.stub = true; }},
      {"--plan", "", [&o](const std::string &) { o.plan = true; }},
      {"--stub-step-ms", "MS", f64(o.stub_step_ms)},
      {"--stub-tokens-per-step", "N", i32(o.stub_tokens_per_step)},
      // sequences, KV, cards
      {"--slots", "N", i32(o.slots)},
      {"--context", "N", i64(o.context)},
      {"--kv-pool-tokens", "auto|N", str(kv_pool)},
      {"--kv", "int8|bf16|int8-v4", str(o.kv)},
      {"--kv-split", "heads|tokens", str(o.kv_split)},
      {"--reserve", "GIB", f64(o.reserve_gib)},
      {"--tp", "1|2|4", i32(o.tp)},
      {"--card", "N", i32(o.card)},
      {"--emu-card", "off|b60|b65", str(o.emu_card)},
      {"--wire", "bf16|fp32|int8", str(o.wire)},
      {"--prefill-wire", "auto|bf16|fp32", str(o.prefill_wire)},
      {"--yarn-tiers", "on|off", str(o.yarn_tiers)},
      {"--threads", "N", i32(o.threads)},
      {"--graphs", "0|1", on(o.graphs)},
      // prefill
      {"--prefill-chunk", "N", i32(o.prefill_chunk)},
      {"--prefill-stall-ms", "MS", f64(o.prefill_stall_ms)},
      {"--prefill-part-ms", "MS", f64(o.prefill_part_ms)},
      {"--prefill-merge", "1..16", i32(o.prefill_merge)},
      {"--prefill-attn", "auto|exact|fast|int8|int8-fast|onednn", str(o.prefill_attn)},
      {"--prefill-w4", "exact|dequant|a8", str(o.prefill_w4)},
      {"--prefill-gemm", "onednn|onednn-all|esimd", str(o.prefill_gemm)},
      {"--attn-tail", "0|1|2", i32(o.attn_tail)},
      {"--prefill-short", "0..32", i32(o.prefill_short)},
      {"--prefill-gemm-rows", "128..4096", i32(o.prefill_gemm_rows)},
      {"--prefill-a8-keep", "SPEC", str(o.prefill_a8_keep)},
      {"--prefill-a8-body", "auto|slm|direct", str(o.prefill_a8_body)},
      {"--prefill-moe", "w4a8|w4a16|w4a16-fast", str(o.prefill_moe)},
      {"--prefill-dense", "w8a8|w8a8-row|w8a16", str(o.prefill_dense)},
      {"--prefill-cold-stage", "N", i32(o.prefill_cold_stage)},
      {"--prefill-fetch", "N", i32(o.prefill_fetch)},
      {"--prefill-graphs", "0|1", on(o.prefill_graphs)},
      {"--kv-lend", "on|off", str(o.kv_lend)},
      {"--kv-lend-keep", "N", i64(o.kv_lend_keep)},
      {"--kv-lend-park-s", "S", f64(o.kv_lend_park_s)},
      {"--kv-lend-pressure", "kv|lru|idle", str(o.kv_lend_pressure)},
      {"--kv-lend-grace-s", "S", f64(o.kv_lend_grace_s)},
      {"--expert-exchange", "on|off", str(o.expert_exchange)},
      // weights
      {"--lm-head", "auto|int8|int4|int4-rescore", str(o.lm_head)},
      {"--lm-head-tiles", "1..32", i32(o.lm_head_tiles)},
      {"--draft-head", "auto|int8|int4", str(o.draft_head)},
      {"--decode-moe", "auto|w4a8|w4a16", str(o.decode_moe)},
      {"--gdn-w4", "int8|int4|int4-qkvz", str(o.gdn_w4)},
      {"--w4-items", "512|1024", i32(o.w4_items)},
      {"--attn-select", "0|32..4096", i32(o.attn_select)},
      {"--select-heads", "auto|same", str(o.select_heads)},
      {"--prefill-select", "0|32..4096", i32(o.prefill_select)},
      {"--decode-attn", "auto|exact|fast|int8|int8-fast", str(o.decode_attn)},
      {"--decode-w4", "exact|a8", str(o.decode_w4)},
      // MoE placement
      {"--hot", "N", i64(o.hot)},
      {"--hot-rank", "FILE", str(o.hot_rank)},
      {"--moe-cpu", "off|auto|avx512|avx2", str(o.moe_cpu)},
      {"--moe-cpu-threads", "N", i32(o.moe_cpu_threads)},
      {"--moe-cpu-cores", "A,B", str(o.moe_cpu_cores)},
      {"--moe-cpu-pct", "P", i32(o.moe_cpu_pct)},
      {"--moe-cpu-min-rows", "N", i32(o.moe_cpu_min_rows)},
      {"--moe-cpu-min-cold", "N", i32(o.moe_cpu_min_cold)},
      {"--moe-cpu-timeout-ms", "MS", i32(o.moe_cpu_timeout_ms)},
      // drafters
      {"--mtp", "DIR", str(o.mtp)},
      {"--mtp-depth", "N", i32(o.mtp_depth)},
      {"--drafter", "0|1", i32(o.drafter)},
      {"--draft-vocab", "K", i32(o.draft_vocab)},
      {"--drafter-model", "DIR", str(o.drafter_model)},
      {"--drafter-defer", "N", i32(o.drafter_defer)},
      {"--drafter-cut", "P", f64(o.drafter_cut)},
      {"--drafter-tree", "N", i32(o.drafter_tree)},
      {"--drafter-tree-seqs", "N", i32(o.drafter_tree_seqs)},
      {"--step-rows", "0|8|16|32|64", i32(o.step_rows)},
      {"--seg-rows", "8|16", i32(o.seg_rows)},
      {"--draft-depths", "D1,D2,..", str(o.draft_depths)},
      {"--drafter-wide", "N", i32(o.drafter_wide)},
      {"--drafter-tree-wide", "N", i32(o.drafter_tree_wide)},
      {"--drafter-mtp-from", "N", i32(o.drafter_mtp_from)},
      {"--drafter-int4", "0|1", i32(o.drafter_int4)},
      {"--adaptive-depth", "0|1", i32(o.adaptive_depth)},
      {"--unequal-segments", "0|1", i32(o.unequal_segments)},
      {"--lookup", "0|2..8", i32(o.lookup)},
      // prefix cache
      {"--prefix-cache", "0|1", i32(o.prefix_cache)},
      {"--prefix-cache-gib", "F", f64(o.prefix_cache_gib)},
      {"--prefix-mark-min", "N", i32(o.prefix_mark_min)},
      {"--prefix-live", "0|1", i32(o.prefix_live)},
      // image input
      {"--vision", "0|1", i32(o.vision)},
      {"--vision-max-pixels", "N", i64(o.vision_max_pixels)},
      {"--vision-images", "N", i32(o.vision_images)},
      {"--media-urls", "0|1", i32(o.media_urls)},
      {"--media-domains", "LIST", str(o.media_domains)},
      {"--media-timeout", "S", f64(o.media_timeout_s)},
      {"--vision-ffmpeg", "PATH", str(o.vision_ffmpeg)},
      {"--video-max-pixels", "N", i64(o.video_max_pixels)},
      {"--video-fps", "F", f64(o.video_fps)},
      // sampling defaults
      {"--temperature", "X", str(o.temperature)},
      {"--top-p", "X", str(o.top_p)},
      {"--top-k", "N", str(o.top_k)},
      {"--min-p", "X", str(o.min_p)},
      {"--presence-penalty", "X", str(o.presence_penalty)},
      {"--repetition-penalty", "X", str(o.repetition_penalty)},
      // serving loop
      {"--warmup", "0|1", on(o.warmup)},
      {"--sample-threads", "N", i32(o.sample_threads)},
      {"--max-queue", "N", i32(o.max_queue)},
      {"--queue-policy", "fifo|sjf", str(o.queue_policy)},
      {"--queue-aging", "S", f64(o.queue_aging_s)},
      {"--queue-timeout", "S", f64(o.queue_timeout_s)},
      {"--idle-timeout", "S", i32(o.idle_timeout_s)},
      {"--cors-origin", "O", str(o.cors_origin)},
      {"--log-interval", "S", f64(o.log_interval_s)},
  };
}

bool one_of(const std::string &v, std::initializer_list<std::string_view> set) {
  for (const auto s : set)
    if (v == s) return true;
  return false;
}

// The checks every backend relies on; arch-specific limits are the backends' (serve:OptionError).
int validate(const serve::Options &o) {
  auto bad = [](const char *msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 2;
  };
  if (o.model_dir.empty()) return bad("--model DIR is required (the tokenizer and chat format come from it, also with --stub)");
  const bool int8_attn = o.prefill_attn.rfind("int8", 0) == 0;
  if (o.slots < 1 || o.slots > 64 || !one_of(o.kv, {"int8", "bf16", "int8-v4"}) || !one_of(o.wire, {"bf16", "fp32", "int8"}) || !one_of(o.prefill_wire, {"auto", "bf16", "fp32"}) || !one_of(o.yarn_tiers, {"on", "off"}) || (o.w4_items != 512 && o.w4_items != 1024) ||
      !one_of(o.prefill_attn, {"auto", "exact", "fast", "int8", "int8-fast", "onednn"}) || (int8_attn && o.kv == "bf16") || !one_of(o.prefill_w4, {"exact", "dequant", "a8"}) || !one_of(o.prefill_gemm, {"onednn", "onednn-all", "esimd"}) || !one_of(o.prefill_a8_body, {"auto", "slm", "direct"}) ||
      !one_of(o.lm_head, {"auto", "int8", "int4", "int4-rescore"}) || o.lm_head_tiles < 1 || o.lm_head_tiles > 32 || !one_of(o.draft_head, {"auto", "int8", "int4"}) ||
      o.prefill_chunk < 0 || o.prefill_chunk % 4 || o.draft_vocab < 0 || o.draft_vocab % 32 || !one_of(o.prefill_moe, {"w4a8", "w4a16", "w4a16-fast"}) ||
      !one_of(o.prefill_dense, {"w8a8", "w8a8-row", "w8a16"}) || !one_of(o.decode_moe, {"w4a8", "w4a16", "auto"}) || !one_of(o.gdn_w4, {"int8", "int4", "int4-qkvz"}) ||
      !one_of(o.kv_lend, {"on", "off"}) || o.kv_lend_keep < 0 || o.kv_lend_park_s < 0 || o.kv_lend_grace_s < 0 ||
      !one_of(o.kv_lend_pressure, {"kv", "lru", "idle"}) || !one_of(o.kv_split, {"heads", "tokens"}) ||
      !one_of(o.expert_exchange, {"on", "off"}))
    return bad("--kv-split heads|tokens, --kv-lend on|off, --kv-lend-keep N >= 0, --kv-lend-park-s / --kv-lend-grace-s S >= 0, --kv-lend-pressure kv|lru|idle, --expert-exchange on|off, --slots takes 1..16 (qwen38 on four cards 1..64, llama / flint 1..8), --kv int8|bf16|int8-v4 (int8-v4: qwen38 and the dense 27B), --wire bf16|fp32|int8, --prefill-wire auto|bf16|fp32, --w4-items 512|1024, --prefill-attn auto|exact|fast|int8|int8-fast|onednn (int8* need --kv int8 | int8-v4), --prefill-w4 exact|dequant|a8, --prefill-gemm onednn|onednn-all|esimd, --prefill-a8-body auto|slm|direct, --lm-head auto|int8|int4|int4-rescore, --lm-head-tiles 1..32, --draft-head auto|int8|int4, --prefill-chunk a multiple of 4 (0 = the arch's), --prefill-moe w4a8|w4a16|w4a16-fast, --decode-moe auto|w4a8|w4a16, --prefill-dense w8a8|w8a8-row|w8a16, --gdn-w4 int8|int4|int4-qkvz, --draft-vocab a multiple of 32");
  if (o.drafter_defer < 0 || (o.drafter_defer > 0 && o.drafter_defer < 8) || o.drafter_defer > 512 || !(o.drafter_cut >= 0 && o.drafter_cut < 1) ||
      o.drafter_mtp_from < 0 || o.drafter_mtp_from > 64 || o.drafter_tree < 0 || o.drafter_tree > 7 ||
      o.drafter_tree_seqs < 1 || o.drafter_tree_seqs > 8 || o.drafter_wide < 0 || o.drafter_wide > 15 || o.drafter_tree_wide < 1 || o.drafter_tree_wide > 15)
    return bad("--drafter-defer takes 0 or 8..512 rows, --drafter-cut a probability in [0, 1), --drafter-mtp-from 0..64 sequences, --drafter-tree 0..7 nodes, --drafter-tree-seqs 1..8 sequences, --drafter-wide 0..15 accepted drafts, --drafter-tree-wide 1..15 nodes");
  if (o.drafter_int4 != 0 && o.drafter_int4 != 1) return bad("--drafter-int4 takes 0 or 1");
  {  // Step rows
    std::vector<int> dd;
    // 0 = the model's default (qwen35 32; qwen38 16, 64 on four cards); the checks below take qwen35's 32 (qwen38 reads no
    // seg rows or draft depths)
    const int step = o.step_rows_or(32);
    // 64 (qwen38 on four cards: the model refuses it elsewhere)
    if ((o.step_rows != 0 && o.step_rows != 8 && o.step_rows != 16 && o.step_rows != 32 && o.step_rows != 64) || (o.seg_rows != 8 && o.seg_rows != 16) ||
        o.seg_rows > step)
      return bad("--step-rows takes 0 (the model's default), 8, 16, 32 or 64, --seg-rows 8 or 16 (--seg-rows <= --step-rows)");
    if (o.step_rows == 8 && o.slots > 8) return bad("--step-rows 8 holds at most 8 sequences a step: --slots 1..8");
    if (!serve::parse_draft_depths(o.draft_depths, dd)) return bad("--draft-depths takes up to 16 comma-separated drafts per sequence, each 0..15 (\"15,7,4,3\")");
    for (size_t i = 0; i < dd.size(); ++i) {
      if (dd[i] > o.seg_rows - 1) return bad(("--draft-depths: " + std::to_string(dd[i]) + " drafts need --seg-rows 16 (8-row segments: at most 7)").c_str());
      if (static_cast<int>(i + 1) * (dd[i] + 1) > step)
        return bad(("--draft-depths: " + std::to_string(i + 1) + " sequences x " + std::to_string(dd[i] + 1) + " rows exceed --step-rows " + std::to_string(step)).c_str());
    }
  }
  if ((o.tp != 1 && o.tp != 2 && o.tp != 4) || o.card < 0 || (o.tp != 1 && o.card != 0))
    return bad("--tp takes 1, 2 or 4 (4: qwen38), --card a device index (TP1 only; TP2 runs on cards 0 and 1, TP4 on 0-3)");
  if (!util::emu_card_valid(o.emu_card) || o.emu_card == "b70") return bad("--emu-card takes off, b60 or b65");
  if (o.attn_select != 0 && (o.attn_select < 32 || o.attn_select > 4096)) return bad("--attn-select takes 0 (dense) or 32..4096 pages");
  if (!one_of(o.decode_attn, {"auto", "exact", "fast", "int8", "int8-fast"}) || (o.decode_attn.rfind("int8", 0) == 0 && o.kv == "bf16"))
    return bad("--decode-attn takes auto|exact|fast|int8|int8-fast (int8* need --kv int8 | int8-v4)");
  if (o.prefill_select != 0 && (o.prefill_select < 32 || o.prefill_select > 4096)) return bad("--prefill-select takes 0 (dense) or 32..4096 pages");
  if (o.prefill_merge < 1 || o.prefill_merge > 16) return bad("--prefill-merge takes 1 (a call per prompt) to 16 prompts per call");
  if (!one_of(o.select_heads, {"same", "auto"})) return bad("--select-heads takes auto (each sequence measures its heads' page counts) or same (N pages for every head)");
  if (o.attn_select != 0 && !one_of(o.decode_attn, {"auto", "exact", "int8-fast"}))
    return bad("--attn-select runs with --decode-attn auto, exact or int8-fast (fast and int8 run on dense attention only)");
  if (!one_of(o.decode_w4, {"exact", "a8"})) return bad("--decode-w4 takes exact|a8");
  if (!one_of(o.moe_cpu, {"off", "auto", "avx512", "avx2"}) || o.moe_cpu_threads < 1 || o.moe_cpu_threads > 64 || o.moe_cpu_pct < 0 || o.moe_cpu_pct > 100 ||
      o.moe_cpu_min_rows < 1 || o.moe_cpu_min_rows > 8 || o.moe_cpu_min_cold < 1 || o.moe_cpu_timeout_ms < 1)
    return bad("--moe-cpu takes off|auto|avx512|avx2, --moe-cpu-threads 1..64, --moe-cpu-pct 0..100, --moe-cpu-min-rows 1..8, --moe-cpu-min-cold >= 1, --moe-cpu-timeout-ms >= 1");
  if (o.adaptive_depth != 0 && o.adaptive_depth != 1) return bad("--adaptive-depth takes 0 or 1");
  if (o.unequal_segments != 0 && o.unequal_segments != 1) return bad("--unequal-segments takes 0 or 1");
  if (o.lookup != 0 && (o.lookup < 2 || o.lookup > 8)) return bad("--lookup takes 0 (off) or a match length 2..8");
  if (o.prefill_short < 0 || o.prefill_short > 32) return bad("--prefill-short takes 0 (off) .. 32 rows");
  if (o.prefill_gemm_rows < 128 || o.prefill_gemm_rows > 4096) return bad("--prefill-gemm-rows takes 128 .. 4096 rows per call");
  if (o.prefill_fetch < 0 || o.prefill_fetch > 512) return bad("--prefill-fetch takes 0 (off) .. 512 expert blobs per layer");
  if (o.hot < 0) return bad("--hot takes 0 (auto) or a positive expert count (the routed experts in VRAM over all layers)");
  if (o.attn_tail < 0 || o.attn_tail > 2 || (o.prefix_cache != 0 && o.prefix_cache != 1) || (o.prefix_live != 0 && o.prefix_live != 1) || !(o.prefix_cache_gib >= 0) || o.prefix_mark_min < 0)
    return bad("--attn-tail takes 0, 1 or 2, --prefix-cache and --prefix-live take 0 or 1, --prefix-cache-gib a size >= 0, --prefix-mark-min a token count >= 0");
  if ((o.vision != 0 && o.vision != 1) || o.vision_max_pixels < 65536 || o.vision_max_pixels > 16777216 || o.vision_images < 1 || o.vision_images > 256)
    return bad("--vision takes 0 or 1, --vision-max-pixels 65,536..16,777,216 pixels, --vision-images 1..256");
  if ((o.media_urls != 0 && o.media_urls != 1) || !(o.media_timeout_s > 0 && o.media_timeout_s <= 600))
    return bad("--media-urls takes 0 or 1, --media-timeout 0 < S <= 600 seconds");
  if (o.video_max_pixels < 65536 || o.video_max_pixels > 268435456 || !(o.video_fps > 0 && o.video_fps <= 30))
    return bad("--video-max-pixels takes 65,536..268,435,456 pixels, --video-fps 0 < F <= 30");
  return -1;
}

}  // namespace

int parse(int argc, char **argv, serve::Options &o) {
  std::string kv_pool = "auto";
  const auto table = flags(o, kv_pool);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::printf("xenithium-server --model DIR [options]   (every setting: docs/server-env.md)\n");
      for (const auto &f : table) std::printf("  %.*s %.*s\n", static_cast<int>(f.name.size()), f.name.data(), static_cast<int>(f.arg.size()), f.arg.data());
      return 0;
    }
    const Flag *hit = nullptr;
    for (const auto &f : table)
      if (f.name == a) hit = &f;
    if (!hit) {
      std::fprintf(stderr, "unknown option %s (xenithium-server --help; docs/server-env.md)\n", a.c_str());
      return 2;
    }
    if (hit->arg.empty()) {
      hit->set("");
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", a.c_str());
      return 2;
    }
    hit->set(argv[++i]);
  }
  if (kv_pool != "auto") {
    char *end = nullptr;
    o.kv_pool_tokens = std::strtoll(kv_pool.c_str(), &end, 10);
    if (kv_pool.empty() || *end || o.kv_pool_tokens < 0) {
      std::fprintf(stderr, "--kv-pool-tokens takes auto or a token count\n");
      return 2;
    }
    if (o.kv_pool_tokens == 0) {  // the fixed per-slot layout (every slot owning --context tokens) is gone from the server
      std::fprintf(stderr, "--kv-pool-tokens 0 (a fixed context per slot) was removed: the slots share one KV pool; pass auto or a token count\n");
      return 2;
    }
  }
  return validate(o);
}

}  // namespace xenithium:server_cli
