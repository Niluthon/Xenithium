#!/usr/bin/env bash
# Starts xenithium-server (build it first: scripts/build-host.sh xenithium-server).
# Settings come from ./server.env when present (KEY=value lines), else the defaults below; every
# setting is documented in docs/server-env.md. SERVE_ENV=FILE reads FILE instead of ./server.env
# (scripts/launch.sh uses it). Extra arguments are appended to the command line (a later flag
# wins). SERVE_DRY_RUN=1 prints the command instead of running it; SERVE_PLAN=1 prints the load's memory plan.
set -euo pipefail
cd "$(dirname "$0")/.."
env_file=${SERVE_ENV:-server.env}
if [[ -n "${SERVE_ENV:-}" && ! -f "$env_file" ]]; then
  echo "start-server: SERVE_ENV=$env_file not found" >&2
  exit 1
fi
# the caller's model choice (XEN_MODEL_ENV=... scripts/start-server.sh) wins over the settings file's
caller_model_env=${XEN_MODEL_ENV-$'\x01'}
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi
[[ "$caller_model_env" != $'\x01' ]] && XEN_MODEL_ENV=$caller_model_env
# Per-model settings (docs/server-env.md, "Per-model settings files"): one checkpoint's settings, under the settings file
# and the caller's environment (a key set there wins), over the defaults below. The shipped files are the presets in
# models/presets/; your own are models/<id>.env (the launcher writes them, git ignores them) and replace the preset of
# the same name, so an update of the project never changes your settings.
# XEN_MODEL_ENV = an id or a path; unset / auto = the file whose XEN_MODEL is this checkpoint; empty = none. A file for
# another checkpoint than XEN_MODEL is not applied (auto lookup instead).
# dir/presets/name.env -> dir/name.env (where your file of that name is); any other path as it is.
user_file() { local d; d=$(dirname "$1"); if [[ "$(basename "$d")" == presets ]]; then printf '%s/%s' "$(dirname "$d")" "$(basename "$1")"; else printf '%s' "$1"; fi; }
# dir/name.env -> itself when it exists, else the preset dir/presets/name.env when that exists, else itself.
settings_file() { local u; u=$(user_file "$1"); local p; p="$(dirname "$u")/presets/$(basename "$u")"; if [[ -f "$u" || ! -f "$p" ]]; then printf '%s' "$u"; else printf '%s' "$p"; fi; }
model_env_model() { env -i HOME="$HOME" bash -c 'source "$1" >/dev/null 2>&1; printf "%s" "${XEN_MODEL/#\~/$HOME}"' _ "$1"; }
same_dir() { [[ -n "$1" && -n "$2" && "$(readlink -f "$1" 2>/dev/null || echo "$1")" == "$(readlink -f "$2" 2>/dev/null || echo "$2")" ]]; }
model_env=${XEN_MODEL_ENV-auto}
model_env_file=""
want_model=${XEN_MODEL:-$HOME/models/qwen38-w4a16}
want_model=${want_model/#\~/$HOME}
if [[ -n "$model_env" && "$model_env" != auto ]]; then
  if [[ "$model_env" == */* || "$model_env" == *.env ]]; then model_env_file=$(settings_file "$model_env"); else model_env_file=$(settings_file "models/$model_env.env"); fi
  if [[ ! -f "$model_env_file" ]]; then
    echo "start-server: XEN_MODEL_ENV=$model_env: $model_env_file not found; no per-model settings" >&2
    model_env_file=""
  elif [[ -n "${XEN_MODEL:-}" ]] && ! same_dir "$want_model" "$(model_env_model "$model_env_file")"; then
    echo "start-server: $model_env_file is for $(model_env_model "$model_env_file"), not XEN_MODEL=$XEN_MODEL; looking up that model's file" >&2
    model_env_file="" model_env=auto
  fi
fi
if [[ "$model_env" == auto ]]; then
  for f in models/*.env models/presets/*.env; do
    [[ "$f" =~ \.tp[0-9]+\.env$ ]] && continue  # a setup file (it names no checkpoint)
    [[ "$f" == models/presets/* && -f "models/${f##*/}" ]] && continue  # your file of that name replaces the preset
    grep -q "^XEN_PRESET_EXPLICIT=1" "$f" && continue  # a variant that needs extra files: only by name (XEN_MODEL_ENV=<id>)
    [[ -f "$f" ]] && same_dir "$want_model" "$(model_env_model "$f")" && { model_env_file=$f; break; }
  done
fi
env_pairs() { env -i HOME="$HOME" bash -c 'set -a; source "$1" >/dev/null 2>&1; for k in $(compgen -v XEN_); do printf "%s\t%s\n" "$k" "${!k}"; done' _ "$1"; }
if [[ -n "$model_env_file" ]]; then
  declare -A model_vals=()
  while IFS=$'\t' read -r k v; do [[ -n "$k" ]] && model_vals[$k]=$v; done < <(env_pairs "$model_env_file")
  # Per-setup settings (docs/server-env.md "Per-setup settings files"): <id>.tp<N>.env holds one card count's own
  # settings over the model's file; N = the setup key's value (XEN_Q38_TP, or XEN_TP for the qwen35 runtime's model
  # types) from the environment, the settings file, the model's file, else 2. Your file replaces the preset of the same
  # name, as for the model's file. The setup file never sets the setup key, the checkpoint or its served name.
  setup_model=${XEN_MODEL:-${model_vals[XEN_MODEL]:-}}
  if grep -qE '"model_type": *"(qwen3_5|llama|flint)"' "${setup_model/#\~/$HOME}/config.json" 2>/dev/null; then setup_key=XEN_TP; else setup_key=XEN_Q38_TP; fi
  setup_value=${!setup_key:-${model_vals[$setup_key]:-2}}
  setup_env_file=$(settings_file "${model_env_file%.env}.tp$setup_value.env")
  if [[ -f "$setup_env_file" ]]; then
    while IFS=$'\t' read -r k v; do
      [[ -n "$k" && "$k" != "$setup_key" && "$k" != XEN_MODEL && "$k" != XEN_SERVED_NAME && "$k" != XEN_MODELS_DIR && -z "${!k+x}" ]] && export "$k=$v"
    done < <(env_pairs "$setup_env_file")
    echo "start-server: per-setup settings from $setup_env_file" >&2
  fi
  # every XEN_ key of the model's file that neither the settings file, the environment nor the setup file set
  for k in "${!model_vals[@]}"; do
    [[ -z "${!k+x}" ]] && export "$k=${model_vals[$k]}"
  done
  echo "start-server: per-model settings from $model_env_file" >&2
fi
XEN_MODEL=${XEN_MODEL:-$HOME/models/qwen38-w4a16}
# The launcher's models folder (the m list scans it); the server itself never reads it (empty = the project's models/).
XEN_MODELS_DIR=${XEN_MODELS_DIR:-}
XEN_HOST=${XEN_HOST:-0.0.0.0}
XEN_PORT=${XEN_PORT:-8000}
XEN_SERVED_NAME=${XEN_SERVED_NAME:-xenithium}
XEN_SLOTS=${XEN_SLOTS:-4}
XEN_CONTEXT=${XEN_CONTEXT:-262144}
XEN_KV_POOL_TOKENS=${XEN_KV_POOL_TOKENS:-auto}
XEN_KV=${XEN_KV:-int8}
XEN_KV_SPLIT=${XEN_KV_SPLIT:-heads}
XEN_HOT=${XEN_HOT:-}
XEN_HOT_RANK=${XEN_HOT_RANK-auto}
XEN_MOE_CPU=${XEN_MOE_CPU:-off}
XEN_MOE_CPU_THREADS=${XEN_MOE_CPU_THREADS:-8}
XEN_MOE_CPU_CORES=${XEN_MOE_CPU_CORES:-}
XEN_MOE_CPU_PCT=${XEN_MOE_CPU_PCT:-70}
XEN_MOE_CPU_MIN_ROWS=${XEN_MOE_CPU_MIN_ROWS:-1}
XEN_MOE_CPU_MIN_COLD=${XEN_MOE_CPU_MIN_COLD:-2}
XEN_MOE_CPU_TIMEOUT_MS=${XEN_MOE_CPU_TIMEOUT_MS:-500}
XEN_RESERVE=${XEN_RESERVE:-0.25}
XEN_GRAPHS=${XEN_GRAPHS:-1}
XEN_PREFILL_CHUNK=${XEN_PREFILL_CHUNK:-0}
XEN_PREFILL_STALL_MS=${XEN_PREFILL_STALL_MS:-100}
XEN_PREFILL_PART_MS=${XEN_PREFILL_PART_MS:--1}
XEN_PREFILL_MERGE=${XEN_PREFILL_MERGE:-16}
XEN_PREFILL_MOE=${XEN_PREFILL_MOE:-auto}
XEN_PREFILL_DENSE=${XEN_PREFILL_DENSE:-w8a8-row}
XEN_DECODE_MOE=${XEN_DECODE_MOE:-w4a16}
XEN_GDN_W4=${XEN_GDN_W4:-int8}
XEN_PREFILL_COLD_STAGE=${XEN_PREFILL_COLD_STAGE:-16}
XEN_PREFILL_FETCH=${XEN_PREFILL_FETCH:-0}
XEN_PREFILL_GRAPHS=${XEN_PREFILL_GRAPHS:-0}
XEN_KV_LEND=${XEN_KV_LEND:-on}
XEN_KV_LEND_KEEP=${XEN_KV_LEND_KEEP:-32768}
XEN_KV_LEND_PARK_S=${XEN_KV_LEND_PARK_S:-300}
XEN_KV_LEND_PRESSURE=${XEN_KV_LEND_PRESSURE:-kv}
XEN_KV_LEND_GRACE_S=${XEN_KV_LEND_GRACE_S:-60}
XEN_EXPERT_EXCHANGE=${XEN_EXPERT_EXCHANGE:-on}
XEN_SAMPLE_THREADS=${XEN_SAMPLE_THREADS:-4}
# the sampling of requests that send no value of their own (empty = the checkpoint's generation_config.json)
XEN_TEMPERATURE=${XEN_TEMPERATURE:-}
XEN_TOP_P=${XEN_TOP_P:-}
XEN_TOP_K=${XEN_TOP_K:-}
XEN_MIN_P=${XEN_MIN_P:-}
XEN_PRESENCE_PENALTY=${XEN_PRESENCE_PENALTY:-}
XEN_REPETITION_PENALTY=${XEN_REPETITION_PENALTY:-}
XEN_MTP=${XEN_MTP:-}
XEN_DRAFTER=${XEN_DRAFTER:-1}
XEN_DRAFTER_MODEL=${XEN_DRAFTER_MODEL:-}
XEN_DRAFT_VOCAB=${XEN_DRAFT_VOCAB:-98304}
XEN_DRAFTER_DEFER=${XEN_DRAFTER_DEFER:-64}
XEN_DRAFTER_CUT=${XEN_DRAFTER_CUT:-0.02}
XEN_DRAFTER_TREE=${XEN_DRAFTER_TREE:-7}
XEN_DRAFTER_TREE_SEQS=${XEN_DRAFTER_TREE_SEQS:-2}
# 0 = the model's default (qwen35 32; qwen38 16, 64 on four cards)
XEN_STEP_ROWS=${XEN_STEP_ROWS:-0}
XEN_SEG_ROWS=${XEN_SEG_ROWS:-16}
XEN_DRAFT_DEPTHS=${XEN_DRAFT_DEPTHS:-}
XEN_DRAFTER_WIDE=${XEN_DRAFTER_WIDE:-7}
XEN_DRAFTER_TREE_WIDE=${XEN_DRAFTER_TREE_WIDE:-15}
XEN_DRAFTER_MTP_FROM=${XEN_DRAFTER_MTP_FROM:-9}
XEN_DRAFTER_INT4=${XEN_DRAFTER_INT4:-1}
XEN_WIRE=${XEN_WIRE:-bf16}
XEN_PREFILL_WIRE=${XEN_PREFILL_WIRE:-auto}
XEN_YARN_TIERS=${XEN_YARN_TIERS:-on}
XEN_TP=${XEN_TP:-2}
XEN_CARD=${XEN_CARD:-0}
XEN_Q38_TP=${XEN_Q38_TP:-2}
XEN_Q38_CARD=${XEN_Q38_CARD:-0}
# run as an Arc Pro B60 / B65 on the B70s (off | b60 | b65): launch shapes and VRAM of that card (docs/server-env.md)
XEN_EMU_CARD=${XEN_EMU_CARD:-off}
# each arch keeps its own TP pair: qwen35 (model_type qwen3_5), llama (Nanbeige4.3, model_type llama) and flint (Recursive
# Flint INT4, model_type flint): the qwen35 runtime, XEN_TP / XEN_CARD; qwen38 XEN_Q38_TP / XEN_Q38_CARD
if grep -qE '"model_type": *"(qwen3_5|llama|flint)"' "${XEN_MODEL/#\~/$HOME}/config.json" 2>/dev/null; then tp=$XEN_TP card=$XEN_CARD
else tp=$XEN_Q38_TP card=$XEN_Q38_CARD; fi
XEN_W4_ITEMS=${XEN_W4_ITEMS:-1024}
XEN_ATTN_SELECT=${XEN_ATTN_SELECT:-0}
XEN_PREFILL_SELECT=${XEN_PREFILL_SELECT:-0}
XEN_SELECT_HEADS=${XEN_SELECT_HEADS:-auto}
XEN_DECODE_ATTN=${XEN_DECODE_ATTN:-auto}
XEN_DECODE_W4=${XEN_DECODE_W4:-exact}
XEN_PREFILL_ATTN=${XEN_PREFILL_ATTN:-auto}
XEN_PREFILL_W4=${XEN_PREFILL_W4:-a8}
XEN_PREFILL_GEMM=${XEN_PREFILL_GEMM:-onednn}
XEN_PREFILL_GEMM_ROWS=${XEN_PREFILL_GEMM_ROWS:-1024}
XEN_ATTN_TAIL=${XEN_ATTN_TAIL:-2}
XEN_PREFILL_SHORT=${XEN_PREFILL_SHORT:-32}
XEN_PREFILL_A8_BODY=${XEN_PREFILL_A8_BODY:-auto}
XEN_LM_HEAD=${XEN_LM_HEAD:-auto}
XEN_LM_HEAD_TILES=${XEN_LM_HEAD_TILES:-32}
XEN_DRAFT_HEAD=${XEN_DRAFT_HEAD:-auto}
XEN_PREFILL_A8_KEEP=${XEN_PREFILL_A8_KEEP:-}
XEN_PREFIX_CACHE=${XEN_PREFIX_CACHE:-1}
XEN_PREFIX_CACHE_GIB=${XEN_PREFIX_CACHE_GIB:-8}
XEN_PREFIX_MARK_MIN=${XEN_PREFIX_MARK_MIN:-1024}
XEN_PREFIX_LIVE=${XEN_PREFIX_LIVE:-0}
XEN_ADAPTIVE_DEPTH=${XEN_ADAPTIVE_DEPTH:-1}
XEN_UNEQUAL_SEGMENTS=${XEN_UNEQUAL_SEGMENTS:-1}
XEN_LOOKUP=${XEN_LOOKUP:-4}
XEN_MTP_DEPTH=${XEN_MTP_DEPTH:-3}
XEN_WARMUP=${XEN_WARMUP:-1}
XEN_MAX_QUEUE=${XEN_MAX_QUEUE:-0}
XEN_QUEUE_POLICY=${XEN_QUEUE_POLICY:-fifo}
XEN_QUEUE_AGING=${XEN_QUEUE_AGING:-10}
XEN_QUEUE_TIMEOUT=${XEN_QUEUE_TIMEOUT:-0}
XEN_IDLE_TIMEOUT=${XEN_IDLE_TIMEOUT:-300}
XEN_LOG_INTERVAL=${XEN_LOG_INTERVAL:-10}
XEN_STUB=${XEN_STUB:-0}
XEN_CHAT_TEMPLATE=${XEN_CHAT_TEMPLATE:-auto}
XEN_VISION=${XEN_VISION:-0}
XEN_VISION_MAX_PIXELS=${XEN_VISION_MAX_PIXELS:-1048576}
XEN_VISION_IMAGES=${XEN_VISION_IMAGES:-16}
XEN_MEDIA_URLS=${XEN_MEDIA_URLS:-1}
XEN_MEDIA_DOMAINS=${XEN_MEDIA_DOMAINS:-}
XEN_MEDIA_TIMEOUT=${XEN_MEDIA_TIMEOUT:-30}
XEN_VISION_FFMPEG=${XEN_VISION_FFMPEG-ffmpeg}
XEN_VIDEO_MAX_PIXELS=${XEN_VIDEO_MAX_PIXELS:-25165824}
XEN_VIDEO_FPS=${XEN_VIDEO_FPS:-2}

cmd=(./build/xenithium-server --model "$XEN_MODEL" --host "$XEN_HOST" --port "$XEN_PORT" --served-model-name "$XEN_SERVED_NAME"
     --slots "$XEN_SLOTS" --context "$XEN_CONTEXT" --kv-pool-tokens "$XEN_KV_POOL_TOKENS" --kv "$XEN_KV" --kv-split "$XEN_KV_SPLIT" --reserve "$XEN_RESERVE" --graphs "$XEN_GRAPHS"
     --prefill-chunk "$XEN_PREFILL_CHUNK" --prefill-stall-ms "$XEN_PREFILL_STALL_MS" --prefill-part-ms "$XEN_PREFILL_PART_MS" --prefill-merge "$XEN_PREFILL_MERGE" --prefill-moe "$XEN_PREFILL_MOE" --prefill-dense "$XEN_PREFILL_DENSE" --decode-moe "$XEN_DECODE_MOE" --gdn-w4 "$XEN_GDN_W4" --prefill-cold-stage "$XEN_PREFILL_COLD_STAGE" --prefill-fetch "$XEN_PREFILL_FETCH" --prefill-graphs "$XEN_PREFILL_GRAPHS" --sample-threads "$XEN_SAMPLE_THREADS" --warmup "$XEN_WARMUP" --max-queue "$XEN_MAX_QUEUE" --queue-policy "$XEN_QUEUE_POLICY"
     --queue-aging "$XEN_QUEUE_AGING" --queue-timeout "$XEN_QUEUE_TIMEOUT" --idle-timeout "$XEN_IDLE_TIMEOUT"
     --log-interval "$XEN_LOG_INTERVAL" --drafter "$XEN_DRAFTER" --draft-vocab "$XEN_DRAFT_VOCAB" --drafter-defer "$XEN_DRAFTER_DEFER" --drafter-cut "$XEN_DRAFTER_CUT" --drafter-tree "$XEN_DRAFTER_TREE" --drafter-tree-seqs "$XEN_DRAFTER_TREE_SEQS" --step-rows "$XEN_STEP_ROWS" --seg-rows "$XEN_SEG_ROWS" --drafter-wide "$XEN_DRAFTER_WIDE" --drafter-tree-wide "$XEN_DRAFTER_TREE_WIDE" --drafter-mtp-from "$XEN_DRAFTER_MTP_FROM" --drafter-int4 "$XEN_DRAFTER_INT4" --wire "$XEN_WIRE" --prefill-wire "$XEN_PREFILL_WIRE" --yarn-tiers "$XEN_YARN_TIERS" --tp "$tp" --card "$card" --emu-card "$XEN_EMU_CARD" --w4-items "$XEN_W4_ITEMS" --attn-select "$XEN_ATTN_SELECT" --prefill-select "$XEN_PREFILL_SELECT" --decode-attn "$XEN_DECODE_ATTN" --decode-w4 "$XEN_DECODE_W4"
     --prefill-attn "$XEN_PREFILL_ATTN" --prefill-w4 "$XEN_PREFILL_W4" --prefill-gemm "$XEN_PREFILL_GEMM" --attn-tail "$XEN_ATTN_TAIL" --prefill-short "$XEN_PREFILL_SHORT" --prefill-a8-body "$XEN_PREFILL_A8_BODY" --lm-head "$XEN_LM_HEAD" --lm-head-tiles "$XEN_LM_HEAD_TILES" --draft-head "$XEN_DRAFT_HEAD"
     --prefill-gemm-rows "$XEN_PREFILL_GEMM_ROWS" --kv-lend "$XEN_KV_LEND" --kv-lend-keep "$XEN_KV_LEND_KEEP" --kv-lend-park-s "$XEN_KV_LEND_PARK_S" --kv-lend-pressure "$XEN_KV_LEND_PRESSURE" --kv-lend-grace-s "$XEN_KV_LEND_GRACE_S" --expert-exchange "$XEN_EXPERT_EXCHANGE" --prefix-cache "$XEN_PREFIX_CACHE" --prefix-cache-gib "$XEN_PREFIX_CACHE_GIB" --prefix-mark-min "$XEN_PREFIX_MARK_MIN" --prefix-live "$XEN_PREFIX_LIVE"
     --adaptive-depth "$XEN_ADAPTIVE_DEPTH" --unequal-segments "$XEN_UNEQUAL_SEGMENTS" --lookup "$XEN_LOOKUP" --mtp-depth "$XEN_MTP_DEPTH" --chat-template "${XEN_CHAT_TEMPLATE/#\~/$HOME}"
     --vision "$XEN_VISION" --vision-max-pixels "$XEN_VISION_MAX_PIXELS" --vision-images "$XEN_VISION_IMAGES"
     --media-urls "$XEN_MEDIA_URLS" --media-timeout "$XEN_MEDIA_TIMEOUT"
     --vision-ffmpeg "${XEN_VISION_FFMPEG/#\~/$HOME}" --video-max-pixels "$XEN_VIDEO_MAX_PIXELS" --video-fps "$XEN_VIDEO_FPS")
[[ -n "$XEN_MEDIA_DOMAINS" ]] && cmd+=(--media-domains "$XEN_MEDIA_DOMAINS")
[[ -n "$XEN_TEMPERATURE" ]] && cmd+=(--temperature "$XEN_TEMPERATURE")
[[ -n "$XEN_TOP_P" ]] && cmd+=(--top-p "$XEN_TOP_P")
[[ -n "$XEN_TOP_K" ]] && cmd+=(--top-k "$XEN_TOP_K")
[[ -n "$XEN_MIN_P" ]] && cmd+=(--min-p "$XEN_MIN_P")
[[ -n "$XEN_PRESENCE_PENALTY" ]] && cmd+=(--presence-penalty "$XEN_PRESENCE_PENALTY")
[[ -n "$XEN_REPETITION_PENALTY" ]] && cmd+=(--repetition-penalty "$XEN_REPETITION_PENALTY")
[[ -n "$XEN_DRAFTER_MODEL" ]] && cmd+=(--drafter-model "${XEN_DRAFTER_MODEL/#\~/$HOME}")
[[ -n "$XEN_DRAFT_DEPTHS" ]] && cmd+=(--draft-depths "$XEN_DRAFT_DEPTHS")
cmd+=(--select-heads "$XEN_SELECT_HEADS")
[[ -n "$XEN_PREFILL_A8_KEEP" ]] && cmd+=(--prefill-a8-keep "$XEN_PREFILL_A8_KEEP")
[[ -n "$XEN_HOT" ]] && cmd+=(--hot "$XEN_HOT")
# MoE hot-expert ranking (qwen38): the server records it while serving and saves it every 60 s of
# decoding and at exit; the next start places the hot set from it. auto = the model's own folder in .cache/experts/ of
# this repo (nothing has to exist: the first run creates it).
moe=0
grep -qE '"model_type": *"qwen4_exp"' "${XEN_MODEL/#\~/$HOME}/config.json" 2>/dev/null && moe=1
expert_dir=$PWD/.cache/experts/$(basename "${XEN_MODEL%/}")
if [[ "$XEN_HOT_RANK" == auto ]]; then
  XEN_HOT_RANK=""
  if [[ "$moe" == 1 ]]; then
    XEN_HOT_RANK=$expert_dir/hot-rank.xrc
  fi
fi
XEN_HOT_RANK=${XEN_HOT_RANK/#\~/$HOME}
[[ -n "$XEN_HOT_RANK" ]] && cmd+=(--hot-rank "$XEN_HOT_RANK")
[[ -n "$XEN_MTP" ]] && cmd+=(--mtp "$XEN_MTP")
# the decode MoE's CPU path for cold experts (qwen38; off by default)
cmd+=(--moe-cpu "$XEN_MOE_CPU" --moe-cpu-threads "$XEN_MOE_CPU_THREADS" --moe-cpu-pct "$XEN_MOE_CPU_PCT" --moe-cpu-min-rows "$XEN_MOE_CPU_MIN_ROWS"
      --moe-cpu-min-cold "$XEN_MOE_CPU_MIN_COLD"
      --moe-cpu-timeout-ms "$XEN_MOE_CPU_TIMEOUT_MS")
[[ -n "$XEN_MOE_CPU_CORES" ]] && cmd+=(--moe-cpu-cores "$XEN_MOE_CPU_CORES")
[[ "$XEN_STUB" == 1 ]] && cmd+=(--stub)
cmd+=("$@")

if [[ "${SERVE_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi
[[ -x build/xenithium-server ]] || { echo "start-server: build/xenithium-server missing; run scripts/build-host.sh xenithium-server" >&2; exit 1; }
# SERVE_PLAN=1: the load's memory plan as JSON (VRAM per card, pinned host RAM, KV tokens, hot experts); no GPU, no
# model load, so it runs beside a live server (the launcher's estimate)
if [[ "${SERVE_PLAN:-0}" == 1 ]]; then exec "${cmd[@]}" --plan; fi
# a running server holds both cards, unless both it and this one are TP1 on different cards (one model per card)
while read -r pid args; do
  [[ -z "$pid" ]] && continue
  if [[ "$tp" == 1 && " $args " == *" --tp 1 "* && " $args " != *" --card $card "* ]]; then continue; fi
  echo "start-server: a xenithium-server is already running (pid $pid)$([[ "$tp" == 1 ]] && echo " on card $card or both cards")" >&2
  exit 1
done < <(pgrep -a -x xenithium-serve || true)
if (exec 3<>"/dev/tcp/127.0.0.1/$XEN_PORT") 2>/dev/null; then
  echo "start-server: port $XEN_PORT is in use" >&2
  exit 1
fi
# other containers hold the cards; this project's own (label org.xenithium, docker/Dockerfile) share the host's
# process table, so a server in one was caught by the check above
others=$(docker ps --format '{{.Label "org.xenithium"}}	{{.Names}}' 2>/dev/null | awk -F'\t' '$1 != "1" {print $2}' | tr '\n' ' ')
if [[ "$XEN_STUB" != 1 && -n "$others" ]]; then
  echo "start-server: containers hold the cards ($others); stop them first" >&2
  exit 1
fi
exec "${cmd[@]}"
