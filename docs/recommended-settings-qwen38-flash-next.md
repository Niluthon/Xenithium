# Recommended settings: Qwen3.8-Flash-Next (W4A16 AutoRound, four cards)

These are the fastest settings we measured for Qwen3.8-Flash-Next (`qwen38-w4a16-intel`, arch qwen38) on four Arc Pro
B70s (32 GB each, every card capped at 220 W with `scripts/gpu-power.sh`), with the command that starts them. The
settings we tried and rejected are listed at the end, so nobody has to test them again.

Every switch is described in [server-env](server-env.md); the launcher (`scripts/launch.sh`, [launcher](launcher.md))
shows the same switches with their explanations.

## Pick a preset

| you have | start with | what you get |
|---|---|---|
| the checkpoint + the DFlash 2 drafter, four cards | `XEN_MODEL_ENV=qwen38-w4a16-intel scripts/start-server.sh` | the fastest setup below |
| the checkpoint only, four cards | the same, with `XEN_DRAFTER_MODEL=` in front | no speculative decoding; or set `XEN_MTP` to an MTP drafter checkpoint |
| one or two cards | `XEN_Q38_TP=2` (or `1`) in front | the `.tp2.env` / `.tp1.env` setups: experts that do not fit in VRAM are read from host memory |

The preset is two files: `models/presets/qwen38-w4a16-intel.env` (the model: folder, served name, `XEN_Q38_TP=4`) and
`models/presets/qwen38-w4a16-intel.tp4.env` (the four-card setup). Your own copies in `models/` (saved by the launcher,
or copied by hand) replace them and are never touched by an update. Anything you put in front of the command overrides
both for that run, for example `XEN_SLOTS=8 XEN_MODEL_ENV=qwen38-w4a16-intel scripts/start-server.sh`.

Before the first start: build the server (`scripts/build-host.sh xenithium-server`) and put the checkpoint in
`~/models/qwen38-w4a16-intel` and the DFlash 2 drafter in `~/models/qwen38-flash-next-dflash2`. Nothing else is needed:
the hot-expert ranking is recorded by the server itself.

## The fastest setup

What the preset sets (`models/presets/qwen38-w4a16-intel.tp4.env`):

```sh
XEN_SLOTS=16                   # 16 conversations decode at once
XEN_CONTEXT=1048576            # one request may grow to 1M tokens (YaRN per request past 262,144)
XEN_KV_SPLIT=tokens            # the KV cache split by tokens: ~1.75x the KV pool, faster long prompts
XEN_DRAFTER_MODEL=~/models/qwen38-flash-next-dflash2   # the DFlash 2 block drafter
XEN_PREFILL_CHUNK=4096         # prompt tokens per prefill call (the model's default, written out)
XEN_PREFIX_MARK_MIN=2048       # chat prompts from 2,048 tokens are split where the reply starts
XEN_PREFIX_CACHE_GIB=24        # host RAM for remembered conversations
XEN_SAMPLE_THREADS=8           # CPU threads that pick the next tokens
XEN_MOE_CPU=off                # every expert fits in VRAM on four cards: no CPU expert path
```

The same as one command (only the switches that matter here; `SERVE_DRY_RUN=1` in front of `scripts/start-server.sh`
prints the whole command with every default):

```sh
xenithium-server --model ~/models/qwen38-w4a16-intel --served-model-name qwen38-w4a16-intel \
  --tp 4 --slots 16 --context 1048576 --yarn-tiers on \
  --kv int8 --kv-split tokens --kv-pool-tokens auto --reserve 0.25 --kv-lend on \
  --drafter 1 --drafter-model ~/models/qwen38-flash-next-dflash2 --drafter-tree 7 --drafter-tree-seqs 2 \
  --step-rows 0 --adaptive-depth 1 --unequal-segments 1 --lookup 4 \
  --graphs 1 --wire bf16 --decode-moe w4a16 --gdn-w4 int8 \
  --prefill-chunk 4096 --prefill-moe auto --prefill-dense w8a8-row --prefill-wire auto --prefill-gemm onednn \
  --prefill-stall-ms 100 --prefill-graphs 0 \
  --prefix-cache 1 --prefix-cache-gib 24 --prefix-mark-min 2048 --prefix-live 0 \
  --moe-cpu off --sample-threads 8 --host 0.0.0.0 --port 8000
```

### What each setting does here, and why this value

| setting | value | why |
|---|---|---|
| `XEN_Q38_TP` (`--tp`) | `4` | Every matrix and every expert in quarters on cards 0-3. Every expert of every layer then fits in VRAM, so no expert is read over PCIe. |
| `XEN_SLOTS` | `16` | Total speed grows up to 16 streams (946-979 tok/s in all at 16, against 763-787 at 8). Each slot costs only its share of the KV pool while it holds a conversation. |
| `XEN_CONTEXT` | `1048576` | A limit, not a reservation: all slots share one KV pool. Past the trained 262,144 tokens YaRN turns on per request (`XEN_YARN_TIERS` on), so chat-sized requests keep the trained numerics and pay nothing for the long limit. |
| `XEN_KV` | `int8` (default) | Half the memory of bf16; the quality change is within run-to-run noise. See `int8-v4` under [Tried and rejected](#tried-and-rejected). |
| `XEN_KV_SPLIT` | `tokens` | The two cards that share a KV head store every other token each: a quarter of the KV per card instead of half. KV pool 1.60M -> 2.83M tokens at the same VRAM (+77 %), long prompts read faster (7,569-8,647 against 7,345-7,902 tok/s by head), and with 17 conversations of 128K the freed VRAM keeps experts hot: decode steps 10.5-11 % shorter. A small numerics change (KL to the reference 0.0449 -> 0.0444). |
| `XEN_KV_POOL_TOKENS` | `auto` (default) | The pool takes the VRAM left after the weights, the drafter, the graphs and `XEN_RESERVE`: about 2.7M tokens with this setup. |
| `XEN_DRAFTER_MODEL` | DFlash 2 | A block drafter: up to 7 guessed tokens a step, a tree of alternatives at 1-2 streams (`XEN_DRAFTER_TREE` 7, `XEN_DRAFTER_TREE_SEQS` 2). About 3.8 tokens per step at one stream, 3 at 4-16 streams. |
| `XEN_STEP_ROWS` | `0` (default: 64 on four cards) | A decode step checks up to 64 rows: 16 rows up to 5 streams, 32 at 6-8, 64 from 9. Against 16-row steps: 16 streams +41..44 % decode summed (+27..31 % in all), 8 streams +12..16 % in all. Costs 0.42 GiB per card of scratch (KV pool -3.4 %); `XEN_STEP_ROWS=32` gives it back. |
| `XEN_PREFILL_CHUNK` | `4096` | The model's default. Larger calls read long prompts faster; the scratch for 4,096 rows is already counted in the pool above. |
| `XEN_PREFILL_MOE` | `auto` (default: `w4a8` on four cards) | The experts read the prompt with int8 activations: +3.1..3.4 % prompt reading over `w4a16`, and closer to the reference model (KL 0.0475 -> 0.0384). On four cards each row's owner card routes it and quantises it once, and shares the int8 row and its routing instead of the bf16 row: +4.9..5.1 % more. |
| `XEN_PREFILL_DENSE` | `w8a8-row` (default) | The dense layers read the prompt with int8 activations, one scale per row. On four cards the linear-attention (GDN) layers' input is quantised once by each row's owner card, which also computes its a / b projections for all four cards, and shares those instead of the bf16 row: +2.0 % (32K) / +2.4 % (128K), the same output. |
| `XEN_PREFILL_WIRE` | `auto` (default: fp32 on qwen38) | Exact sums between the cards. `bf16` halves that traffic but gains only 0.1-0.2 % now (see below). |
| `XEN_PREFILL_STALL_MS` | `100` (default) | While others generate, a new long prompt is read in ~100 ms pieces, so running streams pause at most that long per token. `0` gives the fastest first token of the new prompt and the longest pauses. |
| `XEN_KV_LEND` | `on` (default) | Puts the KV pool's unused pages to work and gives them back when a conversation grows into them. |
| `XEN_PREFIX_CACHE` / `_GIB` / `XEN_PREFIX_MARK_MIN` | `1` / `24` / `2048` | Finished conversations are remembered in host RAM, so an agent's or a chat's next turn reads only what is new (turns 2+ answered in 205-295 ms in the agent test). 24 GiB holds about 150-400 snapshots; lower it on a host with less than ~48 GB of RAM. Prompts from 2,048 tokens are marked where the reply starts, so the next turn resumes exactly there. |
| `XEN_SAMPLE_THREADS` | `8` | Token picking for 16 streams on 8 CPU threads instead of 4. |
| `XEN_MOE_CPU` | `off` | Nothing to do on four cards: no expert is cold. |
| `XEN_GRAPHS` | `1` (default) | Decode steps replay recorded GPU graphs. Off only for debugging. |

### What it measures

Four B70s at 220 W, this setup (16 slots, int8 KV split by tokens, DFlash 2, 64-row steps; measured with a 524,288-token
max context), the server's standard benchmark suite:

| streams | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| decode tok/s per stream | 344-352 | 251-262 | 168-170 | 110-113 | 69-71 |
| total tok/s (decode + prompts) | 307-308 | 433-449 | 575-598 | 763-787 | 946-979 |
| tokens per decode step | 3.8-3.9 | 3.8-3.9 | 3.0 | 2.9-3.0 | 3.0-3.1 |

Answer quality (bench `check`): 34 / 34. Time to first token on a cold prompt: 3.6 s at 32K tokens, 12.4 s at 109K;
the next turn of a cached conversation 205-295 ms.

Prompt reading, one prompt at a time: 2K / 8K / 32K / 64K / 128K tokens at 7,643 / 8,463 / 8,628 / 8,648 / 8,490
tok/s in the server suite, measured before the latest prefill work. The engine's own measure (one 32K and one 128K
prompt, these numerics) read 9,080 and 8,863 tok/s on that build and reads 10,290 and 10,170 on the current one:
+13 % / +15 %.

## By use case

All of these start from the fastest setup; each row lists only what changes.

| use | change | why |
|---|---|---|
| **One person chatting or coding** | nothing | One stream runs draft trees and decodes 344-352 tok/s on the benchmark mix. |
| **Many agents or long chats at once** | nothing; keep 16 slots and the prefix cache | 16 streams decode 69-71 tok/s each, 1,100+ in all. A slot is only a place in the step; the KV pool is shared, so long and short conversations mix. |
| **Users waiting on replies while someone pastes a huge document** | add `XEN_PREFILL_SHARE=0.33` | While others decode, prompt reading gets about a third of the loop's time, so their tokens keep coming at a steadier rate. Not measured on this model; on the 27B a decoder next to a 112K prompt went from 47-51 to 75 tok/s. |
| **The fastest first token for a single long prompt** | `XEN_PREFILL_STALL_MS=0` | Whole 4,096-token chunks: no splitting of the prompt's calls. Running streams pause longer while it is read. |
| **More KV than ~2.7M tokens** | `XEN_KV=int8-v4`, or `XEN_STEP_ROWS=32` | int8-v4 stores V in 4 bits: about 1.3x the tokens, a small quality cost (KL to the reference 0.0390 -> 0.0456 at 548 tokens). Its decode speed on four cards is not measured. 32-row steps give back 0.42 GiB per card, but 16-stream decode drops (below). |
| **Images in requests** | `XEN_VISION=1` | Loads the image encoder: 0.7 GiB per card at the default image size, taken from the hot experts. Some experts then live in host memory, so decode and prompts get slower. |
| **Less host RAM** | `XEN_PREFIX_CACHE_GIB=8` (the default) | Fewer remembered conversations; each snapshot is roughly 60-160 MiB. |

## Tried and rejected

| setting | result | verdict |
|---|---|---|
| `XEN_KV_SPLIT=heads` (the engine default) | KV pool 1.60M instead of 2.83M tokens; long prompts read at 7,345-7,902 against 7,569-8,647 tok/s; at 17 x 128K, decode steps 10.5-11 % longer | keep `tokens` |
| `XEN_STEP_ROWS=16` | 16 streams 788 against 1,104-1,138 tok/s of decode in all | keep `0` (64 rows) |
| `XEN_STEP_ROWS=32` | 16 streams 949-958 against 1,104-1,107 tok/s of decode in all | keep `0`, unless the pool is short |
| `XEN_PREFILL_WIRE=bf16` | +0.1..0.2 % prompt reading on the current build (32K 10,359 against 10,343 tok/s, 128K 10,231 against 10,221), and the sums are no longer exact | keep `auto` (fp32) |
| `XEN_PREFILL_MOE=w4a16` / `w4a16-fast` | w4a16 3.1-3.4 % slower; w4a16-fast is 2.2 % faster than w4a16, still slower than w4a8 on four cards, and changes the output too | keep `auto` (w4a8) |
| `XEN_PREFILL_GRAPHS=1` | 1-2 ms saved per short prompt call (under 128 tokens) at four cards, but each new prompt length costs 10-60 ms to record; it pays only when the same lengths repeat (benchmarks) | keep off for real traffic |
| `XEN_MTP` (the MTP drafter) instead of DFlash 2 | the same at 16 streams (68.8-71.2 against 69.0-69.2 tok/s each); not measured at fewer streams | keep DFlash 2 |
| `XEN_PREFIX_LIVE=1` | faster first token on the next turn, but late tokens may differ from a cache-free run | off where identical replies matter; on if only speed counts |

## Diagnostics: leave them off

`XEN_PF_PROFILE`, `XEN_DEC_PROFILE`, `XEN_PF_LOG` and `XEN_PF_DENSE_ONEDNN_LOG` only add timing or logging, and the
profiles slow the server down: never quote numbers measured with them on. `XEN_REQ_LOG` (one line per finished
request) costs nothing measurable.
