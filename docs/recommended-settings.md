# Recommended settings: Qwen3.8-27B Swift-1.5 (GPTQ int4 baked)

These are the settings we found fastest for each kind of use, with the command that starts them. A contributor measured them on two Arc Pro B60s (24 GB each, TP2, EPYC 7532 host); "our build" below is their fork. The settings we tried and rejected are listed at the end, so nobody has to test them again.

For Qwen3.8-Flash-Next on four cards, see [its recommended settings](recommended-settings-qwen38-flash-next.md).

Every switch is described in [server-env](server-env.md) and [trained K8V4 QSA](trained-k8v4-qsa.md). To see which kernel each setting picks, read the [kernel map](kernel-map.md). [Which PR adds each setting](#which-pr-adds-each-setting) names the pull request behind each switch and how it was measured.

## Pick a preset

| you have | start with | what you get |
|---|---|---|
| the checkpoint + the trained QSA bank + the DSpine b16 drafter | `XEN_MODEL_ENV=swift15-27b-int4-baked-qsa scripts/start-server.sh` | the fastest setup we know, below |
| the checkpoint + the GPTQ DFlash 2 drafter only | no preset ships for it yet: the settings in [Without the bank](#without-the-bank-plain-preset) | no sparse attention; the best settings without the bank |

The QSA preset is only used when you name it, because it needs the bank. Started without the bank, it refuses to load, and the checkpoint lookup never picks it on its own. Anything you put in front of the command overrides the preset for that run, for example `XEN_SLOTS=4 XEN_MODEL_ENV=... scripts/start-server.sh`.

## The fastest setup

This is the configuration we run in production. The model paths are generic, the logging switches are removed (see [Diagnostics](#diagnostics-leave-them-off)), and the KV pool is left to the engine.

**Sizing the KV pool for your cards.** `--kv-pool-tokens auto` gives the KV pool what's left of VRAM after the weights, the drafter, the graphs and `--reserve`. On our 2x 24 GB B60s we pin `--kv-pool-tokens 884416 --context 221120` (6 slots sharing the pool), the largest pool that still leaves room for the prefix cache and graphs there. Don't copy those two numbers to other cards: on 32 GB cards (B70) `auto` gives a much larger pool, for example 1,575,936 tokens on 2x B70. Set `--context` only if you want to cap a single request below the model's 262,144.

```sh
XEN_TRAINED_QSA_DELTA_DIR=/path/to/bank XEN_TRAINED_QSA_TILED_SCORE=1 XEN_TRAINED_QSA_POOL_FP16=1 \
XEN_TRAINED_QSA_TREE=1 XEN_TRAINED_QSA_PREFILL_XMX=0 XEN_TRAINED_QSA_PREFILL_MERGED=0 \
XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG=0 XEN_PF_SUBTILE_SKIP=0 \
XEN_QSA_XMX_SCORE=1 XEN_QSA_GATHER=1 XEN_QSA_RESIDENT_SNAPSHOTS=1 XEN_QSA_RESIDENT_FORK_COPY=1 \
XEN_PREFILL_SHARE=0.33 \
xenithium-server --model ~/models/swift15-27b-int4-baked --tp 2 --card 0 --slots 6 \
  --kv int8-v4 --kv-pool-tokens auto --reserve 2.0 --graphs 1 --warmup 1 \
  --drafter 1 --drafter-model ~/models/qwen38-27b-dspine-b16 --drafter-int4 1 --draft-vocab 98304 \
  --seg-rows 16 --step-rows 0 --draft-depths 15,7,4,3 --adaptive-depth 1 --unequal-segments 1 \
  --drafter-tree 7 --drafter-tree-wide 15 --drafter-wide 7 --drafter-tree-seqs 2 \
  --drafter-defer 64 --drafter-cut 0.02 --drafter-mtp-from 0 --mtp-depth 3 --lookup 4 \
  --decode-attn auto --decode-w4 exact --decode-moe w4a16 --gdn-w4 int8 --wire bf16 \
  --prefill-chunk 2048 --prefill-select 256 --prefill-attn auto --prefill-w4 a8 --prefill-a8-body auto \
  --prefill-gemm onednn-all --prefill-gemm-rows 512 --prefill-wire auto --prefill-moe w4a16 \
  --prefill-dense w8a8-row --prefill-merge 16 --prefill-cold-stage 16 --prefill-stall-ms 100 \
  --prefill-part-ms -1 --prefill-short 32 --attn-tail 2 --attn-select 0 \
  --lm-head auto --lm-head-tiles 32 --draft-head auto --w4-items 1024 --select-heads auto \
  --kv-lend on --kv-lend-keep 32768 --kv-lend-park-s 300 --kv-lend-pressure kv --kv-lend-grace-s 60 \
  --prefix-cache 1 --prefix-cache-gib 5 --prefix-mark-min 2048 --prefix-live 0 \
  --yarn-tiers off --sample-threads 8 --max-queue 0 --queue-policy fifo --idle-timeout 3600
```

**Greedy output is not byte-repeatable with trained QSA on.** Its selection follows the speculative step's shape, which the adaptive depth picks from timing. With this preset on 2x B70, a 16K-token prompt sent six times with the prefix cache off gave 1 of 5 identical repeats (5 of 5 with the bank unset, 5 of 5 with `XEN_ADAPTIVE_DEPTH=0`). Leave the bank unset where identical replies matter.

### What it measures, and on which build

The newest numbers come from our fork build `v0.2.0-xq.21`. That build has kernel work that is **not in any PR yet** (see [Fork-only changes](#fork-only-changes-not-in-a-pr-yet)). With this project's master (which carries #3-#9) and this command, expect the `v0.2.0-xq.11` column.

| measured | xq.21 (current) | xq.11 (master with #3-#9) |
|---|---|---|
| single-stream decode, bench easy mix | **208 tok/s**, 4.95 tokens per step | about 140 tok/s |
| decode step time, 8-row / 16-row step | 24.5 / 28.4 ms | - |
| answer quality (bench `check`) | 33/34 | 97.1%; needles found at 32k and 128k |
| total decode, 1 / 2 / 4 / 6 streams | not re-measured | 140 / 211 / 316 / 379 tok/s |
| 4 users at ~90k context each | not re-measured | 255-261 tok/s total |
| 1 user, decode deep in a document | not re-measured | 107 tok/s at 32k, 108 at 109k |
| single-stream decode by category | not re-measured | chat 104, code 139, reasoning 133, math 176, JSON 232 tok/s |
| prompt reading (prefill) | not re-measured | 3,315 tok/s shallow, about 2,700 at 90k and deeper |

## Which PR adds each setting

Every setting below is on master; the table names the pull request that added it and how that request was measured.

| PR | adds | how it was measured |
|---|---|---|
| #3 | trained K8V4 QSA: `XEN_TRAINED_QSA_DELTA_DIR` (the bank), `XEN_TRAINED_QSA_TILED_SCORE`, `XEN_TRAINED_QSA_POOL_FP16`, `XEN_TRAINED_QSA_PREFILL_XMX` | 2x B60, DSpine b16. Three frozen 8,007-token cold / repeat pairs: identical completions, decode 114.35 → 114.38 tok/s cold, 115.27 → 115.22 on prefix hits |
| #4 | draft trees on `--kv int8-v4` (`--drafter-tree 7` with int8-v4 KV) | 2x B60, one stream, DSpine b16, 8,007 input tokens, 400 greedy outputs, 16,384-token pool. Only `--drafter-tree` changed (0 vs 7): identical text |
| #5 | `XEN_DRAFT_GRAPHS` (cap on the drafter's pass graphs, default 96) | 2x B60, DSpine b16, 6 slots, two `xenithium-bench --suite agents` runs on one boot: VRAM flat once the cap is reached, 0 failed. A reviewer's 2x B70 soak at 16 slots: 0 server errors |
| #6 | DSpine b16 against this checkpoint's GPTQ int4 `lm_head` (`--drafter-model` = DSpine with this model) | 2x B60, TP2, 6 slots, the command above |
| #7 | `--mtp-depth` up to 6 (MTP layer only, no block drafter; on master it sets a lone sequence's depth) | a reviewer's 2x B70: depth 6 gives 3.04 vs 2.65 tokens per step, 22.0 vs 18.1 ms per step, agent mix 137 vs 146 tok/s. The default stays 3 |
| #8 | `XEN_QSA_GATHER`, `XEN_QSA_XMX_SCORE`, `XEN_TRAINED_QSA_TREE`, `XEN_QSA_RESIDENT_SNAPSHOTS`, `XEN_QSA_RESIDENT_FORK_COPY`, `XEN_PREFILL_SHARE`, `XEN_TRAINED_QSA_PREFILL_MERGED`, `XEN_TRAINED_QSA_PREFILL_4SLOT_SINGLE_SEG`, `XEN_PF_SUBTILE_SKIP`, `XEN_REQ_LOG`, `--prefill-select` on int8-v4 KV, and the QSA preset | 2x B60, TP2, DSpine b16, K8V4 KV, 6 slots, the xq.11 column above |
| #9 | `XEN_STEP_GRAPHS`, `XEN_STEP_GRAPHS_MAX`, `XEN_GRAPH_ROOM_MIB` (decode step graphs, opt-in) | 2x B60, TP2, 6 slots, the server held to a quarter of one core: 98 / 177 / 251 → 165 / 283 / 378 tok/s at 1 / 2 / 4 streams; the same as without on a fast host |
| #12 | the DSpine b16 checkpoint's current architecture name (`DSpineDraftModel`) | host tests |
| #13 | none (fixes the `bench` host test as root, for `docker build`) | host tests |

### Fork-only changes (not in a PR yet)

These are in our build between xq.12 and xq.21 and need no settings: they are on by default there.
- **xq.12-xq.20:** faster drafter pass (2.60 → 2.29 ms at 1 user), GDN tree core and 16-row step work.
- **xq.21:** long decode chains attend on the prefill fa kernel (`XEN_ATTN_DEC_FA`, default on there): 16-row chain steps 29.8 → 28.4 ms at 2K context, more at longer context.

They will come upstream as their own PRs, each with its settings and measurements.

## By use case

All of these start from the fastest setup; each row lists only what changes. The numbers are from xq.11.

| use | change | why |
|---|---|---|
| **One person chatting or coding** | nothing | Draft trees run through 2 streams (`--drafter-tree-seqs 2`), and decode stays flat out to 109k. |
| **Several coding agents or long chats at once** | nothing; keep `--slots 6` and the prefix cache | The prefix cache is per slot. With fewer slots, a fifth conversation evicts a parked 90k chat, and that chat's next turn re-reads its whole prompt (67-105 s). 6 slots give 104 / 149 / 178 / 213 / 226 / 240 tok/s total at 1-6 users with 80k context each. 7 or more slots need a smaller pool. |
| **Users waiting on replies while someone pastes a huge document** | `XEN_PREFILL_SHARE=0.33` (already in the command) | While others decode, prompt reading is limited to about a third of the loop's time. A decoder next to a 112k prompt went from 47-51 to 75 tok/s, and the big prompt's time to first token didn't change. |
| **Long documents / RAG (100k-220k prompts)** | nothing; keep `--prefill-select 256` | Reading a 239k prompt goes from 1,521 to 2,424 tok/s, with 8/8 needles at 32k and 121k. Teacher-forced NLL is within noise of exact attention. |
| **Sampling (temperature 1) instead of greedy** | nothing | Expect about 25% less decode speed at short context than greedy benchmarks show (drafts are accepted less often), and no loss at long context. |
| **A weaker CPU, PCIe 3, or an OCuLink / eGPU link** | add `XEN_STEP_GRAPHS=1` (#9, opt-in) | Each decode step becomes a single graph replay, so the host's ~2,000 kernel launches per step stop being the bottleneck. With the server held to a quarter of one core: 98 / 177 / 251 → 165 / 283 / 378 tok/s at 1 / 2 / 4 streams. On a fast host it makes no difference. Keep `--prefill-wire auto` (bf16): fp32 sends 1.34 GB more per chunk, and about 93% of that link time shows up in prompt reading. |
| **Tight VRAM** | lower `--kv-pool-tokens`; keep `--reserve` at least 1.5 | The drafter's pass graphs take up to about 1 GiB per card (`XEN_DRAFT_GRAPHS`, #5). With the old 0.25 GiB reserve, the server runs out of device resources once 16 streams are active. |

## Without the bank (plain preset)

No plain preset ships yet. The settings we use without the bank:
- the GPTQ DFlash 2 drafter with 8-row segments, and draft trees through 2 streams;
- prompt lookup;
- int8 KV;
- prefill chunks of 2,048, and on two cards oneDNN for every prompt matrix from 512 rows;
- a 1.5 GiB reserve.

Compared with the bare defaults on the same build (bench `standard`):

| | bare defaults | preset |
|---|---|---|
| 1 stream | 110 tok/s | 138 tok/s |
| 2 streams, total | 190 | 217 |
| 4 streams, total | 325 | 346 |
| 8 streams, total | 315 | 454 |
| single stream: math / JSON / edit | 134 / 147 / 128 | 200 / 207 / 174 |

## Tried and rejected

| setting | result | verdict |
|---|---|---|
| `--draft-vocab 65536` | the same speed as 98,304 | drop |
| `--draft-depths 15,10,6,4` (deeper shared depth) | 5-15% slower at 4 short streams | keep `15,7,4,3` |
| depth caps at 3-4 streams | no change | keep `15,7,4,3` |
| `--drafter-wide` 0 or 10 | no change on easy single-stream decode | keep 7 |
| `--wire int8` (8-bit tensor-parallel partials) | 0.1-0.16 ms per step | keep `bf16` |
| `--decode-attn exact` | 90k with 1-4 users: 84 / 122 / 126 / 164 tok/s total, against 100 / 148 / 157 / 200 for `int8-fast` | keep `auto` (int8-fast) |
| `--prefill-select 128` | about 10% faster prefill, but fresh NLL above dense at every size (+0.9 to +3.7%) | keep 256 |
| `--prefill-select 0` with trained XMX sparse prefill (`XEN_TRAINED_QSA_PREFILL_XMX=1`) | fresh NLL +6 to +12% over dense; also blocks 5-6 slots | off; use `--prefill-select 256` |
| `--prefill-chunk 4096` | +2.6% prefill, but needs about 0.43 GB per card, so the pool shrinks 4.6% | keep 2048 |
| `--prefill-wire fp32` | 2,420 tok/s, against 2,706 for bf16 | keep `auto` (bf16) |
| `--prefill-dense ...` | qwen38-only; no effect on this model (the command passes it anyway) | harmless |
| 16 slots with the default 0.25 GiB reserve | ran out of device resources at 16 streams | reserve 1.5 or more |

## Diagnostics: leave them off

`XEN_QSA_TRIAL_ROUTE_LOG`, `XEN_PF_LOG`, `XEN_REQ_LOG`, `XEN_QSA_SCORE_CHECK` and `XEN_QSA_PACK_CHECK` only add logging or reference checks. `XEN_QSA_TRIAL_ROUTE_LOG` alone writes one line per stream per step, about 50 MB a day under agent load.
