# Performance: hard mode

The strict measurement: `xenithium-bench` in its former `hard` mode (the tool has since merged its modes into one
method: [reference](bench.md#the-method); these rows predate it and are rerun with it). The [README](../README.md#performance) shows the easy-mode numbers, measured the way the
usual public benchmarks do; this page keeps the hard-mode table, the numbers to compare builds with.

> **These rows are a strict measurement of the machine, not the easiest one.** The measured text is this project's own:
> **real code** and **real documentation**, repeated nothing. That matters most for Decode, where a drafter earns its
> speed by guessing right, and real text is guessed far less well than synthetic, repetitive or empty-prompt
> benchmarks: the accepted tokens per step of the easy ones would let any engine post a higher decode number on the
> same hardware. Compare rows against each other, and against another setup only corpus for corpus.

## Setup

- A live server started from each preset (`models/presets/`), on one machine: 2x Arc Pro B70 for the two-card and
  one-card rows, 4x B70 for the four-card one.
- **The GPUs are power-capped to 220 W per card** (every report's power section records the cap).
- Every row ran with its speculative drafter on; the reports show the accepted tokens per step.
- All runs are from 2026-11-20 on the release build; [benchmark/README.md](benchmark/README.md) names the run
  directory and the settings of every linked report.

## What the columns mean

| Column | Meaning |
| --- | --- |
| Drafter | The preset's speculative drafter: DFlash 2 (a block drafter, up to 7 guesses per step) or DSpine b16 (16-token blocks, up to 15 guesses per step after a good one). |
| Decode, 1 stream | The headline: the agent mix of the sampled scenarios (harmonic mean over chat, code, reasoning, maths, prose, JSON, edit and summary). |
| All streams 1 / 4 / 8 | The closed-loop concurrency sweep: tokens per second over every stream together. |
| Prefill | The prompt-read rate of real code text at the depth named, with the prefix cache defeated. |
| Decode at depth | The same document sent once, then decoded with that context. |
| TTFT at 8K | Time to the first token of the 8K prompt. |

A cell shows `-` where the endpoint's context cannot hold that depth. Each row links two self-contained HTML reports:
the name is the quick run (decode, concurrency, prefill to 32K, decode at 32K); the long-context link is the same
preset's run of the 64K / 128K depths.

## Results

| Server (preset) | Drafter | Decode, 1 stream (tok/s) | All streams 1 / 4 / 8 (tok/s) | Prefill 2K, 8K, 32K (tok/s) | Prefill 64K / 128K (tok/s) | Decode at 32K / 64K / 128K (tok/s) | TTFT at 8K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next, four cards](benchmark/qwen38-flash-next-4-card.html) · [long-context](benchmark/qwen38-flash-next-4-card-long-context.html) | DFlash 2 | 281 | 264/532/733 | 5,916, 7,936, 7,897 | 7,520 / 7,451 | 208 / 203 / 211 | 1.0 s |
| [Qwen3.8-Flash-Next, two cards](benchmark/qwen38-flash-next-2-card.html) · [long-context](benchmark/qwen38-flash-next-2-card-long-context.html) | DFlash 2 | 130 | 132/248/299 | 2,610, 2,994, 2,835 | 2,686 / 2,447 | 102 / 99 / 103 | 2.7 s |
| [Qwen3.8-Flash-Next, one card](benchmark/qwen38-flash-next-1-card.html) · [long-context](benchmark/qwen38-flash-next-1-card-long-context.html) | DFlash 2 | 59 | 54/58/65 | 493, 631, 605 | 577 / 545 | 48 / 46 / 51 | 12.7 s |
| [Qwen3.8-27B Swift-1.5, two cards](benchmark/qwen38-27b-swift15-2-card.html) · [long-context](benchmark/qwen38-27b-swift15-2-card-long-context.html) | DSpine b16 | 187 | 180/433/605 | 4,162, 4,581, 4,103 | 3,661 / 2,893 | 165 / 141 / 130 | 1.8 s |
| [Qwen3.8-27B Swift-1.5, one card](benchmark/qwen38-27b-swift15-1-card.html) · [long-context](benchmark/qwen38-27b-swift15-1-card-long-context.html) | DSpine b16 | 108 | 103/256/353 | 2,456, 2,512, 2,184 | 1,909 / - | 87 / 75 / - | 3.2 s |
| [Qwen3.8-27B AutoRound, two cards](benchmark/qwen38-27b-autoround-2-card.html) · [long-context](benchmark/qwen38-27b-autoround-2-card-long-context.html) | DSpine b16 | 173 | 164/418/601 | 4,441, 4,627, 4,049 | 3,512 / 2,706 | 146 / 130 / 123 | 1.7 s |
| [Qwen3.8-27B AutoRound, one card](benchmark/qwen38-27b-autoround-1-card.html) · [long-context](benchmark/qwen38-27b-autoround-1-card-long-context.html) | DSpine b16 | 102 | 98/259/367 | 2,479, 2,521, 2,208 | 1,920 / - | 85 / 80 / - | 3.2 s |
| [Recursive Flint INT4, two cards](benchmark/flint-int4-2-card.html) · [long-context](benchmark/flint-int4-2-card-long-context.html) | DFlash 2 | 422 | 406/670/831 | 12,498, 11,745, 9,878 | 7,915 / 5,470 | 372 / 259 / 237 | 0.7 s |
| [Recursive Flint INT4, one card](benchmark/flint-int4-1-card.html) · [long-context](benchmark/flint-int4-1-card-long-context.html) | DFlash 2 | 278 | 273/445/536 | 8,166, 7,499, 5,859 | 4,532 / 2,963 | 201 / 175 / 140 | 1.1 s |

Every row is one `xenithium-bench run --mode hard` of a preset; the same command on your cards measures yours. The
four-card setup of Qwen3.8-Flash-Next is the fastest; two cards are the default of its preset. Flint's one-card row
uses `models/presets/flint-int4.tp1.env` (8 slots, a 262,144-token max context: the whole model is on one card).
