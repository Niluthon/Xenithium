# Benchmarking guide

`xenithium-bench` is the engine's benchmark. It is one small program that talks to any server with an
OpenAI-compatible API (`/v1/chat/completions`). It measures this engine, the same engine on another machine, another
engine's server or a hosted API, all in the same way.

This page shows how to use it. [bench.md](bench.md) is the full reference: every option, every scenario parameter,
and how each number is computed.

- [What you get](#what-you-get)
- [Build it](#build-it)
- [Your first run](#your-first-run)
- [How it measures](#how-it-measures)
- [Suites: what to run](#suites-what-to-run)
- [Scenarios](#scenarios)
- [Benchmark another endpoint](#benchmark-another-endpoint)
- [Compare two servers fairly](#compare-two-servers-fairly)
- [From the launcher, or with one script](#from-the-launcher-or-with-one-script)
- [Reading the results](#reading-the-results)
- [Changing a scenario](#changing-a-scenario)
- [When something goes wrong](#when-something-goes-wrong)

---

## What you get

Every run writes one directory, `~/.xenithium/bench/runs/<time>-<name>/`, with three files in it:

| file | what it is |
|---|---|
| `report.html` | One page with the headline numbers, charts, tables and the run's setup. Open it in any browser; it works offline. It hides your home folder and login name ([before sharing](#reading-the-results), set a lab name too). |
| `report.txt` | The same tables as plain text: the tables the console printed while the run went. |
| `results.json` | Every sample of every measurement, the settings and the notes. Other tools read this file. |

What it measures:

- **Speed**: decode tok/s by kind of output, time to first token (TTFT), prefill speed up to 250K tokens, and decode
  speed with a long context.
- **Load**: many users at once, requests arriving at a steady rate, a long prompt arriving while other replies stream.
- **Real applications**: chat users with pauses, questions over retrieved documents, editor autocomplete, agents with
  tool calls.
- **Correctness**: graded answers, finding a fact in a long document, repeatable replies, valid JSON.
- **Power**: on the machine that holds the GPUs it also reads their power draw, so results include tokens per joule.

## Build it

```bash
scripts/build-host.sh xenithium-bench      # builds build/xenithium-bench (seconds; no GPU needed)
```

The program needs no GPU and no Python, and links only the system's C and C++ libraries: copy `build/xenithium-bench`
to another Linux machine of the same or a newer distribution and run it there.

## Your first run

Start a server (see the [launcher guide](launcher.md)), then:

```bash
build/xenithium-bench run --suite quick                  # ~3 min: every headline number once
build/xenithium-bench run                                # ~13 min: the standard suite, the numbers to quote
```

The default address is `http://127.0.0.1:8000/v1`, and the model is the first one the server lists.

Other commands:

```bash
build/xenithium-bench list                       # every suite and scenario, with each parameter's default
build/xenithium-bench report  RUN                # a finished run's tables again (RUN = its directory)
build/xenithium-bench compare RUN_A RUN_B        # B against A, metric by metric
build/xenithium-bench html    RUN                # write the run's HTML page again
build/xenithium-bench help                       # every option
```

## How it measures

An engine with speculative decoding is fast when its draft guesses the next tokens right. How often that happens
depends a lot on the text, on where replies stop, and on how the categories are averaged: the same server can show
180 tok/s on one benchmark and 250 tok/s on another. This tool measures one way, chosen to be what real work looks like:

| | |
|---|---|
| Short tasks (decode, concurrency) | Per kind of output, both kinds of task real work holds: open tasks written from scratch (write a function, an essay, a proof) and tasks on given input (return a diff, fix the given code, turn CSV into JSON, continue a chat) |
| Reply length | Ends where the model ends it |
| Prefix cache | Every request starts with a new line, so no reply comes from the cache |
| Long prompts | This engine's own source code and documentation |
| Decode headline | Two weightings of the same per-category speeds, switchable on the report page: **benchmark** (the default: code 30 %, reasoning 20 %, prose 15 %, JSON 15 %, edits 10 %, summaries 10 %, a plain average, as the usual public benchmarks weigh) and **agentic** (a coding agent's output: 63 % reasoning, 19 % code; the speed that mix of tokens comes out at) |

Some things to know:

- Every run records the method's version. `compare` warns when two runs differ in it: their numbers do not compare.
  Runs made before it had a mode (easy, medium, hard); their reports say so.
- **benchmark or agentic?** Benchmark is the number to put next to other benchmarks'. Agentic is what a coding agent
  gets: it weighs reasoning most and averages by time per token, so slow kinds of output count for the time they take,
  and it reads lower. `--set decode.weights=agentic` makes it the headline; both are in every report either way.
- `--reply-length fixed` makes every reply run to its full budget, past the model's end, so two builds do the same
  work. The text past the end is no user's and drafts unlike real text: it is not the method's number.
- A model that thinks before it answers often spends the whole reply budget thinking: the measured text is then its
  reasoning. `--thinking off` measures the answers alone (where the server has the switch).
- The decode scenario takes 14 samples per category (seven prompts, two passes). A suite's own setting still wins:
  `quick` keeps one pass.

## Suites: what to run

A suite is a ready-made list of scenarios. Choose one with `--suite`:

| suite | time* | what it runs | run it when |
|---|---|---|---|
| `quick` | 3 min | decode, concurrency 1 / 4 / 8, prefill to 32K, decode at 32K | you want a first look |
| `standard` (default) | 13 min | decode, concurrency 1-16, prefill to 128K, decode at 32K and 109K, an agent session, JSON output, answer quality | you want the numbers to quote for a build |
| `long` | 13 min | prefill 32K-250K, decode at depth, the needle, a long agent session | you changed a long-context setting |
| `load` | 12 min | concurrency 1-16, a steady arrival rate, a late long prompt, a burst, mixed depths | you changed scheduling or the slot count |
| `agents` | 25 min | 1, 2 and 4 agents on long documents, a prompt read while agents generate | you serve coding agents |
| `apps` | 7 min | chat users, retrieval questions, autocomplete, tool calls, one long reply | you want real-application numbers |
| `check` | 7 min | graded answers, the needle, repeatable replies, JSON schema, images, tool calls | you changed numerics or a lossy setting |
| `full` | 62 min | every scenario, to the server's full context | before a release |

\* Measured for the dense 27B model on two cards. A slower endpoint takes longer.

To run only some scenarios, list them instead of a suite:

```bash
build/xenithium-bench run --scenario decode,concurrency
build/xenithium-bench run --scenario prefill,needle --set prefill.depths=32000,128000
```

## Scenarios

Each scenario answers one question. They are grouped here by that question; [bench.md](bench.md#scenarios) has the
full details.

**How fast is it for one user?**

| scenario | what it does |
|---|---|
| `decode` | One request at a time over eight kinds of output: chat, code, reasoning, maths, prose, JSON, edits and summaries. Reports decode tok/s, TTFT and tokens per step for each kind, and one headline number. |
| `prefill` | New prompts from 2K to 250K tokens. Reports how fast the prompt is read (tok/s) and the TTFT at each length. |
| `deep-decode` | A long document (32K, 109K, 250K tokens), then the same one again. Reports decode speed with that much context, and how fast a cached prompt starts. |
| `long-reply` | One 8K-token reply, cut into eight stretches. Shows whether the speed drops as the reply grows. |

**How does it hold up under load?**

| scenario | what it does |
|---|---|
| `concurrency` | 1 to 16 users, each sending a new request as soon as the last one ends. Reports total tok/s, tok/s per user and TTFT. |
| `arrivals` | Requests arriving at a steady rate (0.5, 1, 2 per second), whether or not earlier ones are done. Shows the load the server can keep up with. |
| `late-arrival` | A 32K-token prompt arrives while 4 or 8 replies stream. Reports its TTFT and how long the others pause. |
| `burst` | Many questions at once about the same 32K document. Shows whether the shared part is read once or many times. |
| `mixed` | One long-context stream next to three short ones. Reports the speed of each. |

**How does it serve real applications?**

| scenario | what it does |
|---|---|
| `turns` | An agent session: every turn re-sends the whole conversation. Shows how well the prefix cache works. |
| `chat-sessions` | 4, 8 and 16 chat users who pause to read and type. Reports how many users stay within a latency target. |
| `rag` | Questions over eight retrieved passages, 1-8 users at once. Prompt-heavy: this measures prefill under load. |
| `autocomplete` | Developers typing in an editor, cancelling requests as they type. Shows whether cancelled requests stop quickly. |
| `tool-calls` | Tasks that need a tool (read a file, run tests ...). Checks the right tool is called with the right arguments. |
| `long-agents` | 1, 2 and 4 agents each reading a long document, then working on it. The documents are sized to fill the server's memory. |
| `prefill-decode` | A long prompt read while three agents generate. Shows how both sides share the GPUs. |

**Is the output still right?**

| scenario | what it does |
|---|---|
| `quality` | 34 questions with one checkable answer. A healthy model answers nearly all of them; a drop points at the build or a setting. Each wrong answer is listed first in the section, in red: the question, the expected answer and the one given. |
| `needle` | One fact hidden in a long document. Found or not. |
| `repeat` | The same request three times. Checks the replies are identical. |
| `structured` | Replies constrained to a JSON schema. Checks every reply is valid. |
| `vision` | A generated picture and a question about it. Skipped if the server takes no images. |

## Benchmark another endpoint

The tool works with any server that offers `GET /v1/models` and streamed `POST /v1/chat/completions`. Before it starts,
it checks which optional extras the server accepts and only uses those.

**A server on another machine**

```bash
build/xenithium-bench run --url http://192.0.2.10:8000/v1 --suite quick --name lan-box
```

`http://host:port` is enough: the tool adds `/v1`. A URL with its own path keeps it.

**A server that needs an API key**

```bash
export XEN_BENCH_API_KEY=sk-...        # better than --api-key: an argument shows in the process list
build/xenithium-bench run --url https://gateway.example.com/v1 --model my-model --suite quick
```

`$OPENAI_API_KEY` works as well when `XEN_BENCH_API_KEY` is not set.

**A hosted API**

```bash
build/xenithium-bench run --url https://api.example.com/v1 --model some-model --strict --suite quick
```

`--strict` sends standard OpenAI fields only, with nothing extra. Use it for an API that rejects unknown fields.

**A server that does not report its context size**

```bash
build/xenithium-bench run --url http://192.0.2.10:8000/v1 --model my-model --context 131072 --kv-pool 524288
```

Without `--context` the tool tries every prompt length; lengths the server refuses show up as errors.
`--kv-pool` is the number of tokens the server can hold for all requests together. Only `long-agents` needs it.

**What changes without the extras**

| extra | without it |
|---|---|
| `ignore_eos` | Replies may end early, so fixed-length replies are not guaranteed. Rates cover the tokens that arrived. |
| `top_k` | Left out of the sampling. |
| token counts in the stream | Counted as one per streamed chunk and estimated from characters. The report says so. |
| `/metrics` (this engine's counters) | No notes on draft acceptance, prefix-cache hits or preemptions. |

Power is only measured when the URL is this machine (`127.0.0.1`, `localhost`). For a server on this machine reached
through another address, add `--power on`.

## Compare two servers fairly

Run the same command against both, then compare:

```bash
build/xenithium-bench run --url http://192.0.2.10:8000/v1 --suite standard --name server-a
build/xenithium-bench run --url http://192.0.2.11:8000/v1 --suite standard --name server-b

build/xenithium-bench compare ~/.xenithium/bench/runs/*-server-a ~/.xenithium/bench/runs/*-server-b
```

For a fair comparison:

- **Same tool version, same model, same seed.** Requests use seed N + k, so both servers draw the same samples where
  their outputs agree. `compare` warns when the method's version, the reply lengths or the corpora differ.
- **Look at each category, not only the headline.** One weak category can hide inside a good average.
- **Trust a change only when it is marked.** `compare` marks a change `better` or `WORSE` only when it is larger than
  0.5 % and larger than the run's own spread. An unmarked change is noise.
- **A run with failed requests is not a result.** The report names the first error of each scenario.

Add notes so you remember later what each run was:

```bash
build/xenithium-bench run --note gpu=2xB70 --note build=a1b2c3d --note kv=int8
```

## From the launcher, or with one script

**In the launcher** ([launcher.md](launcher.md)), open the Benchmark tab with `3`:

- `r` chooses what to run (Standard, Quick look, Long contexts, Load, Agents, ...) and starts it against the running
  server.
- `Enter` opens a run's report, and `x` stops a running one.

Each run records the server's settings as notes.

**With `scripts/bench.sh`**, one command starts a fresh server, runs the benchmark and stops the server:

```bash
scripts/bench.sh                                        # standard suite on a fresh server from server.env
BENCH_ARGS="--suite quick" scripts/bench.sh quick
URL=http://192.0.2.10:8000/v1 scripts/bench.sh lan      # an endpoint that already runs: nothing is started
```

## Reading the results

| number | what it means |
|---|---|
| **decode tok/s** | Generated tokens per second after the first one. The prompt is not included. |
| **TTFT** | Time to first token: connecting, waiting in the queue and reading the prompt. |
| **prefill tok/s** | Prompt tokens the server had to read (not the cached ones), divided by the TTFT. Missing when the server does not say how many were cached and another client was using its cache at the same time. |
| **tok/update** | Tokens per streamed update. Above 1 means speculative decoding is accepting several tokens per step. |
| **update gap** | Time between two updates: one decode step, as a client sees it. |
| **tok/J** | Tokens per joule of GPU board energy (the same number as tok/s per watt). |

Table cells show the median. Latencies with four or more samples show `median / max`. Yes / no checks show
`yes/total`.

The headline decode number is a mix of the eight categories: by default the benchmark mix, a weighted average led by
code, the figure public benchmarks quote. The agentic mix (the speed of a coding agent's output, 63 % of it reasoning)
is reported beside it; the page's Decode figure switches between the two and its decode section explains both.

**Before sharing a report**, set a lab name so the page does not show your machine's hostname:

```bash
XEN_BENCH_ID=my-lab build/xenithium-bench run --suite quick
```

## Changing a scenario

Every scenario has parameters with defaults. `list` prints them all. Change one with `--set scenario.key=value`:

```bash
build/xenithium-bench run --scenario concurrency --set concurrency.levels=1,8,32,64
build/xenithium-bench run --scenario decode --set decode.passes=5 --set decode.categories=code,json
build/xenithium-bench run --scenario prefill --set prefill.depths=8000,32000,128000
build/xenithium-bench run --scenario decode --set decode.weights=even
```

An unknown scenario, key or value is refused before anything is sent.

Other useful options:

| option | what it does |
|---|---|
| `--name TAG` | The run's directory becomes `<time>-TAG`. |
| `--out DIR` | Write runs somewhere else (or set `$XEN_BENCH_RUNS`). |
| `--temperature 0` | Greedy decoding everywhere (the default is 0.7, as chat clients send). |
| `--thinking on\|off` | Turn the model's thinking on or off, if the server supports the switch. |
| `--reply-length fixed` | Run every reply to its full budget, past the model's end: the same work on every build (not the method's number). |
| `--seed N` | Draw a different, but still repeatable, set of samples. |
| `--timeout SECONDS` | Per request (default 1800). |

## When something goes wrong

| what you see | what to do |
|---|---|
| exit code `2` | A usage or setup problem: wrong option, no server at the URL, a model name the server does not list. The message says which. |
| exit code `1` | Some requests failed. The report names the first error of each scenario. Fix it and run again: this run is not a result. |
| `the endpoint refused a first request` | Usually a wrong `--model` or API key. `curl URL/models` shows the names the server knows. |
| a scenario is "skipped" | The server lacks that feature (images, tool calls) or its memory is too small for the case. The note says why. |
| depths are skipped | They are longer than the server's context. Pass `--context` if the server does not report it. |
| a long context "repeats the text" | The prompt is longer than the built-in text, so the text repeats. Some engines process repeated text faster; the report flags it. |
| token counts are "estimates" | The server sends no usage in the stream. Rates are close but not exact. |

Exit codes in short: `0` = every request answered, `1` = some failed, `2` = usage or setup error.
