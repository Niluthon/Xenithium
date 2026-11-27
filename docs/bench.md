# xenithium-bench

New to the tool? Start with the [benchmarking guide](bench-guide.md): modes, suites, scenarios and how to bench another
endpoint, step by step. This page is the full reference.

The engine's benchmark. One host binary (`build/xenithium-bench`, C++, no GPU and no engine code) that talks to an
OpenAI-compatible endpoint over `/v1/chat/completions` with streaming, so it measures this server, a server on
another machine, or any other engine or hosted API the same way.

It measures the cases the engine is developed against: decode by kind of output, many requests at once (in a closed
loop and at an arrival rate), prompt processing to 250K tokens, decode *at* a long context, an agent session on the
prefix cache, a late long prompt beside running streams, a burst on one document, mixed depths, image input, and
whether the output is still right: graded answers, the long-context needle, reply repeatability and schema-constrained
output. On the machine that holds the cards it also samples their board power and clock, so a run reports tokens per
joule. Every run leaves a `results.json` with every sample, a text report and a one-file HTML page with charts.

```
scripts/build-host.sh xenithium-bench

build/xenithium-bench run                                  # standard suite, http://127.0.0.1:8000/v1
build/xenithium-bench run --url http://192.0.2.10:8000/v1 --suite quick --name lan
build/xenithium-bench run --url https://api.example.com/v1 --model some-model --api-key KEY --strict
build/xenithium-bench run --scenario prefill,needle --set prefill.depths=32000,128000 --set needle.position_pct=80
build/xenithium-bench list                                 # suites, scenarios, every parameter with its default
build/xenithium-bench report  ~/.xenithium/bench/runs/20261116-200152-smoke
build/xenithium-bench compare RUN_A RUN_B                  # B against A, metric by metric
build/xenithium-bench html    RUN [FILE]                   # the run's HTML page again (after an update of the tool)
```

`scripts/bench.sh [name]` wraps a run: with no `URL` it checks that the cards are free, starts this tree's server from
`server.env`, records every effective server flag and the host as notes, samples board power, runs the benchmark and
stops the server; with `URL=...` it only runs against that endpoint. The launcher's Benchmark tab (`3`, then `r`) runs the
same binary against the server it started ([launcher.md](launcher.md)).

## Options of `run`

| option | default | meaning |
|---|---|---|
| `--url URL` | `http://127.0.0.1:8000/v1` | The API base. `http://host:port` gets `/v1`; a URL with a path keeps it (`https://host/openai/v1`). `https` is supported. |
| `--model ID` | the endpoint's first model | The model name sent with every request. |
| `--api-key KEY` | `$XEN_BENCH_API_KEY`, else `$OPENAI_API_KEY` | Sent as `Authorization: Bearer KEY`. Prefer the variable: an argument is visible in the process list. The run file never holds the key: a key from the variables is not recorded at all, one given as `--api-key` is recorded as `(hidden)` and left out of the page's "Run this yourself" command. Notes get the server's rules (`util/secrets.hpp`): a key, token or URL password in a `server_cmd` note or any `--note` is recorded as `(hidden)`, and older runs are hidden the same way on their page. |
| `--suite ID` | `standard` | The scenarios to run and their parameters (below). |
| `--scenario a,b` | | Run these scenarios with their defaults instead of a suite. |
| `--set scenario.key=value` | | Change one parameter of one scenario of the run. Repeatable. An unknown scenario or key is refused before anything is sent. |
| `--name TAG` | the suite's name | The run directory is `<time>-<TAG>`. |
| `XEN_BENCH_ID` | the hostname | The machine name the HTML report prints ("measured from"). Set it to a lab name (`XEN_BENCH_ID=b70-rig`) before copying a report into the repo: the hostname is your machine's identity on your network, and a published report should not carry it. |
| `--note KEY=VALUE` | | Recorded in the run file (a build id, a setting). Repeatable. |
| `--out DIR` | `$XEN_BENCH_RUNS`, else `~/.xenithium/bench/runs` | Where the run directory is created. |
| `--seed N` | `1` | Request k of the run samples with seed N + k, so two builds draw the same tokens where their logits agree. |
| `--temperature` `--top-p` `--top-k` | `0.7` `0.95` `20` | Sampling of the sampled scenarios. `--temperature 0` makes every scenario greedy. |
| `--thinking default\|on\|off` | `default` | The model's thinking switch (`chat_template_kwargs.enable_thinking`), where the endpoint has one. |
| `--mode easy\|medium\|hard` | `hard` | How easy the measured text is for a draft model, and so how high a drafting engine's decode rate reads: `hard` gives the published numbers, `easy` measures the way the usual public benchmarks do. A mode sets the defaults of `--corpus`, `--reply-length` and some scenario parameters; those options still override it. Recorded in the run file; `compare` warns when two runs differ in it. See [Modes](#modes). |
| `--reply-length fixed\|natural` | the mode's: `fixed` for hard and medium, `natural` for easy | The sampled scenarios' replies. `fixed` sends `ignore_eos`: every reply runs to its `max_tokens`, so two builds are timed over the same number of tokens. `natural` lets the model end its reply, as a chat client sees it (rates then cover only the tokens that came). Greedy checks stay fixed. Recorded in the run file; `compare` warns when two runs differ in it. Read below why the two give different decode rates. |
| `--context TOKENS` | the endpoint's `max_model_len` | The longest prompt to send. Depths past it are skipped and named in the report. An endpoint that does not report its context needs this, or every depth is tried. |
| `--kv-pool TOKENS` | the endpoint's `kv_pool_tokens` | Tokens of context the endpoint holds for all requests at once. `long-agents` sizes its documents from it; this server reports it in `/v1/models`. |
| `--gpu NAME` | the endpoint's `devices` | A card the endpoint runs on; give it once per card. Shown in the hardware row of the HTML report's facts, the "runs on" line of the text report, and recorded in the run file. This server lists its cards (name, memory as the driver reports it and the nominal GB from its card table, PCI address) in `/v1/models`; the report shows the nominal size (a B65: 32 GB). Other engines need this flag. |
| `--tp N` | the endpoint's `tensor_parallel` (with `--gpu`: the number of cards) | The endpoint's tensor-parallel width, shown in the parallel row of the HTML report's facts. This server reports it in `/v1/models`. |
| `--timeout SECONDS` | `1800` | Per request. |
| `--corpus hard\|medium\|DIR` | the mode's: `hard` for hard, `medium` for medium and easy | The text long prompts are cut from: `hard` = this engine's own code and documentation, `medium` = conventional template-made code and prose, `DIR` = another corpus (`DIR/<kind>/<files>`). The set, its version and hash are recorded; runs on different corpora do not compare. |
| `--power auto\|on\|off` | `auto` | Sample the GPUs' board power and clock while the run goes (below). `auto` = on when the URL is this machine's loopback address; `on` for a server on this machine reached by another address. |
| `--strict` | off | Send OpenAI fields only: no probing for the extensions below. |

Exit code: 0 = every request was answered, 1 = some failed (the report names the first error per scenario), 2 = a usage
or setup error (no endpoint, unknown suite).

## Other endpoints

The tool needs only `GET /v1/models` and streamed `POST /v1/chat/completions`. Before the run it sends a few small
requests to find out which of these the endpoint takes, uses the ones it does and prints the result:

| extension | used for | without it |
|---|---|---|
| `ignore_eos` | replies of an exact length, so a rate is over the same number of tokens on every build | the reply may end early; rates are over the tokens that came |
| `top_k` | the sampling a chat client sends | left out |
| `chat_template_kwargs.enable_thinking` | `--thinking on\|off` | the endpoint's default |
| `stream_options.include_usage` | the server's own token counts | completion tokens are counted as one per streamed update and prompt lengths are estimated from characters: rates are approximate and the tool says so |
| `GET /metrics` (this engine's counters) | accepted tokens per speculative step, prefix-cache hit tokens and preemptions per scenario | those notes are absent |
| `max_model_len` in `/v1/models` | skipping depths past the context | pass `--context` |
| `devices`, `tensor_parallel` in `/v1/models` | the GPU and TP in the page's facts | pass `--gpu`, `--tp` |
| `machine` in `/v1/models` (OS, kernel, CPU and its threads, host memory) | the page's machine row | the bench's own machine when it measured the cards here (power sampling: the same machine); else no row |
| `build`, `oneapi`, `drafter` in `/v1/models` | the engine row (revision, oneAPI version), the speculation row (DSpine, DFlash 2, MTP or off) and the build step of "Run this yourself" | the launcher's `engine_rev` note; the drafter as the settings name it. A `build` of `unknown` (no checkout: a Docker image built without `XEN_BUILD_REV`, or git refusing the folder) shows as "did not record its revision", and the clone step cannot pin one |
| `launch` (`args`, `env`) in `/v1/models`: xenithium-server with `--report-launch` (`XEN_REPORT_LAUNCH=1`) | the server step of "Run this yourself": its command line with its `XEN_` variables, exact | a launcher run: `scripts/start-server.sh` with its settings; else `--note server_cmd="..."`; else the page says the server was not recorded |

Prompt lengths are given in tokens, but the tool has no tokenizer (the endpoint's is unknown). On first use of a text
kind it sends one prompt of known length, reads `usage.prompt_tokens` and uses that ratio; every row reports the prompt
tokens the server counted, and those are what the rates are computed from.

## Scenarios

`xenithium-bench list` prints every parameter with its default.

| id | what it sends | what it reports |
|---|---|---|
| `decode` | One request at a time over short prompts of eight kinds of output (chat, code, reasoning, maths, prose, JSON, an edit, a summary), sampled. | Decode tok/s, TTFT, tokens per streamed update and the gap between updates per category, and the rate of a mix of the categories (the run's headline; below). Draft acceptance differs widely by category, so the categories are never averaged unweighted. |
| `concurrency` | N requests in flight at once (1, 2, 4, 8, 16), closed loop: a stream sends its next request when its last one ends. | tok/s of all streams together, tok/s per request, TTFT median and worst. A level past the server's slot count queues, and its TTFT shows the wait. |
| `prefill` | New prompts of 2K to 250K tokens of real source text, each starting with a line no earlier request began with (so a prefix cache cannot serve it). | TTFT and prompt tokens per second at each depth. |
| `deep-decode` | A 32K / 109K / 250K-token document once, then the same document again. | The cold TTFT, decode tok/s *with that context* (the number a long-context attention change moves), and the TTFT of the repeat (a prefix-cache hit, or a second prefill where there is no cache). |
| `turns` | An agent session: a 16K-token document, then turns that each re-send the whole conversation plus a new tool result. | TTFT and prompt tokens per turn, and the tokens the server says it took from its cache. |
| `burst` | N requests at once on one 32K-token document with different questions, against one request alone. | Time until all are answered, TTFT each. Shows whether requests that share a prefix wait for one computation or each pay it. |
| `late-arrival` | A 32K-token prompt sent while 4 or 8 streams generate. | The late prompt's TTFT, and the running streams' longest pause and token rate meanwhile. |
| `mixed` | One stream at 64K tokens of context generating with three short streams. | Decode tok/s of each class in the same steps. |
| `arrivals` | An open-loop load: requests arriving at 0.5, 1 and 2 per second with random gaps (a Poisson process, drawn from the seed), whether or not the earlier ones are answered. | TTFT and whole-request latency per rate (median, spread, worst), tokens per second served and the most requests in flight. The rate at which the TTFT's tail and the requests in flight leave the ground is the load the endpoint sustains; a closed loop cannot show it, since there a slow server slows its own load. |
| `vision` | A generated picture (a 3 x 3 grid of coloured squares, a PNG in a `data:` URL) with a question about one square, at 448, 896 and 1,344 pixels. Every picture is new. | TTFT, the prompt tokens a picture of that size costs, and whether the colour was named. An endpoint without image input is reported as skipped, not failed (this server: `XEN_VISION=1`). |
| `quality` | Thirty questions with one checkable answer in five categories (arithmetic, reasoning, reading code, extraction from a passage, following a format), and four on facts planted across an 8K-token document, greedy. The reply ends with `ANSWER: ...`, which is graded. Asked without a reasoning pass unless the run has `--thinking on`. | Correct answers per category and the share over all; every wrong answer with its question, the expected answer and the one given (first in the page's section, linked from the checks strip and the Quality figure; `✗` lines in the text report). The questions are easy to middling on purpose: a healthy model answers nearly all, so a drop points at the build or a lossy setting, not at the model's limits. It is a canary, not a leaderboard score. |
| `long-agents` | 1, 2 and 4 agents at work on long contexts. First every agent's own document is read at once, then every agent produces 10K tokens over tasks of every kind (maths, code, edits, prose, JSON ...), turn by turn, each turn re-sending its conversation. The document length is not fixed: it is sized so that the most agents of the run together fill 75 % of the endpoint's KV pool (`pool_pct`; about 200K tokens each for four agents on a 1M-token pool), because what an endpoint can hold differs with the cards, their number and the KV format. Without a reported pool, 75 % of the context length per agent. `depth=N` fixes it. | Per number of agents: the prompt rate while all documents are read at once and each one's TTFT, decode tok/s per request and of all agents together, tokens per update, the TTFT of a turn on the cached context, power and tokens per joule; and a second table of decode tok/s per kind of output for each number of agents. |
| `prefill-decode` | One agent's 64K-token prompt read alone, then a new one read while three agents generate on 32K-token contexts of their own. | The prompt's rate alone and beside them, the agents' token rate just before and while it is read, their longest pause, and the share of its own speed each side keeps (the two add up to about 100 % when the cards' time is only divided). |
| `needle` | A passphrase sentence at 35 % of a 32K / 128K / 250K-token document, asked for at the end, greedy. | Found or not. The check to run after turning on a lossy long-context setting (`XEN_ATTN_SELECT`, `XEN_PREFILL_SELECT`, an int8-v4 KV cache). |
| `repeat` | One request three times, greedy and sampled with a seed; a 16K-token document computed once and then continued from the prefix cache. | Whether the replies are identical. Greedy replies of this engine are; a reply computed from a cached prefix may differ from the first in the last float bit and so, rarely, in a token. |
| `structured` | Replies constrained to a JSON schema (`response_format`). | Decode tok/s, TTFT, and whether each reply parses as the schema's object. |
| `chat-sessions` | 4, 8 and 16 users each holding a conversation of four turns, staggered, with random pauses to read and type between a reply and the next turn (mean `think_s`); every turn re-sends the whole conversation. | TTFT and reading speed of every turn, the share of turns within the target (first token within `slo_ttft_ms`, the reply at `slo_tps` or faster), tok/s of all replies and of those within the target (goodput), and the most users carried with `slo_pct` of turns in the target. The number to size a chat deployment by. |
| `rag` | 1, 4 and 8 users at once asking questions over eight retrieved passages (about 6,400 tokens, new ones every request) under one shared instruction block; one passage plants the fact the question asks for. | TTFT and request latency, prompt tokens read per second by all requests together, and whether the answer holds the planted fact. A retrieval app is prompt-heavy: this is prefill under load, not decode. |
| `autocomplete` | 1 and 4 developers typing in 2,000-token files: before every completion the developer's typing sends and abandons requests (each a few characters longer, closed after `abandon_ms`), as an editor plugin does. Greedy, short replies. | The completion's whole latency and TTFT, and the prompt tokens the prefix cache served. Shows whether abandoned requests are dropped promptly and whether a prompt that grows by a few characters is read from the cache. |
| `tool-calls` | Five tasks that need a tool (read a file, search code, run tests, the weather, open a ticket) and one that needs none, with the tools sent as an agent sends them (`tools`, `tool_choice: auto`). The tool's result is sent back as a `tool` turn. | Whether the right function was called, whether its arguments parse and hold the task's values, whether no tool was called where none was needed, the time until the call is complete, and whether the answer after the result uses it. An endpoint without tool calls is reported as skipped. |
| `long-reply` | One reply of 8K tokens (a whole source file) cut into eight stretches by the tokens streamed. | The decode step, the decode rate and the longest pause in each stretch: a slowdown as the reply's own context grows, or a stall, shows where it happens. |

Suites:

| id | scenarios | about |
|---|---|---|
| `quick` | decode (1 pass), concurrency 1 / 4 / 8, prefill to 32K, deep-decode at 32K | 3 min |
| `standard` | decode, concurrency 1-16, prefill to 128K, deep-decode at 32K and 109K, turns, structured, quality | 13 min |
| `full` | every scenario, to the endpoint's context | 62 min |
| `long` | prefill 32K-250K, deep-decode, needle, turns from a 64K document | 13 min |
| `load` | concurrency 1-16, arrivals, late-arrival, burst, mixed | 12 min |
| `agents` | long-agents (1 / 2 / 4), prefill-decode, turns from a 64K document | 25 min |
| `check` | quality, needle, repeat, structured, vision, tool-calls | 7 min |
| `apps` | chat-sessions, rag, autocomplete, tool-calls, long-reply | 7 min |

Times are for the dense 27B on two cards; a suite takes as long as the endpoint is slow.

### The decode headline's mix

The decode scenario's one figure is the rate of a mix of its eight categories: `decode.weights` names the mix, each
weight the share of generated tokens of that kind. The figure is the rate at which that mix of tokens comes out, the
weighted harmonic mean of the categories' median rates (each kind's time per token, weighted, inverted). An arithmetic
mean of rates reads high: half the tokens at 300 tok/s and half at 100 come out at 150, not 200.

| `decode.weights` | chat | code | reasoning | maths | prose | json | edit | summary |
|---|---|---|---|---|---|---|---|---|
| `agent` (default) | 4 % | 19 % | 63 % | 0 | 3 % | 3 % | 4 % | 4 % |
| `even` | 12.5 % | 12.5 % | 12.5 % | 12.5 % | 12.5 % | 12.5 % | 12.5 % | 12.5 % |
| `easy` (the easy mode's) | 0 | 30 % | 20 % | 0 | 15 % | 15 % | 10 % | 10 % |

`easy` is the exception to the harmonic mean: its figure is the weighted arithmetic mean of the categories' median
rates, the figure the usual public benchmarks quote, and its row in the report says so.

Or your own shares: `--set decode.weights=code:1,reasoning:3` (normalised to sum 1; a category left out weighs 0; an
unknown name, category or share is refused before anything is sent).

`agent` is the mix of a coding agent's output, not one you choose: what an agent harness reading, editing, building
and testing code generates, split by what was generated: the thinking (reasoning), shell commands, scripts and source
files written (code), patches (edit), other tool-call arguments and the call envelopes (json), notes and docs written
(prose), the narration beside tool calls (chat), and the report that ends a turn of work (summary). Maths is left out. The shares differ by model mostly in the thinking: the 27B spent 69 % of its tokens on it,
Flash-Next 47 %. The mix's row is labelled with its name (`agent mix`), so `compare` never sets two different mixes
against each other; runs made before the mixes have a `weighted` row (code 30 %, the rest 10 % each, an arithmetic
mean) and the report reads it as such.

## Power and clock

With `--power` on, the tool reads every GPU's board energy counter (`energy1_input` of the card's hwmon directory,
cumulative microjoules), its actual clock (`act_freq`), its GPU package temperature (the hwmon input labelled `pkg`)
and what limits its clock (`freq0/throttle/reasons`) four times a second from `/sys/class/drm`, and once its maximum
clock (`rp0_freq`). No root is needed.

- Each scenario reports the mean power per card, the energy, the generated tokens per joule and the clock's mean and
  lowest value. `concurrency`, `arrivals` and `prefill` also report power and tokens per joule per row, so the most
  efficient stream count is visible. Tokens per joule is the same number as tok/s per watt.
- The run file keeps a trace of power over time and the power before the first scenario (`idle_watts`); the HTML page
  draws it under the run's timeline with the cards' power limit. Beside it, per trace point, the hottest card's
  package temperature (`trace_temp_c`) and every throttle reason seen since the point before (`trace_throttle`).
- The counters cover the whole card: anything else running on it is counted, an idle second card too. The host's CPU
  and the rest of the machine are not counted, so this is not the wall-socket figure.
- It says nothing about an endpoint on another machine; `auto` leaves it off there.

## The HTML page

`report.html` in the run directory (`xenithium-bench html RUN` writes it again): the headline figures, a strip of
yes / no checks, the run over time with the power trace, a diagram of one request with the run's own medians, then
every scenario with its charts, its table and its notes, and everything the run recorded.

Under the title, the run's facts as labelled rows (engine, model, KV cache, parallel, speculation, hardware,
sampling, during run, machine) and **Run this yourself** (folded): the run repeated from a fresh machine, for someone who has
never run the engine, each command with a copy button. What is needed (Linux, the `xe` driver, the run's cards), then:
clone the repository at the server's revision (`git describe` of its build; a `-dirty` build is flagged), install the
dependencies (`scripts/install-deps.sh`, then `sycl-ls` to see the cards), build, the model files the server read
(model, drafter, sidecar, with the paths the run used and an `ls -d` to check them), start the server as it was started
(with `--report-launch`: its arguments and `XEN_` variables as it reports them, values of key- or token-like names
hidden; a launcher run: `scripts/start-server.sh` with its settings), run the benchmark as it was typed (`--api-key`
hidden; runs from before are rebuilt from their settings), and open or `compare` the result. With power sampling (the cards are this
machine's), the hardware row also shows the GPU package temperature (mean and peak), how much of the run the clock was
throttled and why (the driver's reasons: `pl1` power limit, `thermal`, `prochot`...) or "never throttled", and the
clock's maximum. The settings the rows show come from the
server's own report first, then the launcher's notes, then a `server_cmd` note. The last section, **Everything the run
recorded**, holds what the rows leave out (endpoint, suite, replies, the API extensions taken or not, power limits,
PCI addresses) and every note and `XEN_` variable as recorded.

It is one file with inline style, inline SVG and one small inline script: no font, no request to anywhere. It opens
offline, prints (light theme), and follows the reader's light or dark setting; the switch at the top overrides it.
With the script, every chart mark answers the pointer with its figures (the value, the spread, the samples, the other
series at that point), a legend entry hides or shows its series, the power trace reads out every card at the pointer's
moment, a span of the timeline jumps to its scenario, the side list follows the scroll, scenarios fold, and tables sort
by any column and copy as tab-separated text. Without the script the page is the same, with the browser's own
tooltips. A scenario's charts are what the scenario declared (`ScenarioResult::charts`: bars, a line over a numeric
label, the spread of a latency, dots for yes / no), so a new scenario appears on the page without touching the page's
code.

## Reading the numbers

- **decode tok/s** is the tokens after the first streamed update over the time from the first update to the last, so
  it excludes the prompt. **TTFT** is from just before the connect to the first update that carries text, so it
  includes the connection, the queue and the prompt.
- **prefill tok/s** is the prompt tokens the server had to process (its count, minus the tokens it reports as cached
  in `usage.prompt_tokens_details.cached_tokens`) over the TTFT. A server that does not report them: a request sent
  alone takes the growth of the `/metrics` prefix-cache counter over the request instead. That counter is
  server-wide, so a growth larger than the prompt (another client's hits) leaves the cached count unknown and the
  row without a prefill rate.
- **tok/update** above 1 means the server sends several tokens per step (speculative decoding); **update gap** is the
  median time between two updates, the decode step as a client sees it.
- A cell shows the median of its samples; latencies with four or more samples show `median / max`. Yes / no checks
  show `yes/total`.
- `results.json` keeps every sample of every metric with its median, mean, min, max, p10 and p90, the run's seed,
  sampling, corpus set, version and hash, the endpoint's extensions and the notes. `summary` holds the headline numbers.
- `compare` shows the change of each metric's median and marks it `better` or `WORSE` only where it exceeds both
  0.5 % and the runs' own spread (half the p10-p90 range over the median). A change inside the spread is printed
  without a mark: it is not a result.
- Sampled runs with the same seed draw the same tokens only while both builds compute the same logits. Compare builds
  that change numerics at `--temperature 0` as well, and judge quality with the engine's KL gates, not here.

### Against other benchmarks

A benchmark that lets the model end its replies reports a higher decode rate for an engine with speculative decoding
than this tool's default does, on the same server. The decode step is the same in both; the tokens per step are not:

- `--reply-length fixed` (the default) sends `ignore_eos`, so every reply runs to its budget. Past the model's own end
  the text is no longer what the draft was trained to predict, and few draft tokens are accepted there. A benchmark
  without `ignore_eos` times only the part of the reply the model meant to write. Run with `--reply-length natural` to
  measure that way; keep `fixed` to compare builds, since natural replies differ in length from build to build.
- The prompts differ: each benchmark has its own texts, and acceptance differs by text (a code prompt drafts better
  than a story).
- The headline is the agent mix (above) as a harmonic mean; a weighted arithmetic mean of the same rates reads higher.
  Compare per category.
- `--mode easy` puts all of these on the common footing: natural replies, tasks whose replies repeat or reshape their
  input, the `easy` mix as a weighted arithmetic mean. Its numbers are the ones to set beside other benchmarks'.
- A run with failed requests has blank or partial rows (`errors` in the report): it is not a result.



## Modes

A drafting engine's decode rate depends on how well its draft guesses the text: the same build reads very differently
on different prompts and with different ways of counting. `--mode` fixes one way, recorded in the run file with its
version (`compare` warns across modes):

| | `hard` (default) | `medium` | `easy` |
|---|---|---|---|
| long prompts | the `hard` corpus: this engine's own code and docs | the `medium` corpus: conventional template-made code and prose | the `medium` corpus |
| short prompts (decode, concurrency, arrivals) | open tasks written from scratch, three per category | the same | tasks whose reply mostly repeats or reshapes the prompt's input (a diff, a whole edited file, JSON from given data, a fix of given code, a chat that goes on), four per category, shorter reply budgets |
| replies | `fixed`: run to `max_tokens` | `fixed` | `natural`: end where the model ends them |
| prefix cache on the short prompts | the same prompt is sent every pass | the same | every request starts with a new line: none is served from the cache |
| decode samples per category | 9 (3 prompts x `passes` 3) | 9 | 20 (4 prompts x 5 passes, after 3 warm-up requests) |
| decode headline | `agent` mix, harmonic mean | the same | `easy` mix, weighted arithmetic mean |

`hard` gives the published numbers. `easy` measures the way the usual public benchmarks do, so its numbers are the ones
to set beside theirs; it is not a measure of an agent's experience. `medium` sits between: the hard method over text
a drafter guesses well, which moves the long-context scenarios more than the decode headline (prefill speed depends on
the token count, not on the text).

A mode only sets defaults. `--corpus`, `--reply-length` and `--set` override them (a suite's own parameter, such as
`quick`'s single decode pass, stays as the suite sets it). The launcher's Benchmark tab picks the mode with `m`
([launcher.md](launcher.md)), easy by default. Changing a prompt of the easy set changes the easy mode's numbers: bump
its version in `tools/bench/mode.cpp` with it.

## The corpus

Long prompts are cut from text embedded in the binary, two sets of `code` and `prose` in `tools/bench/corpus/<set>/`:

- `hard` (the hard mode's, version 4): frozen snapshots of this engine's sources and of its documentation. Real text a
  drafter guesses least well: the strict measurement.
- `medium` (the medium and easy modes', version 1): conventional, template-made code and prose - entity classes, CRUD
  repositories and services, their tests, API clients, UI components, SQL, manifests; a travel guide, product manuals,
  recipes, help-centre articles, local news. Regular text of the kind most public benchmarks use, made to be guessed
  well by a drafter: its long-document numbers do not compare with `hard` runs. Runs made before the modes name this
  set `easy`; the tool reads them as the medium mode.

The sets are not kept in step with the tree, since two builds compare only on the same text. Changing a file of a set
changes every prompt cut from it: bump its version (`kCorpusVersion` for hard, `kMediumCorpusVersion` for medium) and
the hash pinned in `tests/test_bench.cpp` with it. A depth past the set's size repeats the text and the report says so
(a repeated document is easier to process for some engines).


## Adding a scenario

A scenario is one file in `tools/bench/scenarios/` with a class behind `Scenario` (`scenario.hpp`): its id, title, a
one-sentence description, its parameters (`ParamSpec`: key, default, meaning) and `run(Session &, const Params &)`,
which returns rows of labels and metrics. A `check(params)` override refuses a parameter value it cannot use before
anything is sent. Add its `make_<name>()` to `registry.cpp` and its rows to the suites there,
and it has its command-line parameters, its table, its place in `results.json` and in `compare` without further code.
`Session` (`session.hpp`) gives it prompts of a token length, the run's sampling and seeds, and requests sent together
or in a closed loop; `PowerSpan` adds a row's power; `result.charts` says how the rows are drawn on the HTML page; rows
with a `table` caption form a second table (a matrix per category beside the figures per load level).

## Not measured

Wall-socket power (only the GPUs' board power is read), video input, and model quality in the leaderboard sense: the
quality scenario tells a damaged model from a healthy one, it does not rank models. The engine's KL gates remain the
measure for a numerics change.
