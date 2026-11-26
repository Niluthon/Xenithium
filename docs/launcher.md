# The launcher: `scripts/launch.sh`

A full-screen terminal interface for `xenithium-server`: pick a model, edit its settings (each model keeps its own
file, `models/<id>.env`, started from the shipped preset `models/presets/<id>.env`) with their documentation beside them, see the exact command `scripts/start-server.sh`
will run, launch it, and watch the server load and serve.

    scripts/build-host.sh xenithium-launch    # once (host only, no GPU code; a few seconds)
    scripts/launch.sh                         # opens ./server.env: the model it names, with that model's own settings
    scripts/launch.sh --env profiles/long.env # another settings file
    scripts/launch.sh --check                 # no screen: catalog and the file's invalid settings
    docker compose run --rm launch            # the same launcher in the Docker image (README.md, "Run with Docker Compose")

It needs a terminal of at least 64 x 18 (100 columns or more puts the help beside the list), and
uses truecolor when `COLORTERM` says so, else 256 colours; `NO_COLOR` turns colours off. The theme is royal blue (frames, tabs, the chosen option, panels); orange, green, yellow and red keep their meaning (changed, ok, warning, error). The mouse
works (click a setting, click it again to edit, wheel to scroll, click a tab).

## Screens

Four tabs, `1`-`4` or `Tab`; the Chat tab is hidden for now, so the header shows Settings `1`, Logs `2`, Benchmark `3`. The top bar shows the server state, whether the cards are free and the GPUs' power.

**Settings** (`1`)

- the model card, two lines: the checkpoint (its arch from the engine's own registry: `qwen38` / `qwen35` / `llama` /
  `flint`, size on disk, path), then whether it is ready to launch: slots and the max context (plus a `YaRN F` chip when the max context passes the
  trained positions and the load will scale the rotary), for MoE models the
  hot-expert ranking (`✓ experts ranking saved 2 h ago`, or `✓ experts ranking: recorded from this run on` before the
  server has saved one), the drafter, the prefix cache and
  the address it will listen on; a missing server binary shows there in red. Every served model gets a third
  line, the **memory plan**, from `xenithium-server --plan` run through the script (`SERVE_PLAN=1`) for the current
  values, off the UI thread (~0.4 s; "updating..." meanwhile). The dense models (qwen35, llama, flint) lead with the
  **KV context** the launch will have - the pool's tokens and GiB per card, and what it holds (`2 x max context, or
  16 x 69,440`: full-length requests, or every slot at once) - then VRAM per card; MoE models (qwen38) show
  VRAM per card, the KV tokens, and the hot experts out of the model's total. All end with the pinned host RAM against
  the RAM free now. The help panel of every setting the plan reads (slots, max context, KV dtype, KV pool, reserve,
  cards, the drafters and their settings, rows per sequence, prefill chunk, image input, warm-up, hot experts, prefix
  cache, ...) starts with the same numbers **and the KV pool each value of that setting would give**, the current one
  marked, the others with their change in percent (MoE: the hot experts beside it, since their pool is a formula of
  slots and context); a value the load would refuse reads `will not load: <why>`. The table is one plan per value, in
  parallel, started once the selection has rested a quarter of a second; a setting whose values all give the same pool
  says so under its text instead. The list a setting opens with `⏎` shows each value's KV tokens too. Under the text
  the panel shows the plan by part: weights, KV cache, slot states, hot experts,
  scratch, cold experts + embedding in host RAM, and how much prefix cache will fit; the "Capacity & memory" group's
  panel shows it as well. A plan whose pinned
  host memory exceeds the free RAM (minus 2.5 GiB for the server process and 2 GiB for the system) is an error on "Hot experts in VRAM" (a launch would be refused by the
  load); a prefix cache the server will cut and a pinned hot count past what fits (the load cuts it to the fit) are warnings. qwen38's hot experts and the dense models' KV pool are estimates (their loads measure
  the free VRAM) marked with `~`.
  "Hot experts in VRAM" (`XEN_HOT`) counts experts over all MoE layers, each split over both cards. `m` opens the model list: every directory
  under the models folder (the settings file's `XEN_MODELS_DIR`, empty = the project's `models/`; `--models DIR` for another place) with a
  `config.json`, grouped by arch, MTP and DFlash drafter
  directories left out, checkpoints the engine does not serve greyed; type to filter, or "Other directory" for any path
  (Tab completes). Picking another model opens **that model's own settings file** (your `models/<id>.env`, else the preset
  `models/presets/<id>.env`, found by its `XEN_MODEL` line; the defaults when it has neither, created as
  `models/<checkpoint folder>.env` by the next save), so
  one model's values never carry over to another; with unsaved changes it asks first (save them, discard them, or
  cancel). The Settings frame's title names the file (`models/qwen38-flash-next.env`, `(preset)` while a preset is shown, `(new)` when the model has no file yet).
  The served model name (`XEN_SERVED_NAME`) takes the checkpoint's folder name when it is blank or left
  at its default, and follows it again when the model changes; set it to override.
- the settings, grouped in the order you decide them: model and network, capacity and memory, speculative decoding,
  prompt processing and cache, numerics, queue, runtime, diagnostics. Only the settings the server reads for this model
  are listed (`a` shows the others, marked n/a): the other arch's settings, a setting whose switch is off (draft
  vocabulary with the drafter off, SJF aging unless the queue policy is sjf, ...), and with the stub engine on
  everything the stub does not read. A search (`/`) also finds this model's settings whose switch is off, marked n/a
  (`/ https` shows Images by http(s) link while Image input is off). A launch passes the hidden ones at their default. A value that differs from the
  default is orange, an invalid one red, and `●` marks a setting changed since the file was opened. The expert files
  show this model's state next to `auto` (`auto · ranked 2 h ago`, `auto · none yet`, `auto · server creates`).
- the help panel: a problem with the value first (why it is refused), then what the selected setting does, laid out
  the same way for every setting: one or two plain sentences; a line per value worth knowing, the values in one column
  (`on` / `off`, `auto`, `empty`, ...); then labelled lines - `Gives:` (what it improves), `Takes:` (what it costs),
  `Output:` (does the reply change), `Needs:`, `KV context:` (what it does to the pool), `Tip:`, `Note:` - and bullets where
  models differ. Setting names and values are drawn as code. At the bottom the facts: the value now, the default, the
  options or range, which arch reads it, and its key and flag. On a group header it says what the group is about. The
  full row with its history and measurements stays in `docs/server-env.md` (find searches it). `J` `K` scroll a long
  text.
- the command: what `scripts/start-server.sh` prints with `SERVE_DRY_RUN=1` for the current values, refreshed on every
  change, with the script's own notes.
- the bottom line: the keys, and on the right what needs attention: unsaved changes, invalid settings or warnings, or
  `✓ ready to launch`. `i` lists every problem; Enter jumps to the setting.

**Logs** (`2`)

- the state: loading (time and the log's last line), READY with the URL (the LAN address when the
  server listens on `0.0.0.0`), stopping, or exited with its status.
- live numbers from `/metrics` once ready, on two lines. Now: requests running / slots, waiting,
  decode tok/s of the whole server (the tokens of all requests together over the last ~6 s, polled
  every second), per stream (that over the requests running), the peak since the server was seen,
  and for `qwen35` the KV pool use. Since the server started: requests done, tokens decoded and
  prefilled over all requests, and for `qwen35` the preemptions.
- the GPUs, one row: each card's board power now (its energy counter over the last second, from
  the xe driver's hwmon files; no root), its power limit and package temperature. The total is
  also in the top bar. Read-only: the launcher does not change power limits.
- the server's log, following its end (`↑` / PgUp scroll back, `End` follows again, `w` wraps).

**Chat** (hidden for now; was `3`)

- a conversation with the running server: type, `Enter` sends, the answer streams in as it is
  generated. The whole conversation is sent every turn (it is a chat, not single-shot prompts),
  so later answers see the earlier ones.
- the model's thinking streams too, shown dimmed above the answer, where the checkpoint thinks by
  default. The box header shows the live elapsed time while it answers, then the answer's time,
  token count and tok/s (from the stream's usage chunk).
- `Esc` stops a running answer (the connection closes and the server aborts it); the partial
  answer is not kept, so the next send continues from the last finished turn. `F4` starts a new
  conversation. `↑` `↓` PgUp PgDn scroll the transcript, `End` follows again.
- the tab refuses to send while no server runs (`r` launches one) or while it is still loading.

**Benchmark** (`3`)

- The tab splits in two: the runs list on top, and below it a log panel that tails the
  benchmark's output (`build/launch/bench.log`) live while a run is in flight. `j` `k` scroll it,
  `g` jumps to the top, `G` follows the end again, `w` toggles line wrapping, and the mouse
  wheel over the panel scrolls it.
- Benchmark runs, grouped by the model each run hit (`env.model`): one header per model (its run count), its runs
  newest first, each with the headline decode (the figure weighted over the output categories), the deepest prefill
  rate (`pp@depth`), the decode rate at the deepest context (`dec@depth`) and the widest concurrency total
  (`conc@streams`), where the run measured them.
- `Enter` opens a run's report: model, endpoint, host, the weighted decode with its per-category breakdown, the
  prefill, decode-at-depth and concurrency tables, tokens per joule and the quality score where measured (each wrong answer on a red line
  of its own: the question's id, the expected answer and the one given), then the run's notes. In the log a
  finished scenario's wrong answers are red lines marked `✗`. Every scenario's charts are in the run's `report.html` (one self-contained file; its path is shown),
  every table in `report.txt` (`xenithium-bench report <run>`; `xenithium-bench compare A B` sets two runs side by
  side). `Esc` goes back. A run started from the launcher samples the cards' power (the server is on this machine).
- `r` runs `xenithium-bench` ([bench.md](bench.md)) against the running server (the served model's name from
  `/v1/models`) after a pick of what to measure. A suite's tag is the run name's suffix:
  - **Standard** (`standard`, ~13 min): decode by category, concurrency 1-16, prefill to 128K tokens, decode at 32K
    and 109K of context, an agent session on the prefix cache, JSON-schema output, answer quality. The numbers to quote
    for a build;
  - **Quick look** (`quick`, ~3 min): every headline number once;
  - **Full** (`full`, ~62 min): every scenario, to the server's max context;
  - **Long contexts** (`long`, ~13 min): prefill to 250K, decode at depth, the needle, a long agent session. The set
    to run after changing a long-context setting (`XEN_ATTN_SELECT`, `XEN_PREFILL_SELECT`, the KV format);
  - **Load** (`load`, ~12 min): concurrency 1-16, requests arriving at a rate, a long prompt arriving beside running
    streams, a burst on one document, mixed depths. The set to run after changing a scheduler setting. A level past the
    server's slot count (`XEN_SLOTS`) queues the excess requests, so its TTFT includes that wait;
  - **Agents** (`agents`, ~25 min): 1, 2 and 4 agents each reading a long document at once and then producing 10K
    tokens of maths, code, edits and the other kinds of output on it; a long prompt read while three agents generate;
    a long session. The documents are sized from the server's KV pool (75 % of it for four agents), so the case
    follows `XEN_KV`, the card count and the pool setting;
  - **Check the output** (`check`, ~7 min): graded answers, the needle, reply repeatability, JSON-schema output,
    image input (skipped unless `XEN_VISION=1`), tool calls. The set to run after a numerics or a lossy setting changed;
  - **Real-world apps** (`apps`, ~7 min): chat users with pauses against a latency target (the most users carried),
    questions over retrieved passages under load, editor autocomplete with abandoned requests, an agent's tool calls,
    one 8K-token reply cut into stretches;
  - **Decode only**, **Prefill only**, **Concurrency sweep**: that scenario alone (`decode`, `prefill`,
    `concurrency`);
  - **Concurrency at stream counts you enter** (`streams`): asks for the counts (e.g. `1, 8, 24, 48`; 1-256, at most
    12, commas or spaces), then runs the concurrency scenario at them (`--set concurrency.levels=`). The prompt names
    the server's slot count when the launcher started it: a count past it queues. The counts are kept for the next
    run of the session and recorded as the `bench_streams` note.

  Depths past the server's max context are skipped and named in the report.

  Every run records the server's launch settings as notes in its `results.json` (`--note KEY=VALUE`):
  `bench_mode`, `engine_rev` (`git describe --always --dirty` of the tree), `settings_file`, and every non-empty
  setting of `build/launch/launch.env` when the launcher started the server. A server started outside the
  launcher records its command line as `server_cmd` instead. The report (`Enter`) lists the notes as
  `KEY=value` pairs.
- Every run measures by the tool's one method ([bench.md](bench.md#the-method)); a run's report (`Enter`) names its
  version, and a run made before it (with a mode: easy, medium or hard) is marked as older: its numbers do not compare
  with newer runs.
- `x` stops a run the tab started — `Esc` deliberately does not, so an accidental press can't
  kill a run. The binary and the runs directory come from `--bench-bin` and `--bench-runs` (defaults
  `build/xenithium-bench` and `$XEN_BENCH_RUNS`, else `~/.xenithium/bench/runs`).

A first start without a settings file opens a short welcome: pick a model, check the settings, launch.

## Keys

| key | does |
| --- | --- |
| `↑` `↓` / `j` `k`, PgUp PgDn, Home End | move |
| `Enter` | change: a popup for choices and suggestions, a line editor for the rest; fold a group |
| `←` `→` / `h` `l` | previous / next choice (or suggestion) in place; fold / unfold a group |
| `Space` | toggle on / off, next choice |
| `e` (F2) | type a custom value; `Tab` completes paths, `Ctrl-U` clears, `Esc` cancels |
| `d` / `D` | this setting / every setting back to the default of `start-server.sh` |
| `u` | undo |
| `i` | the problems of the current values; `Enter` goes to the setting |
| `/` | find a setting by name, key or flag (this model's settings whose switch is off too, marked n/a); with no name match it searches the help text |
| `m` | the model list (Settings tab) |
| `p` | settings files: `server.env`, the per-model files (`models/*.env` and the presets no file of yours replaces; opening one switches to its model) and the profiles; `n` saves the current values as a new profile, `x` deletes one |
| `s` / `S` | save to the open file (the model's own file, and `server.env` then names it) / save as (the model's file, a profile, a new profile) |
| `r` (F5) | launch, after the checks |
| `x` | stop the server (SIGTERM; after 5 s of stopping, `x` offers SIGKILL) |
| `Tab`, `1`-`4` | switch screens (the only keys besides `?` and `q` that act on every tab; every other key acts on the tab whose bottom line shows it) |
| `?` | keys |
| `q` | quit; with unsaved changes it asks first. A running server or ranking run keeps running |

On the Chat tab the input line owns the keys: `Enter` sends, `Esc` stops a running answer (or
clears the line), `↑` `↓` PgUp PgDn / `End` scroll the transcript, `F4` starts a new conversation.

## Files

- `models/presets/<id>.env` (tracked): the settings shipped for one checkpoint (docs/server-env.md, "Per-model
  settings files"). The launcher reads a preset and never writes it; the title shows `(preset)` while the values come
  from one.
- `models/<id>.env` (yours, git-ignored): the same checkpoint's settings once you saved them. The first save creates it
  from the preset's lines; from then on it replaces the preset (delete it to return to the preset). Saving updates the
  keys in place and keeps comments and other lines; a key the file lists stays listed (also at its default value), a
  new key is written only when it differs from the default, and the file always names its checkpoint (`XEN_MODEL`).
  `--model-envs DIR` points the launcher elsewhere (the script always uses the repo's).
- `models/presets/<id>.tp<N>.env` and your `models/<id>.tp<N>.env`: one setup's settings (N = the card count:
  `XEN_Q38_TP` for qwen38, `XEN_TP` for qwen35 / llama / flint), over the model's file (docs/server-env.md, "Per-setup
  settings files"), with the same preset / your file rule. With setups, the model's file keeps the "Model & network"
  settings and the setup key, and the setup's file keeps the rest. Changing Cards loads the new setup's file (asking
  first about unsaved changes), and `s` saves both files. The setup's file gets only what differs from the model's file
  and the defaults. The title shows both names (`models/<id>.env + <id>.tp4.env`).
- `server.env` (repo root): what `scripts/start-server.sh` reads; the launcher keeps only `XEN_MODEL_ENV=<id>` in it
  (the model to start), comments and other lines stay. Settings an older `server.env` holds win over the model's file
  (as in the script) and move into it on the next save.
- `profiles/NAME.env`: named settings files in the same format, holding every setting themselves (`p` to open, `n` to
  create; a model switch there changes `XEN_MODEL` in place). `scripts/start-server.sh` reads one with
  `SERVE_ENV=profiles/NAME.env` (the model's own file still fills in the keys it does not set).
- `build/launch/`: `launch.env` (the values of the last launch, every setting written out, `XEN_MODEL_ENV=` empty),
  `server.log` (its output; the previous launch's is kept as `server.prev.log`), `server.pid`, `preview.env`,
  `bench.log` (the Benchmark tab's run output, tailed in its log panel), `rank.log` / `rank.pid` (a
  make-hot-rank.sh run).
- `.cache/experts/<model folder>/`: `hot-rank.xrc` (the ranking, recorded by the server while it serves);
  `--experts DIR` points the launcher elsewhere (the script always uses the repo's).

`server.env`, `profiles/` and `.cache/` are local files (`.gitignore`).

## Launching

`r` first checks, and shows each check: the server binary is built, the model is a checkpoint
the engine serves, every setting is valid (the max context at most 1,048,576 on `qwen35` / `llama` / `flint` / `qwen38`; past the
checkpoint's `max_position_embeddings` a warning names the YaRN factor the load will pick, which does not block), the **KV context** the
launch will have (the memory plan's pool and what it holds: `KV context ~1,112,000 tokens - 2 x the 524,288 max context, or 16 x 69,440`; a
plan that fails, e.g. a max context whose KV does not fit, shows its reason as a warning), the port is free, no Docker container holds the cards
(skipped for the stub engine), and whether unsaved values are being launched (they are launched
as shown; the file is not changed). It refuses while any `xenithium-server` runs.

The launch writes `build/launch/launch.env` and runs `SERVE_ENV=build/launch/launch.env
scripts/start-server.sh` in its own session, output to `build/launch/server.log`. The script's own
checks and defaults apply as for a launch by hand, and the command is the one shown under Command.
Settings the model's arch ignores are launched at their default (`--hot-rank` is left off for
`qwen35`). The server outlives the launcher: quitting prints its pid and URL, and the next
`scripts/launch.sh` finds it again (`build/launch/server.pid`) with its log. A server started by
hand is shown too (state, metrics, stop), without a log.

## When the server ends

The Logs tab says why, in words, in its status line and as a `launcher: the server (pid N) ...` line at the end of
`build/launch/server.log` (red in the log pane, kept in the file), plus a note at the bottom:

- **killed by the kernel's OOM killer** (exit status 137): the host ran out of RAM. A SIGKILL leaves the server no
  chance to write anything, so the launcher looks the pid up in the kernel log (`journalctl -k`, else `dmesg`) and
  names the time and the RSS it had (`... at 00:33:50: the host ran out of RAM (the server held 47.3 GiB of 61 GiB)`).
  The server's own log shows the growth before it: every engine status line ends with
  `host RSS X GiB, Y / Z GiB free`, and when the free RAM falls under 8 % of the host (at least 3 GiB) it writes
  `server: WARNING host RAM low: ...` once (a recovery is logged once, and the next fall below the floor warns again).
- **killed by SIGKILL** from this launcher (Stop, then Kill now) or from another process (no OOM entry for the pid).
- **crashed** (SIGSEGV / SIGABRT / SIGBUS / SIGILL / SIGFPE): the server writes `server: crashed on SIGSEGV ...` and a
  backtrace to its log before it dies.
- **exited with status 1 / 2**: the server's own error (status 2 = refused settings) is the last line of its log.

A server started outside the launcher (adopted from `server.pid`) has no exit status; an OOM kill is still found in the
kernel log by its pid.

## Where the settings come from

The launcher does not keep its own copy of the defaults or the long help text: they are read at start
from `scripts/start-server.sh` (the `XEN_X=${XEN_X:-default}` lines) and `docs/server-env.md`
(the setting table and the diagnostic switches table). Its schema in `tools/launch/settings.cpp`
adds the label, group, type, allowed values, which arch reads each setting and a short description (`brief_of`, which must be rewritten when a setting's behaviour changes; its layout - paragraphs split by line breaks, `` `value` `` lines, labelled lines, bullets - is in the comment above it and drawn by `help_lines` in `term`; the `launch` gate fails when one is missing, when a code span is left open, when a choice of a closed list has no line, or when a setting the memory plan reads (`moves_memory`) does not say what it does to memory). `unused_reason` holds the rules that hide a setting for the model and the current values. A setting the
script or docs have and the schema does not still appears, as text under "Other"; `--check` and
the `launch` gate (`xenithium-test-launch`) report any mismatch between the three. Adding or
renaming a `server.env` setting therefore also means a line in `schema()`.

## Code

`tools/launch/`: `term` (terminal: raw mode, alternate screen, diff rendering, key and mouse
decoding; no curses), `settings` (catalog, env files, validation, model scan), `proc` (dry run,
spawn, process and port checks, Docker, metrics), `chat` (the conversation and the worker that
streams the answer over `/v1/chat/completions`, cancellable), `bench` (the Benchmark tab: the runs
scan, the `results.json` parse, the per-model grouping and the run/stop of `xenithium-bench`), `app`
(the screens), `job` (a background command with a log: the ranking run), `main`. The screens draw to
an off-screen canvas, so `tests/test_launch.cpp` drives whole flows by keys without a terminal, and
`xenithium-launch --render 120x36 --keys "/slots<enter>"` prints a frame as text.
