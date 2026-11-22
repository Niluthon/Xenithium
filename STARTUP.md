# Qwen3.8 Flash-Next TP4 startup on haxor

Use the native Intel W4A16 AutoRound checkpoint in `~/models/qwen38-w4a16-intel` (revision `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`) and the block-8 DFlash 2 checkpoint in `~/models/qwen38-flash-next-dflash2`. The record used four Arc Pro B70s and the Xenithium copy-wire changes in this checkout.

## Why the lane was slow

Direct GPU peer mailbox exchanges sometimes delivered stale or torn sectors during TP4 reductions. Non-finite activations propagated into DFlash, which returned effectively empty draft chains. The token-pick mailbox also produced out-of-range IDs that could fault the GPU if used as embedding indices. The failing configuration advanced about 1.95 tokens per speculative step.

Fused kernel mailboxes now allocate pinned host USM. The full 303-check gate kept the same sum order, BF16 rounding points, int8 bounds, and timeout behavior. Two-card and four-card mailboxes and owner exchanges use the same allocation policy. Device buffers still back ordered peer copies and the separately sequenced split transport.

The throughput recipe uses `XEN_Q38_COPY_WIRE=1` for attention, MoE and DFlash reductions. With `XEN_Q38_COPY_SPLIT_ROWS=16`, wide steps copy only each reducer's assigned FP32 rows, then share the BF16 results. The TP4 preset sets `XEN_Q38_COPY_DIRECT_OUT=1` so the peer copies write those BF16 rows straight into the attention and MoE output buffers. This removes one receiver kernel per split reduction and keeps the two sequence barriers and the same peer payload size. The reductions still round once after `(p0+p1)+(p2+p3)`. The full 64-row budget applies from 13 active requests. Short single-stream steps keep their existing budget.

## Build and give Xenithium the cards

On haxor, run:

```bash
cd ~/AI/xenithium
source scripts/oneapi-env.sh
scripts/build-host.sh xenithium-server xenithium-bench
python3 ~/AI/b70-recipe-launcher/switch.py --stop qwen38-flash-next
```

Wait for `Memory Used` to reach **226 MiB** and confirm port 8000 is free before starting. A second process listening on that port can make readiness checks pass against the wrong server.

```bash
for ((attempt=0; attempt<40; ++attempt)); do
  used=$(timeout 8 xpu-smi stats -d 0 | awk '/Memory Used/{print $NF}')
  [[ "$used" == 226 ]] && break
  sleep 3
done
[[ "$used" == 226 ]] && ss -ltn 'sport = :8000'
```

Proceed once the check prints the `ss` header with no listener rows. For an isolated acceptance run, clear the kernel buffer with `sudo -n dmesg -C` just before starting the server. After the run, count reset lines with `sudo -n dmesg -T | grep -c 'Engine reset'`.

## Start the record recipe

Run from `~/AI/xenithium`. Create a fresh tmux session for each independent acceptance gate.

```bash
tmux new-session -d -s q38-xenithium '
  cd ~/AI/xenithium
  source scripts/oneapi-env.sh
  XEN_HOST=127.0.0.1 XEN_Q38_TP=4 \
    XEN_MODEL_ENV=models/record/qwen38-flash-next-intel-dflash.env \
    scripts/start-server.sh --warmup 0 > /tmp/q38-xenithium.log 2>&1 &
  server_pid=$!
  echo "$server_pid" > /tmp/q38-xenithium.pid
  wait "$server_pid"
  printf "DONE %s\n" "$?" >> /tmp/q38-xenithium.log
  rm -f /tmp/q38-xenithium.pid
'
```

Poll until this succeeds and shows model ID `qwen38-flash-next-w4a16-intel`. Watch `/tmp/q38-xenithium.log` if the process exits before it becomes ready.

```bash
for ((attempt=0; attempt<100; ++attempt)); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | jq -e '.data[] | select(.id == "qwen38-flash-next-w4a16-intel")'; then
    break
  fi
  tmux has-session -t q38-xenithium 2>/dev/null || break
  sleep 4
done
```

The local recipe files under `models/record/` are Git-ignored. On a fresh checkout with the same server changes and checkpoints, the equivalent launch command uses the tracked Intel preset and explicit overrides:

```bash
XEN_HOST=127.0.0.1 XEN_Q38_TP=4 \
  XEN_MODEL_ENV=models/presets/qwen38-w4a16-intel.env \
  XEN_Q38_COPY_WIRE=1 XEN_Q38_COPY_DIRECT_OUT=1 XEN_DRAFTER_TREE=0 XEN_SEG_ROWS=8 \
  XEN_STEP_ROWS=64 XEN_Q38_COPY_SPLIT_ROWS=16 XEN_DRAFTER_CUT=0.02 XEN_DRAFTER_DEFER=64 \
  XEN_PREFILL_CHUNK=4096 XEN_DRAFTER_INT4=1 XEN_DRAFT_VOCAB=98304 \
  XEN_LOOKUP=4 XEN_ADAPTIVE_DEPTH=1 XEN_UNEQUAL_SEGMENTS=1 XEN_GRAPHS=1 \
  scripts/start-server.sh --warmup 0
```

The TP4 preset also supplies 16 slots, a 1,048,576-token context limit, and the int8-v4 KV cache. Leave tracing variables unset for performance measurements.

## Reproduce the measurements

From the Xenithium checkout:

```bash
./build/xenithium-bench run --scenario decode --set decode.passes=1 \
  --url http://127.0.0.1:8000/v1 --name q38-intel-tp4
```

On the direct-output build, the single-stream Xenithium agent mix measured **270.41 tok/s**, with **2.84 tokens per speculative step**. Fresh 8K runs measured **8,539 and 8,526 tok/s**, each on 8,007 prompt tokens. Fresh 32K runs processed 31,300 and 31,299 prompt tokens at **8,627 and 8,621 tok/s**. The prefix-cache test reused 3,328 of 3,425 prompt tokens. The temperature-zero test finished 220 tokens, and the quality sample correctly computed an average speed of 78.57 km/h. All gates completed with zero engine resets.

For the standard 16-user closed loop, start a fresh server and run:

```bash
./build/xenithium-bench run --scenario concurrency --set concurrency.levels=16 \
  --url http://127.0.0.1:8000/v1 --name q38-intel-tp4-c16
```

That sends 64 requests at concurrency 16. The direct-output build measured **786.56 and 784.65 aggregate tok/s** on two fresh-server runs. To reproduce the longer run, use the command above as a warmup, then run the same scenario twice with `--set concurrency.requests=192` on that server. The two warmed runs measured **872.67 and 876.02 aggregate tok/s**, generating 92,400 tokens each. All four measurements had zero failed requests and zero engine resets. The 911 tok/s comparison target remains open.

The result JSON files on haxor are in `~/.xenithium/bench/runs/` under `20261121-214153-q38-wide-concurrency-final`, `20261121-214301-q38-wide-final-64`, `20261121-214346-q38-wide-final-192`, and `20261121-214535-q38-wide-final-repeat-192`.

### Diagnosing 16-stream throughput

A control run with ordered peer copies and the 16-row crossover measured **780.92, 852.19, and 871.34 aggregate tok/s** at 64, 96, and 192 requests, respectively, on the same server. The warmed 192-request benchmark generated 92,400 tokens. These rates include short-prompt prefill and the final requests draining. Compare runs at the same request count and after the same warmup sequence.

A separate `XEN_DEC_PROFILE=1` run put the target GPU graph at **39.8 ms** under steady load. MoE accounted for about **18.0 ms**, GDN **11.8 ms**, and the hyper-connection mixers **7.3 ms**. In an additional profiling window with a 36.9 ms target graph, MoE expert computation took **13.3 ms** and its combine plus all-reduce took **2.2 ms**. Use the profiler for phase timing and unset it for throughput runs.

Raising the peer-copy crossover to 24 or 32 rows reached 862.28 and 866.48 tok/s on warmed 192-request runs. Switching wide decode to the pinned-host mailbox path reached 450.07 tok/s on the 64-request run, with busy steps near 81 ms. Ordered peer copies took about 42 ms. Fixed draft depth reached 870.41 tok/s on the warmed 192-request run. The MoE and mixer partition changes tested here increased step time. With direct output enabled, the median of the busy-period step timings went from about 41.74 to 41.52 ms. The final warmed run averaged 891.1 generated tok/s over its 100-second busy middle, based on the server counters. The logs for these comparisons are under `/tmp/q38-wide-*` on haxor.

For BetterBench, use the checkout on haxor and run from tmux. The shipped default sweep ends at concurrency 8. The following extends it to 16 and tests all three phases.

```bash
cd ~/AI/BetterBench
mkdir -p "$HOME/.betterbench"
out=$(mktemp -d "$HOME/.betterbench/q38-intel-tp4-XXXXXX")
jq '.concurrency_levels = [1,2,4,8,16] | .prefill_depths = [2000,8000,32000]' \
  config/default.json > "$out/config.json"
python3 -u -m betterbench.cli run --no-update-check \
  --endpoint http://127.0.0.1:8000/v1 --model qwen38-flash-next-w4a16-intel \
  --config "$out/config.json" --passes 20 --warmup 3 --name q38-intel-tp4-c16 \
  --note tp=4 --note quant=W4A16 --note wire=ordered-peer-copy \
  --out "$out/results.json" 2>&1 | tee "$out/bench.log"
```

### Earlier BetterBench comparison

BetterBench 0.6.1 at `f72bb90`, corpus v1.0, on November 21, 2026, using the earlier 32-row recipe. Single-stream decode used 20 measured passes in each of eight categories, with three warmups per category. The weighted median was **335.2 tok/s** with a weighted median TTFT of **49 ms**. This is BetterBench's own weighted mix and includes reasoning tokens. 104 of 160 requests hit the output cap, so the rate measures token generation under those limits. The Xenithium agent-mix record above uses a different request mix and fixed output lengths.

| Concurrent requests | Completed | Aggregate tok/s | Median per-request decode tok/s | Median TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 48/48 | 311.9 | 354.6 | 49.0 ms |
| 2 | 48/48 | 399.2 | 255.7 | 61.1 ms |
| 4 | 48/48 | 541.8 | 157.7 | 66.6 ms |
| 8 | 48/48 | 657.5 | 98.8 | 77.4 ms |
| 16 | 48/48 | 686.0 | 49.9 | 82.6 ms |

All five levels completed with zero engine resets. BetterBench prefill medians were 6,218.1, 7,885.9, and 8,511.1 tok/s at actual median prompt sizes of 1,550, 5,956, and 23,561 tokens. BetterBench's target depths are approximate. Use the Xenithium results above for the runs that actually processed about 8K and 32K tokens.

Artifacts on haxor are under `~/.betterbench/runs/20261121-q38-intel-tp4-c16/`, including `results.json`, `results.html`, `config.json`, `server.log`, and `kernel.log`. The 275.28 tok/s result is at `~/.xenithium/bench/runs/20261121-194851-q38-decode-final/results.json`.

## Mailbox validation

With the cards idle, the pinned-host mailbox build passed 20 consecutive runs of `XEN_WIRE_CALLS=200 ./build/xenithium-test-wire`. Each run passed all 303 checks. Logs are in `/tmp/xen-wire-pinned-accept/final-long-*.log`. The direct-output split copy passed all 34 checks in five consecutive 200-call runs with `XEN_TEST_COPY_WIRE=1 XEN_WIRE_CALLS=200 XEN_Q38_COPY_SPLIT_ROWS=16 XEN_Q38_COPY_DIRECT_OUT=1`. Those logs are at `/tmp/q38-wide-direct-gate-0*.log`.

## Optional return to the original lane

Stop the server PID recorded by the tmux command. Confirm its log shows `DONE` and port 8000 is closed before starting the original service. Poll until the models endpoint returns HTTP 200.

```bash
kill -TERM "$(cat /tmp/q38-xenithium.pid)"
# Wait for DONE in /tmp/q38-xenithium.log and confirm port 8000 is closed.
python3 ~/AI/b70-recipe-launcher/switch.py --start qwen38-flash-next
curl -fsS --max-time 2 http://127.0.0.1:19622/v1/models
```

The Xenithium record server listens on loopback port 8000 during the GPU handoff. Port 19622 is the original managed service.
