# LoRA adapter support

Load a standard **PEFT LoRA adapter** (`adapter_model.safetensors` + `adapter_config.json`) at startup and apply its
delta during inference, on top of the quantized base checkpoint. No Python at runtime — the adapter is read, sharded
per card, packed and uploaded in C++, and the delta is added inside the existing GEMM outputs.

- **Scope:** the dense path of **qwen35** (Qwen3.8-27B) and **llama**. The MoE trunks (**qwen38 Flash-Next**,
  **flint**) do **not** apply the adapter — `resolve()` rejects `--lora` for them at startup.
- **Startup-only:** one adapter for the server's life, applied to every request. (Drafters — DFlash/DSpine/MTP — are
  not adapted; acceptance may dip slightly, correctness is preserved.)

## The math

At each adapted projection the adapter adds a low-rank delta to the base output:

```
out += scale · B·(A·x)          scale = lora_alpha / r
```

`A` is `[r][k]`, `B` is `[n][r]`. The rank is padded to `r16 = ceil(r/16)·16` (the dense bf16 GEMM body needs
`n % 16 == 0`). The delta is computed in fp32 and accumulated into the caller's existing fp32 buffer **before** the
engine's own rounding/reduce, so it lands at the same precision the base path already has:

1. `mid = A·x` — fp32, then rounded to bf16 **once** (matching how the engine rounds activations).
2. `delta = B·mid` — bf16 DPAS GEMM with fp32 accumulate.
3. `out += scale·delta` — added into the fp32 output buffer.

`lora_apply` (kernels/lora.hpp) runs decode (`m <= 64`) through the dense GEMM body and prefill (`m > 64`) through the
prefill dense body, so it works for both a single-row step and a full prefill chunk.

## Placement (tensor-parallel sharding)

Each module is either **column-parallel** or **row-parallel**, and A/B are sharded per card to match:

| module  | class            | A (per card)          | B (per card)        | where the delta lands |
|---------|------------------|-----------------------|---------------------|-----------------------|
| q, k, v | column-parallel  | replicated `[r16][k]` | row-split `[n/C][r16]` | the fp32 qkv buffer (before norms/rope) |
| o       | row-parallel     | col-split `[r16][k/C]`| replicated `[n][r16]`  | the fp32 o partial (before the pair sum) |
| gate, up| column-parallel  | replicated `[r16][k]` | row-split `[n/C][r16]` | the fp32 gate\|up buffer (before swiglu) |
| down    | row-parallel     | col-split `[r16][k/C]`| replicated `[n][r16]`  | the fp32 part buffer (before pair_reduce) |

Row-parallel deltas are added to the per-card fp32 part buffer **before** the existing two-card reduce, so no new wire
traffic is needed. Column-parallel deltas have no cross-card traffic at all.

## Files

### New

| file | what it is |
|------|------------|
| `include/xenithium/kernels/lora.hpp` | `LoraModule`, `LoraLayer`, `LoraScratch` structs and the `lora_f32_to_bf16` / `lora_accum` / `lora_apply` kernel declarations. |
| `src/kernels/lora/lora.cpp` | The delta kernels: `lora_apply` (mid → bf16 → delta → accumulate), decode via `dense_gemm`, prefill via `pf_dense`. |
| `include/xenithium/load/lora.hpp` | `LoraAdapter` + `LoraGeom`: reads the adapter, validates it, shards per card, uploads, and hands out per-layer `LoraLayer` bundles + per-card scratch. |
| `src/load/lora.cpp` | The loader. Parses `adapter_config.json`, maps each `…lora_{A,B}.weight` to (layer, module, placement), validates shapes, pads r→r16, converts F32→bf16, packs and uploads one buffer per card. `reserve_scratch` sizes the per-card mid/delta buffers. |
| `tests/test_lora.cpp` | GPU gate: config rejects, name→placement mapping, shape rejects, `lora_apply` vs a host reference (decode m=4 + prefill m=100, all modules, 1 and 2 cards), and zero-adapter bit-identical invariance. |

### Modified

| file | change |
|------|--------|
| `include/xenithium/arch/qwen35/model.hpp` | `ModelOptions::lora_dir`; `Model` gains `lora_` and `has_lora()` / `lora_name()` / `lora_layer(c,l)` / `lora_scratch(c)`. |
| `include/xenithium/arch/qwen35/mlp.hpp` | `Mlp::run` / `run_rows` / `gate_up_rows` gain `lora` + `layer` params (null = no adapter). |
| `include/xenithium/kernels/attention/gqa.hpp` | `gqa_decode` / `gqa_rows` / `gqa_rows_o` gain `lora` + `lora_sc` params (null = no adapter). |
| `src/arch/qwen35/model.cpp` | Loads the `LoraAdapter` after `dense_->load`, calls `reserve_scratch`, and passes `lora_layer`/`lora_scratch` into the attention + MLP calls on both the decode and prefill paths. |
| `src/arch/qwen35/mlp.cpp` | Applies the gate/up (column) and down (row) deltas into the fp32 gu/part buffers. |
| `src/kernels/attention/gqa.cpp` | Applies the q/k/v deltas into the fp32 qkv (before norms/rope) and the o delta before the sum; a LoRA layer disables the peer-row split-send path (LoRA uses the plain sum). |
| `src/arch/qwen35/serve.cpp` | `resolve()` rejects `--lora` for the MoE trunks; `model_options` sets `mo.lora_dir = o.lora`. |
| `src/arch/qwen38/serve.cpp` | Rejects `--lora` for the MoE trunk. |
| `include/xenithium/serve/options.hpp` | `Options::lora` (the adapter dir). |
| `tools/server/options.cpp` | `--lora DIR` flag. |
| `scripts/start-server.sh` | `XEN_LORA_ADAPTER` env → `--lora` (with `~` expansion). |
| `tools/launch/settings.cpp` | `XEN_LORA_ADAPTER` launcher setting (Qwen35 scope, path) under an "Adapters" category. |
| `docs/server-env.md` | `XEN_LORA_ADAPTER` row in the settings table. |
| `src/CMakeLists.txt` | `load/lora.cpp` added to `xenithium_arch`. |
| `src/kernels/CMakeLists.txt` | `xenithium_kernel_family(lora … lora.cpp)`. |
| `tests/CMakeLists.txt` | `xenithium_test(lora GPU …)`. |

## Build

The workspace has no SYCL toolchain — build on the Intel Arc build machine (192.168.1.175) in docker.

**Build** in `xen-builder:latest` (has `icpx` **and** `ocloc`, required for `bmg` AOT). Do **not** use
`intel/oneapi:2026.2.0-devel-ubuntu24.04` — it lacks `ocloc` and AOT fails.

```bash
docker run --rm -v ~/xen-lora2:/work -v ~:~ -w /work xen-builder:latest \
  bash -c "source /opt/intel/oneapi/setvars.sh; bash scripts/build-host.sh"
```

Host gates: `ctest -L host` (19/19). The LoRA GPU gate builds as `xenithium-test-lora`.

## Install / use

Point the server at a PEFT adapter directory (it must contain `adapter_model.safetensors` and `adapter_config.json`):

```bash
# env form (scripts/start-server.sh expands ~)
export XEN_LORA_ADAPTER=~/models/hermes-adapter-dl/lora-adapter

# or the flag directly
xenithium-server --lora ~/models/hermes-adapter-dl/lora-adapter …
```

- The adapter's `base_model_name_or_path` is recorded as the served adapter name (reported in `/v1/models` / `/version`).
- **Arch:** qwen35 (Qwen3.8-27B) and llama only. `--lora` on qwen38/flint throws at startup.
- **Supported modules:** q/k/v/o (GQA layers only) and gate/up/down (MLP). Any other target (router, in_proj_a/b,
  conv1d, norms, …) is rejected at load with the offending tensor named.
- **Not supported:** `use_dora`, `use_rslora` (rejected at load).
- The adapter's A/B must be **F32**; they are converted to bf16 at load.

### Real adapter geometry (harness-hub/Qwen3.8-27B-hermes-adapted)

r=32 → r16=32, alpha=64 → scale 2.0. 64 layers, 16 GQA (every 4th: 3,7,…,63) + 48 GDN. 512 tensors (256 modules × A/B),
all F32. hidden 5120, inter 17408, q_heads 48, kv_heads 8, head_dim 128, attn_output_gate true.

## Verification

- **Host gates:** `ctest -L host` — 19/19.
- **LoRA GPU gate:** `xenithium-test-lora` — 62 passed. `lora_apply` vs a host reference on cards 0+1 (decode m=4 +
  prefill m=100, every module q/k/v/o/gate/up/down, worst ~3e-07); zero-adapter invariance is bit-identical (this also
  exercises `reserve_scratch` at m=128).
- **Serve A/B:** the adapter server and a base-only server at `temperature=0` produce different text on the same
  prompts — proving the delta is actually added in the serve path.

## Measured cost

Arc Pro B70 (BMG-G31, 32 Xe cores), one card, Qwen3.8-27B GPTQ INT4 g128, 262144 context, 4 slots, int8 KV, TP1,
DFlash 2 int4 drafter (block 8), `XEN_STEP_GRAPHS=1`. Adapter: r=32, alpha=64, q/k/v/o + gate/up/down.

Single-stream decode, 512-token generations, TTFT taken from a separate 1-token request on the same prompt:

| prompt | decode |
|---|---|
| PCIe / root-complex explanation | 95.9 t/s |
| lock-free ring buffer in Python | 89.0 t/s |
| INT8 vs FP8 KV comparison | 99.9 t/s |
| **median** | **95.9 t/s** |

The same box's bare lane (no adapter) measures ~102 t/s on the same 512-token prompt set, so the adapter costs
**~1.07×** on single-stream decode. Note that every *verified* draft token also runs through the delta, so a
speculative drafter multiplies the number of corrected rows per step.

Prefill (TTFT of a 1-token completion):

| prompt tokens | TTFT | prefill |
|---|---|---|
| 3,623 | 1.99 s | 1,825 tok/s |
| 14,423 | 7.32 s | 1,971 tok/s |
| 43,223 | 19.77 s | 2,186 tok/s |

Concurrency, 256-token requests: 2 streams 146.0 tok/s aggregate (73.0 each); 4 streams 187.7 tok/s aggregate
(46.9 each).

## What is not yet established

The gate proves `lora_apply` matches a host reference for identical inputs, and the serve A/B proves the delta
reaches the output — but neither establishes that the adapted model equals the same LoRA merged into the BF16
weights. A merged-BF16 checkpoint served with no adapter is the remaining reference comparison.
