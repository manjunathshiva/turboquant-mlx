# TurboQuant-MLX

Extreme **weight** and **KV cache** compression for LLMs on Apple Silicon. MLX implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) — Hadamard rotation + Lloyd-Max codebooks applied both to weights (compile time) and the KV cache (run time).

Supports dense models (LLaMA, Qwen, Mistral), **Mixture-of-Experts** (Qwen-MoE, GPT-OSS, Qwen3.5-MoE, Qwen3.6-35B-A3B, Qwen3-235B-A22B, DeepSeek-V2/V3), and **Mamba/attention hybrids** (Nemotron-3-Nano-4B, Nemotron-3-Super-120B). Compatible with hybrid attention architectures, attention sinks, sliding-window attention, and linear attention layers.

**With both weight and KV cache compression at 3-bit, GPT-OSS-120B fits its full 131K context window in 50 GB on a 64 GB MacBook — and KV cache compression actually makes generation *faster* on the 120B (8.7 vs 6.4 tok/s) because the smaller cache cuts memory bandwidth more than dequant costs.**

**Expert streaming (v0.4.0)** runs MoE models whose weights exceed available RAM by paging only the router-selected experts from disk per token — e.g. the 35B-parameter Qwen3.6-35B-A3B runs on a **16 GB Mac mini** in under 4 GB of RAM, with output bit-identical to the fully-resident model. See [Qwen3.6-35B-A3B on a 16 GB Mac mini](#qwen36-35b-a3b-on-a-16-gb-mac-mini-expert-streaming).

**Local coding** — [Qwen3.6-27B](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32), a dense SWE-bench-grade coder, runs **fully resident on a 48 GB Mac** at 3-bit (~13 GB on disk, ~17.5 GB at runtime) and serves to Cursor / VS Code over an OpenAI-compatible endpoint. See [Qwen3.6-27B](#qwen36-27b-dense-coding-model-for-a-48-gb-mac).

## Key Results — Weight Compression

| Model | Method | Bits | PPL | Size | Gen Speed (M4 Max) |
|-------|--------|------|-----|------|---------------------|
| Qwen2.5-7B | TurboQuant | 3 | 8.92 | 3.5 GB | — |
| Qwen2.5-7B | Affine | 3 | 13.37 | 3.3 GB | — |
| GPT-OSS-20B | Affine (mlx-lm) | 4 | — | 11.2 GB | 148 tok/s |
| GPT-OSS-20B | MXFP4 (original) | 4 | 83.04 | 12.8 GB | — |
| GPT-OSS-20B | TurboQuant | 4 | 72.63 | 11.2 GB | — |
| GPT-OSS-20B | TurboQuant | 3 | 78.60 | 9.3 GB | **73 tok/s** |
| GPT-OSS-120B | [Affine 4-bit (mlx-community)](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 | — | 65.8 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | MXFP4 (original) | 4 | — | 63.5 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | TurboQuant | 3 | — | 48 GB | **44 tok/s** |
| **[GPT-OSS-120B (hybrid for 48GB)](https://huggingface.co/manjunathshiva/gpt-oss-120b-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **~35 GB** | **42–50 tok/s** |
| GPT-OSS-120B | TurboQuant | 2 | — | 32 GB | 51 tok/s (poor quality) |
| Qwen3.5-122B-A10B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Qwen3.5-122B-A10B** | **TurboQuant** | **3** | **—** | **~50 GB** | **26.5 tok/s (64 GB) · streams on a 16 GB Mac mini** |
| **[Qwen3.6-35B-A3B](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **~16 GB** | **~60 tok/s (resident) · runs in <4 GB via streaming** |
| **[Qwen3.6-27B (dense coder)](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **~13 GB** | **~14 tok/s (resident) · fits 48 GB, SWE-bench coder** |
| **[Qwen3-235B-A22B-Instruct-2507 (hybrid)](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **70.5 GB** | **~4–6 tok/s (64 GB, 40 GB cache) · converts + streams on a 16 GB Mac mini** |
| **[Qwen3-235B-A22B-Instruct-2507 (full 3-bit)](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **103 GB** | **~1.3 tok/s (64 GB, 40 GB cache) · recall-critical sibling, passes 6/6 stress** |
| **Nemotron-3-Nano-4B** | **TurboQuant** | **3** | **—** | **~2.2 GB** | **75.6 tok/s** |
| Nemotron-3-Super-120B-A12B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Nemotron-3-Super-120B-A12B** | **TurboQuant** | **3** | **—** | **~50 GB** | **18.7 tok/s** |
| **[Nemotron-3-Super-120B-A12B (hybrid for 48GB)](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **~36 GB** | **~27.2 tok/s** |

## Key Results — KV Cache Compression

| Model | KV cache config | KV size | Speed | Notes |
|-------|----------------|---------|-------|-------|
| GPT-OSS-20B (FP16 weights) | FP16 KV | 27.0 MB | 90.6 tok/s | baseline |
| GPT-OSS-20B (FP16 weights) | TQ 3-bit KV | 7.79 MB | 29.9 tok/s | **3.5x cache savings** |
| GPT-OSS-120B (TQ 3-bit weights) | FP16 KV | 45.0 MB | 6.4 tok/s | baseline |
| **GPT-OSS-120B (TQ 3-bit weights)** | **TQ 3-bit KV** | **11.83 MB** | **8.7 tok/s** | **3.8x cache savings — and *faster* than FP16** |
| GPT-OSS-120B (TQ 3-bit weights) | TQ 4-bit KV | 12.21 MB | 16.0 tok/s | also clean |
| Qwen3.5-122B (TQ 3-bit weights) | FP16 KV | 161.06 MB | 5.4 tok/s | baseline |
| **Qwen3.5-122B (TQ 3-bit weights)** | **TQ 3-bit KV** | **150.17 MB** | **5.7 tok/s** | output identical to FP16 |

KV cache compression projects to ~7 GB RAM saved at 131K context on GPT-OSS-120B and ~5 GB at 262K on Qwen3.5-122B. Roundtrip cosine similarity vs FP16: 0.983 at 3-bit, 0.995 at 4-bit.

> **On the Qwen3.5-122B KV rows:** these were measured with **symmetric 3-bit KV** (`demo_kv.py --tq-bits 3`, short prompt), and the rates sit well below the model's fully-resident decode. With the Metal wired cap raised so the ~50 GB model stays fully resident, decode with the recommended **mixed K8/V3** cache is ~24–25 t/s — and there **fp16 edges out compression** at short-to-moderate context, the gap widening as context grows. On this model KV compression is a *memory* win, not a speed win; the genuine decode speed-up is **GPT-OSS-120B-specific**. See the resident long-context sweep in [#19](https://github.com/manjunathshiva/turboquant-mlx/pull/19).

## Key Results — Apple M5 Pro (48 GB, Metal4)

First Metal4 / `MTLGPUFamilyApple10` data point, contributed by [@sbayer2](https://github.com/sbayer2) (#14, #16) — a new 48 GB tier between the 16 GB Mac mini and the 64 GB M4 Max. Reproduce with [`benchmarks/bench_m5_pro.py`](benchmarks/bench_m5_pro.py).

| Model | Mode | Gen t/s | Peak Memory | Notes |
|-------|------|---------|-------------|-------|
| [Qwen3.6-35B-A3B](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) tq3-g32 | resident | **52.0** | 18.1 GB | fp16 KV |
| [Nemotron-3-Super-120B-A12B](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32) tq3a/tq2e-g32 | resident | **20.2** | 41.1 GB | needs `sudo sysctl iogpu.wired_limit_mb=49152` |
| [Qwen3.5-122B-A10B](https://huggingface.co/manjunathshiva/qwen3.5-122b-tq3) tq3 | streaming (30 GB cache) | **9.4** (7.3 e2e) | 34.2 GB | 89.9% expert hit-rate |

**KV cache sweep on the 35B (resident, 3-run averages):**

| KV config | Prompt t/s | Gen t/s | Peak |
|-----------|-----------|---------|------|
| fp16 baseline | 47.8 | **52.0** | 18.131 GB |
| K8 / V3 | 75.5 | 45.7 | 18.123 GB |
| K8 / V3 + sink128 | **76.1** | 45.7 | 18.124 GB |
| K3 / V3 | 75.6 | 45.2 | 18.122 GB |

**122B expert streaming — parallel prefetch** (`--cache-budget-gb 30`, 256 tokens):

| `--prefetch-workers` | Gen t/s | E2E t/s | Disk read | Hit rate |
|----------------------|---------|---------|-----------|----------|
| 1 (serial) | 7.4 | 5.7 | 44.3 GB | 89.5% |
| **8 (parallel)** | **9.1** | **7.6** | 41.9 GB | 90.1% |
| **Speedup** | **1.23×** | **1.33×** | | |

A 1.23× decode speedup from `--prefetch-workers 8`, landing between the Mac mini (1.3×) and the M4 Max (1.67×). The M5 Pro MacBook Pro SSD is the limiter — parallel prefetch helps but doesn't saturate the way the M4 Max's higher-bandwidth SSD does.

**122B expert streaming — cache-budget sweep** (`--prefetch-workers 8`, 256 tokens):

| Budget | Hit rate | Gen t/s | E2E t/s | Peak Metal | Disk read |
|--------|----------|---------|---------|------------|-----------|
| 20 GB | 80.9% | 7.1 | 6.1 | 24.2 GB | 80.7 GB |
| **30 GB** | **90.3%** | **9.1** | **7.6** | 34.2 GB | 40.9 GB |
| 38 GB | 91.0% | 8.9 | 7.5 | 42.2 GB | 38.0 GB |

The hit-rate curve flattens hard past 30 GB (only +0.7% for +8 GB), and throughput actually dips at 38 GB as peak Metal (42.2 GB) crowds the wired cap. **30 GB is the sweet spot on the 48 GB tier** — 90%+ hit rate with ~14 GB of headroom for OS stability; pushing to 38 GB gives negligible gain while peaking uncomfortably close to the wired limit.

KV compression gives a consistent **~1.6× prompt-processing speedup** for a ~12% decode cost (long-context decode behavior is in [The speed flip](#the-speed-flip)). Expert-streaming hit-rate scales with the cache budget — **44.6% at 4 GB (16 GB mini) → 89.9% at 30 GB (48 GB)**, a ~7× throughput jump that fills the gap between the 16 GB and 64 GB tiers.

> **Stability near the memory ceiling:** long-context *prompt prefill* close to the wired cap can starve the kernel watchdog (a `watchdogd` / `AppleARMWatchdogTimer` panic). A rapidly-growing KV cache makes Metal commit pages continuously (`AGXG17XFamilyResidencySet _commitAddedAllocations`), so the binding limit is allocation *rate*, not peak — a static 41 GB resident model (Nemotron) is stable, while a growing ~30 GB KV during a 63K-token prefill can panic. Practical limits on the 48 GB tier (#14): contexts up to ~14.5K are safe on the 35B with either KV config; 63K is feasible with K8/V3 (28.7 GB peak) but not fp16 (30.8 GB → panic). Keep headroom and close other apps for long-context runs near the cap.

## Install

```bash
pip install turboquant-mlx-full
```

The package is published as `turboquant-mlx-full` on PyPI, but importable as
`turboquant_mlx` (without the `-full` suffix) — this matches the original
project name and the examples in the Medium articles.

```python
import turboquant_mlx
from turboquant_mlx.layers import TurboQuantKVCache, convert_cache_to_turboquant
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 64 GB unified memory recommended for 20B+ models

The Metal kernels are JIT-compiled by MLX at first use, so no Xcode / CMake
toolchain is required to install the package.

### Install from source (for development)

```bash
git clone https://github.com/manjunathshiva/turboquant-mlx.git
cd turboquant-mlx
pip install -e .
```

For evaluation utilities (perplexity benchmarking), also install the optional
dependencies:

```bash
pip install "turboquant-mlx-full[eval]"
```

## Quick Start

### 1. Convert a model to TurboQuant format

```bash
# Dense model (e.g., LLaMA 3.2 1B at 3-bit)
python -m turboquant_mlx.convert \
    --hf-path meta-llama/Llama-3.2-1B \
    --mlx-path ./llama-3.2-1b-tq3 \
    --bits 3 --group-size 64

# MoE model (e.g., GPT-OSS-20B at 2-bit)
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq2 \
    --bits 2 --group-size 64

# Very large model whose quantized form won't fit in RAM (200B+): --streaming
# writes each layer to a shard and frees it, so peak memory stays ~one shard
# (5 GB) + one layer — letting 235B/671B-class MoEs convert on a 64 GB Mac.
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path ./qwen3-235b-tq3 \
    --bits 3 --group-size 64 --streaming
```

### 2. Generate text

```bash
turboquant-generate \\
    --model ./gpt-oss-20b-tq2 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200
```

### 3. Evaluate perplexity

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-20b \
    --bits 2 3 4 \
    --num-samples 256 --seq-len 512
```

### 4. Generate with KV cache compression

The production `turboquant-generate` CLI accepts KV-cache flags directly (v0.2+).
Use mixed K/V precision (`--kv-k-bits 8 --kv-v-bits 3`) — required for
TurboQuant-quantized weights, and lossless on stock fp16 weights:

```bash
# v0.2 recommended default: mixed K8/V3 + 128-token fp16 sink
turboquant-generate \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 1024 --temp 0.7 \
    --kv-k-bits 8 --kv-v-bits 3 --kv-min-tokens 128

# Symmetric (legacy) — only safe on fp16 weights
turboquant-generate \
    --model openai/gpt-oss-20b \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --kv-bits 3

# Side-by-side comparison harness (4 configs in one run)
python -m turboquant_mlx.benchmarks.demo_kv_v02 \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 1024 --temp 0.7 --top-p 0.9 --repetition-penalty 1.1
```

### 5. Serve a TurboQuant model over an OpenAI-compatible API

`turboquant-serve` wraps `mlx_lm.server` and patches its loader so any
TurboQuant model (`quantization.mode = "turboquant"` in `config.json`)
loads through the PolarQuant path. Non-TurboQuant models pass through
unchanged, so this is a drop-in replacement for `mlx_lm.server`.

```bash
# Serve a local TQ model
turboquant-serve \
    --model ./NVIDIA-Nemotron-3-Super-120B-A12B-BF16-tq3 \
    --port 8080

# Or serve directly from the Hugging Face Hub
turboquant-serve \
    --model manjunathshiva/Nemotron-3-Super-120B-A12B-tq3 \
    --port 8080
```

Then call it like any OpenAI-compatible endpoint. The `model` field in
the request must match the string passed to `--model`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./NVIDIA-Nemotron-3-Super-120B-A12B-BF16-tq3",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 4096,
    "temperature": 0.7
  }'
```

For Nemotron-3 reasoning models, prefer `max_tokens >= 2048` so the
`<think>` trace and the final answer both fit. mlx-lm splits them into
`message.reasoning` (the thinking) and `message.content` (the answer).

From Python via the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="./NVIDIA-Nemotron-3-Super-120B-A12B-BF16-tq3",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    max_tokens=4096,
    temperature=0.7,
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

All `mlx_lm.server` flags forward unchanged — see `turboquant-serve --help`
for `--host`, `--temp`, `--top-p`, `--prompt-cache-size`, etc.

> **Note**: `mlx_lm.server` is intended for development and local use, not
> production. It does not implement authentication or rate limiting.

#### Memory tuning when serving near the unified-memory ceiling

Serving a 50 GB model on a 64 GB Mac (or any TQ model that fills most of
RAM on a 48 GB / 96 GB Mac) leaves very little headroom for Metal command
buffers and accumulating prompt caches. After 3-4 multi-turn requests the
server can crash with:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

`mlx_lm.server` keeps a **persistent prompt cache per role/conversation**
to speed up follow-up turns. Each new prompt grows that pool, and once
caches + model weights + decode workspace exceed Metal's wired-memory
budget, the next allocation aborts the process.

Two fixes, in order of impact:

**1. Raise Metal's wired-memory ceiling** (biggest lever, requires sudo,
resets on reboot):

```bash
# 64 GB Mac → leave ~7 GB for macOS
sudo sysctl iogpu.wired_limit_mb=57344

# 48 GB Mac → leave ~5 GB for macOS
sudo sysctl iogpu.wired_limit_mb=43008
```

To make it permanent, append `iogpu.wired_limit_mb=57344` to
`/etc/sysctl.conf`.

**2. Cap the prompt cache** (works without sudo, evicts oldest cached
prompts to stay under the cap):

```bash
turboquant-serve \
    --model ./NVIDIA-Nemotron-3-Super-120B-A12B-BF16-tq3 \
    --port 8080 \
    --prompt-cache-bytes 2147483648    # 2 GB hard cap
```

Tighter caps (`536870912` = 512 MB, or `--prompt-cache-size 1` to keep
only one sequence) trade follow-up prefix-cache speedup for stability.

Recommended combo for **Nemotron-3-Super-120B-A12B-tq3 on a 64 GB Mac**:

```bash
sudo sysctl iogpu.wired_limit_mb=57344
turboquant-serve \
    --model ./NVIDIA-Nemotron-3-Super-120B-A12B-BF16-tq3 \
    --port 8080 \
    --prompt-cache-bytes 2147483648
```

Also close any other GPU users (Chrome/Electron apps, Final Cut, Xcode
simulators) before launching — even an idle Chrome can be holding 1-2 GB
of unified memory.

---

## KV Cache Compression

TurboQuant KV cache compression applies the same Hadamard rotation + Lloyd-Max codebook pipeline to KV vectors at runtime. The compressed cache is dequantized to float16 only when attention needs it, so it routes through MLX's standard `scaled_dot_product_attention` and is compatible with attention sinks, sliding windows, and linear attention layers.

### Programmatic usage

```python
from turboquant_mlx.layers import convert_cache_to_turboquant
from mlx_lm.models.cache import make_prompt_cache

# 1. Build per-layer cache (correct types for hybrid models)
cache = make_prompt_cache(model)

# 2. Convert to TurboQuant KV cache (v0.2 mixed K/V + sink protection)
cache = convert_cache_to_turboquant(
    cache,
    k_bits=8, v_bits=3,           # K-precision-critical, V tolerates 3-bit
    min_tokens_before_quant=128,  # keep first 128 tokens fp16 (attention sinks)
    group_size=64,
)

# 3. Process the prompt and generate — cache is compressed from token 128+
model(prompt_tokens, cache=cache)
for token in generate_loop(model, cache):
    ...
```

> **v0.1 → v0.2 migration:** `tq_bits=3` still works (symmetric K=V=3) but is
> not recommended on TurboQuant-quantized weights. Pass `k_bits=8, v_bits=3`
> instead. Pre-existing checkpoints and code paths are fully backward compatible.

### Choosing a bit-width (v0.2)

K precision matters far more than V precision: softmax amplifies any K error,
while V tolerates aggressive quantization. Mixed K8/V3 is the new default.

| Weights | K bits | V bits | sink | When to use |
|---------|-------:|-------:|-----:|-------------|
| FP16 / BF16 | 8 | 3 | 128 | Default — lossless quality, ~4× smaller cache |
| FP16 / BF16 | 4 | 3 | 128 | More aggressive; small quality dip on dense attention |
| **TurboQuant-quantized** | **8** | **3** | **128** | **Required on tq3 weights — symmetric K3 collapses past ~1k generated tokens** |
| Any | 8 | 4 | 128 | Highest fidelity TQ KV setting |

**Why K8 specifically on TurboQuant weights:** stacking 3-bit K cache on top of
already-3-bit weight quantization compounds the noise enough to break long-form
generation on GPT-OSS-20B (we observed total output collapse past ~800 tokens
with `K3_V3` on tq3 weights, while `K8_V3` is clean). The same `K3_V3` cache is
fine on stock fp16 weights — the failure mode is co-compression, not the cache
alone.

### CLI flags

`turboquant-generate` exposes the same controls:

```bash
turboquant-generate --model ./model-tq3 --prompt "..." \
    --kv-k-bits 8 --kv-v-bits 3 \
    --kv-min-tokens 128 \
    --kv-group-size 64
```

| Flag | Purpose |
|------|---------|
| `--kv-bits N` | Symmetric K=V=N (legacy v0.1) |
| `--kv-k-bits` / `--kv-v-bits` | Mixed precision (v0.2 recommended) |
| `--kv-min-tokens N` | Keep the first N cached tokens in fp16 (sink protection) |
| `--kv-group-size N` | Hadamard rotation group size (default 64) |

### The speed flip

Whether KV compression speeds up or slows down decode depends on the **per-token KV cache size**, not the parameter count. When the per-token KV is large (many KV heads and/or long context), its 4x smaller footprint cuts memory bandwidth more than dequant adds, and decode is *faster* than FP16. When it is small (few active params, short context), dequant overhead dominates and compression is *slower* — a pure memory optimization.

| Model | FP16 KV | TQ 3-bit KV | Direction |
|-------|---------|-------------|-----------|
| GPT-OSS-20B | 90.6 tok/s | 29.9 tok/s | TQ is 3x **slower** |
| Qwen3.6-35B-A3B (3B active) | 52.0 tok/s | 45.7 tok/s | TQ is 1.1x **slower** |
| GPT-OSS-120B | 6.4 tok/s | 8.7 tok/s | TQ is 1.4x **faster** |

The penalty also **grows with context** on small-KV models. A community long-context benchmark on M5 Pro (#14) measured Qwen3.6-35B-A3B-tq3-g32 decode at four context lengths (256 gen tokens, except 63K which used 128):

| Context | KV Config | Prompt t/s | Decode t/s | Peak Metal | Memory Saved |
|---------|-----------|-----------|-----------|------------|--------------|
| ~65 tok | fp16 | 47.8 | 52.0 | 18.13 GB | — |
| ~65 tok | K8/V3 | 76.1 | 45.7 | 18.12 GB | 0.01 GB |
| ~2.5K tok | fp16 | 131.9 | 51.5 | 20.50 GB | — |
| ~2.5K tok | K8/V3 | 133.4 | 38.2 | 20.55 GB | -0.05 GB |
| ~14.5K tok | fp16 | 132.8 | 46.0 | 22.50 GB | — |
| ~14.5K tok | K8/V3 | 132.6 | 16.7 | 21.94 GB | 0.56 GB |
| ~63K tok | fp16 | 124.5 | 34.5 | 30.79 GB | — |
| ~63K tok | K8/V3 (no sink) | 123.6 | 5.2 | 28.71 GB | 2.08 GB |

The fp16 decode advantage *widens* with context — 1.14× at 65 tokens → 1.35× at 2.5K → 2.75× at 14.5K → **6.6× at 63K**. So on small-active MoEs, use KV compression to *fit* longer contexts in less RAM (2.08 GB saved at 63K) — not to speed them up. The flip to *faster* shows up only on **GPT-OSS-120B** (8.7 vs 6.4 t/s). The similarly-sized **Qwen3.5-122B does *not* flip** — run resident on a 64 GB M4 Max, fp16 KV beats mixed K8/V3 at every context (1.07× at 256 → 1.20× at 4096; [#19](https://github.com/manjunathshiva/turboquant-mlx/pull/19)) — so the speed-up isn't a general 120B-class property; it's specific to GPT-OSS-120B's KV geometry.

### Compatibility

| Feature | Supported | Notes |
|---------|-----------|-------|
| Attention sinks | Yes | GPT-OSS sink vectors flow through standard SDPA |
| Sliding window attention | Yes | `RotatingKVCache` layers are left untouched |
| Linear attention | Yes | `ArraysCache` (Qwen3.5 GatedDeltaNet) is left untouched |
| Hybrid architectures | Yes | Per-layer cache type is preserved |
| Prompt-first conversion | Yes | Process prompt with FP16, convert before generation |

---

## Running GPT-OSS MoE Models on Apple Silicon

### GPT-OSS-20B (21B total, 32 experts, 3.6B active)

**Hardware:** Apple M4 Max 64GB (or any Apple Silicon with 16GB+ unified memory at 3-bit)

#### Step 1: Convert to TurboQuant 3-bit (recommended)

```bash
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq3 \
    --bits 3 --group-size 32
```

**Model size:** 9.3 GB (vs 12.8 GB MXFP4 original — 28% smaller, lower perplexity)

The converter automatically:
- Detects MoE architecture (SwitchLinear / QuantizedSwitchLinear layers)
- Dequantizes MXFP4 expert weights to float
- Applies Hadamard rotation + Lloyd-Max codebook quantization
- Keeps router weights and attention at full precision
- Handles blockwise Hadamard for 2880-dim experts (2880 = 9 x 320)

#### Step 2: Generate text

```bash
turboquant-generate \\
    --model ./gpt-oss-20b-tq3 \
    --prompt "Explain quantum entanglement to a 10-year-old." \
    --max-tokens 256
```

**Expected:** ~73 tok/s generation, ~85 tok/s prefill on M4 Max

#### Step 3: Run a quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-20b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 64 --seq-len 512
```

#### All bit-widths for GPT-OSS-20B

| Method | Bits | Size | Peak RAM | Gen Speed | Quality |
|--------|------|------|----------|-----------|---------|
| Affine (mlx-lm) | 4 | 11.2 GB | ~14 GB | 148 tok/s | Coherent (but see note below) |
| TurboQuant | 4 | 11.2 GB | ~14 GB | — | Best (PPL 72.63, beats MXFP4) |
| **TurboQuant** | **3** | **9.3 GB** | **~12 GB** | **73 tok/s** | **Recommended (PPL 78.60, beats MXFP4, coherent)** |
| TurboQuant | 2 | 7.5 GB | ~10 GB | — | Poor (incoherent generation on pre-quantized models) |

> **Speed vs quality tradeoff:** Affine 4-bit is ~2x faster on the 20B model due to simpler dequantization, but TurboQuant 3-bit is 28% smaller with lower perplexity than both affine 4-bit and OpenAI's own MXFP4. Crucially, affine 4-bit **cannot scale to 120B** on 64GB hardware — TurboQuant 3-bit is the only option there.

```bash
# 4-bit (best quality, beats OpenAI's MXFP4)
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq4 \
    --bits 4 --group-size 32
```

---

### GPT-OSS-120B (120B total, 128 experts, ~13B active)

**Hardware:** Apple M4 Max 64GB — neither the original MXFP4 (63.5 GB) nor the [mlx-community 4-bit affine](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) (65.8 GB) fit on a 64GB machine. TurboQuant 3-bit is the only way to run this model on consumer hardware.

#### Step 1: Convert to TurboQuant 3-bit (recommended)

```bash
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-120b \
    --mlx-path ./gpt-oss-120b-tq3 \
    --bits 3 --group-size 64
```

**Model size:** 48 GB

> **Note:** The default converter materializes the full quantized model in RAM before saving, so peak memory ≈ the quantized model size (~50–55 GB for a 120B). On a 64 GB machine that caps conversion at ~130B params. For anything larger, add **`--streaming`**: it writes each quantized layer to a shard and frees it, keeping peak memory to ~one 5 GB shard plus the layer being processed — so 200B+ models (Qwen3-235B, DeepSeek-V3) convert on a 64 GB Mac. Output is byte-identical to the in-memory path.

#### Step 2: Generate text

```bash
turboquant-generate \\
    --model ./gpt-oss-120b-tq3 \
    --prompt "Explain quantum computing in simple terms." \
    --max-tokens 200
```

**Expected:** ~44 tok/s generation, ~9.5 tok/s prefill, 52 GB peak memory on M4 Max 64GB

#### Step 3: Quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-120b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 32 --seq-len 512
```

#### All bit-widths for GPT-OSS-120B

| Method | Bits | Size | Peak RAM | Gen Speed | Fits 64 GB? | Quality |
|--------|------|------|----------|-----------|-------------|---------|
| [mlx-community 4-bit](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 (affine) | 65.8 GB | — | — | **No** | — |
| MXFP4 (original) | 4 (mxfp) | 63.5 GB | ~70 GB | — | **No** | — |
| **TurboQuant** | **3** | **48 GB** | **52.3 GB** | **44 tok/s** | **Yes** | **Coherent, well-structured** |
| TurboQuant | 2 | 32 GB | 34.9 GB | 51 tok/s | Yes | Incoherent after ~20 tokens |

> Neither the original MXFP4 format (63.5 GB) nor the mlx-community affine 4-bit re-quantization (65.8 GB) fit on a 64GB Mac. TurboQuant 3-bit (48 GB) is the **only** way to run GPT-OSS-120B on consumer hardware — and at 44 tok/s, it's interactive speed. At 2-bit, the model fits easily but generation quality degrades rapidly — **3-bit is the minimum for coherent output on pre-quantized MoE models.**

---

### Qwen3.5-122B-A10B (122B total, 256 experts, 8 active, ~10B active)

**Hardware:** Apple M4 Max 64GB — the original BF16 model is ~240 GB. TurboQuant 3-bit compresses it to ~50 GB, fitting on a 64GB machine.

This is a brand-new architecture featuring **256 MoE experts** (the most of any model we've tested), **hybrid attention** (GatedDeltaNet linear attention + standard softmax attention), and **thinking/reasoning** capability. The model also has a shared expert per layer alongside the routed experts.

#### Step 1: Convert to TurboQuant 3-bit

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3.5-122B-A10B \
    --mlx-path ./qwen3.5-122b-tq3 \
    --bits 3 --group-size 64
```

**Model size:** ~50 GB | **Conversion time:** ~90 seconds

> **Note:** Conversion requires ~55 GB peak memory. Close all other applications before running. The converter uses memory-efficient processing — each expert layer is replaced immediately after quantization with aggressive garbage collection to handle the 256 experts per layer.

#### Step 2: Generate text

```bash
turboquant-generate \\
    --model ./qwen3.5-122b-tq3 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200
```

**Expected:** ~26.5 tok/s generation, 55 GB peak memory on M4 Max 64GB

#### Benchmark

| Method | Bits | Size | Peak RAM | Gen Speed | Fits 64 GB? | Quality |
|--------|------|------|----------|-----------|-------------|---------|
| BF16 (original) | 16 | ~240 GB | — | — | **No** | — |
| **TurboQuant** | **3** | **~50 GB** | **54.9 GB** | **26.5 tok/s** | **Yes** | **Coherent reasoning with structured thinking** |

> Qwen3.5-122B-A10B is the largest and most complex model TurboQuant has been tested on: 122B parameters, 256 experts (8 active per token), hybrid GatedDeltaNet + softmax attention, and a shared expert per MoE layer. At 3-bit, the model produces structured reasoning with proper analysis steps — demonstrating that TurboQuant preserves thinking capability at extreme compression.

#### Run it on a 16 GB Mac mini (expert streaming)

This 122B model — ~54 GB on disk — also runs on a **16 GB Mac mini** via expert streaming (the same mechanism as [Qwen3.6-35B-A3B](#qwen36-35b-a3b-on-a-16-gb-mac-mini-expert-streaming)). Only the router-selected experts are paged from disk per token (LRU-cached), so the resident footprint stays well under the machine's GPU wired-memory cap, and output is bit-identical to the fully-resident model. Requires `turboquant-mlx-full>=0.4.1`.

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/qwen3.5-122b-tq3 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 128 --cache-budget-gb 4
```

Measured on a **base Apple M4 Mac mini, 16 GB**:

| Cache budget | Expert hit-rate | Disk read / token | Decode | Peak (mlx) |
|---|---|---|---|---|
| `--cache-budget-gb 1` | 0% | ~1.78 GB | ~0.6 tok/s | 6.0 GB |
| `--cache-budget-gb 4` *(recommended)* | **44.6%** | ~0.93 GB | **~1.1 tok/s** | 9.0 GB |

On a 16 GB machine the binding limit is the **Metal GPU wired-memory cap (~10.5 GB)**, not total RAM — and the expert cache counts against it, so `mlx_peak ≈ 5 GB + cache_budget`. `--cache-budget-gb 4` is the sweet spot (~9 GB peak, safe margin); higher budgets risk a Metal out-of-memory error. Throughput is disk-bandwidth-bound (~10B active params/token) → ~1 tok/s on a single mini SSD. Slow, but **a 122B model running on a 16 GB Mac** is the result.

---

### Qwen3.6-35B-A3B on a 16 GB Mac mini (expert streaming)

**Hardware:** Apple M4 Max 64GB to convert; runs **fully resident on 64 GB** or on a **16 GB Mac mini via expert streaming**. Qwen3.6-35B-A3B is a hybrid linear-attention (`qwen3_5_moe`, qwen3_next-style) + MoE model — **256 routed experts (top-8) + 1 shared**, ~35B total / ~3B active. The text-only language model is extracted (the vision tower is dropped during conversion).

A pre-converted 3-bit (group-size 32) model is on the Hub:

→ [`manjunathshiva/Qwen3.6-35B-A3B-tq3-g32`](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) — ~16 GB on disk; ~60 tok/s at ~18 GB peak when fully resident on a 64 GB Mac.

#### Run it fully resident (64 GB)

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512
```

#### Run it on a 16 GB Mac mini (expert streaming)

The model is ~16 GB on disk, so it won't fit fully resident in 16 GB alongside the OS (resident decode peaks ~18 GB). Expert streaming pages only the router-selected experts from disk per token (LRU-cached), keeping resident memory to a few GB. Output is **bit-identical** to the fully-resident model. (`os.pread` + macOS `F_NOCACHE` keep the OS page cache from ballooning while streaming.)

Since `v0.5.0` the missing experts for each layer are read **in parallel** on a thread pool (`--prefetch-workers`, default `8`), hiding SSD latency behind compute — ~1.9× faster decode at a tight cache budget, still bit-identical. Pass `--prefetch-workers 1` for the serial baseline.

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 8
```

#### Benchmark (base Apple M4 Mac mini, 16 GB)

| Config | Expert hit-rate | Disk read / token | Decode | Peak RSS |
|--------|-----------------|-------------------|--------|----------|
| `--cache-budget-gb 2` | ~60% | ~175 MB | ~3.0 tok/s | **3.9 GB** |
| `--cache-budget-gb 8` *(recommended)* | **91%** | ~41 MB | **~4.5 tok/s** | 9.4 GB |

A larger cache keeps more experts resident, raising the hit-rate and cutting SSD reads — the throughput limiter when streaming. `--cache-budget-gb 8` is the sweet spot on a 16 GB machine; drop to `2` if RAM is tight. Streaming targets the SwitchGLU expert layout used by `qwen3_5_moe` and the DeepSeek MLA+MoE family (`deepseek_v2`/`v3`); the loader auto-detects the model's layer-key prefix.

> **Note:** Qwen3.6 is a thinking-mode model — it emits a reasoning trace before the final answer, so give it a generous `--max-tokens` (512+) for tasks that need a concluding answer.

#### Tuning the streaming reader (`v0.6.1`)

Once the cache policy is reasonable, **disk bandwidth is the wall** — for MoE decode the LRU + 8-worker parallel-read pool is already near-optimal, so the big levers are faster storage (Thunderbolt/NVMe) and fewer bytes/token (a hybrid build, a bigger `--cache-budget-gb`), not the read algorithm. A few knobs squeeze the rest:

| Knob | Default | What it does |
|------|---------|--------------|
| read-coalescing | **on** | Merges contiguous missed experts into one `os.pread`. Bit-identical, free, ~5% faster when disk-bound. No flag. |
| `--prefetch-ahead N` | `0` (off) | Speculatively prefetch the next *N* layers' experts (predicted from the previous token's routing) on a background thread. ~+6% on fast NVMe with spare bandwidth; **self-disables** if the drive proves bandwidth-bound (e.g. a saturated USB bus), so it's safe to set `1`. |
| `--pin-file pin.json` | none | Keep a calibrated hot-expert set permanently resident. **Experimental** — measured net-negative vs pure LRU on a 122B (static pinning costs LRU's adaptivity). For experimentation only. |

Generate `pin.json` (and a co-activation `perm.json` for the optional `stream/repack_experts.py` relayout) with `python -m turboquant_mlx.stream.calibrate_experts`.

---

### Qwen3.6-27B (dense coding model for a 48 GB Mac)

**Hardware:** converts on a 64 GB Mac — or off slow USB storage with `v0.6.2+`, which forces the disk read ahead of GPU compute so conversion doesn't trip the Metal GPU watchdog — and runs **fully resident** on a **48 GB** Mac with headroom to spare. [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) is a **dense** (`qwen3_5`) long-context coder — 64 layers, hybrid attention (**48 GatedDeltaNet linear-attention + 16 full-attention** layers), head_dim 256, 262K context — that Qwen positions as competitive on SWE-bench Verified / SWE-bench Pro. Being dense, it has **no experts to stream**: it loads once and stays in RAM, so storage only matters for load time.

A pre-converted 3-bit (group-size 32) build is on the Hub:

→ [`manjunathshiva/Qwen3.6-27B-tq3-g32`](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32) — **~13 GB on disk**, **~17.5 GB peak** at runtime (fits 48 GB with ~30 GB free for KV), ~14 tok/s decode.

#### Run it

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-27B-tq3-g32 \
    --prompt "Write a Python function that merges overlapping intervals." \
    --max-tokens 512 --temp 0.7
```

#### Serve it to Cursor / VS Code (OpenAI-compatible)

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-27B-tq3-g32 --port 8080
```

Point the IDE's custom OpenAI base URL at `http://localhost:8080/v1`. Stock `mlx_lm.server` **can't** load a TurboQuant model (`KeyError: 'turboquant'`) — `turboquant-serve` patches the loader so the weights load through the PolarQuant path.

#### Convert it yourself

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3.6-27B \
    --mlx-path ./Qwen3.6-27B-tq3-g32 \
    --bits 3 --group-size 32 --streaming
```

> **Note:** Qwen3.6 is a thinking-mode model — it emits a reasoning trace before the answer, so give it a generous `--max-tokens` (512+). Only the **16 full-attention layers** keep a growing KV cache (the 48 linear-attention layers use a fixed-size state), so it stays KV-light for long coding context; compress further with `--kv-k-bits 8 --kv-v-bits 3`.

---

### Qwen3-235B-A22B-Instruct-2507 — a 235B MoE that converts on a 16 GB Mac (hybrid + streaming)

**Hardware:** converts on a **16 GB Mac mini** via `--streaming`; runs on a **64 GB Mac** (expert streaming) or fully resident on **96 GB+**. Qwen3-235B-A22B is a `qwen3_moe` Mixture-of-Experts — **94 layers, 128 routed experts (top-8)**, ~235B total / ~22B active.

This is a **hybrid tq3a-tq2e** build: **3-bit attention** (the always-on path, kept safer) + **2-bit experts** (where the parameters — and the savings — live), routers full precision. The 128-expert / top-8 routing carries enough redundancy to absorb 2-bit experts cleanly — the same reason gpt-oss-120b holds at 2-bit while gpt-oss-20b (32 experts) collapses. Result: **~470 GB BF16 → 70.5 GB** (15 shards, 6.7×).

A pre-converted build is on the Hub:

→ [`manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32`](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32)

#### Convert it yourself (streaming, fits in ~8–12 GB RAM)

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path /Volumes/SSD/qwen3-235b-tq3a-tq2e-g32 \
    --bits 3 --mlp-bits 2 -g 32 --streaming
```

`--mlp-bits 2` drops experts to 2-bit while `--bits 3` keeps attention at 3-bit; `--streaming` writes each quantized layer to a shard and frees it, so the full 235B converts in **~8–12 GB of RAM** — it was produced on a **16 GB Mac mini in ~18 minutes**. Point `--mlx-path` at a drive with ≥70 GB free.

#### Run it (expert streaming)

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 40
```

#### Quality + streaming benchmark

A 6-probe stress run passes **5/6**: coherent long-form essay, **correct multi-step math** ($142.80 with a 15% bulk discount), correct memoized Fibonacci, strict-JSON formatting, and clean 1–15 enumeration. The one miss was **exact factual recall** — an in-context password came back with a single flipped digit (`RAVEN-stone-91` → `-51`). Math/reasoning held; verify outputs where an exact literal value matters.

> **Need exact recall?** The full-3-bit sibling [**Qwen3-235B-A22B-Instruct-2507-tq3-g32**](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32) (3-bit experts, 103 GB) fixes the needle flip and passes **6/6** — at ~1.3 tok/s / 86.3% hit-rate on a 64 GB Mac (slower and bigger than this hybrid; the cost of full 3-bit experts). Pick the hybrid for the smallest footprint, the tq3 build when exact literal recall matters.

| Machine | Cache budget | Expert hit-rate | Disk read / token | Decode | Peak memory |
|---------|-------------|-----------------|-------------------|--------|-------------|
| M4 Mac mini, 16 GB | `--cache-budget-gb 6` | ~38% | ~3.2 GB | ~0.2 tok/s | 10.1 GB |
| 64 GB Mac | `--cache-budget-gb 40` | **94.1%** | ~0.28 GB | **~4–6 tok/s** (warm) | 46 GB |

On 64 GB a 40 GB cache holds ~60% of the ~67 GB of experts, but temporal locality lifts the hit-rate to **94.1%**, so warm decode runs at the compute-bound ~4–6 tok/s. Throughput is **bursty**: the first generation and tasks that route into a colder slice of experts stall on the SSD until their experts page in. Bump `sudo sysctl iogpu.wired_limit_mb=57344` to raise the cache past the ~48 GB default Metal wired cap.

---

### Nemotron-3 (Mamba/attention hybrid)

Nemotron-3 is NVIDIA's hybrid Mamba2 + attention architecture. Two variants are tested:

- **Nano-4B** — dense (Mamba + MLP + attention), 42 layers
- **Super-120B-A12B** — hybrid MoE (Mamba + 512-expert latent-MoE + attention), 88 layers, ~12B active per token

Both require **mlx-lm ≥ 0.31.3** for upstream Nemotron-H support (installed automatically).

#### Convert

```bash
# Nano-4B
python -m turboquant_mlx.convert \
    --hf-path nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
    --mlx-path ./nemotron-3-nano-4b-tq3 \
    --bits 3 --group-size 64

# Super-120B
python -m turboquant_mlx.convert \
    --hf-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --mlx-path ./nemotron-3-super-120b-tq3 \
    --bits 3 --group-size 64
```

#### Generate

Nemotron-3's chat template ends in a `<think>\n` scaffold that primes EOS as the top-1 logit at the start of the assistant turn. Pass `--min-tokens` to mask EOS for the first N tokens so the model enters the think phase:

```bash
turboquant-generate \\
    --model ./nemotron-3-super-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --min-tokens 50
```

#### Benchmarks (M4 Max)

| Model | Bits | Size | Peak RAM | Gen Speed | Quality |
|-------|------|------|----------|-----------|---------|
| **Nemotron-3-Nano-4B** | **3** | **~2.2 GB** | **4.3 GB** | **75.6 tok/s** | **Coherent** |
| **Nemotron-3-Super-120B-A12B** | **3** | **~50 GB** | **54.7 GB** | **18.7 tok/s** | **Coherent with structured `<think>` reasoning (974-token answer w/ self-correction, formulas, formatted structure)** |
| **Nemotron-3-Super-120B-A12B (hybrid)** | **3-attn / 2-experts, gs=32** | **~36 GB** | **~40.8 GB** | **~27.2 tok/s** | **Coherent prose, code, format, and long-context recall; math accuracy degraded — see Phase-1 note below** |

#### 48 GB-RAM target: hybrid (3-bit attention / 2-bit experts) at group-size 32

The standard 3-bit Super-120B (~50 GB) needs ~55 GB peak and only fits a 64 GB
Mac after raising `iogpu.wired_limit_mb`. For users on a 64 GB Mac who want
headroom for other applications — or for users on **48 GB Macs** — there is
a **hybrid quantization** that keeps attention at 3-bit (where precision
matters most) and pushes experts to 2-bit (where the bulk of the weights
live), at a smaller group size (g=32) that improves per-group fit.

**Pre-converted model on Hugging Face:**

```bash
hf download manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32 \
    --local-dir ~/models/nemotron-3-super-120b-tq3a-tq2e-g32
```

→ [`manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32`](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32)
on the Hub: ~36 GB on disk, ~40.8 GB peak memory, ~27.2 tok/s decode, fits the
default 48 GB `iogpu.wired_limit_mb` cap.

**Or convert from BF16 source yourself:**

```bash
python -m turboquant_mlx.convert \
    --hf-path nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --mlx-path ./nemotron-3-super-120b-tq3a-tq2e-g32 \
    --bits 2 --attn-bits 3 --mlp-bits 2 --group-size 32
```

For long-form generation, the model needs a small repetition penalty to
avoid degenerate tail loops at >1500 tokens. The recommended decode config
(empirically validated to keep essay, code, format, and long-context
recall clean):

```bash
turboquant-generate \\
    --model ./nemotron-3-super-120b-tq3a-tq2e-g32 \
    --prompt "Why is the sky blue?" \
    --max-tokens 4096 --min-tokens 50 \
    --temp 0.7 --rep-penalty 1.04 --rep-ctx 256
```

> **Phase-1 known limitation: math accuracy.** Step-by-step arithmetic
> on the hybrid degrades under any non-zero `--rep-penalty`. For
> numeric/math prompts in Phase 1, **omit `--rep-penalty`** (you may
> see long-gen tail loops on long prompts, but the arithmetic will land
> correctly more often). A permanent fix is planned for Phase 2 — likely
> first/last-layer bit protection, a calibration-data codebook, or a
> fused QJL Metal kernel. Until Phase 2, use the hybrid for prose,
> coding, format, and long-context tasks; use the standard 3-bit model
> for serious numeric work.

The fused MoE decode kernel transparently chunks expert routings on long
prompts, so this hybrid handles long-context retrieval (e.g. password-
recall over 4000+ tokens of context) without the kernel argument-validation
crash that affected earlier builds.

---

## How It Works

TurboQuant is a two-stage, **calibration-free** quantization pipeline:

1. **Hadamard Rotation** — Multiply weights by a randomized Hadamard matrix, transforming any weight distribution into a near-Gaussian shape. This is data-oblivious (no calibration data needed).

2. **Lloyd-Max Codebook** — Apply information-theoretically optimal quantization for Gaussian distributions. The codebook is a mathematical constant, precomputed once.

The result: near-zero quality loss at 3-bit, and usable 2-bit quantization where standard affine completely breaks down.

For MoE models, all experts within a layer share the same rotation signs and codebook, keeping storage efficient.

## CLI Options

```
python -m turboquant_mlx.convert --help

Options:
  --hf-path TEXT       HuggingFace model path or local path (required)
  --mlx-path TEXT      Output directory (default: mlx_model)
  --bits {2,3,4}       Quantization bit-width (default: 3)
  --group-size {32,64,128}  Elements per quantization group (default: 64)
  --rotation TEXT      Rotation method: hadamard, blockwise_hadamard, none
  --use-qjl           Enable 1-bit QJL residual correction (+1 bit overhead)
  --dtype TEXT         Model dtype before quantization: float16, bfloat16
```

## Supported Architectures

| Architecture | Model Type | MoE | Status |
|-------------|-----------|-----|--------|
| LLaMA / Llama 3 | `llama` | No | Tested |
| Qwen2 / Qwen2.5 | `qwen2` | No | Tested |
| Qwen3.5 | `qwen3_5` | No | Tested |
| Mistral | `mistral` | No | Tested |
| Qwen1.5-MoE | `qwen2_moe` | Yes | Tested |
| GPT-OSS | `gpt_oss` | Yes | Tested |
| Qwen3.5-MoE / Qwen3.6-35B-A3B | `qwen3_5_moe` | Yes (256 experts) | Tested (122B, 35B-A3B); 35B streams on a 16 GB Mac mini |
| Qwen3-MoE | `qwen3_moe` | Yes (128 experts, top-8) | Tested — Qwen3-235B-A22B converted to a hybrid **tq3a-tq2e** build (70.5 GB) on a 16 GB Mac mini via `--streaming`; streams and passes 5/6 quality probes on a 64 GB Mac |
| Nemotron-H (Mamba/attention hybrid) | `nemotron_h` | Yes (512 experts w/ latent MoE on Super-120B) | Tested (Nano-4B, Super-120B) — requires mlx-lm ≥ 0.31.3 |
| DeepSeek-V2 / V3 (MLA + MoE) | `deepseek_v2` / `deepseek_v3` / `deepseek_v32` | Yes (SwitchGLU experts) | Tested (V2-Lite: convert + resident + streaming, coherent at 3-bit); V3/V3.2 share the MLA+MoE layout and reuse the config (untested — need ~250 GB disk) |
| DiffusionGemma (block-diffusion MoE, via **mlx-vlm**) | `diffusion_gemma` | Yes (128 experts, top-8) | Tested (26B-A4B: convert + block-diffusion sampler, coherent at 3-bit — [HF](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32)). **Experimental**: decode is much slower than native 4-bit until a batched codebook gather-GEMM kernel lands |

### mlx-vlm architectures (multimodal / diffusion)

Architectures that live in [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) rather
than mlx-lm convert and run through dedicated entry points (v0.7.0+):

```bash
pip install "turboquant-mlx-full[vlm]"   # adds mlx-vlm >= 0.6.3

# Convert (vision towers, routers, and known quant-sensitive blocks stay full precision)
python -m turboquant_mlx.convert_vlm \
    --hf-path google/diffusiongemma-26B-A4B-it \
    --mlx-path ./diffusiongemma-26B-A4B-it-tq3-g32 --bits 3 -g 32

# Generate (runs mlx-vlm's sampler — block-diffusion denoising for DiffusionGemma)
python -m turboquant_mlx.generate_vlm \
    --model ./diffusiongemma-26B-A4B-it-tq3-g32 \
    --prompt "Write a short paragraph about the ocean." --max-tokens 256
```

## Project Structure

```
turboquant_mlx/
    config.py                 # TurboQuantConfig
    convert.py                # CLI: HF model -> TurboQuant MLX
    generate.py               # Text generation with TurboQuant models
    evaluate.py               # Perplexity evaluation
    quantize_model.py         # Model traversal & layer replacement
    demo_kv.py                # Streaming generation demo with KV cache compression
    test_kv_cache.py          # KV cache roundtrip + integration tests
    core/
        codebook.py           # Lloyd-Max codebooks for Gaussian
        rotation.py           # Randomized Hadamard rotation
        polar_quantize.py     # Rotate + codebook quantize
        packing.py            # Bit-packing into uint32
        qjl.py                # QJL residual correction
    layers/
        polar_linear.py       # PolarQuantizedLinear (dense)
        polar_switch_linear.py # PolarQuantizedSwitchLinear (MoE)
        polar_kv_cache.py     # TurboQuantKVCache (runtime KV compression)
    kernels/
        polar_qmv.py          # Fused Metal kernel (dense decode)
        polar_gather_qmv.py   # Fused Metal kernel (MoE shared input)
        polar_multi_gather_qmv.py  # Fused Metal kernel (MoE per-expert input)
    integration/
        rotation_configs.py   # Per-architecture rotation configs
    stream/                   # Expert streaming — run MoE models beyond RAM (v0.4.0)
        safetensors_reader.py # Per-expert disk slice reads (os.pread + F_NOCACHE; coalesced ranges)
        streaming_switch.py   # StreamingSwitchLinear + byte-budgeted LRU ExpertCache (+ prefetch/pin)
        loader.py             # load_streaming(): swap experts to streaming after lazy load
        stream_generate.py    # CLI: stream-generate (--cache-budget-gb, --prefetch-ahead, --pin-file)
        calibrate_experts.py  # Routing trace → pin.json (hot experts) + perm.json (co-activation)
        repack_experts.py     # Optional co-activation on-disk relayout (byte-identical)
```

## Citation

```bibtex
@misc{turboquant_mlx,
    title={TurboQuant-MLX: Extreme Weight and KV Cache Compression for Apple Silicon},
    year={2025},
    note={MLX implementation of TurboQuant (Zandieh et al., 2025) for both weight quantization and runtime KV cache compression}
}
```

## License

MIT

## Acknowledgments

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (2025)
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Research
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — MLX language model utilities
