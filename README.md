# TurboQuant-MLX

Extreme **weight** and **KV cache** compression for LLMs on Apple Silicon. MLX implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) — Hadamard rotation + Lloyd-Max codebooks applied both to weights (compile time) and the KV cache (run time).

Supports dense models (LLaMA, Qwen, Mistral), **Mixture-of-Experts** (Qwen-MoE, GPT-OSS, Qwen3.5-MoE), and **Mamba/attention hybrids** (Nemotron-3-Nano-4B, Nemotron-3-Super-120B). Compatible with hybrid attention architectures, attention sinks, sliding-window attention, and linear attention layers.

**With both weight and KV cache compression at 3-bit, GPT-OSS-120B fits its full 131K context window in 50 GB on a 64 GB MacBook — and KV cache compression actually makes generation *faster* on the 120B (8.7 vs 6.4 tok/s) because the smaller cache cuts memory bandwidth more than dequant costs.**

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
| GPT-OSS-120B | TurboQuant | 2 | — | 32 GB | 51 tok/s (poor quality) |
| Qwen3.5-122B-A10B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Qwen3.5-122B-A10B** | **TurboQuant** | **3** | **—** | **~50 GB** | **26.5 tok/s** |
| **Nemotron-3-Nano-4B** | **TurboQuant** | **3** | **—** | **~2.2 GB** | **75.6 tok/s** |
| Nemotron-3-Super-120B-A12B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Nemotron-3-Super-120B-A12B** | **TurboQuant** | **3** | **—** | **~50 GB** | **18.7 tok/s** |

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
```

### 2. Generate text

```bash
python -m turboquant_mlx.generate \
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

```bash
# Standard model + KV cache compression
python -m turboquant_mlx.demo_kv \
    --model openai/gpt-oss-20b \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --tq-bits 3

# TQ-compressed model + KV cache compression (full stack)
python -m turboquant_mlx.demo_kv \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --tq-bits 3

# Side-by-side comparison: FP16 KV vs TurboQuant KV
python -m turboquant_mlx.demo_kv \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --compare
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

# 1. Process the prompt with FP16 KV cache (exact)
cache = make_prompt_cache(model)
model(prompt_tokens, cache=cache)

# 2. Convert to TurboQuant KV cache for generation
cache = convert_cache_to_turboquant(cache, tq_bits=3, group_size=64)

# 3. Continue generation — cache is now compressed
for token in generate_loop(model, cache):
    ...
```

### Choosing a bit-width

| Weights | KV cache | Recommendation |
|---------|----------|----------------|
| FP16 / BF16 | TQ 3-bit | Default sweet spot at every model size |
| TQ-compressed (~20B) | TQ 4-bit | Use 4-bit when stacking on TQ weights — small models have a tighter noise budget |
| **TQ-compressed (100B+)** | **TQ 3-bit** | **3-bit on 3-bit works cleanly on GPT-OSS-120B and Qwen3.5-122B — 100B+ models have enough redundancy to absorb the stacked noise** |

### The speed flip

On small fast models (~20B), KV cache compression is a quality-vs-speed tradeoff: the dequant overhead dominates because the model is fast to begin with. On large slow models (100B+), the 4x smaller KV cache reduces memory bandwidth more than dequant adds — generation is *faster* than the FP16 baseline:

| Model | FP16 KV | TQ 3-bit KV | Direction |
|-------|---------|-------------|-----------|
| GPT-OSS-20B | 90.6 tok/s | 29.9 tok/s | TQ is 3x **slower** |
| GPT-OSS-120B | 6.4 tok/s | 8.7 tok/s | TQ is 1.4x **faster** |

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
python -m turboquant_mlx.generate \
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

> **Note:** Conversion requires temporarily loading the full model. With 120B parameters, peak memory during conversion may reach ~50-55 GB. On a 64 GB machine this is tight — close all other applications before running. The converter processes layers sequentially and frees memory after each expert is quantized.

#### Step 2: Generate text

```bash
python -m turboquant_mlx.generate \
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
python -m turboquant_mlx.generate \
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
python -m turboquant_mlx.generate \
    --model ./nemotron-3-super-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --min-tokens 50
```

#### Benchmarks (M4 Max)

| Model | Bits | Size | Peak RAM | Gen Speed | Quality |
|-------|------|------|----------|-----------|---------|
| **Nemotron-3-Nano-4B** | **3** | **~2.2 GB** | **4.3 GB** | **75.6 tok/s** | **Coherent** |
| **Nemotron-3-Super-120B-A12B** | **3** | **~50 GB** | **54.7 GB** | **18.7 tok/s** | **Coherent with structured `<think>` reasoning (974-token answer w/ self-correction, formulas, formatted structure)** |

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
| Qwen3.5-MoE | `qwen3_5_moe` | Yes (256 experts) | Tested (122B) |
| Nemotron-H (Mamba/attention hybrid) | `nemotron_h` | Yes (512 experts w/ latent MoE on Super-120B) | Tested (Nano-4B, Super-120B) — requires mlx-lm ≥ 0.31.3 |

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

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Han, Daliri, Karbasi (2025)
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Research
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — MLX language model utilities
