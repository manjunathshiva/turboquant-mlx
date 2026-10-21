# TurboQuant-MLX

Extreme **weight** and **KV cache** compression for LLMs on Apple Silicon. MLX implementation of Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025): Hadamard rotation + Lloyd-Max codebooks applied both to weights (compile time) and the KV cache (run time). **Data-free**: no calibration set, no retraining.

**📖 Documentation: [manjunathshiva.github.io/turboquant-mlx](https://manjunathshiva.github.io/turboquant-mlx/)**: get started, pick a model for your Mac's memory, and guides for converting, serving, coding agents, vision models and expert streaming.

Supports dense models (LLaMA, Qwen, Mistral), **Mixture-of-Experts** (Qwen-MoE, GPT-OSS, Qwen3.5-MoE, Qwen3.6-35B-A3B, Qwen3-235B-A22B, DeepSeek-V2, Poolside Laguna, Kimi K3 2.8T, Qwen3.8-Flash-Next 180B), **Mamba/attention hybrids** (Nemotron-3-Nano-4B, Nemotron-3-Super-120B, Nemotron 3.5 Super VL 120B), and **multimodal** models (Muse Glimmer, Qwen3.8-27B, Qwen3.8-Flash-Next, Nemotron 3.5 Super VL, and the block-diffusion DiffusionGemma). Compatible with hybrid attention architectures, attention sinks, sliding-window attention, and linear attention layers (including Gated DeltaNet).

**With both weight and KV cache compression at 3-bit, GPT-OSS-120B fits its full 131K context window in 50 GB on a 64 GB MacBook — and KV cache compression actually makes generation *faster* on the 120B (8.7 vs 6.4 tok/s) because the smaller cache cuts memory bandwidth more than dequant costs.**

**Expert streaming** runs MoE models whose weights exceed available RAM by paging only the router-selected experts from disk per token — e.g. the 35B-parameter Qwen3.6-35B-A3B runs on a **16 GB Mac mini** in under 4 GB of RAM, with output bit-identical to the fully-resident model. See [Models bigger than RAM](https://manjunathshiva.github.io/turboquant-mlx/guides/streaming/).

**Local coding** — [Qwen3.6-27B](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32), a dense SWE-bench-grade coder, runs **fully resident on a 48 GB Mac** at 3-bit (~13 GB on disk, ~17.5 GB at runtime) and serves to Cursor / VS Code over an OpenAI-compatible endpoint. See [Local coding agents](https://manjunathshiva.github.io/turboquant-mlx/guides/coding-agents/).

**Ternary (1.58-bit) experts** shrink the biggest MoEs enough to run fully in memory: Qwen3.6-35B-A3B in 9.4 GB on a 16 GB Mac mini, Qwen3.5-122B in 30.9 GB on a 48 GB Mac, and Poolside Laguna S-2.1 118B and NVIDIA Nemotron 3.5 Super VL 120B on a 64 GB Mac. See [Choose a model](https://manjunathshiva.github.io/turboquant-mlx/models/).

## Key Results — Weight Compression

| Model | Method | Bits | PPL | Size | Gen Speed (M4 Max) |
|-------|--------|------|-----|------|---------------------|
| Qwen2.5-7B | TurboQuant | 3 | 8.92 | 3.5 GB | — |
| Qwen2.5-7B | Affine | 3 | 13.37 | 3.3 GB | — |
| GPT-OSS-20B | Affine (mlx-lm) | 4 | — | 11.2 GB | 148 tok/s |
| GPT-OSS-20B | MXFP4 (original) | 4 | 83.04 | 12.8 GB | — |
| GPT-OSS-20B | TurboQuant | 4 | 72.63 | 11.2 GB | — |
| [GPT-OSS-20B](https://huggingface.co/manjunathshiva/gpt-oss-20b-tq3) | TurboQuant | 3 | 78.60 | 9.3 GB | **73 tok/s** |
| GPT-OSS-120B | [Affine 4-bit (mlx-community)](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 | — | 65.8 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | MXFP4 (original) | 4 | — | 63.5 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | TurboQuant | 3 | — | 48 GB | **44 tok/s** |
| **[GPT-OSS-120B (hybrid for 48GB)](https://huggingface.co/manjunathshiva/gpt-oss-120b-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **~35 GB** | **42–50 tok/s** |
| GPT-OSS-120B | TurboQuant | 2 | — | 32 GB | 51 tok/s (poor quality) |
| Qwen3.5-122B-A10B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **[Qwen3.5-122B-A10B](https://huggingface.co/manjunathshiva/qwen3.5-122b-tq3)** | **TurboQuant** | **3** | **—** | **~50 GB** | **26.5 tok/s (64 GB) · streams on a 16 GB Mac mini** |
| **[Qwen3.5-122B-A10B (ternary experts)](https://huggingface.co/manjunathshiva/Qwen3.5-122B-A10B-tq3a-tqTe-g64)** | **TQ 3-attn / ternary trit-packed experts, gs=64** | **1.6/3 mix** | **—** | **30.9 GB** | **~13 tok/s · 32.5 GB peak, resident on 48 GB Macs without sysctl · 6/6 stress** |
| **[Qwen3.6-35B-A3B](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **~16 GB** | **~60 tok/s (resident) · runs in <4 GB via streaming** |
| **[Qwen3.6-35B-A3B (ternary experts)](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64)** | **TQ 3-attn / ternary trit-packed experts, gs=64** | **1.6/3 mix** | **—** | **9.4 GB** | **~10.4 tok/s fully resident on a 16 GB Mac mini (~33 tok/s M4 Max) · chat; not for agent harnesses** |
| **[Qwen3.6-35B-A3B (ternary + 4-bit expert-down)](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64)** | **TQ 3-attn / ternary experts / 4-bit expert-down, gs=64** | **1.6/3/4 mix** | **—** | **12.6 GB** | **15.2–15.6 tok/s on a 16 GB Mac mini (sysctl) · Opencode agentic 3/3** |
| **[Qwen3.6-27B (dense coder)](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **~13 GB** | **~14 tok/s (resident) · fits 48 GB, SWE-bench coder** |
| **[Qwen3-235B-A22B-Instruct-2507 (hybrid)](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **70.5 GB** | **~4–6 tok/s (64 GB, 40 GB cache) · converts + streams on a 16 GB Mac mini** |
| **[Qwen3-235B-A22B-Instruct-2507 (full 3-bit)](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **103 GB** | **~1.3 tok/s (64 GB, 40 GB cache) · recall-critical sibling, passes 6/6 stress** |
| **[Qwen3-235B-A22B-Instruct-2507 (ternary experts, 1.58-bit)](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tqTe-g64)** | **TQ 3-attn / ternary trit-packed experts, gs=64** | **1.6/3 mix** | **—** | **53 GB** | **5.6 tok/s (64 GB) · fully resident, no streaming** |
| **Nemotron-3-Nano-4B** | **TurboQuant** | **3** | **—** | **~2.2 GB** | **75.6 tok/s** |
| Nemotron-3-Super-120B-A12B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Nemotron-3-Super-120B-A12B** | **TurboQuant** | **3** | **—** | **~50 GB** | **18.7 tok/s** |
| **[Nemotron-3-Super-120B-A12B (hybrid for 48GB)](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32)** | **TQ 3-attn / 2-experts, gs=32** | **2/3 mix** | **—** | **~36 GB** | **~27.2 tok/s** |
| **[Laguna-XS.2 (33B agentic coder)](https://huggingface.co/manjunathshiva/Laguna-XS.2-tq3-g64)** | **TurboQuant, gs=64** | **3** | **—** | **13.8 GB** | **46.8 tok/s · 32 GB Mac · Opencode pass** |
| Laguna-S-2.1 (118B) | BF16 (original) | 16 | — | ~219 GB | *Doesn't fit 64GB* |
| Laguna-S-2.1 (118B) | Affine 4-bit (mlx-community; since removed from Hugging Face) | 4 | — | ~64 GB | *Exceeds Metal's ~56 GB working set — won't load* |
| **[Laguna-S-2.1 (118B, ternary experts)](https://huggingface.co/manjunathshiva/Laguna-S-2.1-tqTe-g64)** | **TQ 3-attn / ternary trit-packed experts, gs=64** | **1.6/3 mix** | **—** | **~27 GB** | **~12.5 tok/s · fully resident on 64 GB with ~25 GB spare · Opencode pass** |
| **Kimi K3 (2.8T MoE)** | **TQ 3-attn / ternary experts / 4-bit expert-down, gs=64** | **1.6/3/4 mix** | **—** | **931 GB** | **2.3 tok/s (Mac Studio 512 GB, top-8) · 1.2 tok/s at native top-16 · streams experts from disk** |
| DiffusionGemma-26B-A4B (block-diffusion MoE) | [Affine 4-bit (mlx-community)](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) | 4 | — | 15.0 GB | ~35 tok/s |
| **[DiffusionGemma-26B-A4B](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32)** | **TurboQuant, gs=32** | **3** | **—** | **13.8 GB** | **~4.6 tok/s · experimental: slower than 4-bit** |
| **[DiffusionGemma-26B-A4B (16 GB build)](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-mini-g32)** | **TQ 3-attn / 2-experts, 6 layers protected at 3-bit / 8-bit extras, gs=32** | **2/3 mix** | **—** | **9.79 GB** | **~1.5 tok/s on a 16 GB Mac mini (sysctl) · experimental** |
| Muse Glimmer (30B dense VLM) | [Affine 4-bit](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit) | 4 | 4.3798 | 19.88 GiB | 26.1 tok/s · leaves embedding + vision tower at bf16 |
| **[Muse Glimmer (30B dense VLM)](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq4-g64)** | **TurboQuant, gs=64** | **4** | **4.3315** | **14.88 GiB** | **9.4 tok/s · better PPL than affine 4-bit in 25% less space** |
| **[Muse Glimmer (30B dense VLM)](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq3-g64)** | **TurboQuant, gs=64** | **3** | **5.0454** | **12.53 GiB** | **10.9 tok/s · smallest coherent build** |
| Qwen3.8-27B (dense 27.8B hybrid-GDN VLM) | BF16 (original) | 16 | — | 55.6 GB | *Needs a 64 GB Mac just to load* |
| Qwen3.8-27B | [Affine 4-bit (mlx-community)](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 4 | 7.2685 | 14.95 GiB | **29.7 tok/s** — fastest 4-bit, but peaks 16.16 GiB: *won't fit 16 GB* |
| **[Qwen3.8-27B](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq4-g64)** | **TurboQuant, gs=64** | **4** | **7.2496** | **15.15 GiB** | **11.3 tok/s · best PPL of the three · Opencode agentic pass · vision 4/4** |
| **[Qwen3.8-27B (16 GB build)](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq3-mini-g64)** | **TurboQuant, gs=64** | **3** | **8.0052** | **11.55 GiB** | **12–13.6 tok/s (M4 Max) · verified on a real 16 GB Mac mini: 13.61 GiB peak, 3.7 tok/s, vision passes** |
| Qwen3.8-Flash-Next (180B MoE, 512 experts) | [Affine oQ2 (Vontra)](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ2) | 2 | — | 63.03 GiB | *Too big to load resident on a 64 GB Mac* |
| Qwen3.8-Flash-Next | [imatrix iQ-MLX 3.3 bpw (ddalcu, for mlx-serve)](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw) | 3.3 | — | 50.6 GiB | *Fits 64 GB per its card; not measured here* |
| **[Qwen3.8-Flash-Next](https://huggingface.co/manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64)** | **TQ 4-attn / 2-experts / 2-bit affine n-gram table, data-free** | **2/4 mix** | **9.31** | **52.00 GiB** + 0.84 GiB vision | **16–18 tok/s · fully resident on 64 GB · vision passes** |
| NVIDIA Nemotron 3.5 Super VL 120B (VLM, 512 experts, top-22) | BF16 (original) | 16 | — | 249 GB | *Doesn't fit 64GB* |
| Nemotron 3.5 Super VL 120B | NVFP4 (NVIDIA) | 4 | — | 63.0 GiB | *Doesn't fit resident on a 64 GB Mac* |
| **[Nemotron 3.5 Super VL 120B (ternary experts)](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64)** | **TQ 3-attn / ternary trit-packed experts / 8-bit affine extras, gs=64** | **1.6/3 mix** | **—** | **29.29 GiB** | **~7–8 tok/s · fully resident on 64 GB at default settings · 36K context · text 6/6, vision 3/3** |
| **[Nemotron 3.5 Super VL 120B (3-bit)](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3-g64)** | **TurboQuant, gs=64** | **3** | **—** | **50.22 GiB** | **~19–21 tok/s · 64 GB with `sysctl` + `--prefill-step-size 256`** |

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

> **On the Qwen3.5-122B KV rows:** they used symmetric 3-bit KV on a short prompt. Fully resident with the recommended **K8/V3** cache, FP16 decodes faster on that model, so there KV compression is a *memory* win, not a speed win; the decode speed-up is specific to **GPT-OSS-120B**. See [KV cache compression](https://manjunathshiva.github.io/turboquant-mlx/guides/kv-cache/) and [#19](https://github.com/manjunathshiva/turboquant-mlx/pull/19).

Community results from an **Apple M5 Pro (48 GB)**, contributed by [@sbayer2](https://github.com/sbayer2) (#14, #16), are on the [Benchmarks page](https://manjunathshiva.github.io/turboquant-mlx/benchmarks/#apple-m5-pro-48-gb): resident and streaming speeds, a KV cache sweep against context, and the cache-budget sweep for the 48 GB tier.

## Install

```bash
pip install turboquant-mlx-full
```

The package is published as `turboquant-mlx-full` on PyPI and imported as `turboquant_mlx`.

- macOS on Apple Silicon (M1 or later)
- Python 3.11+ (mlx-lm 0.32 needs it; on 3.10, pip installs turboquant-mlx-full 0.28.x)
- The Metal kernels are compiled by MLX on first use: no Xcode or CMake needed

Optional extras:

```bash
pip install "turboquant-mlx-full[vlm]"    # vision and diffusion models via mlx-vlm (>= 0.7.9)
pip install "turboquant-mlx-full[eval]"   # perplexity benchmarking (datasets, transformers)
pip install "turboquant-mlx-full[kimi]"   # Kimi K3's tiktoken-based tokenizer
```

From source: `git clone https://github.com/manjunathshiva/turboquant-mlx.git && cd turboquant-mlx && pip install -e .` (see [CONTRIBUTING.md](https://github.com/manjunathshiva/turboquant-mlx/blob/main/CONTRIBUTING.md)).

## Quick start

**1. Will it fit?** `turboquant-plan` reads only the safetensors headers (a few hundred KB, over the network for a Hugging Face repo), so you know before a multi-GB download:

```bash
turboquant-plan --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64
```

```
Projection at 16,384 tokens of context
  weights              12.59 GB
  KV cache             0.34 GB  (20.0 KB/token, hybrid: 10/40 full-attention layers)
  prefill workspace    1.07 GB  (estimate, at --prefill-step-size 2048)
  runtime reserve      1.00 GB  (buffer cache, activations, fragmentation)
                       ----------------------------------
  peak                 15.00 GB of 61.85 GB usable   46.85 GB headroom

Verdict: ✅ RESIDENT — fits fully in memory
```

When a model fits only with a raised GPU memory limit or a smaller prefill step, it prints the exact `sysctl` line and flags. `turboquant-doctor` adds a read-only readiness check of the model files and machine.

**2. Run it.**

```bash
turboquant-generate --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64 \
    --prompt "Why is the sky blue? Explain in simple terms." --max-tokens 300
```

**3. Serve it** over an OpenAI-compatible API at `http://localhost:8080/v1`, for apps, Cursor, VS Code or a coding agent:

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64 --port 8080
```

**4. Convert your own** (BF16 or FP16 weights of a [supported architecture](https://manjunathshiva.github.io/turboquant-mlx/reference/architectures/); minutes, no calibration data):

```bash
turboquant-convert --hf-path meta-llama/Llama-3.2-1B --mlx-path ./llama-3.2-1b-tq3 --bits 3 --group-size 64
```

## Documentation

**[manjunathshiva.github.io/turboquant-mlx](https://manjunathshiva.github.io/turboquant-mlx/)**

- [Get started](https://manjunathshiva.github.io/turboquant-mlx/getting-started/) and [Choose a model](https://manjunathshiva.github.io/turboquant-mlx/models/) for your Mac's memory
- Guides: [Convert a model](https://manjunathshiva.github.io/turboquant-mlx/guides/convert/) (mixed precision, ternary experts, the streaming converter for 200B+ models) · [Generate text](https://manjunathshiva.github.io/turboquant-mlx/guides/generate/) · [Serve an OpenAI-compatible API](https://manjunathshiva.github.io/turboquant-mlx/guides/serve/) · [Local coding agents](https://manjunathshiva.github.io/turboquant-mlx/guides/coding-agents/) · [Vision models](https://manjunathshiva.github.io/turboquant-mlx/guides/vision/) · [Models bigger than RAM](https://manjunathshiva.github.io/turboquant-mlx/guides/streaming/) · [KV cache compression](https://manjunathshiva.github.io/turboquant-mlx/guides/kv-cache/) · [Memory and macOS settings](https://manjunathshiva.github.io/turboquant-mlx/guides/memory/)
- Models, one page per family with run commands, measurements and how each build was made: [GPT-OSS](https://manjunathshiva.github.io/turboquant-mlx/model-pages/gpt-oss/) · [Qwen3.5-122B-A10B](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen35-122b/) · [Qwen3.6-35B-A3B](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen36-35b/) · [Qwen3.6-27B](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen36-27b/) · [Qwen3-235B-A22B](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen3-235b/) · [Nemotron-3](https://manjunathshiva.github.io/turboquant-mlx/model-pages/nemotron-3/) · [Nemotron 3.5 Super VL](https://manjunathshiva.github.io/turboquant-mlx/model-pages/nemotron-35-vl/) · [Poolside Laguna](https://manjunathshiva.github.io/turboquant-mlx/model-pages/laguna/) · [Kimi K3](https://manjunathshiva.github.io/turboquant-mlx/model-pages/kimi-k3/) · [Sarvam MoE](https://manjunathshiva.github.io/turboquant-mlx/model-pages/sarvam/) · [Muse Glimmer](https://manjunathshiva.github.io/turboquant-mlx/model-pages/muse-glimmer/) · [Qwen3.8-27B](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen38-27b/) · [Qwen3.8-Flash-Next](https://manjunathshiva.github.io/turboquant-mlx/model-pages/qwen38-flash-next/) · [DiffusionGemma](https://manjunathshiva.github.io/turboquant-mlx/model-pages/diffusiongemma/)
- [How it works](https://manjunathshiva.github.io/turboquant-mlx/how-it-works/) · [Benchmarks](https://manjunathshiva.github.io/turboquant-mlx/benchmarks/) · [Command line](https://manjunathshiva.github.io/turboquant-mlx/reference/cli/) · [Python API](https://manjunathshiva.github.io/turboquant-mlx/reference/python-api/) · [Supported architectures](https://manjunathshiva.github.io/turboquant-mlx/reference/architectures/) · [Changelog](https://github.com/manjunathshiva/turboquant-mlx/blob/main/CHANGELOG.md)

## How it works

1. **Randomized Hadamard rotation** makes any weight distribution close to Gaussian, with no data.
2. **Group and scale**: each group of 32, 64 or 128 weights is divided by its root-mean-square.
3. **Lloyd-Max codebook**: each weight snaps to the nearest centroid of the quantizer that is optimal for a Gaussian, a mathematical constant rather than something fitted.
4. **Pack** the indices into 32-bit words at 2, 3 or 4 bits, or as base-3 trits for the ternary `{-c, 0, +c}` expert tier (~1.6 bits per weight).

Fused Metal kernels decode straight from the packed form, so full-precision weights never exist in memory. The result: near-zero quality loss at 3-bit, and usable 2-bit and ternary experts on large MoEs where affine quantization breaks down. The same rotation and codebook compress the KV cache at run time. [More](https://manjunathshiva.github.io/turboquant-mlx/how-it-works/).

## Supported architectures

LLaMA, Qwen2/2.5, Qwen3.5/3.8 (text and vision), Mistral, Qwen1.5-MoE, GPT-OSS, Qwen3.5-MoE and Qwen3.6, Qwen3-MoE, Nemotron-H, DeepSeek-V2 (MLA + MoE; V3 shares the layout but is untested), and through mlx-vlm Muse Glimmer, NVIDIA Nemotron 3.5 Super VL and DiffusionGemma. Poolside Laguna, Kimi K3 and Sarvam MoE are ported to MLX here, and Qwen3.8-Flash-Next is vendored from an open mlx-lm pull request. Status and tested sizes for each: [Supported architectures](https://manjunathshiva.github.io/turboquant-mlx/reference/architectures/).

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](https://github.com/manjunathshiva/turboquant-mlx/blob/main/CONTRIBUTING.md) (commits need a DCO sign-off) and [SECURITY.md](https://github.com/manjunathshiva/turboquant-mlx/blob/main/SECURITY.md) for reporting vulnerabilities.

## Citation

```bibtex
@misc{turboquant_mlx,
    title={TurboQuant-MLX: Extreme Weight and KV Cache Compression for Apple Silicon},
    year={2025},
    note={MLX implementation of TurboQuant (Zandieh et al., 2025) for both weight quantization and runtime KV cache compression}
}
```

## License

Apache-2.0 (`LICENSE`, `NOTICE`), from v0.3.0. Versions 0.1.x–0.2.0 were released under MIT (`LICENSES/MIT.txt`).

## Acknowledgments

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (2025)
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Research
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — MLX language model utilities
