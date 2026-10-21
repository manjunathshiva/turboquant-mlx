# Benchmarks

Measured results, mostly on a 64 GB M4 Max, plus a 16 GB M4 Mac mini and a 48 GB M5 Pro where noted. For a download list by Mac memory, see [Choose a model](models.md); each model's Hugging Face card has its full measurements.

## Weight compression

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

## KV cache compression

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

## Apple M5 Pro (48 GB)

First Metal4 / `MTLGPUFamilyApple10` data point, contributed by [@sbayer2](https://github.com/sbayer2) (#14, #16) — a new 48 GB tier between the 16 GB Mac mini and the 64 GB M4 Max. Reproduce with [`benchmarks/bench_m5_pro.py`](https://github.com/manjunathshiva/turboquant-mlx/blob/main/benchmarks/bench_m5_pro.py).

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

KV compression gives a consistent **~1.6× prompt-processing speedup** for a ~12% decode cost (long-context decode behavior is in [The speed flip](guides/kv-cache.md#it-saves-memory-it-rarely-saves-time)). Expert-streaming hit-rate scales with the cache budget — **44.6% at 4 GB (16 GB mini) → 89.9% at 30 GB (48 GB)**, a ~7× throughput jump that fills the gap between the 16 GB and 64 GB tiers.

> **Stability near the memory ceiling:** long-context *prompt prefill* close to the wired cap can starve the kernel watchdog (a `watchdogd` / `AppleARMWatchdogTimer` panic). A rapidly-growing KV cache makes Metal commit pages continuously (`AGXG17XFamilyResidencySet _commitAddedAllocations`), so the binding limit is allocation *rate*, not peak — a static 41 GB resident model (Nemotron) is stable, while a growing ~30 GB KV during a 63K-token prefill can panic. Practical limits on the 48 GB tier (#14): contexts up to ~14.5K are safe on the 35B with either KV config; 63K is feasible with K8/V3 (28.7 GB peak) but not fp16 (30.8 GB → panic). Keep headroom and close other apps for long-context runs near the cap.
