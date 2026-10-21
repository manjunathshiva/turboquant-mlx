# Choose a model

Every TurboQuant build published on Hugging Face, grouped by how much unified memory your Mac
has. All are in the
[TurboQuant-MLX collection](https://huggingface.co/collections/manjunathshiva/turboquant-mlx-6ab8b9771f2e996ba1c7ffe3).

Speeds are decode speeds as measured on the Mac named. Before downloading, run
`turboquant-plan --model <repo>` to check the fit on **your** Mac; it reads only the file
headers (see [Get started](getting-started.md#1-check-that-it-fits-before-downloading)).

!!! tip "Reading the Settings column"
    **default** means nothing to change. **sysctl** means raising the GPU memory limit once per
    boot with `sudo sysctl -w iogpu.wired_limit_mb=<value>`; the card gives the value. See
    [Memory and macOS settings](guides/memory.md).

## 16 GB (Mac mini)

Measured on a 16 GB M4 Mac mini.

| Model | Build | Size | Settings | Speed on the mini | Notes |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | [tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64) | 9.4 GB | default | ~10.4 tok/s | Fully in memory. Good for chat; not for agent harnesses. |
| Qwen3.6-35B-A3B | [tq3a-tqTe-down4-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64) | 12.6 GB | sysctl 13824 | 15.2–15.6 tok/s | Fully in memory. Passes an Opencode agentic task 3/3. |
| GPT-OSS-20B | [tq3](https://huggingface.co/manjunathshiva/gpt-oss-20b-tq3) | 9.3 GB | default | 60–80 tok/s (M-series) | ~11 GB peak. |
| Qwen3.8-27B (vision) | [tq3-mini-g64](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq3-mini-g64) | 11.55 GiB | sysctl 14336 + `--prefill-step-size 256` | 3.7 tok/s | 12.95–13.61 GiB peak. Vision passes. Not agentic on 16 GB. |
| DiffusionGemma-26B-A4B | [mini-g32](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-mini-g32) | 9.79 GB | sysctl 13824 | ~1.5 tok/s | Block-diffusion model. Experimental. |

**Bigger than 16 GB, by streaming experts from disk** (see [expert streaming](guides/streaming.md)):

| Model | Build | Size on disk | Speed on the mini | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | [tq3-g32](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) | ~16 GB | — | Runs in under 4 GB of RAM; output bit-identical to resident. |
| Qwen3.5-122B-A10B | [tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/Qwen3.5-122B-A10B-tq3a-tqTe-g64) | 30.9 GB | ~2.1 tok/s | 11.3 GB peak, `--cache-budget-gb 8`. |
| Qwen3-235B-A22B | [tq3a-tq2e-g32](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32) | 70.5 GB | ~0.2 tok/s | Works, but disk-bound on a mini. Much faster on 64 GB. |

## 24–36 GB

Measured on a 64 GB M4 Max unless noted; these fit a 24–36 GB Mac by size. Check with
`turboquant-plan`.

| Model | Build | Size | Speed (M4 Max) | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | [tq3-g32](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) | ~16 GB | ~60 tok/s | ~18 GB peak. |
| Laguna-XS.2 (33B coding agent) | [tq3-g64](https://huggingface.co/manjunathshiva/Laguna-XS.2-tq3-g64) | 13.8 GB | 46.8 tok/s | Fits a 32 GB Mac. Opencode pass. |
| Qwen3.8-27B (vision) | [tq4-g64](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq4-g64) | 15.15 GiB | 11.3 tok/s | Best perplexity of the Qwen3.8-27B builds. Opencode pass, vision 4/4. |
| Muse Glimmer 30B (vision) | [tq4-g64](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq4-g64) | 14.88 GiB | 9.4 tok/s | Better perplexity than affine 4-bit, in 25% less space. |
| Muse Glimmer 30B (vision) | [tq3-g64](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq3-g64) | 12.53 GiB | 10.9 tok/s | Smallest coherent build. |
| DiffusionGemma-26B-A4B | [tq3-g32](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32) | 13.8 GB | ~4.6 tok/s | ~18 GB peak. Experimental; slower than 4-bit. |

## 48 GB

| Model | Build | Size | Speed | Notes |
|---|---|---|---|---|
| Qwen3.6-27B (dense coder) | [tq3-g32](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32) | ~13 GB | ~14 tok/s | ~17.5 GB at run time; room for long context. |
| Qwen3.5-122B-A10B | [tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/Qwen3.5-122B-A10B-tq3a-tqTe-g64) | 30.9 GB | ~13 tok/s | 32.5 GB peak, no sysctl on 48 GB. 6/6 stress. |
| Laguna-S-2.1 (118B coding agent) | [tqTe-g64](https://huggingface.co/manjunathshiva/Laguna-S-2.1-tqTe-g64) | ~27 GB | ~12.5 tok/s | ~30 GB peak. Opencode pass. |
| GPT-OSS-120B | [tq3a-tq2e-g32](https://huggingface.co/manjunathshiva/gpt-oss-120b-tq3a-tq2e-g32) | ~35 GB | 42–50 tok/s | 3-bit attention, 2-bit experts. 6/6 stress. |
| Nemotron-3-Super-120B | [tq3a-tq2e-g32](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3a-tq2e-g32) | ~36 GB | ~27.2 tok/s | ~40.8 GB peak. Math drifts on long generations. |

## 64 GB

| Model | Build | Size | Settings | Speed (M4 Max) | Notes |
|---|---|---|---|---|---|
| NVIDIA Nemotron 3.5 Super VL 120B (vision) | [tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64) | 29.29 GiB | **default** | ~7–8 tok/s | 44.7 GB peak at a 36K-token prompt. Text 6/6, vision 3/3. |
| NVIDIA Nemotron 3.5 Super VL 120B (vision) | [tq3-g64](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3-g64) | 50.22 GiB | sysctl 60416 + `--prefill-step-size 256` | ~19–21 tok/s | The faster build. |
| Qwen3.8-Flash-Next 180B (vision) | [tq4a-tq2e-g64](https://huggingface.co/manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64) | 52.00 GiB | sysctl 60416 | 16–18 tok/s | 56.3 GB peak. `--ngram-offload` cuts the resident footprint. |
| Qwen3-235B-A22B | [tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tqTe-g64) | 53 GB | sysctl 60416 | 5.6 tok/s | Fully in memory, no streaming. |
| Qwen3.5-122B-A10B | [tq3](https://huggingface.co/manjunathshiva/qwen3.5-122b-tq3) | ~54 GB | sysctl | ~25 tok/s | Runs out of GPU memory past ~4–8K context. |
| GPT-OSS-120B | [tq3](https://huggingface.co/manjunathshiva/gpt-oss-120b-tq3) | 48 GB | check with `turboquant-plan` | 44 tok/s | |
| Nemotron-3-Super-120B | [tq3](https://huggingface.co/manjunathshiva/Nemotron-3-Super-120B-A12B-tq3) | ~50 GB | sysctl | 18.7 tok/s | ~55 GB peak. |
| Qwen3-235B-A22B | [tq3a-tq2e-g32](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32) | 70.5 GB | streaming | ~4–6 tok/s | `--cache-budget-gb 40`. 5/6 quality probes. |
| Qwen3-235B-A22B | [tq3-g32](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32) | 103 GB | streaming | ~1.3 tok/s | Recall-critical sibling. 6/6 stress. |

## Which build of the same model?

- **`tq3`** is 3-bit everywhere: the quality and speed reference.
- **`tq3a-tq2e`** keeps attention at 3-bit and drops experts to 2-bit: smaller, faster, but long
  generations can drift on some models.
- **`tqTe`** uses **ternary (1.58-bit) experts**: the smallest build. It needs a model with many
  wide experts (128+), and decodes slower than 3-bit.
- **`down4`** adds a 4-bit expert down projection to a ternary build: a few GB more, and it is
  what made Qwen3.6-35B work in agent harnesses.
- **`g32` / `g64`** is the group size: 32 is more accurate, 64 is smaller.

Each model page under **Models** has the details, and each Hugging Face card has the full
measurements.
