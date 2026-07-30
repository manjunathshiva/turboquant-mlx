---
library_name: mlx
license: other
license_name: modified-mit
pipeline_tag: text-generation
base_model: moonshotai/Kimi-K3
tags:
- mlx
- turboquant
- moe
- kimi
---

# Kimi-K3-tq3a-tqTe-down4-g64

**Ternary-expert** TurboQuant quantization of Moonshot AI's **Kimi K3** — a **2.8-trillion-parameter** Mixture-of-Experts model — produced with [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

This is the largest model TurboQuant-MLX has run to date. The BF16 checkpoint is ~5.6 TB; this build lands at **869 GB on disk** and generates coherent, high-quality text on a **single 512 GB Apple Silicon Mac Studio** by streaming MoE experts from disk — no cluster, no GPUs.

## Model Details

- **Base model**: Moonshot AI Kimi K3 (`model_type: kimi_k3`) — text tower of the `KimiK3ForConditionalGeneration` multimodal reference, vision tower dropped
- **Architecture**: 93 layers, hidden size 7168, **896 routed experts (native top-16) + 2 shared experts**, MoE intermediate 3072, routed-expert latent 3584, vocab 163,840 — ~2.8T total / ~30B active params
  - **Hybrid attention** — 69 KDA (Kimi Delta Attention) linear-attention layers + 24 gated-NoPE MLA layers with q-LoRA, on a mostly-3:1 cadence
  - **Stable LatentMoE** — routed experts operate in a 3584-dim latent space; the top-k weighted sum happens in latent space, then a shared RMSNorm, then the up-projection
  - **`situ` activation**, **Attention Residuals (AttnRes)**, bounded KDA gate — all reproduced faithfully (fp32 parity vs the HF reference ≤ 2.7e-6)
- **Quantization**: **TurboQuant** (Hadamard rotation, seed 42 + Lloyd-Max codebook), group size 64
  - Attention (`q/k/v/o_proj`, q-LoRA, MLA latents) → **3-bit**
  - Routed experts `gate/up_proj` → **ternary** (trit-packed, ~1.58-bit)
  - Routed experts `down_proj` → **4-bit** (the recall-sensitive projection kept wider)
  - Routers (`mlp.gate`) → full precision (auto-skipped — never quantized)
- **Size**: **869 GB** across **186** shards (vs ~5.6 TB BF16). Experts alone are ~903 GB uncompressed-equivalent; the always-resident (non-expert) working set is ~30 GB.

The recipe name decodes as **tq3a** (3-bit attention) / **tqTe** (ternary experts) / **down4** (4-bit expert down-projection) / **g64** (group size 64).

> **Note:** this is an instruct/thinking model. Use the chat template (the streaming generator applies it by default). Thinking-mode output — a reasoning block before the answer — is normal.

## Quality

Even at ternary experts with a **2× K-cut** (top-16 → top-8 routing), real-use quality holds. Chat-template thinking-mode output on "Explain how Rayleigh scattering works, and why the sky is blue but sunsets are red" is accurate throughout — the λ⁻⁴ law, the (700/450)⁴ ≈ 5.9× blue/red scattering ratio, and the violet/ozone/eye-sensitivity nuance are all correct, with a clean reasoning structure.

## Running it (expert streaming)

At 869 GB this model does not fit resident on any single Mac. It runs by **streaming MoE experts from disk** — each token pages in only its router-selected experts (pinned hot-list + LRU cache), so the big expert tensors are never all in memory at once. Non-expert weights (attention, routers, shared experts, embeddings) stay resident.

Measured on a **Mac Studio (M-series, 512 GB)** with `iogpu.wired_limit_mb=512000`, a 506 GB expert cache (`--cache-budget-gb auto`), the shipped hot-expert list, and `--no-chat-template` for controlled A/B (200-token generation):

| Config | Decode speed | Expert hit-rate | Critical (blocking) disk reads | Peak memory |
|---|---|---|---|---|
| top-16, `--prefetch-workers 8` | 0.83 tok/s | 99.6% | 14.9 GB | 534 GB |
| top-16, `--prefetch-workers 16` | **1.18 tok/s** | 99.6% | 13.4 GB | 530 GB |
| top-16, `--prefetch-workers 32` | 1.14 tok/s | 99.6% | 11.7 GB | 533 GB |
| **top-8 (2× K-cut), `--prefetch-workers 16`** | **2.30 tok/s** | **100%** | **0.0 GB** | 400 GB |

The `--max-active-experts 8` lever (native top-16 → top-8) roughly halves per-token expert I/O for a ~2× decode speed-up at no observed quality cost. Prefetch parallelism matters: 16 workers is the knee — at 8 the reads serialize and lose ~30%, past 16 there's no further gain (decode is no longer disk-bandwidth-bound).

### Recommended invocation

```bash
# Raise the Metal wired ceiling on a 512 GB box (resets on reboot)
sudo sysctl iogpu.wired_limit_mb=512000

python3 -m turboquant_mlx.stream.stream_generate \
    --model ./Kimi-K3-tq3a-tqTe-down4-g64 \
    --prompt "Explain how Rayleigh scattering works." \
    --max-tokens 500 --cache-budget-gb auto \
    --max-active-experts 8 --prefetch-workers 16
```

Kimi K3 uses a tiktoken-based tokenizer — install `tiktoken` and `blobfile` alongside TurboQuant-MLX.

## Provenance & verification

- fp32 forward parity vs the HF `transformers` reference: ≤ 2.7e-6 (harness in `scripts/k3/kimi_k3_parity.py`).
- mxfp4 expert unpack is bit-exact with MLX `mode="mxfp4"` (`weight_packed.view(uint32)` + raw u8 scales).
- Streaming output is functionally equivalent to the fully-materialized model; experts are paged, not approximated further.
