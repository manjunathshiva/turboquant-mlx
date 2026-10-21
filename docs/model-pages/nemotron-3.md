# Nemotron-3 (Mamba/attention hybrid)

Nemotron-3 is NVIDIA's hybrid Mamba2 + attention architecture. Two variants are tested:

- **Nano-4B** — dense (Mamba + MLP + attention), 42 layers
- **Super-120B-A12B** — hybrid MoE (Mamba + 512-expert latent-MoE + attention), 88 layers, ~12B active per token

Both require **mlx-lm ≥ 0.31.3** for upstream Nemotron-H support (installed automatically).

## Convert

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

## Generate

Nemotron-3's chat template ends in a `<think>\n` scaffold that primes EOS as the top-1 logit at the start of the assistant turn. Pass `--min-tokens` to mask EOS for the first N tokens so the model enters the think phase:

```bash
turboquant-generate \
    --model ./nemotron-3-super-120b-tq3 \
    --prompt "Why is the sky blue?" \
    --max-tokens 200 --min-tokens 50
```

## Benchmarks (M4 Max)

| Model | Bits | Size | Peak RAM | Gen Speed | Quality |
|-------|------|------|----------|-----------|---------|
| **Nemotron-3-Nano-4B** | **3** | **~2.2 GB** | **4.3 GB** | **75.6 tok/s** | **Coherent** |
| **Nemotron-3-Super-120B-A12B** | **3** | **~50 GB** | **54.7 GB** | **18.7 tok/s** | **Coherent with structured `<think>` reasoning (974-token answer w/ self-correction, formulas, formatted structure)** |
| **Nemotron-3-Super-120B-A12B (hybrid)** | **3-attn / 2-experts, gs=32** | **~36 GB** | **~40.8 GB** | **~27.2 tok/s** | **Coherent prose, code, format, and long-context recall; math accuracy degraded — see Phase-1 note below** |

## 48 GB-RAM target: hybrid (3-bit attention / 2-bit experts) at group-size 32

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
turboquant-generate \
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
