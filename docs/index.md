# TurboQuant-MLX

**Run models that shouldn't fit on your Mac.** TurboQuant-MLX compresses large language and
vision models, including **100B–235B Mixture-of-Experts**, down to 2-bit, 3-bit, 4-bit or
**ternary (1.58-bit)** weights, and runs them on Apple Silicon with fused Metal kernels that
decode straight from the packed form. It is **data-free**: no calibration set, no retraining.
The KV cache can be compressed the same way.

```bash
pip install turboquant-mlx-full
```

```bash
# Will it fit? Answered from the file headers, before you download anything.
turboquant-plan --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64

# Run it.
turboquant-generate --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 \
    --prompt "Explain why the sky is blue." --max-tokens 300

# Serve it over an OpenAI-compatible API.
turboquant-serve --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 --port 8080
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Choose a model](models.md){ .md-button }
[How it works](how-it-works.md){ .md-button }

## What it makes possible

| On a… | Runs | How |
|---|---|---|
| **16 GB Mac mini** | Qwen3.6-35B-A3B, **fully in memory** (9.4 GB, ~10 tok/s) | [ternary experts](model-pages/qwen36-35b.md) |
| **16 GB Mac mini** | Qwen3.8-27B vision model (11.55 GiB, vision passes) | [3-bit](model-pages/qwen38-27b.md) |
| **16 GB Mac mini** | Qwen3.5-122B (~2 tok/s) and Qwen3-235B (~0.2 tok/s), streaming experts from disk | [expert streaming](guides/streaming.md) |
| **48 GB Mac** | Qwen3.5-122B, fully in memory (30.9 GB, ~13 tok/s) | [ternary experts](model-pages/qwen35-122b.md) |
| **64 GB Mac** | NVIDIA Nemotron 3.5 Super VL 120B, a vision model, **at default settings** | [ternary experts](model-pages/nemotron-35-vl.md) |
| **64 GB Mac** | Poolside Laguna S-2.1 118B coding agent (~27 GB, passes Opencode) | [ternary experts](model-pages/laguna.md) |
| **64 GB Mac** | Qwen3.8-Flash-Next 180B, fully in memory (16–18 tok/s) | [mixed 4/2-bit](model-pages/qwen38-flash-next.md) |
| **64 GB Mac** | GPT-OSS-120B with its full 131K context, in 50 GB | [weights + KV cache at 3-bit](guides/kv-cache.md) |

## Find a guide

- **New here?** [Get started](getting-started.md): install, check what fits, run and serve your
  first model.
- **Looking for a download?** [Choose a model](models.md) lists every published build by how
  much memory your Mac has.
- **Converting your own?** [Convert a model](guides/convert.md) covers bit widths, mixed
  attention/expert precision, ternary experts and the streaming converter for 200B+ models.
- **Using it from an app or an agent?** [Serve an OpenAI-compatible API](guides/serve.md) and
  [Local coding agents](guides/coding-agents.md).
- **Images too?** [Vision models](guides/vision.md).
- **Model bigger than your RAM?** [Expert streaming](guides/streaming.md).
- **Out of memory, or asked to run `sysctl`?** [Memory and macOS settings](guides/memory.md).

## In one paragraph

TurboQuant-MLX implements Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et
al., 2025) for MLX. A randomized Hadamard rotation makes weights look Gaussian; each group is
scaled by its RMS and snapped to the nearest centroid of a Lloyd-Max codebook that is optimal
for a Gaussian; the indices are bit-packed (or packed twenty to a 32-bit word as base-3 trits for
the ternary tier, about 1.6 bits each). At run time the layer rotates its *input* instead of un-rotating the weights,
and Metal kernels decode the packed codes inline, so the full-precision weights never exist in
memory. Details in [How it works](how-it-works.md).

Apache-2.0 · [GitHub](https://github.com/manjunathshiva/turboquant-mlx) ·
[PyPI](https://pypi.org/project/turboquant-mlx-full/) ·
[Models on Hugging Face](https://huggingface.co/collections/manjunathshiva/turboquant-mlx-6ab8b9771f2e996ba1c7ffe3)
