# Models

One page per model family: how it was converted, how to run it, and what was measured. To pick
a download by how much memory your Mac has, start with [Choose a model](../models.md).

| Family | Type | Pages cover |
|---|---|---|
| [GPT-OSS](gpt-oss.md) | MoE, 20B and 120B | 2-, 3- and 4-bit builds; the 48 GB hybrid; 131K context on 64 GB |
| [Qwen3.5-122B-A10B](qwen35-122b.md) | MoE, 256 experts | 3-bit and ternary; streaming on a 16 GB mini |
| [Qwen3.6-35B-A3B](qwen36-35b.md) | Hybrid MoE | Resident and streamed; the 16 GB ternary and agentic `down4` builds |
| [Qwen3.6-27B](qwen36-27b.md) | Dense hybrid coder | A local coding model for a 48 GB Mac |
| [Qwen3-235B-A22B](qwen3-235b.md) | MoE, 128 experts | Converting on a 16 GB mini; hybrid, 3-bit and ternary builds |
| [Nemotron-3](nemotron-3.md) | Mamba/attention hybrid | Nano-4B and Super-120B |
| [Nemotron 3.5 Super VL](nemotron-35-vl.md) | Hybrid MoE vision model | 120B resident on a 64 GB Mac |
| [Poolside Laguna](laguna.md) | MoE coding agents | XS.2 33B and S-2.1 118B |
| [Kimi K3](kimi-k3.md) | 2.8T MoE | Streaming from disk on a 512 GB Mac Studio |
| [Sarvam MoE](sarvam.md) | MoE | Port only; no build published |
| [Muse Glimmer](muse-glimmer.md) | Dense vision model, 30B | 4-bit and 3-bit builds that beat affine 4-bit |
| [Qwen3.8-27B](qwen38-27b.md) | Dense hybrid vision model | A 16 GB build and a 4-bit build |
| [Qwen3.8-Flash-Next](qwen38-flash-next.md) | MoE, 180B, 512 experts | Resident on a 64 GB Mac; n-gram offload |
| [DiffusionGemma](diffusiongemma.md) | Block-diffusion MoE | Experimental; a 16 GB build |

Every published build is in the
[TurboQuant-MLX collection](https://huggingface.co/collections/manjunathshiva/turboquant-mlx-6ab8b9771f2e996ba1c7ffe3)
on Hugging Face, and each card has the full measurements.
