# Convert a model

Most people can download a [published build](../models.md). Convert your own when the model
or the bit mix you want isn't published yet.

Conversion is **data-free**: it needs only the model's weights, no calibration text. It reads a
Hugging Face repo or a local folder in **BF16 or FP16** and writes MLX safetensors plus a
`quantization` block in `config.json`.

```bash
turboquant-convert \
    --hf-path meta-llama/Llama-3.2-1B \
    --mlx-path ./llama-3.2-1b-tq3 \
    --bits 3 --group-size 64
```

(`python -m turboquant_mlx.convert` is the same command.) Vision and diffusion models use
`convert_vlm` instead: see [Vision models](vision.md).

## Choosing bits and group size

| Option | Values | Effect |
|---|---|---|
| `--bits`, `-b` | 2, 3, 4 (default 3) | Codebook width for every quantized layer. 3 is the usual choice; 4 for the most quality; 2 only for very large MoEs. |
| `--group-size`, `-g` | 32, 64, 128 (default 64) | How many weights share one scale. 32 is more accurate and a little bigger; 64 is the usual choice. |

## Mixed precision for Mixture-of-Experts

In a big MoE, the routed experts are most of the weights but each token uses only a few of
them, so they tolerate fewer bits than attention.

| Option | What it does |
|---|---|
| `--attn-bits` / `--mlp-bits` | Separate widths for attention and for MLP/expert layers. `--bits 3 --mlp-bits 2` gives the `tq3a-tq2e` builds. |
| `--mlp-group-size` | A finer group size for the expert tier only (16, 32, 64, 128). |
| `--ternary-experts` | Experts on a ternary `{−c, 0, +c}` codebook, about 1.6 bits per weight, packed as base-3 trits. Attention stays at `--bits`. These are the `tqTe` builds. |
| `--expert-down-bits` | Gives the experts' down projection its own width, above the ternary or 2-bit tier. `--ternary-experts --expert-down-bits 4` gives the `down4` builds. |
| `--protect-expert-layers` | Keeps the experts of chosen layers (for example `0-2,45-47`) at `--protect-bits` (default 3). |
| `--expert-layer-tiers` | A JSON file choosing a tier per MoE layer: `ternary`, `2`, `3` or `4`. |

```bash
# Ternary experts: Qwen3-235B in 53 GB, fully resident on a 64 GB Mac
turboquant-convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path ./qwen3-235b-tq3a-tqTe-g64 \
    --bits 3 --group-size 64 --ternary-experts --streaming

# Ternary experts with a 4-bit down projection: the agent-capable 35B for a 16 GB mini
turboquant-convert \
    --hf-path Qwen/Qwen3.6-35B-A3B \
    --mlx-path ./Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64 \
    --bits 3 --group-size 64 --ternary-experts --expert-down-bits 4
```

!!! warning "Sub-2-bit experts only hold on some models"
    Ternary and 2-bit experts rely on many experts sharing the work. Measured so far:

    - **Works** on models with many, wide experts: Qwen3-235B (128 experts, top-8),
      Qwen3.5-122B and Qwen3.6-35B (256), Laguna-S-2.1 (256, 1024 wide), Nemotron 3.5 Super VL
      (512, top-22).
    - **Fails** with narrow experts (Laguna-XS.2, 512 wide: it loops), with few active experts
      on a 4-bit source (GPT-OSS-120B, top-4, MXFP4), and on 32-expert models (GPT-OSS-20B
      becomes word salad at 2-bit).

    Always check a new build on real prompts, not only with the unit tests. A 4-bit down
    projection (`--expert-down-bits 4`) has rescued two failing ternary builds.

!!! note "Nemotron-H layer names"
    Nemotron-H models name their blocks `mixer.*`, which `--attn-bits`, `--mlp-bits` and
    `--mlp-group-size` don't recognise, so on Nemotron-H only `--bits` and `--group-size`
    apply. `--ternary-experts` and `--expert-down-bits` do work, because they pick experts by
    layer type.

## Models too big to convert in memory: `--streaming`

The in-memory converter holds the whole quantized model before saving, which limits a 64 GB
Mac to about 130B parameters at 3-bit. `--streaming` writes each layer to a shard as soon as it
is quantized and frees it, so the peak stays near one shard plus one layer, whatever the model
size. Qwen3-235B converts on a **16 GB Mac mini** this way.

Point `--mlx-path` at a disk with room for the output, and expect the source download to need
more space than the result. (`--dtype` is not supported with `--streaming`.)

## Everything else the polar path skips: `--quantize-extras`

TurboQuant compresses the linear layers. Embeddings, `lm_head` and (in vision models) the vision
tower are left in BF16 unless you add `--quantize-extras`, which stores them as MLX affine
(`--extras-bits` 2, 3, 4 or 8, default 4; `--extras-group-size`). It matters most on models
with a big lookup table: Qwen3.8-Flash-Next's n-gram table is 51.2B parameters, so without it a
ternary build is about 124 GiB instead of 54 GiB. Routers always stay at full precision.

## Other options

| Option | Use |
|---|---|
| `--rotation` | `hadamard` (default), `blockwise_hadamard` or `none`. Leave the default. |
| `--rotation-seed` | Seed for the random rotation signs (default 42). |
| `--use-qjl` | Adds a 1-bit residual correction (about +1 bit). Slow to decode; not recommended. |
| `--dtype` | Cast the source before quantizing: `float16`, `bfloat16`, `float32`. |
| `--keep-mtp` | Copies a multi-token-prediction head (Qwen3.8) into the output. Nothing uses it by default yet. |
| `--trust-remote-code` | Lets the source run its own tokenizer or model code. Needed for Kimi K3 by Hub id. Only use it for sources you trust. |

## Check the result

```bash
turboquant-plan --model ./qwen3-235b-tq3a-tqTe-g64       # does it fit, and with which settings?
turboquant-generate --model ./qwen3-235b-tq3a-tqTe-g64 --prompt "Why is the sky blue?"
```

To measure perplexity against the original, see `python -m turboquant_mlx.evaluate --help`
(needs the `[eval]` extra).
