# Vision models

Vision-language and diffusion models run through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
rather than mlx-lm, so they have their own three commands. Install the extra first:

```bash
pip install "turboquant-mlx-full[vlm]"   # adds mlx-vlm >= 0.7.9
```

Published vision builds: [Nemotron 3.5 Super VL 120B](../model-pages/nemotron-35-vl.md),
[Qwen3.8-27B](../model-pages/qwen38-27b.md), [Muse Glimmer 30B](../model-pages/muse-glimmer.md),
[Qwen3.8-Flash-Next](../model-pages/qwen38-flash-next.md) (vision tower included) and the
block-diffusion [DiffusionGemma](../model-pages/diffusiongemma.md).

## Run it

```bash
python -m turboquant_mlx.generate_vlm \
    --model manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64 \
    --image invoice.png \
    --prompt "What is the invoice number and the total due?" \
    --max-tokens 512 --temp 0.6
```

| Option | Use |
|---|---|
| `--image` | A file path or URL. Leave it out for a text-only prompt. |
| `--max-tokens`, `--temp` | As in `turboquant-generate` (`--temp` defaults to 0 here). |
| `--prefill-step-size` | Smaller steps use less memory on long prompts. Nemotron 3.5 VL's 3-bit build needs 256. |
| `--reasoning LEVEL` | Reasoning effort for templates that take one (Muse Glimmer: `low`, `medium`, `high`, `xhigh`). |
| `--no-think` | Least reasoning: `enable_thinking=False` for Qwen-style templates, `low` for Muse Glimmer. |
| `--max-denoising-steps`, `--max-canvas-length` | DiffusionGemma only. Fewer steps is faster at some cost in quality. |

## Serve it

`turboquant-serve-vlm` runs mlx-vlm's server (OpenAI **and** Anthropic routes, per-model tool
parsers, continuous batching) with TurboQuant's loader:

```bash
turboquant-serve-vlm --model manjunathshiva/Muse-Glimmer-30B-tq4-g64 \
    --reasoning-strength low --port 8080
```

- It binds `127.0.0.1` unless you pass `--host` (mlx-vlm on its own binds `0.0.0.0`).
- **Every request must name a `model`.** mlx-vlm answers HTTP 422 without one; use the id that
  `GET /v1/models` returns.
- Reasoning is separated from the answer: the right thinking delimiters are set per model, so
  the thinking arrives in `reasoning_content`. `--thinking-start-token` and
  `--thinking-end-token` override them.
- `--reasoning-strength` sets the default effort for clients that don't send one (most agent
  harnesses), and requests' `reasoning_effort` is translated for Muse Glimmer. Measured on
  Muse Glimmer `tq4-g64`: a tool-result turn took 54 completion tokens at `low` against 106 at
  the default `high`, for the same answer.
- Every other `mlx_vlm.server` option passes through, for example `--prefill-step-size` and
  `--enable-thinking`.

## Convert one

```bash
python -m turboquant_mlx.convert_vlm \
    --hf-path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --mlx-path ./Nemotron-3.5-Super-VL-tq3a-tqTe-g64 \
    --bits 3 --group-size 64 --ternary-experts --quantize-extras
```

`convert_vlm` takes the same core options as [`turboquant-convert`](convert.md) (`--bits`,
`--group-size`, `--attn-bits`, `--mlp-bits`, `--ternary-experts`, `--expert-down-bits`,
`--protect-expert-layers`) and keeps these at full precision automatically:

- the vision tower and audio towers, plus model-specific parts such as Nemotron VL's projector
  `mlp1`;
- MoE routers;
- **`lm_head` on large-vocabulary models** (Muse Glimmer, Qwen3.5-family, Nemotron VL). A
  polar-quantized `lm_head` would materialize its weights during every prefill longer than 256
  tokens, about 13–19 GB of temporary memory. Affine 8-bit costs up to about 0.8 GB more on
  disk (Muse Glimmer's 1.3B-parameter `lm_head`) and nothing extra at run time.

`--quantize-extras` stores what the polar path skips (embeddings, `lm_head`, the vision tower)
as MLX affine. Here `--extras-bits` defaults to **8** (4 or 8). Layers whose width isn't a
multiple of 64 stay in BF16.
