# DiffusionGemma-26B-A4B (block-diffusion MoE)

!!! warning "Experimental"
    DiffusionGemma generates by denoising and runs through mlx-vlm's diffusion sampler.
    TurboQuant builds work, but decode is much slower than MLX's native 4-bit build. On the
    current release stack (mlx-lm 0.32, mlx-vlm 0.7.9) some short prompts return an empty
    answer, while the stack they were built on (0.28.1 with mlx-vlm 0.6.3) answered correctly.
    That is being investigated. Prose prompts generate normally.

Google's [DiffusionGemma-26B-A4B](https://huggingface.co/google/diffusiongemma-26B-A4B-it) is
a block-diffusion Mixture-of-Experts model: 30 layers, 128 routed experts (top-8), ~25.2B total
and ~3.8B active parameters. Instead of writing one token at a time, it iteratively denoises a
**256-token canvas** (up to 48 steps, ~15–20 tokens per forward pass). It is multimodal.

Diffusion suits TurboQuant's design in one way: every denoising step processes the whole
canvas, so the cost of rotating the input is spread over many tokens. It doesn't suit expert
streaming: each step touches nearly all 128 experts.

## Builds

| Build | Size | Peak | Speed | Mac |
|---|---|---|---|---|
| [mlx-community 4-bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) (reference) | 15.0 GB | ~19 GB | ~35 tok/s | 24 GB+ |
| [`tq3-g32`](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32) | 13.8 GB | ~18 GB | ~4.6 tok/s | 24 GB+ |
| [`mini-g32`](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-mini-g32) | 9.79 GB | 12.9 GB | ~1.5 tok/s | 16 GB mini, with `sudo sysctl -w iogpu.wired_limit_mb=13824` |

The 16 GB build keeps attention at 3-bit, puts experts at 2-bit, and protects the first and
last three layers' experts at 3-bit. Plain 2-bit experts broke arithmetic on this model;
protecting those six layers fixed it for +0.2 GB. Embeddings, the dense per-layer MLP and the
vision tower are 8-bit affine; routers stay at full precision.

Use the 4-bit build if you want the fastest local DiffusionGemma. Use these for the smallest
footprint, or to run it on a 16 GB Mac at all.

## Run it

```bash
pip install "turboquant-mlx-full[vlm]"

python -m turboquant_mlx.generate_vlm \
    --model manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32 \
    --prompt "Write two sentences about the ocean." --max-tokens 256
```

`--max-denoising-steps` trades quality for speed: 24 is about twice as fast as the default cap
of 48 with a mild quality cost, and 16 shows visible artifacts. `--max-canvas-length 128`
reduces memory on 16 GB machines.

## Convert it

```bash
python -m turboquant_mlx.convert_vlm \
    --hf-path google/diffusiongemma-26B-A4B-it \
    --mlx-path ./diffusiongemma-26B-A4B-it-tq3-g32 --bits 3 -g 32
```

The 16 GB build:

```bash
python -m turboquant_mlx.convert_vlm \
    --hf-path google/diffusiongemma-26B-A4B-it \
    --mlx-path ./diffusiongemma-26B-A4B-it-mini-g32 \
    --bits 2 --attn-bits 3 -g 32 \
    --protect-expert-layers 0,1,2,27,28,29 --protect-bits 3 \
    --quantize-extras
```

The routers and the self-conditioning MLP always stay at full precision. The dense per-layer
MLP, the embeddings and the vision tower stay out of the polar path: in BF16 without
`--quantize-extras`, and 8-bit affine with it (layers whose width doesn't fit the group size
stay BF16).
