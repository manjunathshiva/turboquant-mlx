# Muse Glimmer (30B dense VLM)

[Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) is Meta
Superintelligence Lab's 29.8B-parameter dense multimodal model, built for
agentic work on consumer hardware. Architecturally it is Gemma2-style sandwich
norms plus **gated attention** (a sigmoid output gate alongside Q/K/V), a 3:1
sliding(2048)/full attention pattern with **NoPE on the full-attention layers**,
final logit softcapping, and a 1.9B ViT-G/14 perception encoder.

**This is the one case where TurboQuant beats MLX's affine quantizer outright
on quality**, at matched bit width — and does it while quantizing more of the
model:

| build | size | PPL ↓ | decode | notes |
|---|---|---|---|---|
| `mlx-community/Muse-Glimmer-30B-4bit` | 19.88 GiB | 4.3798 | 26.1 tok/s | embedding + vision tower stay bf16 |
| **TurboQuant `tq4-g64`** | **14.88 GiB** | **4.3315** | 9.4 tok/s | everything quantized |
| TurboQuant `tq3-g64` | 12.53 GiB | 5.0454 | 10.9 tok/s | smallest coherent build |

800-token fixed-passage perplexity, greedy, M4 Max. The affine build is 2.4–2.8×
faster to decode — the usual codebook-vs-affine trade. **Do not read the tq3 row
as a TurboQuant deficiency**: the gap there is the 3-bit step, which is exactly
what the matched-4-bit control isolates.

Most of the affine build's extra 5 GB is the embedding and the vision tower,
which mlx-vlm declines to quantize. `embed_tokens` is a `NormedEmbedding` whose
inherited `to_quantized` would silently drop its RMS normalization; TurboQuant
quantizes it through a norm-preserving subclass instead.

```bash
# mlx-vlm 0.6.12 shipped the Muse Glimmer classes, so this is a plain install.
pip install "turboquant-mlx-full[vlm]"

python -m turboquant_mlx.convert_vlm \
  --hf-path meta-models/Muse-Glimmer-30B \
  --mlx-path ./Muse-Glimmer-30B-tq4-g64 \
  --bits 4 --group-size 64 --quantize-extras --extras-bits 4

python -m turboquant_mlx.generate_vlm \
  --model ./Muse-Glimmer-30B-tq4-g64 --prompt "..." --max-tokens 256
```

`--extras-bits 4` matters here: the embedding and vision tower are 3.3B
parameters between them, ~3.9 GB at the 8-bit default versus ~1.8 GB at 4-bit.

> **16 GB Mac mini: yes — measured.** `tq3-g64` runs **resident** on a
> Mac16,10 (macOS 26.5.2) with `iogpu.wired_limit_mb=14336`, at every prompt
> length tried up to 5068 tokens:
>
> | prompt | peak | prefill | decode |
> |---|---|---|---|
> | 268 tok | 13.44 GiB | 19.9 tok/s | 3.70 tok/s |
> | 868 tok | 14.13 GiB | 20.2 tok/s | 3.44 tok/s |
> | 2068 tok | 13.91 GiB | 17.0 tok/s | 3.38 tok/s |
> | 5068 tok | 14.21 GiB | 6.0 tok/s | 3.56 tok/s |
>
> Meta's own smallest Apple Silicon artifact is 17.95 GB, text-only and without
> the drafter — larger than the machine's whole RAM. Two caveats: headroom at
> 5068 tokens is **0.20 GB**, and prefill collapses past ~2000 tokens (20 → 6
> tok/s) while decode stays flat, so long prompts are batch work. The smaller
> alternative does **not** work — a `tq3a-tq2e-g64` hybrid fits at 9.63 GiB but
> measures **PPL 7.5547** (vs 5.0454) and visibly corrupts text, the same way
> 2-bit experts failed on 32-expert GPT-OSS-20B. 2-bit on a *dense* MLP is not
> a usable tier.

Muse Glimmer also ships a [DFlash block-diffusion
drafter](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) (5.11 GB,
5 layers, block 16, tapping target layers {1,13,25,37,49}). It is not wired up
here. Worth noting it is a *dense* target, which is the regime where DFlash
speculation paid off in earlier testing rather than the MoE regime where it
did not.
