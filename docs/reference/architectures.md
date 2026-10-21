# Supported architectures

What has been converted and run with TurboQuant-MLX. "Tested" means converted and generated on real weights; model pages have the details.


| Architecture | Model Type | MoE | Status |
|-------------|-----------|-----|--------|
| LLaMA / Llama 3 | `llama` | No | Tested |
| Qwen2 / Qwen2.5 | `qwen2` | No | Tested |
| Qwen3.5 / Qwen3.8 (text) | `qwen3_5` | No | Tested |
| Qwen3.5-family VLM (hybrid Gated-DeltaNet + full attention, via **mlx-vlm**) | `qwen3_5` | No | Tested (Qwen3.8-27B, 27.8B dense: `tq4-g64` = 15.15 GiB, `tq3-mini-g64` = 11.55 GiB **verified on a real 16 GB Mac mini**; both pass an Opencode agentic task and the vision path). `lm_head` is kept off the polar path — see [Qwen3.8-27B](../model-pages/qwen38-27b.md) |
| Mistral | `mistral` | No | Tested |
| Qwen1.5-MoE | `qwen2_moe` | Yes | Tested |
| GPT-OSS | `gpt_oss` | Yes | Tested |
| Qwen3.5-MoE / Qwen3.6-35B-A3B | `qwen3_5_moe` | Yes (256 experts) | Tested (122B, 35B-A3B); 35B streams on a 16 GB Mac mini |
| Qwen3-MoE | `qwen3_moe` | Yes (128 experts, top-8) | Tested — Qwen3-235B-A22B converted to a hybrid **tq3a-tq2e** build (70.5 GB) on a 16 GB Mac mini via `--streaming`; streams and passes 5/6 quality probes on a 64 GB Mac |
| Nemotron-H (Mamba/attention hybrid) | `nemotron_h` | Yes (512 experts w/ latent MoE on Super-120B) | Tested (Nano-4B, Super-120B) — requires mlx-lm ≥ 0.31.3 |
| Poolside Laguna (SWA + global attn, per-head gating) | `laguna` | Yes (256 experts, top-8/top-10) | Tested (XS.2 33B at 3-bit; S-2.1 118B at ternary 1.58-bit — both convert + generate + Opencode agentic pass; expert streaming wired for both). Custom arch — MLX port + loader shipped here (mlx-lm has no native `laguna`); logit-parity 2e-7 vs transformers |
| DeepSeek-V2 / V3 (MLA + MoE) | `deepseek_v2` / `deepseek_v3` / `deepseek_v32` | Yes (SwitchGLU experts) | V2-Lite tested (convert + resident + streaming, coherent at 3-bit). V3/V3.2 share the MLA+MoE layout and reuse the config, but are **untested** (they need ~250 GB of disk) |
| Kimi K3 (KDA linear-attn + gated-NoPE MLA, Stable LatentMoE) | `kimi_k3` | Yes (896 experts, top-16 + 2 shared) | Tested (2.8T: mxfp4 source → 931 GB `tq3a-tqTe-down4-g64`, streams from disk on a 512 GB Mac Studio; does **not** fit resident on any Mac). Custom arch — MLX port shipped here (mlx-lm has no native `kimi_k3`); fp32 forward parity ≤ 2.7e-6 vs transformers. Needs `[kimi]` extra |
| Sarvam MoE (Megatron-style fused QKV, DeepSeek-V3 routing) | `sarvam_moe` | Yes | Port tested against fp32 (7122/7122 tensors, layer-1 parity 5.4e-7). Custom arch — MLX port shipped here. **No quantized build published**: 4-bit degenerates on long generations (53% of trials; mlx-lm affine 4-bit 100%) |
| Muse Glimmer (dense VLM: gated attention, sandwich norms, sliding/NoPE mix, ViT-G/14) | `muse_glimmer` | No | Tested (29.8B: `tq4-g64` beats affine 4-bit on PPL — 4.3315 vs 4.3798 — in 25% less space; `tq3-g64` = 12.53 GiB). Structural match verified against the checkpoint, 1436/1436 tensors. Needs **mlx-vlm >= 0.6.12** ([#1838](https://github.com/Blaizzy/mlx-vlm/pull/1838), merged) — see [Muse Glimmer](../model-pages/muse-glimmer.md) |
| Qwen3.8-Flash-Next (Gated DeltaNet + Qwen Sparse Attention, n-gram/PLE embedding, hyper-connections; Qwen3-VL tower) | `qwen4_exp` | Yes (512 experts, top-10, 640 wide) | Tested (180B: `tq4a-tq2e-g64` = 52.00 GiB, fully resident on a 64 GB Mac at 16–18 tok/s; vision through mlx-vlm's Qwen3-VL tower). Architecture vendored from the open mlx-lm PR [#1788](https://github.com/ml-explore/mlx-lm/pull/1788) until it merges. `--kv-bits` has no effect on it — see [Qwen3.8-Flash-Next](../model-pages/qwen38-flash-next.md) |
| NVIDIA Nemotron 3.5 Super VL (Mamba-2 + attention + latent MoE, RADIO vision tower, via **mlx-vlm**) | `nemotron_h_omni` | Yes (512 experts, top-22, latent MoE) | Tested (120B total, 12B active: ternary `tq3a-tqTe-g64` = 29.29 GiB, resident on a 64 GB Mac at default settings, text 6/6 and vision 3/3; `tq3-g64` = 50.22 GiB at ~20 tok/s with `sysctl` + `--prefill-step-size 256`). mlx-vlm >= 0.7.9 runs the model; `lm_head` and the vision projector `mlp1` stay off the polar path. Needs turboquant-mlx-full >= 0.29.0 |
| DiffusionGemma (block-diffusion MoE, via **mlx-vlm**) | `diffusion_gemma` | Yes (128 experts, top-8) | Tested (26B-A4B: convert + block-diffusion sampler, coherent at 3-bit — [HF](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32)). **Experimental**: decode is much slower than native 4-bit until a batched codebook gather-GEMM kernel lands |

## mlx-vlm architectures (vision and diffusion)

Architectures that live in [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) rather
than mlx-lm convert and run through dedicated entry points (v0.7.0+):

```bash
pip install "turboquant-mlx-full[vlm]"   # adds mlx-vlm >= 0.7.9

# Convert (vision towers, routers, and known quant-sensitive blocks stay full precision)
python -m turboquant_mlx.convert_vlm \
    --hf-path google/diffusiongemma-26B-A4B-it \
    --mlx-path ./diffusiongemma-26B-A4B-it-tq3-g32 --bits 3 -g 32

# Generate (runs mlx-vlm's sampler — block-diffusion denoising for DiffusionGemma)
python -m turboquant_mlx.generate_vlm \
    --model ./diffusiongemma-26B-A4B-it-tq3-g32 \
    --prompt "Write a short paragraph about the ocean." --max-tokens 256
```
