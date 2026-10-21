# NVIDIA Nemotron 3.5 Super VL (120B VLM, 512 experts)

**Hardware:** a 64 GB Mac (M4 Max, measured). NVIDIA's vision-language model mixes Mamba-2,
attention (8 of 88 layers) and a latent MoE (512 experts, 22 active per token), with a RADIO
vision tower. mlx-vlm >= 0.7.9 runs the architecture (`nemotron_h_omni`); this package sets
the quantization. Neither the BF16 original (249 GB) nor NVIDIA's NVFP4 (63 GiB) fits
resident on a 64 GB Mac. Needs `turboquant-mlx-full[vlm] >= 0.29.0`.

| Build | Size | Settings on a 64 GB Mac | Decode | Notes |
|---|---|---|---|---|
| [`tq3a-tqTe-g64`](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64) (ternary experts) | 29.29 GiB | **none** | ~7–8 tok/s | 44.7 GB peak at a 36K-token prompt; text 6/6, vision 3/3 |
| [`tq3-g64`](https://huggingface.co/manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3-g64) (3-bit) | 50.22 GiB | `sysctl iogpu.wired_limit_mb=60416` **and** `--prefill-step-size 256` | ~19–21 tok/s | without the smaller prefill step, a ~4,000-word prompt runs out of GPU memory |

```bash
pip install "turboquant-mlx-full[vlm]>=0.29.0"

python -m turboquant_mlx.generate_vlm \
    --model manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64 \
    --image invoice.png --prompt "What is the invoice number and the total due?" \
    --max-tokens 512 --temp 0.6

turboquant-serve-vlm --model manjunathshiva/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-tq3a-tqTe-g64 --port 8080
```

To convert it yourself (about 2 minutes, ~41 GB peak memory for the ternary build):

```bash
python -m turboquant_mlx.convert_vlm \
    --hf-path nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --mlx-path ./Nemotron-3.5-Super-VL-tq3a-tqTe-g64 \
    --bits 3 --group-size 64 --ternary-experts --quantize-extras
```

- **Reasoning is off by default:** the chat template thinks only with `enable_thinking=True`.
  With it on, the ternary build kept a 7,442-token trace coherent and answered correctly.
- **Memory barely grows with context:** only 8 layers hold KV (about 8 KB per token), so the
  peak at 36K tokens is close to the peak at 4K. `turboquant-plan` reads the geometry but
  currently projects about 3.5 GB below the measured peak for this model.
- `lm_head` and the vision projector `mlp1` stay off the polar path; the MTP head is dropped.
