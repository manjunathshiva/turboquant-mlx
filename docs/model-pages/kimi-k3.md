# Kimi K3 (2.8T MoE) — the largest model TurboQuant-MLX has run

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is Moonshot's 2.8-trillion-
parameter MoE: 93 layers (69 KDA linear-attention + 24 gated-NoPE MLA with
q-LoRA), 896 routed experts at top-16 plus 2 shared, and a **Stable LatentMoE**
where experts run in a 3584-d latent behind shared per-layer down/up projections
while the router reads the full 7168-d stream. `mlx-lm` has no native `kimi_k3`,
so the MLX port ships here — contributed by
[@anders94](https://github.com/anders94) in
[#71](https://github.com/manjunathshiva/turboquant-mlx/pull/71), verified at
≤ 2.7e-6 fp32 forward parity against `transformers`.

The published source is 1.56 TB of mxfp4 compressed-tensors, which TurboQuant
reads directly (bit-exact unpack, no fp16 round-trip):

```bash
pip install "turboquant-mlx-full[kimi]"   # tiktoken + blobfile for K3's tokenizer

python -m turboquant_mlx.convert --hf-path ./Kimi-K3 \
    --mlx-path ./Kimi-K3-tq3a-tqTe-down4-g64 \
    --bits 3 -g 64 --ternary-experts --expert-down-bits 4 --streaming
```

**This model does not fit resident on any Mac you can buy, at any bit width.**
2.8T parameters want roughly a 768 GB machine; the cheapest recipe here is
617 GB and 512 GB is the current ceiling. It runs by **streaming experts from
disk** — each token pages in only its router-selected experts:

```bash
sudo sysctl iogpu.wired_limit_mb=512000      # 512 GB box; resets on reboot

python -m turboquant_mlx.stream.stream_generate \
    --model ./Kimi-K3-tq3a-tqTe-down4-g64 \
    --prompt "Explain how Rayleigh scattering works." \
    --max-tokens 500 --cache-budget-gb auto \
    --max-active-experts 8 --prefetch-workers 16 --fanout
```

K3's tokenizer is custom code (`tokenization_kimi.py`). It runs without asking
only when `--model` / `--hf-path` is a local K3 directory you passed on the
command line; to load K3 by Hub repo id, add `--trust-remote-code`.

Measured on a **Mac Studio (512 GB)** with a 506 GB expert cache and the shipped
hot-expert list: **2.3 tok/s** at `--max-active-experts 8`, **1.2 tok/s** at
native top-16. Full numbers and the recipe breakdown are in
[`model_card_kimi_k3.md`](https://github.com/manjunathshiva/turboquant-mlx/blob/main/model_card_kimi_k3.md).

> `--fanout` matches how those numbers were measured — it parallelizes each
> layer's expert reads across the pool instead of reading them serially. It is
> off by default because it trades away the coalesced serial read path, which
> wins on bandwidth-bound external storage. It does not self-disable, so measure
> it on your own disk.
>
> `--max-active-experts 8` halves K3's native top-16. That is a 2× cut, the ratio
> validated elsewhere in this repo — but K-reduction changes which experts run
> and has broken agentic tool-calling on other models. Use
> `--max-active-experts 0` for native routing whenever quality matters.
