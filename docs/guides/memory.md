# Memory and macOS settings

On Apple Silicon the CPU and GPU share one pool of memory, but macOS lets the **GPU** use only
part of it by default. For big models, that GPU limit (the "Metal wired limit") is what decides
whether a model fits, not the RAM on the box.

| Mac | GPU limit by default (approx.) |
|---|---|
| 16 GB Mac mini | ~10.5 GB |
| 48 GB | ~36 GB |
| 64 GB | ~48 GB |

`turboquant-plan` reads your Mac's real limit and tells you whether a model fits and which
settings it needs. See [Get started](../getting-started.md#1-check-that-it-fits-before-downloading).

## Raising the GPU limit

```bash
sudo sysctl -w iogpu.wired_limit_mb=60416    # example for a 64 GB Mac
```

It takes effect at once and lasts until a reboot (to keep it, add the line
`iogpu.wired_limit_mb=60416` to `/etc/sysctl.conf`). To go back to the default:

```bash
sudo sysctl -w iogpu.wired_limit_mb=0
```

Values used in these docs and on the model cards:

| Mac | Value | Leaves macOS about | Used for |
|---|---|---|---|
| 64 GB | `60416` | 5 GB | The largest resident builds (Qwen3-235B ternary, Qwen3.8-Flash-Next, Nemotron 3.5 VL 3-bit) |
| 64 GB | `57344` | 8 GB | Serving a ~50 GB model with some headroom |
| 48 GB | `43008` | 6 GB | |
| 16 GB | `13824` / `14336` | 2–2.5 GB | Qwen3.6-35B `down4`, Qwen3.8-27B mini, DiffusionGemma mini |

!!! warning "Leave room for macOS"
    Everything outside the GPU limit is what macOS and your apps have left. Close browsers,
    Electron apps and video editors before running near the limit: even an idle browser can
    hold 1–2 GB.

## What makes up the peak

```
peak = weights + KV cache + prefill workspace + runtime reserve
```

- **Weights** are fixed: the size on disk, near enough.
- **KV cache** grows with context, but only on full-attention layers. Hybrids grow it far more
  slowly (Nemotron 3.5 VL: 8 of 88 layers, about 8 KB per token). [KV compression](kv-cache.md)
  shrinks it about 4×.
- **Prefill workspace** is temporary memory while the prompt is processed, in chunks of
  `--prefill-step-size` tokens (default 2048). **It scales with the chunk size**, which makes
  it the easiest lever:

    ```bash
    turboquant-generate ... --prefill-step-size 256     # or 128 near the limit
    turboquant-serve    ... --prefill-step-size 256
    ```

    For example, the Nemotron 3.5 VL 3-bit build runs out of GPU memory on a ~4,000-word
    prompt at the default step, and fits at 256.

## When it doesn't fit

This error means the GPU limit was exceeded:

```
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

In order:

1. Smaller prefill steps: `--prefill-step-size 256`, then 128.
2. Raise the GPU limit with `sysctl` (above).
3. Compress the KV cache: `--kv-k-bits 8 --kv-v-bits 3`.
4. When serving, cap the prompt cache: `--prompt-cache-max-gb 2`, or `--prompt-cache-size 1`.
   (mlx-lm's `--prompt-cache-bytes` applies only when requests are batched.) See
   [Serve](serve.md#near-the-memory-limit).
5. Pick a smaller build, or [stream the experts](streaming.md).

## Long prompts close to the limit

Prefilling a very long prompt near the GPU limit can trip the macOS kernel watchdog and
restart the Mac. The cause is how fast memory is being committed, not just the peak: a
fixed-size 41 GB model is stable, while a KV cache growing quickly during a 63K-token prefill
was not. On a 48 GB Mac, contexts up to ~14.5K were safe on Qwen3.6-35B; 63K worked with K8/V3
KV (28.7 GB peak) but not with an FP16 cache. Keep some headroom for very long contexts.

## Converting near the memory limit

Converting a model whose quantized size is close to your RAM pushes macOS into swap, and the
swap files live on your system disk. Keep plenty of free disk space during a big conversion (at
least the size of the output plus ~15 GB), or use `turboquant-convert --streaming`, which keeps
memory use low. See [Convert a model](convert.md#models-too-big-to-convert-in-memory-streaming).
