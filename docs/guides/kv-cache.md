# KV cache compression

The KV cache grows with every token of context. TurboQuant compresses it with the same recipe
it uses for weights (Hadamard rotation plus a Lloyd-Max codebook), at run time. The compressed
cache is expanded to FP16 only when attention reads it, so it works with attention sinks,
sliding windows and linear-attention layers.

```bash
turboquant-generate --model ./model-tq3 --prompt "..." \
    --kv-k-bits 8 --kv-v-bits 3 --kv-min-tokens 128
```

The same options work on [`turboquant-serve`](serve.md#options).

## Choosing the bits

Keys need more precision than values: softmax magnifies any error in K, while V tolerates
aggressive quantization. **Mixed K8/V3 is the recommended setting**, about 4× smaller than FP16.

| Weights | K bits | V bits | FP16 sink | When to use |
|---|---:|---:|---:|---|
| FP16 / BF16 | 8 | 3 | 128 | The default: no measurable quality loss. |
| FP16 / BF16 | 4 | 3 | 128 | Smaller; a small quality dip on dense attention. |
| **TurboQuant weights** | **8** | **3** | **128** | **Required.** Symmetric 3-bit K on top of 3-bit weights collapses after about 1K generated tokens. |
| Any | 8 | 4 | 128 | The highest-fidelity compressed setting. |

`--kv-min-tokens 128` keeps the first 128 tokens in FP16. Many models attend heavily to those
"sink" tokens.

## It saves memory; it rarely saves time

Whether compression speeds decoding up or slows it down depends on how big the KV cache is per
token. With a large per-token cache, the 4× smaller cache saves more memory bandwidth than
decompression costs. With a small one, decompression dominates.

| Model | FP16 KV | 3-bit KV | |
|---|---|---|---|
| GPT-OSS-20B | 90.6 tok/s | 29.9 tok/s | 3× slower |
| Qwen3.6-35B-A3B | 52.0 tok/s | 45.7 tok/s | 1.1× slower |
| GPT-OSS-120B | 6.4 tok/s | 8.7 tok/s | **1.4× faster** |

On models with a small cache, the penalty grows with context. Qwen3.6-35B-A3B (measured by a
community member on an M5 Pro) decodes 1.14× faster with FP16 at 65 tokens and 6.6× faster at
63K, while compression saves 2.08 GB at 63K. So on small-active MoEs, use KV compression to
**fit** a longer context, not to go faster. Similar-sized models don't all behave alike:
Qwen3.5-122B doesn't flip the way GPT-OSS-120B does.

The headline case is GPT-OSS-120B: with weights and KV both at 3-bit, its full **131K-token
context fits in 50 GB** on a 64 GB Mac, and decoding is faster than with an FP16 cache.

!!! note "Hybrid models"
    Only full-attention layers hold a growing KV cache. Linear-attention (Gated DeltaNet),
    Mamba and sliding-window layers are left untouched, so on a hybrid such as Qwen3.6 (10 of
    40 layers) or Nemotron (8 of 88), compression saves proportionally less. Laguna's KV cache
    is already small (most layers use a 512-token window), so don't compress it.

## Fused decode+attend (experimental)

Most of the decoding cost above is expanding the cache back to FP16 every token. An
experimental Metal kernel reads the packed cache directly and runs attention on the fly:

```bash
turboquant-serve --model ./model-tq3 --kv-k-bits 8 --kv-v-bits 3 \
    --kv-min-tokens 0 --kv-fused
```

| Model | Context | Standard | Fused | Speedup |
|---|---|---:|---:|---:|
| Llama-3.2-1B (dense) | 4.2K | 70.5 tok/s | 105.2 tok/s | 1.49× |
| Qwen3.6-35B-A3B (hybrid, 10/40 full-attention) | 2.1K | 40.2 tok/s | 43.9 tok/s | 1.09× |

Output is identical. It covers decode only, batch 1, with no FP16 sink
(`--kv-min-tokens 0`). Any step it can't handle (prefill, sink or sliding-window layers) falls
back to the standard path, so it is safe to turn on.

## From Python

```python
from mlx_lm.models.cache import make_prompt_cache
from turboquant_mlx.layers import convert_cache_to_turboquant, enable_fused_attend

cache = make_prompt_cache(model)                 # the right cache type per layer
cache = convert_cache_to_turboquant(
    cache,
    k_bits=8, v_bits=3,                          # keys need the precision
    min_tokens_before_quant=128,                 # first 128 tokens stay FP16
    group_size=64,
)
```

Layers that don't hold a growing KV cache (`RotatingKVCache`, `ArraysCache`) are left as they
are. For the fused kernel, call `enable_fused_attend(cache)` after converting (it needs
`min_tokens_before_quant=0`).
