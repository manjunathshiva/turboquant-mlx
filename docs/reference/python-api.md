# Python API

The command-line tools cover most uses. These are the functions behind them, for using
TurboQuant from your own code. Signatures are from `turboquant-mlx-full` 0.29.0.

## Load and generate

```python
from turboquant_mlx.generate import load_turboquant
from mlx_lm import generate

model, tokenizer = load_turboquant("manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64")
print(generate(model, tokenizer, prompt="Why is the sky blue?", max_tokens=200))
```

**`turboquant_mlx.generate.load_turboquant(model_path, lazy=False, fast=False, ngram_offload=False, tokenizer_config=None)`**

Loads a TurboQuant model and its tokenizer, and returns `(model, tokenizer)`. `model_path` is a
local folder or a Hugging Face repo id. The model is a normal mlx-lm model whose quantized
layers were swapped for TurboQuant layers, so mlx-lm's `generate`, `stream_generate` and
prompt caches all work with it.

- `lazy`: don't evaluate the weights immediately.
- `fast`: skip the QJL correction on builds that have one.
- `ngram_offload`: Qwen3.8-Flash-Next only; serves the n-gram table from memory-mapped files.
- `tokenizer_config`: extra tokenizer options, for example `{"trust_remote_code": True}`.

## Vision models

**`turboquant_mlx.integration.vlm.load_turboquant_vlm(model_path, lazy=False)`**

Loads a TurboQuant vision or diffusion model through mlx-vlm and returns
`(model, processor, config)`. Use it with mlx-vlm's `generate` and
`prompt_utils.apply_chat_template`. Needs the `[vlm]` extra.

## Models bigger than memory

**`turboquant_mlx.stream.loader.load_streaming(model_path, cache_budget_gb=3.0, ...)`**

Loads a Mixture-of-Experts model with its experts streamed from disk, and returns
`(model, tokenizer, cache)`. Keep a reference to `cache`: it is the expert cache. The keyword
options match the [streaming command line](../guides/streaming.md#options):
`prefetch_workers=8`, `prefetch_ahead=0`, `max_active_experts=4` (use `0` for full routing),
`use_page_cache=None` (auto), `use_hotlist=True`, `wire_memory=False`, `learn_experts=True`,
`pin_file=None`, `ngram_offload=False`, `tokenizer_config=None`.

## Convert

**`turboquant_mlx.convert.convert(hf_path, mlx_path="mlx_model", bits=3, group_size=64, ...)`**

What `turboquant-convert` runs. The keyword arguments mirror its options: `rotation`,
`rotation_seed`, `use_qjl`, `dtype`, `attn_bits`, `mlp_bits`, `mlp_group_size`,
`ternary_experts`, `expert_down_bits`, `keep_mtp`, `quantize_extras`, `extras_bits`,
`extras_group_size`, `protect_expert_layers`, `protect_bits`, `expert_layer_tiers` and
`trust_remote_code`. See [Convert a model](../guides/convert.md).

**`turboquant_mlx.TurboQuantConfig`**

A dataclass with the quantization settings: `bits`, `group_size`, `use_qjl`, `rotation`,
`rotation_seed`, `attn_bits`, `mlp_bits`, `mlp_group_size`, `ternary_experts`,
`expert_down_bits`, `protect_expert_layers`, `protect_bits`, `expert_layer_tiers`. It is saved
as the `quantization` block of a converted model's `config.json`
(`TurboQuantConfig.from_dict` reads it back).

**`turboquant_mlx.quantize_model.turboquant_quantize(model, config, tq_config, on_quantized=None)`**

Quantizes an already-loaded MLX model in place and returns `(model, config)`. `on_quantized`
is called with each layer as it is quantized; the streaming converter uses it to write and free
layers one at a time.

## KV cache compression

```python
from mlx_lm.models.cache import make_prompt_cache
from turboquant_mlx.layers import convert_cache_to_turboquant, enable_fused_attend

cache = convert_cache_to_turboquant(
    make_prompt_cache(model),
    k_bits=8, v_bits=3, min_tokens_before_quant=128, group_size=64,
)
```

**`turboquant_mlx.layers.convert_cache_to_turboquant(prompt_cache, tq_bits=None, group_size=64, seed=42, *, k_bits=None, v_bits=None, min_tokens_before_quant=0)`**

Replaces each `KVCache` in a prompt cache with a `TurboQuantKVCache` and leaves other cache
types (sliding-window, linear-attention) alone. Use `k_bits`/`v_bits` (K8/V3 recommended);
`tq_bits` is the older single width for both.

**`turboquant_mlx.layers.enable_fused_attend(caches)`**

Turns on the experimental fused decode+attend kernel for every `TurboQuantKVCache` in the
list. It needs `min_tokens_before_quant=0`, and falls back to the standard path wherever it
doesn't apply. See [KV cache compression](../guides/kv-cache.md#fused-decodeattend-experimental).

**`turboquant_mlx.layers.TurboQuantKVCache(tq_bits=None, group_size=64, seed=42, *, k_bits=None, v_bits=None, min_tokens_before_quant=0)`**

The compressed cache class itself, if you build caches by hand.
