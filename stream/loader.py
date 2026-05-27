"""Load a TurboQuant MoE model with its experts streamed from disk.

Reuses ``load_turboquant(lazy=True)`` to build the full model with weights
left mmap-backed and *unmaterialized*, then swaps every MoE expert layer
(``switch_mlp.{gate,up,down}_proj``) for a ``StreamingSwitchLinear`` before any
forward runs — so the big (num_experts, ...) expert tensors are never
evaluated into RAM. Everything else (embeddings, norms, attention, router,
shared expert) stays resident as usual.
"""

from __future__ import annotations

import mlx.core as mx

from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

from .safetensors_reader import SafetensorsExpertReader
from .streaming_switch import ExpertCache, StreamingSwitchLinear

_PROJS = ("gate_proj", "up_proj", "down_proj")


def load_streaming(model_path, cache_budget_gb: float = 3.0, fast: bool = False,
                   prefetch_workers: int = 8):
    """Returns (model, tokenizer, cache).

    cache_budget_gb bounds total resident expert memory (LRU-evicted).
    prefetch_workers parallelizes per-layer expert reads (1 = serial baseline).
    """
    local_path = str(resolve_model_path(model_path))
    model, tok = load_turboquant(local_path, lazy=True, fast=fast)
    reader = SafetensorsExpertReader(local_path)
    cache = ExpertCache(
        reader, int(cache_budget_gb * 1e9),
        prefetch_workers=prefetch_workers,
    )

    # Locate the transformer layer stack and its weight-key prefix. Multimodal
    # MoEs (qwen3_5_moe) nest it under `language_model.model.layers`; text-only
    # MoEs (deepseek_v2/v3, …) use `model.model.layers`.
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
        prefix = "language_model.model.layers"
    else:
        layers = model.model.layers
        prefix = "model.layers"
    swapped = 0
    for i, layer in enumerate(layers):
        sm = getattr(layer.mlp, "switch_mlp", None)
        if sm is None:
            continue
        for proj in _PROJS:
            res = getattr(sm, proj, None)
            if not isinstance(res, PolarQuantizedSwitchLinear):
                continue
            cb, sg = res.codebook, res.signs
            mx.eval(cb, sg)  # tiny — pin resident, let the rest of res be freed
            st = StreamingSwitchLinear(
                input_dims=res.input_dims,
                output_dims=res.output_dims,
                num_experts=res.num_experts,
                bits=res.bits,
                group_size=res.group_size,
                needs_rotation=res._needs_rotation,
                codebook=cb,
                signs=sg,
                weight_key=f"{prefix}.{i}.mlp.switch_mlp.{proj}.weight",
                scales_key=f"{prefix}.{i}.mlp.switch_mlp.{proj}.scales",
                cache=cache,
            )
            setattr(sm, proj, st)
            swapped += 1

    print(f"[stream] swapped {swapped} expert projections to streaming "
          f"(budget {cache_budget_gb:.1f} GB)")
    return model, tok, cache
