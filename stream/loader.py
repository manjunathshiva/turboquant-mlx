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
                   prefetch_workers: int = 8, prefetch_ahead: int = 0,
                   pin_file: str | None = None):
    """Returns (model, tokenizer, cache).

    cache_budget_gb bounds total resident expert memory (LRU-evicted).
    prefetch_workers parallelizes per-layer expert reads (1 = serial baseline).
    prefetch_ahead speculatively prefetches this many upcoming layers' experts
    (predicted from the previous token's routing); 0 disables prefetch.
    pin_file is an optional JSON {"pin": [[layer, expert], ...]} of hot experts
    to keep permanently resident (never LRU-evicted) — see calibrate_experts.py.
    """
    local_path = str(resolve_model_path(model_path))
    model, tok = load_turboquant(local_path, lazy=True, fast=fast)
    reader = SafetensorsExpertReader(local_path)
    cache = ExpertCache(
        reader, int(cache_budget_gb * 1e9),
        prefetch_workers=prefetch_workers,
        prefetch_ahead=prefetch_ahead,
    )

    # Load the hot-expert pin spec (frequency-based pinning, #2). Keyed by layer
    # so we can pin all three projections of each hot expert.
    pin_layers: dict = {}
    if pin_file:
        import json
        with open(pin_file) as f:
            for layer, expert in json.load(f).get("pin", []):
                pin_layers.setdefault(int(layer), set()).add(int(expert))

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
    pin_keys: set = set()
    for i, layer in enumerate(layers):
        sm = getattr(layer.mlp, "switch_mlp", None)
        if sm is None:
            continue
        proj_keys = []
        for proj in _PROJS:
            res = getattr(sm, proj, None)
            if not isinstance(res, PolarQuantizedSwitchLinear):
                continue
            cb, sg = res.codebook, res.signs
            mx.eval(cb, sg)  # tiny — pin resident, let the rest of res be freed
            wkey = f"{prefix}.{i}.mlp.switch_mlp.{proj}.weight"
            skey = f"{prefix}.{i}.mlp.switch_mlp.{proj}.scales"
            for e in pin_layers.get(i, ()):  # pin every projection of a hot expert
                pin_keys.add((wkey, e))
            st = StreamingSwitchLinear(
                input_dims=res.input_dims,
                output_dims=res.output_dims,
                num_experts=res.num_experts,
                bits=res.bits,
                group_size=res.group_size,
                needs_rotation=res._needs_rotation,
                codebook=cb,
                signs=sg,
                weight_key=wkey,
                scales_key=skey,
                cache=cache,
                layer_idx=i,
                # one trigger per layer fires the next-layer prefetch; gate_proj
                # is first in _PROJS so it fires with maximum lead time.
                is_trigger=(proj == _PROJS[0]),
            )
            setattr(sm, proj, st)
            proj_keys.append((wkey, skey))
            swapped += 1
        if proj_keys:
            cache.register_layer(i, proj_keys)

    cache._pin_keys = pin_keys
    pin_note = f", pinned {len(pin_keys)} hot expert-projections" if pin_keys else ""
    print(f"[stream] swapped {swapped} expert projections to streaming "
          f"(budget {cache_budget_gb:.1f} GB{pin_note})")
    return model, tok, cache
