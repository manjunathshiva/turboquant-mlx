"""Load a TurboQuant MoE model with its experts streamed from disk.

Reuses ``load_turboquant(lazy=True)`` to build the full model with weights
left mmap-backed and *unmaterialized*, then swaps every MoE expert layer
(``switch_mlp.{gate,up,down}_proj``) for a ``StreamingSwitchLinear`` before any
forward runs — so the big (num_experts, ...) expert tensors are never
evaluated into RAM. Everything else (embeddings, norms, attention, router,
shared expert) stays resident as usual.
"""

from __future__ import annotations

import glob
import os

import mlx.core as mx

from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

from .safetensors_reader import SafetensorsExpertReader
from .streaming_switch import ExpertCache, StreamingSwitchLinear

_PROJS = ("gate_proj", "up_proj", "down_proj")

# When the model file fits comfortably in RAM, trusting the OS page cache makes
# LRU-eviction re-reads come back from warm RAM instead of disk — measured 2.44x
# faster decode on a streamed 35B-A3B (scripts/flash_moe/trust_os_ab.py). When
# the model is larger than RAM (16 GB mini on a 70 GB MoE), the page cache would
# thrash and F_NOCACHE is correct. This fraction is the "comfortably fits" line.
_PAGE_CACHE_RAM_FRACTION = 0.6


def _total_ram_bytes() -> int:
    try:  # AttributeError too: os.sysconf is absent on some platforms (Windows)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))


def _auto_page_cache(model_path: str) -> bool:
    """True iff the model's safetensors fit comfortably in RAM (page cache helps)."""
    try:
        # glob.escape so a model_path with [ ] etc. still matches; no files ->
        # we can't size the model, so fail safe to F_NOCACHE rather than 0 bytes.
        files = glob.glob(os.path.join(glob.escape(model_path), "model*.safetensors"))
        if not files:
            return False
        model_bytes = sum(os.path.getsize(f) for f in files)
        ram = _total_ram_bytes()
    except Exception:
        return False  # any uncertainty -> the always-safe F_NOCACHE path
    fits = model_bytes < _PAGE_CACHE_RAM_FRACTION * ram
    print(f"[stream] page-cache auto: model {model_bytes/1e9:.1f} GB vs RAM {ram/1e9:.1f} GB "
          f"-> {'trust-OS (F_NOCACHE off)' if fits else 'F_NOCACHE on'} "
          f"(override with use_page_cache=/--use-page-cache)")
    return fits


def _cap_active_experts(layers, max_active: int) -> None:
    """Cap router top_k on every MoE block to ``min(native, max_active)``.

    The "K-reduction" lever (Flash-MoE): when experts stream from disk, per-token
    disk I/O scales with the number of *active* experts, not the total. Lowering
    top_k is mechanically clean — ``argpartition`` then selects fewer experts and
    ``norm_topk_prob`` renormalizes the gate weights over them — and the streaming
    switch loads only the selected experts. Measured on Qwen3.6-35B-A3B-tq3-g32
    (256 experts, native top_k=8): 8->4 is byte-identical on the 6-test stress
    harness and cuts streamed disk reads ~2x (78.9->37.8 GB) for ~1.4x decode in
    the disk-bound regime; K=2 collapses (broken JSON). Caps only — never raises.
    """
    if not max_active or max_active <= 0:
        return
    changed = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "top_k") or not hasattr(mlp, "switch_mlp"):
            continue
        native = int(mlp.top_k)
        new_k = min(native, max_active)
        if new_k != native:
            mlp.top_k = new_k
            changed.append(native)
    if changed:
        print(f"[stream] K-reduction: capped router top_k {changed[0]}->{min(changed[0], max_active)} "
              f"on {len(changed)} MoE blocks (~2x less disk I/O; pass "
              f"max_active_experts=0 / --max-active-experts 0 to use native routing)")


def load_streaming(model_path, cache_budget_gb: float = 3.0, fast: bool = False,
                   prefetch_workers: int = 8, prefetch_ahead: int = 0,
                   pin_file: str | None = None, max_active_experts: int = 4,
                   use_page_cache: bool | None = None):
    """Returns (model, tokenizer, cache).

    cache_budget_gb bounds total resident expert memory (LRU-evicted).
    prefetch_workers parallelizes per-layer expert reads (1 = serial baseline).
    prefetch_ahead speculatively prefetches this many upcoming layers' experts
    (predicted from the previous token's routing); 0 disables prefetch.
    pin_file is an optional JSON {"pin": [[layer, expert], ...]} of hot experts
    to keep permanently resident (never LRU-evicted) — see calibrate_experts.py.
    max_active_experts caps router top_k to min(native, this) on every MoE block
    (the Flash-MoE K-reduction lever: ~2x less streamed disk I/O at no quality
    cost up to K=4 on validated models). Default 4; set 0 to use native routing.
    use_page_cache controls the OS page cache for expert reads. None (default)
    auto-decides by model-size-vs-RAM: trust the OS (page cache on) when the
    model fits comfortably in RAM (~2.4x faster decode), F_NOCACHE when it does
    not (avoids page-cache thrash on a memory-constrained machine). True/False
    force it.
    """
    local_path = str(resolve_model_path(model_path))
    if use_page_cache is None:
        use_page_cache = _auto_page_cache(local_path)
    model, tok = load_turboquant(local_path, lazy=True, fast=fast)
    reader = SafetensorsExpertReader(local_path, use_page_cache=use_page_cache)
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
    _cap_active_experts(layers, max_active_experts)
    return model, tok, cache
