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
import json
import os
import time

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


def _model_file_bytes(model_path: str) -> int:
    """Total bytes of the model's safetensors (0 when none are found).

    glob.escape so a model_path with [ ] etc. still matches."""
    files = glob.glob(os.path.join(glob.escape(model_path), "model*.safetensors"))
    return sum(os.path.getsize(f) for f in files)


def _auto_page_cache(model_path: str) -> bool:
    """True iff the model's safetensors fit comfortably in RAM (page cache helps)."""
    try:
        # No files -> we can't size the model, so fail safe to F_NOCACHE
        # rather than treating it as 0 bytes.
        model_bytes = _model_file_bytes(model_path)
        if not model_bytes:
            return False
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


# Auto cache budget (ds4-style): size the expert cache from what the GPU can
# actually keep resident — a fraction of Metal's max recommended working set,
# minus the resident (non-expert) weights, minus a reserve for the KV cache
# and the transient prefill workspace — instead of a fixed guess.
_AUTO_WSS_FRACTION = 0.8
_AUTO_RESERVE_BYTES = int(2.0e9)
_AUTO_MIN_BUDGET_BYTES = int(0.5e9)

_SAFETENSORS_ITEMSIZE = {"U32": 4, "I32": 4, "F32": 4, "F16": 2, "BF16": 2,
                         "U8": 1}


def _streamed_expert_bytes(reader) -> int:
    """Total on-disk bytes of the tensors the streaming swap pages from disk
    (MoE switch_mlp weight/scales) — everything else stays resident."""
    total = 0
    for key, loc in reader._index.items():
        if "switch_mlp" not in key:
            continue
        if not (key.endswith(".weight") or key.endswith(".scales")):
            continue
        n = 1
        for d in loc.shape:
            n *= d
        total += n * _SAFETENSORS_ITEMSIZE.get(loc.dtype, 4)
    return total


def _auto_cache_budget(model_bytes: int, expert_bytes: int,
                       wss_bytes: int) -> int:
    """Pure budget math (unit-testable without a model): what's left of the
    working-set fraction after the resident weights and the reserve, clamped
    to [floor, all experts] — a budget past every expert buys nothing."""
    resident = max(0, model_bytes - expert_bytes)
    budget = int(_AUTO_WSS_FRACTION * wss_bytes) - resident - _AUTO_RESERVE_BYTES
    return max(_AUTO_MIN_BUDGET_BYTES, min(budget, expert_bytes))


# A model repo may ship its own routing profile alongside the weights (the
# ds4 hotlist idea): calibrate_experts.py's pin.json, renamed to this, uploaded
# with the checkpoint. Streaming then warm-starts on any machine with just
# --cache-budget-gb.
_HOTLIST_FILENAME = "hot_experts.json"


def _find_hotlist(model_path: str) -> str | None:
    cand = os.path.join(model_path, _HOTLIST_FILENAME)
    return cand if os.path.isfile(cand) else None


def _load_pin_spec(pin_file: str) -> "tuple[dict, list]":
    """Parse ``{"pin": [[layer, expert], ...]}`` keeping BOTH a per-layer set
    (to build pin keys during the layer swap) and the file's rank order
    (hottest first, so a budget-capped preload keeps the hottest).

    Raises ``ValueError`` with the offending file/entry on a malformed spec —
    a shipped hotlist comes from a downloaded repo, so the caller decides
    whether that is fatal (explicit ``--pin-file``) or skippable (auto-found).
    """
    with open(pin_file) as f:
        data = json.load(f)
    pairs = data.get("pin") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        raise ValueError(
            f'{pin_file}: expected {{"pin": [[layer, expert], ...]}}'
        )
    pin_layers: dict = {}
    pin_order: list = []
    seen = set()
    for item in pairs:
        try:
            le = (int(item[0]), int(item[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ValueError(
                f"{pin_file}: bad pin entry {item!r} (want [layer, expert])"
            ) from None
        if le in seen:
            continue
        seen.add(le)
        pin_order.append(le)
        pin_layers.setdefault(le[0], set()).add(le[1])
    return pin_layers, pin_order


def load_streaming(model_path, cache_budget_gb=3.0, fast: bool = False,
                   prefetch_workers: int = 8, prefetch_ahead: int = 0,
                   pin_file: str | None = None, max_active_experts: int = 4,
                   use_page_cache: bool | None = None, use_hotlist: bool = True,
                   preload_pins: bool = True, wire_memory: bool = False):
    """Returns (model, tokenizer, cache).

    cache_budget_gb bounds total resident expert memory (LRU-evicted). Pass
    the string ``"auto"`` to size it from the machine instead: 80% of Metal's
    max recommended working set, minus the resident (non-expert) weights,
    minus a 2 GB reserve for KV + prefill workspace, clamped to
    [0.5 GB, all experts].
    wire_memory=True (--wire-memory) additionally raises MLX's wired-memory
    limit to resident + budget + reserve (capped at the working set) so the
    weights and the expert cache stay resident under memory pressure — the
    ds4 mlock idea. Opt-in: on a roomy machine the OS page cache already
    keeps re-reads warm, and wiring takes memory from other apps.
    prefetch_workers parallelizes per-layer expert reads (1 = serial baseline).
    prefetch_ahead speculatively prefetches this many upcoming layers' experts
    (predicted from the previous token's routing); 0 disables prefetch.
    pin_file is an optional JSON {"pin": [[layer, expert], ...]} of hot experts
    to keep permanently resident (never LRU-evicted) — see calibrate_experts.py.
    When it is None and the model directory ships a ``hot_experts.json`` (same
    schema), that file is used automatically; use_hotlist=False/--no-hotlist
    disables the auto-discovery. Pinned experts are PRELOADED at startup in
    hotness order (coalesced batch reads, capped at 60% of the budget so the
    LRU keeps working room) instead of pinning lazily on first miss;
    preload_pins=False restores the lazy behavior.
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

    expert_bytes = _streamed_expert_bytes(reader)
    resident_bytes = max(0, _model_file_bytes(local_path) - expert_bytes)
    auto = isinstance(cache_budget_gb, str) and cache_budget_gb.lower() == "auto"
    if isinstance(cache_budget_gb, str) and not auto:
        cache_budget_gb = float(cache_budget_gb)
    if auto:
        wss = mx.device_info()["max_recommended_working_set_size"]
        budget_bytes = _auto_cache_budget(resident_bytes + expert_bytes,
                                          expert_bytes, wss)
        cache_budget_gb = budget_bytes / 1e9
        print(f"[stream] auto budget: {_AUTO_WSS_FRACTION:.0%} × working set "
              f"{wss / 1e9:.1f} GB − resident {resident_bytes / 1e9:.1f} GB − "
              f"reserve {_AUTO_RESERVE_BYTES / 1e9:.1f} GB -> cache "
              f"{cache_budget_gb:.1f} GB (experts on disk: "
              f"{expert_bytes / 1e9:.1f} GB)")
    if wire_memory:
        wss = mx.device_info()["max_recommended_working_set_size"]
        want = int(min(wss, resident_bytes + cache_budget_gb * 1e9
                       + _AUTO_RESERVE_BYTES))
        try:
            mx.set_wired_limit(want)
            print(f"[stream] wired-memory limit {want / 1e9:.1f} GB — weights "
                  "and expert cache stay resident under memory pressure")
        except Exception as exc:
            print(f"[stream] could not set wired limit ({exc}); continuing unwired")

    cache = ExpertCache(
        reader, int(cache_budget_gb * 1e9),
        prefetch_workers=prefetch_workers,
        prefetch_ahead=prefetch_ahead,
    )

    # Load the hot-expert pin spec (frequency-based pinning, #2 + shipped
    # hotlist, #6). Keyed by layer so we can pin all three projections of each
    # hot expert; the rank order drives the startup preload. A malformed
    # explicit --pin-file fails loud (the user asked for that exact file); a
    # malformed shipped hotlist only warns — it came with the download, and
    # the model runs fine without it.
    pin_layers: dict = {}
    pin_order: list = []
    if pin_file:
        pin_layers, pin_order = _load_pin_spec(pin_file)
    elif use_hotlist:
        shipped = _find_hotlist(local_path)
        if shipped:
            try:
                pin_layers, pin_order = _load_pin_spec(shipped)
                print(f"[stream] found shipped hot-expert list: {shipped}")
            except ValueError as exc:  # includes json.JSONDecodeError
                print(f"[stream] ignoring malformed shipped hotlist: {exc}")

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
                trit=res.trit,
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
    if pin_keys and preload_pins:
        t0 = time.time()
        specs = [
            (wkey, skey, e)
            for layer, e in pin_order
            for wkey, skey in cache._layer_keys.get(layer, ())
            if (wkey, e) in pin_keys
        ]
        n, dropped = cache.preload(specs)
        drop_note = (f", {dropped} past the 60%-of-budget cap left to the LRU"
                     if dropped else "")
        print(f"[stream] hotlist preload: {n} expert-projections "
              f"({cache.preload_bytes / 1e9:.2f} GB) in {time.time() - t0:.1f}s"
              f"{drop_note}")
    _cap_active_experts(layers, max_active_experts)
    return model, tok, cache
