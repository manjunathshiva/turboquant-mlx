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

from turboquant_mlx.cache_budget import (
    KV_RESERVE_BYTES,
    RUNTIME_OVERHEAD_BYTES,
    auto_cache_budget,
    projected_peak_bytes,
)
from turboquant_mlx.expert_naming import SWITCH_ATTRS, is_streamed_expert_key
from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

from .safetensors_reader import SafetensorsExpertReader
from .streaming_switch import ExpertCache, StreamingSwitchLinear

_PROJS = ("gate_proj", "up_proj", "down_proj")

# Attribute name of the stacked-expert container on an MoE block, by convention.
# qwen3_5_moe/deepseek call it `switch_mlp`; laguna's LagunaMoE holds a plain
# mlx-lm `SwitchGLU` at `experts`. The name is also the weight-key segment
# (`...mlp.<name>.gate_proj.weight`), so whichever attribute matched has to be
# the one the reader is pointed at — hence _find_switch returns both.
# `shared_experts` is deliberately absent: it is a dense always-on MLP that must
# stay resident, and its projections are PolarQuantizedLinear anyway.
# Shared with plan.py so the module swap and the byte accounting agree.
_SWITCH_ATTRS = SWITCH_ATTRS


def _find_switch(mlp):
    """Return ``(attr_name, module)`` for an MoE block's stacked-expert container.

    ``(None, None)`` when the block has none (dense layer, or an expert container
    holding something other than quantized switch projections). The type check is
    deliberate here: the caller splices ``attr_name`` into a weight key and hands
    it to the reader, so claiming a container whose projections aren't quantized
    switch layers would point the reader at a tensor that doesn't exist.
    """
    if mlp is None:
        return None, None
    for name in _SWITCH_ATTRS:
        sm = getattr(mlp, name, None)
        if sm is None:
            continue
        if any(isinstance(getattr(sm, p, None), PolarQuantizedSwitchLinear)
               for p in _PROJS):
            return name, sm
    return None, None


def _has_switch(mlp) -> bool:
    """True if this mlp looks like an MoE block, by container presence alone.

    Deliberately looser than :func:`_find_switch`: routing changes (K-reduction)
    only need to know "is this a router-driven MoE block", and apply equally to
    quantized and unquantized experts.
    """
    return mlp is not None and any(
        getattr(mlp, name, None) is not None for name in _SWITCH_ATTRS)

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
        if mlp is None or not hasattr(mlp, "top_k") or not _has_switch(mlp):
            continue
        native = int(mlp.top_k)
        new_k = min(native, max_active)
        if new_k != native:
            mlp.top_k = new_k
            changed.append(native)
    if changed:
        native = changed[0]
        print(f"[stream] K-reduction: capped router top_k {native}->{min(native, max_active)} "
              f"on {len(changed)} MoE blocks (~2x less disk I/O; pass "
              f"max_active_experts=0 / --max-active-experts 0 to use native routing)")
        if native > 2 * max_active:
            # Halving K was measured byte-identical (Qwen 8->4); deeper cuts
            # are unvalidated. Kimi K3's native top-16 at the default cap of 4
            # would be a 4x truncation — flag it rather than silently degrade.
            print(f"[stream] WARNING: top_k cut {native}->{max_active} is more "
                  f"aggressive than the validated 2x — quality may degrade; "
                  f"consider --max-active-experts {native // 2} or higher")


# Auto cache budget (ds4-style): size the expert cache from what the GPU can
# actually keep resident, rather than a fixed guess. The arithmetic and the
# measured constants behind it are in cache_budget.py, shared with plan.py.
_AUTO_RESERVE_BYTES = RUNTIME_OVERHEAD_BYTES + KV_RESERVE_BYTES

_SAFETENSORS_ITEMSIZE = {"U32": 4, "I32": 4, "F32": 4, "F16": 2, "BF16": 2,
                         "U8": 1}


def _streamed_expert_bytes(reader) -> int:
    """Total on-disk bytes of the tensors the streaming swap pages from disk
    (per-expert weight/scales) — everything else stays resident."""
    total = 0
    for key, loc in reader._index.items():
        if not is_streamed_expert_key(key):
            continue
        n = 1
        for d in loc.shape:
            n *= d
        total += n * _SAFETENSORS_ITEMSIZE.get(loc.dtype, 4)
    return total


def _auto_cache_budget(model_bytes: int, expert_bytes: int,
                       wss_bytes: int) -> int:
    """Pure budget math (unit-testable without a model).

    The arithmetic lives in cache_budget.py so that plan.py predicts exactly
    what happens here — those two drifted apart once already.
    """
    resident = max(0, model_bytes - expert_bytes)
    return auto_cache_budget(wss_bytes, resident, expert_bytes)


# A model repo may ship its own routing profile alongside the weights (the
# ds4 hotlist idea): calibrate_experts.py's pin.json, renamed to this, uploaded
# with the checkpoint. Streaming then warm-starts on any machine with just
# --cache-budget-gb.
_HOTLIST_FILENAME = "hot_experts.json"

# mirrored from usage_profile so the loader can report progress toward maturity
from turboquant_mlx.stream.usage_profile import _MIN_ROUTINGS as _LEARN_MIN_ROUTINGS


def _save_usage(profile, path) -> None:
    """atexit hook: merge this run into the decayed history.

    Never raises: this runs at interpreter shutdown, and a profile is an
    optimisation — losing it must not turn a completed generation into a
    traceback. Failures are reported rather than swallowed silently.
    """
    try:
        if profile.routings and profile.update_on_disk(path):
            print(f"[stream] expert-usage profile updated: {path} "
                  f"(+{profile.routings:,} routings this run)")
    except Exception as exc:                       # noqa: BLE001 - see docstring
        print(f"[stream] could not update expert-usage profile at {path}: {exc}")


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


def _pin_spec_to_layers(spec: dict) -> "tuple[dict, list]":
    """Same shape as _load_pin_spec, for an in-memory spec (learned profile)."""
    pin_layers: dict = {}
    pin_order: list = []
    seen = set()
    for item in spec.get("pin", []):
        le = (int(item[0]), int(item[1]))
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
                   preload_pins: bool = True, wire_memory: bool = False,
                   learn_experts: bool = True, usage_file: str | None = None):
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
        print(f"[stream] auto budget: working set {wss / 1e9:.1f} GB − "
              f"resident {resident_bytes / 1e9:.1f} GB − reserve "
              f"{_AUTO_RESERVE_BYTES / 1e9:.1f} GB -> cache "
              f"{cache_budget_gb:.1f} GB (experts on disk: "
              f"{expert_bytes / 1e9:.1f} GB, projected peak "
              f"{projected_peak_bytes(budget_bytes, resident_bytes) / 1e9:.1f} GB)")
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

    # Learning cache (colibri #3). `history` is what previous runs recorded and
    # decides this run's pins; `profile` collects THIS run and is merged into
    # the history at exit. The profile object is created even when the pins come
    # from a shipped hotlist — that is how a cold machine bootstraps its own
    # list while still benefiting from the shipped prior today.
    from turboquant_mlx.stream.usage_profile import UsageProfile, profile_path
    # abspath: a bare relative --usage-file has no dirname, and the atexit hook
    # runs after any os.chdir the host app did — both would silently lose it.
    profile_file = (os.path.abspath(os.path.expanduser(usage_file))
                    if usage_file else profile_path(local_path))
    history = UsageProfile.load(profile_file)
    profile = UsageProfile(history.num_experts) if learn_experts else None

    cache = ExpertCache(
        reader, int(cache_budget_gb * 1e9),
        prefetch_workers=prefetch_workers,
        prefetch_ahead=prefetch_ahead,
        usage_profile=profile,
    )
    # Fold this run into the persisted history at exit. Registered rather than
    # left to the caller because the win only materialises across runs, and a
    # profile that is never written is a feature that never fires.
    if profile is not None and profile_file:
        import atexit
        atexit.register(_save_usage, profile, profile_file)

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
    else:
        # A learned profile beats the shipped prior once it has seen enough of
        # THIS user's traffic; below that it is noise (a few tokens would pin
        # whatever the greeting touched), so the shipped list still wins.
        if learn_experts and history.is_mature():
            spec = history.pin_spec()
            pin_layers, pin_order = _pin_spec_to_layers(spec)
            print(f"[stream] learned hot-expert list: {len(pin_order)} experts "
                  f"from {history.routings:,} recorded routings "
                  f"({profile_file})")
        elif use_hotlist:
            shipped = _find_hotlist(local_path)
            if shipped:
                try:
                    pin_layers, pin_order = _load_pin_spec(shipped)
                    print(f"[stream] found shipped hot-expert list: {shipped}")
                except ValueError as exc:  # includes json.JSONDecodeError
                    print(f"[stream] ignoring malformed shipped hotlist: {exc}")
        if learn_experts and not history.is_mature():
            print(f"[stream] learning expert usage -> {profile_file} "
                  f"({history.routings:,}/{_LEARN_MIN_ROUTINGS:,} routings; "
                  f"pins from this profile once mature)")

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
        sm_name, sm = _find_switch(getattr(layer, "mlp", None))
        if sm is None:
            continue
        proj_keys = []
        for proj in _PROJS:
            res = getattr(sm, proj, None)
            if not isinstance(res, PolarQuantizedSwitchLinear):
                continue
            cb, sg = res.codebook, res.signs
            mx.eval(cb, sg)  # tiny — pin resident, let the rest of res be freed
            wkey = f"{prefix}.{i}.mlp.{sm_name}.{proj}.weight"
            skey = f"{prefix}.{i}.mlp.{sm_name}.{proj}.scales"
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
                # One trigger per layer fires the same-layer miss fan-out and
                # the next-layer prefetch. It must be the FIRST projection the
                # MoE block EXECUTES — SwitchGLU runs up_proj, then gate_proj,
                # then down_proj — or every up_proj miss is read serially
                # before the fan-out even fires.
                is_trigger=(proj == "up_proj"),
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
