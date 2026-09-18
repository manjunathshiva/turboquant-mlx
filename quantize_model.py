"""Model-level TurboQuant quantization: traverse model and replace linear layers.

Handles the full pipeline:
1. Determine architecture and rotation fusion config
2. Apply Hadamard rotation to weights
3. Fuse rotations into normalization weights where possible
4. Replace nn.Linear layers with PolarQuantizedLinear
"""

import gc
import zlib
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

# Try importing SwitchLinear for MoE detection
try:
    from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
    _HAS_SWITCH_LINEAR = True
except ImportError:
    _HAS_SWITCH_LINEAR = False


def _detect_architecture(config: dict) -> str:
    """Detect model architecture from config dict."""
    model_type = config.get("model_type", "")
    if not model_type:
        # Try text_config for multimodal models
        text_config = config.get("text_config", {})
        model_type = text_config.get("model_type", "")
    return model_type.lower()


def _get_layer_seed(base_seed: int, layer_path: str) -> int:
    """A per-layer rotation seed, the same in every process.

    This used ``hash(layer_path)``, which Python randomizes per process, so
    converting the same model twice gave different rotations and different
    weights. Loading was never affected (each build stores its signs), but builds
    weren't reproducible, and two builds compared in an experiment differed on
    layers that should have been identical. CRC-32 of the path is stable.
    """
    return base_seed + zlib.crc32(layer_path.encode()) % (2**31)


def _should_quantize(path: str, module: nn.Module) -> bool:
    """Determine if a module should be quantized."""
    if isinstance(module, nn.Embedding):
        return False
    if not isinstance(module, nn.Linear):
        return False
    output_dims, input_dims = module.weight.shape
    if input_dims < 32:
        return False
    if output_dims < 32:
        # Scalar/score projections (e.g. Kimi K3's AttnRes *_res_proj rows,
        # shape (1, hidden)) — quantization noise on a softmax score vector
        # is all pain for ~0 bytes saved.
        return False
    return True


def quantize_affine_extras(model, config, bits: int, group_size: int,
                           extra_exclude=None, on_quantized=None) -> int:
    """Quantize the bf16 remainder that the polar path does not touch.

    ``_should_quantize`` handles ``nn.Linear`` / ``SwitchLinear`` only, and skips
    ``nn.Embedding`` outright. On most models that leftover is a rounding error
    — a token embedding and an ``lm_head``, which is why this tier was named
    "extras". On Qwen3.8-Flash-Next it is **51.2B params**: the sharded
    n-gram/PLE table is 128 ``nn.Embedding`` shards, 28% of the model and
    95.4 GiB in bf16. Left alone it puts a ternary-expert build at ~124 GiB and
    off a 64 GB Mac entirely; at 4-bit affine the same build lands near 54 GiB.

    What must NOT be swept up: the modules ``_should_quantize`` skipped *on
    purpose*. The MoE router and Qwen Sparse Attention's block indexer both make
    a discrete choice — which expert, which KV block — so error there changes
    which computation runs rather than degrading it smoothly. ``_is_router``
    already names them, and this pass reuses it so the two cannot drift apart.
    Modules the polar pass DID claim are inert here: they are
    ``PolarQuantized*`` by now and carry no ``to_quantized``.

    The matching read path is ``generate._prepare_affine_extras``, which keys off
    ``.scales`` present with no ``.codebook`` and recovers bits/group size from
    the tensor shapes — so this writes no per-tensor metadata beyond the
    ``affine_extras`` config block.

    Args:
        extra_exclude: optional ``(path) -> bool``; return True to keep a module
            at full precision on top of the ``_is_router`` exclusions.
        on_quantized: optional ``(path, module)`` sink. When given, each module
            is quantized, handed over, then replaced by ``nn.Identity`` so its
            weight can be freed -- the streaming converter's contract. Without
            it the whole tier is quantized in one resident pass.

    Returns:
        The number of modules quantized.
    """
    def _eligible(path, module):
        if not hasattr(module, "to_quantized"):
            return False                       # already polar-quantized
        if _is_router(path):
            return False                       # discrete selection: keep exact
        if isinstance(module, nn.Linear) and not _should_quantize(path, module):
            # The polar path rejects some linears on purpose -- scalar/score
            # projections narrower than 32 (Kimi K3's AttnRes *_res_proj, shape
            # (1, hidden)), where quantization noise costs quality for ~0 bytes.
            # Without this, --quantize-extras quietly re-quantizes exactly those.
            # Embeddings are NOT covered: _should_quantize rejects every
            # nn.Embedding, and catching them is this tier's whole purpose.
            return False
        if extra_exclude is not None and extra_exclude(path):
            return False
        w = getattr(module, "weight", None)
        if w is not None and w.shape[-1] % group_size != 0:
            # Record it: skipping here is silent, and on a big lookup table that
            # silence is the difference between a 54 GiB build and a 124 GiB one.
            # Qwen3.8-Flash-Next's n-gram shards are 160 wide -- fine at g32,
            # excluded at the g64 default -- so this must never pass unremarked.
            # Keyed by path: nn.quantize re-runs the predicate over the tree, so
            # a list would double-count every skipped module.
            skipped[path] = (w.shape[-1], w.size * w.dtype.size)
            return False
        return True

    skipped = {}
    count = 0
    if on_quantized is None:
        # Resident path: let MLX walk the tree in one pass.
        def _predicate(path, module):
            return _eligible(path, module)

        targets = [p for p, m in model.named_modules() if _eligible(p, m)]
        count = len(targets)
        nn.quantize(model, group_size=group_size, bits=bits,
                    class_predicate=_predicate)
        mx.eval(model.parameters())
    else:
        # Streaming path: quantize ONE module, hand it to the writer, drop it.
        # A bulk nn.quantize would materialize every eligible weight at once,
        # and on Qwen3.8-Flash-Next that is the 95.4 GiB n-gram table -- which
        # is the whole reason this tier exists. Per-module, the peak is one
        # shard (0.75 GiB bf16 -> 0.19 GiB at 4-bit).
        #
        # Iterate over PATHS and fetch each module fresh. Iterating
        # `list(model.named_modules())` held a reference to every ORIGINAL
        # module for the whole loop, so each source weight -- materialized by
        # to_quantized -- stayed alive after its module was swapped out. On
        # Qwen3.8-Flash-Next that retained the full 95.4 GiB bf16 n-gram table
        # on a 64 GB machine: three conversions died 71-90% through this phase.
        # Pinned by tests/test_streaming_extras_memory.py.
        paths = [p for p, m in model.named_modules() if _eligible(p, m)]
        for path in paths:
            module = _get_nested_attr(model, path)
            q = module.to_quantized(group_size=group_size, bits=bits)
            mx.eval(q.parameters())
            on_quantized(path, q)
            _set_nested_attr(model, path, nn.Identity())
            del module, q
            gc.collect()
            mx.clear_cache()  # hand freed buffers back to the OS, not MLX's cache
            count += 1

    if skipped:
        tot = sum(b for _, b in skipped.values())
        print(f"[WARNING] {len(skipped)} module(s) left UNQUANTIZED at their "
              f"source dtype because their width is not a multiple of "
              f"group_size={group_size}: {tot / 1024**3:.2f} GiB.")
        widths = sorted({w for w, _ in skipped.values()})
        for cand in (128, 64, 32, 16):
            if cand < group_size and all(w % cand == 0 for w in widths):
                print(f"[WARNING]   every skipped width {widths} divides by "
                      f"{cand} -- re-run with --extras-group-size {cand} to "
                      f"include them.")
                break
        else:
            print(f"[WARNING]   skipped widths: {widths}")
        worst = sorted(skipped.items(), key=lambda kv: -kv[1][1])[:3]
        for path, (w, b) in worst:
            print(f"[WARNING]   {path} (width {w}, {b / 1024**3:.2f} GiB)")

    config.setdefault("quantization", {})["affine_extras"] = {
        "bits": bits, "group_size": group_size,
    }
    return count


def _is_switch_linear(module: nn.Module) -> bool:
    """Check if a module is a SwitchLinear or QuantizedSwitchLinear (MoE expert weights)."""
    if not _HAS_SWITCH_LINEAR:
        return False
    return isinstance(module, (SwitchLinear, QuantizedSwitchLinear))


def _dequantize_switch_expert(module, e: int) -> mx.array:
    """Dequantize a single expert of a QuantizedSwitchLinear to float."""
    return mx.dequantize(
        module.weight[e],
        module.scales[e],
        module.biases[e] if module.biases is not None else None,
        module.group_size,
        module.bits,
        mode=module.mode,
    )


def _dequantize_switch_linear(module) -> mx.array:
    """Dequantize a QuantizedSwitchLinear back to float weights.

    Returns (num_experts, output_dims, input_dims) float16 tensor.
    NOTE: materializes ALL experts — prefer the per-expert path
    (``partial(_dequantize_switch_expert, module)``) for large MoEs.
    """
    return mx.stack(
        [_dequantize_switch_expert(module, e) for e in range(module.num_experts)],
        axis=0,
    )


def _is_router(path: str) -> bool:
    """Check for a discrete-selection layer, which stays at full precision.

    Two kinds qualify, and they share one property: each makes a *discrete*
    choice rather than a smooth projection, so weight error changes **which**
    branch is taken instead of nudging the output. A router that picks the
    wrong expert, or an indexer that picks the wrong KV block, is not a small
    error — it is a different computation.

    - **MoE routers** (``gate`` / ``router`` / ``shared_expert_gate``), which
      select the experts.
    - **Qwen Sparse Attention's block indexer** (``index_qk_proj``, living
      under ``.indexer.``), which top-k selects the compressed KV blocks a
      query may attend to.

    Both are negligible in size: the entire QSA indexer across all 12
    full-attention layers of Qwen3.8-Flash-Next is 19.7M params (0.04 GiB), so
    protecting it costs nothing measurable against a ~50 GiB build.
    """
    parts = path.split(".")
    last = parts[-1]
    if last in ("gate", "router", "shared_expert_gate"):
        return True
    return "indexer" in parts or last == "index_qk_proj"


def _get_nested_attr(model: nn.Module, path: str):
    """Get a nested attribute from a model given a dot-separated path."""
    obj = model
    for p in path.split("."):
        if hasattr(obj, p):
            obj = getattr(obj, p)
        elif p.isdigit():
            obj = obj[int(p)]
        else:
            raise AttributeError(f"Cannot resolve path component '{p}' in '{path}'")
    return obj


def _set_nested_attr(model: nn.Module, path: str, value):
    """Set a nested attribute on a model given a dot-separated path."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if hasattr(parent, p):
            parent = getattr(parent, p)
        elif p.isdigit():
            parent = parent[int(p)]
        else:
            raise AttributeError(f"Cannot resolve path component '{p}' in '{path}'")
    setattr(parent, parts[-1], value)


def turboquant_quantize(
    model: nn.Module,
    config: dict,
    tq_config: TurboQuantConfig,
    on_quantized=None,
) -> tuple[nn.Module, dict]:
    """Apply TurboQuant weight quantization to a model.

    Memory-efficient: replaces each layer immediately after quantization
    and releases references to original weights for garbage collection.

    If ``on_quantized`` is given, it is called as ``on_quantized(path, module)``
    right after each layer is quantized and evaluated, and that layer is then
    replaced on the model with a paramless stub instead of the quantized module.
    This lets a streaming converter write each layer to disk and free it, so the
    full quantized model never has to reside in memory at once. Non-quantized
    params (norms, embeddings, routers) stay on the model for the caller to write
    afterward.
    """
    arch = _detect_architecture(config)

    # Snapshot paths and module types ONLY — don't hold module references
    module_paths = []
    module_types = {}  # path -> "switch" | "switch_quantized" | "linear" | "skip"
    for path, module in model.named_modules():
        if _is_switch_linear(module):
            is_preq = _HAS_SWITCH_LINEAR and isinstance(module, QuantizedSwitchLinear)
            module_types[path] = "switch_quantized" if is_preq else "switch"
            module_paths.append(path)
        elif isinstance(module, nn.Linear):
            module_types[path] = "linear"
            module_paths.append(path)
        # Note: we don't store references to modules, just paths

    n_quantized = 0
    n_skipped = 0
    # Layer protection (see TurboQuantConfig.protect_expert_layers). Matched by
    # the `.layers.N.` index inside the routed-expert branch, which is already
    # selected by module type -- no expert-container name matching.
    import re as _re
    _layer_idx_rx = _re.compile(r"(?:^|\.)layers\.(\d+)\.")
    protected_layers = set(tq_config.protect_expert_layers or ())
    matched_protected_layers = set()
    matched_tier_layers = set()
    n_switch = 0

    for path in module_paths:
        mtype = module_types[path]

        # --- Handle MoE SwitchLinear / QuantizedSwitchLinear layers ---
        if mtype in ("switch", "switch_quantized"):
            # Look up module fresh from model (not from a cached dict)
            module = _get_nested_attr(model, path)

            if mtype == "switch_quantized":
                input_dims = module.scales.shape[-1] * module.group_size
                num_experts = module.num_experts
                output_dims = module.output_dims
                has_bias = "bias" in module
                print(f"[INFO] Dequantizing QuantizedSwitchLinear {path} ({num_experts} experts, {module.mode} {module.bits}b -> float, per-expert)")
                # Lazy per-expert dequant: materializing the whole stacked
                # float tensor costs ~30 GB on an 896-expert layer; one
                # expert at a time stays flat (~0.4 GB).
                float_weight = partial(_dequantize_switch_expert, module)
            else:
                float_weight = module.weight
                input_dims = module.weight.shape[-1]
                num_experts = module.weight.shape[0]
                output_dims = module.weight.shape[1]
                has_bias = "bias" in module

            expert_group_size = tq_config.group_size_for_path(path)
            if input_dims % expert_group_size != 0:
                print(f"[WARNING] Skipping SwitchLinear {path}: input_dims={input_dims} not divisible by group_size={expert_group_size}")
                n_skipped += 1
                del module, float_weight
                continue

            # Rotation is all-or-nothing per model, driven by the config.
            # It can never be "fused into the preceding norm" — a Hadamard
            # does not commute with a diagonal (see rotation.py and
            # test_rotation_cannot_fuse_into_norm). The same flag also
            # decides whether the *weights* get rotated, so the two halves
            # cannot drift apart.
            needs_rotation = tq_config.rotation != "none"

            seed = _get_layer_seed(tq_config.rotation_seed, path)
            use_ternary = tq_config.ternary_experts
            # Ternary experts pack as base-3 trits (3-entry codebook, 20/uint32,
            # ~1.6 bpw); bits=2 is storage/scale semantics only.
            layer_bits = 2 if use_ternary else tq_config.bits_for_path(path)
            if (tq_config.expert_down_bits is not None
                    and path.split(".")[-1] == "down_proj"):
                # Asymmetric expert precision: the down projection carries a
                # higher-precision Gaussian codebook than the up/gate tier.
                # The loader needs no matching rule — per-layer bits are
                # self-describing via the on-disk codebook length.
                use_ternary = False
                layer_bits = tq_config.expert_down_bits
            if tq_config.expert_layer_tiers:
                m = _layer_idx_rx.search(path)
                tier = tq_config.expert_tier_for_layer(int(m.group(1))) if m else None
                if tier is not None:
                    matched_tier_layers.add(int(m.group(1)))
                    layer_bits, use_ternary = tier
            if protected_layers:
                m = _layer_idx_rx.search(path)
                if m and int(m.group(1)) in protected_layers:
                    # Protected layer: leave the ternary / mlp_bits tier for a
                    # protect_bits Gaussian codebook. If --expert-down-bits also
                    # applies to this down projection, keep the higher width.
                    matched_protected_layers.add(int(m.group(1)))
                    down_bits = (tq_config.expert_down_bits
                                 if path.split(".")[-1] == "down_proj" else None)
                    use_ternary = False
                    layer_bits = max(tq_config.protect_bits, down_bits or 0)
            label = "ternary" if use_ternary else f"{layer_bits}b"
            print(f"[INFO] Quantizing SwitchLinear {path} ({num_experts} experts, {input_dims}d, {label} g{expert_group_size})")

            bias_tensor = module.bias if has_bias else None
            pq_switch = PolarQuantizedSwitchLinear.from_switch_linear(
                None,
                bits=layer_bits,
                group_size=expert_group_size,
                seed=seed,
                needs_rotation=needs_rotation,
                float_weight=float_weight,
                bias=bias_tensor,
                ternary=use_ternary,
                weight_shape=(num_experts, output_dims, input_dims),
            )
            # Replace immediately and release all references
            mx.eval(pq_switch.parameters())
            if on_quantized is not None:
                # Streaming convert: write this layer to disk, then drop its
                # params so the full quantized model never resides in memory.
                on_quantized(path, pq_switch)
                _set_nested_attr(model, path, nn.Identity())
            else:
                _set_nested_attr(model, path, pq_switch)
            del float_weight, module, bias_tensor, pq_switch
            gc.collect()
            # Return freed buffers to the OS between expert layers. Without it
            # MLX keeps them in its own cache (limit defaults to the whole
            # Metal working set), which the OS still counts as in use.
            mx.clear_cache()
            n_switch += 1
            n_quantized += 1
            continue

        # --- Handle standard nn.Linear layers ---
        module = _get_nested_attr(model, path)

        if not _should_quantize(path, module):
            del module
            continue

        # Skip MoE router layers (keep higher precision)
        if _is_router(path):
            print(f"[INFO] Skipping router {path} (keeping full precision)")
            del module
            continue

        # Check group_size compatibility against the group size this layer
        # will ACTUALLY be quantized at — --mlp-group-size can differ from the
        # base, and validating the base would let a layer past the check and
        # then fail inside polar_quantize_weight.
        _, input_dims = module.weight.shape
        layer_group_size = tq_config.group_size_for_path(path)
        if input_dims % layer_group_size != 0:
            print(f"[WARNING] Skipping {path}: input_dims={input_dims} not divisible by group_size={layer_group_size}")
            n_skipped += 1
            del module
            continue

        # Rotation is all-or-nothing per model, driven by the config. It can
        # never be folded into the preceding norm: the norm applies a diagonal
        # weight and a Hadamard does not commute with a diagonal (see
        # test_rotation_cannot_fuse_into_norm). The same flag also decides
        # whether the *weights* get rotated, so they cannot drift apart.
        needs_rotation = tq_config.rotation != "none"

        # Quantize the linear layer
        seed = _get_layer_seed(tq_config.rotation_seed, path)
        layer_bits = tq_config.bits_for_path(path)
        pq_layer = PolarQuantizedLinear.from_linear(
            module,
            bits=layer_bits,
            # Per-path, not the base group_size: --mlp-group-size must reach
            # dense MLP linears exactly as --mlp-bits already does. (The loader
            # recovers the real value from the saved scales, so this rule
            # changing cannot desync convert from load.)
            group_size=layer_group_size,
            seed=seed,
            needs_rotation=needs_rotation,
            use_qjl=tq_config.use_qjl,
        )

        # Replace immediately to free original weights
        if on_quantized is not None:
            mx.eval(pq_layer.parameters())
            on_quantized(path, pq_layer)
            _set_nested_attr(model, path, nn.Identity())
        else:
            _set_nested_attr(model, path, pq_layer)
        del module, pq_layer
        n_quantized += 1

    if n_skipped > 0:
        print(f"[INFO] Skipped {n_skipped} layers due to dimension incompatibility")
    if n_switch > 0:
        print(f"[INFO] Quantized {n_switch} SwitchLinear (MoE expert) layers")
    print(f"[INFO] Quantized {n_quantized - n_switch} Linear layers + {n_switch} SwitchLinear layers")

    # Update config — remove any pre-existing quantization keys to avoid
    # mlx_lm trying to re-quantize on load
    from turboquant_mlx.core.codebook import get_codebook
    centroids, _ = get_codebook(tq_config.bits)
    config.pop("quantization_config", None)
    if tq_config.expert_layer_tiers:
        missing = sorted(set(tq_config.expert_layer_tiers) - matched_tier_layers)
        if missing:
            # Never silent, as with protection: a tier map that matches nothing
            # produces a build whose config claims an allocation it doesn't have.
            raise ValueError(f"expert_layer_tiers names layer(s) {missing} that have "
                             "no routed experts in this model")
        counts = {}
        for t in tq_config.expert_layer_tiers.values():
            counts[t] = counts.get(t, 0) + 1
        print(f"[INFO] Expert tiers per layer: {counts}")
    if protected_layers:
        missing = sorted(protected_layers - matched_protected_layers)
        if missing:
            # Never silent: a protection list that matches nothing produces a
            # build that looks protected in config.json and is not.
            print(f"[WARNING] Layer protection requested for layer(s) {missing}, "
                  f"but no routed-expert layer at those indices was quantized -- "
                  f"they are NOT protected. Check the model's layer count.")
        print(f"[INFO] Expert layer protection: layers "
              f"{sorted(matched_protected_layers)} -> "
              f"{tq_config.protect_bits}b codebook")
    config["quantization"] = tq_config.to_dict()
    config["quantization"]["codebook"] = centroids.tolist()

    return model, config
