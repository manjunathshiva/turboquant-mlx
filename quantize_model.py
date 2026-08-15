"""Model-level TurboQuant quantization: traverse model and replace linear layers.

Handles the full pipeline:
1. Determine architecture and rotation fusion config
2. Apply Hadamard rotation to weights
3. Fuse rotations into normalization weights where possible
4. Replace nn.Linear layers with PolarQuantizedLinear
"""

import gc
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
    """Generate a deterministic seed for each layer based on its path."""
    return base_seed + hash(layer_path) % (2**31)


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
    """Check if a path corresponds to a MoE router layer (keep higher precision)."""
    last = path.split(".")[-1]
    return last in ("gate", "router", "shared_expert_gate")


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
    config["quantization"] = tq_config.to_dict()
    config["quantization"]["codebook"] = centroids.tolist()

    return model, config
