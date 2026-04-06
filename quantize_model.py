"""Model-level TurboQuant quantization: traverse model and replace linear layers.

Handles the full pipeline:
1. Determine architecture and rotation fusion config
2. Apply Hadamard rotation to weights
3. Fuse rotations into normalization weights where possible
4. Replace nn.Linear layers with PolarQuantizedLinear
"""

import gc
import math

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.core.rotation import (
    generate_random_signs,
    fuse_rotation_into_norm,
)
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from turboquant_mlx.integration.rotation_configs import (
    get_rotation_config,
    should_fuse_rotation,
    LayerRotationConfig,
)

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
    _, input_dims = module.weight.shape
    if input_dims < 32:
        return False
    return True


def _is_switch_linear(module: nn.Module) -> bool:
    """Check if a module is a SwitchLinear or QuantizedSwitchLinear (MoE expert weights)."""
    if not _HAS_SWITCH_LINEAR:
        return False
    return isinstance(module, (SwitchLinear, QuantizedSwitchLinear))


def _dequantize_switch_linear(module) -> mx.array:
    """Dequantize a QuantizedSwitchLinear back to float weights.

    Returns (num_experts, output_dims, input_dims) float16 tensor.
    """
    experts = []
    for e in range(module.num_experts):
        w_deq = mx.dequantize(
            module.weight[e],
            module.scales[e],
            module.biases[e] if module.biases is not None else None,
            module.group_size,
            module.bits,
            mode=module.mode,
        )
        experts.append(w_deq)
    return mx.stack(experts, axis=0)


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
) -> tuple[nn.Module, dict]:
    """Apply TurboQuant weight quantization to a model.

    Memory-efficient: replaces each layer immediately after quantization
    and releases references to original weights for garbage collection.
    """
    arch = _detect_architecture(config)
    rotation_config = None

    if tq_config.rotation != "none":
        try:
            rotation_config = get_rotation_config(arch)
        except ValueError:
            print(f"[WARNING] No rotation config for arch '{arch}', using online rotation for all layers")

    # Track which norms have been fused (to avoid double-fusing)
    fused_norms: set[str] = set()

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
                print(f"[INFO] Dequantizing QuantizedSwitchLinear {path} ({num_experts} experts, {module.mode} {module.bits}b -> float)")
                float_weight = _dequantize_switch_linear(module)
                mx.eval(float_weight)
            else:
                float_weight = module.weight
                input_dims = module.weight.shape[-1]
                num_experts = module.weight.shape[0]
                output_dims = module.weight.shape[1]
                has_bias = "bias" in module

            if input_dims % tq_config.group_size != 0:
                print(f"[WARNING] Skipping SwitchLinear {path}: input_dims={input_dims} not divisible by group_size={tq_config.group_size}")
                n_skipped += 1
                del module, float_weight
                continue

            # Determine rotation needs for expert layers
            needs_rotation = True
            if rotation_config is not None and tq_config.fuse_rotations:
                can_fuse, norm_name = should_fuse_rotation(path, rotation_config)
                if can_fuse and norm_name:
                    needs_rotation = False

            seed = _get_layer_seed(tq_config.rotation_seed, path)
            print(f"[INFO] Quantizing SwitchLinear {path} ({num_experts} experts, {input_dims}d)")

            bias_tensor = module.bias if has_bias else None
            pq_switch = PolarQuantizedSwitchLinear.from_switch_linear(
                None,
                bits=tq_config.bits,
                group_size=tq_config.group_size,
                seed=seed,
                needs_rotation=needs_rotation,
                float_weight=float_weight,
                bias=bias_tensor,
            )
            # Replace immediately and release all references
            mx.eval(pq_switch.parameters())
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

        # Check group_size compatibility
        _, input_dims = module.weight.shape
        if input_dims % tq_config.group_size != 0:
            print(f"[WARNING] Skipping {path}: input_dims={input_dims} not divisible by group_size={tq_config.group_size}")
            n_skipped += 1
            del module
            continue

        # Determine if rotation can be fused
        needs_rotation = True
        if rotation_config is not None and tq_config.fuse_rotations:
            can_fuse, norm_name = should_fuse_rotation(path, rotation_config)
            if can_fuse and norm_name:
                needs_rotation = False
                # Build the norm path relative to this layer's block
                parts = path.split(".")
                block_parts = []
                for p in parts:
                    if p in ("self_attn", "mlp", "attention", "feed_forward", "linear_attn"):
                        break
                    block_parts.append(p)
                norm_path = ".".join(block_parts + [norm_name])

                if norm_path not in fused_norms:
                    try:
                        norm_module = _get_nested_attr(model, norm_path)
                        if hasattr(norm_module, "weight"):
                            input_dims = module.weight.shape[-1]
                            signs = generate_random_signs(
                                input_dims,
                                seed=_get_layer_seed(tq_config.rotation_seed, path),
                            )
                            fused_weight = fuse_rotation_into_norm(
                                norm_module.weight.astype(mx.float32),
                                signs.astype(mx.float32),
                            ).astype(norm_module.weight.dtype)
                            norm_module.weight = fused_weight
                            fused_norms.add(norm_path)
                            del norm_module
                    except AttributeError:
                        pass

        # Quantize the linear layer
        seed = _get_layer_seed(tq_config.rotation_seed, path)
        pq_layer = PolarQuantizedLinear.from_linear(
            module,
            bits=tq_config.bits,
            group_size=tq_config.group_size,
            seed=seed,
            needs_rotation=needs_rotation,
            use_qjl=tq_config.use_qjl,
        )

        # Replace immediately to free original weights
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
