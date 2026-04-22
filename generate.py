"""Generate text using a TurboQuant-compressed model.

Usage:
    # Accurate mode (default) — uses QJL correction if available
    python -m turboquant_mlx.generate \
        --model ./gpt-oss-20b-tq2-qjl \
        --prompt "Why is the sky blue?" \
        --max-tokens 200

    # Fast mode — skips QJL correction for ~25% faster decode
    python -m turboquant_mlx.generate \
        --model ./gpt-oss-20b-tq2-qjl \
        --prompt "Why is the sky blue?" \
        --max-tokens 200 --fast
"""

import argparse
import glob
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear


def _set_nested_attr(model, path, value):
    """Set a nested attribute on a model given a dot-separated path."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if hasattr(parent, p):
            parent = getattr(parent, p)
        elif p.isdigit():
            parent = parent[int(p)]
        else:
            raise AttributeError(f"Cannot resolve '{p}' in '{path}'")
    setattr(parent, parts[-1], value)


def _prepare_polar_layers(model, weights, tq_config):
    """Replace nn.Linear and SwitchLinear with PolarQuantized versions.

    Detects which layers were quantized by checking for '.codebook' keys
    in the saved weights, then creates matching PolarQuantized layers.
    """
    try:
        from mlx_lm.models.switch_layers import SwitchLinear
        has_switch = True
    except ImportError:
        has_switch = False

    # Find all layers that have codebook in saved weights
    quantized_paths = set()
    for key in weights:
        if key.endswith(".codebook"):
            quantized_paths.add(key.rsplit(".codebook", 1)[0])

    updates = {}
    for path, module in model.named_modules():
        if path not in quantized_paths:
            continue

        has_bias = f"{path}.bias" in weights
        has_qjl = f"{path}.qjl_packed" in weights

        if has_switch and isinstance(module, SwitchLinear):
            num_experts, output_dims, input_dims = module.weight.shape
            pq = PolarQuantizedSwitchLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                num_experts=num_experts,
                bias=has_bias,
                bits=tq_config.bits,
                group_size=tq_config.group_size,
            )
            updates[path] = pq
        elif isinstance(module, nn.Linear):
            output_dims, input_dims = module.weight.shape
            pq = PolarQuantizedLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                bias=has_bias,
                bits=tq_config.bits,
                group_size=tq_config.group_size,
                use_qjl=has_qjl,
            )
            updates[path] = pq

    for path, new_module in updates.items():
        _set_nested_attr(model, path, new_module)

    print(f"[INFO] Replaced {len(updates)} layers with PolarQuantized versions")
    return model


def _disable_qjl(model):
    """Disable QJL correction on all PolarQuantizedLinear layers for fast mode."""
    count = 0
    for _, module in model.named_modules():
        if isinstance(module, PolarQuantizedLinear) and module._use_qjl:
            module._use_qjl = False
            count += 1
    if count > 0:
        print(f"[INFO] Fast mode: disabled QJL on {count} layers")


def resolve_model_path(path_or_hf_repo):
    """Return a local Path for a local directory or an HF repo ID.

    Wraps mlx-lm's internal ``_download`` so that any TurboQuant entry point
    accepts both ``./my-model`` and ``user/repo`` arguments uniformly.
    """
    from mlx_lm.utils import _download
    return Path(_download(str(path_or_hf_repo)))


def load_turboquant(model_path, lazy=False, fast=False):
    """Load a TurboQuant-compressed model.

    Args:
        model_path: Local directory path or HuggingFace repo ID. HF repos
            are downloaded on first use.
        lazy: If True, don't evaluate parameters immediately.
        fast: If True, disable QJL correction for faster inference.

    Returns:
        (model, tokenizer) tuple.
    """
    from mlx_lm.utils import load_config, load_tokenizer, _get_classes

    model_path = resolve_model_path(model_path)
    config = load_config(model_path)

    # Extract TurboQuant config
    tq_dict = config.get("quantization", {})
    tq_config = TurboQuantConfig.from_dict(tq_dict)

    # Remove quantization keys so mlx_lm doesn't try to re-quantize
    config_for_model = dict(config)
    config_for_model.pop("quantization", None)
    config_for_model.pop("quantization_config", None)

    # Load all weights from safetensors
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Create base model (with regular nn.Linear / SwitchLinear layers)
    model_class, model_args_class = _get_classes(config=config_for_model)
    model_args = model_args_class.from_dict(config_for_model)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Replace quantized layers with PolarQuantized versions
    _prepare_polar_layers(model, weights, tq_config)

    # Load weights into the model
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Fast mode: disable QJL correction for speed
    if fast:
        _disable_qjl(model)

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a TurboQuant-compressed model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to TurboQuant model directory",
    )
    parser.add_argument(
        "--prompt", type=str, default="Why is the sky blue?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: skip QJL correction for ~25%% faster decode (slightly lower quality)",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=0,
        help="Mask EOS until at least this many tokens are generated. "
             "Useful for thinking-mode models (Nemotron 3, etc.) whose chat "
             "template primes EOS as the top-1 logit (default: 0)",
    )
    args = parser.parse_args()

    mode = "fast (QJL disabled)" if args.fast else "accurate (QJL enabled)"
    print(f"[INFO] Loading TurboQuant model from {args.model} [{mode}]")
    model, tokenizer = load_turboquant(args.model, fast=args.fast)

    # Apply chat template if available
    prompt = args.prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            print("[INFO] Applied chat template")
        except Exception:
            pass  # Fall back to raw prompt

    # Use mlx_lm's generate function with a sampler for temperature
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    from turboquant_mlx.sampling import (
        eos_token_ids,
        make_min_tokens_logits_processor,
    )

    sampler = make_sampler(temp=args.temp)
    min_tokens_proc = make_min_tokens_logits_processor(
        args.min_tokens, eos_token_ids(tokenizer)
    )
    logits_processors = [min_tokens_proc] if min_tokens_proc is not None else None

    print(f"\nPrompt: {args.prompt}\n")
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True,
    )


if __name__ == "__main__":
    main()
