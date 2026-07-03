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
import json
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

        # Authoritative per-layer bit width: the saved codebook has 2^bits
        # entries. This survives any per-layer bit assignment the converter
        # used (hybrids, layer protection) without re-deriving path rules.
        # A 3-entry codebook is the self-describing marker for ternary experts
        # packed as base-3 trits (20/uint32, ~1.6 bpw): decode base-3 inline.
        cb_size = int(weights[f"{path}.codebook"].shape[-1])
        is_trit = cb_size == 3
        if is_trit:
            layer_bits = 2  # storage semantics only; kernels decode base-3
        else:
            layer_bits = max(1, cb_size.bit_length() - 1)
            if (1 << layer_bits) != cb_size:  # non-power-of-two: fall back
                layer_bits = tq_config.bits_for_path(path)

        if has_switch and isinstance(module, SwitchLinear):
            num_experts, output_dims, input_dims = module.weight.shape
            pq = PolarQuantizedSwitchLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                num_experts=num_experts,
                bias=has_bias,
                bits=layer_bits,
                # Match the convert side, which quantizes experts at
                # group_size_for_path — so a model built with a finer
                # --mlp-group-size loads with the correct scale shape.
                group_size=tq_config.group_size_for_path(path),
                trit=is_trit,
            )
            updates[path] = pq
        elif isinstance(module, nn.Linear):
            output_dims, input_dims = module.weight.shape
            pq = PolarQuantizedLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                bias=has_bias,
                bits=layer_bits,
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
        "--top-p", type=float, default=None,
        help="Nucleus sampling threshold. Defaults to the model's "
             "generation_config.json value when present, else disabled. "
             "Pass 0 to force-disable.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k truncation. Defaults to the model's "
             "generation_config.json value when present, else disabled. "
             "Pass 0 to force-disable. Without truncation, rare tokens like "
             "a stray '</think>' can be sampled in place of EOS.",
    )
    parser.add_argument(
        "--rep-penalty", type=float, default=None,
        help="Repetition penalty (e.g. 1.05). Defaults to the model's "
             "generation_config.json value when present, else disabled. "
             "Pass 0 or 1 to force-disable. Recommended for thinking-mode "
             "models on low-bit builds (breaks think-block loops); omit for "
             "numeric/math prompts.",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="Disable thinking mode via the chat template "
             "(enable_thinking=False). Much faster and immune to "
             "think-block loops on thinking-capable models.",
    )
    parser.add_argument(
        "--multi-think", action="store_true",
        help="Allow more than one </think> token per generation. By default "
             "a second </think> is masked once one has been emitted, which "
             "prevents low-bit thinking models from re-emitting their answer.",
    )
    parser.add_argument(
        "--rep-ctx", type=int, default=256,
        help="Repetition penalty context window in tokens (default: 256). "
             "Only used when --rep-penalty is set.",
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
    # ── TurboQuant KV-cache flags (v0.2) ──────────────────────────────
    parser.add_argument(
        "--kv-bits", type=int, default=None,
        help="Symmetric KV-cache bits (K=V). Use --kv-k-bits / --kv-v-bits "
             "for mixed precision instead. Omit to keep fp16 cache.",
    )
    parser.add_argument(
        "--kv-k-bits", type=int, default=None,
        help="Bits for K cache (recommended: 8). Pair with --kv-v-bits.",
    )
    parser.add_argument(
        "--kv-v-bits", type=int, default=None,
        help="Bits for V cache (recommended: 3). Pair with --kv-k-bits. "
             "Mixed K8/V3 is the v0.2 recommended default.",
    )
    parser.add_argument(
        "--kv-min-tokens", type=int, default=0,
        help="Keep the first N cached tokens in fp16 (attention-sink "
             "protection). 128 is a reasonable default for long-context "
             "generation. Default: 0 (quantize from token 0).",
    )
    parser.add_argument(
        "--kv-group-size", type=int, default=64,
        help="Hadamard rotation group size for KV quantization (default: 64).",
    )
    args = parser.parse_args()

    # Validate KV cache flags
    if args.kv_bits is not None and (args.kv_k_bits is not None or args.kv_v_bits is not None):
        parser.error("--kv-bits is mutually exclusive with --kv-k-bits/--kv-v-bits")
    if (args.kv_k_bits is None) != (args.kv_v_bits is None):
        parser.error("--kv-k-bits and --kv-v-bits must be set together")

    mode = "fast (QJL disabled)" if args.fast else "accurate (QJL enabled)"
    print(f"[INFO] Loading TurboQuant model from {args.model} [{mode}]")
    model, tokenizer = load_turboquant(args.model, fast=args.fast)

    # Apply chat template if available
    prompt = args.prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            template_kwargs = {"enable_thinking": False} if args.no_think else {}
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            print("[INFO] Applied chat template"
                  + (" (thinking disabled)" if args.no_think else ""))
        except Exception:
            pass  # Fall back to raw prompt

    # Use mlx_lm's generate function with a sampler for temperature
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from turboquant_mlx.sampling import (
        eos_token_ids,
        make_min_tokens_logits_processor,
        make_single_think_close_logits_processor,
        think_close_token_id,
    )

    # Sampling defaults come from the model's own generation_config.json
    # (e.g. Qwen ships top_k=20/top_p=0.95; loop-prone low-bit builds can ship
    # repetition_penalty). Untruncated temperature sampling over a 250K vocab
    # occasionally picks a stray special token — seen as a doubled answer when
    # '</think>' is sampled where '<|im_end|>' belongs.
    top_p, top_k, rep_penalty = args.top_p, args.top_k, args.rep_penalty
    if top_p is None or top_k is None or rep_penalty is None:
        # load_turboquant already resolved/downloaded the model, so this is a
        # cache hit; read the json directly rather than pulling in transformers
        # (only an optional [eval] dependency).
        try:
            gen_cfg_file = resolve_model_path(args.model) / "generation_config.json"
            if gen_cfg_file.exists():
                with open(gen_cfg_file, encoding="utf-8") as f:
                    gen_cfg = json.load(f)
                if top_p is None:
                    top_p = gen_cfg.get("top_p")
                if top_k is None:
                    top_k = gen_cfg.get("top_k")
                if rep_penalty is None:
                    cfg_rep = gen_cfg.get("repetition_penalty")
                    # 1.0 is transformers' neutral default — not a real request
                    if cfg_rep and cfg_rep != 1.0:
                        rep_penalty = cfg_rep
                        print(f"[INFO] repetition_penalty {cfg_rep} "
                              f"(from generation_config.json)")
        except Exception as e:
            print(
                f"[INFO] Could not read generation_config.json for "
                f"{args.model}; sampling without truncation defaults ({e})"
            )
    # 0 or 1.0 on the CLI force-disables (1.0 is mathematically neutral)
    if rep_penalty is not None and rep_penalty in (0.0, 1.0):
        rep_penalty = None
    sampler = make_sampler(
        temp=args.temp,
        top_p=top_p if top_p is not None else 0.0,
        top_k=top_k if top_k is not None else 0,
    )
    min_tokens_proc = make_min_tokens_logits_processor(
        args.min_tokens, eos_token_ids(tokenizer)
    )
    logits_processors = []
    if rep_penalty is not None:
        logits_processors.extend(make_logits_processors(
            repetition_penalty=rep_penalty,
            repetition_context_size=args.rep_ctx,
        ))
    if not args.multi_think:
        think_guard = make_single_think_close_logits_processor(
            think_close_token_id(tokenizer)
        )
        if think_guard is not None:
            logits_processors.append(think_guard)
    if min_tokens_proc is not None:
        logits_processors.append(min_tokens_proc)
    if not logits_processors:
        logits_processors = None

    # Build prompt cache. Always start from make_prompt_cache so hybrid
    # models (Nemotron-H, Qwen3.5) get the right per-layer cache type;
    # then convert only the standard KVCache entries to TurboQuant.
    from mlx_lm.models.cache import make_prompt_cache
    prompt_cache = make_prompt_cache(model)

    if args.kv_bits is not None or args.kv_k_bits is not None:
        from turboquant_mlx.layers.polar_kv_cache import (
            convert_cache_to_turboquant,
        )
        prompt_cache = convert_cache_to_turboquant(
            prompt_cache,
            tq_bits=args.kv_bits,
            k_bits=args.kv_k_bits,
            v_bits=args.kv_v_bits,
            group_size=args.kv_group_size,
            min_tokens_before_quant=args.kv_min_tokens,
        )
        if args.kv_bits is not None:
            print(f"[INFO] TurboQuant KV cache: K=V={args.kv_bits}-bit, "
                  f"group={args.kv_group_size}, sink={args.kv_min_tokens}")
        else:
            print(f"[INFO] TurboQuant KV cache: K={args.kv_k_bits}-bit, "
                  f"V={args.kv_v_bits}-bit, group={args.kv_group_size}, "
                  f"sink={args.kv_min_tokens}")

    print(f"\nPrompt: {args.prompt}\n")
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=prompt_cache,
        verbose=True,
    )


if __name__ == "__main__":
    main()
