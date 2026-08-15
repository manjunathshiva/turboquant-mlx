"""Side-by-side generation demo for v0.2 KV-cache features.

Runs the same prompt through several cache configurations on one model
and prints all outputs so you can eyeball quality differences:

    fp16              full-precision baseline
    K8_V3             mixed K/V bits (item #1)         — recommended default
    K8_V3_thr128      K8+V3 with first 128 tokens fp16 (item #2, sink-protected)
    K3_V3             symmetric 3-bit (legacy v0.1.x equivalent)

Usage:
    python -m turboquant_mlx.benchmarks.demo_kv_v02 \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
        --prompt "Why is Sky blue?" \\
        --max-tokens 200
"""

import argparse
import time

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.models.cache import make_prompt_cache

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_kv_cache import convert_cache_to_turboquant


def _is_turboquant_model(path) -> bool:
    """A TurboQuant checkpoint has quantization.mode == 'turboquant' in config.json."""
    try:
        local = resolve_model_path(path)
    except Exception:
        return False
    cfg = local / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(Path(cfg).read_text())
    except Exception:
        return False
    quant = data.get("quantization") or data.get("quantization_config") or {}
    return isinstance(quant, dict) and quant.get("mode") == "turboquant"


CONFIGS = [
    ("fp16",         None),
    ("K8_V3",        {"k_bits": 8, "v_bits": 3, "min_tokens_before_quant": 0}),
    ("K8_V3_thr128", {"k_bits": 8, "v_bits": 3, "min_tokens_before_quant": 128}),
    ("K3_V3",        {"k_bits": 3, "v_bits": 3, "min_tokens_before_quant": 0}),
]


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path (mlx-community/...)")
    ap.add_argument("--prompt", default="Why is Sky blue?")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true",
                    help="For TurboQuant models: disable QJL correction")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Suppress EOS for the first N tokens (Nemotron needs ~50)")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="Sampling temperature; >0 enables stochastic sampling")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="Nucleus sampling top-p")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Penalty for repeated tokens; >1 discourages repetition "
                         "(Nemotron long-gen recommended: 1.05-1.1)")
    ap.add_argument("--repetition-context-size", type=int, default=20)
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    if _is_turboquant_model(args.model):
        print("  → detected TurboQuant checkpoint; using load_turboquant")
        model, tokenizer = load_turboquant(args.model, fast=args.fast)
    else:
        model, tokenizer = load(args.model)
    formatted = _format_prompt(tokenizer, args.prompt)

    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    logits_processors = make_logits_processors(
        repetition_penalty=(
            args.repetition_penalty if args.repetition_penalty != 1.0 else None
        ),
        repetition_context_size=args.repetition_context_size,
    ) or []

    if args.min_tokens > 0:
        from turboquant_mlx.sampling import (
            eos_token_ids,
            make_min_tokens_logits_processor,
        )
        proc = make_min_tokens_logits_processor(
            args.min_tokens, eos_token_ids(tokenizer)
        )
        if proc is not None:
            logits_processors.append(proc)

    if not logits_processors:
        logits_processors = None

    print(f"\nPrompt: {args.prompt}\n" + "=" * 72)

    for name, cfg in CONFIGS:
        mx.random.seed(args.seed)
        # Always start from make_prompt_cache so hybrid models (Nemotron-H,
        # Qwen3.5, etc.) get the correct cache type per layer, then convert
        # only the standard KVCache entries to TurboQuant.
        cache = make_prompt_cache(model)
        if cfg is not None:
            cache = convert_cache_to_turboquant(cache, **cfg)

        print(f"\n[{name}]")
        print("-" * 72, flush=True)
        t0 = time.time()
        generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            prompt_cache=cache,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=True,
        )
        dt = time.time() - t0

        try:
            mb = sum(c.nbytes for c in cache) / (1024 * 1024)
            mb_str = f"{mb:.1f} MiB"
        except Exception:
            mb_str = "n/a"

        print(f"\n  → {dt:.1f}s, cache {mb_str}\n", flush=True)


if __name__ == "__main__":
    main()
