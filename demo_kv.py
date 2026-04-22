"""Demo: TurboQuant KV cache compression — streaming generation.

For video recording, run twice:

  # Without compression (FP16 KV cache)
  python -m turboquant_mlx.demo_kv --model openai/gpt-oss-20b \
      --prompt "Why is the sky blue?" --max-tokens 200

  # With TQ 3-bit KV cache compression
  python -m turboquant_mlx.demo_kv --model openai/gpt-oss-20b \
      --prompt "Why is the sky blue?" --max-tokens 200 --tq-bits 3

Or run both back-to-back:
  python -m turboquant_mlx.demo_kv --model openai/gpt-oss-20b \
      --prompt "Why is the sky blue?" --max-tokens 200 --compare
"""

import argparse
import sys
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import


def stream_generate(model, tokenizer, input_ids, cache, max_tokens, logits_processor=None):
    """Generate tokens one at a time, printing as they arrive."""
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    tokens = []
    t0 = time.time()
    for i in range(max_tokens):
        step_logits = logits[:, -1, :]
        if logits_processor is not None:
            step_logits = logits_processor(mx.array(tokens), step_logits)
        token = mx.argmax(step_logits, axis=-1, keepdims=True)
        mx.eval(token)
        tok_id = token.item()
        if tok_id in getattr(tokenizer, 'eos_token_ids', {tokenizer.eos_token_id}):
            break
        tokens.append(tok_id)
        # Decode and print incrementally
        text = tokenizer.decode([tok_id])
        print(text, end="", flush=True)
        logits = model(token, cache=cache)
        mx.eval(logits)

    elapsed = time.time() - t0
    return tokens, elapsed


def run_demo(model_path, prompt, max_tokens, tq_bits=None, min_tokens=0):
    from mlx_lm.models.cache import make_prompt_cache
    from turboquant_mlx.layers.polar_kv_cache import convert_cache_to_turboquant
    from turboquant_mlx.generate import resolve_model_path
    from turboquant_mlx.sampling import (
        eos_token_ids,
        make_min_tokens_logits_processor,
    )

    # Resolve HF repo IDs to a local directory up front so the TQ-detection
    # check below (which reads config.json) works for both path forms.
    resolved_path = resolve_model_path(model_path)

    # Detect TQ-converted models vs standard models
    import json
    is_tq_model = False
    config_path = resolved_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        quant = cfg.get("quantization", {})
        is_tq_model = quant.get("mode", "") == "turboquant"

    if is_tq_model:
        from turboquant_mlx.generate import load_turboquant
        model, tokenizer = load_turboquant(resolved_path)
    else:
        from mlx_lm import load
        model, tokenizer = load(str(resolved_path))

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        text = prompt
    input_ids = mx.array(tokenizer.encode(text))[None]

    label = f"TurboQuant {tq_bits}-bit KV" if tq_bits else "FP16 KV (standard)"

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Model: {model_path}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Prompt tokens: {input_ids.shape[-1]}")
    print(f"{'='*60}\n")

    # Build min-tokens logits processor (no-op if min_tokens == 0)
    min_tokens_proc = make_min_tokens_logits_processor(
        min_tokens, eos_token_ids(tokenizer)
    )

    # Create cache and process prompt
    cache = make_prompt_cache(model)
    if tq_bits:
        # Process prompt in FP16, then convert to TQ
        logits_prefill = model(input_ids, cache=cache)
        mx.eval(logits_prefill)
        cache = convert_cache_to_turboquant(cache, tq_bits=tq_bits, group_size=64)
        for c in cache:
            if hasattr(c, '_tq_keys') and c._tq_keys is not None:
                mx.eval(*c._tq_keys, *c._tq_values)
        # Generate from the prefilled logits
        tokens = []
        t0 = time.time()
        logits = logits_prefill
        for i in range(max_tokens):
            step_logits = logits[:, -1, :]
            if min_tokens_proc is not None:
                step_logits = min_tokens_proc(mx.array(tokens), step_logits)
            token = mx.argmax(step_logits, axis=-1, keepdims=True)
            mx.eval(token)
            tok_id = token.item()
            if tok_id in getattr(tokenizer, 'eos_token_ids', {tokenizer.eos_token_id}):
                break
            tokens.append(tok_id)
            text_out = tokenizer.decode([tok_id])
            print(text_out, end="", flush=True)
            logits = model(token, cache=cache)
            mx.eval(logits)
        elapsed = time.time() - t0
    else:
        tokens, elapsed = stream_generate(
            model, tokenizer, input_ids, cache, max_tokens,
            logits_processor=min_tokens_proc,
        )

    # Stats
    n = len(tokens)
    speed = n / elapsed if elapsed > 0 else 0
    kv_bytes = sum(c.nbytes for c in cache if hasattr(c, 'nbytes'))

    print(f"\n\n{'─'*60}")
    print(f"  Tokens: {n}  |  Speed: {speed:.1f} tok/s  |  KV cache: {kv_bytes/1024/1024:.2f} MB")
    print(f"{'─'*60}\n")

    return n, speed, kv_bytes


def main():
    parser = argparse.ArgumentParser(description="TurboQuant KV cache demo")
    parser.add_argument("--model", type=str, required=True,
                        help="MLX model path")
    parser.add_argument("--prompt", type=str, default="Why is the sky blue?",
                        help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--tq-bits", type=int, default=None,
                        help="TQ compression bits (3 or 4). Omit for FP16 baseline")
    parser.add_argument("--compare", action="store_true",
                        help="Run both FP16 and TQ 3-bit back-to-back")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Mask EOS until at least this many tokens are generated. "
                             "Useful for thinking-mode models (Nemotron 3, etc.) "
                             "whose chat template primes EOS as the top-1 logit")
    args = parser.parse_args()

    if args.compare:
        # Run FP16 first, then TQ
        n1, s1, m1 = run_demo(args.model, args.prompt, args.max_tokens,
                              tq_bits=None, min_tokens=args.min_tokens)
        mx.clear_cache()
        n2, s2, m2 = run_demo(args.model, args.prompt, args.max_tokens,
                              tq_bits=3, min_tokens=args.min_tokens)

        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"  FP16:     {s1:.1f} tok/s  |  KV cache: {m1/1024/1024:.2f} MB")
        print(f"  TQ 3-bit: {s2:.1f} tok/s  |  KV cache: {m2/1024/1024:.2f} MB")
        print(f"  Memory savings: {m1/max(m2,1):.1f}x")
        print(f"{'='*60}\n")
    else:
        run_demo(args.model, args.prompt, args.max_tokens,
                 tq_bits=args.tq_bits, min_tokens=args.min_tokens)


if __name__ == "__main__":
    main()
