"""Test TurboQuant KV cache compression.

Tests:
1. Correctness: quantize → dequantize roundtrip quality
2. Memory savings: compressed vs float16 cache size
3. End-to-end generation: generate text with TurboQuant KV cache

NOTE: Small models (< 3B) often degrade with ANY KV cache quantization
(including MLX's native kv_bits=4). TQ is designed for large models
(120B+) where more KV heads provide better noise tolerance.

Usage:
    # Quick correctness test (no model needed)
    python -m turboquant_mlx.test_kv_cache --test-roundtrip

    # Generate with a model (compare FP16 vs TQ KV cache)
    python -m turboquant_mlx.test_kv_cache \
        --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt "Why is the sky blue?" --kv-bits 3 --compare

    # Large model test — GPT-OSS (MXFP4, fits in 64GB)
    python -m turboquant_mlx.test_kv_cache \
        --model openai/gpt-oss-120b --kv-bits 3 --compare
"""

import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.layers.polar_kv_cache import (
    TurboQuantKVCache,
    convert_cache_to_turboquant,
)


def test_roundtrip(bits=3, group_size=64, head_dim=128, seq_len=64, n_heads=8):
    """Test quantize → dequantize roundtrip quality."""
    print(f"\n{'='*60}")
    print(f"Roundtrip test: {bits}-bit, group_size={group_size}, "
          f"head_dim={head_dim}, seq_len={seq_len}, n_heads={n_heads}")
    print(f"{'='*60}")

    mx.random.seed(0)
    B = 1
    keys = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal(shape=(B, n_heads, seq_len, head_dim)).astype(mx.float16)
    mx.eval(keys, values)

    cache = TurboQuantKVCache(tq_bits=bits, group_size=group_size, seed=42)
    keys_deq, values_deq = cache.update_and_fetch(keys, values)
    mx.eval(keys_deq, values_deq)

    # Quality metrics
    k_diff = (keys.astype(mx.float32) - keys_deq.astype(mx.float32))
    v_diff = (values.astype(mx.float32) - values_deq.astype(mx.float32))

    k_mse = mx.mean(k_diff * k_diff).item()
    v_mse = mx.mean(v_diff * v_diff).item()
    k_signal = mx.mean(keys.astype(mx.float32) ** 2).item()
    v_signal = mx.mean(values.astype(mx.float32) ** 2).item()

    k_snr = 10 * math.log10(k_signal / max(k_mse, 1e-10))
    v_snr = 10 * math.log10(v_signal / max(v_mse, 1e-10))

    # Cosine similarity
    k_flat = keys.reshape(-1).astype(mx.float32)
    kd_flat = keys_deq.reshape(-1).astype(mx.float32)
    k_cos = (mx.sum(k_flat * kd_flat) / (mx.sqrt(mx.sum(k_flat * k_flat)) * mx.sqrt(mx.sum(kd_flat * kd_flat)))).item()

    v_flat = values.reshape(-1).astype(mx.float32)
    vd_flat = values_deq.reshape(-1).astype(mx.float32)
    v_cos = (mx.sum(v_flat * vd_flat) / (mx.sqrt(mx.sum(v_flat * v_flat)) * mx.sqrt(mx.sum(vd_flat * vd_flat)))).item()

    # Memory comparison
    fp16_bytes = B * n_heads * seq_len * head_dim * 2 * 2
    compressed_bytes = cache.nbytes

    print(f"\nKeys   — MSE: {k_mse:.6f}, cosine: {k_cos:.6f}, SNR: {k_snr:.1f} dB")
    print(f"Values — MSE: {v_mse:.6f}, cosine: {v_cos:.6f}, SNR: {v_snr:.1f} dB")
    print(f"\nMemory: FP16={fp16_bytes/1024:.1f} KB → Compressed={compressed_bytes/1024:.1f} KB "
          f"({fp16_bytes/max(compressed_bytes,1):.1f}x savings)")
    print(f"Cache offset: {cache.offset}, stored tokens: {seq_len}")

    # Test incremental update
    print(f"\nIncremental update test...")
    new_keys = mx.random.normal(shape=(B, n_heads, 1, head_dim)).astype(mx.float16)
    new_values = mx.random.normal(shape=(B, n_heads, 1, head_dim)).astype(mx.float16)
    mx.eval(new_keys, new_values)

    k2, v2 = cache.update_and_fetch(new_keys, new_values)
    mx.eval(k2)

    print(f"After +1 token: cache offset={cache.offset}, "
          f"returned seq_len={k2.shape[-2]}, "
          f"compressed={cache.nbytes/1024:.1f} KB")

    assert k2.shape[-2] == seq_len + 1, f"Expected seq_len={seq_len+1}, got {k2.shape[-2]}"
    assert k2.dtype == mx.float16, f"Expected float16 output, got {k2.dtype}"
    print("✓ Incremental update correct")

    return k_cos, v_cos


def test_multiple_bitwidths():
    """Test all supported bit-widths."""
    print("\n" + "=" * 60)
    print("Testing all bit-widths")
    print("=" * 60)

    for bits in [2, 3, 4]:
        k_cos, v_cos = test_roundtrip(bits=bits, head_dim=128, seq_len=32, n_heads=4)
        status = "✓" if k_cos > 0.95 else "✗"
        print(f"  {bits}-bit: keys cos={k_cos:.4f}, values cos={v_cos:.4f} {status}")


def test_different_head_dims():
    """Test common head dimensions (matching target models)."""
    print("\n" + "=" * 60)
    print("Testing head dimensions for target models")
    print("=" * 60)

    configs = [
        (64, 8, "GPT-OSS-120B"),
        (128, 4, "Standard"),
        (256, 2, "Qwen3.5-122B"),
    ]
    for hd, nh, label in configs:
        gs = min(64, hd)
        k_cos, v_cos = test_roundtrip(bits=3, group_size=gs, head_dim=hd, seq_len=16, n_heads=nh)
        print(f"  {label} (head_dim={hd}, n_kv={nh}): keys cos={k_cos:.4f}, values cos={v_cos:.4f}")


def generate_with_kv_cache(model_path, prompt, kv_bits=3, max_tokens=100, compare=False):
    """Generate text using TurboQuant KV cache."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"\n{'='*60}")
    print(f"Generation test: {model_path}")
    print(f"KV cache: TurboQuant {kv_bits}-bit")
    print(f"{'='*60}")

    model, tokenizer = load(model_path)

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        text = prompt
    input_ids = mx.array(tokenizer.encode(text))[None]
    print(f"Prompt tokens: {input_ids.shape[-1]}")

    # --- Standard FP16 KV cache generation ---
    if compare:
        print(f"\n--- Standard FP16 KV cache ---")
        cache_std = make_prompt_cache(model)
        logits = model(input_ids, cache=cache_std)
        mx.eval(logits)

        t0 = time.time()
        generated_tokens = []
        for i in range(max_tokens):
            token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(token)
            tok_id = token.item()
            if tok_id in getattr(tokenizer, 'eos_token_ids', {tokenizer.eos_token_id}):
                break
            generated_tokens.append(tok_id)
            logits = model(token, cache=cache_std)
            mx.eval(logits)

        t1 = time.time()
        n_gen = len(generated_tokens)
        std_speed = n_gen / (t1 - t0) if (t1 - t0) > 0 else 0

        std_bytes = sum(c.nbytes for c in cache_std if hasattr(c, 'nbytes'))

        output_std = tokenizer.decode(generated_tokens)
        print(f"Generated {n_gen} tokens at {std_speed:.1f} tok/s")
        print(f"KV cache memory: {std_bytes / 1024 / 1024:.2f} MB")
        print(f"Output: {output_std[:200]}")
        del cache_std, logits
        mx.clear_cache()

    # --- TurboQuant KV cache generation ---
    print(f"\n--- TurboQuant {kv_bits}-bit KV cache ---")

    # Process prompt with FP16 cache, then convert to TQ
    # (same pattern as mlx_lm's maybe_quantize_kv_cache)
    cache_fp16 = make_prompt_cache(model)
    logits = model(input_ids, cache=cache_fp16)
    mx.eval(logits)

    cache_tq = convert_cache_to_turboquant(
        cache_fp16, tq_bits=kv_bits, group_size=64, seed=42
    )
    del cache_fp16
    # Evaluate TQ cache state to materialize compressed data
    for c in cache_tq:
        if hasattr(c, '_tq_keys') and c._tq_keys is not None:
            mx.eval(*c._tq_keys, *c._tq_values)

    t0 = time.time()
    generated_tokens = []
    for i in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(token)
        tok_id = token.item()
        if tok_id in getattr(tokenizer, 'eos_token_ids', {tokenizer.eos_token_id}):
            break
        generated_tokens.append(tok_id)
        logits = model(token, cache=cache_tq)
        mx.eval(logits)

    t1 = time.time()
    n_gen = len(generated_tokens)
    tq_speed = n_gen / (t1 - t0) if (t1 - t0) > 0 else 0

    tq_bytes = sum(c.nbytes for c in cache_tq if hasattr(c, 'nbytes'))

    output_tq = tokenizer.decode(generated_tokens)
    print(f"Generated {n_gen} tokens at {tq_speed:.1f} tok/s")
    print(f"KV cache memory: {tq_bytes / 1024 / 1024:.2f} MB")
    print(f"Output: {output_tq[:200]}")

    if compare:
        print(f"\n--- Comparison ---")
        print(f"FP16 KV cache: {std_bytes / 1024 / 1024:.2f} MB, {std_speed:.1f} tok/s")
        print(f"TQ {kv_bits}-bit:     {tq_bytes / 1024 / 1024:.2f} MB, {tq_speed:.1f} tok/s")
        savings = std_bytes / max(tq_bytes, 1)
        print(f"Memory savings: {savings:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TurboQuant KV cache")
    parser.add_argument("--test-roundtrip", action="store_true",
                        help="Run roundtrip quality tests (no model needed)")
    parser.add_argument("--model", type=str, default=None,
                        help="MLX model path for generation test")
    parser.add_argument("--prompt", type=str, default="Why is the sky blue?",
                        help="Prompt for generation")
    parser.add_argument("--kv-bits", type=int, default=3,
                        help="KV cache quantization bits (2, 3, 4)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with standard FP16 KV cache")
    args = parser.parse_args()

    if args.test_roundtrip or args.model is None:
        test_roundtrip(bits=3, head_dim=64, seq_len=32, n_heads=8)
        test_roundtrip(bits=3, head_dim=128, seq_len=32, n_heads=8)
        test_roundtrip(bits=3, head_dim=256, seq_len=32, n_heads=4)
        test_multiple_bitwidths()
        test_different_head_dims()

    if args.model:
        generate_with_kv_cache(
            args.model, args.prompt,
            kv_bits=args.kv_bits,
            max_tokens=args.max_tokens,
            compare=args.compare,
        )
