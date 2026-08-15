"""Per-head_dim PPL sweep for TurboQuantKVCache (v0.2 item #3).

Measures WikiText-2 perplexity and greedy-decode parity for a matrix of
(K bits, V bits, min_tokens_before_quant) configurations across multiple
models. KV-cache quantization quality only manifests during autoregressive
decoding, so PPL is computed token-by-token through the cache rather than
in a single teacher-forcing forward pass.

Usage:
    python -m turboquant_mlx.benchmarks.eval_kv_cache_ppl \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
        --num-windows 4 \\
        --window-len 4096 \\
        --output benchmarks/eval_kv_cache_ppl_llama3_3b.json

Default config matrix (override with --configs custom.json):
    fp16 baseline
    K8 + V4 / V3 / V2
    K4 + V4 / V3 / V2
    K3 + V3 / K2 + V2 (uniform low-bit)
    K8 + V3 with min_tokens_before_quant=128 (sink-protected)

Model suggestions (from project_kv_cache_v02_plan.md):
    mlx-community/Llama-3.2-3B-Instruct-4bit          (D=128, GQA)
    mlx-community/Qwen2.5-7B-Instruct-4bit            (D=128, GQA)
    mlx-community/Mistral-7B-Instruct-v0.3-4bit       (D=128, MHA)
    mlx-community/gemma-3-4b-it-4bit                  (D=256)
    mlx-community/Qwen3.5-35B-A3B-4bit                (D=128, hybrid)
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx

from turboquant_mlx.layers.polar_kv_cache import make_turboquant_cache


# ── Default config matrix ────────────────────────────────────────────────

DEFAULT_CONFIGS = [
    {"name": "fp16",      "kind": "baseline"},
    {"name": "K8_V4",     "kind": "tq", "k_bits": 8, "v_bits": 4, "thr": 0},
    {"name": "K8_V3",     "kind": "tq", "k_bits": 8, "v_bits": 3, "thr": 0},
    {"name": "K8_V2",     "kind": "tq", "k_bits": 8, "v_bits": 2, "thr": 0},
    {"name": "K4_V4",     "kind": "tq", "k_bits": 4, "v_bits": 4, "thr": 0},
    {"name": "K4_V3",     "kind": "tq", "k_bits": 4, "v_bits": 3, "thr": 0},
    {"name": "K4_V2",     "kind": "tq", "k_bits": 4, "v_bits": 2, "thr": 0},
    {"name": "K3_V3",     "kind": "tq", "k_bits": 3, "v_bits": 3, "thr": 0},
    {"name": "K2_V2",     "kind": "tq", "k_bits": 2, "v_bits": 2, "thr": 0},
    {"name": "K8_V3_thr128", "kind": "tq", "k_bits": 8, "v_bits": 3, "thr": 128},
]


def _load_wikitext_windows(tokenizer, num_windows: int, window_len: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    tokens = tokenizer.encode(text)
    if hasattr(tokens, "input_ids"):
        tokens = list(tokens.input_ids)
    if not isinstance(tokens, list):
        tokens = list(tokens)

    stride = window_len // 2  # half-window stride
    windows = []
    pos = 0
    while pos + window_len <= len(tokens) and len(windows) < num_windows:
        windows.append(mx.array(tokens[pos : pos + window_len]))
        pos += stride
    return windows


def _build_cache(model, cfg):
    """Build a per-layer cache list matching the requested config."""
    if cfg["kind"] == "baseline":
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(model)
    return make_turboquant_cache(
        model,
        k_bits=cfg["k_bits"],
        v_bits=cfg["v_bits"],
        min_tokens_before_quant=cfg.get("thr", 0),
    )


def _ppl_through_cache(model, tokens: mx.array, cache, prefill_len: int):
    """Compute PPL on the second half of `tokens` using `cache` for context.

    Prefill the first `prefill_len` tokens into the cache in one pass,
    then score the remaining tokens one at a time. The KV cache state
    affects every per-token logit, so quantization error shows up here.

    Returns: (avg_nll, n_tokens_scored).
    """
    seq_len = tokens.shape[-1]
    if prefill_len >= seq_len:
        raise ValueError("prefill_len must be < seq_len")

    # Prefill: feed the first prefill_len tokens, populate cache
    prefill_ids = tokens[:prefill_len][None, :]
    _ = model(prefill_ids, cache=cache)
    mx.eval(_)

    total_nll = 0.0
    n_scored = 0

    # Score one token at a time
    for i in range(prefill_len, seq_len - 1):
        cur = tokens[i : i + 1][None, :]            # (1, 1)
        target = tokens[i + 1 : i + 2]              # (1,)
        logits = model(cur, cache=cache)            # (1, 1, V)
        log_probs = logits[:, -1, :] - mx.logsumexp(
            logits[:, -1, :], axis=-1, keepdims=True
        )
        target_lp = log_probs[0, target[0]]
        mx.eval(target_lp)
        total_nll -= float(target_lp.item())
        n_scored += 1

    return total_nll / max(n_scored, 1), n_scored


def _greedy_decode_parity(model, tokenizer, prompt: str, cache_cfg, n_tokens: int):
    """Compare greedy decode against fp16 baseline.

    Returns (matching_tokens, total_tokens).
    """
    from mlx_lm.models.cache import make_prompt_cache

    prompt_ids = mx.array(tokenizer.encode(prompt))[None, :]

    def _decode_with(cache_builder):
        cache = cache_builder()
        out_tokens = []
        # Prefill
        _ = model(prompt_ids, cache=cache)
        mx.eval(_)
        # Decode greedily
        last = prompt_ids
        for _i in range(n_tokens):
            logits = model(last[:, -1:], cache=cache)
            nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(nxt)
            tok_id = int(nxt[0, 0].item())
            out_tokens.append(tok_id)
            last = nxt
        return out_tokens

    baseline_tokens = _decode_with(lambda: make_prompt_cache(model))
    cfg_tokens = _decode_with(lambda: _build_cache(model, cache_cfg))

    matches = sum(1 for a, b in zip(baseline_tokens, cfg_tokens) if a == b)
    return matches, len(baseline_tokens)


def _evaluate_config(model, tokenizer, cfg, windows, prefill_len, parity_prompts, parity_tokens):
    """Run PPL + greedy-decode parity for one (model, config)."""
    ppls = []
    n_tokens_total = 0
    t0 = time.time()
    for w_idx, w in enumerate(windows):
        cache = _build_cache(model, cfg)
        nll, n = _ppl_through_cache(model, w, cache, prefill_len)
        ppls.append(math.exp(nll))
        n_tokens_total += n
        print(f"    window {w_idx + 1}/{len(windows)}: ppl={ppls[-1]:.3f}")
    elapsed = time.time() - t0
    avg_ppl = float(sum(math.log(p) for p in ppls) / len(ppls))
    avg_ppl = math.exp(avg_ppl)

    # Greedy parity (skip for the baseline config — that IS the baseline)
    parity_pct = None
    if cfg["kind"] != "baseline":
        match_total = 0
        token_total = 0
        for prompt in parity_prompts:
            m, t = _greedy_decode_parity(
                model, tokenizer, prompt, cfg, parity_tokens
            )
            match_total += m
            token_total += t
        parity_pct = 100.0 * match_total / max(token_total, 1)

    return {
        "name": cfg["name"],
        "ppl": round(avg_ppl, 4),
        "ppl_per_window": [round(p, 4) for p in ppls],
        "n_tokens_scored": n_tokens_total,
        "greedy_parity_pct": (
            round(parity_pct, 2) if parity_pct is not None else None
        ),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF model path (mlx-community/...)")
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--window-len", type=int, default=4096)
    parser.add_argument("--prefill-len", type=int, default=2048,
                        help="First N tokens prefilled in one pass (cache "
                             "fills up); the remaining (window_len - N) "
                             "tokens are scored one at a time.")
    parser.add_argument("--output", required=True,
                        help="Path to write JSON results")
    parser.add_argument("--configs", default=None,
                        help="Optional path to a custom JSON config list "
                             "(see DEFAULT_CONFIGS for shape)")
    parser.add_argument("--parity-tokens", type=int, default=64,
                        help="Tokens to compare in greedy-decode parity")
    args = parser.parse_args()

    from mlx_lm import load

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    if args.configs:
        configs = json.loads(Path(args.configs).read_text())
    else:
        configs = DEFAULT_CONFIGS

    print(f"Loading WikiText-2 ({args.num_windows} windows of {args.window_len})...")
    windows = _load_wikitext_windows(
        tokenizer, args.num_windows, args.window_len
    )

    parity_prompts = [
        "The mitochondrion is",
        "Once upon a time",
        "In computer science, a stack is",
    ]

    results = {
        "model": args.model,
        "window_len": args.window_len,
        "prefill_len": args.prefill_len,
        "num_windows": len(windows),
        "parity_tokens": args.parity_tokens,
        "configs": [],
    }

    for cfg in configs:
        print(f"\n[{cfg['name']}]")
        try:
            r = _evaluate_config(
                model, tokenizer, cfg, windows, args.prefill_len,
                parity_prompts, args.parity_tokens,
            )
            print(f"  → ppl={r['ppl']:.3f} parity={r['greedy_parity_pct']}% "
                  f"({r['elapsed_s']}s)")
            results["configs"].append(r)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results["configs"].append(
                {"name": cfg["name"], "error": f"{type(e).__name__}: {e}"}
            )

        # Write incrementally so a long sweep is safe to interrupt
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
