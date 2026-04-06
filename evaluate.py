"""Evaluate TurboQuant models: perplexity, model size, and generation quality.

Compares TurboQuant (various bit-widths, with/without QJL) against
FP16 baseline and MLX affine quantization on WikiText-2.

Usage:
    python -m turboquant_mlx.evaluate \
        --hf-path meta-llama/Llama-3.2-1B \
        --bits 2 3 4 \
        --include-affine \
        --include-qjl \
        --num-samples 512
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize


def _load_wikitext2(tokenizer, max_samples: int = 512, seq_len: int = 512):
    """Load WikiText-2 test set and tokenize into fixed-length chunks.

    Returns list of token arrays, each of length seq_len.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in ds["text"] if t.strip()])
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for evaluation. "
            "Install it with: pip install datasets"
        )

    tokens = tokenizer.encode(text)
    if isinstance(tokens, list):
        tokens = mx.array(tokens)
    elif hasattr(tokens, "input_ids"):
        tokens = mx.array(tokens.input_ids)

    # Split into fixed-length chunks
    n_chunks = min(max_samples, len(tokens) // seq_len)
    chunks = []
    for i in range(n_chunks):
        start = i * seq_len
        chunks.append(tokens[start : start + seq_len])
    return chunks


def compute_perplexity(model, tokenizer, chunks, batch_size: int = 1):
    """Compute perplexity over tokenized chunks.

    Uses teacher-forcing: feed all tokens, compute cross-entropy loss
    at each position against the next token.
    """
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(chunks), batch_size):
        batch = mx.stack(chunks[i : i + batch_size])  # (B, seq_len)
        inputs = batch[:, :-1]   # (B, seq_len-1)
        targets = batch[:, 1:]   # (B, seq_len-1)

        logits = model(inputs)   # (B, seq_len-1, vocab)
        mx.eval(logits)

        # Cross-entropy loss
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Gather log probs at target positions
        B, T, V = log_probs.shape
        # Flatten for indexing
        log_probs_flat = log_probs.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        target_log_probs = log_probs_flat[mx.arange(B * T), targets_flat]
        mx.eval(target_log_probs)

        nll = -target_log_probs.sum().item()
        total_nll += nll
        total_tokens += B * T

        if (i // batch_size) % 50 == 0:
            running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
            print(f"  [{i+1}/{len(chunks)}] running ppl: {running_ppl:.2f}")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity


def measure_model_size(model):
    """Measure total model size in bytes and original weight count.

    For quantized models, estimates the original parameter count from the
    model config rather than counting packed array elements.
    """
    total_bytes = 0
    leaves = nn.utils.tree_flatten(model.parameters())
    for _, v in leaves:
        total_bytes += v.nbytes

    # Count original (unquantized) parameters by examining model structure
    original_params = 0
    for name, module in model.named_modules():
        if hasattr(module, "num_experts") and hasattr(module, "input_dims") and hasattr(module, "output_dims"):
            # PolarQuantizedSwitchLinear — MoE expert weights
            original_params += module.num_experts * module.input_dims * module.output_dims
            if "bias" in module:
                original_params += module.num_experts * module.output_dims
        elif hasattr(module, "input_dims") and hasattr(module, "output_dims"):
            # PolarQuantizedLinear — has explicit input_dims/output_dims
            original_params += module.input_dims * module.output_dims
            if "bias" in module:
                original_params += module.output_dims
        elif isinstance(module, nn.QuantizedLinear):
            # MLX QuantizedLinear — infer dims from scales shape
            out_features = module.weight.shape[0]
            in_features = module.scales.shape[1] * module.group_size
            original_params += in_features * out_features
            if "bias" in module:
                original_params += out_features
        elif isinstance(module, nn.Linear):
            original_params += module.weight.size
            if "bias" in module:
                original_params += module.bias.size
        elif isinstance(module, nn.QuantizedEmbedding):
            # MLX QuantizedEmbedding — infer dims from scales
            num_emb = module.weight.shape[0]
            emb_dim = module.scales.shape[1] * module.group_size
            original_params += num_emb * emb_dim
        elif isinstance(module, nn.Embedding):
            original_params += module.weight.size
        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            if hasattr(module, "weight"):
                original_params += module.weight.size

    # Fallback: if we didn't find structured params, count raw array elements
    if original_params == 0:
        for _, v in leaves:
            original_params += v.size

    return total_bytes, original_params


def quantize_affine(model, config, bits, group_size):
    """Apply MLX built-in affine quantization for baseline comparison."""
    nn.quantize(model, bits=bits, group_size=group_size)
    config = dict(config)
    config["quantization"] = {"bits": bits, "group_size": group_size, "mode": "affine"}
    return model, config


def evaluate_config(
    hf_path: str,
    config_name: str,
    quantize_fn,
    tokenizer,
    chunks,
    load_fn,
):
    """Evaluate a single quantization configuration.

    Args:
        hf_path: Model path for loading.
        config_name: Human-readable config name.
        quantize_fn: Callable(model, config) -> (model, config), or None for FP16.
        tokenizer: Tokenizer instance.
        chunks: Tokenized evaluation chunks.
        load_fn: Callable() -> (model, config) to load a fresh model.

    Returns:
        Dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*60}")

    # Load fresh model
    print(f"  Loading model...")
    t0 = time.time()
    model, config = load_fn()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Quantize if needed
    if quantize_fn is not None:
        print(f"  Quantizing...")
        t0 = time.time()
        model, config = quantize_fn(model, config)
        mx.eval(model.parameters())
        quant_time = time.time() - t0
        print(f"  Quantized in {quant_time:.1f}s")
    else:
        quant_time = 0.0
        mx.eval(model.parameters())

    # Measure size
    total_bytes, total_params = measure_model_size(model)
    size_gb = total_bytes / (1024**3)
    bpw = (total_bytes * 8) / total_params if total_params > 0 else 0
    print(f"  Size: {size_gb:.3f} GB ({bpw:.2f} bits/weight, {total_params:,} params)")

    # Compute perplexity
    print(f"  Computing perplexity...")
    t0 = time.time()
    ppl = compute_perplexity(model, tokenizer, chunks)
    eval_time = time.time() - t0
    print(f"  Perplexity: {ppl:.2f} (computed in {eval_time:.1f}s)")

    # Free memory
    del model
    mx.clear_cache()

    return {
        "config": config_name,
        "perplexity": round(ppl, 2),
        "size_gb": round(size_gb, 3),
        "bits_per_weight": round(bpw, 2),
        "params": total_params,
        "quant_time_s": round(quant_time, 1),
        "eval_time_s": round(eval_time, 1),
    }


def run_evaluation(
    hf_path: str,
    bits_list: list[int] = (3, 4),
    group_size: int = 64,
    include_affine: bool = True,
    include_qjl: bool = True,
    num_samples: int = 512,
    seq_len: int = 512,
    output_path: str = None,
):
    """Run full evaluation suite on a model."""
    from mlx_lm.utils import load

    print(f"Model: {hf_path}")
    print(f"Configs: bits={bits_list}, gs={group_size}, affine={include_affine}, qjl={include_qjl}")
    print(f"Eval: {num_samples} chunks x {seq_len} tokens")

    # Load tokenizer once (shared across all configs)
    print("\nLoading tokenizer...")
    _, tokenizer = load(hf_path, lazy=True)

    # Prepare evaluation data
    print("Tokenizing WikiText-2...")
    chunks = _load_wikitext2(tokenizer, max_samples=num_samples, seq_len=seq_len)
    print(f"Prepared {len(chunks)} chunks of {seq_len} tokens")

    def load_fresh():
        model, _, config = load(hf_path, return_config=True, lazy=False)
        return model, config

    results = []

    # 1. FP16 baseline
    results.append(evaluate_config(
        hf_path, "FP16 (baseline)", None, tokenizer, chunks, load_fresh,
    ))

    # 2. TurboQuant configs
    for bits in bits_list:
        tq_config = TurboQuantConfig(bits=bits, group_size=group_size, use_qjl=False)

        def make_tq_fn(cfg):
            def fn(model, config):
                return turboquant_quantize(model, config, cfg)
            return fn

        eff_bits = tq_config.effective_bits
        results.append(evaluate_config(
            hf_path,
            f"TurboQuant {bits}-bit (eff {eff_bits:.2f})",
            make_tq_fn(tq_config),
            tokenizer, chunks, load_fresh,
        ))

    # 3. TurboQuant + QJL configs
    if include_qjl:
        for bits in bits_list:
            tq_config = TurboQuantConfig(bits=bits, group_size=group_size, use_qjl=True)

            def make_tq_qjl_fn(cfg):
                def fn(model, config):
                    return turboquant_quantize(model, config, cfg)
                return fn

            eff_bits = tq_config.effective_bits
            results.append(evaluate_config(
                hf_path,
                f"TurboQuant {bits}+QJL (eff {eff_bits:.2f})",
                make_tq_qjl_fn(tq_config),
                tokenizer, chunks, load_fresh,
            ))

    # 4. Affine baselines
    if include_affine:
        for bits in bits_list:
            def make_affine_fn(b, gs):
                def fn(model, config):
                    return quantize_affine(model, config, b, gs)
                return fn

            results.append(evaluate_config(
                hf_path,
                f"Affine {bits}-bit (gs={group_size})",
                make_affine_fn(bits, group_size),
                tokenizer, chunks, load_fresh,
            ))

    # Print summary table
    print(f"\n{'='*80}")
    print(f"RESULTS: {hf_path}")
    print(f"{'='*80}")
    header = f"{'Config':<35} {'PPL':>8} {'Size(GB)':>10} {'BPW':>6} {'QTime':>7}"
    print(header)
    print("-" * 80)
    baseline_ppl = results[0]["perplexity"] if results else None
    for r in results:
        delta = ""
        if baseline_ppl and r["config"] != "FP16 (baseline)":
            ppl_diff = r["perplexity"] - baseline_ppl
            delta_pct = (ppl_diff / baseline_ppl) * 100
            delta = f" ({'+' if ppl_diff >= 0 else ''}{delta_pct:.1f}%)"
        line = (
            f"{r['config']:<35} "
            f"{r['perplexity']:>8.2f}{delta:<10} "
            f"{r['size_gb']:>8.3f} "
            f"{r['bits_per_weight']:>6.2f} "
            f"{r['quant_time_s']:>6.1f}s"
        )
        print(line)

    # Save results
    if output_path is None:
        model_name = hf_path.split("/")[-1].lower().replace("-", "_")
        output_path = f"turboquant_mlx/benchmarks/eval_{model_name}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": hf_path,
            "num_samples": len(chunks),
            "seq_len": seq_len,
            "group_size": group_size,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TurboQuant vs baselines on WikiText-2 perplexity"
    )
    parser.add_argument(
        "--hf-path", "--model",
        type=str, required=True,
        help="HuggingFace model path (e.g. meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--bits", "-b",
        type=int, nargs="+", default=[2, 3, 4],
        help="Bit-widths to evaluate (default: 2 3 4)",
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int, default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--include-affine",
        action="store_true", default=True,
        help="Include MLX affine quantization baselines (default: True)",
    )
    parser.add_argument(
        "--no-affine",
        action="store_true",
        help="Skip affine quantization baselines",
    )
    parser.add_argument(
        "--include-qjl",
        action="store_true", default=True,
        help="Include TurboQuant+QJL configs (default: True)",
    )
    parser.add_argument(
        "--no-qjl",
        action="store_true",
        help="Skip QJL configs",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int, default=512,
        help="Number of evaluation chunks (default: 512)",
    )
    parser.add_argument(
        "--seq-len",
        type=int, default=512,
        help="Sequence length per chunk (default: 512)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Output JSON path (default: auto-generated)",
    )
    args = parser.parse_args()

    run_evaluation(
        hf_path=args.hf_path,
        bits_list=args.bits,
        group_size=args.group_size,
        include_affine=not args.no_affine,
        include_qjl=not args.no_qjl,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
