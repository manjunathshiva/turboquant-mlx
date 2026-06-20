"""Generate from a TurboQuant MoE model with experts streamed from disk.

Runs a model whose weights exceed available RAM by keeping only the resident
tensors in memory and streaming router-selected experts on demand.

Example (Qwen3.6-35B-A3B, ~16 GB on disk, runs in ~5 GB RAM):

    python -m turboquant_mlx.stream.stream_generate \\
        --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \\
        --prompt "Explain why the sky is blue." \\
        --max-tokens 256 --cache-budget-gb 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from .loader import load_streaming


def _rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out) / 1024 / 1024


def main():
    p = argparse.ArgumentParser(
        description="Stream-generate from a TurboQuant MoE model (experts paged from disk)."
    )
    p.add_argument("--model", required=True, help="Local path or HF repo id of a TurboQuant model.")
    p.add_argument("--prompt", default="Why is the sky blue?")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cache-budget-gb", type=float, default=3.0,
                   help="Max resident expert memory (LRU-evicted). Lower = less RAM, more disk reads.")
    p.add_argument("--prefetch-workers", type=int, default=8,
                   help="Threads for parallel per-layer expert reads. 1 = serial baseline.")
    p.add_argument("--prefetch-ahead", type=int, default=0,
                   help="Speculatively prefetch this many upcoming layers' experts "
                        "(predicted from the previous token's routing). 0 = off (default; "
                        "helps only on fast NVMe with spare bandwidth, ~neutral on a "
                        "saturated USB bus). Set 1 to enable; it self-disables if the "
                        "storage proves bandwidth-bound.")
    p.add_argument("--pin-file", default=None,
                   help="JSON {'pin': [[layer, expert], ...]} of hot experts to keep "
                        "permanently resident (from calibrate_experts.py).")
    p.add_argument("--max-active-experts", type=int, default=4,
                   help="Cap router top_k to min(native, this) on every MoE block "
                        "(Flash-MoE K-reduction: ~2x less streamed disk I/O at no quality "
                        "cost up to K=4 on validated models). Default 4; 0 = native routing.")
    p.add_argument("--use-page-cache", dest="use_page_cache", action="store_true",
                   default=None,
                   help="Force the OS page cache ON for expert reads ('trust the OS'; "
                        "~2.4x faster decode when the model fits in RAM). Default: auto "
                        "by model-size-vs-RAM.")
    p.add_argument("--no-page-cache", dest="use_page_cache", action="store_false",
                   help="Force F_NOCACHE (page cache off). Default: auto by model-size-vs-RAM.")
    p.add_argument("--fast", action="store_true", help="Disable QJL correction for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers, prefetch_ahead=args.prefetch_ahead,
        pin_file=args.pin_file, max_active_experts=args.max_active_experts,
        use_page_cache=args.use_page_cache,
    )
    print(f"[stream] loaded in {time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB")

    prompt = args.prompt
    if not args.no_chat_template and hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        )

    sampler = make_sampler(temp=args.temp)
    print("=" * 60)
    t = time.time()
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=sampler, verbose=True)
    dt = time.time() - t
    print("=" * 60)
    # Count tokens actually generated (the model may stop at EOS before
    # max_tokens) — dividing max_tokens by wall-time overstates the rate.
    n = len(tok.encode(text))
    print(f"[stream] {n} generated tok in {dt:.1f}s = {n / dt:.1f} tok/s (end-to-end) | "
          f"peak RSS={_rss_gb():.2f} GB | mlx_peak={mx.get_peak_memory() / 1e9:.2f} GB")
    s = cache.stats()
    print(f"[stream] expert cache: hit_rate={s['hit_rate']:.1%} "
          f"(resident {s['cache_hit_rate']:.1%} + prefetch {s['prefetch_hit_rate']:.1%}) "
          f"resident={s['resident_gb']:.2f} GB")
    print(f"[stream] disk: critical_read={s['bytes_read_gb']:.1f} GB "
          f"prefetched={s['bytes_prefetched_gb']:.1f} GB total={s['bytes_total_gb']:.1f} GB "
          f"| prefetched_experts={s['prefetched']} dropped_unused={s['prefetch_dropped']}")
    print(f"[stream] coalescing: {s['expert_reads']} expert-loads in {s['read_runs']} "
          f"range-reads = {s['experts_per_read']:.2f} experts/read")


if __name__ == "__main__":
    main()
