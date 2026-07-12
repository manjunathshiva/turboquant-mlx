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
import json
import os
import subprocess
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from turboquant_mlx.generate import resolve_model_path
from turboquant_mlx.sampling import (
    make_single_think_close_logits_processor,
    think_close_token_id,
)

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
                        "permanently resident (from calibrate_experts.py). Without it, "
                        "a hot_experts.json shipped in the model directory is used "
                        "automatically. Pins are preloaded at startup.")
    p.add_argument("--no-hotlist", dest="use_hotlist", action="store_false",
                   default=True,
                   help="Ignore a hot_experts.json shipped in the model directory.")
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
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling threshold. Defaults to the model's "
                        "generation_config.json value when present, else disabled. "
                        "Pass 0 to force-disable.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-k truncation. Defaults to the model's "
                        "generation_config.json value when present, else disabled. "
                        "Pass 0 to force-disable.")
    p.add_argument("--rep-penalty", type=float, default=None,
                   help="Repetition penalty (e.g. 1.05). Defaults to the model's "
                        "generation_config.json value when present, else disabled. "
                        "Pass 0 or 1 to force-disable.")
    p.add_argument("--rep-ctx", type=int, default=256,
                   help="Repetition penalty context window in tokens (default: 256).")
    p.add_argument("--no-think", action="store_true",
                   help="Disable thinking mode via the chat template "
                        "(enable_thinking=False). Much faster and immune to "
                        "think-block loops on thinking-capable models.")
    p.add_argument("--multi-think", action="store_true",
                   help="Allow more than one </think> token per generation. By default "
                        "a second </think> is masked once one has been emitted.")
    args = p.parse_args()

    t0 = time.time()
    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers, prefetch_ahead=args.prefetch_ahead,
        pin_file=args.pin_file, max_active_experts=args.max_active_experts,
        use_page_cache=args.use_page_cache, use_hotlist=args.use_hotlist,
    )
    print(f"[stream] loaded in {time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB")

    prompt = args.prompt
    if not args.no_chat_template and hasattr(tok, "apply_chat_template"):
        template_kwargs = {"enable_thinking": False} if args.no_think else {}
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True,
            **template_kwargs,
        )

    # Sampling defaults come from the model's own generation_config.json,
    # mirroring turboquant_mlx.generate: top_p/top_k truncation (a stray
    # '</think>' sampled where EOS belongs doubles the answer) and
    # repetition_penalty for loop-prone low-bit thinking builds.
    top_p, top_k, rep_penalty = args.top_p, args.top_k, args.rep_penalty
    if top_p is None or top_k is None or rep_penalty is None:
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
            print(f"[INFO] Could not read generation_config.json for "
                  f"{args.model}; sampling without truncation defaults ({e})")
    # 0 or 1.0 on the CLI force-disables (1.0 is mathematically neutral)
    if rep_penalty is not None and rep_penalty in (0.0, 1.0):
        rep_penalty = None
    sampler = make_sampler(
        temp=args.temp,
        top_p=top_p if top_p is not None else 0.0,
        top_k=top_k if top_k is not None else 0,
    )
    logits_processors = []
    if rep_penalty is not None:
        logits_processors.extend(make_logits_processors(
            repetition_penalty=rep_penalty,
            repetition_context_size=args.rep_ctx,
        ))
    if not args.multi_think:
        think_guard = make_single_think_close_logits_processor(
            think_close_token_id(tok)
        )
        if think_guard is not None:
            logits_processors.append(think_guard)
    if not logits_processors:
        logits_processors = None

    print("=" * 60)
    t = time.time()
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=sampler, logits_processors=logits_processors,
                        verbose=True)
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
