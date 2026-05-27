"""Long-context KV-compression crossover benchmark for a resident 120B-class model.

Settles the question sbayer2's 48 GB box could not (turboquant-mlx#14): on a large
per-token-KV model, does TQ KV compression stay *faster* than fp16 as context grows,
or does the same widening decode penalty seen on the 35B eventually appear?

Loads the model RESIDENT once, then for each context length runs decode under
fp16 KV and under K8/V3, recording prompt/decode tok/s and peak memory. Results are
written to JSON after every run, so a kernel panic or OOM near the wired cap keeps
all completed rows. The sweep stops at the first OOM (larger contexts would also OOM).

Requires the Metal wired cap raised first, e.g. `sudo sysctl iogpu.wired_limit_mb=58000`.
"""

from __future__ import annotations

import gc
import json
import sys
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_kv_cache import convert_cache_to_turboquant
from turboquant_mlx.sampling import eos_token_ids, make_min_tokens_logits_processor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/qwen3.5-122b-tq3"
CTX_LENS = [256, 1024, 2048, 4096, 8192, 16384]  # ascending; stop at first OOM
GEN = 64                                          # enough to get a stable decode rate
OUT = "benchmarks/bench_122b_kv_longctx_results.json"

# KV configs to compare at each context length.
KV_CONFIGS = [("fp16", None), ("K8/V3", (8, 3))]


def main():
    t0 = time.time()
    path = str(resolve_model_path(MODEL))
    print(f"[bench] loading {MODEL} RESIDENT (lazy=False) ...", flush=True)
    model, tok = load_turboquant(path, lazy=False)
    print(f"[bench] loaded in {time.time() - t0:.1f}s, "
          f"resident peak={mx.get_peak_memory() / 1e9:.1f} GB", flush=True)

    # A long, varied base token stream we slice to hit each target context length.
    base_text = (
        "The history of computing spans centuries of human ingenuity, from the "
        "abacus and mechanical calculators to vacuum tubes, transistors, and the "
        "integrated circuits that power modern processors. Each generation of "
        "hardware unlocked new classes of software, and each new application "
        "demanded faster, denser, more energy-efficient machines in return. "
    )
    base_ids = tok.encode(base_text)

    def prompt_of(n_tokens: int):
        reps = n_tokens // len(base_ids) + 1
        return (base_ids * reps)[:n_tokens]

    results = []

    def save():
        with open(OUT, "w") as f:
            json.dump({"model": MODEL, "gen_tokens": GEN, "runs": results}, f, indent=2)

    def run_one(ctx: int, label: str, kv):
        cache = make_prompt_cache(model)
        if kv is not None:
            cache = convert_cache_to_turboquant(
                cache, tq_bits=None, k_bits=kv[0], v_bits=kv[1],
                group_size=64, min_tokens_before_quant=128,
            )
        sampler = make_sampler(temp=0.0)  # greedy: deterministic, speed is what we measure
        mtp = make_min_tokens_logits_processor(50, eos_token_ids(tok))
        lps = [mtp] if mtp is not None else None

        mx.reset_peak_memory()
        last = None
        for resp in stream_generate(
            model, tok, prompt=prompt_of(ctx), max_tokens=GEN,
            sampler=sampler, logits_processors=lps, prompt_cache=cache,
        ):
            last = resp
        row = {
            "ctx": ctx,
            "kv": label,
            "prompt_tps": round(last.prompt_tps, 1),
            "decode_tps": round(last.generation_tps, 1),
            "gen_tokens": last.generation_tokens,
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        del cache, last
        gc.collect()
        mx.clear_cache()
        return row

    print(f"\n{'ctx':>7} {'kv':>7} {'prompt t/s':>11} {'decode t/s':>11} {'peak GB':>9}")
    print("-" * 50)
    stop = False
    for ctx in CTX_LENS:
        if stop:
            break
        for label, kv in KV_CONFIGS:
            try:
                row = run_one(ctx, label, kv)
                results.append(row)
                save()
                print(f"{row['ctx']:>7} {row['kv']:>7} {row['prompt_tps']:>11} "
                      f"{row['decode_tps']:>11} {row['peak_gb']:>9}", flush=True)
            except Exception as e:  # Metal OOM etc. — keep prior rows, stop escalating
                results.append({"ctx": ctx, "kv": label, "error": repr(e)[:200]})
                save()
                print(f"{ctx:>7} {label:>7}  FAILED: {repr(e)[:120]}", flush=True)
                stop = True
                break

    # Crossover summary: fp16 vs K8/V3 decode at each context.
    print("\n[bench] decode-speed crossover (fp16 vs K8/V3):")
    by_ctx = {}
    for r in results:
        if "error" in r:
            continue
        by_ctx.setdefault(r["ctx"], {})[r["kv"]] = r["decode_tps"]
    for ctx in sorted(by_ctx):
        d = by_ctx[ctx]
        if "fp16" in d and "K8/V3" in d:
            f, k = d["fp16"], d["K8/V3"]
            verdict = f"K8/V3 {k / f:.2f}x faster" if k > f else f"fp16 {f / k:.2f}x faster"
            print(f"  ctx={ctx:>6}: fp16={f:>6} t/s  K8/V3={k:>6} t/s  -> {verdict}")
    print(f"\n[bench] results -> {OUT}")


if __name__ == "__main__":
    main()
