"""Calibrate MoE expert usage to drive frequency-pinning (#2) and co-activation
on-disk layout (#3) for the streaming runtime.

Two stages:

  collect : run a set of representative prompts through the streaming model and
            record, for every *decode* token, the per-layer set of selected
            experts. Dumps a trace JSON of ``[layer, [experts...]]`` records.

  analyze : turn a trace into
              * pin.json  — {"pin": [[layer, expert], ...]} the hottest experts
                            to keep permanently resident, sized to --pin-gb.
              * perm.json — {"perm": {layer: [expert_ids in new disk order]}}
                            a per-layer ordering that places frequently
                            co-activated experts adjacent (for the #3 repack).

Example:
  python -m turboquant_mlx.stream.calibrate_experts collect \
      --model .../qwen3.5-122b-tq3 --budget 30 --out /tmp/trace_122b.json
  python -m turboquant_mlx.stream.calibrate_experts analyze \
      --model .../qwen3.5-122b-tq3 --trace /tmp/trace_122b.json \
      --pin-gb 8 --pin-out /tmp/pin_122b.json --perm-out /tmp/perm_122b.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

# A spread of tasks so the routing trace reflects general use, not one domain.
CALIB_PROMPTS = [
    "Write a detailed, well-structured 300-word essay on the history of the printing press.",
    "Write a Python function that returns the nth Fibonacci number using memoization, with examples.",
    "A store sells notebooks at $7 each. With a 15% bulk discount on orders of 20 or more, how much do 24 notebooks cost? Show your steps.",
    "Explain how a transformer neural network processes a sentence, from tokenization through attention to the output distribution.",
    "List the five largest planets in the Solar System as a strict JSON array of objects with 'name' and 'diameter_km'.",
    "Summarize the causes and consequences of the Industrial Revolution in three paragraphs.",
    "Translate the following into French and explain the grammar: 'The weather is beautiful today and the children are playing outside.'",
    "Describe step by step how to debug a segmentation fault in a C program.",
]


def collect(args):
    import turboquant_mlx.compat  # noqa: F401
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    from .loader import load_streaming

    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.budget,
        prefetch_workers=8, prefetch_ahead=0,   # prefetch off — pure routing trace
    )
    cache.set_trace(True)
    sampler = make_sampler(temp=args.temp)
    t_all = time.time()
    for i, pr in enumerate(CALIB_PROMPTS):
        p = pr
        if hasattr(tok, "apply_chat_template"):
            p = tok.apply_chat_template(
                [{"role": "user", "content": pr}], add_generation_prompt=True
            )
        t = time.time()
        mlx_generate(model, tok, prompt=p, max_tokens=args.max_tokens,
                     sampler=sampler, verbose=False)
        print(f"  prompt {i + 1}/{len(CALIB_PROMPTS)} done in {time.time() - t:.1f}s "
              f"| trace records={len(cache._trace)}", flush=True)
    n = cache.dump_trace(args.out)
    print(f"[calibrate] {n} (layer, experts) records over {time.time() - t_all:.0f}s "
          f"-> {args.out}")
    cache.close()


def _model_expert_info(model_path):
    """Return (bytes_per_expert across the 3 projections, num_experts)."""
    from turboquant_mlx.generate import resolve_model_path
    from .safetensors_reader import SafetensorsExpertReader
    r = SafetensorsExpertReader(str(resolve_model_path(model_path)))
    itemsize = {"U32": 4, "F16": 2, "F32": 4}
    cost = 0
    seen = 0
    num_experts = 0
    for key, loc in r._index.items():
        if "switch_mlp" not in key:
            continue
        if not (key.endswith(".weight") or key.endswith(".scales")):
            continue
        num_experts = loc.shape[0]
        per = 1
        for d in loc.shape[1:]:   # bytes per expert = all dims except the expert axis
            per *= d
        cost += per * itemsize[loc.dtype]
        seen += 1
        if seen >= 6:             # gate/up/down × {weight, scales} of one layer
            break
    r.close()
    return cost, num_experts


def _greedy_order(experts, coact: Counter):
    """Nearest-neighbour ordering: place experts so each is adjacent to its
    strongest remaining co-activation partner (clusters co-fired experts)."""
    experts = sorted(experts)
    if not experts:
        return []
    deg = Counter()
    for (e, f), c in coact.items():
        deg[e] += c
        deg[f] += c
    remaining = set(experts)
    cur = max(experts, key=lambda e: deg.get(e, 0))
    order = [cur]
    remaining.discard(cur)
    while remaining:
        best, best_c = None, -1
        for e in remaining:
            key = (cur, e) if cur < e else (e, cur)
            c = coact.get(key, 0)
            if c > best_c:
                best_c, best = c, e
        order.append(best)
        remaining.discard(best)
        cur = best
    return order


def analyze(args):
    with open(args.trace) as f:
        trace = json.load(f)
    freq = defaultdict(Counter)              # layer -> Counter(expert -> count)
    coact = defaultdict(Counter)             # layer -> Counter((e<f) -> count)
    seen_experts = defaultdict(set)
    for layer, experts in trace:
        experts = list(experts)
        for e in experts:
            freq[layer][e] += 1
            seen_experts[layer].add(e)
        for a in range(len(experts)):
            for b in range(a + 1, len(experts)):
                e, f = experts[a], experts[b]
                if e > f:
                    e, f = f, e
                coact[layer][(e, f)] += 1

    # ---- pin.json: hottest (layer, expert) up to the byte budget ----
    cost, num_experts = _model_expert_info(args.model)
    cap = int(args.pin_gb * 1e9)
    ranked = sorted(((c, l, e) for l in freq for e, c in freq[l].items()),
                    reverse=True)
    pin, used = [], 0
    for _cnt, l, e in ranked:
        if used + cost > cap:
            break
        pin.append([int(l), int(e)])
        used += cost
    with open(args.pin_out, "w") as f:
        json.dump({"pin": pin}, f)
    total_sel = sum(freq[l][e] for l in freq for e in freq[l])
    covered = sum(_cnt for _cnt, _l, _e in ranked[:len(pin)])
    print(f"[analyze] pin {len(pin)} experts (~{used / 1e9:.1f} GB, cost "
          f"{cost / 1e6:.0f} MB/expert) covering {100 * covered / max(1, total_sel):.1f}% "
          f"of all selections -> {args.pin_out}")

    # ---- perm.json: per-layer co-activation ordering for the #3 repack ----
    # Must be a FULL permutation of all E experts — experts that never fired in
    # this short calibration are appended (in id order) so none are dropped.
    perm = {}
    for l in seen_experts:
        order = _greedy_order(seen_experts[l], coact[l])
        order += [e for e in range(num_experts) if e not in seen_experts[l]]
        perm[str(int(l))] = order
    with open(args.perm_out, "w") as f:
        json.dump({"perm": perm}, f)
    print(f"[analyze] co-activation ordering for {len(perm)} layers -> {args.perm_out}")

    # ---- #3 ceiling pre-check: contiguous read-runs per token, identity vs
    # co-activation order. Each run = one coalesced pread; fewer runs = the win.
    def _runs(positions):
        s = sorted(positions)
        r = 1
        for i in range(1, len(s)):
            if s[i] != s[i - 1] + 1:
                r += 1
        return r

    id_runs, perm_runs, ks = [], [], []
    for layer, experts in trace:
        experts = list(experts)
        if len(experts) < 2:
            continue
        ks.append(len(experts))
        id_runs.append(_runs(experts))
        pos = {e: i for i, e in enumerate(perm[str(int(layer))])}
        perm_runs.append(_runs([pos[e] for e in experts]))
    if id_runs:
        ak = sum(ks) / len(ks)
        ai = sum(id_runs) / len(id_runs)
        ap = sum(perm_runs) / len(perm_runs)
        print(f"[analyze] #3 ceiling: avg {ak:.1f} experts/token/layer -> "
              f"identity {ai:.2f} read-runs, co-activation {ap:.2f} read-runs "
              f"({100 * (1 - ap / ai):.0f}% fewer reads, ~{ai / ap:.2f}x larger each)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--budget", type=float, default=30.0)
    c.add_argument("--max-tokens", type=int, default=96)
    c.add_argument("--temp", type=float, default=0.7)
    c.add_argument("--out", default="/tmp/expert_trace.json")
    c.set_defaults(func=collect)

    a = sub.add_parser("analyze")
    a.add_argument("--model", required=True)
    a.add_argument("--trace", required=True)
    a.add_argument("--pin-gb", type=float, default=8.0)
    a.add_argument("--pin-out", default="/tmp/pin.json")
    a.add_argument("--perm-out", default="/tmp/perm.json")
    a.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
