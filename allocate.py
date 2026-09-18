# Copyright 2026 Manjunath Janardhan
"""Data-free per-layer allocation of expert bit widths.

Given a byte budget for the routed experts, choose a tier per MoE layer (ternary,
2-, 3- or 4-bit codebook) so the total quantization error is as small as the budget
allows. It needs no calibration data: each layer's error is measured by
quantizing its own weights with the same ``polar_quantize_weight`` call the
converter uses and comparing against the weights themselves.

The allocation is the standard greedy for this problem: start every layer at the
cheapest tier, then repeatedly upgrade whichever layer removes the most error per
byte added, until the budget is spent.

Reconstruction error is not output error. Calibrated allocators (mlx-serve's
iQ-MLX, llama.cpp's imatrix) weight it by activation statistics; this one can't,
by design. Whether it still beats uniform bits is an experiment, not an assumption
(``paper/expert_allocation_plan.md``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from turboquant_mlx.core.polar_quantize import polar_dequantize_weight, polar_quantize_weight
from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight

# name -> (storage bits, ternary). Ordered cheapest first.
TIERS: dict[str, tuple[int, bool]] = {
    "ternary": (2, True),
    "2": (2, False),
    "3": (3, False),
    "4": (4, False),
}

# Experts quantized per call. Rows quantize independently and the rotation signs
# are per layer, so batching experts is bit-identical to the converter's
# one-expert-at-a-time loop; this only bounds memory.
_EXPERT_CHUNK = 32


@dataclass
class TierCost:
    err: float     # sum of squared reconstruction error, rotated domain
    ref: float     # sum of squared weights, same domain
    nbytes: int    # packed weights + scales, as stored


def expert_tier_cost(weight_3d: mx.array, tier: str, group_size: int, seed: int,
                     rotate: bool = True) -> TierCost:
    """Quantize a layer's experts ``(E, out, in)`` at ``tier`` and measure the error.

    The Hadamard rotation is orthonormal, so the squared error is the same in the
    rotated and the original domain; it is computed in the rotated one, where the
    dequantized weights live.
    """
    bits, ternary = TIERS[tier]
    n_exp, out_dims, in_dims = weight_3d.shape
    signs = (generate_random_signs(in_dims, seed=seed).astype(mx.float32)
             if rotate else None)
    err = ref = 0.0
    nbytes = 0
    for e0 in range(0, n_exp, _EXPERT_CHUNK):
        w = weight_3d[e0:e0 + _EXPERT_CHUNK].reshape(-1, in_dims).astype(mx.float32)
        q = polar_quantize_weight(w, bits=bits, group_size=group_size, seed=seed,
                                  ternary=ternary, rotate=rotate)
        deq = polar_dequantize_weight(q["packed_weight"], q["scales"], q["codebook"],
                                      bits, group_size, in_dims,
                                      trit=ternary).astype(mx.float32)
        target = rotate_weight(w, signs) if rotate else w
        d = target - deq
        e_sum, r_sum = mx.sum(d * d), mx.sum(target * target)
        mx.eval(e_sum, r_sum)
        err += e_sum.item()
        ref += r_sum.item()
        nbytes += q["packed_weight"].nbytes + q["scales"].nbytes
        del w, q, deq, target, d
        mx.clear_cache()
    return TierCost(err=err, ref=ref, nbytes=nbytes)


def allocate(costs: dict[int, dict[str, TierCost]], budget_bytes: int,
             tiers: list[str] | None = None) -> dict[int, str]:
    """Greedy tier choice per layer under ``budget_bytes`` for all experts.

    ``costs[layer][tier]`` must cover every tier in ``tiers`` (default: all, cheapest
    first). Every layer starts at the cheapest tier; the step that removes the most
    error per extra byte is taken while it fits. Ties break toward the lower layer
    index, so the result is deterministic.

    Raises ``ValueError`` if even the cheapest tier everywhere exceeds the budget.
    """
    tiers = tiers or list(TIERS)
    layers = sorted(costs)
    choice = {layer: tiers[0] for layer in layers}
    spent = sum(costs[layer][tiers[0]].nbytes for layer in layers)
    if spent > budget_bytes:
        raise ValueError(f"the cheapest tier everywhere needs {spent} bytes, over the "
                         f"budget of {budget_bytes}")
    while True:
        best = None
        for layer in layers:
            i = tiers.index(choice[layer])
            if i + 1 >= len(tiers):
                continue
            cur, nxt = costs[layer][tiers[i]], costs[layer][tiers[i + 1]]
            extra = nxt.nbytes - cur.nbytes
            if extra <= 0 or spent + extra > budget_bytes:
                continue
            gain = (cur.err - nxt.err) / extra
            if best is None or gain > best[0]:
                best = (gain, layer, extra, tiers[i + 1])
        if best is None or best[0] <= 0:
            return choice
        _, layer, extra, tier = best
        choice[layer] = tier
        spent += extra


def total_bytes(costs: dict[int, dict[str, TierCost]], choice: dict[int, str]) -> int:
    return sum(costs[layer][tier].nbytes for layer, tier in choice.items())


def total_error(costs: dict[int, dict[str, TierCost]], choice: dict[int, str]) -> float:
    """Summed squared error relative to the summed squared weights, over all layers."""
    err = sum(costs[layer][tier].err for layer, tier in choice.items())
    ref = sum(costs[layer][tier].ref for layer, tier in choice.items())
    return err / ref if ref else 0.0


def save_costs(path: str | os.PathLike, costs: dict[int, dict[str, TierCost]]) -> None:
    data = {str(layer): {t: vars(c) for t, c in per.items()} for layer, per in costs.items()}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def load_costs(path: str | os.PathLike) -> dict[int, dict[str, TierCost]]:
    data = json.loads(Path(path).read_text())
    return {int(layer): {t: TierCost(**c) for t, c in per.items()}
            for layer, per in data.items()}


def measure_model(model, out_path: str | os.PathLike, group_size: int,
                  rotation_seed: int, rotate: bool = True,
                  tiers: list[str] | None = None, log=print) -> dict[int, dict[str, TierCost]]:
    """Measure every routed-expert layer of a lazily loaded model at every tier.

    A layer's cost is the sum over its switch projections (gate, up, down), since a
    tier applies to the whole layer's experts, as layer protection does. Results are
    written to ``out_path`` after each layer, and layers already there are skipped,
    so a killed run resumes where it stopped.
    """
    import re
    import time

    from turboquant_mlx.quantize_model import _get_layer_seed, _is_switch_linear

    tiers = tiers or list(TIERS)
    costs = load_costs(out_path) if Path(out_path).exists() else {}
    layer_rx = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    by_layer: dict[int, list] = {}
    for path, module in model.named_modules():
        if _is_switch_linear(module) and "weight" in module:
            m = layer_rx.search(path)
            if m:
                by_layer.setdefault(int(m.group(1)), []).append((path, module))
    for layer in sorted(by_layer):
        if layer in costs and all(t in costs[layer] for t in tiers):
            continue
        t0 = time.time()
        per = {t: TierCost(0.0, 0.0, 0) for t in tiers}
        for path, module in by_layer[layer]:
            seed = _get_layer_seed(rotation_seed, path)
            for t in tiers:
                c = expert_tier_cost(module.weight, t, group_size, seed, rotate)
                per[t] = TierCost(per[t].err + c.err, per[t].ref + c.ref,
                                  per[t].nbytes + c.nbytes)
        costs[layer] = per
        save_costs(out_path, costs)
        rel = "  ".join(f"{t}:{per[t].err / per[t].ref:.4f}" for t in tiers)
        log(f"[allocate] layer {layer}: {rel}  ({time.time() - t0:.0f}s)")
    return costs


def main(argv=None) -> int:
    """``measure`` a source model's per-layer expert costs, then ``choose`` tiers.

        python -m turboquant_mlx.allocate measure --model SRC --out costs.json --group-size 64
        python -m turboquant_mlx.allocate choose --costs costs.json --match-uniform 2 --out tiers.json
        python -m turboquant_mlx.convert ... --expert-layer-tiers tiers.json
    """
    import argparse

    ap = argparse.ArgumentParser(prog="python -m turboquant_mlx.allocate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="quantize each expert layer at every tier; record error and bytes")
    m.add_argument("--model", required=True, help="bf16 source (local dir or HF repo)")
    m.add_argument("--out", required=True, help="costs JSON (resumes if it exists)")
    m.add_argument("--group-size", type=int, default=64, help="expert group size, as --mlp-group-size")
    m.add_argument("--rotation-seed", type=int, default=42)
    m.add_argument("--no-rotation", action="store_true")
    c = sub.add_parser("choose", help="pick a tier per layer under a byte budget")
    c.add_argument("--costs", required=True)
    c.add_argument("--out", required=True, help="tiers JSON for convert --expert-layer-tiers")
    b = c.add_mutually_exclusive_group(required=True)
    b.add_argument("--budget-gib", type=float, help="bytes for all routed experts, GiB")
    b.add_argument("--match-uniform", choices=list(TIERS),
                   help="use exactly the bytes of this tier applied uniformly")
    args = ap.parse_args(argv)

    if args.cmd == "measure":
        from mlx_lm.utils import load

        import turboquant_mlx.compat  # noqa: F401

        model, _ = load(args.model, lazy=True)
        measure_model(model, args.out, args.group_size, args.rotation_seed,
                      rotate=not args.no_rotation)
        return 0

    costs = load_costs(args.costs)
    uniform = {t: total_bytes(costs, {layer: t for layer in costs}) for t in TIERS}
    budget = (uniform[args.match_uniform] if args.match_uniform
              else int(args.budget_gib * 2**30))
    choice = allocate(costs, budget)
    counts = {t: sum(1 for v in choice.values() if v == t) for t in TIERS}
    summary = {
        "budget_bytes": budget,
        "expert_bytes": total_bytes(costs, choice),
        "relative_error": total_error(costs, choice),
        "uniform": {t: {"bytes": uniform[t],
                        "relative_error": total_error(costs, {layer: t for layer in costs})}
                    for t in TIERS},
        "counts": counts,
        "tiers": {str(k): v for k, v in sorted(choice.items())},
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(f"[allocate] {counts} | experts {summary['expert_bytes'] / 2**30:.2f} GiB "
          f"of {budget / 2**30:.2f} | relative error {summary['relative_error']:.5f}")
    for t in TIERS:
        u = summary["uniform"][t]
        print(f"[allocate]   uniform {t:>7}: {u['bytes'] / 2**30:.2f} GiB, "
              f"relative error {u['relative_error']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
