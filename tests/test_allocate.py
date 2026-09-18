"""Per-layer expert allocation: the error measurement and the greedy.

The measurement must use the converter's own quantizer (so it reports the error the
build will actually have), and the greedy must be exact about the budget and
deterministic. Neither needs a real model.
"""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.allocate import (
    TIERS,
    TierCost,
    allocate,
    expert_tier_cost,
    load_costs,
    save_costs,
    total_bytes,
    total_error,
)
from turboquant_mlx.core.polar_quantize import polar_dequantize_weight, polar_quantize_weight
from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight


def _experts(n=5, out=64, inp=128, seed=0):
    return mx.random.normal((n, out, inp), key=mx.random.key(seed)).astype(mx.bfloat16)


def test_error_falls_and_bytes_rise_with_the_tier():
    w = _experts()
    costs = [expert_tier_cost(w, t, group_size=64, seed=7) for t in TIERS]
    errs = [c.err / c.ref for c in costs]
    sizes = [c.nbytes for c in costs]
    assert errs == sorted(errs, reverse=True)
    assert sizes == sorted(sizes)
    # a 4-bit Lloyd-Max codebook on Gaussianized weights: ~1% relative error
    assert errs[-1] < 0.02 and errs[0] > errs[-1] * 5


def test_measurement_matches_quantizing_experts_one_by_one():
    """The converter quantizes one expert at a time; batching them must not change
    the error, because rows quantize independently and the signs are per layer."""
    w = _experts(n=3)
    got = expert_tier_cost(w, "3", group_size=64, seed=11)
    signs = generate_random_signs(128, seed=11).astype(mx.float32)
    err = ref = 0.0
    nbytes = 0
    for e in range(3):
        we = w[e].astype(mx.float32)
        q = polar_quantize_weight(we, bits=3, group_size=64, seed=11)
        deq = polar_dequantize_weight(q["packed_weight"], q["scales"], q["codebook"],
                                      3, 64, 128).astype(mx.float32)
        t = rotate_weight(we, signs)
        err += mx.sum((t - deq) ** 2).item()
        ref += mx.sum(t * t).item()
        nbytes += q["packed_weight"].nbytes + q["scales"].nbytes
    assert got.nbytes == nbytes
    assert got.err == pytest.approx(err, rel=1e-5)
    assert got.ref == pytest.approx(ref, rel=1e-5)


def test_rotation_does_not_change_the_measured_error_scale():
    """Rotation is orthonormal: the weights' energy is the same in both domains."""
    w = _experts(n=2)
    rot = expert_tier_cost(w, "2", group_size=64, seed=3, rotate=True)
    plain = mx.sum(w.astype(mx.float32) ** 2).item()
    assert rot.ref == pytest.approx(plain, rel=1e-4)


def _costs(table):
    """table: {layer: [(err, bytes) per tier, cheapest first]}"""
    return {layer: {t: TierCost(err=e, ref=100.0, nbytes=b)
                    for t, (e, b) in zip(TIERS, rows)}
            for layer, rows in table.items()}


def test_greedy_buys_the_most_error_per_byte_first():
    costs = _costs({
        0: [(10, 100), (4, 150), (2, 200), (1, 250)],   # first upgrade: 6 per 50 bytes
        1: [(10, 100), (9, 150), (8, 200), (7, 250)],   # 1 per 50 bytes
    })
    choice = allocate(costs, budget_bytes=250)
    assert choice == {0: "2", 1: "ternary"}
    assert total_bytes(costs, choice) == 250


def test_budget_is_never_exceeded_and_is_filled_greedily():
    rng = np.random.default_rng(0)
    table = {}
    for layer in range(12):
        e = sorted(rng.uniform(1, 20, 4), reverse=True)
        b = sorted(rng.integers(100, 400, 4))
        table[layer] = list(zip(e, b))
    costs = _costs(table)
    cheapest = total_bytes(costs, {layer: "ternary" for layer in table})
    for budget in (cheapest, cheapest + 300, cheapest + 1500, 10**9):
        choice = allocate(costs, budget)
        assert total_bytes(costs, choice) <= budget
        assert allocate(costs, budget) == choice          # deterministic
    top = allocate(costs, 10**9)
    assert all(t == "4" for t in top.values())


def test_more_budget_never_increases_error():
    rng = np.random.default_rng(1)
    table = {layer: list(zip(sorted(rng.uniform(1, 20, 4), reverse=True),
                             sorted(rng.integers(100, 400, 4))))
             for layer in range(8)}
    costs = _costs(table)
    base = total_bytes(costs, {layer: "ternary" for layer in table})
    errs = [total_error(costs, allocate(costs, base + extra))
            for extra in range(0, 2400, 150)]
    assert errs == sorted(errs, reverse=True)


def test_a_budget_below_the_cheapest_tier_is_an_error():
    costs = _costs({0: [(10, 100), (4, 150), (2, 200), (1, 250)]})
    with pytest.raises(ValueError, match="cheapest tier"):
        allocate(costs, budget_bytes=99)


def test_costs_round_trip_through_json(tmp_path):
    costs = _costs({0: [(10, 100), (4, 150), (2, 200), (1, 250)],
                    5: [(9, 100), (3, 150), (2, 200), (1, 250)]})
    p = tmp_path / "costs.json"
    save_costs(p, costs)
    assert load_costs(p) == costs
