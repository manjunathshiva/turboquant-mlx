"""Tests for the auto cache budget + wired-memory flags (ds4 idea #5).

The budget math is a pure function, so it tests without a model; the CLI
plumbing is tested through serve's argv extractor and stream_generate's
argparse type.
"""

import argparse

import pytest

from turboquant_mlx.stream.loader import (
    _AUTO_MIN_BUDGET_BYTES,
    _AUTO_RESERVE_BYTES,
    _AUTO_WSS_FRACTION,
    _auto_cache_budget,
    _streamed_expert_bytes,
)


def test_auto_budget_mini_case():
    # 16 GB mini serving the 122B ternary: model ~31 GB of which ~28 GB is
    # experts, working set ~11.5 GB. Expected: 0.8*11.5 - 3 - 2 = ~4.2 GB —
    # matching the hand-tuned known-good --cache-budget-gb 4.
    b = _auto_cache_budget(model_bytes=31_000_000_000,
                           expert_bytes=28_000_000_000,
                           wss_bytes=11_500_000_000)
    assert b == int(0.8 * 11_500_000_000) - 3_000_000_000 - _AUTO_RESERVE_BYTES
    assert 3.5e9 < b < 5e9


def test_auto_budget_clamps_to_all_experts_on_roomy_machine():
    # 64 GB box streaming a 9.4 GB model: the working-set slice dwarfs the
    # experts, so the budget caps at "every expert resident".
    b = _auto_cache_budget(model_bytes=9_400_000_000,
                           expert_bytes=8_600_000_000,
                           wss_bytes=55_000_000_000)
    assert b == 8_600_000_000


def test_auto_budget_floor_when_squeezed():
    # Pathologically tight machine: never go below the floor (the LRU needs
    # some working room to make progress at all).
    b = _auto_cache_budget(model_bytes=31_000_000_000,
                           expert_bytes=28_000_000_000,
                           wss_bytes=6_000_000_000)
    assert b == _AUTO_MIN_BUDGET_BYTES


def test_streamed_expert_bytes_counts_only_switch_tensors():
    class Loc:
        def __init__(self, dtype, shape):
            self.dtype = dtype
            self.shape = shape

    class R:
        _index = {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": Loc("U32", (4, 8, 2)),
            "model.layers.0.mlp.switch_mlp.gate_proj.scales": Loc("F16", (4, 8, 1)),
            "model.layers.0.mlp.switch_mlp.gate_proj.codebook": Loc("F16", (3,)),
            "model.layers.0.self_attn.q_proj.weight": Loc("U32", (64, 64)),
        }

    assert _streamed_expert_bytes(R()) == 4 * 8 * 2 * 4 + 4 * 8 * 1 * 2


def test_serve_extractor_accepts_auto_and_wire_memory():
    from turboquant_mlx.serve import _extract_stream_args

    cfg, rem = _extract_stream_args(
        ["--cache-budget-gb", "auto", "--wire-memory", "--model", "m"])
    assert cfg["cache_budget_gb"] == "auto"
    assert cfg["wire_memory"] is True
    assert rem == ["--model", "m"]
    cfg, _ = _extract_stream_args(["--cache-budget-gb", "4"])
    assert cfg["cache_budget_gb"] == 4.0  # numerics parsed in the extractor
    assert cfg["wire_memory"] is False
    cfg, _ = _extract_stream_args(["--model", "m"])
    assert cfg is None  # no budget -> resident loader
    with pytest.raises(SystemExit):
        _extract_stream_args(["--cache-budget-gb", "lots"])


def test_stream_generate_budget_arg_type():
    from turboquant_mlx.stream.stream_generate import _budget_arg

    assert _budget_arg("auto") == "auto"
    assert _budget_arg("AUTO") == "auto"
    assert _budget_arg("3.5") == 3.5
    with pytest.raises(argparse.ArgumentTypeError):
        _budget_arg("lots")


def test_wss_fraction_sane():
    assert 0.5 <= _AUTO_WSS_FRACTION <= 0.9


def test_auto_budget_experts_smaller_than_floor_cap_at_experts():
    # A tiny MoE whose whole expert tier is under the 0.5 GB floor: the
    # ceiling (all experts) must win over the floor — a budget larger than
    # every expert buys nothing.
    b = _auto_cache_budget(model_bytes=10_000_000_000,
                           expert_bytes=200_000_000,
                           wss_bytes=16_000_000_000)
    assert b == 200_000_000
