"""Tests for the auto cache budget + wired-memory flags (ds4 idea #5).

The budget math is a pure function, so it tests without a model; the CLI
plumbing is tested through serve's argv extractor and stream_generate's
argparse type.
"""

import argparse

import pytest

from turboquant_mlx.cache_budget import MIN_BUDGET_BYTES as _AUTO_MIN_BUDGET_BYTES
from turboquant_mlx.cache_budget import projected_peak_bytes
from turboquant_mlx.stream.loader import (
    _AUTO_RESERVE_BYTES,
    _auto_cache_budget,
    _streamed_expert_bytes,
)


def test_auto_budget_mini_case():
    # 16 GB mini serving the 122B ternary: model ~31 GB of which ~28 GB is
    # experts, working set ~11.5 GB.
    #
    # This used to assert 0.8*wss - resident - 2 GB = ~4.2 GB, chosen to match
    # a hand-tuned --cache-budget-gb 4. A later sweep on the same class of
    # machine showed that reserve was double-counted (a 20% haircut *and* a
    # 2 GB subtraction) and cost real throughput, so the budget is now derived
    # from the measured peak model instead: everything the working set has,
    # minus the resident weights, minus what a run actually needs on top.
    b = _auto_cache_budget(model_bytes=31_000_000_000,
                           expert_bytes=28_000_000_000,
                           wss_bytes=11_500_000_000)
    assert b == 11_500_000_000 - 3_000_000_000 - _AUTO_RESERVE_BYTES
    assert 5e9 < b < 6e9
    # and the thing that actually matters: it still fits
    assert projected_peak_bytes(b, 3_000_000_000) < 11_500_000_000


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


def test_reserve_sane():
    """The reserve replaced the working-set fraction. Too small and a streaming
    run peaks over the Metal cap; too large and it repeats the mistake it was
    written to fix."""
    assert 2.0e9 <= _AUTO_RESERVE_BYTES <= 4.0e9


def test_auto_budget_experts_smaller_than_floor_cap_at_experts():
    # A tiny MoE whose whole expert tier is under the 0.5 GB floor: the
    # ceiling (all experts) must win over the floor — a budget larger than
    # every expert buys nothing.
    b = _auto_cache_budget(model_bytes=10_000_000_000,
                           expert_bytes=200_000_000,
                           wss_bytes=16_000_000_000)
    assert b == 200_000_000


# --------------------------------------------------------------------------
# load_streaming's memory block, off and on a Metal device. MLX's CPU backend
# on Linux has no working-set key in device_info(); both `--cache-budget-gb
# auto` and `--wire-memory` indexed it and crashed. The model load is stubbed
# out and the run stops right after the memory block.
# --------------------------------------------------------------------------

_CPU_INFO = {"architecture": "x86_64", "device_name": "cpu"}
_GIB = float(1 << 30)
_RESIDENT, _EXPERTS = 3_000_000_000, 20_000_000_000


class _Stop(Exception):
    pass


@pytest.fixture
def stub_load(monkeypatch, tmp_path):
    import mlx.core as mx

    from turboquant_mlx import cache_budget
    from turboquant_mlx.stream import loader
    from turboquant_mlx.stream.usage_profile import UsageProfile

    monkeypatch.setattr(loader, "resolve_model_path", lambda p: tmp_path)
    monkeypatch.setattr(loader, "load_turboquant",
                        lambda *a, **k: (object(), object()))
    monkeypatch.setattr(loader, "SafetensorsExpertReader",
                        lambda *a, **k: object())
    monkeypatch.setattr(loader, "_streamed_expert_bytes", lambda r: _EXPERTS)
    monkeypatch.setattr(loader, "_model_file_bytes",
                        lambda p: _RESIDENT + _EXPERTS)
    monkeypatch.setattr(loader, "offloaded_ngram_bytes", lambda m: 0)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(UsageProfile, "load", stop)
    wired = []
    monkeypatch.setattr(mx, "set_wired_limit", wired.append)

    def run(info, ram, **kw):
        monkeypatch.setattr(mx, "device_info", lambda: dict(info))
        monkeypatch.setattr(cache_budget, "physical_ram_bytes", lambda: ram)
        with pytest.raises(_Stop):
            loader.load_streaming("m", use_page_cache=False,
                                  usage_file=str(tmp_path / "u.json"), **kw)
        return wired
    return run


def test_auto_budget_without_metal_estimates_from_ram(stub_load, capsys):
    from turboquant_mlx.cache_budget import estimate_wss

    stub_load(_CPU_INFO, 16 * _GIB, cache_budget_gb="auto")
    wss = int(estimate_wss(16 * _GIB))
    budget = _auto_cache_budget(_RESIDENT + _EXPERTS, _EXPERTS, wss)
    out = capsys.readouterr().out
    assert (f"working set {wss / 1e9:.1f} GB estimated from RAM "
            "(no Metal device) − resident 3.0 GB") in out
    assert f"-> cache {budget / 1e9:.1f} GB" in out


def test_auto_budget_without_metal_or_ram_says_what_to_pass(stub_load):
    with pytest.raises(RuntimeError, match=r"--cache-budget-gb <GB>"):
        stub_load(_CPU_INFO, None, cache_budget_gb="auto")


def test_auto_budget_on_metal_log_is_unchanged(stub_load, capsys):
    wss = 12_710_000_000
    stub_load({**_CPU_INFO, "max_recommended_working_set_size": wss}, None,
              cache_budget_gb="auto")
    budget = _auto_cache_budget(_RESIDENT + _EXPERTS, _EXPERTS, wss)
    assert capsys.readouterr().out == (
        f"[stream] auto budget: working set 12.7 GB − resident 3.0 GB − "
        f"reserve {_AUTO_RESERVE_BYTES / 1e9:.1f} GB -> cache "
        f"{budget / 1e9:.1f} GB (experts on disk: 20.0 GB, projected peak "
        f"{projected_peak_bytes(budget, _RESIDENT) / 1e9:.1f} GB)\n")


def test_wire_memory_without_metal_skips(stub_load, capsys):
    # RAM is known, but there is no Metal residency to wire on the CPU backend
    wired = stub_load(_CPU_INFO, 8 * _GIB, cache_budget_gb=4.0,
                      wire_memory=True)
    assert wired == []
    assert "no Metal device — skipping --wire-memory; continuing unwired" \
        in capsys.readouterr().out


def test_wire_memory_without_metal_or_ram_continues_unwired(stub_load, capsys):
    wired = stub_load(_CPU_INFO, None, cache_budget_gb=4.0, wire_memory=True)
    assert wired == []
    assert "no Metal device — skipping --wire-memory; continuing unwired" \
        in capsys.readouterr().out


def test_wire_memory_on_metal_is_unchanged(stub_load, capsys):
    wss = 55_000_000_000
    wired = stub_load({**_CPU_INFO, "max_recommended_working_set_size": wss},
                      None, cache_budget_gb=4.0, wire_memory=True)
    want = int(_RESIDENT + 4.0e9 + _AUTO_RESERVE_BYTES)
    assert wired == [want]
    assert capsys.readouterr().out == (
        f"[stream] wired-memory limit {want / 1e9:.1f} GB — weights and "
        "expert cache stay resident under memory pressure\n")
