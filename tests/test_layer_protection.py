"""Layer protection for expert builds (--protect-expert-layers).

Listed layers' routed experts leave the ternary / mlp_bits tier for a
``protect_bits`` Gaussian codebook. Built for Qwen3.8-Flash-Next, where pure
ternary experts at 640 wide keep answers correct but stop the model ending its
own reasoning (0/5 thinking runs close ``</think>``; a tq3-expert control closes
5/5).

The loader has no matching rule, so these tests pin the contract that makes that
safe: protection changes bit width only, per-layer format is self-describing via
codebook length, and a freshly loaded model computes the same outputs as the
converted one. Selection is by module type plus ``.layers.N.`` index -- never by
expert-container name, which has silently missed three times in this codebase.
"""

import argparse

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from turboquant_mlx.quantize_model import turboquant_quantize


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch_mlp = SwitchGLU(64, 128, 4)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoE()


class _Inner(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = [_Layer() for _ in range(n)]


class TinyModel(nn.Module):
    """Paths look like the real thing: model.layers.N.mlp.switch_mlp.gate_proj."""

    def __init__(self, n=4):
        super().__init__()
        self.model = _Inner(n)


def _quantize(**kw):
    mx.random.seed(0)
    model = TinyModel()
    mx.eval(model.parameters())
    tq = TurboQuantConfig(group_size=32, mlp_group_size=32, **kw)
    model, config = turboquant_quantize(model, {"model_type": "test"}, tq)
    return model, config, tq


def _cb(model, i, proj):
    return getattr(model.model.layers[i].mlp.switch_mlp, proj).codebook.shape[-1]


def test_config_validation_normalization_and_round_trip():
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_bits=1)
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_bits=8)
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_expert_layers=[-1])
    with pytest.raises(ValueError):
        TurboQuantConfig(protect_expert_layers="abc")

    cfg = TurboQuantConfig(ternary_experts=True, protect_expert_layers=[5, 0, 5])
    assert cfg.protect_expert_layers == (0, 5)
    again = TurboQuantConfig.from_dict(cfg.to_dict())
    assert again.protect_expert_layers == (0, 5) and again.protect_bits == 3

    assert TurboQuantConfig(protect_expert_layers=[]).protect_expert_layers is None


def test_to_dict_unchanged_when_protection_is_off():
    d = TurboQuantConfig(ternary_experts=True).to_dict()
    assert "protected_expert_layers" not in d and "protect_bits" not in d


def test_reads_the_keys_convert_vlm_already_writes():
    """Builds that shipped with VLM layer protection must keep loading."""
    legacy = TurboQuantConfig.from_dict(
        {"bits": 2, "protected_expert_layers": [0, 1, 28, 29], "protect_bits": 4})
    assert legacy.protect_expert_layers == (0, 1, 28, 29)
    assert legacy.protect_bits == 4
    assert TurboQuantConfig.from_dict({"bits": 3}).protect_expert_layers is None


def test_protected_layers_leave_the_ternary_tier():
    model, config, _ = _quantize(ternary_experts=True, protect_expert_layers=(0, 3))
    for i in (0, 3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert _cb(model, i, proj) == 8, (i, proj)  # 3-bit Gaussian codebook
    for i in (1, 2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert _cb(model, i, proj) == 3, (i, proj)  # ternary trit codebook
    assert config["quantization"]["protected_expert_layers"] == [0, 3]
    assert config["quantization"]["protect_bits"] == 3


def test_protection_off_is_uniform_ternary():
    model, config, _ = _quantize(ternary_experts=True)
    for i in range(4):
        assert _cb(model, i, "gate_proj") == 3
    assert "protected_expert_layers" not in config["quantization"]


def test_protection_composes_with_expert_down_bits_taking_the_higher_width():
    model, _, _ = _quantize(ternary_experts=True, expert_down_bits=4,
                            protect_expert_layers=(0,), protect_bits=3)
    assert _cb(model, 0, "gate_proj") == 8    # protected: 3-bit
    assert _cb(model, 0, "down_proj") == 16   # max(protect 3, down 4) = 4-bit
    assert _cb(model, 1, "gate_proj") == 3    # unprotected: ternary
    assert _cb(model, 1, "down_proj") == 16   # unprotected down: 4-bit


def test_unmatched_protected_layer_is_never_silent(capsys):
    _quantize(ternary_experts=True, protect_expert_layers=(1, 99))
    out = capsys.readouterr().out
    assert "[WARNING]" in out and "99" in out and "NOT protected" in out


def test_fresh_load_matches_converted_outputs():
    """The real contract: convert and load agree, with no loader rule for it."""
    from mlx.utils import tree_flatten

    from turboquant_mlx.generate import _prepare_polar_layers

    model, _, tq = _quantize(ternary_experts=True, protect_expert_layers=(0, 3))
    weights = dict(tree_flatten(model.parameters()))

    fresh = TinyModel()
    mx.eval(fresh.parameters())
    fresh = _prepare_polar_layers(fresh, weights, tq)
    fresh.load_weights(list(weights.items()), strict=False)

    for i, trit in ((0, False), (1, True), (3, False)):
        layer = fresh.model.layers[i].mlp.switch_mlp.gate_proj
        assert isinstance(layer, PolarQuantizedSwitchLinear)
        assert bool(layer.trit) is trit, i

    x = mx.random.normal((1, 5, 64))
    idx = mx.array([[[0, 2]] * 5])
    for i in range(4):
        a = model.model.layers[i].mlp.switch_mlp(x, idx)
        b = fresh.model.layers[i].mlp.switch_mlp(x, idx)
        assert mx.allclose(a, b, atol=1e-5), i


def test_cli_layer_list_parsing():
    from turboquant_mlx.convert import _parse_layer_list

    assert _parse_layer_list("0-5,42-47") == [0, 1, 2, 3, 4, 5, 42, 43, 44, 45, 46, 47]
    assert _parse_layer_list("3, 1,3") == [1, 3]
    for bad in ("5-2", "a", "-1", "", "1-x"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_layer_list(bad)
