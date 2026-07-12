"""Tests for asymmetric expert precision (--expert-down-bits).

DwarfStar-style expert mix: MoE up/gate projections take the mlp_bits /
ternary tier while the down projections (the SwiGLU summation bottleneck)
keep a higher-precision Gaussian codebook. The loader needs no matching
rule because per-layer bits are self-describing via the on-disk codebook
length — these tests pin both halves of that contract:

1. Config: validation + round-trip through the config.json quantization dict.
2. Converter: down_proj switch layers get a 2^bits codebook, up/gate get the
   3-entry trit codebook (or their own tier when ternary is off), and dense
   (non-expert) down_proj linears are untouched by the override.
3. Loader agreement: rebuilding layers from the saved weights reproduces the
   mixed tier (trit up/gate, codebook down) via codebook-length detection.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm.models.switch_layers import SwitchGLU

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from turboquant_mlx.quantize_model import turboquant_quantize


class TinyMoEBlock(nn.Module):
    def __init__(self, dims=64, hidden=128, num_experts=4):
        super().__init__()
        self.switch_mlp = SwitchGLU(dims, hidden, num_experts)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMoEBlock()
        # A dense down_proj outside the expert tier must not be affected.
        self.dense_mlp = nn.Module()
        self.dense_mlp.down_proj = nn.Linear(64, 64, bias=False)


def _quantize(**cfg_kwargs):
    model = TinyModel()
    mx.eval(model.parameters())
    tq_config = TurboQuantConfig(group_size=32, mlp_group_size=32,
                                 **cfg_kwargs)
    model, config = turboquant_quantize(model, {"model_type": "test"},
                                        tq_config)
    return model, config, tq_config


def test_config_validation_and_round_trip():
    with pytest.raises(ValueError):
        TurboQuantConfig(expert_down_bits=1)
    with pytest.raises(ValueError):
        TurboQuantConfig(expert_down_bits=8)

    cfg = TurboQuantConfig(ternary_experts=True, expert_down_bits=3)
    again = TurboQuantConfig.from_dict(cfg.to_dict())
    assert again.expert_down_bits == 3
    assert again.ternary_experts is True

    # Absent key (models converted before this feature) -> disabled.
    legacy = TurboQuantConfig.from_dict({"bits": 3})
    assert legacy.expert_down_bits is None


def test_ternary_up_gate_with_codebook_down():
    model, _, _ = _quantize(ternary_experts=True, expert_down_bits=3)

    gate = model.mlp.switch_mlp.gate_proj
    up = model.mlp.switch_mlp.up_proj
    down = model.mlp.switch_mlp.down_proj
    for layer in (gate, up, down):
        assert isinstance(layer, PolarQuantizedSwitchLinear)

    # up/gate: ternary -> self-describing 3-entry codebook
    assert gate.codebook.shape[-1] == 3
    assert up.codebook.shape[-1] == 3
    # down: 3-bit Gaussian codebook, not ternary
    assert down.codebook.shape[-1] == 8

    # dense (non-expert) down_proj is untouched by the expert override:
    # quantized as a plain linear at the base width, never at expert_down_bits
    dense = model.dense_mlp.down_proj
    assert not isinstance(dense, PolarQuantizedSwitchLinear)


def test_down_override_without_ternary():
    model, _, _ = _quantize(bits=2, expert_down_bits=4)
    assert model.mlp.switch_mlp.gate_proj.codebook.shape[-1] == 4   # 2-bit
    assert model.mlp.switch_mlp.down_proj.codebook.shape[-1] == 16  # 4-bit


def test_disabled_is_uniform():
    model, _, _ = _quantize(ternary_experts=True)
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(model.mlp.switch_mlp, name).codebook.shape[-1] == 3


def test_loader_rebuilds_mixed_tier_from_codebook_length():
    from mlx.utils import tree_flatten

    model, _, tq_config = _quantize(ternary_experts=True, expert_down_bits=3)
    weights = dict(tree_flatten(model.parameters()))

    # Simulate a fresh load: a float model rebuilt into quantized layers
    # purely from the saved weights (codebook length = the format marker).
    from turboquant_mlx.generate import _prepare_polar_layers

    fresh = TinyModel()
    mx.eval(fresh.parameters())
    fresh = _prepare_polar_layers(fresh, weights, tq_config)

    gate = fresh.mlp.switch_mlp.gate_proj
    down = fresh.mlp.switch_mlp.down_proj
    assert isinstance(gate, PolarQuantizedSwitchLinear)
    assert isinstance(down, PolarQuantizedSwitchLinear)
    assert gate.trit and gate.bits == 2
    assert not down.trit and down.bits == 3
