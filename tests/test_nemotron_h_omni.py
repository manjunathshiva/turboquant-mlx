# Copyright 2026 Manjunath Janardhan
"""Nemotron 3.5 Super VL (``nemotron_h_omni``) conversion rules.

mlx-vlm (>= 0.7.9) maps ``nemotron_h_omni`` onto its ``nemotron_h_nano_omni``
model, so this package only has to get the quantization choices right:

- the vision projector ``mlp1`` and ``lm_head`` stay out of the polar path;
- ``--expert-down-bits`` must find the down projection of Nemotron-H's
  ``SwitchMLP``, which is named ``fc2`` (``fc1`` is the up projection);
- ``convert_vlm`` exposes ``--ternary-experts`` and ``--expert-down-bits``.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchMLP

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from turboquant_mlx.quantize_model import _is_expert_down_proj, turboquant_quantize

_L0 = "language_model.backbone.layers.0.mixer"


def test_skip_patterns_keep_lm_head_and_projector_out_of_the_polar_path():
    pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")
    from turboquant_mlx.integration.vlm import vlm_should_quantize

    predicate = vlm_should_quantize("nemotron_h_omni", lambda p, m: True)
    assert predicate("language_model.lm_head", None) is False
    assert predicate("mlp1.layers.1", None) is False
    assert predicate("mlp1.layers.3", None) is False
    assert predicate("vision_model.radio_model.model.blocks.0.attn.qkv", None) is False
    # the language model still goes through TurboQuant
    for name in ("q_proj", "in_proj", "out_proj", "fc1_latent_proj",
                 "shared_experts.up_proj", "switch_mlp.fc1"):
        assert predicate(f"{_L0}.{name}", None) is True, name


def test_down_projection_is_recognised_under_either_naming():
    assert _is_expert_down_proj("model.layers.3.mlp.switch_mlp.down_proj")
    assert _is_expert_down_proj(f"{_L0}.switch_mlp.fc2")
    assert not _is_expert_down_proj(f"{_L0}.switch_mlp.fc1")
    assert not _is_expert_down_proj("model.layers.3.mlp.switch_mlp.up_proj")
    # an fc2 that is not a SwitchMLP's (e.g. a vision block's MLP) is not one
    assert not _is_expert_down_proj("vision_model.blocks.0.mlp.fc2")


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch_mlp = SwitchMLP(64, 128, 4)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer(), _Layer()]


def test_expert_down_bits_reaches_switch_mlp_fc2():
    mx.random.seed(0)
    model = _Tiny()
    mx.eval(model.parameters())
    tq = TurboQuantConfig(bits=3, group_size=32, ternary_experts=True,
                          expert_down_bits=4)
    model, _ = turboquant_quantize(model, {"model_type": "test"}, tq)
    for layer in model.layers:
        up, down = layer.mixer.switch_mlp.fc1, layer.mixer.switch_mlp.fc2
        assert isinstance(up, PolarQuantizedSwitchLinear)
        assert isinstance(down, PolarQuantizedSwitchLinear)
        assert up.trit, "fc1 keeps the ternary tier"
        assert not down.trit and down.bits == 4, "fc2 takes --expert-down-bits"


def test_convert_vlm_cli_passes_the_expert_tier_flags(monkeypatch):
    pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")
    import sys

    import turboquant_mlx.convert_vlm as cv

    seen = {}
    monkeypatch.setattr(cv, "convert_vlm", lambda **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", [
        "convert_vlm", "--hf-path", "src", "--mlx-path", "out",
        "--ternary-experts", "--expert-down-bits", "4", "-g", "64"])
    cv.main()
    assert seen["ternary_experts"] is True
    assert seen["expert_down_bits"] == 4
    assert seen["group_size"] == 64
