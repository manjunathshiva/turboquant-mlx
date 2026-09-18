"""Per-layer expert tiers (``expert_layer_tiers`` / ``convert --expert-layer-tiers``).

The same contract as layer protection, which is its one-tier special case: tiers
change bit width only, each layer's format is self-describing through its codebook
length, and a freshly loaded model computes what the converted one did.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from tests.test_layer_protection import TinyModel, _cb, _quantize

_TIERS = {0: "4", 1: "ternary", 3: "3"}          # layer 2 keeps the expert tier


def test_each_layer_gets_its_tier():
    model, config, _ = _quantize(mlp_bits=2, expert_layer_tiers=_TIERS)
    want = {0: 16, 1: 3, 2: 4, 3: 8}              # codebook sizes: 4-bit, trit, 2-bit, 3-bit
    for i, n in want.items():
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert _cb(model, i, proj) == n, (i, proj)
    assert config["quantization"]["expert_layer_tiers"] == {"0": "4", "1": "ternary", "3": "3"}


def test_config_round_trips_and_validates():
    tq = TurboQuantConfig(expert_layer_tiers={"3": "3", 0: "4"})
    assert tq.expert_layer_tiers == {0: "4", 3: "3"}
    back = TurboQuantConfig.from_dict(tq.to_dict())
    assert back.expert_layer_tiers == tq.expert_layer_tiers
    assert TurboQuantConfig.from_dict(TurboQuantConfig().to_dict()).expert_layer_tiers is None
    assert "expert_layer_tiers" not in TurboQuantConfig().to_dict()
    with pytest.raises(ValueError, match="invalid entries"):
        TurboQuantConfig(expert_layer_tiers={0: "5"})
    with pytest.raises(ValueError, match="mutually exclusive"):
        TurboQuantConfig(expert_layer_tiers={0: "3"}, protect_expert_layers=(1,))
    with pytest.raises(ValueError, match="expert_down_bits"):
        TurboQuantConfig(expert_layer_tiers={0: "3"}, expert_down_bits=4)


def test_a_tier_for_a_layer_without_experts_fails_before_any_layer_is_quantized():
    """Checked up front: a streaming convert must not write anything first."""
    written = []
    with pytest.raises(ValueError, match=r"\[9\]"):
        mx.random.seed(0)
        model = TinyModel()
        mx.eval(model.parameters())
        from turboquant_mlx.quantize_model import turboquant_quantize

        turboquant_quantize(model, {"model_type": "test"},
                            TurboQuantConfig(group_size=32, mlp_group_size=32, mlp_bits=2,
                                             expert_layer_tiers={0: "3", 9: "4"}),
                            on_quantized=lambda path, module: written.append(path))
    assert written == []


def test_fresh_load_matches_converted_outputs():
    from turboquant_mlx.generate import _prepare_polar_layers

    model, _, tq = _quantize(mlp_bits=2, expert_layer_tiers=_TIERS)
    weights = dict(tree_flatten(model.parameters()))
    fresh = TinyModel()
    mx.eval(fresh.parameters())
    fresh = _prepare_polar_layers(fresh, weights, tq)
    fresh.load_weights(list(weights.items()), strict=False)

    assert bool(fresh.model.layers[1].mlp.switch_mlp.gate_proj.trit)
    assert not bool(fresh.model.layers[0].mlp.switch_mlp.gate_proj.trit)
    x = mx.random.normal((1, 5, 64))
    idx = mx.array([[[0, 2]] * 5])
    for i in range(4):
        assert isinstance(fresh.model.layers[i].mlp.switch_mlp.gate_proj,
                          PolarQuantizedSwitchLinear)
        a = model.model.layers[i].mlp.switch_mlp(x, idx)
        b = fresh.model.layers[i].mlp.switch_mlp(x, idx)
        assert mx.allclose(a, b, atol=1e-5), i


def test_cli_reads_a_plain_map_or_a_tiers_block(tmp_path):
    import json

    from turboquant_mlx.convert import _load_tiers

    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"0": "4", "3": "ternary"}))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"counts": {}, "tiers": {"0": "4", "3": "ternary"}}))
    for p in (plain, wrapped):
        tq = TurboQuantConfig(expert_layer_tiers=_load_tiers(str(p)))
        assert tq.expert_layer_tiers == {0: "4", 3: "ternary"}


@pytest.mark.parametrize("content", ['[1, 2]', '{"tiers": [1]}', '{}', '{"0": "5"}', 'not json'])
def test_cli_rejects_bad_tier_files_as_argument_errors(tmp_path, content):
    import argparse

    from turboquant_mlx.convert import _load_tiers

    p = tmp_path / "bad.json"
    p.write_text(content)
    with pytest.raises(argparse.ArgumentTypeError):
        _load_tiers(str(p))


def test_cli_rejects_a_tier_file_that_is_not_text(tmp_path):
    import argparse

    from turboquant_mlx.convert import _load_tiers

    p = tmp_path / "binary.json"
    p.write_bytes(b"\xff\xfe\x00\x81 not utf-8")
    with pytest.raises(argparse.ArgumentTypeError):
        _load_tiers(str(p))


def test_a_tiered_layer_with_a_projection_that_would_be_skipped_fails_up_front():
    """SwitchGLU(64, 128, 4) at expert group size 128: gate/up (64 wide) would be
    skipped and stay at source precision while down (128 wide) takes the tier, so
    the config would claim a tier the layer only partly has."""
    from turboquant_mlx.quantize_model import turboquant_quantize

    mx.random.seed(0)
    model = TinyModel()
    mx.eval(model.parameters())
    written = []
    with pytest.raises(ValueError, match="gate_proj"):
        turboquant_quantize(model, {"model_type": "test"},
                            TurboQuantConfig(group_size=32, mlp_group_size=128, mlp_bits=2,
                                             expert_layer_tiers={1: "3"}),
                            on_quantized=lambda path, module: written.append(path))
    assert written == []
