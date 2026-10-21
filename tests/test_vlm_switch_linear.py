# Copyright 2026 Manjunath Janardhan
"""mlx-vlm's own expert layers must be quantized on convert and rebuilt on load.

Since mlx-vlm 0.6.12, ``mlx_vlm.models.switch_layers`` defines its own
``SwitchLinear`` / ``QuantizedSwitchLinear`` instead of reusing mlx-lm's, and
every mlx-vlm MoE (DiffusionGemma, Nemotron-H Omni) builds its experts from
them. Detection that knew only mlx-lm's classes skipped every VLM expert on
convert and swapped in none of them on load, so a published TurboQuant VLM MoE
could not be loaded at all.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

vlm_switch = pytest.importorskip(
    "mlx_vlm.models.switch_layers", reason="needs the [vlm] extra")

from turboquant_mlx.compat import switch_linear_classes  # noqa: E402
from turboquant_mlx.config import TurboQuantConfig  # noqa: E402
from turboquant_mlx.layers.polar_switch_linear import (  # noqa: E402
    PolarQuantizedSwitchLinear,
)
from turboquant_mlx.quantize_model import (  # noqa: E402
    _is_switch_linear,
    turboquant_quantize,
)


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        # Nemotron-H's shape: SwitchMLP with fc1 (up) and fc2 (down).
        self.switch_mlp = vlm_switch.SwitchMLP(64, 128, 4)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Mixer()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer(), _Layer()]


class TinyVlmMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()


def _quantize(**kw):
    mx.random.seed(0)
    model = TinyVlmMoE()
    mx.eval(model.parameters())
    tq = TurboQuantConfig(bits=3, group_size=32, **kw)
    model, _ = turboquant_quantize(model, {"model_type": "test"}, tq)
    return model, tq


def test_both_libraries_classes_are_recognised():
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    plain, quantized = switch_linear_classes()
    assert SwitchLinear in plain and vlm_switch.SwitchLinear in plain
    assert QuantizedSwitchLinear in quantized
    assert vlm_switch.QuantizedSwitchLinear in quantized
    assert _is_switch_linear(vlm_switch.SwitchLinear(64, 32, 4))
    assert _is_switch_linear(vlm_switch.QuantizedSwitchLinear(64, 32, 4))


@pytest.mark.parametrize("ternary", [False, True])
def test_convert_quantizes_mlx_vlm_experts(ternary):
    model, _ = _quantize(ternary_experts=ternary)
    for layer in model.backbone.layers:
        for proj in ("fc1", "fc2"):
            pq = getattr(layer.mixer.switch_mlp, proj)
            assert isinstance(pq, PolarQuantizedSwitchLinear), proj
            assert pq.trit == ternary


def test_load_rebuilds_mlx_vlm_experts_and_matches_the_converted_model():
    from turboquant_mlx.generate import _prepare_polar_layers

    model, tq = _quantize(ternary_experts=True)
    weights = dict(tree_flatten(model.parameters()))

    fresh = TinyVlmMoE()
    mx.eval(fresh.parameters())
    fresh = _prepare_polar_layers(fresh, weights, tq)
    fresh.load_weights(list(weights.items()))

    for layer in fresh.backbone.layers:
        for proj in ("fc1", "fc2"):
            assert isinstance(getattr(layer.mixer.switch_mlp, proj),
                              PolarQuantizedSwitchLinear), proj

    x = mx.random.normal((1, 3, 64))
    idx = mx.array([[[0, 2], [1, 3], [2, 0]]])
    a = model.backbone.layers[0].mixer.switch_mlp(x, idx)
    b = fresh.backbone.layers[0].mixer.switch_mlp(x, idx)
    assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("block", ["SwitchGLU", "SwitchMLP"])
@pytest.mark.parametrize("tokens", [1, 16, 256])
def test_polar_experts_compute_the_same_under_either_librarys_block(block, tokens):
    """mlx-vlm's MoE blocks call the expert layers exactly as mlx-lm's do.

    Same float weights, same quantization, mlx-lm's block against mlx-vlm's:
    the PolarQuantizedSwitchLinear outputs must be identical. 1 token takes the
    decode kernel; 256 tokens x top-4 = 1024 routings takes the batched
    gather-GEMM path (a DiffusionGemma canvas forward).
    """
    import mlx_lm.models.switch_layers as lm_switch

    class Blk(nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.experts = cls(256, 512, 16)

    mx.random.seed(0)
    base = Blk(getattr(lm_switch, block))
    mx.eval(base.parameters())
    tq = TurboQuantConfig(bits=4, group_size=64)
    outs = []
    for lib in (lm_switch, vlm_switch):
        m = Blk(getattr(lib, block))
        m.update(base.parameters())
        m, _ = turboquant_quantize(m, {"model_type": "test"}, tq)
        outs.append(m)

    x = mx.random.normal((1, tokens, 256))
    idx = mx.random.randint(0, 16, (1, tokens, 4))
    a, b = (m.experts(x, idx) for m in outs)
    assert mx.array_equal(a, b).item()
