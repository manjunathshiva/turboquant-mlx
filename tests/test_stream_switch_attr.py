"""The streaming loader must find an MoE block's stacked-expert container under
either conventional attribute name.

qwen3_5_moe/deepseek name it ``switch_mlp``; laguna's ``LagunaMoE`` holds a plain
mlx-lm ``SwitchGLU`` at ``experts``. Before this was handled, ``load_streaming``
on a Laguna model swapped **zero** projections and silently fell back to loading
the whole model resident — fine on a 64 GB Mac, an OOM on a 16 GB one. The
attribute name is also the weight-key segment, so a mismatch here reads the
wrong tensor rather than failing loudly; both are pinned below.
"""

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear
from turboquant_mlx.stream.loader import _PROJS, _find_switch


def _switch_linear(num_experts=4, input_dims=64, output_dims=32, bits=3,
                   group_size=64):
    """A minimal PolarQuantizedSwitchLinear with the right shapes/dtypes."""
    packed = input_dims * bits // 32
    layer = PolarQuantizedSwitchLinear(
        input_dims=input_dims,
        output_dims=output_dims,
        num_experts=num_experts,
        bits=bits,
        group_size=group_size,
        needs_rotation=True,
    )
    layer.weight = mx.zeros((num_experts, output_dims, packed), dtype=mx.uint32)
    layer.scales = mx.ones((num_experts, output_dims, input_dims // group_size),
                           dtype=mx.float16)
    layer.codebook = mx.zeros((2 ** bits,), dtype=mx.float16)
    layer.signs = mx.ones((input_dims,), dtype=mx.float16)
    return layer


class _Container(nn.Module):
    """Stands in for SwitchGLU: just holds the three projections."""

    def __init__(self):
        super().__init__()
        for proj in _PROJS:
            setattr(self, proj, _switch_linear())


class _MoEBlock(nn.Module):
    def __init__(self, attr):
        super().__init__()
        self.top_k = 8
        setattr(self, attr, _Container())


def test_finds_switch_mlp_container():
    name, sm = _find_switch(_MoEBlock("switch_mlp"))
    assert name == "switch_mlp"
    assert isinstance(sm.gate_proj, PolarQuantizedSwitchLinear)


def test_finds_experts_container_laguna_style():
    name, sm = _find_switch(_MoEBlock("experts"))
    assert name == "experts"
    assert isinstance(sm.gate_proj, PolarQuantizedSwitchLinear)


def test_returned_name_matches_the_weight_key_segment():
    """The name is spliced into `model.layers.N.mlp.<name>.gate_proj.weight`;
    returning the module without its attribute name would read the wrong key."""
    for attr in ("switch_mlp", "experts"):
        name, _ = _find_switch(_MoEBlock(attr))
        assert f"model.layers.0.mlp.{name}.gate_proj.weight" == (
            f"model.layers.0.mlp.{attr}.gate_proj.weight")


def test_dense_mlp_is_not_a_switch():
    """A block with no expert container must be skipped, not crash."""
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 8)

    assert _find_switch(Dense()) == (None, None)
    assert _find_switch(None) == (None, None)


def test_shared_expert_mlp_is_not_mistaken_for_the_switch():
    """`shared_experts` is a dense always-on MLP that must stay resident. It is
    not in the candidate list, and its projections aren't switch layers anyway —
    streaming it would page a tensor that every single token needs."""
    class WithSharedOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 8
            self.shared_experts = nn.Linear(8, 8)

    assert _find_switch(WithSharedOnly()) == (None, None)


def test_empty_expert_container_is_ignored():
    """An `experts` attribute holding something other than quantized switch
    projections (e.g. an unquantized model) must not be claimed."""
    class NotQuantized(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 8
            self.experts = nn.Linear(8, 8)

    assert _find_switch(NotQuantized()) == (None, None)


def test_k_reduction_sees_moe_blocks_under_either_name():
    """K-reduction only needs "is this a router-driven MoE block", so it uses the
    presence-only check — an unquantized or stubbed expert container still counts.
    Tightening this to _find_switch would silently stop capping top_k."""
    from turboquant_mlx.stream.loader import _cap_active_experts, _has_switch

    class Stub:
        def __init__(self, attr):
            self.top_k = 8
            setattr(self, attr, object())

    class Layer:
        def __init__(self, mlp):
            self.mlp = mlp

    for attr in ("switch_mlp", "experts"):
        assert _has_switch(Stub(attr))
        layers = [Layer(Stub(attr))]
        _cap_active_experts(layers, 4)
        assert layers[0].mlp.top_k == 4, f"top_k not capped for mlp.{attr}"


def test_k_reduction_leaves_dense_blocks_alone():
    from turboquant_mlx.stream.loader import _cap_active_experts, _has_switch

    class Dense:
        top_k = 8  # a top_k without any expert container is not an MoE block

    class Layer:
        def __init__(self, mlp):
            self.mlp = mlp

    assert not _has_switch(Dense())
    layers = [Layer(Dense())]
    _cap_active_experts(layers, 4)
    assert layers[0].mlp.top_k == 8
