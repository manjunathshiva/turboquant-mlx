"""Unit tests for the streaming K-reduction knob (_cap_active_experts).

The Flash-MoE K-reduction lever lowers router top_k so streamed MoEs load fewer
experts per token (~2x less disk I/O at no quality cost up to K=4 on validated
models). The contract this locks: cap only (never raise top_k), apply only to
real MoE blocks (top_k + switch_mlp), and treat 0/None as "use native routing".
"""

from turboquant_mlx.stream.loader import _cap_active_experts


class FakeMoE:
    """A SparseMoeBlock-like stub: has both top_k and switch_mlp."""

    def __init__(self, top_k):
        self.top_k = top_k
        self.switch_mlp = object()


class FakeDense:
    """A non-MoE mlp: no top_k / switch_mlp — must be left untouched."""


class FakeLayer:
    def __init__(self, mlp):
        self.mlp = mlp


def _layers(*top_ks):
    return [FakeLayer(FakeMoE(k)) for k in top_ks]


def test_caps_down_to_target():
    layers = _layers(8, 8, 8)
    _cap_active_experts(layers, 4)
    assert [l.mlp.top_k for l in layers] == [4, 4, 4]


def test_never_raises_above_native():
    # native 2 with a cap of 4 must stay 2 (min), not jump to 4.
    layers = _layers(2, 8)
    _cap_active_experts(layers, 4)
    assert [l.mlp.top_k for l in layers] == [2, 4]


def test_zero_disables():
    layers = _layers(8, 8)
    _cap_active_experts(layers, 0)
    assert [l.mlp.top_k for l in layers] == [8, 8]


def test_dense_layers_untouched():
    dense = FakeLayer(FakeDense())
    moe = FakeLayer(FakeMoE(8))
    _cap_active_experts([dense, moe], 4)
    assert not hasattr(dense.mlp, "top_k")
    assert moe.mlp.top_k == 4
