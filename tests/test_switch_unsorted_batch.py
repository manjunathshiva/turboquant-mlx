"""Parity tests for the small-unsorted-batch dispatch in
PolarQuantizedSwitchLinear.

mlx-lm's SwitchGLU only sorts routings when indices.size >= 64, so short
forwards (2-7 tokens at top-8) arrive unsorted. These used to fall into the
dequantize-all-experts fallback (~34x a single-token forward); they must now
take the fused per-routing kernel and match the dequant + mx.gather_mm
reference in both unsorted row layouts:

  gate/up: one row per token          (B, L, 1, 1, in), idx (B, L, k)
  down:    one row per (token,expert) (B, L, k, 1, in), idx (B, L, k)

Also pins the n_tokens == k collision: 4 unsorted tokens on a top-4 model
used to be misread as flat per-expert rows (misaligned with its k*k indices).
"""

import mlx.core as mx
import pytest

import turboquant_mlx.layers.polar_switch_linear as psl
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear


def _make_layer(num_experts=8, output_dims=64, input_dims=128, bits=3,
                group_size=32, ternary=False, bias=False):
    mx.random.seed(0)
    w = mx.random.normal((num_experts, output_dims, input_dims)).astype(mx.float16)
    b = None
    if bias:
        b = mx.random.normal((num_experts, output_dims)).astype(mx.float16)
    return PolarQuantizedSwitchLinear.from_switch_linear(
        None, bits=bits, group_size=group_size, seed=7, float_weight=w,
        bias=b, ternary=ternary,
    )


def _rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-12))


def _dequant_reference(layer, x, idx):
    """Same inputs through the dequant + gather_mm fallback."""
    saved = psl._GATHER_MM_MIN_ROUTINGS
    psl._GATHER_MM_MIN_ROUTINGS = 0
    try:
        return layer(x, idx)
    finally:
        psl._GATHER_MM_MIN_ROUTINGS = saved


def _fused_only(layer, x, idx):
    """Layer call that fails the test if it falls back to dequant-all."""
    def _boom(self):
        raise AssertionError("small unsorted batch fell back to dequant-all")

    saved = PolarQuantizedSwitchLinear._dequantize_all
    PolarQuantizedSwitchLinear._dequantize_all = _boom
    try:
        y = layer(x, idx)
        mx.eval(y)
        return y
    finally:
        PolarQuantizedSwitchLinear._dequantize_all = saved


@pytest.mark.parametrize("ternary", [False, True])
@pytest.mark.parametrize("n_tokens", [2, 7])
def test_gate_up_layout_parity(ternary, n_tokens):
    layer = _make_layer(ternary=ternary)
    mx.random.seed(1)
    x = mx.random.normal((1, n_tokens, 1, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, n_tokens, 8)).astype(mx.uint32)
    mx.eval(x, idx)

    ref = _dequant_reference(layer, x, idx)
    got = _fused_only(layer, x, idx)
    mx.eval(got, ref)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-3


@pytest.mark.parametrize("ternary", [False, True])
def test_down_layout_parity(ternary):
    layer = _make_layer(ternary=ternary)
    n_tokens, k = 3, 8
    mx.random.seed(2)
    x = mx.random.normal((1, n_tokens, k, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, n_tokens, k)).astype(mx.uint32)
    mx.eval(x, idx)

    ref = _dequant_reference(layer, x, idx)
    got = _fused_only(layer, x, idx)
    mx.eval(got, ref)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-3


def test_bias_applied_on_fused_path():
    layer = _make_layer(bias=True)
    n_tokens = 4
    mx.random.seed(3)
    x = mx.random.normal((1, n_tokens, 1, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, n_tokens, 8)).astype(mx.uint32)
    mx.eval(x, idx)

    ref = _dequant_reference(layer, x, idx)
    got = _fused_only(layer, x, idx)
    mx.eval(got, ref)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-3


def test_top4_collision_n_tokens_equals_k():
    """4 unsorted tokens on a top-4 model: n_tokens == k but indices.size is
    k*k — must be dispatched as an unsorted batch, not as flat routings."""
    layer = _make_layer(num_experts=4)
    n_tokens = k = 4
    mx.random.seed(4)
    x = mx.random.normal((1, n_tokens, 1, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, n_tokens, k)).astype(mx.uint32)
    mx.eval(x, idx)

    ref = _dequant_reference(layer, x, idx)
    got = _fused_only(layer, x, idx)
    mx.eval(got, ref)
    assert got.shape == ref.shape == (1, n_tokens, k, 1, layer.output_dims)
    assert _rel(got, ref) < 1e-3


def test_large_unsorted_batch_keeps_dequant_path():
    """Past _GATHER_MM_MIN_ROUTINGS the per-row kernel loses; the dequant +
    gather_mm path must still be chosen for large unsorted batches."""
    layer = _make_layer()
    n_tokens = 128  # 128 x top-8 = 1024 routings >= 512
    mx.random.seed(5)
    x = mx.random.normal((1, n_tokens, 1, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, n_tokens, 8)).astype(mx.uint32)
    mx.eval(x, idx)

    def _boom(*a, **kw):
        raise AssertionError("large unsorted batch took the per-row kernel")

    saved = psl.polar_multi_gather_qmv
    psl.polar_multi_gather_qmv = _boom
    try:
        y = layer(x, idx)
        mx.eval(y)
    finally:
        psl.polar_multi_gather_qmv = saved
    assert y.shape == (1, n_tokens, 8, 1, layer.output_dims)
