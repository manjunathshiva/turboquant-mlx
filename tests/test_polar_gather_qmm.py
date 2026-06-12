"""Parity tests for polar_gather_qmm (tiled batched gather-GEMM) against the
per-row polar_multi_gather_qmv kernel, plus routing behavior in
PolarQuantizedSwitchLinear."""

import mlx.core as mx
import pytest

from turboquant_mlx.kernels.polar_gather_qmm import polar_gather_qmm, supports
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear


def _make_layer(num_experts=8, output_dims=64, input_dims=128, bits=3,
                group_size=32):
    mx.random.seed(0)
    w = mx.random.normal((num_experts, output_dims, input_dims)).astype(mx.float16)
    return PolarQuantizedSwitchLinear.from_switch_linear(
        None, bits=bits, group_size=group_size, seed=7, float_weight=w,
    )


def _rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float(mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-12))


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("n", [1024, 1000])  # multiple of 16 and not
def test_parity_vs_multi_gather(bits, group_size, n):
    layer = _make_layer(bits=bits, group_size=group_size)
    mx.random.seed(1)
    x = mx.random.normal((n, layer.input_dims)).astype(mx.float16)
    idx = mx.sort(mx.random.randint(0, layer.num_experts, (n,)).astype(mx.uint32))
    mx.eval(x, idx)

    got = polar_gather_qmm(layer.weight, layer.scales, layer.codebook,
                           x, idx, bits, group_size)
    ref = polar_multi_gather_qmv(layer.weight, layer.scales, layer.codebook,
                                 x, idx, bits, group_size)
    mx.eval(got, ref)
    assert got.shape == ref.shape
    assert _rel(got, ref) < 1e-4  # fp32-accumulated both sides


def test_partial_k_chunk():
    # K=160 is well below one packed-word chunk for every bit width,
    # exercising the partial-chunk guards.
    layer = _make_layer(input_dims=160, output_dims=128)
    mx.random.seed(2)
    n = 640
    x = mx.random.normal((n, 160)).astype(mx.float16)
    idx = mx.sort(mx.random.randint(0, 8, (n,)).astype(mx.uint32))
    got = polar_gather_qmm(layer.weight, layer.scales, layer.codebook,
                           x, idx, 3, 32)
    ref = polar_multi_gather_qmv(layer.weight, layer.scales, layer.codebook,
                                 x, idx, 3, 32)
    mx.eval(got, ref)
    assert _rel(got, ref) < 1e-4


def test_single_expert_all_tokens():
    layer = _make_layer()
    n = 768
    x = mx.random.normal((n, layer.input_dims)).astype(mx.float16)
    idx = mx.zeros((n,), dtype=mx.uint32)
    got = polar_gather_qmm(layer.weight, layer.scales, layer.codebook,
                           x, idx, 3, 32)
    ref = polar_multi_gather_qmv(layer.weight, layer.scales, layer.codebook,
                                 x, idx, 3, 32)
    mx.eval(got, ref)
    assert _rel(got, ref) < 1e-4


def test_supports_gate():
    assert supports(64) and supports(1408) and supports(2816)
    assert not supports(96)  # not a multiple of the 64-row block


def test_output_dims_not_multiple_of_block():
    """Direct kernel call with O % 64 != 0: the tail block's extra threads
    clamp their address row and must not corrupt results or read OOB."""
    layer = _make_layer(output_dims=96)
    n = 640
    mx.random.seed(4)
    x = mx.random.normal((n, layer.input_dims)).astype(mx.float16)
    idx = mx.sort(mx.random.randint(0, layer.num_experts, (n,)).astype(mx.uint32))
    got = polar_gather_qmm(layer.weight, layer.scales, layer.codebook,
                           x, idx, 3, 32)
    ref = polar_multi_gather_qmv(layer.weight, layer.scales, layer.codebook,
                                 x, idx, 3, 32)
    mx.eval(got, ref)
    assert got.shape == (n, 96)
    assert _rel(got, ref) < 1e-4


def test_switch_layer_routes_sorted_large_batch():
    """Layer output via the new kernel must match per-chunk gather-kernel calls."""
    layer = _make_layer()
    k = 1024
    mx.random.seed(3)
    x = mx.random.normal((k, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.sort(mx.random.randint(0, layer.num_experts, (k,)).astype(mx.uint32))
    mx.eval(x, idx)

    y_fast = layer(x, idx, sorted_indices=True)
    chunks = [layer(x[s:s + 256], idx[s:s + 256], sorted_indices=True)
              for s in range(0, k, 256)]
    y_ref = mx.concatenate(chunks, axis=0)
    mx.eval(y_fast, y_ref)
    assert y_fast.shape == y_ref.shape
    assert _rel(y_fast, y_ref) < 1e-4
