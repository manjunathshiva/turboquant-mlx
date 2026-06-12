"""Tests for the fused expert-dequant kernel and the large-batch switch routing."""

import math

import mlx.core as mx
import pytest

from turboquant_mlx.kernels.polar_dequant_experts import polar_dequant_experts
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear


def _make_layer(num_experts=4, output_dims=64, input_dims=128, bits=3, group_size=32):
    mx.random.seed(0)
    w = mx.random.normal((num_experts, output_dims, input_dims)).astype(mx.float16)
    return PolarQuantizedSwitchLinear.from_switch_linear(
        None, bits=bits, group_size=group_size, seed=7, float_weight=w,
    )


def _dequantize_all_reference(layer):
    """The pre-0.7.0 multi-op Python dequant path, kept as the test oracle."""
    from turboquant_mlx.core.packing import unpack_indices
    from turboquant_mlx.core.codebook import dequantize_scalar

    n_groups = layer.input_dims // layer.group_size
    indices = unpack_indices(layer.weight, layer.bits, layer.input_dims)
    w_deq = dequantize_scalar(indices, layer.codebook)
    w_deq = w_deq.reshape(layer.num_experts, layer.output_dims, n_groups,
                          layer.group_size)
    w_deq = w_deq * mx.expand_dims(layer.scales, axis=-1)
    return w_deq.reshape(layer.num_experts, layer.output_dims, layer.input_dims)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("group_size", [32, 64])
def test_kernel_matches_python_dequant(bits, group_size):
    layer = _make_layer(bits=bits, group_size=group_size)
    ref = _dequantize_all_reference(layer)
    got = polar_dequant_experts(
        layer.weight, layer.scales, layer.codebook, bits, group_size,
    )
    mx.eval(ref, got)
    assert got.shape == ref.shape
    assert float(mx.abs(got - ref).max()) == 0.0  # bit-identical


def test_dequantize_all_uses_kernel():
    layer = _make_layer()
    ref = _dequantize_all_reference(layer)
    got = layer._dequantize_all()
    mx.eval(ref, got)
    assert float(mx.abs(got - ref).max()) == 0.0


def test_large_batch_routing_matches_gather_kernel():
    """The >=512-routing gather_mm path must match polar_multi_gather_qmv."""
    layer = _make_layer(num_experts=8, output_dims=32, input_dims=64)
    k = 1024  # above _GATHER_MM_MIN_ROUTINGS, sorted-prefill style shapes
    mx.random.seed(1)
    x = mx.random.normal((k, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.sort(mx.random.randint(0, layer.num_experts, (k,)).astype(mx.uint32))
    mx.eval(x, idx)

    y_fast = layer(x, idx, sorted_indices=True)

    # Force the original kernel path by lowering k below the threshold split:
    # call per-chunk so each call stays under the routing threshold.
    chunks = []
    step = 256
    for s in range(0, k, step):
        chunks.append(layer(x[s:s + step], idx[s:s + step], sorted_indices=True))
    y_ref = mx.concatenate(chunks, axis=0)
    mx.eval(y_fast, y_ref)

    assert y_fast.shape == y_ref.shape
    # The gather kernel keeps codebook*x products in fp32; the gather_mm path
    # rounds dequantized weights to fp16 first — so compare relatively.
    a = y_fast.astype(mx.float32)
    b = y_ref.astype(mx.float32)
    rel = float(mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-12))
    assert rel < 2e-3


def test_small_k_keeps_gather_kernel_path():
    """Decode-scale calls (k below threshold) must be unaffected."""
    layer = _make_layer(num_experts=8, output_dims=32, input_dims=64)
    x = mx.random.normal((1, 1, 8, 1, layer.input_dims)).astype(mx.float16)
    idx = mx.random.randint(0, layer.num_experts, (1, 1, 8)).astype(mx.uint32)
    mx.eval(x, idx)
    y = layer(x, idx)
    mx.eval(y)
    assert y.shape == (1, 1, 8, 1, layer.output_dims)
