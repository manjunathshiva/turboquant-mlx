"""The decode/prefill kernel pair must agree bit-for-bit.

Background (colibrì issue #100): their shape-dependent int kernels make a batched
forward round differently from the single-token path, forking greedy output on
3/5 prompts with zero speculation. We dispatch four MoE kernels *by shape*, so
the same (token, expert) routing can be computed by a different kernel purely
because of how many tokens are in the forward.

Audited 2026-07-15 (`scripts/kernel_determinism/`). Result: `polar_gather_qmv`
(decode) and `polar_multi_gather_qmv` (prefill) are **bit-identical** — which is
the pair that matters, because 0.13.0's disk-cache checkpoint ladder restores a
state built on the prefill path and then continues on the decode path. This test
pins that invariant so a kernel refactor can't quietly break it.

(`polar_gather_qmm` reduces in a batch-size-dependent order and differs by
~1e-4..4e-3 — inherent to tiled GEMM, and NOT asserted here. The end-to-end
consequence is documented in the README: it is a property of chunked prefill on
Metal, reproduced on stock mlx-lm with none of our code in the loop.)
"""

import mlx.core as mx
import pytest

from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear


def _layer(bits, group_size, ternary=False, num_experts=8, out_dims=128,
           in_dims=256):
    mx.random.seed(0)
    w = mx.random.normal((num_experts, out_dims, in_dims)).astype(mx.float16)
    return PolarQuantizedSwitchLinear.from_switch_linear(
        None, bits=bits, group_size=group_size, seed=7, float_weight=w,
        ternary=ternary,
    )


@pytest.mark.parametrize("bits,group_size,ternary", [
    (2, 32, False), (3, 64, False), (4, 64, False), (2, 64, True),
])
def test_decode_and_prefill_kernels_are_bit_identical(bits, group_size, ternary):
    """One token routed to k experts: the decode kernel and the prefill kernel
    must produce the same bytes, not merely similar floats."""
    layer = _layer(bits, group_size, ternary)
    mx.random.seed(1)
    k = 8
    x = mx.random.normal((layer.input_dims,)).astype(mx.float16)
    idx = mx.array([0, 1, 2, 3, 4, 5, 6, 7][:k], dtype=mx.uint32)

    decode = polar_gather_qmv(
        layer.weight, layer.scales, layer.codebook, x, idx,
        layer.bits, layer.group_size, trit=layer.trit)

    # The prefill kernel sees one row per (token, expert) routing. Materialise
    # the broadcast: at the real call site x_rows comes from a reshape of a real
    # tensor, so a strided view would be testing a shape we never actually pass.
    x_rows = mx.contiguous(
        mx.broadcast_to(x.reshape(1, -1), (k, layer.input_dims)))
    prefill = polar_multi_gather_qmv(
        layer.weight, layer.scales, layer.codebook, x_rows, idx,
        layer.bits, layer.group_size, trit=layer.trit)

    mx.eval(decode, prefill)
    assert decode.shape == prefill.shape
    assert float(mx.abs(decode - prefill).max()) == 0.0, (
        "decode and prefill kernels disagree — the disk-cache checkpoint "
        "ladder restores a prefill-built state and continues on decode, so "
        "these two must stay bit-identical"
    )
