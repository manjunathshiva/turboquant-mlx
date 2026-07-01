"""Tests for the ternary base-3 (trit) tier: 20 trits/uint32, 3-entry codebook.

Covers pack/unpack round-trip, all four expert Metal kernels (single-token
decode, multi-input, dequant-all, batched gather-GEMM), the quantize path
(3-entry codebook + ~1.6 bpw), and the PolarQuantizedSwitchLinear dispatch.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.core.packing import pack_trits, unpack_trits, TRITS_PER_U32
from turboquant_mlx.core.polar_quantize import (
    polar_quantize_weight, polar_dequantize_weight,
)
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv
from turboquant_mlx.kernels.polar_dequant_experts import polar_dequant_experts
from turboquant_mlx.kernels.polar_gather_qmm import polar_gather_qmm, supports
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

_C = 1.22401
_CB = mx.array([-_C, 0.0, _C], dtype=mx.float16)
_CB_NP = np.array([-_C, 0.0, _C], dtype=np.float32)


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", [(37,), (8, 100), (4, 40, 512)])
def test_pack_unpack_roundtrip(shape):
    idx = mx.array(np.random.randint(0, 3, size=shape).astype(np.uint8))
    packed = pack_trits(idx)
    back = unpack_trits(packed, shape[-1])
    mx.eval(packed, back)
    assert packed.shape[-1] == math.ceil(shape[-1] / TRITS_PER_U32)
    assert packed.dtype == mx.uint32
    assert mx.array_equal(back.astype(mx.uint8), idx.astype(mx.uint8))


def test_all_twos_no_overflow():
    idx = mx.full((TRITS_PER_U32,), 2, dtype=mx.uint8)
    packed = pack_trits(idx)
    mx.eval(packed)
    assert int(packed[0]) == 3 ** TRITS_PER_U32 - 1  # 3486784400 < 2**32


def test_bits_per_weight_under_two():
    n = 4096
    idx = mx.zeros((n,), dtype=mx.uint8)
    packed = pack_trits(idx)
    bpw = packed.shape[-1] * 32 / n
    assert bpw < 1.7  # genuine sub-2-bit (vs 2.0 for the bit-packed slot)


# --------------------------------------------------------------------------- #
# reference helpers
# --------------------------------------------------------------------------- #

def _rand_expert_setup(num_experts, out_dims, in_dims, group_size, seed=0):
    np.random.seed(seed)
    idx = np.random.randint(0, 3, (num_experts, out_dims, in_dims)).astype(np.uint8)
    n_groups = in_dims // group_size
    scales = (0.5 + np.random.rand(num_experts, out_dims, n_groups)).astype(np.float16)
    packed = pack_trits(mx.array(idx))
    mx.eval(packed)
    col_group = np.arange(in_dims) // group_size
    w = _CB_NP[idx] * scales.astype(np.float32)[:, :, col_group]   # (E, out, in)
    return idx, mx.array(scales), packed, w


def _rel(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-6)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #

def test_gather_qmv_trit_matches_reference():
    gs = 32
    idx, scales, packed, w = _rand_expert_setup(6, 40, 96, gs)
    x = np.random.randn(96).astype(np.float16)
    routed = np.array([4, 1, 5], dtype=np.uint32)
    ref = np.einsum("eoi,i->eo", w[routed], x.astype(np.float32))
    out = polar_gather_qmv(packed, scales, _CB, mx.array(x), mx.array(routed),
                           bits=2, group_size=gs, trit=True)
    mx.eval(out)
    assert _rel(np.array(out).astype(np.float32), ref) < 2e-3


def test_multi_gather_qmv_trit_matches_reference():
    gs = 32
    idx, scales, packed, w = _rand_expert_setup(8, 48, 96, gs, seed=1)
    routed = np.array([7, 0, 3, 5], dtype=np.uint32)
    x = np.random.randn(len(routed), 96).astype(np.float16)
    ref = np.stack([w[e] @ x[j].astype(np.float32) for j, e in enumerate(routed)])
    out = polar_multi_gather_qmv(packed, scales, _CB, mx.array(x), mx.array(routed),
                                 bits=2, group_size=gs, trit=True)
    mx.eval(out)
    assert _rel(np.array(out).astype(np.float32), ref) < 2e-3


def test_dequant_experts_trit_matches_reference():
    gs = 32
    idx, scales, packed, w = _rand_expert_setup(5, 128, 96, gs, seed=2)
    deq = polar_dequant_experts(packed, scales, _CB, bits=2, group_size=gs, trit=True)
    mx.eval(deq)
    assert _rel(np.array(deq).astype(np.float32), w) < 2e-3


def test_gather_qmm_trit_matches_reference():
    gs = 32
    idx, scales, packed, w = _rand_expert_setup(5, 128, 96, gs, seed=3)
    assert supports(128)
    N = 40
    routed = np.sort(np.random.randint(0, 5, N)).astype(np.uint32)
    x = np.random.randn(N, 96).astype(np.float16)
    ref = np.stack([w[routed[n]] @ x[n].astype(np.float32) for n in range(N)])
    y = polar_gather_qmm(packed, scales, _CB, mx.array(x), mx.array(routed),
                         bits=2, group_size=gs, trit=True)
    mx.eval(y)
    assert _rel(np.array(y).astype(np.float32), ref) < 3e-3


# --------------------------------------------------------------------------- #
# quantize + layer
# --------------------------------------------------------------------------- #

def test_quantize_emits_trit_format():
    mx.random.seed(3)
    out_dims, in_dims, gs = 64, 256, 32
    w = mx.random.normal((out_dims, in_dims)).astype(mx.float16)
    res = polar_quantize_weight(w, bits=2, group_size=gs, ternary=True)
    mx.eval(res["packed_weight"], res["codebook"])
    assert res["codebook"].shape == (3,)                          # trit marker
    assert res["packed_weight"].shape[1] == math.ceil(in_dims / TRITS_PER_U32)
    # kernel dequant must equal the reference unpack_trits dequant
    ref = polar_dequantize_weight(res["packed_weight"], res["scales"],
                                  res["codebook"], bits=2, group_size=gs,
                                  input_dims=in_dims, trit=True)
    deq = polar_dequant_experts(res["packed_weight"][None], res["scales"][None],
                                res["codebook"], bits=2, group_size=gs, trit=True)[0]
    mx.eval(ref, deq)
    assert float(mx.abs(ref - deq).max()) == 0.0


def test_from_switch_linear_ternary_forces_bits_2():
    """ternary=True with the default bits=3 must not raise (bits forced to 2)."""
    mx.random.seed(9)
    fw = mx.random.normal((8, 64, 128)).astype(mx.float16)
    layer = PolarQuantizedSwitchLinear.from_switch_linear(
        None, group_size=32, seed=1, float_weight=fw, ternary=True,  # bits defaults to 3
    )
    assert layer.trit is True
    assert layer.bits == 2
    assert layer.codebook.shape == (3,)


def test_switch_layer_trit_dispatch():
    mx.random.seed(4)
    E, out_dims, in_dims, gs, k = 16, 128, 256, 32, 4
    fw = mx.random.normal((E, out_dims, in_dims)).astype(mx.float16)
    layer = PolarQuantizedSwitchLinear.from_switch_linear(
        None, bits=2, group_size=gs, seed=42, float_weight=fw, ternary=True,
    )
    assert layer.trit is True
    assert layer.codebook.shape == (3,)
    assert layer.weight.shape == (E, out_dims, math.ceil(in_dims / TRITS_PER_U32))

    w_deq = np.array(layer._dequantize_all()).astype(np.float32)
    from turboquant_mlx.core.rotation import rotate_input

    # single-token decode
    x1 = mx.random.normal((in_dims,)).astype(mx.float16)
    idx1 = mx.array([3, 0, 9, 15], dtype=mx.uint32)
    x1r = np.array(rotate_input(x1, layer.signs)).astype(np.float32)
    ref1 = np.stack([w_deq[e] @ x1r for e in np.array(idx1)])
    y1 = np.array(layer(x1.reshape(1, 1, in_dims), idx1.reshape(1, k)))
    assert _rel(y1.reshape(k, out_dims).astype(np.float32), ref1) < 3e-3

    # multi-input (one vector per expert)
    xk = mx.random.normal((k, 1, in_dims)).astype(mx.float16)
    idxk = mx.array([1, 4, 4, 12], dtype=mx.uint32)
    xkr = np.array(rotate_input(xk, layer.signs)).astype(np.float32).reshape(k, in_dims)
    refk = np.stack([w_deq[e] @ xkr[j] for j, e in enumerate(np.array(idxk))])
    yk = np.array(layer(xk, idxk))
    assert _rel(yk.reshape(k, out_dims).astype(np.float32), refk) < 3e-3
