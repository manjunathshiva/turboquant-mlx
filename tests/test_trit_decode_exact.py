"""Exact trit decoding in all four expert kernels.

With codebook {-1, 0, +1}, unit scales and inputs in {-1, 0, +1}, every
product and partial sum is a small integer, so the kernels' output is exact
in fp16 whatever order they add in. Any wrongly decoded trit then shows up
as a nonzero difference against the numpy reference, not as noise under a
tolerance.

The shapes cover what the decoders must get right: groups smaller than a word
(16), groups that start and end mid-word (64 and 128 are not multiples of
20), groups that end exactly on a
word boundary at the end of a row (in_dims a multiple of 320 = lcm(64, 20)),
a final partial word, and the highest expert index, where reading one word
past the row would leave the buffer.
"""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.core.packing import pack_trits
from turboquant_mlx.kernels.polar_dequant_experts import polar_dequant_experts
from turboquant_mlx.kernels.polar_gather_qmm import OB, polar_gather_qmm
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv

_CB = mx.array([-1.0, 0.0, 1.0], dtype=mx.float16)

# (group_size, in_dims): mid-word groups, word-aligned row ends, partial words.
_SHAPES = [(16, 48), (16, 80), (32, 160), (32, 96), (64, 192), (64, 320),
           (64, 1280), (128, 384), (128, 640)]


def _setup(num_experts, out_dims, in_dims, group_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 3, (num_experts, out_dims, in_dims)).astype(np.uint8)
    packed = pack_trits(mx.array(idx))
    scales = mx.ones((num_experts, out_dims, in_dims // group_size),
                     dtype=mx.float16)
    w = idx.astype(np.int64) - 1                      # codebook {-1, 0, +1}
    return packed, scales, w, rng


def _signs(rng, shape):
    return rng.integers(-1, 2, shape).astype(np.float16)


@pytest.mark.parametrize("group_size,in_dims", _SHAPES)
def test_gather_qmv_trit_exact(group_size, in_dims):
    packed, scales, w, rng = _setup(4, 24, in_dims, group_size, seed=in_dims)
    x = _signs(rng, in_dims)
    routed = np.array([3, 0, 2], dtype=np.uint32)       # includes the last expert
    out = polar_gather_qmv(packed, scales, _CB, mx.array(x), mx.array(routed),
                           bits=2, group_size=group_size, trit=True)
    ref = np.einsum("eoi,i->eo", w[routed], x.astype(np.int64))
    np.testing.assert_array_equal(np.array(out).astype(np.int64), ref)


@pytest.mark.parametrize("group_size,in_dims", _SHAPES)
def test_multi_gather_qmv_trit_exact(group_size, in_dims):
    packed, scales, w, rng = _setup(4, 24, in_dims, group_size, seed=in_dims + 1)
    routed = np.array([3, 1, 3, 0], dtype=np.uint32)
    x = _signs(rng, (len(routed), in_dims))
    out = polar_multi_gather_qmv(packed, scales, _CB, mx.array(x),
                                 mx.array(routed), bits=2,
                                 group_size=group_size, trit=True)
    ref = np.stack([w[e] @ x[j].astype(np.int64) for j, e in enumerate(routed)])
    np.testing.assert_array_equal(np.array(out).astype(np.int64), ref)


@pytest.mark.parametrize("group_size,in_dims", _SHAPES)
def test_dequant_experts_trit_exact(group_size, in_dims):
    packed, scales, w, _ = _setup(4, 24, in_dims, group_size, seed=in_dims + 2)
    deq = polar_dequant_experts(packed, scales, _CB, bits=2,
                                group_size=group_size, trit=True)
    np.testing.assert_array_equal(np.array(deq).astype(np.int64), w)


@pytest.mark.parametrize("group_size,in_dims", _SHAPES)
def test_gather_qmm_trit_exact(group_size, in_dims):
    packed, scales, w, rng = _setup(4, OB, in_dims, group_size, seed=in_dims + 3)
    routed = np.sort(rng.integers(0, 4, 40)).astype(np.uint32)
    routed[-1] = 3                                      # the last expert
    x = _signs(rng, (len(routed), in_dims))
    out = polar_gather_qmm(packed, scales, _CB, mx.array(x), mx.array(routed),
                           bits=2, group_size=group_size, trit=True)
    ref = np.stack([w[e] @ x[n].astype(np.int64) for n, e in enumerate(routed)])
    np.testing.assert_array_equal(np.array(out).astype(np.int64), ref)
