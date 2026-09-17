"""A 2-token forward through PolarQuantizedLinear equals two 1-token forwards.

Speculative decoding verifies two tokens at once and accepts a draft only if the
verify's argmax matches greedy's, so the 2-token path must produce exactly what
the 1-token decode path would. It routes through polar_qmv one row at a time,
the same kernel single-token decode uses, which makes that bit-exact by
construction; these tests pin it, and pin the dispatch boundaries.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import turboquant_mlx.layers.polar_linear as pl
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _layer(out_dims=700, in_dims=256, bits=4, group_size=64):
    mx.random.seed(0)
    return PolarQuantizedLinear.from_linear(nn.Linear(in_dims, out_dims, bias=False),
                                            bits=bits, group_size=group_size)


def _bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 2, 256), (2, 256)])
def test_two_tokens_equal_two_single_token_calls(bits, shape):
    layer = _layer(bits=bits)
    x = mx.random.normal(shape, key=mx.random.key(bits)).astype(mx.float16)
    rows = x.reshape(2, 256)
    got = layer(x).reshape(2, -1)
    want = mx.concatenate([layer(rows[r][None, None, :]).reshape(1, -1) for r in range(2)])
    assert np.array_equal(_bits(got), _bits(want))


def test_dispatch_uses_qmv_for_two_and_qmm_for_three(monkeypatch):
    layer = _layer()
    calls = []
    real_qmv, real_qmm = pl.polar_qmv, pl.polar_qmm
    monkeypatch.setattr(pl, "polar_qmv", lambda *a: calls.append("qmv") or real_qmv(*a))
    monkeypatch.setattr(pl, "polar_qmm", lambda *a: calls.append("qmm") or real_qmm(*a))

    mx.eval(layer(mx.random.normal((1, 1, 256)).astype(mx.float16)))
    assert calls == ["qmv"]
    calls.clear()
    mx.eval(layer(mx.random.normal((1, 2, 256)).astype(mx.float16)))
    assert calls == ["qmv", "qmv"]
    calls.clear()
    mx.eval(layer(mx.random.normal((1, 3, 256)).astype(mx.float16)))
    assert calls == ["qmm"]
