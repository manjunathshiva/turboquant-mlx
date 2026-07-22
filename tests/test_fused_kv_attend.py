"""Fused KV decode+attend kernel: parity with the shipped dequant+SDPA path.

The fused path (``TurboQuantKVCache.fused_attend`` + the SDPA-seam patch) must
be numerically equivalent to ``dequantize -> scaled_dot_product_attention`` on
the same stored (quantized) KV — it reads the identical packed bytes, only the
arithmetic order differs. We assert high cosine similarity + small max-abs (fp16
rounding), across bit widths, head dims, and GQA ratios, plus the applicability
guards and the fail-loud sink/mask behavior.
"""
import math

import mlx.core as mx
import pytest

from turboquant_mlx.layers.polar_kv_cache import (
    TurboQuantKVCache,
    enable_fused_attend,
    install_fused_attend_patch,
)


def _reference(cache, b_total, q, scale):
    """Shipped path: dequantize (with unrotate) then standard SDPA."""
    k_hat = cache._tq_dequantize(
        cache._tq_keys[0][..., :b_total, :], cache._tq_keys[1][..., :b_total, :],
        cache._k_signs, cache._k_block_size, cache._k_gs, cache._k_head_dim,
        cache._k_bits, cache._k_codebook_f16)
    v_hat = cache._tq_dequantize(
        cache._tq_values[0][..., :b_total, :], cache._tq_values[1][..., :b_total, :],
        cache._v_signs, cache._v_block_size, cache._v_gs, cache._v_head_dim,
        cache._v_bits, cache._v_codebook_f16)
    return mx.fast.scaled_dot_product_attention(q, k_hat, v_hat, scale=scale)


def _cos(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float((a * b).sum()
                 / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum())))


@pytest.mark.parametrize("k_bits,v_bits", [(8, 3), (8, 4), (4, 3), (3, 3)])
@pytest.mark.parametrize("n_q,n_kv,D", [(8, 8, 128), (32, 8, 128), (16, 2, 256)])
def test_fused_matches_dequant_sdpa(k_bits, v_bits, n_q, n_kv, D):
    S = 2048
    mx.random.seed(0)
    keys = mx.random.normal((1, n_kv, S, D)).astype(mx.float16)
    values = mx.random.normal((1, n_kv, S, D)).astype(mx.float16)
    q = mx.random.normal((1, n_q, 1, D)).astype(mx.float16)
    scale = 1.0 / math.sqrt(D)

    cache = TurboQuantKVCache(k_bits=k_bits, v_bits=v_bits,
                              group_size=min(64, D), min_tokens_before_quant=0)
    cache.update_and_fetch(keys, values)
    mx.eval(cache._tq_keys, cache._tq_values)

    ref = _reference(cache, cache.offset, q, scale)
    fused = cache.fused_attend(q, scale)
    mx.eval(ref, fused)

    assert _cos(ref, fused) > 0.9999
    max_abs = float(mx.max(mx.abs(ref.astype(mx.float32) - fused.astype(mx.float32))))
    rms = float(mx.sqrt(mx.mean(ref.astype(mx.float32) ** 2)))
    assert max_abs < 5e-3 * rms + 1e-3


def test_fused_respects_model_scale():
    """A non-default softmax scale must be honored (folded into q)."""
    S, D = 1024, 128
    mx.random.seed(1)
    keys = mx.random.normal((1, 4, S, D)).astype(mx.float16)
    values = mx.random.normal((1, 4, S, D)).astype(mx.float16)
    q = mx.random.normal((1, 4, 1, D)).astype(mx.float16)
    scale = 0.137  # deliberately not 1/sqrt(D)

    cache = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64,
                              min_tokens_before_quant=0)
    cache.update_and_fetch(keys, values)
    mx.eval(cache._tq_keys, cache._tq_values)
    ref = _reference(cache, cache.offset, q, scale)
    fused = cache.fused_attend(q, scale)
    mx.eval(ref, fused)
    assert _cos(ref, fused) > 0.9999


def test_applicability_guards():
    cache = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64,
                              min_tokens_before_quant=0)
    cache.update_and_fetch(
        mx.random.normal((1, 4, 16, 128)).astype(mx.float16),
        mx.random.normal((1, 4, 16, 128)).astype(mx.float16))
    cache._use_fused_attend = True
    # decode step, B=1 -> applicable
    assert cache._fused_applicable(B=1, num_steps=1)
    # prefill (num_steps>1) -> not applicable
    assert not cache._fused_applicable(B=1, num_steps=8)
    # batch > 1 -> not applicable
    assert not cache._fused_applicable(B=2, num_steps=1)
    # flag off -> not applicable
    cache._use_fused_attend = False
    assert not cache._fused_applicable(B=1, num_steps=1)
    # fp16 attention-sink window (min_tokens>0) -> not applicable
    c2 = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64,
                           min_tokens_before_quant=64)
    c2._use_fused_attend = True
    c2._k_head_dim = c2._v_head_dim = 128
    assert not c2._fused_applicable(B=1, num_steps=1)


def test_update_and_fetch_skips_dequant_when_fused():
    cache = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64,
                              min_tokens_before_quant=0)
    cache._use_fused_attend = True
    # prefill returns real K/V (fused not applicable for num_steps>1)
    k = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
    v = mx.random.normal((1, 4, 16, 128)).astype(mx.float16)
    ok_k, ok_v = cache.update_and_fetch(k, v)
    assert ok_k is not None and ok_v is not None
    # a decode step returns (None, None) — token stored, dequant skipped
    k1 = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    v1 = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    r_k, r_v = cache.update_and_fetch(k1, v1)
    assert r_k is None and r_v is None
    assert cache.offset == 17  # token was still stored


def test_patch_fails_loud_on_sinks():
    install_fused_attend_patch()
    import mlx_lm.models.base as base
    cache = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64,
                             min_tokens_before_quant=0)
    cache.update_and_fetch(
        mx.random.normal((1, 4, 8, 128)).astype(mx.float16),
        mx.random.normal((1, 4, 8, 128)).astype(mx.float16))
    q = mx.random.normal((1, 4, 1, 128)).astype(mx.float16)
    sinks = mx.zeros((4,), dtype=mx.float16)
    with pytest.raises(RuntimeError, match="attention sinks"):
        base.scaled_dot_product_attention(q, None, None, cache=cache,
                                          scale=1.0, mask=None, sinks=sinks)
