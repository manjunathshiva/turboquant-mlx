"""Tests for TurboQuantKVCache mixed-precision K/V (v0.2 item #1)."""


import mlx.core as mx
import pytest

from turboquant_mlx.layers.polar_kv_cache import (
    TurboQuantKVCache,
    make_turboquant_cache,
)


def _random_kv(batch=1, n_heads=4, seq=8, head_dim=128, seed=0):
    mx.random.seed(seed)
    k = mx.random.normal((batch, n_heads, seq, head_dim)).astype(mx.float16)
    v = mx.random.normal((batch, n_heads, seq, head_dim)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


def _arrays_equal(a, b):
    return bool((a == b).all().item())


# ── Constructor argument validation ──────────────────────────────────────


def test_constructor_legacy_tq_bits_default():
    c = TurboQuantKVCache()
    assert c._k_bits == 3
    assert c._v_bits == 3


def test_constructor_legacy_tq_bits_explicit():
    c = TurboQuantKVCache(tq_bits=4)
    assert c._k_bits == 4
    assert c._v_bits == 4


def test_constructor_mixed_precision():
    c = TurboQuantKVCache(k_bits=8, v_bits=3)
    assert c._k_bits == 8
    assert c._v_bits == 3


def test_constructor_rejects_both_forms():
    with pytest.raises(ValueError, match="Pass either tq_bits"):
        TurboQuantKVCache(tq_bits=3, k_bits=4, v_bits=3)


def test_constructor_rejects_partial_mixed():
    with pytest.raises(ValueError, match="must be set together"):
        TurboQuantKVCache(k_bits=8)
    with pytest.raises(ValueError, match="must be set together"):
        TurboQuantKVCache(v_bits=3)


# ── Backwards compatibility: tq_bits=N == (k_bits=N, v_bits=N) ───────────


def test_legacy_and_explicit_bits_match():
    """Symmetric (k=v=3) must produce byte-identical output to tq_bits=3."""
    k, v = _random_kv(seed=42)

    legacy = TurboQuantKVCache(tq_bits=3, group_size=64, seed=99)
    explicit = TurboQuantKVCache(k_bits=3, v_bits=3, group_size=64, seed=99)

    k_legacy, v_legacy = legacy.update_and_fetch(k, v)
    k_explicit, v_explicit = explicit.update_and_fetch(k, v)
    mx.eval(k_legacy, v_legacy, k_explicit, v_explicit)

    assert _arrays_equal(k_legacy, k_explicit)
    assert _arrays_equal(v_legacy, v_explicit)


def test_back_compat_bits_property():
    """Legacy code reading ``cache._bits`` should still see a value."""
    sym = TurboQuantKVCache(tq_bits=3)
    assert sym._bits == 3
    mixed = TurboQuantKVCache(k_bits=8, v_bits=3)
    # Mixed: returns K bits as the precision-critical lane
    assert mixed._bits == 8


# ── Mixed precision behavior ─────────────────────────────────────────────


def test_mixed_precision_round_trip_shapes():
    """K8+V3 round-trip produces correct shapes."""
    k, v = _random_kv(seed=1)
    c = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64, seed=7)
    k_deq, v_deq = c.update_and_fetch(k, v)
    mx.eval(k_deq, v_deq)
    assert k_deq.shape == k.shape
    assert v_deq.shape == v.shape


def test_mixed_precision_storage_uses_separate_widths():
    """K8 and V3 should produce different packed-dim storage shapes."""
    k, v = _random_kv(seed=2)
    c = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64, seed=7)
    c.update_and_fetch(k, v)

    head_dim = k.shape[-1]
    expected_k_packed = (head_dim + (32 // 8) - 1) // (32 // 8)  # = 32
    expected_v_packed = (head_dim + (32 // 3) - 1) // (32 // 3)  # = 13
    assert c._tq_keys[0].shape[-1] == expected_k_packed
    assert c._tq_values[0].shape[-1] == expected_v_packed


def test_higher_kbits_lower_quantization_error():
    """K8 must reconstruct keys more accurately than K3 on the same input."""
    k, v = _random_kv(seed=3, seq=16)

    c_k3 = TurboQuantKVCache(k_bits=3, v_bits=3, group_size=64, seed=11)
    c_k8 = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=64, seed=11)
    k_deq3, _ = c_k3.update_and_fetch(k, v)
    k_deq8, _ = c_k8.update_and_fetch(k, v)
    mx.eval(k_deq3, k_deq8)

    err3 = float(mx.mean((k.astype(mx.float32) - k_deq3.astype(mx.float32)) ** 2).item())
    err8 = float(mx.mean((k.astype(mx.float32) - k_deq8.astype(mx.float32)) ** 2).item())
    assert err8 < err3, f"Expected K8 MSE ({err8}) < K3 MSE ({err3})"


# ── State / meta_state round-trip ────────────────────────────────────────


def test_meta_state_v02_round_trip():
    """6-tuple meta_state preserves K/V bits + threshold."""
    c = TurboQuantKVCache(k_bits=8, v_bits=3, group_size=32, seed=123)
    k, v = _random_kv(seed=4)
    c.update_and_fetch(k, v)

    meta = c.meta_state
    assert len(meta) == 6
    offset_str, k_bits_str, v_bits_str, gs_str, seed_str, thr_str = meta
    assert int(offset_str) == 8
    assert int(k_bits_str) == 8
    assert int(v_bits_str) == 3
    assert int(gs_str) == 32
    assert int(seed_str) == 123
    assert int(thr_str) == 0

    c2 = TurboQuantKVCache(tq_bits=2)  # any bits — overwritten by setter
    c2.meta_state = meta
    assert c2._k_bits == 8
    assert c2._v_bits == 3
    assert c2.offset == 8
    assert c2._min_tokens_before_quant == 0


def test_meta_state_legacy_4tuple_compat():
    """A 4-tuple from v0.1.x still loads (treated as symmetric)."""
    c = TurboQuantKVCache(tq_bits=4)
    legacy_meta = ("16", "3", "64", "42")  # offset, bits, gs, seed
    c.meta_state = legacy_meta
    assert c._k_bits == 3
    assert c._v_bits == 3
    assert c.offset == 16
    assert c._group_size == 64
    assert c._seed == 42


# ── Factory helpers accept new kwargs ────────────────────────────────────


def test_make_turboquant_cache_accepts_mixed():
    class _StubModel:
        layers = [object(), object(), object()]

    caches = make_turboquant_cache(_StubModel(), k_bits=8, v_bits=3, group_size=64)
    assert len(caches) == 3
    for c in caches:
        assert c._k_bits == 8
        assert c._v_bits == 3


def test_make_turboquant_cache_legacy_default():
    class _StubModel:
        layers = [object()]

    caches = make_turboquant_cache(_StubModel())
    assert caches[0]._k_bits == 3
    assert caches[0]._v_bits == 3


# ── min_tokens_before_quant threshold (v0.2 item #2) ─────────────────────


def test_threshold_zero_is_backwards_compatible():
    """threshold=0 must produce byte-identical output to v0.1.x default."""
    k, v = _random_kv(seed=10)

    legacy = TurboQuantKVCache(tq_bits=3, group_size=64, seed=42)
    new = TurboQuantKVCache(
        tq_bits=3, group_size=64, seed=42, min_tokens_before_quant=0
    )

    k1, v1 = legacy.update_and_fetch(k, v)
    k2, v2 = new.update_and_fetch(k, v)
    mx.eval(k1, v1, k2, v2)

    assert _arrays_equal(k1, k2)
    assert _arrays_equal(v1, v2)
    assert legacy.empty() is False
    assert new.empty() is False
    assert new._fp16_keys is None  # No Tier A allocated when threshold=0


def test_threshold_preserves_first_tokens_bitwise():
    """First N tokens should round-trip with zero error when threshold=N."""
    k, v = _random_kv(seed=11, seq=32)
    threshold = 8

    c = TurboQuantKVCache(
        tq_bits=3, group_size=64, seed=42,
        min_tokens_before_quant=threshold,
    )
    k_fetch, v_fetch = c.update_and_fetch(k, v)
    mx.eval(k_fetch, v_fetch)

    # Tier A (first 8 tokens): bitwise identical
    assert _arrays_equal(k_fetch[..., :threshold, :], k[..., :threshold, :])
    assert _arrays_equal(v_fetch[..., :threshold, :], v[..., :threshold, :])

    # Tier B (remaining 24): not identical (lossy quantization)
    assert not _arrays_equal(k_fetch[..., threshold:, :], k[..., threshold:, :])
    assert c.offset == 32


def test_threshold_smaller_than_seq_splits_batch():
    """A single batch crossing the boundary should split correctly."""
    k, v = _random_kv(seed=12, seq=20)
    threshold = 5

    c = TurboQuantKVCache(
        tq_bits=3, min_tokens_before_quant=threshold,
    )
    k_fetch, v_fetch = c.update_and_fetch(k, v)
    mx.eval(k_fetch, v_fetch)

    # Tier A populated up to threshold
    assert c._fp16_keys.shape[-2] == threshold
    # Tier B holds the remaining 15
    assert c._tq_keys is not None
    # Output sequence length matches input
    assert k_fetch.shape[-2] == 20
    # First 5 tokens bitwise identical
    assert _arrays_equal(k_fetch[..., :threshold, :], k[..., :threshold, :])


def test_threshold_multiple_batches_below_then_crossing():
    """Multiple appends: stay in Tier A, then cross into Tier B."""
    threshold = 10

    c = TurboQuantKVCache(
        tq_bits=3, min_tokens_before_quant=threshold,
    )

    # First batch: 4 tokens, all in Tier A
    k1, v1 = _random_kv(seed=20, seq=4)
    out_k1, out_v1 = c.update_and_fetch(k1, v1)
    mx.eval(out_k1, out_v1)
    assert c.offset == 4
    assert c._tq_keys is None  # No Tier B yet
    assert _arrays_equal(out_k1, k1)

    # Second batch: 4 more tokens, still in Tier A (offset 4→8)
    k2, v2 = _random_kv(seed=21, seq=4)
    out_k2, out_v2 = c.update_and_fetch(k2, v2)
    mx.eval(out_k2, out_v2)
    assert c.offset == 8
    assert c._tq_keys is None

    # Third batch: 6 tokens, crosses boundary at offset 10 (2 to A, 4 to B)
    k3, v3 = _random_kv(seed=22, seq=6)
    out_k3, out_v3 = c.update_and_fetch(k3, v3)
    mx.eval(out_k3, out_v3)
    assert c.offset == 14
    assert c._tq_keys is not None  # Tier B now populated
    assert out_k3.shape[-2] == 14


def test_threshold_meta_state_round_trip():
    c = TurboQuantKVCache(
        k_bits=8, v_bits=3, group_size=32, seed=7,
        min_tokens_before_quant=128,
    )
    k, v = _random_kv(seed=30)
    c.update_and_fetch(k, v)

    meta = c.meta_state
    assert len(meta) == 6
    assert int(meta[5]) == 128

    c2 = TurboQuantKVCache(tq_bits=2)
    c2.meta_state = meta
    assert c2._min_tokens_before_quant == 128
    assert c2._k_bits == 8
    assert c2._v_bits == 3


def test_threshold_meta_state_5tuple_legacy_compat():
    """A 5-tuple from item-#1 still loads (min_tokens defaults unchanged)."""
    c = TurboQuantKVCache(
        tq_bits=3, min_tokens_before_quant=64,
    )
    legacy_meta = ("16", "3", "3", "64", "42")  # offset, k, v, gs, seed
    c.meta_state = legacy_meta
    assert c._min_tokens_before_quant == 64  # unchanged by 5-tuple setter
    assert c.offset == 16


def test_threshold_factory_kwarg():
    class _StubModel:
        layers = [object(), object()]

    caches = make_turboquant_cache(
        _StubModel(), tq_bits=3, min_tokens_before_quant=512,
    )
    for c in caches:
        assert c._min_tokens_before_quant == 512


def test_threshold_rejects_negative():
    with pytest.raises(ValueError, match="must be >= 0"):
        TurboQuantKVCache(tq_bits=3, min_tokens_before_quant=-1)


def test_threshold_nbytes_accounts_both_tiers():
    """nbytes should reflect Tier A fp16 + Tier B compressed storage."""
    threshold = 8
    c = TurboQuantKVCache(
        tq_bits=3, min_tokens_before_quant=threshold,
    )

    # First batch: 4 tokens, only Tier A
    k1, v1 = _random_kv(seed=40, seq=4)
    c.update_and_fetch(k1, v1)
    bytes_after_a_only = c.nbytes
    assert bytes_after_a_only > 0

    # Second batch: 8 more tokens (4 fill A, 4 go to B)
    k2, v2 = _random_kv(seed=41, seq=8)
    c.update_and_fetch(k2, v2)
    bytes_after_both = c.nbytes
    assert bytes_after_both > bytes_after_a_only
