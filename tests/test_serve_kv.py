"""Tests for turboquant-serve TurboQuant KV-cache flags.

Covers the wiring added so `turboquant-serve` (which wraps mlx_lm.server,
whose argparse has no native KV-quant flags) can compress each request's KV
cache:

1. Flag extraction: `--kv-*` flags are peeled off argv, the rest forwarded.
2. Flag validation: mutually-exclusive / paired flags are enforced.
3. Cache patch: `make_prompt_cache` in mlx_lm.server emits TurboQuantKVCache.
4. Integration: a TurboQuant cache survives the server's LRUPromptCache
   reuse machinery (nbytes / deepcopy / insert / fetch) without crashing.
"""

import copy

import mlx.core as mx
import pytest

from turboquant_mlx.serve import _extract_kv_args, _patch_kv_cache
from turboquant_mlx.layers.polar_kv_cache import (
    TurboQuantKVCache,
    make_turboquant_cache,
)


class _DummyModel:
    """Minimal stand-in: make_prompt_cache only needs `.layers` length."""

    def __init__(self, n_layers):
        self.layers = list(range(n_layers))


def _random_kv(batch=1, n_heads=4, seq=8, head_dim=128, seed=0):
    mx.random.seed(seed)
    k = mx.random.normal((batch, n_heads, seq, head_dim)).astype(mx.float16)
    v = mx.random.normal((batch, n_heads, seq, head_dim)).astype(mx.float16)
    mx.eval(k, v)
    return k, v


# ── Flag extraction ───────────────────────────────────────────────────────


def test_extract_no_kv_flags_returns_none():
    kv, remaining = _extract_kv_args(["--model", "foo", "--port", "8080"])
    assert kv is None
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_extract_kv_bits_peeled_off():
    kv, remaining = _extract_kv_args(
        ["--model", "foo", "--kv-bits", "3", "--port", "8080"]
    )
    assert kv is not None
    assert kv["tq_bits"] == 3
    assert kv["k_bits"] is None and kv["v_bits"] is None
    # KV flags removed; everything else forwarded untouched.
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_extract_mixed_precision_and_extras():
    kv, remaining = _extract_kv_args(
        [
            "--model", "foo",
            "--kv-k-bits", "8", "--kv-v-bits", "3",
            "--kv-min-tokens", "128", "--kv-group-size", "32",
        ]
    )
    assert kv["tq_bits"] is None
    assert kv["k_bits"] == 8 and kv["v_bits"] == 3
    assert kv["min_tokens_before_quant"] == 128
    assert kv["group_size"] == 32
    assert remaining == ["--model", "foo"]


def test_extract_defaults_when_only_bits_given():
    kv, _ = _extract_kv_args(["--kv-bits", "2"])
    assert kv["group_size"] == 64
    assert kv["min_tokens_before_quant"] == 0


def test_extract_does_not_eat_server_flags_by_prefix():
    # allow_abbrev=False: a real server flag must never be consumed.
    kv, remaining = _extract_kv_args(["--kv-bits", "3", "--max-tokens", "512"])
    assert kv is not None
    assert "--max-tokens" in remaining and "512" in remaining


def test_extract_rejects_bits_with_split():
    with pytest.raises(SystemExit):
        _extract_kv_args(["--kv-bits", "3", "--kv-k-bits", "8", "--kv-v-bits", "3"])


def test_extract_rejects_unpaired_split():
    with pytest.raises(SystemExit):
        _extract_kv_args(["--kv-k-bits", "8"])


# ── Cache patch ─────────────────────────────────────────────────────────────


def test_patch_make_prompt_cache_emits_turboquant():
    import mlx_lm.server as server_mod

    kv_config, _ = _extract_kv_args(["--kv-bits", "3"])
    orig = server_mod.make_prompt_cache
    try:
        _patch_kv_cache(kv_config)
        cache = server_mod.make_prompt_cache(_DummyModel(4))
        assert len(cache) == 4
        assert all(isinstance(c, TurboQuantKVCache) for c in cache)
        assert all(c._k_bits == 3 and c._v_bits == 3 for c in cache)
    finally:
        server_mod.make_prompt_cache = orig


def test_patch_make_prompt_cache_mixed_precision():
    import mlx_lm.server as server_mod

    kv_config, _ = _extract_kv_args(["--kv-k-bits", "8", "--kv-v-bits", "3"])
    orig = server_mod.make_prompt_cache
    try:
        _patch_kv_cache(kv_config)
        cache = server_mod.make_prompt_cache(_DummyModel(2))
        assert all(c._k_bits == 8 and c._v_bits == 3 for c in cache)
    finally:
        server_mod.make_prompt_cache = orig


def test_turboquant_cache_has_no_merge_so_server_serves_sequentially():
    # The batchability probe is `all(hasattr(c, "merge") ...)`; TurboQuant
    # caches must NOT have merge so the server falls back to single-stream.
    cache = make_turboquant_cache(_DummyModel(2), tq_bits=3)
    assert not any(hasattr(c, "merge") for c in cache)


# ── Integration with the server's LRUPromptCache ────────────────────────────


def test_turboquant_cache_survives_lru_prompt_cache():
    from mlx_lm.models.cache import LRUPromptCache

    cache = make_turboquant_cache(_DummyModel(2), tq_bits=3)
    k, v = _random_kv(seq=8)
    for c in cache:
        c.update_and_fetch(k, v)

    # nbytes must be defined (LRUPromptCache.insert_cache sums it) and > 0.
    total = sum(c.nbytes for c in cache)
    assert total > 0

    # fetch_nearest_cache deepcopies the stored cache — must not crash and
    # must preserve size.
    cp = copy.deepcopy(cache)
    assert sum(c.nbytes for c in cp) == total

    lru = LRUPromptCache(max_size=2)
    tokens = list(range(1, 9))
    lru.insert_cache("model-key", tokens, cache)
    assert len(lru) == 1
    assert lru.nbytes > 0

    got, rest = lru.fetch_nearest_cache("model-key", tokens)
    assert got is not None
    assert rest == []  # exact hit → whole prompt is cached
    assert all(isinstance(c, TurboQuantKVCache) for c in got)
