"""Tests for the serve prompt-cache byte cap (--prompt-cache-max-gb).

Field failure it guards: an agent-harness conversation retains one KV state
per turn in mlx_lm.server's LRU prompt cache (entry-count-bounded only);
8 sequences / 1.14 GB on a 16 GB mini paged the system until the Metal
watchdog killed a stalled prefill command buffer (GPU Timeout).
"""

import mlx.core as mx
import pytest

import mlx_lm.server as server_mod
from mlx_lm.models.cache import KVCache, LRUPromptCache

from turboquant_mlx.serve import (
    _CACHE_LIMIT_AMPLE_HEADROOM_BYTES,
    _PROMPT_CACHE_MIN_BYTES,
    _PROMPT_CACHE_RESERVE_BYTES,
    _auto_prompt_cache_bytes,
    _extract_prompt_cache_max_args,
    _patch_prompt_cache_bytes,
)

GB = 1024**3
MODEL_KEY = ("model-a", None, None)


def _kv_cache(n_tokens, dim=64):
    c = KVCache()
    c.update_and_fetch(
        mx.random.normal((1, 1, n_tokens, dim)),
        mx.random.normal((1, 1, n_tokens, dim)),
    )
    return c


@pytest.fixture
def restore_lru():
    orig = server_mod.LRUPromptCache
    try:
        yield
    finally:
        server_mod.LRUPromptCache = orig


class TestAutoFormula:
    def test_tight_machine_floors_at_min(self):
        # Mini field numbers: wss 14.5 GB, weights+working ~12.6 GiB.
        limit = _auto_prompt_cache_bytes(int(13.5 * GB), int(12.6 * GB))
        assert limit == _PROMPT_CACHE_MIN_BYTES

    def test_roomy_machine_unbounded(self):
        assert _auto_prompt_cache_bytes(int(51.5 * GB), int(13.4 * GB)) is None

    def test_mid_headroom(self):
        wss, active = 20 * GB, 15 * GB
        assert _auto_prompt_cache_bytes(wss, active) == \
            (wss - active) - _PROMPT_CACHE_RESERVE_BYTES

    def test_ample_boundary(self):
        active = 10 * GB
        assert _auto_prompt_cache_bytes(
            active + _CACHE_LIMIT_AMPLE_HEADROOM_BYTES, active) is None


class TestPatchedLRU:
    def test_explicit_cap_evicts_lru(self, restore_lru):
        _patch_prompt_cache_bytes(1e-6)  # ~1 KB budget
        lru = server_mod.LRUPromptCache(max_size=10)
        assert isinstance(lru, LRUPromptCache)
        lru.insert_cache(MODEL_KEY, list(range(100)), [_kv_cache(100)])
        lru.insert_cache(MODEL_KEY, list(range(200, 300)), [_kv_cache(100)])
        # Each entry is ~50 KB >> the 1 KB budget: everything evicts down.
        assert lru.nbytes <= 1e-6 * GB or len(lru) <= 1

    def test_ctor_default_does_not_override_property(self, restore_lru):
        _patch_prompt_cache_bytes(2.0)
        lru = server_mod.LRUPromptCache(max_size=10)
        # Base __init__ assigned max_bytes=1<<63; the property must win.
        assert lru.max_bytes == int(2.0 * GB)

    def test_auto_mode_reads_device_state(self, restore_lru):
        _patch_prompt_cache_bytes("auto")
        lru = server_mod.LRUPromptCache(max_size=10)
        # Just exercises the dynamic path end-to-end on this machine.
        assert lru.max_bytes > 0
        lru.insert_cache(MODEL_KEY, list(range(50)), [_kv_cache(50)])
        assert len(lru) >= 0


class TestFlagParsing:
    def test_default_auto(self):
        mode, remaining = _extract_prompt_cache_max_args(["--model", "m"])
        assert mode == "auto"
        assert remaining == ["--model", "m"]

    def test_off(self):
        mode, _ = _extract_prompt_cache_max_args(
            ["--prompt-cache-max-gb", "off"])
        assert mode is None

    def test_explicit(self):
        mode, _ = _extract_prompt_cache_max_args(
            ["--prompt-cache-max-gb", "0.5"])
        assert mode == 0.5

    def test_garbage_and_nonpositive_error(self):
        with pytest.raises(SystemExit):
            _extract_prompt_cache_max_args(["--prompt-cache-max-gb", "much"])
        with pytest.raises(SystemExit):
            _extract_prompt_cache_max_args(["--prompt-cache-max-gb", "0"])
