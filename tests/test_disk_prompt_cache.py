"""Tests for the disk-persistent prompt cache (turboquant-serve --disk-cache).

Covers:
1. Checkpoint file round-trip for every cache family we serve: standard
   ``KVCache``, non-trimmable ``ArraysCache`` (hybrid GDN/Mamba state), and
   ``TurboQuantKVCache`` (both with and without the fp16 sink tier, whose
   absent tensors are None inside ``state``).
2. ``TurboQuantKVCache.from_state`` — meta_state forms and post-restore
   updates (fp16 tier re-padded to the threshold width).
3. Store save policy: min-tokens gate, save-every stride along a lineage,
   strict-prefix supersede for trimmable checkpoints (and NOT for
   non-trimmable ones), LRU byte-budget eviction protecting the newest write.
4. The restore path through a patched ``LRUPromptCache``: a fresh cache
   instance (simulated server restart) resumes from the on-disk checkpoint —
   strict-prefix extension for any cache type, trim-back reuse only for
   trimmable caches, and no restore when the win is below the gain floor.
"""

import json

import mlx.core as mx
import pytest

from mlx_lm.models.cache import ArraysCache, KVCache, LRUPromptCache

from turboquant_mlx.disk_prompt_cache import (
    _MIN_GAIN_TOKENS,
    DiskPromptCache,
    load_cache_file,
    save_cache_file,
)
from turboquant_mlx.layers.polar_kv_cache import TurboQuantKVCache

MODEL_KEY = ("model-a", None, None)


def _kv_cache(n_tokens, dim=4):
    c = KVCache()
    c.update_and_fetch(
        mx.random.normal((1, 1, n_tokens, dim)),
        mx.random.normal((1, 1, n_tokens, dim)),
    )
    return c


def _arrays_cache():
    c = ArraysCache(2)
    c[0] = mx.random.normal((1, 4))
    c[1] = mx.random.normal((1, 3, 5))
    return c


def _tq_cache(n_tokens=16, min_tok=0):
    c = TurboQuantKVCache(k_bits=8, v_bits=3,
                          min_tokens_before_quant=min_tok)
    c.update_and_fetch(
        mx.random.normal((1, 2, n_tokens, 64)),
        mx.random.normal((1, 2, n_tokens, 64)),
    )
    return c


# ---------------------------------------------------------------------------
# 1+2. Checkpoint file round-trip
# ---------------------------------------------------------------------------

def test_round_trip_standard_and_tq(tmp_path):
    kv = _kv_cache(10)
    arr = _arrays_cache()
    tq_sink = _tq_cache(min_tok=4)   # fp16 tier present
    tq_flat = _tq_cache(min_tok=0)   # fp16 tier None inside state

    path = tmp_path / "ckpt.safetensors"
    save_cache_file(path, [kv, tq_sink, tq_flat, arr], {"tokens": "[1,2]"})
    loaded, meta = load_cache_file(path)
    lkv, ltq_sink, ltq_flat, larr = loaded

    assert meta["tokens"] == "[1,2]"
    assert [type(c).__name__ for c in loaded] == [
        "KVCache", "TurboQuantKVCache", "TurboQuantKVCache", "ArraysCache"
    ]

    assert lkv.offset == kv.offset
    assert mx.allclose(lkv.state[0], kv.state[0]).item()
    assert all(
        mx.allclose(x, y).item() for x, y in zip(larr.state, arr.state)
    )

    # TQ with sink tier: params, offset, packed payload, and the fp16 tier
    # re-padded to the full threshold width for later slice-assign appends.
    assert ltq_sink.offset == tq_sink.offset == 16
    assert ltq_sink._k_bits == 8 and ltq_sink._v_bits == 3
    assert ltq_sink._fp16_keys.shape[-2] == 4
    b_total = tq_sink.offset - 4
    assert mx.array_equal(
        ltq_sink._tq_keys[0], tq_sink._tq_keys[0][..., :b_total, :]
    ).item()
    assert mx.array_equal(
        ltq_sink._tq_values[1], tq_sink._tq_values[1][..., :b_total, :]
    ).item()

    # TQ without sink tier: absent fp16 tensors restore to None.
    assert ltq_flat._fp16_keys is None and ltq_flat._fp16_values is None
    assert mx.array_equal(
        ltq_flat._tq_keys[0], tq_flat._tq_keys[0][..., :16, :]
    ).item()

    # All restored caches must keep working (capacity/expansion paths).
    lkv.update_and_fetch(
        mx.random.normal((1, 1, 6, 4)), mx.random.normal((1, 1, 6, 4))
    )
    ltq_sink.update_and_fetch(
        mx.random.normal((1, 2, 8, 64)), mx.random.normal((1, 2, 8, 64))
    )
    assert lkv.offset == 16 and ltq_sink.offset == 24


def test_tq_from_state_meta_forms():
    tq = _tq_cache(min_tok=4)
    state = tq.state

    # 6-tuple (current)
    c6 = TurboQuantKVCache.from_state(state, tq.meta_state)
    assert (c6.offset, c6._k_bits, c6._v_bits,
            c6._min_tokens_before_quant) == (16, 8, 3, 4)

    # 5-tuple (item-#1 era, no threshold)
    c5 = TurboQuantKVCache.from_state(state, ("16", "8", "3", "64", "42"))
    assert c5._min_tokens_before_quant == 0

    # 4-tuple (legacy uniform bits)
    c4 = TurboQuantKVCache.from_state(state, ("16", "3", "64", "42"))
    assert c4._k_bits == c4._v_bits == 3

    with pytest.raises(ValueError):
        TurboQuantKVCache.from_state(state, ("1", "2"))


# ---------------------------------------------------------------------------
# 3. Save policy
# ---------------------------------------------------------------------------

def _store(tmp_path, **kw):
    kw.setdefault("min_tokens", 4)
    kw.setdefault("save_every", 100)
    kw.setdefault("sync", True)
    return DiskPromptCache(tmp_path, **kw)


def test_min_tokens_gate(tmp_path):
    store = _store(tmp_path, min_tokens=512)
    store.maybe_save(MODEL_KEY, list(range(100)), [_kv_cache(100)])
    assert len(store) == 0


def test_save_and_covered_skip(tmp_path):
    store = _store(tmp_path)
    tokens = list(range(600))
    store.maybe_save(MODEL_KEY, tokens, [_kv_cache(600)])
    assert len(store) == 1
    assert len(list(tmp_path.glob("*.safetensors"))) == 1

    # Same tokens again, and a strict prefix of the stored tokens: covered.
    store.maybe_save(MODEL_KEY, tokens, [_kv_cache(600)])
    store.maybe_save(MODEL_KEY, tokens[:400], [_kv_cache(400)])
    assert len(store) == 1


def test_stride_throttle_and_prefix_supersede(tmp_path):
    store = _store(tmp_path, save_every=100)
    tokens = list(range(600))
    store.maybe_save(MODEL_KEY, tokens, [_kv_cache(600)])

    # Extending the lineage by < save_every is throttled.
    store.maybe_save(MODEL_KEY, tokens + [7] * 50, [_kv_cache(650)])
    assert len(store) == 1

    # Extending by >= save_every writes, and the new trimmable checkpoint
    # supersedes its stored strict prefix.
    longer = tokens + [7] * 150
    store.maybe_save(MODEL_KEY, longer, [_kv_cache(750)])
    assert len(store) == 1
    files = list(tmp_path.glob("*.safetensors"))
    assert len(files) == 1
    loaded, meta = load_cache_file(files[0])
    assert json.loads(meta["tokens"]) == longer


def test_non_trimmable_keeps_prefixes(tmp_path):
    store = _store(tmp_path, save_every=100)
    tokens = list(range(600))
    store.maybe_save(MODEL_KEY, tokens, [_arrays_cache()])
    store.maybe_save(MODEL_KEY, tokens + [7] * 150, [_arrays_cache()])
    # A non-trimmable checkpoint can't be cut back, so the shorter prefix
    # stays as divergence insurance.
    assert len(store) == 2


def test_budget_eviction_protects_newest(tmp_path):
    store = _store(tmp_path, budget_gb=1e-6)  # ~1 KB: everything over budget
    a = list(range(600))
    b = [9] + list(range(599))  # different lineage
    store.maybe_save(MODEL_KEY, a, [_kv_cache(600)])
    store.maybe_save(MODEL_KEY, b, [_kv_cache(600)])
    # Both checkpoints exceed the budget; the older one is evicted, the
    # just-written one is protected.
    assert len(store) == 1
    files = list(tmp_path.glob("*.safetensors"))
    assert len(files) == 1
    _, meta = load_cache_file(files[0])
    assert json.loads(meta["tokens"]) == b


def test_background_writer_and_index_reload(tmp_path):
    # Async mode: payload is evaluated on the caller's thread (MLX graphs
    # can't be evaluated cross-thread), the writer thread only does file I/O.
    store = DiskPromptCache(tmp_path, min_tokens=4, sync=False)
    store.maybe_save(MODEL_KEY, list(range(600)), [_kv_cache(600)])
    store.flush()
    assert len(store) == 1

    # A new store on the same directory (simulated restart) sees the entry.
    store2 = _store(tmp_path)
    assert len(store2) == 1


def test_pending_saves_throttle_rapid_inserts(tmp_path):
    # The server checkpoints each prompt segment back-to-back; while the
    # first write is still queued the stride throttle must already see it,
    # or near-duplicate multi-GB files get written one token apart.
    import time as _time

    store = DiskPromptCache(tmp_path, min_tokens=4, save_every=100,
                            sync=False)
    orig_do_save = store._do_save

    def slow_do_save(*args):
        _time.sleep(0.3)
        orig_do_save(*args)

    store._do_save = slow_do_save
    tokens = list(range(600))
    store.maybe_save(MODEL_KEY, tokens, [_kv_cache(600)])
    store.maybe_save(MODEL_KEY, tokens + [7], [_kv_cache(601)])
    store.flush()
    assert len(store) == 1


def test_corrupt_index_starts_empty_and_prunes_orphans(tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "deadbeef.safetensors").write_bytes(b"orphan")
    store = _store(tmp_path)
    assert len(store) == 0
    assert not (tmp_path / "deadbeef.safetensors").exists()


# ---------------------------------------------------------------------------
# 4. Restore through a patched LRUPromptCache (simulated server restart)
# ---------------------------------------------------------------------------

@pytest.fixture
def installed_store(tmp_path):
    store = _store(tmp_path)
    store.install()
    try:
        yield store
    finally:
        store.uninstall()


def test_restart_restores_strict_prefix(installed_store):
    tokens = list(range(600))
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, tokens, [_kv_cache(600)])
    assert len(installed_store) == 1  # insert mirrored to disk

    # "Restart": a fresh in-memory cache knows nothing...
    lru2 = LRUPromptCache(max_size=10)
    suffix = [7] * 20
    cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, tokens + suffix)
    # ...but the disk checkpoint supplies the 600-token prefix.
    assert cache is not None
    assert rest == suffix
    assert cache[0].offset == 600


def test_restart_restores_non_trimmable_strict_prefix(installed_store):
    tokens = list(range(600))
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, tokens, [_arrays_cache()])

    lru2 = LRUPromptCache(max_size=10)
    cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, tokens + [7] * 20)
    assert cache is not None
    assert rest == [7] * 20


def test_divergent_checkpoint_trims_only_if_trimmable(installed_store):
    stored = list(range(600)) + [1, 2, 3]
    request = list(range(600)) + [8, 8, 8, 8]

    # Trimmable: the divergent checkpoint is loaded and trimmed back.
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, stored, [_kv_cache(603)])
    lru2 = LRUPromptCache(max_size=10)
    cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, request)
    assert cache is not None
    assert rest == request[600:]
    assert cache[0].offset == 600


def test_divergent_non_trimmable_not_restored(tmp_path):
    store = _store(tmp_path)
    store.install()
    try:
        stored = list(range(600)) + [1, 2, 3]
        request = list(range(600)) + [8, 8, 8, 8]
        lru1 = LRUPromptCache(max_size=10)
        lru1.insert_cache(MODEL_KEY, stored, [_arrays_cache()])
        lru2 = LRUPromptCache(max_size=10)
        cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, request)
        assert cache is None
        assert rest == request
    finally:
        store.uninstall()


def test_no_restore_below_gain_floor(installed_store):
    tokens = list(range(_MIN_GAIN_TOKENS // 2))
    # min_tokens=4 lets this checkpoint save, but restoring it can never win
    # more than _MIN_GAIN_TOKENS//2 tokens.
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, tokens, [_kv_cache(len(tokens))])
    assert len(installed_store) == 1

    lru2 = LRUPromptCache(max_size=10)
    cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, tokens + [7] * 10)
    assert cache is None
    assert rest == tokens + [7] * 10


def test_other_model_key_not_restored(installed_store):
    tokens = list(range(600))
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, tokens, [_kv_cache(600)])

    lru2 = LRUPromptCache(max_size=10)
    other = ("model-b", None, None)
    cache, rest = lru2.fetch_nearest_cache(other, tokens + [7] * 20)
    assert cache is None


def test_uninstall_restores_originals(tmp_path):
    orig_fetch = LRUPromptCache.fetch_nearest_cache
    orig_insert = LRUPromptCache.insert_cache
    store = _store(tmp_path)
    store.install()
    store.install()  # double-install must not capture our own wrappers
    assert LRUPromptCache.fetch_nearest_cache is not orig_fetch
    store.uninstall()
    assert LRUPromptCache.fetch_nearest_cache is orig_fetch
    assert LRUPromptCache.insert_cache is orig_insert


def test_tq_kv_cache_restores_through_lru(installed_store):
    tokens = list(range(600))
    lru1 = LRUPromptCache(max_size=10)
    lru1.insert_cache(MODEL_KEY, tokens, [_tq_cache(n_tokens=600, min_tok=4)])

    lru2 = LRUPromptCache(max_size=10)
    cache, rest = lru2.fetch_nearest_cache(MODEL_KEY, tokens + [7] * 20)
    assert cache is not None
    assert rest == [7] * 20
    assert isinstance(cache[0], TurboQuantKVCache)
    assert cache[0].offset == 600
