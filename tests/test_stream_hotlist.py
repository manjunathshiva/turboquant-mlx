"""Tests for the shipped hot-expert list (ds4 idea #6): startup preload of
pinned experts into the ExpertCache, plus the loader-side hotlist discovery
and rank-order-preserving pin-spec parsing.

Uses the FakeReader from test_stream_cache so no model or disk is involved.
"""

import json

import numpy as np

import turboquant_mlx.stream.streaming_switch as ss
from turboquant_mlx.stream.loader import _find_hotlist, _load_pin_spec
from turboquant_mlx.stream.streaming_switch import ExpertCache

from tests.test_stream_cache import FakeReader


def _specs(layers=(0, 1), experts=(3, 1, 5), projs=("gate", "up", "down")):
    """Hotlist-ordered specs: for each (layer, expert) pair, one spec per
    projection — the exact shape load_streaming feeds to preload()."""
    specs, keys = [], set()
    for l in layers:
        for e in experts:
            for proj in projs:
                wkey = f"L{l}.{proj}.weight"
                skey = f"L{l}.{proj}.scales"
                specs.append((wkey, skey, e))
                keys.add((wkey, e))
    return specs, keys


def test_preload_pins_everything_and_gather_never_misses():
    specs, keys = _specs()
    reader = FakeReader()
    cache = ExpertCache(reader, budget_bytes=10**9, prefetch_workers=1,
                        pin_keys=keys)
    n, dropped = cache.preload(specs)
    assert n == len(specs)
    assert dropped == 0
    assert len(cache._pinned) == len(specs)
    assert cache.preload_experts == len(specs)
    assert cache.preload_bytes > 0
    # gather stats stay clean: preload reads are not misses/coalescing stats
    assert cache.misses == 0
    assert cache.read_runs == 0 and cache.expert_reads == 0

    calls_before = reader.read_calls
    w, s = cache.gather("L0.gate.weight", "L0.gate.scales", [1, 3, 5])
    assert reader.read_calls == calls_before  # served entirely from memory
    assert cache.misses == 0
    assert cache._pin_hits == 3

    # byte-identity with a cold, unpinned cache
    cold = ExpertCache(FakeReader(), budget_bytes=10**9, prefetch_workers=1)
    w2, s2 = cold.gather("L0.gate.weight", "L0.gate.scales", [1, 3, 5])
    assert np.array_equal(np.array(w), np.array(w2))
    assert np.array_equal(np.array(s), np.array(s2))


def test_preload_budget_cap_keeps_hottest_and_unpins_rest(monkeypatch):
    # One expert-projection from FakeReader = 8 uint32 + 8 f16 = 48 bytes.
    # Budget 240 -> cap 144 -> 3 entries fit. Chunk of 1 spec makes the cap
    # check exact so the test is deterministic.
    monkeypatch.setattr(ss, "_PRELOAD_CHUNK", 1)
    specs, keys = _specs(layers=(0,), experts=(9, 2, 7, 4, 1, 0), projs=("gate",))
    cache = ExpertCache(FakeReader(), budget_bytes=240, prefetch_workers=1,
                        pin_keys=set(keys))
    n, dropped = cache.preload(specs)
    assert n == 3 and dropped == 3
    # rank order preserved: the first three (hottest) experts are pinned
    assert set(cache._pinned) == {("L0.gate.weight", e) for e in (9, 2, 7)}
    # the dropped ones are no longer pin keys — they age through the LRU
    for e in (4, 1, 0):
        assert ("L0.gate.weight", e) not in cache._pin_keys
    cache.gather("L0.gate.weight", "L0.gate.scales", [4])
    assert ("L0.gate.weight", 4) in cache._od


def test_preload_skips_already_resident():
    specs, keys = _specs(layers=(0,), experts=(1, 2), projs=("gate",))
    reader = FakeReader()
    cache = ExpertCache(reader, budget_bytes=10**9, prefetch_workers=1,
                        pin_keys=keys)
    n1, _ = cache.preload(specs)
    calls = reader.read_calls
    n2, _ = cache.preload(specs)  # second preload is a no-op
    assert n1 == 2 and n2 == 0
    assert reader.read_calls == calls


def test_find_hotlist(tmp_path):
    assert _find_hotlist(str(tmp_path)) is None
    hot = tmp_path / "hot_experts.json"
    hot.write_text(json.dumps({"pin": [[0, 3]]}))
    assert _find_hotlist(str(tmp_path)) == str(hot)


def test_load_pin_spec_preserves_rank_order_and_dedups(tmp_path):
    f = tmp_path / "pin.json"
    f.write_text(json.dumps({"pin": [[4, 17], [0, 3], [4, 17], [2, 9]]}))
    pin_layers, pin_order = _load_pin_spec(str(f))
    assert pin_order == [(4, 17), (0, 3), (2, 9)]  # file order, deduped
    assert pin_layers == {4: {17}, 0: {3}, 2: {9}}


def test_preload_stats_exposed():
    specs, keys = _specs(layers=(0,), experts=(1,), projs=("gate",))
    cache = ExpertCache(FakeReader(), budget_bytes=10**9, prefetch_workers=1,
                        pin_keys=keys)
    cache.preload(specs)
    st = cache.stats()
    assert st["preload_experts"] == 1
    assert st["preload_gb"] > 0
    assert st["pinned_experts"] == 1
