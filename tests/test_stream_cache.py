"""Unit tests for the streaming ExpertCache parallel prefetch.

These exercise the cache logic with a fake reader (no model, no disk), so they
run in CI in milliseconds. The invariant that protects the feature: parallel
prefetch is *byte-identical* to the serial baseline.
"""

import numpy as np
import mlx.core as mx

from turboquant_mlx.stream.streaming_switch import ExpertCache


class FakeReader:
    """Deterministic stand-in for SafetensorsExpertReader.

    weight slices are uint32, scales slices are float16; each expert's content
    is a unique function of (key, expert) so byte-equality is meaningful.
    """

    def __init__(self, cols: int = 8):
        self.cols = cols
        self.read_calls = 0

    def _content(self, key: str, expert: int):
        if "scales" in key:
            return (np.arange(self.cols, dtype=np.float16) + expert), mx.float16
        return (np.arange(self.cols, dtype=np.uint32) + expert * 10), mx.uint32

    def read_expert_np(self, key: str, expert: int):
        self.read_calls += 1
        return self._content(key, expert)

    def read_range_np(self, key: str, e_start: int, count: int):
        # Coalesced read of `count` consecutive experts in one call — mirrors
        # SafetensorsExpertReader.read_range_np: returns (count, *rest), dtype.
        # Stacking _content keeps it byte-identical to the per-expert path.
        self.read_calls += 1
        rows = [self._content(key, e_start + i)[0] for i in range(count)]
        _, mlx_dt = self._content(key, e_start)
        return np.stack(rows), mlx_dt

    def read_expert(self, key: str, expert: int):
        buf, dt = self._content(key, expert)
        return mx.array(buf, dtype=dt)


def _gather_np(workers, experts):
    cache = ExpertCache(FakeReader(), budget_bytes=10**9, prefetch_workers=workers)
    w, s = cache.gather("L0.gate.weight", "L0.gate.scales", experts)
    return np.array(w), np.array(s)


def test_parallel_matches_serial():
    w1, s1 = _gather_np(1, [3, 1, 2, 7, 0])
    w8, s8 = _gather_np(8, [3, 1, 2, 7, 0])
    assert np.array_equal(w1, w8)
    assert np.array_equal(s1, s8)


def test_gather_preserves_expert_order():
    # weight content at column 0 is expert*10, so the stack must echo the
    # requested order regardless of parallel scheduling.
    w, _ = _gather_np(4, [5, 0, 2])
    assert [int(w[i][0]) for i in range(3)] == [50, 0, 20]


def test_hit_miss_accounting():
    cache = ExpertCache(FakeReader(), budget_bytes=10**9, prefetch_workers=1)
    cache.gather("k.weight", "k.scales", [0, 1])
    assert cache.misses == 2 and cache.hits == 0
    cache.gather("k.weight", "k.scales", [0, 1])
    assert cache.hits == 2 and cache.misses == 2


def test_eviction_respects_budget():
    cache = ExpertCache(FakeReader(), budget_bytes=200, prefetch_workers=1)
    for e in range(40):
        cache.gather("k.weight", "k.scales", [e])
    # never grows unbounded: resident bytes stay within ~one entry of budget.
    assert cache.cur <= 200 + 48


def test_prefetch_staging_hit():
    # A staged expert (weight+scales held under one key) is served from the
    # staging area without a critical-path miss, and its content matches a
    # direct read. Guards the single-key staging structure (no orphaned halves).
    cache = ExpertCache(FakeReader(), budget_bytes=10**9,
                        prefetch_workers=1, prefetch_ahead=1)
    cache._prefetch_one("L0.gate.weight", "L0.gate.scales", 2)
    assert ("L0.gate.weight", 2) in cache._staging  # one entry, not two
    w, _ = cache.gather("L0.gate.weight", "L0.gate.scales", [2])
    assert cache.prefetch_hits == 1 and cache.misses == 0
    assert int(np.array(w)[0][0]) == 20  # expert 2 -> 2*10 at column 0
    assert cache._staging_bytes == 0  # claimed, not orphaned
