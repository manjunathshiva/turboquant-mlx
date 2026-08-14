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


class FailingReader(FakeReader):
    """FakeReader that raises on selected (key, expert) reads."""

    def __init__(self, fail_keys=(), **kw):
        super().__init__(**kw)
        self.fail = set(fail_keys)

    def read_expert_np(self, key: str, expert: int):
        if (key, expert) in self.fail:
            raise OSError("injected disk error")
        return super().read_expert_np(key, expert)


def test_fanout_skips_trigger_and_counts_as_critical_path():
    # Same-layer fan-out must (a) skip the trigger projection — its gather
    # runs next on the calling thread and keeps the coalesced read path —
    # and (b) account pool-read claims as misses + fanout_hits, never as
    # prefetch_hits, so hit_rate keeps meaning "no same-token disk read".
    cache = ExpertCache(FakeReader(), budget_bytes=10**9, prefetch_workers=4,
                        fanout=True)
    cache.register_layer(0, [("L0.up.weight", "L0.up.scales"),
                             ("L0.gate.weight", "L0.gate.scales"),
                             ("L0.down.weight", "L0.down.scales")])
    cache.on_layer_start(0, [1, 2], trigger_wkey="L0.up.weight")
    with cache._lock:
        assert ("L0.up.weight", 1) not in cache._inflight
        assert ("L0.up.weight", 1) not in cache._staging
    cache.gather("L0.up.weight", "L0.up.scales", [1, 2])
    assert cache.misses == 2 and cache.fanout_hits == 0
    cache.gather("L0.gate.weight", "L0.gate.scales", [1, 2])
    cache.gather("L0.down.weight", "L0.down.scales", [1, 2])
    assert cache.misses == 6
    assert cache.fanout_hits == 4
    assert cache.prefetch_hits == 0
    assert cache.bytes_prefetched == 0 and cache.bytes_read > 0
    assert cache.stats()["hit_rate"] == 0.0


def test_fanout_is_off_unless_asked_for():
    # Fan-out is opt-in (--fanout): it trades the coalesced serial read path
    # for parallelism, and nothing in-process can tell a fast internal SSD
    # from a saturated external bus. Default must be the pre-fan-out path.
    for kwargs in ({}, {"fanout": False}):
        cache = ExpertCache(FakeReader(), budget_bytes=10**9,
                            prefetch_workers=4, **kwargs)
        assert cache._fanout is False
        cache.register_layer(0, [("a.weight", "a.scales"),
                                 ("b.weight", "b.scales")])
        cache.on_layer_start(0, [0, 1], trigger_wkey="a.weight")
        with cache._lock:
            assert not cache._inflight and not cache._staging
        cache.gather("b.weight", "b.scales", [0, 1])
        assert cache.fanout_hits == 0 and cache.misses == 2


def test_throttle_disables_fanout_too():
    # When the saturation throttle judges storage bandwidth-bound it must
    # stop BOTH speculation and the same-layer fan-out from competing for the
    # bus. This only reaches fan-out while speculation is also running: the
    # throttle judges by rescue rate and fan-out has none, which is exactly
    # why fan-out is opt-in rather than on-with-a-safety-net.
    cache = ExpertCache(FakeReader(), budget_bytes=10**9,
                        prefetch_workers=2, prefetch_ahead=1, fanout=True)
    cache.prefetched = cache._throttle_warmup
    cache.misses = 1000  # rescue rate 0%
    cache.on_layer_start(0, [], trigger_wkey=None)
    assert cache._prefetch_ahead == 0
    assert cache._fanout is False


def test_throttle_cannot_reach_fanout_without_speculation():
    # Documents the gap the opt-in default exists for: with --prefetch-ahead 0
    # (the default) the throttle block never evaluates, so an enabled fan-out
    # runs for the whole session no matter how bandwidth-bound the storage is.
    cache = ExpertCache(FakeReader(), budget_bytes=10**9,
                        prefetch_workers=2, prefetch_ahead=0, fanout=True)
    cache.misses = 10**6  # as bandwidth-bound as it gets
    cache.register_layer(0, [("a.weight", "a.scales")])
    cache.on_layer_start(0, [0], trigger_wkey=None)
    assert cache._fanout is True
    assert cache._throttle_decided is False


def test_background_read_error_counted_and_recovered():
    # A failed background read must be visible (prefetch_errors) and must
    # fall back to a synchronous load with correct content, not hang or lose
    # the expert.
    rd = FailingReader(fail_keys=[("b.weight", 2)])
    cache = ExpertCache(rd, budget_bytes=10**9, prefetch_workers=1)
    cache._prefetch_one("b.weight", "b.scales", 2)  # direct: deterministic
    assert cache.prefetch_errors == 1
    assert cache.stats()["prefetch_errors"] == 1
    rd.fail.clear()
    w, _ = cache.gather("b.weight", "b.scales", [2])
    assert cache.misses == 1
    assert int(np.array(w)[0][0]) == 20  # expert 2 -> 2*10 at column 0


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
