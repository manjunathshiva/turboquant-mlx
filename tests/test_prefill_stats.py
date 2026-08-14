"""Tests for the prefill-redundancy diagnostic (turboquant-serve).

Covers:
1. Flag extraction: `--prefill-stats` / `--prefill-stats-file` peeled off argv,
   the rest forwarded; off by default.
2. Accounting: PrefillStats.record correctly splits reused vs. fresh prefill
   and computes the prev-turn-recoverable count across the cases that matter
   (exact hit, append-only reuse, trim-blocked reuse, divergent prefix).
3. Aggregation: the session summary reports the right "fresh prefill a prefix
   cache would recover" total.
4. Integration: install() patches LRUPromptCache.fetch_nearest_cache and the
   wrapper records on a real (empty) cache without disturbing the return value.
"""

import io
import json


from turboquant_mlx.serve import _extract_prefill_stats_args
from turboquant_mlx.prefill_stats import (
    PrefillStats,
    _longest_common_prefix,
    install,
)


# ── Flag extraction ─────────────────────────────────────────────────────────


def test_extract_no_flags_returns_none():
    cfg, remaining = _extract_prefill_stats_args(["--model", "foo", "--port", "8080"])
    assert cfg is None
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_extract_prefill_stats_flag():
    cfg, remaining = _extract_prefill_stats_args(["--model", "foo", "--prefill-stats"])
    assert cfg == {"stats_file": None}
    assert remaining == ["--model", "foo"]


def test_extract_stats_file_implies_on():
    cfg, remaining = _extract_prefill_stats_args(
        ["--prefill-stats-file", "p.jsonl", "--model", "foo"]
    )
    assert cfg == {"stats_file": "p.jsonl"}
    assert remaining == ["--model", "foo"]


# ── LCP helper ──────────────────────────────────────────────────────────────


def test_lcp_basic():
    assert _longest_common_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert _longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert _longest_common_prefix([1, 2, 3], []) == 0
    assert _longest_common_prefix([9, 1], [1, 9]) == 0
    assert _longest_common_prefix([1, 2], [1, 2, 3, 4]) == 2


# ── Per-request accounting ──────────────────────────────────────────────────


def _stats():
    return PrefillStats(log=io.StringIO())


def test_exact_hit_all_reused():
    s = _stats()
    rec = s.record(tokens=list(range(10)), rest=[])
    assert rec["prompt_tokens"] == 10
    assert rec["reused"] == 10
    assert rec["fresh_prefill"] == 0
    assert rec["recoverable"] == 0


def test_first_request_all_fresh_nothing_recoverable():
    s = _stats()
    rec = s.record(tokens=list(range(100)), rest=list(range(100)))
    assert rec["reused"] == 0
    assert rec["fresh_prefill"] == 100
    # No previous turn → nothing was recoverable.
    assert rec["prev_lcp"] == 0
    assert rec["recoverable"] == 0


def test_append_only_reuse_already_handled():
    # Turn 1: whole prompt fresh. Turn 2: prompt = turn-1 prompt + 5 new
    # tokens, and the server reused the cached prefix (rest = just the 5 new).
    s = _stats()
    s.record(tokens=list(range(10)), rest=list(range(10)))
    rec = s.record(tokens=list(range(15)), rest=[10, 11, 12, 13, 14])
    assert rec["reused"] == 10
    assert rec["fresh_prefill"] == 5
    assert rec["prev_lcp"] == 10          # turn-1 prompt is a prefix of turn-2
    # Server already reused all 10 — a prefix cache buys nothing more here.
    assert rec["recoverable"] == 0


def test_trim_blocked_reuse_is_recoverable():
    # Same append-only growth, but the server re-prefilled the WHOLE prompt
    # (e.g. can_trim=False blocked reuse and no shorter entry matched).
    s = _stats()
    s.record(tokens=list(range(10)), rest=list(range(10)))
    rec = s.record(tokens=list(range(15)), rest=list(range(15)))
    assert rec["reused"] == 0
    assert rec["fresh_prefill"] == 15
    assert rec["prev_lcp"] == 10
    # Those 10 prefix tokens were needlessly re-prefilled → prefix cache prize.
    assert rec["recoverable"] == 10


def test_divergent_prefix_not_recoverable():
    # The first token differs (e.g. a dynamic system-prompt timestamp). A
    # prefix cache cannot help — recoverable must be 0 even though full fresh.
    s = _stats()
    s.record(tokens=[1, 2, 3, 4, 5], rest=[1, 2, 3, 4, 5])
    rec = s.record(tokens=[99, 2, 3, 4, 5, 6], rest=[99, 2, 3, 4, 5, 6])
    assert rec["prev_lcp"] == 0
    assert rec["recoverable"] == 0


# ── Aggregation / summary ───────────────────────────────────────────────────


def test_summary_totals_and_recoverable():
    s = _stats()
    s.record(tokens=list(range(10)), rest=list(range(10)))   # fresh 10, rec 0
    s.record(tokens=list(range(15)), rest=list(range(15)))   # fresh 15, rec 10
    out = s.summary()
    assert "2 requests" in out
    assert s.total_prompt == 25
    assert s.total_fresh == 25
    assert s.total_recoverable == 10
    # A working prefix cache would cut fresh prefill 25 -> 15.
    assert "25 -> 15" in out


def test_empty_summary():
    s = _stats()
    assert "no requests" in s.summary()


def test_jsonl_file_written(tmp_path):
    f = tmp_path / "prefill.jsonl"
    s = PrefillStats(stats_file=str(f), log=io.StringIO())
    s.record(tokens=list(range(10)), rest=list(range(10)))
    s.record(tokens=list(range(15)), rest=[10, 11, 12, 13, 14])
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2
    rec2 = json.loads(lines[1])
    assert rec2["req"] == 2
    assert rec2["reused"] == 10
    assert rec2["recoverable"] == 0


# ── Integration with the real LRUPromptCache ────────────────────────────────


def test_install_patches_and_records_without_changing_return():
    from mlx_lm.models.cache import LRUPromptCache

    orig = LRUPromptCache.fetch_nearest_cache
    try:
        stats = install(log=io.StringIO())
        lru = LRUPromptCache(max_size=2)
        # Empty trie → no reuse: (None, tokens). Wrapper must pass it through
        # unchanged and still record one request of all-fresh prefill.
        cache, rest = lru.fetch_nearest_cache("model-key", [1, 2, 3, 4])
        assert cache is None
        assert rest == [1, 2, 3, 4]
        assert stats.n_requests == 1
        assert stats.total_fresh == 4
        assert stats.total_reused == 0
    finally:
        LRUPromptCache.fetch_nearest_cache = orig
