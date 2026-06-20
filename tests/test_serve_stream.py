"""Tests for turboquant-serve expert-streaming flags.

Covers `_extract_stream_args`: `--cache-budget-gb` is the trigger that turns on
streaming; the Flash-MoE levers (`--max-active-experts`, `--use-page-cache` /
`--no-page-cache`) ride along; everything else forwards to mlx_lm.server
untouched. Also checks streaming + KV flags coexist (both off the same argv).
"""

from turboquant_mlx.serve import _extract_stream_args, _extract_kv_args


def test_no_budget_returns_none():
    sc, remaining = _extract_stream_args(["--model", "foo", "--port", "8080"])
    assert sc is None
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_budget_triggers_streaming_with_defaults():
    sc, remaining = _extract_stream_args(
        ["--model", "foo", "--cache-budget-gb", "4", "--port", "8080"])
    assert sc is not None
    assert sc["cache_budget_gb"] == 4.0
    assert sc["max_active_experts"] == 4       # K-reduction default
    assert sc["use_page_cache"] is None        # auto by model-size-vs-RAM
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_levers_peeled_and_forwarded():
    sc, remaining = _extract_stream_args(
        ["--model", "foo", "--cache-budget-gb", "8",
         "--max-active-experts", "6", "--no-page-cache"])
    assert sc["max_active_experts"] == 6
    assert sc["use_page_cache"] is False
    assert remaining == ["--model", "foo"]


def test_use_page_cache_forces_on():
    sc, _ = _extract_stream_args(["--cache-budget-gb", "4", "--use-page-cache"])
    assert sc["use_page_cache"] is True


def test_does_not_eat_server_flags_by_prefix():
    sc, remaining = _extract_stream_args(
        ["--cache-budget-gb", "4", "--max-tokens", "512"])
    assert sc is not None
    assert "--max-tokens" in remaining and "512" in remaining


def test_streaming_and_kv_flags_coexist():
    # main() peels KV first, then streaming off the remainder.
    kv, remaining = _extract_kv_args(
        ["--model", "foo", "--cache-budget-gb", "4",
         "--kv-k-bits", "8", "--kv-v-bits", "3", "--prompt-concurrency", "1"])
    sc, remaining = _extract_stream_args(remaining)
    assert kv["k_bits"] == 8 and kv["v_bits"] == 3
    assert sc["cache_budget_gb"] == 4.0
    assert remaining == ["--model", "foo", "--prompt-concurrency", "1"]
