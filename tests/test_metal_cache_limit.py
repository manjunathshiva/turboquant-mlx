"""Tests for the serve Metal buffer-cache limit (--metal-cache-limit-gb).

The auto formula is what keeps a long chunked prefill alive on a 16 GB
machine serving a resident ~13 GB model: MLX's unbounded buffer reuse cache
grows ~80 MB per 1K prompt tokens during prefill (new, larger attention
shapes each chunk), which crossed the wired limit mid-prefill in the field
(hard Metal OOM at 14K/21K tokens). These tests pin the formula and the flag
parsing; the end-to-end behavior was validated with a 21K-token prefill probe
(uncapped: 16.5 GB total and climbing; capped at 256 MB: flat 13.5 GB).
"""

import pytest

from turboquant_mlx.serve import (
    _CACHE_LIMIT_AMPLE_HEADROOM_BYTES,
    _CACHE_LIMIT_MIN_BYTES,
    _CACHE_LIMIT_RESERVE_BYTES,
    _auto_metal_cache_limit,
    _extract_metal_cache_limit_args,
)

GB = 1024**3


class TestAutoFormula:
    def test_16gb_mini_case_floors_at_min(self):
        # Measured field numbers: wired cap 13824 MiB (13.5 GiB), resident
        # down4 build ~12.6 GiB -> headroom ~0.9 GiB, under the reserve, so
        # the cap floors at the minimum instead of going negative.
        limit = _auto_metal_cache_limit(int(13.5 * GB), int(12.6 * GB))
        assert limit == _CACHE_LIMIT_MIN_BYTES

    def test_roomy_machine_untouched(self):
        # 64 GB Mac: ~51.5 GB working set, 13.4 GB resident -> ample
        # headroom, keep MLX's unbounded default (fastest).
        assert _auto_metal_cache_limit(int(51.5 * GB), int(13.4 * GB)) is None

    def test_mid_headroom_caps_to_headroom_minus_reserve(self):
        wss, active = 20 * GB, 15 * GB
        limit = _auto_metal_cache_limit(wss, active)
        assert limit == (wss - active) - _CACHE_LIMIT_RESERVE_BYTES

    def test_boundary_exactly_ample_is_untouched(self):
        active = 10 * GB
        wss = active + _CACHE_LIMIT_AMPLE_HEADROOM_BYTES
        assert _auto_metal_cache_limit(wss, active) is None

    def test_just_under_ample_gets_capped(self):
        active = 10 * GB
        wss = active + _CACHE_LIMIT_AMPLE_HEADROOM_BYTES - 1
        limit = _auto_metal_cache_limit(wss, active)
        assert limit is not None
        assert limit >= _CACHE_LIMIT_MIN_BYTES


class TestFlagParsing:
    def test_default_is_auto(self):
        mode, remaining = _extract_metal_cache_limit_args(
            ["--model", "m", "--port", "8080"])
        assert mode == "auto"
        assert remaining == ["--model", "m", "--port", "8080"]

    def test_off_disables(self):
        mode, remaining = _extract_metal_cache_limit_args(
            ["--metal-cache-limit-gb", "off", "--model", "m"])
        assert mode is None
        assert remaining == ["--model", "m"]

    def test_explicit_gb(self):
        mode, _ = _extract_metal_cache_limit_args(
            ["--metal-cache-limit-gb", "1.5"])
        assert mode == 1.5

    def test_zero_means_no_buffer_reuse(self):
        mode, _ = _extract_metal_cache_limit_args(
            ["--metal-cache-limit-gb", "0"])
        assert mode == 0.0

    def test_garbage_errors_loud(self):
        with pytest.raises(SystemExit):
            _extract_metal_cache_limit_args(
                ["--metal-cache-limit-gb", "lots"])

    def test_negative_errors_loud(self):
        with pytest.raises(SystemExit):
            _extract_metal_cache_limit_args(
                ["--metal-cache-limit-gb", "-1"])
