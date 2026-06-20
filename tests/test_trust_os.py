"""Unit tests for the trust-OS page-cache auto-default (_auto_page_cache).

Contract: trust the OS page cache (return True) only when the model's
safetensors fit comfortably in RAM (< _PAGE_CACHE_RAM_FRACTION x total), else
fall back to F_NOCACHE (False) so a memory-constrained machine never thrashes.
Any uncertainty must fail safe to False.

Sizes are mocked (never write multi-GB files): patch glob + getsize so the
decision is a pure function of (model_bytes, ram_bytes).
"""

import turboquant_mlx.stream.loader as loader


def _patch(monkeypatch, model_bytes, ram_bytes):
    monkeypatch.setattr(loader, "_total_ram_bytes", lambda: ram_bytes)
    monkeypatch.setattr(loader.glob, "glob",
                        lambda pat: ["model-00001-of-00002.safetensors",
                                     "model-00002-of-00002.safetensors"])
    monkeypatch.setattr(loader.os.path, "getsize", lambda f: model_bytes // 2)


def test_fits_in_ram_trusts_os(monkeypatch):
    _patch(monkeypatch, model_bytes=20 * 10**9, ram_bytes=64 * 10**9)
    assert loader._auto_page_cache("x") is True


def test_larger_than_ram_uses_nocache(monkeypatch):
    _patch(monkeypatch, model_bytes=40 * 10**9, ram_bytes=16 * 10**9)
    assert loader._auto_page_cache("x") is False


def test_just_over_fraction_uses_nocache(monkeypatch):
    # 0.6 * 64 = 38.4 GB threshold; 40 GB is over -> F_NOCACHE.
    _patch(monkeypatch, model_bytes=40 * 10**9, ram_bytes=64 * 10**9)
    assert loader._auto_page_cache("x") is False


def test_no_files_fails_safe(monkeypatch):
    # special-char path / missing shards -> empty glob -> must NOT trust-OS on 0 bytes
    monkeypatch.setattr(loader, "_total_ram_bytes", lambda: 64 * 10**9)
    monkeypatch.setattr(loader.glob, "glob", lambda pat: [])
    assert loader._auto_page_cache("/weird/[path]") is False


def test_ram_probe_failure_fails_safe(monkeypatch):
    def boom():
        raise OSError("no ram info")
    monkeypatch.setattr(loader, "_total_ram_bytes", boom)
    monkeypatch.setattr(loader.glob, "glob", lambda pat: ["a.safetensors"])
    monkeypatch.setattr(loader.os.path, "getsize", lambda f: 10**9)
    assert loader._auto_page_cache("x") is False
