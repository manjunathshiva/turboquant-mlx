"""`turboquant-serve --help` lists TurboQuant's own options next to mlx-lm's.

The TurboQuant options are peeled off argv by permissive pre-parsers before
mlx_lm.server parses the rest, so mlx-lm's help alone never showed them. The
help printer and the pre-parsers share the same ``_add_*_args`` functions.
"""

import argparse
import re
import sys

import pytest

from turboquant_mlx import serve


def _help_text(monkeypatch, capsys, flag="--help"):
    monkeypatch.setattr(sys, "argv", ["turboquant-serve", flag])
    serve.main()
    return capsys.readouterr().out


def _options(add_args):
    parser = argparse.ArgumentParser(add_help=False)
    add_args(parser)
    return [s for a in parser._actions for s in a.option_strings]


def _listed(option, text):
    """True when ``option`` starts an entry in the options listing."""
    return re.search(rf"^  {re.escape(option)}(\s|,|$)", text, re.M) is not None


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_lists_every_turboquant_option(monkeypatch, capsys, flag):
    text = _help_text(monkeypatch, capsys, flag)
    adders = [getattr(serve, name) for name in dir(serve)
              if name.startswith("_add_") and name.endswith("_args")]
    assert adders
    for add_args in adders:
        for option in _options(add_args):
            assert _listed(option, text), option
    for _, _, add_args in serve._OPTION_GROUPS:
        assert _options(add_args)
    # mlx-lm's own options are still listed.
    assert _listed("--model", text)
    assert _listed("--prompt-concurrency", text)


def test_help_replaces_mlx_lm_kv_options(monkeypatch, capsys):
    text = _help_text(monkeypatch, capsys)
    assert len(re.findall(r"^  --kv-bits ", text, re.M)) == 1
    assert len(re.findall(r"^  --kv-group-size ", text, re.M)) == 1
    assert "Rotation group size for KV quantization" in text
    assert text.startswith("usage: turboquant-serve ")


def test_help_restores_argparse(monkeypatch, capsys):
    original = argparse.ArgumentParser.parse_args
    _help_text(monkeypatch, capsys)
    assert argparse.ArgumentParser.parse_args is original


def test_extractors_still_parse_their_options():
    kv_config, remaining = serve._extract_kv_args(
        ["--model", "m", "--kv-k-bits", "8", "--kv-v-bits", "3",
         "--kv-min-tokens", "128"])
    assert remaining == ["--model", "m"]
    assert (kv_config["k_bits"], kv_config["v_bits"]) == (8, 3)
    assert kv_config["min_tokens_before_quant"] == 128

    disk, remaining = serve._extract_disk_cache_args(
        ["--disk-cache", "/tmp/c", "--disk-cache-budget-gb", "4", "--port", "1"])
    assert remaining == ["--port", "1"]
    assert disk["cache_dir"] == "/tmp/c" and disk["budget_gb"] == 4.0

    stream, remaining = serve._extract_stream_args(
        ["--cache-budget-gb", "auto", "--max-active-experts", "0",
         "--no-page-cache"])
    assert remaining == []
    assert stream["cache_budget_gb"] == "auto"
    assert stream["max_active_experts"] == 0
    assert stream["use_page_cache"] is False
