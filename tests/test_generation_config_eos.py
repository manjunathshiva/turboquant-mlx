# Copyright 2026 Manjunath Janardhan
"""Regression tests for generation_config.json EOS resolution.

A tokenizer only ever exposes a single ``eos_token_id``, so models that declare
several turn terminators in ``generation_config.json`` used to lose all but the
first. Laguna is the motivating case: it ends an assistant turn with
``</assistant>`` (24) and reserves ``〈|EOS|〉`` (2) for document level, so
generation ran past the end of the turn and answered again.
"""
import json

import pytest

from turboquant_mlx.sampling import (
    apply_generation_config_eos,
    eos_token_ids,
    resolve_stop_token,
)


class StubTokenizer:
    """Minimal stand-in for mlx-lm's TokenizerWrapper eos surface."""

    def __init__(self, ids):
        self.eos_token_ids = set(ids)

    def add_eos_token(self, token: str):
        # mirrors TokenizerWrapper: an id may be passed as a string
        self.eos_token_ids.add(int(token))


def _write(tmp_path, payload):
    (tmp_path / "generation_config.json").write_text(json.dumps(payload))
    return tmp_path


def test_adds_missing_terminator(tmp_path):
    tok = StubTokenizer({2})
    added = apply_generation_config_eos(tok, _write(tmp_path, {"eos_token_id": [2, 24]}))
    assert added == {24}
    assert eos_token_ids(tok) == {2, 24}


def test_scalar_eos_token_id(tmp_path):
    tok = StubTokenizer({2})
    added = apply_generation_config_eos(tok, _write(tmp_path, {"eos_token_id": 7}))
    assert added == {7}
    assert eos_token_ids(tok) == {2, 7}


def test_noop_when_already_present(tmp_path):
    tok = StubTokenizer({2, 24})
    assert apply_generation_config_eos(tok, _write(tmp_path, {"eos_token_id": [2, 24]})) == set()
    assert eos_token_ids(tok) == {2, 24}


@pytest.mark.parametrize("payload", [{}, {"eos_token_id": None}])
def test_no_eos_declared(tmp_path, payload):
    tok = StubTokenizer({2})
    assert apply_generation_config_eos(tok, _write(tmp_path, payload)) == set()
    assert eos_token_ids(tok) == {2}


def test_missing_file_is_not_an_error(tmp_path):
    tok = StubTokenizer({2})
    assert apply_generation_config_eos(tok, tmp_path) == set()
    assert eos_token_ids(tok) == {2}


def test_malformed_json_is_not_an_error(tmp_path):
    (tmp_path / "generation_config.json").write_text("{not json")
    tok = StubTokenizer({2})
    assert apply_generation_config_eos(tok, tmp_path) == set()
    assert eos_token_ids(tok) == {2}


class StubVocab:
    """Tokenizer whose encode() maps unknown text to the UNK id, like HF."""

    unk_token_id = 0
    unk_token = "<unk>"
    _vocab = {"</assistant>": 24, "</think>": 19, "<unk>": 0}

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        # unknown text splits into one UNK per word, as a real BPE would not,
        # but single unknown words must still resolve to UNK
        return [self.unk_token_id] * max(1, len(text.split()))


def test_resolve_stop_token_by_id():
    assert resolve_stop_token(StubVocab(), "24") == 24


def test_resolve_stop_token_by_string():
    assert resolve_stop_token(StubVocab(), "</assistant>") == 24


def test_resolve_stop_token_rejects_multi_token():
    with pytest.raises(ValueError, match="tokens for this tokenizer"):
        resolve_stop_token(StubVocab(), "not a real token at all")


def test_resolve_stop_token_rejects_unk():
    """A typo must not silently make UNK terminal (add_eos_token would)."""
    with pytest.raises(ValueError, match="not a token"):
        resolve_stop_token(StubVocab(), "</assistan>")


def test_resolve_stop_token_allows_explicit_unk():
    assert resolve_stop_token(StubVocab(), "<unk>") == 0
