"""Tests for the single-think-block guard and sampling default plumbing."""

import mlx.core as mx

from turboquant_mlx.sampling import (
    make_single_think_close_logits_processor,
    think_close_token_id,
)

CLOSE = 42  # stand-in for the </think> token id
VOCAB = 100


def _logits():
    return mx.zeros((1, VOCAB))


def test_guard_disabled_without_token_id():
    assert make_single_think_close_logits_processor(None) is None
    assert make_single_think_close_logits_processor(-1) is None


def test_guard_allows_first_close():
    proc = make_single_think_close_logits_processor(CLOSE)
    tokens = mx.array([1, 2, 3])  # no </think> yet
    out = proc(tokens, _logits())
    assert out[0, CLOSE].item() == 0.0  # untouched


def test_guard_masks_after_first_close():
    proc = make_single_think_close_logits_processor(CLOSE)
    tokens = mx.array([1, 2, CLOSE, 4])
    out = proc(tokens, _logits())
    assert out[0, CLOSE].item() == -float("inf")
    # other logits untouched
    assert out[0, CLOSE - 1].item() == 0.0


def test_guard_stays_latched_once_seen():
    proc = make_single_think_close_logits_processor(CLOSE)
    proc(mx.array([CLOSE]), _logits())  # latch
    # later steps mask even if the (hypothetical) window no longer shows it
    out = proc(mx.array([7, 8, 9]), _logits())
    assert out[0, CLOSE].item() == -float("inf")


def test_guard_handles_empty_tokens():
    proc = make_single_think_close_logits_processor(CLOSE)
    out = proc(mx.array([], dtype=mx.int32), _logits())
    assert out[0, CLOSE].item() == 0.0


class _FakeTokenizer:
    def __init__(self, mapping):
        self._mapping = mapping

    def encode(self, text, add_special_tokens=False):
        return self._mapping[text]


def test_think_close_token_id_single_token():
    tok = _FakeTokenizer({"</think>": [248069]})
    assert think_close_token_id(tok) == 248069


def test_think_close_token_id_multi_token_returns_none():
    tok = _FakeTokenizer({"</think>": [12, 13, 14]})
    assert think_close_token_id(tok) is None
