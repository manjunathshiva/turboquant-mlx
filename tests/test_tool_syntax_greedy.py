"""Tests for the tool-syntax greedy logits processor.

A character tokenizer (one token id per code point) drives the processor
through scripted generations; the assertion at each step is whether the
processor left the logits untouched (sampled) or masked them to argmax
(greedy).
"""

import math

import mlx.core as mx
import pytest

from turboquant_mlx.tool_syntax_greedy import ToolSyntaxGreedyProcessor


class CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _encode(text):
    return [ord(c) for c in text]


VOCAB = 4096


def _logits():
    """Fixed non-trivial logits: argmax at index 7."""
    row = mx.zeros((1, VOCAB))
    row = mx.where(mx.arange(VOCAB) == 7, mx.array(2.0), row)
    row = mx.where(mx.arange(VOCAB) == 11, mx.array(1.0), row)
    return row


def _is_greedy(out):
    """True when all but one entry are -inf."""
    finite = mx.isinf(out).astype(mx.int32).sum().item()
    return finite == VOCAB - 1


def _drive(prompt, generated, **proc_kwargs):
    """Run the processor over a scripted generation.

    Returns a list of (position, char_about_to_be_emitted, greedy_bool) for
    each generated character: the processor is consulted with the history
    *before* that character, exactly like real decoding.
    """
    proc = ToolSyntaxGreedyProcessor(CharTokenizer(), **proc_kwargs)
    history = _encode(prompt)
    decisions = []
    for i, ch in enumerate(generated):
        out = proc(mx.array(history), _logits())
        decisions.append((i, ch, _is_greedy(out)))
        history.append(ord(ch))
    return decisions


def _decision_str(decisions):
    """Render decisions as a string of s/G aligned with the generated text."""
    return "".join("G" if g else "s" for _, _, g in decisions)


def test_outside_tool_block_never_masks():
    gen = "Sure, let me check the weather for you."
    decisions = _drive("What is the weather?", gen)
    assert _decision_str(decisions) == "s" * len(gen)


def test_structural_greedy_value_sampled():
    gen = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    decisions = _drive("", gen)
    txt = _decision_str(decisions)
    # The decision to open a tool call is the sampler's: positions emitting
    # the opening tag itself are sampled until the tag completes.
    assert txt[: len("<tool_call>")] == "s" * len("<tool_call>")
    by_pos = dict(enumerate(txt))

    def span(sub):
        start = gen.index(sub)
        return "".join(by_pos[i] for i in range(start, start + len(sub)))

    # Keys and punctuation are greedy.
    assert span('"name"') == "G" * 6
    assert span('"arguments"') == "G" * 11
    assert span(": {") == "GGG"
    # Value string *contents* are sampled: the opening quote is emitted at a
    # greedy position, the characters inside (and the closing quote's
    # position, decided while still in-string) are sampled.
    start = gen.index('"Paris"')
    assert txt[start] == "G"                       # opening quote
    assert txt[start + 1 : start + 7] == "ssssss"  # P a r i s + closing pos
    # After the value closes, structure is greedy again (closing braces).
    tail = gen.index("}}")
    assert txt[tail] == "G" and txt[tail + 1] == "G"


def test_after_close_tag_returns_to_sampled():
    gen = '<tool_call>{"name": "x"}</tool_call> Done thinking.'
    decisions = _drive("", gen)
    txt = _decision_str(decisions)
    after = gen.index(" Done")
    assert txt[after:] == "s" * (len(gen) - after)


def test_nested_containers_and_arrays():
    gen = '<tool_call>{"arguments": {"items": [1, "two", {"k": "v"}]}}</tool_call>'
    decisions = _drive("", gen)
    txt = _decision_str(decisions)

    def at(sub, offset=0):
        return txt[gen.index(sub) + offset]

    assert at("[1, ") == "G"          # array punctuation greedy
    # "two" is an array element -> value string, contents sampled
    start = gen.index('"two"')
    assert txt[start] == "G"
    assert txt[start + 1 : start + 5] == "ssss"
    # nested object key "k" greedy, its value "v" sampled inside
    kstart = gen.index('"k"')
    assert txt[kstart : kstart + 3] == "GGG"
    vstart = gen.index('"v"')
    assert txt[vstart] == "G"
    assert txt[vstart + 1] == "s"


def test_escaped_quote_stays_in_string():
    gen = '<tool_call>{"a": "say \\"hi\\" ok"}</tool_call>'
    decisions = _drive("", gen)
    txt = _decision_str(decisions)
    # Every char between the value's opening quote and its true closing
    # quote is sampled — the escaped quotes must not end the string.
    start = gen.index('"say')
    end = gen.index('"}', start)
    assert txt[start + 1 : end + 1] == "s" * (end - start)
    # After the true close, structure is greedy again.
    assert txt[gen.index("}")] == "G"


def test_prompt_tool_call_is_ignored():
    prompt = 'Earlier: <tool_call>{"name": "old"}'  # unclosed block in prompt
    gen = "A normal sentence."
    decisions = _drive(prompt, gen)
    assert _decision_str(decisions) == "s" * len(gen)


def test_custom_tags():
    gen = "<fn>{\"name\": \"x\"}</fn>"
    decisions = _drive("", gen, open_tag="<fn>", close_tag="</fn>")
    txt = _decision_str(decisions)
    assert txt[gen.index('{"name"')] == "G"
    assert "G" in txt


def test_rewind_rebuilds_state():
    proc = ToolSyntaxGreedyProcessor(CharTokenizer())
    prompt = _encode("hi ")
    gen = '<tool_call>{"a"'
    history = prompt + []
    for ch in gen:
        proc(mx.array(history), _logits())
        history.append(ord(ch))
    # Inside the block now: structural position -> greedy.
    assert _is_greedy(proc(mx.array(history), _logits()))
    # Rewind to just the prompt (rejected speculative draft) and emit plain
    # text instead: must be sampled again.
    history = prompt + _encode("hello")
    out = proc(mx.array(history), _logits())
    assert not _is_greedy(out)


def test_greedy_mask_preserves_argmax():
    proc = ToolSyntaxGreedyProcessor(CharTokenizer())
    history = _encode('<tool_call>{')
    proc(mx.array(_encode("")), _logits())  # first call sets prompt base
    out = proc(mx.array(history), _logits())
    assert _is_greedy(out)
    assert mx.argmax(out, axis=-1).item() == 7
    assert out[0, 7].item() == pytest.approx(2.0)
    assert math.isinf(out[0, 11].item()) and out[0, 11].item() < 0


def test_install_is_idempotent_and_appends_processor():
    import mlx_lm.server as server_mod

    import turboquant_mlx.tool_syntax_greedy as tsg

    orig_sampler = server_mod._make_sampler
    orig_procs = server_mod._make_logits_processors
    try:
        tsg.install()
        first_sampler = server_mod._make_sampler
        tsg.install()  # second install must not re-wrap
        assert server_mod._make_sampler is first_sampler

        class _Args:
            class sampling:
                temperature = 0.7
                top_p = 0.8
                top_k = 20
                min_p = 0.0
                xtc_probability = 0.0
                xtc_threshold = 0.0

            class logits:
                logit_bias = None
                repetition_penalty = None
                repetition_context_size = 20
                presence_penalty = None
                presence_context_size = 20
                frequency_penalty = None
                frequency_context_size = 20

        class _Tok:
            eos_token_id = 0

            def encode(self, s):
                return [1]

            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        server_mod._make_sampler(_Args, _Tok())
        procs = server_mod._make_logits_processors(_Args)
        assert any(isinstance(p, ToolSyntaxGreedyProcessor) for p in procs)
        # A second request gets a *fresh* processor (stateful, per-request).
        procs2 = server_mod._make_logits_processors(_Args)
        p1 = [p for p in procs if isinstance(p, ToolSyntaxGreedyProcessor)][0]
        p2 = [p for p in procs2 if isinstance(p, ToolSyntaxGreedyProcessor)][0]
        assert p1 is not p2
    finally:
        server_mod._make_sampler = orig_sampler
        server_mod._make_logits_processors = orig_procs
        tsg._INSTALLED = False
