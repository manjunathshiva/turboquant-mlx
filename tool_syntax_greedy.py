"""Greedy sampling on tool-call structural tokens (DwarfStar-style).

Low-bit builds serve agent harnesses at temperature (greedy loops across
turns), but sampled *structure* is exactly where they fabricate tool calls:
a wrong brace, a hallucinated key, a truncated tag. DwarfStar's answer is to
force temp=0 whenever the model is emitting tool-call syntax and keep the
configured sampler for the free-text argument payloads (greedy on long
bodies causes repetition). This module is the serve-side port: a stateful
logits processor that masks logits to the argmax token while the generation
sits inside a ``<tool_call>``...``</tool_call>`` block, except inside JSON
*value* strings.

Position classification, from the generated text so far:

- outside a tool block                          -> sampled (the *decision* to
  call a tool stays with the sampler)
- inside a block, structural (braces, colons,
  commas, whitespace, tags, numbers, literals)  -> greedy
- inside a block, JSON key string               -> greedy
- inside a block, JSON value string             -> sampled

The processor composes with any sampler because an all-``-inf``-but-argmax
logit row survives temperature/top-p/top-k unchanged. It runs after the
other processors (repetition penalties), so "argmax" means argmax of the
penalized distribution.

State is tracked incrementally: each call decodes only the token ids added
since the previous call and advances a small text scanner. If the token
history rewinds (a rejected speculative-decoding draft), the scanner rebuilds
from the start of the generation. Prompt tokens are never scanned — a
``<tool_call>`` in the prompt (few-shot examples, prior turns) cannot flip
the state.

A malformed block that opens a value string and never closes it leaves the
tail of that block sampled (the close tag is only recognized outside
strings); generation still terminates through the normal stop tokens.
"""

from __future__ import annotations

import math
from typing import List, Optional

import mlx.core as mx

_DEFAULT_OPEN_TAG = "<tool_call>"
_DEFAULT_CLOSE_TAG = "</tool_call>"


class _ToolBlockScanner:
    """Incremental text scanner deciding sampled-vs-greedy for the next token.

    Tracks whether the cursor sits inside an open ``open_tag`` block and, if
    so, a minimal JSON context: container stack (objects remember whether the
    next string is a key), in-string flag, and backslash escapes. Tags are
    matched on a rolling ``endswith`` so they may arrive split across tokens.
    """

    def __init__(self, open_tag: str, close_tag: str) -> None:
        self.open_tag = open_tag
        self.close_tag = close_tag
        self._tail_keep = max(len(open_tag), len(close_tag))
        self.reset()

    def reset(self) -> None:
        self.in_tool = False
        self._tail = ""
        # JSON context, valid while in_tool:
        self._stack: List[List] = []  # ["obj", expect_key] | ["arr"]
        self._in_string = False
        self._string_is_value = False
        self._escape = False

    def consume(self, text: str) -> None:
        for ch in text:
            self._consume_char(ch)

    def _consume_char(self, ch: str) -> None:
        self._tail = (self._tail + ch)[-self._tail_keep:]
        if not self.in_tool:
            if self._tail.endswith(self.open_tag):
                self.in_tool = True
                self._stack = []
                self._in_string = False
                self._escape = False
            return

        if self._in_string:
            if self._escape:
                self._escape = False
            elif ch == "\\":
                self._escape = True
            elif ch == '"':
                self._in_string = False
            return

        if self._tail.endswith(self.close_tag):
            self.in_tool = False
            return

        if ch == '"':
            self._in_string = True
            self._escape = False
            top = self._stack[-1] if self._stack else None
            # A string right after '{' or ',' in an object is a key; any
            # other position (after ':', in an array, bare) is a value.
            self._string_is_value = not (top is not None and top[0] == "obj" and top[1])
        elif ch == "{":
            self._stack.append(["obj", True])
        elif ch == "[":
            self._stack.append(["arr"])
        elif ch in "}]":
            if self._stack:
                self._stack.pop()
        elif ch == ":":
            if self._stack and self._stack[-1][0] == "obj":
                self._stack[-1][1] = False
        elif ch == ",":
            if self._stack and self._stack[-1][0] == "obj":
                self._stack[-1][1] = True

    @property
    def force_greedy(self) -> bool:
        return self.in_tool and not (self._in_string and self._string_is_value)


class ToolSyntaxGreedyProcessor:
    """Logits processor: argmax while inside tool-call structural syntax.

    One instance per request — it carries generation state. ``tokens`` on the
    first call is the prompt; everything before that length is skipped so
    only generated text drives the scanner.
    """

    def __init__(
        self,
        tokenizer,
        open_tag: str = _DEFAULT_OPEN_TAG,
        close_tag: str = _DEFAULT_CLOSE_TAG,
    ) -> None:
        self.tokenizer = tokenizer
        self._scanner = _ToolBlockScanner(open_tag, close_tag)
        self._base: Optional[int] = None  # prompt length, set on first call
        self._pos = 0  # token ids consumed into the scanner so far

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        n = tokens.size
        if self._base is None:
            self._base = n
            self._pos = n
        elif n < self._pos:
            # History rewound (rejected speculative draft): rebuild.
            self._scanner.reset()
            self._pos = self._base
        if n > self._pos:
            new_ids = tokens[self._pos:].tolist()
            self._scanner.consume(self.tokenizer.decode(new_ids))
            self._pos = n
        if not self._scanner.force_greedy:
            return logits
        idx = mx.argmax(logits, axis=-1, keepdims=True)
        mask = mx.arange(logits.shape[-1]) == idx
        return mx.where(mask, logits, mx.array(-math.inf, dtype=logits.dtype))


_INSTALLED = False


def install(
    open_tag: str = _DEFAULT_OPEN_TAG,
    close_tag: str = _DEFAULT_CLOSE_TAG,
) -> None:
    """Patch ``mlx_lm.server`` so every request gets a fresh processor.

    ``_make_logits_processors(args)`` never sees the tokenizer, but the
    server calls ``_make_sampler(args, tokenizer)`` immediately before it on
    the same thread in both the batched and single-stream paths — so the
    sampler patch stashes the tokenizer for the processor patch to read.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    import mlx_lm.server as _server_mod

    _orig_make_sampler = _server_mod._make_sampler
    _orig_make_processors = _server_mod._make_logits_processors
    _ctx: dict = {}

    def _make_sampler(args, tokenizer):
        _ctx["tokenizer"] = tokenizer
        return _orig_make_sampler(args, tokenizer)

    def _make_logits_processors(args):
        processors = list(_orig_make_processors(args))
        tokenizer = _ctx.get("tokenizer")
        if tokenizer is not None:
            processors.append(
                ToolSyntaxGreedyProcessor(tokenizer, open_tag, close_tag)
            )
        return processors

    _server_mod._make_sampler = _make_sampler
    _server_mod._make_logits_processors = _make_logits_processors
    _INSTALLED = True
