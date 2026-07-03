"""Sampling helpers for TurboQuant-MLX."""

from typing import Callable, Iterable, Optional

import mlx.core as mx


def make_min_tokens_logits_processor(
    min_tokens: int,
    eos_token_ids: Iterable[int],
) -> Optional[Callable[[mx.array, mx.array], mx.array]]:
    """Build a logits processor that masks EOS tokens until ``min_tokens``
    tokens have been generated.

    Some models — Nemotron 3 in particular, and other "thinking-mode" models
    whose chat template ends with a think-scaffold like ``<think>\\n`` — have
    an EOS token sitting at the top of the first-step logits. With greedy or
    near-greedy sampling the model then terminates before emitting a single
    visible token. This processor keeps the sampler honest by forcing at
    least ``min_tokens`` tokens of output before EOS can win.

    Args:
        min_tokens: Number of tokens to produce before allowing EOS. ``0``
            (or any non-positive value) disables the processor entirely.
        eos_token_ids: The token ids considered terminal. Pass the tokenizer's
            full EOS set, not just the primary id (some models have multiple).

    Returns:
        A ``(tokens, logits) -> logits`` callable suitable for mlx-lm's
        ``logits_processors`` kwarg, or ``None`` if no-op.
    """
    if min_tokens <= 0:
        return None
    eos = list(eos_token_ids)
    if not eos:
        return None
    eos_arr = mx.array(eos)
    neg_inf = -float("inf")

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size < min_tokens:
            logits[..., eos_arr] = neg_inf
        return logits

    return processor


def make_single_think_close_logits_processor(
    think_close_id: Optional[int],
) -> Optional[Callable[[mx.array, mx.array], mx.array]]:
    """Build a logits processor that allows at most one ``</think>`` token.

    Low-bit thinking models occasionally sample a second ``</think>`` where
    ``<|im_end|>`` belongs — the model then "reopens" its answer and emits it
    again verbatim (observed on ternary-expert builds). After the first
    ``</think>`` appears in the generated tokens, this processor masks it,
    so the only sensible continuation after the answer is EOS.

    Args:
        think_close_id: Token id of ``</think>``. Pass ``None`` (or a
            negative id) when the tokenizer has no such single token —
            the processor is disabled and ``None`` is returned.

    Returns:
        A ``(tokens, logits) -> logits`` callable for mlx-lm's
        ``logits_processors`` kwarg, or ``None`` if no-op.
    """
    if think_close_id is None or think_close_id < 0:
        return None
    neg_inf = -float("inf")
    # Kept as an mx.array so the guard never forces a CPU-GPU sync inside
    # the decode loop; it latches True once </think> appears.
    seen = mx.array(False)

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        nonlocal seen
        if tokens is not None and tokens.size:
            seen = mx.logical_or(seen, mx.any(tokens == think_close_id))
        logits[..., think_close_id] = mx.where(
            seen, neg_inf, logits[..., think_close_id]
        )
        return logits

    return processor


def think_close_token_id(tokenizer) -> Optional[int]:
    """Return the id of ``</think>`` when it encodes to a single token."""
    try:
        ids = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:
        return None
    return ids[0] if len(ids) == 1 else None


def eos_token_ids(tokenizer) -> set:
    """Return the full set of EOS token ids a tokenizer considers terminal.

    Mirrors the lookup pattern used across the generation code paths so we
    don't have the fallback logic sprinkled in five places.
    """
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    primary = getattr(tokenizer, "eos_token_id", None)
    return {primary} if primary is not None else set()
