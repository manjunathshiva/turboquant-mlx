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
