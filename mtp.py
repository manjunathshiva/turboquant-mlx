# Copyright 2026 Manjunath Janardhan
"""Preserve the Qwen3.5/3.8 multi-token-prediction (MTP) head through conversion.

Why this module exists
----------------------
``Qwen/Qwen3.8-27B`` ships a real MTP head — 15 tensors: a fused-input projection
``mtp.fc``, two input norms, one complete self-attention + MLP decoder layer, and
an output norm — intended for self-speculative decoding.

**mlx-lm throws it away.** ``mlx_lm/models/qwen3_5.py`` drops every ``mtp.*`` key
inside ``sanitize()``::

    weights = {k: v for k, v in weights.items() if "mtp." not in k}

using their mere *presence* as a hint for norm-weight shifting and nothing else.
No model in mlx-lm implements MTP (``mimo.py`` strips it too). Because
``convert.py`` loads through ``mlx_lm.load()``, the head is gone before
quantization begins, so the converted checkpoint cannot contain it no matter what
the converter does to the module tree.

This module reads those tensors straight from the source shards, bypassing
``sanitize()`` entirely, and appends them to the converted checkpoint.

What is here, and what is not
-----------------------------
``preserve_mtp`` / ``load_mtp`` move the weights through conversion, ``MTPHead``
runs them, and ``speculative_generate`` is a complete self-speculative loop.
mlx-lm's own ``speculative_generate_step`` refuses this model outright, because
``ArraysCache.is_trimmable()`` is ``False``; the loop here works around that by
snapshotting and restoring the 48 Gated-DeltaNet recurrent states (a fixed
~147 MiB, independent of sequence length) and replaying on a miss.

**Nothing calls it, deliberately.** It is verified bit-identical to greedy
decoding, but the measured draft acceptance is ~69% on ordinary prose, and one
draft token verified two-at-a-time returns ``(1+p)/(2-p)`` — about 1.30x at best
before overheads, against a 1.5x bar. So this ships as preserved weights plus a
working reference implementation, not as a decoding path anyone gets by default.
Beware measuring acceptance on repetitive text: the same head scored 93.7% on a
paragraph repeated six times purely because the model was copying.

The head stays at its **source dtype (bf16, 810 MiB)**, ~1.9% of the bf16 model.
Quantizing it is a separate decision that wants its own quality gate, and
quantizing an unexercised head would mix two risks in one change.

One useful accident of the architecture: ``mtp.layers.0`` is shape-identical to a
main full-attention layer (gated ``q_proj`` 12288×5120, GQA ``k/v_proj``
1024×5120, ``o_proj`` 5120×6144, 256-wide q/k norms), so ``MTPHead`` reuses
``DecoderLayer`` rather than defining a new block. It is full attention, not
Gated DeltaNet — so its own cache is a plain trimmable ``KVCache``, which matters
for speculative decoding.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten

#: Prefix these tensors carry in the **source** checkpoint.
MTP_PREFIX = "mtp."

#: Prefix they are stored under in a **converted** checkpoint, and it must not
#: contain the substring ``"mtp."``.
#:
#: This is not cosmetic — it prevents silent model corruption. ``sanitize()``
#: decides whether to add 1.0 to every norm weight from::
#:
#:     has_mtp_weights = any("mtp." in k for k in weights)
#:     should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
#:
#: A converted checkpoint's norms were **already shifted** at conversion time
#: (verified: source layer-3 ``input_layernorm`` mean +0.2493 → converted
#: +1.2492). Storing the head under any name containing ``"mtp."`` makes that
#: predicate fire again on load and shifts them a *second* time, silently
#: breaking every norm in the model. Note the test is a substring match, so
#: ``tq_mtp.`` and ``turboquant_mtp.`` are both unsafe.
STORED_PREFIX = "tq_speculator."

#: Filename the preserved head is written to. The ``model*`` prefix matters:
#: turboquant's loader discovers weights with ``glob("model*.safetensors")``, so
#: this file is picked up without the index needing to be consulted.
MTP_SHARD = "model-mtp.safetensors"


def to_stored(key: str) -> str:
    """``mtp.fc.weight`` -> ``tq_speculator.fc.weight``."""
    return STORED_PREFIX + key[len(MTP_PREFIX):] if key.startswith(MTP_PREFIX) else key


def from_stored(key: str) -> str:
    """``tq_speculator.fc.weight`` -> ``mtp.fc.weight``."""
    return (MTP_PREFIX + key[len(STORED_PREFIX):]
            if key.startswith(STORED_PREFIX) else key)


def _resolve_source(hf_path: str) -> Path:
    """Local directory for ``hf_path``, downloading only what MTP needs.

    Two stages, on purpose. ``allow_patterns=["*.safetensors"]`` would match
    every shard and pull the whole 52 GB source to copy 810 MiB out of it, so the
    first stage fetches only the small files — which include the shard index —
    and that names the single shard actually holding the head. An already-
    complete local cache short-circuits both stages.
    """
    p = Path(hf_path)
    if p.is_dir():
        return p
    from mlx_lm.utils import _download

    meta = Path(_download(hf_path, allow_patterns=["*.json"]))
    index = meta / "model.safetensors.index.json"
    if not index.exists():
        # Unsharded, or no index to narrow with: nothing to be clever about.
        return Path(_download(hf_path, allow_patterns=["*.json", "*.safetensors"]))

    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(MTP_PREFIX)})
    if not shards:
        return meta          # no head in this checkpoint; fetch no weights at all
    return Path(_download(hf_path, allow_patterns=["*.json", *shards]))


def extract_mtp(hf_path: str) -> Dict[str, mx.array]:
    """Return every ``mtp.*`` tensor from the source checkpoint.

    Reads the shards directly. Never goes through ``mlx_lm.load()`` or any
    model's ``sanitize()``, which is the whole point.
    """
    src = _resolve_source(hf_path)
    index = src / "model.safetensors.index.json"

    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({weight_map[k] for k in weight_map if k.startswith(MTP_PREFIX)})
    else:
        shards = sorted(f.name for f in src.glob("*.safetensors"))

    out: Dict[str, mx.array] = {}
    for shard in shards:
        loaded = mx.load(str(src / shard))
        out.update({k: v for k, v in loaded.items() if k.startswith(MTP_PREFIX)})
    return out


def append_to_checkpoint(mlx_path: str | Path, tensors: Dict[str, mx.array]) -> Path:
    """Write ``tensors`` into ``mlx_path`` as an extra shard and fix the index.

    Kept out of ``mlx_lm.utils.save`` deliberately. That function serialises a
    module's ``parameters()``, and these tensors belong to no module — appending
    afterwards is less invasive than teaching it about orphans.
    """
    dst = Path(mlx_path)
    # Renamed on the way in, so no key in the checkpoint contains "mtp." — see
    # STORED_PREFIX for why that would otherwise double-shift every norm weight.
    stored = {to_stored(k): v for k, v in tensors.items()}
    # Not an assert: python -O strips those, and this guards a silent corruption
    # of every norm weight in the model rather than a mere programming slip.
    leaked = sorted(k for k in stored if MTP_PREFIX in k)
    if leaked:
        raise RuntimeError(
            f"stored MTP keys must not contain {MTP_PREFIX!r} or sanitize() will "
            f"re-shift every norm weight: {', '.join(leaked[:5])}"
        )

    shard_path = dst / MTP_SHARD
    mx.save_safetensors(str(shard_path), stored, metadata={"format": "mlx"})

    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index.setdefault("weight_map", {}).update({k: MTP_SHARD for k in stored})
        meta = index.setdefault("metadata", {})
        added = sum(v.nbytes for v in stored.values())
        meta["total_size"] = int(meta.get("total_size", 0)) + added
        index_path.write_text(json.dumps(index, indent=2))
    return shard_path


def load_mtp(mlx_path: str | Path) -> Dict[str, mx.array]:
    """Read a preserved head back, keyed by its original ``mtp.*`` names.

    Adds 1.0 to **every 1-D tensor** in the head, converting its norm weights
    from HF convention to the MLX convention the rest of the loaded model uses.

    All seven 1-D tensors in this head are RMSNorm weights and all eight 2-D ones
    are projections, so ``ndim == 1`` is an exact test for "is a norm" here.

    The rule is wider than mlx-lm's, deliberately, and it is measured. ``sanitize()``
    shifts only four suffixes and says nothing about ``mtp.norm`` or
    ``mtp.pre_fc_norm_{embedding,hidden}``, because it discards the head before
    reaching them. Top-1 agreement with the target over a 462-token passage
    (``Qwen3.8-27B-tq4``, causal mask, ``embedding_first``)::

        shift                             top-1
        mlx-lm's 4 suffixes only          2.81%   (12/462)
        + mtp.norm                        3.03%   (14/462)
        every 1-D tensor                 93.72%  (433/462)

    Two warnings for anyone re-deriving this. Measure with a **causal mask**: a
    multi-position teacher-forced pass without one lets the layer see the future
    and inflates every variant. And measure on **hundreds** of positions: on a
    64-token passage these three land within a few tokens of each other and the
    ordering scrambles, which points at the wrong rule with apparent confidence.
    """
    dst = Path(mlx_path)
    shard = dst / MTP_SHARD
    if not shard.exists():
        return {}
    raw = {from_stored(k): v for k, v in mx.load(str(shard)).items()}
    return {k: (v + 1.0 if v.ndim == 1 else v) for k, v in raw.items()}


def preserve_mtp(hf_path: str, mlx_path: str | Path) -> Tuple[int, int]:
    """Copy the source MTP head into a converted checkpoint.

    Returns ``(tensor_count, bytes)``. ``(0, 0)`` means the source has no MTP
    head, which is the normal case for most architectures and not an error.
    """
    tensors = extract_mtp(hf_path)
    if not tensors:
        return 0, 0
    nbytes = sum(v.nbytes for v in tensors.values())
    append_to_checkpoint(mlx_path, tensors)
    return len(tensors), nbytes


def checkpoint_has_mtp(mlx_path: str | Path) -> bool:
    """True if a converted checkpoint carries a preserved MTP head."""
    dst = Path(mlx_path)
    if (dst / MTP_SHARD).exists():
        return True
    index = dst / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text()).get("weight_map", {})
        return any(k.startswith(STORED_PREFIX) for k in wm)
    return any(
        k.startswith(STORED_PREFIX)
        for f in glob.glob(str(dst / "model*.safetensors"))
        for k in mx.load(f)
    )


# ── Stage 2: the forward pass ────────────────────────────────────────────────
#
# Neither mlx-lm nor transformers implements this head — transformers declares
# `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` and mlx-lm filters it in
# sanitize() — so there is no reference implementation to copy and one detail is
# genuinely undetermined by the checkpoint: which half goes first into `fc`.
#
# `fc.weight` is (hidden, 2*hidden), so it consumes a concatenation of the
# normed next-token embedding and the normed previous hidden state, but nothing
# in the tensor names or shapes fixes the order. DeepSeek-V3, the canonical MTP,
# uses [hidden ; embedding]; Qwen's parameter names list embedding first. Rather
# than guess, `concat_order` is explicit and `probe_concat_order()` decides it by
# measuring which one predicts real tokens.


def _text_args(model):
    """Args for the text tower, whether or not the model is multimodal."""
    inner = getattr(model, "language_model", model)
    args = getattr(inner, "args", None) or getattr(model, "args", None)
    return getattr(args, "text_config", args)


class MTPHead:
    """Qwen3.5/3.8 multi-token-prediction head, as a draft model.

    Deliberately not an ``nn.Module`` subclass holding the target's embedding and
    output projection: it *borrows* both. Making them submodules would duplicate
    a 248,320-row embedding and an equally large ``lm_head`` in the parameter
    tree, which on this model is most of a gigabyte for no reason.

    The wrapped decoder layer is built with ``layer_idx = full_attention_interval
    - 1`` so ``DecoderLayer`` takes its full-attention branch — the head is a
    plain attention layer, not Gated DeltaNet, which is also why its cache is a
    trimmable ``KVCache``.
    """

    def __init__(self, args, concat_order: str = "embedding_first"):
        import mlx.nn as nn
        from mlx_lm.models.qwen3_5 import DecoderLayer

        if concat_order not in ("embedding_first", "hidden_first"):
            raise ValueError(f"bad concat_order: {concat_order!r}")
        self.concat_order = concat_order
        self.args = args

        h, eps = args.hidden_size, args.rms_norm_eps
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
        # layer_idx chosen so is_linear is False -> full attention branch.
        self.layer = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
        self.norm = nn.RMSNorm(h, eps=eps)

    # -- weight loading ------------------------------------------------------
    def load(self, tensors: Dict[str, mx.array]) -> None:
        """Load from ``mtp.*``-keyed tensors, e.g. ``load_mtp()`` output."""
        import mlx.nn as nn

        def _set(mod, key, name="weight"):
            if key not in tensors:
                raise KeyError(f"missing MTP tensor: {key}")
            setattr(mod, name, tensors[key])

        _set(self.fc, "mtp.fc.weight")
        _set(self.pre_fc_norm_embedding, "mtp.pre_fc_norm_embedding.weight")
        _set(self.pre_fc_norm_hidden, "mtp.pre_fc_norm_hidden.weight")
        _set(self.norm, "mtp.norm.weight")
        pre = "mtp.layers.0."
        layer_w = {k[len(pre):]: v for k, v in tensors.items() if k.startswith(pre)}

        # strict=False on its own would accept an empty or partial layer_w and
        # leave the DecoderLayer on its random initialization — the head would
        # then draft fluent-looking nonsense and simply read as a low acceptance
        # rate, never as an error. So check coverage explicitly first, and keep
        # strict=False only to tolerate buffers the checkpoint does not carry.
        expected = {k for k, _ in tree_flatten(self.layer.parameters())}
        missing = sorted(expected - set(layer_w))
        if missing:
            raise KeyError(
                f"MTP decoder layer is missing {len(missing)} of {len(expected)} "
                f"tensors, e.g. {', '.join(pre + m for m in missing[:3])}. "
                "The head would otherwise stay randomly initialized."
            )
        self.layer.load_weights(list(layer_w.items()), strict=False)

    # -- forward -------------------------------------------------------------
    def __call__(self, hidden, next_ids, embed_tokens, lm_head, cache=None):
        """Logits for the token *after* ``next_ids``.

        Args:
            hidden: target hidden states at positions i, shape (B, T, H).
            next_ids: token ids at positions i+1, shape (B, T).
            embed_tokens / lm_head: borrowed from the target model.
            cache: a single ``KVCache``, not a list. ``DecoderLayer`` passes it
                straight through to the attention block, which reads
                ``cache.offset``.

        A causal mask is built whenever more than one position is processed.
        Passing ``mask=None`` there would let the layer attend to *future*
        positions, which silently inflates any teacher-forced agreement score.
        """
        from mlx_lm.models.base import create_attention_mask

        emb = embed_tokens(next_ids)
        a = self.pre_fc_norm_embedding(emb.astype(hidden.dtype))
        b = self.pre_fc_norm_hidden(hidden)
        pair = (a, b) if self.concat_order == "embedding_first" else (b, a)
        h = self.fc(mx.concatenate(pair, axis=-1))
        mask = create_attention_mask(h, cache) if h.shape[1] > 1 else None
        h = self.layer(h, mask, cache)
        return lm_head(self.norm(h))


# ── Stage 3a: rolling the cache back ─────────────────────────────────────────
#
# Speculative decoding has to undo a rejected token's effect on the cache.
# On this architecture that is two different problems:
#
#   * the 16 full-attention layers use KVCache, whose trim(n) is O(1) offset
#     arithmetic — the stale entries are simply overwritten next update; and
#   * the 48 Gated-DeltaNet layers use ArraysCache, which reports
#     is_trimmable() == False and has no trim() at all, because a recurrent
#     state has no per-token slice to drop.
#
# So the GDN half is snapshotted before drafting and restored on rejection.
# That is affordable only because the state is FIXED SIZE, independent of
# sequence length: 48 layers x (3x10240 conv + 48x128x128 fp32 recurrent) =
# 146.81 MiB, and speculation needs exactly one live snapshot at a time. It is
# the same state that makes prefix caching unusable here — APC accumulates one
# snapshot per block across a whole prompt and runs out of memory; this keeps one.


def snapshot_recurrent(cache) -> list:
    """Capture the state of every non-trimmable layer in ``cache``.

    Trimmable layers get ``None`` — they are rolled back with ``trim()`` instead,
    which is far cheaper than copying keys and values.

    The snapshot is evaluated eagerly. MLX arrays are immutable so holding
    references is safe, but leaving them lazy would pin the graph that produced
    them and defeat the point of a fixed-size snapshot.
    """
    snap = []
    for c in cache:
        trimmable = getattr(c, "is_trimmable", None)
        if callable(trimmable) and trimmable():
            snap.append(None)
            continue
        st = c.state
        snap.append(list(st) if isinstance(st, (list, tuple)) else st)

    flat = [a for s in snap if s is not None
            for a in (s if isinstance(s, list) else [s]) if a is not None]
    if flat:
        mx.eval(flat)
    return snap


def restore_recurrent(cache, snap: list) -> None:
    """Undo drafting on the non-trimmable layers, in place."""
    if len(snap) != len(cache):
        raise ValueError(f"snapshot is for {len(snap)} layers, cache has {len(cache)}")
    for c, s in zip(cache, snap):
        if s is not None:
            c.state = list(s) if isinstance(s, list) else s


def trim_trimmable(cache, n: int) -> None:
    """Drop ``n`` positions from every trimmable layer in ``cache``."""
    for c in cache:
        trimmable = getattr(c, "is_trimmable", None)
        if callable(trimmable) and trimmable():
            c.trim(n)


def rollback(cache, snap: list, n: int) -> None:
    """Return ``cache`` to its pre-draft state: trim what can be, restore the rest."""
    trim_trimmable(cache, n)
    restore_recurrent(cache, snap)


def snapshot_bytes(snap: list) -> int:
    """Size of a snapshot, for reporting against a machine's headroom."""
    return sum(a.nbytes for s in snap if s is not None
               for a in (s if isinstance(s, list) else [s]) if a is not None)


# ── Stage 3b: the speculative decode loop ────────────────────────────────────
#
# k = 1. One MTP layer drafts exactly one token, so each iteration runs the
# target once over TWO positions and confirms two tokens on a hit, one on a miss.
# For a bandwidth-bound decode a 2-token batched forward costs about the same as
# a 1-token one, which is where the win comes from.
#
# Per iteration, with `x` the last confirmed token and `y` already known from the
# previous step's logits:
#
#   draft   z' = MTP(h(x), y)
#   verify  target on [y, z'] -> h(y), h(z')
#           z = argmax(head(h(y)))      <- the truth about the token after y
#     hit   z' == z: confirm z AND argmax(head(h(z'))). Two tokens, no rollback.
#     miss  z' != z: confirm z anyway, since it came from the target. Roll the
#           cache back one position to drop z'. One token, and no wasted round
#           trip — the next iteration's verify does double duty.
#
# So a rejected draft costs a cache rollback, not a forward pass.
#
# The MTP cache is kept aligned with the target's, but staleness there could only
# lower the acceptance rate, never change the output: the target verifies every
# token that is emitted. That is what makes the bit-identical gate meaningful.


def _make_cache(text, make_cache=None):
    """Build a cache for the decoder stack.

    ``make_cache`` must be given when ``text`` does not carry the method, which is
    the normal case here: on Qwen3.5 it lives on the outer ``TextModel`` while the
    stack itself is the inner ``Qwen3_5TextModel``. Falling back to
    ``make_prompt_cache`` would be silently wrong — it hands every one of the 64
    layers a ``KVCache``, including the 48 that need ``ArraysCache``.
    """
    if make_cache is not None:
        return make_cache()
    if hasattr(text, "make_cache"):
        return text.make_cache()
    raise TypeError(
        f"{type(text).__name__} has no make_cache(); pass make_cache= explicitly. "
        "Do not fall back to make_prompt_cache on a hybrid model — it would give "
        "the Gated-DeltaNet layers a KVCache and silently corrupt decoding."
    )


def speculative_generate(ids, text, mtp, embed, head, *,
                         max_tokens: int = 64, eos_ids=(), make_cache=None):
    """Greedy decode with MTP speculation. Yields ``(token, from_draft)``.

    ``text`` is the decoder stack, ``embed`` / ``head`` the embedding and output
    projection, ``mtp`` a loaded :class:`MTPHead`. They are passed in rather than
    pulled off a model object so the loop stays usable against a bare decoder
    stack, and so the gate scripts can drive it with their own reference
    embedding and projection.

    Single sequence only: ``ids`` must have batch size 1. The scalar ``.item()``
    calls and the ``mx.array([[y]])`` re-feeds below collapse a batch to its
    first row, so a batched call would return quietly wrong tokens rather than
    fail — hence the explicit check.
    """
    from mlx_lm.models.cache import KVCache

    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(
            f"speculative_generate handles one sequence at a time; got ids with "
            f"shape {tuple(ids.shape)}, expected (1, T)"
        )

    cache = _make_cache(text, make_cache)
    mtp_cache = KVCache()
    eos = {int(e) for e in eos_ids}

    hidden = text(ids, cache=cache, input_embeddings=embed(ids))
    y = int(mx.argmax(head(hidden[:, -1:, :]), axis=-1).item())
    h_prev = hidden[:, -1:, :]
    mx.eval(h_prev)

    # Prime the draft head over positions 0..n-2 so its next call lands on n-1.
    if ids.shape[1] > 1:
        mtp(hidden[:, :-1, :], ids[:, 1:], embed, head, cache=mtp_cache)

    yield y, False
    produced = 1
    if y in eos:
        return

    while produced < max_tokens:
        draft = mtp(h_prev, mx.array([[y]]), embed, head, cache=mtp_cache)
        z_draft = int(mx.argmax(draft[:, -1, :], axis=-1).item())

        # Snapshot is taken BEFORE the two-position verify, so it corresponds to
        # the state before `y` as well as before `z_draft`. On rejection both
        # positions therefore have to come off the target cache -- trimming only
        # the rejected one would leave the 16 KV layers a position ahead of the
        # 48 recurrent layers, which is exactly the kind of desync that shows up
        # as divergence rather than a crash. `y` is then replayed on its own.
        snap = snapshot_recurrent(cache)
        pair = mx.array([[y, z_draft]])
        h2 = text(pair, cache=cache, input_embeddings=embed(pair))
        logits2 = head(h2)
        z = int(mx.argmax(logits2[:, 0, :], axis=-1).item())
        mx.eval(h2)

        if z == z_draft:
            yield z, True
            produced += 1
            if z in eos or produced >= max_tokens:
                return
            y_next = int(mx.argmax(logits2[:, 1, :], axis=-1).item())
            yield y_next, False
            produced += 1
            if y_next in eos:
                return
            # Exactly ONE draft-cache position is missing: (h(y), z). The next
            # iteration's draft call appends the one after it. Advancing by two
            # here would duplicate a position and silently wreck the alignment,
            # which costs acceptance rate rather than correctness.
            mtp(h2[:, 0:1, :], mx.array([[z]]), embed, head, cache=mtp_cache)
            h_prev, y = h2[:, 1:2, :], y_next
        else:
            # Undo the whole verify, then replay the one token that was real.
            rollback(cache, snap, 2)
            replay = mx.array([[y]])
            h1 = text(replay, cache=cache, input_embeddings=embed(replay))
            mx.eval(h1)
            yield z, False
            produced += 1
            if z in eos:
                return
            h_prev, y = h1[:, -1:, :], z
        mx.eval(h_prev)


def greedy_generate(ids, text, embed, head, *, max_tokens: int = 64,
                    eos_ids=(), make_cache=None):
    """Plain greedy decode on the same plumbing — reference for the equality gate.

    Single sequence only, for the same reason as :func:`speculative_generate`;
    the two must agree token for token, so they take the same restriction.
    """
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(
            f"greedy_generate handles one sequence at a time; got ids with "
            f"shape {tuple(ids.shape)}, expected (1, T)"
        )

    cache = _make_cache(text, make_cache)
    eos = {int(e) for e in eos_ids}

    hidden = text(ids, cache=cache, input_embeddings=embed(ids))
    y = int(mx.argmax(head(hidden[:, -1:, :]), axis=-1).item())
    yield y
    if y in eos:
        return
    for _ in range(max_tokens - 1):
        cur = mx.array([[y]])
        h = text(cur, cache=cache, input_embeddings=embed(cur))
        y = int(mx.argmax(head(h[:, -1:, :]), axis=-1).item())
        yield y
        if y in eos:
            return
