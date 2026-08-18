# Copyright 2026 Manjunath Janardhan
"""Tests for MTP-head preservation through conversion.

Hermetic: the real source is a 52 GB download, so these synthesise a source
checkpoint with an `mtp.*` head and a converted checkpoint without one, then
check the head is copied across and the index stays coherent.

The thing under test is a workaround for mlx-lm dropping `mtp.*` in
`sanitize()`, so the test that matters most is `test_extract_bypasses_sanitize`:
it asserts we read the tensors even though the sanitiser would have removed them.
"""

import json
import re

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest
from turboquant_mlx.mtp import (
    MTP_PREFIX,
    MTP_SHARD,
    STORED_PREFIX,
    append_to_checkpoint,
    checkpoint_has_mtp,
    extract_mtp,
    from_stored,
    load_mtp,
    preserve_mtp,
    to_stored,
)

# Real Qwen3.8-27B MTP shapes, scaled down 40x on the hidden dim so the test is
# instant. Structure (one fused input projection, two input norms, a full
# attention + MLP layer, an output norm) is the shape that matters.
H, INTER, QO, KVO = 128, 344, 288, 24


def _mtp_tensors():
    return {
        "mtp.fc.weight": mx.zeros((H, 2 * H), mx.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((H,), mx.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((H,), mx.bfloat16),
        "mtp.norm.weight": mx.ones((H,), mx.bfloat16),
        "mtp.layers.0.input_layernorm.weight": mx.ones((H,), mx.bfloat16),
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((H,), mx.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((QO, H), mx.bfloat16),
        "mtp.layers.0.self_attn.k_proj.weight": mx.zeros((KVO, H), mx.bfloat16),
        "mtp.layers.0.self_attn.v_proj.weight": mx.zeros((KVO, H), mx.bfloat16),
        "mtp.layers.0.self_attn.o_proj.weight": mx.zeros((H, QO // 2), mx.bfloat16),
        "mtp.layers.0.self_attn.q_norm.weight": mx.ones((16,), mx.bfloat16),
        "mtp.layers.0.self_attn.k_norm.weight": mx.ones((16,), mx.bfloat16),
        "mtp.layers.0.mlp.gate_proj.weight": mx.zeros((INTER, H), mx.bfloat16),
        "mtp.layers.0.mlp.up_proj.weight": mx.zeros((INTER, H), mx.bfloat16),
        "mtp.layers.0.mlp.down_proj.weight": mx.zeros((H, INTER), mx.bfloat16),
    }


@pytest.fixture
def source(tmp_path):
    """A two-shard source checkpoint whose LAST shard holds the MTP head.

    Mirrors the real layout, where all 15 tensors live in shard 18 of 18 — so a
    reader that only opens the first shard would silently find nothing.
    """
    src = tmp_path / "src"
    src.mkdir()
    body = {"model.layers.0.self_attn.q_proj.weight": mx.zeros((H, H), mx.bfloat16)}
    mtp = _mtp_tensors()
    mx.save_safetensors(str(src / "model-00001-of-00002.safetensors"), body)
    mx.save_safetensors(str(src / "model-00002-of-00002.safetensors"), mtp)
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(v.nbytes for v in {**body, **mtp}.values())},
        "weight_map": {**{k: "model-00001-of-00002.safetensors" for k in body},
                       **{k: "model-00002-of-00002.safetensors" for k in mtp}},
    }))
    return src


@pytest.fixture
def converted(tmp_path):
    """A converted checkpoint with no MTP head, as mlx-lm's save() would leave it."""
    dst = tmp_path / "dst"
    dst.mkdir()
    body = {"model.layers.0.self_attn.q_proj.weight": mx.zeros((H, H), mx.bfloat16)}
    mx.save_safetensors(str(dst / "model.safetensors"), body)
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(v.nbytes for v in body.values())},
        "weight_map": {k: "model.safetensors" for k in body},
    }))
    return dst


def test_extract_finds_the_head_in_a_later_shard(source):
    got = extract_mtp(str(source))
    assert len(got) == 15
    assert all(k.startswith(MTP_PREFIX) for k in got)


def test_extract_bypasses_sanitize(source):
    """The whole point: sanitize() would drop these, extract_mtp must not.

    Applies mlx-lm's actual filter expression to the same tensors and asserts the
    two disagree. If mlx-lm ever starts keeping `mtp.*`, this test fails loudly
    and the workaround can be reconsidered rather than quietly duplicating work.
    """
    got = extract_mtp(str(source))
    sanitized = {k: v for k, v in got.items() if "mtp." not in k}
    assert sanitized == {}, "sanitize() drops every MTP key"
    assert len(got) == 15, "extract_mtp keeps them"


def test_extract_returns_empty_without_a_head(converted):
    assert extract_mtp(str(converted)) == {}


def test_preserve_writes_shard_and_updates_index(source, converted):
    before = json.loads((converted / "model.safetensors.index.json").read_text())
    n, nbytes = preserve_mtp(str(source), converted)

    assert n == 15
    assert nbytes == sum(v.nbytes for v in _mtp_tensors().values())
    assert (converted / MTP_SHARD).exists()

    after = json.loads((converted / "model.safetensors.index.json").read_text())
    assert len(after["weight_map"]) == len(before["weight_map"]) + 15
    assert {v for k, v in after["weight_map"].items()
            if k.startswith(STORED_PREFIX)} == {MTP_SHARD}
    assert int(after["metadata"]["total_size"]) \
        == int(before["metadata"]["total_size"]) + nbytes
    # Pre-existing entries must survive untouched.
    for k, v in before["weight_map"].items():
        assert after["weight_map"][k] == v


def test_preserved_shard_is_discoverable_by_the_loader_glob(source, converted):
    """turboquant's loader globs `model*.safetensors`; the name must match it."""
    preserve_mtp(str(source), converted)
    found = sorted(p.name for p in converted.glob("model*.safetensors"))
    assert MTP_SHARD in found


def test_round_trip_values_are_unchanged(source, converted):
    """Raw bytes must survive verbatim; the norm shift is applied on load, not here."""
    preserve_mtp(str(source), converted)
    got = mx.load(str(converted / MTP_SHARD))
    want = {to_stored(k): v for k, v in _mtp_tensors().items()}
    assert set(got) == set(want)
    for k in want:
        assert got[k].shape == want[k].shape
        assert got[k].dtype == want[k].dtype
        assert mx.array_equal(got[k], want[k])


def test_checkpoint_has_mtp(source, converted):
    assert not checkpoint_has_mtp(converted)
    preserve_mtp(str(source), converted)
    assert checkpoint_has_mtp(converted)


def test_preserve_is_a_noop_without_a_source_head(converted, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    mx.save_safetensors(str(empty / "model.safetensors"),
                        {"model.norm.weight": mx.ones((H,), mx.bfloat16)})
    assert preserve_mtp(str(empty), converted) == (0, 0)
    assert not (converted / MTP_SHARD).exists()


def test_stored_keys_never_contain_mtp_dot(source, converted):
    """The bug this guards is silent model corruption, not a cosmetic naming nit.

    ``sanitize()`` adds 1.0 to every norm weight when ``any("mtp." in k)``. A
    converted checkpoint's norms are ALREADY shifted, so a stored key containing
    ``"mtp."`` re-fires that predicate on load and shifts them twice, quietly
    breaking every norm in the model. Note it is a substring test: ``tq_mtp.``
    and ``turboquant_mtp.`` would both trip it.
    """
    preserve_mtp(str(source), converted)

    stored = mx.load(str(converted / MTP_SHARD))
    assert stored, "nothing was written"
    offenders = [k for k in stored if "mtp." in k]
    assert offenders == [], f"these keys would re-trigger the norm shift: {offenders}"

    index = json.loads((converted / "model.safetensors.index.json").read_text())
    assert [k for k in index["weight_map"] if "mtp." in k] == []

    # And the exact predicate mlx-lm uses must stay False across the whole
    # checkpoint, index and shards alike.
    every_key = set(index["weight_map"])
    for f in converted.glob("model*.safetensors"):
        every_key |= set(mx.load(str(f)))
    assert not any("mtp." in k for k in every_key)


def test_the_leak_guard_raises_runtimeerror_not_assertionerror(converted):
    """It must survive `python -O`, which strips asserts. A stored key carrying
    "mtp." re-fires the norm shift on load, so this is silent corruption, not a
    programming slip — the distinction that decides which one it has to be."""
    with pytest.raises(RuntimeError, match="re-shift every norm weight"):
        # `to_stored` only rewrites a leading "mtp.", so an embedded one survives
        # renaming and is exactly what the guard exists to catch.
        append_to_checkpoint(converted, {"model.mtp.fc.weight": mx.zeros((2, 2))})


def test_stored_prefix_is_itself_safe():
    assert "mtp." not in STORED_PREFIX


def test_name_mapping_round_trips():
    for k in _mtp_tensors():
        assert from_stored(to_stored(k)) == k
        assert to_stored(k).startswith(STORED_PREFIX)
    # Non-MTP keys pass through untouched.
    assert to_stored("model.norm.weight") == "model.norm.weight"
    assert from_stored("model.norm.weight") == "model.norm.weight"


def test_load_mtp_restores_original_names_and_shifts_every_norm(source, converted):
    """EVERY 1-D tensor gets +1; no 2-D projection does.

    Wider than mlx-lm's four-suffix rule on purpose, and measured: over a
    462-token passage, shifting every 1-D tensor scores 93.72% top-1 agreement
    against 2.81% for mlx-lm's four suffixes alone. See load_mtp's docstring.
    """
    preserve_mtp(str(source), converted)
    got = load_mtp(converted)
    want = _mtp_tensors()

    assert set(got) == set(want), "original mtp.* names must be restored"

    ones = [k for k, v in want.items() if v.ndim == 1]
    twos = [k for k, v in want.items() if v.ndim != 1]
    assert len(ones) == 7 and len(twos) == 8, "head shape changed; re-check the rule"

    for k in ones:
        assert mx.allclose(got[k].astype(mx.float32),
                           want[k].astype(mx.float32) + 1.0), f"{k} should be +1"
    for k in twos:
        assert mx.array_equal(got[k], want[k]), f"{k} must not be shifted"


def test_the_three_norms_mlx_lm_ignores_are_still_shifted(source, converted):
    """Regression guard on the specific finding that unblocked Stage 2.

    These three carry no suffix mlx-lm recognises, so an implementation that
    simply mirrored sanitize() would leave them unshifted and score 2.81%
    instead of 93.72%.
    """
    preserve_mtp(str(source), converted)
    got, want = load_mtp(converted), _mtp_tensors()
    for k in ("mtp.norm.weight",
              "mtp.pre_fc_norm_embedding.weight",
              "mtp.pre_fc_norm_hidden.weight"):
        assert mx.allclose(got[k].astype(mx.float32),
                           want[k].astype(mx.float32) + 1.0), f"{k} must be +1"


def test_load_mtp_is_empty_without_a_head(converted):
    assert load_mtp(converted) == {}


def test_mlx_lm_still_keys_the_norm_shift_off_the_mtp_substring():
    """If mlx-lm stops using `any("mtp." in k)`, the STORED_PREFIX workaround
    may be unnecessary or actively wrong. Fail here rather than drift."""
    import inspect

    from mlx_lm.models import qwen3_5

    # TextModel, not Model: Model.sanitize only re-keys and delegates.
    src = inspect.getsource(qwen3_5.TextModel.sanitize)
    assert 'any("mtp." in k for k in weights)' in src, (
        "mlx-lm no longer keys the norm shift off the substring 'mtp.'"
    )


def test_append_survives_a_missing_index(tmp_path):
    """Single-file checkpoints have no index; the shard must still be written."""
    dst = tmp_path / "noindex"
    dst.mkdir()
    mx.save_safetensors(str(dst / "model.safetensors"),
                        {"model.norm.weight": mx.ones((H,), mx.bfloat16)})
    append_to_checkpoint(dst, _mtp_tensors())
    assert (dst / MTP_SHARD).exists()
    assert checkpoint_has_mtp(dst)


# ── Stage 3a: cache rollback ─────────────────────────────────────────────────

def _hybrid_cache():
    """A Qwen3.8-shaped cache: 48 ArraysCache + 16 KVCache, one full in four."""
    from mlx_lm.models.cache import ArraysCache, KVCache
    return [KVCache() if (i + 1) % 4 == 0 else ArraysCache(size=2) for i in range(64)]


def test_the_hybrid_cache_really_is_untrimmable():
    """The premise of this whole mechanism. If it stops holding, simplify."""
    from mlx_lm.models.cache import ArraysCache, KVCache, can_trim_prompt_cache
    assert KVCache().is_trimmable() is True
    assert ArraysCache(size=2).is_trimmable() is False
    assert can_trim_prompt_cache(_hybrid_cache()) is False
    assert can_trim_prompt_cache([KVCache() for _ in range(16)]) is True


def test_snapshot_covers_exactly_the_untrimmable_layers():
    from turboquant_mlx.mtp import snapshot_recurrent
    cache = _hybrid_cache()
    for i, c in enumerate(cache):
        if not c.is_trimmable():
            c[0] = mx.full((1, 3, 8), float(i))
            c[1] = mx.full((1, 2, 4, 4), float(i))
    snap = snapshot_recurrent(cache)
    assert len(snap) == 64
    assert sum(s is None for s in snap) == 16, "trimmable layers use trim(), not a copy"
    assert sum(s is not None for s in snap) == 48


def test_restore_undoes_a_draft_on_the_recurrent_layers():
    from turboquant_mlx.mtp import restore_recurrent, snapshot_recurrent
    cache = _hybrid_cache()
    for i, c in enumerate(cache):
        if not c.is_trimmable():
            c[0] = mx.full((1, 3, 8), float(i))
            c[1] = mx.full((1, 2, 4, 4), float(i))
    snap = snapshot_recurrent(cache)

    # Draft: every recurrent state advances to garbage.
    for c in cache:
        if not c.is_trimmable():
            c[0] = mx.zeros((1, 3, 8))
            c[1] = mx.ones((1, 2, 4, 4)) * 99.0

    restore_recurrent(cache, snap)
    for i, c in enumerate(cache):
        if not c.is_trimmable():
            assert mx.array_equal(c[0], mx.full((1, 3, 8), float(i))), f"layer {i}"
            assert mx.array_equal(c[1], mx.full((1, 2, 4, 4), float(i))), f"layer {i}"


def test_rollback_trims_offsets_and_restores_states_together():
    from turboquant_mlx.mtp import rollback, snapshot_recurrent
    cache = _hybrid_cache()
    for c in cache:
        if not c.is_trimmable():
            c[0] = mx.zeros((1, 3, 8))
            c[1] = mx.zeros((1, 2, 4, 4))
        else:
            c.update_and_fetch(mx.zeros((1, 4, 5, 16)), mx.zeros((1, 4, 5, 16)))
    snap = snapshot_recurrent(cache)
    before = [c.offset for c in cache if c.is_trimmable()]

    # Draft one token through both halves.
    for c in cache:
        if c.is_trimmable():
            c.update_and_fetch(mx.zeros((1, 4, 1, 16)), mx.zeros((1, 4, 1, 16)))
        else:
            c[1] = mx.ones((1, 2, 4, 4)) * 7.0

    rollback(cache, snap, 1)
    assert [c.offset for c in cache if c.is_trimmable()] == before
    for c in cache:
        if not c.is_trimmable():
            assert mx.array_equal(c[1], mx.zeros((1, 2, 4, 4)))


def test_snapshot_size_is_independent_of_sequence_length():
    """Why one live snapshot is affordable where prefix caching was not."""
    from turboquant_mlx.mtp import snapshot_bytes, snapshot_recurrent
    sizes = []
    for seq in (8, 512, 4096):
        cache = _hybrid_cache()
        for c in cache:
            if not c.is_trimmable():
                c[0] = mx.zeros((1, 3, 8))
                c[1] = mx.zeros((1, 2, 4, 4))
            else:
                c.update_and_fetch(mx.zeros((1, 4, seq, 16)), mx.zeros((1, 4, seq, 16)))
        sizes.append(snapshot_bytes(snapshot_recurrent(cache)))
    assert len(set(sizes)) == 1, f"snapshot grew with context: {sizes}"
    assert sizes[0] > 0


def test_restore_rejects_a_mismatched_snapshot():
    from turboquant_mlx.mtp import restore_recurrent, snapshot_recurrent
    snap = snapshot_recurrent(_hybrid_cache())
    with pytest.raises(ValueError):
        restore_recurrent(_hybrid_cache()[:32], snap)


def _tiny_args():
    qwen35 = pytest.importorskip("mlx_lm.models.qwen3_5")
    return qwen35.TextModelArgs.from_dict(dict(
        hidden_size=64, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, rms_norm_eps=1e-6, full_attention_interval=4,
    ))


def _scalar_tensors(h):
    return {"mtp.fc.weight": mx.zeros((h, 2 * h)),
            "mtp.pre_fc_norm_embedding.weight": mx.zeros((h,)),
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((h,)),
            "mtp.norm.weight": mx.zeros((h,))}


def test_a_partially_loaded_head_is_refused_not_left_random():
    """`load_weights(strict=False)` accepts an empty or partial layer and leaves
    the DecoderLayer randomly initialized. That head still runs and still emits
    fluent-looking drafts, so it reads as a poor acceptance rate rather than a
    fault — the one failure mode that would never surface on its own."""
    from turboquant_mlx.mtp import MTPHead

    args = _tiny_args()
    scalars = _scalar_tensors(args.hidden_size)

    with pytest.raises(KeyError, match="randomly initialized"):
        MTPHead(args).load(dict(scalars))          # no mtp.layers.0.* at all

    head = MTPHead(args)
    full = dict(scalars)
    full.update({f"mtp.layers.0.{k}": v
                 for k, v in tree_flatten(head.layer.parameters())})
    head.load(full)                                 # complete: must not raise

    victim = sorted(k for k in full if k.startswith("mtp.layers.0."))[0]
    partial = {k: v for k, v in full.items() if k != victim}
    with pytest.raises(KeyError, match=re.escape(victim)):
        MTPHead(args).load(partial)
