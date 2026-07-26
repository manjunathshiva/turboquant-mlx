"""Repacking experts must be a pure *relabeling*, under either naming scheme.

`repack_experts` reorders each layer's expert stacks and permutes the router
rows by the same permutation, so the model picks the same expert for the same
input and only the on-disk position changes. That argument holds only while the
two move *together*.

They didn't. The expert-stack branch matched the container by `switch_mlp`
alone, while the router branch matched `mlp.gate.*` unconditionally — so on a
laguna checkpoint (`mlp.experts`) the router rows were permuted and the expert
stacks were left in place. The result loads cleanly, passes every shape check,
and routes every token to the wrong expert.

These tests run the real `main()` over a tiny synthetic checkpoint, so they
exercise the tool rather than just its predicate.
"""

import json
import os
import sys

import mlx.core as mx
import pytest

from turboquant_mlx.stream.repack_experts import main

N_EXPERTS, OUT, PACKED, HIDDEN = 4, 3, 2, 5
PERM = [2, 0, 3, 1]          # order[p] = old expert id now at position p
PROJS = ("gate_proj", "up_proj", "down_proj")


def _checkpoint(tmp_path, container):
    """One layer of a `container`-style MoE, with distinguishable values."""
    src = tmp_path / f"src_{container}"
    src.mkdir()
    t = {}
    for i, proj in enumerate(PROJS):
        # expert e is filled with a value that identifies it
        t[f"model.layers.0.mlp.{container}.{proj}.weight"] = mx.broadcast_to(
            mx.arange(N_EXPERTS, dtype=mx.uint32).reshape(N_EXPERTS, 1, 1),
            (N_EXPERTS, OUT, PACKED)) + i * 100
        t[f"model.layers.0.mlp.{container}.{proj}.scales"] = mx.broadcast_to(
            mx.arange(N_EXPERTS, dtype=mx.float16).reshape(N_EXPERTS, 1, 1),
            (N_EXPERTS, OUT, PACKED))
        # the dense always-on MLP: no expert axis, must never be touched
        t[f"model.layers.0.mlp.shared_experts.{proj}.weight"] = mx.arange(
            OUT * PACKED, dtype=mx.uint32).reshape(OUT, PACKED)
    t["model.layers.0.mlp.gate.weight"] = mx.broadcast_to(
        mx.arange(N_EXPERTS, dtype=mx.float16).reshape(N_EXPERTS, 1),
        (N_EXPERTS, HIDDEN))
    t["model.layers.0.mlp.gate.e_score_correction_bias"] = mx.arange(
        N_EXPERTS, dtype=mx.float16)
    t["model.embed_tokens.weight"] = mx.arange(HIDDEN, dtype=mx.float16)
    mx.eval(*t.values())
    mx.save_safetensors(str(src / "model.safetensors"), t)
    (src / "config.json").write_text("{}")

    perm_file = tmp_path / f"perm_{container}.json"
    perm_file.write_text(json.dumps({"perm": {"0": PERM}}))
    return src, perm_file, t


def _run(src, perm_file, out):
    argv = sys.argv
    sys.argv = ["repack_experts", "--model", str(src), "--perm",
                str(perm_file), "--out", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    return mx.load(os.path.join(str(out), "model.safetensors"))


@pytest.mark.parametrize("container", ["switch_mlp", "experts"])
def test_experts_and_router_move_together(tmp_path, container):
    """The invariant the whole tool rests on: after repacking, position p holds
    the expert that used to be at PERM[p] — in BOTH the weight stacks and the
    router rows. Permuting one alone is a rerouting, not a relabeling."""
    src, perm_file, before = _checkpoint(tmp_path, container)
    after = _run(src, perm_file, tmp_path / f"out_{container}")

    for p, old in enumerate(PERM):
        for proj in PROJS:
            for suffix in ("weight", "scales"):
                key = f"model.layers.0.mlp.{container}.{proj}.{suffix}"
                assert mx.array_equal(after[key][p], before[key][old]), (
                    f"{key}: position {p} should hold old expert {old}")
        for key in ("model.layers.0.mlp.gate.weight",
                    "model.layers.0.mlp.gate.e_score_correction_bias"):
            assert mx.array_equal(after[key][p], before[key][old]), (
                f"{key}: router row {p} should hold old expert {old} — router "
                f"and experts are out of step, which silently reroutes tokens")


@pytest.mark.parametrize("container", ["switch_mlp", "experts"])
def test_shared_experts_and_non_expert_tensors_are_untouched(tmp_path,
                                                             container):
    """`shared_experts` is dense and always-on — it has no expert axis, so
    permuting its axis 0 would scramble rows of a matrix every token uses."""
    src, perm_file, before = _checkpoint(tmp_path, container)
    after = _run(src, perm_file, tmp_path / f"out2_{container}")

    for proj in PROJS:
        key = f"model.layers.0.mlp.shared_experts.{proj}.weight"
        assert mx.array_equal(after[key], before[key]), f"{key} was permuted"
    assert mx.array_equal(after["model.embed_tokens.weight"],
                          before["model.embed_tokens.weight"])


@pytest.mark.parametrize("container", ["switch_mlp", "experts"])
def test_repack_preserves_the_tensor_set(tmp_path, container):
    """Names and shards are supposed to be untouched — only values move."""
    src, perm_file, before = _checkpoint(tmp_path, container)
    after = _run(src, perm_file, tmp_path / f"out3_{container}")
    assert set(after) == set(before)
    for k in before:
        assert after[k].shape == before[k].shape
    assert os.path.exists(str(tmp_path / f"out3_{container}" / "config.json"))
