# Copyright 2026 Manjunath Janardhan
"""Tests for the affine "extras" tier on the load path.

The bug these pin: TurboQuant leaves the token embedding and ``lm_head`` to MLX's
affine quantizer, so they arrive as weight/scales/biases with no ``.codebook``.
``_prepare_polar_layers`` keys off ``.codebook`` and never sees them, the freshly
built model still holds a plain ``nn.Embedding``/``nn.Linear``, and
``load_weights(strict=False)`` then discards their scales without a word. The
result runs and returns nonsense — measured on Qwen3.8-27B, ``embed_tokens(ids)``
came back ``(1, T, 1280) uint32`` instead of ``(1, T, 5120)`` floats.

A silent wrong answer is worse than a crash, hence both a fix and a guard.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from turboquant_mlx.generate import (
    _assert_no_orphan_quant_params,
    _get_nested_module,
    _prepare_affine_extras,
)

VOCAB, DIM, OUT, GS, BITS = 128, 64, 32, 32, 8


class Tiny(nn.Module):
    """Minimal stand-in for the shape of a real text tower."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIM)
        self.lm_head = nn.Linear(DIM, OUT, bias=False)
        self.plain = nn.Linear(DIM, OUT, bias=False)


def _affine(shape, group_size=GS, bits=BITS):
    """Quantize a real array so scales/biases have honest shapes."""
    w = mx.random.normal(shape).astype(mx.bfloat16)
    return mx.quantize(w, group_size=group_size, bits=bits)


def _checkpoint(group_size=GS, bits=BITS):
    ew, es, eb = _affine((VOCAB, DIM), group_size, bits)
    hw, hs, hb = _affine((OUT, DIM), group_size, bits)
    return {
        "embed_tokens.weight": ew, "embed_tokens.scales": es,
        "embed_tokens.biases": eb,
        "lm_head.weight": hw, "lm_head.scales": hs, "lm_head.biases": hb,
        "plain.weight": mx.zeros((OUT, DIM), mx.bfloat16),
    }


def test_get_nested_module_resolves_and_returns_none():
    m = Tiny()
    assert _get_nested_module(m, "embed_tokens") is m.embed_tokens
    assert _get_nested_module(m, "nope") is None
    assert _get_nested_module(m, "embed_tokens.nope.deeper") is None


def test_affine_extras_become_quantized_modules():
    m, w = Tiny(), _checkpoint()
    specs = _prepare_affine_extras(m, w)
    assert set(specs) == {"embed_tokens", "lm_head"}
    assert isinstance(m.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(m.lm_head, nn.QuantizedLinear)
    # A module with no scales in the checkpoint must be left alone.
    assert isinstance(m.plain, nn.Linear)
    assert not isinstance(m.plain, nn.QuantizedLinear)


@pytest.mark.parametrize("group_size,bits", [(32, 4), (32, 8), (64, 8)])
def test_group_size_and_bits_are_recovered_from_shapes(group_size, bits):
    """Shapes alone fix both, because the plain module knows its unpacked width."""
    m, w = Tiny(), _checkpoint(group_size, bits)
    specs = _prepare_affine_extras(m, w)
    for path in ("embed_tokens", "lm_head"):
        assert specs[path] == {"group_size": group_size, "bits": bits}
    assert m.embed_tokens.bits == bits
    assert m.embed_tokens.group_size == group_size


def test_the_weights_actually_load_and_produce_real_floats():
    """The end the bug was breaking: a lookup returning packed uint32."""
    m, w = Tiny(), _checkpoint()
    _prepare_affine_extras(m, w)
    m.load_weights(list(w.items()), strict=False)

    out = m.embed_tokens(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, DIM), "a dropped scales gives the PACKED width here"
    assert out.dtype in (mx.bfloat16, mx.float16, mx.float32)

    logits = m.lm_head(out)
    mx.eval(logits)
    assert logits.shape == (1, 3, OUT)


def test_polar_tensors_are_left_to_the_polar_pass():
    """A .codebook alongside .scales means the polar tier owns that module."""
    m = Tiny()
    w = _checkpoint()
    w["embed_tokens.codebook"] = mx.zeros((16,), mx.bfloat16)
    specs = _prepare_affine_extras(m, w)
    assert "embed_tokens" not in specs
    assert isinstance(m.embed_tokens, nn.Embedding)
    assert "lm_head" in specs


def test_noop_when_there_are_no_affine_extras():
    m = Tiny()
    assert _prepare_affine_extras(m, {"plain.weight": mx.zeros((OUT, DIM))}) == {}
    assert isinstance(m.embed_tokens, nn.Embedding)


def test_orphan_guard_raises_instead_of_returning_garbage():
    """The guard is the real protection: strict=False cannot be dropped here,
    because the checkpoint legitimately carries polar keys the base modules do
    not declare. So an unhandled scales must be caught explicitly."""
    m, w = Tiny(), _checkpoint()
    with pytest.raises(RuntimeError, match="no quantized module"):
        _assert_no_orphan_quant_params(m, w, prepared={})


def test_orphan_guard_passes_once_extras_are_prepared():
    m, w = Tiny(), _checkpoint()
    specs = _prepare_affine_extras(m, w)
    _assert_no_orphan_quant_params(m, w, specs)   # must not raise


def test_orphan_guard_ignores_paths_with_no_module():
    """Sanitize can leave keys for modules a given config never builds."""
    m = Tiny()
    w = {"vision.absent.weight": mx.zeros((4, 4)),
         "vision.absent.scales": mx.zeros((4, 1)),
         "vision.absent.biases": mx.zeros((4, 1))}
    _assert_no_orphan_quant_params(m, w, prepared={})   # must not raise


def test_missing_biases_raises_instead_of_keeping_random_init():
    """`load_weights` does not replace omitted parameters, so a dropped `biases`
    would leave the freshly initialized ones in place and dequantize to garbage."""
    m, w = Tiny(), _checkpoint()
    del w["embed_tokens.biases"]
    with pytest.raises(RuntimeError, match=r"incomplete.*embed_tokens\.biases"):
        _prepare_affine_extras(m, w)


def test_missing_weight_raises_a_named_error_not_a_bare_keyerror():
    m, w = Tiny(), _checkpoint()
    del w["lm_head.weight"]
    with pytest.raises(RuntimeError, match=r"incomplete.*lm_head\.weight"):
        _prepare_affine_extras(m, w)


def test_settings_are_per_module_not_global():
    """Pins the contract that each module keeps its own recovered settings.

    Every checkpoint shipped so far records one `affine_extras` group_size/bits
    for the whole tier, so mixed settings are not a case real weights exercise
    today — this fixes the behaviour the shape-recovery promises, so that a
    checkpoint which ever does mix them cannot regress silently.

    It cannot distinguish the two implementations on a current install:
    `nn.quantize(class_predicate=...)` ignores a per-module dict below MLX 0.22,
    and mlx-lm pins mlx>=0.31.2 transitively. The assertion is on the contract,
    not on the version bug that motivated it."""
    ew, es, eb = _affine((VOCAB, DIM), group_size=64, bits=8)
    hw, hs, hb = _affine((OUT, DIM), group_size=32, bits=4)
    w = {"embed_tokens.weight": ew, "embed_tokens.scales": es,
         "embed_tokens.biases": eb,
         "lm_head.weight": hw, "lm_head.scales": hs, "lm_head.biases": hb}
    m = Tiny()
    specs = _prepare_affine_extras(m, w)

    assert specs["embed_tokens"] == {"group_size": 64, "bits": 8}
    assert specs["lm_head"] == {"group_size": 32, "bits": 4}
    assert (m.embed_tokens.group_size, m.embed_tokens.bits) == (64, 8)
    assert (m.lm_head.group_size, m.lm_head.bits) == (32, 4)

    # And they still load and run, which a mismatched group_size would break.
    m.load_weights(list(w.items()), strict=False)
    out = m.lm_head(m.embed_tokens(mx.array([[1, 2, 3]])))
    mx.eval(out)
    assert out.shape == (1, 3, OUT)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, OUT, bias=False)


class _Tower(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.blocks = [_Block() for _ in range(n)]


class TinyVLM(nn.Module):
    """Shaped like a real VLM: a text tower plus an indexed list of vision blocks."""

    def __init__(self):
        super().__init__()
        self.language_model = Tiny()
        self.vision_tower = _Tower()


def test_modules_behind_a_list_index_are_prepared_and_installed():
    """The vision tower reaches its modules through a list — `vision_tower.
    blocks.0.qkv` — and preparation now installs each one with
    `_set_nested_attr` rather than letting `nn.quantize` walk the tree. Setting a
    replacement back through a list index is the part that has no equivalent in
    the old path, so it gets its own test rather than riding on the flat case.
    """
    m = TinyVLM()
    w = {}
    for path in ("language_model.embed_tokens", "language_model.lm_head",
                 "vision_tower.blocks.0.qkv", "vision_tower.blocks.1.qkv"):
        shape = (VOCAB, DIM) if path.endswith("embed_tokens") else (OUT, DIM)
        q, s, b = _affine(shape)
        w[f"{path}.weight"], w[f"{path}.scales"], w[f"{path}.biases"] = q, s, b

    specs = _prepare_affine_extras(m, w)
    assert set(specs) == {"language_model.embed_tokens", "language_model.lm_head",
                          "vision_tower.blocks.0.qkv", "vision_tower.blocks.1.qkv"}

    # Installed on the real tree, not on a detached copy.
    assert isinstance(m.vision_tower.blocks[0].qkv, nn.QuantizedLinear)
    assert isinstance(m.vision_tower.blocks[1].qkv, nn.QuantizedLinear)
    assert isinstance(m.language_model.embed_tokens, nn.QuantizedEmbedding)

    _assert_no_orphan_quant_params(m, w, specs)     # must not raise
    m.load_weights(list(w.items()), strict=False)
    out = m.vision_tower.blocks[0].qkv(mx.zeros((1, 3, DIM), mx.bfloat16))
    mx.eval(out)
    assert out.shape == (1, 3, OUT)
