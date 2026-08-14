"""Muse Glimmer (Meta, dense 30B VLM) support.

Two things here are easy to get silently wrong, and both produce a model that
loads and generates fluent-looking garbage rather than raising:

1. The decoder block contains TWO projections named ``gate_proj`` —
   ``self_attn.gate_proj`` (the sigmoid attention output gate) and
   ``mlp.gate_proj`` (the SwiGLU gate). They read DIFFERENT norms, and they
   belong to different bit tiers.
2. ``embed_tokens`` is a ``NormedEmbedding`` — an ``nn.Embedding`` subclass that
   RMS-normalizes the row it looks up. The inherited ``to_quantized`` returns a
   plain ``QuantizedEmbedding`` and drops the normalization.

Point 2 describes the **pre-merge** mlx-vlm layout. mlx-vlm 0.6.12 released
Blaizzy/mlx-vlm#1838 with the norm hoisted out of the embedding: ``embed_tokens``
is a plain ``nn.Embedding`` and ``TextModel`` owns a paramless ``embed_norm``
that it applies in ``__call__``. Both layouts are supported, so the tests below
skip the ones that do not apply to the installed build.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.integration.rotation_configs import (
    get_rotation_config,
    should_fuse_rotation,
)

_L0 = "language_model.model.layers.0"


def _muse_language_module():
    """The installed build's muse_glimmer language module, or skip."""
    return pytest.importorskip(
        "mlx_vlm.models.muse_glimmer.language",
        reason="mlx-vlm build without muse_glimmer",
    )


def _require_normed_embedding():
    """Skip on mlx-vlm >= 0.6.12, where there is no NormedEmbedding to patch."""
    mg_language = _muse_language_module()
    if not hasattr(mg_language, "NormedEmbedding"):
        pytest.skip(
            "mlx-vlm >= 0.6.12 applies the embedding norm in TextModel, so "
            "there is no NormedEmbedding to patch"
        )
    return mg_language


def test_rotation_config_registered():
    for arch in ("muse_glimmer", "muse_glimmer_text"):
        assert get_rotation_config(arch) is not None


def test_attention_gate_and_mlp_gate_fuse_into_different_norms():
    """The two `gate_proj`s must not collapse onto one norm."""
    cfg = get_rotation_config("muse_glimmer")

    attn_fuse, attn_norm = should_fuse_rotation(f"{_L0}.self_attn.gate_proj", cfg)
    mlp_fuse, mlp_norm = should_fuse_rotation(f"{_L0}.mlp.gate_proj", cfg)

    assert (attn_fuse, attn_norm) == (True, "input_layernorm")
    assert (mlp_fuse, mlp_norm) == (True, "pre_feedforward_layernorm")
    assert attn_norm != mlp_norm


@pytest.mark.parametrize(
    "proj,expected",
    [
        ("self_attn.q_proj", (True, "input_layernorm")),
        ("self_attn.k_proj", (True, "input_layernorm")),
        ("self_attn.v_proj", (True, "input_layernorm")),
        # o_proj reads attn_out * sigmoid(gate) -> non-linear -> online
        ("self_attn.o_proj", (False, None)),
        ("mlp.up_proj", (True, "pre_feedforward_layernorm")),
        # down_proj reads swiglu(gate, up) -> non-linear -> online
        ("mlp.down_proj", (False, None)),
    ],
)
def test_rotation_fusion_targets(proj, expected):
    cfg = get_rotation_config("muse_glimmer")
    assert should_fuse_rotation(f"{_L0}.{proj}", cfg) == expected


def test_post_norms_absorb_nothing():
    """The sandwich `post_*` norms sit on the residual branch, after the
    sublayer, so no projection may fuse into them."""
    cfg = get_rotation_config("muse_glimmer")
    targets = set(cfg.fuse_norm_to_projs)
    assert "post_attention_layernorm" not in targets
    assert "post_feedforward_layernorm" not in targets


def test_qualified_entries_do_not_regress_bare_name_archs():
    """Adding parent-qualified matching must not change LLaMA-style configs."""
    cfg = get_rotation_config("llama")
    assert should_fuse_rotation("m.layers.0.self_attn.q_proj", cfg) == (
        True, "input_layernorm")
    assert should_fuse_rotation("m.layers.0.mlp.gate_proj", cfg) == (
        True, "post_attention_layernorm")
    assert should_fuse_rotation("m.layers.0.mlp.down_proj", cfg) == (False, None)
    assert should_fuse_rotation("m.layers.0.self_attn.o_proj", cfg) == (False, None)


def test_gated_attention_takes_the_attention_bit_tier():
    """`self_attn.gate_proj` is attention, not MLP — a hybrid build must not
    quietly drop it into the MLP tier (or vice versa)."""
    cfg = TurboQuantConfig(bits=3, group_size=64, attn_bits=3, mlp_bits=2)
    assert cfg.bits_for_path(f"{_L0}.self_attn.gate_proj") == 3
    assert cfg.bits_for_path(f"{_L0}.self_attn.q_proj") == 3
    assert cfg.bits_for_path(f"{_L0}.mlp.gate_proj") == 2
    assert cfg.bits_for_path(f"{_L0}.mlp.down_proj") == 2


def test_lm_head_is_kept_out_of_the_polar_path():
    """lm_head must fall to affine, not PolarQuantizedLinear.

    PolarQuantizedLinear only has a fused kernel for single-vector decode; any
    batch > 1 dequantizes the weight through several full-size intermediates
    (~14 bytes/param measured). At 202048x6656 = 1.345B params that is ~19 GB of
    transient peak on every prefill, to save 0.21 GB on disk. Measured end to
    end: prefill peak over weights fell 20.06 GB -> 3.86 GB when lm_head moved
    to affine.
    """
    from turboquant_mlx.integration.vlm import VLM_SKIP_PATTERNS

    assert "lm_head" in VLM_SKIP_PATTERNS["muse_glimmer"]


def test_vlm_skip_predicate_excludes_lm_head_and_vision_tower():
    pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")
    from turboquant_mlx.integration.vlm import vlm_should_quantize

    predicate = vlm_should_quantize("muse_glimmer", lambda p, m: True)
    assert predicate("language_model.lm_head", None) is False
    assert predicate("vision_tower.layers.0.attn.q_proj", None) is False
    # the text stack still goes through TurboQuant
    assert predicate(f"{_L0}.self_attn.q_proj", None) is True
    assert predicate(f"{_L0}.self_attn.gate_proj", None) is True
    assert predicate(f"{_L0}.mlp.down_proj", None) is True


def test_normed_embedding_keeps_its_norm_when_quantized():
    """Quantizing `embed_tokens` must preserve the RMS normalization.

    Without the patch, `nn.quantize` swaps in a plain QuantizedEmbedding and the
    normalization vanishes — no error, just wrong activations everywhere.
    """
    mg_language = _require_normed_embedding()
    from turboquant_mlx.integration.vlm import patch_vlm_arch

    patch_vlm_arch("muse_glimmer")

    vocab, dims, eps = 512, 256, 1e-5
    emb = mg_language.NormedEmbedding(vocab, dims, eps)
    mx.eval(emb.parameters())

    idx = mx.array([1, 7, 300])
    ref = emb(idx)

    q = emb.to_quantized(group_size=64, bits=8)
    mx.eval(q.parameters())
    got = q(idx)

    # The norm survived: a plain QuantizedEmbedding would return unnormalized
    # rows, whose RMS is nowhere near 1.
    rms = mx.sqrt(mx.mean(mx.square(got), axis=-1))
    assert float(mx.max(mx.abs(rms - 1.0))) < 5e-2

    # And 8-bit quantization alone should stay close to the bf16 reference.
    assert float(mx.max(mx.abs(got - ref))) < 5e-2


def test_normed_embedding_patch_is_idempotent():
    mg_language = _require_normed_embedding()
    from turboquant_mlx.integration.vlm import patch_vlm_arch

    patch_vlm_arch("muse_glimmer")
    first = mg_language.NormedEmbedding.to_quantized
    patch_vlm_arch("muse_glimmer")
    assert mg_language.NormedEmbedding.to_quantized is first


def test_quantized_normed_embedding_is_not_a_bare_quantized_embedding():
    mg_language = _require_normed_embedding()
    from turboquant_mlx.integration.vlm import patch_vlm_arch

    patch_vlm_arch("muse_glimmer")
    emb = mg_language.NormedEmbedding(128, 64, 1e-5)
    mx.eval(emb.parameters())
    q = emb.to_quantized(group_size=64, bits=8)

    assert isinstance(q, nn.QuantizedEmbedding)
    assert type(q) is not nn.QuantizedEmbedding
    assert hasattr(q, "embed_norm")


def test_patch_vlm_arch_survives_the_merged_mlx_vlm_layout():
    """`patch_vlm_arch` must be a no-op, not a crash, on mlx-vlm >= 0.6.12.

    The released #1838 dropped `NormedEmbedding` and applies the norm inside
    `TextModel.__call__` instead. An earlier version of the patch reached
    straight for `_mg.NormedEmbedding` and raised AttributeError, which broke
    BOTH `convert_vlm` and `load_turboquant_vlm` — every Muse Glimmer entry
    point — the moment a user installed a current mlx-vlm.
    """
    mg_language = _muse_language_module()
    from turboquant_mlx.integration.vlm import patch_vlm_arch

    had = hasattr(mg_language, "NormedEmbedding")
    saved = getattr(mg_language, "NormedEmbedding", None)
    if had:  # simulate the merged layout on a pre-merge build
        delattr(mg_language, "NormedEmbedding")
    try:
        patch_vlm_arch("muse_glimmer")  # must not raise
        patch_vlm_arch("muse_glimmer_text")
    finally:
        if had:
            mg_language.NormedEmbedding = saved


def test_merged_layout_applies_the_embedding_norm_outside_the_embedding():
    """On >= 0.6.12 the norm must live in TextModel, or quantizing drops it.

    This is the invariant that makes the no-op above safe: a plain
    `QuantizedEmbedding` is only correct because something else normalizes.
    """
    mg_language = _muse_language_module()
    if hasattr(mg_language, "NormedEmbedding"):
        pytest.skip("pre-merge mlx-vlm: the norm lives inside the embedding")

    import inspect

    src = inspect.getsource(mg_language.TextModel)
    assert "self.embed_norm" in src
    assert "self.embed_norm(self.embed_tokens(" in src
    # and it must be paramless, or it would add on-disk keys our converted
    # checkpoints do not carry
    assert mg_language.RMSNormNoScale(1e-6).parameters() == {}
