"""Vision path for Qwen3.8-Flash-Next: rope threading, 3-D positions, splicing.

The text model gained one optional argument (``rope_cs``) so the vision path can
supply 3-D MRoPE positions for image tokens. These tests pin the two things that
could silently go wrong: that the argument is actually *used* (a no-op would make
the whole vision-position story fiction), and that it is genuinely optional so
the text path is unchanged.
"""

import mlx.core as mx
import pytest

from turboquant_mlx.models.qwen4_exp_vision import (
    build_position_ids,
    splice_image_features,
)

IMG = 99  # stand-in image_token_id


def _tiny_model():
    import turboquant_mlx.compat  # noqa: F401
    from mlx_lm.models import qwen4_exp

    args = qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=1000, rms_norm_eps=1e-6,
        full_attention_interval=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, hc_count=4, hc_lowrank=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8,
        indexer_compress_ratio=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=101, split_ngram_parts=4, ple_embed_dim=64,
        ple_layer_ids=[2], eos_token_id=1,
        rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25}))
    model = qwen4_exp.Model(args)
    mx.eval(model.parameters())
    return model, args


def test_rope_cs_is_optional_and_text_path_is_unchanged():
    model, _ = _tiny_model()
    ids = mx.array([[5, 6, 7, 8, 9, 10]])
    a = model(ids)
    b = model(ids, rope_cs=None)
    assert mx.allclose(a, b), "passing rope_cs=None must be identical to omitting it"


def test_rope_cs_actually_reaches_attention():
    """A threaded-but-ignored argument would make the vision positions fiction."""
    model, args = _tiny_model()
    ids = mx.array([[5, 6, 7, 8, 9, 10]])
    base = model(ids)
    rotary_dim = int(args.text.head_dim * args.text.partial_rotary_factor)
    S = ids.shape[1]
    # cos=1, sin=0 => identity rotation, which no real position produces.
    cos = mx.ones((1, S, rotary_dim))
    sin = mx.zeros((1, S, rotary_dim))
    changed = model(ids, rope_cs=(cos, sin))
    assert not mx.allclose(base, changed, atol=1e-5), \
        "rope_cs did not change the output — it is not reaching Attention"


def test_build_position_ids_text_only_is_plain_sequence():
    ids = [1, 2, 3, 4]
    pos = build_position_ids(ids, (1, 4, 4), IMG, merge_size=2)
    assert pos.shape == (3, 1, 4)
    for d in range(3):
        assert pos[d, 0].tolist() == [0, 1, 2, 3]


def test_build_position_ids_gives_image_tokens_grid_coordinates():
    # grid 1 x 4 x 4, merge 2 -> 2x2 = 4 image tokens between two text tokens
    ids = [1] + [IMG] * 4 + [2]
    pos = build_position_ids(ids, (1, 4, 4), IMG, merge_size=2)
    assert pos.shape == (3, 1, 6)
    t, h, w = (pos[d, 0].tolist() for d in range(3))
    assert (t[0], h[0], w[0]) == (0, 0, 0)              # leading text
    assert h[1:5] == [1, 1, 2, 2]                       # rows, from origin 1
    assert w[1:5] == [1, 2, 1, 2]                       # cols
    assert t[1:5] == [1, 1, 1, 1]                       # single frame
    # trailing text resumes past the image's span (origin + max(t, gh, gw))
    assert (t[5], h[5], w[5]) == (3, 3, 3)


def test_splice_replaces_only_placeholder_rows():
    ids = mx.array([[1, IMG, IMG, 2]])
    emb = mx.ones((1, 4, 8))
    feats = mx.full((2, 8), 7.0)
    out = splice_image_features(emb, ids, feats, IMG)
    assert out[0, 0].tolist() == [1.0] * 8
    assert out[0, 1].tolist() == [7.0] * 8
    assert out[0, 2].tolist() == [7.0] * 8
    assert out[0, 3].tolist() == [1.0] * 8


def test_splice_refuses_a_placeholder_count_mismatch():
    ids = mx.array([[1, IMG, 2]])
    with pytest.raises(ValueError, match="placeholders"):
        splice_image_features(mx.ones((1, 3, 8)), ids, mx.zeros((5, 8)), IMG)


def test_mrope_cos_sin_shape():
    pytest.importorskip("mlx_vlm")
    from turboquant_mlx.models.qwen4_exp_vision import mrope_cos_sin

    tcfg = {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
    S = 7
    pos = mx.broadcast_to(mx.arange(S)[None, None, :], (3, 1, S))
    cos, sin = mrope_cos_sin(tcfg, mx.zeros((1, S, 64)), pos)
    rotary_dim = int(tcfg["head_dim"] * tcfg["rope_parameters"]["partial_rotary_factor"])
    assert cos.shape[-1] == rotary_dim and sin.shape[-1] == rotary_dim
    assert sum(tcfg["rope_parameters"]["mrope_section"]) == rotary_dim // 2
