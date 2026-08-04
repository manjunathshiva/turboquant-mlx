# Copyright 2026 Manjunath Janardhan
"""Tests for the Sarvam AI MoE port (``model_type: "sarvam_moe"``).

The load-bearing risk in this port is :meth:`Model.sanitize`: the checkpoint is
Megatron-named, so a wrong fused-QKV split or expert stacking order still produces
*correctly shaped* parameters and would only show up as garbage generations. These
tests pin the mapping against the reference semantics rather than against shapes.
"""

import numpy as np
import pytest

import mlx.core as mx

from turboquant_mlx.models.sarvam_moe import Model, ModelArgs


def _tiny_config(**over):
    cfg = dict(
        model_type="sarvam_moe",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        rope_theta=8000000.0,
        first_k_dense_replace=1,
        num_shared_experts=1,
        routed_scaling_factor=2.5,
        score_function="sigmoid",
        norm_topk_prob=True,
        use_qk_norm=True,
    )
    cfg.update(over)
    return cfg


def _model(**over):
    return Model(ModelArgs.from_dict(_tiny_config(**over)))


def test_get_classes_resolves_sarvam_moe():
    """importing compat must make mlx-lm dispatch `sarvam_moe` to our port."""
    import turboquant_mlx.compat  # noqa: F401 — registers the alias
    from mlx_lm.utils import _get_classes

    ModelCls, ArgsCls = _get_classes({"model_type": "sarvam_moe"})
    assert ModelCls is Model
    assert ArgsCls is ModelArgs


def test_fused_qkv_split_matches_reference_semantics():
    """The row split must reproduce the reference's head-axis split exactly.

    Reference does ``qkv.view(B, L, n_heads + 2*n_kv, head_dim).split(...,dim=-2)``.
    Applied to the weight, that means rows are contiguous ``[Q | K | V]`` blocks.
    Encoding each row with its own index makes a wrong (e.g. interleaved) split
    detectable, where a shape-only check would pass.
    """
    m = _model()
    a = m.args
    q_rows = a.num_attention_heads * a.head_dim
    kv_rows = a.num_key_value_heads * a.head_dim
    total = q_rows + 2 * kv_rows

    # row i is filled with the value i
    w = mx.array(np.arange(total, dtype=np.float32)[:, None]
                 * np.ones((1, a.hidden_size), dtype=np.float32))
    out = m.sanitize({"model.layers.0.attention.query_key_value.weight": w})

    q = np.array(out["model.layers.0.self_attn.q_proj.weight"])[:, 0]
    k = np.array(out["model.layers.0.self_attn.k_proj.weight"])[:, 0]
    v = np.array(out["model.layers.0.self_attn.v_proj.weight"])[:, 0]

    assert np.array_equal(q, np.arange(0, q_rows))
    assert np.array_equal(k, np.arange(q_rows, q_rows + kv_rows))
    assert np.array_equal(v, np.arange(q_rows + kv_rows, total))


def test_fused_qkv_rejects_wrong_row_count():
    m = _model()
    bad = mx.zeros((7, m.args.hidden_size))
    with pytest.raises(ValueError, match="fused QKV"):
        m.sanitize({"model.layers.0.attention.query_key_value.weight": bad})


def test_experts_stack_in_expert_order():
    """Per-expert 2D tensors must stack by index, not by dict iteration order."""
    m = _model()
    a = m.args
    weights = {}
    # feed them in shuffled order; expert i is filled with the value i
    for i in reversed(range(a.num_experts)):
        weights[f"model.layers.1.mlp.experts.{i}.gate_proj.weight"] = mx.full(
            (a.moe_intermediate_size, a.hidden_size), float(i)
        )
    out = m.sanitize(weights)
    stacked = np.array(out["model.layers.1.mlp.experts.gate_proj.weight"])

    assert stacked.shape == (a.num_experts, a.moe_intermediate_size, a.hidden_size)
    assert np.array_equal(stacked[:, 0, 0], np.arange(a.num_experts, dtype=np.float32))


def test_sanitize_renames():
    m = _model()
    h = m.args.hidden_size
    out = m.sanitize({
        "model.word_embeddings.weight": mx.zeros((m.args.vocab_size, h)),
        "model.layers.0.attention.dense.weight": mx.zeros((h, h)),
        "model.layers.0.attention.query_layernorm.weight": mx.zeros((m.args.head_dim,)),
        "model.layers.0.attention.key_layernorm.weight": mx.zeros((m.args.head_dim,)),
    })
    assert set(out) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    }


def test_first_k_dense_replace_places_dense_layers():
    m = _model(first_k_dense_replace=2)
    kinds = [type(layer.mlp).__name__ for layer in m.layers]
    assert kinds == ["MLP", "MLP", "SarvamMoE"]


def test_router_bias_steers_selection_only():
    """expert_bias must change *which* experts run, never the combine weights.

    Driven through the real ``SarvamMoE.__call__`` and discriminated against the
    wrong answer: with a large bias on one expert, combining with *biased* weights
    and with *unbiased* weights give visibly different outputs, so the test can
    assert the module matches one and not the other. (Renormalization hides the
    difference in the weight *sum* — both sum to ``route_scale`` — so a test that
    only checked the sum would pass either way.)
    """
    m = _model()
    moe = m.layers[1].mlp
    hidden, n_exp = m.args.hidden_size, m.args.num_experts

    bias = np.zeros(n_exp, dtype=np.float32)
    bias[4] = 10.0
    moe.gate.expert_bias = mx.array(bias)
    mx.eval(m.parameters())

    x = mx.random.normal((1, 3, hidden))
    y = moe(x)

    # reference: selection from biased scores, combine from raw scores
    scores = mx.sigmoid(moe.gate(x.astype(mx.float32)))
    sel = scores + moe.gate.expert_bias.astype(mx.float32)
    inds = mx.argpartition(-sel, kth=moe.top_k - 1, axis=-1)[..., : moe.top_k]
    assert 4 in np.array(inds).ravel().tolist(), "biased expert must be selected"

    expert_out = moe.experts(x, inds)
    shared = moe.shared_experts(x)

    def combine(src):
        w = mx.take_along_axis(src, inds, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20) * moe.route_scale
        return (expert_out * w[..., None]).sum(axis=-2).astype(x.dtype) + shared

    right, wrong = combine(scores), combine(sel)
    mx.eval(y, right, wrong)

    # the two references must actually differ, or the assertion below is vacuous
    assert float(mx.abs(right - wrong).max()) > 1e-3
    assert float(mx.abs(y - right).max()) < 1e-5
    assert float(mx.abs(y - wrong).max()) > 1e-3


def test_qk_norm_applied_before_rope():
    """Pin the QK-norm / RoPE order to the reference's.

    RMSNorm and RoPE do not commute, and both orderings are internally consistent
    (decode still matches prefill either way), so nothing else in this file would
    notice a flip. This pins the order read off the reference
    ``SarvamMoEAttention.forward``: split -> QK-norm -> RoPE.

    The norm weights must be non-trivial for this to test anything: RMSNorm's
    ``1/RMS`` factor is a scalar and RoPE is a rotation (norm-preserving), so with
    the default all-ones weight the two orderings genuinely coincide and any such
    test is vacuous. Only the learned per-dimension weight breaks the commutation.

    Note this pins the *ordering*, it does not prove the port as a whole — that is
    what the logit-parity check against the HF reference on real weights is for.
    """
    m = _model()
    attn = m.layers[0].self_attn
    attn.q_norm.weight = mx.random.uniform(0.5, 1.5, (m.args.head_dim,))
    attn.k_norm.weight = mx.random.uniform(0.5, 1.5, (m.args.head_dim,))
    mx.eval(m.parameters())

    B, L = 1, 4
    x = mx.random.normal((B, L, m.args.hidden_size))
    got = attn(x)

    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim

    def manual(norm_first: bool):
        q = attn.q_proj(x).reshape(B, L, nh, -1).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
        if norm_first:
            q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = attn.rope(q), attn.rope(k)
        else:
            q, k = attn.rope(q), attn.rope(k)
            q, k = attn.q_norm(q), attn.k_norm(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=None)
        return attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))

    right, wrong = manual(True), manual(False)
    mx.eval(got, right, wrong)

    assert float(mx.abs(right - wrong).max()) > 1e-4, "orderings must differ"
    assert float(mx.abs(got - right).max()) < 1e-5
    assert float(mx.abs(got - wrong).max()) > 1e-4


def test_decode_matches_prefill():
    """Incremental decode must reproduce the last prefill position (RoPE offset)."""
    from mlx_lm.models.cache import make_prompt_cache

    m = _model()
    mx.eval(m.parameters())
    toks = [1, 2, 3, 4, 5]

    full = m(mx.array([toks]))
    cache = make_prompt_cache(m)
    step = None
    for t in toks:
        step = m(mx.array([[t]]), cache=cache)
    mx.eval(full, step)

    assert float(mx.abs(step[0, -1] - full[0, -1]).max()) < 2e-4


def test_attn_bits_reaches_attention_paths():
    """Guard the Nemotron trap: --attn-bits must not be a silent no-op here.

    ``bits_for_path`` keys on a ``self_attn`` / ``attention`` path segment, so the
    port's attention container name has to be one of those.
    """
    from turboquant_mlx.config import TurboQuantConfig

    cfg = TurboQuantConfig(bits=2, attn_bits=3)
    assert cfg.bits_for_path("model.layers.0.self_attn.q_proj") == 3
    assert cfg.bits_for_path("model.layers.1.mlp.experts.gate_proj") == 2

    m = _model()
    assert hasattr(m.layers[0], "self_attn")


def test_router_excluded_from_casts():
    m = _model()
    pred = m.cast_predicate
    assert not pred("model.layers.1.mlp.gate.weight")
    assert not pred("model.layers.1.mlp.gate.expert_bias")
    # expert projections named *_proj must still be castable
    assert pred("model.layers.1.mlp.experts.gate_proj.weight")


@pytest.mark.parametrize("over", [
    {"score_function": "softmax"},
    {"n_group": 8, "topk_group": 4},
])
def test_unsupported_routing_rejected(over):
    """Fail loudly rather than silently mis-routing on a config we cannot serve."""
    with pytest.raises(ValueError):
        ModelArgs.from_dict(_tiny_config(**over))
