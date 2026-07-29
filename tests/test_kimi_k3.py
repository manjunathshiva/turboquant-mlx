# Copyright 2026 Manjunath Janardhan
"""Tiny-config tests for the Kimi K3 model port (torch-free, CI-safe).

Real-weight logit parity vs the HF reference (fp32, layers 0-3, 32-expert
subset) is validated separately by a local torch-dependent harness — these
tests pin the structural contracts: model_type dispatch, hybrid layer/cache
typing, the AttnRes forward protocol, and sanitize's key normalization
(per-expert -> stacked, w1/w3/w2 -> gate/up/down_proj, mxfp4 packed -> MLX
quantized layout, A_log de-padding, vision-tower drop, idempotence).
"""

import mlx.core as mx
import pytest

import turboquant_mlx.compat  # noqa: F401 — registers kimi_k3 with mlx-lm
from turboquant_mlx.models import kimi_k3 as k3

HIDDEN = 64
LATENT = 32
MOE_INTER = 64
N_EXPERTS = 8


def tiny_config():
    return {
        "model_type": "kimi_k3",
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 128,
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 16,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
                "kda_layers": [1, 2, 3, 5],  # 1-based; 0-based layer 3 is MLA
                "full_attn_layers": [4],
            },
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts": N_EXPERTS,
            "num_experts_per_token": 2,
            "num_shared_experts": 2,
            "moe_intermediate_size": MOE_INTER,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": LATENT,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,  # boundaries at layers 0, 2, 4
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
        },
    }


def make_model():
    cfg = tiny_config()
    model = k3.Model(k3.ModelArgs.from_dict(cfg))
    mx.eval(model.parameters())
    return model


def test_get_classes_resolves_kimi_k3():
    from mlx_lm.utils import _get_classes

    model_class, args_class = _get_classes(tiny_config())
    assert model_class is k3.Model
    assert args_class is k3.ModelArgs


def test_layer_and_mlp_typing():
    model = make_model()
    kinds = ["KDA" if l.is_linear else "MLA" for l in model.layers]
    assert kinds == ["KDA", "KDA", "KDA", "MLA", "KDA"]
    mlps = [type(l.mlp).__name__ for l in model.layers]
    assert mlps == ["KimiMLP"] + ["LatentMoE"] * 4


def test_make_cache_matches_layer_types():
    from mlx_lm.models.cache import ArraysCache, KVCache

    model = make_model()
    cache = model.make_cache()
    assert [type(c) for c in cache] == [
        ArraysCache,
        ArraysCache,
        ArraysCache,
        KVCache,
        ArraysCache,
    ]


def test_forward_prefill_and_decode():
    model = make_model()
    cache = model.make_cache()
    tokens = mx.array([[1, 5, 9, 3, 7, 2]])
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 6, 128)
    assert not bool(mx.isnan(logits).any())

    tok = mx.argmax(logits[:, -1:], axis=-1)
    step = model(tok, cache=cache)
    mx.eval(step)
    assert step.shape == (1, 1, 128)
    assert not bool(mx.isnan(step).any())


def test_prefill_matches_incremental_decode():
    model = make_model()
    tokens = mx.array([[1, 5, 9, 3, 7, 2]])

    cache_a = model.make_cache()
    full = model(tokens, cache=cache_a)

    cache_b = model.make_cache()
    model(tokens[:, :-1], cache=cache_b)
    step = model(tokens[:, -1:], cache=cache_b)

    diff = float(mx.abs(full[:, -1] - step[:, 0]).max())
    assert diff < 2e-2, f"prefill/decode divergence {diff}"


def _checkpoint_style_weights(model):
    """Synthesize a checkpoint-shaped weight dict (per-expert w1/w3/w2 under
    block_sparse_moe, padded A_log, conv1d layout, kv_b_proj unsplit, plus a
    vision tower to drop) matching the tiny config."""
    args = model.text_args
    lac = args.linear_attn_config
    n_heads, head_dim = lac["num_heads"], lac["head_dim"]
    proj = n_heads * head_dim
    w = {}

    def rnd(*shape):
        return mx.random.normal(shape=shape) * 0.02

    w["vision_tower.encoder.blocks.0.wqkv.weight"] = rnd(8, 8)
    w["mm_projector.proj.0.weight"] = rnd(8, 8)

    for i, layer in enumerate(model.layers):
        p = f"language_model.model.layers.{i}"
        w[f"{p}.input_layernorm.weight"] = mx.ones((HIDDEN,))
        w[f"{p}.post_attention_layernorm.weight"] = mx.ones((HIDDEN,))
        for res in ("self_attention_res", "mlp_res"):
            w[f"{p}.{res}_norm.weight"] = mx.ones((HIDDEN,))
            w[f"{p}.{res}_proj.weight"] = rnd(1, HIDDEN)

        if layer.is_linear:
            for name in ("q_proj", "k_proj", "v_proj"):
                w[f"{p}.self_attn.{name}.weight"] = rnd(proj, HIDDEN)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                w[f"{p}.self_attn.{name}.weight"] = rnd(proj, 1, 4)
            w[f"{p}.self_attn.f_a_proj.weight"] = rnd(head_dim, HIDDEN)
            w[f"{p}.self_attn.f_b_proj.weight"] = rnd(proj, head_dim)
            w[f"{p}.self_attn.b_proj.weight"] = rnd(n_heads, HIDDEN)
            w[f"{p}.self_attn.g_proj.weight"] = rnd(proj, HIDDEN)
            # zero-padded to head_dim, as in the real checkpoint
            w[f"{p}.self_attn.A_log"] = mx.concatenate(
                [rnd(n_heads), mx.zeros((head_dim - n_heads,))]
            )
            w[f"{p}.self_attn.dt_bias"] = rnd(proj)
            w[f"{p}.self_attn.o_norm.weight"] = mx.ones((head_dim,))
            w[f"{p}.self_attn.o_proj.weight"] = rnd(HIDDEN, proj)
        else:
            q_head = args.qk_nope_head_dim + args.qk_rope_head_dim
            w[f"{p}.self_attn.q_a_proj.weight"] = rnd(args.q_lora_rank, HIDDEN)
            w[f"{p}.self_attn.q_a_layernorm.weight"] = mx.ones((args.q_lora_rank,))
            w[f"{p}.self_attn.q_b_proj.weight"] = rnd(
                args.num_attention_heads * q_head, args.q_lora_rank
            )
            w[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = rnd(
                args.kv_lora_rank + args.qk_rope_head_dim, HIDDEN
            )
            w[f"{p}.self_attn.kv_a_layernorm.weight"] = mx.ones((args.kv_lora_rank,))
            w[f"{p}.self_attn.kv_b_proj.weight"] = rnd(
                args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
                args.kv_lora_rank,
            )
            w[f"{p}.self_attn.g_proj.weight"] = rnd(
                args.num_attention_heads * args.v_head_dim, HIDDEN
            )
            w[f"{p}.self_attn.o_proj.weight"] = rnd(
                HIDDEN, args.num_attention_heads * args.v_head_dim
            )

        if type(layer.mlp).__name__ == "LatentMoE":
            mp = f"{p}.block_sparse_moe"
            w[f"{mp}.gate.weight"] = rnd(N_EXPERTS, HIDDEN)
            w[f"{mp}.gate.e_score_correction_bias"] = mx.zeros(
                (N_EXPERTS,), dtype=mx.float32
            )
            w[f"{mp}.routed_expert_down_proj.weight"] = rnd(LATENT, HIDDEN)
            w[f"{mp}.routed_expert_up_proj.weight"] = rnd(HIDDEN, LATENT)
            w[f"{mp}.routed_expert_norm.weight"] = mx.ones((LATENT,))
            for name, (o, i_) in (
                ("gate_proj", (MOE_INTER * 2, HIDDEN)),
                ("up_proj", (MOE_INTER * 2, HIDDEN)),
                ("down_proj", (HIDDEN, MOE_INTER * 2)),
            ):
                w[f"{mp}.shared_experts.{name}.weight"] = rnd(o, i_)
            for e in range(N_EXPERTS):
                w[f"{mp}.experts.{e}.w1.weight"] = rnd(MOE_INTER, LATENT)
                w[f"{mp}.experts.{e}.w3.weight"] = rnd(MOE_INTER, LATENT)
                w[f"{mp}.experts.{e}.w2.weight"] = rnd(LATENT, MOE_INTER)
        else:
            for name, (o, i_) in (
                ("gate_proj", (128, HIDDEN)),
                ("up_proj", (128, HIDDEN)),
                ("down_proj", (HIDDEN, 128)),
            ):
                w[f"{p}.mlp.{name}.weight"] = rnd(o, i_)

    w["language_model.model.embed_tokens.weight"] = rnd(128, HIDDEN)
    w["language_model.model.norm.weight"] = mx.ones((HIDDEN,))
    w["language_model.model.output_attn_res_norm.weight"] = mx.ones((HIDDEN,))
    w["language_model.model.output_attn_res_proj.weight"] = rnd(1, HIDDEN)
    w["language_model.lm_head.weight"] = rnd(128, HIDDEN)
    return w


def test_sanitize_normalizes_checkpoint_layout():
    model = make_model()
    raw = _checkpoint_style_weights(model)
    out = model.sanitize(dict(raw))

    assert not any(k.startswith(("vision_tower.", "mm_projector.")) for k in out)
    assert not any(".block_sparse_moe." in k for k in out)

    stacked = out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    assert stacked.shape == (N_EXPERTS, MOE_INTER, LATENT)
    assert (
        "language_model.model.layers.1.mlp.gate.e_score_correction_bias" in out
    )
    a_log = out["language_model.model.layers.0.self_attn.A_log"]
    assert a_log.shape == (4,)  # de-padded from head_dim=16 to num_heads=4
    conv = out["language_model.model.layers.0.self_attn.q_conv.conv.weight"]
    assert conv.shape == (64, 4, 1)  # (C, K, 1) after moveaxis
    assert "language_model.model.layers.3.self_attn.embed_q.weight" in out
    assert "language_model.model.layers.3.self_attn.kv_b_proj.weight" not in out

    # the sanitized dict must load cleanly and run
    model.load_weights(list(out.items()), strict=True)
    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert not bool(mx.isnan(logits).any())


def test_sanitize_is_idempotent():
    model = make_model()
    once = model.sanitize(_checkpoint_style_weights(model))
    twice = model.sanitize(dict(once))
    assert set(once) == set(twice)
    for k in once:
        assert once[k].shape == twice[k].shape
        assert bool((once[k] == twice[k]).all()), f"{k} changed on second sanitize"


def test_sanitize_mxfp4_packed_experts():
    """Packed uint8 experts view to uint32 weights + raw uint8 scales."""
    model = make_model()
    raw = _checkpoint_style_weights(model)
    # replace layer 1's float experts with an mxfp4-packed layout
    mp = "language_model.model.layers.1.block_sparse_moe"
    for e in range(N_EXPERTS):
        for mat, (o, i_) in (
            ("w1", (MOE_INTER, LATENT)),
            ("w3", (MOE_INTER, LATENT)),
            ("w2", (LATENT, MOE_INTER)),
        ):
            raw.pop(f"{mp}.experts.{e}.{mat}.weight")
            wq, sq = mx.quantize(
                mx.random.normal(shape=(o, i_)) * 0.02,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            raw[f"{mp}.experts.{e}.{mat}.weight_packed"] = wq.view(mx.uint8)
            raw[f"{mp}.experts.{e}.{mat}.weight_scale"] = sq

    out = model.sanitize(raw)
    w = out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    s = out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.scales"]
    assert w.dtype == mx.uint32 and w.shape == (N_EXPERTS, MOE_INTER, LATENT // 8)
    assert s.dtype == mx.uint8 and s.shape == (N_EXPERTS, MOE_INTER, LATENT // 32)


def test_situ_activation_matches_reference_formula():
    import math

    beta, lb = 4.0, 25.0
    g, u = 1.7, -3.2
    want = (beta * math.tanh(g / beta) * (1 / (1 + math.exp(-g)))) * (
        lb * math.tanh(u / lb)
    )
    got = float(
        k3.situ_mul(mx.array([g]), mx.array([u]), beta, lb)[0]
    )
    assert abs(got - want) < 1e-6
