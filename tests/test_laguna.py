# Copyright 2026 Manjunath Janardhan
"""Laguna MLX model: loader-dispatch registration + a forward smoke test.

The numerical logit-parity check against the transformers HF reference lives in
``scripts/laguna_parity.py`` (it needs torch); these tests are torch-free so they
run in CI. They cover the two things the port must guarantee to be loadable:
``_get_classes`` resolves ``model_type: "laguna"`` to our module, and the model
builds and runs on the real XS.2 layer schedule (mixed dense/MoE, mixed
full/sliding attention, per-layer head counts).
"""
import mlx.core as mx
import pytest

import turboquant_mlx.compat  # noqa: F401 — registers the laguna dispatch on import
from turboquant_mlx.models.laguna import Model, ModelArgs


def _tiny_config():
    """Small config that still spans both attention + both MLP layer types."""
    return dict(
        model_type="laguna",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        moe_routed_scaling_factor=2.5,
        moe_router_logit_softcapping=0.0,
        sliding_window=4,
        max_position_embeddings=262144,
        tie_word_embeddings=False,
        attention_bias=False,
        layer_types=[
            "full_attention", "sliding_attention",
            "sliding_attention", "full_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        num_attention_heads_per_layer=[4, 6, 6, 4],
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn", "rope_theta": 500000.0, "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 64.0, "beta_slow": 1.0, "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default", "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    )


def test_get_classes_resolves_laguna():
    """Importing compat aliases our module so mlx-lm dispatch finds it."""
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes({"model_type": "laguna"})
    assert model_cls is Model
    assert args_cls is ModelArgs


def test_forward_smoke():
    args = ModelArgs.from_dict(_tiny_config())
    model = Model(args)
    mx.eval(model.parameters())

    ids = mx.array([[7, 13, 42, 5, 88, 61]])  # L=6 > sliding_window=4
    out = model(ids)
    mx.eval(out)

    assert out.shape == (1, 6, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)))


def test_make_cache_matches_layer_types():
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    args = ModelArgs.from_dict(_tiny_config())
    model = Model(args)
    caches = model.make_cache()

    assert len(caches) == args.num_hidden_layers
    # full_attention -> KVCache, sliding_attention -> RotatingKVCache
    kinds = [type(c) for c in caches]
    assert kinds[0] is KVCache and kinds[3] is KVCache
    assert kinds[1] is RotatingKVCache and kinds[2] is RotatingKVCache


def test_decode_step_with_cache():
    """A prefill + single decode step runs through the cache path."""
    args = ModelArgs.from_dict(_tiny_config())
    model = Model(args)
    mx.eval(model.parameters())
    cache = model.make_cache()

    prompt = mx.array([[7, 13, 42, 5, 88]])
    logits = model(prompt, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 5, args.vocab_size)

    nxt = mx.array([[61]])
    step = model(nxt, cache=cache)
    mx.eval(step)
    assert step.shape == (1, 1, args.vocab_size)
    assert bool(mx.all(mx.isfinite(step)))
