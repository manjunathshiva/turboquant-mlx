"""Qwen4-Exp (Qwen3.8-Flash-Next) path-resolution regressions.

Qwen3.8-Flash-Next is a 180B hybrid MoE: 512 routed experts at top-10 with a
narrow ``moe_intermediate_size`` of 640, a per-layer 3x GatedDeltaNet -> 1x
Qwen Sparse Attention pattern, and an n-gram/PLE embedding table that is 28% of
the model. Two of its naming choices differ from every model shipped so far,
and both silently mis-resolved before these tests existed.

Layer paths here mirror mlx-lm PR #1788's ``qwen4_exp.py``.
"""

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import _is_router

_L0 = "model.layers.0"  # linear_attention layer
_L3 = "model.layers.3"  # full_attention layer (every 4th)


def test_hybrid_attention_blocks_both_take_the_attention_tier():
    """Qwen4-Exp alternates ``linear_attn`` (GatedDeltaNet) and ``self_attn``
    (QSA). Both are attention and must take ``attn_bits``.

    This is the check that nemotron_h failed: its ``mixer.*`` paths matched
    neither name, so ``--attn-bits`` was a silent no-op and a "hybrid" build
    was really pure ``mlp_bits`` throughout.
    """
    cfg = TurboQuantConfig(bits=4, group_size=64, attn_bits=4, mlp_bits=2)
    assert cfg.bits_for_path(f"{_L0}.linear_attn.q_proj") == 4
    assert cfg.bits_for_path(f"{_L0}.linear_attn.in_proj_qkvz") == 4
    assert cfg.bits_for_path(f"{_L3}.self_attn.q_proj") == 4
    assert cfg.bits_for_path(f"{_L3}.self_attn.o_proj") == 4


def test_singular_shared_expert_is_exempt_from_the_expert_tier():
    """``shared_expert`` (singular) must NOT fall into ``mlp_bits``.

    Qwen3/Kimi name it ``shared_experts``; Qwen4-Exp drops the plural. The
    exemption used to test equality against the plural form, so the singular
    landed in the sub-2-bit expert tier — and unlike a routed expert (1 of 512,
    top-10) the shared expert runs on *every* token. It is 0.44 GiB bf16 across
    all 48 layers, so protecting it is free.
    """
    cfg = TurboQuantConfig(bits=4, group_size=64, attn_bits=4, mlp_bits=2)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert cfg.bits_for_path(f"{_L0}.mlp.shared_expert.{proj}") == 4, proj
        # The plural spelling used by Qwen3/Kimi must keep working.
        assert cfg.bits_for_path(f"{_L0}.mlp.shared_experts.{proj}") == 4, proj
    # Routed experts still take the expert tier — the exemption is not a blanket.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert cfg.bits_for_path(f"{_L0}.mlp.switch_mlp.{proj}") == 2, proj


def test_qsa_block_indexer_is_kept_at_full_precision():
    """``index_qk_proj`` top-k selects which KV blocks a query attends to.

    That is a discrete choice, so weight error changes *which* blocks are read
    rather than degrading the output smoothly — the same hazard as a MoE
    router, which is already protected. The whole indexer is 19.7M params
    (0.04 GiB) across the 12 full-attention layers, so this costs nothing.
    """
    assert _is_router(f"{_L3}.self_attn.indexer.index_qk_proj")
    assert _is_router(f"{_L3}.self_attn.indexer.some_future_proj")
    # The MoE router and the shared-expert gate stay protected.
    assert _is_router(f"{_L0}.mlp.gate")
    assert _is_router(f"{_L0}.mlp.shared_expert_gate")
    # Ordinary attention and expert projections are still quantized.
    assert not _is_router(f"{_L3}.self_attn.q_proj")
    assert not _is_router(f"{_L0}.mlp.switch_mlp.gate_proj")
    assert not _is_router(f"{_L0}.mlp.shared_expert.down_proj")


# --------------------------------------------------------------------------- forward


def _tiny_args():
    """A miniature qwen4_exp, shaped like mlx-lm PR #1788's own test.

    ``indexer_budget=8`` is the point: a 22-token prompt is well past it, so the
    QSA sparse path actually runs instead of falling back to causal masking.
    """
    import turboquant_mlx.compat  # noqa: F401  -- registers the alias
    from mlx_lm.models import qwen4_exp

    return qwen4_exp, qwen4_exp.ModelArgs(
        model_type="qwen4_exp",
        text_config=dict(
            hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, vocab_size=10_000,
            rms_norm_eps=1e-6, full_attention_interval=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, hc_count=4, hc_lowrank=16,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
            indexer_budget=8, indexer_compress_ratio=4,
            ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=101,
            split_ngram_parts=4, ple_embed_dim=64, ple_layer_ids=[2],
            eos_token_id=1,
            rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        ),
    )


def test_vendored_module_is_what_mlx_lm_resolves():
    """compat.py must alias our copy in until mlx-lm ships qwen4_exp itself."""
    import importlib

    import turboquant_mlx.compat  # noqa: F401

    mod = importlib.import_module("mlx_lm.models.qwen4_exp")
    assert mod.__name__ == "turboquant_mlx.models.qwen4_exp"


def test_qsa_sparse_path_runs_and_stays_causal():
    """Past ``indexer_budget`` the sparse path engages; it must remain causal.

    22 tokens against a budget of 8, and 22 is not a multiple of the compression
    ratio, so the ragged final block is covered too. Changing the *last* token
    must not move any earlier logit — if the block top-k leaked future keys, it
    would.
    """
    import mlx.core as mx
    from mlx.utils import tree_map

    qwen4_exp, args = _tiny_args()
    model = qwen4_exp.Model(args)
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

    inputs = mx.array([list(range(2, 24))])
    prefill = model(inputs)
    assert prefill.shape == (1, 22, args.text.vocab_size)

    other = mx.concatenate([inputs[:, :-1], mx.array([[997]])], axis=-1)
    assert mx.allclose(prefill[:, :-1], model(other)[:, :-1])


def test_prefill_matches_token_by_token_decode():
    """One-shot prefill and 22 single-token steps must agree.

    This is the check that catches a cache bug in the sparse indexer: the
    indexer keeps its own ``_IndexerCache`` of raw keys alongside the KV cache,
    and a mismatch between the two shows up here and essentially nowhere else.
    """
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.models.cache import make_prompt_cache

    qwen4_exp, args = _tiny_args()
    model = qwen4_exp.Model(args)
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

    inputs = mx.array([list(range(2, 24))])
    prefill = model(inputs)

    cache = make_prompt_cache(model)
    steps = [model(inputs[:, i : i + 1], cache=cache) for i in range(22)]
    assert mx.allclose(prefill, mx.concatenate(steps, axis=1), atol=1e-4)


# ----------------------------------------------------------------- affine extras


def _extras_toy():
    """A stand-in for the pieces the polar path never claims.

    Width 160 is the real Qwen3.8-Flash-Next n-gram shard width, and it is the
    whole point of these tests: 160 is divisible by 32 but **not** by the
    default group_size of 64.
    """
    import mlx.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.shard_0 = nn.Embedding(1000, 160)
            self.shard_1 = nn.Embedding(1000, 160)
            self.embed_tokens = nn.Embedding(500, 2560)
            self.gate = nn.Linear(2560, 512, bias=False)
            # Narrow projection: the polar path rejects it on purpose
            # (output_dims < 32). Qwen3.8-Flash-Next has 96 of these --
            # hyper-connection block_inject_weight, shape (4, 640).
            self.block_inject_weight = nn.Linear(640, 4, bias=False)

    return Toy()


def test_extras_quantizes_embeddings_but_never_the_router():
    """nn.Embedding is invisible to the polar path; this tier is what catches it.

    On Qwen3.8-Flash-Next that matters more than anywhere else: the n-gram/PLE
    table is 51.2B params (95.4 GiB bf16, 28% of the model), so leaving it at
    source dtype makes a ternary build ~124 GiB instead of ~54 GiB. The router
    must still come through untouched.
    """
    import mlx.nn as nn

    from turboquant_mlx.quantize_model import quantize_affine_extras

    model, cfg = _extras_toy(), {}
    n = quantize_affine_extras(model, cfg, bits=4, group_size=32)
    kinds = {p: type(m).__name__ for p, m in model.named_modules()}

    assert kinds["shard_0"] == "QuantizedEmbedding"
    assert kinds["shard_1"] == "QuantizedEmbedding"
    assert kinds["embed_tokens"] == "QuantizedEmbedding"
    assert kinds["gate"] == "Linear", "the MoE router must stay full precision"
    assert n == 3
    assert cfg["quantization"]["affine_extras"] == {"bits": 4, "group_size": 32}


def test_default_group_size_silently_misses_the_ngram_table():
    """160 is not a multiple of 64, so the g64 default excludes the shards.

    This is the trap worth a regression test: the pass would report success and
    quietly leave 95 GiB at bf16. The skip is legitimate (a group must divide
    the width) but it must never be silent, so the warning names a group size
    that works.
    """
    from turboquant_mlx.quantize_model import quantize_affine_extras

    model, cfg = _extras_toy(), {}
    n = quantize_affine_extras(model, cfg, bits=4, group_size=64)
    kinds = {p: type(m).__name__ for p, m in model.named_modules()}

    # Only the 2560-wide token embedding survives the divisibility check.
    assert n == 1
    assert kinds["shard_0"] == "Embedding"
    assert kinds["embed_tokens"] == "QuantizedEmbedding"


def test_extras_does_not_requantize_linears_the_polar_path_rejects():
    """`_should_quantize` skips scalar/score projections narrower than 32 because
    quantization noise there costs quality for ~0 bytes. The extras tier must not
    quietly undo that -- but it must still claim embeddings, which
    `_should_quantize` also rejects and which are this tier's entire purpose.
    """
    from turboquant_mlx.quantize_model import quantize_affine_extras

    model, cfg = _extras_toy(), {}
    quantize_affine_extras(model, cfg, bits=4, group_size=32)
    kinds = {p: type(m).__name__ for p, m in model.named_modules()}

    assert kinds["block_inject_weight"] == "Linear", \
        "a 4-wide projection was re-quantized by the extras tier"
    assert kinds["shard_0"] == "QuantizedEmbedding"      # embeddings still claimed
    assert kinds["embed_tokens"] == "QuantizedEmbedding"
    assert kinds["gate"] == "Linear"                     # router still exact


def test_streaming_extras_frees_each_module_after_handing_it_over():
    """The streaming sink must see every tensor and leave nothing resident.

    A bulk nn.quantize would materialize all eligible weights at once — 95.4 GiB
    on the real model, which is exactly what --streaming exists to avoid.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from turboquant_mlx.quantize_model import quantize_affine_extras

    model, cfg = _extras_toy(), {}
    written = {}

    def sink(path, module):
        for sub, arr in tree_flatten(module.parameters()):
            written[f"{path}.{sub}"] = arr.shape

    n = quantize_affine_extras(model, cfg, bits=4, group_size=32, on_quantized=sink)
    assert n == 3
    # Each quantized embedding streams weight + scales + biases.
    for name in ("shard_0", "shard_1", "embed_tokens"):
        assert f"{name}.weight" in written, name
        assert f"{name}.scales" in written, name
        assert f"{name}.biases" in written, name
    # ...and is dropped from the tree so its memory can be reclaimed.
    kinds = {p: type(m).__name__ for p, m in model.named_modules()}
    assert kinds["shard_0"] == "Identity"
    assert kinds["gate"] == "Linear", "the router is neither streamed nor replaced"


def test_resident_and_streaming_extras_agree_on_what_they_claim():
    """Two code paths, one decision — they must not drift."""
    from turboquant_mlx.quantize_model import quantize_affine_extras

    a, cfg_a = _extras_toy(), {}
    n_a = quantize_affine_extras(a, cfg_a, bits=4, group_size=32)

    b, cfg_b = _extras_toy(), {}
    n_b = quantize_affine_extras(b, cfg_b, bits=4, group_size=32,
                                 on_quantized=lambda p, m: None)

    assert n_a == n_b
    assert cfg_a["quantization"] == cfg_b["quantization"]


# ----------------------------------------------------------------- KV quantization


def _tiny_model():
    import mlx.core as mx
    from mlx.utils import tree_map

    qwen4_exp, args = _tiny_args()
    model = qwen4_exp.Model(args)
    model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
    return qwen4_exp, model


def test_kv_quantization_keeps_the_qsa_indexer_cache(capsys):
    """`--kv-bits` must not turn QSA sparse attention into dense attention.

    `_AttnCache` subclasses KVCache and carries the indexer's raw keys. Swapping
    it for a plain TurboQuantKVCache dropped them: each decode step then saw only
    its own key, fell under `indexer_budget`, and attended densely -- no error,
    just a different model (max logit drift 0.22 on this toy). Such layers are
    now left as they are, with a warning.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from turboquant_mlx.layers import polar_kv_cache as pkc

    _, model = _tiny_model()
    inputs = mx.array([list(range(2, 24))])

    getattr(pkc, "_WARNED_KV_SUBCLASSES", set()).discard("_AttnCache")
    cache = pkc.convert_cache_to_turboquant(
        make_prompt_cache(model), k_bits=8, v_bits=8, group_size=32)
    assert ([type(c).__name__ for c in cache]
            == [type(c).__name__ for c in make_prompt_cache(model)])
    assert "_AttnCache" in capsys.readouterr().out

    # prefill 12 (under nothing), then decode 10 past indexer_budget=8
    steps = [model(inputs[:, :12], cache=cache)]
    steps += [model(inputs[:, i : i + 1], cache=cache) for i in range(12, 22)]
    assert mx.allclose(model(inputs), mx.concatenate(steps, axis=1), atol=1e-4)


def test_kv_quantization_still_converts_plain_kvcache_and_warns_once(capsys):
    """Only exact KVCache converts; `serve` converts per request, warn once."""
    from mlx_lm.models.cache import KVCache

    from turboquant_mlx.layers import polar_kv_cache as pkc

    qwen4_exp, _ = _tiny_args()
    getattr(pkc, "_WARNED_KV_SUBCLASSES", set()).discard("_AttnCache")
    for _ in range(3):
        out = pkc.convert_cache_to_turboquant(
            [KVCache(), qwen4_exp._AttnCache()], tq_bits=4)
        assert isinstance(out[0], pkc.TurboQuantKVCache)
        assert type(out[1]) is qwen4_exp._AttnCache
    assert capsys.readouterr().out.count("_AttnCache") == 1


def test_attention_resolves_sdpa_through_the_base_module(monkeypatch):
    """The fused-KV patch replaces `mlx_lm.models.base.scaled_dot_product_attention`.

    A `from base import` copy misses that whenever this module is imported
    after the patch is installed (CodeQL py/import-of-mutable-attribute).
    """
    import mlx.core as mx
    import mlx_lm.models.base as base

    _, model = _tiny_model()
    orig, calls = base.scaled_dot_product_attention, []

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(base, "scaled_dot_product_attention", spy)
    model(mx.array([list(range(2, 12))]))
    assert calls
