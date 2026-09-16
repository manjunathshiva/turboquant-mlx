"""Serving the n-gram table from disk must be bit-identical to the GPU path.

The table is 17.9 GiB of Qwen3.8-Flash-Next's 52 GiB resident build, so moving it
off the GPU is only worth doing if nothing about the output changes. Every check
here compares bits, not closeness.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from turboquant_mlx.ngram_offload import (
    HostShardedEmbedding,
    _read_header,
    _Shard,
    bf16_to_float32,
    offload_ngram_tables,
    round_to_bf16,
)


def _bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _index(path):
    header, data_start = _read_header(path)
    return {k: (path, m, data_start) for k, m in header.items()}


def test_bf16_rounding_matches_mlx_cast():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=50_000) * 10.0 ** rng.uniform(-40, 38, 50_000)).astype(np.float32)
    x = np.concatenate([x, np.array([0.0, -0.0, np.inf, -np.inf, 3.4e38, 1e-44], np.float32)])
    with np.errstate(all="ignore"):
        want = mx.array(x).astype(mx.bfloat16).astype(mx.float32)
        mx.eval(want)
    assert np.array_equal(round_to_bf16(x).view(np.uint32), _bits(want))


def test_bf16_widening_is_bit_exact_including_specials():
    rng = np.random.default_rng(0)
    x = (rng.normal(size=20_000) * 10.0 ** rng.uniform(-40, 38, 20_000)).astype(np.float32)
    x = np.concatenate([x, np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40], np.float32)])
    with np.errstate(all="ignore"):
        b = mx.array(x).astype(mx.bfloat16)
        mx.eval(b)
    raw = np.frombuffer(bytes(memoryview(b)), dtype="<u2")
    assert np.array_equal(bf16_to_float32(raw).view(np.uint32),
                          _bits(b.astype(mx.float32)))


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_quantized_rows_match_quantized_embedding(tmp_path, bits, group_size, dtype):
    """MLX dequantizes in float32 and rounds into the scales' dtype; at 4 and 8 bits,
    and for float16, that rounding changes values, so it has to be reproduced. Rows
    are scaled over five orders of magnitude so the rounding is exercised."""
    rows, dim = 300, 128
    base = _embedding(rows, dim, mx.float32)
    magnitude = mx.power(10.0, mx.linspace(-3, 2, rows))[:, None]
    base.weight = (base.weight * magnitude).astype(dtype)
    emb = nn.QuantizedEmbedding.from_embedding(base, group_size=group_size, bits=bits)
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {"t.weight": emb.weight, "t.scales": emb.scales,
                                    "t.biases": emb.biases})
    shard = _Shard([path], "t", _index(path), dim)

    idx = np.array([0, 1, 7, 150, 299])
    assert np.array_equal(_bits(shard.rows(idx)),
                          _bits(emb(mx.array(idx)).astype(mx.float32)))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_unquantized_rows_match_embedding(tmp_path, dtype):
    emb = _embedding(50, 32, dtype)
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {"t.weight": emb.weight})
    shard = _Shard([path], "t", _index(path), 32)

    idx = np.arange(50)
    assert np.array_equal(_bits(shard.rows(idx)),
                          _bits(emb(mx.array(idx)).astype(mx.float32)))


def test_three_bit_tables_are_refused_not_guessed(tmp_path):
    emb = nn.QuantizedEmbedding.from_embedding(_embedding(10, 64, mx.float32),
                                               group_size=32, bits=3)
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {"t.weight": emb.weight, "t.scales": emb.scales,
                                    "t.biases": emb.biases})
    with pytest.raises(NotImplementedError, match="3-bit"):
        _Shard([path], "t", _index(path), 64)


def test_a_missing_shard_is_a_clear_error(tmp_path):
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {"other.weight": mx.zeros((2, 2))})
    with pytest.raises(KeyError, match="t.weight"):
        _Shard([path], "t", _index(path), 2)


def test_a_header_whose_range_disagrees_with_its_shape_is_refused(tmp_path):
    path = tmp_path / "shard.safetensors"
    mx.save_safetensors(str(path), {"t.weight": mx.zeros((4, 8))})
    index = _index(path)
    index["t.weight"][1]["shape"] = [8, 8]      # claims twice the bytes it has
    with pytest.raises(ValueError, match="does not match"):
        _Shard([path], "t", index, 8)


def test_lookup_handles_repeats_and_every_shard(tmp_path):
    """Repeated ids (common in a prompt) and ids spread over all shards, in an
    arbitrary order and a 2-D shape, come back in the caller's order and shape."""
    n_shards, rows, dim = 3, 40, 32
    tensors, embs = {}, []
    for i in range(n_shards):
        e = nn.QuantizedEmbedding.from_embedding(_embedding(rows, dim, mx.bfloat16, seed=i),
                                                 group_size=32, bits=2)
        embs.append(e)
        for part in ("weight", "scales", "biases"):
            tensors[f"s.shard_{i}.{part}"] = getattr(e, part)
    path = tmp_path / "t.safetensors"
    mx.save_safetensors(str(path), tensors)
    index = _index(path)
    host = HostShardedEmbedding(
        [_Shard([path], f"s.shard_{i}", index, dim) for i in range(n_shards)], rows, dim)

    gid = np.array([[119, 0, 45, 0], [45, 80, 119, 3]])
    got = host(mx.array(gid))
    assert got.shape == (2, 4, dim)
    want = np.stack([np.array(embs[g // rows](mx.array([g % rows])).astype(mx.float32))[0]
                     for g in gid.reshape(-1).tolist()]).reshape(2, 4, dim)
    assert np.array_equal(_bits(got), _bits(want))


# ------------------------------------------------------------- whole model


def _tiny_model():
    """A miniature qwen4_exp whose n-gram table is 2-bit/group-32, like the real one.

    ``ple_embed_dim`` is 128 so each of the 4 n-gram heads is 32 wide, the smallest
    width a 32-element group divides.
    """
    import turboquant_mlx.compat  # noqa: F401  -- registers the alias
    from mlx_lm.models import qwen4_exp

    args = qwen4_exp.ModelArgs(
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
            split_ngram_parts=4, ple_embed_dim=128, ple_layer_ids=[2],
            eos_token_id=1,
            rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        ),
    )

    def build():
        mx.random.seed(0)
        model = qwen4_exp.Model(args)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))
        return model

    return build


def _quantize_tables(model):
    for name, module in model.named_modules():
        if type(module).__name__ != "NGramEmbedding":
            continue
        table = module.ngram_embedding
        for i in range(table.n_shards):
            shard = getattr(table, f"shard_{i}")
            shard.weight = shard.weight.astype(mx.bfloat16)
            setattr(table, f"shard_{i}",
                    nn.QuantizedEmbedding.from_embedding(shard, group_size=32, bits=2))


def test_offloaded_model_is_bit_identical(tmp_path):
    build = _tiny_model()
    ref = build()
    _quantize_tables(ref)
    mx.eval(ref.parameters())
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(ref.parameters())))

    model = build()
    weights = mx.load(str(path))
    n_keys = len(weights)
    moved = offload_ngram_tables(model, weights, [path])
    assert moved > 0
    assert len(weights) == n_keys - 4 * 3        # 4 shards x weight/scales/biases
    assert not any("ngram_embedding" in k for k in weights)
    model.load_weights(list(weights.items()), strict=False)
    assert not any("ngram_embedding" in k for k, _ in tree_flatten(model.parameters()))

    # EOS (id 1) mid-prompt: the n-gram hash must not look across it.
    inputs = mx.array([[5, 17, 17, 1, 17, 17, 300, 42, 9, 9, 9, 1234, 17, 5]])
    assert np.array_equal(_bits(ref(inputs)), _bits(model(inputs)))

    from mlx_lm.models.cache import make_prompt_cache

    cr, cm = make_prompt_cache(ref), make_prompt_cache(model)
    ref(inputs, cache=cr)
    model(inputs, cache=cm)
    for tok in (17, 1, 5, 5):
        step = mx.array([[tok]])
        assert np.array_equal(_bits(ref(step, cache=cr)), _bits(model(step, cache=cm)))


def test_streaming_budget_sees_the_offloaded_bytes(tmp_path):
    """The streaming loader subtracts the offloaded table from resident bytes, as
    plan.py does, so the auto expert budget matches plan's prediction."""
    from turboquant_mlx.stream.loader import offloaded_ngram_bytes

    build = _tiny_model()
    ref = build()
    _quantize_tables(ref)
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(ref.parameters())))
    model = build()
    assert offloaded_ngram_bytes(model) == 0
    moved = offload_ngram_tables(model, mx.load(str(path)), [path])
    assert offloaded_ngram_bytes(model) == moved > 0


def test_models_without_a_table_are_untouched(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4))
    path = tmp_path / "m.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.parameters())))
    weights = mx.load(str(path))
    assert offload_ngram_tables(model, weights, [path]) == 0
    assert len(weights) == 2


def test_serve_peels_the_flag_off_before_mlx_lm_sees_it():
    from turboquant_mlx.serve import _extract_ngram_offload_args

    on, rest = _extract_ngram_offload_args(["--model", "m", "--ngram-offload", "--port", "1"])
    assert on is True and rest == ["--model", "m", "--port", "1"]
    off, rest = _extract_ngram_offload_args(["--model", "m"])
    assert off is False and rest == ["--model", "m"]


def _embedding(rows, dim, dtype, seed=0):
    mx.random.seed(seed)
    e = nn.Embedding(rows, dim)
    e.weight = e.weight.astype(dtype)
    mx.eval(e.weight)
    return e
