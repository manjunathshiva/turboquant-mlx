"""polar_qmm: dense batched GEMM on packed weights.

Verified against the dequantize + GEMM path it replaces, which is the reference
by construction: the kernel must produce the same numbers without ever
materializing the weight.
"""

import mlx.core as mx
import pytest

from turboquant_mlx.core.polar_quantize import polar_dequantize_weight
from turboquant_mlx.core.packing import pack_indices, pack_trits
from turboquant_mlx.kernels.polar_qmm import polar_qmm


def _make(out_dims, in_dims, bits, group_size, seed=0, trit=False):
    """Build a random packed weight plus its dequantized reference."""
    mx.random.seed(seed)
    n_groups = in_dims // group_size
    n_codes = 3 if trit else (1 << bits)
    if trit:
        codebook = mx.array([-0.7, 0.0, 0.7], dtype=mx.float16)
    else:
        codebook = mx.array(
            [(i - (n_codes - 1) / 2) * 0.5 for i in range(n_codes)],
            dtype=mx.float16,
        )
    idx = mx.random.randint(0, n_codes, (out_dims, in_dims)).astype(mx.uint32)
    scales = (mx.random.uniform(shape=(out_dims, n_groups)) * 0.5
              + 0.25).astype(mx.float16)
    packed = pack_trits(idx) if trit else pack_indices(idx, bits)
    ref_w = polar_dequantize_weight(packed, scales, codebook, bits, group_size,
                                    in_dims, trit=trit)
    return packed, scales, codebook, ref_w


def _check(out_dims, in_dims, n_tokens, bits, group_size, trit=False, tol=2e-2):
    packed, scales, codebook, ref_w = _make(out_dims, in_dims, bits,
                                            group_size, trit=trit)
    x = mx.random.normal((n_tokens, in_dims)).astype(mx.float16)

    got = polar_qmm(packed, scales, codebook, x, bits, group_size, trit=trit)
    ref = x @ ref_w.T
    mx.eval(got, ref)

    assert got.shape == (n_tokens, out_dims)
    denom = mx.maximum(mx.abs(ref).max(), mx.array(1e-3, dtype=mx.float16))
    rel = float((mx.abs(got - ref) / denom).max())
    assert rel < tol, f"max rel err {rel:.4g} (bits={bits} g={group_size} " \
                      f"O={out_dims} K={in_dims} N={n_tokens})"


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_matches_dequant_gemm(bits, group_size):
    _check(128, 256, 8, bits, group_size)


@pytest.mark.parametrize("n_tokens", [1, 2, 7, 15, 16, 17, 31, 32, 64, 129])
def test_token_counts_including_tile_boundaries(n_tokens):
    """Tokens are processed in tiles of 16 — check the tail handling."""
    _check(128, 256, n_tokens, 3, 64)


@pytest.mark.parametrize("out_dims", [64, 128, 192, 256])
def test_output_block_multiples(out_dims):
    _check(out_dims, 256, 8, 3, 64)


@pytest.mark.parametrize("out_dims", [65, 100, 130])
def test_output_dims_not_multiple_of_64(out_dims):
    """The safe_o clamp must keep partial output blocks in bounds."""
    _check(out_dims, 256, 8, 3, 64)


@pytest.mark.parametrize("in_dims", [256, 320, 512, 640, 1024])
def test_k_chunk_boundaries(in_dims):
    _check(128, in_dims, 8, 3, 64)


def test_trit_packing():
    _check(128, 320, 8, 3, 64, trit=True)


@pytest.mark.parametrize("n_tokens", [1, 16, 33])
def test_trit_token_counts(n_tokens):
    _check(128, 320, n_tokens, 3, 64, trit=True)


def test_real_muse_glimmer_shapes():
    """The shapes that actually matter on a 30B dense model."""
    for out_dims, in_dims in [(4096, 6656),    # q_proj / attn gate_proj
                              (256, 6656),     # k_proj / v_proj (GQA 2 heads)
                              (6656, 4096),    # o_proj
                              (19968, 6656)]:  # mlp gate/up
        _check(out_dims, in_dims, 4, 3, 64, tol=3e-2)


def test_single_token_matches_qmv():
    """N=1 must agree with the dedicated decode kernel."""
    from turboquant_mlx.kernels.polar_qmv import polar_qmv

    packed, scales, codebook, _ = _make(256, 512, 3, 64)
    x = mx.random.normal((1, 512)).astype(mx.float16)
    a = polar_qmm(packed, scales, codebook, x, 3, 64)
    b = polar_qmv(packed, scales, codebook, x, 3, 64)
    mx.eval(a, b)
    denom = mx.maximum(mx.abs(b).max(), mx.array(1e-3, dtype=mx.float16))
    assert float((mx.abs(a - b) / denom).max()) < 2e-2
