"""QJL: Quantized Johnson-Lindenstrauss residual correction (Stage 2).

Provides unbiased inner product estimation using 1-bit sign quantization
of the PolarQuant residual projected through a randomized Hadamard transform.

Mathematical basis:
  For residual r = w_rot - w_deq and structured projection S = H·D
  (normalized Hadamard × random diagonal signs):

  Quantize:   b = sign(S · r) ∈ {-1, +1}^d
  Dequantize: r̃ = α · ||r|| · S^T · b   where α = √(π/2)/√d

  Unbiasedness: E[⟨r̃, x⟩] = ⟨r, x⟩  (exact for Gaussian projections,
  near-exact for randomized Hadamard by concentration of measure).

Storage overhead: 1 bit per element + 1 float16 per row ≈ 1.004 bits/element.
"""

import math

import mlx.core as mx

from .rotation import _find_hadamard_block_size


def _blockwise_hadamard(x: mx.array) -> mx.array:
    """Apply Hadamard transform along the last axis, using blockwise fallback
    for dimensions not directly supported by mx.hadamard_transform."""
    dim = x.shape[-1]
    block_size = _find_hadamard_block_size(dim)
    scale = 1.0 / math.sqrt(block_size)
    if block_size == dim:
        return mx.hadamard_transform(x, scale=scale)
    n_blocks = dim // block_size
    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], n_blocks, block_size)
    x = mx.hadamard_transform(x, scale=scale)
    return x.reshape(orig_shape)


def _generate_qjl_signs(dim: int, seed: int) -> mx.array:
    """Generate deterministic random ±1 signs for the QJL projection."""
    key = mx.random.key(seed)
    b = mx.random.bernoulli(key=key, shape=(dim,))
    return (2 * b.astype(mx.float32) - 1).astype(mx.float16)


def pack_1bit(bits_array: mx.array) -> mx.array:
    """Pack a {0,1} array into uint32 (32 elements per uint32).

    Args:
        bits_array: (..., N) uint8 array of 0/1 values.

    Returns:
        (..., ceil(N/32)) uint32 packed array.
    """
    *batch_shape, n = bits_array.shape
    # Pad to multiple of 32
    remainder = n % 32
    if remainder != 0:
        pad_size = 32 - remainder
        bits_array = mx.concatenate(
            [bits_array, mx.zeros((*batch_shape, pad_size), dtype=bits_array.dtype)],
            axis=-1,
        )
        n = n + pad_size

    n_packed = n // 32
    bits_array = bits_array.astype(mx.uint32).reshape(*batch_shape, n_packed, 32)

    packed = mx.zeros((*batch_shape, n_packed), dtype=mx.uint32)
    for i in range(32):
        packed = packed | (bits_array[..., i] << i)
    return packed


def unpack_1bit(packed: mx.array, count: int) -> mx.array:
    """Unpack uint32 array back to {0,1} values.

    Args:
        packed: (..., M) uint32 packed array.
        count: Number of bits to unpack.

    Returns:
        (..., count) uint8 array of 0/1 values.
    """
    *batch_shape, m = packed.shape
    packed_expanded = mx.expand_dims(packed, axis=-1)  # (..., M, 1)
    shifts = mx.arange(32, dtype=mx.uint32)  # (32,)
    bits = (packed_expanded >> shifts) & mx.array(1, dtype=mx.uint32)  # (..., M, 32)
    bits = bits.reshape(*batch_shape, -1)  # (..., M*32)
    if count < bits.shape[-1]:
        bits = bits[..., :count]
    return bits.astype(mx.uint8)


def qjl_quantize(
    residual: mx.array,
    seed: int = 137,
) -> dict:
    """Apply QJL 1-bit quantization to the PolarQuant residual.

    For each row r of the residual matrix:
      1. Project: p = H(σ ⊙ r)  (randomized Hadamard)
      2. Store: b = (p >= 0)     (1-bit signs, packed into uint32)
      3. Store: ν = ||r||_2      (per-row L2 norm)

    Args:
        residual: (output_dims, input_dims) float — PolarQuant residual
                  in the rotated domain (w_rot - w_deq).
        seed: Random seed for QJL projection signs.

    Returns:
        Dict with keys:
            qjl_packed: (output_dims, ceil(input_dims/32)) uint32
            qjl_norms: (output_dims,) float16
            qjl_signs: (input_dims,) float16 — random ±1 projection signs
    """
    output_dims, input_dims = residual.shape
    residual = residual.astype(mx.float32)

    # Generate random signs for the structured projection S = H · diag(σ)
    qjl_signs = _generate_qjl_signs(input_dims, seed)

    # Per-row L2 norms
    qjl_norms = mx.sqrt(mx.sum(residual * residual, axis=-1))  # (output_dims,)

    # Project: S · r = H(σ ⊙ r)
    projected = residual * qjl_signs.astype(mx.float32)  # (out, d)
    projected = _blockwise_hadamard(projected)  # H applied along last dim

    # 1-bit quantization: store sign as {0, 1}
    bits_01 = (projected >= 0).astype(mx.uint8)  # (out, d)

    # Pack into uint32
    qjl_packed = pack_1bit(bits_01)

    return {
        "qjl_packed": qjl_packed,
        "qjl_norms": qjl_norms.astype(mx.float16),
        "qjl_signs": qjl_signs,
    }


def qjl_correct(
    qjl_packed: mx.array,
    qjl_norms: mx.array,
    qjl_random_signs: mx.array,
    x: mx.array,
    input_dims: int,
) -> mx.array:
    """Compute QJL residual correction for the inner product.

    Estimates ⟨r_i, x⟩ for each weight row i using stored sign bits and norms.

    The correction for row i is:
      α · ν_i · ⟨b_i, H(σ ⊙ x)⟩
    where α = √(π/2)/√d, ν_i = ||r_i||, b_i ∈ {-1,+1}^d are stored signs.

    Args:
        qjl_packed: (output_dims, packed_cols) uint32 — packed 1-bit signs.
        qjl_norms: (output_dims,) float16 — per-row residual norms.
        qjl_random_signs: (input_dims,) float16 — random ±1 projection signs.
        x: (..., input_dims) float16 — input tensor (any leading dims).
        input_dims: Original input dimension.

    Returns:
        (..., output_dims) float16 — correction tensor matching x's leading dims.
    """
    d = input_dims
    alpha = math.sqrt(math.pi / 2.0) / math.sqrt(d)

    # Flatten leading dims for computation, restore at end
    orig_shape = x.shape
    if x.ndim > 1:
        x_flat = x.reshape(-1, d)  # (batch, d)
    else:
        x_flat = x.reshape(1, d)  # (1, d)

    # Compute S · x = H(σ ⊙ x) — done once, shared across all rows
    sx = _blockwise_hadamard(
        (qjl_random_signs * x_flat).astype(mx.float32)
    ).astype(mx.float16)  # (batch, d)

    # Unpack QJL bits: {0,1} → {-1, +1}
    b_01 = unpack_1bit(qjl_packed, input_dims).astype(mx.float16)  # (out, d)
    b_signed = 2.0 * b_01 - 1.0  # {-1, +1}

    # 1-bit matvec: sx @ B_signed.T → (batch, out)
    dot = sx @ b_signed.T  # (batch, out)

    # Scale by norms and alpha: (batch, out) * (out,) → (batch, out)
    correction = mx.array(alpha, dtype=mx.float16) * qjl_norms * dot

    # Restore original leading dims
    out_dims = qjl_packed.shape[0]
    if x.ndim <= 1:
        correction = correction.squeeze(0)  # (out,)
    elif x.ndim > 2:
        correction = correction.reshape(*orig_shape[:-1], out_dims)

    return correction
