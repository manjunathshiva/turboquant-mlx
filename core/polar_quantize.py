"""PolarQuant: Rotation + Lloyd-Max codebook weight quantization pipeline.

Orchestrates the full Stage 1 quantization:
1. Apply randomized Hadamard rotation to Gaussianize weights
2. Group-wise normalization (compute per-group scale)
3. Lloyd-Max codebook quantization (optimal for Gaussian distribution)
4. Bit-pack indices for compact storage
"""

import mlx.core as mx

from turboquant_mlx.core.codebook import get_codebook, quantize_scalar, dequantize_scalar
from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight
from turboquant_mlx.core.packing import pack_indices, unpack_indices


def polar_quantize_weight(
    weight: mx.array,
    bits: int = 3,
    group_size: int = 64,
    seed: int = 42,
) -> dict:
    """Quantize a weight matrix using the PolarQuant pipeline.

    Args:
        weight: Weight matrix of shape (output_dims, input_dims).
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Number of elements per quantization group.
        seed: Random seed for Hadamard rotation signs.

    Returns:
        Dict with keys:
            packed_weight: uint32 packed indices, shape (out, packed_in)
            scales: float16 per-group scales, shape (out, n_groups)
            codebook: float16 centroids, shape (2^bits,)
            signs: float16 random signs, shape (input_dims,)
            bits: int bit-width
            group_size: int group size
            input_dims: int original input dimension
    """
    output_dims, input_dims = weight.shape

    if input_dims % group_size != 0:
        raise ValueError(
            f"input_dims ({input_dims}) must be divisible by group_size ({group_size})"
        )

    n_groups = input_dims // group_size

    # 1. Generate random signs for randomized Hadamard
    signs = generate_random_signs(input_dims, seed=seed)

    # 2. Rotate weight matrix (Gaussianize the distribution)
    w_rot = rotate_weight(weight.astype(mx.float32), signs.astype(mx.float32))

    # 3. Group-wise normalization
    # Reshape to (output_dims, n_groups, group_size)
    w_grouped = w_rot.reshape(output_dims, n_groups, group_size)

    centroids, boundaries = get_codebook(bits, dtype=mx.float32)

    # Per-group scale: use RMS (optimal for Gaussian-distributed values after rotation)
    # For N(0, sigma^2), RMS = sigma, and Lloyd-Max centroids are for N(0,1)
    rms = mx.sqrt(mx.mean(w_grouped * w_grouped, axis=-1, keepdims=True))  # (out, n_groups, 1)
    rms = mx.maximum(rms, mx.array(1e-7))

    # Normalize to unit variance for Lloyd-Max codebook
    w_normalized = w_grouped / rms  # (out, n_groups, gs)

    # Clip to prevent extreme outliers from saturating the codebook
    max_centroid = mx.max(mx.abs(centroids))
    w_normalized = mx.clip(w_normalized, -max_centroid * 1.5, max_centroid * 1.5)

    # 4. Lloyd-Max codebook quantization
    indices = quantize_scalar(w_normalized, boundaries)  # (out, n_groups, gs) uint8

    # Flatten groups back: (out, input_dims)
    indices_flat = indices.reshape(output_dims, input_dims)

    # 5. Pack indices into uint32
    packed_weight = pack_indices(indices_flat, bits)

    # Store scales as (out, n_groups) in float16
    # Scale = RMS, so dequant is: centroid * rms
    scales_out = rms.squeeze(-1).astype(mx.float16)

    # Get codebook in float16
    codebook_f16 = centroids.astype(mx.float16)

    return {
        "packed_weight": packed_weight,
        "scales": scales_out,
        "codebook": codebook_f16,
        "signs": signs.astype(mx.float16),
        "bits": bits,
        "group_size": group_size,
        "input_dims": input_dims,
    }


def polar_dequantize_weight(
    packed_weight: mx.array,
    scales: mx.array,
    codebook: mx.array,
    bits: int,
    group_size: int,
    input_dims: int,
) -> mx.array:
    """Dequantize packed weight back to float values (without un-rotating).

    This returns the weight in the rotated domain. For inference, either:
    - Apply rotation to the input instead (preferred)
    - Call unrotate_weight() to get back to original domain

    Args:
        packed_weight: uint32 packed indices, shape (out, packed_in).
        scales: float16 per-group scales, shape (out, n_groups).
        codebook: float16 centroids, shape (2^bits,).
        bits: Quantization bit-width.
        group_size: Elements per group.
        input_dims: Original input dimension (for unpack count).

    Returns:
        Dequantized weight in rotated domain, shape (out, input_dims), float16.
    """
    output_dims = packed_weight.shape[0]
    n_groups = input_dims // group_size

    # Unpack indices
    indices = unpack_indices(packed_weight, bits, input_dims)  # (out, input_dims)
    indices = indices.reshape(output_dims, input_dims)

    # Dequantize via codebook lookup
    w_deq = dequantize_scalar(indices, codebook)  # (out, input_dims) float16

    # Apply per-group scales
    w_deq = w_deq.reshape(output_dims, n_groups, group_size)
    scales_expanded = mx.expand_dims(scales, axis=-1)  # (out, n_groups, 1)
    w_deq = w_deq * scales_expanded

    return w_deq.reshape(output_dims, input_dims)
