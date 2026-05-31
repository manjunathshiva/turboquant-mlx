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


# Cap on the number of weight elements quantized in a single MLX graph. Large
# tensors (e.g. a big-vocab lm_head — 248k x 5120 ~ 1.3B params) are split along
# the output-row axis into blocks of at most this many elements, each evaluated
# as its own command buffer, so a single GPU submission never exceeds the Metal
# command-buffer watchdog timeout (kIOGPUCommandBufferCallbackErrorTimeout).
# Bit-identical to a single-pass quantization: each output row is quantized
# independently (rotation is per-row; normalization + codebook assignment are
# per-group within a row), so splitting rows and concatenating changes nothing.
_MAX_QUANT_BLOCK_ELEMS = 64_000_000


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

    # Materialize the weight off disk BEFORE any GPU compute. With a lazily
    # mmap'd checkpoint on slow storage (e.g. a USB HDD), MLX would otherwise
    # fuse the multi-second weight read into the rotation/quantize command
    # buffer, and the Metal GPU watchdog kills any command buffer that stalls
    # on I/O that long (kIOGPUCommandBufferCallbackErrorTimeout). Forcing the
    # read as its own eval keeps the subsequent GPU kernels pure-compute (fast).
    mx.eval(weight)

    # 1. Generate random signs for randomized Hadamard (shared across all rows)
    signs = generate_random_signs(input_dims, seed=seed)
    signs_f32 = signs.astype(mx.float32)

    centroids, boundaries = get_codebook(bits, dtype=mx.float32)
    # Clip threshold to prevent extreme outliers from saturating the codebook
    max_centroid = mx.max(mx.abs(centroids))

    # Quantize the output-row axis in blocks so no single command buffer
    # exceeds the GPU watchdog timeout (see _MAX_QUANT_BLOCK_ELEMS). Each block
    # is evaluated on its own; rows quantize independently, so concatenating the
    # per-block results is bit-identical to a single pass. Small tensors stay a
    # single block (no overhead).
    rows_per_block = max(1, _MAX_QUANT_BLOCK_ELEMS // input_dims)
    packed_parts = []
    scale_parts = []
    for r0 in range(0, output_dims, rows_per_block):
        wb = weight[r0:r0 + rows_per_block].astype(mx.float32)
        rows = wb.shape[0]

        # 2. Rotate weight rows (Gaussianize the distribution)
        w_rot = rotate_weight(wb, signs_f32)

        # 3. Group-wise normalization: reshape to (rows, n_groups, group_size).
        # Per-group scale uses RMS (= sigma for N(0, sigma^2) after rotation;
        # Lloyd-Max centroids are for N(0,1)).
        w_grouped = w_rot.reshape(rows, n_groups, group_size)
        rms = mx.sqrt(mx.mean(w_grouped * w_grouped, axis=-1, keepdims=True))
        rms = mx.maximum(rms, mx.array(1e-7))
        w_normalized = w_grouped / rms
        w_normalized = mx.clip(w_normalized, -max_centroid * 1.5, max_centroid * 1.5)

        # 4. Lloyd-Max codebook quantization, then 5. pack indices into uint32
        indices = quantize_scalar(w_normalized, boundaries)
        indices_flat = indices.reshape(rows, input_dims)
        packed = pack_indices(indices_flat, bits)
        scales_b = rms.squeeze(-1).astype(mx.float16)  # (rows, n_groups)

        # Force this block out as its own command buffer (bounds GPU work).
        mx.eval(packed, scales_b)
        packed_parts.append(packed)
        scale_parts.append(scales_b)

    packed_weight = (packed_parts[0] if len(packed_parts) == 1
                     else mx.concatenate(packed_parts, axis=0))
    # Scale = RMS, so dequant is: centroid * rms
    scales_out = (scale_parts[0] if len(scale_parts) == 1
                  else mx.concatenate(scale_parts, axis=0))

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
