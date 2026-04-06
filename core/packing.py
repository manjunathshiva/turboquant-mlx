"""Bit-packing utilities for quantization indices.

Packs b-bit integer indices into uint32 arrays for compact storage.
Follows LSB-first convention to match MLX's internal packing format.
"""

import mlx.core as mx


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack b-bit indices into uint32 arrays.

    Each uint32 stores floor(32/bits) indices, packed LSB-first.

    Args:
        indices: Integer indices of shape (..., N) where N is divisible
                 by elements_per_uint32. Values must be in [0, 2^bits).
        bits: Bits per index (2, 3, or 4).

    Returns:
        Packed uint32 array of shape (..., N * bits / 32).
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")

    elems_per_u32 = 32 // bits
    *batch_shape, n = indices.shape

    # For 3-bit: 10 elements per uint32 (30 bits, 2 wasted)
    # Pad N to multiple of elems_per_u32 if needed
    remainder = n % elems_per_u32
    if remainder != 0:
        pad_size = elems_per_u32 - remainder
        indices = mx.concatenate(
            [indices, mx.zeros((*batch_shape, pad_size), dtype=indices.dtype)],
            axis=-1,
        )
        n = n + pad_size

    n_packed = n // elems_per_u32
    indices = indices.astype(mx.uint32)

    # Reshape to (..., n_packed, elems_per_u32) for vectorized packing
    indices = indices.reshape(*batch_shape, n_packed, elems_per_u32)

    # Pack: shift each element by (index_within_group * bits) and OR together
    packed = mx.zeros((*batch_shape, n_packed), dtype=mx.uint32)
    for i in range(elems_per_u32):
        packed = packed | (indices[..., i] << (i * bits))

    return packed


def unpack_indices(packed: mx.array, bits: int, count: int) -> mx.array:
    """Unpack uint32 array back to b-bit indices.

    Args:
        packed: Packed uint32 array of shape (..., M).
        bits: Bits per index (2, 3, or 4).
        count: Number of indices to unpack (may be less than M * 32/bits).

    Returns:
        Unpacked indices of shape (..., count), dtype uint8.
    """
    if bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")

    elems_per_u32 = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    *batch_shape, m = packed.shape

    # Expand each uint32 into elems_per_u32 indices
    packed_expanded = mx.expand_dims(packed, axis=-1)  # (..., M, 1)
    shifts = mx.arange(elems_per_u32, dtype=mx.uint32) * bits  # (elems_per_u32,)
    indices = (packed_expanded >> shifts) & mask  # (..., M, elems_per_u32)
    indices = indices.reshape(*batch_shape, -1)  # (..., M * elems_per_u32)

    # Trim to requested count
    if count < indices.shape[-1]:
        indices = indices[..., :count]

    return indices.astype(mx.uint8)
