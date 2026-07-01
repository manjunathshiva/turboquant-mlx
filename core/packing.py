"""Bit-packing utilities for quantization indices.

Packs b-bit integer indices into uint32 arrays for compact storage.
Follows LSB-first convention to match MLX's internal packing format.

Also provides base-3 (trit) packing for the ternary {-c, 0, +c} tier: 20
ternary indices fit in one uint32 (3**20 < 2**32), giving 32/20 = 1.6 bits per
weight vs 2.0 for the bit-packed 2-bit slot -- the memory win that lets a
ternary-expert model fit resident. The layout is little-endian base 3:
packed = sum_i trit[i] * 3**i.
"""

import mlx.core as mx

# 20 trits per uint32: 3**20 = 3,486,784,401 <= 2**32 = 4,294,967,296.
TRITS_PER_U32 = 20
_POW3 = [3 ** i for i in range(TRITS_PER_U32)]  # little-endian base-3 place values


def pack_trits(indices: mx.array) -> mx.array:
    """Pack ternary indices (values in {0, 1, 2}) as base-3 digits into uint32.

    Each uint32 holds 20 trits (LSB place first): ``packed = sum_i t[i]*3**i``.

    Args:
        indices: Integer indices of shape (..., N), values in [0, 3).

    Returns:
        Packed uint32 array of shape (..., ceil(N / 20)).
    """
    *batch_shape, n = indices.shape
    remainder = n % TRITS_PER_U32
    if remainder != 0:
        pad = TRITS_PER_U32 - remainder
        indices = mx.concatenate(
            [indices, mx.zeros((*batch_shape, pad), dtype=indices.dtype)], axis=-1)
        n += pad

    n_packed = n // TRITS_PER_U32
    idx = indices.astype(mx.uint32).reshape(*batch_shape, n_packed, TRITS_PER_U32)

    # packed[..., j] = sum_i idx[..., j, i] * 3**i, vectorized over the 20 trits.
    # Max sum is 3**20 - 1 = 3,486,784,400 < 2**32, so uint32 never overflows.
    pow3 = mx.array(_POW3, dtype=mx.uint32)  # (20,)
    return mx.sum(idx * pow3, axis=-1)  # (..., n_packed)


def unpack_trits(packed: mx.array, count: int) -> mx.array:
    """Unpack base-3 (trit) uint32 array back to ternary indices.

    Args:
        packed: Packed uint32 array of shape (..., M).
        count: Number of trits to return (<= M * 20).

    Returns:
        Unpacked indices of shape (..., count), dtype uint8, values in {0,1,2}.
    """
    *batch_shape, _ = packed.shape
    pe = mx.expand_dims(packed, axis=-1)  # (..., M, 1)
    pow3 = mx.array(_POW3, dtype=mx.uint32)  # (20,)
    trits = (pe // pow3) % mx.array(3, dtype=mx.uint32)  # (..., M, 20)
    trits = trits.reshape(*batch_shape, -1)
    if count < trits.shape[-1]:
        trits = trits[..., :count]
    return trits.astype(mx.uint8)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack b-bit indices into uint32 arrays.

    Each uint32 stores floor(32/bits) indices, packed LSB-first.

    Args:
        indices: Integer indices of shape (..., N) where N is divisible
                 by elements_per_uint32. Values must be in [0, 2^bits).
        bits: Bits per index (2..8 inclusive).

    Returns:
        Packed uint32 array of shape (..., N * bits / 32).
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

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
        bits: Bits per index (2..8 inclusive).
        count: Number of indices to unpack (may be less than M * 32/bits).

    Returns:
        Unpacked indices of shape (..., count), dtype uint8.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

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
