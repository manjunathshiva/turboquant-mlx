"""Precomputed Lloyd-Max optimal codebooks for N(0,1) distribution.

These centroids are mathematically optimal (minimize MSE) for quantizing
Gaussian-distributed values. After Hadamard rotation, weight coordinates
follow approximately N(0, sigma^2), so scaling these by sigma gives
optimal quantization for any variance.
"""

import mlx.core as mx

# Lloyd-Max optimal centroids for standard normal N(0,1)
# Computed via iterative Lloyd-Max algorithm with scipy reference
CENTROIDS = {
    2: [
        -1.5104174569,
        -0.4527799975,
        0.4527799975,
        1.5104174569,
    ],
    3: [
        -2.1519452850,
        -1.3439090860,
        -0.7560051861,
        -0.2450941497,
        0.2450941497,
        0.7560051861,
        1.3439090860,
        2.1519452850,
    ],
    4: [
        -2.7332986608,
        -2.0698191883,
        -1.6188648437,
        -1.2570025732,
        -0.9430078288,
        -0.6572738605,
        -0.3883729570,
        -0.1285059463,
        0.1285059463,
        0.3883729570,
        0.6572738605,
        0.9430078288,
        1.2570025732,
        1.6188648437,
        2.0698191883,
        2.7332986608,
    ],
}

# Decision boundaries (midpoints between adjacent centroids)
BOUNDARIES = {
    2: [
        -0.9815987272,
        0.0,
        0.9815987272,
    ],
    3: [
        -1.7479271855,
        -1.0499571360,
        -0.5005496679,
        0.0,
        0.5005496679,
        1.0499571360,
        1.7479271855,
    ],
    4: [
        -2.4015589245,
        -1.8443420160,
        -1.4379337084,
        -1.1000052010,
        -0.8001408446,
        -0.5228234087,
        -0.2584394517,
        0.0,
        0.2584394517,
        0.5228234087,
        0.8001408446,
        1.1000052010,
        1.4379337084,
        1.8443420160,
        2.4015589245,
    ],
}

# MSE distortion for each bit-width (for unit-variance Gaussian)
MSE = {
    2: 0.1174818198,
    3: 0.0345477324,
    4: 0.0095009960,
}

# Cache for mx.array versions
_centroids_cache: dict[tuple[int, mx.Dtype], mx.array] = {}
_boundaries_cache: dict[tuple[int, mx.Dtype], mx.array] = {}


def get_codebook(bits: int, dtype: mx.Dtype = mx.float16) -> tuple[mx.array, mx.array]:
    """Get Lloyd-Max centroids and boundaries for a given bit-width.

    Args:
        bits: Quantization bit-width (2, 3, or 4).
        dtype: Output dtype for the arrays.

    Returns:
        (centroids, boundaries) as mx.arrays of shape (2^bits,) and (2^bits - 1,).
    """
    if bits not in CENTROIDS:
        raise ValueError(f"Unsupported bit-width {bits}. Must be 2, 3, or 4.")

    c_key = (bits, dtype)
    if c_key not in _centroids_cache:
        _centroids_cache[c_key] = mx.array(CENTROIDS[bits], dtype=dtype)
    if c_key not in _boundaries_cache:
        _boundaries_cache[c_key] = mx.array(BOUNDARIES[bits], dtype=dtype)

    return _centroids_cache[c_key], _boundaries_cache[c_key]


def quantize_scalar(values: mx.array, boundaries: mx.array) -> mx.array:
    """Quantize values using precomputed decision boundaries.

    For each value, counts how many boundaries it exceeds to determine
    the bin index. Equivalent to searchsorted but uses MLX primitives.

    Args:
        values: Input values to quantize. Any shape.
        boundaries: Sorted decision boundaries of shape (2^bits - 1,).

    Returns:
        Integer indices of shape matching values, dtype uint8.
    """
    orig_shape = values.shape
    flat = values.reshape(-1, 1)  # (N, 1)
    # Compare each value against all boundaries: sum of (value >= boundary)
    # boundaries shape: (B,) -> (1, B) for broadcasting
    indices = (flat >= boundaries.reshape(1, -1)).sum(axis=-1)  # (N,)
    return indices.reshape(orig_shape).astype(mx.uint8)


def dequantize_scalar(indices: mx.array, centroids: mx.array) -> mx.array:
    """Dequantize indices back to centroid values.

    Args:
        indices: Integer indices from quantize_scalar. Any shape.
        centroids: Codebook centroids of shape (2^bits,).

    Returns:
        Dequantized values of shape matching indices.
    """
    return mx.take(centroids, indices.reshape(-1).astype(mx.uint32), axis=0).reshape(indices.shape)
