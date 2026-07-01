"""Precomputed Lloyd-Max optimal codebooks for N(0,1) distribution.

These centroids are mathematically optimal (minimize MSE) for quantizing
Gaussian-distributed values. After Hadamard rotation, weight coordinates
follow approximately N(0, sigma^2), so scaling these by sigma gives
optimal quantization for any variance.

Tables for bits 2/3/4 are precomputed for backwards compatibility.
Higher bit widths (5-8) are computed on first request via Lloyd-Max
iteration on N(0,1) using closed-form truncated-Gaussian conditional
means, then cached.
"""

import math

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
    1: 0.3633802276,  # 1 - 2/pi, optimal 1-bit (sign + single magnitude)
    2: 0.1174818198,
    3: 0.0345477324,
    4: 0.0095009960,
}

# Cache for mx.array versions
_centroids_cache: dict[tuple[int, mx.Dtype], mx.array] = {}
_boundaries_cache: dict[tuple[int, mx.Dtype], mx.array] = {}


_SUPPORTED_BITS = range(1, 9)  # 1..8 inclusive (1-bit = sign codebook for sub-2bit tiers)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _compute_lloyd_max_gaussian(bits: int, max_iter: int = 500, tol: float = 1e-12):
    """Iterative Lloyd-Max for N(0,1).

    Uses closed-form conditional means of truncated Gaussian regions to
    update centroids each step. Converges in ~30 iterations for bits up
    to 8. Returns (centroids, boundaries) as Python lists.
    """
    K = 1 << bits  # 2 ** bits
    # Initialize centroids on a uniform grid spanning ~±3σ
    centroids = [(-1.0 + 2.0 * (i + 0.5) / K) * 3.0 for i in range(K)]

    for _ in range(max_iter):
        # Boundaries are midpoints of adjacent centroids
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(K - 1)]

        new_centroids = [0.0] * K

        # First region: (-inf, b0]
        b0 = boundaries[0]
        cdf_b0 = _norm_cdf(b0)
        if cdf_b0 > 1e-30:
            new_centroids[0] = -_norm_pdf(b0) / cdf_b0
        else:
            new_centroids[0] = centroids[0]

        # Last region: [b_last, +inf)
        bL = boundaries[-1]
        sf_bL = 1.0 - _norm_cdf(bL)
        if sf_bL > 1e-30:
            new_centroids[-1] = _norm_pdf(bL) / sf_bL
        else:
            new_centroids[-1] = centroids[-1]

        # Middle regions: [b[i-1], b[i]]
        for i in range(1, K - 1):
            lo = boundaries[i - 1]
            hi = boundaries[i]
            num = _norm_pdf(lo) - _norm_pdf(hi)
            den = _norm_cdf(hi) - _norm_cdf(lo)
            if den > 1e-30:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = centroids[i]

        max_delta = max(abs(a - b) for a, b in zip(new_centroids, centroids))
        centroids = new_centroids
        if max_delta < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(K - 1)]
    return centroids, boundaries


def _ensure_table(bits: int):
    """Populate CENTROIDS[bits] / BOUNDARIES[bits] on first request."""
    if bits in CENTROIDS:
        return
    if bits not in _SUPPORTED_BITS:
        raise ValueError(
            f"Unsupported bit-width {bits}. Must be in {list(_SUPPORTED_BITS)}."
        )
    centroids, boundaries = _compute_lloyd_max_gaussian(bits)
    CENTROIDS[bits] = centroids
    BOUNDARIES[bits] = boundaries


def get_codebook(bits: int, dtype: mx.Dtype = mx.float16) -> tuple[mx.array, mx.array]:
    """Get Lloyd-Max centroids and boundaries for a given bit-width.

    Args:
        bits: Quantization bit-width (2..8). Tables for 2/3/4 are
            precomputed; 5..8 are computed on first request and cached.
        dtype: Output dtype for the arrays.

    Returns:
        (centroids, boundaries) as mx.arrays of shape (2^bits,) and
        (2^bits - 1,).
    """
    _ensure_table(bits)

    c_key = (bits, dtype)
    if c_key not in _centroids_cache:
        _centroids_cache[c_key] = mx.array(CENTROIDS[bits], dtype=dtype)
    if c_key not in _boundaries_cache:
        _boundaries_cache[c_key] = mx.array(BOUNDARIES[bits], dtype=dtype)

    return _centroids_cache[c_key], _boundaries_cache[c_key]


# Ternary (1.58-bit) codebook: {-c, 0, +c}, padded to 4 entries so it rides the
# existing 2-bit storage / packing / kernel / load paths unchanged (emitted
# indices are only ever in {0, 1, 2}). Optimal Lloyd-Max levels for N(0,1):
# c = 1.22401, decision threshold c/2, normalized MSE 0.1898 -- half of 1-bit's
# 0.363. The zero level is what clears the 1-bit (2-centroid) cardinality wall.
_TERNARY_C = 1.22401


def get_ternary_codebook(dtype: mx.Dtype = mx.float16) -> tuple[mx.array, mx.array]:
    """Ternary {-c, 0, +c} codebook stored in a 2-bit (4-entry) slot.

    Returns a 4-entry centroid array ``[-c, 0, +c, +c]`` and 3 boundaries
    ``[-c/2, +c/2, BIG]``. Quantization emits only indices {0, 1, 2}; the 4th
    centroid is padding (a duplicate of +c, so even a stray index 3 is harmless)
    and its boundary is far above the clip range so it is never selected. Because
    the shape matches a normal 2-bit layer, the packing, Metal kernel, on-disk
    format, and loader need no changes -- ternary experts ride the 2-bit path.
    This is the viable data-free sub-2-bit tier (see scripts/onebit).
    """
    c = _TERNARY_C
    centroids = mx.array([-c, 0.0, c, c], dtype=dtype)
    # Last boundary sits well above the ±1.5c quantize clip, so index 3 (the
    # padding centroid) is never chosen; kept finite to stay valid in float16.
    boundaries = mx.array([-c / 2.0, c / 2.0, 30.0], dtype=dtype)
    return centroids, boundaries


def get_trit_codebook(dtype: mx.Dtype = mx.float16) -> tuple[mx.array, mx.array]:
    """Ternary {-c, 0, +c} codebook for genuine base-3 (trit) packing.

    Returns a 3-entry centroid array ``[-c, 0, +c]`` and 2 boundaries
    ``[-c/2, +c/2]``; ``quantize_scalar`` emits indices {0, 1, 2}. The 3-entry
    codebook length is the self-describing marker for the trit path: the loader
    keys off ``codebook.shape[-1] == 3`` to decode base-3 packing (20 trits per
    uint32, ~1.6 bpw) instead of the 2-bit slot (2.0 bpw) used by the 4-entry
    ``get_ternary_codebook``. Same centroids, but the real sub-2-bit storage win.
    """
    c = _TERNARY_C
    centroids = mx.array([-c, 0.0, c], dtype=dtype)
    boundaries = mx.array([-c / 2.0, c / 2.0], dtype=dtype)
    return centroids, boundaries


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
