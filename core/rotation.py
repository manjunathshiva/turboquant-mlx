"""Randomized Hadamard rotation for Gaussianizing weight distributions.

The key insight: multiplying by a randomized Hadamard matrix (H @ diag(signs))
transforms any weight distribution into one that is approximately Gaussian
in each coordinate, enabling optimal Lloyd-Max quantization.
"""

import math
import mlx.core as mx


def generate_random_signs(dim: int, seed: int = 42) -> mx.array:
    """Generate deterministic random +1/-1 signs for randomized Hadamard.

    Args:
        dim: Dimension of the sign vector.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (dim,) with values in {-1, +1}, dtype float16.
    """
    key = mx.random.key(seed)
    # Generate uniform random, threshold at 0.5 to get +1/-1
    r = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(r < 0.5, mx.array(-1.0), mx.array(1.0))
    return signs.astype(mx.float16)


def _find_hadamard_block_size(dim: int) -> int:
    """Find the largest Hadamard-compatible block size that divides dim.

    mx.hadamard_transform supports n = m * 2^k for m in {1, 12, 20, 28}.
    We find the largest such n that divides dim.
    """
    valid_m = [1, 12, 20, 28]
    best = 1
    for m in valid_m:
        k = 0
        while True:
            size = m * (2 ** k)
            if size > dim:
                break
            if dim % size == 0:
                best = max(best, size)
            k += 1
    return best


def rotate_weight(
    weight: mx.array,
    signs: mx.array,
    scale: float = None,
) -> mx.array:
    """Apply randomized Hadamard rotation to weight matrix columns.

    Computes: W_rot = hadamard(W * signs[None, :]) along last axis.
    If the last dimension is not Hadamard-compatible, uses blockwise rotation.

    Args:
        weight: Weight matrix of shape (output_dims, input_dims).
        signs: Random +1/-1 signs of shape (input_dims,).
        scale: Hadamard normalization scale. Default 1/sqrt(input_dims).

    Returns:
        Rotated weight matrix of same shape.
    """
    input_dims = weight.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(input_dims)

    # Apply random sign flip
    w = weight * signs

    # Check if dimension is directly Hadamard-compatible
    block_size = _find_hadamard_block_size(input_dims)

    if block_size == input_dims:
        # Direct Hadamard transform
        return mx.hadamard_transform(w, scale=scale)
    else:
        # Blockwise Hadamard: split into blocks, transform each
        n_blocks = input_dims // block_size
        block_scale = 1.0 / math.sqrt(block_size)
        # Reshape to (..., n_blocks, block_size), transform, reshape back
        orig_shape = w.shape
        w = w.reshape(*orig_shape[:-1], n_blocks, block_size)
        w = mx.hadamard_transform(w, scale=block_scale)
        return w.reshape(orig_shape)


def rotate_input(
    x: mx.array,
    signs: mx.array,
    scale: float = None,
) -> mx.array:
    """Apply randomized Hadamard rotation to input activations (online).

    This is the inverse-transpose operation needed at inference time
    for layers where rotation cannot be fused into normalization.

    Args:
        x: Input tensor of shape (..., input_dims).
        signs: Random +1/-1 signs of shape (input_dims,).
        scale: Hadamard normalization scale. Default 1/sqrt(input_dims).

    Returns:
        Rotated input of same shape.
    """
    input_dims = x.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(input_dims)

    x = x * signs
    block_size = _find_hadamard_block_size(input_dims)

    if block_size == input_dims:
        return mx.hadamard_transform(x, scale=scale)
    else:
        n_blocks = input_dims // block_size
        block_scale = 1.0 / math.sqrt(block_size)
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], n_blocks, block_size)
        x = mx.hadamard_transform(x, scale=block_scale)
        return x.reshape(orig_shape)


def fuse_rotation_into_norm(
    norm_weight: mx.array,
    signs: mx.array,
) -> mx.array:
    """Fuse Hadamard rotation into RMSNorm/LayerNorm weight.

    Since RMSNorm computes: y = (x / rms(x)) * weight, and our rotation
    is: x_rot = H @ diag(signs) @ x, we can absorb the rotation into
    the norm weight: weight_new = H @ diag(signs) @ diag(weight)
    = H @ diag(signs * weight).

    But actually the correct fusion for y = norm(x) * w followed by
    W_rot @ y is: we need x_rot = H @ (signs * (norm(x) * w)) which equals
    H @ (signs * w) * norm(x). So the fused norm weight is:
    w_fused = hadamard(signs * w).

    Args:
        norm_weight: LayerNorm/RMSNorm weight of shape (dim,).
        signs: Random +1/-1 signs of shape (dim,).

    Returns:
        Fused norm weight of shape (dim,).
    """
    dim = norm_weight.shape[0]
    scale = 1.0 / math.sqrt(dim)
    fused = signs * norm_weight
    block_size = _find_hadamard_block_size(dim)

    if block_size == dim:
        return mx.hadamard_transform(fused.reshape(1, -1), scale=scale).reshape(-1)
    else:
        n_blocks = dim // block_size
        block_scale = 1.0 / math.sqrt(block_size)
        fused = fused.reshape(n_blocks, block_size)
        fused = mx.hadamard_transform(fused, scale=block_scale)
        return fused.reshape(-1)
