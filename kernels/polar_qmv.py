"""Fused Metal kernel for PolarQuant matrix-vector multiplication.

Combines index unpacking, codebook lookup, scale multiplication, and
dot product into a single Metal kernel — avoiding materialization of
the full dequantized weight matrix.  Optimized for the decode path
(single-token inference, batch size 1).

Uses threadgroup parallelism: 32 threads cooperate on each output row,
dividing groups across threads and using shared memory for reduction.

Memory-bandwidth savings vs software dequant:
  - Reads compressed weights (b bits/elem) instead of FP16 (16 bits)
  - For 3-bit/group32: ~5.4 MB per 2880×2880 layer vs ~16 MB for FP16
  - ~3× less memory traffic → proportional speedup on bandwidth-bound decode
"""

import math
from typing import Optional

import mlx.core as mx

# Cache compiled kernels keyed by (bits, group_size)
_kernel_cache: dict[tuple[int, int], object] = {}

THREADS_PER_ROW = 32


def _build_kernel_source(bits: int, group_size: int) -> str:
    """Generate Metal shader with threadgroup parallel reduction."""
    n_codes = 1 << bits
    elems_per_u32 = 32 // bits
    mask = (1 << bits) - 1

    return f"""
    // Each threadgroup handles one output row.
    // thread_position_in_threadgroup.x = lane within the row (0..31)
    // threadgroup_position_in_grid.x = which row
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint n_rows = packed_weight_shape[0];
    if (row >= n_rows) return;

    // Shared memory for reduction
    threadgroup float shared_sums[{THREADS_PER_ROW}];

    // Load codebook into registers ({n_codes} entries)
    float cb[{n_codes}];
    for (uint i = 0; i < {n_codes}u; i++) {{
        cb[i] = float(codebook[i]);
    }}

    uint n_groups = scales_shape[1];
    uint pw_stride = packed_weight_strides[0];

    float accum = 0.0f;

    // Distribute groups across threads in the threadgroup
    for (uint g = lane; g < n_groups; g += {THREADS_PER_ROW}u) {{
        float scale = float(scales[row * n_groups + g]);
        uint base_col = g * {group_size}u;
        float group_accum = 0.0f;

        for (uint e = 0; e < {group_size}u; e++) {{
            uint col = base_col + e;
            uint packed_col = col / {elems_per_u32}u;
            uint bit_pos = (col % {elems_per_u32}u) * {bits}u;

            uint packed_val = packed_weight[row * pw_stride + packed_col];
            uint code_idx = (packed_val >> bit_pos) & {mask}u;

            group_accum += cb[code_idx] * float(x[col]);
        }}

        accum += group_accum * scale;
    }}

    // Write partial sum to shared memory
    shared_sums[lane] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    if (lane < 16u) shared_sums[lane] += shared_sums[lane + 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8u) shared_sums[lane] += shared_sums[lane + 8u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 4u) shared_sums[lane] += shared_sums[lane + 4u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 2u) shared_sums[lane] += shared_sums[lane + 2u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {{
        out[row] = T(shared_sums[0] + shared_sums[1]);
    }}
"""


def _get_kernel(bits: int, group_size: int):
    """Get (or compile and cache) the Metal kernel for given parameters."""
    key = (bits, group_size)
    if key not in _kernel_cache:
        source = _build_kernel_source(bits, group_size)
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"polar_qmv_{bits}bit_gs{group_size}",
            input_names=["packed_weight", "scales", "codebook", "x"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def polar_qmv(
    packed_weight: mx.array,
    scales: mx.array,
    codebook: mx.array,
    x: mx.array,
    bits: int,
    group_size: int,
) -> mx.array:
    """Fused quantized matrix-vector product via custom Metal kernel.

    Computes  y = dequant(packed_weight, scales, codebook) @ x
    without materializing the full FP16 weight matrix.

    Uses threadgroup parallelism: 32 threads cooperate per output row
    with shared memory tree reduction.

    Args:
        packed_weight: (output_dims, packed_cols) uint32 — packed b-bit indices.
        scales: (output_dims, n_groups) float16 — per-group RMS scales.
        codebook: (n_codes,) float16 — Lloyd-Max centroids.
        x: (input_dims,) or (1, input_dims) float16 — input vector.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Elements per quantization group.

    Returns:
        (output_dims,) or (1, output_dims) float16 — result vector.
    """
    had_batch = False
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError(
                f"polar_qmv is for single-token decode (batch=1), got {x.shape}"
            )
        x = x.squeeze(0)
        had_batch = True

    output_dims = packed_weight.shape[0]
    kernel = _get_kernel(bits, group_size)

    # grid = total threads, so multiply by THREADS_PER_ROW to get
    # one threadgroup (of 32 threads) per output row
    outputs = kernel(
        inputs=[packed_weight, scales, codebook, x],
        template=[("T", x.dtype)],
        grid=(output_dims * THREADS_PER_ROW, 1, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        output_shapes=[(output_dims,)],
        output_dtypes=[x.dtype],
    )

    out = outputs[0]
    if had_batch:
        out = mx.expand_dims(out, axis=0)
    return out
