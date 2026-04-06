"""Fused Metal kernel for PolarQuant expert-routed matrix-vector multiply.

Combines index unpacking, codebook lookup, scale multiplication, and
dot product for MoE expert layers — reading directly from packed weights
without materializing the full dequantized expert weight matrices.

Uses threadgroup parallelism: 32 threads cooperate on each (expert, row)
pair with shared memory tree reduction.

For single-token decode with k=4 experts (GPT-OSS), this avoids
dequantizing all 32 experts and instead only touches the 4 selected ones.

Memory savings: reads k/N of expert weights vs dequantize-all.
"""

import math
from typing import Optional

import mlx.core as mx

_kernel_cache: dict[tuple[int, int], object] = {}

THREADS_PER_ROW = 32


def _build_kernel_source(bits: int, group_size: int) -> str:
    """Generate Metal shader for fused expert-routed polar QMV with
    threadgroup parallel reduction.

    Each threadgroup handles one (expert, row) pair.
    Grid: (k * output_dims, 1, 1) threadgroups.
    """
    n_codes = 1 << bits
    elems_per_u32 = 32 // bits
    mask = (1 << bits) - 1

    return f"""
    // Each threadgroup handles one (expert_local, row) pair.
    uint lane = thread_position_in_threadgroup.x;
    uint work_id = threadgroup_position_in_grid.x;

    uint out_dims = packed_weight_shape[1];
    uint k = indices_shape[0];
    uint total_work = k * out_dims;
    if (work_id >= total_work) return;

    uint expert_local = work_id / out_dims;
    uint row = work_id % out_dims;

    // Look up actual expert index
    uint expert_id = indices[expert_local];

    // Shared memory for reduction
    threadgroup float shared_sums[{THREADS_PER_ROW}];

    // Load codebook into registers ({n_codes} entries)
    float cb[{n_codes}];
    for (uint i = 0; i < {n_codes}u; i++) {{
        cb[i] = float(codebook[i]);
    }}

    uint n_groups = scales_shape[2];
    uint pw_cols = packed_weight_shape[2];

    // Base offsets for this expert and row
    uint pw_base = expert_id * out_dims * pw_cols + row * pw_cols;
    uint sc_base = expert_id * out_dims * n_groups + row * n_groups;

    float accum = 0.0f;

    // Distribute groups across threads
    for (uint g = lane; g < n_groups; g += {THREADS_PER_ROW}u) {{
        float scale = float(scales[sc_base + g]);
        uint base_col = g * {group_size}u;
        float group_accum = 0.0f;

        for (uint e = 0; e < {group_size}u; e++) {{
            uint col = base_col + e;
            uint packed_col = col / {elems_per_u32}u;
            uint bit_pos = (col % {elems_per_u32}u) * {bits}u;

            uint packed_val = packed_weight[pw_base + packed_col];
            uint code_idx = (packed_val >> bit_pos) & {mask}u;

            group_accum += cb[code_idx] * float(x[col]);
        }}

        accum += group_accum * scale;
    }}

    // Write partial sum to shared memory
    shared_sums[lane] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    if (lane < 16u) shared_sums[lane] += shared_sums[lane + 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8u) shared_sums[lane] += shared_sums[lane + 8u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 4u) shared_sums[lane] += shared_sums[lane + 4u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 2u) shared_sums[lane] += shared_sums[lane + 2u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {{
        out[expert_local * out_dims + row] = T(shared_sums[0] + shared_sums[1]);
    }}
"""


def _get_kernel(bits: int, group_size: int):
    """Get (or compile and cache) the Metal kernel for given parameters."""
    key = (bits, group_size)
    if key not in _kernel_cache:
        source = _build_kernel_source(bits, group_size)
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"polar_gather_qmv_{bits}bit_gs{group_size}",
            input_names=["packed_weight", "scales", "codebook", "x", "indices"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def polar_gather_qmv(
    packed_weight: mx.array,
    scales: mx.array,
    codebook: mx.array,
    x: mx.array,
    indices: mx.array,
    bits: int,
    group_size: int,
) -> mx.array:
    """Fused expert-routed quantized matrix-vector product via Metal kernel.

    Only reads the k selected expert weights from packed format — never
    materializes or dequantizes the full expert weight tensor.

    Uses threadgroup parallelism: 32 threads cooperate per (expert, row)
    with shared memory tree reduction.

    Args:
        packed_weight: (num_experts, output_dims, packed_cols) uint32.
        scales: (num_experts, output_dims, n_groups) float16.
        codebook: (n_codes,) float16 — Lloyd-Max centroids.
        x: (input_dims,) float16 — input vector (single token).
        indices: (k,) or (1, k) uint32 — selected expert indices.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Elements per quantization group.

    Returns:
        (k, output_dims) float16 — output for each selected expert.
    """
    if indices.ndim == 2:
        indices = indices.squeeze(0)

    indices = indices.astype(mx.uint32)
    k = indices.shape[0]
    output_dims = packed_weight.shape[1]

    kernel = _get_kernel(bits, group_size)

    total_work = k * output_dims
    # grid = total threads, so multiply by THREADS_PER_ROW to get
    # one threadgroup (of 32 threads) per (expert, row) pair
    outputs = kernel(
        inputs=[packed_weight, scales, codebook, x, indices],
        template=[("T", x.dtype)],
        grid=(total_work * THREADS_PER_ROW, 1, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        output_shapes=[(k, output_dims)],
        output_dtypes=[x.dtype],
    )

    return outputs[0]
