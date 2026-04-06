"""Fused Metal kernel for multi-input expert-routed matrix-vector multiply.

Like polar_gather_qmv but each expert reads from its OWN input vector
instead of sharing a single input. Used for MoE down_proj where each
expert's activation is different after gate/up + SwiGLU.

x layout: (k, input_dims) — k separate input vectors, one per expert.
"""

import math

import mlx.core as mx

_kernel_cache: dict[tuple[int, int], object] = {}

THREADS_PER_ROW = 32


def _build_kernel_source(bits: int, group_size: int) -> str:
    """Generate Metal shader for multi-input expert-routed polar QMV."""
    n_codes = 1 << bits
    elems_per_u32 = 32 // bits
    mask = (1 << bits) - 1

    return f"""
    uint lane = thread_position_in_threadgroup.x;
    uint work_id = threadgroup_position_in_grid.x;

    uint out_dims = packed_weight_shape[1];
    uint k = indices_shape[0];
    uint in_dims = x_shape[1];
    uint total_work = k * out_dims;
    if (work_id >= total_work) return;

    uint expert_local = work_id / out_dims;
    uint row = work_id % out_dims;
    uint expert_id = indices[expert_local];

    threadgroup float shared_sums[{THREADS_PER_ROW}];

    float cb[{n_codes}];
    for (uint i = 0; i < {n_codes}u; i++) {{
        cb[i] = float(codebook[i]);
    }}

    uint n_groups = scales_shape[2];
    uint pw_cols = packed_weight_shape[2];

    uint pw_base = expert_id * out_dims * pw_cols + row * pw_cols;
    uint sc_base = expert_id * out_dims * n_groups + row * n_groups;
    uint x_base = expert_local * in_dims;

    float accum = 0.0f;

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

            group_accum += cb[code_idx] * float(x[x_base + col]);
        }}

        accum += group_accum * scale;
    }}

    shared_sums[lane] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    key = (bits, group_size)
    if key not in _kernel_cache:
        source = _build_kernel_source(bits, group_size)
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"polar_multi_gather_qmv_{bits}bit_gs{group_size}",
            input_names=["packed_weight", "scales", "codebook", "x", "indices"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def polar_multi_gather_qmv(
    packed_weight: mx.array,
    scales: mx.array,
    codebook: mx.array,
    x: mx.array,
    indices: mx.array,
    bits: int,
    group_size: int,
) -> mx.array:
    """Multi-input expert-routed quantized matrix-vector product.

    Like polar_gather_qmv but each expert reads from its own input vector.
    Used for MoE down_proj where each expert's activation is different.

    Args:
        packed_weight: (num_experts, output_dims, packed_cols) uint32.
        scales: (num_experts, output_dims, n_groups) float16.
        codebook: (n_codes,) float16.
        x: (k, input_dims) — one input vector per selected expert.
        indices: (k,) uint32 — selected expert indices.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Elements per quantization group.

    Returns:
        (k, output_dims) — output for each selected expert.
    """
    indices = indices.astype(mx.uint32)
    k = indices.shape[0]
    output_dims = packed_weight.shape[1]

    kernel = _get_kernel(bits, group_size)

    total_work = k * output_dims
    outputs = kernel(
        inputs=[packed_weight, scales, codebook, x, indices],
        template=[("T", x.dtype)],
        grid=(total_work * THREADS_PER_ROW, 1, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        output_shapes=[(k, output_dims)],
        output_dtypes=[x.dtype],
    )

    return outputs[0]
