"""Fused Metal kernel for multi-input expert-routed matrix-vector multiply.

Like polar_gather_qmv but each expert reads from its OWN input vector
instead of sharing a single input. Used for MoE down_proj where each
expert's activation is different after gate/up + SwiGLU.

x layout: (k, input_dims) — k separate input vectors, one per expert.
"""

import math

import mlx.core as mx

_kernel_cache: dict[tuple[int, int, bool], object] = {}

THREADS_PER_ROW = 32


def _build_kernel_source(bits: int, group_size: int, trit: bool = False) -> str:
    """Generate Metal shader for multi-input expert-routed polar QMV.

    trit=True decodes base-3 (ternary) packing: 20 trits per uint32, 3-entry
    codebook — see ``core.packing.pack_trits`` and ``polar_gather_qmv``.
    """
    if trit:
        n_codes = 3
        pow3_init = ", ".join(f"{3 ** i}u" for i in range(20))
        pow3_decl = f"    const uint pw3[20] = {{{pow3_init}}};\n"
        extract = """
            uint packed_col = col / 20u;
            uint slot = col % 20u;
            uint packed_val = packed_weight[pw_base + packed_col];
            uint code_idx = (packed_val / pw3[slot]) % 3u;"""
    else:
        n_codes = 1 << bits
        elems_per_u32 = 32 // bits
        mask = (1 << bits) - 1
        pow3_decl = ""
        extract = f"""
            uint packed_col = col / {elems_per_u32}u;
            uint bit_pos = (col % {elems_per_u32}u) * {bits}u;
            uint packed_val = packed_weight[pw_base + packed_col];
            uint code_idx = (packed_val >> bit_pos) & {mask}u;"""

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
{pow3_decl}
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
            uint col = base_col + e;{extract}
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


def _get_kernel(bits: int, group_size: int, trit: bool = False):
    key = (bits, group_size, trit)
    if key not in _kernel_cache:
        source = _build_kernel_source(bits, group_size, trit)
        name = (
            f"polar_multi_gather_qmv_trit_gs{group_size}"
            if trit
            else f"polar_multi_gather_qmv_{bits}bit_gs{group_size}"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=name,
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
    trit: bool = False,
) -> mx.array:
    """Multi-input expert-routed quantized matrix-vector product.

    Like polar_gather_qmv but each expert reads from its own input vector.
    Used for MoE down_proj where each expert's activation is different.

    Args:
        packed_weight: (num_experts, output_dims, packed_cols) uint32.
        scales: (num_experts, output_dims, n_groups) float16.
        codebook: (n_codes,) float16 — 3 entries if trit.
        x: (k, input_dims) — one input vector per selected expert.
        indices: (k,) uint32 — selected expert indices.
        bits: Quantization bit-width (2, 3, or 4); ignored when trit=True.
        group_size: Elements per quantization group.
        trit: If True, decode base-3 (ternary) packing — 20 trits/uint32.

    Returns:
        (k, output_dims) — output for each selected expert.
    """
    indices = indices.astype(mx.uint32)
    k = int(indices.shape[0])
    output_dims = int(packed_weight.shape[1])

    kernel = _get_kernel(bits, group_size, trit)

    # Cap k per kernel call. Empirically the kernel succeeds on small N
    # (e.g. 11k token×expert routings) but fails inside mlx for very large
    # multi-chunk prefills (issue #1). Splitting the call along k keeps
    # grid sizes modest and is safe — the kernel computes each row
    # independently. K_CHUNK is a conservative cap that keeps grid_x well
    # under any plausible Metal limit.
    K_CHUNK = 4096
    if k <= K_CHUNK:
        outputs = kernel(
            inputs=[packed_weight, scales, codebook, x, indices],
            template=[("T", x.dtype)],
            grid=(k * output_dims * THREADS_PER_ROW, 1, 1),
            threadgroup=(THREADS_PER_ROW, 1, 1),
            output_shapes=[(k, output_dims)],
            output_dtypes=[x.dtype],
        )
        return outputs[0]

    chunks = []
    for start in range(0, k, K_CHUNK):
        end = min(start + K_CHUNK, k)
        n = end - start
        x_chunk = x[start:end]
        idx_chunk = indices[start:end]
        out_chunk = kernel(
            inputs=[packed_weight, scales, codebook, x_chunk, idx_chunk],
            template=[("T", x.dtype)],
            grid=(n * output_dims * THREADS_PER_ROW, 1, 1),
            threadgroup=(THREADS_PER_ROW, 1, 1),
            output_shapes=[(n, output_dims)],
            output_dtypes=[x.dtype],
        )[0]
        chunks.append(out_chunk)
    return mx.concatenate(chunks, axis=0)
