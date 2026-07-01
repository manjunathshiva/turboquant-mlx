"""Fused Metal kernel: dequantize all experts of a PolarQuantized switch layer.

Unpacks the codebook indices, applies the per-group scales, and writes fp16
expert weights in one pass — one thread per (expert, row, group). This is
pure-bandwidth work and replaces the multi-op Python unpack path (~11x faster
at DiffusionGemma expert shapes, bit-identical output).

Used by PolarQuantizedSwitchLinear._dequantize_all and by the large-batch
routing path (dequant + mx.gather_mm), where materializing fp16 weights once
per call beats per-row gather kernels that re-read activations per output row.
"""

import mlx.core as mx

_kernel_cache: dict[tuple[int, int, bool], object] = {}


def _get_kernel(bits: int, group_size: int, trit: bool = False):
    key = (bits, group_size, trit)
    if key in _kernel_cache:
        return _kernel_cache[key]

    if trit:
        n_codes = 3
        pow3_init = ", ".join(f"{3 ** i}u" for i in range(20))
        pow3_decl = f"    const uint pw3[20] = {{{pow3_init}}};\n"
        # base-3 digit at position (col % 20): (word / 3**slot) % 3
        decode = """
        uint word = packed_weight[pw_base + col / 20u];
        uint code = (word / pw3[col % 20u]) % 3u;"""
    else:
        n_codes = 1 << bits
        elems_per_u32 = 32 // bits
        mask = (1 << bits) - 1
        pow3_decl = ""
        decode = f"""
        uint word = packed_weight[pw_base + col / {elems_per_u32}u];
        uint code = (word >> ((col % {elems_per_u32}u) * {bits}u)) & {mask}u;"""

    source = f"""
    uint gid = thread_position_in_grid.x;
    uint n_groups = scales_shape[2];
    uint out_rows = scales_shape[1];
    uint pw_cols = packed_weight_shape[2];
    uint in_dims = n_groups * {group_size}u;
    uint total = scales_shape[0] * out_rows * n_groups;
    if (gid >= total) return;

    uint g = gid % n_groups;
    uint row = (gid / n_groups) % out_rows;
    uint e = gid / (n_groups * out_rows);

    float cb[{n_codes}];
    for (uint i = 0; i < {n_codes}u; i++) {{ cb[i] = float(codebook[i]); }}
{pow3_decl}
    float scale = float(scales[gid]);
    uint pw_base = (e * out_rows + row) * pw_cols;
    uint out_base = (e * out_rows + row) * in_dims + g * {group_size}u;

    for (uint t = 0; t < {group_size}u; t++) {{
        uint col = g * {group_size}u + t;{decode}
        out[out_base + t] = T(cb[code] * scale);
    }}
    """
    name = (
        f"polar_dequant_experts_trit_gs{group_size}"
        if trit
        else f"polar_dequant_experts_{bits}bit_gs{group_size}"
    )
    _kernel_cache[key] = mx.fast.metal_kernel(
        name=name,
        input_names=["packed_weight", "scales", "codebook"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )
    return _kernel_cache[key]


def polar_dequant_experts(
    packed_weight: mx.array,
    scales: mx.array,
    codebook: mx.array,
    bits: int,
    group_size: int,
    trit: bool = False,
) -> mx.array:
    """Dequantize packed expert weights to float.

    Args:
        packed_weight: (num_experts, output_dims, packed_cols) uint32.
        scales: (num_experts, output_dims, n_groups) float16.
        codebook: (2^bits,) float16 — 3 entries if trit.
        bits: Quantization bit-width (2, 3, or 4); ignored when trit=True.
        group_size: Elements per quantization group.
        trit: If True, decode base-3 (ternary) packing — 20 trits/uint32.

    Returns:
        (num_experts, output_dims, n_groups * group_size) in scales.dtype.
    """
    kernel = _get_kernel(bits, group_size, trit)
    num_experts, output_dims, n_groups = scales.shape
    input_dims = n_groups * group_size
    total = num_experts * output_dims * n_groups
    return kernel(
        inputs=[packed_weight, scales, codebook],
        template=[("T", scales.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(num_experts, output_dims, input_dims)],
        output_dtypes=[scales.dtype],
    )[0]
