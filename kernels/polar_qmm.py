"""polar_qmm: batched dense GEMM on packed TurboQuant weights.

Y[n, o] = sum_k x[n, k] * W[o, k]

The dense counterpart to :mod:`polar_gather_qmm`, which does the same thing for
MoE experts. Without this, ``PolarQuantizedLinear`` had a fused kernel only for
the single-vector decode path (:mod:`polar_qmv`) and fell back to
``polar_dequantize_weight`` + ``x @ w.T`` for ANY batch > 1 -- that is, for all
of prefill. That fallback materializes the weight through several full-size
intermediates, measured at ~14 bytes per parameter, which is both a large
transient memory spike and wasted bandwidth. Peak memory is what makes it
fatal: it is a FIXED cost that appears the instant n_tokens >= 2 and is
independent of prompt length.

Structure follows polar_gather_qmm: one threadgroup per (16-token tile,
64-row output block); 256 threads = 64 output rows x 4 word-partitions. Each
thread walks its output row's packed words with stride 4 (coalesced across
partitions), unpacks the codes in each word, and FMAs into 16 statically
indexed per-token accumulators held in registers, against an x tile staged once
in threadgroup memory. The packed weight is read once per tile and decoded in
registers; the dequantized matrix is never written to memory.
"""

import math

import mlx.core as mx

TT = 16    # tokens per tile
OB = 64    # output rows per block
NT = 256   # threads per threadgroup
WC = 32    # packed words per K-chunk

_kernel_cache: dict[tuple, object] = {}


def _build_source(bits: int, group_size: int, trit: bool = False) -> str:
    if trit:
        n_codes = 3
        epu = 20              # trits per packed word
        pow3_init = ", ".join(f"{3 ** i}u" for i in range(20))
        pow3_decl = f"    const uint pw3[20] = {{{pow3_init}}};\n"
        code_expr = "(word / pw3[j]) % 3u"   # base-3 digit at slot j
    else:
        n_codes = 1 << bits
        epu = 32 // bits          # codes per packed word
        mask = (1 << bits) - 1
        pow3_decl = ""
        code_expr = f"(word >> (j * {bits}u)) & {mask}u"
    kc = WC * epu             # cols per chunk

    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint oblk = threadgroup_position_in_grid.y;

    uint N = x_shape[0];
    uint K = x_shape[1];
    uint O = packed_weight_shape[0];
    uint pw_cols = packed_weight_shape[1];
    uint n_groups = scales_shape[1];

    uint t0 = tile * {TT}u;
    if (t0 >= N) return;                           // tail tile beyond the batch

    uint o = oblk * {OB}u + tid / 4u;
    uint wpart = tid % 4u;

    threadgroup half xs[{kc}][{TT}];
    threadgroup float red[{NT}];

    float cb[{n_codes}];
    #pragma unroll
    for (uint i = 0; i < {n_codes}u; i++) cb[i] = float(codebook[i]);
{pow3_decl}
    float acc[{TT}];
    #pragma unroll
    for (uint t = 0; t < {TT}u; t++) acc[t] = 0.0f;

    // Clamp the row used for ADDRESS computation: when O is not a multiple of
    // {OB}, threads with o >= O must keep participating in the barriers below,
    // so they compute on row O-1 and the o < O write guard discards their
    // result. Without the clamp they would read out of bounds.
    uint safe_o = min(o, O - 1u);
    uint pw_base = safe_o * pw_cols;
    uint sc_base = safe_o * n_groups;
    uint n_chunks = (K + {kc}u - 1u) / {kc}u;

    for (uint c = 0u; c < n_chunks; c++) {{
        uint k0 = c * {kc}u;
        uint cols = min((uint){kc}, K - k0);

        // Cooperative stage of the x chunk: xs[kk][t]. Clamp the address
        // operands so tail tokens and tail columns can never form an
        // out-of-bounds address even under predicated execution.
        for (uint i = tid; i < {kc}u * {TT}u; i += {NT}u) {{
            uint kk = i / {TT}u;
            uint tt = i % {TT}u;
            uint tok = t0 + tt;
            uint safe_tok = tok < N ? tok : 0u;
            uint safe_kk = kk < cols ? kk : 0u;
            xs[kk][tt] = (kk < cols && tok < N)
                ? x[safe_tok * K + k0 + safe_kk] : half(0.0f);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint w0 = k0 / {epu}u;                     // chunk's first word
        uint n_words = (cols + {epu}u - 1u) / {epu}u;
        for (uint wi = wpart; wi < n_words; wi += 4u) {{
            uint word = packed_weight[pw_base + w0 + wi];
            uint col0 = wi * {epu}u;               // within chunk
            #pragma unroll
            for (uint j = 0; j < {epu}u; j++) {{
                uint col = col0 + j;
                if (col >= cols) break;
                uint code = {code_expr};
                float w = cb[code]
                    * float(scales[sc_base + (k0 + col) / {group_size}u]);
                #pragma unroll
                for (uint t = 0; t < {TT}u; t++) {{
                    acc[t] = fma(w, float(xs[col][t]), acc[t]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // reduce the 4 word-partitions per output row, write Y[tok, o]
    #pragma unroll
    for (uint t = 0u; t < {TT}u; t++) {{
        red[tid] = acc[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wpart == 0u && o < O && (t0 + t) < N) {{
            float v = red[tid] + red[tid + 1u] + red[tid + 2u] + red[tid + 3u];
            out[(t0 + t) * O + o] = T(v);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


def _get_kernel(bits: int, group_size: int, trit: bool = False):
    key = (bits, group_size, trit)
    if key not in _kernel_cache:
        name = (
            f"polar_qmm_trit_gs{group_size}"
            if trit
            else f"polar_qmm_{bits}b_gs{group_size}"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=name,
            input_names=["packed_weight", "scales", "codebook", "x"],
            output_names=["out"],
            source=_build_source(bits, group_size, trit),
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def polar_qmm(packed_weight, scales, codebook, x, bits, group_size, trit=False):
    """Fused batched quantized matmul: (N, K) @ dequant(W).T -> (N, O).

    Args:
        packed_weight: (O, pw_cols) uint32 — packed b-bit indices (or trits).
        scales: (O, n_groups) float16 — per-group RMS scales.
        codebook: (n_codes,) float16 — Lloyd-Max centroids (3 entries if trit).
        x: (N, K) float16 — input, N >= 1.
        bits: Quantization bit-width (2, 3, or 4). Ignored when trit=True.
        group_size: Elements per quantization group.
        trit: If True, decode base-3 packing (20 trits/uint32).

    Returns:
        (N, O) — same dtype as x.
    """
    N = int(x.shape[0])
    O = int(packed_weight.shape[0])
    kernel = _get_kernel(bits, group_size, trit)
    return kernel(
        inputs=[packed_weight, scales, codebook, x],
        template=[("T", x.dtype)],
        grid=(math.ceil(N / TT) * NT, math.ceil(O / OB), 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(N, O)],
        output_dtypes=[x.dtype],
    )[0]
