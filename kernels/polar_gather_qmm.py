"""polar_gather_qmm: batched expert-routed GEMM on packed TQ weights.

Y[n, o] = sum_k x[n, k] * W[indices[n], o, k]   with indices SORTED ascending.

Host side (vectorized mx ops, no GPU->CPU sync): tokens are grouped into
16-token tiles that are SINGLE-EXPERT, padding each expert's segment up to a
multiple of 16. Tile capacity is the static worst case N + 16*num_experts, so
shapes stay concrete; empty tail tiles exit immediately.

Kernel: one threadgroup per (tile, 64-row output block). 256 threads =
64 output rows x 4 word-partitions. Each thread walks its row's packed words
(stride 4 -> coalesced across partitions), unpacks the codes per word, and
FMAs into 16 statically-indexed per-token accumulators (registers, fully
unrolled) against an x tile staged once in threadgroup memory.
"""

import math

import mlx.core as mx

TT = 16    # tokens per tile
OB = 64    # output rows per block
NT = 256   # threads per threadgroup
WC = 32    # packed words per K-chunk

_kernel_cache: dict[tuple, object] = {}


def _build_source(bits: int, group_size: int) -> str:
    n_codes = 1 << bits
    epu = 32 // bits          # codes per packed word
    mask = (1 << bits) - 1
    kc = WC * epu             # cols per chunk

    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint tile = threadgroup_position_in_grid.x;
    uint oblk = threadgroup_position_in_grid.y;

    uint K = x_shape[1];
    uint O = packed_weight_shape[1];
    uint pw_cols = packed_weight_shape[2];
    uint n_groups = scales_shape[2];

    uint t0 = tile * {TT}u;
    int first_tok = tile_token[t0];
    if (first_tok < 0) return;                     // empty padding tile
    uint e = indices[uint(first_tok)];

    uint o = oblk * {OB}u + tid / 4u;
    uint wpart = tid % 4u;

    threadgroup half xs[{kc}][{TT}];
    threadgroup float red[{NT}];

    int toks[{TT}];
    #pragma unroll
    for (uint t = 0; t < {TT}u; t++) toks[t] = tile_token[t0 + t];

    float cb[{n_codes}];
    #pragma unroll
    for (uint i = 0; i < {n_codes}u; i++) cb[i] = float(codebook[i]);

    float acc[{TT}];
    #pragma unroll
    for (uint t = 0; t < {TT}u; t++) acc[t] = 0.0f;

    // Clamp the row used for ADDRESS computation: when O is not a multiple
    // of {OB}, threads with o >= O must keep participating in the barriers
    // below, so they compute on row O-1 and the o < O write guard discards
    // their result. Without the clamp they would read out of bounds.
    uint safe_o = min(o, O - 1u);
    uint pw_base = (e * O + safe_o) * pw_cols;
    uint sc_base = (e * O + safe_o) * n_groups;
    uint n_chunks = (K + {kc}u - 1u) / {kc}u;

    for (uint c = 0u; c < n_chunks; c++) {{
        uint k0 = c * {kc}u;
        uint cols = min((uint){kc}, K - k0);

        // cooperative stage of the x chunk: xs[kk][t]. Clamp the address
        // operands so padding tokens (tok = -1) and tail columns can never
        // form an out-of-bounds address even under predicated execution.
        for (uint i = tid; i < {kc}u * {TT}u; i += {NT}u) {{
            uint kk = i / {TT}u;
            uint tt = i % {TT}u;
            int tok = toks[tt];
            uint safe_tok = tok >= 0 ? uint(tok) : 0u;
            uint safe_kk = kk < cols ? kk : 0u;
            xs[kk][tt] = (kk < cols && tok >= 0)
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
                uint code = (word >> (j * {bits}u)) & {mask}u;
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
        if (wpart == 0u && o < O && toks[t] >= 0) {{
            float v = red[tid] + red[tid + 1u] + red[tid + 2u] + red[tid + 3u];
            out[uint(toks[t]) * O + o] = T(v);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


def _get_kernel(bits: int, group_size: int):
    key = (bits, group_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"polar_gather_qmm_{bits}b_gs{group_size}",
            input_names=["packed_weight", "scales", "codebook", "x", "indices",
                         "tile_token"],
            output_names=["out"],
            source=_build_source(bits, group_size),
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def _build_tiles(indices: mx.array, num_experts: int) -> mx.array:
    """Map sorted routings to single-expert 16-token tiles (padded with -1).

    Pure mx ops, fixed output shape N_pad = N + TT*num_experts -> no sync.
    """
    n = indices.shape[0]
    counts = mx.zeros((num_experts,), dtype=mx.int32).at[indices].add(1)
    padded = ((counts + TT - 1) // TT) * TT
    raw_off = mx.cumsum(counts) - counts        # segment start, unpadded
    pad_off = mx.cumsum(padded) - padded        # segment start, padded
    ar = mx.arange(n, dtype=mx.int32)
    pos = pad_off[indices] + (ar - raw_off[indices])
    cap = n + TT * num_experts
    tile_token = mx.full((cap,), -1, dtype=mx.int32).at[pos].add(ar + 1)
    return tile_token


def supports(output_dims: int) -> bool:
    return output_dims % OB == 0


def polar_gather_qmm(packed_weight, scales, codebook, x, indices,
                     bits, group_size):
    """x: (N, K) f16, indices: (N,) u32 SORTED. Returns (N, O) f16."""
    N = int(x.shape[0])
    O = int(packed_weight.shape[1])
    E = int(packed_weight.shape[0])
    tile_token = _build_tiles(indices.astype(mx.int32), E)
    n_tiles = tile_token.shape[0] // TT
    kernel = _get_kernel(bits, group_size)
    return kernel(
        inputs=[packed_weight, scales, codebook, x,
                indices.astype(mx.uint32), tile_token],
        template=[("T", x.dtype)],
        grid=(n_tiles * NT, math.ceil(O / OB), 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(N, O)],
        output_dtypes=[x.dtype],
    )[0]
