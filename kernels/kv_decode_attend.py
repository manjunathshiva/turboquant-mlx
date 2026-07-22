"""Fused KV decode+attend Metal kernel for the TurboQuant KV cache.

Single-token decode (S_q=1) attention that reads the *packed* TurboQuant KV
cache directly — decoding keys/values on the fly inside a FlashAttention-style
online-softmax pass, never materializing the fp16 K/V tensors that the standard
``dequantize -> scaled_dot_product_attention`` path builds.

The inverse-Hadamard rotation is kept OUTSIDE the kernel via the orthonormal
identity: since H is orthonormal + symmetric,
    q . k_hat = H(q (x) signs_k) . k_ds        (rotate the query once)
    out = signs_v (x) H(sum_s p_s v_ds[s])      (rotate the output once)
so the kernel operates purely in the rotated domain (per-dim decode + dot +
online softmax) and the caller applies ``_rotate`` to q and ``_unrotate`` to the
output. This removes the per-key WHT that dominates the dequantize path.

Design (flash-decoding / split-K): grid = (n_q_heads * n_chunks) simdgroups of
32 lanes; each lane owns D/32 dims. Each (q_head, chunk) simdgroup emits a
partial (m, l, acc); a second Metal kernel merges the partials with the standard
online-softmax combine. Bit-packed K and V only (2/3/4/8-bit); ``D`` must be a
multiple of 32.

Scope: B=1, S_q=1, single-tier (no fp16 attention-sink window), no sliding-window
mask, no attention sinks. Callers must check applicability before dispatching.
"""

import math

import mlx.core as mx

_kernel_cache: dict[tuple, object] = {}

WARP = 32


def _extract(prefix: str, bits: int, var: str) -> str:
    elems = 32 // bits
    mask = (1 << bits) - 1
    return f"""
            uint {var}_pcol = d / {elems}u;
            uint {var}_bpos = (d % {elems}u) * {bits}u;
            uint {var}_word = {prefix}[{prefix}_base + {var}_pcol];
            uint {var}_idx  = ({var}_word >> {var}_bpos) & {mask}u;"""


def _build_partial_source(k_bits, v_bits, gs_k, gs_v, D, chunk):
    n_codes_k = 1 << k_bits
    n_codes_v = 1 << v_bits
    dpl = D // WARP
    scale = 1.0 / math.sqrt(D)
    ext_k = _extract("packed_k", k_bits, "k")
    ext_v = _extract("packed_v", v_bits, "v")
    return f"""
    uint lane = thread_position_in_threadgroup.x;         // 0..31
    uint work = threadgroup_position_in_grid.x;           // q_head*n_chunks+chunk
    uint n_q_heads = q_rot_shape[0];
    uint n_kv_heads = packed_k_shape[0];
    uint S_kv   = packed_k_shape[1];
    uint n_chunks = (S_kv + {chunk}u - 1u) / {chunk}u;
    if (work >= n_q_heads * n_chunks) return;

    uint q_head = work / n_chunks;
    uint chunk  = work % n_chunks;
    uint s0 = chunk * {chunk}u;
    uint s1 = min(s0 + {chunk}u, S_kv);

    uint pk_cols = packed_k_shape[2];
    uint pv_cols = packed_v_shape[2];
    uint ngk = scales_k_shape[2];
    uint ngv = scales_v_shape[2];
    const uint D = {D}u;
    const uint DPL = {dpl}u;

    uint kv_head = q_head / (n_q_heads / n_kv_heads);

    float cbk[{n_codes_k}];
    for (uint i = 0; i < {n_codes_k}u; i++) cbk[i] = float(codebook_k[i]);
    float cbv[{n_codes_v}];
    for (uint i = 0; i < {n_codes_v}u; i++) cbv[i] = float(codebook_v[i]);

    float qd[DPL];
    for (uint i = 0; i < DPL; i++) qd[i] = float(q_rot[q_head * D + lane + i * 32u]);

    float acc[DPL];
    for (uint i = 0; i < DPL; i++) acc[i] = 0.0f;
    float run_m = -INFINITY;
    float run_l = 0.0f;

    uint pk_hbase = kv_head * S_kv * pk_cols;
    uint sk_hbase = kv_head * S_kv * ngk;
    uint pv_hbase = kv_head * S_kv * pv_cols;
    uint sv_hbase = kv_head * S_kv * ngv;

    for (uint s = s0; s < s1; s++) {{
        uint packed_k_base = pk_hbase + s * pk_cols;
        uint sk_base = sk_hbase + s * ngk;
        float pdot = 0.0f;
        for (uint i = 0; i < DPL; i++) {{
            uint d = lane + i * 32u;{ext_k}
            float sc = float(scales_k[sk_base + d / {gs_k}u]);
            pdot += qd[i] * (cbk[k_idx] * sc);
        }}
        float score = simd_sum(pdot) * {scale}f;
        float new_m = max(run_m, score);
        float corr = exp(run_m - new_m);
        float p = exp(score - new_m);
        run_l = run_l * corr + p;
        run_m = new_m;

        uint packed_v_base = pv_hbase + s * pv_cols;
        uint sv_base = sv_hbase + s * ngv;
        for (uint i = 0; i < DPL; i++) {{
            uint d = lane + i * 32u;{ext_v}
            float sc = float(scales_v[sv_base + d / {gs_v}u]);
            acc[i] = acc[i] * corr + p * (cbv[v_idx] * sc);
        }}
    }}

    uint acc_base = (q_head * n_chunks + chunk) * D;
    for (uint i = 0; i < DPL; i++) {{
        uint d = lane + i * 32u;
        partial_acc[acc_base + d] = partial_acc_t(acc[i]);
    }}
    if (lane == 0) {{
        partial_m[q_head * n_chunks + chunk] = run_m;
        partial_l[q_head * n_chunks + chunk] = run_l;
    }}
"""


def _build_combine_source(D):
    dpl = D // WARP
    return f"""
    uint lane = thread_position_in_threadgroup.x;
    uint q_head = threadgroup_position_in_grid.x;
    uint n_q_heads = partial_m_shape[0];
    if (q_head >= n_q_heads) return;
    uint n_chunks = partial_m_shape[1];
    const uint D = {D}u;
    const uint DPL = {dpl}u;

    uint pm_base = q_head * n_chunks;
    uint pa_base = q_head * n_chunks * D;

    float m = -INFINITY;
    for (uint j = 0; j < n_chunks; j++) m = max(m, partial_m[pm_base + j]);

    float num[DPL];
    for (uint i = 0; i < DPL; i++) num[i] = 0.0f;
    float l = 0.0f;
    for (uint j = 0; j < n_chunks; j++) {{
        float w = exp(partial_m[pm_base + j] - m);
        l += w * partial_l[pm_base + j];
        uint pa = pa_base + j * D;
        for (uint i = 0; i < DPL; i++)
            num[i] += w * float(partial_acc[pa + lane + i * 32u]);
    }}
    float inv_l = 1.0f / l;
    for (uint i = 0; i < DPL; i++)
        out[q_head * D + lane + i * 32u] = out_t(num[i] * inv_l);
"""


def _get_partial_kernel(k_bits, v_bits, gs_k, gs_v, D, chunk):
    key = ("partial", k_bits, v_bits, gs_k, gs_v, D, chunk)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"kv_da_k{k_bits}v{v_bits}_gk{gs_k}gv{gs_v}_d{D}_c{chunk}",
            input_names=["q_rot", "packed_k", "scales_k", "codebook_k",
                         "packed_v", "scales_v", "codebook_v"],
            output_names=["partial_acc", "partial_m", "partial_l"],
            source=_build_partial_source(k_bits, v_bits, gs_k, gs_v, D, chunk),
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def _get_combine_kernel(D):
    key = ("combine", D)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"kv_da_combine_d{D}",
            input_names=["partial_acc", "partial_m", "partial_l"],
            output_names=["out"],
            source=_build_combine_source(D),
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def _adaptive_chunk(S_kv, target_chunks=32):
    raw = max(1.0, S_kv / target_chunks)
    p2 = 1 << int(round(math.log2(raw)))
    return min(max(p2, 256), 2048)


def kv_decode_attend(q_rot, packed_k, scales_k, codebook_k,
                     packed_v, scales_v, codebook_v,
                     k_bits, v_bits, gs_k, gs_v, chunk=None):
    """Fused decode+attend.

    q_rot: (n_q_heads, D) — query already rotated (H(q (x) signs_k)) and
        pre-scaled so the kernel's baked 1/sqrt(D) yields the model's scale.
    packed_k/scales_k/codebook_k: (n_kv_heads, S_kv, pk_cols) uint32 /
        (n_kv_heads, S_kv, n_groups) f16 / (2^k_bits,) f16.
    packed_v/... : likewise for V.
    Returns (n_q_heads, D) in the rotated-V domain (caller un-rotates).
    """
    n_q_heads, D = q_rot.shape
    S_kv = packed_k.shape[1]
    if chunk is None:
        chunk = _adaptive_chunk(S_kv)
    n_chunks = (S_kv + chunk - 1) // chunk

    partial = _get_partial_kernel(k_bits, v_bits, gs_k, gs_v, D, chunk)
    partial_acc, partial_m, partial_l = partial(
        inputs=[q_rot, packed_k, scales_k, codebook_k,
                packed_v, scales_v, codebook_v],
        template=[("partial_acc_t", q_rot.dtype)],
        grid=(n_q_heads * n_chunks * WARP, 1, 1),
        threadgroup=(WARP, 1, 1),
        output_shapes=[(n_q_heads, n_chunks, D),
                       (n_q_heads, n_chunks), (n_q_heads, n_chunks)],
        output_dtypes=[q_rot.dtype, mx.float32, mx.float32],
    )
    combine = _get_combine_kernel(D)
    out = combine(
        inputs=[partial_acc, partial_m, partial_l],
        template=[("out_t", q_rot.dtype)],
        grid=(n_q_heads * WARP, 1, 1),
        threadgroup=(WARP, 1, 1),
        output_shapes=[(n_q_heads, D)],
        output_dtypes=[q_rot.dtype],
    )
    return out[0]
