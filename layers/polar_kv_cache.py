"""TurboQuant KV Cache: Hadamard rotation + Lloyd-Max codebook compression.

Adapts the original TurboQuant KV cache compression (Zandieh et al., 2025)
for Apple Silicon using MLX.

Architecture:
  - Storage: TurboQuant compressed (rotation + codebook), independent
    per-K and per-V bit widths
  - Attention: Dequantize to float16 → standard mx.fast.scaled_dot_product_attention
    This preserves compatibility with all model features (attention sinks,
    sliding windows, etc.) and lets MLX's Metal kernel handle precision.

Memory savings come from the persistent compressed storage. The temporary
float16 dequantized arrays for attention are part of the lazy eval graph
and don't persist between steps.

Mixed precision (v0.2):
  - K and V can carry independent bit widths via (k_bits, v_bits)
  - K is more sensitive to score perturbation (softmax amplifies error);
    keep K8 unless calibration shows otherwise
  - V tolerates 3-bit safely on most architectures (D >= 128)
  - Recommended default: K8 + V3 (~2.7x compression, near-lossless)

Compatible with models that use attention sinks (GPT-OSS), hybrid
attention (Qwen3.5 linear + full), and sliding windows.
"""

import math

import mlx.core as mx

from turboquant_mlx.core.codebook import (
    get_codebook,
    quantize_scalar,
    dequantize_scalar,
)
from turboquant_mlx.core.rotation import (
    generate_random_signs,
    _find_hadamard_block_size,
)
from turboquant_mlx.core.packing import pack_indices, unpack_indices


class TurboQuantKVCache:
    """KV Cache with TurboQuant compression for storage, float16 for attention.

    Stores keys/values in TurboQuant format (rotation + codebook, no bias)
    for maximum compression. For attention computation, dequantizes to
    float16 for use with mx.fast.scaled_dot_product_attention.

    Does NOT expose ``self.bits`` — routes through standard SDPA, which
    supports attention sinks (GPT-OSS) and all other attention features.

    Parameters
    ----------
    tq_bits : int, optional
        Legacy: applies the same bit width to both K and V. Mutually
        exclusive with ``k_bits`` / ``v_bits``.
    k_bits, v_bits : int, optional
        Independent bit widths for keys and values. Both must be set
        together. Mutually exclusive with ``tq_bits``.
    group_size : int
        Per-group RMS scaling group size (default 64).
    seed : int
        Rotation seed.
    min_tokens_before_quant : int
        Number of leading tokens kept in fp16 (Tier A) before
        TQ-compressed storage (Tier B) takes over. Protects attention
        sinks at the cost of a fixed fp16 buffer. 0 disables the tier
        and is byte-equivalent to v0.1.x. Recommended: 1024.

    Storage compression vs float16 (head_dim=128, group_size=64,
    accounting for fp16 group RMS scales):
      tq_bits=3       ~4.6x   uniform 3-bit
      tq_bits=4       ~3.8x   uniform 4-bit
      tq_bits=2       ~7.1x   uniform 2-bit (quality-degrading)
      k_bits=8 v=4    ~2.6x   safe across architectures
      k_bits=8 v=3    ~2.7x   recommended default
      k_bits=4 v=3    ~4.0x   K starts to degrade greedy decode
    """

    step = 256

    def __init__(
        self,
        tq_bits: int | None = None,
        group_size: int = 64,
        seed: int = 42,
        *,
        k_bits: int | None = None,
        v_bits: int | None = None,
        min_tokens_before_quant: int = 0,
    ):
        if tq_bits is not None and (k_bits is not None or v_bits is not None):
            raise ValueError(
                "Pass either tq_bits (legacy) OR (k_bits, v_bits) — not both."
            )
        if (k_bits is None) ^ (v_bits is None):
            raise ValueError("k_bits and v_bits must be set together.")
        if tq_bits is None and k_bits is None:
            tq_bits = 3  # historical default
        if tq_bits is not None:
            k_bits = tq_bits
            v_bits = tq_bits
        if min_tokens_before_quant < 0:
            raise ValueError("min_tokens_before_quant must be >= 0")

        # Storage state — Tier B (TQ-compressed)
        self._tq_keys = None
        self._tq_values = None
        # Storage state — Tier A (fp16, attention-sink window)
        self._fp16_keys = None
        self._fp16_values = None
        self._min_tokens_before_quant = int(min_tokens_before_quant)
        self.offset = 0
        self._k_bits = int(k_bits)
        self._v_bits = int(v_bits)
        self._group_size = int(group_size)
        self._seed = int(seed)
        # Opt-in fused decode+attend Metal kernel (reads packed KV directly,
        # skips the fp16 dequantize). Set via enable_fused_attend().
        self._use_fused_attend = False

        # Precompute K codebook
        self._k_codebook_f32, self._k_boundaries_f32 = get_codebook(
            self._k_bits, dtype=mx.float32
        )
        self._k_codebook_f16 = self._k_codebook_f32.astype(mx.float16)
        self._k_max_centroid = float(mx.max(mx.abs(self._k_codebook_f32)).item())

        # Precompute V codebook
        self._v_codebook_f32, self._v_boundaries_f32 = get_codebook(
            self._v_bits, dtype=mx.float32
        )
        self._v_codebook_f16 = self._v_codebook_f32.astype(mx.float16)
        self._v_max_centroid = float(mx.max(mx.abs(self._v_codebook_f32)).item())

        # Rotation state (lazy-initialized on first update)
        self._k_signs = None
        self._v_signs = None
        self._k_head_dim = None
        self._v_head_dim = None
        self._k_block_size = None
        self._v_block_size = None
        self._k_gs = None
        self._v_gs = None

    # Back-compat alias: some external code may read ``cache._bits``.
    @property
    def _bits(self):
        if self._k_bits == self._v_bits:
            return self._k_bits
        # Mixed precision — return K bits as the "primary" width since K is
        # the precision-critical lane. Caller should prefer _k_bits/_v_bits.
        return self._k_bits

    def _ensure_rotation(self, k_head_dim, v_head_dim):
        if self._k_signs is None or self._k_head_dim != k_head_dim:
            self._k_head_dim = k_head_dim
            self._k_signs = generate_random_signs(k_head_dim, seed=self._seed)
            self._k_block_size = _find_hadamard_block_size(k_head_dim)
            self._k_gs = min(self._group_size, k_head_dim)
            while k_head_dim % self._k_gs != 0 and self._k_gs > 1:
                self._k_gs //= 2

        if self._v_signs is None or self._v_head_dim != v_head_dim:
            self._v_head_dim = v_head_dim
            self._v_signs = generate_random_signs(v_head_dim, seed=self._seed + 1)
            self._v_block_size = _find_hadamard_block_size(v_head_dim)
            self._v_gs = min(self._group_size, v_head_dim)
            while v_head_dim % self._v_gs != 0 and self._v_gs > 1:
                self._v_gs //= 2

    def _hadamard_fwd(self, x, block_size):
        dim = x.shape[-1]
        if block_size == dim:
            return mx.hadamard_transform(x, scale=1.0 / math.sqrt(dim))
        n_blocks = dim // block_size
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], n_blocks, block_size)
        x = mx.hadamard_transform(x, scale=1.0 / math.sqrt(block_size))
        return x.reshape(orig_shape)

    def _rotate(self, x, signs, block_size):
        return self._hadamard_fwd(x * signs, block_size)

    def _unrotate(self, x, signs, block_size):
        return self._hadamard_fwd(x, block_size) * signs

    def _tq_quantize(
        self, vectors, signs, block_size, gs, head_dim,
        bits, boundaries, max_centroid,
    ):
        """TurboQuant quantize: rotate → normalize → codebook → pack."""
        B, H, S, D = vectors.shape
        n_groups = D // gs

        v_rot = self._rotate(
            vectors.astype(mx.float32), signs.astype(mx.float32), block_size
        )
        v_grouped = v_rot.reshape(B, H, S, n_groups, gs)
        rms = mx.sqrt(mx.mean(v_grouped * v_grouped, axis=-1, keepdims=True))
        rms = mx.maximum(rms, mx.array(1e-7))
        v_norm = v_grouped / rms
        v_norm = mx.clip(v_norm, -max_centroid * 1.5, max_centroid * 1.5)

        indices = quantize_scalar(v_norm, boundaries)
        indices_flat = indices.reshape(B, H, S, D)
        packed = pack_indices(indices_flat, bits)
        scales = rms.squeeze(-1).astype(mx.float16)
        return packed, scales

    def _tq_dequantize(
        self, packed, scales, signs, block_size, gs, head_dim,
        bits, codebook_f16,
    ):
        """TurboQuant dequantize: unpack → codebook → scale → unrotate."""
        B, H, S = packed.shape[:3]
        n_groups = head_dim // gs

        indices = unpack_indices(packed, bits, head_dim)
        indices = indices.reshape(B, H, S, head_dim)
        v_deq = dequantize_scalar(indices, codebook_f16)
        v_deq = v_deq.reshape(B, H, S, n_groups, gs)
        v_deq = v_deq * mx.expand_dims(scales, axis=-1)
        v_deq = v_deq.reshape(B, H, S, head_dim)
        v_deq = self._unrotate(
            v_deq.astype(mx.float32), signs.astype(mx.float32), block_size
        )
        return v_deq.astype(mx.float16)

    def _fused_applicable(self, B, num_steps):
        """True when the fused decode+attend kernel can serve this step.

        Requires: opt-in flag; a single new token (decode); batch 1; single tier
        (no fp16 attention-sink window); equal K/V head dims that are a multiple
        of 32; and bit-packed (2^bits) codebooks — never trit for KV.
        """
        return (
            self._use_fused_attend
            and num_steps == 1
            and B == 1
            and self._min_tokens_before_quant == 0
            and self._k_head_dim is not None
            and self._k_head_dim == self._v_head_dim
            and self._k_head_dim % 32 == 0
        )

    def fused_attend(self, queries, scale):
        """Fused decode attention straight from packed storage.

        ``queries``: (1, n_q_heads, 1, D). Returns (1, n_q_heads, 1, D) fp16.
        Rotation stays outside the kernel via the orthonormal identity; the
        model's ``scale`` is folded into the pre-rotated query so the kernel's
        baked 1/sqrt(D) reproduces it.
        """
        from turboquant_mlx.kernels.kv_decode_attend import kv_decode_attend

        B, n_q, S_q, D = queries.shape
        b_total = self.offset  # threshold == 0 -> all tokens are in Tier B

        q_rot = self._rotate(
            queries.astype(mx.float32), self._k_signs.astype(mx.float32),
            self._k_block_size,
        ).astype(mx.float16).reshape(n_q, D)
        # fold the model scale into q so the kernel's baked 1/sqrt(D) == scale
        q_rot = q_rot * mx.array(scale * math.sqrt(D), dtype=mx.float16)

        out = kv_decode_attend(
            q_rot,
            mx.contiguous(self._tq_keys[0][0, :, :b_total, :]),
            mx.contiguous(self._tq_keys[1][0, :, :b_total, :]),
            self._k_codebook_f16,
            mx.contiguous(self._tq_values[0][0, :, :b_total, :]),
            mx.contiguous(self._tq_values[1][0, :, :b_total, :]),
            self._v_codebook_f16,
            self._k_bits, self._v_bits, self._k_gs, self._v_gs,
        ).reshape(1, n_q, 1, D)
        out = self._unrotate(
            out.astype(mx.float32), self._v_signs.astype(mx.float32),
            self._v_block_size,
        )
        return out.astype(mx.float16)

    def update_and_fetch(self, keys, values):
        """Store new KV across two tiers, return float16 for SDPA.

        Tier A — fp16 buffer for tokens [0, min_tokens_before_quant). Holds
        attention sinks at full precision so they don't pay quantization
        error.

        Tier B — TQ-compressed for tokens [min_tokens_before_quant, offset).

        When ``min_tokens_before_quant=0`` (default), Tier A is empty and
        all tokens flow through Tier B exactly like v0.1.x.

        Returns float16 (keys, values) compatible with standard
        mx.fast.scaled_dot_product_attention including attention sinks.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset
        new_offset = prev + num_steps
        threshold = self._min_tokens_before_quant

        self._ensure_rotation(k_head_dim, v_head_dim)

        # Lazy-init Tier A buffer (size fixed by threshold)
        if threshold > 0 and self._fp16_keys is None:
            self._fp16_keys = mx.zeros(
                (B, n_kv_heads, threshold, k_head_dim), dtype=mx.float16
            )
            self._fp16_values = mx.zeros(
                (B, n_kv_heads, threshold, v_head_dim), dtype=mx.float16
            )

        # Split this batch into A-bound and B-bound segments
        a_end = min(new_offset, threshold)
        a_count = max(0, a_end - prev)
        b_count = num_steps - a_count

        # Tier A append (fp16, no quantization)
        if a_count > 0:
            self._fp16_keys[..., prev : prev + a_count, :] = (
                keys[..., :a_count, :].astype(mx.float16)
            )
            self._fp16_values[..., prev : prev + a_count, :] = (
                values[..., :a_count, :].astype(mx.float16)
            )

        # Tier B append (TQ-compressed). b_prev counts from the start of
        # Tier B (absolute position threshold maps to b-position 0).
        if b_count > 0:
            b_keys = keys[..., a_count:, :]
            b_values = values[..., a_count:, :]
            b_prev = max(0, prev - threshold)
            b_new = b_prev + b_count

            k_gs = self._k_gs
            v_gs = self._v_gs
            k_el_per_int = 32 // self._k_bits
            v_el_per_int = 32 // self._v_bits
            packed_k_dim = (k_head_dim + k_el_per_int - 1) // k_el_per_int
            packed_v_dim = (v_head_dim + v_el_per_int - 1) // v_el_per_int
            n_groups_k = k_head_dim // k_gs
            n_groups_v = v_head_dim // v_gs

            # Allocate or expand TQ storage
            if (
                self._tq_keys is None
                or b_new > self._tq_keys[0].shape[-2]
            ):
                new_steps = (self.step + b_count - 1) // self.step * self.step
                # Round up to fit b_new
                while b_prev + new_steps < b_new:
                    new_steps += self.step
                shape = (B, n_kv_heads, new_steps)

                def _init(pd, ng):
                    return (
                        mx.zeros((*shape, pd), dtype=mx.uint32),
                        mx.zeros((*shape, ng), dtype=mx.float16),
                    )

                def _expand(x):
                    pad = mx.zeros(
                        (B, n_kv_heads, new_steps, x.shape[-1]), dtype=x.dtype
                    )
                    return mx.concatenate([x, pad], axis=-2)

                if self._tq_keys is not None:
                    if b_prev % self.step != 0:
                        self._tq_keys = tuple(
                            x[..., :b_prev, :] for x in self._tq_keys
                        )
                        self._tq_values = tuple(
                            x[..., :b_prev, :] for x in self._tq_values
                        )
                    self._tq_keys = tuple(_expand(x) for x in self._tq_keys)
                    self._tq_values = tuple(_expand(x) for x in self._tq_values)
                else:
                    self._tq_keys = _init(packed_k_dim, n_groups_k)
                    self._tq_values = _init(packed_v_dim, n_groups_v)

            # Quantize and store the B-bound segment
            k_packed, k_scales = self._tq_quantize(
                b_keys, self._k_signs, self._k_block_size, k_gs, k_head_dim,
                self._k_bits, self._k_boundaries_f32, self._k_max_centroid,
            )
            v_packed, v_scales = self._tq_quantize(
                b_values, self._v_signs, self._v_block_size, v_gs, v_head_dim,
                self._v_bits, self._v_boundaries_f32, self._v_max_centroid,
            )
            self._tq_keys[0][..., b_prev:b_new, :] = k_packed
            self._tq_keys[1][..., b_prev:b_new, :] = k_scales
            self._tq_values[0][..., b_prev:b_new, :] = v_packed
            self._tq_values[1][..., b_prev:b_new, :] = v_scales

        self.offset = new_offset

        # Fused decode path: the token is stored; skip the fp16 dequantize and
        # signal the patched SDPA seam (keys is None) to run the fused kernel.
        if self._fused_applicable(B, num_steps):
            return None, None

        # Fetch: concat Tier A (fp16) + Tier B (dequantized)
        b_total = max(0, new_offset - threshold)
        if b_total > 0:
            k_deq_b = self._tq_dequantize(
                self._tq_keys[0][..., :b_total, :],
                self._tq_keys[1][..., :b_total, :],
                self._k_signs, self._k_block_size, self._k_gs, k_head_dim,
                self._k_bits, self._k_codebook_f16,
            )
            v_deq_b = self._tq_dequantize(
                self._tq_values[0][..., :b_total, :],
                self._tq_values[1][..., :b_total, :],
                self._v_signs, self._v_block_size, self._v_gs, v_head_dim,
                self._v_bits, self._v_codebook_f16,
            )
        else:
            k_deq_b = None
            v_deq_b = None

        if a_end > 0:
            a_keys_view = self._fp16_keys[..., :a_end, :]
            a_values_view = self._fp16_values[..., :a_end, :]
            if k_deq_b is not None:
                return (
                    mx.concatenate([a_keys_view, k_deq_b], axis=-2),
                    mx.concatenate([a_values_view, v_deq_b], axis=-2),
                )
            return a_keys_view, a_values_view
        return k_deq_b, v_deq_b

    @property
    def state(self):
        # Returns (tq_keys, tq_values, fp16_keys, fp16_values). Older
        # 2-tuple form is also accepted by the setter for backwards compat.
        threshold = self._min_tokens_before_quant
        if self._tq_keys is None and self._fp16_keys is None:
            return []

        b_total = max(0, self.offset - threshold)
        if self._tq_keys is None:
            tq_pair = (None, None)
        elif b_total == self._tq_keys[0].shape[2]:
            tq_pair = (self._tq_keys, self._tq_values)
        else:
            tq_pair = (
                tuple(x[..., :b_total, :] for x in self._tq_keys),
                tuple(x[..., :b_total, :] for x in self._tq_values),
            )

        a_end = min(self.offset, threshold)
        if self._fp16_keys is None or a_end == 0:
            fp_pair = (None, None)
        else:
            fp_pair = (
                self._fp16_keys[..., :a_end, :],
                self._fp16_values[..., :a_end, :],
            )

        return (tq_pair[0], tq_pair[1], fp_pair[0], fp_pair[1])

    @state.setter
    def state(self, v):
        if not v:
            return

        def _norm(x):
            # A deserialized state (mlx tree_unflatten) hands back lists in
            # place of tuples and may use zero-size arrays for absent tiers.
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return tuple(x)
            if isinstance(x, mx.array) and x.size == 0:
                return None
            return x

        # 2-tuple legacy form: (tq_keys, tq_values)
        if len(v) == 2:
            self._tq_keys = _norm(v[0])
            self._tq_values = _norm(v[1])
            return
        # 4-tuple form: (tq_keys, tq_values, fp16_keys, fp16_values)
        if len(v) == 4:
            tq_k, tq_v, fp_k, fp_v = v
            self._tq_keys = _norm(tq_k)
            self._tq_values = _norm(tq_v)
            self._fp16_keys = _norm(fp_k)
            self._fp16_values = _norm(fp_v)
            return
        raise ValueError(
            f"Unexpected state length {len(v)}; expected 2 or 4."
        )

    @property
    def meta_state(self):
        # Forward-compat 6-tuple:
        # (offset, k_bits, v_bits, group_size, seed, min_tokens_before_quant)
        return tuple(
            map(str, (
                self.offset, self._k_bits, self._v_bits,
                self._group_size, self._seed,
                self._min_tokens_before_quant,
            ))
        )

    @meta_state.setter
    def meta_state(self, v):
        # Accept legacy 4-tuple, item-#1 5-tuple, and item-#2 6-tuple forms.
        if len(v) == 4:
            self.offset = int(v[0])
            self._k_bits = int(v[1])
            self._v_bits = int(v[1])
            self._group_size = int(v[2])
            self._seed = int(v[3])
        elif len(v) == 5:
            self.offset = int(v[0])
            self._k_bits = int(v[1])
            self._v_bits = int(v[2])
            self._group_size = int(v[3])
            self._seed = int(v[4])
        elif len(v) == 6:
            self.offset = int(v[0])
            self._k_bits = int(v[1])
            self._v_bits = int(v[2])
            self._group_size = int(v[3])
            self._seed = int(v[4])
            self._min_tokens_before_quant = int(v[5])
        else:
            raise ValueError(
                f"Unexpected meta_state length {len(v)}; expected 4, 5, or 6."
            )

    @classmethod
    def from_state(cls, state, meta_state):
        """Recreate a cache from ``(state, meta_state)``.

        This is the ``mlx_lm.models.cache`` deserialization protocol, used by
        the turboquant-serve disk prompt cache. Constructor params come from
        ``meta_state`` (any of the 4/5/6-tuple forms the setter accepts).
        """
        v = list(meta_state)
        if len(v) == 4:
            k_bits = v_bits = int(v[1])
            group_size, seed, min_tok = int(v[2]), int(v[3]), 0
        elif len(v) in (5, 6):
            k_bits, v_bits = int(v[1]), int(v[2])
            group_size, seed = int(v[3]), int(v[4])
            min_tok = int(v[5]) if len(v) == 6 else 0
        else:
            raise ValueError(
                f"Unexpected meta_state length {len(v)}; expected 4, 5, or 6."
            )
        obj = cls(
            k_bits=k_bits,
            v_bits=v_bits,
            group_size=group_size,
            seed=seed,
            min_tokens_before_quant=min_tok,
        )
        obj.state = state
        obj.offset = int(v[0])
        # The serialized fp16 tier is trimmed to the tokens it holds, but the
        # update path slice-assigns into a buffer preallocated at the full
        # threshold width — re-pad it so later appends fit.
        if obj._fp16_keys is not None:
            held = obj._fp16_keys.shape[-2]
            if held < min_tok:
                pad = [(0, 0)] * (obj._fp16_keys.ndim - 2) + [
                    (0, min_tok - held), (0, 0)
                ]
                obj._fp16_keys = mx.pad(obj._fp16_keys, pad)
                obj._fp16_values = mx.pad(obj._fp16_values, pad)
        return obj

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def empty(self):
        return self._tq_keys is None and self._fp16_keys is None

    @property
    def nbytes(self):
        """Return total cache size in bytes (Tier A fp16 + Tier B compressed)."""
        threshold = self._min_tokens_before_quant
        total = 0

        # Tier A
        a_end = min(self.offset, threshold)
        if self._fp16_keys is not None and a_end > 0:
            total += self._fp16_keys[..., :a_end, :].nbytes
            total += self._fp16_values[..., :a_end, :].nbytes

        # Tier B
        b_total = max(0, self.offset - threshold)
        if self._tq_keys is not None and b_total > 0:
            for arr in self._tq_keys:
                total += arr[..., :b_total, :].nbytes
            for arr in self._tq_values:
                total += arr[..., :b_total, :].nbytes
        return total

    def make_mask(self, N, window_size=None, return_array=False):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array=return_array,
                                     window_size=window_size)


def install_fused_attend_patch():
    """Route the model's SDPA seam to the fused kernel on fused decode steps.

    mlx-lm models call ``scaled_dot_product_attention(q, k, v, cache=..., ...)``
    (imported into each model module). When a fused-enabled TurboQuantKVCache
    served a decode step it returned ``(None, None)`` from ``update_and_fetch``;
    here ``keys is None`` is the signal to run ``cache.fused_attend``. All other
    calls fall through to the original. Idempotent (the installed wrapper is
    tagged and re-detected, so it survives module reloads); call after load.
    """
    import sys
    import mlx_lm.models.base as base

    orig = base.scaled_dot_product_attention
    if getattr(orig, "_turboquant_fused", False):
        return  # already installed

    def patched(queries, keys, values, cache=None, scale=1.0, mask=None,
                sinks=None):
        if keys is None and isinstance(cache, TurboQuantKVCache):
            # update_and_fetch skipped the fp16 dequant before this call
            # revealed mask/sinks. The fused kernel does plain causal decode
            # (a single query attends to all stored tokens), so it cannot honor
            # attention sinks or a sliding-window array mask. Fail loud rather
            # than return silently wrong output — the user opted in explicitly.
            if sinks is not None or (mask is not None and not isinstance(mask, str)):
                raise RuntimeError(
                    "fused KV decode+attend is incompatible with attention "
                    "sinks or a non-causal/sliding-window mask; do not call "
                    "enable_fused_attend() for this model."
                )
            return cache.fused_attend(queries, scale)
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask,
                    sinks=sinks)

    patched._turboquant_fused = True
    base.scaled_dot_product_attention = patched
    for name, mod in list(sys.modules.items()):
        if (name.startswith("mlx_lm.models.")
                and getattr(mod, "scaled_dot_product_attention", None) is orig):
            mod.scaled_dot_product_attention = patched


def enable_fused_attend(caches):
    """Turn on the fused decode+attend path for every TurboQuantKVCache in the
    list (and install the SDPA patch). Other cache types are ignored."""
    install_fused_attend_patch()
    for c in caches:
        if isinstance(c, TurboQuantKVCache):
            c._use_fused_attend = True
    return caches


def make_turboquant_cache(
    model,
    tq_bits: int | None = None,
    group_size: int = 64,
    seed: int = 42,
    *,
    k_bits: int | None = None,
    v_bits: int | None = None,
    min_tokens_before_quant: int = 0,
):
    """Create TurboQuant KV caches for all layers.

    Pass ``tq_bits`` for symmetric K/V (legacy) or ``(k_bits, v_bits)`` for
    mixed precision. ``min_tokens_before_quant`` keeps the first N tokens
    in fp16 (recommended: 1024) to protect attention sinks. See
    ``TurboQuantKVCache`` for details.
    """
    num_layers = len(model.layers)
    return [
        TurboQuantKVCache(
            tq_bits=tq_bits, group_size=group_size, seed=seed,
            k_bits=k_bits, v_bits=v_bits,
            min_tokens_before_quant=min_tokens_before_quant,
        )
        for _ in range(num_layers)
    ]


def convert_cache_to_turboquant(
    prompt_cache,
    tq_bits: int | None = None,
    group_size: int = 64,
    seed: int = 42,
    *,
    k_bits: int | None = None,
    v_bits: int | None = None,
    min_tokens_before_quant: int = 0,
):
    """Convert KVCache entries in a prompt cache list to TurboQuantKVCache.

    Only converts standard KVCache instances. Other cache types
    (RotatingKVCache, ArraysCache, etc.) are left unchanged so this
    works with hybrid-attention models like GPT-OSS and Qwen3.5.
    """
    from mlx_lm.models.cache import KVCache

    new_cache = []
    for c in prompt_cache:
        if not isinstance(c, KVCache):
            # Leave RotatingKVCache, ArraysCache, etc. as-is
            new_cache.append(c)
            continue

        tq = TurboQuantKVCache(
            tq_bits=tq_bits, group_size=group_size, seed=seed,
            k_bits=k_bits, v_bits=v_bits,
            min_tokens_before_quant=min_tokens_before_quant,
        )
        if c.keys is not None and c.offset > 0:
            keys = c.keys[..., : c.offset, :]
            values = c.values[..., : c.offset, :]
            tq.update_and_fetch(keys, values)
        new_cache.append(tq)
    return new_cache
