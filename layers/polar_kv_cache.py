"""TurboQuant KV Cache: Hadamard rotation + Lloyd-Max codebook compression.

Adapts the original TurboQuant KV cache compression (Zandieh et al., 2025)
for Apple Silicon using MLX.

Architecture:
  - Storage: TurboQuant compressed (rotation + codebook), ~4.6x at 3-bit
  - Attention: Dequantize to float16 → standard mx.fast.scaled_dot_product_attention
    This preserves compatibility with all model features (attention sinks,
    sliding windows, etc.) and lets MLX's Metal kernel handle precision.

Memory savings come from the persistent compressed storage. The temporary
float16 dequantized arrays for attention are part of the lazy eval graph
and don't persist between steps.

Compatible with models that use attention sinks (GPT-OSS), hybrid
attention (Qwen3.5 linear + full), and sliding windows.
"""

import math

import mlx.core as mx
from mlx.utils import tree_map

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

    Storage compression vs float16:
      - 3-bit TQ: ~4.6x  (3 bits + scale only, no bias)
      - 4-bit TQ: ~3.8x
      - 2-bit TQ: ~7.1x
    """

    step = 256

    def __init__(
        self,
        tq_bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
    ):
        # TQ storage parameters
        self._tq_keys = None
        self._tq_values = None
        self.offset = 0
        self._bits = tq_bits
        self._group_size = group_size
        self._seed = seed

        # Precompute codebook
        self._codebook_f32, self._boundaries_f32 = get_codebook(tq_bits, dtype=mx.float32)
        self._codebook_f16 = self._codebook_f32.astype(mx.float16)
        self._max_centroid = float(mx.max(mx.abs(self._codebook_f32)).item())

        # Rotation state (lazy-initialized on first update)
        self._k_signs = None
        self._v_signs = None
        self._k_head_dim = None
        self._v_head_dim = None
        self._k_block_size = None
        self._v_block_size = None

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

    def _tq_quantize(self, vectors, signs, block_size, gs, head_dim):
        """TurboQuant quantize: rotate → normalize → codebook → pack."""
        B, H, S, D = vectors.shape
        n_groups = D // gs

        v_rot = self._rotate(vectors.astype(mx.float32), signs.astype(mx.float32), block_size)
        v_grouped = v_rot.reshape(B, H, S, n_groups, gs)
        rms = mx.sqrt(mx.mean(v_grouped * v_grouped, axis=-1, keepdims=True))
        rms = mx.maximum(rms, mx.array(1e-7))
        v_norm = v_grouped / rms
        v_norm = mx.clip(v_norm, -self._max_centroid * 1.5, self._max_centroid * 1.5)

        indices = quantize_scalar(v_norm, self._boundaries_f32)
        indices_flat = indices.reshape(B, H, S, D)
        packed = pack_indices(indices_flat, self._bits)
        scales = rms.squeeze(-1).astype(mx.float16)
        return packed, scales

    def _tq_dequantize(self, packed, scales, signs, block_size, gs, head_dim):
        """TurboQuant dequantize: unpack → codebook → scale → unrotate."""
        B, H, S = packed.shape[:3]
        n_groups = head_dim // gs

        indices = unpack_indices(packed, self._bits, head_dim)
        indices = indices.reshape(B, H, S, head_dim)
        v_deq = dequantize_scalar(indices, self._codebook_f16)
        v_deq = v_deq.reshape(B, H, S, n_groups, gs)
        v_deq = v_deq * mx.expand_dims(scales, axis=-1)
        v_deq = v_deq.reshape(B, H, S, head_dim)
        v_deq = self._unrotate(v_deq.astype(mx.float32), signs.astype(mx.float32), block_size)
        return v_deq.astype(mx.float16)

    def update_and_fetch(self, keys, values):
        """Store new KV in TQ compressed format, return float16 for SDPA.

        Returns float16 arrays (keys, values) compatible with standard
        mx.fast.scaled_dot_product_attention including attention sinks.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        self._ensure_rotation(k_head_dim, v_head_dim)

        k_gs = self._k_gs
        v_gs = self._v_gs
        el_per_int = 32 // self._bits
        packed_k_dim = (k_head_dim + el_per_int - 1) // el_per_int
        packed_v_dim = (v_head_dim + el_per_int - 1) // el_per_int
        n_groups_k = k_head_dim // k_gs
        n_groups_v = v_head_dim // v_gs

        # Allocate or expand TQ storage
        if self._tq_keys is None or (prev + num_steps) > self._tq_keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def _init(pd, ng):
                return (
                    mx.zeros((*shape, pd), dtype=mx.uint32),
                    mx.zeros((*shape, ng), dtype=mx.float16),
                )

            def _expand(x):
                pad = mx.zeros((B, n_kv_heads, new_steps, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, pad], axis=-2)

            if self._tq_keys is not None:
                if prev % self.step != 0:
                    self._tq_keys = tuple(x[..., :prev, :] for x in self._tq_keys)
                    self._tq_values = tuple(x[..., :prev, :] for x in self._tq_values)
                self._tq_keys = tuple(_expand(x) for x in self._tq_keys)
                self._tq_values = tuple(_expand(x) for x in self._tq_values)
            else:
                self._tq_keys = _init(packed_k_dim, n_groups_k)
                self._tq_values = _init(packed_v_dim, n_groups_v)

        self.offset += num_steps

        # TQ quantize incoming tokens
        k_packed, k_scales = self._tq_quantize(
            keys, self._k_signs, self._k_block_size, k_gs, k_head_dim
        )
        v_packed, v_scales = self._tq_quantize(
            values, self._v_signs, self._v_block_size, v_gs, v_head_dim
        )

        # Store in TQ format
        self._tq_keys[0][..., prev : self.offset, :] = k_packed
        self._tq_keys[1][..., prev : self.offset, :] = k_scales
        self._tq_values[0][..., prev : self.offset, :] = v_packed
        self._tq_values[1][..., prev : self.offset, :] = v_scales

        # Dequantize full TQ cache → float16 for attention
        k_deq = self._tq_dequantize(
            self._tq_keys[0][..., : self.offset, :],
            self._tq_keys[1][..., : self.offset, :],
            self._k_signs, self._k_block_size, k_gs, k_head_dim,
        )
        v_deq = self._tq_dequantize(
            self._tq_values[0][..., : self.offset, :],
            self._tq_values[1][..., : self.offset, :],
            self._v_signs, self._v_block_size, v_gs, v_head_dim,
        )

        return k_deq, v_deq

    @property
    def state(self):
        if self._tq_keys is None:
            return []
        if self.offset == self._tq_keys[0].shape[2]:
            return self._tq_keys, self._tq_values
        return (
            tuple(x[..., : self.offset, :] for x in self._tq_keys),
            tuple(x[..., : self.offset, :] for x in self._tq_values),
        )

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._tq_keys, self._tq_values = v

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.offset, self._bits, self._group_size, self._seed))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self._bits = int(v[1])
        self._group_size = int(v[2])
        self._seed = int(v[3])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def empty(self):
        return self._tq_keys is None

    @property
    def nbytes(self):
        """Return TQ compressed cache size in bytes (persistent storage)."""
        if self._tq_keys is None:
            return 0
        total = 0
        for arr in self._tq_keys:
            total += arr[..., : self.offset, :].nbytes
        for arr in self._tq_values:
            total += arr[..., : self.offset, :].nbytes
        return total

    def make_mask(self, N, window_size=None, return_array=False):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(N, self.offset, return_array=return_array,
                                     window_size=window_size)


def make_turboquant_cache(model, tq_bits=3, group_size=64, seed=42):
    """Create TurboQuant KV caches for all layers."""
    num_layers = len(model.layers)
    return [
        TurboQuantKVCache(tq_bits=tq_bits, group_size=group_size, seed=seed)
        for _ in range(num_layers)
    ]


def convert_cache_to_turboquant(prompt_cache, tq_bits=3, group_size=64, seed=42):
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

        tq = TurboQuantKVCache(tq_bits=tq_bits, group_size=group_size, seed=seed)
        if c.keys is not None and c.offset > 0:
            keys = c.keys[..., : c.offset, :]
            values = c.values[..., : c.offset, :]
            tq.update_and_fetch(keys, values)
        new_cache.append(tq)
    return new_cache
