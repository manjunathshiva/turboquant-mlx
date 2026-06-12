"""PolarQuantizedSwitchLinear: TurboQuant-compressed MoE expert weights.

Drop-in replacement for SwitchLinear that stores expert weights using
PolarQuant (Hadamard rotation + Lloyd-Max codebook). At inference time,
dequantizes weights and uses mx.gather_mm for expert-routed matmul.

Expert weights are stored as 3D packed arrays:
  - weight: (num_experts, output_dims, packed_cols) uint32
  - scales: (num_experts, output_dims, n_groups) float16
  - codebook: (2^bits,) float16 — shared across all experts
  - signs: (input_dims,) float16 — shared rotation signs
"""

import math

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.codebook import get_codebook
from turboquant_mlx.core.polar_quantize import polar_quantize_weight, polar_dequantize_weight
from turboquant_mlx.core.rotation import rotate_input

# Use Python kernels - native C++ extension has ABI issues with MLX
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv
from turboquant_mlx.kernels.polar_dequant_experts import polar_dequant_experts

# Above this many (token, expert) routings, materializing fp16 expert weights
# once (fused dequant + mx.gather_mm) beats the per-row gather kernels, which
# re-read the activation vector from global memory per output row. Diffusion
# canvas forwards (e.g. DiffusionGemma: 256 tokens x top-8 = 2048 routings)
# sit far above this; autoregressive decode sits far below.
_GATHER_MM_MIN_ROUTINGS = 512

# ... but only when the full dequantized expert tensor stays small. Models
# with hundreds of large experts (e.g. 512-expert LatentMoE) would OOM on the
# materialization (issue #1), so they keep the gather kernels at any k.
_GATHER_MM_MAX_DEQUANT_BYTES = 2 << 30


class PolarQuantizedSwitchLinear(nn.Module):
    """MoE expert linear layer with PolarQuant weight compression.

    Stores N expert weight matrices in packed codebook format.
    At inference, dequantizes and delegates to mx.gather_mm.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        num_experts: Number of expert weight matrices.
        bias: Whether to use a bias term.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Elements per quantization group.
        needs_rotation: Whether to apply online Hadamard rotation to inputs.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = False,
        bits: int = 3,
        group_size: int = 64,
        needs_rotation: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.bits = bits
        self.group_size = group_size
        self._needs_rotation = needs_rotation

        # Placeholder weights (replaced by from_switch_linear)
        codebook, _ = get_codebook(bits, dtype=mx.float16)
        n_groups = input_dims // group_size
        elems_per_u32 = 32 // bits
        packed_cols = math.ceil(input_dims / elems_per_u32)

        self.weight = mx.zeros((num_experts, output_dims, packed_cols), dtype=mx.uint32)
        self.scales = mx.ones((num_experts, output_dims, n_groups), dtype=mx.float16)
        self.codebook = codebook
        self.signs = mx.ones((input_dims,), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((num_experts, output_dims), dtype=mx.float16)

        self.freeze()

    def _dequantize_all(self) -> mx.array:
        """Dequantize all expert weights to float16 (fused Metal kernel).

        Single-pass unpack + codebook lookup + group scaling — bit-identical
        to the previous multi-op Python path and ~11x faster at MoE shapes.

        Returns:
            (num_experts, output_dims, input_dims) float16 tensor.
        """
        return polar_dequant_experts(
            self.weight, self.scales, self.codebook,
            self.bits, self.group_size,
        )

    def _dequant_bytes(self) -> int:
        """Size of the fully materialized fp16 expert tensor."""
        return self.num_experts * self.output_dims * self.input_dims * 2

    def __call__(self, x, indices, sorted_indices=False):
        # Apply online rotation to input if not fused into preceding norm
        if self._needs_rotation:
            x = rotate_input(x, self.signs)

        # Decode path: fused Metal kernel (no weight materialization)
        # Only reads the k selected experts, not all num_experts.
        # NOTE: when mlx-lm's SwitchMLP triggers do_sort, x.ndim drops to 3
        # and indices.ndim drops to 1, so n_tokens and k both equal
        # n_real_tokens * k_real and the n_tokens==k branch fires. That is
        # the correct behavior — multi_gather_qmv handles N (token,expert)
        # routings without materializing all experts. We DO NOT fall back to
        # _dequantize_all here because this model has 512 experts (top-22)
        # and full dequant of one layer is ~10 GB → instant OOM.  See
        # issue #1; the kernel itself chunks for very large N.
        n_tokens = 1 if x.ndim <= 2 else math.prod(x.shape[:-2])
        k = indices.shape[-1] if indices.ndim >= 1 else 1

        if n_tokens == 1:
            # Single token, shared input: use polar_gather_qmv
            # x shape from SwitchGLU: (..., 1, 1, input_dims)
            # indices shape: (..., k)
            x_flat = x.reshape(-1)  # (input_dims,)
            idx_flat = indices.reshape(-1)  # (k,)

            y = polar_gather_qmv(
                self.weight, self.scales, self.codebook,
                x_flat, idx_flat,
                self.bits, self.group_size,
            )  # (k, output_dims)

            # Reshape to match gather_mm output: (..., k, 1, output_dims)
            target_shape = list(indices.shape) + [1, self.output_dims]
            y = y.reshape(target_shape)

            if "bias" in self:
                y = y + mx.expand_dims(self["bias"][indices], -2)
            return y

        if n_tokens == k:
            # Large batched routing (diffusion canvas / big prefill via
            # SwitchMLP's do_sort): the per-row gather kernel re-reads x per
            # output row, so past _GATHER_MM_MIN_ROUTINGS it loses to a fused
            # dequant + native gather_mm — but only when the materialized
            # fp16 expert tensor stays small (512-expert models would OOM).
            if (k >= _GATHER_MM_MIN_ROUTINGS
                    and self._dequant_bytes() <= _GATHER_MM_MAX_DEQUANT_BYTES):
                w_deq = self._dequantize_all()
                y = mx.gather_mm(
                    x,
                    w_deq.swapaxes(-1, -2),
                    rhs_indices=indices,
                    sorted_indices=sorted_indices,
                )
                if "bias" in self:
                    y = y + mx.expand_dims(self["bias"][indices], -2)
                return y

            # Multi-input decode: k expert vectors (down_proj path)
            # x shape: (..., k, 1, input_dims) — one vector per expert
            # Use polar_multi_gather_qmv to avoid dequantizing all experts
            orig_shape = x.shape
            x_2d = x.reshape(k, self.input_dims)  # (k, input_dims)
            idx_flat = indices.reshape(-1)  # (k,)

            y = polar_multi_gather_qmv(
                self.weight, self.scales, self.codebook,
                x_2d, idx_flat,
                self.bits, self.group_size,
            )  # (k, output_dims)

            # Reshape to match gather_mm output: (..., k, 1, output_dims)
            target_shape = list(indices.shape) + [1, self.output_dims]
            y = y.reshape(target_shape)

            if "bias" in self:
                y = y + mx.expand_dims(self["bias"][indices], -2)
            return y

        # Prefill path: vectorized dequant + gather_mm
        w_deq = self._dequantize_all()  # (num_experts, out, in)
        y = mx.gather_mm(
            x,
            w_deq.swapaxes(-1, -2),  # (num_experts, input_dims, output_dims)
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            y = y + mx.expand_dims(self["bias"][indices], -2)
        return y

    def _extra_repr(self):
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"num_experts={self.num_experts}, bias={'bias' in self}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"rotation={'online' if self._needs_rotation else 'fused'}"
        )

    @classmethod
    def from_switch_linear(
        cls,
        switch_linear,
        bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
        needs_rotation: bool = True,
        float_weight: mx.array = None,
        bias: mx.array = None,
    ) -> "PolarQuantizedSwitchLinear":
        """Create from an existing SwitchLinear with FP16/BF16 weights.

        Quantizes each expert's weight matrix independently using the same
        rotation signs and codebook, then stacks into 3D storage.

        Args:
            switch_linear: Source SwitchLinear (or None if float_weight provided).
            bits: Quantization bit-width (2, 3, or 4).
            group_size: Elements per quantization group.
            seed: Random seed for Hadamard rotation signs.
            needs_rotation: Whether this layer needs online input rotation.
            float_weight: Optional pre-dequantized weight (N, out, in) float.
            bias: Optional bias tensor (N, out) float.

        Returns:
            New PolarQuantizedSwitchLinear with quantized expert weights.
        """
        if float_weight is not None:
            weight_3d = float_weight
            has_bias = bias is not None
        else:
            weight_3d = switch_linear.weight
            has_bias = "bias" in switch_linear
            if has_bias:
                bias = switch_linear.bias
        num_experts, output_dims, input_dims = weight_3d.shape

        if input_dims % group_size != 0:
            raise ValueError(
                f"input_dims ({input_dims}) must be divisible by "
                f"group_size ({group_size})"
            )

        # Pre-allocate output arrays to avoid accumulating large lists
        from turboquant_mlx.core.packing import pack_indices as _pack
        n_groups = input_dims // group_size
        elems_per_u32 = 32 // bits
        packed_cols = math.ceil(input_dims / elems_per_u32)

        packed_3d = mx.zeros((num_experts, output_dims, packed_cols), dtype=mx.uint32)
        scales_3d = mx.zeros((num_experts, output_dims, n_groups), dtype=mx.float16)
        codebook = None
        signs = None

        # Process experts one-by-one, eval each to flush graph immediately
        for e in range(num_experts):
            expert_w = weight_3d[e]  # view into (output_dims, input_dims)
            result = polar_quantize_weight(
                expert_w,
                bits=bits,
                group_size=group_size,
                seed=seed,
            )
            packed_3d[e] = result["packed_weight"]
            scales_3d[e] = result["scales"]
            if codebook is None:
                codebook = result["codebook"]
                signs = result["signs"]
            # Eval every expert to keep graph small and free intermediates
            mx.eval(packed_3d[e], scales_3d[e])
            del result, expert_w

        # Free original weight reference
        del weight_3d

        # Create layer
        layer = cls(
            input_dims, output_dims, num_experts,
            bias=has_bias, bits=bits, group_size=group_size,
            needs_rotation=needs_rotation,
        )
        layer.weight = packed_3d
        layer.scales = scales_3d
        layer.codebook = codebook
        layer.signs = signs

        if has_bias and bias is not None:
            layer.bias = bias.astype(mx.float16)

        layer.freeze()
        return layer
