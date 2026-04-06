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

# Use native C++ extension if available, fall back to Python kernel
try:
    from turboquant_mlx._ext import polar_gather_qmv, polar_multi_gather_qmv
except ImportError:
    from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
    from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv


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
        """Dequantize all expert weights to float16 (vectorized).

        Processes all experts in a single batched operation instead of
        looping, which is much faster on Metal.

        Returns:
            (num_experts, output_dims, input_dims) float16 tensor.
        """
        from turboquant_mlx.core.packing import unpack_indices
        from turboquant_mlx.core.codebook import dequantize_scalar

        # self.weight: (num_experts, output_dims, packed_cols)
        # self.scales: (num_experts, output_dims, n_groups)
        n_groups = self.input_dims // self.group_size

        # Unpack all experts at once — unpack_indices handles batch dims
        indices = unpack_indices(self.weight, self.bits, self.input_dims)
        # indices: (num_experts, output_dims, input_dims) uint8

        # Codebook lookup — vectorized across all experts
        w_deq = dequantize_scalar(indices, self.codebook)
        # w_deq: (num_experts, output_dims, input_dims) float16

        # Apply per-group scales
        w_deq = w_deq.reshape(self.num_experts, self.output_dims, n_groups, self.group_size)
        scales_expanded = mx.expand_dims(self.scales, axis=-1)
        w_deq = w_deq * scales_expanded

        return w_deq.reshape(self.num_experts, self.output_dims, self.input_dims)

    def __call__(self, x, indices, sorted_indices=False):
        # Apply online rotation to input if not fused into preceding norm
        if self._needs_rotation:
            x = rotate_input(x, self.signs)

        # Decode path: fused Metal kernel (no weight materialization)
        # Only reads the k selected experts, not all num_experts
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
