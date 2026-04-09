"""PolarQuantizedLinear: Drop-in replacement for nn.QuantizedLinear using TurboQuant.

Uses PolarQuant (randomized Hadamard rotation + Lloyd-Max codebook) for
weight quantization, with optional QJL residual correction (Stage 2).
Achieves much better quality at 2-3 bits than standard affine quantization.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.codebook import get_codebook, dequantize_scalar
from turboquant_mlx.core.packing import unpack_indices
from turboquant_mlx.core.rotation import rotate_input, rotate_weight
from turboquant_mlx.core.polar_quantize import polar_quantize_weight, polar_dequantize_weight
from turboquant_mlx.core.qjl import qjl_quantize, qjl_correct
# Use Python kernels - native C++ extension has ABI issues with MLX
from turboquant_mlx.kernels.polar_qmv import polar_qmv


class PolarQuantizedLinear(nn.Module):
    """Linear layer with PolarQuant weight compression + optional QJL correction.

    Stores weights as packed b-bit indices with per-group scales and a
    shared codebook. At inference, dequantizes weights via codebook lookup
    and optionally applies Hadamard rotation to inputs.

    When use_qjl=True, additionally stores 1-bit QJL sign corrections and
    per-row residual norms for unbiased inner product estimation.

    Args:
        input_dims: Input feature dimension.
        output_dims: Output feature dimension.
        bias: Whether to use a bias term.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Elements per quantization group.
        needs_rotation: Whether to apply online Hadamard rotation to inputs.
        use_qjl: Whether this layer has QJL residual correction.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = False,
        bits: int = 3,
        group_size: int = 64,
        needs_rotation: bool = True,
        use_qjl: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bits = bits
        self.group_size = group_size
        self._needs_rotation = needs_rotation
        self._use_qjl = use_qjl

        # Initialize with placeholder weights (replaced by from_linear)
        codebook, _ = get_codebook(bits, dtype=mx.float16)
        n_groups = input_dims // group_size
        elems_per_u32 = 32 // bits
        packed_cols = math.ceil(input_dims / elems_per_u32)

        self.weight = mx.zeros((output_dims, packed_cols), dtype=mx.uint32)
        self.scales = mx.ones((output_dims, n_groups), dtype=mx.float16)
        self.codebook = codebook
        self.signs = mx.ones((input_dims,), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((output_dims,), dtype=mx.float16)

        if use_qjl:
            qjl_packed_cols = math.ceil(input_dims / 32)
            self.qjl_packed = mx.zeros((output_dims, qjl_packed_cols), dtype=mx.uint32)
            self.qjl_norms = mx.zeros((output_dims,), dtype=mx.float16)
            self.qjl_signs = mx.ones((input_dims,), dtype=mx.float16)

        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # Apply online rotation if not fused into preceding norm
        if self._needs_rotation:
            x = rotate_input(x, self.signs)

        # Decode path: fused Metal kernel (no weight materialization)
        # Prefill path: software dequant + optimized GEMM
        n_vectors = 1 if x.ndim <= 1 else math.prod(x.shape[:-1])
        if n_vectors == 1:
            orig_shape = x.shape
            x_vec = x.reshape(x.shape[-1])
            y = polar_qmv(
                self.weight, self.scales, self.codebook,
                x_vec, self.bits, self.group_size,
            )
            y = y.reshape(*orig_shape[:-1], -1) if x.ndim >= 2 else y
        else:
            # Batched: dequantize and use MLX's optimized GEMM
            w = polar_dequantize_weight(
                self.weight, self.scales, self.codebook,
                self.bits, self.group_size, self.input_dims,
            )
            y = x @ w.T

        # QJL residual correction (Stage 2)
        if self._use_qjl and "qjl_packed" in self:
            correction = qjl_correct(
                self.qjl_packed, self.qjl_norms, self.qjl_signs,
                x, self.input_dims,
            )
            y = y + correction

        if "bias" in self:
            y = y + self.bias

        return y

    def _extra_repr(self):
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, bits={self.bits}, group_size={self.group_size}, "
            f"rotation={'online' if self._needs_rotation else 'fused'}, "
            f"qjl={self._use_qjl}"
        )

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Module,
        bits: int = 3,
        group_size: int = 64,
        seed: int = 42,
        needs_rotation: bool = True,
        use_qjl: bool = False,
        qjl_seed: int = 137,
    ) -> "PolarQuantizedLinear":
        """Create a PolarQuantizedLinear from an existing nn.Linear layer.

        Args:
            linear_layer: Source nn.Linear layer with float weights.
            bits: Quantization bit-width (2, 3, or 4).
            group_size: Elements per quantization group.
            seed: Random seed for Hadamard rotation signs.
            needs_rotation: Whether this layer needs online input rotation.
            use_qjl: Whether to apply QJL residual correction (Stage 2).
            qjl_seed: Random seed for QJL projection signs.

        Returns:
            New PolarQuantizedLinear with quantized weights.
        """
        weight = linear_layer.weight  # (output_dims, input_dims)
        output_dims, input_dims = weight.shape
        has_bias = "bias" in linear_layer

        # Stage 1: PolarQuant
        result = polar_quantize_weight(weight, bits, group_size, seed)

        # Create layer
        layer = cls(
            input_dims, output_dims,
            bias=has_bias, bits=bits, group_size=group_size,
            needs_rotation=needs_rotation, use_qjl=use_qjl,
        )
        layer.weight = result["packed_weight"]
        layer.scales = result["scales"]
        layer.codebook = result["codebook"]
        layer.signs = result["signs"]

        # Stage 2: QJL residual correction
        if use_qjl:
            # Compute residual in rotated space
            w_rot = rotate_weight(
                weight.astype(mx.float32),
                result["signs"].astype(mx.float32),
            )
            w_deq = polar_dequantize_weight(
                result["packed_weight"], result["scales"], result["codebook"],
                bits, group_size, input_dims,
            )
            residual = w_rot - w_deq.astype(mx.float32)
            mx.eval(residual)

            qjl_result = qjl_quantize(residual, seed=qjl_seed)
            layer.qjl_packed = qjl_result["qjl_packed"]
            layer.qjl_norms = qjl_result["qjl_norms"]
            layer.qjl_signs = qjl_result["qjl_signs"]

        if has_bias:
            layer.bias = linear_layer.bias.astype(mx.float16)

        layer.freeze()
        return layer

    @classmethod
    def from_quantized_dict(
        cls,
        params: dict,
        input_dims: int,
        output_dims: int,
        bias: bool = False,
        bits: int = 3,
        group_size: int = 64,
        needs_rotation: bool = True,
        use_qjl: bool = False,
    ) -> "PolarQuantizedLinear":
        """Create from a dict of pre-quantized parameters (for model loading).

        Args:
            params: Dict with keys 'weight', 'scales', 'codebook', 'signs',
                    and optionally 'bias', 'qjl_packed', 'qjl_norms', 'qjl_signs'.
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether bias is present.
            bits: Quantization bit-width.
            group_size: Elements per quantization group.
            needs_rotation: Whether this layer needs online input rotation.
            use_qjl: Whether QJL correction is present.

        Returns:
            New PolarQuantizedLinear with loaded parameters.
        """
        layer = cls(
            input_dims, output_dims,
            bias=bias, bits=bits, group_size=group_size,
            needs_rotation=needs_rotation, use_qjl=use_qjl,
        )
        layer.weight = params["weight"]
        layer.scales = params["scales"]
        layer.codebook = params["codebook"]
        layer.signs = params["signs"]
        if use_qjl:
            layer.qjl_packed = params["qjl_packed"]
            layer.qjl_norms = params["qjl_norms"]
            layer.qjl_signs = params["qjl_signs"]
        if bias and "bias" in params:
            layer.bias = params["bias"]
        layer.freeze()
        return layer
