"""PolarQuantizedLinear: Drop-in replacement for nn.QuantizedLinear using TurboQuant.

Uses PolarQuant (randomized Hadamard rotation + Lloyd-Max codebook) for
weight quantization, with optional QJL residual correction (Stage 2).
Achieves much better quality at 2-3 bits than standard affine quantization.
"""

import math
import os

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.codebook import get_codebook
from turboquant_mlx.core.rotation import rotate_input, rotate_weight
from turboquant_mlx.core.polar_quantize import polar_quantize_weight, polar_dequantize_weight
from turboquant_mlx.core.qjl import qjl_quantize, qjl_correct
# Use Python kernels - native C++ extension has ABI issues with MLX
from turboquant_mlx.kernels.polar_qmv import polar_qmv
from turboquant_mlx.kernels.polar_qmm import polar_qmm

# Batched dispatch bounds for the fused polar_qmm kernel, measured on an M4 Max
# against the dequantize + GEMM path it replaces (3-bit, g64):
#
#   shape                N=2    N=8   N=32  N=128  N=512  N=2048   dq spike
#   mlp gate/up (19968) 9.30x  9.19x  5.81x  1.95x  0.75x   0.41x    1.35 GB
#   o_proj      ( 6656) 5.81x  5.84x  4.29x  1.81x  0.73x   0.42x    0.12 GB
#   q_proj      ( 4096) 2.18x  4.44x  2.81x  1.35x  1.88x   0.42x    0.27 GB
#   k/v_proj    (  256) 0.71x  0.75x  0.79x  0.84x  0.60x   0.51x    0.00 GB
#
# Above ~256 tokens MLX's tuned GEMM beats the fused kernel on time, so the
# default keeps the old path there. Below 512 output dims there are too few
# 64-row output blocks to fill the GPU, and the weight is small enough that
# materializing it costs nothing anyway.
_QMM_MAX_TOKENS = int(os.environ.get("TURBOQUANT_QMM_MAX_TOKENS", "256"))
_QMM_MIN_OUTPUT_DIMS = 512


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
        # The fused Metal kernels cannot compile against bfloat16 activations.
        # Nothing used to enforce that, because rotate_input multiplied by the
        # float16 `signs` vector and MLX promoted bf16 -> float32 on the way
        # out. With rotation="none" the raw bf16 reached the kernel and Metal
        # failed to build the library. Promote to float32 here, which is
        # exactly what the rotated path already hands the kernel for a bf16
        # model. Deliberately narrow: after rotate_input the dtype is float32
        # or float16, never bfloat16, so this is provably a no-op whenever
        # rotation ran.
        if x.dtype == mx.bfloat16:
            x = x.astype(mx.float32)

        # Decode path (1 token): fused matrix-vector kernel.
        # Small batch: fused matrix-matrix kernel — faster AND allocates
        #   nothing, because it decodes the packed weight in registers.
        # Large batch: dequantize + MLX GEMM, which wins on time above
        #   _QMM_MAX_TOKENS but materializes the full weight (~14 bytes per
        #   parameter transient — the reason prefill peak dwarfs the model).
        n_vectors = 1 if x.ndim <= 1 else math.prod(x.shape[:-1])
        if n_vectors == 1:
            orig_shape = x.shape
            x_vec = x.reshape(x.shape[-1])
            y = polar_qmv(
                self.weight, self.scales, self.codebook,
                x_vec, self.bits, self.group_size,
            )
            y = y.reshape(*orig_shape[:-1], -1) if x.ndim >= 2 else y
        elif (n_vectors <= _QMM_MAX_TOKENS
                and self.output_dims >= _QMM_MIN_OUTPUT_DIMS):
            orig_shape = x.shape
            y = polar_qmm(
                self.weight, self.scales, self.codebook,
                x.reshape(n_vectors, orig_shape[-1]),
                self.bits, self.group_size,
            )
            y = y.reshape(*orig_shape[:-1], -1)
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
        # One flag drives both halves: the weights are rotated iff the layer
        # will rotate its input. They can never disagree.
        result = polar_quantize_weight(weight, bits, group_size, seed,
                                       rotate=needs_rotation)

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

        # Stage 2: QJL residual correction.
        # The residual must be measured in the SAME domain the weights were
        # packed in, because at inference qjl_correct() is applied to the very
        # x the matmul saw — rotated iff needs_rotation. rotate_weight() cannot
        # be used unconditionally: with rotation off the signs are all-ones,
        # but it still applies the Hadamard, so the target would be rotated
        # while w_deq is not, and the "correction" would be noise.
        if use_qjl:
            w_target = (
                rotate_weight(
                    weight.astype(mx.float32),
                    result["signs"].astype(mx.float32),
                )
                if needs_rotation
                else weight.astype(mx.float32)
            )
            w_deq = polar_dequantize_weight(
                result["packed_weight"], result["scales"], result["codebook"],
                bits, group_size, input_dims,
            )
            residual = w_target - w_deq.astype(mx.float32)
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
