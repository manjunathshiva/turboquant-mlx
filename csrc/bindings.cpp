// Copyright 2025 TurboQuant Authors
// nanobind Python bindings for TurboQuant native ops.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "polar_ops.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "TurboQuant native Metal ops for quantized inference";

  m.def(
      "polar_qmv",
      &turboquant::polar_qmv,
      "packed_weight"_a,
      "scales"_a,
      "codebook"_a,
      "x"_a,
      "bits"_a,
      "group_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused quantized matrix-vector product via native Metal kernel.

        Computes y = dequant(packed_weight, scales, codebook) @ x
        using hardware SIMD group reduction (simd_sum).

        Args:
            packed_weight: (output_dims, packed_cols) uint32.
            scales: (output_dims, n_groups) float16.
            codebook: (n_codes,) float16.
            x: (input_dims,) float16.
            bits: Quantization bit-width (2, 3, or 4).
            group_size: Elements per group (32 or 64).

        Returns:
            (output_dims,) float16 result.
      )");

  m.def(
      "polar_gather_qmv",
      &turboquant::polar_gather_qmv,
      "packed_weight"_a,
      "scales"_a,
      "codebook"_a,
      "x"_a,
      "indices"_a,
      "bits"_a,
      "group_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused expert-routed quantized matrix-vector product (MoE decode).

        Only reads k selected experts from packed format. Uses hardware
        SIMD group reduction for maximum throughput.

        Args:
            packed_weight: (num_experts, output_dims, packed_cols) uint32.
            scales: (num_experts, output_dims, n_groups) float16.
            codebook: (n_codes,) float16.
            x: (input_dims,) float16.
            indices: (k,) uint32 — selected expert indices.
            bits: Quantization bit-width (2, 3, or 4).
            group_size: Elements per group (32 or 64).

        Returns:
            (k, output_dims) float16 result.
      )");

  m.def(
      "polar_multi_gather_qmv",
      &turboquant::polar_multi_gather_qmv,
      "packed_weight"_a,
      "scales"_a,
      "codebook"_a,
      "x"_a,
      "indices"_a,
      "bits"_a,
      "group_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multi-input expert-routed quantized matrix-vector product.

        Like polar_gather_qmv but each expert reads from its own input
        vector. Used for MoE down_proj where each expert's activation
        differs.

        Args:
            packed_weight: (num_experts, output_dims, packed_cols) uint32.
            scales: (num_experts, output_dims, n_groups) float16.
            codebook: (n_codes,) float16.
            x: (k, input_dims) — one input vector per selected expert.
            indices: (k,) uint32 — selected expert indices.
            bits: Quantization bit-width (2, 3, or 4).
            group_size: Elements per group (32 or 64).

        Returns:
            (k, output_dims) float16 result.
      )");
}
