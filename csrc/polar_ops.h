// Copyright 2025 TurboQuant Authors
// Custom MLX primitives for PolarQuant quantized matrix-vector products.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace turboquant {

// Locate the directory of the current binary (to find the .metallib)
std::string current_binary_dir();

///////////////////////////////////////////////////////////////////////////////
// Operations (Python-facing)
///////////////////////////////////////////////////////////////////////////////

mx::array polar_qmv(
    const mx::array& packed_weight, // (output_dims, packed_cols) uint32
    const mx::array& scales,        // (output_dims, n_groups) float16
    const mx::array& codebook,      // (n_codes,) float16
    const mx::array& x,             // (input_dims,) float16
    int bits,
    int group_size,
    mx::StreamOrDevice s = {});

mx::array polar_gather_qmv(
    const mx::array& packed_weight, // (num_experts, output_dims, packed_cols)
    const mx::array& scales,        // (num_experts, output_dims, n_groups)
    const mx::array& codebook,      // (n_codes,) float16
    const mx::array& x,             // (input_dims,) float16
    const mx::array& indices,       // (k,) uint32
    int bits,
    int group_size,
    mx::StreamOrDevice s = {});

mx::array polar_multi_gather_qmv(
    const mx::array& packed_weight, // (num_experts, output_dims, packed_cols)
    const mx::array& scales,        // (num_experts, output_dims, n_groups)
    const mx::array& codebook,      // (n_codes,) float16
    const mx::array& x,             // (k, input_dims) — one vector per expert
    const mx::array& indices,       // (k,) uint32
    int bits,
    int group_size,
    mx::StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitives
///////////////////////////////////////////////////////////////////////////////

class PolarQMV : public mx::Primitive {
 public:
  explicit PolarQMV(mx::Stream stream, int bits, int group_size)
      : mx::Primitive(stream), bits_(bits), group_size_(group_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  const char* name() const override { return "PolarQMV"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int bits_;
  int group_size_;
};

class PolarGatherQMV : public mx::Primitive {
 public:
  explicit PolarGatherQMV(mx::Stream stream, int bits, int group_size)
      : mx::Primitive(stream), bits_(bits), group_size_(group_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  const char* name() const override { return "PolarGatherQMV"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int bits_;
  int group_size_;
};

class PolarMultiGatherQMV : public mx::Primitive {
 public:
  explicit PolarMultiGatherQMV(mx::Stream stream, int bits, int group_size)
      : mx::Primitive(stream), bits_(bits), group_size_(group_size) {}

  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;

  const char* name() const override { return "PolarMultiGatherQMV"; }
  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  int bits_;
  int group_size_;
};

}  // namespace turboquant
