// Copyright 2025 TurboQuant Authors
// PolarQMV and PolarGatherQMV — fused Metal kernels for quantized MoE decode.
// Uses precompiled .metallib with hardware SIMD group reduction.

#include <dlfcn.h>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

#include "polar_ops.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace turboquant {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

///////////////////////////////////////////////////////////////////////////////
// Python-facing operations
///////////////////////////////////////////////////////////////////////////////

mx::array polar_qmv(
    const mx::array& packed_weight,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& x,
    int bits,
    int group_size,
    mx::StreamOrDevice s) {
  int output_dims = packed_weight.shape(0);
  return mx::array(
      mx::Shape{output_dims},
      x.dtype(),
      std::make_shared<PolarQMV>(mx::to_stream(s), bits, group_size),
      {packed_weight, scales, codebook, x});
}

mx::array polar_gather_qmv(
    const mx::array& packed_weight,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& x,
    const mx::array& indices,
    int bits,
    int group_size,
    mx::StreamOrDevice s) {
  int k = indices.shape(0);
  int output_dims = packed_weight.shape(1);
  return mx::array(
      mx::Shape{k, output_dims},
      x.dtype(),
      std::make_shared<PolarGatherQMV>(mx::to_stream(s), bits, group_size),
      {packed_weight, scales, codebook, x, indices});
}

mx::array polar_multi_gather_qmv(
    const mx::array& packed_weight,
    const mx::array& scales,
    const mx::array& codebook,
    const mx::array& x,
    const mx::array& indices,
    int bits,
    int group_size,
    mx::StreamOrDevice s) {
  int k = indices.shape(0);
  int output_dims = packed_weight.shape(1);
  return mx::array(
      mx::Shape{k, output_dims},
      x.dtype(),
      std::make_shared<PolarMultiGatherQMV>(mx::to_stream(s), bits, group_size),
      {packed_weight, scales, codebook, x, indices});
}

///////////////////////////////////////////////////////////////////////////////
// CPU fallback — not implemented (GPU-only)
///////////////////////////////////////////////////////////////////////////////

void PolarQMV::eval_cpu(const std::vector<mx::array>&,
                        std::vector<mx::array>&) {
  throw std::runtime_error("PolarQMV is GPU-only.");
}

void PolarGatherQMV::eval_cpu(const std::vector<mx::array>&,
                              std::vector<mx::array>&) {
  throw std::runtime_error("PolarGatherQMV is GPU-only.");
}

void PolarMultiGatherQMV::eval_cpu(const std::vector<mx::array>&,
                                   std::vector<mx::array>&) {
  throw std::runtime_error("PolarMultiGatherQMV is GPU-only.");
}

///////////////////////////////////////////////////////////////////////////////
// Metal GPU implementation — precompiled metallib
///////////////////////////////////////////////////////////////////////////////

#ifdef _METAL_

void PolarQMV::eval_gpu(const std::vector<mx::array>& inputs,
                        std::vector<mx::array>& outputs) {
  auto& packed_weight = inputs[0];
  auto& scales = inputs[1];
  auto& codebook = inputs[2];
  auto& x = inputs[3];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  int output_dims = packed_weight.shape(0);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  // Kernel name matches [[host_name(...)]] in polar_kernels.metal
  std::string kname = "polar_qmv_";
  kname += std::to_string(bits_) + "bit_gs";
  kname += std::to_string(group_size_) + "_";
  kname += mx::type_to_name(out);

  auto lib = d.get_library("turboquant_ext", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(packed_weight, 0);
  enc.set_input_array(scales, 1);
  enc.set_input_array(codebook, 2);
  enc.set_input_array(x, 3);
  enc.set_output_array(out, 4);

  uint32_t n_rows = static_cast<uint32_t>(output_dims);
  uint32_t n_groups = static_cast<uint32_t>(scales.shape(1));
  uint32_t pw_stride = static_cast<uint32_t>(packed_weight.shape(1));
  enc.set_bytes(n_rows, 5);
  enc.set_bytes(n_groups, 6);
  enc.set_bytes(pw_stride, 7);

  constexpr int SIMD_SIZE = 32;
  MTL::Size grid_dims = MTL::Size(output_dims * SIMD_SIZE, 1, 1);
  MTL::Size group_dims = MTL::Size(SIMD_SIZE, 1, 1);
  enc.dispatch_threads(grid_dims, group_dims);
}

void PolarGatherQMV::eval_gpu(const std::vector<mx::array>& inputs,
                              std::vector<mx::array>& outputs) {
  auto& packed_weight = inputs[0];
  auto& scales = inputs[1];
  auto& codebook = inputs[2];
  auto& x = inputs[3];
  auto& indices = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  int k = indices.shape(0);
  int output_dims = packed_weight.shape(1);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  std::string kname = "polar_gather_qmv_";
  kname += std::to_string(bits_) + "bit_gs";
  kname += std::to_string(group_size_) + "_";
  kname += mx::type_to_name(out);

  auto lib = d.get_library("turboquant_ext", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(packed_weight, 0);
  enc.set_input_array(scales, 1);
  enc.set_input_array(codebook, 2);
  enc.set_input_array(x, 3);
  enc.set_input_array(indices, 4);
  enc.set_output_array(out, 5);

  uint32_t out_dims = static_cast<uint32_t>(output_dims);
  uint32_t n_groups = static_cast<uint32_t>(scales.shape(2));
  uint32_t pw_cols = static_cast<uint32_t>(packed_weight.shape(2));
  uint32_t k_val = static_cast<uint32_t>(k);
  enc.set_bytes(out_dims, 6);
  enc.set_bytes(n_groups, 7);
  enc.set_bytes(pw_cols, 8);
  enc.set_bytes(k_val, 9);

  constexpr int SIMD_SIZE = 32;
  size_t total_work = static_cast<size_t>(k) * output_dims;
  MTL::Size grid_dims = MTL::Size(total_work * SIMD_SIZE, 1, 1);
  MTL::Size group_dims = MTL::Size(SIMD_SIZE, 1, 1);
  enc.dispatch_threads(grid_dims, group_dims);
}

void PolarMultiGatherQMV::eval_gpu(const std::vector<mx::array>& inputs,
                                   std::vector<mx::array>& outputs) {
  auto& packed_weight = inputs[0];
  auto& scales = inputs[1];
  auto& codebook = inputs[2];
  auto& x = inputs[3];
  auto& indices = inputs[4];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  int k = indices.shape(0);
  int output_dims = packed_weight.shape(1);
  int input_dims = x.shape(1);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  std::string kname = "polar_multi_gather_qmv_";
  kname += std::to_string(bits_) + "bit_gs";
  kname += std::to_string(group_size_) + "_";
  kname += mx::type_to_name(out);

  auto lib = d.get_library("turboquant_ext", current_binary_dir());
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(packed_weight, 0);
  enc.set_input_array(scales, 1);
  enc.set_input_array(codebook, 2);
  enc.set_input_array(x, 3);
  enc.set_input_array(indices, 4);
  enc.set_output_array(out, 5);

  uint32_t out_dims = static_cast<uint32_t>(output_dims);
  uint32_t n_groups = static_cast<uint32_t>(scales.shape(2));
  uint32_t pw_cols = static_cast<uint32_t>(packed_weight.shape(2));
  uint32_t k_val = static_cast<uint32_t>(k);
  uint32_t in_dims = static_cast<uint32_t>(input_dims);
  enc.set_bytes(out_dims, 6);
  enc.set_bytes(n_groups, 7);
  enc.set_bytes(pw_cols, 8);
  enc.set_bytes(k_val, 9);
  enc.set_bytes(in_dims, 10);

  constexpr int SIMD_SIZE = 32;
  size_t total_work = static_cast<size_t>(k) * output_dims;
  MTL::Size grid_dims = MTL::Size(total_work * SIMD_SIZE, 1, 1);
  MTL::Size group_dims = MTL::Size(SIMD_SIZE, 1, 1);
  enc.dispatch_threads(grid_dims, group_dims);
}

#else

void PolarQMV::eval_gpu(const std::vector<mx::array>&,
                        std::vector<mx::array>&) {
  throw std::runtime_error("PolarQMV requires Metal.");
}
void PolarGatherQMV::eval_gpu(const std::vector<mx::array>&,
                              std::vector<mx::array>&) {
  throw std::runtime_error("PolarGatherQMV requires Metal.");
}
void PolarMultiGatherQMV::eval_gpu(const std::vector<mx::array>&,
                                   std::vector<mx::array>&) {
  throw std::runtime_error("PolarMultiGatherQMV requires Metal.");
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Equivalence
///////////////////////////////////////////////////////////////////////////////

bool PolarQMV::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const PolarQMV&>(other);
  return bits_ == o.bits_ && group_size_ == o.group_size_;
}

bool PolarGatherQMV::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const PolarGatherQMV&>(other);
  return bits_ == o.bits_ && group_size_ == o.group_size_;
}

bool PolarMultiGatherQMV::is_equivalent(const mx::Primitive& other) const {
  auto& o = static_cast<const PolarMultiGatherQMV&>(other);
  return bits_ == o.bits_ && group_size_ == o.group_size_;
}

}  // namespace turboquant
