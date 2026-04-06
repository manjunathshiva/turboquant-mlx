// Copyright 2025 TurboQuant Authors
// Fused Metal kernels for PolarQuant matrix-vector multiplication.
// Uses SIMD groups (simd_sum) for hardware-accelerated reduction.
//
// Scales and codebook are always half-precision (float16).
// Input x and output use type T (half, bfloat, or float).

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

static constant constexpr int SIMD_SIZE = 32;

///////////////////////////////////////////////////////////////////////////////
// polar_qmv: Single-layer quantized matrix-vector product
//
// One simdgroup (32 threads) handles one output row.
// Groups are distributed across lanes; simd_sum reduces.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int BITS, int GROUP_SIZE>
[[kernel]] void polar_qmv(
    device const uint32_t* packed_weight [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const half* codebook         [[buffer(2)]],
    device const T* x                   [[buffer(3)]],
    device T* out                       [[buffer(4)]],
    constant uint& n_rows               [[buffer(5)]],
    constant uint& n_groups             [[buffer(6)]],
    constant uint& pw_stride            [[buffer(7)]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint tg_id       [[threadgroup_position_in_grid]]) {

  constexpr int N_CODES = 1 << BITS;
  constexpr int ELEMS_PER_U32 = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;

  uint row = tg_id;
  if (row >= n_rows) return;

  // Load codebook into registers
  float cb[N_CODES];
  for (int i = 0; i < N_CODES; i++) {
    cb[i] = float(codebook[i]);
  }

  float accum = 0.0f;

  // Each lane processes groups: lane, lane+32, lane+64, ...
  for (uint g = simd_lane; g < n_groups; g += SIMD_SIZE) {
    float scale = float(scales[row * n_groups + g]);
    uint base_col = g * GROUP_SIZE;
    float group_accum = 0.0f;

    for (uint e = 0; e < uint(GROUP_SIZE); e++) {
      uint col = base_col + e;
      uint packed_col = col / uint(ELEMS_PER_U32);
      uint bit_pos = (col % uint(ELEMS_PER_U32)) * uint(BITS);

      uint packed_val = packed_weight[row * pw_stride + packed_col];
      uint code_idx = (packed_val >> bit_pos) & MASK;

      group_accum += cb[code_idx] * float(x[col]);
    }

    accum += group_accum * scale;
  }

  // Hardware SIMD reduction — single-cycle 32-thread sum
  accum = simd_sum(accum);

  if (simd_lane == 0) {
    out[row] = T(accum);
  }
}

///////////////////////////////////////////////////////////////////////////////
// polar_gather_qmv: Expert-routed quantized matrix-vector product (MoE)
//
// One simdgroup handles one (expert, output_row) pair.
// Only reads the k selected experts from packed format.
///////////////////////////////////////////////////////////////////////////////

template <typename T, int BITS, int GROUP_SIZE>
[[kernel]] void polar_gather_qmv(
    device const uint32_t* packed_weight [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const half* codebook         [[buffer(2)]],
    device const T* x                   [[buffer(3)]],
    device const uint32_t* indices      [[buffer(4)]],
    device T* out                       [[buffer(5)]],
    constant uint& out_dims             [[buffer(6)]],
    constant uint& n_groups             [[buffer(7)]],
    constant uint& pw_cols              [[buffer(8)]],
    constant uint& k                    [[buffer(9)]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint tg_id       [[threadgroup_position_in_grid]]) {

  constexpr int N_CODES = 1 << BITS;
  constexpr int ELEMS_PER_U32 = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;

  uint total_work = k * out_dims;
  if (tg_id >= total_work) return;

  uint expert_local = tg_id / out_dims;
  uint row = tg_id % out_dims;
  uint expert_id = indices[expert_local];

  // Load codebook into registers
  float cb[N_CODES];
  for (int i = 0; i < N_CODES; i++) {
    cb[i] = float(codebook[i]);
  }

  // Base offsets for this expert and row
  uint pw_base = expert_id * out_dims * pw_cols + row * pw_cols;
  uint sc_base = expert_id * out_dims * n_groups + row * n_groups;

  float accum = 0.0f;

  for (uint g = simd_lane; g < n_groups; g += SIMD_SIZE) {
    float scale = float(scales[sc_base + g]);
    uint base_col = g * GROUP_SIZE;
    float group_accum = 0.0f;

    for (uint e = 0; e < uint(GROUP_SIZE); e++) {
      uint col = base_col + e;
      uint packed_col = col / uint(ELEMS_PER_U32);
      uint bit_pos = (col % uint(ELEMS_PER_U32)) * uint(BITS);

      uint packed_val = packed_weight[pw_base + packed_col];
      uint code_idx = (packed_val >> bit_pos) & MASK;

      group_accum += cb[code_idx] * float(x[col]);
    }

    accum += group_accum * scale;
  }

  // Hardware SIMD reduction
  accum = simd_sum(accum);

  if (simd_lane == 0) {
    out[expert_local * out_dims + row] = T(accum);
  }
}

///////////////////////////////////////////////////////////////////////////////
// polar_multi_gather_qmv: Multi-input expert-routed quantized MV (down_proj)
//
// Like polar_gather_qmv but each expert reads from its OWN input vector
// instead of sharing a single input. Used for MoE down_proj where each
// expert's activation is different.
//
// x layout: (k, input_dims) — k separate input vectors
///////////////////////////////////////////////////////////////////////////////

template <typename T, int BITS, int GROUP_SIZE>
[[kernel]] void polar_multi_gather_qmv(
    device const uint32_t* packed_weight [[buffer(0)]],
    device const half* scales           [[buffer(1)]],
    device const half* codebook         [[buffer(2)]],
    device const T* x                   [[buffer(3)]],
    device const uint32_t* indices      [[buffer(4)]],
    device T* out                       [[buffer(5)]],
    constant uint& out_dims             [[buffer(6)]],
    constant uint& n_groups             [[buffer(7)]],
    constant uint& pw_cols              [[buffer(8)]],
    constant uint& k                    [[buffer(9)]],
    constant uint& in_dims              [[buffer(10)]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint tg_id       [[threadgroup_position_in_grid]]) {

  constexpr int N_CODES = 1 << BITS;
  constexpr int ELEMS_PER_U32 = 32 / BITS;
  constexpr uint MASK = (1u << BITS) - 1u;

  uint total_work = k * out_dims;
  if (tg_id >= total_work) return;

  uint expert_local = tg_id / out_dims;
  uint row = tg_id % out_dims;
  uint expert_id = indices[expert_local];

  // Load codebook into registers
  float cb[N_CODES];
  for (int i = 0; i < N_CODES; i++) {
    cb[i] = float(codebook[i]);
  }

  // Base offsets for this expert and row
  uint pw_base = expert_id * out_dims * pw_cols + row * pw_cols;
  uint sc_base = expert_id * out_dims * n_groups + row * n_groups;

  // Base offset for this expert's input vector
  uint x_base = expert_local * in_dims;

  float accum = 0.0f;

  for (uint g = simd_lane; g < n_groups; g += SIMD_SIZE) {
    float scale = float(scales[sc_base + g]);
    uint base_col = g * GROUP_SIZE;
    float group_accum = 0.0f;

    for (uint e = 0; e < uint(GROUP_SIZE); e++) {
      uint col = base_col + e;
      uint packed_col = col / uint(ELEMS_PER_U32);
      uint bit_pos = (col % uint(ELEMS_PER_U32)) * uint(BITS);

      uint packed_val = packed_weight[pw_base + packed_col];
      uint code_idx = (packed_val >> bit_pos) & MASK;

      // Key difference: read from expert_local's input vector
      group_accum += cb[code_idx] * float(x[x_base + col]);
    }

    accum += group_accum * scale;
  }

  // Hardware SIMD reduction
  accum = simd_sum(accum);

  if (simd_lane == 0) {
    out[expert_local * out_dims + row] = T(accum);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Template instantiations
//
// Scales/codebook are always half. T controls x and output dtype.
// Instantiate for half, bfloat, and float with common bit/group combos.
///////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_QMV(T_TYPE, BITS, GS, SUFFIX)                            \
  template [[host_name("polar_qmv_" #BITS "bit_gs" #GS "_" SUFFIX)]]        \
  [[kernel]] void polar_qmv<T_TYPE, BITS, GS>(                              \
      device const uint32_t*, device const half*, device const half*,        \
      device const T_TYPE*, device T_TYPE*, constant uint&, constant uint&,  \
      constant uint&, uint, uint);                                           \
  template [[host_name("polar_gather_qmv_" #BITS "bit_gs" #GS "_" SUFFIX)]] \
  [[kernel]] void polar_gather_qmv<T_TYPE, BITS, GS>(                       \
      device const uint32_t*, device const half*, device const half*,        \
      device const T_TYPE*, device const uint32_t*, device T_TYPE*,          \
      constant uint&, constant uint&, constant uint&, constant uint&,        \
      uint, uint);                                                           \
  template [[host_name("polar_multi_gather_qmv_" #BITS "bit_gs" #GS "_" SUFFIX)]] \
  [[kernel]] void polar_multi_gather_qmv<T_TYPE, BITS, GS>(                 \
      device const uint32_t*, device const half*, device const half*,        \
      device const T_TYPE*, device const uint32_t*, device T_TYPE*,          \
      constant uint&, constant uint&, constant uint&, constant uint&,        \
      constant uint&, uint, uint);

// float16
INSTANTIATE_QMV(half, 2, 32, "float16")
INSTANTIATE_QMV(half, 2, 64, "float16")
INSTANTIATE_QMV(half, 3, 32, "float16")
INSTANTIATE_QMV(half, 3, 64, "float16")
INSTANTIATE_QMV(half, 4, 32, "float16")
INSTANTIATE_QMV(half, 4, 64, "float16")

// bfloat16
INSTANTIATE_QMV(bfloat, 2, 32, "bfloat16")
INSTANTIATE_QMV(bfloat, 2, 64, "bfloat16")
INSTANTIATE_QMV(bfloat, 3, 32, "bfloat16")
INSTANTIATE_QMV(bfloat, 3, 64, "bfloat16")
INSTANTIATE_QMV(bfloat, 4, 32, "bfloat16")
INSTANTIATE_QMV(bfloat, 4, 64, "bfloat16")

// float32
INSTANTIATE_QMV(float, 2, 32, "float32")
INSTANTIATE_QMV(float, 2, 64, "float32")
INSTANTIATE_QMV(float, 3, 32, "float32")
INSTANTIATE_QMV(float, 3, 64, "float32")
INSTANTIATE_QMV(float, 4, 32, "float32")
INSTANTIATE_QMV(float, 4, 64, "float32")
