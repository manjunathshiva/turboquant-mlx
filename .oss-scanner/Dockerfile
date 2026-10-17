# OSS Scanner build: https://github.com/anthropics/oss-scanner
# The scan runs on Linux with no GPU and no network. MLX ships CPU-only Linux
# wheels (`mlx[cpu]`), so the Python package, converters, loaders, servers and
# stream/ readers run here; the Metal kernels (kernels/) cannot execute, so the
# tests that dispatch them are left out below. No model weights are fetched:
# the tests build tiny synthetic models in a temp dir.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential: miniaudio (via mlx-vlm) has no aarch64 wheel and builds from
# source; libgl1/libglib2.0-0: opencv-python (via mlx-vlm) needs them to import.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Editable install, as CI does (the repo root IS the turboquant_mlx package).
# [vlm] for serve_vlm/convert_vlm, [kimi] for the Kimi K3 tokenizer; [eval] and
# [dev] are left out. Pins are the versions of the last green CI run (mlx-vlm:
# the version developed against); bump them together with CI.
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[kimi,vlm]" \
         "mlx[cpu]==0.32.2" mlx-lm==0.31.3 transformers==5.17.0 mlx-vlm==0.6.14 pytest \
    && python -c "import turboquant_mlx, mlx.core as mx; print(turboquant_mlx.__version__, mx.default_device())"

# Tests that need Metal (mx.fast.metal_kernel raises "No Metal back-end" on
# Linux, or compare bit-exactly against the Metal path). Run the rest with:
#   python -m pytest tests/ -k "not requires_model" \
#     --deselect tests/test_core.py::test_qjl_unbiasedness $TQ_METAL_ONLY_TESTS
ENV TQ_METAL_ONLY_TESTS="\
--ignore=tests/test_kernel_determinism.py \
--ignore=tests/test_polar_dequant_experts.py \
--ignore=tests/test_polar_gather_qmm.py \
--ignore=tests/test_polar_qmm.py \
--ignore=tests/test_switch_unsorted_batch.py \
--deselect=tests/test_core.py::test_metal_kernel_correctness \
--deselect=tests/test_expert_layer_tiers.py::test_fresh_load_matches_converted_outputs \
--deselect=tests/test_flag_effects.py::test_mixed_group_sizes_survive_a_save_load_round_trip \
--deselect=tests/test_fused_kv_attend.py::test_fused_matches_dequant_sdpa \
--deselect=tests/test_fused_kv_attend.py::test_fused_respects_model_scale \
--deselect=tests/test_layer_protection.py::test_fresh_load_matches_converted_outputs \
--deselect=tests/test_ngram_offload.py::test_quantized_rows_match_quantized_embedding \
--deselect=tests/test_ngram_offload.py::test_lookup_handles_repeats_and_every_shard \
--deselect=tests/test_ngram_offload.py::test_offloaded_model_is_bit_identical \
--deselect=tests/test_qwen3_5_vlm.py::test_the_fused_kernel_bound_is_what_makes_the_skip_load_bearing \
--deselect=tests/test_trit_packing.py::test_dequant_experts_trit_matches_reference \
--deselect=tests/test_trit_packing.py::test_gather_qmm_trit_matches_reference \
--deselect=tests/test_trit_packing.py::test_gather_qmv_trit_matches_reference \
--deselect=tests/test_trit_packing.py::test_multi_gather_qmv_trit_matches_reference \
--deselect=tests/test_trit_packing.py::test_quantize_emits_trit_format \
--deselect=tests/test_trit_packing.py::test_switch_layer_trit_dispatch"

# Cheap check that the image works; a failure warns, it does not fail the build.
RUN timeout 900 python -m pytest tests/ -q -p no:cacheprovider -k "not requires_model" \
      --deselect tests/test_core.py::test_qjl_unbiasedness $TQ_METAL_ONLY_TESTS \
    || echo "WARNING: pytest reported failures"
