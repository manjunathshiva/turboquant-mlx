# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-06-12

### Added — 16 GB Mac support for VLM/diffusion models

- **`convert_vlm --protect-expert-layers 0,1,2,27,28,29 --protect-bits 3`** —
  expert layer protection: keep the listed layers' experts at 3-bit while the
  rest drop to `--bits` (e.g. 2-bit). On DiffusionGemma-26B-A4B, unprotected
  2-bit experts break arithmetic entirely (17×23 → "3"); protecting the
  first/last three layers restores it (391, correct multi-step chains) for
  +0.2 GB.
- **`convert_vlm --quantize-extras`** — quantizes the remaining bf16 modules
  (embeddings, dense per-layer MLP, vision tower) to 8-bit affine; routers
  and self-conditioning stay full precision. `load_turboquant_vlm` applies
  the matching `nn.quantize` on load (affine modules are recognized by
  having `.scales` without `.codebook`). Mini build of DiffusionGemma:
  **9.79 GB** on disk, ~12.4 GB peak at `--max-tokens 120` — vs 13.84 GB /
  OOM before.
- **`generate_vlm --max-denoising-steps / --max-canvas-length`** — speed and
  memory knobs for diffusion sampling (quantized models need more denoise
  steps to converge; capping trades quality for speed).

### Changed

- `_prepare_polar_layers` now infers each layer's bit width from its saved
  **codebook size** (2^bits entries) instead of re-deriving it from path
  rules — required for per-layer bit assignments (layer protection) and
  immune to config/path drift.

## [0.7.1] - 2026-06-12

### Added

- **`kernels/polar_gather_qmm.py`** — tiled batched gather-GEMM that runs
  expert-routed matmuls **directly on packed TurboQuant weights** (no fp16
  materialization). Sorted routings are grouped host-side into single-expert
  16-token tiles (vectorized mx ops, no sync); each threadgroup stages the
  x tile in threadgroup memory and unpacks each weight word once for 16
  fully-unrolled per-token FMAs. ~5.8x faster than the per-row gather kernel
  at diffusion-canvas scale (gate_up 5.7 ms vs 33 ms at 2048 routings).

### Changed

- `PolarQuantizedSwitchLinear` large-batch routing now prefers
  `polar_gather_qmm` for sorted routings (any expert count — nothing is
  materialized), keeping fused dequant + `mx.gather_mm` only as the unsorted
  fallback under the 2 GiB cap. End-to-end on DiffusionGemma-26B-A4B tq3-g32:
  **1.6 -> 4.6 tok/s** (2.9x), peak memory **23.3 -> 17.8 GB**.

## [0.7.0] - 2026-06-12

### Added

- **mlx-vlm architecture support (multimodal / diffusion LLMs)** — first
  target: Google **DiffusionGemma-26B-A4B** (`model_type: diffusion_gemma`,
  block-diffusion MoE, 25.2B total / 3.8B active). New optional dependency:
  `pip install "turboquant-mlx-full[vlm]"` (mlx-vlm >= 0.6.3).
  - `python -m turboquant_mlx.convert_vlm` — converts architectures that live
    in mlx-vlm rather than mlx-lm. Reuses the model-agnostic
    `turboquant_quantize` core; applies per-arch full-precision skips
    (`integration/vlm.py::VLM_SKIP_PATTERNS`): vision/audio towers, MoE
    routers, and for `diffusion_gemma` the dense per-layer MLP and
    self-conditioning block (quant-sensitive per the upstream
    `quant_predicate`).
  - `python -m turboquant_mlx.generate_vlm` — loads TurboQuant checkpoints
    through mlx-vlm (`integration/vlm.py::load_turboquant_vlm`; the stock
    mlx-vlm loader would mis-apply affine `nn.quantize`) and runs mlx-vlm's
    generation dispatch, including the block-diffusion denoising sampler.
  - `diffusion_gemma` rotation-config registry entry.
- **`kernels/polar_dequant_experts.py`** — fused Metal kernel that
  dequantizes all experts of a `PolarQuantizedSwitchLinear` in one pass
  (unpack + codebook + group scales). Bit-identical to the previous multi-op
  Python path and ~11x faster at MoE shapes; now backs `_dequantize_all`.

### Changed

- `PolarQuantizedSwitchLinear` routes **large batched expert calls** (>= 512
  token-expert routings, e.g. diffusion canvas forwards and large sorted
  prefills) through fused dequant + `mx.gather_mm` instead of the per-row
  gather kernel, which re-reads activations per output row at that scale
  (~2x end-to-end on DiffusionGemma denoising). Guarded by a 2 GiB cap on
  the materialized expert tensor so many-large-expert models (e.g.
  512-expert LatentMoE) keep the memory-safe gather kernels (issue #1).

## [0.6.2] - 2026-05-31

### Fixed

- **Convert no longer hits the Metal GPU watchdog when quantizing a lazily-mmap'd
  checkpoint off slow storage (e.g. a USB HDD).** Symptom: `convert` (in-memory
  *and* `--streaming`) aborted with
  `[METAL] ... kIOGPUCommandBufferCallbackErrorTimeout` shortly after start.
  Root cause: MLX fused the multi-second weight *read* from the slow disk into
  the same GPU command buffer as the rotate/quantize, and the watchdog kills any
  command buffer that stalls on I/O that long — it was **not** tensor size or a
  slow kernel (rotating a resident 1.3B `lm_head` is ~0.2 s). Fix in
  `core/polar_quantize.py`: (1) `mx.eval(weight)` at the top of
  `polar_quantize_weight` forces the disk read as its own step before any GPU
  compute, keeping the kernels pure-compute; (2) the output-row axis is quantized
  in ≤64M-element blocks (`_MAX_QUANT_BLOCK_ELEMS`) as a secondary bound on
  per-command-buffer work. **Bit-identical** to the previous single-pass
  quantization (each output row quantizes independently; validated 4-block vs
  1-block exact match). Fast storage never tripped this, which is why prior
  large-model converts succeeded. Helps any dense / big-vocab model converted
  off slow storage (validated converting Qwen3.6-27B tq3 g32).

## [0.6.1] - 2026-05-30

### Added

- **Read-coalescing for the expert-streaming reader (default-on).** When a token
  routes to several experts that sit at *contiguous* positions in a shard, the
  streaming cache now merges them into a single `os.pread` (`read_range_np` +
  `_load_coalesced`) instead of one syscall per expert. Bit-identical to the
  per-expert path; cuts read syscalls by up to ~22% and lifts throughput ~5% in
  the disk-bound regime (low cache budget on fast storage), ~0 when cache-warm.
  No flag — every streaming run benefits automatically.
- **Cross-layer speculative prefetch — `--prefetch-ahead N` (opt-in, default 0).**
  Predicts an upcoming layer's experts from the previous token's routing and
  reads them on a background thread into a staging buffer, claiming them on the
  main thread (MLX is never touched off-thread) → bit-identical. Helps only when
  the storage has spare bandwidth (~+6% on fast NVMe); it **self-disables** after
  a warmup window if the measured rescue rate shows the drive is bandwidth-bound
  (e.g. a saturated USB bus), so it is safe to leave on.
- **Hot-expert pinning + calibration tooling — `--pin-file` (experimental).**
  `stream/calibrate_experts.py` records a routing trace and emits `pin.json`
  (hottest experts) and `perm.json` (co-activation order); `--pin-file` keeps the
  hot set permanently resident. **Not recommended as a default** — measured
  net-negative vs pure LRU on a 122B (static pinning removes LRU's adaptivity).
  Shipped as opt-in tooling for experimentation only.
- **Co-activation on-disk relayout — `stream/repack_experts.py` (optional).**
  Reorders the expert axis of `switch_mlp.{gate,up,down}_proj.{weight,scales}`
  and the matching router rows by co-activation order so co-selected experts land
  adjacent (feeding the coalescing reader). Pure relabeling → byte-identical
  output; benefit is fast-storage + low-budget only.

> **Finding (2026-05-30):** for MoE expert streaming the LRU + parallel-read cache
> is already near-optimal on the policy axis; the dominant limiter is raw disk
> bandwidth (a USB SSD bus saturates at ~0.6 GB/s under the 8-worker read pool).
> The genuine levers are hardware (Thunderbolt/NVMe) and fewer bytes/token
> (hybrid models, larger cache budget), not the read algorithm. These knobs are
> the squeeze that remains once the bus is the wall.

## [0.6.0] - 2026-05-28

### Added

- **Memory-bounded (streaming) converter — `convert --streaming`.** The default
  converter materializes the entire quantized model in RAM before saving, which
  caps conversion at ~130B params on a 64 GB Mac. The new path writes each
  quantized layer to a safetensors shard and frees it during the quantization
  loop (via a `turboquant_quantize(on_quantized=…)` callback + a
  `StreamingShardWriter`), keeping peak memory to ~one shard (5 GB) plus the
  single layer being processed — so 200B+ MoEs (Qwen3-235B, DeepSeek-V3) convert
  on a 64 GB machine. Output is **byte-identical** to the in-memory converter
  (verified on DeepSeek-V2-Lite-Chat: 1181/1181 tensors at fixed `PYTHONHASHSEED`).
- **DeepSeek (MLA + MoE) conversion + streaming support.** Added a rotation
  config for the DeepSeek Multi-head Latent Attention + SwitchGLU-MoE family
  (`deepseek_v2`, `deepseek_v3`, `deepseek_v32`). MLA's input projections
  (`q_proj`/`q_a_proj`/`kv_a_proj_with_mqa`) fuse into `input_layernorm`; the
  `q_b_proj`/`kv_b_proj` (nested-norm inputs) and `o_proj` use online rotation;
  the MoE/MLP fuses exactly like `qwen3_5_moe`. Validated end-to-end on
  DeepSeek-V2-Lite-Chat: converted to tq3 (6.6 GB) and coherent both resident
  (~84 tok/s) and via expert streaming. V3/V3.2 reuse the same config (untested,
  pending a conversion). The streaming loader (`turboquant_mlx.stream`) now
  auto-detects the layer-key prefix, supporting both the multimodal
  `language_model.model.layers` layout (qwen3_5_moe) and the text-only
  `model.model.layers` layout (DeepSeek).
- **`qwen3_moe` rotation config registered** (standard attention + SwitchGLU, =
  `MOE_LLAMA_CONFIG`). Validated on Qwen3-235B-A22B-Instruct-2507: converted to
  a hybrid **tq3a-tq2e g32** build (3-bit attention, 2-bit experts, full-precision
  routers — 70.51 GB across 15 shards) on a 16 GB Mac mini via `--streaming`, and
  generates coherent text through expert streaming. First `qwen3_moe` validation
  and confirmation that 2-bit experts hold at Qwen3's 128-expert / top-8 routing.

## [0.5.0] - 2026-05-26

### Added

- **Parallel expert prefetch for streaming MoE (`turboquant_mlx.stream`).**
  The experts missing for a layer are now `pread` concurrently on a thread
  pool instead of one at a time. Because `pread` is positional and releases
  the GIL, the per-layer disk stall drops from the sum of the slice reads to
  roughly the slowest one; MLX array construction and `eval` stay on the
  calling thread, so output is **bit-identical** to the serial path. Controlled
  by `--prefetch-workers` (default `8`; `1` restores the old serial behavior).
  Measured on Qwen3.6-35B-A3B-tq3-g32 at a 1 GB cache budget: decode **3.2 →
  6.0 tok/s (~1.9×)** and prefill **5.4 → 13.9 tok/s (~2.6×)**, same 3.48 GB
  peak, identical generated text.

### Notes

- Frequency-based hot-expert *pinning* was prototyped alongside prefetch and
  **rejected**: reserving budget for a frozen "hot" set consistently lowered
  the cache hit rate versus a plain adaptive LRU in single-stream decode
  (49.0% → 38.8% at the same 1 GB budget), because the tight-budget streaming
  regime is exactly where starving the LRU hurts most. Not shipped.

## [0.4.1] - 2026-05-25

### Fixed

- **Packaging: `turboquant_mlx.stream` was omitted from the 0.4.0
  distribution.** The `[tool.setuptools] packages` list enumerates packages
  explicitly and the new `stream` subpackage was not added, so 0.4.0 shipped
  without the streaming code (`import turboquant_mlx.stream` failed after a
  PyPI install; the feature was only reachable from a source checkout). Added
  `turboquant_mlx.stream` to the packages list. Expert streaming now works
  from `pip install turboquant-mlx-full`.

## [0.4.0] - 2026-05-25

### Added

- **Expert streaming for MoE models (`turboquant_mlx.stream`).** Run MoE
  checkpoints whose weights exceed available RAM by paging only the
  router-selected experts from disk per token (LRU-cached), keeping the full
  `(num_experts, ...)` expert tensors out of memory. Output is bit-identical
  to the fully-resident model. New CLI:
  `python -m turboquant_mlx.stream.stream_generate --model <repo> --cache-budget-gb <GB>`.
  - Validated on a 16 GB Mac mini running the ~16 GB
    `Qwen3.6-35B-A3B-tq3-g32` (`qwen3_5_moe`, 256 experts): 3.9 GB peak RSS
    at `--cache-budget-gb 2` (~3 tok/s) up to 9.4 GB / ~4.5 tok/s at
    `--cache-budget-gb 8`. Disk read is the throughput limiter; a larger cache
    budget raises the expert hit-rate and cuts per-token SSD traffic.
  - Uses `os.pread` + macOS `F_NOCACHE` so streaming tens of GB of expert
    slices doesn't balloon resident page cache; RSS tracks MLX managed memory.

### Fixed

- `stream_generate` reported throughput by dividing `--max-tokens` by
  wall-time, overstating tok/s whenever the model stopped at EOS before the
  cap. It now counts the tokens actually generated.

### Notes

- Streaming currently targets the `qwen3_5_moe` expert layout
  (`language_model.model.layers[*].mlp.switch_mlp.{gate,up,down}_proj`);
  generalizing to other MoE architectures is future work.

## [0.3.0] - 2026-05-03

### Changed

- **License relicensed from MIT to Apache-2.0.** The MIT license that covered
  versions 0.1.x and 0.2.0 is preserved verbatim in `LICENSE-MIT` for
  reference; users of those versions retain their MIT rights. All new
  contributions and releases (0.3.0 and later) are governed by `LICENSE`
  (Apache-2.0).
- `pyproject.toml` `license` field updated to `Apache-2.0`; classifier
  updated accordingly. Author field corrected to legal name
  (`Manjunath Janardhan`).

### Added

- `NOTICE` file with copyright + attribution boilerplate (Apache-2.0
  requires this to propagate into derivative works).
- `CITATION.cff` so GitHub renders a "Cite this repository" widget and
  academic users have a defined citation form.
- `CONTRIBUTING.md` with the Developer Certificate of Origin (DCO) and
  `Signed-off-by` instructions for contributors.

### Notes

- Why the relicense: Apache-2.0 adds an explicit patent grant, mandates
  NOTICE propagation in derivative works, and is the standard open-source
  license for ML/quantization tooling. MIT remains valid for all 0.1.x and
  0.2.0 releases.

## [0.2.0] - 2026-04-30

KV cache v0.2: mixed K/V bits (`--kv-k-bits`/`--kv-v-bits`),
attention-sink protection (`--kv-min-tokens`), per-head_dim PPL harness,
production CLI for `turboquant-generate` and `turboquant-serve`. Validated
on GPT-OSS-20B/120B and Qwen3.5-122B.

## [0.1.6] - 2026-04-12

Hybrid quantization (`--attn-bits`/`--mlp-bits`) targeting 48 GB Apple
Silicon Macs. Long-context Metal kernel fixes.

## Earlier versions

See `git log` for the full history of versions 0.1.0 through 0.1.5.
