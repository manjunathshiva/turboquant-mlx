# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
