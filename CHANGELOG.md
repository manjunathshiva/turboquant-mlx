# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.25.0] - 2026-08-18

A loader fix, and a negative result recorded with its evidence.

Affine-quantized embeddings and `lm_head` were dropped on load without a word,
so a VLM checkpoint opened through the *text* loader returned nonsense instead
of refusing. Both loaders now recover those modules from the tensor shapes, and
a missed one raises rather than loading garbage.

The multi-token-prediction head that ships with Qwen3.5/3.8 can now be preserved
and is fully implemented — and runs at 0.52x plain greedy, so nothing enables it.

### Fixed

- **Affine-quantized embeddings and `lm_head` were silently discarded on load.**
  TurboQuant leaves the token embedding and `lm_head` to MLX's affine quantizer,
  so they arrive as `weight`/`scales`/`biases` with no `.codebook`.
  `_prepare_polar_layers` keys off `.codebook` and so never saw them: the freshly
  built model still held a plain `nn.Embedding`/`nn.Linear` there, and
  `load_weights(strict=False)` dropped their scales and biases without a word.

  Nothing errored. The model returned nonsense — on Qwen3.8-27B,
  `embed_tokens(ids)` came back `(1, T, 1280)` uint32 instead of `(1, T, 5120)`
  floats. It bites when a VLM checkpoint is opened through the **text** loader
  (`turboquant_mlx.generate`), which mlx-lm accepts because it knows `qwen3_5`
  as a text architecture — which is exactly why it loaded and was wrong instead
  of refusing. Muse Glimmer was luckier: mlx-lm has no `muse_glimmer` class at
  all, so it failed loudly.

  `_prepare_affine_extras` now converts those modules before the load, with both
  parameters recovered exactly from the tensor shapes — the plain module's
  `weight.shape[-1]` is the unpacked width, so `group_size = width /
  scales.shape[-1]` and `bits = packed_words * 32 / width`.
  `_assert_no_orphan_quant_params` then refuses to load at all if any `.scales`
  still has nowhere to go. `strict=False` cannot simply be dropped instead,
  because the checkpoint legitimately carries polar keys the base modules do not
  declare, so a missed module has to be caught explicitly.

- **The VLM loader no longer gates affine preparation on a config key.**
  `load_turboquant_vlm` read `quantization.affine_extras` and skipped preparation
  entirely when it was absent, which would have dropped every scale and bias in
  silence. Nothing shipped could reach it — `convert_vlm` writes that key in the
  same branch that quantizes, so the key and the tensors have always travelled
  together (verified across all five local builds) — so this removes the coupling
  rather than fixing a live failure. Both loaders now share the one shape-based
  mechanism, which cannot be skipped by a missing key, and the VLM path gains the
  orphan guard it never had. Verified to agree with the config it replaces on
  Qwen3.8-27B `tq4` (gs64/8-bit) and `tq3-x4` (gs64/4-bit), 86 modules each.

### Added

- **`convert --keep-mtp` preserves the Qwen3.5/3.8 multi-token-prediction head**,
  and `turboquant_mlx.mtp` implements self-speculative decoding on it. Both are
  opt-in and **nothing enables them**: mlx-lm discards the head in `sanitize()`
  and implements MTP nowhere, transformers declares
  `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, so there was no reference
  implementation to copy. The head is copied unquantized at its source dtype
  (810 MiB on Qwen3.8-27B, ~1.9% of the bf16 model).

  The decode loop is bit-identical to greedy and runs at **0.52x** its speed. The
  cost is structural: the Gated-DeltaNet snapshot is taken before both verify
  positions, so a rejection must discard both — trimming only the rejected one
  desyncs the 16 KV layers from the 48 recurrent ones — and replay the one real
  token. That puts the ceiling at `(1+p)/(2-p)`: 0.94x at the measured **45.7%**
  acceptance, and it needs p >= 0.80 to reach 1.5x. Removing the replay needs an
  invertible Gated-DeltaNet update or a cheap mid-sequence checkpoint; both are
  research. Recorded rather than shipped.

  Two facts the checkpoint does not determine, both settled by measurement:
  `concat_order` is `embedding_first` (`hidden_first` scores exactly 0.00%, so
  DeepSeek-V3's `[hidden ; embedding]` convention is wrong here), and every 1-D
  tensor needs the +1 norm shift, not just the four suffixes mlx-lm names —
  **69.49%** top-1 agreement against 3.73%, on 295 tokens of prose.

  The head is stored under `tq_speculator.` rather than `mtp.`, which is
  load-bearing: `sanitize()` keys its +1 norm shift off `any("mtp." in k)`, and a
  converted checkpoint's norms are *already* shifted, so any key containing that
  substring re-fires the predicate and shifts every norm a second time.

- **`benchmarks/eval_vlm_perplexity.py`** — WikiText-2 perplexity for any mlx-vlm
  checkpoint, TurboQuant or affine. `evaluate.py` cannot do this: it loads through
  `mlx_lm.utils.load` and re-quantizes in-process, so it reaches neither a VLM nor
  a checkpoint that arrived pre-quantized from the Hub. Teacher-forced, so no
  sampler or seed is involved, and it *asserts* both models tokenize the corpus
  identically rather than silently reporting incomparable numbers.

- **`benchmarks/eval_vlm_vision.py`** — four drawn-to-order images (OCR, counting,
  chart reading, spatial relations) with ground truth we set, so scoring is a
  substring match rather than a judgement. Picks the engine from the checkpoint:
  TurboQuant builds go through `generate_vlm`, everything else through
  `mlx_vlm.generate`.

### Documentation

- **Qwen3.8-27B's 16 GB claim is now measured on a 16 GB M4 Mac mini** rather than
  reasoned from peak-under-cap on a 64 GB M4 Max. macOS 26.5.2 at
  `iogpu.wired_limit_mb=14336` and `--prefill-step-size 256`: peaks 12.95-13.61
  GiB from 220 to 5,017 tokens, vision correct at two image sizes, 3.7 tok/s
  decode. The peaks reproduced identically across two sweeps and land 0.09-0.27
  GiB *below* the same prompts on the M4 Max, so the larger machine was the
  pessimistic estimator. The README's summary table drops the unpublished
  `tq3-g64` row for the shipped `tq3-mini-g64` (11.55 GiB) and adds
  mlx-community's affine 4-bit for a like-for-like comparison; harness landed as
  `benchmarks/bench_mini_qwen38.py`.

- **The README recommends the affine 4-bit build for Qwen3.8-27B**, on the
  numbers: 29.7 against 11.3 tok/s decode at a lower peak (16.16 against 17.80
  GiB), for 0.3% perplexity (7.2685 against 7.2496 at 16K). TurboQuant's one
  structural edge here is storage — the Hadamard rotation symmetrizes each group
  so there is no zero-point to keep, 4.251 against 4.500 bits/weight. Two
  cautions are recorded with it, because both would otherwise be misread: the
  3/4-against-4/4 vision split is ONE string (a five-string probe put affine at
  4/5, so it is not an OCR weakness), and agentic wall-clock is not a speed
  metric — the faster-decoding build finished slower by spending more tool turns.

- **Corrected the reason for the `[vlm]` floor of `mlx-vlm>=0.6.12`.** The
  comment claimed 0.6.12 was the first release carrying the Muse Glimmer model
  classes, and that `[vlm]` could not load `muse_glimmer` below it. Both halves
  are wrong: 0.6.11 carries `mlx_vlm.models.muse_glimmer` and loads it fine.
  What 0.6.12 actually shipped (Blaizzy/mlx-vlm#1838) is the *norm hoist* — the
  RMS norm moved out of `NormedEmbedding` into `TextModel.embed_norm` — which is
  why `_patch_muse_glimmer_normed_embedding` is a no-op there. The constraint is
  unchanged; only its justification was wrong.

## [0.24.0] - 2026-08-15

Qwen3.5-family VLM support (`model_type: qwen3_5`, e.g. Qwen3.8-27B), and the
one thing that stopped those builds being usable.

No architecture work was needed — mlx-lm 0.31.3 and mlx-vlm already carry
`qwen3_5`. The substance is that a 248K-vocabulary `lm_head` must not go through
the polar path, which costs 10.5 GB of runtime peak to save 0.63 GB on disk.

### Fixed

- **`qwen3_5` VLMs keep `lm_head` off the polar path.** Qwen3.5-family
  multimodal models (e.g. Qwen3.8-27B) have a 248,320-token vocabulary, making
  `lm_head` 248320x5120 = 1.271B parameters — the largest matrix in the model.
  `PolarQuantizedLinear` is fused only up to `_QMM_MAX_TOKENS` (256) tokens;
  above that it dequantizes the weight through full-size intermediates. Measured
  on that exact shape at 3-bit/g64: **+0.116 GiB at 256 tokens, +13.141 GiB at
  257**. End to end on a 1342-token prompt the polar build peaks at **33.34 GB
  against 22.85 GB** for the affine one, while saving only 0.63 GB on disk —
  enough to put an otherwise-fine build over the limit on a 36 GB Mac.

  It hides because mlx-vlm's chunked prefill discards the chunk forward's return
  value, so MLX never evaluates the matmul. But chunking only engages *above*
  `prefill_step_size` (default 2048), and a prompt of 257..2048 tokens takes the
  single-shot branch in `generate/ar.py`, which slices `logits[:, -1, :]` and
  does evaluate it. `muse_glimmer` already carried this skip for the same
  reason; `qwen3_5` did not.

### Added

- **`generate_vlm --prefill-step-size`.** The VLM generate path had no way to
  set the prefill chunk size, so the planner could recommend a value that was
  not runnable there (`turboquant-plan` suggests one whenever the default would
  not fit). It is the main memory knob for long prompts: on Qwen3.8-27B with a
  1342-token prompt, `--prefill-step-size 256` measures **16.77 GB peak against
  21.28 GB** at the 2048 default, for 2.1x slower prefill. mlx-vlm only chunks
  *above* the step size, so prompts shorter than it run unchunked.

- **Qwen3.8-27B validated** (dense 27.8B hybrid Gated-DeltaNet VLM, 48 linear +
  16 full attention layers). `tq4-g64` = 15.15 GiB, `tq3-g64` = 12.88 GiB; both
  pass an Opencode agentic task and the vision path. `tq4` is both faster and
  better than `tq3` here. No architecture work was needed — mlx-lm 0.31.3
  already carries `qwen3_5`. Verified on **both mlx-vlm 0.6.3 and 0.6.13**
  (identical peaks and identical vision results), so the `[vlm]` floor of
  0.6.12 reproduces the measurements.

## [0.23.0] - 2026-08-15

A correctness release. Every item below is a setting that was accepted,
validated, stored in `config.json` and printed at conversion time — and then
did not reach the quantizer. Two produced word salad when used; one made a
layer 10.6x worse than not using it at all.

**No shipped model is affected and no default output changes.** Verified by
converting Llama-3.2-1B before and after under a fixed `PYTHONHASHSEED`: the
`model.safetensors` are byte-identical (SHA256 `99def7a6…`). Additionally
validated on Laguna-S-2.1 tqTe (256-expert MoE + ternary), Muse Glimmer 30B
tq4 (VLM), and Laguna under expert streaming.

The last entry is what keeps this from recurring: a mechanical audit of the
flag surface, which is what found the `--mlp-group-size` bug.

### Fixed

- **The dense group-size compatibility check validated the wrong value.** It
  tested `input_dims % tq_config.group_size` while the layer is quantized at
  `group_size_for_path(path)`, so with e.g. `--group-size 64
  --mlp-group-size 128` an MLP whose `input_dims` divides 64 but not 128
  passed the check and then raised inside `polar_quantize_weight`. The MoE
  branch already checked the per-path value; the dense branch now does too.

- **`--mlp-group-size` silently did nothing on dense models.** Found by a
  systematic audit of every converter flag (see below). `bits_for_path`
  applies `--mlp-bits` to dense MLP linears, but the dense branch of the
  quantizer asked for `tq_config.group_size` directly instead of
  `group_size_for_path`, so its documented sibling `--mlp-group-size` reached
  MoE experts only. The two are documented as a matched pair; they now behave
  like one. No shipped model is affected — the loader read the base group
  size too, so convert and load agreed and no checkpoint was mis-scaled, and
  no local or published model sets the flag. Default conversions remain
  byte-identical (Llama-3.2-1B, SHA256 `99def7a6…`).

- **The loader now recovers each layer's group size from the saved scales
  rather than re-deriving it from the config rules.** `scales` are
  `(..., output_dims, n_groups)` with `n_groups = input_dims // group_size`,
  so the on-disk tensor states the group size exactly. Re-deriving it meant
  any change to the path rules would silently mis-scale every older
  checkpoint — the failure mode `CLAUDE.md` records as having already caused
  one real bug, and the reason the fix above would otherwise have been unsafe
  to ship. This also makes per-layer group sizes work end to end: a model
  converted with `--group-size 64 --mlp-group-size 32` now loads with
  attention at 64 and MLP at 32 and generates correctly.

### Added

- **`tests/test_flag_effects.py` — a mechanical audit of the whole flag
  surface**, in two independent layers, because a flag can break at either.
  A *static* pass asserts every `--flag` declared by an entry point is
  actually read off the `parse_args()` namespace (catches "declared and
  forgotten"), and a
  *behavioural* pass asserts each config field actually changes the quantized
  bytes (catches "plumbed all the way through, then ignored" — exactly how
  `--rotation` passed every review for the project's entire history).
  Documented no-ops are asserted to stay no-ops, so a flag leaking into
  layers it should not touch fails too.

  The fingerprint hashes parameter **bytes**, not just shapes. Shapes stay
  correct no matter which domain the numbers live in, which is precisely why
  the existing suite missed the rotation bugs.

  15 effect cases across dense and MoE, covering `bits`, `group_size`,
  `rotation`, `rotation_seed`, `attn_bits`, `mlp_bits`, `mlp_group_size`,
  `use_qjl`, `ternary_experts` and `expert_down_bits`, plus a save/load
  round-trip pinning the per-layer group-size derivation. All pass.

  The fingerprint hashes each parameter in its **own** dtype rather than
  widening to float32. Packed weights are `uint32` and reach ~1.07e9 in
  practice, 64x past float32's exact-integer limit of 2^24, so widening would
  round distinct packings onto the same value and report "unchanged" for
  genuinely different weights — a false pass in exactly the direction that
  hides the bug being hunted.

- **`--rotation` was accepted, stored, printed — and ignored.** All three
  values produced identical output. `polar_quantize_weight` had no rotation
  parameter at all, so the randomized Hadamard was applied unconditionally;
  `TurboQuantConfig.rotation` was validated and round-tripped through
  `config.json` without ever reaching the quantizer. Measured on
  Llama-3.2-1B under a fixed `PYTHONHASHSEED`, `--rotation hadamard`,
  `none`, and `blockwise_hadamard` all produced `model.safetensors` with
  SHA256 `99def7a6…`. This predates the `fuse_rotations` removal below —
  verified by converting at the parent commit, which gives the same hash.

  `rotation="none"` now genuinely disables the rotation, on **both** sides:
  weights are quantized unrotated and the layer skips `rotate_input`. One
  config field drives both (`quantize_model.py` on the convert side,
  `generate.py::_prepare_polar_layers` on the load side), so they cannot
  drift apart — which is exactly how `fuse_rotations` produced noise. The
  streaming loader inherits `_needs_rotation` from the resident layer and
  the VLM loader shares `_prepare_polar_layers`, so every load path agrees.

  The default path is untouched: `--rotation hadamard` still hashes to
  `99def7a6…`. Verified on real models beyond the dense case: **Laguna-S-2.1
  tqTe** (256-expert MoE + ternary/trit, 526 quantized layers) and **Muse
  Glimmer 30B tq4** (VLM, bfloat16 source) both load and generate normally,
  covering `PolarQuantizedSwitchLinear`, the trit decode path, and the VLM
  loader. A Qwen3-0.6B tq3 build produces output identical to `main`.

  **Legacy guard.** Because the flag was ignored, a model converted *before*
  this release with `--rotation none` records `"rotation": "none"` but has
  ROTATED weights — trusting the config alone would skip the input rotation
  and produce noise. The loader therefore checks the saved `signs`, which are
  the ground truth: the genuinely unrotated path writes all-ones, the old
  path always wrote randomized ±1. A negative entry anywhere means the
  weights really were rotated, so rotation stays on. Verified by forging such
  a model (rotated weights, config edited to `"none"`): it still generates
  coherently.

  As an ablation it confirms the rotation is load-bearing. Reconstruction
  error on real Llama-3.2-1B weights at 3-bit/g64 is a **uniform 0.1828**
  with rotation on — the same value for every layer, which is the
  Gaussianization signature: after the Hadamard each group looks N(0,1), so
  the Lloyd-Max codebook hits its designed error regardless of the layer's
  native distribution. With rotation off it rises and scatters,
  0.1862–0.1978 depending on the layer. End to end at 3-bit that compounds
  into word salad; at 4-bit the same build is degraded but partly
  grammatical. (A converter/loader mismatch would destroy both bit-widths
  equally — that monotonic curve is what distinguishes real quality loss
  from a wiring bug.)

  `"blockwise_hadamard"` is now documented as an **alias** for `"hadamard"`,
  not a separate mode. Blockwise was never a choice: `rotate_weight` picks
  the largest Hadamard-compatible block dividing `input_dims` and blocks
  automatically when the full dimension is not compatible. The alias is
  still accepted so older `config.json` files keep loading.

- **QJL measured its residual in the wrong domain when rotation was
  disabled.** `from_linear` built the correction target with `rotate_weight`
  unconditionally, but that applies the Hadamard even when the signs are
  all-ones — so with `rotation="none"` the target was rotated while the
  dequantized weights were not, making the residual `rotated - unrotated`.
  At inference `qjl_correct` is applied to the same `x` the matmul saw, so
  the "correction" was noise: measured relative error on a 3-bit layer went
  from 0.1712 without QJL to **1.8200** with it, a 10.6x degradation.
  The target now follows `needs_rotation`. Pinned by
  `test_qjl_residual_matches_the_packed_domain`, whose invariant is simply
  that QJL must never make a layer worse — in either rotation mode. Only
  reachable via `--use-qjl` combined with `--rotation none`; the shipped
  default leaves QJL off.

- **The fused Metal kernels could not compile against bfloat16
  activations.** Nothing enforced their documented `float16` input
  contract; it was met only as a side effect of `rotate_input` multiplying
  by the float16 `signs` vector, which promoted bf16 activations to
  float32. With rotation disabled the raw bfloat16 reached the kernel and
  Metal failed to build the library outright (`incompatible operand types
  ('bfloat16_t' and 'half')`). Both quantized layers now promote bfloat16
  to float32 explicitly — the same dtype the rotated path already hands the
  kernel. The check is deliberately narrow (`== mx.bfloat16`): after
  `rotate_input` the dtype is float32 or float16 and never bfloat16, so it
  is provably a no-op whenever rotation ran. `StreamingSwitchLinear` carries
  the same guard — it has its own `__call__` with its own rotation branch, so
  fixing only the two resident layers would have left the streaming path
  exposed. Streaming was verified end-to-end on Laguna-S-2.1 tqTe (141 expert
  projections swapped, 2.74 GB resident vs 28.9 GB fully resident, 36.1 GB of
  expert reads, coherent output).

### Removed

- **`--fuse-rotations` / `fuse_rotations`, which produced pure noise whenever
  it was enabled.** The option claimed to fold a layer's Hadamard rotation
  into the preceding RMSNorm weight, letting inference skip the online
  rotation. That is not possible. A norm applies a *diagonal* weight
  (`y = n * w`), and folding the rotation in would require pulling an
  element-wise multiply through a Hadamard transform:

      hadamard(signs * w) * n  ==  hadamard(signs * (n * w))     # false

  A Hadamard mixes across dimensions, so the identity does not hold, and the
  error is not a small approximation — measured cosine similarity between the
  two sides is **+0.18 at dim=64, +0.04 at dim=128, and +0.004 at dim=4096**.
  At model scale the "fused" result is orthogonal to the correct one. A
  converted Llama-3.2-1B emitted textbook word salad (`"followed direct
  Roberts cầuChildren packaged epidemi-k set done Liu visiting..."`) where the
  same build with online rotation was coherent.

  Two further defects sat on top of the broken identity, both of which would
  have had to be fixed even if it had held: the dense path derived the
  rotation seed per *projection* while `q/k/v` share one `input_layernorm`, so
  only `q_proj`'s rotation was ever fused and `k_proj`/`v_proj` silently
  inherited it; and the MoE path set `needs_rotation = False` without fusing
  anything at all. Sharing the seed was tried and confirmed **not** to fix the
  word salad, as the math predicts.

  The escape hatch that let this survive was a unit test asserting only
  `fused_weight.shape == (dim,)`. It is replaced by
  `test_rotation_cannot_fuse_into_norm`, which pins the impossibility from
  both sides: the Hadamard case must stay far from the truth, and the control
  — a diagonal-only sign flip, which *does* commute with the norm weight —
  must fuse exactly. A sign flip alone does not Gaussianize the weights, which
  is the entire purpose of the rotation, so there is no salvageable middle
  ground. Fusing a rotation correctly requires rotating the *residual stream*
  (QuaRot-style: one shared orthogonal `Q` absorbed into the embeddings and
  every output projection, with norm weights folded into the following linear
  first) — a whole-model transform, not a per-layer one.

  **No shipped model is affected and no output changes.** The CLI flag was
  `store_true`, so every documented conversion already used online rotation,
  and `convert_vlm` passed `fuse_rotations=False` explicitly. Verified by
  converting Llama-3.2-1B before and after this change under a fixed
  `PYTHONHASHSEED`: the resulting `model.safetensors` are byte-identical
  (SHA256 `99def7a6…`). Only the now-meaningless `"fuse_rotations": false`
  key disappears from the `quantization` block in `config.json`; older
  `config.json` files carrying it still load unchanged.

  `integration/rotation_configs.py` is kept — it encodes real per-architecture
  structure (which projection reads which norm) and is the exact input a
  correct residual-stream rotation would need — but it is now descriptive
  only, with no runtime consumer.

## [0.22.0] - 2026-08-14

`turboquant-plan` now models prefill the way it actually runs. Two
corrections, both measured rather than reasoned about: a term it charged for
that is never paid, and a term it never charged for that always is. Nothing
outside the planner changes — no conversion, kernel, or serving behaviour is
touched.

### Fixed

- **`turboquant-plan` over-estimated prefill workspace on big-vocab models by
  charging for an lm_head output that is never computed.** The projection
  billed `chunk x vocab x 2` for every prefill chunk. Both engines chunk
  prefill (`mlx_lm.generate_step`, mlx-vlm `generate/ar.py`) and both *discard*
  the chunk forward's return value, evaluating only the KV cache state — and
  MLX is lazy, so an lm_head matmul nobody asks for never runs. Measured on an
  M4 Max at 2048 x 202048, against a floor doing the cache work alone:
  dropping the output costs **0 MB**, while slicing `[:, -1:]` out of it costs
  the full **763 MiB**. Both loops also leave exactly one token for the scoring
  step.

  So the term is real only when the prompt fits in a *single* forward
  (`context <= step`), where the slice happens after the matmul. On
  Muse-Glimmer-30B at 5,068 tokens this cut the projected workspace from 5.21
  GB to 4.39 GB, and it was making `--prefill-step-size` look like it bought
  headroom it does not buy. The estimate is now the max of the widest chunk
  pass and the one-token scoring step, rather than a sum of costs that never
  coexist.

  Also corrected: the chunk can no longer exceed the prompt, so a 900-token
  prompt is billed as a 900-token forward instead of a 2048-token one.

  The field-calibrated mini datapoints are unaffected — they were measured on
  a config with no `vocab_size`, so they never depended on this term.

### Added

- **`turboquant-plan` now models the MoE prefill term**, the last one it left
  out. It is not what the old TODO assumed. Expert *dequantization* is
  essentially never paid during prefill: mlx-lm's SwitchGLU sorts any batch of
  64+ routings, and sorted batches go to `polar_gather_qmm`, which tiles over
  packed weights and materializes nothing. Measured on a 256-expert block at
  chunk 2048: **357.8 MB** peak, against a **933.8 MB** control that does
  dequantize all experts — the 768 MB tensor never appears.

  What is actually there is routing-expanded activations. SwitchGLU fans every
  token out to `top_k` rows before the expert matmuls, so the transient scales
  with `chunk x top_k`, not `chunk`. Measured bytes per routing on one
  gate/up/down block, against `8 x (hidden + moe_intermediate)`:

  | geometry (experts, hidden, moe_inter) | measured | bound |
  |---|---|---|
  | 256, 2048, 768 (Qwen3.6-35B-A3B) | 21,077 | 22,528 |
  | 128, 1024, 2048 (inverted ratio) | 20,564 | 24,576 |
  | 64, 4096, 1024 (wide hidden) | 38,984 | 40,960 |
  | 256, 2048, 512 (narrow expert) | 19,541 | 20,480 |
  | 256, 3072, 1024 (Laguna-S-2.1) | 30,819 | 32,768 |

  The bound holds on all of them with 3–16% slack, and is what the eight live
  routing-expanded buffers cost if the allocator reused none. On Laguna-S-2.1
  at 16K context this adds 0.67 GB to the projection at the default chunk.

  The dequant fallback is still charged where it can genuinely fire: expert
  output dims that `polar_gather_qmm` cannot tile (not a multiple of 64), and
  only below the switch layer's 2 GiB cap, above which it reverts to gather
  kernels that materialize nothing.

  Expert width is read from `moe_intermediate_size`, never `intermediate_size`
  — on Laguna those are 1024 and 12288, a 12x error. Models whose config does
  not state `num_experts_per_tok` get no term rather than a guessed one.

## [0.21.1] - 2026-08-11

Two `turboquant-serve-vlm` fixes found by using it: an agent harness never
received the assistant's final message, and a documented flag was missing from
`--help`.

### Fixed

- **Muse Glimmer returned an empty final message to any tools-enabled client.**
  While streaming, mlx-vlm suppresses content deltas from the moment the tool
  parser's `tool_call_start` appears in the output, and `in_tool_call` has no
  release path — once latched, it stays latched for the rest of the generation.
  ATEM's `tool_call_start` is `to=self<|message|>`, which is exactly the channel
  router Muse Glimmer emits at the start of *every* turn, tool call or not. So
  the latch closed on the first reasoning token of every request that declared
  tools, and nothing the model said afterwards reached the client.

  The asymmetry made it look like a model problem: a plain `curl` with no tools
  answered correctly, tool calls kept working (those are parsed from the full
  output, not the stream), and only prose went missing. In OpenCode the model
  would find and fix the bug, run the tests green, and then say nothing.

  `turboquant-serve-vlm` now maps that trigger to `<atem:function_calls>`, the
  tag that genuinely opens a call — and the same string mlx-vlm uses to detect
  the ATEM format in the first place. Tool-call *parsing* is untouched, since
  `process_tool_calls` reads the marker from the parser itself. Parsers whose
  `tool_call_start` is not a known collision pass through unchanged.

- **`--reasoning-strength` was missing from `--help`.** `--help` falls through
  to mlx-vlm's parser, which owns every other flag and exits before ours is
  mentioned, so a flag the model cards document looked nonexistent.

## [0.21.0] - 2026-08-11

Muse Glimmer becomes usable from a plain `pip install`, and serveable to an
agent. [Blaizzy/mlx-vlm#1838](https://github.com/Blaizzy/mlx-vlm/pull/1838)
merged the day after the port shipped, which collapsed the install to one line —
and, because the released code differs from the PR head the port was written
against, broke every Muse Glimmer entry point until fixed here.

### Added

- **`turboquant-serve-vlm`: an OpenAI-compatible server for multimodal
  TurboQuant models.** `turboquant-serve` wraps `mlx_lm.server`, which has no
  notion of VLM architectures, so Muse Glimmer and DiffusionGemma could not be
  served at all. Rather than reimplement a server, this drives *mlx-vlm's* —
  OpenAI and Anthropic routes, per-model tool parsers, continuous batching — and
  teaches it the three things it cannot infer about a polar checkpoint:

  1. **How to load it.** mlx-vlm's server reaches the model through exactly one
     seam (`mlx_vlm.server.generation.load`), so binding that name to a
     TurboQuant-aware wrapper covers the whole stack. Non-TurboQuant models keep
     mlx-vlm's own loader, so one server can still serve either.
  2. **Where the reasoning channel ends** (see Fixed).
  3. **What the reasoning knob is called** (see Fixed).

  Verified end-to-end on `Muse-Glimmer-30B-tq4-g64`: tool calls return
  `finish_reason: tool_calls` with correct name and arguments, tool-result round
  trips answer correctly, and the streaming path emits no protocol markup.

### Fixed

- **Muse Glimmer leaked its entire reasoning channel into `message.content`.**
  It answers in a harmony-style channel format
  (`to=self<|message|>…<|eom|><|start|>assistant to=user<|message|>…`), and
  mlx-vlm's ATEM parser only strips that envelope when a tool call was parsed —
  so tool turns looked fine while ordinary turns handed the caller the model's
  private deliberation as its reply, with `reasoning_content` left `None`. An
  agent harness would ingest it as the answer.

  mlx-vlm's generic thinking splitter handles this correctly once pointed at the
  right delimiters, so `turboquant-serve-vlm` supplies them per architecture.
  The span deliberately ends at the routing header rather than at `<|eom|>`, so
  content starts at the first real answer token instead of carrying
  `<|start|>assistant to=user<|message|>` on every reply. An explicit
  `--thinking-start-token`/`--thinking-end-token` always wins.

- **`reasoning_effort` was a silent no-op on Muse Glimmer.** The server passes
  OpenAI's `reasoning_effort` to the chat template; Muse Glimmer's template only
  reads `reasoning_strength` (`reasoning_effort` and `enable_thinking` appear
  zero times in it), so the request was dropped without a warning and the model
  always deliberated at its `high` default. Requests are now translated
  (`minimal`/`none` → `low`, since Muse has no "off" level), and
  `--reasoning-strength` sets the default for clients that ask for nothing —
  which is most agent harnesses. Measured on `tq4-g64`: a tool-result turn cost
  **54 completion tokens at `low` versus 106 at `high`**, same answer.

- **`--reasoning LEVEL` and `--no-think` for `generate_vlm`.** Muse Glimmer's
  chat template defaults to `reasoning_strength: high`, which spends hundreds of
  tokens deliberating before a short answer — on a 16 GB mini at ~3.5 tok/s that
  is most of the wall clock. `--reasoning low|medium|high|xhigh` sets it, and
  `--no-think` asks for the least reasoning a template supports (it also passes
  `enable_thinking=False` for Qwen-style templates). Templates that do not
  reference these keys ignore them. Note that Muse Glimmer has no "off" level,
  so `--no-think` reduces but does not eliminate the thinking channel.

### Changed

- **Muse Glimmer installs with a plain `pip install "turboquant-mlx-full[vlm]"`.**
  [Blaizzy/mlx-vlm#1838](https://github.com/Blaizzy/mlx-vlm/pull/1838) merged on
  2026-08-10 and shipped in **mlx-vlm 0.6.12**, so the git pin previously
  documented in the README is obsolete and has been removed. `[vlm]` now
  requires `mlx-vlm>=0.6.12`.

- **The `transformers<5.13` cap is now a `!=5.13.*` exclusion.** The
  `AutoTokenizer.register` breakage is confined to the 5.13 series — verified
  against mlx-lm 0.31.3, where 5.12, 5.14 and 5.15 all import cleanly and only
  5.13 raises. The old cap was incompatible with mlx-vlm 0.6.12
  (`transformers>=5.14`), so a resolver honouring it would silently downgrade
  mlx-vlm to a build with no `muse_glimmer` — the same failure mode that broke
  the documented install order in 0.20.1.

### Fixed

- **`patch_vlm_arch` crashed on mlx-vlm >= 0.6.12**, taking `convert_vlm` and
  `load_turboquant_vlm` — every Muse Glimmer entry point — down with it. The
  released #1838 differs from the PR head this support was written against:
  `NormedEmbedding` is gone, and `TextModel` now owns a paramless `embed_norm`
  that it applies in `__call__`. The patch reached straight for
  `_mg.NormedEmbedding` and raised `AttributeError`. It is now a no-op on the
  merged layout, where a plain `QuantizedEmbedding` is already correct because
  the norm is applied outside it.

  **Checkpoints are unaffected in both directions**: `RMSNormNoScale` has no
  parameters, so neither layout contributes an `embed_norm` key, and models
  converted against either one load against both. Verified end-to-end by
  generating with the published `tq4-g64` build on mlx-vlm 0.6.12.

## [0.20.1] - 2026-08-10

A planner correctness fix and the field data behind it. `turboquant-plan`
told a 16 GB Mac it could not run a model that then ran **resident** on that
exact machine — a units bug in `--ram-gb`. Also corrects the documented
Muse Glimmer install order, which silently produced a broken environment.

### Fixed

- **`turboquant-plan --ram-gb` read its argument as decimal GB**, but a machine
  sold as "16 GB" reports `hw.memsize` = 17.18e9 bytes. Planning for a 16 GB
  Mac therefore understated its ceiling by 7.4% (14.40 vs 15.46 GB usable) and
  returned `WILL NOT RUN` for Muse-Glimmer-30B `tq3-g64` — a model that then
  ran **resident** on exactly that machine (Mac16,10, macOS 26.5.2,
  `iogpu.wired_limit_mb=14336`), peaking at 14.21 GiB at 5068 tokens of prompt.
  `--ram-gb` is now interpreted as the size the machine is sold as, which is
  what anyone typing it means, and `--ram-gb 16` reproduces the real mini's
  numbers exactly. Verdicts for machines planned this way become slightly more
  optimistic; verdicts read from the local device are unchanged.

  Fixing this exposed a second, opposite error the first was masking: with the
  corrected ceiling, the chooser now picks `--prefill-step-size 512` for a MoE
  at 21K context where the field measured 128 as necessary. The likely cause is
  that MoE expert dequantization is still not modelled, so the undersized-RAM
  bug had been compensating for it. The field test has been narrowed to the
  part that was actually measured (2048 OOMs) with the open question recorded
  in place; **re-measuring the real step ceiling for a MoE at 21K is
  outstanding.**

- **The documented Muse Glimmer install order was broken.** It installed the
  pinned `mlx-vlm` from git *first* and `turboquant-mlx-full[vlm]` second — but
  `[vlm]` depends on `mlx-vlm>=0.6.3`, so the resolver replaced the git build
  with a PyPI one. Measured in clean venvs: the documented order yields
  mlx-vlm **0.6.4** and `ImportError: cannot import name 'muse_glimmer'`; the
  corrected order yields 0.6.11 and imports fine. Now documented as PyPI first,
  pinned overrides last, with `--force-reinstall` (the PyPI build already
  satisfies the constraint, so uv would skip the git one) and `--no-deps` (stops
  it re-resolving transformers back below 5.15). README and both model cards
  updated.

### Added

- Field validation for **Muse-Glimmer-30B `tq3-g64` on a 16 GB Mac mini**
  (Mac16,10, macOS 26.5.2, `iogpu.wired_limit_mb=14336`): runs resident at
  every prompt length tried up to 5068 tokens, peaking at 14.21 GiB with
  0.20 GB of headroom, decode flat at ~3.5 tok/s. Prefill degrades from
  20 tok/s at 868 tokens to 6.0 at 5068 — memory pressure during prefill, not
  the kernel — so ~2000 tokens is the practical interactive limit. Meta's own
  smallest Apple Silicon artifact is 17.95 GB, text-only, which does not fit
  that machine at all.

## [0.20.0] - 2026-08-10

Meta's **Muse Glimmer** (dense 30B VLM) lands, and with it the first case where
TurboQuant beats MLX's affine quantizer on quality *and* size at matched bit
width — PPL 4.3315 in 14.88 GiB against 4.3798 in 19.88 GiB. Alongside it,
**`polar_qmm`** closes a long-standing hole in the kernel set: dense prefill no
longer materializes dequantized weights, which cuts prefill peak memory for
*every* dense TurboQuant model. `turboquant-plan` gains two prefill terms it was
missing, which changes its output for existing models.

### Added

- **Muse Glimmer (Meta, dense 30B VLM) support** — `muse_glimmer`, a
  29.8B-parameter dense multimodal model with Gemma2-style sandwich norms,
  gated attention, a 3:1 sliding(2048)/full attention pattern with NoPE on the
  full-attention layers, and a 1.9B ViT-G/14 perception encoder. Converts
  through `convert_vlm`.

  At **matched 4-bit**, `tq4-g64` measures **PPL 4.3315 in 14.88 GiB** against
  `mlx-community/Muse-Glimmer-30B-4bit`'s **4.3798 in 19.88 GiB** — better
  quality in 25% less space, while quantizing the embedding and vision tower
  that the affine build leaves at bf16. Decode is 2.4–2.8× slower, the usual
  codebook-vs-affine trade.

  The MLX model classes come from
  [Blaizzy/mlx-vlm#1838](https://github.com/Blaizzy/mlx-vlm/pull/1838), which is
  **not merged and not on PyPI** — see the README for the pinned install.

- **`polar_qmm`: fused dense batched GEMM on packed weights.**
  `PolarQuantizedLinear` previously had a fused kernel only for single-vector
  decode; every batch > 1, meaning all of prefill, fell back to
  `polar_dequantize_weight` + GEMM and materialized the weight at ~14 bytes per
  parameter. The new kernel decodes in registers and writes nothing, and is
  dispatched for `2 <= n_tokens <= 256` with `output_dims >= 512` — bounds set
  by measurement (9.3× faster than dequant+GEMM at N=2–8 on wide MLP
  projections, 0.41× at N=2048 where MLX's tuned GEMM wins, 0.5–0.8× when the
  output is too narrow to fill the GPU). `TURBOQUANT_QMM_MAX_TOKENS` overrides
  the token bound for memory-constrained machines.

  On Muse-Glimmer-30B tq3, prefill peak over resident weights at 64 tokens
  drops **3.86 GB → 0.62 GB** and prefill runs **23.1 → 73.9 tok/s**. Benefits
  every dense TurboQuant model, not just this one. Perplexity is unchanged
  (5.0454 dequant vs 5.0452 fused — fp32 accumulate vs fp16 GEMM).

- Rotation-fusion config entries may be **parent-qualified**
  (`"self_attn.gate_proj"`), and qualified entries are matched before bare leaf
  names. Muse Glimmer is the first architecture with the same leaf name under
  two parents in one block — a sigmoid attention output gate and a SwiGLU gate,
  reading different norms — which the old leaf-only match would have fused into
  the wrong norm.

- `convert_vlm` gained `--extras-bits` / `--extras-group-size`.
  `--quantize-extras` was hard-wired to 8-bit; on models where the extras are a
  large share (Muse Glimmer's embedding + vision tower are 3.3B params) that is
  the difference between ~3.9 GB and ~1.8 GB.

### Fixed

- **`turboquant-plan` under-estimated prefill workspace**, which on small
  machines turned a will-not-run into a false `RESIDENT` verdict. It modelled
  attention scratch only, missing (1) the `lm_head` output for the chunk —
  often the largest term on a big-vocab model, 0.77 GB at 202048 vocab × 2048
  tokens — and (2) polar weight dequantization for dense TurboQuant layers
  above the fused kernel's token bound. On Muse-Glimmer-30B at 16K context the
  estimate goes 2.15 GB → 6.70 GB, against a measured 6.39 GB. **Existing
  plan output for other models will change**: every model gains the logits
  term, and dense TurboQuant models gain the dequantization term. MoE expert
  dequantization is still not modelled.

- `muse_glimmer`'s `lm_head` is kept out of the polar path. At
  202048 × 6656 = 1.345B parameters it is 10× the largest MLP matrix, so the
  dequantize-on-prefill fallback cost **18.84 GB of transient peak alone** to
  save 0.21 GB on disk. Routing it to affine cut prefill peak over weights
  from 20.06 GB to 3.86 GB for +0.16 GiB, with prefill slightly faster.

- Muse Glimmer's `embed_tokens` is a `NormedEmbedding` — an `nn.Embedding`
  subclass that RMS-normalizes the row it looks up. It inherits
  `to_quantized`, which returns a plain `QuantizedEmbedding` and **silently
  drops the normalization**. It now quantizes through a norm-preserving
  subclass on both the convert and load sides. (mlx-vlm sidesteps this by
  refusing to quantize the module at all, which is most of why a "4-bit" Muse
  Glimmer weighs 19.88 GiB.)

## [0.19.0] - 2026-08-04

Two new architectures, both of which mlx-lm has no module for. **Kimi K3**
(2.8T MoE) is the largest model TurboQuant-MLX has run — an external
contribution from [@anders94](https://github.com/anders94), streaming 931 GB of
experts from disk on a single 512 GB Mac Studio. **Sarvam MoE** converts and
runs, but ships with a documented negative result rather than a published
build. Same-layer read fan-out arrives as an opt-in `--fanout`.

### Added

- **Kimi K3 (2.8T MoE) support**, contributed by
  [@anders94](https://github.com/anders94) in
  [#71](https://github.com/manjunathshiva/turboquant-mlx/pull/71). MLX model
  port (69 KDA linear-attention + 24 gated-NoPE MLA layers, LatentMoE experts
  behind shared down/up projections), an mxfp4 compressed-tensors conversion
  path, planner and expert-streaming coverage, and a model card for a 931 GB
  `tq3a-tqTe-down4-g64` build that streams from disk on a 512 GB Mac Studio.
  Verified against the HF reference at ≤ 2.7e-6 fp32 forward parity, with a
  bit-exact mxfp4 unpack check.

- **`sarvam_moe` architecture support** (Sarvam AI's sarvam-30b and any later
  model of that type). mlx-lm has no module for it, so `compat.py` aliases our
  port in the same way it already does for Laguna. The port is verified three
  ways against the fp32 reference: full key/shape coverage of all 7122 source
  tensors, layer-1 output parity at 5.4e-7, and whole-model logit agreement.

  Two details of this architecture are easy to get wrong and are pinned by
  tests. The checkpoint is Megatron-style, so attention arrives as a single
  fused `attention.query_key_value` matrix — the reference views it as
  `(heads + 2*kv_heads, head_dim)` and splits on the head axis, which makes a
  plain row split exact, but a Q/K/V mis-slice still produces a model that
  loads and generates plausible-looking text. Routing is DeepSeek-V3-style:
  sigmoid scores with `expert_bias` added **for selection only**, so the bias
  must not leak into the combine weights. The router is deliberately not an
  `nn.Linear`, which keeps the quantizer's module walk from quantizing it.

  Note on quantization: at 4-bit this model degenerates on long generations,
  measured as a rate over repeated trials at temp 0.7 / top_p 0.9 —
  **TurboQuant 4-bit degenerated 8/15 (53%), mlx-lm affine 4-bit 15/15
  (100%)**. TurboQuant is clearly the better of the two, and neither is good
  enough to ship, so no quantized sarvam build is published. Use a higher bit
  width and validate long-form output on your own prompts.

- **`--fanout` on `stream_generate` and `serve`** — same-layer read fan-out:
  once a layer's router has chosen its experts, the other projections' misses
  are submitted to the read pool at layer start so their reads overlap the
  earlier projections' compute. Off by default. It trades away the coalesced
  serial read path, which is a measured win on a fast internal SSD with spare
  bandwidth and a loss on bandwidth-bound external storage, and unlike
  `--prefetch-ahead` it has no self-disable: the saturation throttle judges by
  rescue rate, and fan-out has none to judge (its claims are accounted as
  misses by design). Measure it on your own storage.

### Changed

- **Always-on MoE plumbing is now exempt from the `--mlp-bits` tier.**
  `bits_for_path` returns the base `--bits` for `shared_experts` and
  latent-MoE `routed_expert_*` projections: unlike a routed expert (1 of many,
  chosen top-k), these run on every token, so dropping them into a sub-2-bit
  expert tier costs quality across the whole stream for a rounding-error size
  saving. Affects new hybrid conversions of models with shared experts
  (DeepSeek family); existing checkpoints are unaffected, since bit width is
  recovered from the on-disk codebook rather than re-derived.

- **`trust_remote_code` is auto-injected only for local Kimi K3 checkpoints.**
  The new `compat.is_local_kimi_k3()` gate requires a local *directory* whose
  `config.json` declares `model_type: "kimi_k3"`; hub repo ids, other local
  models, and missing or corrupt configs are left to transformers' explicit
  opt-in, and a `trust_remote_code` passed by the caller always wins.
  Downloading tokenizer code and executing it are separate trust decisions.

- **`--cache-budget-gb auto` is no longer double-conservative, and `plan` now
  predicts what the loader actually does.** The two computed the budget
  differently: `plan.py` sized from the cap a `sysctl` bump *could* reach and
  subtracted a 1 GB reserve, while `stream/loader.py` sized from the cap the
  machine has and subtracted 2 GB. On a 16 GB M4 mini planning Laguna-S-2.1,
  `turboquant-plan` advertised "~9.1 GB here" where the loader chose 5.89 GB —
  and 9.1 GB would have peaked near 12.4 GB against that machine's 12.71 GB
  cap. The arithmetic now lives in one module, `cache_budget.py`, imported by
  both, and sizes from the *current* working set: a default must not assume a
  sysctl the user was never told to run.

  The formula also stopped taking 80% of the working set *and* subtracting a
  2 GB reserve, which double-counted safety. A `--cache-budget-gb` sweep on
  that mini (2/4/5.89/6/8 GB) put mlx's peak at **cache budget + 3.33 GB**
  every time, within 40 MB, so the budget is now derived from that measured
  relation rather than a blanket fraction. `auto` picks **7.4 GB** where it
  used to pick 5.89 on the same machine; measured throughput across that range
  was 1.36 → 1.58 tok/s. `turboquant-plan` prints the projected peak alongside
  the budget so the number can be checked against a real run.

### Fixed

- **The streaming layer trigger fired on the wrong projection.** It fired on
  `_PROJS[0]` (`gate_proj`), but `SwitchGLU.__call__` executes `up_proj` first,
  so every layer's prefetch was kicked off one projection late. Applies to
  every streaming model, not just K3. Caught by @anders94.
- **`turboquant-plan --model ~/typo` says the directory is missing.** Anything
  that was not a directory was handed to the Hub, so a mistyped path came back
  as "Repo id must be in the form 'repo_name' or 'namespace/repo_name'" — which
  sends you looking for a naming rule when a directory is simply not there.
- **`release.yml` verifies sdist contents too.** The guard added in 0.18.1 ran
  only in `ci.yml`; the job whose output actually reaches PyPI now runs it as
  well.

## [0.18.1] - 2026-07-26

Hotfix. **0.17.0 and 0.18.0 installed from PyPI cannot run any command** — the
`models/` subpackage that 0.17.0 added for Poolside Laguna was never listed in
`pyproject.toml`, so it was absent from the distribution. `compat.py` imports
`turboquant_mlx.models.laguna` at module scope with no guard and eleven modules
import `compat`, so `turboquant-generate`, `turboquant-convert`,
`turboquant-serve` and `stream_generate` all raised
`ModuleNotFoundError: No module named 'turboquant_mlx.models'` on first import.
`turboquant-plan` and `turboquant-doctor` were unaffected — they are the only
entry points that do not import `compat`. Upgrade from 0.17.0 or 0.18.0.

### Fixed

- **`turboquant_mlx.models` is included in the distribution.** The flat-package
  layout (the repo root *is* the `turboquant_mlx` package) means subpackages
  cannot be auto-discovered and must be enumerated in `[tool.setuptools]
  packages`; `models/` was not. Nothing caught it because the test suite runs
  from the source tree, where the directory is present regardless of what gets
  packaged.

### Added

- **Two guards against a repeat.** `tests/test_packaging.py` cross-checks the
  declared package list against the subpackages actually on disk (in both
  directions), and CI's `build sdist` job now asserts the built tarball really
  contains an `__init__.py` for every declared package. The second is the one
  that tests the artifact rather than the source tree — verified against the
  published 0.18.0 tarball, which it rejects.

## [0.18.0] - 2026-07-26

Expert streaming on Poolside Laguna. A MoE block names its stacked-expert
container either `switch_mlp` (qwen3_5_moe, deepseek) or, as a plain mlx-lm
`SwitchGLU`, `experts` (laguna, gpt-oss). Four places had to know that and only
one did — so Laguna could not be streamed, planned, repacked or calibrated
correctly. The rule now lives in one module, `expert_naming.py`, imported by all
four. Laguna-S-2.1 (118B) streams on a 16 GB Mac at a measured 7.3 GB peak.

### Fixed

- **Expert streaming engages on Laguna models at all.** `load_streaming` looked
  for the expert container under `switch_mlp` only, so on Laguna it swapped
  **zero** projections and silently fell back to loading the whole model
  resident — harmless on a 64 GB Mac, an out-of-memory crash on a 16 GB one.
  Check the loader's `[stream] swapped N expert projections` line: `swapped 0`
  means streaming did not engage.
- **Laguna's experts are recognised as streamable by the planner and the cache
  budget.** The module swap already handled both container names
  (`switch_mlp` and mlx-lm's `SwitchGLU` at `experts`), but the two places that
  *count* expert bytes — `plan.py:footprint` and
  `stream/loader.py:_streamed_expert_bytes` — still matched `switch_mlp` alone.
  Laguna therefore streamed correctly while reporting **zero** streamable bytes:
  `turboquant-plan` called a streamable model resident-only and printed
  "❌ WILL NOT RUN" for a 16 GB Mac that in fact runs Laguna-S-2.1 at a 7.3 GB
  peak, and `--cache-budget-gb auto` sized its cache from a resident figure that
  wrongly included all 26.4 GB of experts (explicit `--cache-budget-gb` was
  unaffected). The rule now lives in one place, `expert_naming.py`, imported by
  both. `experts` matches only anchored as `.mlp.experts.`, so the dense
  always-on `shared_experts` MLP stays resident as it must.
- **`stream/repack_experts.py` no longer corrupts routing on a Laguna
  checkpoint.** Its expert-stack branch matched the container by `switch_mlp`
  while the router branch matched `mlp.gate.*` unconditionally, so on Laguna it
  permuted the router rows and left the expert stacks in place — for all 47 MoE
  layers. The tool's correctness argument is that the two move *together*
  (a relabeling); permuting one alone is a rerouting. The output loaded cleanly
  and passed every shape assertion, so the damage was silent.
- **`stream/calibrate_experts.py` sizes the hot-expert pin list correctly on
  Laguna.** `_model_expert_info` returned 0 bytes/expert, so the
  `used + cost > cap` budget check never fired and *every* expert was written
  to `hot_experts.json` claiming ~0 GB — which the streaming loader then tries
  to pin into a wired cache sized for a fraction of it.

## [0.17.0] - 2026-07-24

### Added

- **Poolside Laguna support** (`model_type: "laguna"`) — MLX port in
  `models/laguna.py` (per-layer query-head counts, 3:1 full/sliding attention,
  partial+YaRN RoPE on global layers, per-head softplus attention gate, QK-norm,
  sigmoid router with selection-only correction bias, dense layer 0). Registered
  through `compat.py` so `convert`/`generate`/`serve` resolve it. Logit parity
  vs transformers CPU/fp32 is 2e-7. `Model.sanitize` handles both the on-disk
  per-expert layout (`experts.{i}.gate_proj.weight`, singular `shared_expert`)
  and the transformers in-memory packed layout.
- **`turboquant-generate --stop`** — register an extra terminator for one run,
  as a token string (`--stop '</assistant>'`) or id (`--stop 24`). Repeatable.

### Fixed

- **EOS ids declared in `generation_config.json` are now honored.** A tokenizer
  exposes only its single `eos_token_id`, so models declaring several turn
  terminators (Laguna ends a turn with `</assistant>` = 24, not `〈|EOS|〉` = 2)
  lost all but the first: generation ran past the end of the turn and the model
  answered the same question two or three more times, up to `--max-tokens`.
  `load_turboquant` now unions the model's declared ids into the tokenizer, so
  `generate`, `serve` (resident and streaming) and `evaluate` all stop correctly.
- **GLM-style tool-call function names are stripped.** mlx-lm's `glm47` tool
  parser (auto-selected for Laguna and GLM-4.7) captured the function name with
  a trailing newline (`"list_dir\n"`), so agent harnesses failed to match the
  tool and string arguments were wrongly deserialized. `compat.py` patches the
  parser to strip the name (self-disables once upstream fixes it).

## [0.16.0] - 2026-07-22

### Added

- **`turboquant-serve --kv-fused`** — enable the fused KV decode+attend kernel on
  the server. Requires KV quantization and `--kv-min-tokens 0` (single-tier); a
  non-zero sink window keeps decode on the standard path and the flag warns. On
  attention-sink / sliding-window layers the fused path **gracefully falls back**
  to dequantize+SDPA for that step (the earlier fail-loud guard became a
  fallback via `TurboQuantKVCache.fused_fallback_fetch`), so it is safe to enable
  on any model including GPT-OSS. The startup banner reports fused on/inactive.

- **Fused KV decode+attend Metal kernel** (`turboquant_mlx.kernels.kv_decode_attend`,
  opt-in via `layers.enable_fused_attend`). At each decode token it reads the
  **packed** TurboQuant KV cache directly and runs a FlashAttention-style
  online-softmax pass — decoding K/V on the fly instead of materializing the
  fp16 tensors that `dequantize -> scaled_dot_product_attention` builds. The
  inverse-Hadamard rotation is kept outside the kernel via an orthonormal
  identity (rotate the query once, un-rotate the output once), so the
  per-token rotation collapses too. Numerically equivalent to the dequant path
  (cosine 1.0; same fp16 rounding). Split-K / flash-decoding layout keeps the
  GPU saturated at long context. Measured end-to-end decode speedup **over the
  standard TQ-KV path**, byte-identical greedy output: **1.49×** on Llama-3.2-1B
  at 4.2K (70.5 → 105.2 t/s), 1.09× on the hybrid Qwen3.6-35B-A3B (only 10/40
  layers are full-attention); the isolated attention-layer win reaches ≈10× at
  32K. Scope: decode-only (`S_q=1`), batch 1, single-tier
  (`min_tokens_before_quant=0`), bit-packed K/V, equal K/V head dims that are a
  multiple of 32; prefill and non-applicable steps fall back to the standard
  path. Plain causal attention only — on attention-sink / sliding-window layers
  it gracefully falls back to dequantize+SDPA (see `--kv-fused` below). Parity +
  guard tests in `tests/test_fused_kv_attend.py`.

### Documentation

- Recorded the **measured** learning-cache result on a genuinely disk-bound 122B
  (`Qwen3.5-122B-A10B-tq3a-tqTe-g64`, 30.9 GB, `F_NOCACHE`, on a 16 GB mini):
  learned pins vs pure LRU lift the cache hit rate 54.0 → 55.9% and cut expert
  bytes read 84.4 → 81.0 GB (−4.0%), but decode speed stays inside run-to-run
  noise. Confirms the 0.15.0 "no speedup" claim on the hardware built to show a
  benefit — pinning helps hit-rate and disk, not throughput; disk bandwidth is
  the wall. No code change.

## [0.15.1] - 2026-07-16

### Fixed

- **`turboquant-plan`/`turboquant-doctor` on a HuggingFace repo id worked once,
  then failed forever** with `error: no *.safetensors in ...`. Planning a repo
  fetches its `config.json`, and that alone creates a snapshot directory in the
  HF cache. The next run got a cache hit on that directory, took it for a
  downloaded model, and died looking for weights that were never there — the
  tool poisoning its own cache. A cache hit is now used only once it proves it
  holds the whole checkpoint; otherwise planning stays on the network, where it
  belongs.

  Same check closes a quieter hole: a **partially downloaded** model (some
  shards present, interrupted transfer) had its weights summed from only the
  headers that arrived, so a half-fetched 30 GB model would report ~15 GB and a
  confident `RESIDENT` verdict — the exact false green light this tool exists to
  prevent. A local path is still planned on request, but now carries a warning
  that the numbers are under-counted.

- **A malformed `model.safetensors.index.json` now degrades instead of raising.**
  The index is fetched from an arbitrary repo id, so it is untrusted input: a
  top-level list/string/number, a non-dict `weight_map`, or a non-string shard
  name each raised an uncaught `AttributeError`/`TypeError` out of the new
  completeness check. An index that cannot be read no longer gets to vouch for
  the checkpoint.

## [0.15.0] - 2026-07-15

### Added

- **Learning expert cache for streaming** (`--no-learn-experts` opts out,
  `--usage-file` relocates it): the shipped `hot_experts.json` is one profile,
  baked by whoever converted the model. TurboQuant now also records which
  experts *your own* traffic routes to, persists that across runs, and pins from
  it once it has seen enough evidence — so a streamed MoE gets faster the more
  you use it, and adapts if your workload changes (chat → agent, another
  language).

  Precedence is explicit `--pin-file` > learned (once mature) > shipped
  hotlist; below the maturity threshold a profile is noise (a few tokens would
  pin whatever the greeting happened to touch), so the shipped prior still
  wins — but recording starts at the first token, so a cold machine bootstraps
  its own list while still benefiting from the shipped one. History decays on
  merge so a workload change re-pins within a few runs. The emitted spec is the
  same `{"pin": [[layer, expert], ...]}` schema as `hot_experts.json`, so it
  feeds the existing pin/preload path unchanged — and a good learned profile can
  simply be uploaded as a shipped hotlist.

  The profile lives in `~/.cache/turboquant-mlx/usage/` (XDG- and
  `TURBOQUANT_USAGE_DIR`-aware), **not** in the model directory: a model dir is
  usually a shared, content-addressed HuggingFace snapshot, and a stale file
  there would survive a re-download and silently mis-pin. A read-only cache dir
  is never fatal — a profile is an optimisation, not a dependency.

  Measured on a streamed ternary 35B (3 identical runs, 2 GB budget, fresh
  profile): cache hit 75.5% → **78.0%**, disk read 8.4 → **7.5 GB (−11%)** once
  the profile matures and takes over from the shipped list. **Decode speed was
  flat** (13.05 → 13.07 tok/s) — that box is a 64 GB M4 Max whose page cache
  holds the whole 9.4 GB model, so it is not disk-bound and a better pin cannot
  help it. The −11% fewer reads is what converts to tok/s on a machine that *is*
  disk-bound (a mini streaming a 30-50 GB build); that measurement is still
  outstanding, and no speedup is claimed until it exists.

### Documentation

- **Reproducibility and `--prefill-step-size`.** Changing the prefill chunk size
  can change a greedy answer by a token: measured on a ~3.9K-token prompt,
  `--prefill-step-size 2048` produced "...which *explains why* the sky reads
  blue" where 512/256/128 produced "...which *is why*...". Both are valid argmax
  continuations — the chunk size only changes the order the GPU reduces in, and
  fp16 rounding occasionally lands on the other side of a near-tie.

  **This is a property of chunked prefill on Metal, not of TurboQuant**: the
  same test on stock `mlx-community/Llama-3.2-1B-Instruct-4bit` (plain mlx-lm,
  plain affine 4-bit, none of our kernels) forks the same way. Our own decode
  and prefill MoE kernels are *bit-identical* to each other, now pinned by
  `tests/test_kernel_determinism.py` — that is the invariant `--disk-cache`
  checkpoint restores depend on, since a restore continues a prefill-built state
  on the decode path. For byte-reproducible greedy output, hold
  `--prefill-step-size` fixed; quality and correctness are unaffected.

## [0.14.1] - 2026-07-15

### Fixed

- **`turboquant-plan`: KV geometry for Mamba hybrids and sliding-window
  layers.** Found by sweeping the planner across all 16 published TurboQuant
  repos. It ran on every one, but two families were wrong:
  - **Nemotron-H** (`nemotron_h`) has no `layer_types` — its attention layers
    live in `hybrid_override_pattern` (`MEMEMEM*EMEM…`, `*` = attention,
    `M` = Mamba, `E` = MLP). The planner fell through to `num_hidden_layers`
    and assumed all 88 layers were full attention. **Only 8 are** — an 11x KV
    over-prediction that made the 120B look far heavier than it is
    (88.0 → **8.0 KB/token**).
  - **GPT-OSS and Gemma/DiffusionGemma** interleave `sliding_attention`
    layers, which hold real KV capped at `sliding_window`. Only `full` layers
    were counted — an **under-prediction**, the direction that OOMs.
    DiffusionGemma 40.0 → **65.0 KB/token** (25 sliding layers, window 1024);
    gpt-oss 36.0 → 36.6 (window 128).

  `kv_bytes(cfg, context, kv_bits)` now returns the total at a context rather
  than a flat per-token rate, since sliding layers stop growing at the window;
  `_attention_layers()` centralises the three ways a config declares which
  layers hold KV. The calibrated Qwen3.6-35B path is unchanged at exactly
  20.0 KB/token, so the field validation (0.67 GB predicted vs 0.62 measured
  at 32K) still holds.

- **`turboquant-plan`: hardened config parsing.** `config.json` comes from
  whatever HF repo id the caller passes, so a malformed value must degrade
  rather than quietly under-predict. `sliding_window: -1` made sliding layers
  *subtract* KV; a fractional `full_attention_interval` raised
  `ZeroDivisionError` (`int(0.5)` → 0); and a config with `layer_types` but no
  `num_hidden_layers` lost its projection unnecessarily. Anything that is not
  a positive integer now means "not set". All 16 published repos re-swept
  byte-identical — the guards touch only malformed configs.

## [0.14.0] - 2026-07-15

### Added

- **`turboquant-plan` / `turboquant-doctor` — preflight placement projection.**
  Answers "will this model run on this Mac, and with what flags?" *before* a
  multi-GB download or a five-minute load, by reading **only safetensors
  headers** and `config.json` — it allocates no tensors, starts no engine, and
  imports no model framework. Reports the exact weight split (total / streamable
  experts / resident backbone), a KV projection derived from the config's
  attention geometry (hybrid GatedDeltaNet models only grow KV on their
  full-attention layers — 10 of 40 on Qwen3.6-35B), an estimate of the transient
  prefill workspace (which scales with *both* chunk size and context), and a
  verdict: resident / resident-after-a-wired-bump / streaming / won't-run —
  plus the flags to use. `--wired-gb` / `--ram-gb` plan for a machine you are
  not sitting at; `--json` is machine-readable; `turboquant-doctor` adds a
  read-only readiness check (files, tokenizer, quantization block, mlx/mlx-lm)
  with stable check ids and exit codes (0 ok/warn, 1 won't fit / missing,
  2 usage). The projection is **calibrated against the 16 GB mini
  measurements**, not guessed: it predicts a 10.44 GB peak where the 9.4 GB
  ternary build measures 10.42 GB and correctly needs no `sudo`; it recommends
  `iogpu.wired_limit_mb=13721` for the 12.6 GB down4 build where 13824 is what
  works; and at 21K context it rejects the default 2048 prefill chunk that
  OOMed in the field. The 0.13.0 auto-guards fix tight-memory serving at
  runtime — this predicts it instead.

  A HuggingFace repo id is planned **over the network from its headers** (range
  requests via `get_safetensors_metadata`), never by downloading it: the 12.6 GB
  down4 repo answers in ~2.5 s and pulls 232 KB into a cold cache — so the
  question is answered *before* the download, which is the whole point. An
  already-cached repo is used from disk.

## [0.13.0] - 2026-07-13

### Added

- **Prompt-cache byte cap on `turboquant-serve`** (`--prompt-cache-max-gb`,
  default `auto`): mlx-lm's in-memory LRU prompt cache is bounded by entry
  count only, and an agent-harness conversation retains one KV state per
  turn — measured on a 16 GB mini serving the resident 12.6 GB down4 35B,
  8 retained sequences (1.14 GB) paged the system until a prefill command
  buffer stalled on swap I/O and the Metal watchdog killed the server (GPU
  Timeout). `auto` computes the byte budget at insert time (working-set
  headroom excluding the cache's own bytes, minus a 2 GB reserve, floored
  at 256 MB) and lets the upstream LRU evict down to it; roomy machines
  stay unbounded. Pair with `--disk-cache`: evicted states restore from
  disk instead of re-prefilling.

- **Mid-prefill disk-cache checkpoints** (`--disk-cache`, on by default;
  `--disk-cache-no-prefill-checkpoints` opts out): the disk prompt cache now
  also checkpoints every `--disk-cache-save-every` tokens *during* prompt
  processing, not just after the request completes. This fixes a measured
  total-reuse failure on hybrid GDN/Mamba models: the end-of-request
  checkpoint includes the generated assistant tail, and chat templates
  re-render that tail differently on the next turn (Qwen's empty `<think>`
  block appears at generation time but not in history), so the checkpoint is
  never a strict prefix of turn N+1 and the non-trimmable cache gets **zero**
  reuse — live on a 16 GB mini, a 21,250-token turn 2 re-prefilled from
  token 0 over a 4-token divergence. The mid-prefill ladder (1024, 2048, …)
  restores from the newest checkpoint below the divergence, and a *crash*
  mid-prefill resumes the same way instead of starting over.

- **Metal buffer-cache limit on `turboquant-serve`**
  (`--metal-cache-limit-gb`, default `auto`): MLX's GPU buffer reuse cache
  is unbounded by default, and a long chunked prefill allocates attention
  workspace with a new, larger shape every chunk — so stale buffers
  accumulate ~80 MB per 1K prompt tokens instead of being reused. Measured
  on a 16 GB Mac mini serving the resident 12.6 GB down4 35B: active memory
  flat at ~13.2 GB, buffer cache 1.9 → 3.5 GB, hard Metal OOM crash at
  14K/21K prompt tokens. `auto` caps the cache via `mx.set_cache_limit`
  right after the model loads whenever working-set headroom is under 8 GB
  (roomy machines keep the fast unbounded default); with the cap the same
  21K-token prefill runs flat at ~13.5 GB total. Pass a number (GB) to
  force a cap or `off` to disable. On 16 GB machines pair with
  `--prefill-step-size 256` — the *per-chunk* transient workspace scales
  with chunk size, and mlx-lm's default of 2048 OOMs tight boxes on the
  first chunk.

- **Auto cache budget + wired memory for streaming** (ds4-style):
  `--cache-budget-gb auto` sizes the expert cache from the machine — 80% of
  Metal's max recommended working set, minus the resident (non-expert)
  weights (computed from the safetensors index, nothing loaded), minus a
  2 GB KV/prefill reserve, clamped to [0.5 GB, all experts]. On a 16 GB mini
  with the 122B ternary this lands on ~4.2 GB, matching the hand-tuned
  known-good value. `--wire-memory` (opt-in) raises MLX's wired-memory limit
  to resident + budget + reserve so weights and the expert cache stay
  resident under memory pressure — the MLX-native equivalent of ds4's
  mlock'd cache chunks. Off by default: on a roomy machine the OS page cache
  already keeps re-reads warm, and wiring takes memory from other apps.

- **Shipped hot-expert list for streaming** (ds4-style): a model repo can
  now carry its own routing profile — drop `calibrate_experts.py`'s pin
  output into the model directory as `hot_experts.json` and the streaming
  loader pins **and preloads** those experts at startup (`--no-hotlist`
  opts out; explicit `--pin-file` still overrides). Preload reads the list
  in hotness order with coalesced batch reads, capped at 60% of
  `--cache-budget-gb` so the LRU keeps working room; experts past the cap
  are un-pinned and age through the LRU normally. Previously pinned experts
  loaded lazily on first miss, so the profile did nothing for cold-start
  latency. Cache stats gained `preload_experts` / `preload_gb`.

- **Greedy tool-call syntax on `turboquant-serve`** (`--tool-syntax-greedy`,
  optional `--tool-syntax-tags "<open>,</close>"`): a per-request logits
  processor masks logits to argmax while the generation is inside a
  `<tool_call>`...`</tool_call>` block — braces, keys, colons, tags — and
  leaves the configured sampler in charge of JSON *value* strings (free-text
  payloads) and of the decision to emit a tool call at all, so agent
  harnesses can serve low-bit builds at temperature without fabricated
  tool-call structure. Composes with any sampler and with the repetition
  penalties (argmax of the penalized distribution). Validated on the 35B
  asymmetric-ternary build: well-formed `tool_calls` via curl, and no
  regression on the Opencode agent smoke test (2/2 pass, same latency).

- **Asymmetric expert precision** (`convert --expert-down-bits {2,3,4}`):
  quantize MoE expert down-projections at a higher-precision Gaussian
  codebook while up/gate take the `--mlp-bits` / `--ternary-experts` tier —
  the down projection is the SwiGLU summation bottleneck, and llama.cpp-family
  2-bit mixes (up/gate IQ2_XXS, down Q2_K) rely on exactly this asymmetry.
  Applies only to MoE SwitchLinear experts (dense MLPs are untouched); the
  loader needs no matching rule because per-layer bits are self-describing
  via the on-disk codebook length. Works with both the in-memory and
  `--streaming` converters. Validated on Qwen3.6-35B-A3B: `--ternary-experts
  --expert-down-bits 4` (11.7 GB, vs 9.4 GB pure ternary / 16.4 GB tq3) is
  the first sub-2-bit-expert build to pass an agent-harness smoke test
  (Opencode 3/3, matching the 3-bit control's trajectory and latency), where
  pure ternary scores 0/4 and `--expert-down-bits 3` still fails on
  multi-step error recovery.
- **Disk-persistent prompt cache on `turboquant-serve`** (`--disk-cache
  [DIR]`, plus `--disk-cache-budget-gb` / `--disk-cache-min-tokens` /
  `--disk-cache-save-every`): each completed request's KV cache is
  checkpointed to disk (background writer, LRU byte-budget eviction) and
  restored on a prompt-cache miss — most importantly the first request after
  a server restart, which resumes from the longest saved token prefix
  instead of re-prefilling the whole conversation (a ~13-minute cold 22K
  prefill on a streaming 122B on a 16 GB mini becomes a checkpoint load plus
  a short suffix prefill). Token-level longest-common-prefix matching keeps
  the exact semantics of the in-memory cache: strict-prefix checkpoints
  extend any cache type, longer/divergent checkpoints are used only for
  trimmable caches. TurboQuant-quantized KV caches serialize too
  (`TurboQuantKVCache.from_state` added). Measured on a 64 GB M4 with
  Qwen3.6-35B-A3B-tq3-g32 and a 16.3K-token conversation: the first turn
  after a restart drops 21.1 s → 3.0 s (6.9x; 99.9% of the prompt served
  from the 700 MB checkpoint, 22 tokens freshly prefilled, greedy output
  byte-identical to the never-restarted server).

## [0.12.4] - 2026-07-04

### Added

- **`--no-think` and sampling defaults on the streaming CLI**
  (`python -m turboquant_mlx.stream.stream_generate`), matching the 0.12.3
  generate CLI: `--no-think` (disable thinking via the chat template),
  `--multi-think`, and `--top-p` / `--top-k` / `--rep-penalty` / `--rep-ctx`
  with defaults read from the model's `generation_config.json` (the neutral
  1.0 repetition penalty is ignored; 0 or 1 on the CLI force-disables), plus
  the single-`</think>` guard against re-emitted answers.

### Fixed

- **Small unsorted MoE batches no longer dequantize every expert.** mlx-lm's
  `SwitchGLU` only sorts routings once `indices.size >= 64`, so 2-7-token
  forwards (e.g. a short prompt extension on a warm prefix cache) arrive
  unsorted and fell into the dequantize-all fallback — ~34x the cost of a
  single-token forward (measured 1.17 s vs 34 ms per forward on the ternary
  Qwen3.6-35B-A3B). Both unsorted row layouts (gate/up: one row per token;
  down: one row per routing) now flatten into the fused per-routing kernel:
  M=2 1177→233 ms (5.0x), M=4 1156→269 ms (4.3x); single-token decode and
  sorted-batch paths are unchanged. Applies to both the resident and the
  streaming expert layers.
- **Top-k routing collision on unsorted batches**: a batch of exactly k
  unsorted tokens on a top-k model (e.g. 4 tokens on GPT-OSS's top-4)
  satisfied the flat-routings dispatch (`n_tokens == k`) and misaligned its
  rows against the k*k routing indices; the flat path is now additionally
  guarded by `indices.size == k`.
- **Pinned `transformers<5.13`**: transformers 5.13 dropped string keys in
  `AutoTokenizer.register`, which mlx-lm <= 0.31.3 still uses — every
  `import mlx_lm` fails. The cap will be lifted once a compatible mlx-lm
  release exists.

## [0.12.3] - 2026-07-03

### Added — thinking-mode ergonomics for low-bit builds

- **`--no-think` on the generate CLI**: disables thinking via the chat
  template (`enable_thinking=False`). On loop-prone low-bit thinking builds
  this answers instantly instead of deliberating for minutes. (The server
  equivalent already exists: `--chat-template-args '{"enable_thinking": false}'`.)
- **`--rep-penalty` now defaults from the model's `generation_config.json`**
  (ignoring the neutral 1.0), so a build that ships `repetition_penalty: 1.05`
  gets it automatically; `--rep-penalty 0` (or 1) force-disables. A 12-trial
  A/B/C/D sweep on the ternary 35B showed a light repetition penalty is the
  only sampling change that reliably exits `<think>` (3/3 vs 2/3 for every
  other config, including the pre-0.12.2 sampler).
- **`turboquant-serve --rep-penalty` / `--rep-ctx`**: server-side default
  `repetition_penalty` for OpenAI clients that never send the field
  (mlx_lm.server hardcodes the request default to 0.0 with no flag). When the
  flag is omitted, the default is read from the model's
  `generation_config.json`, resolved once on the first request.

### Fixed — doubled answers from a second `</think>`

- **Single-think-block guard** in the generate CLI: once `</think>` has been
  emitted, its logit is masked, so the model can't "reopen" the answer and
  emit it twice (observed sporadically on ternary builds even with top-k/top-p
  truncation). Disable with `--multi-think` for models that legitimately emit
  several think blocks.

## [0.12.2] - 2026-07-03

### Fixed — generate CLI sampler now honors the model's generation_config

- **`python -m turboquant_mlx.generate` sampled with temperature only** — no
  top-p/top-k truncation — so every token in the vocabulary kept nonzero
  probability at each step. Over long generations this occasionally sampled a
  stray special token; the visible symptom was a **doubled answer** when
  `</think>` was drawn where `<|im_end|>` belonged (seen on
  `Qwen3.6-35B-A3B-tq3a-tqTe-g64`). Low-bit builds are the most exposed since
  their logits are the noisiest. New `--top-p` / `--top-k` flags now default
  from the model's own `generation_config.json` (e.g. Qwen ships
  `top_k=20, top_p=0.95`), can be overridden per run, and `0` force-disables.
  Models without a `generation_config.json` keep the previous behavior.

## [0.12.1] - 2026-07-02

### Fixed — ternary experts on the streaming path

- **Streaming a ternary (trit) MoE now works.** `StreamingSwitchLinear` was
  trit-blind: its prefill dequant (`_dequantize_selected`) unpacked the base-3
  trit weights as bit-packed 2-bit indices (16 vs 20 values per uint32), which
  crashed with a `reshape` size mismatch, and its decode/prefill kernel calls
  never passed `trit=`. The streaming layer now detects the 3-entry codebook
  (with an explicit `trit` flag from the loader), decodes with `unpack_trits`,
  and threads `trit=` into `polar_gather_qmv` / `polar_multi_gather_qmv`. This
  is the path a >RAM ternary MoE takes — e.g. the 53 GB `Qwen3-235B tq3a-tqTe`
  streamed on a 16 GB Mac mini. The resident (fits-in-RAM) path was unaffected.

## [0.12.0] - 2026-07-01

### Added — ternary (1.58-bit) experts with base-3 trit packing

- **`convert --ternary-experts`** quantizes routed MoE experts to the data-free
  ternary `{-c, 0, +c}` codebook (optimal Lloyd-Max for N(0,1): c≈1.224) and
  packs the `{0,1,2}` indices as **genuine base-3 trits — 20 per uint32 (3²⁰ <
  2³²) = ~1.6 bpw**, vs 2.0 for the bit-packed 2-bit slot. Attention and
  `lm_head` stay at `--bits`; only the experts go sub-2-bit. The 3-entry
  codebook is self-describing: the loader decodes base-3 automatically (no new
  config field). All four expert Metal kernels (`polar_gather_qmv`,
  `polar_multi_gather_qmv`, `polar_dequant_experts`, `polar_gather_qmm`) decode
  trits inline, so the weight stays ~1.6 bpw in memory — never unpacked to a
  wider format. Needs expert redundancy: strong on 128-expert models
  (Qwen3-235B → **53 GB, fully resident on a 64 GB Mac at ~5.6 tok/s**, vs
  ~2 tok/s streaming the 70.5 GB hybrid), tq2e-class on 64 experts.
- Non-streaming `convert()` now honors **`--ternary-experts`** and
  **`--mlp-group-size`** (previously wired only on the `--streaming` path).

## [0.11.0] - 2026-06-20

### Added — expert streaming in `turboquant-serve`

- **`turboquant-serve --cache-budget-gb N`** routes the OpenAI-compatible server
  through `load_streaming`, so a MoE whose weights exceed RAM can be *served*
  over the API — only router-selected experts are paged from disk per token.
  This puts a ~50 GB **122B on a 16 GB Mac mini** behind Claude Code / Aider.
  The Flash-MoE levers from 0.10.0 ride along: **`--max-active-experts`**
  (K-reduction, default 4) and **`--use-page-cache` / `--no-page-cache`**
  (auto by model-size-vs-RAM). Composes with the `--kv-*` KV-quant flags;
  streaming is single-user, so pair with `--prompt-concurrency 1`. Verified
  end-to-end: the streamed model serves coherent completions over
  `/v1/chat/completions`.

## [0.10.0] - 2026-06-20

### Added — Flash-MoE streaming levers

- **`load_streaming(max_active_experts=4)` / `stream_generate --max-active-experts`**
  — K-reduction: caps router `top_k` to `min(native, K)` on every MoE block so
  the streaming switch pages fewer experts per token. `argpartition` selects
  fewer experts and `norm_topk_prob` renormalizes the gates, so it stays a clean
  reduced-K MoE. On streamed Qwen3.6-35B-A3B (256 experts, native top-8),
  **K=8→4 is byte-identical** on the 6-test stress harness and cuts streamed
  disk reads **~2.09×** (1.4× faster decode in the disk-bound regime); K=2
  collapses (broken JSON). Default `4` is a safe floor; `0` restores native
  routing. Ported from [Flash-MoE](https://github.com/danveloper/flash-moe).
- **`load_streaming(use_page_cache=…)` / `stream_generate --use-page-cache` /
  `--no-page-cache`** — "trust the OS": the reader's `F_NOCACHE` flag is now
  optional. On a machine where the model fits in free RAM, leaving the OS page
  cache on returns LRU-eviction re-reads from warm RAM instead of disk —
  measured **2.44× faster decode** on the 35B at a small cache budget (7.58 →
  18.50 tok/s), same hit-rate and RSS. The default auto-enables it only when the
  model's safetensors are `< 0.6×` total RAM, so a 16 GB mini streaming a 70 GB
  MoE keeps `F_NOCACHE` and never thrashes.

### Changed

- Streaming now defaults to `max_active_experts=4` (K-reduction on) and
  size-aware page-cache selection. Both are quality-neutral on validated models
  and overridable; pass `max_active_experts=0` / `--no-page-cache` for the prior
  behavior.

## [0.9.0] - 2026-06-17

### Added

- **`turboquant-serve` KV-cache quantization** — the server now accepts the
  same KV-quant flags as `turboquant-generate` (`--kv-bits`, `--kv-k-bits` /
  `--kv-v-bits`, `--kv-min-tokens`, `--kv-group-size`). `mlx_lm.server` has no
  native KV-quant option, so these are parsed by the wrapper and stripped
  before the remaining flags forward on; when set, every per-request prompt
  cache has its standard `KVCache` layers swapped for `TurboQuantKVCache`
  (other cache types — sliding-window / Mamba — pass through untouched, so
  hybrid models like GPT-OSS and Nemotron-H keep working). This shrinks each
  in-flight request's KV ~4x, which is the dominant memory lever for agentic
  loops (Aider, Claude Code) on memory-constrained boxes — distinct from
  `--prompt-cache-bytes`, which only bounds the cross-request reuse pool.
  Enabling KV-quant forces single-stream serving (TurboQuant caches lack the
  `merge` the batch generator needs).

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
