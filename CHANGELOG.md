# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **`--fanout` on `stream_generate` and `serve`** — same-layer read fan-out:
  once a layer's router has chosen its experts, the other projections' misses
  are submitted to the read pool at layer start so their reads overlap the
  earlier projections' compute. Off by default. It trades away the coalesced
  serial read path, which is a measured win on a fast internal SSD with spare
  bandwidth and a loss on bandwidth-bound external storage, and unlike
  `--prefetch-ahead` it has no self-disable: the saturation throttle judges by
  rescue rate, and fan-out has none to judge (its claims are accounted as
  misses by design). Measure it on your own storage.

### Fixed

- **The streaming layer trigger fired on the wrong projection.** It fired on
  `_PROJS[0]` (`gate_proj`), but `SwitchGLU.__call__` executes `up_proj` first,
  so every layer's prefetch was kicked off one projection late. Applies to
  every streaming model, not just K3. Caught by @anders94.

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
