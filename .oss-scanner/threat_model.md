# Threat model

TurboQuant-MLX (PyPI: `turboquant-mlx-full`) is a local tool: data-free weight
quantization for MLX on Apple Silicon, plus the loaders, CLIs and an
OpenAI-compatible server that run the compressed models. The repository root is
the importable `turboquant_mlx` package (see `pyproject.toml`).

## What this project does and where untrusted input enters

Treat everything that comes from a **model** or an **HTTP request** as
untrusted. A user typically runs `--model user/repo` on a Hub repo they did not
write, or a folder somebody shared.

- **Model repositories and folders** (local path or Hugging Face repo id; repo
  ids are fetched via mlx-lm's `_download` / `huggingface_hub`, see
  `generate.py:resolve_model_path`):
  - `config.json`: its `quantization` block drives per-layer bits and group
    size (`config.py:bits_for_path` / `group_size_for_path`, re-derived in
    `generate.py:_prepare_polar_layers`); `model_type` picks the model class
    (`compat.py:_register_local_model`, `models/`).
  - `*.safetensors`: loaded with `mx.load` (`generate.py`, `integration/vlm.py`,
    `mtp.py`, `stream/repack_experts.py`), and also parsed **by our own code**:
    the 8-byte header length, JSON header and `data_offsets` in
    `stream/safetensors_reader.py` (expert slices read with `os.pread` at
    computed offsets), `ngram_offload.py` (`_read_header`, `np.memmap`) and
    `plan.py`. The codebook length on disk selects bit-packed vs base-3 trit
    decode (`generate.py`, `stream/streaming_switch.py`, `core/packing.py`).
  - `model.safetensors.index.json`, `generation_config.json`, tokenizer files
    and chat templates (loaded by transformers / mlx-lm), expert hotlist/pin
    JSON shipped in a repo (`stream/loader.py:_find_hotlist`, `_load_pin_spec`),
    usage profiles (`stream/usage_profile.py`), `--expert-layer-tiers` JSON.
- **Remote code.** The package itself does not call `pickle`, `torch.load`,
  `np.load`, `eval` or `exec`, and leaves transformers' `trust_remote_code`
  off unless the user opts in (`serve` passes through mlx-lm's
  `--trust-remote-code`), with one deliberate exception: a local Kimi K3
  directory the user named on the command line runs its own tokenizer code.
  The CLIs (`generate`, `stream_generate`, `serve`'s operator `--model`,
  `calibrate_experts`, `convert` / `convert_streaming`'s `--hf-path`) register
  that argument with `compat.py:trust_local_kimi_k3` before anything is
  downloaded, and only when it is an existing local directory whose
  `config.json` declares `model_type: "kimi_k3"`; the tokenizer shim
  (`compat.py:_patch_kimi_k3_tokenizer_trust`) then injects
  `trust_remote_code=True` only for a path whose realpath was registered
  (`is_trusted_kimi_k3`). A Hub repo id, the cache snapshot it resolves to, and
  a model named in a server request are never registered. Before 0.28.1 the
  gate looked at the resolved directory instead
  ([GHSA-xccr-42wp-gxr2](https://github.com/manjunathshiva/turboquant-mlx/security/advisories/GHSA-xccr-42wp-gxr2)).
  See severity below.
- **HTTP servers.** `python -m turboquant_mlx.serve` wraps `mlx_lm.server` and
  patches its loader, prompt cache, KV cache and sampling defaults (`serve.py`).
  Inherited defaults: binds `127.0.0.1:8080`, no authentication,
  `--allowed-origins` defaults to `*`, and a request's `model` field is passed
  to the (patched) loader. `python -m turboquant_mlx.serve_vlm` wraps mlx-vlm's
  server (`serve_vlm.py`), which binds `127.0.0.1` by default from 0.28.1
  (mlx-vlm's own default is `0.0.0.0`). Request bodies, messages, tools and the
  model's tool-call output are parsed by the servers and by our patches
  (`tool_syntax_greedy.py`, `compat.py:_patch_glm47_tool_name_strip`).
- **Files the tool writes.** Converters write MLX safetensors plus `config.json`
  into `--mlx-path` and copy tokenizer/config files from the source
  (`convert_streaming.py`, `convert_vlm.py`, `mtp.py`). `--disk-cache` stores
  prompt KV checkpoints and an `index.json` under
  `~/.cache/turboquant/prompt-cache/` by default (`disk_prompt_cache.py`);
  `--prefill-stats-file` appends JSONL.
- **Shell-outs:** fixed argv only, no shell: `sysctl -n hw.memsize`
  (`plan.py`, `stream/loader.py`) and `ps -o rss= -p <pid>`
  (`stream/stream_generate.py`). `plan` / `doctor` print
  `sudo sysctl -w iogpu.wired_limit_mb=...` as advice and never run it.

## Components that matter most / least

- Most: the model-loading path (`generate.py`, `compat.py`, `models/`,
  `integration/vlm.py`, `mtp.py`, `ngram_offload.py`), the `stream/` readers,
  the servers and our patches to them (`serve.py`, `serve_vlm.py`,
  `disk_prompt_cache.py`, `tool_syntax_greedy.py`, `prefill_stats.py`), and the
  converters.
- Metal kernels (`kernels/*.py`): Metal source built as Python f-strings per
  `(bits, group_size, trit)` and run through
  `mx.fast.metal_kernel`. They **cannot execute** in the scan environment
  (Linux, CPU-only MLX), but their source is in scope for review: index math
  that reads or writes out of bounds when tensor shapes, `group_size`, expert
  indices or codebook length from a crafted model disagree with what the kernel
  assumes (e.g. expert `input_dims` a multiple of `group_size`).
- Less: `plan.py` / `doctor` (read-only sizing), `evaluate.py`, `demo_kv.py`.
- Out of scope: `benchmarks/` and `tests/` (developer scripts), docs and model
  cards. Bugs that live entirely in MLX, mlx-lm, mlx-vlm, transformers or
  huggingface_hub belong upstream; how this package wires, patches or sets
  defaults for them is in scope.

## How to exercise it

- The image is built from `.oss-scanner/Dockerfile`: Python 3.12, CPU-only
  `mlx[cpu]`, the package installed editable with `[kimi,vlm]`. No model
  weights are present and the scan is offline.
- CPU-safe tests (about 620, ~15 s):
  `python -m pytest tests/ -k "not requires_model" --deselect tests/test_core.py::test_qjl_unbiasedness $TQ_METAL_ONLY_TESTS`.
  `TQ_METAL_ONLY_TESTS` (set in the image) leaves out the tests that need Metal.
  Known Linux failure until it is fixed:
  `tests/test_tokenizer_trust.py::test_serve_main_registers_the_operator_model`
  starts `serve` with the default `--prompt-cache-max-gb auto`, which reads a
  Metal-only device field and raises `KeyError` (see the server note below).
- Tests build tiny synthetic models in a temp dir and are good templates for
  crafted inputs: `tests/test_flag_effects.py` and
  `tests/test_expert_layer_tiers.py` (convert, save, reload),
  `tests/test_convert_streaming.py`, `tests/test_kimi_k3.py`,
  `tests/test_tokenizer_trust.py` (the K3 trust gate),
  `tests/test_safetensors_reader_teardown.py`, `tests/test_ngram_offload.py`,
  `tests/test_disk_prompt_cache.py`, `tests/test_serve_*.py`.
- Server on Linux: `python -m turboquant_mlx.serve --host 127.0.0.1 --port 8080
  --prompt-cache-max-gb off --metal-cache-limit-gb off` (the `auto` defaults
  read Metal device info and fail on Linux). It starts without `--model`; point
  a request's `model` at a crafted local directory.
- CLIs, each with `--help`: `python -m turboquant_mlx.convert`,
  `python -m turboquant_mlx.generate`, `python -m turboquant_mlx.plan`,
  `python -m turboquant_mlx.serve_vlm` and
  `python -m turboquant_mlx.stream.stream_generate`.

## How you rate severity

- **Critical:** code execution from loading a model without the user passing
  `--trust-remote-code`, other than a local Kimi K3 directory the user named on
  the command line (for example: via a Hub repo id or its cache snapshot, via a
  model or draft model named in an HTTP request, or via a non-K3 model); code
  execution or arbitrary file write reachable from an HTTP request, including
  from a web page against a server bound to localhost.
- **High:** memory-unsafe behaviour (out-of-bounds read/write) from a crafted
  model file, including in our Metal kernel index math; reading or writing files
  outside the model, output or cache directory from model contents or request
  fields; leaking another user's prompts or cached KV state through the server.
- **Medium:** path traversal or file deletion that needs a tampered local cache
  or output directory; a crash of the server that one request can trigger.
- **Low:** denial of service from a model the user chose to load (huge headers,
  memory exhaustion, slow loads); local-only information leaks.
- Exposure: a finding that needs a server bound to a non-loopback address
  ranks one level lower (`serve` and `serve_vlm` both default to `127.0.0.1`).

## Anything to leave alone

- Numerical quality, quantization error, perplexity, and output differences
  between the Metal and CPU paths. "A malicious model gives bad or harmful
  answers" is not a vulnerability.
- Performance of the Metal kernels and anything that only fails because Metal
  is missing on Linux.
- The deliberate, documented trust of a local Kimi K3 directory the user named
  on the command line (above), as long as it stays within that boundary.

## Reports

One issue per report: the input that triggers it (a small script that builds
the crafted model directory or sends the request), the file and line, and a
minimal patch.
