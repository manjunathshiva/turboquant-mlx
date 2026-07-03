"""OpenAI-compatible HTTP server for TurboQuant-quantized MLX models.

Wraps `mlx_lm.server` and patches its loader so that models whose
config.json declares `quantization.mode == "turboquant"` are loaded
through `turboquant_mlx.generate.load_turboquant` (which knows how to
build PolarQuantizedLinear / PolarQuantizedSwitchLinear modules) rather
than mlx-core's built-in quantizer (which only knows affine/mxfp4/etc.
and crashes on `mode = "turboquant"`).

Non-TurboQuant models pass straight through to the standard mlx-lm
loader, so this server works as a drop-in replacement for
`mlx_lm.server` regardless of model type.

TurboQuant KV-cache compression
--------------------------------
`mlx_lm.server` has no native KV-quantization flags, so this wrapper adds
its own (`--kv-bits`, `--kv-k-bits`/`--kv-v-bits`, `--kv-min-tokens`,
`--kv-group-size`) — the same set exposed by `turboquant-generate`. When
enabled, every per-request prompt cache has its standard ``KVCache`` layers
swapped for ``TurboQuantKVCache`` (other cache types — RotatingKVCache for
sliding-window / Mamba layers — are left untouched, so hybrid models like
GPT-OSS and Nemotron-H keep working). This shrinks each request's KV
footprint ~4x, which is the real lever on memory-constrained boxes (e.g. a
streaming 120B on 16 GB) where Aider-style agentic loops grow context fast.

Note: enabling KV quantization forces single-stream (non-batched) serving,
because TurboQuant caches do not support the cross-request ``merge`` the
batch generator needs. That is the right trade-off for single-user setups;
pair it with ``--prompt-concurrency 1`` for a multi-client server.

Expert streaming
----------------
Passing ``--cache-budget-gb`` routes the loader through
``turboquant_mlx.stream.load_streaming`` instead of the resident loader, so a
MoE whose weights exceed RAM (e.g. a 122B on a 16 GB Mac mini) can be *served*
over the OpenAI API — only the router-selected experts are paged from disk per
token. The Flash-MoE streaming levers come with it: ``--max-active-experts``
(K-reduction, default 4 → ~2x less disk I/O) and ``--use-page-cache`` /
``--no-page-cache`` (auto by model-size-vs-RAM; trust-OS is ~2.4x faster decode
when the model fits free RAM, F_NOCACHE otherwise). Streaming is a single-user
path — pair with ``--prompt-concurrency 1``.

Usage:
    turboquant-serve --model manjunathshiva/Nemotron-3-Super-120B-A12B-tq3
    turboquant-serve --model <path> --host 0.0.0.0 --port 8080
    # K8/V3 mixed-precision KV (recommended default), sink-protect first 128:
    turboquant-serve --model <tq-path> --kv-k-bits 8 --kv-v-bits 3 \
        --kv-min-tokens 128 --prompt-concurrency 1
    # Stream a 122B on a 16 GB mini (+ KV-quant for long agentic context):
    turboquant-serve --model manjunathshiva/qwen3.5-122b-tq3 \
        --cache-budget-gb 4 --kv-k-bits 8 --kv-v-bits 3 --prompt-concurrency 1
    # Diagnostic: measure redundant prefill of an agentic client (Claude
    # Code / Aider) — per-request + exit summary, hardware-invariant:
    turboquant-serve --model <tq-path> --prompt-concurrency 1 \
        --prefill-stats --prefill-stats-file prefill.jsonl
    # Survive a multi-minute cold prefill behind claude-code-router (Claude
    # Code) — mirror prefill keepalives as real SSE data chunks:
    turboquant-serve --model manjunathshiva/qwen3.5-122b-tq3 \
        --cache-budget-gb 4 --prefill-keepalive --prompt-concurrency 1

Prefill keepalive (long cold prefills behind a proxy)
-----------------------------------------------------
A streaming agentic client (Claude Code via claude-code-router) aborts a
request whose connection sends no bytes for ~1 min and then retries — so a
multi-minute *cold prefill* (e.g. a 22K-token prompt on a streaming 122B over
a slow disk) never completes: each retry restarts the prefill from scratch.
``mlx_lm.server`` already emits keepalives during prompt processing, but as SSE
*comment* lines (``: keepalive p/t``), which a proxy parsing the SSE stream
drops before they reach the client. ``--prefill-keepalive`` mirrors each of
those (throttled by ``--prefill-keepalive-interval``, default 10s) as a real
``data:`` chunk (an empty assistant delta — no visible content), which proxies
forward and clients count as stream activity, so the client rides out the
prefill instead of timing out.

All other flags forward to `mlx_lm.server`; see `--help`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

_MIN_MLX_LM = (0, 31, 3)


def _check_mlx_lm_version() -> None:
    try:
        import mlx_lm
    except ImportError:
        sys.stderr.write(
            "ERROR: mlx-lm is not installed. Install with:\n"
            "    pip install 'mlx-lm>=0.31.3'\n"
        )
        sys.exit(1)

    raw = getattr(mlx_lm, "__version__", "0.0.0")
    parts = raw.split(".")
    try:
        version = tuple(int(p.split("+")[0].split("-")[0]) for p in parts[:3])
    except ValueError:
        return

    if version < _MIN_MLX_LM:
        need = ".".join(str(x) for x in _MIN_MLX_LM)
        sys.stderr.write(
            f"ERROR: mlx-lm {raw} is too old to load TurboQuant weights.\n"
            f"Nemotron-H latent-MoE projections (fc1_latent_proj / fc2_latent_proj)\n"
            f"and the MTP head landed in mlx-lm {need}. Upgrade with:\n"
            f"    pip install -U 'mlx-lm>={need}'\n"
        )
        sys.exit(1)


def _patch_loader(stream_config=None) -> None:
    """Replace `mlx_lm.server.load` with a TurboQuant-aware wrapper.

    The server calls the bare name `load(...)` after `from .utils import
    load`, so patching the binding inside `mlx_lm.server` is what we need.
    We also patch `mlx_lm.utils.load` for any other callers that import
    from utils directly.

    When ``stream_config`` is given, TurboQuant models are loaded through
    ``load_streaming`` (experts paged from disk) instead of the resident
    ``load_turboquant``; non-TurboQuant models always pass through to mlx-lm.
    """
    import mlx_lm.server as _server_mod
    import mlx_lm.utils as _utils_mod
    from mlx_lm.utils import _download, load_config

    _orig_load = _utils_mod.load

    # Importing turboquant_mlx.compat applies upstream shims (e.g. the
    # NemotronHConfig MLP-block-type patch) before any model loads.
    import turboquant_mlx.compat  # noqa: F401
    from turboquant_mlx.generate import load_turboquant
    if stream_config is not None:
        from turboquant_mlx.stream.loader import load_streaming

    def _tq_aware_load(
        path_or_hf_repo,
        tokenizer_config=None,
        model_config=None,
        adapter_path=None,
        lazy=False,
        return_config=False,
        revision=None,
    ):
        model_path = _download(path_or_hf_repo, revision=revision)
        cfg = load_config(model_path)
        is_tq = cfg.get("quantization", {}).get("mode") == "turboquant"

        if not is_tq:
            return _orig_load(
                path_or_hf_repo,
                tokenizer_config=tokenizer_config,
                model_config=model_config,
                adapter_path=adapter_path,
                lazy=lazy,
                return_config=return_config,
                revision=revision,
            )

        if adapter_path is not None:
            sys.stderr.write(
                "WARNING: --adapter-path is not supported for TurboQuant "
                "models; ignoring.\n"
            )

        if stream_config is not None:
            sys.stderr.write(
                f"[turboquant-serve] Streaming TurboQuant model from {model_path} "
                f"(cache_budget={stream_config['cache_budget_gb']} GB)\n"
            )
            # load_streaming returns (model, tok, cache); the cache stays alive
            # via the StreamingSwitchLinear modules that reference it.
            model, tokenizer, _cache = load_streaming(model_path, **stream_config)
        else:
            sys.stderr.write(
                f"[turboquant-serve] Loading TurboQuant model from {model_path}\n"
            )
            model, tokenizer = load_turboquant(model_path, lazy=lazy)
        if return_config:
            return model, tokenizer, cfg
        return model, tokenizer

    _server_mod.load = _tq_aware_load
    _utils_mod.load = _tq_aware_load


def _extract_kv_args(argv):
    """Peel TurboQuant KV-cache flags off ``argv`` before mlx_lm.server sees it.

    `mlx_lm.server`'s argparse would reject these as unknown, so we parse
    them here with a permissive pre-parser and hand the *remaining* args back
    to the server unchanged.

    Returns ``(kv_config | None, remaining_argv)``. ``kv_config`` is ``None``
    when no KV flag was given (KV quantization stays off), otherwise a kwargs
    dict for ``convert_cache_to_turboquant``.
    """
    # add_help=False: let -h/--help fall through to mlx_lm.server's parser.
    # allow_abbrev=False: never consume a server flag via prefix matching.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-k-bits", type=int, default=None)
    parser.add_argument("--kv-v-bits", type=int, default=None)
    parser.add_argument("--kv-min-tokens", type=int, default=0)
    parser.add_argument("--kv-group-size", type=int, default=64)
    ns, remaining = parser.parse_known_args(argv)

    if ns.kv_bits is not None and (
        ns.kv_k_bits is not None or ns.kv_v_bits is not None
    ):
        sys.stderr.write(
            "ERROR: --kv-bits is mutually exclusive with "
            "--kv-k-bits/--kv-v-bits\n"
        )
        sys.exit(2)
    if (ns.kv_k_bits is None) != (ns.kv_v_bits is None):
        sys.stderr.write(
            "ERROR: --kv-k-bits and --kv-v-bits must be set together\n"
        )
        sys.exit(2)

    if ns.kv_bits is None and ns.kv_k_bits is None:
        return None, remaining

    kv_config = dict(
        tq_bits=ns.kv_bits,
        k_bits=ns.kv_k_bits,
        v_bits=ns.kv_v_bits,
        group_size=ns.kv_group_size,
        min_tokens_before_quant=ns.kv_min_tokens,
    )
    return kv_config, remaining


def _extract_stream_args(argv):
    """Peel expert-streaming flags off ``argv`` before mlx_lm.server sees them.

    ``--cache-budget-gb`` is the trigger: when present, TurboQuant models load
    through ``load_streaming`` with the given budget and Flash-MoE levers.
    Returns ``(stream_config | None, remaining_argv)``. The other streaming
    flags are no-ops without a budget (and are documented as such).
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cache-budget-gb", type=float, default=None)
    parser.add_argument("--max-active-experts", type=int, default=4)
    parser.add_argument("--prefetch-workers", type=int, default=8)
    parser.add_argument("--prefetch-ahead", type=int, default=0)
    parser.add_argument("--pin-file", default=None)
    parser.add_argument("--use-page-cache", dest="use_page_cache",
                        action="store_true", default=None)
    parser.add_argument("--no-page-cache", dest="use_page_cache",
                        action="store_false")
    ns, remaining = parser.parse_known_args(argv)

    if ns.cache_budget_gb is None:
        return None, remaining

    stream_config = dict(
        cache_budget_gb=ns.cache_budget_gb,
        max_active_experts=ns.max_active_experts,
        use_page_cache=ns.use_page_cache,
        prefetch_workers=ns.prefetch_workers,
        prefetch_ahead=ns.prefetch_ahead,
        pin_file=ns.pin_file,
    )
    return stream_config, remaining


def _extract_prefill_stats_args(argv):
    """Peel the prefill-redundancy diagnostic flags off ``argv``.

    ``--prefill-stats`` (or just giving ``--prefill-stats-file``) turns on
    per-request logging of how much of each prompt is reused vs. re-prefilled,
    plus an exit summary of how much fresh prefill a working prefix cache would
    recover. Diagnostic only; no effect on generation.

    Returns ``(stats_config | None, remaining_argv)``.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prefill-stats", dest="prefill_stats",
                        action="store_true", default=False)
    parser.add_argument("--prefill-stats-file", dest="prefill_stats_file",
                        default=None)
    ns, remaining = parser.parse_known_args(argv)

    if not ns.prefill_stats and ns.prefill_stats_file is None:
        return None, remaining

    return dict(stats_file=ns.prefill_stats_file), remaining


def _extract_prefill_keepalive_args(argv):
    """Peel the prefill-keepalive flags off ``argv``.

    ``--prefill-keepalive`` mirrors mlx_lm's prefill keepalive comments as real
    SSE ``data:`` chunks so a proxy that drops comments (claude-code-router)
    keeps an agentic client (Claude Code) alive through a multi-minute cold
    prefill instead of timing out. ``--prefill-keepalive-interval`` throttles
    them (seconds; default 10). Returns ``(config | None, remaining_argv)``.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prefill-keepalive", dest="prefill_keepalive",
                        action="store_true", default=False)
    parser.add_argument("--prefill-keepalive-interval", dest="interval",
                        type=float, default=10.0)
    ns, remaining = parser.parse_known_args(argv)

    if not ns.prefill_keepalive:
        return None, remaining

    return dict(interval=ns.interval), remaining


def _patch_kv_cache(kv_config) -> None:
    """Wrap `mlx_lm.server.make_prompt_cache` to emit TurboQuant KV caches.

    Both the batchability probe and the per-request cache build call the bare
    name `make_prompt_cache(model)` inside `mlx_lm.server`, so patching that
    binding is sufficient. Standard ``KVCache`` layers become
    ``TurboQuantKVCache``; other cache types are left as-is. Because the
    converted caches lack ``merge``, the server's batchability probe sees them
    and falls back to sequential serving automatically.
    """
    import mlx_lm.server as _server_mod
    from turboquant_mlx.layers.polar_kv_cache import (
        convert_cache_to_turboquant,
    )

    _orig_make = _server_mod.make_prompt_cache

    def _tq_make_prompt_cache(model, *args, **kwargs):
        cache = _orig_make(model, *args, **kwargs)
        return convert_cache_to_turboquant(cache, **kv_config)

    _server_mod.make_prompt_cache = _tq_make_prompt_cache


class _KeepaliveSSEWriter:
    """Wrap an SSE ``wfile`` to mirror mlx_lm's prefill keepalives as data chunks.

    During a long prompt prefill ``mlx_lm.server`` writes SSE *comment* lines
    (``: keepalive p/t``) to keep the socket warm. Comments keep a *direct*
    client alive, but a proxy that parses the SSE stream (claude-code-router)
    drops comment lines — so a downstream agentic client sees a byte-less
    connection through a multi-minute cold prefill and aborts. This writer
    passes every byte through unchanged, and additionally mirrors each
    ``: keepalive`` comment (throttled to ``interval`` seconds) as a real
    ``data:`` chunk built by ``make_chunk`` — which proxies forward and clients
    count as activity. The chunk is an empty assistant delta, so no visible
    content is added.
    """

    def __init__(self, wfile, make_chunk, interval, clock=time.monotonic):
        self._wfile = wfile
        self._make_chunk = make_chunk
        self._interval = interval
        self._clock = clock
        self._last = 0.0

    def write(self, b):
        n = self._wfile.write(b)
        raw = bytes(b) if isinstance(b, (bytes, bytearray)) else b
        if isinstance(raw, bytes) and raw.startswith(b": keepalive"):
            now = self._clock()
            if now - self._last >= self._interval:
                self._last = now
                try:
                    chunk = self._make_chunk()
                    self._wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self._wfile.flush()
                except Exception:
                    pass  # keepalive is best-effort; never break the response
        return n

    def flush(self):
        return self._wfile.flush()

    def __getattr__(self, name):
        return getattr(self._wfile, name)


def _extract_rep_penalty_args(argv):
    """Peel the server-side repetition-penalty default flags off ``argv``.

    ``--rep-penalty`` sets a default ``repetition_penalty`` for requests that
    don't specify one (OpenAI clients rarely do); ``--rep-ctx`` sets the
    matching default context window. When ``--rep-penalty`` is omitted the
    default is looked up lazily from the model's ``generation_config.json``
    (loop-prone low-bit thinking builds ship one). ``--rep-penalty 0`` (or 1)
    disables the fallback entirely.

    Returns ``(rep_config | None, remaining_argv)``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rep-penalty", type=float, default=None)
    parser.add_argument("--rep-ctx", type=int, default=256)
    ns, remaining = parser.parse_known_args(argv)
    if ns.rep_penalty is not None and ns.rep_penalty in (0.0, 1.0):
        return None, remaining
    return {"penalty": ns.rep_penalty, "context_size": ns.rep_ctx}, remaining


def _patch_default_rep_penalty(rep_config) -> None:
    """Give requests without a ``repetition_penalty`` a server-side default.

    mlx_lm.server hardcodes the request default to 0.0 (disabled) with no CLI
    flag, so OpenAI clients that never send the field can't benefit. Wraps
    ``APIHandler.validate_model_parameters`` (which runs right after the body
    fields are read) and fills in the default only when the client did not
    send the field. The default is ``--rep-penalty`` when given, else the
    model's ``generation_config.json`` ``repetition_penalty`` (ignoring the
    neutral 1.0), resolved once on the first request.
    """
    import mlx_lm.server as _server_mod

    APIHandler = _server_mod.APIHandler
    _orig = APIHandler.validate_model_parameters
    resolved: dict = {}

    def _default_penalty(handler):
        if "penalty" in resolved:
            return resolved["penalty"]
        penalty = rep_config["penalty"]
        if penalty is None:
            try:
                from turboquant_mlx.generate import resolve_model_path

                # APIHandler exposes cli_args via response_generator in
                # mlx-lm 0.31.x; fall back to model_provider for other
                # versions rather than pinning one attribute name.
                provider = getattr(handler, "response_generator", None) \
                    or getattr(handler, "model_provider", None)
                model = provider.cli_args.model
                cfg_file = resolve_model_path(model) / "generation_config.json"
                if cfg_file.exists():
                    with open(cfg_file, encoding="utf-8") as f:
                        cfg_rep = json.load(f).get("repetition_penalty")
                    if cfg_rep and cfg_rep != 1.0:
                        penalty = cfg_rep
            except Exception as e:
                sys.stderr.write(
                    f"[turboquant-serve] Could not read generation_config "
                    f"repetition_penalty ({e})\n"
                )
        if penalty is not None:
            sys.stderr.write(
                f"[turboquant-serve] Default repetition_penalty={penalty} "
                f"(ctx={rep_config['context_size']}) for requests that don't "
                "set one\n"
            )
        resolved["penalty"] = penalty
        return penalty

    def validate_model_parameters(self):
        penalty = _default_penalty(self)
        if penalty is not None and "repetition_penalty" not in self.body:
            self.repetition_penalty = penalty
            if "repetition_context_size" not in self.body:
                self.repetition_context_size = rep_config["context_size"]
        return _orig(self)

    APIHandler.validate_model_parameters = validate_model_parameters


def _patch_prefill_keepalive(interval: float = 10.0) -> None:
    """Mirror mlx_lm's prefill keepalive *comments* as real SSE ``data:`` chunks.

    Patches ``APIHandler.handle_completion`` to wrap ``self.wfile`` in a
    ``_KeepaliveSSEWriter`` for the duration of a streaming request, so a proxy
    that drops SSE comments (and the agentic client behind it) still sees stream
    activity during a multi-minute cold prefill. No-op for non-streaming
    requests. Reuses mlx_lm's existing per-prefill-step callback for timing.
    """
    import mlx_lm.server as _server_mod

    APIHandler = _server_mod.APIHandler
    _orig = APIHandler.handle_completion

    def _handle_completion(self, request, stop_words):
        if not getattr(self, "stream", False):
            return _orig(self, request, stop_words)
        real_wfile = self.wfile
        self.wfile = _KeepaliveSSEWriter(
            real_wfile,
            make_chunk=lambda: self.generate_response("", None),
            interval=interval,
        )
        try:
            return _orig(self, request, stop_words)
        finally:
            self.wfile = real_wfile

    APIHandler.handle_completion = _handle_completion


def main() -> None:
    _check_mlx_lm_version()

    kv_config, remaining = _extract_kv_args(sys.argv[1:])
    stream_config, remaining = _extract_stream_args(remaining)
    prefill_stats_config, remaining = _extract_prefill_stats_args(remaining)
    prefill_keepalive_config, remaining = _extract_prefill_keepalive_args(remaining)
    rep_config, remaining = _extract_rep_penalty_args(remaining)
    _patch_loader(stream_config)
    if rep_config is not None:
        _patch_default_rep_penalty(rep_config)
    if kv_config is not None:
        _patch_kv_cache(kv_config)
    if prefill_stats_config is not None:
        from turboquant_mlx.prefill_stats import install as _install_prefill_stats
        _install_prefill_stats(**prefill_stats_config)
    if prefill_keepalive_config is not None:
        _patch_prefill_keepalive(**prefill_keepalive_config)
    # Hand mlx_lm.server only the args it understands.
    sys.argv = [sys.argv[0], *remaining]

    from mlx_lm.server import main as _mlx_lm_server_main

    sys.stderr.write(
        "TurboQuant-MLX serve  ·  OpenAI-compatible HTTP server "
        "(backend: mlx_lm.server, TurboQuant-aware loader)\n"
    )
    if stream_config is not None:
        pc = "auto" if stream_config["use_page_cache"] is None else (
            "on" if stream_config["use_page_cache"] else "off")
        sys.stderr.write(
            f"[turboquant-serve] Expert streaming: cache_budget="
            f"{stream_config['cache_budget_gb']} GB, "
            f"max_active_experts={stream_config['max_active_experts']}, "
            f"page_cache={pc} (single-user; use --prompt-concurrency 1)\n"
        )
    if kv_config is not None:
        if kv_config["tq_bits"] is not None:
            desc = f"K=V={kv_config['tq_bits']}-bit"
        else:
            desc = f"K={kv_config['k_bits']}-bit, V={kv_config['v_bits']}-bit"
        sys.stderr.write(
            f"[turboquant-serve] TurboQuant KV cache: {desc}, "
            f"group={kv_config['group_size']}, "
            f"sink={kv_config['min_tokens_before_quant']} "
            "(forces single-stream serving)\n"
        )
    if prefill_stats_config is not None:
        dest = prefill_stats_config["stats_file"] or "stderr only"
        sys.stderr.write(
            f"[turboquant-serve] Prefill-redundancy stats: ON "
            f"(per-request + exit summary; jsonl -> {dest}). "
            "Pair with --prompt-concurrency 1 for clean single-conversation "
            "numbers.\n"
        )
    if prefill_keepalive_config is not None:
        sys.stderr.write(
            f"[turboquant-serve] Prefill keepalive: ON (mirror prefill "
            f"keepalives as real SSE data chunks, throttled to "
            f"{prefill_keepalive_config['interval']:.0f}s) — keeps Claude Code "
            "and SSE-comment-dropping proxies alive through a long cold "
            "prefill.\n"
        )
    _mlx_lm_server_main()


if __name__ == "__main__":
    main()
