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

Usage:
    turboquant-serve --model manjunathshiva/Nemotron-3-Super-120B-A12B-tq3
    turboquant-serve --model <path> --host 0.0.0.0 --port 8080
    # K8/V3 mixed-precision KV (recommended default), sink-protect first 128:
    turboquant-serve --model <tq-path> --kv-k-bits 8 --kv-v-bits 3 \
        --kv-min-tokens 128 --prompt-concurrency 1

All other flags forward to `mlx_lm.server`; see `--help`.
"""

from __future__ import annotations

import argparse
import sys

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


def _patch_loader() -> None:
    """Replace `mlx_lm.server.load` with a TurboQuant-aware wrapper.

    The server calls the bare name `load(...)` after `from .utils import
    load`, so patching the binding inside `mlx_lm.server` is what we need.
    We also patch `mlx_lm.utils.load` for any other callers that import
    from utils directly.
    """
    import mlx_lm.server as _server_mod
    import mlx_lm.utils as _utils_mod
    from mlx_lm.utils import _download, load_config

    _orig_load = _utils_mod.load

    # Importing turboquant_mlx.compat applies upstream shims (e.g. the
    # NemotronHConfig MLP-block-type patch) before any model loads.
    import turboquant_mlx.compat  # noqa: F401
    from turboquant_mlx.generate import load_turboquant

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


def main() -> None:
    _check_mlx_lm_version()
    _patch_loader()

    kv_config, remaining = _extract_kv_args(sys.argv[1:])
    if kv_config is not None:
        _patch_kv_cache(kv_config)
    # Hand mlx_lm.server only the args it understands.
    sys.argv = [sys.argv[0], *remaining]

    from mlx_lm.server import main as _mlx_lm_server_main

    sys.stderr.write(
        "TurboQuant-MLX serve  ·  OpenAI-compatible HTTP server "
        "(backend: mlx_lm.server, TurboQuant-aware loader)\n"
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
    _mlx_lm_server_main()


if __name__ == "__main__":
    main()
