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

Usage:
    turboquant-serve --model manjunathshiva/Nemotron-3-Super-120B-A12B-tq3
    turboquant-serve --model <path> --host 0.0.0.0 --port 8080

All flags forward to `mlx_lm.server`; see `--help`.
"""

from __future__ import annotations

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


def main() -> None:
    _check_mlx_lm_version()
    _patch_loader()

    from mlx_lm.server import main as _mlx_lm_server_main

    sys.stderr.write(
        "TurboQuant-MLX serve  ·  OpenAI-compatible HTTP server "
        "(backend: mlx_lm.server, TurboQuant-aware loader)\n"
    )
    _mlx_lm_server_main()


if __name__ == "__main__":
    main()
