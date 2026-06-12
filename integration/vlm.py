"""mlx-vlm integration: convert/load TurboQuant models whose architectures
live in mlx-vlm rather than mlx-lm (multimodal and diffusion LLMs).

mlx-vlm's stock loader cannot be used directly on a TurboQuant checkpoint:
it sees ``config["quantization"]`` and applies ``nn.quantize`` (affine),
which does not understand the polar codebook format. ``load_turboquant_vlm``
replicates its model-construction steps and swaps in PolarQuantized layers
instead (mirroring ``turboquant_mlx.generate.load_turboquant`` for mlx-lm).

Requires the optional dependency:  pip install "turboquant-mlx-full[vlm]"
"""

import glob
from pathlib import Path

import mlx.core as mx

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.generate import _prepare_polar_layers


def _require_mlx_vlm():
    try:
        import mlx_vlm  # noqa: F401
        from mlx_vlm import utils  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "mlx-vlm >= 0.6.3 is required for VLM/diffusion architectures. "
            'Install with: pip install "turboquant-mlx-full[vlm]"'
        ) from e


# Per-architecture layers kept at full precision IN ADDITION to the always-on
# skips (vision/audio towers via mlx_vlm.utils.skip_multimodal_module, MoE
# routers, embeddings). Substring match on the module path.
#
# diffusion_gemma: the model's own upstream quant_predicate pins the router
# and the dense per-layer MLP at >= 8-bit (quant-sensitive); the
# self-conditioning MLP feeds every denoise step and is tiny. ".mlp." matches
# only the dense MLP — experts live under ".experts." as SwitchLinear.
VLM_SKIP_PATTERNS: dict[str, tuple[str, ...]] = {
    "diffusion_gemma": ("router", "self_conditioning", "embed_vision", ".mlp."),
}


def vlm_should_quantize(arch: str, base_predicate):
    """Wrap the converter's _should_quantize with VLM-specific skips."""
    _require_mlx_vlm()
    from mlx_vlm.utils import skip_multimodal_module

    skip_patterns = VLM_SKIP_PATTERNS.get(arch, ())

    def predicate(path, module):
        if skip_multimodal_module(path):
            return False
        if any(s in path for s in skip_patterns):
            return False
        return base_predicate(path, module)

    return predicate


def load_turboquant_vlm(model_path, lazy=False):
    """Load a TurboQuant-compressed mlx-vlm model.

    Args:
        model_path: Local directory containing the TurboQuant checkpoint.
        lazy: If True, don't evaluate parameters immediately.

    Returns:
        (model, processor, config) tuple. ``config`` is the raw dict with the
        "quantization" key removed (as mlx-vlm's prompt utilities expect).
    """
    _require_mlx_vlm()
    from mlx_vlm.utils import (
        apply_generation_config_defaults,
        get_model_and_args,
        load_config,
        load_processor,
        update_module_configs,
    )

    model_path = Path(model_path)
    config = load_config(model_path)

    tq_dict = config.pop("quantization", None)
    if tq_dict is None or tq_dict.get("mode") != "turboquant":
        raise ValueError(f"{model_path} is not a TurboQuant checkpoint")
    tq_config = TurboQuantConfig.from_dict(tq_dict)
    config.pop("quantization_config", None)

    model_class, _ = get_model_and_args(config=config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model_config = apply_generation_config_defaults(model_config, config)
    model = model_class.Model(model_config)

    weights = {}
    for wf in sorted(glob.glob(str(model_path / "model*.safetensors"))):
        weights.update(mx.load(wf))

    # TurboQuant checkpoints are saved from the model tree (mlx format),
    # so no weight sanitization is needed before loading.
    _prepare_polar_layers(model, weights, tq_config)
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())
    model.model_path = model_path
    model.eval()

    processor = load_processor(model_path)
    return model, processor, config
