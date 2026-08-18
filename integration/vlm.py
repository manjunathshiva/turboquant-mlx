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
from turboquant_mlx.generate import (
    _assert_no_orphan_quant_params,
    _prepare_affine_extras,
    _prepare_polar_layers,
)


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
# muse_glimmer: `lm_head` is deliberately left to the affine-extras path rather
# than polar-quantized. PolarQuantizedLinear has a fused Metal kernel only for
# the single-vector decode path; ANY batch > 1 (i.e. all of prefill) falls back
# to polar_dequantize_weight + GEMM, which materializes the weight through
# several full-size intermediates — measured at ~14 bytes per parameter.
# lm_head is 202048x6656 = 1.345B params, by far the largest matrix in the
# model, so that fallback costs **~19 GB of transient peak** during any prefill,
# versus ~1.9 GB for the largest MLP projection. Polar only buys 0.21 GB of
# steady-state size over 4-bit affine here. MLX's affine path has a fused
# quantized matmul for batched input and materializes nothing.
# Net: -19 GB prefill peak for +0.21 GB on disk.
# qwen3_5: the same lm_head reasoning, and the arithmetic is if anything worse
# — 248320x5120 = 1.271B params against a 248K vocab. Measured on a real
# PolarQuantizedLinear of that exact shape at 3-bit/g64:
#
#     tokens    1  ->   +0.000 GiB      (fused kernel, materializes nothing)
#     tokens  256  ->   +0.116 GiB
#     tokens  257  ->  +13.141 GiB      <-- fused kernel switches off
#     tokens 2048  ->  +13.953 GiB
#
# against +1.895 GiB for the same shape as an 8-bit affine layer, which is the
# logits array itself and carries no weight-dequant term at all.
#
# mlx-vlm's chunked prefill discards the chunk forward's return value, so MLX
# never evaluates that matmul and the cliff stays hidden — but chunking only
# engages above `prefill_step_size` (default 2048). A prompt of 257..2048
# tokens takes the single-shot branch in `generate/ar.py`, which slices
# `logits[:, -1, :]` and so DOES evaluate lm_head over the whole sequence.
# That is an ordinary chat turn, not an edge case.
VLM_SKIP_PATTERNS: dict[str, tuple[str, ...]] = {
    "diffusion_gemma": ("router", "self_conditioning", "embed_vision", ".mlp."),
    "muse_glimmer": ("lm_head",),
    "qwen3_5": ("lm_head",),
}


def _patch_muse_glimmer_normed_embedding() -> None:
    """Teach Muse Glimmer's ``NormedEmbedding`` how to quantize itself.

    Muse Glimmer's ``embed_tokens`` is an ``nn.Embedding`` subclass that
    RMS-normalizes the row it looks up. It inherits ``nn.Embedding.to_quantized``,
    which returns a *plain* ``QuantizedEmbedding`` — silently dropping the
    normalization and corrupting every activation downstream. mlx-vlm's own
    ``quant_predicate`` avoids that by refusing to quantize the module at all,
    which leaves 1.35B params (2.7 GB at bf16) uncompressed and is most of the
    reason a "4-bit" Muse Glimmer weighs 21.4 GB.

    Returning a norm-preserving subclass instead keeps the maths bit-for-bit
    identical and makes the module compressible. Both ``convert_vlm`` and
    ``load_turboquant_vlm`` reach the embedding through ``nn.quantize``, so
    patching ``to_quantized`` fixes the write and read sides at once.

    Only needed for the pre-merge layout. mlx-vlm 0.6.12 shipped
    [#1838](https://github.com/Blaizzy/mlx-vlm/pull/1838) with the norm hoisted
    out of the embedding — ``TextModel`` holds a paramless ``embed_norm`` and
    applies it in ``__call__``, so a plain ``QuantizedEmbedding`` already keeps
    the maths right and there is nothing to patch. The on-disk key set is the
    same either way (``RMSNormNoScale`` has no parameters), so models converted
    against either layout load against both.

    Idempotent, and a no-op if mlx-vlm has no muse_glimmer module or already
    applies the norm itself.
    """
    try:
        from mlx_vlm.models.muse_glimmer import language as _mg
    except ImportError:
        return
    # >=0.6.12: no NormedEmbedding to patch — the norm lives in TextModel.
    if not hasattr(_mg, "NormedEmbedding"):
        return
    if getattr(_mg.NormedEmbedding, "_tq_patched", False):
        return

    import mlx.core as mx
    import mlx.nn as nn

    class QuantizedNormedEmbedding(nn.QuantizedEmbedding):
        """QuantizedEmbedding that keeps NormedEmbedding's RMS normalization."""

        def __call__(self, x):
            return self.embed_norm(super().__call__(x))

    def to_quantized(self, group_size=None, bits=None, mode="affine",
                     quantize_input=False):
        if quantize_input:
            raise ValueError("Quantized input is not supported.")
        num_embeddings, dims = self.weight.shape
        ql = QuantizedNormedEmbedding(num_embeddings, dims, group_size, bits,
                                      mode=mode)
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None
        # Paramless RMSNormNoScale — carries only `eps`, so sharing is safe.
        ql.embed_norm = self.embed_norm
        return ql

    _mg.NormedEmbedding.to_quantized = to_quantized
    _mg.QuantizedNormedEmbedding = QuantizedNormedEmbedding
    _mg.NormedEmbedding._tq_patched = True


def patch_vlm_arch(arch: str) -> None:
    """Apply per-architecture fixes needed before quantizing OR loading.

    Must run on both sides: the converter decides what ``nn.quantize`` produces,
    and the loader has to rebuild the very same classes.
    """
    if arch in ("muse_glimmer", "muse_glimmer_text"):
        _patch_muse_glimmer_normed_embedding()


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

    patch_vlm_arch(config.get("model_type", "").lower())

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

    # Checkpoints also carry affine-quantized modules — the embeddings, lm_head
    # and, on a VLM, the entire vision tower. Those have `.scales` but no
    # `.codebook`, so `_prepare_polar_layers` never sees them.
    #
    # Recovered from the tensor shapes rather than read from the config's
    # `affine_extras` block. That block is not guaranteed to be present, and the
    # old `if affine:` meant a checkpoint without it skipped affine preparation
    # entirely: `load_weights(strict=False)` then dropped every scales and biases
    # in silence and the modules returned packed uint32 instead of floats. Shapes
    # are always there, so this cannot be skipped by a missing key.
    prepared_affine = _prepare_affine_extras(model, weights)
    _assert_no_orphan_quant_params(model, weights, prepared_affine)

    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())
    model.model_path = model_path
    model.eval()

    processor = load_processor(model_path)
    return model, processor, config
