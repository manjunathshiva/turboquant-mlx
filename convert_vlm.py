"""Convert an mlx-vlm-architecture model (multimodal / diffusion LLM) to
TurboQuant-compressed MLX format.

For architectures that live in mlx-vlm rather than mlx-lm — e.g. Google's
DiffusionGemma (model_type ``diffusion_gemma``). The quantize core
(``turboquant_quantize``) is model-agnostic; this entry point only swaps the
load/save plumbing to mlx-vlm and applies per-architecture full-precision
skips (vision towers, routers, known quant-sensitive blocks).

Usage:
    python -m turboquant_mlx.convert_vlm \\
        --hf-path google/diffusiongemma-26B-A4B-it \\
        --mlx-path ./diffusiongemma-26B-A4B-it-tq3-g32 \\
        --bits 3 --group-size 32

Requires:  pip install "turboquant-mlx-full[vlm]"
"""

import argparse
import re
import shutil
import time
import types
from pathlib import Path

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
import turboquant_mlx.quantize_model as _qm
from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize, _detect_architecture
from turboquant_mlx.integration.vlm import _require_mlx_vlm, vlm_should_quantize

_AUX_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "processor_config.json", "generation_config.json",
    "preprocessor_config.json", "special_tokens_map.json",
)

# --quantize-extras: affine quantization for the non-TurboQuant remainder.
_EXTRAS_BITS = 8
_EXTRAS_GROUP = 64


def convert_vlm(
    hf_path: str,
    mlx_path: str,
    bits: int = 3,
    group_size: int = 32,
    rotation: str = "hadamard",
    rotation_seed: int = 42,
    attn_bits: int = None,
    mlp_bits: int = None,
    quantize_extras: bool = False,
    protect_expert_layers: list = None,
    protect_bits: int = 3,
):
    """Convert an mlx-vlm model to TurboQuant format. See module docstring."""
    _require_mlx_vlm()
    from mlx_vlm.utils import (
        get_model_path, load_config, load_model, save_config, save_weights,
    )

    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to {mlx_path} as it already exists. "
            "Delete it or specify a new path."
        )

    hf_path = Path(get_model_path(hf_path))

    tq_config = TurboQuantConfig(
        bits=bits,
        group_size=group_size,
        rotation=rotation,
        rotation_seed=rotation_seed,
        fuse_rotations=False,  # online rotation everywhere (production default)
        use_qjl=False,
        attn_bits=attn_bits,
        mlp_bits=mlp_bits,
    )

    if protect_expert_layers:
        # Layer protection: keep the experts of the listed layers at
        # `protect_bits` while the rest use the default `bits`. The cheapest
        # quality lift for 2-bit experts — the first/last layers carry the
        # most quant-sensitive expert work. The loader needs no knowledge of
        # this: per-layer bits are inferred from each saved codebook's size.
        protected = set(int(i) for i in protect_expert_layers)
        layer_rx = re.compile(r"\.layers\.(\d+)\.")
        base_bfp = tq_config.bits_for_path

        def _bits_for_path(self, path):
            if ".experts." in path:
                m = layer_rx.search(path)
                if m and int(m.group(1)) in protected:
                    return protect_bits
            return base_bfp(path)

        tq_config.bits_for_path = types.MethodType(_bits_for_path, tq_config)
        print(f"[INFO] Expert layer protection: layers {sorted(protected)} "
              f"-> {protect_bits}b")

    print(f"[INFO] Loading {hf_path} via mlx-vlm (lazy)")
    config = load_config(hf_path)
    model = load_model(hf_path, lazy=True)

    arch = _detect_architecture(config)
    print(f"[INFO] Architecture: {arch}")
    print(f"[INFO] Quantizing with TurboQuant ({bits}-bit, gs={group_size}, "
          f"rotation={rotation})")

    orig_should_quantize = _qm._should_quantize
    _qm._should_quantize = vlm_should_quantize(arch, orig_should_quantize)
    try:
        t0 = time.time()
        model, config = turboquant_quantize(model, config, tq_config)
        mx.eval(model.parameters())
        print(f"[INFO] Quantization completed in {time.time() - t0:.1f}s")
    finally:
        _qm._should_quantize = orig_should_quantize

    if quantize_extras:
        # Memory-constrained targets (e.g. 16 GB Mac mini): quantize the
        # remaining bf16 modules (embeddings, dense MLP, vision tower) to
        # mlx 8-bit affine. Polar layers are untouched (no `to_quantized`);
        # routers and self-conditioning stay full precision — both are tiny
        # and routing/conditioning fidelity matters more than their bytes.
        import mlx.nn as nn

        n_extra = [0]

        def _extras_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            if "router" in p or "self_conditioning" in p:
                return False
            if hasattr(m, "weight") and m.weight.shape[-1] % _EXTRAS_GROUP != 0:
                return False  # e.g. vision MLP down (4304-wide) stays bf16
            n_extra[0] += 1
            return True

        nn.quantize(model, group_size=_EXTRAS_GROUP, bits=_EXTRAS_BITS,
                    class_predicate=_extras_predicate)
        mx.eval(model.parameters())
        config["quantization"]["affine_extras"] = {
            "bits": _EXTRAS_BITS, "group_size": _EXTRAS_GROUP,
        }
        print(f"[INFO] Quantized {n_extra[0]} extra modules to "
              f"{_EXTRAS_BITS}-bit affine (embeddings/dense MLP/vision)")

    if protect_expert_layers:
        config["quantization"]["protected_expert_layers"] = sorted(
            int(i) for i in protect_expert_layers)
        config["quantization"]["protect_bits"] = protect_bits

    print(f"[INFO] Saving to {mlx_path}")
    save_weights(mlx_path, model, donate_weights=True)
    config.pop("quantization_config", None)
    save_config(config, mlx_path / "config.json")
    for fname in _AUX_FILES:
        src = hf_path / fname
        if src.exists():
            shutil.copy(src, mlx_path / fname)

    total = sum(f.stat().st_size for f in mlx_path.glob("*.safetensors"))
    print(f"[INFO] Done! Model saved to {mlx_path} ({total / 1024**3:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an mlx-vlm model to TurboQuant-compressed MLX format"
    )
    parser.add_argument("--hf-path", "--model", type=str, required=True,
                        help="HuggingFace model path or local path")
    parser.add_argument("--mlx-path", type=str, required=True,
                        help="Output directory for the MLX model")
    parser.add_argument("--bits", "-b", type=int, default=3, choices=[2, 3, 4],
                        help="Quantization bit-width (default: 3)")
    parser.add_argument("--group-size", "-g", type=int, default=32,
                        choices=[32, 64, 128], help="Group size (default: 32)")
    parser.add_argument("--rotation", type=str, default="hadamard",
                        choices=["hadamard", "blockwise_hadamard", "none"])
    parser.add_argument("--rotation-seed", type=int, default=42)
    parser.add_argument("--attn-bits", type=int, default=None, choices=[2, 3, 4],
                        help="Override bits for attention-block linears")
    parser.add_argument("--mlp-bits", type=int, default=None, choices=[2, 3, 4],
                        help="Override bits for MLP / MoE expert linears")
    parser.add_argument("--quantize-extras", action="store_true",
                        help="Also quantize embeddings, dense MLP and vision "
                             "tower to 8-bit affine (for memory-constrained "
                             "machines, e.g. 16 GB Macs)")
    parser.add_argument("--protect-expert-layers", type=str, default=None,
                        help="Comma-separated layer indices whose EXPERTS keep "
                             "--protect-bits instead of --bits (quality lift "
                             "for 2-bit experts, e.g. '0,1,2,27,28,29')")
    parser.add_argument("--protect-bits", type=int, default=3,
                        choices=[3, 4],
                        help="Bit width for protected expert layers (default 3)")
    args = parser.parse_args()

    convert_vlm(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        bits=args.bits,
        group_size=args.group_size,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        attn_bits=args.attn_bits,
        mlp_bits=args.mlp_bits,
        quantize_extras=args.quantize_extras,
        protect_expert_layers=(
            [int(i) for i in args.protect_expert_layers.split(",")]
            if args.protect_expert_layers else None),
        protect_bits=args.protect_bits,
    )


if __name__ == "__main__":
    main()
