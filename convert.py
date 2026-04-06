"""CLI tool to convert HuggingFace models to TurboQuant-compressed MLX format.

Usage:
    python -m turboquant_mlx.convert \\
        --hf-path meta-llama/Llama-3.2-1B \\
        --mlx-path ./llama-3.2-1b-tq3 \\
        --bits 3 --group-size 64
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    bits: int = 3,
    group_size: int = 64,
    rotation: str = "hadamard",
    rotation_seed: int = 42,
    fuse_rotations: bool = True,
    use_qjl: bool = False,
    dtype: str = None,
):
    """Convert a HuggingFace model to TurboQuant-compressed MLX format.

    Args:
        hf_path: HuggingFace model path or local path.
        mlx_path: Output directory for the MLX model.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Quantization group size.
        rotation: Rotation method ("hadamard", "blockwise_hadamard", "none").
        rotation_seed: Random seed for rotation signs.
        fuse_rotations: Whether to fuse rotations into norm weights.
        use_qjl: Whether to enable QJL residual correction.
        dtype: Optional dtype override ("float16", "bfloat16", "float32").
    """
    from mlx_lm.utils import load, save

    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to {mlx_path} as it already exists. "
            "Delete it or specify a new path."
        )

    tq_config = TurboQuantConfig(
        bits=bits,
        group_size=group_size,
        rotation=rotation,
        rotation_seed=rotation_seed,
        fuse_rotations=fuse_rotations,
        use_qjl=use_qjl,
    )

    # Load model
    print(f"[INFO] Loading model from {hf_path}")
    model, tokenizer, config = load(
        hf_path,
        return_config=True,
        lazy=True,
    )

    # Apply dtype if specified
    if dtype is not None:
        target_dtype = getattr(mx, dtype)
        model.update(
            {k: v.astype(target_dtype) for k, v in model.parameters().items()
             if mx.issubdtype(v.dtype, mx.floating)}
        )

    # Quantize
    arch = config.get("model_type", "unknown")
    print(f"[INFO] Quantizing with TurboQuant ({bits}-bit, gs={group_size}, rotation={rotation})")
    print(f"[INFO] Architecture: {arch}")
    print(f"[INFO] Effective bits/weight: {tq_config.effective_bits:.2f}")

    t0 = time.time()
    model, config = turboquant_quantize(model, config, tq_config)
    mx.eval(model.parameters())
    t1 = time.time()

    print(f"[INFO] Quantization completed in {t1 - t0:.1f}s")

    # Save
    print(f"[INFO] Saving to {mlx_path}")
    save(mlx_path, hf_path, model, tokenizer, config)

    # Print summary
    from mlx.nn.utils import tree_flatten
    leaves = tree_flatten(model.parameters())
    total_params = sum(v.size for _, v in leaves)
    total_bytes = sum(v.nbytes for _, v in leaves)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Model size: {total_bytes / 1024**3:.2f} GB")
    print(f"[INFO] Done! Model saved to {mlx_path}")


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to TurboQuant-compressed MLX format"
    )
    parser.add_argument(
        "--hf-path", "--model",
        type=str, required=True,
        help="HuggingFace model path or local path",
    )
    parser.add_argument(
        "--mlx-path",
        type=str, default="mlx_model",
        help="Output directory for MLX model (default: mlx_model)",
    )
    parser.add_argument(
        "--bits", "-b",
        type=int, default=3, choices=[2, 3, 4],
        help="Quantization bit-width (default: 3)",
    )
    parser.add_argument(
        "--group-size", "-g",
        type=int, default=64, choices=[32, 64, 128],
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--rotation",
        type=str, default="hadamard",
        choices=["hadamard", "blockwise_hadamard", "none"],
        help="Rotation method (default: hadamard)",
    )
    parser.add_argument(
        "--rotation-seed",
        type=int, default=42,
        help="Random seed for rotation signs (default: 42)",
    )
    parser.add_argument(
        "--fuse-rotations",
        action="store_true",
        help="Fuse rotations into normalization weights (experimental, may degrade quality)",
    )
    parser.add_argument(
        "--use-qjl",
        action="store_true",
        help="Enable QJL 1-bit residual correction (adds ~1 bit overhead)",
    )
    parser.add_argument(
        "--dtype",
        type=str, default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype before quantization",
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        bits=args.bits,
        group_size=args.group_size,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        fuse_rotations=args.fuse_rotations,
        use_qjl=args.use_qjl,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
