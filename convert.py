"""CLI tool to convert HuggingFace models to TurboQuant-compressed MLX format.

Usage:
    python -m turboquant_mlx.convert \\
        --hf-path meta-llama/Llama-3.2-1B \\
        --mlx-path ./llama-3.2-1b-tq3 \\
        --bits 3 --group-size 64
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from turboquant_mlx.compat import is_local_kimi_k3
from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize


def _load_tiers(path: str) -> dict:
    """Read a ``{layer: tier}`` JSON file for --expert-layer-tiers."""
    import json

    with open(path) as f:
        data = json.load(f)
    return data.get("tiers", data)


def _parse_layer_list(text: str) -> list:
    """Parse a layer list for --protect-expert-layers: '0-5,42-47' or '0,1,47'.

    Ranges are inclusive. Raises argparse.ArgumentTypeError on malformed input
    so the CLI reports a clean usage error instead of a traceback.
    """
    import argparse

    layers = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo > hi or lo < 0:
                    raise ValueError
                layers.update(range(lo, hi + 1))
            else:
                v = int(part)
                if v < 0:
                    raise ValueError
                layers.add(v)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"bad layer spec {part!r}: use indices and inclusive ranges, "
                f"e.g. '0-5,42-47'")
    if not layers:
        raise argparse.ArgumentTypeError("empty layer list")
    return sorted(layers)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    bits: int = 3,
    group_size: int = 64,
    rotation: str = "hadamard",
    rotation_seed: int = 42,
    use_qjl: bool = False,
    dtype: str = None,
    attn_bits: int = None,
    mlp_bits: int = None,
    mlp_group_size: int = None,
    ternary_experts: bool = False,
    expert_down_bits: int = None,
    keep_mtp: bool = False,
    quantize_extras: bool = False,
    extras_bits: int = 4,
    extras_group_size: int = 64,
    protect_expert_layers: list = None,
    protect_bits: int = 3,
    expert_layer_tiers: dict = None,
):
    """Convert a HuggingFace model to TurboQuant-compressed MLX format.

    Args:
        hf_path: HuggingFace model path or local path.
        mlx_path: Output directory for the MLX model.
        bits: Quantization bit-width (2, 3, or 4).
        group_size: Quantization group size.
        rotation: Rotation method ("hadamard", "blockwise_hadamard", "none").
        rotation_seed: Random seed for rotation signs.
        use_qjl: Whether to enable QJL residual correction.
        dtype: Optional dtype override ("float16", "bfloat16", "float32").
        mlp_group_size: Optional finer group size for MoE expert tensors.
        ternary_experts: If True, quantize routed MoE experts to the ternary
            {-c,0,+c} codebook packed as base-3 trits (~1.6 bpw).
        keep_mtp: If True, copy the source model's multi-token-prediction head
            into the output. mlx-lm's ``sanitize()`` drops every ``mtp.*`` key,
            so without this the head cannot survive conversion at all. The head
            is copied **unquantized, at its source dtype** (810 MiB on
            Qwen3.8-27B, ~1.9% of the bf16 model) — quantizing it is a separate
            decision that wants its own quality gate. Off by default because
            nothing in the decode path consumes it; see ``turboquant_mlx.mtp``.
            No-op if the source has no head.
        quantize_extras: If True, quantize the bf16 remainder the polar path
            never touches -- ``nn.Embedding`` above all -- to MLX affine at
            ``extras_bits``/``extras_group_size``. Off by default because on a
            dense model the remainder is a token embedding and an ``lm_head``.
            It is **not** optional on models that keep a large lookup table:
            Qwen3.8-Flash-Next's sharded n-gram/PLE table is 51.2B params
            (95.4 GiB bf16, 28% of the model), so without this a ternary-expert
            build is ~124 GiB instead of ~54 GiB. Routers and the QSA block
            indexer stay full precision either way.
        extras_bits: Bit-width for the affine extras tier.
        extras_group_size: Group size for the affine extras tier.
        protect_expert_layers: Layer indices whose routed experts use a
            ``protect_bits`` Gaussian codebook instead of the expert tier
            (ternary or mlp_bits). See TurboQuantConfig.protect_expert_layers.
        protect_bits: Codebook width for protected expert layers.
        expert_layer_tiers: ``{layer: "ternary" | "2" | "3" | "4"}``, a hand-chosen
            tier per MoE layer. See TurboQuantConfig.expert_layer_tiers.
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
        use_qjl=use_qjl,
        attn_bits=attn_bits,
        mlp_bits=mlp_bits,
        mlp_group_size=mlp_group_size,
        ternary_experts=ternary_experts,
        expert_down_bits=expert_down_bits,
        protect_expert_layers=protect_expert_layers,
        protect_bits=protect_bits,
        expert_layer_tiers=expert_layer_tiers,
    )

    # Load model
    print(f"[INFO] Loading model from {hf_path}")
    model, tokenizer, config = load(
        hf_path,
        # K3's custom tokenizer class (tokenization_kimi.py) needs the code
        # opt-in; gated on a local kimi_k3 config so no other model gets it.
        tokenizer_config=(
            {"trust_remote_code": True} if is_local_kimi_k3(hf_path) else {}
        ),
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
    if tq_config.is_hybrid:
        eff_attn = tq_config.attn_bits if tq_config.attn_bits is not None else tq_config.bits
        eff_mlp = tq_config.mlp_bits if tq_config.mlp_bits is not None else tq_config.bits
        print(f"[INFO] Quantizing with TurboQuant (hybrid: attn={eff_attn}b mlp={eff_mlp}b default={bits}b, gs={group_size}, rotation={rotation})")
    else:
        print(f"[INFO] Quantizing with TurboQuant ({bits}-bit, gs={group_size}, rotation={rotation})")
    print(f"[INFO] Architecture: {arch}")
    print(f"[INFO] Effective bits/weight: {tq_config.effective_bits:.2f}")

    t0 = time.time()
    model, config = turboquant_quantize(model, config, tq_config)
    mx.eval(model.parameters())
    t1 = time.time()

    print(f"[INFO] Quantization completed in {t1 - t0:.1f}s")

    if quantize_extras:
        from turboquant_mlx.quantize_model import quantize_affine_extras

        n_extra = quantize_affine_extras(
            model, config, bits=extras_bits, group_size=extras_group_size
        )
        print(f"[INFO] Quantized {n_extra} extra modules to {extras_bits}-bit "
              f"affine g{extras_group_size} (embeddings and any layer the polar "
              f"path skipped; routers and the QSA indexer excluded)")

    # Save
    print(f"[INFO] Saving to {mlx_path}")
    save(mlx_path, hf_path, model, tokenizer, config)

    # Must run AFTER save(). mlx-lm's qwen3_5 sanitize() drops every `mtp.*` key,
    # and load() above already applied it, so the head is not in `model` and
    # cannot be saved from it -- it has to be copied from the source shards.
    if keep_mtp:
        from turboquant_mlx.mtp import preserve_mtp

        n, nbytes = preserve_mtp(hf_path, mlx_path)
        if n:
            print(f"[INFO] Preserved MTP head: {n} tensors, "
                  f"{nbytes / 1024**2:.0f} MiB (source dtype, not quantized)")
        else:
            print("[INFO] --keep-mtp requested but the source has no MTP head")

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
    parser.add_argument(
        "--attn-bits",
        type=int, default=None, choices=[2, 3, 4],
        help="Override bits for attention-block linears (q/k/v/o_proj). "
             "Defaults to --bits when omitted.",
    )
    parser.add_argument(
        "--mlp-bits",
        type=int, default=None, choices=[2, 3, 4],
        help="Override bits for MLP and MoE expert linears. "
             "Defaults to --bits when omitted.",
    )
    parser.add_argument(
        "--mlp-group-size",
        type=int, default=None, choices=[16, 32, 64, 128],
        help="Override group size (block-scale granularity) for MLP/MoE expert "
             "linears, so a sub-2-bit expert tier can carry a finer scale than "
             "attention. Defaults to --group-size when omitted.",
    )
    parser.add_argument(
        "--ternary-experts",
        action="store_true",
        help="Quantize MoE expert weights to the ternary {-c,0,+c} codebook "
             "(1.58-bit), stored in the 2-bit slot. The zero level clears the "
             "1-bit cardinality wall; data-free sub-2-bit expert tier (tq2a-tqTe). "
             "Attention stays at --attn-bits/--bits.",
    )
    parser.add_argument(
        "--expert-down-bits",
        type=int, default=None, choices=[2, 3, 4],
        help="Asymmetric expert precision: quantize MoE expert down "
             "projections at this Gaussian-codebook width while up/gate take "
             "the --mlp-bits / --ternary-experts tier. The down projection is "
             "the SwiGLU summation bottleneck; llama.cpp-family 2-bit mixes "
             "keep it above up/gate for exactly this reason.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Memory-bounded conversion: write each quantized layer to a shard "
             "and free it, so the full quantized model never resides in RAM. "
             "Use for models too big to convert in memory (e.g. 200B+ on 64 GB). "
             "(--dtype override is not supported in this mode.)",
    )
    parser.add_argument(
        "--protect-expert-layers",
        type=_parse_layer_list, default=None,
        help="Layer indices whose routed experts use a --protect-bits Gaussian "
             "codebook instead of the expert tier (ternary or --mlp-bits). "
             "Comma list with inclusive ranges, e.g. '0-5,42-47'. Changes bit "
             "width only, never the expert group size, so the loader needs no "
             "matching rule. Warns if a listed layer matches nothing.",
    )
    parser.add_argument(
        "--expert-layer-tiers",
        type=_load_tiers, default=None, metavar="JSON",
        help="Per-layer expert tiers: a JSON file mapping MoE layer index to "
             "'ternary', '2', '3' or '4' (or {\"tiers\": {...}}). Layers not "
             "listed keep the expert tier. Exclusive with --protect-expert-layers and "
             "--expert-down-bits.",
    )
    parser.add_argument(
        "--protect-bits",
        type=int, default=3, choices=[2, 3, 4],
        help="Codebook width for --protect-expert-layers (default 3).",
    )
    parser.add_argument(
        "--quantize-extras",
        action="store_true",
        help="Also quantize the bf16 remainder the polar path never touches -- "
             "nn.Embedding above all -- to MLX affine. Off by default because on "
             "a dense model that remainder is just a token embedding and an "
             "lm_head. REQUIRED on models with a large lookup table: "
             "Qwen3.8-Flash-Next's sharded n-gram/PLE table is 51.2B params "
             "(95.4 GiB bf16, 28%% of the model), so without this a "
             "ternary-expert build is ~124 GiB instead of ~54 GiB. Routers and "
             "the QSA block indexer stay full precision either way.",
    )
    parser.add_argument(
        "--extras-bits",
        type=int, default=4, choices=[2, 3, 4, 8],
        help="Bit-width for --quantize-extras (default 4).",
    )
    parser.add_argument(
        "--extras-group-size",
        type=int, default=64, choices=[32, 64, 128],
        help="Group size for --quantize-extras (default 64).",
    )
    parser.add_argument(
        "--keep-mtp",
        action="store_true",
        help="Copy the source model's multi-token-prediction head into the "
             "output, for self-speculative decoding. mlx-lm discards these "
             "tensors in sanitize(), so without this flag the head is lost. "
             "Opt-in because it adds the head at its source dtype (810 MiB on "
             "Qwen3.8-27B) and nothing consumes it yet. No-op if the source has "
             "no MTP head.",
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    if args.streaming:
        if args.dtype is not None:
            print("[WARNING] --dtype is ignored in --streaming mode")
        from turboquant_mlx.convert_streaming import convert_streaming
        convert_streaming(
            hf_path=args.hf_path,
            mlx_path=args.mlx_path,
            bits=args.bits,
            group_size=args.group_size,
            rotation=args.rotation,
            rotation_seed=args.rotation_seed,
            use_qjl=args.use_qjl,
            attn_bits=args.attn_bits,
            mlp_bits=args.mlp_bits,
            mlp_group_size=args.mlp_group_size,
            ternary_experts=args.ternary_experts,
            expert_down_bits=args.expert_down_bits,
            quantize_extras=args.quantize_extras,
            extras_bits=args.extras_bits,
            extras_group_size=args.extras_group_size,
            protect_expert_layers=args.protect_expert_layers,
            protect_bits=args.protect_bits,
            expert_layer_tiers=args.expert_layer_tiers,
        )
        # Applied here rather than threaded through convert_streaming: the head
        # is copied from the source shards after the fact either way, so the
        # streaming path needs no knowledge of it.
        if args.keep_mtp:
            from turboquant_mlx.mtp import preserve_mtp

            n, nbytes = preserve_mtp(args.hf_path, args.mlx_path)
            print(f"[INFO] Preserved MTP head: {n} tensors, "
                  f"{nbytes / 1024**2:.0f} MiB (source dtype, not quantized)"
                  if n else
                  "[INFO] --keep-mtp requested but the source has no MTP head")
        return
    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        bits=args.bits,
        group_size=args.group_size,
        rotation=args.rotation,
        rotation_seed=args.rotation_seed,
        use_qjl=args.use_qjl,
        dtype=args.dtype,
        attn_bits=args.attn_bits,
        mlp_bits=args.mlp_bits,
        mlp_group_size=args.mlp_group_size,
        ternary_experts=args.ternary_experts,
        expert_down_bits=args.expert_down_bits,
        keep_mtp=args.keep_mtp,
        quantize_extras=args.quantize_extras,
        extras_bits=args.extras_bits,
        extras_group_size=args.extras_group_size,
        protect_expert_layers=args.protect_expert_layers,
        protect_bits=args.protect_bits,
        expert_layer_tiers=args.expert_layer_tiers,
    )


if __name__ == "__main__":
    main()
