"""Generate with a TurboQuant-compressed mlx-vlm model (multimodal/diffusion).

Loads the model via turboquant_mlx.integration.vlm (PolarQuantized layers),
then runs mlx-vlm's generation dispatch — for diffusion architectures such as
DiffusionGemma that is the block-diffusion denoising sampler.

Usage:
    python -m turboquant_mlx.generate_vlm \\
        --model manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32 \\
        --prompt "Write a short paragraph about the ocean." \\
        --max-tokens 256 --temp 0.0

Requires:  pip install "turboquant-mlx-full[vlm]"
"""

import argparse
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from turboquant_mlx.generate import resolve_model_path
from turboquant_mlx.integration.vlm import _require_mlx_vlm, load_turboquant_vlm


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a TurboQuant-compressed mlx-vlm model"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="TurboQuant model directory or HF repo ID")
    parser.add_argument("--prompt", type=str,
                        default="Write a short paragraph about the ocean.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", "--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional image path/URL for multimodal prompts")
    parser.add_argument("--max-denoising-steps", type=int, default=None,
                        help="Cap diffusion denoising steps (model default 48)."
                             " Lower = faster, mild quality cost (try 24)")
    parser.add_argument("--max-canvas-length", type=int, default=None,
                        help="Cap the diffusion canvas length (model default "
                             "256). Lower = smaller activation memory, useful "
                             "on 16 GB machines (try 128)")
    args = parser.parse_args()

    _require_mlx_vlm()
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path = resolve_model_path(args.model)

    t0 = time.time()
    model, processor, config = load_turboquant_vlm(model_path)
    print(f"[INFO] Loaded in {time.time() - t0:.1f}s")

    num_images = 1 if args.image else 0
    formatted = apply_chat_template(processor, config, args.prompt,
                                    num_images=num_images)
    gen_kwargs = {}
    if args.max_denoising_steps is not None:
        gen_kwargs["max_denoising_steps"] = args.max_denoising_steps
    if args.max_canvas_length is not None:
        gen_kwargs["diffusion_max_canvas_length"] = args.max_canvas_length
    generate(
        model, processor, formatted,
        image=[args.image] if args.image else None,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        verbose=True,
        **gen_kwargs,
    )
    print(f"peak memory: {mx.get_peak_memory() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
