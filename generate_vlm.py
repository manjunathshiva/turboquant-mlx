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


def _positive_int(value: str) -> int:
    """An int > 0, rejected at parse time rather than deep inside prefill.

    mlx-vlm's chunked-prefill loop is
    ``while inputs_embeds.shape[1] > 1: n = min(step, len - 1); embeds = embeds[:, n:]``.
    At ``step=0`` that slices nothing off, the length never falls, and the loop
    spins forever; a negative step slices from the wrong end. mlx-vlm guards
    this in its diffusion path (``prefill_step_size must be a positive
    integer``) but not in the autoregressive one, so guard it here.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {ivalue}"
        )
    return ivalue


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
    parser.add_argument("--prefill-step-size", type=_positive_int, default=None,
                        help="Tokens per prefill chunk (mlx-vlm default 2048). "
                             "This is the memory knob for long prompts: the "
                             "transient prefill workspace scales with the "
                             "chunk, and a chunk of 256 or less also keeps "
                             "every TurboQuant layer on the fused Metal "
                             "kernel, which materializes nothing. "
                             "`turboquant-plan` recommends a value when the "
                             "default would not fit.")
    parser.add_argument("--max-denoising-steps", type=int, default=None,
                        help="Cap diffusion denoising steps (model default 48)."
                             " Lower = faster, mild quality cost (try 24)")
    parser.add_argument("--max-canvas-length", type=int, default=None,
                        help="Cap the diffusion canvas length (model default "
                             "256). Lower = smaller activation memory, useful "
                             "on 16 GB machines (try 128)")
    parser.add_argument("--reasoning", type=str, default=None,
                        metavar="LEVEL",
                        help="Reasoning effort for models whose chat template "
                             "takes one (Muse Glimmer: low/medium/high/xhigh, "
                             "default high). Lower spends far fewer tokens "
                             "thinking before answering — worth a lot on slow "
                             "machines. Ignored by templates without it.")
    parser.add_argument("--no-think", action="store_true",
                        help="Minimise reasoning: shorthand for the lowest "
                             "effort the template supports, plus "
                             "`enable_thinking=False` for Qwen-style "
                             "templates. NOTE this does not guarantee zero "
                             "reasoning — Muse Glimmer has no 'off' level and "
                             "may still emit a short thinking channel.")
    args = parser.parse_args()

    _require_mlx_vlm()
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model_path = resolve_model_path(args.model)

    t0 = time.time()
    model, processor, config = load_turboquant_vlm(model_path)
    print(f"[INFO] Loaded in {time.time() - t0:.1f}s")

    num_images = 1 if args.image else 0

    # Chat-template knobs for reasoning effort. Templates differ: Qwen-style
    # ones take `enable_thinking`, Muse Glimmer takes `reasoning_strength`
    # (low/medium/high/xhigh, defaulting to **high** — which spends hundreds of
    # tokens deliberating before a short answer, and at a few tok/s that is most
    # of the wall clock). Unknown keys are ignored by a template that does not
    # reference them, so passing both is safe.
    template_kwargs = {}
    if args.reasoning:
        template_kwargs["reasoning_strength"] = args.reasoning
    if args.no_think:
        template_kwargs.setdefault("reasoning_strength", "low")
        template_kwargs["enable_thinking"] = False
    if template_kwargs:
        print(f"[INFO] chat template: {template_kwargs}")

    formatted = apply_chat_template(processor, config, args.prompt,
                                    num_images=num_images, **template_kwargs)
    gen_kwargs = {}
    if args.prefill_step_size is not None:
        gen_kwargs["prefill_step_size"] = args.prefill_step_size
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
