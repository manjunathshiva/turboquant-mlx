"""OpenAI-compatible server for TurboQuant-compressed mlx-vlm models.

``turboquant-serve`` wraps ``mlx_lm.server``, which knows nothing about
multimodal architectures. mlx-vlm ships its own, considerably richer server
(OpenAI *and* Anthropic routes, per-model tool parsers, continuous batching,
speculative decoding), so this module drives that one instead of reimplementing
it — it only teaches it the three things it cannot know about a TurboQuant VLM:

1. **How to load the checkpoint.** mlx-vlm's server reaches the model through
   exactly one seam: ``mlx_vlm.server.generation.load_model_resources`` calls
   ``load(...)``, which ``generation`` imported into its own namespace. Binding
   that name to a TurboQuant-aware wrapper is enough for the whole stack.

2. **Where the reasoning channel ends.** Muse Glimmer answers in a harmony-style
   channel format, and mlx-vlm's ATEM parser only strips that envelope when a
   tool call was parsed — so ordinary turns leak the entire thinking channel
   into ``message.content``. mlx-vlm's generic thinking splitter handles it
   correctly once pointed at the right delimiters, so we supply them.

3. **What the reasoning knob is called.** The server sends OpenAI's
   ``reasoning_effort``; Muse Glimmer's chat template only reads
   ``reasoning_strength`` and otherwise deliberates at its ``high`` default.

Usage:
    turboquant-serve-vlm --model <tq-dir-or-repo> --port 8080
    turboquant-serve-vlm --model <tq-dir> --reasoning-strength low

Requires:  pip install "turboquant-mlx-full[vlm]"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("turboquant.serve_vlm")

# Per-architecture (thinking_start, thinking_end) delimiters.
#
# Muse Glimmer emits:
#     to=self<|message|> …reasoning… <|eom|><|start|>assistant to=user<|message|> …answer…
#
# Ending the span at the *routing header* rather than at `<|eom|>` is deliberate:
# it consumes the `<|start|>assistant to=user<|message|>` preamble too, so
# content begins at the first real answer token. Stopping at `<|eom|>` would
# leave that header at the front of every reply.
CHANNEL_DELIMITERS: Dict[str, Tuple[str, str]] = {
    "muse_glimmer": ("to=self<|message|>", "<|start|>assistant to=user<|message|>"),
    "muse_glimmer_text": (
        "to=self<|message|>",
        "<|start|>assistant to=user<|message|>",
    ),
}

# OpenAI's reasoning_effort vocabulary -> Muse Glimmer's reasoning_strength.
# Muse has no "off" level, so the floor is "low"; anything already valid for the
# template passes through untouched.
_EFFORT_TO_STRENGTH = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def map_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """Translate an OpenAI reasoning_effort into a reasoning_strength."""
    if not effort:
        return None
    return _EFFORT_TO_STRENGTH.get(str(effort).lower(), str(effort).lower())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def is_turboquant_checkpoint(model_path) -> bool:
    """True if `model_path` is a TurboQuant (polar) checkpoint directory."""
    config = _read_json(Path(model_path) / "config.json")
    return (config.get("quantization") or {}).get("mode") == "turboquant"


def model_architecture(model_path) -> Optional[str]:
    return _read_json(Path(model_path) / "config.json").get("model_type")


def chat_template_text(model_path) -> str:
    """The model's chat template, from either place it may be stored."""
    model_path = Path(model_path)
    jinja = model_path / "chat_template.jinja"
    if jinja.exists():
        try:
            return jinja.read_text()
        except OSError:
            return ""
    template = _read_json(model_path / "tokenizer_config.json").get("chat_template")
    return template if isinstance(template, str) else ""


def install_turboquant_loader() -> None:
    """Teach mlx-vlm's server to load TurboQuant checkpoints.

    Idempotent. Non-TurboQuant models keep taking mlx-vlm's own loader, so one
    server can still serve an ordinary mlx-vlm model.
    """
    import mlx_vlm.server.generation as generation

    if getattr(generation.load, "_turboquant_wrapper", False):
        return

    original_load = generation.load

    def load(model_path, adapter_path=None, **kwargs):
        from turboquant_mlx.generate import resolve_model_path
        from turboquant_mlx.integration.vlm import load_turboquant_vlm

        resolved = resolve_model_path(str(model_path))
        if is_turboquant_checkpoint(resolved):
            logger.info("loading TurboQuant checkpoint: %s", resolved)
            model, processor, _config = load_turboquant_vlm(resolved)
            return model, processor
        return original_load(model_path, adapter_path, **kwargs)

    load._turboquant_wrapper = True
    generation.load = load


def install_reasoning_strength_mapping(default_strength: Optional[str] = None) -> None:
    """Feed the chat template `reasoning_strength` instead of `reasoning_effort`.

    Only the name differs, so a request that already says
    ``reasoning_effort: "low"`` starts working rather than being silently
    dropped. `default_strength` applies when a request says nothing at all —
    which is the common case for agent harnesses, and the difference between a
    short answer and several hundred tokens of deliberation first.
    """
    from mlx_vlm.server.generation import GenerationArguments

    original = GenerationArguments.to_template_kwargs
    if getattr(original, "_turboquant_wrapper", False):
        return

    def to_template_kwargs(self):
        kwargs = original(self)
        strength = map_reasoning_effort(kwargs.get("reasoning_effort"))
        if strength is None:
            strength = default_strength
        if strength is not None:
            kwargs["reasoning_strength"] = strength
        return kwargs

    to_template_kwargs._turboquant_wrapper = True
    GenerationArguments.to_template_kwargs = to_template_kwargs


def channel_delimiters_for(model_path) -> Optional[Tuple[str, str]]:
    """The thinking delimiters this model needs, if we know of any."""
    return CHANNEL_DELIMITERS.get(model_architecture(model_path) or "")


def _extract_option(argv, name: str) -> Optional[str]:
    """Read `--name value` or `--name=value` out of an argv list."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _has_option(argv, name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def build_server_argv(argv, model_path) -> list:
    """Add the channel delimiters this architecture needs, if not already set.

    An explicit `--thinking-start-token` / `--thinking-end-token` always wins.
    """
    delimiters = channel_delimiters_for(model_path) if model_path else None
    if delimiters is None:
        return argv
    if _has_option(argv, "--thinking-start-token") or _has_option(
        argv, "--thinking-end-token"
    ):
        return argv

    start, end = delimiters
    logger.info(
        "%s: splitting the reasoning channel on %r -> %r",
        model_architecture(model_path),
        start,
        end,
    )
    return argv + ["--thinking-start-token", start, "--thinking-end-token", end]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--reasoning-strength",
        type=str,
        default=None,
        help="Default reasoning effort for models whose template takes one "
        "(Muse Glimmer: low/medium/high/xhigh, default high). Applies when a "
        "request does not ask for one. Every other flag is mlx-vlm's; see "
        "`python -m mlx_vlm.server --help`.",
    )
    args, server_argv = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    install_turboquant_loader()

    model_path = _extract_option(server_argv, "--model")
    resolved = None
    if model_path:
        try:
            from turboquant_mlx.generate import resolve_model_path

            resolved = resolve_model_path(model_path)
        except Exception:  # a repo id we cannot resolve offline, or a typo
            resolved = model_path

    default_strength = args.reasoning_strength
    if resolved and (
        default_strength is not None
        or "reasoning_strength" in chat_template_text(resolved)
    ):
        install_reasoning_strength_mapping(default_strength)

    if resolved:
        server_argv = build_server_argv(server_argv, resolved)

    from mlx_vlm.server.cli import main as server_main

    sys.argv = [sys.argv[0]] + server_argv
    return server_main()


if __name__ == "__main__":
    main()
