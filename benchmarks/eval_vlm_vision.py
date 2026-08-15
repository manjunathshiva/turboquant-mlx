"""Four-case vision battery for any mlx-vlm checkpoint, TurboQuant or affine.

OCR, counting, chart reading, spatial relations. Every image is drawn here, so
the ground truth is something we set rather than something a judge model
decides — the scoring is a substring match against an accept list, and the
whole thing is greedy, so a run is reproducible.

The engine is chosen from the checkpoint: a TurboQuant build goes through
`turboquant_mlx.generate_vlm`, anything else through `mlx_vlm.generate`. Same
images, same prompts, same accept lists either way, which is what makes a
TurboQuant-vs-affine comparison meaningful.

One caution learned the hard way: a single failing case is not a capability
claim. On Qwen3.8-27B the affine build missed `text.png` while TurboQuant
passed it, but a follow-up five-string probe put affine at 4/5 — the miss was
one string, not an OCR weakness. Treat a 3/4-vs-4/4 split as a tie until a
wider probe says otherwise.

Usage:
    python -m turboquant_mlx.benchmarks.eval_vlm_vision <model> [--json out.json]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# The same interpreter that launched this module, so the subprocess inherits
# the environment the caller actually set up.
PYTHON = sys.executable
OUT = Path(tempfile.mkdtemp(prefix="tq_vision_"))


def resolved_identity(spec):
    """A record of exactly which bytes were measured.

    Repo ids and dataset defaults are mutable — `mlx-community/X-4bit` can be
    re-uploaded and a published number would silently start describing
    different weights. For a Hub repo this pins the commit sha; for a local
    directory it records the path and the total size of its safetensors, which
    is enough to notice a swap.
    """
    from pathlib import Path as _P

    p = _P(spec)
    if p.is_dir():
        shards = sorted(p.glob("*.safetensors"))
        return {"spec": str(spec), "kind": "local",
                "safetensors_bytes": sum(f.stat().st_size for f in shards),
                "n_shards": len(shards)}
    try:
        from huggingface_hub import HfApi

        return {"spec": str(spec), "kind": "hub",
                "revision": HfApi().model_info(str(spec)).sha}
    except Exception as exc:  # offline, private, or not a repo id
        return {"spec": str(spec), "kind": "unresolved", "error": str(exc)[:120]}


def engine_for(model):
    """(module, temp_flag) — TurboQuant checkpoints need our loader, others mlx-vlm's.

    Same images, same prompts, same accept lists either way; only the entry
    point differs, because `generate_vlm` refuses non-TurboQuant checkpoints
    and `mlx_vlm.generate` cannot read a polar one.
    """
    from turboquant_mlx.serve_vlm import is_turboquant_checkpoint
    from turboquant_mlx.generate import resolve_model_path

    if is_turboquant_checkpoint(Path(resolve_model_path(str(model)))):
        return "turboquant_mlx.generate_vlm", "--temp"
    return "mlx_vlm.generate", "--temperature"


# Scalable fonts to draw the OCR and chart labels with, most-preferred first.
# Covers macOS and the usual Linux packages. Override with TQ_VISION_FONT.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)


def _font(size):
    """A scalable font at `size`, or a hard failure.

    Deliberately no `ImageFont.load_default()` fallback. That returns a small
    fixed-size bitmap face, so the text would render a few pixels tall and the
    OCR case would measure nothing — while still reporting a tidy PASS/FAIL as
    if the result meant something. A missing font is a setup problem and should
    look like one.

    Not bundled in the repo: shipping a font means carrying its licence and a
    few hundred KB for a benchmark most people never run. Point
    TQ_VISION_FONT at one instead.
    """
    explicit = os.environ.get("TQ_VISION_FONT")
    for path in ((explicit,) if explicit else ()) + _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit(
        "no scalable font found — the OCR and chart cases need one to render "
        "legible text. Set TQ_VISION_FONT=/path/to/font.ttf, or install e.g. "
        "fonts-dejavu. Looked in:\n  " + "\n  ".join(_FONT_CANDIDATES)
    )


def build_images():
    """Synthetic images with ground truth we drew, so scoring is objective."""
    OUT.mkdir(exist_ok=True)

    img = Image.new("RGB", (640, 320), "white")
    ImageDraw.Draw(img).text((60, 120), "VOLTAGE 47", fill="black", font=_font(72))
    img.save(OUT / "text.png")

    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    for i in range(3):
        x = 80 + i * 160
        d.ellipse([x, 60, x + 100, 160], fill="red")
    for i in range(2):
        x = 160 + i * 200
        d.rectangle([x, 240, x + 100, 340], fill="blue")
    img.save(OUT / "count.png")

    img = Image.new("RGB", (640, 420), "white")
    d = ImageDraw.Draw(img)
    f = _font(28)
    for i, (label, h) in enumerate([("A", 120), ("B", 200), ("C", 320)]):
        x = 100 + i * 160
        d.rectangle([x, 360 - h, x + 90, 360], fill="#3366cc")
        d.text((x + 30, 370), label, fill="black", font=f)
    d.line([80, 360, 600, 360], fill="black", width=3)
    img.save(OUT / "chart.png")

    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    d.polygon([(120, 60), (60, 170), (180, 170)], fill="green")
    d.ellipse([460, 240, 580, 360], fill="orange")
    img.save(OUT / "spatial.png")


CASES = [
    ("text.png", "What text is written in this image? Answer with just the text.",
     ["voltage 47", "voltage47"]),
    ("count.png", "How many red circles are in this image? Answer with just the number.",
     ["3", "three"]),
    ("chart.png", "Which bar is tallest: A, B, or C? Answer with just the letter.",
     ["c"]),
    ("spatial.png", "What shape is in the top-left of this image? Answer with one word.",
     ["triangle"]),
]

_NOISE = ("[INFO]", "[transformers]", "Prompt:", "Generation:", "Files:",
          "peak memory", "Peak memory", "==========", "<|im_start|>",
          "<|im_end|>", "Fetching")


def extract_answer(text):
    """The answer after the prompt echo and the `<think>` channel."""
    if "==========" in text:
        text = text.split("==========")[-2] if text.count("==========") >= 2 else text
    # Qwen3.5 templates open a reasoning channel; the answer follows it.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = text.replace("<think>", " ").replace("</think>", " ")
    if "assistant" in text:
        text = text.rsplit("assistant", 1)[1]
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith(_NOISE)]
    return " ".join(lines).strip()


def peak_gib(text):
    """Peak memory in GiB.

    Two reporters print a line and they disagree on units, distinguished only
    by capitalisation: mlx-vlm's `Peak memory:` is decimal GB, turboquant's
    lowercase `peak memory:` is already GiB. Prefer the lowercase one.
    """
    m = re.search(r"\bpeak memory:\s*([\d.]+)\s*GB", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\bPeak memory:\s*([\d.]+)\s*GB", text)
    if m:
        return float(m.group(1)) / 1.073741824
    return None


def run(model, max_tokens=128):
    build_images()
    module, temp_flag = engine_for(model)
    print(f"model: {model}   (via {module})\n")
    cases, peaks = [], []
    for name, prompt, accept in CASES:
        t0 = time.time()
        proc = subprocess.run(
            [PYTHON, "-m", module,
             "--model", str(model), "--image", str(OUT / name),
             "--prompt", prompt, "--max-tokens", str(max_tokens), temp_flag, "0"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # Never score a crash. A failed load or an OOM would otherwise be
            # extracted as a wrong "answer" and recorded as a failed case,
            # which reads as a model capability result rather than the setup
            # failure it is.
            raise SystemExit(
                f"{model}: `{module}` exited {proc.returncode} on case "
                f"{name!r} — this is an execution failure, not a wrong answer."
                f"\n--- stderr ---\n{proc.stderr[-2000:]}"
            )
        blob = proc.stdout + proc.stderr
        answer = extract_answer(blob)
        peak = peak_gib(blob)
        if peak:
            peaks.append(peak)
        ok = any(a in answer.lower() for a in accept)
        cases.append({"case": name, "passed": ok, "answer": answer[:200],
                      "accept": accept, "seconds": round(time.time() - t0, 1),
                      "peak_gib": peak})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<12} {cases[-1]['seconds']:5.1f}s  "
              f"peak {peak if peak else float('nan'):5.2f} GiB")
        print(f"         expected one of {accept} -> got {answer[:110]!r}")
    n_pass = sum(c["passed"] for c in cases)
    print(f"\nvision: {n_pass}/{len(CASES)} passed"
          + (f", peak {max(peaks):.2f} GiB" if peaks else ""))
    return {"model": str(model), "engine": module, "passed": n_pass,
            "total": len(CASES), "peak_gib": max(peaks) if peaks else None,
            "identity": resolved_identity(model), "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="+", help="checkpoint dirs or HF repo ids")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    results = [run(m, args.max_tokens) for m in args.models]
    if len(results) > 1:
        print(f"\n{'model':<44}{'vision':>10}{'peak GiB':>11}")
        print("-" * 65)
        for r in results:
            score = f"{r['passed']}/{r['total']}"
            print(f"{r['model']:<44}{score:>10}"
                  f"{r['peak_gib'] or float('nan'):>11.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if all(r["passed"] == r["total"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
