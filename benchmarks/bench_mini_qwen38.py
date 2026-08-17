"""Run the Qwen3.8-27B 3-bit build on a 16 GB Mac mini. Self-contained.

Why this exists
---------------
Everything published about `manjunathshiva/Qwen3.8-27B-tq3-mini-g64` was first
measured on a 64 GB M4 Max, which made the 16 GB claim an inference —
peak-under-cap — rather than a reproduction. This script settles it on the
actual hardware.

**It has now been run, and it passes.** See `bench_mini_qwen38_results.json` for
the full record (16 GB M4 Mac mini, macOS 26.5.2):

    prompt      peak        vs 64 GB M4 Max    prefill      decode
      220    12.95 GiB          -0.27          20.7 t/s     3.9 t/s
      818    12.96 GiB          -0.09          19.0 t/s     3.9 t/s
    2,014    13.14 GiB          -0.16          21.0 t/s     3.6 t/s
    5,017    13.61 GiB          -0.20          20.6 t/s     3.7 t/s

    vision   700x200  12.68 GiB  OCR correct
             350x100  12.44 GiB  OCR correct

Read the delta column as approximate, not as an A/B. The 64 GB reference sweep
realized 246 / 844 / 2,040 / 5,043 prompt tokens where this one realizes 220 /
818 / 2,014 / 5,017 — a constant 26-token offset, worth 1.62 MiB of KV at this
model's 64 KiB/token. That is 55-170x smaller than the deltas below, so the
finding holds, but these are not byte-identical prompts.

Two findings worth keeping. Peaks land **below** the comparable prompts on the
64 GB machine, so the big Mac is the pessimistic estimator — and peak reproduces
to the hundredth of a GiB across independent sweeps while prefill moves ~7%, so
quote peak and decode flat but give prefill a range.

**An out-of-memory failure is still a perfectly good result** if you are
adapting this for another build. Whatever happens, the JSON it writes is the
real data. One caveat if you see a kill: the *first* load after boot faults
11.55 GiB off disk while Metal wires the same pages, and on a 16 GB machine
that transient double-count can trip the kernel's memory killer (exit -9). Re-run
it; the second start is warm. That is exactly what happened here on the first
attempt, and every length passed on the repeat.

Usage (on the mini)
-------------------
    pip install "turboquant-mlx-full[vlm]>=0.24.0" pillow
    sudo sysctl -w iogpu.wired_limit_mb=14336
    python3 bench_mini_qwen38.py

Quit Chrome and everything else first. Headroom at the longest prompt is about
0.19 GiB — other apps are not background noise at that margin.

Note on the cap: `turboquant-plan` suggests 13863, projecting the workspace at
8K context. That is 13.54 GiB, and both machines overrun it at 5K tokens — the
mini's 13.61 GiB by 0.07, the 64 GB M4 Max's 13.81 GiB by 0.27. Use 14336.
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("Q38_MODEL", "manjunathshiva/Qwen3.8-27B-tq3-mini-g64")
OUT = Path(__file__).parent / "mini_qwen38_results"
STEP = "256"                      # mandatory: keeps the fused kernel on
CAP_GIB = 14336 / 1024            # what the docs tell you to set

# Reference peaks from the 64 GB M4 Max, so the mini numbers have something to
# sit against in the writeup. Keyed by loop target, NOT by realized prompt
# tokens: that sweep landed on 246/844/2040/5043 tokens where this one lands on
# 220/818/2014/5017, a constant 26-token offset. At 64 KiB/token that gap is
# 1.62 MiB, so it cannot account for deltas measured in tenths of a GiB -- but
# the comparison is approximate and the printed column says so.
REFERENCE = {246: 13.22, 844: 13.05, 2040: 13.30, 5043: 13.81}
REFERENCE_TOKENS = {246: 246, 844: 844, 2040: 2040, 5043: 5043}
REFERENCE_NOTE = (
    "reference_peaks_64gb is keyed by loop target. The 64 GB sweep realized "
    "246/844/2040/5043 prompt tokens where this one realizes 220/818/2014/5017 "
    "- a constant 26-token offset, 1.62 MiB of KV at 64 KiB/token. That is "
    "55-170x smaller than the measured peak deltas, so the comparison holds, "
    "but the prompts are not byte-identical."
)
MUSE_MINI_SURVIVED = 14.21        # highest peak a 16 GB mini is known to survive


def _run(args):
    return subprocess.run([sys.executable, "-m", *args],
                          capture_output=True, text=True)


def peak_gib(text):
    """turboquant prints `peak memory:` in GiB; mlx-vlm's `Peak memory:` is decimal GB."""
    m = re.search(r"\bpeak memory:\s*([\d.]+)\s*GB", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\bPeak memory:\s*([\d.]+)\s*GB", text)
    return float(m.group(1)) / 1.073741824 if m else None


def rate(text, label):
    m = re.search(rf"{label}:\s*(\d+) tokens,\s*([\d.]+) tokens-per-sec", text)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)


def wired_cap_mib():
    try:
        return int(subprocess.check_output(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip())
    except Exception:
        return None


def machine_str():
    """e.g. 'Mac16,10, 16 GB' — platform.platform() alone omits the RAM."""
    try:
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        gb = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip()) // 1024**3
        return f"{model}, {gb} GB"
    except Exception:
        return platform.machine()


def tq_version():
    try:
        import turboquant_mlx
        return turboquant_mlx.__version__
    except Exception:
        return "unknown"


def text_sweep():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    unit = "Quantization maps each group of weights onto a small codebook. "
    rows = []
    print(f"\n{'prompt':>8}{'peak GiB':>11}{'vs 64GB*':>10}{'prefill':>10}{'decode':>9}")
    print("-" * 48)
    for target in (246, 844, 2040, 5043):
        n = 1
        while len(tok.encode(unit * n)) < target - 40:
            n += 1
        t0 = time.time()
        out = _run(["turboquant_mlx.generate_vlm", "--model", MODEL,
                    "--prompt", unit * n, "--max-tokens", "24", "--temp", "0",
                    "--prefill-step-size", STEP])
        blob = out.stdout + out.stderr
        if out.returncode != 0:
            print(f"{target:>8}   FAILED (exit {out.returncode})")
            rows.append({"target": target, "ok": False,
                         "stderr": blob[-1500:]})
            continue
        pk = peak_gib(blob)
        ptok, prate = rate(blob, "Prompt")
        _, drate = rate(blob, "Generation")
        ref = REFERENCE.get(target)
        delta = f"{pk - ref:+.2f}" if (pk and ref) else "—"
        print(f"{ptok or target:>8}{pk or float('nan'):>11.2f}{delta:>10}"
              f"{prate or float('nan'):>10.1f}{drate or float('nan'):>9.1f}")
        rows.append({"target": target, "ok": True, "prompt_tokens": ptok,
                     "peak_gib": pk, "prefill_tps": prate, "decode_tps": drate,
                     "seconds": round(time.time() - t0, 1),
                     "reference_prompt_tokens": REFERENCE_TOKENS.get(target)})
    print("\n* approximate: the 64 GB reference prompts realized "
          f"{'/'.join(str(REFERENCE_TOKENS[t]) for t in sorted(REFERENCE_TOKENS))}"
          " tokens, ~26 more than these (1.62 MiB of KV).")
    return rows


def vision_check():
    """Vision at two image sizes — the large one is expected to be tight."""
    from PIL import Image, ImageDraw, ImageFont

    OUT.mkdir(exist_ok=True)
    font = None
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            font = ImageFont.truetype(p, 72)
            break
        except OSError:
            continue
    if font is None:
        print("\n(no scalable font found — skipping vision)")
        return []

    big = Image.new("RGB", (700, 200), "white")
    ImageDraw.Draw(big).text((40, 60), "VOLTAGE 47", fill="black", font=font)
    big.save(OUT / "ocr_big.png")
    big.resize((350, 100)).save(OUT / "ocr_small.png")

    rows = []
    print(f"\n{'image':>12}{'peak GiB':>11}   answer")
    print("-" * 48)
    for name, note in (("ocr_big.png", "700x200"), ("ocr_small.png", "350x100")):
        out = _run(["turboquant_mlx.generate_vlm", "--model", MODEL,
                    "--image", str(OUT / name), "--max-tokens", "16",
                    "--temp", "0", "--prefill-step-size", STEP,
                    "--prompt",
                    "What text is written in this image? Answer with just the text."])
        blob = out.stdout + out.stderr
        if out.returncode != 0:
            print(f"{note:>12}     FAILED (exit {out.returncode})")
            rows.append({"image": note, "ok": False, "stderr": blob[-1500:]})
            continue
        pk = peak_gib(blob)
        # A run that completes but misreads the image is a failed vision case,
        # not a passing one -- `ok` has to track the answer, or the JSON can
        # publish `"ok": true` next to `"read": "(not read)"`. A crash is still
        # distinguishable from a misread: only the crash branch carries stderr.
        matched = bool(re.search(r"\bVOLTAGE\s*47\b", blob, re.IGNORECASE))
        ans = "VOLTAGE 47" if matched else "(not read)"
        print(f"{note:>12}{pk or float('nan'):>11.2f}   {ans}")
        rows.append({"image": note, "ok": matched, "peak_gib": pk, "read": ans})
    return rows


def main():
    cap = wired_cap_mib()
    print(f"model : {MODEL}")
    print(f"machine: {platform.platform()}")
    print(f"iogpu.wired_limit_mb = {cap}"
          + ("" if cap and cap >= 14336 else
             "   <-- run: sudo sysctl -w iogpu.wired_limit_mb=14336"))

    text = text_sweep()
    vision = vision_check()

    peaks = [r["peak_gib"] for r in text if r.get("peak_gib")]
    ok = all(r["ok"] for r in text)
    print("\n" + "=" * 48)
    if ok and peaks:
        print(f"TEXT: ran at every length. max peak {max(peaks):.2f} GiB "
              f"of {CAP_GIB:.2f} GiB cap ({CAP_GIB - max(peaks):+.2f} headroom)")
        print(f"      (Muse tq3 survived {MUSE_MINI_SURVIVED:.2f} GiB on this machine class)")
    else:
        print("TEXT: did NOT complete — see the JSON. This is a real result, send it.")

    # Write the artifact the README cites, not a private copy under the image
    # directory: a reproduction that leaves the published record untouched is
    # not a reproduction. Same schema for generated and checked-in results.
    dest = Path(__file__).parent / "bench_mini_qwen38_results.json"
    payload = {"model": MODEL,
               "machine": machine_str(),
               "platform": platform.platform(),
               "turboquant_mlx": tq_version(),
               "wired_limit_mb": cap, "prefill_step_size": int(STEP),
               "reference_peaks_64gb": REFERENCE,
               "reference_prompt_tokens_64gb": REFERENCE_TOKENS,
               "reference_note": REFERENCE_NOTE,
               "text": text, "vision": vision}
    # `note` and `repeatability` are observations ACROSS runs. A single run
    # cannot regenerate them, so carry them forward rather than silently
    # dropping them on every overwrite.
    try:
        if dest.exists():
            prior = json.loads(dest.read_text())
            for key in ("note", "repeatability"):
                if key in prior:
                    payload[key] = prior[key]
    except (OSError, ValueError) as exc:
        print(f"(could not carry forward prior provenance: {exc})")
    dest.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {dest}")
    return 0 if (ok and all(r["ok"] for r in vision)) else 1


if __name__ == "__main__":
    sys.exit(main())
