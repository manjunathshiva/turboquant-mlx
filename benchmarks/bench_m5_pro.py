"""
M5 Pro (48 GB) benchmark for TurboQuant-MLX.

Runs Qwen3.6-35B-A3B-tq3-g32 through multiple configurations and
reports prompt/generation speed, peak memory, and KV cache stats.

Configurations tested:
  1. fp16 KV baseline (no cache compression)
  2. Mixed K8/V3 (recommended default)
  3. Mixed K8/V3 + 128-token fp16 sink
  4. Symmetric K3/V3

Usage:
    python bench_m5_pro.py
    python bench_m5_pro.py --model manjunathshiva/qwen3.5-122b-tq3 --stream --cache-budget-gb 30
    python bench_m5_pro.py --max-tokens 512 --runs 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

CONFIGS = {
    "fp16_baseline": [],
    "K8_V3": ["--kv-k-bits", "8", "--kv-v-bits", "3"],
    "K8_V3_sink128": ["--kv-k-bits", "8", "--kv-v-bits", "3", "--kv-min-tokens", "128"],
    "K3_V3": ["--kv-k-bits", "3", "--kv-v-bits", "3"],
}

PROMPTS = {
    "short": "Explain why the sky is blue in two sentences.",
    "medium": (
        "You are a finance advisor. Long government bond yields are at a "
        "10-year high above 5%. Provide a 6-month capital protection plan "
        "for clients trading heavily in long-term US government bonds."
    ),
    "long": (
        "Write a detailed technical comparison of transformer attention "
        "mechanisms: multi-head attention, grouped-query attention, "
        "multi-query attention, and linear attention. Cover computational "
        "complexity, memory requirements, quality tradeoffs, and which "
        "architectures use each. Include concrete examples with dimensions."
    ),
}

DEFAULT_MODEL = "manjunathshiva/Qwen3.6-35B-A3B-tq3-g32"


def get_hardware_info() -> dict:
    """Collect Apple Silicon hardware metadata."""
    info = {}
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if "Chip:" in line:
                info["chip"] = line.split(":", 1)[1].strip()
            elif "Memory:" in line:
                info["memory_gb"] = line.split(":", 1)[1].strip()
            elif "Total Number of Cores:" in line:
                info["cores"] = line.split(":", 1)[1].strip()
    except Exception:
        # host details are best-effort; skip if system_profiler is unavailable
        pass

    try:
        try:
            dinfo = mx.device_info()
        except AttributeError:
            dinfo = mx.metal.device_info()
        info["metal_working_set_mb"] = round(
            dinfo.get("recommendedMaxWorkingSetSize", 0) / 1e6, 1
        )
        info["gpu_family"] = dinfo.get("deviceName", "unknown")
    except Exception:
        # MLX built without Metal, or device_info API changed
        info["metal_working_set_mb"] = 0.0
        info["gpu_family"] = "unknown"
    return info


def run_single(model: str, prompt: str, max_tokens: int, kv_args: list,
               stream: bool = False, cache_budget: float = None,
               rep_penalty: float = 1.1) -> dict:
    """Run a single generation and capture timing from stderr/stdout."""
    python = sys.executable
    if stream:
        cmd = [python, "-m", "turboquant_mlx.stream.stream_generate",
               "--model", model, "--prompt", prompt,
               "--max-tokens", str(max_tokens)]
        if cache_budget is not None:
            cmd += ["--cache-budget-gb", str(cache_budget)]
    else:
        cmd = [python, "-m", "turboquant_mlx.generate",
               "--model", model, "--prompt", prompt,
               "--max-tokens", str(max_tokens),
               "--temp", "0.3",
               "--rep-penalty", str(rep_penalty)]
        cmd += kv_args

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_time = time.perf_counter() - t0
        if result.returncode != 0:
            return {
                "wall_time_s": round(wall_time, 2),
                "exit_code": result.returncode,
                "error": f"non-zero exit code: {result.returncode}",
            }
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        stats = {"wall_time_s": round(wall_time, 2), "exit_code": 0}
    except subprocess.TimeoutExpired:
        return {
            "wall_time_s": round(time.perf_counter() - t0, 2),
            "exit_code": -1,
            "error": "timed out after 600s",
        }
    except Exception as e:  # defensive: don't let one bad run abort the suite
        return {
            "wall_time_s": round(time.perf_counter() - t0, 2),
            "exit_code": -2,
            "error": str(e),
        }

    for line in output.splitlines():
        line_s = line.strip()
        if "tokens-per-sec" in line_s:
            if line_s.startswith("Prompt:") and "tokens," in line_s:
                try:
                    stats["prompt_tps"] = float(line_s.split(",")[1].strip().split()[0])
                except (IndexError, ValueError):
                    # stat line missing/unparseable this run; leave the metric unset
                    pass
            elif line_s.startswith("Generation:"):
                try:
                    stats["gen_tps"] = float(line_s.split(",")[1].strip().split()[0])
                except (IndexError, ValueError):
                    # stat line missing/unparseable this run; leave the metric unset
                    pass
        elif "Peak memory:" in line_s:
            try:
                stats["peak_memory_gb"] = float(line_s.split("Peak memory:")[1].strip().split()[0])
            except (IndexError, ValueError):
                # stat line missing/unparseable this run; leave the metric unset
                pass

    return stats


def run_benchmark(model: str, max_tokens: int, runs: int,
                  prompt_key: str, stream: bool, cache_budget: float):
    """Run all configurations and collect results."""
    prompt = PROMPTS.get(prompt_key, prompt_key)
    hw = get_hardware_info()

    print("=" * 70)
    print("TurboQuant-MLX Benchmark — M5 Pro")
    print("=" * 70)
    print(f"Hardware:    {hw.get('chip', '?')} / {hw.get('memory_gb', '?')} / "
          f"{hw.get('cores', '?')} cores")
    print(f"Metal:       {hw.get('gpu_family', '?')} / "
          f"{hw.get('metal_working_set_mb', '?')} MB working set")
    print(f"Model:       {model}")
    print(f"Prompt:      {prompt_key} ({len(prompt)} chars)")
    print(f"Max tokens:  {max_tokens}")
    print(f"Runs/config: {runs}")
    if stream:
        print(f"Mode:        expert streaming (cache budget: {cache_budget} GB)")
    print("=" * 70)

    results = {"hardware": hw, "model": model, "prompt_key": prompt_key,
               "max_tokens": max_tokens, "configs": {}}

    configs = CONFIGS if not stream else {"streaming": []}

    for config_name, kv_args in configs.items():
        print(f"\n--- {config_name} ---")
        config_runs = []

        for i in range(runs):
            print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)
            stats = run_single(model, prompt, max_tokens, kv_args,
                               stream=stream, cache_budget=cache_budget)
            config_runs.append(stats)
            if "error" in stats:
                print(f"FAILED: {stats['error']} (wall={stats['wall_time_s']}s)")
            else:
                gen = stats.get("gen_tps", "?")
                prompt_tps = stats.get("prompt_tps", "?")
                mem = stats.get("peak_memory_gb", "?")
                print(f"prompt={prompt_tps} t/s, gen={gen} t/s, "
                      f"mem={mem} GB, wall={stats['wall_time_s']}s")

        if config_runs:
            gen_values = [r["gen_tps"] for r in config_runs if "gen_tps" in r]
            prompt_values = [r["prompt_tps"] for r in config_runs if "prompt_tps" in r]
            mem_values = [r["peak_memory_gb"] for r in config_runs if "peak_memory_gb" in r]

            summary = {}
            if gen_values:
                summary["gen_tps_avg"] = round(sum(gen_values) / len(gen_values), 1)
                summary["gen_tps_min"] = round(min(gen_values), 1)
                summary["gen_tps_max"] = round(max(gen_values), 1)
            if prompt_values:
                summary["prompt_tps_avg"] = round(sum(prompt_values) / len(prompt_values), 1)
            if mem_values:
                summary["peak_memory_gb"] = round(max(mem_values), 3)

            results["configs"][config_name] = {"runs": config_runs, "summary": summary}
            print(f"  Summary: gen={summary.get('gen_tps_avg', '?')} t/s avg, "
                  f"mem={summary.get('peak_memory_gb', '?')} GB peak")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Prompt t/s':>10} {'Gen t/s':>10} {'Peak GB':>10}")
    print("-" * 55)
    for name, data in results["configs"].items():
        s = data["summary"]
        print(f"{name:<20} {s.get('prompt_tps_avg', '-'):>10} "
              f"{s.get('gen_tps_avg', '-'):>10} {s.get('peak_memory_gb', '-'):>10}")

    out_path = Path(__file__).parent / "bench_m5_pro_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser(description="M5 Pro benchmark for TurboQuant-MLX")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model path (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3,
                    help="Runs per configuration for averaging")
    ap.add_argument("--prompt", default="medium", choices=list(PROMPTS.keys()),
                    help="Prompt length to test (default: medium)")
    ap.add_argument("--stream", action="store_true",
                    help="Use expert streaming mode (for models > RAM)")
    ap.add_argument("--cache-budget-gb", type=float, default=30,
                    help="Expert cache budget in GB for streaming mode")
    args = ap.parse_args()

    run_benchmark(args.model, args.max_tokens, args.runs,
                  args.prompt, args.stream, args.cache_budget_gb)


if __name__ == "__main__":
    main()
