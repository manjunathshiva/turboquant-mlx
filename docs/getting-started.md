# Get started

## Requirements

- A Mac with Apple Silicon (M1 or later).
- Python 3.11 or newer. (On Python 3.10, pip installs the older 0.28.x.)
- Enough unified memory for the model you pick. [Choose a model](models.md) sorts every published
  build by Mac memory, from 16 GB up.

The Metal kernels are compiled by MLX the first time they run, so there is no Xcode or CMake
step.

## Install

```bash
pip install turboquant-mlx-full
```

The package is published as `turboquant-mlx-full` and imported as `turboquant_mlx`.

Optional extras:

```bash
pip install "turboquant-mlx-full[vlm]"    # vision and diffusion models (mlx-vlm >= 0.7.9)
pip install "turboquant-mlx-full[eval]"   # perplexity benchmarking (datasets, transformers)
pip install "turboquant-mlx-full[kimi]"   # Kimi K3's tiktoken tokenizer
```

## 1. Check that it fits, before downloading

```bash
turboquant-plan --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64
```

```
Projection at 16,384 tokens of context
  weights              12.59 GB
  KV cache             0.34 GB  (20.0 KB/token, hybrid: 10/40 full-attention layers)
  prefill workspace    1.07 GB  (estimate, at --prefill-step-size 2048)
  runtime reserve      1.00 GB  (buffer cache, activations, fragmentation)
                       ----------------------------------
  peak                 15.00 GB of 61.85 GB usable   46.85 GB headroom

Verdict: ✅ RESIDENT — fits fully in memory
```

`turboquant-plan` reads only the safetensors **headers**: for a Hugging Face repo id it fetches a
few hundred KB over the network (the 12.6 GB repo above answers in about 2.5 s), so you know
before the download. Weights are exact; the KV cache comes from the model's attention geometry;
the prefill workspace is an estimate.

To plan for a Mac you're not sitting at, give it that Mac's limits, and it tells you which
settings to use:

```bash
turboquant-plan --model <repo> --wired-gb 10.5 --ram-gb 16 --context 21000 --kv-bits 8
```

```
Verdict: ⚠️  RESIDENT — fits, but only after raising the Metal wired cap

Recommended:
  sudo sysctl -w iogpu.wired_limit_mb=13721   (raises the 10.50 GB Metal cap —
      the binding limit here, not your 16 GB of RAM; resets on reboot)
  --prefill-step-size 128   (the default 2048 needs 1.38 GB of transient
      workspace at this context)
  --kv-bits 8
```

`turboquant-doctor` runs the same projection plus a read-only readiness check (model files,
tokenizer, quantization block, MLX imports, machine limits). Both commands take `--json`.

!!! note "What the `sysctl` line means"
    macOS lets the GPU use only part of unified memory by default: about 48 GB on a 64 GB
    Mac, and roughly 10.5 GB on a 16 GB Mac mini.
    Raising `iogpu.wired_limit_mb` lets it use more, until the next reboot. See
    [Memory and macOS settings](guides/memory.md).

## 2. Generate text

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 300
```

The model downloads from Hugging Face on first use. `--model` also accepts a local folder.
Useful flags: `--temp`, `--top-p`, `--no-think` (skip the reasoning trace on thinking models)
and `--kv-bits` (compress the KV cache). See [Generate text](guides/generate.md).

## 3. Serve it to apps

`turboquant-serve` is `mlx_lm.server` with a loader that understands TurboQuant models, so
anything that speaks the OpenAI API can use it:

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 --port 8080
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64",   # must match --model
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    max_tokens=1024,
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Vision models use `turboquant-serve-vlm` instead. See [Serve an OpenAI-compatible API](guides/serve.md)
and [Vision models](guides/vision.md).

## Next steps

- [Choose a model](models.md) for your Mac.
- [Convert a model](guides/convert.md) yourself from its Hugging Face weights.
- [Local coding agents](guides/coding-agents.md): Claude Code, Opencode, Cursor and VS Code
  against a local model.
