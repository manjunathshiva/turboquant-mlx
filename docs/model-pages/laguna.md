# Poolside Laguna (33B and 118B MoE for agentic coding)

Two models share this architecture: **XS.2** (33B) and **S-2.1** (118B). The port
covers both; the licence does **not** — XS.2 is Apache-2.0, S-2.1 is OpenMDW-1.1.
Check per repo before redistributing derived weights.

## Laguna-XS.2 (33B)

[Laguna-XS.2](https://huggingface.co/poolside/Laguna-XS.2) is Poolside's 33B
Mixture-of-Experts (256 experts, top-8, 3B active) built for **agentic coding
and long-horizon work on a local machine**. It is a custom architecture —
mixed sliding-window / global attention in a 3:1 ratio, per-head attention
gating, QK-norm, partial+YaRN RoPE on the global layers, and a sigmoid router
with a selection-only correction bias. `mlx-lm` has **no native `laguna`
model**; the MLX port and loader ship here (registered via `compat.py`), with
logit-parity of 2e-7 against the transformers reference.

```bash
# Convert bf16 -> TurboQuant 3-bit. bf16 (~67 GB) exceeds 64 GB RAM, so use --streaming.
python -m turboquant_mlx.convert \
    --hf-path poolside/Laguna-XS.2 \
    --mlx-path ./laguna-xs2-tq3-g64 \
    --bits 3 --group-size 64 --streaming

# Generate. Laguna's generation_config ships top_p 1.0; ALWAYS pass --top-p 0.9
# or the untruncated 100k-vocab tail injects rare junk tokens.
python -m turboquant_mlx.generate \
    --model ./laguna-xs2-tq3-g64 \
    --prompt "Write a mergesort in Python." --temp 0.7 --top-p 0.9

# Serve (agentic use). Do NOT pass --kv-bits: 30/40 layers are sliding-window(512)
# so KV is already tiny (peak 16.8 GB @2K -> 19.4 GB @31K); KV-quant saves nothing
# here and can cost up to ~4x decode.
turboquant-serve --model ./laguna-xs2-tq3-g64 --port 8080 \
    --temp 0.7 --top-p 0.9 --prompt-concurrency 1 \
    --chat-template-args '{"enable_thinking": false}'
```

- **Runs on a 32 GB Mac** at 13.8 GB on disk / ~16 GB peak; MMLU-Redux 76.9 %,
  GSM8K 79.0 % (3-bit, no-think), and passes the Opencode observe→read→edit→verify
  loop.
- **Tool calls are GLM-XML**, not Hermes-JSON; `mlx-lm`'s `glm47` parser is
  auto-selected and patched here for a trailing-newline function-name bug.
- **3-bit vs 4-bit**: this data-free codebook 3-bit build is 4 GB smaller than
  MLX-native affine 4-bit but ~2× slower per token (codebook decode + online
  Hadamard rotation, not bandwidth). Prefer it when memory-bound; prefer affine
  4-bit for the fastest interactive loop.

## Laguna-S-2.1 (118B) — ternary experts, resident on a 64 GB Mac

[Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) is the 118B sibling
(256 experts, top-10 + 1 shared, `moe_intermediate` 1024, 48 layers). At bf16 it
is ~219 GB. **Ternary (1.58-bit) experts bring it to ~27 GB, which is the only
build of this model that runs resident on a 64 GB Mac** — affine 4-bit (~64 GB)
exceeds Metal's ~56 GB working set and won't load at all, and TurboQuant's own
3-bit (48 GB) peaks at 52.7 GB and starves the OS.

```bash
python -m turboquant_mlx.convert --hf-path poolside/Laguna-S-2.1 \
    --mlx-path ./laguna-s21-tqTe-g64 \
    --attn-bits 3 --ternary-experts --group-size 64 --streaming

# Resident on 64 GB. --no-think matters: Laguna reasons out loud by default, so
# a plain generate call can spend its whole budget deliberating and never answer.
python -m turboquant_mlx.generate --model ./laguna-s21-tqTe-g64 \
    --prompt "Write a mergesort in Python." \
    --temp 0.7 --top-p 0.9 --max-tokens 1024 --no-think
```

- **~30 GB peak resident, ~12.5 tok/s** on an M4 Max, leaving ~25 GB for the OS.
- **MMLU-Redux 79.5 %, GSM8K 82.5 %** — at 1.58-bit experts this 118B still edges
  the 33B XS.2 at 3-bit (76.9 / 79.0). The larger, wider expert pool absorbs the
  deeper quantization.
- **Passes Opencode agentic coding at 1.58-bit**, where the 512-wide-expert 35B
  ternary build fails the same harness. Expert **width** (1024 vs 512), not just
  count, is what sustains sub-2-bit — see [Sub-2-bit ternary experts](../benchmarks.md).

**Expert streaming** works on Laguna too, so 27 GB resident is not the floor —
only 2.28 GB of this model has to stay resident, the other 26.42 GB pages from
disk on demand:

```bash
python -m turboquant_mlx.stream.stream_generate --model ./laguna-s21-tqTe-g64 \
    --cache-budget-gb 4 --max-active-experts 0 --prompt "..." --max-tokens 256
```

Measured on an M4 Max with a 4 GB expert cache: **peak 7.3 GB** (RSS 6.6 GB),
141 of 144 expert projections streamed, ~5 tok/s, 56.8 % cache hit rate — a
footprint that fits a **16 GB Mac mini**. Use `--max-active-experts 0` to keep
native top-10 routing (lowering it reads less from disk but changes which experts
run, and breaks agentic tool-calling).

> Requires `turboquant-mlx-full >= 0.18.0`. Earlier versions matched expert
> tensors by the `switch_mlp` name only, so Laguna's `mlp.experts` container
> counted as resident: `turboquant-plan` reported "won't run" on small Macs and
> `--cache-budget-gb auto` sized itself from a bad figure. Explicit
> `--cache-budget-gb` was unaffected.
