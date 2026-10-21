# Qwen3.6-27B (dense coding model for a 48 GB Mac)

**Hardware:** converts on a 64 GB Mac — or off slow USB storage with `v0.6.2+`, which forces the disk read ahead of GPU compute so conversion doesn't trip the Metal GPU watchdog — and runs **fully resident** on a **48 GB** Mac with headroom to spare. [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) is a **dense** (`qwen3_5`) long-context coder — 64 layers, hybrid attention (**48 GatedDeltaNet linear-attention + 16 full-attention** layers), head_dim 256, 262K context — that Qwen positions as competitive on SWE-bench Verified / SWE-bench Pro. Being dense, it has **no experts to stream**: it loads once and stays in RAM, so storage only matters for load time.

A pre-converted 3-bit (group-size 32) build is on the Hub:

→ [`manjunathshiva/Qwen3.6-27B-tq3-g32`](https://huggingface.co/manjunathshiva/Qwen3.6-27B-tq3-g32) — **~13 GB on disk**, **~17.5 GB peak** at runtime (fits 48 GB with ~30 GB free for KV), ~14 tok/s decode.

## Run it

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-27B-tq3-g32 \
    --prompt "Write a Python function that merges overlapping intervals." \
    --max-tokens 512 --temp 0.7
```

## Serve it to Cursor / VS Code (OpenAI-compatible)

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-27B-tq3-g32 --port 8080
```

Point the IDE's custom OpenAI base URL at `http://localhost:8080/v1`. Stock `mlx_lm.server` **can't** load a TurboQuant model (`KeyError: 'turboquant'`) — `turboquant-serve` patches the loader so the weights load through the PolarQuant path.

## Convert it yourself

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3.6-27B \
    --mlx-path ./Qwen3.6-27B-tq3-g32 \
    --bits 3 --group-size 32 --streaming
```

> **Note:** Qwen3.6 is a thinking-mode model — it emits a reasoning trace before the answer, so give it a generous `--max-tokens` (512+). Only the **16 full-attention layers** keep a growing KV cache (the 48 linear-attention layers use a fixed-size state), so it stays KV-light for long coding context; compress further with `--kv-k-bits 8 --kv-v-bits 3`.
