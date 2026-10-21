# Serve an OpenAI-compatible API

`turboquant-serve` is `mlx_lm.server` with a loader that understands TurboQuant models
(`quantization.mode = "turboquant"` in `config.json`). Other models pass through unchanged, so
it is a drop-in replacement for `mlx_lm.server`.

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 --port 8080
```

It listens on `127.0.0.1:8080`. Add `--host 0.0.0.0` to accept other machines; there is no
authentication or rate limiting, so only do that on a network you trust.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 1024
  }'
```

The `model` field is the string you passed to `--model`, or `default_model`, mlx-lm's name for it.
A request that names a different model makes the server load that one instead. Reasoning models return their
thinking in `message.reasoning` and the answer in `message.content`; give them
`max_tokens >= 2048` so both fit.

For vision models, use [`turboquant-serve-vlm`](vision.md#serve-it).

## Options

All `mlx_lm.server` options work unchanged: `--host`, `--port`, `--temp`, `--top-p`,
`--top-k`, `--max-tokens`, `--chat-template-args`, `--prefill-step-size`,
`--prompt-concurrency`, `--prompt-cache-size`, `--prompt-cache-bytes` and so on (see
`turboquant-serve --help`). TurboQuant adds the options below. They're handled before the
rest are passed on, so `--help` doesn't list them.

| Option | Default | What it does |
|---|---|---|
| `--kv-k-bits N --kv-v-bits M` | off | Compress each request's KV cache. K8/V3 is the recommended mix. See [KV cache compression](kv-cache.md). |
| `--kv-bits N` | off | The same width for K and V (legacy). |
| `--kv-min-tokens N` | 0 | Keep the first N cached tokens in FP16 (attention sinks). 128 is a good value. |
| `--kv-group-size G` | 64 | Rotation group size for KV quantization. |
| `--kv-fused` | off | Fused decode+attend kernel. Experimental; needs `--kv-min-tokens 0`. |
| `--disk-cache [DIR]` | off | Save prompt-cache checkpoints to disk, so a later request or a restart resumes instead of re-prefilling. |
| `--disk-cache-budget-gb` | 10 | Disk space for checkpoints (oldest evicted first). |
| `--disk-cache-min-tokens` | 512 | Don't checkpoint shorter prompts. |
| `--disk-cache-save-every` | 1024 | Tokens between checkpoints, including mid-prefill ones. |
| `--disk-cache-no-prefill-checkpoints` | off | Only checkpoint at the end of a request. |
| `--prompt-cache-max-gb` | `auto` | A byte cap on the in-memory prompt cache. `auto` caps it only when memory is tight. |
| `--metal-cache-limit-gb` | `auto` | Caps MLX's buffer-reuse cache after the model loads. `auto` acts only when memory is tight. |
| `--rep-penalty`, `--rep-ctx` | from the model; 256 | A default repetition penalty for every request. |
| `--tool-syntax-greedy` | off | Inside a `<tool_call>` block, pick the most likely token, so low-bit builds don't garble tool-call JSON. Tags via `--tool-syntax-tags "<open>,</close>"`. |
| `--prefill-keepalive` | off | Sends real SSE chunks during a long prefill, so a proxy (for example claude-code-router) doesn't time a client out. `--prefill-keepalive-interval` (default 10 s). |
| `--cache-budget-gb N\|auto` | off | Serve a MoE bigger than RAM by streaming experts from disk. See [below](#a-model-bigger-than-ram). |
| `--max-active-experts K` | 4 | Streaming only: read at most K experts per token. Use `0` for full routing in quality-sensitive or agent work. |
| `--use-page-cache` / `--no-page-cache` | auto | Streaming only: read through the OS page cache, or bypass it. |
| `--wire-memory` | off | Streaming only: wire weights and expert cache against memory pressure. |
| `--no-hotlist` | — | Streaming only: ignore a `hot_experts.json` in the model folder. |
| `--ngram-offload` | off | Qwen3.8-Flash-Next only: serve its n-gram table from disk-backed memory. |
| `--prefill-stats [--prefill-stats-file F]` | off | Diagnostic: log how much of each prompt is reused or re-prefilled. |

!!! note "KV compression serves one request at a time"
    TurboQuant KV caches can't be merged across requests, so any `--kv-*` option makes the
    server process requests one after another. That's right for one user; concurrent clients
    queue instead of batching.

## Keep the cache across restarts: `--disk-cache`

With `--disk-cache`, KV-cache checkpoints are written per conversation by a background writer,
and any later request, including the first after a restart or crash, resumes from the longest
saved prefix. Measured: the first turn of a 16.3K-token conversation after a restart dropped
from **21.1 s to 3.0 s** on a 64 GB Mac, with identical output.

Hybrid models with a recurrent state (Qwen3.6, Qwen3.8, Nemotron) can't trim a checkpoint
back to an earlier point, so they rely on **mid-prefill checkpoints** taken every
`--disk-cache-save-every` tokens. Those are taken on the one-request-at-a-time path, so for
hybrid models combine `--disk-cache` with a `--kv-*` option. On a 16 GB mini this let turn 2
of a 21K-token conversation reuse 20,480 of 21,250 tokens.

## Near the memory limit

A server that fills most of memory can fail after a few turns with:

```
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The in-memory prompt cache grows with every conversation. In order of impact:

1. **Raise the GPU memory limit**, once per boot: `sudo sysctl -w iogpu.wired_limit_mb=57344`
   on a 64 GB Mac, `43008` on 48 GB. See [Memory and macOS settings](memory.md).
2. **Use a smaller prefill step** on tight machines: `--prefill-step-size 256` (or 128).
3. **Cap the prompt cache:** `--prompt-cache-bytes 2147483648` (2 GB), or
   `--prompt-cache-size 1`. The `auto` guards above already do this when headroom is small.
4. **Close other GPU users** (browsers, Electron apps, video editors).

## A model bigger than RAM

`--cache-budget-gb` loads the model through the expert-streaming path, so only the experts each
token routes to are read from disk. This puts a ~50 GB 122B behind an OpenAI endpoint on a
16 GB Mac mini:

```bash
turboquant-serve \
    --model manjunathshiva/qwen3.5-122b-tq3 \
    --cache-budget-gb 4 \
    --kv-k-bits 8 --kv-v-bits 3 --kv-min-tokens 128 \
    --prompt-concurrency 1 --port 8080
```

`--cache-budget-gb auto` sizes the expert cache from the machine. See
[Models bigger than RAM](streaming.md) for the speed you can expect and the tuning options.

## Recipes

- **A coding agent on a 16 GB Mac mini:** see [Local coding agents](coding-agents.md#a-16-gb-mac-mini).
- **Nemotron-3-Super 3-bit on a 64 GB Mac:**

    ```bash
    sudo sysctl -w iogpu.wired_limit_mb=57344
    turboquant-serve --model manjunathshiva/Nemotron-3-Super-120B-A12B-tq3 \
        --port 8080 --prompt-cache-bytes 2147483648
    ```
