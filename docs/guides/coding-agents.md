# Local coding agents

Agent harnesses (Opencode, Aider, Cursor, VS Code extensions, Claude Code through a proxy) talk
to a model over an OpenAI-compatible endpoint, so `turboquant-serve` is all they need. The hard
part is picking a model that holds up through a long tool-calling loop at low bit widths.

## Models that pass an agent loop

Each of these passed an Opencode task end to end (observe, read, edit, run the tests, fix, run
again). Measured on the Mac shown.

| Model | Build | Size | Smallest Mac | Notes |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | [tq3a-tqTe-down4-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64) | 12.6 GB | 16 GB mini | 3/3, at 15.2–15.6 tok/s on the mini. |
| Qwen3.6-35B-A3B | [tq3-g32](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) | ~16 GB | 24 GB | The 3-bit control for the build above. |
| Laguna-XS.2 33B | [tq3-g64](https://huggingface.co/manjunathshiva/Laguna-XS.2-tq3-g64) | 13.8 GB | 32 GB | ~16 GB peak. |
| Qwen3.8-27B | [tq4-g64](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq4-g64) | 15.15 GiB | 32 GB | Also a vision model. |
| Muse Glimmer 30B | [tq4-g64](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq4-g64), [tq3-g64](https://huggingface.co/manjunathshiva/Muse-Glimmer-30B-tq3-g64) | 14.9 / 12.5 GiB | 32 GB | Serve with `turboquant-serve-vlm`. |
| Laguna-S-2.1 118B | [tqTe-g64](https://huggingface.co/manjunathshiva/Laguna-S-2.1-tqTe-g64) | ~27 GB | 48 GB | Passes at 1.58-bit experts. |

!!! warning "Not for agents"
    The pure-ternary [Qwen3.6-35B-A3B-tq3a-tqTe-g64](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64)
    (9.4 GB) is fine for chat but failed the same Opencode task 0/4: command loops and invented
    tool calls. The `down4` build above spends 3.2 GB more on the experts' down projection and
    passes.

## Server settings that matter for agents

- **Keep a temperature of about 0.7.** Greedy decoding tends to repeat itself across turns on
  low-bit builds.
- **`--tool-syntax-greedy`** picks the single most likely token inside `<tool_call>` blocks, so
  the JSON structure stays valid while the values are still sampled.
- **Turn thinking off** for speed: `--chat-template-args '{"enable_thinking": false}'`.
- **`--disk-cache`** makes repeated agent turns much faster, and survives restarts.
- **`--prompt-concurrency 1`**: one user, one request at a time.
- **Expert streaming with agents:** add `--max-active-experts 0` so every token gets its full
  set of experts.

## A 16 GB Mac mini

```bash
sudo sysctl -w iogpu.wired_limit_mb=14336   # 14 GiB; 13824 is enough below ~16K context

turboquant-serve \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64 \
    --port 8080 \
    --kv-bits 8 --tool-syntax-greedy --disk-cache \
    --prefill-step-size 128 \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    --chat-template-args '{"enable_thinking": false}' \
    --prompt-concurrency 1
```

Measured on the mini, with the model on an external SSD: 15.2–15.6 tok/s at a 13.6 GB peak, and
a 21K-token cold prefill in about 234 s. The Opencode fix-the-failing-test task passed 3/3,
faster each run as the disk cache warmed. At the default GPU limit the practical context
ceiling is about 16–18K tokens; the 128-token prefill step and the 14 GiB limit above are what
carry 21K and more. Point the harness at `http://localhost:8080/v1` with
the model id `default_model` (mlx-lm's name for the `--model` you started with), and turn off
auto-discovered skills. If the harness runs on another machine, add `--host 0.0.0.0`, but only
on a network you trust: the server has no authentication.

`--kv-bits 8` (8-bit keys and values) is what this recipe was measured with. On TurboQuant
weights avoid low symmetric widths such as `--kv-bits 3`; see [KV cache compression](kv-cache.md).

## Laguna (32 GB and up)

```bash
turboquant-serve --model manjunathshiva/Laguna-XS.2-tq3-g64 --port 8080 \
    --temp 0.7 --top-p 0.9 --prompt-concurrency 1 \
    --chat-template-args '{"enable_thinking": false}'
```

Laguna's tool calls are GLM-style XML; mlx-lm picks the right parser automatically. **Don't
add `--kv-bits` for Laguna:** most of its layers use a short sliding window, so its KV cache is
already small, and quantizing it slows decoding.

## Cursor, VS Code and other OpenAI clients

```bash
turboquant-serve --model manjunathshiva/Qwen3.6-27B-tq3-g32 --port 8080
```

Set the editor's custom OpenAI base URL to `http://localhost:8080/v1`. Plain `mlx_lm.server`
can't load a TurboQuant model (`KeyError: 'turboquant'`); `turboquant-serve` can.

## Claude Code

Claude Code speaks Anthropic's API, so put a translating proxy such as claude-code-router in
front of `turboquant-serve`. Add `--prefill-keepalive`: long cold prefills can take minutes,
and some proxies drop the server's keepalive comments, so the client times out. This option
sends real SSE chunks instead.
