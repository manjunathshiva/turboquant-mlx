# Models bigger than RAM (expert streaming)

A Mixture-of-Experts model uses only a few experts per token. Expert streaming keeps the rest of
the model in memory and reads just the experts each token routes to from disk, through an LRU
cache. Memory use falls to a few GB, and the output is **bit-identical** to running the whole
model in memory (with full routing; see `--max-active-experts` below).

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 8
```

To serve a streamed model over the OpenAI API, add `--cache-budget-gb` to
[`turboquant-serve`](serve.md#a-model-bigger-than-ram).

## What to expect

Measured on a base 16 GB M4 Mac mini:

| Model | Disk | Cache budget | Hit rate | Decode | Peak |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B `tq3-g32` | ~16 GB | 2 GB | ~60% | ~3.0 tok/s | 3.9 GB |
| Qwen3.6-35B-A3B `tq3-g32` | ~16 GB | **8 GB** | **91%** | **~4.5 tok/s** | 9.4 GB |
| Qwen3.5-122B-A10B `tq3a-tqTe-g64` | 30.9 GB | 8 GB | 82% | ~2.1 tok/s | 11.3 GB |
| Qwen3-235B-A22B `tq3a-tq2e-g32` | 70.5 GB | 6 GB | 38% | ~0.2 tok/s | 10.1 GB |

On a 64 GB Mac with a 40 GB cache, the same 235B build reaches a 94% hit rate and about 4–6
tok/s once warm.

**Disk speed is the limit.** Once the cache is sensible, a faster SSD (Thunderbolt NVMe) or
fewer bytes per token (a smaller build, a bigger cache) are what help, not reader tweaks. If a
model fits in memory, running it in memory is always faster: for the 9.4 GB ternary 35B on a
16 GB mini, use `turboquant-generate`, not streaming.

## Options

| Option | Default | What it does |
|---|---|---|
| `--cache-budget-gb N\|auto` | 3 | Memory for cached experts. `auto` sizes it from the Mac's GPU working set, minus the resident weights and a reserve. 8 is the sweet spot on a 16 GB mini; 2 if memory is tight. |
| `--max-active-experts K` | 4 | Reads at most K experts per token (the router's top-K, renormalized). On Qwen3.6-35B, 8→4 matched full routing on the stress tests and halved disk reads; K=2 broke JSON output. **`0` keeps the model's own routing**: use it for quality tests and agents. |
| `--use-page-cache` / `--no-page-cache` | auto | Whether expert reads go through the macOS page cache. Auto turns it on only when the model files are under 0.6× of RAM; then decoding was 2.4× faster on the 35B. |
| `--prefetch-workers` | 8 | Parallel reads of a layer's missing experts (about 1.9× faster than one at a time). |
| `--prefetch-ahead N` | 0 | Speculatively reads the next N layers' experts. About +6% on fast NVMe; switches itself off on a saturated drive. |
| `--wire-memory` | off | Wires weights and the expert cache against memory pressure. |
| `--no-hotlist` | — | Ignore a `hot_experts.json` shipped with the model (those experts are otherwise pinned and preloaded at start). |
| `--no-learn-experts`, `--usage-file` | learning on | The cache records which experts your traffic uses and warm-starts from that profile next time. |
| `--pin-file` | — | Experimental: keep a fixed set of experts resident. It measured worse than plain LRU on a 122B. |

`stream_generate` also takes the generation options from
[Generate text](generate.md): `--temp`, `--top-p`, `--top-k`, `--rep-penalty`, `--no-think`,
`--multi-think`, `--fast`, `--trust-remote-code`, `--ngram-offload`.

## Which models stream

Streaming handles the SwitchGLU expert layout used by `qwen3_5_moe` (Qwen3.5 and Qwen3.6),
`qwen3_moe` (Qwen3-235B), the DeepSeek MLA+MoE family, Laguna and Kimi K3. The loader finds the
model's layer names automatically. Diffusion models are a poor fit: a denoising step touches
nearly every expert, so every step would read them all from disk.
