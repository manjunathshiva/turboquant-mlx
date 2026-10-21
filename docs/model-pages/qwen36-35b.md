# Qwen3.6-35B-A3B on a 16 GB Mac mini (expert streaming)

**Hardware:** Apple M4 Max 64GB to convert; runs **fully resident on 64 GB** or on a **16 GB Mac mini via expert streaming**. Qwen3.6-35B-A3B is a hybrid linear-attention (`qwen3_5_moe`, qwen3_next-style) + MoE model — **256 routed experts (top-8) + 1 shared**, ~35B total / ~3B active. The text-only language model is extracted (the vision tower is dropped during conversion).

A pre-converted 3-bit (group-size 32) model is on the Hub:

→ [`manjunathshiva/Qwen3.6-35B-A3B-tq3-g32`](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3-g32) — ~16 GB on disk; ~60 tok/s at ~18 GB peak when fully resident on a 64 GB Mac.

Two smaller builds run **fully resident on a 16 GB Mac mini**, with no streaming:

- → [`Qwen3.6-35B-A3B-tq3a-tqTe-g64`](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64): ternary experts, **9.4 GB**, ~10.4 tok/s on the mini at the default memory limit. Good for chat; it does not hold up in agent harnesses.
- → [`Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64`](https://huggingface.co/manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-down4-g64): ternary experts with a 4-bit expert `down_proj`, **12.6 GB**, 15.2–15.6 tok/s on the mini (needs `sudo sysctl -w iogpu.wired_limit_mb=13824`). Passes an Opencode agentic task 3/3.

## Run it fully resident (64 GB)

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512
```

## Run it on a 16 GB Mac mini (expert streaming)

The model is ~16 GB on disk, so it won't fit fully resident in 16 GB alongside the OS (resident decode peaks ~18 GB). Expert streaming pages only the router-selected experts from disk per token (LRU-cached), keeping resident memory to a few GB. With `--max-active-experts 0` (the model's own top-8 routing) the output is **bit-identical** to the fully-resident model; the default of 4 reads about half as many experts and matched full routing on the stress tests, but is not identical in general. (`os.pread` + macOS `F_NOCACHE` keep the OS page cache from ballooning while streaming.)

Since `v0.5.0` the missing experts for each layer are read **in parallel** on a thread pool (`--prefetch-workers`, default `8`), hiding SSD latency behind compute — ~1.9× faster decode at a tight cache budget, still bit-identical. Pass `--prefetch-workers 1` for the serial baseline.

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 8
```

## Benchmark (base Apple M4 Mac mini, 16 GB)

| Config | Expert hit-rate | Disk read / token | Decode | Peak RSS |
|--------|-----------------|-------------------|--------|----------|
| `--cache-budget-gb 2` | ~60% | ~175 MB | ~3.0 tok/s | **3.9 GB** |
| `--cache-budget-gb 8` *(recommended)* | **91%** | ~41 MB | **~4.5 tok/s** | 9.4 GB |

A larger cache keeps more experts resident, raising the hit-rate and cutting SSD reads — the throughput limiter when streaming. `--cache-budget-gb 8` is the sweet spot on a 16 GB machine; drop to `2` if RAM is tight. Streaming targets the SwitchGLU expert layout used by `qwen3_5_moe` and the DeepSeek MLA+MoE family (`deepseek_v2`/`v3`); the loader auto-detects the model's layer-key prefix.

> **Note:** Qwen3.6 is a thinking-mode model — it emits a reasoning trace before the final answer, so give it a generous `--max-tokens` (512+) for tasks that need a concluding answer.

## Tuning the streaming reader (`v0.6.1`)

Once the cache policy is reasonable, **disk bandwidth is the wall** — for MoE decode the LRU + 8-worker parallel-read pool is already near-optimal, so the big levers are faster storage (Thunderbolt/NVMe) and fewer bytes/token (a hybrid build, a bigger `--cache-budget-gb`), not the read algorithm. A few knobs squeeze the rest:

> **How much can a smarter cache policy actually buy? (measured, 122B on a 16 GB mini.)** The 0.15.0 **learning expert cache** (`learn_experts=True`, default) records which experts *your* traffic routes to and pins from that profile once mature — the adaptive cousin of a shipped `hot_experts.json`. On the genuinely disk-bound case it was built for — `Qwen3.5-122B-A10B-tq3a-tqTe-g64` (30.9 GB, `F_NOCACHE`, auto budget 4.3 GB), learned pins vs pure LRU over identical prompts — it lifted the cache hit rate **54.0 → 55.9%** and cut expert bytes read **84.4 → 81.0 GB (−4.0%)**, but decode speed stayed **inside run-to-run noise** (two byte-identical unpinned runs themselves varied ±7%). The reason is structural, and it is the same wall: 4.3 GB of cache against **27 GB of experts is 16%**, and 256-expert top-8 routing is too spread for a small pin to beat recency. At **~0.47 GB/token off the SSD**, trimming 4% is nothing. **The pin keeps better experts and reads a little less disk; it does not make streaming faster.** Faster storage and fewer bytes/token do — the learned profile's real value is warm-starting the cache (skipping the cold-record run), not throughput.


| Knob | Default | What it does |
|------|---------|--------------|
| `--max-active-experts K` | `4` | **K-reduction** — caps router `top_k` to `min(native, K)` per MoE block, so the switch streams fewer experts/token. `argpartition` selects fewer and `norm_topk_prob` renormalizes the gates, so it stays a clean reduced-K MoE. On Qwen3.6-35B-A3B (native top-8) **K=8→4 is byte-identical** on the 6-test stress harness and cuts streamed disk reads **~2.09×** (1.4× faster decode in the disk-bound regime); K=2 collapses (broken JSON). `4` = safe floor; `0` = native routing. |
| `--use-page-cache` / `--no-page-cache` | **auto** | **Trust-OS** — whether expert reads use the OS page cache (vs `F_NOCACHE`). On a roomy machine where the model fits in free RAM, leaving the page cache on returns LRU-eviction re-reads from warm RAM instead of disk: **2.44× faster decode** on the 35B at a small budget (7.58 → 18.50 tok/s), same hit-rate/RSS. Auto-enables only when model files are `< 0.6× total RAM`, so a 16 GB mini on a 70 GB MoE keeps `F_NOCACHE` and never thrashes. |
| read-coalescing | **on** | Merges contiguous missed experts into one `os.pread`. Bit-identical, free, ~5% faster when disk-bound. No flag. |
| `--prefetch-ahead N` | `0` (off) | Speculatively prefetch the next *N* layers' experts (predicted from the previous token's routing) on a background thread. ~+6% on fast NVMe with spare bandwidth; **self-disables** if the drive proves bandwidth-bound (e.g. a saturated USB bus), so it's safe to set `1`. |
| `--pin-file pin.json` | none | Keep a calibrated hot-expert set permanently resident. **Experimental** — measured net-negative vs pure LRU on a 122B (static pinning costs LRU's adaptivity). For experimentation only. |

The `--max-active-experts` and page-cache levers are ports of [Flash-MoE](https://github.com/danveloper/flash-moe)'s K-reduction and "trust the OS" findings (blueprint: Apple [*LLM in a Flash*](https://arxiv.org/abs/2312.11514)), measured on the TurboQuant streaming path.

Generate `pin.json` (and a co-activation `perm.json` for the optional `stream/repack_experts.py` relayout) with `python -m turboquant_mlx.stream.calibrate_experts`.
