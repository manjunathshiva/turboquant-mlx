# Qwen3-235B-A22B-Instruct-2507 — a 235B MoE that converts on a 16 GB Mac (hybrid + streaming)

**Hardware:** converts on a **16 GB Mac mini** via `--streaming`; runs on a **64 GB Mac** (expert streaming) or fully resident on **96 GB+**. Qwen3-235B-A22B is a `qwen3_moe` Mixture-of-Experts — **94 layers, 128 routed experts (top-8)**, ~235B total / ~22B active.

This is a **hybrid tq3a-tq2e** build: **3-bit attention** (the always-on path, kept safer) + **2-bit experts** (where the parameters — and the savings — live), routers full precision. The 128-expert / top-8 routing carries enough redundancy to absorb 2-bit experts cleanly — the same reason gpt-oss-120b holds at 2-bit while gpt-oss-20b (32 experts) collapses. Result: **~470 GB BF16 → 70.5 GB** (15 shards, 6.7×).

A pre-converted build is on the Hub:

→ [`manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32`](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32)

## Convert it yourself (streaming, fits in ~8–12 GB RAM)

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path /Volumes/SSD/qwen3-235b-tq3a-tq2e-g32 \
    --bits 3 --mlp-bits 2 -g 32 --streaming
```

`--mlp-bits 2` drops experts to 2-bit while `--bits 3` keeps attention at 3-bit; `--streaming` writes each quantized layer to a shard and frees it, so the full 235B converts in **~8–12 GB of RAM** — it was produced on a **16 GB Mac mini in ~18 minutes**. Point `--mlx-path` at a drive with ≥70 GB free.

## Run it (expert streaming)

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 40
```

## Quality + streaming benchmark

A 6-probe stress run passes **5/6**: coherent long-form essay, **correct multi-step math** ($142.80 with a 15% bulk discount), correct memoized Fibonacci, strict-JSON formatting, and clean 1–15 enumeration. The one miss was **exact factual recall** — an in-context password came back with a single flipped digit (`RAVEN-stone-91` → `-51`). Math/reasoning held; verify outputs where an exact literal value matters.

> **Need exact recall?** The full-3-bit sibling [**Qwen3-235B-A22B-Instruct-2507-tq3-g32**](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32) (3-bit experts, 103 GB) fixes the needle flip and passes **6/6** — at ~1.3 tok/s / 86.3% hit-rate on a 64 GB Mac (slower and bigger than this hybrid; the cost of full 3-bit experts). Pick the hybrid for the smallest footprint, the tq3 build when exact literal recall matters.

| Machine | Cache budget | Expert hit-rate | Disk read / token | Decode | Peak memory |
|---------|-------------|-----------------|-------------------|--------|-------------|
| M4 Mac mini, 16 GB | `--cache-budget-gb 6` | ~38% | ~3.2 GB | ~0.2 tok/s | 10.1 GB |
| 64 GB Mac | `--cache-budget-gb 40` | **94.1%** | ~0.28 GB | **~4–6 tok/s** (warm) | 46 GB |

On 64 GB a 40 GB cache holds ~60% of the ~67 GB of experts, but temporal locality lifts the hit-rate to **94.1%**, so warm decode runs at the compute-bound ~4–6 tok/s. Throughput is **bursty**: the first generation and tasks that route into a colder slice of experts stall on the SSD until their experts page in. Bump `sudo sysctl iogpu.wired_limit_mb=57344` to raise the cache past the ~48 GB default Metal wired cap.
