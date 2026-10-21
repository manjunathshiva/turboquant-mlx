# Qwen3.8-27B (dense 27.8B hybrid Gated-DeltaNet VLM)

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) reports
`model_type: qwen3_5` despite the name — a **dense 27.8B multimodal** model whose
64 decoder layers are **48 Gated-DeltaNet (linear attention) + 16 full
attention**, one full layer in every four. Hidden 5120, intermediate 17408,
**vocabulary 248,320**, plus a 27-block vision tower. mlx-lm 0.31.3 and mlx-vlm
0.6.3 already carry the architecture, so nothing needed porting; conversion
takes about 18 seconds.

Two builds are published, and they target different machines.

| build | size | peak (1.3K prompt) | decode | Opencode agentic |
|---|---|---|---|---|
| **[`tq4-g64`](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq4-g64)** | **15.15 GiB** | 22.85 GB | 11.4 tok/s | **pass** (6m12s) |
| [`tq3-mini-g64`](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq3-mini-g64) | 11.55 GiB | 13.30 GiB | 12.2 tok/s | pass (6m32s)¹ |
| `tq3-g64`, 8-bit extras (unpublished) | 12.88 GiB | 20.60 GB | 9.2 tok/s | pass (6m32s) |

¹ the agentic run was done on the 8-bit-extras `tq3`; the published `tq3-mini`
differs only in the extras tier, at an identical 8.0052-vs-8.0551 perplexity.

Both fix a planted off-by-one end to end: given the task's pytest command, each build
runs it, reads the file, makes the minimal edit, re-runs to green and explains with `file:line`. Vision
works on both — shapes, colours, positions and grounded bounding boxes.

**Take `tq4` if it fits; take `tq3-mini` if it does not.** At 4-bit, `tq4` is
*faster as well as better* than the 8-bit-extras `tq3` for 2.3 GiB more — the
same result the [speed flip](../guides/kv-cache.md#it-saves-memory-it-rarely-saves-time) predicts, since the codebook path
only earns its keep at 2-bit and 3-bit's 10-indices-per-`uint32` packing costs
more to unpack than 4-bit's clean 8.

The exception is the 16 GB tier, where nothing else reaches.
[`tq3-mini-g64`](https://huggingface.co/manjunathshiva/Qwen3.8-27B-tq3-mini-g64)
is **verified on a real 16 GB M4 Mac mini** — not projected down from a bigger
machine — at `iogpu.wired_limit_mb=14336` and `--prefill-step-size 256`:

| prompt | peak, 16 GB mini | peak, 64 GB M4 Max\* | prefill | decode |
|---|---|---|---|---|
| 220 tok | **12.95 GiB** | 13.22 GiB | 20.7 tok/s | 3.9 tok/s |
| 818 tok | **12.96 GiB** | 13.05 GiB | 19.0 tok/s | 3.9 tok/s |
| 2,014 tok | **13.14 GiB** | 13.30 GiB | 21.0 tok/s | 3.6 tok/s |
| 5,017 tok | **13.61 GiB** | 13.81 GiB | 20.6 tok/s | 3.7 tok/s |

\* The M4 Max sweep realized 246 / 844 / 2,040 / 5,043 prompt tokens — a
constant 26 more than the mini's, worth 1.62 MiB of KV at 64 KiB/token. Read the
comparison as close, not as an exact A/B.

Vision passes there too — OCR correct at 350×100 (12.44 GiB) and 700×200
(12.68 GiB). Peaks reproduced identically across two independent sweeps, and
they run **0.09–0.27 GiB *below*** the comparable prompts on the M4 Max — 55–170×
more than that 26-token gap can account for — so the larger machine is the
pessimistic estimate. Reproduce with
[`benchmarks/bench_mini_qwen38.py`](https://github.com/manjunathshiva/turboquant-mlx/blob/main/benchmarks/bench_mini_qwen38.py); the full
record is in
[`bench_mini_qwen38_results.json`](https://github.com/manjunathshiva/turboquant-mlx/blob/main/benchmarks/bench_mini_qwen38_results.json).

Neither 4-bit build reaches that tier: affine 4-bit peaks at **16.16 GiB**, more
than a 16 GB machine has in total. Decode there is **3.7 tok/s** — fine for
document Q&A, summarization and vision calls, too slow for interactive chat.

The hybrid stack makes KV very cheap — only 16 of 64 layers cache anything, so
**64 KiB/token**, or 2.0 GiB at 32K context.

```bash
pip install "turboquant-mlx-full[vlm]"

python -m turboquant_mlx.convert_vlm \
  --hf-path Qwen/Qwen3.8-27B \
  --mlx-path ./Qwen3.8-27B-tq4-g64 \
  --bits 4 --group-size 64 --quantize-extras --extras-bits 8

python -m turboquant_mlx.generate_vlm \
  --model ./Qwen3.8-27B-tq4-g64 --prompt "..." --max-tokens 256
```

Measured on mlx-lm 0.31.3 with **both mlx-vlm 0.6.3 and 0.6.13** — identical
peaks and identical vision results on each, so the `[vlm]` floor of 0.6.12
reproduces these numbers.

`--prefill-step-size` is the memory knob for long prompts, and it is a large
one. mlx-vlm only chunks prefill *above* the step size, so a 1342-token prompt
runs unchunked at the 2048 default:

| `--prefill-step-size` | peak | prefill |
|---|---|---|
| default (2048 — unchunked here) | 21.28 GB | 162 tok/s |
| **256** | **16.77 GB** | 77 tok/s |

**−4.5 GB for 2.1× slower prefill.** 256 also keeps every TurboQuant layer on
the fused Metal kernel, which materializes nothing. This is what makes the
24 GB target work.

## `lm_head` stays off the polar path

With a 248,320-token vocabulary `lm_head` is 248320×5120 = **1.271B
parameters**, the largest matrix in the model. `PolarQuantizedLinear` has a
fused Metal kernel only up to `_QMM_MAX_TOKENS` (256) tokens; past that it
dequantizes the weight through full-size intermediates. Measured on exactly that
shape at 3-bit/g64:

| tokens | transient peak |
|---|---|
| 256 | +0.116 GiB |
| **257** | **+13.141 GiB** |
| 2048 | +13.953 GiB |

The same shape as an 8-bit affine layer costs +1.895 GiB at 2048 tokens — that
is the logits array itself, with no weight-dequant term at all. So `qwen3_5`
joins `muse_glimmer` in `VLM_SKIP_PATTERNS`, and `lm_head` is left to the
`--quantize-extras` affine path. End to end on a 1342-token prompt:

| build | size | peak |
|---|---|---|
| `lm_head` affine | 15.15 GiB | **22.85 GB** |
| `lm_head` polar | 14.52 GiB | **33.34 GB** |

Polar buys 0.63 GB on disk and costs 10.5 GB at runtime — and that 33 GB peak is
the difference between fitting a 36 GB Mac and not.

This is easy to miss because mlx-vlm's chunked prefill *discards* the chunk
forward's return value, so MLX never evaluates the matmul at all. But chunking
only engages **above** `prefill_step_size` (default 2048); a prompt of 257–2048
tokens takes the single-shot branch in `generate/ar.py`, which slices
`logits[:, -1, :]` and therefore does evaluate `lm_head` across the whole
sequence. That is an ordinary chat turn, not an edge case.

## Against MLX's own affine 4-bit

[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
is the same architecture at the same 4-bit / group-size-64 setting, with no MTP
weights on either side — so this is a like-for-like comparison, run through the
same harnesses and the same server.

The two TurboQuant columns differ only in what the *non-polar* remainder
(embeddings, `lm_head`, vision tower) is stored at — `--extras-bits 8` versus
`4`. They are separate builds with separate numbers, so they get separate
columns.

| | affine 4-bit | `tq4-g64` extras 8 | `tq4-g64` extras 4 |
|---|---|---|---|
| on disk | 14.95 GiB | 15.15 GiB | **13.81 GiB** |
| bits/weight (polar tier) | 4.500 | **4.251** | **4.251** |
| PPL, 16K tok | 7.2685 | **7.2496** | 7.3043 |
| PPL, 32K tok | 8.3593 | **8.3321** | 8.4061 |
| decode | **29.7 tok/s** | 11.3 tok/s | 11.5 tok/s |
| peak, same harness | **16.16 GiB** | 17.80 GiB | 16.47 GiB |
| vision battery | 3/4 | 4/4 | — |
| Opencode agentic | **pass** (6m42s) | **pass** (6m12s) | — |

Only the `extras 8` build is published, and the vision/agentic runs were done
on it; the `extras 4` column is the size-matched control.

**Take the affine build if you want 4-bit Qwen3.8-27B.** It is 2.6× faster to
decode at a lower peak, and the perplexity difference is 0.3% — real and stable
in ordering across both corpus sizes, but far too small to trade that speed for.

The one place TurboQuant is structurally ahead is storage. At identical bit
width and group size, on the same 17408×5120 projection:

| | weight | scales | biases | bits/weight |
|---|---|---|---|---|
| affine 4-bit | 42.50 MiB | 2.66 | **2.66** | **4.500** |
| polar 4-bit | 42.50 MiB | 2.66 | — | **4.251** |

The Hadamard rotation symmetrizes each group, so there is no zero-point to
store — 5.5% leaner for free. That margin is a rounding error at 4-bit and the
reason the codebook only earns its keep at 2-bit and below (see
[the speed flip](../guides/kv-cache.md#it-saves-memory-it-rarely-saves-time)).

Two cautions on reading the table. The vision split is **not** a capability
gap: affine's single miss is one string (`VOLTAGE 47` → `VOLTAGE`), and a
five-string follow-up probe put it at 4/5, so treat 3/4-vs-4/4 as a tie. And
the agentic wall-clock is **not** a speed metric — affine decodes 2.6× faster
yet finished *slower*, because it spent more tool turns.

Reproduce both halves:

```bash
# 16K-token column (--chunks 32) and 32K-token column (--chunks 64);
# both records are checked in under benchmarks/
python -m turboquant_mlx.benchmarks.eval_vlm_perplexity \
    ./Qwen3.8-27B-tq4-g64 ./Qwen3.8-27B-tq4-g64-x4 \
    mlx-community/Qwen3.8-27B-4bit --chunks 32
python -m turboquant_mlx.benchmarks.eval_vlm_perplexity \
    ./Qwen3.8-27B-tq4-g64 ./Qwen3.8-27B-tq4-g64-x4 \
    mlx-community/Qwen3.8-27B-4bit --chunks 64
python -m turboquant_mlx.benchmarks.eval_vlm_vision \
    ./Qwen3.8-27B-tq4-g64 mlx-community/Qwen3.8-27B-4bit
```

## Will it fit?

`turboquant-plan --model <path>` at 16K context, per machine:

| Mac | verdict |
|---|---|
| 16 GB | ❌ **Will not run.** Peak 18.44 GB against 15.46 GB usable — over by 2.98 GB. A dense model has no expert tier to stream, so the [16 GB recipe](../guides/coding-agents.md#a-16-gb-mac-mini) does not apply |
| 24 GB | ⚠️ Fits **after raising the Metal wired cap**, with `--prefill-step-size 256` — that knob measures 16.77 GB peak on a 1342-token prompt, against 21.28 GB at the default |
| 36 GB | ✅ Resident, 12.34 GB headroom |
| 64 GB | ✅ Resident, 39.40 GB headroom |

> **Speed is the caveat.** At ~11 tok/s decode and ~110–160 tok/s prefill an
> agentic turn takes about six minutes, against 33 s for Laguna-XS.2 at 3-bit.
> It passes, but this is a model to serve for quality rather than latency.
