# Qwen3.8-Flash-Next (180B MoE, 512 experts)

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`model_type: qwen4_exp`) is a 180B MoE: 512 routed experts at top-10 but only
**640 wide**, 48 layers alternating three Gated DeltaNet layers with one **Qwen
Sparse Attention** layer, hyper-connection gated residuals, and a sharded
**n-gram/PLE embedding table of 51.2B parameters, 28% of the model**. It also
carries a Qwen3-VL vision tower. No mlx-lm release has `qwen4_exp` yet
([#1788](https://github.com/ml-explore/mlx-lm/pull/1788) is open), so the
architecture is vendored here and aliased in through `compat.py`, which switches
itself off once mlx-lm ships the type.

**[`tq4a-tq2e-g64`](https://huggingface.co/manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64)**
runs fully resident on a 64 GB Mac:

| | |
|---|---|
| size | 52.00 GiB text + 0.84 GiB bf16 vision tower |
| decode, thinking disabled | 16.2–17.9 tok/s (M4 Max, resident) |
| peak | 56.30 GB short prompt · 59.43 GB at 15.7K tokens · 60.93 GB at 29.8K |
| WikiText-2 perplexity (32 × 512 tokens) | 9.31 |
| vision | reading text, left/right and counting: 3/3 on mlx-vlm 0.6.14 and 0.7.0 |
| with `--ngram-offload` | **34.13 GiB** resident instead of 52.01 (the 17.88 GiB n-gram table stays in the page cache), bit-identical output, 0 swapouts inside requests on 64 GB |
| streaming instead (12 GB expert cache) | 34.4 GB peak, 7.6 tok/s, no `sysctl` |

The tiers: 4-bit codebook attention and `lm_head`; **2-bit codebook routed
experts**; **2-bit affine** for the n-gram table and token embedding (the extras
tier; without it the build is ~124 GiB); routers, the QSA block indexer and the 96
hyper-connection gating matrices at full precision.

```bash
pip install "turboquant-mlx-full>=0.27.0"
sudo sysctl -w iogpu.wired_limit_mb=60416   # resident runs; resets on reboot

turboquant-generate --model manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64 --ngram-offload \
    --no-think --prompt "Explain sparse attention in two sentences." --max-tokens 300

# Reproduce the build: needs the 336 GiB bf16 source; converts on a 64 GB Mac
python -m turboquant_mlx.convert \
  --hf-path Qwen/Qwen3.8-Flash-Next \
  --mlx-path ./Qwen3.8-Flash-Next-tq4a-tq2e-g64 --streaming \
  --bits 4 --group-size 64 --mlp-bits 2 --mlp-group-size 64 \
  --quantize-extras --extras-bits 2 --extras-group-size 32
```

The converter builds the text model only, because upstream's `sanitize` drops
`model.visual.*`. The published repo adds those tensors, unquantized, as
`vision.safetensors`, and `turboquant_mlx.models.qwen4_exp_vision` runs them
through mlx-vlm's Qwen3-VL tower. The
[model card](https://huggingface.co/manjunathshiva/Qwen3.8-Flash-Next-tq4a-tq2e-g64)
has the vision code, the server command and the fit plan.

**`--ngram-offload` (generate, serve, streaming; 0.27.0+).** The n-gram table is 17.88
GiB of the build, yet a token reads only 16 of its 320M rows. With the flag,
the table stays in the checkpoint's safetensors files, memory-mapped, and the
rows each step needs are dequantized on the CPU. The pages sit in the OS page
cache, which macOS can reclaim, instead of wired GPU memory, which it can't.
Nothing needs re-downloading or reconverting. It is off by default.

Measured on an M4 Max 64 GB, `turboquant-serve --prefill-step-size 512`, thinking off:

| | without | with `--ngram-offload` |
|---|---|---|
| MLX active memory after load | 52.01 GiB | **34.13 GiB** |
| logits (2 × 512 WikiText-2 tokens), 48 greedy tokens | — | **bit-identical** |
| warm decode, short / code prompt | 17.6 / 17.2 tok/s | 17.2 / 16.8 tok/s |
| 16K prompt, cold first token | 62.8 s | 70.1 s (the table's rows are read from SSD the first time) |
| system wired peak at 16K | 63.4 GB | 47.1 GB |
| swapouts inside requests | 621K–706K | **0** |

Without the flag this build swapped on a 64 GB Mac in both sessions measured. The
decode figures come from separate runs on a fresh machine, because an arm that
runs right after a swapping one is ~15% slower. The flag's catch: **keep the model
directory in place while it is loaded**, since the rows are read from those files.

*Under a 48 GB-class GPU cap* (tested by lowering `iogpu.wired_limit_mb` to 39,648
on the same 64 GB Mac, so the page cache still had 64 GB of RAM behind it; not a
48 GB Mac), it loads and chats at 17.3 / 16.8 tok/s with 0 swapouts inside requests. A 16K
context is answered correctly, but at that cap the server could not keep its
prompt cache, so every repeat re-read the whole prompt (61–235 s). At the cap `turboquant-plan
--ngram-offload` now recommends (41,603 MB), the cache holds: a cold 16K prompt
answers in 62.3 s and each repeat in 0.27 s, decoding at 17.6 tok/s with 0
swapouts inside requests.

Three things to know:

- **Run with thinking disabled.** With thinking on, only 7 of 15 seeded runs close
  `</think>` within 5,000 tokens; with it off, every run terminated cleanly.
- **Don't use `--kv-bits`.** Each attention layer's cache also holds the
  sparse-attention indexer keys, so TurboQuant leaves it at full precision, with a
  warning, rather than silently turning QSA dense. KV is small anyway: 24 KB per
  token, since only 12 of 48 layers keep one.
- **Other builds in this size class:** the 2-bit affine
  [oQ2](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-oQ2) (63.0 GiB) does
  not load resident on a 64 GB Mac. ddalcu's imatrix-calibrated
  [iQ-MLX 3.3 bpw](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-3.3bpw)
  (50.6 GiB, for `mlx-serve`) does fit, and is faster. Same M4 Max, both
  servers with thinking off: `mlx-serve` decodes **2.3–2.4×** faster (39.6 / 41.5
  vs 17.6 / 17.2 tok/s warm, without MTP), prefills a cold 16K prompt 1.8×
  faster, and peaks ~3 GB lower. `mlx-serve --mtp` raises its decode further on
  this pack. What this build offers instead: a calibration-free recipe, a vision
  tower, and with `--ngram-offload`, a 34 GiB resident footprint.
