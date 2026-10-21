# Generate text

```bash
turboquant-generate \
    --model manjunathshiva/Qwen3.6-35B-A3B-tq3a-tqTe-g64 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 300
```

`--model` takes a Hugging Face repo id (downloaded on first use) or a local folder. The prompt
goes through the model's chat template. The command prints the answer, then prompt and
generation speed and peak memory.

## Sampling

| Option | Default | Notes |
|---|---|---|
| `--max-tokens` | 200 | Raise it for reasoning models: the thinking trace counts. |
| `--temp` | 0.7 | `0` is greedy. |
| `--top-p`, `--top-k` | from the model's `generation_config.json` | Pass `0` to switch one off. Without any truncation, rare tokens can be sampled, such as a stray `</think>`. |
| `--rep-penalty`, `--rep-ctx` | from `generation_config.json`; window 256 | A small penalty (about 1.05) stops think-block loops on low-bit builds. Leave it off for math. |
| `--stop TOKEN` | — | An extra stop token (text or id), repeatable. The model's own stop tokens are applied automatically. |

## Reasoning ("thinking") models

| Option | Use |
|---|---|
| `--no-think` | Turns thinking off through the chat template. Much faster, and immune to think-block loops. |
| `--multi-think` | Allows a second `</think>`. By default it is masked, which stops low-bit thinking models re-emitting their answer. |
| `--min-tokens N` | Blocks end-of-sequence for the first N tokens. Nemotron-3 needs `--min-tokens 50`: its template makes EOS the most likely first token. |

## Memory

| Option | Use |
|---|---|
| `--kv-k-bits 8 --kv-v-bits 3` | Compresses the KV cache (the recommended mix). Add `--kv-min-tokens 128` to keep the first tokens in FP16. See [KV cache compression](kv-cache.md). |
| `--kv-bits N` | The same width for keys and values. 8 is safe everywhere; at 3 bits, symmetric KV breaks long generations on TurboQuant weights, so use the K8/V3 mix there. |
| `--ngram-offload` | Qwen3.8-Flash-Next only: serves its n-gram table from memory-mapped files, ~17.9 GiB less GPU memory, identical output. |
| `--fast` | Skips the QJL correction on builds that have it (about 25% faster). |

## Reproducible greedy output

Changing `--prefill-step-size` can change a greedy answer by a token. On a ~3.9K-token prompt,
a step of 2048 produced "…which *explains why* the sky reads blue" where 512, 256 and 128
produced "…which *is why* the sky reads blue". Both are valid: the chunk size changes the order
the GPU adds numbers in, and FP16 rounding sometimes tips a near-tie the other way.

This comes from chunked prefill on Metal, not from TurboQuant: stock
`mlx-community/Llama-3.2-1B-Instruct-4bit` on plain mlx-lm forks the same way. TurboQuant's
decode and prefill expert kernels give bit-identical results to each other
(`tests/test_kernel_determinism.py`), which is what keeps `--disk-cache` restores consistent.
For byte-for-byte repeatable greedy output, keep `--prefill-step-size` the same across runs.

## Models bigger than your RAM

`turboquant-generate` loads the whole model. For a Mixture-of-Experts model that doesn't fit,
use the streaming generator, which pages experts from disk:

```bash
python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3.5-122B-A10B-tq3a-tqTe-g64 \
    --cache-budget-gb 8 --prompt "Why is the sky blue?" --no-think
```

See [Models bigger than RAM](streaming.md).

## Vision models

Use `python -m turboquant_mlx.generate_vlm` with `--image`. See [Vision models](vision.md).
