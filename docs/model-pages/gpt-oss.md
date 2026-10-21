# GPT-OSS

## GPT-OSS-20B (21B total, 32 experts, 3.6B active)

**Hardware:** Apple M4 Max 64GB (or any Apple Silicon with 16GB+ unified memory at 3-bit)

### Step 1: Convert to TurboQuant 3-bit (recommended)

```bash
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq3 \
    --bits 3 --group-size 32
```

**Model size:** 9.3 GB (vs 12.8 GB MXFP4 original — 28% smaller, lower perplexity)

The converter automatically:
- Detects MoE architecture (SwitchLinear / QuantizedSwitchLinear layers)
- Dequantizes MXFP4 expert weights to float
- Applies Hadamard rotation + Lloyd-Max codebook quantization
- Keeps router weights and attention at full precision
- Handles blockwise Hadamard for 2880-dim experts (2880 = 9 x 320)

### Step 2: Generate text

```bash
turboquant-generate \
    --model ./gpt-oss-20b-tq3 \
    --prompt "Explain quantum entanglement to a 10-year-old." \
    --max-tokens 256
```

**Expected:** ~73 tok/s generation, ~85 tok/s prefill on M4 Max

### Step 3: Run a quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-20b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 64 --seq-len 512
```

### All bit-widths for GPT-OSS-20B

| Method | Bits | Size | Peak RAM | Gen Speed | Quality |
|--------|------|------|----------|-----------|---------|
| Affine (mlx-lm) | 4 | 11.2 GB | ~14 GB | 148 tok/s | Coherent (but see note below) |
| TurboQuant | 4 | 11.2 GB | ~14 GB | — | Best (PPL 72.63, beats MXFP4) |
| **TurboQuant** | **3** | **9.3 GB** | **~12 GB** | **73 tok/s** | **Recommended (PPL 78.60, beats MXFP4, coherent)** |
| TurboQuant | 2 | 7.5 GB | ~10 GB | — | Poor (incoherent generation on pre-quantized models) |

> **Speed vs quality tradeoff:** Affine 4-bit is ~2x faster on the 20B model due to simpler dequantization, but TurboQuant 3-bit is 28% smaller with lower perplexity than both affine 4-bit and OpenAI's own MXFP4. Crucially, affine 4-bit **cannot scale to 120B** on 64GB hardware — TurboQuant 3-bit is the only option there.

```bash
# 4-bit (best quality, beats OpenAI's MXFP4)
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq4 \
    --bits 4 --group-size 32
```

## GPT-OSS-120B (120B total, 128 experts, ~13B active)

**Hardware:** Apple M4 Max 64GB — neither the original MXFP4 (63.5 GB) nor the [mlx-community 4-bit affine](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) (65.8 GB) fit on a 64GB machine. TurboQuant 3-bit is the only way to run this model on consumer hardware.

### Step 1: Convert to TurboQuant 3-bit (recommended)

```bash
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-120b \
    --mlx-path ./gpt-oss-120b-tq3 \
    --bits 3 --group-size 64
```

**Model size:** 48 GB

> **Note:** The default converter materializes the full quantized model in RAM before saving, so peak memory ≈ the quantized model size (~50–55 GB for a 120B). On a 64 GB machine that caps conversion at ~130B params. For anything larger, add **`--streaming`**: it writes each quantized layer to a shard and frees it, keeping peak memory to ~one 5 GB shard plus the layer being processed — so 200B+ models (Qwen3-235B, DeepSeek-V3) convert on a 64 GB Mac. Output is byte-identical to the in-memory path.

### Step 2: Generate text

```bash
turboquant-generate \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Explain quantum computing in simple terms." \
    --max-tokens 200
```

**Expected:** ~44 tok/s generation, ~9.5 tok/s prefill, 52 GB peak memory on M4 Max 64GB

### Step 3: Quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-120b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 32 --seq-len 512
```

### All bit-widths for GPT-OSS-120B

| Method | Bits | Size | Peak RAM | Gen Speed | Fits 64 GB? | Quality |
|--------|------|------|----------|-----------|-------------|---------|
| [mlx-community 4-bit](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 (affine) | 65.8 GB | — | — | **No** | — |
| MXFP4 (original) | 4 (mxfp) | 63.5 GB | ~70 GB | — | **No** | — |
| **TurboQuant** | **3** | **48 GB** | **52.3 GB** | **44 tok/s** | **Yes** | **Coherent, well-structured** |
| TurboQuant | 2 | 32 GB | 34.9 GB | 51 tok/s | Yes | Incoherent after ~20 tokens |

> Neither the original MXFP4 format (63.5 GB) nor the mlx-community affine 4-bit re-quantization (65.8 GB) fit on a 64GB Mac. TurboQuant 3-bit (48 GB) is the **only** way to run GPT-OSS-120B on consumer hardware — and at 44 tok/s, it's interactive speed. At 2-bit, the model fits easily but generation quality degrades rapidly — **3-bit is the minimum for coherent output on pre-quantized MoE models.**
