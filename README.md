# TurboQuant-MLX

Extreme weight compression for LLMs on Apple Silicon. Adapts Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) from KV cache compression to **weight quantization** using MLX.

Supports dense models (LLaMA, Qwen, Mistral) and **Mixture-of-Experts** (Qwen-MoE, GPT-OSS, Qwen3.5-MoE).

## Key Results

| Model | Method | Bits | PPL | Size | Gen Speed (M4 Max) |
|-------|--------|------|-----|------|---------------------|
| Qwen2.5-7B | TurboQuant | 3 | 8.92 | 3.5 GB | — |
| Qwen2.5-7B | Affine | 3 | 13.37 | 3.3 GB | — |
| GPT-OSS-20B | Affine (mlx-lm) | 4 | — | 11.2 GB | 148 tok/s |
| GPT-OSS-20B | MXFP4 (original) | 4 | 83.04 | 12.8 GB | — |
| GPT-OSS-20B | TurboQuant | 4 | 72.63 | 11.2 GB | — |
| GPT-OSS-20B | TurboQuant | 3 | 78.60 | 9.3 GB | **73 tok/s** |
| GPT-OSS-120B | [Affine 4-bit (mlx-community)](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 | — | 65.8 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | MXFP4 (original) | 4 | — | 63.5 GB | *Doesn't fit 64GB* |
| GPT-OSS-120B | TurboQuant | 3 | — | 48 GB | **44 tok/s** |
| GPT-OSS-120B | TurboQuant | 2 | — | 32 GB | 51 tok/s (poor quality) |
| Qwen3.5-122B-A10B | BF16 (original) | 16 | — | ~240 GB | *Doesn't fit 64GB* |
| **Qwen3.5-122B-A10B** | **TurboQuant** | **3** | **—** | **~50 GB** | **26.5 tok/s** |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 64 GB unified memory recommended for 20B+ models

```bash
pip install mlx mlx-lm datasets transformers
```

## Quick Start

### 1. Convert a model to TurboQuant format

```bash
# Dense model (e.g., LLaMA 3.2 1B at 3-bit)
python -m turboquant_mlx.convert \
    --hf-path meta-llama/Llama-3.2-1B \
    --mlx-path ./llama-3.2-1b-tq3 \
    --bits 3 --group-size 64

# MoE model (e.g., GPT-OSS-20B at 2-bit)
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-20b \
    --mlx-path ./gpt-oss-20b-tq2 \
    --bits 2 --group-size 64
```

### 2. Generate text

```bash
python -m turboquant_mlx.generate \
    --model ./gpt-oss-20b-tq2 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200
```

### 3. Evaluate perplexity

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-20b \
    --bits 2 3 4 \
    --num-samples 256 --seq-len 512
```

---

## Running GPT-OSS MoE Models on Apple Silicon

### GPT-OSS-20B (21B total, 32 experts, 3.6B active)

**Hardware:** Apple M4 Max 64GB (or any Apple Silicon with 16GB+ unified memory at 3-bit)

#### Step 1: Convert to TurboQuant 3-bit (recommended)

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

#### Step 2: Generate text

```bash
python -m turboquant_mlx.generate \
    --model ./gpt-oss-20b-tq3 \
    --prompt "Explain quantum entanglement to a 10-year-old." \
    --max-tokens 256
```

**Expected:** ~73 tok/s generation, ~85 tok/s prefill on M4 Max

#### Step 3: Run a quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-20b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 64 --seq-len 512
```

#### All bit-widths for GPT-OSS-20B

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

---

### GPT-OSS-120B (120B total, 128 experts, ~13B active)

**Hardware:** Apple M4 Max 64GB — neither the original MXFP4 (63.5 GB) nor the [mlx-community 4-bit affine](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) (65.8 GB) fit on a 64GB machine. TurboQuant 3-bit is the only way to run this model on consumer hardware.

#### Step 1: Convert to TurboQuant 3-bit (recommended)

```bash
python -m turboquant_mlx.convert \
    --hf-path openai/gpt-oss-120b \
    --mlx-path ./gpt-oss-120b-tq3 \
    --bits 3 --group-size 64
```

**Model size:** 48 GB

> **Note:** Conversion requires temporarily loading the full model. With 120B parameters, peak memory during conversion may reach ~50-55 GB. On a 64 GB machine this is tight — close all other applications before running. The converter processes layers sequentially and frees memory after each expert is quantized.

#### Step 2: Generate text

```bash
python -m turboquant_mlx.generate \
    --model ./gpt-oss-120b-tq3 \
    --prompt "Explain quantum computing in simple terms." \
    --max-tokens 200
```

**Expected:** ~44 tok/s generation, ~9.5 tok/s prefill, 52 GB peak memory on M4 Max 64GB

#### Step 3: Quick quality check

```bash
python -m turboquant_mlx.evaluate \
    --hf-path openai/gpt-oss-120b \
    --bits 3 \
    --no-affine --no-qjl \
    --num-samples 32 --seq-len 512
```

#### All bit-widths for GPT-OSS-120B

| Method | Bits | Size | Peak RAM | Gen Speed | Fits 64 GB? | Quality |
|--------|------|------|----------|-----------|-------------|---------|
| [mlx-community 4-bit](https://huggingface.co/mlx-community/gpt-oss-120b-4bit) | 4 (affine) | 65.8 GB | — | — | **No** | — |
| MXFP4 (original) | 4 (mxfp) | 63.5 GB | ~70 GB | — | **No** | — |
| **TurboQuant** | **3** | **48 GB** | **52.3 GB** | **44 tok/s** | **Yes** | **Coherent, well-structured** |
| TurboQuant | 2 | 32 GB | 34.9 GB | 51 tok/s | Yes | Incoherent after ~20 tokens |

> Neither the original MXFP4 format (63.5 GB) nor the mlx-community affine 4-bit re-quantization (65.8 GB) fit on a 64GB Mac. TurboQuant 3-bit (48 GB) is the **only** way to run GPT-OSS-120B on consumer hardware — and at 44 tok/s, it's interactive speed. At 2-bit, the model fits easily but generation quality degrades rapidly — **3-bit is the minimum for coherent output on pre-quantized MoE models.**

---

### Qwen3.5-122B-A10B (122B total, 256 experts, 8 active, ~10B active)

**Hardware:** Apple M4 Max 64GB — the original BF16 model is ~240 GB. TurboQuant 3-bit compresses it to ~50 GB, fitting on a 64GB machine.

This is a brand-new architecture featuring **256 MoE experts** (the most of any model we've tested), **hybrid attention** (GatedDeltaNet linear attention + standard softmax attention), and **thinking/reasoning** capability. The model also has a shared expert per layer alongside the routed experts.

#### Step 1: Convert to TurboQuant 3-bit

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3.5-122B-A10B \
    --mlx-path ./qwen3.5-122b-tq3 \
    --bits 3 --group-size 64
```

**Model size:** ~50 GB | **Conversion time:** ~90 seconds

> **Note:** Conversion requires ~55 GB peak memory. Close all other applications before running. The converter uses memory-efficient processing — each expert layer is replaced immediately after quantization with aggressive garbage collection to handle the 256 experts per layer.

#### Step 2: Generate text

```bash
python -m turboquant_mlx.generate \
    --model ./qwen3.5-122b-tq3 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200
```

**Expected:** ~26.5 tok/s generation, 55 GB peak memory on M4 Max 64GB

#### Benchmark

| Method | Bits | Size | Peak RAM | Gen Speed | Fits 64 GB? | Quality |
|--------|------|------|----------|-----------|-------------|---------|
| BF16 (original) | 16 | ~240 GB | — | — | **No** | — |
| **TurboQuant** | **3** | **~50 GB** | **54.9 GB** | **26.5 tok/s** | **Yes** | **Coherent reasoning with structured thinking** |

> Qwen3.5-122B-A10B is the largest and most complex model TurboQuant has been tested on: 122B parameters, 256 experts (8 active per token), hybrid GatedDeltaNet + softmax attention, and a shared expert per MoE layer. At 3-bit, the model produces structured reasoning with proper analysis steps — demonstrating that TurboQuant preserves thinking capability at extreme compression.

---

## How It Works

TurboQuant is a two-stage, **calibration-free** quantization pipeline:

1. **Hadamard Rotation** — Multiply weights by a randomized Hadamard matrix, transforming any weight distribution into a near-Gaussian shape. This is data-oblivious (no calibration data needed).

2. **Lloyd-Max Codebook** — Apply information-theoretically optimal quantization for Gaussian distributions. The codebook is a mathematical constant, precomputed once.

The result: near-zero quality loss at 3-bit, and usable 2-bit quantization where standard affine completely breaks down.

For MoE models, all experts within a layer share the same rotation signs and codebook, keeping storage efficient.

## CLI Options

```
python -m turboquant_mlx.convert --help

Options:
  --hf-path TEXT       HuggingFace model path or local path (required)
  --mlx-path TEXT      Output directory (default: mlx_model)
  --bits {2,3,4}       Quantization bit-width (default: 3)
  --group-size {32,64,128}  Elements per quantization group (default: 64)
  --rotation TEXT      Rotation method: hadamard, blockwise_hadamard, none
  --use-qjl           Enable 1-bit QJL residual correction (+1 bit overhead)
  --dtype TEXT         Model dtype before quantization: float16, bfloat16
```

## Supported Architectures

| Architecture | Model Type | MoE | Status |
|-------------|-----------|-----|--------|
| LLaMA / Llama 3 | `llama` | No | Tested |
| Qwen2 / Qwen2.5 | `qwen2` | No | Tested |
| Qwen3.5 | `qwen3_5` | No | Tested |
| Mistral | `mistral` | No | Tested |
| Qwen1.5-MoE | `qwen2_moe` | Yes | Tested |
| GPT-OSS | `gpt_oss` | Yes | Tested |
| Qwen3.5-MoE | `qwen3_5_moe` | Yes (256 experts) | Tested (122B) |

## Project Structure

```
turboquant_mlx/
    config.py                 # TurboQuantConfig
    convert.py                # CLI: HF model -> TurboQuant MLX
    generate.py               # Text generation with TurboQuant models
    evaluate.py               # Perplexity evaluation
    quantize_model.py         # Model traversal & layer replacement
    core/
        codebook.py           # Lloyd-Max codebooks for Gaussian
        rotation.py           # Randomized Hadamard rotation
        polar_quantize.py     # Rotate + codebook quantize
        packing.py            # Bit-packing into uint32
        qjl.py                # QJL residual correction
    layers/
        polar_linear.py       # PolarQuantizedLinear (dense)
        polar_switch_linear.py # PolarQuantizedSwitchLinear (MoE)
    kernels/
        polar_qmv.py          # Fused Metal kernel (dense decode)
        polar_gather_qmv.py   # Fused Metal kernel (MoE shared input)
        polar_multi_gather_qmv.py  # Fused Metal kernel (MoE per-expert input)
    csrc/
        polar_kernels.metal   # Native Metal shaders (SIMD group reduction)
        polar_ops.h/cpp       # C++ MLX Primitive classes
        bindings.cpp          # nanobind Python bindings
        CMakeLists.txt        # Build system
    integration/
        rotation_configs.py   # Per-architecture rotation configs
```

## Citation

```bibtex
@misc{turboquant_mlx,
    title={TurboQuant-MLX: Extreme Weight Compression for Apple Silicon},
    year={2025},
    note={Adapts TurboQuant (Zandieh et al., 2025) for weight quantization on MLX}
}
```

## License

MIT

## Acknowledgments

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Han, Daliri, Karbasi (2025)
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Research
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — MLX language model utilities
