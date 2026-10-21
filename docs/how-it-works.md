# How it works

TurboQuant-MLX is an MLX implementation of Google's
[TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian and Mirrokni, 2025),
extended for large Mixture-of-Experts models on Apple Silicon.

## Quantizing a weight, with no data

Every quantized layer goes through the same steps, and none of them looks at any input text:

1. **Randomized Hadamard rotation.** Multiplying by a Hadamard matrix with random ±1 signs
   spreads every weight's value across the whole group, so whatever shape the weights had, the
   result is close to a Gaussian.
2. **Group and scale.** Weights are split into groups along the input dimension (the group
   size: 32, 64 or 128, and also 16 for MLP and expert layers via `--mlp-group-size`), and
   each group is divided by its root-mean-square.
3. **Snap to a codebook.** Each scaled weight is replaced by the nearest centroid of a
   **Lloyd-Max codebook**, the quantizer that minimizes error for a Gaussian. Because the
   rotation made the weights Gaussian, one fixed codebook per bit width works for every layer
   of every model. It's a mathematical constant, not something fitted.
4. **Pack.** The centroid indices are bit-packed into 32-bit words (2, 3 or 4 bits each), and
   the per-group scales are stored alongside.

That's why conversion needs no calibration set and runs in minutes: a 120B model converts in
about two minutes on a 64 GB Mac.

In a Mixture-of-Experts layer, all experts share the same rotation signs and codebook, so the
extra storage is only the packed indices and scales.

## The ternary tier (1.58 bits)

A 1-bit codebook (two levels, `{−c, +c}`) can't represent "this weight is close to zero", and
1-bit experts produce gibberish. Adding a zero level, `{−c, 0, +c}`, fixes that: three levels
carry log₂3 ≈ 1.58 bits. Twenty of these "trits" are packed into one 32-bit word in base 3,
about 1.6 bits per weight.

The format marks itself: a codebook with 2ᵇ entries means bit-packed; a codebook with **3**
entries means trit-packed. The loader and the kernels switch on that, so a model can mix tiers
layer by layer (ternary experts, 4-bit down projections, 3-bit protected layers) without any
extra configuration.

Ternary works when a model has **many, wide experts** to share the work: 128 or more experts
and wide expert layers. On narrow or few experts it breaks down (see
[Convert a model](guides/convert.md#mixed-precision-for-mixture-of-experts)).

## Running a quantized layer

Rotation is orthonormal, so instead of un-rotating the weights, a quantized layer **rotates its
input** and multiplies by the packed weights. Fused Metal kernels decode the packed codes inline
during the matrix multiply, so the full-precision weights never exist in memory.

Mixture-of-Experts layers pick a kernel by the shape of the work:

| Situation | Kernel |
|---|---|
| Decoding one token | `polar_gather_qmv`: each routed expert reads the shared input |
| Prefill with a different input per expert | `polar_multi_gather_qmv` |
| A large batch sorted by expert | `polar_gather_qmm`, a batched gather-GEMM |
| Anything else | `polar_dequant_experts`, then a regular matmul |

Each kernel is compiled per combination of bit width, group size and trit-or-not, and tested
against a reference dequantize-and-multiply.

**The cost:** rotating the input every step is extra work. On decode-bound runs, a 3-bit
TurboQuant model is usually slower than MLX's native affine 4-bit (about 2× on Laguna-XS.2),
and ternary is slower again (about 1.9× slower than 3-bit). TurboQuant buys **size and quality
at low bit widths**: Muse Glimmer at 4-bit beats affine 4-bit on perplexity in 25% less space,
and at 2 and 3 bits affine quantization falls apart where TurboQuant holds.

## The KV cache

The same rotation and codebook compress the KV cache at run time. Keys need more precision
than values, so the recommended setting is 8-bit keys and 3-bit values, about 4× smaller than
FP16. See [KV cache compression](guides/kv-cache.md).

## Configuration that converter and loader share

Bit widths are chosen per layer from its name (attention, MLP or expert), and the loader
recomputes the same choices when it builds the model, so the two must agree. The bit width of
each layer is recovered from the size of its saved codebook, never trusted from a config
field. The `quantization` block in `config.json` records the rest: rotation, seed, group sizes.

## Two converters

- **In memory** (`turboquant-convert`): quantizes the whole model, then saves it. Peak memory
  is about the size of the quantized model, which caps a 64 GB Mac at roughly 130B parameters
  at 3-bit.
- **Streaming** (`turboquant-convert --streaming`): writes each quantized layer to a shard and
  frees it, so memory stays near one shard plus one layer. Qwen3-235B converts on a 16 GB Mac
  mini this way, and the output is byte-identical to the in-memory converter.

## Expert streaming

For a Mixture-of-Experts model bigger than memory, only the attention, the router and the
shared parts stay resident. The experts each token routes to are read from the safetensors
files on disk (in parallel, with an LRU cache), and the result is bit-identical to running it
in memory. See [Models bigger than RAM](guides/streaming.md).
