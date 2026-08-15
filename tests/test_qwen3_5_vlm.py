"""Qwen3.5-family VLM support (``model_type: qwen3_5``, e.g. Qwen3.8-27B).

The one thing that is easy to get silently wrong here is ``lm_head``. With a
248320-token vocabulary it is 248320x5120 = 1.271B params, the largest matrix
in the model by a wide margin. ``PolarQuantizedLinear`` has a fused Metal
kernel only up to ``_QMM_MAX_TOKENS`` tokens; above that it falls back to
dequantize-then-GEMM, which materializes the weight through several full-size
intermediates (~14 bytes per parameter).

Measured on a real PolarQuantizedLinear of that exact shape, 3-bit / g64:

    tokens    1  ->   +0.000 GiB
    tokens  256  ->   +0.116 GiB
    tokens  257  ->  +13.141 GiB     <-- fused kernel switches off
    tokens 2048  ->  +13.953 GiB

The same shape as an 8-bit affine layer costs +1.895 GiB at 2048 tokens, which
is just the logits array itself — affine has a fused batched matmul and
materializes no weight at all.

Why this hides: mlx-vlm's chunked prefill calls the model and *discards* the
return value, so MLX never evaluates the lm_head matmul. But chunking only
engages when the prompt exceeds ``prefill_step_size`` (default 2048). A prompt
of 257..2048 tokens takes the single-shot branch in ``generate/ar.py``, which
slices ``logits[:, -1, :]`` and therefore does evaluate lm_head across the whole
sequence. That is an ordinary chat turn.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

import turboquant_mlx.quantize_model as _qm
from turboquant_mlx.integration.vlm import VLM_SKIP_PATTERNS
from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear, _QMM_MAX_TOKENS


def _predicate():
    """The converter's real predicate for qwen3_5, or skip without mlx-vlm."""
    pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")
    from turboquant_mlx.integration.vlm import vlm_should_quantize

    # Read the base predicate off the module rather than binding it at import.
    # `convert_vlm` swaps `_qm._should_quantize` for the duration of a
    # conversion, so a module-level `from ... import _should_quantize` would
    # pin the pre-patch value and stop reflecting what the converter uses.
    return vlm_should_quantize("qwen3_5", _qm._should_quantize)


def test_lm_head_is_kept_off_the_polar_path():
    assert "lm_head" in VLM_SKIP_PATTERNS["qwen3_5"]

    predicate = _predicate()
    lm_head = nn.Linear(5120, 248320, bias=False)
    assert not predicate("language_model.lm_head", lm_head), (
        "qwen3_5 lm_head must not be polar-quantized — above the fused "
        "kernel's token bound it materializes ~14 bytes per parameter, "
        "measured at +13.1 GiB for this shape"
    )


def test_the_rest_of_the_model_is_still_quantized():
    """The skip must be surgical: everything else still has to compress."""
    predicate = _predicate()
    base = "language_model.model.layers.0"
    for path, (out_dims, in_dims) in {
        f"{base}.mlp.gate_proj": (17408, 5120),
        f"{base}.mlp.down_proj": (5120, 17408),
        f"{base}.self_attn.q_proj": (12288, 5120),
        f"{base}.linear_attn.in_proj_qkv": (10240, 5120),
        f"{base}.linear_attn.out_proj": (5120, 6144),
    }.items():
        assert predicate(path, nn.Linear(in_dims, out_dims, bias=False)), (
            f"{path} should still be polar-quantized"
        )


def test_vision_tower_stays_full_precision():
    """mlx-vlm's own multimodal skip has to survive our wrapper."""
    predicate = _predicate()
    assert not predicate(
        "vision_tower.blocks.0.attn.qkv", nn.Linear(1152, 3456, bias=True)
    )


@pytest.mark.parametrize("bad", ["0", "-1", "-256"])
def test_prefill_step_size_rejects_non_positive(bad):
    """``--prefill-step-size 0`` would spin mlx-vlm's prefill loop forever.

    The loop is ``while inputs_embeds.shape[1] > 1: n = min(step, len - 1);
    embeds = embeds[:, n:]``. At step 0 the slice removes nothing, so the length
    never falls and it never terminates; a negative step slices from the wrong
    end. mlx-vlm validates this in its diffusion path but not the
    autoregressive one, so the flag has to reject it at parse time.
    """
    from turboquant_mlx.generate_vlm import _positive_int

    with pytest.raises(Exception) as exc:
        _positive_int(bad)
    assert "positive" in str(exc.value)


@pytest.mark.parametrize("good", ["1", "256", "2048"])
def test_prefill_step_size_accepts_positive(good):
    from turboquant_mlx.generate_vlm import _positive_int

    assert _positive_int(good) == int(good)


def test_the_fused_kernel_bound_is_what_makes_the_skip_load_bearing():
    """The cliff is real, and it is a step function at ``_QMM_MAX_TOKENS``.

    Run at a small shape so this stays cheap — the mechanism is the point, and
    it is the same mechanism that costs 13 GiB at lm_head's true size. If MLX
    ever gains a fused batched polar matmul this test fails, and the lm_head
    skip can be revisited.
    """
    dims = 512
    layer = PolarQuantizedLinear.from_linear(
        nn.Linear(dims, dims, bias=False),
        bits=3, group_size=64, seed=1, needs_rotation=True,
    )
    mx.eval(layer.parameters())

    def transient(tokens):
        x = mx.random.normal((tokens, dims)).astype(mx.float16)
        mx.eval(x)
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        mx.eval(layer(x))
        return mx.get_peak_memory() - before

    fused = transient(_QMM_MAX_TOKENS)
    fallback = transient(_QMM_MAX_TOKENS + 1)
    assert fallback > fused, (
        f"expected a materialization step above {_QMM_MAX_TOKENS} tokens; "
        f"got {fused} bytes at the bound and {fallback} just past it"
    )
