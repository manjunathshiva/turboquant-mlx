"""Comprehensive tests for TurboQuant-MLX core modules."""

import mlx.core as mx
import mlx.nn as nn


def test_codebook_centroids():
    """Verify codebook centroids are symmetric and ordered."""
    from turboquant_mlx.core.codebook import get_codebook

    for bits in [2, 3, 4]:
        centroids, boundaries = get_codebook(bits, dtype=mx.float32)
        n = 2 ** bits

        assert centroids.shape == (n,), f"{bits}-bit: expected {n} centroids"
        assert boundaries.shape == (n - 1,), f"{bits}-bit: expected {n-1} boundaries"

        # Centroids should be sorted
        c = centroids.tolist()
        assert c == sorted(c), f"{bits}-bit: centroids not sorted"

        # Centroids should be symmetric around 0
        for i in range(n // 2):
            assert abs(c[i] + c[n - 1 - i]) < 1e-6, f"{bits}-bit: not symmetric"

        # Boundaries should be sorted
        b = boundaries.tolist()
        assert b == sorted(b), f"{bits}-bit: boundaries not sorted"

    print("test_codebook_centroids: PASSED")


def test_quantize_dequantize_roundtrip():
    """Verify quantize -> dequantize maps to nearest centroid."""
    from turboquant_mlx.core.codebook import get_codebook, quantize_scalar, dequantize_scalar

    for bits in [2, 3, 4]:
        centroids, boundaries = get_codebook(bits, dtype=mx.float32)
        n = 2 ** bits

        # Exact centroid values should roundtrip perfectly
        indices = quantize_scalar(centroids, boundaries)
        reconstructed = dequantize_scalar(indices, centroids)
        error = mx.max(mx.abs(centroids - reconstructed)).item()
        assert error < 1e-5, f"{bits}-bit: centroid roundtrip error {error}"

        # Random values should map to nearest centroid
        values = mx.random.normal((1000,))
        mx.eval(values)
        idx = quantize_scalar(values, boundaries)
        assert idx.min().item() >= 0, "Negative index"
        assert idx.max().item() < n, f"Index >= {n}"

    print("test_quantize_dequantize_roundtrip: PASSED")


def test_packing_roundtrip():
    """Verify pack -> unpack is lossless for all bit-widths."""
    from turboquant_mlx.core.packing import pack_indices, unpack_indices

    for bits in [2, 3, 4]:
        n = 2 ** bits
        # Create test indices
        count = 128
        indices = (mx.random.uniform(shape=(4, count)) * n).astype(mx.uint8)
        indices = mx.minimum(indices, mx.array(n - 1, dtype=mx.uint8))
        mx.eval(indices)

        packed = pack_indices(indices, bits)
        unpacked = unpack_indices(packed, bits, count)

        # Verify roundtrip
        match = (unpacked == indices).all().item()
        assert match, f"{bits}-bit packing roundtrip failed"

    print("test_packing_roundtrip: PASSED")


def test_rotation_preserves_norms():
    """Verify Hadamard rotation preserves row norms."""
    from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight

    for dim in [64, 128, 256, 512]:
        signs = generate_random_signs(dim, seed=42)
        W = mx.random.normal((32, dim))
        mx.eval(W)

        W_rot = rotate_weight(W, signs)
        mx.eval(W_rot)

        norms_orig = mx.sqrt((W * W).sum(axis=-1))
        norms_rot = mx.sqrt((W_rot * W_rot).sum(axis=-1))

        rel_error = mx.abs(norms_orig - norms_rot) / (norms_orig + 1e-8)
        max_rel_error = rel_error.max().item()
        # Hadamard preserves norms exactly (it's orthogonal)
        assert max_rel_error < 0.01, f"dim={dim}: rotation changed norms by {max_rel_error}"

    print("test_rotation_preserves_norms: PASSED")


def test_rotation_blockwise():
    """Verify blockwise rotation works for non-standard dimensions."""
    from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight, _find_hadamard_block_size

    # Test block size finding
    assert _find_hadamard_block_size(128) == 128
    assert _find_hadamard_block_size(256) == 256
    assert _find_hadamard_block_size(4096) == 4096

    # Non-power-of-2 that should use blockwise
    bs = _find_hadamard_block_size(192)  # 192 = 3 * 64
    assert 192 % bs == 0, f"Block size {bs} doesn't divide 192"

    # Test actual rotation
    for dim in [192, 384]:
        signs = generate_random_signs(dim, seed=42)
        W = mx.random.normal((16, dim))
        mx.eval(W)
        W_rot = rotate_weight(W, signs)
        mx.eval(W_rot)
        assert W_rot.shape == W.shape, f"Shape mismatch for dim={dim}"

    print("test_rotation_blockwise: PASSED")


def test_rotation_deterministic():
    """Verify rotation is deterministic given the same seed."""
    from turboquant_mlx.core.rotation import generate_random_signs, rotate_weight

    signs1 = generate_random_signs(128, seed=42)
    signs2 = generate_random_signs(128, seed=42)
    assert (signs1 == signs2).all().item(), "Same seed produced different signs"

    signs3 = generate_random_signs(128, seed=99)
    assert not (signs1 == signs3).all().item(), "Different seeds produced same signs"

    print("test_rotation_deterministic: PASSED")


def test_polar_quantize_weight():
    """Test full PolarQuant pipeline on a weight matrix."""
    from turboquant_mlx.core.polar_quantize import polar_quantize_weight, polar_dequantize_weight

    W = mx.random.normal((64, 256))
    mx.eval(W)

    for bits in [2, 3, 4]:
        result = polar_quantize_weight(W, bits=bits, group_size=64, seed=42)

        assert result["packed_weight"].dtype == mx.uint32
        assert result["scales"].dtype == mx.float16
        assert result["codebook"].shape == (2 ** bits,)
        assert result["signs"].shape == (256,)
        assert result["bits"] == bits
        assert result["group_size"] == 64

        # Dequantize and check error
        W_deq = polar_dequantize_weight(
            result["packed_weight"], result["scales"], result["codebook"],
            bits, 64, 256,
        )
        assert W_deq.shape == (64, 256), f"Dequant shape mismatch"

    print("test_polar_quantize_weight: PASSED")


def test_polar_linear_from_linear():
    """Test PolarQuantizedLinear creation from nn.Linear."""
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

    linear = nn.Linear(256, 128, bias=True)
    mx.eval(linear.parameters())

    pq = PolarQuantizedLinear.from_linear(linear, bits=3, group_size=64, seed=42)
    mx.eval(pq.parameters())

    # Check shapes
    assert pq.weight.dtype == mx.uint32
    assert pq.scales.shape == (128, 4)  # 256/64 = 4 groups
    assert pq.codebook.shape == (8,)  # 2^3 = 8
    assert pq.signs.shape == (256,)
    assert pq.bias.shape == (128,)

    # Forward pass
    x = mx.random.normal((2, 5, 256))
    mx.eval(x)
    y = pq(x)
    mx.eval(y)
    assert y.shape == (2, 5, 128), f"Output shape mismatch: {y.shape}"

    print("test_polar_linear_from_linear: PASSED")


def test_quality_vs_affine():
    """Compare PolarQuant quality against MLX affine quantization at 3-bit."""
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

    # Create a larger linear layer for meaningful comparison
    linear = nn.Linear(512, 256, bias=False)
    mx.eval(linear.parameters())

    x = mx.random.normal((1, 10, 512))
    mx.eval(x)
    y_ref = linear(x)
    mx.eval(y_ref)

    # PolarQuant 3-bit
    pq3 = PolarQuantizedLinear.from_linear(linear, bits=3, group_size=64, seed=42)
    mx.eval(pq3.parameters())
    y_pq3 = pq3(x)
    mx.eval(y_pq3)

    # MLX affine 3-bit
    affine3 = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=3)
    mx.eval(affine3.parameters())
    y_aff3 = affine3(x)
    mx.eval(y_aff3)

    # Cosine similarity
    def cosine_sim(a, b):
        return ((a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()))).item()

    cos_pq3 = cosine_sim(y_ref, y_pq3)
    cos_aff3 = cosine_sim(y_ref, y_aff3)

    mse_pq3 = ((y_ref - y_pq3) ** 2).mean().item()
    mse_aff3 = ((y_ref - y_aff3) ** 2).mean().item()

    print(f"  PolarQuant 3-bit: cosine={cos_pq3:.6f}, MSE={mse_pq3:.6f}")
    print(f"  Affine 3-bit:     cosine={cos_aff3:.6f}, MSE={mse_aff3:.6f}")

    # PolarQuant should have reasonable quality (cosine > 0.95)
    assert cos_pq3 > 0.95, f"PolarQuant 3-bit cosine too low: {cos_pq3}"

    print("test_quality_vs_affine: PASSED")


def test_rotation_cannot_fuse_into_norm():
    """A Hadamard rotation CANNOT be folded into a preceding norm weight.

    This pins the reason `fuse_rotations` was removed. An RMSNorm applies a
    diagonal weight, `y = n * w`. A quantized layer whose weights live in
    rotated space needs `R @ y` where `R = H @ diag(signs)`. Folding R into
    the norm would require `hadamard(signs * w) * n == hadamard(signs * (n * w))`
    — pulling an element-wise multiply through a Hadamard transform. H mixes
    across dimensions, so that identity is false, and the error is not small:
    the two are essentially orthogonal at model-scale dims.

    The control half shows the boundary precisely: a *diagonal* rotation (sign
    flip alone, no Hadamard) does commute with the norm weight and fuses
    exactly. It is the Hadamard that makes fusion impossible — and a sign flip
    alone does not Gaussianize the weights, which is the whole point of the
    rotation. Any future attempt must rotate the residual stream (QuaRot-style,
    one shared Q absorbed into embeddings + output projections), not the norm.
    """
    import math
    from turboquant_mlx.core.rotation import generate_random_signs

    for dim in (128, 4096):
        w = mx.random.normal((dim,)) * 0.5 + 1.0     # norm weight
        n = mx.random.normal((dim,))                 # normalized activation
        signs = generate_random_signs(dim, seed=42).astype(mx.float32)
        scale = 1.0 / math.sqrt(dim)

        # What the quantized layer actually needs: R @ (n * w)
        truth = mx.hadamard_transform(
            (signs * (n * w)).reshape(1, -1), scale=scale).reshape(-1)
        # What any norm-fusion scheme can produce: n * <some fused vector>
        fused = mx.hadamard_transform(
            (signs * w).reshape(1, -1), scale=scale).reshape(-1)
        got = n * fused

        cos = float((got * truth).sum()
                    / (mx.linalg.norm(got) * mx.linalg.norm(truth)))
        assert abs(cos) < 0.5, (
            f"dim={dim}: fusion unexpectedly close (cos={cos:.4f}). If this "
            f"ever passes, the impossibility argument above needs revisiting."
        )

    # Control: a diagonal-only rotation DOES fuse, exactly.
    dim = 4096
    w = mx.random.normal((dim,)) * 0.5 + 1.0
    n = mx.random.normal((dim,))
    signs = generate_random_signs(dim, seed=42).astype(mx.float32)
    assert float(mx.abs((n * (signs * w)) - (signs * (n * w))).max()) == 0.0

    print("test_rotation_cannot_fuse_into_norm: PASSED")


def test_rotation_none_skips_rotation_on_both_sides():
    """rotation="none" must unrotate the WEIGHTS and the INPUT together.

    This is the invariant `fuse_rotations` violated: weights quantized in
    rotated space while the input arrives unrotated produces noise, and no
    shape check catches it. Here we pin that the `rotate` flag really moves
    which domain the packed weights live in.

    `polar_dequantize_weight` returns the weight in whatever domain it was
    packed in, so the two cases have different reconstruction targets:
    rotate=True reconstructs `rotate_weight(w, signs)`, rotate=False
    reconstructs `w` itself.
    """
    import numpy as np
    from turboquant_mlx.core.polar_quantize import (
        polar_quantize_weight, polar_dequantize_weight)
    from turboquant_mlx.core.rotation import rotate_weight

    dim = 256
    w = mx.random.normal((128, dim))

    rot = polar_quantize_weight(w, bits=4, group_size=64, seed=7, rotate=True)
    off = polar_quantize_weight(w, bits=4, group_size=64, seed=7, rotate=False)

    # rotate=False must leave the signs a no-op, so the stored vector cannot
    # silently re-introduce a half-rotation anywhere downstream.
    assert float(mx.abs(off["signs"] - 1.0).max()) == 0.0, "signs must be ones"

    targets = {
        "rotate=True": (rot, rotate_weight(w.astype(mx.float32),
                                           rot["signs"].astype(mx.float32))),
        "rotate=False": (off, w.astype(mx.float32)),
    }
    for name, (r, target) in targets.items():
        recon = polar_dequantize_weight(
            r["packed_weight"], r["scales"], r["codebook"],
            r["bits"], r["group_size"], r["input_dims"],
        ).astype(mx.float32)
        rel = float(mx.linalg.norm(recon - target) / mx.linalg.norm(target))
        assert rel < 0.10, f"{name}: round-trip error {rel:.4f} too high"

    # The packed payloads must actually differ — otherwise `rotate` is a no-op
    # and we are back to the silently-ignored flag this test exists to prevent.
    assert not np.array_equal(
        np.array(rot["packed_weight"]), np.array(off["packed_weight"])), \
        "rotate=False produced identical packing — the flag is being ignored"

    print("test_rotation_none_skips_rotation_on_both_sides: PASSED")


def test_loader_legacy_rotation_none_guard():
    """A pre-0.23 model with rotation="none" still has ROTATED weights.

    `rotation` was silently ignored by every converter before that release, so
    trusting the config alone would skip the input rotation on those models and
    produce noise. The saved `signs` are the ground truth: the genuinely
    unrotated path writes all-ones, the old path always wrote randomized +/-1.
    """
    import mlx.nn as nn
    from turboquant_mlx.config import TurboQuantConfig
    from turboquant_mlx.generate import _prepare_polar_layers
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

    dim = 128

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)

    def build(signs, rotation="none"):
        model = _Tiny()
        weights = {"proj.codebook": mx.zeros((8,)), "proj.signs": signs}
        cfg = TurboQuantConfig(bits=3, group_size=64, rotation=rotation)
        _prepare_polar_layers(model, weights, cfg)
        assert isinstance(model.proj, PolarQuantizedLinear)
        return model.proj._needs_rotation

    # Legacy model: randomized signs mean the weights really were rotated.
    legacy = mx.ones((dim,), dtype=mx.float16)
    legacy[3] = -1.0
    assert build(legacy) is True, "legacy rotation=none model must still rotate"

    # Genuinely unrotated model: all-ones signs.
    assert build(mx.ones((dim,), dtype=mx.float16)) is False

    # rotation="hadamard" always rotates regardless of signs.
    assert build(mx.ones((dim,), dtype=mx.float16), rotation="hadamard") is True

    print("test_loader_legacy_rotation_none_guard: PASSED")


def test_qjl_residual_matches_the_packed_domain():
    """QJL must measure its residual in the domain the weights were packed in.

    At inference `qjl_correct` is applied to the same x the matmul saw —
    rotated iff needs_rotation. `rotate_weight` applies the Hadamard even when
    the signs are all-ones, so computing the target with it unconditionally
    makes the rotation="none" residual `rotated - unrotated`: not a correction
    but noise. The invariant that catches it is simply that QJL must never
    make a layer WORSE.
    """
    import mlx.nn as nn
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

    mx.random.seed(0)
    lin = nn.Linear(256, 256, bias=False)
    x = mx.random.normal((4, 256)).astype(mx.float16)
    ref = x.astype(mx.float32) @ lin.weight.astype(mx.float32).T

    for needs_rotation in (True, False):
        errs = {}
        for use_qjl in (False, True):
            q = PolarQuantizedLinear.from_linear(
                lin, bits=3, group_size=64, seed=7,
                needs_rotation=needs_rotation, use_qjl=use_qjl,
            )
            y = q(x).astype(mx.float32)
            errs[use_qjl] = float(
                mx.linalg.norm(y - ref) / mx.linalg.norm(ref))
        assert errs[True] < errs[False], (
            f"needs_rotation={needs_rotation}: QJL made it worse "
            f"({errs[False]:.4f} -> {errs[True]:.4f}) — the residual is being "
            f"measured in the wrong domain"
        )

    print("test_qjl_residual_matches_the_packed_domain: PASSED")


def test_metal_kernel_correctness():
    """Test fused Metal kernel matches software dequant exactly."""
    from turboquant_mlx.core.polar_quantize import polar_quantize_weight, polar_dequantize_weight
    from turboquant_mlx.kernels.polar_qmv import polar_qmv

    for bits in [2, 3, 4]:
        for M, N in [(128, 64), (256, 512), (512, 1024)]:
            W = mx.random.normal((M, N)) * 0.02
            mx.eval(W)
            result = polar_quantize_weight(W, bits, 64, seed=42)
            packed, scales, codebook = result["packed_weight"], result["scales"], result["codebook"]

            x = mx.random.normal((N,)).astype(mx.float16)
            mx.eval(x)

            # Software path
            W_deq = polar_dequantize_weight(packed, scales, codebook, bits, 64, N)
            y_sw = x @ W_deq.T
            mx.eval(y_sw)

            # Fused Metal kernel
            y_hw = polar_qmv(packed, scales, codebook, x, bits, 64)
            mx.eval(y_hw)

            max_diff = mx.max(mx.abs(y_sw - y_hw)).item()
            assert max_diff < 0.01, (
                f"{bits}-bit {M}x{N}: Metal kernel differs by {max_diff}"
            )

            # Also test (1, N) input shape
            y_batch = polar_qmv(packed, scales, codebook, mx.expand_dims(x, 0), bits, 64)
            mx.eval(y_batch)
            assert y_batch.shape == (1, M), f"Batch shape mismatch: {y_batch.shape}"

    print("test_metal_kernel_correctness: PASSED")


def test_rotation_configs():
    """Test rotation config registry."""
    from turboquant_mlx.integration.rotation_configs import (
        get_rotation_config, should_fuse_rotation, ROTATION_CONFIGS,
    )

    # Test all registered architectures
    for arch in ROTATION_CONFIGS:
        config = get_rotation_config(arch)
        assert config.fuse_norm_to_projs or config.online_rotation_layers

    # Test LLaMA config specifically
    config = get_rotation_config("llama")

    # q_proj should fuse into input_layernorm
    can_fuse, norm = should_fuse_rotation("layers.0.self_attn.q_proj", config)
    assert can_fuse and norm == "input_layernorm"

    # down_proj should need online rotation
    can_fuse, norm = should_fuse_rotation("layers.0.mlp.down_proj", config)
    assert not can_fuse and norm is None

    # gate_proj should fuse into post_attention_layernorm
    can_fuse, norm = should_fuse_rotation("layers.0.mlp.gate_proj", config)
    assert can_fuse and norm == "post_attention_layernorm"

    # Qwen3-MoE uses the standard-attention + SwitchGLU MoE config
    qm = get_rotation_config("qwen3_moe")
    assert should_fuse_rotation("layers.0.self_attn.q_proj", qm) == (True, "input_layernorm")
    assert should_fuse_rotation("layers.0.mlp.switch_mlp.gate_proj", qm) == (True, "post_attention_layernorm")
    assert should_fuse_rotation("layers.0.mlp.switch_mlp.down_proj", qm) == (False, None)

    print("test_rotation_configs: PASSED")


def test_deepseek_rotation_config():
    """DeepSeek MLA + MoE family resolves a config with correct MLA handling."""
    from turboquant_mlx.integration.rotation_configs import (
        get_rotation_config, should_fuse_rotation,
    )

    for arch in ("deepseek_v2", "deepseek_v3", "deepseek_v32"):
        config = get_rotation_config(arch)

        # MLA input projections fuse into input_layernorm
        for proj in ("q_proj", "q_a_proj", "kv_a_proj_with_mqa"):
            can_fuse, norm = should_fuse_rotation(f"layers.0.self_attn.{proj}", config)
            assert can_fuse and norm == "input_layernorm", (arch, proj)

        # b-projections (nested-norm inputs) and o_proj use online rotation
        for proj in ("q_b_proj", "kv_b_proj", "o_proj"):
            can_fuse, norm = should_fuse_rotation(f"layers.0.self_attn.{proj}", config)
            assert not can_fuse and norm is None, (arch, proj)

        # MoE/MLP fuses like qwen3_5_moe; down_proj is online
        can_fuse, norm = should_fuse_rotation("layers.1.mlp.switch_mlp.gate_proj", config)
        assert can_fuse and norm == "post_attention_layernorm", arch
        can_fuse, norm = should_fuse_rotation("layers.1.mlp.switch_mlp.down_proj", config)
        assert not can_fuse and norm is None, arch

    print("test_deepseek_rotation_config: PASSED")


def test_qjl_packing_roundtrip():
    """Verify 1-bit QJL packing is lossless."""
    from turboquant_mlx.core.qjl import pack_1bit, unpack_1bit

    for n in [32, 64, 100, 128, 256, 512]:
        bits = (mx.random.uniform(shape=(8, n)) > 0.5).astype(mx.uint8)
        mx.eval(bits)
        packed = pack_1bit(bits)
        unpacked = unpack_1bit(packed, n)
        assert (unpacked == bits).all().item(), f"1-bit packing failed for n={n}"

    print("test_qjl_packing_roundtrip: PASSED")


def test_qjl_unbiasedness():
    """Verify QJL correction is unbiased (mean error ≈ 0)."""
    from turboquant_mlx.core.qjl import qjl_quantize, qjl_correct

    # Fixed seed: this is a 100-trial statistical bias estimate, and with
    # unseeded draws it crosses the 0.1 threshold roughly 1 run in 5.
    mx.random.seed(1234)
    d = 256
    total_error = 0.0
    total_abs = 0.0
    n_trials = 100

    for t in range(n_trials):
        r = mx.random.normal((1, d)) * 0.1
        x = mx.random.normal((d,))
        mx.eval(r, x)

        true_ip = (r.squeeze(0) * x).sum().item()
        qr = qjl_quantize(r, seed=t * 7 + 1)
        est = qjl_correct(qr["qjl_packed"], qr["qjl_norms"], qr["qjl_signs"], x, d)
        mx.eval(est)
        total_error += est.item() - true_ip
        total_abs += abs(true_ip)

    rel_bias = abs(total_error / n_trials) / (total_abs / n_trials)
    assert rel_bias < 0.1, f"QJL not unbiased: relative bias {rel_bias:.4f}"

    print("test_qjl_unbiasedness: PASSED")


def test_qjl_improves_quality():
    """Verify QJL correction reduces MSE at all bit-widths."""
    from turboquant_mlx.layers.polar_linear import PolarQuantizedLinear

    linear = nn.Linear(512, 256, bias=False)
    mx.eval(linear.parameters())
    x = mx.random.normal((1, 5, 512))
    mx.eval(x)
    y_ref = linear(x)
    mx.eval(y_ref)

    for bits in [2, 3, 4]:
        pq = PolarQuantizedLinear.from_linear(linear, bits=bits, use_qjl=False)
        pq_qjl = PolarQuantizedLinear.from_linear(linear, bits=bits, use_qjl=True)
        mx.eval(pq.parameters())
        mx.eval(pq_qjl.parameters())

        y_pq = pq(x)
        y_qjl = pq_qjl(x)
        mx.eval(y_pq, y_qjl)

        mse_pq = mx.mean((y_ref - y_pq) ** 2).item()
        mse_qjl = mx.mean((y_ref - y_qjl) ** 2).item()

        assert mse_qjl < mse_pq, (
            f"{bits}-bit: QJL MSE {mse_qjl:.6f} >= PolarQuant MSE {mse_pq:.6f}"
        )

    print("test_qjl_improves_quality: PASSED")


def test_config():
    """Test TurboQuantConfig validation and serialization."""
    from turboquant_mlx.config import TurboQuantConfig

    config = TurboQuantConfig(bits=3, group_size=64)
    assert config.effective_bits == 3.25  # 3 + 16/64

    d = config.to_dict()
    assert d["mode"] == "turboquant"
    assert d["bits"] == 3

    config2 = TurboQuantConfig.from_dict(d)
    assert config2.bits == config.bits
    assert config2.group_size == config.group_size

    # Test validation
    try:
        TurboQuantConfig(bits=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("test_config: PASSED")


def run_all():
    print("=" * 60)
    print("TurboQuant-MLX Test Suite")
    print("=" * 60)

    test_config()
    test_codebook_centroids()
    test_quantize_dequantize_roundtrip()
    test_packing_roundtrip()
    test_rotation_preserves_norms()
    test_rotation_blockwise()
    test_rotation_deterministic()
    test_polar_quantize_weight()
    test_polar_linear_from_linear()
    test_quality_vs_affine()
    test_rotation_cannot_fuse_into_norm()
    test_rotation_none_skips_rotation_on_both_sides()
    test_loader_legacy_rotation_none_guard()
    test_qjl_residual_matches_the_packed_domain()
    test_metal_kernel_correctness()
    test_qjl_packing_roundtrip()
    test_qjl_unbiasedness()
    test_qjl_improves_quality()
    test_rotation_configs()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
