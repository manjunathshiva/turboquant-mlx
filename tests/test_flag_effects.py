"""Every knob must actually do something.

The failure mode this file exists to catch is the one that hid three separate
bugs: a flag that is accepted, validated, stored into ``config.json``, printed
at conversion time — and never reaches the quantizer. ``--rotation`` shipped
that way for the project's entire history; ``--fuse-rotations`` shipped a
variant of it that produced word salad; ``--mlp-group-size`` silently did
nothing on dense models.

None of those were exotic code paths. They were documented CLI flags, and the
reason none of the existing tests caught them is that every test asserted on
*shapes*, which stay correct no matter which domain the numbers live in.

Two independent layers here, because a flag can break at either:

1. :func:`test_every_cli_flag_is_referenced` — static. Does the argparse flag
   reach anything at all? Catches "defined and forgotten".
2. ``test_flag_changes_output`` — behavioural. Does the config field change
   the quantized bytes? Catches "plumbed all the way through and then
   ignored", which is exactly how ``--rotation`` passed every review.
"""

import ast
import hashlib
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize

try:
    from mlx_lm.models.switch_layers import SwitchLinear
    HAVE_SWITCH = True
except ImportError:  # pragma: no cover - mlx_lm is a hard dep in practice
    HAVE_SWITCH = False

_DIM, _HIDDEN, _EXPERTS = 128, 256, 4


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)


class _Experts(nn.Module):
    def __init__(self, d, h, e):
        super().__init__()
        self.gate_proj = SwitchLinear(d, h, e, bias=False)
        self.up_proj = SwitchLinear(d, h, e, bias=False)
        self.down_proj = SwitchLinear(h, d, e, bias=False)


class _MoEMLP(nn.Module):
    def __init__(self, d, h, e):
        super().__init__()
        self.switch_mlp = _Experts(d, h, e)


class _Block(nn.Module):
    def __init__(self, d, h, e):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MoEMLP(d, h, e) if e else _DenseMLP(d, h)


class _Tiny(nn.Module):
    def __init__(self, e=0, n=2):
        super().__init__()
        self.layers = [_Block(_DIM, _HIDDEN, e) for _ in range(n)]


def _flatten(prefix, obj, out):
    if isinstance(obj, mx.array):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v, out)


def _fingerprint(model):
    """Hash every quantized parameter: name, shape, dtype and native bytes.

    Shape alone is not enough — that is precisely what let a rotated-vs-
    unrotated mixup through. The bytes are the part that has to move.

    Hashed in the parameter's OWN dtype, never widened to float32. Packed
    weights are uint32, and float32 has a 24-bit mantissa, so casting would
    round distinct packings onto the same value and make this audit report
    "unchanged" for genuinely different weights — a false pass in the exact
    direction that hides the bug being hunted.
    """
    items = []
    _flatten("", model.parameters(), items)
    h = hashlib.sha256()
    for name, arr in sorted(items):
        mx.eval(arr)
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(np.ascontiguousarray(np.array(arr, copy=False)).tobytes())
    return h.hexdigest()


def _quantize(moe, **overrides):
    mx.random.seed(0)
    model = _Tiny(e=_EXPERTS if moe else 0)
    cfg = {"model_type": "llama", "hidden_size": _DIM,
           "num_hidden_layers": 2, "num_attention_heads": 4}
    tq = TurboQuantConfig(**{"bits": 3, "group_size": 64, **overrides})
    model, _ = turboquant_quantize(model, cfg, tq)
    mx.eval(model.parameters())
    return _fingerprint(model)


# (id, moe?, config override, must the output change?)
_CASES = [
    ("bits=4",                 False, dict(bits=4),                        True),
    ("bits=2",                 False, dict(bits=2),                        True),
    ("group_size=32",          False, dict(group_size=32),                 True),
    ("rotation=none",          False, dict(rotation="none"),               True),
    ("rotation_seed=7",        False, dict(rotation_seed=7),               True),
    ("attn_bits=2",            False, dict(attn_bits=2),                   True),
    ("mlp_bits=2",             False, dict(mlp_bits=2),                    True),
    ("mlp_group_size=32",      False, dict(mlp_group_size=32),             True),
    ("use_qjl",                False, dict(use_qjl=True),                  True),
    ("mlp_bits=2 (moe)",       True,  dict(mlp_bits=2),                    True),
    ("ternary_experts (moe)",  True,  dict(ternary_experts=True),          True),
    ("expert_down_bits (moe)", True,  dict(expert_down_bits=4),            True),
    # Documented no-ops. These must NOT change anything, or the flag is
    # leaking into layers it was never meant to touch.
    ("blockwise is an alias",  False, dict(rotation="blockwise_hadamard"), False),
    ("ternary_experts (dense)", False, dict(ternary_experts=True),         False),
    ("expert_down_bits (dense)", False, dict(expert_down_bits=4),          False),
]


@pytest.mark.parametrize("label,moe,override,should_change",
                         _CASES, ids=[c[0] for c in _CASES])
def test_flag_changes_output(label, moe, override, should_change):
    if moe and not HAVE_SWITCH:
        pytest.skip("mlx_lm SwitchLinear unavailable")
    base = _quantize(moe)
    varied = _quantize(moe, **override)
    if should_change:
        assert base != varied, (
            f"{label} did not change the quantized output — the setting is "
            f"being accepted and then ignored"
        )
    else:
        assert base == varied, (
            f"{label} changed the quantized output but is documented as a "
            f"no-op — it is leaking into layers it should not touch"
        )


def test_mixed_group_sizes_survive_a_save_load_round_trip():
    """A checkpoint whose layers use DIFFERENT group sizes must reload right.

    ``--group-size 64 --mlp-group-size 32`` produces attention and MLP tiers
    with different scale shapes in one file. The loader must take each layer's
    group size from its saved scales rather than re-deriving it from the
    config rules — so the reload here is deliberately given a config whose
    rules would answer 64 for the MLP. If the derivation regresses to reading
    the config, the MLP scales are misread and the outputs diverge.
    """
    from turboquant_mlx.generate import _prepare_polar_layers

    mx.random.seed(0)
    cfg = {"model_type": "llama", "hidden_size": _DIM,
           "num_hidden_layers": 2, "num_attention_heads": 4}
    quantized, _ = turboquant_quantize(
        _Tiny(e=0), cfg,
        TurboQuantConfig(bits=3, group_size=64, mlp_group_size=32),
    )
    mx.eval(quantized.parameters())

    items = []
    _flatten("", quantized.parameters(), items)
    weights = dict(items)

    # The two tiers really are stored at different granularities.
    assert weights["layers.0.self_attn.q_proj.scales"].shape[-1] == _DIM // 64
    assert weights["layers.0.mlp.gate_proj.scales"].shape[-1] == _DIM // 32

    # Reload with a config that does NOT mention mlp_group_size: its rules say
    # 64 everywhere, so only the saved scales can give the MLP its true 32.
    reloaded = _Tiny(e=0)
    _prepare_polar_layers(reloaded, weights,
                          TurboQuantConfig(bits=3, group_size=64))
    reloaded.load_weights(list(weights.items()), strict=False)
    mx.eval(reloaded.parameters())

    assert reloaded.layers[0].self_attn.q_proj.group_size == 64
    assert reloaded.layers[0].mlp.gate_proj.group_size == 32

    x = mx.random.normal((2, _DIM)).astype(mx.float16)
    for getter in (lambda m: m.layers[0].self_attn.q_proj,
                   lambda m: m.layers[0].mlp.gate_proj):
        want = getter(quantized)(x)
        got = getter(reloaded)(x)
        assert mx.allclose(want, got, atol=1e-4).item(), (
            "mixed-group-size checkpoint did not reload identically — the "
            "loader is not taking the group size from the saved scales"
        )

    print("test_mixed_group_sizes_survive_a_save_load_round_trip: PASSED")


_ENTRY_POINTS = ["convert.py", "convert_vlm.py"]


@pytest.mark.parametrize("module", _ENTRY_POINTS)
def test_every_cli_flag_is_referenced(module):
    """Each ``--flag`` must be read somewhere in its own module.

    Static half of the audit: catches a flag that is declared and then never
    consulted. Cheap, and it runs without touching MLX at all.
    """
    src = (pathlib.Path(__file__).resolve().parents[1] / module).read_text()
    tree = ast.parse(src)

    declared = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        explicit = next(
            (str(kw.value.value) for kw in node.keywords
             if kw.arg == "dest" and isinstance(kw.value, ast.Constant)),
            None,
        )
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                # argparse derives the attribute from the option name unless
                # dest= overrides it; check whichever one actually gets read.
                declared.add(explicit or str(arg.value)[2:].replace("-", "_"))

    # Reads only. Deliberately NOT seeded with the dest= values themselves:
    # declaring `dest="x"` is not evidence that anything ever reads `args.x`,
    # and counting it as such would make every dest-renamed flag pass for free.
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    orphans = sorted(f for f in declared if f not in used)
    assert not orphans, (
        f"{module}: CLI flags declared but never read: {orphans}"
    )
