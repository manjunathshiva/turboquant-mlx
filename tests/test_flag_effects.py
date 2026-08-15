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
    """Hash every quantized parameter: name, shape, dtype and bytes.

    Shape alone is not enough — that is precisely what let a rotated-vs-
    unrotated mixup through. The bytes are the part that has to move.
    """
    items = []
    _flatten("", model.parameters(), items)
    h = hashlib.sha256()
    for name, arr in sorted(items):
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(bytes(memoryview(mx.array(arr).astype(mx.float32))))
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
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    declared.add(str(arg.value)[2:].replace("-", "_"))

    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {kw.arg for n in ast.walk(tree)
             if isinstance(n, ast.Call) for kw in n.keywords if kw.arg}

    # `dest=` renames the attribute, so honour it before declaring a miss.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    used.add(str(kw.value.value))

    orphans = sorted(f for f in declared if f not in used)
    assert not orphans, (
        f"{module}: CLI flags declared but never read: {orphans}"
    )
