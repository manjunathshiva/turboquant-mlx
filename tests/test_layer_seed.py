"""Rotation seeds must not depend on the Python process.

Two conversions of the same model must produce the same weights, and two builds
compared in an experiment must agree on every layer they quantize the same way.
"""

import subprocess
import sys

from turboquant_mlx.quantize_model import _get_layer_seed


def test_seed_is_the_same_in_a_fresh_process():
    path = "model.layers.7.mlp.switch_mlp.down_proj"
    here = _get_layer_seed(42, path)
    code = ("from turboquant_mlx.quantize_model import _get_layer_seed;"
            f"print(_get_layer_seed(42, {path!r}))")
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True).stdout.strip()
        assert int(out) == here


def test_seeds_differ_by_layer_and_base_seed():
    a = _get_layer_seed(42, "model.layers.0.self_attn.q_proj")
    assert a != _get_layer_seed(42, "model.layers.1.self_attn.q_proj")
    assert _get_layer_seed(43, "model.layers.0.self_attn.q_proj") == a + 1
