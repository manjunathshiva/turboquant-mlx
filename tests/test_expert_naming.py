"""Which safetensors keys count as streamable expert weights.

The companion to ``test_stream_switch_attr.py``: that one pins the *module*
swap, this one pins the *byte accounting* over the same two naming conventions.
Both halves have to agree, and once they didn't — the swap learned laguna's
``mlp.experts`` container while ``plan.py`` and ``_streamed_expert_bytes`` kept
matching only ``switch_mlp``. Laguna therefore streamed correctly but reported
zero streamable bytes, so ``turboquant-plan`` called a streamable model
resident-only (❌ on a 16 GB Mac that in fact runs it at a 7.3 GB peak) and
``--cache-budget-gb auto`` sized its cache from a resident figure that wrongly
included all 28.7 GB of experts.
"""

import json
import os

from turboquant_mlx.expert_naming import is_streamed_expert_key
from turboquant_mlx.plan import footprint

# Real key shapes, as emitted by the converter.
_QWEN = "model.layers.3.mlp.switch_mlp.{proj}.{suffix}"      # qwen3_5_moe/deepseek
_LAGUNA = "model.layers.3.mlp.experts.{proj}.{suffix}"       # laguna SwitchGLU
_SHARED = "model.layers.3.mlp.shared_experts.{proj}.{suffix}"  # dense, resident


def test_counts_switch_mlp_keys():
    for suffix in ("weight", "scales"):
        assert is_streamed_expert_key(_QWEN.format(proj="gate_proj",
                                                   suffix=suffix))


def test_counts_laguna_experts_keys():
    """The regression: `mlp.experts` is a stacked-expert container too."""
    for suffix in ("weight", "scales"):
        assert is_streamed_expert_key(_LAGUNA.format(proj="down_proj",
                                                     suffix=suffix))


def test_never_counts_shared_experts():
    """`shared_experts` is a dense always-on MLP. It contains the substring
    "experts", so an unanchored match would page a tensor every token needs —
    and would overstate the streaming saving by the whole shared-expert stack."""
    for suffix in ("weight", "scales"):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert not is_streamed_expert_key(_SHARED.format(proj=proj,
                                                             suffix=suffix))


def test_never_counts_resident_codebook_and_signs():
    """A trit/codebook expert layer keeps its 3-entry codebook and rotation
    signs resident on the StreamingSwitchLinear — they are not paged."""
    for suffix in ("codebook", "signs"):
        assert not is_streamed_expert_key(_LAGUNA.format(proj="gate_proj",
                                                         suffix=suffix))
        assert not is_streamed_expert_key(_QWEN.format(proj="gate_proj",
                                                       suffix=suffix))


def test_never_counts_dense_or_router_keys():
    for key in ("model.layers.3.mlp.gate_proj.weight",       # dense layer 0 MLP
                "model.layers.3.mlp.gate.weight",            # router
                "model.layers.3.mlp.gate.e_score_correction_bias",
                "model.layers.3.self_attn.q_proj.weight",
                "model.embed_tokens.weight"):
        assert not is_streamed_expert_key(key)


def _index(template, n_layers=2, itemsize_shape=(4, 8)):
    """A fake safetensors index: {name: (dtype, shape)}, as plan.footprint wants."""
    idx = {}
    for i in range(n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix, dtype in (("weight", "uint32"), ("scales", "float16"),
                                  ("codebook", "float16"), ("signs", "float16")):
                key = template.format(proj=proj, suffix=suffix).replace(
                    "layers.3", f"layers.{i}")
                idx[key] = (dtype, itemsize_shape)
    return idx


def test_footprint_reports_laguna_experts_as_streamable():
    """plan.footprint must not call a streamable MoE resident-only."""
    fp = footprint(_index(_LAGUNA))
    assert fp["expert_bytes"] > 0, "laguna experts counted as resident"
    assert fp["resident_bytes"] == fp["total_bytes"] - fp["expert_bytes"]
    # is_moe is derived from this in plan.build_plan
    assert fp["expert_bytes"] > 0


def test_footprint_keeps_shared_experts_resident():
    fp = footprint(_index(_SHARED))
    assert fp["expert_bytes"] == 0
    assert fp["resident_bytes"] == fp["total_bytes"]


def test_plan_and_loader_agree_on_the_same_index():
    """The drift guard. Both modules must classify identical keys identically —
    a planner that disagrees with the loader mispredicts the thing it exists to
    predict."""
    from turboquant_mlx.stream.loader import _streamed_expert_bytes

    class _Loc:
        def __init__(self, dtype, shape):
            self.dtype, self.shape = dtype, shape

    class _Reader:
        def __init__(self, idx):
            self._index = {k: _Loc(*v) for k, v in idx.items()}

    for template in (_QWEN, _LAGUNA, _SHARED):
        idx = _index(template)
        assert footprint(idx)["expert_bytes"] == _streamed_expert_bytes(
            _Reader(idx)), f"plan/loader disagree on {template}"


def test_calibrate_expert_cost_is_nonzero_for_either_container(tmp_path):
    """`_model_expert_info` sizes the hot-expert pin list. Returning 0 bytes per
    expert makes the `used + cost > cap` budget check never fire, so *every*
    expert lands in hot_experts.json claiming ~0 GB — and the loader then tries
    to pin the whole expert stack into a wired cache sized for a fraction of it."""
    import mlx.core as mx

    from turboquant_mlx.stream.calibrate_experts import _model_expert_info

    for container in ("switch_mlp", "experts"):
        d = tmp_path / container
        d.mkdir()
        t = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            t[f"model.layers.0.mlp.{container}.{proj}.weight"] = mx.zeros(
                (8, 4, 2), dtype=mx.uint32)
            t[f"model.layers.0.mlp.{container}.{proj}.scales"] = mx.zeros(
                (8, 4, 1), dtype=mx.float16)
        mx.eval(*t.values())
        mx.save_safetensors(str(d / "model.safetensors"), t)
        (d / "config.json").write_text("{}")

        cost, num_experts = _model_expert_info(str(d))
        assert num_experts == 8, f"{container}: expert axis not found"
        assert cost > 0, f"{container}: 0 bytes/expert defeats the pin budget"


def test_router_and_experts_are_classified_consistently():
    """Cross-module invariant: any layer whose router rows repack_experts will
    permute must also have expert stacks it will permute. Violating this is not
    a missed optimization — it reroutes every token."""
    from turboquant_mlx.stream.repack_experts import _reorder_axis0_keys

    for container in ("switch_mlp", "experts"):
        router = "model.layers.0.mlp.gate.weight"
        expert = f"model.layers.0.mlp.{container}.gate_proj.weight"
        assert _reorder_axis0_keys(router), "router always permutes"
        assert _reorder_axis0_keys(expert), (
            f"router permutes but mlp.{container} experts do not — "
            "the checkpoint would load fine and route to the wrong expert")
    # and the dense always-on MLP must stay put under either scheme
    assert not _reorder_axis0_keys(
        "model.layers.0.mlp.shared_experts.gate_proj.weight")


def test_real_laguna_index_if_present():
    """Against the actual converted repo when it's on this machine — the case
    that was wrong in the field. Skipped in CI."""
    path = os.path.expanduser(
        "~/RandD/laguna-s21-tqTe-g64/model.safetensors.index.json")
    if not os.path.exists(path):
        return
    keys = json.load(open(path))["weight_map"]
    streamed = [k for k in keys if is_streamed_expert_key(k)]
    assert streamed, "no streamable experts found in a real Laguna MoE repo"
    assert not any("shared_experts" in k for k in streamed)
    assert all(k.endswith((".weight", ".scales")) for k in streamed)

    # repack's per-layer invariant, over real key names: no layer may have its
    # router permuted without its experts. Before the fix this held for 0 of 47.
    import collections
    import re

    from turboquant_mlx.stream.repack_experts import _reorder_axis0_keys

    layer_re = re.compile(r"\.layers\.(\d+)\.")
    router, expert = collections.Counter(), collections.Counter()
    for k in keys:
        m = layer_re.search(k)
        if not m or not _reorder_axis0_keys(k):
            continue
        (router if ".mlp.gate." in k else expert)[int(m.group(1))] += 1
    orphaned = sorted(l for l in router if expert[l] == 0)
    assert not orphaned, (
        f"layers {orphaned} would permute the router but not the experts")
