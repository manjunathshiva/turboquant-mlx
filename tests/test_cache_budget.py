"""`--cache-budget-gb auto` means one number, not two.

`turboquant-plan` exists to tell you what the streaming loader will do before
you download the model. It was telling you something else: on a 16 GB M4 mini
planning Laguna-S-2.1 it advertised "~9.1 GB here" while the loader chose
5.89 GB, because the two computed the budget from different ceilings with
different reserves. The measured peak at 9.1 GB would have been ~12.4 GB
against that machine's 12.71 GB cap.

The sweep the constants come from (16 GB M4 mini, Metal working set 12.71 GB,
Laguna-S-2.1 ternary, 2.28 GB resident, 26.42 GB of experts):

    budget  2.00  4.00  5.89  6.00  8.00   GB
    peak    5.31  7.32  9.23  9.34 11.35   GB
    tok/s   1.15  1.26  1.36  1.35  1.58
"""

import os

from turboquant_mlx.cache_budget import (
    MIN_BUDGET_BYTES,
    auto_cache_budget,
    projected_peak_bytes,
)

_GB = 1e9

# The machine the constants were measured on.
_MINI_WSS = 12.71 * _GB
_MINI_RESIDENT = 2.28 * _GB
_MINI_EXPERTS = 26.42 * _GB

# (budget GB, measured mlx peak GB) — every point of the real sweep.
_SWEEP = [(2.00, 5.31), (4.00, 7.32), (5.89, 9.23), (6.00, 9.34), (8.00, 11.35)]


def test_projected_peak_matches_every_measured_point():
    """The peak model is the whole basis for deriving a budget. If it drifts
    from what the hardware does, the budget silently stops being safe."""
    for budget_gb, measured_gb in _SWEEP:
        predicted = projected_peak_bytes(budget_gb * _GB, _MINI_RESIDENT) / _GB
        assert abs(predicted - measured_gb) < 0.2, (
            f"budget {budget_gb} GB: predicted peak {predicted:.2f} GB vs "
            f"measured {measured_gb:.2f} GB")


def test_auto_budget_fits_under_the_cap():
    """The point of the reserve: whatever auto picks must leave room."""
    budget = auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, _MINI_EXPERTS)
    peak = projected_peak_bytes(budget, _MINI_RESIDENT)
    assert peak < _MINI_WSS, "auto budget projects a peak over the Metal cap"
    assert _MINI_WSS - peak > 1.0 * _GB, "less than 1 GB of headroom is not a reserve"


def test_auto_budget_is_not_wastefully_small():
    """The regression this replaced. The old formula took 80% of the working
    set *and* subtracted a 2 GB reserve — double-counted safety that cost ~15%
    of throughput on the mini (5.89 GB -> 1.36 tok/s where 8 GB -> 1.58)."""
    budget = auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, _MINI_EXPERTS)
    old = int(0.8 * _MINI_WSS) - _MINI_RESIDENT - 2.0 * _GB   # 5.89 GB
    assert budget > old, "auto is no better than the formula it replaced"
    assert budget / _GB > 7.0, f"auto picked {budget / _GB:.2f} GB, still timid"


def test_never_exceeds_the_experts_that_exist():
    """Upper clamp must beat the floor: caching more than every expert is
    reserved memory that can never be used."""
    tiny = 0.2 * _GB
    assert auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, tiny) == int(tiny)


def test_floor_applies_on_a_machine_with_no_room():
    """A cramped machine still gets a usable (if bad) budget rather than a
    negative one."""
    budget = auto_cache_budget(4 * _GB, 3.5 * _GB, _MINI_EXPERTS)
    assert budget == MIN_BUDGET_BYTES


def test_monotonic_in_working_set():
    """More GPU, more cache — never the reverse."""
    budgets = [auto_cache_budget(w * _GB, _MINI_RESIDENT, _MINI_EXPERTS)
               for w in (12.71, 16, 24, 32)]
    assert budgets == sorted(budgets)


def test_known_kv_size_shrinks_the_reserve():
    """A caller that knows its KV cache shouldn't pay the blanket estimate."""
    blind = auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, _MINI_EXPERTS)
    known = auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, _MINI_EXPERTS,
                              kv_bytes=int(0.88 * _GB))  # laguna @16K
    assert known > blind


def test_plan_and_loader_pick_the_same_budget():
    """The drift guard. `plan.py` promises what `stream/loader.py` does; the
    two must be the same arithmetic, not two copies of it."""
    from turboquant_mlx.stream.loader import _auto_cache_budget

    model_bytes = _MINI_RESIDENT + _MINI_EXPERTS
    loader = _auto_cache_budget(model_bytes, _MINI_EXPERTS, _MINI_WSS)
    plan = auto_cache_budget(_MINI_WSS, _MINI_RESIDENT, _MINI_EXPERTS)
    assert loader == plan, (
        f"loader would use {loader / _GB:.2f} GB but plan advertises "
        f"{plan / _GB:.2f} GB")


def test_plan_advertises_it_in_the_flag_a_user_reads(tmp_path):
    """End-to-end over the rendered flag string. The drift guard above proves
    the two functions agree; this proves plan actually *prints* that result
    rather than a stale expression alongside it — which is exactly the shape
    the original bug had."""
    import json
    import struct

    from turboquant_mlx.plan import build_plan

    # A 30 GB MoE in 300 bytes. `turboquant-plan` reads only the safetensors
    # *header* — that is its whole reason to exist — so a hand-written header
    # declaring large shapes with no tensor data behind it exercises the real
    # code path. Reaching the streaming branch needs a model genuinely bigger
    # than the machine, which is not something you can allocate in a test.
    hdr, off = {}, 0
    for name, shape in (
            ("model.layers.0.mlp.switch_mlp.gate_proj.weight",
             (256, 4096, 7168)),                       # ~30 GB of experts
            ("model.layers.0.self_attn.q_proj.weight", (4096, 4096))):
        n = 1
        for s in shape:
            n *= s
        hdr[name] = {"dtype": "U32", "shape": list(shape),
                     "data_offsets": [off, off + n * 4]}
        off += n * 4
    blob = json.dumps(hdr).encode()
    d = tmp_path / "moe"
    d.mkdir()
    with open(d / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
    (d / "config.json").write_text(
        '{"model_type": "qwen3_next", "num_hidden_layers": 4, '
        '"hidden_size": 4096, "num_attention_heads": 32, '
        '"num_key_value_heads": 4}')

    # the machine the constants were measured on
    plan = build_plan(str(d), wired_gb=_MINI_WSS / _GB, ram_gb=17.18)
    assert plan["verdict"]["mode"] == "streaming", (
        "a 30 GB model on a 12.71 GB machine must plan as streaming; this test "
        "checks nothing otherwise")

    flag = next(f for f in plan["flags"] if "--cache-budget-gb" in f)
    expected = auto_cache_budget(plan["machine"]["wss_bytes"],
                                 plan["model"]["resident_bytes"],
                                 plan["model"]["expert_bytes"])
    assert f"{expected / _GB:.1f} GB" in flag, (
        f"plan prints {flag!r}, loader would use {expected / _GB:.2f} GB")


def test_local_path_typo_says_so(tmp_path):
    """A path that isn't there is a typo, not a repo id.

    `_resolve` used to hand anything non-directory to the hub, so
    `turboquant-plan --model ~/typo` reported "Repo id must be in the form
    'repo_name' or 'namespace/repo_name'" — which sends you hunting for a
    naming rule when a directory is simply missing.
    """
    import pytest

    from turboquant_mlx.plan import _resolve

    with pytest.raises(FileNotFoundError, match="no such directory"):
        _resolve(str(tmp_path / "not-there"))

    # a real repo id must still be treated as remote, not as a broken path
    target, remote = _resolve("manjunathshiva/Laguna-S-2.1-tqTe-g64")
    assert remote or os.path.isdir(target)
