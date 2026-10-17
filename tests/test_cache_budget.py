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

import pytest

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


# --------------------------------------------------------------------------
# The working set itself. MLX's CPU backend on Linux reports no Metal working
# set (device_info() is just architecture + device_name), and every `auto`
# default used to index it and die with KeyError.
# --------------------------------------------------------------------------

_GIB = float(1 << 30)
_CPU_DEVICE_INFO = {"architecture": "x86_64", "device_name": "cpu"}


@pytest.fixture(autouse=True)
def _no_host_cgroup(monkeypatch):
    # the host running the tests may itself be a memory-limited container
    from turboquant_mlx import cache_budget

    monkeypatch.setattr(cache_budget, "_PROC_SELF_CGROUP",
                        "/nonexistent/proc/self/cgroup")


def _fake_sysconf(page, pages):
    def sysconf(name):
        return {"SC_PAGE_SIZE": page, "SC_PHYS_PAGES": pages}[name]
    return sysconf


def test_working_set_is_metals_figure_when_there_is_one(monkeypatch):
    import mlx.core as mx

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(mx, "device_info", lambda: {
        **_CPU_DEVICE_INFO, "max_recommended_working_set_size": 12_710_000_000})
    monkeypatch.setattr(cache_budget, "physical_ram_bytes",
                        lambda: pytest.fail("RAM read with Metal present"))
    assert cache_budget.working_set_bytes() == (12_710_000_000, False)


def test_working_set_without_metal_is_the_ram_estimate(monkeypatch):
    import mlx.core as mx

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(mx, "device_info", lambda: dict(_CPU_DEVICE_INFO))
    monkeypatch.setattr(cache_budget, "physical_ram_bytes", lambda: 16 * _GIB)
    wss, estimated = cache_budget.working_set_bytes()
    assert estimated is True
    assert wss == int(cache_budget.estimate_wss(16 * _GIB))


def test_working_set_survives_device_info_raising(monkeypatch):
    import mlx.core as mx

    from turboquant_mlx import cache_budget

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(mx, "device_info", boom)
    monkeypatch.setattr(cache_budget, "physical_ram_bytes", lambda: 16 * _GIB)
    assert cache_budget.working_set_bytes()[1] is True


def test_working_set_is_none_without_metal_or_ram(monkeypatch):
    import mlx.core as mx

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(mx, "device_info", lambda: dict(_CPU_DEVICE_INFO))
    monkeypatch.setattr(cache_budget, "physical_ram_bytes", lambda: None)
    assert cache_budget.working_set_bytes() is None


def test_physical_ram_on_linux_comes_from_sysconf(monkeypatch):
    import subprocess
    import sys

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: pytest.fail("sysctl run on Linux"))
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(4096, 4_194_304))
    assert cache_budget.physical_ram_bytes() == 4096 * 4_194_304


def test_physical_ram_unreadable_is_none(monkeypatch):
    import sys

    from turboquant_mlx import cache_budget

    def no_sysconf(name):
        raise ValueError(name)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "sysconf", no_sysconf)
    assert cache_budget.physical_ram_bytes() is None
    # sysconf answers -1 for a value it does not know
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(4096, -1))
    assert cache_budget.physical_ram_bytes() is None


def test_physical_ram_on_macos_comes_from_sysctl(monkeypatch):
    import subprocess
    import sys

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "check_output",
                        lambda argv: b"17179869184\n"
                        if argv == ["sysctl", "-n", "hw.memsize"] else None)
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(1, 1))
    assert cache_budget.physical_ram_bytes() == 17179869184


def _cgroups(monkeypatch, tmp_path, proc_self_cgroup, files):
    """A fake cgroup mount: ``files`` maps paths under it to limits."""
    from turboquant_mlx import cache_budget

    root = tmp_path / "cgroup"
    for rel, value in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(value + "\n")
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(proc_self_cgroup)
    monkeypatch.setattr(cache_budget, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(cache_budget, "_PROC_SELF_CGROUP", str(proc))


def _limit_files(monkeypatch, tmp_path, v2=None, v1=None):
    # a container with its own cgroup namespace: its cgroup is the mount root
    files = {}
    if v2 is not None:
        files["memory.max"] = v2
    if v1 is not None:
        files["memory/memory.limit_in_bytes"] = v1
    _cgroups(monkeypatch, tmp_path, "0::/\n" if v1 is None else
             "7:memory:/\n", files)


@pytest.fixture
def linux_16gib(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(4096, 4_194_304))
    return 4096 * 4_194_304


def test_physical_ram_in_a_container_is_its_cgroup_v2_limit(
        monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    _limit_files(monkeypatch, tmp_path, str(int(4 * _GIB)), None)
    assert cache_budget.physical_ram_bytes() == 4 * _GIB


def test_physical_ram_reads_cgroup_v1_when_there_is_no_v2(
        monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    _limit_files(monkeypatch, tmp_path, None, str(int(2 * _GIB)))
    assert cache_budget.physical_ram_bytes() == 2 * _GIB


@pytest.mark.parametrize("unlimited", ["max", str(2**63 - 4096)])
def test_physical_ram_without_a_cgroup_limit_is_the_hosts(
        monkeypatch, tmp_path, linux_16gib, unlimited):
    from turboquant_mlx import cache_budget

    # v2 says "max"; v1 says a page-rounded 2**63
    _limit_files(monkeypatch, tmp_path, unlimited)
    assert cache_budget.physical_ram_bytes() == linux_16gib


def test_cgroup_limit_stands_in_when_sysconf_fails(monkeypatch, tmp_path):
    import sys

    from turboquant_mlx import cache_budget

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "sysconf", _fake_sysconf(4096, -1))
    _limit_files(monkeypatch, tmp_path, str(int(3 * _GIB)))
    assert cache_budget.physical_ram_bytes() == 3 * _GIB


def test_working_set_in_a_container_is_estimated_from_its_limit(
        monkeypatch, tmp_path, linux_16gib):
    import mlx.core as mx

    from turboquant_mlx import cache_budget
    from turboquant_mlx.cache_budget import estimate_wss

    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "cpu"})
    _limit_files(monkeypatch, tmp_path, str(int(4 * _GIB)))
    assert cache_budget.working_set_bytes() == (int(estimate_wss(4 * _GIB)),
                                                True)


def test_physical_ram_in_a_systemd_service_is_its_cgroup_limit(
        monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    # MemoryMax= on the service; the slice above it and the root are unlimited
    _cgroups(monkeypatch, tmp_path, "0::/system.slice/tq.service\n", {
        "system.slice/tq.service/memory.max": str(int(6 * _GIB)),
        "system.slice/memory.max": "max",
    })
    assert cache_budget.physical_ram_bytes() == 6 * _GIB


def test_an_ancestors_tighter_limit_wins(monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    _cgroups(monkeypatch, tmp_path, "0::/user.slice/user-501.slice/app\n", {
        "user.slice/user-501.slice/app/memory.max": str(int(8 * _GIB)),
        "user.slice/memory.max": str(int(5 * _GIB)),
    })
    assert cache_budget.physical_ram_bytes() == 5 * _GIB


def test_v1_host_path_missing_in_the_container_falls_back_to_the_root(
        monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    # without a cgroup namespace the container sees the host's path, but its
    # own cgroup is mounted at the root
    _cgroups(monkeypatch, tmp_path, "12:cpu,cpuacct:/docker/abc\n"
             "7:memory:/docker/abc\n", {
                 "memory/memory.limit_in_bytes": str(int(2 * _GIB)),
             })
    assert cache_budget.physical_ram_bytes() == 2 * _GIB



def test_a_malformed_proc_self_cgroup_line_is_skipped(
        monkeypatch, tmp_path, linux_16gib):
    from turboquant_mlx import cache_budget

    _cgroups(monkeypatch, tmp_path, "garbage\n0::/\n",
             {"memory.max": str(int(4 * _GIB))})
    assert cache_budget.physical_ram_bytes() == 4 * _GIB
