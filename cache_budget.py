"""How big the streaming expert cache should be — in one place.

``--cache-budget-gb auto`` is answered twice: once by ``stream/loader.py``,
which actually sizes the cache, and once by ``plan.py``, which tells you what
the loader is going to pick before you download 28 GB to find out. They
disagreed. On a 16 GB M4 mini planning Laguna-S-2.1, ``turboquant-plan``
advertised "~9.1 GB here" while the loader chose 5.89 GB — because plan sized
from the cap you *could* raise the wired limit to and subtracted a 1 GB
reserve, and the loader sized from the cap you actually have and subtracted 2.
A planner that contradicts the thing it predicts is worse than no planner, so
the arithmetic now lives here and both import it.

The constants are measured, not guessed. Sweeping ``--cache-budget-gb`` over
2 / 4 / 5.89 / 6 / 8 GB on a 16 GB M4 mini (Metal working set 12.71 GB,
Laguna-S-2.1 ternary, 2.28 GB resident) put mlx's peak at **budget + 3.33 GB**
every single time, within 40 MB:

    budget  2.00  4.00  5.89  6.00  8.00
    peak    5.31  7.32  9.23  9.34 11.35

2.28 GB of that constant is the resident weights, which the caller already
knows, leaving ~1.05 GB of runtime overhead that tracks neither the cache nor
the model. That is what ``RUNTIME_OVERHEAD_BYTES`` is, and it is why a budget
can be derived rather than guessed at with a blanket fraction.

The old formula took 80% of the working set *and then* subtracted a 2 GB
reserve, which double-counts safety: on that mini it reserved 4.5 GB against a
measured need of ~1.05 GB plus KV, and cost ~15% of achievable throughput
(5.89 GB → 1.36 tok/s where 8 GB → 1.58 tok/s). The reserve here is explicit
about what each gigabyte is for.

The working set itself is answered here too (``working_set_bytes``), for the
same reason: plan, the streaming loader and the server's ``auto`` defaults all
need it, and MLX's CPU backend on Linux has no Metal working set to report.
"""

import os
import subprocess
import sys

# Runtime cost that scales with neither the cache nor the resident weights:
# activations, buffer-cache slack, fragmentation. Measured at ~1.05 GB; carried
# at 1.1 to avoid pricing the floor exactly at the observation.
RUNTIME_OVERHEAD_BYTES = int(1.1e9)

# KV cache plus margin. KV grows with context and the loader does not know the
# context in advance, so this is deliberately generous: at 16K tokens Laguna's
# hybrid attention wants 0.88 GB, and a model with less sliding-window
# attention wants more. Callers that *do* know their KV size should pass it as
# `kv_bytes` and this shrinks to pure margin.
KV_RESERVE_BYTES = int(1.9e9)

# Below this a cache is not worth having: every token would miss.
MIN_BUDGET_BYTES = int(0.5e9)


def auto_cache_budget(wss_bytes: int, resident_bytes: int, expert_bytes: int,
                      kv_bytes: int = 0) -> int:
    """Bytes of expert cache to use when the user says ``auto``.

    ``wss_bytes`` is Metal's max recommended working set — the *current* one,
    not what a ``sysctl iogpu.wired_limit_mb`` bump could make it. Sizing a
    default against a limit the user has not raised is how plan.py came to
    recommend a budget that would have peaked at 12.4 GB against a 12.71 GB
    cap.

    Clamped to ``[MIN_BUDGET_BYTES, expert_bytes]``. The upper clamp must win:
    when the experts are smaller than the floor, a budget bigger than every
    expert that exists is just wasted reservation.
    """
    reserve = RUNTIME_OVERHEAD_BYTES + (kv_bytes or KV_RESERVE_BYTES)
    budget = int(wss_bytes) - int(resident_bytes) - reserve
    return min(int(expert_bytes), max(MIN_BUDGET_BYTES, budget))


def projected_peak_bytes(budget_bytes: int, resident_bytes: int,
                         kv_bytes: int = 0) -> int:
    """What a streaming run will actually peak at — the inverse of the above.

    Exposed because it is the number a user can check against their own
    machine: measured peak was within 40 MB of this across a 4x range of
    budgets. `turboquant-plan` prints it; the streaming loader could assert on
    it.
    """
    return int(budget_bytes) + int(resident_bytes) + RUNTIME_OVERHEAD_BYTES \
        + int(kv_bytes)


def estimate_wss(ram_bytes: float | None) -> float | None:
    """Estimate Metal's default working-set cap from RAM.

    Apple's `recommendedMaxWorkingSetSize` is roughly two-thirds of RAM on small
    machines and a larger fraction on big ones — measured here: a 16 GB mini
    reports 10.5 GB (66%), a 68.7 GB M4 Max reports 55.7 GB (81%). A single
    fraction can't serve both, so this is piecewise and deliberately biased LOW:
    under-promising the cap costs a user an unnecessary flag, over-promising
    costs them an OOM.
    """
    if not ram_bytes:
        return None
    frac = 0.66 if ram_bytes <= 36e9 else 0.75
    return frac * ram_bytes


# sysconf reports the host's RAM inside a container or a memory-limited
# systemd service, and sizing against that gets the process OOM-killed. The
# limit that applies is the tightest one on this process's cgroup and its
# ancestors, which /proc/self/cgroup locates under the cgroup mount.
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_ROOT = "/sys/fs/cgroup"


def _cgroup_memory_limit() -> int | None:
    """The tightest cgroup memory limit on this process, in bytes, or None
    when there is none (v2 says ``max``; v1 says a huge number, which the
    caller's min() drops) or this isn't Linux.

    A path from /proc/self/cgroup that doesn't exist under the mount (a v1
    container without its own cgroup namespace sees the host's path) walks up
    to the mount root, which is then that container's own cgroup.
    """
    try:
        with open(_PROC_SELF_CGROUP) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, cgroup = parts
        if controllers == "":  # v2: "0::/path"
            base, name = _CGROUP_ROOT, "memory.max"
        elif "memory" in controllers.split(","):  # v1: "N:memory:/path"
            base, name = os.path.join(_CGROUP_ROOT, "memory"), \
                "memory.limit_in_bytes"
        else:
            continue
        rel = cgroup.strip("/")
        while True:
            try:
                with open(os.path.join(base, rel, name)) as f:
                    value = f.read().strip()
            except OSError:
                value = ""
            if value.isdigit() and int(value) > 0:
                limits.append(int(value))
            if not rel:
                break
            rel = os.path.dirname(rel)
    return min(limits) if limits else None


def physical_ram_bytes() -> int | None:
    """RAM this process can use, in bytes, or None when it can't be read.

    macOS reports it via `sysctl hw.memsize`; elsewhere (Linux, where MLX runs
    on its CPU backend) `sysconf` does, capped at a container's cgroup limit.
    """
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            pass
    try:  # AttributeError: os.sysconf is absent on some platforms (Windows)
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 0
    limit = _cgroup_memory_limit()
    if limit and (ram <= 0 or limit < ram):
        ram = limit
    return ram if ram > 0 else None


def working_set_bytes() -> tuple[int, bool] | None:
    """``(bytes, estimated)`` for the memory a run may plan against, or None.

    Metal's `max_recommended_working_set_size` when the device reports one
    (``estimated`` False). Without a Metal device — MLX's CPU backend on Linux,
    whose `device_info()` has no such key — the cap is estimated from RAM with
    `estimate_wss` (``estimated`` True). None only when RAM can't be read
    either; callers decide what ``auto`` means then.

    mlx is imported here, not at module level, so plan.py stays importable
    without loading a framework.
    """
    try:
        import mlx.core as mx
        wss = mx.device_info().get("max_recommended_working_set_size")
    except Exception:
        wss = None
    if wss:
        return int(wss), False
    est = estimate_wss(physical_ram_bytes())
    return (int(est), True) if est else None
