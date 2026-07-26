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
"""

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
