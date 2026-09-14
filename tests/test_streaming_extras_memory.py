"""The streaming affine-extras pass must release each source weight as it goes.

Qwen3.8-Flash-Next's n-gram/PLE table is 128 nn.Embedding shards, 95.4 GiB in
bf16. The streaming pass quantizes one shard at a time precisely so that table is
never resident. An earlier loop iterated ``list(model.named_modules())``, which
held a reference to every ORIGINAL module for the whole loop -- so each source
weight, materialized by ``to_quantized``, stayed alive after its module was
swapped for nn.Identity. The whole bf16 table accumulated, the machine went deep
into swap, and three real conversions died 71-90% of the way through this phase.

Embedding weights here are left lazy (never evaluated before the pass), exactly
like a lazily loaded checkpoint, so retention shows up as active-memory growth.
"""

import gc

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.quantize_model import quantize_affine_extras

_K, _ROWS, _DIMS = 12, 50_000, 160  # 160 = the real shard width; 32 MB each in float32


class _Shards(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(_K):
            setattr(self, f"shard_{i}", nn.Embedding(_ROWS, _DIMS))


def test_streaming_extras_does_not_accumulate_source_weights():
    gc.collect()
    mx.clear_cache()
    model = _Shards()  # weights stay lazy: nothing evaluated yet
    per_module = _ROWS * _DIMS * 4
    active_after_each = []

    def sink(path, module):
        # Discard the quantized module, like a writer that has flushed it.
        active_after_each.append(mx.get_active_memory())

    n = quantize_affine_extras(model, {}, bits=4, group_size=32, on_quantized=sink)
    assert n == _K

    # The sink runs before the loop drops its references, so allow a couple of
    # modules' worth of slack. Retaining every source weight grows by ~K modules.
    growth = active_after_each[-1] - active_after_each[0]
    assert growth < 3 * per_module, (
        f"active memory grew {growth / 2**20:.0f} MiB across {_K} modules "
        f"({per_module / 2**20:.0f} MiB each) -- source weights are being retained")
