"""Unit test for the streaming shard writer (no model / disk-of-a-model needed).

The end-to-end guarantee — that ``--streaming`` conversion is byte-identical to
the in-memory converter — is validated manually on real models (it depends on a
fixed PYTHONHASHSEED, since the per-layer rotation seed uses ``hash()``). Here we
just exercise the shard writer's sharding, naming, index, and reload-fidelity.
"""

import glob
import json
import tempfile
from pathlib import Path

import mlx.core as mx

from turboquant_mlx.convert_streaming import StreamingShardWriter


def test_streaming_shard_writer_multishard():
    tensors = {
        "model.a.weight": mx.arange(16, dtype=mx.uint32).reshape(4, 4),
        "model.a.scales": (mx.ones((4, 1)) * 2).astype(mx.float16),
        "model.a.codebook": mx.arange(8, dtype=mx.float32),
        "model.b.weight": mx.arange(8, dtype=mx.uint32),
    }
    with tempfile.TemporaryDirectory() as d:
        # tiny threshold -> one tensor per shard, exercising the -of-N naming
        w = StreamingShardWriter(d, max_file_size_gb=1e-9)
        for k, v in tensors.items():
            w.add(k, v)
        n = w.finalize()
        assert n >= 2, f"expected multiple shards, got {n}"

        idx = json.load(open(Path(d) / "model.safetensors.index.json"))
        assert set(idx["weight_map"]) == set(tensors)
        assert idx["metadata"]["total_size"] == sum(v.nbytes for v in tensors.values())
        # every referenced shard file actually exists
        for fname in set(idx["weight_map"].values()):
            assert (Path(d) / fname).exists(), fname

        # reloading the shards reproduces every tensor exactly
        loaded = {}
        for f in glob.glob(str(Path(d) / "*.safetensors")):
            loaded.update(mx.load(f))
        assert set(loaded) == set(tensors)
        for k, v in tensors.items():
            assert mx.array_equal(loaded[k], v).item(), k

    print("test_streaming_shard_writer_multishard: PASSED")
