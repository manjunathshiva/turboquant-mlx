"""Memory-bounded (sharded-streaming) TurboQuant conversion.

The standard converter (:mod:`turboquant_mlx.convert`) materializes the entire
quantized model in RAM before saving, which caps practical conversion at
~130B params on a 64 GB Mac (the quantized model itself must fit in memory).

This path writes each quantized layer to a safetensors shard and frees it
*during* the quantization loop, so peak memory stays at roughly one shard
(~5 GB) plus the single layer being processed. That lets 200B+ MoEs convert on
a 64 GB machine. The output is identical to ``convert.py`` (same quantization,
same seeds) — only *when* tensors are written to disk differs.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import (
    MAX_FILE_SIZE_GB,
    create_model_card,
    hf_repo_to_path,
    load,
    save_config,
)

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from turboquant_mlx.config import TurboQuantConfig
from turboquant_mlx.quantize_model import turboquant_quantize


class StreamingShardWriter:
    """Write tensors into safetensors shards incrementally, mlx-lm style.

    Tensors are buffered until the buffer would exceed ``max_file_size_gb``,
    then flushed to a shard file and freed. The final ``model.safetensors.index.json``
    is written by :meth:`finalize` once the total shard count is known.
    """

    def __init__(self, dst_path, max_file_size_gb: int = MAX_FILE_SIZE_GB):
        self.dst = Path(dst_path)
        self.dst.mkdir(parents=True, exist_ok=True)
        # int() so tests can pass a fractional GB to force small shards
        self.max_bytes = int(max_file_size_gb * (2 ** 30))
        self._buf: dict = {}
        self._buf_bytes = 0
        self._shard_idx = 0
        self._tmp_files: list[Path] = []
        self._name_to_shard: dict[str, int] = {}
        self.total_size = 0
        self.total_params = 0

    def add(self, name: str, arr) -> None:
        mx.eval(arr)
        nb = arr.nbytes
        # Mirror mlx_lm.make_shards: start a new shard before adding a tensor
        # that would overflow the current one.
        if self._buf and self._buf_bytes + nb > self.max_bytes:
            self._flush()
        self._buf[name] = arr
        self._buf_bytes += nb
        self._name_to_shard[name] = self._shard_idx
        self.total_size += nb
        self.total_params += arr.size

    def _flush(self) -> None:
        tmp = self.dst / f"__tq_shard_{self._shard_idx:05d}.safetensors"
        mx.save_safetensors(str(tmp), self._buf, metadata={"format": "mlx"})
        self._tmp_files.append(tmp)
        self._buf = {}
        self._buf_bytes = 0
        self._shard_idx += 1

    def finalize(self) -> int:
        if self._buf:
            self._flush()
        n = len(self._tmp_files)
        if n == 0:
            raise RuntimeError("StreamingShardWriter: no tensors were written")
        single = n == 1
        idx_to_name: dict[int, str] = {}
        for i, tmp in enumerate(self._tmp_files):
            final = "model.safetensors" if single else \
                f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.replace(tmp, self.dst / final)
            idx_to_name[i] = final
        weight_map = {k: idx_to_name[self._name_to_shard[k]]
                      for k in sorted(self._name_to_shard)}
        index = {
            "metadata": {
                "total_size": self.total_size,
                "total_parameters": self.total_params,
            },
            "weight_map": weight_map,
        }
        with open(self.dst / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=4)
        return n


def convert_streaming(
    hf_path: str,
    mlx_path: str = "mlx_model",
    bits: int = 3,
    group_size: int = 64,
    rotation: str = "hadamard",
    rotation_seed: int = 42,
    fuse_rotations: bool = True,
    use_qjl: bool = False,
    attn_bits: int = None,
    mlp_bits: int = None,
    mlp_group_size: int = None,
    ternary_experts: bool = False,
    max_file_size_gb: int = MAX_FILE_SIZE_GB,
):
    """Convert an HF model to TurboQuant MLX format with bounded peak memory.

    Identical output to :func:`turboquant_mlx.convert.convert`, but each
    quantized layer is streamed to disk and freed, so the full quantized model
    is never resident. Use this for models whose quantized size exceeds RAM.
    """
    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to {mlx_path} as it already exists. "
            "Delete it or specify a new path."
        )

    tq_config = TurboQuantConfig(
        bits=bits, group_size=group_size, rotation=rotation,
        rotation_seed=rotation_seed, fuse_rotations=fuse_rotations,
        use_qjl=use_qjl, attn_bits=attn_bits, mlp_bits=mlp_bits,
        mlp_group_size=mlp_group_size, ternary_experts=ternary_experts,
    )

    print(f"[INFO] Loading model from {hf_path} (lazy)")
    model, tokenizer, config = load(hf_path, return_config=True, lazy=True)
    arch = config.get("model_type", "unknown")
    print(f"[INFO] Streaming TurboQuant convert: {bits}-bit, gs={group_size}, "
          f"arch={arch}, shard={max_file_size_gb} GB")

    writer = StreamingShardWriter(mlx_path, max_file_size_gb)

    def on_quantized(path, module):
        # Write every parameter of this just-quantized layer, then it gets freed.
        for sub, arr in tree_flatten(module.parameters()):
            writer.add(f"{path}.{sub}", arr)

    t0 = time.time()
    # Pass 1: quantized layers are streamed + freed via the callback.
    model, config = turboquant_quantize(
        model, config, tq_config, on_quantized=on_quantized,
    )
    # Pass 2: the remaining (non-quantized) params — norms (with fused rotations),
    # embeddings, routers, any dimension-skipped layers. Small vs. the experts.
    for name, arr in tree_flatten(model.parameters()):
        writer.add(name, arr)
    n_shards = writer.finalize()
    print(f"[INFO] Quantized + streamed in {time.time() - t0:.1f}s "
          f"({writer.total_size / 1024**3:.2f} GB across {n_shards} shard(s))")

    # Config + tokenizer + aux files — mirror the tail of mlx_lm.utils.save.
    save_config(config, config_path=mlx_path / "config.json")
    tokenizer.save_pretrained(mlx_path)

    src_path = Path(hf_path)
    hf_repo = None
    if not src_path.exists():
        hf_repo = hf_path
        src_path = hf_repo_to_path(hf_path)
    for pattern in ("*.py", "generation_config.json"):
        for fpath in glob.glob(str(src_path / pattern)):
            shutil.copy(fpath, mlx_path)
    create_model_card(mlx_path, hf_repo)

    print(f"[INFO] Done! Model saved to {mlx_path}")
