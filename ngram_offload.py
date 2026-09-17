# Copyright 2026 Manjunath Janardhan
"""Serve an n-gram embedding table from disk instead of GPU memory.

Qwen3.8-Flash-Next carries a hashed n-gram table (``NGramEmbedding``) of 128
shards x 2.5M rows. Even at 2 bits it is ~17.9 GiB, about a third of the whole
resident model, yet each token reads only 16 rows (~1 KB). This module leaves the
table in the checkpoint's safetensors files, memory-maps it, and gathers and
dequantizes just the rows a step needs on the CPU. The pages live in the OS file
cache, which macOS can evict, instead of wired GPU memory, which it cannot.

The arithmetic reproduces ``nn.QuantizedEmbedding`` bit for bit. MLX computes
``q * scale + bias`` as a fused multiply-add in float32, then rounds the result to
the scales' dtype. Here the sum is done in float64 (so it is correctly rounded, as
the fused op is) and then rounded the same two steps. On the shipped 2-bit BF16
table the last rounding never changes a value, but 4- and 8-bit or float16 tables
do need it. The tests cover every combination.

Idea borrowed from ddalcu/mlx-serve (MIT), which memory-maps its merged table the
same way; the code here is independent.

Keep the model directory in place while a model is loaded: the maps read from the
original files, and a file removed underneath them makes the next read fail.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_NUMPY_DTYPES = {"U32": "<u4", "F32": "<f4", "F16": "<f2", "BF16": "<u2"}


def _read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _normalize_key(key: str) -> str:
    """Apply the qwen4_exp ``sanitize`` prefix rules, which are the only renames
    an n-gram key can go through."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith("language_model."):
        return key[len("language_model."):]
    return key


def _map(files, key: str, index: dict) -> np.memmap:
    if key not in index:
        raise KeyError(f"{key} is not in any of the {len(files)} safetensors files")
    path, meta, data_start = index[key]
    if meta["dtype"] not in _NUMPY_DTYPES:
        raise NotImplementedError(f"{key}: dtype {meta['dtype']} is not supported")
    start, end = meta["data_offsets"]
    dtype = np.dtype(_NUMPY_DTYPES[meta["dtype"]])
    shape = tuple(meta["shape"])
    if end - start != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
        # A header whose byte range disagrees with its shape would otherwise map
        # the wrong bytes and return plausible-looking garbage.
        raise ValueError(f"{key}: header byte range {end - start} does not match "
                         f"shape {list(shape)} of {meta['dtype']}")
    return np.memmap(path, dtype=dtype, mode="r", offset=data_start + start,
                     shape=shape)


def bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    """Widen raw BF16 bits (as uint16) to float32, exactly.

    BF16 is 1 sign, 8 exponent and 7 mantissa bits, with the same exponent bias as
    float32, so the float32 bit pattern is the same fields with the mantissa
    shifted to the top: the whole conversion is a 16-bit left shift. Zeros,
    subnormals, infinities and NaNs all carry over.
    """
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _widen(a: np.ndarray, dtype: str) -> np.ndarray:
    return bf16_to_float32(a) if dtype == "BF16" else a.astype(np.float32)


def round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 values to the nearest BF16 value (ties to even), as float32.

    BF16 keeps 7 mantissa bits below the leading one, with float32's exponent range.
    Scaling by a power of two is exact, so rounding to a multiple of the value's own
    step does the job. Float32 subnormals go to a signed zero, as MLX's GPU cast
    does (its CPU cast rounds them instead). A dequantized row lands in that range
    only if its scale and bias are themselves near 1e-38, which no quantized table
    has.
    """
    x32 = np.asarray(x, dtype=np.float32)
    x = x32.astype(np.float64)
    _, k = np.frexp(x)
    step = np.ldexp(1.0, k - 8)
    subnormal = (x32.view(np.uint32) & 0x7F800000) == 0
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.round(x / step) * step
        r = np.where(np.isfinite(x), r, x)
        return np.where(subnormal, np.copysign(0.0, x), r).astype(np.float32)


# Bounds the float64 temporaries of a long prefill (256K rows x 160 x 8 bytes would
# otherwise be ~330 MB at once).
_ROW_BLOCK = 16_384


class _Shard:
    """One shard's tensors, memory-mapped, and how to turn rows into float32."""

    def __init__(self, files, prefix: str, index: dict, dim: int,
                 rows: int | None = None):
        self.weight = _map(files, f"{prefix}.weight", index)
        # Check the shape here, at load: a table with the wrong row count would
        # otherwise load cleanly and fail with an IndexError on the first token.
        if self.weight.ndim != 2 or (rows is not None and self.weight.shape[0] != rows):
            raise ValueError(f"{prefix}: weight has shape {list(self.weight.shape)}, "
                             f"expected {rows if rows is not None else 'N'} rows")
        self.scale_key = f"{prefix}.scales"
        if self.scale_key in index:
            self.scales = _map(files, self.scale_key, index)
            self.biases = _map(files, f"{prefix}.biases", index)
            self.scale_dtype = index[self.scale_key][1]["dtype"]
            self.bias_dtype = index[f"{prefix}.biases"][1]["dtype"]
            weight_dtype = index[f"{prefix}.weight"][1]["dtype"]
            if weight_dtype != "U32":
                raise ValueError(f"{prefix}: quantized weight is {weight_dtype}, "
                                 "expected packed U32")
            floats = ("F32", "F16", "BF16")
            if (self.scale_dtype not in floats or self.bias_dtype not in floats
                    or self.scales.ndim != 2 or self.biases.ndim != 2):
                raise ValueError(f"{prefix}: scales and biases must be 2-D floats, got "
                                 f"{self.scale_dtype} {list(self.scales.shape)} and "
                                 f"{self.bias_dtype} {list(self.biases.shape)}")
            groups = self.scales.shape[-1]
            if groups == 0 or dim % groups:
                raise ValueError(f"{prefix}: {groups} groups do not divide width {dim}")
            self.group_size = dim // groups
            packed_bits = self.weight.shape[-1] * 32
            if packed_bits % dim:
                raise ValueError(f"{prefix}: {self.weight.shape[-1]} packed words "
                                 f"do not hold a whole number of bits per value "
                                 f"for width {dim}")
            self.bits = packed_bits // dim
            if self.bits not in (2, 4, 8):
                # 3- and 6-bit values straddle word boundaries; not needed for any
                # shipped table, so refuse rather than guess the bit order.
                raise NotImplementedError(
                    f"{prefix}: {self.bits}-bit tables are not supported off the GPU "
                    "(2, 4 and 8 are)")
            if not (self.scales.shape[0] == self.biases.shape[0]
                    == self.weight.shape[0]) or self.biases.shape != self.scales.shape:
                raise ValueError(f"{prefix}: weight, scales and biases disagree on "
                                 "rows or groups")
            self.shifts = np.arange(0, 32, self.bits, dtype=np.uint32)
            self.mask = np.uint32((1 << self.bits) - 1)
        else:
            self.scales = None
            if self.weight.shape[-1] != dim:
                raise ValueError(f"{prefix}: weight is {self.weight.shape[-1]} wide, "
                                 f"the model's table is {dim}")
            if index[f"{prefix}.weight"][1]["dtype"] == "U32":
                raise ValueError(f"{prefix}: packed U32 weight without scales")
            self.weight_bf16 = index[f"{prefix}.weight"][1]["dtype"] == "BF16"
        self.dim = dim
        self.nbytes = sum(a.nbytes for a in (self.weight, self.scales, getattr(self, "biases", None))
                          if a is not None)

    def rows(self, idx: np.ndarray) -> np.ndarray:
        if self.scales is None:
            w = self.weight[idx]
            return bf16_to_float32(w) if self.weight_bf16 else w.astype(np.float32)
        if len(idx) <= _ROW_BLOCK:
            return self._dequant(idx)
        return np.concatenate([self._dequant(idx[i:i + _ROW_BLOCK])
                               for i in range(0, len(idx), _ROW_BLOCK)])

    def _dequant(self, idx: np.ndarray) -> np.ndarray:
        n = len(idx)
        w = self.weight[idx]
        s = _widen(self.scales[idx], self.scale_dtype).astype(np.float64)
        b = _widen(self.biases[idx], self.bias_dtype).astype(np.float64)
        q = (w[:, :, None] >> self.shifts) & self.mask
        q = q.reshape(n, -1)[:, : self.dim].astype(np.float64).reshape(n, -1, self.group_size)
        out = (q * s[:, :, None] + b[:, :, None]).reshape(n, self.dim).astype(np.float32)
        if self.scale_dtype == "BF16":
            return round_to_bf16(out)
        if self.scale_dtype == "F16":
            return out.astype(np.float16).astype(np.float32)
        return out


class HostShardedEmbedding(nn.Module):
    """Drop-in for ``qwen4_exp._ShardedEmbedding`` that reads from disk.

    Same call: global row ids in, float32 rows out. Holds no MLX arrays, so it adds
    nothing to ``parameters()`` and nothing is wired.
    """

    def __init__(self, shards: list, rows: int, dim: int):
        super().__init__()
        self._shards = shards
        self.n_shards = len(shards)
        self.rows = rows
        self.dim = dim

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._shards)

    def __call__(self, gid: mx.array) -> mx.array:
        flat = np.array(gid, copy=False).reshape(-1).astype(np.int64)
        shard_of = flat // self.rows
        row_of = flat % self.rows
        out = np.empty((flat.size, self.dim), dtype=np.float32)
        for s in np.unique(shard_of).tolist():
            sel = np.nonzero(shard_of == s)[0]
            # Repeated n-grams are common in a prompt: read each distinct row once,
            # in file order, which also keeps the disk access sequential-ish.
            rows, inverse = np.unique(row_of[sel], return_inverse=True)
            out[sel] = self._shards[s].rows(rows)[inverse]
        return mx.array(out).reshape(*gid.shape, self.dim)


def offload_ngram_tables(model, weights: dict, weight_files) -> int:
    """Move every ``NGramEmbedding`` table in ``model`` onto memory-mapped files.

    Swaps each table's ``ngram_embedding`` for a :class:`HostShardedEmbedding` and
    removes the table's tensors from ``weights``, so the loader neither creates GPU
    modules for them nor loads them. Call it after ``sanitize`` and before the
    quantized-layer passes. Returns the bytes moved off the GPU (0 if the model has
    no n-gram table).
    """
    files = [Path(f) for f in weight_files]
    index = {}
    for path in files:
        header, data_start = _read_header(path)
        for key, meta in header.items():
            index[_normalize_key(key)] = (path, meta, data_start)

    moved = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "NGramEmbedding":
            continue
        table = module.ngram_embedding
        prefix = f"{name}.ngram_embedding"
        shards = [_Shard(files, f"{prefix}.shard_{i}", index, table.dim, table.rows)
                  for i in range(table.n_shards)]
        for i in range(table.n_shards):
            for part in ("weight", "scales", "biases"):
                weights.pop(f"{prefix}.shard_{i}.{part}", None)
        host = HostShardedEmbedding(shards, table.rows, table.dim)
        module.ngram_embedding = host
        moved += host.nbytes
    return moved
