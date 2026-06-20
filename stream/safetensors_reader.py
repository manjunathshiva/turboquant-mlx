"""Read individual MoE expert slices out of mmap'd safetensors shards.

TurboQuant stores each expert projection as a stacked tensor

    weight: (num_experts, output_dims, packed_cols)  uint32
    scales: (num_experts, output_dims, n_groups)      float16

contiguous along the expert axis, so expert ``e`` is a single contiguous byte
range. This reader builds an offset index across all shards and hands back
just the bytes for one expert as an ``mx.array`` — never materializing the
full (256, ...) tensor. That is the primitive the streaming MoE block uses to
keep resident memory bounded.

Only the dtypes that actually get streamed are supported (U32 weights, F16
scales); everything else in the model is loaded resident the normal way.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import struct
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

# macOS: tell the OS not to keep this fd's data in the unified buffer cache,
# so streaming 14+ GB of expert slices doesn't balloon resident page cache on
# a memory-constrained machine. Advisory; pairs with our own LRU expert cache.
_F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)

# safetensors dtype string -> (numpy dtype, itemsize, mlx dtype)
_DTYPES = {
    "U32": (np.uint32, 4, mx.uint32),
    "F16": (np.float16, 2, mx.float16),
    "F32": (np.float32, 4, mx.float32),
}


@dataclass
class _TensorLoc:
    file_idx: int          # index into self._mmaps
    dtype: str             # safetensors dtype string
    shape: tuple           # full tensor shape
    abs_begin: int         # absolute byte offset of element 0 in the file


class SafetensorsExpertReader:
    """Offset index + mmap pool for reading per-expert weight/scale slices.

    Parameters
    ----------
    model_path : str
        Directory holding ``model*.safetensors``.
    use_page_cache : bool
        If False (default), set ``F_NOCACHE`` so expert reads bypass the OS
        unified buffer cache — correct when the model is much larger than RAM
        (16 GB mini streaming a big MoE), where letting the page cache grow
        would thrash. If True ("trust the OS", Flash-MoE's finding), leave the
        page cache enabled so LRU-eviction re-reads come back from warm RAM
        instead of disk — a win on roomy machines where the model file fits in
        free RAM. See ``trust_os_ab.py`` for the measured tradeoff.
    """

    def __init__(self, model_path: str, use_page_cache: bool = False):
        self.model_path = model_path
        self.use_page_cache = use_page_cache
        # glob.escape so a model_path containing [ ] etc. still matches literally.
        files = sorted(glob.glob(os.path.join(glob.escape(model_path), "model*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No model*.safetensors in {model_path}")
        self._files = files
        self._fds: list[int] = []
        self._index: dict[str, _TensorLoc] = {}

        for fi, f in enumerate(files):
            fd = os.open(f, os.O_RDONLY)
            if not use_page_cache:
                try:
                    fcntl.fcntl(fd, _F_NOCACHE, 1)
                except OSError:
                    pass  # F_NOCACHE is best-effort
            self._fds.append(fd)
            header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_len, 8))
            data_start = 8 + header_len  # tensor data section begins here
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, _end = meta["data_offsets"]
                self._index[name] = _TensorLoc(
                    file_idx=fi,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    abs_begin=data_start + begin,
                )

    # -- introspection -------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self._index

    def shape(self, key: str) -> tuple:
        return self._index[key].shape

    def num_experts(self, key: str) -> int:
        return self._index[key].shape[0]

    # -- the hot path --------------------------------------------------
    def read_expert_np(self, key: str, expert: int):
        """Return ``(numpy_array, mlx_dtype)`` for slice ``[expert]``.

        This is the disk half of :meth:`read_expert`: a positional ``os.pread``
        (which releases the GIL and ignores the file offset, so it is safe to
        call concurrently on the same fd) plus a zero-copy ``np.frombuffer``
        view. The MLX array — which we keep on a single thread — is built by
        the caller. Splitting it this way lets the expert cache fan the
        per-layer ``pread``s across a thread pool without ever touching MLX
        from a worker thread.
        """
        loc = self._index[key]
        np_dt, itemsize, mlx_dt = _DTYPES[loc.dtype]
        per_expert = 1
        for d in loc.shape[1:]:
            per_expert *= d
        nbytes = per_expert * itemsize
        start = loc.abs_begin + expert * nbytes
        # pread on an F_NOCACHE fd: copies just this expert's bytes, without
        # growing the resident page cache.
        raw = os.pread(self._fds[loc.file_idx], nbytes, start)
        buf = np.frombuffer(raw, dtype=np_dt, count=per_expert).reshape(loc.shape[1:])
        return buf, mlx_dt

    def read_range_np(self, key: str, e_start: int, count: int):
        """Read ``count`` *consecutive* experts [e_start, e_start+count) in ONE
        positional ``pread`` and return ``(numpy_array, mlx_dtype)`` of shape
        ``(count, d0, d1, ...)``. Coalescing adjacent experts into a single
        larger read is the win behind the co-activation on-disk layout (#3):
        the streaming cache calls this once per contiguous run instead of once
        per expert, cutting syscalls and turning random reads into sequential.
        """
        loc = self._index[key]
        np_dt, itemsize, mlx_dt = _DTYPES[loc.dtype]
        per_expert = 1
        for d in loc.shape[1:]:
            per_expert *= d
        each = per_expert * itemsize
        start = loc.abs_begin + e_start * each
        raw = os.pread(self._fds[loc.file_idx], each * count, start)
        buf = np.frombuffer(raw, dtype=np_dt, count=per_expert * count).reshape(
            (count, *loc.shape[1:]))
        return buf, mlx_dt

    def read_expert(self, key: str, expert: int) -> mx.array:
        """Return slice ``[expert]`` of a stacked tensor as an mx.array.

        For a tensor of shape (E, d0, d1, ...) this returns shape (d0, d1, ...)
        and copies only that expert's bytes into an MLX buffer.
        """
        buf, mlx_dt = self.read_expert_np(key, expert)
        return mx.array(buf, dtype=mlx_dt)

    def close(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self._fds = []

    def __del__(self):
        self.close()
