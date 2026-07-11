"""Disk-persistent prompt cache for turboquant-serve.

Makes the server's prompt-prefix cache a *disk* citizen, not just an
in-memory one: every completed request's KV cache is checkpointed to a
cache directory, and on a cache miss (most importantly: the first request
after a server restart) the best on-disk checkpoint is loaded back and
reused, so only the token suffix is prefilled. This turns the killer cold
start on slow boxes (a ~13-minute 22K-token prefill on a streaming 122B on
a 16 GB mini) into a file load plus a short suffix prefill.

Design (inspired by DwarfStar's "the KV cache is a first-class disk
citizen", adapted to mlx_lm's token-level prompt trie):

- ``mlx_lm.server`` routes all prompt-cache traffic through
  ``LRUPromptCache.fetch_nearest_cache`` / ``insert_cache`` with the full
  token list. We wrap both (same pattern as ``prefill_stats``): inserts are
  mirrored to disk, and a fetch that reuses little consults the disk index
  for a checkpoint with a longer usable prefix, loads it into the in-memory
  LRU, and re-runs the fetch.
- Matching is token-level longest-common-prefix, exactly like the in-memory
  trie — so there is no byte/retokenization keying problem: a checkpoint
  whose tokens are a strict prefix of the request extends any cache type,
  and a *longer/divergent* checkpoint is usable only when the cache is
  trimmable (full-attention KV; hybrid GDN/Mamba states are not).
- Save policy: only prompts of at least ``min_tokens``; at most one
  checkpoint per ``save_every`` new tokens along a conversation lineage;
  when the new checkpoint is trimmable, stored strict-prefix checkpoints
  are superseded (deleted) — mirroring the trie's ``pop_prefixes``.
- Storage: one ``.safetensors`` per checkpoint via the same state/meta_state
  protocol as ``mlx_lm.models.cache.save_prompt_cache``, plus an
  ``index.json`` holding the token lists. ``None`` entries inside a cache's
  ``state`` (e.g. an absent fp16 tier in ``TurboQuantKVCache``) are stored
  as placeholder arrays whose key paths are recorded in the file metadata
  and put back to ``None`` on load.
- Eviction: least-recently-used by total bytes (``budget_gb``).
- Writes happen on a background thread so the next request is never blocked
  behind a multi-GB checkpoint write; a flush runs at exit (the "shutdown
  save"). Single-server-process per cache directory is assumed.

Enable with ``turboquant-serve --disk-cache [DIR]``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np

_FORMAT_VERSION = 1
_INDEX_NAME = "index.json"
# Loading a checkpoint must win at least this many tokens over what the
# in-memory cache already covers — below that the file I/O isn't worth it.
_MIN_GAIN_TOKENS = 256


def _model_key_str(model: Any) -> str:
    """Stable string form of the server's model key (a tuple of paths)."""
    if isinstance(model, (tuple, list)):
        return json.dumps([None if x is None else str(x) for x in model])
    return str(model)


def _digest(model_key: str, tokens: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(model_key.encode())
    h.update(b"|")
    h.update(np.ascontiguousarray(tokens, dtype=np.int64).tobytes())
    return h.hexdigest()[:32]


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    n = min(a.size, b.size)
    if n == 0:
        return 0
    neq = np.nonzero(a[:n] != b[:n])[0]
    return int(neq[0]) if neq.size else n


# ---------------------------------------------------------------------------
# Checkpoint file format (mlx_lm save_prompt_cache layout + None sentinels)
# ---------------------------------------------------------------------------

_NONE_KEYS_META = "__tq_none_state_keys__"


def _replace_nones(flat):
    """Swap ``None`` state leaves for placeholder arrays (safetensors can
    store neither None nor zero-size arrays); returns the affected key paths
    so load can put the Nones back."""
    import mlx.core as mx

    none_keys = [k for k, v in flat if v is None]
    flat = [
        (k, mx.zeros((1,), dtype=mx.float32) if v is None else v)
        for k, v in flat
    ]
    return flat, none_keys


def _set_none_at(tree, dotted_key: str) -> None:
    """Set the leaf at a tree_flatten-style dotted index path to ``None``."""
    parts = dotted_key.split(".")
    node = tree
    for p in parts[:-1]:
        node = node[int(p)]
    node[int(parts[-1])] = None


def _cache_class(name: str):
    """Resolve a cache class by name: mlx_lm's registry plus TurboQuant's."""
    import mlx_lm.models.cache as _cache_mod

    if name == "TurboQuantKVCache":
        from turboquant_mlx.layers.polar_kv_cache import TurboQuantKVCache

        return TurboQuantKVCache
    cls = getattr(_cache_mod, name, None)
    if cls is None or not hasattr(cls, "from_state"):
        raise ValueError(f"Unknown prompt-cache class in checkpoint: {name!r}")
    return cls


def prepare_cache_payload(prompt_cache: List[Any], metadata: Dict[str, str]):
    """Extract and *evaluate* a cache's serializable payload.

    Must run on the thread that owns the cache's MLX graph (the generation
    thread): lazy arrays built on one thread cannot be evaluated from
    another ("There is no Stream(gpu, 0) in current thread"). The returned
    payload is fully materialized, so the background writer only does file
    I/O. Returns ``(data_dict, meta_dict)`` for ``mx.save_safetensors``.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    cache_data = [c.state for c in prompt_cache]
    cache_info = [c.meta_state for c in prompt_cache]
    cache_classes = [type(c).__name__ for c in prompt_cache]
    flat_data, none_keys = _replace_nones(tree_flatten(cache_data))
    metadata = {**metadata, _NONE_KEYS_META: json.dumps(none_keys)}
    flat_meta = dict(tree_flatten([cache_info, metadata, cache_classes]))
    data = dict(flat_data)
    mx.eval(list(data.values()))
    return data, flat_meta


def save_cache_file(path, prompt_cache: List[Any],
                    metadata: Dict[str, str]) -> None:
    """``save_prompt_cache``-compatible writer that tolerates None state leaves."""
    import mlx.core as mx

    data, meta = prepare_cache_payload(prompt_cache, metadata)
    mx.save_safetensors(str(path), data, meta)


def load_cache_file(path):
    """Load a checkpoint written by ``save_cache_file``.

    Returns ``(prompt_cache, metadata)``.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    arrays, meta = mx.load(str(path), return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    meta = tree_unflatten(list(meta.items()))
    info, metadata, classes = meta
    for key in json.loads(metadata.pop(_NONE_KEYS_META, "[]")):
        _set_none_at(arrays, key)
    cache = [
        _cache_class(name).from_state(state, meta_state)
        for name, state, meta_state in zip(classes, arrays, info)
    ]
    return cache, metadata


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    digest: str
    model_key: str
    tokens: np.ndarray
    nbytes: int
    trimmable: bool
    created: float
    last_used: float

    def to_json(self) -> dict:
        return dict(
            digest=self.digest,
            model_key=self.model_key,
            tokens=self.tokens.tolist(),
            nbytes=self.nbytes,
            trimmable=self.trimmable,
            created=self.created,
            last_used=self.last_used,
        )

    @classmethod
    def from_json(cls, d: dict) -> "_Entry":
        return cls(
            digest=d["digest"],
            model_key=d["model_key"],
            tokens=np.asarray(d["tokens"], dtype=np.int64),
            nbytes=int(d["nbytes"]),
            trimmable=bool(d["trimmable"]),
            created=float(d["created"]),
            last_used=float(d["last_used"]),
        )


class DiskPromptCache:
    """Persist prompt-cache checkpoints under ``cache_dir`` and restore on miss.

    Parameters
    ----------
    cache_dir : str | Path
        Directory for ``index.json`` + per-checkpoint ``.safetensors`` files.
    budget_gb : float
        Max total checkpoint bytes; least-recently-used entries are evicted.
    min_tokens : int
        Never checkpoint prompts shorter than this.
    save_every : int
        Along one conversation lineage, only checkpoint after at least this
        many new tokens since the stored prefix (a restart loses at most this
        much prefill).
    sync : bool
        Write checkpoints synchronously instead of on the background thread
        (used by tests).
    """

    def __init__(self, cache_dir, budget_gb: float = 10.0,
                 min_tokens: int = 512, save_every: int = 1024,
                 sync: bool = False, log: TextIO = sys.stderr):
        self.dir = Path(cache_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget_bytes = int(budget_gb * (1 << 30))
        self.min_tokens = int(min_tokens)
        self.save_every = int(save_every)
        self.log = log
        self._lock = threading.Lock()
        self._entries: Dict[str, _Entry] = {}
        # Enqueued-but-unwritten checkpoints, keyed like _entries. The save
        # policy must see these too, or rapid-fire inserts (the server
        # checkpoints each prompt segment) race past the stride throttle and
        # write near-duplicate multi-GB files.
        self._pending: Dict[str, tuple] = {}
        self._load_index()

        self._sync = sync
        self._queue: Optional[queue.Queue] = None
        if not sync:
            self._queue = queue.Queue(maxsize=4)
            worker = threading.Thread(
                target=self._worker, daemon=True, name="tq-disk-cache"
            )
            worker.start()

        self._orig_fetch = None
        self._orig_insert = None

    # -- index ------------------------------------------------------------

    def _file(self, digest: str) -> Path:
        return self.dir / f"{digest}.safetensors"

    def _load_index(self) -> None:
        index_path = self.dir / _INDEX_NAME
        entries: Dict[str, _Entry] = {}
        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data.get("entries", []):
                entry = _Entry.from_json(d)
                if self._file(entry.digest).exists():
                    entries[entry.digest] = entry
        except FileNotFoundError:
            pass
        except Exception as e:
            self.log.write(
                f"[disk-cache] index unreadable ({e}); starting empty\n"
            )
        self._entries = entries
        # Prune checkpoint files the index doesn't know about (e.g. a write
        # that crashed before the index update).
        for f in self.dir.glob("*.safetensors"):
            if f.stem not in self._entries:
                try:
                    f.unlink()
                except OSError:
                    pass

    def _save_index(self) -> None:
        # Called with self._lock held. Atomic via tmp + rename.
        tmp = self.dir / (_INDEX_NAME + ".tmp")
        data = dict(
            version=_FORMAT_VERSION,
            entries=[e.to_json() for e in self._entries.values()],
        )
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.dir / _INDEX_NAME)

    def _remove_entry(self, digest: str) -> None:
        # Called with self._lock held.
        self._entries.pop(digest, None)
        try:
            self._file(digest).unlink()
        except OSError:
            pass

    def _evict_over_budget(self, protect: str) -> None:
        # Called with self._lock held.
        total = sum(e.nbytes for e in self._entries.values())
        victims = sorted(self._entries.values(), key=lambda e: e.last_used)
        for e in victims:
            if total <= self.budget_bytes:
                break
            if e.digest == protect:
                continue
            total -= e.nbytes
            self._remove_entry(e.digest)
            self.log.write(
                f"[disk-cache] evicted {e.tokens.size}-token checkpoint "
                f"({e.nbytes / (1 << 20):.0f} MB, LRU)\n"
            )

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(e.nbytes for e in self._entries.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- save path ----------------------------------------------------------

    def maybe_save(self, model: Any, tokens: List[int],
                   prompt_cache: List[Any]) -> None:
        """Checkpoint ``prompt_cache`` if the save policy says it's worth it."""
        toks = np.asarray(tokens, dtype=np.int64)
        n = int(toks.size)
        if n < self.min_tokens:
            return
        mk = _model_key_str(model)
        digest = _digest(mk, toks)
        with self._lock:
            best_prefix = 0
            known = list(self._entries.values()) + [
                _e for _e in self._pending.values()
            ]
            for e in known:
                if e.model_key != mk:
                    continue
                lcp = _lcp(e.tokens, toks)
                if lcp == n:
                    return  # a stored checkpoint already covers these tokens
                if lcp == e.tokens.size:
                    best_prefix = max(best_prefix, lcp)
            if best_prefix and n - best_prefix < self.save_every:
                return

        from mlx_lm.models.cache import can_trim_prompt_cache

        try:
            trimmable = bool(can_trim_prompt_cache(prompt_cache))
        except Exception:
            trimmable = False

        # Extract + evaluate the payload here, on the thread that owns the
        # cache's MLX graph (lazy arrays cannot be evaluated cross-thread);
        # the background writer is left with pure file I/O — the slow part.
        metadata = dict(
            version=str(_FORMAT_VERSION),
            model_key=mk,
            tokens=json.dumps(toks.tolist()),
        )
        try:
            data, meta = prepare_cache_payload(prompt_cache, metadata)
        except Exception as e:
            self.log.write(f"[disk-cache] save skipped (serialize: {e})\n")
            return

        now = time.time()
        placeholder = _Entry(
            digest=digest, model_key=mk, tokens=toks, nbytes=0,
            trimmable=trimmable, created=now, last_used=now,
        )
        job = (mk, toks, data, meta, trimmable)
        if self._sync:
            self._do_save(*job)
        else:
            with self._lock:
                self._pending[digest] = placeholder
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                with self._lock:
                    self._pending.pop(digest, None)
                self.log.write(
                    "[disk-cache] save skipped (writer busy)\n"
                )

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._do_save(*job)
            except Exception as e:
                self.log.write(f"[disk-cache] save failed: {e}\n")
            finally:
                self._queue.task_done()

    def _do_save(self, mk: str, toks: np.ndarray, data: dict, meta: dict,
                 trimmable: bool) -> None:
        import mlx.core as mx

        digest = _digest(mk, toks)
        path = self._file(digest)
        t0 = time.time()
        try:
            mx.save_safetensors(str(path), data, meta)
        except Exception as e:
            try:
                path.unlink()
            except OSError:
                pass
            with self._lock:
                self._pending.pop(digest, None)
            self.log.write(f"[disk-cache] save failed: {e}\n")
            return

        nbytes = path.stat().st_size
        now = time.time()
        n = int(toks.size)
        with self._lock:
            self._pending.pop(digest, None)
            self._entries[digest] = _Entry(
                digest=digest, model_key=mk, tokens=toks, nbytes=nbytes,
                trimmable=trimmable, created=now, last_used=now,
            )
            if trimmable:
                # A trimmable checkpoint can always be cut back, so shorter
                # strict-prefix checkpoints of the same lineage are redundant
                # (mirrors the in-memory trie's pop_prefixes).
                for e in list(self._entries.values()):
                    if e.digest == digest or e.model_key != mk:
                        continue
                    if e.tokens.size < n and _lcp(e.tokens, toks) == e.tokens.size:
                        self._remove_entry(e.digest)
            self._evict_over_budget(protect=digest)
            self._save_index()
        self.log.write(
            f"[disk-cache] saved {n}-token checkpoint "
            f"({nbytes / (1 << 20):.0f} MB in {time.time() - t0:.1f}s)\n"
        )

    def flush(self) -> None:
        """Wait for queued checkpoint writes to hit disk (the shutdown save)."""
        if self._queue is not None:
            self._queue.join()

    # -- restore path -------------------------------------------------------

    def restore_into(self, lru, model: Any, tokens: List[int],
                     reused: int) -> bool:
        """Load the best on-disk checkpoint into ``lru`` if it beats ``reused``.

        Returns True when a checkpoint was inserted (caller should re-run the
        LRU fetch to pick it up).
        """
        toks = np.asarray(tokens, dtype=np.int64)
        n = int(toks.size)
        mk = _model_key_str(model)
        with self._lock:
            best = None
            best_usable = reused + _MIN_GAIN_TOKENS
            for e in self._entries.values():
                if e.model_key != mk:
                    continue
                lcp = _lcp(e.tokens, toks)
                if lcp == e.tokens.size:
                    usable = lcp  # checkpoint is a (weak) prefix of the request
                elif e.trimmable:
                    usable = min(lcp, n - 1)  # divergent/longer: trim after load
                else:
                    continue
                if usable >= best_usable:
                    best, best_usable = e, usable
            if best is None:
                return False
            path = self._file(best.digest)

        t0 = time.time()
        try:
            cache, _meta = load_cache_file(path)
        except Exception as e:
            self.log.write(
                f"[disk-cache] checkpoint unreadable ({e}); dropping it\n"
            )
            with self._lock:
                self._remove_entry(best.digest)
                self._save_index()
            return False

        with self._lock:
            if best.digest in self._entries:
                self._entries[best.digest].last_used = time.time()
                self._save_index()
        lru.insert_cache(model, best.tokens.tolist(), cache)
        self.log.write(
            f"[disk-cache] restored {best.tokens.size}-token checkpoint "
            f"({best.nbytes / (1 << 20):.0f} MB in {time.time() - t0:.1f}s) — "
            f"prefill resumes from token {best_usable}/{n}\n"
        )
        return True

    # -- server integration ---------------------------------------------------

    def install(self) -> "DiskPromptCache":
        """Wrap ``LRUPromptCache`` so all servers persist through this store."""
        from mlx_lm.models.cache import LRUPromptCache

        store = self
        self._orig_fetch = LRUPromptCache.fetch_nearest_cache
        self._orig_insert = LRUPromptCache.insert_cache

        def fetch_nearest_cache(lru, model, tokens):
            cache, rest = store._orig_fetch(lru, model, tokens)
            try:
                reused = len(tokens) - len(rest)
                if store.restore_into(lru, model, tokens, reused):
                    cache, rest = store._orig_fetch(lru, model, tokens)
            except Exception as e:  # persistence must never break serving
                store.log.write(f"[disk-cache] restore skipped: {e}\n")
            return cache, rest

        def insert_cache(lru, model, tokens, prompt_cache, **kwargs):
            store._orig_insert(lru, model, tokens, prompt_cache, **kwargs)
            try:
                store.maybe_save(model, tokens, prompt_cache)
            except Exception as e:
                store.log.write(f"[disk-cache] save skipped: {e}\n")

        LRUPromptCache.fetch_nearest_cache = fetch_nearest_cache
        LRUPromptCache.insert_cache = insert_cache
        atexit.register(self.flush)
        return self

    def uninstall(self) -> None:
        from mlx_lm.models.cache import LRUPromptCache

        if self._orig_fetch is not None:
            LRUPromptCache.fetch_nearest_cache = self._orig_fetch
        if self._orig_insert is not None:
            LRUPromptCache.insert_cache = self._orig_insert
        self._orig_fetch = None
        self._orig_insert = None


def install(cache_dir, budget_gb: float = 10.0, min_tokens: int = 512,
            save_every: int = 1024, log: TextIO = sys.stderr) -> DiskPromptCache:
    """Create a ``DiskPromptCache`` and patch it into ``mlx_lm.server``."""
    store = DiskPromptCache(
        cache_dir, budget_gb=budget_gb, min_tokens=min_tokens,
        save_every=save_every, log=log,
    )
    return store.install()
