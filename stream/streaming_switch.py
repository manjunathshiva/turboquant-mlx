"""Streaming replacement for PolarQuantizedSwitchLinear.

Holds no resident expert weights. On each forward it discovers the
router-selected experts, loads *only those* expert slices from the mmap'd
safetensors (through an LRU ``ExpertCache``), and runs the exact same fused
polar kernels as the resident layer — assembling a small (n_selected, ...)
weight stack and remapping the routing indices to local positions.

Output is numerically identical to PolarQuantizedSwitchLinear because it uses
the same kernels on the same expert bytes; only the *set* of experts present
in memory at any moment differs.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.rotation import rotate_input
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv


class ExpertCache:
    """Process-wide LRU cache of per-expert (weight, scales) mx.arrays, with
    optional cross-layer speculative prefetch.

    Shared across every streaming layer so a single byte budget bounds the
    total resident expert memory. Keyed by (weight_key, expert) which is
    globally unique (weight_key already encodes layer + projection).

    The experts missing for a single ``gather`` are read in one batch and,
    when ``prefetch_workers > 1``, their ``pread``s are fanned across a thread
    pool. ``pread`` releases the GIL, so the per-layer disk stall drops from the
    sum of the slice reads to roughly the slowest one. MLX array construction
    and ``eval`` stay on the calling thread, so the produced tensors are
    bit-identical to the serial path.

    **Speculative prefetch** (``prefetch_ahead > 0``): when a layer starts we
    kick off *background* disk reads for the experts the previous token used at
    the next layer(s) — a near-free, high-accuracy predictor because MoE routing
    is strongly correlated token-to-token. The reads land in ``_staging`` as raw
    numpy buffers; the next ``gather`` claims them on the calling thread (where
    it builds the mx.array), so the disk latency overlaps with compute and *no
    MLX is ever touched off the main thread*. Output stays bit-identical:
    prefetch changes only *when* bytes are read, never *which* experts a gather
    returns (mispredicted slices simply age out of staging unused).
    """

    def __init__(self, reader, budget_bytes: int, *, prefetch_workers: int = 8,
                 prefetch_ahead: int = 1,
                 staging_budget_bytes: int = 1_500_000_000,
                 pin_keys=None):
        self.reader = reader
        self.budget = budget_bytes
        self.cur = 0
        self._od: "OrderedDict[tuple, tuple]" = OrderedDict()
        # stats
        self.hits = 0             # served resident from _od
        self.misses = 0           # critical-path disk reads
        self.prefetch_hits = 0    # served from a completed background prefetch
        self.bytes_read = 0           # critical-path bytes
        self.bytes_prefetched = 0     # background prefetch bytes
        self.prefetched = 0           # experts prefetched
        self.prefetch_dropped = 0     # prefetched then evicted from staging unused
        self.read_runs = 0            # coalesced range reads issued (critical path)
        self.expert_reads = 0         # experts loaded on the critical path
        # parallel read pool (shared by critical-path misses and prefetch)
        self._workers = max(1, int(prefetch_workers))
        self._pool = (ThreadPoolExecutor(max_workers=self._workers)
                      if self._workers > 1 else None)
        # speculative-prefetch state (needs the pool to run in background)
        self._prefetch_ahead = max(0, int(prefetch_ahead)) if self._pool else 0
        # saturation throttle: after warming up, if prefetch is rescuing too few
        # of the would-be misses (its reads arrive too late because the bus is
        # saturated), turn prefetch OFF so it stops wasting bandwidth. Judged by
        # the RESCUE RATE — prefetch_hits / (prefetch_hits + misses), NOT raw
        # utilization, since prefetch over-fetches but can still rescue a large
        # share of misses on fast storage. Self-disables on the bus-bound USB
        # 235B (~0.1% rescue); stays on for fast NVMe (tens of %).
        self._throttle_warmup = 2000
        self._throttle_min_rescue = 0.03
        self._throttle_decided = False
        self._staging_cap = max(0, int(staging_budget_bytes))
        self._staging: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (np_buf, dt)
        self._staging_bytes = 0
        self._inflight: set = set()
        self._lock = threading.Lock()        # guards _staging / _inflight / stats
        self._layer_keys: dict = {}          # layer_idx -> [(wkey, skey), ...]
        self._last_experts: dict = {}        # layer_idx -> experts (previous token)
        # calibration trace: per-decode-token expert selections, for #2/#3
        self._trace_on = False
        self._trace: list = []               # [(layer_idx, tuple(sorted experts)), ...]
        # frequency-based pinning (#2): these (wkey, expert) keys never evict;
        # they live in a separate dict so eviction stays O(1) on the LRU set.
        self._pin_keys: set = set(pin_keys or ())
        self._pinned: dict = {}              # (wkey, e) -> (w, s, nbytes)
        self._pin_hits = 0

    # -- speculative prefetch -----------------------------------------
    def register_layer(self, layer_idx: int, proj_keys: list):
        """Record the (weight_key, scales_key) of every projection in a layer so
        the predictor can prefetch all of them for the next layer at once."""
        self._layer_keys[layer_idx] = list(proj_keys)

    def on_layer_start(self, layer_idx: int, experts: list):
        """Called once per layer per token (from the trigger projection).

        Prefetches the next layer(s)' predicted experts in the background, then
        records this layer's selection for the *next* token to predict from.
        """
        # once enough has been prefetched, judge whether it's worth continuing
        if (self._prefetch_ahead and not self._throttle_decided
                and self.prefetched >= self._throttle_warmup):
            self._throttle_decided = True
            rescue = self.prefetch_hits / max(1, self.prefetch_hits + self.misses)
            if rescue < self._throttle_min_rescue:
                self._prefetch_ahead = 0
                print(f"[stream] prefetch self-disabled: rescue rate {rescue:.0%} < "
                      f"{self._throttle_min_rescue:.0%} (storage appears bandwidth-bound)")
        if self._prefetch_ahead:
            for nxt in range(layer_idx + 1, layer_idx + 1 + self._prefetch_ahead):
                pred = self._last_experts.get(nxt)
                if pred:
                    self._prefetch_layer(nxt, pred)
        self._last_experts[layer_idx] = experts

    def _prefetch_layer(self, layer_idx: int, predicted: list):
        keys = self._layer_keys.get(layer_idx)
        if not keys or self._pool is None:
            return
        with self._lock:
            for wkey, skey in keys:
                for e in predicted:
                    ck = (wkey, e)
                    # already resident (LRU or pinned), staged, or being read
                    if (ck in self._od or ck in self._pinned
                            or ck in self._staging or ck in self._inflight):
                        continue
                    self._inflight.add(ck)
                    self._pool.submit(self._prefetch_one, wkey, skey, e)

    # -- calibration trace (#2 frequencies / #3 co-activation) --------
    def set_trace(self, on: bool = True):
        self._trace_on = on

    def record_trace(self, layer_idx: int, experts):
        # called once per decode token per layer with that token's selected set
        if self._trace_on:
            self._trace.append((layer_idx, tuple(experts)))

    def dump_trace(self, path: str) -> int:
        import json
        with open(path, "w") as f:
            json.dump(self._trace, f)
        return len(self._trace)

    def _prefetch_one(self, wkey: str, skey: str, e: int):
        # disk-only background work; never touches MLX (built later on main thread)
        try:
            wbuf = self.reader.read_expert_np(wkey, e)   # (np_array, mlx_dtype)
            sbuf = self.reader.read_expert_np(skey, e)
            nb = wbuf[0].nbytes + sbuf[0].nbytes
            with self._lock:
                self._inflight.discard((wkey, e))
                # Store both projections under one key so eviction can never
                # orphan half a pair (drop the weight but keep its scales).
                self._staging[(wkey, e)] = (wbuf, sbuf)
                self._staging_bytes += nb
                self.bytes_prefetched += nb
                self.prefetched += 1
                # bound staging: drop oldest (mispredicted) entries FIFO
                while self._staging_bytes > self._staging_cap and len(self._staging) > 1:
                    _, (wb, sb) = self._staging.popitem(last=False)
                    self._staging_bytes -= (wb[0].nbytes + sb[0].nbytes)
                    self.prefetch_dropped += 1
        except Exception:
            with self._lock:
                self._inflight.discard((wkey, e))

    # -- loading -------------------------------------------------------
    def _load_coalesced(self, miss_pairs):
        """Load every missed expert for one projection, coalescing *contiguous*
        expert positions into a single range ``pread`` (the #3 layout win).

        All pairs in one gather share the same (wkey, skey) — gather is called
        per projection — so we sort the missed experts, split them into runs of
        consecutive positions, and read each run once (weight + scales) instead
        of once per expert. On a co-activation-ordered checkpoint a token's
        misses fall into far fewer runs, so this turns many small random reads
        into a few larger sequential ones. The per-expert slices are then built
        into MLX arrays on this (the calling) thread, as before.
        """
        wkey, skey = miss_pairs[0][0], miss_pairs[0][1]
        experts = sorted(p[2] for p in miss_pairs)
        runs = []
        i = 0
        while i < len(experts):
            j = i
            while j + 1 < len(experts) and experts[j + 1] == experts[j] + 1:
                j += 1
            runs.append((experts[i], experts[j] - experts[i] + 1))
            i = j + 1
        self.read_runs += len(runs)
        self.expert_reads += len(experts)

        def _read(t):
            key, start, count = t
            buf, dt = self.reader.read_range_np(key, start, count)
            return key, start, buf, dt

        tasks = []
        for start, count in runs:
            tasks.append((wkey, start, count))
            tasks.append((skey, start, count))
        results = (list(self._pool.map(_read, tasks)) if self._pool
                   else [_read(t) for t in tasks])

        wb, sb = {}, {}
        for key, start, buf, dt in results:
            tgt = wb if key == wkey else sb
            for k in range(buf.shape[0]):
                tgt[start + k] = (buf[k], dt)

        out, arrs = {}, []
        for e in experts:
            wbuf, wdt = wb[e]
            sbuf, sdt = sb[e]
            w = mx.array(wbuf, dtype=wdt)
            s = mx.array(sbuf, dtype=sdt)
            out[(wkey, e)] = (w, s, w.nbytes + s.nbytes)
            arrs.append(w)
            arrs.append(s)
        mx.eval(*arrs)  # force the slice copies out of the read buffers
        return out

    def _evict(self):
        while self.cur > self.budget and len(self._od) > 1:
            _, (_w, _s, nb) = self._od.popitem(last=False)  # oldest
            self.cur -= nb

    def _insert(self, ck, entry):
        """Place a freshly loaded (w, s, nbytes) entry. Pinned keys go to the
        never-evicted dict; everything else into the LRU set."""
        if ck in self._pin_keys:
            self._pinned[ck] = entry
        else:
            self._od[ck] = entry
        self.cur += entry[2]

    def gather(self, wkey: str, skey: str, experts: list[int]):
        """Return stacked (n, out, packed) weight + (n, out, ng) scales for
        ``experts`` (in the given order)."""
        # classify each requested expert against the pinned set, the LRU
        # resident set, and the prefetch staging area; load the rest in a batch.
        miss_pairs = []
        staged = []  # (expert, wbuf, sbuf) — read in background, build here
        with self._lock:
            for e in experts:
                ck = (wkey, e)
                if ck in self._pinned:
                    self.hits += 1
                    self._pin_hits += 1
                    continue
                if ck in self._od:
                    self.hits += 1
                    continue
                if ck in self._staging:
                    wbuf, sbuf = self._staging.pop(ck)
                    self._staging_bytes -= (wbuf[0].nbytes + sbuf[0].nbytes)
                    staged.append((e, wbuf, sbuf))
                    self.prefetch_hits += 1
                else:
                    self.misses += 1
                    miss_pairs.append((wkey, skey, e))

        # Build prefetch-staged slices into MLX arrays on THIS thread (no disk;
        # the bytes were already read in the background). Lock released above so
        # background reads keep flowing while we build.
        for e, wbuf, sbuf in staged:
            w = mx.array(wbuf[0], dtype=wbuf[1])
            s = mx.array(sbuf[0], dtype=sbuf[1])
            mx.eval(w, s)
            self._insert((wkey, e), (w, s, w.nbytes + s.nbytes))

        if miss_pairs:
            loaded = self._load_coalesced(miss_pairs)
            for ck, entry in loaded.items():
                self._insert(ck, entry)
                self.bytes_read += entry[2]

        # Build the stack *before* evicting so freshly loaded slices are still
        # present (and held by ``ws``/``ss``, so eviction can't free them out
        # from under this call).
        ws, ss = [], []
        for e in experts:
            ck = (wkey, e)
            entry = self._pinned.get(ck) or self._od.get(ck)
            w, s, _ = entry
            if ck in self._od:
                self._od.move_to_end(ck)
            ws.append(w)
            ss.append(s)

        self._evict()
        return mx.stack(ws, axis=0), mx.stack(ss, axis=0)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        with self._lock:
            self._staging.clear()
            self._staging_bytes = 0
            self._inflight.clear()

    def stats(self):
        served = self.hits + self.prefetch_hits   # neither incurred critical-path disk
        tot = served + self.misses
        return {
            # effective (latency) hit-rate: fraction served without a blocking read
            "hit_rate": (served / tot) if tot else 0.0,
            "cache_hit_rate": (self.hits / tot) if tot else 0.0,
            "prefetch_hit_rate": (self.prefetch_hits / tot) if tot else 0.0,
            "hits": self.hits,
            "prefetch_hits": self.prefetch_hits,
            "pin_hits": self._pin_hits,
            "misses": self.misses,
            "prefetched": self.prefetched,
            "prefetch_dropped": self.prefetch_dropped,
            "resident_experts": len(self._od) + len(self._pinned),
            "pinned_experts": len(self._pinned),
            "resident_gb": self.cur / 1e9,
            "bytes_read_gb": self.bytes_read / 1e9,
            "bytes_prefetched_gb": self.bytes_prefetched / 1e9,
            "bytes_total_gb": (self.bytes_read + self.bytes_prefetched) / 1e9,
            "read_runs": self.read_runs,
            "expert_reads": self.expert_reads,
            # experts per coalesced read: 1.0 = no coalescing, higher = #3 working
            "experts_per_read": (self.expert_reads / self.read_runs) if self.read_runs else 0.0,
        }


class StreamingSwitchLinear(nn.Module):
    """Drop-in for PolarQuantizedSwitchLinear that streams experts from disk."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bits: int,
        group_size: int,
        needs_rotation: bool,
        codebook: mx.array,
        signs: mx.array,
        weight_key: str,
        scales_key: str,
        cache: ExpertCache,
        layer_idx: int = -1,
        is_trigger: bool = False,
        trit: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.bits = bits
        self.group_size = group_size
        # Ternary (1.58-bit) experts pack 20 base-3 trits per uint32 instead of
        # bit-packing; the kernels and dequant switch to base-3 decode off this.
        # Detect from the 3-entry codebook if the caller didn't say (self-describing).
        self.trit = bool(trit) or codebook.size == 3
        if self.trit:
            # The Metal kernels hardcode n_codes=3 under trit; a mismatched
            # codebook would read out of bounds on the GPU. Fail loud instead.
            if codebook.size != 3:
                raise ValueError(
                    "Ternary (trit) mode requires a 3-entry codebook, "
                    f"got size {codebook.size}."
                )
            # Match the resident layer (ternary is always bit-width 2) so the
            # (bits, group_size, trit) kernel-cache key stays consistent.
            self.bits = 2
        self._needs_rotation = needs_rotation
        # small resident tensors
        self.codebook = codebook
        self.signs = signs
        # streaming wiring (underscored so they are not treated as params)
        self._weight_key = weight_key
        self._scales_key = scales_key
        self._cache = cache
        # speculative-prefetch wiring: exactly one projection per layer is the
        # trigger (fires the next-layer prefetch once per token).
        self._layer_idx = layer_idx
        self._is_trigger = is_trigger
        self.freeze()

    def _dequantize_selected(self, w_sel: mx.array, s_sel: mx.array) -> mx.array:
        """Dequantize only the selected experts to float16 (prefill path).

        Mirrors PolarQuantizedSwitchLinear._dequantize_all but on an
        (n_sel, out, packed) stack instead of all num_experts.
        """
        from turboquant_mlx.core.packing import unpack_indices, unpack_trits
        from turboquant_mlx.core.codebook import dequantize_scalar

        n_sel = w_sel.shape[0]
        n_groups = self.input_dims // self.group_size
        if self.trit:
            idx = unpack_trits(w_sel, self.input_dims)
        else:
            idx = unpack_indices(w_sel, self.bits, self.input_dims)
        w_deq = dequantize_scalar(idx, self.codebook)
        w_deq = w_deq.reshape(n_sel, self.output_dims, n_groups, self.group_size)
        w_deq = w_deq * mx.expand_dims(s_sel, axis=-1)
        return w_deq.reshape(n_sel, self.output_dims, self.input_dims)

    def __call__(self, x, indices, sorted_indices=False):
        if self._needs_rotation:
            x = rotate_input(x, self.signs)

        n_tokens = 1 if x.ndim <= 2 else math.prod(x.shape[:-2])
        k = indices.shape[-1] if indices.ndim >= 1 else 1

        # Discover selected experts (forces a tiny GPU->CPU sync on the
        # routing indices — unavoidable: we must know what to load).
        idx_global = np.asarray(
            indices.reshape(-1).astype(mx.uint32).tolist(), dtype=np.int64
        )
        sel = sorted(set(int(v) for v in idx_global))
        # Kick off the next layer's speculative prefetch as soon as this layer's
        # expert set is known (only the trigger projection, once per token), and
        # record the selection for calibration on decode steps (1 token).
        if self._is_trigger:
            self._cache.on_layer_start(self._layer_idx, sel)
            if n_tokens == 1:
                self._cache.record_trace(self._layer_idx, sel)
        remap = {e: i for i, e in enumerate(sel)}
        w_sel, s_sel = self._cache.gather(self._weight_key, self._scales_key, sel)
        idx_local_flat = mx.array([remap[int(v)] for v in idx_global], dtype=mx.uint32)

        if n_tokens == 1:
            x_flat = x.reshape(-1)
            y = polar_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_flat, idx_local_flat, self.bits, self.group_size,
                trit=self.trit,
            )
            return y.reshape(list(indices.shape) + [1, self.output_dims])

        if n_tokens == k:
            x_2d = x.reshape(k, self.input_dims)
            y = polar_multi_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_2d, idx_local_flat, self.bits, self.group_size,
                trit=self.trit,
            )
            return y.reshape(list(indices.shape) + [1, self.output_dims])

        # Prefill: dequantize only the selected experts, gather_mm locally.
        w_deq = self._dequantize_selected(w_sel, s_sel)
        idx_local = idx_local_flat.reshape(indices.shape)
        return mx.gather_mm(
            x, w_deq.swapaxes(-1, -2),
            rhs_indices=idx_local, sorted_indices=False,
        )

    def _extra_repr(self):
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"num_experts={self.num_experts}, bits={self.bits}, "
            f"group_size={self.group_size}, STREAMING key={self._weight_key}"
        )
