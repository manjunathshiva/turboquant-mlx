"""Learning cache: pin the experts *your* traffic actually routes to (colibrì ③).

The shipped `hot_experts.json` (0.13.0) is a good cold-start prior, but it is one
profile baked by whoever converted the model. colibrì instead records which
experts each user's own prompts route to and auto-pins the hottest at startup —
"it gets faster the more you use it", and their learned pins measure 66-98%
cache-hit against our ~45% at a comparable budget.

This is that idea on our stack, with two deliberate differences:

* **We do not write into the model directory.** colibrì drops `.coli_usage` next
  to the weights; ours are usually a *HuggingFace snapshot* — shared, content-
  addressed, and not ours to pollute (a stale file there would also survive a
  re-download and silently mis-pin). The profile lives in a per-user cache dir,
  keyed by the model, and a read-only or missing directory is never fatal.
* **Counting is per-routing, not per-forward.** A prefill forward's *unique*
  expert set is close to "all of them" on a 256-expert MoE, so counting the set
  once per forward would flatten the very signal we want. We count every
  (token, expert) routing, which is what makes hot experts actually hot.

Schema is deliberately the same `{"pin": [[layer, expert], ...]}` that
`calibrate_experts.py` emits and `hot_experts.json` uses, so a learned profile
feeds the existing pin/preload path unchanged — and can be uploaded as a shipped
hotlist if it turns out to be a good one.
"""

import hashlib
import json
import os
import tempfile

import numpy as np

_PROFILE_VERSION = 1
_ENV_DIR = "TURBOQUANT_USAGE_DIR"

# Old evidence is multiplied by this when a run's counts are merged in, so the
# profile tracks a drifting workload (switching from chat to agent work re-pins
# within a few runs) instead of being frozen by its first heavy session.
_DECAY = 0.7

# Below this many recorded routings the profile is noise — a handful of tokens
# would pin whatever the greeting happened to touch. Keep using the shipped
# hotlist until the user's own evidence beats it.
_MIN_ROUTINGS = 20_000


def default_dir() -> str:
    if os.environ.get(_ENV_DIR):
        return os.environ[_ENV_DIR]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "turboquant-mlx", "usage")


def profile_path(model_path: str, directory: str | None = None) -> str:
    """A stable per-model filename. The model path can be a long HF snapshot
    dir, so key on a hash but keep a readable prefix for humans poking around."""
    name = os.path.basename(os.path.abspath(model_path.rstrip("/")))[:40]
    h = hashlib.sha1(os.path.abspath(model_path).encode()).hexdigest()[:8]
    return os.path.join(directory or default_dir(), f"{name}-{h}.json")


class UsageProfile:
    """Per-(layer, expert) routing counts, persisted across runs."""

    def __init__(self, num_experts: int = 0):
        self.num_experts = num_experts
        self._counts: dict[int, np.ndarray] = {}   # layer -> float64[num_experts]
        self.routings = 0                          # total (token, expert) pairs
        self._dirty = False
        self._saved = False

    # -- recording -----------------------------------------------------
    def record(self, layer_idx: int, routed: np.ndarray) -> None:
        """`routed` is the flat routing array for one forward (may repeat an
        expert once per token). Vectorised: one bincount per layer per forward."""
        if routed.size == 0:
            return
        # Width must track the widest expert id ever SEEN, not the first
        # forward's. `self.num_experts or ...` froze it at whatever the first
        # call happened to route to, and every higher expert id was then
        # silently truncated away by the bincount slice below — a profile that
        # quietly forgets the top half of a 256-expert MoE.
        n = max(self.num_experts, int(routed.max()) + 1)
        self.num_experts = n
        c = self._counts.get(layer_idx)
        if c is None or c.size < n:
            grown = np.zeros(n, dtype=np.float64)
            if c is not None:
                grown[:c.size] = c
            c = grown
            self._counts[layer_idx] = c
        c += np.bincount(routed, minlength=c.size)[:c.size]
        self.routings += int(routed.size)
        self._dirty = True

    # -- pin selection -------------------------------------------------
    def is_mature(self, min_routings: int = _MIN_ROUTINGS) -> bool:
        return self.routings >= min_routings

    def pin_spec(self, limit: int | None = None) -> dict:
        """`{"pin": [[layer, expert], ...]}`, hottest first — the same schema as
        hot_experts.json, so the loader's existing pin/preload path takes it."""
        ranked = []
        for layer, c in self._counts.items():
            for e in np.nonzero(c)[0]:
                ranked.append((float(c[e]), int(layer), int(e)))
        ranked.sort(key=lambda t: -t[0])
        if limit:
            ranked = ranked[:limit]
        return {"pin": [[l, e] for _, l, e in ranked],
                "routings": self.routings, "version": _PROFILE_VERSION}

    # -- persistence ---------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "UsageProfile":
        p = cls()
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return p                                  # absent/corrupt -> empty
        # A truncated/edited cache file can be valid JSON that is not an object
        # ("[]", "null", "42"). d.get() would then raise AttributeError, which
        # the clause above does not catch — crashing the loader at startup over
        # a corrupt *optimisation* file.
        if not isinstance(d, dict) or d.get("version") != _PROFILE_VERSION:
            return p
        try:
            p.num_experts = int(d["num_experts"])
            p.routings = int(d.get("routings", 0))
            for k, v in d["counts"].items():
                p._counts[int(k)] = np.asarray(v, dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return cls()                              # malformed -> start over
        return p

    def merged_with(self, older: "UsageProfile", decay: float = _DECAY
                    ) -> "UsageProfile":
        """This run's counts on top of a decayed history."""
        out = UsageProfile(max(self.num_experts, older.num_experts))
        for layer in set(self._counts) | set(older._counts):
            size = out.num_experts
            acc = np.zeros(size, dtype=np.float64)
            old = older._counts.get(layer)
            if old is not None:
                acc[:old.size] += decay * old
            new = self._counts.get(layer)
            if new is not None:
                acc[:new.size] += new
            out._counts[layer] = acc
        out.routings = int(decay * older.routings) + self.routings
        return out

    def save(self, path: str) -> bool:
        """Atomic write; never raises — a profile is an optimisation, and a
        read-only cache dir must not take the run down with it."""
        if not self._dirty and self.routings == 0:
            return False
        tmp = None
        try:
            # dirname("profile.json") is "" -> makedirs("") raises, and the
            # OSError guard below turned that into a silent "never persisted".
            # A bare relative --usage-file is a reasonable thing to pass.
            d_name = os.path.dirname(path) or "."
            os.makedirs(d_name, exist_ok=True)
            d = {
                "version": _PROFILE_VERSION,
                "num_experts": self.num_experts,
                "routings": self.routings,
                "counts": {str(k): v.tolist() for k, v in self._counts.items()},
            }
            fd, tmp = tempfile.mkstemp(dir=d_name, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(d, f)
            os.replace(tmp, path)                     # atomic
            return True
        except OSError:
            # don't leave orphaned .tmp files in the user's cache dir
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return False

    def update_on_disk(self, path: str, decay: float = _DECAY) -> bool:
        """Fold this run into whatever history is already there.

        Idempotent: an explicit flush plus the atexit hook (or two
        load_streaming calls in one process) must not merge the same counts
        twice and double-weight this session against the decayed history.
        """
        if self._saved:
            return False
        ok = self.merged_with(UsageProfile.load(path), decay).save(path)
        self._saved = ok
        return ok
