"""Preflight: will this model run on this Mac, and with what flags?

`turboquant-plan` answers that **before** a multi-GB download or a five-minute
load, by reading only safetensors *headers* and `config.json` — it allocates no
tensors, starts no engine, and imports no model framework.

Why this exists: the 0.13.0 auto-guards (`--metal-cache-limit-gb auto`,
`--prompt-cache-max-gb auto`, `--cache-budget-gb auto`) fix tight-memory serving
*at runtime*. They can't answer the question a user asks first — "will this fit,
and what do I pass?" Every crash behind those guards was predictable from the
model headers plus one machine number.

The projection is deliberately explicit about what is exact and what is an
estimate:

  weights          EXACT   — summed from the safetensors headers
  KV cache         DERIVED — 2 (K+V) x n_kv_heads x head_dim x full-attn layers
                             x itemsize x context. Validated on the ternary 35B:
                             predicts 0.67 GB at 32K vs 0.62 GB measured (+8%,
                             i.e. conservative, which is what a planner should be).
  prefill workspace ESTIMATE — chunk x context x n_attn_heads x 2 bytes, the
                             transient attention scratch that scales with BOTH
                             the chunk size and the context. Calibrated against
                             the measured 16 GB mini boundary: at 21K context it
                             puts --prefill-step-size 2048 over the wired cap
                             (observed: OOM) and 256/128 under it (observed: runs).

A HuggingFace repo id is planned **over the network from its headers** — never
by downloading it, which would defeat the point. Measured: the 12.6 GB down4 repo
plans in ~2.5 s and pulls 232 KB into a cold cache.

Usage:
    turboquant-plan --model <path_or_repo>
    turboquant-plan --model <path> --context 32768 --kv-bits 8
    turboquant-plan --model <path> --wired-gb 13.5 --ram-gb 16   # plan for ANOTHER Mac
    turboquant-plan --model <path> --json

Exit codes: 0 = runnable (warnings allowed), 1 = won't fit / missing files,
2 = bad usage.
"""

import argparse
import glob
import json
import os
import struct
import subprocess
import sys

from turboquant_mlx.cache_budget import auto_cache_budget, projected_peak_bytes
from turboquant_mlx.expert_naming import is_streamed_expert_key

_ITEMSIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
             "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "F8_E4M3": 1,
             "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1}

_GB = 1e9

# What we do NOT model line-by-line: the (capped) MLX buffer-reuse cache,
# per-layer activations, and allocator fragmentation. Calibrated, not guessed:
# the down4 35B measures a 13.60 GB peak on the mini against 12.59 GB of weights
# and ~0.1 GB of KV, so the residue is ~0.9 GB. NOTE this is deliberately NOT the
# 2 GB that serve.py/loader.py reserve — theirs has to *cover* KV and prefill
# workspace, which this projection already counts explicitly; reusing 2 GB here
# would double-count and wrongly report "streaming" for models that run resident.
_RESERVE_BYTES = 1.0 * _GB
_EXIT_OK, _EXIT_UNFIT, _EXIT_USAGE = 0, 1, 2


# --------------------------------------------------------------------------
# header-only model inspection
# --------------------------------------------------------------------------

def read_safetensors_index(model_path: str) -> dict:
    """{tensor_name: (dtype, shape)} from shard headers. Opens each file, reads
    the 8-byte length + JSON header, closes it. No payload is touched."""
    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {model_path}")
    index = {}
    for shard in shards:
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            index[name] = (meta["dtype"], tuple(meta["shape"]))
    return index


def read_remote_index(repo_id: str) -> dict:
    """{tensor_name: (dtype, shape)} for an un-downloaded HF repo.

    `get_safetensors_metadata` fetches each shard's header with an HTTP range
    request — bytes, not gigabytes. This is what makes "plan *before* the
    download" true rather than a slogan: the 12.6 GB down4 repo answers in
    ~2.5 s over the network with 2,026 tensors described and nothing cached.
    """
    from huggingface_hub import get_safetensors_metadata
    meta = get_safetensors_metadata(repo_id)
    index = {}
    for shard in meta.files_metadata.values():
        for name, t in shard.tensors.items():
            index[name] = (t.dtype, tuple(t.shape))
    if not index:
        raise FileNotFoundError(f"{repo_id}: no safetensors metadata")
    return index


def read_remote_config(repo_id: str) -> dict:
    """config.json only — a few KB, not the checkpoint."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(repo_id, "config.json")) as f:
        return json.load(f)


def footprint(index: dict) -> dict:
    """Exact byte split: total / streamable experts / always-resident."""
    total = expert = 0
    for name, (dtype, shape) in index.items():
        n = 1
        for d in shape:
            n *= d
        b = n * _ITEMSIZE.get(dtype, 4)
        total += b
        # what the streaming swap pages from disk — same rule as
        # stream/loader.py:_streamed_expert_bytes, now shared so they can't
        # drift again (they did: both missed laguna's `mlp.experts`)
        if is_streamed_expert_key(name):
            expert += b
    return {"total_bytes": total, "expert_bytes": expert,
            "resident_bytes": total - expert}


def _text_config(cfg: dict) -> dict:
    """Multimodal repos nest the LM config under text_config."""
    return cfg.get("text_config", cfg)


def _pos_int(value, fallback):
    """Config values come from arbitrary HF repos — treat anything that is not a
    positive integer as 'not set'. Guards the real cases: `sliding_window: -1`
    (a negative window subtracted KV, under-predicting), and a fractional
    `full_attention_interval` (int() -> 0 -> ZeroDivisionError)."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _attention_layers(cfg: dict) -> tuple:
    """(n_full_attention, n_sliding_attention, n_layers, note).

    Three families, three different ways of saying which layers hold KV:

    * `layer_types` — Qwen3.5/3.6 (`linear_attention` = GatedDeltaNet, fixed-size
      recurrent state, no KV growth), GPT-OSS and Gemma (`sliding_attention`,
      whose KV is real but capped at `sliding_window`).
    * `hybrid_override_pattern` — Nemotron-H's Mamba hybrid, where `*` is an
      attention layer and `M`/`E` are Mamba/MLP. Only 8 of its 88 layers are
      attention; reading `num_hidden_layers` instead over-predicts KV by 11x.
    * `full_attention_interval` — older hybrids.
    * `linear_attn_config.full_attn_layers` — Kimi Linear / Kimi K3 KDA
      hybrids: an explicit (1-based) list of the full-attention (MLA) layers;
      everything else is KDA linear attention with a fixed-size recurrent
      state. K3: 24 of 93 layers — assuming dense over-predicts KV ~4x even
      before MLA's latent geometry (see kv_bytes) is accounted for.

    Everything else is assumed dense full-attention.
    """
    c = _text_config(cfg)
    n_layers = c.get("num_hidden_layers")

    lac = c.get("linear_attn_config")
    if isinstance(lac, dict) and isinstance(lac.get("full_attn_layers"), list):
        n_full = len(lac["full_attn_layers"])
        n_kda = len(lac.get("kda_layers") or [])
        n_total = n_layers or (n_full + n_kda)
        return n_full, 0, n_total, \
            f"KDA hybrid: {n_full}/{n_total} full-attention (MLA)"

    types = c.get("layer_types")
    if isinstance(types, list) and types:
        n_full = sum(1 for t in types if "full" in str(t))
        n_slide = sum(1 for t in types if "sliding" in str(t))
        bits = [f"{n_full}/{len(types)} full-attention"]
        if n_slide:
            bits.append(f"{n_slide} sliding")
        # layer_types is itself an authoritative layer count
        return n_full, n_slide, n_layers or len(types), \
            "hybrid: " + ", ".join(bits)

    pat = c.get("hybrid_override_pattern")
    if isinstance(pat, str) and "*" in pat:
        n_full = pat.count("*")
        return n_full, 0, n_layers or len(pat), \
            f"Mamba hybrid: {n_full}/{len(pat)} attention layers"

    iv = _pos_int(c.get("full_attention_interval"), 0)
    if iv and n_layers:
        return n_layers // iv, 0, n_layers, \
            f"hybrid: every {iv}th layer full-attention"

    return n_layers, 0, n_layers, "all layers full-attention"


def kv_bytes(cfg: dict, context: int, kv_bits: int | None = None) -> tuple:
    """(total_bytes_at_context, effective_bytes_per_token, note).

    Full-attention layers grow with the context; sliding-window layers stop at
    `sliding_window` tokens, so their cost is bounded — small, but ignoring them
    under-predicts, and under-predicting is the direction that OOMs.
    """
    c = _text_config(cfg)
    n_kv = c.get("num_key_value_heads") or c.get("num_attention_heads")
    head_dim = c.get("head_dim")
    if head_dim is None and c.get("hidden_size") and c.get("num_attention_heads"):
        head_dim = c["hidden_size"] // c["num_attention_heads"]
    n_full, n_slide, n_layers, note = _attention_layers(cfg)
    if not (n_layers and n_kv and head_dim):
        return None, None, "config lacks KV geometry"

    itemsize = 2.0  # fp16 KV
    if kv_bits:
        # TurboQuant KV-quant stores kv_bits/16 of the fp16 payload (+ scales,
        # which are small); --kv-bits 8 halving KV matches the measured 35B.
        itemsize = 2.0 * (kv_bits / 16.0)

    kv_rank = _pos_int(c.get("kv_lora_rank"), 0)
    if kv_rank:
        # MLA caches the shared latent (kv_lora_rank) + the MQA rope key —
        # once per layer, NOT per head, and K/V share it. Kimi K3: 576 dims
        # ≈ 1.2 KB/token/layer vs the 48 KB the dense formula would charge.
        rope_dim = _pos_int(c.get("qk_rope_head_dim"), 0)
        per_layer_token = (kv_rank + rope_dim) * itemsize
        note += ", MLA latent KV"
    else:
        per_layer_token = 2 * n_kv * head_dim * itemsize

    total = n_full * context * per_layer_token

    lac = c.get("linear_attn_config")
    if isinstance(lac, dict) and isinstance(lac.get("kda_layers"), list):
        # KDA layers hold a fixed-size recurrent state (heads × head_dim² fp32
        # + short-conv tails) that doesn't grow with context but isn't free:
        # ~450 MB on K3. Charge it as a constant.
        kh = _pos_int(lac.get("num_heads"), n_kv or 0)
        kd = _pos_int(lac.get("head_dim"), head_dim or 0)
        ck = _pos_int(lac.get("short_conv_kernel_size"), 4)
        total += len(lac["kda_layers"]) * (
            kh * kd * kd * 4.0 + 3 * (ck - 1) * kh * kd * 2.0
        )
        note += f" + fixed KDA state ({len(lac['kda_layers'])} layers)"
    if n_slide:
        # an absent/zero/negative/garbage window means "not windowed" -> full
        # context. Never let it shrink the total: that under-predicts, and
        # under-prediction is the direction that OOMs.
        window = _pos_int(c.get("sliding_window"), context)
        total += n_slide * min(context, window) * per_layer_token
        note += f" (window {window})"
    return total, (total / context if context else 0.0), note


def prefill_workspace_bytes(cfg: dict, context: int, step: int) -> float:
    """Transient attention scratch for one prefill chunk (ESTIMATE).

    Scales with chunk size AND context — which is why a long agent turn can OOM
    at a chunk size that was fine when the conversation was short.
    """
    c = _text_config(cfg)
    heads = c.get("num_attention_heads") or 16
    return float(step) * float(context) * float(heads) * 2.0


# --------------------------------------------------------------------------
# machine
# --------------------------------------------------------------------------

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
    frac = 0.66 if ram_bytes <= 36 * _GB else 0.75
    return frac * ram_bytes


def machine(wired_gb: float | None = None, ram_gb: float | None = None) -> dict:
    """Metal working-set cap + system RAM. Overridable so you can plan for a
    machine you're not sitting at ("will this run on my 16 GB mini?").

    The two numbers MUST describe the same machine. Reading this device's
    working set while the caller has assumed someone else's RAM is how you tell
    a 16 GB mini that a 30 GB model fits — so an assumed --ram-gb estimates the
    cap from that RAM instead of querying Metal here.
    """
    out = {"assumed": bool(wired_gb or ram_gb), "wss_estimated": False}
    if ram_gb:
        out["ram_bytes"] = ram_gb * _GB
    else:
        try:
            out["ram_bytes"] = float(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            out["ram_bytes"] = None

    if wired_gb:
        out["wss_bytes"] = wired_gb * _GB
    elif ram_gb:
        # planning for another machine: this device's cap is irrelevant
        out["wss_bytes"] = estimate_wss(out["ram_bytes"])
        out["wss_estimated"] = True
    else:
        try:
            import mlx.core as mx
            out["wss_bytes"] = float(
                mx.device_info()["max_recommended_working_set_size"])
        except Exception:
            # no Metal device (Linux/CI): fall back to the RAM estimate rather
            # than leaving it unknown, which made every verdict "needs a bump"
            out["wss_bytes"] = estimate_wss(out["ram_bytes"])
            out["wss_estimated"] = out["wss_bytes"] is not None
    return out


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def build_plan(model_path: str, context: int = 16384, kv_bits: int | None = None,
               step: int | None = None, wired_gb: float | None = None,
               ram_gb: float | None = None, remote: bool = False) -> dict:
    incomplete = False
    if remote:
        cfg = read_remote_config(model_path)
        index = read_remote_index(model_path)
    else:
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"{cfg_path} not found")
        with open(cfg_path) as f:
            cfg = json.load(f)
        index = read_safetensors_index(model_path)
        # An explicit local path is taken at the user's word — but if shards are
        # missing, every byte below is undercounted and the verdict is optimistic
        # in the one direction that gets someone an OOM instead of an answer.
        incomplete = not is_complete_checkpoint(model_path)
    fp = footprint(index)
    mach = machine(wired_gb, ram_gb)
    c = _text_config(cfg)

    kv_total, kv_pt, kv_note = kv_bytes(cfg, context, kv_bits)
    _, _, n_layers, _ = _attention_layers(cfg)
    kv_total = kv_total or 0

    wss = mach["wss_bytes"]
    ram = mach["ram_bytes"]
    is_moe = fp["expert_bytes"] > 0
    warnings, flags = [], []
    if incomplete:
        warnings.append("shards are missing from this directory — the download "
                        "looks incomplete, so every size below is UNDER-counted")

    # The Metal wired cap is NOT fixed — `sysctl iogpu.wired_limit_mb` raises it,
    # which is exactly what the 16 GB mini does to run a 12.6 GB model resident.
    # So the hard ceiling is RAM (minus what the OS needs), and the cap is a
    # knob. Raising it too far starves the OS and can panic a small box, so cap
    # the recommendation at 90% of RAM — the measured mini runs at 14 GiB / 16 GB
    # (87.5%).
    raisable = min(ram * 0.90, ram - 1.5 * _GB) if ram else None
    ceiling = max(wss or 0, raisable or 0) or wss

    def peak_at(s):
        # weights + KV + this chunk's transient + a runtime reserve. The reserve
        # is not slack: MLX's buffer-reuse cache alone grew ~80 MB per 1K prompt
        # tokens on the mini (~1.7 GB at 21K) before --metal-cache-limit-gb auto
        # capped it, and activations/fragmentation live here too. Same 2 GB the
        # auto-guards reserve.
        return (fp["total_bytes"] + kv_total
                + prefill_workspace_bytes(cfg, context, s) + _RESERVE_BYTES)

    # Largest chunk whose projection keeps real headroom under the (possibly
    # raised) ceiling. The 0.95 is deliberate: a chunk size chosen to *just* fit
    # is the crash we already shipped fixes for, and prefill throughput barely
    # moves between 512 and 128 while peak transient scales linearly. Bias
    # conservative — this tool exists to prevent OOMs, not to win benchmarks.
    steps = [2048, 512, 256, 128] if step is None else [step]
    chosen = None
    for s in steps:
        if step is not None or ceiling is None or peak_at(s) <= ceiling * 0.95:
            chosen = s
            break
    if chosen is None:
        chosen = steps[-1]
    peak = peak_at(chosen)
    workspace = prefill_workspace_bytes(cfg, context, chosen)

    # ---- verdict -------------------------------------------------------
    needs_bump = False
    if wss is None and ram is None:
        mode, ok = "unknown", True
        warnings.append("no Metal device info — machine limits unknown")
    elif wss and peak <= wss:
        mode, ok = "resident", True
    elif ceiling and peak <= ceiling:
        mode, ok, needs_bump = "resident", True, True
    elif is_moe and ceiling and (fp["resident_bytes"] + 0.5 * _GB + kv_total
                                 + workspace + _RESERVE_BYTES) <= ceiling:
        mode, ok = "streaming", True
        needs_bump = bool(wss and (fp["resident_bytes"] + 0.5 * _GB + kv_total
                                   + workspace + _RESERVE_BYTES) > wss)
    else:
        mode, ok = "no-fit", False

    # ---- recommended flags --------------------------------------------
    if needs_bump and ram:
        want = min(peak + 0.5 * _GB, raisable)
        mb = int(want / (1024**2))
        flags.append(f"sudo sysctl -w iogpu.wired_limit_mb={mb}   "
                     f"(raises the {_gb(wss)} Metal cap — the binding limit "
                     f"here, not your {ram/_GB:.0f} GB of RAM; resets on reboot)")
    if mode == "streaming":
        flags.append("--streaming   (weights exceed even the raised cap; "
                     "experts page from disk)")
        # Predict what the loader will choose, by calling the same function it
        # calls. Note `wss`, not `ceiling`: the loader sizes against the cap
        # this machine has right now, and a default must not assume the user
        # ran a sysctl they were never told to run. Getting that wrong is how
        # this line came to advertise 9.1 GB where the loader picked 5.89.
        budget_bytes = auto_cache_budget(wss or 0, fp["resident_bytes"],
                                         fp["expert_bytes"])
        peak_bytes = projected_peak_bytes(budget_bytes, fp["resident_bytes"])
        flags.append(f"--cache-budget-gb auto   (~{budget_bytes / _GB:.1f} GB "
                     f"here, peaking near {peak_bytes / _GB:.1f} GB)")
    if chosen != 2048:
        flags.append(f"--prefill-step-size {chosen}   (the default 2048 needs "
                     f"{_gb(prefill_workspace_bytes(cfg, context, 2048))} of "
                     f"transient workspace at this context)")
    if kv_bits:
        flags.append(f"--kv-bits {kv_bits}")
    elif kv_total > 1.0 * _GB:
        flags.append("--kv-bits 8   (KV is over 1 GB at this context; "
                     "halves it)")

    if mode == "no-fit" and is_moe:
        warnings.append("even streaming does not fit — the resident backbone "
                        "alone exceeds what this machine can wire")
    if mode == "resident" and ceiling and peak > ceiling * 0.93:
        warnings.append("tight: under ~7% headroom — a longer context or a "
                        "background app can still push it over")

    return {
        "schema": 1,
        "model": {
            "path": model_path,
            "source": "huggingface" if remote else "local",
            "name": (model_path if remote
                     else os.path.basename(os.path.abspath(model_path))),
            "type": cfg.get("model_type") or c.get("model_type"),
            "quantization": cfg.get("quantization", {}).get("mode"),
            "bits": cfg.get("quantization", {}).get("bits"),
            "group_size": cfg.get("quantization", {}).get("group_size"),
            "expert_down_bits": cfg.get("quantization", {}).get(
                "expert_down_bits"),
            "n_layers": n_layers,
            "n_experts": c.get("num_experts"),
            "experts_per_tok": c.get("num_experts_per_tok")
            or c.get("num_experts_per_token"),  # Kimi K3 spelling
            "is_moe": is_moe,
            **fp,
        },
        "machine": mach,
        "projection": {
            "context": context,
            "kv_bits": kv_bits,
            "kv_bytes_per_token": kv_pt,
            "kv_bytes": kv_total,
            "kv_note": kv_note,
            "prefill_step_size": chosen,
            "prefill_workspace_bytes": workspace,
            "runtime_reserve_bytes": _RESERVE_BYTES,
            "peak_bytes": peak,
            "ceiling_bytes": ceiling,
            "headroom_bytes": (ceiling - peak) if ceiling else None,
        },
        "verdict": {"mode": mode, "runnable": ok, "needs_wired_bump": needs_bump},
        "flags": flags,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _gb(b) -> str:
    return "?" if b is None else f"{b / _GB:.2f} GB"


def render(p: dict) -> str:
    m, mach, pr, v = p["model"], p["machine"], p["projection"], p["verdict"]
    L = []
    L.append(f"TurboQuant plan — {m['name']}")
    L.append("")
    L.append("Model")
    q = f"{m['quantization']} {m['bits']}-bit g{m['group_size']}" if m[
        "quantization"] else "unquantized"
    if m.get("expert_down_bits"):
        q += f", expert down_proj {m['expert_down_bits']}-bit"
    L.append(f"  {'type':<20} {m['type']}  ({q})")
    if m["is_moe"]:
        L.append(f"  {'MoE':<20} {m['n_experts']} experts, "
                 f"top-{m['experts_per_tok']}, {m['n_layers']} layers")
    L.append(f"  {'weights (exact)':<20} {_gb(m['total_bytes'])}")
    if m["is_moe"]:
        L.append(f"  {'  experts':<20} {_gb(m['expert_bytes'])}  (streamable)")
        L.append(f"  {'  resident':<20} {_gb(m['resident_bytes'])}  "
                 f"(attention, embeddings, routers)")
    L.append("")
    L.append("Machine" + ("  (assumed, not this one)" if mach["assumed"] else ""))
    L.append(f"  {'Metal working set':<20} {_gb(mach['wss_bytes'])}   "
             f"← the real ceiling")
    L.append(f"  {'system RAM':<20} {_gb(mach['ram_bytes'])}")
    L.append("")
    L.append(f"Projection at {pr['context']:,} tokens of context")
    L.append(f"  {'weights':<20} {_gb(m['total_bytes'])}")
    kvb = f"  ({pr['kv_bytes_per_token']/1024:.1f} KB/token, {pr['kv_note']})" \
        if pr["kv_bytes_per_token"] else f"  ({pr['kv_note']})"
    L.append(f"  {'KV cache':<20} {_gb(pr['kv_bytes'])}{kvb}")
    L.append(f"  {'prefill workspace':<20} {_gb(pr['prefill_workspace_bytes'])}"
             f"  (estimate, at --prefill-step-size {pr['prefill_step_size']})")
    L.append(f"  {'runtime reserve':<20} {_gb(pr['runtime_reserve_bytes'])}"
             f"  (buffer cache, activations, fragmentation)")
    L.append(f"  {'':<20} {'-' * 34}")
    head = ""
    if pr["headroom_bytes"] is not None:
        head = (f"   {_gb(pr['headroom_bytes'])} headroom"
                if pr["headroom_bytes"] > 0 else
                f"   OVER by {_gb(-pr['headroom_bytes'])}")
    L.append(f"  {'peak':<20} {_gb(pr['peak_bytes'])}"
             f" of {_gb(pr['ceiling_bytes'])} usable{head}")
    L.append("")
    icon = {"resident": "✅", "streaming": "⚠️ ", "no-fit": "❌",
            "unknown": "❔"}[v["mode"]]
    label = {"resident": "RESIDENT — fits fully in memory",
             "streaming": "STREAMING — too big to hold; experts page from disk",
             "no-fit": "WILL NOT RUN on this machine",
             "unknown": "UNKNOWN — could not read the Metal working set"}[v["mode"]]
    if v["mode"] == "resident" and v.get("needs_wired_bump"):
        label = "RESIDENT — fits, but only after raising the Metal wired cap"
        icon = "⚠️ "
    L.append(f"Verdict: {icon} {label}")
    for w in p["warnings"]:
        L.append(f"  ! {w}")
    if p["flags"]:
        L.append("")
        L.append("Recommended:")
        for f in p["flags"]:
            L.append(f"  {f}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# doctor — readiness, not just arithmetic
# --------------------------------------------------------------------------

def run_doctor(model_path: str, plan: dict) -> list:
    """[(check_id, status, message)] with stable ids for automation."""
    checks = []
    if plan["model"].get("source") == "huggingface":
        checks.append(("model.remote", "warn",
                       f"{model_path} planned from remote headers — not "
                       f"downloaded, so on-disk checks are skipped"))

    def add(cid, ok, msg, warn=False):
        checks.append((cid, "warn" if warn and not ok else
                       ("pass" if ok else "fail"), msg))

    local = plan["model"].get("source") != "huggingface"
    if local:
        add("model.dir", os.path.isdir(model_path),
            f"model directory {model_path}")
    if local:
        add("model.config",
            os.path.isfile(os.path.join(model_path, "config.json")),
            "config.json present")
        shards = glob.glob(os.path.join(model_path, "*.safetensors"))
        add("model.weights", bool(shards), f"{len(shards)} safetensors shard(s)")
        tok = any(os.path.isfile(os.path.join(model_path, f)) for f in
                  ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
        add("model.tokenizer", tok, "tokenizer files present")

    q = plan["model"]["quantization"]
    add("model.turboquant", q == "turboquant",
        f"quantization: {q}" if q == "turboquant" else
        f"quantization block: {q or 'none'} — not a TurboQuant model "
        f"(a plain MLX model runs with mlx_lm, not turboquant)")

    try:
        import mlx.core as mx
        add("env.mlx", True, f"mlx {getattr(mx, '__version__', '?')}")
    except Exception as e:
        add("env.mlx", False, f"mlx not importable: {e}")

    try:
        import mlx_lm
        add("env.mlx_lm", True, f"mlx-lm {getattr(mlx_lm, '__version__', '?')}")
    except Exception as e:
        add("env.mlx_lm", False, f"mlx-lm not importable: {e}")

    mach = plan["machine"]
    add("machine.wss", mach["wss_bytes"] is not None,
        f"Metal working set {_gb(mach['wss_bytes'])}")

    v = plan["verdict"]["mode"]
    pk = _gb(plan["projection"]["peak_bytes"])
    if plan["verdict"].get("needs_wired_bump"):
        msg = (f"placement: {v} — peak {pk} exceeds the current "
               f"{_gb(mach['wss_bytes'])} Metal cap; needs the sysctl raise below")
        add("fit.projection", False, msg, warn=True)
    else:
        add("fit.projection", plan["verdict"]["runnable"],
            f"placement: {v} — peak {pk} vs working set {_gb(mach['wss_bytes'])}")

    if local and plan["model"]["is_moe"] and v == "streaming":
        add("fit.hotlist",
            os.path.isfile(os.path.join(model_path, "hot_experts.json")),
            "hot_experts.json (warm-starts the streaming cache)", warn=True)
    return checks


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def is_complete_checkpoint(path: str) -> bool:
    """Does this directory hold every weight shard, not just some of them?

    Two ways a directory can look like a model and not be one:

    1. It has no shards at all. Planning a repo id caches `config.json`, and
       that alone creates a snapshot dir — so `turboquant-plan --model <repo>`
       used to succeed once and then fail forever, because the second run took
       its own cache droppings for a downloaded model.
    2. It has *some* shards, from a download that was interrupted. This is the
       dangerous one: the weight total is summed from whatever headers are
       present, so a half-downloaded 30 GB model would report ~15 GB and a
       confident "RESIDENT" verdict — precisely the false green light this
       tool exists to prevent.

    The index's weight_map names every shard the checkpoint needs, so when it's
    there, use it. Without one, a single-shard checkpoint is all-or-nothing.
    """
    if not os.path.isdir(path):
        return False
    index_file = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_file):
        # The index comes from an arbitrary repo id — untrusted input. Every
        # shape below is a real observed failure: a top-level list/str/int has
        # no .get, a non-dict weight_map has no .values, an unhashable value
        # breaks set(), and a non-str value breaks os.path.join. An index we
        # cannot read is an index we cannot verify against, so it does not get
        # to vouch for the checkpoint.
        try:
            with open(index_file) as f:
                weight_map = json.load(f).get("weight_map") or {}
            shards = set(weight_map.values())
            if shards:
                return all(os.path.isfile(os.path.join(path, s)) for s in shards)
        except (ValueError, OSError, AttributeError, TypeError):
            return False
    return bool(glob.glob(os.path.join(path, "*.safetensors")))


def _resolve(path: str) -> tuple:
    """(target, remote). A repo id is planned over the network from its headers —
    never by downloading it, which would defeat the entire point of the tool.
    A local cache hit is used only once it proves it holds the whole checkpoint;
    a partial one falls back to the network rather than misreporting the size."""
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        return p, False                     # an explicit local path is the user's word
    # Something that is clearly meant as a path, and isn't one, is a typo — not
    # a repo id. Handing it to the hub gets you "Repo id must be in the form
    # 'repo_name' or 'namespace/repo_name'", which sends you looking for a
    # naming rule when the real problem is a directory that isn't there.
    if os.path.isabs(p) or path.startswith((".", "~")) or path.count("/") > 1:
        raise FileNotFoundError(
            f"no such directory: {p}\n"
            "(looks like a local path; an HF repo id would be "
            "'namespace/repo_name' with no leading '/', '.' or '~')")
    try:                                    # already downloaded? use the cache
        from huggingface_hub import snapshot_download
        local = snapshot_download(path, local_files_only=True)
    except Exception:
        return path, True                   # plan it remotely, headers only
    return (local, False) if is_complete_checkpoint(local) else (path, True)


def _common_args(ap):
    ap.add_argument("--model", required=True, help="local dir or HF repo id")
    ap.add_argument("--context", type=int, default=16384,
                    help="context length to plan for (default 16384)")
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="plan with TurboQuant KV quantization")
    ap.add_argument("--prefill-step-size", type=int, default=None,
                    help="force a chunk size instead of picking the largest safe one")
    ap.add_argument("--wired-gb", type=float, default=None,
                    help="assume this Metal working set (plan for another Mac)")
    ap.add_argument("--ram-gb", type=float, default=None,
                    help="assume this much system RAM")
    ap.add_argument("--json", action="store_true", help="machine-readable output")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="turboquant-plan",
        description="Will this model run on this Mac, and with what flags? "
                    "Reads only safetensors headers — no tensors are loaded.")
    _common_args(ap)
    args = ap.parse_args(argv)
    try:
        path, remote = _resolve(args.model)
        p = build_plan(path, context=args.context, kv_bits=args.kv_bits,
                       step=args.prefill_step_size, wired_gb=args.wired_gb,
                       ram_gb=args.ram_gb, remote=remote)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return _EXIT_UNFIT
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return _EXIT_USAGE
    print(json.dumps(p, indent=2) if args.json else render(p))
    return _EXIT_OK if p["verdict"]["runnable"] else _EXIT_UNFIT


def doctor_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="turboquant-doctor",
        description="Read-only readiness check: model files, environment, and "
                    "whether the projected placement is runnable.")
    _common_args(ap)
    args = ap.parse_args(argv)
    try:
        path, remote = _resolve(args.model)
        p = build_plan(path, context=args.context, kv_bits=args.kv_bits,
                       step=args.prefill_step_size, wired_gb=args.wired_gb,
                       ram_gb=args.ram_gb, remote=remote)
    except Exception as e:
        if args.json:
            print(json.dumps({"schema": 1, "checks": [
                ["model.readable", "fail", str(e)]]}, indent=2))
        else:
            print(f"✗ model.readable   {e}", file=sys.stderr)
        return _EXIT_UNFIT

    checks = run_doctor(path, p)
    if args.json:
        print(json.dumps({"schema": 1, "verdict": p["verdict"],
                          "checks": [list(c) for c in checks],
                          "flags": p["flags"]}, indent=2))
    else:
        print(f"turboquant-doctor — {p['model']['name']}\n")
        icon = {"pass": "✓", "warn": "!", "fail": "✗"}
        for cid, status, msg in checks:
            print(f"  {icon[status]} {cid:<20} {msg}")
        if p["flags"]:
            print("\nRecommended:")
            for f in p["flags"]:
                print(f"  {f}")
    return _EXIT_OK if not any(s == "fail" for _, s, _ in checks) else _EXIT_UNFIT


if __name__ == "__main__":
    sys.exit(main())
