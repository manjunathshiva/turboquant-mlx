"""Repack a TurboQuant MoE checkpoint so co-activated experts are adjacent on
disk (#3 layout optimization).

Given a per-layer permutation ``order`` (from calibrate_experts.py, where
``order[p] = old_expert_id`` now stored at new position ``p``), this reorders,
for every MoE layer:

  * switch_mlp.{gate,up,down}_proj.{weight,scales}  — the expert weight stacks
  * mlp.gate.{weight,bias,e_score_correction_bias}  — the router output rows

by the SAME permutation. Because the router rows are permuted to match the
expert rows, this is a pure relabeling of experts: the model selects the same
expert for the same input and computes the identical result — only the on-disk
position of each expert changes. A token's selected experts then fall into
fewer, longer contiguous runs, which the streaming reader coalesces into fewer
larger reads.

Sharding, tensor names, and the safetensors index are preserved unchanged
(only tensor *values* move), so the result is a drop-in checkpoint.

Example:
  python -m turboquant_mlx.stream.repack_experts \
      --model .../qwen3.5-122b-tq3 --perm /tmp/perm_122b.json \
      --out /tmp/qwen3.5-122b-tq3-coact
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct

import mlx.core as mx

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _st_metadata(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__", None)


def _reorder_axis0_keys(name: str) -> bool:
    """True if this tensor's axis 0 is the expert axis and must be permuted."""
    if name.endswith(("gate_proj.weight", "gate_proj.scales",
                       "up_proj.weight", "up_proj.scales",
                       "down_proj.weight", "down_proj.scales")):
        return "switch_mlp" in name
    # router (NOT switch_mlp.*_proj): mlp.gate.weight / .bias / correction bias
    return name.endswith(("mlp.gate.weight", "mlp.gate.bias",
                          "mlp.gate.e_score_correction_bias"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Input model dir (local path).")
    ap.add_argument("--perm", required=True, help="perm.json from calibrate_experts.")
    ap.add_argument("--out", required=True, help="Output model dir.")
    args = ap.parse_args()

    from turboquant_mlx.generate import resolve_model_path
    src = str(resolve_model_path(args.model))
    os.makedirs(args.out, exist_ok=True)

    with open(args.perm) as f:
        perm = {int(k): v for k, v in json.load(f)["perm"].items()}
    perm_idx = {l: mx.array(order, dtype=mx.uint32) for l, order in perm.items()}

    shards = sorted(glob.glob(os.path.join(src, "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    # copy everything that is NOT a weight shard (config, tokenizer, index, ...)
    for f in os.listdir(src):
        if f.endswith(".safetensors"):
            continue
        s = os.path.join(src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, f))

    n_perm = 0
    for shard in shards:
        meta = _st_metadata(shard)
        tensors = mx.load(shard)
        out = {}
        for name, arr in tensors.items():
            m = _LAYER_RE.search(name)
            if m and _reorder_axis0_keys(name):
                layer = int(m.group(1))
                idx = perm_idx.get(layer)
                if idx is not None:
                    assert arr.shape[0] == idx.shape[0], (
                        f"{name}: axis0 {arr.shape[0]} != perm len {idx.shape[0]}")
                    arr = mx.take(arr, idx, axis=0)
                    n_perm += 1
            out[name] = arr
        mx.eval(*out.values())
        dst = os.path.join(args.out, os.path.basename(shard))
        mx.save_safetensors(dst, out, metadata=meta or {})
        print(f"  {os.path.basename(shard)}: {len(out)} tensors "
              f"({n_perm} permuted so far)", flush=True)

    print(f"[repack] done -> {args.out} ({n_perm} expert/router tensors reordered)")


if __name__ == "__main__":
    main()
