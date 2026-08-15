"""WikiText-2 perplexity for mlx-vlm checkpoints, TurboQuant or affine.

`turboquant_mlx.evaluate` loads through `mlx_lm.utils.load` and re-quantizes
in-process, so it cannot touch a VLM checkpoint or a pre-quantized one from the
Hub. This harness takes an already-quantized directory of either kind and
scores the same text through it.

What makes the comparison fair, and why each part matters:

* **Same tokenizer.** Both builds derive from Qwen/Qwen3.8-27B, so token
  boundaries are identical and the per-token NLL is directly comparable. The
  script asserts this rather than assuming it.
* **Same text, same chunking.** A fixed number of non-overlapping `seq_len`
  chunks taken from the front of the concatenated corpus, so both models see
  byte-identical inputs.
* **Teacher forcing, no sampling.** Perplexity is a property of the
  distribution, so nothing here depends on a sampler or a seed.
* **Loss over the whole corpus, not a mean of per-chunk means.** Chunks are
  equal length here, so the two agree — but summing NLL and dividing by token
  count is the definition, and stays right if chunking ever changes.

Usage:
    python -m turboquant_mlx.benchmarks.eval_vlm_perplexity \\
        ./Qwen3.8-27B-tq4-g64 mlx-community/Qwen3.8-27B-4bit --chunks 32
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def resolved_identity(spec):
    """A record of exactly which bytes were measured.

    Repo ids and dataset defaults are mutable — `mlx-community/X-4bit` can be
    re-uploaded and a published number would silently start describing
    different weights. For a Hub repo this pins the commit sha; for a local
    directory it records the path and the total size of its safetensors, which
    is enough to notice a swap.
    """
    from pathlib import Path as _P

    p = _P(spec)
    if p.is_dir():
        shards = sorted(p.glob("*.safetensors"))
        return {"spec": str(spec), "kind": "local",
                "safetensors_bytes": sum(f.stat().st_size for f in shards),
                "n_shards": len(shards)}
    try:
        from huggingface_hub import HfApi

        return {"spec": str(spec), "kind": "hub",
                "revision": HfApi().model_info(str(spec)).sha}
    except Exception as exc:  # offline, private, or not a repo id
        return {"spec": str(spec), "kind": "unresolved", "error": str(exc)[:120]}


def load_any(path):
    """Load a TurboQuant or a stock mlx-vlm checkpoint, returning (model, kind)."""
    from mlx_vlm.utils import get_model_path, load_config

    resolved = Path(get_model_path(path))
    cfg = load_config(resolved)
    quant = cfg.get("quantization") or {}

    if quant.get("mode") == "turboquant":
        from turboquant_mlx.integration.vlm import load_turboquant_vlm

        model, _processor, _cfg = load_turboquant_vlm(resolved)
        bits = quant.get("bits")
        gs = quant.get("group_size")
        extras = quant.get("affine_extras")
        kind = f"turboquant {bits}-bit g{gs}"
        if extras:
            kind += f" + {extras['bits']}-bit affine extras"
        return model, kind, resolved

    from mlx_vlm import load as vlm_load

    model, _processor = vlm_load(str(resolved))
    kind = f"{quant.get('mode', 'affine')} {quant.get('bits')}-bit g{quant.get('group_size')}"
    return model, kind, resolved


def corpus_ids(tokenizer, seq_len, n_chunks):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    need = seq_len * n_chunks
    # Tokenize a generous prefix; 6 chars/token is a safe overestimate of density.
    ids = tokenizer.encode(text[: need * 6])
    if len(ids) < need + 1:
        raise SystemExit(f"corpus too short: {len(ids)} tokens, need {need + 1}")
    return ids[: need + 1]


def perplexity(model, ids, seq_len, n_chunks, label):
    """Total NLL / total predicted tokens, teacher-forced."""
    total_nll, total_tok = 0.0, 0
    t0 = time.time()
    for i in range(n_chunks):
        chunk = ids[i * seq_len : (i + 1) * seq_len + 1]
        inp = mx.array(chunk[:-1])[None]
        tgt = mx.array(chunk[1:])[None]
        out = model.language_model(inputs=inp)
        logits = (out.logits if hasattr(out, "logits") else out).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, tgt, reduction="sum")
        mx.eval(nll)
        total_nll += float(nll.item())
        total_tok += tgt.size
        print(f"\r  {label}: chunk {i+1}/{n_chunks}  "
              f"running ppl {math.exp(total_nll/total_tok):7.4f}", end="", flush=True)
    dt = time.time() - t0
    print(f"\r  {label}: {n_chunks} chunks x {seq_len} tok  "
          f"ppl {math.exp(total_nll/total_tok):7.4f}  ({dt:.0f}s)        ")
    return math.exp(total_nll / total_tok), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="checkpoint dirs or HF repo ids")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    results, ref_ids = [], None
    for spec in args.models:
        print(f"\n=== {spec} ===")
        model, kind, resolved = load_any(spec)
        tok = AutoTokenizer.from_pretrained(str(resolved))
        ids = corpus_ids(tok, args.seq_len, args.chunks)

        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            raise SystemExit(
                f"{spec}: tokenization differs from the first model — the "
                "perplexities would not be comparable. Aborting rather than "
                "reporting a misleading number."
            )

        mx.reset_peak_memory()
        ppl, n = perplexity(model, ids, args.seq_len, args.chunks, kind)
        peak = mx.get_peak_memory() / 1024**3
        results.append({"model": spec, "kind": kind, "ppl": ppl,
                        "tokens": n, "peak_gib": peak,
                        "identity": resolved_identity(spec)})
        print(f"  peak {peak:.2f} GiB")
        del model
        import gc

        gc.collect()
        mx.clear_cache()

    print(f"\n{'model':<44}{'quantization':<34}{'PPL':>9}")
    print("-" * 88)
    for r in sorted(results, key=lambda r: r["ppl"]):
        print(f"{r['model']:<44}{r['kind']:<34}{r['ppl']:>9.4f}")
    if len(results) == 2:
        a, b = results
        d = (a["ppl"] - b["ppl"]) / b["ppl"] * 100
        print(f"\n{a['model']} vs {b['model']}: {d:+.2f}% PPL "
              f"({'better' if d < 0 else 'worse'})")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "corpus": "wikitext-2-raw-v1 test",
            "seq_len": args.seq_len,
            "chunks": args.chunks,
            "note": ("teacher-forced NLL over the whole corpus slice; "
                     "tokenization asserted identical across models"),
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
