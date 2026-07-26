"""Where a MoE block keeps its stacked experts — the one naming rule.

Two modules have to agree on which safetensors keys hold *streamable* expert
weights: ``stream/loader.py`` (which pages them from disk) and ``plan.py``
(which projects how much RAM that saves). They drifted — both matched only
``switch_mlp``, so Laguna, whose experts live at ``mlp.experts``, reported zero
streamable bytes: the planner called a streamable model resident-only, and
``--cache-budget-gb auto`` sized itself from a resident figure that included
every expert. Keeping the rule in one stdlib-only module is what stops that
happening again; ``plan.py`` imports no model framework, so this must not
either.

The two spellings are not symmetric, and that asymmetry is the point:

* ``switch_mlp`` is ours (qwen3_5_moe, deepseek) and unambiguous anywhere in a
  key, so it matches unanchored — that also preserves any prefix layout that
  already worked before this module existed.
* ``experts`` is a plain mlx-lm ``SwitchGLU`` attribute (laguna, gpt-oss) and is
  a *substring of* ``shared_experts`` — a dense, always-on MLP that must stay
  resident. So it only matches anchored as ``.mlp.experts.``, which is exactly
  how ``loader.py`` splices the attribute back into a key
  (``{prefix}.{i}.mlp.{attr}.{proj}.weight``).
"""

# Attribute name of the stacked-expert container on an MoE block, by convention.
SWITCH_ATTRS = ("switch_mlp", "experts")

# Only these two are paged per-expert. A trit/codebook layer also ships
# `.codebook` (3 entries) and `.signs` (the rotation sign vector); both are tiny
# and held resident on the StreamingSwitchLinear, so counting them as streamable
# would overstate what streaming actually saves.
STREAMED_SUFFIXES = (".weight", ".scales")


def is_streamed_expert_key(key: str) -> bool:
    """True if this safetensors key is a per-expert tensor the swap pages in.

    Excludes ``shared_experts`` (dense, always resident) and the resident
    codebook/signs of a quantized expert layer.
    """
    if not key.endswith(STREAMED_SUFFIXES):
        return False
    return "switch_mlp" in key or ".mlp.experts." in key
