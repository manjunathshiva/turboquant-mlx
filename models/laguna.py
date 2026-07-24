# Copyright 2026 Manjunath Janardhan
"""MLX implementation of Poolside's Laguna architecture (``model_type: "laguna"``).

Ported from the ``transformers`` ``LagunaForCausalLM`` reference. The architecture
spans several non-standard pieces that this module reproduces faithfully so the
weights load bit-for-bit and the forward pass matches HF logits:

* **Hybrid attention** — a 3:1 schedule of sliding-window (512) and full-attention
  layers (``layer_types``), with a **per-layer query-head count**
  (``num_attention_heads_per_layer``: 48 on global, 64 on sliding) over a fixed
  GQA KV-head count.
* **Partial + YaRN RoPE** — global layers rotate the first ``head_dim * 0.5`` dims
  with YaRN scaling (``factor`` 64, ``attention_factor`` folded into ``mscale``);
  sliding layers rotate the full head with plain ``theta`` 10000.
* **QK-norm** — per-head RMSNorm on queries and keys before RoPE.
* **Per-head softplus attention gate** — ``g_proj: hidden -> num_heads`` produces a
  scalar gate per head, ``softplus``-activated, multiplied into the attention output.
* **Sigmoid-router MoE with a shared expert** — a leading dense MLP layer
  (``mlp_layer_types[0] == "dense"``) then MoE blocks: sigmoid routing scores,
  ``e_score_correction_bias`` added for selection only, top-k weights renormalized
  to sum 1, scaled by ``moe_routed_scaling_factor`` (2.5), plus an always-on shared
  expert.

The real checkpoint stores experts *per expert* (``experts.7.gate_proj.weight``);
:meth:`Model.sanitize` stacks them into ``SwitchGLU``'s ``(E, out, in)`` form (and
also handles the transformers in-memory packed ``experts.gate_up_proj`` layout the
logit-parity harness feeds it).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn

import mlx_lm.models.base as base
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import YarnRoPE
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    moe_routed_scaling_factor: float
    layer_types: List[str]
    mlp_layer_types: List[str]
    num_attention_heads_per_layer: List[int]
    rope_parameters: Dict[str, Any]
    sliding_window: int
    max_position_embeddings: int = 262144
    moe_router_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False


def _make_rope(head_dim: int, layer_type: str, rope_parameters: Dict[str, Any],
               max_pos: int):
    """Build the RoPE module for a layer type from Laguna's nested rope config."""
    params = rope_parameters[layer_type]
    partial = params.get("partial_rotary_factor", 1.0)
    dims = int(head_dim * partial)
    base = params.get("rope_theta", 10000.0)
    rope_type = params.get("rope_type", "default")

    if rope_type == "yarn":
        # mlx-lm's YarnRoPE reproduces HF YaRN exactly: its period-space freq blend
        # is algebraically identical to HF's inv_freq blend, and its
        # mscale = 0.1*ln(factor)+1 equals Laguna's attention_factor (RoPE is linear,
        # so scaling x vs. cos/sin is equivalent).
        return YarnRoPE(
            dims=dims,
            traditional=False,
            max_position_embeddings=max_pos,
            base=base,
            scaling_factor=params["factor"],
            original_max_position_embeddings=params.get(
                "original_max_position_embeddings", 4096
            ),
            beta_fast=params.get("beta_fast", 32),
            beta_slow=params.get("beta_slow", 1),
        )
    # plain RoPE (non-interleaved), full or partial dims
    return nn.RoPE(dims, traditional=False, base=base)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, num_heads: int, layer_type: str):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = layer_type == "sliding_attention"

        bias = args.attention_bias
        self.q_proj = nn.Linear(args.hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, args.hidden_size, bias=bias)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        # per-head scalar attention gate
        self.g_proj = nn.Linear(args.hidden_size, num_heads, bias=False)

        self.rope = _make_rope(
            self.head_dim, layer_type, args.rope_parameters,
            args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        # QK-norm over head_dim, then to (B, H, L, D)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = base.scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3)  # (B, L, H, D)

        # per-head softplus gate (computed in fp32 like HF)
        g = self.g_proj(x).astype(mx.float32)  # (B, L, H)
        gate = mx.logaddexp(mx.array(0.0, dtype=mx.float32), g).astype(out.dtype)
        out = out * gate[..., None]

        out = out.reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaRouter(nn.Module):
    """MoE router. Named to match the HF ``mlp.gate`` weight + correction bias."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size))
        self.e_score_correction_bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class LagunaMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.moe_routed_scaling_factor
        self.softcap = args.moe_router_logit_softcapping

        self.gate = LagunaRouter(args.hidden_size, args.num_experts)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x).astype(mx.float32)
        if self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        scores = mx.sigmoid(logits)

        # correction bias affects selection only, not the combine weights
        selection = scores + self.gate.e_score_correction_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]

        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        weights = (weights * self.route_scale).astype(x.dtype)

        y = self.experts(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        num_heads = args.num_attention_heads_per_layer[layer_idx]
        layer_type = args.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        self.self_attn = Attention(args, num_heads, layer_type)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = LagunaMoE(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # representative layer index per attention type, for mask construction
        self._full_idx = self.layer_types.index("full_attention")
        self._swa_idx = next(
            (i for i, t in enumerate(self.layer_types) if t == "sliding_attention"),
            None,
        )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self._full_idx])
        swa_mask = full_mask
        if self._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map on-disk / HF weight names onto this module's parameter names.

        Two expert layouts are normalized to ``SwitchGLU``'s stacked
        ``experts.{gate,up,down}_proj.weight`` (E, out, in):

        * **On-disk (real checkpoint):** experts are stored *per expert*, e.g.
          ``mlp.experts.7.gate_proj.weight`` (2D). These are stacked in expert
          order along a new leading axis.
        * **transformers in-memory:** experts come packed as
          ``mlp.experts.gate_up_proj`` (E, 2*inter, hidden) + ``experts.down_proj``
          (E, hidden, inter); split/relabel these. (Kept so the logit-parity
          harness, which copies the HF ``state_dict``, still loads.)

        Also relocates the router correction bias (on disk it sits under
        ``mlp.experts``; our router owns it at ``mlp.gate``) and the singular
        ``mlp.shared_expert`` -> our ``mlp.shared_experts``.
        """
        import re

        inter = self.args.moe_intermediate_size
        # collect per-expert tensors: prefix "...mlp.experts" -> proj -> {idx: w}
        per_expert: Dict[str, Dict[str, Dict[int, mx.array]]] = {}
        expert_re = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")

        new: Dict[str, mx.array] = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue

            m = expert_re.match(k)
            if m is not None:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                per_expert.setdefault(prefix, {}).setdefault(proj, {})[idx] = v
                continue

            if ".mlp.experts.e_score_correction_bias" in k:
                # router owns the correction bias in our module (mlp.gate)
                new[k.replace(".mlp.experts.e_score_correction_bias",
                              ".mlp.gate.e_score_correction_bias")] = v
                continue

            if ".mlp.shared_expert." in k:
                new[k.replace(".mlp.shared_expert.", ".mlp.shared_experts.")] = v
                continue

            if k.endswith("mlp.experts.gate_up_proj"):
                # transformers in-memory: (E, 2*inter, hidden) -> split
                base = k[: -len("gate_up_proj")]
                new[base + "gate_proj.weight"] = v[:, :inter, :]
                new[base + "up_proj.weight"] = v[:, inter:, :]
            elif k.endswith("mlp.experts.down_proj"):
                # transformers in-memory: (E, hidden, inter) already (out, in)
                new[k + ".weight"] = v
            else:
                new[k] = v

        # stack per-expert 2D weights -> (E, out, in) for SwitchGLU
        for prefix, projs in per_expert.items():
            for proj, by_idx in projs.items():
                stacked = mx.stack([by_idx[i] for i in range(len(by_idx))], axis=0)
                new[f"{prefix}.{proj}_proj.weight"] = stacked

        if self.args.tie_word_embeddings:
            new.pop("lm_head.weight", None)
        return new

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.args.sliding_window)
            if layer.is_sliding
            else KVCache()
            for layer in self.model.layers
        ]

    @property
    def cast_predicate(self):
        # keep the router correction bias out of low-precision casts
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
