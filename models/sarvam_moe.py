# Copyright 2026 Manjunath Janardhan
"""MLX implementation of Sarvam AI's MoE architecture (``model_type: "sarvam_moe"``).

Ported from the ``transformers`` ``SarvamMoEForCausalLM`` reference shipped in the
checkpoint's own ``modeling_sarvam_moe.py``. The forward pass itself is ordinary —
GQA attention with QK-norm, then a sigmoid-routed MoE with a shared expert — but the
*checkpoint naming* is Megatron-style rather than Llama-style, so most of the work
happens in :meth:`Model.sanitize`:

* **Fused QKV** — the checkpoint stores one ``attention.query_key_value.weight``
  of shape ``(num_heads + 2 * num_kv_heads) * head_dim, hidden``. The reference
  reshapes to ``(..., n_heads_total, head_dim)`` and splits along the *head* axis,
  so the rows are contiguous blocks of Q, then K, then V, and a plain row split
  reproduces it exactly. We split it here into standard ``q_proj`` / ``k_proj`` /
  ``v_proj`` so downstream tooling sees conventional names.
* **``attention.dense`` -> ``self_attn.o_proj``**, ``query_layernorm`` /
  ``key_layernorm`` -> ``q_norm`` / ``k_norm``, ``model.word_embeddings`` ->
  ``model.embed_tokens``.
* **Per-expert tensors** — ``mlp.experts.7.gate_proj.weight`` (2D, one per expert)
  are stacked into ``SwitchGLU``'s ``(E, out, in)`` form.

Naming matters beyond aesthetics: :meth:`TurboQuantConfig.bits_for_path` resolves
``--attn-bits`` by looking for an ``self_attn`` / ``attention`` path segment, so a
non-standard attention container would make the flag a silent no-op (the bug that
bit us on Nemotron's ``mixer.*`` paths).

Routing follows DeepSeek-V3: sigmoid scores, ``expert_bias`` added for *selection
only*, top-k weights renormalized to sum 1, then scaled by ``routed_scaling_factor``.
``n_group``/``topk_group`` are both 1 in the released config, which makes the
reference's ``group_limited_topk`` degenerate to a plain top-k; we assert that rather
than reimplementing the grouped path.
"""

from dataclasses import dataclass
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
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
    rope_theta: float
    first_k_dense_replace: int = 1
    num_shared_experts: int = 1
    moe_shared_expert_intermediate_size: int = 0
    routed_scaling_factor: float = 1.0
    score_function: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    use_qk_norm: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.score_function != "sigmoid":
            raise ValueError(
                f"unsupported score_function {self.score_function!r} "
                "(only 'sigmoid' is implemented)"
            )
        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError(
                f"grouped expert routing (n_group={self.n_group}, "
                f"topk_group={self.topk_group}) is not implemented; the released "
                "sarvam_moe configs use 1/1, which is plain top-k"
            )
        if not self.moe_shared_expert_intermediate_size:
            # reference: moe_intermediate_size * num_shared_experts
            self.moe_shared_expert_intermediate_size = (
                self.moe_intermediate_size * self.num_shared_experts
            )


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.head_dim, bias=args.use_qkv_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.use_qkv_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_kv_heads * self.head_dim, bias=args.use_qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, args.hidden_size, bias=args.use_bias
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        # reference applies QK-norm per head *before* RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class Router(nn.Module):
    """Sigmoid router with a selection-only bias.

    Deliberately *not* an ``nn.Linear``: the quantizer walks the module tree for
    linear layers, and the router must stay in full precision (the checkpoint keeps
    it fp32 even in otherwise-bf16 mirrors, matching ``router_dtype: fp32``).
    """

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size))
        self.expert_bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class SarvamMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = Router(args.hidden_size, args.num_experts)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size, args.moe_shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(logits)

        # expert_bias steers *selection* only; the combine weights use raw scores
        selection = scores + self.gate.expert_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]

        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob and k > 1:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.route_scale).astype(x.dtype)

        y = self.experts(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SarvamMoE(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class SarvamMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SarvamMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Map Megatron-style checkpoint names onto this module's parameters.

        The fused ``query_key_value`` split is the load-bearing part. The reference
        views the projection output as ``(B, L, n_heads + 2 * n_kv, head_dim)`` and
        splits on the head axis, so along the weight's *row* axis the layout is
        ``[Q heads | K heads | V heads]`` in contiguous blocks — a plain row split at
        ``n_heads * head_dim`` and ``+ n_kv * head_dim`` is exact, not an approximation.
        """
        import re

        a = self.args
        q_rows = a.num_attention_heads * a.head_dim
        kv_rows = a.num_key_value_heads * a.head_dim

        expert_re = re.compile(
            r"^(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$"
        )
        per_expert: Dict[str, Dict[str, Dict[int, mx.array]]] = {}
        new: Dict[str, mx.array] = {}

        renames = {
            ".attention.dense.": ".self_attn.o_proj.",
            ".attention.query_layernorm.": ".self_attn.q_norm.",
            ".attention.key_layernorm.": ".self_attn.k_norm.",
        }

        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue

            m = expert_re.match(k)
            if m is not None:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                per_expert.setdefault(prefix, {}).setdefault(proj, {})[idx] = v
                continue

            if k.endswith(".attention.query_key_value.weight"):
                base = k[: -len("attention.query_key_value.weight")] + "self_attn."
                expected = q_rows + 2 * kv_rows
                if v.shape[0] != expected:
                    raise ValueError(
                        f"{k}: expected {expected} rows for fused QKV "
                        f"(Q {q_rows} + K {kv_rows} + V {kv_rows}), got {v.shape[0]}"
                    )
                new[base + "q_proj.weight"] = v[:q_rows]
                new[base + "k_proj.weight"] = v[q_rows : q_rows + kv_rows]
                new[base + "v_proj.weight"] = v[q_rows + kv_rows :]
                continue

            if k == "model.word_embeddings.weight":
                new["model.embed_tokens.weight"] = v
                continue

            for old, sub in renames.items():
                if old in k:
                    k = k.replace(old, sub)
                    break
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

    @property
    def cast_predicate(self):
        # the router is fp32 by design (`router_dtype: fp32`); keep it out of casts
        def predicate(k):
            return not (k.endswith("mlp.gate.weight") or k.endswith("expert_bias"))

        return predicate
