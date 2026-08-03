# Copyright 2026 Manjunath Janardhan
"""MLX implementation of Moonshot AI's Kimi K3 (``model_type: "kimi_k3"``), text-only.

Ported from the ``transformers`` ``KimiK3ForConditionalGeneration`` reference
(``modeling_kimi_k3.py`` / ``modeling_kimi_linear.py``) and cross-checked against
the Moonshot-contributed vLLM K3 implementation. K3 is a 2.8T-parameter MoE
(93 layers, 896 experts, top-16) whose text tower extends the Kimi Linear
architecture with several pieces this module reproduces faithfully:

* **Hybrid attention** — 69 KDA (Kimi Delta Attention) linear-attention layers
  and 24 MLA layers on a mostly-3:1 cadence (``linear_attn_config.kda_layers`` /
  ``full_attn_layers``, both **1-indexed**; the cadence breaks at the tail —
  layers 92 and 93 are consecutive MLA).
* **Bounded KDA gate** — with ``gate_lower_bound`` set (K3: -5.0) the log-decay is
  ``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`` — a *replacement* for the
  standard ``-exp(A_log) * softplus(...)`` form, bounded to ``(lower_bound, 0)``.
  The checkpoint stores ``A_log`` zero-padded to ``head_dim`` (128); only the
  first ``num_heads`` (96) entries are real — :meth:`Model.sanitize` slices them.
* **Full-rank KDA output gate** — ``g_proj: hidden -> heads*head_dim``
  (``use_full_rank_gate``) instead of Kimi Linear 48B's low-rank pair.
* **Gated NoPE MLA with q-LoRA** — ``q_a_proj -> q_a_layernorm -> q_b_proj``,
  no positional encoding anywhere (position comes from the KDA layers), and a
  sigmoid output gate ``g_proj`` applied between the head-merge and ``o_proj``.
* **Stable LatentMoE** — routed experts live in a ``routed_expert_hidden_size``
  (3584) latent space behind shared per-layer ``routed_expert_down_proj`` /
  ``routed_expert_up_proj`` projections; the top-k weighted sum happens in latent
  space, then a shared RMSNorm (``latent_moe_use_norm``), then the up-projection.
  The router reads the full 7168-dim hidden state, NOT the latent.
* **``situ`` activation** — ``[beta*tanh(g/beta)*sigmoid(g)] * [lb*tanh(u/lb)]``
  computed in float32 (K3: beta=4.0, linear beta=25.0), used by the dense MLP,
  the shared experts, and every routed expert.
* **Attention Residuals (AttnRes)** — every sublayer input is a softmax-weighted
  mix of the current residual stream and snapshots stashed at each
  ``attn_res_block_size`` (12) boundary, where the stream also resets. The mixing
  scores come from RMS-normalized candidates dotted with the fused product
  ``norm.weight * proj.weight``, but the returned mixture is over the
  *un-normalized* candidates. Recomputed per token — never cache state.

The checkpoint stores routed experts *per expert* in compressed-tensors
mxfp4 (``experts.E.w{1,3,2}.weight_packed``/``weight_scale``); ``sanitize``
stacks them into ``SwitchGLU``'s ``(E, out, in)`` layout, views the packed
uint8 as uint32 (bit-identical to MLX's ``mode="mxfp4"``; verified against
compressed-tensors), renames ``block_sparse_moe`` -> ``mlp`` and
``w1/w3/w2`` -> ``gate/up/down_proj``, and drops the vision tower.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

import mlx_lm.models.base as base
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 33792
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    rms_norm_eps: float = 1e-5
    hidden_act: str = "situ"
    linear_attn_config: Dict[str, Any] = field(default_factory=dict)
    tie_word_embeddings: bool = False
    # MLA
    q_lora_rank: Optional[int] = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    # MoE
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    moe_intermediate_size: int = 3072
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    num_expert_group: int = 1
    topk_group: int = 1
    routed_expert_hidden_size: Optional[int] = 3584
    latent_moe_use_norm: bool = True
    # K3 extras
    attn_res_block_size: Optional[int] = 12
    activation_situ_beta: Optional[float] = 4.0
    activation_situ_linear_beta: Optional[float] = 25.0


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "kimi_k3"
    text_config: Dict[str, Any] = field(default_factory=dict)
    vision_config: Optional[Dict[str, Any]] = None


def situ_mul(gate: mx.array, up: mx.array, beta: float,
             linear_beta: Optional[float]) -> mx.array:
    """K3's SwiGLU variant with tanh soft-clipping on both legs, in float32."""
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * mx.tanh(u / linear_beta)
    return (a * u).astype(gate.dtype)


class Situ(nn.Module):
    """SwitchGLU-compatible activation module: called as ``activation(x_up, x_gate)``."""

    def __init__(self, beta: float, linear_beta: Optional[float]):
        super().__init__()
        self._beta = beta
        self._linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return situ_mul(gate, x, self._beta, self._linear_beta)


@partial(mx.compile, shapeless=True)
def _compute_g_softplus(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def _compute_g_bounded(A_log, a, dt_bias, lower_bound):
    # K3 safe gate: log-decay = lower_bound * sigmoid(exp(A_log) * (a + dt_bias)),
    # bounded to (lower_bound, 0) — NOT a clamp of the softplus form.
    return mx.exp(
        lower_bound * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a + dt_bias))
    )


class KimiMLP(nn.Module):
    def __init__(self, args: TextArgs, hidden_size: Optional[int] = None,
                 intermediate_size: Optional[int] = None):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self._beta = args.activation_situ_beta or 1.0
        self._linear_beta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            situ_mul(self.gate_proj(x), self.up_proj(x), self._beta, self._linear_beta)
        )


# Ported from mlx_lm.models.kimi_linear (MIT); selection on biased scores,
# weights gathered from the raw sigmoid scores (the noaux_tc trick).
@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class Router(nn.Module):
    """MoE router. A plain module (not nn.Linear) so the quantizers skip it, with
    the correction bias kept under ``mlp.gate.*`` where the streaming repack's
    router whitelist expects it."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size))
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # fp32 router matmul, as in the reference.
        return mx.matmul(x.astype(mx.float32), self.weight.astype(mx.float32).T)


class LatentMoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        latent = args.routed_expert_hidden_size or hidden

        # Mutable so the streaming loader's K-reduction lever
        # (_cap_active_experts) can lower it.
        self.top_k = args.num_experts_per_token
        self.gate = Router(hidden, args.num_experts)
        self.switch_mlp = SwitchGLU(
            latent,
            args.moe_intermediate_size,
            args.num_experts,
            activation=Situ(args.activation_situ_beta or 1.0,
                            args.activation_situ_linear_beta),
        )

        self.use_latent = args.routed_expert_hidden_size is not None
        if self.use_latent:
            self.routed_expert_down_proj = nn.Linear(hidden, latent, bias=False)
            self.routed_expert_up_proj = nn.Linear(latent, hidden, bias=False)
            if args.latent_moe_use_norm:
                self.routed_expert_norm = nn.RMSNorm(latent, eps=args.rms_norm_eps)
            else:
                self.routed_expert_norm = None

        if args.num_shared_experts:
            self.shared_experts = KimiMLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        # Router reads the full hidden state, BEFORE the latent down-projection.
        inds, weights = _group_expert_select(
            self.gate(x),
            self.gate.e_score_correction_bias,
            self.top_k,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )

        z = self.routed_expert_down_proj(x) if self.use_latent else x
        y = self.switch_mlp(z, inds)
        # Top-k weighted sum in latent space, then norm once, then up-project.
        y = (y * weights[..., None].astype(y.dtype)).sum(axis=-2)
        if self.use_latent:
            if self.routed_expert_norm is not None:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input))
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return out, new_state


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)
        self.gate_lower_bound = cfg.get("gate_lower_bound", None)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        # Checkpoint value is zero-padded to head_dim; sanitize slices to (num_heads,).
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = k_state = v_state = ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state, k_state, v_state = s, s, s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        # True L2-norm of q/k (fla's use_qk_l2norm_in_kernel: eps=1e-6 on the
        # SUM of squares, not the mean), with the readout scale folded into q.
        q = q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6) * self.scale
        k = k * mx.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        beta = mx.sigmoid(self.b_proj(x).reshape(B, T, self.num_heads))

        A_log = self.A_log.reshape(self.num_heads, 1)
        dt_bias = self.dt_bias.reshape(self.num_heads, self.head_dim)
        if self.gate_lower_bound is not None:
            g = _compute_g_bounded(A_log, a_logits, dt_bias, self.gate_lower_bound)
        else:
            g = _compute_g_softplus(A_log, a_logits, dt_bias)

        if ssm_state is None:
            ssm_state = mx.zeros(
                (B, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )

        use_kernel = (
            not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )
        if use_kernel:
            out, ssm_state = gated_delta_kernel(q, k, v, g, beta, ssm_state, mask)
        else:
            out, ssm_state = gated_delta_ops(q, k, v, g, beta, ssm_state, mask)

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, T, self.num_heads, self.head_dim)

        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


class KimiMLAAttention(nn.Module):
    """DeepSeek-style MLA, fully NoPE, with q-LoRA and a sigmoid output gate."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.use_output_gate = args.mla_use_output_gate
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        if args.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.args.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # NoPE: q_pe/k_pe carry no rotation; their scores fold into the mask.
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = base.scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


def _apply_attn_res(
    h: mx.array,
    block_residual: mx.array,
    proj_weight: mx.array,
    norm_weight: mx.array,
    eps: float,
) -> mx.array:
    """Softmax-mix the current stream with the block-boundary snapshots.

    Scores: RMS-normalized (weightless) candidates dotted with the fused
    ``norm.weight * proj.weight`` vector. The mixture is over the RAW
    (un-normalized) candidates — mixing the normalized ones is wrong.
    """
    v = mx.concatenate([block_residual, h[..., None, :]], axis=-2)  # (B,L,nb+1,H)
    vf = v.astype(mx.float32)
    k = vf * mx.rsqrt(mx.mean(vf * vf, axis=-1, keepdims=True) + eps)
    score_w = norm_weight.astype(mx.float32) * proj_weight.reshape(-1).astype(
        mx.float32
    )
    scores = (k * score_w).sum(axis=-1)  # (B,L,nb+1)
    probs = mx.softmax(scores, axis=-1, precise=True)
    out = (probs[..., None] * vf).sum(axis=-2)
    return out.astype(h.dtype)


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_res_block_size = args.attn_res_block_size
        self.rms_norm_eps = args.rms_norm_eps

        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = LatentMoE(args)
        else:
            self.mlp = KimiMLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        if self.attn_res_block_size is not None:
            hidden = args.hidden_size
            self.self_attention_res_norm = nn.RMSNorm(hidden, eps=args.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(hidden, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(hidden, eps=args.rms_norm_eps)
            self.mlp_res_proj = nn.Linear(hidden, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        block_residual: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        if self.attn_res_block_size is None:
            y = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + y
            return h + self.mlp(self.post_attention_layernorm(h)), block_residual

        prefix_sum = x
        if block_residual.shape[-2] > 0:
            h = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj.weight,
                self.self_attention_res_norm.weight,
                self.rms_norm_eps,
            )
        else:
            h = prefix_sum

        if self.layer_idx % self.attn_res_block_size == 0:
            # Stash the PRE-mix stream and reset it: the old stream survives
            # only inside block_residual from here on.
            block_residual = mx.concatenate(
                [block_residual, prefix_sum[..., None, :]], axis=-2
            )
            prefix_sum = None

        a = self.self_attn(self.input_layernorm(h), mask, cache)
        prefix_sum = a if prefix_sum is None else prefix_sum + a

        h = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.rms_norm_eps,
        )
        f = self.mlp(self.post_attention_layernorm(h))
        return prefix_sum + f, block_residual


class KimiK3TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.attn_res_block_size is not None:
            self.output_attn_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1 if kda_layers else None
        self.attn_idx = None
        for i in range(args.num_hidden_layers):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = (
            create_ssm_mask(h, cache[self.ssm_idx]) if self.ssm_idx is not None
            else None
        )
        attn_mask = (
            create_attention_mask(h, cache[self.attn_idx], return_array=True)
            if self.attn_idx is not None
            else None
        )

        block_residual = None
        if self.args.attn_res_block_size is not None:
            B, L = h.shape[:2]
            block_residual = mx.zeros((B, L, 0, h.shape[-1]), dtype=h.dtype)

        for layer, layer_cache in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            h, block_residual = layer(
                h, block_residual=block_residual, mask=mask, cache=layer_cache
            )

        if self.args.attn_res_block_size is not None:
            h = _apply_attn_res(
                h,
                block_residual,
                self.output_attn_res_proj.weight,
                self.output_attn_res_norm.weight,
                self.args.rms_norm_eps,
            )
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = KimiK3TextModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text_args = TextArgs.from_dict(args.text_config or {})
        self.language_model = LanguageModel(self.text_args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        return self.language_model(inputs, cache)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        args = self.text_args
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(("vision_tower.", "mm_projector."))
            and ".mtp" not in k
        }

        if args.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"language_model.model.layers.{layer_idx}"

            if isinstance(layer.mlp, LatentMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    packed = f"{src_prefix}.experts.0.{src}.weight_packed"
                    plain = f"{src_prefix}.experts.0.{src}.weight"
                    if packed in weights:
                        # compressed-tensors mxfp4: uint8 nibbles view as uint32
                        # (bit-identical to mx.quantize mode="mxfp4"); E8M0
                        # uint8 scales pass through raw. No biases.
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                            [
                                weights.pop(
                                    f"{src_prefix}.experts.{e}.{src}.weight_packed"
                                ).view(mx.uint32)
                                for e in range(args.num_experts)
                            ]
                        )
                        weights[f"{dst_prefix}.switch_mlp.{dst}.scales"] = mx.stack(
                            [
                                weights.pop(
                                    f"{src_prefix}.experts.{e}.{src}.weight_scale"
                                )
                                for e in range(args.num_experts)
                            ]
                        )
                    elif plain in weights:
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                            [
                                weights.pop(f"{src_prefix}.experts.{e}.{src}.weight")
                                for e in range(args.num_experts)
                            ]
                        )

                for name in (
                    "shared_experts.gate_proj",
                    "shared_experts.up_proj",
                    "shared_experts.down_proj",
                    "routed_expert_down_proj",
                    "routed_expert_up_proj",
                    "routed_expert_norm",
                    "gate",
                ):
                    src_key = f"{src_prefix}.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.{name}.weight"] = weights.pop(src_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.gate.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

            attn = getattr(layer, "self_attn", None)
            attn_prefix = f"{prefix}.self_attn"
            if isinstance(attn, KimiDeltaAttention):
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                a_key = f"{attn_prefix}.A_log"
                if a_key in weights:
                    a = weights[a_key].reshape(-1)
                    if a.shape[0] != attn.num_heads:
                        # Stored zero-padded to head_dim; only the first
                        # num_heads entries are real (verified: tail is all 0).
                        a = a[: attn.num_heads]
                    weights[a_key] = a
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights and weights[dt_key].ndim > 1:
                    weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = args.qk_nope_head_dim
                v_head = args.v_head_dim
                head_dim = qk_nope + v_head
                num_heads = args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases", None)
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return False
            return True

        return predicate
