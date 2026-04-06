"""Per-architecture rotation fusion configurations.

Defines which layers can have their Hadamard rotation fused into a
preceding normalization (LayerNorm/RMSNorm) layer, and which require
online rotation at inference time.

Fusion is possible when:
- The layer's input comes directly from a normalization layer (element-wise scale)
- There's no non-linearity between the norm and the projection

Online rotation is needed when:
- The layer's input passes through a non-linearity (SiLU, GeLU, etc.)
- The layer's input comes from attention (softmax + value multiply)
"""

from dataclasses import dataclass, field


@dataclass
class LayerRotationConfig:
    """Rotation configuration for a single transformer block.

    fuse_norm_to_projs: Maps norm layer name -> list of projection layer names
        whose rotation can be fused into that norm's weight.
    online_rotation_layers: Projection layer names that need online rotation
        (their inputs pass through non-linearities).
    """
    fuse_norm_to_projs: dict[str, list[str]] = field(default_factory=dict)
    online_rotation_layers: list[str] = field(default_factory=list)


# LLaMA-style architectures (LLaMA, Mistral, Qwen2, CodeLlama, Yi)
# Block structure:
#   h = x + attn(input_layernorm(x))
#   out = h + mlp(post_attention_layernorm(h))
# Where:
#   attn: q_proj, k_proj, v_proj -> attention -> o_proj
#   mlp: gate_proj, up_proj -> silu(gate) * up -> down_proj
LLAMA_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        # input_layernorm output feeds directly into QKV projections
        "input_layernorm": ["q_proj", "k_proj", "v_proj"],
        # post_attention_layernorm output feeds directly into gate/up projections
        "post_attention_layernorm": ["gate_proj", "up_proj"],
    },
    online_rotation_layers=[
        # o_proj input comes from attention (softmax @ V), non-linear
        "o_proj",
        # down_proj input comes from silu(gate) * up, non-linear
        "down_proj",
    ],
)

# Gemma uses the same structure as LLaMA but with different norm naming
GEMMA_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        "input_layernorm": ["q_proj", "k_proj", "v_proj"],
        "post_feedforward_layernorm": ["gate_proj", "up_proj"],
    },
    online_rotation_layers=["o_proj", "down_proj"],
)

# Phi-2/3 style (uses LayerNorm instead of RMSNorm, different MLP)
# Block structure:
#   attn_out = attn(input_layernorm(x))
#   mlp_out = mlp(input_layernorm(x))  # parallel attention + MLP
#   out = x + attn_out + mlp_out
PHI_PARALLEL_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        # Single norm feeds both attention and MLP
        "input_layernorm": ["q_proj", "k_proj", "v_proj", "fc1"],
    },
    online_rotation_layers=["o_proj", "fc2"],
)

# Phi-3/3.5 (sequential, like LLaMA)
PHI3_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        "input_layernorm": ["qkv_proj"],
        "post_attention_layernorm": ["gate_up_proj"],
    },
    online_rotation_layers=["o_proj", "down_proj"],
)

# Starcoder2 / GPT-style with post-LN
STARCODER2_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        "input_layernorm": ["q_proj", "k_proj", "v_proj"],
        "post_attention_layernorm": ["c_fc"],
    },
    online_rotation_layers=["o_proj", "c_proj"],
)


# MoE architectures with SwitchGLU experts
# Same block structure as LLaMA:
#   input_layernorm -> attn(q,k,v,o) -> post_attention_layernorm -> MoE(experts + router)
# Expert layers (gate_proj, up_proj, down_proj) inside SwitchGLU follow the same
# rotation logic as dense MLP: gate/up inputs come from norm, down input is non-linear.
MOE_LLAMA_CONFIG = LayerRotationConfig(
    fuse_norm_to_projs={
        "input_layernorm": ["q_proj", "k_proj", "v_proj"],
        "post_attention_layernorm": ["gate_proj", "up_proj"],
    },
    online_rotation_layers=["o_proj", "down_proj"],
)


# Registry mapping architecture names to rotation configs
ROTATION_CONFIGS: dict[str, LayerRotationConfig] = {
    "llama": LLAMA_CONFIG,
    "mistral": LLAMA_CONFIG,
    "qwen2": LLAMA_CONFIG,
    "qwen3": LLAMA_CONFIG,
    "yi": LLAMA_CONFIG,
    "codellama": LLAMA_CONFIG,
    "internlm2": LLAMA_CONFIG,
    "gemma": GEMMA_CONFIG,
    "gemma2": GEMMA_CONFIG,
    "gemma3": GEMMA_CONFIG,
    "phi": PHI_PARALLEL_CONFIG,
    "phi3": PHI3_CONFIG,
    "phi3small": PHI3_CONFIG,
    "starcoder2": STARCODER2_CONFIG,
    # Qwen3.5 (hybrid attention: GatedDeltaNet + standard attention, same norm structure)
    "qwen3_5": LLAMA_CONFIG,
    # MoE architectures
    "qwen2_moe": MOE_LLAMA_CONFIG,
    "qwen3_5_moe": MOE_LLAMA_CONFIG,
    "gpt_oss": MOE_LLAMA_CONFIG,
}


def get_rotation_config(arch: str) -> LayerRotationConfig:
    """Get rotation fusion config for a model architecture.

    Args:
        arch: Architecture name (e.g., "llama", "mistral", "gemma").

    Returns:
        LayerRotationConfig for the architecture.

    Raises:
        ValueError: If architecture is not supported.
    """
    arch_lower = arch.lower()
    if arch_lower not in ROTATION_CONFIGS:
        supported = ", ".join(sorted(ROTATION_CONFIGS.keys()))
        raise ValueError(
            f"Unsupported architecture '{arch}'. Supported: {supported}. "
            f"Use rotation='none' in TurboQuantConfig to skip rotation."
        )
    return ROTATION_CONFIGS[arch_lower]


def should_fuse_rotation(
    layer_path: str,
    config: LayerRotationConfig,
) -> tuple[bool, str | None]:
    """Determine if a layer's rotation should be fused or applied online.

    Args:
        layer_path: Dot-separated path like "layers.0.self_attn.q_proj".
        config: Rotation config for the architecture.

    Returns:
        (can_fuse, norm_name): If can_fuse is True, norm_name is the norm layer
        whose weight should absorb the rotation. If False, norm_name is None
        and online rotation is needed.
    """
    # Extract the projection name from the path
    proj_name = layer_path.split(".")[-1]

    # Check if this projection can be fused into a norm
    for norm_name, proj_list in config.fuse_norm_to_projs.items():
        if proj_name in proj_list:
            return True, norm_name

    # Check if this requires online rotation
    if proj_name in config.online_rotation_layers:
        return False, None

    # Unknown layer - default to online rotation for safety
    return False, None
