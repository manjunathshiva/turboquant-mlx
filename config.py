"""Configuration for TurboQuant weight quantization."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant weight quantization.

    Args:
        bits: Base quantization bits (2, 3, or 4). Default 3.
        group_size: Number of weights sharing a scale factor. Default 64.
        use_qjl: Enable QJL 1-bit residual correction (adds ~1 bit overhead). Default False.
        rotation: Rotation method - "hadamard", "blockwise_hadamard", or "none". Default "hadamard".
        rotation_seed: Seed for random rotation signs (deterministic). Default 42.
        fuse_rotations: Whether to fuse rotations into LayerNorm weights. Default False.
            Note: norm fusion is currently disabled by default because fusing a
            Hadamard rotation into a diagonal norm weight is not mathematically
            valid (H(diag(w) @ x) != diag(H@w) @ x). Online rotation is used
            instead, with negligible overhead (~0.3% FLOPs).
        attn_bits: Optional override for attention-block linears (q/k/v/o_proj).
            None falls back to ``bits``. Useful for hybrid configs that keep
            attention sharper than MLP/expert weights.
        mlp_bits: Optional override for MLP / MoE expert linears
            (gate/up/down_proj, experts.*). None falls back to ``bits``.
        ternary_experts: If True, MoE expert (SwitchLinear) weights are quantized
            to the ternary {-c, 0, +c} codebook (1.58-bit) instead of a Gaussian
            codebook. Stored in the 2-bit slot, so it rides the existing packing /
            kernel / on-disk format unchanged (no memory win yet -- a real ~1.58
            bpw trit packing is a later step). The zero level clears the 1-bit
            cardinality wall; attention stays at ``bits``. This is the data-free
            sub-2-bit expert tier (``tq2a-tqTe``).
    """

    bits: int = 3
    group_size: int = 64
    use_qjl: bool = False
    rotation: str = "hadamard"
    rotation_seed: int = 42
    fuse_rotations: bool = False
    attn_bits: Optional[int] = None
    mlp_bits: Optional[int] = None
    mlp_group_size: Optional[int] = None  # block-scale override for the MLP/expert tier
    ternary_experts: bool = False  # ternary (1.58-bit) MoE experts, stored 2-bit

    def __post_init__(self):
        if self.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.bits}")
        # The 1-bit (sign + block scale) tier is only valid as an *override* for
        # the highly-redundant MoE expert weights, never as the base width.
        if self.attn_bits is not None and self.attn_bits not in (1, 2, 3, 4):
            raise ValueError(f"attn_bits must be 1, 2, 3, or 4, got {self.attn_bits}")
        if self.mlp_bits is not None and self.mlp_bits not in (1, 2, 3, 4):
            raise ValueError(f"mlp_bits must be 1, 2, 3, or 4, got {self.mlp_bits}")
        if self.group_size not in (16, 32, 64, 128):
            raise ValueError(f"group_size must be 16, 32, 64, or 128, got {self.group_size}")
        if self.mlp_group_size is not None and self.mlp_group_size not in (16, 32, 64, 128):
            raise ValueError(
                f"mlp_group_size must be 16, 32, 64, or 128, got {self.mlp_group_size}")
        if self.rotation not in ("hadamard", "blockwise_hadamard", "none"):
            raise ValueError(f"rotation must be 'hadamard', 'blockwise_hadamard', or 'none', got {self.rotation}")
        if not isinstance(self.ternary_experts, bool):
            raise ValueError(f"ternary_experts must be a bool, got {self.ternary_experts!r}")

    def bits_for_path(self, path: str) -> int:
        """Resolve the bit-width for a layer based on its dotted path.

        Attention-block linears use ``attn_bits`` when set; MLP / MoE expert
        linears use ``mlp_bits`` when set; everything else (and either
        override left as None) falls back to ``bits``.
        """
        for p in path.split("."):
            if p in ("self_attn", "attention", "linear_attn"):
                return self.attn_bits if self.attn_bits is not None else self.bits
            if p in ("mlp", "feed_forward"):
                return self.mlp_bits if self.mlp_bits is not None else self.bits
        return self.bits

    def group_size_for_path(self, path: str) -> int:
        """Resolve the quantization group size (block-scale granularity).

        MLP / MoE expert linears use ``mlp_group_size`` when set so a sub-2-bit
        expert tier can carry a finer block scale than the attention tier;
        everything else uses ``group_size``.
        """
        if self.mlp_group_size is not None:
            for p in path.split("."):
                if p in ("mlp", "feed_forward"):
                    return self.mlp_group_size
        return self.group_size

    @property
    def is_hybrid(self) -> bool:
        eff_attn = self.attn_bits if self.attn_bits is not None else self.bits
        eff_mlp = self.mlp_bits if self.mlp_bits is not None else self.bits
        return eff_attn != eff_mlp

    def to_dict(self) -> dict:
        return {
            "mode": "turboquant",
            "bits": self.bits,
            "group_size": self.group_size,
            "use_qjl": self.use_qjl,
            "rotation": self.rotation,
            "rotation_seed": self.rotation_seed,
            "fuse_rotations": self.fuse_rotations,
            "attn_bits": self.attn_bits,
            "mlp_bits": self.mlp_bits,
            "mlp_group_size": self.mlp_group_size,
            "ternary_experts": self.ternary_experts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TurboQuantConfig":
        return cls(
            bits=d.get("bits", 3),
            group_size=d.get("group_size", 64),
            use_qjl=d.get("use_qjl", False),
            rotation=d.get("rotation", "hadamard"),
            rotation_seed=d.get("rotation_seed", 42),
            fuse_rotations=d.get("fuse_rotations", False),
            attn_bits=d.get("attn_bits", None),
            mlp_bits=d.get("mlp_bits", None),
            mlp_group_size=d.get("mlp_group_size", None),
            ternary_experts=d.get("ternary_experts", False),
        )

    @property
    def effective_bits(self) -> float:
        """Effective bits per weight including overhead."""
        # b bits for indices + 16 bits per group for scale
        bpw = self.bits + 16.0 / self.group_size
        if self.use_qjl:
            # +1 bit for sign + 16 bits per group for residual norm
            bpw += 1.0 + 16.0 / self.group_size
        return bpw
