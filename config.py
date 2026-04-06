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
    """

    bits: int = 3
    group_size: int = 64
    use_qjl: bool = False
    rotation: str = "hadamard"
    rotation_seed: int = 42
    fuse_rotations: bool = False

    def __post_init__(self):
        if self.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.bits}")
        if self.group_size not in (32, 64, 128):
            raise ValueError(f"group_size must be 32, 64, or 128, got {self.group_size}")
        if self.rotation not in ("hadamard", "blockwise_hadamard", "none"):
            raise ValueError(f"rotation must be 'hadamard', 'blockwise_hadamard', or 'none', got {self.rotation}")

    def to_dict(self) -> dict:
        return {
            "mode": "turboquant",
            "bits": self.bits,
            "group_size": self.group_size,
            "use_qjl": self.use_qjl,
            "rotation": self.rotation,
            "rotation_seed": self.rotation_seed,
            "fuse_rotations": self.fuse_rotations,
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
