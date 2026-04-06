"""TurboQuant-MLX: Extreme weight compression for MLX on Apple Silicon.

Adapts Google's TurboQuant (PolarQuant + QJL) technique for weight quantization,
achieving 3-bit quality matching 4-bit affine with no calibration data needed.
"""

__version__ = "0.1.0"

from turboquant_mlx.config import TurboQuantConfig
