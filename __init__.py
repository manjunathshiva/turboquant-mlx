"""TurboQuant-MLX: Extreme weight compression for MLX on Apple Silicon.

Adapts Google's TurboQuant (PolarQuant + QJL) technique for weight quantization,
achieving 3-bit quality matching 4-bit affine with no calibration data needed.
"""

__version__ = "0.22.0"

from turboquant_mlx.config import TurboQuantConfig

# Re-exported as the package's public surface: `from turboquant_mlx import
# TurboQuantConfig`. Listing it in __all__ makes that intent explicit — and
# stops static analysers reading the import as dead code.
__all__ = ["TurboQuantConfig", "__version__"]
