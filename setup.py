"""Build script for the TurboQuant native Metal extension.

All package metadata lives in pyproject.toml. This file exists only to declare
the C++/Metal extension via mlx.extension.CMakeExtension, which still requires
the setuptools setup() call for compiled extensions.

Note: The native C++ extension is currently disabled due to ABI issues with MLX.
The Python kernels in turboquant_mlx/kernels/ are used instead.
"""

from setuptools import setup
from mlx.extension import CMakeBuild, CMakeExtension

# Check if nanobind is available - if not, skip C++ extension build
# The Python kernels will be used instead
try:
    import nanobind
    ext_modules = [CMakeExtension("turboquant_mlx._ext", sourcedir="csrc")]
    cmdclass = {"build_ext": CMakeBuild}
except ImportError:
    ext_modules = []
    cmdclass = {}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
