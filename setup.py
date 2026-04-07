"""Build script for the TurboQuant native Metal extension.

All package metadata lives in pyproject.toml. This file exists only to declare
the C++/Metal extension via mlx.extension.CMakeExtension, which still requires
the setuptools setup() call for compiled extensions.
"""

from setuptools import setup
from mlx.extension import CMakeBuild, CMakeExtension

setup(
    ext_modules=[
        CMakeExtension("turboquant_mlx._ext", sourcedir="csrc"),
    ],
    cmdclass={"build_ext": CMakeBuild},
)
