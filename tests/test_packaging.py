"""Every runtime subpackage is declared in pyproject's `packages` list.

The flat-package layout (the repo root *is* the importable `turboquant_mlx`
package, via a package-dir trick) means subpackages must be enumerated by hand
— setuptools cannot discover them. Adding a directory therefore silently does
nothing to the distribution until someone remembers this list.

That failed once, badly: 0.17.0 added `models/` for Poolside Laguna but not the
list entry, so the sdist shipped without it. `compat.py` imports
`turboquant_mlx.models.laguna` at import time with no guard, and eleven modules
import `compat` — so every console script (`turboquant-generate`,
`-convert`, `-serve`, `stream_generate`) raised ModuleNotFoundError on a fresh
pip install of 0.17.0 and 0.18.0. The test suite passed throughout, because
pytest runs from the source tree where `models/` is right there on disk.

This test reads the same source tree, so it cannot prove the *tarball* is
correct — CI's `sdist contents` step does that. It does catch the mistake at
its origin: a new directory with no list entry.
"""

import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Not shipped, deliberately: pytest suite and dev benchmarks. Keep this list
# tiny — anything else with an __init__.py is runtime code a user can import.
_NOT_SHIPPED = {"tests", "benchmarks"}


def _declared_packages() -> set:
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.10
        tomllib = pytest.importorskip("tomli", reason="needs tomllib or tomli")
    with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as f:
        return set(tomllib.load(f)["tool"]["setuptools"]["packages"])


def _on_disk_subpackages() -> set:
    """Directories that are importable subpackages (have an __init__.py)."""
    return {
        d for d in os.listdir(_ROOT)
        if os.path.isfile(os.path.join(_ROOT, d, "__init__.py"))
    }


def test_every_subpackage_on_disk_is_declared():
    declared = _declared_packages()
    missing = sorted(
        d for d in _on_disk_subpackages()
        if d not in _NOT_SHIPPED and f"turboquant_mlx.{d}" not in declared
    )
    assert not missing, (
        f"subpackage(s) {missing} exist on disk but are not in pyproject's "
        "[tool.setuptools] packages — they will be missing from the sdist and "
        "any import of them will fail on a pip install while passing here. "
        f"Add 'turboquant_mlx.{missing[0]}' to that list, or to _NOT_SHIPPED "
        "in this test if it is genuinely dev-only."
    )


def test_declared_packages_exist_on_disk():
    """The other direction: a stale entry breaks the build outright."""
    declared = _declared_packages() - {"turboquant_mlx"}
    on_disk = _on_disk_subpackages()
    stale = sorted(p for p in declared if p.split(".", 1)[1] not in on_disk)
    assert not stale, f"declared but not on disk: {stale}"


def test_models_is_packaged():
    """The specific regression, named so a bisect points straight at it."""
    assert "turboquant_mlx.models" in _declared_packages()
