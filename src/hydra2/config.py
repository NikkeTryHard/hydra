"""Frozen WP-01 configuration: parity tolerances, forbidden packages,
artifact-root resolution.

Dependency pins are declared exactly once in ``pyproject.toml``
(``[tool.pixi.pypi-dependencies]`` / pixi feature) — this module never
restates versions; it references the lock and pyproject for verification.

Portability:
- ``artifact_root()`` is lazy and honors ``HYDRA2_ARTIFACT_ROOT`` > ``XDG_CACHE_HOME``
  per XDG Base Directory spec https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
  (``$XDG_CACHE_HOME`` defines cache base; default ``$HOME/.cache`` if unset/empty)
  > ``tempfile.gettempdir()`` per https://docs.python.org/3/library/tempfile.html
  which respects ``TMPDIR``/``TEMP``/``TMP`` via candidate list in
  https://github.com/python/cpython/blob/main/Lib/tempfile.py and documented in
  https://github.com/python/cpython/blob/main/Doc/library/tempfile.rst.
  Final fallback ``Path.home()/.cache`` per https://docs.python.org/3/library/pathlib.html#pathlib.Path.home.
- ``repo_root()`` walks parents for ``pyproject.toml`` or ``.git`` marker, cached.
"""

from __future__ import annotations

import os
import tempfile
from functools import cache, lru_cache
from pathlib import Path

# Frozen fp32 parity tolerance between plain-PyTorch and Fabric adapters.
PARITY_REL_TOL = 1e-5
PARITY_ABS_TOL = 1e-6

# Trainer meta-packages that MUST NOT be installed (lightning-fabric only).
TRAINER_FORBIDDEN_PACKAGES = ("lightning", "pytorch-lightning")

# MahJax is pinned by full Git SHA via pixi; no PyPI release matches it.
MAHJAX_GIT_URL = "https://github.com/nissymori/mahjax.git"
MAHJAX_PIN_SHA = "52228723901a4ace44b745afd25141acc25405ec"

__all__ = [
    "MAHJAX_GIT_URL",
    "MAHJAX_PIN_SHA",
    "PARITY_ABS_TOL",
    "PARITY_REL_TOL",
    "TRAINER_FORBIDDEN_PACKAGES",
    "artifact_root",
    "repo_root",
]


def _default_artifact_root() -> Path:
    """Portable default artifact root without hardcoded user paths.

    Resolution (lazy, no module-level constant):
    1. ``platformdirs.user_cache_dir("hydra2")`` if available (XDG on Linux,
       ``LOCALAPPDATA`` on Windows, ``Library/Caches`` on macOS) — cross-platform.
       Evidence: https://github.com/tox-dev/platformdirs + https://pypi.org/project/platformdirs/
       Evidence: https://specifications.freedesktop.org/basedir/latest/ + https://docs.python.org/3/library/pathlib.html#pathlib.Path.home
       Cons: adds dep weight (not in pixi.lock) — mitigated by fallback try/except so no hard dep.
    2. ``XDG_CACHE_HOME`` if set and non-empty -> ``$XDG_CACHE_HOME/hydra2/artifacts``
       (XDG spec https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
    3. ``tempfile.gettempdir()`` (respects ``TMPDIR`` per https://docs.python.org/3/library/tempfile.html
       and candidate list https://github.com/python/cpython/blob/main/Lib/tempfile.py)
       -> ``<tmp>/hydra2-artifacts``
    4. ``Path.home()/.cache/hydra2/artifacts`` fallback (XDG default).
    """
    # Prefer platformdirs when installed — respects XDG_CACHE_HOME on Linux already,
    # returns LOCALAPPDATA/ProgramData on Windows, Library/Caches on macOS.
    # Keep manual chain as fallback so missing platformdirs is not a hard failure.
    try:
        from platformdirs import user_cache_dir  # type: ignore[import-not-found]

        # ensure_exists=False keeps lazy (no mkdir here); .resolve() done by caller artifact_root().
        return Path(user_cache_dir("hydra2", ensure_exists=False)) / "artifacts"
    except (ImportError, OSError):
        pass
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg is not None:
        stripped = xdg.strip()
        if stripped != "":
            return Path(stripped) / "hydra2" / "artifacts"
    try:
        tmpdir = tempfile.gettempdir()
    except Exception:
        tmpdir = ""
    if tmpdir and tmpdir.strip() != "":
        return Path(tmpdir) / "hydra2-artifacts"
    return Path.home() / ".cache" / "hydra2" / "artifacts"


def artifact_root() -> Path:
    """Lazily resolve artifact root.

    Honors ``HYDRA2_ARTIFACT_ROOT`` if set and non-empty, else delegates to
    ``_default_artifact_root()``. No module-level hardcoded ``/home/...`` path
    is allocated; env is read on each call so late ``os.environ`` mutations
    (e.g., ``tests/conftest.py`` or ``pixi run`` wrappers) are respected.
    """
    value = os.environ.get("HYDRA2_ARTIFACT_ROOT")
    if value is not None:
        stripped = value.strip()
        if stripped != "":
            return Path(stripped).resolve()
    return _default_artifact_root().resolve()


@cache
def _find_repo_root(start: Path) -> Path:
    """Walk ``start`` and its parents for ``pyproject.toml`` or ``.git`` marker.

    Cached. Falls back to legacy ``parents[2]``-like heuristic only if no marker
    is found (e.g., installed wheel where ``src/hydra2/config.py`` depth differs).
    """
    cur = start.resolve()
    for cand in (cur, *cur.parents):
        if (cand / "pyproject.toml").is_file() or (cand / ".git").exists():
            return cand
    # Fallback for non-checkout installs: try historical depth but verify existence.
    try:
        legacy = Path(__file__).resolve().parents[2]
        if legacy.exists():
            return legacy
    except Exception:
        pass
    return cur


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return repository root via marker walk (pyproject.toml/.git), cached."""
    return _find_repo_root(Path(__file__).resolve())


# Back-compat lazy aliases via PEP 562 (module __getattr__).
# ``from hydra2.config import DEFAULT_ARTIFACT_ROOT`` remains importable but
# recomputes lazily; direct ``hydra2.config.DEFAULT_ARTIFACT_ROOT`` also lazy.
def __getattr__(name: str) -> Path:
    if name == "DEFAULT_ARTIFACT_ROOT":
        return _default_artifact_root().resolve()
    if name == "REPO_ROOT":
        return repo_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        [*__all__, "DEFAULT_ARTIFACT_ROOT", "REPO_ROOT", "_default_artifact_root", "_find_repo_root"]  # noqa: E501
    )
