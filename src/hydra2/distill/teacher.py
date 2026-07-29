"""WP-10 Candidate 7 Teacher Distillation — re-export (deduplicated, P-005).

Canonical implementation lives in ``hydra2.distillation.teacher``; this module
re-exports to eliminate drift and remove hardcoded home fallback.
Path now via ``hydra2.config.repo_root`` + importlib.resources fallback.
Evidence: importlib.resources https://docs.python.org/3/library/importlib.resources.html
Portable ``repo_root`` uses marker walk (pyproject.toml/.git), not ``parents[2]``.
Keep payload small; dedup comment.
"""

from __future__ import annotations

from hydra2.distillation import teacher as _teacher

__all__ = getattr(_teacher, "__all__", [])


def __getattr__(name: str) -> object:
    """Delegate re-export to canonical teacher (PEP 562, avoids star import)."""
    return getattr(_teacher, name)
