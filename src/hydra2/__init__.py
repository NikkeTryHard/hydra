"""Hydra2: locked runtime, canonical contracts, engines, and search.

WP-01 bootstrap scope: contracts/common primitives (SPEC 2.1 + 3) and the
runtime subpackage (SPEC 10). Later work packages add the remaining
subpackages; nothing here is a placeholder.
"""

from hydra2.config import (
    MAHJAX_GIT_URL,
    MAHJAX_PIN_SHA,
    PARITY_ABS_TOL,
    PARITY_REL_TOL,
    TRAINER_FORBIDDEN_PACKAGES,
    artifact_root,
    repo_root,
)

__all__ = [
    "MAHJAX_GIT_URL",
    "MAHJAX_PIN_SHA",
    "PARITY_ABS_TOL",
    "PARITY_REL_TOL",
    "TRAINER_FORBIDDEN_PACKAGES",
    "artifact_root",
    "repo_root",
]
