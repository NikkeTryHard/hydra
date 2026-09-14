# ruff: noqa: F401  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — shared: seeds, feature flags, guarded fallbacks.

Single home for the deterministic master seed, the strategy-key firewall set, the
module logger, and the optional-dependency fallbacks (contracts, randomness, belief,
observation) shared by the ``local_*`` split. The common search contract stays
authoritative in :mod:`hydra2.search.common`; the fallbacks below only keep unit
import paths working where the full stack is absent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hydra2.search.common import (
    DEPLOYABLE_DEADLINE_MS,
    CandidateSpec,
    Planner,
    ResourceBudget,
    SearchRequest,
    SearchResult,
)

_COMMON_AVAILABLE = True  # common.py is the single authority, imported directly above

try:
    from hydra2.artifacts.canonical import canonical_bytes
    from hydra2.contracts.common import ContractError, DigestText, make_digest_text

    _HAS_CONTRACTS = True
except ImportError:  # pragma: no cover
    _HAS_CONTRACTS = False
    ContractError = RuntimeError  # type: ignore[no-redef]
    DigestText = str  # type: ignore[no-redef]

    def make_digest_text(v: str) -> str:  # type: ignore[no-redef]
        if not isinstance(v, str) or not v.startswith("sha256:") or len(v) != 71:
            raise RuntimeError(f"bad digest {v!r}")
        return v

    def canonical_bytes(v: Any) -> bytes:  # type: ignore[no-redef]
        import json as _json

        return _json.dumps(v, sort_keys=True, separators=(",", ":")).encode()


try:
    from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

    _HAS_RANDOM = True
except ImportError:  # pragma: no cover
    _HAS_RANDOM = False
    RandomStream = Any  # type: ignore[no-redef]

try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief

    _HAS_BELIEF = True
except ImportError:  # pragma: no cover
    _HAS_BELIEF = False
    BeliefEpoch = Any  # type: ignore[no-redef]
    NaturalBelief = Any  # type: ignore[no-redef]

try:
    from hydra2.contracts.observation import make_actor_observation

    _HAS_OBS = True
except ImportError:
    _HAS_OBS = False

logger = logging.getLogger(__name__)

_MASTER_SEED = b"wp09d_local_resolving_v1"
FORBIDDEN_IN_STRATEGY_KEY: frozenset[str] = frozenset({"world_id", "full_hidden", "privileged"})

__all__ = [
    "FORBIDDEN_IN_STRATEGY_KEY",
    "_COMMON_AVAILABLE",
    "_HAS_BELIEF",
    "_HAS_CONTRACTS",
    "_HAS_OBS",
    "_HAS_RANDOM",
    "_MASTER_SEED",
    "logger",
]
