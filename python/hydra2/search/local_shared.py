# ruff: noqa: F401  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — shared: seeds, fail-closed guards.

Single home for the deterministic master seed, the strategy-key firewall set, the
module logger, and the fail-closed dependency guards (contracts, randomness, belief,
observation) shared by the ``local_*`` split. The common search contract stays
authoritative in :mod:`hydra2.search.common`; missing dependencies raise ImportError
with a build-ext hint on use (never silent degrade).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
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
    from hydra2.contracts.common import ContractError, DigestText

    _CONTRACTS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    canonical_bytes = Any  # type: ignore[no-redef]  # placeholder; _require_contracts() raises on use
    ContractError = Any  # type: ignore[no-redef]
    DigestText = Any  # type: ignore[no-redef]
    _CONTRACTS_IMPORT_ERROR = exc


def _require_contracts() -> None:
    """Fail-closed contracts access (lazy ImportError with build-ext hint)."""
    if _CONTRACTS_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.common/artifacts not importable "
            f"({_CONTRACTS_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before local resolving search"
        ) from _CONTRACTS_IMPORT_ERROR


try:
    from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

    _RANDOM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    RandomStream = Any  # type: ignore[no-redef]  # placeholder; _require_random_stream() raises on use
    make_random_stream_key = Any
    semantic_seed = Any
    _RANDOM_IMPORT_ERROR = exc


def _require_random_stream() -> Any:
    """Fail-closed RNG access (lazy ImportError with build-ext hint)."""
    if _RANDOM_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.randomness not importable "
            f"({_RANDOM_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before local resolving search"
        ) from _RANDOM_IMPORT_ERROR
    return RandomStream


try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief

    _BELIEF_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    BeliefEpoch = Any  # type: ignore[no-redef]  # placeholder; _require_belief() raises on use
    NaturalBelief = Any  # type: ignore[no-redef]
    _BELIEF_IMPORT_ERROR = exc


def _require_belief() -> None:
    """Fail-closed belief access (lazy ImportError with build-ext hint)."""
    if _BELIEF_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief.natural not importable "
            f"({_BELIEF_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before local resolving search"
        ) from _BELIEF_IMPORT_ERROR


try:
    from hydra2.contracts.observation_actor import make_actor_observation

    _OBS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    make_actor_observation = Any  # placeholder; _require_obs() raises on use
    _OBS_IMPORT_ERROR = exc


def _require_obs() -> Any:
    """Fail-closed observation access (lazy ImportError with build-ext hint)."""
    if _OBS_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.observation not importable "
            f"({_OBS_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before local resolving search"
        ) from _OBS_IMPORT_ERROR
    return make_actor_observation


logger = logging.getLogger(__name__)

_MASTER_SEED: bytes = _bridge_search.LOCAL_RESOLVING_MASTER_SEED  # pyrefly: ignore[unknown-variable-type] # untyped bridge constant
FORBIDDEN_IN_STRATEGY_KEY: frozenset[str] = _bridge_search.FORBIDDEN_IN_STRATEGY_KEY  # pyrefly: ignore[unknown-variable-type] # untyped bridge constant


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(s: str) -> DigestText:
    return _bridge_contracts.make_digest_text("sha256:" + _h(s.encode()))


__all__ = [
    "FORBIDDEN_IN_STRATEGY_KEY",
    "_COMMON_AVAILABLE",
    "_MASTER_SEED",
    "logger",
]
