"""Candidate 6 Gumbel configs — frozen hyper-parameters and per-action stats.

Owns the frozen ``GumbelSearchConfig`` and ``PuctConfig`` hyper-parameters
(part of CandidateSpec.parameters) plus the ``_ActionStats`` four-seat
accumulator the search loop backs up into. The shared vocabulary lives in
:mod:`hydra2.search.gumbel_core`, the search loop in
:mod:`hydra2.search.gumbel_search`, and the factories that freeze these
configs into CandidateSpec parameters in :mod:`hydra2.search.gumbel_spec`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError
from hydra2.search.gumbel_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.gumbel_core import scalarize_vector as scalarize_vector

__all__ = [
    "GumbelSearchConfig",
    "PuctConfig",
]


# ---------------------------------------------------------------------------
# Frozen planner config — part of CandidateSpec.parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GumbelSearchConfig:
    """Frozen Candidate 6 hyper-parameters (part of CandidateSpec.parameters)."""

    halving_rounds: int = _bridge_search.GUMBEL_HALVING_ROUNDS
    visits_per_round: tuple[int, ...] = tuple(_bridge_search.GUMBEL_VISITS_PER_ROUND)  # pyrefly: ignore[unknown-argument-type] # untyped bridge constant
    max_depth: int = _bridge_search.GUMBEL_MAX_DEPTH
    max_model_calls: int | None = _bridge_search.GUMBEL_MAX_MODEL_CALLS
    max_transitions: int | None = _bridge_search.GUMBEL_MAX_TRANSITIONS
    tie_break: str = _bridge_search.GUMBEL_TIE_BREAK
    candidate_id: str = _bridge_search.GUMBEL_CANDIDATE_ID
    resource_view: str = _bridge_search.GUMBEL_RESOURCE_VIEW
    seed_material: bytes = _MASTER_SEED
    puct_c: float | None = None  # reserved, not used in gumbel core

    def __post_init__(self) -> None:
        if (
            not isinstance(self.halving_rounds, int)
            or isinstance(self.halving_rounds, bool)
            or self.halving_rounds <= 0
            or self.halving_rounds > 5
        ):
            raise ContractError(f"halving_rounds must be int 1..5, got {self.halving_rounds!r}")
        if (
            not isinstance(self.visits_per_round, tuple)
            or len(self.visits_per_round) != self.halving_rounds
        ):
            raise ContractError(
                f"visits_per_round must be tuple length halving_rounds ({self.halving_rounds}), got {self.visits_per_round!r}"
            )
        for idx, v in enumerate(self.visits_per_round):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v > 64:
                raise ContractError(f"visits_per_round[{idx}] must be 1..64 int, got {v!r}")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth <= 0
            or self.max_depth > 32
        ):
            raise ContractError(f"max_depth must be int 1..32, got {self.max_depth!r}")
        for name in ("max_model_calls", "max_transitions"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ContractError(f"{name} must be positive int or None, got {v!r}")
        if self.tie_break not in ("lowest_action_id", "stable_hash", "lexicographic"):
            raise ContractError(
                f"tie_break must be lowest_action_id/stable_hash/lexicographic, got {self.tie_break!r}"
            )
        if not isinstance(self.candidate_id, str) or self.candidate_id == "":
            raise ContractError("candidate_id must be non-empty str")
        if self.resource_view not in ("calls", "transitions", "joules"):
            raise ContractError("resource_view must be calls/transitions/joules")
        if not isinstance(self.seed_material, (bytes, bytearray)) or len(self.seed_material) == 0:
            raise ContractError("seed_material must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class PuctConfig:
    """Frozen PUCT baseline config for matched comparator."""

    puct_c: float = _bridge_search.PUCT_C
    max_depth: int = _bridge_search.PUCT_MAX_DEPTH
    max_model_calls: int | None = _bridge_search.PUCT_MAX_MODEL_CALLS
    max_transitions: int | None = _bridge_search.PUCT_MAX_TRANSITIONS
    num_simulations: int = _bridge_search.PUCT_NUM_SIMULATIONS
    tie_break: str = _bridge_search.PUCT_TIE_BREAK
    candidate_id: str = _bridge_search.PUCT_CANDIDATE_ID
    resource_view: str = _bridge_search.PUCT_RESOURCE_VIEW
    seed_material: bytes = _MASTER_SEED

    def __post_init__(self) -> None:
        if not isinstance(self.puct_c, float) or not math.isfinite(self.puct_c) or self.puct_c <= 0:
            raise ContractError(f"puct_c must be finite >0, got {self.puct_c!r}")
        if (
            not isinstance(self.max_depth, int)
            or isinstance(self.max_depth, bool)
            or self.max_depth <= 0
        ):
            raise ContractError(f"max_depth must be int 1..32, got {self.max_depth!r}")
        if (
            not isinstance(self.num_simulations, int)
            or isinstance(self.num_simulations, bool)
            or self.num_simulations <= 0
        ):
            raise ContractError(
                f"num_simulations must be positive int, got {self.num_simulations!r}"
            )
        for name in ("max_model_calls", "max_transitions"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ContractError(f"{name} must be positive int or None, got {v!r}")
        if self.tie_break not in ("lowest_action_id", "stable_hash", "lexicographic"):
            raise ContractError(f"tie_break must be ... got {self.tie_break!r}")


# ---------------------------------------------------------------------------
# Per-action statistics — vector backup
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ActionStats:
    visits: int = 0
    value_sum: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def mean_vector(self) -> tuple[float, float, float, float] | None:
        if self.visits == 0:
            return None
        return tuple(v / self.visits for v in self.value_sum)  # type: ignore[return]

    def scalar_mean(self, root_seat: int) -> float | None:
        mv = self.mean_vector()
        if mv is None:
            return None
        return scalarize_vector(mv, root_seat)
