"""Candidate profiles + cost-gated admission + Gumbel/PUCT accounting (SPEC 16.7 PR3).

Profiles are LABELED PRIORS, not measured capacities: the rows below promote to
capacities only via the RTX pilot fixture. The admission gate selects the largest
profile passing the synchronized cost gate on disjoint pilot states; nothing
fits -> Candidate 0 (always reachable, never an error, never a forced profile).
Gate selection from held-out win rates is prohibited.

Jobs identity (exact): round r holds M/2^r survivors x 4.2^r added jobs = 4M per
round over log2(M) rounds, so jobs = 4M.log2(M); exact transitions <= jobs x H.
The comparator below measures accounting only; wall-block confirmation
decides (wall block = atomic confirmation unit sharing one wall;
wall = undealt tile stock: live = drawable, dead = dora reserve).

Pure accounting (row table, jobs/transitions math, admission scan, Gumbel/PUCT
comparison) executes in ``hydra2._native.search``
(``crates/bridge/src/search_profiles.rs``); this module keeps the
``CandidateProfile`` dataclass, ``ContractError`` shaping, and thin delegates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from hydra2._native import search as _profiles_bridge  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError

__all__ = [
    "PROFILES",
    "CandidateProfile",
    "admit",
    "compare_gumbel_puct",
    "jobs_for",
    "transitions_bound",
]


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    """Frozen candidate profile (provisional prior until RTX pilot promotes it)."""

    name: str
    candidate_cap: int
    horizon: int
    carry_quota: int
    halving_rounds: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ContractError("profile name must be a non-empty str")
        mval: Any = self.candidate_cap
        if isinstance(mval, bool) or not isinstance(mval, int) or mval < 2:
            raise ContractError("candidate_cap must be an int >= 2")
        if mval & (mval - 1) != 0:
            raise ContractError("candidate_cap must be a power of two (halving schedule)")
        rounds = mval.bit_length() - 1
        if self.halving_rounds != rounds:
            raise ContractError(f"halving_rounds must equal log2(M) = {rounds} for M = {mval}")
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or not 1 <= self.horizon <= 16
        ):
            raise ContractError("horizon must be an int in 1..16")
        if (
            not isinstance(self.carry_quota, int)
            or isinstance(self.carry_quota, bool)
            or self.carry_quota <= 0
        ):
            raise ContractError("carry_quota must be a positive int")


def _leaf(name: str) -> Any:
    """Resolve one ``hydra2._native.search`` profile leaf (fail closed)."""
    try:
        return getattr(_profiles_bridge, name)
    except AttributeError as exc:
        raise ImportError(
            f"hydra2._native.search.{name} missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


#: Provisional profile rows. Pilot-promotion condition attached (see module
#: docstring); do not read these as measured capacities. Single source is the
#: bridge ``PROFILES_ROWS`` table; Python rebuilds validated dataclass rows.
_profile_rows: tuple[tuple[str, int, int, int, int], ...] = _leaf("PROFILES_ROWS")
PROFILES: tuple[CandidateProfile, ...] = tuple(
    CandidateProfile(
        name=row[0],
        candidate_cap=row[1],
        horizon=row[2],
        carry_quota=row[3],
        halving_rounds=row[4],
    )
    for row in _profile_rows
)


def jobs_for(profile: CandidateProfile) -> int:
    """Exact added rollout jobs: 4M.log2(M)."""
    if not isinstance(profile, CandidateProfile):
        raise ContractError("profile must be CandidateProfile")
    try:
        return _leaf("profiles_jobs_for")(profile.candidate_cap, profile.halving_rounds)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def transitions_bound(profile: CandidateProfile) -> int:
    """Exact upper bound on exact transitions: jobs x H."""
    if not isinstance(profile, CandidateProfile):
        raise ContractError("profile must be CandidateProfile")
    try:
        return _leaf("profiles_transitions_bound")(
            profile.candidate_cap, profile.halving_rounds, profile.horizon
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def admit(
    profiles: tuple[CandidateProfile, ...],
    *,
    deadline_ms: int,
    fallback_margin_ms: int,
    seconds_per_transition: float,
    max_transitions: int | None = None,
) -> CandidateProfile | Literal["candidate0"]:
    """Select the largest fitting profile, else Candidate 0.

    Fit is decided in the transitions view (exact bound) against the
    deadline margin and an optional transition cap. The model-call view needs
    per-call pilot measurements and is NOT decided here. Deterministic:
    profiles considered largest-M first.
    """
    if not isinstance(profiles, tuple) or len(profiles) == 0:
        raise ContractError("profiles must be a non-empty tuple")
    for entry in profiles:
        if not isinstance(entry, CandidateProfile):
            raise ContractError("profiles must contain CandidateProfile only")
    if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or deadline_ms <= 0:
        raise ContractError("deadline_ms must be a positive int")
    if (
        isinstance(fallback_margin_ms, bool)
        or not isinstance(fallback_margin_ms, int)
        or not 0 <= fallback_margin_ms < deadline_ms
    ):
        raise ContractError("fallback_margin_ms must satisfy 0 <= margin < deadline")
    if (
        not isinstance(seconds_per_transition, float)
        or not math.isfinite(seconds_per_transition)
        or seconds_per_transition <= 0
    ):
        raise ContractError("seconds_per_transition must be a positive finite float")
    if max_transitions is not None and (
        isinstance(max_transitions, bool)
        or not isinstance(max_transitions, int)
        or max_transitions <= 0
    ):
        raise ContractError("max_transitions must be a positive int or None")
    try:
        picked: int | None = _leaf("profiles_admit")(
            [entry.candidate_cap for entry in profiles],
            [entry.horizon for entry in profiles],
            [entry.halving_rounds for entry in profiles],
            deadline_ms,
            fallback_margin_ms,
            seconds_per_transition,
            max_transitions,
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    if picked is None:
        return "candidate0"
    return profiles[picked]  # pyrefly: ignore[unknown-argument-type] # bridge-picked index


def compare_gumbel_puct(
    *,
    n_actions: int,
    gumbel_visits: tuple[int, ...],
    puct_simulations: int,
) -> dict[str, object]:
    """Accounting-level Gumbel-vs-PUCT comparison (measures, never promotes).

    Gumbel jobs use the closed form sum_r ceil(n/2^r).v_r; PUCT jobs equal its
    simulation count. Both arms must share n_actions; costs are tallied, and any
    strength claim stays with wall-block confirmation.
    """
    if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions <= 0:
        raise ContractError("n_actions must be a positive int")
    if (
        not isinstance(gumbel_visits, tuple)
        or len(gumbel_visits) == 0
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in gumbel_visits)
    ):
        raise ContractError("gumbel_visits must be a non-empty tuple of positive ints")
    if (
        isinstance(puct_simulations, bool)
        or not isinstance(puct_simulations, int)
        or puct_simulations <= 0
    ):
        raise ContractError("puct_simulations must be a positive int")
    try:
        gumbel_jobs, puct_jobs, cheaper = _leaf("profiles_compare_gumbel_puct")(
            n_actions, list(gumbel_visits), puct_simulations
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return {
        "n_actions": n_actions,
        "gumbel_jobs": gumbel_jobs,
        "puct_jobs": puct_jobs,
        "cheaper": cheaper,
    }
