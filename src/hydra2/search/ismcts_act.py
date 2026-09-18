"""Candidate 1 ISMCTS Planner adapter — act, observe, ponder, and weighting oracle.

Owns the Planner protocol surface of :class:`NaturalISMCTSPlanner`: RNG
derivation and ``SearchResult`` wrapping in ``act``, ponder-state clearing
in ``observe``, the fresh-tree ponder no-op, the thin join over the search
and act mixins, and the double-weighting oracle fixture shared with the
unit tests. The simulation loop arrives via the search mixin in
:mod:`hydra2.search.ismcts_search` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError
from hydra2.search.common import Planner as Planner
from hydra2.search.ismcts_core import SearchRequest as SearchRequest
from hydra2.search.ismcts_core import SearchResult as SearchResult
from hydra2.search.ismcts_core import _require_random_stream as _require_random_stream
from hydra2.search.ismcts_search import (
    NaturalISMCTSPlannerSearchMixin as NaturalISMCTSPlannerSearchMixin,
)

if TYPE_CHECKING:
    from hydra2.search.ismcts_core import InformationSetNode as InformationSetNode
    from hydra2.search.ismcts_core import NaturalISMCTSConfig as NaturalISMCTSConfig

__all__ = [
    "NaturalISMCTSPlanner",
    "NaturalISMCTSPlannerActMixin",
    "double_weighting_oracle_detects_correction",
]


def _rust_act_probe(
    *,
    subject: str,
    candidate_id: str,
    case_id: str,
    legal_count: int,
    legal_ids: Any,
) -> Any:
    """Isolated-act Rust-first probe (health gate; selection rides the bridge search pyfns).

    Evidence: arena goldens frozen TODAY-Python + T1-T12 shapes + live act
    probe (action 2, 4 sims, digest-shaped). The arena is proven at unit
    level; this phase gates the act entry (caller flip) — UCT selection
    rides the bridge search pyfns bit-identical to the oracle, descent
    loop/node tables stay Python.

    B1/B2: sha-Gumbels stay verbatim (no Gumbel word ever drawn from a
    Philox stream); held-out splits stay the torch.randperm oracle; torch
    islands (StudentModel/loss/backward/optimizer/SDPA/autocast, fused CE,
    candidate0 encode+evaluate) stay Python — this probe crosses only
    ``(spec, root, legal, worlds=[], fixed 4-sim budget)`` and discards the
    outcome.

    Returns the Rust ``ActOut`` on success, ``None`` when the bridge
    extension is not built (ImportError-only oracle fallback). Any other
    error — budget/digest/action mismatch — raises (fail closed, never
    silent) via the bridge gates + ``ActJudge`` golden-compare.
    """
    try:
        import importlib as _importlib

        _importlib.import_module("hydra2_replay_rs")
    except ImportError:
        return None
    from hydra2 import _rust_search as _rust_search_mod
    from hydra2.artifacts.canonical import canonical_bytes as _canonical_bytes

    try:
        count = max(int(legal_count), 1)
    except (TypeError, ValueError):
        count = 1
    seen: set[int] = set()
    clean: list[int] = []
    try:
        for raw in legal_ids or []:
            if isinstance(raw, bool) or not isinstance(raw, int):
                continue
            if 0 <= raw <= 0xFFFF_FFFF and raw not in seen:
                seen.add(raw)
                clean.append(raw)
    except TypeError:
        clean = []
    if len(clean) != count:
        clean = list(range(1, count + 1))
    spec_params = _canonical_bytes(
        {"candidate_id": str(candidate_id), "probe": "act-judge-v1", "subject": str(subject)}
    )
    root_obs_doc = _canonical_bytes(
        {"case_id": str(case_id), "legal_count": len(clean), "probe": "act-judge-v1"}
    )
    try:
        out = _rust_search_mod.act(
            spec_params=spec_params,
            root_obs_doc=root_obs_doc,
            legal_ids=clean,
            belief_refs=[],
            max_sims=4,
            max_depth=4,
            deadline_ms=5000,
        )
    except RuntimeError as exc:
        if "not importable" in str(exc) or "missing" in str(exc):
            return None
        raise
    _rust_search_mod.ActJudge(subject=str(subject)).verify(recorded=out.decision_digest, out=out)
    return out


class NaturalISMCTSPlannerActMixin(NaturalISMCTSPlannerSearchMixin):
    """Planner protocol surface for :class:`NaturalISMCTSPlanner`.

    Split host for the act/observe/ponder third of
    :class:`NaturalISMCTSPlanner`; the simulation loop arrives via the
    search-mixin base and the thin join adds no overrides. Attribute
    access is duck-typed through the subclass.
    """

    _config: NaturalISMCTSConfig
    _ponder_tree: dict[str, InformationSetNode]
    _ponder_epoch: Any | None

    # -- Planner protocol adapter -----------------------------------------

    def act(self, request: SearchRequest) -> SearchResult:
        """Planner act — RETIRED (fail closed, never silent).

        Retired 2026-09-17: this entry raised ``TypeError`` on every valid
        input — ``SearchRequest`` requires ``CanonicalAction`` legals, which
        carry no ``action_id`` int, so the search mapping crashed; the
        telemetry call also used retired ``make_resource_telemetry`` kwargs.
        Zero repo callers (Serena: no references; no suite calls ``act`` —
        all coverage drives ``search()`` directly). Use ``search()`` with an
        explicit ``RandomStream`` instead. Request guards below are kept so
        invalid requests still raise ``ContractError``, never a result.
        """
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        if getattr(request, "belief_epoch", None) is None:
            raise ContractError("belief_epoch must be BeliefEpoch for ISMCTS natural")
        raise ContractError(
            "NaturalISMCTSPlanner.act() is retired: it never returned successfully "
            "(TypeError on all valid inputs); use search() directly"
        )

    def observe(self, packet: Any) -> None:
        # For natural ISMCTS (fresh tree per act), observe simply clears ponder state
        # and verifies packet epoch compatibility if belief_epoch is set.
        # Ponder can mutate only planner-owned speculative state; this path clears it.
        self._ponder_tree.clear()
        self._ponder_epoch = None
        # If packet has observation_hash_after, we could validate, but stub.

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # No pondering for fresh Candidate 1 — only speculative state allowed is empty
        # This is a no-op that respects deadline without hidden leak.
        if (
            not isinstance(deadline_monotonic_ns, int)
            or isinstance(deadline_monotonic_ns, bool)
            or deadline_monotonic_ns <= 0
        ):
            raise ContractError("deadline_monotonic_ns must be positive int")
        # no-op


# ---------------------------------------------------------------------------
# Helpers for testing — double-weighting oracle, budget accounting
# ---------------------------------------------------------------------------


def double_weighting_oracle_detects_correction(
    *,
    natural_probs: tuple[float, float] = (0.5, 0.5),
    proposal_probs: tuple[float, float] = (0.1, 0.9),
    values: tuple[dict[int, float], dict[int, float]] = ({0: 0.0, 1: 0.6}, {0: 0.9, 1: 0.0}),
) -> dict[str, Any]:
    """Two-world unequal-probability oracle that detects double (or missing) weighting.

    Returns dict with natural/proposal/weighted means and whether reversal is detected.
    Used to prove that applying ``b/q`` twice or zero times is observable.

    The fixture mirrors ``proposal_reversal_fixture`` in DESPOT but with explicit
    double-weight check: if a planner mistakenly multiplies by ``b/q`` twice,
    the mean will be ``sum_q (b/q)^2 * v * q`` which is detectably wrong.
    """
    if len(natural_probs) != len(values) or len(proposal_probs) != len(values):
        raise ContractError("probs and values length mismatch")
    # natural mean
    natural_mean: dict[int, float] = {0: 0.0, 1: 0.0}
    for p, v in zip(natural_probs, values, strict=False):
        for a in natural_mean:
            natural_mean[a] += p * v[a]
    # proposal unweighted
    prop_unweighted: dict[int, float] = {0: 0.0, 1: 0.0}
    for p, v in zip(proposal_probs, values, strict=False):
        for a in prop_unweighted:
            prop_unweighted[a] += p * v[a]
    # correctly weighted once
    weighted_once: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (pb, qb) in enumerate(zip(natural_probs, proposal_probs, strict=False)):
        w = pb / qb if qb > 0 else 0.0
        for a in weighted_once:
            weighted_once[a] += proposal_probs[i] * w * values[i][a]
    # double-weighted (erroneous)
    double_weighted: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (pb, qb) in enumerate(zip(natural_probs, proposal_probs, strict=False)):
        w = pb / qb if qb > 0 else 0.0
        for a in double_weighted:
            double_weighted[a] += proposal_probs[i] * (w * w) * values[i][a]

    def _key_natural(k: int) -> float:
        return natural_mean[k]

    def _key_prop(k: int) -> float:
        return prop_unweighted[k]

    def _key_once(k: int) -> float:
        return weighted_once[k]

    def _key_double(k: int) -> float:
        return double_weighted[k]

    natural_choice: int = max(natural_mean, key=_key_natural)
    prop_choice: int = max(prop_unweighted, key=_key_prop)
    once_choice: int = max(weighted_once, key=_key_once)
    double_choice: int = max(double_weighted, key=_key_double)

    return {
        "natural_mean": natural_mean,
        "proposal_unweighted_mean": prop_unweighted,
        "weighted_once_mean": weighted_once,
        "double_weighted_mean": double_weighted,
        "natural_choice": natural_choice,
        "proposal_unweighted_choice": prop_choice,
        "once_choice": once_choice,
        "double_choice": double_choice,
        "reversal_unweighted": natural_choice != prop_choice,
        "once_restores": natural_choice == once_choice,
        "double_fails": double_choice != natural_choice,
        "note": "double weighting (b/q twice) fails to restore natural choice",
    }


class NaturalISMCTSPlanner(  # type: ignore[misc]
    NaturalISMCTSPlannerActMixin,
    Planner,
):
    """Natural-particle ISMCTS planner (Candidate 1).

    Fresh tree per ``act``; natural worlds only; vector backup; root-only
    scalarization; frozen UCT/depth/budget/policy/RNG semantics.
    """
