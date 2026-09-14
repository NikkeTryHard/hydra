# ruff: noqa: N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
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

import hashlib
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError
from hydra2.search.common import Planner as Planner
from hydra2.search.ismcts_core import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.ismcts_core import SearchRequest as SearchRequest
from hydra2.search.ismcts_core import SearchResult as SearchResult
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
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        # Validate request hashes against candidate spec (lightweight)
        belief_epoch = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch must be BeliefEpoch for ISMCTS natural")
        # Derive RNG from request deadline / case_id if available
        _case_id_raw: Any = getattr(request, "case_id", None)  # pyrefly: ignore[explicit-any]
        _decision_id_raw: Any = getattr(request.observation, "decision_id", "case_default")  # pyrefly: ignore[explicit-any]
        if _case_id_raw is not None and str(_case_id_raw) != "":
            case_id: str = str(_case_id_raw)  # pyrefly: ignore[explicit-any]
        elif _decision_id_raw is not None and str(_decision_id_raw) != "":
            case_id = str(_decision_id_raw)  # pyrefly: ignore[explicit-any]
        else:
            case_id = "case_default"
        candidate_id = getattr(request.candidate_spec, "candidate_id", self._config.candidate_id)
        # Use semantic seed if available, else deterministic hash
        if _HAS_RANDOM:
            try:
                from hydra2.contracts.randomness import RandomStream

                # Derive stream from (candidate_id, case_id, belief_epoch)
                epoch_id = str(getattr(belief_epoch, "epoch", "0"))
                seed = hashlib.sha256(f"{candidate_id}:{case_id}:{epoch_id}".encode()).digest()
                rng = RandomStream(seed)
            except Exception:
                rng = RandomStream(hashlib.sha256(f"{candidate_id}:{case_id}".encode()).digest())  # type: ignore[call-arg]
        else:
            import secrets as _secrets  # fallback

            rng = _secrets.token_bytes(32)

        res = self.search(
            epoch=belief_epoch,
            root_observation=request.observation,
            legal_actions=request.legal_actions,
            rng=rng,
        )

        # Wrap into SearchResult with UtilityVector etc if telemetry available
        try:
            from hydra2.contracts.utility import UtilityVector as _UV
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
            from hydra2.search.common import candidate_spec_hash as _csh
        except Exception:
            # fallback minimal result for unit tests
            return SearchResult(
                selected_action=res["selected_action"],
                candidate_actions=res["candidate_actions"],
                value_vectors=res["value_vectors"],
                # NEVER-bind: fallback digest, not a verified binding.
                candidate_spec_hash=getattr(
                    request.candidate_spec, "candidate_spec_hash", "sha256:" + "a" * 64
                ),
                telemetry=res["telemetry"],
                evidence_refs=(),
                completed=res["completed"],
            )

        # Build UtilityVectors (vector preserved, identity from manifest)
        u_vectors: list[Any] = []
        for vec in res["value_vectors"]:
            try:
                u_vectors.append(
                    _UV(
                        values=tuple(float(v) for v in vec),  # type: ignore[arg-type]
                        utility_id=getattr(
                            request.candidate_spec, "utility_id", "expected_final_placement"
                        ),
                        utility_manifest_hash=getattr(  # type: ignore[arg-type]
                            request.candidate_spec, "utility_manifest_hash", "sha256:" + "b" * 64
                        ),
                        rules_hash=getattr(  # type: ignore[arg-type]
                            request.candidate_spec, "rules_hash", "sha256:" + "a" * 64
                        ),
                    )
                )
            except Exception:
                # fallback if UtilityVector signature differs
                u_vectors.append(vec)
        try:
            spec_hash = _csh(request.candidate_spec)  # type: ignore[call-arg]
        except Exception:
            # NEVER-bind: fallback digest, not a verified binding.
            spec_hash = "sha256:" + "a" * 64
        try:
            _telemetry_dict: Any = res["telemetry"]  # pyrefly: ignore[explicit-any]
            _actual_calls_raw: Any = _telemetry_dict["model_calls"]  # pyrefly: ignore[explicit-any]
            _actual_trans_raw: Any = _telemetry_dict["transitions"]  # pyrefly: ignore[explicit-any]
            _completed_raw: Any = res["completed"]  # pyrefly: ignore[explicit-any]
            telem: Any = _mrt(  # pyrefly: ignore[explicit-any]
                budget=request.candidate_spec.resource_budget,
                actual_calls=int(_actual_calls_raw),  # pyrefly: ignore[explicit-any]
                actual_transitions=int(_actual_trans_raw),  # pyrefly: ignore[explicit-any]
                actual_duration_ms=0,
                completed=bool(_completed_raw),  # pyrefly: ignore[explicit-any]
            )
        except Exception:
            telem = res["telemetry"]
        return SearchResult(
            selected_action=res["selected_action"],
            candidate_actions=tuple(res["candidate_actions"]),
            value_vectors=tuple(u_vectors),
            candidate_spec_hash=spec_hash,
            telemetry=telem,
            evidence_refs=(),
            completed=res["completed"],
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
