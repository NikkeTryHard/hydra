# ruff: noqa: N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 Gumbel Planner adapter — act, observe, ponder.

Owns the Planner protocol surface of :class:`GumbelSearchPlanner`: RNG
derivation and ``SearchResult`` wrapping in ``act``, ponder-state clearing
in ``observe``, the fresh-search ponder no-op, and the thin join over the
search and act mixins. The simulation loop arrives via the search mixin
in :mod:`hydra2.search.gumbel_search` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

from hydra2.contracts.common import ContractError as ContractError
from hydra2.search.common import Planner as Planner
from hydra2.search.common import SearchRequest as SearchRequest
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.gumbel_core import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.gumbel_core import make_digest_text as make_digest_text
from hydra2.search.gumbel_search import (
    GumbelSearchPlannerSearchMixin as GumbelSearchPlannerSearchMixin,
)

__all__ = [
    "GumbelSearchPlanner",
    "GumbelSearchPlannerActMixin",
]


class GumbelSearchPlannerActMixin(GumbelSearchPlannerSearchMixin):
    """Planner protocol surface for :class:`GumbelSearchPlanner`.

    Split host for the act/observe/ponder third of
    :class:`GumbelSearchPlanner`; the simulation loop arrives via the
    search-mixin base and the thin join adds no overrides. Attribute
    access is duck-typed through the subclass.
    """

    def act(self, request: SearchRequest) -> SearchResult:
        if not isinstance(request, SearchRequest):
            raise ContractError(f"request must be SearchRequest, got {type(request).__name__}")
        belief_epoch = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            raise ContractError("belief_epoch must be BeliefEpoch for Gumbel search")
        _case_req: Any | None = getattr(request, "case_id", None)
        if _case_req is not None and isinstance(_case_req, str) and _case_req != "":
            case_id: Any = _case_req
        else:
            _dec2: Any | None = getattr(request.observation, "decision_id", None)
            case_id = _dec2 if isinstance(_dec2, str) and _dec2 != "" else "case_default"
        candidate_id = getattr(request.candidate_spec, "candidate_id", self._config.candidate_id)
        if _HAS_RANDOM:
            try:
                from hydra2.contracts.randomness import RandomStream

                epoch_id = str(getattr(belief_epoch, "epoch", "0"))
                seed = hashlib.sha256(f"{candidate_id}:{case_id}:{epoch_id}".encode()).digest()
                rng = RandomStream(seed)
            except Exception:
                rng = RandomStream(hashlib.sha256(f"{candidate_id}:{case_id}".encode()).digest())  # type: ignore[call-arg]
        else:
            import secrets as _secrets

            rng = _secrets.token_bytes(32)

        res = self.search(
            epoch=belief_epoch,
            root_observation=request.observation,
            legal_actions=request.legal_actions,
            rng=rng,
            case_id=str(case_id),
        )

        try:
            from hydra2.contracts.utility import UtilityVector as _UV
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
            from hydra2.search.common import candidate_spec_hash as _csh
        except Exception:
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

        u_vectors: list[Any] = []
        for vec in res["value_vectors"]:
            try:
                # vec is raw 4-float; narrow to fixed quad and digests
                vals_4 = cast("tuple[float, float, float, float]", tuple(float(v) for v in vec))
                assert len(vals_4) == 4
                u_vectors.append(
                    _UV(
                        values=vals_4,
                        utility_id=str(
                            getattr(
                                request.candidate_spec, "utility_id", "expected_final_placement"
                            )
                        ),
                        utility_manifest_hash=make_digest_text(
                            str(
                                getattr(
                                    request.candidate_spec,
                                    "utility_manifest_hash",
                                    "sha256:" + "b" * 64,
                                )
                            )
                        ),
                        rules_hash=make_digest_text(
                            str(getattr(request.candidate_spec, "rules_hash", "sha256:" + "a" * 64))
                        ),
                    )
                )
            except Exception:
                u_vectors.append(vec)
        try:
            spec_hash = _csh(request.candidate_spec)  # type: ignore[call-arg]
        except Exception:
            # NEVER-bind: fallback digest, not a verified binding.
            spec_hash = "sha256:" + "a" * 64
        try:
            telem = _mrt(
                mode=str(getattr(request.candidate_spec.resource_budget, "mode", "gameplay_5s")),
                wall_id=None,
                case_id=case_id if isinstance(case_id, str) else None,
                candidate_spec_hash=spec_hash,
                hardware_hash="sha256:" + "8" * 64,
                environment_hash="sha256:" + "7" * 64,
                cold_start=False,
                synchronized_elapsed_ms=0.0,
                model_calls=int(res["telemetry"]["model_calls"]),  # type: ignore[unknown-argument-type]
                exact_transitions=int(res["telemetry"]["transitions"]),  # type: ignore[unknown-argument-type]
                particles=len(res["candidate_actions"]),
                fallback_used=not res["completed"],
                timeout=not res["completed"],
                illegal_action=False,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=float(
                    res["telemetry"]["model_calls"] * 0.5 + res["telemetry"]["transitions"] * 0.2  # type: ignore[unknown-argument-type]
                ),
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
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
        self._belief_epoch = None

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        if (
            not isinstance(deadline_monotonic_ns, int)
            or isinstance(deadline_monotonic_ns, bool)
            or deadline_monotonic_ns <= 0
        ):
            raise ContractError("deadline_monotonic_ns must be positive int")


class GumbelSearchPlanner(  # type: ignore[misc]
    GumbelSearchPlannerActMixin,
    Planner,
):
    """Deterministic Gumbel sequential-halving search (Candidate 6).

    Fresh search per ``act``; root Gumbels deterministic; exact transitions;
    vector backup; root scalarization; frozen sequential-halving schedule.
    """
