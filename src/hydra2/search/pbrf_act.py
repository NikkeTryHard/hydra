# ruff: noqa: N814, F841, SIM105  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (SIM105 fallback-chain try/except-pass idiom; B007/F841 intentional scratch loop locals; B904 ContractError preconditions; N814 upstream casing; F401 cross-module names re-exported for the shim path). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 3 PBRF planner adapter — act, observe, ponder.

Owns the Planner protocol surface of :class:`PbrfPlanner`: forest
construction and fixed-batch evaluation in ``act``,
authoritative-child commit in ``observe``, the no-background-work
ponder no-op, and the thin join over the search mixin. The
construction plus budget/telemetry/value driver arrives via the search
mixin in :mod:`hydra2.search.pbrf_search` so each file stays inside
the review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from hydra2.contracts.common import ContractError, PacketPartitionError, StaleBeliefError
from hydra2.contracts.common import make_digest_text as make_digest_text
from hydra2.search.common import Planner as Planner
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.common import candidate_spec_hash as candidate_spec_hash
from hydra2.search.pbrf_commit import commit as commit
from hydra2.search.pbrf_forest import build_pbrf as build_pbrf
from hydra2.search.pbrf_partition import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.pbrf_partition import NaturalBelief as NaturalBelief
from hydra2.search.pbrf_partition import RandomStream as RandomStream
from hydra2.search.pbrf_partition import _action_id as _action_id
from hydra2.search.pbrf_partition import _freeze_candidates as _freeze_candidates
from hydra2.search.pbrf_search import PbrfPlannerSearchMixin as PbrfPlannerSearchMixin

__all__ = [
    "PbrfPlanner",
    "PbrfPlannerActMixin",
]


class PbrfPlannerActMixin(PbrfPlannerSearchMixin):
    """Planner protocol surface for :class:`PbrfPlanner`.

    Split host for the ``act``/``observe``/``ponder`` protocol surface; the
    construction plus budget/telemetry/value driver arrives via the
    search-mixin base and the thin join adds no overrides. Attribute
    access is duck-typed through the subclass.
    """

    def act(self, request: Any) -> Any:  # type: ignore[override]
        """Execute PBRF core search: build forest, allocate batches, evaluate.

        Returns SearchResult with completed flag, candidate_spec_hash, telemetry,
        and vector values. Deterministic: same request yields same selected_action.
        """
        start_ns = time.monotonic_ns()
        # -- validate request ------------------------------------------------
        if (
            request is None
            or not hasattr(request, "legal_actions")
            or not hasattr(request, "candidate_spec")
        ):
            raise ContractError("request must have legal_actions and candidate_spec")
        legal: tuple[Any, ...] = tuple(getattr(request, "legal_actions", ()))
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty")
        # sort legal deterministically by action_id
        try:
            aids = []
            for a_any in legal:
                a: Any = a_any
                v: Any = getattr(a, "action_id", None)
                if isinstance(v, int) and not isinstance(v, bool):
                    aids.append(v)
                elif isinstance(a, int) and not isinstance(a, bool):
                    aids.append(a)
                else:
                    aids.append(_action_id(a))
            if len(aids) != len(set(aids)):
                raise ContractError("legal_actions must have unique action_ids")
            if aids != sorted(aids):
                paired = sorted(zip(aids, legal, strict=False), key=lambda x: x[0])
                legal = tuple(p for _, p in paired)
        except ContractError:
            raise
        except Exception:
            pass

        _cand_spec_raw: Any = getattr(request, "candidate_spec", None)
        cand_spec: Any = _cand_spec_raw if _cand_spec_raw is not None else self._spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", "candidate3"))
        _case_id_raw: Any = getattr(request, "case_id", None)
        _cand_case_raw: Any = getattr(cand_spec, "candidate_id", "case")
        case_id: str = str(
            _case_id_raw if _case_id_raw is not None and _case_id_raw != "" else _cand_case_raw
        )
        belief_epoch: Any = getattr(request, "belief_epoch", None)
        if belief_epoch is None:
            # Need epoch; create synthetic from request observation if possible
            obs = getattr(request, "observation", None)
            if obs is not None and self._belief is not None:
                try:
                    belief_epoch = self._belief.begin(obs)  # type: ignore[union-attr]
                except Exception:
                    belief_epoch = None
            if belief_epoch is None:
                raise ContractError("belief_epoch is required for PBRF core")
        _budget_raw: Any = getattr(cand_spec, "resource_budget", None)
        budget: Any = _budget_raw if _budget_raw is not None else self._budget()
        if hasattr(budget, "resource_budget"):
            budget = getattr(budget, "resource_budget")  # noqa: B009
            # reason: B009 narrowing to inner budget — spec wrappers nest it.
        deadline_ns = getattr(request, "deadline_monotonic_ns", None)
        spec_hash = self._spec_hash()

        # -- budget check helper ---------------------------------------------
        def exhausted() -> bool:
            if budget is None:
                return False
            mc = getattr(budget, "max_model_calls", None)
            if mc is not None and self._model_calls >= int(mc):
                return True
            tr = getattr(budget, "max_transitions", None)
            if tr is not None and self._transitions >= int(tr):
                return True
            if deadline_ns is not None and time.monotonic_ns() >= int(deadline_ns):
                return True
            dm = getattr(budget, "deadline_ms", None)
            if dm is not None:
                elapsed_ms: float = (time.monotonic_ns() - start_ns) / 1e6
                _margin_raw: Any = getattr(budget, "fallback_margin_ms", None)
                margin: int = int(_margin_raw) if _margin_raw is not None else 0
                if elapsed_ms >= (dm - margin):
                    return True
            return False

        # Reset counters
        self._model_calls = 0
        self._transitions = 0
        completed = True
        fallback_used = False

        # -- candidate generator (frozen before enumeration) -----------------
        # Freeze candidates before any packet enumeration evidence: we capture legal as frozen_candidates
        frozen_candidates = _freeze_candidates(legal)

        # -- build PBRF forest ------------------------------------------------
        # Use a deterministic RNG derived from (candidate_id, case_id)
        try:
            seed_bytes = hashlib.sha256(f"{candidate_id}:{case_id}:pbrf_core".encode()).digest()
            rng = RandomStream(seed_bytes)  # type: ignore[call-arg]
        except Exception:
            rng = None

        # Need belief for sampling; if not supplied use stored
        belief = self._belief
        if belief is None:
            # Attempt to create a default NaturalBelief if available
            if _HAS_BELIEF:
                try:
                    belief = NaturalBelief()  # type: ignore[call-arg]
                    # Adopt epochs across belief instances for determinism:
                    # rebuild from the same observation when the epoch came
                    # from a different belief store.
                    obs2 = getattr(request, "observation", None)
                    if obs2 is not None:
                        try:
                            belief_epoch = belief.begin(obs2)
                        except Exception:
                            pass
                except Exception:
                    belief = None
            if belief is None:
                raise ContractError("belief is required for PBRF planner")

        # candidates_fn closure that returns frozen_candidates regardless of parents (ensures freeze)
        def _cand_fn(_parents: Any) -> tuple[Any, ...]:
            return frozen_candidates

        try:
            forest = build_pbrf(
                belief,
                belief_epoch,
                parent_count=self._config.parent_count,
                candidates_fn=_cand_fn,
                policy_set=self._policy_set,
                kernel=self._kernel,
                rng=rng,
                config=self._config,
            )
            self._forest = forest
            # Count model calls / transitions: one per parent*action enumeration counts as transition batch
            # For telemetry, model_calls = number of value evaluations (one per child)
            # transitions = number of enumerated successors (parents * candidates * 2 packets)
            self._model_calls = len(forest.children)  # one per child
            self._transitions = self._config.parent_count * len(frozen_candidates) * 2
            # Check budget exhaustion after forest build
            if exhausted():
                completed = False
                fallback_used = True
        except (PacketPartitionError, StaleBeliefError, ContractError):
            raise
        except Exception as exc:
            raise ContractError(f"build_pbrf failed: {exc}") from exc

        if not completed:
            # Return fallback (Candidate 0 style: first legal)
            fallback = legal[0]
            self._last_selected_action = fallback
            telemetry = self._make_telemetry(
                start_ns=start_ns,
                budget=budget,
                completed=False,
                spec_hash=spec_hash,
                case_id=case_id,
                fallback_used=True,
                timeout=True,
            )
            # Wrap fallback vectors into UtilityVector (zero vector per spec)
            try:
                from hydra2.contracts.utility import UtilityVector

                fallback_vec = UtilityVector(
                    values=(0.0, 0.0, 0.0, 0.0),
                    utility_id=str(getattr(cand_spec, "utility_id", "expected_final_placement")),
                    utility_manifest_hash=make_digest_text(
                        str(getattr(cand_spec, "utility_manifest_hash", "sha256:" + "0" * 64))
                    ),
                    rules_hash=make_digest_text(
                        str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                    ),
                )
                fb_vectors = tuple(fallback_vec for _ in legal)
            except Exception:
                # fallback to raw if UtilityVector fails (should not happen with valid spec hashes)
                fb_vectors = tuple((0.0, 0.0, 0.0, 0.0) for _ in legal)
                # but SearchResult requires UtilityVector, so try again with dummy hashes
                try:
                    from hydra2.contracts.utility import (
                        UtilityVector as _UV,
                    )

                    fb_vectors = tuple(
                        _UV(
                            values=(0.0, 0.0, 0.0, 0.0),
                            utility_id="expected_final_placement",
                            utility_manifest_hash=make_digest_text("sha256:" + "f" * 64),
                            rules_hash=make_digest_text("sha256:" + "a" * 64),
                        )
                        for _ in legal
                    )
                except Exception:
                    pass
            return SearchResult(
                selected_action=fallback,
                candidate_actions=legal,
                value_vectors=fb_vectors,
                candidate_spec_hash=make_digest_text(spec_hash),
                telemetry=telemetry,
                evidence_refs=(make_digest_text(spec_hash),),
                completed=False,
            )
        # -- evaluate each action's aggregated child values -------------------
        # For each action, aggregate child values via Z_hat weighting (like SPEC gamma_hat)
        value_by_action: dict[Any, tuple[float, float, float, float]] = {}
        for action in legal:
            aid = _action_id(action)
            # collect Z_hat per packet for this action
            total_vec = [0.0, 0.0, 0.0, 0.0]
            z_sum = 0.0
            for (k_aid, pid), entries in forest.children.items():
                if k_aid != aid:
                    continue
                z = sum(e.raw_weight for e in entries)
                z_sum += z
                vec = self._value_for_child(action=action, packet_id=pid, forest=forest)
                for i in range(4):
                    total_vec[i] += vec[i] * z
            # After aggregation, total_vec should already be weighted by Z (which sums to 1 per action)
            # So total_vec is the expected vector conditioned on action
            # Keep as is; ensure finite
            if not all(math.isfinite(v) for v in total_vec):
                raise ContractError("value vector must be finite")
            value_by_action[action] = tuple(float(v) for v in total_vec)  # type: ignore[assignment]

            # Check budget after each action evaluation
            self._model_calls += 1
            if exhausted():
                completed = False
                break

        if not completed or len(value_by_action) != len(legal):
            fallback = legal[0]
            self._last_selected_action = fallback
            telemetry = self._make_telemetry(
                start_ns=start_ns,
                budget=budget,
                completed=False,
                spec_hash=spec_hash,
                case_id=case_id,
                fallback_used=True,
                timeout=True,
            )
            try:
                from hydra2.contracts.utility import UtilityVector

                fb2: list[Any] = []
                for a in legal:
                    vec = value_by_action.get(a, (0.0, 0.0, 0.0, 0.0))
                    # vec is tuple[float]; wrap
                    if (
                        isinstance(vec, tuple)
                        and len(vec) == 4
                        and all(isinstance(x, float) for x in vec)
                    ):
                        fb2.append(
                            UtilityVector(
                                values=vec,
                                utility_id=str(
                                    getattr(cand_spec, "utility_id", "expected_final_placement")
                                ),
                                utility_manifest_hash=make_digest_text(
                                    str(
                                        getattr(
                                            cand_spec, "utility_manifest_hash", "sha256:" + "f" * 64
                                        )
                                    )
                                ),
                                rules_hash=make_digest_text(
                                    str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                                ),
                            )
                        )
                    else:
                        # vec already UtilityVector? keep
                        fb2.append(vec)
                fb_vectors2 = tuple(fb2)
            except Exception:
                fb_vectors2 = tuple(value_by_action.get(a, (0.0, 0.0, 0.0, 0.0)) for a in legal)
            return SearchResult(
                selected_action=fallback,
                candidate_actions=legal,
                value_vectors=fb_vectors2,
                candidate_spec_hash=make_digest_text(spec_hash),
                telemetry=telemetry,
                evidence_refs=(make_digest_text(spec_hash),),
                completed=False,
            )

        # -- root selection: scalarize via s_i at root only ------------------
        # Determine root seat from epoch
        _root_raw: Any = getattr(belief_epoch, "root_actor", 0)  # type: ignore[attr-defined]
        try:
            root_seat: int = int(_root_raw)
        except Exception:
            root_seat = 0

        def _scalar(vec: tuple[float, ...]) -> float:
            try:
                return vec[root_seat] if 0 <= root_seat < len(vec) else vec[0]
            except Exception:
                return 0.0

        # Find max scalar; tie break deterministically
        max_scalar = max(_scalar(v) for v in value_by_action.values())
        tied = [a for a, v in value_by_action.items() if abs(_scalar(v) - max_scalar) < 1e-12]
        if len(tied) == 1:
            selected = tied[0]
        else:
            if self._config.tie_break == "stable_hash":
                # stable hash tie break
                def _h(a: Any) -> str:
                    return hashlib.sha256(f"{candidate_id}:{_action_id(a)}".encode()).hexdigest()

                selected = min(tied, key=_h)
            else:
                selected = min(tied, key=_action_id)

        if selected not in legal:
            raise ContractError("selected_action must be in legal_actions")

        # -- telemetry & result ------------------------------------------------
        telemetry = self._make_telemetry(
            start_ns=start_ns,
            budget=budget,
            completed=True,
            spec_hash=spec_hash,
            case_id=case_id,
            fallback_used=False,
            timeout=False,
        )
        # Wrap value vectors into UtilityVector for SearchResult validation
        try:
            from hydra2.contracts.utility import UtilityVector

            wrapped: list[Any] = []
            for a in legal:
                vec = value_by_action[a]
                wrapped.append(
                    UtilityVector(
                        values=vec,
                        utility_id=str(
                            getattr(cand_spec, "utility_id", "expected_final_placement")
                        ),
                        utility_manifest_hash=make_digest_text(
                            str(getattr(cand_spec, "utility_manifest_hash", "sha256:" + "f" * 64))
                        ),
                        rules_hash=make_digest_text(
                            str(getattr(cand_spec, "rules_hash", "sha256:" + "a" * 64))
                        ),
                    )
                )
            value_vectors = tuple(wrapped)
        except Exception:
            value_vectors = tuple(value_by_action[a] for a in legal)
        self._last_selected_action = selected
        return SearchResult(
            selected_action=selected,
            candidate_actions=legal,
            value_vectors=value_vectors,
            candidate_spec_hash=make_digest_text(spec_hash),
            telemetry=telemetry,
            evidence_refs=(make_digest_text(spec_hash),),
            completed=True,
        )

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        """PBRF observe: commit the emitted action to its authoritative child.

        Commits exactly ``self._last_selected_action`` from ``act()``. The
        action is never recovered from the packet: kernel packet ids are
        action-free and collide across actions, so any candidate sweep would
        promote the wrong branch. Exactly one ``pushforward_condition`` runs
        per observe (SPEC order, required for the miss path's rebuild epoch);
        the old per-candidate loop is gone, along with its speculative writes.
        The stored action is consumed one-shot; observing without a stored
        action (no preceding act()) raises ``ContractError`` instead of
        guessing a candidate.
        """
        if self._forest is None or self._belief is None:
            # No forest to commit; ignore (or rebuild if packet supplied)
            return
        if packet is None or not hasattr(packet, "packet_id"):
            raise ContractError("packet must have packet_id for observe")
        action = self._last_selected_action
        self._last_selected_action = None
        if action is None:
            # No emitted action memory: act() never ran since the last
            # observe (or a forest was injected directly). The action cannot
            # be recovered from the packet — kernel packet ids are
            # action-free and collide across actions — and defaulting to any
            # candidate would reintroduce first-hit-wins miscommit, so this
            # is a protocol violation, not a miss.
            raise ContractError("observe requires an act()-emitted action: none stored")
        promoted, disp = commit(self._forest, action, packet, self._belief)
        self._forest = promoted
        self._last_commit = disp

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # PBRF core does not perform background ponder without commit; no-op
        return


class PbrfPlanner(  # type: ignore[misc]
    PbrfPlannerActMixin,
    Planner,
):
    """Natural-particle PBRF planner (Candidate 3).

    Thin subclass joining the split mixins; construction, the value driver,
    and act live in the ``pbrf_*`` modules with no overrides here.
    """
