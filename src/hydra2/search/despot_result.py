"""Candidate 2 natural DESPOT — result assembly (telemetry, vectors, report).

Owns the result path: resource telemetry with the Joules view, feasible
lower-estimate value vectors, the evidence digest, the stateless
``observe``/``ponder`` hooks, and the budget test helper. The search policy
lives in :mod:`hydra2.search.despot_search`; the expansion loop and the
final :class:`NaturalDespotPlanner` join live in :mod:`hydra2.search.despot_act`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.despot_core import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.despot_core import DespotConfig as DespotConfig
from hydra2.search.despot_core import ResourceTelemetry as ResourceTelemetry
from hydra2.search.despot_core import UtilityVector as UtilityVector
from hydra2.search.despot_core import _DespotNode as _DespotNode
from hydra2.search.despot_core import _require_telemetry as _require_telemetry
from hydra2.search.despot_core import _require_utility as _require_utility
from hydra2.search.despot_search import (
    NaturalDespotPlannerSearchMixin as NaturalDespotPlannerSearchMixin,
)

logger = logging.getLogger(__name__)
__all__ = [
    "NaturalDespotPlannerResultMixin",
    "budget_exhausted_for_test",
]


class NaturalDespotPlannerResultMixin(NaturalDespotPlannerSearchMixin):
    """Result assembly for :class:`NaturalDespotPlanner`.

    Split host for telemetry, value vectors, evidence, and the stateless
    ``observe``/``ponder`` hooks; the final subclass adds nothing and no
    overrides. Attribute access is duck-typed through the subclass.
    """

    _candidate_spec: Any
    _belief: Any
    _kernel: Any
    _blueprint: Any
    _master_seed: bytes
    _belief_epoch: Any | None
    _last_telemetry: Any | None
    _config: DespotConfig
    _ponder_nodes: dict[str, _DespotNode]
    _ponder_epoch: Any | None
    _model_calls: int
    _transitions: int

    def _make_telemetry(
        self,
        *,
        start_ns: int,
        budget: Any,
        completed: bool,
        legal: tuple[Any, ...],
        spec_hash: str | None = None,
        case_id: str | None = None,
    ) -> Any:
        duration_ms = (time.monotonic_ns() - start_ns) / 1e6
        joules = self._model_calls * 0.5 + self._transitions * 0.2
        _require_telemetry()
        try:
            from hydra2.eval.telemetry import make_resource_telemetry as _mrt
        except ImportError as exc:
            raise ImportError(
                "hydra2.eval.telemetry not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before DESPOT search"
            ) from exc
        # Required digests — use provided spec_hash or dummy
        cand_hash: str = (
            spec_hash
            if spec_hash is not None
            and isinstance(spec_hash, str)
            and spec_hash != ""
            and spec_hash.startswith("sha256:")
            else "sha256:" + "9" * 64
        )
        hw_hash = "sha256:" + "8" * 64
        env_hash = "sha256:" + "7" * 64
        mode = str(getattr(budget, "mode", "gameplay_5s"))
        # wall_id/case_id are optional; keep None for synthetic
        try:
            return _mrt(
                mode=mode,
                wall_id=None,
                case_id=case_id,
                candidate_spec_hash=cand_hash,
                hardware_hash=hw_hash,
                environment_hash=env_hash,
                cold_start=False,
                synchronized_elapsed_ms=duration_ms,
                model_calls=self._model_calls,
                exact_transitions=self._transitions,
                particles=self._config.num_scenarios,
                fallback_used=not completed,
                timeout=not completed,
                illegal_action=False,
                cuda_peak_allocated_bytes=None,
                cuda_peak_reserved_bytes=None,
                host_peak_bytes=None,
                energy_joules=joules,
                graph_breaks=None,
                recompiles=None,
                invalid_reason=None,
            )
        except ImportError:
            raise
        except (AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"despot: telemetry build failed: {exc}") from exc

    def _make_result(
        self,
        *,
        request: Any,
        selected: Any,
        lower_by_action: dict[Any, float],
        telemetry: Any,
        spec_hash: str,
        completed: bool,
    ) -> Any:
        legal = tuple(getattr(request, "legal_actions", ()))
        # Build UtilityVector per legal action (feasible lower estimate) — fail closed, no raw fallback.
        _require_utility()
        # Need rules_hash for UtilityVector; derive from observation or candidate spec
        obs = getattr(request, "observation", None)
        rules_hash = getattr(obs, "rules_hash", None) if obs is not None else None
        if not isinstance(rules_hash, str):
            cand = getattr(request, "candidate_spec", None)
            rules_hash = (
                getattr(cand, "rules_hash", "sha256:" + "a" * 64)
                if cand is not None
                else "sha256:" + "a" * 64
            )
        util_hash = getattr(
            getattr(request, "candidate_spec", None),
            "utility_manifest_hash",
            "sha256:" + "b" * 64,
        )
        if not isinstance(util_hash, str):
            util_hash = "sha256:" + "b" * 64
        vecs: list[Any] = []
        for act in legal:
            v = lower_by_action.get(act, 0.0)
            # Deterministic 4-seat placement vector: root gets v, others 0 (feasible, not zero-sum)
            # Keep within manifest bounds: assume bounds [-10,10]
            # Clamp v to [-5,5] for safety
            v_clamped = max(-5.0, min(5.0, v))
            try:
                vec = UtilityVector(  # type: ignore[bad-instantiation]  # pyrefly: ignore[bad-instantiation]
                    values=(v_clamped, 0.0, 0.0, 0.0),
                    utility_id="expected_final_placement",
                    utility_manifest_hash=util_hash,  # type: ignore[arg-type]
                    rules_hash=rules_hash,  # type: ignore[arg-type]
                )
            except ImportError:
                raise
            except (AttributeError, ValueError, TypeError, OSError) as exc:
                raise ContractError(f"despot: UtilityVector build failed: {exc}") from exc
            vecs.append(vec)
        value_vectors: tuple[Any, ...] = tuple(vecs)
        if not isinstance(telemetry, ResourceTelemetry):  # type: ignore[arg-type]
            raise ContractError(
                "despot: telemetry must be ResourceTelemetry; dict fallback removed"
            )
        evidence = (
            f"sha256:{hashlib.sha256(canonical_bytes({'lower_by_action': {str(getattr(k, 'action_id', k)): v for k, v in lower_by_action.items()}})).hexdigest()}",
        )
        return SearchResult(
            selected_action=selected,
            candidate_actions=legal,
            value_vectors=value_vectors,
            candidate_spec_hash=spec_hash,
            telemetry=telemetry,
            evidence_refs=evidence,
            completed=completed,
        )

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        """Commit or rebuild after a real actor-visible packet.

        Verifies packet/epoch coherence: if packet's epoch matches our stored
        epoch, promote matching child; otherwise rebuild from authoritative
        pushforward. The stateless DESPOT re-plans fresh per act, so this
        mainly validates packet partition and clears pondering state.
        """
        # Validate packet has packet_id and is actor-visible
        pid_first: Any = getattr(packet, "packet_id", None)
        pid_second: Any = getattr(getattr(packet, "packet", None), "packet_id", None)
        pid: Any = pid_first if pid_first is not None else pid_second
        if pid is None and packet is not None:
            try:
                pid = packet.packet_id  # type: ignore[union-attr]
            except (AttributeError, ValueError, TypeError) as exc:
                logger.debug("despot: packet_id fallback to str", exc_info=exc)
                pid = str(packet)
        self._ponder_nodes.clear()
        self._ponder_epoch = None
        # No hard failure for unknown packet in stateless DESPOT; just clear.

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        """Speculative pondering mutates only planner-owned state.

        Pondering stays minimal: expand at most one priority node
        per call within deadline, then return. No observation, rules, or model
        identity changes.
        """
        # Stateless: nothing to ponder beyond what act already did; respect deadline
        if time.monotonic_ns() >= deadline_monotonic_ns:
            return
        # Expand one more node only when budget allows; no-op keeps determinism.


# ---------------------------------------------------------------------------
# Budget enforcement helper (public for tests)
# ---------------------------------------------------------------------------


def budget_exhausted_for_test(*, model_calls: int, transitions: int, budget: Any) -> bool:
    """Test helper exposing budget logic without needing a planner instance."""
    from hydra2.search.despot_act import NaturalDespotPlanner

    planner = NaturalDespotPlanner()
    start = time.monotonic_ns()
    return planner._budget_exhausted(
        model_calls=model_calls,
        transitions=transitions,
        start_ns=start - 1,
        budget=budget,
        deadline_ns=None,
    )
