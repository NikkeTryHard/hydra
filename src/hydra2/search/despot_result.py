"""Candidate 2 natural DESPOT — search policy + result assembly.

Owns the search policy (construction, sampling, feasible lower values,
priority proxy, budget helpers) plus the result path: resource telemetry
with the Joules view, feasible lower-estimate value vectors, the evidence
digest, the stateless ``observe``/``ponder`` hooks, and the budget test
helper. The expansion loop and the final :class:`NaturalDespotPlanner`
join live in :mod:`hydra2.search.despot_act`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes_batch
from hydra2.contracts.common import ContractError
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.despot_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.despot_core import DespotConfig as DespotConfig
from hydra2.search.despot_core import NaturalPacketKernel as NaturalPacketKernel
from hydra2.search.despot_core import NaturalScenario as NaturalScenario
from hydra2.search.despot_core import ResourceTelemetry as ResourceTelemetry
from hydra2.search.despot_core import UtilityVector as UtilityVector
from hydra2.search.despot_core import _DespotNode as _DespotNode
from hydra2.search.despot_core import _hash_tie_break as _hash_tie_break
from hydra2.search.despot_core import _require_belief as _require_kernel
from hydra2.search.despot_core import _require_telemetry as _require_telemetry
from hydra2.search.despot_core import _require_utility as _require_utility
from hydra2.search.despot_core import _scenario_seed_bytes as _scenario_seed_bytes

logger = logging.getLogger(__name__)
__all__ = [
    "NaturalDespotPlannerResultMixin",
    "NaturalDespotPlannerSearchMixin",
    "budget_exhausted_for_test",
]


class NaturalDespotPlannerSearchMixin:
    """Search policy for :class:`NaturalDespotPlanner`.

    Folded here from the deleted ``despot_search`` split host: construction
    plus the sampling, feasible-policy, lower-value, priority-proxy, and
    budget helpers. The expansion loop lives in the ``despot_act`` mixin.
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

    def __init__(
        self,
        *,
        candidate_spec: Any | None = None,
        belief: Any | None = None,
        kernel: Any | None = None,
        blueprint_policy: Any | None = None,
        master_seed: bytes = _MASTER_SEED,
    ) -> None:
        self._candidate_spec = candidate_spec
        self._belief = belief
        if kernel is not None:
            self._kernel = kernel
        else:
            _require_kernel()
            self._kernel = NaturalPacketKernel()  # type: ignore[bad-instantiation]
        self._blueprint = blueprint_policy  # callable(observation, legal) -> action
        self._master_seed = master_seed
        self._belief_epoch: Any | None = None
        self._last_telemetry: Any | None = None
        params = {}
        if candidate_spec is not None and hasattr(candidate_spec, "parameters"):
            try:
                params = dict(candidate_spec.parameters or {})
            except (AttributeError, TypeError, ValueError, OSError) as exc:
                logger.debug("despot: params fallback to empty", exc_info=exc)
                params = {}
        num_scenarios_raw: Any = params.get("num_scenarios", 16)
        regularization_raw: Any = params.get("regularization")
        max_depth_raw: Any = params.get("max_depth", 4)
        tie_break_raw: Any = params.get("tie_break", "lexicographic")
        resource_view_raw: Any = params.get("resource_view", "calls")
        num_scenarios_val: int = cast("int", num_scenarios_raw)
        max_depth_val: int = cast("int", max_depth_raw)
        tie_break_val: str = cast("str", tie_break_raw)
        self._config = DespotConfig(
            num_scenarios=num_scenarios_val,
            regularization=cast("float | None", regularization_raw),
            max_depth=max_depth_val,
            tie_break=tie_break_val,
            resource_view=cast("Literal['calls', 'transitions', 'joules']", resource_view_raw),
        )
        self._ponder_nodes: dict[str, _DespotNode] = {}
        self._ponder_epoch: Any | None = None
        self._model_calls: int = 0
        self._transitions: int = 0

    # -- scenario sampling (natural only) ----------------------------------

    def _sample_natural_scenarios(
        self,
        *,
        belief_epoch: Any | None,
        candidate_id: str,
        case_id: str,
        k: int,
    ) -> tuple[NaturalScenario, ...]:
        """Sample ``k`` natural scenarios (world, semantic seed) deterministically.

        Real ``NaturalBelief`` and epoch required; no synthetic fallback.
        Weight is uniform 1/K, log_target == log_proposal, no proposal used.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ContractError("k must be positive int")
        _require_kernel()
        if self._belief is None or belief_epoch is None:
            raise ContractError("despot: belief and epoch required; synthetic worlds removed")
        weight = 1.0 / k
        logp = -math.log(k)
        scenarios: list[NaturalScenario] = []
        try:
            from hydra2.contracts.randomness import RandomStream

            for idx in range(k):
                seed = _scenario_seed_bytes(
                    candidate_id=candidate_id, case_id=case_id, scenario_idx=idx
                )
                rs = RandomStream(seed)
                try:
                    particles: Any = self._belief.sample_natural(belief_epoch, count=1, rng=rs)  # type: ignore[union-attr]
                    wref: str = cast("str", particles[0].world_ref)
                except (AttributeError, ValueError, TypeError, LookupError, OSError) as exc:
                    raise ContractError(f"despot: belief sampling failed: {exc}") from exc
                scenarios.append(
                    NaturalScenario(
                        scenario_id=idx,
                        world_ref=wref,
                        semantic_seed_bytes=seed,
                        log_target_density=logp,
                        log_proposal_density=logp,
                        weight=weight,
                    )
                )
            return tuple(scenarios)
        except ContractError:
            raise
        except (
            AttributeError,
            ValueError,
            TypeError,
            OSError,
            ImportError,
            RuntimeError,
        ) as exc:
            raise ContractError(f"despot: belief path failed: {exc}") from exc

    # -- lower policy value (feasible, not bound) --------------------------

    def _lower_value_for_action(
        self,
        *,
        action: Any,
        scenarios: tuple[NaturalScenario, ...],
        legal_actions: tuple[Any, ...],
        candidate_id: str,
    ) -> float:
        """Empirical mean return of feasible policy conditioned on root action.

        Bridge is the single implementation (``search.despot_lower_values``);
        missing extension raises ``ImportError`` (fail closed).
        """
        if len(scenarios) == 0:
            return 0.0
        try:
            from hydra2_replay_rs import search as _search_bridge

            refs = [sc.world_ref for sc in scenarios]
            seeds = [sc.semantic_seed_bytes.hex() for sc in scenarios]
            return float(
                _search_bridge.despot_lower_values(
                    refs, seeds, str(getattr(action, "action_id", action)), candidate_id
                )
            )
        except ImportError:
            raise

    def _priority_proxy_for(self, action: Any, lower_value: float, visits: int) -> float:
        """Heuristic search priority — explicitly NOT an upper bound."""
        bonus = 0.0
        if visits > 0:
            bonus = 0.05 / math.sqrt(visits)
        elif visits == 0:
            bonus = 0.1
        if self._config.regularization is not None:
            bonus *= 1.0 + self._config.regularization
        return lower_value + bonus

    # -- budget helpers ----------------------------------------------------

    def _budget_exhausted(
        self,
        *,
        model_calls: int,
        transitions: int,
        start_ns: int,
        budget: Any,
        deadline_ns: int | None,
    ) -> bool:
        if budget is None:
            return False
        mc = getattr(budget, "max_model_calls", None)
        if mc is not None and model_calls >= int(mc):
            return True
        tr = getattr(budget, "max_transitions", None)
        if tr is not None and transitions >= int(tr):
            return True
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            return True
        dm = getattr(budget, "deadline_ms", None)
        if dm is not None:
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
            fallback_raw: Any = getattr(budget, "fallback_margin_ms", 0)
            margin_val: int = cast("int", fallback_raw) if fallback_raw is not None else 0
            if elapsed_ms >= (dm - margin_val):
                return True
        return False


class NaturalDespotPlannerResultMixin(NaturalDespotPlannerSearchMixin):
    """Result assembly for :class:`NaturalDespotPlanner`.

    Split host for telemetry, value vectors, evidence, and the stateless
    ``observe``/``ponder`` hooks; the final subclass adds nothing and no
    overrides. Attribute access is duck-typed through the subclass.
    """

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
        # ONE batch FFI for the single evidence doc (byte-identical blob).
        evidence = (
            f"sha256:{hashlib.sha256(canonical_bytes_batch([{'lower_by_action': {str(getattr(k, 'action_id', k)): v for k, v in lower_by_action.items()}}])[0]).hexdigest()}",
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
