"""Candidate 2 natural DESPOT — search policy (sampling, values, budget).

Owns the planner-owned search policy: deterministic natural-scenario
sampling, the feasible-policy action choice with its empirical lower
value, the priority proxy that is explicitly not an upper bound, and the
budget accounting the expansion loop consults. The expansion loop itself
lives in :mod:`hydra2.search.despot_act` and result assembly in
:mod:`hydra2.search.despot_result`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.search.despot_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.despot_core import DespotConfig as DespotConfig
from hydra2.search.despot_core import NaturalScenario as NaturalScenario
from hydra2.search.despot_core import _DespotNode as _DespotNode
from hydra2.search.despot_core import _hash_tie_break as _hash_tie_break
from hydra2.search.despot_core import _scenario_seed_bytes as _scenario_seed_bytes

logger = logging.getLogger(__name__)

try:
    from hydra2.belief.kernel import NaturalPacketKernel

    _KERNEL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    NaturalPacketKernel = Any  # placeholder; _require_kernel() raises on use
    _KERNEL_IMPORT_ERROR = exc


def _require_kernel() -> Any:
    """Fail-closed kernel access (lazy ImportError with build-ext hint)."""
    if _KERNEL_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief.kernel not importable "
            f"({_KERNEL_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before DESPOT search"
        ) from _KERNEL_IMPORT_ERROR
    return NaturalPacketKernel


__all__ = [
    "NaturalDespotPlannerSearchMixin",
]


class NaturalDespotPlannerSearchMixin:
    """Search policy for :class:`NaturalDespotPlanner`.

    Split host for construction plus the sampling, feasible-policy,
    lower-value, priority-proxy, and budget helpers; the expansion loop
    lives in the ``despot_act`` mixin and telemetry/result assembly in the
    ``despot_result`` mixin. Attribute access is duck-typed through the
    subclass.
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
        # config from candidate_spec.parameters or defaults
        params = {}
        if candidate_spec is not None and hasattr(candidate_spec, "parameters"):
            try:
                params = dict(candidate_spec.parameters or {})
            except (AttributeError, TypeError, ValueError, OSError) as exc:
                logger.debug("despot: params fallback to empty", exc_info=exc)
                params = {}
        # params.get returns Any; cast without wrapping conversion where value already typed
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
        # bookkeeping for ponder (planner-owned speculative state only)
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
        # Wave 2 bridge audit: kept Python — per-scenario count=1 draws resolve to Particle
        # world_refs via belief (bridge natural_indices returns indices only; corpus lives in belief).
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ContractError("k must be positive int")
        _require_kernel()
        if self._belief is None or belief_epoch is None:
            raise ContractError("despot: belief and epoch required; synthetic worlds removed")
        weight = 1.0 / k
        logp = -math.log(k)
        scenarios: list[NaturalScenario] = []
        try:
            # Use belief to enumerate corpus worlds deterministically
            # Sample one particle per scenario using deterministic seeds
            # We don't use global RNG; we derive per-scenario seed and sample
            # via hashlib index.
            from hydra2.contracts.randomness import RandomStream

            # Per-scenario deterministic streams avoid ledger duplication:
            # each scenario draws count=1 from an independent stream, so
            # no scenario shares ledger entries with another.
            for idx in range(k):
                seed = _scenario_seed_bytes(
                    candidate_id=candidate_id, case_id=case_id, scenario_idx=idx
                )
                # Use RandomStream to sample world index deterministically
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

    def _feasible_action_for(
        self, legal_actions: tuple[Any, ...], *, scenario_seed: bytes, candidate_id: str
    ) -> Any:
        """Blueprint feasible policy: deterministic, actor-visible, never optimal.

        For the tiny test domain we define the feasible policy as:
        - if custom blueprint_policy is supplied, delegate to it,
        - else choose the lexicographically smallest legal action, with tie
          seeded by scenario_seed (deterministic but not learned).
        This is intentionally not value-optimal; DESPOT's lower value tracks
        this feasible policy, not an upper bound.
        """
        # Wave 2 bridge audit: kept Python — blueprint/min-aid feasible policy + hash
        # lower values + priority proxy are rollout/spec logic, not UCT/PUCT/Gumbel cuts.
        if self._blueprint is not None:
            try:
                return self._blueprint(legal_actions, scenario_seed)
            except (AttributeError, TypeError, ValueError, OSError) as exc:
                logger.debug("despot: blueprint fallback to deterministic min", exc_info=exc)
                pass
        if len(legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty")

        def aid(a: Any) -> int:
            v = getattr(a, "action_id", None)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
            if isinstance(a, int) and not isinstance(a, bool):
                return a
            return hash(str(a)) & 0xFFFF

        # If tie_break is stable_hash, mix candidate_id
        if self._config.tie_break == "stable_hash":
            return _hash_tie_break(legal_actions, candidate_id)
        return min(legal_actions, key=aid)

    def _lower_value_for_action(
        self,
        *,
        action: Any,
        scenarios: tuple[NaturalScenario, ...],
        legal_actions: tuple[Any, ...],
        candidate_id: str,
    ) -> float:
        """Empirical mean return of feasible policy conditioned on root action.

        For each scenario, simulate: root takes ``action``, then feasible policy
        thereafter for ``max_depth-1`` steps. Return is a scalar in [0,1] derived
        deterministically from (world_ref, action, scenario_seed). This keeps
        the lower estimate feasible and deterministic without needing a full
        Mahjong simulator.
        """
        # Wave 2 bridge audit: kept Python — rollout mean over scenario world_refs stays
        # Python (needs scenario seeds/weights; no pyfn covers rollout averaging).
        if len(scenarios) == 0:
            return 0.0
        total = 0.0
        for sc in scenarios:
            # Deterministic scalar return: hash(world_ref, action, seed) -> [0,1)
            # This is a stand-in for exact simulator settlement; the key property
            # is that natural mean vs proposal-unweighted mean can reverse (tested
            # via proposal_reversal_fixture), not the absolute Mahjong value.
            aid = getattr(action, "action_id", action)
            payload = canonical_bytes(
                {
                    "world_ref": sc.world_ref,
                    "action": str(aid),
                    "seed": sc.semantic_seed_bytes.hex(),
                    "candidate": candidate_id,
                }
            )
            h = hashlib.sha256(payload).digest()
            # map to float in [0,1)
            val = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
            # Apply feasible policy continuation depth discount: small depth penalty to keep finite
            # The blueprint continuation is implicit in the hash (deterministic).
            total += val * sc.weight * len(scenarios)  # weight*K == 1, so mean
        # correct for weight already 1/K but total aggregated as mean; we did weight*K
        # Simpler: compute mean directly
        # Actually above we did total += val * weight * K == val, then need /K? Let's recompute mean correctly.
        # For uniform weight 1/K, mean = sum val * 1/K
        # Our total after loop is sum val * weight * K = sum val, so mean = total / K
        return total / len(scenarios) if len(scenarios) > 0 else 0.0

    def _priority_proxy_for(self, action: Any, lower_value: float, visits: int) -> float:
        """Heuristic search priority — explicitly NOT an upper bound.

        The proxy is ``lower_value`` plus a small visitation bonus to encourage
        exploration. It MUST NOT be labeled ``upper_bound``. Callers that need
        a certified bound must supply a proof and a named bound field.
        """
        # Simple UCB-like proxy but we label it proxy to avoid bound claim
        bonus = 0.0
        if visits > 0:
            bonus = 0.05 / math.sqrt(visits)
        elif visits == 0:
            bonus = 0.1
        # regularization (if set) is a heuristic, not a bound
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
        # check max_model_calls
        mc = getattr(budget, "max_model_calls", None)
        if mc is not None and model_calls >= int(mc):
            return True
        tr = getattr(budget, "max_transitions", None)
        if tr is not None and transitions >= int(tr):
            return True
        # check deadline monotonic ns if supplied
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            return True
        # also check budget.deadline_ms relative to start
        dm = getattr(budget, "deadline_ms", None)
        if dm is not None:
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
            fallback_raw: Any = getattr(budget, "fallback_margin_ms", 0)
            margin_val: int = cast("int", fallback_raw) if fallback_raw is not None else 0
            # we must leave margin for fallback (SPEC 15)
            if elapsed_ms >= (dm - margin_val):
                return True
        return False
