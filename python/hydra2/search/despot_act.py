"""Candidate 2 natural DESPOT — trigger-only drive (Candidate 2 act path)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream
from hydra2.search._drive_trigger import pack_rng as _pack_rng
from hydra2.search._drive_trigger import pack_worlds as _pack_worlds
from hydra2.search._drive_trigger import wrap_result as _wrap_result
from hydra2.search.common import Planner as Planner
from hydra2.search.despot_core import ResourceBudget as ResourceBudget
from hydra2.search.despot_core import _default_budget as _default_budget
from hydra2.search.despot_result import (
    NaturalDespotPlannerResultMixin as NaturalDespotPlannerResultMixin,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NaturalDespotPlanner",
    "NaturalDespotPlannerActMixin",
]


class NaturalDespotPlannerActMixin(NaturalDespotPlannerResultMixin):
    """Expansion loop for :class:`NaturalDespotPlanner`.

    Split host for the ``act`` expansion loop plus the spec-hash helper;
    telemetry/result assembly lives in the ``despot_result`` mixin and the
    final subclass adds nothing and no overrides. Attribute access is
    duck-typed through the subclass.
    """

    # -- Planner interface -------------------------------------------------

    def act(self, request: Any) -> Any:  # type: ignore[override]
        """Trigger-only DESPOT act: validate, pack frozen ticket, one Rust drive call, wrap."""
        if (
            request is None
            or not hasattr(request, "legal_actions")
            or not hasattr(request, "candidate_spec")
        ):
            raise ContractError("request must have legal_actions and candidate_spec")
        legal = tuple(getattr(request, "legal_actions", ()))
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        try:
            aids: list[int] = []
            for _a in legal:
                _v = getattr(_a, "action_id", None)
                if isinstance(_v, int) and not isinstance(_v, bool):
                    aids.append(_v)
                elif isinstance(_a, int) and not isinstance(_a, bool):
                    aids.append(_a)
                else:
                    import hashlib as _ahl

                    _h = _ahl.sha256(str(_a).encode()).hexdigest()
                    aids.append(int(_h[:8], 16) & 0xFFFF)
            if len(aids) != len(set(aids)):
                raise ContractError("legal_actions must have unique action_ids")
            if aids != sorted(aids):
                paired = sorted(zip(aids, legal, strict=False), key=lambda x: x[0])
                legal = tuple(p for _, p in paired)
                aids = sorted(aids)
        except ContractError:
            raise
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"despot: legal_actions ids unreadable: {exc}") from exc
        cand_spec: Any = request.candidate_spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", "candidate2"))
        case_raw: Any = getattr(request, "case_id", None)
        cand_fallback: Any = getattr(cand_spec, "candidate_id", "case_default")
        case_val: Any = case_raw if case_raw is not None else cand_fallback
        if isinstance(case_val, str) and case_val == "":
            case_val = cand_fallback
        case_id: str = str(case_val) if case_val is not None else "case_default"
        belief_epoch: Any | None = getattr(request, "belief_epoch", None)
        if belief_epoch is None or getattr(self, "_belief", None) is None:
            raise ContractError("despot: belief and epoch required; synthetic worlds removed")
        budget_raw: Any = getattr(cand_spec, "resource_budget", None)
        budget: Any = (
            budget_raw if budget_raw is not None else getattr(request, "candidate_spec", None)
        )
        if hasattr(budget, "resource_budget"):
            budget = budget.resource_budget
        if budget is None or not isinstance(budget, ResourceBudget):
            try:
                budget = _default_budget()
            except (AttributeError, ValueError, TypeError, OSError):
                budget = ResourceBudget(
                    mode="gameplay_5s",
                    deadline_ms=5000,
                    fallback_margin_ms=200,
                    max_model_calls=64,
                    max_transitions=256,
                    max_particles=16,
                    max_memory_bytes=None,
                )
        start_ns: int = time.monotonic_ns()
        k: int = int(getattr(getattr(self, "_config", None), "num_scenarios", 16))
        try:
            params_any: Any = getattr(cand_spec, "parameters", {})
            if isinstance(params_any, dict) and "num_scenarios" in params_any:
                k = int(params_any.get("num_scenarios", k))  # pyrefly: ignore[unknown-argument-type] # Any spec params
        except (AttributeError, TypeError, ValueError):
            pass
        if k <= 0:
            raise ContractError("k must be positive int")
        max_depth: int = int(getattr(getattr(self, "_config", None), "max_depth", 4))
        tie_break: str = str(getattr(getattr(self, "_config", None), "tie_break", "lexicographic"))
        deadline_ms: int = int(getattr(budget, "deadline_ms", 5000))
        fallback_margin_ms: int = int(getattr(budget, "fallback_margin_ms", 200))
        max_calls_raw: Any = getattr(budget, "max_model_calls", 64)
        max_trans_raw: Any = getattr(budget, "max_transitions", 256)
        max_model_calls: int | None = int(max_calls_raw) if max_calls_raw is not None else None
        max_transitions: int | None = int(max_trans_raw) if max_trans_raw is not None else None
        seed_bytes = hashlib.sha256(f"{candidate_id}:{case_id}:despot_drive".encode()).digest()
        rng = RandomStream(seed_bytes)
        try:
            particles: Any = self._belief.sample_natural(belief_epoch, count=k, rng=rng)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"despot: belief sampling failed: {exc}") from exc
        try:
            raw_particles: list[Any] = list(particles)
        except TypeError as exc:
            raise ContractError(f"despot: belief sampling malformed: {exc}") from exc
        cur_worlds: list[Any] = []
        for _p in raw_particles:
            _wref: Any = getattr(_p, "world_ref", None)
            if not isinstance(_wref, str) or _wref == "":
                raise ContractError("despot: particle missing world_ref")
            try:
                _w: Any = self._belief._worlds[_wref]  # type: ignore[attr-defined]
            except Exception as exc:
                raise ContractError(f"despot: belief world missing for particle: {exc}") from exc
            cur_worlds.append(_w)
        worlds_json = _pack_worlds(cur_worlds)
        seed_frozen, cursor_frozen = _pack_rng(rng)
        rules_hash: str = str(getattr(belief_epoch, "rules_hash", ""))
        obs_hash: str = str(getattr(belief_epoch, "observation_hash", ""))
        ticket = {
            "worlds": worlds_json,
            "rules_hash": rules_hash,
            "observation_hash": obs_hash,
            "root_legal": sorted(aids),
            "candidate_id": candidate_id,
            "case_id": case_id,
            "attempt_id": 0,
            "num_scenarios": k,
            "max_sims": k,
            "max_depth": max_depth,
            "max_transitions": max_transitions,
            "max_model_calls": max_model_calls,
            "deadline_ms": deadline_ms,
            "fallback_margin_ms": fallback_margin_ms,
            "tie_break": tie_break,
            "seed_hex": seed_frozen.hex(),
            "cursor": cursor_frozen,
        }
        try:
            from hydra2._native import search as _drive_bridge
        except ImportError as exc:
            raise ImportError(
                f"hydra2._native.search not importable ({exc}); build the bridge with `pixi run build-ext` before despot drive"
            ) from exc
        try:
            out: Any = _drive_bridge.despot_search_batch(json.dumps(ticket).encode())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(f"despot bridge drive failed: {exc}") from exc
        try:
            end_cursor: int = out.end_cursor
        except Exception:
            end_cursor = cursor_frozen
        try:
            rng.jump_to(end_cursor)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ContractError(f"despot: rng jump_to failed: {exc}") from exc
        return _wrap_result(
            out,  # pyrefly: ignore[unknown-argument-type] # native DespotOut
            legal=legal,
            candidate_spec=cand_spec,
            start_ns=start_ns,
            case_id=case_id,
            particles=k,
        )


class NaturalDespotPlanner(  # type: ignore[misc]
    NaturalDespotPlannerActMixin,
    Planner,
):
    """Natural-scenario DESPOT planner (Candidate 2).

    Thin subclass joining the split mixins; construction, the search
    policy, the expansion loop, and result assembly live in the
    ``despot_*`` modules with no overrides here.
    """
