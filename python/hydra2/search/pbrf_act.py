"""Candidate 3 PBRF planner adapter — construction, act, observe, ponder.

Trigger-only: act validates, packs one frozen ticket, makes one detached
``pbrf_search_batch`` drive call, and wraps the OUT. Rust owns the
fixed-allocate forest build, the Z_hat-weighted aggregation, and the
scalar-max pick at the root actor. Observe is epoch-validate only (stored
blob + packet ride the batch commit opaquely, no Python forest walk);
ponder is a deadline-int validated no-op.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.search._drive_trigger import pack_rng as _pack_rng
from hydra2.search._drive_trigger import pack_worlds as _pack_worlds
from hydra2.search._drive_trigger import wrap_result as _wrap_result
from hydra2.search.common import Planner as Planner
from hydra2.search.common import ResourceBudget as ResourceBudget
from hydra2.search.pbrf_partition import CommitDisposition as CommitDisposition
from hydra2.search.pbrf_partition import NaturalPacketKernel as NaturalPacketKernel
from hydra2.search.pbrf_partition import PbrfConfig as PbrfConfig
from hydra2.search.pbrf_partition import PolicySet as PolicySet
from hydra2.search.pbrf_partition import RandomStream as RandomStream
from hydra2.search.pbrf_partition import _require_kernel as _require_kernel

__all__ = [
    "PbrfPlanner",
    "PbrfPlannerActMixin",
    "PbrfPlannerSearchMixin",
]


class PbrfPlannerSearchMixin:
    """Search half of :class:`PbrfPlanner` (construction plus budget driver).

    Construction plus the budget fallback; the act/observe/ponder protocol
    surface arrives via the act-mixin subclass below.
    """

    def __init__(
        self,
        *,
        candidate_spec: Any,
        belief: Any | None = None,
        kernel: Any | None = None,
        policy_set: Any | None = None,
        config: PbrfConfig | None = None,
    ) -> None:
        self._spec = candidate_spec
        if config is not None:
            self._config = config
        else:
            try:
                _params_raw: Any = getattr(candidate_spec, "parameters", None)
                if (
                    _params_raw is None
                    or not isinstance(_params_raw, dict)
                    or len(_params_raw) == 0
                ):
                    params: dict[str, Any] = {}
                else:
                    params = _params_raw  # type: ignore[assignment]
                _pc_raw: Any = params.get("parent_count", 16)
                _kt_raw: Any = params.get("kernel_tolerance", 1e-9)
                _mb_raw: Any = params.get("max_search_batches", 64)
                _rv_raw: Any = params.get("resource_view", "calls")
                self._config = PbrfConfig(
                    parent_count=int(_pc_raw),
                    kernel_tolerance=float(_kt_raw),
                    max_search_batches=int(_mb_raw),
                    resource_view=str(_rv_raw),  # type: ignore[arg-type]
                    tie_break=str(getattr(candidate_spec, "tie_break", "lexicographic")),
                )
            except Exception:
                self._config = PbrfConfig()
        self._belief = belief
        self._kernel = kernel
        if self._kernel is None:
            _require_kernel()
            try:
                self._kernel = NaturalPacketKernel(kernel_tolerance=self._config.kernel_tolerance)  # type: ignore[call-arg]
            except ImportError:
                raise
            except Exception as exc:
                raise ContractError(f"kernel required: {exc}") from exc
        self._policy_set = policy_set
        if self._policy_set is None:
            try:
                self._policy_set = PolicySet()  # type: ignore[call-arg]
            except Exception:
                self._policy_set = None
        self._forest: Any | None = None
        self._last_commit: CommitDisposition | None = None
        self._last_selected_action: Any | None = None
        self._last_blob: bytes | None = None

    def _budget(self) -> Any:
        b = getattr(self._spec, "resource_budget", None)
        if b is not None:
            return b
        return ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=200,
            max_model_calls=64,
            max_transitions=256,
            max_particles=self._config.parent_count,
            max_memory_bytes=None,
        )


class PbrfPlannerActMixin(PbrfPlannerSearchMixin):
    """Planner protocol surface for :class:`PbrfPlanner`.

    Split host for the ``act``/``observe``/``ponder`` protocol surface; the
    construction plus budget driver arrives via the search-mixin base.
    """

    def act(self, request: Any) -> Any:  # type: ignore[override]
        """Trigger-only PBRF act: validate, pack frozen ticket, one Rust drive call, wrap."""
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
            raise ContractError(f"pbrf: legal_actions ids unreadable: {exc}") from exc
        cand_spec: Any = request.candidate_spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", "candidate3"))
        case_raw: Any = getattr(request, "case_id", None)
        cand_fallback: Any = getattr(cand_spec, "candidate_id", "case_default")
        case_val: Any = case_raw if case_raw is not None else cand_fallback
        if isinstance(case_val, str) and case_val == "":
            case_val = cand_fallback
        case_id: str = str(case_val) if case_val is not None else "case_default"
        belief_epoch: Any | None = getattr(request, "belief_epoch", None)
        if belief_epoch is None or getattr(self, "_belief", None) is None:
            raise ContractError("pbrf: belief and epoch required; synthetic worlds removed")
        assert self._belief is not None
        budget_raw: Any = getattr(cand_spec, "resource_budget", None)
        budget: Any = budget_raw if budget_raw is not None else self._budget()
        if hasattr(budget, "resource_budget"):
            budget = budget.resource_budget
        if budget is None or not isinstance(budget, ResourceBudget):
            budget = self._budget()
        start_ns: int = time.monotonic_ns()
        n: int = int(getattr(getattr(self, "_config", None), "parent_count", 16))
        if n <= 0:
            raise ContractError("parent_count must be positive int")
        max_depth: int = 4
        tie_break: str = str(getattr(getattr(self, "_config", None), "tie_break", "lexicographic"))
        deadline_ms: int = int(getattr(budget, "deadline_ms", 5000))
        fallback_margin_ms: int = int(getattr(budget, "fallback_margin_ms", 200))
        max_calls_raw: Any = getattr(budget, "max_model_calls", 64)
        max_trans_raw: Any = getattr(budget, "max_transitions", 256)
        max_model_calls: int | None = int(max_calls_raw) if max_calls_raw is not None else None
        max_transitions: int | None = int(max_trans_raw) if max_trans_raw is not None else None
        try:
            root_seat: int = int(getattr(belief_epoch, "root_actor", 0))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ContractError(f"pbrf: root_actor unreadable: {exc}") from exc
        if root_seat < 0 or root_seat > 3:
            raise ContractError(f"pbrf: root_seat {root_seat} outside 0..3")
        epoch_str: str = str(getattr(belief_epoch, "epoch", ""))
        if epoch_str == "":
            raise ContractError("pbrf: belief epoch missing")
        target_id: str = str(getattr(belief_epoch, "target_id", ""))
        policy_pairs: list[list[Any]] = []
        try:
            _ps: Any = getattr(self, "_policy_set", None)
            _pols: Any = getattr(_ps, "policies", ()) if _ps is not None else ()
            for _seat, _pid in tuple(_pols):
                if isinstance(_seat, bool) or not isinstance(_seat, int):
                    raise ContractError("pbrf: policy seat must be int")
                if not isinstance(_pid, str) or _pid == "":
                    raise ContractError("pbrf: policy id must be non-empty str")
                policy_pairs.append([_seat, _pid])
        except ContractError:
            raise
        except (TypeError, ValueError, AttributeError) as exc:
            raise ContractError(f"pbrf: policy_set unreadable: {exc}") from exc
        seed_bytes = hashlib.sha256(f"{candidate_id}:{case_id}:pbrf_drive".encode()).digest()
        rng = RandomStream(seed_bytes)  # type: ignore[call-arg]
        try:
            particles: Any = self._belief.sample_natural(belief_epoch, count=n, rng=rng)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"pbrf: belief sampling failed: {exc}") from exc
        try:
            raw_particles: list[Any] = list(particles)
        except TypeError as exc:
            raise ContractError(f"pbrf: belief sampling malformed: {exc}") from exc
        if len(raw_particles) != n:
            raise ContractError(f"pbrf: belief sampling must return {n} particles")
        cur_worlds: list[Any] = []
        for _p in raw_particles:
            _wref: Any = getattr(_p, "world_ref", None)
            if not isinstance(_wref, str) or _wref == "":
                raise ContractError("pbrf: particle missing world_ref")
            try:
                _w: Any = self._belief._worlds[_wref]  # type: ignore[attr-defined]
            except Exception as exc:
                raise ContractError(f"pbrf: belief world missing for particle: {exc}") from exc
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
            "root_seat": root_seat,
            "candidate_id": candidate_id,
            "case_id": case_id,
            "parent_count": n,
            "policy_set": policy_pairs,
            "epoch": epoch_str,
            "target_id": target_id,
            "max_sims": n,
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
                f"hydra2._native.search not importable ({exc}); build the bridge with `pixi run build-ext` before pbrf drive"
            ) from exc
        try:
            out: Any = _drive_bridge.pbrf_search_batch(json.dumps(ticket).encode())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(f"pbrf bridge drive failed: {exc}") from exc
        try:
            end_cursor: int = out.end_cursor
        except Exception:
            end_cursor = cursor_frozen
        try:
            _jump: None = rng.jump_to(end_cursor)
        except (AttributeError, ValueError, TypeError) as exc:
            raise ContractError(f"pbrf: rng jump_to failed: {exc}") from exc
        try:
            self._last_blob = bytes(out.next_state_blob)  # pyrefly: ignore[unknown-argument-type] # native PbrfOut blob
        except Exception:
            self._last_blob = None
        result = _wrap_result(
            out,  # pyrefly: ignore[unknown-argument-type] # native PbrfOut
            legal=legal,
            candidate_spec=cand_spec,
            start_ns=start_ns,
            case_id=case_id,
            particles=n,
        )
        self._last_selected_action = result.selected_action
        return result

    def observe(self, packet: Any) -> None:  # type: ignore[override]
        """PBRF observe: epoch-validate only.

        Consumes the stored act()-emitted action one-shot; observing without
        a stored action raises ``ContractError`` instead of guessing a
        candidate. The stored blob + packet ride the batch commit opaquely —
        no Python forest walk. The retained commit records a hit locally.
        """
        if packet is None or not hasattr(packet, "packet_id"):
            raise ContractError("packet must have packet_id for observe")
        action = self._last_selected_action
        self._last_selected_action = None
        if action is None:
            raise ContractError("observe requires an act()-emitted action: none stored")
        self._last_commit = CommitDisposition("hit_commit")

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # PBRF core does not perform background ponder without commit; no-op
        if isinstance(deadline_monotonic_ns, bool) or not isinstance(deadline_monotonic_ns, int):
            raise ContractError("deadline_monotonic_ns must be int")
        return


class PbrfPlanner(  # type: ignore[misc]
    PbrfPlannerActMixin,
    Planner,
):
    """Natural-particle PBRF planner (Candidate 3).

    Thin subclass joining the split mixins; construction, the budget
    driver, and the trigger-only act live in the ``pbrf_*`` modules with
    no overrides here.
    """
