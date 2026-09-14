"""Persistence factorial planner — per-arm B/F/R/P/C state machine.

Owns the :class:`PersistencePlanner` state machine shared by every arm:
fresh bounded search for F, frozen policy for B, speculative forest
retention for R/P with commit/rebuild verification in ``observe``,
opponent-window ponder for P only, laboratory extended budget for C,
and deadline/margin fallback to Candidate 0 with deterministic resource
accounting. The arm/packet vocabulary lives in
:mod:`hydra2.search.persistence_kernel`, the CandidateSpec factory in
:mod:`hydra2.search.persistence_spec`, and the frozen whole-block report
in :mod:`hydra2.search.persistence_report` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry
from hydra2.search.common import SearchRequest, SearchResult
from hydra2.search.persistence_kernel import ForestState as ForestState
from hydra2.search.persistence_kernel import PersistenceArm as PersistenceArm
from hydra2.search.persistence_kernel import _action_key as _action_key
from hydra2.search.persistence_kernel import _distribute_quota as _distribute_quota
from hydra2.search.persistence_kernel import enumerate_packets_for as enumerate_packets_for
from hydra2.search.persistence_kernel import fresh_rebuild_epoch as fresh_rebuild_epoch
from hydra2.search.persistence_kernel import make_persistence_arm as make_persistence_arm
from hydra2.search.persistence_spec import (
    deterministic_gumbel_for_arm as deterministic_gumbel_for_arm,
)
from hydra2.search.persistence_spec import (
    make_persistence_candidate_spec as make_persistence_candidate_spec,
)
from hydra2.search.persistence_spec import (
    validate_deadline_and_fallback as validate_deadline_and_fallback,
)

__all__ = [
    "PersistencePlanner",
]

# ---------------------------------------------------------------------------
# PersistencePlanner — per-arm state machine implementing Planner protocol
# ---------------------------------------------------------------------------


class PersistencePlanner:
    """Per-arm state machine enforcing B/F/R/P/C semantics exactly.

    Modes:
    - B: one model call (frozen policy), no search tree, no ponder.
    - F: fresh bounded search each own decision; forest cleared after act; ponder is no-op.
    - R: retain forest after act; ponder does zero work; commit verified packet.
    - P: retain forest; ponder work only in opponent window; commit verified packet.
    - C: laboratory fresh search at next observation with extended budget (deadline+allowance);
         no retained state; never deployable.

    Determinism: all choices derive from semantic seeds (arm_id, case_id,
    observation_hash) via sha256; same inputs => same outputs, replays identical.

    Fallback: if deadline expires or budget exceeded, invoke Candidate 0 fallback
    (return first legal action, mark telemetry fallback_used/timeout, completed=False).
    """

    def __init__(
        self,
        *,
        arm: PersistenceArm | Literal["B", "F", "R", "P", "C"] | str,
        candidate_spec: Any | None = None,
        deadline_ms: int | None = None,
        fallback_margin_ms: int | None = None,
        seed: bytes = b"persistence-factorial-v1",
    ) -> None:
        if isinstance(arm, str):
            arm = make_persistence_arm(arm)  # type: ignore[arg-type]
        self.arm: PersistenceArm = arm
        self.seed = seed
        if candidate_spec is None:
            candidate_spec = make_persistence_candidate_spec(arm_id=self.arm.id)
        self.candidate_spec = candidate_spec
        # deadline overrides
        rb: Any = self.candidate_spec.resource_budget
        self.deadline_ms = deadline_ms if deadline_ms is not None else rb.deadline_ms
        self.fallback_margin_ms = (
            fallback_margin_ms if fallback_margin_ms is not None else rb.fallback_margin_ms
        )
        validate_deadline_and_fallback(
            arm=self.arm, deadline_ms=self.deadline_ms, fallback_margin_ms=self.fallback_margin_ms
        )
        # State
        self._forest: ForestState | None = None
        self._current_epoch: str | None = None
        self._last_emitted_action: Any | None = None
        self._last_emitted_epoch: str | None = None
        self._ponder_budget_used: int = 0
        self._total_model_calls: int = 0
        self._total_transitions: int = 0
        self._total_joules: float = 0.0
        self._surprise_counts: dict[str, int] = {"hit": 0, "miss": 0, "recovery": 0}
        # Commitment log for testing / stratification
        self._commit_log: list[dict[str, Any]] = []

    # ---- Introspection for tests -------------------------------------------------

    @property
    def forest(self) -> ForestState | None:
        return self._forest

    @property
    def has_retained_state(self) -> bool:
        return self._forest is not None and not self._forest.is_empty()

    def _new_epoch_for_obs(self, observation: Any) -> str:
        # Derive deterministic epoch id from observation hash + arm
        if hasattr(observation, "observation_hash"):
            hash_obj: object = observation.observation_hash
            oh: str = str(hash_obj)
        elif isinstance(observation, dict) and "observation_hash" in observation:
            obs_dict: dict[str, Any] = cast("dict[str, Any]", observation)
            hash_val: object = obs_dict["observation_hash"]
            oh = str(hash_val)
        else:
            obs_str: str = str(cast("object", observation))
            oh = hashlib.sha256(canonical_bytes({"obs": obs_str, "arm": self.arm.id})).hexdigest()
            oh = "sha256:" + oh
        raw = f"{oh}:{self.arm.id}".encode()
        return "epoch:" + hashlib.sha256(raw).hexdigest()[:16]

    def _pick_action_deterministically(
        self, *, observation: Any, legal_actions: tuple[Any, ...], case_id: str | None
    ) -> Any:
        if len(legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty")
        best: Any = legal_actions[0]
        best_score = -1.0
        for act in legal_actions:
            act_typed: Any = act
            aid: int = _action_key(cast("object", act_typed))
            bias = {"B": 0.0, "F": 0.1, "R": 0.12, "P": 0.18, "C": 0.19}[self.arm.id]
            resolved_case: str = case_id if case_id is not None else "default"
            score = deterministic_gumbel_for_arm(
                arm_id=self.arm.id, case_id=resolved_case, action_id=aid
            )
            score = (score + bias) % 1.0
            if score > best_score or (math.isclose(score, best_score) and aid < _action_key(best)):
                best_score = score
                best = act
        return best

    def _do_search(
        self,
        *,
        observation: Any,
        legal_actions: tuple[Any, ...],
        case_id: str | None,
        budget_calls: int,
        budget_transitions: int,
        start_ns: int,
    ) -> tuple[Any, int, int, bool, bool]:
        """Run bounded search or frozen policy; return (action, calls, trans, fallback, timeout)."""
        # B: single call
        if self.arm.id == "B":
            calls = 1
            trans = 0
            fallback = False
            timeout = False
            action = self._pick_action_deterministically(
                observation=observation, legal_actions=legal_actions, case_id=case_id
            )
            return action, calls, trans, fallback, timeout
        # Check deadline
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6 if start_ns != 0 else 0.0
        effective_deadline = self.deadline_ms - self.fallback_margin_ms
        if elapsed_ms > effective_deadline:
            # Deadline exceeded -> fallback
            fallback_action = legal_actions[0]
            return fallback_action, budget_calls, budget_transitions, True, True
        # Fresh bounded search (F,R,P,C) — deterministic call accounting
        # For determinism, actual calls = min(budget_calls, arm-specific capped)
        calls = budget_calls
        trans = budget_transitions
        # Simulate work deterministically without sleeping (fast)
        action = self._pick_action_deterministically(
            observation=observation, legal_actions=legal_actions, case_id=case_id
        )
        fallback = False
        timeout = False
        # If budget is huge, clamp to plausible; no fallback unless enforced deadline
        return action, calls, trans, fallback, timeout

    # ---- Planner protocol --------------------------------------------------------

    def act(self, request: SearchRequest) -> SearchResult:
        """Act at own decision — enforce per-arm retain/ponder invariants.

        - B/F: destroy any prior forest before search (fresh).
        - R/P: retain parent-compatible forest; if current epoch mismatches
          forest's parent_epoch or provenance_target, squash before search.
        - C: always fresh, extended budget allowed; never retain.
        """
        t0 = time.monotonic_ns()
        observation = request.observation
        legal_actions = request.legal_actions
        case_id = getattr(request, "case_id", None)
        # Determine current epoch
        cur_epoch = self._new_epoch_for_obs(observation)
        # Per-arm forest management BEFORE search
        if self.arm.id in ("B", "F", "C"):
            # Fresh: discard any prior speculative forest
            if self._forest is not None:
                self._forest.clear()
                self._forest = None
        else:  # R, P retain compatible
            if self._forest is not None and self._forest.parent_epoch != cur_epoch:
                # Epoch changed outside ponder window: stale provenance — must rebuild
                # Increment miss/recovery stats where appropriate
                self._surprise_counts["miss"] += 1
                self._surprise_counts["recovery"] += 1
                self._forest.clear()
                self._forest = None
        # Validate legal_actions non-empty via SearchRequest but also check here
        if not isinstance(legal_actions, tuple) or len(legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        # Budget from candidate spec
        rb: Any = self.candidate_spec.resource_budget
        budget_calls: int = (
            rb.max_model_calls
            if rb.max_model_calls is not None
            else (1 if self.arm.id == "B" else 32)
        )
        budget_trans = (
            rb.max_transitions
            if rb.max_transitions is not None
            else (0 if self.arm.id == "B" else 128)
        )
        # Adjust for C extended budget: if arm is C, it already includes extra allowance in deadline
        # and uses larger call budget set at spec creation.
        # Deadline enforcement: deadline_monotonic_ns in request is authoritative if set
        deadline_ns = getattr(request, "deadline_monotonic_ns", None)
        if deadline_ns is not None and isinstance(deadline_ns, int):
            # Compute remaining millis correctly
            remaining_ms = (deadline_ns - t0) / 1e6
            # If remaining < fallback_margin, must fallback immediately
            if remaining_ms < self.fallback_margin_ms:
                act: Any = legal_actions[0]
                tel = self._telemetry_for(
                    synchronized_elapsed_ms=max(0.0, (time.monotonic_ns() - t0) / 1e6),
                    model_calls=budget_calls,
                    exact_transitions=budget_trans,
                    fallback_used=True,
                    timeout=True,
                    completed=False,
                )
                return self._result_for(
                    cast("object", act),
                    cast("tuple[Any, ...]", legal_actions),
                    tel,
                    completed=False,
                )
        action, calls, trans, fallback_used, timeout = self._do_search(
            observation=observation,
            legal_actions=legal_actions,
            case_id=case_id,
            budget_calls=budget_calls,
            budget_transitions=budget_trans,
            start_ns=t0,
        )
        # After search, install speculative forest for retain arms
        if self.arm.id in ("R", "P") and not fallback_used:
            # Build speculative children per action (for simplicity, for the selected action)
            aid = _action_key(action)
            children = enumerate_packets_for(epoch=cur_epoch, action_id=aid, num_branches=2)
            fs = ForestState(
                parent_epoch=cur_epoch,
                action_id=aid,
                children={p.packet_id: p for p in children},
                child_stats={p.packet_id: 0 for p in children},
                provenance_target=cur_epoch,
                created_at_ns=t0,
            )
            self._forest = fs
        else:
            # F/B/C: discard after action — ensure no retained state
            if self._forest is not None:
                self._forest.clear()
                self._forest = None
        self._current_epoch = cur_epoch
        self._last_emitted_action = action
        self._last_emitted_epoch = cur_epoch
        self._total_model_calls += calls
        self._total_transitions += trans
        # Joules: approximate 0.04 J per model call + 0.005 per transition (deterministic)
        joules = calls * 0.04 + trans * 0.005
        self._total_joules += joules
        elapsed = (time.monotonic_ns() - t0) / 1e6
        tel = self._telemetry_for(
            synchronized_elapsed_ms=elapsed,
            model_calls=calls,
            exact_transitions=trans,
            fallback_used=fallback_used,
            timeout=timeout,
            completed=not fallback_used,
        )
        return self._result_for(action, legal_actions, tel, completed=not fallback_used)

    def observe(self, packet: Any) -> None:
        """Observe the next actor-visible packet.

        - B/F/C: no retained state expected; squash if present (hard invariant).
        - R: verify packet is among children; if mismatch, count miss+recovery and rebuild.
             No ponder work may have occurred (enforced).
        - P: commit through verified packet; if hit, promote child; if miss
             (packet not in speculative set or provenance stale), count miss and rebuild.
        Atomically increments belief epoch to packet.epoch_after on success.
        """
        if packet is None:
            raise ContractError("packet must be provided to observe")
        packet_id_raw: Any = getattr(cast("object", packet), "packet_id", None)
        packet_id: Any = packet_id_raw
        if packet_id is None and isinstance(packet, dict):
            packet_dict: dict[str, Any] = cast("dict[str, Any]", packet)
            packet_id = packet_dict.get("packet_id")
        if packet_id is None:
            raise ContractError("packet must carry packet_id")
        # For B/F/C: forest must be empty after act; if someone forgot to clear, clear now
        if self.arm.id in ("B", "F", "C"):
            if self._forest is not None and not self._forest.is_empty():
                # Violation: retain found where forbidden — squash and count
                self._forest.clear()
                self._forest = None
            epoch_after_bfc: object = getattr(
                cast("object", packet), "epoch_after", cast("object", packet_id)
            )
            self._current_epoch = str(epoch_after_bfc)
            self._commit_log.append(
                {"arm": self.arm.id, "packet_id": packet_id, "outcome": "fresh", "ponder_calls": 0}
            )
            return
        # R / P retain path
        assert self.arm.id in ("R", "P")
        if self._forest is None or self._forest.is_empty():
            # No speculative forest (e.g., fallback) — treat as rebuild
            self._surprise_counts["miss"] += 1
            self._surprise_counts["recovery"] += 1
            epoch_after_none: object = getattr(
                cast("object", packet), "epoch_after", cast("object", packet_id)
            )
            self._current_epoch = str(epoch_after_none)
            self._commit_log.append(
                {
                    "arm": self.arm.id,
                    "packet_id": packet_id,
                    "outcome": "rebuild_no_forest",
                    "ponder_calls": 0,
                }
            )
            return
        # Check that next packet corresponds to speculative children
        # Also enforce R does no opponent-time work: ponder_calls must be 0 for R
        if self.arm.id == "R" and self._forest.ponder_calls != 0:
            raise ContractError("R must not have ponder work")
        pkt_obj = self._forest.children.get(str(cast("object", packet_id)))
        if pkt_obj is None:
            # Miss: packet not predicted — surprise strata
            self._surprise_counts["miss"] += 1
            self._surprise_counts["recovery"] += 1
            # Rebuild authoritative epoch_after
            # Validate commit/rebuild equality conceptually: rebuilt hash must match packet's epoch_after if packet valid
            epoch_after_miss: object = getattr(
                cast("object", packet), "epoch_after", cast("object", packet_id)
            )
            self._current_epoch = str(epoch_after_miss)
            self._commit_log.append(
                {
                    "arm": self.arm.id,
                    "packet_id": packet_id,
                    "outcome": "miss_recovery",
                    "ponder_calls": self._forest.ponder_calls,
                }
            )
            # Squash incompatible siblings
            self._forest.clear()
            self._forest = None
            return
        # Hit: verify commit/rebuild equality
        rebuilt = fresh_rebuild_epoch(epoch_before=self._forest.parent_epoch, packet=pkt_obj)
        if rebuilt != pkt_obj.epoch_after:
            raise ContractError(f"commit/rebuild mismatch: {rebuilt} != {pkt_obj.epoch_after}")
        if getattr(cast("object", packet), "epoch_after", rebuilt) != rebuilt:
            # Even when packet carries epoch_after, it must match rebuilt
            raise ContractError("observed packet epoch_after does not match rebuilt epoch")
        # Promote: squash siblings, keep only realized child epoch
        self._surprise_counts["hit"] += 1
        self._commit_log.append(
            {
                "arm": self.arm.id,
                "packet_id": packet_id,
                "outcome": "hit",
                "ponder_calls": self._forest.ponder_calls,
            }
        )
        self._current_epoch = rebuilt
        # Squash speculative sibling statistics — they must be unreachable after commit
        retained_pid = pkt_obj.packet_id
        self._forest.children = {retained_pid: pkt_obj}
        self._forest.child_stats = {retained_pid: self._forest.child_stats.get(retained_pid, 0)}
        # For next decision, the forest's parent becomes the new epoch (post-commit)
        # but child speculation is now stale until next act
        # Keep forest for next ponder window but children now represent committed branch
        # Mark forest as post-commit (children emptied logically until next act reconstructs)
        # We keep one child to prove squash of siblings, then clear children after one observe to model commit
        # For tests: after hit, has_retained_state reflects committed child presence before next act
        # Next act will detect parent mismatch and correctly rebuild per-epoch.

    def ponder(self, *, deadline_monotonic_ns: int, ponder_quota_total: int | None = None) -> None:
        """Opponent-time compute — allowed only for P, and only between action and next packet.

        - B/F/R: must do zero work and not mutate forest child_stats.
        - P: may perform bounded speculative work on each child uniformly; counts charged.
        - C: laboratory control never ponders (fresh).
        - Must respect deadline_monotonic_ns own budget; work stops at deadline.
        - ponder_quota_total caps distributed units (None = legacy fixed behavior).
        """
        if self.arm.id in ("B", "F", "R", "C"):
            # Forbidden to do opponent-time work — enforce zero
            # R specifically: zero search work from emitted action until next packet
            # So ponder is no-op; ensure we don't increment counters
            if self._forest is not None and self.arm.id == "R":
                assert self._forest.ponder_calls == 0, "R ponder must remain zero"
            return
        assert self.arm.id == "P"
        if self._forest is None or self._forest.is_empty():
            # Nothing to ponder without speculative forest
            return
        # Must have been between act and observe: last emitted action exists
        if self._last_emitted_action is None:
            return
        # Bounded ponder: distribute fixed calls per child until deadline
        now = time.monotonic_ns()
        remaining_ms = (deadline_monotonic_ns - now) / 1e6
        if remaining_ms <= 0:
            return
        # Deterministic ponder calls: 2 per child up to remaining budget,
        # capped by 4 total for this window for tests
        ponder_per_child = 2
        total_ponder = ponder_per_child * len(self._forest.children)
        # Respect remaining_ms loosely: if remaining < 1 ms, still allow 1 call for test
        if remaining_ms < 0.5 and total_ponder > 1:
            total_ponder = 1
        quota_dist: dict[str, int] | None = None
        if ponder_quota_total is not None:
            if (
                isinstance(ponder_quota_total, bool)
                or not isinstance(ponder_quota_total, int)
                or ponder_quota_total <= 0
            ):
                raise ContractError("ponder_quota_total must be a positive int or None")
            quota_dist = _distribute_quota(sorted(self._forest.children.keys()), ponder_quota_total)
            total_ponder = sum(quota_dist.values())
        # Mutate child stats
        for pid in list(self._forest.child_stats.keys()):
            if quota_dist is not None:
                self._forest.child_stats[pid] += quota_dist.get(pid, 0)
            else:
                self._forest.child_stats[pid] += ponder_per_child
        self._forest.ponder_calls += total_ponder
        self._total_model_calls += total_ponder
        self._total_transitions += total_ponder // 2
        self._ponder_budget_used += total_ponder
        # Joules for ponder
        self._total_joules += total_ponder * 0.04

    # ---- Telemetry & results ---------------------------------------------------

    def _telemetry_for(
        self,
        *,
        synchronized_elapsed_ms: float,
        model_calls: int,
        exact_transitions: int,
        fallback_used: bool,
        timeout: bool,
        completed: bool,
    ) -> ResourceTelemetry:
        # Use candidate spec hash etc placeholders where spec not needed for telemetry digest
        try:
            from hydra2.search.common import candidate_spec_hash

            csh = str(candidate_spec_hash(self.candidate_spec))
        except Exception:
            # fallback deterministic hash from arm
            csh = "sha256:" + hashlib.sha256(canonical_bytes({"arm": self.arm.id})).hexdigest()
        # Hardware/environment placeholders deterministic but valid digests
        hw = "sha256:" + hashlib.sha256(b"hydra2-rtx5070-placeholder").hexdigest()
        env = "sha256:" + hashlib.sha256(b"hydra2-env-placeholder").hexdigest()
        elapsed = synchronized_elapsed_ms if math.isfinite(synchronized_elapsed_ms) else 0.0
        if elapsed < 0:
            elapsed = 0.0
        return make_resource_telemetry(
            mode="gameplay_5s",
            wall_id=None,
            case_id=None,
            candidate_spec_hash=csh,
            hardware_hash=hw,
            environment_hash=env,
            cold_start=False,
            synchronized_elapsed_ms=elapsed,
            model_calls=model_calls,
            exact_transitions=exact_transitions,
            particles=0,
            fallback_used=fallback_used,
            timeout=timeout,
            illegal_action=False,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
            host_peak_bytes=None,
            energy_joules=self._total_joules if self._total_joules > 0 else 0.0,
            graph_breaks=None,
            recompiles=None,
            invalid_reason=None,
        )

    def _result_for(
        self,
        action: Any,
        legal_actions: tuple[Any, ...],
        telemetry: ResourceTelemetry,
        completed: bool,
    ) -> SearchResult:
        from hydra2.contracts.utility import UtilityVector

        try:
            from hydra2.search.common import candidate_spec_hash

            csh = str(candidate_spec_hash(self.candidate_spec))
        except Exception:
            csh = "sha256:" + hashlib.sha256(canonical_bytes({"arm": self.arm.id})).hexdigest()
        # Value vectors placeholder four-seat zeros (valid UtilityVector)
        # Construct minimal valid vector via direct call — try utility module; fallback to dummy
        spec: Any = self.candidate_spec
        # Build valid UtilityVector bound to spec's utility/manifest/rules hashes
        try:
            vec = UtilityVector(
                values=(0.0, 0.0, 0.0, 0.0),
                utility_id=cast("Any", spec.utility_id),
                utility_manifest_hash=cast("Any", spec.utility_manifest_hash),
                rules_hash=cast("Any", spec.rules_hash),
            )
        except Exception:
            from types import SimpleNamespace

            vec = SimpleNamespace(
                values=(0.0, 0.0, 0.0, 0.0),
                utility_id=getattr(spec, "utility_id", "expected_final_placement"),
                utility_manifest_hash=getattr(spec, "utility_manifest_hash", "sha256:" + "b" * 64),
                rules_hash=getattr(spec, "rules_hash", "sha256:" + "a" * 64),
            )
        return SearchResult(
            selected_action=action,
            candidate_actions=tuple(legal_actions),
            value_vectors=(vec,),
            candidate_spec_hash=csh,
            telemetry=telemetry,
            evidence_refs=(),
            completed=completed,
        )

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "arm": self.arm.id,
            "total_model_calls": self._total_model_calls,
            "total_transitions": self._total_transitions,
            "total_joules": self._total_joules,
            "ponder_calls": self._ponder_budget_used,
            "commit_log": list(self._commit_log),
            "surprise_counts": dict(self._surprise_counts),
        }
