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
import json
import math
import time
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry
from hydra2.search._drive_trigger import _aid_for_trigger as _trigger_aid
from hydra2.search._drive_trigger import wrap_result as _wrap_result
from hydra2.search.common import SearchRequest, SearchResult
from hydra2.search.persistence_kernel import ForestState as ForestState
from hydra2.search.persistence_kernel import PersistenceArm as PersistenceArm
from hydra2.search.persistence_kernel import _distribute_quota as _distribute_quota
from hydra2.search.persistence_kernel import make_persistence_arm as make_persistence_arm
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
        # Opaque retain blob owned by Rust (stateless-plus-blob): stored and
        # passed back verbatim each act, never inspected in Python.
        self._forest_blob: bytes = b""
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
        # Retain lives in Rust behind the opaque blob (stateless-plus-blob):
        # R/P with non-empty blob retain, B/F/C never retain.
        if self.arm.id in ("R", "P"):
            return len(getattr(self, "_forest_blob", b"")) > 0
        return False

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
        # Bridge-owned final hash (owner ``epoch_for_obs``), oracle fallback.
        if isinstance(oh, str) and self.arm.id in ("B", "F", "R", "P", "C"):
            try:
                from hydra2._native import (
                    search as _epoch_bridge,  # pyrefly: ignore[missing-import]
                )

                _fn = _epoch_bridge.persistence_epoch_for_obs  # type: ignore[attr-defined]  # reason: persistence_kernel leaf lands with MAIN wiring; AttributeError fallback covers stale .so
            except (ImportError, AttributeError):
                pass  # stale .so: fall through to the oracle below (same bytes)
            else:
                try:
                    return _fn(self.arm.id, oh)  # pyrefly: ignore[unknown-argument-type] # untyped bridge epoch fn
                except (ValueError, TypeError) as exc:
                    raise ContractError(f"persistence epoch rejected: {exc}") from exc
        raw = f"{oh}:{self.arm.id}".encode()
        return "epoch:" + hashlib.sha256(raw).hexdigest()[:16]

    # ---- Planner protocol --------------------------------------------------------

    def act(self, request: SearchRequest) -> SearchResult:
        """Trigger-only persistence act: validate, pack frozen ticket, one Rust drive call, wrap.

        Rust owns per-arm retain rules (B/F/C destroy, R/P retain-or-squash),
        the bounded pick, and all counters via ``persistence_search_batch``.
        Python only validates, packs the ticket (epoch hash owner kept
        verbatim), crosses once, stores the opaque blob, and wraps the OUT.
        """
        t0 = time.monotonic_ns()
        observation = request.observation
        legal_actions = request.legal_actions
        # Determine current epoch (hash owner, kept verbatim — not driving).
        cur_epoch = self._new_epoch_for_obs(observation)
        # Validate legal_actions non-empty via SearchRequest but also check here.
        if not isinstance(legal_actions, tuple) or len(legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        # Int ids via the shared trigger mapping (bit-exact with wrap_result);
        # sort legal by aids, never leave aids empty.
        try:
            _aids_raw: list[Any] = [_trigger_aid(_a) for _a in legal_actions]  # pyrefly: ignore[unknown-argument-type] # Any legal action
            aids: list[int] = _aids_raw
            if len(aids) != len(set(aids)):
                raise ContractError("legal_actions must have unique action_ids")
            legal: tuple[Any, ...] = tuple(legal_actions)
            if aids != sorted(aids):

                def _sort_key(pair: tuple[int, Any]) -> int:
                    return pair[0]

                paired = sorted(zip(aids, legal, strict=False), key=_sort_key)
                legal = tuple(p for _, p in paired)
                aids = sorted(aids)
        except ContractError:
            raise
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError(f"persistence: legal_actions ids unreadable: {exc}") from exc
        cand_spec: Any = request.candidate_spec
        candidate_id: str = str(getattr(cand_spec, "candidate_id", f"persistence-{self.arm.id}"))
        case_raw: Any = getattr(request, "case_id", None)
        cand_fallback: Any = getattr(cand_spec, "candidate_id", "case_default")
        case_val: Any = case_raw if case_raw is not None else cand_fallback
        if isinstance(case_val, str) and case_val == "":
            case_val = cand_fallback
        case_id: str = str(case_val) if case_val is not None else "case_default"
        # Budget from candidate spec (spec defaults already per-arm).
        rb: Any = self.candidate_spec.resource_budget
        budget_calls: int = (
            rb.max_model_calls
            if rb.max_model_calls is not None
            else (1 if self.arm.id == "B" else 32)
        )
        budget_trans: int = (
            rb.max_transitions
            if rb.max_transitions is not None
            else (0 if self.arm.id == "B" else 128)
        )
        # Deadline enforcement: deadline_monotonic_ns in request is authoritative
        # if set; past-deadline falls back immediately without crossing.
        deadline_ns = getattr(request, "deadline_monotonic_ns", None)
        if deadline_ns is not None and isinstance(deadline_ns, int):
            remaining_ms = (deadline_ns - t0) / 1e6
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
        # Observation hash for the ticket (shape-checked arena-side).
        oh_attr: Any = getattr(observation, "observation_hash", None)
        if oh_attr is None and isinstance(observation, dict):
            oh_attr = observation.get("observation_hash")
        if oh_attr is None:
            obs_hash = (
                "sha256:"
                + hashlib.sha256(
                    canonical_bytes({"obs": str(observation), "arm": self.arm.id})
                ).hexdigest()
            )
        else:
            obs_hash = str(oh_attr)
        rules_hash: str = str(getattr(cand_spec, "rules_hash", ""))
        # Control-plane ticket: worlds unused arena-side (shape-validated only);
        # the forest rides the opaque blob, never a live handle.
        ticket = {
            "worlds": [],
            "rules_hash": rules_hash,
            "observation_hash": obs_hash,
            "root_legal": list(aids),
            "candidate_id": candidate_id,
            "case_id": case_id,
            "arm_id": self.arm.id,
            "epoch": cur_epoch,
            "forest_blob": list(self._forest_blob),
            "ponder_quota": None,
            "max_sims": 1,
            "max_depth": 4,
            "max_transitions": budget_trans,
            "max_model_calls": budget_calls,
            "deadline_ms": int(self.deadline_ms),
            "fallback_margin_ms": int(self.fallback_margin_ms),
            "tie_break": "lexicographic",
            "seed_hex": hashlib.sha256(
                f"{self.arm.id}:{case_id}:persist_drive".encode()
            ).hexdigest(),
            "cursor": 0,
        }
        try:
            from hydra2._native import search as _drive_bridge  # pyrefly: ignore[missing-import]
        except ImportError as exc:
            raise ImportError(
                "hydra2._native.search not importable "
                f"({exc}); build the bridge with `pixi run build-ext` before persistence drive"
            ) from exc
        try:
            out: Any = _drive_bridge.persistence_search_batch(json.dumps(ticket).encode())
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(f"persistence bridge drive failed: {exc}") from exc
        try:
            calls: int = out.model_calls
            trans: int = out.transitions
            blob: bytes = out.next_state_blob
        except Exception as exc:
            raise ContractError(f"persistence: OUT malformed: {exc}") from exc
        # Counters from Rust only, plus elapsed measured once for the report.
        self._forest_blob = blob
        self._current_epoch = cur_epoch
        self._last_emitted_epoch = cur_epoch
        try:
            self._last_emitted_action = next(
                a for a, aid in zip(legal, aids, strict=True) if aid == out.selected_id
            )
        except Exception:
            self._last_emitted_action = legal[0]
        self._total_model_calls += calls
        self._total_transitions += trans
        joules = float(calls) * 0.04 + float(trans) * 0.005
        self._total_joules += joules
        # Retain rides the opaque blob only (never a Python forest mirror):
        # R/P keep the Rust-owned blob for the next act, B/F/C clear both.
        if self.arm.id in ("R", "P") and out.completed:
            pass
        else:
            self._forest = None
            if self.arm.id in ("B", "F", "C"):
                self._forest_blob = b""
        return _wrap_result(
            out,  # pyrefly: ignore[unknown-argument-type] # native PersistenceOut
            legal=legal,
            candidate_spec=cand_spec,
            start_ns=t0,
            case_id=case_id,
            energy_joules=joules,
            particles=1,
        )

    def observe(self, packet: Any) -> None:
        """Observe the next actor-visible packet (epoch-validate only + blob commit).

        Rust owns the forest: Python only squashes or keeps the opaque blob
        and records the commit outcome. No forest walk, no rebuild math.

        - B/F/C: squash the blob (hard invariant: never retain).
        - R/P: empty blob means nothing retained (miss + recovery); a stored
          blob kept on epoch continuity (hit), squashed on a stale
          ``epoch_before`` (miss + recovery).
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
        epoch_after: object = getattr(
            cast("object", packet), "epoch_after", cast("object", packet_id)
        )
        if self.arm.id in ("B", "F", "C"):
            if self._forest is not None and not self._forest.is_empty():
                self._forest.clear()
                self._forest = None
            self._forest_blob = b""
            self._current_epoch = str(epoch_after)
            self._commit_log.append(
                {"arm": self.arm.id, "packet_id": packet_id, "outcome": "fresh", "ponder_calls": 0}
            )
            return
        assert self.arm.id in ("R", "P")
        if len(self._forest_blob) == 0:
            # Nothing retained (e.g., fallback path) — treat as rebuild.
            self._surprise_counts["miss"] += 1
            self._surprise_counts["recovery"] += 1
            self._current_epoch = str(epoch_after)
            self._commit_log.append(
                {
                    "arm": self.arm.id,
                    "packet_id": packet_id,
                    "outcome": "rebuild_no_forest",
                    "ponder_calls": 0,
                }
            )
            return
        epoch_before: Any = getattr(cast("object", packet), "epoch_before", None)
        if epoch_before is not None and str(epoch_before) != str(self._current_epoch):
            # Stale provenance — squash the blob, count miss + recovery.
            self._surprise_counts["miss"] += 1
            self._surprise_counts["recovery"] += 1
            self._current_epoch = str(epoch_after)
            self._commit_log.append(
                {
                    "arm": self.arm.id,
                    "packet_id": packet_id,
                    "outcome": "miss_recovery",
                    "ponder_calls": self._ponder_budget_used,
                }
            )
            if self._forest is not None:
                self._forest.clear()
                self._forest = None
            self._forest_blob = b""
            return
        # Hit: keep the blob for the next act's commit path.
        self._surprise_counts["hit"] += 1
        self._commit_log.append(
            {
                "arm": self.arm.id,
                "packet_id": packet_id,
                "outcome": "hit",
                "ponder_calls": self._ponder_budget_used,
            }
        )
        self._current_epoch = str(epoch_after)

    def ponder(self, *, deadline_monotonic_ns: int, ponder_quota_total: int | None = None) -> None:
        """Opponent-time compute — allowed only for P, and only between action and next packet.

        Blob-only: ponder work is counted, never a forest walk. R/B/F/C do zero
        work. P with a retained blob charges bounded speculative work.
        """
        if ponder_quota_total is not None and (
            isinstance(ponder_quota_total, bool)
            or not isinstance(ponder_quota_total, int)
            or ponder_quota_total <= 0
        ):
            raise ContractError("ponder_quota_total must be a positive int or None")
        if self.arm.id in ("B", "F", "R", "C"):
            return
        assert self.arm.id == "P"
        if len(getattr(self, "_forest_blob", b"")) == 0:
            return
        if self._last_emitted_action is None:
            return
        now = time.monotonic_ns()
        remaining_ms = (deadline_monotonic_ns - now) / 1e6
        if remaining_ms <= 0:
            return
        total_ponder = 4
        if remaining_ms < 0.5 and total_ponder > 1:
            total_ponder = 1
        if ponder_quota_total is not None:
            total_ponder = ponder_quota_total
        self._total_model_calls += total_ponder
        self._total_transitions += total_ponder // 2
        self._ponder_budget_used += total_ponder
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
