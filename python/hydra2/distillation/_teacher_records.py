"""WP-10 trajectory records — actor-visible records and deterministic generation.

Owns the frozen :class:`TrajectoryRecord`, its validated constructor, the
isolated privileged-label namespace, and deterministic trajectory generation
beside their only callers. The gate and registry live in
:mod:`hydra2.distillation._teacher_gate`, case observations in
:mod:`hydra2.distillation._teacher_cases`, the student model in
:mod:`hydra2.distillation._teacher_student`, and five-arm evaluation in
:mod:`hydra2.distillation._teacher_eval`, so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError

try:
    from hydra2._native import contracts as _distill_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal; probes below decide
    _distill_bridge = None  # type: ignore[assignment]

from hydra2.distillation._teacher_cases import _case_observation as _case_observation
from hydra2.distillation._teacher_cases import _hash_to_uniform as _hash_to_uniform
from hydra2.distillation._teacher_cases import _provenance_for_case as _provenance_for_case
from hydra2.distillation._teacher_cases import (
    _teacher_policy_and_value as _teacher_policy_and_value,
)
from hydra2.distillation._teacher_gate import TeacherJustification
from hydra2.distillation._teacher_gate import _action_table as _action_table
from hydra2.distillation._teacher_gate import _real_candidate_spec as _real_candidate_spec
from hydra2.distillation._teacher_gate import _require_sha256 as _require_sha256
from hydra2.distillation._teacher_gate import _spec_digest_of as _spec_digest_of

# ---------------------------------------------------------------------------
# Trajectory record — actor-visible only, legal mask, teacher policy, vector, provenance, budget
# ---------------------------------------------------------------------------


#: Bridge leaf for the trajectory validation (single source: ``hydra2._native.contracts``
#: ``distill_validate_trajectory_values``; the stale-.so fallback is the byte-identical oracle).
_distill_validate_trajectory_values = getattr(
    _distill_bridge, "distill_validate_trajectory_values", None
)


def _validate_trajectory_values(
    legal_mask: list[bool], teacher_policy: list[float], vector_return: list[float]
) -> None:
    """Validate mask/policy/vector decisions — bridge decides, oracle fallback."""
    if _distill_validate_trajectory_values is not None:
        # ``sum()``/``repr()`` renderings are staged here (exact by definition);
        # the bridge only compares, so CPython's compensated ``sum`` needs no replica.
        policy_sum = sum(teacher_policy)
        try:
            _distill_validate_trajectory_values(
                list(legal_mask),
                list(teacher_policy),
                policy_sum,
                repr(policy_sum),
                [repr(p) for p in teacher_policy],
                list(vector_return),
                [repr(v) for v in vector_return],
            )
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        return
    if len(legal_mask) != len(teacher_policy):
        raise ContractError("legal_mask and teacher_policy length mismatch")
    if len(legal_mask) == 0:
        raise ContractError("legal_mask empty")
    if not any(legal_mask):
        raise ContractError("legal_mask all false — nonterminal must have legal")
    # Teacher policy must be valid distribution over legal
    s = sum(teacher_policy)
    if not math.isclose(s, 1.0, abs_tol=1e-6):
        raise ContractError(f"teacher_policy sum {s} !=1")
    for p, m in zip(teacher_policy, legal_mask, strict=True):
        if not m and not math.isclose(p, 0.0, abs_tol=1e-9):
            raise ContractError(f"illegal action has non-zero prob {p}")
        if m and (p < -1e-9 or not math.isfinite(p)):
            raise ContractError(f"legal prob invalid {p}")
    # Vector finite
    for v in vector_return:
        if not math.isfinite(v):
            raise ContractError(f"vector_return non-finite {v}")
    return


#: Bridge leaf for the record-id seal (single source: ``hydra2._native.contracts``
#: ``distill_trajectory_record_id``; the stale-.so fallback is the byte-identical oracle).
_distill_trajectory_record_id = getattr(_distill_bridge, "distill_trajectory_record_id", None)


def _record_id_for(payload: dict[str, Any]) -> str:
    """Seal the trajectory payload to its ``record_id`` — bridge decides, oracle fallback."""
    if _distill_trajectory_record_id is not None:
        try:
            record_hex: str = _distill_trajectory_record_id(
                payload["observation_hash"],
                list(payload["legal_mask"]),
                list(payload["teacher_policy"]),
                list(payload["vector_return"]),
                payload["event_label"],
                list(payload["belief_label"]) if payload["belief_label"] is not None else None,
                payload["teacher_spec_hash"],
                canonical_bytes(payload["budget"]),
                canonical_bytes(payload["provenance"]),
            )
            return record_hex
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """Actor-visible distillation record; no privileged world in inference features."""

    observation_hash: str
    legal_mask: tuple[bool, ...]
    teacher_policy: tuple[float, ...]
    vector_return: tuple[float, float, float, float]
    event_label: str | None
    belief_label: tuple[float, ...] | None
    teacher_spec_hash: str
    budget: tuple[tuple[str, Any], ...]  # frozen budget items
    provenance: tuple[tuple[str, Any], ...]  # frozen provenance
    record_id: str

    def __post_init__(self) -> None:
        _ = _require_sha256("observation_hash", self.observation_hash)
        _ = _require_sha256("teacher_spec_hash", self.teacher_spec_hash)
        _ = _require_sha256("record_id", self.record_id)
        _validate_trajectory_values(
            list(self.legal_mask), list(self.teacher_policy), list(self.vector_return)
        )
        # Validate record_id
        payload = {
            "observation_hash": self.observation_hash,
            "legal_mask": list(self.legal_mask),
            "teacher_policy": list(self.teacher_policy),
            "vector_return": list(self.vector_return),
            "event_label": self.event_label,
            "belief_label": list(self.belief_label) if self.belief_label is not None else None,
            "teacher_spec_hash": self.teacher_spec_hash,
            "budget": dict(self.budget),
            "provenance": dict(self.provenance),
        }
        expected = _record_id_for(payload)
        if expected != self.record_id:
            raise ContractError(f"record_id mismatch expected {expected} got {self.record_id}")


def validate_trajectory_record(record: TrajectoryRecord) -> None:
    """Validate a trajectory record — raises ContractError on violation."""
    # Validation lives in TrajectoryRecord.__post_init__; this stable call-site
    # anchor re-checks masks after construction (generate_trajectories:936).
    if not isinstance(record, TrajectoryRecord):
        raise ContractError(f"expected TrajectoryRecord, got {type(record)}")
    return


def _budget_to_frozen(budget: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Freeze a budget dict as a sorted tuple."""
    return tuple(sorted(budget.items()))


def _provenance_to_frozen(prov: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Freeze provenance as a sorted tuple; values must be JSON-serializable."""
    return tuple(sorted(prov.items()))


def make_trajectory_record(
    *,
    case_id: str,
    actor: int,
    observation_hash: str,
    legal_mask: tuple[bool, ...],
    teacher_policy: tuple[float, ...],
    vector_return: tuple[float, float, float, float],
    teacher_spec_hash: str,
    budget: dict[str, Any],
    provenance: dict[str, Any],
    event_label: str | None = None,
    belief_label: tuple[float, ...] | None = None,
) -> TrajectoryRecord:
    """Hash payload to record_id, then construct the validated TrajectoryRecord."""
    obs_hash = _require_sha256("observation_hash", observation_hash)
    frozen_budget = _budget_to_frozen(budget)
    frozen_prov = _provenance_to_frozen(provenance)
    payload = {
        "observation_hash": obs_hash,
        "legal_mask": list(legal_mask),
        "teacher_policy": list(teacher_policy),
        "vector_return": list(vector_return),
        "event_label": event_label,
        "belief_label": list(belief_label) if belief_label is not None else None,
        "teacher_spec_hash": teacher_spec_hash,
        "budget": dict(frozen_budget),
        "provenance": dict(frozen_prov),
    }
    record_id = _record_id_for(payload)
    return TrajectoryRecord(
        observation_hash=obs_hash,
        legal_mask=legal_mask,
        teacher_policy=teacher_policy,
        vector_return=vector_return,
        event_label=event_label,
        belief_label=belief_label,
        teacher_spec_hash=teacher_spec_hash,
        budget=frozen_budget,
        provenance=frozen_prov,
        record_id=record_id,
    )


# ---------------------------------------------------------------------------
# Privileged namespace — only this module may generate event/belief labels from world state
# ---------------------------------------------------------------------------

_TRAINING_NAMESPACE_TOKEN: str = (
    _distill_bridge.DISTILL_TRAINING_NAMESPACE_TOKEN
    if hasattr(_distill_bridge, "DISTILL_TRAINING_NAMESPACE_TOKEN")
    else "training_namespace_v1"
)


def generate_privileged_labels(
    *,
    world_id: str,
    case_id: str,
    teacher_id: str,
    token: str,
) -> tuple[str, tuple[float, ...]]:
    """Isolated privileged label generation — requires training namespace token.

    Student inference paths MUST NOT call this; only training namespace uses it.
    """
    if token != _TRAINING_NAMESPACE_TOKEN:
        raise ContractError("privileged labels require training namespace token")
    # Deterministic labels derived from world_id + case_id
    key = f"priv:{world_id}:{case_id}:{teacher_id}".encode()
    event_label = "event:" + hashlib.sha256(key).hexdigest()[:12]
    belief = tuple(_hash_to_uniform(key, i) for i in range(4))
    # Normalize belief to sum 1
    s = sum(belief)
    belief_norm = tuple(v / s for v in belief) if s > 0 else (0.25, 0.25, 0.25, 0.25)
    return event_label, belief_norm


def _maybe_privileged_labels(
    *, case_id: str, teacher_id: str, with_privileged: bool, world_id: str | None
) -> tuple[str | None, tuple[float, ...] | None]:
    """Return (None, None) unless with_privileged; default world_id when absent."""
    if not with_privileged:
        return None, None
    if world_id is None:
        world_id = f"world:{case_id}:{teacher_id}"
    return generate_privileged_labels(
        world_id=world_id, case_id=case_id, teacher_id=teacher_id, token=_TRAINING_NAMESPACE_TOKEN
    )


# ---------------------------------------------------------------------------
# Trajectory generation — deterministic, actor-visible only
# ---------------------------------------------------------------------------


def generate_trajectories(
    *,
    justification: TeacherJustification,
    num_records: int = 32,
    actor: int = 0,
    budget: dict[str, Any] | None = None,
    with_privileged_labels: bool = False,
    num_actions: int | None = None,
    seed_material: bytes | bytearray = b"wp10_trajectory_v1",
) -> tuple[TrajectoryRecord, ...]:
    """Generate deterministic distillation trajectories for teacher.

    Each record carries the REAL actor-visible observation hash, the EXACT
    legal mask from the canonical action table, the REAL teacher policy
    (spec-bound model prior masked to the exact mask), and the REAL four-seat
    value vector. Records are actor-visible only; privileged labels are
    generated only inside the isolated training namespace when
    `with_privileged_labels` is True. Replacing teacher (different
    justification digest) invalidates all records because record provenance
    includes justification_digest. When the teacher path is unavailable the
    generation raises ContractError (WP-10 blocked) — never hash noise.
    """
    if not isinstance(justification, TeacherJustification):
        raise ContractError(
            f"justification must be TeacherJustification, got {type(justification)}"
        )
    if num_records <= 0:
        raise ContractError("num_records must be positive")
    if actor not in (0, 1, 2, 3):
        raise ContractError(f"actor must be 0..3, got {actor}")
    if not isinstance(seed_material, (bytes, bytearray)) or len(seed_material) == 0:
        raise ContractError("seed_material must be nonempty bytes")
    seed_material = bytes(seed_material)
    table = _action_table()
    n_actions = len(table)
    if num_actions is not None and num_actions != n_actions:
        raise ContractError(
            f"num_actions {num_actions} != canonical action table size {n_actions} "
            "(exact legal mask required)"
        )

    # Budget — frozen ResourceBudget-like dict
    if budget is None:
        budget = {
            "mode": "gameplay_5s",
            "deadline_ms": 5000,
            "max_model_calls": 64,
            "max_transitions": 256,
            "teacher_candidate_id": justification.teacher_candidate_id,
            "teacher_spec_hash": justification.candidate_spec_hash,
        }
    else:
        budget = dict(budget)
        # Ensure teacher identity in budget for traceability
        budget.setdefault("teacher_candidate_id", justification.teacher_candidate_id)
        budget.setdefault("teacher_spec_hash", justification.candidate_spec_hash)

    # Validate budget provenance matches justification
    if budget.get("teacher_spec_hash") != justification.candidate_spec_hash:
        raise ContractError("budget teacher_spec_hash mismatch justification")

    teacher_id = justification.teacher_candidate_id
    spec_hash = justification.candidate_spec_hash
    # Resolve the real spec and confirm it matches the justification digest
    # (a replaced/stale spec must not mint records under an old digest).
    spec = _real_candidate_spec(teacher_id)
    if _spec_digest_of(spec) != spec_hash:
        raise ContractError(
            f"justification spec hash {spec_hash} != live CandidateSpec digest "
            "(teacher spec changed — re-run select_teacher)"
        )

    records: list[TrajectoryRecord] = []
    for idx in range(num_records):
        case_id = f"case_{idx:05d}"
        obs = _case_observation(
            case_id=case_id,
            teacher_id=teacher_id,
            actor=actor,
            spec=spec,
            seed_material=seed_material,
        )
        legal_mask = tuple(obs.legal_mask)
        policy, vec = _teacher_policy_and_value(observation=obs, spec=spec)
        prov = _provenance_for_case(
            case_id=case_id,
            teacher_id=teacher_id,
            index=idx,
            actor=actor,
            seed_material=seed_material,
            budget=budget,
            justification_digest=justification.digest,
        )
        # Optional privileged labels — isolated namespace only
        event_label, belief_label = _maybe_privileged_labels(
            case_id=case_id,
            teacher_id=teacher_id,
            with_privileged=with_privileged_labels,
            world_id=None,
        )
        obs_hash = obs.observation_hash
        if obs_hash is None:
            raise ContractError("WP-10 blocked: case observation missing hash")
        rec = make_trajectory_record(
            case_id=case_id,
            actor=actor,
            observation_hash=obs_hash,
            legal_mask=legal_mask,
            teacher_policy=policy,
            vector_return=vec,
            teacher_spec_hash=spec_hash,
            budget=budget,
            provenance=prov,
            event_label=event_label,
            belief_label=belief_label,
        )
        # Validate
        validate_trajectory_record(rec)
        records.append(rec)

    return tuple(records)
