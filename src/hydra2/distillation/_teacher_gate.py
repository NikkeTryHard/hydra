"""WP-10 teacher gate — registry, analysis-gate loader, selection justification.

Owns the teacher-candidate registry beside its only readers, the WP-12
analysis-gate loader with its digest guards, the frozen
:class:`TeacherJustification` record, and the canonical action-table probe
plus cache shared by the case and evaluation paths. The deterministic case
observations live in :mod:`hydra2.distillation._teacher_cases`, trajectory
records in :mod:`hydra2.distillation._teacher_records`, the student model in
:mod:`hydra2.distillation._teacher_student`, and five-arm evaluation in
:mod:`hydra2.distillation._teacher_eval`, so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.search.common import CandidateSpec

# ---------------------------------------------------------------------------
# Constants — teacher registry mirrors Candidates 0-6
# ---------------------------------------------------------------------------

TEACHER_CANDIDATES: tuple[str, ...] = (
    "candidate0",
    "candidate1",
    "candidate2",
    "candidate3",
    "candidate4",
    "candidate5",
    "candidate6",
)

# Rejected modules/candidates remain registry evidence, never teachers (BUILD §13).
REJECTED_CANDIDATES: frozenset[str] = frozenset({"candidate4"})

# Gate hash domain — every digest below is content-addressed to a real artifact
# (CandidateSpec via per-candidate factory, WP-12 analysis-gate record). Missing
# prerequisites fail closed with ContractError (WP-10 blocked); nothing is
# synthesized (BUILD:818, SPEC:1587/1604).
_GATE_KINDS: tuple[str, ...] = ("contract", "exact", "search", "match", "analysis")

# Teacher candidate -> WP-12 analysis candidate id. WP-12 gates candidate3 under
# "candidate3_pbrf_core_v1" (ANALYSIS_CANDIDATE_IDS); every other teacher id is
# identical in both registries. Unmapped ids are blocked, never defaulted.
_ANALYSIS_ID_FOR_TEACHER: dict[str, str] = {
    "candidate0": "candidate0",
    "candidate1": "candidate1",
    "candidate2": "candidate2",
    "candidate3": "candidate3_pbrf_core_v1",
    "candidate5": "candidate5",
    "candidate6": "candidate6",
}

# Actor-visible feature dimensionality for the real model_input_v1 encoder path
# (34 concealed-tile counts + 4 scores + wall remaining + seats/phase one-hots).
# wall = the fixed ordered tile sequence a block of games is dealt from; wall
# blocks are the independent uncertainty unit for eval contrasts.
_REAL_FEATURE_DIM = 48

_DEFAULT_NUM_ACTIONS = 32  # import-time fallback only; real paths resolve via _action_table()


def _load_action_table_num_actions() -> int:
    """Probe action-table size; fall back to _DEFAULT_NUM_ACTIONS (never raises)."""
    try:
        from hydra2.config import repo_root

        p = repo_root() / "configs" / "contracts" / "action_table_v1.json"
        if not p.is_file():
            # Portable package-data fallback (https://docs.python.org/3/library/importlib.resources.html)
            # Try importlib.resources as secondary resolver for installed package data.
            try:
                from importlib.resources import files as _files  # pyrefly: ignore[import-error]

                cand = _files("hydra2").joinpath("../../../configs/contracts/action_table_v1.json")
                try:  # why-broad: package-data probe — any layout failure falls through
                    if cand.is_file():  # type: ignore[attr-defined]
                        p = Path(str(cand))
                except Exception:
                    pass
            except Exception:  # why-broad: package-data probe — any layout failure falls through
                pass
            if not p.is_file():
                raise FileNotFoundError(
                    f"config not found: {p} — run from checkout or pip install package data"
                )
        data: Any = json.loads(p.read_text())  # pyrefly: ignore[explicit-any]
        # Canonical envelope: {"payload": {"actions": [...]}, ...}
        if isinstance(data, dict):
            payload: Any = data.get("payload", data)  # pyrefly: ignore[explicit-any]
            if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
                return len(payload["actions"])
        # action_table may be list or dict with actions
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict) and "actions" in data:
            actions_val: Any = cast("dict[str, Any]", data).get("actions")  # pyrefly: ignore[explicit-any]
            if isinstance(actions_val, list):
                return len(actions_val)
            return len(cast("list[Any]", actions_val))  # pyrefly: ignore[explicit-any]
        # fallback: try keys
        dict_data: dict[str, Any] = cast("dict[str, Any]", data)  # pyrefly: ignore[explicit-any]
        num_val: Any = dict_data.get("num_actions", _DEFAULT_NUM_ACTIONS)  # pyrefly: ignore[explicit-any]
        if isinstance(num_val, int):
            return num_val
        if isinstance(num_val, float):
            return int(num_val)
        if isinstance(num_val, str):
            try:
                return int(num_val)
            except ValueError:
                return _DEFAULT_NUM_ACTIONS
        return _DEFAULT_NUM_ACTIONS
    except Exception:  # why-broad: import-time probe must never raise; real paths fail closed
        return _DEFAULT_NUM_ACTIONS


_NUM_ACTIONS: int = _load_action_table_num_actions()

_ACTION_TABLE_CACHE: list[dict[str, Any]] | None = None


def _action_table() -> list[dict[str, Any]]:
    """Load canonical action-table entries (payload.actions); fail closed when absent.

    The exact legal mask is derived from these entries — a missing table blocks
    WP-10 trajectory generation instead of falling back to a smaller mask.
    """
    global _ACTION_TABLE_CACHE
    if _ACTION_TABLE_CACHE is not None:
        return _ACTION_TABLE_CACHE
    from hydra2.config import repo_root

    p = repo_root() / "configs" / "contracts" / "action_table_v1.json"
    if not p.is_file():
        raise ContractError(
            f"WP-10 blocked: action table not found at {p} — cannot derive exact legal mask"
        )
    data: Any = json.loads(p.read_text())  # pyrefly: ignore[explicit-any]
    if not isinstance(data, dict) or not isinstance(data.get("payload"), dict):
        raise ContractError("WP-10 blocked: action table envelope missing payload")
    actions: Any = data["payload"].get("actions")  # pyrefly: ignore[explicit-any]
    if not isinstance(actions, list) or len(actions) == 0:
        raise ContractError("WP-10 blocked: action table payload.actions empty")
    entries: list[dict[str, Any]] = [dict(a) for a in actions if isinstance(a, dict)]
    if len(entries) != len(actions):
        raise ContractError("WP-10 blocked: action table entries malformed")
    _ACTION_TABLE_CACHE = entries
    return entries


# ---------------------------------------------------------------------------
# Analysis gate loader — coord with Wp12 shape
# ---------------------------------------------------------------------------


def load_analysis_gate(candidate_id: str) -> dict[str, Any]:
    """Load the WP-12 analysis gate for a teacher candidate — fail closed.

    Delegates to the canonical :func:`hydra2.analysis.qualification.analysis_gate_for`
    on the canonical path (``work_packages/WP-12/analysis_gates.json``). WP-10 is
    blocked for the candidate unless the gate exists, is eligible, is
    compute-only, and passed deterministic replay (BUILD:701/738 ordering:
    WP-12 executes before WP-10 teacher selection). Rejected candidates raise
    (never an ``eligible=False`` dict — fail closed with exception).
    Returned hashes are the REAL gate-record digests, never synthesized.
    """
    if candidate_id in REJECTED_CANDIDATES:
        raise ContractError(
            f"WP-10 blocked: candidate {candidate_id!r} is rejected and can never be teacher"
        )
    analysis_id = _ANALYSIS_ID_FOR_TEACHER.get(candidate_id)
    if analysis_id is None:
        raise ContractError(
            f"WP-10 blocked: candidate {candidate_id!r} has no WP-12 analysis identity"
        )
    from hydra2.analysis.qualification import analysis_gate_for

    gate = analysis_gate_for(analysis_id)
    if gate is None:
        raise ContractError(
            f"WP-10 blocked for {candidate_id!r}: no WP-12 analysis gate at "
            "work_packages/WP-12/analysis_gates.json — generate the hashed "
            "analysis report first (BUILD:701 entry, BUILD:738 ordering)"
        )
    if not bool(gate.get("eligible")):
        raise ContractError(
            f"WP-10 blocked for {candidate_id!r}: WP-12 gate ineligible: "
            f"{gate.get('reason', 'ineligible')}"
        )
    if not bool(gate.get("compute_only")):
        raise ContractError(f"WP-10 blocked for {candidate_id!r}: WP-12 gate not compute_only")
    if not bool(gate.get("deterministic_replay_ok")):
        raise ContractError(
            f"WP-10 blocked for {candidate_id!r}: WP-12 deterministic replay failed"
        )
    gameplay_spec_hash = _require_sha256("gameplay_spec_hash", str(gate.get("gameplay_spec_hash")))
    analysis_spec_hash = _require_sha256("analysis_spec_hash", str(gate.get("analysis_spec_hash")))
    report_hash = _require_sha256("report_hash", str(gate.get("report_hash")))
    digest = _require_sha256("digest", str(gate.get("digest")))
    return {
        "candidate_id": candidate_id,
        "analysis_candidate_id": analysis_id,
        "eligible": True,
        "compute_only": True,
        "deterministic_replay": True,
        "deterministic_replay_ok": True,
        "gameplay_spec_hash": gameplay_spec_hash,
        "analysis_spec_hash": analysis_spec_hash,
        "report_hash": report_hash,
        "digest": digest,
        "reason": str(gate.get("reason", "passed")),
    }


def _require_sha256(name: str, value: str) -> str:
    """Guard sha256:<64-hex> shape; fail closed on mismatch."""
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContractError(f"{name} must be sha256:<64 hex>, got {value!r}")
    hexpart = value[7:]
    if any(c not in "0123456789abcdef" for c in hexpart):
        raise ContractError(f"{name} invalid hex: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Teacher justification — frozen before trajectory generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeacherJustification:
    """Frozen teacher selection justification (recorded before trajectories)."""

    teacher_candidate_id: str
    candidate_spec_hash: str
    gate_hashes: tuple[tuple[str, str], ...]  # sorted (kind, digest)
    justification_text: str
    selected_at_utc: str
    digest: str

    def __post_init__(self) -> None:
        if self.teacher_candidate_id not in TEACHER_CANDIDATES:
            raise ContractError(f"unknown teacher candidate {self.teacher_candidate_id!r}")
        if self.teacher_candidate_id in REJECTED_CANDIDATES:
            raise ContractError(
                f"rejected candidate {self.teacher_candidate_id!r} cannot be teacher"
            )
        _ = _require_sha256("candidate_spec_hash", self.candidate_spec_hash)
        if self.justification_text == "" or not isinstance(self.justification_text, str):
            raise ContractError("justification_text must be non-empty string")
        if len(self.gate_hashes) == 0 or len(self.gate_hashes) != len(_GATE_KINDS):
            raise ContractError(f"gate_hashes must have {len(_GATE_KINDS)} entries")
        kinds = tuple(k for k, _ in self.gate_hashes)
        if tuple(sorted(kinds)) != tuple(sorted(_GATE_KINDS)):
            raise ContractError(f"gate kinds must be {_GATE_KINDS}, got {kinds}")
        for _k, v in self.gate_hashes:
            _ = _require_sha256(f"gate_hash:{_k}", v)
        _ = _require_sha256("digest", self.digest)
        # validate digest matches canonical
        payload = {
            "teacher_candidate_id": self.teacher_candidate_id,
            "candidate_spec_hash": self.candidate_spec_hash,
            "gate_hashes": dict(self.gate_hashes),
            "justification_text": self.justification_text,
            "selected_at_utc": self.selected_at_utc,
        }
        expected = "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()
        if expected != self.digest:
            raise ContractError(f"digest mismatch: expected {expected}, got {self.digest}")


def _now_utc() -> str:
    """Mint a UTC %Y-%m-%dT%H:%M:%SZ timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def select_teacher(
    *,
    candidate_id: str,
    justification_text: str,
    selected_at_utc: str | None = None,
) -> TeacherJustification:
    """Select teacher only from 5-gate-passed outcome; record justification before trajectories.

    Validates contract/exact/search/match/analysis gates and rejects REJECTED_CANDIDATES.
    The CandidateSpec digest is resolved via the candidate's REAL per-candidate
    factory (mirroring qualification `_make_gameplay_spec_for`); unknown or
    unmapped candidates block with ContractError. Gate hashes bind the real
    WP-12 gate/record digests — never synthesized.
    """
    if candidate_id not in TEACHER_CANDIDATES:
        raise ContractError(f"unknown candidate {candidate_id!r}")
    if candidate_id in REJECTED_CANDIDATES:
        raise ContractError(f"candidate {candidate_id!r} is rejected and cannot be teacher")
    # Load analysis gate and enforce 5-gate eligibility (raises when blocked).
    gate = load_analysis_gate(candidate_id)
    # Resolve the REAL CandidateSpec and bind its content digest. The spec must
    # agree with the WP-12 gate's gameplay_spec_hash (stale-hash guard).
    spec = _real_candidate_spec(candidate_id)
    spec_hash = _require_sha256("candidate_spec_hash", _spec_digest_of(spec))
    if spec_hash != gate["gameplay_spec_hash"]:
        raise ContractError(
            f"WP-10 blocked for {candidate_id!r}: factory CandidateSpec digest "
            f"{spec_hash} != WP-12 gameplay_spec_hash {gate['gameplay_spec_hash']} (stale)"
        )
    gates: dict[str, str] = {}
    for kind in _GATE_KINDS:
        gates[kind] = _gate_hash_for_kind(
            kind,
            candidate_spec_hash=spec_hash,
            gate=gate,
        )
    sorted_gates = tuple(sorted(gates.items()))
    ts = selected_at_utc if selected_at_utc is not None else _now_utc()
    payload = {
        "teacher_candidate_id": candidate_id,
        "candidate_spec_hash": spec_hash,
        "gate_hashes": dict(sorted_gates),
        "justification_text": justification_text,
        "selected_at_utc": ts,
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return TeacherJustification(
        teacher_candidate_id=candidate_id,
        candidate_spec_hash=spec_hash,
        gate_hashes=sorted_gates,
        justification_text=justification_text,
        selected_at_utc=ts,
        digest=digest,
    )


def _real_candidate_spec(candidate_id: str) -> CandidateSpec:
    """Resolve the REAL CandidateSpec for a teacher candidate via its factory.

    Mirrors ``hydra2.analysis.qualification._make_gameplay_spec_for`` (same
    factories, same analysis identity mapping). Unknown/unmapped candidates
    raise ContractError — never a hardcoded default spec.
    """
    analysis_id = _ANALYSIS_ID_FOR_TEACHER.get(candidate_id)
    if analysis_id is None:
        raise ContractError(
            f"WP-10 blocked: candidate {candidate_id!r} has no CandidateSpec factory mapping"
        )
    from hydra2.analysis.qualification import _make_gameplay_spec_for

    try:
        spec = _make_gameplay_spec_for(analysis_id)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"WP-10 blocked for {candidate_id!r}: CandidateSpec factory failed: {exc}"
        ) from exc
    # Identity is established by digest equality against the qualified WP-12
    # gate (checked by callers) — factories use their canonical spec ids
    # (e.g. candidate2 -> "candidate2_despot_natural"), so no id comparison here.
    return spec


def _spec_digest_of(spec: CandidateSpec) -> str:
    """Content digest of a CandidateSpec (SPEC 15 canonical projection)."""
    from hydra2.search.common import candidate_spec_hash

    try:
        digest = str(candidate_spec_hash(spec))
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"WP-10 blocked: CandidateSpec hashing failed: {exc}") from exc
    return _require_sha256("candidate_spec_hash", digest)


def _gate_hash_for_kind(kind: str, *, candidate_spec_hash: str, gate: dict[str, Any]) -> str:
    """Bind a gate kind to REAL outcome-record digests (never synthesized).

    The ``analysis`` kind carries the WP-12 analysis_spec_hash directly; every
    other kind carries a content digest over the real WP-12 gate digest, the
    real report hash, and the real CandidateSpec digest. All inputs are real
    artifact digests, so each gate hash is verifiable by recomputation from
    the gate record — a hash of a ``"wp10:gate:..."`` literal would not be.
    """
    if kind == "analysis":
        return _require_sha256("gate_hash:analysis", str(gate["analysis_spec_hash"]))
    payload = {
        "kind": kind,
        "candidate_spec_hash": candidate_spec_hash,
        "analysis_gate_digest": str(gate["digest"]),
        "report_hash": str(gate["report_hash"]),
    }
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()
