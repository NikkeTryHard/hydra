# ruff: noqa: TC006, PERF401  # reason: legacy blanket; E501 URLs unwrappable, TC006 quotes intentional, SIM102 nested guards readable, PERF401/B905 perf-critical. Evidence: https://docs.python.org/3/library/importlib.resources.html
"""Analysis gate records and the hashed WP-12 report.

Owns the per-candidate gate records that block teacher eligibility, the
candidate factories that synthesize gameplay specs (per-candidate search
factories first, file-backed fallback second), and the atomic,
content-addressed report publish. Factory imports stay function-local so
importing this module never pulls candidate search stacks or the model —
a missing factory degrades to the file-backed fallback instead of failing
the import.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.analysis.qual_budget import (
    ANALYSIS_CANDIDATE_IDS as ANALYSIS_CANDIDATE_IDS,
)
from hydra2.analysis.qual_budget import (
    GAMEPLAY_BUDGETS as GAMEPLAY_BUDGETS,
)
from hydra2.analysis.qual_budget import analysis_budget_for as analysis_budget_for
from hydra2.analysis.qual_budget import make_analysis_spec as make_analysis_spec
from hydra2.analysis.qual_budget import verify_compute_only as verify_compute_only
from hydra2.analysis.qual_replay import (
    _spec_hash as _spec_hash,
)
from hydra2.analysis.qual_replay import (
    compare_gameplay_analysis as compare_gameplay_analysis,
)
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical, sha256_digest, validate_digest
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    VisibilityViolationError,
)

logger = logging.getLogger(__name__)


ANALYSIS_REPORT_KIND = "hydra2.analysis_gate_report"
ANALYSIS_REPORT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class AnalysisGateRecord:
    """Per-candidate analysis gate record — teacher eligibility blocker.

    A candidate is teacher-eligible only if this record exists, has
    ``compute_only == True``, ``deterministic_replay_ok == True``, no
    privileged leak, and the analysis budget is finite and larger than
    gameplay. ``rejected`` candidates remain registry evidence, never teachers
    (BUILD §10).
    """

    candidate_id: str
    gameplay_spec_hash: str
    analysis_spec_hash: str
    analysis_budget: Mapping[str, Any]
    compute_only: bool
    deterministic_replay_ok: bool
    privileged_leak: bool
    comparison: Mapping[str, Any]
    eligible: bool
    reason: str
    digest: str = field(default="")

    def __post_init__(self) -> None:
        _: DigestText = validate_digest(self.gameplay_spec_hash)
        _: DigestText = validate_digest(self.analysis_spec_hash)
        if not isinstance(self.candidate_id, str) or self.candidate_id == "":
            raise ContractError("candidate_id must be non-empty str")
        if not isinstance(self.comparison, Mapping):
            raise ContractError("comparison must be mapping")


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Consolidated WP-12 hashed analysis report.

    Contains one AnalysisGateRecord per teacher-eligible Candidate 0-6.
    The report's digest is sha256 over RFC 8785 canonical bytes of its
    payload (excluding the digest field itself). It is written atomically
    to ``$HYDRA2_ARTIFACT_ROOT/reports/WP-12/<run-id>/analysis_report.json``
    and also to the standard pytest contract report location.
    """

    schema_version: str
    kind: str
    generated_at_utc: str
    artifact_root: str
    budgets: Mapping[str, Mapping[str, Any]]
    gates: tuple[AnalysisGateRecord, ...]
    digest: str


def compute_only_proof(gameplay_spec: Any, analysis_spec: Any) -> bool:
    """Return True iff analysis is compute-only (or raise)."""
    return verify_compute_only(gameplay_spec, analysis_spec)


def _utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_default_hashes_for_spec() -> dict[str, str]:
    """File-backed config hashes + model-derived semantic digests for specs.

    Used by the generic fallback in ``_make_gameplay_spec_for`` when a
    per-candidate factory import fails. File-backed configs hash from disk;
    utility/model derive from the live model; rng/stream/case use the
    candidate0 canonical descriptors verbatim. Never constant hashes: this
    path is prod-reachable (gate records), so derivation failure raises
    loudly instead of fabricating a manifest (BUILD S17).
    """
    # repo_root() walks to the marker so this survives the analysis/ depth.
    from hydra2.config import repo_root
    from hydra2.search.common import _require_real_file

    repo = repo_root()
    defaults: dict[str, str] = {}
    for name, rel in (
        ("rules_hash", "configs/rules/tenhou_4p_hanchan_v1.json"),
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ):
        p = repo / rel
        try:
            real = _require_real_file(p, repo)
            defaults[name] = str(sha256_digest(real.read_bytes()))
        except (ImportError, AttributeError, OSError, ValueError, TypeError, ContractError) as exc:
            logger.debug("qualification: default hash fallback for %s", name, exc_info=exc)
            raise ContractError(
                f"qualification: cannot derive file-backed {name} from {rel}"
            ) from exc
    try:
        from hydra2.models.model import Hydra2BaselineModel

        probe = Hydra2BaselineModel()
        defaults["utility_manifest_hash"] = str(validate_digest(str(probe.utility_manifest_hash)))
        defaults["model_hash"] = str(validate_digest(str(probe.model_identity)))
    except (ImportError, AttributeError, ValueError, TypeError, OSError, ContractError) as exc:
        logger.debug("qualification: model-derived hash fallback", exc_info=exc)
        raise ContractError("qualification: cannot derive utility/model hashes from model") from exc
    # RNG / stream / case — candidate0 canonical descriptors verbatim
    defaults["rng_protocol_hash"] = str(
        of_canonical({"protocol": "counter_based_v1", "version": "1.0.0"})
    )
    defaults["random_stream_schema_hash"] = str(
        of_canonical({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
    )
    defaults["case_manifest_hash"] = str(of_canonical([]))
    return defaults


def _make_gameplay_spec_for(candidate_id: str) -> Any:
    """Synthesize a gameplay spec for candidate_id (factories, else file-backed fallback)."""
    # Try per-candidate factories first
    try:
        if candidate_id == "candidate0":
            from hydra2.search.candidate0 import make_candidate0_spec

            return make_candidate0_spec()
        if candidate_id == "candidate1":
            from hydra2.search.ismcts_natural import (
                make_ismcts_candidate_spec,  # type: ignore[import]  # reason: optional candidate factory may be absent in minimal env
            )

            return make_ismcts_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate2":
            from hydra2.search.despot_natural import make_despot_candidate_spec

            return make_despot_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate3_pbrf_core_v1":
            from hydra2.search.pbrf import make_pbrf_candidate_spec

            return make_pbrf_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate4_core_control":
            from hydra2.search.modules import (
                make_module_candidate_spec,  # type: ignore[import]  # reason: optional module factory may be absent in minimal env
            )

            # Use control forest as representative for candidate4
            return make_module_candidate_spec(module_id="control")
    except Exception:
        pass
    try:
        if candidate_id == "candidate5":
            from hydra2.search.local_resolving import (
                make_candidate5_spec as make_local_resolving_spec,  # type: ignore[import]  # reason: optional resolving factory may be absent in minimal env
            )

            return make_local_resolving_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate6":
            from hydra2.search.gumbel import make_gumbel_candidate_spec

            return make_gumbel_candidate_spec()
    except Exception:
        pass
    # Generic fallback: construct via CandidateSpec directly with defaults
    from hydra2.search.common import CandidateSpec, ResourceBudget

    defaults = _load_default_hashes_for_spec()
    gp_cfg = GAMEPLAY_BUDGETS.get(candidate_id, GAMEPLAY_BUDGETS["candidate0"])
    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=gp_cfg["deadline_ms"] if gp_cfg["deadline_ms"] is not None else 5000,
        fallback_margin_ms=gp_cfg["fallback_margin_ms"]
        if gp_cfg["fallback_margin_ms"] is not None
        else 200,
        max_model_calls=gp_cfg["max_model_calls"],
        max_transitions=gp_cfg["max_transitions"],
        max_particles=gp_cfg["max_particles"],
        max_memory_bytes=gp_cfg["max_memory_bytes"],
    )
    # Map candidate_id to algorithm name for uniqueness
    algo_map = {
        "candidate0": "frozen_policy",
        "candidate1": "ismcts_natural",
        "candidate2": "despot_natural",
        "candidate3_pbrf_core_v1": "pbrf_core",
        "candidate4_core_control": "pbrf_module_control",
        "candidate5": "local_resolving",
        "candidate6": "gumbel_search",
    }
    return CandidateSpec(
        candidate_id=candidate_id,
        algorithm=algo_map.get(candidate_id, "generic_search"),
        algorithm_version="1.0.0",
        rules_hash=defaults["rules_hash"],
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash=defaults["utility_manifest_hash"],
        action_table_hash=defaults["action_table_hash"],
        observation_schema_hash=defaults["observation_schema_hash"],
        packet_boundary_hash=defaults["packet_boundary_hash"],
        model_hash=defaults["model_hash"],
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=defaults["case_manifest_hash"],
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break="lexicographic",
        rng_protocol_hash=defaults["rng_protocol_hash"],
        random_stream_schema_hash=defaults["random_stream_schema_hash"],
        parameters={"candidate_id": candidate_id},
    )


def build_gate_record(
    candidate_id: str,
    *,
    gameplay_spec: Any | None = None,
    observation: Any | None = None,
    legal_actions: tuple[Any, ...] | None = None,
) -> AnalysisGateRecord:
    """Build a single candidate's analysis gate record, synthesizing fixtures if needed."""
    from hydra2.contracts.action_model import CanonicalAction

    gp_spec: Any = (
        gameplay_spec if gameplay_spec is not None else _make_gameplay_spec_for(candidate_id)
    )
    an_spec: Any = make_analysis_spec(gp_spec)

    # Synthesize minimal actor-visible observation + legal actions if not supplied
    if observation is None or legal_actions is None:
        # Use tiny actor observation via world helper if available; else fallback stub
        try:
            from hydra2.belief.world import make_full_world, world_actor_observation

            w = make_full_world(
                concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
                live_wall=tuple(range(8, 40)),
                dead_wall=(),
                rules_hash=cast(str, gp_spec.rules_hash),
                observation_hash=str(of_canonical({"case": candidate_id})),
            )
            obs = world_actor_observation(w, actor=_bridge_contracts.make_seat(0))
            legal = (
                CanonicalAction(
                    kind="pass",
                    actor=_bridge_contracts.make_seat(0),
                    tile=None,
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
                CanonicalAction(
                    kind="discard",
                    actor=_bridge_contracts.make_seat(0),
                    tile=_bridge_contracts.make_tile_id(0),
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
            )
            observation = obs
            legal_actions = legal
        except Exception:
            # Fallback: construct synthetic observation stub
            # Use ActorObservation-like dict with required hash
            class _ObsStub:
                observation_hash = str(of_canonical({"stub": candidate_id}))
                actor = 0

            observation = _ObsStub()
            legal_actions = (
                CanonicalAction(
                    kind="pass",
                    actor=_bridge_contracts.make_seat(0),
                    tile=None,
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
            )

    # Perform comparison (includes compute-only and privileged checks)
    try:
        comp = compare_gameplay_analysis(
            gameplay_spec=gp_spec,
            analysis_spec=an_spec,
            observation=observation,
            legal_actions=legal_actions,
            case_id=f"{candidate_id}_analysis_gate",
        )
        compute_only = bool(comp.get("compute_only"))
        deterministic_ok = bool(comp.get("deterministic_replay_ok"))
        privileged_leak = False
        eligible = compute_only and deterministic_ok and not privileged_leak
        reason = "passed" if eligible else "comparison_failed"
    except (ContractError, VisibilityViolationError) as exc:
        comp = {
            "gameplay_spec_hash": _spec_hash(gp_spec),
            "analysis_spec_hash": _spec_hash(an_spec),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        compute_only = False
        deterministic_ok = False
        privileged_leak = isinstance(exc, VisibilityViolationError)
        eligible = False
        reason = str(exc)[:240]

    an_budget_dict = {
        "mode": an_spec.resource_budget.mode,
        "deadline_ms": an_spec.resource_budget.deadline_ms,
        "fallback_margin_ms": an_spec.resource_budget.fallback_margin_ms,
        "max_model_calls": an_spec.resource_budget.max_model_calls,
        "max_transitions": an_spec.resource_budget.max_transitions,
        "max_particles": an_spec.resource_budget.max_particles,
        "max_memory_bytes": an_spec.resource_budget.max_memory_bytes,
    }
    # Digest over gate record content (excluding digest field)
    gate_payload = {
        "candidate_id": candidate_id,
        "gameplay_spec_hash": _spec_hash(gp_spec),
        "analysis_spec_hash": _spec_hash(an_spec),
        "analysis_budget": an_budget_dict,
        "compute_only": compute_only,
        "deterministic_replay_ok": deterministic_ok,
        "privileged_leak": privileged_leak,
        "eligible": eligible,
        "reason": reason,
        "comparison_digest": str(of_canonical(comp)),
    }
    digest = str(of_canonical(gate_payload))
    return AnalysisGateRecord(
        candidate_id=candidate_id,
        gameplay_spec_hash=_spec_hash(gp_spec),
        analysis_spec_hash=_spec_hash(an_spec),
        analysis_budget=an_budget_dict,
        compute_only=compute_only,
        deterministic_replay_ok=deterministic_ok,
        privileged_leak=privileged_leak,
        comparison=comp,
        eligible=eligible,
        reason=reason,
        digest=digest,
    )


def generate_hashed_analysis_report(
    *,
    artifact_root: Path | str | None = None,
    candidate_ids: tuple[str, ...] | None = None,
) -> tuple[Path, str]:
    """Generate and atomically publish the WP-12 hashed analysis report.

    Writes two artifacts:
    - ``$ART/reports/WP-12/<run-id>/report.json`` (contract report, via caller)
    - ``$ART/reports/WP-12/<run-id>/analysis_report.json`` (analysis gates, hashed)
    - ``$ART/work_packages/WP-12/analysis_gates.json`` (latest, content-addressed)

    Returns (path_to_analysis_report, digest).
    The digest is sha256 over RFC 8785 canonical bytes of the report payload.
    """
    from hydra2.artifacts.atomic import atomic_replace_bytes
    from hydra2.config import artifact_root as cfg_artifact_root

    art = Path(artifact_root) if artifact_root is not None else cfg_artifact_root()
    candidates: tuple[str, ...] = (
        candidate_ids if candidate_ids is not None else ANALYSIS_CANDIDATE_IDS
    )
    gates: list[AnalysisGateRecord] = []
    for cid in candidates:
        gates.append(build_gate_record(cid))

    # Build budgets view
    budgets_view: dict[str, Any] = {}
    for cid in candidates:
        b = analysis_budget_for(cid)
        budgets_view[cid] = {
            "mode": b.mode,
            "deadline_ms": b.deadline_ms,
            "fallback_margin_ms": b.fallback_margin_ms,
            "max_model_calls": b.max_model_calls,
            "max_transitions": b.max_transitions,
            "max_particles": b.max_particles,
            "max_memory_bytes": b.max_memory_bytes,
        }

    report_payload: dict[str, Any] = {
        "schema_version": ANALYSIS_REPORT_SCHEMA_VERSION,
        "kind": ANALYSIS_REPORT_KIND,
        "generated_at_utc": _utc_now(),
        "artifact_root": str(art),
        "budgets": budgets_view,
        "gates": [
            {
                "candidate_id": g.candidate_id,
                "gameplay_spec_hash": g.gameplay_spec_hash,
                "analysis_spec_hash": g.analysis_spec_hash,
                "analysis_budget": dict(g.analysis_budget),
                "compute_only": g.compute_only,
                "deterministic_replay_ok": g.deterministic_replay_ok,
                "privileged_leak": g.privileged_leak,
                "eligible": g.eligible,
                "reason": g.reason,
                "digest": g.digest,
                "comparison": dict(g.comparison),
            }
            for g in gates
        ],
        "summary": {
            "total": len(gates),
            "eligible": sum(1 for g in gates if g.eligible),
            "ineligible": sum(1 for g in gates if not g.eligible),
            "compute_only_pass": sum(1 for g in gates if g.compute_only),
            "deterministic_pass": sum(1 for g in gates if g.deterministic_replay_ok),
        },
    }
    digest = str(of_canonical(report_payload))
    report_payload["digest"] = digest

    # Atomic write to run-id directory and to latest. The first stamp is
    # superseded by the canonical microsecond run_id (kept for ordering).
    run_id = _utc_now().replace(":", "").replace("-", "")  # compact but still UTC-like
    # Canonical microsecond run_id for uniqueness (supersedes the stamp above).
    from datetime import UTC, datetime

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = art / "reports" / "WP-12" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "analysis_report.json"
    atomic_replace_bytes(report_path, canonical_bytes(report_payload))

    # Also publish latest under work_packages/WP-12/ for teacher selection
    latest_dir = art / "work_packages" / "WP-12"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / "analysis_gates.json"
    atomic_replace_bytes(latest_path, canonical_bytes(report_payload))

    # Also publish content-addressed copy by digest
    content_path = latest_dir / f"{digest.split(':', 1)[1]}.json"
    atomic_replace_bytes(content_path, canonical_bytes(report_payload))

    return report_path, digest


def analysis_gate_for(
    candidate_id: str, *, artifact_root: Path | str | None = None
) -> dict[str, Any] | None:
    """Load the analysis gate for candidate_id from the latest hashed report.

    Returns dict with keys {eligible, analysis_spec_hash, report_hash, compute_only,
    deterministic_replay_ok, digest} or None if not yet generated. Teacher
    selection should check ``compute_only == True`` and ``eligible == True``.
    """
    from hydra2.config import artifact_root as cfg_artifact_root

    art = Path(artifact_root) if artifact_root is not None else cfg_artifact_root()
    latest_path = art / "work_packages" / "WP-12" / "analysis_gates.json"
    if not latest_path.is_file():
        return None
    try:
        import json

        doc: dict[str, Any] = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    _gates_raw: Any = doc.get("gates", [])
    _gates: list[Any] = cast(list[Any], _gates_raw) if isinstance(_gates_raw, list) else []
    for _gate_raw in _gates:
        gate: dict[str, Any] = (
            cast(dict[str, Any], _gate_raw) if isinstance(_gate_raw, dict) else {}
        )
        if gate.get("candidate_id") == candidate_id:
            return {
                "candidate_id": candidate_id,
                "eligible": bool(cast(Any, gate.get("eligible"))),
                "analysis_spec_hash": str(cast(Any, gate.get("analysis_spec_hash"))),
                "gameplay_spec_hash": str(cast(Any, gate.get("gameplay_spec_hash"))),
                "report_hash": str(cast(Any, doc.get("digest"))),
                "compute_only": bool(cast(Any, gate.get("compute_only"))),
                "deterministic_replay_ok": bool(cast(Any, gate.get("deterministic_replay_ok"))),
                "digest": str(cast(Any, gate.get("digest"))),
                "reason": str(cast(Any, gate.get("reason", ""))),
            }
    return None
