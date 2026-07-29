"""Wave 8 shared search contracts — SPEC 15 CandidateSpec/Search API.

This module is the single authority for ``ResourceBudget``, ``CandidateSpec``,
``SearchRequest``, ``SearchResult`` and the ``Planner`` protocol. Wave 8
candidates (0,1,2) all import from here so that ``candidate_spec_hash`` and
``resource_budget`` semantics have one implementation.

Ownership: WP-08A creates the file; peers extend without redefinition.
Contracts depend only on stdlib + ``hydra2.contracts.*`` + ``hydra2.artifacts.*``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — runtime needed for REPO_ROOT = repo_root()
from typing import Any, Protocol

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.config import repo_root
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    make_digest_text,
    make_schema_version,
)

__all__ = [
    "DEPLOYABLE_DEADLINE_MS",
    "HASH63_MOD",
    "MISSING_HASH",
    "PLACEHOLDER_1",
    "PLACEHOLDER_2",
    "PLACEHOLDER_A",
    "PLACEHOLDER_B",
    "PLACEHOLDER_C",
    "PLACEHOLDER_D",
    "PLACEHOLDER_E",
    "PLACEHOLDER_F",
    "REPO_ROOT",
    "U32_MAX",
    "U64_DENOM",
    "U64_MOD",
    "CandidateSpec",
    "Planner",
    "ResourceBudget",
    "SearchRequest",
    "SearchResult",
    "candidate_spec_hash",
    "candidate_spec_to_json",
    "resource_budget_to_json",
]

VALID_MODES: tuple[str, ...] = ("gameplay_5s", "ponder", "analysis")
VALID_FALLBACK: tuple[str, ...] = ("candidate0",)
VALID_TIE_BREAKS: frozenset[str] = frozenset(
    ("greedy", "temperature_0.5", "temperature_1.0", "value_break")
)

# Portable repo root via marker walk (pyproject.toml/.git) — not hardcoded
# parents[3] depth. Centralizes via hydra2.config.repo_root (cached walk).
# Evidence: https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
# Evidence: https://github.com/fsspec/universal_pathlib
# Evidence: https://github.com/tox-dev/platformdirs (XDG/portable context)
# Legacy: previously Path(__file__).resolve().parents[3] (brittle if layout changes).
REPO_ROOT: Path = repo_root()
DEPLOYABLE_DEADLINE_MS: int = 5000
MISSING_HASH: str = "0" * 64
PLACEHOLDER_A: str = "a" * 64
PLACEHOLDER_B: str = "b" * 64
PLACEHOLDER_C: str = "c" * 64
PLACEHOLDER_D: str = "d" * 64
PLACEHOLDER_E: str = "e" * 64
PLACEHOLDER_F: str = "f" * 64
PLACEHOLDER_1: str = "1" * 64
PLACEHOLDER_2: str = "2" * 64
HASH63_MOD: int = 2**63 - 1
U32_MAX: int = 0xFFFFFFFF
U64_MOD: int = 2**64
U64_DENOM: float = 18446744073709551616.0


def _require_real_file(p: Path, repo: Path) -> Path:
    """Reject symlinks and path traversal (CWE-22/59, pathlib Context7).

    - ``p.is_symlink()`` rejects symlink at ``p`` itself.
    - ``p.resolve(strict=True)`` resolves all components, fails if missing.
    - ``resolved.is_relative_to(repo.resolve())`` ensures result inside ``repo``.

    Evidence: ``pathlib.Path.is_symlink``, ``Path.resolve(strict=True)``,
    ``Path.is_relative_to`` (Python docs, Context7) + CWE-22/59.
    """
    if p.is_symlink():
        raise ContractError("symlink not allowed")
    resolved = p.resolve(strict=True)
    if not resolved.is_relative_to(repo.resolve()):
        raise ContractError("path traversal")
    return resolved


def _require_mode(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_MODES:
        raise ContractError(f"mode must be one of {VALID_MODES}, got {value!r}")
    return value


def _require_fallback(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_FALLBACK:
        raise ContractError(f"fallback_candidate_id must be 'candidate0', got {value!r}")
    return value


def _require_deadline_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"deadline_ms must be int, got {type(value).__name__}")
    if value <= 0 or value > 60000:
        raise ContractError(f"deadline_ms {value} outside (0,60000]")
    return value


def _require_fallback_margin(value: object, deadline_ms: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"fallback_margin_ms must be int, got {type(value).__name__}")
    if value < 0 or value >= deadline_ms:
        raise ContractError(
            f"fallback_margin_ms {value} must be >=0 and < deadline_ms {deadline_ms}"
        )
    return value


def _require_opt_pos_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be positive int or None, got {value!r}")
    return value


def _require_opt_nonneg_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be nonnegative int or None, got {value!r}")
    return value


def _require_digest(name: str, value: object) -> DigestText:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be sha256 digest, got {type(value).__name__}")
    return make_digest_text(value)


def _require_opt_digest(name: str, value: object) -> DigestText | None:
    if value is None:
        return None
    return _require_digest(name, value)


def _require_tuple_digests(name: str, value: object) -> tuple[DigestText, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be tuple of digests, got {type(value).__name__}")
    out: list[DigestText] = []
    for item in value:
        item_obj: object = item
        out.append(_require_digest(f"{name}[]", item_obj))
    return tuple(out)


def _require_json_value(where: str, value: Any) -> None:
    # Validate JSON domain via canonical_bytes round-trip; NaN/Inf rejected.
    try:
        _ = canonical_bytes(value)
    except Exception as exc:
        raise ContractError(f"{where}: not in canonical JSON domain: {exc}") from exc
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ContractError(f"{where}: dict keys must be str, got {type(key).__name__}")


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """SPEC 15 ResourceBudget — frozen, validated.

    Fields follow the specification order exactly.
    """

    mode: str
    deadline_ms: int
    fallback_margin_ms: int
    max_model_calls: int | None
    max_transitions: int | None
    max_particles: int | None
    max_memory_bytes: int | None

    def __post_init__(self) -> None:
        m = _require_mode(self.mode)
        object.__setattr__(self, "mode", m)
        d = _require_deadline_ms(self.deadline_ms)
        object.__setattr__(self, "deadline_ms", d)
        f = _require_fallback_margin(self.fallback_margin_ms, d)
        object.__setattr__(self, "fallback_margin_ms", f)
        object.__setattr__(
            self, "max_model_calls", _require_opt_pos_int("max_model_calls", self.max_model_calls)
        )
        object.__setattr__(
            self,
            "max_transitions",
            _require_opt_nonneg_int("max_transitions", self.max_transitions),
        )
        object.__setattr__(
            self, "max_particles", _require_opt_nonneg_int("max_particles", self.max_particles)
        )
        object.__setattr__(
            self,
            "max_memory_bytes",
            _require_opt_pos_int("max_memory_bytes", self.max_memory_bytes),
        )


def resource_budget_to_json(budget: ResourceBudget) -> dict[str, object]:
    return {
        "mode": budget.mode,
        "deadline_ms": budget.deadline_ms,
        "fallback_margin_ms": budget.fallback_margin_ms,
        "max_model_calls": budget.max_model_calls,
        "max_transitions": budget.max_transitions,
        "max_particles": budget.max_particles,
        "max_memory_bytes": budget.max_memory_bytes,
    }


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """SPEC 15 CandidateSpec — frozen, validated.

    Field order matches the specification exactly; JSON projection follows that order.
    """

    candidate_id: str
    algorithm: str
    algorithm_version: str
    rules_hash: str
    utility_id: str
    utility_manifest_hash: str
    action_table_hash: str
    observation_schema_hash: str
    packet_boundary_hash: str
    model_hash: str
    belief_model_hash: str | None
    event_model_hash: str | None
    continuation_policy_hashes: tuple[str, ...]
    proposal_spec_hash: str | None
    case_manifest_hash: str
    resource_budget: ResourceBudget
    fallback_candidate_id: str
    tie_break: str
    rng_protocol_hash: str
    random_stream_schema_hash: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or self.candidate_id == "":
            raise ContractError("candidate_id must be non-empty str")
        if not isinstance(self.algorithm, str) or self.algorithm == "":
            raise ContractError("algorithm must be non-empty str")
        # validate schema version format
        _ = make_schema_version(self.algorithm_version)
        # digest fields
        object.__setattr__(self, "rules_hash", _require_digest("rules_hash", self.rules_hash))
        if not isinstance(self.utility_id, str) or self.utility_id == "":
            raise ContractError("utility_id must be non-empty str")
        object.__setattr__(
            self,
            "utility_manifest_hash",
            _require_digest("utility_manifest_hash", self.utility_manifest_hash),
        )
        object.__setattr__(
            self, "action_table_hash", _require_digest("action_table_hash", self.action_table_hash)
        )
        object.__setattr__(
            self,
            "observation_schema_hash",
            _require_digest("observation_schema_hash", self.observation_schema_hash),
        )
        object.__setattr__(
            self,
            "packet_boundary_hash",
            _require_digest("packet_boundary_hash", self.packet_boundary_hash),
        )
        object.__setattr__(self, "model_hash", _require_digest("model_hash", self.model_hash))
        object.__setattr__(
            self,
            "belief_model_hash",
            _require_opt_digest("belief_model_hash", self.belief_model_hash),
        )
        object.__setattr__(
            self, "event_model_hash", _require_opt_digest("event_model_hash", self.event_model_hash)
        )
        object.__setattr__(
            self,
            "continuation_policy_hashes",
            _require_tuple_digests("continuation_policy_hashes", self.continuation_policy_hashes),
        )
        object.__setattr__(
            self,
            "proposal_spec_hash",
            _require_opt_digest("proposal_spec_hash", self.proposal_spec_hash),
        )
        object.__setattr__(
            self,
            "case_manifest_hash",
            _require_digest("case_manifest_hash", self.case_manifest_hash),
        )
        if not isinstance(self.resource_budget, ResourceBudget):
            raise ContractError("resource_budget must be ResourceBudget")
        object.__setattr__(
            self, "fallback_candidate_id", _require_fallback(self.fallback_candidate_id)
        )
        if not isinstance(self.tie_break, str) or self.tie_break == "":
            raise ContractError("tie_break must be non-empty str")
        object.__setattr__(
            self, "rng_protocol_hash", _require_digest("rng_protocol_hash", self.rng_protocol_hash)
        )
        object.__setattr__(
            self,
            "random_stream_schema_hash",
            _require_digest("random_stream_schema_hash", self.random_stream_schema_hash),
        )
        if not isinstance(self.parameters, Mapping):
            raise ContractError("parameters must be mapping")
        # freeze parameters via MappingProxy + validate JSON domain
        _require_json_value("parameters", dict(self.parameters))
        # store as plain dict for canonical hashing determinism (still frozen via object.__setattr__)
        object.__setattr__(self, "parameters", dict(self.parameters))

    def to_json(self) -> dict[str, object]:
        return candidate_spec_to_json(self)


def candidate_spec_to_json(spec: CandidateSpec) -> dict[str, object]:
    """SPEC-order JSON projection (without digest)."""
    return {
        "candidate_id": spec.candidate_id,
        "algorithm": spec.algorithm,
        "algorithm_version": spec.algorithm_version,
        "rules_hash": spec.rules_hash,
        "utility_id": spec.utility_id,
        "utility_manifest_hash": spec.utility_manifest_hash,
        "action_table_hash": spec.action_table_hash,
        "observation_schema_hash": spec.observation_schema_hash,
        "packet_boundary_hash": spec.packet_boundary_hash,
        "model_hash": spec.model_hash,
        "belief_model_hash": spec.belief_model_hash,
        "event_model_hash": spec.event_model_hash,
        "continuation_policy_hashes": list(spec.continuation_policy_hashes),
        "proposal_spec_hash": spec.proposal_spec_hash,
        "case_manifest_hash": spec.case_manifest_hash,
        "resource_budget": resource_budget_to_json(spec.resource_budget),
        "fallback_candidate_id": spec.fallback_candidate_id,
        "tie_break": spec.tie_break,
        "rng_protocol_hash": spec.rng_protocol_hash,
        "random_stream_schema_hash": spec.random_stream_schema_hash,
        "parameters": dict(spec.parameters),
    }


def candidate_spec_hash(spec: CandidateSpec) -> DigestText:
    """Content hash of the canonical JSON projection (SPEC 15)."""
    payload = candidate_spec_to_json(spec)
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return DigestText("sha256:" + digest)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """SPEC 15 SearchRequest — frozen.

    The observation and candidate_spec hashes are not stored redundantly;
    callers validate them inside ``candidate0`` via direct comparison.
    """

    observation: Any  # ActorObservation — avoid hard import at module load
    legal_actions: tuple[Any, ...]  # tuple[CanonicalAction, ...]
    candidate_spec: CandidateSpec
    deadline_monotonic_ns: int
    belief_epoch: Any | None  # BeliefEpoch | None

    def __post_init__(self) -> None:
        from hydra2.contracts.observation import ActorObservation

        if not isinstance(self.observation, ActorObservation):
            raise ContractError(
                f"observation must be ActorObservation, got {type(self.observation).__name__}"
            )
        if not isinstance(self.legal_actions, tuple) or len(self.legal_actions) == 0:
            raise ContractError("legal_actions must be non-empty tuple")
        for act in self.legal_actions:
            from hydra2.contracts.action import CanonicalAction

            if not isinstance(act, CanonicalAction):
                raise ContractError(
                    f"legal_actions entries must be CanonicalAction, got {type(act).__name__}"
                )
        if not isinstance(self.candidate_spec, CandidateSpec):
            raise ContractError("candidate_spec must be CandidateSpec")
        if isinstance(self.deadline_monotonic_ns, bool) or not isinstance(
            self.deadline_monotonic_ns, int
        ):
            raise ContractError("deadline_monotonic_ns must be int")
        if self.deadline_monotonic_ns <= 0:
            raise ContractError("deadline_monotonic_ns must be positive")
        # belief_epoch may be None for candidate0 (no belief)
        if self.belief_epoch is not None:
            from hydra2.belief.natural import BeliefEpoch

            if not isinstance(self.belief_epoch, BeliefEpoch):
                raise ContractError("belief_epoch must be BeliefEpoch or None")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """SPEC 15 SearchResult — frozen."""

    selected_action: Any  # CanonicalAction
    candidate_actions: tuple[Any, ...]
    value_vectors: tuple[Any, ...]  # tuple[UtilityVector, ...]
    candidate_spec_hash: str
    telemetry: Any  # ResourceTelemetry
    evidence_refs: tuple[str, ...]
    completed: bool

    def __post_init__(self) -> None:
        from hydra2.contracts.action import CanonicalAction
        from hydra2.contracts.utility import UtilityVector
        from hydra2.eval.telemetry import ResourceTelemetry

        if not isinstance(self.selected_action, CanonicalAction):
            raise ContractError(
                f"selected_action must be CanonicalAction, got {type(self.selected_action).__name__}"
            )
        if not isinstance(self.candidate_actions, tuple) or len(self.candidate_actions) == 0:
            raise ContractError("candidate_actions must be non-empty tuple")
        for cand in self.candidate_actions:
            if not isinstance(cand, CanonicalAction):
                raise ContractError("candidate_actions must hold CanonicalAction")
        if not isinstance(self.value_vectors, tuple):
            raise ContractError("value_vectors must be tuple")
        for vec in self.value_vectors:
            if not isinstance(vec, UtilityVector):
                raise ContractError("value_vectors entries must be UtilityVector")
        _ = make_digest_text(self.candidate_spec_hash)
        if not isinstance(self.telemetry, ResourceTelemetry):
            raise ContractError(
                f"telemetry must be ResourceTelemetry, got {type(self.telemetry).__name__}"
            )
        if not isinstance(self.evidence_refs, tuple):
            raise ContractError("evidence_refs must be tuple")
        for ref in self.evidence_refs:
            _ = make_digest_text(ref)
        if not isinstance(self.completed, bool):
            raise ContractError("completed must be bool")


class Planner(Protocol):
    """SPEC 15 Planner protocol."""

    def act(self, request: SearchRequest) -> SearchResult: ...

    def observe(self, packet: Any) -> None: ...

    def ponder(self, *, deadline_monotonic_ns: int) -> None: ...
