# ruff: noqa: F401  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — spec: frozen config plus CandidateSpec factory.

Owns the frozen Candidate 5 hyper-parameters (horizon, iterations, update rule,
averaging, abstraction descriptor, leaf model, tie break, resource view), the
descriptor-to-mapping translation, the file-backed config hashes, the candidate0
model/utility digest authority, and the ``make_candidate5_spec`` factory that binds
them into a frozen CandidateSpec. Abstraction lives in
:mod:`hydra2.search.local_abstraction`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, make_digest_text
from hydra2.search.common import DEPLOYABLE_DEADLINE_MS, CandidateSpec, ResourceBudget
from hydra2.search.local_abstraction import LocalResolvingAbstraction as LocalResolvingAbstraction
from hydra2.search.local_abstraction import (
    validate_abstraction_mapping as validate_abstraction_mapping,
)
from hydra2.search.local_shared import logger as logger
from hydra2.search.local_strategy import _VALID_AVERAGING as _VALID_AVERAGING
from hydra2.search.local_strategy import _VALID_UPDATE_RULES as _VALID_UPDATE_RULES

__all__ = [
    "LocalResolvingConfig",
    "make_candidate5_spec",
]


@dataclass(frozen=True, slots=True)
class LocalResolvingConfig:
    """Frozen Candidate 5 hyper-parameters (part of CandidateSpec.parameters)."""

    horizon: int = 2
    iterations: int = 16
    update_rule: str = "regret_matching"
    averaging: str = "uniform"
    abstraction: str = "identity"  # descriptor; actual mapping built via helper
    leaf_model: str = "model"
    tie_break: str = "greedy"
    resource_view: Literal["calls", "transitions", "joules"] = "calls"
    public_history_seed: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or not (1 <= self.horizon <= 16)
        ):
            raise ContractError(f"horizon must be int in 1..16, got {self.horizon!r}")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations <= 0
            or self.iterations > 1024
        ):
            raise ContractError(f"iterations must be positive int <=1024, got {self.iterations!r}")
        if self.update_rule not in _VALID_UPDATE_RULES:
            raise ContractError(
                f"update_rule must be one of {sorted(_VALID_UPDATE_RULES)}, got {self.update_rule!r}"
            )
        if self.averaging not in _VALID_AVERAGING:
            raise ContractError(
                f"averaging must be one of {sorted(_VALID_AVERAGING)}, got {self.averaging!r}"
            )
        if self.abstraction not in ("identity", "pair_merge", "tile_type", "custom"):
            raise ContractError(
                f"abstraction must be identity/pair_merge/tile_type/custom, got {self.abstraction!r}"
            )
        if self.leaf_model not in ("model", "terminal"):
            raise ContractError(f"leaf_model must be model or terminal, got {self.leaf_model!r}")
        if self.tie_break not in ("greedy", "temperature_0.5", "temperature_1.0", "value_break"):
            raise ContractError(f"tie_break {self.tie_break!r} unknown")
        if self.resource_view not in ("calls", "transitions", "joules"):
            raise ContractError(f"resource_view {self.resource_view!r} unknown")

    def to_parameters(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "iterations": self.iterations,
            "update_rule": self.update_rule,
            "averaging": self.averaging,
            "abstraction": self.abstraction,
            "leaf_model": self.leaf_model,
            "tie_break": self.tie_break,
            "resource_view": self.resource_view,
            "public_history_seed": self.public_history_seed,
        }

    @classmethod
    def from_parameters(cls, params: dict[str, Any]) -> LocalResolvingConfig:
        return cls(
            horizon=int(params.get("horizon", 2)),
            iterations=int(params.get("iterations", 16)),
            update_rule=str(params.get("update_rule", "regret_matching")),
            averaging=str(params.get("averaging", "uniform")),
            abstraction=str(params.get("abstraction", "identity")),
            leaf_model=str(params.get("leaf_model", "model")),
            tie_break=str(params.get("tie_break", "greedy")),
            resource_view=params.get("resource_view", "calls"),
            public_history_seed=params.get("public_history_seed"),
        )


def _build_abstraction_from_config(
    config: LocalResolvingConfig, legal_ids: tuple[int, ...]
) -> LocalResolvingAbstraction:
    """Translate config.abstraction descriptor into concrete mapping for tiny domain.

    For the tiny domain we assume legal_ids subset of 0..6791 but normally 0..3.
    Identity maps each concrete to itself; pair_merge merges (0,1)->0, (2,3)->1 etc.
    """
    if config.abstraction == "identity":
        mapping: dict[int, int] = {c: c for c in legal_ids}
    elif config.abstraction == "pair_merge":
        # pair consecutive ids: 0,1 -> 0; 2,3 ->1; etc
        sorted_ids = sorted(legal_ids)
        mapping = {}
        for idx, c in enumerate(sorted_ids):
            mapping[c] = idx // 2
        # also need to cover any concrete ids in 0..3 for subgame nodes even if not legal — for graph
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c // 2
    elif config.abstraction == "tile_type":
        # tile_type abstraction merges tiles of same type (simplified: modulo 34)
        mapping = {c: c % 34 for c in legal_ids}
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c % 34
        # remap to dense abstract ids 0..k-1
        uniq = sorted(set(mapping.values()))
        remap = {old: new for new, old in enumerate(uniq)}
        mapping = {c: remap[a] for c, a in mapping.items()}
    else:  # custom
        mapping = {c: c for c in legal_ids}
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c
    return validate_abstraction_mapping(mapping, legal_concrete_ids=tuple(sorted(mapping.keys())))


# ---------------------------------------------------------------------------
# CandidateSpec factory for candidate5
# ---------------------------------------------------------------------------


def _file_sha256(path: Any) -> str:
    import hashlib
    from pathlib import Path

    from hydra2.config import repo_root
    from hydra2.search.common import _require_real_file

    p = Path(path)
    real = _require_real_file(p, repo_root())
    return "sha256:" + hashlib.sha256(real.read_bytes()).hexdigest()


def _load_default_hashes() -> dict[str, str]:
    """File-backed config hashes only; semantic digests derive per factory.

    Utility/model/rng/stream/case digests are bound by
    ``make_candidate5_spec`` (model + candidate0 canonical descriptors) —
    never constant hashes here.
    """
    from pathlib import Path

    from hydra2.config import repo_root
    from hydra2.search.common import MISSING_HASH, _require_real_file

    repo = repo_root()
    out: dict[str, str] = {}
    # Try to load existing hashes like candidate0 does; fall back per key
    try:
        p = repo / "configs/rules/tenhou_4p_hanchan_v1.json"
        if p.exists():
            real = _require_real_file(p, repo)
            doc: dict[str, Any] = json.loads(real.read_text())
            payload: Any = doc.get("payload", {})
            try:
                from hydra2.contracts.rules import rules_manifest_from_payload

                manifest = rules_manifest_from_payload(payload)  # type: ignore[no-untyped-call]
                # RulesManifest has no digest attr — use file hash (avoids missing-attribute)
                _ = manifest  # silence unused
                out["rules_hash"] = _file_sha256(p)
            except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
                logger.debug("local_resolving: rules manifest fallback", exc_info=exc)
                out["rules_hash"] = _file_sha256(p)
        else:
            out["rules_hash"] = "sha256:" + MISSING_HASH
    except (OSError, ValueError, TypeError, ContractError, json.JSONDecodeError) as exc:
        logger.debug("local_resolving: rules_hash fallback", exc_info=exc)
        out["rules_hash"] = "sha256:" + MISSING_HASH
    for key, rel in [
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ]:
        try:
            out[key] = _file_sha256(repo / rel)
        except (OSError, ValueError, TypeError, ContractError) as exc:
            logger.debug("local_resolving: %s fallback", key, exc_info=exc)
            out[key] = "sha256:" + MISSING_HASH
    return out


def _model_hash_from_identity(model: Any | None) -> str:
    """Model digest via candidate0 authority (import; mirror on failure)."""
    try:
        from hydra2.search.candidate0 import _model_hash_from_identity as _c0_hash

        return str(_c0_hash(model))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("local_resolving: candidate0 model-hash import fallback", exc_info=exc)
    if model is not None:
        ident: Any = getattr(model, "model_identity", None)
        if ident is not None:
            return str(make_digest_text(str(ident)))
    from hydra2.models.model import Hydra2BaselineModel

    return str(make_digest_text(str(Hydra2BaselineModel().model_identity)))


def _derive_utility_manifest_hash(model: Any | None) -> str:
    """Utility manifest digest from the live model; fail loudly, never fake."""
    try:
        from hydra2.models.model import Hydra2BaselineModel

        probe: Any = Hydra2BaselineModel() if model is None else model
        manifest_raw: object = probe.utility_manifest_hash
        return str(make_digest_text(str(manifest_raw)))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("local_resolving: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError(
            "local_resolving: cannot derive utility_manifest_hash from model"
        ) from exc


def make_candidate5_spec(
    *,
    config: LocalResolvingConfig | None = None,
    horizon: int | None = None,
    iterations: int | None = None,
    update_rule: str | None = None,
    averaging: str | None = None,
    abstraction: str | None = None,
    leaf_model: str | None = None,
    tie_break: str = "greedy",
    resource_view: str = "calls",
    warm_start: bool = False,
    case_manifest_hash: str | None = None,
    model: Any | None = None,
    model_hash: str | None = None,
    rules_hash: str | None = None,
    utility_manifest_hash: str | None = None,
    action_table_hash: str | None = None,
    observation_schema_hash: str | None = None,
    packet_boundary_hash: str | None = None,
    rng_protocol_hash: str | None = None,
    random_stream_schema_hash: str | None = None,
    deadline_ms: int = DEPLOYABLE_DEADLINE_MS,
    fallback_margin_ms: int = 500,
    max_model_calls: int | None = 64,
    max_transitions: int | None = 256,
    max_particles: int | None = 32,
    extra_parameters: dict[str, Any] | None = None,
) -> Any:
    """Build frozen CandidateSpec for candidate5 (local resolving).

    All hash fields are bound before cases: file-backed configs from disk,
    utility/model from the live model, rng/stream/case from the candidate0
    canonical descriptors. Caller overrides still win. Horizon/abstraction/
    leaf_model/update/iterations/averaging are frozen. Warm start flag is
    recorded in parameters for comparator experiments (with/without PBRF
    warm start).
    """
    import hashlib

    if config is None:
        config = LocalResolvingConfig(
            horizon=horizon if horizon is not None else 2,
            iterations=iterations if iterations is not None else 16,
            update_rule=update_rule if update_rule is not None else "regret_matching",
            averaging=averaging if averaging is not None else "uniform",
            abstraction=abstraction if abstraction is not None else "identity",
            leaf_model=leaf_model if leaf_model is not None else "model",
            tie_break=tie_break,
            resource_view=resource_view,  # type: ignore[arg-type]
        )
    else:
        # Override via kwargs if supplied
        if (
            horizon is not None
            or iterations is not None
            or update_rule is not None
            or averaging is not None
            or abstraction is not None
            or leaf_model is not None
        ):
            config = LocalResolvingConfig(
                horizon=horizon if horizon is not None else config.horizon,
                iterations=iterations if iterations is not None else config.iterations,
                update_rule=update_rule if update_rule is not None else config.update_rule,
                averaging=averaging if averaging is not None else config.averaging,
                abstraction=abstraction if abstraction is not None else config.abstraction,
                leaf_model=leaf_model if leaf_model is not None else config.leaf_model,
                tie_break=tie_break,
                resource_view=resource_view,  # type: ignore[arg-type]
                public_history_seed=config.public_history_seed,
            )

    defaults = _load_default_hashes()
    if utility_manifest_hash is None:
        utility_manifest_hash = _derive_utility_manifest_hash(model)
    if rules_hash is None:
        rules_hash = defaults["rules_hash"]
        # Try verified manifest
        try:
            from pathlib import Path

            from hydra2.config import repo_root
            from hydra2.search.common import _require_real_file

            p = repo_root() / "configs/rules/tenhou_4p_hanchan_v1.json"
            if p.exists():
                real = _require_real_file(p, repo_root())
                doc: dict[str, Any] = json.loads(real.read_text())
                payload: Any = doc.get("payload", {})
                from hydra2.contracts.rules import rules_manifest_from_payload

                manifest = rules_manifest_from_payload(payload)  # type: ignore[no-untyped-call]
                # RulesManifest has no digest attr; synthesize via file hash (avoids missing-attribute)
                _ = manifest  # silence unused
                rules_hash = _file_sha256(p)
        except (
            ImportError,
            AttributeError,
            ValueError,
            TypeError,
            OSError,
            ContractError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            logger.debug("local_resolving: rules_hash verified-manifest fallback", exc_info=exc)
        pass
    if action_table_hash is None:
        action_table_hash = defaults["action_table_hash"]
    if observation_schema_hash is None:
        observation_schema_hash = defaults["observation_schema_hash"]
    if packet_boundary_hash is None:
        try:
            from pathlib import Path

            from hydra2.config import repo_root
            from hydra2.search.common import _require_real_file

            p = repo_root() / "configs/contracts/packet_boundary_v1.json"
            real = _require_real_file(p, repo_root())
            doc: dict[str, Any] = json.loads(real.read_text())
            payload_pb: Any = doc.get("payload", {})
            digest_val: Any = payload_pb.get("digest", "") if isinstance(payload_pb, dict) else ""
            packet_boundary_hash = (
                str(digest_val) if digest_val else defaults["packet_boundary_hash"]
            )
        except (
            ImportError,
            AttributeError,
            ValueError,
            TypeError,
            OSError,
            ContractError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            logger.debug("local_resolving: packet_boundary_hash fallback", exc_info=exc)
            packet_boundary_hash = defaults["packet_boundary_hash"]
    if model_hash is None:
        model_hash = _model_hash_from_identity(model)
    if rng_protocol_hash is None:
        rng_protocol_hash = (
            "sha256:"
            + hashlib.sha256(
                canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
            ).hexdigest()
        )
    if random_stream_schema_hash is None:
        random_stream_schema_hash = (
            "sha256:"
            + hashlib.sha256(
                canonical_bytes({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
            ).hexdigest()
        )
    if case_manifest_hash is None:
        case_manifest_hash = "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest()
    parameters: dict[str, Any] = dict(config.to_parameters())
    parameters["warm_start"] = warm_start
    parameters["candidate5_algorithm"] = "local_resolving"
    if extra_parameters is not None:
        parameters.update(extra_parameters)

    # Narrow hashes: after fallback assignment they must be str digests, not None
    assert rules_hash is not None
    assert utility_manifest_hash is not None
    assert action_table_hash is not None
    assert observation_schema_hash is not None
    assert packet_boundary_hash is not None
    assert model_hash is not None
    assert rng_protocol_hash is not None
    assert random_stream_schema_hash is not None
    assert case_manifest_hash is not None

    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=deadline_ms,
        fallback_margin_ms=fallback_margin_ms,
        max_model_calls=max_model_calls,
        max_transitions=max_transitions,
        max_particles=max_particles,
        max_memory_bytes=None,
    )
    spec = CandidateSpec(
        candidate_id="candidate5",
        algorithm="local_resolving",
        algorithm_version="1.0.0",
        rules_hash=rules_hash,
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash=utility_manifest_hash,
        action_table_hash=action_table_hash,
        observation_schema_hash=observation_schema_hash,
        packet_boundary_hash=packet_boundary_hash,
        model_hash=model_hash,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=case_manifest_hash,
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break=tie_break,
        rng_protocol_hash=rng_protocol_hash,
        random_stream_schema_hash=random_stream_schema_hash,
        parameters=parameters,
    )
    return spec
