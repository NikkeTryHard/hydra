# ruff: noqa: N814, F841  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (SIM105 fallback-chain try/except-pass idiom; B007/F841 intentional scratch loop locals; B904 ContractError preconditions; N814 upstream casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 3 PBRF spec factory — file-backed hashes plus CandidateSpec builder.

Owns the file-backed config-hash loader, the candidate0-authority
model/utility digest binders, the canonical rng/stream/case hashes, and
the frozen :func:`make_pbrf_candidate_spec` factory that binds them
into a CandidateSpec. Hash binding mirrors the candidate0 canonical
descriptors: file-backed configs from disk, utility/model from the live
model, rng/stream/case from the canonical descriptors; caller overrides
still win. The partition vocabulary and guarded dependency flags live
in :mod:`hydra2.search.pbrf_partition` and the Planner runner in
:mod:`hydra2.search.pbrf_search` plus :mod:`hydra2.search.pbrf_act`.

The frozen factory scalars, the canonical rng/stream/case descriptors, and the
override-or-default pick execute in ``hydra2._native.search``
(``crates/bridge/src/pbrf_spec.rs``); this module keeps the ``ContractError``
shaping, the file reads, the model probes, and the ``CandidateSpec`` build.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal, cast

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, DigestText
from hydra2.search.common import ResourceBudget
from hydra2.search.pbrf_partition import PbrfConfig as PbrfConfig

logger = logging.getLogger(__name__)

__all__ = [
    "_canonical_hashes",
    "_derive_utility_manifest_hash",
    "_file_sha256",
    "_load_default_hashes",
    "_model_hash_from_identity",
    "make_pbrf_candidate_spec",
]

# ---------------------------------------------------------------------------
# Frozen factory defaults — Rust-owned mirrors with literal fallback
# ---------------------------------------------------------------------------
# Single source is ``hydra2._native.search`` (``crates/bridge/src/pbrf_spec.rs``);
# the literals are the stale-.so fallback so this module still imports before
# MAIN wires the new attrs (``eval/schedule.py`` hasattr-fallback shape).
_PBRF_DEFAULT_PARENT_COUNT: int = int(getattr(_bridge_search, "PBRF_SPEC_DEFAULT_PARENT_COUNT", 16))
_PBRF_DEFAULT_KERNEL_TOLERANCE: float = float(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_KERNEL_TOLERANCE", 1e-9)
)
_PBRF_DEFAULT_MAX_SEARCH_BATCHES: int = int(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_MAX_SEARCH_BATCHES", 64)
)
_PBRF_DEFAULT_RESOURCE_VIEW: Literal["calls", "transitions", "joules"] = getattr(
    _bridge_search, "PBRF_SPEC_DEFAULT_RESOURCE_VIEW", "calls"
)
_PBRF_DEFAULT_CANDIDATE_ID: str = str(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_CANDIDATE_ID", "candidate3_pbrf_core_v1")
)
_PBRF_DEFAULT_UTILITY_ID: str = str(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_UTILITY_ID", "expected_final_placement")
)
_PBRF_DEFAULT_TIE_BREAK: str = str(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_TIE_BREAK", "lexicographic")
)
_PBRF_ALGORITHM: str = str(getattr(_bridge_search, "PBRF_SPEC_ALGORITHM", "pbrf_core"))
_PBRF_ALGORITHM_VERSION: str = str(getattr(_bridge_search, "PBRF_SPEC_ALGORITHM_VERSION", "1.0.0"))
_PBRF_FALLBACK_CANDIDATE_ID: str = str(
    getattr(_bridge_search, "PBRF_SPEC_FALLBACK_CANDIDATE_ID", "candidate0")
)
_PBRF_DEFAULT_BUDGET_MODE: str = str(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_BUDGET_MODE", "gameplay_5s")
)
_PBRF_DEFAULT_BUDGET_DEADLINE_MS: int = int(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_BUDGET_DEADLINE_MS", 5000)
)
_PBRF_DEFAULT_BUDGET_FALLBACK_MARGIN_MS: int = int(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_BUDGET_FALLBACK_MARGIN_MS", 200)
)
_PBRF_DEFAULT_BUDGET_MAX_MODEL_CALLS: int = int(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_BUDGET_MAX_MODEL_CALLS", 64)
)
_PBRF_DEFAULT_BUDGET_MAX_TRANSITIONS: int = int(
    getattr(_bridge_search, "PBRF_SPEC_DEFAULT_BUDGET_MAX_TRANSITIONS", 256)
)
_PBRF_RULES_CONFIG_REL: str = str(
    getattr(_bridge_search, "PBRF_SPEC_RULES_CONFIG_REL", "configs/rules/tenhou_4p_hanchan_v1.json")
)
_PBRF_ACTION_TABLE_CONFIG_REL: str = str(
    getattr(
        _bridge_search,
        "PBRF_SPEC_ACTION_TABLE_CONFIG_REL",
        "configs/contracts/action_table_v1.json",
    )
)
_PBRF_OBSERVATION_SCHEMA_CONFIG_REL: str = str(
    getattr(
        _bridge_search,
        "PBRF_SPEC_OBSERVATION_SCHEMA_CONFIG_REL",
        "configs/models/model_input_v1.json",
    )
)
_PBRF_PACKET_BOUNDARY_CONFIG_REL: str = str(
    getattr(
        _bridge_search,
        "PBRF_SPEC_PACKET_BOUNDARY_CONFIG_REL",
        "configs/contracts/packet_boundary_v1.json",
    )
)
_PBRF_OBSERVATION_SCHEMA_CONTRACT_REL: str = str(
    getattr(
        _bridge_search,
        "PBRF_SPEC_OBSERVATION_SCHEMA_CONTRACT_REL",
        "configs/contracts/observation_schema_v1.json",
    )
)

# ---------------------------------------------------------------------------
# CandidateSpec factory
# ---------------------------------------------------------------------------


def _file_sha256(path: Any) -> DigestText:
    import hashlib
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise ContractError(f"pbrf: required config missing: {p}")
    return _bridge_contracts.make_digest_text(
        "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    )


def _load_default_hashes() -> dict[str, str]:
    """File-backed config hashes only; semantic digests derive per factory.

    Utility/model/rng/stream/case digests are bound by
    ``make_pbrf_candidate_spec`` (model + candidate0 canonical descriptors) —
    never constant hashes here. Portable repo root via marker walk.
    """
    from hydra2.config import repo_root

    repo = repo_root()
    defaults: dict[str, str] = {}
    for key, rel in [
        ("rules_hash", _PBRF_RULES_CONFIG_REL),
        ("action_table_hash", _PBRF_ACTION_TABLE_CONFIG_REL),
        ("observation_schema_hash", _PBRF_OBSERVATION_SCHEMA_CONFIG_REL),
        ("packet_boundary_hash", _PBRF_PACKET_BOUNDARY_CONFIG_REL),
    ]:
        try:
            p = repo / rel
            if not p.exists():
                raise ContractError(f"pbrf: required config missing: {p}")
            defaults[key] = str(_file_sha256(p))
        except ContractError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise ContractError(f"pbrf: default hash required for {key}: {exc}") from exc
    # also try observation schema contract path (upgrade when present)
    try:
        p = repo / _PBRF_OBSERVATION_SCHEMA_CONTRACT_REL
        if p.exists():
            defaults["observation_schema_hash"] = str(_file_sha256(p))
    except (OSError, ValueError, TypeError, ContractError) as exc:
        logger.debug("pbrf: observation_schema contract fallback", exc_info=exc)
        pass
    return defaults


def _model_hash_from_identity(model: Any | None) -> str:
    """Model digest via candidate0 authority (import; mirror on failure)."""
    try:
        from hydra2.search.candidate0_frozen import _model_hash_from_identity as _c0_hash

        return _c0_hash(model)  # pyrefly: ignore[unknown-argument-type] # untyped candidate0 hash
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("pbrf: candidate0 model-hash import fallback", exc_info=exc)
    if model is not None:
        ident: Any = getattr(model, "model_identity", None)
        if ident is not None:
            return _bridge_contracts.make_digest_text(str(ident))  # pyrefly: ignore[unknown-argument-type] # untyped bridge digest
    from hydra2.models.model import Hydra2BaselineModel

    return _bridge_contracts.make_digest_text(str(Hydra2BaselineModel().model_identity))  # pyrefly: ignore[unknown-argument-type] # untyped bridge digest


def _derive_utility_manifest_hash(model: Any | None) -> str:
    """Utility manifest digest from the live model; fail loudly, never fake."""
    try:
        from hydra2.models.model import Hydra2BaselineModel

        probe: Any = Hydra2BaselineModel() if model is None else model
        manifest_raw: object = probe.utility_manifest_hash
        return _bridge_contracts.make_digest_text(str(manifest_raw))  # pyrefly: ignore[unknown-argument-type] # untyped bridge digest
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("pbrf: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError("pbrf: cannot derive utility_manifest_hash from model") from exc


def _canonical_hashes() -> dict[str, str]:
    """RNG/stream/case digests verbatim from candidate0 authority descriptors.

    Thin bridge delegate: ``hydra2._native.search.pbrf_spec_canonical_hashes``
    seals the frozen descriptors through the feed canon/digest owners; the
    oracle body below is the stale-.so fallback (``eval/statistics.py``
    try-leaf shape).
    """
    try:
        leaf: Any = _bridge_search.pbrf_spec_canonical_hashes
    except AttributeError:
        pass
    else:
        try:
            hashes: tuple[str, str, str] = leaf()
            rng_hash, stream_hash, case_hash = hashes
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        return {
            "rng_protocol_hash": rng_hash,
            "random_stream_schema_hash": stream_hash,
            "case_manifest_hash": case_hash,
        }
    return {
        "rng_protocol_hash": "sha256:"
        + hashlib.sha256(
            canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
        ).hexdigest(),
        "random_stream_schema_hash": "sha256:"
        + hashlib.sha256(
            canonical_bytes({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
        ).hexdigest(),
        "case_manifest_hash": "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest(),
    }


def _resolve_hash(override: str | None, default: str) -> str:
    """Override-or-default pick via the bridge with oracle fallback.

    A non-empty override wins; ``None``/empty reads as the bound default.
    Digest-shape validation stays in :func:`make_digest_text` (contracts owns
    digests); the bridge never defaults a malformed hash.
    """
    try:
        leaf = _bridge_search.pbrf_spec_resolve_hash  # pyrefly: ignore[unknown-variable-type] # untyped bridge leaf
    except AttributeError:
        return override if override is not None and override != "" else default
    try:
        return leaf(override, default)  # pyrefly: ignore[unknown-argument-type] # untyped bridge hash fn
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def make_pbrf_candidate_spec(
    *,
    parent_count: int = _PBRF_DEFAULT_PARENT_COUNT,
    kernel_tolerance: float = _PBRF_DEFAULT_KERNEL_TOLERANCE,
    max_search_batches: int = _PBRF_DEFAULT_MAX_SEARCH_BATCHES,
    resource_view: Literal["calls", "transitions", "joules"] = _PBRF_DEFAULT_RESOURCE_VIEW,
    candidate_id: str = _PBRF_DEFAULT_CANDIDATE_ID,
    rules_hash: str | None = None,
    utility_id: str = _PBRF_DEFAULT_UTILITY_ID,
    utility_manifest_hash: str | None = None,
    tie_break: str = _PBRF_DEFAULT_TIE_BREAK,
    resource_budget: Any | None = None,
    model: Any | None = None,
    model_hash: str | None = None,
    case_manifest_hash: str | None = None,
    rng_protocol_hash: str | None = None,
    random_stream_schema_hash: str | None = None,
) -> Any:
    """Build frozen CandidateSpec for PBRF core (Candidate 3).

    Mirrors ``make_candidate0_spec`` style but for PBRF. ``parent_count``,
    ``kernel_tolerance``, ``max_search_batches`` and ``resource_view`` are
    frozen into ``parameters`` and also reflected in ``PbrfConfig``.

    All hash fields are bound before cases: file-backed configs from disk,
    utility/model from the live model, rng/stream/case from the candidate0
    canonical descriptors. Caller overrides still win.
    """
    defaults = _load_default_hashes()
    canonical = _canonical_hashes()
    rh: DigestText = _bridge_contracts.make_digest_text(
        _resolve_hash(rules_hash, defaults["rules_hash"])
    )
    ah: DigestText = _bridge_contracts.make_digest_text(defaults["action_table_hash"])
    oh: DigestText = _bridge_contracts.make_digest_text(defaults["observation_schema_hash"])
    ph: DigestText = _bridge_contracts.make_digest_text(defaults["packet_boundary_hash"])
    # NOTE: model/utility keep the lazy ternary (never the eager _resolve_hash
    # call): the probes must not run when a caller override wins, and a failing
    # model must not break an explicitly overridden hash.
    mh: DigestText = _bridge_contracts.make_digest_text(
        model_hash
        if model_hash is not None and model_hash != ""
        else _model_hash_from_identity(model)
    )
    uh: DigestText = _bridge_contracts.make_digest_text(
        utility_manifest_hash
        if utility_manifest_hash is not None and utility_manifest_hash != ""
        else _derive_utility_manifest_hash(model)
    )
    ch: DigestText = _bridge_contracts.make_digest_text(
        _resolve_hash(case_manifest_hash, canonical["case_manifest_hash"])
    )
    rngh: DigestText = _bridge_contracts.make_digest_text(
        _resolve_hash(rng_protocol_hash, canonical["rng_protocol_hash"])
    )
    strh: DigestText = _bridge_contracts.make_digest_text(
        _resolve_hash(random_stream_schema_hash, canonical["random_stream_schema_hash"])
    )
    # Validate config
    cfg = PbrfConfig(
        parent_count=parent_count,
        kernel_tolerance=kernel_tolerance,
        max_search_batches=max_search_batches,
        resource_view=resource_view,
        tie_break=tie_break,
    )
    if resource_budget is None:
        resource_budget = ResourceBudget(
            mode=_PBRF_DEFAULT_BUDGET_MODE,
            deadline_ms=_PBRF_DEFAULT_BUDGET_DEADLINE_MS,
            fallback_margin_ms=_PBRF_DEFAULT_BUDGET_FALLBACK_MARGIN_MS,
            max_model_calls=_PBRF_DEFAULT_BUDGET_MAX_MODEL_CALLS,
            max_transitions=_PBRF_DEFAULT_BUDGET_MAX_TRANSITIONS,
            max_particles=parent_count,
            max_memory_bytes=None,
        )
    # Ensure max_particles reflects parent_count if not set
    try:
        if getattr(resource_budget, "max_particles", None) is None:
            # keep as parent_count
            pass
    except Exception:
        pass
    from hydra2.search.common import CandidateSpec as _CS  # local to avoid circular
    from hydra2.search.common import ResourceBudget as _CommonRB

    # Narrow resource_budget to common type — pbrf fallback and Any are handled via cast
    _rb_common: _CommonRB = cast("_CommonRB", resource_budget)
    return _CS(
        candidate_id=candidate_id,
        algorithm=_PBRF_ALGORITHM,
        algorithm_version=_PBRF_ALGORITHM_VERSION,
        rules_hash=rh,
        utility_id=utility_id,
        utility_manifest_hash=uh,
        action_table_hash=ah,
        observation_schema_hash=oh,
        packet_boundary_hash=ph,
        model_hash=mh,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=ch,
        resource_budget=_rb_common,
        fallback_candidate_id=_PBRF_FALLBACK_CANDIDATE_ID,
        tie_break=tie_break,
        rng_protocol_hash=rngh,
        random_stream_schema_hash=strh,
        parameters={
            "parent_count": parent_count,
            "kernel_tolerance": kernel_tolerance,
            "max_search_batches": max_search_batches,
            "resource_view": resource_view,
        },
    )
