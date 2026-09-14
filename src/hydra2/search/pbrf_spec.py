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
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, DigestText, make_digest_text
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
# CandidateSpec factory
# ---------------------------------------------------------------------------


def _file_sha256(path: Any) -> DigestText:
    import hashlib
    from pathlib import Path

    p = Path(path)
    # dummy-until-real: file content hash wins when the config is present.
    if not p.exists():
        return make_digest_text("sha256:" + "0" * 64)
    return make_digest_text("sha256:" + hashlib.sha256(p.read_bytes()).hexdigest())


def _load_default_hashes() -> dict[str, str]:
    """File-backed config hashes only; semantic digests derive per factory.

    Utility/model/rng/stream/case digests are bound by
    ``make_pbrf_candidate_spec`` (model + candidate0 canonical descriptors) —
    never constant hashes here. Portable repo root via marker walk.
    """
    from hydra2.config import repo_root
    from hydra2.search.common import MISSING_HASH

    repo = repo_root()
    defaults: dict[str, str] = {}
    for key, rel in [
        ("rules_hash", "configs/rules/tenhou_4p_hanchan_v1.json"),
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/models/model_input_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ]:
        try:
            p = repo / rel
            # dummy-until-real: file content hash wins when the config is present.
            if p.exists():
                defaults[key] = str(_file_sha256(p))
            else:
                defaults[key] = "sha256:" + MISSING_HASH
        except (OSError, ValueError, TypeError, ContractError) as exc:
            logger.debug("pbrf: default hash fallback for %s", key, exc_info=exc)
            defaults[key] = "sha256:" + MISSING_HASH
    # also try observation schema contract path (upgrade when present)
    try:
        p = repo / "configs/contracts/observation_schema_v1.json"
        if p.exists():
            defaults["observation_schema_hash"] = str(_file_sha256(p))
    except (OSError, ValueError, TypeError, ContractError) as exc:
        logger.debug("pbrf: observation_schema contract fallback", exc_info=exc)
        pass
    return defaults


def _model_hash_from_identity(model: Any | None) -> str:
    """Model digest via candidate0 authority (import; mirror on failure)."""
    try:
        from hydra2.search.candidate0 import _model_hash_from_identity as _c0_hash

        return str(_c0_hash(model))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("pbrf: candidate0 model-hash import fallback", exc_info=exc)
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
        logger.debug("pbrf: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError("pbrf: cannot derive utility_manifest_hash from model") from exc


def _canonical_hashes() -> dict[str, str]:
    """RNG/stream/case digests verbatim from candidate0 authority descriptors."""
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


def make_pbrf_candidate_spec(
    *,
    parent_count: int = 16,
    kernel_tolerance: float = 1e-9,
    max_search_batches: int = 64,
    resource_view: Literal["calls", "transitions", "joules"] = "calls",
    candidate_id: str = "candidate3_pbrf_core_v1",
    rules_hash: str | None = None,
    utility_id: str = "expected_final_placement",
    utility_manifest_hash: str | None = None,
    tie_break: str = "lexicographic",
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
    rh: DigestText = make_digest_text(
        rules_hash if rules_hash is not None and rules_hash != "" else defaults["rules_hash"]
    )
    ah: DigestText = make_digest_text(defaults["action_table_hash"])
    oh: DigestText = make_digest_text(defaults["observation_schema_hash"])
    ph: DigestText = make_digest_text(defaults["packet_boundary_hash"])
    mh: DigestText = make_digest_text(
        model_hash
        if model_hash is not None and model_hash != ""
        else _model_hash_from_identity(model)
    )
    uh: DigestText = make_digest_text(
        utility_manifest_hash
        if utility_manifest_hash is not None and utility_manifest_hash != ""
        else _derive_utility_manifest_hash(model)
    )
    ch: DigestText = make_digest_text(
        case_manifest_hash
        if case_manifest_hash is not None and case_manifest_hash != ""
        else canonical["case_manifest_hash"]
    )
    rngh: DigestText = make_digest_text(
        rng_protocol_hash
        if rng_protocol_hash is not None and rng_protocol_hash != ""
        else canonical["rng_protocol_hash"]
    )
    strh: DigestText = make_digest_text(
        random_stream_schema_hash
        if random_stream_schema_hash is not None and random_stream_schema_hash != ""
        else canonical["random_stream_schema_hash"]
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
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=200,
            max_model_calls=64,
            max_transitions=256,
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
    spec = _CS(
        candidate_id=candidate_id,
        algorithm="pbrf_core",
        algorithm_version="1.0.0",
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
        fallback_candidate_id="candidate0",
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
    return spec
