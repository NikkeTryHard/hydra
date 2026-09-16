# ruff: noqa: F401, N814  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; SIM102 nested contract guards; B905 intentionally non-strict action/legal zips; N814 upstream belief symbol casing). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 6 Gumbel factories — CandidateSpec builders for Gumbel and PUCT.

Owns the file-backed config-hash loader, the model/utility digest
binders, the candidate0-authority canonical hashes, and the frozen
``make_gumbel_candidate_spec`` / ``make_puct_candidate_spec`` factories.
Hash binding mirrors the candidate0 canonical descriptors: file-backed
configs from disk, utility/model from the live model, rng/stream/case
from the canonical descriptors; caller overrides still win.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from hydra2.artifacts.canonical import canonical_bytes as canonical_bytes
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_digest_text as make_digest_text
from hydra2.search.common import DEPLOYABLE_DEADLINE_MS as DEPLOYABLE_DEADLINE_MS
from hydra2.search.common import MISSING_HASH as MISSING_HASH
from hydra2.search.common import REPO_ROOT as REPO_ROOT
from hydra2.search.common import CandidateSpec as CandidateSpec
from hydra2.search.common import ResourceBudget as ResourceBudget
from hydra2.search.gumbel_config import GumbelSearchConfig as GumbelSearchConfig
from hydra2.search.gumbel_config import PuctConfig as PuctConfig
from hydra2.search.gumbel_core import logger as logger

__all__ = [
    "make_gumbel_candidate_spec",
    "make_puct_candidate_spec",
]


# ---------------------------------------------------------------------------
# Factories — CandidateSpec builders for gumbel and PUCT comparator
# ---------------------------------------------------------------------------


def _load_default_hashes() -> dict[str, str]:
    """File-backed config hashes only; semantic digests derive per factory.

    Utility/model/rng/stream/case digests are bound by the factories below
    (model + candidate0 canonical descriptors) — never placeholders here.
    """
    from pathlib import Path  # noqa: TC003

    repo = REPO_ROOT
    defaults: dict[str, str] = {}
    try:
        import hashlib as _hl

        from hydra2.search.common import _require_real_file

        def _sha(p: Path) -> str:
            real = _require_real_file(p, REPO_ROOT)
            return "sha256:" + _hl.sha256(real.read_bytes()).hexdigest()

        mapping = {
            "rules_hash": repo / "configs" / "rules" / "tenhou_4p_hanchan_v1.json",
            "action_table_hash": repo / "configs" / "contracts" / "action_table_v1.json",
            "observation_schema_hash": repo
            / "configs"
            / "contracts"
            / "observation_schema_v1.json",
            "packet_boundary_hash": repo / "configs" / "contracts" / "packet_boundary_v1.json",
        }
        for key, path in mapping.items():
            if not path.exists():
                raise ContractError(f"gumbel: required config missing: {path}")
            defaults[key] = _sha(path)
    except ContractError:
        raise
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        raise ContractError(f"gumbel: file-backed default hashes required: {exc}") from exc
    return defaults


def _model_hash_from_identity(model: Any | None) -> str:
    """Model digest via candidate0 authority (import; mirror on failure)."""
    try:
        from hydra2.search.candidate0 import _model_hash_from_identity as _c0_hash

        return str(_c0_hash(model))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("gumbel: candidate0 model-hash import fallback", exc_info=exc)
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
        logger.debug("gumbel: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError("gumbel: cannot derive utility_manifest_hash from model") from exc


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


def make_gumbel_candidate_spec(
    *,
    halving_rounds: int = 2,
    visits_per_round: tuple[int, ...] = (8, 8),
    max_depth: int = 6,
    max_model_calls: int | None = 32,
    max_transitions: int | None = 64,
    tie_break: str = "lowest_action_id",
    resource_view: str = "calls",
    candidate_id: str = "candidate6",
    case_manifest_hash: str | None = None,
    model_hash: str | None = None,
    rules_hash: str | None = None,
) -> CandidateSpec:
    """Build frozen CandidateSpec for Gumbel search.

    All hash fields are bound before cases: file-backed configs from disk,
    utility/model from the live model, rng/stream/case from the candidate0
    canonical descriptors. Caller ``rules_hash``/``model_hash``/
    ``case_manifest_hash`` overrides still win.
    """
    defaults = _load_default_hashes()
    canonical = _canonical_hashes()
    utility_manifest_hash = _derive_utility_manifest_hash(None)
    bound_model_hash = (
        model_hash
        if model_hash is not None and model_hash != ""
        else _model_hash_from_identity(None)
    )
    bound_case_hash = (
        case_manifest_hash
        if case_manifest_hash is not None and case_manifest_hash != ""
        else canonical["case_manifest_hash"]
    )
    cfg = GumbelSearchConfig(
        halving_rounds=halving_rounds,
        visits_per_round=visits_per_round,
        max_depth=max_depth,
        max_model_calls=max_model_calls,
        max_transitions=max_transitions,
        tie_break=tie_break,
        candidate_id=candidate_id,
        resource_view=resource_view,
    )
    from hydra2.search.common import CandidateSpec as _CS
    from hydra2.search.common import ResourceBudget as _RB

    budget = _RB(
        mode="gameplay_5s",
        deadline_ms=DEPLOYABLE_DEADLINE_MS,
        fallback_margin_ms=200,
        max_model_calls=cfg.max_model_calls,
        max_transitions=cfg.max_transitions,
        max_particles=16,
        max_memory_bytes=None,
    )
    spec = _CS(
        candidate_id=cfg.candidate_id,
        algorithm="gumbel_search",
        algorithm_version="1.0.0",
        rules_hash=rules_hash
        if rules_hash is not None and rules_hash != ""
        else defaults["rules_hash"],
        utility_id="expected_final_placement",
        utility_manifest_hash=utility_manifest_hash,
        action_table_hash=defaults["action_table_hash"],
        observation_schema_hash=defaults["observation_schema_hash"],
        packet_boundary_hash=defaults["packet_boundary_hash"],
        model_hash=bound_model_hash,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=bound_case_hash,
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break=cfg.tie_break,
        rng_protocol_hash=canonical["rng_protocol_hash"],
        random_stream_schema_hash=canonical["random_stream_schema_hash"],
        parameters={
            "halving_rounds": cfg.halving_rounds,
            "visits_per_round": list(cfg.visits_per_round),
            "max_depth": cfg.max_depth,
            "max_model_calls": cfg.max_model_calls,
            "max_transitions": cfg.max_transitions,
            "tie_break": cfg.tie_break,
            "candidate_id": cfg.candidate_id,
            "resource_view": cfg.resource_view,
        },
    )
    return spec


def make_puct_candidate_spec(
    *,
    puct_c: float = 1.5,
    max_depth: int = 6,
    max_model_calls: int | None = 32,
    max_transitions: int | None = 64,
    num_simulations: int = 16,
    tie_break: str = "lowest_action_id",
    resource_view: str = "calls",
    candidate_id: str = "puct_baseline",
    case_manifest_hash: str | None = None,
) -> CandidateSpec:
    """Build frozen CandidateSpec for PUCT comparator (matched budget).

    Hash binding mirrors ``make_gumbel_candidate_spec``: file-backed configs
    from disk, utility/model from the live model, rng/stream/case from the
    candidate0 canonical descriptors. Caller ``case_manifest_hash`` wins.
    """
    defaults = _load_default_hashes()
    canonical = _canonical_hashes()
    bound_case_hash = (
        case_manifest_hash
        if case_manifest_hash is not None and case_manifest_hash != ""
        else canonical["case_manifest_hash"]
    )
    cfg = PuctConfig(
        puct_c=puct_c,
        max_depth=max_depth,
        max_model_calls=max_model_calls,
        max_transitions=max_transitions,
        num_simulations=num_simulations,
        tie_break=tie_break,
        candidate_id=candidate_id,
        resource_view=resource_view,
    )
    from hydra2.search.common import CandidateSpec as _CS
    from hydra2.search.common import ResourceBudget as _RB

    budget = _RB(
        mode="gameplay_5s",
        deadline_ms=DEPLOYABLE_DEADLINE_MS,
        fallback_margin_ms=200,
        max_model_calls=cfg.max_model_calls,
        max_transitions=cfg.max_transitions,
        max_particles=16,
        max_memory_bytes=None,
    )
    spec = _CS(
        candidate_id=cfg.candidate_id,
        algorithm="puct_search",
        algorithm_version="1.0.0",
        rules_hash=defaults["rules_hash"],
        utility_id="expected_final_placement",
        utility_manifest_hash=_derive_utility_manifest_hash(None),
        action_table_hash=defaults["action_table_hash"],
        observation_schema_hash=defaults["observation_schema_hash"],
        packet_boundary_hash=defaults["packet_boundary_hash"],
        model_hash=_model_hash_from_identity(None),
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=bound_case_hash,
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break=cfg.tie_break,
        rng_protocol_hash=canonical["rng_protocol_hash"],
        random_stream_schema_hash=canonical["random_stream_schema_hash"],
        parameters={
            "puct_c": cfg.puct_c,
            "max_depth": cfg.max_depth,
            "max_model_calls": cfg.max_model_calls,
            "max_transitions": cfg.max_transitions,
            "num_simulations": cfg.num_simulations,
            "tie_break": cfg.tie_break,
            "candidate_id": cfg.candidate_id,
            "resource_view": cfg.resource_view,
        },
    )
    return spec
