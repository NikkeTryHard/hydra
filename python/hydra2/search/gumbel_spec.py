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

import logging
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError as ContractError
from hydra2.search.common import (
    DEPLOYABLE_DEADLINE_MS as DEPLOYABLE_DEADLINE_MS,
)
from hydra2.search.common import (
    MISSING_HASH as MISSING_HASH,
)
from hydra2.search.common import REPO_ROOT as REPO_ROOT
from hydra2.search.common import CandidateSpec as CandidateSpec
from hydra2.search.gumbel_config import (
    GumbelSearchConfig as GumbelSearchConfig,
)
from hydra2.search.gumbel_config import (
    PuctConfig as PuctConfig,
)
from hydra2.search.gumbel_core import logger as logger

__all__ = [
    "make_gumbel_candidate_spec",
    "make_puct_candidate_spec",
]


# ---------------------------------------------------------------------------
# Factories — CandidateSpec builders for gumbel and PUCT comparator
# ---------------------------------------------------------------------------


def _leaf(name: str) -> Any:
    """Resolve one ``hydra2._native.search`` gumbel-spec leaf (fail closed)."""
    try:
        return getattr(_bridge_search, name)
    except AttributeError as exc:
        raise ImportError(
            f"hydra2._native.search.{name} missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _pick_hash(caller_value: str | None, default_value: str) -> str:
    """Caller-override-wins hash pick via the bridge (None/empty → default)."""
    try:
        return _leaf("gumbel_spec_pick_hash")(caller_value, default_value)  # pyrefly: ignore[unknown-argument-type] # untyped bridge pick fn
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


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
        from hydra2.search.candidate0_frozen import _model_hash_from_identity as _c0_hash

        return _c0_hash(model)  # pyrefly: ignore[unknown-argument-type] # untyped candidate0 hash
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("gumbel: candidate0 model-hash import fallback", exc_info=exc)
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
        logger.debug("gumbel: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError("gumbel: cannot derive utility_manifest_hash from model") from exc


def _canonical_hashes() -> dict[str, str]:
    """RNG/stream/case digests verbatim from candidate0 authority descriptors."""
    try:
        hashes: tuple[str, str, str] = _leaf("gumbel_spec_canonical_hashes")()  # pyrefly: ignore[unknown-argument-type] # untyped bridge hashes fn
        rng_hash, stream_hash, case_hash = hashes
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return {
        "rng_protocol_hash": rng_hash,
        "random_stream_schema_hash": stream_hash,
        "case_manifest_hash": case_hash,
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
    bound_model_hash = _pick_hash(model_hash, _model_hash_from_identity(None))
    bound_case_hash = _pick_hash(case_manifest_hash, canonical["case_manifest_hash"])
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
        mode=_leaf("GUMBEL_SPEC_BUDGET_MODE"),
        deadline_ms=DEPLOYABLE_DEADLINE_MS,
        fallback_margin_ms=_leaf("GUMBEL_SPEC_FALLBACK_MARGIN_MS"),
        max_model_calls=cfg.max_model_calls,
        max_transitions=cfg.max_transitions,
        max_particles=_leaf("GUMBEL_SPEC_MAX_PARTICLES"),
        max_memory_bytes=None,
    )
    return _CS(
        candidate_id=cfg.candidate_id,
        algorithm=_leaf("GUMBEL_SPEC_ALGORITHM"),
        algorithm_version=_leaf("GUMBEL_SPEC_ALGORITHM_VERSION"),
        rules_hash=_pick_hash(rules_hash, defaults["rules_hash"]),
        # Utility id owned by the contracts bridge (utility::UTILITY_OBJECTIVE);
        # kept literal here, never duplicated.
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
        fallback_candidate_id=_leaf("GUMBEL_SPEC_FALLBACK_CANDIDATE_ID"),
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
    bound_case_hash = _pick_hash(case_manifest_hash, canonical["case_manifest_hash"])
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
        mode=_leaf("GUMBEL_SPEC_BUDGET_MODE"),
        deadline_ms=DEPLOYABLE_DEADLINE_MS,
        fallback_margin_ms=_leaf("GUMBEL_SPEC_FALLBACK_MARGIN_MS"),
        max_model_calls=cfg.max_model_calls,
        max_transitions=cfg.max_transitions,
        max_particles=_leaf("GUMBEL_SPEC_MAX_PARTICLES"),
        max_memory_bytes=None,
    )
    return _CS(
        candidate_id=cfg.candidate_id,
        algorithm=_leaf("PUCT_SPEC_ALGORITHM"),
        algorithm_version=_leaf("GUMBEL_SPEC_ALGORITHM_VERSION"),
        rules_hash=defaults["rules_hash"],
        # Utility id owned by the contracts bridge (utility::UTILITY_OBJECTIVE);
        # kept literal here, never duplicated.
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
        fallback_candidate_id=_leaf("GUMBEL_SPEC_FALLBACK_CANDIDATE_ID"),
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
