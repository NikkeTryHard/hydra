"""Candidate 0 frozen policy — deterministic choice and spec factory.

Owns the frozen ``frozen_choice`` tie-break arms (greedy, frozen-temperature,
value-break), the file-backed config-hash loaders, the model-identity digest
binder, and the ``make_candidate0_spec`` factory that binds them into a frozen
CandidateSpec (SPEC 16.1). All hash fields bind before cases; tie-break and
fallback margin stay frozen. The single-evaluation act path lives in
:mod:`hydra2.search.candidate0_act`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

import torch

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, DigestText, make_digest_text
from hydra2.search.common import (
    DEPLOYABLE_DEADLINE_MS,
    HASH63_MOD,
    REPO_ROOT,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_file_sha256",
    "_load_default_hashes",
    "_model_hash_from_identity",
    "frozen_choice",
    "make_candidate0_spec",
]

# ---------------------------------------------------------------------------
# frozen choice
# ---------------------------------------------------------------------------


@torch.inference_mode()
def frozen_choice(
    probs: torch.Tensor,
    value_vector: torch.Tensor,
    tie_break: str,
    *,
    observation_hash: str | None = None,
    actor_seat: int = 0,
) -> int:
    """Deterministic masked choice bound to CandidateSpec.tie_break.

    - greedy: argmax, first max on ties (deterministic)
    - temperature_*: deterministic categorical sample with frozen seed derived
      from observation_hash + tie_break (call-order independent)
    - value_break: among max-prob ties within eps, pick max value_vector[actor]

    ``probs`` must already be masked (illegal == 0) and sum to 1 over legal.
    ``value_vector`` is [A] or [4] for the batch row.
    """
    if probs.ndim != 1:
        raise ContractError("frozen_choice expects 1D probs")
    if not isinstance(tie_break, str) or tie_break == "":
        raise ContractError("tie_break must be non-empty str")
    # greedy — argmax, first tie wins (torch.argmax is first)
    if tie_break == "greedy":
        return int(torch.argmax(probs).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
    if tie_break.startswith("temperature_"):
        # temperature frozen in the tie_break string itself, e.g. temperature_0.5
        try:
            temp_str = tie_break.split("_", 1)[1]
            temperature = float(temp_str)
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            raise ContractError(f"bad temperature tie_break {tie_break!r}: {exc}") from exc
        if temperature <= 0:
            raise ContractError(f"temperature must be >0, got {temperature}")
        # Apply temperature to probs: tempered ∝ p^(1/T) over legal support, renormalize.
        # Guard against zero probs: keep legal support only.
        legal_mask = probs > 0
        if not bool(legal_mask.any().item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            raise ContractError("frozen_choice: no legal entries in probs")
        tempered = torch.zeros_like(probs)
        # p^(1/T) for legal entries; illegal stays 0
        tempered[legal_mask] = torch.pow(probs[legal_mask].clamp(min=1e-12), 1.0 / temperature)
        tempered = tempered / tempered.sum().clamp(min=1e-12)
        # Deterministic sampling derived from observation_hash
        seed_material = (
            (
                observation_hash
                if observation_hash is not None and observation_hash != ""
                else "no_hash"
            )
            + ":"
            + tie_break
        )
        seed_hex = hashlib.sha256(seed_material.encode()).hexdigest()[:16]
        seed = int(seed_hex, 16) % HASH63_MOD
        gen = torch.Generator(device=probs.device)
        _ = gen.manual_seed(seed)
        # torch.multinomial is deterministic with generator
        sampled = torch.multinomial(tempered, num_samples=1, generator=gen)
        return int(sampled.item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
    if tie_break == "value_break":
        # Tie among max-prob entries within eps; break via value_vector[actor]
        max_prob = float(probs.max().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
        eps = 1e-9
        probs_list: list[float] = cast("list[float]", probs.tolist())
        candidates: list[int] = [i for i, p in enumerate(probs_list) if abs(p - max_prob) <= eps]
        if len(candidates) == 1:
            return candidates[0]
        # value_vector may be [4] per seat or [A] per action; handle both
        # For [4], we cannot map action tie to seat value — fall back to prob tie break via value_vector magnitude per action if sized A,
        # otherwise use first candidate.
        if value_vector.ndim == 1 and value_vector.numel() == len(probs):
            # per-action value proxy — pick max value among tied candidates
            vals = value_vector[candidates]
            best_local = int(torch.argmax(vals).item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call]
            return candidates[best_local]
        if value_vector.ndim == 1 and value_vector.numel() == 4:
            # per-seat vector: value_break is degenerate (single action vs seats); keep greedy tie order
            # Documented: value_break with per-seat vector keeps first max (no hidden info).
            return candidates[0]
        return candidates[0]
    raise ContractError(f"unknown tie_break {tie_break!r}")


# ---------------------------------------------------------------------------
# helpers: digest loaders for default spec
# ---------------------------------------------------------------------------


def _bridge_sha256_file(path: Path) -> str:
    """Chunked file digest via the bridge (bit-identical to the retired hashlib loop)."""
    try:
        from hydra2_replay_rs import canon_rng as bridge  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2 digest authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        ) from exc
    try:
        text = bridge.sha256_file(str(path))  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.canon_rng.sha256_file missing (stale .so); "
            "rebuild the bridge (`pixi run build-ext`)"
        ) from exc
    return str(text)


def _bridge_sha256_hex(data: bytes) -> str:
    """In-memory digest via the bridge (bit-identical to hashlib.sha256 hexdigest)."""
    try:
        from hydra2_replay_rs import canon_rng as bridge  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2 digest authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        ) from exc
    try:
        text = bridge.sha256_hex(bytes(data))  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.canon_rng.sha256_hex missing (stale .so); "
            "rebuild the bridge (`pixi run build-ext`)"
        ) from exc
    return str(text)


def _file_sha256(path: Path) -> DigestText:
    from hydra2.search.common import _require_real_file

    real = _require_real_file(Path(path), REPO_ROOT)
    return DigestText(_bridge_sha256_file(real))


def _load_default_hashes() -> dict[str, str]:
    repo = REPO_ROOT
    # Fall back to file sha if contract modules not importable at spec creation time
    out: dict[str, str] = {}
    for key, rel in (
        ("rules_hash", "configs/rules/tenhou_4p_hanchan_v1.json"),
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("event_schema_hash", "configs/contracts/event_schema_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
        ("model_input_hash", "configs/models/model_input_v1.json"),
    ):
        p = repo / rel
        if not p.exists():
            raise ContractError(f"candidate0: required config missing: {p}")
        out[key] = _file_sha256(p)
    # Try to upgrade to canonical contract digests where modules available
    try:
        from hydra2.contracts.observation import observation_schema_digest

        out["observation_schema_hash"] = str(observation_schema_digest())
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        logger.debug("candidate0: observation_schema_digest fallback", exc_info=exc)
        pass
    try:
        from hydra2.contracts.action import load_action_table

        tbl = load_action_table(repo / "configs/contracts/action_table_v1.json")
        out["action_table_hash"] = str(tbl.digest)
    except (
        ImportError,
        AttributeError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug("candidate0: load_action_table fallback", exc_info=exc)
        pass
    try:
        from hydra2.models.schema import model_input_schema_digest

        out["model_input_hash"] = str(model_input_schema_digest())
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        logger.debug("candidate0: model_input_schema_digest fallback", exc_info=exc)
        pass
    try:
        from hydra2.contracts.event_schema import load_event_schema

        evt: Any = load_event_schema(repo / "configs/contracts/event_schema_v1.json")
        tmp_digest: Any = getattr(evt, "digest", None)
        tmp_payload: Any = getattr(evt, "payload", {}).get("digest", "")
        digest: Any = tmp_digest if tmp_digest is not None and tmp_digest != "" else tmp_payload
        if digest is not None and digest != "":
            out["event_schema_hash"] = str(digest)
    except (
        ImportError,
        AttributeError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug("candidate0: load_event_schema fallback", exc_info=exc)
        pass
    return out


def _model_hash_from_identity(model: Any | None) -> DigestText:
    if model is not None:
        ident: Any = getattr(model, "model_identity", None)
        if ident is not None:
            return make_digest_text(str(ident))
        # Fallback: hash of model state dict keys (bridge digest, bit-identical)
        try:
            state: Any = model.state_dict()  # type: ignore[union-attr]
            keys_raw: Any = state.keys()
            keys_sorted: list[str] = sorted(keys_raw)
            payload: dict[str, list[str]] = {"keys": keys_sorted}
            return DigestText(_bridge_sha256_hex(canonical_bytes(payload)))
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            logger.debug("candidate0: model state_dict fallback", exc_info=exc)
            pass
    from hydra2.models.model import Hydra2BaselineModel

    m = Hydra2BaselineModel()
    return make_digest_text(str(m.model_identity))


# ---------------------------------------------------------------------------
# CandidateSpec factory for candidate0
# ---------------------------------------------------------------------------


def make_candidate0_spec(
    *,
    tie_break: str = "greedy",
    parameters: dict[str, Any] | None = None,
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
    max_model_calls: int | None = 1,
) -> Any:
    """Build the frozen CandidateSpec for candidate0.

    All hash fields are bound before cases; tie_break and fallback margin are
    frozen. Missing digests are derived from current repo configs/model so that
    a bare ``make_candidate0_spec()`` is reproducible and contract-bound.
    """
    from hydra2.search.common import CandidateSpec, ResourceBudget

    defaults = _load_default_hashes()
    # Utility manifest: live model digest required; no placeholder fallback.
    if utility_manifest_hash is None:
        try:
            from hydra2.models.model import Hydra2BaselineModel

            probe: Any = Hydra2BaselineModel() if model is None else model
            probe_hash_raw: Any = getattr(probe, "utility_manifest_hash", None)
            if probe_hash_raw is None or str(probe_hash_raw) == "":
                raise ContractError("candidate0: utility_manifest_hash missing from model")
            utility_manifest_hash = str(probe_hash_raw)
        except ContractError:
            raise
        except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
            raise ContractError(f"candidate0: utility_manifest_hash required: {exc}") from exc
        rules_hash = defaults["rules_hash"]
        # Prefer verified manifest digest when file contains envelope
        try:
            from hydra2.search.common import _require_real_file

            p = REPO_ROOT / "configs/rules/tenhou_4p_hanchan_v1.json"
            real = _require_real_file(p, REPO_ROOT)
            doc: Any = json.loads(real.read_text())
            payload: Any = doc.get("payload", {})
            # The file's payload digest is the rules manifest digest in hydra2 sense
            # but the repo stores it as artifact envelope; derive via file sha fallback is acceptable
            # Try to compute via rules module if available
            from hydra2.contracts.rules import rules_manifest_from_payload

            manifest: Any = rules_manifest_from_payload(payload)  # type: ignore[no-untyped-call]
            manifest_digest: Any = getattr(manifest, "digest", None)
            if manifest_digest is not None:
                rules_hash = str(manifest_digest)  # type: ignore[attr-defined]
        except (
            AttributeError,
            ValueError,
            TypeError,
            OSError,
            ImportError,
            json.JSONDecodeError,
        ) as exc:
            logger.debug("candidate0: rules_hash fallback", exc_info=exc)
            pass
        action_table_hash = defaults["action_table_hash"]
    if observation_schema_hash is None:
        observation_schema_hash = defaults["observation_schema_hash"]
    if packet_boundary_hash is None:
        # packet file envelope vs payload digest distinction: use payload digest
        try:
            from hydra2.search.common import _require_real_file

            p = REPO_ROOT / "configs/contracts/packet_boundary_v1.json"
            real = _require_real_file(p, REPO_ROOT)
            doc2: Any = json.loads(real.read_text())
            payload2: Any = doc2["payload"]
            digest_val: Any = payload2["digest"]
            packet_boundary_hash = str(digest_val)
        except (
            AttributeError,
            ValueError,
            TypeError,
            OSError,
            ImportError,
            json.JSONDecodeError,
            KeyError,
        ) as exc:
            logger.debug("candidate0: packet_boundary_hash fallback", exc_info=exc)
            packet_boundary_hash = defaults["packet_boundary_hash"]
        model_hash = str(_model_hash_from_identity(model))
    # RNG / stream schema placeholders — canonical JSON hashes of fixed descriptors
    # (bridge digests, bit-identical to the retired hashlib lines).
    if rng_protocol_hash is None:
        rng_protocol_hash = _bridge_sha256_hex(
            canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
        )
    if random_stream_schema_hash is None:
        random_stream_schema_hash = _bridge_sha256_hex(
            canonical_bytes({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
        )
    if case_manifest_hash is None:
        # Empty manifest hash (frozen before cases would be set externally)
        case_manifest_hash = _bridge_sha256_hex(canonical_bytes([]))
    # Unconditional narrowing: rules/action/model hashes are only defaulted inside
    # the utility/packet branches above, so callers passing those manifests but
    # omitting these hashes would otherwise flow str|None into CandidateSpec.
    if rules_hash is None:
        rules_hash = defaults["rules_hash"]
    if action_table_hash is None:
        action_table_hash = defaults["action_table_hash"]
    if model_hash is None:
        model_hash = str(_model_hash_from_identity(model))
    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=deadline_ms,
        fallback_margin_ms=fallback_margin_ms,
        max_model_calls=max_model_calls,
        max_transitions=0 if max_model_calls is not None else None,
        max_particles=0 if max_model_calls is not None else None,
        max_memory_bytes=None,
    )
    params_effective: dict[str, Any] = (
        parameters if parameters is not None else {"temperature": 0.0, "tie_break": tie_break}
    )
    spec = CandidateSpec(
        candidate_id="candidate0",
        algorithm="frozen_policy",
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
        parameters=dict(params_effective),
    )
    return spec
