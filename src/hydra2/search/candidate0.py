"""WP-08A Candidate 0 Frozen Policy — SPEC 16.1, Blueprint §7.

One model evaluation only, no particles/search/pondering/learning. Greedy,
frozen-temperature and value tie-break arms. Deadline fallback is Candidate 0
itself. Freeze selection/tie-break and fallback margin before cases.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, DigestText, make_digest_text
from hydra2.search.common import (
    DEPLOYABLE_DEADLINE_MS,
    HASH63_MOD,
    MISSING_HASH,
    PLACEHOLDER_1,
    REPO_ROOT,
)

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from hydra2.contracts.observation import ActorObservation

__all__ = [
    "FrozenCandidate0",
    "candidate0",
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
            observation_hash if observation_hash is not None and observation_hash != "" else "no_hash"
        ) + ":" + tie_break
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
        candidates: list[int] = [
            i for i, p in enumerate(probs_list) if abs(p - max_prob) <= eps
        ]
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


def _file_sha256(path: Path) -> DigestText:
    from hydra2.search.common import _require_real_file

    real = _require_real_file(Path(path), REPO_ROOT)
    return DigestText("sha256:" + hashlib.sha256(real.read_bytes()).hexdigest())


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
        out[key] = _file_sha256(p) if p.exists() else "sha256:" + MISSING_HASH
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
    except (ImportError, AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.debug("candidate0: load_action_table fallback", exc_info=exc)
        pass
    try:
        from hydra2.models.schema import model_input_schema_digest

        out["model_input_hash"] = str(model_input_schema_digest())
    except (ImportError, AttributeError, OSError, ValueError, TypeError) as exc:
        logger.debug("candidate0: model_input_schema_digest fallback", exc_info=exc)
        pass
    try:
        from hydra2.contracts.event import load_event_schema

        evt: Any = load_event_schema(repo / "configs/contracts/event_schema_v1.json")
        tmp_digest: Any = getattr(evt, "digest", None)
        tmp_payload: Any = getattr(evt, "payload", {}).get("digest", "")
        digest: Any = tmp_digest if tmp_digest is not None and tmp_digest != "" else tmp_payload
        if digest is not None and digest != "":
            out["event_schema_hash"] = str(digest)
    except (ImportError, AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.debug("candidate0: load_event_schema fallback", exc_info=exc)
        pass
    return out


def _model_hash_from_identity(model: Any | None) -> DigestText:
    if model is not None:
        ident: Any = getattr(model, "model_identity", None)
        if ident is not None:
            return make_digest_text(str(ident))
        # Fallback: hash of model state dict keys
        try:
            state: Any = model.state_dict()  # type: ignore[union-attr]
            keys_raw: Any = state.keys()
            keys_sorted: list[str] = sorted(keys_raw)
            payload: dict[str, list[str]] = {"keys": keys_sorted}
            return DigestText("sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest())
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
    # Utility manifest: derive from Synthetic golden manifest identical to model init
    if utility_manifest_hash is None:
        try:
            from hydra2.models.model import Hydra2BaselineModel

            probe: Any = Hydra2BaselineModel() if model is None else model
            probe_hash_raw: Any = getattr(probe, "utility_manifest_hash", "sha256:" + MISSING_HASH)
            utility_manifest_hash = str(probe_hash_raw)
        except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
            logger.debug("candidate0: utility_manifest_hash fallback", exc_info=exc)
            utility_manifest_hash = "sha256:" + PLACEHOLDER_1
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
        except (AttributeError, ValueError, TypeError, OSError, ImportError, json.JSONDecodeError) as exc:
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
        except (AttributeError, ValueError, TypeError, OSError, ImportError, json.JSONDecodeError, KeyError) as exc:
            logger.debug("candidate0: packet_boundary_hash fallback", exc_info=exc)
            packet_boundary_hash = defaults["packet_boundary_hash"]
        model_hash = str(_model_hash_from_identity(model))
    # RNG / stream schema placeholders — canonical JSON hashes of fixed descriptors
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
        # Empty manifest hash (frozen before cases would be set externally)
        case_manifest_hash = "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest()
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


# ---------------------------------------------------------------------------
# core candidate0 act
# ---------------------------------------------------------------------------


def _action_context_from_obs(observation: Any) -> Any:
    from hydra2.contracts.action import ActionContext

    # Build the full context required by the codec. For frozen candidate0 the
    # legal set is already filtered by observation.legal_mask, so the context
    # only needs to make the selected action decodeable.
    # Own concealed tiles: concealed_hand (+ drawn tile) sorted unique
    concealed_hand_raw: Any = observation.concealed_hand
    concealed: tuple[Any, ...] = tuple(concealed_hand_raw)
    own_drawn: Any = observation.own_drawn_tile
    if own_drawn is not None:
        concealed = tuple(sorted({*concealed, own_drawn}))
    # Visible melds: flatten all seats (codec filters by owner)
    flat_melds: list[Any] = []
    visible_melds_raw: Any = observation.visible_melds
    for row in visible_melds_raw:
        row_typed: Any = row
        flat_melds.extend(row_typed)
    # Offered tile / source for claim phases — derive from phase. For
    # draw_decision phases no offer; for discard_response we expose the last
    # discard tile if available. Synthetic tests use draw_decision, so this
    # stays None there. When a discard_response observation carries a pending
    # discard, we surface it.
    offered_tile: Any = None
    offered_by: Any = None
    phase_raw: Any = observation.phase
    if phase_raw in ("discard_response", "kan_response"):
        # Try pending_declaration_discard first, then last visible discard tile
        pending: Any = observation.pending_declaration_discard
        if pending is not None:
            offered_tile = pending
            # Source is the player who discarded — derive from history last discard
            # Fallback to next seat if unknown
            actor_raw: Any = observation.actor
            offered_by = (int(actor_raw) + 3) % 4
            with contextlib.suppress(Exception):
                turn_actor_raw: Any = observation.turn_actor
                offered_by = int(turn_actor_raw)  # type: ignore[arg-type]
        else:
            # Search visible_history for last discard
            visible_history_raw: Any = observation.visible_history
            for ev in reversed(visible_history_raw):
                ev_typed: Any = ev
                payload_obj: Any = getattr(ev_typed, "payload", None)
                kind_from_payload: Any = getattr(payload_obj, "kind", None) if payload_obj is not None else None
                kind_from_ev: Any = getattr(ev_typed, "kind", None)
                kind: Any = kind_from_payload if kind_from_payload is not None and kind_from_payload != "" else kind_from_ev
                if kind == "discard":
                    tile_from_payload: Any = getattr(payload_obj, "tile", None) if payload_obj is not None else None
                    actor_from_payload: Any = getattr(payload_obj, "actor", None) if payload_obj is not None else None
                    if tile_from_payload is not None:
                        offered_tile = tile_from_payload
                        offered_by = (
                            actor_from_payload if actor_from_payload is not None else offered_by
                        )
                    break
    actor_for_ctx: Any = observation.actor
    action_table_hash_raw: Any = observation.action_table_hash
    phase_for_ctx: Any = observation.phase
    return ActionContext(
        phase=phase_for_ctx,
        actor=actor_for_ctx,  # type: ignore[arg-type]
        action_table_hash=action_table_hash_raw,  # type: ignore[arg-type]
        offered_tile=offered_tile,
        offered_by=offered_by,  # type: ignore[arg-type]
        own_concealed_tiles=concealed,
        visible_melds=tuple(flat_melds),
    )


def candidate0(
    request: Any,
    *,
    model: Any,
    encoder: Any | None = None,
    action_table: Any,
    action_codec: Any,
) -> Any:
    """Exact SPEC 16.1 Candidate 0 API — one model evaluation.

    Validates every hash binding before touching the model (ContractError on mismatch),
    does one ``encoder.encode`` + ``model.evaluate``, masked policy, frozen_choice,
    codec decode, and returns a ``SearchResult`` with telemetry.

    No belief, particles, search, pondering, online adaptation, or hidden state.
    """
    from hydra2.eval.telemetry import make_resource_telemetry
    from hydra2.search.common import SearchResult, candidate_spec_hash

    start_ns = time.monotonic_ns()
    spec: Any = request.candidate_spec
    obs: ActorObservation = request.observation

    # ---- hash binding validation (SPEC 15 Contract gate) ----
    obs_rules_hash: Any = obs.rules_hash
    spec_rules_hash: Any = spec.rules_hash
    if str(obs_rules_hash) != str(spec_rules_hash):
        raise ContractError(f"observation rules_hash {obs_rules_hash} != spec {spec_rules_hash}")
    obs_action_table_hash: Any = obs.action_table_hash
    spec_action_table_hash: Any = spec.action_table_hash
    if str(obs_action_table_hash) != str(spec_action_table_hash):
        raise ContractError(
            f"observation action_table_hash {obs_action_table_hash} != spec {spec_action_table_hash}"
        )
    obs_observation_schema_hash: Any = obs.observation_schema_hash
    spec_observation_schema_hash: Any = spec.observation_schema_hash
    if str(obs_observation_schema_hash) != str(spec_observation_schema_hash):
        raise ContractError(
            f"observation observation_schema_hash {obs_observation_schema_hash} != spec {spec_observation_schema_hash}"
        )
    obs_packet_boundary_hash: Any = obs.packet_boundary_hash
    spec_packet_boundary_hash: Any = spec.packet_boundary_hash
    if str(obs_packet_boundary_hash) != str(spec_packet_boundary_hash):
        raise ContractError(
            f"observation packet_boundary_hash {obs_packet_boundary_hash} != spec {spec_packet_boundary_hash}"
        )
    # model identity must match
    model_ident_raw: Any = getattr(model, "model_identity", "")
    model_ident: str = str(model_ident_raw)
    spec_model_hash: Any = spec.model_hash
    if model_ident != "" and model_ident != str(spec_model_hash):
        raise ContractError(f"model identity {model_ident} != spec {spec_model_hash}")
    # utility manifest hash inside model vs spec
    model_util_raw: Any = getattr(model, "utility_manifest_hash", "")
    model_util: str = str(model_util_raw)
    spec_utility_manifest_hash: Any = spec.utility_manifest_hash
    if model_util != "" and model_util != str(spec_utility_manifest_hash):
        raise ContractError(
            f"model utility_manifest_hash {model_util} != spec {spec_utility_manifest_hash}"
        )

    # ---- encode ----
    if encoder is None:
        from hydra2.models.encoder import encode_observations as default_encode

        encode_fn: Any = default_encode
    else:
        # encoder may be a callable or module with encode_observations
        encode_fn = getattr(encoder, "encode_observations", encoder)
    batch: Any = encode_fn([obs])  # type: ignore[operator]  # one row
    # ---- one model evaluation (exactly one) ----
    if not hasattr(model, "evaluate"):
        raise ContractError("model must expose evaluate(batch) -> ModelOutput")
    out: Any = model.evaluate(batch)  # type: ignore[operator]

    # ---- masked policy ----
    from hydra2.models.model import masked_policy

    # legal_mask is tuple[bool] on observation but batch has tensor; use batch tensor
    legal_mask_tensor: Any = batch.legal_mask  # [1, A]
    # Validate requested legal_actions align with mask? For spec we ensure decode will validate.
    policy_logits: Any = out.policy_logits  # [1, A]
    # Use explicit shape check via tensor attributes
    policy_shape_1: Any = policy_logits.shape[1] if hasattr(policy_logits, "shape") else 0
    legal_shape_1: Any = legal_mask_tensor.shape[1] if hasattr(legal_mask_tensor, "shape") else 0
    if policy_shape_1 != legal_shape_1:
        raise ContractError(
            f"policy_logits A {policy_shape_1} != legal_mask A {legal_shape_1}"
        )
    probs_1: Any = masked_policy(policy_logits, legal_mask_tensor)  # [1, A]
    probs: Any = probs_1[0]  # [A]
    value_vec: Any = out.value_vector[0]  # [4]

    # ---- frozen choice ----
    obs_hash_raw: Any = getattr(obs, "observation_hash", "")
    obs_hash_str: str = str(obs_hash_raw if obs_hash_raw is not None and obs_hash_raw != "" else "")
    obs_hash: str = obs_hash_str
    actor_raw: Any = obs.actor
    actor_seat: int = int(actor_raw)
    spec_tie_break: Any = spec.tie_break
    action_id: int = frozen_choice(
        probs, value_vec, str(spec_tie_break), observation_hash=obs_hash, actor_seat=actor_seat
    )

    # ---- decode to CanonicalAction ----
    context: Any = _action_context_from_obs(obs)
    # Codec expects ActionId index aligned to table; validate that mask's true indices match table decode domain
    # Verify that selected action is legal per observation.legal_mask tuple
    legal_mask_obs: Any = obs.legal_mask
    legal_val: Any = legal_mask_obs[action_id]
    if not bool(legal_val):
        raise ContractError(f"selected action {action_id} is illegal per observation.legal_mask")
    try:
        selected: Any = action_codec.decode(action_id, table=action_table, context=context)
    except (AttributeError, ValueError, TypeError, OSError, RuntimeError) as exc:
        raise ContractError(f"codec decode failed for action {action_id}: {exc}") from exc
    # ---- telemetry ----
    elapsed_ns = time.monotonic_ns() - start_ns
    elapsed_ms = elapsed_ns / 1e6
    # Fallback margin: if elapsed would exceed deadline - margin, speculative timeout would fire.
    # Since fallback is self, we never actually fallback to a different policy, but we record fallback_used if we
    # would have exceeded the budget.
    deadline_raw: Any = request.deadline_monotonic_ns
    deadline_ns: int = int(deadline_raw)
    now_ns = time.monotonic_ns()
    # If deadline already passed at entry, mark timeout but still return completed result (fallback is self)
    timeout = now_ns > deadline_ns
    # If model_calls would exceed budget, mark invalid but candidate0's budget is 1 so it never exceeds
    fallback_used = False
    if timeout:
        fallback_used = False  # fallback is self, so not counted as distinct fallback

    # Build telemetry via the eval telemetry contract
    # hardware/environment hashes are derived from the search spec payload so binding is explicit.
    # For deterministic reporting we use stable placeholders hashed from spec hash.
    spec_hash: Any = candidate_spec_hash(spec)
    spec_hash_str: str = str(spec_hash)
    hw_hash: str = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"hardware": "rtx5070", "spec": spec_hash_str})
        ).hexdigest()
    )
    env_hash: str = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"env": "pixi_py312_cuda", "spec": spec_hash_str})
        ).hexdigest()
    )
    resource_budget_mode: Any = spec.resource_budget.mode
    telem: Any = make_resource_telemetry(
        mode=str(resource_budget_mode),
        wall_id=None,
        case_id=None,
        candidate_spec_hash=spec_hash_str,
        hardware_hash=hw_hash,
        environment_hash=env_hash,
        cold_start=False,
        synchronized_elapsed_ms=elapsed_ms,
        model_calls=1,
        exact_transitions=0,
        particles=0,
        fallback_used=bool(fallback_used),
        timeout=timeout,
        illegal_action=False,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
        host_peak_bytes=None,
        energy_joules=None,
        graph_breaks=None,
        recompiles=None,
        invalid_reason=None,
    )

    # Build result
    from hydra2.contracts.utility import UtilityVector

    # Value vectors: wrap model's per-seat value_vector into UtilityVector for spec compliance
    # The model outputs value_vector [B,4] which is already the expected placement vector.
    value_vec_list: Any = value_vec.tolist()
    vec_values: tuple[float, ...] = tuple(float(v) for v in value_vec_list)
    # Validate finite
    for val in vec_values:
        if not isinstance(val, float) or not (val == val and abs(val) != float("inf")):
            raise ContractError(f"value_vector entry {val!r} not finite")
    utility_id_raw: Any = spec.utility_id
    rules_hash_raw: Any = spec.rules_hash
    utility_manifest_hash_raw: Any = spec.utility_manifest_hash
    utility_vec = UtilityVector(
        values=vec_values,  # type: ignore[arg-type]
        utility_id=str(utility_id_raw),
        rules_hash=cast("DigestText", rules_hash_raw),
        utility_manifest_hash=cast("DigestText", utility_manifest_hash_raw),
    )

    result: Any = SearchResult(
        selected_action=selected,
        candidate_actions=(selected,),
        value_vectors=(utility_vec,),
        candidate_spec_hash=spec_hash_str,
        telemetry=telem,
        evidence_refs=(),
        completed=True,
    )
    # Runner validates legal mask; we already validated.
    return result


class FrozenCandidate0:
    """Planner wrapper for frozen Candidate 0 — stateless, deterministic.

    Satisfies ``Planner`` protocol: ``act`` is the only stateful path (none),
    ``observe`` and ``ponder`` are no-ops. History is not retained between calls.
    """

    def __init__(
        self,
        spec: Any,
        model: Any,
        action_table: Any,
        action_codec: Any,
        *,
        encoder: Any | None = None,
    ) -> None:
        spec_candidate_id: Any = spec.candidate_id
        if spec_candidate_id != "candidate0":
            raise ContractError(
                f"FrozenCandidate0 requires candidate0 spec, got {spec_candidate_id!r}"
            )
        spec_fallback: Any = spec.fallback_candidate_id
        if spec_fallback != "candidate0":
            raise ContractError("fallback must be candidate0")
        self._spec: Any = spec
        self._model: Any = model
        self._action_table: Any = action_table
        self._action_codec: Any = action_codec
        self._encoder: Any | None = encoder
        # No hidden state
        self._history: tuple[Any, ...] = ()

    @property
    def spec(self) -> Any:
        return self._spec

    def act(self, request: Any) -> Any:
        # Validate request spec matches owned spec (identity)
        from hydra2.search.common import candidate_spec_hash

        request_spec: Any = request.candidate_spec
        self_spec: Any = self._spec
        req_hash: Any = candidate_spec_hash(request_spec)
        self_hash: Any = candidate_spec_hash(self_spec)
        request_model_hash: Any = getattr(request_spec, "model_hash", None)
        self_model_hash: Any = getattr(self_spec, "model_hash", None)
        if req_hash != self_hash and str(request_model_hash) != str(self_model_hash):
            raise ContractError("request candidate_spec does not match planner spec")
        return candidate0(
            request,
            model=self._model,
            encoder=self._encoder,
            action_table=self._action_table,
            action_codec=self._action_codec,
        )

    def observe(self, packet: Any) -> None:
        # Candidate 0 has no speculative belief state; observe is no-op but validates packet type
        # Packet must be actor-visible; we accept any object with 'visibility' or is ActorVisiblePacket
        if packet is None:
            return
        # Stateless — append to local history for diagnostics only, never influences act()
        self._history = (*self._history, packet)
        # Trim to small bound to avoid unbounded growth (payload small)
        if len(self._history) > 16:
            self._history = self._history[-16:]

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        # No particles/search/pondering — explicitly no-op per BUILD checklist
        return
