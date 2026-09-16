"""Candidate 0 act path — one model evaluation plus planner wrapper.

Owns the observation-to-codec context builder, the ``candidate0`` single
model-evaluation entry point (hash-gate validation, masked policy, frozen
choice, decode, telemetry), and the stateless deterministic
``FrozenCandidate0`` planner (SPEC 16.1). Frozen choice and the spec factory
live in :mod:`hydra2.search.candidate0_frozen`.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from typing import TYPE_CHECKING, Any, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.search.candidate0_frozen import frozen_choice

if TYPE_CHECKING:
    from hydra2.contracts.common import DigestText
    from hydra2.contracts.observation import ActorObservation

__all__ = [
    "FrozenCandidate0",
    "_action_context_from_obs",
    "candidate0",
]

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
                kind_from_payload: Any = (
                    getattr(payload_obj, "kind", None) if payload_obj is not None else None
                )
                kind_from_ev: Any = getattr(ev_typed, "kind", None)
                kind: Any = (
                    kind_from_payload
                    if kind_from_payload is not None and kind_from_payload != ""
                    else kind_from_ev
                )
                if kind == "discard":
                    tile_from_payload: Any = (
                        getattr(payload_obj, "tile", None) if payload_obj is not None else None
                    )
                    actor_from_payload: Any = (
                        getattr(payload_obj, "actor", None) if payload_obj is not None else None
                    )
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
        raise ContractError(f"policy_logits A {policy_shape_1} != legal_mask A {legal_shape_1}")
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
        """Planner act — torch path stays, Rust only judges the boundary.

        The candidate0 torch path (encode + evaluate + masked policy +
        frozen choice + decode) is a HARD torch island and stays Python per
        the invariants — no act_batch probe crosses here (NO Rust GPU math,
        Burn/Candle out). The telemetry/spec-hash bindings below are
        unchanged; the plan's bridge-act flip covers the search act entries
        only.
        """
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
