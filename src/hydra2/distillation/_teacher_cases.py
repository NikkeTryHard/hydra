"""WP-10 teacher cases — deterministic case observations and teacher priors.

Owns the case RNG, the exact legal-mask derivation from the canonical action
table, the actor-visible case observation builder, and the spec-bound teacher
prior beside its only caller (:func:`_teacher_policy_and_value`). The gate
and registry live in :mod:`hydra2.distillation._teacher_gate`, trajectory
records in :mod:`hydra2.distillation._teacher_records`, the student model in
:mod:`hydra2.distillation._teacher_student`, and five-arm evaluation in
:mod:`hydra2.distillation._teacher_eval`, so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import TYPE_CHECKING, Any

import torch
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import ContractError
from hydra2.distillation._teacher_gate import _require_sha256 as _require_sha256


def _require_census_bridge() -> object:
    """Resolve the census bridge, fail closed when not built (no JSON fallback)."""
    try:
        from hydra2_replay_rs import contracts as bridge  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2 census authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        ) from exc
    if not hasattr(bridge, "action_census") or not hasattr(bridge, "action_census_count"):
        raise ImportError(
            "hydra2_replay_rs.contracts census surface missing (stale .so); "
            "rebuild the bridge (`pixi run build-ext`)"
        )
    return bridge


if TYPE_CHECKING:
    from hydra2.contracts.observation_actor import ActorObservation
    from hydra2.models.model import Hydra2BaselineModel, ModelOutput
    from hydra2.search.common import CandidateSpec

# ---------------------------------------------------------------------------
# Hash utilities (real math) + REAL teacher case path (exact mask, model priors)
# ---------------------------------------------------------------------------


def _hash_to_uniform(key: bytes, index: int) -> float:
    """Map key+index to a top-32-bits uniform in [0, 1] for deterministic draws."""
    b = hashlib.sha256(key + index.to_bytes(4, "big")).digest()
    # 32-bit uniform
    v = int.from_bytes(b[:4], "big") / 0xFFFFFFFF
    return v


def _masked_softmax(logits: tuple[float, ...], mask: tuple[bool, ...]) -> tuple[float, ...]:
    """Masked softmax with exact-zero illegal mass; raises on empty/non-finite support."""
    if len(logits) != len(mask):
        raise ContractError(f"logits len {len(logits)} != mask len {len(mask)}")
    # Zero out illegal by -inf
    mx = max((lv for lv, m in zip(logits, mask, strict=True) if m), default=0.0)
    exps: list[float] = []
    for lv, m in zip(logits, mask, strict=True):
        if not m:
            exps.append(0.0)
        else:
            exps.append(math.exp(lv - mx))
    s = sum(exps)
    if s <= 0 or not math.isfinite(s):
        raise ContractError(f"masked softmax sum non-finite {s}")
    return tuple(e / s for e in exps)


def _provenance_for_case(
    *,
    case_id: str,
    teacher_id: str,
    index: int,
    actor: int,
    seed_material: bytes,
    budget: dict[str, Any],
    justification_digest: str,
) -> dict[str, Any]:
    """Mint case reconstruction provenance (case/actor/seed/budget/justification)."""
    return {
        "teacher_candidate_id": teacher_id,
        "justification_digest": justification_digest,
        "case_id": case_id,
        "trajectory_index": index,
        "actor": actor,
        "seed_material_hex": seed_material.hex(),
        "budget": dict(budget),
        "provenance_version": "1.0.0",
    }


_CASE_RNG_DOMAIN = b"wp10_teacher_case_v1"


def _case_hand_tiles(*, case_id: str, teacher_id: str, seed_material: bytes) -> tuple[int, ...]:
    """Deterministic 13-tile concealed hand for a case (distinct physical tile ids).

    The hand is case material (which tiles the actor holds), not a policy — it
    is drawn from a case-scoped RNG so trajectories are reproducible and vary
    across cases, teachers, and seed materials.
    """
    seed = hashlib.sha256(
        _CASE_RNG_DOMAIN
        + b"|"
        + seed_material
        + b"|"
        + teacher_id.encode()
        + b"|"
        + case_id.encode()
    ).digest()
    rng = random.Random(int.from_bytes(seed, "big"))
    tiles = list(range(136))
    rng.shuffle(tiles)
    return tuple(sorted(tiles[:13]))


def _discard_mask_for_hand(hand: tuple[int, ...]) -> tuple[bool, ...]:
    """Exact legal mask for a discard-phase case: discards of held tiles only.

    Derived from the canonical action census via the bridge (kind == "discard",
    matching tile), never a hash coin-flip. Length equals the full census;
    every other action is illegal in this fixture phase. Bit-identical to the
    retired JSON-table path (verified: bridge order == payload.actions order).
    """
    bridge = _require_census_bridge()
    records = bridge.action_census()  # type: ignore[attr-defined]
    index_by_tile: dict[int, int] = {}
    for idx, entry in enumerate(records):
        if entry.kind == "discard" and entry.tile is not None:
            tile_id: int = int(entry.tile)
            if tile_id not in index_by_tile:
                index_by_tile[tile_id] = idx
    mask = [False] * len(records)
    for tile in hand:
        idx = index_by_tile.get(tile)
        if idx is None:
            raise ContractError(f"WP-10 blocked: no canonical discard action for held tile {tile}")
        mask[idx] = True
    if not any(mask):
        raise ContractError("WP-10 blocked: exact legal mask empty")
    return tuple(mask)


def _case_observation(
    *, case_id: str, teacher_id: str, actor: int, spec: CandidateSpec, seed_material: bytes
) -> ActorObservation:
    """Build the REAL actor-visible observation for a distillation case.

    Concealed hand from the case RNG, exact discard mask from the canonical
    action table, contract hashes bound from the teacher CandidateSpec, and the
    observation_hash computed over the identity document (SPEC 8). No hash
    coin-flips, no fabricated digests.
    """
    if actor not in (0, 1, 2, 3):
        raise ContractError(f"actor must be 0..3, got {actor}")
    from hydra2.contracts.event_schema import (
        build_event_schema_payload,
        compute_event_schema_digest,
    )
    from hydra2.contracts.observation_actor import (
        make_actor_observation,
    )
    from hydra2.contracts.observation_schema import (
        observation_schema_digest,
    )
    from hydra2.contracts.observation_types import (
        DORA_SENTINEL,
    )

    hand = _case_hand_tiles(case_id=case_id, teacher_id=teacher_id, seed_material=seed_material)
    legal_mask = _discard_mask_for_hand(hand)
    wall_byte = hashlib.sha256(
        _CASE_RNG_DOMAIN + b"|wall|" + teacher_id.encode() + b"|" + case_id.encode()
    ).digest()[0]
    live_remaining = 70 - (wall_byte % 40)
    try:
        obs = make_actor_observation(
            game_id=f"wp10:{teacher_id}",
            decision_id=case_id,
            sequence=0,
            actor=actor,
            rules_id="tenhou_4p_hanchan_v1",
            rules_hash=_bridge_contracts.make_digest_text(spec.rules_hash),
            action_table_hash=_bridge_contracts.make_digest_text(spec.action_table_hash),
            event_schema_hash=compute_event_schema_digest(build_event_schema_payload()),
            observation_schema_hash=observation_schema_digest(),
            packet_boundary_hash=_bridge_contracts.make_digest_text(spec.packet_boundary_hash),
            round_index=0,
            round_wind=27,
            hand_number=0,
            seat_winds=(27, 28, 29, 30),
            honba=0,
            riichi_sticks=0,
            dealer=0,
            scores=(25000, 25000, 25000, 25000),
            turn_actor=actor,
            phase="discard_response",
            live_wall_tiles_remaining=live_remaining,
            kan_count=0,
            ippatsu_active=(False, False, False, False),
            actor_furiten="none",
            actor_can_tsumo=True,
            actor_can_riichi=False,
            pending_declaration_discard=None,
            concealed_hand=hand,
            own_drawn_tile=None,
            visible_discards=((), (), (), ()),
            visible_melds=((), (), (), ()),
            riichi_states=("none", "none", "none", "none"),
            dora_indicators=(
                DORA_SENTINEL,
                DORA_SENTINEL,
                DORA_SENTINEL,
                DORA_SENTINEL,
                DORA_SENTINEL,
            ),
            visible_history=(),
            legal_mask=legal_mask,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"WP-10 blocked: case observation invalid: {exc}") from exc
    return obs


_TEACHER_PRIOR_CACHE: dict[str, Hydra2BaselineModel] = {}


def _teacher_prior_model(spec: CandidateSpec) -> Hydra2BaselineModel:
    """Real teacher prior: baseline transformer bound to the CandidateSpec.

    Weights are deterministically seeded from the spec's model_hash, so the
    same teacher always yields the same priors and distinct teachers yield
    distinct policies. This is the shared model-prior path the candidates'
    search policies build on (SPEC 16.2-16.7 leaf priors).
    """
    from hydra2.models.model import Hydra2BaselineModel

    model_hash = spec.model_hash
    _ = _require_sha256("spec.model_hash", model_hash)
    cached = _TEACHER_PRIOR_CACHE.get(model_hash)
    if cached is not None:
        return cached
    seed = int(hashlib.sha256(b"wp10_teacher_prior_v1|" + model_hash.encode()).hexdigest()[:16], 16)
    gen_state = torch.random.get_rng_state()
    try:
        _ = torch.manual_seed(seed)
        model = Hydra2BaselineModel()
        _ = model.eval()
    finally:
        torch.random.set_rng_state(gen_state)
    _TEACHER_PRIOR_CACHE[model_hash] = model
    return model


def _teacher_policy_and_value(
    *, observation: ActorObservation, spec: CandidateSpec
) -> tuple[tuple[float, ...], tuple[float, float, float, float]]:
    """REAL teacher policy + four-seat return for a case observation.

    Encodes the actor-visible observation via the model_input_v1 encoder,
    runs the spec-bound teacher prior, and masks to the EXACT legal mask.
    Raises ContractError when the teacher path is unavailable (WP-10 blocked).
    """
    from hydra2.models.encoder import encode_observations

    census_len = int(_require_census_bridge().action_census_count())  # type: ignore[attr-defined]
    if len(observation.legal_mask) != census_len:
        raise ContractError(
            "WP-10 blocked: observation legal mask does not match canonical action table"
        )
    model = _teacher_prior_model(spec)
    try:
        batch = encode_observations([observation])
        with torch.no_grad():
            out: ModelOutput = model.evaluate(batch)
            logits: list[float] = out.policy_logits[0].tolist()
            values: list[float] = out.value_vector[0].tolist()
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"WP-10 blocked: teacher prior failed: {exc}") from exc
    mask = tuple(observation.legal_mask)
    policy = _masked_softmax(tuple(logits), mask)
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ContractError("WP-10 blocked: teacher value vector invalid")
    vector = (values[0], values[1], values[2], values[3])
    return policy, vector
