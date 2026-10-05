"""Shared WP-05A model/inference observation builders (test-only)."""

from __future__ import annotations

from typing import Any

import torch

from hydra2.contracts.event_envelope import (
    EventEnvelope,
    EventPayload,
)
from hydra2.contracts.observation_actor import make_actor_observation
from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.schema import BASELINE_ACTION_COUNT


def _make_turn_advance(sequence: int, actor: int = 0) -> EventEnvelope:
    payload = EventPayload(
        kind="turn_advance",
        actor=actor,
        tile=None,
        action_id=None,
        source_seat=None,
        consumed_tiles=(),
        offered_action_ids=(),
        accepted_action_ids=(),
        round_index=None,
        scores=None,
        reason=None,
    )
    return EventEnvelope(
        game_id="g-wp05a",
        sequence=sequence,
        kind="turn_advance",
        actor=actor,
        visibility="public",
        visible_to=(0, 1, 2, 3),
        payload=payload,
        public_delta=(),
        rules_hash="sha256:" + "ab" * 32,
        schema_hash="sha256:" + "ac" * 32,
    )


def _make_observation(
    *,
    actor: int = 0,
    history: tuple[EventEnvelope, ...] = (),
    legal_mask: tuple[bool, ...] | None = None,
    concealed: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
    dora: tuple[int, ...] = (-1, -1, -1, -1, -1),
) -> Any:
    if legal_mask is None:
        mask = [False] * BASELINE_ACTION_COUNT
        mask[0] = True
        mask[10] = True
        legal_mask = tuple(mask)
    seq = int(history[-1].sequence) if history else 1
    return make_actor_observation(
        game_id="g-wp05a",
        decision_id=f"d-{actor}-{seq}",
        sequence=seq,
        actor=actor,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "ab" * 32,
        action_table_hash="sha256:" + "ac" * 32,
        event_schema_hash="sha256:" + "ad" * 32,
        observation_schema_hash="sha256:" + "ae" * 32,
        packet_boundary_hash="sha256:" + "af" * 32,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=concealed,
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=dora,
        visible_history=tuple(history),
        legal_mask=legal_mask,
    )


def _history_of_length(n: int) -> tuple[EventEnvelope, ...]:
    return tuple(_make_turn_advance(i + 1, actor=i % 4) for i in range(n))


def _batch_to_device(batch: ActorTensorBatch, device: torch.device) -> ActorTensorBatch:
    """Move an ActorTensorBatch to device (test-only; mirrors loop._move_batch_to_device)."""
    return ActorTensorBatch(
        features={
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in dict(batch.features).items()
        },
        history_mask=batch.history_mask.to(device),
        legal_mask=batch.legal_mask.to(device),
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats.to(device),
    )
