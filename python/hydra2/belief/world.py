"""WP-07A natural belief — FullWorld and helpers.

Privileged FullWorld: NEVER serialized into actor model/search keys/logs.
Search sandbox may hold worlds behind opaque string refs (world_id).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    TileId,
)
from hydra2.contracts.observation_actor import make_actor_observation

__all__ = [
    "FullWorld",
    "compute_world_id",
    "make_full_world",
    "world_actor_observation",
]

#: Frozen tiny-observation literals (single source: ``hydra2._native.contracts``;
#: this module keeps the builder + record roots, ``__all__`` unchanged).
_DEFAULT_GAME_ID: str = _bridge_contracts.WORLD_DEFAULT_GAME_ID
_DEFAULT_PHASE: str = _bridge_contracts.WORLD_DEFAULT_PHASE
_DEFAULT_ROUND_WIND: int = _bridge_contracts.WORLD_DEFAULT_ROUND_WIND
_SEAT_WINDS: tuple[int, int, int, int] = _bridge_contracts.WORLD_SEAT_WINDS
_DEFAULT_SCORES: tuple[int, int, int, int] = _bridge_contracts.WORLD_DEFAULT_SCORES
_TINY_LEGAL_MASK: tuple[bool, bool, bool] = _bridge_contracts.WORLD_TINY_LEGAL_MASK


def _world_id_from_identity_doc(identity_doc: dict[str, Any]) -> DigestText:
    """Hash canonical identity bytes via the feed digest owner (never rehash here)."""
    hex_text: str = _bridge_contracts.world_id_from_doc(canonical_bytes(identity_doc))
    return DigestText(hex_text)


@dataclass(frozen=True, slots=True)
class FullWorld:
    """Privileged complete tile-conserving hidden completion (SPEC 14.1)."""

    world_id: DigestText
    simulator_snapshot: str
    concealed_hands: tuple[tuple[TileId, ...], ...]
    live_wall: tuple[TileId, ...]
    dead_wall: tuple[TileId, ...]
    latent_state: Mapping[str, Any]
    rules_hash: DigestText
    observation_hash: DigestText

    def __post_init__(self) -> None:
        # Re-verify stored id: stale/forged worlds fail closed with digest mismatch.
        if self.world_id is not None:
            validated_id: DigestText = _bridge_contracts.make_digest_text(self.world_id)
            object.__setattr__(self, "world_id", validated_id)
            expected = compute_world_id(self)
            if self.world_id != expected:
                raise DigestMismatchError(
                    f"world_id mismatch: recorded {self.world_id} != recomputed {expected}"
                    " [PBRF_DIGEST_WORLD_ID]"
                )
        # Require exactly 4 seats; fewer would silently drop hidden tiles.
        if not isinstance(self.concealed_hands, tuple) or len(self.concealed_hands) != 4:
            raise ContractError("concealed_hands must be 4-tuple of tile tuples")
        normalized_hands = []
        for seat, hand in enumerate(self.concealed_hands):
            if not isinstance(hand, tuple):
                raise ContractError(f"concealed_hands[{seat}] must be tuple")
            normalized_hand = tuple(_bridge_contracts.make_tile_id(v) for v in hand)
            if list(normalized_hand) != sorted(normalized_hand):
                raise ContractError(f"concealed_hands[{seat}] must be sorted ascending")
            normalized_hands.append(normalized_hand)
        object.__setattr__(self, "concealed_hands", tuple(normalized_hands))
        # live/dead wall
        object.__setattr__(
            self, "live_wall", tuple(_bridge_contracts.make_tile_id(v) for v in self.live_wall)
        )
        object.__setattr__(
            self, "dead_wall", tuple(_bridge_contracts.make_tile_id(v) for v in self.dead_wall)
        )
        # latent_state must be mapping with json-serializable values (checked lightly)
        if not isinstance(self.latent_state, Mapping):
            raise ContractError("latent_state must be mapping")
        # hashes
        validated_rules: DigestText = _bridge_contracts.make_digest_text(self.rules_hash)
        object.__setattr__(self, "rules_hash", validated_rules)
        validated_obs: DigestText = _bridge_contracts.make_digest_text(self.observation_hash)
        object.__setattr__(self, "observation_hash", validated_obs)
        # simulator_snapshot is opaque string
        if not isinstance(self.simulator_snapshot, str):
            raise ContractError("simulator_snapshot must be str")

    def to_json(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "simulator_snapshot": self.simulator_snapshot,
            "concealed_hands": [[int(t) for t in hand] for hand in self.concealed_hands],
            "live_wall": [int(t) for t in self.live_wall],
            "dead_wall": [int(t) for t in self.dead_wall],
            "latent_state": dict(self.latent_state),
            "rules_hash": self.rules_hash,
            "observation_hash": self.observation_hash,
        }


def world_identity_document(world: FullWorld) -> dict[str, Any]:
    doc = world.to_json()
    doc.pop("world_id", None)
    return doc


def compute_world_id(world: FullWorld) -> DigestText:
    return _world_id_from_identity_doc(world_identity_document(world))


def make_full_world(
    *,
    concealed_hands: tuple[tuple[int, ...], ...],
    live_wall: tuple[int, ...],
    dead_wall: tuple[int, ...] = (),
    latent_state: Mapping[str, Any] | None = None,
    rules_hash: str,
    observation_hash: str,
    simulator_snapshot: str | None = None,
) -> FullWorld:
    """Construct FullWorld with world_id bound to identity bytes."""
    concealed = tuple(tuple(t for t in hand) for hand in concealed_hands)
    live = tuple(t for t in live_wall)
    dead = tuple(t for t in dead_wall)
    latent_copy = dict(latent_state) if latent_state is not None else {}
    snapshot = simulator_snapshot if simulator_snapshot is not None else f"snap:{concealed}:{live}"
    # Invariant: world_id binds identity bytes; mismatch fails closed in __post_init__.
    identity_doc = {
        "simulator_snapshot": snapshot,
        "concealed_hands": [list(hand) for hand in concealed],
        "live_wall": list(live),
        "dead_wall": list(dead),
        "latent_state": latent_copy,
        "rules_hash": rules_hash,
        "observation_hash": observation_hash,
    }
    world_id = _world_id_from_identity_doc(identity_doc)
    return FullWorld(
        world_id=world_id,
        simulator_snapshot=snapshot,
        concealed_hands=concealed,  # type: ignore[arg-type]
        live_wall=live,  # type: ignore[arg-type]
        dead_wall=dead,  # type: ignore[arg-type]
        latent_state=latent_copy,
        rules_hash=rules_hash,  # type: ignore[arg-type]
        observation_hash=observation_hash,  # type: ignore[arg-type]
    )


def world_actor_observation(
    world: FullWorld,
    actor: int,
    *,
    game_id: str = _DEFAULT_GAME_ID,
    decision_id: str | None = None,
    sequence: int = 0,
    action_table_hash: str | None = None,
    event_schema_hash: str | None = None,
    observation_schema_hash: str | None = None,
    packet_boundary_hash: str | None = None,
) -> Any:
    """Derive ActorObservation for ``actor`` from FullWorld.

    Only actor's concealed hand and public histories are exposed; hidden tiles
    of other seats remain unobservable, giving hidden-permutation invariance.
    """
    from hydra2.contracts.observation_types import DORA_SENTINEL

    seat_idx: int = _bridge_contracts.make_seat(actor)
    hand = world.concealed_hands[seat_idx]
    # Derive a stable id when the caller supplies none: binds hand + seat.
    decision_text = (
        decision_id
        if decision_id is not None
        else f"dec_hand_{'_'.join(str(int(t)) for t in hand)}_{seat_idx}"
    )
    # Dummy hashes default to fixed sentinels; never derive from hidden tiles.
    action_hash = action_table_hash if action_table_hash is not None else "sha256:" + "b" * 64
    event_hash = event_schema_hash if event_schema_hash is not None else "sha256:" + "c" * 64
    obs_hash = (
        observation_schema_hash if observation_schema_hash is not None else "sha256:" + "d" * 64
    )
    boundary_hash = (
        packet_boundary_hash if packet_boundary_hash is not None else "sha256:" + "e" * 64
    )
    return make_actor_observation(
        game_id=game_id,
        decision_id=decision_text,
        sequence=sequence,
        actor=seat_idx,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash=world.rules_hash,
        action_table_hash=action_hash,
        event_schema_hash=event_hash,
        observation_schema_hash=obs_hash,
        packet_boundary_hash=boundary_hash,
        round_index=0,
        round_wind=_DEFAULT_ROUND_WIND,
        hand_number=0,
        seat_winds=_SEAT_WINDS,
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=_DEFAULT_SCORES,
        turn_actor=seat_idx,
        phase=_DEFAULT_PHASE,
        live_wall_tiles_remaining=len(world.live_wall),
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
        dora_indicators=(DORA_SENTINEL, DORA_SENTINEL, DORA_SENTINEL, DORA_SENTINEL, DORA_SENTINEL),
        visible_history=(),
        legal_mask=_TINY_LEGAL_MASK,
    )
