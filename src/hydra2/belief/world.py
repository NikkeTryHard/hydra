"""WP-07A natural belief — FullWorld and helpers.

Privileged FullWorld: NEVER serialized into actor model/search keys/logs.
Search sandbox may hold worlds behind opaque string refs (world_id).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    TileId,
    make_digest_text,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.observation import make_actor_observation

__all__ = [
    "FullWorld",
    "compute_world_id",
    "make_full_world",
    "world_actor_observation",
]


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
        # world_id validated if present
        if self.world_id is not None:
            object.__setattr__(self, "world_id", make_digest_text(self.world_id))
            expected = compute_world_id(self)
            if self.world_id != expected:
                raise DigestMismatchError(
                    f"world_id mismatch: recorded {self.world_id} != recomputed {expected}"
                    " [PBRF_DIGEST_WORLD_ID]"
                )
        # concealed_hands must be 4 seats
        if not isinstance(self.concealed_hands, tuple) or len(self.concealed_hands) != 4:
            raise ContractError("concealed_hands must be 4-tuple of tile tuples")
        hands = []
        for seat, hand in enumerate(self.concealed_hands):
            if not isinstance(hand, tuple):
                raise ContractError(f"concealed_hands[{seat}] must be tuple")
            t = tuple(make_tile_id(v) for v in hand)
            if list(t) != sorted(t):
                raise ContractError(f"concealed_hands[{seat}] must be sorted ascending")
            hands.append(t)
        object.__setattr__(self, "concealed_hands", tuple(hands))
        # live/dead wall
        object.__setattr__(self, "live_wall", tuple(make_tile_id(v) for v in self.live_wall))
        object.__setattr__(self, "dead_wall", tuple(make_tile_id(v) for v in self.dead_wall))
        # latent_state must be mapping with json-serializable values (checked lightly)
        if not isinstance(self.latent_state, Mapping):
            raise ContractError("latent_state must be mapping")
        # hashes
        object.__setattr__(self, "rules_hash", make_digest_text(self.rules_hash))
        object.__setattr__(self, "observation_hash", make_digest_text(self.observation_hash))
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
    identity = canonical_bytes(world_identity_document(world))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


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
    ch = tuple(tuple(t for t in hand) for hand in concealed_hands)
    lw = tuple(t for t in live_wall)
    dw = tuple(t for t in dead_wall)
    latent = dict(latent_state) if latent_state is not None else {}
    snap = simulator_snapshot if simulator_snapshot is not None else f"snap:{ch}:{lw}"
    # Compute world_id via canonical bytes of identity document (without world_id)
    # Identity document mirrors FullWorld.to_json() without world_id
    identity_doc = {
        "simulator_snapshot": snap,
        "concealed_hands": [list(hand) for hand in ch],
        "live_wall": list(lw),
        "dead_wall": list(dw),
        "latent_state": latent,
        "rules_hash": rules_hash,
        "observation_hash": observation_hash,
    }
    world_id = DigestText("sha256:" + hashlib.sha256(canonical_bytes(identity_doc)).hexdigest())
    return FullWorld(
        world_id=world_id,
        simulator_snapshot=snap,
        concealed_hands=ch,  # type: ignore[arg-type]
        live_wall=lw,  # type: ignore[arg-type]
        dead_wall=dw,  # type: ignore[arg-type]
        latent_state=latent,
        rules_hash=rules_hash,  # type: ignore[arg-type]
        observation_hash=observation_hash,  # type: ignore[arg-type]
    )


def world_actor_observation(
    world: FullWorld,
    actor: int,
    *,
    game_id: str = "game_tiny_001",
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
    from hydra2.contracts.observation import DORA_SENTINEL

    a = int(make_seat(actor))
    hand = world.concealed_hands[a]
    did = (
        decision_id
        if decision_id is not None
        else f"dec_hand_{'_'.join(str(int(t)) for t in hand)}_{a}"
    )
    # Dummy hashes default to world hashes
    ath = action_table_hash if action_table_hash is not None else "sha256:" + "b" * 64
    esh = event_schema_hash if event_schema_hash is not None else "sha256:" + "c" * 64
    osh = observation_schema_hash if observation_schema_hash is not None else "sha256:" + "d" * 64
    pbh = packet_boundary_hash if packet_boundary_hash is not None else "sha256:" + "e" * 64

    obs = make_actor_observation(
        game_id=game_id,
        decision_id=did,
        sequence=sequence,
        actor=a,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash=world.rules_hash,
        action_table_hash=ath,
        event_schema_hash=esh,
        observation_schema_hash=osh,
        packet_boundary_hash=pbh,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=a,
        phase="discard_response",
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
        legal_mask=(True, False, True),
    )
    return obs
