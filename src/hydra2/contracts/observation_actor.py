"""SPEC 8 actor observation — dataclass, identity hash, and factory.

Owns the exact SPEC 8 field list in declaration order: the frozen dataclass
firewall, the identity document with its hash binding, and the closed
factory that stamps the digest. Schema tables and the versioned artifact
live in the build module; the validator shell lives there too so both
directions of the layer stay acyclic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    Seat,
    SequenceNo,
    TileId,
    TileType,
    VisibilityViolationError,
    make_digest_text,
    make_seat,
    make_sequence_no,
    make_tile_id,
    make_tile_type,
)
from hydra2.contracts.event import EventEnvelope, visible_to_actor
from hydra2.contracts.observation_types import (
    _FURIETEN_STATES,
    _RIICHI_STATES,
    _WIND_TILE_TYPES,
    DORA_SENTINEL,
    DORA_SHAPE,
    PHASES,
    Phase,
    VisibleMeld,
    _quad,
    _require_bool,
    _require_enum,
    _require_plain_int,
    _require_str,
    _tile_tuple,
    _validate_score,
    _validate_wind_type,
)

__all__ = [
    "_OBSERVATION_FIELDS",
    "ActorObservation",
    "compute_observation_hash",
    "make_actor_observation",
    "observation_identity_document",
]

# ---------------------------------------------------------------------------
# ActorObservation - exact SPEC 8 fields, in declaration order.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActorObservation:
    """One actor's complete legal view at one decision point (SPEC 8).

    The closed slot set IS the visibility boundary: wall (136-tile stack) /
    dead wall (14-tile reserve), opponent concealed tiles, unrevealed dora/ura,
    engine RNG, future events, server-private events, opponent legal masks,
    and privileged labels have no field to occupy. ``observation_hash`` binds
    the identity document.
    """

    game_id: str
    decision_id: str
    sequence: SequenceNo
    actor: Seat
    rules_id: str
    rules_hash: DigestText
    action_table_hash: DigestText
    event_schema_hash: DigestText
    observation_schema_hash: DigestText
    packet_boundary_hash: DigestText
    round_index: int
    round_wind: TileType
    hand_number: int
    seat_winds: tuple[TileType, TileType, TileType, TileType]
    honba: int
    riichi_sticks: int
    dealer: Seat
    scores: tuple[int, int, int, int]
    turn_actor: Seat
    phase: Phase
    live_wall_tiles_remaining: int
    kan_count: int
    ippatsu_active: tuple[bool, bool, bool, bool]
    actor_furiten: Literal["none", "temporary", "riichi", "discard"]
    actor_can_tsumo: bool
    actor_can_riichi: bool
    pending_declaration_discard: TileId | None
    concealed_hand: tuple[TileId, ...]
    own_drawn_tile: TileId | None
    visible_discards: tuple[tuple[TileId, ...], ...]  # four seats
    visible_melds: tuple[tuple[VisibleMeld, ...], ...]  # four seats
    riichi_states: tuple[str, str, str, str]
    dora_indicators: tuple[int, int, int, int, int]
    visible_history: tuple[EventEnvelope, ...]
    legal_mask: tuple[bool, ...]
    observation_hash: DigestText | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_id", _require_str(self.game_id, name="game_id"))
        if self.game_id == "":
            raise ContractError("game_id must be non-empty")
        object.__setattr__(self, "decision_id", _require_str(self.decision_id, name="decision_id"))
        if self.decision_id == "":
            raise ContractError("decision_id must be non-empty")
        object.__setattr__(self, "sequence", make_sequence_no(self.sequence))
        object.__setattr__(self, "actor", make_seat(self.actor))
        object.__setattr__(self, "rules_id", _require_str(self.rules_id, name="rules_id"))
        if self.rules_id == "":
            raise ContractError("rules_id must be non-empty")
        for name in (
            "rules_hash",
            "action_table_hash",
            "event_schema_hash",
            "observation_schema_hash",
            "packet_boundary_hash",
        ):
            object.__setattr__(self, name, make_digest_text(getattr(self, name)))
        object.__setattr__(
            self,
            "round_index",
            _require_plain_int(self.round_index, name="round_index", minimum=0, maximum=None),
        )
        object.__setattr__(self, "round_wind", make_tile_type(self.round_wind))
        object.__setattr__(
            self,
            "hand_number",
            _require_plain_int(self.hand_number, name="hand_number", minimum=0, maximum=None),
        )
        winds = _quad(
            self.seat_winds,
            name="seat_winds",
            validator=_validate_wind_type,
        )
        if sorted(int(w) for w in winds) != list(_WIND_TILE_TYPES):  # pyrefly: ignore[unknown-argument-type]  # reason: winds quad-validated above; int() coerces each entry
            raise ContractError("seat_winds must permute East/South/West/North aligned by seat")
        object.__setattr__(self, "seat_winds", winds)
        object.__setattr__(
            self, "honba", _require_plain_int(self.honba, name="honba", minimum=0, maximum=None)
        )
        object.__setattr__(
            self,
            "riichi_sticks",
            _require_plain_int(self.riichi_sticks, name="riichi_sticks", minimum=0, maximum=None),
        )
        object.__setattr__(self, "dealer", make_seat(self.dealer))
        object.__setattr__(
            self,
            "scores",
            _quad(
                self.scores,
                name="scores",
                validator=_validate_score,
            ),
        )
        object.__setattr__(self, "turn_actor", make_seat(self.turn_actor))
        object.__setattr__(self, "phase", _require_enum(self.phase, name="phase", allowed=PHASES))
        object.__setattr__(
            self,
            "live_wall_tiles_remaining",
            _require_plain_int(
                self.live_wall_tiles_remaining,
                name="live_wall_tiles_remaining",
                minimum=0,
                maximum=None,
            ),
        )
        object.__setattr__(
            self,
            "kan_count",
            _require_plain_int(self.kan_count, name="kan_count", minimum=0, maximum=4),
        )
        object.__setattr__(
            self,
            "ippatsu_active",
            _quad(self.ippatsu_active, name="ippatsu_active", validator=_require_bool),
        )
        object.__setattr__(
            self,
            "actor_furiten",
            _require_enum(self.actor_furiten, name="actor_furiten", allowed=_FURIETEN_STATES),
        )
        object.__setattr__(
            self, "actor_can_tsumo", _require_bool(self.actor_can_tsumo, name="actor_can_tsumo")
        )
        object.__setattr__(
            self, "actor_can_riichi", _require_bool(self.actor_can_riichi, name="actor_can_riichi")
        )
        if self.pending_declaration_discard is not None:
            object.__setattr__(
                self,
                "pending_declaration_discard",
                make_tile_id(self.pending_declaration_discard),
            )
        hand = _tile_tuple(self.concealed_hand, name="concealed_hand")
        if list(hand) != sorted(hand):
            raise ContractError(
                "concealed_hand must be ascending by physical TileId "
                "(duplicates allowed; the drawn tile stays separate)"
            )
        object.__setattr__(self, "concealed_hand", hand)
        if self.own_drawn_tile is not None:
            object.__setattr__(self, "own_drawn_tile", make_tile_id(self.own_drawn_tile))
        if (
            isinstance(self.visible_discards, (str, bytes))
            or not isinstance(self.visible_discards, Sequence)
            or len(self.visible_discards) != 4
        ):
            raise ContractError("visible_discards must hold exactly four seat rivers")
        rivers = tuple(
            _tile_tuple(river, name=f"visible_discards[{seat}]")
            for seat, river in enumerate(self.visible_discards)
        )
        object.__setattr__(self, "visible_discards", rivers)
        if (
            isinstance(self.visible_melds, (str, bytes))
            or not isinstance(self.visible_melds, Sequence)
            or len(self.visible_melds) != 4
        ):
            raise ContractError("visible_melds must hold exactly four seat meld rows")
        meld_rows = []
        for seat, row in enumerate(self.visible_melds):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise ContractError(f"visible_melds[{seat}] must be a sequence of VisibleMeld")
            for meld in row:
                if not isinstance(meld, VisibleMeld):
                    raise ContractError(
                        f"visible_melds[{seat}] entries must be VisibleMeld instances"
                    )
            meld_rows.append(tuple(row))
        object.__setattr__(self, "visible_melds", tuple(meld_rows))
        object.__setattr__(
            self,
            "riichi_states",
            _quad(
                self.riichi_states,
                name="riichi_states",
                validator=lambda v, name: _require_enum(v, name=name, allowed=_RIICHI_STATES),  # pyrefly: ignore[unknown-argument-type]  # reason: snapshot statically object; _require_enum validates the value
            ),
        )
        indicators = self.dora_indicators
        if (
            isinstance(indicators, (str, bytes))
            or not isinstance(indicators, Sequence)
            or len(indicators) != DORA_SHAPE[0]
        ):
            raise ContractError(
                f"dora_indicators must hold exactly {DORA_SHAPE[0]} entries, got "
                f"{len(indicators) if isinstance(indicators, Sequence) else 'non-sequence'}; "
                "the shape is fixed and NEVER padded"
            )
        checked: list[int] = []
        for index, value in enumerate(indicators):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"dora_indicators[{index}] must be an int")
            if value != DORA_SENTINEL:
                _ = make_tile_id(value)
            checked.append(value)
        revealed = [v for v in checked if v != DORA_SENTINEL]
        if checked[: len(revealed)] != revealed or DORA_SENTINEL in revealed:
            raise ContractError(
                "revealed dora indicators must be contiguous from index 0; "
                "sentinels fill the unrevealed tail"
            )
        object.__setattr__(self, "dora_indicators", tuple(checked))
        if isinstance(self.visible_history, (str, bytes)) or not isinstance(
            self.visible_history, Sequence
        ):
            raise ContractError("visible_history must be a sequence of EventEnvelope")
        for event in self.visible_history:
            if not isinstance(event, EventEnvelope):
                raise ContractError("visible_history entries must be EventEnvelope instances")
            if not visible_to_actor(event, self.actor):
                raise VisibilityViolationError(
                    f"history holds a {event.visibility} event this actor may not see"
                )
        mask = self.legal_mask
        if isinstance(mask, (str, bytes)) or not isinstance(mask, Sequence) or len(mask) == 0:
            raise ContractError("legal_mask must be a non-empty sequence of booleans")
        for index, flag in enumerate(mask):
            if not isinstance(flag, bool):
                raise ContractError(f"legal_mask[{index}] must be a bool")
            if not flag:
                continue
            break
        else:
            raise ContractError("legal_mask must contain at least one True at a decision")
        object.__setattr__(self, "legal_mask", tuple(mask))
        if self.observation_hash is not None:
            object.__setattr__(self, "observation_hash", make_digest_text(self.observation_hash))
            recomputed = compute_observation_hash(self)
            if self.observation_hash != recomputed:
                raise DigestMismatchError(
                    f"observation_hash mismatch: recorded {self.observation_hash} != "
                    f"recomputed {recomputed}"
                )

    def to_json(self) -> dict[str, object]:
        """Deterministic SPEC-order document; the concealed hand serializes sorted."""
        document: dict[str, object] = {
            "game_id": self.game_id,
            "decision_id": self.decision_id,
            "sequence": int(self.sequence),
            "actor": int(self.actor),
            "rules_id": self.rules_id,
            "rules_hash": self.rules_hash,
            "action_table_hash": self.action_table_hash,
            "event_schema_hash": self.event_schema_hash,
            "observation_schema_hash": self.observation_schema_hash,
            "packet_boundary_hash": self.packet_boundary_hash,
            "round_index": self.round_index,
            "round_wind": int(self.round_wind),
            "hand_number": self.hand_number,
            "seat_winds": [int(w) for w in self.seat_winds],
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "dealer": int(self.dealer),
            "scores": list(self.scores),
            "turn_actor": int(self.turn_actor),
            "phase": self.phase,
            "live_wall_tiles_remaining": self.live_wall_tiles_remaining,
            "kan_count": self.kan_count,
            "ippatsu_active": list(self.ippatsu_active),
            "actor_furiten": self.actor_furiten,
            "actor_can_tsumo": self.actor_can_tsumo,
            "actor_can_riichi": self.actor_can_riichi,
            "pending_declaration_discard": (
                None
                if self.pending_declaration_discard is None
                else int(self.pending_declaration_discard)
            ),
            "concealed_hand": sorted(int(t) for t in self.concealed_hand),
            "own_drawn_tile": (None if self.own_drawn_tile is None else int(self.own_drawn_tile)),
            "visible_discards": [[int(t) for t in river] for river in self.visible_discards],
            "visible_melds": [[meld.to_json() for meld in row] for row in self.visible_melds],
            "riichi_states": list(self.riichi_states),
            "dora_indicators": list(self.dora_indicators),
            "visible_history": [event.to_json() for event in self.visible_history],
            "legal_mask": list(self.legal_mask),
            "observation_hash": self.observation_hash,
        }
        return document

    def __repr__(self) -> str:
        """Leak-safe: routing facts plus digests, never tile or payload contents."""
        return (
            f"ActorObservation(game_id={self.game_id!r}, decision_id={self.decision_id!r}, "
            f"sequence={int(self.sequence)}, actor={int(self.actor)}, phase={self.phase!r}, "
            f"observation_hash={self.observation_hash})"
        )


_OBSERVATION_FIELDS = tuple(field.name for field in fields(ActorObservation))


def observation_identity_document(observation: ActorObservation) -> dict[str, object]:
    """Serialized field mapping WITHOUT ``observation_hash`` (hash input)."""
    document = observation.to_json()
    _ = document.pop("observation_hash", None)
    return document


def compute_observation_hash(observation: ActorObservation) -> DigestText:
    """sha256 over canonical bytes of the field dict minus observation_hash."""
    identity = canonical_json_bytes(observation_identity_document(observation))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def make_actor_observation(**field_values: object) -> ActorObservation:
    """Construct an observation with its observation_hash bound to the identity."""
    staged_names = tuple(field_values)
    if set(staged_names) | {"observation_hash"} != set(_OBSERVATION_FIELDS):
        missing = sorted(set(_OBSERVATION_FIELDS) - set(staged_names) - {"observation_hash"})
        unknown = sorted(set(staged_names) - set(_OBSERVATION_FIELDS))
        raise ContractError(
            f"make_actor_observation field mismatch; missing={missing}, unknown={unknown}"
        )
    body = {k: v for k, v in field_values.items() if k != "observation_hash"}
    staged = ActorObservation(**body, observation_hash=None)  # type: ignore[arg-type]  # reason: body keys pre-checked against _OBSERVATION_FIELDS above
    # Perf-C P1: the staged instance already ran the full __post_init__ firewall
    # (closed slots, visibility filter, dora contiguity, mask positivity), so
    # bind the digest in place instead of reconstructing (which re-validated
    # every field and re-serialized the history for a second hash). The bound
    # object is field-identical to a reconstructed one: normalization in
    # __post_init__ is idempotent and the digest is verified independently by
    # VISIBILITY_VALIDATOR at capture time.
    digest = compute_observation_hash(staged)
    object.__setattr__(staged, "observation_hash", digest)
    return staged
