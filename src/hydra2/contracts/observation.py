"""SPEC 8 actor observation, serialization, and visibility — WP-02D.

This module is the CANONICAL home (owner decision D-WP02D-1) of:

- ``DORA_SENTINEL`` / ``DORA_SHAPE``: fixed ``(5,)`` indicator shape whose
  revealed values stay contiguous from index 0; unrevealed slots carry
  ``DORA_SENTINEL`` (-1). Shape violations are rejected and NEVER padded.
- ``Phase`` / ``PHASES``: the SPEC section 8 phase union (action.py re-imports).
- ``MeldKind`` / ``MELD_KINDS`` / ``VisibleMeld`` / ``visible_meld_id``: the
  exposed-meld record with the SPEC 8 exact field set and order
  (meld_id, kind, owner, source_seat, called_tile, tiles). ``meld_id`` may be
  omitted (``None``) and is then derived through :func:`visible_meld_id`.
- ``ActorObservation``: the exact SPEC 8 field list. Its identity document is
  the serialized field mapping WITHOUT ``observation_hash``;
  ``observation_hash = sha256(canonical_json_bytes(identity))``
  (owner decision D-WP02D-7). Concealed hands serialize sorted by physical
  TileId while the drawn tile stays separate.
- ``ObservationBuilder``: builds observations from four ISOLATED per-seat
  caches (owner decision D-WP02D-6). Events are filtered by visibility BEFORE
  storage: public events enter every seat's cache, ``draw_tile`` enters only
  the drawing seat's cache, and server_private events are never stored
  anywhere. Public scalar state arrives through :meth:`update_public_state`;
  concealed hands through :meth:`set_concealed_hand` into that seat slot only.
  Discard rivers, melds, dora indicators, riichi states, kan counts, and drawn
  tiles derive exclusively from ingested events, so the observation can never
  contain wall/dead-wall, opponent concealed tiles, unrevealed dora/ura, RNG,
  future events, server-private events, opponent legal masks, or privileged
  labels: those fields do not exist on the type.
- ``VisibilityValidator``: rejects any event a seat may not hold and validates
  assembled observations against the same boundary. Debug representations and
  exception messages expose routing facts and digests only — never payload
  contents.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
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
from hydra2.contracts.event import (
    EVENT_SCHEMA_SCHEMA_VERSION,
    EventEnvelope,
    filter_events_for_actor,
    visible_to_actor,
)

__all__ = [
    "DORA_SENTINEL",
    "DORA_SHAPE",
    "MELD_KINDS",
    "OBSERVATION_SCHEMA_ARTIFACT_TYPE",
    "OBSERVATION_SCHEMA_RELPATH",
    "OBSERVATION_SCHEMA_SCHEMA_VERSION",
    "PHASES",
    "VISIBILITY_VALIDATOR",
    "ActorObservation",
    "MeldKind",
    "ObservationBuilder",
    "Phase",
    "VisibilityValidator",
    "build_observation_schema_envelope",
    "build_observation_schema_payload",
    "compute_observation_hash",
    "compute_observation_schema_digest",
    "load_observation_schema",
    "make_actor_observation",
    "observation_identity_document",
    "observation_schema_digest",
    "parse_observation_schema",
    "visible_meld_id",
]

# ---------------------------------------------------------------------------
# Fixed dora indicator shape (BUILD checklist: `(5,)`, declared sentinel).
# ---------------------------------------------------------------------------

#: Sentinel for an unrevealed dora indicator slot.
DORA_SENTINEL = -1
#: Exact indicator shape; rejected otherwise, NEVER padded implicitly.
DORA_SHAPE = (5,)

# ---------------------------------------------------------------------------
# SPEC 8 phase union - canonical definition (action.py re-imports).
# ---------------------------------------------------------------------------

Phase = Literal[
    "round_start",
    "draw_decision",
    "discard_response",
    "kan_response",
    "round_end",
    "game_end",
]

#: Every phase literal accepted by :class:`ActionContext` and observations.
PHASES: tuple[Phase, ...] = (
    "round_start",
    "draw_decision",
    "discard_response",
    "kan_response",
    "round_end",
    "game_end",
)

# ---------------------------------------------------------------------------
# Visible melds - canonical home (SPEC 8).
# ---------------------------------------------------------------------------

#: Meld kinds carried by :class:`VisibleMeld`.
MeldKind = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
MELD_KINDS: tuple[MeldKind, ...] = ("chi", "pon", "daiminkan", "ankan", "kakan")

_FURIETEN_STATES = ("none", "temporary", "riichi", "discard")
_RIICHI_STATES = ("none", "declared", "accepted")

_WIND_TILE_TYPES = (27, 28, 29, 30)  # East, South, West, North logical types


def _tile_type_of(tile: int) -> int:
    """Logical tile type 0..33 of a validated physical id."""
    return tile // 4


def _require_plain_int(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name}={value} outside [{minimum}, {maximum}]")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_enum(value: object, *, name: str, allowed: tuple[str, ...]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {list(allowed)}")
    return text


def _tile_tuple(values: Sequence[int], *, name: str) -> tuple[TileId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of tile ids")
    return tuple(make_tile_id(v) for v in values)


def _quad(values: object, *, name: str, validator) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
        raise ContractError(f"{name} must be exactly four entries")
    return tuple(validator(v, name=f"{name}[{i}]") for i, v in enumerate(values))


@dataclass(frozen=True, slots=True)
class VisibleMeld:
    """An exposed meld visible to every actor (SPEC 8 exact field order).

    ``tiles`` are strictly ascending physical ids composing the meld. Chi/pon/
    daiminkan carry the offering ``source_seat`` and claimed ``called_tile``;
    ankan/kakan are self-contained. ``meld_id`` may be ``None``, in which case
    the canonical :func:`visible_meld_id` reference is derived.
    """

    meld_id: str | None
    kind: MeldKind
    owner: Seat
    source_seat: Seat | None = None
    called_tile: TileId | None = None
    tiles: tuple[TileId, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in MELD_KINDS:
            raise ContractError(f"meld kind must be one of {MELD_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "owner", make_seat(self.owner))
        tiles = tuple(make_tile_id(t) for t in self.tiles)
        if len(tiles) == 0 or list(tiles) != sorted(set(tiles)):
            raise ContractError(
                f"{self.kind} meld tiles must be non-empty, unique, ascending: {tiles!r}"
            )
        object.__setattr__(self, "tiles", tiles)
        expected_len = {"chi": 3, "pon": 3, "daiminkan": 4, "ankan": 4, "kakan": 4}[self.kind]
        if len(tiles) != expected_len:
            raise ContractError(
                f"{self.kind} meld must hold {expected_len} tiles, got {len(tiles)}"
            )
        types = [_tile_type_of(t) for t in tiles]
        if self.kind == "chi":
            if (
                any(t >= 27 for t in types)
                or len({t // 9 for t in types}) != 1
                or max(types) - min(types) != 2
                or len(set(types)) != 3
            ):
                raise ContractError(f"chi meld is not a same-suit run: {tiles!r}")
        elif len(set(types)) != 1:
            raise ContractError(f"{self.kind} meld tiles must share one logical type: {tiles!r}")
        if self.kind in ("ankan", "kakan"):
            if self.called_tile is not None or self.source_seat is not None:
                raise ContractError(f"{self.kind} meld has no called tile or source seat")
            if self.kind == "ankan":
                base = 4 * types[0]
                if tiles != tuple(range(base, base + 4)):
                    raise ContractError(
                        f"ankan meld tiles {tiles!r} must be consecutive 4 of type {types[0]}"
                    )
        else:
            if self.called_tile is None or self.source_seat is None:
                raise ContractError(f"{self.kind} meld requires called_tile and source_seat")
            if isinstance(self.called_tile, bool) or not isinstance(self.called_tile, int):
                raise ContractError(
                    "called_tile must be a tile id int 0..135, got "
                    f"{type(self.called_tile).__name__}"
                )
            if isinstance(self.source_seat, bool) or not isinstance(self.source_seat, int):
                raise ContractError(
                    f"source_seat must be a seat int 0..3, got {type(self.source_seat).__name__}"
                )
            called = make_tile_id(int(self.called_tile))
            source = make_seat(int(self.source_seat))
            if source == self.owner:
                raise ContractError(f"{self.kind} meld source seat equals owner")
            if called not in tiles:
                raise ContractError(f"{self.kind} meld called tile {called} not among tiles")
        resolved = self.meld_id if self.meld_id is not None else visible_meld_id(self)
        if not isinstance(resolved, str) or resolved == "":
            raise ContractError("meld_id must resolve to a non-empty string")
        object.__setattr__(self, "meld_id", resolved)

    def to_json(self) -> dict[str, object]:
        return {
            "meld_id": self.meld_id,
            "kind": self.kind,
            "owner": int(self.owner),
            "source_seat": None if self.source_seat is None else int(self.source_seat),
            "called_tile": None if self.called_tile is None else int(self.called_tile),
            "tiles": [int(t) for t in self.tiles],
        }


def visible_meld_id(meld: VisibleMeld) -> str:
    """Canonical prior-meld reference used by kakan metadata (SPEC 6.2)."""
    return f"{meld.kind}:{'.'.join(str(int(t)) for t in meld.tiles)}"


# ---------------------------------------------------------------------------
# ActorObservation - exact SPEC 8 fields, in declaration order.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActorObservation:
    """One actor's complete legal view at one decision point (SPEC 8).

    The closed slot set IS the visibility boundary: wall/dead wall, opponent
    concealed tiles, unrevealed dora/ura, engine RNG, future events,
    server-private events, opponent legal masks, and privileged labels have no
    field to occupy. ``observation_hash`` binds the identity document.
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
            validator=lambda v, name: TileType(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                _require_plain_int(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                    v, name=name, minimum=_WIND_TILE_TYPES[0], maximum=_WIND_TILE_TYPES[-1]  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                )
            ),
        )
        if sorted(int(w) for w in winds) != list(_WIND_TILE_TYPES):  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
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
                validator=lambda v, name: _require_plain_int(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                    v, name=name, minimum=-(10**9), maximum=10**9  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                ),
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
                validator=lambda v, name: _require_enum(v, name=name, allowed=_RIICHI_STATES),  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
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
    digest = compute_observation_hash(staged)
    return ActorObservation(**body, observation_hash=digest)  # type: ignore[arg-type]  # reason: body keys pre-checked against _OBSERVATION_FIELDS above


# ---------------------------------------------------------------------------
# VisibilityValidator (SPEC 8 protocol).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ObservationSchema - closed versioned artifact derived from ActorObservation
# (SPEC 8 tail: fixes field order, enum/range constraints, serialization;
# SPEC 21 config category). Contracts never import hydra2.artifacts, so the
# digest is computed through the byte-identical leaf canonicalizer.
# ---------------------------------------------------------------------------

OBSERVATION_SCHEMA_ARTIFACT_TYPE = "hydra2.observation_schema"
OBSERVATION_SCHEMA_SCHEMA_VERSION = "1.0.0"
OBSERVATION_SCHEMA_RELPATH = Path("configs") / "contracts" / "observation_schema_v1.json"

_OBSERVATION_SCHEMA_ENVELOPE_FIELDS = (
    "artifact_type",
    "schema_version",
    "compatibility",
    "payload",
)

_SEAT_SPEC: dict[str, object] = {"dtype": "seat", "minimum": 0, "maximum": 3}
_TILE_ID_SPEC: dict[str, object] = {"dtype": "tile_id", "minimum": 0, "maximum": 135}
_DIGEST_SPEC: dict[str, object] = {"dtype": "digest_text", "pattern": "sha256:[0-9a-f]{64}"}
_NONEMPTY_STRING_SPEC: dict[str, object] = {"dtype": "string", "min_length": 1}

#: Serialization row for one exposed meld (SPEC 8 ``VisibleMeld`` field order).
_VISIBLE_MELD_ROW: dict[str, object] = {
    "field_order": ["meld_id", "kind", "owner", "source_seat", "called_tile", "tiles"],
    "fields": {
        "meld_id": {"dtype": "string", "nullable": True, "derived_when_null": "visible_meld_id"},
        "kind": {"dtype": "enum", "values": list(MELD_KINDS)},
        "owner": dict(_SEAT_SPEC),
        "source_seat": {**_SEAT_SPEC, "nullable": True},
        "called_tile": {**_TILE_ID_SPEC, "nullable": True},
        "tiles": {
            "dtype": "tile_id_array",
            "item_minimum": 0,
            "item_maximum": 135,
            "constraints": ["nonempty_unique_ascending"],
        },
    },
    "kind_tile_counts": {"chi": 3, "pon": 3, "daiminkan": 4, "ankan": 4, "kakan": 4},
    "kind_shapes": {
        "chi": "three consecutive logical types inside one suit",
        "pon_daiminkan_ankan_kakan": "all tiles share one logical type",
    },
    "ankan_requires_all_four_copies": True,
    "call_kinds_with_source_and_called_tile": ["chi", "pon", "daiminkan"],
    "self_kinds_without_source_or_called_tile": ["ankan", "kakan"],
    "source_seat_must_differ_from_owner": True,
}

#: One closed constraint row per ``ActorObservation`` field. The table is
#: checked against the live dataclass when the payload is built: adding,
#: removing, renaming, or reordering a field without extending the schema
#: raises instead of silently drifting from the published artifact.
_FIELD_CONSTRAINTS: dict[str, dict[str, object]] = {
    "game_id": dict(_NONEMPTY_STRING_SPEC),
    "decision_id": dict(_NONEMPTY_STRING_SPEC),
    "sequence": {"dtype": "sequence_no", "minimum": 0},
    "actor": dict(_SEAT_SPEC),
    "rules_id": dict(_NONEMPTY_STRING_SPEC),
    "rules_hash": dict(_DIGEST_SPEC),
    "action_table_hash": dict(_DIGEST_SPEC),
    "event_schema_hash": dict(_DIGEST_SPEC),
    "observation_schema_hash": dict(_DIGEST_SPEC),
    "packet_boundary_hash": dict(_DIGEST_SPEC),
    "round_index": {"dtype": "integer", "minimum": 0},
    "round_wind": {"dtype": "tile_type", "minimum": 0, "maximum": 33},
    "hand_number": {"dtype": "integer", "minimum": 0},
    "seat_winds": {
        "dtype": "tile_type",
        "length": 4,
        "item_minimum": 27,
        "item_maximum": 30,
        "constraints": ["permutes_east_south_west_north_aligned_by_seat"],
    },
    "honba": {"dtype": "integer", "minimum": 0},
    "riichi_sticks": {"dtype": "integer", "minimum": 0},
    "dealer": dict(_SEAT_SPEC),
    "scores": {
        "dtype": "integer",
        "length": 4,
        "item_minimum": -(10**9),
        "item_maximum": 10**9,
        "int32_safe": True,
    },
    "turn_actor": dict(_SEAT_SPEC),
    "phase": {"dtype": "enum", "values": list(PHASES)},
    "live_wall_tiles_remaining": {"dtype": "integer", "minimum": 0},
    "kan_count": {"dtype": "integer", "minimum": 0, "maximum": 4},
    "ippatsu_active": {"dtype": "boolean", "length": 4},
    "actor_furiten": {"dtype": "enum", "values": list(_FURIETEN_STATES)},
    "actor_can_tsumo": {"dtype": "boolean"},
    "actor_can_riichi": {"dtype": "boolean"},
    "pending_declaration_discard": {**_TILE_ID_SPEC, "nullable": True},
    "concealed_hand": {
        "dtype": "tile_id_array",
        "item_minimum": 0,
        "item_maximum": 135,
        "constraints": ["serialized_ascending_duplicates_allowed"],
    },
    "own_drawn_tile": {**_TILE_ID_SPEC, "nullable": True},
    "visible_discards": {
        "dtype": "array",
        "length": 4,
        "items": {"dtype": "tile_id_array", "item_minimum": 0, "item_maximum": 135},
    },
    "visible_melds": {
        "dtype": "array",
        "length": 4,
        "items": {"dtype": "visible_meld_array"},
        "element_row_ref": "visible_meld_row",
    },
    "riichi_states": {"dtype": "enum", "length": 4, "values": list(_RIICHI_STATES)},
    "dora_indicators": {
        "dtype": "dora_slot_array",
        "length": 5,
        "sentinel": DORA_SENTINEL,
        "revealed_minimum": 0,
        "revealed_maximum": 135,
        "constraints": [
            "shape_exactly_five_never_padded",
            "revealed_contiguous_from_index_zero",
            "sentinel_tail_only",
        ],
    },
    "visible_history": {
        "dtype": "event_envelope_array",
        "element_artifact": {
            "artifact_type": "hydra2.event_schema",
            "schema_version": EVENT_SCHEMA_SCHEMA_VERSION,
            "relpath": "configs/contracts/event_schema_v1.json",
        },
        "constraints": ["every_entry_visible_to_the_observed_actor"],
    },
    "legal_mask": {
        "dtype": "boolean_array",
        "min_length": 1,
        "constraints": [
            "at_least_one_true_at_every_decision",
            "length_equals_published_action_table_actions",
        ],
    },
    "observation_hash": dict(_DIGEST_SPEC),
}


def build_observation_schema_payload() -> dict[str, object]:
    """Deterministic ObservationSchema payload WITHOUT the digest field."""
    names = tuple(field.name for field in fields(ActorObservation))
    declared = set(_FIELD_CONSTRAINTS)
    if set(names) != declared:
        raise ContractError(
            "observation schema must stay closed over ActorObservation: "
            f"fields without constraint rows={sorted(set(names) - declared)} "
            f"constraint rows without fields={sorted(declared - set(names))}"
        )
    return {
        "schema_version": OBSERVATION_SCHEMA_SCHEMA_VERSION,
        "field_order": list(names),
        "fields": {name: _FIELD_CONSTRAINTS[name] for name in names},
        "visible_meld_row": _VISIBLE_MELD_ROW,
        "enums": {
            "phase": list(PHASES),
            "actor_furiten": list(_FURIETEN_STATES),
            "riichi_state": list(_RIICHI_STATES),
            "meld_kind": list(MELD_KINDS),
        },
        "identity": {
            "hash_field": "observation_hash",
            "excluded_fields": ["observation_hash"],
            "hash_rule": (
                "sha256 over RFC 8785 canonical bytes of the serialized field "
                "document WITHOUT observation_hash"
            ),
        },
        "serialization": {
            "encoding": "rfc8785_canonical_json_utf8_no_bom",
            "document_order": "ActorObservation declaration order",
            "concealed_hand": "ascending physical TileId; drawn tile stays separate",
            "seats_tiles_sequences": "serialized as JSON integers",
            "enums": "serialized as literal strings",
            "null_semantics": "explicit null; absent keys prohibited",
        },
        "visibility_boundary": {
            "forbidden_content": [
                "wall",
                "dead_wall",
                "opponent_concealed_tiles",
                "unrevealed_dora_indicators",
                "ura_dora",
                "engine_rng_state",
                "future_events",
                "server_private_events",
                "opponent_legal_masks",
                "privileged_labels",
            ],
        },
    }


def compute_observation_schema_digest(payload_without_digest: Mapping[str, object]) -> DigestText:
    """sha256 over canonical bytes of the digest-stripped payload."""
    identity = canonical_json_bytes(dict(payload_without_digest))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def observation_schema_digest() -> DigestText:
    """Digest of the compiled schema; stamped onto every built observation."""
    return compute_observation_schema_digest(build_observation_schema_payload())


def build_observation_schema_envelope() -> dict[str, object]:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    payload = build_observation_schema_payload()
    payload["digest"] = compute_observation_schema_digest(payload)
    return {
        "artifact_type": OBSERVATION_SCHEMA_ARTIFACT_TYPE,
        "schema_version": OBSERVATION_SCHEMA_SCHEMA_VERSION,
        "compatibility": "exact",
        "payload": payload,
    }


def _reject_json_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_observation_schema(raw_bytes: bytes) -> dict[str, object]:
    """Verify observation-schema artifact bytes; returns the envelope document."""
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"observation_schema artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping) or tuple(sorted(document)) != tuple(
        sorted(_OBSERVATION_SCHEMA_ENVELOPE_FIELDS)
    ):
        raise ContractError("observation_schema artifact must be a SPEC 2.2 envelope")
    if document["artifact_type"] != OBSERVATION_SCHEMA_ARTIFACT_TYPE:
        raise ContractError("artifact_type must be 'hydra2.observation_schema'")
    if document["compatibility"] != "exact":
        raise ContractError("observation_schema compatibility must be exact")
    payload: object = document["payload"]  # type: ignore[index]  # reason: document Mapping-checked above; checker cannot narrow object index
    if not isinstance(payload, Mapping) or "digest" not in payload:  # type: ignore[attr-defined]  # reason: isinstance-narrowed Mapping; checker flags 'in' on bare Mapping
        raise ContractError("observation_schema payload missing digest")
    expected = compute_observation_schema_digest(
        {k: v for k, v in payload.items() if k != "digest"}  # type: ignore[attr-defined]  # reason: payload Mapping-narrowed above; checker flags .items on bare Mapping
    )
    recorded = make_digest_text(str(payload["digest"]))  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict  # type: ignore[index]  # reason: payload Mapping-narrowed above; index on bare Mapping
    if not hmac.compare_digest(str(recorded), str(expected)):
        raise DigestMismatchError(
            f"observation_schema digest mismatch: recorded {recorded} != recomputed {expected}"
        )
    compiled = build_observation_schema_payload()
    if {k: v for k, v in payload.items() if k != "digest"} != compiled:
        raise ContractError("observation_schema artifact diverges from the compiled schema")
    return dict(document)


def load_observation_schema(path: Path) -> dict[str, object]:
    """Read and verify the published artifact at ``path``."""
    return parse_observation_schema(Path(path).read_bytes())


class VisibilityValidator:
    """Guards the actor-visible boundary for events and assembled observations."""

    __slots__ = ()

    def validate_event_for_actor(self, event: EventEnvelope, actor: Seat) -> None:
        """Reject any event ``actor`` may not legitimately hold."""
        if not isinstance(event, EventEnvelope):
            raise ContractError("event must be an EventEnvelope")
        seat = make_seat(int(actor))
        if not visible_to_actor(event, seat):
            raise VisibilityViolationError(
                f"seat {int(seat)} may not hold {event.visibility} event "
                f"{event.kind!r} at sequence {int(event.sequence)}"
            )

    def validate_observation(self, observation: ActorObservation) -> None:
        """Re-check the assembled observation against the visibility boundary."""
        if not isinstance(observation, ActorObservation):
            raise ContractError("observation must be an ActorObservation")
        # Structural fields already validated in __post_init__ (closed slot set,
        # per-seat history filtering, dora contiguity, mask positivity, hash).
        # Re-derive the hash here so a tampered frozen instance cannot pass.
        recomputed = compute_observation_hash(observation)
        if observation.observation_hash != recomputed:
            raise DigestMismatchError(
                f"observation_hash mismatch: recorded {observation.observation_hash} != "
                f"recomputed {recomputed}"
            )


#: Shared stateless validator instance (SPEC 8 protocol object).
VISIBILITY_VALIDATOR = VisibilityValidator()


# ---------------------------------------------------------------------------
# ObservationBuilder - four isolated per-seat caches, filtered before storage.
# ---------------------------------------------------------------------------

#: Closed public-snapshot vocabulary supplied through ``update_public_state``.
_PUBLIC_SNAPSHOT_FIELDS = (
    "decision_id",
    "round_index",
    "round_wind",
    "hand_number",
    "seat_winds",
    "honba",
    "riichi_sticks",
    "dealer",
    "scores",
    "turn_actor",
    "phase",
    "live_wall_tiles_remaining",
    "ippatsu_active",
)

_CALL_MELD_KINDS = ("chi", "pon", "daiminkan")


class ObservationBuilder:
    """Ingests the event stream once; serves isolated per-seat observations.

    One history/cache per seat exists from the first ingestion; the builder
    never materializes a full-state object to strip later. Visibility is
    applied BEFORE storage:

    - ``public`` events append to all four seat caches and drive the derived
      round caches (discard rivers, melds, dora indicators, riichi states,
      kan count);
    - ``actor_private`` events (``draw_tile``) append to the drawing seat's
      cache only and set that seat's ``own_drawn_tile``;
    - ``server_private`` events are dropped unconditionally and leave no
      trace in any cache, repr, or error message.

    Round-scoped caches reset on the public ``round_start`` event so one
    builder can serve a whole game deterministically. Every built observation
    carries the published ObservationSchema artifact digest
    (``observation_schema_digest()``) as ``observation_schema_hash``; callers
    cannot supply divergent lineage.
    """

    __slots__ = (
        "_action_table_hash",
        "_can_riichi",
        "_can_tsumo",
        "_concealed",
        "_discards",
        "_dora",
        "_drawn",
        "_event_schema_hash",
        "_furiten",
        "_game_id",
        "_histories",
        "_kan_count",
        "_last_sequence",
        "_mask_length",
        "_melds",
        "_obs_schema_hash",
        "_packet_boundary_hash",
        "_pending_discard",
        "_public",
        "_riichi_states",
        "_rules_hash",
        "_rules_id",
    )

    def __init__(
        self,
        *,
        game_id: str,
        rules_id: str,
        rules_hash: DigestText,
        action_table_hash: DigestText,
        expected_legal_mask_length: int,
        event_schema_hash: DigestText,
        packet_boundary_hash: DigestText,
    ) -> None:
        if game_id == "" or not isinstance(game_id, str):
            raise ContractError("game_id must be a non-empty string")
        if (
            isinstance(expected_legal_mask_length, bool)
            or not isinstance(expected_legal_mask_length, int)
            or expected_legal_mask_length <= 0
        ):
            raise ContractError("expected_legal_mask_length must be a positive int")
        self._game_id = game_id
        self._rules_id = _require_str(rules_id, name="rules_id")
        self._rules_hash = make_digest_text(rules_hash)
        self._action_table_hash = make_digest_text(action_table_hash)
        self._mask_length = expected_legal_mask_length
        self._event_schema_hash = make_digest_text(event_schema_hash)
        self._obs_schema_hash = observation_schema_digest()
        self._packet_boundary_hash = make_digest_text(packet_boundary_hash)
        self._histories: tuple[list[EventEnvelope], ...] = ([], [], [], [])
        self._concealed: list[tuple[TileId, ...] | None] = [None, None, None, None]
        self._drawn: list[int | None] = [None, None, None, None]
        self._discards: tuple[list[int], ...] = ([], [], [], [])
        self._melds: tuple[list[VisibleMeld], ...] = ([], [], [], [])
        self._riichi_states: list[str] = ["none", "none", "none", "none"]
        self._dora: list[int] = []
        self._kan_count = 0
        self._furiten: list[str] = ["none", "none", "none", "none"]
        self._can_tsumo: list[bool] = [False, False, False, False]
        self._can_riichi: list[bool] = [False, False, False, False]
        self._pending_discard: list[int | None] = [None, None, None, None]
        self._public: dict[str, object] = {}
        self._last_sequence: int | None = None

    def __repr__(self) -> str:
        """Leak-safe: cache occupancy counts only, never contents."""
        return (
            f"ObservationBuilder(game_id={self._game_id!r}, "
            f"histories={[len(h) for h in self._histories]}, "
            f"dora_revealed={len([d for d in self._dora if d != DORA_SENTINEL])}, "
            f"last_sequence={self._last_sequence})"
        )

    # -- ingestion ---------------------------------------------------------

    def append_visible(self, event: EventEnvelope) -> None:
        """Route one sequenced event into exactly the caches allowed to hold it."""
        if not isinstance(event, EventEnvelope):
            raise ContractError("event must be an EventEnvelope")
        sequence = int(event.sequence)
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise ContractError(
                f"sequence {sequence} does not strictly increase past {self._last_sequence}"
            )
        self._last_sequence = sequence
        if event.visibility == "server_private":
            return  # never stored anywhere (owner decision D-WP02D-6)
        if event.visibility == "actor_private":
            seat = int(event.visible_to[0])
            self._histories[seat].append(event)
            if event.kind == "draw_tile":
                self._drawn[seat] = int(event.payload.tile)  # type: ignore[arg-type]  # reason: draw_tile shape guarantees non-None tile; int() validates
            return
        for seat in range(4):
            self._histories[seat].append(event)
        self._apply_public_effects(event)

    def _apply_public_effects(self, event: EventEnvelope) -> None:
        kind = event.kind
        actor = None if event.payload.actor is None else int(event.payload.actor)
        if kind == "round_start":
            self._reset_round_caches()
            return
        if kind == "discard":
            self._discards[actor].append(int(event.payload.tile))  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType
        elif kind in _CALL_MELD_KINDS:
            claimed = event.payload.tile
            assert claimed is not None  # kind-shape validation guarantees the tile
            meld = VisibleMeld(
                meld_id=None,
                kind=kind,
                owner=event.payload.actor,  # type: ignore[arg-type]  # reason: kind-shape validation guarantees seat actor; checker sees object
                source_seat=event.payload.source_seat,
                called_tile=claimed,
                tiles=tuple(sorted([*event.payload.consumed_tiles, claimed])),
            )
            self._melds[actor].append(meld)  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType
            self._kan_count += 1
        elif kind == "ankan":
            meld = VisibleMeld(
                meld_id=None,
                kind="ankan",
                owner=event.payload.actor,  # type: ignore[arg-type]  # reason: kind-shape validation guarantees seat actor; checker sees object
                tiles=tuple(sorted(event.payload.consumed_tiles)),
            )
            self._melds[actor].append(meld)  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType
            self._kan_count += 1
        elif kind == "dora_revealed":
            revealed = event.payload.tile
            assert revealed is not None  # kind-shape validation guarantees the tile
            if len(self._dora) >= DORA_SHAPE[0]:
                raise ContractError(
                    f"dora indicator {int(revealed)} exceeds the fixed "
                    f"{DORA_SHAPE[0]}-slot shape; never padded or truncated"
                )
            self._dora.append(int(revealed))
        elif kind == "riichi_declared":
            self._riichi_states[actor] = "declared"  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType
        elif kind == "riichi_accepted":
            self._riichi_states[actor] = "accepted"  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType

    def _upgrade_kakan(self, actor: int, added_tile: int) -> None:
        """Replace the owner's prior pon with the upgraded kakan meld in place."""
        added_type = _tile_type_of(added_tile)
        for index, meld in enumerate(self._melds[actor]):
            if meld.kind == "pon" and _tile_type_of(int(meld.tiles[0])) == added_type:
                tiles = tuple(sorted((*meld.tiles, make_tile_id(added_tile))))
                self._melds[actor][index] = VisibleMeld(
                    meld_id=None, kind="kakan", owner=meld.owner, tiles=tiles
                )
                return
        raise ContractError(
            f"kakan of tile {added_tile}: no prior pon of type {added_type} owned by seat {actor}"
        )

    def _reset_round_caches(self) -> None:
        self._discards = ([], [], [], [])
        self._melds = ([], [], [], [])
        self._riichi_states = ["none", "none", "none", "none"]
        self._dora = []
        self._kan_count = 0
        self._drawn = [None, None, None, None]

    # -- explicit state supplies ---------------------------------------------

    def set_concealed_hand(self, actor: Seat, tiles: Sequence[int]) -> None:
        """Store ONE seat's concealed hand; other seats' slots are untouched."""
        seat = int(make_seat(int(actor)))
        hand = _tile_tuple(tiles, name="concealed_hand")
        self._concealed[seat] = tuple(sorted(hand))

    def set_actor_state(
        self,
        actor: Seat,
        *,
        furiten: str | None = None,
        can_tsumo: bool | None = None,
        can_riichi: bool | None = None,
        pending_declaration_discard: int | None = None,
    ) -> None:
        """Update one seat's eligibility facts (partial updates allowed)."""
        seat = int(make_seat(int(actor)))
        if furiten is not None:
            self._furiten[seat] = _require_enum(furiten, name="furiten", allowed=_FURIETEN_STATES)
        if can_tsumo is not None:
            self._can_tsumo[seat] = _require_bool(can_tsumo, name="can_tsumo")
        if can_riichi is not None:
            self._can_riichi[seat] = _require_bool(can_riichi, name="can_riichi")
        if pending_declaration_discard is not None:
            self._pending_discard[seat] = int(make_tile_id(pending_declaration_discard))

    def update_public_state(self, **snapshot: object) -> None:
        """Supply the authoritative public scalar state (closed vocabulary)."""
        unknown = sorted(set(snapshot) - set(_PUBLIC_SNAPSHOT_FIELDS))
        if len(unknown) > 0:
            raise ContractError(f"unknown public snapshot fields: {unknown}")
        if "decision_id" in snapshot:
            decision_id = _require_str(snapshot["decision_id"], name="decision_id")
            if decision_id == "":
                raise ContractError("decision_id must be non-empty")
            self._public["decision_id"] = decision_id
        if "round_index" in snapshot:
            self._public["round_index"] = _require_plain_int(
                snapshot["round_index"], name="round_index", minimum=0, maximum=None
            )
        if "round_wind" in snapshot:
            self._public["round_wind"] = make_tile_type(snapshot["round_wind"])  # type: ignore[arg-type]  # reason: snapshot value statically object; validated inside make_tile_type
        if "hand_number" in snapshot:
            self._public["hand_number"] = _require_plain_int(
                snapshot["hand_number"], name="hand_number", minimum=0, maximum=None
            )
        if "seat_winds" in snapshot:
            winds = _quad(
                snapshot["seat_winds"],
                name="seat_winds",
                validator=lambda v, name: TileType(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                    _require_plain_int(v, name=name, minimum=27, maximum=33)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                ),
            )
            if sorted(int(w) for w in winds) != list(_WIND_TILE_TYPES):  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                raise ContractError("seat_winds must permute East/South/West/North")
            self._public["seat_winds"] = winds
        if "honba" in snapshot:
            self._public["honba"] = _require_plain_int(
                snapshot["honba"], name="honba", minimum=0, maximum=None
            )
        if "riichi_sticks" in snapshot:
            self._public["riichi_sticks"] = _require_plain_int(
                snapshot["riichi_sticks"], name="riichi_sticks", minimum=0, maximum=None
            )
        if "dealer" in snapshot:
            self._public["dealer"] = make_seat(snapshot["dealer"])  # type: ignore[arg-type]  # reason: snapshot value statically object; validated inside make_seat
        if "scores" in snapshot:
            self._public["scores"] = _quad(
                snapshot["scores"],
                name="scores",
                validator=lambda v, name: _require_plain_int(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                    v, name=name, minimum=-(10**9), maximum=10**9  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                ),
            )
        if "turn_actor" in snapshot:
            self._public["turn_actor"] = make_seat(snapshot["turn_actor"])  # type: ignore[arg-type]  # reason: snapshot value statically object; validated inside make_seat
        if "phase" in snapshot:
            self._public["phase"] = _require_enum(snapshot["phase"], name="phase", allowed=PHASES)
        if "live_wall_tiles_remaining" in snapshot:
            self._public["live_wall_tiles_remaining"] = _require_plain_int(
                snapshot["live_wall_tiles_remaining"],
                name="live_wall_tiles_remaining",
                minimum=0,
                maximum=None,
            )
        if "ippatsu_active" in snapshot:
            self._public["ippatsu_active"] = _quad(
                snapshot["ippatsu_active"], name="ippatsu_active", validator=_require_bool
            )

    # -- assembly ------------------------------------------------------------

    def build(self, *, actor: Seat, legal_mask: Sequence[bool]) -> ActorObservation:
        """Assemble the observation for ONE seat from its isolated cache."""
        seat = int(make_seat(int(actor)))
        mask = tuple(legal_mask)
        if len(mask) != self._mask_length:
            raise ContractError(
                f"legal_mask length {len(mask)} != action table length {self._mask_length}; "
                "masks align with the canonical action vocabulary and are NEVER padded"
            )
        flags: list[bool] = []
        for index, flag in enumerate(mask):
            if not isinstance(flag, bool):
                raise ContractError(f"legal_mask[{index}] must be a bool")
            flags.append(flag)
        if not any(flags):
            raise ContractError("legal_mask must contain at least one True at a decision")
        missing = [name for name in _PUBLIC_SNAPSHOT_FIELDS if name not in self._public]
        if len(missing) > 0:
            raise ContractError(f"public snapshot incomplete; missing {missing}")
        indicators = tuple(self._dora) + (DORA_SENTINEL,) * (DORA_SHAPE[0] - len(self._dora))
        conceal = self._concealed[seat]
        pending = self._pending_discard[seat]
        history = filter_events_for_actor(self._histories[seat], make_seat(seat))
        return make_actor_observation(
            game_id=self._game_id,
            decision_id=str(self._public["decision_id"]),
            sequence=make_sequence_no(history[-1].sequence if len(history) > 0 else 0),
            actor=make_seat(seat),
            rules_id=self._rules_id,
            rules_hash=self._rules_hash,
            action_table_hash=self._action_table_hash,
            event_schema_hash=self._event_schema_hash,
            observation_schema_hash=self._obs_schema_hash,
            packet_boundary_hash=self._packet_boundary_hash,
            round_index=int(self._public["round_index"]),  # type: ignore[arg-type]  # reason: _public store is dict[str, object]; int() validates
            round_wind=self._public["round_wind"],
            hand_number=int(self._public["hand_number"]),  # type: ignore[arg-type]  # reason: _public store is dict[str, object]; int() validates
            seat_winds=self._public["seat_winds"],
            honba=int(self._public["honba"]),  # type: ignore[arg-type]  # reason: _public store is dict[str, object]; int() validates
            riichi_sticks=int(self._public["riichi_sticks"]),  # type: ignore[arg-type]  # reason: _public store is dict[str, object]; int() validates
            dealer=self._public["dealer"],
            scores=self._public["scores"],
            turn_actor=self._public["turn_actor"],
            phase=self._public["phase"],
            live_wall_tiles_remaining=int(self._public["live_wall_tiles_remaining"]),  # type: ignore[arg-type]  # reason: _public store is dict[str, object]; int() validates
            kan_count=self._kan_count,
            ippatsu_active=self._public["ippatsu_active"],
            actor_furiten=self._furiten[seat],
            actor_can_tsumo=self._can_tsumo[seat],
            actor_can_riichi=self._can_riichi[seat],
            pending_declaration_discard=(None if pending is None else make_tile_id(pending)),
            concealed_hand=conceal if conceal is not None else (),
            own_drawn_tile=(
                None if self._drawn[seat] is None else make_tile_id(self._drawn[seat])  # type: ignore[arg-type]  # reason: None filtered by ternary; range validated inside make_tile_id
            ),
            visible_discards=tuple(tuple(river) for river in self._discards),
            visible_melds=tuple(tuple(row) for row in self._melds),
            riichi_states=tuple(self._riichi_states),
            dora_indicators=indicators,
            visible_history=tuple(history),
            legal_mask=tuple(flags),
        )
