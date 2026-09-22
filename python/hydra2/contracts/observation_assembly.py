"""SPEC 8 observation assembly — validator and per-seat builder.

Owns the isolated per-seat assembly: the stateless validator shell with
its shared instance and history cap, and the event stream ingestion with
per-seat filtering before storage. Schema tables and the versioned
artifact live in the sibling schema module; the dataclass lives in the
actor module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    Seat,
    TileId,
    TileType,
    VisibilityViolationError,
    make_sequence_no,
    make_tile_type,
)
from hydra2.contracts.event_envelope import (
    EventEnvelope,
    filter_events_for_actor,
    visible_to_actor,
)
from hydra2.contracts.observation_actor import (
    ActorObservation,
    compute_observation_hash,
    make_actor_observation,
)
from hydra2.contracts.observation_schema import observation_schema_digest
from hydra2.contracts.observation_types import (
    _FURIETEN_STATES,
    _WIND_TILE_TYPES,
    DORA_SENTINEL,
    DORA_SHAPE,
    PHASES,
    VisibleMeld,
    _quad,
    _require_bool,
    _require_enum,
    _require_plain_int,
    _require_str,
    _tile_tuple,
    _tile_type_of,
    _validate_score,
)

__all__ = [
    "HISTORY_EVENT_CAP",
    "VISIBILITY_VALIDATOR",
    "ObservationBuilder",
    "VisibilityValidator",
]

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# VisibilityValidator (SPEC 8 protocol).
# ---------------------------------------------------------------------------
class VisibilityValidator:
    """Guards the actor-visible boundary for events and assembled observations."""

    __slots__ = ()

    def validate_event_for_actor(self, event: EventEnvelope, actor: Seat) -> None:
        """Reject any event ``actor`` may not legitimately hold."""
        if not isinstance(event, EventEnvelope):
            raise ContractError("event must be an EventEnvelope")
        seat: Seat = _bridge_contracts.make_seat(int(actor))
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

#: Max visible-history events a row may carry. Mirrors
#: ``HISTORY_BUCKET_LENGTHS[-1]`` (models/schema, the model cap); pinned
#: equal by test. Histories are per-kyoku and NEVER truncated silently --
#: over-cap rows fail closed at capture/encode time instead. Single source:
#: ``hydra2._native.contracts.HISTORY_EVENT_CAP`` — the literal fallback keeps
#: stale-.so environments importable and is byte-identical.
HISTORY_EVENT_CAP: int = int(getattr(_bridge_contracts, "HISTORY_EVENT_CAP", 256))

# ---------------------------------------------------------------------------
# ObservationBuilder - four isolated per-seat caches, filtered before storage.
# ---------------------------------------------------------------------------

#: Closed public-snapshot vocabulary supplied through ``update_public_state``.
#: Single source: ``hydra2._native.contracts._PUBLIC_SNAPSHOT_FIELDS``
#: (oracle order) — the literal fallback keeps stale-.so environments
#: importable and is byte-identical.
_PUBLIC_SNAPSHOT_FIELDS: tuple[str, ...] = tuple(
    getattr(
        _bridge_contracts,
        "_PUBLIC_SNAPSHOT_FIELDS",
        (
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
        ),
    )
)

_CALL_MELD_KINDS = ("chi", "pon", "daiminkan")


def _check_dora_append(dora_len: object, revealed_tile: object) -> None:
    """Gate one dora-indicator append (bridge ``obs_dora_append_slot``)."""
    gate = getattr(_bridge_contracts, "obs_dora_append_slot", None)
    if gate is None:  # stale .so: oracle gate inline (byte-identical text)
        if int(dora_len) >= DORA_SHAPE[0]:  # type: ignore[arg-type]  # reason: fallback receives len() ints; int() preserves oracle coercion
            raise ContractError(
                f"dora indicator {int(revealed_tile)} exceeds the fixed "  # type: ignore[arg-type]  # reason: fallback receives int-coerced tiles; int() preserves oracle rendering
                f"{DORA_SHAPE[0]}-slot shape; never padded or truncated"
            )
        return
    try:
        gate(dora_len, revealed_tile)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _check_public_unknown(supplied: object) -> None:
    """Gate supplied snapshot keys (bridge ``obs_public_unknown``)."""
    gate = getattr(_bridge_contracts, "obs_public_unknown", None)
    if gate is None:  # stale .so: oracle gate inline (byte-identical text)
        unknown = sorted(set(supplied) - set(_PUBLIC_SNAPSHOT_FIELDS))  # type: ignore[arg-type]  # reason: fallback receives the snapshot mapping; set() reads its keys like the oracle
        if len(unknown) > 0:
            raise ContractError(f"unknown public snapshot fields: {unknown}")
        return
    try:
        gate(list(supplied))  # type: ignore[arg-type]  # reason: supplied is the snapshot key iterable; bridge validates element types
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def _check_public_missing(present: object) -> None:
    """Gate snapshot completeness (bridge ``obs_public_missing``)."""
    gate = getattr(_bridge_contracts, "obs_public_missing", None)
    if gate is None:  # stale .so: oracle gate inline (byte-identical text)
        missing = [name for name in _PUBLIC_SNAPSHOT_FIELDS if name not in present]  # type: ignore[operator]  # reason: fallback receives the public-state mapping; `in` reads its keys like the oracle
        if len(missing) > 0:
            raise ContractError(f"public snapshot incomplete; missing {missing}")
        return
    try:
        gate(list(present))  # type: ignore[arg-type]  # reason: present is the public-state key iterable; bridge validates element types
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


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
        self._rules_hash: DigestText = _bridge_contracts.make_digest_text(rules_hash)
        self._action_table_hash: DigestText = _bridge_contracts.make_digest_text(action_table_hash)
        self._mask_length = expected_legal_mask_length
        self._event_schema_hash: DigestText = _bridge_contracts.make_digest_text(event_schema_hash)
        self._obs_schema_hash = observation_schema_digest()
        self._packet_boundary_hash: DigestText = _bridge_contracts.make_digest_text(
            packet_boundary_hash
        )
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
            if kind == "daiminkan":
                # Chi/pon open the hand but are not kans (grammar delta_paths agrees).
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
        elif kind == "kakan":
            added = event.payload.tile
            assert added is not None  # kind-shape validation guarantees the tile
            assert actor is not None  # grammar requires actor for kakan
            self._upgrade_kakan(actor, int(added))
            self._kan_count += 1
        elif kind == "dora_revealed":
            revealed = event.payload.tile
            assert revealed is not None  # kind-shape validation guarantees the tile
            tile_int = int(revealed)
            _check_dora_append(len(self._dora), tile_int)
            self._dora.append(tile_int)
        elif kind == "riichi_declared":
            self._riichi_states[actor] = "declared"  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType
        elif kind == "riichi_accepted":
            self._riichi_states[actor] = "accepted"  # type: ignore[index]  # reason: actor is a validated seat int; container keyed by Seat NewType

    def _upgrade_kakan(self, actor: int, added_tile: int) -> None:
        """Replace the owner's prior pon with the upgraded kakan meld in place."""
        added_type = _tile_type_of(added_tile)
        for index, meld in enumerate(self._melds[actor]):
            if meld.kind == "pon" and _tile_type_of(int(meld.tiles[0])) == added_type:
                tiles = tuple(sorted((*meld.tiles, _bridge_contracts.make_tile_id(added_tile))))
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
        seat: Seat = _bridge_contracts.make_seat(int(actor))
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
        seat: Seat = _bridge_contracts.make_seat(int(actor))
        if furiten is not None:
            self._furiten[seat] = _require_enum(furiten, name="furiten", allowed=_FURIETEN_STATES)
        if can_tsumo is not None:
            self._can_tsumo[seat] = _require_bool(can_tsumo, name="can_tsumo")
        if can_riichi is not None:
            self._can_riichi[seat] = _require_bool(can_riichi, name="can_riichi")
        if pending_declaration_discard is not None:
            tile_id: TileId = _bridge_contracts.make_tile_id(pending_declaration_discard)
            self._pending_discard[seat] = int(tile_id)

    def update_public_state(self, **snapshot: object) -> None:
        """Supply the authoritative public scalar state (closed vocabulary)."""
        _check_public_unknown(snapshot)
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
                validator=lambda v, name: TileType(  # pyrefly: ignore[unknown-argument-type]  # reason: snapshot statically object; range-validated to 27..33 below
                    _require_plain_int(v, name=name, minimum=27, maximum=33)  # pyrefly: ignore[unknown-argument-type]  # reason: snapshot statically object; range-validated to 27..33 here
                ),
            )
            if sorted(int(w) for w in winds) != list(_WIND_TILE_TYPES):  # pyrefly: ignore[unknown-argument-type]  # reason: winds quad-validated above; int() coerces each entry
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
            self._public["dealer"] = _bridge_contracts.make_seat(snapshot["dealer"])  # type: ignore[arg-type]  # reason: snapshot value statically object; validated inside make_seat
        if "scores" in snapshot:
            self._public["scores"] = _quad(
                snapshot["scores"],
                name="scores",
                validator=_validate_score,
            )
        if "turn_actor" in snapshot:
            self._public["turn_actor"] = _bridge_contracts.make_seat(snapshot["turn_actor"])  # type: ignore[arg-type]  # reason: snapshot value statically object; validated inside make_seat
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
        seat: Seat = _bridge_contracts.make_seat(int(actor))
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
        _check_public_missing(self._public)
        indicators: tuple[int, ...] = tuple(self._dora) + (DORA_SENTINEL,) * (
            DORA_SHAPE[0] - len(self._dora)
        )
        conceal = self._concealed[seat]
        pending = self._pending_discard[seat]
        filter_seat: Seat = _bridge_contracts.make_seat(seat)
        history = filter_events_for_actor(self._histories[seat], filter_seat)
        pending_tile: TileId | None = (
            None if pending is None else _bridge_contracts.make_tile_id(pending)
        )
        return make_actor_observation(
            game_id=self._game_id,
            decision_id=str(self._public["decision_id"]),
            sequence=make_sequence_no(history[-1].sequence if len(history) > 0 else 0),
            actor=filter_seat,
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
            pending_declaration_discard=pending_tile,
            concealed_hand=conceal if conceal is not None else (),
            own_drawn_tile=(
                None
                if self._drawn[seat] is None
                else _bridge_contracts.make_tile_id(self._drawn[seat])  # type: ignore[arg-type]  # reason: None filtered by ternary; range validated inside make_tile_id
            ),
            visible_discards=tuple(tuple(river) for river in self._discards),
            visible_melds=tuple(tuple(row) for row in self._melds),
            riichi_states=tuple(self._riichi_states),
            dora_indicators=indicators,
            visible_history=tuple(history),
            legal_mask=tuple(flags),
        )
