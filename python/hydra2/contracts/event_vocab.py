"""SPEC 7.1 event vocabulary — frozen literal sets and delta-value rules.

The closed inventories live in the ``hydra2._native.contracts`` Rust surface
(single source); this module keeps thin re-exports plus its ``Literal`` types,
delta-value rules, and the small field validators shared by the envelope and
schema modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import (
    ActionId,
    ContractError,
    Seat,
    TileId,
)

__all__ = [
    "DELTA_OPERATIONS",
    "DELTA_PATH_VOCABULARY",
    "EVENT_KINDS",
    "PAYLOAD_SCALAR_FIELDS",
    "PAYLOAD_TUPLE_FIELDS",
    "VISIBILITIES",
    "DeltaOperation",
    "EventKind",
    "Visibility",
]

Visibility = Literal["public", "actor_private", "server_private"]
VISIBILITIES: tuple[str, ...] = _bridge_contracts.VISIBILITIES

#: Every EventKind literal in SPEC declaration order; frozen vocabulary v1.
EventKind = Literal[
    "game_start",
    "round_start",
    "turn_advance",
    "draw_tile",
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "call_window",
    "call_resolved",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
    "draw_end",
    "abortive_draw",
    "round_end",
    "game_end",
]
EVENT_KINDS: tuple[str, ...] = _bridge_contracts.EVENT_KINDS

DeltaOperation = Literal["set", "append", "increment"]
DELTA_OPERATIONS: tuple[str, ...] = _bridge_contracts.DELTA_OPERATIONS
_PUBLIC_VISIBLE_TO = (0, 1, 2, 3)
#: Ceiling of the bridge action-id domain (``make_action_id`` returns u32).
#: Python ints above this would wrap modulo 2**32 inside the bridge instead of
#: rejecting, so the validators below fail closed here and preserve the oracle
#: accept-set (the oracle accepted arbitrary-precision ints and rejected the
#: huge ones downstream; wrapping would silently alias them onto live ids).
_ACTION_ID_MAX = 0xFFFF_FFFF

#: Scalar payload fields besides ``kind``/``actor`` (closed inventory, re-exported).
PAYLOAD_SCALAR_FIELDS: tuple[str, ...] = _bridge_contracts.PAYLOAD_SCALAR_FIELDS
#: Tuple payload fields (closed inventory, re-exported).
PAYLOAD_TUPLE_FIELDS: tuple[str, ...] = _bridge_contracts.PAYLOAD_TUPLE_FIELDS


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_enum(value: object, *, name: str, allowed: tuple[str, ...] | frozenset[str]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {sorted(allowed)}")
    return text


def _require_plain_int(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name}={value} outside [{minimum}, {maximum}]")
    return value


def _optional_seat(value: object, *, name: str) -> Seat | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be a seat int 0..3, got {type(value).__name__}")
    return _bridge_contracts.make_seat(value)


def _optional_tile(value: object, *, name: str) -> TileId | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be a tile id int 0..135, got {type(value).__name__}")
    return _bridge_contracts.make_tile_id(value)


def _optional_action(value: object, *, name: str) -> ActionId | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{name} must be an action id nonnegative int, got {type(value).__name__}"
        )
    if value > _ACTION_ID_MAX:
        raise ContractError(f"{name}={value} exceeds u32 action id domain")
    return _bridge_contracts.make_action_id(value)


def _tile_tuple(values: Sequence[int], *, name: str) -> tuple[TileId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of tile ids")
    return tuple(_bridge_contracts.make_tile_id(v) for v in values)


def _action_tuple(values: Sequence[int], *, name: str) -> tuple[ActionId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of action ids")
    for element in values:
        if isinstance(element, int) and not isinstance(element, bool) and element > _ACTION_ID_MAX:
            raise ContractError(f"{name} element {element!r} exceeds u32 action id domain")
    return tuple(_bridge_contracts.make_action_id(v) for v in values)


def _score_quad(values: object, *, name: str) -> tuple[int, int, int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
        raise ContractError(f"{name} must be exactly four score ints")
    scores = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{name}[{index}] must be an int")
        scores.append(value)
    first, second, third, fourth = scores
    return (first, second, third, fourth)


def _validate_json_value(value: object) -> None:
    """Restrict delta values to the canonical JSON domain (SPEC 2.2)."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if abs(value) >= 2**53:
            raise ContractError("delta int value exceeds the canonical safe range")
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractError("delta float value must be finite")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("delta object keys must be strings")
        for item in value.values():
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    raise ContractError(f"delta value outside JSON domain: {type(value).__name__}")


# ---------------------------------------------------------------------------
# PublicStateDelta paths: closed vocabulary with typed operations/values.
# ---------------------------------------------------------------------------

_DELTA_ROOTS: frozenset[str] = frozenset(
    {
        "scores",
        "honba",
        "riichi_sticks",
        "round_index",
        "dora_indicators",
        "melds",
        "riichi_states",
        "ippatsu",
        "kan_count",
        "live_wall_tiles_remaining",  # tiles left in the live wall (wall = 136-tile stack).
    }
)
_PATH_ACTOR_PLACEHOLDER = "actor"
_RIICHI_STATE_VALUES = ("none", "declared", "accepted")

_MELD_OBJECT_KEYS = ("meld_id", "kind", "owner", "source_seat", "called_tile", "tiles")


def _validate_delta_path_element(element: object, *, depth: int) -> str | int:
    if isinstance(element, bool):
        raise ContractError("delta path elements must be str or int, not bool")
    if isinstance(element, str):
        if element == "":
            raise ContractError("delta path string elements must be non-empty")
        if depth > 0 and element != _PATH_ACTOR_PLACEHOLDER:
            raise ContractError(f"delta path element {element!r} invalid at position {depth}")
        return element
    if isinstance(element, int):
        return _require_plain_int(element, name="delta path seat", minimum=0, maximum=3)
    raise ContractError(f"delta path element must be str or int: {type(element).__name__}")


def _validate_delta_path(path: Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(path, str) or not isinstance(path, Sequence) or len(path) == 0:
        raise ContractError("delta path must be a non-empty sequence of str|int")
    if len(path) > 2:
        raise ContractError("delta paths are at most two elements deep in schema v1")
    root = path[0]
    if not isinstance(root, str) or root not in _DELTA_ROOTS:
        raise ContractError(f"delta path root {root!r} outside the closed vocabulary")
    validated: list[str | int] = [root]
    for depth, element in enumerate(path[1:], start=1):
        validated.append(_validate_delta_path_element(element, depth=depth))
    second = validated[1] if len(validated) > 1 else None
    if root in ("melds", "riichi_states", "ippatsu"):
        if second is None or not isinstance(second, (str, int)):
            raise ContractError(f"delta path {root!r} requires a seat element")
    elif root == "scores":
        # Both ("scores",) quad-set and ("scores", seat) int deltas are
        # published in DELTA_PATH_VOCABULARY; any second element must be a
        # validated seat.
        if second is None:
            pass
        elif not isinstance(second, int):
            raise ContractError("delta path scores second element must be a seat 0..3")
    elif second is not None:
        raise ContractError(f"delta path root {root!r} takes no second element")
    return tuple(validated)


def _validate_delta_value(path: tuple[str | int, ...], operation: str, value: object) -> None:
    """Closed (path, operation) -> value-type rules; undeclared pairs rejected."""
    _validate_json_value(value)
    root = path[0]
    seated_second = path[1] if len(path) > 1 else None

    def _plain_int(candidate: object, name: str) -> None:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ContractError(f"delta {name} requires an int value")

    def _tile_value(candidate: object) -> None:
        _plain_int(candidate, "dora_indicators.append")
        _bridge_contracts.make_tile_id(candidate)  # type: ignore[arg-type]  # reason: _plain_int bool-checks candidate above; range validated inside make_tile_id

    if root == "scores":
        if seated_second is None:
            if operation != "set":
                raise ContractError("delta scores supports operation 'set' only")
            _ = _score_quad(value, name="scores.set")
            return
        _plain_int(value, "scores[..]")
        return
    if root in ("honba", "riichi_sticks", "kan_count", "live_wall_tiles_remaining"):
        _plain_int(value, str(root))
        return
    if root == "round_index":
        if operation != "set":
            raise ContractError("delta round_index supports operation 'set' only")
        _plain_int(value, "round_index.set")
        return
    # Dora = bonus-indicator tiles; each append reveals one indicator TileId.
    if root == "dora_indicators":
        if operation != "append":
            raise ContractError("delta dora_indicators supports operation 'append' only")
        _tile_value(value)
        return
    if root == "melds":
        if operation != "append":
            raise ContractError("delta melds supports operation 'append' only")
        if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(
            sorted(_MELD_OBJECT_KEYS)
        ):
            raise ContractError("melds.append requires the canonical meld object shape")
        _ = _require_str(value["meld_id"], name="meld.meld_id")  # pyrefly: ignore[unknown-argument-type]  # reason: Mapping shape-checked above; field validators narrow each field
        meld_kind: object = value["kind"]
        _ = _require_enum(
            meld_kind,
            name="meld.kind",
            allowed=("chi", "pon", "daiminkan", "ankan", "kakan"),
        )
        _bridge_contracts.make_seat(value["owner"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_seat
        if value["source_seat"] is not None:
            _bridge_contracts.make_seat(value["source_seat"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_seat
        if value["called_tile"] is not None:
            _bridge_contracts.make_tile_id(value["called_tile"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_tile_id
        _ = _tile_tuple(value["tiles"], name="meld.tiles")  # pyrefly: ignore[unknown-argument-type]  # reason: Mapping shape-checked above; field validators narrow each field
        return
    if root == "riichi_states":
        if operation != "set":
            raise ContractError("delta riichi_states supports operation 'set' only")
        _ = _require_enum(value, name="riichi_states.set", allowed=_RIICHI_STATE_VALUES)
        del seated_second
        return
    if root == "ippatsu":
        if operation != "set":
            raise ContractError("delta ippatsu supports operation 'set' only")
        _ = _require_bool(value, name="ippatsu.set")
        return
    raise ContractError(f"delta path root {root!r} has no value rules")


_ENVELOPE_JSON_FIELDS = (
    "artifact_type",
    "schema_version",
    "compatibility",
    "payload",
)


def _reject_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


#: Serialized vocabulary published inside event_schema_v1.json: every allowed
#: (path, operation, value_type) triple. ``path`` elements: str root, then a
#: seat int 0..3 or the "actor" placeholder where applicable.
DELTA_PATH_VOCABULARY: tuple[dict[str, object], ...] = _bridge_contracts.DELTA_PATH_VOCABULARY
