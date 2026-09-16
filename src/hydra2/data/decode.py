"""One-game-per-object JSONL decode — checklist item 3.

Hard dependency (shrink end-state): :func:`decode_game_object` is a thin
delegate over the Rust bridge (``hydra2_replay_rs.packet_decode``
``decode_frames_batch`` as a single-item batch; ``hydra-feed`` decode owns
the gates: utf-8, trailing newline, blank-line, JSON, shape, start/end,
game-id fallback, wall extraction, raw sha). ``ImportError`` (extension not
built) raises with a ``build-ext`` hint — NO oracle fallback, never silent.
Stems are parallel params (``stem_of`` per file, never synthesized here).
Evidence: packet 283/283 + raw sha every game; seals migrated (B3).

:class:`GameRecord` stays the thin transport type (fields are bridge-echoed
identity plus transport-materialized events). :func:`decode_json_line`
stays transport-only JSON parse (no bridge line-parse pyfn exists).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from hydra2.contracts.common import ContractError, CorruptArtifactError

__all__ = ["GameRecord", "decode_game_object", "decode_json_line"]


def _require_packet_decode() -> Any:
    """Import the built ``packet_decode`` bridge surface (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with packet_decode not importable; "
            "build the bridge with `pixi run build-ext` before decoding games"
        ) from exc
    try:
        return _ext.packet_decode
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.packet_decode submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def decode_json_line(line: str | bytes) -> object:
    """Parse one JSON line (transport-only; no bridge line-parse pyfn)."""
    return json.loads(line)


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_id: str
    object_id: str
    packaged_object_id: str
    events: tuple[dict[str, object], ...]
    raw_bytes_sha256: str
    wall_tiles: tuple[int, ...] | None
    source: dict[str, object]


def decode_game_object(
    *, object_id: str, packaged_object_id: str, decoded_bytes: bytes
) -> GameRecord:
    """Decode exactly one game per object via the Rust bridge (fail closed).

    Rules (``hydra-feed`` decode owns them; rejects raise, never skip):
    utf-8 payload, trailing newline, no blank lines, each line a JSON
    object with a ``type`` field, exactly one start/end boundary with no
    trailing data. ``blank_line`` maps to :class:`CorruptArtifactError`
    (oracle parity); every other reject maps to :class:`ContractError`.
    """
    if not isinstance(decoded_bytes, (bytes, bytearray)):
        raise ContractError(
            f"decoded payload for {object_id} must be bytes, got {type(decoded_bytes).__name__}"
        )
    raw = bytes(decoded_bytes)
    packet_decode = _require_packet_decode()
    try:
        slots = packet_decode.decode_frames_batch([raw], [(object_id, packaged_object_id)])
    except ValueError as exc:
        raise ContractError(f"decode rejected for {object_id}: {exc}") from exc
    slot = slots[0]
    if not slot["ok"]:
        error_class = slot.get("error_class")
        event_index = slot.get("event_index")
        detail = slot.get("detail")
        message = f"decode {error_class} for {object_id} (event {event_index}): {detail}"
        if error_class == "blank_line":
            raise CorruptArtifactError(message)
        raise ContractError(message)
    # Transport materialization for the thin type (gate already passed, so
    # utf-8/split cannot fail; dict shape re-checked fail-closed).
    parsed: list[dict[str, object]] = []
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"decode shape reject for {object_id}: line must be object")
        parsed.append(cast("dict[str, object]", value))
    if len(parsed) == 0:
        raise ContractError(f"empty payload for {object_id}: no game records")
    first = parsed[0]
    first_type = first.get("type")
    source: dict[str, object] = {"type": first_type} if isinstance(first_type, str) else {}
    wall_raw = slot.get("wall_tiles")
    wall = tuple(int(x) for x in wall_raw) if wall_raw is not None else None
    return GameRecord(
        game_id=str(slot["game_id"]),
        object_id=str(slot["object_id"]),
        packaged_object_id=str(slot["packaged_object_id"]),
        events=tuple(parsed),
        raw_bytes_sha256=str(slot["raw_bytes_sha256"]),
        wall_tiles=wall,
        source=source,
    )
