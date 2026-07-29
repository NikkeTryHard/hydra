"""One-game-per-object JSONL decode — checklist item 3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hydra2.contracts.common import ContractError, CorruptArtifactError

__all__ = ["GameRecord", "decode_game_object"]


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_id: str
    object_id: str
    packaged_object_id: str
    events: tuple[dict[str, object], ...]
    raw_bytes_sha256: str
    wall_tiles: tuple[int, ...] | None
    source: dict[str, object]


def _game_id_from_events(events: tuple[dict[str, object], ...], object_id: str) -> str:
    # Prefer explicit game_id field, else derive from object_id
    for ev in events:
        gid = ev.get("game_id")
        if isinstance(gid, str) and gid != "":
            return gid
        gid2 = ev.get("gameId")
        if isinstance(gid2, str) and gid2 != "":
            return gid2
    # Synthetic fallback: hash object_id
    return "game-" + hashlib.sha256(object_id.encode()).hexdigest()[:12]


def decode_game_object(
    *, object_id: str, packaged_object_id: str, decoded_bytes: bytes
) -> GameRecord:
    """Decode exactly one game per object; rejects partial/trailing/blank.

    Rules (mirrors packager canonical_jsonl but stricter for games):
      - decoded_bytes must be utf-8
      - split by newline, no blank lines allowed (WP-00B canonical_jsonl would be False)
      - each line must be valid JSON object with a 'type' field
      - exactly one 'start_game' / 'startGame' as first record
      - exactly one 'end_game' / 'endGame' as last record
      - no data after end_game (no trailing data)
      - record_count >=2
    """
    try:
        text = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"decoded bytes not utf-8 for {object_id}: {exc}") from exc
    if not text.endswith("\n"):
        raise ContractError(
            f"decoded payload for {object_id} must end with newline (missing trailing newline)"
        )
    # Check for trailing data after end_game is handled after parsing
    lines = text.splitlines()
    if len(lines) == 0:
        raise ContractError(f"empty payload for {object_id}: no game records")
    # Reject blank lines anywhere: they would be silent skip risk
    for idx, line in enumerate(lines):
        if line.strip() == "":
            raise CorruptArtifactError(
                f"blank line at index {idx} for {object_id}: blank lines forbidden"
            )
    events: list[dict[str, object]] = []
    for idx, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"line {idx} invalid JSON for {object_id}: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(
                f"line {idx} must be JSON object for {object_id}, got {type(value).__name__}"
            )
        if "type" not in value:
            raise ContractError(f"line {idx} missing 'type' field for {object_id}")
        events.append(value)
    # One-game-per-object
    first_type = str(events[0].get("type"))
    last_type = str(events[-1].get("type"))
    if first_type not in ("start_game", "startGame", "game_start", "start"):
        raise ContractError(f"first record must be start_game for {object_id}, got {first_type!r}")
    if last_type not in ("end_game", "endGame", "game_end", "end"):
        raise ContractError(f"last record must be end_game for {object_id}, got {last_type!r}")
    start_count = sum(
        1 for e in events if str(e.get("type")) in ("start_game", "startGame", "game_start")
    )
    end_count = sum(1 for e in events if str(e.get("type")) in ("end_game", "endGame", "game_end"))
    if start_count != 1 or end_count != 1:
        raise ContractError(
            "exactly one start_game and one end_game required for "
            f"{object_id}, got {start_count}/{end_count}"
        )
    # No trailing data already ensured: last is end_game; also ensure no extra bytes beyond newline
    # Already checked text ends with newline and splitlines removed it.
    game_id = _game_id_from_events(tuple(events), object_id)
    # Optional wall extraction
    wall: tuple[int, ...] | None = None
    for ev in events:
        _w = ev.get("wall")
        if _w is None:
            _w = ev.get("wall_tiles")
        if _w is None:
            _w = ev.get("tiles")
        w = _w
        if isinstance(w, list) and len(w) == 136 and all(isinstance(x, int) for x in w):
            wall = tuple(int(x) for x in w)  # type: ignore[arg-type]
            break
    raw_sha = "sha256:" + hashlib.sha256(decoded_bytes).hexdigest()
    return GameRecord(
        game_id=game_id,
        object_id=object_id,
        packaged_object_id=packaged_object_id,
        events=tuple(events),
        raw_bytes_sha256=raw_sha,
        wall_tiles=wall,
        source={"type": first_type},
    )
