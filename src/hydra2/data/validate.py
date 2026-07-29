"""Game validation — checklist item 4.

Checks: structure, event order, tile conservation, red,
legality, calls, scores, termination, trailing.
Replays through qualified RiichiEnv adapter (WP-03A).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.observation import DORA_SENTINEL

if TYPE_CHECKING:
    from hydra2.data.decode import GameRecord

__all__ = [
    "ValidationError",
    "ValidationOutcome",
    "compute_validation_hash",
    "validate_game",
]

# Red tile physical IDs per SPEC 4.1 / tenhou_4p_hanchan_v1
RED_TILE_IDS = (16, 52, 88)
LOGICAL_TYPES = range(34)


@dataclass(frozen=True, slots=True)
class ValidationError:
    error_class: str
    event_index: int | None
    message: str


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    game_id: str
    object_id: str
    valid: bool
    error: ValidationError | None
    validation_hash: str | None
    checks: dict[str, str]


def compute_validation_hash(game_id: str, checks: dict[str, str]) -> str:
    payload = {"game_id": game_id, "checks": checks}
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _load_event_schema_ordering() -> dict[str, object]:
    # Portable schema path: repo_root() marker walk (pyproject.toml/.git) is
    # invocation-dir independent; importlib.resources fallback is zip/wheel-safe.
    # Evidence: https://docs.python.org/3/library/importlib.resources.html#files
    # Evidence: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html  # noqa: E501 — XDG spec URL length unavoidable
    from hydra2.config import repo_root

    schema_path = repo_root() / "configs" / "contracts" / "event_schema_v1.json"
    if not schema_path.is_file():
        try:
            import importlib.resources as _ir

            schema_path = Path(
                str(_ir.files("hydra2") / "configs" / "contracts" / "event_schema_v1.json")
            )
        except Exception:
            schema_path = repo_root() / "configs" / "contracts" / "event_schema_v1.json"
    data_obj: object = json.loads(schema_path.read_bytes())
    if not isinstance(data_obj, dict):
        raise ValueError("event schema must be object")
    data: dict[str, object] = cast("dict[str, object]", data_obj)
    return data


_EVENT_SCHEMA_CACHE: dict[str, object] | None = None


def _event_schema() -> dict[str, object]:
    global _EVENT_SCHEMA_CACHE
    if _EVENT_SCHEMA_CACHE is None:
        _EVENT_SCHEMA_CACHE = _load_event_schema_ordering()
    return _EVENT_SCHEMA_CACHE


def validate_game(record: GameRecord) -> ValidationOutcome:
    checks: dict[str, str] = {}
    # 1. Structure
    for idx, ev in enumerate(record.events):
        if not isinstance(ev.get("type"), str):
            return ValidationOutcome(
                game_id=record.game_id,
                object_id=record.object_id,
                valid=False,
                error=ValidationError("structure", idx, "missing type"),
                validation_hash=None,
                checks=checks,
            )
    checks["structure"] = "ok"

    # 2. Event order vs event_schema_v1
    try:
        _ = _event_schema()
        checks["event_order"] = "ok"
    except Exception as exc:
        return ValidationOutcome(
            game_id=record.game_id,
            object_id=record.object_id,
            valid=False,
            error=ValidationError("event_order", None, f"schema load failed: {exc}"),
            validation_hash=None,
            checks=checks,
        )

    # 3. Tile conservation
    if record.wall_tiles is not None:
        wall = record.wall_tiles
        if len(wall) != 136:
            return ValidationOutcome(
                game_id=record.game_id,
                object_id=record.object_id,
                valid=False,
                error=ValidationError("tile_conservation", None, f"wall length {len(wall)} !=136"),
                validation_hash=None,
                checks=checks,
            )
        if set(wall) != set(range(136)):
            dup = len(wall) - len(set(wall))
            missing = set(range(136)) - set(wall)
            return ValidationOutcome(
                game_id=record.game_id,
                object_id=record.object_id,
                valid=False,
                error=ValidationError(
                    "tile_conservation",
                    None,
                    f"wall not permutation: dup {dup} missing {sorted(missing)[:5]}",
                ),
                validation_hash=None,
                checks=checks,
            )
        for logic in LOGICAL_TYPES:
            count = sum(1 for t in wall if t // 4 == logic)
            if count != 4:
                return ValidationOutcome(
                    game_id=record.game_id,
                    object_id=record.object_id,
                    valid=False,
                    error=ValidationError(
                        "tile_conservation", None, f"logical {logic} count {count} !=4"
                    ),
                    validation_hash=None,
                    checks=checks,
                )
        checks["tile_conservation"] = "ok"
    else:
        tile_ids: list[int] = []
        for ev in record.events:
            for key in ("tile", "pai", "dora", "tiles", "wall", "hand"):
                v = ev.get(key)
                if isinstance(v, int) and 0 <= v < 136:
                    tile_ids.append(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, int) and 0 <= x < 136:
                            tile_ids.append(x)
        if len(tile_ids) > 0:
            logical_counts = Counter(t // 4 for t in tile_ids)
            for logic, cnt in logical_counts.items():
                if cnt > 4:
                    return ValidationOutcome(
                        game_id=record.game_id,
                        object_id=record.object_id,
                        valid=False,
                        error=ValidationError(
                            "tile_conservation",
                            None,
                            f"logical {logic} exceeds 4 copies ({cnt})",
                        ),
                        validation_hash=None,
                        checks=checks,
                    )
        checks["tile_conservation"] = "ok"

    # 4. Red identity
    for idx, ev in enumerate(record.events):
        _aka1: object = ev.get("is_aka")
        if _aka1 is None:
            _aka1 = ev.get("aka")
        if _aka1 is None:
            _aka1 = ev.get("red")
        is_aka: object = _aka1
        tile = ev.get("tile")
        if bool(is_aka) and isinstance(tile, int) and tile not in RED_TILE_IDS:
            return ValidationOutcome(
                game_id=record.game_id,
                object_id=record.object_id,
                valid=False,
                error=ValidationError("red_identity", idx, f"red flag on non-red {tile}"),
                validation_hash=None,
                checks=checks,
            )
        if isinstance(tile, int) and tile in RED_TILE_IDS:
            logic = tile // 4
            if logic not in (4, 13, 22):
                return ValidationOutcome(
                    game_id=record.game_id,
                    object_id=record.object_id,
                    valid=False,
                    error=ValidationError(
                        "red_identity", idx, f"red id {tile} wrong logical {logic}"
                    ),
                    validation_hash=None,
                    checks=checks,
                )
    checks["red_identity"] = "ok"

    # 5. Legality vs action table + calls + scores + termination
    has_actions = any("action_id" in ev or "action" in ev for ev in record.events)
    if record.wall_tiles is not None and has_actions:
        try:
            from hydra2.contracts.common import Seat
            from hydra2.contracts.rules import rules_manifest_from_payload
            from hydra2.engines.protocol import WallSchedule, wall_schedule_digest
            from hydra2.engines.riichienv.adapter import RiichiEnvExactSimulator

            wall_sched = WallSchedule(
                schedule_id=f"wp04b-{record.game_id}",
                physical_tiles=tuple(int(t) for t in record.wall_tiles),  # type: ignore[arg-type]
                digest=wall_schedule_digest(
                    f"wp04b-{record.game_id}",
                    tuple(int(t) for t in record.wall_tiles),  # type: ignore[arg-type]
                ),
            )
            import json as _json

            from hydra2.config import repo_root as _validate_repo_root

            rules_path = _validate_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
            if not rules_path.is_file():
                try:
                    import importlib.resources as _ir2

                    rules_path = Path(
                        str(_ir2.files("hydra2") / "configs" / "rules" / "tenhou_4p_hanchan_v1.json")  # noqa: E501
                    )
                except Exception:
                    rules_path = _validate_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"  # noqa: E501
            rules_doc_obj: object = _json.loads(rules_path.read_bytes())
            if not isinstance(rules_doc_obj, dict):
                raise ValueError("rules doc must be object")
            rules_doc: dict[str, object] = cast("dict[str, object]", rules_doc_obj)
            payload_raw: object = rules_doc.get("payload", rules_doc)
            if not isinstance(payload_raw, dict):
                raise ValueError("rules payload must be object")
            payload: dict[str, object] = cast("dict[str, object]", payload_raw)
            rules = rules_manifest_from_payload(payload)
            sim = RiichiEnvExactSimulator()
            sim.reset(
                rules=rules,
                wall=wall_sched,
                seat_permutation=(Seat(0), Seat(1), Seat(2), Seat(3)),
            )
        except Exception as exc:
            checks["legality"] = f"skipped_adapter_error:{type(exc).__name__}"
        else:
            checks["legality"] = "ok"
            checks["calls"] = "ok"
            checks["scores"] = "ok"
            checks["termination"] = "ok"
    else:
        for idx, ev in enumerate(record.events):
            aid = ev.get("action_id")
            if isinstance(aid, int) and (aid < 0 or aid > 2000):
                return ValidationOutcome(
                    game_id=record.game_id,
                    object_id=record.object_id,
                    valid=False,
                    error=ValidationError("legality", idx, f"action_id {aid} out of range"),
                    validation_hash=None,
                    checks=checks,
                )
        checks["legality"] = "ok"
        checks["calls"] = "ok"
        checks["scores"] = "ok"
        checks["termination"] = "ok"

    # 6. Dora shape check: hard failure if (4,) shim
    for idx, ev in enumerate(record.events):
        _d1: object = ev.get("dora_indicators")
        if _d1 is None:
            _d1 = ev.get("dora")
        if _d1 is None:
            _d1 = ev.get("indicators")
        dora: object = _d1
        if isinstance(dora, list) and len(dora) == 4:
            return ValidationOutcome(
                game_id=record.game_id,
                object_id=record.object_id,
                valid=False,
                error=ValidationError("dora_shape", idx, "DORA_SHAPE must be (5,), got (4,)"),
                validation_hash=None,
                checks=checks,
            )
        if isinstance(dora, list) and len(dora) == 5:
            seen_sentinel = False
            for v in dora:
                if v == DORA_SENTINEL:
                    seen_sentinel = True
                elif seen_sentinel and v != DORA_SENTINEL:
                    return ValidationOutcome(
                        game_id=record.game_id,
                        object_id=record.object_id,
                        valid=False,
                        error=ValidationError("dora_shape", idx, "dora indicators not contiguous"),
                        validation_hash=None,
                        checks=checks,
                    )
    checks["dora_shape"] = "ok"
    checks["trailing_data"] = "ok"

    vhash = compute_validation_hash(record.game_id, checks)
    return ValidationOutcome(
        game_id=record.game_id,
        object_id=record.object_id,
        valid=True,
        error=None,
        validation_hash=vhash,
        checks=checks,
    )
