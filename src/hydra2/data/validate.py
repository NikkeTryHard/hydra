"""Game validation — checklist item 4.

Hard dependency (shrink end-state): :func:`validate_game` is a thin
delegate over the Rust bridge (``hydra2_replay_rs.packet_decode``
``decode_frames_batch`` for the opaque handle, then ``validate_batch``;
``hydra-feed`` validate owns the 9-check pipeline plus the
``sha256(canonical({game_id, checks}))`` seal). ``ImportError`` (extension
not built) raises with a ``build-ext`` hint — NO oracle fallback, never
silent. Evidence: packet 283/283 + seals migrated (B3). B1/B2 untouched
(no RNG/Gumbel/randperm here).

Trap-8: the ``RiichiEnvExactSimulator.reset`` probe stays Python (engine
slice owns it); Rust takes the ``adapter_ok`` flag and labels the skipped
path ``skipped_adapter_error:NotWired``. :func:`compute_validation_hash`
is a thin delegate over the hard-Rust digest owner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.data.decode import GameRecord

__all__ = [
    "ValidationError",
    "ValidationOutcome",
    "compute_validation_hash",
    "validate_game",
]

#: Red tile physical IDs per SPEC 4.1 / tenhou_4p_hanchan_v1 (aka-dora: ids
#: 16/52/88 are the red fives). Kept as data (no bridge replacement).
RED_TILE_IDS = (16, 52, 88)
#: Logical tile types. Kept as data (no bridge replacement).
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


def _require_packet_decode() -> Any:
    """Import the built ``packet_decode`` bridge surface (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with packet_decode not importable; "
            "build the bridge with `pixi run build-ext` before validating games"
        ) from exc
    try:
        return _ext.packet_decode
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.packet_decode submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def compute_validation_hash(game_id: str, checks: dict[str, str]) -> str:
    """Seal over ``{game_id, checks}`` via the hard-Rust digest owner."""
    return str(of_canonical({"game_id": game_id, "checks": dict(checks)}))


def _adapter_ok(record: GameRecord) -> bool:
    """Trap-8 sim verdict for the Rust ``adapter_ok`` flag (stays Python).

    Only consulted on the wall+actions path; every other path ignores the
    probe. ``True`` = full ``legality|calls|scores|termination = ok``;
    ``False`` pins the bridge ``NotWired`` skipped label.
    """
    if record.wall_tiles is None:
        return True
    if not any(
        isinstance(ev, dict) and ("action_id" in ev or "action" in ev) for ev in record.events
    ):
        return True
    try:
        from hydra2.contracts.common import Seat, TileId
        from hydra2.contracts.rules_manifest import rules_manifest_from_payload
        from hydra2.engines.protocol import WallSchedule, wall_schedule_digest
        from hydra2.engines.riichienv.adapter_core import RiichiEnvExactSimulator

        wall_tiles = tuple(TileId(int(t)) for t in record.wall_tiles)
        wall_sched = WallSchedule(
            schedule_id=f"wp04b-{record.game_id}",
            physical_tiles=wall_tiles,
            digest=wall_schedule_digest(f"wp04b-{record.game_id}", wall_tiles),
        )
        from hydra2.config import repo_root as _validate_repo_root

        rules_path = _validate_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
        if not rules_path.is_file():
            try:
                import importlib.resources as _ir2

                rules_path = Path(
                    str(_ir2.files("hydra2") / "configs" / "rules" / "tenhou_4p_hanchan_v1.json")
                )
            except Exception:  # why-broad: resource probe failed; retry repo
                rules_path = (
                    _validate_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
                )
        rules_doc_obj: object = json.loads(rules_path.read_bytes())
        if not isinstance(rules_doc_obj, dict):
            return False
        payload_raw: object = rules_doc_obj.get("payload", rules_doc_obj)
        if not isinstance(payload_raw, dict):
            return False
        rules = rules_manifest_from_payload(payload_raw)
        sim = RiichiEnvExactSimulator()
        sim.reset(
            rules=rules,
            wall=wall_sched,
            seat_permutation=(Seat(0), Seat(1), Seat(2), Seat(3)),
        )
        return True
    except Exception:  # why-broad: adapter probe may fail anywhere -> skipped
        return False


def _invalid(
    record: GameRecord, error_class: str, event_index: int | None, message: str
) -> ValidationOutcome:
    return ValidationOutcome(
        game_id=record.game_id,
        object_id=record.object_id,
        valid=False,
        error=ValidationError(error_class, event_index, message),
        validation_hash=None,
        checks={},
    )


def validate_game(record: GameRecord) -> ValidationOutcome:
    """Validate one decoded game via the Rust bridge (fail closed on content).

    Content rejects arrive as invalid outcomes (never raises); only a
    malformed bridge call itself raises :class:`ContractError`.
    """
    packet_decode = _require_packet_decode()
    try:
        payload = (
            "\n".join(
                json.dumps(ev, separators=(",", ":"), ensure_ascii=False) for ev in record.events
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return _invalid(record, "structure", None, f"unserializable event: {exc}")
    try:
        slots = packet_decode.decode_frames_batch(
            [payload], [(record.object_id, record.packaged_object_id)]
        )
    except ValueError as exc:
        return _invalid(record, "structure", None, f"decode rejected: {exc}")
    slot = slots[0]
    if not slot["ok"]:
        return _invalid(
            record,
            "structure",
            slot.get("event_index"),
            str(slot.get("detail")),
        )
    try:
        outs = packet_decode.validate_batch([slot["record"]], _adapter_ok(record))
    except ValueError as exc:
        raise ContractError(f"validate rejected for {record.object_id}: {exc}") from exc
    out = outs[0]
    error = None
    if not out["valid"]:
        raw_index = out.get("event_index")
        error = ValidationError(
            str(out.get("error_class") or "structure"),
            int(raw_index) if raw_index is not None else None,
            str(out.get("message") or "invalid"),
        )
    raw_hash = out.get("validation_hash")
    checks = {str(k): str(v) for k, v in dict(out.get("checks") or {}).items()}
    return ValidationOutcome(
        game_id=str(out["game_id"]),
        object_id=str(out["object_id"]),
        valid=bool(out["valid"]),
        error=error,
        validation_hash=str(raw_hash) if raw_hash is not None else None,
        checks=checks,
    )
