"""Game validation — checklist item 4.

Hard dependency (shrink end-state): :func:`validate_game` is a thin
delegate over the Rust bridge (``hydra2._native.packet_decode``
``decode_frames_batch`` for the opaque handle, then ``validate_batch``;
``hydra-feed`` validate owns the 9-check pipeline plus the
``sha256(canonical({game_id, checks}))`` seal). ``ImportError`` (extension
not built) raises with a ``build-ext`` hint — NO oracle fallback, never
silent. Evidence: packet 283/283 + seals migrated (B3). B1/B2 untouched
(no RNG/Gumbel/randperm here).

Trap-8: the ``RiichiEnvExactSimulator.reset`` probe stays Python (engine
slice owns it); Rust takes the ``adapter_ok`` flag and labels the skipped
path ``skipped_adapter_error:NotWired``. :func:`compute_validation_hash`
is a thin delegate over the ``hydra2._native.contracts`` ``validation_hash``
leaf (``feed::digest`` owner underneath; taxonomy consts live on the same
submodule as ``VALIDATION_*``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from collections.abc import Callable

    from hydra2.data.decode import GameRecord

__all__ = [
    "ValidationError",
    "ValidationOutcome",
    "compute_validation_hash",
    "validate_game",
]


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
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with packet_decode not importable; "
            "build the bridge with `pixi run build-ext` before validating games"
        ) from exc
    try:
        return _ext.packet_decode
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.packet_decode submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _require_contracts() -> Any:
    """Import the built ``contracts`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "build the bridge with `pixi run build-ext` before sealing games"
        ) from exc
    try:
        return _ext.contracts
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.contracts submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def compute_validation_hash(game_id: str, checks: dict[str, str]) -> str:
    """Seal over ``{game_id, checks}`` via the bridge ``validation_hash`` leaf."""
    try:
        leaf: Callable[[str, dict[str, str]], str] = _require_contracts().validation_hash
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.contracts.validation_hash missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    digest_text: str = leaf(game_id, dict(checks))
    return digest_text


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

        wall_tiles = tuple(TileId(t) for t in record.wall_tiles)
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
        slots: list[dict[str, Any]] = packet_decode.decode_frames_batch(
            [payload], [(record.object_id, record.packaged_object_id)]
        )
    except ValueError as exc:
        return _invalid(record, "structure", None, f"decode rejected: {exc}")
    slot: dict[str, Any] = slots[0]
    if not slot["ok"]:
        event_index: int | None = slot.get("event_index")
        detail: object = slot.get("detail")
        return _invalid(
            record,
            "structure",
            event_index,
            str(detail),
        )
    try:
        record_obj: object = slot["record"]
        record_list: list[object] = [record_obj]
        outs: list[dict[str, Any]] = packet_decode.validate_batch(record_list, _adapter_ok(record))
    except ValueError as exc:
        raise ContractError(f"validate rejected for {record.object_id}: {exc}") from exc
    out: dict[str, Any] = outs[0]
    error = None
    if not out["valid"]:
        raw_index: int | None = out.get("event_index")
        error_class_raw: object = out.get("error_class")
        message_raw: object = out.get("message")
        error_text: str = (
            error_class_raw
            if isinstance(error_class_raw, str) and error_class_raw != ""
            else "structure"
        )
        message_text: str = (
            message_raw if isinstance(message_raw, str) and message_raw != "" else "invalid"
        )
        error = ValidationError(
            error_text,
            raw_index,
            message_text,
        )
    raw_hash: str | None = out.get("validation_hash")
    checks_raw: object = out.get("checks")
    checks_dict: dict[object, object] = (
        cast("dict[object, object]", checks_raw) if isinstance(checks_raw, dict) else {}
    )
    checks = {str(k): str(v) for k, v in checks_dict.items()}
    out_game_id: str = out["game_id"]
    out_object_id: str = out["object_id"]
    out_valid: bool = out["valid"]
    return ValidationOutcome(
        game_id=out_game_id,
        object_id=out_object_id,
        valid=out_valid,
        error=error,
        validation_hash=raw_hash,
        checks=checks,
    )
