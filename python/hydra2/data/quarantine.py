"""Quarantine invalid records with reason and lineage — checklist item 5.

Thin delegates over the Rust bridge (``hydra2._native.columnar``
``quarantine_project_invalid`` + ``quarantine_manifest_frame``; feed-owned
canon/digest math): :func:`quarantine_invalid` projects invalid outcomes to
:class:`QuarantinedRecord` slots, :func:`write_quarantine_manifest` frames
the sorted digest-bearing envelope. ``ImportError`` (extension not built)
raises with a ``build-ext`` hint — NO oracle fallback, never silent.
:class:`QuarantinedRecord` / :class:`QuarantineManifest` stay the thin
transport types, and the atomic publish stays here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.data.decode import GameRecord
    from hydra2.data.rows import RawObjectRow
    from hydra2.data.validate import ValidationOutcome


def _require_columnar() -> Any:
    """Import the built ``columnar`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with columnar not importable; "
            "build the bridge with `pixi run build-ext` before quarantining invalid games"
        ) from exc
    try:
        sub = _ext.columnar
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.columnar submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    for fn_name in ("quarantine_project_invalid", "quarantine_manifest_frame"):
        if getattr(sub, fn_name, None) is None:
            raise ImportError(
                f"hydra2._native.columnar.{fn_name} missing (stale .so); "
                "rebuild the bridge with `pixi run build-ext`"
            )
    return sub


#: Missing-outcome dispatch: the bridge owns the no-silent-skip gate and
#: reports it as ``ValueError``; only this exact text re-raises as the oracle
#: ``RuntimeError`` — every other bridge reject maps to :class:`ContractError`.
_MISSING_OUTCOME_PREFIX = "missing validation outcome for "


def _lineage_json(lineage: dict[str, object]) -> bytes:
    """Serialize one lineage dict for the manifest-frame bridge (keys gated pre-cross).

    JSON objects only carry string keys post-transport, so non-``str`` keys
    are rejected here (fail closed) rather than silently coerced by the
    encoder. Key order is irrelevant: Rust re-parses through the feed
    I-JSON boundary and re-canonicalizes.
    """
    for key in lineage:
        if not isinstance(key, str):
            raise ContractError(f"quarantine lineage keys must be str, got {key!r}")
    try:
        return json.dumps(lineage, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"quarantine lineage not JSON-serializable: {exc}") from exc


__all__ = [
    "QuarantineManifest",
    "QuarantinedRecord",
    "quarantine_invalid",
    "write_quarantine_manifest",
]


@dataclass(frozen=True, slots=True)
class QuarantinedRecord:
    object_id: str
    packaged_object_id: str
    game_id: str | None
    error_class: str
    error_event_index: int | None
    lineage: dict[str, object]
    validation_hash: str | None


@dataclass(frozen=True, slots=True)
class QuarantineManifest:
    quarantined: tuple[QuarantinedRecord, ...]
    digest: str


def quarantine_invalid(
    *,
    raw_rows: list[RawObjectRow],
    game_records: dict[str, GameRecord],
    outcomes: dict[str, ValidationOutcome],
) -> list[QuarantinedRecord]:
    """Build quarantine list for invalid games; preserves lineage.

    Thin bridge delegate: ``hydra2._native.columnar.quarantine_project_invalid``
    filters + projects over plain parallel vectors in one detach (missing
    outcome is a hard error — do not silently skip — surfaced here as the
    oracle ``RuntimeError``; every other bridge reject maps to
    :class:`ContractError`), then each invalid slot is wrapped in its
    :class:`QuarantinedRecord` with the 8-key lineage literal below.

    - Every invalid outcome becomes a QuarantinedRecord.
    - Lineage includes packaged_object_id, object_id, source, attestation, parent_ids.
    - No silent skip: every input raw row must appear as either valid or quarantined.
    """
    columnar = _require_columnar()
    object_ids: list[str] = []
    packaged_object_ids: list[str] = []
    confidential_source_ids: list[str] = []
    authorization_attestation_ids: list[str] = []
    permitted_purposes: list[list[str]] = []
    parent_id_lists: list[list[str]] = []
    game_ids: list[str] = []
    valids: list[bool] = []
    has_outcomes: list[bool] = []
    error_classes: list[str | None] = []
    error_event_indices: list[int | None] = []
    game_event_counts: list[int] = []
    validation_checks: list[list[tuple[str, str]]] = []
    validation_hashes: list[str | None] = []
    for raw in raw_rows:
        outcome = outcomes.get(raw.object_id)
        if outcome is not None and not outcome.valid:
            assert outcome.error is not None
        rec = game_records.get(raw.object_id)
        object_ids.append(raw.object_id)
        packaged_object_ids.append(raw.packaged_object_id)
        confidential_source_ids.append(raw.confidential_source_id)
        authorization_attestation_ids.append(raw.authorization_attestation_id)
        permitted_purposes.append(list(raw.permitted_purpose))
        parent_id_lists.append(list(raw.parent_ids))
        game_event_counts.append(len(rec.events) if rec is not None else 0)
        if outcome is None:
            # Dummy siblings: the bridge owns the missing-outcome gate and
            # fails the call (never a silent skip).
            game_ids.append("")
            valids.append(True)
            has_outcomes.append(False)
            error_classes.append(None)
            error_event_indices.append(None)
            validation_checks.append([])
            validation_hashes.append(None)
        else:
            game_ids.append(outcome.game_id)
            valids.append(outcome.valid)
            has_outcomes.append(True)
            error_classes.append(outcome.error.error_class if outcome.error is not None else None)
            error_event_indices.append(
                outcome.error.event_index if outcome.error is not None else None
            )
            validation_checks.append(list(outcome.checks.items()))
            validation_hashes.append(outcome.validation_hash)
    try:
        slots: list[dict[str, Any]] = columnar.quarantine_project_invalid(
            object_ids,
            packaged_object_ids,
            confidential_source_ids,
            authorization_attestation_ids,
            permitted_purposes,
            parent_id_lists,
            game_ids,
            valids,
            has_outcomes,
            error_classes,
            error_event_indices,
            game_event_counts,
            validation_checks,
            validation_hashes,
        )
    except ValueError as exc:
        text = str(exc)
        if text.startswith(_MISSING_OUTCOME_PREFIX):
            raise RuntimeError(text) from exc
        raise ContractError(text) from exc
    except (TypeError, OverflowError) as exc:
        raise ContractError(str(exc)) from exc
    out: list[QuarantinedRecord] = []
    for slot in slots:
        object_id: str = slot["object_id"]
        packaged_object_id: str = slot["packaged_object_id"]
        game_id: str | None = slot["game_id"]
        error_class: str = slot["error_class"]
        error_event_index: int | None = slot["error_event_index"]
        confidential_source_id: str = slot["confidential_source_id"]
        auth_id: str = slot["authorization_attestation_id"]
        permitted_raw: list[str] = cast("list[str]", slot["permitted_purpose"])
        parent_raw: list[str] = cast("list[str]", slot["parent_ids"])
        game_events: object = slot["game_events"]
        checks_raw: dict[str, str] = cast("dict[str, str]", slot["validation_checks"])
        validation_hash: str | None = slot["validation_hash"]
        out.append(
            QuarantinedRecord(
                object_id=object_id,
                packaged_object_id=packaged_object_id,
                game_id=game_id,
                error_class=error_class,
                error_event_index=error_event_index,
                lineage={
                    "object_id": object_id,
                    "packaged_object_id": packaged_object_id,
                    "confidential_source_id": confidential_source_id,
                    "authorization_attestation_id": auth_id,
                    "permitted_purpose": list(permitted_raw),
                    "parent_ids": list(parent_raw),
                    "game_events": game_events,
                    "validation_checks": dict(checks_raw),
                },
                validation_hash=validation_hash,
            )
        )
    return out


def write_quarantine_manifest(destination: Path, records: list[QuarantinedRecord]) -> str:
    """Frame the sorted quarantine envelope via the bridge, then publish it.

    Thin bridge delegate: ``hydra2._native.columnar.quarantine_manifest_frame``
    sorts by ``object_id`` and seals ``(digest, canonical_bytes)`` through the
    feed-owned canon/digest math in one detach (bridge rejects map to
    :class:`ContractError`); the atomic file publish stays here.
    """
    columnar = _require_columnar()
    try:
        frame: tuple[str, bytes] = columnar.quarantine_manifest_frame(
            [r.object_id for r in records],
            [r.packaged_object_id for r in records],
            [r.game_id for r in records],
            [r.error_class for r in records],
            [r.error_event_index for r in records],
            [_lineage_json(r.lineage) for r in records],
            [r.validation_hash for r in records],
        )
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    digest, blob = frame
    atomic_replace_bytes(destination, blob)
    return digest
