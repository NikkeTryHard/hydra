"""Quarantine invalid records with reason and lineage — checklist item 5."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.data.decode import GameRecord
    from hydra2.data.rows import RawObjectRow
    from hydra2.data.validate import ValidationOutcome

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

    - Every invalid outcome becomes a QuarantinedRecord.
    - Lineage includes packaged_object_id, object_id, source, attestation, parent_ids.
    - No silent skip: every input raw row must appear as either valid or quarantined.
    """
    quarantined: list[QuarantinedRecord] = []
    for raw in raw_rows:
        outcome = outcomes.get(raw.object_id)
        if outcome is None:
            # Missing outcome is a hard error — do not silently skip
            raise RuntimeError(
                f"missing validation outcome for {raw.object_id} (would be silent skip)"
            )
        if not outcome.valid:
            assert outcome.error is not None
            rec = game_records.get(raw.object_id)
            quarantined.append(
                QuarantinedRecord(
                    object_id=raw.object_id,
                    packaged_object_id=raw.packaged_object_id,
                    game_id=outcome.game_id,
                    error_class=outcome.error.error_class,
                    error_event_index=outcome.error.event_index,
                    lineage={
                        "object_id": raw.object_id,
                        "packaged_object_id": raw.packaged_object_id,
                        "confidential_source_id": raw.confidential_source_id,
                        "authorization_attestation_id": raw.authorization_attestation_id,
                        "permitted_purpose": list(raw.permitted_purpose),
                        "parent_ids": list(raw.parent_ids),
                        "game_events": len(rec.events) if rec is not None else 0,
                        "validation_checks": outcome.checks,
                    },
                    validation_hash=outcome.validation_hash,
                )
            )
    return quarantined


def write_quarantine_manifest(destination: Path, records: list[QuarantinedRecord]) -> str:
    payload = {
        "schema_version": "1.0.0",
        "quarantined": [
            {
                "object_id": r.object_id,
                "packaged_object_id": r.packaged_object_id,
                "game_id": r.game_id,
                "error_class": r.error_class,
                "error_event_index": r.error_event_index,
                "lineage": r.lineage,
                "validation_hash": r.validation_hash,
            }
            for r in sorted(records, key=lambda x: x.object_id)
        ],
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()
    payload_with_digest = {**payload, "digest": digest}
    atomic_replace_bytes(destination, canonical_bytes(payload_with_digest))
    return digest
