"""WP-04B RawObjectRow join — SPEC 12.1 immutable join over WP-00B transport rows.

PackagedObjectRow is transport-only (WP-00B authority). RawObjectRow joins one
immutable packaged row with exactly one attestation; the transport row is never
mutated. ObjectId hashes the join (SPEC 12.1: "Its object_id hashes that join").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from hydra2.contracts.common import ContractError, DigestText, make_digest_text, make_utc_timestamp

if TYPE_CHECKING:
    from pathlib import Path

SourceKind = Literal["raw", "archive_member", "precompressed"]
SemanticState = Literal["unvalidated", "valid", "quarantined"]

__all__ = [
    "PackagedObjectRow",
    "RawObjectRow",
    "load_packaged_manifest",
    "make_raw_object_row",
]


def _sha_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_digest(text: str, *, name: str) -> DigestText:
    try:
        return make_digest_text(text)
    except Exception as exc:
        raise ContractError(f"{name} must be sha256:<64 hex>, got {text!r}") from exc


def _canonical_row_bytes(fields: list[tuple[str, object]], *, include_id: bool) -> bytes:
    # Mirrors integrity.rs canonical_bytes: field order is normative, compact
    # separators, json-encoded keys/values. Top-level object ordered as passed.
    parts: list[bytes] = [b"{"]
    first = True
    for key, value in fields:
        # skip packaged_object_id when include_id is False is handled by caller
        if not first:
            parts.append(b",")
        first = False
        parts.append(json.dumps(key, separators=(",", ":"), ensure_ascii=False).encode())
        parts.append(b":")
        parts.append(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())
    parts.append(b"}")
    return b"".join(parts)


@dataclass(frozen=True, slots=True)
class PackagedObjectRow:
    """WP-00B transport authority — field order normative for canonical bytes."""

    packaged_object_id: str
    source_kind: SourceKind
    source_container_sha256: str | None
    source_member_path: str | None
    source_bytes_sha256: str
    source_bytes_length: int
    compressed_path: str
    compressed_bytes_sha256: str
    compressed_bytes_length: int
    decoded_bytes_sha256: str
    decoded_bytes_length: int
    record_count: int
    canonical_jsonl: bool
    packager_identity: str
    packager_config_hash: str
    created_at_utc: str

    def __post_init__(self) -> None:
        _ = _require_digest(self.packaged_object_id, name="packaged_object_id")
        if self.source_kind not in ("raw", "archive_member", "precompressed"):
            raise ContractError(f"source_kind invalid {self.source_kind!r}")
        if self.source_container_sha256 is not None:
            _ = _require_digest(self.source_container_sha256, name="source_container_sha256")
        _ = _require_digest(self.source_bytes_sha256, name="source_bytes_sha256")
        _ = _require_digest(self.compressed_bytes_sha256, name="compressed_bytes_sha256")
        _ = _require_digest(self.decoded_bytes_sha256, name="decoded_bytes_sha256")
        _ = _require_digest(self.packager_identity, name="packager_identity")
        _ = _require_digest(self.packager_config_hash, name="packager_config_hash")
        _ = make_utc_timestamp(self.created_at_utc)
        for name in (
            "source_bytes_length",
            "compressed_bytes_length",
            "decoded_bytes_length",
            "record_count",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ContractError(f"{name} must be nonnegative int, got {v!r}")
        if not isinstance(self.canonical_jsonl, bool):
            raise ContractError("canonical_jsonl must be bool")
        if not isinstance(self.compressed_path, str) or self.compressed_path == "":
            raise ContractError("compressed_path must be non-empty string")
        if self.source_kind == "archive_member":
            if self.source_container_sha256 is None or self.source_member_path is None:
                raise ContractError("archive_member requires container and member")
        else:
            if self.source_container_sha256 is not None or self.source_member_path is not None:
                raise ContractError(f"{self.source_kind} must have no container/member")

    def canonical_bytes(self, *, include_id: bool = True) -> bytes:
        def strip(v: object) -> object:
            if isinstance(v, str) and v.startswith("sha256:"):
                return v.removeprefix("sha256:")
            return v

        fields: list[tuple[str, object]] = []
        if include_id:
            fields.append(("packaged_object_id", strip(self.packaged_object_id)))
        fields.extend(
            [
                ("source_kind", self.source_kind),
                ("source_container_sha256", strip(self.source_container_sha256)),
                ("source_member_path", self.source_member_path),
                ("source_bytes_sha256", strip(self.source_bytes_sha256)),
                ("source_bytes_length", self.source_bytes_length),
                ("compressed_path", self.compressed_path),
                ("compressed_bytes_sha256", strip(self.compressed_bytes_sha256)),
                ("compressed_bytes_length", self.compressed_bytes_length),
                ("decoded_bytes_sha256", strip(self.decoded_bytes_sha256)),
                ("decoded_bytes_length", self.decoded_bytes_length),
                ("record_count", self.record_count),
                ("canonical_jsonl", self.canonical_jsonl),
                ("packager_identity", strip(self.packager_identity)),
                ("packager_config_hash", strip(self.packager_config_hash)),
                ("created_at_utc", self.created_at_utc),
            ]
        )
        return _canonical_row_bytes(fields, include_id=True)

    def verify_seal(self) -> None:
        expected = _sha_hex(self.canonical_bytes(include_id=False))
        # stored id includes sha256: prefix in python but rust stores bare hex?
        # Rust stores hex without prefix? Check: integrity.rs to_hex => bare hex,
        # but python spec says DigestText is sha256:<hex>. We normalize both.
        stored_hex = self.packaged_object_id.removeprefix("sha256:")
        if stored_hex != expected:
            raise ContractError(
                f"transport row failed self-hash for {self.compressed_path}: "
                f"expected {expected} got {stored_hex}"
            )


def _parse_packaged_row(raw: object) -> PackagedObjectRow:
    if not isinstance(raw, dict):
        raise ContractError("packaged row must be object")
    raw_dict: dict[str, Any] = cast("dict[str, Any]", raw)
    required = {
        "packaged_object_id",
        "source_kind",
        "source_container_sha256",
        "source_member_path",
        "source_bytes_sha256",
        "source_bytes_length",
        "compressed_path",
        "compressed_bytes_sha256",
        "compressed_bytes_length",
        "decoded_bytes_sha256",
        "decoded_bytes_length",
        "record_count",
        "canonical_jsonl",
        "packager_identity",
        "packager_config_hash",
        "created_at_utc",
    }
    missing = required - set(raw_dict.keys())
    if len(missing) > 0:
        raise ContractError(f"packaged row missing keys {missing}")
    extra = set(raw_dict.keys()) - required
    if len(extra) > 0:
        raise ContractError(f"packaged row extra keys {extra}")

    # Normalize digests to sha256: form if bare hex supplied (pre-WP-01 compatibility)
    def norm(d: object) -> str | None:
        if d is None:
            return None
        if not isinstance(d, str):
            raise ContractError(f"digest must be str, got {d!r}")
        if d.startswith("sha256:"):
            return d
        if len(d) == 64 and all(c in "0123456789abcdef" for c in d):
            return "sha256:" + d
        return d

    _norm_packaged = norm(cast("object", raw_dict["packaged_object_id"]))
    _norm_source = norm(cast("object", raw_dict["source_bytes_sha256"]))
    _norm_compressed = norm(cast("object", raw_dict["compressed_bytes_sha256"]))
    _norm_decoded = norm(cast("object", raw_dict["decoded_bytes_sha256"]))
    _norm_packager = norm(cast("object", raw_dict["packager_identity"]))
    _norm_config = norm(cast("object", raw_dict["packager_config_hash"]))
    packaged_id_raw: Any = raw_dict["packaged_object_id"]
    source_bytes_raw: Any = raw_dict["source_bytes_sha256"]
    compressed_bytes_raw: Any = raw_dict["compressed_bytes_sha256"]
    decoded_bytes_raw: Any = raw_dict["decoded_bytes_sha256"]
    packager_id_raw: Any = raw_dict["packager_identity"]
    packager_cfg_raw: Any = raw_dict["packager_config_hash"]
    return PackagedObjectRow(
        packaged_object_id=(
            _norm_packaged
            if _norm_packaged is not None
            else str(cast("object", packaged_id_raw))
        ),
        source_kind=cast("SourceKind", raw_dict["source_kind"]),
        source_container_sha256=norm(cast("object", raw_dict["source_container_sha256"])),
        source_member_path=cast(
            "str | None", raw_dict["source_member_path"]
        ),
        source_bytes_sha256=(
            _norm_source
            if _norm_source is not None
            else str(cast("object", source_bytes_raw))
        ),
        source_bytes_length=int(cast("str | int", raw_dict["source_bytes_length"])),
        compressed_path=str(cast("object", raw_dict["compressed_path"])),
        compressed_bytes_sha256=(
            _norm_compressed
            if _norm_compressed is not None
            else str(cast("object", compressed_bytes_raw))
        ),
        compressed_bytes_length=int(
            cast("str | int", raw_dict["compressed_bytes_length"])
        ),
        decoded_bytes_sha256=(
            _norm_decoded
            if _norm_decoded is not None
            else str(cast("object", decoded_bytes_raw))
        ),
        decoded_bytes_length=int(cast("str | int", raw_dict["decoded_bytes_length"])),
        record_count=int(cast("str | int", raw_dict["record_count"])),
        canonical_jsonl=bool(cast("object", raw_dict["canonical_jsonl"])),
        packager_identity=(
            _norm_packager
            if _norm_packager is not None
            else str(cast("object", packager_id_raw))
        ),
        packager_config_hash=(
            _norm_config
            if _norm_config is not None
            else str(cast("object", packager_cfg_raw))
        ),
        created_at_utc=str(cast("object", raw_dict["created_at_utc"])),
    )

def load_packaged_manifest(path: Path) -> list[PackagedObjectRow]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    rows: list[PackagedObjectRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"manifest line {lineno} invalid JSON: {exc}") from exc
        row = _parse_packaged_row(raw)
        row.verify_seal()
        rows.append(row)
    # dedup by compressed_path
    seen: set[str] = set()
    for r in rows:
        if r.compressed_path in seen:
            raise ContractError(f"duplicate compressed_path {r.compressed_path!r} in manifest")
        seen.add(r.compressed_path)
    return rows


@dataclass(frozen=True, slots=True)
class RawObjectRow:
    """WP-04B authority — join of one packaged row + one attestation."""

    object_id: str
    packaged_object_id: str
    confidential_source_id: str
    authorization_attestation_id: str
    permitted_purpose: tuple[str, ...]
    disclosure_class: str
    acquisition_metadata: dict[str, object]
    semantic_state: SemanticState
    semantic_validation_hash: str | None
    first_error_class: str | None
    first_error_event_index: int | None
    parent_ids: tuple[str, ...]
    created_at_utc: str

    def __post_init__(self) -> None:
        _ = _require_digest(self.object_id, name="object_id")
        _ = _require_digest(self.packaged_object_id, name="packaged_object_id")
        if not isinstance(self.confidential_source_id, str) or self.confidential_source_id == "":
            raise ContractError("confidential_source_id must be non-empty str")
        if (
            not isinstance(self.authorization_attestation_id, str)
            or self.authorization_attestation_id == ""
        ):
            raise ContractError("authorization_attestation_id must be non-empty str")
        if not isinstance(self.permitted_purpose, tuple) or not all(
            isinstance(s, str) and s for s in self.permitted_purpose
        ):
            raise ContractError("permitted_purpose must be tuple of non-empty str")
        if not isinstance(self.disclosure_class, str) or self.disclosure_class == "":
            raise ContractError("disclosure_class must be non-empty str")
        if not isinstance(self.acquisition_metadata, dict):
            raise ContractError("acquisition_metadata must be dict")
        if self.semantic_state not in ("unvalidated", "valid", "quarantined"):
            raise ContractError(f"semantic_state invalid {self.semantic_state!r}")
        if self.semantic_validation_hash is not None:
            _ = _require_digest(self.semantic_validation_hash, name="semantic_validation_hash")
        if self.first_error_event_index is not None and (
            not isinstance(self.first_error_event_index, int) or self.first_error_event_index < 0
        ):
            raise ContractError("first_error_event_index must be nonnegative int or None")
        for pid in self.parent_ids:
            _ = _require_digest(pid, name="parent_id")
        _ = make_utc_timestamp(self.created_at_utc)

    def canonical_bytes_without_id(self) -> bytes:
        fields: list[tuple[str, object]] = [
            ("packaged_object_id", self.packaged_object_id),
            ("confidential_source_id", self.confidential_source_id),
            ("authorization_attestation_id", self.authorization_attestation_id),
            ("permitted_purpose", list(self.permitted_purpose)),
            ("disclosure_class", self.disclosure_class),
            ("acquisition_metadata", self.acquisition_metadata),
            ("semantic_state", self.semantic_state),
            ("semantic_validation_hash", self.semantic_validation_hash),
            ("first_error_class", self.first_error_class),
            ("first_error_event_index", self.first_error_event_index),
            ("parent_ids", list(self.parent_ids)),
            ("created_at_utc", self.created_at_utc),
        ]
        return _canonical_row_bytes(fields, include_id=True)


def _raw_object_id_for(
    *,
    packaged_object_id: str,
    confidential_source_id: str,
    authorization_attestation_id: str,
    permitted_purpose: tuple[str, ...],
    disclosure_class: str,
    acquisition_metadata: dict[str, object],
    semantic_state: SemanticState,
    semantic_validation_hash: str | None,
    first_error_class: str | None,
    first_error_event_index: int | None,
    parent_ids: tuple[str, ...],
    created_at_utc: str,
) -> str:
    tmp = RawObjectRow(
        object_id="sha256:" + "0" * 64,
        packaged_object_id=packaged_object_id,
        confidential_source_id=confidential_source_id,
        authorization_attestation_id=authorization_attestation_id,
        permitted_purpose=permitted_purpose,
        disclosure_class=disclosure_class,
        acquisition_metadata=acquisition_metadata,
        semantic_state=semantic_state,
        semantic_validation_hash=semantic_validation_hash,
        first_error_class=first_error_class,
        first_error_event_index=first_error_event_index,
        parent_ids=parent_ids,
        created_at_utc=created_at_utc,
    )
    raw = tmp.canonical_bytes_without_id()
    return "sha256:" + _sha_hex(raw)


def make_raw_object_row(
    packaged: PackagedObjectRow,
    *,
    confidential_source_id: str,
    authorization_attestation_id: str,
    permitted_purpose: tuple[str, ...],
    disclosure_class: str,
    acquisition_metadata: dict[str, object],
    semantic_state: SemanticState = "unvalidated",
    semantic_validation_hash: str | None = None,
    first_error_class: str | None = None,
    first_error_event_index: int | None = None,
    parent_ids: tuple[str, ...] = (),
    created_at_utc: str | None = None,
) -> RawObjectRow:
    """Join one immutable packaged row with one attestation — never mutates packaged."""
    # Defensive copy: ensure we don't mutate packaged (frozen anyway)
    packaged_id = packaged.packaged_object_id
    # Attestation presence is mandatory; missing cannot be represented
    if authorization_attestation_id == "":
        raise ContractError(
            "authorization_attestation_id is required; missing attestation cannot be represented"
        )
    if created_at_utc is None:
        from datetime import UTC, datetime

        created_at_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        _ = make_utc_timestamp(created_at_utc)
    object_id = _raw_object_id_for(
        packaged_object_id=packaged_id,
        confidential_source_id=confidential_source_id,
        authorization_attestation_id=authorization_attestation_id,
        permitted_purpose=permitted_purpose,
        disclosure_class=disclosure_class,
        acquisition_metadata=acquisition_metadata,
        semantic_state=semantic_state,
        semantic_validation_hash=semantic_validation_hash,
        first_error_class=first_error_class,
        first_error_event_index=first_error_event_index,
        parent_ids=parent_ids,
        created_at_utc=created_at_utc,
    )
    return RawObjectRow(
        object_id=object_id,
        packaged_object_id=packaged_id,
        confidential_source_id=confidential_source_id,
        authorization_attestation_id=authorization_attestation_id,
        permitted_purpose=permitted_purpose,
        disclosure_class=disclosure_class,
        acquisition_metadata=dict(acquisition_metadata),
        semantic_state=semantic_state,
        semantic_validation_hash=semantic_validation_hash,
        first_error_class=first_error_class,
        first_error_event_index=first_error_event_index,
        parent_ids=parent_ids,
        created_at_utc=created_at_utc,
    )
