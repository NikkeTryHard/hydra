"""Immutable artifact registry — SPEC 2.2 envelopes and (type, version, hash) lookup.

Layout under the registry root::

    artifacts/<artifact_type>/<64-hex artifact id>.json   # immutable bytes
    artifacts/index.json                                  # mutable rows

Identity: ``artifact_id(envelope) = sha256(canonical_bytes(to_json(envelope)))``.
Registry rows (compatibility declaration, migration metadata) are metadata and
MUST NOT change identity. Readers recompute the digest of stored bytes before
decoding, then reject unknown major versions, unknown required fields, invalid
enums, noncanonical digests, identity collisions, truncation, tampering, and
overwrite attempts with the SPEC 3 typed errors. Unknown OPTIONAL fields are
accepted only when the envelope's own ``compatibility == "backward_read"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hydra2.artifacts.atomic import atomic_replace_bytes, publish_atomic
from hydra2.artifacts.canonical import canonical_bytes, loads_canonical
from hydra2.artifacts.digest import sha256_digest, validate_digest
from hydra2.contracts.common import (
    ContractError,
    CorruptArtifactError,
    IncompatibleSchemaError,
    LineageError,
    SchemaVersion,
    make_schema_version,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "COMPATIBILITY_VALUES",
    "INDEX_SCHEMA_VERSION",
    "ArtifactEnvelope",
    "ArtifactRegistry",
    "MigrationMetadata",
    "RegistryEntry",
    "artifact_id",
    "envelope_from_json",
    "to_json",
]

COMPATIBILITY_VALUES = ("exact", "backward_read")
INDEX_SCHEMA_VERSION = "1.0.0"

_ENVELOPE_REQUIRED_FIELDS = (
    "artifact_type",
    "schema_version",
    "compatibility",
    "payload",
)


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    """SPEC 2.2 identity envelope for every versioned Hydra2 artifact."""

    artifact_type: str
    schema_version: SchemaVersion
    compatibility: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, str) or self.artifact_type == "":
            raise ContractError(f"artifact_type must be a non-empty str: {self.artifact_type!r}")
        object.__setattr__(self, "schema_version", make_schema_version(self.schema_version))
        if self.compatibility not in COMPATIBILITY_VALUES:
            raise ContractError(
                f"compatibility must be one of {list(COMPATIBILITY_VALUES)}, "
                f"got {self.compatibility!r}"
            )
        if not isinstance(self.payload, dict):
            raise ContractError("payload must be a string-keyed JSON object")


def to_json(envelope: ArtifactEnvelope) -> dict[str, Any]:
    """Plain JSON document for ``envelope`` (fresh mapping; payload shallow-copied)."""
    return {
        "artifact_type": envelope.artifact_type,
        "schema_version": envelope.schema_version,
        "compatibility": envelope.compatibility,
        "payload": dict(envelope.payload),
    }


def artifact_id(envelope: ArtifactEnvelope) -> str:
    """SPEC 2.2 identity: sha256 over RFC 8785 canonical bytes of the envelope."""
    return sha256_digest(canonical_bytes(to_json(envelope)))


_envelope_identity = artifact_id


def _major(version: str) -> int:
    return int(version.split(".", 1)[0])


def envelope_from_json(
    raw: Any, *, supported_majors: frozenset[int] | None = None
) -> ArtifactEnvelope:
    """Validate a parsed document as an :class:`ArtifactEnvelope`.

    Unknown fields are rejected unless the document itself declares
    ``compatibility="backward_read"``. A major version outside
    ``supported_majors`` raises :class:`IncompatibleSchemaError`.
    """
    if not isinstance(raw, dict):
        raise ContractError("artifact envelope must be a JSON object")
    raw_dict: dict[str, Any] = cast("dict[str, Any]", raw)
    for field in _ENVELOPE_REQUIRED_FIELDS:
        if field not in raw_dict:
            raise ContractError(f"artifact envelope missing required field {field!r}")
    unknown: list[str] = sorted(set(raw_dict) - set(_ENVELOPE_REQUIRED_FIELDS))
    if len(unknown) != 0 and cast("str", raw_dict["compatibility"]) != "backward_read":
        raise ContractError(
            f"unknown envelope field(s) {unknown}; compatibility "
            f"{cast('str', raw_dict['compatibility'])!r} does not allow them"
        )
    if not isinstance(raw_dict.get("payload"), dict):
        raise ContractError("payload must be a string-keyed JSON object")
    envelope = ArtifactEnvelope(
        artifact_type=cast("str", raw_dict["artifact_type"]),
        schema_version=SchemaVersion(cast("str", raw_dict["schema_version"])),
        compatibility=cast("str", raw_dict["compatibility"]),
        payload=cast("dict[str, Any]", raw_dict["payload"]),
    )
    if supported_majors is not None and _major(envelope.schema_version) not in supported_majors:
        raise IncompatibleSchemaError(
            f"unsupported schema major version for {envelope.artifact_type}: "
            f"{envelope.schema_version}"
        )
    return envelope


@dataclass(frozen=True, slots=True)
class MigrationMetadata:
    """Provenance of an artifact migrated from an older schema version.

    Lives in the registry ROW only — never in the envelope — so migration
    bookkeeping cannot change artifact identity (SPEC 2.2).
    """

    source_schema_version: SchemaVersion
    migration_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_schema_version", make_schema_version(self.source_schema_version)
        )
        if not isinstance(self.migration_id, str) or self.migration_id == "":
            raise ContractError(f"migration_id must be a non-empty str: {self.migration_id!r}")

    def to_json(self) -> dict[str, str]:
        return {
            "source_schema_version": self.source_schema_version,
            "migration_id": self.migration_id,
        }

    @classmethod
    def from_json(cls, raw: Any) -> MigrationMetadata:
        if not isinstance(raw, dict):
            raise ContractError("migration metadata must be a JSON object")
        raw_dict: dict[str, Any] = cast("dict[str, Any]", raw)
        missing: set[str] = {"source_schema_version", "migration_id"} - set(raw_dict)
        if len(missing) != 0:
            raise ContractError(f"migration metadata missing required field(s) {sorted(missing)}")
        unknown: list[str] = sorted(set(raw_dict) - {"source_schema_version", "migration_id"})
        if len(unknown) != 0:
            raise ContractError(f"unknown migration metadata field(s) {unknown}")
        return cls(
            source_schema_version=SchemaVersion(cast("str", raw_dict["source_schema_version"])),
            migration_id=cast("str", raw_dict["migration_id"]),
        )


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Index row describing one published immutable artifact."""

    artifact_type: str
    schema_version: SchemaVersion
    compatibility: str
    artifact_id: str
    migration: MigrationMetadata | None


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Verified artifact plus its registry row."""

    envelope: ArtifactEnvelope
    entry: RegistryEntry


class ArtifactRegistry:
    """Content-addressed immutable store keyed by (type, version, hash)."""

    def __init__(self, root: Path | str, *, supported_majors: frozenset[int] | None = None) -> None:
        self.root = Path(root)
        self._supported_majors = (
            supported_majors if supported_majors is not None else frozenset({1})
        )

    # -- paths -------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / "artifacts" / "index.json"

    def artifact_path(self, artifact_type: str, digest: str) -> Path:
        hex_part = validate_digest(digest).split(":", 1)[1]
        return self.root / "artifacts" / artifact_type / f"{hex_part}.json"

    # -- publication ---------------------------------------------------------

    def publish(
        self,
        envelope: ArtifactEnvelope,
        *,
        migration: MigrationMetadata | None = None,
    ) -> RegistryEntry:
        """Publish immutable bytes, then record the row (SPEC 2.3 ordering)."""
        self._require_supported_major(envelope)
        document = to_json(envelope)
        data = canonical_bytes(document)
        digest = sha256_digest(data)
        destination = self.artifact_path(envelope.artifact_type, digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_atomic(destination=destination, data=data, expected=digest)
        entry = RegistryEntry(
            artifact_type=envelope.artifact_type,
            schema_version=envelope.schema_version,
            compatibility=envelope.compatibility,
            artifact_id=digest,
            migration=migration,
        )
        self._upsert_index_row(entry)
        return entry

    # -- lookup ---------------------------------------------------------------

    def lookup(self, *, artifact_type: str, schema_version: str, artifact_id: str) -> LookupResult:
        """Resolve and fully verify one artifact by its identity triple."""
        digest = validate_digest(artifact_id)
        _ = make_schema_version(schema_version)
        if _major(schema_version) not in self._supported_majors:
            raise IncompatibleSchemaError(
                f"unsupported requested schema major for {artifact_type}: {schema_version}"
            )
        row = self._index_rows().get((artifact_type, schema_version, digest))
        if row is None:
            raise LineageError(f"no registered artifact {artifact_type} v{schema_version} {digest}")
        path = self.artifact_path(artifact_type, digest)
        if not path.is_file():
            raise CorruptArtifactError(f"registered artifact file is missing: {path}")
        stored_bytes = path.read_bytes()
        recomputed = sha256_digest(stored_bytes)  # BEFORE decoding (SPEC 2.2)
        if recomputed != digest:
            raise CorruptArtifactError(
                f"stored bytes of {path} hash to {recomputed}, registered {digest}"
            )
        document = loads_canonical(stored_bytes)
        envelope = envelope_from_json(document, supported_majors=self._supported_majors)
        stored_identity = _envelope_identity(envelope)
        if stored_identity != digest:
            raise CorruptArtifactError(
                f"identity collision at {path}: content hashes to {stored_identity}, "
                f"addressed as {digest}"
            )
        if envelope.artifact_type != artifact_type or envelope.schema_version != schema_version:
            raise CorruptArtifactError(
                f"identity collision at {path}: envelope declares "
                f"{envelope.artifact_type} v{envelope.schema_version}"
            )
        return LookupResult(envelope=envelope, entry=row)

    # -- index ----------------------------------------------------------------

    def _require_supported_major(self, envelope: ArtifactEnvelope) -> None:
        if _major(envelope.schema_version) not in self._supported_majors:
            raise IncompatibleSchemaError(
                f"unsupported schema major for {envelope.artifact_type}: {envelope.schema_version}"
            )

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return {"schema_version": INDEX_SCHEMA_VERSION, "rows": {}}
        raw = loads_canonical(self.index_path.read_bytes())
        if not isinstance(raw, dict) or not isinstance(raw.get("rows"), dict):
            raise CorruptArtifactError(f"registry index is malformed: {self.index_path}")
        return raw

    def _index_rows(self) -> dict[tuple[str, str, Any], RegistryEntry]:
        rows: dict[tuple[str, str, Any], RegistryEntry] = {}
        index_rows: dict[str, Any] = cast("dict[str, Any]", self._load_index()["rows"])
        for artifact_type, versions in index_rows.items():
            versions_dict: dict[str, Any] = cast("dict[str, Any]", versions)
            for version, entries in versions_dict.items():
                entries_dict: dict[str, Any] = cast("dict[str, Any]", entries)
                for hex_id, meta in entries_dict.items():
                    digest = validate_digest(f"sha256:{hex_id}")
                    meta_dict: dict[str, Any] = (
                        cast("dict[str, Any]", meta)
                        if isinstance(meta, dict)
                        else cast("dict[str, Any]", {})
                    )
                    migration_raw: Any = (
                        cast("Any", meta_dict.get("migration"))
                        if isinstance(meta, dict)
                        else None
                    )
                    entry = RegistryEntry(
                        artifact_type=artifact_type,
                        schema_version=cast("SchemaVersion", version),
                        compatibility=cast("str", meta_dict["compatibility"]),
                        artifact_id=digest,
                        migration=(
                            MigrationMetadata.from_json(cast("Any", migration_raw))
                            if migration_raw is not None
                            else None
                        ),
                    )
                    rows[(artifact_type, version, digest)] = entry
        return rows

    def _upsert_index_row(self, entry: RegistryEntry) -> None:
        index: dict[str, Any] = self._load_index()
        rows: dict[str, Any] = cast("dict[str, Any]", index["rows"])
        type_rows: dict[str, Any] = cast(
            "dict[str, Any]", rows.setdefault(entry.artifact_type, {})
        )
        version_rows: dict[str, Any] = cast(
            "dict[str, Any]", type_rows.setdefault(entry.schema_version, {})
        )
        hex_id: str = entry.artifact_id.split(":", 1)[1]
        existed: bool = hex_id in version_rows
        if not existed:
            version_rows[hex_id] = {
                "compatibility": entry.compatibility,
                **({"migration": entry.migration.to_json()} if entry.migration is not None else {}),
            }
            atomic_replace_bytes(
                self.index_path,
                canonical_bytes({"schema_version": INDEX_SCHEMA_VERSION, "rows": rows}),
            )
