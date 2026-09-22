"""WP-04B RawObjectRow join — SPEC 12.1 immutable join over WP-00B transport rows.

PackagedObjectRow is transport-only (WP-00B authority). RawObjectRow joins one
immutable packaged row with exactly one attestation; the transport row is never
mutated. ObjectId hashes the join (SPEC 12.1: "Its object_id hashes that join").

Hard dependency (shrink end-state): seal math is DELETED here — the B3
single printer (``hydra-feed`` rows via ``feed::canon`` serde_jcs 0.2.0
exact + ``feed::digest`` SHA-256 ONLY) is reached through the columnar
bridge (``packaged_seal_bytes`` / ``packaged_seal_digest`` frame the seal
doc, ``raw_join_seal_bytes`` / ``raw_join_seal_digest`` mint the join id;
all raise with a ``build-ext`` hint when the extension is not built).
NO oracle fallback, never silent. Evidence: packet 283/283 + raw sha
every game; seals migrated (B3 JCS re-mint + seal-replay green).

Kept thin types: :class:`PackagedObjectRow` / :class:`RawObjectRow`
(field framing + contract validation), :func:`_parse_packaged_row`
(exact 16-key set, bare-hex norm, container/member rule),
:func:`load_packaged_manifest` (JSONL framing + seal verify + dedup),
:func:`make_raw_object_row` (join framing; id via the hard digest owner).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError, DigestText, make_utc_timestamp

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


def _require_digest(text: str, *, name: str) -> DigestText:
    try:
        return _bridge_contracts.make_digest_text(text)
    except Exception as exc:  # why-broad: any digest-shape failure is one
        # ContractError with the offending name and value.
        raise ContractError(f"{name} must be sha256:<64 hex>, got {text!r}") from exc


def _seal_bridge() -> Any:
    """Resolve the columnar seal bridge, fail closed when not built.

    Hard-dependency rule (mirrors ``artifacts.canonical._require_bridge``):
    a missing extension or stale ``.so`` without the seal pyfns raises
    ``ImportError`` with a ``build-ext`` hint — NO oracle fallback, never
    silent. Compute rejects from Rust surface as ``ValueError`` and are
    mapped to :class:`ContractError` at the call sites below.
    """
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with columnar not importable; "
            "run `pixi run build-ext` to build the bridge before use"
        ) from exc
    sub = getattr(_ext, "columnar", None)
    if sub is None:
        raise ImportError(
            "hydra2._native.columnar submodule missing (stale .so); "
            "run `pixi run build-ext` to rebuild the bridge"
        )
    for fn_name in (
        "packaged_seal_bytes",
        "packaged_seal_digest",
        "raw_join_seal_bytes",
        "raw_join_seal_digest",
    ):
        if getattr(sub, fn_name, None) is None:
            raise ImportError(
                f"hydra2._native.columnar.{fn_name} missing (stale .so); "
                "run `pixi run build-ext` to rebuild the bridge"
            )
    return sub


def _metadata_json(metadata: dict[str, object]) -> bytes:
    """Serialize join metadata for the seal bridge (keys gated pre-cross).

    JSON objects only carry string keys post-transport, so non-``str`` keys
    are rejected here (fail closed) rather than silently coerced by the
    encoder. Key order is irrelevant: Rust re-parses through the feed
    I-JSON boundary and re-canonicalizes.
    """
    for key in metadata:
        if not isinstance(key, str):
            raise ContractError(f"acquisition_metadata keys must be str, got {key!r}")
    try:
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"acquisition_metadata not JSON-serializable: {exc}") from exc


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

    def _packaged_seal_args(self, *, include_id: bool) -> tuple[object, ...]:
        """Scalar seal args in bridge order (stored form; Rust strips)."""
        return (
            include_id,
            self.packaged_object_id,
            self.source_kind,
            self.source_container_sha256,
            self.source_member_path,
            self.source_bytes_sha256,
            self.source_bytes_length,
            self.compressed_path,
            self.compressed_bytes_sha256,
            self.compressed_bytes_length,
            self.decoded_bytes_sha256,
            self.decoded_bytes_length,
            self.record_count,
            self.canonical_jsonl,
            self.packager_identity,
            self.packager_config_hash,
            self.created_at_utc,
        )

    def canonical_bytes(self, *, include_id: bool = True) -> bytes:
        """Seal bytes via the columnar bridge (B3 single printer in Rust).

        Field order in :meth:`_packaged_seal_args` is normative for
        readability (JCS sorts keys on the wire); digests cross WITH their
        ``sha256:`` prefixes (stored dataclass form) and Rust strips to
        bare hex. ``include_id`` stays for call-site compat; seal bytes
        exclude the id (caller omits it).
        """
        sub = _seal_bridge()
        try:
            sealed: bytes = sub.packaged_seal_bytes(
                *self._packaged_seal_args(include_id=include_id)
            )
            return sealed
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"packaged seal bytes rejected: {exc}") from exc

    def verify_seal(self) -> None:
        """Self-hash check over id-excluded seal bytes (hard digest owner)."""
        sub = _seal_bridge()
        try:
            digest: str = sub.packaged_seal_digest(*self._packaged_seal_args(include_id=False))
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"packaged seal digest rejected: {exc}") from exc
        expected_hex = digest.removeprefix("sha256:")
        stored_hex = self.packaged_object_id.removeprefix("sha256:")
        if stored_hex != expected_hex:
            raise ContractError(
                f"transport row failed self-hash for {self.compressed_path}: "
                f"expected {expected_hex} got {stored_hex}"
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
            _norm_packaged if _norm_packaged is not None else str(cast("object", packaged_id_raw))
        ),
        source_kind=cast("SourceKind", raw_dict["source_kind"]),
        source_container_sha256=norm(cast("object", raw_dict["source_container_sha256"])),
        source_member_path=cast("str | None", raw_dict["source_member_path"]),
        source_bytes_sha256=(
            _norm_source if _norm_source is not None else str(cast("object", source_bytes_raw))
        ),
        source_bytes_length=int(cast("str | int", raw_dict["source_bytes_length"])),
        compressed_path=str(cast("object", raw_dict["compressed_path"])),
        compressed_bytes_sha256=(
            _norm_compressed
            if _norm_compressed is not None
            else str(cast("object", compressed_bytes_raw))
        ),
        compressed_bytes_length=int(cast("str | int", raw_dict["compressed_bytes_length"])),
        decoded_bytes_sha256=(
            _norm_decoded if _norm_decoded is not None else str(cast("object", decoded_bytes_raw))
        ),
        decoded_bytes_length=int(cast("str | int", raw_dict["decoded_bytes_length"])),
        record_count=int(cast("str | int", raw_dict["record_count"])),
        canonical_jsonl=bool(cast("object", raw_dict["canonical_jsonl"])),
        packager_identity=(
            _norm_packager if _norm_packager is not None else str(cast("object", packager_id_raw))
        ),
        packager_config_hash=(
            _norm_config if _norm_config is not None else str(cast("object", packager_cfg_raw))
        ),
        created_at_utc=str(cast("object", raw_dict["created_at_utc"])),
    )


def load_packaged_manifest(path: Path) -> list[PackagedObjectRow]:
    """Load + seal-verify one JSONL manifest (framing kept; seals are hard)."""
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

    def _join_seal_args(self) -> tuple[object, ...]:
        """Scalar join-seal args in bridge order (digests verbatim)."""
        return (
            self.packaged_object_id,
            self.confidential_source_id,
            self.authorization_attestation_id,
            list(self.permitted_purpose),
            self.disclosure_class,
            _metadata_json(self.acquisition_metadata),
            self.semantic_state,
            self.semantic_validation_hash,
            self.first_error_class,
            self.first_error_event_index,
            list(self.parent_ids),
            self.created_at_utc,
        )

    def canonical_bytes_without_id(self) -> bytes:
        """Join seal bytes via the columnar bridge (verbatim digests)."""
        sub = _seal_bridge()
        try:
            joined: bytes = sub.raw_join_seal_bytes(*self._join_seal_args())
            return joined
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"raw join seal bytes rejected: {exc}") from exc


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
    """Join id via the columnar bridge (seal math in Rust, one detach)."""
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
    sub = _seal_bridge()
    try:
        join_digest: str = sub.raw_join_seal_digest(*tmp._join_seal_args())
        return join_digest
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"raw join seal digest rejected: {exc}") from exc


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
