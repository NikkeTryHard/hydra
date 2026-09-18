"""Ingest from WP-00B Rust packager — manifest + zstd verification.

Implements checklist items 2 & 3: ingest via --manifest hidden flag, full zstd
decode (magic bytes alone never authorize), decode one-game-per-object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.data.attestation import Attestation, require_attestation
from hydra2.data.rows import (
    PackagedObjectRow,
    RawObjectRow,
    load_packaged_manifest,
    make_raw_object_row,
)

__all__ = ["IngestedObject", "decode_zstd_verified", "ingest_packaged_objects"]


@dataclass(frozen=True, slots=True)
class IngestedObject:
    packaged: PackagedObjectRow
    raw: RawObjectRow
    decoded_bytes: bytes
    decoded_path: Path


def _require_packet() -> Any:
    """Import the built ``packet`` bridge surface (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with packet not importable; "
            "build the bridge with `pixi run build-ext` before ingesting objects"
        ) from exc
    try:
        return _ext.packet
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.packet submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def decode_zstd_verified(path: Path, expected_sha256: str, expected_len: int) -> bytes:
    """Fully decode a zstd file and verify hash/length; never trusts magic alone.

    Hard dependency (Wave 1 R4): the zstd plane delegates to the Rust bridge
    (``hydra2_replay_rs.packet`` ``decode_zstd_verified``: 64 KiB reads,
    512 MiB fail-closed guard, ``sha256:<hex>``/length mismatch → error).
    ``ImportError`` (extension not built) raises with a ``build-ext`` hint —
    NO oracle fallback, never silent. Record counting stays Python (the
    ``len`` over decoded lines in :func:`ingest_packaged_objects`): no bridge
    count pyfn exists.
    """
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise CorruptArtifactError(f"cannot read compressed object {path}: {exc}") from exc
    try:
        return _require_packet().decode_zstd_verified(compressed, expected_sha256, expected_len)
    except ValueError as exc:
        raise CorruptArtifactError(f"zstd decode failed for {path}: {exc}") from exc


def ingest_packaged_objects(
    *,
    manifest_path: Path,
    output_root: Path,
    attestation: Attestation,
    allow_synthetic: bool = True,
) -> list[IngestedObject]:
    """Ingest authoritative rows from WP-00B manifest + output root.

    For each PackagedObjectRow:
      - verifies seal (load_packaged_manifest already does)
      - verifies compressed file exists and hashes match
      - fully zstd-decodes and verifies decoded hash
      - joins with attestation to produce RawObjectRow (never mutates transport)
    No silent skip: any missing/corrupt file is a hard error.
    """
    att = require_attestation(attestation, allow_synthetic=allow_synthetic)
    rows = load_packaged_manifest(manifest_path)
    if len(rows) == 0:
        raise ContractError("manifest contains no rows; refusing silent empty ingest")
    results: list[IngestedObject] = []
    for pkg in rows:
        compressed_path = Path(pkg.compressed_path)
        if not compressed_path.is_absolute():
            compressed_path = output_root / compressed_path
        if not compressed_path.is_file():
            raise CorruptArtifactError(
                f"compressed object missing: {compressed_path} (no silent skip)"
            )
        data = compressed_path.read_bytes()
        c_sha = "sha256:" + hashlib.sha256(data).hexdigest()
        if c_sha != pkg.compressed_bytes_sha256:
            raise CorruptArtifactError(
                f"compressed hash mismatch for {compressed_path}: "
                f"{c_sha} != {pkg.compressed_bytes_sha256}"
            )
        if len(data) != pkg.compressed_bytes_length:
            raise CorruptArtifactError(f"compressed length mismatch for {compressed_path}")
        decoded = decode_zstd_verified(
            compressed_path, pkg.decoded_bytes_sha256, pkg.decoded_bytes_length
        )
        lines = decoded.splitlines()
        parsed_count = 0
        for line in lines:
            if line.strip() == b"":
                continue
            try:
                json.loads(line)
                parsed_count += 1
            except json.JSONDecodeError:
                continue
        if parsed_count != pkg.record_count:
            raise CorruptArtifactError(
                f"record_count mismatch for {compressed_path}: "
                f"manifest {pkg.record_count} vs parsed {parsed_count}"
            )
        raw = make_raw_object_row(
            pkg,
            confidential_source_id=att.confidential_source_id,
            authorization_attestation_id=att.attestation_id,
            permitted_purpose=att.permitted_purpose,
            disclosure_class=att.disclosure_class,
            acquisition_metadata=dict(att.acquisition_metadata),
            semantic_state="unvalidated",
        )
        results.append(
            IngestedObject(
                packaged=pkg, raw=raw, decoded_bytes=decoded, decoded_path=compressed_path
            )
        )
    return results
