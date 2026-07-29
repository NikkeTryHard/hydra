"""SHA-256 digest identity — SPEC 2.2 textual form ``sha256:<64 lowercase hex>``.

Two independent recomputation paths are provided on purpose (BUILD WP-02A
exit): :func:`sha256_digest` hashes in-memory bytes, :func:`sha256_file`
streams a file in chunks. Golden tests require both to agree.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    DigestMismatchError,
    DigestText,
    make_digest_text,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "of_bytes",
    "of_canonical",
    "require_digest_match",
    "sha256_digest",
    "sha256_file",
    "validate_digest",
]

_CHUNK_SIZE = 1 << 20


def sha256_digest(data: bytes) -> DigestText:
    """Digest of raw bytes in canonical ``sha256:<hex>`` form."""
    return DigestText("sha256:" + hashlib.sha256(data).hexdigest())


def of_bytes(data: bytes) -> DigestText:
    """Task-facing alias of :func:`sha256_digest`."""
    return sha256_digest(data)


def of_canonical(value: object) -> DigestText:
    """Digest over RFC 8785 canonical bytes of ``value``."""
    return sha256_digest(canonical_bytes(value))


def validate_digest(text: str) -> DigestText:
    """Validate ``sha256:<64 lowercase hex>``; raises ContractError otherwise."""
    return make_digest_text(text)


def sha256_file(path: str | Path) -> DigestText:
    """Chunked streaming digest of a file (independent second hash path)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return DigestText("sha256:" + digest.hexdigest())


def require_digest_match(*, recorded: str, recomputed: DigestText, subject: str) -> None:
    """Raise :class:`DigestMismatchError` unless ``recorded == recomputed``."""
    try:
        recorded_digest = make_digest_text(recorded)
    except Exception as exc:
        raise DigestMismatchError(
            f"{subject}: recorded digest {recorded!r} is not a valid sha256 digest"
        ) from exc
    if recorded_digest != recomputed:
        raise DigestMismatchError(
            f"{subject}: recorded {recorded_digest} != recomputed {recomputed}"
        )
