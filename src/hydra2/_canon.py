"""Bootstrap canonical-serialization helpers — superseded by WP-02A.

The qualified implementation lives in :mod:`hydra2.artifacts` (RFC 8785 JCS
canonical bytes, ``sha256:`` digests, atomic publication). This module is a
pure re-export shim so WP-01 callers (``completion``, ``runtime``, ``probe``,
``config``) keep importing their historical names; the canonicalization is now
full RFC 8785 instead of the bootstrap stdlib approximation.

Behavior deltas vs the WP-01 bootstrap (intended, per BUILD WP-02A):
numbers follow ECMA-262 ``Number::toString`` (e.g. ``1.0 -> 1``,
``1e-07 -> 1e-7``, ``-0.0 -> 0``); key ordering is UTF-16 code-unit based;
rejections raise the typed ``CanonicalizationError``.
"""

from pathlib import Path

from hydra2.artifacts.atomic import _mkstemp_o_excl, atomic_replace_bytes
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import (
    of_canonical,
    require_digest_match,
    sha256_digest,
    sha256_file,
)
from hydra2.contracts.common import CanonicalizationError, DigestMismatchError, DigestText

# Deprecated WP-01 compatibility surface: ``runtime.checkpoint`` predates the
# artifacts package and imports these names from here. They delegate to the
# qualified implementation; new code MUST use hydra2.artifacts directly and a
# future contract migration will retire the aliases.
NonFiniteNumberError = CanonicalizationError

__all__ = [
    "CanonicalizationError",
    "DigestMismatchError",
    "DigestText",
    "NonFiniteNumberError",
    "_mkstemp_o_excl",
    "atomic_write_bytes",
    "canonical_bytes",
    "canonical_json_bytes",
    "require_digest_match",
    "sha256_digest",
    "sha256_digest_of_json",
    "sha256_file",
]


def canonical_json_bytes(value: object) -> bytes:
    """RFC 8785 canonical UTF-8 JSON bytes (delegates to hydra2.artifacts)."""
    return canonical_bytes(value)


def sha256_digest_of_json(value: object) -> DigestText:
    """Digest over RFC 8785 canonical bytes of ``value``."""
    return of_canonical(value)


def atomic_write_bytes(destination: Path, data: bytes) -> None:
    """Publish mutable control bytes atomically (unique temp/fsync/rename/fsync dir)."""
    atomic_replace_bytes(Path(destination), data)
