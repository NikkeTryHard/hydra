"""SHA-256 digest identity — textual form ``sha256:<64 lowercase hex>``.

Two independent recomputation paths are provided on purpose:
:func:`sha256_digest` hashes in-memory bytes, :func:`sha256_file`
streams a file in chunks. Golden tests require both to agree.
Hard dependency (shrink end-state): :func:`sha256_digest`,
:func:`of_canonical`, and :func:`sha256_file` are thin hard-Rust delegates —
the bridge (``hydra2._native.canon_rng`` detached batch API) computes and
its text is returned directly. ``ImportError`` (extension not built) raises
with a ``build-ext`` hint — NO oracle fallback, never silent.
:func:`require_digest_match` stays the Python judge by ownership (Rust
computes, Python compares). Evidence: canon arrays ~2.8-3.8x / flats ~1.4x
Rust-faster (linear); digest parity on every doc.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import _require_bridge, canonical_bytes
from hydra2.contracts.common import (
    DigestMismatchError,
    DigestText,
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


def sha256_digest(data: bytes) -> DigestText:
    """Digest of raw bytes in canonical ``sha256:<hex>`` form.

    Thin hard-Rust delegate: computes via the bridge 1:1 pyfn (``sha256_hex``)
    and returns the bridge text directly. ``ImportError`` raises with a
    ``build-ext`` hint — NO oracle fallback. Evidence: digest parity on every
    doc.
    """
    bridge = _require_bridge()
    rust_text: str = bridge._canon_rng().sha256_hex(data)
    return DigestText(rust_text)


def of_bytes(data: bytes) -> DigestText:
    """Task-facing alias of :func:`sha256_digest`."""
    return sha256_digest(data)


def of_canonical(value: object) -> DigestText:
    """Digest over RFC 8785 canonical bytes of ``value``.

    Thin hard-Rust delegate: ``canonical_bytes`` (Python canon authority,
    bridge-verified) frames the doc, then the bridge 1:1 pyfn
    (``of_canonical_json``: parse + JCS + hash) computes the digest returned
    directly. ``ImportError`` raises with a ``build-ext`` hint — NO oracle
    fallback. Evidence: canon arrays ~2.8-3.8x / flats ~1.4x Rust-faster
    (linear); digest parity on every doc.
    """
    document = canonical_bytes(value)
    bridge = _require_bridge()
    rust_text: str = bridge._canon_rng().of_canonical_json(document)
    return DigestText(rust_text)


def validate_digest(text: str) -> DigestText:
    """Validate ``sha256:<64 lowercase hex>``; raises ValueError otherwise."""
    return _bridge_contracts.make_digest_text(text)


def sha256_file(path: str | Path) -> DigestText:
    """Chunked streaming digest of a file (independent second hash path).

    Thin hard-Rust delegate: computes via the bridge file path (native 1 MiB
    chunks, same digest) and returns the bridge text directly. ``ImportError``
    raises with a ``build-ext`` hint — NO oracle fallback. Evidence: digest
    parity on every doc.
    """
    bridge = _require_bridge()
    rust_text: str = bridge._canon_rng().sha256_file(os.fspath(path))
    return DigestText(rust_text)


def require_digest_match(*, recorded: str, recomputed: DigestText, subject: str) -> None:
    """Raise :class:`DigestMismatchError` unless ``recorded == recomputed``.

    Ownership: this stays the Python judge — Rust computes, Python compares.
    """
    try:
        recorded_digest: DigestText = _bridge_contracts.make_digest_text(recorded)
    except Exception as exc:
        raise DigestMismatchError(
            f"{subject}: recorded digest {recorded!r} is not a valid sha256 digest"
        ) from exc
    if recorded_digest != recomputed:
        raise DigestMismatchError(
            f"{subject}: recorded {recorded_digest} != recomputed {recomputed}"
        )
