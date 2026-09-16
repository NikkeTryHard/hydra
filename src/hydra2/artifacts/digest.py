"""SHA-256 digest identity — SPEC 2.2 textual form ``sha256:<64 lowercase hex>``.

Two independent recomputation paths are provided on purpose (BUILD WP-02A
exit): :func:`sha256_digest` hashes in-memory bytes, :func:`sha256_file`
streams a file in chunks. Golden tests require both to agree.

Cutover (minimal-Python end-state, Phase 1): :func:`sha256_digest`,
:func:`of_canonical`, and :func:`sha256_file` are Rust-first — the bridge
(``hydra2_replay_rs.canon_rng`` detached batch API) computes, Python compares
via :func:`require_digest_match`. Fallback rule: ``ImportError`` (extension
not built) → Python oracle; any mismatch raises, never silent.
:func:`require_digest_match` stays the Python judge by ownership (Rust
computes, Python compares). Evidence: canon arrays ~2.8-3.8x / flats ~1.4x
Rust-faster (linear); digest parity on every doc.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

from hydra2.artifacts.canonical import _rust_bridge_or_none, canonical_bytes
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


def _sha256_digest_oracle(data: bytes) -> DigestText:
    """Python oracle for :func:`sha256_digest` (hashlib; kept for fallback)."""
    return DigestText("sha256:" + hashlib.sha256(data).hexdigest())


def sha256_digest(data: bytes) -> DigestText:
    """Digest of raw bytes in canonical ``sha256:<hex>`` form.

    Rust-first cutover: computes via the bridge detached batch path
    (``batch_sha256`` batch of one) and compares against the oracle —
    mismatch raises. Fallback rule: ``ImportError`` → oracle. Evidence:
    digest parity on every doc.
    """
    oracle = _sha256_digest_oracle(data)
    bridge = _rust_bridge_or_none()
    if bridge is None:
        return oracle
    rust_text = str(bridge._canon_rng().batch_sha256([bytes(data)])[0])
    require_digest_match(recorded=rust_text, recomputed=oracle, subject="sha256_digest")
    return DigestText(rust_text)


def of_bytes(data: bytes) -> DigestText:
    """Task-facing alias of :func:`sha256_digest`."""
    return sha256_digest(data)


def of_canonical(value: object) -> DigestText:
    """Digest over RFC 8785 canonical bytes of ``value``.

    Rust-first cutover: ``canonical_bytes`` (already Rust-judged) frames the
    doc, then the bridge detached batch path (``batch_of_canonical`` batch of
    one: parse + JCS + hash) recomputes for Python comparison — mismatch
    raises. Fallback rule: ``ImportError`` → oracle. Evidence: canon arrays
    ~2.8-3.8x / flats ~1.4x Rust-faster (linear); digest parity on every doc.
    """
    document = canonical_bytes(value)
    oracle = _sha256_digest_oracle(document)
    bridge = _rust_bridge_or_none()
    if bridge is None:
        return oracle
    rust_text = str(bridge._canon_rng().batch_of_canonical([bytes(document)])[0])
    require_digest_match(recorded=rust_text, recomputed=oracle, subject="of_canonical")
    return DigestText(rust_text)


def validate_digest(text: str) -> DigestText:
    """Validate ``sha256:<64 lowercase hex>``; raises ContractError otherwise."""
    return make_digest_text(text)


def _sha256_file_oracle(path: str | Path) -> DigestText:
    """Python oracle for :func:`sha256_file` (chunked streaming; kept)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return DigestText("sha256:" + digest.hexdigest())


def sha256_file(path: str | Path) -> DigestText:
    """Chunked streaming digest of a file (independent second hash path).

    Rust-first cutover: computes via the bridge file path (native 1 MiB
    chunks, same digest) and compares against the oracle — mismatch raises.
    Fallback rule: ``ImportError`` → oracle. Evidence: digest parity on
    every doc.
    """
    oracle = _sha256_file_oracle(path)
    bridge = _rust_bridge_or_none()
    if bridge is None:
        return oracle
    rust_text = str(bridge._canon_rng().sha256_file(os.fspath(path)))
    require_digest_match(recorded=rust_text, recomputed=oracle, subject="sha256_file")
    return DigestText(rust_text)


def require_digest_match(*, recorded: str, recomputed: DigestText, subject: str) -> None:
    """Raise :class:`DigestMismatchError` unless ``recorded == recomputed``.

    Ownership: this stays the Python judge — Rust computes, Python compares.
    """
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
