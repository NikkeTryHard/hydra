"""Canonical identity bytes for contract artifacts — shared leaf module.

Cutover (shrink end-state): the RFC 8785 recipe is owned by the
``hydra2.artifacts.canonical.canonical_bytes`` authority (Python serializer
as canon authority — no bytes-returning ``canon_rng`` pyfn exists — with a
hard bridge judge; ``ImportError`` raises with a ``build-ext`` hint, NO
oracle fallback). This module keeps its name and signature as a thin
delegate with no logic; the import stays lazy so contract import time still
pulls stdlib only (SPEC 1 import-time property preserved). Evidence: canon
arrays ~2.8-3.8x / flats ~1.4x Rust-faster (linear); digest parity on every
doc.
"""

from __future__ import annotations

__all__ = ["canonical_json_bytes"]


def canonical_json_bytes(document: object) -> bytes:
    """RFC 8785 bytes (thin delegate of the flipped artifacts authority; no logic)."""
    from hydra2.artifacts.canonical import canonical_bytes

    return canonical_bytes(document)
