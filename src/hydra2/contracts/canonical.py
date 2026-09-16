"""Canonical identity bytes for contract artifacts — shared leaf module.

Cutover (minimal-Python end-state, Phase 1): the RFC 8785 recipe is now owned
by the flipped ``hydra2.artifacts.canonical.canonical_bytes`` authority
(Rust-first judge: bridge computes, Python compares; ``ImportError`` →
Python oracle; mismatch raises). This module keeps its name and signature as
a thin delegate with no logic; the import stays lazy so contract import time
still pulls stdlib only (SPEC 1 import-time property preserved). Byte
equality with the authority holds by construction (same function); digest
parity on every doc (arrays ~2.8-3.8x / flats ~1.4x Rust-faster evidence).
"""

from __future__ import annotations

__all__ = ["canonical_json_bytes"]


def canonical_json_bytes(document: object) -> bytes:
    """RFC 8785 bytes (thin delegate of the flipped artifacts authority; no logic)."""
    from hydra2.artifacts.canonical import canonical_bytes

    return canonical_bytes(document)
