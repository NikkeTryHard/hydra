"""Canonical identity bytes for contract artifacts — shared leaf module.

Contracts modules import stdlib and sibling contract modules only (SPEC 1), so
the RFC 8785 recipe used by every contract artifact lives here as a dependency
-free leaf. ``canonical_json_bytes`` was moved verbatim from
``hydra2.contracts.action`` (WP-02D clean cutover, owner decision D-WP02D-8);
action.py re-exports it so existing importers are unaffected. Byte equality
with the WP-02A authority ``hydra2.artifacts.canonical.canonical_bytes`` stays
pinned by tests; contracts themselves never import ``hydra2.artifacts``.
"""

from __future__ import annotations

import json

__all__ = ["canonical_json_bytes"]


def canonical_json_bytes(document: object) -> bytes:
    """RFC 8785 bytes restricted to this artifact's JSON domain.

    Contract payloads contain only nonnegative ints below 2**53,
    lowercase-hex digests, ASCII enum strings, booleans, null, arrays and
    string-keyed objects; for that domain ``json.dumps`` with sorted keys and
    tight separators reproduces RFC 8785 exactly (no float forms occur).
    """
    text = json.dumps(
        document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    return text.encode("utf-8")
