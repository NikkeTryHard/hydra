"""Public-state hash chain over actor-visible events (stdlib-light split).

Owns the chained state digest the packet partitioner stamps before/after
every packet: one fold step per public event, plus the incremental prefix
form replacing per-segment full re-walks. Split from
:mod:`hydra2.contracts.event_packet` under the module-size ceiling; the fold
bytes are unchanged (byte-identical digests, same bridge-free ``hashlib``
path). Failure mode: a caller folding non-public events, or reordering the
stream, silently binds a different chain — inputs must arrive in stream
order with visibility already gated.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from hydra2.artifacts.canonical import canonical_bytes as canonical_json_bytes
from hydra2.contracts.common import DigestText
from hydra2.contracts.event_envelope import envelope_identity_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.event_envelope import EventEnvelope
__all__ = [
    "public_state_chain_hash",
    "public_state_chain_hash_prefixes",
]

#: Chain digest of the empty prefix (constant anchor for every stream).
_EMPTY_CHAIN_DIGEST = DigestText("sha256:" + hashlib.sha256(b"").hexdigest())


def _fold_chain_digest(prefix: DigestText, event: EventEnvelope) -> DigestText:
    """One public fold step (shared by the single and prefix-batch chain paths)."""
    fold_doc = {"prefix": str(prefix), "event": envelope_identity_document(event)}
    fold_bytes = canonical_json_bytes(fold_doc)
    return DigestText("sha256:" + hashlib.sha256(fold_bytes).hexdigest())


def public_state_chain_hash(events: Sequence[EventEnvelope]) -> DigestText:
    """Fold public event identities into a chained state hash (deterministic)."""
    digest = _EMPTY_CHAIN_DIGEST
    for event in events:
        if event.visibility == "public":
            digest = _fold_chain_digest(digest, event)
    return digest


def public_state_chain_hash_prefixes(
    events: Sequence[EventEnvelope],
) -> tuple[DigestText, ...]:
    """Chain digest after each prefix: ``out[k] == public_state_chain_hash(events[:k])``.

    Incremental single pass over the stream (same fold bytes as the single
    path, hashed with ``hashlib``): replaces the per-segment full re-walk in
    the packet partitioner (O(segments*events) serializations) with
    O(events). Non-public events carry the running digest forward unchanged.
    """
    out: list[DigestText] = [_EMPTY_CHAIN_DIGEST]
    digest = _EMPTY_CHAIN_DIGEST
    for event in events:
        if event.visibility == "public":
            digest = _fold_chain_digest(digest, event)
        out.append(digest)
    return tuple(out)
