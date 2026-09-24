"""Natural-belief target identity derivation (split half)."""

from __future__ import annotations

import hashlib

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes_batch
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import DigestText, Seat

__all__ = [
    "_target_digest",
    "_target_doc_for",
    "_target_id_for",
    "_target_ids_for",
    "_target_seat",
]


def _target_digest(value: DigestText | str) -> DigestText:
    """Validate one digest via the bridge (precise return, no str() wrap)."""
    validated: DigestText = _bridge_contracts.make_digest_text(value)
    return validated


def _target_seat(value: Seat | int) -> Seat:
    """Validate one seat via the bridge (precise return, no int() wrap)."""
    checked: Seat = _bridge_contracts.make_seat(value)
    return checked


def _target_id_for(
    *,
    observation_hash: DigestText,
    rules_hash: DigestText,
    belief_model_hash: DigestText,
    event_model_hash: DigestText,
    proposal_spec_hash: DigestText,
) -> DigestText:
    doc = _target_doc_for(
        observation_hash=observation_hash,
        rules_hash=rules_hash,
        belief_model_hash=belief_model_hash,
        event_model_hash=event_model_hash,
        proposal_spec_hash=proposal_spec_hash,
    )
    # Digest line via the canon bridge (byte-identical to the retired
    # hashlib-over-canonical_bytes loop; ImportError with build-ext hint,
    # no oracle fallback). Math/doc shape untouched.
    return of_canonical(doc)


def _target_ids_for(
    docs: list[dict[str, DigestText]],
) -> list[DigestText]:
    """Target digests for pre-built identity docs via ONE bridge FFI.

    Byte-identical to ``[_target_id_for(**kw) for ...]`` built from the same
    field values: the docs serialize in one ``canonical_bytes_batch``
    call, then hash with ``hashlib`` exactly like the single ``of_canonical``
    line (Python-framed + bridge-hashed). Empty input returns ``[]`` without
    touching the bridge. Bridge rejects raise, never silent.
    """
    if len(docs) == 0:
        return []
    blobs = canonical_bytes_batch([dict(doc) for doc in docs])
    return [DigestText("sha256:" + hashlib.sha256(blob).hexdigest()) for blob in blobs]


def _target_doc_for(
    *,
    observation_hash: DigestText,
    rules_hash: DigestText,
    belief_model_hash: DigestText,
    event_model_hash: DigestText,
    proposal_spec_hash: DigestText,
) -> dict[str, DigestText]:
    """Target identity doc (shared by the single and batch digest paths)."""
    return {
        "observation_hash": observation_hash,
        "rules_hash": rules_hash,
        "belief_model_hash": belief_model_hash,
        "event_model_hash": event_model_hash,
        "proposal_spec_hash": proposal_spec_hash,
    }
