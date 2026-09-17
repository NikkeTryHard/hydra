"""SPEC 6 action-table artifact — document, digest, and loaders.

Owns the published action-table document: the sha256 digest freezing
template indices, the deterministic builder, the envelope, and the
verified file/bytes loaders that recompute the digest before decoding.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from hydra2.contracts.action_kinds import (
    ACTION_KIND_ORDINALS,
)
from hydra2.contracts.action_model import (
    CanonicalActionTemplate,
    generate_action_templates,
    template_sort_key,
)
from hydra2.contracts.action_table import ActionTable
from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    IncompatibleSchemaError,
    SchemaVersion,
    make_schema_version,
)

__all__ = [
    "ACTION_TABLE_ARTIFACT_TYPE",
    "ACTION_TABLE_RELPATH",
    "ACTION_TABLE_SCHEMA_VERSION",
    "action_table_envelope",
    "build_action_table",
    "load_action_table",
]

# ---------------------------------------------------------------------------
# Versioned artifact document (configs/contracts/action_table_v1.json).
# ---------------------------------------------------------------------------

ACTION_TABLE_ARTIFACT_TYPE = "hydra2.action_table"
ACTION_TABLE_SCHEMA_VERSION = SchemaVersion("1.0.0")
ACTION_TABLE_RELPATH = Path("configs") / "contracts" / "action_table_v1.json"

_TEMPLATE_JSON_FIELDS = (
    "called_tile",
    "consumed_tiles",
    "declares_riichi",
    "kind",
    "meld_ref_required",
    "source_offset",
    "tile",
)


def _reject_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _template_to_json(template: CanonicalActionTemplate) -> dict[str, object]:
    return {
        "called_tile": template.called_tile,
        "consumed_tiles": list(template.consumed_tiles),
        "declares_riichi": template.declares_riichi,
        "kind": template.kind,
        "meld_ref_required": template.meld_ref_required,
        "source_offset": template.source_offset,
        "tile": template.tile,
    }


def _identity_document(schema_version: str, actions: tuple[CanonicalActionTemplate, ...]) -> dict:
    return {
        "schema_version": schema_version,
        "actions": [_template_to_json(t) for t in actions],
    }


def compute_table_digest(
    actions: tuple[CanonicalActionTemplate, ...],
    schema_version: SchemaVersion = ACTION_TABLE_SCHEMA_VERSION,
) -> DigestText:
    """sha256 over the RFC 8785 canonical bytes of the digest-free payload."""
    identity = canonical_json_bytes(_identity_document(schema_version, actions))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def build_action_table(
    actions: tuple[CanonicalActionTemplate, ...] | None = None,
    *,
    schema_version: SchemaVersion = ACTION_TABLE_SCHEMA_VERSION,
) -> ActionTable:
    """Deterministically build the versioned table (digest included)."""
    ordered = tuple(
        sorted(
            actions if actions is not None else generate_action_templates(), key=template_sort_key
        )
    )
    return ActionTable(
        schema_version=schema_version,
        actions=ordered,
        digest=compute_table_digest(ordered, schema_version),
    )


def action_table_envelope(
    table: ActionTable, *, compatibility: Literal["exact", "backward_read"] = "exact"
) -> dict:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    if compatibility != "exact":
        raise ContractError("action_table_v1 publishes exact compatibility only")
    payload = _identity_document(table.schema_version, table.actions)
    payload["digest"] = table.digest
    return {
        "artifact_type": ACTION_TABLE_ARTIFACT_TYPE,
        "schema_version": table.schema_version,
        "compatibility": compatibility,
        "payload": payload,
    }


def load_action_table(path: Path) -> ActionTable:
    """Parse and fully verify an action-table artifact; returns the table.

    Recomputes the digest before decoding the semantic payload (SPEC 2.2),
    validates the envelope, template ordering, and cross-checks the declared
    digest against a fresh :func:`build_action_table`.
    """
    raw_bytes = Path(path).read_bytes()
    try:
        document = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: not valid JSON: {exc}") from exc
    except ContractError:
        raise
    return _table_from_document(document, origin=str(path))


def _table_from_document(document: object, *, origin: str) -> ActionTable:
    if not isinstance(document, Mapping):
        raise ContractError(f"{origin}: artifact must be a JSON object")
    if document.get("artifact_type") != ACTION_TABLE_ARTIFACT_TYPE:
        raise ContractError(
            f"{origin}: artifact_type must be {ACTION_TABLE_ARTIFACT_TYPE!r}, "
            f"got {document.get('artifact_type')!r}"
        )
    compatibility = document.get("compatibility")
    if compatibility != "exact":
        raise ContractError(f"{origin}: unsupported compatibility {compatibility!r}")
    envelope_version = document.get("schema_version")
    if not isinstance(envelope_version, str):
        raise ContractError(f"{origin}: schema_version must be a string")
    if envelope_version.split(".")[0] != ACTION_TABLE_SCHEMA_VERSION.split(".")[0]:
        raise IncompatibleSchemaError(
            f"{origin}: unknown major schema version {envelope_version!r}"
        )
    if envelope_version != ACTION_TABLE_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"{origin}: schema_version {envelope_version!r} newer than supported "
            f"{ACTION_TABLE_SCHEMA_VERSION!r}"
        )
    payload = document.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "actions", "digest"}:
        raise ContractError(f"{origin}: payload must hold exactly schema_version/actions/digest")
    if payload["schema_version"] != envelope_version:
        raise ContractError(
            f"{origin}: payload schema_version {payload['schema_version']!r} != envelope "
            f"{envelope_version!r}"
        )
    declared_digest: object = payload["digest"]
    if not isinstance(declared_digest, str):
        raise ContractError(f"{origin}: digest must be a string")

    raw_actions: object = payload["actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) == 0:
        raise ContractError(f"{origin}: actions must be a non-empty array")
    templates = tuple(
        _template_from_json(entry, origin=origin)  # pyrefly: ignore[unknown-argument-type]  # reason: raw_actions list-checked above; element type unchecked until _template_from_json validates
        for entry in raw_actions
    )  # type: ignore[attr-defined]  # reason: raw_actions isinstance-checked as list above; checker cannot narrow object
    recomputed = compute_table_digest(templates, make_schema_version(envelope_version))
    if not hmac.compare_digest(str(recomputed), declared_digest):
        raise DigestMismatchError(
            f"{origin}: declared digest {declared_digest!r} != recomputed {recomputed!r}"
        )
    table = build_action_table(templates, schema_version=make_schema_version(envelope_version))
    if not hmac.compare_digest(str(table.digest), declared_digest):
        raise DigestMismatchError(
            f"{origin}: rebuilt table digest {table.digest!r} != declared {declared_digest!r}"
        )
    return table


def _template_from_json(entry: object, *, origin: str) -> CanonicalActionTemplate:
    if not isinstance(entry, Mapping) or set(entry) != set(_TEMPLATE_JSON_FIELDS):
        raise ContractError(
            f"{origin}: template entries must hold exactly {list(_TEMPLATE_JSON_FIELDS)}"
        )
    consumed_raw: object = entry["consumed_tiles"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if not isinstance(consumed_raw, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in consumed_raw
    ):
        raise ContractError(f"{origin}: consumed_tiles must be an array of ints")
    for name in ("tile", "called_tile"):
        value: object = entry[name]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ContractError(f"{origin}: {name} must be null or int")
    offset: object = entry["source_offset"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if offset is not None and (isinstance(offset, bool) or offset not in (-1, 0, 1, 2)):  # pyrefly: ignore[unknown-argument-type]  # reason: offset None/bool-filtered here; membership check is the validation
        raise ContractError(f"{origin}: source_offset invalid: {offset!r}")
    kind: object = entry["kind"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if not isinstance(kind, str) or kind not in ACTION_KIND_ORDINALS:
        raise ContractError(f"{origin}: unknown template kind {kind!r}")
    for name in ("declares_riichi", "meld_ref_required"):
        if not isinstance(entry[name], bool):
            raise ContractError(f"{origin}: {name} must be a bool")
    return CanonicalActionTemplate(
        kind=kind,  # type: ignore[arg-type]  # reason: kind isinstance-checked as str against ACTION_KIND_ORDINALS above
        tile=entry["tile"],  # type: ignore[arg-type]  # reason: null-or-int checked above; ctor re-validates
        called_tile=entry["called_tile"],  # type: ignore[arg-type]  # reason: null-or-int checked above; ctor re-validates
        consumed_tiles=tuple(consumed_raw),  # type: ignore[arg-type]  # reason: list-of-ints checked above
        source_offset=offset,  # type: ignore[arg-type]  # reason: offset range-checked above
        declares_riichi=entry["declares_riichi"],  # pyrefly: ignore[unknown-argument-type]  # reason: bool-checked in the loop above; ctor re-validates
        meld_ref_required=entry["meld_ref_required"],  # pyrefly: ignore[unknown-argument-type]  # reason: bool-checked in the loop above; ctor re-validates
    )
