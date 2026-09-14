"""SPEC 6 canonical action vocabulary: kinds, invariants, table, codec.

Re-export facade over the split modules: :mod:`hydra2.contracts.action_kinds`
(frozen kind ordinals, metadata domain, phase gating, tile validators),
:mod:`hydra2.contracts.action_model` (validated action record, template with
its lexicographic order, census),
:mod:`hydra2.contracts.action_table` (indexed table, context, bidirectional
codec), and :mod:`hydra2.contracts.action_artifact` (versioned table document
with digest and loaders). Import from this path; it preserves every public
name and ``__all__``.
"""

from __future__ import annotations

from hydra2.contracts.action_artifact import (
    ACTION_TABLE_ARTIFACT_TYPE as ACTION_TABLE_ARTIFACT_TYPE,
)
from hydra2.contracts.action_artifact import (
    ACTION_TABLE_RELPATH as ACTION_TABLE_RELPATH,
)
from hydra2.contracts.action_artifact import (
    ACTION_TABLE_SCHEMA_VERSION as ACTION_TABLE_SCHEMA_VERSION,
)
from hydra2.contracts.action_artifact import (
    action_table_envelope as action_table_envelope,
)
from hydra2.contracts.action_artifact import (
    build_action_table as build_action_table,
)
from hydra2.contracts.action_artifact import (
    load_action_table as load_action_table,
)
from hydra2.contracts.action_kinds import (
    ACTION_KIND_ORDINALS as ACTION_KIND_ORDINALS,
)
from hydra2.contracts.action_kinds import (
    ACTION_KINDS as ACTION_KINDS,
)
from hydra2.contracts.action_kinds import (
    ACTION_PHASES as ACTION_PHASES,
)
from hydra2.contracts.action_kinds import (
    CLAIM_KINDS as CLAIM_KINDS,
)
from hydra2.contracts.action_kinds import (
    KAKAN_METADATA_KEYS as KAKAN_METADATA_KEYS,
)
from hydra2.contracts.action_kinds import (
    MELD_KINDS as MELD_KINDS,
)
from hydra2.contracts.action_kinds import (
    METADATA_KEYS_BY_KIND as METADATA_KEYS_BY_KIND,
)
from hydra2.contracts.action_kinds import (
    PHASES as PHASES,
)
from hydra2.contracts.action_kinds import (
    JsonValue as JsonValue,
)
from hydra2.contracts.action_kinds import (
    Phase as Phase,
)
from hydra2.contracts.action_kinds import (
    VisibleMeld as VisibleMeld,
)
from hydra2.contracts.action_kinds import (
    visible_meld_id as visible_meld_id,
)
from hydra2.contracts.action_model import (
    CanonicalAction as CanonicalAction,
)
from hydra2.contracts.action_model import (
    CanonicalActionTemplate as CanonicalActionTemplate,
)
from hydra2.contracts.action_model import (
    generate_action_templates as generate_action_templates,
)
from hydra2.contracts.action_model import (
    template_sort_key as template_sort_key,
)
from hydra2.contracts.action_table import (
    ActionContext as ActionContext,
)
from hydra2.contracts.action_table import (
    ActionTable as ActionTable,
)
from hydra2.contracts.action_table import (
    CanonicalActionCodec as CanonicalActionCodec,
)
from hydra2.contracts.action_table import (
    canonical_action_codec as canonical_action_codec,
)

__all__ = [
    "ACTION_KINDS",
    "ACTION_KIND_ORDINALS",
    "ACTION_PHASES",
    "ACTION_TABLE_ARTIFACT_TYPE",
    "ACTION_TABLE_RELPATH",
    "ACTION_TABLE_SCHEMA_VERSION",
    "CLAIM_KINDS",
    "KAKAN_METADATA_KEYS",
    "MELD_KINDS",
    "METADATA_KEYS_BY_KIND",
    "PHASES",
    "ActionContext",
    "ActionTable",
    "CanonicalAction",
    "CanonicalActionCodec",
    "CanonicalActionTemplate",
    "JsonValue",
    "Phase",
    "VisibleMeld",
    "action_table_envelope",
    "build_action_table",
    "canonical_action_codec",
    "generate_action_templates",
    "load_action_table",
    "template_sort_key",
    "visible_meld_id",
]

# Names importable from this path before the split that live in the
# submodules now (kept so engine helpers and type-checking imports resolve).
from hydra2.contracts.action_artifact import compute_table_digest as compute_table_digest
from hydra2.contracts.action_artifact import parse_action_table as parse_action_table
from hydra2.contracts.action_kinds import ActionKind as ActionKind
from hydra2.contracts.action_model import SourceOffset as SourceOffset
from hydra2.contracts.action_table import ActionCodec as ActionCodec
from hydra2.contracts.canonical import canonical_json_bytes as canonical_json_bytes
