"""SPEC 6 action vocabulary — kinds, records, template census.

Re-export facade over the split modules: :mod:`hydra2.contracts.action_kinds`
(frozen kind ordinals, metadata domain, phase gating, tile validators) and
:mod:`hydra2.contracts.action_model` (validated action record, template with
its lexicographic order, census). Import from this path; it preserves every
public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.contracts.action_kinds import (
    _SOURCE_OFFSETS_CLAIM as _SOURCE_OFFSETS_CLAIM,
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
    ActionKind as ActionKind,
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
    _all_same_type as _all_same_type,
)
from hydra2.contracts.action_kinds import (
    _consumed_pair_forms_run as _consumed_pair_forms_run,
)
from hydra2.contracts.action_kinds import (
    _tile_type as _tile_type,
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
    SourceOffset as SourceOffset,
)
from hydra2.contracts.action_model import (
    generate_action_templates as generate_action_templates,
)
from hydra2.contracts.action_model import (
    template_sort_key as template_sort_key,
)

__all__ = [
    "ACTION_KINDS",
    "ACTION_KIND_ORDINALS",
    "ACTION_PHASES",
    "CLAIM_KINDS",
    "KAKAN_METADATA_KEYS",
    "MELD_KINDS",
    "METADATA_KEYS_BY_KIND",
    "PHASES",
    "CanonicalAction",
    "CanonicalActionTemplate",
    "JsonValue",
    "Phase",
    "VisibleMeld",
    "generate_action_templates",
    "template_sort_key",
]
