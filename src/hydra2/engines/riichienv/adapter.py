"""RiichiEnvExactSimulator: the WP-03A reference ExactSimulator.

Re-export facade over the split modules:
:mod:`hydra2.engines.riichienv.adapter_identity` (design notes, shared
caches, rules gates), :mod:`hydra2.engines.riichienv.adapter_core`
(public API, construction, apply, persistence),
:mod:`hydra2.engines.riichienv.adapter_step` (decision core),
:mod:`hydra2.engines.riichienv.adapter_events_a` (round and call
handlers), and :mod:`hydra2.engines.riichienv.adapter_events_b`
(outcomes and exact-tile helpers). Import from this path; it preserves
every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.engines.riichienv.adapter_core import RiichiEnvExactSimulator as RiichiEnvExactSimulator
from hydra2.engines.riichienv.adapter_events_a import AdapterEventsAMixin as AdapterEventsAMixin
from hydra2.engines.riichienv.adapter_events_b import AdapterEventsBMixin as AdapterEventsBMixin
from hydra2.engines.riichienv.adapter_identity import _AT as _AT
from hydra2.engines.riichienv.adapter_identity import _BAKAZE_TO_TILE_TYPE as _BAKAZE_TO_TILE_TYPE
from hydra2.engines.riichienv.adapter_identity import _EVENT_SCHEMA_CACHE as _EVENT_SCHEMA_CACHE
from hydra2.engines.riichienv.adapter_identity import _TABLE_CACHE as _TABLE_CACHE
from hydra2.engines.riichienv.adapter_identity import _action_table as _action_table
from hydra2.engines.riichienv.adapter_identity import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv.adapter_identity import _rules_identity as _rules_identity
from hydra2.engines.riichienv.adapter_identity import _validate_rules as _validate_rules
from hydra2.engines.riichienv.adapter_step import AdapterStepMixin as AdapterStepMixin

__all__ = [
    "RiichiEnvExactSimulator",
]
