"""Candidate 2 natural DESPOT — natural scenarios only.

Re-export facade over the split modules: :mod:`hydra2.search.despot_core`
(scenarios, packet guards, seeding, spec factory),
:mod:`hydra2.search.despot_search` (search policy),
:mod:`hydra2.search.despot_act` (expansion loop plus
:class:`NaturalDespotPlanner`), and
:mod:`hydra2.search.despot_result` (result assembly). Import from this path;
public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.despot_act import NaturalDespotPlanner as NaturalDespotPlanner
from hydra2.search.despot_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.despot_core import DespotConfig as DespotConfig
from hydra2.search.despot_core import NaturalScenario as NaturalScenario
from hydra2.search.despot_core import _DespotNode as _DespotNode
from hydra2.search.despot_core import _hash_tie_break as _hash_tie_break
from hydra2.search.despot_core import _scenario_seed_bytes as _scenario_seed_bytes
from hydra2.search.despot_core import make_despot_candidate_spec as make_despot_candidate_spec
from hydra2.search.despot_core import packet_aliasing_rejected as packet_aliasing_rejected
from hydra2.search.despot_core import proposal_reversal_fixture as proposal_reversal_fixture
from hydra2.search.despot_core import validate_packet_partition as validate_packet_partition
from hydra2.search.despot_result import budget_exhausted_for_test as budget_exhausted_for_test

__all__ = [
    "DespotConfig",
    "NaturalDespotPlanner",
    "NaturalScenario",
    "packet_aliasing_rejected",
    "proposal_reversal_fixture",
    "validate_packet_partition",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.common import Planner as Planner
from hydra2.search.common import SearchRequest as SearchRequest
from hydra2.search.common import SearchResult as SearchResult
from hydra2.search.despot_act import NaturalDespotPlannerActMixin as NaturalDespotPlannerActMixin
from hydra2.search.despot_core import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.despot_core import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.despot_core import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.despot_core import _HAS_TELEMETRY as _HAS_TELEMETRY
from hydra2.search.despot_core import _HAS_UTILITY as _HAS_UTILITY
from hydra2.search.despot_core import Any as Any
from hydra2.search.despot_core import BeliefEpoch as BeliefEpoch
from hydra2.search.despot_core import CandidateSpec as CandidateSpec
from hydra2.search.despot_core import ContractError as ContractError
from hydra2.search.despot_core import Literal as Literal
from hydra2.search.despot_core import NaturalBelief as NaturalBelief
from hydra2.search.despot_core import NaturalPacketKernel as NaturalPacketKernel
from hydra2.search.despot_core import PacketPartitionError as PacketPartitionError
from hydra2.search.despot_core import RandomStream as RandomStream
from hydra2.search.despot_core import ResourceBudget as ResourceBudget
from hydra2.search.despot_core import ResourceTelemetry as ResourceTelemetry
from hydra2.search.despot_core import UtilityVector as UtilityVector
from hydra2.search.despot_core import _default_budget as _default_budget
from hydra2.search.despot_core import canonical_bytes as canonical_bytes
from hydra2.search.despot_core import cast as cast
from hydra2.search.despot_core import dataclass as dataclass
from hydra2.search.despot_core import field as field
from hydra2.search.despot_core import hashlib as hashlib
from hydra2.search.despot_core import logger as logger
from hydra2.search.despot_core import logging as logging
from hydra2.search.despot_core import make_resource_telemetry as make_resource_telemetry
from hydra2.search.despot_core import math as math
from hydra2.search.despot_result import (
    NaturalDespotPlannerResultMixin as NaturalDespotPlannerResultMixin,
)
from hydra2.search.despot_search import (
    NaturalDespotPlannerSearchMixin as NaturalDespotPlannerSearchMixin,
)
from hydra2.search.despot_search import time as time
