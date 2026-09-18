"""Candidate 1 natural ISMCTS — natural worlds only, vector backup.

Re-export facade over the split modules: :mod:`hydra2.search.ismcts_core`
(firewall vocabulary, frozen config, tree nodes, continuation policy, UCT
selection), :mod:`hydra2.search.ismcts_search` (trajectory predicates,
construction, particle materialization, Rust-batch search driver), and
:mod:`hydra2.search.ismcts_act` (Planner protocol adapter
plus the double-weighting oracle). Import from this path; it preserves
every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.ismcts_act import (
    NaturalISMCTSPlanner as NaturalISMCTSPlanner,
)
from hydra2.search.ismcts_act import (
    NaturalISMCTSPlannerActMixin as NaturalISMCTSPlannerActMixin,
)
from hydra2.search.ismcts_act import (
    double_weighting_oracle_detects_correction as double_weighting_oracle_detects_correction,
)
from hydra2.search.ismcts_core import (
    FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY,
)
from hydra2.search.ismcts_core import (
    InformationSetNode as InformationSetNode,
)
from hydra2.search.ismcts_core import (
    NaturalISMCTSConfig as NaturalISMCTSConfig,
)
from hydra2.search.ismcts_core import (
    UniformContinuationPolicy as UniformContinuationPolicy,
)
from hydra2.search.ismcts_core import (
    attempt_redeterminize as attempt_redeterminize,
)
from hydra2.search.ismcts_core import (
    info_key_for_observation as info_key_for_observation,
)
from hydra2.search.ismcts_core import (
    is_redeterminization_enabled as is_redeterminization_enabled,
)
from hydra2.search.ismcts_core import (
    model_vector_for_world as model_vector_for_world,
)
from hydra2.search.ismcts_core import (
    scalarize_vector as scalarize_vector,
)
from hydra2.search.ismcts_core import (
    terminal_vector_for_world as terminal_vector_for_world,
)
from hydra2.search.ismcts_core import (
    validate_tree_keys_contain_no_world_id as validate_tree_keys_contain_no_world_id,
)
from hydra2.search.ismcts_search import (
    NaturalISMCTSPlannerSearchMixin as NaturalISMCTSPlannerSearchMixin,
)

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "InformationSetNode",
    "NaturalISMCTSConfig",
    "NaturalISMCTSPlanner",
    "UniformContinuationPolicy",
    "info_key_for_observation",
    "is_redeterminization_enabled",
    "model_vector_for_world",
    "scalarize_vector",
    "terminal_vector_for_world",
    "validate_tree_keys_contain_no_world_id",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.ismcts_core import _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR
from hydra2.search.ismcts_core import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.ismcts_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.ismcts_core import _RANDOM_IMPORT_ERROR as _RANDOM_IMPORT_ERROR
from hydra2.search.ismcts_core import _TELEMETRY_IMPORT_ERROR as _TELEMETRY_IMPORT_ERROR
from hydra2.search.ismcts_core import ActorObservation as ActorObservation
from hydra2.search.ismcts_core import Any as Any
from hydra2.search.ismcts_core import BeliefEpoch as BeliefEpoch
from hydra2.search.ismcts_core import CandidateSpec as CandidateSpec
from hydra2.search.ismcts_core import CanonicalAction as CanonicalAction
from hydra2.search.ismcts_core import ContractError as ContractError
from hydra2.search.ismcts_core import FullWorld as FullWorld
from hydra2.search.ismcts_core import Literal as Literal
from hydra2.search.ismcts_core import NaturalBelief as NaturalBelief
from hydra2.search.ismcts_core import Planner as Planner
from hydra2.search.ismcts_core import RandomStream as RandomStream
from hydra2.search.ismcts_core import ResourceBudget as ResourceBudget
from hydra2.search.ismcts_core import ResourceTelemetry as ResourceTelemetry
from hydra2.search.ismcts_core import SearchRequest as SearchRequest
from hydra2.search.ismcts_core import SearchResult as SearchResult
from hydra2.search.ismcts_core import UtilityVector as UtilityVector
from hydra2.search.ismcts_core import VisibilityViolationError as VisibilityViolationError
from hydra2.search.ismcts_core import _ActionStats as _ActionStats
from hydra2.search.ismcts_core import _uct_select as _uct_select
from hydra2.search.ismcts_core import canonical_bytes as canonical_bytes
from hydra2.search.ismcts_core import dataclass as dataclass
from hydra2.search.ismcts_core import field as field
from hydra2.search.ismcts_core import hashlib as hashlib
from hydra2.search.ismcts_core import make_full_world as make_full_world
from hydra2.search.ismcts_core import make_random_stream_key as make_random_stream_key
from hydra2.search.ismcts_core import make_resource_telemetry as make_resource_telemetry
from hydra2.search.ismcts_core import math as math
from hydra2.search.ismcts_core import observation_identity_document as observation_identity_document
from hydra2.search.ismcts_core import semantic_seed as semantic_seed
from hydra2.search.ismcts_core import world_actor_observation as world_actor_observation
from hydra2.search.ismcts_search import _actor_to_move as _actor_to_move
from hydra2.search.ismcts_search import _is_terminal as _is_terminal
from hydra2.search.ismcts_search import (
    _legal_ids_for_observation as _legal_ids_for_observation,
)
