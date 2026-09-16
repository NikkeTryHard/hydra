"""Candidate 6 Gumbel search — deterministic root Gumbels, sequential halving, exact rules.

Re-export facade over the split modules: :mod:`hydra2.search.gumbel_core`
(firewall vocabulary, deterministic roots, vectors, simulator),
:mod:`hydra2.search.gumbel_config` (frozen configs, per-action stats),
:mod:`hydra2.search.gumbel_search` (continuation policy plus the
:class:`GumbelSearchPlanner` construction/search driver),
:mod:`hydra2.search.gumbel_act` (:class:`GumbelSearchPlanner` Planner
protocol adapter), :mod:`hydra2.search.gumbel_puct` (matched PUCT
comparator), and :mod:`hydra2.search.gumbel_spec` (CandidateSpec
factories). Import from this path; public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.gumbel_act import GumbelSearchPlanner as GumbelSearchPlanner
from hydra2.search.gumbel_act import GumbelSearchPlannerActMixin as GumbelSearchPlannerActMixin
from hydra2.search.gumbel_config import GumbelSearchConfig as GumbelSearchConfig
from hydra2.search.gumbel_config import PuctConfig as PuctConfig
from hydra2.search.gumbel_core import FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY
from hydra2.search.gumbel_core import cached_full_history_agreement as cached_full_history_agreement
from hydra2.search.gumbel_core import deterministic_gumbel as deterministic_gumbel
from hydra2.search.gumbel_core import deterministic_root_gumbels as deterministic_root_gumbels
from hydra2.search.gumbel_core import exact_transition as exact_transition
from hydra2.search.gumbel_core import info_key_for_observation as info_key_for_observation
from hydra2.search.gumbel_core import (
    learned_rules_transition_rejected as learned_rules_transition_rejected,
)
from hydra2.search.gumbel_core import model_vector_for_world as model_vector_for_world
from hydra2.search.gumbel_core import scalarize_vector as scalarize_vector
from hydra2.search.gumbel_core import terminal_vector_for_world as terminal_vector_for_world
from hydra2.search.gumbel_core import (
    validate_hidden_permutation_invariance as validate_hidden_permutation_invariance,
)
from hydra2.search.gumbel_core import validate_packet_partition as validate_packet_partition
from hydra2.search.gumbel_puct import PuctBaselinePlanner as PuctBaselinePlanner
from hydra2.search.gumbel_search import (
    GumbelSearchPlannerSearchMixin as GumbelSearchPlannerSearchMixin,
)
from hydra2.search.gumbel_spec import make_gumbel_candidate_spec as make_gumbel_candidate_spec
from hydra2.search.gumbel_spec import make_puct_candidate_spec as make_puct_candidate_spec

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "GumbelSearchConfig",
    "GumbelSearchPlanner",
    "PuctBaselinePlanner",
    "PuctConfig",
    "cached_full_history_agreement",
    "deterministic_gumbel",
    "deterministic_root_gumbels",
    "exact_transition",
    "info_key_for_observation",
    "learned_rules_transition_rejected",
    "make_gumbel_candidate_spec",
    "make_puct_candidate_spec",
    "model_vector_for_world",
    "scalarize_vector",
    "terminal_vector_for_world",
    "validate_hidden_permutation_invariance",
    "validate_packet_partition",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.gumbel_config import _ActionStats as _ActionStats
from hydra2.search.gumbel_core import _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR
from hydra2.search.gumbel_core import _GUMBEL_SEED_DOMAIN as _GUMBEL_SEED_DOMAIN
from hydra2.search.gumbel_core import _MASTER_SEED as _MASTER_SEED
from hydra2.search.gumbel_core import _RANDOM_IMPORT_ERROR as _RANDOM_IMPORT_ERROR
from hydra2.search.gumbel_core import _TELEMETRY_IMPORT_ERROR as _TELEMETRY_IMPORT_ERROR
from hydra2.search.gumbel_core import DEPLOYABLE_DEADLINE_MS as DEPLOYABLE_DEADLINE_MS
from hydra2.search.gumbel_core import REPO_ROOT as REPO_ROOT
from hydra2.search.gumbel_core import U64_DENOM as U64_DENOM
from hydra2.search.gumbel_core import ActorObservation as ActorObservation
from hydra2.search.gumbel_core import Any as Any
from hydra2.search.gumbel_core import BeliefEpoch as BeliefEpoch
from hydra2.search.gumbel_core import CandidateSpec as CandidateSpec
from hydra2.search.gumbel_core import ContractError as ContractError
from hydra2.search.gumbel_core import DigestText as DigestText
from hydra2.search.gumbel_core import FullWorld as FullWorld
from hydra2.search.gumbel_core import Literal as Literal
from hydra2.search.gumbel_core import NaturalBelief as NaturalBelief
from hydra2.search.gumbel_core import Planner as Planner
from hydra2.search.gumbel_core import RandomStream as RandomStream
from hydra2.search.gumbel_core import ResourceBudget as ResourceBudget
from hydra2.search.gumbel_core import ResourceTelemetry as ResourceTelemetry
from hydra2.search.gumbel_core import SearchRequest as SearchRequest
from hydra2.search.gumbel_core import SearchResult as SearchResult
from hydra2.search.gumbel_core import UtilityVector as UtilityVector
from hydra2.search.gumbel_core import VisibilityViolationError as VisibilityViolationError
from hydra2.search.gumbel_core import _actor_to_move as _actor_to_move
from hydra2.search.gumbel_core import _is_terminal as _is_terminal
from hydra2.search.gumbel_core import _legal_ids_for_observation as _legal_ids_for_observation
from hydra2.search.gumbel_core import canonical_bytes as canonical_bytes
from hydra2.search.gumbel_core import cast as cast
from hydra2.search.gumbel_core import dataclass as dataclass
from hydra2.search.gumbel_core import field as field
from hydra2.search.gumbel_core import hashlib as hashlib
from hydra2.search.gumbel_core import logger as logger
from hydra2.search.gumbel_core import logging as logging
from hydra2.search.gumbel_core import make_digest_text as make_digest_text
from hydra2.search.gumbel_core import make_full_world as make_full_world
from hydra2.search.gumbel_core import make_resource_telemetry as make_resource_telemetry
from hydra2.search.gumbel_core import math as math
from hydra2.search.gumbel_core import observation_identity_document as observation_identity_document
from hydra2.search.gumbel_core import time as time
from hydra2.search.gumbel_core import world_actor_observation as world_actor_observation
from hydra2.search.gumbel_spec import _canonical_hashes as _canonical_hashes
from hydra2.search.gumbel_spec import _derive_utility_manifest_hash as _derive_utility_manifest_hash
from hydra2.search.gumbel_spec import _load_default_hashes as _load_default_hashes
from hydra2.search.gumbel_spec import _model_hash_from_identity as _model_hash_from_identity
