# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — public-history subgame, information-set strategies.

Re-export facade over the split modules: :mod:`hydra2.search.local_shared`
(seeds, feature flags, guarded fallbacks),
:mod:`hydra2.search.local_abstraction` (information keys, vectors, abstraction,
subgame declaration), :mod:`hydra2.search.local_strategy` (strategy tables,
update rules, leaf replay, exhaustive reference),
:mod:`hydra2.search.local_spec` (frozen config, CandidateSpec factory),
:mod:`hydra2.search.local_search` (resolving loop), and
:mod:`hydra2.search.local_act` (Planner act plus
:class:`LocalResolvingPlanner`). Import from this path; it preserves every
public name and ``__all__``.

Implements Blueprint §12 (Candidate 5) and SPEC 16.6:

- Tables key ``(actor, information_node_hash)``, never root world.
- Every actor update uses only that actor's information set.
- Return vectors remain four-seat, settlement-preserving, exact.
- Subgame horizon, abstraction, leaf model, update, iteration count, and averaging
  are CandidateSpec fields and frozen.
- Cycle and abstraction failure abort candidate.
- Output is empirical optimizer result; never equilibrium or exploitability certificate.
- PBRF warm start variant compared without claiming superiority.

Deterministic, CPU-only, no hidden-state leakage. All randomness via semantic
counter-based seeds derived from ``(case_id, root_seat, candidate_id)``.
"""

from __future__ import annotations

from hydra2.search.local_abstraction import AbstractMappingError as AbstractMappingError
from hydra2.search.local_abstraction import CycleDetectedError as CycleDetectedError
from hydra2.search.local_abstraction import LocalResolvingAbstraction as LocalResolvingAbstraction
from hydra2.search.local_abstraction import PublicSubgame as PublicSubgame
from hydra2.search.local_abstraction import _actor_to_key as _actor_to_key
from hydra2.search.local_abstraction import _digest as _digest
from hydra2.search.local_abstraction import _h as _h
from hydra2.search.local_abstraction import _seed_bytes as _seed_bytes
from hydra2.search.local_abstraction import abstraction_round_trip as abstraction_round_trip
from hydra2.search.local_abstraction import build_public_subgame as build_public_subgame
from hydra2.search.local_abstraction import detect_cycle as detect_cycle
from hydra2.search.local_abstraction import (
    info_key_for_actor_observation as info_key_for_actor_observation,
)
from hydra2.search.local_abstraction import model_vector_for_world as model_vector_for_world
from hydra2.search.local_abstraction import preserves_vector_returns as preserves_vector_returns
from hydra2.search.local_abstraction import terminal_vector_for_world as terminal_vector_for_world
from hydra2.search.local_abstraction import (
    validate_abstraction_mapping as validate_abstraction_mapping,
)
from hydra2.search.local_act import LocalResolvingPlanner as LocalResolvingPlanner
from hydra2.search.local_act import (
    LocalResolvingPlannerActMixin as LocalResolvingPlannerActMixin,
)
from hydra2.search.local_search import (
    LocalResolvingPlannerSearchMixin as LocalResolvingPlannerSearchMixin,
)
from hydra2.search.local_spec import LocalResolvingConfig as LocalResolvingConfig
from hydra2.search.local_spec import (
    _build_abstraction_from_config as _build_abstraction_from_config,
)
from hydra2.search.local_spec import _derive_utility_manifest_hash as _derive_utility_manifest_hash
from hydra2.search.local_spec import _file_sha256 as _file_sha256
from hydra2.search.local_spec import _load_default_hashes as _load_default_hashes
from hydra2.search.local_spec import make_candidate5_spec as make_candidate5_spec
from hydra2.search.local_strategy import StrategyTable as StrategyTable
from hydra2.search.local_strategy import _fictitious_play_update as _fictitious_play_update
from hydra2.search.local_strategy import _hedge_update as _hedge_update
from hydra2.search.local_strategy import _regret_matching_update as _regret_matching_update
from hydra2.search.local_strategy import apply_update as apply_update
from hydra2.search.local_strategy import averaging_weights as averaging_weights
from hydra2.search.local_strategy import (
    exhaustive_tiny_game_values as exhaustive_tiny_game_values,
)
from hydra2.search.local_strategy import frozen_averaging_rule_names as frozen_averaging_rule_names
from hydra2.search.local_strategy import frozen_update_rule_names as frozen_update_rule_names
from hydra2.search.local_strategy import is_equilibrium_claimed as is_equilibrium_claimed
from hydra2.search.local_strategy import leaf_vector_replay as leaf_vector_replay
from hydra2.search.local_strategy import make_uniform_strategy as make_uniform_strategy

__all__ = [
    "AbstractMappingError",
    "CycleDetectedError",
    "LocalResolvingAbstraction",
    "LocalResolvingConfig",
    "LocalResolvingPlanner",
    "PublicSubgame",
    "StrategyTable",
    "abstraction_round_trip",
    "build_public_subgame",
    "detect_cycle",
    "info_key_for_actor_observation",
    "is_equilibrium_claimed",
    "leaf_vector_replay",
    "make_candidate5_spec",
    "model_vector_for_world",
    "terminal_vector_for_world",
    "validate_abstraction_mapping",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.local_shared import _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR
from hydra2.search.local_shared import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.local_shared import _CONTRACTS_IMPORT_ERROR as _CONTRACTS_IMPORT_ERROR
from hydra2.search.local_shared import _MASTER_SEED as _MASTER_SEED
from hydra2.search.local_shared import _OBS_IMPORT_ERROR as _OBS_IMPORT_ERROR
from hydra2.search.local_shared import _RANDOM_IMPORT_ERROR as _RANDOM_IMPORT_ERROR
from hydra2.search.local_shared import DEPLOYABLE_DEADLINE_MS as DEPLOYABLE_DEADLINE_MS
from hydra2.search.local_shared import FORBIDDEN_IN_STRATEGY_KEY as FORBIDDEN_IN_STRATEGY_KEY
from hydra2.search.local_shared import Any as Any
from hydra2.search.local_shared import BeliefEpoch as BeliefEpoch
from hydra2.search.local_shared import CandidateSpec as CandidateSpec
from hydra2.search.local_shared import ContractError as ContractError
from hydra2.search.local_shared import DigestText as DigestText
from hydra2.search.local_shared import Literal as Literal
from hydra2.search.local_shared import NaturalBelief as NaturalBelief
from hydra2.search.local_shared import Planner as Planner
from hydra2.search.local_shared import RandomStream as RandomStream
from hydra2.search.local_shared import ResourceBudget as ResourceBudget
from hydra2.search.local_shared import SearchRequest as SearchRequest
from hydra2.search.local_shared import SearchResult as SearchResult
from hydra2.search.local_shared import canonical_bytes as canonical_bytes
from hydra2.search.local_shared import cast as cast
from hydra2.search.local_shared import dataclass as dataclass
from hydra2.search.local_shared import field as field
from hydra2.search.local_shared import hashlib as hashlib
from hydra2.search.local_shared import json as json
from hydra2.search.local_shared import logger as logger
from hydra2.search.local_shared import logging as logging
from hydra2.search.local_shared import make_actor_observation as make_actor_observation
from hydra2.search.local_shared import make_digest_text as make_digest_text
from hydra2.search.local_shared import make_random_stream_key as make_random_stream_key
from hydra2.search.local_shared import math as math
from hydra2.search.local_shared import semantic_seed as semantic_seed
from hydra2.search.local_shared import time as time
from hydra2.search.local_strategy import _VALID_AVERAGING as _VALID_AVERAGING
from hydra2.search.local_strategy import _VALID_UPDATE_RULES as _VALID_UPDATE_RULES
