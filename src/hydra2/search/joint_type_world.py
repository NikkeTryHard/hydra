"""Candidate 8 joint type/world model — observation-only opponent types, joint posterior, robust set.

Re-export facade over the split modules: :mod:`hydra2.search.joint_types`
(info keys, type policy, joint particles), :mod:`hydra2.search.joint_uncertainty`
(exact oracle, coherent trajectory, uncertainty set, spec factory), and
:mod:`hydra2.search.joint_planner` (:class:`JointTypeWorldPlanner`). Import from
this path; public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.joint_planner import JointTypeWorldPlanner as JointTypeWorldPlanner
from hydra2.search.joint_types import FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY
from hydra2.search.joint_types import JointParticle as JointParticle
from hydra2.search.joint_types import JointPosterior as JointPosterior
from hydra2.search.joint_types import OpponentTypePolicy as OpponentTypePolicy
from hydra2.search.joint_types import deterministic_joint_gumbel as deterministic_joint_gumbel
from hydra2.search.joint_types import info_key_for_observation as info_key_for_observation
from hydra2.search.joint_types import (
    validate_hidden_permutation_invariance as validate_hidden_permutation_invariance,
)
from hydra2.search.joint_types import (
    validate_same_information_equality as validate_same_information_equality,
)
from hydra2.search.joint_uncertainty import JointTypeWorldConfig as JointTypeWorldConfig
from hydra2.search.joint_uncertainty import UncertaintySet as UncertaintySet
from hydra2.search.joint_uncertainty import coherent_trajectory as coherent_trajectory
from hydra2.search.joint_uncertainty import (
    exact_joint_posterior_oracle as exact_joint_posterior_oracle,
)
from hydra2.search.joint_uncertainty import hidden_marginalization as hidden_marginalization
from hydra2.search.joint_uncertainty import (
    make_joint_type_world_candidate_spec as make_joint_type_world_candidate_spec,
)
from hydra2.search.joint_uncertainty import preserve_correlation_check as preserve_correlation_check
from hydra2.search.joint_uncertainty import sequential_joint_update as sequential_joint_update

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "JointParticle",
    "JointPosterior",
    "JointTypeWorldConfig",
    "JointTypeWorldPlanner",
    "OpponentTypePolicy",
    "UncertaintySet",
    "coherent_trajectory",
    "deterministic_joint_gumbel",
    "exact_joint_posterior_oracle",
    "hidden_marginalization",
    "info_key_for_observation",
    "make_joint_type_world_candidate_spec",
    "preserve_correlation_check",
    "sequential_joint_update",
    "validate_hidden_permutation_invariance",
    "validate_same_information_equality",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.joint_types import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.joint_types import _HAS_BELIEF as _HAS_BELIEF
from hydra2.search.joint_types import _HAS_OBS as _HAS_OBS
from hydra2.search.joint_types import _HAS_RANDOM as _HAS_RANDOM
from hydra2.search.joint_types import _INFO_KEY_DOMAIN as _INFO_KEY_DOMAIN
from hydra2.search.joint_types import _JOINT_GUMBEL_DOMAIN as _JOINT_GUMBEL_DOMAIN
from hydra2.search.joint_types import _MASTER_SEED as _MASTER_SEED
from hydra2.search.joint_types import DIVERGENCE_DIRECTIONS as DIVERGENCE_DIRECTIONS
from hydra2.search.joint_types import RATIONALITY_RULES as RATIONALITY_RULES
from hydra2.search.joint_types import SUPPORT_CLASSES as SUPPORT_CLASSES
from hydra2.search.joint_types import THETA_IDS as THETA_IDS
from hydra2.search.joint_types import ActorObservation as ActorObservation
from hydra2.search.joint_types import Any as Any
from hydra2.search.joint_types import BeliefEpoch as BeliefEpoch
from hydra2.search.joint_types import CandidateSpec as CandidateSpec
from hydra2.search.joint_types import ContractError as ContractError
from hydra2.search.joint_types import FullWorld as FullWorld
from hydra2.search.joint_types import Literal as Literal
from hydra2.search.joint_types import NaturalBelief as NaturalBelief
from hydra2.search.joint_types import Planner as Planner
from hydra2.search.joint_types import RandomStream as RandomStream
from hydra2.search.joint_types import ResourceBudget as ResourceBudget
from hydra2.search.joint_types import SearchRequest as SearchRequest
from hydra2.search.joint_types import SearchResult as SearchResult
from hydra2.search.joint_types import VisibilityViolationError as VisibilityViolationError
from hydra2.search.joint_types import candidate_spec_hash as candidate_spec_hash
from hydra2.search.joint_types import canonical_bytes as canonical_bytes
from hydra2.search.joint_types import cast as cast
from hydra2.search.joint_types import dataclass as dataclass
from hydra2.search.joint_types import field as field
from hydra2.search.joint_types import hashlib as hashlib
from hydra2.search.joint_types import make_digest_text as make_digest_text
from hydra2.search.joint_types import make_full_world as make_full_world
from hydra2.search.joint_types import math as math
from hydra2.search.joint_types import observation_identity_document as observation_identity_document
from hydra2.search.joint_types import time as time
from hydra2.search.joint_types import world_actor_observation as world_actor_observation
