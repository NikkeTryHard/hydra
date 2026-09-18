"""Candidate 8 joint type/world model — observation-only opponent types, joint posterior, robust set.

Re-export facade over the split modules: :mod:`hydra2.search.joint_types`
(info keys, type policy, joint particles), :mod:`hydra2.search.joint_uncertainty`
(exact oracle, coherent trajectory, uncertainty set, spec factory), and
:mod:`hydra2.search.joint_planner` (:class:`JointTypeWorldPlanner`). Import from
this path; public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.joint_planner import JointTypeWorldPlanner as JointTypeWorldPlanner
from hydra2.search.joint_types import (
    FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY,
)
from hydra2.search.joint_types import (
    THETA_IDS as THETA_IDS,
)
from hydra2.search.joint_types import (
    JointParticle as JointParticle,
)
from hydra2.search.joint_types import (
    JointPosterior as JointPosterior,
)
from hydra2.search.joint_types import (
    OpponentTypePolicy as OpponentTypePolicy,
)
from hydra2.search.joint_types import (
    deterministic_joint_gumbel as deterministic_joint_gumbel,
)
from hydra2.search.joint_types import (
    info_key_for_observation as info_key_for_observation,
)
from hydra2.search.joint_types import (
    validate_hidden_permutation_invariance as validate_hidden_permutation_invariance,
)
from hydra2.search.joint_types import (
    validate_same_information_equality as validate_same_information_equality,
)
from hydra2.search.joint_uncertainty import (
    JointTypeWorldConfig as JointTypeWorldConfig,
)
from hydra2.search.joint_uncertainty import (
    UncertaintySet as UncertaintySet,
)
from hydra2.search.joint_uncertainty import (
    coherent_trajectory as coherent_trajectory,
)
from hydra2.search.joint_uncertainty import (
    exact_joint_posterior_oracle as exact_joint_posterior_oracle,
)
from hydra2.search.joint_uncertainty import (
    hidden_marginalization as hidden_marginalization,
)
from hydra2.search.joint_uncertainty import (
    make_joint_type_world_candidate_spec as make_joint_type_world_candidate_spec,
)
from hydra2.search.joint_uncertainty import (
    preserve_correlation_check as preserve_correlation_check,
)
from hydra2.search.joint_uncertainty import (
    sequential_joint_update as sequential_joint_update,
)

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
from hydra2.search.joint_types import (
    _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR,
)
from hydra2.search.joint_types import (
    _COMMON_AVAILABLE as _COMMON_AVAILABLE,
)
