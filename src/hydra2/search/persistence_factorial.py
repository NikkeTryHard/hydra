"""Persistence factorial — B/F/R/P/C state-machine, commitment, reporting.

Re-export facade over the split modules: :mod:`hydra2.search.persistence_kernel`
(arms, packet kernel, forest state), :mod:`hydra2.search.persistence_spec`
(CandidateSpec factory per arm), :mod:`hydra2.search.persistence_planner`
(per-arm state machine plus :class:`PersistencePlanner`), and
:mod:`hydra2.search.persistence_report` (frozen whole-block report).
Import from this path; public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.persistence_kernel import ARM_DEFS as ARM_DEFS
from hydra2.search.persistence_kernel import CandidateSpec as CandidateSpec
from hydra2.search.persistence_kernel import FinitePacket as FinitePacket
from hydra2.search.persistence_kernel import ForestState as ForestState
from hydra2.search.persistence_kernel import PersistenceArm as PersistenceArm
from hydra2.search.persistence_kernel import commit_equals_rebuild as commit_equals_rebuild
from hydra2.search.persistence_kernel import compute_packet_id as compute_packet_id
from hydra2.search.persistence_kernel import enumerate_packets_for as enumerate_packets_for
from hydra2.search.persistence_kernel import fresh_rebuild_epoch as fresh_rebuild_epoch
from hydra2.search.persistence_kernel import make_persistence_arm as make_persistence_arm
from hydra2.search.persistence_planner import PersistencePlanner as PersistencePlanner
from hydra2.search.persistence_report import FactorialContrasts as FactorialContrasts
from hydra2.search.persistence_report import FactorialReport as FactorialReport
from hydra2.search.persistence_report import factorial_contrasts as factorial_contrasts
from hydra2.search.persistence_report import generate_factorial_report as generate_factorial_report
from hydra2.search.persistence_report import (
    stratify_surprise_miss_recovery as stratify_surprise_miss_recovery,
)
from hydra2.search.persistence_spec import (
    deterministic_gumbel_for_arm as deterministic_gumbel_for_arm,
)
from hydra2.search.persistence_spec import (
    make_persistence_candidate_spec as make_persistence_candidate_spec,
)
from hydra2.search.persistence_spec import (
    validate_deadline_and_fallback as validate_deadline_and_fallback,
)

__all__ = [
    "ARM_DEFS",
    "CandidateSpec",
    "FactorialContrasts",
    "FactorialReport",
    "FinitePacket",
    "ForestState",
    "PersistenceArm",
    "PersistencePlanner",
    "commit_equals_rebuild",
    "compute_packet_id",
    "deterministic_gumbel_for_arm",
    "enumerate_packets_for",
    "factorial_contrasts",
    "fresh_rebuild_epoch",
    "generate_factorial_report",
    "make_persistence_arm",
    "make_persistence_candidate_spec",
    "stratify_surprise_miss_recovery",
    "validate_deadline_and_fallback",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.persistence_kernel import _COMMON_AVAILABLE as _COMMON_AVAILABLE
from hydra2.search.persistence_kernel import BeliefEpochLite as BeliefEpochLite
from hydra2.search.persistence_kernel import _action_key as _action_key
from hydra2.search.persistence_kernel import _distribute_quota as _distribute_quota
from hydra2.search.persistence_kernel import _obs_hash_from_epoch as _obs_hash_from_epoch
from hydra2.search.persistence_spec import _default_hashes as _default_hashes
