"""WP-07B oracle privileged loader — separate namespace/process boundary.

Re-export facade over the split modules: :mod:`hydra2.belief.oracle_guard`
(privileged firewall, split/wall leakage gates, shard-path helpers),
:mod:`hydra2.belief.oracle_targets` (deterministic teacher targets from
privileged rows), :mod:`hydra2.belief.oracle_store` (train-split-only
privileged parquet loader), and :mod:`hydra2.belief.oracle_join`
(decision_id joins, ranks exporter, subprocess boundary, encoder isolation
proof). Import from this path; it preserves every public name and
``__all__``.
"""

from __future__ import annotations

from hydra2.belief.oracle_guard import (
    AUTHORIZED_SPLITS_FOR_INFERENCE as AUTHORIZED_SPLITS_FOR_INFERENCE,
)
from hydra2.belief.oracle_guard import (
    AUTHORIZED_TRAIN_SPLIT as AUTHORIZED_TRAIN_SPLIT,
)
from hydra2.belief.oracle_guard import (
    FORBIDDEN_IN_ACTOR_KEYS as FORBIDDEN_IN_ACTOR_KEYS,
)
from hydra2.belief.oracle_guard import (
    PRIVILEGED_KEYS as PRIVILEGED_KEYS,
)
from hydra2.belief.oracle_guard import (
    _actor_shard_paths as _actor_shard_paths,
)
from hydra2.belief.oracle_guard import (
    _privileged_shard_paths as _privileged_shard_paths,
)
from hydra2.belief.oracle_guard import (
    _require_train_split as _require_train_split,
)
from hydra2.belief.oracle_guard import (
    assert_no_privileged_leakage_in_actor_row as assert_no_privileged_leakage_in_actor_row,
)
from hydra2.belief.oracle_guard import (
    check_split_disjoint as check_split_disjoint,
)
from hydra2.belief.oracle_guard import (
    check_wall_leakage as check_wall_leakage,
)
from hydra2.belief.oracle_guard import (
    validate_actor_batch_no_privileged as validate_actor_batch_no_privileged,
)
from hydra2.belief.oracle_join import (
    assert_privileged_loader_isolated_from_encoder as assert_privileged_loader_isolated_from_encoder,  # noqa: E501  # reason: self-alias re-export single logical; splitting harms grep
)
from hydra2.belief.oracle_join import (
    join_oracle_targets as join_oracle_targets,
)
from hydra2.belief.oracle_join import (
    load_oracle_batch_in_subprocess as load_oracle_batch_in_subprocess,
)
from hydra2.belief.oracle_join import (
    ranks_from_final_scores as ranks_from_final_scores,
)
from hydra2.belief.oracle_store import (
    PrivilegedOracleLoader as PrivilegedOracleLoader,
)
from hydra2.belief.oracle_targets import (
    OracleTarget as OracleTarget,
)
from hydra2.belief.oracle_targets import (
    _belief_target_from_privileged as _belief_target_from_privileged,
)
from hydra2.belief.oracle_targets import (
    _oracle_utility_manifest as _oracle_utility_manifest,
)
from hydra2.belief.oracle_targets import (
    _teacher_logits_from_targets as _teacher_logits_from_targets,
)
from hydra2.belief.oracle_targets import (
    _value_from_ranks_via_utility as _value_from_ranks_via_utility,
)
from hydra2.belief.oracle_targets import (
    _value_target_from_privileged as _value_target_from_privileged,
)

__all__ = [
    "AUTHORIZED_TRAIN_SPLIT",
    "FORBIDDEN_IN_ACTOR_KEYS",
    "PRIVILEGED_KEYS",
    "OracleTarget",
    "PrivilegedOracleLoader",
    "assert_privileged_loader_isolated_from_encoder",
    "check_split_disjoint",
    "check_wall_leakage",
    "join_oracle_targets",
    "load_oracle_batch_in_subprocess",
    "ranks_from_final_scores",
    "validate_actor_batch_no_privileged",
]
