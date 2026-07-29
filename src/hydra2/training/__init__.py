"""WP-05B training loop package entry.

Exports the project-owned supervised loop, authoritative parquet dataset,
masked objectives and reporting helpers.  Inference loaders cannot import
privileged row constructors (enforced by __all__ and separate modules).
WP-11 replay is optionally exported when available.
"""

from __future__ import annotations

from hydra2.training.dataset import AuthoritativeParquetDataset, SamplerState, tensorize_actor_row
from hydra2.training.loop import SupervisedLoop, TrainingLoopConfig, TrainingState
from hydra2.training.objectives import (
    compute_metrics,
    compute_supervised_loss,
    masked_cross_entropy,
    masked_topk_accuracy,
)
from hydra2.training.replay import (
    ActorLearnerReplay,
    PrivilegedLabelStore,
    ReplayConfig,
    ReplayState,
)

__all__ = [
    "ActorLearnerReplay",
    "AuthoritativeParquetDataset",
    "PrivilegedLabelStore",
    "ReplayConfig",
    "ReplayState",
    "SamplerState",
    "SupervisedLoop",
    "TrainingLoopConfig",
    "TrainingState",
    "compute_metrics",
    "compute_supervised_loss",
    "masked_cross_entropy",
    "masked_topk_accuracy",
    "tensorize_actor_row",
]
