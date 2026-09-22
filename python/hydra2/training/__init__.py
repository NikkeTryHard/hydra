"""WP-05B training loop package entry.

Exports the project-owned supervised loop, authoritative parquet dataset,
masked objectives and reporting helpers.  Inference loaders cannot import
privileged row constructors (enforced by __all__ and separate modules).
WP-11 replay is optionally exported when available.
"""

from __future__ import annotations

from hydra2.training.dataset_encode import tensorize_actor_row
from hydra2.training.dataset_store import (
    AuthoritativeParquetDataset,
    SamplerState,
)
from hydra2.training.loop_state import TrainingLoopConfig, TrainingState
from hydra2.training.loop_train import SupervisedLoop
from hydra2.training.objectives_loss import (
    compute_supervised_loss,
    masked_cross_entropy,
)
from hydra2.training.objectives_metrics import (
    compute_metrics,
    masked_topk_accuracy,
)
from hydra2.training.replay_engine import (
    ActorLearnerReplay,
)
from hydra2.training.replay_state import (
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
