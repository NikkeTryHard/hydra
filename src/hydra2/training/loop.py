"""WP-05B project-owned supervised loop over authoritative data.

Re-export facade over the split modules: :mod:`hydra2.training.loop_state`
(firewall vocabulary, config, manifest keys, best-checkpoint helpers),
:mod:`hydra2.training.loop_batch` (forward bridge, device move, telemetry
rows), :mod:`hydra2.training.loop_engine` (construction, sampler, fetch),
:mod:`hydra2.training.loop_loss` (validated-loss step),
:mod:`hydra2.training.loop_train` (accumulation loop plus
:class:`SupervisedLoop`), and :mod:`hydra2.training.loop_checkpoint`
(persistence, report, selection). Import from this path; it preserves
every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.training.loop_batch import (
    MicrobatchTelemetry as MicrobatchTelemetry,
)
from hydra2.training.loop_batch import (
    UpdateTelemetry as UpdateTelemetry,
)
from hydra2.training.loop_batch import (
    _batch_action_kinds as _batch_action_kinds,
)
from hydra2.training.loop_batch import (
    _model_forward as _model_forward,
)
from hydra2.training.loop_batch import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.loop_batch import (
    _quantile_sorted as _quantile_sorted,
)
from hydra2.training.loop_batch import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)
from hydra2.training.loop_batch import (
    _window_means as _window_means,
)
from hydra2.training.loop_batch import (
    summarize_telemetry as summarize_telemetry,
)
from hydra2.training.loop_batch import (
    summarize_update_telemetry as summarize_update_telemetry,
)
from hydra2.training.loop_checkpoint import (
    SupervisedLoopCheckpointMixin as SupervisedLoopCheckpointMixin,
)
from hydra2.training.loop_engine import (
    SupervisedLoopEngineMixin as SupervisedLoopEngineMixin,
)
from hydra2.training.loop_loss import (
    SupervisedLoopLossMixin as SupervisedLoopLossMixin,
)
from hydra2.training.loop_state import (
    _REQUIRED_MANIFEST_KEYS as _REQUIRED_MANIFEST_KEYS,
)
from hydra2.training.loop_state import (
    FORBIDDEN_BATCH_KEYS as FORBIDDEN_BATCH_KEYS,
)
from hydra2.training.loop_state import (
    TrainingLoopConfig as TrainingLoopConfig,
)
from hydra2.training.loop_state import (
    TrainingState as TrainingState,
)
from hydra2.training.loop_state import (
    _atomic_publish_best as _atomic_publish_best,
)
from hydra2.training.loop_state import (
    _best_ckpt_digest_for as _best_ckpt_digest_for,
)
from hydra2.training.loop_state import (
    _require_sha256 as _require_sha256,
)
from hydra2.training.loop_state import (
    _verify_best_ckpt as _verify_best_ckpt,
)
from hydra2.training.loop_train import (
    SupervisedLoop as SupervisedLoop,
)
from hydra2.training.loop_train import (
    SupervisedLoopTrainMixin as SupervisedLoopTrainMixin,
)

__all__ = [
    "FORBIDDEN_BATCH_KEYS",
    "MicrobatchTelemetry",
    "SupervisedLoop",
    "TrainingLoopConfig",
    "TrainingState",
    "summarize_telemetry",
]
