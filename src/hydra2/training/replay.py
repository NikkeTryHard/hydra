"""WP-11 Actor-Learner Replay — project-owned deterministic replay over authorized data.

Re-export facade over the split modules: :mod:`hydra2.training.replay_state`
(firewall vocabulary, config, batch helpers, label store),
:mod:`hydra2.training.replay_engine` (construction, sampling, stepping,
training), and :mod:`hydra2.training.replay_checkpoint` (best-ckpt helpers,
persistence, gated selection). Import from this path; it preserves every
public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.training.replay_checkpoint import (
    ActorLearnerReplayCheckpointMixin as ActorLearnerReplayCheckpointMixin,
)
from hydra2.training.replay_checkpoint import (
    _atomic_publish_best as _atomic_publish_best,
)
from hydra2.training.replay_checkpoint import (
    _best_ckpt_digest_for as _best_ckpt_digest_for,
)
from hydra2.training.replay_checkpoint import (
    _verify_best_ckpt as _verify_best_ckpt,
)
from hydra2.training.replay_engine import (
    ActorLearnerReplay as ActorLearnerReplay,
)
from hydra2.training.replay_state import (
    _REQUIRED_MANIFEST_KEYS as _REQUIRED_MANIFEST_KEYS,
)
from hydra2.training.replay_state import (
    FORBIDDEN_REPLAY_KEYS as FORBIDDEN_REPLAY_KEYS,
)
from hydra2.training.replay_state import (
    PrivilegedLabelStore as PrivilegedLabelStore,
)
from hydra2.training.replay_state import (
    ReplayConfig as ReplayConfig,
)
from hydra2.training.replay_state import (
    ReplayState as ReplayState,
)
from hydra2.training.replay_state import (
    _batch_action_kinds as _batch_action_kinds,
)
from hydra2.training.replay_state import (
    _model_forward as _model_forward,
)
from hydra2.training.replay_state import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.replay_state import (
    _require_sha256 as _require_sha256,
)
from hydra2.training.replay_state import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)

__all__ = [
    "FORBIDDEN_REPLAY_KEYS",
    "ActorLearnerReplay",
    "PrivilegedLabelStore",
    "ReplayConfig",
    "ReplayState",
]
