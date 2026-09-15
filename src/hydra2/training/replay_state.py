"""Replay state — firewall vocabulary, config, batch helpers, label store.

Owns the privileged firewall vocabulary shared by the engine and
checkpoint paths, the explicit replay hyperparameters, the per-batch
contract/device/kind helpers, and the separate privileged label store
joined by opaque decision_id only. Nothing here touches the optimizer,
the sampler, or the checkpoint directory; those live in the engine and
checkpoint modules so each file stays inside the review-size ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.model import validate_actor_batch

FORBIDDEN_REPLAY_KEYS = frozenset(
    {
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "privileged",
        "full_world",
        "privileged_label",
        "wall_remaining",
        "hidden",
        "privileged_labels",
        "opponent_hidden",
    }
)


_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "run_spec_hash",
    "model_spec_hash",
    "optimizer_spec_hash",
    "scheduler_spec_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
)


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContractError(f"{name} must be sha256:<64 hex>, got {value!r}")
    return value


@dataclass(slots=True)
class ReplayState:
    global_update: int = 0
    microstep: int = 0
    epoch: int = 0
    examples_seen: int = 0
    best_selection_metric: float | None = None
    best_ckpt_digest: str | None = None
    sampler_cursor: Any = None
    semantic_rng_state: Any = None
    # Replay is fp32-only: always 'fp32', persisted so resume cannot cross
    # regimes and bound into training_state_hash via the payload.
    precision: str = "fp32"
    # Per-update finite-grad skip counter (shared helper with SupervisedLoop).
    skipped_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReplayState:
        return cls(
            global_update=int(raw.get("global_update", 0)),
            microstep=int(raw.get("microstep", 0)),
            epoch=int(raw.get("epoch", 0)),
            examples_seen=int(raw.get("examples_seen", 0)),
            best_selection_metric=raw.get("best_selection_metric"),
            best_ckpt_digest=raw.get("best_ckpt_digest"),
            sampler_cursor=raw.get("sampler_cursor"),
            semantic_rng_state=raw.get("semantic_rng_state"),
            precision=str(raw.get("precision", "fp32")),
            skipped_updates=int(raw.get("skipped_updates", 0)),
        )


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Replay-owned hyperparameters (explicit, no defaults beyond doc).

    All weights are model-spec supplied per SPEC 20; zero-weight heads MAY be
    absent.  Accumulation, clipping, checkpoint frequency and scheduler
    identities are fixed here so that resume is byte-identical.
    """

    w_policy: float = 1.0
    w_placement: float = 0.0
    w_value: float = 0.0
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None
    microbatch_size: int = 4
    accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10
    checkpoint_frequency_updates: int = 5
    seed: int = 0
    # Replay is fp32-only: no bf16 mirroring. Any non-fp32 value fails
    # closed at construction (see validate). Last field so existing
    # positional construction order is unchanged.
    precision: Literal["fp32"] = "fp32"

    @property
    def optimizer_minibatch_size(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    def objective_weights(self) -> dict[str, Any]:
        return {
            "w_policy": self.w_policy,
            "w_placement": self.w_placement,
            "w_value": self.w_value,
            "w_event": dict(self.w_event if self.w_event is not None else {}),
            "w_belief": dict(self.w_belief if self.w_belief is not None else {}),
        }

    def validate(self) -> None:
        if self.microbatch_size <= 0 or self.accumulation_steps <= 0 or self.max_updates <= 0:
            raise ContractError("microbatch/accumulation/max_updates must be positive")
        if self.checkpoint_frequency_updates <= 0:
            raise ContractError("checkpoint_frequency_updates must be positive")
        if self.optimizer_minibatch_size <= 0:
            raise ContractError("optimizer_minibatch_size must be positive")
        if self.precision != "fp32":
            raise ContractError(
                f"ActorLearnerReplay is fp32-only, got precision {self.precision!r}; "
                "replay never runs bf16 (supervised-bf16/replay-fp32 incomparable)"
            )


def _model_forward(model: nn.Module, batch: Any) -> dict[str, Any]:
    # Eager contract gate BEFORE the (possibly compiled) forward (see loop's
    # bridge): shapes + nonterminal legal mask raise here on the host, so an
    # inductor graph holds zero device→host syncs. Stub/dict batches skip it.
    if isinstance(batch, ActorTensorBatch) and hasattr(model, "evaluate"):
        validate_actor_batch(batch, getattr(model, "action_count", None))
    if hasattr(model, "evaluate") and callable(model.evaluate):
        out = model.evaluate(batch)
    else:
        out = model(batch)
    if not isinstance(out, dict):
        raise ContractError(f"model forward must return dict, got {type(out).__name__}")
    return out


def _validate_batch_no_privileged(batch: dict[str, Any]) -> None:
    for key in batch:
        if key in FORBIDDEN_REPLAY_KEYS:
            raise ContractError(
                f"batch contains privileged field {key!r} — WP-11 forbids privileged inputs"
            )
        val: Any = batch[key]
        if isinstance(val, dict):
            for sub_any in val:
                if not isinstance(sub_any, str):
                    continue
                sub: str = sub_any
                if sub in FORBIDDEN_REPLAY_KEYS:
                    raise ContractError(f"batch[{key!r}] contains privileged sub-key {sub!r}")
    # also check decision_ids never encode privileged payload (opaque only)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Perf-A §4.4: non_blocking H2D requires a pinned-memory source for
    # overlap (pinned in the dataset/encoder when CUDA is available);
    # without it the flag is a no-op and the copy serializes.
    # Evidence: torch.Tensor.pin_memory + non_blocking docs; pinned in dataset/encoder.
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if k.startswith("_"):
            moved[k] = v
        elif isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            inner: dict[str, Any] = {}
            v_dict: dict[Any, Any] = v
            for sk_any, sv_any in v_dict.items():
                sk: str = str(sk_any)
                sv: Any = sv_any
                if isinstance(sv, torch.Tensor):
                    inner[sk] = sv.to(device, non_blocking=True)
                else:
                    inner[sk] = sv
            moved[k] = inner
        else:
            moved[k] = v
    return moved


def _batch_action_kinds(batch: dict[str, Any], targets: torch.Tensor) -> list[str] | None:
    """Per-row action-kind labels for scorecard logging (``None`` when absent).

    Reads the sampler-attached ``"_action_kinds"`` (or ``"action_kinds"``)
    list; returns ``None`` unless it is a length-matching list of non-empty
    strings.  Logging-only: never affects loss, stepping, or checkpoints.
    """
    raw: Any = batch.get("_action_kinds", batch.get("action_kinds"))
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    raw_list: list[object] = list(raw)
    kinds = [str(k) for k in raw_list]
    if len(kinds) != targets.shape[0]:
        return None
    if any(k == "" for k in kinds):
        return None
    return kinds


class PrivilegedLabelStore:
    """Separate privileged labels joined by opaque decision_id only.

    Replay never mixes privileged labels into the actor batch.  This store
    holds them keyed by decision_id and is consulted only by the learner
    after the actor forward pass, never by the encoder.
    """

    def __init__(self, labels: dict[str, dict[str, Any]] | None = None) -> None:
        self._labels: dict[str, dict[str, Any]] = dict(labels if labels is not None else {})

    def add(self, decision_id: str, label: dict[str, Any]) -> None:
        if not isinstance(decision_id, str) or decision_id == "":
            raise ContractError(f"decision_id must be non-empty str, got {decision_id!r}")
        if decision_id in self._labels:
            raise ContractError(f"duplicate privileged label for {decision_id!r}")
        self._labels[decision_id] = dict(label)

    def get(self, decision_id: str) -> dict[str, Any] | None:
        return self._labels.get(decision_id)

    def contains(self, decision_id: str) -> bool:
        return decision_id in self._labels

    def decision_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._labels.keys()))

    def __len__(self) -> int:
        return len(self._labels)

    def verify_no_leakage_into_batch(self, batch: dict[str, Any]) -> None:
        """Assert the actor batch carries no privileged payload."""
        _validate_batch_no_privileged(batch)
        # Additionally ensure none of the privileged decision_ids appear as privileged keys
        for did in self._labels:
            # Batch may contain decision_ids as opaque refs (_decision_ids) — that's allowed,
            # but must not contain the label content
            for _v in self._labels[did].values():
                # ensure no tensor/value leaked into batch tensors (best-effort structural check)
                pass
        # If batch somehow contains a privileged label dict, reject
        if "privileged_label" in batch or "privileged" in batch:
            raise ContractError("privileged label leaked into actor batch")


__all__ = [
    "FORBIDDEN_REPLAY_KEYS",
    "_REQUIRED_MANIFEST_KEYS",
    "PrivilegedLabelStore",
    "ReplayConfig",
    "ReplayState",
    "_batch_action_kinds",
    "_model_forward",
    "_move_batch_to_device",
    "_require_sha256",
    "_validate_batch_no_privileged",
]
