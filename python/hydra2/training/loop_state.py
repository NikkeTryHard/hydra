"""Supervised-loop state: firewall vocabulary, config, identity helpers.

Owns the privileged firewall vocabulary shared by the engine and
checkpoint paths, the explicit loop hyperparameters, the manifest-key
tuple every checkpoint manifest requires, and the best-checkpoint
digest / publish / verify-closed helpers. Nothing here touches the
optimizer, the sampler, or the checkpoint directory; those live in the
engine, train, and checkpoint modules so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from hydra2._native import contracts as _loop_state_bridge  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import sha256_digest as sha256_digest
from hydra2.contracts.common import ContractError, CorruptArtifactError

#: Privileged batch fields (single source: ``hydra2._native.contracts``
#: ``LOOP_FORBIDDEN_BATCH_KEYS``; the stale-.so fallback is the byte-identical
#: oracle literal, per the ``randomness.py`` discipline).
_BRIDGE_FORBIDDEN_KEYS: frozenset[str] = _loop_state_bridge.LOOP_FORBIDDEN_BATCH_KEYS
FORBIDDEN_BATCH_KEYS = (
    frozenset(_BRIDGE_FORBIDDEN_KEYS)
    if hasattr(_loop_state_bridge, "LOOP_FORBIDDEN_BATCH_KEYS")
    else frozenset(
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
        }
    )
)


#: Checkpoint manifest keys (single source: ``hydra2._native.contracts``
#: ``LOOP_REQUIRED_MANIFEST_KEYS``; the stale-.so fallback is byte-identical).
_BRIDGE_MANIFEST_KEYS: tuple[str, ...] = _loop_state_bridge.LOOP_REQUIRED_MANIFEST_KEYS
_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    tuple(_BRIDGE_MANIFEST_KEYS)
    if hasattr(_loop_state_bridge, "LOOP_REQUIRED_MANIFEST_KEYS")
    else (
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
)


#: Bridge leaf for the sha gate (single source: ``hydra2._native.contracts``
#: ``loop_require_sha256``; the stale-.so fallback is the byte-identical oracle).
_loop_require_sha256 = getattr(_loop_state_bridge, "loop_require_sha256", None)

#: Bridge leaves for the int/enum config gates (single source:
#: ``hydra2._native.contracts`` ``training_*`` / ``loop_precision_gate``;
#: the stale-.so fallback is the byte-identical oracle below).
_training_minibatch_size = getattr(_loop_state_bridge, "training_minibatch_size", None)
_training_sizes_gate = getattr(_loop_state_bridge, "training_sizes_gate", None)
_training_prefetch_gate = getattr(_loop_state_bridge, "training_prefetch_gate", None)
_loop_precision_gate = getattr(_loop_state_bridge, "loop_precision_gate", None)


def _require_sha256(name: str, value: str) -> str:
    if _loop_require_sha256 is not None:
        try:
            bound: str = _loop_require_sha256(name, value)
            return bound
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContractError(f"{name} must be sha256:<64 hex>, got {value!r}")
    return value


def _best_ckpt_digest_for(data: bytes) -> str:
    return str(sha256_digest(data))


def _atomic_publish_best(source: Path, dest: Path) -> str:
    """Atomically publish ``source`` bytes to ``dest``; return its digest."""
    src = Path(source)
    dst = Path(dest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise ContractError(f"selection ckpt missing: {src}")
    data = src.read_bytes()
    digest = _best_ckpt_digest_for(data)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        _ = fh.write(data)  # intentionally discarded: byte count unneeded after fsync
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    try:
        dir_fd = os.open(dst.parent, os.O_DIRECTORY)
    except Exception:
        return digest
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest


def _verify_best_ckpt(checkpoint_dir: Path, *, metric: float | None, digest: str | None) -> None:
    """Fail closed when a promoted best has no matching ``best-ckpt.pt``."""
    if metric is None and digest is None:
        return
    dest = Path(checkpoint_dir) / "best-ckpt.pt"
    if not dest.is_file():
        raise CorruptArtifactError(
            f"best-ckpt.pt missing for best_selection_metric={metric!r} in {dest.parent}"
        )
    if digest is not None:
        actual = _best_ckpt_digest_for(dest.read_bytes())
        if actual != digest:
            raise CorruptArtifactError(
                f"best-ckpt.pt digest mismatch: expected {digest}, got {actual}"
            )


@dataclass(slots=True)
class TrainingState:
    global_update: int = 0
    microstep: int = 0
    epoch: int = 0
    examples_seen: int = 0
    best_selection_metric: float | None = None
    best_ckpt_digest: str | None = None
    sampler_cursor: Any = None  # JSON value — dict with offset/seed/total
    semantic_rng_state: Any = None
    # Precision regime this state was produced under (persisted so resume
    # cannot silently cross fp32<->bf16). Bound into training_state_hash via
    # the checkpoint payload; run_spec_hash binds it at the manifest level.
    precision: str = "fp32"
    # Per-update finite-grad skip counter (non-finite grad norm skipped step).
    skipped_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingState:
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
class TrainingLoopConfig:
    """Loop-owned hyperparameters and objective weights (explicit, no defaults).

    All weights are model-spec supplied per SPEC 20; zero-weight heads MAY be
    absent.  Accumulation, clipping, checkpoint frequency and scheduler
    identities are fixed here so that resume is byte-identical.
    """

    # Objective weights (explicit)
    w_policy: float = 1.0
    w_placement: float = 0.0
    w_value: float = 0.0
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None

    # Optimizer microbatching — project owned
    microbatch_size: int = 4
    accumulation_steps: int = 8

    # Optimization dynamics
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10
    checkpoint_frequency_updates: int = 5

    # Determinism
    seed: int = 0

    # Mixed precision — loop-owned bf16 autocast around forward+loss only.
    # fp32 default preserves byte-identical behavior; bf16_mixed opts into
    # torch.autocast(cuda, bfloat16) with fp32 master weights, fp32
    # accumulate/optimizer/clip and NO GradScaler. fp16 excluded by design.
    precision: Literal["fp32", "bf16_mixed"] = "fp32"
    # Recipe threading (defaults carry the pinned constants below):
    # legal-only label-smoothing eps in [0, 1) (0.0 = disabled, plain CE;
    # default 0.03 spreads mass over the LEGAL set only, illegal mass stays
    # exactly zero); sampler/scorecard knobs carried for observability
    # (stream single-pass keeps stream order; parquet dataset honors
    # stratification).
    label_smoothing: float = 0.03
    stratified_sampling: bool = False
    sampling_ratios: dict[str, float] | None = None
    log_per_type_metrics: bool = True
    fit_temperature: bool = True
    fetch_prefetch_depth: int = 3

    @property
    def optimizer_minibatch_size(self) -> int:
        if _training_minibatch_size is not None:
            try:
                size: int = _training_minibatch_size(self.microbatch_size, self.accumulation_steps)
                return size
            except (ValueError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        return self.microbatch_size * self.accumulation_steps

    def objective_weights(self) -> dict[str, Any]:
        return {
            "w_policy": self.w_policy,
            "w_placement": self.w_placement,
            "w_value": self.w_value,
            "w_event": dict(self.w_event if self.w_event is not None else {}),
            "w_belief": dict(self.w_belief if self.w_belief is not None else {}),
            "label_smoothing": self.label_smoothing,
        }

    def validate(self) -> None:
        if _training_sizes_gate is not None:
            try:
                _training_sizes_gate(
                    self.microbatch_size,
                    self.accumulation_steps,
                    self.max_updates,
                    self.checkpoint_frequency_updates,
                )
            except (ValueError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        elif self.microbatch_size <= 0 or self.accumulation_steps <= 0 or self.max_updates <= 0:
            raise ContractError("microbatch/accumulation/max_updates must be positive")
        if _training_sizes_gate is None and self.checkpoint_frequency_updates <= 0:
            raise ContractError("checkpoint_frequency_updates must be positive")
        if _training_minibatch_size is not None:
            try:
                _training_minibatch_size(self.microbatch_size, self.accumulation_steps)
            except (ValueError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        elif self.optimizer_minibatch_size <= 0:
            raise ContractError("optimizer_minibatch_size must be positive")
        if self.w_policy < 0.0 or self.w_placement < 0.0 or self.w_value < 0.0:
            raise ContractError("w_policy/w_placement/w_value must be nonnegative")
        event_vals = (self.w_event if self.w_event is not None else {}).values()
        belief_vals = (self.w_belief if self.w_belief is not None else {}).values()
        for _w in list(event_vals) + list(belief_vals):
            if not isinstance(_w, (int, float)) or _w < 0.0:
                raise ContractError("w_event/w_belief values must be nonnegative")
        if _loop_precision_gate is not None:
            try:
                _loop_precision_gate(self.precision)
            except (ValueError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        elif self.precision not in ("fp32", "bf16_mixed"):
            raise ContractError(f"precision must be 'fp32' or 'bf16_mixed', got {self.precision!r}")
        _eps = self.label_smoothing
        if (
            isinstance(_eps, bool)
            or not isinstance(_eps, (int, float))
            or not (0.0 <= float(_eps) < 1.0)
            or float(_eps) != float(_eps)
        ):
            raise ContractError(f"label_smoothing must lie in [0, 1), got {self.label_smoothing!r}")
        if not isinstance(self.stratified_sampling, bool):
            raise ContractError(
                f"stratified_sampling must be a bool, got {self.stratified_sampling!r}"
            )
        if self.sampling_ratios is not None:
            if not isinstance(self.sampling_ratios, dict):
                raise ContractError("sampling_ratios must be a mapping or null")
            for _k, _v in self.sampling_ratios.items():
                if not isinstance(_k, str) or _k == "":
                    raise ContractError("sampling_ratios keys must be non-empty strings")
                if (
                    isinstance(_v, bool)
                    or not isinstance(_v, (int, float))
                    or not (float(_v) == float(_v))
                    or float(_v) in (float("inf"), float("-inf"))
                    or float(_v) <= 0.0
                ):
                    raise ContractError(f"sampling_ratios[{_k!r}] must be positive and finite")
        if not isinstance(self.log_per_type_metrics, bool):
            raise ContractError(
                f"log_per_type_metrics must be a bool, got {self.log_per_type_metrics!r}"
            )
        if not isinstance(self.fit_temperature, bool):
            raise ContractError(f"fit_temperature must be a bool, got {self.fit_temperature!r}")
        if _training_prefetch_gate is not None:
            try:
                _training_prefetch_gate(self.fetch_prefetch_depth)
            except (ValueError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        elif (
            isinstance(self.fetch_prefetch_depth, bool)
            or not isinstance(self.fetch_prefetch_depth, int)
            or not (1 <= self.fetch_prefetch_depth <= 16)
        ):
            raise ContractError(
                f"fetch_prefetch_depth must be an int in [1, 16], got {self.fetch_prefetch_depth!r}"
            )


__all__ = [
    "FORBIDDEN_BATCH_KEYS",
    "_REQUIRED_MANIFEST_KEYS",
    "TrainingLoopConfig",
    "TrainingState",
    "_atomic_publish_best",
    "_best_ckpt_digest_for",
    "_require_sha256",
    "_verify_best_ckpt",
]
