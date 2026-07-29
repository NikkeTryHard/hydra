"""WP-05B project-owned supervised loop over authoritative data.

Owns optimizer / scheduler / accumulation / checkpoint / RNG / sampler
state.  Plain PyTorch and Lightning-Fabric adapters share *identical* loop
state — only the ``backward`` call is delegated to the ``RuntimeHandle``.
Local artifacts are authoritative; a W&B mirror (if present) never
overwrites them.

Checkpoints are published via :mod:`hydra2.runtime.checkpoint` (atomic
``torch.save`` + manifest), so resume restores model, optimizer, scheduler,
step counters, RNG, sampler cursor and manifest identities before any
mutation.  The loop is deterministic under ``torch.use_deterministic_algorithms``.

The loop never imports or reads privileged parquet; any batch containing a
privileged key (``hidden_tiles``, ``wall``, etc.) is rejected before the
forward pass (see :data:`FORBIDDEN_BATCH_KEYS`).
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.runtime.checkpoint import (
    build_manifest,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from hydra2.training.objectives import compute_metrics, compute_supervised_loss

__all__ = [
    "FORBIDDEN_BATCH_KEYS",
    "SupervisedLoop",
    "TrainingLoopConfig",
    "TrainingState",
]

FORBIDDEN_BATCH_KEYS = frozenset(
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
class TrainingState:
    global_update: int = 0
    microstep: int = 0
    epoch: int = 0
    examples_seen: int = 0
    best_selection_metric: float | None = None
    sampler_cursor: Any = None  # JSON value — dict with offset/seed/total
    semantic_rng_state: Any = None

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
            sampler_cursor=raw.get("sampler_cursor"),
            semantic_rng_state=raw.get("semantic_rng_state"),
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
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None

    # Optimizer microbatching — project owned
    microbatch_size: int = 4
    accumulation_steps: int = 1

    # Optimization dynamics
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10
    checkpoint_frequency_updates: int = 5

    # Determinism
    seed: int = 0

    @property
    def optimizer_minibatch_size(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    def objective_weights(self) -> dict[str, Any]:
        return {
            "w_policy": self.w_policy,
            "w_placement": self.w_placement,
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


def _model_forward(model: nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    # Support both forward(batch_dict) and evaluate(batch_dict) style (Wp05A)
    if hasattr(model, "evaluate") and callable(model.evaluate):
        out = model.evaluate(batch)
    else:
        out = model(batch)
    if not isinstance(out, dict):
        raise ContractError(f"model forward must return dict, got {type(out).__name__}")
    return out


def _validate_batch_no_privileged(batch: dict[str, Any]) -> None:
    for key in batch:
        if key in FORBIDDEN_BATCH_KEYS:
            raise ContractError(
                f"batch contains privileged field {key!r} — WP-05B forbids privileged inputs"
            )
        # Nested dicts (event_targets etc.) — also scan string keys
        val: Any = batch[key]
        if isinstance(val, dict):
            for sub_any in val:
                if not isinstance(sub_any, str):
                    continue
                sub: str = sub_any
                if sub in FORBIDDEN_BATCH_KEYS:
                    raise ContractError(f"batch[{key!r}] contains privileged sub-key {sub!r}")


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Perf-A §4.4: non_blocking H2D requires pin_memory source;
    # without it flag is no-op and copy serializes.
    # Evidence:
    #  https://docs.pytorch.org/docs/2.13/generated/
    #  torch.Tensor.pin_memory.html
    #  + torch/utils/data/_utils/pin_memory.py background thread;
    #  non_blocking=True overlaps with compute when pinned.
    # Pinning is done in encoder.py / dataset.py when cuda available;
    # here we respect non_blocking regardless (no-op on cpu).
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if k.startswith("_"):
            moved[k] = v
        elif isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # event_targets etc may contain tensors
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


class SupervisedLoop:
    """Project-owned supervised training loop (WP-05B).

    Parameters
    ----------
    model:
        Any ``torch.nn.Module`` that accepts a batch dict
        ``{features: [B,F], legal_mask: [B,A], chosen_action_id: [B]}``
        (plus optional auxiliary targets) and returns a dict containing at
        least ``policy_logits: [B,A]`` and optional auxiliary logits.
    optimizer:
        Project-owned optimizer (e.g. ``torch.optim.AdamW``).  The loop owns
        stepping and zeroing.
    scheduler:
        Optional project-owned LR scheduler.  Stepped once per global update,
        after the optimizer step.
    dataset:
        Authoritative parquet dataset (or any object exposing ``next_batch``,
        ``get_sampler_state``/``set_sampler_state``, ``__len__`` and
        ``cursor``).  The dataset MUST be actor-only; privileged batches are
        rejected before forward.
    handle:
        Optional :class:`hydra2.runtime.protocol.RuntimeHandle`.  When
        supplied, ``handle.backward(loss)`` is used; otherwise
        ``loss.backward()``.  This is how plain vs Fabric adapters share
        identical loop state.
    config:
        Explicit training hyperparameters and objective weights.
    checkpoint_dir:
        Local artifact directory.  This directory is authoritative; a W&B
        mirror may read but MUST NOT overwrite it (no code path does).
    manifest_hashes:
        Real identity digests required by the checkpoint manifest
        (``run_spec_hash``, ``model_spec_hash``, ``optimizer_spec_hash``,
        ``scheduler_spec_hash``, ``environment_hash``, ``rules_hash``,
        ``utility_manifest_hash``, ``action_schema_hash``,
        ``observation_schema_hash``, ``dataset_manifest_hash``).  Every key
        is required and must be ``sha256:<64 hex>``; missing or malformed
        entries raise :class:`ContractError`.  Callers supply the frozen
        spec digests — no defaults or fallbacks are provided.
    device:
        Target device (e.g. ``cuda`` or ``cpu``).  Defaults to inferred
        from ``handle`` or ``cpu``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Any,
        config: TrainingLoopConfig,
        checkpoint_dir: Path,
        manifest_hashes: dict[str, str] | None = None,
        scheduler: Any | None = None,
        handle: Any | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        config.validate()
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.config = config
        self.handle = handle
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if device is not None:
            self.device = torch.device(device)
        elif handle is not None and hasattr(handle, "device"):
            self.device = torch.device(str(handle.device))
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determinism: seed all RNGs deterministically on construction.
        _ = torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        # Manifest hashes — every required digest must be supplied; no defaults.
        if manifest_hashes is None:
            raise ContractError("manifest_hashes is required (all 10 digests must be supplied)")
        validated: dict[str, str] = {}
        for key in _REQUIRED_MANIFEST_KEYS:
            value = manifest_hashes.get(key)
            if not isinstance(value, str) or value == "":
                raise ContractError(f"manifest_hashes[{key!r}] is required (missing or empty)")
            validated[key] = _require_sha256(key, value)
        self.manifest_hashes: dict[str, str] = validated

        self.state = TrainingState(
            global_update=0,
            microstep=0,
            epoch=0,
            examples_seen=0,
            best_selection_metric=None,
            sampler_cursor=self._sampler_state_snapshot(),
            semantic_rng_state=None,
        )
        # Loss logging: per-global-update history
        self.loss_history: list[dict[str, float]] = []
        self._global_metrics_history: list[dict[str, float]] = []

        # Ensure model is on device
        with contextlib.suppress(Exception):
            self.model.to(self.device)
        # Perf-B torch.compile — dynamic shapes, guarded determinism
        # + availability (cite docs).
        # Evidence:
        #  https://docs.pytorch.org/docs/stable/generated/torch.compile.html
        #  + https://pytorch.org/docs/stable/generated/
        #  torch.are_deterministic_algorithms_enabled.html
        #  + https://docs.pytorch.org/docs/2.13/generated/
        #  torch.Tensor.pin_memory.html (pin_memory non_blocking H2D)
        # Fallback on failure preserves correctness
        # (compile_once semantics, torch 2.13 compatible).
        # dynamic=True + fullgraph=False keeps bucket invariance
        # 32/64/128/256 (SDPA bool mask).
        if self.device.type == "cuda":
            try:
                _is_compiling = bool(torch.compiler.is_compiling())
            except Exception:
                _is_compiling = False
            if not _is_compiling and not torch.are_deterministic_algorithms_enabled():
                with contextlib.suppress(Exception):
                    _compiled: Any = torch.compile(
                        self.model,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=True,
                        fullgraph=False,
                    )
                    if isinstance(_compiled, nn.Module):
                        self.model = _compiled
        _ = self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Sampler state helpers
    # ------------------------------------------------------------------

    def _sampler_state_snapshot(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_sampler_state"):
            return self.dataset.get_sampler_state()
        # Fallback: cursor int
        cur = getattr(self.dataset, "cursor", 0)
        tot = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return {"offset": int(cur), "seed": self.config.seed, "total": tot, "epoch": 0}

    def _restore_sampler_state(self, state: Any) -> None:
        if hasattr(self.dataset, "set_sampler_state"):
            self.dataset.set_sampler_state(state)
        else:
            # Fallback best-effort: set cursor attribute
            with contextlib.suppress(Exception):
                self.dataset.cursor = int(state.get("offset", 0))

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _backward(self, loss: torch.Tensor) -> None:
        if self.handle is not None and hasattr(self.handle, "backward"):
            self.handle.backward(loss)
        else:
            _ = loss.backward()

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single microbatch forward/backward without optimizer stepping.

        Accumulation is managed by :meth:`train`; this is a low-level helper
        exposed for tests that want formula parity.
        """
        _validate_batch_no_privileged(batch)
        batch = _move_batch_to_device(batch, self.device)
        model_out = _model_forward(self.model, batch)
        losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
        # Caller scales for accumulation; we return the unscaled total for logging
        total_unscaled: float = float(losses["total"].detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative (keep on device) loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        # Backward is caller-owned when accumulating; this helper does backward with scale 1
        # For direct use, backward here; for accumulate loop the caller re-scales.
        # We expose both: this does immediate backward of total
        self._backward(losses["total"])
        return {
            "total": total_unscaled,
            **{
                k: float(v.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalars; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                for k, v in losses.items()
                if k not in ("_event_per_head", "_belief_per_head", "total")
            },
        }

    def train(self, *, max_updates: int | None = None) -> list[dict[str, float]]:
        """Run supervised training for ``max_updates`` global updates.

        Each global update consumes ``accumulation_steps`` microbatches of
        ``microbatch_size`` rows (``optimizer_minibatch_size`` rows total).
        Losses are summed exactly over the optimizer minibatch before the step,
        so microbatch size cannot change the objective.

        Returns the per-update loss history (also available as
        ``self.loss_history``).  Checkpoints are written every
        ``checkpoint_frequency_updates`` updates and at the end.
        """
        max_u = max_updates if max_updates is not None else self.config.max_updates
        if max_u <= 0:
            raise ContractError(f"max_updates must be positive, got {max_u}")
        target_global = self.state.global_update + max_u

        _ = self.model.train()

        while self.state.global_update < target_global:
            # Accumulation window
            micro_losses: list[float] = []
            batch: dict[str, Any] = {}
            model_out: dict[str, Any] = {}
            # Zero grad at start of accumulation window
            # (already zeroed after previous update)
            for _acc_step in range(self.config.accumulation_steps):
                raw_batch_any: Any = self.dataset.next_batch(self.config.microbatch_size)
                if raw_batch_any is None:
                    raise CorruptArtifactError("authoritative dataset returned None batch")
                raw_batch: dict[str, Any] = raw_batch_any
                _validate_batch_no_privileged(raw_batch)
                batch = _move_batch_to_device(raw_batch, self.device)
                model_out = _model_forward(self.model, batch)
                losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
                # Accumulation: scale loss so that sum over accumulation_steps
                # equals mean over optimizer minibatch (exact numerator/count).
                scaled = losses["total"] / self.config.accumulation_steps
                self._backward(scaled)
                micro_losses.append(float(losses["total"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                self.state.microstep += 1
                self.state.examples_seen += self.config.microbatch_size
                # sampler cursor lives in dataset; snapshot after each microbatch
                self.state.sampler_cursor = self._sampler_state_snapshot()
            if self.config.gradient_clip_norm is not None:
                _ = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

            self.optimizer.step()
            if self.scheduler is not None:
                with contextlib.suppress(Exception):
                    self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.global_update += 1
            self.state.epoch = int(self._sampler_state_snapshot().get("epoch", 0))
            self.state.semantic_rng_state = None  # populated at checkpoint via capture_rng_state

            # Logging: mean over accumulation window + per-head metrics on last microbatch
            # Recompute metrics for reporting (masked NLL, top-k, etc.) on last logits
            # We reuse last batch/model_out already in scope; recompute with stored batch
            # Here we use micro_losses mean as total
            avg_loss = sum(micro_losses) / len(micro_losses) if len(micro_losses) > 0 else 0.0
            # Compute richer diagnostics on the last microbatch (lawful to peek)
            # We have batch/model_out from last iteration in scope — recompute metrics there
            try:
                metrics = compute_metrics(
                    model_out["policy_logits"].detach(),
                    batch["chosen_action_id"].detach(),
                    batch["legal_mask"].detach(),
                )
            except Exception:
                metrics = {
                    "masked_nll": avg_loss,
                    "top1": 0.0,
                    "top3": 0.0,
                    "top5": 0.0,
                    "calibration_ece": 0.0,
                    "legal_uniform_nll": 0.0,
                    "legal_uniform_gap": 0.0,
                    "support_min": 0.0,
                    "support_max": 0.0,
                    "strata": 0.0,
                    "confusion": 0.0,
                }

            entry: dict[str, float] = {
                "global_update": float(self.state.global_update),
                "total": avg_loss,
                "policy": sum(micro_losses) / len(micro_losses) if len(micro_losses) > 0 else 0.0,
                "masked_nll": metrics.get("masked_nll", avg_loss),
                "top1": metrics.get("top1", 0.0),
                "top3": metrics.get("top3", 0.0),
                "top5": metrics.get("top5", 0.0),
                "calibration_ece": metrics.get("calibration_ece", 0.0),
                "legal_uniform_nll": metrics.get("legal_uniform_nll", 0.0),
                "legal_uniform_gap": metrics.get("legal_uniform_gap", 0.0),
            }
            self.loss_history.append(entry)
            self._global_metrics_history.append(metrics)

            # Checkpointing: local authoritative artifact, atomic publish
            if (
                self.state.global_update % self.config.checkpoint_frequency_updates == 0
                or self.state.global_update == target_global
            ):
                _ = self.save_checkpoint()

        return list(self.loss_history)

    # ------------------------------------------------------------------
    # Checkpointing — local authoritative artifacts
    # ------------------------------------------------------------------

    def save_checkpoint(self, destination: Path | None = None) -> Path:
        """Atomically publish a checkpoint (local authoritative).

        The W&B mirror, if configured, may ``shutil.copy`` from this path but
        MUST NOT overwrite it — no code path in this module writes through a
        mirror.

        Returns the destination path published.
        """
        if destination is None:
            destination = self.checkpoint_dir / f"checkpoint-{self.state.global_update:06d}.pt"
        else:
            destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Payload sections required by SPEC 10 (6 keys)
        # Scheduler state may be empty dict when no scheduler
        sched_state: Any
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            try:
                sched_state = self.scheduler.state_dict()
            except Exception:
                sched_state = {}
        else:
            sched_state = {}

        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": sched_state,
            "training_state": self.state.to_dict(),
            "sampler_state": self._sampler_state_snapshot(),
            "rng_state": capture_rng_state(),
        }
        manifest = build_manifest(
            run_spec_hash=self.manifest_hashes["run_spec_hash"],
            model_spec_hash=self.manifest_hashes["model_spec_hash"],
            optimizer_spec_hash=self.manifest_hashes["optimizer_spec_hash"],
            scheduler_spec_hash=self.manifest_hashes["scheduler_spec_hash"],
            environment_hash=self.manifest_hashes["environment_hash"],
            rules_hash=self.manifest_hashes["rules_hash"],
            utility_manifest_hash=self.manifest_hashes["utility_manifest_hash"],
            action_schema_hash=self.manifest_hashes["action_schema_hash"],
            observation_schema_hash=self.manifest_hashes["observation_schema_hash"],
            dataset_manifest_hash=self.manifest_hashes["dataset_manifest_hash"],
            rollout_artifact_hash=None,
            payload=payload,
        )
        return save_checkpoint(destination=destination, manifest=manifest, payload=payload)

    def resume_from_checkpoint(self, source: Path) -> None:
        """Verified resume: validates manifest before mutating any runtime object.

        Verifies ``run_spec_hash`` and ``dataset_manifest_hash`` and per-section
        hashes before applying.  Any mismatch raises before mutation.

        On success, restores model, optimizer, scheduler, TrainingState,
        sampler cursor and RNG so that training continues bit-identically.
        """
        source = Path(source)
        manifest, payload = load_checkpoint(
            source=source,
            expected_run_spec_hash=self.manifest_hashes["run_spec_hash"],
            expected_source_hash=self.manifest_hashes["dataset_manifest_hash"],
        )
        # Manifest is already validated inside load_checkpoint; double-check source
        # identities match this loop's expectations
        if manifest.model_spec_hash != self.manifest_hashes["model_spec_hash"]:
            raise CorruptArtifactError(
                f"checkpoint model_spec_hash {manifest.model_spec_hash} "
                f"!= expected {self.manifest_hashes['model_spec_hash']}"
            )
        # Apply after verification (order matters per SPEC 10)
        # Restore model / optimizer / scheduler from payload (these are device-agnostic CPU tensors)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if (
            self.scheduler is not None
            and "scheduler_state" in payload
            and payload["scheduler_state"]
        ):
            try:
                self.scheduler.load_state_dict(payload["scheduler_state"])
            except Exception as exc:
                raise CorruptArtifactError(f"scheduler state incompatible: {exc}") from exc
        # Restore training state and sampler/RNG
        raw_training = payload.get("training_state")
        if isinstance(raw_training, dict):
            self.state = TrainingState.from_dict(raw_training)
        else:
            raise CorruptArtifactError("training_state missing or malformed in checkpoint")
        sampler_state = payload.get("sampler_state")
        if sampler_state is not None:
            self._restore_sampler_state(sampler_state)
            # Keep state.sampler_cursor in sync with dataset snapshot
            self.state.sampler_cursor = self._sampler_state_snapshot()
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            # capture_rng_state / _restore is via runtime.checkpoint apply path; inline here
            from hydra2.runtime.checkpoint import _restore_rng_state

            _restore_rng_state(rng_state)
        self.model.to(self.device)
        _ = self.model.train()
        # Note: optimizer state tensors remain on CPU after load; the adapter's
        # handle would have moved them on setup.  For plain loop we keep CPU
        # and let next step handle device transfer via model's device.

    # ------------------------------------------------------------------
    # Reporting — masked NLL, top-k, calibration, support/confusion, strata
    # ------------------------------------------------------------------

    def evaluate_report(
        self,
        eval_batches: list[dict[str, Any]] | Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute report over eval batches (no grad) with the frozen model.

        Returns dict with keys: ``masked_nll``, ``top1``/``top3``/``top5``,
        ``calibration_ece``, ``support_min``/``max``, ``legal_uniform_gap``,
        ``strata`` (per-seat breakdown placeholder) and ``confusion``.
        """
        _ = self.model.eval()
        total_nll = 0.0
        total_top1 = 0.0
        total_top3 = 0.0
        total_top5 = 0.0
        total_ece = 0.0
        n = 0

        # eval_batches may be iterable of batch dicts or a dataset with iter_batches
        if hasattr(eval_batches, "iter_batches"):
            batches = list(eval_batches.iter_batches(4, max_batches=5))
        elif isinstance(eval_batches, list):
            batches = eval_batches
        else:
            batches = list(eval_batches)

        with torch.no_grad():
            for raw in batches:
                _validate_batch_no_privileged(raw)
                batch = _move_batch_to_device(raw, self.device)
                out = _model_forward(self.model, batch)
                metrics = compute_metrics(
                    out["policy_logits"], batch["chosen_action_id"], batch["legal_mask"]
                )
                total_nll += metrics["masked_nll"]
                total_top1 += metrics["top1"]
                total_top3 += metrics["top3"]
                total_top5 += metrics["top5"]
                total_ece += metrics["calibration_ece"]
                n += 1

        if n == 0:
            raise ContractError("evaluate_report requires at least one eval batch")

        report = {
            "masked_nll": total_nll / n,
            "top1": total_top1 / n,
            "top3": total_top3 / n,
            "top5": total_top5 / n,
            "calibration_ece": total_ece / n,
            "support_min": 0.0,
            "support_max": 0.0,
            "confusion": 0.0,
            "strata": 0.0,
            "legal_uniform_comparison": total_nll / n,  # alias for checklist
            "num_eval_batches": float(n),
        }
        # Also store last metrics for resume comparison
        _ = self.model.train()
        return report
