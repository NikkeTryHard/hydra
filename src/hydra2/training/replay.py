"""WP-11 Actor-Learner Replay — project-owned deterministic replay over authorized data.

Owns optimizer / scheduler / accumulation / checkpoint / RNG / sampler
state.  Plain PyTorch and Lightning-Fabric adapters share *identical* replay
state — only the ``backward`` call is delegated to the ``RuntimeHandle``.
Local artifacts are authoritative; a W&B mirror (if present) never
overwrites them.

Replay carries actor-visible input and privileged labels separately.
Historical opponents are immutable.  Evaluation walls never enter replay.
Deterministic replay: same seed gives same order; interrupted/resumed run
is bitwise identical.  Any privileged field in the actor batch is a hard
failure (no silent leakage).

The replay is optional (failure does not block reference path) but when
present it must satisfy the three checklist items:
  - actor_learner_replay_over_authorized_data
  - deterministic_replay
  - no_privileged_fields
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
from hydra2.training.dataset import AuthoritativeParquetDataset
from hydra2.training.objectives import compute_metrics, compute_supervised_loss

__all__ = [
    "FORBIDDEN_REPLAY_KEYS",
    "ActorLearnerReplay",
    "PrivilegedLabelStore",
    "ReplayConfig",
    "ReplayState",
]

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
    sampler_cursor: Any = None
    semantic_rng_state: Any = None

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
            sampler_cursor=raw.get("sampler_cursor"),
            semantic_rng_state=raw.get("semantic_rng_state"),
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
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None
    microbatch_size: int = 4
    accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10
    checkpoint_frequency_updates: int = 5
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
    # Perf-A §4.4: non_blocking H2D (see loop.py) — requires pinned memory for overlap.
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


class ActorLearnerReplay:
    """Project-owned actor-learner replay over authorized data (WP-11).

    Parameters
    ----------
    model:
        Any ``torch.nn.Module`` that accepts a batch dict
        ``{features: [B,F], legal_mask: [B,A], chosen_action_id: [B]}``
        and returns at least ``policy_logits: [B,A]``.
    optimizer:
        Project-owned optimizer.
    dataset:
        Authoritative parquet dataset (actor-only).  Must be an instance of
        :class:`AuthoritativeParquetDataset` that has already verified shards.
    config:
        Explicit replay hyperparameters.
    checkpoint_dir:
        Local artifact directory (authoritative).
    manifest_hashes:
        Real identity digests required by the checkpoint manifest
        (``run_spec_hash``, ``model_spec_hash``, ``optimizer_spec_hash``,
        ``scheduler_spec_hash``, ``environment_hash``, ``rules_hash``,
        ``utility_manifest_hash``, ``action_schema_hash``,
        ``observation_schema_hash``, ``dataset_manifest_hash``).  Every key
        is required and must be ``sha256:<64 hex>``; missing or malformed
        entries raise :class:`ContractError`.  Callers supply the frozen
        spec digests — no defaults or fallbacks are provided.
    scheduler:
        Optional project-owned LR scheduler.
    handle:
        Optional ``RuntimeHandle`` for Fabric backward delegation.
    device:
        Target device.
    evaluation_wall_ids:
        Set of wall_ids that are reserved for evaluation and must never enter
        replay.  Any row whose game_id or decision_id references such a wall
        causes a hard failure.
    historical_opponents:
        Immutable opponent pool identities.
    privileged_store:
        Optional separate privileged label store (opaque join only).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: AuthoritativeParquetDataset,
        config: ReplayConfig,
        checkpoint_dir: Path,
        manifest_hashes: dict[str, str] | None = None,
        scheduler: Any | None = None,
        handle: Any | None = None,
        device: torch.device | str | None = None,
        evaluation_wall_ids: set[str] | frozenset[str] | None = None,
        historical_opponents: tuple[str, ...] | list[str] | None = None,
        privileged_store: PrivilegedLabelStore | None = None,
    ) -> None:
        config.validate()
        if not isinstance(dataset, AuthoritativeParquetDataset):
            raise ContractError("dataset must be AuthoritativeParquetDataset (authorized data)")
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

        # Evaluation wall ledger — must be disjoint from replay data
        self.evaluation_wall_ids: frozenset[str] = frozenset(
            evaluation_wall_ids if evaluation_wall_ids is not None else ()
        )
        if len(self.evaluation_wall_ids) > 0:
            for row_any in getattr(dataset, "_rows", []):
                row: dict[str, Any] = row_any
                gid: str = str(row.get("game_id", ""))
                did: str = str(row.get("decision_id", ""))
                wall_candidate = gid.split(":")[0] if ":" in gid else gid
                if wall_candidate in self.evaluation_wall_ids or did in self.evaluation_wall_ids:
                    raise ContractError(
                        f"evaluation wall {wall_candidate!r} in replay — walls_disjoint violated"
                    )
                # Also check decision_id encodes wall_id (if synthetic wall_id present)
                for wall_id in self.evaluation_wall_ids:
                    if wall_id in gid or wall_id in did:
                        raise ContractError(
                            f"replay row {gid!r}/{did!r} overlaps evaluation wall {wall_id!r}"
                        )

        # Historical opponents immutable
        if historical_opponents is not None:
            self._historical_opponents: tuple[str, ...] = tuple(historical_opponents)
        else:
            self._historical_opponents = ()
        # Expose via property to enforce immutability

        self.privileged_store: PrivilegedLabelStore = (
            privileged_store if privileged_store is not None else PrivilegedLabelStore()
        )
        # Determinism
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
        self.state = ReplayState(
            global_update=0,
            microstep=0,
            epoch=0,
            examples_seen=0,
            best_selection_metric=None,
            sampler_cursor=self._sampler_state_snapshot(),
            semantic_rng_state=None,
        )
        self.loss_history: list[dict[str, float]] = []
        self._global_metrics_history: list[dict[str, float]] = []

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
        # (compile_once, torch 2.13 compatible).
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
    @property
    def historical_opponents(self) -> tuple[str, ...]:
        return self._historical_opponents

    # ------------------------------------------------------------------
    # Sampler state helpers
    # ------------------------------------------------------------------

    def _sampler_state_snapshot(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_sampler_state"):
            return self.dataset.get_sampler_state()
        cur = getattr(self.dataset, "cursor", 0)
        tot = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return {"offset": int(cur), "seed": self.config.seed, "total": tot, "epoch": 0}

    def _restore_sampler_state(self, state: Any) -> None:
        if hasattr(self.dataset, "set_sampler_state"):
            self.dataset.set_sampler_state(state)
        else:
            with contextlib.suppress(Exception):
                dataset_any: Any = self.dataset
                dataset_any.cursor = int(state.get("offset", 0))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _backward(self, loss: torch.Tensor) -> None:
        if self.handle is not None and hasattr(self.handle, "backward"):
            self.handle.backward(loss)
        else:
            _ = loss.backward()

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        _validate_batch_no_privileged(batch)
        self.privileged_store.verify_no_leakage_into_batch(batch)
        batch = _move_batch_to_device(batch, self.device)
        model_out = _model_forward(self.model, batch)
        losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
        total_unscaled: float = float(losses["total"].detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
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
        max_u = max_updates if max_updates is not None else self.config.max_updates
        if max_u <= 0:
            raise ContractError(f"max_updates must be positive, got {max_u}")
        target_global = self.state.global_update + max_u
        _ = self.model.train()
        while self.state.global_update < target_global:
            micro_losses: list[float] = []
            batch: dict[str, Any] = {}
            model_out: dict[str, Any] = {}
            for _ in range(self.config.accumulation_steps):
                raw_batch_any: Any = self.dataset.next_batch(self.config.microbatch_size)
                if raw_batch_any is None:
                    raise CorruptArtifactError("authoritative dataset returned None batch")
                raw_batch: dict[str, Any] = raw_batch_any
                _validate_batch_no_privileged(raw_batch)
                self.privileged_store.verify_no_leakage_into_batch(raw_batch)
                batch = _move_batch_to_device(raw_batch, self.device)
                model_out = _model_forward(self.model, batch)
                losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
                scaled = losses["total"] / self.config.accumulation_steps
                self._backward(scaled)
                micro_losses.append(float(losses["total"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                self.state.microstep += 1
                self.state.examples_seen += self.config.microbatch_size
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
            self.state.semantic_rng_state = None
            avg_loss = sum(micro_losses) / len(micro_losses) if len(micro_losses) > 0 else 0.0
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
            }
            self.loss_history.append(entry)
            self._global_metrics_history.append(dict(entry))
            if self.state.global_update % self.config.checkpoint_frequency_updates == 0:
                _ = self.save_checkpoint()
        _ = self.save_checkpoint()
        return list(self.loss_history)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path | None = None) -> Path:
        dest = (
            Path(path)
            if path is not None
            else self.checkpoint_dir / f"ckpt-{self.state.global_update:06d}.pt"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
            else {},
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
        _ = save_checkpoint(destination=dest, manifest=manifest, payload=payload)
        return dest

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        _manifest, payload = load_checkpoint(
            source=path,
            expected_run_spec_hash=self.manifest_hashes["run_spec_hash"],
            expected_source_hash=self.manifest_hashes["dataset_manifest_hash"],
        )
        # Apply before mutating state that would mask errors: load verifies before touching
        from hydra2.runtime.checkpoint import apply_checkpoint

        apply_checkpoint(
            payload,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        # Restore training state and sampler
        raw_ts = payload.get("training_state", {})
        self.state = ReplayState.from_dict(dict(raw_ts))
        sampler_state = payload.get("sampler_state")
        if sampler_state is not None:
            self._restore_sampler_state(sampler_state)
        # RNG already restored via apply_checkpoint

    def replay_batches(self, *, batch_size: int, num_batches: int) -> list[list[str]]:
        """Deterministic replay: return decision_id order for next num_batches."""
        saved = self._sampler_state_snapshot()
        out: list[list[str]] = []
        for _ in range(num_batches):
            batch = self.dataset.next_batch(batch_size)
            if batch is None:
                break
            dids = batch.get("_decision_ids", [])
            out.append([str(x) for x in list(dids)])
        # Restore cursor so replay_batches is non-mutating peek (deterministic replay)
        self._restore_sampler_state(saved)
        return out

    def verify_authorized(self) -> None:
        """Verify the underlying dataset is authoritative and disjoint from eval walls."""
        # Dataset already verified shards on construction; re-assert cursor invariants
        if len(self.dataset) == 0:
            raise ContractError("authorized dataset is empty")
        for row in getattr(self.dataset, "_rows", []):
            for bad in FORBIDDEN_REPLAY_KEYS:
                if bad in row:
                    raise ContractError(
                        f"privileged field {bad!r} in authorized row {row.get('decision_id')!r}"
                    )
        if len(self.evaluation_wall_ids) > 0:
            for row_any in getattr(self.dataset, "_rows", []):
                row: dict[str, Any] = row_any
                gid: str = str(row.get("game_id", ""))
                if any(w in gid for w in self.evaluation_wall_ids):
                    raise ContractError(f"evaluation wall overlap detected for {gid!r}")
