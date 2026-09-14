"""Replay engine — construction, stepping, and training.

Owns :class:`ActorLearnerReplay` construction (fp32-only gates, wall
ledger, manifest-hash validation), the Fabric-delegated backward, the
opaque oracle join, and the accumulation / finite-grad-skip /
mirror-logging train loop. Sampler snapshots, checkpoint persistence, and
held-out selection arrive via the checkpoint mixin base so the engine file
stays inside the review-size ceiling.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.training.dataset import AuthoritativeParquetDataset
from hydra2.training.objectives import (
    compute_metrics,
    compute_supervised_loss,
    global_grad_norm_is_finite,
)
from hydra2.training.replay_checkpoint import ActorLearnerReplayCheckpointMixin
from hydra2.training.replay_state import (
    _REQUIRED_MANIFEST_KEYS as _REQUIRED_MANIFEST_KEYS,
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

if TYPE_CHECKING:
    from hydra2.tracking.clearml_mirror import ClearmlMirror


class ActorLearnerReplay(ActorLearnerReplayCheckpointMixin):
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
        mirror: ClearmlMirror | None = None,
        runtime_spec: Any | None = None,
    ) -> None:
        config.validate()
        # Replay is fp32-only: any bf16 request fails closed here (config
        # already validated above; runtime/handle checked below).
        if runtime_spec is not None:
            rt_precision = getattr(runtime_spec, "precision", None)
            if rt_precision != "fp32":
                raise ContractError(
                    f"ActorLearnerReplay is fp32-only, got RuntimeSpec precision {rt_precision!r}; "
                    "replay never runs bf16 (supervised-bf16/replay-fp32 incomparable)"
                )
        if handle is not None:
            h_precision = getattr(handle, "precision", None)
            if h_precision is not None and h_precision != "fp32":
                raise ContractError(
                    f"ActorLearnerReplay is fp32-only, got RuntimeHandle precision {h_precision!r}"
                )
        if not isinstance(dataset, AuthoritativeParquetDataset):
            raise ContractError("dataset must be AuthoritativeParquetDataset (authorized data)")
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.config = config
        self.handle = handle
        self.runtime_spec = runtime_spec
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if device is not None:
            self.device = torch.device(device)
        elif handle is not None and hasattr(handle, "device"):
            _device_attr: object = handle.device
            self.device = torch.device(str(_device_attr))
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
            precision="fp32",
            skipped_updates=0,
        )
        self.loss_history: list[dict[str, float]] = []
        self._global_metrics_history: list[dict[str, float]] = []

        with contextlib.suppress(Exception):
            _ = self.model.to(self.device)
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
                _is_compiling = torch.compiler.is_compiling()
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
        # Observer mirror (see hydra2.tracking): disabled by default; when
        # enabled it copies allowlisted scalars + digests, never feeds back.
        if mirror is None:
            from hydra2.tracking.clearml_mirror import make_mirror

            mirror = make_mirror(
                manifest_hashes=dict(validated),
                loop_config={
                    "microbatch_size": config.microbatch_size,
                    "accumulation_steps": config.accumulation_steps,
                    "max_updates": config.max_updates,
                    "checkpoint_frequency_updates": config.checkpoint_frequency_updates,
                    "seed": config.seed,
                    "w_policy": config.w_policy,
                    "w_placement": config.w_placement,
                    "w_value": config.w_value,
                    "precision": "fp32",
                },
            )
            _ = mirror.start_run()
        self._mirror: ClearmlMirror = mirror

    @property
    def historical_opponents(self) -> tuple[str, ...]:
        return self._historical_opponents

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _backward(self, loss: torch.Tensor) -> None:
        if self.handle is not None and hasattr(self.handle, "backward"):
            self.handle.backward(loss)
        else:
            _ = loss.backward()

    def _maybe_join_oracle_targets(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Join privileged placement/value targets by opaque decision_id only.
        # Called AFTER verify_no_leakage_into_batch; merged batch is re-validated
        # so the actor batch stays privileged-free (FORBIDDEN_REPLAY_KEYS assert
        # post-merge). Defaults unchanged: empty store or zero auxiliary weights
        # is a no-op. Belief 34-dist distribution support explicitly DEFERRED:
        # belief heads keep the index-CE contract and are never joined here.
        if len(self.privileged_store) == 0:
            return batch
        if self.config.w_placement == 0.0 and self.config.w_value == 0.0:
            return batch
        decision_ids_any: Any = batch.get("_decision_ids", [])
        if not isinstance(decision_ids_any, (list, tuple)) or len(decision_ids_any) == 0:
            return batch
        decision_ids: list[str] = [str(x) for x in list(decision_ids_any)]
        from hydra2.belief.oracle_loader import join_oracle_targets

        joined = join_oracle_targets(
            decision_ids, self.privileged_store, evaluation_wall_ids=self.evaluation_wall_ids
        )
        merged: dict[str, Any] = dict(batch)
        if self.config.w_placement != 0.0 and "placement_target" in joined:
            merged["placement_target"] = joined["placement_target"]
        if self.config.w_value != 0.0 and "value_target" in joined:
            merged["value_target"] = joined["value_target"]
        _validate_batch_no_privileged(merged)
        self.privileged_store.verify_no_leakage_into_batch(merged)
        return merged

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        _validate_batch_no_privileged(batch)
        self.privileged_store.verify_no_leakage_into_batch(batch)
        batch = self._maybe_join_oracle_targets(batch)
        batch = _move_batch_to_device(batch, self.device)
        model_out = _model_forward(self.model, batch)
        losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
        total_tensor: torch.Tensor = losses["total"]
        total_unscaled: float = float(total_tensor.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        self._backward(total_tensor)
        aux_scalars: dict[str, float] = {}
        for _aux_key, _aux_value in losses.items():
            if _aux_key in ("_event_per_head", "_belief_per_head", "total"):
                continue
            _aux_tensor: torch.Tensor = _aux_value
            aux_scalars[_aux_key] = float(_aux_tensor.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalars; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return {
            "total": total_unscaled,
            **aux_scalars,
        }

    def train(self, *, max_updates: int | None = None) -> list[dict[str, float]]:
        max_u = max_updates if max_updates is not None else self.config.max_updates
        if max_u <= 0:
            raise ContractError(f"max_updates must be positive, got {max_u}")
        target_global = self.state.global_update + max_u
        _ = self.model.train()
        entry: dict[str, float] = {}
        while self.state.global_update < target_global:
            micro_losses: list[float] = []
            micro_policy: list[float] = []
            micro_placement: list[float] = []
            micro_value: list[float] = []
            micro_event: list[float] = []
            micro_belief: list[float] = []
            micro_event_heads: dict[str, list[float]] = {}
            micro_belief_heads: dict[str, list[float]] = {}
            batch: dict[str, Any] = {}
            model_out: dict[str, Any] = {}
            for _ in range(self.config.accumulation_steps):
                raw_batch_any: Any = self.dataset.next_batch(self.config.microbatch_size)
                if raw_batch_any is None:
                    raise CorruptArtifactError("authoritative dataset returned None batch")
                raw_batch: dict[str, Any] = raw_batch_any
                _validate_batch_no_privileged(raw_batch)
                self.privileged_store.verify_no_leakage_into_batch(raw_batch)
                raw_batch = self._maybe_join_oracle_targets(raw_batch)
                batch = _move_batch_to_device(raw_batch, self.device)
                model_out = _model_forward(self.model, batch)
                losses = compute_supervised_loss(model_out, batch, self.config.objective_weights())
                step_total: torch.Tensor = losses["total"]
                scaled: torch.Tensor = step_total / self.config.accumulation_steps
                self._backward(scaled)
                micro_losses.append(float(step_total.detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                micro_policy.append(float(losses["policy"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                micro_placement.append(float(losses["placement"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                micro_value.append(float(losses["value"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                micro_event.append(float(losses["event"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                micro_belief.append(float(losses["belief"].detach().cpu().item()))  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                _event_heads_any: Any = losses.get("_event_per_head", {})
                if isinstance(_event_heads_any, dict):
                    for _head_id, _head_loss in _event_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _head_scalar = float(_head_loss.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalars; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                            _head_id_obj: object = _head_id
                            micro_event_heads.setdefault(str(_head_id_obj), []).append(_head_scalar)
                _belief_heads_any: Any = losses.get("_belief_per_head", {})
                if isinstance(_belief_heads_any, dict):
                    for _head_id, _head_loss in _belief_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _head_scalar = float(_head_loss.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalars; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
                            _head_id_obj2: object = _head_id
                            micro_belief_heads.setdefault(str(_head_id_obj2), []).append(
                                _head_scalar
                            )
                self.state.microstep += 1
                self.state.examples_seen += self.config.microbatch_size
                self.state.sampler_cursor = self._sampler_state_snapshot()
            # Fail-closed finite-grad skip (shared helper with SupervisedLoop):
            # per-update global grad-norm finite check BEFORE clip/step.
            # Non-finite grads skip the step, zero grads, count the skip.
            _grads_finite, _grad_norm = global_grad_norm_is_finite(self.model)
            if not _grads_finite:
                self.optimizer.zero_grad(set_to_none=True)
                self.state.skipped_updates += 1
                self.state.global_update += 1
                self.state.epoch = int(self._sampler_state_snapshot().get("epoch", 0))
                self.state.semantic_rng_state = None
                _skip_avg = sum(micro_losses) / len(micro_losses) if len(micro_losses) > 0 else 0.0
                _skip_entry: dict[str, float] = {
                    "global_update": float(self.state.global_update),
                    "total": _skip_avg,
                    "policy": sum(micro_policy) / len(micro_policy)
                    if len(micro_policy) > 0
                    else 0.0,
                    "placement": sum(micro_placement) / len(micro_placement)
                    if len(micro_placement) > 0
                    else 0.0,
                    "value": sum(micro_value) / len(micro_value) if len(micro_value) > 0 else 0.0,
                    "event": sum(micro_event) / len(micro_event) if len(micro_event) > 0 else 0.0,
                    "belief": sum(micro_belief) / len(micro_belief)
                    if len(micro_belief) > 0
                    else 0.0,
                    "masked_nll": _skip_avg,
                    "top1": 0.0,
                    "top3": 0.0,
                    "top5": 0.0,
                    "skipped_updates": float(self.state.skipped_updates),
                    "skipped_this_update": 1.0,
                }
                self.loss_history.append(_skip_entry)
                self._global_metrics_history.append(dict(_skip_entry))
                entry = _skip_entry
                if self.state.global_update % self.config.checkpoint_frequency_updates == 0:
                    dest = self.save_checkpoint()
                    self._mirror.log_update(_skip_entry, step=self.state.global_update)
                    self._mirror.log_checkpoint(
                        checkpoint_path=dest,
                        manifest_json={
                            "checkpoint_file": dest.name,
                            "global_update": self.state.global_update,
                            "manifest_hashes": dict(self.manifest_hashes),
                        },
                    )
                continue
            if self.config.gradient_clip_norm is not None:
                _ = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.global_update += 1
            self.state.epoch = int(self._sampler_state_snapshot().get("epoch", 0))
            self.state.semantic_rng_state = None
            avg_loss = sum(micro_losses) / len(micro_losses) if len(micro_losses) > 0 else 0.0
            try:
                train_logits: torch.Tensor = model_out["policy_logits"]
                train_targets: torch.Tensor = batch["chosen_action_id"]
                train_mask: torch.Tensor = batch["legal_mask"]
                metrics = compute_metrics(
                    train_logits.detach(),
                    train_targets.detach(),
                    train_mask.detach(),
                    action_kinds=_batch_action_kinds(batch, train_targets),
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
                "policy": sum(micro_policy) / len(micro_policy) if len(micro_policy) > 0 else 0.0,
                "placement": sum(micro_placement) / len(micro_placement)
                if len(micro_placement) > 0
                else 0.0,
                "value": sum(micro_value) / len(micro_value) if len(micro_value) > 0 else 0.0,
                "event": sum(micro_event) / len(micro_event) if len(micro_event) > 0 else 0.0,
                "belief": sum(micro_belief) / len(micro_belief) if len(micro_belief) > 0 else 0.0,
                "masked_nll": metrics.get("masked_nll", avg_loss),
                "top1": metrics.get("top1", 0.0),
                "top3": metrics.get("top3", 0.0),
                "top5": metrics.get("top5", 0.0),
                "calibration_ece": metrics.get("calibration_ece", 0.0),
                "legal_uniform_nll": metrics.get("legal_uniform_nll", 0.0),
                "legal_uniform_gap": metrics.get("legal_uniform_gap", 0.0),
                "strata": metrics.get("strata", 0.0),
                "confusion": metrics.get("confusion", 0.0),
                "skipped_updates": float(self.state.skipped_updates),
                "skipped_this_update": 0.0,
            }
            # Per-type scorecards: flattened ``per_type/<kind>/*`` floats ride
            # the entry when the batch carries kind labels (mirrors
            # ``SupervisedLoop``; report-only, never loss).
            for _mkey, _mval in metrics.items():
                if _mkey.startswith("per_type/") and isinstance(_mval, float):
                    entry[_mkey] = _mval
            for _event_head in sorted(micro_event_heads):
                _event_vals = micro_event_heads[_event_head]
                entry[f"event_{_event_head}"] = (
                    sum(_event_vals) / len(_event_vals) if len(_event_vals) > 0 else 0.0
                )
            for _belief_head in sorted(micro_belief_heads):
                _belief_vals = micro_belief_heads[_belief_head]
                entry[f"belief_{_belief_head}"] = (
                    sum(_belief_vals) / len(_belief_vals) if len(_belief_vals) > 0 else 0.0
                )
            self.loss_history.append(entry)
            self._global_metrics_history.append(dict(entry))
            if self.state.global_update % self.config.checkpoint_frequency_updates == 0:
                dest = self.save_checkpoint()
                self._mirror.log_update(entry, step=self.state.global_update)
                self._mirror.log_checkpoint(
                    checkpoint_path=dest,
                    manifest_json={
                        "checkpoint_file": dest.name,
                        "global_update": self.state.global_update,
                        "manifest_hashes": dict(self.manifest_hashes),
                    },
                )
        dest = self.save_checkpoint()
        self._mirror.log_update(entry, step=self.state.global_update)
        self._mirror.log_checkpoint(
            checkpoint_path=dest,
            manifest_json={
                "checkpoint_file": dest.name,
                "global_update": self.state.global_update,
                "manifest_hashes": dict(self.manifest_hashes),
            },
        )
        return list(self.loss_history)
