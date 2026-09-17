"""Supervised-loop validated-loss step: autocast, oracle join, train step.

Owns the per-microbatch forward/loss half of :class:`SupervisedLoop`:
the loop-owned bf16 autocast scope, the Fabric-delegated backward, the
opaque oracle-target join, the eager pre-validation plus compiled-kernel
loss gate, and the single-microbatch train step the accumulation loop
calls. Optimizer stepping and checkpointing live in the train and
checkpoint modules so each file stays inside the review-size ceiling.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch

from hydra2.training.loop_batch import (
    _model_forward as _model_forward,
)
from hydra2.training.loop_batch import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.loop_batch import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)
from hydra2.training.objectives_loss import (
    _check_total_finite as _check_total_finite,
)
from hydra2.training.objectives_loss import (
    validate_supervised_inputs as validate_supervised_inputs,
)

if TYPE_CHECKING:
    from hydra2.training.loop_state import TrainingLoopConfig as TrainingLoopConfig
    from hydra2.training.loop_state import TrainingState as TrainingState


class SupervisedLoopLossMixin:
    """Validated-loss step for :class:`SupervisedLoop`.

    Split host for the forward/loss half of :class:`SupervisedLoop`;
    the train subclass adds the accumulation loop and the frozen-model
    report and no overrides. Attribute access is duck-typed through the
    subclass.
    """

    model: Any
    config: TrainingLoopConfig
    state: TrainingState
    device: Any
    handle: Any | None
    privileged_source: Any | None
    evaluation_wall_ids: frozenset[str]
    _compiled_loss: Any

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _forward_autocast(self) -> Any:
        """Loop-owned bf16 autocast scope for forward+loss only.

        Returns ``torch.autocast(device_type="cuda", dtype=torch.bfloat16)``
        when ``precision == "bf16_mixed"`` on a CUDA device; otherwise a
        no-op ``nullcontext`` (fp32 default is byte-identical, CPU path
        never autocasts). Backward/clip/step always stay outside in fp32
        with fp32 master weights and NO GradScaler.
        """
        if self.config.precision == "bf16_mixed" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _backward(self, loss: torch.Tensor) -> None:
        if self.handle is not None and hasattr(self.handle, "backward"):
            self.handle.backward(loss)
        else:
            _ = loss.backward()

    def _maybe_join_oracle_targets(
        self, batch: dict[str, Any], evaluation_wall_ids: Any | None = None
    ) -> dict[str, Any]:
        # Join privileged placement/value targets by opaque decision_id only.
        # Called AFTER _validate_batch_no_privileged; merged batch is
        # re-validated so the actor batch stays privileged-free
        # (FORBIDDEN_BATCH_KEYS assert post-merge). Defaults unchanged:
        # absent/empty source or zero auxiliary weights is a no-op (the loss
        # then raises ContractError on missing targets, as before). Belief
        # 34-dist distribution support explicitly DEFERRED: belief heads keep
        # the index-CE contract and are never joined here.
        src: Any | None = self.privileged_source
        # Default to the ctor ledger so both join sites fail closed without
        # requiring an explicit arg; explicit arg still wins (tests + replay mirror).
        if evaluation_wall_ids is None:
            evaluation_wall_ids = getattr(self, "evaluation_wall_ids", None)
        if src is None:
            return batch
        if hasattr(src, "__len__"):
            try:
                if len(src) == 0:  # type: ignore[arg-type]
                    return batch
            except TypeError:
                pass
        if self.config.w_placement == 0.0 and self.config.w_value == 0.0:
            return batch
        decision_ids_any: Any = batch.get("_decision_ids", [])
        if not isinstance(decision_ids_any, (list, tuple)) or len(decision_ids_any) == 0:
            return batch
        decision_ids_raw: list[object] = list(decision_ids_any)
        decision_ids: list[str] = [str(x) for x in decision_ids_raw]
        from hydra2.belief.oracle_join import join_oracle_targets

        joined = join_oracle_targets(decision_ids, src, evaluation_wall_ids=evaluation_wall_ids)
        merged: dict[str, Any] = dict(batch)
        if self.config.w_placement != 0.0 and "placement_target" in joined:
            merged["placement_target"] = joined["placement_target"]
        if self.config.w_value != 0.0 and "value_target" in joined:
            merged["value_target"] = joined["value_target"]
        _validate_batch_no_privileged(merged)
        return merged

    def _validated_loss(self, model_out: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Eager pre-validation + compiled kernel + total-finite gate.

        Runs :func:`validate_supervised_inputs` on the host (same errors the
        kernel skips under compile), invokes ``self._compiled_loss`` (kernel
        on CUDA, validating public fn on fallback paths), then gates the
        weighted total with the identical non-finite error.  Both call sites
        (``train_step`` and the accumulation loop) share this so the fail-
        closed contract cannot drift between them.
        """
        weights = self.config.objective_weights()
        validate_supervised_inputs(model_out, batch, weights)
        losses = self._compiled_loss(model_out, batch, weights)
        _check_total_finite(losses["total"])
        return losses

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single microbatch forward/backward without optimizer stepping.

        Accumulation is managed by :meth:`train`; this is a low-level helper
        exposed for tests that want formula parity.
        """
        _validate_batch_no_privileged(batch)
        batch = self._maybe_join_oracle_targets(batch, evaluation_wall_ids=self.evaluation_wall_ids)
        batch = _move_batch_to_device(batch, self.device)
        # AMP: autocast covers forward+loss only; backward below stays fp32.
        with self._forward_autocast():
            model_out = _model_forward(self.model, batch)
            losses = self._validated_loss(model_out, batch)
        # Caller scales for accumulation; we return the unscaled total for logging
        total_tensor: torch.Tensor = losses["total"]
        total_unscaled: float = float(total_tensor.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative (keep on device) loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        # Backward is caller-owned when accumulating; this helper does backward with scale 1
        # For direct use, backward here; for accumulate loop the caller re-scales.
        # We expose both: this does immediate backward of total
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


__all__ = [
    "SupervisedLoopLossMixin",
]
