"""Supervised-loop train path: accumulation loop over authoritative data.

Owns the :class:`SupervisedLoop` accumulation loop (fetch, forward,
validated loss, finite-grad skip, optimizer step, per-update logging,
and cadence checkpointing). Construction and the fetch hooks arrive via
the engine mixin, the validated-loss step via the loss mixin, and
sampler plus checkpoint persistence, the frozen-model report, and gated
selection via the checkpoint mixin base, so the train file stays inside
the review-size ceiling.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.training.loop_batch import (
    _model_forward as _model_forward,
)
from hydra2.training.loop_batch import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)
from hydra2.training.loop_batch import (
    _window_means as _window_means,
)
from hydra2.training.loop_engine import (
    SupervisedLoopEngineMixin as SupervisedLoopEngineMixin,
)
from hydra2.training.loop_loss import (
    SupervisedLoopLossMixin as SupervisedLoopLossMixin,
)
from hydra2.training.objectives_loss import (
    global_grad_norm_is_finite as global_grad_norm_is_finite,
)
from hydra2.training.objectives_metrics import (
    compute_hot_scalars as compute_hot_scalars,
)

if TYPE_CHECKING:
    from pathlib import Path

    from hydra2.training.loop_state import TrainingLoopConfig as TrainingLoopConfig
    from hydra2.training.loop_state import TrainingState as TrainingState


def _save_checkpoint_with_mirror(loop: Any, entry: dict[str, float]) -> Path:
    """Submit one checkpoint generation with its mirror announcement.

    Captures the update number now (the hook may fire updates later, and
    must announce the generation it belongs to, not the then-current
    one). The hook receives the published destination from the loop's
    ``save_checkpoint``, so no box is needed regardless of when it fires.
    """
    update = int(loop.state.global_update)
    hashes = dict(loop.manifest_hashes)

    def _fire(dest: Path) -> None:
        manifest_json = {
            "checkpoint_file": dest.name,
            "global_update": update,
            "manifest_hashes": hashes,
        }
        loop._mirror.log_update(entry, step=update)
        loop._mirror.log_checkpoint(
            checkpoint_path=dest,
            manifest_json=manifest_json,
        )
        loop._mlflow_mirror.log_update(entry, step=update)
        loop._mlflow_mirror.log_checkpoint(
            checkpoint_path=dest,
            manifest_json=manifest_json,
        )

    return loop.save_checkpoint(on_published=_fire)


class SupervisedLoopTrainMixin(SupervisedLoopEngineMixin, SupervisedLoopLossMixin):
    """Accumulation loop for :class:`SupervisedLoop`.

    Split host for the train half of :class:`SupervisedLoop`; the
    loop subclass adds nothing and no overrides. Attribute access is
    duck-typed through the subclass.
    """

    model: Any
    optimizer: Any
    scheduler: Any | None
    dataset: Any
    config: TrainingLoopConfig
    handle: Any | None
    checkpoint_dir: Path
    device: Any
    manifest_hashes: dict[str, str]
    state: TrainingState
    loss_history: list[dict[str, float]]
    _global_metrics_history: list[dict[str, float]]
    privileged_source: Any | None
    evaluation_wall_ids: frozenset[str]
    telemetry_path: Path | None
    feed: Any | None
    telemetry_records: Any
    update_records: Any
    _compiled_loss: Any
    _mirror: Any
    _mlflow_mirror: Any
    save_checkpoint: Any
    evaluate_report: Any
    evaluate_selection: Any
    maybe_promote_best: Any
    resume_from_checkpoint: Any

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

        self.model.train()
        # Fresh run owns its telemetry: reset records, truncate the JSONL file.
        self.telemetry_records = []
        self.update_records = []
        telemetry_path = self.telemetry_path
        if telemetry_path is not None:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            # intentionally discarded: char count unneeded for truncation
            _ = telemetry_path.write_text("", encoding="utf-8")

        # Gated fetch prefetch (feed attached only; depth from
        # ``fetch_prefetch_depth``): the whole next_batch chain (stream pull
        # + expand + encode) runs on one background thread while the main
        # thread does H2D + compute, so fetch_decode overlaps compute and
        # _h2d_batch queue_wait covers the slot-recycle wait. Single worker
        # FIFO preserves order exactly (byte-identical batches, deterministic
        # cursor); sampler snapshots travel with the batch (consumed, not
        # live head) so checkpoints resume without skipping
        # prefetched-but-unconsumed rows. Feed None keeps the serial sync
        # fallback (no thread, queue_wait 0.0).
        use_prefetch = self.feed is not None
        total_mb = max_u * self.config.accumulation_steps
        _prefetch_ex: ThreadPoolExecutor | None = None
        _pending: deque[Any] = deque()
        _fetched = 0
        last_snap: dict[str, Any] | None = None
        if use_prefetch:
            _prefetch_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h2d-prefetch")
            _depth = max(1, min(self.config.fetch_prefetch_depth, total_mb))
            while _fetched < _depth:
                _pending.append(
                    _prefetch_ex.submit(
                        self._acquire_cpu_batch_with_state, self.config.microbatch_size
                    )
                )
                _fetched += 1
        while self.state.global_update < target_global:
            # Accumulation window — loss scalars stay on-device as
            # tensors (no per-microbatch .item() syncs); logging aggregates
            # once per update in the logging phase below.
            micro_total_tensors: list[torch.Tensor] = []
            micro_policy_tensors: list[torch.Tensor] = []
            micro_placement_tensors: list[torch.Tensor] = []
            micro_value_tensors: list[torch.Tensor] = []
            micro_event_tensors: list[torch.Tensor] = []
            micro_belief_tensors: list[torch.Tensor] = []
            micro_event_head_tensors: dict[str, list[torch.Tensor]] = {}
            micro_belief_head_tensors: dict[str, list[torch.Tensor]] = {}
            batch: dict[str, Any] = {}
            model_out: dict[str, Any] = {}
            # Zero grad at start of accumulation window
            # (already zeroed after previous update)
            for _acc_step in range(self.config.accumulation_steps):
                if len(_pending) > 0:
                    raw_batch, fetch_decode_ms, _snap = _pending.popleft().result()
                    last_snap = _snap
                    # Lookahead: refill to configured depth while main does H2D+compute.
                    if _fetched < total_mb and _prefetch_ex is not None:
                        _pending.append(
                            _prefetch_ex.submit(
                                self._acquire_cpu_batch_with_state,
                                self.config.microbatch_size,
                            )
                        )
                        _fetched += 1
                else:
                    raw_batch, fetch_decode_ms = self._acquire_cpu_batch(
                        self.config.microbatch_size
                    )
                    last_snap = None
                _validate_batch_no_privileged(raw_batch)
                raw_batch = self._maybe_join_oracle_targets(
                    raw_batch, evaluation_wall_ids=self.evaluation_wall_ids
                )
                batch, queue_wait_ms, h2d_ms = self._h2d_batch(raw_batch)
                # AMP: autocast covers forward+loss only; backward/clip/step below stay fp32.
                # Stage split: forward vs loss timed separately inside
                # the shared autocast scope; backward outside; compute_ms keeps
                # the total wall for back-compat.
                compute_t0 = time.perf_counter()
                with self._forward_autocast():
                    _fwd_t0 = time.perf_counter()
                    model_out = _model_forward(self.model, batch)
                    forward_ms = (time.perf_counter() - _fwd_t0) * 1000.0
                    _loss_t0 = time.perf_counter()
                    losses = self._validated_loss(model_out, batch)
                    loss_ms = (time.perf_counter() - _loss_t0) * 1000.0
                # Accumulation: scale loss so that sum over accumulation_steps
                # equals mean over optimizer minibatch (exact numerator/count).
                step_total: torch.Tensor = losses["total"]
                scaled: torch.Tensor = step_total / self.config.accumulation_steps
                _bwd_t0 = time.perf_counter()
                self._backward(scaled)
                backward_ms = (time.perf_counter() - _bwd_t0) * 1000.0
                compute_ms = (time.perf_counter() - compute_t0) * 1000.0
                self._record_microbatch_telemetry(
                    queue_wait_ms=queue_wait_ms,
                    fetch_decode_ms=fetch_decode_ms,
                    h2d_ms=h2d_ms,
                    compute_ms=compute_ms,
                    forward_ms=forward_ms,
                    loss_ms=loss_ms,
                    backward_ms=backward_ms,
                )
                micro_total_tensors.append(step_total.detach())
                placement_loss: torch.Tensor = losses["placement"]
                value_loss: torch.Tensor = losses["value"]
                event_loss: torch.Tensor = losses["event"]
                belief_loss: torch.Tensor = losses["belief"]
                policy_loss: torch.Tensor = losses["policy"]
                micro_placement_tensors.append(placement_loss.detach())
                micro_value_tensors.append(value_loss.detach())
                micro_event_tensors.append(event_loss.detach())
                micro_belief_tensors.append(belief_loss.detach())
                micro_policy_tensors.append(policy_loss.detach())
                _event_heads_any: Any = losses.get("_event_per_head", {})
                if isinstance(_event_heads_any, dict):
                    for _head_id, _head_loss in _event_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _event_head_id: object = _head_id
                            _dst = micro_event_head_tensors.setdefault(str(_event_head_id), [])
                            _dst.append(_head_loss.detach())
                _belief_heads_any: Any = losses.get("_belief_per_head", {})
                if isinstance(_belief_heads_any, dict):
                    for _head_id, _head_loss in _belief_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _belief_head_id: object = _head_id
                            _dst = micro_belief_head_tensors.setdefault(str(_belief_head_id), [])
                            _dst.append(_head_loss.detach())
                self.state.sampler_cursor = (
                    last_snap
                    if (use_prefetch and last_snap is not None)
                    else self._sampler_state_snapshot()
                )
            # Fail-closed finite-grad skip: per-update global grad-norm finite
            # check BEFORE clip/step. Non-finite grads skip the optimizer (and
            # scheduler) step, zero grads, and count the skip — weights are
            # never poisoned. Mid-window non-finite *loss* still raises
            # ContractError inside compute_supervised_loss (before backward),
            # so there is no double-count: this gate only sees grads.
            # Optimizer phase timed (probe+clip+step+scheduler+zero);
            # logging phase timed separately below (deferred single-sync means).
            _opt_t0 = time.perf_counter()
            _opt_debug = os.environ.get("HYDRA2_OPT_DEBUG") == "1" and self.state.global_update < 8
            _grads_finite, _grad_norm = global_grad_norm_is_finite(self.model)
            _t_probe = time.perf_counter()
            if not _grads_finite:
                self.optimizer.zero_grad(set_to_none=True)
                _optimizer_ms = (time.perf_counter() - _opt_t0) * 1000.0
                _log_t0 = time.perf_counter()
                self.state.skipped_updates += 1
                self.state.global_update += 1
                snap_epoch: int = (
                    last_snap.get("epoch", 0)
                    if use_prefetch and last_snap is not None
                    else self._sampler_state_snapshot().get("epoch", 0)
                )
                self.state.epoch = snap_epoch
                self.state.semantic_rng_state = None

                # Deferred single-sync means over the on-device window tensors
                # (one host sync for all heads; see _window_means).
                _skip_means = _window_means(
                    micro_total_tensors,
                    micro_policy_tensors,
                    micro_placement_tensors,
                    micro_value_tensors,
                    micro_event_tensors,
                    micro_belief_tensors,
                )
                _skip_avg = _skip_means[0]
                _skip_entry: dict[str, float] = {
                    "global_update": float(self.state.global_update),
                    "total": _skip_avg,
                    "policy": _skip_means[1],
                    "placement": _skip_means[2],
                    "value": _skip_means[3],
                    "event": _skip_means[4],
                    "belief": _skip_means[5],
                    "masked_nll": _skip_avg,
                    "top1": 0.0,
                    "skipped_updates": float(self.state.skipped_updates),
                    "skipped_this_update": 1.0,
                }
                # Optimizer health (successor observability; no new host sync):
                # grad probe is non-finite here so pre/post keys are omitted
                # (sidecar + mirror drop non-finite); lr reads pre-skip value
                # since the scheduler is not stepped on skip — correct.
                try:
                    _skip_lr = float(self.optimizer.param_groups[0]["lr"])
                except (KeyError, IndexError, TypeError, ValueError):
                    _skip_lr = float("nan")
                if _skip_lr == _skip_lr and abs(_skip_lr) != float("inf"):
                    _skip_entry["lr_now"] = _skip_lr
                self.loss_history.append(_skip_entry)
                self._global_metrics_history.append({"masked_nll": _skip_avg})
                _logging_ms = (time.perf_counter() - _log_t0) * 1000.0
                self._record_update_telemetry(optimizer_ms=_optimizer_ms, logging_ms=_logging_ms)
                if (
                    self.state.global_update % self.config.checkpoint_frequency_updates == 0
                    or self.state.global_update == target_global
                ):
                    _ = _save_checkpoint_with_mirror(self, _skip_entry)
                # Per-update writer poll: cheap when idle; fail closed on error.
                self._poll_checkpoint_writer()
                continue
            if self.config.gradient_clip_norm is not None:
                model_params: Any = self.model.parameters()
                _clipped: torch.Tensor = torch.nn.utils.clip_grad_norm_(
                    model_params, self.config.gradient_clip_norm
                )

            _t_clip = time.perf_counter()
            self.optimizer.step()
            _t_step = time.perf_counter()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            _optimizer_ms = (time.perf_counter() - _opt_t0) * 1000.0
            if _opt_debug:
                print(
                    f"phase: opt u={self.state.global_update} "
                    f"probe={(_t_probe - _opt_t0) * 1000:.1f}ms "
                    f"clip={(_t_clip - _t_probe) * 1000:.1f}ms "
                    f"step={(_t_step - _t_clip) * 1000:.1f}ms "
                    f"tail={(_optimizer_ms - (_t_step - _opt_t0) * 1000):.1f}ms",
                    flush=True,
                )

            self.state.global_update += 1
            snap_epoch2: int = (
                last_snap.get("epoch", 0)
                if use_prefetch and last_snap is not None
                else self._sampler_state_snapshot().get("epoch", 0)
            )
            self.state.epoch = snap_epoch2
            self.state.semantic_rng_state = None  # populated at checkpoint via capture_rng_state

            # Logging: mean over accumulation window + per-head metrics on last microbatch
            # Single-sync means over the deferred window tensors (one
            # host sync per head per update, not per microbatch); metrics on
            # the last microbatch stay lawful peeks. Checkpoint save stays
            # outside logging_ms (cadence stall reviewed separately).
            _log_t0 = time.perf_counter()

            # Recompute metrics for reporting (masked NLL, top-k, etc.) on last logits
            # We reuse last batch/model_out already in scope; recompute with stored batch
            # One host sync for every head mean (bitwise-identical values; see
            # _window_means). Fixed-order lists: 6 trunk heads, then per-head
            # event/belief windows in sorted-head order.
            _event_head_keys = sorted(micro_event_head_tensors)
            _belief_head_keys = sorted(micro_belief_head_tensors)
            _all_means = _window_means(
                micro_total_tensors,
                micro_policy_tensors,
                micro_placement_tensors,
                micro_value_tensors,
                micro_event_tensors,
                micro_belief_tensors,
                *[micro_event_head_tensors[_k] for _k in _event_head_keys],
                *[micro_belief_head_tensors[_k] for _k in _belief_head_keys],
            )
            avg_loss = _all_means[0]
            avg_policy = _all_means[1]
            avg_placement = _all_means[2]
            avg_value = _all_means[3]
            avg_event = _all_means[4]
            avg_belief = _all_means[5]
            _extra_means = _all_means[6:]
            # Hot diagnostics: masked NLL + top-1 only (2 host syncs). Richer
            # metrics (top-k/ECE/uniform/support/per-type) ride the eval
            # report; the batch was pre-validated this microbatch, so no
            # re-validation here. We have batch/model_out from last iteration
            # in scope — recompute hot scalars there.
            try:
                train_logits: torch.Tensor = model_out["policy_logits"]
                train_targets: torch.Tensor = batch["chosen_action_id"]
                train_mask: torch.Tensor = batch["legal_mask"]
                metrics = compute_hot_scalars(
                    train_logits.detach(),
                    train_targets.detach(),
                    train_mask.detach(),
                )
            except Exception:
                metrics = {
                    "masked_nll": avg_loss,
                    "top1": 0.0,
                }

            entry: dict[str, float] = {
                "global_update": float(self.state.global_update),
                "total": avg_loss,
                "policy": avg_policy,
                "placement": avg_placement,
                "value": avg_value,
                "event": avg_event,
                "belief": avg_belief,
                "masked_nll": metrics.get("masked_nll", avg_loss),
                "top1": metrics.get("top1", 0.0),
                "skipped_updates": float(self.state.skipped_updates),
                "skipped_this_update": 0.0,
            }
            # Optimizer health from already-in-scope values (no new host sync):
            # pre is the fused probe at :281, post is pure-arithmetic mirror of
            # clip_grad_norm_ at :333-337, lr is group-0 trunk base after
            # scheduler.step() (head multiplier rides separately).
            try:
                _pre = float(_grad_norm)
            except (TypeError, ValueError):
                _pre = float("nan")
            if _pre == _pre and abs(_pre) != float("inf"):
                entry["grad_norm_pre"] = _pre
                _clip = self.config.gradient_clip_norm
                if isinstance(_clip, (int, float)) and float(_clip) > 0:
                    _post = min(float(_clip), _pre)
                else:
                    _post = _pre
                entry["grad_norm_post"] = _post
            try:
                _lr_now = float(self.optimizer.param_groups[0]["lr"])
            except (KeyError, IndexError, TypeError, ValueError):
                _lr_now = float("nan")
            if _lr_now == _lr_now and abs(_lr_now) != float("inf"):
                entry["lr_now"] = _lr_now
            for _event_pos, _event_head in enumerate(_event_head_keys):
                entry[f"event_{_event_head}"] = _extra_means[_event_pos]
            _belief_base = len(_event_head_keys)
            for _belief_pos, _belief_head in enumerate(_belief_head_keys):
                entry[f"belief_{_belief_head}"] = _extra_means[_belief_base + _belief_pos]
            self.loss_history.append(entry)
            self._global_metrics_history.append(metrics)
            _logging_ms = (time.perf_counter() - _log_t0) * 1000.0
            self._record_update_telemetry(optimizer_ms=_optimizer_ms, logging_ms=_logging_ms)

            # Checkpointing: local authoritative artifact, atomic publish.
            # The save snapshots to CPU and publishes on the background
            # writer; the mirror hook fires only after that generation lands
            # (observer-only, never ahead of the bytes). The per-update poll
            # below surfaces a failed save at the next update boundary.
            if (
                self.state.global_update % self.config.checkpoint_frequency_updates == 0
                or self.state.global_update == target_global
            ):
                _ = _save_checkpoint_with_mirror(self, entry)
            # Per-update writer poll: cheap when idle; fail closed on error.
            self._poll_checkpoint_writer()

        # Flush any generation that landed on the final update so one-shot
        # train() callers observe its mirror announcement on return.
        self._poll_checkpoint_writer()
        if _prefetch_ex is not None:
            with contextlib.suppress(Exception):
                _prefetch_ex.shutdown(wait=True)
        if self.telemetry_path is not None:
            summary_line: dict[str, Any] = {
                "kind": "summary",
                "microbatches": len(self.telemetry_records),
                **self.telemetry_summary(),
            }
            with open(self.telemetry_path, "a", encoding="utf-8") as sink:
                payload = json.dumps(summary_line, sort_keys=True) + "\n"
                _ = sink.write(payload)  # intentionally discarded: byte count unneeded

        return list(self.loss_history)


class SupervisedLoop(SupervisedLoopTrainMixin):
    """Project-owned supervised training loop (masked behavior cloning with
    explicit weights; resume restores model, optimizer, scheduler, step, RNG,
    sampler, and manifest identities; masked NLL, top-k, calibration
    reported).

    Thin subclass joining the split mixins; construction, the
    validated-loss step, the accumulation loop, checkpoint persistence,
    the frozen-model report, and the gated selection pair all live in
    the ``loop_*`` modules with no overrides here.
    """


__all__ = [
    "SupervisedLoop",
    "SupervisedLoopTrainMixin",
]
