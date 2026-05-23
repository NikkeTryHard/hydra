#!/usr/bin/env python3
"""Hydra PyTorch base-head BC learner."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import IO, Literal, cast, override

import torch
import torch.cuda.profiler as cuda_profiler
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
from hydra_learner.checkpoint import (
    EmaConfig,
    ModelConfig,
    OptimizerConfig,
    ResumeState,
    RuntimeConfig,
    load_checkpoint,
    save_checkpoint,
)
from hydra_learner.losses import (
    BaseTargets,
    LossWeights,
    base_loss,
    bce_logits_mean,
    danger_focal_bce,
    loss_breakdown_dict,
    masked_policy_ce_indices,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    soft_ce,
    target_coverage_dict,
    value_mse,
)
from hydra_learner.metrics import EvalStats, StepStats, summarize_eval, summarize_steps
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_DEFAULT,
    BACKBONE_PROFILES,
    CONV_MEMORY_FORMAT_DEFAULT,
    CONV_MEMORY_FORMATS,
    DEFAULT_BLOCKS,
    DEFAULT_HIDDEN,
    DEFAULT_SE_BOTTLENECK,
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILES,
    HydraPolicyNet,
)
from hydra_learner.raw_mjai_stream import (
    RAW_MJAI_TRANSPORT_PINNED_PYO3,
    RAW_MJAI_TRANSPORT_STDOUT,
    BuildProgress,
    PinnedPolicyBatch,
    RawMjaiBridgeStats,
    RawMjaiDirectStream,
    RawMjaiPinnedQueueStats,
    RawMjaiPinnedStream,
    add_raw_mjai_args,
    build_progress_json,
    default_raw_mjai_pinned_library_path,
    raw_mjai_config_from_args,
    validate_raw_mjai_source_args,
)
from hydra_learner.shards import BcShardDataset, ManifestSummary, PolicyBatch, validate_manifest

VARIANTS = ("eager_fp32", "eager_bf16", "compile_default", "compile_reduce_overhead", "compile_max_autotune")
PYTHON_VARIANT_DEFAULT = "compile_max_autotune"
LOSS_MODES = ("policy_only", "full_base")
COMPILED_LOSS_MODES = ("policy_only", "full_base")
ADAMW_FLAG_MODES = ("auto", "on", "off")
LR_SCHEDULES = ("constant", "cosine")
VALIDATION_SOURCE_MODES = ("fixed", "streaming")


WARMUP_MODE = "non_mutating_replay_first_batch"
COMPILE_DRY_RUN_MODE = "snapshot_restore_first_batch"


@dataclass(frozen=True)
class PreMainAccounting:
    compile_dry_run: bool = False
    compile_dry_run_changed_weights: bool = False
    warmup_mode: str = WARMUP_MODE
    warmup_steps_counted: int = 0
    warmup_steps_run: int = 0
    samples_consumed_pre_main: int = 0
    batches_consumed_pre_main: int = 0
    pre_main_batches_changed_weights: bool = False

    def as_log_fields(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StagedTrainBatch:
    obs: torch.Tensor
    targets: BaseTargets
    input_timing: InputTiming
    pinned_batch: PinnedPolicyBatch | None = None


@dataclass(frozen=True)
class ValidationSourceInfo:
    mode: str
    requested_batches: int
    actual_batches: int
    requested_samples: int | None
    actual_samples: int
    sample_cap_overrun: int
    full_batches: bool
    augment: bool


CachedValidationBatch = tuple[PolicyBatch, float]


class RawMjaiValidationSource:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        events: JsonlLogger,
    ) -> None:
        self.mode: str = args.validation_source_mode
        self._cached: list[CachedValidationBatch] = []
        self._stream: RawMjaiDirectStream | None = None
        self._index = 0
        self.info = ValidationSourceInfo(
            mode=self.mode,
            requested_batches=0,
            actual_batches=0,
            requested_samples=args.validation_max_samples,
            actual_samples=0,
            sample_cap_overrun=0,
            full_batches=True,
            augment=args.raw_mjai_validation_augment,
        )
        if args.validation_steps <= 0 or not args.raw_mjai_data_dirs:
            return
        stream = RawMjaiDirectStream(
            data_dirs=args.raw_mjai_data_dirs,
            batch_size=args.batch,
            prefetch_batches=args.raw_mjai_prefetch_batches,
            queue_bound=args.raw_mjai_queue_bound,
            worker_threads=args.raw_mjai_worker_threads,
            max_games=args.raw_mjai_max_games,
            max_samples=args.validation_max_samples if self.mode == "fixed" else None,
            train_fraction=args.raw_mjai_train_fraction,
            augment=args.raw_mjai_validation_augment,
            split="validation",
        )
        stream.start()
        if self.mode == "fixed":
            try:
                for _ in range(args.validation_steps):
                    self._cached.append(stream.next_batch())
            except StopIteration as exc:
                raise ValueError(
                    "raw MJAI fixed validation window exhausted before --validation-steps batches"
                ) from exc
            finally:
                stream.close()
            samples = sum(batch.actions.shape[0] for batch, _fetch_ms in self._cached)
            batches = len(self._cached)
            full_batches = all(batch.actions.shape[0] == args.batch for batch, _fetch_ms in self._cached)
        else:
            self._stream = stream
            samples = 0
            batches = 0
            full_batches = True
        overrun = 0 if args.validation_max_samples is None else max(0, samples - args.validation_max_samples)
        self.info = ValidationSourceInfo(
            mode=self.mode,
            requested_batches=args.validation_steps,
            actual_batches=batches,
            requested_samples=args.validation_max_samples,
            actual_samples=samples,
            sample_cap_overrun=overrun,
            full_batches=full_batches,
            augment=args.raw_mjai_validation_augment,
        )
        events.write("validation_source", asdict(self.info))

    def next_batch(self) -> CachedValidationBatch:
        if self.mode == "fixed":
            if not self._cached:
                raise StopIteration("fixed validation window is empty")
            item = self._cached[self._index]
            self._index = (self._index + 1) % len(self._cached)
            return item
        if self._stream is None:
            raise StopIteration("streaming validation source is not open")
        return self._stream.next_batch()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def collect_validation_batches(source: RawMjaiValidationSource, *, steps: int) -> list[CachedValidationBatch]:
    return [source.next_batch() for _ in range(steps)]


def evaluate_validation_batches(
    batches: list[CachedValidationBatch],
    *,
    model: nn.Module,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
) -> dict[str, object]:
    step_eval: list[EvalStats] = []
    for val_batch, val_fetch_ms in batches:
        val_obs, _val_legal, _val_labels, val_targets, _val_input_timing = tensors_from_policy_batch(
            val_batch, device, val_fetch_ms
        )
        val_targets = targets_for_compiled_loss(val_targets, weights)
        step_eval.append(evaluate_batch(model, val_obs, val_targets, weights, autocast))
    return summarize_eval(step_eval)


def evaluate_validation_source(
    source: RawMjaiValidationSource,
    *,
    steps: int,
    model: nn.Module,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
) -> dict[str, object]:
    batches = collect_validation_batches(source, steps=steps)
    return evaluate_validation_batches(
        batches,
        model=model,
        device=device,
        weights=weights,
        autocast=autocast,
    )


RAW_MJAI_CURSOR_RESUME_ERROR = (
    "raw-MJAI resume is unsupported: checkpoint restores weights but raw stream cursor resume is "
    "unsupported; use fresh output dir or BC shards"
)


def adamw_flag_value(mode: str) -> bool | None:
    if mode == "auto":
        return None
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"unsupported AdamW flag mode {mode!r}")


@dataclass(frozen=True)
class LrSchedulerConfig:
    base_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int | None
    schedule: str


class LrScheduler:
    def __init__(self, config: LrSchedulerConfig) -> None:
        self.config = config

    def lr_for_step(self, completed_steps: int) -> float:
        if self.config.schedule == "constant":
            return self.config.base_lr
        if self.config.warmup_steps > 0 and completed_steps < self.config.warmup_steps:
            return self.config.base_lr * (completed_steps / self.config.warmup_steps)
        if self.config.schedule != "cosine":
            raise ValueError(f"unsupported LR schedule {self.config.schedule!r}")
        total_steps = self.config.total_steps
        if total_steps is None:
            raise ValueError("--lr-schedule cosine requires --schedule-total-steps or --steps")
        decay_steps = max(1, total_steps - self.config.warmup_steps)
        decay_index = min(max(0, completed_steps - self.config.warmup_steps), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_index / decay_steps))
        return self.config.min_lr + (self.config.base_lr - self.config.min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_optimizer_config(args: argparse.Namespace) -> OptimizerConfig:
    return OptimizerConfig(
        name="AdamW",
        lr=args.lr,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        schedule_total_steps=args.schedule_total_steps,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
        foreach=adamw_flag_value(args.adamw_foreach),
        fused=adamw_flag_value(args.adamw_fused),
    )


def build_lr_scheduler_config(args: argparse.Namespace) -> LrSchedulerConfig:
    return LrSchedulerConfig(
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.lr_warmup_steps,
        total_steps=args.schedule_total_steps,
        schedule=args.lr_schedule,
    )


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=config.foreach,
        fused=config.fused,
    )


@dataclass(frozen=True)
class InputTiming:
    fetch_decode_ms: float = math.nan
    h2d_wall_ms: float = math.nan


def make_synthetic_batch(
    batch: int, actions: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(0xC0FFEE)
    obs = torch.randn(batch, 192, 34, device=device, generator=gen)
    legal = torch.rand(batch, actions, device=device, generator=gen) > 0.20
    labels = torch.randint(actions, (batch,), device=device, generator=gen)
    legal[torch.arange(batch, device=device), labels] = True
    return obs, legal, labels


def synthetic_targets(obs: torch.Tensor, legal: torch.Tensor, labels: torch.Tensor) -> BaseTargets:
    batch = obs.shape[0]
    device = obs.device
    grp = torch.zeros(batch, 24, device=device)
    grp[:, 0] = 1.0
    tenpai = torch.zeros(batch, 3, device=device)
    danger = torch.zeros(batch, 3, 34, device=device)
    danger_mask = torch.ones(batch, 3, 34, device=device)
    opp_next = torch.zeros(batch, 3, 34, device=device)
    opp_next[:, :, 0] = 1.0
    score_pdf = torch.zeros(batch, 64, device=device)
    score_pdf[:, 0] = 1.0
    score_cdf = torch.ones(batch, 64, device=device)
    return BaseTargets(
        policy_target=labels,
        legal_mask=legal,
        value_target=torch.zeros(batch, device=device),
        grp_target=grp,
        tenpai_target=tenpai,
        danger_target=danger,
        danger_mask=danger_mask,
        opp_next_target=opp_next,
        score_pdf_target=score_pdf,
        score_cdf_target=score_cdf,
        oracle_target=torch.zeros(batch, 4, device=device),
        oracle_target_mask=torch.ones(batch, device=device),
        safety_target=None,
        safety_mask=None,
    )


def tensors_from_policy_batch(
    batch: PolicyBatch, device: torch.device, fetch_decode_ms: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    h2d_started = time.perf_counter()
    obs = torch.from_numpy(batch.obs).to(device=device, non_blocking=True)
    legal = torch.from_numpy(batch.legal_mask).to(device=device, non_blocking=True)
    labels = torch.from_numpy(batch.actions).to(device=device, non_blocking=True)
    targets = targets_from_policy_batch(batch, device, labels, legal)
    h2d_wall_ms = (time.perf_counter() - h2d_started) * 1000.0
    if obs.shape[1:] != (192, 34) or obs.dtype != torch.float32:
        raise ValueError(f"real shard obs contract mismatch: shape={tuple(obs.shape)} dtype={obs.dtype}")
    if legal.shape != (obs.shape[0], ACTION_SPACE) or legal.dtype != torch.bool:
        raise ValueError(f"real shard legal-mask contract mismatch: shape={tuple(legal.shape)} dtype={legal.dtype}")
    if labels.shape != (obs.shape[0],) or labels.dtype != torch.int64:
        raise ValueError(f"real shard action contract mismatch: shape={tuple(labels.shape)} dtype={labels.dtype}")
    return (
        obs,
        legal,
        labels,
        targets,
        InputTiming(
            fetch_decode_ms=fetch_decode_ms,
            h2d_wall_ms=h2d_wall_ms,
        ),
    )


def tensors_from_pinned_policy_batch(
    batch: PinnedPolicyBatch, device: torch.device, fetch_decode_ms: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    h2d_started = time.perf_counter()
    obs = batch.obs.to(device=device, non_blocking=True)
    legal = batch.legal_mask.to(device=device, non_blocking=True)
    labels = batch.actions.to(device=device, non_blocking=True)
    targets = BaseTargets(
        policy_target=labels,
        legal_mask=legal,
        value_target=batch.value_target.to(device=device, non_blocking=True),
        grp_target=batch.grp_target.to(device=device, non_blocking=True),
        tenpai_target=batch.tenpai.to(device=device, non_blocking=True),
        danger_target=batch.danger.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        danger_mask=batch.danger_mask.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        opp_next_target=batch.opp_next.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        score_pdf_target=batch.score_pdf.to(device=device, non_blocking=True),
        score_cdf_target=batch.score_cdf.to(device=device, non_blocking=True),
        oracle_target=batch.oracle_target.to(device=device, non_blocking=True),
        oracle_target_mask=batch.oracle_target_mask.to(device=device, non_blocking=True),
        safety_target=None,
        safety_mask=None,
    )
    h2d_wall_ms = (time.perf_counter() - h2d_started) * 1000.0
    if obs.shape[1:] != (192, 34) or obs.dtype != torch.float32:
        raise ValueError(f"pinned raw MJAI obs contract mismatch: shape={tuple(obs.shape)} dtype={obs.dtype}")
    if legal.shape != (obs.shape[0], ACTION_SPACE) or legal.dtype != torch.bool:
        raise ValueError(
            f"pinned raw MJAI legal-mask contract mismatch: shape={tuple(legal.shape)} dtype={legal.dtype}"
        )
    if labels.shape != (obs.shape[0],) or labels.dtype != torch.int64:
        raise ValueError(f"pinned raw MJAI action contract mismatch: shape={tuple(labels.shape)} dtype={labels.dtype}")
    return (
        obs,
        legal,
        labels,
        targets,
        InputTiming(fetch_decode_ms=fetch_decode_ms, h2d_wall_ms=h2d_wall_ms),
    )


def tensors_from_real_batch(
    dataset: BcShardDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    batch = dataset.next_batch()
    return tensors_from_policy_batch(batch, device, dataset.last_fetch_decode_ms)


def stage_train_batch(
    obs: torch.Tensor,
    targets: BaseTargets,
    input_timing: InputTiming,
    weights: LossWeights,
    pinned_batch: PinnedPolicyBatch | None = None,
) -> StagedTrainBatch:
    return StagedTrainBatch(
        obs=obs,
        targets=targets_for_compiled_loss(targets, weights),
        input_timing=input_timing,
        pinned_batch=pinned_batch,
    )


def next_staged_train_batch(
    *,
    real_dataset: BcShardDataset | None,
    raw_stream: RawMjaiDirectStream | None,
    raw_pinned: RawMjaiPinnedStream | None,
    device: torch.device,
    weights: LossWeights,
) -> StagedTrainBatch:
    if raw_stream is not None:
        batch, fetch_ms = raw_stream.next_batch()
        obs, _legal, _labels, targets, input_timing = tensors_from_policy_batch(batch, device, fetch_ms)
        return stage_train_batch(obs, targets, input_timing, weights)
    if raw_pinned is not None:
        pinned_batch, fetch_ms = raw_pinned.next_batch()
        obs, _legal, _labels, targets, input_timing = tensors_from_pinned_policy_batch(pinned_batch, device, fetch_ms)
        return stage_train_batch(obs, targets, input_timing, weights, pinned_batch)
    if real_dataset is not None:
        obs, _legal, _labels, targets, input_timing = tensors_from_real_batch(real_dataset, device)
        return stage_train_batch(obs, targets, input_timing, weights)
    raise ValueError("internal data source selection failed")


def targets_from_policy_batch(
    batch: PolicyBatch, device: torch.device, labels: torch.Tensor | None = None, legal: torch.Tensor | None = None
) -> BaseTargets:
    if labels is None:
        labels = torch.from_numpy(batch.actions).to(device=device)
    if legal is None:
        legal = torch.from_numpy(batch.legal_mask).to(device=device)
    return BaseTargets(
        policy_target=labels,
        legal_mask=legal,
        value_target=torch.from_numpy(batch.value_target).to(device=device),
        grp_target=torch.from_numpy(batch.grp_target).to(device=device),
        tenpai_target=torch.from_numpy(batch.tenpai).to(device=device),
        danger_target=torch.from_numpy(batch.danger.reshape(batch.danger.shape[0], 3, 34)).to(device=device),
        danger_mask=torch.from_numpy(batch.danger_mask.reshape(batch.danger_mask.shape[0], 3, 34)).to(device=device),
        opp_next_target=torch.from_numpy(batch.opp_next.reshape(batch.opp_next.shape[0], 3, 34)).to(device=device),
        score_pdf_target=torch.from_numpy(batch.score_pdf).to(device=device),
        score_cdf_target=torch.from_numpy(batch.score_cdf).to(device=device),
        oracle_target=torch.from_numpy(batch.oracle_target).to(device=device),
        oracle_target_mask=torch.from_numpy(batch.oracle_target_mask).to(device=device),
        safety_target=None if batch.safety_target is None else torch.from_numpy(batch.safety_target).to(device=device),
        safety_mask=None if batch.safety_mask is None else torch.from_numpy(batch.safety_mask).to(device=device),
    )


def targets_for_compiled_loss(targets: BaseTargets, weights: LossWeights) -> BaseTargets:
    batch = targets.policy_target.shape[0]
    if weights.oracle_critic > 0.0:
        oracle_target = targets.oracle_target
        oracle_target_mask = targets.oracle_target_mask
        if oracle_target is None or oracle_target_mask is None:
            raise ValueError("oracle targets are required when oracle_critic loss weight is positive")
    else:
        oracle_target = targets.oracle_target
        if oracle_target is None:
            oracle_target = targets.value_target.new_zeros((batch, 4))
        oracle_target_mask = targets.oracle_target_mask
        if oracle_target_mask is None:
            oracle_target_mask = targets.value_target.new_zeros((batch,))
    if weights.safety_residual > 0.0:
        safety_target = targets.safety_target
        safety_mask = targets.safety_mask
        if safety_target is None or safety_mask is None:
            raise ValueError("safety targets are required when safety_residual loss weight is positive")
    else:
        safety_target = targets.safety_target
        if safety_target is None:
            safety_target = targets.value_target.new_zeros((batch, ACTION_SPACE))
        safety_mask = targets.safety_mask
        if safety_mask is None:
            safety_mask = targets.value_target.new_zeros((batch, ACTION_SPACE))
    return BaseTargets(
        policy_target=targets.policy_target,
        legal_mask=targets.legal_mask,
        value_target=targets.value_target,
        grp_target=targets.grp_target,
        tenpai_target=targets.tenpai_target,
        danger_target=targets.danger_target,
        danger_mask=targets.danger_mask,
        opp_next_target=targets.opp_next_target,
        score_pdf_target=targets.score_pdf_target,
        score_cdf_target=targets.score_cdf_target,
        oracle_target=oracle_target,
        oracle_target_mask=oracle_target_mask,
        safety_target=safety_target,
        safety_mask=safety_mask,
    )


def cuda_event_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return start.elapsed_time(end)


def time_cuda(fn: Callable[[], torch.Tensor | None]) -> tuple[float, torch.Tensor | None]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    return cuda_event_elapsed(start, end), out


class HydraCompiledLossStep(nn.Module):
    def __init__(self, model: nn.Module, loss_mode: str, weights: LossWeights) -> None:
        super().__init__()
        self.model = model
        self.loss_mode = loss_mode
        self.weights = weights

    @override
    def forward(
        self,
        obs: torch.Tensor,
        policy_target: torch.Tensor,
        legal_mask: torch.Tensor,
        value_target: torch.Tensor,
        grp_target: torch.Tensor,
        tenpai_target: torch.Tensor,
        danger_target: torch.Tensor,
        danger_mask: torch.Tensor,
        opp_next_target: torch.Tensor,
        score_pdf_target: torch.Tensor,
        score_cdf_target: torch.Tensor,
        oracle_target: torch.Tensor,
        oracle_target_mask: torch.Tensor,
        safety_target: torch.Tensor,
        safety_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(obs)
        if self.loss_mode == "policy_only":
            return masked_policy_ce_indices(outputs.policy_logits, policy_target, legal_mask).mean()
        l_policy = masked_policy_ce_indices(outputs.policy_logits, policy_target, legal_mask).mean()
        l_value = value_mse(outputs.value, value_target).mean()
        l_grp = soft_ce(outputs.grp, grp_target).mean()
        l_tenpai = bce_logits_mean(outputs.opp_tenpai, tenpai_target, dim=1).mean()
        l_danger = danger_focal_bce(outputs.danger, danger_target, danger_mask).mean()
        l_opp = opp_next_ce(outputs.opp_next_discard, opp_next_target).mean()
        l_pdf = soft_ce(outputs.score_pdf, score_pdf_target).mean()
        l_cdf = bce_logits_mean(outputs.score_cdf, score_cdf_target, dim=1).mean()
        total = (
            l_policy * self.weights.policy
            + l_value * self.weights.value
            + l_grp * self.weights.grp
            + l_tenpai * self.weights.tenpai
            + l_danger * self.weights.danger
            + l_opp * self.weights.opp_next
            + l_pdf * self.weights.score
            + l_cdf * self.weights.score
        )
        if self.weights.oracle_critic > 0.0:
            total = (
                total
                + oracle_critic_loss(outputs.oracle_critic, oracle_target, oracle_target_mask)
                * self.weights.oracle_critic
            )
        if self.weights.safety_residual > 0.0:
            total = (
                total
                + safety_residual_loss(outputs.safety_residual, safety_target, safety_mask)
                * self.weights.safety_residual
            )
        return total


def loss_step_args(obs: torch.Tensor, targets: BaseTargets, start: int, end: int) -> tuple[torch.Tensor, ...]:
    oracle_target = targets.oracle_target
    if oracle_target is None:
        raise ValueError("compiled loss targets missing oracle_target")
    oracle_target_mask = targets.oracle_target_mask
    if oracle_target_mask is None:
        raise ValueError("compiled loss targets missing oracle_target_mask")
    safety_target = targets.safety_target
    if safety_target is None:
        raise ValueError("compiled loss targets missing safety_target")
    safety_mask = targets.safety_mask
    if safety_mask is None:
        raise ValueError("compiled loss targets missing safety_mask")
    return (
        obs[start:end],
        targets.policy_target[start:end],
        targets.legal_mask[start:end],
        targets.value_target[start:end],
        targets.grp_target[start:end],
        targets.tenpai_target[start:end],
        targets.danger_target[start:end],
        targets.danger_mask[start:end],
        targets.opp_next_target[start:end],
        targets.score_pdf_target[start:end],
        targets.score_cdf_target[start:end],
        oracle_target[start:end],
        oracle_target_mask[start:end],
        safety_target[start:end],
        safety_mask[start:end],
    )


def clone_state_for_restore(state: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(state)


def restore_train_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_state: dict[str, object],
    optimizer_state: dict[str, object],
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def run_step(
    loss_step: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    loss_mode: str,
    microbatch: int,
    autocast: bool,
    timed: bool,
    grad_clip_norm: float | None = None,
    collect_diagnostics: bool = False,
) -> StepStats:
    logical = obs.shape[0]
    if getattr(loss_step, "_hydra_compiled", False):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    step_start = torch.cuda.Event(enable_timing=True) if obs.device.type == "cuda" else None
    step_end = torch.cuda.Event(enable_timing=True) if obs.device.type == "cuda" else None
    step_start_wall = time.perf_counter()
    if step_start is not None:
        step_start.record()
    fwd_ms = 0.0
    bwd_ms = 0.0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    logical_loss = obs.new_zeros(())
    for start_idx in range(0, logical, microbatch):
        end_idx = min(start_idx + microbatch, logical)
        scale = (end_idx - start_idx) / logical

        def fwd_loss() -> torch.Tensor:
            with amp_ctx:
                loss = loss_step(*loss_step_args(obs, targets, start_idx, end_idx))
            return loss * scale

        if timed:
            ms, loss = time_cuda(fwd_loss)
            fwd_ms += ms
            assert loss is not None
        else:
            loss = fwd_loss()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite BC loss: {loss_value}")
        if timed:
            ms, _ = time_cuda(loss.backward)
            bwd_ms += ms
        else:
            loss.backward()
        logical_loss = logical_loss + loss.detach()

    grad_norm = math.nan
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(loss_step.parameters(), grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    if timed:
        opt_ms, _ = time_cuda(optimizer.step)
    else:
        optimizer.step()
        opt_ms = 0.0
    if step_start is not None and step_end is not None:
        step_end.record()
        step_ms = cuda_event_elapsed(step_start, step_end)
    else:
        step_ms = (time.perf_counter() - step_start_wall) * 1000.0
    if collect_diagnostics:
        with torch.inference_mode(), amp_ctx:
            outputs = model(obs)
            breakdown = base_loss(outputs, targets, weights)
        head_losses = loss_breakdown_dict(breakdown, weights, loss_mode)
        target_coverage = target_coverage_dict(targets, weights, loss_mode)
    else:
        head_losses: dict[str, float] = {}
        target_coverage: dict[str, dict[str, float | str]] = {}
    loss_value = float(logical_loss.detach())
    stat = StepStats(
        step_ms=step_ms,
        fwd_loss_ms=fwd_ms,
        backward_ms=bwd_ms,
        optimizer_ms=opt_ms,
        loss=loss_value,
        head_losses=head_losses,
        target_coverage=target_coverage,
        grad_norm=grad_norm,
    )
    stat.train_gpu_ms = step_ms
    return stat


def run_non_mutating_train_step(
    loss_step: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    loss_mode: str,
    microbatch: int,
    autocast: bool,
    timed: bool,
    grad_clip_norm: float | None = None,
) -> StepStats:
    model_state = clone_state_for_restore(model.state_dict())
    optimizer_state = clone_state_for_restore(optimizer.state_dict())
    try:
        return run_step(
            loss_step, model, optimizer, obs, targets, weights, loss_mode, microbatch, autocast, timed, grad_clip_norm
        )
    finally:
        restore_train_state(model, optimizer, model_state=model_state, optimizer_state=optimizer_state)


class EmaTracker:
    def __init__(self, model: nn.Module, config: EmaConfig, training_device: torch.device) -> None:
        self.config = config
        self.device = self._resolve_device(config.device, training_device)
        model_state = model.state_dict()
        self.keys = tuple(key for key, tensor in model_state.items() if tensor.is_floating_point())
        self.state: dict[str, torch.Tensor] = {
            key: model_state[key].detach().to(device=self.device, dtype=torch.float32).clone() for key in self.keys
        }
        self.update_count = 0
        self.last_update_step = 0
        self.last_update_ms = math.nan

    @staticmethod
    def _resolve_device(requested: str, training_device: torch.device) -> torch.device:
        if requested == "auto":
            return (
                torch.device("cuda", training_device.index) if training_device.type == "cuda" else torch.device("cpu")
            )
        if requested == "cuda":
            if training_device.type != "cuda":
                raise ValueError("--ema-device cuda requires CUDA training device")
            return torch.device("cuda", training_device.index)
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"unsupported EMA device {requested!r}")

    @property
    def device_name(self) -> str:
        return self.device.type if self.device.index is None else f"{self.device.type}:{self.device.index}"

    def load_state(self, state: dict[str, torch.Tensor], update_count: int, last_update_step: int = 0) -> None:
        if set(state) != set(self.state):
            missing = sorted(set(self.state).difference(state))
            extra = sorted(set(state).difference(self.state))
            raise ValueError(f"EMA state keys mismatch: missing={missing} extra={extra}")
        self.state = {key: state[key].detach().to(device=self.device, dtype=torch.float32).clone() for key in self.keys}
        self.update_count = update_count
        self.last_update_step = last_update_step

    def maybe_update(self, model: nn.Module, global_step: int) -> None:
        if global_step < self.config.start_step or global_step % self.config.update_every_steps != 0:
            return
        started = time.perf_counter()
        decay = self.config.decay
        one_minus_decay = 1.0 - decay
        model_state = model.state_dict()
        for key in self.keys:
            source = model_state[key].detach().to(device=self.device, dtype=torch.float32)
            self.state[key].mul_(decay).add_(source, alpha=one_minus_decay)
        self.update_count += 1
        self.last_update_step = global_step
        self.last_update_ms = (time.perf_counter() - started) * 1000.0

    def metrics(self) -> dict[str, object]:
        return {
            "ema/enabled": True,
            "ema/device": self.device_name,
            "ema/device_code": 1 if self.device.type == "cuda" else 0,
            "ema/update_count": self.update_count,
            "ema/active": self.update_count > 0,
            "ema/last_update_step": self.last_update_step,
            "ema/last_update_ms": self.last_update_ms,
        }


@contextmanager
def ema_weights(model: nn.Module, tracker: EmaTracker | None) -> Generator[None, None, None]:
    if tracker is None:
        yield
        return
    backup = {key: value.detach().clone() for key, value in model.state_dict().items() if key in tracker.state}
    try:
        target_state = {
            key: tensor.to(device=backup[key].device, dtype=backup[key].dtype) for key, tensor in tracker.state.items()
        }
        model.load_state_dict(target_state, strict=False)
        yield
    finally:
        model.load_state_dict(backup, strict=False)


def build_ema_config(args: argparse.Namespace) -> EmaConfig | None:
    if not args.ema_enabled:
        return None
    return EmaConfig(
        enabled=True,
        decay=args.ema_decay,
        start_step=args.ema_start_step,
        update_every_steps=args.ema_update_every_steps,
        device=args.ema_device,
    )


def prefixed_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def evaluate_raw_and_ema(
    source: RawMjaiValidationSource,
    *,
    steps: int,
    model: HydraPolicyNet,
    device: torch.device,
    weights: LossWeights,
    autocast: bool,
    ema_tracker: EmaTracker | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    batches = collect_validation_batches(source, steps=steps)
    raw_metrics = evaluate_validation_batches(
        batches,
        model=model,
        device=device,
        weights=weights,
        autocast=autocast,
    )
    if ema_tracker is None or ema_tracker.update_count == 0:
        return raw_metrics, None
    with ema_weights(model, ema_tracker):
        ema_metrics = evaluate_validation_batches(
            batches,
            model=model,
            device=device,
            weights=weights,
            autocast=autocast,
        )

    return raw_metrics, ema_metrics


def evaluate_batch(
    model: nn.Module,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    autocast: bool,
) -> EvalStats:
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    with torch.inference_mode(), amp_ctx:
        outputs = model(obs)
        breakdown = base_loss(outputs, targets, weights)
        masked_logits = outputs.policy_logits.masked_fill(~targets.legal_mask.to(dtype=torch.bool), -1.0e9)
        pred = masked_logits.argmax(dim=1)
        target = targets.policy_target.to(dtype=torch.int64)
        accuracy = (pred == target).to(dtype=torch.float32).mean()
        topk = masked_logits.topk(k=min(5, masked_logits.shape[1]), dim=1).indices
        top3_accuracy = (topk[:, : min(3, topk.shape[1])] == target[:, None]).any(dim=1).to(dtype=torch.float32).mean()
        top5_accuracy = (topk == target[:, None]).any(dim=1).to(dtype=torch.float32).mean()
        probs = torch.softmax(masked_logits, dim=1)
        target_probs = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1.0e-12)
        confidence = probs.max(dim=1).values
        correct = (pred == target).to(dtype=torch.float32)
        ece = obs.new_zeros(())
        for bucket in range(10):
            lower = bucket / 10.0
            upper = (bucket + 1) / 10.0
            if bucket == 9:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)
            if mask.any():
                ece = ece + mask.to(dtype=torch.float32).mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    total = float(breakdown.total.detach())
    if not math.isfinite(total):
        raise RuntimeError(f"non-finite validation BC loss: {total}")
    return EvalStats(
        loss=total,
        policy=float(breakdown.policy.detach()),
        value=float(breakdown.value.detach()),
        grp=float(breakdown.grp.detach()),
        tenpai=float(breakdown.tenpai.detach()),
        danger=float(breakdown.danger.detach()),
        opp_next=float(breakdown.opp_next.detach()),
        score_pdf=float(breakdown.score_pdf.detach()),
        score_cdf=float(breakdown.score_cdf.detach()),
        oracle_critic=float(breakdown.oracle_critic.detach()),
        safety_residual=float(breakdown.safety_residual.detach()),
        target_coverage=target_coverage_dict(targets, weights, "full_base"),
        policy_accuracy=float(accuracy.detach()),
        policy_top3_accuracy=float(top3_accuracy.detach()),
        policy_top5_accuracy=float(top5_accuracy.detach()),
        policy_nll=float((-target_probs.log()).mean().detach()),
        policy_confidence=float(confidence.mean().detach()),
        policy_ece=float(ece.detach()),
        samples=obs.shape[0],
    )


def slice_targets(targets: BaseTargets, start: int, end: int) -> BaseTargets:
    return BaseTargets(
        policy_target=targets.policy_target[start:end],
        legal_mask=targets.legal_mask[start:end],
        value_target=targets.value_target[start:end],
        grp_target=targets.grp_target[start:end],
        tenpai_target=targets.tenpai_target[start:end],
        danger_target=targets.danger_target[start:end],
        danger_mask=targets.danger_mask[start:end],
        opp_next_target=targets.opp_next_target[start:end],
        score_pdf_target=targets.score_pdf_target[start:end],
        score_cdf_target=targets.score_cdf_target[start:end],
        oracle_target=None if targets.oracle_target is None else targets.oracle_target[start:end],
        oracle_target_mask=None if targets.oracle_target_mask is None else targets.oracle_target_mask[start:end],
        safety_target=None if targets.safety_target is None else targets.safety_target[start:end],
        safety_mask=None if targets.safety_mask is None else targets.safety_mask[start:end],
    )


class ScalarEventWriter:
    def __init__(self, path: Path | None) -> None:
        self._writer: SummaryWriter | None = None
        self._file: IO[str] | None = None
        if path is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(path))
        else:
            self._file = (path / "scalars.jsonl").open("a", encoding="utf-8")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._file is not None:
            self._file.close()

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
            return
        if self._file is None:
            return
        record = {"wall_time": time.time(), "step": step, "tag": tag, "value": value}
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        if self._file is not None:
            self._file.flush()

    @property
    def enabled(self) -> bool:
        return self._writer is not None or self._file is not None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def add_scalars(
    writer: ScalarEventWriter, prefix: str, metrics: dict[str, object] | dict[str, float], step: int
) -> None:
    for key, value in metrics.items():
        number = _finite_number(value)
        if number is not None:
            writer.add_scalar(f"{prefix}/{key}", number, step)


def log_step_scalars(
    writer: ScalarEventWriter, stat: StepStats, *, batch: int, samples_seen: int, global_step: int
) -> None:
    metrics: dict[str, object] = {
        "loss": stat.loss,
        "head_losses": stat.head_losses,
        "target_coverage": stat.target_coverage,
        "step_ms": stat.step_ms,
        "fwd_loss_ms": stat.fwd_loss_ms,
        "backward_ms": stat.backward_ms,
        "optimizer_ms": stat.optimizer_ms,
        "train_gpu_ms": stat.train_gpu_ms,
        "fetch_decode_ms": stat.fetch_decode_ms,
        "h2d_wall_ms": stat.h2d_wall_ms,
        "samples_seen": samples_seen,
        "global_step": global_step,
        "lr": stat.lr,
        "grad_norm": stat.grad_norm,
    }
    for head, loss in stat.head_losses.items():
        metrics[f"loss/{head}"] = loss
    for head, coverage in stat.target_coverage.items():
        metrics[f"coverage/{head}/fraction"] = coverage["fraction"]
        metrics[f"coverage/{head}/active"] = 1.0 if coverage["active"] else 0.0
        status = coverage["status"]
        metrics[f"coverage/{head}/status_code"] = float(
            ("absent", "present_zero", "present_positive").index(str(status))
        )
    if stat.step_ms > 0.0 and math.isfinite(stat.step_ms):
        metrics["samples_per_s"] = batch * 1000.0 / stat.step_ms
    if math.isfinite(stat.fetch_decode_ms) and math.isfinite(stat.h2d_wall_ms):
        input_pipeline_ms = stat.fetch_decode_ms + stat.h2d_wall_ms
        metrics["input_pipeline_wall_ms"] = input_pipeline_ms
        if math.isfinite(stat.train_gpu_ms):
            total_wall_ms = input_pipeline_ms + stat.train_gpu_ms
            metrics["total_wall_ms"] = total_wall_ms
            if total_wall_ms > 0.0:
                metrics["end_to_end_samples_per_s"] = batch * 1000.0 / total_wall_ms
    add_scalars(writer, "train", metrics, global_step)


def log_validation_scalars(
    writer: ScalarEventWriter, metrics: dict[str, object], global_step: int, *, final: bool
) -> None:
    add_scalars(writer, "final_validation" if final else "validation", metrics, global_step)


def raw_mjai_cursor_resume_supported() -> bool:
    return False


# Raw-MJAI stream counters are local to the currently opened stream. Resume-total
# counters are labelled separately because no raw cursor has been applied.
def _progress_scalars(progress: BuildProgress | None, offsets: RawMjaiResumeOffsets) -> dict[str, object]:
    if progress is None:
        return {}
    return {
        "progress/complete": progress.complete,
        "progress/build_seconds": progress.build_seconds,
        "progress/stream_local_loaded_games": progress.loaded_games,
        "progress/stream_local_skipped_games": progress.skipped_games,
        "progress/stream_local_samples": progress.samples,
        "progress/stream_local_batches": progress.batches,
        "progress/resume_total_loaded_games": progress.loaded_games + offsets.loaded_games,
        "progress/resume_total_skipped_games": progress.skipped_games + offsets.skipped_games,
        "progress/resume_total_samples": progress.samples + offsets.samples,
        "progress/resume_total_batches": progress.batches + offsets.batches,
        "progress/max_games_reached": progress.max_games_reached,
        "progress/max_samples_reached": progress.max_samples_reached,
    }


@dataclass(frozen=True)
class RawMjaiResumeOffsets:
    loaded_games: int = 0
    skipped_games: int = 0
    samples: int = 0
    batches: int = 0

    @classmethod
    def from_resume(cls, resume_state: ResumeState | None, batch: int) -> RawMjaiResumeOffsets:
        if resume_state is None:
            return cls()
        progress = resume_state.raw_mjai_progress
        if progress:
            return cls(
                loaded_games=progress.get("loaded_games", 0),
                skipped_games=progress.get("skipped_games", 0),
                samples=progress.get("samples", resume_state.samples_seen),
                batches=progress.get("batches", resume_state.samples_seen // batch),
            )
        return cls(samples=resume_state.samples_seen, batches=resume_state.samples_seen // batch)


def apply_progress_offsets(progress: BuildProgress | None, offsets: RawMjaiResumeOffsets) -> BuildProgress | None:
    if progress is None:
        return None
    return BuildProgress(
        manifest_path=progress.manifest_path,
        complete=progress.complete,
        build_seconds=progress.build_seconds,
        loaded_games=progress.loaded_games + offsets.loaded_games,
        skipped_games=progress.skipped_games + offsets.skipped_games,
        samples=progress.samples + offsets.samples,
        batches=progress.batches + offsets.batches,
        max_games_reached=progress.max_games_reached,
        max_samples_reached=progress.max_samples_reached,
    )


def raw_mjai_progress_dict(progress: BuildProgress | None) -> dict[str, int] | None:
    if progress is None:
        return None
    data = build_progress_json(progress)
    return {key: value for key, value in data.items() if isinstance(value, int)}


def raw_mjai_progress_sections(
    progress: BuildProgress | None, offsets: RawMjaiResumeOffsets
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    return json_raw_mjai_progress(progress), json_raw_mjai_progress(apply_progress_offsets(progress, offsets))


def _bridge_scalars(stats: RawMjaiBridgeStats | None) -> dict[str, object]:
    if stats is None:
        return {}
    return {
        "bridge/open_count": stats.open_count,
        "bridge/open_scan_plan_ms": stats.open_scan_plan_ms,
        "bridge/last_next_fill_ms": stats.last_next_fill_ms,
        "bridge/last_queue_wait_ms": stats.last_queue_wait_ms,
        "bridge/last_bytes_filled": stats.last_bytes_filled,
        "bridge/last_games_consumed": stats.last_games_consumed,
    }


def _queue_scalars(stats: RawMjaiPinnedQueueStats | None) -> dict[str, object]:
    if stats is None:
        return {}
    return {
        "queue/ready_wait_ms_total": stats.ready_wait_ms_total,
        "queue/ready_wait_count": stats.ready_wait_count,
        "queue/mean_ready_wait_ms": stats.mean_ready_wait_ms,
        "queue/producer_fill_ms_total": stats.producer_fill_ms_total,
        "queue/produced_batches": stats.produced_batches,
        "queue/mean_producer_fill_ms": stats.mean_producer_fill_ms,
        "queue/producer_free_wait_ms_total": stats.producer_free_wait_ms_total,
        "queue/producer_free_wait_count": stats.producer_free_wait_count,
        "queue/mean_producer_free_wait_ms": stats.mean_producer_free_wait_ms,
        "queue/ready_queue_size": stats.ready_queue_size,
        "queue/free_queue_size": stats.free_queue_size,
    }


def raw_mjai_scalar_snapshot(
    raw_stream: RawMjaiDirectStream | None,
    raw_pinned: RawMjaiPinnedStream | None,
    offsets: RawMjaiResumeOffsets,
) -> dict[str, object]:
    if raw_stream is None and raw_pinned is None:
        return {}
    if raw_stream is not None:
        progress = raw_stream.progress()
        bridge_stats = None
        queue_stats = None
    else:
        assert raw_pinned is not None
        progress = raw_pinned.progress()
        bridge_stats = raw_pinned.bridge_stats()
        queue_stats = raw_pinned.queue_stats()
    return _progress_scalars(progress, offsets) | _bridge_scalars(bridge_stats) | _queue_scalars(queue_stats)


def log_final_scalars(writer: ScalarEventWriter, result: dict[str, object], global_step: int) -> None:
    summary = result.get("summary")
    if isinstance(summary, dict):
        add_scalars(writer, "summary", summary, global_step)
    memory = result.get("memory")
    if isinstance(memory, dict):
        add_scalars(writer, "memory", memory, global_step)
    add_scalars(
        writer,
        "run",
        {
            "compile_s": result.get("compile_s"),
            "global_step": result.get("global_step"),
            "samples_seen": result.get("samples_seen"),
            "raw_mjai_training": result.get("raw_mjai_training"),
            "raw_mjai_pinned_pyo3": result.get("raw_mjai_pinned_pyo3"),
            "compile_dry_run": result.get("compile_dry_run"),
            "warmup_steps_counted": result.get("warmup_steps_counted"),
            "samples_consumed_pre_main": result.get("samples_consumed_pre_main"),
            "pre_main_batches_changed_weights": result.get("pre_main_batches_changed_weights"),
        },
        global_step,
    )
    raw_progress = result.get("raw_mjai_progress")
    if isinstance(raw_progress, dict):
        add_scalars(writer, "raw_mjai", {f"progress/stream_local_{k}": v for k, v in raw_progress.items()}, global_step)
    raw_total_progress = result.get("raw_mjai_resume_total_progress")
    if isinstance(raw_total_progress, dict):
        add_scalars(
            writer,
            "raw_mjai",
            {f"progress/resume_total_{k}": v for k, v in raw_total_progress.items()},
            global_step,
        )
    bridge_stats = result.get("raw_mjai_bridge_stats")
    if isinstance(bridge_stats, dict):
        add_scalars(writer, "raw_mjai", {f"bridge/{k}": v for k, v in bridge_stats.items()}, global_step)
    queue_stats = result.get("raw_mjai_queue_stats")
    if isinstance(queue_stats, dict):
        add_scalars(writer, "raw_mjai", {f"queue/{k}": v for k, v in queue_stats.items()}, global_step)


class JsonlLogger:
    def __init__(self, path: Path | None) -> None:
        self._file = None if path is None else path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def write(self, event: str, payload: dict[str, object] | None = None) -> None:
        if self._file is None:
            return
        record: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
        }
        if payload is not None:
            record.update(payload)
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()


def atomic_save_training_checkpoint(
    path: Path,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    manifest_path: Path | None,
    global_step: int,
    samples_seen: int,
    raw_mjai_progress: dict[str, int] | None = None,
    ema_tracker: EmaTracker | None = None,
    ema_config: EmaConfig | None = None,
    weight_source: Literal["raw", "ema"] = "raw",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=optimizer_config,
        runtime_config=runtime_config,
        loss_weights=loss_weights,
        manifest_path=manifest_path,
        global_step=global_step,
        samples_seen=samples_seen,
        raw_mjai_progress=raw_mjai_progress,
        ema_tracker=ema_tracker,
        ema_config=ema_config,
        weight_source=weight_source,
    )
    tmp_path.replace(path)


def checkpoint_paths(args: argparse.Namespace, global_step: int) -> tuple[Path | None, Path | None]:
    if args.checkpoint_dir is None:
        return args.checkpoint_out, None
    latest = args.checkpoint_dir / "latest.pt"
    step_path = args.checkpoint_dir / f"step_{global_step}.pt" if args.keep_step_checkpoints else None
    return latest, step_path


def best_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.checkpoint_dir is None:
        return None
    return args.checkpoint_dir / "best.pt"


def checkpoint_raw_progress(
    raw_stream: RawMjaiDirectStream | None,
    raw_pinned: RawMjaiPinnedStream | None,
    offsets: RawMjaiResumeOffsets,
) -> dict[str, int] | None:
    progress = (
        raw_stream.progress() if raw_stream is not None else raw_pinned.progress() if raw_pinned is not None else None
    )
    return raw_mjai_progress_dict(apply_progress_offsets(progress, offsets))


def torch_env() -> dict[str, object]:
    env: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "optimizer": "AdamW",
    }
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name()
        env["device_capability"] = torch.cuda.get_device_capability()
    return env


def json_manifest_summary(summary: ManifestSummary | None) -> dict[str, object] | None:
    if summary is None:
        return None
    data = asdict(summary)
    data["path"] = str(data["path"])
    return data


def json_raw_mjai_progress(progress: object | None) -> dict[str, object] | None:
    if progress is None:
        return None
    return build_progress_json(cast("BuildProgress", progress))


def save_training_checkpoint(
    path: Path,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    manifest_path: Path | None,
    global_step: int,
    samples_seen: int,
    raw_mjai_progress: dict[str, int] | None = None,
    ema_tracker: EmaTracker | None = None,
    ema_config: EmaConfig | None = None,
    weight_source: Literal["raw", "ema"] = "raw",
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=optimizer_config,
        runtime_config=runtime_config,
        loss_weights=loss_weights,
        manifest_path=manifest_path,
        global_step=global_step,
        samples_seen=samples_seen,
        raw_mjai_progress=raw_mjai_progress,
        ema_config=ema_config,
        ema_state=None if ema_tracker is None else ema_tracker.state,
        ema_update_count=0 if ema_tracker is None else ema_tracker.update_count,
        ema_last_update_step=0 if ema_tracker is None else ema_tracker.last_update_step,
        weight_source=weight_source,
    )


def json_config(args: argparse.Namespace, effective_raw_mjai_max_samples: int | None = None) -> dict[str, object]:
    return {
        "variant": args.variant,
        "loss_mode": args.loss_mode,
        "batch": args.batch,
        "microbatch": args.microbatch,
        "hidden": args.hidden,
        "blocks": args.blocks,
        "bottleneck": args.bottleneck,
        "actions": args.actions,
        "residual_profile": args.residual_profile,
        "backbone_profile": args.backbone_profile,
        "conv_memory_format": args.conv_memory_format,
        "warmup": args.warmup,
        "steps": args.steps,
        "profile": args.profile,
        "profile_coarse": args.profile_coarse,
        "torch_profiler_trace": str(args.torch_profiler_trace) if args.torch_profiler_trace else None,
        "torch_profiler_start_step": args.torch_profiler_start_step,
        "torch_profiler_stop_step": args.torch_profiler_stop_step,
        "manifest": str(args.manifest) if args.manifest else None,
        "check_shard_files": args.check_shard_files,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "schedule_total_steps": args.schedule_total_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "adamw_foreach": args.adamw_foreach,
        "adamw_fused": args.adamw_fused,
        "out": str(args.out),
        "w_oracle_critic": args.w_oracle_critic,
        "w_safety_residual": args.w_safety_residual,
        "compile_fullgraph_check": args.compile_fullgraph_check,
        "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "keep_step_checkpoints": args.keep_step_checkpoints,
        "resume": str(args.resume) if args.resume else None,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "log_dir": str(args.log_dir) if args.log_dir else None,
        "log_every_steps": args.log_every_steps,
        "tensorboard_dir": str(args.tensorboard_dir) if args.tensorboard_dir else None,
        "tensorboard_url": args.tensorboard_url,
        "validation_steps": args.validation_steps,
        "validation_max_samples": args.validation_max_samples,
        "validation_every": args.validation_every,
        "validation_source_mode": args.validation_source_mode,
        "ema": None
        if not args.ema_enabled
        else {
            "enabled": True,
            "decay": args.ema_decay,
            "start_step": args.ema_start_step,
            "update_every_steps": args.ema_update_every_steps,
            "device": args.ema_device,
        },
        "raw_mjai": raw_mjai_config_from_args(args),
        "effective_raw_mjai_max_samples": effective_raw_mjai_max_samples,
        "quiet": args.quiet,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=PYTHON_VARIANT_DEFAULT)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="full_base")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--bottleneck", type=int, default=DEFAULT_SE_BOTTLENECK)
    parser.add_argument("--actions", type=int, default=ACTION_SPACE)
    parser.add_argument("--residual-profile", choices=RESIDUAL_PROFILES, default=RESIDUAL_PROFILE_DEFAULT)
    parser.add_argument("--backbone-profile", choices=BACKBONE_PROFILES, default=BACKBONE_PROFILE_DEFAULT)
    parser.add_argument("--conv-memory-format", choices=CONV_MEMORY_FORMATS, default=CONV_MEMORY_FORMAT_DEFAULT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--profile", action="store_true", help="emit NVTX ranges around measured steps")
    parser.add_argument(
        "--profile-coarse", action="store_true", help="measure whole-step time only to reduce profiler overhead"
    )
    parser.add_argument(
        "--torch-profiler-trace",
        type=Path,
        help="write a Chrome trace for a scheduled measured-step window",
    )
    parser.add_argument(
        "--torch-profiler-start-step",
        type=int,
        default=0,
        help="0-based measured step where torch profiler capture starts",
    )
    parser.add_argument(
        "--torch-profiler-stop-step",
        type=int,
        default=1,
        help="0-based measured step where torch profiler capture stops, exclusive",
    )
    parser.add_argument("--manifest", type=Path, help="train from a Hydra BC shard manifest instead of synthetic data")
    add_raw_mjai_args(parser)
    parser.add_argument(
        "--check-shard-files", action="store_true", help="also validate shard headers named by --manifest"
    )
    parser.add_argument("--w-oracle-critic", type=float, default=0.0)
    parser.add_argument("--w-safety-residual", type=float, default=0.0)
    parser.add_argument("--compile-fullgraph-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/home/cachybtw/tmp/hydra_py_bc_result.json"))
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="cosine")
    parser.add_argument("--schedule-total-steps", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--adamw-foreach", choices=ADAMW_FLAG_MODES, default="auto")
    parser.add_argument("--adamw-fused", choices=ADAMW_FLAG_MODES, default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--tensorboard-url")
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--validation-max-samples", type=int)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-source-mode", choices=VALIDATION_SOURCE_MODES, default="fixed")
    parser.add_argument("--ema-enabled", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--ema-update-every-steps", type=int, default=1)
    parser.add_argument("--ema-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--cuda-profiler-range", action="store_true", help="start CUDA profiler only around measured steps"
    )
    parser.add_argument("--quiet", action="store_true", help="write JSON result without printing it to stdout")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (("batch", args.batch), ("microbatch", args.microbatch))
    for name, value in positive_ints:
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be >= 1")
    non_negative_ints = (
        ("warmup", args.warmup),
        ("checkpoint_every_steps", args.checkpoint_every_steps),
        ("log_every_steps", args.log_every_steps),
        ("validation_steps", args.validation_steps),
        ("validation_every", args.validation_every),
        ("validation_max_samples", args.validation_max_samples if args.validation_max_samples is not None else 0),
        ("lr_warmup_steps", args.lr_warmup_steps),
        ("ema_start_step", args.ema_start_step),
    )
    for name, value in non_negative_ints:
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.schedule_total_steps is not None and args.schedule_total_steps < 1:
        raise ValueError("--schedule-total-steps must be >= 1")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0")
    if args.min_lr < 0.0:
        raise ValueError("--min-lr must be >= 0")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must be <= --lr")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError("--grad-clip-norm must be > 0")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.ema_update_every_steps < 1:
        raise ValueError("--ema-update-every-steps must be >= 1")
    if (
        args.lr_schedule == "cosine"
        and args.schedule_total_steps is None
        and args.steps is None
        and args.manifest is None
    ):
        raise ValueError("--lr-schedule cosine requires --schedule-total-steps, --steps, or --manifest")
    if args.checkpoint_out is not None and args.checkpoint_dir is not None:
        raise ValueError("--checkpoint-out and --checkpoint-dir cannot be combined")
    if args.keep_step_checkpoints and args.checkpoint_dir is None:
        raise ValueError("--keep-step-checkpoints requires --checkpoint-dir")
    if args.microbatch > args.batch:
        raise ValueError("--microbatch must be <= --batch")
    if args.actions != ACTION_SPACE:
        raise ValueError(f"--actions must equal Hydra action space {ACTION_SPACE}")
    if args.torch_profiler_start_step < 0:
        raise ValueError("--torch-profiler-start-step must be >= 0")
    if args.torch_profiler_stop_step < 1:
        raise ValueError("--torch-profiler-stop-step must be >= 1")
    if args.torch_profiler_trace is not None:
        if args.torch_profiler_start_step >= args.torch_profiler_stop_step:
            raise ValueError("--torch-profiler-start-step must be < --torch-profiler-stop-step")
        if args.torch_profiler_stop_step > args.steps:
            raise ValueError("--torch-profiler-stop-step must be <= --steps")
    if args.full_epoch and args.manifest is not None:
        raise ValueError("--full-epoch is only supported with raw MJAI input")
    if args.full_epoch and args.raw_mjai_max_samples is not None:
        raise ValueError("--full-epoch cannot be combined with --raw-mjai-max-samples")
    if args.resume is not None and args.raw_mjai_data_dirs and not raw_mjai_cursor_resume_supported():
        raise ValueError(RAW_MJAI_CURSOR_RESUME_ERROR)


def main() -> int:
    args = parse_args()
    validate_args(args)
    validate_raw_mjai_source_args(args)
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard_dir is not None:
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    events = JsonlLogger(None if args.log_dir is None else args.log_dir / "events.jsonl")
    train_log = JsonlLogger(None if args.log_dir is None else args.log_dir / "train_steps.jsonl")
    scalar_writer = ScalarEventWriter(args.tensorboard_dir)
    events.write("run_start", {"config": json_config(args)})
    manifest_summary = None
    real_dataset = None
    raw_stream = None
    raw_pinned = None
    staged_initial_batch: StagedTrainBatch | None = None
    raw_train_max_samples = args.raw_mjai_max_samples
    if args.raw_mjai_data_dirs and raw_train_max_samples is None and not args.full_epoch and args.steps is not None:
        raw_train_batches = args.steps + 1
        raw_train_max_samples = raw_train_batches * args.batch

    events.write(
        "input_setup_start",
        {
            "manifest": str(args.manifest) if args.manifest is not None else None,
            "raw_mjai_transport": args.raw_mjai_transport if args.raw_mjai_data_dirs else None,
            "raw_mjai_data_dir_count": len(args.raw_mjai_data_dirs or ()),
            "raw_mjai_max_games": args.raw_mjai_max_games,
            "raw_mjai_max_samples": raw_train_max_samples,
            "raw_mjai_prefetch_batches": args.raw_mjai_prefetch_batches,
            "raw_mjai_queue_bound": args.raw_mjai_queue_bound,
            "raw_mjai_worker_threads": args.raw_mjai_worker_threads,
        },
    )
    if args.manifest is not None:
        manifest_summary = validate_manifest(args.manifest, check_files=args.check_shard_files)
        real_dataset = BcShardDataset(args.manifest, batch_size=args.batch, split="train")
        events.write("input_setup_complete", {"kind": "bc_shards"})
    elif args.raw_mjai_data_dirs:
        if args.raw_mjai_transport == RAW_MJAI_TRANSPORT_STDOUT:
            raw_stream = RawMjaiDirectStream(
                data_dirs=args.raw_mjai_data_dirs,
                batch_size=args.batch,
                prefetch_batches=args.raw_mjai_prefetch_batches,
                queue_bound=args.raw_mjai_queue_bound,
                worker_threads=args.raw_mjai_worker_threads,
                max_games=args.raw_mjai_max_games,
                max_samples=raw_train_max_samples,
                train_fraction=args.raw_mjai_train_fraction,
                augment=args.raw_mjai_augment,
                split=args.raw_mjai_split,
            )
            raw_stream.start()
            events.write("input_setup_complete", {"kind": "raw_mjai_stdout"})
        elif args.raw_mjai_transport == RAW_MJAI_TRANSPORT_PINNED_PYO3:
            raw_pinned = RawMjaiPinnedStream(
                data_dirs=args.raw_mjai_data_dirs,
                batch_size=args.batch,
                queue_bound=args.raw_mjai_queue_bound,
                worker_threads=args.raw_mjai_worker_threads,
                max_games=args.raw_mjai_max_games,
                max_samples=raw_train_max_samples,
                train_fraction=args.raw_mjai_train_fraction,
                augment=args.raw_mjai_augment,
                split=args.raw_mjai_split,
                library_path=args.raw_mjai_pinned_ffi or default_raw_mjai_pinned_library_path(),
                ring_size=args.raw_mjai_prefetch_batches,
            )
            events.write("input_setup_complete", {"kind": "raw_mjai_pinned_pyo3"})
        else:
            raise ValueError(f"unsupported raw MJAI transport {args.raw_mjai_transport!r}")
    validation_source = RawMjaiValidationSource(args=args, events=events)
    events.write("validation_setup_complete", {"actual_batches": validation_source.info.actual_batches})

    if args.schedule_total_steps is None:
        if args.steps is not None:
            args.schedule_total_steps = args.steps
        elif manifest_summary is not None:
            args.schedule_total_steps = max(1, math.ceil(manifest_summary.train_samples / args.batch))
    if args.lr_schedule == "cosine" and args.schedule_total_steps is None:
        raise ValueError("--lr-schedule cosine requires --schedule-total-steps when run horizon is unbounded")
    events.write("torch_setup_start", {})

    if not torch.cuda.is_available():
        result = {"variant": args.variant, "env": torch_env(), "error": "CUDA unavailable"}
        events.write("run_error", result)
        events.close()
        train_log.close()
        scalar_writer.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if not args.quiet:
            print(json.dumps(result, indent=2))
        return 2

    torch.manual_seed(0x51A7E)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    model = HydraPolicyNet(
        args.hidden,
        args.blocks,
        args.bottleneck,
        args.actions,
        args.residual_profile,
        args.backbone_profile,
        args.conv_memory_format,
    ).to(device)
    events.write(
        "model_setup_complete",
        {
            "hidden": args.hidden,
            "blocks": args.blocks,
            "bottleneck": args.bottleneck,
            "residual_profile": args.residual_profile,
            "conv_memory_format": args.conv_memory_format,
        },
    )
    optimizer_config = build_optimizer_config(args)
    optimizer = build_optimizer(model, optimizer_config)
    lr_scheduler = LrScheduler(build_lr_scheduler_config(args))
    if real_dataset is None and raw_stream is None and raw_pinned is None:
        obs, legal, labels = make_synthetic_batch(args.batch, args.actions, device)
        targets = synthetic_targets(obs, legal, labels)
        input_timing = InputTiming()
    else:
        obs = legal = labels = None
        targets = None
        input_timing = InputTiming()
    autocast = args.variant != "eager_fp32"
    weights = LossWeights(oracle_critic=args.w_oracle_critic, safety_residual=args.w_safety_residual)
    if args.loss_mode not in COMPILED_LOSS_MODES:
        raise ValueError(f"unsupported loss mode {args.loss_mode!r}")
    loss_step: nn.Module = HydraCompiledLossStep(model, args.loss_mode, weights)
    model_config = ModelConfig(
        hidden=args.hidden,
        blocks=args.blocks,
        bottleneck=args.bottleneck,
        actions=args.actions,
        residual_profile=args.residual_profile,
        backbone_profile=args.backbone_profile,
        conv_memory_format=args.conv_memory_format,
    )
    precision_mode = "fp32" if args.variant == "eager_fp32" else "bf16_autocast"
    runtime_config = RuntimeConfig(
        variant=args.variant,
        loss_mode=args.loss_mode,
        precision_mode=precision_mode,
        compile_fullgraph_check=args.compile_fullgraph_check,
        compile_dry_run_mode=COMPILE_DRY_RUN_MODE,
        warmup_mode=WARMUP_MODE,
    )
    ema_config = build_ema_config(args)
    ema_tracker = None if ema_config is None else EmaTracker(model, ema_config, device)
    resume_state = None
    if args.resume is not None:
        resume_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_optimizer_config=optimizer_config,
            expected_runtime_config=runtime_config,
            expected_loss_weights=weights,
            expected_manifest_path=args.manifest if raw_stream is None else raw_stream.manifest_path,
            expected_ema_config=ema_config,
        )
        if ema_tracker is not None:
            if resume_state.ema is None:
                raise ValueError("checkpoint EMA state missing after EMA resume validation")
            ema_tracker.load_state(
                resume_state.ema.state_dict,
                resume_state.ema.update_count,
                resume_state.ema.last_update_step,
            )
        events.write(
            "resume_loaded",
            {
                "checkpoint_path": str(args.resume),
                "global_step": resume_state.global_step,
                "samples_seen": resume_state.samples_seen,
            },
        )
        scalar_writer.add_scalar("checkpoint/resumed", 1.0, resume_state.global_step)
        scalar_writer.add_scalar("checkpoint/resumed_step", float(resume_state.global_step), resume_state.global_step)
        scalar_writer.add_scalar(
            "checkpoint/resumed_samples_seen",
            float(resume_state.samples_seen),
            resume_state.global_step,
        )
        scalar_writer.add_scalar("run/resume_global_step", float(resume_state.global_step), resume_state.global_step)

    global_step = 0 if resume_state is None else resume_state.global_step
    samples_seen = 0 if resume_state is None else resume_state.samples_seen
    raw_mjai_offsets = RawMjaiResumeOffsets.from_resume(resume_state, args.batch)

    compile_error = None
    compile_s = 0.0
    events.write("first_batch_fetch_start", {})
    if real_dataset is not None or raw_stream is not None or raw_pinned is not None:
        staged_initial_batch = next_staged_train_batch(
            real_dataset=real_dataset,
            raw_stream=raw_stream,
            raw_pinned=raw_pinned,
            device=device,
            weights=weights,
        )
        obs = staged_initial_batch.obs
        targets = staged_initial_batch.targets
        input_timing = staged_initial_batch.input_timing
        events.write(
            "first_batch_fetch_complete",
            {
                "fetch_decode_ms": input_timing.fetch_decode_ms,
                "h2d_wall_ms": input_timing.h2d_wall_ms,
                "rows": int(obs.shape[0]),
            },
        )
    else:
        assert obs is not None and targets is not None
        targets = targets_for_compiled_loss(targets, weights)
        events.write("first_batch_fetch_complete", {"kind": "synthetic", "rows": int(obs.shape[0])})
    pre_main = PreMainAccounting(
        compile_dry_run=args.variant.startswith("compile_"),
        warmup_steps_run=args.warmup,
        samples_consumed_pre_main=(obs.shape[0] if staged_initial_batch is not None else 0),
        batches_consumed_pre_main=(1 if staged_initial_batch is not None else 0),
    )
    staged_initial_pinned_marked = False
    if args.variant.startswith("compile_"):
        events.write(
            "compile_start",
            {"variant": args.variant, "compile_dry_run": True, "compile_dry_run_mode": COMPILE_DRY_RUN_MODE},
        )
        mode = None
        if args.variant == "compile_reduce_overhead":
            mode = "reduce-overhead"
        elif args.variant == "compile_max_autotune":
            mode = "max-autotune"
        t0 = time.perf_counter()
        try:
            loss_step = cast("nn.Module", torch.compile(loss_step, mode=mode, fullgraph=args.compile_fullgraph_check))
            setattr(loss_step, "_hydra_compiled", True)
            set_optimizer_lr(optimizer, lr_scheduler.lr_for_step(global_step))
            run_non_mutating_train_step(
                loss_step,
                model,
                optimizer,
                obs,
                targets,
                weights,
                args.loss_mode,
                args.microbatch,
                autocast,
                False,
                args.grad_clip_norm,
            )
            torch.cuda.synchronize()
            if raw_pinned is not None and staged_initial_batch is not None:
                assert staged_initial_batch.pinned_batch is not None
                raw_pinned.mark_inflight(staged_initial_batch.pinned_batch)
                staged_initial_pinned_marked = True
            compile_s = time.perf_counter() - t0
            events.write("compile_complete", {"compile_s": compile_s} | pre_main.as_log_fields())
            scalar_writer.add_scalar("runtime/compile_s", compile_s, global_step)
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            error_compile_s = time.perf_counter() - t0
            events.write("compile_error", {"error": compile_error, "compile_s": error_compile_s})
            scalar_writer.add_scalar("runtime/compile_error", 1.0, global_step)

    env = torch_env()
    if compile_error is not None:
        result = {
            "variant": args.variant,
            "env": env,
            "compile_error": compile_error,
            "compile_s": compile_s,
            "fullgraph_check": args.compile_fullgraph_check,
        }
        events.write("run_error", result)
        events.close()
        train_log.close()
        scalar_writer.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if not args.quiet:
            print(json.dumps(result, indent=2))
        return 2

    torch.cuda.reset_peak_memory_stats()
    eval_stats: list[dict[str, object]] = []
    events.write("warmup_start", {"warmup_steps": args.warmup})
    if args.warmup > 0:
        set_optimizer_lr(optimizer, lr_scheduler.lr_for_step(global_step))
    for _ in range(args.warmup):
        run_non_mutating_train_step(
            loss_step,
            model,
            optimizer,
            obs,
            targets,
            weights,
            args.loss_mode,
            args.microbatch,
            autocast,
            False,
            args.grad_clip_norm,
        )
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        cuda_profiler.start()

    profiler_ctx: AbstractContextManager[torch.profiler.profile | None]
    if args.torch_profiler_trace is None:
        profiler_ctx = nullcontext(None)
    else:
        args.torch_profiler_trace.parent.mkdir(parents=True, exist_ok=True)
        active_steps = args.torch_profiler_stop_step - args.torch_profiler_start_step

        def write_trace(prof: torch.profiler.profile) -> None:
            prof.export_chrome_trace(str(args.torch_profiler_trace))

        profiler_ctx = torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            schedule=torch.profiler.schedule(
                wait=args.torch_profiler_start_step,
                warmup=0,
                active=active_steps,
                repeat=1,
            ),
            on_trace_ready=write_trace,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )

    events.write("warmup_complete", {"warmup_steps": args.warmup} | pre_main.as_log_fields())
    scalar_writer.add_scalar("runtime/warmup_steps", float(args.warmup), global_step)
    scalar_writer.add_scalar("runtime/warmup_steps_counted", 0.0, global_step)
    scalar_writer.add_scalar(
        "runtime/samples_consumed_pre_main", float(pre_main.samples_consumed_pre_main), global_step
    )
    stats: list[StepStats] = []
    best_policy_nll = math.inf
    best_policy_nll_step = 0
    step_range = itertools.count() if args.steps is None else range(args.steps)
    with profiler_ctx as torch_profiler:
        for i in step_range:
            if args.full_epoch and i > 0 and raw_pinned is not None and raw_pinned.progress().complete:
                break
            if args.full_epoch and i > 0 and raw_stream is not None and raw_stream.progress().complete:
                break
            if args.profile:
                torch.cuda.nvtx.range_push(f"hydra_bc_step_{i}")
            pinned_batch: PinnedPolicyBatch | None = None
            pinned_batch_already_marked = False
            if staged_initial_batch is not None:
                staged = staged_initial_batch
                staged_initial_batch = None
                obs = staged.obs
                targets = staged.targets
                input_timing = staged.input_timing
                pinned_batch = staged.pinned_batch
                pinned_batch_already_marked = staged_initial_pinned_marked
            else:
                input_timing = InputTiming()
                if raw_stream is not None:
                    try:
                        staged = next_staged_train_batch(
                            real_dataset=real_dataset,
                            raw_stream=raw_stream,
                            raw_pinned=raw_pinned,
                            device=device,
                            weights=weights,
                        )
                    except StopIteration:
                        if args.full_epoch:
                            break
                        raise
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
                elif raw_pinned is not None:
                    try:
                        staged = next_staged_train_batch(
                            real_dataset=real_dataset,
                            raw_stream=raw_stream,
                            raw_pinned=raw_pinned,
                            device=device,
                            weights=weights,
                        )
                    except StopIteration:
                        if args.full_epoch:
                            break
                        raise
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
                    pinned_batch = staged.pinned_batch
                elif real_dataset is not None:
                    staged = next_staged_train_batch(
                        real_dataset=real_dataset,
                        raw_stream=raw_stream,
                        raw_pinned=raw_pinned,
                        device=device,
                        weights=weights,
                    )
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
            lr = lr_scheduler.lr_for_step(global_step)
            set_optimizer_lr(optimizer, lr)
            will_log = args.log_every_steps > 0 and (
                (global_step + 1) % args.log_every_steps == 0 or (args.steps is not None and i + 1 == args.steps)
            )
            stat = run_step(
                loss_step,
                model,
                optimizer,
                obs,
                targets,
                weights,
                args.loss_mode,
                args.microbatch,
                autocast,
                timed=not args.profile_coarse,
                grad_clip_norm=args.grad_clip_norm,
                collect_diagnostics=will_log,
            )
            stat.lr = lr
            if real_dataset is not None or raw_stream is not None or raw_pinned is not None:
                stat.fetch_decode_ms = input_timing.fetch_decode_ms
                stat.h2d_wall_ms = input_timing.h2d_wall_ms
            stats.append(stat)
            global_step += 1
            samples_seen += obs.shape[0]
            if ema_tracker is not None:
                ema_tracker.maybe_update(model, global_step)
            ema_metrics = (
                {
                    "ema/enabled": False,
                    "ema/device": "off",
                    "ema/device_code": -1,
                    "ema/update_count": 0,
                    "ema/active": False,
                    "ema/last_update_step": 0,
                    "ema/last_update_ms": math.nan,
                }
                if ema_tracker is None
                else ema_tracker.metrics()
            )
            if will_log:
                train_log.write(
                    "train_step",
                    {
                        "step": i + 1,
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                        "batch": args.batch,
                        "loss": stat.loss,
                        "head_losses": stat.head_losses,
                        "target_coverage": stat.target_coverage,
                        "step_ms": stat.step_ms,
                        "fwd_loss_ms": stat.fwd_loss_ms,
                        "backward_ms": stat.backward_ms,
                        "optimizer_ms": stat.optimizer_ms,
                        "train_gpu_ms": stat.train_gpu_ms,
                        "fetch_decode_ms": stat.fetch_decode_ms,
                        "h2d_wall_ms": stat.h2d_wall_ms,
                        "lr": stat.lr,
                        "grad_norm": stat.grad_norm,
                        "compile_dry_run": pre_main.compile_dry_run,
                        "warmup_mode": pre_main.warmup_mode,
                        "warmup_steps_counted": pre_main.warmup_steps_counted,
                        "samples_consumed_pre_main": pre_main.samples_consumed_pre_main,
                        "pre_main_batches_changed_weights": pre_main.pre_main_batches_changed_weights,
                    }
                    | ema_metrics
                    | raw_mjai_scalar_snapshot(raw_stream, raw_pinned, raw_mjai_offsets),
                )
                log_step_scalars(
                    scalar_writer,
                    stat,
                    batch=args.batch,
                    samples_seen=samples_seen,
                    global_step=global_step,
                )
                add_scalars(
                    scalar_writer,
                    "raw_mjai",
                    raw_mjai_scalar_snapshot(raw_stream, raw_pinned, raw_mjai_offsets),
                    global_step,
                )
                add_scalars(
                    scalar_writer,
                    "ema",
                    {key.removeprefix("ema/"): value for key, value in ema_metrics.items()},
                    global_step,
                )
                scalar_writer.flush()
            if raw_pinned is not None and not pinned_batch_already_marked:
                assert pinned_batch is not None
                raw_pinned.mark_inflight(pinned_batch)
            if (
                validation_source.info.actual_batches > 0
                and args.validation_every > 0
                and global_step % args.validation_every == 0
            ):
                metrics, ema_metrics = evaluate_raw_and_ema(
                    validation_source,
                    steps=args.validation_steps,
                    model=model,
                    device=device,
                    weights=weights,
                    autocast=autocast,
                    ema_tracker=ema_tracker,
                )
                event_metrics = {"raw": metrics}
                scalar_metrics = prefixed_metrics("raw", metrics)
                if ema_metrics is not None:
                    event_metrics["ema"] = ema_metrics
                    scalar_metrics |= prefixed_metrics("ema", ema_metrics)
                eval_stats.append({"step": i + 1, "metrics": event_metrics})
                events.write(
                    "validation",
                    {
                        "step": i + 1,
                        "global_step": global_step,
                        "source": asdict(validation_source.info),
                        "metrics": event_metrics,
                    },
                )
                log_validation_scalars(scalar_writer, scalar_metrics, global_step, final=False)
                scalar_writer.flush()
                raw_policy_nll_value = metrics["policy_nll"]
                if not isinstance(raw_policy_nll_value, Real):
                    raise TypeError("validation policy_nll metric must be numeric")
                best_metrics = metrics
                best_weight_source: Literal["raw", "ema"] = "raw"
                if ema_metrics is not None:
                    ema_policy_nll_value = ema_metrics["policy_nll"]
                    if not isinstance(ema_policy_nll_value, Real):
                        raise TypeError("EMA validation policy_nll metric must be numeric")
                    if float(ema_policy_nll_value) < float(raw_policy_nll_value):
                        best_metrics = ema_metrics
                        best_weight_source = "ema"
                best_policy_nll_value = best_metrics["policy_nll"]
                if not isinstance(best_policy_nll_value, Real):
                    raise TypeError("best validation policy_nll metric must be numeric")
                policy_nll = float(best_policy_nll_value)
                if policy_nll < best_policy_nll:
                    best_policy_nll = policy_nll
                    best_policy_nll_step = global_step
                    best_path = best_checkpoint_path(args)
                    if best_path is not None:
                        weight_ctx = ema_weights(model, ema_tracker) if best_weight_source == "ema" else nullcontext()
                        with weight_ctx:
                            atomic_save_training_checkpoint(
                                best_path,
                                model=model,
                                optimizer=optimizer,
                                model_config=model_config,
                                optimizer_config=optimizer_config,
                                runtime_config=runtime_config,
                                loss_weights=weights,
                                manifest_path=args.manifest if raw_stream is None else raw_stream.manifest_path,
                                global_step=global_step,
                                samples_seen=samples_seen,
                                raw_mjai_progress=checkpoint_raw_progress(raw_stream, raw_pinned, raw_mjai_offsets),
                                ema_tracker=ema_tracker,
                                ema_config=ema_config,
                                weight_source=best_weight_source,
                            )
                        events.write(
                            "best_checkpoint_saved",
                            {
                                "path": str(best_path),
                                "metric": "policy_nll",
                                "metric_value": best_policy_nll,
                                "global_step": global_step,
                                "samples_seen": samples_seen,
                                "weight_source": best_weight_source,
                            },
                        )
                        scalar_writer.add_scalar("checkpoint/best_policy_nll", best_policy_nll, global_step)
                        scalar_writer.add_scalar("checkpoint/best_step", float(best_policy_nll_step), global_step)
            latest_checkpoint, step_checkpoint = checkpoint_paths(args, global_step)
            if (
                latest_checkpoint is not None
                and args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            ):
                for checkpoint_path in (latest_checkpoint, step_checkpoint):
                    if checkpoint_path is None:
                        continue
                    atomic_save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        runtime_config=runtime_config,
                        loss_weights=weights,
                        manifest_path=args.manifest if raw_stream is None else raw_stream.manifest_path,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        raw_mjai_progress=raw_mjai_progress_dict(
                            apply_progress_offsets(
                                raw_stream.progress()
                                if raw_stream is not None
                                else raw_pinned.progress()
                                if raw_pinned is not None
                                else None,
                                raw_mjai_offsets,
                            )
                        ),
                        ema_tracker=ema_tracker,
                        ema_config=ema_config,
                    )
                events.write(
                    "checkpoint_saved",
                    {
                        "path": str(latest_checkpoint),
                        "step_path": None if step_checkpoint is None else str(step_checkpoint),
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                    },
                )
                scalar_writer.add_scalar("checkpoint/saved", 1.0, global_step)
                scalar_writer.add_scalar("checkpoint/samples_seen", float(samples_seen), global_step)
            if torch_profiler is not None:
                torch_profiler.step()
            if args.profile:
                torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        cuda_profiler.stop()

    final_validation = None
    if validation_source.info.actual_batches > 0:
        raw_final_validation, ema_final_validation = evaluate_raw_and_ema(
            validation_source,
            steps=args.validation_steps,
            model=model,
            device=device,
            weights=weights,
            autocast=autocast,
            ema_tracker=ema_tracker,
        )
        final_validation = {"raw": raw_final_validation}
        final_scalar_metrics = prefixed_metrics("raw", raw_final_validation)
        if ema_final_validation is not None:
            final_validation["ema"] = ema_final_validation
            final_scalar_metrics |= prefixed_metrics("ema", ema_final_validation)
        log_validation_scalars(scalar_writer, final_scalar_metrics, global_step, final=True)
    raw_progress: BuildProgress | None = None
    if raw_stream is not None:
        raw_progress = raw_stream.progress()
    elif raw_pinned is not None:
        raw_progress = raw_pinned.progress()
    raw_stream_local_progress, raw_resume_total_progress = raw_mjai_progress_sections(raw_progress, raw_mjai_offsets)

    final_global_step = global_step
    final_samples_seen = samples_seen
    result = {
        "variant": args.variant,
        "env": env,
        "config": json_config(args, raw_train_max_samples),
        "manifest_summary": json_manifest_summary(
            raw_stream.manifest_summary
            if raw_stream is not None
            else (raw_pinned.manifest_summary if raw_pinned is not None else manifest_summary)
        ),
        "compile_s": compile_s,
        "compile_dry_run": pre_main.compile_dry_run,
        "warmup_mode": pre_main.warmup_mode,
        "warmup_steps_counted": pre_main.warmup_steps_counted,
        "samples_consumed_pre_main": pre_main.samples_consumed_pre_main,
        "pre_main_batches_changed_weights": pre_main.pre_main_batches_changed_weights,
        "summary": summarize_steps(stats, args.batch),
        "validation": {
            "every": args.validation_every,
            "source": asdict(validation_source.info),
            "history": eval_stats,
            "final": final_validation,
        },
        "memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "step_stats": [asdict(s) for s in stats],
        "real_shard_training": real_dataset is not None,
        "raw_mjai_training": raw_stream is not None or raw_pinned is not None,
        "raw_mjai_pinned_pyo3": raw_pinned is not None,
        "raw_mjai_transport": args.raw_mjai_transport,
        "raw_mjai_progress": raw_stream_local_progress,
        "raw_mjai_resume_total_progress": raw_resume_total_progress,
        "raw_mjai_bridge_stats": None if raw_pinned is None else asdict(raw_pinned.bridge_stats()),
        "raw_mjai_queue_stats": None if raw_pinned is None else asdict(raw_pinned.queue_stats()),
        "checkpoint_path": None,
        "resumed_step": None if resume_state is None else resume_state.global_step,
        "resumed_samples_seen": None if resume_state is None else resume_state.samples_seen,
        "global_step": final_global_step,
        "samples_seen": final_samples_seen,
        "ema": {
            "enabled": ema_tracker is not None,
            "device": None if ema_tracker is None else ema_tracker.device_name,
            "update_count": 0 if ema_tracker is None else ema_tracker.update_count,
            "last_update_step": 0 if ema_tracker is None else ema_tracker.last_update_step,
            "last_update_ms": math.nan if ema_tracker is None else ema_tracker.last_update_ms,
        },
    }
    global_step = final_global_step
    samples_seen = final_samples_seen
    latest_checkpoint, step_checkpoint = checkpoint_paths(args, global_step)
    if latest_checkpoint is not None:
        for checkpoint_path in (latest_checkpoint, step_checkpoint):
            if checkpoint_path is None:
                continue
            atomic_save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                optimizer_config=optimizer_config,
                runtime_config=runtime_config,
                loss_weights=weights,
                manifest_path=args.manifest if raw_stream is None else raw_stream.manifest_path,
                global_step=global_step,
                samples_seen=samples_seen,
                raw_mjai_progress=raw_mjai_progress_dict(apply_progress_offsets(raw_progress, raw_mjai_offsets)),
                ema_tracker=ema_tracker,
                ema_config=ema_config,
            )
        result["checkpoint_path"] = str(latest_checkpoint)
        events.write(
            "checkpoint_saved",
            {
                "path": str(latest_checkpoint),
                "step_path": None if step_checkpoint is None else str(step_checkpoint),
                "global_step": global_step,
                "samples_seen": samples_seen,
            },
        )
        scalar_writer.add_scalar("checkpoint/saved", 1.0, global_step)
        scalar_writer.add_scalar("checkpoint/samples_seen", float(samples_seen), global_step)
    validation_source.close()
    if raw_stream is not None:
        raw_stream.close()
    if raw_pinned is not None:
        raw_pinned.close()
    log_final_scalars(scalar_writer, result, global_step)
    scalar_writer.flush()
    events.write(
        "run_complete",
        {"global_step": result["global_step"], "samples_seen": result["samples_seen"], "result_path": str(args.out)},
    )
    events.close()
    train_log.close()
    scalar_writer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
