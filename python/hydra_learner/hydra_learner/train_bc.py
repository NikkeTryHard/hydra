#!/usr/bin/env python3
"""Hydra PyTorch base-head BC learner."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast, override

import torch
import torch.cuda.profiler as cuda_profiler
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None
from hydra_learner.checkpoint import (
    ModelConfig,
    OptimizerConfig,
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
    masked_policy_ce_indices,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    soft_ce,
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


def adamw_flag_value(mode: str) -> bool | None:
    if mode == "auto":
        return None
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"unsupported AdamW flag mode {mode!r}")


def build_optimizer_config(args: argparse.Namespace) -> OptimizerConfig:
    return OptimizerConfig(
        name="AdamW",
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
        foreach=adamw_flag_value(args.adamw_foreach),
        fused=adamw_flag_value(args.adamw_fused),
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


def run_step(
    loss_step: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    microbatch: int,
    autocast: bool,
    timed: bool,
) -> StepStats:
    logical = obs.shape[0]
    if getattr(loss_step, "_hydra_compiled", False):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
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

    if timed:
        opt_ms, _ = time_cuda(optimizer.step)
    else:
        optimizer.step()
        opt_ms = 0.0
    step_end.record()
    step_ms = cuda_event_elapsed(step_start, step_end)
    loss_value = float(logical_loss.detach())
    stat = StepStats(step_ms=step_ms, fwd_loss_ms=fwd_ms, backward_ms=bwd_ms, optimizer_ms=opt_ms, loss=loss_value)
    stat.train_gpu_ms = step_ms
    return stat


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
        accuracy = (pred == targets.policy_target.to(dtype=torch.int64)).to(dtype=torch.float32).mean()
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
        policy_accuracy=float(accuracy.detach()),
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
        "step_ms": stat.step_ms,
        "fwd_loss_ms": stat.fwd_loss_ms,
        "backward_ms": stat.backward_ms,
        "optimizer_ms": stat.optimizer_ms,
        "train_gpu_ms": stat.train_gpu_ms,
        "fetch_decode_ms": stat.fetch_decode_ms,
        "h2d_wall_ms": stat.h2d_wall_ms,
        "samples_seen": samples_seen,
        "global_step": global_step,
    }
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
    writer: ScalarEventWriter, metrics: dict[str, float], global_step: int, *, final: bool
) -> None:
    add_scalars(writer, "final_validation" if final else "validation", metrics, global_step)


def _progress_scalars(progress: BuildProgress | None) -> dict[str, object]:
    if progress is None:
        return {}
    return {
        "progress/complete": progress.complete,
        "progress/build_seconds": progress.build_seconds,
        "progress/loaded_games": progress.loaded_games,
        "progress/skipped_games": progress.skipped_games,
        "progress/samples": progress.samples,
        "progress/batches": progress.batches,
        "progress/max_games_reached": progress.max_games_reached,
        "progress/max_samples_reached": progress.max_samples_reached,
    }


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
    raw_stream: RawMjaiDirectStream | None, raw_pinned: RawMjaiPinnedStream | None
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
    return _progress_scalars(progress) | _bridge_scalars(bridge_stats) | _queue_scalars(queue_stats)


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
        },
        global_step,
    )
    raw_progress = result.get("raw_mjai_progress")
    if isinstance(raw_progress, dict):
        add_scalars(writer, "raw_mjai", {f"progress/{k}": v for k, v in raw_progress.items()}, global_step)
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
    )
    tmp_path.replace(path)


def checkpoint_paths(args: argparse.Namespace, global_step: int) -> tuple[Path | None, Path | None]:
    if args.checkpoint_dir is None:
        return args.checkpoint_out, None
    latest = args.checkpoint_dir / "latest.pt"
    step_path = args.checkpoint_dir / f"step_{global_step}.pt" if args.keep_step_checkpoints else None
    return latest, step_path


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
        "validation_every": args.validation_every,
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
    parser.add_argument("--steps", type=int, default=30)
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
    parser.add_argument("--lr", type=float, default=3.0e-4)
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
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument(
        "--cuda-profiler-range", action="store_true", help="start CUDA profiler only around measured steps"
    )
    parser.add_argument("--quiet", action="store_true", help="write JSON result without printing it to stdout")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (("batch", args.batch), ("microbatch", args.microbatch), ("steps", args.steps))
    for name, value in positive_ints:
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    non_negative_ints = (
        ("warmup", args.warmup),
        ("checkpoint_every_steps", args.checkpoint_every_steps),
        ("log_every_steps", args.log_every_steps),
        ("validation_steps", args.validation_steps),
        ("validation_every", args.validation_every),
    )
    for name, value in non_negative_ints:
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
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
    raw_first_batch: PinnedPolicyBatch | None = None
    raw_train_max_samples = args.raw_mjai_max_samples
    if args.raw_mjai_data_dir is not None and raw_train_max_samples is None:
        raw_train_batches = 1 + args.warmup + args.steps
        raw_train_max_samples = raw_train_batches * args.batch
    if args.manifest is not None:
        manifest_summary = validate_manifest(args.manifest, check_files=args.check_shard_files)
        real_dataset = BcShardDataset(args.manifest, batch_size=args.batch, split="train")
    elif args.raw_mjai_data_dir is not None:
        if args.raw_mjai_transport == RAW_MJAI_TRANSPORT_STDOUT:
            raw_stream = RawMjaiDirectStream(
                data_dir=args.raw_mjai_data_dir,
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
        elif args.raw_mjai_transport == RAW_MJAI_TRANSPORT_PINNED_PYO3:
            raw_pinned = RawMjaiPinnedStream(
                data_dir=args.raw_mjai_data_dir,
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
        else:
            raise ValueError(f"unsupported raw MJAI transport {args.raw_mjai_transport!r}")
    validation_stream = None
    if args.validation_steps > 0 and args.raw_mjai_data_dir is not None:
        validation_stream = RawMjaiDirectStream(
            data_dir=args.raw_mjai_data_dir,
            batch_size=args.batch,
            prefetch_batches=args.raw_mjai_prefetch_batches,
            queue_bound=args.raw_mjai_queue_bound,
            worker_threads=args.raw_mjai_worker_threads,
            max_games=args.raw_mjai_max_games,
            max_samples=None,
            train_fraction=args.raw_mjai_train_fraction,
            augment=args.raw_mjai_augment,
            split="validation",
        )
        validation_stream.start()

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
    optimizer_config = build_optimizer_config(args)
    optimizer = build_optimizer(model, optimizer_config)
    if real_dataset is None and raw_stream is None and raw_pinned is None:
        obs, legal, labels = make_synthetic_batch(args.batch, args.actions, device)
        targets = synthetic_targets(obs, legal, labels)
    elif raw_stream is not None:
        first_batch, first_fetch_ms = raw_stream.next_batch()
        obs, legal, labels, targets, _input_timing = tensors_from_policy_batch(first_batch, device, first_fetch_ms)
    elif raw_pinned is not None:
        raw_first_batch, first_fetch_ms = raw_pinned.next_batch()
        obs, legal, labels, targets, _input_timing = tensors_from_pinned_policy_batch(
            raw_first_batch, device, first_fetch_ms
        )
    elif real_dataset is not None:
        obs, legal, labels, targets, _input_timing = tensors_from_real_batch(real_dataset, device)
    else:
        raise ValueError("internal data source selection failed")
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
    )
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

    global_step = 0
    compile_error = None
    compile_s = 0.0
    if args.variant.startswith("compile_"):
        events.write("compile_start", {"variant": args.variant})
        mode = None
        if args.variant == "compile_reduce_overhead":
            mode = "reduce-overhead"
        elif args.variant == "compile_max_autotune":
            mode = "max-autotune"
        t0 = time.perf_counter()
        try:
            loss_step = cast("nn.Module", torch.compile(loss_step, mode=mode, fullgraph=args.compile_fullgraph_check))
            setattr(loss_step, "_hydra_compiled", True)
            targets = targets_for_compiled_loss(targets, weights)
            run_step(loss_step, optimizer, obs, targets, args.microbatch, autocast, False)
            torch.cuda.synchronize()
            if raw_pinned is not None:
                assert raw_first_batch is not None
                raw_pinned.mark_inflight(raw_first_batch)
            compile_s = time.perf_counter() - t0
            events.write("compile_complete", {"compile_s": compile_s})
            scalar_writer.add_scalar("runtime/compile_s", compile_s, 0)
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            error_compile_s = time.perf_counter() - t0
            events.write("compile_error", {"error": compile_error, "compile_s": error_compile_s})
            scalar_writer.add_scalar("runtime/compile_error", 1.0, 0)

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
    for _ in range(args.warmup):
        pinned_batch: PinnedPolicyBatch | None = None
        if raw_stream is not None:
            batch, fetch_ms = raw_stream.next_batch()
            obs, legal, labels, targets, _input_timing = tensors_from_policy_batch(batch, device, fetch_ms)
        elif raw_pinned is not None:
            pinned_batch, fetch_ms = raw_pinned.next_batch()
            obs, legal, labels, targets, _input_timing = tensors_from_pinned_policy_batch(
                pinned_batch, device, fetch_ms
            )
        elif real_dataset is not None:
            obs, legal, labels, targets, _input_timing = tensors_from_real_batch(real_dataset, device)
        targets = targets_for_compiled_loss(targets, weights)
        run_step(loss_step, optimizer, obs, targets, args.microbatch, autocast, False)
        if raw_pinned is not None:
            assert pinned_batch is not None
            raw_pinned.mark_inflight(pinned_batch)
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

    events.write("warmup_complete", {"warmup_steps": args.warmup})
    scalar_writer.add_scalar("runtime/warmup_steps", float(args.warmup), 0)
    stats: list[StepStats] = []
    with profiler_ctx as torch_profiler:
        for i in range(args.steps):
            if args.profile:
                torch.cuda.nvtx.range_push(f"hydra_bc_step_{i}")
            input_timing = InputTiming()
            pinned_batch: PinnedPolicyBatch | None = None
            if raw_stream is not None:
                batch, fetch_ms = raw_stream.next_batch()
                obs, legal, labels, targets, input_timing = tensors_from_policy_batch(batch, device, fetch_ms)
            elif raw_pinned is not None:
                pinned_batch, fetch_ms = raw_pinned.next_batch()
                obs, legal, labels, targets, input_timing = tensors_from_pinned_policy_batch(
                    pinned_batch, device, fetch_ms
                )
            elif real_dataset is not None:
                obs, legal, labels, targets, input_timing = tensors_from_real_batch(real_dataset, device)
            targets = targets_for_compiled_loss(targets, weights)
            stat = run_step(
                loss_step,
                optimizer,
                obs,
                targets,
                args.microbatch,
                autocast,
                timed=not args.profile_coarse,
            )
            if real_dataset is not None or raw_stream is not None or raw_pinned is not None:
                stat.fetch_decode_ms = input_timing.fetch_decode_ms
                stat.h2d_wall_ms = input_timing.h2d_wall_ms
            stats.append(stat)
            global_step = (0 if resume_state is None else resume_state.global_step) + i + 1
            samples_seen = (0 if resume_state is None else resume_state.samples_seen) + (i + 1) * args.batch
            if args.log_every_steps > 0 and ((i + 1) % args.log_every_steps == 0 or i + 1 == args.steps):
                train_log.write(
                    "train_step",
                    {
                        "step": i + 1,
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                        "batch": args.batch,
                        "loss": stat.loss,
                        "step_ms": stat.step_ms,
                        "fwd_loss_ms": stat.fwd_loss_ms,
                        "backward_ms": stat.backward_ms,
                        "optimizer_ms": stat.optimizer_ms,
                        "train_gpu_ms": stat.train_gpu_ms,
                        "fetch_decode_ms": stat.fetch_decode_ms,
                        "h2d_wall_ms": stat.h2d_wall_ms,
                    }
                    | raw_mjai_scalar_snapshot(raw_stream, raw_pinned),
                )
                log_step_scalars(
                    scalar_writer,
                    stat,
                    batch=args.batch,
                    samples_seen=samples_seen,
                    global_step=global_step,
                )
                add_scalars(scalar_writer, "raw_mjai", raw_mjai_scalar_snapshot(raw_stream, raw_pinned), global_step)
                scalar_writer.flush()
            if raw_pinned is not None:
                assert pinned_batch is not None
                raw_pinned.mark_inflight(pinned_batch)
            if validation_stream is not None and args.validation_every > 0 and (i + 1) % args.validation_every == 0:
                step_eval = []
                for _ in range(args.validation_steps):
                    val_batch, val_fetch_ms = validation_stream.next_batch()
                    val_obs, _val_legal, _val_labels, val_targets, _val_input_timing = tensors_from_policy_batch(
                        val_batch, device, val_fetch_ms
                    )
                    val_targets = targets_for_compiled_loss(val_targets, weights)
                    step_eval.append(evaluate_batch(model, val_obs, val_targets, weights, autocast))
                metrics = summarize_eval(step_eval)
                eval_stats.append({"step": i + 1, "metrics": metrics})
                events.write("validation", {"step": i + 1, "global_step": global_step, "metrics": metrics})
                log_validation_scalars(scalar_writer, metrics, global_step, final=False)
                scalar_writer.flush()
            latest_checkpoint, step_checkpoint = checkpoint_paths(args, global_step)
            if (
                latest_checkpoint is not None
                and args.checkpoint_every_steps > 0
                and (i + 1) % args.checkpoint_every_steps == 0
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
    if validation_stream is not None:
        final_eval = []
        for _ in range(args.validation_steps):
            val_batch, val_fetch_ms = validation_stream.next_batch()
            val_obs, _val_legal, _val_labels, val_targets, _val_input_timing = tensors_from_policy_batch(
                val_batch, device, val_fetch_ms
            )
            val_targets = targets_for_compiled_loss(val_targets, weights)
            final_eval.append(evaluate_batch(model, val_obs, val_targets, weights, autocast))
        final_validation = summarize_eval(final_eval)
        log_validation_scalars(scalar_writer, final_validation, global_step, final=True)
    raw_progress: BuildProgress | None = None
    if raw_stream is not None:
        raw_progress = raw_stream.progress()
    elif raw_pinned is not None:
        raw_progress = raw_pinned.progress()

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
        "summary": summarize_steps(stats, args.batch),
        "validation": {"every": args.validation_every, "history": eval_stats, "final": final_validation},
        "memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "step_stats": [asdict(s) for s in stats],
        "real_shard_training": real_dataset is not None,
        "raw_mjai_training": raw_stream is not None or raw_pinned is not None,
        "raw_mjai_pinned_pyo3": raw_pinned is not None,
        "raw_mjai_transport": args.raw_mjai_transport,
        "raw_mjai_progress": json_raw_mjai_progress(raw_progress),
        "raw_mjai_bridge_stats": None if raw_pinned is None else asdict(raw_pinned.bridge_stats()),
        "raw_mjai_queue_stats": None if raw_pinned is None else asdict(raw_pinned.queue_stats()),
        "checkpoint_path": None,
        "resumed_step": None if resume_state is None else resume_state.global_step,
        "resumed_samples_seen": None if resume_state is None else resume_state.samples_seen,
        "global_step": (0 if resume_state is None else resume_state.global_step) + args.steps,
        "samples_seen": (0 if resume_state is None else resume_state.samples_seen) + args.steps * args.batch,
    }
    global_step = (0 if resume_state is None else resume_state.global_step) + args.steps
    samples_seen = (0 if resume_state is None else resume_state.samples_seen) + args.steps * args.batch
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
    if validation_stream is not None:
        validation_stream.close()
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
