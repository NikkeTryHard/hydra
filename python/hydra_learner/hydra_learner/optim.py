from __future__ import annotations

import argparse
import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from hydra_learner.checkpoint import EmaConfig, OptimizerConfig


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
    target_games: int | None
    schedule: str


class LrScheduler:
    def __init__(self, config: LrSchedulerConfig) -> None:
        self.config = config

    def lr_for_step(self, completed_steps: int, completed_games: int | None = None) -> float:
        if self.config.schedule == "constant":
            return self.config.base_lr
        if self.config.warmup_steps > 0 and completed_steps < self.config.warmup_steps:
            return self.config.base_lr * (completed_steps / self.config.warmup_steps)
        if self.config.schedule != "cosine":
            raise ValueError(f"unsupported LR schedule {self.config.schedule!r}")
        horizon = self.config.target_games if self.config.target_games is not None else self.config.total_steps
        if horizon is None:
            raise ValueError("cosine LR schedule requires total_steps or target_games")
        if self.config.target_games is not None and completed_games is not None:
            decay_progress = completed_games
            decay_steps = max(1, horizon)
        else:
            decay_progress = completed_steps - self.config.warmup_steps
            decay_steps = max(1, horizon - self.config.warmup_steps)
        decay_index = min(max(0, decay_progress), decay_steps)
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
        target_games=args.schedule_target_games,
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
        target_games=args.schedule_target_games,
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
