#!/usr/bin/env python3
"""Compatibility entrypoint for the Hydra PyTorch BC learner."""

from __future__ import annotations

from hydra_learner.checkpointing.core import EmaConfig
from hydra_learner.cli import main
from hydra_learner.constants import WARMUP_MODE
from hydra_learner.data.batches import BaseTargets, synthetic_targets, targets_for_compiled_loss
from hydra_learner.model.optim import EmaTracker, LrScheduler, LrSchedulerConfig, set_optimizer_lr
from hydra_learner.telemetry.logging import SummaryWriter
from hydra_learner.training.loop import PreMainAccounting
from hydra_learner.training.step import HydraCompiledLossStep, run_non_mutating_train_step, run_step

__all__ = (
    "WARMUP_MODE",
    "BaseTargets",
    "EmaConfig",
    "EmaTracker",
    "HydraCompiledLossStep",
    "LrScheduler",
    "LrSchedulerConfig",
    "PreMainAccounting",
    "SummaryWriter",
    "main",
    "run_non_mutating_train_step",
    "run_step",
    "set_optimizer_lr",
    "synthetic_targets",
    "targets_for_compiled_loss",
)

if __name__ == "__main__":
    raise SystemExit(main())
