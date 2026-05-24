from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hydra_learner.checkpoint import ModelConfig, OptimizerConfig, ResumeState, RuntimeConfig, save_checkpoint
from hydra_learner.hydra_logging import json_raw_mjai_progress
from hydra_learner.raw_mjai import (
    BuildProgress,
    RawMjaiDirectStream,
    RawMjaiPinnedStream,
    build_progress_json,
)

if TYPE_CHECKING:
    import torch

    from hydra_learner.checkpoint import EmaConfig
    from hydra_learner.losses import LossWeights
    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.optim import EmaTracker


@dataclass(frozen=True)
class RawMjaiResumeOffsets:
    """Raw-MJAI resume cursor skips deterministic completed games before streaming."""

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

    @property
    def completed_games(self) -> int:
        return self.loaded_games + self.skipped_games


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
