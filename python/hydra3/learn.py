"""Public full-match PPO and behavior-cloning entry points."""

from pathlib import Path
from time import perf_counter_ns

import torch

from .learn_batch import MetricValue, finite_metrics
from .learn_bc import update_bc
from .learn_collect import collect_generation
from .learn_dataset import load_dataset, save_dataset
from .learn_ppo import update_ppo
from .learn_schema import (
    DATASET_FORMAT,
    DATASET_VERSION,
    PROVENANCE_SELF_PLAY,
    DecisionDataset,
    GameRecord,
)

__all__ = [
    "DATASET_FORMAT",
    "DATASET_VERSION",
    "PROVENANCE_SELF_PLAY",
    "DecisionDataset",
    "GameRecord",
    "bc",
    "load_dataset",
    "ppo",
    "save_dataset",
]


def _positive(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return value


def _device(value: torch.device) -> torch.device:
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported training device: {device}")
    return device


def ppo(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    games: int,
    seed: int,
    device: torch.device,
    epochs: int,
    minibatch: int,
    record: Path | None = None,
    telemetry: dict[str, int] | None = None,
    compile_learner: bool = False,
    graph_actor: bool = False,
    compile_actor: bool = False,
) -> dict[str, MetricValue]:
    """Collect one frozen-policy generation and apply its PPO update."""
    games = _positive("games", games)
    epochs = _positive("epochs", epochs)
    minibatch = _positive("minibatch", minibatch)
    seed = _seed(seed)
    training_device = _device(device)
    if (compile_learner or graph_actor or compile_actor) and training_device.type != "cuda":
        raise ValueError("CUDA optimizations require CUDA")

    model.eval()
    start = perf_counter_ns() if telemetry is not None else 0
    if telemetry is not None:
        telemetry["collect_started_ns"] = start
    # The rollout policy is frozen. Clear cached casts before the learner
    # mutates weights, or inference could reuse stale BF16 parameters.
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=training_device.type,
            dtype=torch.bfloat16,
            enabled=training_device.type == "cuda",
        ),
    ):
        rollout = collect_generation(
            model,
            games=games,
            seed=seed,
            device=training_device,
            telemetry=telemetry,
            graph_actor=graph_actor,
            compile_actor=compile_actor,
        )
    if telemetry is not None:
        telemetry["collect_ns"] = perf_counter_ns() - start
    if record is not None:
        save_dataset(
            Path(record),
            (game.record for game in rollout.games),
            PROVENANCE_SELF_PLAY,
        )

    model.train()
    start = perf_counter_ns() if telemetry is not None else 0
    if telemetry is not None:
        telemetry["learn_started_ns"] = start
    try:
        metrics = update_ppo(
            model,
            optimizer,
            rollout,
            device=training_device,
            epochs=epochs,
            minibatch=minibatch,
            seed=seed,
            telemetry=telemetry,
            compile_learner=compile_learner,
        )
    finally:
        model.eval()
    if telemetry is not None:
        telemetry["learn_ns"] = perf_counter_ns() - start
        if training_device.type == "cuda":
            telemetry["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                training_device
            )

    ranks = rollout.ranks.float()
    metrics["mean_rank"] = float(ranks.mean())
    metrics["mean_utility"] = float((2.5 - ranks).mean())
    metrics["seed"] = seed
    return finite_metrics(metrics)


def bc(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    source: Path,
    device: torch.device,
    epochs: int,
    minibatch: int,
    seed: int,
) -> dict[str, MetricValue]:
    """Imitate an explicitly supplied corpus.

    The caller asserts the corpus is actor-visible and authorized for training.
    """
    epochs = _positive("epochs", epochs)
    minibatch = _positive("minibatch", minibatch)
    seed = _seed(seed)
    training_device = _device(device)
    dataset = load_dataset(Path(source))

    model.train()
    try:
        metrics = update_bc(
            model,
            optimizer,
            dataset.games,
            device=training_device,
            epochs=epochs,
            minibatch=minibatch,
            seed=seed,
        )
    finally:
        model.eval()
    metrics["source_provenance"] = dataset.provenance
    metrics["seed"] = seed
    return finite_metrics(metrics)
