"""Atomic, weights-only model and optimizer snapshots for the versioned actor wire."""

from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint_io import atomic_torch_save
from .model import ModelConfig, Policy
from .model_schema import WIRE_VERSION

FORMAT = "hydra3.policy.v1"
UTILITY = "tenhou_4p_hanchan_centered_placement_v1"


def save_checkpoint(
    path: Path,
    model: Policy,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    updates: int = 0,
) -> None:
    """Save a completed generation boundary; updates counts generations, not optimizer steps."""
    payload = {
        "format": FORMAT,
        "wire": WIRE_VERSION,
        "utility": UTILITY,
        "config": asdict(model.config),
        "weights": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "updates": updates,
    }
    atomic_torch_save(Path(path), payload)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Policy, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("not a Hydra3 policy checkpoint")
    if payload.get("wire") != WIRE_VERSION or payload.get("utility") != UTILITY:
        raise ValueError("checkpoint action/observation or placement contract is incompatible")
    if type(payload.get("updates")) is not int or payload["updates"] < 0:
        raise ValueError("invalid checkpoint update count")
    model = Policy(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["weights"], strict=True)
    return model.to(device).eval(), payload
