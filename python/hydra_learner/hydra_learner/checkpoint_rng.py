"""RNG capture/restore helpers for data-only checkpoints."""

from __future__ import annotations

import random
from typing import Any, cast

import numpy as np
import torch


def _numpy_rng_state() -> tuple[Any, ...]:
    return cast("tuple[Any, ...]", np.random.get_state())


def capture_rng_state() -> dict[str, Any]:
    np_name, np_keys, np_pos, np_has_gauss, np_cached_gaussian = _numpy_rng_state()
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.astype(np.uint32, copy=False).copy()),
            "pos": np_pos,
            "has_gauss": np_has_gauss,
            "cached_gaussian": np_cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    numpy_state = state["numpy_random"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda", [])
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
