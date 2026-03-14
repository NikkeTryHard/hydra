from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import orbax.checkpoint as ocp


@dataclass(frozen=True)
class RuntimeResumeContract:
    manifest_fingerprint: str
    batch_size: int
    train_microbatch_size: int
    validation_microbatch_size: int
    model_fingerprint: str
    optimizer_fingerprint: str


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    next_batch_index: int
    runtime: RuntimeResumeContract


def save_checkpoint(checkpoint_dir: Path, state, metadata: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dir = checkpoint_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(state_dir, state)
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def restore_checkpoint(checkpoint_dir: Path, item=None):
    checkpointer = ocp.PyTreeCheckpointer()
    if item is None:
        return checkpointer.restore(checkpoint_dir / "state")
    return checkpointer.restore(checkpoint_dir / "state", item=item)


def write_resume_state(checkpoint_dir: Path, state: ResumeState) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "resume_state.json").write_text(
        json.dumps(asdict(state), indent=2, sort_keys=True)
    )


def read_resume_state(checkpoint_dir: Path) -> ResumeState:
    raw = json.loads((checkpoint_dir / "resume_state.json").read_text())
    runtime = RuntimeResumeContract(**raw["runtime"])
    return ResumeState(
        epoch=raw["epoch"],
        global_step=raw["global_step"],
        next_batch_index=raw["next_batch_index"],
        runtime=runtime,
    )
