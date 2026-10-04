"""Generation restart gates: paired capture, bounded writer, verified resume.

Consumer-visible gates over ``hydra2.training.rl_restart`` plus the native
restart validators (needs ``pixi run build-ext`` first; the import fails
loud otherwise):

- capture freezes paired learner-final/actor CPU states: later live
  mutations must not move the saved snapshot;
- the writer holds at most one active plus one deferred checkpoint: a newer
  boundary replaces only the unstarted save, worker errors surface, and the
  prior pointer stays authoritative;
- resume loads the exact container behind the pointer, verifies
  training/sampler sections against the descriptor, restores RNG last, and
  rejects missing pointers and foreign schemas loudly.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.training.rl_restart import (
    RESTART_POINTER_NAME,
    _RestartWriter,
    capture_restart,
    load_restart,
    save_restart,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-11")

_C0 = "sha256:" + "22" * 32
_PIPELINE = "sha256:" + "aa" * 32


def _descriptor() -> dict[str, Any]:
    """Minimal run descriptor carrying exactly what restart validation reads."""
    return {
        "pipeline_spec_id": _PIPELINE,
        "initial_checkpoint_id": _C0,
        "cohort": {"generations": 8},
    }


def _cold() -> SimpleNamespace:
    """Tiny CPU learner/actor/optimizer/scheduler triple (deterministic)."""
    torch.manual_seed(1234)
    learner = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    actor = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    actor.load_state_dict(learner.state_dict())
    for module in (learner, actor):
        module.eval()
    optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    return SimpleNamespace(learner=learner, actor=actor, optimizer=optimizer, scheduler=scheduler)


def _native() -> Any:
    from hydra2._native import rl as native_rl  # pyrefly: ignore[missing-import]

    return native_rl


def _lineage(native: Any, descriptor: dict[str, Any], generation: int, parent: str) -> str:
    return str(
        native.policy_identity(
            run_descriptor_json=json.dumps(descriptor).encode(),
            generation=int(generation),
            parent_policy_id=str(parent),
        )
    )


def _capture(
    cold: SimpleNamespace,
    descriptor: dict[str, Any],
    *,
    next_generation: int,
    completed_updates: int,
    behavior: str,
    previous: str,
) -> dict[str, Any]:
    return capture_restart(
        cold=cold,
        descriptor=descriptor,
        next_generation=next_generation,
        completed_updates=completed_updates,
        behavior_policy_id=behavior,
        previous_behavior_policy_id=previous,
    )


def _clone_state(model: Any) -> dict[str, Any]:
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def _states_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if set(first) != set(second):
        return False
    return all(torch.equal(first[key], second[key]) for key in first)


def test_capture_freezes_snapshot_against_live_mutation() -> None:
    """Later learner/actor steps must not move the captured boundary."""
    native = _native()
    descriptor = _descriptor()
    cold = _cold()
    before_learner = _clone_state(cold.learner)
    before_actor = _clone_state(cold.actor)
    container = _capture(
        cold,
        descriptor,
        next_generation=1,
        completed_updates=29,
        behavior=_lineage(native, descriptor, 1, _C0),
        previous=_C0,
    )
    assert container["schema"] == "hydra2.rl_restart/v1"
    assert set(container["learner_payload"]) == {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "training_state",
        "sampler_state",
        "rng_state",
    }
    assert set(container["actor_state"]) == {"model_state"}
    assert container["descriptor"]["next_generation"] == 1
    assert container["descriptor"]["completed"] is False
    # Mutate both live modules through real optimizer steps.
    for _ in range(5):
        cold.optimizer.zero_grad()
        loss = cold.learner(torch.randn(4, 8)).square().mean()
        loss.backward()
        cold.optimizer.step()
    with torch.no_grad():
        for parameter in cold.actor.parameters():
            parameter.add_(1.0)
    assert not _states_equal(
        container["learner_payload"]["model_state"], dict(cold.learner.state_dict())
    )
    assert _states_equal(container["learner_payload"]["model_state"], before_learner)
    assert _states_equal(container["actor_state"]["model_state"], before_actor)


def test_writer_bounds_pending_and_newest_wins(tmp_path: Path) -> None:
    """One active plus one deferred: gen3 replaces gen2, gen1+gen3 land."""
    native = _native()
    descriptor = _descriptor()
    cold = _cold()
    c1 = _lineage(native, descriptor, 1, _C0)
    containers = {
        gen: _capture(
            cold,
            descriptor,
            next_generation=gen,
            completed_updates=10 * gen,
            behavior=_lineage(native, descriptor, gen, _C0) if gen > 1 else c1,
            previous=_C0,
        )
        for gen in (1, 2, 3)
    }
    root = tmp_path / "run"
    release = threading.Event()
    started: list[int] = []
    entered: list[int] = []
    writer = _RestartWriter()
    try:

        def _job(gen: int) -> Any:
            if gen == 1:
                started.append(gen)
                assert release.wait(timeout=30.0)
            entered.append(gen)
            return save_restart(root=root, run_descriptor=descriptor, container=containers[gen])

        writer.submit(lambda: _job(1), generation=1)
        # Wait until the first job is blocked inside the worker (deadline,
        # never a fixed sleep): later submits then deterministically defer.
        deadline = time.monotonic() + 30.0
        while not started:
            assert time.monotonic() < deadline, "first save never started"
            time.sleep(0.01)
        writer.submit(lambda: _job(2), generation=2)
        writer.submit(lambda: _job(3), generation=3)
        with pytest.raises(ContractError, match="older than submitted"):
            writer.submit(lambda: _job(1), generation=1)
        release.set()
        writer.drain()
        assert entered[0] == 1
        assert 2 not in entered
        assert entered[-1] == 3
        assert (root / "checkpoints" / "restart-g01.pt").exists()
        assert not (root / "checkpoints" / "restart-g02.pt").exists()
        assert (root / "checkpoints" / "restart-g03.pt").exists()
        pointer = json.loads((root / RESTART_POINTER_NAME).read_bytes())
        assert pointer["next_generation"] == 3
        loaded = load_restart(root=root, run_descriptor=descriptor)
        assert loaded["descriptor"]["completed_updates"] == 30
    finally:
        writer.close()


def test_writer_error_surfaces_and_pointer_stays_authoritative(tmp_path: Path) -> None:
    """A failed save raises on poll; no pointer is published."""
    root = tmp_path / "run"

    def _boom() -> None:
        raise RuntimeError("boom-save")

    writer = _RestartWriter()
    try:
        writer.submit(_boom, generation=1)
        deadline = time.monotonic() + 30.0
        while True:
            try:
                writer.poll()
            except RuntimeError:
                break
            assert time.monotonic() < deadline, "worker error never surfaced"
            time.sleep(0.01)
        assert not (root / RESTART_POINTER_NAME).exists()
    finally:
        writer.close()
    # Without a poll, close itself surfaces the pending worker error.
    writer = _RestartWriter()
    try:
        writer.submit(_boom, generation=2)
        deadline = time.monotonic() + 30.0
        while not writer._active.done():
            assert time.monotonic() < deadline, "worker never finished"
            time.sleep(0.01)
        with pytest.raises(RuntimeError):
            writer.close()
    finally:
        writer.close()


def test_resume_restores_weights_and_rng_last(tmp_path: Path) -> None:
    """Saved states apply strictly to fresh modules with RNG restored."""
    from hydra2.training.rl_checkpoint import restore_rng_last

    native = _native()
    descriptor = _descriptor()
    torch.manual_seed(777)
    cold = _cold()
    fixed_input = torch.randn(4, 8)
    with torch.no_grad():
        expected_actor = cold.actor(fixed_input).clone()
    c1 = _lineage(native, descriptor, 1, _C0)
    container = _capture(
        cold,
        descriptor,
        next_generation=1,
        completed_updates=29,
        behavior=c1,
        previous=_C0,
    )
    root = tmp_path / "run"
    save_restart(root=root, run_descriptor=descriptor, container=container)
    torch.randn(16, 16)
    loaded = load_restart(root=root, run_descriptor=descriptor)
    fresh = _cold()
    fresh.learner.load_state_dict(loaded["learner_payload"]["model_state"], strict=True)
    fresh.optimizer.load_state_dict(loaded["learner_payload"]["optimizer_state"])
    fresh.scheduler.load_state_dict(loaded["learner_payload"]["scheduler_state"])
    fresh.actor.load_state_dict(loaded["actor_state"]["model_state"], strict=True)
    with torch.no_grad():
        assert torch.equal(fresh.actor(fixed_input), expected_actor)
    cpu_before = torch.get_rng_state().clone()
    restore_rng_last(rng_state=loaded["learner_payload"]["rng_state"])
    assert not torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.get_rng_state(), loaded["learner_payload"]["rng_state"]["cpu"])


def test_resume_rejects_missing_and_foreign_schema(tmp_path: Path) -> None:
    """Missing pointers and foreign containers reject with fixed messages."""
    descriptor = _descriptor()
    with pytest.raises(ContractError, match="RL generation restart checkpoint missing"):
        load_restart(root=tmp_path / "absent", run_descriptor=descriptor)
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True)
    (root / RESTART_POINTER_NAME).write_text(
        json.dumps(
            {
                "schema": "hydra2.rl_restart_pointer/v1",
                "pipeline_spec_id": _PIPELINE,
                "next_generation": 1,
                "locator": "checkpoints/restart-g01.pt",
            }
        )
    )
    # Garbage bytes are unreadable (weights-only unpickler rejects them).
    (root / "checkpoints" / "restart-g01.pt").write_bytes(b"not-a-container")
    with pytest.raises(ContractError, match="rl_restart container unreadable"):
        load_restart(root=root, run_descriptor=descriptor)
    # A valid pickle with a foreign schema is incompatible, not unreadable.
    torch.save({"schema": "hydra2.foreign/v1"}, root / "checkpoints" / "restart-g01.pt")
    with pytest.raises(ContractError, match="RL generation restart schema incompatible"):
        load_restart(root=root, run_descriptor=descriptor)
