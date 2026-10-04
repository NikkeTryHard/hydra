"""Torch-only generation restart for native-live RL (no canonical rollout).

Capture/save/load orchestration for the paired generation-boundary
container (learner-final plus separately frozen actor); descriptor and
pointer validation/encoding live in ``hydra2._native.rl`` (Rust). The
ordinary supervised checkpoint manifest and generic writer behavior are
untouched: this container is the live-training restart authority, not a
second ordinary checkpoint convention.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from hydra2.contracts.common import ContractError
from hydra2.runtime.checkpoint import PAYLOAD_SECTION_KEYS
from hydra2.runtime.checkpoint_publish import _snapshot_payload_to_cpu
from hydra2.training.rl_checkpoint import _fsync_dir, ordinary_snapshot

RESTART_SCHEMA = "hydra2.rl_restart/v1"
RESTART_DESCRIPTOR_SCHEMA = "hydra2.rl_restart_descriptor/v1"
RESTART_POINTER_SCHEMA = "hydra2.rl_restart_pointer/v1"
RESTART_POINTER_NAME = "checkpoints/restart.json"


def _require_native_rl() -> Any:
    """Resolve the native live surface, fail closed when unbuilt."""
    from hydra2._native import rl as native_rl  # pyrefly: ignore[missing-import]

    for name in (
        "validate_restart_descriptor",
        "validate_restart_pointer",
        "publish_restart_pointer",
    ):
        if not hasattr(native_rl, name):
            raise ImportError(
                "hydra2._native extension with rl live restart not importable; "
                "run `pixi run build-ext` to build the extension before use"
            )
    return native_rl


def _encode(document: Any) -> bytes:
    try:
        return json.dumps(document).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"rl_restart document not JSON-encodable: {exc}") from exc


def _decode(blob: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(bytes(blob).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"rl_restart {name} not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"rl_restart {name} must be an object")
    return value


def _validate_descriptor(
    native: Any, run_descriptor: dict[str, Any], descriptor: dict[str, Any]
) -> dict[str, Any]:
    """Run the descriptor through Rust validation (never Python identity math)."""
    try:
        blob = bytes(
            native.validate_restart_descriptor(
                run_descriptor_json=_encode(run_descriptor),
                descriptor_json=_encode(descriptor),
            )
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return _decode(blob, name="restart descriptor")


def capture_restart(
    *,
    cold: Any,
    descriptor: dict[str, Any],
    next_generation: int,
    completed_updates: int,
    behavior_policy_id: str,
    previous_behavior_policy_id: str,
) -> dict[str, Any]:
    """Capture the paired boundary container on the owner (compute-stream ordered).

    Call after a completed generation before further learner mutation, on the
    actual compute stream after ``learner.boundary()``. All worker inputs are
    the immutable CPU clones returned here; workers never touch Torch CUDA
    state. Scalar/container/RNG facts come from the six-section payload, so
    only learner-final RNG is captured and restored.
    """
    if (
        not isinstance(next_generation, int)
        or isinstance(next_generation, bool)
        or not 0 <= next_generation <= 8
    ):
        raise ContractError(
            f"rl_restart next_generation must be an int in 0..8, got {next_generation!r}"
        )
    if (
        not isinstance(completed_updates, int)
        or isinstance(completed_updates, bool)
        or completed_updates < 0
    ):
        raise ContractError(
            f"rl_restart completed_updates must be a non-bool int >= 0, got {completed_updates!r}"
        )
    training_state = {
        "generation": next_generation,
        "completed_updates": completed_updates,
        "epoch": 0,
    }
    sampler_state = {
        "pipeline_spec_id": descriptor["pipeline_spec_id"],
        "generation": next_generation,
        "epoch": 0,
        "optimizer_cursor": 0,
        "micro_offset": 0,
        "source": "native_live_v1",
    }
    payload = ordinary_snapshot(
        model=cold.learner,
        optimizer=cold.optimizer,
        scheduler=cold.scheduler,
        sampler_state=sampler_state,
        training_state=training_state,
    )
    # Clone both states through the existing recursive law (actor weights
    # only, never actor RNG/optimizer); the extra key leaves with the CPU
    # snapshot step so the six-section law is undisturbed.
    payload["actor_state"] = {"model_state": cold.actor.state_dict()}
    snapped = _snapshot_payload_to_cpu(payload)
    actor_state = snapped.pop("actor_state")
    descriptor_doc = {
        "schema": RESTART_DESCRIPTOR_SCHEMA,
        "pipeline_spec_id": descriptor["pipeline_spec_id"],
        "initial_checkpoint_id": descriptor["initial_checkpoint_id"],
        "next_generation": next_generation,
        "completed_updates": completed_updates,
        "behavior_policy_id": behavior_policy_id,
        "previous_behavior_policy_id": previous_behavior_policy_id,
        "completed": next_generation == 8,
    }
    native = _require_native_rl()
    validated = _validate_descriptor(native, descriptor, descriptor_doc)
    return {
        "schema": RESTART_SCHEMA,
        "descriptor": validated,
        "learner_payload": snapped,
        "actor_state": actor_state,
    }


def _restart_path(root: Path, next_generation: int) -> Path:
    return root / f"checkpoints/restart-g{int(next_generation):02}.pt"


def save_restart(*, root: Path, run_descriptor: dict[str, Any], container: dict[str, Any]) -> Path:
    """Atomically save one container plus its pointer (worker thread entry).

    The file holds both states in one atomic unit (same-dir temp, fsync,
    replace, dir fsync); the pointer lands only after the file, via the
    native atomic publisher. Failures leave the prior pointer authoritative.
    Rerunning the same boundary atomically replaces its restart file.
    """
    import torch

    if not isinstance(container, dict) or container.get("schema") != RESTART_SCHEMA:
        raise ContractError("rl_restart container schema mismatch")
    for key in ("descriptor", "learner_payload", "actor_state"):
        if key not in container:
            raise ContractError(f"rl_restart container lacks {key}")
    native = _require_native_rl()
    validated = _validate_descriptor(native, run_descriptor, container["descriptor"])
    next_generation = int(validated["next_generation"])
    destination = _restart_path(Path(root), next_generation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=destination.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(container, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise
    _fsync_dir(destination.parent)
    try:
        native.publish_restart_pointer(
            root_dir=str(root),
            run_descriptor_json=_encode(run_descriptor),
            descriptor_json=_encode(validated),
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return destination


def _require_finite_model_state(model_state: Any) -> None:
    """Fail closed on non-finite saved weights (never resume corrupted math)."""
    import torch

    def _visit(node: Any, *, where: str) -> None:
        if isinstance(node, torch.Tensor):
            if not bool(torch.isfinite(node).all()):
                raise ContractError(f"rl_restart non-finite model state at {where}")
            return
        if isinstance(node, dict):
            for key, item in node.items():
                _visit(item, where=f"{where}.{key}")
            return
        if isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                _visit(item, where=f"{where}[{index}]")
            return

    _visit(model_state, where="model_state")


def load_restart(*, root: Path, run_descriptor: dict[str, Any]) -> dict[str, Any]:
    """Load and verify the restart container behind the pointer (owner entry).

    Missing pointer rejects ``RL generation restart checkpoint missing``;
    foreign schemas reject ``RL generation restart schema incompatible``.
    Training/sampler sections must equal the descriptor (generation, update
    count, pipeline, zeroed cursors); the pointer generation must equal the
    descriptor generation. RNG restores last at apply time, after
    shell/module/optimizer/slot init and state loading.
    """
    import torch

    native = _require_native_rl()
    pointer_path = Path(root) / RESTART_POINTER_NAME
    try:
        pointer_raw = pointer_path.read_bytes()
    except OSError:
        raise ContractError("RL generation restart checkpoint missing") from None
    try:
        pointer = _decode(pointer_raw, name="restart pointer")
        validated_pointer = _decode(
            bytes(
                native.validate_restart_pointer(
                    run_descriptor_json=_encode(run_descriptor),
                    pointer_json=_encode(pointer),
                )
            ),
            name="restart pointer",
        )
    except (ValueError, ContractError) as exc:
        raise ContractError("RL generation restart schema incompatible") from exc
    container_path = Path(root) / str(validated_pointer["locator"])
    try:
        container = torch.load(container_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ContractError(f"rl_restart container unreadable: {exc}") from exc
    if (
        not isinstance(container, dict)
        or container.get("schema") != RESTART_SCHEMA
        or not isinstance(container.get("descriptor"), dict)
        or not isinstance(container.get("learner_payload"), dict)
        or not isinstance(container.get("actor_state"), dict)
    ):
        raise ContractError("RL generation restart schema incompatible")
    try:
        validated = _validate_descriptor(native, run_descriptor, container["descriptor"])
    except (ValueError, ContractError) as exc:
        raise ContractError("RL generation restart schema incompatible") from exc
    if validated_pointer["next_generation"] != validated["next_generation"]:
        raise ContractError("RL generation restart schema incompatible")
    payload = container["learner_payload"]
    if set(payload.keys()) != set(PAYLOAD_SECTION_KEYS):
        raise ContractError("RL generation restart schema incompatible")
    training_state, sampler_state = payload["training_state"], payload["sampler_state"]
    next_generation, completed_updates = (
        int(validated["next_generation"]),
        int(validated["completed_updates"]),
    )

    def _field(section: Any, name: str) -> Any:
        if not isinstance(section, dict) or name not in section:
            raise ContractError("RL generation restart schema incompatible")
        return section[name]

    if (
        _field(training_state, "generation") != next_generation
        or _field(training_state, "completed_updates") != completed_updates
        or _field(training_state, "epoch") != 0
        or _field(sampler_state, "pipeline_spec_id") != validated["pipeline_spec_id"]
        or _field(sampler_state, "generation") != next_generation
        or _field(sampler_state, "epoch") != 0
        or _field(sampler_state, "optimizer_cursor") != 0
        or _field(sampler_state, "micro_offset") != 0
        or _field(sampler_state, "source") != "native_live_v1"
    ):
        raise ContractError("RL generation restart schema incompatible")
    actor_state = container["actor_state"]
    if set(actor_state.keys()) != {"model_state"}:
        raise ContractError("RL generation restart schema incompatible")
    _require_finite_model_state(payload["model_state"])
    _require_finite_model_state(actor_state["model_state"])
    return {
        "schema": RESTART_SCHEMA,
        "descriptor": validated,
        "learner_payload": payload,
        "actor_state": actor_state,
    }


class _RestartWriter:
    """Single-worker publisher of generation restart containers (Torch-only).

    At most one worker future active plus one deferred immutable job that was
    never submitted: ``submit`` never waits, ``poll`` surfaces worker errors
    then starts the deferred job, and a newer snapshot replaces only the
    unstarted deferred checkpoint (never skips RL updates or actor
    handoffs). The last durable checkpoint stays authoritative until the
    next publication lands. Memory is bounded to two retained CPU snapshots
    plus the transient capture; no unbounded queue, no error suppression.
    """

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active: Any = None
        self._deferred: Callable[[], Any] | None = None
        self._deferred_generation: int | None = None
        self._last_generation: int | None = None
        self._closed = False

    def submit(self, job: Callable[[], Any], generation: int) -> None:
        """Queue ``job`` (zero-arg callable) behind at most one active save."""
        if self._closed:
            raise ContractError("restart writer is closed")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ContractError(
                f"restart generation must be a non-negative int, got {generation!r}"
            )
        if self._last_generation is not None and generation < self._last_generation:
            raise ContractError(
                f"restart generation {generation} older than submitted {self._last_generation}"
            )
        self._last_generation = generation
        if self._active is None:
            self._active = self._executor.submit(job)
            return
        # One deferred slot: a newer boundary replaces the unstarted save.
        self._deferred = job
        self._deferred_generation = generation

    def poll(self) -> None:
        """Surface the active worker error, then start the deferred save."""
        if self._active is None:
            return
        if not self._active.done():
            return
        try:
            self._active.result()
        finally:
            self._active = None
        if self._deferred is not None:
            job, self._deferred = self._deferred, None
            self._deferred_generation = None
            self._active = self._executor.submit(job)

    def drain(self) -> None:
        """Block until the active and newest deferred saves land (in order)."""
        while self._active is not None or self._deferred is not None:
            if self._active is None:
                pending = self._deferred
                self._deferred = None
                self._deferred_generation = None
                if pending is None:
                    raise ContractError("restart writer lost its deferred save")
                self._active = self._executor.submit(pending)
                continue
            try:
                self._active.result()
            finally:
                self._active = None
            if self._deferred is not None:
                pending = self._deferred
                self._deferred = None
                self._deferred_generation = None
                if pending is None:
                    raise ContractError("restart writer lost its deferred save")
                self._active = self._executor.submit(pending)

    def close(self) -> None:
        """Drain once, then shut the worker down (idempotent)."""
        if self._closed:
            return
        try:
            self.drain()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True)


__all__ = [
    "RESTART_DESCRIPTOR_SCHEMA",
    "RESTART_POINTER_NAME",
    "RESTART_POINTER_SCHEMA",
    "RESTART_SCHEMA",
    "capture_restart",
    "load_restart",
    "save_restart",
]
