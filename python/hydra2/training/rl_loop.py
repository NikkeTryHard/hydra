"""Single CUDA owner dispatching native cohorts and complete PPO updates."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hydra2.contracts.common import ContractError
from hydra2.training._rc_digest import run_dir_for
from hydra2.training.rl_checkpoint import (
    prepare_torch_state,
    restore_actor_weights,
    restore_rng_last,
)
from hydra2.training.rl_checkpoint_native import (
    live_collector_spec,
    native_choose_work,
    native_ppo_report,
    policy_identity,
)
from hydra2.training.rl_collect import Collector
from hydra2.training.rl_learner import LearnerService
from hydra2.training.rl_objective import PpoParameters
from hydra2.training.rl_restart import (
    _RestartWriter,
    capture_restart,
    load_restart,
    save_restart,
)


def _encode(value: Any) -> bytes:
    return json.dumps(value).encode("utf-8")


@dataclass(slots=True)
class _Generation:
    spec: dict[str, Any]
    policy_id: str
    tape: int | None = None
    replay: int | None = None
    row_count: int = 0
    batches: Any = None
    prep: Any = None


def run_rl_training(
    config: Any, resume: str | None = None, *, initial_checkpoint: Path | None = None
) -> dict[str, Any]:
    from hydra2._native import rl as native  # pyrefly: ignore[missing-import]
    from hydra2._native import rl_artifact as artifacts  # pyrefly: ignore[missing-import]
    from hydra2.runtime.plain import configure_torch_matmul

    _ = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_torch_matmul()
    torch.use_deterministic_algorithms(True)
    # Precision note: bf16 reduced-precision GEMM reductions stay at their
    # default (True). Disabling them measured 329.9s versus 332.7-335.2s but
    # with reforked trajectories from update 1 (235 versus 237 updates), so the
    # wall delta is different work, not faster compute, and the bitwise
    # trajectory contract breaks. This default is load-bearing, not hardening.
    root = run_dir_for(config)
    source = initial_checkpoint if initial_checkpoint is not None else root / "checkpoints/c0.pt"
    device = torch.device(config.runtime.device)
    cold = prepare_torch_state(
        config=config,
        source=source,
        device=device,
        repo_root=Path(__file__).resolve().parents[3],
        run_root=root,
    )
    descriptor, recipe = cold.native_run_descriptor, config.rl
    derivation = descriptor["row_derivation_hash"]
    pipeline_spec_id = descriptor["pipeline_spec_id"]
    current_generation = completed_updates = batch_cursor = 0
    behavior_id = previous_behavior_id = descriptor["initial_checkpoint_id"]

    restart_rng_state: Any = None
    if resume is not None:
        container = load_restart(root=root, run_descriptor=descriptor)
        restarted = container["descriptor"]
        if restarted["completed"]:
            return {
                "start_update": 0,
                "end_update": int(restarted["completed_updates"]),
                "train_games": recipe.generations * recipe.games_per_generation,
                "val_games": 0,
                "quarantined": 0,
                "checkpoints": [],
                "stopped": False,
                "elapsed_s": 0.0,
            }
        current_generation = int(restarted["next_generation"])
        completed_updates = int(restarted["completed_updates"])
        batch_cursor = 0
        behavior_id = str(restarted["behavior_policy_id"])
        previous_behavior_id = str(restarted["previous_behavior_policy_id"])
        payload = container["learner_payload"]
        cold.learner.load_state_dict(payload["model_state"], strict=True)
        cold.optimizer.load_state_dict(payload["optimizer_state"])
        cold.scheduler.load_state_dict(payload["scheduler_state"])
        restore_actor_weights(model=cold.actor, weights=container["actor_state"]["model_state"])
        restart_rng_state = payload["rng_state"]
    spans: list[tuple[str, float]] = []
    seen_marks: set[str] = set()

    def mark(event: str) -> None:
        spans.append((event, time.monotonic()))

    def mark_once(event: str) -> None:
        if event not in seen_marks:
            seen_marks.add(event)
            mark(event)

    current_spec = live_collector_spec(
        run_descriptor=descriptor,
        generation=current_generation,
        policy_id=behavior_id,
    )
    mark(f"collector-open-start-g{current_generation}-cold")
    collector = Collector(
        native,
        current_spec,
        derivation,
        gpu=None,
        device=device,
        capacity=recipe.inference_cap,
        pipeline_spec_id=pipeline_spec_id,
        compile_mode=config.runtime.compile_mode,
        span_sink=mark,
    )
    mark(f"collector-open-end-g{current_generation}-cold")
    gpu = collector.gpu
    learner = LearnerService(
        cold.learner,
        cold.optimizer,
        cold.scheduler,
        native=native,
        compute_stream=gpu.compute_stream,
        copy_stream=gpu.copy_stream,
        device=device,
        parameters=PpoParameters(
            recipe.clip_eps, recipe.eps_std, recipe.w_value, recipe.w_bc, recipe.ppo_alpha
        ),
        microbatch_cap=recipe.microbatch_cap,
        optimizer_cap=recipe.optimizer_minibatch_cap,
        gradient_clip_norm=recipe.gradient_clip_norm,
        cache_byte_budget=recipe.gpu_cache_byte_budget,
        compile_mode=config.runtime.compile_mode,
        span_sink=mark,
    )
    current = _Generation(spec=current_spec, policy_id=behavior_id)
    future: _Generation | None = None
    checkpoint_paths: list[str] = []
    writer = _RestartWriter()
    stopped = False
    if restart_rng_state is not None:
        restore_rng_last(rng_state=restart_rng_state)
    start_update, started = completed_updates, time.monotonic()
    preferred_inference = True
    prepared_slot: int | None = None
    active_slot = 0
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    def activate_future() -> None:
        nonlocal future, collector, preferred_inference
        # Old collection is fully complete/drained (current replay exists);
        # learner is between updates. Drain issued actor work, publish the
        # frozen actor in memory, and admit the one future cohort. No CPU
        # tensor snapshot, checkpoint hash, fsync, or writer future gates
        # this activation. No further actor mutation until the next handoff.
        _ = gpu.drain(wait=True)
        mark(f"actor-copy-start-g{current_generation + 1}")
        gpu.copy_actor_at_boundary(cold.learner, cold.actor)
        mark(f"actor-copy-end-g{current_generation + 1}")
        next_policy = policy_identity(
            run_descriptor=descriptor,
            generation=current_generation + 1,
            parent_policy_id=behavior_id,
        )
        future_spec = live_collector_spec(
            run_descriptor=descriptor,
            generation=current_generation + 1,
            policy_id=next_policy,
        )
        pending = _Generation(spec=future_spec, policy_id=next_policy)
        mark(f"collector-open-start-g{current_generation + 1}")
        new_collector: Any = None
        try:
            new_collector = Collector(
                native,
                future_spec,
                derivation,
                gpu=gpu,
                device=device,
                capacity=recipe.inference_cap,
                pipeline_spec_id=pipeline_spec_id,
                compile_mode=config.runtime.compile_mode,
                span_sink=mark,
            )
            new_collector.resume_admission()
        except BaseException:
            # Construction may fail before Collector binds its pool (open_live
            # raises inside __init__): nothing to close there. After binding,
            # close the orphan before surfacing so its threads never leak.
            if new_collector is not None:
                with contextlib.suppress(Exception):
                    new_collector.pool.close()
                if collector is not new_collector:
                    gpu.pool = collector.pool
            raise
        mark(f"collector-open-end-g{current_generation + 1}")
        collector = new_collector
        future = pending

    loop_error: BaseException | None = None

    def _write_spans(path: Path) -> None:
        _ = path.write_text(json.dumps([(event, t) for event, t in spans]))

    def _cleanup(stage: str, action: Any) -> None:
        nonlocal loop_error
        try:
            action()
        except BaseException as exc:  # keep cleaning; surface everything below
            if loop_error is None:
                loop_error = exc
            else:
                loop_error.add_note(f"{stage} cleanup: {exc!r}")
            mark(f"cleanup-error-{stage}")

    def _join_generation_prep(generation: Any) -> None:
        if generation is not None and generation.prep is not None:
            # Shutdown joins preparation before tape/arena release.
            handle, _ = native.wait_live_replay(job_id=int(generation.prep))
            generation.prep = None
            artifacts.drop_replay(handle)

    try:
        while current_generation < recipe.generations:
            writer.poll()
            inference_ready = collector.poll() if not collector.complete else False
            target = future if future is not None else current
            if target.replay is None and target.prep is None and collector.complete:
                mark(f"cohort-complete-g{target.spec['generation']}")
                mark(f"drain-start-g{target.spec['generation']}")
                target.tape = collector.export()
                mark(f"drain-end-g{target.spec['generation']}")
                mark(f"freeze-start-g{target.spec['generation']}")
                _ = gpu.freeze_terminal_advantages(artifacts, target.tape)
                mark(f"freeze-end-g{target.spec['generation']}")
                mark(f"prep-begin-g{target.spec['generation']}")
                target.prep = int(
                    native.begin_live_replay(
                        tape_handle=target.tape,
                        collector_spec_json=_encode(target.spec),
                        run_descriptor_json=_encode(descriptor),
                    )
                )
            elif target.prep is not None:
                status = json.loads(bytes(native.poll_live_replay(job_id=int(target.prep))))
                if status.get("status") == "done":
                    live = status["descriptor"]
                    target.replay = int(status["replay_handle"])
                    target.row_count = int(live["row_count"])
                    target.batches = list(live["batches"])
                    target.prep = None
                    mark(f"prep-done-g{target.spec['generation']}")
            batch_count = len(current.batches) if current.batches is not None else 0
            train_available = current.replay is not None and batch_cursor < batch_count
            if train_available and learner.active is None:
                if current.tape is None or current.replay is None:
                    raise ContractError(
                        "training admission requires a drained, frozen, registered replay"
                    )
                if prepared_slot is None:
                    mark(f"fill-start-g{current_generation}-b{batch_cursor}")
                    _ = learner.fill(
                        active_slot, native, current.tape, current.replay, batch_cursor
                    )
                    mark(f"fill-end-g{current_generation}-b{batch_cursor}")
                    mark_once(f"first-learner-g{current_generation}")
                else:
                    active_slot, prepared_slot = prepared_slot, None
                learner.begin(active_slot)
            if not collector.complete:
                collector.admit()
            kind, preferred_inference = native_choose_work(
                preferred_inference=preferred_inference,
                inference_eligible=inference_ready,
                learner_eligible=train_available,
            )
            if kind == "inference":
                collecting = current_generation + 1 if future is not None else current_generation
                mark_once(f"first-actor-g{collecting}")
                collector.infer(cold.actor)
            elif kind == "learner":
                if current.tape is None or current.replay is None:
                    raise ContractError("learner dispatch requires a registered live replay")
                finished = learner.microstep()
                if prepared_slot is None and batch_cursor + 1 < batch_count:
                    prepared_slot = 1 - active_slot
                    _ = learner.fill(
                        prepared_slot, native, current.tape, current.replay, batch_cursor + 1
                    )
                if finished:
                    count, terms = learner.finish()
                    completed_updates += 1
                    batch_cursor += 1
                    mark_once(f"first-update-g{current_generation}")
                    with torch.cuda.stream(gpu.compute_stream):
                        stats = (
                            torch.cat([tensor.reshape(-1).float() for tensor in terms])
                            .cpu()
                            .tolist()
                        )
                    report = native_ppo_report(count=count, stats=stats)
                    print(
                        f"rl: generation={current_generation} update={completed_updates} "
                        f"rows={count} loss={report['loss']:.6f}",
                        flush=True,
                    )
                    if (
                        batch_cursor == 1
                        and current_generation + 1 < recipe.generations
                        and future is None
                    ):
                        activate_future()

            collector.flush_if_no_next_ticket()
            if stopped and learner.active is None:
                collector.quiesce()
                writer.drain()
                break

            if (
                current.batches is not None
                and batch_cursor == batch_count
                and learner.active is None
            ):
                learner.boundary()
                mark(f"training-end-g{current_generation}")
                mark(f"capture-start-g{current_generation}")
                if future is not None:
                    restart_behavior, restart_previous = future.policy_id, behavior_id
                else:
                    restart_behavior, restart_previous = behavior_id, previous_behavior_id
                container = capture_restart(
                    cold=cold,
                    descriptor=descriptor,
                    next_generation=current_generation + 1,
                    completed_updates=completed_updates,
                    behavior_policy_id=restart_behavior,
                    previous_behavior_policy_id=restart_previous,
                )
                mark(f"capture-end-g{current_generation}")

                def _save(_container: Any = container, _gen: int = current_generation + 1) -> Any:
                    spans.append((f"save-start-g{_gen}", time.monotonic()))
                    try:
                        landed = save_restart(
                            root=root, run_descriptor=descriptor, container=_container
                        )
                        checkpoint_paths.append(str(landed))
                        return landed
                    finally:
                        spans.append((f"save-end-g{_gen}", time.monotonic()))

                writer.submit(_save, generation=current_generation + 1)
                artifacts.drop_replay(current.replay)
                artifacts.drop_tape(current.tape)
                current.replay = None
                current.tape = None
                if current_generation + 1 == recipe.generations:
                    current_generation += 1
                    break
                if future is None:
                    raise ContractError("generation handoff requires its future cohort")
                previous_behavior_id, behavior_id = behavior_id, future.policy_id
                current, future = future, None
                current_generation += 1
                batch_cursor = 0
                active_slot, prepared_slot = 0, None
                continue
            if kind == "idle":
                if target.prep is not None:
                    # Both work classes unavailable while preparation runs:
                    # join it here instead of idling.
                    handle, descriptor_blob = native.wait_live_replay(job_id=int(target.prep))
                    live = json.loads(bytes(descriptor_blob))
                    target.replay = int(handle)
                    target.row_count = int(live["row_count"])
                    target.batches = list(live["batches"])
                    target.prep = None
                    mark(f"prep-done-g{target.spec['generation']}")
                elif not collector.complete:
                    collector.wait()
                else:
                    raise ContractError(
                        "continuous RL has no work or pending source capable of progress"
                    )
    except BaseException as exc:
        # Exception exit runs the same owned cleanup below: the loop error
        # becomes the primary failure and every cleanup stage still runs.
        if loop_error is None:
            loop_error = exc
        else:  # pragma: no cover - defensive: _cleanup never raises
            exc.add_note(f"loop: {loop_error!r}")
            loop_error = exc
    _cleanup("writer-drain", writer.drain)
    _cleanup("gpu-quiesce", gpu.quiesce)
    _cleanup("pool-close", collector.pool.close)
    for generation in (current, future):
        if generation is None:
            continue
        _cleanup("prep-join", lambda generation=generation: _join_generation_prep(generation))
        if generation.replay is not None:
            _cleanup(
                "drop-replay",
                lambda generation=generation: artifacts.drop_replay(generation.replay),
            )
        if generation.tape is not None:
            _cleanup(
                "drop-tape",
                lambda generation=generation: artifacts.drop_tape(generation.tape),
            )
    spans_path = root / "handoff-spans.json"
    _cleanup("spans-write", lambda: _write_spans(spans_path))
    try:
        _cleanup("writer-close", writer.close)
    finally:
        for signum, handler in previous_handlers.items():
            _ = signal.signal(signum, handler)
    if loop_error is not None:
        raise loop_error
    return {
        "start_update": start_update,
        "end_update": completed_updates,
        "train_games": current_generation * recipe.games_per_generation,
        "val_games": 0,
        "quarantined": 0,
        "checkpoints": checkpoint_paths,
        "stopped": stopped,
        "elapsed_s": time.monotonic() - started,
    }
