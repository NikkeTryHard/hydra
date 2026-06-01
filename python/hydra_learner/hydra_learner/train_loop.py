from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Literal, cast

import torch
import torch.cuda.profiler as cuda_profiler
import torch.nn as nn

from hydra_learner.batches import (
    InputTiming,
    StagedTrainBatch,
    make_synthetic_batch,
    next_staged_train_batch,
    synthetic_targets,
    targets_for_compiled_loss,
)
from hydra_learner.checkpoint import (
    ModelConfig,
    RuntimeConfig,
    load_checkpoint,
    load_checkpoint_metadata,
    target_contract_from_manifest,
)
from hydra_learner.checkpointing import (
    RawMjaiResumeOffsets,
    apply_progress_offsets,
    atomic_save_training_checkpoint,
    best_checkpoint_path,
    checkpoint_paths,
    checkpoint_raw_progress,
    raw_mjai_progress_dict,
    raw_mjai_progress_sections,
)
from hydra_learner.config import json_config
from hydra_learner.constants import COMPILE_DRY_RUN_MODE, COMPILED_LOSS_MODES, WARMUP_MODE
from hydra_learner.hydra_logging import (
    JsonlLogger,
    ScalarEventWriter,
    add_scalars,
    json_manifest_summary,
    log_final_scalars,
    log_step_scalars,
    log_validation_scalars,
    raw_mjai_scalar_snapshot,
    torch_env,
)
from hydra_learner.losses import LossWeights
from hydra_learner.metrics import StepStats, summarize_steps
from hydra_learner.model import HydraPolicyNet
from hydra_learner.optim import (
    EmaTracker,
    LrScheduler,
    build_ema_config,
    build_lr_scheduler_config,
    build_optimizer,
    build_optimizer_config,
    ema_weights,
    set_optimizer_lr,
)
from hydra_learner.raw_mjai import (
    RAW_MJAI_TRANSPORT_PINNED_PYO3,
    RAW_MJAI_TRANSPORT_STDOUT,
    BuildProgress,
    PinnedPolicyBatch,
    RawMjaiDirectStream,
    RawMjaiPinnedStream,
    default_raw_mjai_pyo3_library_path,
)
from hydra_learner.shard_manifest import validate_manifest
from hydra_learner.shard_reader import BcShardDataset
from hydra_learner.step import HydraCompiledLossStep, run_non_mutating_train_step, run_step
from hydra_learner.train_validation import ValidationConvergenceState, _validation_scalar_metrics
from hydra_learner.validation import RawMjaiValidationSource, evaluate_raw_and_ema


@dataclass(frozen=True)
class PreMainAccounting:
    compile_dry_run: bool = False
    compile_dry_run_changed_weights: bool = False
    warmup_mode: str = WARMUP_MODE
    warmup_steps_counted: int = 0
    warmup_steps_run: int = 0
    samples_consumed_pre_main: int = 0
    batches_consumed_pre_main: int = 0
    pre_main_batches_changed_weights: bool = False

    def as_log_fields(self) -> dict[str, object]:
        return asdict(self)


def run_training(args: argparse.Namespace) -> int:
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard_dir is not None:
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    events = JsonlLogger(None if args.log_dir is None else args.log_dir / "events.jsonl")
    train_log = JsonlLogger(None if args.log_dir is None else args.log_dir / "train_steps.jsonl")
    scalar_writer = ScalarEventWriter(args.tensorboard_dir)
    events.write("run_start", {"config": json_config(args)})
    if args.resume is not None:
        events.write("resume_requested", {"checkpoint_path": str(args.resume)})
    manifest_summary = None
    real_dataset = None
    raw_stream = None
    raw_pinned = None
    staged_initial_batch: StagedTrainBatch | None = None
    raw_train_max_samples = args.raw_mjai_max_samples
    if args.raw_mjai_data_dirs and raw_train_max_samples is None and not args.full_epoch and args.steps is not None:
        raw_train_batches = args.steps + 1
        raw_train_max_samples = raw_train_batches * args.batch
    raw_resume_offsets = RawMjaiResumeOffsets()
    raw_resume_state = None
    if args.raw_mjai_data_dirs and args.resume is not None:
        raw_resume_state = load_checkpoint_metadata(args.resume)
        raw_resume_offsets = RawMjaiResumeOffsets.from_resume(raw_resume_state, args.batch)
        args.raw_mjai_skip_games += raw_resume_offsets.completed_games

    events.write(
        "input_setup_start",
        {
            "manifest": str(args.manifest) if args.manifest is not None else None,
            "raw_mjai_transport": args.raw_mjai_transport if args.raw_mjai_data_dirs else None,
            "raw_mjai_data_dir_count": len(args.raw_mjai_data_dirs or ()),
            "raw_mjai_max_games": args.raw_mjai_max_games,
            "raw_mjai_max_samples": raw_train_max_samples,
            "raw_mjai_prefetch_batches": args.raw_mjai_prefetch_batches,
            "raw_mjai_queue_bound": args.raw_mjai_queue_bound,
            "raw_mjai_worker_threads": args.raw_mjai_worker_threads,
            "resume_requested": args.resume is not None,
            "raw_mjai_skip_games": args.raw_mjai_skip_games,
        },
    )
    if args.manifest is not None:
        events.write("manifest_validate_start", {"manifest": str(args.manifest)})
        manifest_summary = validate_manifest(args.manifest, check_files=args.check_shard_files)
        events.write("manifest_validate_complete", {"train_samples": manifest_summary.train_samples})
        real_dataset = BcShardDataset(args.manifest, batch_size=args.batch, split="train")
        events.write("input_setup_complete", {"kind": "bc_shards"})
    elif args.raw_mjai_data_dirs:
        events.write(
            "raw_mjai_stream_open_start",
            {
                "transport": args.raw_mjai_transport,
                "data_dir_count": len(args.raw_mjai_data_dirs),
                "resume_requested": args.resume is not None,
                "raw_mjai_skip_games": args.raw_mjai_skip_games,
            },
        )
        if args.raw_mjai_transport == RAW_MJAI_TRANSPORT_STDOUT:
            raw_stream = RawMjaiDirectStream(
                data_dirs=args.raw_mjai_data_dirs,
                batch_size=args.batch,
                prefetch_batches=args.raw_mjai_prefetch_batches,
                queue_bound=args.raw_mjai_queue_bound,
                worker_threads=args.raw_mjai_worker_threads,
                max_games=args.raw_mjai_max_games,
                max_samples=raw_train_max_samples,
                train_fraction=args.raw_mjai_train_fraction,
                augment=args.raw_mjai_augment,
                split=args.raw_mjai_split,
                skip_games=args.raw_mjai_skip_games,
            )
            raw_stream.start()
            events.write("input_setup_complete", {"kind": "raw_mjai_stdout"})
            events.write("raw_mjai_stream_open_complete", {"kind": "raw_mjai_stdout"})
        elif args.raw_mjai_transport == RAW_MJAI_TRANSPORT_PINNED_PYO3:
            raw_pinned = RawMjaiPinnedStream(
                data_dirs=args.raw_mjai_data_dirs,
                batch_size=args.batch,
                queue_bound=args.raw_mjai_queue_bound,
                worker_threads=args.raw_mjai_worker_threads,
                max_games=args.raw_mjai_max_games,
                max_samples=raw_train_max_samples,
                train_fraction=args.raw_mjai_train_fraction,
                augment=args.raw_mjai_augment,
                split=args.raw_mjai_split,
                library_path=args.raw_mjai_pyo3_lib or default_raw_mjai_pyo3_library_path(),
                ring_size=args.raw_mjai_prefetch_batches,
                skip_games=args.raw_mjai_skip_games,
            )
            events.write("input_setup_complete", {"kind": "raw_mjai_pinned_pyo3"})
            events.write("raw_mjai_stream_open_complete", {"kind": "raw_mjai_pinned_pyo3"})
        else:
            raise ValueError(f"unsupported raw MJAI transport {args.raw_mjai_transport!r}")
    events.write(
        "validation_setup_start",
        {"validation_steps": args.validation_steps, "validation_max_samples": args.validation_max_samples},
    )
    validation_source = RawMjaiValidationSource(args=args, events=events)
    events.write("validation_setup_complete", {"actual_batches": validation_source.info.actual_batches})

    if args.schedule_total_steps is None:
        if args.steps is not None:
            args.schedule_total_steps = args.steps
        elif manifest_summary is not None:
            args.schedule_total_steps = max(1, math.ceil(manifest_summary.train_samples / args.batch))
    if args.lr_schedule == "cosine" and args.schedule_total_steps is None and args.schedule_target_games is None:
        raise ValueError("cosine LR needs --schedule-total-steps or --schedule-target-games for unbounded runs")
    events.write("torch_setup_start", {})

    if not torch.cuda.is_available():
        result = {"variant": args.variant, "env": torch_env(), "error": "CUDA unavailable"}
        events.write("run_error", result)
        events.close()
        train_log.close()
        scalar_writer.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if not args.quiet:
            print(json.dumps(result, indent=2))
        return 2

    torch.manual_seed(0x51A7E)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    model = HydraPolicyNet(
        args.hidden,
        args.blocks,
        args.bottleneck,
        args.actions,
        args.residual_profile,
        args.backbone_profile,
        args.conv_memory_format,
    ).to(device)
    events.write(
        "model_setup_complete",
        {
            "hidden": args.hidden,
            "blocks": args.blocks,
            "bottleneck": args.bottleneck,
            "residual_profile": args.residual_profile,
            "conv_memory_format": args.conv_memory_format,
        },
    )
    optimizer_config = build_optimizer_config(args)
    optimizer = build_optimizer(model, optimizer_config)
    lr_scheduler = LrScheduler(build_lr_scheduler_config(args))
    if real_dataset is None and raw_stream is None and raw_pinned is None:
        obs, legal, labels = make_synthetic_batch(args.batch, args.actions, device)
        targets = synthetic_targets(obs, legal, labels)
        input_timing = InputTiming()
    else:
        obs = legal = labels = None
        targets = None
        input_timing = InputTiming()
    autocast = args.variant != "eager_fp32"
    weights = LossWeights(
        oracle_critic=args.w_oracle_critic,
        safety_residual=args.w_safety_residual,
        exit=args.w_exit,
        deltaq=args.w_deltaq,
    )
    target_manifest_path = args.manifest if raw_stream is None else raw_stream.manifest_path
    target_contract = target_contract_from_manifest(target_manifest_path, weights)
    if args.loss_mode not in COMPILED_LOSS_MODES:
        raise ValueError(f"unsupported loss mode {args.loss_mode!r}")
    loss_step: nn.Module = HydraCompiledLossStep(model, args.loss_mode, weights)
    model_config = ModelConfig(
        hidden=args.hidden,
        blocks=args.blocks,
        bottleneck=args.bottleneck,
        actions=args.actions,
        residual_profile=args.residual_profile,
        backbone_profile=args.backbone_profile,
        conv_memory_format=args.conv_memory_format,
    )
    precision_mode = "fp32" if args.variant == "eager_fp32" else "bf16_autocast"
    runtime_config = RuntimeConfig(
        variant=args.variant,
        loss_mode=args.loss_mode,
        precision_mode=precision_mode,
        compile_fullgraph_check=args.compile_fullgraph_check,
        compile_dry_run_mode=COMPILE_DRY_RUN_MODE,
        warmup_mode=WARMUP_MODE,
    )
    ema_config = build_ema_config(args)
    ema_tracker = None if ema_config is None else EmaTracker(model, ema_config, device)
    resume_state = None
    if args.resume is not None:
        events.write("checkpoint_load_start", {"checkpoint_path": str(args.resume)})
        resume_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_optimizer_config=optimizer_config,
            expected_runtime_config=runtime_config,
            expected_loss_weights=weights,
            expected_manifest_path=target_manifest_path,
            expected_ema_config=ema_config,
            expected_target_contract=target_contract,
        )
        if ema_tracker is not None:
            if resume_state.ema is None:
                raise ValueError("checkpoint EMA state missing after EMA resume validation")
            ema_tracker.load_state(
                resume_state.ema.state_dict,
                resume_state.ema.update_count,
                resume_state.ema.last_update_step,
            )
        events.write(
            "resume_loaded",
            {
                "checkpoint_path": str(args.resume),
                "global_step": resume_state.global_step,
                "samples_seen": resume_state.samples_seen,
            },
        )
        scalar_writer.add_scalar("checkpoint/resumed", 1.0, resume_state.global_step)
        scalar_writer.add_scalar("checkpoint/resumed_step", float(resume_state.global_step), resume_state.global_step)
        scalar_writer.add_scalar(
            "checkpoint/resumed_samples_seen",
            float(resume_state.samples_seen),
            resume_state.global_step,
        )
        scalar_writer.add_scalar("run/resume_global_step", float(resume_state.global_step), resume_state.global_step)
        events.write("checkpoint_load_complete", {"global_step": resume_state.global_step})
        if args.raw_mjai_data_dirs:
            events.write(
                "raw_mjai_checkpoint_resume_cursor",
                {
                    "global_step": resume_state.global_step,
                    "samples_seen": resume_state.samples_seen,
                    "skip_games": args.raw_mjai_skip_games,
                },
            )

    global_step = 0 if resume_state is None else resume_state.global_step
    samples_seen = 0 if resume_state is None else resume_state.samples_seen
    raw_mjai_offsets = RawMjaiResumeOffsets.from_resume(resume_state, args.batch)

    compile_error = None
    compile_s = 0.0
    events.write("first_batch_fetch_start", {})
    if real_dataset is not None or raw_stream is not None or raw_pinned is not None:
        staged_initial_batch = next_staged_train_batch(
            real_dataset=real_dataset,
            raw_stream=raw_stream,
            raw_pinned=raw_pinned,
            device=device,
            weights=weights,
        )
        obs = staged_initial_batch.obs
        targets = staged_initial_batch.targets
        input_timing = staged_initial_batch.input_timing
        events.write(
            "first_batch_fetch_complete",
            {
                "fetch_decode_ms": input_timing.fetch_decode_ms,
                "h2d_wall_ms": input_timing.h2d_wall_ms,
                "rows": obs.shape[0],
            },
        )
    else:
        assert obs is not None and targets is not None
        targets = targets_for_compiled_loss(targets, weights)
        events.write("first_batch_fetch_complete", {"kind": "synthetic", "rows": obs.shape[0]})
    pre_main = PreMainAccounting(
        compile_dry_run=args.variant.startswith("compile_"),
        warmup_steps_run=args.warmup,
        samples_consumed_pre_main=(obs.shape[0] if staged_initial_batch is not None else 0),
        batches_consumed_pre_main=(1 if staged_initial_batch is not None else 0),
    )
    staged_initial_pinned_marked = False
    if args.variant.startswith("compile_"):
        events.write(
            "compile_start",
            {"variant": args.variant, "compile_dry_run": True, "compile_dry_run_mode": COMPILE_DRY_RUN_MODE},
        )
        mode = None
        if args.variant == "compile_reduce_overhead":
            mode = "reduce-overhead"
        elif args.variant == "compile_max_autotune":
            mode = "max-autotune"
        t0 = time.perf_counter()
        try:
            loss_step = cast("nn.Module", torch.compile(loss_step, mode=mode, fullgraph=args.compile_fullgraph_check))
            setattr(loss_step, "_hydra_compiled", True)
            set_optimizer_lr(optimizer, lr_scheduler.lr_for_step(global_step))
            run_non_mutating_train_step(
                loss_step,
                model,
                optimizer,
                obs,
                targets,
                weights,
                args.loss_mode,
                args.microbatch,
                autocast,
                False,
                args.grad_clip_norm,
            )
            torch.cuda.synchronize()
            if raw_pinned is not None and staged_initial_batch is not None:
                assert staged_initial_batch.pinned_batch is not None
                raw_pinned.mark_inflight(staged_initial_batch.pinned_batch)
                staged_initial_pinned_marked = True
            compile_s = time.perf_counter() - t0
            events.write("compile_complete", {"compile_s": compile_s} | pre_main.as_log_fields())
            scalar_writer.add_scalar("runtime/compile_s", compile_s, global_step)
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            error_compile_s = time.perf_counter() - t0
            events.write("compile_error", {"error": compile_error, "compile_s": error_compile_s})
            scalar_writer.add_scalar("runtime/compile_error", 1.0, global_step)

    env = torch_env()
    if compile_error is not None:
        result = {
            "variant": args.variant,
            "env": env,
            "compile_error": compile_error,
            "compile_s": compile_s,
            "fullgraph_check": args.compile_fullgraph_check,
        }
        events.write("run_error", result)
        events.close()
        train_log.close()
        scalar_writer.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if not args.quiet:
            print(json.dumps(result, indent=2))
        return 2

    torch.cuda.reset_peak_memory_stats()
    eval_stats: list[dict[str, object]] = []
    events.write("warmup_start", {"warmup_steps": args.warmup})
    if args.warmup > 0:
        set_optimizer_lr(optimizer, lr_scheduler.lr_for_step(global_step))
    for _ in range(args.warmup):
        run_non_mutating_train_step(
            loss_step,
            model,
            optimizer,
            obs,
            targets,
            weights,
            args.loss_mode,
            args.microbatch,
            autocast,
            False,
            args.grad_clip_norm,
        )
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        cuda_profiler.start()

    profiler_ctx: AbstractContextManager[torch.profiler.profile | None]
    if args.torch_profiler_trace is None:
        profiler_ctx = nullcontext(None)
    else:
        args.torch_profiler_trace.parent.mkdir(parents=True, exist_ok=True)
        active_steps = args.torch_profiler_stop_step - args.torch_profiler_start_step

        def write_trace(prof: torch.profiler.profile) -> None:
            prof.export_chrome_trace(str(args.torch_profiler_trace))

        profiler_ctx = torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            schedule=torch.profiler.schedule(
                wait=args.torch_profiler_start_step,
                warmup=0,
                active=active_steps,
                repeat=1,
            ),
            on_trace_ready=write_trace,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )

    events.write("warmup_complete", {"warmup_steps": args.warmup} | pre_main.as_log_fields())
    scalar_writer.add_scalar("runtime/warmup_steps", float(args.warmup), global_step)
    scalar_writer.add_scalar("runtime/warmup_steps_counted", 0.0, global_step)
    scalar_writer.add_scalar(
        "runtime/samples_consumed_pre_main", float(pre_main.samples_consumed_pre_main), global_step
    )
    stats: list[StepStats] = []
    best_policy_nll = math.inf
    best_policy_nll_step = 0
    validation_convergence = ValidationConvergenceState()
    step_range = itertools.count() if args.steps is None else range(args.steps)
    with profiler_ctx as torch_profiler:
        for i in step_range:
            if args.full_epoch and i > 0 and raw_pinned is not None and raw_pinned.progress().complete:
                break
            if args.full_epoch and i > 0 and raw_stream is not None and raw_stream.progress().complete:
                break
            if args.profile:
                torch.cuda.nvtx.range_push(f"hydra_bc_step_{i}")
            pinned_batch: PinnedPolicyBatch | None = None
            pinned_batch_already_marked = False
            if staged_initial_batch is not None:
                staged = staged_initial_batch
                staged_initial_batch = None
                obs = staged.obs
                targets = staged.targets
                input_timing = staged.input_timing
                pinned_batch = staged.pinned_batch
                pinned_batch_already_marked = staged_initial_pinned_marked
            else:
                input_timing = InputTiming()
                if raw_stream is not None:
                    try:
                        staged = next_staged_train_batch(
                            real_dataset=real_dataset,
                            raw_stream=raw_stream,
                            raw_pinned=raw_pinned,
                            device=device,
                            weights=weights,
                        )
                    except StopIteration:
                        if args.full_epoch:
                            break
                        raise
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
                elif raw_pinned is not None:
                    try:
                        staged = next_staged_train_batch(
                            real_dataset=real_dataset,
                            raw_stream=raw_stream,
                            raw_pinned=raw_pinned,
                            device=device,
                            weights=weights,
                        )
                    except StopIteration:
                        if args.full_epoch:
                            break
                        raise
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
                    pinned_batch = staged.pinned_batch
                elif real_dataset is not None:
                    staged = next_staged_train_batch(
                        real_dataset=real_dataset,
                        raw_stream=raw_stream,
                        raw_pinned=raw_pinned,
                        device=device,
                        weights=weights,
                    )
                    obs = staged.obs
                    targets = staged.targets
                    input_timing = staged.input_timing
            completed_games_for_lr: int | None = None
            progress_for_lr = raw_pinned.progress() if raw_pinned is not None else None
            if progress_for_lr is None and raw_stream is not None:
                progress_for_lr = raw_stream.progress()
            if progress_for_lr is not None:
                completed_games_for_lr = progress_for_lr.loaded_games + raw_mjai_offsets.loaded_games

            lr = lr_scheduler.lr_for_step(global_step, completed_games_for_lr)
            set_optimizer_lr(optimizer, lr)
            will_log = args.log_every_steps > 0 and (
                (global_step + 1) % args.log_every_steps == 0 or (args.steps is not None and i + 1 == args.steps)
            )
            stat = run_step(
                loss_step,
                model,
                optimizer,
                obs,
                targets,
                weights,
                args.loss_mode,
                args.microbatch,
                autocast,
                timed=not args.profile_coarse,
                grad_clip_norm=args.grad_clip_norm,
                collect_diagnostics=will_log,
            )
            stat.lr = lr
            stat.lr_progress_games = math.nan if completed_games_for_lr is None else float(completed_games_for_lr)
            if real_dataset is not None or raw_stream is not None or raw_pinned is not None:
                stat.fetch_decode_ms = input_timing.fetch_decode_ms
                stat.h2d_wall_ms = input_timing.h2d_wall_ms
            stats.append(stat)
            global_step += 1
            samples_seen += obs.shape[0]
            if ema_tracker is not None:
                ema_tracker.maybe_update(model, global_step)
            ema_metrics = (
                {
                    "ema/enabled": False,
                    "ema/device": "off",
                    "ema/device_code": -1,
                    "ema/update_count": 0,
                    "ema/active": False,
                    "ema/last_update_step": 0,
                    "ema/last_update_ms": math.nan,
                }
                if ema_tracker is None
                else ema_tracker.metrics()
            )
            if will_log:
                train_log.write(
                    "train_step",
                    {
                        "step": i + 1,
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                        "batch": args.batch,
                        "loss": stat.loss,
                        "head_losses": stat.head_losses,
                        "target_coverage": stat.target_coverage,
                        "policy_nll": stat.policy_nll,
                        "policy_accuracy": stat.policy_accuracy,
                        "policy_top3_accuracy": stat.policy_top3_accuracy,
                        "policy_top5_accuracy": stat.policy_top5_accuracy,
                        "policy_confidence": stat.policy_confidence,
                        "policy_entropy": stat.policy_entropy,
                        "policy_target_prob": stat.policy_target_prob,
                        "policy_margin": stat.policy_margin,
                        "step_ms": stat.step_ms,
                        "fwd_loss_ms": stat.fwd_loss_ms,
                        "backward_ms": stat.backward_ms,
                        "optimizer_ms": stat.optimizer_ms,
                        "train_gpu_ms": stat.train_gpu_ms,
                        "fetch_decode_ms": stat.fetch_decode_ms,
                        "h2d_wall_ms": stat.h2d_wall_ms,
                        "lr": stat.lr,
                        "grad_norm": stat.grad_norm,
                        "lr_progress_games": stat.lr_progress_games,
                        "compile_dry_run": pre_main.compile_dry_run,
                        "warmup_mode": pre_main.warmup_mode,
                        "warmup_steps_counted": pre_main.warmup_steps_counted,
                        "samples_consumed_pre_main": pre_main.samples_consumed_pre_main,
                        "pre_main_batches_changed_weights": pre_main.pre_main_batches_changed_weights,
                    }
                    | ema_metrics
                    | raw_mjai_scalar_snapshot(raw_stream, raw_pinned, raw_mjai_offsets),
                )
                log_step_scalars(
                    scalar_writer,
                    stat,
                    batch=args.batch,
                    samples_seen=samples_seen,
                    global_step=global_step,
                )
                add_scalars(
                    scalar_writer,
                    "raw_mjai",
                    raw_mjai_scalar_snapshot(raw_stream, raw_pinned, raw_mjai_offsets),
                    global_step,
                )
                add_scalars(
                    scalar_writer,
                    "ema",
                    {key.removeprefix("ema/"): value for key, value in ema_metrics.items()},
                    global_step,
                )
                scalar_writer.flush()
            if raw_pinned is not None and not pinned_batch_already_marked:
                assert pinned_batch is not None
                raw_pinned.mark_inflight(pinned_batch)
            if args.validation_steps > 0 and args.validation_every > 0 and global_step % args.validation_every == 0:
                metrics, ema_metrics = evaluate_raw_and_ema(
                    validation_source,
                    steps=args.validation_steps,
                    model=model,
                    device=device,
                    weights=weights,
                    autocast=autocast,
                    ema_tracker=ema_tracker,
                )
                event_metrics = {"raw": metrics}
                if ema_metrics is not None:
                    event_metrics["ema"] = ema_metrics
                raw_policy_nll_value = metrics["policy_nll"]
                if not isinstance(raw_policy_nll_value, Real):
                    raise TypeError("validation policy_nll metric must be numeric")
                best_metrics = metrics
                best_weight_source: Literal["raw", "ema"] = "raw"
                if ema_metrics is not None:
                    ema_policy_nll_value = ema_metrics["policy_nll"]
                    if not isinstance(ema_policy_nll_value, Real):
                        raise TypeError("EMA validation policy_nll metric must be numeric")
                    if float(ema_policy_nll_value) < float(raw_policy_nll_value):
                        best_metrics = ema_metrics
                        best_weight_source = "ema"
                best_policy_nll_value = best_metrics["policy_nll"]
                if not isinstance(best_policy_nll_value, Real):
                    raise TypeError("best validation policy_nll metric must be numeric")
                policy_nll = float(best_policy_nll_value)
                convergence_metrics = validation_convergence.update(policy_nll, global_step)
                scalar_metrics = _validation_scalar_metrics(
                    raw_metrics=metrics,
                    ema_metrics=ema_metrics,
                    source_info=validation_source.info,
                    convergence_metrics=convergence_metrics,
                )
                eval_stats.append({"step": i + 1, "metrics": event_metrics})
                events.write(
                    "validation",
                    {
                        "step": i + 1,
                        "global_step": global_step,
                        "source": asdict(validation_source.info),
                        "metrics": event_metrics,
                    },
                )
                log_validation_scalars(scalar_writer, scalar_metrics, global_step, final=False)
                scalar_writer.flush()
                if policy_nll < best_policy_nll:
                    best_policy_nll = policy_nll
                    best_policy_nll_step = global_step
                    best_path = best_checkpoint_path(args)
                    if best_path is not None:
                        weight_ctx = ema_weights(model, ema_tracker) if best_weight_source == "ema" else nullcontext()
                        with weight_ctx:
                            atomic_save_training_checkpoint(
                                best_path,
                                model=model,
                                optimizer=optimizer,
                                model_config=model_config,
                                optimizer_config=optimizer_config,
                                runtime_config=runtime_config,
                                loss_weights=weights,
                                manifest_path=target_manifest_path,
                                global_step=global_step,
                                samples_seen=samples_seen,
                                raw_mjai_progress=checkpoint_raw_progress(raw_stream, raw_pinned, raw_mjai_offsets),
                                ema_tracker=ema_tracker,
                                ema_config=ema_config,
                                weight_source=best_weight_source,
                                target_contract=target_contract,
                            )
                        events.write(
                            "best_checkpoint_saved",
                            {
                                "path": str(best_path),
                                "metric": "policy_nll",
                                "metric_value": best_policy_nll,
                                "global_step": global_step,
                                "samples_seen": samples_seen,
                                "weight_source": best_weight_source,
                            },
                        )
                        scalar_writer.add_scalar("checkpoint/best_policy_nll", best_policy_nll, global_step)
                        scalar_writer.add_scalar("checkpoint/best_step", float(best_policy_nll_step), global_step)
            latest_checkpoint, step_checkpoint = checkpoint_paths(args, global_step)
            if (
                latest_checkpoint is not None
                and args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            ):
                for checkpoint_path in (latest_checkpoint, step_checkpoint):
                    if checkpoint_path is None:
                        continue
                    atomic_save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        runtime_config=runtime_config,
                        loss_weights=weights,
                        manifest_path=target_manifest_path,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        raw_mjai_progress=raw_mjai_progress_dict(
                            apply_progress_offsets(
                                raw_stream.progress()
                                if raw_stream is not None
                                else raw_pinned.progress()
                                if raw_pinned is not None
                                else None,
                                raw_mjai_offsets,
                            )
                        ),
                        ema_tracker=ema_tracker,
                        ema_config=ema_config,
                        target_contract=target_contract,
                    )
                events.write(
                    "checkpoint_saved",
                    {
                        "path": str(latest_checkpoint),
                        "step_path": None if step_checkpoint is None else str(step_checkpoint),
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                    },
                )
                scalar_writer.add_scalar("checkpoint/saved", 1.0, global_step)
                scalar_writer.add_scalar("checkpoint/samples_seen", float(samples_seen), global_step)
            if torch_profiler is not None:
                torch_profiler.step()
            if args.profile:
                torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    if args.cuda_profiler_range:
        cuda_profiler.stop()

    final_validation = None
    if args.validation_steps > 0:
        raw_final_validation, ema_final_validation = evaluate_raw_and_ema(
            validation_source,
            steps=args.validation_steps,
            model=model,
            device=device,
            weights=weights,
            autocast=autocast,
            ema_tracker=ema_tracker,
        )
        final_validation = {"raw": raw_final_validation}
        if ema_final_validation is not None:
            final_validation["ema"] = ema_final_validation
        final_best_metrics = raw_final_validation
        if ema_final_validation is not None:
            final_raw_policy_nll_value = raw_final_validation["policy_nll"]
            final_ema_policy_nll_value = ema_final_validation["policy_nll"]
            if not isinstance(final_raw_policy_nll_value, Real):
                raise TypeError("final raw validation policy_nll metric must be numeric")
            if not isinstance(final_ema_policy_nll_value, Real):
                raise TypeError("final EMA validation policy_nll metric must be numeric")
            if float(final_ema_policy_nll_value) < float(final_raw_policy_nll_value):
                final_best_metrics = ema_final_validation
        final_policy_nll_value = final_best_metrics["policy_nll"]
        if not isinstance(final_policy_nll_value, Real):
            raise TypeError("final validation policy_nll metric must be numeric")
        final_convergence_metrics = validation_convergence.update(float(final_policy_nll_value), global_step)
        final_scalar_metrics = _validation_scalar_metrics(
            raw_metrics=raw_final_validation,
            ema_metrics=ema_final_validation,
            source_info=validation_source.info,
            convergence_metrics=final_convergence_metrics,
        )
        log_validation_scalars(scalar_writer, final_scalar_metrics, global_step, final=True)
    raw_progress: BuildProgress | None = None
    if raw_stream is not None:
        raw_progress = raw_stream.progress()
    elif raw_pinned is not None:
        raw_progress = raw_pinned.progress()
    raw_stream_local_progress, raw_resume_total_progress = raw_mjai_progress_sections(raw_progress, raw_mjai_offsets)

    final_global_step = global_step
    final_samples_seen = samples_seen
    result = {
        "variant": args.variant,
        "env": env,
        "config": json_config(args, raw_train_max_samples),
        "manifest_summary": json_manifest_summary(
            raw_stream.manifest_summary
            if raw_stream is not None
            else (raw_pinned.manifest_summary if raw_pinned is not None else manifest_summary)
        ),
        "compile_s": compile_s,
        "compile_dry_run": pre_main.compile_dry_run,
        "warmup_mode": pre_main.warmup_mode,
        "warmup_steps_counted": pre_main.warmup_steps_counted,
        "samples_consumed_pre_main": pre_main.samples_consumed_pre_main,
        "pre_main_batches_changed_weights": pre_main.pre_main_batches_changed_weights,
        "summary": summarize_steps(stats, args.batch),
        "validation": {
            "every": args.validation_every,
            "source": asdict(validation_source.info),
            "history": eval_stats,
            "final": final_validation,
        },
        "memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "step_stats": [asdict(s) for s in stats],
        "real_shard_training": real_dataset is not None,
        "raw_mjai_training": raw_stream is not None or raw_pinned is not None,
        "raw_mjai_pinned_pyo3": raw_pinned is not None,
        "raw_mjai_transport": args.raw_mjai_transport,
        "raw_mjai_progress": raw_stream_local_progress,
        "raw_mjai_resume_total_progress": raw_resume_total_progress,
        "raw_mjai_bridge_stats": None if raw_pinned is None else asdict(raw_pinned.bridge_stats()),
        "raw_mjai_queue_stats": None if raw_pinned is None else asdict(raw_pinned.queue_stats()),
        "checkpoint_path": None,
        "resumed_step": None if resume_state is None else resume_state.global_step,
        "resumed_samples_seen": None if resume_state is None else resume_state.samples_seen,
        "global_step": final_global_step,
        "samples_seen": final_samples_seen,
        "ema": {
            "enabled": ema_tracker is not None,
            "device": None if ema_tracker is None else ema_tracker.device_name,
            "update_count": 0 if ema_tracker is None else ema_tracker.update_count,
            "last_update_step": 0 if ema_tracker is None else ema_tracker.last_update_step,
            "last_update_ms": math.nan if ema_tracker is None else ema_tracker.last_update_ms,
        },
    }
    global_step = final_global_step
    samples_seen = final_samples_seen
    latest_checkpoint, step_checkpoint = checkpoint_paths(args, global_step)
    if latest_checkpoint is not None:
        for checkpoint_path in (latest_checkpoint, step_checkpoint):
            if checkpoint_path is None:
                continue
            atomic_save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                optimizer_config=optimizer_config,
                runtime_config=runtime_config,
                loss_weights=weights,
                manifest_path=target_manifest_path,
                global_step=global_step,
                samples_seen=samples_seen,
                raw_mjai_progress=raw_mjai_progress_dict(apply_progress_offsets(raw_progress, raw_mjai_offsets)),
                ema_tracker=ema_tracker,
                ema_config=ema_config,
                target_contract=target_contract,
            )
        result["checkpoint_path"] = str(latest_checkpoint)
        events.write(
            "checkpoint_saved",
            {
                "path": str(latest_checkpoint),
                "step_path": None if step_checkpoint is None else str(step_checkpoint),
                "global_step": global_step,
                "samples_seen": samples_seen,
            },
        )
        scalar_writer.add_scalar("checkpoint/saved", 1.0, global_step)
        scalar_writer.add_scalar("checkpoint/samples_seen", float(samples_seen), global_step)
    validation_source.close()
    if raw_stream is not None:
        raw_stream.close()
    if raw_pinned is not None:
        raw_pinned.close()
    log_final_scalars(scalar_writer, result, global_step)
    scalar_writer.flush()
    events.write(
        "run_complete",
        {"global_step": result["global_step"], "samples_seen": result["samples_seen"], "result_path": str(args.out)},
    )
    events.close()
    train_log.close()
    scalar_writer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(result, indent=2))
    return 0
