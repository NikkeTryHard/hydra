from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra_learner.constants import (
    ADAMW_FLAG_MODES,
    LOSS_MODES,
    LR_SCHEDULES,
    PYTHON_VARIANT_DEFAULT,
    VALIDATION_SOURCE_MODES,
    VARIANTS,
)
from hydra_learner.data.raw_mjai import add_raw_mjai_args, validate_raw_mjai_source_args
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_DEFAULT,
    BACKBONE_PROFILES,
    CONV_MEMORY_FORMAT_DEFAULT,
    CONV_MEMORY_FORMATS,
    DEFAULT_BLOCKS,
    DEFAULT_HIDDEN,
    DEFAULT_SE_BOTTLENECK,
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILES,
)
from hydra_learner.ppo_control import main as ppo_control_main
from hydra_learner.training.loop import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default=PYTHON_VARIANT_DEFAULT)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="full_base")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--bottleneck", type=int, default=DEFAULT_SE_BOTTLENECK)
    parser.add_argument("--actions", type=int, default=ACTION_SPACE)
    parser.add_argument("--residual-profile", choices=RESIDUAL_PROFILES, default=RESIDUAL_PROFILE_DEFAULT)
    parser.add_argument("--backbone-profile", choices=BACKBONE_PROFILES, default=BACKBONE_PROFILE_DEFAULT)
    parser.add_argument("--conv-memory-format", choices=CONV_MEMORY_FORMATS, default=CONV_MEMORY_FORMAT_DEFAULT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--profile", action="store_true", help="emit NVTX ranges around measured steps")
    parser.add_argument(
        "--profile-coarse", action="store_true", help="measure whole-step time only to reduce profiler overhead"
    )
    parser.add_argument(
        "--torch-profiler-trace",
        type=Path,
        help="write a Chrome trace for a scheduled measured-step window",
    )
    parser.add_argument(
        "--torch-profiler-start-step",
        type=int,
        default=0,
        help="0-based measured step where torch profiler capture starts",
    )
    parser.add_argument(
        "--torch-profiler-stop-step",
        type=int,
        default=1,
        help="0-based measured step where torch profiler capture stops, exclusive",
    )
    parser.add_argument("--manifest", type=Path, help="train from a Hydra BC shard manifest instead of synthetic data")
    add_raw_mjai_args(parser)
    parser.add_argument(
        "--check-shard-files", action="store_true", help="also validate shard headers named by --manifest"
    )
    parser.add_argument("--w-oracle-critic", type=float, default=0.0)
    parser.add_argument("--w-safety-residual", type=float, default=0.0)
    parser.add_argument("--w-exit", type=float, default=0.0)
    parser.add_argument("--w-deltaq", type=float, default=0.0)
    parser.add_argument("--compile-fullgraph-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/home/cachybtw/tmp/hydra_py_bc_result.json"))
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="cosine")
    parser.add_argument("--schedule-total-steps", type=int)
    parser.add_argument("--schedule-target-games", type=int)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--adamw-foreach", choices=ADAMW_FLAG_MODES, default="auto")
    parser.add_argument("--adamw-fused", choices=ADAMW_FLAG_MODES, default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--tensorboard-url")
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--validation-max-samples", type=int)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--validation-source-mode", choices=VALIDATION_SOURCE_MODES, default="fixed")
    parser.add_argument("--ema-enabled", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--ema-update-every-steps", type=int, default=1)
    parser.add_argument("--ema-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--cuda-profiler-range", action="store_true", help="start CUDA profiler only around measured steps"
    )
    parser.add_argument("--quiet", action="store_true", help="write JSON result without printing it to stdout")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (("batch", args.batch), ("microbatch", args.microbatch))
    for name, value in positive_ints:
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be >= 1")
    non_negative_ints = (
        ("warmup", args.warmup),
        ("checkpoint_every_steps", args.checkpoint_every_steps),
        ("log_every_steps", args.log_every_steps),
        ("validation_steps", args.validation_steps),
        ("validation_every", args.validation_every),
        ("validation_max_samples", args.validation_max_samples if args.validation_max_samples is not None else 0),
        ("lr_warmup_steps", args.lr_warmup_steps),
        ("ema_start_step", args.ema_start_step),
    )
    for name, value in non_negative_ints:
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.schedule_total_steps is not None and args.schedule_total_steps < 1:
        raise ValueError("--schedule-total-steps must be >= 1")
    if args.schedule_target_games is not None and args.schedule_target_games < 1:
        raise ValueError("--schedule-target-games must be >= 1")
    if args.lr <= 0.0:
        raise ValueError("--lr must be > 0")
    if args.min_lr < 0.0:
        raise ValueError("--min-lr must be >= 0")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must be <= --lr")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError("--grad-clip-norm must be > 0")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema-decay must be in [0, 1)")
    if getattr(args, "w_exit", 0.0) < 0.0:
        raise ValueError("--w-exit must be >= 0")
    if getattr(args, "w_deltaq", 0.0) < 0.0:
        raise ValueError("--w-deltaq must be >= 0")
    if args.ema_update_every_steps < 1:
        raise ValueError("--ema-update-every-steps must be >= 1")
    if (
        args.lr_schedule == "cosine"
        and args.schedule_total_steps is None
        and args.schedule_target_games is None
        and args.steps is None
        and args.manifest is None
    ):
        raise ValueError(
            "--lr-schedule cosine requires --schedule-total-steps, --schedule-target-games, --steps, or --manifest"
        )
    if args.checkpoint_out is not None and args.checkpoint_dir is not None:
        raise ValueError("--checkpoint-out and --checkpoint-dir cannot be combined")
    if args.keep_step_checkpoints and args.checkpoint_dir is None:
        raise ValueError("--keep-step-checkpoints requires --checkpoint-dir")
    if args.microbatch > args.batch:
        raise ValueError("--microbatch must be <= --batch")
    if args.actions != ACTION_SPACE:
        raise ValueError(f"--actions must equal Hydra action space {ACTION_SPACE}")
    if args.torch_profiler_start_step < 0:
        raise ValueError("--torch-profiler-start-step must be >= 0")
    if args.torch_profiler_stop_step < 1:
        raise ValueError("--torch-profiler-stop-step must be >= 1")
    if args.torch_profiler_trace is not None:
        if args.torch_profiler_start_step >= args.torch_profiler_stop_step:
            raise ValueError("--torch-profiler-start-step must be < --torch-profiler-stop-step")
        if args.torch_profiler_stop_step > args.steps:
            raise ValueError("--torch-profiler-stop-step must be <= --steps")
    if args.full_epoch and args.manifest is not None:
        raise ValueError("--full-epoch is only supported with raw MJAI input")
    if args.full_epoch and args.raw_mjai_max_samples is not None:
        raise ValueError("--full-epoch cannot be combined with --raw-mjai-max-samples")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "ppo-control":
        return ppo_control_main(sys.argv[2:])
    args = parse_args()
    validate_args(args)
    validate_raw_mjai_source_args(args)
    return run_training(args)
