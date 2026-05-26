from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

OBJECTIVE = "ppo_control"
RANK_UTILITY = "U_A"


@dataclass(frozen=True)
class PpoControlConfig:
    init_checkpoint: Path
    output_dir: Path
    steps: int
    games_per_update: int
    seed: int
    device: str
    temperature: float
    arena_batch_decisions: int
    arena_threads: int
    extension_path: Path | None
    hidden: int
    blocks: int
    bottleneck: int
    residual_profile: str
    backbone_profile: str
    conv_memory_format: str
    lr: float
    min_lr: float
    lr_warmup_steps: int
    grad_clip_norm: float | None
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    adamw_fused: str
    adamw_foreach: str
    bc_kl_reverse_coef: float
    entropy_alpha: float
    entropy_beta: float
    entropy_alpha_max: float
    log_every_steps: int
    checkpoint_every_steps: int
    keep_step_checkpoints: bool
    resume: Path | None
    tensorboard_dir: Path | None
    quiet: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production T1 PPO-control self-play training.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output dir")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--games-per-update", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--arena-batch-decisions", type=int, default=1024)
    parser.add_argument("--arena-threads", type=int, default=0)
    parser.add_argument("--extension-path", type=Path)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--bottleneck", type=int, required=True)
    parser.add_argument("--residual-profile", required=True)
    parser.add_argument("--backbone-profile", required=True)
    parser.add_argument("--conv-memory-format", required=True)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--adamw-fused", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--adamw-foreach", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--bc-kl-reverse-coef", type=float, default=0.01)
    parser.add_argument("--entropy-alpha", type=float, default=1.0e-3)
    parser.add_argument("--entropy-beta", type=float, default=1.0e-2)
    parser.add_argument("--entropy-alpha-max", type=float, default=0.05)
    parser.add_argument("--log-every-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> PpoControlConfig:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.games_per_update < 1:
        raise ValueError("--games-per-update must be >= 1")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("--temperature must be finite and > 0")
    if args.arena_batch_decisions < 1:
        raise ValueError("--arena-batch-decisions must be >= 1")
    if args.lr <= 0.0 or args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("invalid learning-rate bounds")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be >= 0")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError("--grad-clip-norm must be > 0")
    if args.log_every_steps < 1 or args.checkpoint_every_steps < 1:
        raise ValueError("log/checkpoint cadence must be >= 1")
    if args.backbone_profile != "conv2d_local3":
        raise ValueError("T1 PPO native rollout/export requires backbone_profile=conv2d_local3")
    if not args.init_checkpoint.is_file():
        raise ValueError(f"--init-checkpoint does not exist: {args.init_checkpoint}")
    return PpoControlConfig(
        init_checkpoint=args.init_checkpoint,
        output_dir=args.out,
        steps=args.steps,
        games_per_update=args.games_per_update,
        seed=args.seed,
        device=args.device,
        temperature=args.temperature,
        arena_batch_decisions=args.arena_batch_decisions,
        arena_threads=args.arena_threads,
        extension_path=args.extension_path,
        hidden=args.hidden,
        blocks=args.blocks,
        bottleneck=args.bottleneck,
        residual_profile=args.residual_profile,
        backbone_profile=args.backbone_profile,
        conv_memory_format=args.conv_memory_format,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        adamw_fused=args.adamw_fused,
        adamw_foreach=args.adamw_foreach,
        bc_kl_reverse_coef=args.bc_kl_reverse_coef,
        entropy_alpha=args.entropy_alpha,
        entropy_beta=args.entropy_beta,
        entropy_alpha_max=args.entropy_alpha_max,
        log_every_steps=args.log_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        keep_step_checkpoints=args.keep_step_checkpoints,
        resume=args.resume,
        tensorboard_dir=args.tensorboard_dir,
        quiet=args.quiet,
    )


def _json_config(config: PpoControlConfig) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _config_digest(config: PpoControlConfig) -> str:
    payload = _json_config(config)
    payload["resume"] = None
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
