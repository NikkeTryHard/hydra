from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

OBJECTIVE = "ppo_control"
RANK_UTILITY = "U_A"


@dataclass(frozen=True)
class PpoControlConfig:
    init_checkpoint: Path
    output_dir: Path
    steps: int | None
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
    lr_warmup_samples: int
    lr_decay_samples: int | None
    grad_clip_norm: float | None
    microbatch_size: int
    epochs: int
    target_kl: float | None
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
    rollout_inference: str
    ppo_pipeline_depth: int
    rollout_device: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production T1 PPO-control self-play training.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--run-forever", action="store_true")
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
    parser.add_argument("--lr-warmup-samples", type=int, default=1_000_000)
    parser.add_argument("--lr-decay-samples", type=int)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--microbatch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--target-kl", type=float)
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
    parser.add_argument("--rollout-inference", choices=("torch-callback", "rust-ort"), default="torch-callback")
    parser.add_argument("--ppo-pipeline-depth", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ppo-rollout-device")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> PpoControlConfig:
    if args.run_forever:
        args.steps = None
    elif args.steps is None or args.steps < 1:
        raise ValueError("--steps must be >= 1 unless --run-forever is set")
    if args.games_per_update < 1:
        raise ValueError("--games-per-update must be >= 1")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("--temperature must be finite and > 0")
    if args.arena_batch_decisions < 1:
        raise ValueError("--arena-batch-decisions must be >= 1")
    if args.lr <= 0.0 or args.min_lr < 0.0 or args.min_lr > args.lr:
        raise ValueError("invalid learning-rate bounds")
    if args.lr_warmup_samples < 0:
        raise ValueError("--lr-warmup-samples must be >= 0")
    if args.lr_decay_samples is not None and args.lr_decay_samples < 1:
        raise ValueError("--lr-decay-samples must be >= 1")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0.0:
        raise ValueError("--grad-clip-norm must be > 0")
    if args.microbatch_size < 1:
        raise ValueError("--microbatch-size must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.target_kl is not None and args.target_kl <= 0.0:
        raise ValueError("--target-kl must be > 0")
    if args.log_every_steps < 1 or args.checkpoint_every_steps < 1:
        raise ValueError("log/checkpoint cadence must be >= 1")
    if args.backbone_profile != "conv2d_local3":
        raise ValueError("T1 PPO native rollout/export requires backbone_profile=conv2d_local3")
    if args.ppo_pipeline_depth not in (0, 1):
        raise ValueError("--ppo-pipeline-depth must be 0 or 1")
    _validate_device(args.device, "--device")
    if args.ppo_rollout_device is not None:
        _validate_device(args.ppo_rollout_device, "--ppo-rollout-device")
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
        lr_warmup_samples=args.lr_warmup_samples,
        lr_decay_samples=args.lr_decay_samples,
        grad_clip_norm=args.grad_clip_norm,
        microbatch_size=args.microbatch_size,
        epochs=args.epochs,
        target_kl=args.target_kl,
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
        rollout_inference=args.rollout_inference,
        ppo_pipeline_depth=args.ppo_pipeline_depth,
        rollout_device=args.ppo_rollout_device,
    )


def _validate_device(device: str, label: str) -> None:
    try:
        parsed = torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"{label} must be cpu, cuda, or cuda:N") from exc
    if parsed.type == "cpu":
        if parsed.index is not None:
            raise ValueError(f"{label} must be cpu, cuda, or cuda:N")
        return
    if parsed.type != "cuda":
        raise ValueError(f"{label} must be cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise ValueError(f"{label} requested CUDA but CUDA is unavailable")
    index = 0 if parsed.index is None else parsed.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"{label} requested unavailable CUDA device {device}")


def _json_config(config: PpoControlConfig) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _config_digest(config: PpoControlConfig) -> str:
    payload = _json_config(config)
    payload["resume"] = None
    return _payload_digest(payload)


def _payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_RESUME_RUN_LOCAL_DIGEST_FIELDS = ("output_dir", "tensorboard_dir", "steps")


def _legacy_resume_run_local_payloads(payload: dict[str, object]) -> list[dict[str, object]]:
    init_checkpoint = payload.get("init_checkpoint")
    if not isinstance(init_checkpoint, str):
        return []
    checkpoint_suffix = "/logs/checkpoints/best.pt"
    if not init_checkpoint.endswith(checkpoint_suffix):
        return []
    run_dir = init_checkpoint[: -len(checkpoint_suffix)] + "/stages/T1_ppo_control/runs/latest_run"
    variant = dict(payload)
    variant["output_dir"] = run_dir
    variant["tensorboard_dir"] = run_dir + "/tensorboard"
    variant["steps"] = None
    return [variant]


def _add_resume_config_digest_variants(
    digests: set[str],
    payload: dict[str, object],
    *,
    omit_lr_decay_samples: bool,
    omit_legacy_rollout_fields: bool,
    omit_run_local_fields: bool,
) -> None:
    variant = dict(payload)
    if omit_lr_decay_samples:
        variant["lr_decay_samples"] = None
    if omit_legacy_rollout_fields:
        del variant["ppo_pipeline_depth"]
        del variant["rollout_device"]
    if omit_run_local_fields:
        for field in _RESUME_RUN_LOCAL_DIGEST_FIELDS:
            del variant[field]
    digests.add(_payload_digest(variant))


def _compatible_resume_config_digests(config: PpoControlConfig) -> set[str]:
    payload = _json_config(config)
    payload["resume"] = None
    digests: set[str] = set()
    payloads = [payload]
    payloads.extend(_legacy_resume_run_local_payloads(payload))
    can_omit_lr_decay_samples = config.lr_decay_samples is not None
    can_omit_legacy_rollout_fields = config.ppo_pipeline_depth == 0 and config.rollout_device is None
    for resume_payload in payloads:
        digests.add(_payload_digest(resume_payload))
        for omit_run_local_fields in (False, True):
            if omit_run_local_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=False,
                    omit_legacy_rollout_fields=False,
                    omit_run_local_fields=True,
                )
            if can_omit_lr_decay_samples:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=True,
                    omit_legacy_rollout_fields=False,
                    omit_run_local_fields=omit_run_local_fields,
                )
            if can_omit_legacy_rollout_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=False,
                    omit_legacy_rollout_fields=True,
                    omit_run_local_fields=omit_run_local_fields,
                )
            if can_omit_lr_decay_samples and can_omit_legacy_rollout_fields:
                _add_resume_config_digest_variants(
                    digests,
                    resume_payload,
                    omit_lr_decay_samples=True,
                    omit_legacy_rollout_fields=True,
                    omit_run_local_fields=omit_run_local_fields,
                )
    return digests
