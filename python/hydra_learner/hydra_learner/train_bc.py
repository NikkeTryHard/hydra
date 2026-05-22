#!/usr/bin/env python3
"""Experimental Hydra PyTorch base-head BC throughput oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_learner.checkpoint import (
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    load_checkpoint,
    save_checkpoint,
)
from hydra_learner.losses import (
    BaseTargets,
    LossWeights,
    bce_logits_mean,
    danger_focal_bce,
    masked_policy_ce,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    soft_ce,
    value_mse,
)
from hydra_learner.metrics import StepStats, summarize_steps
from hydra_learner.model import ACTION_SPACE, DEFAULT_BLOCKS, DEFAULT_HIDDEN, DEFAULT_SE_BOTTLENECK, HydraPolicyNet
from hydra_learner.shards import BcShardDataset, ManifestSummary, PolicyBatch, validate_manifest

VARIANTS = ("eager_fp32", "eager_bf16", "compile_default", "compile_reduce_overhead", "compile_max_autotune")
LOSS_MODES = ("policy_only", "full_base")
COMPILED_LOSS_MODES = ("policy_only", "full_base")


def make_synthetic_batch(
    batch: int, actions: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(0xC0FFEE)
    obs = torch.randn(batch, 192, 34, device=device, generator=gen)
    legal = torch.rand(batch, actions, device=device, generator=gen) > 0.20
    labels = torch.randint(actions, (batch,), device=device, generator=gen)
    legal[torch.arange(batch, device=device), labels] = True
    return obs, legal, labels


def synthetic_targets(obs: torch.Tensor, legal: torch.Tensor, labels: torch.Tensor) -> BaseTargets:
    batch = obs.shape[0]
    device = obs.device
    policy = F.one_hot(labels, num_classes=ACTION_SPACE).to(dtype=torch.float32)
    grp = torch.zeros(batch, 24, device=device)
    grp[:, 0] = 1.0
    tenpai = torch.zeros(batch, 3, device=device)
    danger = torch.zeros(batch, 3, 34, device=device)
    danger_mask = torch.ones(batch, 3, 34, device=device)
    opp_next = torch.zeros(batch, 3, 34, device=device)
    opp_next[:, :, 0] = 1.0
    score_pdf = torch.zeros(batch, 64, device=device)
    score_pdf[:, 0] = 1.0
    score_cdf = torch.ones(batch, 64, device=device)
    return BaseTargets(
        policy_target=policy,
        legal_mask=legal,
        value_target=torch.zeros(batch, device=device),
        grp_target=grp,
        tenpai_target=tenpai,
        danger_target=danger,
        danger_mask=danger_mask,
        opp_next_target=opp_next,
        score_pdf_target=score_pdf,
        score_cdf_target=score_cdf,
        oracle_target=torch.zeros(batch, 4, device=device),
        oracle_target_mask=torch.ones(batch, device=device),
        safety_target=None,
        safety_mask=None,
    )


def tensors_from_real_batch(
    dataset: BcShardDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets]:
    batch = dataset.next_batch()
    obs = torch.from_numpy(batch.obs).to(device=device, non_blocking=True)
    legal = torch.from_numpy(batch.legal_mask).to(device=device, non_blocking=True)
    labels = torch.from_numpy(batch.actions).to(device=device, non_blocking=True)
    targets = targets_from_policy_batch(batch, device)
    if obs.shape != (dataset.batch_size, 192, 34) or obs.dtype != torch.float32:
        raise ValueError(f"real shard obs contract mismatch: shape={tuple(obs.shape)} dtype={obs.dtype}")
    if legal.shape != (dataset.batch_size, ACTION_SPACE) or legal.dtype != torch.bool:
        raise ValueError(f"real shard legal-mask contract mismatch: shape={tuple(legal.shape)} dtype={legal.dtype}")
    if labels.shape != (dataset.batch_size,) or labels.dtype != torch.int64:
        raise ValueError(f"real shard action contract mismatch: shape={tuple(labels.shape)} dtype={labels.dtype}")
    return obs, legal, labels, targets


def targets_from_policy_batch(batch: PolicyBatch, device: torch.device) -> BaseTargets:
    return BaseTargets(
        policy_target=F.one_hot(torch.from_numpy(batch.actions).to(device=device), num_classes=ACTION_SPACE).to(
            dtype=torch.float32
        ),
        legal_mask=torch.from_numpy(batch.legal_mask).to(device=device),
        value_target=torch.from_numpy(batch.value_target).to(device=device),
        grp_target=torch.from_numpy(batch.grp_target).to(device=device),
        tenpai_target=torch.from_numpy(batch.tenpai).to(device=device),
        danger_target=torch.from_numpy(batch.danger.reshape(batch.danger.shape[0], 3, 34)).to(device=device),
        danger_mask=torch.from_numpy(batch.danger_mask.reshape(batch.danger_mask.shape[0], 3, 34)).to(device=device),
        opp_next_target=torch.from_numpy(batch.opp_next.reshape(batch.opp_next.shape[0], 3, 34)).to(device=device),
        score_pdf_target=torch.from_numpy(batch.score_pdf).to(device=device),
        score_cdf_target=torch.from_numpy(batch.score_cdf).to(device=device),
        oracle_target=torch.from_numpy(batch.oracle_target).to(device=device),
        oracle_target_mask=torch.from_numpy(batch.oracle_target_mask).to(device=device),
        safety_target=None if batch.safety_target is None else torch.from_numpy(batch.safety_target).to(device=device),
        safety_mask=None if batch.safety_mask is None else torch.from_numpy(batch.safety_mask).to(device=device),
    )


def cuda_event_elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return start.elapsed_time(end)


def time_cuda(fn: Callable[[], torch.Tensor | None]) -> tuple[float, torch.Tensor | None]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    return cuda_event_elapsed(start, end), out


class HydraCompiledLossStep(nn.Module):
    def __init__(self, model: nn.Module, loss_mode: str, weights: LossWeights) -> None:
        super().__init__()
        self.model = model
        self.loss_mode = loss_mode
        self.weights = weights

    @override
    def forward(
        self,
        obs: torch.Tensor,
        policy_target: torch.Tensor,
        legal_mask: torch.Tensor,
        value_target: torch.Tensor,
        grp_target: torch.Tensor,
        tenpai_target: torch.Tensor,
        danger_target: torch.Tensor,
        danger_mask: torch.Tensor,
        opp_next_target: torch.Tensor,
        score_pdf_target: torch.Tensor,
        score_cdf_target: torch.Tensor,
        oracle_target: torch.Tensor,
        oracle_target_mask: torch.Tensor,
        safety_target: torch.Tensor,
        safety_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(obs)
        if self.loss_mode == "policy_only":
            return masked_policy_ce(outputs.policy_logits, policy_target, legal_mask).mean()
        l_policy = masked_policy_ce(outputs.policy_logits, policy_target, legal_mask).mean()
        l_value = value_mse(outputs.value, value_target).mean()
        l_grp = soft_ce(outputs.grp, grp_target).mean()
        l_tenpai = bce_logits_mean(outputs.opp_tenpai, tenpai_target, dim=1).mean()
        l_danger = danger_focal_bce(outputs.danger, danger_target, danger_mask).mean()
        l_opp = opp_next_ce(outputs.opp_next_discard, opp_next_target).mean()
        l_pdf = soft_ce(outputs.score_pdf, score_pdf_target).mean()
        l_cdf = bce_logits_mean(outputs.score_cdf, score_cdf_target, dim=1).mean()
        total = (
            l_policy * self.weights.policy
            + l_value * self.weights.value
            + l_grp * self.weights.grp
            + l_tenpai * self.weights.tenpai
            + l_danger * self.weights.danger
            + l_opp * self.weights.opp_next
            + l_pdf * self.weights.score
            + l_cdf * self.weights.score
        )
        if self.weights.oracle_critic > 0.0:
            total = (
                total
                + oracle_critic_loss(outputs.oracle_critic, oracle_target, oracle_target_mask)
                * self.weights.oracle_critic
            )
        if self.weights.safety_residual > 0.0:
            total = (
                total
                + safety_residual_loss(outputs.safety_residual, safety_target, safety_mask)
                * self.weights.safety_residual
            )
        return total


def loss_step_args(obs: torch.Tensor, targets: BaseTargets, start: int, end: int) -> tuple[torch.Tensor, ...]:
    oracle_target = targets.oracle_target
    if oracle_target is None:
        oracle_target = obs.new_zeros((obs.shape[0], 4))
    oracle_target_mask = targets.oracle_target_mask
    if oracle_target_mask is None:
        oracle_target_mask = obs.new_zeros((obs.shape[0],))
    safety_target = targets.safety_target
    if safety_target is None:
        safety_target = obs.new_zeros((obs.shape[0], ACTION_SPACE))
    safety_mask = targets.safety_mask
    if safety_mask is None:
        safety_mask = obs.new_zeros((obs.shape[0], ACTION_SPACE))
    return (
        obs[start:end],
        targets.policy_target[start:end],
        targets.legal_mask[start:end],
        targets.value_target[start:end],
        targets.grp_target[start:end],
        targets.tenpai_target[start:end],
        targets.danger_target[start:end],
        targets.danger_mask[start:end],
        targets.opp_next_target[start:end],
        targets.score_pdf_target[start:end],
        targets.score_cdf_target[start:end],
        oracle_target[start:end],
        oracle_target_mask[start:end],
        safety_target[start:end],
        safety_mask[start:end],
    )


def run_step(
    loss_step: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    microbatch: int,
    autocast: bool,
    timed: bool,
) -> StepStats:
    logical = obs.shape[0]
    if getattr(loss_step, "_hydra_compiled", False):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    step_start.record()
    fwd_ms = 0.0
    bwd_ms = 0.0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    for start_idx in range(0, logical, microbatch):
        end_idx = min(start_idx + microbatch, logical)
        scale = (end_idx - start_idx) / logical

        def fwd_loss() -> torch.Tensor:
            with amp_ctx:
                loss = loss_step(*loss_step_args(obs, targets, start_idx, end_idx))
            return loss * scale

        if timed:
            ms, loss = time_cuda(fwd_loss)
            fwd_ms += ms
            assert loss is not None
            ms, _ = time_cuda(loss.backward)
            bwd_ms += ms
        else:
            loss = fwd_loss()
            loss.backward()

    if timed:
        opt_ms, _ = time_cuda(optimizer.step)
    else:
        optimizer.step()
        opt_ms = 0.0
    step_end.record()
    step_ms = cuda_event_elapsed(step_start, step_end)
    return StepStats(step_ms=step_ms, fwd_loss_ms=fwd_ms, backward_ms=bwd_ms, optimizer_ms=opt_ms)


def slice_targets(targets: BaseTargets, start: int, end: int) -> BaseTargets:
    return BaseTargets(
        policy_target=targets.policy_target[start:end],
        legal_mask=targets.legal_mask[start:end],
        value_target=targets.value_target[start:end],
        grp_target=targets.grp_target[start:end],
        tenpai_target=targets.tenpai_target[start:end],
        danger_target=targets.danger_target[start:end],
        danger_mask=targets.danger_mask[start:end],
        opp_next_target=targets.opp_next_target[start:end],
        score_pdf_target=targets.score_pdf_target[start:end],
        score_cdf_target=targets.score_cdf_target[start:end],
        oracle_target=None if targets.oracle_target is None else targets.oracle_target[start:end],
        oracle_target_mask=None if targets.oracle_target_mask is None else targets.oracle_target_mask[start:end],
        safety_target=None if targets.safety_target is None else targets.safety_target[start:end],
        safety_mask=None if targets.safety_mask is None else targets.safety_mask[start:end],
    )


def torch_env() -> dict[str, object]:
    env: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "optimizer": "AdamW",
    }
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name()
        env["device_capability"] = torch.cuda.get_device_capability()
    return env


def json_manifest_summary(summary: ManifestSummary | None) -> dict[str, object] | None:
    if summary is None:
        return None
    data = asdict(summary)
    data["path"] = str(data["path"])
    return data


def json_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "variant": args.variant,
        "loss_mode": args.loss_mode,
        "batch": args.batch,
        "microbatch": args.microbatch,
        "hidden": args.hidden,
        "blocks": args.blocks,
        "bottleneck": args.bottleneck,
        "actions": args.actions,
        "warmup": args.warmup,
        "steps": args.steps,
        "profile": args.profile,
        "profile_coarse": args.profile_coarse,
        "manifest": str(args.manifest) if args.manifest else None,
        "check_shard_files": args.check_shard_files,
        "out": str(args.out),
        "w_oracle_critic": args.w_oracle_critic,
        "w_safety_residual": args.w_safety_residual,
        "compile_fullgraph_check": args.compile_fullgraph_check,
        "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
        "resume": str(args.resume) if args.resume else None,
        "checkpoint_every_steps": args.checkpoint_every_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="eager_bf16")
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="full_base")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--bottleneck", type=int, default=DEFAULT_SE_BOTTLENECK)
    parser.add_argument("--actions", type=int, default=ACTION_SPACE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--profile", action="store_true", help="emit NVTX ranges around measured steps")
    parser.add_argument(
        "--profile-coarse", action="store_true", help="measure whole-step time only to reduce profiler overhead"
    )
    parser.add_argument("--manifest", type=Path, help="train from a Hydra BC shard manifest instead of synthetic data")
    parser.add_argument(
        "--check-shard-files", action="store_true", help="also validate shard headers named by --manifest"
    )
    parser.add_argument("--w-oracle-critic", type=float, default=0.0)
    parser.add_argument("--w-safety-residual", type=float, default=0.0)
    parser.add_argument("--compile-fullgraph-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/home/cachybtw/tmp/hydra_py_oracle_result.json"))
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_summary = None
    real_dataset = None
    if args.manifest is not None:
        manifest_summary = validate_manifest(args.manifest, check_files=args.check_shard_files)
        real_dataset = BcShardDataset(args.manifest, batch_size=args.batch, split="train")

    if not torch.cuda.is_available():
        result = {"variant": args.variant, "env": torch_env(), "error": "CUDA unavailable"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    torch.manual_seed(0x51A7E)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    model = HydraPolicyNet(args.hidden, args.blocks, args.bottleneck, args.actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    if real_dataset is None:
        obs, legal, labels = make_synthetic_batch(args.batch, args.actions, device)
        targets = synthetic_targets(obs, legal, labels)
    else:
        obs, legal, labels, targets = tensors_from_real_batch(real_dataset, device)
    autocast = args.variant != "eager_fp32"
    weights = LossWeights(oracle_critic=args.w_oracle_critic, safety_residual=args.w_safety_residual)
    if args.loss_mode not in COMPILED_LOSS_MODES:
        raise ValueError(f"unsupported loss mode {args.loss_mode!r}")
    loss_step: nn.Module = HydraCompiledLossStep(model, args.loss_mode, weights)
    model_config = ModelConfig(hidden=args.hidden, blocks=args.blocks, bottleneck=args.bottleneck, actions=args.actions)
    optimizer_config = OptimizerConfig(name="AdamW", lr=3.0e-4)
    precision_mode = "fp32" if args.variant == "eager_fp32" else "bf16_autocast"
    runtime_config = RuntimeConfig(
        variant=args.variant,
        loss_mode=args.loss_mode,
        precision_mode=precision_mode,
        compile_fullgraph_check=args.compile_fullgraph_check,
    )
    resume_state = None
    if args.resume is not None:
        resume_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_optimizer_config=optimizer_config,
            expected_runtime_config=runtime_config,
            expected_loss_weights=weights,
            expected_manifest_path=args.manifest,
        )

    compile_error = None
    compile_s = 0.0
    if args.variant.startswith("compile_"):
        mode = None
        if args.variant == "compile_reduce_overhead":
            mode = "reduce-overhead"
        elif args.variant == "compile_max_autotune":
            mode = "max-autotune"
        try:
            t0 = time.perf_counter()
            loss_step = cast("nn.Module", torch.compile(loss_step, mode=mode, fullgraph=args.compile_fullgraph_check))
            setattr(loss_step, "_hydra_compiled", True)
            run_step(loss_step, optimizer, obs, targets, args.microbatch, autocast, False)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t0
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"

    env = torch_env()
    if compile_error is not None:
        result = {
            "variant": args.variant,
            "env": env,
            "compile_error": compile_error,
            "compile_s": compile_s,
            "fullgraph_check": args.compile_fullgraph_check,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        if real_dataset is not None:
            obs, legal, labels, targets = tensors_from_real_batch(real_dataset, device)
        run_step(loss_step, optimizer, obs, targets, args.microbatch, autocast, False)
    torch.cuda.synchronize()

    stats: list[StepStats] = []
    for i in range(args.steps):
        if args.profile:
            torch.cuda.nvtx.range_push(f"hydra_oracle_step_{i}")
        if real_dataset is not None:
            obs, legal, labels, targets = tensors_from_real_batch(real_dataset, device)
        stats.append(
            run_step(
                loss_step,
                optimizer,
                obs,
                targets,
                args.microbatch,
                autocast,
                timed=not args.profile_coarse,
            )
        )
        if args.profile:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    result = {
        "variant": args.variant,
        "env": env,
        "config": json_config(args),
        "manifest_summary": json_manifest_summary(manifest_summary),
        "compile_s": compile_s,
        "summary": summarize_steps(stats, args.batch),
        "memory": {
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "step_stats": [asdict(s) for s in stats],
        "real_shard_training": real_dataset is not None,
        "checkpoint_path": None,
        "resumed_step": None if resume_state is None else resume_state.global_step,
        "resumed_samples_seen": None if resume_state is None else resume_state.samples_seen,
        "global_step": (0 if resume_state is None else resume_state.global_step) + args.steps,
        "samples_seen": (0 if resume_state is None else resume_state.samples_seen) + args.steps * args.batch,
    }
    if args.checkpoint_out is not None:
        global_step = (0 if resume_state is None else resume_state.global_step) + args.steps
        samples_seen = (0 if resume_state is None else resume_state.samples_seen) + args.steps * args.batch
        save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            optimizer_config=optimizer_config,
            runtime_config=runtime_config,
            loss_weights=weights,
            manifest_path=args.manifest,
            global_step=global_step,
            samples_seen=samples_seen,
        )
        result["checkpoint_path"] = str(args.checkpoint_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
