from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import override

import torch
import torch.nn as nn

from hydra_learner.losses import (
    BaseTargets,
    LossWeights,
    base_loss,
    bce_logits_mean,
    danger_focal_bce,
    loss_breakdown_dict,
    masked_policy_ce_indices,
    opp_next_ce,
    oracle_critic_loss,
    safety_residual_loss,
    soft_ce,
    target_coverage_dict,
    value_mse,
)
from hydra_learner.metrics import EvalStats, StepStats


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
            return masked_policy_ce_indices(outputs.policy_logits, policy_target, legal_mask).mean()
        l_policy = masked_policy_ce_indices(outputs.policy_logits, policy_target, legal_mask).mean()
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
        raise ValueError("compiled loss targets missing oracle_target")
    oracle_target_mask = targets.oracle_target_mask
    if oracle_target_mask is None:
        raise ValueError("compiled loss targets missing oracle_target_mask")
    safety_target = targets.safety_target
    if safety_target is None:
        raise ValueError("compiled loss targets missing safety_target")
    safety_mask = targets.safety_mask
    if safety_mask is None:
        raise ValueError("compiled loss targets missing safety_mask")
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


def clone_state_for_restore(state: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(state)


def restore_train_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_state: dict[str, object],
    optimizer_state: dict[str, object],
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def run_step(
    loss_step: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    loss_mode: str,
    microbatch: int,
    autocast: bool,
    timed: bool,
    grad_clip_norm: float | None = None,
    collect_diagnostics: bool = False,
) -> StepStats:
    logical = obs.shape[0]
    if getattr(loss_step, "_hydra_compiled", False):
        torch.compiler.cudagraph_mark_step_begin()
    optimizer.zero_grad(set_to_none=True)
    step_start = torch.cuda.Event(enable_timing=True) if obs.device.type == "cuda" else None
    step_end = torch.cuda.Event(enable_timing=True) if obs.device.type == "cuda" else None
    step_start_wall = time.perf_counter()
    if step_start is not None:
        step_start.record()
    fwd_ms = 0.0
    bwd_ms = 0.0
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    logical_loss = obs.new_zeros(())
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
        else:
            loss = fwd_loss()
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite BC loss: {loss_value}")
        if timed:
            ms, _ = time_cuda(loss.backward)
            bwd_ms += ms
        else:
            loss.backward()
        logical_loss = logical_loss + loss.detach()

    grad_norm = math.nan
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(loss_step.parameters(), grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    if timed:
        opt_ms, _ = time_cuda(optimizer.step)
    else:
        optimizer.step()
        opt_ms = 0.0
    if step_start is not None and step_end is not None:
        step_end.record()
        step_ms = cuda_event_elapsed(step_start, step_end)
    else:
        step_ms = (time.perf_counter() - step_start_wall) * 1000.0
    if collect_diagnostics:
        with torch.inference_mode(), amp_ctx:
            outputs = model(obs)
            breakdown = base_loss(outputs, targets, weights)
        head_losses = loss_breakdown_dict(breakdown, weights, loss_mode)
        target_coverage = target_coverage_dict(targets, weights, loss_mode)
    else:
        head_losses: dict[str, float] = {}
        target_coverage: dict[str, dict[str, float | str]] = {}
    loss_value = float(logical_loss.detach())
    stat = StepStats(
        step_ms=step_ms,
        fwd_loss_ms=fwd_ms,
        backward_ms=bwd_ms,
        optimizer_ms=opt_ms,
        loss=loss_value,
        head_losses=head_losses,
        target_coverage=target_coverage,
        grad_norm=grad_norm,
    )
    stat.train_gpu_ms = step_ms
    return stat


def run_non_mutating_train_step(
    loss_step: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    loss_mode: str,
    microbatch: int,
    autocast: bool,
    timed: bool,
    grad_clip_norm: float | None = None,
) -> StepStats:
    model_state = clone_state_for_restore(model.state_dict())
    optimizer_state = clone_state_for_restore(optimizer.state_dict())
    try:
        return run_step(
            loss_step, model, optimizer, obs, targets, weights, loss_mode, microbatch, autocast, timed, grad_clip_norm
        )
    finally:
        restore_train_state(model, optimizer, model_state=model_state, optimizer_state=optimizer_state)


def evaluate_batch(
    model: nn.Module,
    obs: torch.Tensor,
    targets: BaseTargets,
    weights: LossWeights,
    autocast: bool,
) -> EvalStats:
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    with torch.inference_mode(), amp_ctx:
        outputs = model(obs)
        breakdown = base_loss(outputs, targets, weights)
        masked_logits = outputs.policy_logits.masked_fill(~targets.legal_mask.to(dtype=torch.bool), -1.0e9)
        pred = masked_logits.argmax(dim=1)
        target = targets.policy_target.to(dtype=torch.int64)
        accuracy = (pred == target).to(dtype=torch.float32).mean()
        topk = masked_logits.topk(k=min(5, masked_logits.shape[1]), dim=1).indices
        top3_accuracy = (topk[:, : min(3, topk.shape[1])] == target[:, None]).any(dim=1).to(dtype=torch.float32).mean()
        top5_accuracy = (topk == target[:, None]).any(dim=1).to(dtype=torch.float32).mean()
        probs = torch.softmax(masked_logits, dim=1)
        target_probs = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1.0e-12)
        confidence = probs.max(dim=1).values
        correct = (pred == target).to(dtype=torch.float32)
        ece = obs.new_zeros(())
        for bucket in range(10):
            lower = bucket / 10.0
            upper = (bucket + 1) / 10.0
            if bucket == 9:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)
            if mask.any():
                ece = ece + mask.to(dtype=torch.float32).mean() * (confidence[mask].mean() - correct[mask].mean()).abs()
    total = float(breakdown.total.detach())
    if not math.isfinite(total):
        raise RuntimeError(f"non-finite validation BC loss: {total}")
    return EvalStats(
        loss=total,
        policy=float(breakdown.policy.detach()),
        value=float(breakdown.value.detach()),
        grp=float(breakdown.grp.detach()),
        tenpai=float(breakdown.tenpai.detach()),
        danger=float(breakdown.danger.detach()),
        opp_next=float(breakdown.opp_next.detach()),
        score_pdf=float(breakdown.score_pdf.detach()),
        score_cdf=float(breakdown.score_cdf.detach()),
        oracle_critic=float(breakdown.oracle_critic.detach()),
        safety_residual=float(breakdown.safety_residual.detach()),
        target_coverage=target_coverage_dict(targets, weights, "full_base"),
        policy_accuracy=float(accuracy.detach()),
        policy_top3_accuracy=float(top3_accuracy.detach()),
        policy_top5_accuracy=float(top5_accuracy.detach()),
        policy_nll=float((-target_probs.log()).mean().detach()),
        policy_confidence=float(confidence.mean().detach()),
        policy_ece=float(ece.detach()),
        samples=obs.shape[0],
    )


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
