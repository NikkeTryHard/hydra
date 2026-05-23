from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from hydra_learner.losses import BaseTargets, LossWeights
from hydra_learner.shard_contracts import ACTION_SPACE, NUM_CHANNELS, TILE_WIDTH, PolicyBatch

if TYPE_CHECKING:
    from hydra_learner.raw_mjai import PinnedPolicyBatch
    from hydra_learner.raw_mjai_direct import RawMjaiDirectStream
    from hydra_learner.raw_mjai_pinned import RawMjaiPinnedStream
    from hydra_learner.shard_reader import BcShardDataset


@dataclass
class StagedTrainBatch:
    obs: torch.Tensor
    targets: BaseTargets
    input_timing: InputTiming
    pinned_batch: PinnedPolicyBatch | None = None


def _require_shape_dtype(value: object, shape: tuple[int, ...], dtype: object, name: str, context: str) -> None:
    actual_shape = tuple(getattr(value, "shape", ()))
    actual_dtype = getattr(value, "dtype", None)
    if actual_shape != shape or actual_dtype != dtype:
        raise ValueError(
            f"{context} {name} must have shape {shape} and dtype {dtype}, got shape {actual_shape} dtype {actual_dtype}"
        )


def validate_policy_batch_shapes(batch: PolicyBatch, *, context: str) -> None:
    """Validate host PolicyBatch tensor contracts before any device transfer."""
    rows = batch.obs.shape[0] if len(batch.obs.shape) >= 1 else -1
    _require_shape_dtype(batch.obs, (rows, NUM_CHANNELS, TILE_WIDTH), np.float32, "obs (192x34)", context)
    _require_shape_dtype(batch.actions, (rows,), np.int64, "actions", context)
    _require_shape_dtype(batch.legal_mask, (rows, ACTION_SPACE), np.bool_, "legal_mask (46)", context)
    for name, field, shapes in (
        ("value_target", "value_target", ((rows,), (rows, 1))),
        ("grp_target", "grp_target", ((rows, 24),)),
        ("tenpai", "tenpai", ((rows, 3),)),
        ("score_pdf", "score_pdf", ((rows, 64),)),
        ("score_cdf", "score_cdf", ((rows, 64),)),
    ):
        value = getattr(batch, field)
        if tuple(value.shape) not in shapes or value.dtype != np.float32:
            raise ValueError(
                f"{context} {name} must have shape {shapes} and dtype {np.float32}, "
                f"got shape {tuple(value.shape)} dtype {value.dtype}"
            )
    for name, field, shapes in (
        ("oracle_target", "oracle_target", ((rows, 4),)),
        ("oracle_target_mask", "oracle_target_mask", ((rows,), (rows, 4))),
        ("safety_target", "safety_target", ((rows, ACTION_SPACE),)),
        ("safety_mask", "safety_mask", ((rows, ACTION_SPACE),)),
    ):
        value = getattr(batch, field)
        if value is not None and (tuple(value.shape) not in shapes or value.dtype != np.float32):
            raise ValueError(
                f"{context} {name} must have shape {shapes} and dtype {np.float32}, "
                f"got shape {tuple(value.shape)} dtype {value.dtype}"
            )
    for name, shapes in (
        ("opp_next", ((rows, 3, TILE_WIDTH), (rows, 3 * TILE_WIDTH))),
        ("danger", ((rows, 3, TILE_WIDTH), (rows, 3 * TILE_WIDTH))),
        ("danger_mask", ((rows, 3, TILE_WIDTH), (rows, 3 * TILE_WIDTH))),
        ("tenpai", ((rows,), (rows, 3))),
    ):
        value = getattr(batch, name)
        if value is not None and (tuple(value.shape) not in shapes or value.dtype != np.float32):
            raise ValueError(
                f"{context} {name} must have shape {shapes} and dtype {np.float32}, "
                f"got shape {tuple(value.shape)} dtype {value.dtype}"
            )


def validate_base_targets(targets: BaseTargets, *, batch: int, device: torch.device | None, context: str) -> None:
    """Validate BaseTargets batch/device contracts at API boundaries, not per microbatch."""
    for name, tensor in vars(targets).items():
        if tensor is None:
            continue
        if tensor.shape[0] != batch:
            raise ValueError(f"{context} {name} batch dimension must be {batch}, got {tensor.shape[0]}")
        if device is not None and tensor.device != device:
            raise ValueError(f"{context} {name} must be on {device}, got {tensor.device}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{context} {name} contains non-finite values")


@dataclass(frozen=True)
class InputTiming:
    fetch_decode_ms: float = math.nan
    h2d_wall_ms: float = math.nan


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
        policy_target=labels,
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


def tensors_from_policy_batch(
    batch: PolicyBatch, device: torch.device, fetch_decode_ms: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    validate_policy_batch_shapes(batch, context="policy batch")
    h2d_started = time.perf_counter()
    obs = torch.from_numpy(batch.obs).to(device=device, non_blocking=True)
    legal = torch.from_numpy(batch.legal_mask).to(device=device, non_blocking=True)
    labels = torch.from_numpy(batch.actions).to(device=device, non_blocking=True)
    targets = targets_from_policy_batch(batch, device, labels, legal)
    h2d_wall_ms = (time.perf_counter() - h2d_started) * 1000.0
    if obs.shape[1:] != (192, 34) or obs.dtype != torch.float32:
        raise ValueError(f"real shard obs contract mismatch: shape={tuple(obs.shape)} dtype={obs.dtype}")
    if legal.shape != (obs.shape[0], ACTION_SPACE) or legal.dtype != torch.bool:
        raise ValueError(f"real shard legal-mask contract mismatch: shape={tuple(legal.shape)} dtype={legal.dtype}")
    if labels.shape != (obs.shape[0],) or labels.dtype != torch.int64:
        raise ValueError(f"real shard action contract mismatch: shape={tuple(labels.shape)} dtype={labels.dtype}")
    return (
        obs,
        legal,
        labels,
        targets,
        InputTiming(
            fetch_decode_ms=fetch_decode_ms,
            h2d_wall_ms=h2d_wall_ms,
        ),
    )


def tensors_from_pinned_policy_batch(
    batch: PinnedPolicyBatch, device: torch.device, fetch_decode_ms: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    if (
        batch.obs.shape != (batch.rows, NUM_CHANNELS, TILE_WIDTH)
        or batch.actions.shape != (batch.rows,)
        or batch.legal_mask.shape != (batch.rows, ACTION_SPACE)
    ):
        raise ValueError("pinned policy batch must have obs 192x34, actions N, legal_mask width 46 before device move")
    h2d_started = time.perf_counter()
    obs = batch.obs.to(device=device, non_blocking=True)
    legal = batch.legal_mask.to(device=device, non_blocking=True)
    labels = batch.actions.to(device=device, non_blocking=True)
    targets = BaseTargets(
        policy_target=labels,
        legal_mask=legal,
        value_target=batch.value_target.to(device=device, non_blocking=True),
        grp_target=batch.grp_target.to(device=device, non_blocking=True),
        tenpai_target=batch.tenpai.to(device=device, non_blocking=True),
        danger_target=batch.danger.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        danger_mask=batch.danger_mask.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        opp_next_target=batch.opp_next.reshape(obs.shape[0], 3, 34).to(device=device, non_blocking=True),
        score_pdf_target=batch.score_pdf.to(device=device, non_blocking=True),
        score_cdf_target=batch.score_cdf.to(device=device, non_blocking=True),
        oracle_target=batch.oracle_target.to(device=device, non_blocking=True),
        oracle_target_mask=batch.oracle_target_mask.to(device=device, non_blocking=True),
        safety_target=None,
        safety_mask=None,
    )
    h2d_wall_ms = (time.perf_counter() - h2d_started) * 1000.0
    if obs.shape[1:] != (192, 34) or obs.dtype != torch.float32:
        raise ValueError(f"pinned raw MJAI obs contract mismatch: shape={tuple(obs.shape)} dtype={obs.dtype}")
    if legal.shape != (obs.shape[0], ACTION_SPACE) or legal.dtype != torch.bool:
        raise ValueError(
            f"pinned raw MJAI legal-mask contract mismatch: shape={tuple(legal.shape)} dtype={legal.dtype}"
        )
    if labels.shape != (obs.shape[0],) or labels.dtype != torch.int64:
        raise ValueError(f"pinned raw MJAI action contract mismatch: shape={tuple(labels.shape)} dtype={labels.dtype}")
    return (
        obs,
        legal,
        labels,
        targets,
        InputTiming(fetch_decode_ms=fetch_decode_ms, h2d_wall_ms=h2d_wall_ms),
    )


def tensors_from_real_batch(
    dataset: BcShardDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BaseTargets, InputTiming]:
    batch = dataset.next_batch()
    return tensors_from_policy_batch(batch, device, dataset.last_fetch_decode_ms)


def stage_train_batch(
    obs: torch.Tensor,
    targets: BaseTargets,
    input_timing: InputTiming,
    weights: LossWeights,
    pinned_batch: PinnedPolicyBatch | None = None,
) -> StagedTrainBatch:
    return StagedTrainBatch(
        obs=obs,
        targets=targets_for_compiled_loss(targets, weights),
        input_timing=input_timing,
        pinned_batch=pinned_batch,
    )


def next_staged_train_batch(
    *,
    real_dataset: BcShardDataset | None,
    raw_stream: RawMjaiDirectStream | None,
    raw_pinned: RawMjaiPinnedStream | None,
    device: torch.device,
    weights: LossWeights,
) -> StagedTrainBatch:
    if raw_stream is not None:
        batch, fetch_ms = raw_stream.next_batch()
        obs, _legal, _labels, targets, input_timing = tensors_from_policy_batch(batch, device, fetch_ms)
        return stage_train_batch(obs, targets, input_timing, weights)
    if raw_pinned is not None:
        pinned_batch, fetch_ms = raw_pinned.next_batch()
        obs, _legal, _labels, targets, input_timing = tensors_from_pinned_policy_batch(pinned_batch, device, fetch_ms)
        return stage_train_batch(obs, targets, input_timing, weights, pinned_batch)
    if real_dataset is not None:
        obs, _legal, _labels, targets, input_timing = tensors_from_real_batch(real_dataset, device)
        return stage_train_batch(obs, targets, input_timing, weights)
    raise ValueError("internal data source selection failed")


def targets_from_policy_batch(
    batch: PolicyBatch, device: torch.device, labels: torch.Tensor | None = None, legal: torch.Tensor | None = None
) -> BaseTargets:
    if labels is None:
        labels = torch.from_numpy(batch.actions).to(device=device)
    if legal is None:
        legal = torch.from_numpy(batch.legal_mask).to(device=device)
    return BaseTargets(
        policy_target=labels,
        legal_mask=legal,
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


def targets_for_compiled_loss(targets: BaseTargets, weights: LossWeights) -> BaseTargets:
    batch = targets.policy_target.shape[0]
    if weights.oracle_critic > 0.0:
        oracle_target = targets.oracle_target
        oracle_target_mask = targets.oracle_target_mask
        if oracle_target is None or oracle_target_mask is None:
            raise ValueError("oracle targets are required when oracle_critic loss weight is positive")
    else:
        oracle_target = targets.oracle_target
        if oracle_target is None:
            oracle_target = targets.value_target.new_zeros((batch, 4))
        oracle_target_mask = targets.oracle_target_mask
        if oracle_target_mask is None:
            oracle_target_mask = targets.value_target.new_zeros((batch,))
    if weights.safety_residual > 0.0:
        safety_target = targets.safety_target
        safety_mask = targets.safety_mask
        if safety_target is None or safety_mask is None:
            raise ValueError("safety targets are required when safety_residual loss weight is positive")
    else:
        safety_target = targets.safety_target
        if safety_target is None:
            safety_target = targets.value_target.new_zeros((batch, ACTION_SPACE))
        safety_mask = targets.safety_mask
        if safety_mask is None:
            safety_mask = targets.value_target.new_zeros((batch, ACTION_SPACE))
    return BaseTargets(
        policy_target=targets.policy_target,
        legal_mask=targets.legal_mask,
        value_target=targets.value_target,
        grp_target=targets.grp_target,
        tenpai_target=targets.tenpai_target,
        danger_target=targets.danger_target,
        danger_mask=targets.danger_mask,
        opp_next_target=targets.opp_next_target,
        score_pdf_target=targets.score_pdf_target,
        score_cdf_target=targets.score_cdf_target,
        oracle_target=oracle_target,
        oracle_target_mask=oracle_target_mask,
        safety_target=safety_target,
        safety_mask=safety_mask,
    )
