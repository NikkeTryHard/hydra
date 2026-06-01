"""Reusable convolutional blocks for HydraPolicyNet."""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_learner.model_profiles import (
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE,
    RESIDUAL_PROFILE_RELU_NO_SE,
    RESIDUAL_PROFILE_RELU_SE,
    RESIDUAL_PROFILE_SILU_SE,
    profile_uses_eca,
    profile_uses_norm,
    profile_uses_se,
    validate_residual_profile,
)


def _activation(profile: str, x: torch.Tensor) -> torch.Tensor:
    if profile == RESIDUAL_PROFILE_SILU_SE:
        return F.silu(x)
    if profile in (RESIDUAL_PROFILE_RELU_SE, RESIDUAL_PROFILE_RELU_NO_SE, RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE):
        return F.relu(x)
    return F.mish(x)


def _group_norm_groups(hidden: int) -> int:
    groups = min(32, hidden)
    if hidden % groups != 0:
        raise ValueError(f"GroupNorm requires hidden channels divisible by groups: hidden={hidden} groups={groups}")
    return groups


def _make_norm(profile: str, hidden: int) -> nn.Module:
    if not profile_uses_norm(profile):
        return nn.Identity()
    return nn.GroupNorm(_group_norm_groups(hidden), hidden)


def _norm_activation(profile: str, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _activation(profile, norm(x))


class ResidualSeBlock(nn.Module):
    """Conv2d residual block selected by checkpoint-stable residual profile."""

    def __init__(self, hidden: int, bottleneck: int, residual_profile: str = RESIDUAL_PROFILE_DEFAULT) -> None:
        super().__init__()
        self.residual_profile = validate_residual_profile(residual_profile)
        self.norm1 = _make_norm(self.residual_profile, hidden)
        self.conv1 = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.norm2 = _make_norm(self.residual_profile, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        if profile_uses_se(self.residual_profile):
            self.se_fc1 = nn.Linear(hidden, bottleneck)
            self.se_fc2 = nn.Linear(bottleneck, hidden)
        if profile_uses_eca(self.residual_profile):
            self.eca_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(_norm_activation(self.residual_profile, self.norm1, x))
        y = self.conv2(_norm_activation(self.residual_profile, self.norm2, y))
        if profile_uses_eca(self.residual_profile):
            eca = y.mean(dim=(2, 3)).unsqueeze(1)
            eca = self.eca_conv(eca).sigmoid().squeeze(1).view(y.shape[0], y.shape[1], 1, 1)
            return residual + y * eca
        if not profile_uses_se(self.residual_profile):
            return residual + y
        se = y.mean(dim=(2, 3))
        se = (
            self.se_fc2(_activation(self.residual_profile, self.se_fc1(se)))
            .sigmoid()
            .view(y.shape[0], y.shape[1], 1, 1)
        )
        return residual + y * se
