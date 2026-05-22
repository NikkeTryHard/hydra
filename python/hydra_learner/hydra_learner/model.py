"""PyTorch model pieces for Hydra BC learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_CHANNELS = 192
TILE_WIDTH = 34
ACTION_SPACE = 46
SCORE_BINS = 64
OPPONENTS = 3
GRP_CLASSES = 24
BASE_LINEAR_HEADS = ACTION_SPACE + 1 + SCORE_BINS + SCORE_BINS + OPPONENTS + GRP_CLASSES + 4 + ACTION_SPACE
DEFAULT_HIDDEN = 256
DEFAULT_BLOCKS = 10
DEFAULT_SE_BOTTLENECK = 64
BACKBONE_PROFILE_CONV2D_LOCAL3 = "conv2d_local3"
BACKBONE_PROFILE_DEFAULT = BACKBONE_PROFILE_CONV2D_LOCAL3
BACKBONE_PROFILES = (BACKBONE_PROFILE_CONV2D_LOCAL3,)
CONV_MEMORY_FORMAT_CONTIGUOUS = "contiguous"
CONV_MEMORY_FORMAT_CHANNELS_LAST = "channels_last"
CONV_MEMORY_FORMAT_DEFAULT = CONV_MEMORY_FORMAT_CONTIGUOUS
CONV_MEMORY_FORMATS = (CONV_MEMORY_FORMAT_CONTIGUOUS, CONV_MEMORY_FORMAT_CHANNELS_LAST)
RESIDUAL_PROFILE_DEFAULT = "mish_se"
RESIDUAL_PROFILE_MISH_NO_SE = "mish_no_se"
RESIDUAL_PROFILE_SILU_SE = "silu_se"
RESIDUAL_PROFILE_RELU_SE = "relu_se"
RESIDUAL_PROFILE_RELU_NO_SE = "relu_no_se"
RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE = "relu_no_norm_no_se"
RESIDUAL_PROFILES = (
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILE_SILU_SE,
    RESIDUAL_PROFILE_RELU_SE,
    RESIDUAL_PROFILE_MISH_NO_SE,
    RESIDUAL_PROFILE_RELU_NO_SE,
    RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE,
)


def validate_residual_profile(profile: str) -> str:
    if profile not in RESIDUAL_PROFILES:
        raise ValueError(f"unsupported residual profile {profile!r}")
    return profile


def validate_backbone_profile(profile: str) -> str:
    if profile not in BACKBONE_PROFILES:
        raise ValueError(f"unsupported backbone profile {profile!r}")
    return profile


def validate_conv_memory_format(memory_format: str) -> str:
    if memory_format not in CONV_MEMORY_FORMATS:
        raise ValueError(f"unsupported conv memory format {memory_format!r}")
    return memory_format


def _profile_uses_se(profile: str) -> bool:
    return profile in (RESIDUAL_PROFILE_DEFAULT, RESIDUAL_PROFILE_SILU_SE, RESIDUAL_PROFILE_RELU_SE)


def _profile_uses_norm(profile: str) -> bool:
    return profile != RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE


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
    if not _profile_uses_norm(profile):
        return nn.Identity()
    return nn.GroupNorm(_group_norm_groups(hidden), hidden)


def _norm_activation(profile: str, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _activation(profile, norm(x))


class Conv2dBackbone(nn.Module):
    """Conv2d local-3 backbone over singleton height and tile width."""

    def __init__(self, hidden: int, blocks: int, bottleneck: int, residual_profile: str, memory_format: str) -> None:
        super().__init__()
        self.input = nn.Conv2d(OBS_CHANNELS, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.blocks = nn.ModuleList([ResidualSeBlock(hidden, bottleneck, residual_profile) for _ in range(blocks)])
        self.final_norm = _make_norm(residual_profile, hidden)
        self.residual_profile = residual_profile
        self.memory_format = validate_conv_memory_format(memory_format)

    @override
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.unsqueeze(2)
        if self.memory_format == CONV_MEMORY_FORMAT_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return _norm_activation(self.residual_profile, self.final_norm, x)


@dataclass(frozen=True)
class HydraBaseOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    score_pdf: torch.Tensor
    score_cdf: torch.Tensor
    opp_tenpai: torch.Tensor
    grp: torch.Tensor
    oracle_critic: torch.Tensor
    safety_residual: torch.Tensor
    opp_next_discard: torch.Tensor
    danger: torch.Tensor


class ResidualSeBlock(nn.Module):
    """Conv2d residual block selected by checkpoint-stable residual profile."""

    def __init__(self, hidden: int, bottleneck: int, residual_profile: str = RESIDUAL_PROFILE_DEFAULT) -> None:
        super().__init__()
        self.residual_profile = validate_residual_profile(residual_profile)
        self.norm1 = _make_norm(self.residual_profile, hidden)
        self.conv1 = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.norm2 = _make_norm(self.residual_profile, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        if _profile_uses_se(self.residual_profile):
            self.se_fc1 = nn.Linear(hidden, bottleneck)
            self.se_fc2 = nn.Linear(bottleneck, hidden)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(_norm_activation(self.residual_profile, self.norm1, x))
        y = self.conv2(_norm_activation(self.residual_profile, self.norm2, y))
        if not _profile_uses_se(self.residual_profile):
            return residual + y
        se = y.mean(dim=(2, 3))
        se = (
            self.se_fc2(_activation(self.residual_profile, self.se_fc1(se)))
            .sigmoid()
            .view(y.shape[0], y.shape[1], 1, 1)
        )
        return residual + y * se


class HydraPolicyNet(nn.Module):
    """Hydra base-head hot-path topology used by the PyTorch learner."""

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        blocks: int = DEFAULT_BLOCKS,
        bottleneck: int = DEFAULT_SE_BOTTLENECK,
        actions: int = ACTION_SPACE,
        residual_profile: str = RESIDUAL_PROFILE_DEFAULT,
        backbone_profile: str = BACKBONE_PROFILE_DEFAULT,
        conv_memory_format: str = CONV_MEMORY_FORMAT_DEFAULT,
    ) -> None:
        super().__init__()
        if actions != ACTION_SPACE:
            raise ValueError(f"Hydra base-head model requires action space {ACTION_SPACE}, got {actions}")
        residual_profile = validate_residual_profile(residual_profile)
        backbone_profile = validate_backbone_profile(backbone_profile)
        conv_memory_format = validate_conv_memory_format(conv_memory_format)
        self.backbone = Conv2dBackbone(hidden, blocks, bottleneck, residual_profile, conv_memory_format)
        self.residual_profile = residual_profile
        self.backbone_profile = backbone_profile
        self.base_heads = nn.Linear(hidden, BASE_LINEAR_HEADS)
        self.opp_next_head = nn.Conv2d(hidden, OPPONENTS, kernel_size=1)
        self.danger_head = nn.Conv2d(hidden, OPPONENTS, kernel_size=1)

    @override
    def forward(self, obs: torch.Tensor) -> HydraBaseOutput:
        spatial = self.backbone(obs)
        pooled = spatial.mean(dim=(2, 3))
        packed = self.base_heads(pooled)
        return HydraBaseOutput(
            policy_logits=packed[:, 0:46],
            value=torch.tanh(packed[:, 46:47]),
            score_pdf=packed[:, 47:111],
            score_cdf=packed[:, 111:175],
            opp_tenpai=packed[:, 175:178],
            grp=packed[:, 178:202],
            oracle_critic=packed[:, 202:206],
            safety_residual=packed[:, 206:252],
            opp_next_discard=self.opp_next_head(spatial).squeeze(2),
            danger=self.danger_head(spatial).squeeze(2),
        )
