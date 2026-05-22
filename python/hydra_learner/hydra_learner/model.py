"""PyTorch model pieces for Hydra experimental learner."""

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
DEFAULT_BLOCKS = 12
DEFAULT_SE_BOTTLENECK = 64


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
    """Conv/GN/Mish residual block with squeeze-excitation gating."""

    def __init__(self, hidden: int, bottleneck: int) -> None:
        super().__init__()
        groups = min(32, hidden)
        self.norm1 = nn.GroupNorm(groups, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.se_fc1 = nn.Linear(hidden, bottleneck)
        self.se_fc2 = nn.Linear(bottleneck, hidden)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(F.mish(self.norm1(x)))
        y = self.conv2(F.mish(self.norm2(y)))
        se = y.mean(dim=2)
        se = self.se_fc2(F.mish(self.se_fc1(se))).sigmoid().unsqueeze(2)
        return residual + y * se


class HydraPolicyNet(nn.Module):
    """Hydra base-head hot-path topology used by the PyTorch learner."""

    def __init__(
        self,
        hidden: int = DEFAULT_HIDDEN,
        blocks: int = DEFAULT_BLOCKS,
        bottleneck: int = DEFAULT_SE_BOTTLENECK,
        actions: int = ACTION_SPACE,
    ) -> None:
        super().__init__()
        if actions != ACTION_SPACE:
            raise ValueError(f"Hydra base-head model requires action space {ACTION_SPACE}, got {actions}")
        self.input = nn.Conv1d(OBS_CHANNELS, hidden, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList([ResidualSeBlock(hidden, bottleneck) for _ in range(blocks)])
        self.final_norm = nn.GroupNorm(min(32, hidden), hidden)
        self.base_heads = nn.Linear(hidden, BASE_LINEAR_HEADS)
        self.opp_next_head = nn.Conv1d(hidden, OPPONENTS, kernel_size=1)
        self.danger_head = nn.Conv1d(hidden, OPPONENTS, kernel_size=1)

    @override
    def forward(self, obs: torch.Tensor) -> HydraBaseOutput:
        x = self.input(obs)
        for block in self.blocks:
            x = block(x)
        spatial = F.mish(self.final_norm(x))
        pooled = spatial.mean(dim=2)
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
            opp_next_discard=self.opp_next_head(spatial),
            danger=self.danger_head(spatial),
        )
