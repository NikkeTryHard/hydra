"""PyTorch model surface for Hydra BC learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_learner.model.backbones import Conv2dBackbone, TileformerBiasBackbone
from hydra_learner.model.blocks import ResidualSeBlock
from hydra_learner.model.profiles import (
    ACTION_SPACE,
    BACKBONE_PROFILE_CONV2D_LOCAL3,
    BACKBONE_PROFILE_CONVNEXT_TILE_K7,
    BACKBONE_PROFILE_DEFAULT,
    BACKBONE_PROFILE_GLOBAL_POOL_BIAS,
    BACKBONE_PROFILE_TILEFORMER_BIAS,
    BACKBONE_PROFILES,
    BASE_LINEAR_HEADS,
    CONV_MEMORY_FORMAT_CHANNELS_LAST,
    CONV_MEMORY_FORMAT_CONTIGUOUS,
    CONV_MEMORY_FORMAT_DEFAULT,
    CONV_MEMORY_FORMATS,
    DEFAULT_BLOCKS,
    DEFAULT_HIDDEN,
    DEFAULT_SE_BOTTLENECK,
    GRP_CLASSES,
    OBS_CHANNELS,
    OPPONENTS,
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILE_MISH_ECA,
    RESIDUAL_PROFILE_MISH_NO_SE,
    RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE,
    RESIDUAL_PROFILE_RELU_NO_SE,
    RESIDUAL_PROFILE_RELU_SE,
    RESIDUAL_PROFILE_SILU_SE,
    RESIDUAL_PROFILES,
    SCORE_BINS,
    TILE_WIDTH,
    validate_backbone_profile,
    validate_conv_memory_format,
    validate_residual_profile,
)

__all__ = [
    "ACTION_SPACE",
    "BACKBONE_PROFILES",
    "BACKBONE_PROFILE_CONV2D_LOCAL3",
    "BACKBONE_PROFILE_CONVNEXT_TILE_K7",
    "BACKBONE_PROFILE_DEFAULT",
    "BACKBONE_PROFILE_GLOBAL_POOL_BIAS",
    "BACKBONE_PROFILE_TILEFORMER_BIAS",
    "BASE_LINEAR_HEADS",
    "CONV_MEMORY_FORMATS",
    "CONV_MEMORY_FORMAT_CHANNELS_LAST",
    "CONV_MEMORY_FORMAT_CONTIGUOUS",
    "CONV_MEMORY_FORMAT_DEFAULT",
    "DEFAULT_BLOCKS",
    "DEFAULT_HIDDEN",
    "DEFAULT_SE_BOTTLENECK",
    "GRP_CLASSES",
    "OBS_CHANNELS",
    "OPPONENTS",
    "RESIDUAL_PROFILES",
    "RESIDUAL_PROFILE_DEFAULT",
    "RESIDUAL_PROFILE_MISH_ECA",
    "RESIDUAL_PROFILE_MISH_NO_SE",
    "RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE",
    "RESIDUAL_PROFILE_RELU_NO_SE",
    "RESIDUAL_PROFILE_RELU_SE",
    "RESIDUAL_PROFILE_SILU_SE",
    "SCORE_BINS",
    "TILE_WIDTH",
    "HydraBaseOutput",
    "HydraPolicyNet",
    "ResidualSeBlock",
    "validate_backbone_profile",
    "validate_conv_memory_format",
    "validate_residual_profile",
]


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


class HydraPolicyNet(nn.Module):
    """Hydra base-head hot-path topology used by the PyTorch learner."""

    policy_value: ClassVar

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
        self.backbone: TileformerBiasBackbone | Conv2dBackbone
        if backbone_profile == BACKBONE_PROFILE_TILEFORMER_BIAS:
            self.backbone = TileformerBiasBackbone(hidden, blocks)
        else:
            self.backbone = Conv2dBackbone(
                hidden, blocks, bottleneck, residual_profile, conv_memory_format, backbone_profile
            )
        self.residual_profile = residual_profile
        self.backbone_profile = backbone_profile
        self.base_heads = nn.Linear(hidden, BASE_LINEAR_HEADS)
        self.opp_next_head = nn.Conv2d(hidden, OPPONENTS, kernel_size=1)
        self.danger_head = nn.Conv2d(hidden, OPPONENTS, kernel_size=1)

    def _features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_profile == BACKBONE_PROFILE_TILEFORMER_BIAS:
            return self.backbone(obs)
        spatial = self.backbone(obs)
        return spatial, spatial.mean(dim=(2, 3))

    def policy_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only policy logits and scalar value for PPO/arena hot paths."""
        _, pooled = self._features(obs)
        packed = F.linear(pooled, self.base_heads.weight[:47], self.base_heads.bias[:47])
        return packed[:, 0:46], torch.tanh(packed[:, 46:47])

    @override
    def forward(self, obs: torch.Tensor) -> HydraBaseOutput:
        spatial, pooled = self._features(obs)
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
