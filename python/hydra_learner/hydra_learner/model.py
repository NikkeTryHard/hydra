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
BACKBONE_PROFILE_TILEFORMER_BIAS = "tileformer_bias"
BACKBONE_PROFILE_CONVNEXT_TILE_K7 = "convnext_tile_k7"
BACKBONE_PROFILE_GLOBAL_POOL_BIAS = "global_pool_bias"
BACKBONE_PROFILE_DEFAULT = BACKBONE_PROFILE_CONV2D_LOCAL3
BACKBONE_PROFILES = (
    BACKBONE_PROFILE_CONV2D_LOCAL3,
    BACKBONE_PROFILE_TILEFORMER_BIAS,
    BACKBONE_PROFILE_CONVNEXT_TILE_K7,
    BACKBONE_PROFILE_GLOBAL_POOL_BIAS,
)
CONV_MEMORY_FORMAT_CONTIGUOUS = "contiguous"
CONV_MEMORY_FORMAT_CHANNELS_LAST = "channels_last"
CONV_MEMORY_FORMAT_DEFAULT = CONV_MEMORY_FORMAT_CONTIGUOUS
CONV_MEMORY_FORMATS = (CONV_MEMORY_FORMAT_CONTIGUOUS, CONV_MEMORY_FORMAT_CHANNELS_LAST)
RESIDUAL_PROFILE_DEFAULT = "mish_se"
RESIDUAL_PROFILE_MISH_NO_SE = "mish_no_se"
RESIDUAL_PROFILE_MISH_ECA = "mish_eca"
RESIDUAL_PROFILE_SILU_SE = "silu_se"
RESIDUAL_PROFILE_RELU_SE = "relu_se"
RESIDUAL_PROFILE_RELU_NO_SE = "relu_no_se"
RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE = "relu_no_norm_no_se"
RESIDUAL_PROFILES = (
    RESIDUAL_PROFILE_DEFAULT,
    RESIDUAL_PROFILE_SILU_SE,
    RESIDUAL_PROFILE_RELU_SE,
    RESIDUAL_PROFILE_MISH_NO_SE,
    RESIDUAL_PROFILE_MISH_ECA,
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


def _profile_uses_eca(profile: str) -> bool:
    return profile == RESIDUAL_PROFILE_MISH_ECA


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


class TileformerAttention(nn.Module):
    """Multi-head tile attention with learned static Mahjong relation bias."""

    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"tileformer d_model must be divisible by heads: d_model={d_model} heads={heads}")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(heads, 6))
        self.register_buffer("relation_mask", _tileformer_relation_mask(), persistent=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        attn = q @ k.transpose(-2, -1)
        attn = attn * (self.head_dim**-0.5)
        bias = torch.einsum("hr,rij->hij", self.relation_bias, self.relation_mask.to(device=x.device, dtype=x.dtype))
        attn = attn + bias.unsqueeze(0)
        y = attn.softmax(dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.out(y)


class TileformerBlock(nn.Module):
    """Pre-norm Transformer block for 34 tile tokens plus CLS."""

    def __init__(self, d_model: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TileformerAttention(d_model, heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp_gate = nn.Linear(d_model, hidden)
        self.mlp_value = nn.Linear(d_model, hidden)
        self.mlp_out = nn.Linear(hidden, d_model)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        y = self.norm2(x)
        return x + self.mlp_out(F.silu(self.mlp_gate(y)) * self.mlp_value(y))


class TileformerBiasBackbone(nn.Module):
    """Tile-token backbone with static Mahjong relation attention bias."""

    def __init__(self, d_model: int, layers: int) -> None:
        super().__init__()
        heads = _tileformer_heads(d_model)
        self.input = nn.Linear(OBS_CHANNELS, d_model, bias=False)
        self.tile_embedding = nn.Embedding(TILE_WIDTH, d_model)
        self.suit_embedding = nn.Embedding(5, d_model)
        self.rank_embedding = nn.Embedding(10, d_model)
        self.terminal_embedding = nn.Embedding(2, d_model)
        self.honor_embedding = nn.Embedding(2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TileformerBlock(d_model, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.register_buffer("tile_id", torch.arange(TILE_WIDTH, dtype=torch.long), persistent=False)
        self.register_buffer("suit_id", _tileformer_suit_ids(), persistent=False)
        self.register_buffer("rank_id", _tileformer_rank_ids(), persistent=False)
        self.register_buffer("terminal_id", _tileformer_terminal_ids(), persistent=False)
        self.register_buffer("honor_id", _tileformer_honor_ids(), persistent=False)

    @override
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input(obs.transpose(1, 2))
        x = x + self.tile_embedding(self.tile_id)
        x = x + self.suit_embedding(self.suit_id)
        x = x + self.rank_embedding(self.rank_id)
        x = x + self.terminal_embedding(self.terminal_id)
        x = x + self.honor_embedding(self.honor_id)
        cls = self.cls.expand(obs.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        tiles = x[:, 1:, :].transpose(1, 2).unsqueeze(2)
        return tiles, x[:, 0, :]


def _tileformer_heads(d_model: int) -> int:
    if d_model % 12 == 0:
        return 12
    if d_model % 8 == 0:
        return 8
    raise ValueError(f"tileformer d_model must be divisible by 12 or 8, got {d_model}")


def _tileformer_suit_ids() -> torch.Tensor:
    suits = torch.full((TILE_WIDTH,), 4, dtype=torch.long)
    suits[0:9] = 0
    suits[9:18] = 1
    suits[18:27] = 2
    suits[27:34] = 3
    return suits


def _tileformer_rank_ids() -> torch.Tensor:
    ranks = torch.empty(TILE_WIDTH, dtype=torch.long)
    ranks[0:9] = torch.arange(9)
    ranks[9:18] = torch.arange(9)
    ranks[18:27] = torch.arange(9)
    ranks[27:34] = torch.arange(7)
    return ranks


def _tileformer_terminal_ids() -> torch.Tensor:
    terminal = torch.zeros(TILE_WIDTH, dtype=torch.long)
    terminal[[0, 8, 9, 17, 18, 26]] = 1
    return terminal


def _tileformer_honor_ids() -> torch.Tensor:
    honor = torch.zeros(TILE_WIDTH, dtype=torch.long)
    honor[27:34] = 1
    return honor


def _tileformer_relation_mask() -> torch.Tensor:
    suits = _tileformer_suit_ids()
    ranks = _tileformer_rank_ids()
    terminal = _tileformer_terminal_ids()
    honor = _tileformer_honor_ids()
    mask = torch.zeros(6, TILE_WIDTH + 1, TILE_WIDTH + 1)
    for i in range(TILE_WIDTH):
        ti = i + 1
        for j in range(TILE_WIDTH):
            tj = j + 1
            same_suit = suits[i] < 3 and suits[i] == suits[j]
            if i == j:
                mask[0, ti, tj] = 1.0
            if same_suit:
                dist = abs(int(ranks[i]) - int(ranks[j]))
                if dist == 1:
                    mask[1, ti, tj] = 1.0
                elif dist == 2:
                    mask[2, ti, tj] = 1.0
            if suits[i] < 3 and suits[j] < 3 and suits[i] != suits[j] and ranks[i] == ranks[j]:
                mask[3, ti, tj] = 1.0
            if honor[i] == 1 and honor[j] == 1:
                mask[4, ti, tj] = 1.0
            if terminal[i] == 1 and terminal[j] == 1:
                mask[5, ti, tj] = 1.0
    return mask


class ConvNextTileK7Block(nn.Module):
    """ConvNeXt-style tile-width residual block preserving 1x34 spatial layout."""

    def __init__(self, hidden: int, expand_ratio: int = 4) -> None:
        super().__init__()
        expanded = hidden * expand_ratio
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=(1, 7), padding=(0, 3), groups=hidden, bias=False)
        self.norm = nn.GroupNorm(_group_norm_groups(hidden), hidden)
        self.expand = nn.Conv2d(hidden, expanded, kernel_size=1)
        self.project = nn.Conv2d(expanded, hidden, kernel_size=1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x)
        y = self.norm(y)
        y = self.expand(y)
        y = F.gelu(y)
        y = self.project(y)
        return residual + y


class GlobalPoolChannelBias(nn.Module):
    """Mean+max global channel summary projected back as a spatial channel bias."""

    def __init__(self, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden * 2, bottleneck)
        self.fc2 = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        bias = self.fc2(F.silu(self.fc1(summary))).view(x.shape[0], x.shape[1], 1, 1)
        return x + bias


class Conv2dBackbone(nn.Module):
    """Conv2d backbone over singleton height and tile width."""

    def __init__(
        self,
        hidden: int,
        blocks: int,
        bottleneck: int,
        residual_profile: str,
        memory_format: str,
        backbone_profile: str = BACKBONE_PROFILE_CONV2D_LOCAL3,
    ) -> None:
        super().__init__()
        self.backbone_profile = validate_backbone_profile(backbone_profile)
        self.input = nn.Conv2d(OBS_CHANNELS, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        if self.backbone_profile == BACKBONE_PROFILE_CONVNEXT_TILE_K7:
            self.blocks = nn.ModuleList([ConvNextTileK7Block(hidden) for _ in range(blocks)])
        else:
            self.blocks = nn.ModuleList([ResidualSeBlock(hidden, bottleneck, residual_profile) for _ in range(blocks)])
        if self.backbone_profile == BACKBONE_PROFILE_GLOBAL_POOL_BIAS:
            self.channel_bias = GlobalPoolChannelBias(hidden, bottleneck)
        self.final_norm = _make_norm(residual_profile, hidden)
        self.residual_profile = residual_profile
        self.memory_format = validate_conv_memory_format(memory_format)

    @override
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.unsqueeze(2)
        if self.memory_format == CONV_MEMORY_FORMAT_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.input(x)
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if self.backbone_profile == BACKBONE_PROFILE_GLOBAL_POOL_BIAS and index % 4 == 0:
                x = self.channel_bias(x)
        if self.backbone_profile == BACKBONE_PROFILE_GLOBAL_POOL_BIAS and len(self.blocks) % 4 != 0:
            x = self.channel_bias(x)
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
        if _profile_uses_eca(self.residual_profile):
            self.eca_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(_norm_activation(self.residual_profile, self.norm1, x))
        y = self.conv2(_norm_activation(self.residual_profile, self.norm2, y))
        if _profile_uses_eca(self.residual_profile):
            eca = y.mean(dim=(2, 3)).unsqueeze(1)
            eca = self.eca_conv(eca).sigmoid().squeeze(1).view(y.shape[0], y.shape[1], 1, 1)
            return residual + y * eca
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
