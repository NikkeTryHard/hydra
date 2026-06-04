"""Checkpoint-stable model profile constants and validators."""

from __future__ import annotations

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


def profile_uses_se(profile: str) -> bool:
    return profile in (RESIDUAL_PROFILE_DEFAULT, RESIDUAL_PROFILE_SILU_SE, RESIDUAL_PROFILE_RELU_SE)


def profile_uses_eca(profile: str) -> bool:
    return profile == RESIDUAL_PROFILE_MISH_ECA


def profile_uses_norm(profile: str) -> bool:
    return profile != RESIDUAL_PROFILE_RELU_NO_NORM_NO_SE


_profile_uses_se = profile_uses_se
_profile_uses_eca = profile_uses_eca
_profile_uses_norm = profile_uses_norm
