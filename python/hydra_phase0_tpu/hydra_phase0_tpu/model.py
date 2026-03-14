from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp


def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(nn.softplus(x))


@dataclass(frozen=True)
class HydraOutput:
    policy_logits: jnp.ndarray
    value: jnp.ndarray
    score_pdf: jnp.ndarray
    score_cdf: jnp.ndarray
    opp_tenpai: jnp.ndarray
    grp: jnp.ndarray
    opp_next_discard: jnp.ndarray
    danger: jnp.ndarray
    oracle_critic: jnp.ndarray
    belief_fields: jnp.ndarray
    mixture_weight_logits: jnp.ndarray
    opponent_hand_type: jnp.ndarray
    delta_q: jnp.ndarray
    safety_residual: jnp.ndarray


class SEBlock(nn.Module):
    channels: int
    bottleneck: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.mean(x, axis=1)
        scale = mish(
            nn.Dense(self.bottleneck, dtype=self.dtype, param_dtype=jnp.float32)(scale)
        )
        scale = nn.sigmoid(
            nn.Dense(self.channels, dtype=self.dtype, param_dtype=jnp.float32)(scale)
        )
        return x * scale[:, None, :]


class SEResBlock(nn.Module):
    channels: int
    num_groups: int
    se_bottleneck: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nn.GroupNorm(
            num_groups=self.num_groups, dtype=self.dtype, param_dtype=jnp.float32
        )(x)
        y = mish(y)
        y = nn.Conv(
            self.channels,
            kernel_size=(3,),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(y)
        y = nn.GroupNorm(
            num_groups=self.num_groups, dtype=self.dtype, param_dtype=jnp.float32
        )(y)
        y = mish(y)
        y = nn.Conv(
            self.channels,
            kernel_size=(3,),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(y)
        y = SEBlock(self.channels, self.se_bottleneck, dtype=self.dtype)(y)
        return y + residual


class SEResNet(nn.Module):
    num_blocks: int
    input_channels: int
    hidden_channels: int
    num_groups: int
    se_bottleneck: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        y = nn.Conv(
            self.hidden_channels,
            kernel_size=(3,),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        y = nn.GroupNorm(
            num_groups=self.num_groups, dtype=self.dtype, param_dtype=jnp.float32
        )(y)
        y = mish(y)
        for _ in range(self.num_blocks):
            y = SEResBlock(
                self.hidden_channels,
                self.num_groups,
                self.se_bottleneck,
                dtype=self.dtype,
            )(y)
        spatial = mish(
            nn.GroupNorm(
                num_groups=self.num_groups, dtype=self.dtype, param_dtype=jnp.float32
            )(y)
        )
        pooled = jnp.mean(spatial, axis=1)
        return spatial, pooled


class HydraModel(nn.Module):
    num_blocks: int = 24
    input_channels: int = 192
    hidden_channels: int = 256
    num_groups: int = 32
    se_bottleneck: int = 64
    action_space: int = 46
    score_bins: int = 64
    num_opponents: int = 3
    grp_classes: int = 24
    num_belief_components: int = 4
    opponent_hand_type_classes: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> HydraOutput:
        x = jnp.swapaxes(x, 1, 2)
        spatial, pooled = SEResNet(
            num_blocks=self.num_blocks,
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            num_groups=self.num_groups,
            se_bottleneck=self.se_bottleneck,
            dtype=self.dtype,
        )(x)
        oracle_input = jax.lax.stop_gradient(pooled)
        opp_shape = self.num_opponents * self.opponent_hand_type_classes
        belief_channels = self.num_belief_components * 4
        opp_next = nn.Conv(
            self.num_opponents,
            kernel_size=(1,),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(spatial)
        danger = nn.Conv(
            self.num_opponents,
            kernel_size=(1,),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(spatial)
        belief = nn.Conv(
            belief_channels,
            kernel_size=(1,),
            padding="VALID",
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(spatial)
        return HydraOutput(
            policy_logits=nn.Dense(
                self.action_space, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            value=jnp.tanh(
                nn.Dense(1, dtype=self.dtype, param_dtype=jnp.float32)(pooled)
            ),
            score_pdf=nn.Dense(
                self.score_bins, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            score_cdf=nn.Dense(
                self.score_bins, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            opp_tenpai=nn.Dense(
                self.num_opponents, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            grp=nn.Dense(self.grp_classes, dtype=self.dtype, param_dtype=jnp.float32)(
                pooled
            ),
            opp_next_discard=jnp.swapaxes(opp_next, 1, 2),
            danger=jnp.swapaxes(danger, 1, 2),
            oracle_critic=nn.Dense(4, dtype=self.dtype, param_dtype=jnp.float32)(
                oracle_input
            ),
            belief_fields=jnp.swapaxes(belief, 1, 2),
            mixture_weight_logits=nn.Dense(
                self.num_belief_components, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            opponent_hand_type=nn.Dense(
                opp_shape, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            delta_q=nn.Dense(
                self.action_space, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
            safety_residual=nn.Dense(
                self.action_space, dtype=self.dtype, param_dtype=jnp.float32
            )(pooled),
        )


import jax
