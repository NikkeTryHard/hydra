from __future__ import annotations

from flax import struct
import jax.numpy as jnp
import optax

from .model import HydraOutput


NEG_INF = -1e9


@struct.dataclass
class LossBreakdown:
    policy: jnp.ndarray
    value: jnp.ndarray
    grp: jnp.ndarray
    tenpai: jnp.ndarray
    danger: jnp.ndarray
    opp_next: jnp.ndarray
    score_pdf: jnp.ndarray
    score_cdf: jnp.ndarray
    safety_residual: jnp.ndarray
    total: jnp.ndarray


def _soft_cross_entropy(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target * log_probs, axis=-1)


def policy_ce(
    logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    masked = logits + (1.0 - mask) * NEG_INF
    return _soft_cross_entropy(masked, target)


def value_mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    diff = pred - target
    return 0.5 * diff * diff


def grp_ce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return _soft_cross_entropy(logits, target)


def tenpai_bce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, target), axis=-1)


def danger_focal_bce(
    logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    alpha = 0.25
    gamma = 2.0
    probs = jax.nn.sigmoid(logits)
    bce = optax.sigmoid_binary_cross_entropy(logits, target)
    p_t = target * probs + (1.0 - target) * (1.0 - probs)
    focal = (1.0 - p_t) ** gamma * alpha * bce * mask
    return jnp.sum(focal, axis=(1, 2))


def opp_next_ce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(target * log_probs, axis=-1), axis=-1)


def score_pdf_ce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return _soft_cross_entropy(logits, target)


def score_cdf_bce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, target), axis=-1)


def masked_action_mse(
    pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    sq = 0.5 * jnp.square(pred - target) * mask
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(sq) / denom


def total_loss(output: HydraOutput, batch: dict[str, jnp.ndarray]) -> LossBreakdown:
    l_pi = jnp.mean(
        policy_ce(output.policy_logits, batch["policy_target"], batch["legal_mask"])
    )
    l_v = jnp.mean(value_mse(jnp.squeeze(output.value, axis=-1), batch["value_target"]))
    l_grp = jnp.mean(grp_ce(output.grp, batch["grp_target"]))
    l_tenpai = jnp.mean(tenpai_bce(output.opp_tenpai, batch["tenpai_target"]))
    l_danger = jnp.mean(
        danger_focal_bce(output.danger, batch["danger_target"], batch["danger_mask"])
    )
    l_opp = jnp.mean(opp_next_ce(output.opp_next_discard, batch["opp_next_target"]))
    l_pdf = jnp.mean(score_pdf_ce(output.score_pdf, batch["score_pdf_target"]))
    l_cdf = jnp.mean(score_cdf_bce(output.score_cdf, batch["score_cdf_target"]))
    safety_presence = jnp.asarray(batch["safety_residual_present"], dtype=jnp.float32)
    safety_mask = batch["safety_residual_mask"] * safety_presence[:, None]
    l_sr = masked_action_mse(
        output.safety_residual,
        batch["safety_residual_target"],
        safety_mask,
    )

    total = (
        l_pi * 1.0
        + l_v * 0.5
        + l_grp * 0.2
        + l_tenpai * 0.1
        + l_danger * 0.1
        + l_opp * 0.1
        + l_pdf * 0.025
        + l_cdf * 0.025
        + l_sr * 0.1
    )
    return LossBreakdown(
        policy=l_pi,
        value=l_v,
        grp=l_grp,
        tenpai=l_tenpai,
        danger=l_danger,
        opp_next=l_opp,
        score_pdf=l_pdf,
        score_cdf=l_cdf,
        safety_residual=l_sr,
        total=total,
    )


import jax
