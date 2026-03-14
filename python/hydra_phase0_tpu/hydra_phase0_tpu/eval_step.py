from __future__ import annotations

import jax

from .losses import total_loss


def eval_step(params, batch, model):
    output = model.apply({"params": params}, batch["obs"])
    return total_loss(output, batch)


eval_step = jax.jit(eval_step, static_argnames=("model",))
