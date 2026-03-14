from __future__ import annotations

from flax.training.train_state import TrainState
import optax


def create_learning_rate_schedule(
    learning_rate: float, min_learning_rate: float, warmup_steps: int
):
    warmup_steps = max(int(warmup_steps), 0)
    if warmup_steps == 0 or abs(learning_rate - min_learning_rate) <= 1e-12:
        return optax.constant_schedule(learning_rate)

    warmup = optax.linear_schedule(
        init_value=min_learning_rate,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    hold = optax.constant_schedule(learning_rate)
    return optax.join_schedules([warmup, hold], boundaries=[warmup_steps])


def create_optimizer(
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    warmup_steps: int,
):
    schedule = create_learning_rate_schedule(
        learning_rate, min_learning_rate, warmup_steps
    )
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay, eps=1e-8),
    )


def create_train_state(
    apply_fn,
    params,
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    warmup_steps: int,
):
    tx = create_optimizer(
        learning_rate, min_learning_rate, weight_decay, grad_clip_norm, warmup_steps
    )
    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def restore_train_state(
    apply_fn,
    params,
    opt_state,
    step,
    learning_rate: float,
    min_learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    warmup_steps: int,
):
    tx = create_optimizer(
        learning_rate, min_learning_rate, weight_decay, grad_clip_norm, warmup_steps
    )
    return TrainState(
        step=step,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
    )
