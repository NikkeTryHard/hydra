from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .losses import LossBreakdown, total_loss


def train_step(state, batch, model):
    def loss_fn(params):
        output = model.apply({"params": params}, batch["obs"])
        breakdown = total_loss(output, batch)
        return breakdown.total, breakdown

    (loss, breakdown), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, breakdown


train_step = jax.jit(train_step, static_argnames=("model",))


@functools.partial(
    jax.pmap,
    axis_name="device",
    static_broadcasted_argnums=(2,),
    in_axes=(0, 0, None, None),
)
def _data_parallel_microbatch_grad(params, chunk, model, chunk_weight):
    def loss_fn(inner_params):
        output = model.apply({"params": inner_params}, chunk["obs"])
        breakdown = total_loss(output, chunk)
        return breakdown.total * chunk_weight, breakdown

    (weighted_loss, breakdown), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params
    )
    grads = jax.lax.pmean(grads, axis_name="device")
    weighted_loss = jax.lax.pmean(weighted_loss, axis_name="device")
    breakdown = jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="device"), breakdown
    )
    return (weighted_loss, breakdown), grads


def _microbatch_grad(params, chunk, model, chunk_weight: float):
    def loss_fn(inner_params):
        output = model.apply({"params": inner_params}, chunk["obs"])
        breakdown = total_loss(output, chunk)
        return breakdown.total * chunk_weight, breakdown

    return jax.value_and_grad(loss_fn, has_aux=True)(params)


_microbatch_grad = jax.jit(_microbatch_grad, static_argnames=("model",))


def _scale_breakdown(breakdown: LossBreakdown, weight) -> LossBreakdown:
    return jax.tree_util.tree_map(lambda x: x * weight, breakdown)


def _add_breakdowns(left: LossBreakdown, right: LossBreakdown) -> LossBreakdown:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def train_logical_batch(state, logical_batch_np, model, microbatch_size: int):
    total_size = int(logical_batch_np["obs"].shape[0])
    if total_size == 0:
        raise ValueError("logical batch must be non-empty")

    grads_acc = None
    breakdown_acc = None
    total_loss_value = 0.0

    for start in range(0, total_size, microbatch_size):
        end = min(start + microbatch_size, total_size)
        chunk = {key: value[start:end] for key, value in logical_batch_np.items()}
        chunk_weight = (end - start) / total_size

        (weighted_loss, breakdown), grads = _microbatch_grad(
            state.params, chunk, model, chunk_weight
        )
        total_loss_value += float(weighted_loss)
        weighted_breakdown = _scale_breakdown(breakdown, chunk_weight)
        breakdown_acc = (
            weighted_breakdown
            if breakdown_acc is None
            else _add_breakdowns(breakdown_acc, weighted_breakdown)
        )
        if grads_acc is None:
            grads_acc = grads
        else:
            grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)

    if grads_acc is None or breakdown_acc is None:
        raise ValueError("logical batch produced no gradients")

    state = state.apply_gradients(grads=grads_acc)
    return state, total_loss_value, breakdown_acc


def train_logical_batch_data_parallel(
    state, logical_batch, model, microbatch_size: int
):
    total_size = int(logical_batch["obs"].shape[1])
    if total_size == 0:
        raise ValueError("logical batch must be non-empty")

    grads_acc = None
    breakdown_acc = None
    total_loss_value = 0.0

    for start in range(0, total_size, microbatch_size):
        end = min(start + microbatch_size, total_size)
        chunk = {key: value[:, start:end] for key, value in logical_batch.items()}
        chunk_weight = jnp.asarray((end - start) / total_size, dtype=jnp.float32)
        (weighted_loss, breakdown), grads = _data_parallel_microbatch_grad(
            state.params, chunk, model, chunk_weight
        )
        total_loss_value += float(jax.device_get(weighted_loss[0]))
        weighted_breakdown = _scale_breakdown(breakdown, chunk_weight)
        breakdown_acc = (
            weighted_breakdown
            if breakdown_acc is None
            else _add_breakdowns(breakdown_acc, weighted_breakdown)
        )
        if grads_acc is None:
            grads_acc = grads
        else:
            grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)

    if grads_acc is None or breakdown_acc is None:
        raise ValueError("logical batch produced no gradients")

    state = state.apply_gradients(grads=grads_acc)
    return state, total_loss_value, breakdown_acc
