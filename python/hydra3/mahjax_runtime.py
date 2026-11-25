"""Device-resident MahJax matches with DLPack exchange at the actor boundary."""

import os
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import torch

from .runtime import ArenaBatch


@cache
def _program():
    from . import engine

    engine.select()
    import mahjax

    from .mahjax_actions import action_descriptors
    from .mahjax_tokens import actor_tokens

    env = mahjax.make("red_4p", round_mode="half", order_points=[0, 0, 0, 0])
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)
    fold = jax.vmap(jax.random.fold_in)

    @jax.jit
    def start(words):
        roots = fold(jax.vmap(jax.random.key)(words[:, 0]), words[:, 1])
        deals = jnp.zeros(words.shape[0], dtype=jnp.uint32)
        return roots, deals, init(fold(roots, deals))

    @jax.jit
    def packet(state):
        def encode(row):
            descriptors = action_descriptors(row)
            active = ~(row.terminated | row.truncated)
            legal = (descriptors[:, 0] >= 0) & active
            positions = jnp.where(legal, jnp.cumsum(legal) - 1, descriptors.shape[0])
            packed = jnp.full_like(descriptors, -1).at[positions].set(descriptors, mode="drop")
            metadata = jnp.stack((row.current_player, legal.sum(), active))
            return metadata, actor_tokens(row), packed

        return jax.vmap(encode)(state)

    # Donate `deals` and `state`: the arena overwrites both with the returned
    # values and never reads the arguments again. Donation lets XLA reuse the
    # same device buffers each turn instead of re-parameterising CUDA-graph
    # nodes for freshly allocated addresses.
    @jax.jit(donate_argnums=(1, 2))
    def advance(roots, deals, state, actions):
        mask = state.legal_action_mask
        in_range = (actions >= 0) & (actions < mask.shape[1])
        allowed = jnp.take_along_axis(
            mask, jnp.clip(actions, 0, mask.shape[1] - 1)[:, None], axis=1
        )[:, 0]
        invalid = ~(state.terminated | state.truncated) & ~(in_range & allowed)
        updated = step(state, jnp.where(invalid, 0, actions), fold(roots, deals + jnp.uint32(1)))
        updated = updated.replace(truncated=updated.truncated | invalid)
        before, after = state.round_state, updated.round_state
        # Dealer repeats increment honba; all other new deals advance the round.
        # Key walls by deal ordinal so policies taking different numbers of turns still share walls.
        redealt = (before.round != after.round) | (before.honba != after.honba)
        return deals + (redealt & ~updated.terminated).astype(jnp.uint32), updated

    return start, packet, advance


class MahJaxArena:
    """Keep lane shapes fixed as matches finish so the rules are compiled once per cohort."""

    def __init__(self, seeds: list[int], device: torch.device):
        # JAX and Torch share VRAM; prevent JAX from reserving it before Torch runs.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        platform = "gpu" if device.type == "cuda" else "cpu"
        self._device = jax.devices(platform)[device.index or 0]
        self._size = len(seeds)
        self._slots = torch.arange(self._size, dtype=torch.int64)
        self._step_count = 0
        words = np.asarray([(seed & 0xFFFFFFFF, seed >> 32) for seed in seeds], dtype=np.uint32)
        start, self._packet, self._advance = _program()
        self._roots, self._deals, self._state = start(jax.device_put(words, self._device))

    def batch(self) -> ArenaBatch | None:
        metadata, tokens, actions = self._packet(self._state)
        seats, counts, playing = torch.from_dlpack(metadata).cpu().unbind(dim=1)
        active = playing.nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        if self._step_count >= 4096:
            raise RuntimeError("MahJax match did not finish within 4096 decisions")
        if bool((counts[active] == 0).any()):
            raise RuntimeError("MahJax returned an active seat with no legal action")
        width = int(counts.max())
        tokens_t = torch.from_dlpack(tokens)
        actions_t = torch.from_dlpack(actions)[:, :width]
        if active.numel() == self._size:
            return ArenaBatch(self._slots, seats, tokens_t, actions_t)
        index = active.to(tokens_t.device)
        return ArenaBatch(active, seats[active], tokens_t[index], actions_t[index])

    def step(self, slots: torch.Tensor, ids: torch.Tensor) -> None:
        if slots.numel() != self._size:
            padded = torch.zeros((self._size,), dtype=torch.int32, device=ids.device)
            padded.index_copy_(0, slots.to(ids.device), ids)
            ids = padded
        actions = jax.dlpack.from_dlpack(ids.contiguous())
        self._deals, self._state = self._advance(self._roots, self._deals, self._state, actions)
        self._step_count += 1

    def outcomes(self) -> tuple[torch.Tensor, torch.Tensor]:
        # MahJax settles leftover riichi sticks before marking the round terminal.
        # Use the settled score so reported points include those sticks.
        scores, winds, done, truncated = jax.device_get(
            (
                self._state.round_state.score,
                self._state.round_state.init_wind,
                self._state.terminated,
                self._state.truncated,
            )
        )
        if not done.all() or truncated.any():
            raise RuntimeError("MahJax returned an incomplete or truncated match")
        order = np.lexsort((winds, -scores), axis=1)
        ranks = np.empty_like(scores)
        np.put_along_axis(ranks, order, np.arange(1, 5, dtype=np.int32)[None, :], axis=1)
        return torch.from_numpy(scores * 100), torch.from_numpy(ranks)
