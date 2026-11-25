# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union, get_args, overload

import jax.numpy as jnp

from mahjax._src.struct import dataclass
from mahjax._src.types import Array, ArrayLike, PRNGKey

if TYPE_CHECKING:
    from mahjax.mahjong_3p.env import Mahjong3p
    from mahjax.mahjong_4p.env import Mahjong4p

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EnvId = Literal[
    "no_red_4p",
    "red_4p",
    "no_red_3p",
    "red_3p",
]


@dataclass
class State(abc.ABC):
    """Base state class of all Mahjax game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action, key)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (Array): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        rewards (Array): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.rewards` is zero for all players.
        terminated (Array): denotes that the state is terminal state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminate within a limited number of states (following AlphaGo).
        truncated (Array): indicates that the episode ends with the reason other than termination.
            Note that current Mahjax environments do not invoke truncation but users can use `TimeLimit` wrapper
            to truncate the environment. In Mahjax environments, some MinAtar games may not terminate within a finite timestep.
            However, the other environments are supposed to terminate within a finite timestep with probability one.
        legal_action_mask (Array): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

    current_player: Array
    rewards: Array
    terminated: Array
    truncated: Array
    legal_action_mask: Array
    _step_count: Array

    @property
    @abc.abstractmethod
    def env_id(self) -> EnvId:
        """Environment id (e.g. "go_19x19")"""
        ...


class Env(abc.ABC):
    """Environment class API.

    !!! example "Example usage"

        ```py
        env: Env = mahjax.make("no_red_4p")
        state = env.init(jax.random.PRNGKey(0))
        action = jax.random.int32(4)
        state = env.step(state, action, jax.random.PRNGKey(1))
        ```

    """

    def __init__(self): ...

    @abc.abstractmethod
    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.

        Returns:
            State: initial state of environment
        """
        ...

    @abc.abstractmethod
    def step(
        self,
        state: State,
        action: ArrayLike,
        key: Optional[PRNGKey] = None,
    ) -> State:
        """Step function.

        Args:
            state: State: Current state of the game
            action: ArrayLike: Action to be performed
            key: PRNGKey: Pseudo-random generator key in JAX. Required — a step
                that ends a round deals the next wall from it, and the state
                carries no rng of its own.

        Returns:
            State: State after processing the action
        """
        ...

    @abc.abstractmethod
    def observe(self, state: State) -> Dict[str, Array]:
        """
        Observation function.

        Args:
            state: State: Current state of the game

        Returns:
            Dict[str, Array]: Observation of the state
        """
        ...

    @property
    @abc.abstractmethod
    def id(self) -> EnvId:
        """Environment id."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (3 or 4 in Mahjong)"""
        ...

    @property
    @abc.abstractmethod
    def num_actions(self) -> int:
        """Return the size of action space"""
        ...

    @property
    @abc.abstractmethod
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """The shape of each key of the observation."""
        ...

    @property
    @abc.abstractmethod
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        ...

    @abc.abstractmethod
    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        """Step function with illegal action."""
        ...


def available_envs() -> Tuple[EnvId, ...]:
    """List up all environment id available in `mahjax.make` function.

    !!! example "Example usage"

        ```py
        mahjax.available_envs()
        ('no_red_4p', 'red_4p', 'no_red_3p', 'red_3p')
        ```
    """
    games = get_args(EnvId)
    return games


@overload
def make(env_id: Literal["no_red_4p", "red_4p"], **kwargs: Any) -> "Mahjong4p": ...


@overload
def make(env_id: Literal["no_red_3p", "red_3p"], **kwargs: Any) -> "Mahjong3p": ...


@overload
def make(env_id: EnvId, **kwargs: Any) -> "Union[Mahjong4p, Mahjong3p]": ...


def make(env_id: EnvId, **kwargs) -> "Union[Mahjong4p, Mahjong3p]":  # noqa: C901
    """Load the specified environment.

    !!! example "Example usage"

        ```py
        env = mahjax.make(
            "no_red_4p",
            round_mode="single",  # "single", "east" (tonpuusen), or "half" (hanchan)
            observe_type="dict", # "dict" for Transformer, "2D" for CNN
            order_points=[0, 0, 0, 0],  # Final score bonuses (uma), in hundreds of points
        )
        ```
    """
    from mahjax.mahjong_3p.env import Mahjong3p
    from mahjax.mahjong_4p.env import Mahjong4p

    if env_id in ("no_red_4p", "red_4p", "no_red_3p", "red_3p"):
        env_cls = Mahjong4p if env_id.endswith("_4p") else Mahjong3p
        # no_red_* is red_* with every optional rule off: no red fives, no double ron,
        # no pao and no special abortive draws.
        if env_id.startswith("no_red_"):
            return env_cls(game_config=env_cls.NO_RED, **kwargs)
        return env_cls(**kwargs)
    else:
        raise ValueError(
            f"Wrong env_id '{env_id}' is passed. Available ids are: \n{available_envs()}"
        )


def rule_based_player(state: Any, rng: PRNGKey) -> Array:
    """The rule-based player of the state's table.

    Each table has its own, in ``mahjax.mahjong_4p.players`` and
    ``mahjax.mahjong_3p.players``; this picks the one for ``state``, so one jitted
    function plays either env.

    !!! example "Example usage"

        ```py
        act = jax.jit(mahjax.rule_based_player)
        action = act(state, jax.random.PRNGKey(0))
        ```
    """
    from mahjax.mahjong_3p.players import rule_based_player as play_3p
    from mahjax.mahjong_4p.players import rule_based_player as play_4p

    play = play_3p if state.constants.NUM_PLAYERS == 3 else play_4p
    return play(state, rng)
