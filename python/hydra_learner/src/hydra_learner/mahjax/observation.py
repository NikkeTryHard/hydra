from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hydra_learner.mahjax import safety as mahjax_safety_adapter
from hydra_learner.mahjax.shanten import hydra_discard_shanten_masks_jax
from hydra_learner.mahjax.shanten_bridge import exact_shanten_mask_planes

HYDRA_OBS_CHANNELS: Final = 192
HYDRA_TILE_WIDTH: Final = 34
HYDRA_BASELINE_CHANNELS: Final = 85
MAHJAX_DEFAULT_BLOCKED_CHANNELS: Final = ()
TEMPORAL_DECAY_TABLE = np.exp(np.arange(31, dtype=np.float32) * np.float32(-0.2))


CH_HAND_START: Final = 0
CH_DRAWN: Final = 8
CH_SHANTEN_MASK: Final = 9
CH_DISCARDS_START: Final = 11
CH_MELDS_START: Final = 23
CH_DORA_START: Final = 35
CH_RIICHI_START: Final = 43
CH_META_START: Final = 43

STATE_CHANNELS: Final = (
    4,
    5,
    6,
    7,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    43,
    44,
    45,
    46,
)
SUPPORTED_CHANNELS: Final = (
    0,
    1,
    2,
    3,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
)
SAFETY_CHANNELS: Final = mahjax_safety_adapter.MAHJAX_EXACT_SAFETY_CHANNELS
SUPPORTED_CHANNELS_WITH_SAFETY: Final = tuple(sorted((*SUPPORTED_CHANNELS, *SAFETY_CHANNELS)))
SUPPORTED_CHANNELS_WITH_STATE_AND_SAFETY: Final = tuple(
    sorted((*SUPPORTED_CHANNELS, *STATE_CHANNELS, *SAFETY_CHANNELS))
)

SUPPORTED_CHANNELS_WITH_STATE: Final = tuple(sorted((*SUPPORTED_CHANNELS, *STATE_CHANNELS)))


@dataclass(frozen=True)
class HydraObsAdapterResult:
    obs: Any
    supported_channel_mask: Any


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def _jax() -> Any:
    return importlib.import_module("jax")


def _base_tile_jax(tile: Any) -> Any:
    jnp = _jnp()
    tile_i = jnp.asarray(tile, dtype=jnp.int32)
    return jnp.select(
        [tile_i == 34, tile_i == 35, tile_i == 36],
        [jnp.asarray(4, dtype=jnp.int32), jnp.asarray(13, dtype=jnp.int32), jnp.asarray(22, dtype=jnp.int32)],
        default=tile_i,
    )


def _counts_from_red_tile_ids_jax(tile_ids: Any) -> Any:
    jnp = _jnp()
    tiles = jnp.asarray(tile_ids, dtype=jnp.int32)
    valid = (tiles >= 0) & (tiles <= 36)
    base = _base_tile_jax(tiles)
    safe_base = jnp.where(valid, base, 0)
    return jnp.bincount(safe_base, weights=valid.astype(jnp.int32), length=HYDRA_TILE_WIDTH).astype(jnp.float32)


def diagnostic_exact_shanten_mask_planes(tile_ids: list[int]) -> tuple[list[float], list[float]]:
    """Return exact channel 9/10 planes through the Rust bridge; diagnostic CPU path only."""
    # Do not wire this into mahjax-gpu rollout: per-hand CPU bridge calls would
    # add host synchronization and break the device-resident rollout contract.
    counts = [0] * HYDRA_TILE_WIDTH
    for tile_id in tile_ids:
        if 0 <= tile_id <= 36:
            base = 4 if tile_id == 34 else 13 if tile_id == 35 else 22 if tile_id == 36 else tile_id
            counts[base] += 1
    return exact_shanten_mask_planes(counts)


def _threshold_planes_jax(counts34: Any) -> Any:
    jnp = _jnp()
    counts = jnp.asarray(counts34, dtype=jnp.float32)
    return jnp.stack([counts >= 1.0, counts >= 2.0, counts >= 3.0, counts >= 4.0]).astype(jnp.float32)


def _shanten_mask_planes_jax(counts34: Any) -> Any:
    jnp = _jnp()
    _, non_increase, decrease = hydra_discard_shanten_masks_jax(counts34)
    return jnp.stack([non_increase, decrease]).astype(jnp.float32)


def _aka_flags_from_red_tile_ids_jax(tile_ids: Any) -> Any:
    jnp = _jnp()
    tiles = jnp.asarray(tile_ids, dtype=jnp.int32)
    return jnp.asarray([jnp.any(tiles == 34), jnp.any(tiles == 35), jnp.any(tiles == 36)], dtype=jnp.float32)


def _discard_planes_jax(action_history: Any) -> Any:
    jnp = _jnp()
    history = jnp.asarray(action_history, dtype=jnp.int32)
    players = history[0]
    tiles = history[1]
    tsumogiri = history[2]
    idx = jnp.arange(history.shape[1], dtype=jnp.int32)
    is_discard = (players >= 0) & (players < 4) & (tiles >= 0) & (tiles <= 36) & ((tsumogiri == 0) | (tsumogiri == 1))
    base_tiles = _base_tile_jax(tiles)
    safe_tiles = jnp.where(is_discard, base_tiles, 0)

    planes = []
    for player in range(4):
        player_discard = is_discard & (players == player)
        counts = jnp.bincount(safe_tiles, weights=player_discard.astype(jnp.int32), length=HYDRA_TILE_WIDTH)
        presence = counts > 0
        tedashi_counts = jnp.bincount(
            safe_tiles,
            weights=(player_discard & (tsumogiri == 0)).astype(jnp.int32),
            length=HYDRA_TILE_WIDTH,
        )
        tedashi = tedashi_counts > 0
        t_max = jnp.max(jnp.where(player_discard, idx, 0))
        age = jnp.minimum(jnp.maximum(t_max - idx, 0), 30)
        weights = jnp.asarray(TEMPORAL_DECAY_TABLE)[age]
        temporal = jnp.zeros((HYDRA_TILE_WIDTH,), dtype=jnp.float32)
        temporal = temporal.at[safe_tiles].max(jnp.where(player_discard, weights, 0.0))
        planes.extend([presence.astype(jnp.float32), tedashi.astype(jnp.float32), temporal])
    return jnp.stack(planes)


def _meld_action_jax(meld: Any) -> Any:
    jnp = _jnp()
    meld_i = jnp.asarray(meld, dtype=jnp.uint16)
    return jnp.where(meld_i == jnp.uint16(0xFFFF), jnp.int32(-1), (meld_i & jnp.uint16(0b1111111)).astype(jnp.int32))


def _meld_target_jax(meld: Any) -> Any:
    jnp = _jnp()
    meld_i = jnp.asarray(meld, dtype=jnp.uint16)
    return jnp.where(
        meld_i == jnp.uint16(0xFFFF),
        jnp.int32(-1),
        ((meld_i >> jnp.uint16(7)) & jnp.uint16(0b111111)).astype(jnp.int32),
    )


def _meld_chi_start_jax(action: Any, target: Any) -> Any:
    jnp = _jnp()
    chi_index = jnp.select(
        [(action == 78) | (action == 79), (action == 80) | (action == 81), (action == 82) | (action == 83)],
        [jnp.int32(0), jnp.int32(1), jnp.int32(2)],
        default=jnp.int32(-1),
    )
    return target - chi_index


def _state_meld_planes_jax(state: Any) -> tuple[Any, Any]:
    jnp = _jnp()
    current = jnp.asarray(state.current_player, dtype=jnp.int32)
    relative_players = (current + jnp.arange(4, dtype=jnp.int32)) % 4
    melds = jnp.take(jnp.asarray(state.players.melds, dtype=jnp.uint16), relative_players, axis=0)

    open_counts = jnp.zeros((HYDRA_TILE_WIDTH,), dtype=jnp.float32)
    meld_planes = jnp.zeros((12, HYDRA_TILE_WIDTH), dtype=jnp.float32)
    for player in range(4):
        for slot in range(4):
            meld = melds[player, slot]
            action = _meld_action_jax(meld)
            target = _meld_target_jax(meld)
            valid_target = (target >= 0) & (target < HYDRA_TILE_WIDTH)
            is_chi = ((action >= 78) & (action <= 83)) & valid_target
            is_pon = ((action == 75) | (action == 76)) & valid_target
            is_kan = ((action == 77) | ((action >= 37) & (action <= 70))) & valid_target
            chi_start = _meld_chi_start_jax(action, target)
            chi_tiles = chi_start + jnp.arange(3, dtype=jnp.int32)
            chi_valid = is_chi & (chi_start >= 0) & (chi_start + 2 < 27)
            meld_planes = meld_planes.at[player * 3 + 0, chi_tiles].set(jnp.where(chi_valid, 1.0, 0.0))
            meld_planes = meld_planes.at[player * 3 + 1, target].set(jnp.where(is_pon, 1.0, 0.0))
            meld_planes = meld_planes.at[player * 3 + 2, target].set(jnp.where(is_kan, 1.0, 0.0))
            if player == 0:
                open_counts = open_counts.at[chi_tiles].add(jnp.where(chi_valid, 1.0, 0.0))
                open_counts = open_counts.at[target].add(jnp.where(is_pon, 3.0, 0.0))
                open_counts = open_counts.at[target].add(jnp.where(is_kan, 4.0, 0.0))
    return _threshold_planes_jax(open_counts), meld_planes


def _relative_riichi_jax(state: Any) -> Any:
    jnp = _jnp()
    current = jnp.asarray(state.current_player, dtype=jnp.int32)
    relative_players = (current + jnp.arange(4, dtype=jnp.int32)) % 4
    return jnp.take(jnp.asarray(state.players.riichi, dtype=jnp.bool_), relative_players).astype(jnp.float32)


def supported_channel_mask_jax(*, include_state: bool = False, include_safety: bool = False) -> Any:
    jnp = _jnp()
    if include_state and include_safety:
        channels = SUPPORTED_CHANNELS_WITH_STATE_AND_SAFETY
    elif include_safety:
        channels = SUPPORTED_CHANNELS_WITH_SAFETY
    elif include_state:
        channels = SUPPORTED_CHANNELS_WITH_STATE
    else:
        channels = SUPPORTED_CHANNELS
    mask = jnp.zeros((HYDRA_OBS_CHANNELS,), dtype=jnp.bool_)
    return mask.at[jnp.asarray(channels, dtype=jnp.int32)].set(True)


def assert_supported_channels(required_mask: Any, supported_mask: Any) -> None:
    required = np.asarray(required_mask, dtype=bool)
    supported = np.asarray(supported_mask, dtype=bool)
    if required.shape != (HYDRA_OBS_CHANNELS,) or supported.shape != (HYDRA_OBS_CHANNELS,):
        raise ValueError(f"channel masks must have shape ({HYDRA_OBS_CHANNELS},)")
    missing = np.flatnonzero(required & ~supported)
    if missing.size:
        raise ValueError(f"MahJAX observation adapter does not support required Hydra channels: {missing.tolist()}")


def mahjax_observation_to_hydra_jax(
    observation: Any,
    state: Any | None = None,
    safety_state: mahjax_safety_adapter.MahjaxSafetyState | None = None,
) -> HydraObsAdapterResult:
    jnp = _jnp()
    obs = jnp.zeros((HYDRA_OBS_CHANNELS, HYDRA_TILE_WIDTH), dtype=jnp.float32)

    hand_counts = _counts_from_red_tile_ids_jax(observation["hand"])
    obs = obs.at[CH_HAND_START : CH_HAND_START + 4, :].set(_threshold_planes_jax(hand_counts))
    obs = obs.at[CH_SHANTEN_MASK : CH_SHANTEN_MASK + 2, :].set(_shanten_mask_planes_jax(hand_counts))

    last_draw = jnp.asarray(observation["last_draw"], dtype=jnp.int32)
    last_draw_valid = (last_draw >= 0) & (last_draw <= 36)
    drawn_base = _base_tile_jax(last_draw)
    drawn_row = (jnp.arange(HYDRA_TILE_WIDTH, dtype=jnp.int32) == drawn_base) & last_draw_valid
    obs = obs.at[CH_DRAWN, :].set(drawn_row.astype(jnp.float32))

    obs = obs.at[CH_DISCARDS_START : CH_DISCARDS_START + 12, :].set(_discard_planes_jax(observation["action_history"]))

    dora_counts = _counts_from_red_tile_ids_jax(observation["dora_indicators"])
    dora_planes = jnp.stack(
        [dora_counts >= 1.0, dora_counts >= 2.0, dora_counts >= 3.0, dora_counts >= 4.0, dora_counts >= 5.0]
    ).astype(jnp.float32)
    obs = obs.at[CH_DORA_START : CH_DORA_START + 5, :].set(dora_planes)
    aka_flags = _aka_flags_from_red_tile_ids_jax(observation["hand"])
    obs = obs.at[40:43, :].set(aka_flags[:, None].repeat(HYDRA_TILE_WIDTH, axis=1))

    if state is not None:
        open_meld_planes, meld_planes = _state_meld_planes_jax(state)
        obs = obs.at[4:8, :].set(open_meld_planes)
        obs = obs.at[CH_MELDS_START : CH_MELDS_START + 12, :].set(meld_planes)
        obs = obs.at[CH_RIICHI_START : CH_RIICHI_START + 4, :].set(
            _relative_riichi_jax(state)[:, None].repeat(HYDRA_TILE_WIDTH, axis=1)
        )

    if safety_state is not None:
        obs = obs.at[62:85, :].set(mahjax_safety_adapter.encode_safety_channels_jax(safety_state))

    scores = jnp.asarray(observation["scores"], dtype=jnp.float32)
    obs = obs.at[CH_META_START + 4 : CH_META_START + 8, :].set(
        (scores[:, None] / 1000.0).repeat(HYDRA_TILE_WIDTH, axis=1)
    )
    score_gaps = (scores[0] - scores) / 300.0
    obs = obs.at[CH_META_START + 8 : CH_META_START + 12, :].set(score_gaps[:, None].repeat(HYDRA_TILE_WIDTH, axis=1))

    shanten = jnp.clip(jnp.asarray(observation["shanten_count"], dtype=jnp.int32), 0, 3)
    shanten_rows = (jnp.arange(4, dtype=jnp.int32) == shanten).astype(jnp.float32)
    obs = obs.at[CH_META_START + 12 : CH_META_START + 16, :].set(shanten_rows[:, None].repeat(HYDRA_TILE_WIDTH, axis=1))

    obs = obs.at[CH_META_START + 16, :].set(jnp.asarray(observation["round"], dtype=jnp.float32) / 8.0)
    obs = obs.at[CH_META_START + 17, :].set(jnp.asarray(observation["honba"], dtype=jnp.float32) / 10.0)
    obs = obs.at[CH_META_START + 18, :].set(jnp.asarray(observation["kyotaku"], dtype=jnp.float32) / 10.0)

    return HydraObsAdapterResult(
        obs=obs,
        supported_channel_mask=supported_channel_mask_jax(
            include_state=state is not None, include_safety=safety_state is not None
        ),
    )


def _mahjax_observation_to_hydra_batch_state_safety_jax(
    observation: Any, state: Any, safety_state: mahjax_safety_adapter.MahjaxSafetyState
) -> Any:
    jax = _jax()
    jnp = _jnp()
    batch = observation["hand"].shape[0]
    obs = jnp.zeros((batch, HYDRA_OBS_CHANNELS, HYDRA_TILE_WIDTH), dtype=jnp.float32)

    hand_counts = jax.vmap(_counts_from_red_tile_ids_jax)(observation["hand"])
    obs = obs.at[:, CH_HAND_START : CH_HAND_START + 4, :].set(jax.vmap(_threshold_planes_jax)(hand_counts))
    obs = obs.at[:, CH_SHANTEN_MASK : CH_SHANTEN_MASK + 2, :].set(jax.vmap(_shanten_mask_planes_jax)(hand_counts))

    last_draw = jnp.asarray(observation["last_draw"], dtype=jnp.int32)
    last_draw_valid = (last_draw >= 0) & (last_draw <= 36)
    drawn_base = _base_tile_jax(last_draw)
    drawn_row = (jnp.arange(HYDRA_TILE_WIDTH, dtype=jnp.int32)[None, :] == drawn_base[:, None]) & last_draw_valid[
        :, None
    ]
    obs = obs.at[:, CH_DRAWN, :].set(drawn_row.astype(jnp.float32))

    obs = obs.at[:, CH_DISCARDS_START : CH_DISCARDS_START + 12, :].set(
        jax.vmap(_discard_planes_jax)(observation["action_history"])
    )
    open_meld_planes, meld_planes = jax.vmap(_state_meld_planes_jax)(state)
    obs = obs.at[:, 4:8, :].set(open_meld_planes)
    obs = obs.at[:, CH_MELDS_START : CH_MELDS_START + 12, :].set(meld_planes)

    dora_counts = jax.vmap(_counts_from_red_tile_ids_jax)(observation["dora_indicators"])
    dora_planes = jnp.stack(
        [
            dora_counts >= 1.0,
            dora_counts >= 2.0,
            dora_counts >= 3.0,
            dora_counts >= 4.0,
            dora_counts >= 5.0,
        ],
        axis=1,
    ).astype(jnp.float32)
    obs = obs.at[:, CH_DORA_START : CH_DORA_START + 5, :].set(dora_planes)

    aka_flags = jax.vmap(_aka_flags_from_red_tile_ids_jax)(observation["hand"])
    obs = obs.at[:, 40:43, :].set(aka_flags[:, :, None].repeat(HYDRA_TILE_WIDTH, axis=2))
    obs = obs.at[:, CH_RIICHI_START : CH_RIICHI_START + 4, :].set(
        jax.vmap(_relative_riichi_jax)(state)[:, :, None].repeat(HYDRA_TILE_WIDTH, axis=2)
    )
    obs = obs.at[:, 62:85, :].set(jax.vmap(mahjax_safety_adapter.encode_safety_channels_jax)(safety_state))

    scores = jnp.asarray(observation["scores"], dtype=jnp.float32)
    obs = obs.at[:, CH_META_START + 4 : CH_META_START + 8, :].set(
        (scores[:, :, None] / 1000.0).repeat(HYDRA_TILE_WIDTH, axis=2)
    )
    score_gaps = (scores[:, 0:1] - scores) / 300.0
    obs = obs.at[:, CH_META_START + 8 : CH_META_START + 12, :].set(
        score_gaps[:, :, None].repeat(HYDRA_TILE_WIDTH, axis=2)
    )

    shanten = jnp.clip(jnp.asarray(observation["shanten_count"], dtype=jnp.int32), 0, 3)
    shanten_rows = (jnp.arange(4, dtype=jnp.int32)[None, :] == shanten[:, None]).astype(jnp.float32)
    obs = obs.at[:, CH_META_START + 12 : CH_META_START + 16, :].set(
        shanten_rows[:, :, None].repeat(HYDRA_TILE_WIDTH, axis=2)
    )
    obs = obs.at[:, CH_META_START + 16, :].set(jnp.asarray(observation["round"], dtype=jnp.float32)[:, None] / 8.0)
    obs = obs.at[:, CH_META_START + 17, :].set(jnp.asarray(observation["honba"], dtype=jnp.float32)[:, None] / 10.0)
    return obs.at[:, CH_META_START + 18, :].set(jnp.asarray(observation["kyotaku"], dtype=jnp.float32)[:, None] / 10.0)


def mahjax_observation_to_hydra_batch_jax(
    observation: Any,
    state: Any | None = None,
    safety_state: mahjax_safety_adapter.MahjaxSafetyState | None = None,
) -> HydraObsAdapterResult:
    jax = _jax()
    if state is None and safety_state is None:
        obs = jax.vmap(lambda row: mahjax_observation_to_hydra_jax(row).obs)(observation)
    elif safety_state is None:
        obs = jax.vmap(lambda row, state_row: mahjax_observation_to_hydra_jax(row, state_row).obs)(observation, state)
    elif state is None:
        obs = jax.vmap(lambda row, safety_row: mahjax_observation_to_hydra_jax(row, None, safety_row).obs)(
            observation, safety_state
        )
    else:
        obs = _mahjax_observation_to_hydra_batch_state_safety_jax(observation, state, safety_state)
    return HydraObsAdapterResult(
        obs=obs,
        supported_channel_mask=supported_channel_mask_jax(
            include_state=state is not None, include_safety=safety_state is not None
        ),
    )
