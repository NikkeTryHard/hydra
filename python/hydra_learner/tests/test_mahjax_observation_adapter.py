from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from hydra_learner.mahjax import observation as adapter
from hydra_learner.mahjax.shanten_bridge import exact_shanten_mask_planes, has_shanten_bridge

pytest.importorskip("jax")


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def _fixture_observation() -> dict[str, Any]:
    jnp = _jnp()
    return {
        "hand": jnp.asarray([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 34, -1, -1], dtype=jnp.int32),
        "last_draw": jnp.asarray(34, dtype=jnp.int32),
        "action_history": jnp.asarray(
            [
                [0, 0, 1, 1, 2] + [-1] * 195,
                [0, 34, 1, 1, 36] + [-1] * 195,
                [0, 1, 0, 1, 0] + [-1] * 195,
            ],
            dtype=jnp.int32,
        ),
        "dora_indicators": jnp.asarray([22, 36, -1, -1], dtype=jnp.int32),
        "scores": jnp.asarray([250, 260, 240, 200], dtype=jnp.int32),
        "shanten_count": jnp.asarray(2, dtype=jnp.int32),
        "round": jnp.asarray(3, dtype=jnp.int32),
        "honba": jnp.asarray(2, dtype=jnp.int32),
        "kyotaku": jnp.asarray(1, dtype=jnp.int32),
    }


def test_supported_channel_mask_is_fail_closed_contract() -> None:
    mask = np.asarray(adapter.supported_channel_mask_jax())

    assert mask.shape == (adapter.HYDRA_OBS_CHANNELS,)
    assert mask.dtype == np.bool_
    assert set(np.flatnonzero(mask).tolist()) == set(adapter.SUPPORTED_CHANNELS)
    assert not mask[85:].any()


def test_observation_adapter_shape_and_supported_zeros() -> None:
    result = adapter.mahjax_observation_to_hydra_jax(_fixture_observation())
    obs = np.asarray(result.obs)
    supported = np.asarray(result.supported_channel_mask)

    assert obs.shape == (192, 34)
    assert obs.dtype == np.float32
    assert supported.shape == (192,)
    assert np.all(obs[~supported] == 0.0)


def test_closed_hand_planes_fold_red_fives() -> None:
    obs = np.asarray(adapter.mahjax_observation_to_hydra_jax(_fixture_observation()).obs)

    assert obs[0, 0] == 1.0
    assert obs[0, 1] == 1.0
    assert obs[1, 1] == 1.0
    assert obs[0, 2] == 1.0
    assert obs[1, 2] == 1.0
    assert obs[2, 2] == 1.0
    assert obs[0, 3] == 1.0
    assert obs[1, 3] == 1.0
    assert obs[2, 3] == 1.0
    assert obs[3, 3] == 1.0
    # base 5m plus red 5m fold into tile kind 4 count=2.
    assert obs[0, 4] == 1.0
    assert obs[1, 4] == 1.0
    assert obs[2, 4] == 0.0


def test_aka_channels_are_hand_only_full_plane_flags() -> None:
    obs = np.asarray(adapter.mahjax_observation_to_hydra_jax(_fixture_observation()).obs)

    assert obs[40].sum() == 34.0
    assert obs[41].sum() == 0.0
    assert obs[42].sum() == 0.0


def test_discard_planes_use_relative_players_tedashi_and_temporal_decay() -> None:
    obs = np.asarray(adapter.mahjax_observation_to_hydra_jax(_fixture_observation()).obs)

    assert obs[11, 0] == 1.0
    assert obs[11, 4] == 1.0
    assert obs[12, 0] == 1.0
    assert obs[12, 4] == 0.0
    np.testing.assert_allclose(obs[13, 0], np.exp(-0.2), rtol=1e-6)
    assert obs[13, 4] == 1.0

    assert obs[14, 1] == 1.0
    assert obs[15, 1] == 1.0
    assert obs[16, 1] == 1.0

    assert obs[17, 22] == 1.0
    assert obs[18, 22] == 1.0
    assert obs[19, 22] == 1.0


def test_drawn_dora_and_metadata_planes() -> None:
    obs = np.asarray(adapter.mahjax_observation_to_hydra_jax(_fixture_observation()).obs)

    assert obs[8, 4] == 1.0
    assert obs[8].sum() == 1.0
    # Dora 5s base + red 5s fold to tile kind 22 count=2.
    assert obs[35, 22] == 1.0
    assert obs[36, 22] == 1.0
    assert obs[37, 22] == 0.0
    np.testing.assert_allclose(obs[47:51, 0], np.asarray([0.25, 0.26, 0.24, 0.20], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        obs[51:55, 0], np.asarray([0.0, -10 / 300, 10 / 300, 50 / 300], dtype=np.float32), rtol=1e-6
    )
    assert obs[57, 0] == 1.0
    assert obs[59, 0] == 3 / 8
    assert obs[60, 0] == 0.2
    assert obs[61, 0] == 0.1


def test_batch_adapter_matches_single_outputs() -> None:
    jnp = _jnp()
    one = _fixture_observation()
    two = {key: jnp.stack([value, value]) for key, value in one.items()}

    batch = adapter.mahjax_observation_to_hydra_batch_jax(two)
    single = adapter.mahjax_observation_to_hydra_jax(one)

    assert np.asarray(batch.obs).shape == (2, 192, 34)
    np.testing.assert_array_equal(np.asarray(batch.obs[0]), np.asarray(single.obs))
    np.testing.assert_array_equal(np.asarray(batch.obs[1]), np.asarray(single.obs))
    np.testing.assert_array_equal(np.asarray(batch.supported_channel_mask), np.asarray(single.supported_channel_mask))


def test_shanten_mask_channels_match_exact_rust_bridge() -> None:
    if not has_shanten_bridge():
        pytest.skip("hydra_raw_mjai_pyo3 extension is not built")
    obs = np.asarray(adapter.mahjax_observation_to_hydra_jax(_fixture_observation()).obs)
    counts = [0] * adapter.HYDRA_TILE_WIDTH
    for tile in (0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 34):
        counts[4 if tile == 34 else tile] += 1
    expected_ch9, expected_ch10 = exact_shanten_mask_planes(counts)

    np.testing.assert_array_equal(obs[9], np.asarray(expected_ch9, dtype=np.float32))
    np.testing.assert_array_equal(obs[10], np.asarray(expected_ch10, dtype=np.float32))
    assert np.asarray(adapter.supported_channel_mask_jax())[9]
    assert np.asarray(adapter.supported_channel_mask_jax())[10]


def test_diagnostic_exact_shanten_masks_fold_red_fives() -> None:
    if not has_shanten_bridge():
        pytest.skip("hydra_raw_mjai_pyo3 extension is not built")
    ch9_base, ch10_base = adapter.diagnostic_exact_shanten_mask_planes(
        [4, 4, 4, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 28]
    )
    ch9_red, ch10_red = adapter.diagnostic_exact_shanten_mask_planes(
        [4, 4, 34, 9, 10, 11, 18, 19, 20, 27, 27, 28, 28, 28]
    )

    assert ch9_base == ch9_red
    assert ch10_base == ch10_red
    assert len(ch9_base) == adapter.HYDRA_TILE_WIDTH
    assert len(ch10_base) == adapter.HYDRA_TILE_WIDTH
    assert adapter.MAHJAX_DEFAULT_BLOCKED_CHANNELS == ()


def test_supported_channel_mask_with_state_and_safety_excludes_only_known_blockers() -> None:
    # Channels 0..84 now have exact support; default blockers only return if parity regresses.
    mask = np.asarray(adapter.supported_channel_mask_jax(include_state=True, include_safety=True))
    expected = set(range(adapter.HYDRA_BASELINE_CHANNELS))

    assert set(np.flatnonzero(mask).tolist()) == expected


def test_full_baseline_required_mask_passes_after_shanten_parity() -> None:
    supported = np.asarray(adapter.supported_channel_mask_jax(include_state=True, include_safety=True))
    required = np.zeros((adapter.HYDRA_OBS_CHANNELS,), dtype=bool)
    required[: adapter.HYDRA_BASELINE_CHANNELS] = True

    adapter.assert_supported_channels(required, supported)


def test_required_mask_guard_fails_closed() -> None:
    supported = np.asarray(adapter.supported_channel_mask_jax())
    required = np.zeros((adapter.HYDRA_OBS_CHANNELS,), dtype=bool)
    required[0] = True
    adapter.assert_supported_channels(required, supported)

    required[85] = True
    with pytest.raises(ValueError, match="does not support"):
        adapter.assert_supported_channels(required, supported)


def test_live_state_riichi_channels_are_marked_and_filled_when_state_passed() -> None:
    jax = importlib.import_module("jax")
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(players=state.players.replace(riichi=_jnp().asarray([False, True, False, True])))
    observation = env.observe(state)

    result = adapter.mahjax_observation_to_hydra_jax(observation, state)
    obs = np.asarray(result.obs)
    supported = np.asarray(result.supported_channel_mask)
    current = int(state.current_player)
    expected = np.asarray([False, True, False, True], dtype=np.float32)[[(current + i) % 4 for i in range(4)]]

    assert supported[43:47].all()
    np.testing.assert_array_equal(obs[43:47, 0], expected)


def test_state_meld_channels_decode_chi_pon_kan_and_observer_open_counts() -> None:
    jax = importlib.import_module("jax")
    jnp = _jnp()
    mahjax = importlib.import_module("mahjax")
    meld_mod = importlib.import_module("mahjax.red_mahjong.meld")
    env = mahjax.make("red_mahjong", observe_type="dict")
    state = env.init(jax.random.PRNGKey(0))
    current = int(state.current_player)
    melds = np.full((4, 4), 0xFFFF, dtype=np.uint16)
    melds[current, 0] = int(meld_mod.Meld.init(jnp.asarray(78), jnp.asarray(3), jnp.asarray(1)))
    melds[(current + 1) % 4, 0] = int(meld_mod.Meld.init(jnp.asarray(75), jnp.asarray(13), jnp.asarray(2)))
    melds[(current + 2) % 4, 0] = int(meld_mod.Meld.init(jnp.asarray(77), jnp.asarray(22), jnp.asarray(3)))
    state = state.replace(players=state.players.replace(melds=jnp.asarray(melds)))
    observation = env.observe(state)

    result = adapter.mahjax_observation_to_hydra_jax(observation, state)
    obs = np.asarray(result.obs)
    supported = np.asarray(result.supported_channel_mask)

    assert supported[4:8].all()
    assert supported[23:35].all()
    assert obs[4, 3] == 1.0
    assert obs[4, 4] == 1.0
    assert obs[4, 5] == 1.0
    assert obs[23, 3] == 1.0
    assert obs[23, 4] == 1.0
    assert obs[23, 5] == 1.0
    assert obs[27, 13] == 1.0
    assert obs[31, 22] == 1.0


def test_live_mahjax_observation_smoke() -> None:
    jax = importlib.import_module("jax")
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    state = env.init(jax.random.PRNGKey(0))
    observation = env.observe(state)

    result = adapter.mahjax_observation_to_hydra_jax(observation)

    assert tuple(result.obs.shape) == (192, 34)
    assert tuple(result.supported_channel_mask.shape) == (192,)
    assert bool(np.isfinite(np.asarray(result.obs)).all())
