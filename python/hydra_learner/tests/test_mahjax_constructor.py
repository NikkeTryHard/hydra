from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
import pytest

from hydra_learner.mahjax.compat import (
    HYDRA_ACTION_SPACE,
    MAHJAX_CHI_LEFT,
    MAHJAX_CHI_LEFT_RED,
    MAHJAX_CHI_MID,
    MAHJAX_CHI_RIGHT,
    MAHJAX_OPEN_KAN,
    MAHJAX_PASS,
    MAHJAX_PON,
    MAHJAX_PON_RED,
    MAHJAX_RIICHI,
    MAHJAX_RON,
    MAHJAX_SELF_KAN_START,
    MAHJAX_TSUMO,
    MAHJAX_TSUMOGIRI,
)
from hydra_learner.mahjax.constructor import (
    apply_mjai_event_to_mahjax_state,
    mahjax_action_from_mjai_chi,
    mahjax_action_from_mjai_dahai,
    mahjax_action_from_mjai_hora,
    mahjax_action_from_mjai_kan,
    mahjax_action_from_mjai_pon,
    mahjax_action_from_mjai_reach,
    mahjax_action_from_mjai_self_kan,
    mahjax_state_from_start_kyoku,
    mjai_tile_to_mahjax_id,
)
from hydra_learner.mahjax.jax_compat import mahjax_mask_to_hydra_jax
from hydra_learner.mahjax.observation import mahjax_observation_to_hydra_jax

if TYPE_CHECKING:
    from hydra_learner.typing_boundaries import MahjaxEnv, MahjaxState


@pytest.fixture(scope="module")
def mahjax_env() -> MahjaxEnv:
    mahjax = importlib.import_module("mahjax")
    return mahjax.make("red_mahjong", observe_type="dict")


pytest.importorskip("jax")


def test_mjai_tile_to_mahjax_id() -> None:
    assert mjai_tile_to_mahjax_id("1m") == 0
    assert mjai_tile_to_mahjax_id("9m") == 8
    assert mjai_tile_to_mahjax_id("1p") == 9
    assert mjai_tile_to_mahjax_id("1s") == 18
    assert mjai_tile_to_mahjax_id("E") == 27
    assert mjai_tile_to_mahjax_id("5mr") == 34
    assert mjai_tile_to_mahjax_id("5pr") == 35
    assert mjai_tile_to_mahjax_id("5sr") == 36


START_TEHAIS = [
    ["1m", "2m", "3m", "4m", "5m", "5mr", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
    ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N"],
    ["P", "F", "C", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m"],
    ["6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s", "1s", "2s", "2s", "3s"],
]

PARITY_JSONL = """
{"type":"start_game","names":["a","b","c","d"],"id":"mahjax-parity"}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","5mr","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}
{"type":"tsumo","actor":0,"pai":"5sr"}
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
{"type":"none"}
""".strip()

PARITY_AUTHORITY_JSON = """
{"action_space":46,"rows":[{"index":0,"action_id":0,"legal_mask":[1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0]},{"index":1,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":2,"action_id":0,"legal_mask":[1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":3,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]},{"index":4,"action_id":0,"legal_mask":[0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0]},{"index":5,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":6,"action_id":0,"legal_mask":[1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":7,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]},{"index":8,"action_id":0,"legal_mask":[0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0]},{"index":9,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":10,"action_id":0,"legal_mask":[1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":11,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]},{"index":12,"action_id":0,"legal_mask":[0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0]},{"index":13,"action_id":0,"legal_mask":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]},{"index":14,"action_id":0,"legal_mask":[1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]}]}
""".strip()


def _parse_mjai_jsonl(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line]


def _parse_authority_rows(text: str) -> dict[int, UInt8Array]:
    data = json.loads(text)
    assert data["action_space"] == HYDRA_ACTION_SPACE
    rows: dict[int, UInt8Array] = {}
    for row in data["rows"]:
        index = int(row["index"])
        assert index not in rows
        mask = np.asarray(row["legal_mask"], dtype=np.uint8)
        assert mask.shape == (HYDRA_ACTION_SPACE,)
        assert bool(mask.any())
        rows[index] = mask
    assert set(rows) == set(range(len(rows)))
    return rows


def _require_sequence_of_strings(value: object, name: str) -> Sequence[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{name} entries must be strings")
    return value


def _require_nested_strings(value: object, name: str) -> Sequence[Sequence[str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    return [_require_sequence_of_strings(item, name) for item in value]


def _require_ints(value: object, name: str) -> Sequence[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be a sequence")
    out: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(f"{name} entries must be ints")
        out.append(item)
    return out


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    return value


def _assert_projected_mask_matches_authority(state: MahjaxState, rows: dict[int, UInt8Array], index: int) -> None:
    np.testing.assert_array_equal(_projected_mask(state), rows[index])


def _state_from_start_kyoku_event(event: Mapping[str, object], *, first_draw: str) -> MahjaxState:
    return mahjax_state_from_start_kyoku(
        tehais=_require_nested_strings(event["tehais"], "tehais"),
        scores=_require_ints(event["scores"], "scores"),
        dora_marker=str(event["dora_marker"]),
        oya=_require_int(event["oya"], "oya"),
        kyoku=_require_int(event["kyoku"], "kyoku"),
        honba=_require_int(event["honba"], "honba"),
        kyotaku=_require_int(event["kyotaku"], "kyotaku"),
        first_draw=first_draw,
    )


def _apply_mjai_events(env: MahjaxEnv, state: MahjaxState, events: list[dict[str, object]]) -> MahjaxState:
    for event in events:
        state = apply_mjai_event_to_mahjax_state(env, state, event)
    return state


def _projected_mask(state: MahjaxState) -> UInt8Array:
    return np.asarray(mahjax_mask_to_hydra_jax(state.legal_action_mask, state.round_state.last_draw), dtype=np.uint8)


def _apply_implicit_passes(env: MahjaxEnv, state: MahjaxState) -> MahjaxState:
    while bool(state.legal_action_mask[MAHJAX_PASS]):
        state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})
    return state


def _assert_projected_mask_has_ones(state: MahjaxState, expected_ones: list[int]) -> None:
    expected = np.zeros((HYDRA_ACTION_SPACE,), dtype=np.uint8)
    expected[np.asarray(expected_ones, dtype=np.int32)] = 1
    np.testing.assert_array_equal(_projected_mask(state), expected)


@pytest.mark.slow
def test_real_mjai_replay_prefix_projected_masks_match_hydra_authority(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["3m", "5m", "7m", "8m", "4p", "5p", "6p", "2s", "4s", "7s", "P", "F", "C"],
            ["2m", "5mr", "7m", "1p", "3p", "8p", "1s", "1s", "8s", "9s", "S", "N", "P"],
            ["7m", "8m", "1p", "6p", "7p", "8p", "1s", "4s", "7s", "8s", "9s", "9s", "W"],
            ["1m", "2m", "2m", "4m", "9m", "2p", "3p", "4p", "2s", "4s", "6s", "6s", "F"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="S",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="7p",
    )
    deck = state.round_state.deck
    for index, tile in zip(
        range(82, 62, -1),
        [
            "4p",
            "9p",
            "2s",
            "4m",
            "1p",
            "5pr",
            "3s",
            "2m",
            "5p",
            "6m",
            "7p",
            "5s",
            "N",
            "2s",
            "1s",
            "E",
            "W",
            "4p",
            "5m",
            "2p",
        ],
        strict=True,
    ):
        deck = deck.at[index].set(mjai_tile_to_mahjax_id(tile))
    state = state.replace(round_state=state.round_state.replace(deck=deck))

    expected_masks = [
        [2, 4, 6, 7, 12, 13, 14, 15, 19, 21, 24, 31, 32, 33],
        [1, 6, 9, 11, 12, 16, 18, 25, 26, 28, 30, 31, 34],
        [6, 7, 9, 14, 15, 16, 17, 18, 21, 24, 25, 26, 29],
        [0, 1, 3, 8, 10, 11, 12, 19, 21, 23, 32],
        [2, 3, 4, 6, 7, 12, 13, 14, 15, 19, 21, 24, 32, 33],
        [1, 6, 9, 11, 12, 16, 18, 25, 26, 30, 31, 34],
        [6, 7, 9, 14, 15, 16, 17, 21, 24, 25, 26, 29, 35],
        [0, 1, 3, 10, 11, 12, 19, 20, 21, 23, 32],
        [1, 2, 3, 4, 6, 7, 12, 13, 14, 15, 19, 21, 24, 32],
        [1, 6, 9, 11, 12, 13, 16, 18, 25, 26, 30, 34],
        [5, 6, 7, 14, 15, 16, 17, 21, 24, 25, 26, 29, 35],
        [1, 3, 10, 11, 12, 15, 19, 20, 21, 23, 32],
        [1, 2, 3, 4, 6, 7, 12, 13, 14, 15, 19, 21, 22, 24],
        [1, 6, 9, 11, 12, 13, 16, 18, 25, 26, 30, 34],
        [5, 6, 7, 14, 15, 16, 17, 19, 24, 25, 26, 29, 35],
        [1, 3, 10, 11, 12, 15, 18, 19, 20, 21, 23],
        [1, 2, 3, 4, 6, 7, 12, 13, 14, 15, 21, 22, 24, 27],
        [1, 6, 9, 11, 12, 13, 18, 25, 26, 29, 30, 34],
        [26, 29],
        [1, 3, 4, 10, 11, 12, 15, 18, 19, 20, 21, 23],
        [1, 2, 3, 4, 6, 7, 10, 12, 13, 14, 15, 21, 22, 24],
    ]
    steps = [
        ([{"type": "dahai", "actor": 0, "pai": "P", "tsumogiri": False}], expected_masks[1]),
        ([{"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": False}], expected_masks[2]),
        ([{"type": "dahai", "actor": 2, "pai": "1s", "tsumogiri": False}], expected_masks[3]),
        ([{"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": False}], expected_masks[4]),
        ([{"type": "dahai", "actor": 0, "pai": "C", "tsumogiri": False}], expected_masks[5]),
        ([{"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": False}], expected_masks[6]),
        ([{"type": "dahai", "actor": 2, "pai": "1p", "tsumogiri": False}], expected_masks[7]),
        ([{"type": "dahai", "actor": 3, "pai": "1m", "tsumogiri": False}], expected_masks[8]),
        ([{"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": False}], expected_masks[9]),
        ([{"type": "dahai", "actor": 1, "pai": "N", "tsumogiri": False}], expected_masks[10]),
        ([{"type": "dahai", "actor": 2, "pai": "4s", "tsumogiri": False}], expected_masks[11]),
        ([{"type": "dahai", "actor": 3, "pai": "F", "tsumogiri": False}], expected_masks[12]),
        ([{"type": "dahai", "actor": 0, "pai": "2s", "tsumogiri": False}], expected_masks[13]),
        ([{"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": False}], expected_masks[14]),
        ([{"type": "dahai", "actor": 2, "pai": "2s", "tsumogiri": True}], expected_masks[15]),
        ([{"type": "dahai", "actor": 3, "pai": "2s", "tsumogiri": False}], expected_masks[16]),
        ([{"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True}], expected_masks[17]),
        ([{"type": "dahai", "actor": 1, "pai": "2m", "tsumogiri": False}], None),
        ([{"type": "reach", "actor": 2}], expected_masks[18]),
        ([{"type": "dahai", "actor": 2, "pai": "W", "tsumogiri": False}], expected_masks[19]),
        (
            [
                {"type": "reach_accepted", "actor": 2},
                {"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False},
            ],
            expected_masks[20],
        ),
    ]

    _assert_projected_mask_has_ones(state, expected_masks[0])
    for events, expected in steps:
        for event in events:
            state = apply_mjai_event_to_mahjax_state(env, state, event)
        state = _apply_implicit_passes(env, state)
        if expected is not None:
            _assert_projected_mask_has_ones(state, expected)


def test_start_kyoku_constructor_sets_observable_fields(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=START_TEHAIS,
        scores=[25000, 26000, 24000, 20000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=2,
        kyotaku=1,
        first_draw="5sr",
    )
    obs = env.observe(state)
    hydra_obs = np.asarray(mahjax_observation_to_hydra_jax(obs, state).obs)

    assert int(state.current_player) == 0
    assert int(state.round_state.honba) == 2
    assert int(state.round_state.kyotaku) == 1
    assert int(state.round_state.score[1]) == 260
    assert int(state.round_state.last_draw) == 36
    assert bool(state.legal_action_mask.any())
    assert hydra_obs.shape == (192, 34)
    assert hydra_obs[8, 22] == 1.0
    assert hydra_obs[40].sum() == 34.0
    assert bool(np.isfinite(hydra_obs).all())


class _Round:
    def __init__(self, *, target: int, last_draw: int = -1) -> None:
        self.target = target
        self.last_draw = last_draw
        self.terminated_round = False

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


class _State:
    def __init__(
        self, legal_actions: list[int], *, target: int = 0, last_draw: int = -1, current_player: int = 0
    ) -> None:
        self.current_player = current_player
        self.legal_action_mask = np.zeros(87, dtype=np.bool_)
        self.legal_action_mask[legal_actions] = True
        self.round_state = _Round(target=target, last_draw=last_draw)
        self.players = type("_Players", (), {"legal_action_mask": self.legal_action_mask})()

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


def test_mjai_call_and_terminal_action_translation_is_fail_closed() -> None:
    state = _State([MAHJAX_PON, MAHJAX_PON_RED, MAHJAX_OPEN_KAN], target=mjai_tile_to_mahjax_id("5m"))
    assert mahjax_action_from_mjai_pon(state, pai="5m", consumed=["5m", "5m"]) == MAHJAX_PON
    assert mahjax_action_from_mjai_pon(state, pai="5m", consumed=["5m", "5mr"]) == MAHJAX_PON_RED
    assert mahjax_action_from_mjai_kan(state, pai="5m", consumed=["5m", "5m", "5mr"]) == MAHJAX_OPEN_KAN
    with pytest.raises(ValueError, match="response target"):
        mahjax_action_from_mjai_pon(state, pai="6m", consumed=["6m", "6m"])
    with pytest.raises(ValueError, match="exactly three"):
        mahjax_action_from_mjai_kan(state, pai="5m", consumed=["5m", "5m"])

    chi_state = _State([MAHJAX_CHI_LEFT, MAHJAX_CHI_LEFT_RED, MAHJAX_CHI_MID, MAHJAX_CHI_RIGHT], target=3)
    assert mahjax_action_from_mjai_chi(chi_state, pai="4m", consumed=["5m", "6m"]) == MAHJAX_CHI_LEFT
    assert mahjax_action_from_mjai_chi(chi_state, pai="4m", consumed=["5mr", "6m"]) == MAHJAX_CHI_LEFT_RED
    assert mahjax_action_from_mjai_chi(chi_state, pai="4m", consumed=["3m", "5m"]) == MAHJAX_CHI_MID
    assert mahjax_action_from_mjai_chi(chi_state, pai="4m", consumed=["2m", "3m"]) == MAHJAX_CHI_RIGHT
    with pytest.raises(ValueError, match="sequence"):
        mahjax_action_from_mjai_chi(chi_state, pai="4m", consumed=["6m", "7m"])

    assert mahjax_action_from_mjai_self_kan(_State([MAHJAX_SELF_KAN_START + 4]), pai="5m", consumed=()) == (
        MAHJAX_SELF_KAN_START + 4
    )
    assert mahjax_action_from_mjai_self_kan(
        _State([MAHJAX_SELF_KAN_START + 4]), pai=None, consumed=["5mr", "5m", "5m", "5m"]
    ) == (MAHJAX_SELF_KAN_START + 4)
    assert mahjax_action_from_mjai_reach(_State([MAHJAX_RIICHI])) == MAHJAX_RIICHI
    assert mahjax_action_from_mjai_hora(_State([MAHJAX_TSUMO]), actor=2, target=2) == MAHJAX_TSUMO
    assert mahjax_action_from_mjai_hora(_State([MAHJAX_RON]), actor=2, target=1) == MAHJAX_RON


@pytest.mark.slow
def test_mjai_event_applier_applies_real_chi_response_window(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["4p", "4p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1s", "2s"],
            ["2m", "3m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s"],
            ["9s", "9s", "9s", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p"],
            ["E", "E", "E", "E", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="4p",
    )

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "4p", "tsumogiri": True})
    assert int(state.current_player) == 1
    assert int(state.round_state.target) == mjai_tile_to_mahjax_id("4p")

    state = apply_mjai_event_to_mahjax_state(
        env,
        state,
        {"type": "chi", "pai": "4p", "consumed": ["3p", "5p"]},
    )
    assert int(state.current_player) == 1
    assert int(state.players.meld_counts[1]) == 1
    assert not bool(state.round_state.draw_next)


@pytest.mark.slow
def test_mjai_event_applier_applies_real_pon_response_window(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["4p", "4p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S"],
            ["4p", "4p", "5p", "5p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "P"],
            ["4p", "4p", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m", "C"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="4p",
    )

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "4p", "tsumogiri": True})
    assert int(state.current_player) == 1
    assert int(state.round_state.target) == mjai_tile_to_mahjax_id("4p")

    state = apply_mjai_event_to_mahjax_state(
        env,
        state,
        {"type": "pon", "pai": "4p", "consumed": ["4p", "4p"]},
    )
    assert int(state.current_player) == 1
    assert int(state.players.meld_counts[1]) == 1
    assert not bool(state.round_state.draw_next)


@pytest.mark.slow
def test_mjai_event_applier_applies_real_open_kan_response_window(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
            ["4p", "4p", "4p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E"],
            ["5p", "5p", "5p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "P"],
            ["6p", "6p", "6p", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "C"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="4p",
    )
    state = state.replace(
        round_state=state.round_state.replace(deck=state.round_state.deck.at[10].set(mjai_tile_to_mahjax_id("7m")))
    )

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "4p", "tsumogiri": True})
    assert int(state.current_player) == 1
    assert int(state.round_state.target) == mjai_tile_to_mahjax_id("4p")

    state = apply_mjai_event_to_mahjax_state(
        env,
        state,
        {"type": "kan", "pai": "4p", "consumed": ["4p", "4p", "4p"]},
    )
    assert int(state.current_player) == 1
    assert int(state.players.meld_counts[1]) == 1
    assert int(state.players.n_kan[1]) == 1
    assert not bool(state.round_state.kan_declared)
    assert not bool(state.round_state.draw_next)


@pytest.mark.slow
def test_mjai_event_applier_applies_real_closed_kan_after_draw(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["4p", "4p", "4p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1s"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N"],
            ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "P", "F", "C", "1m"],
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="4p",
    )
    state = state.replace(
        round_state=state.round_state.replace(deck=state.round_state.deck.at[10].set(mjai_tile_to_mahjax_id("7m")))
    )

    state = apply_mjai_event_to_mahjax_state(
        env,
        state,
        {"type": "ankan", "consumed": ["4p", "4p", "4p", "4p"]},
    )
    assert int(state.current_player) == 0
    assert int(state.round_state.last_draw) == mjai_tile_to_mahjax_id("7m")
    assert int(state.players.meld_counts[0]) == 1
    assert int(state.players.n_kan[0]) == 1
    assert not bool(state.round_state.kan_declared)
    assert not bool(state.round_state.draw_next)


@pytest.mark.slow
def test_mjai_event_applier_applies_real_riichi_after_draw(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m", "6m", "6m", "7m"],
            ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S", "W", "N"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "P", "F", "C", "1m"],
            ["1p", "1p", "2p", "2p", "3p", "3p", "4p", "4p", "5p", "5p", "6p", "6p", "7p"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="7m",
    )

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "reach", "actor": 0})
    assert int(state.current_player) == 0
    assert bool(state.players.riichi_declared[0])
    assert not bool(state.round_state.draw_next)
    assert bool(state.legal_action_mask[MAHJAX_TSUMOGIRI])


@pytest.mark.slow
def test_mjai_event_applier_applies_real_tsumo_after_draw(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"],
            ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S", "W", "N"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "P", "F", "C", "1m"],
            ["1p", "1p", "2p", "2p", "3p", "3p", "4p", "4p", "5p", "5p", "6p", "6p", "7p"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="5p",
    )

    assert bool(state.legal_action_mask[MAHJAX_TSUMO])
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "hora", "actor": 0, "target": 0})
    assert state.rewards[0] > 0
    assert state.round_state.score[0] > 250


@pytest.mark.slow
def test_mjai_event_applier_applies_real_ron_response_window(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=[
            ["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "9p"],
            ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "5p"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "P", "F", "C", "1m"],
            ["1p", "1p", "2p", "2p", "3p", "3p", "4p", "4p", "5p", "5p", "6p", "6p", "7p"],
        ],
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="5p",
    )

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "5p", "tsumogiri": True})
    assert bool(state.legal_action_mask[MAHJAX_RON])
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "hora", "actor": 1, "target": 0})
    assert state.rewards[1] > 0
    assert state.rewards[0] < 0
    assert state.round_state.score[1] > 250


class _Env:
    def step(self, state: _State, action: object) -> tuple[_State, int]:
        if not isinstance(action, int):
            action = int(np.asarray(action))
        return state, action


def test_mjai_event_applier_dispatches_supported_actions_fail_closed() -> None:
    env = _Env()
    state = _State([MAHJAX_PON, MAHJAX_CHI_LEFT, MAHJAX_OPEN_KAN], target=mjai_tile_to_mahjax_id("4m"))
    _, action = apply_mjai_event_to_mahjax_state(env, state, {"type": "pon", "pai": "4m", "consumed": ["4m", "4m"]})
    assert action == MAHJAX_PON
    _, action = apply_mjai_event_to_mahjax_state(env, state, {"type": "chi", "pai": "4m", "consumed": ["5m", "6m"]})
    assert action == MAHJAX_CHI_LEFT
    _, action = apply_mjai_event_to_mahjax_state(
        env,
        state,
        {"type": "kan", "pai": "4m", "consumed": ["4m", "4m", "4m"]},
    )
    assert action == MAHJAX_OPEN_KAN

    assert apply_mjai_event_to_mahjax_state(env, _State([MAHJAX_RIICHI]), {"type": "reach"})[1] == MAHJAX_RIICHI
    assert (
        apply_mjai_event_to_mahjax_state(env, _State([MAHJAX_TSUMO]), {"type": "hora", "actor": 1, "target": 1})[1]
        == MAHJAX_TSUMO
    )
    assert (
        apply_mjai_event_to_mahjax_state(env, _State([MAHJAX_RON]), {"type": "hora", "actor": 1, "target": 2})[1]
        == MAHJAX_RON
    )
    assert apply_mjai_event_to_mahjax_state(
        env, _State([MAHJAX_SELF_KAN_START + 4]), {"type": "ankan", "consumed": ["5m", "5m", "5m", "5m"]}
    )[1] == (MAHJAX_SELF_KAN_START + 4)
    drawn_state = _State([], target=0, last_draw=mjai_tile_to_mahjax_id("5p"))
    assert apply_mjai_event_to_mahjax_state(env, drawn_state, {"type": "tsumo", "actor": 0, "pai": "5p"}) is drawn_state
    with pytest.raises(ValueError, match="tsumo tile"):
        apply_mjai_event_to_mahjax_state(env, drawn_state, {"type": "tsumo", "actor": 0, "pai": "6p"})

    with pytest.raises(ValueError, match="already terminated"):
        apply_mjai_event_to_mahjax_state(env, state, {"type": "ryukyoku"})
    terminal_state = _State([], target=0)
    terminal_state.round_state.terminated_round = True
    assert apply_mjai_event_to_mahjax_state(env, terminal_state, {"type": "ryukyoku"}) is terminal_state


def test_static_mjai_jsonl_replay_prefix_runs_through_event_stream() -> None:
    events = _parse_mjai_jsonl(PARITY_JSONL)
    start = events[1]
    first_draw = events[2]
    state = _state_from_start_kyoku_event(start, first_draw=str(first_draw["pai"]))

    assert apply_mjai_event_to_mahjax_state(None, state, first_draw) is state
    rows = _parse_authority_rows(PARITY_AUTHORITY_JSON)
    _assert_projected_mask_matches_authority(state, rows, 0)
    fake_discard_state = _State([mjai_tile_to_mahjax_id("1m")], last_draw=mjai_tile_to_mahjax_id("5sr"))
    _, action = apply_mjai_event_to_mahjax_state(_Env(), fake_discard_state, events[3])
    assert action == mjai_tile_to_mahjax_id("1m")
    fake_response_state = _State([MAHJAX_PASS], target=mjai_tile_to_mahjax_id("1m"))
    _, action = apply_mjai_event_to_mahjax_state(_Env(), fake_response_state, events[4])
    assert action == MAHJAX_PASS


def test_authority_row_parser_rejects_malformed_fixture() -> None:
    with pytest.raises(AssertionError):
        _parse_authority_rows('{"action_space":45,"rows":[]}')
    with pytest.raises(AssertionError):
        _parse_authority_rows('{"action_space":46,"rows":[{"index":1,"legal_mask":[1]}]}')
    with pytest.raises(AssertionError):
        bad_mask = (
            "{"
            + f'"action_space":46,"rows":[{{"index":0,"legal_mask":[{",".join(["0"] * HYDRA_ACTION_SPACE)}]}}]'
            + "}"
        )
        _parse_authority_rows(bad_mask)


@pytest.mark.slow
def test_start_kyoku_projected_mask_matches_hydra_authority_fixture_row() -> None:
    state = mahjax_state_from_start_kyoku(
        tehais=START_TEHAIS,
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="5sr",
    )

    rows = _parse_authority_rows(PARITY_AUTHORITY_JSON)
    _assert_projected_mask_matches_authority(state, rows, 0)


@pytest.mark.slow
def test_replay_slice_projected_masks_match_hydra_authority_fixture_rows(mahjax_env: MahjaxEnv) -> None:
    env = mahjax_env
    state = mahjax_state_from_start_kyoku(
        tehais=START_TEHAIS,
        scores=[25000, 25000, 25000, 25000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="5sr",
    )
    player_1_draw = mjai_tile_to_mahjax_id("P")
    player_2_draw = mjai_tile_to_mahjax_id("6p")
    player_3_draw = mjai_tile_to_mahjax_id("7p")
    player_0_second_draw = mjai_tile_to_mahjax_id("6m")
    player_1_second_draw = mjai_tile_to_mahjax_id("F")
    player_2_second_draw = mjai_tile_to_mahjax_id("7p")
    player_3_second_draw = mjai_tile_to_mahjax_id("8p")
    player_0_third_draw = mjai_tile_to_mahjax_id("3m")
    player_1_third_draw = mjai_tile_to_mahjax_id("C")
    player_2_third_draw = mjai_tile_to_mahjax_id("9p")
    player_3_third_draw = mjai_tile_to_mahjax_id("9p")
    player_0_fourth_draw = mjai_tile_to_mahjax_id("4m")
    player_1_fourth_draw = mjai_tile_to_mahjax_id("4s")
    player_2_fourth_draw = mjai_tile_to_mahjax_id("6m")
    state = state.replace(
        round_state=state.round_state.replace(
            deck=state.round_state.deck.at[82]
            .set(player_1_draw)
            .at[81]
            .set(player_2_draw)
            .at[80]
            .set(player_3_draw)
            .at[79]
            .set(player_0_second_draw)
            .at[78]
            .set(player_1_second_draw)
            .at[77]
            .set(player_2_second_draw)
            .at[76]
            .set(player_3_second_draw)
            .at[75]
            .set(player_0_third_draw)
            .at[74]
            .set(player_1_third_draw)
            .at[73]
            .set(player_2_third_draw)
            .at[72]
            .set(player_3_third_draw)
            .at[71]
            .set(player_0_fourth_draw)
            .at[70]
            .set(player_1_fourth_draw)
            .at[69]
            .set(player_2_fourth_draw)
        )
    )
    state = _apply_mjai_events(
        env,
        state,
        [
            {"type": "dahai", "pai": "1m", "tsumogiri": False},
            {"type": "none"},
        ],
    )

    rows = _parse_authority_rows(PARITY_AUTHORITY_JSON)
    # Row 1 from the matching replay slice through tsumo(P).
    _assert_projected_mask_matches_authority(state, rows, 1)

    assert int(state.current_player) == 1
    assert int(state.round_state.last_draw) == player_1_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "1s", "tsumogiri": False})
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})

    # Row 2 from the matching replay slice through dahai(1s) + tsumo(6p).
    _assert_projected_mask_matches_authority(state, rows, 2)

    assert int(state.current_player) == 2
    assert int(state.round_state.last_draw) == player_2_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "P", "tsumogiri": False})

    # Row 3 from the matching replay slice through dahai(P) + tsumo(7p).
    _assert_projected_mask_matches_authority(state, rows, 3)

    assert int(state.current_player) == 3
    assert int(state.round_state.last_draw) == player_3_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "6p", "tsumogiri": False})

    # Row 4 from the matching replay slice through dahai(6p) + tsumo(6m).
    _assert_projected_mask_matches_authority(state, rows, 4)

    assert int(state.current_player) == 0
    assert int(state.round_state.last_draw) == player_0_second_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "2m", "tsumogiri": False})
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})

    # Row 5 from the matching replay slice through dahai(2m) + tsumo(F).
    _assert_projected_mask_matches_authority(state, rows, 5)

    assert int(state.current_player) == 1
    assert int(state.round_state.last_draw) == player_1_second_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "2s", "tsumogiri": False})
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})

    # Row 6 from the matching replay slice through dahai(2s) + tsumo(7p).
    _assert_projected_mask_matches_authority(state, rows, 6)

    assert int(state.current_player) == 2
    assert int(state.round_state.last_draw) == player_2_second_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "F", "tsumogiri": False})

    # Row 7 from the matching replay slice through dahai(F) + tsumo(8p).
    _assert_projected_mask_matches_authority(state, rows, 7)

    assert int(state.current_player) == 3
    assert int(state.round_state.last_draw) == player_3_second_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "7p", "tsumogiri": False})

    # Row 8 from the matching replay slice through dahai(7p) + tsumo(3m).
    _assert_projected_mask_matches_authority(state, rows, 8)

    assert int(state.current_player) == 0
    assert int(state.round_state.last_draw) == player_0_third_draw

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "3m", "tsumogiri": True})
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})

    # Row 9 from the matching replay slice through tsumogiri(3m) + tsumo(C).
    _assert_projected_mask_matches_authority(state, rows, 9)

    assert int(state.current_player) == 1
    assert int(state.round_state.last_draw) == player_1_third_draw
    legal_before_player_1_discard = np.asarray(state.legal_action_mask, dtype=bool)
    player_1_hand_after_c = np.asarray(state.players.hand_with_red[1], dtype=np.int8)
    assert not bool(state.terminated)
    assert bool(legal_before_player_1_discard[mjai_tile_to_mahjax_id("3s")])
    assert not bool(legal_before_player_1_discard[player_1_third_draw])
    assert int(player_1_hand_after_c[mjai_tile_to_mahjax_id("3s")]) == 1
    assert int(player_1_hand_after_c[player_1_third_draw]) == 1
    assert int(state.round_state.next_deck_ix) == 73

    with pytest.raises(ValueError, match="tedashi tile is not legal"):
        mahjax_action_from_mjai_dahai(state, pai="C", tsumogiri=False)
    assert mahjax_action_from_mjai_dahai(state, pai="C", tsumogiri=True) == MAHJAX_TSUMOGIRI
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "3s", "tsumogiri": False})
    # Row 10 root cause was selected-action translation, not constructor accounting:
    # last-draw discards must use MahJAX TSUMOGIRI, while tedashi uses the tile id.
    # That distinction lets replay events advance without tripping the illegal-action sentinel.

    # Row 10 from the matching replay slice through tedashi(3s) + tsumo(9p).
    _assert_projected_mask_matches_authority(state, rows, 10)

    assert not bool(state.terminated)
    assert int(state.current_player) == 2
    assert int(state.round_state.last_draw) == player_2_third_draw
    assert int(state.round_state.next_deck_ix) == 72
    assert int(state.round_state.target) == -1
    assert int(state.round_state.last_player) == 1
    assert int(state.round_state.honba) == 0
    assert int(state.round_state.kyotaku) == 0
    np.testing.assert_array_equal(np.asarray(state.round_state.score), np.asarray([250, 250, 250, 250]))

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "1m", "tsumogiri": False})

    # Row 11 from the matching replay slice through tedashi(1m) + tsumo(9p).
    _assert_projected_mask_matches_authority(state, rows, 11)

    assert not bool(state.terminated)
    assert int(state.current_player) == 3
    assert int(state.round_state.last_draw) == player_3_third_draw
    assert int(state.round_state.next_deck_ix) == 71
    assert int(state.round_state.target) == -1
    assert int(state.round_state.last_player) == 2

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "8p", "tsumogiri": False})

    # Row 12 from the matching replay slice through tedashi(8p) + tsumo(4m).
    _assert_projected_mask_matches_authority(state, rows, 12)

    assert not bool(state.terminated)
    assert int(state.current_player) == 0
    assert int(state.round_state.last_draw) == player_0_fourth_draw
    assert int(state.round_state.next_deck_ix) == 70
    assert int(state.round_state.target) == -1
    assert int(state.round_state.last_player) == 3

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "4m", "tsumogiri": True})
    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "none"})

    # Row 13 from the matching replay slice through tsumogiri(4m), no-call response, and tsumo(4s).
    _assert_projected_mask_matches_authority(state, rows, 13)

    assert not bool(state.terminated)
    assert int(state.current_player) == 1
    assert int(state.round_state.last_draw) == player_1_fourth_draw
    assert int(state.round_state.next_deck_ix) == 69
    assert int(state.round_state.target) == -1
    assert int(state.round_state.last_player) == 0

    state = apply_mjai_event_to_mahjax_state(env, state, {"type": "dahai", "pai": "4s", "tsumogiri": True})

    # Row 14 from the matching replay slice through tsumogiri(4s), no-call response, and tsumo(6m).
    _assert_projected_mask_matches_authority(state, rows, 14)

    assert not bool(state.terminated)
    assert int(state.current_player) == 2
    assert int(state.round_state.last_draw) == player_2_fourth_draw
    assert int(state.round_state.next_deck_ix) == 68
    assert int(state.round_state.target) == -1
    assert int(state.round_state.last_player) == 1
