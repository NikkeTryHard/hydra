"""WP-14 wall-less sim replay: hand-built invented kyoku deals.

Covers deals with no mount bytes framed as complete games: an unoffered
ankan alternative, single-step reach plus ron, chi through reach acceptance
with a flag-less tsumo win, reason-less aborts and wall exhaustion, double
ron and marker-less kan-dora quarantines, a repeated kan-dora marker, and
turn-ordered versus out-of-turn third draws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hydra2._native import tiles as _tiles_bridge
from hydra2.contracts.common import ContractError
from hydra2.engines.riichienv._lr_end import replay_game
from tests.unit.test_log_replay_wp14 import _assert_valid_rows, _mask_action_kinds, _record

mjai_string_of = _tiles_bridge.mjai_string_of

if TYPE_CHECKING:
    from hydra2.data.decode import GameRecord

pytestmark = pytest.mark.contract_package("WP-14")


def _invented_kyoku_game(
    game_id: str, tehais: list[list[str]], header: dict[str, object], body: list[dict[str, object]]
) -> GameRecord:
    """One hand-built invented kyoku framed as a complete game (no mount bytes)."""
    events: list[dict[str, object]] = [
        {"type": "start_game"},
        {"type": "start_kyoku", "tehais": tehais, **header},
        *body,
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [25000, 25000, 25000, 25000]},
    ]
    return _record(game_id, events)


def test_invented_quad_hand_ankan_alternative_emits_rows() -> None:
    """Closed quad with an unoffered ankan alternative (hand-built deal).

    Seat 0 holds four ``E`` and tsumogiri-discards an unrelated draw while
    the engine also offers the closed kan; the mask must carry the ankan
    alternative with copies resolved from the tracked hand.
    """
    tehais = [
        ["E", "E", "E", "E", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
        ["S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "C", "C", "1p"],
        ["1p", "1p", "2p", "2p", "3p", "3p", "4p", "4p", "5p", "5p", "6p", "6p", "7p"],
        ["1s", "1s", "2s", "2s", "3s", "3s", "4s", "4s", "5s", "5s", "6s", "6s", "7s"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
    ]
    game = _invented_kyoku_game("wp14-invented-quad", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-quad", rows, [0])
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import (
        ACTION_TABLE_RELPATH,
        load_action_table,
    )

    table = load_action_table(repo_root() / ACTION_TABLE_RELPATH)
    assert ("ankan", None, None, ("E", "E", "E", "E")) in _mask_action_kinds(rows[0], table)


def test_invented_single_step_reach_and_ron_emit_rows() -> None:
    """Single-step reach declaration plus ron on a later discard (hand-built deal).

    Seat 0 reaches with a shanpon wait where the drawn tile completes the
    structure, so the oracle yields the reach with no following declaration
    step; the discard resolves from the tracked hand. Seat 1 later deals
    into the wait and seat 0 rons.
    """
    tehais = [
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "E", "E", "5m", "W"],
        ["E", "E", "S", "S", "N", "N", "P", "P", "F", "F", "C", "C", "1m"],
        ["1p", "1p", "2p", "2p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s"],
        ["1s", "1s", "5s", "5s", "6s", "6s", "7s", "7s", "8s", "8s", "9s", "9s", "C"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        {"type": "dahai", "actor": 2, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "9m"},
        {"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "W", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "S"},
        {"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "N"},
        {"type": "dahai", "actor": 2, "pai": "N", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "P"},
        {"type": "dahai", "actor": 3, "pai": "P", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "F"},
        {"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5m"},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [3900, -3900, 0, 0]},
    ]
    game = _invented_kyoku_game("wp14-invented-reach", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-reach", rows, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0])
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import (
        ACTION_TABLE_RELPATH,
        load_action_table,
    )

    table = load_action_table(repo_root() / ACTION_TABLE_RELPATH)
    kinds = _mask_action_kinds(rows[4], table)
    assert ("riichi_discard", "W", None, ()) in kinds


def test_invented_chi_on_reach_and_flagless_tsumo_emit_rows() -> None:
    """Chi on the reach declaration plus a flag-less tsumo win (hand-built deal).

    Seat 1 chis seat 0's declaration discard straight through the reach
    acceptance, and seat 2 later wins by self-draw with ``target == actor``
    and no ``tsumo`` flag.
    """
    tehais = [
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "3s"],
        ["1s", "2s", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F"],
        ["1m", "1m", "1m", "9m", "9m", "9m", "1p", "1p", "1p", "6s", "7s", "8s", "5s"],
        ["S", "S", "W", "W", "N", "N", "P", "F", "C", "C", "9p", "9p", "6p"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "7p"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3s", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "chi", "actor": 1, "target": 0, "pai": "3s", "consumed": ["1s", "2s"]},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "1m"},
        {"type": "dahai", "actor": 2, "pai": "1m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "9m"},
        {"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "F"},
        {"type": "dahai", "actor": 1, "pai": "F", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5s"},
        {"type": "hora", "actor": 2, "target": 2, "deltas": [-2000, -2000, 8000, -2000]},
    ]
    game = _invented_kyoku_game("wp14-invented-chi-tsumo", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-chi-tsumo", rows, [0, 1, 1, 2, 3, 0, 1, 2])
    melds = rows[2].actor_observation["visible_melds"]
    assert melds[1] and melds[1][0]["kind"] == "chi"


def test_invented_kyushu_abort_without_reason_emits_no_rows() -> None:
    """Reason-less first-turn abort on a nine-terminal deal (hand-built deal).

    The dealer's fourteen tiles span thirteen terminal/honor kinds, so the
    oracle offers the nine-terminals abort; the walk must complete the game
    with no rows rather than desync on the empty queue.
    """
    tehais = [
        ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"],
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "6p"],
        ["1p", "1p", "2p", "2p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "7s"],
        ["5s", "5s", "6s", "6s", "7s", "7s", "8s", "8s", "9s", "9s", "2m", "2m", "4m"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    game = _invented_kyoku_game("wp14-invented-kyushu", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-kyushu", rows, [])


def test_invented_exhaustive_draw_without_reason_emits_rows() -> None:
    """Reason-less wall exhaustion over sixty invented tsumogiri draws.

    Hand-built deal (no mount bytes): closed honor/terminal soup hands that
    never interact, plus sixty tile-conserving tsumogiri draws (fifteen
    middle-suit kinds times four copies, disjoint from the deal). The full
    wall of draws classifies the reason-less ending as exhaustive without
    guessing at abortives.
    """
    tehais = [
        ["E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "C"],
        ["E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "1m"],
        ["C", "C", "C", "1m", "1m", "1m", "9m", "9m", "9m", "9m", "1p", "1p", "1p"],
        ["1p", "9p", "9p", "9p", "9p", "1s", "1s", "1s", "1s", "9s", "9s", "9s", "9s"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "5s",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "dahai", "actor": 1, "pai": "2m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "2m"},
        {"type": "dahai", "actor": 2, "pai": "2m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "2m"},
        {"type": "dahai", "actor": 3, "pai": "2m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "3m"},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "3m"},
        {"type": "dahai", "actor": 1, "pai": "3m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "3m"},
        {"type": "dahai", "actor": 2, "pai": "3m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "3m"},
        {"type": "dahai", "actor": 3, "pai": "3m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "4m"},
        {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "4m"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "4m"},
        {"type": "dahai", "actor": 2, "pai": "4m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "4m"},
        {"type": "dahai", "actor": 3, "pai": "4m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6m"},
        {"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "6m"},
        {"type": "dahai", "actor": 2, "pai": "6m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "6m"},
        {"type": "dahai", "actor": 3, "pai": "6m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "7m"},
        {"type": "dahai", "actor": 0, "pai": "7m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "7m"},
        {"type": "dahai", "actor": 1, "pai": "7m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "7m"},
        {"type": "dahai", "actor": 2, "pai": "7m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "7m"},
        {"type": "dahai", "actor": 3, "pai": "7m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "8m"},
        {"type": "dahai", "actor": 0, "pai": "8m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "8m"},
        {"type": "dahai", "actor": 1, "pai": "8m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "8m"},
        {"type": "dahai", "actor": 2, "pai": "8m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "8m"},
        {"type": "dahai", "actor": 3, "pai": "8m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "2p"},
        {"type": "dahai", "actor": 0, "pai": "2p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2p"},
        {"type": "dahai", "actor": 1, "pai": "2p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "2p"},
        {"type": "dahai", "actor": 2, "pai": "2p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "2p"},
        {"type": "dahai", "actor": 3, "pai": "2p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "3p"},
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "3p"},
        {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "3p"},
        {"type": "dahai", "actor": 2, "pai": "3p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "3p"},
        {"type": "dahai", "actor": 3, "pai": "3p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "4p"},
        {"type": "dahai", "actor": 1, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "4p"},
        {"type": "dahai", "actor": 2, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "4p"},
        {"type": "dahai", "actor": 3, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "dahai", "actor": 0, "pai": "6p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "6p"},
        {"type": "dahai", "actor": 1, "pai": "6p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "6p"},
        {"type": "dahai", "actor": 2, "pai": "6p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "6p"},
        {"type": "dahai", "actor": 3, "pai": "6p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "7p"},
        {"type": "dahai", "actor": 0, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "7p"},
        {"type": "dahai", "actor": 1, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "7p"},
        {"type": "dahai", "actor": 2, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "7p"},
        {"type": "dahai", "actor": 3, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "8p"},
        {"type": "dahai", "actor": 0, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
        {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "8p"},
        {"type": "dahai", "actor": 2, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "8p"},
        {"type": "dahai", "actor": 3, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "2s"},
        {"type": "dahai", "actor": 0, "pai": "2s", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "2s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "2s"},
        {"type": "dahai", "actor": 2, "pai": "2s", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "2s"},
        {"type": "dahai", "actor": 3, "pai": "2s", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "3s"},
        {"type": "dahai", "actor": 0, "pai": "3s", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "3s"},
        {"type": "dahai", "actor": 1, "pai": "3s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "3s"},
        {"type": "dahai", "actor": 2, "pai": "3s", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "3s"},
        {"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "4s"},
        {"type": "dahai", "actor": 0, "pai": "4s", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "4s"},
        {"type": "dahai", "actor": 1, "pai": "4s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "4s"},
        {"type": "dahai", "actor": 2, "pai": "4s", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "4s"},
        {"type": "dahai", "actor": 3, "pai": "4s", "tsumogiri": True},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    game = _invented_kyoku_game("wp14-invented-exhaust", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-exhaust", rows, [i % 4 for i in range(60)])


def test_invented_double_ron_second_winner_quarantined() -> None:
    """Second ron winner on one discard quarantines with a distinct code (hand-built deal).

    Double ron is log-valid but unhandled: the single-winner pipeline emits
    the first ron row, then quarantines on the second hora instead of the
    generic post-decision message.
    """
    tehais = [
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "E", "E", "5m", "W"],
        ["E", "E", "S", "S", "N", "N", "P", "P", "F", "F", "C", "C", "1m"],
        ["1p", "1p", "2p", "2p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "9p"],
        ["1s", "1s", "5s", "5s", "6s", "6s", "7s", "7s", "8s", "8s", "9s", "9s", "C"],
    ]
    header = {
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "9m"},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "9m"},
        {"type": "dahai", "actor": 2, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "9m"},
        {"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "W", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "5m"},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [3900, -3900, 0, 0]},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -3900, 3900, 0]},
    ]
    game = _invented_kyoku_game("wp14-invented-double-ron", tehais, header, body)
    with pytest.raises(ContractError, match="double ron on one discard is quarantined"):
        replay_game(game)


def test_invented_bare_kan_dora_quarantined() -> None:
    """Kan-dora event without a marker quarantines as unrecoverable (hand-built deal).

    A marker-less ``{"type": "dora"}`` names no recoverable indicator, so the
    walk fails closed with a distinct code instead of resolving or guessing
    it. Never observed in the Tenhou corpus (conversion audit: every dora
    event carries a marker); kept as fail-closed defense-in-depth.
    """
    tehais = [
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "3s"],
        ["1s", "2s", "S", "S", "W", "W", "N", "N", "P", "P", "F", "C", "1m"],
        ["E", "E", "E", "E", "1p", "2p", "3p", "6s", "7s", "8s", "5p", "6p", "7p"],
        ["9m", "9p", "1m", "9s", "S", "W", "N", "P", "F", "C", "4m", "5m", "6s"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "5p"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "7p"},
        {"type": "dahai", "actor": 1, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "5s"},
        {"type": "ankan", "actor": 2, "consumed": ["E", "E", "E", "E"]},
        {"type": "dora"},
    ]
    game = _invented_kyoku_game("wp14-invented-bare-dora", tehais, header, body)
    with pytest.raises(ContractError, match="kan-dora indicator unrecoverable"):
        replay_game(game)


def test_invented_double_kan_same_marker_emits_rows() -> None:
    """Two kans revealing one marker string emit faithful rows (hand-built deal).

    Regression for the refuted 'no fresh indicator' quarantine: the log's
    ordered kan-dora markers are faithful (conversion audit), so the second
    reveal of the same string resolves against the used indicator and rows
    carry both reveals. Filler determinism fixes the indicator list for
    this deal (``F`` is single-fresh here); any wall-layout change fails
    loudly at replay time.
    """
    tehais = [
        ["E", "E", "E", "E", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
        ["S", "S", "W", "W", "N", "N", "P", "P", "F", "F", "C", "C", "1s"],
        ["9p", "9p", "9p", "9p", "2p", "3p", "4p", "6p", "7p", "8p", "1s", "2s", "3s"],
        ["4s", "5s", "6s", "7s", "8s", "9s", "1m", "2m", "3m", "4p", "5p", "6p", "7s"],
    ]
    header = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }
    body = [
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "ankan", "actor": 0, "consumed": ["E", "E", "E", "E"]},
        {"type": "dora", "dora_marker": "F"},
        {"type": "tsumo", "actor": 0, "pai": "9s"},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "1m"},
        {"type": "ankan", "actor": 2, "consumed": ["9p", "9p", "9p", "9p"]},
        {"type": "dora", "dora_marker": "F"},
        {"type": "tsumo", "actor": 2, "pai": "9s"},
        {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
    ]
    game = _invented_kyoku_game("wp14-invented-double-kan-marker", tehais, header, body)
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-double-kan-marker", rows, [0, 0, 1, 2, 2])
    revealed = [d for d in rows[4].actor_observation["dora_indicators"] if d != -1]
    assert sorted(mjai_string_of(int(t)) for t in revealed) == ["F", "F"]


def _three_exchange_tehais() -> list[list[str]]:
    return [
        ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "3s"],
        ["1s", "2s", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F"],
        ["1m", "1m", "1m", "9m", "9m", "9m", "1p", "1p", "1p", "6s", "7s", "8s", "5s"],
        ["S", "S", "W", "W", "N", "N", "P", "F", "C", "C", "9p", "9p", "6p"],
    ]


def _three_exchange_header() -> dict[str, object]:
    return {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    }


def test_invented_round_robin_draws_emit_rows() -> None:
    """Legal third draw (turn order 0-1-2) emits three rows (hand-built deal).

    Guards the drawer check against over-tightening: a third tsumo/dahai
    pair is sound when the turn order reaches that seat.
    """
    body = [
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "9s"},
        {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
    ]
    game = _invented_kyoku_game(
        "wp14-invented-round-robin", _three_exchange_tehais(), _three_exchange_header(), body
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-invented-round-robin", rows, [0, 1, 2])


def test_invented_out_of_turn_tsumo_quarantined_at_draw() -> None:
    """Turn-illegal third draw quarantines at the tsumo (hand-built deal).

    Seat 0 draws right after seat 1's discard with no intervening claim, so
    the engine has already dealt that wall tile to seat 2; the draw used to
    string-match seat 2's tile and desync a later row, and now fail-closes
    at the offending draw instead.
    """
    body = [
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "dahai", "actor": 0, "pai": "6p", "tsumogiri": True},
    ]
    game = _invented_kyoku_game(
        "wp14-invented-out-of-turn", _three_exchange_tehais(), _three_exchange_header(), body
    )
    with pytest.raises(ContractError, match="out-of-turn draw"):
        replay_game(game)
