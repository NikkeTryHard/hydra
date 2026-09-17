"""WP-14 wall-less sim replay: framed MJAI -> DecisionRow via MjaiReplay.

Covers the wall-less path the engine-reset expander cannot run (real Tenhou
MJAI carries no wall field):

- a one-kyoku reach + pon fixture replays to the expected row count with
  actor-masked hands, legal chosen actions, and ``(5,)`` dora;
- bytes/path inputs decode strictly and replay identically to records;
- split/seat validation fails closed;
- seat filtering keeps positional decision ids (privileged join alignment);
- engine-reset vs sim-replay agreement on a walled fixture: identical
  decision ids, seats, chosen actions (where copies coincide), string-level
  observation content, normalized masks, history kinds, and envelope
  counters -- differing exactly by the documented classes (wall-derived vs
  log game identity, copy-identity folding, refit numbering, wall-less
  derivation marker);
- a tampered-dahai desync raises ``ContractError`` naming the game;
- chi / ankan (+ kan-dora growth) / daiminkan / kakan / ron / tsumo-win
  fixtures replay to structurally valid rows (fail-closed anywhere else).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from hydra2_replay_rs import tiles as _tiles_bridge

from hydra2.contracts.common import ContractError
from hydra2.data.decode import GameRecord
from hydra2.data.parquet import FORBIDDEN_IN_ACTOR
from hydra2.data.replay_expand import expand_game, expand_privileged_rows
from hydra2.engines.riichienv.log_replay import SIM_DERIVATION_MARK, replay_game

mjai_string_of = _tiles_bridge.mjai_string_of

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-14")


GAME_ID = "wp14-sim-replay-01"


def _record(
    game_id: str,
    events: list[dict[str, object]],
    *,
    wall_tiles: tuple[int, ...] | None = None,
    object_id: str = "wp14-sim-obj",
) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        object_id=object_id,
        packaged_object_id=f"{object_id}-pkg",
        events=tuple(events),
        raw_bytes_sha256="sha256:" + "0" * 64,
        wall_tiles=wall_tiles,
        source={"type": "start_game"},
    )


def _frame(events: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


def _fix_tehais() -> list[list[str]]:
    return [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "E", "E", "5p", "6p", "S"],
        ["S", "S", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "6p", "7p", "1s"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "6p", "2m"],
        ["E", "2p", "W", "N", "P", "F", "C", "7m", "8m", "9m", "6p", "7p", "8p"],
    ]


def _fix_events() -> list[dict[str, object]]:
    tehais = _fix_tehais()
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": tehais,
        },
        {"type": "tsumo", "actor": 0, "pai": "3p"},
        {"type": "dahai", "actor": 0, "pai": "S", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 0, "pai": "S", "consumed": ["S", "S"]},
        {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "7p"},
        {"type": "dahai", "actor": 2, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "3s"},
        {"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
        {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [1500, 1500, -1500, -1500]},
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [27000, 25000, 24000, 23000]},
    ]


def _agree_events() -> list[dict[str, object]]:
    tehais = _fix_tehais()
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": tehais,
        },
        {"type": "tsumo", "actor": 0, "pai": "3p"},
        {"type": "dahai", "actor": 0, "pai": "S", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
        {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "7p"},
        {"type": "dahai", "actor": 2, "pai": "7p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "3s"},
        {"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "9m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "tsumo", "actor": 1, "pai": "9p"},
        {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [1500, 1500, -1500, -1500]},
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [27000, 25000, 24000, 23000]},
    ]


def _arranged_wall(events: list[dict[str, object]]) -> tuple[int, ...]:
    """Unique-permutation wall playing the log's strings in deal/draw order.

    Copy pools are consumed base-first across tehais then draws, so the engine
    path deals and draws deterministic distinct copies the assertions pin.
    """
    from hydra2_replay_rs import tiles as _tiles_bridge

    physical_of = _tiles_bridge.physical_of

    start = next(e for e in events if e.get("type") == "start_kyoku")
    tehais = cast("list[list[str]]", start["tehais"])
    pools: dict[str, list[int]] = {}

    def pool(pai: str) -> list[int]:
        if pai not in pools:
            first = int(physical_of(pai))
            if pai in ("5mr", "0m"):
                ids = [16]
            elif pai in ("5pr", "0p"):
                ids = [52]
            elif pai in ("5sr", "0s"):
                ids = [88]
            else:
                base = (first // 4) * 4
                ids = [base, base + 1, base + 2, base + 3]
                if len(pai) == 2 and pai[0] == "5":
                    ids = [i for i in ids if i != base]
            pools[pai] = ids
        return pools[pai]

    used: dict[str, int] = {}

    def take(pai: str) -> int:
        choices = pool(pai)
        cursor = used.get(pai, 0)
        if cursor >= len(choices):
            raise AssertionError(f"fixture overuses tile string {pai!r}")
        used[pai] = cursor + 1
        return choices[cursor]

    wall = [-1] * 136
    for seat in range(4):
        for j in range(12):
            block, slot = divmod(j, 4)
            wall[4 * (seat + 4 * block) + slot] = take(tehais[seat][j])
        wall[48 + seat] = take(tehais[seat][12])
    draw_index = 0
    for event in events:
        if event.get("type") == "tsumo":
            wall[52 + draw_index] = take(str(event["pai"]))
            draw_index += 1
    wall[131] = take("F")
    leftovers = (t for t in range(136) if t not in set(wall) - {-1})
    wall = [t if t != -1 else next(leftovers) for t in wall]
    assert sorted(wall) == list(range(136))
    return tuple(wall)


def _strings(ids: Any) -> list[str]:
    return sorted(mjai_string_of(int(t)) for t in ids)


def test_wall_less_counts_masks_and_dora() -> None:
    game = _record(GAME_ID, _fix_events())
    rows = replay_game(game)
    assert [r.seat for r in rows] == [0, 1, 1, 2, 3, 0, 1]
    assert [r.decision_id for r in rows] == [f"{GAME_ID}:d{i:04d}" for i in range(7)]
    assert all(r.round_id == f"{GAME_ID}:h00" for r in rows)
    assert all(r.game_id == GAME_ID and r.split == "train" for r in rows)
    assert all(r.source_object_id == "wp14-sim-obj" for r in rows)
    assert all(r.privileged_label_ref == r.decision_id for r in rows)
    for row in rows:
        obs = row.actor_observation
        mask = obs["legal_mask"]
        assert isinstance(mask, list) and mask[row.chosen_action_id] is True
        assert any(mask) and len(mask) > row.chosen_action_id
        dora = obs["dora_indicators"]
        assert isinstance(dora, list) and len(dora) == 5
        concealed = obs["concealed_hand"]
        assert isinstance(concealed, list) and len(concealed) > 0
        assert concealed == sorted(concealed)
        assert obs["actor"] == row.seat
        assert obs["game_id"] == GAME_ID
        for key in FORBIDDEN_IN_ACTOR:
            assert key not in obs


def test_privileged_join_alignment_wall_less() -> None:
    game = _record(GAME_ID, _fix_events())
    rows = replay_game(game)
    privileged = expand_privileged_rows(game)
    assert [p.decision_id for p in privileged] == [r.decision_id for r in rows]
    for entry in privileged:
        assert "wall_id" not in entry.privileged_label


def _row_key(row: Any) -> Any:
    doc = dict(row.actor_observation)
    doc.pop("game_id", None)
    doc.pop("decision_id", None)
    doc.pop("observation_hash", None)
    doc.pop("sequence", None)
    history = [dict(e) for e in doc.pop("visible_history", [])]
    for envelope in history:
        envelope.pop("game_id", None)
        envelope.pop("sequence", None)
    return (row.seat, row.chosen_action_id, doc, history)


def test_bytes_and_path_inputs_replay_identically(tmp_path: Path) -> None:
    events = _fix_events()
    game = _record(GAME_ID, events)
    expected = [_row_key(r) for r in replay_game(game)]
    from_bytes = replay_game(_frame(events))
    # Bytes without a game_id field decode to a synthetic identity, but the
    # decisions, seats, and observations replay identically.
    assert from_bytes[0].game_id.startswith("game-")
    assert [_row_key(r) for r in from_bytes] == expected
    staged = tmp_path / "game.mjai.jsonl"
    staged.write_bytes(_frame(events))
    from_path = replay_game(staged)
    assert [_row_key(r) for r in from_path] == expected


def test_split_and_seat_validation() -> None:
    game = _record(GAME_ID, _fix_events())
    with pytest.raises(ContractError):
        replay_game(game, split="")
    with pytest.raises(ContractError):
        replay_game(game, seat=4)
    with pytest.raises(ContractError):
        replay_game(game, seat=-1)


def test_seat_filter_keeps_positional_ids() -> None:
    game = _record(GAME_ID, _fix_events())
    rows = replay_game(game, seat=2)
    assert len(rows) == 1
    assert rows[0].seat == 2
    assert rows[0].decision_id == f"{GAME_ID}:d0003"
    assert rows[0].actor_observation["actor"] == 2


def _mask_action_kinds(row: Any, table: Any) -> set[tuple[Any, ...]]:
    """Legal-mask content with copy identity folded to MJAI strings."""
    kinds: set[tuple[Any, ...]] = set()
    for pos, flag in enumerate(row.actor_observation["legal_mask"]):
        if not flag:
            continue
        action = table.actions[pos]
        tile = mjai_string_of(int(action.tile)) if action.tile is not None else None
        called = mjai_string_of(int(action.called_tile)) if action.called_tile is not None else None
        consumed = tuple(sorted(mjai_string_of(int(t)) for t in action.consumed_tiles))
        kinds.add((action.kind, tile, called, consumed))
    return kinds


def test_wall_less_marker_value_pinned() -> None:
    assert SIM_DERIVATION_MARK == "sim-replay-wall-less-v1"


def test_engine_sim_agreement_semantic() -> None:
    """Engine-reset vs sim-replay agree decision-for-decision.

    The oracle reports string-canonical tile ids (one id per MJAI string)
    while the engine deals distinct physical copies from its unique wall, so
    raw hashes differ exactly by copy identity (plus wall-derived vs log game
    identity, refit numbering, and the wall-less derivation marker). This
    test pins everything that must agree: decision ids, seats, round ids,
    chosen actions where the copy coincides, string-level observation
    content, copy-folded masks, history kinds, and envelope counters.
    """
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import (
        ACTION_TABLE_RELPATH,
        load_action_table,
    )

    events = _agree_events()
    wall = _arranged_wall(events)
    table = load_action_table(repo_root() / ACTION_TABLE_RELPATH)
    engine_rows = expand_game(_record("agree-wp14", events, wall_tiles=wall))
    sim_rows = replay_game(_record("agree-wp14", events))
    assert len(engine_rows) == len(sim_rows) == 6
    for engine, sim in zip(engine_rows, sim_rows, strict=True):
        assert (sim.decision_id, sim.seat, sim.round_id) == (
            engine.decision_id,
            engine.seat,
            engine.round_id,
        )
        assert sim.source_object_id == engine.source_object_id == "wp14-sim-obj"
        assert sim.split == engine.split == "train"
        assert sim.privileged_label_ref == engine.privileged_label_ref
        # Wall-less provenance observes the log game id; the engine path
        # observes its wall-derived builder identity. Never equal here.
        assert sim.actor_observation["game_id"] == "agree-wp14"
        assert engine.actor_observation["game_id"] != "agree-wp14"
        assert sim.actor_observation["decision_id"] != engine.actor_observation["decision_id"]
        # Derivations differ exactly by the wall binding (digest vs marker).
        assert sim.derivation_hash != engine.derivation_hash
        assert sim.observation_hash != engine.observation_hash
        for key in (
            "actor",
            "round_index",
            "hand_number",
            "honba",
            "riichi_sticks",
            "dealer",
            "seat_winds",
            "round_wind",
            "scores",
            "turn_actor",
            "phase",
            "live_wall_tiles_remaining",
            "kan_count",
            "ippatsu_active",
            "actor_furiten",
            "actor_can_tsumo",
            "actor_can_riichi",
            "riichi_states",
        ):
            assert sim.actor_observation[key] == engine.actor_observation[key], key
        assert _strings(sim.actor_observation["concealed_hand"]) == _strings(
            engine.actor_observation["concealed_hand"]
        )
        drawn_sim = sim.actor_observation["own_drawn_tile"]
        drawn_engine = engine.actor_observation["own_drawn_tile"]
        if drawn_sim is None or drawn_engine is None:
            assert drawn_sim == drawn_engine
        else:
            assert mjai_string_of(int(drawn_sim)) == mjai_string_of(int(drawn_engine))
        assert [_strings(river) for river in sim.actor_observation["visible_discards"]] == [
            _strings(river) for river in engine.actor_observation["visible_discards"]
        ]
        assert (
            sim.actor_observation["dora_indicators"] == engine.actor_observation["dora_indicators"]
        )
        assert _mask_action_kinds(sim, table) == _mask_action_kinds(engine, table)
        kinds_sim = [e["kind"] for e in sim.actor_observation["visible_history"]]
        kinds_engine = [e["kind"] for e in engine.actor_observation["visible_history"]]
        assert kinds_sim == kinds_engine
    # Chosen actions agree wherever the copy coincides (deal/draw order took
    # the base copy); folded copies diverge by physical id only.
    assert sim_rows[0].chosen_action_id == engine_rows[0].chosen_action_id
    assert sim_rows[5].chosen_action_id == engine_rows[5].chosen_action_id
    for sim, engine in zip(sim_rows, engine_rows, strict=True):
        assert sim.actor_observation["legal_mask"][sim.chosen_action_id] is True
        assert engine.actor_observation["legal_mask"][engine.chosen_action_id] is True


def test_tampered_dahai_desync_raises() -> None:
    events = _fix_events()
    tampered = [dict(e) for e in events]
    for event in tampered:
        if event.get("type") == "dahai" and event.get("actor") == 2:
            event["pai"] = "W"
            break
    game = _record("fix-desync", tampered)
    with pytest.raises(ContractError, match="fix-desync"):
        replay_game(game)


def _kyoku_game(game_id: str, tehais: list[list[str]], body: list[dict[str, object]]) -> GameRecord:
    events: list[dict[str, object]] = [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "F",
            "honba": 0,
            "kyoku": 1,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": tehais,
        },
        *body,
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [25000, 25000, 25000, 25000]},
    ]
    return _record(game_id, events)


def _assert_valid_rows(game_id: str, rows: Any, seats: list[int]) -> None:
    assert [r.seat for r in rows] == seats
    assert [r.decision_id for r in rows] == [f"{game_id}:d{i:04d}" for i in range(len(rows))]
    for row in rows:
        obs = row.actor_observation
        assert obs["legal_mask"][row.chosen_action_id] is True
        assert len(obs["dora_indicators"]) == 5
        assert len(obs["concealed_hand"]) > 0
        assert obs["actor"] == row.seat
        for key in FORBIDDEN_IN_ACTOR:
            assert key not in obs


def test_chi_claim_structural() -> None:
    tehais = [
        ["5s", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
        ["5m", "6m", "7m", "1p", "2p", "3p", "7p", "8p", "9p", "1s", "2s", "3s", "4s"],
        ["5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P", "F", "2p", "3p"],
        ["E", "S", "W", "N", "P", "F", "C", "9m", "9m", "1p", "1p", "4p", "6p"],
    ]
    game = _kyoku_game(
        "wp14-chi",
        tehais,
        [
            {"type": "tsumo", "actor": 0, "pai": "5s"},
            {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "4s"},
            {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": False},
            {"type": "chi", "actor": 2, "target": 1, "pai": "1p", "consumed": ["2p", "3p"]},
            {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": False},
            {
                "type": "ryukyoku",
                "reason": "exhaustive_draw",
                "deltas": [0, 0, 0, 0],
            },
        ],
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-chi", rows, [0, 1, 2, 2])
    # Capture-before-apply: the claim row itself predates its meld; the
    # following row of the same seat carries it.
    assert rows[2].actor_observation["visible_melds"][2] == []
    melds = rows[3].actor_observation["visible_melds"]
    assert melds[2] and melds[2][0]["kind"] == "chi"


def test_ankan_grows_dora_structural() -> None:
    tehais = [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
        ["5m", "6m", "7m", "8m", "9m", "1p", "1p", "1p", "2p", "3p", "4p", "5p", "6p"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "E", "E"],
        ["2p", "S", "W", "N", "P", "F", "C", "7m", "8m", "9m", "6p", "7p", "8p"],
    ]
    game = _kyoku_game(
        "wp14-ankan",
        tehais,
        [
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "7p"},
            {"type": "dahai", "actor": 1, "pai": "7p", "tsumogiri": True},
            {"type": "tsumo", "actor": 2, "pai": "5s"},
            {"type": "ankan", "actor": 2, "consumed": ["E", "E", "E", "E"]},
            {"type": "dora", "dora_marker": "6m"},
            {"type": "tsumo", "actor": 2, "pai": "9s"},
            {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
            {
                "type": "ryukyoku",
                "reason": "exhaustive_draw",
                "deltas": [0, 0, 0, 0],
            },
        ],
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-ankan", rows, [0, 1, 2, 2])
    assert rows[0].actor_observation["dora_indicators"] == [-1, -1, -1, -1, -1]
    revealed = [d for d in rows[3].actor_observation["dora_indicators"] if d != -1]
    assert len(revealed) == 1
    assert rows[2].actor_observation["visible_melds"][2] == []
    melds = rows[3].actor_observation["visible_melds"]
    assert melds[2] and melds[2][0]["kind"] == "ankan"


def test_daiminkan_and_kakan_structural() -> None:
    tehais = [
        ["7p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p"],
        ["7p", "7p", "5m", "6m", "8m", "9m", "1p", "1p", "2p", "3p", "4p", "5p", "6p"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "S", "S"],
        ["E", "S", "W", "N", "P", "F", "C", "9m", "9m", "2p", "2p", "4p", "6p"],
    ]
    game = _kyoku_game(
        "wp14-kan",
        tehais,
        [
            {"type": "tsumo", "actor": 0, "pai": "4p"},
            {"type": "dahai", "actor": 0, "pai": "7p", "tsumogiri": False},
            {"type": "pon", "actor": 1, "target": 0, "pai": "7p", "consumed": ["7p", "7p"]},
            {"type": "dahai", "actor": 1, "pai": "6p", "tsumogiri": False},
            {"type": "tsumo", "actor": 2, "pai": "9s"},
            {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": True},
            {"type": "tsumo", "actor": 3, "pai": "9p"},
            {"type": "dahai", "actor": 3, "pai": "9p", "tsumogiri": True},
            {"type": "tsumo", "actor": 0, "pai": "5p"},
            {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "7p"},
            {"type": "kakan", "actor": 1, "pai": "7p"},
            {"type": "tsumo", "actor": 1, "pai": "8p"},
            {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
            {
                "type": "ryukyoku",
                "reason": "exhaustive_draw",
                "deltas": [0, 0, 0, 0],
            },
        ],
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-kan", rows, [0, 1, 1, 2, 3, 0, 1, 1])


def test_ron_structural() -> None:
    # seat1 holds a closed tanyao tanki-5p tenpai; seat0 deals into it.
    tehais = [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"],
        ["2p", "3p", "4p", "5s", "6s", "7s", "6m", "7m", "8m", "2m", "3m", "4m", "5p"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "S", "S"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "4s", "S", "W", "N"],
    ]
    game = _kyoku_game(
        "wp14-ron",
        tehais,
        [
            {"type": "tsumo", "actor": 0, "pai": "4p"},
            {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False},
            {
                "type": "hora",
                "actor": 1,
                "target": 0,
                "pai": "5p",
                "deltas": [-8000, 12000, -2000, -2000],
            },
        ],
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-ron", rows, [0, 1])


def test_tsumo_win_structural() -> None:
    tehais = [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "E", "E", "S", "S"],
        ["5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "2p", "3p", "4p", "5p", "6p"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "4s", "S", "W", "N"],
        ["E", "S", "W", "N", "P", "F", "C", "7m", "8m", "9m", "6p", "7p", "8p"],
    ]
    game = _kyoku_game(
        "wp14-tsumo",
        tehais,
        [
            {"type": "tsumo", "actor": 0, "pai": "E"},
            {
                "type": "hora",
                "actor": 0,
                "target": 0,
                "pai": "E",
                "tsumo": True,
                "deltas": [12000, -4000, -4000, -4000],
            },
        ],
    )
    rows = replay_game(game)
    _assert_valid_rows("wp14-tsumo", rows, [0])


def _two_kyoku_events() -> list[dict[str, object]]:
    tehais = _fix_tehais()
    head: dict[str, object] = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": tehais,
    }
    second: dict[str, object] = {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "6m",
        "honba": 1,
        "kyoku": 2,
        "kyotaku": 0,
        "oya": 1,
        "scores": [28000, 22000, 25000, 25000],
        "tehais": tehais,
    }
    return [
        {"type": "start_game"},
        head,
        {"type": "tsumo", "actor": 0, "pai": "3p"},
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "8p"},
        {"type": "dahai", "actor": 1, "pai": "8p", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        second,
        {"type": "tsumo", "actor": 1, "pai": "5s"},
        {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "6s"},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
        {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [28000, 22000, 25000, 25000]},
    ]


def test_two_kyoku_carry_structural() -> None:
    game = _record("wp14-2kyoku", _two_kyoku_events())
    rows = replay_game(game)
    assert [(r.decision_id, r.seat, r.round_id) for r in rows] == [
        ("wp14-2kyoku:d0000", 0, "wp14-2kyoku:h00"),
        ("wp14-2kyoku:d0001", 1, "wp14-2kyoku:h00"),
        ("wp14-2kyoku:d0002", 1, "wp14-2kyoku:h01"),
        ("wp14-2kyoku:d0003", 2, "wp14-2kyoku:h01"),
    ]
    assert rows[2].actor_observation["dealer"] == 1
    assert rows[2].actor_observation["honba"] == 1
    assert rows[2].actor_observation["hand_number"] == 2
    for row in rows:
        assert row.actor_observation["legal_mask"][row.chosen_action_id] is True
        assert len(row.actor_observation["dora_indicators"]) == 5
