"""WP-14 single-pass wall-less replay: live-engine offer-semantics pins.

``single_pass.replay_game`` walks one live ``riichienv.RiichiEnv`` per kyoku
instead of draining ``riichienv.MjaiReplay`` queues (see
:mod:`hydra2.engines.riichienv.log_replay`). Decisions, seats, chosen actions,
and observation content agree decision-for-decision with the drained path;
the legal masks differ exactly by the offer-source classes below, because the
live engine enforces rules the drained oracle does not (and surfaces options
the drain narrows away):

- kuikae withholding: after chi 5-6-7m the live engine withholds the
  kuikae discard 4m while the drained oracle still offers it;
- chi offer-completeness: the live engine offers every chi variant for the
  called tile while the drain surfaces only the taken one;
- kyushu timing: both paths surface nine-terminals abort on a seat's first
  draw with 9+ distinct terminals/honors even after another seat's plain
  discard — only a meld blocks kyuushu (Tenhou rule), so the pre-discard-only
  drain distinction from the previous pin no longer holds at 0.4.10
  (re-baselined with MjaiReplay evidence; upstream abortive-resolution
  rework #231).

A fourth test pins the derivation-marker rev bump (``v2``): single-pass rows
never mix silently with ``v1`` drained rows.
"""

from __future__ import annotations

from typing import Any

import pytest
from hydra2_replay_rs import tiles as _tiles_bridge

from hydra2.contracts.common import ContractError
from hydra2.data.decode import GameRecord
from hydra2.engines.riichienv import (
    _lr_end as drained,
)
from hydra2.engines.riichienv import (
    _sp_game as live,
)
from hydra2.engines.riichienv._lr_rows import SIM_DERIVATION_MARK as _DRAINED_SIM_MARK
from hydra2.engines.riichienv._sp_records import SIM_DERIVATION_MARK as _LIVE_SIM_MARK

mjai_string_of = _tiles_bridge.mjai_string_of

pytestmark = pytest.mark.contract_package("WP-14")


def _record(game_id: str, events: list[dict[str, object]]) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        object_id="wp14-sp-obj",
        packaged_object_id="wp14-sp-obj-pkg",
        events=tuple(events),
        raw_bytes_sha256="sha256:" + "0" * 64,
        wall_tiles=None,
        source={"type": "start_game"},
    )


def _invented(
    game_id: str, tehais: list[list[str]], body: list[dict[str, object]], *, oya: int = 0
) -> GameRecord:
    header: dict[str, object] = {
        "bakaze": "E",
        "dora_marker": "F",
        "honba": 0,
        "kyoku": 1,
        "kyotaku": 0,
        "oya": oya,
        "scores": [25000, 25000, 25000, 25000],
    }
    return _record(
        game_id,
        [
            {"type": "start_game"},
            {"type": "start_kyoku", "tehais": tehais, **header},
            *body,
            {"type": "end_kyoku"},
            {"type": "end_game", "scores": [25000, 25000, 25000, 25000]},
        ],
    )


def _mask_kinds(row: Any, table: Any) -> set[tuple[Any, ...]]:
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


def _table() -> Any:
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import (
        ACTION_TABLE_RELPATH,
        load_action_table,
    )

    return load_action_table(repo_root() / ACTION_TABLE_RELPATH)


def _both(game: GameRecord) -> tuple[list[Any], list[Any]]:
    quiet = {"split": "train"}
    old_rows = drained.replay_game(game, **quiet)
    new_rows = live.replay_game(game, **quiet)
    assert [r.seat for r in new_rows] == [r.seat for r in old_rows]
    assert [r.decision_id for r in new_rows] == [r.decision_id for r in old_rows]
    assert [r.chosen_action_id for r in new_rows] == [r.chosen_action_id for r in old_rows]
    return old_rows, new_rows


def test_kuikae_discard_withheld_live_offered_drained() -> None:
    """Chi 5-6-7m then 4m: engine withholds the kuikae discard, drain offers it."""
    tehais = [
        ["7m", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F"],
        ["4m", "5m", "6m", "P", "C", "1p", "2p", "3p", "1s", "2s", "3s", "E", "S"],
        ["1m", "1m", "9m", "9m", "1p", "1p", "9p", "9p", "1s", "1s", "9s", "9s", "E"],
        ["2m", "2m", "3m", "3m", "4p", "4p", "5p", "5p", "6p", "6p", "7p", "7p", "8p"],
    ]
    body = [
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "7m", "tsumogiri": False},
        {"type": "chi", "actor": 1, "target": 0, "pai": "7m", "consumed": ["5m", "6m"]},
        {"type": "dahai", "actor": 1, "pai": "P", "tsumogiri": False},
    ]
    old_rows, new_rows = _both(_invented("wp14-sp-kuikae", tehais, body))
    assert [r.seat for r in new_rows] == [0, 1, 1]
    table = _table()
    old, new = _mask_kinds(old_rows[2], table), _mask_kinds(new_rows[2], table)
    # The drained oracle still offers the kuikae discard; the live engine owns
    # the stricter offer, so the single-pass mask must not carry it while the
    # chosen discard stays legal on both sides.
    assert ("discard", "4m", None, ()) in old
    assert ("discard", "4m", None, ()) not in new
    assert new_rows[2].actor_observation["legal_mask"][new_rows[2].chosen_action_id] is True


def test_chi_variants_complete_live_taken_only_drained() -> None:
    """Called 5p with 3-4/4-6/6-7 in hand: live offers all, drain the taken."""
    tehais = [
        ["5p", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F"],
        ["3p", "4p", "6p", "7p", "C", "1m", "2m", "3m", "1s", "2s", "3s", "E", "S"],
        ["1m", "1m", "9m", "9m", "1p", "1p", "9p", "9p", "1s", "1s", "9s", "9s", "E"],
        ["2m", "2m", "3m", "3m", "4p", "4p", "6p", "6p", "7p", "7p", "8p", "8p", "C"],
    ]
    body = [
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": False},
        {"type": "chi", "actor": 1, "target": 0, "pai": "5p", "consumed": ["3p", "4p"]},
        {"type": "dahai", "actor": 1, "pai": "C", "tsumogiri": False},
    ]
    old_rows, new_rows = _both(_invented("wp14-sp-chi-variants", tehais, body))
    assert [r.seat for r in new_rows] == [0, 1, 1]
    table = _table()
    old_chi = {k for k in _mask_kinds(old_rows[1], table) if k[0] == "chi"}
    new_chi = {k for k in _mask_kinds(new_rows[1], table) if k[0] == "chi"}
    assert old_chi == {("chi", None, "5p", ("3p", "4p"))}
    assert new_chi == {
        ("chi", None, "5p", ("3p", "4p")),
        ("chi", None, "5p", ("4p", "6p")),
        ("chi", None, "5p", ("6p", "7p")),
    }


def test_kyushu_offered_live_past_first_discard() -> None:
    """9-terminal haipai, first draw after a discard: both paths offer kyushu."""
    tehais = [
        ["4p", "5p", "6s", "7s", "2s", "3s", "E", "S", "S", "W", "W", "N", "C"],
        ["2m", "3m", "4m", "2p", "3p", "6m", "7m", "8m", "5s", "6s", "7s", "8s", "C"],
        ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "2m", "3m", "4m", "5m"],
        ["6m", "7m", "8m", "5s", "6s", "7s", "8s", "3s", "4s", "5p", "6p", "7p", "C"],
    ]
    body = [
        {"type": "tsumo", "actor": 1, "pai": "8s"},
        {"type": "dahai", "actor": 1, "pai": "8s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "6m"},
        {"type": "dahai", "actor": 2, "pai": "6m", "tsumogiri": True},
    ]
    old_rows, new_rows = _both(_invented("wp14-sp-kyushu", tehais, body, oya=1))
    assert [r.seat for r in new_rows] == [1, 2]
    table = _table()
    old_kinds = {k[0] for k in _mask_kinds(old_rows[1], table)}
    new_kinds = {k[0] for k in _mask_kinds(new_rows[1], table)}
    # Engine 0.4.10 offers the nine-terminals abort on seat 2's first draw
    # with 9+ terminal/honor kinds even though seat 1 already discarded: only
    # a meld blocks kyuushu (Tenhou rule; no calls here), so the drained
    # oracle and the live engine agree. Raw-engine evidence at 0.4.10:
    # MjaiReplay yields KYUSHU_KYUHAI for seat 2's 6m draw; both row paths
    # carry abort_nine_terminals (upstream abortive-resolution rework #231
    # touches legal_actions.rs + replay state).
    assert "abort_nine_terminals" in old_kinds
    assert "abort_nine_terminals" in new_kinds
    assert new_rows[1].actor_observation["legal_mask"][new_rows[1].chosen_action_id] is True


def test_derivation_marker_rev_bump_pinned() -> None:
    """Single-pass rows carry marker rev v2, never silently mixing with v1."""
    assert _LIVE_SIM_MARK == "sim-replay-wall-less-v2"
    assert _DRAINED_SIM_MARK == "sim-replay-wall-less-v1"
    assert _LIVE_SIM_MARK != _DRAINED_SIM_MARK


def test_tampered_dahai_still_fails_closed() -> None:
    """Fail-closed desync contract survives the offer-selection change."""
    tehais = [
        ["7m", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F", "F"],
        ["4m", "5m", "6m", "P", "C", "1p", "2p", "3p", "1s", "2s", "3s", "E", "S"],
        ["1m", "1m", "9m", "9m", "1p", "1p", "9p", "9p", "1s", "1s", "9s", "9s", "E"],
        ["2m", "2m", "3m", "3m", "4p", "4p", "5p", "5p", "6p", "6p", "7p", "7p", "8p"],
    ]
    body = [
        {"type": "tsumo", "actor": 0, "pai": "C"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]
    game = _invented("wp14-sp-tamper", tehais, body)
    with pytest.raises(ContractError):
        live.replay_game(game)


def _two_kyoku_tehais() -> list[list[str]]:
    return [
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "E", "S", "W", "N", "P"],
        ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "E", "S", "W", "N", "F"],
        ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "E", "S", "W", "N", "C"],
        ["9m", "9p", "9s", "E", "S", "W", "N", "P", "F", "C", "1m", "1p", "1s"],
    ]


def _two_kyoku_game(game_id: str) -> Any:
    tehais = _two_kyoku_tehais()

    def head(kyoku: int, oya: int, honba: int, dora: str, scores: list[int]) -> dict[str, object]:
        return {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": dora,
            "honba": honba,
            "kyoku": kyoku,
            "kyotaku": 0,
            "oya": oya,
            "scores": scores,
            "tehais": tehais,
        }

    return _record(
        game_id,
        [
            {"type": "start_game"},
            head(1, 0, 0, "F", [25000, 25000, 25000, 25000]),
            {"type": "tsumo", "actor": 0, "pai": "9m"},
            {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True},
            {"type": "tsumo", "actor": 1, "pai": "9p"},
            {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
            {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
            {"type": "end_kyoku"},
            head(2, 1, 1, "6m", [25000, 25000, 25000, 25000]),
            {"type": "tsumo", "actor": 1, "pai": "5s"},
            {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True},
            {"type": "tsumo", "actor": 2, "pai": "6s"},
            {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
            {"type": "ryukyoku", "reason": "exhaustive_draw", "deltas": [0, 0, 0, 0]},
            {"type": "end_kyoku"},
            {"type": "end_game", "scores": [25000, 25000, 25000, 25000]},
        ],
    )


def _history_kinds(row: Any) -> list[str]:
    return [str(e["kind"]) for e in row.actor_observation["visible_history"]]


def test_histories_reset_per_kyoku_both_paths() -> None:
    """Histories live exactly one kyoku on both wall-less paths.

    Kyoku 0 opens ``[game_start, round_start, ...]`` (adapter-reset parity);
    later kyokus open ``[round_start, ...]`` with no game_start and none of
    the prior kyoku's envelopes. Decisions, seats, and chosen actions agree
    across the two paths throughout.
    """
    game = _two_kyoku_game("wp14-sp-two-kyoku")
    old_rows = drained.replay_game(game)
    new_rows = live.replay_game(game)
    assert [r.seat for r in new_rows] == [r.seat for r in old_rows] == [0, 1, 1, 2]
    assert [r.decision_id for r in new_rows] == [r.decision_id for r in old_rows]
    assert [r.chosen_action_id for r in new_rows] == [r.chosen_action_id for r in old_rows]
    for rows in (old_rows, new_rows):
        first, second = (
            [r for r in rows if r.round_id.endswith(":h00")],
            [r for r in rows if r.round_id.endswith(":h01")],
        )
        assert len(first) == len(second) == 2
        assert _history_kinds(first[0])[:2] == ["game_start", "round_start"]
        assert _history_kinds(second[0])[0] == "round_start"
        for row in second:
            assert "game_start" not in _history_kinds(row)
        assert max(len(r.actor_observation["visible_history"]) for r in second) <= max(
            len(r.actor_observation["visible_history"]) for r in first
        )


def test_history_cap_pinned_to_model_buckets() -> None:
    """The fail-closed cap equals the model's top history bucket (frozen)."""
    from hydra2.contracts.observation_assembly import HISTORY_EVENT_CAP
    from hydra2.models.schema import HISTORY_BUCKET_LENGTHS

    assert HISTORY_EVENT_CAP == HISTORY_BUCKET_LENGTHS[-1] == 256


def test_history_overflow_guard_fails_closed() -> None:
    """A 257-envelope history raises at encode; 256 encodes (never truncated)."""
    from hydra2.contracts.observation_assembly import (
        HISTORY_EVENT_CAP,
        ObservationBuilder,
    )
    from hydra2.engines.riichienv.events import make_envelope
    from hydra2.engines.riichienv.state import seat_winds_for_dealer
    from hydra2.models.encoder import encode_observations

    digest = "sha256:" + "0" * 64

    def fresh_builder() -> Any:
        return ObservationBuilder(
            game_id="wp14-sp-cap",
            rules_id="r",
            rules_hash=digest,
            action_table_hash=digest,
            expected_legal_mask_length=6792,
            event_schema_hash=digest,
            packet_boundary_hash=digest,
        )

    def ready(builder: Any) -> None:
        builder.update_public_state(
            decision_id="wp14-sp-cap:d0000",
            round_index=0,
            round_wind=27,
            hand_number=1,
            seat_winds=seat_winds_for_dealer(0),
            honba=0,
            riichi_sticks=0,
            dealer=0,
            scores=(25000, 25000, 25000, 25000),
            turn_actor=0,
            phase="draw_decision",
            live_wall_tiles_remaining=70,
            ippatsu_active=(False, False, False, False),
        )
        builder.set_concealed_hand(0, [])
        builder.set_actor_state(0, furiten="none", can_tsumo=True, can_riichi=False)

    def observe_with(n: int) -> Any:
        builder = fresh_builder()
        for seq in range(1, n + 1):
            builder.append_visible(
                make_envelope(
                    game_id="wp14-sp-cap",
                    sequence=seq,
                    kind="game_start",
                    visibility="public",
                    rules_hash=digest,
                    schema_hash=digest,
                    round_index=0,
                    scores=[25000, 25000, 25000, 25000],
                )
            )
        ready(builder)
        return builder.build(actor=0, legal_mask=(True,) * 6792)

    assert len(observe_with(HISTORY_EVENT_CAP).visible_history) == 256
    encode_observations([observe_with(HISTORY_EVENT_CAP)])
    with pytest.raises(ContractError, match="exceeds model bucket cap"):
        encode_observations([observe_with(HISTORY_EVENT_CAP + 1)])
