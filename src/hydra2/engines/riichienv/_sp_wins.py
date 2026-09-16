"""Single-engine wall-less replay: wins and boundaries."""

from __future__ import annotations

from dataclasses import replace as replace
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.engines.riichienv._sp_records import _emit as _emit
from hydra2.engines.riichienv._sp_records import _safe_mjai_type as _safe_mjai_type
from hydra2.engines.riichienv._sp_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._sp_windows import _check_drawer as _check_drawer
from hydra2.engines.riichienv._sp_windows import _clear_stash_no_claim as _clear_stash_no_claim
from hydra2.engines.riichienv._sp_windows import _emit_call_resolved as _emit_call_resolved
from hydra2.engines.riichienv._sp_windows import _live_step as _live_step
from hydra2.engines.riichienv._sp_windows import _require_actor as _require_actor
from hydra2.engines.riichienv.events import make_delta as make_delta
from hydra2.engines.riichienv.events import reason_kind as reason_kind

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_walk import _KyokuWalk as _KyokuWalk


# ---------------------------------------------------------------------------
# Wins, draws, boundaries.
# ---------------------------------------------------------------------------


def _do_hora(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    winner = _require_actor(state, kyoku, event, where="hora")
    tsumo_flag = bool(event.get("tsumo", False))
    target = event.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
        raise state.fail(kyoku, "hora", "hora without a valid target seat")
    # Tenhou-real tsumo wins carry no flag (and no pai); target==actor marks
    # them, while ron names the discarder.
    self_draw = tsumo_flag or target == winner
    if state.decided:
        # Double ron (two winners, one discard) is log-valid but unhandled:
        # the single-winner pipeline emits one ron row and quarantines here.
        # Bias note: co-winners never contribute rows, so ron rows skew
        # slightly toward single winners (rare: single-digit games per hundred).
        # TODO(multi-ron): emit second-winner rows in the DecisionRow schema.
        if (
            not self_draw
            and state.decided_by is not None
            and state.decided_by[0] == "ron"
            and state.decided_by[1] != winner
            and state.last_discard[0] is not None
            and target == state.last_discard[0]
        ):
            raise state.fail(
                kyoku, "hora", "double ron on one discard is quarantined (single-winner pipeline)"
            )
        raise state.fail(kyoku, "hora", "win after the kyoku was decided")
    deltas_raw = event.get("deltas")
    if not isinstance(deltas_raw, (list, tuple)) or len(deltas_raw) != 4:
        raise state.fail(kyoku, "hora", "hora without a 4-seat deltas quad")
    try:
        deltas = [int(d) for d in deltas_raw]
    except (TypeError, ValueError) as exc:
        raise state.fail(kyoku, "hora", f"hora deltas malformed: {exc}") from exc
    if self_draw:
        _check_drawer(state, walk, kyoku, winner, where="hora")
        step = _live_step(state, walk, kyoku, winner, mjai_type="hora")
        if step.drawn is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        drawn_str = tiles.mjai_string_of(step.drawn)
        offer: Any = None
        for offer_kind in ("hora", "tsumo"):
            try:
                offer = walk.tw._find(winner, mjai_type=offer_kind, tile_str=drawn_str)
            except ContractError:
                continue
            break
        if offer is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        tile = step.drawn
        if offer.tile is None or tiles.mjai_string_of(int(offer.tile)) != tiles.mjai_string_of(
            tile
        ):
            raise state.fail(kyoku, "hora", "tsumo tile differs from the drawn tile")
        step = replace(step, mjai_type="hora", tile=tile)
        expected = CanonicalAction(
            kind=cast("Any", "tsumo"),
            actor=make_seat(winner),
            tile=make_tile_id(tile),
            called_tile=None,
            consumed_tiles=(),
            source_seat=None,
            declares_riichi=False,
            metadata=(),
        )
        live_offered: tuple[int | None, int | None] = (
            state.last_discard[1],
            state.last_discard[0],
        )
        chosen_id = _strict_row(
            state,
            walk,
            kyoku,
            winner,
            step,
            expected,
            phase="draw_decision",
            turn_actor=winner,
            offered=live_offered,
        )
        try:
            walk.tw.do_tsumo_win(winner, tiles.mjai_string_of(tile))
        except (ContractError, ValueError) as exc:
            raise state.fail(kyoku, "hora", f"oracle win failed: {exc}") from exc
        source: int | None = None
    else:
        if state.last_discard[0] is None or state.last_discard[1] is None:
            raise state.fail(kyoku, "hora", "ron without a live discard offer")
        discarder = state.last_discard[0]
        if target != discarder:
            raise state.fail(kyoku, "hora", f"ron target {target} != discarder {discarder}")
        step = walk.stash.pop(winner, None)
        if step is None:
            raise state.fail(kyoku, "hora", f"seat {winner} holds no window step for ron")
        tile = state.last_discard[1]
        tile_str = tiles.mjai_string_of(tile)
        if not any(
            _safe_mjai_type(raw) == "hora"
            and raw.tile is not None
            and tiles.mjai_string_of(int(raw.tile)) == tile_str
            for raw in step.raw_legals
        ):
            raise state.fail(kyoku, "hora", f"seat {winner} oracle holds no hora offer")
        step = replace(step, mjai_type="hora", tile=tile)
        expected = CanonicalAction(
            kind=cast("Any", "ron"),
            actor=make_seat(winner),
            tile=make_tile_id(tile),
            called_tile=None,
            consumed_tiles=(),
            source_seat=make_seat(discarder),
            declares_riichi=False,
            metadata=(),
        )
        phase = "kan_response" if not state.opened_by_discard else "discard_response"
        chosen_id = _strict_row(
            state,
            walk,
            kyoku,
            winner,
            step,
            expected,
            phase=phase,
            turn_actor=discarder,
            offered=(tile, discarder),
        )
        source = discarder
        walk.stash.clear()
        if state.cr_pending:
            _emit_call_resolved(state, accepted=[chosen_id])
    scores = list(state.tracked_scores)
    state.tracked_scores = cast(
        "tuple[int, int, int, int]", tuple(scores[s] + deltas[s] for s in range(4))
    )
    _emit(
        state,
        kind="tsumo" if self_draw else "ron",
        visibility="public",
        actor=winner,
        tile=tile,
        action_id=chosen_id,
        source_seat=source,
        public_delta=(
            make_delta(("scores",), "set", list(deltas)),
            *(
                (make_delta(("riichi_sticks",), "increment", -step.sticks),)
                if step.sticks != 0
                else ()
            ),
        ),
    )
    state.decided = True
    state.decided_by = ("tsumo" if self_draw else "ron", winner)
    walk.drawer = None


def _do_ryukyoku(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    if state.decided:
        raise state.fail(kyoku, "ryukyoku", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="ryukyoku")
    reason = event.get("reason")
    if not isinstance(reason, str) or reason == "":
        # Tenhou-real MJAI omits the reason for wall exhaustion, first-turn
        # kyushu aborts, and four-reach aborts (the 4th reach emits without
        # its acceptance because the abort ends the kyoku first: never
        # synthesize it). Exhaustion always needs a full wall of draws;
        # kyushu is verified against the single engine's own nine-terminals
        # offer; yonin matches four declarations with three acceptances.
        # Anything between stays quarantined.
        if all(state.riichi_declared) and state.reach_accepted == 3:
            reason = "suucha_riichi"
        elif state.draws >= 60:
            reason = "exhaustive_draw"
        elif state.draws <= 4 and walk.drawer is not None and walk.tw.offers_kyushu(walk.drawer):
            reason = "kyushu_kyuhai"
        else:
            raise state.fail(
                kyoku, "ryukyoku", "draw without a reason string outside kyushu/exhaustive shape"
            )
    try:
        classification = reason_kind(reason)
    except ContractError as exc:
        raise state.fail(kyoku, "ryukyoku", str(exc)) from exc
    deltas_raw = event.get("deltas")
    if not isinstance(deltas_raw, (list, tuple)) or len(deltas_raw) != 4:
        raise state.fail(kyoku, "ryukyoku", "draw without a 4-seat deltas quad")
    try:
        deltas = [int(d) for d in deltas_raw]
    except (TypeError, ValueError) as exc:
        raise state.fail(kyoku, "ryukyoku", f"draw deltas malformed: {exc}") from exc
    pre = list(state.tracked_scores)
    post = [pre[s] + deltas[s] for s in range(4)]
    state.tracked_scores = cast("tuple[int, int, int, int]", tuple(post))
    if classification == "abortive_draw":
        from hydra2.engines.riichienv import events as _events

        try:
            mapped = _events.ABORTIVE_REASONS[reason]
        except KeyError as exc:
            raise state.fail(kyoku, "ryukyoku", f"unmapped abortive reason: {exc}") from exc
        _emit(
            state,
            kind="abortive_draw",
            visibility="public",
            round_index=state.hand_index,
            scores=post,
            reason=mapped,
            public_delta=(make_delta(("scores",), "set", list(deltas)),),
        )
    else:
        _emit(
            state,
            kind="draw_end",
            visibility="public",
            scores=post,
            reason=reason,
            public_delta=(make_delta(("scores",), "set", list(deltas)),),
        )
    state.decided = True
    walk.drawer = None


def _do_end_kyoku(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    following: dict[str, object] | None,
) -> None:
    _clear_stash_no_claim(state, walk, kyoku, where="end_kyoku")
    final = list(state.tracked_scores)
    next_round = state.hand_index + (1 if following is not None else 0)
    deltas: list[Any] = [
        make_delta(("scores",), "set", list(final)),
        make_delta(("round_index",), "set", next_round),
        *(make_delta(("riichi_states", seat), "set", "none") for seat in range(4)),
    ]
    if following is not None and str(following.get("type")) == "start_kyoku":
        try:
            deltas.append(make_delta(("honba",), "set", int(cast("Any", following["honba"]))))
            deltas.append(
                make_delta(("riichi_sticks",), "set", int(cast("Any", following["kyotaku"])))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise state.fail(kyoku, "end_kyoku", f"carry malformed: {exc}") from exc
    for seat in range(4):
        deltas.append(make_delta(("ippatsu", seat), "set", False))
    _emit(
        state,
        kind="round_end",
        visibility="public",
        round_index=state.hand_index,
        scores=final,
        public_delta=tuple(deltas),
    )
    walk.drawer = None


def _do_end_game(state: _GameState, walk: _KyokuWalk, kyoku: int) -> None:
    # The framed log always closes with end_game, but the engine path only
    # carries a game_end envelope when its own simulation went terminal (a
    # complete hanchan). Truncated wall-less streams must not invent one, so
    # the envelope is never emitted here -- only the completion flag.
    _clear_stash_no_claim(state, walk, kyoku, where="end_game")
    state.terminal = True
