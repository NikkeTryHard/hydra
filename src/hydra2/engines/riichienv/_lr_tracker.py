"""Wall-less sim replay: per-kyoku tracker mirrors."""

from __future__ import annotations

from typing import TYPE_CHECKING as TYPE_CHECKING

from hydra2.contracts.common import ContractError as ContractError
from hydra2.engines.riichienv._lr_frame import _END as _END
from hydra2.engines.riichienv._lr_track import _take_next as _take_next
from hydra2.engines.riichienv._lr_track import _track_remove as _track_remove
from hydra2.engines.riichienv._lr_track import _wall_filler_ids as _wall_filler_ids
from hydra2.engines.riichienv._lr_walk import _split_kyoku_draws as _split_kyoku_draws
from hydra2.engines.riichienv._oracle_base import _TRANSPARENT_KINDS as _TRANSPARENT_KINDS
from hydra2.engines.riichienv.tiles import mjai_string_of as mjai_string_of

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.engines.riichienv._lr_rows import _GameState as _GameState
    from hydra2.engines.riichienv._lr_walk import _KyokuWalk as _KyokuWalk


def _tracker_init(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    tehais: tuple[Any, ...],
    events: Sequence[dict[str, object]],
    start_idx: int,
) -> None:
    """Seed the log-faithful tracker from a kyoku's dealt hands (fail closed).

    Tile takes mirror the engine wall build exactly (tehais seats 0..3, then
    live draws in log order, then rinshan draws), so tracked physical copies
    agree with the engine tile-for-tile. Per-seat draw queues hold those
    copies in log order; the filler list covers pre-draws past the log.
    """
    if len(tehais) != 4:
        raise state.fail(kyoku, "start_kyoku", f"tehais cover {len(tehais)} seats, not 4")
    walk.take_taken = {}
    walk.tehais = tuple([str(t) for t in hand] for hand in tehais)
    walk.hands = []
    for seat in range(4):
        hand_tiles = list(tehais[seat])
        if len(hand_tiles) != 13:
            raise state.fail(
                kyoku, "start_kyoku", f"seat {seat} tehais hold {len(hand_tiles)} tiles, not 13"
            )
        tracked: list[int] = []
        for pai in hand_tiles:
            tracked.append(_take_next(walk, str(pai), state, kyoku, where="start_kyoku"))
        walk.hands.append(tracked)
    live_draws, rinshan_draws = _split_kyoku_draws(state, kyoku, events, start_idx)
    live_copies = [_take_next(walk, pai, state, kyoku, where="tsumo") for pai in live_draws]
    rinshan_copies = [_take_next(walk, pai, state, kyoku, where="tsumo") for pai in rinshan_draws]
    walk.live_total = len(live_copies)
    walk.rinshan_total = len(rinshan_copies)
    walk.live_used = 0
    walk.rinshan_used = 0
    walk.filler_ids = _wall_filler_ids(walk)
    walk.rivers = [[], [], [], []]
    walk.missed = set()
    walk.tsumo_counts = (0, 0, 0, 0)
    walk.first_draws = (None, None, None, None)
    walk.exp_drawer = None
    walk.kan_pending = False
    walk.kakan_added = [{}, {}, {}, {}]
    walk.sticks = state.kyotaku
    queues: list[list[list[int]]] = [[], [], [], []]
    live_pos = 0
    rinshan_pos = 0
    prev_kind = "start_kyoku"
    idx = start_idx + 1
    total = len(events)
    while idx < total:
        item = events[idx]
        if not isinstance(item, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = str(item.get("type", ""))
        if kind in _END or kind in ("start_kyoku", "end_kyoku"):
            break
        if kind in _TRANSPARENT_KINDS:
            idx += 1
            continue
        if kind == "tsumo":
            drawer = item.get("actor")
            drawn_pai = item.get("pai")
            if (
                isinstance(drawer, bool)
                or not isinstance(drawer, int)
                or not 0 <= drawer <= 3
                or not isinstance(drawn_pai, str)
                or drawn_pai == ""
            ):
                raise state.fail(kyoku, "tsumo", "malformed draw event")
            if prev_kind in ("ankan", "kakan", "daiminkan"):
                if rinshan_pos >= len(rinshan_copies):
                    raise state.fail(kyoku, "tsumo", "rinshan draw without a rinshan tile")
                queues[drawer].append([rinshan_copies[rinshan_pos], False])
                rinshan_pos += 1
            else:
                if live_pos >= len(live_copies):
                    raise state.fail(kyoku, "tsumo", "live draw without a live tile")
                queues[drawer].append([live_copies[live_pos], False])
                live_pos += 1
        prev_kind = kind
        idx += 1
    walk.draw_queues = queues


def _track_draw(state: _GameState, walk: _KyokuWalk, kyoku: int, actor: int, pai: str) -> None:
    """Mirror one logged draw in the tracker: turn order, furiten clear, hand."""
    if walk.kan_pending:
        expected: int | None = walk.exp_drawer
    elif state.last_discard[0] is not None:
        expected = (state.last_discard[0] + 1) % 4
    elif walk.exp_drawer is None:
        expected = state.dealer
    else:
        expected = walk.exp_drawer
    if actor != expected:
        raise state.fail(
            kyoku,
            "tsumo",
            f"seat {actor} tsumo breaks turn order (expected seat {expected}, out-of-turn draw)",
        )
    counts = list(walk.tsumo_counts)
    counts[actor] += 1
    walk.tsumo_counts = (counts[0], counts[1], counts[2], counts[3])
    if walk.first_draws[actor] is None:
        first = list(walk.first_draws)
        first[actor] = pai
        walk.first_draws = (first[0], first[1], first[2], first[3])
    if state.draws >= 70:
        raise state.fail(kyoku, "tsumo", "draw past the end of the wall")
    queue = walk.draw_queues[actor]
    if len(queue) == 0:
        raise state.fail(kyoku, "tsumo", f"seat {actor} drew a different tile than logged")
    copy, marked = queue[0]
    if mjai_string_of(copy) != pai:
        raise state.fail(kyoku, "tsumo", f"seat {actor} drew a different tile than logged")
    _ = queue.pop(0)  # discard consumed draw entry; copy/marked already captured
    if marked != 0:
        if copy not in walk.hands[actor]:
            raise state.fail(kyoku, "tsumo", "pre-drawn tile missing from tracked hand")
    else:
        walk.hands[actor].append(copy)
    if walk.kan_pending:
        walk.rinshan_used += 1
    else:
        walk.live_used += 1
    walk.exp_drawer = actor
    walk.kan_pending = False


def _mirror_predraw(
    state: _GameState, walk: _KyokuWalk, kyoku: int, discarder: int, *, chankan: bool
) -> None:
    """Mirror the engine's closed-window pre-draw for the next seat.

    Closed windows draw the next seat at the discard step. Past the logged
    draws the countdown wall deals deterministic filler. Entries are marked,
    never popped early, so each queued draw lands once; wall pointers advance
    at draw time for logged tiles and at mirror time for filler.
    """
    if chankan:
        _mirror_rinshan(state, walk, kyoku, discarder)
        return
    drawer = (discarder + 1) % 4
    queue = walk.draw_queues[drawer]
    if len(queue) > 0:
        if queue[0][1] != 0:
            raise state.fail(kyoku, "window", f"double pre-draw for seat {drawer}")
        if walk.live_used < walk.live_total:
            walk.hands[drawer].append(queue[0][0])
            queue[0][1] = True
            return
        # Live wall exhausted but this seat still queues rinshan tiles: the
        # engine deals live-pointer filler here (its rinshan waits for a kan).
    elif walk.live_used < walk.live_total:
        # Live wall lives on but this seat draws no more: the engine deals a
        # live tile owned by a later draw elsewhere (a forthcoming desync the
        # turn/draw checks own). Nothing log-faithful to mirror here.
        return
    if walk.filler_live_n >= len(walk.filler_ids):
        raise state.fail(kyoku, "window", "wall layout overlap past the live wall end")
    walk.hands[drawer].append(walk.filler_ids[walk.filler_live_n])
    walk.filler_live_n += 1
    walk.live_used += 1


def _mirror_rinshan(state: _GameState, walk: _KyokuWalk, kyoku: int, actor: int) -> None:
    """Mirror the engine's kan-step rinshan pre-draw (observed ahead of chankan)."""
    queue = walk.draw_queues[actor]
    if len(queue) > 0:
        if queue[0][1] != 0:
            raise state.fail(kyoku, "window", f"double pre-draw for seat {actor}")
        walk.hands[actor].append(queue[0][0])
        queue[0][1] = True
        return
    slot = 135 - walk.rinshan_total - walk.filler_rin_n
    index = slot - 52 - walk.live_total
    if index < 0 or index >= len(walk.filler_ids):
        raise state.fail(kyoku, "window", "wall layout overlap past the dead wall end")
    walk.hands[actor].append(walk.filler_ids[index])
    walk.filler_rin_n += 1
    walk.rinshan_used += 1


def _track_discard(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    actor: int,
    tile: int,
    *,
    drawn: int | None,
    where: str,
) -> None:
    """Mirror one logged discard: hand removal plus river, then drawer update."""
    if tile not in walk.hands[actor]:
        _ = _track_remove(
            walk.hands[actor], mjai_string_of(tile), state, kyoku, drawn=drawn, where=where
        )  # discard removed tile id; hand mutation is the effect
    else:
        walk.hands[actor].remove(tile)
    walk.rivers[actor].append(tile)
    walk.missed.discard(actor)
    walk.exp_drawer = actor
