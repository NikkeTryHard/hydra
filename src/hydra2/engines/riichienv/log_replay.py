"""Wall-less sim replay: framed MJAI games -> actor ``DecisionRow`` rows (WP-14).

Real Tenhou MJAI carries no wall field, so the engine-reset path
(:mod:`hydra2.data.replay_expand`, which injects a full 136-tile
:class:`~hydra2.engines.protocol.WallSchedule`) cannot run. This module
replays through the PINNED simulator's own wall-less replay instead:
``riichienv.MjaiReplay`` installs ``start_kyoku`` tehais directly and treats
its RNG-shuffled wall as a pure countdown (hydra1 precedent
``event_handler.rs`` StartKyoku: hands installed, ``tile_count = 136 - 52``,
no 136-tile wall; actor-masked observe per ``state/mod.rs``
``mjai_log_per_player``). The engine owns the firewall: this module consumes
ONLY seat-filtered yielded observations (acting seat hand non-empty, every
other seat empty -- asserted per step, fail closed) and never touches
``Kyoku.hands`` / ``paishan`` / unfiltered event streams.

Row construction reuses the adapter's EXISTING conversion machinery, never a
parallel implementation:

- legal sets: the yielded ``Observation.legal_actions()`` (the engine's own
  seat-filtered offer) expanded through
  :func:`~hydra2.engines.riichienv.actions.expand_engine_legals` and
  :func:`~hydra2.engines.riichienv.actions.legal_view`;
- chosen actions: yielded ``Action`` objects mapped to canonical form (twin
  resolution follows the logged flags plus deterministic copy rules, exactly
  like the engine path's log matching) and encoded with the canonical codec;
- observations: assembled with the canonical
  :class:`~hydra2.contracts.observation.ObservationBuilder` from the same
  envelope constructors (:mod:`hydra2.engines.riichienv.events`) the adapter
  uses, in the same emission order, so a walled fixture agrees with the
  engine path decision-for-decision (same decision ids, seats, chosen
  actions, and string-level observation content);
- history: the framed log's MJAI events translated envelope-for-envelope in
  adapter order (turn/draw, discard, claims, windows, dora, wins, draws).

Copy-identity folding: ``MjaiReplay`` reports string-canonical tile ids (the
same id for every copy of one MJAI string), while a live engine deals
distinct physical copies from its unique wall. Rows therefore fold
physical-copy variants (smallest-copy choice everywhere, exactly like the
engine path's deterministic copy resolution) and rebuild meld tiles from the
logged strings deterministically. Consequence: ``observation_hash`` values
differ from the engine path exactly by copy identity (same tile strings,
different physical copies), plus the wall-less derivation marker. Agreement
tests assert the identical part (ids, seats, actions, string-level content)
and pin the folding explicitly.

CRITICAL wall binding: a placeholder-wall digest is NEVER bound. ``wall_id``
is always ``None`` (the :func:`_wall_id_for` None-rule: never invented) and
the derivation carries the wall-less marker
:data:`SIM_DERIVATION_MARK` instead of a wall digest. Desyncs (riichienv
``InvalidState`` / ``Replay desync``, illegal offered actions, firewall
breaches) raise :class:`~hydra2.contracts.common.ContractError` naming
game + kyoku + step, so upstream callers quarantine and count them instead
of silently dropping rows.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.action import (
    ActionContext,
    CanonicalAction,
    canonical_action_codec,
    load_action_table,
)
from hydra2.contracts.common import (
    ContractError,
    IllegalActionError,
    make_digest_text,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.observation import (
    HISTORY_EVENT_CAP,
    ObservationBuilder,
    VisibleMeld,
    visible_meld_id,
)
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.parquet import DecisionRow
from hydra2.data.stream import verify_no_privileged_leakage
from hydra2.engines.riichienv.actions import legal_view
from hydra2.engines.riichienv.events import (
    make_delta,
    make_envelope,
    meld_delta_value,
    reason_kind,
)
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY
from hydra2.engines.riichienv.state import seat_winds_for_dealer
from hydra2.engines.riichienv.tiles import mjai_string_of, physical_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.rules import RulesManifest

__all__ = ["SIM_DERIVATION_MARK", "replay_game"]

#: Wall-less derivation marker bound into ``derivation_hash`` instead of a
#: wall digest (placeholder digests are never bound).
SIM_DERIVATION_MARK = "sim-replay-wall-less-v1"

#: Live-wall countdown base: 136 tiles minus 4x13 dealt minus the 14-tile
#: dead wall (mirrors the tehais-install countdown semantics).
_LIVE_WALL_BASE = 136 - 52 - 14

#: Bakaze letter -> round-wind TileType (same scale as seat winds, 27..30).
_BAKAZE_TO_WIND = {"E": 27, "S": 28, "W": 29, "N": 30}
#: Red-five aliases folded for offer counting (aka counts as its kind).
_RED_NORM = {"5mr": "5m", "0m": "5m", "5pr": "5p", "0p": "5p", "5sr": "5s", "0s": "5s"}


def _norm_pai(pai: str) -> str:
    """Red-normalized MJAI string (``5pr`` folds to ``5p``)."""
    return _RED_NORM.get(pai, pai)


#: Terminal/honor kinds for the kyushu nine-terminals offer (normalized).
_YAOCHU = frozenset({"1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"})

#: Round-wind letter -> Conditions wind int (E/S/W/N = 0/1/2/3).
_ROUND_WIND_TO_INT = {"E": 0, "S": 1, "W": 2, "N": 3}

#: Every tile kind for furiten wait enumeration (fixed order, deterministic).
_ALL_TILE_KINDS = (
    *(f"{value}{suit}" for suit in "mps" for value in range(1, 10)),
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
)

_MELD_TYPE_BY_KIND = {
    "chi": riichienv.MeldType.Chi,
    "pon": riichienv.MeldType.Pon,
    "daiminkan": riichienv.MeldType.Daiminkan,
    "kakan": riichienv.MeldType.Kakan,
    "ankan": riichienv.MeldType.Ankan,
}


def _take_next(walk: _KyokuWalk, pai: str, state: _GameState, kyoku: int, *, where: str) -> int:
    """Take the next pool copy of ``pai`` in global wall order (fail closed).

    The take sequence mirrors the engine wall build exactly (tehais seats
    0..3, then live draws in log order, then rinshan draws), so tracked
    physical copies agree with the engine tile-for-tile, aka included.
    """
    pool = _copies_of_string(pai)
    taken = walk.take_taken.get(pai, 0)
    if taken >= len(pool):
        raise state.fail(kyoku, where, f"tile string {pai!r} overused (tile conservation)")
    walk.take_taken[pai] = taken + 1
    return pool[taken]


def _wall_filler_ids(walk: _KyokuWalk) -> list[int]:
    """Deterministic filler ids: numeric complement of every taken copy."""
    used: set[int] = set()
    for pai, count in walk.take_taken.items():
        used.update(_copies_of_string(pai)[:count])
    return sorted(set(range(136)) - used)


def _track_remove(
    hand: list[int], pai: str, state: _GameState, kyoku: int, *, drawn: int | None, where: str
) -> int:
    """Remove one tracked copy rendering ``pai`` (drawn tile preferred, fail closed)."""
    if drawn is not None and mjai_string_of(drawn) == pai and drawn in hand:
        hand.remove(drawn)
        return drawn
    for tile in hand:
        if mjai_string_of(tile) == pai:
            hand.remove(tile)
            return tile
    raise state.fail(kyoku, where, f"no tracked copy of discard {pai!r} in hand")


def _track_remove_consumed(
    hand: list[int], consumed: Sequence[str], state: _GameState, kyoku: int, *, where: str
) -> None:
    """Remove one tracked copy per logged consumed string (fail closed)."""
    for pai in sorted(s for s in consumed):
        for tile in hand:
            if mjai_string_of(tile) == pai:
                hand.remove(tile)
                break
        else:
            raise state.fail(kyoku, where, f"no tracked copy left for meld tile {pai!r}")


def _chi_offered(hand: Sequence[int], tile: str, *, kamicha: bool) -> bool:
    """Whether a concealed hand can chi ``tile`` (normalized counting, kamicha only)."""
    if not kamicha or len(tile) != 2 or tile[0] not in "123456789" or tile[1] not in "mps":
        return False
    counts: dict[str, int] = {}
    for raw in hand:
        rendered = _norm_pai(mjai_string_of(raw))
        counts[rendered] = counts.get(rendered, 0) + 1
    value = int(tile[0])
    suit = tile[1]
    patterns: list[tuple[int, int]] = []
    if value >= 3:
        patterns.append((value - 2, value - 1))
    if 2 <= value <= 8:
        patterns.append((value - 1, value + 1))
    if value <= 7:
        patterns.append((value + 1, value + 2))
    return any(
        counts.get(f"{low}{suit}", 0) > 0 and counts.get(f"{high}{suit}", 0) > 0
        for low, high in patterns
    )


def _eval_melds(state: _GameState, walk: _KyokuWalk, seat: int) -> list[Any]:
    """Tracked melds as evaluator melds (kakan upgrades join their prior pon)."""
    upgraded = walk.kakan_added[seat]
    out: list[Any] = []
    for meld in state.melds[seat]:
        kind = str(meld.kind)
        tiles = sorted(int(t) for t in meld.tiles)
        if kind == "pon" and (int(meld.tiles[0]) // 4) in upgraded:
            kind = "kakan"
            tiles = sorted([*tiles, upgraded[int(meld.tiles[0]) // 4]])
        out.append(
            riichienv.Meld(
                tiles=tiles,
                meld_type=_MELD_TYPE_BY_KIND[kind],
                opened=(kind != "ankan"),
            )
        )
    return out


def _win_is_win(
    concealed: Sequence[int],
    melds: Sequence[Any],
    tile: int,
    *,
    riichi: bool,
    player_wind: int,
    round_wind: int,
    chankan: bool,
) -> bool:
    """Yaku-aware win check through the pinned evaluator (fail closed on misuse)."""
    evaluator = riichienv.HandEvaluator(sorted(t for t in concealed), list(melds))
    conditions = riichienv.Conditions(
        riichi=riichi, player_wind=player_wind, round_wind=round_wind, chankan=chankan
    )
    try:
        return evaluator.calc(tile, conditions=conditions).is_win
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"win evaluation failed: {exc}") from exc


def _shape_is_win(concealed: Sequence[int], melds: Sequence[Any], tile: int) -> bool:
    """Yaku-blind shape check (furiten wait enumeration)."""
    evaluator = riichienv.HandEvaluator(sorted(t for t in concealed), list(melds))
    try:
        return evaluator.calc(tile).has_win_shape
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"shape evaluation failed: {exc}") from exc


def _responder_eval(
    state: _GameState,
    walk: _KyokuWalk,
    seat: int,
    tile: str,
    *,
    chankan: bool,
) -> tuple[bool, bool, bool]:
    """Per-responder window facts (normalized): ``(shape, win, r2clean)``."""
    concealed = walk.hands[seat]
    melds = _eval_melds(state, walk, seat)
    win_tile = _copies_of_string(tile)[0]
    if not _shape_is_win(concealed, melds, win_tile):
        return (False, False, False)
    river = {_norm_pai(mjai_string_of(t)) for t in walk.rivers[seat]}
    if _norm_pai(tile) in river:
        return (True, False, False)
    for kind in river:
        if _shape_is_win(concealed, melds, _copies_of_string(kind)[0]):
            return (True, False, False)
    win = _win_is_win(
        concealed,
        melds,
        win_tile,
        riichi=state.riichi_declared[seat],
        player_wind=(seat - state.dealer) % 4,
        round_wind=_ROUND_WIND_TO_INT[state.bakaze],
        chankan=chankan,
    )
    return (True, win, True)


def _thin_window_open(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    discarder: int,
    tile: int,
    *,
    chankan: bool,
) -> bool:
    """Python claim-window predicate over tracked hands (no engine)."""
    del kyoku
    tile_norm = _norm_pai(mjai_string_of(tile))
    for seat in range(4):
        if seat == discarder:
            continue
        if state.riichi_declared[seat]:
            _, win, r2clean = _responder_eval(state, walk, seat, tile_norm, chankan=chankan)
            if win and r2clean and seat not in walk.missed:
                return True
            continue
        counts: dict[str, int] = {}
        for raw in walk.hands[seat]:
            rendered = _norm_pai(mjai_string_of(raw))
            counts[rendered] = counts.get(rendered, 0) + 1
        if not chankan and counts.get(tile_norm, 0) >= 2:
            return True
        if (
            not chankan
            and seat == (discarder + 1) % 4
            and _chi_offered(walk.hands[seat], tile_norm, kamicha=True)
        ):
            return True
    return False


def _thin_furiten(state: _GameState, walk: _KyokuWalk, seat: int) -> str:
    """Python furiten state: riichi declarations plus tracked ron passes."""
    if state.riichi_declared[seat]:
        return "riichi"
    if seat in walk.missed:
        return "temporary"
    return "none"


def _thin_kyushu(state: _GameState, walk: _KyokuWalk, seat: int) -> bool:
    """Python nine-terminals offer: first turn with 9+ terminal/honor kinds."""
    del state
    if walk.tsumo_counts[seat] != 1:
        return False
    first = walk.first_draws[seat]
    if first is None:
        return False
    kinds = {
        rendered
        for raw in [*list(walk.tehais[seat]), first]
        if (rendered := _norm_pai(raw)) in _YAOCHU
    }
    return len(kinds) >= 9


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


def _assert_tracker_matches_engine(
    state: _GameState, walk: _KyokuWalk, kyoku: int, *, where: str
) -> None:
    """Slice-1 tripwire: tracked hands/rivers equal the throwaway engine state."""
    try:
        hands: Any = walk.tw._engine.hands
        rivers: Any = walk.tw._engine.discards
    except Exception as exc:
        raise state.fail(kyoku, where, f"tracker cross-check query failed: {exc}") from exc
    for seat in range(4):
        engine_hand = sorted(mjai_string_of(int(t)) for t in hands[seat])
        tracked_hand = sorted(mjai_string_of(t) for t in walk.hands[seat])
        if engine_hand != tracked_hand:
            # The countdown wall pre-draws ahead of the log: once logged draws
            # run out the engine holds filler the log never names. Filler is
            # invisible to rows (actor-masked) and offer-guarded by the thin
            # window assert; anything the log DOES name must match exactly.
            remaining = list(engine_hand)
            for rendered in tracked_hand:
                if rendered in remaining:
                    remaining.remove(rendered)
                else:
                    raise state.fail(
                        kyoku, where, f"tracker hand differs from engine state for seat {seat}"
                    )
            queued = [mjai_string_of(entry[0]) for entry in walk.draw_queues[seat]]
            for rendered in remaining:
                if rendered in queued:
                    raise state.fail(
                        kyoku,
                        where,
                        f"tracker missed a logged pre-draw for seat {seat}",
                    )
        engine_river = [mjai_string_of(int(t)) for t in rivers[seat]]
        tracked_river = [mjai_string_of(t) for t in walk.rivers[seat]]
        if engine_river != tracked_river:
            raise state.fail(
                kyoku, where, f"tracker river differs from engine state for seat {seat}"
            )


def _assert_thin_row_matches(
    state: _GameState, walk: _KyokuWalk, kyoku: int, seat: int, step: _SimStep, furiten: str
) -> None:
    """Slice-1 tripwire: thin furiten/counters equal oracle/engine values at rows."""
    thin = _thin_furiten(state, walk, seat)
    if thin != furiten:
        raise state.fail(
            kyoku,
            f"seat {seat} row",
            f"furiten derivation mismatch: thin {thin!r} engine {furiten!r}",
        )
    if walk.sticks != step.sticks:
        raise state.fail(kyoku, f"seat {seat} row", "tracked sticks differ from oracle sticks")
    if tuple(state.riichi_declared) != tuple(step.riichi):
        raise state.fail(
            kyoku, f"seat {seat} row", "tracked riichi differs from oracle declarations"
        )
    if walk.live_used < 70:
        _assert_tracker_matches_engine(state, walk, kyoku, where=f"seat {seat} row")


#: MJAI framing vocabulary: the single shared source is replay_expand's
#: closed sets (do not fork a second vocabulary here). Bound once at import
#: (module constants, UPPER by convention).
def _vocabulary() -> tuple[
    frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]
]:
    from hydra2.data import replay_expand as _re

    return (_re._START_TYPES, _re._END_TYPES, _re._SKIP_TYPES, _re._ROW_TYPES, _re._CLAIM_TYPES)


_START, _END, _SKIP, _ROW, _CLAIM = _vocabulary()


def _rules() -> RulesManifest:
    from hydra2.data import replay_expand as _re

    return _re._load_rules()


def _adapter_hash() -> str:
    from hydra2.data import replay_expand as _re

    return _re._adapter_hash()


_TABLE_CACHE: dict[str, Any] = {}
_EVENT_SCHEMA_HASH_CACHE: dict[str, str] = {}
_PACKET_BOUNDARY_HASH_CACHE: dict[str, str] = {}
_RULES_HASH_CACHE: dict[str, str] = {}


def _table() -> Any:
    from hydra2.config import repo_root
    from hydra2.contracts.action import ACTION_TABLE_RELPATH

    root = str(repo_root())
    if root not in _TABLE_CACHE:
        _TABLE_CACHE[root] = load_action_table(Path(root) / ACTION_TABLE_RELPATH)
    return _TABLE_CACHE[root]


def _event_schema_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event_schema import EVENT_SCHEMA_RELPATH, parse_event_schema

    root = str(repo_root())
    if root not in _EVENT_SCHEMA_HASH_CACHE:
        document: dict[str, Any] = cast(
            "dict[str, Any]",
            parse_event_schema((Path(root) / EVENT_SCHEMA_RELPATH).read_bytes()),
        )
        payload: Any = document["payload"]
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ContractError("event schema artifact lacks a digest")
        _EVENT_SCHEMA_HASH_CACHE[root] = str(cast("Any", payload["digest"]))
    return _EVENT_SCHEMA_HASH_CACHE[root]


def _packet_boundary_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event_packet import (
        build_packet_boundary_payload,
        compute_event_schema_digest,
    )

    # Perf-C P1b: the boundary payload is process-constant; hashing it per
    # builder (per kyoku) re-ran canonicalization for an identical digest.
    root = str(repo_root())
    if root not in _PACKET_BOUNDARY_HASH_CACHE:
        _PACKET_BOUNDARY_HASH_CACHE[root] = str(
            compute_event_schema_digest(build_packet_boundary_payload())
        )
    return _PACKET_BOUNDARY_HASH_CACHE[root]


def _rules_hash(manifest: RulesManifest, recomputed: str) -> str:
    """Published rules bytes win when present (same authority as the adapter)."""
    import hashlib

    from hydra2.config import repo_root

    # Perf-C P1b: published rules bytes are immutable mid-run; hash once per
    # (root, rules_id). The no-published-file branch stays uncached and
    # returns ``recomputed`` verbatim, exactly like before.
    root = str(repo_root())
    key = f"{root}\x00{manifest.rules_id}"
    if key not in _RULES_HASH_CACHE:
        published = Path(root) / "configs" / "rules" / f"{manifest.rules_id}.json"
        if published.is_file():
            _RULES_HASH_CACHE[key] = "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
    cached = _RULES_HASH_CACHE.get(key)
    return cached if cached is not None else recomputed


def _sim_game_id(game: GameRecord, *, rules_hash: str) -> str:
    """Builder game id with the engine path's exact scheme when walled.

    A walled game replays row-identical to :func:`expand_game`, so the
    builder identity (wall-derived seed material) is reproduced exactly from
    the real wall. A wall-less game has no wall to bind -- the log's game id
    is carried instead (never a placeholder digest).
    """
    if game.wall_tiles is None:
        if game.game_id == "":
            raise ContractError("wall-less game has no game_id for sim replay")
        return game.game_id
    if len(game.wall_tiles) != 136:
        raise ContractError(f"wall_tiles must carry 136 tiles, got {len(game.wall_tiles)}")
    from hydra2.contracts.common import make_tile_id as _tid
    from hydra2.engines.protocol import wall_schedule_digest

    physical = tuple(_tid(t) for t in game.wall_tiles)
    schedule_id = f"replay-{game.game_id}"
    wall_digest = str(wall_schedule_digest(schedule_id, physical))
    seed_material = of_canonical(
        {
            "rules_hash": rules_hash,
            "wall_digest": wall_digest,
            "seat_permutation": [0, 1, 2, 3],
            "adapter_version": str(ENGINE_IDENTITY.adapter_version),
        }
    )
    return f"hydra2-riichienv-{str(seed_material).removeprefix('sha256:')[:16]}"


# ---------------------------------------------------------------------------
# Oracle records (frozen primitives extracted eagerly per yielded step).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SimStep:
    """One yielded ``(Observation, Action)`` pair as frozen primitives."""

    seat: int
    mjai_type: str
    tile: int | None
    consume: tuple[int, ...]
    mjai: dict[str, object]
    raw_action: Any
    raw_legals: tuple[Any, ...]
    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    discards: tuple[tuple[int, ...], ...]
    hands_lens: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    sticks: int
    oya: int
    honba: int
    round_wind: int
    player_id: int
    riichi: tuple[bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class _TablePosition:
    """Frozen throwaway-engine snapshot for oracle-less forced rows."""

    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    rivers: tuple[tuple[int, ...], ...]
    lens: tuple[int, int, int, int]
    sticks: int
    declared: tuple[bool, bool, bool, bool]
    legals: tuple[Any, ...]


def _coerce_game(game: GameRecord | bytes | str | Path) -> GameRecord:
    if isinstance(game, GameRecord):
        return game
    if isinstance(game, (str, Path)):
        raw = Path(game).read_bytes()
        object_id = f"simreplay-path-{Path(game).name}"
    else:
        raw = game
        import hashlib

        object_id = "simreplay-bytes-" + hashlib.sha256(raw).hexdigest()[:16]
    return decode_game_object(object_id=object_id, packaged_object_id=object_id, decoded_bytes=raw)


def _frame_bytes(game: GameRecord) -> bytes:
    return ("\n".join(json.dumps(event) for event in game.events) + "\n").encode("utf-8")


def _drain_game_steps(staged_path: str, *, game_id: str) -> list[list[list[_SimStep]]]:
    """Drain ``steps(seat)`` for seats 0..3 of every kyoku (fail closed).

    One ``from_jsonl``/``take_kyokus`` pass per game; each ``(kyoku, seat)``
    step iterator replays independently (verified: sequential drains of one
    kyoku object yield identical per-seat queues to fresh replays).
    """

    def _fail(kyoku_index: int, why: str) -> ContractError:
        return ContractError(f"sim replay desync game {game_id!r} kyoku {kyoku_index}: {why}")

    try:
        replay = riichienv.MjaiReplay.from_jsonl(staged_path)
    except ContractError:
        raise
    except Exception as exc:
        raise _fail(-1, f"MjaiReplay.from_jsonl failed: {exc}") from exc
    try:
        kyokus = list(replay.take_kyokus())
    except ContractError:
        raise
    except Exception as exc:
        raise _fail(-1, f"take_kyokus failed: {exc}") from exc
    drained: list[list[list[_SimStep]]] = []
    for kyoku_index, kyoku in enumerate(kyokus):
        queues: list[list[_SimStep]] = [[], [], [], []]
        for seat in range(4):
            try:
                pairs = list(kyoku.steps(seat=seat))
            except ContractError:
                raise
            except Exception as exc:
                raise _fail(kyoku_index, f"seat {seat} steps failed: {exc}") from exc
            for position, (obs, act) in enumerate(pairs):
                try:
                    queues[seat].append(_extract_step(seat, obs, act, game_id=game_id))
                except ContractError:
                    raise
                except Exception as exc:
                    raise _fail(kyoku_index, f"seat {seat} step {position}: {exc}") from exc
        drained.append(queues)
    return drained


def _extract_step(seat: int, obs: Any, act: Any, *, game_id: str) -> _SimStep:
    """Freeze one yielded pair; the engine-masked firewall is asserted here."""
    where = f"game {game_id!r} seat {seat}"
    hands: Any = obs.hands
    lens = tuple(len(h) for h in hands)
    if len(lens) != 4:
        raise ContractError(f"{where}: seat-filtered hands must cover 4 seats")
    for other in range(4):
        if other != seat and lens[other] != 0:
            raise ContractError(
                f"{where}: firewall breach -- seat {other} hand leaks "
                f"{lens[other]} tiles into seat {seat}'s observation"
            )
    if lens[seat] == 0:
        raise ContractError(f"{where}: acting-seat hand is empty (desync)")
    try:
        mjai_raw: Any = act.to_mjai()
    except Exception as exc:
        raise ContractError(f"{where}: yielded action has no MJAI mapping: {exc}") from exc
    if isinstance(mjai_raw, str):
        try:
            mjai_raw = json.loads(mjai_raw)
        except ValueError as exc:
            raise ContractError(f"{where}: yielded action MJAI unparseable: {exc}") from exc
    mjai: Any = mjai_raw
    if not isinstance(mjai, dict):
        raise ContractError(f"{where}: yielded action has no MJAI mapping")
    kind = mjai.get("type")
    if not isinstance(kind, str) or kind == "":
        raise ContractError(f"{where}: yielded action without a type: {mjai!r}")
    tile_raw: Any = act.tile
    tile = None if tile_raw is None else int(tile_raw)
    consume = tuple(sorted(int(t) for t in act.consume_tiles))
    hand = _distinct_copies(tuple(int(t) for t in obs.hand))
    drawn_raw: Any = obs.drawn_tile
    drawn = None if drawn_raw is None else int(drawn_raw)
    dora = tuple(int(t) for t in obs.dora_indicators)
    discards = tuple(tuple(int(t) for t in river) for river in obs.discards)
    scores_raw: Any = obs.scores
    scores = tuple(int(s) for s in scores_raw)
    if len(scores) != 4:
        raise ContractError(f"{where}: scores must cover 4 seats")
    riichi = tuple(bool(v) for v in obs.riichi_declared)
    if len(riichi) != 4:
        raise ContractError(f"{where}: riichi_declared must cover 4 seats")
    legals: Any = obs.legal_actions()
    return _SimStep(
        seat=seat,
        mjai_type=kind,
        tile=tile,
        consume=consume,
        mjai=dict(mjai),
        raw_action=act,
        raw_legals=tuple(legals),
        hand=hand,
        drawn=drawn,
        dora=dora,
        discards=discards,
        hands_lens=lens,
        scores=scores,
        sticks=int(obs.riichi_sticks),
        oya=int(obs.oya),
        honba=int(obs.honba),
        round_wind=int(obs.round_wind),
        player_id=int(obs.player_id),
        riichi=riichi,
    )


# ---------------------------------------------------------------------------
# Tile-copy helpers (string-canonical ids + deterministic meld rebuild).
# ---------------------------------------------------------------------------


def _copies_of_string(pai: str) -> list[int]:
    """Ordered physical copies for one MJAI string (red-aware)."""

    first = int(physical_of(pai))
    if pai in ("5mr", "0m"):
        return [16]
    if pai in ("5pr", "0p"):
        return [52]
    if pai in ("5sr", "0s"):
        return [88]
    base = (first // 4) * 4
    if first == base + 1 and pai[0] == "5":
        # Plain five: the red copy (base) belongs to the "5xr" string.
        return [base + 1, base + 2, base + 3]
    return [base, base + 1, base + 2, base + 3]


# ---------------------------------------------------------------------------
# Per-game replay state.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GameState:
    game: GameRecord
    split: str
    seat_filter: int | None
    rules: Any
    rules_hash: str
    table: Any
    sim_game_id: str
    builder: ObservationBuilder
    rows: list[DecisionRow] = None  # type: ignore[assignment]
    seq: int = 0
    event_count: int = 0
    hand_index: int = -1
    round_idx: int = -1
    draws: int = 0
    declarations: int = 0
    riichi_declared: tuple[bool, bool, bool, bool] = (False, False, False, False)
    ippatsu: tuple[bool, bool, bool, bool] = (False, False, False, False)
    tracked_scores: tuple[int, int, int, int] = (0, 0, 0, 0)
    melds: tuple[list[VisibleMeld], ...] = None  # type: ignore[assignment]
    last_discard: tuple[int | None, int | None] = (None, None)
    opened_by_discard: bool = False
    cr_pending: bool = False
    dealer: int = 0
    bakaze: str = "E"
    honba: int = 0
    kyotaku: int = 0
    hand_number: int = 1
    decided: bool = False
    decided_by: tuple[str, int] | None = None
    terminal: bool = False
    last_kind: str | None = None

    def fail(self, kyoku: int, step: str, why: str) -> ContractError:
        return ContractError(
            f"sim replay desync game {self.game.game_id!r} kyoku {kyoku} {step}: {why}"
        )


def _ippatsu_open(state: _GameState, actor: int) -> None:
    """Open one seat's ippatsu window (adapter e80184a: riichi_accepted)."""
    flags = list(state.ippatsu)
    flags[actor] = True
    state.ippatsu = cast("tuple[bool, bool, bool, bool]", tuple(flags))


def _ippatsu_interrupt(state: _GameState, seat: int | None) -> None:
    """Clear ippatsu (adapter e80184a): seat's own next discard, or every
    seat on any meld/kan interrupt (seat=None)."""
    if seat is None:
        state.ippatsu = (False, False, False, False)
    else:
        flags = list(state.ippatsu)
        flags[seat] = False
        state.ippatsu = cast("tuple[bool, bool, bool, bool]", tuple(flags))


# ---------------------------------------------------------------------------
# Envelope emission (adapter order, adapter constructors).
# ---------------------------------------------------------------------------


def _emit(
    state: _GameState,
    *,
    kind: str,
    visibility: str,
    actor: int | None = None,
    tile: int | None = None,
    action_id: int | None = None,
    source_seat: int | None = None,
    consumed_tiles: Sequence[int] = (),
    offered_action_ids: Sequence[int] = (),
    accepted_action_ids: Sequence[int] = (),
    round_index: int | None = None,
    scores: Sequence[int] | None = None,
    reason: str | None = None,
    public_delta: Sequence[Any] = (),
) -> None:

    envelope = make_envelope(
        game_id=state.sim_game_id,
        sequence=state.event_count,
        kind=cast("Any", kind),
        visibility=cast("Any", visibility),
        rules_hash=state.rules_hash,
        schema_hash=_event_schema_hash(),
        actor=actor,
        tile=tile,
        action_id=action_id,
        source_seat=source_seat,
        consumed_tiles=tuple(consumed_tiles),
        offered_action_ids=tuple(offered_action_ids),
        accepted_action_ids=tuple(accepted_action_ids),
        round_index=round_index,
        scores=None if scores is None else tuple(scores),
        reason=reason,
        public_delta=tuple(public_delta),
    )
    state.builder.append_visible(envelope)
    state.event_count += 1
    state.last_kind = kind


def _snapshot_at_row(state: _GameState, *, phase: str, turn_actor: int, obs: _SimStep) -> None:
    wind = _BAKAZE_TO_WIND[state.bakaze]
    state.builder.update_public_state(
        decision_id=f"{state.sim_game_id}:{state.event_count}",
        round_index=max(0, state.hand_index),
        round_wind=wind,
        hand_number=state.hand_number,
        seat_winds=seat_winds_for_dealer(state.dealer),
        honba=state.honba,
        riichi_sticks=obs.sticks,
        dealer=state.dealer,
        scores=tuple(obs.scores),
        turn_actor=turn_actor,
        phase=cast("Any", phase),
        live_wall_tiles_remaining=max(0, _LIVE_WALL_BASE - state.draws),
        # Adapter parity (e80184a): the window opens on riichi_accepted and
        # clears per the interrupt rules; snapshots read live state.
        ippatsu_active=state.ippatsu,
    )


def _context_for(
    state: _GameState,
    seat: int,
    *,
    phase: str,
    offered: tuple[int | None, int | None],
    extra_concealed: Sequence[int] = (),
    obs: _SimStep,
) -> ActionContext:
    concealed = sorted(set(obs.hand) | set(extra_concealed))
    if obs.drawn is not None:
        concealed = sorted(set(concealed) | {obs.drawn})
    offered_tile, offered_by = offered
    return ActionContext(
        actor=make_seat(seat),
        action_table_hash=state.table.digest,
        phase=cast("Any", phase),
        offered_tile=None if offered_tile is None else make_tile_id(offered_tile),
        offered_by=None if offered_by is None else make_seat(offered_by),
        own_concealed_tiles=tuple(make_tile_id(t) for t in concealed),
        visible_melds=tuple(m for row in state.melds for m in row),
    )


def _legal_mjai_type(raw: Any) -> str:
    """MJAI type string of a yielded engine action (public mapping only)."""
    try:
        mjai: Any = raw.to_mjai()
    except Exception as exc:
        raise ContractError(f"engine action without an MJAI mapping: {exc}") from exc
    if not isinstance(mjai, dict) or not isinstance(mjai.get("type"), str):
        raise ContractError(f"engine action without a type string: {mjai!r}")
    return mjai["type"]


def _expand_nonclaim_legals(
    state: _GameState,
    seat: int,
    step: _SimStep,
    *,
    phase: str,
    offered: tuple[int | None, int | None],
    extra_concealed: Sequence[int] = (),
) -> tuple[tuple[CanonicalAction, ...], tuple[bool, ...], ActionContext]:
    """Expand one step's yielded legals through the adapter's legal view.

    Yielded claim legals use meld-style consume sets (called tile included),
    which are invalid as canonical consumed sets -- ``pon``/``daiminkan`` also
    collapse same-string copies -- so claims rebuild deterministically on the
    claim path (string-distinct chi variants rejoin the mask there); every
    other kind flows through untouched.
    """
    context = _context_for(
        state, seat, phase=phase, offered=offered, extra_concealed=extra_concealed, obs=step
    )
    actor_melds = list(state.melds[seat])
    kept = [
        raw
        for raw in step.raw_legals
        if _legal_mjai_type(raw) not in ("chi", "pon", "daiminkan", "kakan")
    ]
    # MjaiReplay renders taken ron tiles with distinct copies while discards
    # collapse per string; the tracked offer owns the canonical copy, so ron
    # slots are re-pointed at it when the strings agree (anything else stays
    # verbatim and fails closed at encode time).
    from types import SimpleNamespace  # local single-use adapter (see _capture_row)

    offered_tile, _ = offered
    fixed: list[Any] = []
    for raw in kept:
        tile_raw: Any = raw.tile
        try:
            is_ron = int(cast("Any", raw.action_type)) == int(riichienv.ActionType.RON)
        except (TypeError, ValueError):
            is_ron = False
        if (
            offered_tile is not None
            and tile_raw is not None
            and is_ron
            and mjai_string_of(int(tile_raw)) == mjai_string_of(offered_tile)
            and int(tile_raw) != offered_tile
        ):
            raw = SimpleNamespace(
                action_type=raw.action_type,
                tile=offered_tile,
                consume_tiles=tuple(raw.consume_tiles),
            )
        fixed.append(raw)
    actions, mask = legal_view(
        table=state.table,
        context=context,
        engine_actions=fixed,
        drawn_tile=step.drawn,
        own_hand=list(step.hand),
        melds_of_actor=actor_melds,
        offered_by=offered[1],
    )
    return actions, mask, context


def _legal_mjai_type(raw: Any) -> str:
    """MJAI type string of a yielded engine action (public mapping only)."""
    try:
        mjai: Any = raw.to_mjai()
    except Exception as exc:
        raise ContractError(f"engine action without an MJAI mapping: {exc}") from exc
    if isinstance(mjai, str):
        try:
            mjai = json.loads(mjai)
        except ValueError as exc:
            raise ContractError(f"engine action MJAI unparseable: {exc}") from exc
    if not isinstance(mjai, dict) or not isinstance(mjai.get("type"), str):
        raise ContractError(f"engine action without a type string: {mjai!r}")
    return mjai["type"]


# ---------------------------------------------------------------------------
# Deterministic claim variants.
# ---------------------------------------------------------------------------


def _distinct_copies(ids: tuple[int, ...]) -> tuple[int, ...]:
    """Expand copy-collapsed oracle ids to distinct physical copies.

    ``MjaiReplay`` renders every occurrence of one tile string with the same
    base physical id (four ``2p`` read as ``[40, 40, 40, 40]``), while the
    contract space tracks distinct copies (``[40, 41, 42, 43]``). Ordering
    copies per string preserves the exact string multiset, so string-level
    agreement is untouched and ownership sees tile-valid ids. Red fives keep
    their string (``5mr``/``5m`` pools stay disjoint). Overused strings keep
    the verbatim id and fail closed downstream.
    """
    counts: dict[str, int] = {}
    out: list[int] = []
    for tile in ids:
        pai = mjai_string_of(tile)
        pool = _copies_of_string(pai)
        seen = counts.get(pai, 0)
        out.append(pool[seen] if seen < len(pool) else tile)
        counts[pai] = seen + 1
    return tuple(out)


def _tracked_consumed(
    hand: Sequence[int],
    consumed_strings: Sequence[str],
    *,
    needed: int,
    called: int | None = None,
) -> tuple[int, ...]:
    """Resolve meld copies from the tracked hand state, never guessed.

    Each logged string consumes one tracked tile rendering that exact string
    (red-aware); absence fails closed instead of inventing pool copies the
    actor was never dealt. Collapsed oracle ids can repeat the called copy
    inside the tracked hand, so each such collision is swapped for an unused
    pool copy of the same string, keeping meld tiles distinct; exhaustion
    fails closed.
    """
    pool = list(hand)
    picked: list[int] = []
    for pai in sorted(s for s in consumed_strings):
        for index, candidate in enumerate(pool):
            if mjai_string_of(candidate) == pai:
                picked.append(pool.pop(index))
                break
        else:
            raise ContractError(f"no tracked copy left for meld tile {pai!r}")
    if len(picked) != needed:
        raise ContractError(f"claim needs {needed} consumed tiles, got {len(picked)}")
    if called is not None:
        for pos, tile in enumerate(picked):
            if tile == called:
                pai = mjai_string_of(tile)
                used = set(picked) | {called}
                for candidate in _copies_of_string(pai):
                    if candidate not in used:
                        picked[pos] = candidate
                        used.add(candidate)
                        break
                else:
                    raise ContractError(f"no distinct copy left for meld tile {pai!r}")
    return tuple(sorted(picked))


def _tracked_discard_tile(step: _SimStep, pai: str) -> int:
    """Resolve one logged discard to a tracked copy (drawn tile preferred).

    The drawn tile leads so tsumogiri resolves to the draw itself, otherwise
    the first tracked copy rendering the string. Absence fails closed.
    """
    if step.drawn is not None and mjai_string_of(step.drawn) == pai:
        return step.drawn
    for tile in step.hand:
        if mjai_string_of(tile) == pai:
            return tile
    raise ContractError(f"no tracked copy of discard {pai!r} in hand")


def _claim_canonical(
    *,
    kind: str,
    seat: int,
    called: int,
    consumed: tuple[int, ...],
    source: int,
) -> CanonicalAction:
    return CanonicalAction(
        kind=cast("Any", kind),
        actor=make_seat(seat),
        tile=None,
        called_tile=make_tile_id(called),
        consumed_tiles=tuple(make_tile_id(t) for t in consumed),
        source_seat=make_seat(source),
        declares_riichi=False,
        metadata=(),
    )


# ---------------------------------------------------------------------------
# Row capture.
# ---------------------------------------------------------------------------


def _concealed_for_build(step: _SimStep) -> list[int]:
    """Concealed hand for the builder: oracle hand minus the drawn tile once."""
    hand = list(step.hand)
    if step.drawn is not None:
        for index, tile in enumerate(hand):
            if tile == step.drawn:
                del hand[index]
                break
    return hand


def _capture_row(
    state: _GameState,
    kyoku: int,
    *,
    seat: int,
    step: _SimStep,
    canonical: CanonicalAction,
    chosen_id: int,
    mask: Sequence[bool],
    phase: str,
    turn_actor: int,
    can_tsumo: bool,
    can_riichi: bool,
    furiten: str,
) -> None:
    """Assemble one actor row through the canonical builder (capture-then-emit).

    The observation is captured BEFORE the row's envelopes are emitted, exactly
    like the engine path captures before applying.
    """
    seq = state.seq
    state.seq += 1
    decision_id = f"{state.game.game_id}:d{seq:04d}"
    if state.seat_filter is not None and seat != state.seat_filter:
        return
    _snapshot_at_row(state, phase=phase, turn_actor=turn_actor, obs=step)
    state.builder.set_concealed_hand(make_seat(seat), _concealed_for_build(step))
    state.builder.set_actor_state(
        make_seat(seat), furiten=furiten, can_tsumo=can_tsumo, can_riichi=can_riichi
    )
    try:
        observation = state.builder.build(actor=make_seat(seat), legal_mask=tuple(mask))
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation build failed: {exc}") from exc
    from hydra2.contracts.observation import VISIBILITY_VALIDATOR as _VV

    try:
        _VV.validate_observation(observation)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation rejected: {exc}") from exc
    doc = observation.to_json()
    # Perf-C P1a: hand the validated live object to the encoder out of band
    # (row dict content below is byte-identical either way).
    from hydra2.data import replay_expand as _re

    _re.stash_live_observation(decision_id, observation)
    try:
        verify_no_privileged_leakage(doc)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"actor row leaks privilege: {exc}") from exc
    if len(cast("list[object]", doc.get("visible_history", []))) > HISTORY_EVENT_CAP:
        raise state.fail(
            kyoku,
            f"seat {seat} row",
            f"visible history exceeds model cap {HISTORY_EVENT_CAP} (never truncated)",
        )
    obs_hash = str(observation.observation_hash)
    derivation = str(
        of_canonical(
            {
                "game_id": state.game.game_id,
                "decision_id": decision_id,
                "observation_hash": obs_hash,
                "chosen_action_id": chosen_id,
                "wall_digest": None,
                "adapter_hash": _adapter_hash(),
                "derivation": SIM_DERIVATION_MARK,
            }
        )
    )
    state.rows.append(
        DecisionRow(
            game_id=state.game.game_id,
            round_id=f"{state.game.game_id}:h{state.round_idx:02d}",
            decision_id=decision_id,
            seat=seat,
            source_object_id=state.game.object_id,
            split=state.split,
            rules_hash=str(observation.rules_hash),
            adapter_hash=_adapter_hash(),
            observation_hash=obs_hash,
            action_table_hash=str(observation.action_table_hash),
            derivation_hash=derivation,
            actor_observation=doc,
            chosen_action_id=chosen_id,
            privileged_label_ref=decision_id,
        )
    )


# ---------------------------------------------------------------------------
# Per-kyoku walk (queues + window stash).
# ---------------------------------------------------------------------------

_WINDOW_HEAD_TYPES = frozenset({"none", "chi", "pon", "daiminkan", "hora"})


@dataclass(slots=True)
class _KyokuWalk:
    queues: list[list[_SimStep]]
    stash: dict[int, _SimStep]
    tw: _WindowOracle
    drawer: int | None = None
    last_oracle_dora: tuple[int, ...] = ()
    dora_used: set[int] = None  # type: ignore[assignment]
    # Log-faithful Python tracker (single-pass counters, no engine):
    hands: list[list[int]] = None  # type: ignore[assignment]
    rivers: list[list[int]] = None  # type: ignore[assignment]
    missed: set[int] = None  # type: ignore[assignment]
    tehais: tuple[list[str], ...] = ()
    tsumo_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    first_draws: tuple[str | None, str | None, str | None, str | None] = (None, None, None, None)
    exp_drawer: int | None = None
    kan_pending: bool = False
    kakan_added: list[dict[int, int]] = None  # type: ignore[assignment]
    draw_queues: list[list[list[int]]] = None  # type: ignore[assignment]
    take_taken: dict[str, int] = None  # type: ignore[assignment]
    filler_ids: list[int] = None  # type: ignore[assignment]
    live_total: int = 0
    rinshan_total: int = 0
    live_used: int = 0
    rinshan_used: int = 0
    filler_live_n: int = 0
    filler_rin_n: int = 0
    sticks: int = 0


def _peek(walk: _KyokuWalk, seat: int) -> _SimStep | None:
    queue = walk.queues[seat]
    return queue[0] if len(queue) > 0 else None


def _note_pop(state: _GameState, walk: _KyokuWalk, step: _SimStep) -> None:
    """Record oracle facts from every consumed step (dora visibility)."""
    del state
    walk.last_oracle_dora = step.dora


def _pop(state: _GameState, walk: _KyokuWalk, kyoku: int, seat: int, *, why: str) -> _SimStep:
    queue = walk.queues[seat]
    if len(queue) == 0:
        raise state.fail(kyoku, f"seat {seat} pop", f"oracle queue empty ({why})")
    step = queue.pop(0)
    _note_pop(state, walk, step)
    return step


def _drop_orphans(state: _GameState, walk: _KyokuWalk, seat: int) -> None:
    """Discard queued window leftovers ahead of a seat's next live step.

    Placeholder ``none`` steps and claim steps from already-closed windows
    carry no rows for the logged game; the log stays authoritative (a logged
    claim without oracle support still fails closed at claim-match time). A
    passed ron is an orphan, but a post-reach ron is a live future win
    (forced discards yield no steps, so nothing precedes it). Draw decisions
    and queued tsumo wins are live steps and stop the scan.
    """
    while True:
        head = _peek(walk, seat)
        if head is None:
            return
        if head.mjai_type in ("none", "chi", "pon", "daiminkan"):
            dropped = walk.queues[seat].pop(0)
            _note_pop(state, walk, dropped)
            continue
        if head.mjai_type == "hora" and head.drawn is None and not state.riichi_declared[seat]:
            dropped = walk.queues[seat].pop(0)
            _note_pop(state, walk, dropped)
            continue
        return


def _pop_draw_head(
    state: _GameState, walk: _KyokuWalk, kyoku: int, seat: int, *, why: str
) -> _SimStep:
    """Pop one draw-decision step after discarding orphaned leftovers."""
    _drop_orphans(state, walk, seat)
    return _pop(state, walk, kyoku, seat, why=why)


def _pop_window_heads(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    discarder: int,
    tile: int,
    claim_ev: dict[str, object] | None,
) -> None:
    """Pop this window's responder steps into the stash.

    Pass steps pop unconditionally (placeholder noise is harmless once
    discarded). Claim and ron steps pop only for the logged claim of this
    window: the same tile can be offered (and passed) on an earlier window
    while the log claims a later one, so an unmatched claim must stay queued
    for its own discard instead of being eaten here and cleared as an orphan.
    """
    del kyoku
    pai = mjai_string_of(tile)
    claim_kind = str(claim_ev.get("type", "")) if isinstance(claim_ev, dict) else ""
    claim_actor = claim_ev.get("actor") if isinstance(claim_ev, dict) else None
    for seat in range(4):
        if seat == discarder:
            continue
        head = _peek(walk, seat)
        if head is None or head.mjai_type not in _WINDOW_HEAD_TYPES:
            continue
        if head.mjai_type in ("chi", "pon", "daiminkan", "hora") and (
            claim_kind not in ("chi", "pon", "daiminkan", "hora")
            or claim_actor != seat
            or head.mjai_type != claim_kind
        ):
            continue
        if head.mjai_type in ("chi", "pon", "daiminkan") and (
            head.tile is None or mjai_string_of(head.tile) != pai
        ):
            continue
        if head.mjai_type == "hora" and not (
            head.drawn is None
            and head.tile is not None
            and mjai_string_of(head.tile) == mjai_string_of(tile)
        ):
            # A future tsumo win is not part of this window; leave it queued.
            continue
        popped = walk.queues[seat].pop(0)
        _note_pop(state, walk, popped)
        walk.stash[seat] = popped


def _strict_row(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    seat: int,
    step: _SimStep,
    expected: CanonicalAction,
    *,
    phase: str,
    turn_actor: int,
    offered: tuple[int | None, int | None] = (None, None),
    extra_concealed: Sequence[int] = (),
    match_offered: bool = True,
) -> int:
    """Capture a row whose canonical form must sit in the expanded offer.

    Runs pre-step: the oracle cross-check and furiten read observe the
    throwaway state before this decision is stepped through it. Kakan passes
    ``match_offered=False``: the yielded kakan tile is string-collapsed while
    the true added copy is resolved separately, so membership is validated by
    encoding (ownership plus prior-pon metadata) instead.
    """
    try:
        walk.tw.check_row(seat, step, melds=state.melds, hand_check=True)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"oracle cross-check: {exc}") from exc
    try:
        furiten = walk.tw.furiten(seat)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"furiten query failed: {exc}") from exc
    try:
        actions, mask_t, context = _expand_nonclaim_legals(
            state, seat, step, phase=phase, offered=offered, extra_concealed=extra_concealed
        )
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"legal expansion failed: {exc}") from exc
    if match_offered and all(a != expected for a in actions):
        raise state.fail(
            kyoku, f"seat {seat} row", f"action {expected.kind}:{expected.tile} not offered"
        )
    try:
        chosen_id = int(canonical_action_codec.encode(expected, table=state.table, context=context))
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"action encode failed: {exc}") from exc
    if match_offered and not mask_t[chosen_id]:
        raise IllegalActionError(
            f"sim replay desync game {state.game.game_id!r} kyoku {kyoku} seat {seat} row: "
            f"chosen action id {chosen_id} is not legal"
        )
    mask: Sequence[bool] = mask_t
    if not match_offered:
        mutable = list(mask_t)
        mutable[chosen_id] = True
        mask = tuple(mutable)
    can_tsumo = any(a.kind == "tsumo" for a in actions)
    can_riichi = any(a.kind == "riichi_discard" for a in actions)
    _capture_row(
        state,
        kyoku,
        seat=seat,
        step=step,
        canonical=expected,
        chosen_id=chosen_id,
        mask=mask,
        phase=phase,
        turn_actor=turn_actor,
        can_tsumo=can_tsumo,
        can_riichi=can_riichi,
        furiten=furiten,
    )
    return chosen_id


# ---------------------------------------------------------------------------
# Event handlers (adapter envelope order mirrored).
# ---------------------------------------------------------------------------


def _split_kyoku_draws(
    state: _GameState, kyoku: int, events: Sequence[dict[str, object]], start: int
) -> tuple[list[str], list[str]]:
    """Partition a kyoku's tsumo pais into live draws and rinshan draws.

    A tsumo whose previous significant event (past transparent dora/reach
    markers) is a kan is the rinshan replacement; every other tsumo draws
    live. Scans to the kyoku boundary (end_kyoku / next hand / game end).
    """
    from hydra2.data import replay_expand as _re

    live: list[str] = []
    rinshan: list[str] = []
    prev_kind = "start_kyoku"
    idx = start + 1
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = str(event.get("type", ""))
        if kind in _re._END_TYPES or kind == "start_kyoku":
            break
        if kind == "end_kyoku":
            break
        if kind in _TRANSPARENT_KINDS:
            idx += 1
            continue
        if kind == "tsumo":
            actor = event.get("actor")
            pai = event.get("pai")
            if (
                isinstance(actor, bool)
                or not isinstance(actor, int)
                or not 0 <= actor <= 3
                or not isinstance(pai, str)
                or pai == ""
            ):
                raise state.fail(kyoku, "tsumo", "malformed draw event")
            (rinshan if prev_kind in ("ankan", "kakan", "daiminkan") else live).append(pai)
        prev_kind = kind
        idx += 1
    return live, rinshan


def _fresh_builder(state: _GameState) -> ObservationBuilder:
    """One observation builder per kyoku (walled-adapter parity).

    Histories live exactly one kyoku (model cap
    :data:`~hydra2.contracts.observation.HISTORY_EVENT_CAP`); rebirthing the
    builder per hand is what the walled adapter does via ``_open_hand``.
    """
    return ObservationBuilder(
        game_id=state.sim_game_id,
        rules_id=state.rules.rules_id,
        rules_hash=make_digest_text(state.rules_hash),
        action_table_hash=state.table.digest,
        expected_legal_mask_length=len(state.table.actions),
        event_schema_hash=make_digest_text(_event_schema_hash()),
        packet_boundary_hash=make_digest_text(_packet_boundary_hash()),
    )


def _do_start_kyoku(
    state: _GameState,
    drained: list[list[list[_SimStep]]],
    kyoku_ordinal: int,
    event: dict[str, object],
    events: Sequence[dict[str, object]],
    start_idx: int,
) -> _KyokuWalk:
    try:
        oya = int(cast("Any", event["oya"]))
        honba = int(cast("Any", event["honba"]))
        kyotaku = int(cast("Any", event["kyotaku"]))
        scores = tuple(int(s) for s in cast("Any", event["scores"]))
        bakaze = str(cast("Any", event["bakaze"]))
        kyoku_no = int(cast("Any", event["kyoku"]))
        tehais = tuple(cast("Any", event["tehais"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise state.fail(kyoku_ordinal, "start_kyoku", f"malformed hand header: {exc}") from exc
    if bakaze not in _BAKAZE_TO_WIND:
        raise state.fail(kyoku_ordinal, "start_kyoku", f"unknown bakaze {bakaze!r}")
    if len(scores) != 4:
        raise state.fail(kyoku_ordinal, "start_kyoku", "scores must cover 4 seats")
    state.hand_index += 1
    state.round_idx += 1
    state.dealer = oya
    state.bakaze = bakaze
    state.honba = honba
    state.kyotaku = kyotaku
    state.hand_number = kyoku_no
    state.tracked_scores = scores
    state.draws = 0
    state.riichi_declared = (False, False, False, False)
    state.ippatsu = (False, False, False, False)
    state.melds = ([], [], [], [])
    state.last_discard = (None, None)
    state.cr_pending = False
    state.decided = False
    state.decided_by = None
    # Per-kyoku observation scope (walled-adapter parity): rebirth the
    # builder before round_start lands in it. Kyoku 0 also carries the
    # once-per-game game_start first, exactly like the adapter's reset.
    state.builder = _fresh_builder(state)
    if kyoku_ordinal == 0:
        _emit(
            state,
            kind="game_start",
            visibility="public",
            round_index=0,
            scores=[state.rules.starting_points] * 4,
        )
    if not 0 <= kyoku_ordinal < len(drained):
        raise state.fail(kyoku_ordinal, "start_kyoku", "kyoku missing from replay")
    queues = drained[kyoku_ordinal]
    tw = _WindowOracle(game_id=state.game.game_id)
    live_draws, rinshan_draws = _split_kyoku_draws(state, kyoku_ordinal, events, start_idx)
    try:
        tw.reset_kyoku(
            ordinal=kyoku_ordinal,
            oya=oya,
            scores=state.tracked_scores,
            honba=honba,
            kyotaku=kyotaku,
            bakaze=bakaze,
            tehais=tehais,
            live_draws=live_draws,
            rinshan_draws=rinshan_draws,
        )
    except ContractError as exc:
        raise state.fail(kyoku_ordinal, "start_kyoku", f"oracle reset failed: {exc}") from exc
    walk = _KyokuWalk(queues=queues, stash={}, tw=tw)
    walk.dora_used = set()
    _emit(
        state,
        kind="round_start",
        visibility="public",
        actor=oya,
        round_index=state.hand_index,
        scores=list(scores),
        public_delta=(
            make_delta(("round_index",), "set", state.hand_index),
            make_delta(("honba",), "set", honba),
            make_delta(("riichi_sticks",), "set", kyotaku),
            make_delta(("scores",), "set", list(scores)),
        ),
    )
    wind = _BAKAZE_TO_WIND[bakaze]
    state.builder.update_public_state(round_wind=wind)
    return walk


# ---------------------------------------------------------------------------
# Windows, draws, declarations.
# ---------------------------------------------------------------------------


def _require_actor(state: _GameState, kyoku: int, event: dict[str, object], *, where: str) -> int:
    actor = event.get("actor")
    if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor <= 3:
        raise state.fail(kyoku, where, f"event actor must be 0..3, got {actor!r}")
    return actor


def _check_drawer(
    state: _GameState, walk: _KyokuWalk, kyoku: int, seat: int, *, where: str
) -> None:
    if walk.drawer is None:
        walk.drawer = seat
    elif walk.drawer != seat:
        raise state.fail(kyoku, where, f"actor {seat} != expected decision seat {walk.drawer}")


def _resolve_dora(state: _GameState, walk: _KyokuWalk, kyoku: int, marker: str) -> int:
    """Resolve a kan-dora marker to the exact physical indicator."""
    reused: int | None = None
    for tile in walk.last_oracle_dora:
        try:
            rendered = mjai_string_of(tile)
        except ContractError:
            continue
        if rendered != marker:
            continue
        if tile in walk.dora_used:
            if reused is None:
                reused = tile
            continue
        walk.dora_used.add(tile)
        return tile
    if reused is not None:
        # Real Tenhou re-emits one marker string for successive kan-dora
        # reveals (audit: ordered markers faithful to the XML reveal order),
        # while the filler-built dead wall carries a single fresh copy of
        # that string. Resolve the repeat against the used indicator: the
        # emitted strings stay faithful and copy identity is folded by
        # design (wall-less derivation marker, never a wall digest). Only a
        # marker naming no indicator at all stays fail-closed below.
        return reused
    raise state.fail(kyoku, "dora", f"dora marker {marker!r} matches no fresh indicator")


def _emit_call_resolved(state: _GameState, *, accepted: Sequence[int]) -> None:
    """Emit the filtered window-resolution envelope (content-free by design).

    ``call_resolved`` is ``server_private``: filtered from every actor history
    by construction, so only its existence and position matter (decision-id
    counting). The grammar requires ``accepted ⊆ offered``, hence the offered
    list repeats the accepted singleton (all-pass resolutions carry neither).
    """
    accepted_ids = tuple(accepted)
    _emit(
        state,
        kind="call_resolved",
        visibility="server_private",
        offered_action_ids=accepted_ids,
        accepted_action_ids=accepted_ids,
    )
    state.cr_pending = False


def _open_window(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    discarder: int,
    tile: int,
    claim_ev: dict[str, object] | None,
) -> None:
    """Pop responder heads and resolve the window through the oracle engine.

    Pass steps are placeholder noise (the countdown wall leaks stale legals
    through them), so they are popped and discarded unconditionally; the
    engine's own claim-window predicate decides ``call_window`` /
    ``call_resolved`` existence. A stash holding a claim the log never shows
    fails closed at window close.
    """
    _pop_window_heads(state, walk, kyoku, discarder, tile, claim_ev)
    try:
        window_open = walk.tw.window_open(discarder)
    except ContractError as exc:
        raise state.fail(kyoku, "window", f"window query failed: {exc}") from exc
    if window_open:
        try:
            walk.tw.resolve_window(discarder, claim_ev)
        except ContractError as exc:
            raise state.fail(kyoku, "window", f"window resolution failed: {exc}") from exc
    elif claim_ev is not None:
        raise state.fail(kyoku, "window", "logged claim without an open window")
    if window_open and state.opened_by_discard and state.last_kind == "discard":
        _emit(state, kind="call_window", visibility="public")
    state.cr_pending = window_open and state.opened_by_discard


def _clear_stash_no_claim(state: _GameState, walk: _KyokuWalk, kyoku: int, *, where: str) -> None:
    """Close a window nobody claimed in the log.

    Stashed passes and unlogged claim/ron steps are discarded: the log is the
    authoritative decision record, and rows follow logged decisions only. A
    logged claim without oracle support still fails closed at claim-match
    time. (In particular, a passed ron leaves its step here; the missed
    furiten it implies is tracked by the oracle engine, not this stash.)
    """
    del kyoku, where
    walk.stash.clear()
    if state.cr_pending:
        _emit_call_resolved(state, accepted=())


# ---------------------------------------------------------------------------
# Decision handlers.
# ---------------------------------------------------------------------------


def _note_forced_draw(
    state: _GameState, walk: _KyokuWalk, kyoku: int, actor: int, pai: str
) -> None:
    """Validate a post-reach forced draw against the wall and emit its envelopes."""
    try:
        walk.tw.note_tsumo(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "tsumo", f"oracle draw mismatch: {exc}") from exc
    if state.last_discard[0] == actor:
        state.last_discard = (None, None)
    state.draws += 1
    walk.drawer = actor
    _emit(state, kind="turn_advance", visibility="public", actor=actor)
    _emit(
        state,
        kind="draw_tile",
        visibility="actor_private",
        actor=actor,
        tile=int(physical_of(pai)),
    )


def _do_tsumo(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    event: dict[str, object],
    events: Sequence[object],
    idx: int,
) -> None:
    actor = _require_actor(state, kyoku, event, where="tsumo")
    if state.decided:
        raise state.fail(kyoku, "tsumo", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="tsumo")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, "tsumo", "draw without a pai string")
    _drop_orphans(state, walk, actor)
    head = _peek(walk, actor)
    if head is None:
        # An empty queue ends the kyoku early (kyushu-style abortive
        # declaration) or continues a post-reach forced run; anything else
        # is a desync. Transparent markers carry no decisions.
        nxt = idx + 1
        total = len(events)
        while nxt < total:
            following = events[nxt]
            if not isinstance(following, dict):
                break
            if str(following.get("type", "")) in _TRANSPARENT_KINDS:
                nxt += 1
                continue
            break
        following = events[nxt] if nxt < total else None
        if isinstance(following, dict) and following.get("type") == "ryukyoku":
            _note_forced_draw(state, walk, kyoku, actor, pai)
            return
        if (
            state.riichi_declared[actor]
            and isinstance(following, dict)
            and following.get("type") == "dahai"
            and following.get("actor") == actor
        ):
            _note_forced_draw(state, walk, kyoku, actor, pai)
            return
        raise state.fail(kyoku, "tsumo", f"seat {actor} oracle queue empty")
    drawn = head.drawn
    if drawn is None or mjai_string_of(drawn) != pai:
        # A queued head from a later decision (or none at all) means every
        # intervening turn was a post-reach forced discard the oracle omits;
        # only a declared reach sanctions that shape.
        if not state.riichi_declared[actor]:
            raise state.fail(kyoku, "tsumo", f"seat {actor} draw differs from oracle draw")
        _note_forced_draw(state, walk, kyoku, actor, pai)
        return
    try:
        walk.tw.note_tsumo(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "tsumo", f"oracle draw mismatch: {exc}") from exc
    if state.last_discard[0] == actor:
        state.last_discard = (None, None)
    state.draws += 1
    walk.drawer = actor
    _emit(state, kind="turn_advance", visibility="public", actor=actor)
    _emit(
        state,
        kind="draw_tile",
        visibility="actor_private",
        actor=actor,
        tile=drawn,
    )


def _do_forced_dahai(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    actor: int,
    pai: str,
    pos: _TablePosition,
    claim_ev: dict[str, object] | None,
) -> None:
    """Replay a post-reach forced discard the oracle yields no step for.

    The logged tile must be the pending drawn tile (tsumogiri by rule, no
    matter what the writer flag claims); the observation is sourced from the
    throwaway engine, whose wall-validated state mirrors the log, and the
    legal mask translates its live legals. The oracle queues are never
    touched here: a queued head belongs to a later decision.
    """
    if pos.drawn is None or mjai_string_of(pos.drawn) != pai:
        raise state.fail(kyoku, "dahai", f"seat {actor} forced discard is not the drawn tile")
    step = _SimStep(
        seat=actor,
        mjai_type="dahai",
        tile=pos.drawn,
        consume=(),
        mjai={"type": "dahai", "actor": actor, "pai": pai},
        raw_action=None,
        raw_legals=pos.legals,
        hand=pos.hand,
        drawn=pos.drawn,
        dora=pos.dora,
        discards=pos.rivers,
        hands_lens=pos.lens,
        scores=tuple(state.tracked_scores),
        sticks=pos.sticks,
        oya=state.dealer,
        honba=state.honba,
        round_wind=_BAKAZE_TO_WIND[state.bakaze],
        player_id=actor,
        riichi=pos.declared,
    )
    expected = CanonicalAction(
        kind=cast("Any", "tsumogiri"),
        actor=make_seat(actor),
        tile=make_tile_id(pos.drawn),
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
        actor,
        step,
        expected,
        phase="draw_decision",
        turn_actor=actor,
        offered=live_offered,
    )
    try:
        walk.tw.do_dahai(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "dahai", f"oracle discard failed: {exc}") from exc
    _emit(
        state,
        kind="discard",
        visibility="public",
        actor=actor,
        tile=pos.drawn,
        action_id=chosen_id,
    )
    state.last_discard = (actor, pos.drawn)
    state.opened_by_discard = True
    walk.drawer = actor
    _open_window(state, walk, kyoku, actor, pos.drawn, claim_ev)


def _do_dahai(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    event: dict[str, object],
    claim_ev: dict[str, object] | None,
) -> None:
    actor = _require_actor(state, kyoku, event, where="dahai")
    if state.decided:
        raise state.fail(kyoku, "dahai", "discard after the kyoku was decided")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, "dahai", "discard without a pai string")
    tsumogiri = bool(event.get("tsumogiri", False))
    _check_drawer(state, walk, kyoku, actor, where="dahai")
    _drop_orphans(state, walk, actor)
    head = _peek(walk, actor)
    try:
        pos = walk.tw.position(actor)
    except ContractError as exc:
        raise state.fail(kyoku, "dahai", str(exc)) from exc
    if (
        head is not None
        and head.mjai_type == "dahai"
        and head.tile is not None
        and mjai_string_of(head.tile) == pai
        and head.drawn is not None
        and pos.drawn is not None
        and mjai_string_of(head.drawn) == mjai_string_of(pos.drawn)
    ):
        step = _pop(state, walk, kyoku, actor, why="dahai")
    elif (
        pos.drawn is not None
        and mjai_string_of(pos.drawn) == pai
        and state.riichi_declared[actor]
        and (head is None or head.mjai_type in ("hora", "ankan", "kakan"))
    ):
        _do_forced_dahai(state, walk, kyoku, actor, pai, pos, claim_ev)
        _ippatsu_interrupt(state, actor)  # declarer discarded again: window gone
        return
    else:
        step = _pop(state, walk, kyoku, actor, why="dahai")
    if step.mjai_type != "dahai":
        raise state.fail(kyoku, "dahai", f"seat {actor} oracle holds {step.mjai_type}")
    if step.tile is None or mjai_string_of(step.tile) != pai:
        raise state.fail(kyoku, "dahai", f"seat {actor} oracle discards a different tile")
    kind = "tsumogiri" if tsumogiri else "discard"
    expected = CanonicalAction(
        kind=cast("Any", kind),
        actor=make_seat(actor),
        tile=make_tile_id(step.tile),
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(),
    )
    # Draw-row contexts carry the live (possibly stale) offer, exactly like the
    # adapter's pre-apply context; FIX1 already cleared the self-offer case.
    live_offered: tuple[int | None, int | None] = (
        state.last_discard[1],
        state.last_discard[0],
    )
    chosen_id = _strict_row(
        state,
        walk,
        kyoku,
        actor,
        step,
        expected,
        phase="draw_decision",
        turn_actor=actor,
        offered=live_offered,
    )
    try:
        walk.tw.do_dahai(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "dahai", f"oracle discard failed: {exc}") from exc
    _emit(
        state,
        kind="discard",
        visibility="public",
        actor=actor,
        tile=step.tile,
        action_id=chosen_id,
    )
    state.last_discard = (actor, step.tile)
    state.opened_by_discard = True
    walk.drawer = actor
    _open_window(state, walk, kyoku, actor, step.tile, claim_ev)
    _ippatsu_interrupt(state, actor)  # declarer discarded again: window gone


# ---------------------------------------------------------------------------
# Reach, claims, kans.
# ---------------------------------------------------------------------------


def _do_reach(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    event: dict[str, object],
    declaration: dict[str, object],
    claim_ev: dict[str, object] | None,
) -> None:
    actor = _require_actor(state, kyoku, event, where="reach")
    if state.decided:
        raise state.fail(kyoku, "reach", "declaration after the kyoku was decided")
    dactor = _require_actor(state, kyoku, declaration, where="reach-declaration")
    if dactor != actor:
        raise state.fail(kyoku, "reach", "declaration actor differs from reach actor")
    pai = declaration.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, "reach", "declaration without a pai string")
    _check_drawer(state, walk, kyoku, actor, where="reach")
    reach_step = _pop_draw_head(state, walk, kyoku, actor, why="reach")
    if reach_step.mjai_type != "reach":
        raise state.fail(kyoku, "reach", f"seat {actor} oracle holds {reach_step.mjai_type}")
    # Declaration-discard dialect: house logs yield the declaration as a second
    # oracle step sharing the reach draw; Tenhou-real logs collapse it (every
    # post-reach discard is forced). Consume the second step only when it names
    # this exact draw and discard; otherwise resolve the logged discard from
    # the tracked reach hand.
    decl_head = _peek(walk, actor)
    if (
        decl_head is not None
        and decl_head.mjai_type == "dahai"
        and decl_head.drawn is not None
        and reach_step.drawn is not None
        and decl_head.drawn == reach_step.drawn
        and decl_head.tile is not None
        and mjai_string_of(decl_head.tile) == pai
    ):
        _ = _pop(state, walk, kyoku, actor, why="reach-declaration")  # consume decl; head held
        declaration_tile = decl_head.tile
    else:
        try:
            declaration_tile = _tracked_discard_tile(reach_step, pai)
        except ContractError as exc:
            raise state.fail(kyoku, "reach", f"declaration discard not owned: {exc}") from exc
    expected = CanonicalAction(
        kind=cast("Any", "riichi_discard"),
        actor=make_seat(actor),
        tile=make_tile_id(declaration_tile),
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=True,
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
        actor,
        reach_step,
        expected,
        phase="draw_decision",
        turn_actor=actor,
        offered=live_offered,
    )
    try:
        walk.tw.do_reach(actor)
    except ContractError as exc:
        raise state.fail(kyoku, "reach", f"oracle reach failed: {exc}") from exc
    try:
        walk.tw.do_dahai(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "reach", f"oracle declaration failed: {exc}") from exc
    # Replay parity: the adapter translates the declaration discard through
    # _on_dahai (a discard envelope stamped with the riichi id), never
    # _on_reach (the RIICHI and DISCARD steps land in separate batches, so the
    # paired reach branch never fires). No riichi_declared envelope, no
    # declared state; the declarer's ippatsu clears like any discard.
    _emit(
        state,
        kind="discard",
        visibility="public",
        actor=actor,
        tile=declaration_tile,
        action_id=chosen_id,
    )
    state.last_discard = (actor, declaration_tile)
    state.opened_by_discard = True
    state.declarations += 1
    reached = list(state.riichi_declared)
    reached[actor] = True
    state.riichi_declared = cast("tuple[bool, bool, bool, bool]", tuple(reached))
    scores = list(state.tracked_scores)
    scores[actor] -= 1000
    state.tracked_scores = cast("tuple[int, int, int, int]", tuple(scores))
    walk.drawer = actor
    _open_window(state, walk, kyoku, actor, declaration_tile, claim_ev)


def _match_stashed_claim(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    seat: int,
    kind: str,
    event: dict[str, object],
    discarder: int,
) -> _SimStep:
    """Validate the claimant's stashed window step against the logged claim."""
    step = walk.stash.pop(seat, None)
    if step is None:
        raise state.fail(kyoku, kind, f"seat {seat} holds no window step for the claim")
    if step.mjai_type != kind:
        raise state.fail(kyoku, kind, f"seat {seat} oracle holds {step.mjai_type}, log says {kind}")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, kind, "logged claim without a pai string")
    if step.tile is None or mjai_string_of(step.tile) != pai:
        raise state.fail(kyoku, kind, "claim names a different tile than the oracle")
    consumed = event.get("consumed")
    if not isinstance(consumed, (list, tuple)):
        raise state.fail(kyoku, kind, "logged claim without consumed tiles")
    target = event.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
        raise state.fail(kyoku, kind, "logged claim without a valid target seat")
    if target != discarder:
        raise state.fail(kyoku, kind, f"claim target {target} != discarder {discarder}")
    yielded_strings = sorted(mjai_string_of(t) for t in step.consume)
    logged_strings = sorted(str(t) for t in consumed)
    # Yielded consume repeats the called tile (degenerate); the log lists the
    # called tile once plus the hand tiles.
    if yielded_strings != sorted([pai, *logged_strings]):
        raise state.fail(kyoku, kind, "claim tiles differ from the oracle")
    for other, other_step in list(walk.stash.items()):
        if other_step.mjai_type in ("chi", "pon", "daiminkan"):
            del walk.stash[other]
    return step


def _do_claim(
    state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object], kind: str
) -> None:
    actor = _require_actor(state, kyoku, event, where=kind)
    if state.decided:
        raise state.fail(kyoku, kind, "claim after the kyoku was decided")
    if state.last_discard[0] is None or state.last_discard[1] is None:
        raise state.fail(kyoku, kind, "claim without a live discard offer")
    discarder = state.last_discard[0]
    called = state.last_discard[1]
    step = _match_stashed_claim(state, walk, kyoku, actor, kind, event, discarder)
    consumed_raw = cast("Any", event["consumed"])
    needed = 2 if kind in ("chi", "pon") else 3
    try:
        consumed = _tracked_consumed(
            step.hand, [str(t) for t in consumed_raw], needed=needed, called=called
        )
    except ContractError as exc:
        raise state.fail(kyoku, kind, f"claim tiles not owned: {exc}") from exc
    canonical = _claim_canonical(
        kind=kind, seat=actor, called=called, consumed=consumed, source=discarder
    )
    offered: tuple[int | None, int | None] = (called, discarder)
    try:
        actions, mask_t, context = _expand_nonclaim_legals(
            state,
            actor,
            step,
            phase="discard_response",
            offered=offered,
            extra_concealed=consumed,
        )
    except ContractError as exc:
        raise state.fail(kyoku, kind, f"legal expansion failed: {exc}") from exc
    try:
        chosen_id = int(
            canonical_action_codec.encode(canonical, table=state.table, context=context)
        )
    except ContractError as exc:
        raise state.fail(kyoku, kind, f"claim encode failed: {exc}") from exc
    state.melds[actor].append(
        VisibleMeld(
            meld_id=None,
            kind=cast("Any", kind),
            owner=make_seat(actor),
            source_seat=make_seat(discarder),
            called_tile=make_tile_id(called),
            tiles=tuple(make_tile_id(t) for t in sorted([*consumed, called])),
        )
    )
    try:
        walk.tw.check_row(actor, step, melds=state.melds, hand_check=False)
    except ContractError as exc:
        raise state.fail(kyoku, kind, f"oracle cross-check failed: {exc}") from exc
    try:
        furiten = walk.tw.furiten(actor)
    except ContractError as exc:
        raise state.fail(kyoku, kind, f"furiten query failed: {exc}") from exc
    mask = list(mask_t)
    mask[chosen_id] = True
    # String-distinct chi variants rejoin the mask exactly (their consumed
    # sets are already true physicals); pon/daiminkan copy-variants stay
    # folded by the copy-identity rule.
    for raw in step.raw_legals:
        if _legal_mjai_type(raw) != "chi":
            continue
        tile_raw: Any = raw.tile
        if tile_raw is None or int(tile_raw) != called:
            continue
        variant_consumed = sorted({int(t) for t in raw.consume_tiles} - {called})
        if len(variant_consumed) != 2:
            continue
        try:
            variant = _claim_canonical(
                kind="chi",
                seat=actor,
                called=called,
                consumed=(variant_consumed[0], variant_consumed[1]),
                source=discarder,
            )
            variant_id = int(
                canonical_action_codec.encode(variant, table=state.table, context=context)
            )
            mask[variant_id] = True
        except ContractError:
            continue
    can_tsumo = any(a.kind == "tsumo" for a in actions)
    can_riichi = any(a.kind == "riichi_discard" for a in actions)
    _capture_row(
        state,
        kyoku,
        seat=actor,
        step=step,
        canonical=canonical,
        chosen_id=chosen_id,
        mask=tuple(mask),
        phase="discard_response",
        turn_actor=discarder,
        can_tsumo=can_tsumo,
        can_riichi=can_riichi,
        furiten=furiten,
    )
    source = discarder
    if state.cr_pending:
        _emit_call_resolved(state, accepted=[chosen_id])
    _emit(
        state,
        kind=kind,
        visibility="public",
        actor=actor,
        tile=called,
        action_id=chosen_id,
        source_seat=source,
        consumed_tiles=consumed,
        public_delta=(
            make_delta(
                ("melds", actor),
                "append",
                meld_delta_value(
                    kind=kind,
                    owner=actor,
                    source_seat=source,
                    called_tile=called,
                    tiles=[*consumed, called],
                ),
            ),
            *((make_delta(("kan_count",), "increment", 1),) if kind == "daiminkan" else ()),
        ),
    )
    state.last_discard = (None, None)
    walk.drawer = actor
    walk.stash.clear()
    walk.exp_drawer = actor
    walk.kan_pending = kind == "daiminkan"
    _ippatsu_interrupt(state, None)  # any call interrupts every chance


# ---------------------------------------------------------------------------
# Kans, dora, reach acceptance.
# ---------------------------------------------------------------------------


def _do_ankan(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    actor = _require_actor(state, kyoku, event, where="ankan")
    if state.decided:
        raise state.fail(kyoku, "ankan", "kan after the kyoku was decided")
    consumed = event.get("consumed")
    if not isinstance(consumed, (list, tuple)):
        raise state.fail(kyoku, "ankan", "logged ankan without consumed tiles")
    _check_drawer(state, walk, kyoku, actor, where="ankan")
    step = _pop_draw_head(state, walk, kyoku, actor, why="ankan")
    if step.mjai_type != "ankan":
        raise state.fail(kyoku, "ankan", f"seat {actor} oracle holds {step.mjai_type}")
    yielded_strings = sorted(mjai_string_of(t) for t in step.consume)
    if yielded_strings != sorted(str(t) for t in consumed):
        raise state.fail(kyoku, "ankan", "ankan tiles differ from the oracle")
    try:
        block = _tracked_consumed(step.hand, [str(t) for t in consumed], needed=4)
    except ContractError as exc:
        raise state.fail(kyoku, "ankan", f"ankan tiles not owned: {exc}") from exc
    base = (block[0] // 4) * 4
    if tuple(block) != (base, base + 1, base + 2, base + 3):
        raise state.fail(kyoku, "ankan", f"ankan tiles {block!r} are not one block")
    expected = CanonicalAction(
        kind=cast("Any", "ankan"),
        actor=make_seat(actor),
        tile=None,
        called_tile=None,
        consumed_tiles=tuple(make_tile_id(t) for t in block),
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
        actor,
        step,
        expected,
        phase="draw_decision",
        turn_actor=actor,
        offered=live_offered,
        extra_concealed=block,
    )
    try:
        walk.tw.do_ankan(actor, [str(t) for t in consumed])
    except ContractError as exc:
        raise state.fail(kyoku, "ankan", f"oracle kan failed: {exc}") from exc
    _emit(
        state,
        kind="ankan",
        visibility="public",
        actor=actor,
        action_id=chosen_id,
        consumed_tiles=block,
        public_delta=(
            make_delta(
                ("melds", actor),
                "append",
                meld_delta_value(
                    kind="ankan",
                    owner=actor,
                    source_seat=None,
                    called_tile=None,
                    tiles=list(block),
                ),
            ),
            make_delta(("kan_count",), "increment", 1),
        ),
    )
    state.melds[actor].append(
        VisibleMeld(
            meld_id=None,
            kind=cast("Any", "ankan"),
            owner=make_seat(actor),
            tiles=tuple(make_tile_id(t) for t in block),
        )
    )
    walk.exp_drawer = actor
    walk.kan_pending = True
    walk.drawer = actor
    _ippatsu_interrupt(state, None)  # kan interrupts every chance


def _find_prior_pon(state: _GameState, seat: int, added: int) -> VisibleMeld:
    added_type = added // 4
    for meld in state.melds[seat]:
        if meld.kind == "pon" and (int(meld.tiles[0]) // 4) == added_type:
            return meld
    raise ContractError(f"kakan of tile {added}: no prior pon owned by seat {seat}")


def _do_kakan(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    event: dict[str, object],
    claim_ev: dict[str, object] | None,
) -> None:
    actor = _require_actor(state, kyoku, event, where="kakan")
    if state.decided:
        raise state.fail(kyoku, "kakan", "kan after the kyoku was decided")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, "kakan", "logged kakan without a pai string")
    _check_drawer(state, walk, kyoku, actor, where="kakan")
    step = _pop_draw_head(state, walk, kyoku, actor, why="kakan")
    if step.mjai_type != "kakan":
        raise state.fail(kyoku, "kakan", f"seat {actor} oracle holds {step.mjai_type}")
    if step.tile is None or mjai_string_of(step.tile) != pai:
        raise state.fail(kyoku, "kakan", "kakan names a different tile than the oracle")
    # The added fourth copy may be the drawn tile or a tile held since the
    # deal; either way the collapsed id equals the logged string.
    try:
        prior = _find_prior_pon(state, actor, step.tile)
    except ContractError as exc:
        raise state.fail(kyoku, "kakan", str(exc)) from exc
    # The yielded tile id is string-collapsed; the true added copy is the one
    # pool copy of this type absent from the prior pon triple (deterministic,
    # contract-valid, and exact on min-rule-disciplined logs).
    pool = _copies_of_string(pai)
    if pai[0] == "5" and len(pai) == 2:
        base = (int(physical_of(pai)) // 4) * 4
        pool = [base, base + 1, base + 2, base + 3]
    missing = [c for c in pool if c not in {int(t) for t in prior.tiles}]
    if len(missing) != 1:
        raise state.fail(kyoku, "kakan", "prior pon leaves no single added copy")
    added = missing[0]
    expected = CanonicalAction(
        kind=cast("Any", "kakan"),
        actor=make_seat(actor),
        tile=make_tile_id(added),
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(("prior_pon_meld_id", visible_meld_id(prior)),),
    )
    live_offered: tuple[int | None, int | None] = (
        state.last_discard[1],
        state.last_discard[0],
    )
    chosen_id = _strict_row(
        state,
        walk,
        kyoku,
        actor,
        step,
        expected,
        phase="draw_decision",
        turn_actor=actor,
        offered=live_offered,
        extra_concealed=[added],
        match_offered=False,
    )
    try:
        walk.tw.do_kakan(actor, pai)
    except ContractError as exc:
        raise state.fail(kyoku, "kakan", f"oracle kan failed: {exc}") from exc
    # The tracked pon triple IS the consumed set: the added fourth copy shares
    # the collapsed id, so value-filtering would eat a pon tile.
    consumed = tuple(prior.tiles)
    _emit(
        state,
        kind="kakan",
        visibility="public",
        actor=actor,
        tile=added,
        action_id=chosen_id,
        public_delta=(
            make_delta(
                ("melds", actor),
                "append",
                meld_delta_value(
                    kind="kakan",
                    owner=actor,
                    source_seat=None,
                    called_tile=None,
                    tiles=[*consumed, added],
                ),
            ),
            make_delta(("kan_count",), "increment", 1),
        ),
    )
    state.opened_by_discard = False
    state.last_discard = (actor, added)
    walk.drawer = actor
    walk.exp_drawer = actor
    walk.kan_pending = True
    # Kan windows never emit envelopes (adapter grammar routes kakan -> ron
    # directly), but the oracle still resolves responders to advance.
    _open_window(state, walk, kyoku, actor, added, claim_ev)
    _ippatsu_interrupt(state, None)  # kakan interrupts every chance


def _do_dora(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    marker = event.get("dora_marker")
    if not isinstance(marker, str) or marker == "":
        # Tenhou emits a bare {"type": "dora"} with no marker on kan-dora; the
        # indicator is unrecoverable from the log, so fail closed here rather
        # than resolve or guess it. Rare in real corpus (single-digit games
        # per hundred).
        raise state.fail(
            kyoku, "dora", "kan-dora indicator unrecoverable: dora event without a dora_marker"
        )
    tile = _resolve_dora(state, walk, kyoku, marker)
    _emit(
        state,
        kind="dora_revealed",
        visibility="public",
        tile=tile,
        public_delta=(make_delta(("dora_indicators",), "append", tile),),
    )


def _do_reach_accepted(
    state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]
) -> None:
    actor = _require_actor(state, kyoku, event, where="reach_accepted")
    walk.sticks += 1
    _emit(
        state,
        kind="riichi_accepted",
        visibility="public",
        actor=actor,
        public_delta=(
            make_delta(("riichi_states", actor), "set", "accepted"),
            make_delta(("riichi_sticks",), "increment", 1),
            make_delta(("ippatsu", actor), "set", True),
        ),
    )
    _ippatsu_open(state, actor)  # accepted reach opens the window


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
        step = _pop_draw_head(state, walk, kyoku, winner, why="hora")
        if step.mjai_type != "hora" or step.drawn is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        tile = step.drawn
        if step.tile is not None and step.tile != tile:
            raise state.fail(kyoku, "hora", "tsumo tile differs from the drawn tile")
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
            walk.tw.do_tsumo_win(winner, mjai_string_of(tile))
        except ContractError as exc:
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
        if step.mjai_type != "hora":
            raise state.fail(kyoku, "hora", f"seat {winner} oracle holds {step.mjai_type}")
        tile = state.last_discard[1]
        if step.tile is not None and mjai_string_of(step.tile) != mjai_string_of(tile):
            raise state.fail(kyoku, "hora", "ron tile differs from the offered discard")
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
        for other, other_step in list(walk.stash.items()):
            if other_step.mjai_type == "hora":
                del walk.stash[other]
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
    walk.exp_drawer = None
    walk.kan_pending = False


def _do_ryukyoku(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    if state.decided:
        raise state.fail(kyoku, "ryukyoku", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="ryukyoku")
    reason = event.get("reason")
    if not isinstance(reason, str) or reason == "":
        # Tenhou-real MJAI omits the reason for wall exhaustion and first-turn
        # kyushu aborts; every other abort carries one. Exhaustion always needs
        # a full wall of draws; kyushu is verified against the oracle engine's
        # own nine-terminals offer. Anything between stays quarantined.
        if state.draws >= 60:
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
    walk.exp_drawer = None
    walk.kan_pending = False


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


# ---------------------------------------------------------------------------
# Game walk + entry point.
# ---------------------------------------------------------------------------


def _walk_game(state: _GameState, drained: list[list[list[_SimStep]]]) -> list[DecisionRow]:
    events = list(state.game.events)
    walk: _KyokuWalk | None = None
    kyoku_ordinal = -1
    idx = 0
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        ev = event
        kind_raw = ev.get("type")
        if not isinstance(kind_raw, str) or kind_raw == "":
            raise ContractError(f"mjai event [{idx}] without a string type")
        kind = kind_raw
        if kind in _END:
            if kind == "end_game" or kind in ("endGame", "game_end", "end"):
                if walk is None or kyoku_ordinal < 0:
                    raise state.fail(-1, "end_game", "game ends before any kyoku")
                _do_end_game(state, walk, kyoku_ordinal)
            break
        if kind == "start_kyoku":
            kyoku_ordinal += 1
            walk = _do_start_kyoku(state, drained, kyoku_ordinal, ev, events, idx)
            idx += 1
            continue
        if kind in _START:
            idx += 1
            continue
        if walk is None or kyoku_ordinal < 0:
            raise state.fail(-1, kind, "decision before the first start_kyoku")
        assert walk is not None
        if kind == "ryukyoku":
            _do_ryukyoku(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind == "end_kyoku":
            following_ev: dict[str, object] | None = None
            if idx + 1 < total:
                candidate = events[idx + 1]
                if isinstance(candidate, dict):
                    following_ev = candidate
            _do_end_kyoku(state, walk, kyoku_ordinal, following_ev)
            idx += 1
            continue
        if kind == "tsumo":
            _do_tsumo(state, walk, kyoku_ordinal, ev, events, idx)
            idx += 1
            continue
        if kind == "dora":
            _do_dora(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind == "reach_accepted":
            _do_reach_accepted(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind in _SKIP:
            idx += 1
            continue
        if kind == "reach":
            nxt = idx + 1
            while nxt < total:
                following = events[nxt]
                if not isinstance(following, dict):
                    raise ContractError(f"mjai event [{nxt}] must be an object")
                following_kind = following.get("type")
                if following_kind in _SKIP:
                    nxt += 1
                    continue
                break
            else:
                raise state.fail(kyoku_ordinal, "reach", "declaration missing")
            declaration = events[nxt]
            if declaration.get("type") != "dahai":
                raise state.fail(kyoku_ordinal, "reach", "declaration is not a dahai")
            decl_actor = _require_actor(state, kyoku_ordinal, declaration, where="dahai")
            claim_after = _peek_window_claim(
                cast("Sequence[dict[str, object]]", events), nxt + 1, decl_actor
            )
            _do_reach(state, walk, kyoku_ordinal, ev, declaration, claim_after)
            idx = nxt + 1
            continue
        if kind in _ROW:
            if kind in _CLAIM:
                _do_claim(state, walk, kyoku_ordinal, ev, kind)
            elif kind in ("ankan", "kakan"):
                if kind == "ankan":
                    _do_ankan(state, walk, kyoku_ordinal, ev)
                else:
                    actor = _require_actor(state, kyoku_ordinal, ev, where="kakan")
                    claim_after = _peek_window_claim(
                        cast("Sequence[dict[str, object]]", events), idx + 1, actor
                    )
                    _do_kakan(state, walk, kyoku_ordinal, ev, claim_after)
            elif kind == "hora":
                _do_hora(state, walk, kyoku_ordinal, ev)
            else:  # dahai
                actor = _require_actor(state, kyoku_ordinal, ev, where="dahai")
                claim_after = _peek_window_claim(
                    cast("Sequence[dict[str, object]]", events), idx + 1, actor
                )
                _do_dahai(state, walk, kyoku_ordinal, ev, claim_after)
            idx += 1
            continue
        raise state.fail(kyoku_ordinal, kind, f"unmapped mjai event type {kind!r}")
    if not state.terminal:
        raise state.fail(kyoku_ordinal, "walk", "game never reached end_game")
    return state.rows


def replay_game(
    game: GameRecord | bytes | str | Path, *, split: str = "train", seat: int | None = None
) -> list[DecisionRow]:
    """Replay one framed MJAI game into actor ``DecisionRow`` rows.

    ``game`` is a :class:`~hydra2.data.decode.GameRecord` (as carried by
    :class:`~hydra2.data.stream.StreamGame`), framed game bytes, or a path to
    framed game bytes. Bytes/paths are decoded strictly first. The framed
    bytes are staged through a system temporary file (never the corpus) for
    ``MjaiReplay.from_jsonl``; ``take_kyokus`` plus ``steps(seat)`` for seats
    0..3 drive the row walk. ``split`` rides the rows opaquely (non-empty).
    ``seat`` optionally restricts emitted rows to one actor (decision ids stay
    positional over every decision so privileged joins still align).

    Wall-less provenance: ``wall_id`` is never bound (always ``None``); the
    derivation carries :data:`SIM_DERIVATION_MARK`. Any desync, illegal
    offered action, or firewall breach raises :class:`ContractError` naming
    game + kyoku + step.
    """
    record = _coerce_game(game)
    if split == "":
        raise ContractError("split must be a non-empty string")
    if seat is not None and (isinstance(seat, bool) or not isinstance(seat, int)):
        raise ContractError(f"seat must be a seat int or None, got {seat!r}")
    if seat is not None and not 0 <= seat <= 3:
        raise ContractError(f"seat must be 0..3, got {seat!r}")
    manifest = _rules()
    from hydra2.engines.riichienv.state import rules_identity_hash

    rules_hash = _rules_hash(manifest, str(rules_identity_hash(manifest)))
    table = _table()
    sim_game_id = _sim_game_id(record, rules_hash=rules_hash)
    builder = ObservationBuilder(
        game_id=sim_game_id,
        rules_id=manifest.rules_id,
        rules_hash=make_digest_text(rules_hash),
        action_table_hash=table.digest,
        expected_legal_mask_length=len(table.actions),
        event_schema_hash=make_digest_text(_event_schema_hash()),
        packet_boundary_hash=make_digest_text(_packet_boundary_hash()),
    )
    state = _GameState(
        game=record,
        split=split,
        seat_filter=seat,
        rules=manifest,
        rules_hash=rules_hash,
        table=table,
        sim_game_id=sim_game_id,
        builder=builder,
    )
    state.rows = []
    state.melds = ([], [], [], [])
    # Tenhou emits a bare {"type": "dora"} with no marker on kan-dora, which
    # the drain rejects whole-game with an external parse error. Own the
    # reason code here instead: the indicator is unrecoverable from the log.
    kyoku_idx = -1
    pending_ron: tuple[int, int] | None = None
    for event in record.events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "start_kyoku":
            kyoku_idx += 1
            pending_ron = None
        elif kind == "dora" and (
            not isinstance(event.get("dora_marker"), str) or event.get("dora_marker") == ""
        ):
            raise state.fail(
                kyoku_idx,
                "dora",
                "kan-dora indicator unrecoverable: dora event without a dora_marker",
            )
        elif kind == "hora":
            actor = event.get("actor")
            target = event.get("target")
            if (
                not bool(event.get("tsumo", False))
                and isinstance(actor, int)
                and not isinstance(actor, bool)
                and 0 <= actor <= 3
                and isinstance(target, int)
                and not isinstance(target, bool)
                and 0 <= target <= 3
                and target != actor
            ):
                if pending_ron is not None and pending_ron[1] == target and pending_ron[0] != actor:
                    raise state.fail(
                        kyoku_idx,
                        "hora",
                        "double ron on one discard is quarantined (single-winner pipeline)",
                    )
                pending_ron = (actor, target)
            else:
                pending_ron = None
        elif kind in _TRANSPARENT_KINDS:
            pass
        else:
            pending_ron = None
    framed = _frame_bytes(record)
    with tempfile.TemporaryDirectory(prefix="hydra2-simreplay-") as tmpdir:
        staged = str(Path(tmpdir) / "game.mjai.jsonl")
        _ = Path(staged).write_bytes(framed)  # staging write; byte count unneeded
        drained = _drain_game_steps(staged, game_id=record.game_id)
        rows = _walk_game(state, drained)
    return rows


# ---------------------------------------------------------------------------
# Raw-engine window/furiten oracle.
# ---------------------------------------------------------------------------

#: Log event kinds skipped when looking behind/ahead through a kyoku.
_TRANSPARENT_KINDS = frozenset({"dora", "reach_accepted"})


def _peek_window_claim(
    events: Sequence[dict[str, object]], start: int, discarder: int
) -> dict[str, object] | None:
    """Next window claim on ``discarder``'s tile, if the log shows one.

    Scans forward past transparent events; a draw decision, another offer, a
    terminal, or any boundary ends the window with no claim.
    """
    idx = start
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = str(event.get("type", ""))
        if kind in _TRANSPARENT_KINDS:
            idx += 1
            continue
        if kind in ("chi", "pon", "daiminkan"):
            target = event.get("target")
            if target == discarder:
                return event
            return None
        if kind == "hora" and not bool(event.get("tsumo", False)):
            # Tenhou-real tsumo wins carry no flag; target==actor marks them
            # and they never open a discard window.
            if event.get("target") == event.get("actor"):
                return None
            if event.get("target") == discarder:
                return event
            return None
        return None
    return None


class _WindowOracle:
    """Throwaway raw-engine oracle answering windows and furiten exactly.

    One pinned ``riichienv.RiichiEnv`` per kyoku, reset with an ephemeral
    countdown wall built from the log itself (tehais in deal order, logged
    tsumo in draw order with rinshan tiles at the dead-wall tail, filler
    elsewhere). The wall is purely mechanical and its digest is never bound
    anywhere. The log's events are translated to raw steps with string-level
    matching; any mismatch fails closed (quarantine upstream), never
    approximated.

    Answered queries (the ONLY surfaces consumed):

    - ``window_open()``: the engine's own claim-window predicate (phase plus
      responder legals) after a discard or kakan step;
    - ``furiten(pid)``: :func:`furiten_of` on the live engine at row time;
    - ``check_row(...)``: string-level hand/river/meld agreement between the
      seat-filtered oracle observation and the throwaway state.

    Wins, draws, dora reveals, and reach acceptances are never translated:
    the kyoku is decided by then and no further query can occur before the
    next reset.
    """

    def __init__(self, *, game_id: str) -> None:
        self._game_id = game_id
        self._env: Any = None
        self._kyoku = -1
        self._missed: set[int] = set()

    def _fail(self, why: str) -> ContractError:
        return ContractError(f"sim replay desync game {self._game_id!r} kyoku {self._kyoku}: {why}")

    def reset_kyoku(
        self,
        *,
        ordinal: int,
        oya: int,
        scores: tuple[int, int, int, int],
        honba: int,
        kyotaku: int,
        bakaze: str,
        tehais: tuple[Any, ...],
        live_draws: list[str],
        rinshan_draws: list[str],
    ) -> None:
        self._kyoku = ordinal
        self._missed = set()
        if len(tehais) != 4:
            raise self._fail(f"tehais cover {len(tehais)} seats, not 4")
        # Distinct physical copies per occurrence (the engine's multiset
        # accounting assumes unique tiles; collapsed ids corrupt it). Tile
        # conservation bounds every string to its four copies -- exhausting a
        # pool means an impossible log and fails closed.
        pools: dict[str, list[int]] = {}
        taken: dict[str, int] = {}

        def take(pai: str) -> int:
            if pai not in pools:
                first = int(physical_of(pai))
                if pai in ("5mr", "0m"):
                    pools[pai] = [16]
                elif pai in ("5pr", "0p"):
                    pools[pai] = [52]
                elif pai in ("5sr", "0s"):
                    pools[pai] = [88]
                else:
                    base = (first // 4) * 4
                    ids = [base, base + 1, base + 2, base + 3]
                    if len(pai) == 2 and pai[0] == "5":
                        ids = [i for i in ids if i != base]
                    pools[pai] = ids
            cursor = taken.get(pai, 0)
            if cursor >= len(pools[pai]):
                raise self._fail(f"tile string {pai!r} overused (tile conservation)")
            taken[pai] = cursor + 1
            return pools[pai][cursor]

        wall: list[int] = [-1] * 136
        # Deal order rotates with the dealer: block b feeds seat
        # (oya + b) mod 4, so seat s takes blocks congruent to (s - oya).
        for seat in range(4):
            hand = list(tehais[seat])
            if len(hand) != 13:
                raise self._fail(f"seat {seat} tehais hold {len(hand)} tiles, not 13")
            rel = (seat - oya) % 4
            for j in range(12):
                k, m = divmod(j, 4)
                wall[4 * (rel + 4 * k) + m] = take(str(hand[j]))
            wall[48 + rel] = take(str(hand[12]))
        for i, pai in enumerate(live_draws):
            wall[52 + i] = take(pai)
        for i, pai in enumerate(rinshan_draws):
            wall[135 - i] = take(pai)
        used = {t for t in wall if t != -1}
        filler = (t for t in range(136) if t not in used)
        wall = [t if t != -1 else next(filler) for t in wall]
        wind = {"E": 0, "S": 1, "W": 2, "N": 3}[bakaze]
        env = riichienv.RiichiEnv(
            game_mode=riichienv.GameType.YON_HANCHAN,
            rule=riichienv.GameRule.default_tenhou(),
            seed=ordinal,
        )
        try:
            _ = env.reset(
                oya=oya,
                wall=wall,
                scores=list(scores),
                honba=honba,
                kyotaku=kyotaku,
                round_wind=wind,
            )  # discard initial obs; reset side-effect installs the position
        except Exception as exc:
            raise self._fail(f"engine reset failed: {exc}") from exc
        self._env = env

    @property
    def _engine(self) -> Any:
        if self._env is None:
            raise self._fail("oracle engine used before reset")
        return self._env

    def _legals(self, pid: int) -> list[Any]:
        try:
            return list(self._engine.get_observation(pid).legal_actions())
        except Exception as exc:
            raise self._fail(f"seat {pid} legal query failed: {exc}") from exc

    @staticmethod
    def _mjai(raw: Any) -> dict[str, Any]:
        mjai: Any = raw.to_mjai()
        if isinstance(mjai, str):
            mjai = json.loads(mjai)
        if not isinstance(mjai, dict):
            raise ContractError(f"engine action without an MJAI mapping: {mjai!r}")
        return dict(mjai)

    def _find(
        self,
        pid: int,
        *,
        mjai_type: str,
        tile_str: str | None = None,
        consumed_strs: Sequence[str] | None = None,
    ) -> Any:
        matches: list[Any] = []
        for raw in self._legals(pid):
            try:
                mjai = self._mjai(raw)
            except (ContractError, ValueError):
                continue
            if str(mjai.get("type", "")) != mjai_type:
                continue
            if tile_str is not None:
                tile_raw: Any = raw.tile
                if tile_raw is None or mjai_string_of(int(tile_raw)) != tile_str:
                    continue
            if consumed_strs is not None:
                got = sorted(mjai_string_of(int(t)) for t in raw.consume_tiles)
                if got != sorted(consumed_strs):
                    continue
            matches.append(raw)
        if len(matches) == 0:
            raise self._fail(f"seat {pid} has no {mjai_type} offer for {tile_str!r}")
        return matches[0]

    def _step(self, moves: dict[int, Any], *, where: str) -> None:
        try:
            self._engine.step(moves)
        except Exception as exc:
            raise self._fail(f"engine step failed at {where}: {exc}") from exc

    def note_tsumo(self, actor: int, pai: str) -> None:
        """Assert-or-draw the logged tsumo (rinshan draws included)."""
        self._missed.discard(actor)
        drawn_raw: Any = self._engine.drawn_tile
        if drawn_raw is not None and mjai_string_of(int(drawn_raw)) == pai:
            self._confirm_draw_holder(actor, pai, drawn_raw)
            return
        if drawn_raw is not None:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._step({}, where=f"tsumo seat {actor}")
        drawn_raw = self._engine.drawn_tile
        if drawn_raw is None or mjai_string_of(int(drawn_raw)) != pai:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._confirm_draw_holder(actor, pai, drawn_raw)

    def _confirm_draw_holder(self, actor: int, pai: str, drawn_raw: Any) -> None:
        """Fail closed when the drawn tile sits in another seat's hand.

        The engine auto-draws the next seat on discard steps while draws
        match by string, so an out-of-turn log tsumo can string-match a tile
        the engine dealt elsewhere; physical-id membership names the true
        drawer at the tsumo instead of desyncing a later row.
        """
        try:
            hands: Any = self._engine.hands
            owned = int(drawn_raw) in {int(t) for t in hands[actor]}
        except Exception as exc:
            raise self._fail(f"seat {actor} drawer query failed: {exc}") from exc
        if not owned:
            raise self._fail(
                f"seat {actor} tsumo {pai!r} sits in another seat's hand (out-of-turn draw)"
            )

    def do_dahai(self, actor: int, pai: str) -> None:
        """Step one logged discard (tsumogiri and tedashi share a slot)."""
        action = self._find(actor, mjai_type="dahai", tile_str=pai)
        self._step({actor: action}, where=f"dahai seat {actor}")

    def do_reach(self, actor: int) -> None:
        """Step the riichi declaration (the discard follows separately)."""
        action = self._find(actor, mjai_type="reach")
        self._step({actor: action}, where=f"reach seat {actor}")

    def do_ankan(self, actor: int, consumed: Sequence[str]) -> None:
        """Step one logged closed kan."""
        action = self._find(actor, mjai_type="ankan", consumed_strs=list(consumed))
        self._step({actor: action}, where=f"ankan seat {actor}")

    def do_kakan(self, actor: int, pai: str) -> None:
        """Step one logged added kan."""
        action = self._find(actor, mjai_type="kakan", tile_str=pai)
        self._step({actor: action}, where=f"kakan seat {actor}")

    def do_tsumo_win(self, actor: int, pai: str) -> None:
        """Step one logged self-draw win."""
        for mjai_type in ("hora", "tsumo"):
            try:
                action = self._find(actor, mjai_type=mjai_type, tile_str=pai)
            except ContractError:
                continue
            self._step({actor: action}, where=f"tsumo seat {actor}")
            return
        raise self._fail(f"seat {actor} has no tsumo win offer for {pai!r}")

    def window_open(self, discarder: int) -> bool:
        """The engine's own claim-window predicate after a discard/kan step."""
        env = self._engine
        try:
            waiting = int(env.phase) == int(riichienv.Phase.WaitResponse)
        except Exception as exc:
            raise self._fail(f"phase query failed: {exc}") from exc
        if not waiting:
            return False
        for pid in range(4):
            if pid == discarder:
                continue
            if len(self._legals(pid)) > 0:
                return True
        return False

    def resolve_window(self, discarder: int, claim: dict[str, object] | None) -> None:
        """Step one combined window resolution (claim or all-pass).

        Ron windows are never stepped: winning terminates the engine into an
        auto-dealt next hand (all subsequent queries would read garbage), and
        the kyoku is decided by the logged hora anyway. Responders passing a
        genuine ron offer are recorded as temporarily furiten manually; the
        winner is excluded (they won, not passed).
        """
        if claim is not None and str(claim.get("type", "")) == "hora":
            actor_raw = claim.get("actor")
            winner = None if isinstance(actor_raw, bool) else actor_raw
            for pid in range(4):
                if pid in (discarder, winner):
                    continue
                try:
                    legals = self._legals(pid)
                except ContractError:
                    continue
                for raw in legals:
                    try:
                        mjai = self._mjai(raw)
                    except (ContractError, ValueError):
                        continue
                    if str(mjai.get("type", "")) in ("hora", "ron"):
                        self._missed.add(pid)
                        break
            return
        moves: dict[int, Any] = {}
        if claim is not None:
            kind = str(claim.get("type", ""))
            actor_raw = claim.get("actor")
            if isinstance(actor_raw, bool) or not isinstance(actor_raw, int):
                raise self._fail("window claim without an actor")
            actor = actor_raw
            if kind not in ("chi", "pon", "daiminkan"):
                raise self._fail(f"window claim of unexpected kind {kind!r}")
            pai = str(claim.get("pai", ""))
            consumed = [str(t) for t in cast("Any", claim.get("consumed", []))]
            moves[actor] = self._find(actor, mjai_type=kind, tile_str=pai, consumed_strs=consumed)
        for pid in range(4):
            if pid == discarder or pid in moves:
                continue
            if len(self._legals(pid)) == 0:
                continue
            moves[pid] = self._find(pid, mjai_type="none")
        if len(moves) == 0:
            return
        self._step(moves, where="window resolution")

    def furiten(self, pid: int) -> str:
        """Exact furiten state from the live engine flags plus ron passes.

        Ron windows are never stepped (winning would terminate the engine),
        so responders passing a genuine ron offer are recorded manually; the
        engine's own doujun/riichi flags cover everything else.
        """
        from hydra2.engines.riichienv.state import furiten_of

        try:
            state = furiten_of(self._engine, pid)
        except Exception as exc:
            raise self._fail(f"seat {pid} furiten query failed: {exc}") from exc
        if state == "none" and pid in self._missed:
            return "temporary"
        return state

    def offers_kyushu(self, pid: int) -> bool:
        """Whether the live engine offers the nine-terminals abort to ``pid``."""
        for raw in self._legals(pid):
            try:
                action_type = int(cast("Any", raw.action_type))
            except (TypeError, ValueError):
                continue
            if action_type == int(riichienv.ActionType.KYUSHU_KYUHAI):
                return True
        return False

    def position(self, pid: int) -> _TablePosition:
        """Frozen live-table snapshot backing oracle-less forced rows."""
        env = self._engine
        try:
            hands_raw: Any = env.hands
            rivers_raw: Any = env.discards
            drawn_raw: Any = env.drawn_tile
            dora_raw: Any = env.dora_indicators
            sticks_raw: Any = env.riichi_sticks
            declared_raw: Any = env.riichi_declared
            legals = list(self._legals(pid))
            hand = tuple(int(t) for t in hands_raw[pid])
            rivers = tuple(tuple(int(t) for t in river) for river in rivers_raw)
            lens = tuple(len(cast("Any", item)) for item in hands_raw)
            dora = tuple(int(t) for t in dora_raw)
            sticks = int(sticks_raw)
            declared = tuple(bool(v) for v in declared_raw)
            drawn = None if drawn_raw is None else int(drawn_raw)
        except ContractError:
            raise
        except Exception as exc:
            raise self._fail(f"seat {pid} position query failed: {exc}") from exc
        if len(declared) != 4 or len(lens) != 4 or len(rivers) != 4:
            raise self._fail(f"seat {pid} position must cover 4 seats")
        return _TablePosition(
            hand=hand,
            drawn=drawn,
            dora=dora,
            rivers=rivers,
            lens=lens,
            sticks=sticks,
            declared=declared,
            legals=tuple(legals),
        )

    def check_row(
        self,
        seat: int,
        step: _SimStep,
        *,
        melds: Sequence[Sequence[VisibleMeld]],
        hand_check: bool,
    ) -> None:
        """String-level agreement between oracle observation and engine state.

        Draw rows compare hands (with drawn tile), rivers, and meld counts;
        claim rows compare rivers and meld counts only (hands move through
        the meld at different points of the two pipelines).
        """
        env = self._engine
        try:
            hands: Any = env.hands
            rivers: Any = env.discards
            meld_rows: Any = env.melds
        except Exception as exc:
            raise self._fail(f"seat {seat} state query failed: {exc}") from exc
        if hand_check:
            hand_strings = sorted(mjai_string_of(t) for t in step.hand)
            engine_strings = sorted(mjai_string_of(int(t)) for t in hands[seat])
            if hand_strings != engine_strings:
                raise self._fail(f"seat {seat} hand differs from the engine state")
            if step.drawn is not None:
                drawn_raw: Any = env.drawn_tile
                if drawn_raw is None or mjai_string_of(int(drawn_raw)) != mjai_string_of(
                    step.drawn
                ):
                    raise self._fail(f"seat {seat} drawn tile differs from engine state")
        for other in range(4):
            river_strings = [mjai_string_of(int(t)) for t in rivers[other]]
            oracle_strings = [mjai_string_of(t) for t in step.discards[other]]
            if river_strings != oracle_strings:
                raise self._fail(f"seat {other} river differs from engine state")
        for owner in range(4):
            if len(meld_rows[owner]) != len(melds[owner]):
                raise self._fail(f"seat {owner} meld count differs from engine state")
