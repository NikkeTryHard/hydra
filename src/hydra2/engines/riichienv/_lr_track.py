"""Wall-less sim replay: wall-tracking estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING as TYPE_CHECKING

import riichienv
from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import ContractError as ContractError

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep
    from hydra2.engines.riichienv._lr_rows import _GameState as _GameState
    from hydra2.engines.riichienv._lr_walk import _KyokuWalk as _KyokuWalk


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
    pool = tiles.copies_of_string(pai)
    taken = walk.take_taken.get(pai, 0)
    if taken >= len(pool):
        raise state.fail(kyoku, where, f"tile string {pai!r} overused (tile conservation)")
    walk.take_taken[pai] = taken + 1
    return pool[taken]


def _wall_filler_ids(walk: _KyokuWalk) -> list[int]:
    """Deterministic filler ids: numeric complement of every taken copy."""
    used: set[int] = set()
    for pai, count in walk.take_taken.items():
        used.update(tiles.copies_of_string(pai)[:count])
    return sorted(set(range(136)) - used)


def _track_remove(
    hand: list[int], pai: str, state: _GameState, kyoku: int, *, drawn: int | None, where: str
) -> int:
    """Remove one tracked copy rendering ``pai`` (drawn tile preferred, fail closed)."""
    if drawn is not None and tiles.mjai_string_of(drawn) == pai and drawn in hand:
        hand.remove(drawn)
        return drawn
    for tile in hand:
        if tiles.mjai_string_of(tile) == pai:
            hand.remove(tile)
            return tile
    raise state.fail(kyoku, where, f"no tracked copy of discard {pai!r} in hand")


def _track_remove_consumed(
    hand: list[int], consumed: Sequence[str], state: _GameState, kyoku: int, *, where: str
) -> None:
    """Remove one tracked copy per logged consumed string (fail closed)."""
    for pai in sorted(s for s in consumed):
        for tile in hand:
            if tiles.mjai_string_of(tile) == pai:
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
        rendered = _norm_pai(tiles.mjai_string_of(raw))
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
    win_tile = tiles.copies_of_string(tile)[0]
    if not _shape_is_win(concealed, melds, win_tile):
        return (False, False, False)
    river = {_norm_pai(tiles.mjai_string_of(t)) for t in walk.rivers[seat]}
    if _norm_pai(tile) in river:
        return (True, False, False)
    for kind in river:
        if _shape_is_win(concealed, melds, tiles.copies_of_string(kind)[0]):
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
    tile_norm = _norm_pai(tiles.mjai_string_of(tile))
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
            rendered = _norm_pai(tiles.mjai_string_of(raw))
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
        engine_hand = sorted(tiles.mjai_string_of(int(t)) for t in hands[seat])
        tracked_hand = sorted(tiles.mjai_string_of(t) for t in walk.hands[seat])
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
            queued = [tiles.mjai_string_of(entry[0]) for entry in walk.draw_queues[seat]]
            for rendered in remaining:
                if rendered in queued:
                    raise state.fail(
                        kyoku,
                        where,
                        f"tracker missed a logged pre-draw for seat {seat}",
                    )
        engine_river = [tiles.mjai_string_of(int(t)) for t in rivers[seat]]
        tracked_river = [tiles.mjai_string_of(t) for t in walk.rivers[seat]]
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
