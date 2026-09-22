"""Wall-less replay end-of-kyoku and end-of-game handlers (WP-14).

Single home for `_do_end_kyoku` plus `_do_end_game` shared by wall-less
replay drivers. The engine owns the firewall; this file only emits the
public `round_end` deltas and the terminal flag with no wall digest.
Failure mode is fail-closed `ContractError` naming game plus kyoku plus
step via `state.fail`, never silent drop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2.engines.riichienv._lr_act import (
    _clear_stash_no_claim as _clear_stash_no_claim,
)
from hydra2.engines.riichienv.events import (
    make_delta as make_delta,
)

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.engines.riichienv._lr_rows import (
        _GameState as _GameState,
    )
    from hydra2.engines.riichienv._lr_walk import (
        _KyokuWalk as _KyokuWalk,
    )


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
    from hydra2.engines.riichienv._lr_rows import _emit as _emit

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
