"""Wall-less sim replay: window, draw, declaration handlers."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep
from hydra2.engines.riichienv._lr_rows import (
    _emit as _emit,
)
from hydra2.engines.riichienv._lr_rows import (
    _ippatsu_interrupt as _ippatsu_interrupt,
)
from hydra2.engines.riichienv._lr_walk import (
    _drop_orphans as _drop_orphans,
)
from hydra2.engines.riichienv._lr_walk import (
    _peek as _peek,
)
from hydra2.engines.riichienv._lr_walk import _pop as _pop
from hydra2.engines.riichienv._lr_walk import _pop_window_heads as _pop_window_heads
from hydra2.engines.riichienv._lr_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._oracle_base import (
    _BAKAZE_TO_WIND as _BAKAZE_TO_WIND,
)
from hydra2.engines.riichienv._oracle_base import (
    _TRANSPARENT_KINDS as _TRANSPARENT_KINDS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.engines.riichienv._lr_frame import _TablePosition as _TablePosition
    from hydra2.engines.riichienv._lr_rows import _GameState as _GameState
    from hydra2.engines.riichienv._lr_walk import _KyokuWalk as _KyokuWalk


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
            rendered = tiles.mjai_string_of(tile)
        except ValueError:
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
        tile=int(tiles.physical_of(pai)),
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
    if drawn is None or tiles.mjai_string_of(drawn) != pai:
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
    if pos.drawn is None or tiles.mjai_string_of(pos.drawn) != pai:
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
        actor=_bridge_contracts.make_seat(actor),
        tile=_bridge_contracts.make_tile_id(pos.drawn),
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
        and tiles.mjai_string_of(head.tile) == pai
        and head.drawn is not None
        and pos.drawn is not None
        and tiles.mjai_string_of(head.drawn) == tiles.mjai_string_of(pos.drawn)
    ):
        step = _pop(state, walk, kyoku, actor, why="dahai")
    elif (
        pos.drawn is not None
        and tiles.mjai_string_of(pos.drawn) == pai
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
    if step.tile is None or tiles.mjai_string_of(step.tile) != pai:
        raise state.fail(kyoku, "dahai", f"seat {actor} oracle discards a different tile")
    kind = "tsumogiri" if tsumogiri else "discard"
    expected = CanonicalAction(
        kind=cast("Any", kind),
        actor=_bridge_contracts.make_seat(actor),
        tile=_bridge_contracts.make_tile_id(step.tile),
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
