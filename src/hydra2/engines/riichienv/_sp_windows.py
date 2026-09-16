"""Single-engine wall-less replay: live windows and draws."""

from __future__ import annotations

from dataclasses import replace as replace
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.engines.riichienv._oracle_base import _BAKAZE_TO_WIND as _BAKAZE_TO_WIND
from hydra2.engines.riichienv._oracle_base import _legal_mjai_type as _legal_mjai_type
from hydra2.engines.riichienv._sp_capture import _distinct_copies as _distinct_copies
from hydra2.engines.riichienv._sp_records import _emit as _emit
from hydra2.engines.riichienv._sp_records import _ippatsu_interrupt as _ippatsu_interrupt
from hydra2.engines.riichienv._sp_records import _safe_mjai_type as _safe_mjai_type
from hydra2.engines.riichienv._sp_records import _SimStep as _SimStep
from hydra2.engines.riichienv._sp_walk import _strict_row as _strict_row

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_walk import _KyokuWalk as _KyokuWalk


def _snapshot_window(state: _GameState, walk: _KyokuWalk, kyoku: int, discarder: int) -> None:
    """Snapshot every responder's live pre-resolution decision point.

    The engine sits in WaitResponse with this window's offers live; freezing
    each responder now replaces the drained queue pops (matching happens at
    claim time against the frozen legals, exactly like the stashed heads).
    """
    for seat in range(4):
        if seat == discarder:
            continue
        walk.stash[seat] = _live_step(state, walk, kyoku, seat, mjai_type="none")


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


def _fold_offer(raw: Any) -> Any:
    """Fold one live offer's tile/consume ids to rendering-rule copies.

    The drained oracle rendered discards at string base and claim consumes
    as occurrence pools; the live engine deals take-ordered physicals.
    Folding here (never the wall) keeps expansion, matching, and masks on
    identical ids. Anything unparseable passes through untouched and fails
    closed at its existing use-site.
    """
    from types import SimpleNamespace  # local single-use adapter (see _expand_nonclaim_legals)

    try:
        _ = _legal_mjai_type(raw)  # validate MJAI mapping; type string unneeded
        tile_raw: Any = raw.tile
        folded_tile = (
            None
            if tile_raw is None
            else int(tiles.physical_of(tiles.mjai_string_of(int(tile_raw))))
        )
        folded_consume = _distinct_copies(tuple(int(t) for t in raw.consume_tiles))
    except Exception:  # why-broad: folding never invents ids; use-site fails closed
        return raw
    return SimpleNamespace(
        action_type=raw.action_type,
        tile=folded_tile,
        consume_tiles=folded_consume,
        to_mjai=raw.to_mjai,
    )


def _live_step(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    seat: int,
    *,
    mjai_type: str,
    tile: int | None = None,
    consume: tuple[int, ...] = (),
    fold: bool = True,
) -> _SimStep:
    """Freeze the live engine's current decision point as a row source.

    Folded mode renders hands as per-string occurrence pools, the drawn tile
    at its string base, and offers folded likewise -- exactly the drained
    oracle's rendering (the log says what was chosen, the engine owns what
    was offered and held; copy identity is a rendering rule, never wall
    content). Raw mode (post-reach forced discards, which the drain never
    yielded) binds the engine's take-ordered physicals verbatim.
    """
    try:
        pos = walk.tw.position(seat)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} step", f"position query failed: {exc}") from exc
    if fold:
        hand = _distinct_copies(pos.hand)
        drawn: int | None = None
        if pos.drawn is not None:
            drawn = tiles.physical_of(tiles.mjai_string_of(pos.drawn))
        legals = tuple(_fold_offer(raw) for raw in pos.legals)
    else:
        hand = tuple(t for t in pos.hand)
        drawn = None if pos.drawn is None else pos.drawn
        legals = pos.legals
    return _SimStep(
        seat=seat,
        mjai_type=mjai_type,
        tile=tile,
        consume=tuple(consume),
        mjai={},
        raw_action=None,
        raw_legals=legals,
        hand=hand,
        drawn=drawn,
        dora=pos.dora,
        discards=pos.rivers,
        hands_lens=pos.lens,
        scores=pos.scores,
        sticks=pos.sticks,
        oya=state.dealer,
        honba=state.honba,
        round_wind=_BAKAZE_TO_WIND[state.bakaze],
        player_id=seat,
        riichi=pos.declared,
    )


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
    """Snapshot responders and resolve the window through the single engine.

    Pre-resolution snapshots replace the drained queue pops; the engine's own
    claim-window predicate decides ``call_window`` / ``call_resolved``
    existence. A stash holding a claim the log never shows fails closed at
    window close.
    """
    _snapshot_window(state, walk, kyoku, discarder)
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

    Stashed passes and unlogged claim/ron snapshots are discarded: the log is
    the authoritative decision record, and rows follow logged decisions only.
    A logged claim without engine support still fails closed at claim-match
    time. (In particular, a passed ron leaves its snapshot here; the missed
    furiten it implies is tracked by the engine, not this stash.)
    """
    del kyoku, where
    walk.stash.clear()
    if state.cr_pending:
        _emit_call_resolved(state, accepted=())


def _do_tsumo(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    event: dict[str, object],
) -> None:
    actor = _require_actor(state, kyoku, event, where="tsumo")
    if state.decided:
        raise state.fail(kyoku, "tsumo", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="tsumo")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, "tsumo", "draw without a pai string")
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
    if state.riichi_declared[actor]:
        # Post-reach forced discard (the drain yields no step for these):
        # raw take-ordered ids, tsumogiri by rule. A non-drawn discard here
        # has no oracle step either and fails closed like the empty queue.
        step = _live_step(state, walk, kyoku, actor, mjai_type="dahai", fold=False)
        if step.drawn is None or tiles.mjai_string_of(step.drawn) != pai:
            raise state.fail(kyoku, "dahai", f"seat {actor} pop: oracle queue empty (dahai)")
        tile_raw = step.drawn
        step = replace(step, mjai_type="dahai", tile=tile_raw, consume=())
        kind = "tsumogiri"
    else:
        step = _live_step(state, walk, kyoku, actor, mjai_type="dahai")
        # Firewall: the engine must offer a dahai of the logged string.
        if not any(
            _safe_mjai_type(raw) == "dahai"
            and raw.tile is not None
            and tiles.mjai_string_of(int(raw.tile)) == pai
            for raw in step.raw_legals
        ):
            raise state.fail(kyoku, "dahai", f"seat {actor} oracle discards a different tile")
        # Tile identity is the string base (the oracle renders every normal
        # discard at its string's first copy).
        tile_raw = int(tiles.physical_of(pai))
        step = replace(step, mjai_type="dahai", tile=tile_raw, consume=())
        kind = "tsumogiri" if tsumogiri else "discard"
    expected = CanonicalAction(
        kind=cast("Any", kind),
        actor=make_seat(actor),
        tile=make_tile_id(tile_raw),
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
        tile=tile_raw,
        action_id=chosen_id,
    )
    state.last_discard = (actor, tile_raw)
    state.opened_by_discard = True
    walk.drawer = actor
    _open_window(state, walk, kyoku, actor, tile_raw, claim_ev)
    _ippatsu_interrupt(state, actor)  # declarer discarded again: window gone
