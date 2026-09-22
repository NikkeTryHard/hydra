"""Single-engine wall-less replay: kans and dora."""

from __future__ import annotations

from dataclasses import replace as replace
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import tiles  # pyrefly: ignore[missing-import]
from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.observation_types import (
    VisibleMeld as VisibleMeld,
)
from hydra2.contracts.observation_types import (
    visible_meld_id as visible_meld_id,
)
from hydra2.engines.riichienv._sp_capture import _tracked_consumed as _tracked_consumed
from hydra2.engines.riichienv._sp_records import (
    _emit as _emit,
)
from hydra2.engines.riichienv._sp_records import _ippatsu_interrupt as _ippatsu_interrupt
from hydra2.engines.riichienv._sp_records import _ippatsu_open as _ippatsu_open
from hydra2.engines.riichienv._sp_records import _safe_mjai_type as _safe_mjai_type
from hydra2.engines.riichienv._sp_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._sp_windows import (
    _check_drawer as _check_drawer,
)
from hydra2.engines.riichienv._sp_windows import (
    _live_step as _live_step,
)
from hydra2.engines.riichienv._sp_windows import _open_window as _open_window
from hydra2.engines.riichienv._sp_windows import _require_actor as _require_actor
from hydra2.engines.riichienv._sp_windows import _resolve_dora as _resolve_dora
from hydra2.engines.riichienv.events import (
    make_delta as make_delta,
)
from hydra2.engines.riichienv.events import (
    meld_delta_value as meld_delta_value,
)

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.contracts.common import Seat as Seat
    from hydra2.contracts.common import TileId as TileId
    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_walk import _KyokuWalk as _KyokuWalk


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
    logged_raw: list[object] = list(consumed)
    logged = sorted(str(t) for t in logged_raw)
    try:
        offer = walk.tw._find(actor, mjai_type="ankan", consumed_strs=logged)
    except ContractError as exc:
        raise state.fail(
            kyoku, "ankan", f"seat {actor} oracle holds no ankan offer: {exc}"
        ) from exc
    offer_consume: list[int] = list(offer.consume_tiles)  # pyrefly: ignore[unknown-argument-type] # engine action consume tiles dynamic
    step = _live_step(
        state,
        walk,
        kyoku,
        actor,
        mjai_type="ankan",
        consume=tuple(sorted(offer_consume)),
    )
    try:
        block = _tracked_consumed(step.hand, [str(t) for t in logged_raw], needed=4)
    except ContractError as exc:
        raise state.fail(kyoku, "ankan", f"ankan tiles not owned: {exc}") from exc
    base = (block[0] // 4) * 4
    if tuple(block) != (base, base + 1, base + 2, base + 3):
        raise state.fail(kyoku, "ankan", f"ankan tiles {block!r} are not one block")
    ankan_seat: Seat = _bridge_contracts.make_seat(actor)
    expected = CanonicalAction(
        kind=cast("Any", "ankan"),
        actor=ankan_seat,
        tile=None,
        called_tile=None,
        consumed_tiles=tuple(_bridge_contracts.make_tile_id(t) for t in block),
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
        walk.tw.do_ankan(actor, [str(t) for t in logged_raw])
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
    meld_owner: Seat = _bridge_contracts.make_seat(actor)
    state.melds[actor].append(
        VisibleMeld(
            meld_id=None,
            kind=cast("Any", "ankan"),
            owner=meld_owner,
            tiles=tuple(_bridge_contracts.make_tile_id(t) for t in block),
        )
    )
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
    step = _live_step(state, walk, kyoku, actor, mjai_type="kakan")
    kakan_legals: list[Any] = list(step.raw_legals)
    if not any(
        _safe_mjai_type(raw) == "kakan"
        and raw.tile is not None
        and tiles.mjai_string_of(int(cast("Any", raw.tile))) == pai
        for raw in kakan_legals
    ):
        raise state.fail(kyoku, "kakan", "kakan names a different tile than the oracle")
    # The added fourth copy is the one pool copy of this type absent from the
    # prior pon triple (deterministic, contract-valid, and exact on
    # min-rule-disciplined logs); meld tiles are folded, so the trick lands
    # on the same copy as the drained pipeline.
    pool: list[int] = tiles.copies_of_string(pai)
    if pai[0] == "5" and len(pai) == 2:
        physical: int = tiles.physical_of(pai)
        base = (physical // 4) * 4
        pool = [base, base + 1, base + 2, base + 3]
    try:
        prior = _find_prior_pon(state, actor, pool[0])
    except ContractError as exc:
        raise state.fail(kyoku, "kakan", str(exc)) from exc
    prior_tiles: list[int] = [int(t) for t in prior.tiles]
    missing = [c for c in pool if c not in set(prior_tiles)]
    if len(missing) != 1:
        raise state.fail(kyoku, "kakan", "prior pon leaves no single added copy")
    added: int = missing[0]
    step = replace(step, mjai_type="kakan", tile=added)
    kakan_seat: Seat = _bridge_contracts.make_seat(actor)
    kakan_tile: TileId = _bridge_contracts.make_tile_id(added)
    expected = CanonicalAction(
        kind=cast("Any", "kakan"),
        actor=kakan_seat,
        tile=kakan_tile,
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
    # Kan-dora indicators are log-faithful first-copies per marker (census:
    # the drain binds the first physical copy of every marker string); the
    # engine's filler dead wall can never name them. Track reveals in Python.
    indicator: int = tiles.physical_of(marker)
    walk.dora_revealed.append(indicator)
    walk.last_oracle_dora = tuple(walk.dora_revealed)
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
    state.reach_accepted += 1
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
