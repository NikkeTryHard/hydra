"""Single-engine wall-less replay: reach and claims."""

from __future__ import annotations

from dataclasses import replace as replace
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.action import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.contracts.observation import VisibleMeld as VisibleMeld
from hydra2.engines.riichienv._oracle_base import _legal_mjai_type as _legal_mjai_type
from hydra2.engines.riichienv._sp_capture import _capture_row as _capture_row
from hydra2.engines.riichienv._sp_capture import _claim_canonical as _claim_canonical
from hydra2.engines.riichienv._sp_capture import _tracked_consumed as _tracked_consumed
from hydra2.engines.riichienv._sp_records import _emit as _emit
from hydra2.engines.riichienv._sp_records import _expand_nonclaim_legals as _expand_nonclaim_legals
from hydra2.engines.riichienv._sp_records import _ippatsu_interrupt as _ippatsu_interrupt
from hydra2.engines.riichienv._sp_records import _safe_mjai_type as _safe_mjai_type
from hydra2.engines.riichienv._sp_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._sp_windows import _check_drawer as _check_drawer
from hydra2.engines.riichienv._sp_windows import _emit_call_resolved as _emit_call_resolved
from hydra2.engines.riichienv._sp_windows import _live_step as _live_step
from hydra2.engines.riichienv._sp_windows import _open_window as _open_window
from hydra2.engines.riichienv._sp_windows import _require_actor as _require_actor
from hydra2.engines.riichienv.events import make_delta as make_delta
from hydra2.engines.riichienv.events import meld_delta_value as meld_delta_value
from hydra2.engines.riichienv.tiles import mjai_string_of as mjai_string_of

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_records import _SimStep as _SimStep
    from hydra2.engines.riichienv._sp_walk import _KyokuWalk as _KyokuWalk


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
    try:
        walk.tw._find(actor, mjai_type="reach")
    except ContractError as exc:
        raise state.fail(kyoku, "reach", f"seat {actor} oracle holds no reach offer") from exc
    # The declaration discard is the live dahai offer naming the logged tile
    # (both house dialects collapse here: the engine offers the discard of
    # this turn whether or not the log splits reach and dahai in two). Its
    # identity resolves through the folded hand, drawn copy preferred.
    step = _live_step(state, walk, kyoku, actor, mjai_type="reach")
    if not any(
        _safe_mjai_type(raw) == "dahai"
        and raw.tile is not None
        and mjai_string_of(int(raw.tile)) == pai
        for raw in step.raw_legals
    ):
        raise state.fail(kyoku, "reach", "declaration discard not owned: no offer")
    declaration_tile: int | None = None
    for owned in step.hand:
        if mjai_string_of(owned) != pai:
            continue
        if declaration_tile is None:
            declaration_tile = owned
        if step.drawn is not None and owned == step.drawn:
            declaration_tile = owned
            break
    if declaration_tile is None:
        raise state.fail(kyoku, "reach", "declaration discard not owned: tile mismatch")
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
        step,
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
    """Validate the claimant's stashed window snapshot against the logged claim."""
    step = walk.stash.pop(seat, None)
    if step is None:
        raise state.fail(kyoku, kind, f"seat {seat} holds no window step for the claim")
    pai = event.get("pai")
    if not isinstance(pai, str) or pai == "":
        raise state.fail(kyoku, kind, "logged claim without a pai string")
    candidates: list[Any] = []
    for raw in step.raw_legals:
        try:
            is_kind = _legal_mjai_type(raw) == kind
        except (ContractError, ValueError):
            continue
        if not is_kind:
            continue
        tile_raw: Any = raw.tile
        if tile_raw is None or mjai_string_of(int(tile_raw)) != pai:
            continue
        candidates.append(raw)
    if len(candidates) == 0:
        raise state.fail(kyoku, kind, f"seat {seat} oracle holds no {kind} offer, log says {kind}")
    consumed = event.get("consumed")
    if not isinstance(consumed, (list, tuple)):
        raise state.fail(kyoku, kind, "logged claim without consumed tiles")
    logged_strings = sorted(str(t) for t in consumed)
    offer: Any = None
    for raw in candidates:
        # Live consume lists the hand tiles taken (the called tile is the
        # offer's own tile, not repeated inside); the log lists the same
        # hand tiles. Several chi variants can share one called tile, so
        # the logged consumed set selects the variant, never list order.
        yielded_strings = sorted(mjai_string_of(int(t)) for t in raw.consume_tiles)
        if yielded_strings == logged_strings:
            offer = raw
            break
    if offer is None:
        raise state.fail(kyoku, kind, "claim tiles differ from the oracle")
    target = event.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
        raise state.fail(kyoku, kind, "logged claim without a valid target seat")
    if target != discarder:
        raise state.fail(kyoku, kind, f"claim target {target} != discarder {discarder}")
    offer_consume = tuple(sorted(int(t) for t in offer.consume_tiles))
    for other, other_step in list(walk.stash.items()):
        if any(
            _legal_mjai_type(raw) in ("chi", "pon", "daiminkan") for raw in other_step.raw_legals
        ):
            del walk.stash[other]
    return replace(
        step,
        mjai_type=kind,
        tile=int(offer.tile),
        consume=offer_consume,
    )


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
    _ippatsu_interrupt(state, None)  # any call interrupts every chance
