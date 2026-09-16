"""Wall-less sim replay: reach, claim, kan handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.action import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.contracts.observation import VisibleMeld as VisibleMeld
from hydra2.contracts.observation import visible_meld_id as visible_meld_id
from hydra2.engines.riichienv._lr_act import _check_drawer as _check_drawer
from hydra2.engines.riichienv._lr_act import _emit_call_resolved as _emit_call_resolved
from hydra2.engines.riichienv._lr_act import _open_window as _open_window
from hydra2.engines.riichienv._lr_act import _require_actor as _require_actor
from hydra2.engines.riichienv._lr_act import _resolve_dora as _resolve_dora
from hydra2.engines.riichienv._lr_rows import _capture_row as _capture_row
from hydra2.engines.riichienv._lr_rows import _claim_canonical as _claim_canonical
from hydra2.engines.riichienv._lr_rows import _copies_of_string as _copies_of_string
from hydra2.engines.riichienv._lr_rows import _emit as _emit
from hydra2.engines.riichienv._lr_rows import _expand_nonclaim_legals as _expand_nonclaim_legals
from hydra2.engines.riichienv._lr_rows import _ippatsu_interrupt as _ippatsu_interrupt
from hydra2.engines.riichienv._lr_rows import _ippatsu_open as _ippatsu_open
from hydra2.engines.riichienv._lr_rows import _tracked_consumed as _tracked_consumed
from hydra2.engines.riichienv._lr_rows import _tracked_discard_tile as _tracked_discard_tile
from hydra2.engines.riichienv._lr_walk import _peek as _peek
from hydra2.engines.riichienv._lr_walk import _pop as _pop
from hydra2.engines.riichienv._lr_walk import _pop_draw_head as _pop_draw_head
from hydra2.engines.riichienv._lr_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._oracle_base import _legal_mjai_type as _legal_mjai_type
from hydra2.engines.riichienv.events import make_delta as make_delta
from hydra2.engines.riichienv.events import meld_delta_value as meld_delta_value

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep
    from hydra2.engines.riichienv._lr_rows import _GameState as _GameState
    from hydra2.engines.riichienv._lr_walk import _KyokuWalk as _KyokuWalk


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
        and tiles.mjai_string_of(decl_head.tile) == pai
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
    if step.tile is None or tiles.mjai_string_of(step.tile) != pai:
        raise state.fail(kyoku, kind, "claim names a different tile than the oracle")
    consumed = event.get("consumed")
    if not isinstance(consumed, (list, tuple)):
        raise state.fail(kyoku, kind, "logged claim without consumed tiles")
    target = event.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
        raise state.fail(kyoku, kind, "logged claim without a valid target seat")
    if target != discarder:
        raise state.fail(kyoku, kind, f"claim target {target} != discarder {discarder}")
    yielded_strings = sorted(tiles.mjai_string_of(t) for t in step.consume)
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
    yielded_strings = sorted(tiles.mjai_string_of(t) for t in step.consume)
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
    if step.tile is None or tiles.mjai_string_of(step.tile) != pai:
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
        base = (int(tiles.physical_of(pai)) // 4) * 4
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
