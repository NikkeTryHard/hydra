"""Wall-less sim replay: per-kyoku walk."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import IllegalActionError as IllegalActionError
from hydra2.contracts.common import make_digest_text as make_digest_text
from hydra2.contracts.observation import ObservationBuilder as ObservationBuilder
from hydra2.engines.riichienv._lr_frame import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv._lr_frame import _packet_boundary_hash as _packet_boundary_hash
from hydra2.engines.riichienv._lr_oracle import _WindowOracle as _WindowOracle
from hydra2.engines.riichienv._lr_rows import _capture_row as _capture_row
from hydra2.engines.riichienv._lr_rows import _emit as _emit
from hydra2.engines.riichienv._lr_rows import _expand_nonclaim_legals as _expand_nonclaim_legals
from hydra2.engines.riichienv._oracle_base import _BAKAZE_TO_WIND as _BAKAZE_TO_WIND
from hydra2.engines.riichienv._oracle_base import _TRANSPARENT_KINDS as _TRANSPARENT_KINDS
from hydra2.engines.riichienv._oracle_base import _WINDOW_HEAD_TYPES as _WINDOW_HEAD_TYPES
from hydra2.engines.riichienv.events import make_delta as make_delta

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.action import CanonicalAction as CanonicalAction
    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep
    from hydra2.engines.riichienv._lr_rows import _GameState as _GameState


# ---------------------------------------------------------------------------
# Per-kyoku walk (queues + window stash).
# ---------------------------------------------------------------------------


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
    pai = tiles.mjai_string_of(tile)
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
            head.tile is None or tiles.mjai_string_of(head.tile) != pai
        ):
            continue
        if head.mjai_type == "hora" and not (
            head.drawn is None
            and head.tile is not None
            and tiles.mjai_string_of(head.tile) == tiles.mjai_string_of(tile)
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
