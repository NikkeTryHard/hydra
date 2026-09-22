"""Single-engine wall-less replay: per-kyoku walk."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.action_table import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import (
    IllegalActionError as IllegalActionError,
)
from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder
from hydra2.engines.riichienv._oracle_base import (
    _BAKAZE_TO_WIND as _BAKAZE_TO_WIND,
)
from hydra2.engines.riichienv._oracle_base import (
    _TRANSPARENT_KINDS as _TRANSPARENT_KINDS,
)
from hydra2.engines.riichienv._sp_capture import _capture_row as _capture_row
from hydra2.engines.riichienv._sp_oracle import _WindowOracle as _WindowOracle
from hydra2.engines.riichienv._sp_records import (
    _emit as _emit,
)
from hydra2.engines.riichienv._sp_records import (
    _event_schema_hash as _event_schema_hash,
)
from hydra2.engines.riichienv._sp_records import _expand_nonclaim_legals as _expand_nonclaim_legals
from hydra2.engines.riichienv._sp_records import _packet_boundary_hash as _packet_boundary_hash
from hydra2.engines.riichienv.events import make_delta as make_delta

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
    from hydra2.contracts.common import DigestText as DigestText
    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_records import _SimStep as _SimStep


# ---------------------------------------------------------------------------
# Per-kyoku walk (single engine + window stash).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _KyokuWalk:
    stash: dict[int, _SimStep]
    tw: _WindowOracle
    drawer: int | None = None
    last_oracle_dora: tuple[int, ...] = ()
    dora_used: set[int] = None  # type: ignore[assignment]
    dora_revealed: list[int] = None  # type: ignore[assignment]


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

    Runs pre-step: the furiten read observes the single engine state before
    this decision is stepped through it. Kakan passes ``match_offered=False``:
    the offered kakan tile is the engine's added copy while the true added
    copy is resolved separately, so membership is validated by encoding
    (ownership plus prior-pon metadata) instead.
    """
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
    :data:`~hydra2.contracts.observation_assembly.HISTORY_EVENT_CAP`); rebirthing the
    builder per hand is what the walled adapter does via ``_open_hand``.
    """
    return ObservationBuilder(
        game_id=state.sim_game_id,
        rules_id=state.rules.rules_id,
        rules_hash=cast("DigestText", _bridge_contracts.make_digest_text(state.rules_hash)),
        action_table_hash=state.table.digest,
        expected_legal_mask_length=len(state.table.actions),
        event_schema_hash=cast(
            "DigestText", _bridge_contracts.make_digest_text(_event_schema_hash())
        ),
        packet_boundary_hash=cast(
            "DigestText", _bridge_contracts.make_digest_text(_packet_boundary_hash())
        ),
    )


def _do_start_kyoku(
    state: _GameState,
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
    state.reach_accepted = 0
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
    walk = _KyokuWalk(stash={}, tw=tw)
    walk.dora_used = set()
    walk.dora_revealed = []
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
