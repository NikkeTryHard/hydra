"""Wall-less sim replay: row-construction core."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

import riichienv
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
from hydra2.contracts.action_table import ActionContext as ActionContext
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.observation_assembly import HISTORY_EVENT_CAP as HISTORY_EVENT_CAP
from hydra2.data.parquet import DecisionRow as DecisionRow
from hydra2.data.stream_decode import verify_no_privileged_leakage as verify_no_privileged_leakage
from hydra2.engines.riichienv._lr_frame import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv._oracle_base import (
    _BAKAZE_TO_WIND as _BAKAZE_TO_WIND,
)
from hydra2.engines.riichienv._oracle_base import (
    _LIVE_WALL_BASE as _LIVE_WALL_BASE,
)
from hydra2.engines.riichienv._oracle_base import _adapter_hash as _adapter_hash
from hydra2.engines.riichienv._oracle_base import _legal_mjai_type as _legal_mjai_type
from hydra2.engines.riichienv.actions import legal_view as legal_view
from hydra2.engines.riichienv.events import make_envelope as make_envelope
from hydra2.engines.riichienv.state import seat_winds_for_dealer as seat_winds_for_dealer

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder
    from hydra2.contracts.observation_types import VisibleMeld as VisibleMeld
    from hydra2.data.decode import GameRecord as GameRecord
    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep


#: Wall-less derivation marker bound into ``derivation_hash`` instead of a
#: wall digest (placeholder digests are never bound).
SIM_DERIVATION_MARK = "sim-replay-wall-less-v1"


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
        actor=_bridge_contracts.make_seat(seat),
        action_table_hash=state.table.digest,
        phase=cast("Any", phase),
        offered_tile=None if offered_tile is None else _bridge_contracts.make_tile_id(offered_tile),
        offered_by=None if offered_by is None else _bridge_contracts.make_seat(offered_by),
        own_concealed_tiles=tuple(_bridge_contracts.make_tile_id(t) for t in concealed),
        visible_melds=tuple(m for row in state.melds for m in row),
    )


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
            and tiles.mjai_string_of(int(tile_raw)) == tiles.mjai_string_of(offered_tile)
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


# ---------------------------------------------------------------------------
# Deterministic claim variants.
# ---------------------------------------------------------------------------


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
            if tiles.mjai_string_of(candidate) == pai:
                picked.append(pool.pop(index))
                break
        else:
            raise ContractError(f"no tracked copy left for meld tile {pai!r}")
    if len(picked) != needed:
        raise ContractError(f"claim needs {needed} consumed tiles, got {len(picked)}")
    if called is not None:
        for pos, tile in enumerate(picked):
            if tile == called:
                pai = tiles.mjai_string_of(tile)
                used = set(picked) | {called}
                for candidate in tiles.copies_of_string(pai):
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
    if step.drawn is not None and tiles.mjai_string_of(step.drawn) == pai:
        return step.drawn
    for tile in step.hand:
        if tiles.mjai_string_of(tile) == pai:
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
        actor=_bridge_contracts.make_seat(seat),
        tile=None,
        called_tile=_bridge_contracts.make_tile_id(called),
        consumed_tiles=tuple(_bridge_contracts.make_tile_id(t) for t in consumed),
        source_seat=_bridge_contracts.make_seat(source),
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
    state.builder.set_concealed_hand(_bridge_contracts.make_seat(seat), _concealed_for_build(step))
    state.builder.set_actor_state(
        _bridge_contracts.make_seat(seat),
        furiten=furiten,
        can_tsumo=can_tsumo,
        can_riichi=can_riichi,
    )
    try:
        observation = state.builder.build(
            actor=_bridge_contracts.make_seat(seat), legal_mask=tuple(mask)
        )
    except (ContractError, ValueError) as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation build failed: {exc}") from exc
    from hydra2.contracts.observation_assembly import VISIBILITY_VALIDATOR as _VV

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
