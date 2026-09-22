"""Single-engine wall-less replay: claim variants, capture."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import tiles  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.observation_assembly import HISTORY_EVENT_CAP as HISTORY_EVENT_CAP
from hydra2.data.parquet import DecisionRow as DecisionRow
from hydra2.data.stream_decode import verify_no_privileged_leakage as verify_no_privileged_leakage
from hydra2.engines.riichienv._oracle_base import _adapter_hash as _adapter_hash
from hydra2.engines.riichienv._sp_records import (
    SIM_DERIVATION_MARK as SIM_DERIVATION_MARK,
)
from hydra2.engines.riichienv._sp_records import _snapshot_at_row as _snapshot_at_row

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.common import Seat as Seat
    from hydra2.contracts.common import TileId as TileId
    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_records import _SimStep as _SimStep


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
            rendered: str = tiles.mjai_string_of(candidate)
            if rendered == pai:
                picked.append(pool.pop(index))
                break
        else:
            raise ContractError(f"no tracked copy left for meld tile {pai!r}")
    if len(picked) != needed:
        raise ContractError(f"claim needs {needed} consumed tiles, got {len(picked)}")
    if called is not None:
        for pos, tile in enumerate(picked):
            if tile == called:
                swap_pai: str = tiles.mjai_string_of(tile)
                used = set(picked) | {called}
                pool_copies: list[int] = tiles.copies_of_string(swap_pai)
                for candidate in pool_copies:
                    if candidate not in used:
                        picked[pos] = candidate
                        used.add(candidate)
                        break
                else:
                    raise ContractError(f"no distinct copy left for meld tile {swap_pai!r}")
    return tuple(sorted(picked))


def _claim_canonical(
    *,
    kind: str,
    seat: int,
    called: int,
    consumed: tuple[int, ...],
    source: int,
) -> CanonicalAction:
    seat_bound: Seat = _bridge_contracts.make_seat(seat)
    called_bound: TileId = _bridge_contracts.make_tile_id(called)
    source_bound: Seat = _bridge_contracts.make_seat(source)
    return CanonicalAction(
        kind=cast("Any", kind),
        actor=seat_bound,
        tile=None,
        called_tile=called_bound,
        consumed_tiles=tuple(_bridge_contracts.make_tile_id(t) for t in consumed),
        source_seat=source_bound,
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
    seat_bound: Seat = _bridge_contracts.make_seat(seat)
    state.builder.set_concealed_hand(seat_bound, _concealed_for_build(step))
    state.builder.set_actor_state(
        seat_bound,
        furiten=furiten,
        can_tsumo=can_tsumo,
        can_riichi=can_riichi,
    )
    try:
        observation = state.builder.build(actor=seat_bound, legal_mask=tuple(mask))
    except (ContractError, ValueError) as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation build failed: {exc}") from exc
    from hydra2.contracts.observation_assembly import VISIBILITY_VALIDATOR as _VV

    try:
        _VV.validate_observation(observation)
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation rejected: {exc}") from exc
    doc = observation.to_json()
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
