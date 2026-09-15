"""Single-engine wall-less replay: claim variants, capture."""

from __future__ import annotations

from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.contracts.observation import HISTORY_EVENT_CAP as HISTORY_EVENT_CAP
from hydra2.data.parquet import DecisionRow as DecisionRow
from hydra2.data.stream import verify_no_privileged_leakage as verify_no_privileged_leakage
from hydra2.engines.riichienv._oracle_base import _adapter_hash as _adapter_hash
from hydra2.engines.riichienv._sp_records import SIM_DERIVATION_MARK as SIM_DERIVATION_MARK
from hydra2.engines.riichienv._sp_records import _copies_of_string as _copies_of_string
from hydra2.engines.riichienv._sp_records import _snapshot_at_row as _snapshot_at_row
from hydra2.engines.riichienv.tiles import mjai_string_of as mjai_string_of

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.engines.riichienv._sp_records import _GameState as _GameState
    from hydra2.engines.riichienv._sp_records import _SimStep as _SimStep


# ---------------------------------------------------------------------------
# Deterministic claim variants.
# ---------------------------------------------------------------------------


def _distinct_copies(ids: tuple[int, ...]) -> tuple[int, ...]:
    """Fold ids to per-string occurrence pools (the oracle's rendering rule).

    Ordering copies per string preserves the exact string multiset, so
    string-level agreement is untouched and ownership sees tile-valid ids.
    Red fives keep their string (``5mr``/``5m`` pools stay disjoint).
    Overused strings keep the verbatim id and fail closed downstream.
    """
    counts: dict[str, int] = {}
    out: list[int] = []
    for tile in ids:
        pai = mjai_string_of(tile)
        pool = _copies_of_string(pai)
        seen = counts.get(pai, 0)
        out.append(pool[seen] if seen < len(pool) else tile)
        counts[pai] = seen + 1
    return tuple(out)


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
            if mjai_string_of(candidate) == pai:
                picked.append(pool.pop(index))
                break
        else:
            raise ContractError(f"no tracked copy left for meld tile {pai!r}")
    if len(picked) != needed:
        raise ContractError(f"claim needs {needed} consumed tiles, got {len(picked)}")
    if called is not None:
        for pos, tile in enumerate(picked):
            if tile == called:
                pai = mjai_string_of(tile)
                used = set(picked) | {called}
                for candidate in _copies_of_string(pai):
                    if candidate not in used:
                        picked[pos] = candidate
                        used.add(candidate)
                        break
                else:
                    raise ContractError(f"no distinct copy left for meld tile {pai!r}")
    return tuple(sorted(picked))


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
        actor=make_seat(seat),
        tile=None,
        called_tile=make_tile_id(called),
        consumed_tiles=tuple(make_tile_id(t) for t in consumed),
        source_seat=make_seat(source),
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
    state.builder.set_concealed_hand(make_seat(seat), _concealed_for_build(step))
    state.builder.set_actor_state(
        make_seat(seat), furiten=furiten, can_tsumo=can_tsumo, can_riichi=can_riichi
    )
    try:
        observation = state.builder.build(actor=make_seat(seat), legal_mask=tuple(mask))
    except ContractError as exc:
        raise state.fail(kyoku, f"seat {seat} row", f"observation build failed: {exc}") from exc
    from hydra2.contracts.observation import VISIBILITY_VALIDATOR as _VV

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
