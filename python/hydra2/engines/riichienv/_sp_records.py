"""Single-engine wall-less replay: records and emission."""

from __future__ import annotations

from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

import riichienv

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import tiles  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.action_artifact import load_action_table as load_action_table
from hydra2.contracts.action_table import ActionContext as ActionContext
from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.decode import (
    GameRecord as GameRecord,
)
from hydra2.data.decode import (
    decode_game_object as decode_game_object,
)
from hydra2.engines.riichienv._oracle_base import (
    _BAKAZE_TO_WIND as _BAKAZE_TO_WIND,
)
from hydra2.engines.riichienv._oracle_base import (
    _LIVE_WALL_BASE as _LIVE_WALL_BASE,
)
from hydra2.engines.riichienv._oracle_base import _legal_mjai_type as _legal_mjai_type
from hydra2.engines.riichienv.actions import legal_view as legal_view
from hydra2.engines.riichienv.events import make_envelope as make_envelope
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY as ENGINE_IDENTITY
from hydra2.engines.riichienv.state import seat_winds_for_dealer as seat_winds_for_dealer

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
    from hydra2.contracts.action_table import ActionTable as ActionTable
    from hydra2.contracts.common import Seat as Seat
    from hydra2.contracts.common import TileId as TileId
    from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder
    from hydra2.contracts.observation_types import VisibleMeld as VisibleMeld
    from hydra2.contracts.rules_manifest import RulesManifest as RulesManifest
    from hydra2.data.parquet import DecisionRow as DecisionRow


#: Wall-less derivation marker bound into ``derivation_hash`` instead of a
#: wall digest (placeholder digests are never bound). Rev ``v2``: single-pass
#: rows carry live-engine offer semantics (kuikae-strict discards, complete
#: chi/kyushu offers), which differ from the ``v1`` drained-oracle masks by
#: the pinned classes only -- old/new rows never mix silently.
SIM_DERIVATION_MARK = "sim-replay-wall-less-v2"


#: MJAI framing vocabulary: the single shared source is replay_expand's
#: closed sets (do not fork a second vocabulary here). Bound once at import
#: (module constants, UPPER by convention).
def _vocabulary() -> tuple[
    frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]
]:
    from hydra2.data import replay_expand as _re

    return (_re._START_TYPES, _re._END_TYPES, _re._SKIP_TYPES, _re._ROW_TYPES, _re._CLAIM_TYPES)


_START, _END, _SKIP, _ROW, _CLAIM = _vocabulary()


def _rules() -> RulesManifest:
    from hydra2.data import replay_expand as _re

    return _re._load_rules()


def _table() -> ActionTable:
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import ACTION_TABLE_RELPATH

    return load_action_table(Path(repo_root()) / ACTION_TABLE_RELPATH)


def _event_schema_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event_schema import EVENT_SCHEMA_RELPATH, parse_event_schema

    document: dict[str, Any] = cast(
        "dict[str, Any]",
        parse_event_schema((Path(repo_root()) / EVENT_SCHEMA_RELPATH).read_bytes()),
    )
    payload: Any = document["payload"]
    if not isinstance(payload, dict) or "digest" not in payload:
        raise ContractError("event schema artifact lacks a digest")
    return str(cast("Any", payload["digest"]))


def _packet_boundary_hash() -> str:
    from hydra2.contracts.event_packet import build_packet_boundary_payload
    from hydra2.contracts.event_schema import (
        compute_event_schema_digest as compute_event_schema_digest,
    )

    return str(compute_event_schema_digest(build_packet_boundary_payload()))


def _rules_hash(manifest: RulesManifest, recomputed: str) -> str:
    """Published rules bytes win when present (same authority as the adapter)."""
    import hashlib

    from hydra2.config import repo_root

    published = Path(repo_root()) / "configs" / "rules" / f"{manifest.rules_id}.json"
    if published.is_file():
        return "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
    return recomputed


def _sim_game_id(game: GameRecord, *, rules_hash: str) -> str:
    """Builder game id with the engine path's exact scheme when walled.

    A walled game replays row-identical to :func:`expand_game`, so the
    builder identity (wall-derived seed material) is reproduced exactly from
    the real wall. A wall-less game has no wall to bind -- the log's game id
    is carried instead (never a placeholder digest).
    """
    if game.wall_tiles is None:
        if game.game_id == "":
            raise ContractError("wall-less game has no game_id for sim replay")
        return game.game_id
    if len(game.wall_tiles) != 136:
        raise ContractError(f"wall_tiles must carry 136 tiles, got {len(game.wall_tiles)}")
    from hydra2.engines.protocol import wall_schedule_digest

    physical = tuple(_bridge_contracts.make_tile_id(t) for t in game.wall_tiles)
    schedule_id = f"replay-{game.game_id}"
    wall_digest = str(wall_schedule_digest(schedule_id, physical))
    seed_material = of_canonical(
        {
            "rules_hash": rules_hash,
            "wall_digest": wall_digest,
            "seat_permutation": [0, 1, 2, 3],
            "adapter_version": str(ENGINE_IDENTITY.adapter_version),
        }
    )
    return f"hydra2-riichienv-{str(seed_material).removeprefix('sha256:')[:16]}"


# ---------------------------------------------------------------------------
# Oracle records (frozen primitives extracted eagerly per yielded step).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SimStep:
    """One live-engine decision point as frozen primitives (row source)."""

    seat: int
    mjai_type: str
    tile: int | None
    consume: tuple[int, ...]
    mjai: dict[str, object]
    raw_action: Any
    raw_legals: tuple[Any, ...]
    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    discards: tuple[tuple[int, ...], ...]
    hands_lens: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    sticks: int
    oya: int
    honba: int
    round_wind: int
    player_id: int
    riichi: tuple[bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class _TablePosition:
    """Frozen live-engine snapshot: the single engine's state at a decision."""

    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    rivers: tuple[tuple[int, ...], ...]
    lens: tuple[int, int, int, int]
    sticks: int
    declared: tuple[bool, bool, bool, bool]
    scores: tuple[int, int, int, int]
    legals: tuple[Any, ...]


def _coerce_game(game: GameRecord | bytes | str | Path) -> GameRecord:
    if isinstance(game, GameRecord):
        return game
    if isinstance(game, (str, Path)):
        raw = Path(game).read_bytes()
        object_id = f"simreplay-path-{Path(game).name}"
    else:
        raw = game
        import hashlib

        object_id = "simreplay-bytes-" + hashlib.sha256(raw).hexdigest()[:16]
    return decode_game_object(object_id=object_id, packaged_object_id=object_id, decoded_bytes=raw)


# ---------------------------------------------------------------------------
# Per-game replay state.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _GameState:
    game: GameRecord
    split: str
    seat_filter: int | None
    rules: RulesManifest
    rules_hash: str
    table: ActionTable
    sim_game_id: str
    builder: ObservationBuilder
    rows: list[DecisionRow] = None  # type: ignore[assignment]
    seq: int = 0
    event_count: int = 0
    hand_index: int = -1
    round_idx: int = -1
    draws: int = 0
    declarations: int = 0
    reach_accepted: int = 0
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
    seat_bound: Seat = _bridge_contracts.make_seat(seat)
    offered_tile_bound: TileId | None = (
        None if offered_tile is None else _bridge_contracts.make_tile_id(offered_tile)
    )
    offered_by_bound: Seat | None = (
        None if offered_by is None else _bridge_contracts.make_seat(offered_by)
    )
    return ActionContext(
        actor=seat_bound,
        action_table_hash=state.table.digest,
        phase=cast("Any", phase),
        offered_tile=offered_tile_bound,
        offered_by=offered_by_bound,
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
            action_type_raw: Any = raw.action_type
            is_ron = int(action_type_raw) == int(riichienv.ActionType.RON)
        except (TypeError, ValueError):
            is_ron = False
        tile_int: int | None = None if tile_raw is None else int(cast("Any", tile_raw))
        if (
            offered_tile is not None
            and tile_int is not None
            and is_ron
            and tiles.mjai_string_of(tile_int) == tiles.mjai_string_of(offered_tile)
            and tile_int != offered_tile
        ):
            raw_consume: list[int] = list(raw.consume_tiles)  # pyrefly: ignore[unknown-argument-type] # engine action consume tiles dynamic
            raw_type: Any = raw.action_type
            raw = SimpleNamespace(
                action_type=raw_type,
                tile=offered_tile,
                consume_tiles=tuple(raw_consume),
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


def _safe_mjai_type(raw: Any) -> str:
    """MJAI type string, or ``""`` for actions without a public mapping."""
    try:
        return _legal_mjai_type(raw)
    except (ContractError, ValueError):
        return ""
