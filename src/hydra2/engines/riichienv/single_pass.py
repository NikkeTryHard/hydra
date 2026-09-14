"""Single-engine wall-less replay: framed MJAI games -> actor ``DecisionRow`` rows (WP-14).

Real Tenhou MJAI carries no wall field, so the engine-reset path
(:mod:`hydra2.data.replay_expand`, which injects a full 136-tile
:class:`~hydra2.engines.protocol.WallSchedule`) cannot run. This module walks
ONE live ``riichienv.RiichiEnv`` per kyoku through the logged events instead:
the staged wall deals tehais in log order and answers draws in log order (a
pure countdown; no synthesized content ever reaches a row), and every label
needing win evaluation (furiten, missed passes, claim windows, kyushu offers,
chankan windows) is queried natively from that engine at row time. There is
no second engine pass and no Python win/yaku reimplementation: the log drives
which decision happens when, the engine owns what was offered and what holds.

Row construction reuses the adapter's EXISTING conversion machinery, never a
parallel implementation:

- legal sets: the live engine's seat-filtered offer expanded through
  :func:`~hydra2.engines.riichienv.actions.expand_engine_legals` and
  :func:`~hydra2.engines.riichienv.actions.legal_view`;
- chosen actions: engine ``Action`` objects mapped to canonical form and
  encoded with the canonical codec;
- observations: assembled with the canonical
  :class:`~hydra2.contracts.observation.ObservationBuilder` from the same
  envelope constructors (:mod:`hydra2.engines.riichienv.events`) the adapter
  uses, in the same emission order;
- history: the framed log's MJAI events translated envelope-for-envelope in
  adapter order (turn/draw, discard, claims, windows, dora, wins, draws).

CRITICAL wall binding: a placeholder-wall digest is NEVER bound. ``wall_id``
is always ``None`` and the derivation carries the wall-less marker
:data:`SIM_DERIVATION_MARK` instead of a wall digest. Desyncs (riichienv
``InvalidState`` / ``Replay desync``, illegal offered actions, firewall
breaches) raise :class:`~hydra2.contracts.common.ContractError` naming
game + kyoku + step, so upstream callers quarantine and count them instead
of silently dropping rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.action import (
    ActionContext,
    CanonicalAction,
    canonical_action_codec,
    load_action_table,
)
from hydra2.contracts.common import (
    ContractError,
    IllegalActionError,
    make_digest_text,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.observation import (
    HISTORY_EVENT_CAP,
    ObservationBuilder,
    VisibleMeld,
    visible_meld_id,
)
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.parquet import DecisionRow
from hydra2.data.stream import verify_no_privileged_leakage
from hydra2.engines.riichienv.actions import legal_view
from hydra2.engines.riichienv.events import (
    make_delta,
    make_envelope,
    meld_delta_value,
    reason_kind,
)
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY
from hydra2.engines.riichienv.state import seat_winds_for_dealer
from hydra2.engines.riichienv.tiles import mjai_string_of, physical_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.rules import RulesManifest

__all__ = ["SIM_DERIVATION_MARK", "replay_game"]

#: Wall-less derivation marker bound into ``derivation_hash`` instead of a
#: wall digest (placeholder digests are never bound). Rev ``v2``: single-pass
#: rows carry live-engine offer semantics (kuikae-strict discards, complete
#: chi/kyushu offers), which differ from the ``v1`` drained-oracle masks by
#: the pinned classes only -- old/new rows never mix silently.
SIM_DERIVATION_MARK = "sim-replay-wall-less-v2"

#: Live-wall countdown base: 136 tiles minus 4x13 dealt minus the 14-tile
#: dead wall (mirrors the tehais-install countdown semantics).
_LIVE_WALL_BASE = 136 - 52 - 14

#: Bakaze letter -> round-wind TileType (same scale as seat winds, 27..30).
_BAKAZE_TO_WIND = {"E": 27, "S": 28, "W": 29, "N": 30}


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


def _adapter_hash() -> str:
    from hydra2.data import replay_expand as _re

    return _re._adapter_hash()


def _table() -> Any:
    from hydra2.config import repo_root
    from hydra2.contracts.action import ACTION_TABLE_RELPATH

    return load_action_table(Path(repo_root()) / ACTION_TABLE_RELPATH)


def _event_schema_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event import EVENT_SCHEMA_RELPATH, parse_event_schema

    document: dict[str, Any] = cast(
        "dict[str, Any]",
        parse_event_schema((Path(repo_root()) / EVENT_SCHEMA_RELPATH).read_bytes()),
    )
    payload: Any = document["payload"]
    if not isinstance(payload, dict) or "digest" not in payload:
        raise ContractError("event schema artifact lacks a digest")
    return str(cast("Any", payload["digest"]))


def _packet_boundary_hash() -> str:
    from hydra2.contracts.event import build_packet_boundary_payload, compute_event_schema_digest

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
    from hydra2.contracts.common import make_tile_id as _tid
    from hydra2.engines.protocol import wall_schedule_digest

    physical = tuple(_tid(t) for t in game.wall_tiles)
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
# Tile-copy helpers (string-canonical ids + deterministic meld rebuild).
# ---------------------------------------------------------------------------


def _copies_of_string(pai: str) -> list[int]:
    """Ordered physical copies for one MJAI string (red-aware)."""

    first = int(physical_of(pai))
    if pai in ("5mr", "0m"):
        return [16]
    if pai in ("5pr", "0p"):
        return [52]
    if pai in ("5sr", "0s"):
        return [88]
    base = (first // 4) * 4
    if first == base + 1 and pai[0] == "5":
        # Plain five: the red copy (base) belongs to the "5xr" string.
        return [base + 1, base + 2, base + 3]
    return [base, base + 1, base + 2, base + 3]


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
    return ActionContext(
        actor=make_seat(seat),
        action_table_hash=state.table.digest,
        phase=cast("Any", phase),
        offered_tile=None if offered_tile is None else make_tile_id(offered_tile),
        offered_by=None if offered_by is None else make_seat(offered_by),
        own_concealed_tiles=tuple(make_tile_id(t) for t in concealed),
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
            and mjai_string_of(int(tile_raw)) == mjai_string_of(offered_tile)
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


def _legal_mjai_type(raw: Any) -> str:
    """MJAI type string of a yielded engine action (public mapping only)."""
    try:
        mjai: Any = raw.to_mjai()
    except Exception as exc:
        raise ContractError(f"engine action without an MJAI mapping: {exc}") from exc
    if isinstance(mjai, str):
        try:
            mjai = json.loads(mjai)
        except ValueError as exc:
            raise ContractError(f"engine action MJAI unparseable: {exc}") from exc
    if not isinstance(mjai, dict) or not isinstance(mjai.get("type"), str):
        raise ContractError(f"engine action without a type string: {mjai!r}")
    return mjai["type"]


def _safe_mjai_type(raw: Any) -> str:
    """MJAI type string, or ``""`` for actions without a public mapping."""
    try:
        return _legal_mjai_type(raw)
    except (ContractError, ValueError):
        return ""


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
            rendered = mjai_string_of(tile)
        except ContractError:
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
        folded_tile = None if tile_raw is None else int(physical_of(mjai_string_of(int(tile_raw))))
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
            drawn = physical_of(mjai_string_of(pos.drawn))
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
        tile=int(physical_of(pai)),
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
        if step.drawn is None or mjai_string_of(step.drawn) != pai:
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
            and mjai_string_of(int(raw.tile)) == pai
            for raw in step.raw_legals
        ):
            raise state.fail(kyoku, "dahai", f"seat {actor} oracle discards a different tile")
        # Tile identity is the string base (the oracle renders every normal
        # discard at its string's first copy).
        tile_raw = int(physical_of(pai))
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
    logged = sorted(str(t) for t in consumed)
    try:
        offer = walk.tw._find(actor, mjai_type="ankan", consumed_strs=logged)
    except ContractError as exc:
        raise state.fail(
            kyoku, "ankan", f"seat {actor} oracle holds no ankan offer: {exc}"
        ) from exc
    step = _live_step(
        state,
        walk,
        kyoku,
        actor,
        mjai_type="ankan",
        consume=tuple(sorted(int(t) for t in offer.consume_tiles)),
    )
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
    if not any(
        _safe_mjai_type(raw) == "kakan"
        and raw.tile is not None
        and mjai_string_of(int(raw.tile)) == pai
        for raw in step.raw_legals
    ):
        raise state.fail(kyoku, "kakan", "kakan names a different tile than the oracle")
    # The added fourth copy is the one pool copy of this type absent from the
    # prior pon triple (deterministic, contract-valid, and exact on
    # min-rule-disciplined logs); meld tiles are folded, so the trick lands
    # on the same copy as the drained pipeline.
    pool = _copies_of_string(pai)
    if pai[0] == "5" and len(pai) == 2:
        base = (int(physical_of(pai)) // 4) * 4
        pool = [base, base + 1, base + 2, base + 3]
    try:
        prior = _find_prior_pon(state, actor, pool[0])
    except ContractError as exc:
        raise state.fail(kyoku, "kakan", str(exc)) from exc
    missing = [c for c in pool if c not in {int(t) for t in prior.tiles}]
    if len(missing) != 1:
        raise state.fail(kyoku, "kakan", "prior pon leaves no single added copy")
    added = missing[0]
    step = replace(step, mjai_type="kakan", tile=added)
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
    indicator = int(physical_of(marker))
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


# ---------------------------------------------------------------------------
# Wins, draws, boundaries.
# ---------------------------------------------------------------------------


def _do_hora(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    winner = _require_actor(state, kyoku, event, where="hora")
    tsumo_flag = bool(event.get("tsumo", False))
    target = event.get("target")
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
        raise state.fail(kyoku, "hora", "hora without a valid target seat")
    # Tenhou-real tsumo wins carry no flag (and no pai); target==actor marks
    # them, while ron names the discarder.
    self_draw = tsumo_flag or target == winner
    if state.decided:
        # Double ron (two winners, one discard) is log-valid but unhandled:
        # the single-winner pipeline emits one ron row and quarantines here.
        # Bias note: co-winners never contribute rows, so ron rows skew
        # slightly toward single winners (rare: single-digit games per hundred).
        # TODO(multi-ron): emit second-winner rows in the DecisionRow schema.
        if (
            not self_draw
            and state.decided_by is not None
            and state.decided_by[0] == "ron"
            and state.decided_by[1] != winner
            and state.last_discard[0] is not None
            and target == state.last_discard[0]
        ):
            raise state.fail(
                kyoku, "hora", "double ron on one discard is quarantined (single-winner pipeline)"
            )
        raise state.fail(kyoku, "hora", "win after the kyoku was decided")
    deltas_raw = event.get("deltas")
    if not isinstance(deltas_raw, (list, tuple)) or len(deltas_raw) != 4:
        raise state.fail(kyoku, "hora", "hora without a 4-seat deltas quad")
    try:
        deltas = [int(d) for d in deltas_raw]
    except (TypeError, ValueError) as exc:
        raise state.fail(kyoku, "hora", f"hora deltas malformed: {exc}") from exc
    if self_draw:
        _check_drawer(state, walk, kyoku, winner, where="hora")
        step = _live_step(state, walk, kyoku, winner, mjai_type="hora")
        if step.drawn is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        drawn_str = mjai_string_of(step.drawn)
        offer: Any = None
        for offer_kind in ("hora", "tsumo"):
            try:
                offer = walk.tw._find(winner, mjai_type=offer_kind, tile_str=drawn_str)
            except ContractError:
                continue
            break
        if offer is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        tile = step.drawn
        if offer.tile is None or mjai_string_of(int(offer.tile)) != mjai_string_of(tile):
            raise state.fail(kyoku, "hora", "tsumo tile differs from the drawn tile")
        step = replace(step, mjai_type="hora", tile=tile)
        expected = CanonicalAction(
            kind=cast("Any", "tsumo"),
            actor=make_seat(winner),
            tile=make_tile_id(tile),
            called_tile=None,
            consumed_tiles=(),
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
            winner,
            step,
            expected,
            phase="draw_decision",
            turn_actor=winner,
            offered=live_offered,
        )
        try:
            walk.tw.do_tsumo_win(winner, mjai_string_of(tile))
        except ContractError as exc:
            raise state.fail(kyoku, "hora", f"oracle win failed: {exc}") from exc
        source: int | None = None
    else:
        if state.last_discard[0] is None or state.last_discard[1] is None:
            raise state.fail(kyoku, "hora", "ron without a live discard offer")
        discarder = state.last_discard[0]
        if target != discarder:
            raise state.fail(kyoku, "hora", f"ron target {target} != discarder {discarder}")
        step = walk.stash.pop(winner, None)
        if step is None:
            raise state.fail(kyoku, "hora", f"seat {winner} holds no window step for ron")
        tile = state.last_discard[1]
        tile_str = mjai_string_of(tile)
        if not any(
            _safe_mjai_type(raw) == "hora"
            and raw.tile is not None
            and mjai_string_of(int(raw.tile)) == tile_str
            for raw in step.raw_legals
        ):
            raise state.fail(kyoku, "hora", f"seat {winner} oracle holds no hora offer")
        step = replace(step, mjai_type="hora", tile=tile)
        expected = CanonicalAction(
            kind=cast("Any", "ron"),
            actor=make_seat(winner),
            tile=make_tile_id(tile),
            called_tile=None,
            consumed_tiles=(),
            source_seat=make_seat(discarder),
            declares_riichi=False,
            metadata=(),
        )
        phase = "kan_response" if not state.opened_by_discard else "discard_response"
        chosen_id = _strict_row(
            state,
            walk,
            kyoku,
            winner,
            step,
            expected,
            phase=phase,
            turn_actor=discarder,
            offered=(tile, discarder),
        )
        source = discarder
        walk.stash.clear()
        if state.cr_pending:
            _emit_call_resolved(state, accepted=[chosen_id])
    scores = list(state.tracked_scores)
    state.tracked_scores = cast(
        "tuple[int, int, int, int]", tuple(scores[s] + deltas[s] for s in range(4))
    )
    _emit(
        state,
        kind="tsumo" if self_draw else "ron",
        visibility="public",
        actor=winner,
        tile=tile,
        action_id=chosen_id,
        source_seat=source,
        public_delta=(
            make_delta(("scores",), "set", list(deltas)),
            *(
                (make_delta(("riichi_sticks",), "increment", -step.sticks),)
                if step.sticks != 0
                else ()
            ),
        ),
    )
    state.decided = True
    state.decided_by = ("tsumo" if self_draw else "ron", winner)
    walk.drawer = None


def _do_ryukyoku(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    if state.decided:
        raise state.fail(kyoku, "ryukyoku", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="ryukyoku")
    reason = event.get("reason")
    if not isinstance(reason, str) or reason == "":
        # Tenhou-real MJAI omits the reason for wall exhaustion, first-turn
        # kyushu aborts, and four-reach aborts (the 4th reach emits without
        # its acceptance because the abort ends the kyoku first: never
        # synthesize it). Exhaustion always needs a full wall of draws;
        # kyushu is verified against the single engine's own nine-terminals
        # offer; yonin matches four declarations with three acceptances.
        # Anything between stays quarantined.
        if all(state.riichi_declared) and state.reach_accepted == 3:
            reason = "suucha_riichi"
        elif state.draws >= 60:
            reason = "exhaustive_draw"
        elif state.draws <= 4 and walk.drawer is not None and walk.tw.offers_kyushu(walk.drawer):
            reason = "kyushu_kyuhai"
        else:
            raise state.fail(
                kyoku, "ryukyoku", "draw without a reason string outside kyushu/exhaustive shape"
            )
    try:
        classification = reason_kind(reason)
    except ContractError as exc:
        raise state.fail(kyoku, "ryukyoku", str(exc)) from exc
    deltas_raw = event.get("deltas")
    if not isinstance(deltas_raw, (list, tuple)) or len(deltas_raw) != 4:
        raise state.fail(kyoku, "ryukyoku", "draw without a 4-seat deltas quad")
    try:
        deltas = [int(d) for d in deltas_raw]
    except (TypeError, ValueError) as exc:
        raise state.fail(kyoku, "ryukyoku", f"draw deltas malformed: {exc}") from exc
    pre = list(state.tracked_scores)
    post = [pre[s] + deltas[s] for s in range(4)]
    state.tracked_scores = cast("tuple[int, int, int, int]", tuple(post))
    if classification == "abortive_draw":
        from hydra2.engines.riichienv import events as _events

        try:
            mapped = _events.ABORTIVE_REASONS[reason]
        except KeyError as exc:
            raise state.fail(kyoku, "ryukyoku", f"unmapped abortive reason: {exc}") from exc
        _emit(
            state,
            kind="abortive_draw",
            visibility="public",
            round_index=state.hand_index,
            scores=post,
            reason=mapped,
            public_delta=(make_delta(("scores",), "set", list(deltas)),),
        )
    else:
        _emit(
            state,
            kind="draw_end",
            visibility="public",
            scores=post,
            reason=reason,
            public_delta=(make_delta(("scores",), "set", list(deltas)),),
        )
    state.decided = True
    walk.drawer = None


def _do_end_kyoku(
    state: _GameState,
    walk: _KyokuWalk,
    kyoku: int,
    following: dict[str, object] | None,
) -> None:
    _clear_stash_no_claim(state, walk, kyoku, where="end_kyoku")
    final = list(state.tracked_scores)
    next_round = state.hand_index + (1 if following is not None else 0)
    deltas: list[Any] = [
        make_delta(("scores",), "set", list(final)),
        make_delta(("round_index",), "set", next_round),
        *(make_delta(("riichi_states", seat), "set", "none") for seat in range(4)),
    ]
    if following is not None and str(following.get("type")) == "start_kyoku":
        try:
            deltas.append(make_delta(("honba",), "set", int(cast("Any", following["honba"]))))
            deltas.append(
                make_delta(("riichi_sticks",), "set", int(cast("Any", following["kyotaku"])))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise state.fail(kyoku, "end_kyoku", f"carry malformed: {exc}") from exc
    for seat in range(4):
        deltas.append(make_delta(("ippatsu", seat), "set", False))
    _emit(
        state,
        kind="round_end",
        visibility="public",
        round_index=state.hand_index,
        scores=final,
        public_delta=tuple(deltas),
    )
    walk.drawer = None


def _do_end_game(state: _GameState, walk: _KyokuWalk, kyoku: int) -> None:
    # The framed log always closes with end_game, but the engine path only
    # carries a game_end envelope when its own simulation went terminal (a
    # complete hanchan). Truncated wall-less streams must not invent one, so
    # the envelope is never emitted here -- only the completion flag.
    _clear_stash_no_claim(state, walk, kyoku, where="end_game")
    state.terminal = True


# ---------------------------------------------------------------------------
# Game walk + entry point.
# ---------------------------------------------------------------------------


def _walk_game(state: _GameState) -> list[DecisionRow]:
    events = list(state.game.events)
    walk: _KyokuWalk | None = None
    kyoku_ordinal = -1
    idx = 0
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        ev = event
        kind_raw = ev.get("type")
        if not isinstance(kind_raw, str) or kind_raw == "":
            raise ContractError(f"mjai event [{idx}] without a string type")
        kind = kind_raw
        if kind in _END:
            if kind == "end_game" or kind in ("endGame", "game_end", "end"):
                if walk is None or kyoku_ordinal < 0:
                    raise state.fail(-1, "end_game", "game ends before any kyoku")
                _do_end_game(state, walk, kyoku_ordinal)
            break
        if kind == "start_kyoku":
            kyoku_ordinal += 1
            walk = _do_start_kyoku(state, kyoku_ordinal, ev, events, idx)
            idx += 1
            continue
        if kind in _START:
            idx += 1
            continue
        if walk is None or kyoku_ordinal < 0:
            raise state.fail(-1, kind, "decision before the first start_kyoku")
        assert walk is not None
        if kind == "ryukyoku":
            _do_ryukyoku(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind == "end_kyoku":
            following_ev: dict[str, object] | None = None
            if idx + 1 < total:
                candidate = events[idx + 1]
                if isinstance(candidate, dict):
                    following_ev = candidate
            _do_end_kyoku(state, walk, kyoku_ordinal, following_ev)
            idx += 1
            continue
        if kind == "tsumo":
            _do_tsumo(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind == "dora":
            _do_dora(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind == "reach_accepted":
            _do_reach_accepted(state, walk, kyoku_ordinal, ev)
            idx += 1
            continue
        if kind in _SKIP:
            idx += 1
            continue
        if kind == "reach":
            nxt = idx + 1
            while nxt < total:
                following = events[nxt]
                if not isinstance(following, dict):
                    raise ContractError(f"mjai event [{nxt}] must be an object")
                following_kind = following.get("type")
                if following_kind in _SKIP:
                    nxt += 1
                    continue
                break
            else:
                raise state.fail(kyoku_ordinal, "reach", "declaration missing")
            declaration = events[nxt]
            if declaration.get("type") != "dahai":
                raise state.fail(kyoku_ordinal, "reach", "declaration is not a dahai")
            decl_actor = _require_actor(state, kyoku_ordinal, declaration, where="dahai")
            claim_after = _peek_window_claim(
                cast("Sequence[dict[str, object]]", events), nxt + 1, decl_actor
            )
            _do_reach(state, walk, kyoku_ordinal, ev, declaration, claim_after)
            idx = nxt + 1
            continue
        if kind in _ROW:
            if kind in _CLAIM:
                _do_claim(state, walk, kyoku_ordinal, ev, kind)
            elif kind in ("ankan", "kakan"):
                if kind == "ankan":
                    _do_ankan(state, walk, kyoku_ordinal, ev)
                else:
                    actor = _require_actor(state, kyoku_ordinal, ev, where="kakan")
                    claim_after = _peek_window_claim(
                        cast("Sequence[dict[str, object]]", events), idx + 1, actor
                    )
                    _do_kakan(state, walk, kyoku_ordinal, ev, claim_after)
            elif kind == "hora":
                _do_hora(state, walk, kyoku_ordinal, ev)
            else:  # dahai
                actor = _require_actor(state, kyoku_ordinal, ev, where="dahai")
                claim_after = _peek_window_claim(
                    cast("Sequence[dict[str, object]]", events), idx + 1, actor
                )
                _do_dahai(state, walk, kyoku_ordinal, ev, claim_after)
            idx += 1
            continue
        raise state.fail(kyoku_ordinal, kind, f"unmapped mjai event type {kind!r}")
    if not state.terminal:
        raise state.fail(kyoku_ordinal, "walk", "game never reached end_game")
    return state.rows


def replay_game(
    game: GameRecord | bytes | str | Path, *, split: str = "train", seat: int | None = None
) -> list[DecisionRow]:
    """Replay one framed MJAI game into actor ``DecisionRow`` rows.

    ``game`` is a :class:`~hydra2.data.decode.GameRecord` (as carried by
    :class:`~hydra2.data.stream.StreamGame`), framed game bytes, or a path to
    framed game bytes. Bytes/paths are decoded strictly first. One live
    ``riichienv.RiichiEnv`` per kyoku walks the logged events (staged wall in
    log order, native furiten/window/legal queries); ``split`` rides the rows
    opaquely (non-empty).
    ``seat`` optionally restricts emitted rows to one actor (decision ids stay
    positional over every decision so privileged joins still align).

    Wall-less provenance: ``wall_id`` is never bound (always ``None``); the
    derivation carries :data:`SIM_DERIVATION_MARK`. Any desync, illegal
    offered action, or firewall breach raises :class:`ContractError` naming
    game + kyoku + step.
    """
    record = _coerce_game(game)
    if split == "":
        raise ContractError("split must be a non-empty string")
    if seat is not None and (isinstance(seat, bool) or not isinstance(seat, int)):
        raise ContractError(f"seat must be a seat int or None, got {seat!r}")
    if seat is not None and not 0 <= seat <= 3:
        raise ContractError(f"seat must be 0..3, got {seat!r}")
    manifest = _rules()
    from hydra2.engines.riichienv.state import rules_identity_hash

    rules_hash = _rules_hash(manifest, str(rules_identity_hash(manifest)))
    table = _table()
    sim_game_id = _sim_game_id(record, rules_hash=rules_hash)
    builder = ObservationBuilder(
        game_id=sim_game_id,
        rules_id=manifest.rules_id,
        rules_hash=make_digest_text(rules_hash),
        action_table_hash=table.digest,
        expected_legal_mask_length=len(table.actions),
        event_schema_hash=make_digest_text(_event_schema_hash()),
        packet_boundary_hash=make_digest_text(_packet_boundary_hash()),
    )
    state = _GameState(
        game=record,
        split=split,
        seat_filter=seat,
        rules=manifest,
        rules_hash=rules_hash,
        table=table,
        sim_game_id=sim_game_id,
        builder=builder,
    )
    state.rows = []
    state.melds = ([], [], [], [])
    # Tenhou emits a bare {"type": "dora"} with no marker on kan-dora. Own the
    # reason code here instead: the indicator is unrecoverable from the log.
    kyoku_idx = -1
    pending_ron: tuple[int, int] | None = None
    for event in record.events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "start_kyoku":
            kyoku_idx += 1
            pending_ron = None
        elif kind == "dora" and (
            not isinstance(event.get("dora_marker"), str) or event.get("dora_marker") == ""
        ):
            raise state.fail(
                kyoku_idx,
                "dora",
                "kan-dora indicator unrecoverable: dora event without a dora_marker",
            )
        elif kind == "hora":
            actor = event.get("actor")
            target = event.get("target")
            if (
                not bool(event.get("tsumo", False))
                and isinstance(actor, int)
                and not isinstance(actor, bool)
                and 0 <= actor <= 3
                and isinstance(target, int)
                and not isinstance(target, bool)
                and 0 <= target <= 3
                and target != actor
            ):
                if pending_ron is not None and pending_ron[1] == target and pending_ron[0] != actor:
                    raise state.fail(
                        kyoku_idx,
                        "hora",
                        "double ron on one discard is quarantined (single-winner pipeline)",
                    )
                pending_ron = (actor, target)
            else:
                pending_ron = None
        elif kind in _TRANSPARENT_KINDS:
            pass
        else:
            pending_ron = None
    rows = _walk_game(state)
    return rows


# ---------------------------------------------------------------------------
# Raw-engine window/furiten oracle.
# ---------------------------------------------------------------------------

#: Log event kinds skipped when looking behind/ahead through a kyoku.
_TRANSPARENT_KINDS = frozenset({"dora", "reach_accepted"})


def _peek_window_claim(
    events: Sequence[dict[str, object]], start: int, discarder: int
) -> dict[str, object] | None:
    """Next window claim on ``discarder``'s tile, if the log shows one.

    Scans forward past transparent events; a draw decision, another offer, a
    terminal, or any boundary ends the window with no claim.
    """
    idx = start
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = str(event.get("type", ""))
        if kind in _TRANSPARENT_KINDS:
            idx += 1
            continue
        if kind in ("chi", "pon", "daiminkan"):
            target = event.get("target")
            if target == discarder:
                return event
            return None
        if kind == "hora" and not bool(event.get("tsumo", False)):
            # Tenhou-real tsumo wins carry no flag; target==actor marks them
            # and they never open a discard window.
            if event.get("target") == event.get("actor"):
                return None
            if event.get("target") == discarder:
                return event
            return None
        return None
    return None


class _WindowOracle:
    """Live raw engine: the single pass's wall, offers, and labels.

    One pinned ``riichienv.RiichiEnv`` per kyoku, reset with a countdown wall
    built from the log itself (tehais in deal order, logged tsumo in draw
    order with rinshan tiles at the dead-wall tail, filler elsewhere). The
    wall is purely mechanical and its digest is never bound anywhere. The
    log's events are translated to raw steps with string-level matching; any
    mismatch fails closed (quarantine upstream), never approximated.

    Answered queries (the ONLY surfaces consumed):

    - ``window_open()``: the engine's own claim-window predicate (phase plus
      responder legals) after a discard or kakan step;
    - ``furiten(pid)``: :func:`furiten_of` on the live engine at row time,
      plus manually recorded ron passes;
    - ``position(pid)``: frozen live-table snapshot feeding row sources;
    - ``offers_kyushu(pid)``: the engine's own nine-terminals offer.

    Wins, draws, dora reveals, and reach acceptances are never translated:
    the kyoku is decided by then and no further query can occur before the
    next reset.
    """

    def __init__(self, *, game_id: str) -> None:
        self._game_id = game_id
        self._env: Any = None
        self._kyoku = -1
        self._missed: set[int] = set()

    def _fail(self, why: str) -> ContractError:
        return ContractError(f"sim replay desync game {self._game_id!r} kyoku {self._kyoku}: {why}")

    def reset_kyoku(
        self,
        *,
        ordinal: int,
        oya: int,
        scores: tuple[int, int, int, int],
        honba: int,
        kyotaku: int,
        bakaze: str,
        tehais: tuple[Any, ...],
        live_draws: list[str],
        rinshan_draws: list[str],
    ) -> None:
        self._kyoku = ordinal
        self._missed = set()
        if len(tehais) != 4:
            raise self._fail(f"tehais cover {len(tehais)} seats, not 4")
        # Distinct physical copies per occurrence (the engine's multiset
        # accounting assumes unique tiles; collapsed ids corrupt it). Tile
        # conservation bounds every string to its four copies -- exhausting a
        # pool means an impossible log and fails closed.
        pools: dict[str, list[int]] = {}
        taken: dict[str, int] = {}

        def take(pai: str) -> int:
            if pai not in pools:
                first = int(physical_of(pai))
                if pai in ("5mr", "0m"):
                    pools[pai] = [16]
                elif pai in ("5pr", "0p"):
                    pools[pai] = [52]
                elif pai in ("5sr", "0s"):
                    pools[pai] = [88]
                else:
                    base = (first // 4) * 4
                    ids = [base, base + 1, base + 2, base + 3]
                    if len(pai) == 2 and pai[0] == "5":
                        ids = [i for i in ids if i != base]
                    pools[pai] = ids
            cursor = taken.get(pai, 0)
            if cursor >= len(pools[pai]):
                raise self._fail(f"tile string {pai!r} overused (tile conservation)")
            taken[pai] = cursor + 1
            return pools[pai][cursor]

        wall: list[int] = [-1] * 136
        # Deal order rotates with the dealer: block b feeds seat
        # (oya + b) mod 4, so seat s takes blocks congruent to (s - oya).
        for seat in range(4):
            hand = list(tehais[seat])
            if len(hand) != 13:
                raise self._fail(f"seat {seat} tehais hold {len(hand)} tiles, not 13")
            rel = (seat - oya) % 4
            for j in range(12):
                k, m = divmod(j, 4)
                wall[4 * (rel + 4 * k) + m] = take(str(hand[j]))
            wall[48 + rel] = take(str(hand[12]))
        for i, pai in enumerate(live_draws):
            wall[52 + i] = take(pai)
        for i, pai in enumerate(rinshan_draws):
            wall[135 - i] = take(pai)
        used = {t for t in wall if t != -1}
        filler = (t for t in range(136) if t not in used)
        wall = [t if t != -1 else next(filler) for t in wall]
        wind = {"E": 0, "S": 1, "W": 2, "N": 3}[bakaze]
        env = riichienv.RiichiEnv(
            game_mode=riichienv.GameType.YON_HANCHAN,
            rule=riichienv.GameRule.default_tenhou(),
            seed=ordinal,
        )
        try:
            _ = env.reset(
                oya=oya,
                wall=wall,
                scores=list(scores),
                honba=honba,
                kyotaku=kyotaku,
                round_wind=wind,
            )  # discard initial obs; reset side-effect installs the position
        except Exception as exc:
            raise self._fail(f"engine reset failed: {exc}") from exc
        self._env = env

    @property
    def _engine(self) -> Any:
        if self._env is None:
            raise self._fail("oracle engine used before reset")
        return self._env

    def _legals(self, pid: int) -> list[Any]:
        try:
            return list(self._engine.get_observation(pid).legal_actions())
        except Exception as exc:
            raise self._fail(f"seat {pid} legal query failed: {exc}") from exc

    @staticmethod
    def _mjai(raw: Any) -> dict[str, Any]:
        mjai: Any = raw.to_mjai()
        if isinstance(mjai, str):
            mjai = json.loads(mjai)
        if not isinstance(mjai, dict):
            raise ContractError(f"engine action without an MJAI mapping: {mjai!r}")
        return dict(mjai)

    def _find(
        self,
        pid: int,
        *,
        mjai_type: str,
        tile_str: str | None = None,
        consumed_strs: Sequence[str] | None = None,
    ) -> Any:
        matches: list[Any] = []
        for raw in self._legals(pid):
            try:
                mjai = self._mjai(raw)
            except (ContractError, ValueError):
                continue
            if str(mjai.get("type", "")) != mjai_type:
                continue
            if tile_str is not None:
                tile_raw: Any = raw.tile
                if tile_raw is None or mjai_string_of(int(tile_raw)) != tile_str:
                    continue
            if consumed_strs is not None:
                got = sorted(mjai_string_of(int(t)) for t in raw.consume_tiles)
                if got != sorted(consumed_strs):
                    continue
            matches.append(raw)
        if len(matches) == 0:
            raise self._fail(f"seat {pid} has no {mjai_type} offer for {tile_str!r}")
        return matches[0]

    def _step(self, moves: dict[int, Any], *, where: str) -> None:
        try:
            self._engine.step(moves)
        except Exception as exc:
            raise self._fail(f"engine step failed at {where}: {exc}") from exc

    def note_tsumo(self, actor: int, pai: str) -> None:
        """Assert-or-draw the logged tsumo (rinshan draws included)."""
        self._missed.discard(actor)
        drawn_raw: Any = self._engine.drawn_tile
        if drawn_raw is not None and mjai_string_of(int(drawn_raw)) == pai:
            self._confirm_draw_holder(actor, pai, drawn_raw)
            return
        if drawn_raw is not None:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._step({}, where=f"tsumo seat {actor}")
        drawn_raw = self._engine.drawn_tile
        if drawn_raw is None or mjai_string_of(int(drawn_raw)) != pai:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._confirm_draw_holder(actor, pai, drawn_raw)
        return

    def _confirm_draw_holder(self, actor: int, pai: str, drawn_raw: Any) -> None:
        """Fail closed when the drawn tile sits in another seat's hand.

        The engine auto-draws the next seat on discard steps while draws
        match by string, so an out-of-turn log tsumo can string-match a tile
        the engine dealt elsewhere; physical-id membership names the true
        drawer at the tsumo instead of desyncing a later row.
        """
        try:
            hands: Any = self._engine.hands
            owned = int(drawn_raw) in {int(t) for t in hands[actor]}
        except Exception as exc:
            raise self._fail(f"seat {actor} drawer query failed: {exc}") from exc
        if not owned:
            raise self._fail(
                f"seat {actor} tsumo {pai!r} sits in another seat's hand (out-of-turn draw)"
            )

    def do_dahai(self, actor: int, pai: str) -> None:
        """Step one logged discard (tsumogiri and tedashi share a slot)."""
        action = self._find(actor, mjai_type="dahai", tile_str=pai)
        self._step({actor: action}, where=f"dahai seat {actor}")

    def do_reach(self, actor: int) -> None:
        """Step the riichi declaration (the discard follows separately)."""
        action = self._find(actor, mjai_type="reach")
        self._step({actor: action}, where=f"reach seat {actor}")

    def do_ankan(self, actor: int, consumed: Sequence[str]) -> None:
        """Step one logged closed kan."""
        action = self._find(actor, mjai_type="ankan", consumed_strs=list(consumed))
        self._step({actor: action}, where=f"ankan seat {actor}")

    def do_kakan(self, actor: int, pai: str) -> None:
        """Step one logged added kan."""
        action = self._find(actor, mjai_type="kakan", tile_str=pai)
        self._step({actor: action}, where=f"kakan seat {actor}")

    def do_tsumo_win(self, actor: int, pai: str) -> None:
        """Step one logged self-draw win."""
        for mjai_type in ("hora", "tsumo"):
            try:
                action = self._find(actor, mjai_type=mjai_type, tile_str=pai)
            except ContractError:
                continue
            self._step({actor: action}, where=f"tsumo seat {actor}")
            return
        raise self._fail(f"seat {actor} has no tsumo win offer for {pai!r}")

    def window_open(self, discarder: int) -> bool:
        """The engine's own claim-window predicate after a discard/kan step."""
        env = self._engine
        try:
            waiting = int(env.phase) == int(riichienv.Phase.WaitResponse)
        except Exception as exc:
            raise self._fail(f"phase query failed: {exc}") from exc
        if not waiting:
            return False
        for pid in range(4):
            if pid == discarder:
                continue
            if len(self._legals(pid)) > 0:
                return True
        return False

    def resolve_window(self, discarder: int, claim: dict[str, object] | None) -> None:
        """Step one combined window resolution (claim or all-pass).

        Ron windows are never stepped: winning terminates the engine into an
        auto-dealt next hand (all subsequent queries would read garbage), and
        the kyoku is decided by the logged hora anyway. Responders passing a
        genuine ron offer are recorded as temporarily furiten manually; the
        winner is excluded (they won, not passed).
        """
        if claim is not None and str(claim.get("type", "")) == "hora":
            actor_raw = claim.get("actor")
            winner = None if isinstance(actor_raw, bool) else actor_raw
            for pid in range(4):
                if pid in (discarder, winner):
                    continue
                try:
                    legals = self._legals(pid)
                except ContractError:
                    continue
                for raw in legals:
                    try:
                        mjai = self._mjai(raw)
                    except (ContractError, ValueError):
                        continue
                    if str(mjai.get("type", "")) in ("hora", "ron"):
                        self._missed.add(pid)
                        break
            return
        moves: dict[int, Any] = {}
        if claim is not None:
            kind = str(claim.get("type", ""))
            actor_raw = claim.get("actor")
            if isinstance(actor_raw, bool) or not isinstance(actor_raw, int):
                raise self._fail("window claim without an actor")
            actor = actor_raw
            if kind not in ("chi", "pon", "daiminkan"):
                raise self._fail(f"window claim of unexpected kind {kind!r}")
            pai = str(claim.get("pai", ""))
            consumed = [str(t) for t in cast("Any", claim.get("consumed", []))]
            moves[actor] = self._find(actor, mjai_type=kind, tile_str=pai, consumed_strs=consumed)
        for pid in range(4):
            if pid == discarder or pid in moves:
                continue
            if len(self._legals(pid)) == 0:
                continue
            moves[pid] = self._find(pid, mjai_type="none")
        if len(moves) == 0:
            return
        self._step(moves, where="window resolution")

    def furiten(self, pid: int) -> str:
        """Exact furiten state from the live engine flags plus ron passes.

        Ron windows are never stepped (winning would terminate the engine),
        so responders passing a genuine ron offer are recorded manually; the
        engine's own doujun/riichi flags cover everything else.
        """
        from hydra2.engines.riichienv.state import furiten_of

        try:
            state = furiten_of(self._engine, pid)
        except Exception as exc:
            raise self._fail(f"seat {pid} furiten query failed: {exc}") from exc
        if state == "none" and pid in self._missed:
            return "temporary"
        return state

    def offers_kyushu(self, pid: int) -> bool:
        """Whether the live engine offers the nine-terminals abort to ``pid``."""
        for raw in self._legals(pid):
            try:
                action_type = int(cast("Any", raw.action_type))
            except (TypeError, ValueError):
                continue
            if action_type == int(riichienv.ActionType.KYUSHU_KYUHAI):
                return True
        return False

    def position(self, pid: int) -> _TablePosition:
        """Frozen live-table snapshot for engine-sourced rows."""
        env = self._engine
        try:
            hands_raw: Any = env.hands
            rivers_raw: Any = env.discards
            drawn_raw: Any = env.drawn_tile
            dora_raw: Any = env.dora_indicators
            sticks_raw: Any = env.riichi_sticks
            declared_raw: Any = env.riichi_declared
            scores_raw: Any = env.scores()
            legals = list(self._legals(pid))
            hand = tuple(int(t) for t in hands_raw[pid])
            rivers = tuple(tuple(int(t) for t in river) for river in rivers_raw)
            lens = tuple(len(cast("Any", item)) for item in hands_raw)
            dora = tuple(int(t) for t in dora_raw)
            sticks = int(sticks_raw)
            declared = tuple(bool(v) for v in declared_raw)
            scores = tuple(int(s) for s in scores_raw)
            drawn = None if drawn_raw is None else int(drawn_raw)
        except ContractError:
            raise
        except Exception as exc:
            raise self._fail(f"seat {pid} position query failed: {exc}") from exc
        return _TablePosition(
            hand=hand,
            drawn=drawn,
            dora=dora,
            rivers=rivers,
            lens=cast("tuple[int, int, int, int]", lens),
            sticks=sticks,
            declared=cast("tuple[bool, bool, bool, bool]", declared),
            scores=cast("tuple[int, int, int, int]", scores),
            legals=tuple(legals),
        )
