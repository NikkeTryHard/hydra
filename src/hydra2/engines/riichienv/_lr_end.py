"""Wall-less sim replay: framed MJAI games -> actor ``DecisionRow`` rows (WP-14).

Real Tenhou MJAI carries no wall field, so the engine-reset path
(:mod:`hydra2.data.replay_expand`, which injects a full 136-tile
:class:`~hydra2.engines.protocol.WallSchedule`) cannot run. This module
replays through the PINNED simulator's own wall-less replay instead:
``riichienv.MjaiReplay`` installs ``start_kyoku`` tehais directly and treats
its RNG-shuffled wall as a pure countdown (hydra1 precedent
``event_handler.rs`` StartKyoku: hands installed, ``tile_count = 136 - 52``,
no 136-tile wall; actor-masked observe per ``state/mod.rs``
``mjai_log_per_player``). The engine owns the firewall: this module consumes
ONLY seat-filtered yielded observations (acting seat hand non-empty, every
other seat empty -- asserted per step, fail closed) and never touches
``Kyoku.hands`` / ``paishan`` / unfiltered event streams.

Row construction reuses the adapter's EXISTING conversion machinery, never a
parallel implementation:

- legal sets: the yielded ``Observation.legal_actions()`` (the engine's own
  seat-filtered offer) expanded through
  :func:`~hydra2.engines.riichienv.actions.expand_engine_legals` and
  :func:`~hydra2.engines.riichienv.actions.legal_view`;
- chosen actions: yielded ``Action`` objects mapped to canonical form (twin
  resolution follows the logged flags plus deterministic copy rules, exactly
  like the engine path's log matching) and encoded with the canonical codec;
- observations: assembled with the canonical
  :class:`~hydra2.contracts.observation.ObservationBuilder` from the same
  envelope constructors (:mod:`hydra2.engines.riichienv.events`) the adapter
  uses, in the same emission order, so a walled fixture agrees with the
  engine path decision-for-decision (same decision ids, seats, chosen
  actions, and string-level observation content);
- history: the framed log's MJAI events translated envelope-for-envelope in
  adapter order (turn/draw, discard, claims, windows, dora, wins, draws).

Copy-identity folding: ``MjaiReplay`` reports string-canonical tile ids (the
same id for every copy of one MJAI string), while a live engine deals
distinct physical copies from its unique wall. Rows therefore fold
physical-copy variants (smallest-copy choice everywhere, exactly like the
engine path's deterministic copy resolution) and rebuild meld tiles from the
logged strings deterministically. Consequence: ``observation_hash`` values
differ from the engine path exactly by copy identity (same tile strings,
different physical copies), plus the wall-less derivation marker. Agreement
tests assert the identical part (ids, seats, actions, string-level content)
and pin the folding explicitly.

CRITICAL wall binding: a placeholder-wall digest is NEVER bound. ``wall_id``
is always ``None`` (the :func:`_wall_id_for` None-rule: never invented) and
the derivation carries the wall-less marker
:data:`SIM_DERIVATION_MARK` instead of a wall digest. Desyncs (riichienv
``InvalidState`` / ``Replay desync``, illegal offered actions, firewall
breaches) raise :class:`~hydra2.contracts.common.ContractError` naming
game + kyoku + step, so upstream callers quarantine and count them instead
of silently dropping rows.
"""

from __future__ import annotations

import tempfile
from pathlib import Path as Path
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

from hydra2.contracts.action import CanonicalAction as CanonicalAction
from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_digest_text as make_digest_text
from hydra2.contracts.common import make_seat as make_seat
from hydra2.contracts.common import make_tile_id as make_tile_id
from hydra2.contracts.observation import ObservationBuilder as ObservationBuilder
from hydra2.engines.riichienv._lr_act import _check_drawer as _check_drawer
from hydra2.engines.riichienv._lr_act import _clear_stash_no_claim as _clear_stash_no_claim
from hydra2.engines.riichienv._lr_act import _do_dahai as _do_dahai
from hydra2.engines.riichienv._lr_act import _do_tsumo as _do_tsumo
from hydra2.engines.riichienv._lr_act import _emit_call_resolved as _emit_call_resolved
from hydra2.engines.riichienv._lr_act import _require_actor as _require_actor
from hydra2.engines.riichienv._lr_claim import _do_ankan as _do_ankan
from hydra2.engines.riichienv._lr_claim import _do_claim as _do_claim
from hydra2.engines.riichienv._lr_claim import _do_dora as _do_dora
from hydra2.engines.riichienv._lr_claim import _do_kakan as _do_kakan
from hydra2.engines.riichienv._lr_claim import _do_reach as _do_reach
from hydra2.engines.riichienv._lr_claim import _do_reach_accepted as _do_reach_accepted
from hydra2.engines.riichienv._lr_frame import _CLAIM as _CLAIM
from hydra2.engines.riichienv._lr_frame import _END as _END
from hydra2.engines.riichienv._lr_frame import _ROW as _ROW
from hydra2.engines.riichienv._lr_frame import _SKIP as _SKIP
from hydra2.engines.riichienv._lr_frame import _START as _START
from hydra2.engines.riichienv._lr_frame import _coerce_game as _coerce_game
from hydra2.engines.riichienv._lr_frame import _drain_game_steps as _drain_game_steps
from hydra2.engines.riichienv._lr_frame import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv._lr_frame import _frame_bytes as _frame_bytes
from hydra2.engines.riichienv._lr_frame import _packet_boundary_hash as _packet_boundary_hash
from hydra2.engines.riichienv._lr_frame import _rules as _rules
from hydra2.engines.riichienv._lr_frame import _rules_hash as _rules_hash
from hydra2.engines.riichienv._lr_frame import _sim_game_id as _sim_game_id
from hydra2.engines.riichienv._lr_frame import _table as _table
from hydra2.engines.riichienv._lr_oracle import _peek_window_claim as _peek_window_claim
from hydra2.engines.riichienv._lr_rows import _emit as _emit
from hydra2.engines.riichienv._lr_rows import _GameState as _GameState
from hydra2.engines.riichienv._lr_walk import _do_start_kyoku as _do_start_kyoku
from hydra2.engines.riichienv._lr_walk import _pop_draw_head as _pop_draw_head
from hydra2.engines.riichienv._lr_walk import _strict_row as _strict_row
from hydra2.engines.riichienv._oracle_base import _TRANSPARENT_KINDS as _TRANSPARENT_KINDS
from hydra2.engines.riichienv.events import make_delta as make_delta
from hydra2.engines.riichienv.events import reason_kind as reason_kind

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.data.decode import GameRecord as GameRecord
    from hydra2.data.parquet import DecisionRow as DecisionRow
    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep
    from hydra2.engines.riichienv._lr_walk import _KyokuWalk as _KyokuWalk


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
        step = _pop_draw_head(state, walk, kyoku, winner, why="hora")
        if step.mjai_type != "hora" or step.drawn is None:
            raise state.fail(kyoku, "hora", "tsumo win without a drawn winning step")
        tile = step.drawn
        if step.tile is not None and step.tile != tile:
            raise state.fail(kyoku, "hora", "tsumo tile differs from the drawn tile")
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
            walk.tw.do_tsumo_win(winner, tiles.mjai_string_of(tile))
        except (ContractError, ValueError) as exc:
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
        if step.mjai_type != "hora":
            raise state.fail(kyoku, "hora", f"seat {winner} oracle holds {step.mjai_type}")
        tile = state.last_discard[1]
        if step.tile is not None and tiles.mjai_string_of(step.tile) != tiles.mjai_string_of(tile):
            raise state.fail(kyoku, "hora", "ron tile differs from the offered discard")
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
        for other, other_step in list(walk.stash.items()):
            if other_step.mjai_type == "hora":
                del walk.stash[other]
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
    walk.exp_drawer = None
    walk.kan_pending = False


def _do_ryukyoku(state: _GameState, walk: _KyokuWalk, kyoku: int, event: dict[str, object]) -> None:
    if state.decided:
        raise state.fail(kyoku, "ryukyoku", "draw after the kyoku was decided")
    _clear_stash_no_claim(state, walk, kyoku, where="ryukyoku")
    reason = event.get("reason")
    if not isinstance(reason, str) or reason == "":
        # Tenhou-real MJAI omits the reason for wall exhaustion and first-turn
        # kyushu aborts; every other abort carries one. Exhaustion always needs
        # a full wall of draws; kyushu is verified against the oracle engine's
        # own nine-terminals offer. Anything between stays quarantined.
        if state.draws >= 60:
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
    walk.exp_drawer = None
    walk.kan_pending = False


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


def _walk_game(state: _GameState, drained: list[list[list[_SimStep]]]) -> list[DecisionRow]:
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
            walk = _do_start_kyoku(state, drained, kyoku_ordinal, ev, events, idx)
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
            _do_tsumo(state, walk, kyoku_ordinal, ev, events, idx)
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
    framed game bytes. Bytes/paths are decoded strictly first. The framed
    bytes are staged through a system temporary file (never the corpus) for
    ``MjaiReplay.from_jsonl``; ``take_kyokus`` plus ``steps(seat)`` for seats
    0..3 drive the row walk. ``split`` rides the rows opaquely (non-empty).
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
    # Tenhou emits a bare {"type": "dora"} with no marker on kan-dora, which
    # the drain rejects whole-game with an external parse error. Own the
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
    framed = _frame_bytes(record)
    with tempfile.TemporaryDirectory(prefix="hydra2-simreplay-") as tmpdir:
        staged = str(Path(tmpdir) / "game.mjai.jsonl")
        _ = Path(staged).write_bytes(framed)  # staging write; byte count unneeded
        drained = _drain_game_steps(staged, game_id=record.game_id)
        rows = _walk_game(state, drained)
    return rows
