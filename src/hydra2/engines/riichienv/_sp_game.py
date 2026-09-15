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

from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

from hydra2.contracts.common import ContractError as ContractError
from hydra2.contracts.common import make_digest_text as make_digest_text
from hydra2.contracts.observation import ObservationBuilder as ObservationBuilder
from hydra2.engines.riichienv._oracle_base import _TRANSPARENT_KINDS as _TRANSPARENT_KINDS
from hydra2.engines.riichienv._sp_kans import _do_ankan as _do_ankan
from hydra2.engines.riichienv._sp_kans import _do_dora as _do_dora
from hydra2.engines.riichienv._sp_kans import _do_kakan as _do_kakan
from hydra2.engines.riichienv._sp_kans import _do_reach_accepted as _do_reach_accepted
from hydra2.engines.riichienv._sp_oracle import _peek_window_claim as _peek_window_claim
from hydra2.engines.riichienv._sp_reach import _do_claim as _do_claim
from hydra2.engines.riichienv._sp_reach import _do_reach as _do_reach
from hydra2.engines.riichienv._sp_records import _CLAIM as _CLAIM
from hydra2.engines.riichienv._sp_records import _END as _END
from hydra2.engines.riichienv._sp_records import _ROW as _ROW
from hydra2.engines.riichienv._sp_records import _SKIP as _SKIP
from hydra2.engines.riichienv._sp_records import _START as _START
from hydra2.engines.riichienv._sp_records import _coerce_game as _coerce_game
from hydra2.engines.riichienv._sp_records import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv._sp_records import _GameState as _GameState
from hydra2.engines.riichienv._sp_records import _packet_boundary_hash as _packet_boundary_hash
from hydra2.engines.riichienv._sp_records import _rules as _rules
from hydra2.engines.riichienv._sp_records import _rules_hash as _rules_hash
from hydra2.engines.riichienv._sp_records import _sim_game_id as _sim_game_id
from hydra2.engines.riichienv._sp_records import _table as _table
from hydra2.engines.riichienv._sp_walk import _do_start_kyoku as _do_start_kyoku
from hydra2.engines.riichienv._sp_windows import _do_dahai as _do_dahai
from hydra2.engines.riichienv._sp_windows import _do_tsumo as _do_tsumo
from hydra2.engines.riichienv._sp_windows import _require_actor as _require_actor
from hydra2.engines.riichienv._sp_wins import _do_end_game as _do_end_game
from hydra2.engines.riichienv._sp_wins import _do_end_kyoku as _do_end_kyoku
from hydra2.engines.riichienv._sp_wins import _do_hora as _do_hora
from hydra2.engines.riichienv._sp_wins import _do_ryukyoku as _do_ryukyoku

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from pathlib import Path as Path

    from hydra2.data.decode import GameRecord as GameRecord
    from hydra2.data.parquet import DecisionRow as DecisionRow
    from hydra2.engines.riichienv._sp_walk import _KyokuWalk as _KyokuWalk


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
