"""RiichiEnv outcomes and exact tiles — dispatcher, hora, helpers.

Owns the second translation half of :class:`RiichiEnvExactSimulator`:
the batch dispatcher with hora-run merging, hora and draw outcomes,
hand and game boundaries, window resolution, stamp and claim-id
helpers, and the exact physical-tile resolvers. The round and call
handlers live in :mod:`hydra2.engines.riichienv.adapter_events_a`, the
decision driver in :mod:`hydra2.engines.riichienv.adapter_step`, and
the public API in :mod:`hydra2.engines.riichienv.adapter_core`, so each
file stays inside the review-size ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
from hydra2.contracts.action_table import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import (
    make_sequence_no as make_sequence_no,
)
from hydra2.engines.riichienv import events as events
from hydra2.engines.riichienv.events import (
    make_delta as make_delta,
)
from hydra2.engines.riichienv.events import (
    make_envelope as make_envelope,
)
from hydra2.engines.riichienv.events import meld_delta_value as meld_delta_value
from hydra2.engines.riichienv.events import reason_kind as reason_kind
from hydra2.engines.riichienv.state import (
    raw_outcome_from_final as raw_outcome_from_final,
)
from hydra2.engines.riichienv.state import (
    settlement_facts_from_deltas as settlement_facts_from_deltas,
)
from hydra2.engines.riichienv.walls import derive_hand_wall as derive_hand_wall

if TYPE_CHECKING:
    from collections.abc import Callable as Callable

    import riichienv as riichienv

    from hydra2.contracts.action_table import ActionContext as ActionContext
    from hydra2.contracts.action_table import ActionTable as ActionTable
    from hydra2.contracts.common import DigestText as DigestText
    from hydra2.contracts.common import Seat as Seat
    from hydra2.contracts.common import TileId as TileId
    from hydra2.contracts.event_envelope import EventEnvelope as EventEnvelope
    from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder
    from hydra2.contracts.rules_manifest import RulesManifest as RulesManifest
    from hydra2.contracts.utility import RawOutcome as RawOutcome
    from hydra2.contracts.utility import SettlementFact as SettlementFact

__all__ = [
    "AdapterEventsBMixin",
]


class AdapterEventsBMixin:
    """Outcomes and exact tiles for :class:`RiichiEnvExactSimulator`.

    Split host for the second translation half; the round and call
    handlers live in the events-a sibling, the decision driver in the
    step sibling. Attribute access is duck-typed through the subclass.
    """

    _buffered: dict[int, tuple[CanonicalAction, object, int]]
    _builder: ObservationBuilder | None
    _context_for: Any
    _cursor: int
    _dahai_cursor: list[int]
    _decision_seat: int | None
    _draw_queue: list[int]
    _emit: Any
    _env: riichienv.RiichiEnv | None
    _event_schema_hash: str
    _events: list[EventEnvelope]
    _game_id: str
    _hand_index: int
    _inv: list[int]
    _ippatsu: list[bool]
    _kakan_added: dict[int, int]
    _last_discard: tuple[int | None, int | None]
    _mode: str | None
    _next_seq: Callable[[], int]
    _open_hand: Any
    _pending: list[int]
    _perm: tuple[Seat, ...]
    _pre_step_hands: dict[int, tuple[int, ...]]
    _process_until_decision: Any
    _raw_outcome: RawOutcome | None
    _refresh_public_snapshot: Any
    _reopen_hand: Any
    _rules: RulesManifest | None
    _rules_hash: str
    _schedule_digest: str
    _schedule_id: str
    _schedule_tiles: tuple[TileId, ...]
    _settlements: list[SettlementFact]
    _staging: list[EventEnvelope] | None
    _stamp_actor: int | None
    _stamp_id: int | None
    _stamp_kind: str | None
    _starting_scores: tuple[int, int, int, int]
    _table: ActionTable
    _terminal: bool
    _to_canonical_scores: Any
    _to_engine_order: Any
    _translate_batch: Any
    _views: dict[int, tuple[tuple[CanonicalAction, ...], tuple[bool, ...]]]
    _window_offered_ids: tuple[int, ...]
    _window_opened_by_discard: bool

    @property
    def _engine(self) -> riichienv.RiichiEnv: ...

    def _on_hora(self, horas: Any) -> None:
        first: Any = cast("Any", horas[0])
        winner = self._inv[int(cast("Any", first["actor"]))]
        self_draw = bool(cast("Any", first.get("tsumo")))
        loser = self._inv[int(cast("Any", first["target"]))]
        drawn = self._engine.drawn_tile
        offered = self._last_discard[1]
        tile = drawn if self_draw and drawn is not None else (-1 if offered is None else offered)
        # Chankan rons resolve inside a kakan window: the winner's RON sat in
        # the buffer, so its canonical id comes from the buffered decision
        # rather than the apply-time stamp (already cleared by the kakan).
        action_id = self._stamp_if(winner, "tsumo" if self_draw else "ron")
        if action_id is None and not self_draw and len(self._buffered) > 0:
            try:
                action_id = self._accepted_claim_id(winner)
            except (ContractError, ValueError):
                action_id = None
        winners: list[int] = sorted(
            self._inv[int(cast("Any", h["actor"]))] for h in cast("Any", horas)
        )
        deltas = [0, 0, 0, 0]
        for hora in cast("Any", horas):
            for seat, value in enumerate(cast("Any", hora["deltas"])):
                deltas[seat] += int(cast("Any", value))
        kind = "tsumo" if self_draw else "ron"
        public = [
            make_delta(("scores",), "set", [deltas[seat] for seat in range(4)]),
        ]
        sticks_before = self._engine.riichi_sticks
        if not self_draw and sticks_before != 0:
            public.append(make_delta(("riichi_sticks",), "increment", -sticks_before))
        payload_kwargs: dict[str, int | None] = {}
        if self_draw:
            payload_kwargs["source_seat"] = None
        else:
            payload_kwargs["source_seat"] = loser
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind=kind,
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=cast("int", _bridge_contracts.make_seat(winner)),
                tile=tile,
                action_id=None if action_id is None else action_id,
                public_delta=tuple(public),
                source_seat=payload_kwargs["source_seat"],
            )
        )
        self._settlements.extend(
            settlement_facts_from_deltas(
                kind=kind,
                deltas=deltas,
                payer_seat=None if self_draw else loser,
                winner_seats=winners,
            )
        )
        self._stamp_id = None
        self._stamp_kind = None
        self._refresh_public_snapshot()

    def _on_ryukyoku(self, event: Any) -> None:
        classification = reason_kind(str(cast("Any", event["reason"])))
        deltas: list[int] = [int(cast("Any", d)) for d in cast("Any", event["deltas"])]
        pre: tuple[int, int, int, int] = self._to_canonical_scores(self._engine.scores())
        post = tuple(pre[seat] + deltas[seat] for seat in range(4))
        public = [
            make_delta(("scores",), "set", [deltas[seat] for seat in range(4)]),
        ]
        if classification == "abortive_draw":
            self._emit(
                make_envelope(
                    game_id=self._game_id,
                    sequence=self._next_seq(),
                    kind="abortive_draw",
                    visibility="public",
                    rules_hash=self._rules_hash,
                    schema_hash=self._event_schema_hash,
                    round_index=self._hand_index,
                    scores=post,
                    reason=events.ABORTIVE_REASONS[str(cast("Any", event["reason"]))],
                    public_delta=tuple(public),
                )
            )
        else:
            self._emit(
                make_envelope(
                    game_id=self._game_id,
                    sequence=self._next_seq(),
                    kind="draw_end",
                    visibility="public",
                    rules_hash=self._rules_hash,
                    schema_hash=self._event_schema_hash,
                    scores=post,
                    reason=str(cast("Any", event["reason"])),
                    public_delta=tuple(public),
                )
            )
        self._settlements.extend(
            settlement_facts_from_deltas(
                kind=classification,
                deltas=deltas,
                payer_seat=None,
                winner_seats=tuple(seat for seat in range(4) if deltas[seat] > 0),
            )
        )
        self._refresh_public_snapshot()

    def _on_end_kyoku(self, event: Any, following: Any) -> None:
        del event
        final_scores: tuple[int, int, int, int] = self._to_canonical_scores(self._engine.scores())
        next_round_index = self._hand_index + (1 if following is not None else 0)
        deltas = [
            make_delta(("scores",), "set", list(final_scores)),
            *(make_delta(("riichi_states", seat), "set", "none") for seat in range(4)),
        ]
        deltas.insert(1, make_delta(("round_index",), "set", next_round_index))
        if following is not None and str(cast("Any", following.get("type"))) == "start_kyoku":
            deltas.append(make_delta(("honba",), "set", int(cast("Any", following["honba"]))))
            deltas.append(
                make_delta(("riichi_sticks",), "set", int(cast("Any", following["kyotaku"])))
            )
        for seat in range(4):
            deltas.append(make_delta(("ippatsu", seat), "set", False))
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="round_end",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                round_index=self._hand_index,
                scores=tuple(final_scores),
                public_delta=tuple(deltas),
            )
        )
        self._refresh_public_snapshot(phase="round_end")

    def _on_end_game(self, event: Any) -> None:
        del event
        end_scores: tuple[int, int, int, int] = self._to_canonical_scores(self._engine.scores())
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="game_end",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                round_index=max(0, self._hand_index),
                scores=tuple(end_scores),
                reason="hanchan_complete",
                public_delta=(make_delta(("scores",), "set", list(end_scores)),),
            )
        )
        assert self._rules is not None
        self._raw_outcome = raw_outcome_from_final(
            final_scores=end_scores,
            starting_scores=self._starting_scores,
            settlements=self._settlements,
            rules_id=self._rules.rules_id,
            rules_hash=cast("DigestText", _bridge_contracts.make_digest_text(self._rules_hash)),
        )
        self._terminal = True
        self._refresh_public_snapshot(phase="game_end")

    # ------------------------------------------------------- boundaries

    def _reopen_hand(self, carry: Any) -> None:
        assert self._rules is not None
        hand_index = self._hand_index + 1
        derived = derive_hand_wall(
            schedule_digest=self._schedule_digest,
            schedule_id=self._schedule_id,
            hand_index=hand_index,
        )
        tiles = self._schedule_tiles if hand_index == 0 else derived
        wind_letter = {"E": 0, "S": 1, "W": 2, "N": 3}[str(cast("Any", carry["bakaze"]))]
        self._hand_index = hand_index
        self._open_hand(
            tiles,
            oya_engine=int(cast("Any", carry["oya"])),
            honba=int(cast("Any", carry["honba"])),
            kyotaku=int(cast("Any", carry["kyotaku"])),
            scores_engine=self._to_engine_order(
                [int(cast("Any", s)) for s in cast("Any", carry["scores"])]
            ),
            round_wind_int=wind_letter,
        )

    # ------------------------------------------------------- resolution

    def _flush_window_resolution(self) -> None:
        assert self._env is not None and self._builder is not None
        batch = self._env.mjai_log[self._cursor :]
        self._cursor = len(self._env.mjai_log)
        staging: list[EventEnvelope] = []
        self._staging = staging
        try:
            outcome: str | dict[str, Any] | None = self._translate_batch(batch)
        finally:
            self._staging = None
        accepted: list[int] = []
        for envelope in staging:
            if envelope.kind in ("chi", "pon", "daiminkan", "ron") and (
                envelope.payload.action_id is not None
            ):
                accepted = [int(cast("Any", envelope.payload.action_id))]
                break
        offered = self._window_offered_ids if len(accepted) > 0 else ()
        # D-WP02D-2: offered ids are only representable alongside exactly one
        # accepted id; all-pass resolutions carry neither.
        if not self._window_opened_by_discard:
            resolved: list[EventEnvelope] = staging
        else:
            resolution = make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="call_resolved",
                visibility="server_private",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                offered_action_ids=offered,
                accepted_action_ids=accepted,
            )
            resolved = [resolution, *staging]
        resolved_boundary = outcome if isinstance(outcome, dict) else None
        for envelope in resolved:
            refitted = dataclasses.replace(envelope, sequence=make_sequence_no(self._next_seq()))
            self._emit(refitted)
        self._window_offered_ids = ()
        if resolved_boundary is not None:
            self._reopen_hand(resolved_boundary)
            self._process_until_decision()

    # ------------------------------------------------------- helpers

    def _stamp_if(self, actor: int, kind: str) -> int | None:
        """The applied action's canonical id when it matches actor and kind.

        Discard-family probes accept either spelling (the engine folds
        hand-discards and tsumogiri into one DISCARD slot) and the
        declaration discard inherits the riichi_discard id.
        """
        if self._stamp_id is None or self._stamp_actor != actor:
            return None
        applied = self._stamp_kind
        if applied == kind or {applied, kind} <= {"discard", "tsumogiri"}:
            return self._stamp_id
        if applied == "riichi_discard" and kind in ("discard", "tsumogiri"):
            return self._stamp_id
        return None

    def _accepted_claim_id(self, actor: int) -> int:
        if len(self._buffered) > 0:
            for seat, (_action, _engine, buffered_id) in self._buffered.items():
                if seat == actor:
                    return buffered_id
        stamped = self._stamp_if(actor, "chi")
        if stamped is None:
            stamped = self._stamp_if(actor, "pon")
        if stamped is None:
            stamped = self._stamp_if(actor, "daiminkan")
        if stamped is not None:
            return stamped
        raise ContractError(f"claim event for seat {actor} without a buffered decision")

    def _encode_fallback_discard(self, actor: int, tile: int) -> int:
        # The physical discard was offered while its decision existed; encode
        # under draw_decision so the coarse phase gate cannot reject history.
        action = CanonicalAction(
            kind="discard",
            actor=cast("Seat", _bridge_contracts.make_seat(actor)),
            tile=cast("TileId", _bridge_contracts.make_tile_id(tile)),
            called_tile=None,
            consumed_tiles=(),
            source_seat=None,
            declares_riichi=False,
            metadata=(),
        )
        # D-WP03A-10: engine-auto discards without a live applied-action link
        # get a deterministic id from a minimal synthetic context (the tile
        # itself as the concealed set). Ids stay pure functions of the action.
        context: ActionContext = self._context_for(actor, phase_override="draw_decision")
        if tile not in {int(t) for t in context.own_concealed_tiles}:
            from hydra2.contracts.action_table import ActionContext

            context = ActionContext(
                actor=cast("Seat", _bridge_contracts.make_seat(actor)),
                action_table_hash=self._table.digest,
                phase="draw_decision",
                offered_tile=None,
                offered_by=None,
                own_concealed_tiles=(_bridge_contracts.make_tile_id(tile),),
                visible_melds=(),
            )
        return int(canonical_action_codec.encode(action, table=self._table, context=context))

    # -- exact physical-tile resolution (mjai strings lose copy identity) --

    def _note_step_draws(self, previous_log_length: int) -> None:
        assert self._env is not None
        new_events = self._env.mjai_log[previous_log_length:]
        if any(e["type"] == "tsumo" for e in new_events) and self._env.drawn_tile is not None:
            self._draw_queue.append(self._env.drawn_tile)

    def _exact_draw_int(self) -> int:
        assert len(self._draw_queue) > 0, "tsumo event without a captured draw id"
        return self._draw_queue.pop(0)

    def _on_kakan(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        engine_pid = int(cast("Any", event["actor"]))
        # D-WP03A-11: the exact added tile was captured at apply() time; the
        # pon triple it upgraded is still the seat's latest meld at mjai
        # translation time (engine 0.4.10 upgrades that meld IN PLACE, so
        # scanning for a separate 3-tile pon would fail after any later meld).
        meld_tiles: list[int] = self._latest_meld_tiles(engine_pid)
        added: int | None = self._kakan_added.get(engine_pid)
        if added is None:  # pragma: no cover - apply-time capture guarantees presence
            raise ContractError(
                f"kakan by engine seat {engine_pid}: no captured added tile; "
                "translation ran without a preceding canonical apply"
            )
        del self._kakan_added[engine_pid]
        consumed = tuple(sorted(t for t in meld_tiles if t != added))
        action_id: int | None = self._stamp_if(actor, "kakan")
        self._window_opened_by_discard = False
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="kakan",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=cast("int", _bridge_contracts.make_seat(actor)),
                tile=added,
                action_id=action_id,
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
        )
        self._ippatsu = [False] * 4  # kakan interrupts every ippatsu chance
        # D-WP04A-FIX1 (Main-authorized root-cause fix): the kakan tile is
        # the offer a chankan ron claims. Recording it as the live window
        # context keeps `_context_for` consistent with the engine's RON slot
        # (offered tile/source must match the winning tile), so the canonical
        # ron encodes and applies instead of failing validation.
        self._last_discard = (actor, added)
        self._stamp_id = None
        self._stamp_kind = None
        self._refresh_public_snapshot()

    def _peek_discard_int(self, engine_pid: int, mjai_pai: str) -> int:
        """Exact physical id of the seat's pending declaration discard."""
        assert self._env is not None
        river = self._env.discards[engine_pid]
        index = self._dahai_cursor[engine_pid]
        if index < len(river):
            return river[index]
        pre = self._pre_step_hands.get(engine_pid)
        if pre is not None:
            post = tuple(sorted(t for t in self._env.hands[engine_pid]))
            missing = [t for t in pre if post.count(t) < pre.count(t)]
            if len(missing) == 1:
                return missing[0]
            raise ContractError(
                f"cannot resolve declaration discard of {mjai_pai!r} on engine "
                f"seat {engine_pid}: hand diff {missing!r} is not a single tile"
            )
        raise ContractError(
            f"cannot resolve declaration discard of {mjai_pai!r} on engine seat "
            f"{engine_pid}: no pre-step hand captured"
        )

    def _capture_hands_for_step(self, actions: dict[int, Any]) -> None:
        """Snapshot the hands a combined step will draw from (exact discards)."""
        assert self._env is not None
        for pid in actions:
            self._pre_step_hands[pid] = tuple(sorted(t for t in self._env.hands[pid]))

    def _next_discard_int(self, engine_pid: int, mjai_pai: str) -> int:
        """Exact physical id of the seat's next translated discard.

        The engine river is authoritative while current; when it lags the
        mjai log (riichi declaration, boundary batches) resolve via the
        pre-step hand snapshot, then string parsing.
        """
        assert self._env is not None
        river = self._env.discards[engine_pid]
        index = self._dahai_cursor[engine_pid]
        if index < len(river):
            self._dahai_cursor[engine_pid] = index + 1
            return river[index]
        # River lags behind the mjai log on riichi-declaration turns and
        # boundary batches: resolve via the pre-step hand snapshot, falling
        # back to string parsing (exact after the corrected five-copy fix).
        pre = self._pre_step_hands.get(engine_pid)
        if pre is not None:
            post = tuple(sorted(t for t in self._env.hands[engine_pid]))
            missing = [t for t in pre if post.count(t) < pre.count(t)]
            if len(missing) == 1:
                return missing[0]
        from hydra2._native import tiles  # pyrefly: ignore[missing-import]

        return int(cast("Any", tiles.physical_of(mjai_pai)))

    def _last_discard_int(self, engine_pid: int) -> int:
        """Exact physical id of the engine seat's latest offered discard."""
        river = self._engine.discards[engine_pid]
        if len(river) == 0:
            raise ContractError(f"seat {engine_pid} has no discard to claim")
        return river[-1]

    def _latest_meld_tiles(self, engine_pid: int) -> list[int]:
        """Tiles of the engine seat's most recent meld (call or kan)."""
        melds = self._engine.melds[engine_pid]
        if len(melds) == 0:
            raise ContractError(f"seat {engine_pid} has no meld to translate")
        return list(melds[-1].tiles)
