"""RiichiEnv mjai translation — round, draw, discard, call handlers.

Owns the first translation half of :class:`RiichiEnvExactSimulator`:
round-start, draw, discard, riichi declaration/acceptance, open-call,
closed-kan, dora, and hora-run merging. The outcome handlers plus the
batch dispatcher live in
:mod:`hydra2.engines.riichienv.adapter_events_b`, the decision driver
in :mod:`hydra2.engines.riichienv.adapter_step`, and the public API in
:mod:`hydra2.engines.riichienv.adapter_core`, so each file stays inside
the review-size ceiling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import (
    TileId as TileId,
)
from hydra2.engines.riichienv.adapter_identity import _BAKAZE_TO_TILE_TYPE as _BAKAZE_TO_TILE_TYPE
from hydra2.engines.riichienv.events import (
    make_delta as make_delta,
)
from hydra2.engines.riichienv.events import (
    make_envelope as make_envelope,
)
from hydra2.engines.riichienv.events import meld_delta_value as meld_delta_value

if TYPE_CHECKING:
    import riichienv as riichienv

    from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder

__all__ = [
    "AdapterEventsAMixin",
]


class AdapterEventsAMixin:
    """Round/draw/discard/call handlers for :class:`RiichiEnvExactSimulator`.

    Split host for the first translation half; the batch dispatcher and
    outcome handlers arrive via the events-b subclass, which adds the
    translation entry points the step driver calls. Attribute access is
    duck-typed through the subclass.
    """

    _accepted_claim_id: Any
    _builder: ObservationBuilder | None
    _emit: Any
    _encode_fallback_discard: Any
    _env: riichienv.RiichiEnv | None
    _event_schema_hash: str
    _exact_draw_int: Any
    _game_id: str
    _hand_index: int
    _inv: list[int]
    _ippatsu: list[bool]
    _last_discard: tuple[int | None, int | None]
    _last_discard_int: Any
    _latest_meld_tiles: Any
    _next_discard_int: Any
    _next_seq: Any
    _note_step_draws: Any
    _on_end_game: Any
    _on_end_kyoku: Any
    _on_hora: Any
    _on_kakan: Any
    _on_ryukyoku: Any
    _peek_discard_int: Any
    _refresh_public_snapshot: Any
    _rules_hash: str
    _stamp_id: int | None
    _stamp_if: Any
    _stamp_kind: str | None
    _window_offered_ids: tuple[int, ...]
    _window_opened_by_discard: bool

    @property
    def _engine(self) -> riichienv.RiichiEnv: ...

    # ------------------------------------------------------- translation

    def _consume_new_events(self) -> Any:
        assert self._env is not None
        batch = self._env.mjai_log[self._cursor :]
        self._cursor = len(self._env.mjai_log)
        return self._translate_batch(batch)

    def _translate_batch(self, batch: Any) -> Any:
        index: int = 0
        batch_seq: Any = batch
        while index < len(cast("Any", batch_seq)):
            event: Any = batch_seq[index]
            kind: Any = event["type"]
            if kind == "start_game":
                index += 1
                continue
            if kind == "start_kyoku":
                self._on_start_kyoku(cast("Any", event))
                index += 1
                continue
            if kind == "tsumo":
                self._on_tsumo(cast("Any", event))
            elif kind == "dahai":
                self._on_dahai(cast("Any", event))
            elif kind == "reach":
                if index + 1 >= len(batch):
                    # Declaration discard sits in the NEXT batch (two-step
                    # riichi crosses the mjai batch boundary); stop consuming
                    # and let the following translation pair them.
                    return None
                declaration: Any = batch_seq[index + 1]
                self._on_reach(cast("Any", event), cast("Any", declaration))
                index += 1
            elif kind == "reach_accepted":
                self._on_reach_accepted(cast("Any", event))
            elif kind in ("chi", "pon", "daiminkan"):
                self._on_call(cast("Any", event))
            elif kind == "ankan":
                self._on_ankan(cast("Any", event))
            elif kind == "kakan":
                self._on_kakan(cast("Any", event))
            elif kind == "dora":
                self._on_dora(cast("Any", event))
            elif kind == "hora":
                merged, index = self._merge_horas(batch, index)
                self._on_hora(merged)
                # D-WP03A-12: _merge_horas already advanced index past the
                # whole hora run; the bottom index += 1 would swallow the
                # event right after it (typically end_kyoku, dropping
                # round_end and the schedule-derived hand reopen).
                continue
            elif kind == "ryukyoku":
                self._on_ryukyoku(cast("Any", event))
            elif kind == "end_kyoku":
                following: Any | None = (
                    batch_seq[index + 1] if index + 1 < len(cast("Any", batch_seq)) else None
                )
                self._on_end_kyoku(cast("Any", event), cast("Any", following))
                if following is not None and cast("Any", following["type"]) == "start_kyoku":
                    following_dict: dict[str, Any] = cast("dict[str, Any]", following)
                    return {
                        "oya": int(cast("Any", following_dict["oya"])),
                        "honba": int(cast("Any", following_dict["honba"])),
                        "kyotaku": int(cast("Any", following_dict["kyotaku"])),
                        "scores": [
                            int(cast("Any", s)) for s in cast("Any", following_dict["scores"])
                        ],
                        "bakaze": str(cast("Any", following_dict["bakaze"])),
                    }
                if following is None:
                    # Boundary sits in the NEXT step batch; force the native
                    # advance and capture any immediate dealer draw it deals.
                    _pre = len(self._engine.mjai_log)
                    _ = self._engine.step({})
                    self._note_step_draws(_pre)
                    return None
            elif kind == "end_game":
                self._on_end_game(cast("Any", event))
                return "terminal"
            else:  # pragma: no cover - closed mjai vocabulary
                raise ContractError(f"unexpected mjai event type {kind!r}")
            index += 1
        return None

    def _on_start_kyoku(self, event: Any) -> None:
        assert self._env is not None
        self._hand_index += 1
        self._ippatsu = [False] * 4  # new hand: no ippatsu window open
        dealer = self._inv[int(cast("Any", event["oya"]))]
        scores: tuple[int, ...] = tuple(int(cast("Any", s)) for s in cast("Any", event["scores"]))
        wind_type = _BAKAZE_TO_TILE_TYPE[str(cast("Any", event["bakaze"]))]
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="round_start",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(dealer),
                round_index=self._hand_index,
                scores=scores,
                public_delta=(
                    make_delta(("round_index",), "set", self._hand_index),
                    make_delta(("honba",), "set", int(cast("Any", event["honba"]))),
                    make_delta(("riichi_sticks",), "set", int(cast("Any", event["kyotaku"]))),
                    make_delta(("scores",), "set", list(scores)),
                ),
            )
        )
        self._refresh_public_snapshot(phase="round_start", turn_actor=dealer)
        assert self._builder is not None
        self._builder.update_public_state(round_wind=wind_type)

    def _on_tsumo(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        # D-WP04A-FIX1 scoping: a kakan records its added tile as the live
        # chankan offer, but once the NEXT turn's draw happens the kan actor
        # (or any rinshan continuation) is drawing - no discard sits on the
        # table. Leaving the record set would build contexts with
        # offered_by == actor (the kan drawer) and violate ActionContext.
        if self._last_discard[0] == actor and self._window_offered_ids == ():
            self._last_discard = (None, None)
        tile = self._exact_draw_int()
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="turn_advance",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
            )
        )
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="draw_tile",
                visibility="actor_private",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                tile=tile,
            )
        )
        self._refresh_public_snapshot(turn_actor=actor)

    def _on_dahai(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        tile = self._next_discard_int(
            int(cast("Any", event["actor"])), str(cast("Any", event["pai"]))
        )
        action_id = self._stamp_if(actor, "discard")
        if action_id is None:
            action_id = self._stamp_if(actor, "tsumogiri")
        if action_id is None:
            action_id = self._encode_fallback_discard(actor, tile)
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="discard",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                tile=tile,
                action_id=action_id,
            )
        )
        self._last_discard = (actor, tile)
        # The next claim window (if any responder acts on this tile) opens
        # from a discard; kakan windows reset it to False at their own site.
        self._window_opened_by_discard = True
        self._ippatsu[actor] = False  # declarer discarded again: ippatsu gone
        self._stamp_id = None
        self._stamp_kind = None
        self._refresh_public_snapshot()

    def _on_reach(self, event: Any, declaration: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        tile = self._peek_discard_int(
            int(cast("Any", declaration["actor"])), str(cast("Any", declaration["pai"]))
        )
        action_id = self._stamp_if(actor, "riichi_discard")
        if action_id is None:  # pragma: no cover - declarations always follow apply()
            raise ContractError("reach event without an applied riichi_discard")
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="riichi_declared",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                tile=tile,
                action_id=action_id,
                public_delta=(make_delta(("riichi_states", actor), "set", "declared"),),
            )
        )

    def _on_reach_accepted(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="riichi_accepted",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                public_delta=(
                    make_delta(("riichi_states", actor), "set", "accepted"),
                    make_delta(("riichi_sticks",), "increment", 1),
                    make_delta(("ippatsu", actor), "set", True),
                ),
            )
        )
        self._ippatsu[actor] = True  # accepted reach opens the ippatsu window
        self._refresh_public_snapshot()

    def _on_call(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        called = self._last_discard_int(int(cast("Any", event["target"])))
        meld_tiles = self._latest_meld_tiles(int(cast("Any", event["actor"])))
        consumed = tuple(sorted(t for t in meld_tiles if t != called))
        action_id = self._accepted_claim_id(actor)
        kind = str(cast("Any", event["type"]))
        # The claimed tile belongs to the seat that discarded it (the engine's
        # call target); kan-offered claims have no discard source.
        source = (
            self._inv[int(cast("Any", event["target"]))]
            if kind == "daiminkan" or self._last_discard[0] is not None
            else None
        )
        deltas = [
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
            )
        ]
        if kind == "daiminkan":
            deltas.append(make_delta(("kan_count",), "increment", 1))
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind=kind,
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                tile=called,
                action_id=action_id,
                source_seat=None if source is None else _bridge_contracts.make_seat(source),
                consumed_tiles=consumed,
                public_delta=tuple(deltas),
            )
        )
        self._last_discard = (None, None)
        self._ippatsu = [False] * 4  # any call interrupts every ippatsu chance
        self._stamp_id = None
        self._stamp_kind = None
        self._refresh_public_snapshot()

    def _on_ankan(self, event: Any) -> None:
        actor = self._inv[int(cast("Any", event["actor"]))]
        consumed = self._latest_meld_tiles(int(cast("Any", event["actor"])))
        action_id = self._stamp_if(actor, "ankan")
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="ankan",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=_bridge_contracts.make_seat(actor),
                action_id=action_id,
                consumed_tiles=consumed,
                public_delta=(
                    make_delta(
                        ("melds", actor),
                        "append",
                        meld_delta_value(
                            kind="ankan",
                            owner=actor,
                            source_seat=None,
                            called_tile=None,
                            tiles=consumed,
                        ),
                    ),
                    make_delta(("kan_count",), "increment", 1),
                ),
            )
        )
        self._ippatsu = [False] * 4  # kan interrupts every ippatsu chance
        self._stamp_id = None
        self._stamp_kind = None
        self._refresh_public_snapshot()

    def _on_dora(self, event: Any) -> None:
        tile = TileId(self._engine.dora_indicators[-1])
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="dora_revealed",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                tile=tile,
                public_delta=(make_delta(("dora_indicators",), "append", tile),),
            )
        )
        self._refresh_public_snapshot()

    def _merge_horas(self, batch: Any, index: int) -> tuple[list[Any], int]:
        merged: list[Any] = [cast("Any", batch[index])]
        position: int = index + 1
        batch_any: Any = batch
        while (
            position < len(cast("Any", batch_any))
            and cast("Any", batch_any[position]["type"]) == "hora"
        ):
            merged.append(cast("Any", batch_any[position]))
            position += 1
        return merged, position
