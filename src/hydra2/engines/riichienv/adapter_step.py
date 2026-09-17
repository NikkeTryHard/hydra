"""RiichiEnv decision core — readiness, views, detection, game loop.

Owns the decision-loop third of :class:`RiichiEnvExactSimulator`: the
readiness and sequencing helpers, per-hand engine opening, public
snapshot refresh, codec contexts, legal-view caching, decision
detection, and the process-until-decision driver. Mjai translation
arrives via the events-a subclass, outcomes and exact-tile helpers via
the events-b base, and the public API via the core subclass, so each
file stays inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import riichienv
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.action_table import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import (
    IllegalActionError as IllegalActionError,
)
from hydra2.contracts.observation_assembly import ObservationBuilder as ObservationBuilder
from hydra2.engines.protocol import TransitionResult as TransitionResult
from hydra2.engines.riichienv.actions import legal_view as legal_view
from hydra2.engines.riichienv.adapter_events_a import AdapterEventsAMixin as AdapterEventsAMixin
from hydra2.engines.riichienv.adapter_events_b import AdapterEventsBMixin as AdapterEventsBMixin
from hydra2.engines.riichienv.adapter_identity import _BAKAZE_TO_TILE_TYPE as _BAKAZE_TO_TILE_TYPE
from hydra2.engines.riichienv.events import make_envelope as make_envelope
from hydra2.engines.riichienv.state import (
    live_wall_remaining as live_wall_remaining,
)
from hydra2.engines.riichienv.state import (
    seat_winds_for_dealer as seat_winds_for_dealer,
)
from hydra2.engines.riichienv.state import state_digest as state_digest
from hydra2.engines.riichienv.walls import WALL_STREAM_NAME as WALL_STREAM_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence

    from hydra2.contracts.action_kinds import Phase as Phase
    from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
    from hydra2.contracts.common import Seat as Seat
    from hydra2.contracts.common import TileId as TileId
    from hydra2.contracts.event_envelope import EventEnvelope as EventEnvelope
    from hydra2.contracts.rules_manifest import RulesManifest as RulesManifest
    from hydra2.contracts.utility import RawOutcome as RawOutcome

__all__ = [
    "AdapterStepMixin",
]


class AdapterStepMixin(AdapterEventsAMixin, AdapterEventsBMixin):
    """Decision core for :class:`RiichiEnvExactSimulator`.

    Split host for the decision-loop third; translation dispatch and
    outcome handlers arrive via the events subclasses, which add the
    batch translator and hand-boundary reopen the driver calls.
    """

    _builder: ObservationBuilder | None
    _cursor: int
    _dahai_cursor: list[int]
    _decision_seat: int | None
    _draw_queue: list[int]
    _engine_observation: Any
    _env: riichienv.RiichiEnv | None
    _event_schema_hash: str
    _events: list[EventEnvelope]
    _game_id: str
    _hand_index: int
    _inv: list[int]
    _ippatsu: list[bool]
    _last_discard: tuple[int | None, int | None]
    _last_masks: dict[int, tuple[bool, ...]]
    _mode: str | None
    _packet_boundary_hash: str
    _pending: list[int]
    _perm: tuple[Seat, ...]
    _pre_step_hands: dict[int, tuple[int, ...]]
    _raw_outcome: RawOutcome | None
    _rules: RulesManifest | None
    _rules_hash: str
    _schedule_digest: str
    _schedule_id: str
    _seq: int
    _staging: list[EventEnvelope] | None
    _table: Any
    _terminal: bool
    _views: dict[int, tuple[tuple[CanonicalAction, ...], tuple[bool, ...]]]
    _window_opened_by_discard: bool

    # ------------------------------------------------------- decision core

    def _require_ready(self) -> None:
        if self._env is None or self._rules is None:
            raise ContractError("simulator is not initialized; call reset() first")

    def _next_seq(self) -> int:
        value = self._seq
        self._seq += 1
        return value

    def _to_engine_order(self, canonical_scores: Sequence[int]) -> list[int]:
        # Minor destructure: local-bind inv to avoid repeated attribute lookup
        # and unpack to direct indices; keeps semantics, small hot-path win
        # (called per hand open and per score refresh).
        inv0, inv1, inv2, inv3 = self._inv  # type: ignore[assignment]  # reason: inv is 4-tuple at runtime; checker sees generic Sequence
        return [
            canonical_scores[inv0],
            canonical_scores[inv1],
            canonical_scores[inv2],
            canonical_scores[inv3],
        ]

    def _to_canonical_scores(self, engine_scores: Sequence[int]) -> tuple[int, int, int, int]:
        return cast(
            "tuple[int, int, int, int]",
            tuple(engine_scores[int(self._perm[s])] for s in range(4)),
        )

    def _hand_rng_seed(self, hand_index: int) -> int:
        """D-WP03A-1 refinement: the engine's internal RNG participates in
        mid-game resolution even with fully injected walls (verified:
        seedless instances diverge on identical action streams). Pin it per
        hand from the same named stream family as the walls so the complete
        WallSchedule still determines every stochastic outcome.
        """
        material = "|".join(
            (
                WALL_STREAM_NAME,
                self._schedule_id,
                self._schedule_digest.removeprefix("sha256:"),
                f"rng:{hand_index}",
            )
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    def _open_hand(
        self,
        tiles: Sequence[TileId],
        *,
        oya_engine: int,
        honba: int,
        kyotaku: int,
        scores_engine: Sequence[int],
        round_wind_int: int,
    ) -> None:
        env = riichienv.RiichiEnv(
            game_mode=riichienv.GameType.YON_HANCHAN,
            rule=riichienv.GameRule.default_tenhou(),
            seed=self._hand_rng_seed(max(0, self._hand_index)),
        )
        _ = env.reset(
            oya=oya_engine,
            wall=[int(t) for t in tiles],
            scores=list(scores_engine),
            honba=honba,
            kyotaku=kyotaku,
            round_wind=round_wind_int,
        )
        self._env = env
        self._cursor = 0
        assert self._rules is not None
        self._builder = ObservationBuilder(
            game_id=self._game_id,
            rules_id=self._rules.rules_id,
            rules_hash=_bridge_contracts.make_digest_text(self._rules_hash),
            action_table_hash=self._table.digest,
            expected_legal_mask_length=len(self._table.actions),
            event_schema_hash=_bridge_contracts.make_digest_text(self._event_schema_hash),
            packet_boundary_hash=_bridge_contracts.make_digest_text(self._packet_boundary_hash),
        )
        self._mode = None
        self._decision_seat = None
        self._pending = []
        self._views = {}
        self._offered_reset()
        self._pre_step_hands = {}
        self._dahai_cursor = [0, 0, 0, 0]
        initial_draws = sum(1 for e in env.mjai_log if e["type"] == "tsumo")
        self._draw_queue = (
            [env.drawn_tile] if initial_draws != 0 and env.drawn_tile is not None else []
        )

    def _offered_reset(self) -> None:
        self._last_discard = (None, None)

    def _emit(self, envelope: EventEnvelope) -> None:
        staging = self._staging
        if staging is not None:
            staging.append(envelope)
            return  # builder feed happens once at resolution-commit time
        self._events.append(envelope)
        assert self._builder is not None
        self._builder.append_visible(envelope)

    def _refresh_public_snapshot(
        self, *, phase: str | None = None, turn_actor: int | None = None
    ) -> None:
        assert self._builder is not None and self._env is not None and self._rules is not None
        dealer = self._inv[self._env.oya]
        wind_letter = {0: "E", 1: "S", 2: "W", 3: "N"}[self._env.round_wind]
        fields = {
            "decision_id": f"{self._game_id}:{len(self._events)}",
            "round_index": max(0, self._hand_index),
            "round_wind": _BAKAZE_TO_TILE_TYPE[wind_letter],
            "hand_number": self._env.kyoku_idx + 1,
            "seat_winds": seat_winds_for_dealer(dealer),
            "honba": self._env.honba,
            "riichi_sticks": self._env.riichi_sticks,
            "dealer": dealer,
            "scores": self._to_canonical_scores(self._env.scores()),
            "turn_actor": self._inv[self._env.current_player]
            if self._env.current_player >= 0
            else 0,
            "phase": (
                phase
                if phase is not None
                else ("discard_response" if self._mode == "window" else "draw_decision")
            ),
            "live_wall_tiles_remaining": live_wall_remaining(self._env),
            "ippatsu_active": tuple(self._ippatsu),
        }
        if turn_actor is not None:
            fields["turn_actor"] = turn_actor
        self._builder.update_public_state(**fields)

    def _context_for(self, seat: int, *, phase_override: str | None = None):
        assert self._env is not None
        from hydra2.contracts.action_table import ActionContext

        engine_pid = int(self._perm[seat])
        hand = list(self._env.hands[engine_pid])
        # ActionContext carries the FULL hand (drawn tile included): canonical
        # discard/tsumogiri of the drawn tile must encode against it. The
        # observation-side concealment stays drawn-stripped (SPEC 8).
        melds = []
        assert self._builder is not None
        for owner in range(4):
            melds.extend(self._builder_melds(owner))
        if phase_override is not None:
            phase: str = phase_override
        else:
            phase = "discard_response"
            if self._mode == "window":
                phase = "kan_response" if not self._window_opened_by_discard else "discard_response"
            elif self._mode == "draw":
                phase = "draw_decision"
        offered = self._last_discard[1]
        return ActionContext(
            actor=_bridge_contracts.make_seat(seat),
            action_table_hash=self._table.digest,
            phase=cast("Phase", phase),
            offered_tile=None if offered is None else _bridge_contracts.make_tile_id(offered),
            offered_by=None
            if self._last_discard[0] is None
            else _bridge_contracts.make_seat(self._last_discard[0]),
            own_concealed_tiles=tuple(_bridge_contracts.make_tile_id(t) for t in sorted(hand)),
            visible_melds=tuple(melds),
        )

    def _view_for(self, seat: int) -> tuple[tuple[CanonicalAction, ...], tuple[bool, ...]]:
        cached = self._views.get(seat)
        if cached is not None:
            return cached
        assert self._env is not None
        engine_pid = int(self._perm[seat])
        hand = list(self._env.hands[engine_pid])
        melds = []
        drawn = self._env.drawn_tile
        engine_legals = self._engine_observation(engine_pid).legal_actions()
        assert self._builder is not None
        for owner in range(4):
            melds.extend(self._builder_melds(owner))
        # The discarder's canonical seat: prefer the live window record; when
        # it is unset (e.g. views rebuilt mid-window), read the engine's own
        # ``last_discard`` (engine seat, physical tile) so claim sources stay
        # attributed to the true previous seat.
        offered_by = self._last_discard[0]
        if offered_by is None:
            engine_last = self._env.last_discard
            if engine_last is not None and engine_last[0] >= 0:
                offered_by = self._inv[engine_last[0]]
        actions, mask = legal_view(
            table=self._table,
            context=self._context_for(seat),
            engine_actions=engine_legals,
            drawn_tile=None if drawn is None else drawn,
            own_hand=hand,
            melds_of_actor=melds,
            offered_by=offered_by,
        )
        self._last_masks[seat] = mask
        self._views[seat] = (actions, mask)
        return actions, mask

    def _expected_actor_or_none(self) -> int | None:
        if self._terminal:
            return None
        if self._mode == "draw":
            return self._decision_seat
        if self._mode == "window":
            return min(self._pending)
        return None

    def _builder_melds(self, owner: int) -> list[Any]:
        melds: Any = getattr(self._builder, "_melds", None)
        if melds is None:  # pragma: no cover - contract change guard
            raise ContractError("ObservationBuilder meld cache is unavailable")
        return list(cast("Any", melds[owner]))

    def _expected_actor(self) -> int:
        actor = self._expected_actor_or_none()
        if actor is None:
            raise IllegalActionError("no decision is pending")
        return actor

    def _result(self, marker: int, *, next_actor: int | None) -> TransitionResult:
        return TransitionResult(
            events=tuple(self._events[marker:]),
            next_actor=None if next_actor is None else _bridge_contracts.make_seat(next_actor),
            terminal=self._terminal,
            raw_outcome=self._raw_outcome,
            state_digest=_bridge_contracts.make_digest_text(self._state_digest()),
        )

    def _state_digest(self) -> str:
        return str(
            state_digest(self._env, hand_index=max(0, self._hand_index), permutation=self._perm)
        )

    def _process_until_decision(self) -> None:
        while True:
            if self._terminal:
                return
            assert self._env is not None
            batch = self._env.mjai_log[self._cursor :]
            if len(batch) > 0:
                self._cursor = len(self._env.mjai_log)
                outcome = self._translate_batch(batch)
                if outcome == "terminal":
                    return
                if isinstance(outcome, dict):  # hand boundary carry parameters
                    self._reopen_hand(outcome)
                    continue
            decision = self._detect_decision()
            if decision is not None:
                return

    def _detect_decision(self) -> bool:
        assert self._env is not None
        if self._env.phase == int(riichienv.Phase.WaitResponse):
            responders = [
                seat
                for seat in range(4)
                if len(self._engine_observation(int(self._perm[seat])).legal_actions()) > 0
            ]
            if len(responders) > 0:
                self._pending = sorted(responders)
                self._mode = "window"
                self._decision_seat = None
                self._views = {}
                # D-WP03A-11: a superseded window's buffered decisions are
                # stale here - their engine slots died with the old offer.
                # Carrying them into the combined step of the NEW window
                # makes the engine reject them (Illegal Action ryukyoku).
                self._buffered = {}
                offered_ids: list[int] = []
                for seat in responders:
                    actions, _mask = self._view_for(seat)
                    offered_ids.extend(
                        int(
                            canonical_action_codec.encode(
                                a, table=self._table, context=self._context_for(seat)
                            )
                        )
                        for a in actions
                        if a.kind != "pass"
                    )
                self._window_offered_ids = tuple(sorted(set(offered_ids)))
                if self._window_opened_by_discard and self._events[-1].kind == "discard":
                    self._emit(
                        make_envelope(
                            game_id=self._game_id,
                            sequence=self._next_seq(),
                            kind="call_window",
                            visibility="public",
                            rules_hash=self._rules_hash,
                            schema_hash=self._event_schema_hash,
                        )
                    )
                return True
        current = self._env.current_player
        if (
            self._env.phase == int(riichienv.Phase.WaitAct)
            and current >= 0
            and len(self._engine_observation(current).legal_actions()) > 0
        ):
            seat = self._inv[current]
            self._mode = "draw"
            self._decision_seat = seat
            self._views = {}
            _ = self._view_for(seat)
            return True
        return False
