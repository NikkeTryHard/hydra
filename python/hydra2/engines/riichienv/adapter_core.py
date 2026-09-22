"""RiichiEnv simulator — public API, construction, apply, persistence.

Owns :class:`RiichiEnvExactSimulator` beside its only readers: SPEC 9
construction with the cyclic-seating gate, the legal-action and
observation surface, the canonical apply path with buffered response
windows, and snapshot/restore/clone persistence. The decision helpers
it calls live in :mod:`hydra2.engines.riichienv.adapter_step`, mjai
translation in :mod:`hydra2.engines.riichienv.adapter_events_a` and
:mod:`hydra2.engines.riichienv.adapter_events_b`, and the shared caches
and rules gates in :mod:`hydra2.engines.riichienv.adapter_identity`, so
each file stays inside the review-size ceiling.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.config import repo_root as repo_root
from hydra2.contracts.action_table import canonical_action_codec as canonical_action_codec
from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import DigestText as DigestText
from hydra2.contracts.common import (
    IllegalActionError as IllegalActionError,
)
from hydra2.contracts.common import InvalidActionError as InvalidActionError
from hydra2.contracts.common import Seat as Seat
from hydra2.contracts.common import TileId as TileId
from hydra2.contracts.common import UnsupportedRuleError as UnsupportedRuleError
from hydra2.contracts.event_packet import (
    build_packet_boundary_payload as build_packet_boundary_payload,
)
from hydra2.contracts.event_schema import (
    compute_event_schema_digest as compute_event_schema_digest,
)
from hydra2.contracts.observation_assembly import (
    VISIBILITY_VALIDATOR as VISIBILITY_VALIDATOR,
)
from hydra2.contracts.observation_assembly import (
    ObservationBuilder as ObservationBuilder,
)
from hydra2.contracts.observation_schema import (
    observation_schema_digest as observation_schema_digest,
)
from hydra2.engines.protocol import (
    SimulatorSnapshot as SimulatorSnapshot,
)
from hydra2.engines.protocol import (
    TransitionResult as TransitionResult,
)
from hydra2.engines.protocol import WallSchedule as WallSchedule
from hydra2.engines.protocol import validate_seat_permutation as validate_seat_permutation
from hydra2.engines.protocol import wall_schedule_digest as wall_schedule_digest
from hydra2.engines.riichienv.actions import (
    engine_matches_canonical as engine_matches_canonical,
)
from hydra2.engines.riichienv.actions import (
    legal_view as legal_view,
)
from hydra2.engines.riichienv.adapter_identity import (
    _AT as _AT,
)
from hydra2.engines.riichienv.adapter_identity import (
    _action_table as _action_table,
)
from hydra2.engines.riichienv.adapter_identity import _event_schema_hash as _event_schema_hash
from hydra2.engines.riichienv.adapter_identity import _rules_identity as _rules_identity
from hydra2.engines.riichienv.adapter_identity import _validate_rules as _validate_rules
from hydra2.engines.riichienv.adapter_step import AdapterStepMixin as AdapterStepMixin
from hydra2.engines.riichienv.events import make_envelope as make_envelope
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY as ENGINE_IDENTITY
from hydra2.engines.riichienv.state import (
    furiten_of as furiten_of,
)
from hydra2.engines.riichienv.state import (
    rules_identity_hash as rules_identity_hash,
)

if TYPE_CHECKING:
    from hydra2.contracts.action_model import CanonicalAction as CanonicalAction
    from hydra2.contracts.event_envelope import EventEnvelope as EventEnvelope
    from hydra2.contracts.observation_actor import ActorObservation as ActorObservation
    from hydra2.contracts.rules_manifest import RulesManifest as RulesManifest
    from hydra2.contracts.utility import RawOutcome as RawOutcome
    from hydra2.contracts.utility import SettlementFact as SettlementFact

__all__ = [
    "RiichiEnvExactSimulator",
]


class RiichiEnvExactSimulator(AdapterStepMixin):  # type: ignore[misc]
    """SPEC 9 exact simulator backed by pinned RiichiEnv 0.4.10."""

    def __init__(self) -> None:
        self._rules: RulesManifest | None = None
        self._rules_hash: str = ""
        self._perm: tuple[Seat, ...] = (
            _bridge_contracts.make_seat(0),
            _bridge_contracts.make_seat(1),
            _bridge_contracts.make_seat(2),
            _bridge_contracts.make_seat(3),
        )
        self._inv: list[int] = [0, 1, 2, 3]
        self._table = _action_table()
        self._event_schema_hash: str = _event_schema_hash()
        self._observation_schema_hash: str = str(observation_schema_digest())
        self._packet_boundary_hash: str = str(
            compute_event_schema_digest(build_packet_boundary_payload())
        )
        self._game_id: str = ""
        self._schedule_id: str = ""
        self._schedule_digest: str = ""
        self._schedule_tiles: tuple[TileId, ...] = ()
        self._env: riichienv.RiichiEnv | None = None
        self._cursor = 0
        self._hand_index = -1
        self._events: list[EventEnvelope] = []
        self._seq = 0
        self._applied: list[CanonicalAction] = []
        self._terminal = False
        self._raw_outcome: RawOutcome | None = None
        self._settlements: list[SettlementFact] = []
        self._starting_scores: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._builder: ObservationBuilder | None = None
        self._mode: str | None = None  # None | "draw" | "window"
        self._decision_seat: int | None = None
        self._pending: list[int] = []
        self._buffered: dict[int, tuple[CanonicalAction, object, int]] = {}
        self._window_opened_by_discard = False
        self._window_offered_ids: tuple[int, ...] = ()
        self._views: dict[int, tuple[tuple[CanonicalAction, ...], tuple[bool, ...]]] = {}
        self._last_masks: dict[int, tuple[bool, ...]] = {}
        self._last_discard: tuple[int | None, int | None] = (None, None)
        self._stamp_id: int | None = None
        self._stamp_actor: int | None = None
        self._stamp_kind: str | None = None
        self._staging: list[EventEnvelope] | None = None
        self._dahai_cursor: list[int] = [0, 0, 0, 0]
        # Exact physical ids of draws, captured at step time (drawn_tile is
        # current-state only and goes stale across batched translations).
        self._draw_queue: list[int] = []
        # Hands captured immediately before each engine step; a discard is
        # resolved exactly as (pre-step hand - post-step hand) whenever the
        # engine river lags behind the mjai log (D-WP03A-11). String parsing
        # cannot disambiguate the two plain copies of a suit five.
        self._pre_step_hands: dict[int, tuple[int, ...]] = {}
        # D-WP03A-11: exact physical id each seat adds in a kakan, captured
        # at apply() time. Engine 0.4.10 upgrades the prior pon meld IN PLACE
        # (same list slot), so by mjai-translation time the pon triple is no
        # longer recoverable from engine state alone.
        self._kakan_added: dict[int, int] = {}
        # D-WP03A-9: engine 0.4.10 exposes no runtime ippatsu property (stale
        # stub lists one); the adapter derives the flags from canonical events:
        # set at riichi_accepted, cleared by any meld interrupt and by the
        # declarer's own next discard, reset every hand.
        self._ippatsu: list[bool] = [False, False, False, False]

    # ------------------------------------------------------------------ API

    @property
    def identity(self) -> object:
        return ENGINE_IDENTITY

    def reset(
        self, *, rules: RulesManifest, wall: WallSchedule, seat_permutation: tuple[Seat, ...]
    ) -> None:
        _validate_rules(rules)
        perm = validate_seat_permutation(seat_permutation)
        # D-WP03A-9: only cyclic rotations keep canonical adjacency equal to
        # engine turn order (chi source = previous seat in BOTH spaces).
        if any(int(perm[seat]) != (perm[0] + seat) % 4 for seat in range(4)):
            raise UnsupportedRuleError(
                "non-cyclic seat_permutation; the reference seating space is "
                "engine-numbered and turn-cyclic, only rotations of (0,1,2,3) "
                "are supported"
            )
        rules_hash = _rules_identity(rules, str(rules_identity_hash(rules)))
        inv = [0, 0, 0, 0]
        for canonical_seat, engine_pid in enumerate(perm):
            inv[int(engine_pid)] = canonical_seat
        seed_material = of_canonical(
            {
                "rules_hash": rules_hash,
                "wall_digest": str(wall.digest),
                "seat_permutation": [int(s) for s in perm],
                "adapter_version": str(ENGINE_IDENTITY.adapter_version),
            }
        )
        self._rules = rules
        self._rules_hash = rules_hash
        self._perm = perm
        self._inv = inv
        self._game_id = f"hydra2-riichienv-{str(seed_material).removeprefix('sha256:')[:16]}"
        self._schedule_id = wall.schedule_id
        self._schedule_digest = str(wall.digest)
        self._events = []
        self._seq = 0
        self._applied = []
        self._terminal = False
        self._raw_outcome = None
        self._settlements = []
        self._schedule_tiles = tuple(wall.physical_tiles)
        self._hand_index = -1
        self._last_masks = {}
        self._stamp_id = None
        self._pre_step_hands = {}
        self._stamp_kind = None
        self._stamp_actor = None
        self._last_discard = (None, None)
        self._kakan_added = {}
        starting: tuple[int, int, int, int] = (
            rules.starting_points,
            rules.starting_points,
            rules.starting_points,
            rules.starting_points,
        )
        self._starting_scores = starting
        self._open_hand(
            wall.physical_tiles,
            oya_engine=int(perm[0]),
            honba=0,
            kyotaku=0,
            scores_engine=self._to_engine_order(starting),
            round_wind_int=0,
        )
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="game_start",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                round_index=0,
                scores=tuple(starting),
            )
        )
        self._process_until_decision()

    def legal_actions(self, actor: Seat) -> tuple[CanonicalAction, ...]:
        view = self._view_for(int(actor))
        return view[0]

    def legal_mask(self, actor: Seat) -> tuple[bool, ...]:
        return self._view_for(int(actor))[1]

    def actor_observation(self, actor: Seat) -> ActorObservation:
        seat_bound: Seat = _bridge_contracts.make_seat(int(actor))
        seat = int(seat_bound)
        mask = self._last_masks.get(seat)
        if mask is None:
            raise ContractError(f"seat {seat} has no recorded decision to observe")
        assert self._builder is not None and self._env is not None
        deciding = seat in self._views
        if deciding:
            engine_pid = int(self._perm[seat])
            hand = list(self._env.hands[engine_pid])
            drawn = self._env.drawn_tile
            if self._mode == "draw" and self._decision_seat == seat and drawn is not None:
                hand = [t for t in hand if t != drawn]
            self._builder.set_concealed_hand(seat_bound, hand)
            engine_legals = self._engine_observation(engine_pid).legal_actions()
            expanded = {a.kind for a in self._view_for(seat)[0]}
            self._builder.set_actor_state(
                seat_bound,
                furiten=furiten_of(self._env, engine_pid),
                can_tsumo="tsumo" in expanded,
                can_riichi="riichi_discard" in expanded,
            )
            _ = engine_legals
        self._refresh_public_snapshot()
        observation = self._builder.build(actor=seat_bound, legal_mask=mask)
        VISIBILITY_VALIDATOR.validate_observation(observation)
        for event in observation.visible_history:
            VISIBILITY_VALIDATOR.validate_event_for_actor(event, seat_bound)
        return observation

    def apply(self, action: CanonicalAction) -> TransitionResult:
        if self._terminal or self._env is None:
            raise IllegalActionError("simulation is terminal; no further actions exist")
        expected = self._expected_actor()
        if action.actor != expected:
            raise IllegalActionError(
                f"action actor {int(action.actor)} != expected decision actor {expected}"
            )
        actions, _mask = self._view_for(expected)
        try:
            stamp = int(
                canonical_action_codec.encode(
                    action, table=self._table, context=self._context_for(expected)
                )
            )
        except (ContractError, ValueError) as exc:
            raise IllegalActionError(f"action rejected by codec context: {exc}") from exc
        if all(a != action for a in actions):
            raise IllegalActionError(
                f"action {action.kind}:{action.tile} is not in the legal mask slot set (id {stamp})"
            )
        engine_pid = int(self._perm[expected])
        engine_legals = self._engine_observation(engine_pid).legal_actions()
        candidates = frozenset(
            t for t in riichienv.check_riichi_candidates(list(self._env.hands[engine_pid]))
        )
        match = next(
            (
                eng
                for eng in engine_legals
                if engine_matches_canonical(eng, action, riichi_candidate_tiles=candidates)
            ),
            None,
        )
        if match is None:
            raise InvalidActionError(
                f"canonical action {action.kind}:{action.tile} has no engine counterpart"
            )

        marker = len(self._events)
        self._stamp_id = stamp
        self._stamp_actor = expected
        self._stamp_kind = str(action.kind)
        if str(action.kind) == "kakan":
            assert action.tile is not None
            self._kakan_added[engine_pid] = int(action.tile)
            # D-WP04A-FIX8 (chankan, commit-then-collect): stepping the kakan
            # alone leaves the engine in WaitResponse; _detect_decision then
            # opens a kan_response window through the EXISTING claim machinery
            # (real legals, real views, buffered responses, one combined
            # response step). RiichiEnv auto-passes only when the window is
            # never observed - which cannot happen here because the tail
            # _process_until_decision always runs detection.
            self._window_opened_by_discard = False
        if self._mode == "window":
            # Freeze the canonical id NOW: after the combined step executes,
            # the claimer's concealed hand no longer owns the consumed tiles
            # (they moved into the meld), so a late re-encode would fail.
            buffered_id = int(
                canonical_action_codec.encode(
                    action, table=self._table, context=self._context_for(expected)
                )
            )
            self._buffered[expected] = (action, match, buffered_id)
            self._pending = [s for s in self._pending if s != expected]
            _ = self._views.pop(expected, None)
            if len(self._pending) > 0:
                return self._result(marker, next_actor=min(self._pending))
            combined: dict[int, riichienv.Action] = {
                int(self._perm[s]): eng
                for s, (_a, eng, _bid) in self._buffered.items()
                if isinstance(eng, riichienv.Action)
            }
            if len(combined) != len(self._buffered):
                raise ContractError("window buffered a 3-player action in a 4-player game")
            for buffered_action, _eng, _bid in self._buffered.values():
                self._applied.append(buffered_action)
            _pre = len(self._env.mjai_log)
            self._capture_hands_for_step(combined)
            _ = self._env.step(cast("dict[int, Any]", combined))
            self._note_step_draws(_pre)
            self._flush_window_resolution()
        else:
            self._applied.append(action)
            if action.kind == "riichi_discard":
                # D-WP03A-9: reuse the engine's own RIICHI slot (tile=None);
                # synthesizing Action(_AT.RIICHI, 0, []) corrupts engine
                # internal riichi state and aborts the following hand.
                declaration = match
                _pre = len(self._env.mjai_log)
                self._capture_hands_for_step({engine_pid: declaration})
                _ = self._env.step({engine_pid: declaration})
                self._note_step_draws(_pre)
                _ = self._consume_new_events()
                inner = self._engine_observation(engine_pid).legal_actions()
                # D-WP04A-FIX6: a bare riichi_discard (tile=None) declares and
                # tosses the just-drawn tile; otherwise the inner matcher must
                # find the exact declared physical tile.
                if action.tile is None:
                    drawn = getattr(self._env, "drawn_tile", None)
                    declared_tile = None if drawn is None else int(drawn)
                else:
                    declared_tile = int(action.tile)
                discard_match = next(
                    (
                        eng
                        for eng in inner
                        if int(eng.action_type) == int(_AT.DISCARD) and eng.tile == declared_tile
                    ),
                    None,
                )
                if discard_match is None:  # pragma: no cover - engine contract
                    raise InvalidActionError("engine lost the declared discard candidate")
                _pre_inner = len(self._env.mjai_log)
                self._capture_hands_for_step({engine_pid: discard_match})
                _ = self._env.step(cast("dict[int, Any]", {engine_pid: discard_match}))
                self._note_step_draws(_pre_inner)
            else:
                _pre = len(self._env.mjai_log)
                self._capture_hands_for_step({engine_pid: match})
                _ = self._env.step(cast("dict[int, Any]", {engine_pid: match}))
                self._note_step_draws(_pre)
            if str(action.kind) == "kakan":
                # FIX8: the kan_response window opens right after this step;
                # responders need offered_tile/offered_by = (owner, added).
                assert action.tile is not None
                self._last_discard = (expected, int(action.tile))
            else:
                self._mode = None
                self._decision_seat = None
                _ = self._views.pop(expected, None)
                self._last_discard = (None, None)
        self._process_until_decision()
        return self._result(marker, next_actor=self._expected_actor_or_none())

    def snapshot(self) -> SimulatorSnapshot:
        self._require_ready()
        assert self._rules is not None
        snapshot_hash: DigestText = _bridge_contracts.make_digest_text(self._rules_hash)
        return SimulatorSnapshot(
            engine_name=ENGINE_IDENTITY.name,
            engine_version=ENGINE_IDENTITY.version,
            rules_hash=snapshot_hash,
            game_id=self._game_id,
            seat_permutation=self._perm,
            schedule_id=self._schedule_id,
            schedule_physical_tiles=self._schedule_tiles,
            applied_actions=tuple(self._applied),
            rules_manifest=self._rules,
        )

    def restore(self, snapshot: SimulatorSnapshot) -> None:
        wall = WallSchedule(
            schedule_id=snapshot.schedule_id,
            physical_tiles=snapshot.schedule_physical_tiles,
            digest=wall_schedule_digest(snapshot.schedule_id, snapshot.schedule_physical_tiles),
        )
        self.reset(
            rules=snapshot.rules_manifest,
            wall=wall,
            seat_permutation=snapshot.seat_permutation,
        )
        for replay_action in snapshot.applied_actions:
            _ = self.apply(replay_action)
        if self._game_id != snapshot.game_id:  # pragma: no cover - identity drift
            raise ContractError(
                f"restored simulator game_id {self._game_id!r} != snapshot {snapshot.game_id!r}"
            )

    @property
    def _engine(self) -> riichienv.RiichiEnv:
        """Narrowed engine handle; gameplay paths all run post-``reset``."""
        if self._env is None:
            raise ContractError("simulator is not initialized; call reset() first")
        return self._env

    def _engine_observation(self, engine_pid: int) -> riichienv.Observation:
        """Per-seat observation (the wheel exposes this; its stub omits it)."""
        return cast(
            "riichienv.Observation",
            cast("Any", self._engine).get_observation(engine_pid),
        )

    def clone(self) -> RiichiEnvExactSimulator:
        cls = type(self)
        replica = cls.__new__(cls)
        for name, value in self.__dict__.items():
            setattr(replica, name, value)
        replica._env = (
            cast(
                "riichienv.RiichiEnv",
                cast("Any", self._engine).clone(),
            )
            if self._env is not None
            else None
        )
        # P-B13: hot-path clone uses shallow copy + copy-on-write for builder.
        # copy.deepcopy(builder) recursively walks immutable EventEnvelope/
        # VisibleMeld objects; copy.copy + manual list/dict shims isolates the
        # mutable containers (histories/discards/melds/public) while sharing the
        # immutable payloads. Preserves deep-equal semantics but avoids O(N)
        # pickle-style traversal on every MCTS expansion (hundreds clones/search).
        # Evidence: https://docs.python.org/3/library/copy.html
        # - copy.copy(x): shallow copy, new container shares references
        # - copy.deepcopy(x): recursively copies, expensive for frozen dataclasses
        builder = self._builder
        if builder is None:
            replica._builder = None  # type: ignore[assignment]  # reason: replica via __new__; builder Optional at runtime
        else:
            try:
                new_builder = copy.copy(builder)
                # Histories/discards/melds are tuple[list[...]]; copy each inner list
                new_builder._histories = tuple(list(h) for h in builder._histories)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._discards = tuple(list(d) for d in builder._discards)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._melds = tuple(list(m) for m in builder._melds)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._concealed = list(builder._concealed)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._drawn = list(builder._drawn)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._dora = list(builder._dora)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._furiten = list(builder._furiten)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._can_tsumo = list(builder._can_tsumo)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._can_riichi = list(builder._can_riichi)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._pending_discard = list(builder._pending_discard)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._riichi_states = list(builder._riichi_states)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                new_builder._public = dict(builder._public)  # type: ignore[attr-defined]  # reason: external builder lacks stubs; attrs exist at runtime
                replica._builder = new_builder  # type: ignore[assignment]  # reason: replica via __new__; builder assignment valid at runtime
            except Exception:  # pragma: no cover - defensive fallback
                # why-broad: shallow-copy shim touches untyped builder attrs;
                # any shape falls back to deepcopy.
                replica._builder = copy.deepcopy(builder)  # type: ignore[assignment]  # reason: fallback preserves semantics; checker cannot narrow __new__ replica
        replica._events = list(self._events)
        replica._applied = list(self._applied)
        replica._settlements = list(self._settlements)
        replica._pending = list(self._pending)
        replica._buffered = dict(self._buffered)
        replica._views = dict(self._views)
        replica._last_masks = dict(self._last_masks)
        return replica
