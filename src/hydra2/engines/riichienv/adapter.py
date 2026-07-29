"""RiichiEnvExactSimulator: the WP-03A reference ExactSimulator.

Design decisions recorded here (full rationale in work_packages/WP-03A):

D-WP03A-1 Per-hand engines with adapter-driven chaining. RiichiEnv 0.4.8
    honours ``reset(wall=...)`` for the first hand only; later hands come from
    engine-internal RNG (verified: identical injected walls diverge at kyoku
    2). The adapter therefore plays each hand on a fresh engine instance fed
    with ``reset(oya/honba/kyotaku/scores/round_wind/wall=...)``. Carry
    parameters between hands are read from the ENGINE's own native
    ``start_kyoku`` advance (emitted inside the same step batch that closed
    the previous hand), so renchan/honba/stick/agari-yame/sudden-death/tobi
    logic stays engine-faithful while walls stay fully injected. The engine's
    RNG-dealt follow-up hand is discarded unplayed; nothing derived from it
    reaches any public or private surface. Continuation walls derive from the
    pinned WallSchedule via :mod:`hydra2.engines.riichienv.walls` (named
    stream ``hydra2.wall_continuation_v1``); the seed parameter is never
    touched on formal paths.

D-WP03A-5 Buffered response windows. RiichiEnv resolves claims from ONE
    simultaneous ``step`` over all responders (verified: partial submission
    resolves immediately and silently drops other claimants). The adapter
    buffers individual responder decisions and submits one combined step;
    ``call_window`` opens a discard-offered window and one server-private
    ``call_resolved`` closes it ahead of the outcome envelopes (accepted id
    taken from what the engine actually executed). Kan-offered windows
    (chankan) emit no window pair because the grammar routes ``kakan -> ron``
    directly.

D-WP03A-6 Multi-ron attribution. Concurrent hora events of one resolution
    merge into a single ``ron`` envelope: the first winner owns
    actor/action-id (matching the engine's stick rule), deltas sum, and
    SettlementFact entries carry every winner.

D-WP03A-8 Hand-scoped observation builders. One ObservationBuilder per hand
    keeps ``visible_history`` inside the current hand's public stream and
    avoids unresettable per-seat caches leaking across hands; determinism is
    unaffected because hands chain through injected walls.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2.artifacts.digest import of_canonical
from hydra2.config import repo_root
from hydra2.contracts.action import (
    ACTION_TABLE_RELPATH,
    CanonicalAction,
    canonical_action_codec,
    load_action_table,
)
from hydra2.contracts.common import (
    ContractError,
    IllegalActionError,
    InvalidActionError,
    Seat,
    TileId,
    UnsupportedRuleError,
    make_digest_text,
    make_seat,
    make_sequence_no,
    make_tile_id,
)
from hydra2.contracts.event import (
    EVENT_SCHEMA_RELPATH,
    build_packet_boundary_payload,
    compute_event_schema_digest,
    parse_event_schema,
)
from hydra2.contracts.observation import (
    VISIBILITY_VALIDATOR,
    ObservationBuilder,
    observation_schema_digest,
)
from hydra2.engines.protocol import (
    SimulatorSnapshot,
    TransitionResult,
    WallSchedule,
    validate_seat_permutation,
    wall_schedule_digest,
)
from hydra2.engines.riichienv import events
from hydra2.engines.riichienv.actions import engine_matches_canonical, legal_view
from hydra2.engines.riichienv.events import make_delta, make_envelope, meld_delta_value, reason_kind
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY
from hydra2.engines.riichienv.state import (
    furiten_of,
    live_wall_remaining,
    raw_outcome_from_final,
    rules_identity_hash,
    seat_winds_for_dealer,
    settlement_facts_from_deltas,
    state_digest,
)
from hydra2.engines.riichienv.walls import WALL_STREAM_NAME, derive_hand_wall

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.action import ActionTable, Phase
    from hydra2.contracts.event import EventEnvelope
    from hydra2.contracts.observation import ActorObservation
    from hydra2.contracts.rules import RulesManifest
    from hydra2.contracts.utility import RawOutcome, SettlementFact
__all__ = ["RiichiEnvExactSimulator"]

_BAKAZE_TO_TILE_TYPE = {"E": 27, "S": 28, "W": 29, "N": 30}
_AT = riichienv.ActionType

_TABLE_CACHE: dict[str, ActionTable] = {}
_EVENT_SCHEMA_CACHE: dict[str, str] = {}


def _action_table() -> ActionTable:
    root = str(repo_root())
    if root not in _TABLE_CACHE:
        _TABLE_CACHE[root] = load_action_table(Path(root) / ACTION_TABLE_RELPATH)
    return _TABLE_CACHE[root]


def _event_schema_hash() -> str:
    root = str(repo_root())
    if root not in _EVENT_SCHEMA_CACHE:
        document: dict[str, Any] = cast(
            "dict[str, Any]", parse_event_schema((Path(root) / EVENT_SCHEMA_RELPATH).read_bytes())
        )
        payload: Any = document["payload"]
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ContractError("event schema artifact lacks a digest")
        payload_dict: dict[str, Any] = cast("dict[str, Any]", payload)
        _EVENT_SCHEMA_CACHE[root] = str(cast("Any", payload_dict["digest"]))
    return _EVENT_SCHEMA_CACHE[root]


def _validate_rules(rules: RulesManifest) -> None:
    """Structural support gate; failures happen BEFORE any game starts."""
    if rules.players != 4:
        raise UnsupportedRuleError(
            f"reference adapter supports 4-player games, got {rules.players}"
        )
    if rules.match_length != "hanchan":
        raise UnsupportedRuleError(
            f"reference adapter pins match_length='hanchan', got {rules.match_length!r}"
        )
    if tuple(rules.red_tile_ids) != (16, 52, 88):
        raise UnsupportedRuleError(f"unsupported red-five encoding {rules.red_tile_ids!r}")
    if rules.kuikae_policy != "forbidden":
        raise UnsupportedRuleError(
            f"RiichiEnv hard-forbids kuikae; manifest declares {rules.kuikae_policy!r}"
        )
    for entry in rules.adapter_compatibility:
        if entry.adapter_id == "riichienv" and entry.status not in ("supported", "qualified"):
            raise UnsupportedRuleError(
                f"manifest marks adapter riichienv as {entry.status!r}; refusing to run"
            )


def _rules_identity(manifest: RulesManifest, recomputed: str) -> str:
    """Published artifact bytes win when present (D-WP03A-4 refinement).

    The published configs/rules file is the authority its digest was recorded
    from; the payload recompute stays as the fallback for manifests without a
    published artifact.
    """
    published = Path(repo_root()) / "configs" / "rules" / f"{manifest.rules_id}.json"
    if not published.is_file():
        return recomputed
    return "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()


class RiichiEnvExactSimulator:
    """SPEC 9 exact simulator backed by pinned RiichiEnv 0.4.8."""

    def __init__(self) -> None:
        self._rules: RulesManifest | None = None
        self._rules_hash: str = ""
        self._perm: tuple[Seat, ...] = (
            make_seat(0),
            make_seat(1),
            make_seat(2),
            make_seat(3),
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
        # at apply() time. Engine 0.4.8 upgrades the prior pon meld IN PLACE
        # (same list slot), so by mjai-translation time the pon triple is no
        # longer recoverable from engine state alone.
        self._kakan_added: dict[int, int] = {}
        # D-WP03A-9: engine 0.4.8 exposes no runtime ippatsu property (stale
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
        seat = int(make_seat(int(actor)))
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
            self._builder.set_concealed_hand(make_seat(seat), hand)
            engine_legals = self._engine_observation(engine_pid).legal_actions()
            expanded = {a.kind for a in self._view_for(seat)[0]}
            self._builder.set_actor_state(
                make_seat(seat),
                furiten=furiten_of(self._env, engine_pid),
                can_tsumo="tsumo" in expanded,
                can_riichi="riichi_discard" in expanded,
            )
            _ = engine_legals
        self._refresh_public_snapshot()
        observation = self._builder.build(actor=make_seat(seat), legal_mask=mask)
        VISIBILITY_VALIDATOR.validate_observation(observation)
        for event in observation.visible_history:
            VISIBILITY_VALIDATOR.validate_event_for_actor(event, make_seat(seat))
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
        except ContractError as exc:
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
        return SimulatorSnapshot(
            engine_name=ENGINE_IDENTITY.name,
            engine_version=ENGINE_IDENTITY.version,
            rules_hash=make_digest_text(self._rules_hash),
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
        replica = RiichiEnvExactSimulator.__new__(RiichiEnvExactSimulator)
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
                replica._builder = copy.deepcopy(builder)  # type: ignore[assignment]  # reason: fallback preserves semantics; checker cannot narrow __new__ replica
        replica._events = list(self._events)
        replica._applied = list(self._applied)
        replica._settlements = list(self._settlements)
        replica._pending = list(self._pending)
        replica._buffered = dict(self._buffered)
        replica._views = dict(self._views)
        replica._last_masks = dict(self._last_masks)
        return replica

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
            rules_hash=make_digest_text(self._rules_hash),
            action_table_hash=self._table.digest,
            expected_legal_mask_length=len(self._table.actions),
            event_schema_hash=make_digest_text(self._event_schema_hash),
            packet_boundary_hash=make_digest_text(self._packet_boundary_hash),
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
        from hydra2.contracts.action import ActionContext

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
            actor=make_seat(seat),
            action_table_hash=self._table.digest,
            phase=cast("Phase", phase),
            offered_tile=None if offered is None else make_tile_id(offered),
            offered_by=None if self._last_discard[0] is None else make_seat(self._last_discard[0]),
            own_concealed_tiles=tuple(make_tile_id(t) for t in sorted(hand)),
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

    def _expected_actor_or_none(self) -> int | None:
        if self._terminal:
            return None
        if self._mode == "draw":
            return self._decision_seat
        if self._mode == "window":
            return min(self._pending)
        return None

    def _result(self, marker: int, *, next_actor: int | None) -> TransitionResult:
        return TransitionResult(
            events=tuple(self._events[marker:]),
            next_actor=None if next_actor is None else make_seat(next_actor),
            terminal=self._terminal,
            raw_outcome=self._raw_outcome,
            state_digest=make_digest_text(self._state_digest()),
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
                actor=make_seat(dealer),
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
                actor=make_seat(actor),
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
                actor=make_seat(actor),
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
                actor=make_seat(actor),
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
                actor=make_seat(actor),
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
                actor=make_seat(actor),
                public_delta=(
                    make_delta(("riichi_states", actor), "set", "accepted"),
                    make_delta(("riichi_sticks",), "increment", 1),
                    make_delta(("ippatsu", actor), "set", True),
                ),
            )
        )
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
                actor=make_seat(actor),
                tile=called,
                action_id=action_id,
                source_seat=None if source is None else make_seat(source),
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
                actor=make_seat(actor),
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
            except ContractError:
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
                actor=make_seat(winner),
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
        pre = self._to_canonical_scores(self._engine.scores())
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
        final_scores = self._to_canonical_scores(self._engine.scores())
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
                scores=final_scores,
                public_delta=tuple(deltas),
            )
        )
        self._refresh_public_snapshot(phase="round_end")

    def _on_end_game(self, event: Any) -> None:
        del event
        final_scores = self._to_canonical_scores(self._engine.scores())
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="game_end",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                round_index=max(0, self._hand_index),
                scores=final_scores,
                reason="hanchan_complete",
                public_delta=(make_delta(("scores",), "set", list(final_scores)),),
            )
        )
        assert self._rules is not None
        self._raw_outcome = raw_outcome_from_final(
            final_scores=final_scores,
            starting_scores=self._starting_scores,
            settlements=self._settlements,
            rules_id=self._rules.rules_id,
            rules_hash=make_digest_text(self._rules_hash),
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
            outcome = self._translate_batch(batch)
        finally:
            self._staging = None
        accepted: list[int] = []
        for envelope in staging:
            if envelope.kind in ("chi", "pon", "daiminkan", "ron") and (
                envelope.payload.action_id is not None
            ):
                accepted = [int(envelope.payload.action_id)]
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
            actor=make_seat(actor),
            tile=make_tile_id(tile),
            called_tile=None,
            consumed_tiles=(),
            source_seat=None,
            declares_riichi=False,
            metadata=(),
        )
        # D-WP03A-10: engine-auto discards without a live applied-action link
        # get a deterministic id from a minimal synthetic context (the tile
        # itself as the concealed set). Ids stay pure functions of the action.
        context = self._context_for(actor, phase_override="draw_decision")
        if tile not in {int(t) for t in context.own_concealed_tiles}:
            from hydra2.contracts.action import ActionContext

            context = ActionContext(
                actor=make_seat(actor),
                action_table_hash=self._table.digest,
                phase="draw_decision",
                offered_tile=None,
                offered_by=None,
                own_concealed_tiles=(make_tile_id(tile),),
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
        # translation time (engine 0.4.8 upgrades that meld IN PLACE, so
        # scanning for a separate 3-tile pon would fail after any later meld).
        meld_tiles = self._latest_meld_tiles(engine_pid)
        added = self._kakan_added.get(engine_pid)
        if added is None:  # pragma: no cover - apply-time capture guarantees presence
            raise ContractError(
                f"kakan by engine seat {engine_pid}: no captured added tile; "
                "translation ran without a preceding canonical apply"
            )
        del self._kakan_added[engine_pid]
        consumed = tuple(sorted(t for t in meld_tiles if t != added))
        action_id = self._stamp_if(actor, "kakan")
        self._window_opened_by_discard = False
        self._emit(
            make_envelope(
                game_id=self._game_id,
                sequence=self._next_seq(),
                kind="kakan",
                visibility="public",
                rules_hash=self._rules_hash,
                schema_hash=self._event_schema_hash,
                actor=make_seat(actor),
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
        from hydra2.engines.riichienv.tiles import physical_of

        return int(physical_of(mjai_pai))

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
