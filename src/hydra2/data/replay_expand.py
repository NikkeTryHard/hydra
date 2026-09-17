"""Streaming-first replay expansion: GameRecord -> DecisionRow (WP-14).

Ephemeral rows carry the identical contracts as the materialized parquet path
(contract lock): validate/quarantine-shape lineage, actor/privileged split,
wall-bound provenance enabling wall-disjoint manifests plus the leakage check,
the actor-privileged firewall, counter-free deterministic derivation (no RNG),
``(5,)`` dora, and positional decision ids that compose with a resumable
cursor (prefix catch-up is a slice; see :func:`iter_microbatches`).

Mechanism: reset one :class:`RiichiEnvExactSimulator` with a
:class:`WallSchedule` built from ``game.wall_tiles``, walk the framed MJAI
events in log order, and at each logged player action capture the actor view
(:meth:`actor_observation` + :meth:`legal_mask`) with ``chosen_action_id``
encoded from the log's actual played tile. Fail closed: unmapped event kinds
and illegal logged actions raise; nothing is ever synthesized.

Judgment calls recorded explicitly (never silent picks):

- Window auto-pass is mechanical, not synthesis: MJAI logs never record
  passes, but the engine opens a claim window after some discards. Passes
  applied to advance such windows emit no rows and no labels; only logged
  actions (discards, claims, kans, wins) produce DecisionRows.
- Physical-copy resolution: several physical copies share one MJAI string
  (``5m`` is three copies; only the red five is distinct). When the log names
  a string with several legal copies, the smallest physical id wins. The
  choice is deterministic and documented; tsumogiri and red fives are exact.
- Identity seating only: ``seat_permutation`` is fixed to ``(0, 1, 2, 3)`` so
  MJAI actor indices equal canonical seats. Rotated seatings are rejected by
  the adapter itself; replay does not remap actors.
- Truncated logs are honest: replay stops at ``end_game`` without requiring
  simulator terminal (a streamed game may end mid-hanchan). A log that still
  carries row-events after simulator terminal raises instead of inventing
  outcomes.
- No new dependencies; YAML/output placement stays under artifact-root by
  construction (this module writes no files and hard-codes no paths).

Perf (keep the GPU fed): :class:`ReplayExpander` owns one engine instance
reused across games via ``reset``; rules/manifest/adapter digests are loaded
once; per-decision work reuses the adapter's cached legal view (one
``legal_actions`` call feeds selection, observation, and encoding). Batch
callers slice with :func:`iter_microbatches` without re-decode. Throughput
(``decisions/sec``) is reported by tests as informational output, never as
an assert.

Bridge-routing boundary (Wave 1 R3, named keep-reason): this per-decision
walk stays Python — no bridge pyfn produces :class:`DecisionRow` content.
The Rust walk (``hydra2_replay_rs.expand_games`` /
``replay_game_planes[_wall]``) serves decision identity plus tensor planes
only (``chosen_action_id`` columns and feature blobs); it never materializes
the per-row ``actor_observation`` JSON, the live :class:`ActorObservation`
handoff (:func:`stash_live_observation`), or the ``derivation_hash`` /
``adapter_hash`` / ``observation_hash`` seals :func:`expand_game` commits.
Routing the oracle through planes would drop those fields and break
byte-exactness, so the training feed's Rust path
(:mod:`hydra2.training.stream_expand`) consumes planes while this module
remains the parity anchor. The per-event scans
(:func:`_count_row_decisions`, :func:`_skip_index`, :func:`_final_scores`,
the walk itself) likewise stay Python: no pyfn covers them. Field gap for a
future pyfn: full DecisionRow materialization (observation JSON plus
derivation/adapter seals) from staged planes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.action_table import canonical_action_codec
from hydra2.contracts.common import (
    ContractError,
    IllegalActionError,
    TileId,
)
from hydra2.data.parquet import DecisionRow, PrivilegedRow, validate_privileged_ranks
from hydra2.engines.protocol import WallSchedule, wall_schedule_digest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from hydra2.contracts.action_model import CanonicalAction
    from hydra2.contracts.observation_actor import ActorObservation
    from hydra2.contracts.rules_manifest import RulesManifest
    from hydra2.data.decode import GameRecord
    from hydra2.engines.riichienv.adapter_core import RiichiEnvExactSimulator

__all__ = [
    "ReplayExpander",
    "expand_game",
    "expand_privileged_rows",
    "iter_microbatches",
    "pop_live_observation",
    "stash_live_observation",
]

# ---------------------------------------------------------------------------
# Event vocabulary (closed; anything else fails closed).
# ---------------------------------------------------------------------------

#: Framing aliases accepted at the game boundaries (mirrors decode.py).
_START_TYPES = frozenset({"start_game", "startGame", "game_start", "start"})
_END_TYPES = frozenset({"end_game", "endGame", "game_end", "end"})

#: Non-decision events: consumed for ordering/round tracking, never rows.
_SKIP_TYPES = frozenset(
    {
        "start_kyoku",
        "tsumo",
        "dora",
        "reach_accepted",
        "ryukyoku",
        "end_kyoku",
        *(_START_TYPES),
    }
)

#: One row per occurrence (``reach`` collapses with its following ``dahai``).
_ROW_TYPES = frozenset({"dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "hora"})

_CLAIM_TYPES = frozenset({"chi", "pon", "daiminkan"})

_RULES_RELPATH = Path("configs") / "rules" / "tenhou_4p_hanchan_v1.json"

_RULES_CACHE: RulesManifest | None = None
_ADAPTER_HASH_CACHE: str | None = None
_EXPANDER_CACHE: ReplayExpander | None = None
#: Live-observation handoff: per-decision_id cache of the validated
#: :class:`ActorObservation` built at capture time, so the encoder consumes
#: live objects instead of re-serializing/re-parsing/re-validating row
#: dicts. Keyed by row ``decision_id`` (``{game_id}:d{seq}``, unique per
#: game), consume-once via :func:`pop_live_observation` so steady-state size
#: tracks the in-flight buffer, never the epoch. A miss (parquet rows,
#: cross-process rows not yet re-stashed, over-cap drops) falls back to the
#: validating parse path — never wrong, only slower. Row CONTENT is
#: untouched: the cache rides out-of-band, never inside the row dict.
_LIVE_OBSERVATION_CAP = 65536
_LIVE_OBSERVATIONS: dict[str, ActorObservation] = {}


def stash_live_observation(decision_id: str, observation: ActorObservation) -> None:
    """Offer one built observation for later consume-once encoder pickup."""
    if len(_LIVE_OBSERVATIONS) >= _LIVE_OBSERVATION_CAP:
        return
    _LIVE_OBSERVATIONS[decision_id] = observation


def pop_live_observation(decision_id: str) -> ActorObservation | None:
    """Take the stashed observation for ``decision_id`` (``None`` on miss)."""
    return _LIVE_OBSERVATIONS.pop(decision_id, None)


def _load_rules() -> RulesManifest:
    """Load the pinned Tenhou rules manifest once (same source as validate)."""
    global _RULES_CACHE
    cached = _RULES_CACHE
    if cached is not None:
        return cached
    from hydra2.config import repo_root
    from hydra2.contracts.rules_manifest import rules_manifest_from_payload

    rules_path = repo_root() / _RULES_RELPATH
    if not rules_path.is_file():
        try:
            import importlib.resources as _ir

            rules_path = Path(
                str(_ir.files("hydra2") / "configs" / "rules" / "tenhou_4p_hanchan_v1.json")
            )
        except Exception:  # why-broad: rules-path probe failed; retry repo
            rules_path = repo_root() / _RULES_RELPATH
    raw: object = json.loads(rules_path.read_bytes())
    if not isinstance(raw, dict):
        raise ContractError("rules document must be an object")
    payload_raw: object = raw.get("payload", raw)
    if not isinstance(payload_raw, dict):
        raise ContractError("rules payload must be an object")
    manifest = rules_manifest_from_payload(cast("dict[str, object]", payload_raw))
    _RULES_CACHE = manifest
    return manifest


def _adapter_hash() -> str:
    """Machine-stable adapter digest (identity minus environment/source)."""
    global _ADAPTER_HASH_CACHE
    cached = _ADAPTER_HASH_CACHE
    if cached is not None:
        return cached
    from hydra2.engines.riichienv.identity import ENGINE_IDENTITY

    digest = str(
        of_canonical(
            {
                "engine": ENGINE_IDENTITY.name,
                "engine_version": ENGINE_IDENTITY.version,
                "adapter_version": str(ENGINE_IDENTITY.adapter_version),
            }
        )
    )
    _ADAPTER_HASH_CACHE = digest
    return digest


def _schedule_for(game_id: str, wall_tiles: tuple[int, ...]) -> WallSchedule:
    """Build the game wall; raises instead of synthesizing a missing wall."""
    if len(wall_tiles) != 136:
        raise ContractError(f"wall_tiles must carry 136 tiles, got {len(wall_tiles)}")
    physical = tuple(_bridge_contracts.make_tile_id(t) for t in wall_tiles)
    schedule_id = f"replay-{game_id}"
    return WallSchedule(
        schedule_id=schedule_id,
        physical_tiles=physical,
        digest=wall_schedule_digest(schedule_id, physical),
    )


def _event_kind(event: dict[str, object]) -> str:
    kind = event.get("type")
    if not isinstance(kind, str) or kind == "":
        raise ContractError(f"mjai event without a string type: {event!r}")
    return kind


def _require_actor(event: dict[str, object], *, where: str) -> int:
    actor = event.get("actor")
    if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor <= 3:
        raise ContractError(f"{where}: event actor must be 0..3, got {actor!r}")
    return actor


def _mjai_strings(tile_ids: Sequence[TileId]) -> list[str]:
    from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

    return [tiles.mjai_string_of(int(t)) for t in tile_ids]


def _wall_id_for(game_id: str, wall_tiles: tuple[int, ...] | None) -> str | None:
    if wall_tiles is None:
        return None
    return str(_schedule_for(game_id, wall_tiles).digest)


def _final_scores(events: Sequence[dict[str, object]]) -> tuple[object, ...]:
    """Extract the terminal 4-score quad from the closing end_game event."""
    for event in reversed(events):
        kind = event.get("type")
        if isinstance(kind, str) and kind in _END_TYPES:
            for key in ("scores", "final_scores"):
                scores = event.get(key)
                if isinstance(scores, (list, tuple)) and len(scores) == 4:
                    return tuple(scores)
            raise ContractError("end_game event carries no 4-score quad (scores/final_scores)")
    raise ContractError("game has no closing end_game event for privileged ranks")


def _count_row_decisions(events: Sequence[dict[str, object]]) -> int:
    """Count row-producing decisions without a simulator (reach collapses)."""
    count = 0
    idx = 0
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = _event_kind(event)
        if kind in _END_TYPES:
            break
        if kind == "reach":
            nxt = _skip_index(events, idx + 1)
            if nxt is None:
                raise ContractError("reach event without a following dahai")
            following = events[nxt]
            if not isinstance(following, dict):
                raise ContractError("reach event must be followed by its declaration dahai")
            if _event_kind(following) != "dahai":
                raise ContractError("reach event must be followed by its declaration dahai")
            count += 1
            idx = nxt + 1
            continue
        if kind in _ROW_TYPES:
            count += 1
        elif kind not in _SKIP_TYPES:
            raise ContractError(f"unmapped mjai event type {kind!r} at index {idx}")
        idx += 1
    return count


def _skip_index(events: Sequence[dict[str, object]], start: int) -> int | None:
    """First index >= start that is a decision, reach, or end (None if none)."""
    idx = start
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = _event_kind(event)
        if kind in _END_TYPES or kind in _ROW_TYPES or kind == "reach":
            return idx
        if kind not in _SKIP_TYPES:
            raise ContractError(f"unmapped mjai event type {kind!r} at index {idx}")
        idx += 1
    return None


class ReplayExpander:
    """Owns one engine instance reused across games via ``reset``.

    Not thread-safe: create one per thread for parallel expansion. Module
    :func:`expand_game` reuses a process-wide instance for the serial path.
    """

    __slots__ = ("_rules", "_sim")

    def __init__(self) -> None:
        from hydra2.engines.riichienv.adapter_core import RiichiEnvExactSimulator

        self._sim: RiichiEnvExactSimulator = RiichiEnvExactSimulator()
        self._rules: RulesManifest = _load_rules()

    def expand(self, game: GameRecord, *, split: str = "train") -> list[DecisionRow]:
        """Expand one game into actor DecisionRows (see :func:`expand_game`)."""
        if split == "":
            raise ContractError("split must be a non-empty string")
        wall_tiles = game.wall_tiles
        if wall_tiles is None:
            raise ContractError(f"game {game.game_id!r} has no wall_tiles; cannot replay")
        sim = self._sim
        sim.reset(
            rules=self._rules,
            wall=_schedule_for(game.game_id, tuple(wall_tiles)),
            seat_permutation=(
                _bridge_contracts.make_seat(0),
                _bridge_contracts.make_seat(1),
                _bridge_contracts.make_seat(2),
                _bridge_contracts.make_seat(3),
            ),
        )
        events = game.events
        rows: list[DecisionRow] = []
        idx = 0
        total = len(events)
        round_idx = 0
        kyoku_seen = 0
        while idx < total:
            event = events[idx]
            if not isinstance(event, dict):
                raise ContractError(f"mjai event [{idx}] must be an object")
            ev = event
            kind = _event_kind(ev)
            if kind in _END_TYPES:
                break
            if kind == "start_kyoku":
                round_idx = kyoku_seen
                kyoku_seen += 1
                idx += 1
                continue
            if kind in _SKIP_TYPES:
                idx += 1
                continue
            if kind == "reach" or kind in _ROW_TYPES:
                idx = self._step_decision(game, ev, kind, idx, round_idx, rows, split=split)
                continue
            raise ContractError(f"unmapped mjai event type {kind!r} at index {idx}")
        if sim._terminal:
            rest = _skip_index(events, idx)
            if rest is not None:
                raise ContractError("log carries row-events past simulator terminal")
        return rows

    def _step_decision(
        self,
        game: GameRecord,
        ev: dict[str, object],
        kind: str,
        idx: int,
        round_idx: int,
        rows: list[DecisionRow],
        *,
        split: str,
    ) -> int:
        sim = self._sim
        expected = sim._expected_actor_or_none()
        if expected is None:
            if sim._terminal:
                raise ContractError("log carries row-events past simulator terminal")
            raise ContractError(f"no decision pending for mjai event [{idx}] ({kind})")
        mode = sim._mode
        if mode == "window":
            return self._step_window(game, ev, kind, idx, round_idx, rows, split=split)
        if mode != "draw":
            raise ContractError(f"simulator in unexpected mode {mode!r} at event [{idx}]")
        return self._step_draw(game, ev, kind, idx, round_idx, rows, split=split)

    def _step_window(
        self,
        game: GameRecord,
        ev: dict[str, object],
        kind: str,
        idx: int,
        round_idx: int,
        rows: list[DecisionRow],
        *,
        split: str,
    ) -> int:
        sim = self._sim
        pending = sorted(sim._pending)
        claim = self._window_claim(ev, kind, pending)
        if claim is None:
            # No logged claim here: mechanically pass every responder so the
            # engine advances. No rows, no labels — never synthesized.
            for seat in pending:
                self._apply_pass(seat)
            return idx
        seat, action = claim
        for earlier in pending:
            if earlier == seat:
                break
            self._apply_pass(earlier)
        self._emit_row(game, seat, action, len(rows), round_idx, rows, split=split)
        applied = self._select_for_apply(seat, action)
        # Engine advance is the effect; result discarded.
        _ = sim.apply(applied)
        return idx + 1

    def _window_claim(
        self, ev: dict[str, object], kind: str, pending: list[int]
    ) -> tuple[int, CanonicalAction] | None:
        """Map a window claim event to (seat, legal action), or None to pass."""
        sim = self._sim
        if kind in _CLAIM_TYPES:
            actor = _require_actor(ev, where=kind)
            if actor not in pending:
                raise IllegalActionError(
                    f"logged {kind} by seat {actor} outside the claim window {pending}"
                )
            legals = sim.legal_actions(_bridge_contracts.make_seat(actor))
            return actor, self._match_claim(actor, kind, ev, legals)
        if kind == "hora" and not bool(ev.get("tsumo", False)):
            actor = _require_actor(ev, where="hora")
            if actor not in pending:
                return None
            legals = sim.legal_actions(_bridge_contracts.make_seat(actor))
            return actor, self._match_ron(actor, ev, legals)
        return None

    def _step_draw(
        self,
        game: GameRecord,
        ev: dict[str, object],
        kind: str,
        idx: int,
        round_idx: int,
        rows: list[DecisionRow],
        *,
        split: str,
    ) -> int:
        sim = self._sim
        expected = sim._expected_actor_or_none()
        assert expected is not None
        if kind in _CLAIM_TYPES or (kind == "hora" and not bool(ev.get("tsumo", False))):
            raise IllegalActionError(f"logged {kind} outside any claim window at index {idx}")
        if kind == "reach":
            events = game.events
            nxt = _skip_index(events, idx + 1)
            if nxt is None:
                raise ContractError("reach event without a following dahai")
            following = events[nxt]
            if not isinstance(following, dict):
                raise ContractError("reach event must be followed by its declaration dahai")
            dev = following
            if _event_kind(dev) != "dahai":
                raise ContractError("reach event must be followed by its declaration dahai")
            actor = _require_actor(ev, where="reach")
            dactor = _require_actor(dev, where="dahai")
            if actor != expected or dactor != expected or actor != dactor:
                raise IllegalActionError(
                    f"reach/dahai actor {actor}/{dactor} != expected decision seat {expected}"
                )
            legals = sim.legal_actions(_bridge_contracts.make_seat(expected))
            action = self._match_riichi(expected, dev, legals)
            self._emit_row(game, expected, action, len(rows), round_idx, rows, split=split)
            # Engine advance is the effect; result discarded.
            _ = sim.apply(self._select_for_apply(expected, action))
            return nxt + 1
        if kind == "dahai":
            actor = _require_actor(ev, where="dahai")
            if actor != expected:
                raise IllegalActionError(f"dahai by seat {actor} != expected {expected} [{idx}]")
            legals = sim.legal_actions(_bridge_contracts.make_seat(expected))
            action = self._match_dahai(expected, ev, legals, idx=idx)
            self._emit_row(game, expected, action, len(rows), round_idx, rows, split=split)
            # Engine advance is the effect; result discarded.
            _ = sim.apply(self._select_for_apply(expected, action))
            return idx + 1
        if kind in ("ankan", "kakan"):
            actor = _require_actor(ev, where=kind)
            if actor != expected:
                raise IllegalActionError(f"{kind} by seat {actor} != expected {expected} [{idx}]")
            legals = sim.legal_actions(_bridge_contracts.make_seat(expected))
            action = self._match_kan(expected, kind, ev, legals, idx=idx)
            self._emit_row(game, expected, action, len(rows), round_idx, rows, split=split)
            # Engine advance is the effect; result discarded.
            _ = sim.apply(self._select_for_apply(expected, action))
            return idx + 1
        if kind == "hora":
            actor = _require_actor(ev, where="hora")
            if actor != expected:
                raise IllegalActionError(f"hora by seat {actor} != expected {expected} [{idx}]")
            if not bool(ev.get("tsumo", False)):
                raise IllegalActionError(f"logged ron outside any claim window at index {idx}")
            legals = sim.legal_actions(_bridge_contracts.make_seat(expected))
            action = self._match_tsumo(expected, ev, legals, idx=idx)
            self._emit_row(game, expected, action, len(rows), round_idx, rows, split=split)
            # Engine advance is the effect; result discarded.
            _ = sim.apply(self._select_for_apply(expected, action))
            return idx + 1
        raise ContractError(f"unmapped mjai event type {kind!r} at index {idx}")

    def _apply_pass(self, seat: int) -> None:
        sim = self._sim
        legals = sim.legal_actions(_bridge_contracts.make_seat(seat))
        for action in legals:
            if action.kind == "pass":
                # Mechanical pass advances the engine; result discarded.
                _ = sim.apply(action)
                return
        raise ContractError(f"seat {seat} has no pass action in the claim window")

    def _select_for_apply(self, seat: int, action: CanonicalAction) -> CanonicalAction:
        """Return the cached legal twin to apply (identity for the adapter)."""
        sim = self._sim
        legals = sim.legal_actions(_bridge_contracts.make_seat(seat))
        for legal in legals:
            if legal == action:
                return legal
        return action

    def _emit_row(
        self,
        game: GameRecord,
        seat: int,
        action: CanonicalAction,
        seq: int,
        round_idx: int,
        rows: list[DecisionRow],
        *,
        split: str,
    ) -> None:
        sim = self._sim
        observation = sim.actor_observation(_bridge_contracts.make_seat(seat))
        context = sim._context_for(seat)
        chosen_id = int(canonical_action_codec.encode(action, table=sim._table, context=context))
        mask = tuple(observation.legal_mask)
        if not 0 <= chosen_id < len(mask) or mask[chosen_id] is not True:
            raise IllegalActionError(
                f"chosen action id {chosen_id} is not legal for seat {seat} at decision {seq}"
            )
        obs_hash = str(observation.observation_hash)
        decision_id = f"{game.game_id}:d{seq:04d}"
        # Hand the validated live object to the encoder out of band; row
        # dict bytes below are identical either way, so skipping the
        # re-serialize/re-parse/re-validate round trip cannot diverge output.
        stash_live_observation(decision_id, observation)
        wall_digest = sim._schedule_digest
        derivation = str(
            of_canonical(
                {
                    "game_id": game.game_id,
                    "decision_id": decision_id,
                    "observation_hash": obs_hash,
                    "chosen_action_id": chosen_id,
                    "wall_digest": wall_digest,
                    "adapter_hash": _adapter_hash(),
                }
            )
        )
        rows.append(
            DecisionRow(
                game_id=game.game_id,
                round_id=f"{game.game_id}:h{round_idx:02d}",
                decision_id=decision_id,
                seat=seat,
                source_object_id=game.object_id,
                split=split,
                rules_hash=str(observation.rules_hash),
                adapter_hash=_adapter_hash(),
                observation_hash=obs_hash,
                action_table_hash=str(observation.action_table_hash),
                derivation_hash=derivation,
                actor_observation=observation.to_json(),
                chosen_action_id=chosen_id,
                privileged_label_ref=decision_id,
            )
        )

    # -- log -> legal matching (fail closed; deterministic copy resolution) --

    def _match_dahai(
        self, seat: int, ev: dict[str, object], legals: tuple[CanonicalAction, ...], *, idx: int
    ) -> CanonicalAction:
        pai = ev.get("pai")
        if not isinstance(pai, str) or pai == "":
            raise ContractError(f"dahai event [{idx}] carries no pai string")
        if bool(ev.get("tsumogiri", False)):
            matches = [a for a in legals if a.kind == "tsumogiri" and _matches_pai(a.tile, pai)]
            if len(matches) == 0:
                raise IllegalActionError(f"tsumogiri {pai!r} illegal for seat {seat} [{idx}]")
            return min(matches, key=lambda a: cast("int", a.tile))
        matches = [a for a in legals if a.kind == "discard" and _matches_pai(a.tile, pai)]
        if len(matches) == 0:
            raise IllegalActionError(f"discard {pai!r} illegal for seat {seat} [{idx}]")
        return min(matches, key=lambda a: cast("int", a.tile))

    def _match_riichi(
        self, seat: int, ev: dict[str, object], legals: tuple[CanonicalAction, ...]
    ) -> CanonicalAction:
        pai = ev.get("pai")
        if not isinstance(pai, str) or pai == "":
            raise ContractError("riichi declaration dahai carries no pai string")
        matches = [a for a in legals if a.kind == "riichi_discard" and _matches_pai(a.tile, pai)]
        if len(matches) == 0:
            raise IllegalActionError(f"riichi discard {pai!r} illegal for seat {seat}")
        return min(matches, key=lambda a: cast("int", a.tile))

    def _match_claim(
        self, seat: int, kind: str, ev: dict[str, object], legals: tuple[CanonicalAction, ...]
    ) -> CanonicalAction:
        pai = ev.get("pai")
        consumed = ev.get("consumed")
        target = ev.get("target")
        if not isinstance(pai, str) or pai == "":
            raise ContractError(f"logged {kind} carries no pai string")
        if not isinstance(consumed, (list, tuple)):
            raise ContractError(f"logged {kind} carries no consumed tile list")
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 3:
            raise ContractError(f"logged {kind} carries no valid target seat")
        consumed_seq: Sequence[object] = consumed
        wanted = sorted(str(t) for t in consumed_seq)
        matches = []
        for action in legals:
            if action.kind != kind or action.called_tile is None:
                continue
            if _tile_string(int(action.called_tile)) != pai:
                continue
            if sorted(_mjai_strings(tuple(action.consumed_tiles))) != wanted:
                continue
            if action.source_seat is not None and int(action.source_seat) != target:
                continue
            matches.append(action)
        if len(matches) == 0:
            raise IllegalActionError(f"logged {kind} {pai!r} is illegal for seat {seat}")
        return min(matches, key=_consumed_tiles_key)

    def _match_kan(
        self,
        seat: int,
        kind: str,
        ev: dict[str, object],
        legals: tuple[CanonicalAction, ...],
        *,
        idx: int,
    ) -> CanonicalAction:
        if kind == "ankan":
            consumed = ev.get("consumed")
            if not isinstance(consumed, (list, tuple)):
                raise ContractError(f"logged ankan [{idx}] carries no consumed tile list")
            consumed_seq: Sequence[object] = consumed
            wanted = sorted(str(t) for t in consumed_seq)
            matches = [
                a
                for a in legals
                if a.kind == "ankan" and sorted(_mjai_strings(tuple(a.consumed_tiles))) == wanted
            ]
            if len(matches) == 0:
                raise IllegalActionError(f"logged ankan is illegal for seat {seat} at [{idx}]")
            return min(matches, key=_consumed_tiles_key)
        pai = ev.get("pai")
        if not isinstance(pai, str) or pai == "":
            raise ContractError(f"logged kakan [{idx}] carries no pai string")
        matches = [a for a in legals if a.kind == "kakan" and _matches_pai(a.tile, pai)]
        if len(matches) == 0:
            raise IllegalActionError(f"kakan {pai!r} illegal for seat {seat} at [{idx}]")
        return min(matches, key=lambda a: cast("int", a.tile))

    def _match_tsumo(
        self, seat: int, ev: dict[str, object], legals: tuple[CanonicalAction, ...], *, idx: int
    ) -> CanonicalAction:
        pai = ev.get("pai")
        matches = [a for a in legals if a.kind == "tsumo"]
        if isinstance(pai, str) and pai != "":
            narrowed = [a for a in matches if _matches_pai(a.tile, pai)]
            if len(narrowed) > 0:
                matches = narrowed
        if len(matches) == 0:
            raise IllegalActionError(f"tsumo win illegal for seat {seat} at [{idx}]")
        return min(matches, key=_optional_tile_key)

    def _match_ron(
        self, seat: int, ev: dict[str, object], legals: tuple[CanonicalAction, ...]
    ) -> CanonicalAction:
        pai = ev.get("pai")
        matches = [a for a in legals if a.kind == "ron"]
        if isinstance(pai, str) and pai != "":
            narrowed = [a for a in matches if _matches_pai(a.tile, pai)]
            if len(narrowed) > 0:
                matches = narrowed
        if len(matches) == 0:
            raise IllegalActionError(f"logged ron is illegal for seat {seat}")
        return min(matches, key=_optional_tile_key)


def _consumed_tiles_key(action: CanonicalAction) -> tuple[int, ...]:
    """Deterministic copy-resolution order: ascending physical ids."""
    return tuple(sorted(int(t) for t in action.consumed_tiles))


def _optional_tile_key(action: CanonicalAction) -> int:
    """Tile-less wins first, else ascending physical id."""
    return -1 if action.tile is None else int(action.tile)


def _tile_string(tile: int) -> str:
    from hydra2_replay_rs import tiles  # pyrefly: ignore[missing-import]

    return tiles.mjai_string_of(tile)


def _matches_pai(tile: int | None, pai: str) -> bool:
    """Exact MJAI-string match for one physical tile (red fives distinct)."""
    return tile is not None and _tile_string(tile) == pai


def _get_expander() -> ReplayExpander:
    global _EXPANDER_CACHE
    cached = _EXPANDER_CACHE
    if cached is None:
        cached = ReplayExpander()
        _EXPANDER_CACHE = cached
    return cached


def expand_game(game: GameRecord, *, split: str = "train") -> list[DecisionRow]:
    """Expand one game into actor-visible DecisionRows.

    Reuses the process-wide :class:`ReplayExpander` (one engine instance via
    ``reset``); for parallel work construct one expander per thread instead.
    """
    return _get_expander().expand(game, split=split)


def expand_privileged_rows(
    game: GameRecord, *, split: str = "train", wall_id: str | None = None
) -> list[PrivilegedRow]:
    """Build privileged placement rows whose decision ids match :func:`expand_game`.

    Ranks derive from the terminal ``end_game`` scores via
    :func:`ranks_from_final_scores` (ties raise there — Tenhou-resolve first).
    ``wall_id``/``split`` ride inside the opaque label for the existing
    privileged writer; ``wall_id`` defaults to the game wall digest when
    ``game.wall_tiles`` is present and is omitted otherwise (never invented).
    """
    from hydra2.belief.oracle_join import ranks_from_final_scores

    if split == "":
        raise ContractError("split must be a non-empty string")
    events = list(game.events)
    count = _count_row_decisions(events)
    scores = _final_scores(events)
    ranks = ranks_from_final_scores(list(scores))
    ordered = validate_privileged_ranks(list(ranks), f"{game.game_id}:privileged")
    resolved_wall = wall_id
    if resolved_wall is None:
        resolved_wall = _wall_id_for(game.game_id, game.wall_tiles)
    elif resolved_wall == "":
        raise ContractError("wall_id must be a non-empty string")
    rows: list[PrivilegedRow] = []
    for seq in range(count):
        decision_id = f"{game.game_id}:d{seq:04d}"
        label: dict[str, object] = {"ranks": list(ordered), "split": split}
        if resolved_wall is not None:
            label["wall_id"] = resolved_wall
        rows.append(PrivilegedRow(decision_id=decision_id, privileged_label=label))
    return rows


def iter_microbatches(rows: Sequence[DecisionRow], microbatch: int) -> Iterator[list[DecisionRow]]:
    """Yield microbatch slices sharing the decoded rows (no re-decode).

    Slices reference the same frozen rows; only the pointer list per batch is
    allocated. Positional order is preserved, so prefix catch-up resume is
    ``rows[k:]`` without re-expansion.
    """
    if isinstance(microbatch, bool) or not isinstance(microbatch, int) or microbatch <= 0:
        raise ContractError(f"microbatch must be a positive int, got {microbatch!r}")
    total = len(rows)
    start = 0
    while start < total:
        yield list(rows[start : start + microbatch])
        start += microbatch
