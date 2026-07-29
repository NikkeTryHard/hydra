"""SPEC 9 Engine Protocol: identity, wall schedule, transition, snapshots.

Canonical home of the ``ExactSimulator`` surface (WP-03A owns this module).
Dataclasses are closed and validated; the protocol itself is structural so
adapters need not inherit from anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    SchemaVersion,
    Seat,
    TileId,
    make_digest_text,
    make_schema_version,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.rules import RULES_ID as _CANONICAL_RULES_ID  # noqa: F401

if TYPE_CHECKING:
    from hydra2.contracts.action import CanonicalAction
    from hydra2.contracts.event import EventEnvelope
    from hydra2.contracts.observation import ActorObservation
    from hydra2.contracts.rules import RulesManifest
    from hydra2.contracts.utility import RawOutcome

__all__ = [
    "WALL_TILE_COUNT",
    "EngineIdentity",
    "ExactSimulator",
    "SimulatorSnapshot",
    "TransitionResult",
    "WallSchedule",
]

#: A complete WallSchedule carries exactly one full live+dead wall.
WALL_TILE_COUNT = 136


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    """Pinned engine identity (SPEC 9); verified at import by each adapter."""

    name: str
    version: str
    adapter_version: SchemaVersion
    source_revision: str | None
    environment_hash: DigestText

    def __post_init__(self) -> None:
        if len(self.name) == 0 or not isinstance(self.name, str):
            raise ContractError("EngineIdentity.name must be a non-empty str")
        if len(self.version) == 0 or not isinstance(self.version, str):
            raise ContractError("EngineIdentity.version must be a non-empty str")
        object.__setattr__(self, "adapter_version", make_schema_version(self.adapter_version))
        if self.source_revision is not None and (
            not isinstance(self.source_revision, str) or len(self.source_revision) == 0
        ):
            raise ContractError("EngineIdentity.source_revision must be None or a non-empty str")
        object.__setattr__(self, "environment_hash", make_digest_text(self.environment_hash))


def wall_schedule_digest(schedule_id: str, physical_tiles: tuple[TileId, ...]) -> DigestText:
    """sha256 over canonical bytes of the digest-free schedule document."""
    return of_canonical(
        {"schedule_id": schedule_id, "physical_tiles": [int(t) for t in physical_tiles]}
    )


@dataclass(frozen=True, slots=True)
class WallSchedule:
    """Deterministic physical tile supply (SPEC 9).

    ``physical_tiles`` is the exact engine draw-order deck: a permutation of
    the 136 physical tile ids. The digest binds schedule identity so derived
    semantic streams and snapshot records can reference it.
    """

    schedule_id: str
    physical_tiles: tuple[TileId, ...]
    digest: DigestText

    def __post_init__(self) -> None:
        if len(self.schedule_id) == 0 or not isinstance(self.schedule_id, str):
            raise ContractError("WallSchedule.schedule_id must be a non-empty str")
        tiles = tuple(make_tile_id(t) for t in self.physical_tiles)
        if len(tiles) != WALL_TILE_COUNT:
            raise ContractError(
                f"WallSchedule must carry exactly {WALL_TILE_COUNT} physical tiles, "
                f"got {len(tiles)}"
            )
        if sorted(tiles) != list(range(WALL_TILE_COUNT)):
            raise ContractError("WallSchedule.physical_tiles must permute 0..135 exactly once")
        object.__setattr__(self, "physical_tiles", tiles)
        expected = wall_schedule_digest(self.schedule_id, tiles)
        recorded = make_digest_text(self.digest)
        if recorded != expected:
            raise ContractError(
                f"WallSchedule.digest mismatch: recorded {recorded} != recomputed {expected}"
            )


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of applying one canonical action (SPEC 9)."""

    events: tuple[EventEnvelope, ...]
    next_actor: Seat | None
    terminal: bool
    raw_outcome: RawOutcome | None
    state_digest: DigestText


@dataclass(frozen=True, slots=True)
class SimulatorSnapshot:
    """Privileged serialized simulator state (SPEC 9).

    The reference adapter stores the full deterministic replay inputs rather
    than engine internals: replaying ``applied_actions`` over
    ``(rules_hash, seat_permutation, wall schedule)`` regenerates byte-equal
    state because every stochastic outcome derives from injected walls.
    """

    engine_name: str
    engine_version: str
    rules_hash: DigestText
    game_id: str
    seat_permutation: tuple[Seat, ...]
    schedule_id: str
    schedule_physical_tiles: tuple[TileId, ...]
    applied_actions: tuple[CanonicalAction, ...]
    rules_manifest: RulesManifest

    def __post_init__(self) -> None:
        perms = tuple(make_seat(s) for s in self.seat_permutation)
        if sorted(perms) != [0, 1, 2, 3]:
            raise ContractError("snapshot seat_permutation must permute seats 0..3")
        object.__setattr__(self, "seat_permutation", perms)
        object.__setattr__(
            self,
            "schedule_physical_tiles",
            tuple(make_tile_id(t) for t in self.schedule_physical_tiles),
        )
        object.__setattr__(self, "rules_hash", make_digest_text(self.rules_hash))


@runtime_checkable
class ExactSimulator(Protocol):
    """SPEC 9 exact-simulator protocol (structural)."""

    @property
    def identity(self) -> EngineIdentity: ...

    def reset(
        self, *, rules: RulesManifest, wall: WallSchedule, seat_permutation: tuple[Seat, ...]
    ) -> None: ...

    def actor_observation(self, actor: Seat) -> ActorObservation: ...

    def legal_actions(self, actor: Seat) -> tuple[CanonicalAction, ...]: ...

    def legal_mask(self, actor: Seat) -> tuple[bool, ...]: ...

    def apply(self, action: CanonicalAction) -> TransitionResult: ...

    def snapshot(self) -> SimulatorSnapshot: ...

    def restore(self, snapshot: SimulatorSnapshot) -> None: ...

    def clone(self) -> ExactSimulator: ...


def validate_seat_permutation(seat_permutation: tuple[Seat, ...]) -> tuple[Seat, ...]:
    """Validate ``seat_permutation[canonical_seat] = engine_player_index``.

    Structural check only (a bijection of 0..3). D-WP03A-9 additionally
    constrains simulators to CYCLIC ROTATIONS of the identity seating:
    canonical seat numbering equals engine numbering and turn order is
    engine-cyclic, so reversed/shuffled seatings would break the canonical
    action vocabulary's adjacency (chi source = previous seat). Adapters
    MUST reject non-rotations with UnsupportedRuleError before any game
    starts; duplicate-wall seat protocols rotate agents at the eval layer.
    """
    if not isinstance(seat_permutation, tuple) or len(seat_permutation) != 4:
        raise ContractError("seat_permutation must be a tuple of four entries")
    normalized = tuple(make_seat(int(s)) for s in seat_permutation)
    if sorted(normalized) != [0, 1, 2, 3]:
        raise ContractError(f"seat_permutation must permute 0..3 exactly once, got {normalized!r}")
    return normalized


def seat_permutation_literal(kind: Literal["identity", "shift1", "shift2", "shift3", "reverse"]):
    """Named permutation builders used by tests and schedules (canary support)."""
    identities = {
        "identity": (0, 1, 2, 3),
        "shift1": (1, 2, 3, 0),
        "shift2": (2, 3, 0, 1),
        "shift3": (3, 0, 1, 2),
        "reverse": (3, 2, 1, 0),
    }
    try:
        return tuple(make_seat(s) for s in identities[kind])
    except KeyError as exc:  # pragma: no cover - literal guard
        raise ContractError(f"unknown seat permutation kind {kind!r}") from exc
