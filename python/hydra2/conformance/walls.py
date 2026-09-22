"""WP-04A deterministic wall construction for edge-case corpus cases.

A corpus case pins a COMPLETE 136-tile :class:`~hydra2.engines.protocol.WallSchedule`.
The builder here assembles such walls from explicit per-seat hand multisets plus
live-draw and dead-wall placements, using one global cycle-safe assignment so
that no placement silently corrupts another seat's tiles (probe-verified
failure mode).

Layout facts of RiichiEnv 0.4.10 under ``reset(wall=...)`` (probed with an
identity wall; attestation metadata carries the layout proof):

- haipai: seat ``k`` draws indices ``[4k..4k+3, 16+4k..19+4k, 32+4k..35+4k,
  48+k]`` (13 tiles); the dealer's 14th tile is live draw index 52;
- live draws consume indices 52, 53, 54, ... in strict turn order;
- rinshan replacement draws pop indices 135, 134, 133, ... descending;
- the first kan-dora indicator sits at index 131 and each further kan reveals
  the indicator two slots lower (129, 127, ...); ura markers occupy the
  interleaved slots below each indicator.

Physical tile identity follows SPEC §4.1: ``type == id // 4`` with man types
0..8, pin 9..17, sou 18..26 and honors E/S/W/N/P/F/C at 27..33; red fives are
ids 16 (5mr), 52 (5pr) and 88 (5sr).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import TileId

if TYPE_CHECKING:
    from collections.abc import Mapping

HAIPAI_SLOTS: tuple[int, ...] = _bridge_contracts.HAIPAI_SLOTS


__all__ = [
    "DEALER_DRAW_INDEX",
    "FIRST_DORA_INDICATOR_INDEX",
    "HAIPAI_SLOTS",
    "LIVE_DRAW_BASE",
    "RINSHAN_TOP_INDEX",
    "TILE_C",
    "TILE_E",
    "TILE_F",
    "TILE_N",
    "TILE_P",
    "TILE_S",
    "TILE_W",
    "build_wall",
    "copies",
    "haipai_index",
    "type_id",
]


#: First index consumed by a live (non-rinshan) draw.
LIVE_DRAW_BASE: int = _bridge_contracts.LIVE_DRAW_BASE
#: Dealer's opening 14th tile.
DEALER_DRAW_INDEX: int = _bridge_contracts.DEALER_DRAW_INDEX
#: Index of the initially revealed dora indicator.
FIRST_DORA_INDICATOR_INDEX: int = _bridge_contracts.FIRST_DORA_INDICATOR_INDEX
#: Highest rinshan tile index (draws descend from here).
RINSHAN_TOP_INDEX: int = _bridge_contracts.RINSHAN_TOP_INDEX

# Honor physical ids used throughout case definitions (first copy of each).
TILE_E: int = _bridge_contracts.TILE_E
TILE_S: int = _bridge_contracts.TILE_S
TILE_W: int = _bridge_contracts.TILE_W
TILE_N: int = _bridge_contracts.TILE_N
TILE_P: int = _bridge_contracts.TILE_P  # haku
TILE_F: int = _bridge_contracts.TILE_F  # hatsu
TILE_C: int = _bridge_contracts.TILE_C  # chun


def type_id(physical: int) -> int:
    """Logical tile type of a physical id (SPEC §4.1)."""
    return _bridge_contracts.type_id(physical)


def copies(tile_type: int):
    """The four physical ids of ``tile_type`` in ascending order."""
    return _bridge_contracts.copies(tile_type)


def haipai_index(seat: int, position: int) -> int:
    """Wall index dealt as ``position`` (0-based) of ``seat``'s opening hand."""
    return _bridge_contracts.haipai_index(seat, position)


class WallPlan:
    """Accumulates index -> physical-id requirements, then resolves them."""

    def __init__(self) -> None:
        self._requirements: dict[int, set[int]] = {}
        self._used_copies: dict[int, int] = {}

    def require(self, index: int, physical: int) -> None:
        """Pin ``index`` to physical tile ``physical``."""
        if not 0 <= index <= 135:
            raise ValueError(f"wall index {index} out of range")
        self._requirements.setdefault(index, set()).add(physical)

    def require_hand(self, seat: int, hand: Mapping[int, int]) -> list[int]:
        """Pin an EXACT multiset ``{physical_id: count}`` onto haipai slots.

        D-WP04A-FIX5 (root-cause fix): keys are literal physical ids and are
        preserved verbatim - the previous type-normalising allocator dealt
        lowest-free copies instead, silently swapping same-type tiles across
        seats and breaking any scenario logic keyed on specific physical
        tiles (aka identity, chankan waits, furiten rivers). Per-type supply
        (four copies) is validated across the WHOLE plan at resolve time.
        Fewer than 13 tiles leaves the remaining slots free. Returns the
        exact indices used, sorted.
        """
        wanted: list[int] = []
        for physical, count in sorted(hand.items()):
            pid = physical
            if not 0 <= pid <= 135:
                raise ValueError(f"seat {seat}: physical id {pid} out of range")
            wanted.extend([pid] * count)
        if len(wanted) > 13:
            raise ValueError(f"seat {seat} hand exceeds 13 tiles: {len(wanted)}")
        if len(wanted) == 0:
            return []
        slots = [haipai_index(seat, position) for position in range(len(wanted))]
        for slot, physical in zip(slots, sorted(wanted), strict=True):
            self.require(slot, physical)
        return sorted(slots)

    def resolve(self) -> tuple[TileId, ...]:
        """Solve the assignment and return the full 136-tile wall."""
        supply: dict[int, int] = {}
        for ids in self._requirements.values():
            for physical in ids:
                supply[physical // 4] = supply.get(physical // 4, 0) + 1
        over = {t: n for t, n in supply.items() if n > 4}
        if len(over) != 0:
            raise ValueError(f"tile types exceed four-copy supply: {over}")
        conflicts = {index: ids for index, ids in self._requirements.items() if len(ids) > 1}
        if len(conflicts) != 0:
            raise ValueError(f"conflicting wall requirements: {conflicts}")
        wall = list(range(136))
        # Resolve by moving each demanded physical id into its slot with a
        # cycle-safe swap walk; repeated passes settle chains.
        for _pass in range(200):
            misplaced = [
                (index, next(iter(ids)))
                for index, ids in sorted(self._requirements.items())
                if wall[index] != next(iter(ids))
            ]
            if len(misplaced) == 0:
                break
            for index, physical in misplaced:
                if wall[index] == physical:
                    continue
                source = wall.index(physical)
                wall[source], wall[index] = wall[index], physical
        else:  # pragma: no cover - defensive
            raise ValueError("wall assignment did not converge")
        remaining = {
            index: next(iter(ids))
            for index, ids in self._requirements.items()
            if wall[index] != next(iter(ids))
        }
        if len(remaining) != 0:  # pragma: no cover - defensive
            raise ValueError(f"unresolved wall slots: {remaining}")
        return tuple(TileId(t) for t in wall)


def build_wall(
    *,
    hands: Mapping[int, Mapping[int, int]],
    dealer_draw: int | None = None,
    live_draws: Mapping[int, int] | None = None,
    dead_wall: Mapping[int, int] | None = None,
) -> tuple[TileId, ...]:
    """Build a complete wall from per-seat hands plus extra placements.

    ``hands`` maps seat -> {physical_id: count} (13 tiles each). ``dealer_draw``
    pins the dealer's 14th tile (index 52). ``live_draws`` maps absolute wall
    indices >= 52 to physical ids (the dealer draw may be given either way).
    ``dead_wall`` maps indices 120..135 directly (rinshan stack 135 down and
    indicator slots 131/129/...).
    """
    ids: list[int] = _bridge_contracts.build_wall(
        hands=hands, dealer_draw=dealer_draw, live_draws=live_draws, dead_wall=dead_wall
    )
    return tuple(TileId(t) for t in ids)
