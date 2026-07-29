"""WP-04A deterministic wall construction for edge-case corpus cases.

A corpus case pins a COMPLETE 136-tile :class:`~hydra2.engines.protocol.WallSchedule`.
The builder here assembles such walls from explicit per-seat hand multisets plus
live-draw and dead-wall placements, using one global cycle-safe assignment so
that no placement silently corrupts another seat's tiles (probe-verified
failure mode).

Layout facts of RiichiEnv 0.4.8 under ``reset(wall=...)`` (probed with an
identity wall, see docs/wp04a-progress.md or attestation metadata):

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

from hydra2.contracts.common import TileId

if TYPE_CHECKING:
    from collections.abc import Mapping

HAIPAI_SLOTS: tuple[int, ...] = tuple(
    idx
    for seat in range(4)
    for idx in (
        4 * seat,
        4 * seat + 1,
        4 * seat + 2,
        4 * seat + 3,
        16 + 4 * seat,
        17 + 4 * seat,
        18 + 4 * seat,
        19 + 4 * seat,
        32 + 4 * seat,
        33 + 4 * seat,
        34 + 4 * seat,
        35 + 4 * seat,
        48 + seat,
    )
)


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
    "first_copy_of_type",
    "haipai_index",
    "type_id",
]


#: First index consumed by a live (non-rinshan) draw.
LIVE_DRAW_BASE = 52
#: Dealer's opening 14th tile.
DEALER_DRAW_INDEX = LIVE_DRAW_BASE
#: Index of the initially revealed dora indicator.
FIRST_DORA_INDICATOR_INDEX = 131
#: Highest rinshan tile index (draws descend from here).
RINSHAN_TOP_INDEX = 135

# Honor physical ids used throughout case definitions (first copy of each).
TILE_E = 108
TILE_S = 112
TILE_W = 116
TILE_N = 120
TILE_P = 124  # haku
TILE_F = 128  # hatsu
TILE_C = 132  # chun


def type_id(physical: int) -> int:
    """Logical tile type of a physical id (SPEC §4.1)."""
    return physical // 4


def copies(tile_type: int):
    """The four physical ids of ``tile_type`` in ascending order."""
    return tuple(4 * tile_type + offset for offset in range(4))


def first_copy_of_type(tile_type: int) -> int:
    return 4 * tile_type


def haipai_index(seat: int, position: int) -> int:
    """Wall index dealt as ``position`` (0-based) of ``seat``'s opening hand."""
    if not 0 <= seat <= 3:
        raise ValueError("seat must be 0..3")
    if not 0 <= position <= 12:
        raise ValueError("position must be 0..12")
    row, column = divmod(position, 4)
    if row < 3:
        return 16 * row + 4 * seat + column
    return 48 + seat


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
    plan = WallPlan()
    for seat in range(4):
        hand = hands.get(seat, {})
        _ = plan.require_hand(seat, hand)
    if dealer_draw is not None:
        plan.require(DEALER_DRAW_INDEX, dealer_draw)
    for index, physical in sorted((live_draws if live_draws is not None else {}).items()):
        if index < LIVE_DRAW_BASE:
            raise ValueError("live draws start at index 52")
        plan.require(index, physical)
    for index, physical in sorted((dead_wall if dead_wall is not None else {}).items()):
        if index < 120:
            raise ValueError("dead-wall indices start at 120")
        plan.require(index, physical)
    return plan.resolve()
