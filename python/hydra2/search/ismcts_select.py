"""ISMCTS UCT selection over information-set nodes (split half)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra2.contracts.common import ContractError
from hydra2.search.ismcts_core import _require_search_bridge as _require_search_bridge

if TYPE_CHECKING:
    from hydra2.search.ismcts_core import InformationSetNode

__all__ = [
    "_uct_select",
]


def _uct_select(
    node: InformationSetNode, legal: tuple[int, ...], root_seat: int, uct_c: float, tie_break: str
) -> int:
    # UCT pick rides the bridge (unvisited-first + q+u with 1e-12 eps + tie
    # arm, bit-identical to the oracle below); node tables stay Python —
    # only visited-arm scalars cross, never hidden worlds (info-keys only).
    try:
        stat_ids = sorted(node.action_stats.keys())
    except Exception as exc:
        raise ContractError(f"uct node stats unreadable: {exc}") from exc
    actions: list[int] = []
    visits: list[int] = []
    sums_flat: list[float] = []
    for aid in stat_ids:
        st = node.action_stats.get(aid)
        if st is None:
            continue
        try:
            n = int(st.visits)  # type: ignore[arg-type]
            quad = tuple(st.value_sum)
        except Exception as exc:
            raise ContractError(f"uct node stats malformed for {aid!r}: {exc}") from exc
        if len(quad) != 4:
            raise ContractError(f"uct node value_sum must hold 4 entries for {aid!r}")
        actions.append(aid)
        visits.append(n)
        sums_flat.extend(quad)
    try:
        total = int(node.visits)  # type: ignore[arg-type]
        selected: int = _require_search_bridge().uct_select(
            actions,
            visits,
            sums_flat,
            list(legal),
            root_seat,
            total,
            uct_c,
            tie_break,
        )
        return selected
    except ImportError:
        raise
    except Exception as exc:
        raise ContractError(f"ismcts bridge uct failed: {exc}") from exc
