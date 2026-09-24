"""Public-history graph validation for resolving subgames (split half)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from hydra2.search.local_abstraction import PublicSubgame

__all__ = [
    "CycleDetectedError",
    "detect_cycle",
]


class CycleDetectedError(ContractError):  # type: ignore[misc]
    """Public-history graph contains a cycle — subgame invalid."""


def _graph_bridge() -> Any | None:
    """Built ``search`` bridge surface or ``None`` when unavailable.

    Same fallback contract as the abstraction's optional accessor (missing
    extensions and stale ``.so`` builds fall back to the pure-Python oracle
    below); kept local so this module never imports its parent and no
    import cycle forms.
    """
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]

        return _ext.search
    except (ImportError, AttributeError):
        return None


def detect_cycle(subgame: PublicSubgame) -> None:
    """DFS cycle detection on directed public-history graph.

    Raises CycleDetectedError if any directed cycle exists. Horizon-bounded
    DAG should be acyclic; abstraction that aliases distinct public histories
    to same hash can introduce cycles, which must be rejected.
    """
    bridge = _graph_bridge()
    detect = None
    if bridge is not None:
        detect = getattr(bridge, "local_graph_detect_cycle", None)
    if detect is not None:
        try:
            detect(subgame.nodes, subgame.edges)
        except ValueError as exc:
            raise CycleDetectedError(str(exc)) from exc
        return
    # Fallback oracle (bridge absent/stale): identical DFS, identical text.
    # Build adjacency
    adj: dict[str, list[str]] = {n: [] for n in subgame.nodes}
    for fr, to, _ in subgame.edges:
        adj[fr].append(to)
    WHITE, GRAY, BLACK = 0, 1, 2  # noqa: N806
    color: dict[str, int] = dict.fromkeys(subgame.nodes, WHITE)

    def dfs(u: str) -> None:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == GRAY:
                raise CycleDetectedError(f"cycle detected: {u} -> {v} closes loop")
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for n in subgame.nodes:
        if color[n] == WHITE:
            dfs(n)
