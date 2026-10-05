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


class CycleDetectedError(ContractError):
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
    bridge_detect = None
    if bridge is not None:
        # Bridge owns the fast path when available; else use the oracle below.
        bridge_detect = getattr(bridge, "local_graph_detect_cycle", None)
    if bridge_detect is not None:
        try:
            bridge_detect(subgame.nodes, subgame.edges)
        except ValueError as exc:
            raise CycleDetectedError(str(exc)) from exc
        return
    # Fallback oracle (bridge absent/stale): identical DFS, identical text.
    # Adjacency keyed by node; missing keys mean no outgoing edges.
    adjacency: dict[str, list[str]] = {node: [] for node in subgame.nodes}
    for source, target, _edge_payload in subgame.edges:
        adjacency[source].append(target)
    WHITE, GRAY, BLACK = 0, 1, 2  # noqa: N806
    color: dict[str, int] = dict.fromkeys(subgame.nodes, WHITE)

    def dfs(node_id: str) -> None:
        # GRAY marks the current DFS path; re-entry means a directed cycle.
        color[node_id] = GRAY
        for neighbor in adjacency.get(node_id, []):
            if color[neighbor] == GRAY:
                raise CycleDetectedError(f"cycle detected: {node_id} -> {neighbor} closes loop")
            if color[neighbor] == WHITE:
                dfs(neighbor)
        color[node_id] = BLACK

    for start_node in subgame.nodes:
        if color[start_node] == WHITE:
            dfs(start_node)
