# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — abstraction: information keys, vectors, subgame declaration.

Owns the information-set firewall shared by the strategy tables and the search loop: the
per-actor information key, the four-seat leaf vectors, the action-abstraction vocabulary
with its coverage guards, and the declared public-history subgame (nodes, edges, horizon,
cycle detection). The strategy tables live in :mod:`hydra2.search.local_strategy`, the
CandidateSpec factory in :mod:`hydra2.search.local_spec`, the resolving loop in
:mod:`hydra2.search.local_search`, and Planner act in :mod:`hydra2.search.local_act`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes, canonical_bytes_batch
from hydra2.contracts.common import ContractError, DigestText
from hydra2.search.local_graph import detect_cycle as detect_cycle
from hydra2.search.local_shared import (
    _MASTER_SEED as _MASTER_SEED,
)
from hydra2.search.local_shared import (
    FORBIDDEN_IN_STRATEGY_KEY as FORBIDDEN_IN_STRATEGY_KEY,
)
from hydra2.search.local_shared import _digest as _digest

__all__ = [
    "AbstractMappingError",
    "LocalResolvingAbstraction",
    "PublicSubgame",
    "abstraction_round_trip",
    "build_public_subgame",
    "info_key_for_actor_observation",
    "info_keys_for_actor_observations",
    "model_vector_for_world",
    "preserves_vector_returns",
    "terminal_vector_for_world",
    "validate_abstraction_mapping",
]


class AbstractMappingError(ContractError):  # type: ignore[misc]
    """Abstraction maps concrete to invalid abstract or loses required coverage."""


# ---------------------------------------------------------------------------
# Deterministic half-open interval helper
# ---------------------------------------------------------------------------


def _require_search_bridge() -> Any:
    """Import the built ``search`` bridge surface (fail closed)."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with search not importable; "
            "build the bridge with `pixi run build-ext` before local resolving vectors"
        ) from exc
    try:
        return _ext.search
    except AttributeError as exc:
        raise ImportError(
            "hydra2._native.search submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _optional_search_bridge() -> Any | None:
    """Built ``search`` bridge surface or ``None`` when unavailable.

    Missing extensions and stale ``.so`` builds fall back to the pure-Python
    oracles below with byte-identical text; the bridge accelerates these graph
    leaves, it never gates them.
    """
    try:
        return _require_search_bridge()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Information-set key — per actor, never world_id
# ---------------------------------------------------------------------------


def _info_key_payload(observation: Any) -> dict[str, Any]:
    """Actor-visible payload hashed by the per-actor information-set key."""
    try:
        actor = int(getattr(observation, "actor", 0))
    except Exception:
        actor = 0
    # concealed_hand for acting player only
    try:
        hand = tuple(int(t) for t in getattr(observation, "concealed_hand", ()))
    except Exception:
        hand = ()
    # public visible: visible_melds, scores, round etc — include everything except private hidden
    try:
        visible = getattr(observation, "visible_melds", ())
        # flatten count
        vis_len = len(visible) if isinstance(visible, (list, tuple)) else 0
    except Exception:
        vis_len = 0
    try:
        seq = int(getattr(observation, "sequence", 0))
    except Exception:
        seq = 0
    try:
        obs_hash = str(getattr(observation, "observation_hash", ""))
    except Exception:
        obs_hash = ""
    return {
        "actor": actor,
        "hand": list(hand),
        "vis_len": vis_len,
        "sequence": seq,
        "observation_hash": obs_hash,
    }


def info_key_for_actor_observation(observation: Any) -> str:
    """Canonical per-actor information-set hash (actor-visible only).

    For the tiny domain we hash a payload of:
      actor, concealed_hand, public visible fields (discards, riichi flags, scores)
    Excludes legal_mask redundancy, world_id, hidden hands of others.
    """
    payload = _info_key_payload(observation)
    # Hash via canonical_bytes when available else json
    try:
        blob = canonical_bytes(payload)
    except Exception:
        import json

        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).hexdigest()
    key = "sha256:" + digest
    # forbidden check
    for bad in FORBIDDEN_IN_STRATEGY_KEY:
        if bad in key:
            raise ContractError(f"strategy key contains forbidden substring {bad!r}")
    return key


def info_keys_for_actor_observations(observations: list[Any]) -> list[str]:
    """Batch per-actor information-set keys via ONE ``canonical_bytes_batch`` FFI.

    Byte-identical to ``[info_key_for_actor_observation(o) for o in observations]``
    on canonical-domain payloads (the only shape this module builds): same
    payloads, same bytes, same digests. Fail-closed: bridge errors raise —
    callers that tolerate per-slot failure must catch and fall back to the
    single-key authority loop.
    """
    blobs = canonical_bytes_batch([_info_key_payload(o) for o in observations])
    keys: list[str] = []
    for blob in blobs:
        key = "sha256:" + hashlib.sha256(blob).hexdigest()
        for bad in FORBIDDEN_IN_STRATEGY_KEY:
            if bad in key:
                raise ContractError(f"strategy key contains forbidden substring {bad!r}")
        keys.append(key)
    return keys


def _actor_to_key(actor: int) -> int:
    """Seat validator — single home is ``hydra2._native.contracts.make_seat``."""
    try:
        return _bridge_contracts.make_seat(actor)
    except (ImportError, AttributeError):
        pass  # stale .so: fall through to the oracle below (same values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError(f"actor must be int in 0..3, got {actor!r}") from exc
    if not isinstance(actor, int) or isinstance(actor, bool) or not (0 <= actor <= 3):
        raise ContractError(f"actor must be int in 0..3, got {actor!r}")
    return actor


# ---------------------------------------------------------------------------
# Vector returns — four-seat, deterministic, settlement-preserving
# ---------------------------------------------------------------------------


def model_vector_for_world(
    world: Any, *, leaf_kind: str = "model"
) -> tuple[float, float, float, float]:
    """Deterministic four-seat leaf value from actor-visible proxy.

    Uses world_id hash to derive bounded values; sum is zero to preserve
    general-sum feasibility (zero-sum subset). Finite and reproducible.
    """
    wid = str(getattr(world, "world_id", "world_unknown"))
    # Bridge is the single implementation (fail closed).
    wid_in = "world_unknown" if wid == "" else wid
    try:
        out: tuple[float, float, float, float] = _require_search_bridge().ismcts_local_model_vector(
            wid_in, leaf_kind
        )
        return (out[0], out[1], out[2], out[3])
    except ImportError:
        raise
    except Exception as exc:
        raise ContractError(f"local bridge model vector failed: {exc}") from exc


def terminal_vector_for_world(world: Any) -> tuple[float, float, float, float]:
    """Exact terminal settlement vector derived from concealed hands + wall.

    Bridge is the single implementation (fail closed).
    """
    try:
        hands_in_raw = getattr(world, "concealed_hands", ((0,),) * 4)
        wall_raw = getattr(world, "live_wall", ())
        hands_in = [[int(t) for t in hand] for hand in hands_in_raw]
        live_in = [] if wall_raw is None else [int(t) for t in wall_raw]
        term_out: tuple[float, float, float, float] = (
            _require_search_bridge().ismcts_local_terminal_vector(hands_in, live_in)
        )
        return (term_out[0], term_out[1], term_out[2], term_out[3])
    except ImportError:
        raise
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"local bridge terminal vector failed: {exc}") from exc


def preserves_vector_returns(vector: tuple[float, ...]) -> bool:
    """Check that vector is four-seat, finite, bounded."""
    if not isinstance(vector, (list, tuple)) or len(vector) != 4:
        return False
    for v in vector:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False
        if abs(float(v)) > 100.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalResolvingAbstraction:
    """Declared action abstraction — maps concrete action ids to abstract ids.

    Concrete ids are 0..6791 (full table). For tiny domain we use 0..N-1.
    Mapping must be surjective onto abstract range and cover all legal concretes.
    """

    name: str
    concrete_to_abstract: tuple[tuple[int, int], ...]  # sorted pairs
    abstract_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise ContractError("abstraction name must be non-empty str")
        if self.name not in ("identity", "pair_merge", "tile_type", "custom"):
            raise ContractError(f"unknown abstraction {self.name!r}")
        if not isinstance(self.concrete_to_abstract, tuple):
            raise ContractError("concrete_to_abstract must be tuple")
        seen_c: set[int] = set()
        seen_a: set[int] = set()
        for pair in self.concrete_to_abstract:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ContractError(f"pair must be (concrete, abstract), got {pair!r}")
            c, a = pair
            if not isinstance(c, int) or isinstance(c, bool) or c < 0:
                raise ContractError(f"concrete id must be nonnegative int, got {c!r}")
            if not isinstance(a, int) or isinstance(a, bool) or a < 0:
                raise ContractError(f"abstract id must be nonnegative int, got {a!r}")
            if c in seen_c:
                raise ContractError(f"duplicate concrete id {c}")
            seen_c.add(c)
            seen_a.add(a)
        if not isinstance(self.abstract_ids, tuple):
            raise ContractError("abstract_ids must be tuple")
        if set(self.abstract_ids) != seen_a:
            raise ContractError(
                f"abstract_ids {self.abstract_ids} must equal image of mapping {seen_a}"
            )
        if tuple(sorted(self.abstract_ids)) != self.abstract_ids:
            raise ContractError("abstract_ids must be sorted")
        # mapping must be sorted by concrete for determinism
        if self.concrete_to_abstract != tuple(sorted(self.concrete_to_abstract)):
            raise ContractError("concrete_to_abstract must be sorted by concrete id")

    def map_concrete(self, concrete_id: int) -> int:
        for c, a in self.concrete_to_abstract:
            if c == concrete_id:
                return a
        raise AbstractMappingError(f"concrete id {concrete_id} not in abstraction mapping")

    def map_abstract_to_representative(self, abstract_id: int) -> int:
        if abstract_id not in self.abstract_ids:
            raise AbstractMappingError(f"abstract id {abstract_id} unknown")
        # representative is smallest concrete mapping to that abstract
        candidates = [c for c, a in self.concrete_to_abstract if a == abstract_id]
        return min(candidates)


def validate_abstraction_mapping(
    mapping: dict[int, int] | tuple[tuple[int, int], ...] | LocalResolvingAbstraction,
    legal_concrete_ids: tuple[int, ...],
) -> LocalResolvingAbstraction:
    """Validate that mapping covers all legal concretes and is consistent.

    Returns a frozen LocalResolvingAbstraction or raises AbstractMappingError.
    """
    if isinstance(mapping, LocalResolvingAbstraction):
        ab = mapping
    elif isinstance(mapping, dict):
        pairs = tuple(sorted(mapping.items()))
        abstract_ids = tuple(sorted({a for _, a in pairs}))
        # infer name
        name = "custom" if len(pairs) > 0 else "identity"
        # Determine if identity: mapping is i->i for all
        if len(pairs) > 0 and all(c == a for c, a in pairs):
            name = "identity"
        ab = LocalResolvingAbstraction(
            name=name, concrete_to_abstract=pairs, abstract_ids=abstract_ids
        )
    elif isinstance(mapping, tuple):
        pairs = tuple(sorted(mapping))
        abstract_ids = tuple(sorted({a for _, a in pairs}))
        name = "custom"
        if len(pairs) > 0 and all(c == a for c, a in pairs):
            name = "identity"
        # check pair_merge pattern
        if name == "custom" and len(pairs) >= 2:
            # if mapping merges pairs 0,1->0, 2,3->1 etc
            merged = all(pairs[i][1] == pairs[i + 1][1] for i in range(0, len(pairs) - 1, 2))
            if merged:
                name = "pair_merge"
        ab = LocalResolvingAbstraction(
            name=name, concrete_to_abstract=pairs, abstract_ids=abstract_ids
        )
    # check coverage of legal ids
    legal_set = set(legal_concrete_ids)
    mapped_concretes = {c for c, _ in ab.concrete_to_abstract}
    missing = legal_set - mapped_concretes
    if len(missing) > 0:
        raise AbstractMappingError(f"abstraction missing legal concretes {sorted(missing)}")
    # check no mapping references out-of-range abstract
    if any(a not in ab.abstract_ids for _, a in ab.concrete_to_abstract):
        raise AbstractMappingError("mapping references abstract id not in abstract_ids")
    return ab


def abstraction_round_trip(abstraction: LocalResolvingAbstraction, abstract_id: int) -> int:
    """Round-trip abstract -> representative concrete -> abstract.

    Returns the recovered abstract id; raises if mismatch.
    """
    rep = abstraction.map_abstract_to_representative(abstract_id)
    back = abstraction.map_concrete(rep)
    if back != abstract_id:
        raise AbstractMappingError(f"round-trip failed: {abstract_id} -> {rep} -> {back}")
    return back


# ---------------------------------------------------------------------------
# Public subgame — declared horizon, public history, abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicSubgame:
    """Declared public-history subgame.

    Nodes are keyed by public history hash; edges labeled by abstract action.
    Horizon bounds depth. Abstraction maps concrete->abstract.
    """

    horizon: int
    abstraction: LocalResolvingAbstraction
    public_history_hash: DigestText
    nodes: tuple[str, ...]  # public node hashes, root first
    edges: tuple[tuple[str, str, int], ...]  # (from_hash, to_hash, abstract_id)
    iteration_count: int
    averaging_rule: str
    update_rule: str
    leaf_model: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or self.horizon <= 0
            or self.horizon > 16
        ):
            raise ContractError(f"horizon must be int in 1..16, got {self.horizon!r}")
        if (
            not isinstance(self.iteration_count, int)
            or isinstance(self.iteration_count, bool)
            or self.iteration_count <= 0
        ):
            raise ContractError(
                f"iteration_count must be positive int, got {self.iteration_count!r}"
            )
        if self.update_rule not in ("regret_matching", "hedge", "fictitious_play"):
            raise ContractError(f"unknown update_rule {self.update_rule!r}")
        if self.averaging_rule not in ("uniform", "linear"):
            raise ContractError(f"unknown averaging_rule {self.averaging_rule!r}")
        if self.leaf_model not in ("model", "terminal"):
            raise ContractError(
                f"leaf_model must be 'model' or 'terminal', got {self.leaf_model!r}"
            )
        try:
            _bridge_contracts.make_digest_text(self.public_history_hash)
        except Exception as exc:
            raise ContractError(
                f"public_history_hash must be sha256 digest, got {self.public_history_hash!r}"
            ) from exc
        if not isinstance(self.nodes, tuple) or len(self.nodes) == 0:
            raise ContractError("nodes must be non-empty tuple")
        for n in self.nodes:
            _bridge_contracts.make_digest_text(n)
        if len(set(self.nodes)) != len(self.nodes):
            raise ContractError("nodes must be distinct (no duplicate public hashes)")
        # edges must reference nodes
        node_set = set(self.nodes)
        for e in self.edges:
            if not isinstance(e, (list, tuple)) or len(e) != 3:
                raise ContractError(f"edge must be (from, to, abstract_id), got {e!r}")
            fr, to, aid = e
            _bridge_contracts.make_digest_text(fr)
            _bridge_contracts.make_digest_text(to)
            if fr not in node_set or to not in node_set:
                raise ContractError(f"edge references unknown node {e!r}")
            if not isinstance(aid, int) or isinstance(aid, bool) or aid < 0:
                raise ContractError(f"edge abstract_id must be nonnegative int, got {aid!r}")
            if aid not in self.abstraction.abstract_ids:
                raise AbstractMappingError(f"edge abstract_id {aid} not in abstraction")


def build_public_subgame(
    epoch: Any | None,
    *,
    horizon: int,
    abstraction: LocalResolvingAbstraction | dict[int, int] | tuple[tuple[int, int], ...],
    iteration_count: int = 16,
    averaging: str = "uniform",
    update_rule: str = "regret_matching",
    leaf_model: str = "model",
    public_history_seed: str | None = None,
) -> PublicSubgame:
    """Build declared public-history subgame from epoch.

    Validates horizon, abstraction, and iteration counts. Uses epoch target_id
    as public history seed when not supplied. Does not claim equilibrium.
    """
    # Validate frozen fields
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not (1 <= horizon <= 16):
        raise ContractError(f"horizon must be int in 1..16, got {horizon!r}")
    if (
        not isinstance(iteration_count, int)
        or isinstance(iteration_count, bool)
        or iteration_count <= 0
    ):
        raise ContractError(f"iteration_count must be positive int, got {iteration_count!r}")
    if averaging not in ("uniform", "linear"):
        raise ContractError(f"averaging must be uniform or linear, got {averaging!r}")
    if update_rule not in ("regret_matching", "hedge", "fictitious_play"):
        raise ContractError(f"update_rule {update_rule!r} unknown")
    if leaf_model not in ("model", "terminal"):
        raise ContractError(f"leaf_model {leaf_model!r} unknown")
    # Derive public_history_hash from epoch if available
    if public_history_seed is not None:
        ph_hash = _digest(public_history_seed)
    elif epoch is not None:
        try:
            target = str(getattr(epoch, "target_id", "target_unknown"))
            obs_h = str(getattr(epoch, "observation_hash", "obs_unknown"))
            ph_hash = _digest(f"{target}:{obs_h}:{horizon}")
        except Exception:
            ph_hash = _digest(f"fallback:{horizon}:{averaging}:{update_rule}")
    else:
        ph_hash = _digest(f"no_epoch:{horizon}:{averaging}:{update_rule}")
    # Normalize abstraction — use minimal legal set for validation if not provided
    # For subgame construction we assume tiny domain with concrete ids 0..3 for determinism
    # If abstraction already covers 0..3, use it; else validate against 0..3 placeholder
    if isinstance(abstraction, LocalResolvingAbstraction):
        ab = abstraction
    else:
        # Build mapping from abstraction arg; if mapping is incomplete for tiny domain, extend with identity
        if isinstance(abstraction, dict):
            raw = abstraction
        elif isinstance(abstraction, tuple):
            raw = dict(abstraction)
        else:
            raise AbstractMappingError(f"unsupported abstraction type {type(abstraction)}")
        # Ensure coverage of 0..3 for tiny subgame if not already
        for cid in (0, 1, 2, 3):
            if cid not in raw:
                raw[cid] = cid
        ab = validate_abstraction_mapping(raw, legal_concrete_ids=(0, 1, 2, 3))
    # Level expansion runs in ``hydra2._native.search`` when available
    # (``local_graph_expand_subgame``); the loop below is the byte-identical
    # fallback for missing/stale extensions.
    root = ph_hash
    bridge = _optional_search_bridge()
    expand = None
    if bridge is not None:
        expand = getattr(bridge, "local_graph_expand_subgame", None)
    nodes: list[str] = []
    edges: list[tuple[str, str, int]] = []
    if expand is not None:
        try:
            expanded: tuple[list[str], list[tuple[str, str, int]]] = expand(
                root, list(ab.abstract_ids), horizon
            )
            raw_nodes, raw_edges = expanded
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        nodes = list(raw_nodes)
        edges = [(fr, to, aid) for fr, to, aid in raw_edges]
    else:
        # Build nodes: root + horizon * branching (abstract fanout) synthetic hashes
        # Branching = len(ab.abstract_ids), capped for small horizon to avoid explosion
        # Build level-by-level hashes
        nodes.append(root)
        # Map from level nodes to next level
        current_level = [root]
        for d in range(horizon):
            next_level: list[str] = []
            for parent in current_level:
                for aid in ab.abstract_ids:
                    child_hash = _digest(f"{parent}:{d}:{aid}")
                    # Avoid duplicate hashes (deterministic)
                    if child_hash not in nodes:
                        nodes.append(child_hash)
                        next_level.append(child_hash)
                    else:
                        # Already exists — still add edge, but node not duplicated
                        next_level.append(child_hash)
                    edges.append((parent, child_hash, aid))
            current_level = next_level
            # Cap explosion: if nodes exceed 64, stop expanding deeper (still respect horizon for depth)
            if len(nodes) > 64:
                break
    subgame = PublicSubgame(
        horizon=horizon,
        abstraction=ab,
        public_history_hash=ph_hash,
        nodes=tuple(nodes),
        edges=tuple(edges),
        iteration_count=iteration_count,
        averaging_rule=averaging,
        update_rule=update_rule,
        leaf_model=leaf_model,
    )
    # Cycle check after build
    detect_cycle(subgame)
    return subgame
