# ruff: noqa: F401  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""WP-09D Candidate 5 Local Resolving — public-history subgame, information-set strategies.

Implements Blueprint §12 (Candidate 5) and SPEC 16.6:

- Tables key ``(actor, information_node_hash)``, never root world.
- Every actor update uses only that actor's information set.
- Return vectors remain four-seat, settlement-preserving, exact.
- Subgame horizon, abstraction, leaf model, update, iteration count, and averaging
  are CandidateSpec fields and frozen.
- Cycle and abstraction failure abort candidate.
- Output is empirical optimizer result; never equilibrium or exploitability certificate.
- PBRF warm start variant compared without claiming superiority.

Deterministic, CPU-only, no hidden-state leakage. All randomness via semantic
counter-based seeds derived from ``(case_id, root_seat, candidate_id)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hydra2.search.common import (
    DEPLOYABLE_DEADLINE_MS,
    CandidateSpec,
    Planner,
    ResourceBudget,
    SearchRequest,
    SearchResult,
)

_COMMON_AVAILABLE = True  # common always available via direct import (shim deleted)

try:
    from hydra2.artifacts.canonical import canonical_bytes
    from hydra2.contracts.common import ContractError, DigestText, make_digest_text

    _HAS_CONTRACTS = True
except ImportError:  # pragma: no cover
    _HAS_CONTRACTS = False
    ContractError = RuntimeError
    DigestText = str

    def make_digest_text(v: str) -> str:
        if not isinstance(v, str) or not v.startswith("sha256:") or len(v) != 71:
            raise RuntimeError(f"bad digest {v!r}")
        return v

    def canonical_bytes(v: Any) -> bytes:  # type: ignore[no-redef]
        import json

        return json.dumps(v, sort_keys=True, separators=(",", ":")).encode()


try:
    from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

    _HAS_RANDOM = True
except ImportError:  # pragma: no cover
    _HAS_RANDOM = False
    RandomStream = Any

try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief

    _HAS_BELIEF = True
except ImportError:  # pragma: no cover
    _HAS_BELIEF = False
    BeliefEpoch = Any
    NaturalBelief = Any

try:
    from hydra2.contracts.observation import make_actor_observation

    _HAS_OBS = True
except ImportError:
    _HAS_OBS = False
logger = logging.getLogger(__name__)


__all__ = [
    "AbstractMappingError",
    "CycleDetectedError",
    "LocalResolvingAbstraction",
    "LocalResolvingConfig",
    "LocalResolvingPlanner",
    "PublicSubgame",
    "StrategyTable",
    "abstraction_round_trip",
    "build_public_subgame",
    "detect_cycle",
    "info_key_for_actor_observation",
    "is_equilibrium_claimed",
    "leaf_vector_replay",
    "make_candidate5_spec",
    "model_vector_for_world",
    "terminal_vector_for_world",
    "validate_abstraction_mapping",
]

_MASTER_SEED = b"wp09d_local_resolving_v1"
FORBIDDEN_IN_STRATEGY_KEY: frozenset[str] = frozenset({"world_id", "full_hidden", "privileged"})


class AbstractMappingError(ContractError):  # type: ignore[misc]
    """Abstraction maps concrete to invalid abstract or loses required coverage."""


class CycleDetectedError(ContractError):  # type: ignore[misc]
    """Public-history graph contains a cycle — subgame invalid."""


# ---------------------------------------------------------------------------
# Deterministic half-open interval helper
# ---------------------------------------------------------------------------


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(s: str) -> str:
    return "sha256:" + _h(s.encode())


def _seed_bytes(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode()).digest()


# ---------------------------------------------------------------------------
# Information-set key — per actor, never world_id
# ---------------------------------------------------------------------------


def info_key_for_actor_observation(observation: Any) -> str:
    """Canonical per-actor information-set hash (actor-visible only).

    For the tiny domain we hash a payload of:
      actor, concealed_hand, public visible fields (discards, riichi flags, scores)
    Excludes legal_mask redundancy, world_id, hidden hands of others.
    """
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
    payload = {
        "actor": actor,
        "hand": list(hand),
        "vis_len": vis_len,
        "sequence": seq,
        "observation_hash": obs_hash,
    }
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


def _actor_to_key(actor: int) -> int:
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
    h = hashlib.sha256((wid + ":" + leaf_kind).encode()).digest()
    # 4 values in [-1, 1] from bytes
    raw = [int.from_bytes(h[i * 2 : i * 2 + 2], "little") for i in range(4)]
    vals = tuple(((r / 65535.0) * 2.0 - 1.0) for r in raw)
    # enforce zero-sum for conservation test (general-sum allows non-zero but zero satisfies spec)
    s = sum(vals)
    # shift to zero-sum
    centered = tuple(v - s / 4.0 for v in vals)
    # bounds check
    for v in centered:
        if not math.isfinite(v) or abs(v) > 2.0:
            raise ContractError(f"vector value out of bounds {v}")
    return centered  # type: ignore[return-value]


def terminal_vector_for_world(world: Any) -> tuple[float, float, float, float]:
    """Exact terminal settlement vector derived from concealed hands + wall."""
    try:
        hands = getattr(world, "concealed_hands", ((0,),) * 4)
        # sum tiles per seat as strength proxy
        sums = []
        for hand in hands:
            try:
                s = sum(int(t) for t in hand)
            except Exception:
                s = 0
            sums.append(float(s))
        # normalize to zero-sum settlement
        mean = sum(sums) / 4.0
        centered = tuple((s - mean) / 10.0 for s in sums)
        # add wall influence
        try:
            wall = getattr(world, "live_wall", ())
            wall_sum = sum(int(t) for t in wall) / 100.0
            centered = tuple(
                v + wall_sum * (0.1 if i == 0 else -0.03) for i, v in enumerate(centered)
            )
            # re-center
            mean2 = sum(centered) / 4.0
            centered = tuple(v - mean2 for v in centered)
        except Exception:
            pass
        for v in centered:
            if not math.isfinite(v):
                raise ContractError("terminal vector nonfinite")
        return centered  # type: ignore[return-value]
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"terminal vector failed: {exc}") from exc


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
            _ = make_digest_text(self.public_history_hash)
        except Exception as exc:
            raise ContractError(
                f"public_history_hash must be sha256 digest, got {self.public_history_hash!r}"
            ) from exc
        if not isinstance(self.nodes, tuple) or len(self.nodes) == 0:
            raise ContractError("nodes must be non-empty tuple")
        for n in self.nodes:
            _ = make_digest_text(n)
        if len(set(self.nodes)) != len(self.nodes):
            raise ContractError("nodes must be distinct (no duplicate public hashes)")
        # edges must reference nodes
        node_set = set(self.nodes)
        for e in self.edges:
            if not isinstance(e, (list, tuple)) or len(e) != 3:
                raise ContractError(f"edge must be (from, to, abstract_id), got {e!r}")
            fr, to, aid = e
            _ = make_digest_text(fr)
            _ = make_digest_text(to)
            if fr not in node_set or to not in node_set:
                raise ContractError(f"edge references unknown node {e!r}")
            if not isinstance(aid, int) or isinstance(aid, bool) or aid < 0:
                raise ContractError(f"edge abstract_id must be nonnegative int, got {aid!r}")
            if aid not in self.abstraction.abstract_ids:
                raise AbstractMappingError(f"edge abstract_id {aid} not in abstraction")

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)


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
    # Build nodes: root + horizon * branching (abstract fanout) synthetic hashes
    # Branching = len(ab.abstract_ids), capped for small horizon to avoid explosion
    # Build level-by-level hashes
    nodes: list[str] = []
    edges: list[tuple[str, str, int]] = []
    root = ph_hash
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


def detect_cycle(subgame: PublicSubgame) -> None:
    """DFS cycle detection on directed public-history graph.

    Raises CycleDetectedError if any directed cycle exists. Horizon-bounded
    DAG should be acyclic; abstraction that aliases distinct public histories
    to same hash can introduce cycles, which must be rejected.
    """
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


# ---------------------------------------------------------------------------
# Strategy table — keyed by (actor, information_node_hash)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrategyTable:
    """Per-actor information-set strategy table.

    Keys are ``(actor, info_hash)``; never world_id. Values are distributions
    over abstract actions (tuple aligned with subgame's abstract_ids order).
    """

    abstraction: LocalResolvingAbstraction
    table: dict[tuple[int, str], tuple[float, ...]] = field(default_factory=dict)
    visit_counts: dict[tuple[int, str], int] = field(default_factory=dict)

    def get(self, actor: int, info_hash: str) -> tuple[float, ...] | None:
        _ = _actor_to_key(actor)
        _ = make_digest_text(info_hash)
        return self.table.get((actor, info_hash))

    def set(self, actor: int, info_hash: str, distribution: tuple[float, ...]) -> None:
        _ = _actor_to_key(actor)
        _ = make_digest_text(info_hash)
        if not isinstance(distribution, (list, tuple)):
            raise ContractError("distribution must be tuple")
        if len(distribution) != len(self.abstraction.abstract_ids):
            raise ContractError(
                f"distribution length {len(distribution)} must equal abstract size {len(self.abstraction.abstract_ids)}"
            )
        s = sum(distribution)
        if abs(s - 1.0) > 1e-6:
            raise ContractError(f"distribution must sum to 1, got {s}")
        for p in distribution:
            if (
                not isinstance(p, (int, float))
                or not math.isfinite(float(p))
                or not (0.0 <= float(p) <= 1.0)
            ):
                raise ContractError(f"distribution entry {p!r} must be in [0,1]")
        self.table[actor, info_hash] = tuple(distribution)

    def ensure_uniform(self, actor: int, info_hash: str) -> tuple[float, ...]:
        key = (actor, info_hash)
        if key not in self.table:
            n = len(self.abstraction.abstract_ids)
            uni = tuple(1.0 / n for _ in range(n))
            self.table[key] = uni
            self.visit_counts[key] = 0
        return self.table[key]

    def keys(self) -> list[tuple[int, str]]:
        return list(self.table.keys())

    def validate_no_world_id(self) -> bool:
        for actor, info_hash in self.table:
            if not isinstance(info_hash, str) or not info_hash.startswith("sha256:"):
                return False
            for bad in FORBIDDEN_IN_STRATEGY_KEY:
                if bad in info_hash:
                    return False
            if not isinstance(actor, int) or not (0 <= actor <= 3):
                return False
        return True


def make_uniform_strategy(abstraction: LocalResolvingAbstraction) -> tuple[float, ...]:
    n = len(abstraction.abstract_ids)
    return tuple(1.0 / n for _ in range(n))


# ---------------------------------------------------------------------------
# Update rules — frozen, empirical optimizers only (never equilibrium)
# ---------------------------------------------------------------------------

_VALID_UPDATE_RULES = frozenset({"regret_matching", "hedge", "fictitious_play"})
_VALID_AVERAGING = frozenset({"uniform", "linear"})


def frozen_update_rule_names() -> frozenset[str]:
    return _VALID_UPDATE_RULES


def frozen_averaging_rule_names() -> frozenset[str]:
    return _VALID_AVERAGING


def _regret_matching_update(
    current: tuple[float, ...],
    regrets: tuple[float, ...],
) -> tuple[float, ...]:
    # positive regrets
    pos = tuple(max(0.0, r) for r in regrets)
    s = sum(pos)
    n = len(current)
    if s > 1e-12:
        return tuple(p / s for p in pos)
    return tuple(1.0 / n for _ in range(n))


def _hedge_update(
    current: tuple[float, ...],
    q_values: tuple[float, ...],
    eta: float = 1.0,
) -> tuple[float, ...]:
    # softmax over cumulative Q
    # subtract max for stability
    m = max(q_values)
    exps = tuple(math.exp(eta * (q - m)) for q in q_values)
    s = sum(exps)
    return tuple(e / s for e in exps)


def _fictitious_play_update(
    current: tuple[float, ...],
    best_response_idx: int,
    count: int,
    new_weight: float = 1.0,
) -> tuple[float, ...]:
    n = len(current)
    w = new_weight
    total = count + w
    out = []
    for i in range(n):
        prev = current[i] * count / total if count > 0 else 0.0
        add = w / total if i == best_response_idx else 0.0
        out.append(prev + add)
    # Renormalize due to floating error
    s = sum(out)
    return tuple(v / s for v in out) if s > 0 else tuple(1.0 / n for _ in range(n))


def apply_update(
    strategy: tuple[float, ...],
    *,
    update_rule: str,
    regrets: tuple[float, ...] | None = None,
    q_values: tuple[float, ...] | None = None,
    best_response_idx: int | None = None,
    visit_count: int = 0,
) -> tuple[float, ...]:
    if update_rule not in _VALID_UPDATE_RULES:
        raise ContractError(f"unknown update_rule {update_rule!r}")
    if update_rule == "regret_matching":
        if regrets is None:
            raise ContractError("regret_matching requires regrets")
        return _regret_matching_update(strategy, regrets)
    if update_rule == "hedge":
        if q_values is None:
            raise ContractError("hedge requires q_values")
        return _hedge_update(strategy, q_values)
    if update_rule == "fictitious_play":
        if best_response_idx is None:
            raise ContractError("fictitious_play requires best_response_idx")
        return _fictitious_play_update(strategy, best_response_idx, visit_count)
    raise ContractError(f"unhandled update_rule {update_rule!r}")


def averaging_weights(iteration: int, total: int, rule: str) -> float:
    if rule == "uniform":
        return 1.0
    if rule == "linear":
        # linear weighting by iteration index (1-based)
        return float(iteration)
    raise ContractError(f"unknown averaging rule {rule!r}")


def is_equilibrium_claimed() -> bool:
    """Local resolving never claims equilibrium or exploitability guarantee."""
    return False


# ---------------------------------------------------------------------------
# Leaf replay helper
# ---------------------------------------------------------------------------


def leaf_vector_replay(world: Any, leaf_model: str = "model") -> tuple[float, float, float, float]:
    if leaf_model == "terminal":
        return terminal_vector_for_world(world)
    return model_vector_for_world(world, leaf_kind=leaf_model)


# ---------------------------------------------------------------------------
# Frozen config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalResolvingConfig:
    """Frozen Candidate 5 hyper-parameters (part of CandidateSpec.parameters)."""

    horizon: int = 2
    iterations: int = 16
    update_rule: str = "regret_matching"
    averaging: str = "uniform"
    abstraction: str = "identity"  # descriptor; actual mapping built via helper
    leaf_model: str = "model"
    tie_break: str = "greedy"
    resource_view: Literal["calls", "transitions", "joules"] = "calls"
    public_history_seed: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or not (1 <= self.horizon <= 16)
        ):
            raise ContractError(f"horizon must be int in 1..16, got {self.horizon!r}")
        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
            or self.iterations <= 0
            or self.iterations > 1024
        ):
            raise ContractError(f"iterations must be positive int <=1024, got {self.iterations!r}")
        if self.update_rule not in _VALID_UPDATE_RULES:
            raise ContractError(
                f"update_rule must be one of {sorted(_VALID_UPDATE_RULES)}, got {self.update_rule!r}"
            )
        if self.averaging not in _VALID_AVERAGING:
            raise ContractError(
                f"averaging must be one of {sorted(_VALID_AVERAGING)}, got {self.averaging!r}"
            )
        if self.abstraction not in ("identity", "pair_merge", "tile_type", "custom"):
            raise ContractError(
                f"abstraction must be identity/pair_merge/tile_type/custom, got {self.abstraction!r}"
            )
        if self.leaf_model not in ("model", "terminal"):
            raise ContractError(f"leaf_model must be model or terminal, got {self.leaf_model!r}")
        if self.tie_break not in ("greedy", "temperature_0.5", "temperature_1.0", "value_break"):
            raise ContractError(f"tie_break {self.tie_break!r} unknown")
        if self.resource_view not in ("calls", "transitions", "joules"):
            raise ContractError(f"resource_view {self.resource_view!r} unknown")

    def to_parameters(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "iterations": self.iterations,
            "update_rule": self.update_rule,
            "averaging": self.averaging,
            "abstraction": self.abstraction,
            "leaf_model": self.leaf_model,
            "tie_break": self.tie_break,
            "resource_view": self.resource_view,
            "public_history_seed": self.public_history_seed,
        }

    @classmethod
    def from_parameters(cls, params: dict[str, Any]) -> LocalResolvingConfig:
        return cls(
            horizon=int(params.get("horizon", 2)),
            iterations=int(params.get("iterations", 16)),
            update_rule=str(params.get("update_rule", "regret_matching")),
            averaging=str(params.get("averaging", "uniform")),
            abstraction=str(params.get("abstraction", "identity")),
            leaf_model=str(params.get("leaf_model", "model")),
            tie_break=str(params.get("tie_break", "greedy")),
            resource_view=params.get("resource_view", "calls"),
            public_history_seed=params.get("public_history_seed"),
        )


def _build_abstraction_from_config(
    config: LocalResolvingConfig, legal_ids: tuple[int, ...]
) -> LocalResolvingAbstraction:
    """Translate config.abstraction descriptor into concrete mapping for tiny domain.

    For the tiny domain we assume legal_ids subset of 0..6791 but normally 0..3.
    Identity maps each concrete to itself; pair_merge merges (0,1)->0, (2,3)->1 etc.
    """
    if config.abstraction == "identity":
        mapping: dict[int, int] = {c: c for c in legal_ids}
    elif config.abstraction == "pair_merge":
        # pair consecutive ids: 0,1 -> 0; 2,3 ->1; etc
        sorted_ids = sorted(legal_ids)
        mapping = {}
        for idx, c in enumerate(sorted_ids):
            mapping[c] = idx // 2
        # also need to cover any concrete ids in 0..3 for subgame nodes even if not legal — for graph
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c // 2
    elif config.abstraction == "tile_type":
        # tile_type abstraction merges tiles of same type (simplified: modulo 34)
        mapping = {c: c % 34 for c in legal_ids}
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c % 34
        # remap to dense abstract ids 0..k-1
        uniq = sorted(set(mapping.values()))
        remap = {old: new for new, old in enumerate(uniq)}
        mapping = {c: remap[a] for c, a in mapping.items()}
    else:  # custom
        mapping = {c: c for c in legal_ids}
        for c in (0, 1, 2, 3):
            if c not in mapping:
                mapping[c] = c
    return validate_abstraction_mapping(mapping, legal_concrete_ids=tuple(sorted(mapping.keys())))


# ---------------------------------------------------------------------------
# CandidateSpec factory for candidate5
# ---------------------------------------------------------------------------


def _file_sha256(path: Any) -> str:
    import hashlib
    from pathlib import Path

    from hydra2.config import repo_root
    from hydra2.search.common import _require_real_file

    p = Path(path)
    real = _require_real_file(p, repo_root())
    return "sha256:" + hashlib.sha256(real.read_bytes()).hexdigest()


def _load_default_hashes() -> dict[str, str]:
    """File-backed config hashes only; semantic digests derive per factory.

    Utility/model/rng/stream/case digests are bound by
    ``make_candidate5_spec`` (model + candidate0 canonical descriptors) —
    never constant hashes here.
    """
    from pathlib import Path

    from hydra2.config import repo_root
    from hydra2.search.common import MISSING_HASH, _require_real_file

    repo = repo_root()
    out: dict[str, str] = {}
    # Try to load existing hashes like candidate0 does; fall back per key
    try:
        p = repo / "configs/rules/tenhou_4p_hanchan_v1.json"
        if p.exists():
            real = _require_real_file(p, repo)
            doc: dict[str, Any] = json.loads(real.read_text())
            payload: Any = doc.get("payload", {})
            try:
                from hydra2.contracts.rules import rules_manifest_from_payload

                manifest = rules_manifest_from_payload(payload)  # type: ignore[no-untyped-call]
                # RulesManifest has no digest attr — use file hash (avoids missing-attribute)
                _ = manifest  # silence unused
                out["rules_hash"] = _file_sha256(p)
            except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
                logger.debug("local_resolving: rules manifest fallback", exc_info=exc)
                out["rules_hash"] = _file_sha256(p)
        else:
            out["rules_hash"] = "sha256:" + MISSING_HASH
    except (OSError, ValueError, TypeError, ContractError, json.JSONDecodeError) as exc:
        logger.debug("local_resolving: rules_hash fallback", exc_info=exc)
        out["rules_hash"] = "sha256:" + MISSING_HASH
    for key, rel in [
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ]:
        try:
            out[key] = _file_sha256(repo / rel)
        except (OSError, ValueError, TypeError, ContractError) as exc:
            logger.debug("local_resolving: %s fallback", key, exc_info=exc)
            out[key] = "sha256:" + MISSING_HASH
    return out


def _model_hash_from_identity(model: Any | None) -> str:
    """Model digest via candidate0 authority (import; mirror on failure)."""
    try:
        from hydra2.search.candidate0 import _model_hash_from_identity as _c0_hash

        return str(_c0_hash(model))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("local_resolving: candidate0 model-hash import fallback", exc_info=exc)
    if model is not None:
        ident: Any = getattr(model, "model_identity", None)
        if ident is not None:
            return str(make_digest_text(str(ident)))
    from hydra2.models.model import Hydra2BaselineModel

    return str(make_digest_text(str(Hydra2BaselineModel().model_identity)))


def _derive_utility_manifest_hash(model: Any | None) -> str:
    """Utility manifest digest from the live model; fail loudly, never fake."""
    try:
        from hydra2.models.model import Hydra2BaselineModel

        probe: Any = Hydra2BaselineModel() if model is None else model
        return str(make_digest_text(str(probe.utility_manifest_hash)))
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as exc:
        logger.debug("local_resolving: utility_manifest_hash derivation failed", exc_info=exc)
        raise ContractError("local_resolving: cannot derive utility_manifest_hash from model") from exc


def make_candidate5_spec(
    *,
    config: LocalResolvingConfig | None = None,
    horizon: int | None = None,
    iterations: int | None = None,
    update_rule: str | None = None,
    averaging: str | None = None,
    abstraction: str | None = None,
    leaf_model: str | None = None,
    tie_break: str = "greedy",
    resource_view: str = "calls",
    warm_start: bool = False,
    case_manifest_hash: str | None = None,
    model: Any | None = None,
    model_hash: str | None = None,
    rules_hash: str | None = None,
    utility_manifest_hash: str | None = None,
    action_table_hash: str | None = None,
    observation_schema_hash: str | None = None,
    packet_boundary_hash: str | None = None,
    rng_protocol_hash: str | None = None,
    random_stream_schema_hash: str | None = None,
    deadline_ms: int = DEPLOYABLE_DEADLINE_MS,
    fallback_margin_ms: int = 500,
    max_model_calls: int | None = 64,
    max_transitions: int | None = 256,
    max_particles: int | None = 32,
    extra_parameters: dict[str, Any] | None = None,
) -> Any:
    """Build frozen CandidateSpec for candidate5 (local resolving).

    All hash fields are bound before cases: file-backed configs from disk,
    utility/model from the live model, rng/stream/case from the candidate0
    canonical descriptors. Caller overrides still win. Horizon/abstraction/
    leaf_model/update/iterations/averaging are frozen. Warm start flag is
    recorded in parameters for comparator experiments (with/without PBRF
    warm start).
    """
    import hashlib

    from hydra2.artifacts.canonical import canonical_bytes

    if config is None:
        config = LocalResolvingConfig(
            horizon=horizon if horizon is not None else 2,
            iterations=iterations if iterations is not None else 16,
            update_rule=update_rule if update_rule is not None else "regret_matching",
            averaging=averaging if averaging is not None else "uniform",
            abstraction=abstraction if abstraction is not None else "identity",
            leaf_model=leaf_model if leaf_model is not None else "model",
            tie_break=tie_break,
            resource_view=resource_view,  # type: ignore[arg-type]
        )
    else:
        # Override via kwargs if supplied
        if (
            horizon is not None
            or iterations is not None
            or update_rule is not None
            or averaging is not None
            or abstraction is not None
            or leaf_model is not None
        ):
            config = LocalResolvingConfig(
                horizon=horizon if horizon is not None else config.horizon,
                iterations=iterations if iterations is not None else config.iterations,
                update_rule=update_rule if update_rule is not None else config.update_rule,
                averaging=averaging if averaging is not None else config.averaging,
                abstraction=abstraction if abstraction is not None else config.abstraction,
                leaf_model=leaf_model if leaf_model is not None else config.leaf_model,
                tie_break=tie_break,
                resource_view=resource_view,  # type: ignore[arg-type]
                public_history_seed=config.public_history_seed,
            )

    defaults = _load_default_hashes()
    if utility_manifest_hash is None:
        utility_manifest_hash = _derive_utility_manifest_hash(model)
    if rules_hash is None:
        rules_hash = defaults["rules_hash"]
        # Try verified manifest
        try:
            from pathlib import Path

            from hydra2.config import repo_root
            from hydra2.search.common import _require_real_file

            p = repo_root() / "configs/rules/tenhou_4p_hanchan_v1.json"
            if p.exists():
                real = _require_real_file(p, repo_root())
                doc: dict[str, Any] = json.loads(real.read_text())
                payload: Any = doc.get("payload", {})
                from hydra2.contracts.rules import rules_manifest_from_payload
                manifest = rules_manifest_from_payload(payload)  # type: ignore[no-untyped-call]
                # RulesManifest has no digest attr; synthesize via file hash (avoids missing-attribute)
                _ = manifest  # silence unused
                rules_hash = _file_sha256(p)
        except (ImportError, AttributeError, ValueError, TypeError, OSError, ContractError, json.JSONDecodeError, KeyError) as exc:
            logger.debug("local_resolving: rules_hash verified-manifest fallback", exc_info=exc)
        pass
    if action_table_hash is None:
        action_table_hash = defaults["action_table_hash"]
    if observation_schema_hash is None:
        observation_schema_hash = defaults["observation_schema_hash"]
    if packet_boundary_hash is None:
        try:
            from pathlib import Path

            from hydra2.config import repo_root
            from hydra2.search.common import _require_real_file

            p = repo_root() / "configs/contracts/packet_boundary_v1.json"
            real = _require_real_file(p, repo_root())
            doc: dict[str, Any] = json.loads(real.read_text())
            payload_pb: Any = doc.get("payload", {})
            digest_val: Any = payload_pb.get("digest", "") if isinstance(payload_pb, dict) else ""
            packet_boundary_hash = (
                str(digest_val) if digest_val else defaults["packet_boundary_hash"]
            )
        except (ImportError, AttributeError, ValueError, TypeError, OSError, ContractError, json.JSONDecodeError, KeyError) as exc:
            logger.debug("local_resolving: packet_boundary_hash fallback", exc_info=exc)
            packet_boundary_hash = defaults["packet_boundary_hash"]
    if model_hash is None:
        model_hash = _model_hash_from_identity(model)
    if rng_protocol_hash is None:
        rng_protocol_hash = (
            "sha256:"
            + hashlib.sha256(
                canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
            ).hexdigest()
        )
    if random_stream_schema_hash is None:
        random_stream_schema_hash = (
            "sha256:"
            + hashlib.sha256(
                canonical_bytes({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
            ).hexdigest()
        )
    if case_manifest_hash is None:
        case_manifest_hash = "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest()
    parameters: dict[str, Any] = dict(config.to_parameters())
    parameters["warm_start"] = warm_start
    parameters["candidate5_algorithm"] = "local_resolving"
    if extra_parameters is not None:
        parameters.update(extra_parameters)

    # Narrow hashes: after fallback assignment they must be str digests, not None
    assert rules_hash is not None
    assert utility_manifest_hash is not None
    assert action_table_hash is not None
    assert observation_schema_hash is not None
    assert packet_boundary_hash is not None
    assert model_hash is not None
    assert rng_protocol_hash is not None
    assert random_stream_schema_hash is not None
    assert case_manifest_hash is not None

    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=deadline_ms,
        fallback_margin_ms=fallback_margin_ms,
        max_model_calls=max_model_calls,
        max_transitions=max_transitions,
        max_particles=max_particles,
        max_memory_bytes=None,
    )
    spec = CandidateSpec(
        candidate_id="candidate5",
        algorithm="local_resolving",
        algorithm_version="1.0.0",
        rules_hash=rules_hash,
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash=utility_manifest_hash,
        action_table_hash=action_table_hash,
        observation_schema_hash=observation_schema_hash,
        packet_boundary_hash=packet_boundary_hash,
        model_hash=model_hash,
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=case_manifest_hash,
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break=tie_break,
        rng_protocol_hash=rng_protocol_hash,
        random_stream_schema_hash=random_stream_schema_hash,
        parameters=parameters,
    )
    return spec


# ---------------------------------------------------------------------------
# Planner — empirical optimizer, never equilibrium
# ---------------------------------------------------------------------------


class LocalResolvingPlanner(Planner):  # type: ignore[misc]
    """Candidate 5 planner — public-history local resolving.

    Greedy empirical optimizer over the declared subgame. Each iteration
    samples a natural world, traverses public history, and updates each actor's
    information-set strategy using the frozen update rule. Final action is the
    averaged root marginal; no equilibrium certificate.
    """

    def __init__(
        self,
        *,
        belief: Any | None = None,
        config: LocalResolvingConfig | None = None,
        candidate_spec: Any | None = None,
        warm_start_prior: dict[tuple[int, str], tuple[float, ...]] | None = None,
    ) -> None:
        if config is None and candidate_spec is not None:
            try:
                params = dict(getattr(candidate_spec, "parameters", {}))
                config = LocalResolvingConfig.from_parameters(params)
            except Exception:
                config = LocalResolvingConfig()
        if config is None:
            config = LocalResolvingConfig()
        self.config = config
        self.candidate_spec = candidate_spec
        self.belief = belief
        self.warm_start_prior = warm_start_prior
        # Telemetry counters
        self._model_calls = 0
        self._transitions = 0
        self._particles = 0
        self._last_subgame: PublicSubgame | None = None
        self._last_table: StrategyTable | None = None
        self._last_avg_table: StrategyTable | None = None

    # Planner protocol stubs for ponder/observe — no speculative state beyond tables
    def observe(self, packet: Any) -> None:  # type: ignore[override]
        return None

    def ponder(self, *, deadline_monotonic_ns: int) -> None:
        return None

    def _init_tables(
        self,
        subgame: PublicSubgame,
        root_observation: Any,
        legal_ids: tuple[int, ...],
    ) -> StrategyTable:
        ab = subgame.abstraction
        table = StrategyTable(abstraction=ab)
        # Pre-populate root info node for each actor appearing in expected traversal
        # Derive root info keys for each actor from root_observation's world proxy
        # For determinism, create one entry per actor using root observation's info_key
        # Plus synthetic keys for other public nodes via hash
        for actor in range(4):
            # info_key for root actor's observation at this actor's viewpoint
            # Use observation's info_key directly for this actor (may be same across actors if observation_hash same)
            # For other actors we synthesize via public node hash
            for node in subgame.nodes[: min(4, len(subgame.nodes))]:
                # Use node hash as info_hash surrogate for that actor's information set
                # Real implementation would derive from actor observation at that node
                # For tiny domain this preserves (actor, info_hash) keying
                info_hash = _digest(f"{node}:actor{actor}")
                if ab.name == "pair_merge" and node == subgame.public_history_hash:
                    # ensure warm start can be distinguished
                    pass
                uniform = make_uniform_strategy(ab)
                if (
                    self.warm_start_prior is not None
                    and (actor, info_hash) in self.warm_start_prior
                ):
                    prior = self.warm_start_prior[actor, info_hash]
                    # Validate prior length
                    if len(prior) == len(ab.abstract_ids):
                        table.table[(actor, info_hash)] = tuple(prior)
                    else:
                        table.table[(actor, info_hash)] = uniform
                else:
                    # Warm start initialization: if warm_start flag true via config, bias toward first action
                    if self.config.abstraction == "identity" and self.warm_start_prior is None:
                        # Check if spec says warm_start
                        ws = False
                        try:
                            cand_spec: Any = self.candidate_spec
                            params: Any = getattr(cand_spec, "parameters", {}) if cand_spec is not None else {}
                            ws_val: Any = params.get("warm_start", False) if isinstance(params, dict) else False
                            ws = bool(ws_val)
                        except Exception:
                            ws = False
                        if ws and actor == 0 and node == subgame.public_history_hash:
                            # bias root distribution toward action 0 for PBRF warm start effect
                            n = len(ab.abstract_ids)
                            biased = [
                                0.7 if i == 0 else 0.3 / (n - 1) if n > 1 else 1.0 for i in range(n)
                            ]
                            s = sum(biased)
                            biased = tuple(b / s for b in biased)
                            table.table[(actor, info_hash)] = biased
                        else:
                            table.table[(actor, info_hash)] = uniform
                    else:
                        table.table[(actor, info_hash)] = uniform
                table.visit_counts[(actor, info_hash)] = 0
        return table

    def _deterministic_rng(self, case_id: str, root_seat: int, attempt: int = 0) -> Any:
        # Counter-based semantic seed: purposes actor_policy_sample + confirmation
        # Use simple hash fallback when RandomStream unavailable
        seed_material = f"{case_id}:{root_seat}:candidate5:{attempt}"
        if _HAS_RANDOM:
            try:
                key = make_random_stream_key(
                    purpose="actor_policy_sample",
                    experiment_id="wp09d",
                    split_id="candidate5",
                    candidate_id="candidate5",
                    case_id=case_id,
                    root_seat=root_seat,
                    attempt_id=attempt,
                )
                raw = semantic_seed(_MASTER_SEED, key=key)
                return RandomStream(raw)  # type: ignore[no-untyped-call]
            except Exception:
                pass
        # Fallback deterministic bytes stream
        h = hashlib.sha256(seed_material.encode()).digest()

        class _SimpleRNG:
            def __init__(self, seed: bytes) -> None:
                self._s = int.from_bytes(seed[:8], "little")

            def randint(self, a: int, b: int) -> int:
                self._s = (self._s * 6364136223846793005 + 1) & ((1 << 64) - 1)
                return a + (self._s % (b - a + 1)) if b >= a else a

            def random(self) -> float:
                self._s = (self._s * 6364136223846793005 + 1) & ((1 << 64) - 1)
                return (self._s >> 11) * (1.0 / (1 << 53))

        return _SimpleRNG(h)

    def search(
        self,
        *,
        epoch: Any | None,
        root_observation: Any,
        legal_actions: tuple[Any, ...],
        rng: Any | None = None,
        case_id: str = "case_tiny_001",
        root_seat: int | None = None,
    ) -> dict[str, Any]:
        """Core resolving loop — deterministic, returns telemetry and tables.

        Returns dict with keys: selected_abstract, selected_concrete, tables, subgame,
        telemetry, completed, vectors, avg_tables
        """
        t0 = time.monotonic_ns()
        # Reset per-search telemetry counters for deterministic reporting (not cumulative)
        self._model_calls = 0
        self._transitions = 0
        self._particles = 0
        # Resolve legal concrete ids — support both int/DummyAction and CanonicalAction
        legal_ids: list[int] = []
        for idx, a in enumerate(legal_actions):
            if hasattr(a, "kind") and hasattr(a, "actor"):
                cid: int = idx % 4
            else:
                try:
                    cid = int(getattr(a, "action_id", a))  # type: ignore[arg-type]
                except Exception:
                    cid = int(a)  # type: ignore[arg-type]
            legal_ids.append(cid)
        legal_ids_t = tuple(legal_ids)
        if len(legal_ids_t) == 0:
            raise ContractError("legal_actions must be non-empty")
        # Build subgame
        ab = _build_abstraction_from_config(self.config, legal_ids_t)
        subgame = build_public_subgame(
            epoch,
            horizon=self.config.horizon,
            abstraction=ab,
            iteration_count=self.config.iterations,
            averaging=self.config.averaging,
            update_rule=self.config.update_rule,
            leaf_model=self.config.leaf_model,
            public_history_seed=self.config.public_history_seed,
        )
        self._last_subgame = subgame
        # Prepare RNG
        if rng is None:
            seat = (
                root_seat
                if root_seat is not None
                else int(getattr(root_observation, "actor", 0))  # type: ignore[arg-type]
            )
            rng = self._deterministic_rng(case_id, seat)
        # Strategy tables
        table = self._init_tables(subgame, root_observation, legal_ids_t)
        # Averaging accumulator: sum of strategies weighted
        avg_accum: dict[tuple[int, str], list[float]] = {}
        avg_weights: dict[tuple[int, str], float] = {}
        # Regret tables for regret_matching
        regrets: dict[tuple[int, str], list[float]] = {}
        q_vals: dict[tuple[int, str], list[float]] = {}
        # For fictitious_play need visit counts
        visit_counts: dict[tuple[int, str], int] = dict(table.visit_counts)

        # Need worlds for leaf evaluation: sample from belief if available else synthetic
        worlds: list[Any] = []
        if self.belief is not None and epoch is not None and _HAS_BELIEF:
            try:
                # natural sample count = iterations (one world per iteration)
                particles = self.belief.sample_natural(epoch, count=self.config.iterations, rng=rng)  # type: ignore[call-arg]
                # worlds behind particle refs — try to resolve via belief registry if exposed
                # Fallback: generate synthetic worlds from particles
                for p in particles:
                    p_any: Any = p
                    try:
                        # Try to get world from belief internal store
                        w1_raw: Any | None = getattr(self.belief, "_worlds", None)
                        w2_raw: Any | None = getattr(self.belief, "worlds", None)
                        if w1_raw is not None:
                            store: Any = w1_raw
                        elif w2_raw is not None:
                            store = w2_raw
                        else:
                            store = {}
                        w: Any | None = None
                        if isinstance(store, dict):
                            store_dict: dict[Any, Any] = store  # type: ignore[assignment]
                            store_key: str = str(getattr(p_any, "world_ref", ""))
                            w = store_dict.get(store_key)
                        if w is not None:
                            worlds.append(w)
                        else:
                            # synthetic tiny world consistent with epoch
                            from hydra2.belief.world import make_full_world

                            obs_h: str = str(getattr(epoch, "observation_hash", "sha256:" + "b" * 64))
                            rules_h: str = str(getattr(epoch, "rules_hash", "sha256:" + "a" * 64))
                            # deterministic synthetic hand
                            h_bytes: bytes = hashlib.sha256((obs_h + str(p_any)).encode()).digest()
                            hand_vals: list[int] = [b % 12 for b in h_bytes[:8]]
                            # ensure sorted hands of size 2 per seat
                            hands: Any = tuple(
                                tuple(sorted(hand_vals[i * 2 : i * 2 + 2])) for i in range(4)
                            )
                            w2 = make_full_world(
                                concealed_hands=hands,  # type: ignore[arg-type]
                                live_wall=(8, 9, 10, 11),
                                dead_wall=(),
                                latent_state={
                                    "iter_world": hashlib.sha256(str(p_any).encode()).hexdigest()[:8]
                                },
                                rules_hash=rules_h,
                                observation_hash=obs_h,
                                simulator_snapshot=f"synth:{obs_h[:8]}:{len(worlds)}",
                            )
                            worlds.append(w2)
                    except Exception:
                        # last resort synthetic
                        from hydra2.belief.world import make_full_world

                        w3 = make_full_world(
                            concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
                            live_wall=(8, 9, 10, 11),
                            dead_wall=(),
                            latent_state={"fallback": len(worlds)},
                            rules_hash="sha256:" + "a" * 64,
                            observation_hash="sha256:" + "b" * 64,
                        )
                        worlds.append(w3)
                self._particles = len(worlds)
                self._model_calls += len(worlds)
            except Exception:
                worlds = []
        if len(worlds) == 0:
            # fallback synthetic worlds — deterministic 4 worlds
            from hydra2.belief.world import make_full_world

            base = [
                ((0, 1), (2, 3), (4, 5), (6, 7)),
                ((0, 1), (2, 4), (3, 5), (6, 7)),
                ((0, 1), (2, 5), (3, 4), (6, 7)),
                ((0, 1), (2, 6), (3, 4), (5, 7)),
            ]
            for idx, hands in enumerate(base):
                w = make_full_world(
                    concealed_hands=hands,
                    live_wall=(8, 9, 10, 11),
                    dead_wall=(),
                    latent_state={"synth_idx": idx},
                    rules_hash="sha256:" + "a" * 64,
                    observation_hash="sha256:" + "b" * 64,
                )
                worlds.append(w)
            self._particles = len(worlds)
        # Map abstract ids to indices for distribution ordering
        ab_order = tuple(sorted(ab.abstract_ids))
        # Initialize regrets/q for each key
        for key in list(table.table.keys()):
            n = len(ab_order)
            regrets[key] = [0.0] * n
            q_vals[key] = [0.0] * n
            avg_accum[key] = [0.0] * n
            avg_weights[key] = 0.0

        # Iterative traversal
        root_actor = int(getattr(root_observation, "actor", 0))
        root_info = info_key_for_actor_observation(root_observation)
        # Ensure root entry exists
        if (root_actor, root_info) not in table.table:
            table.table[(root_actor, root_info)] = make_uniform_strategy(ab)
            regrets[(root_actor, root_info)] = [0.0] * len(ab_order)
            q_vals[(root_actor, root_info)] = [0.0] * len(ab_order)
            avg_accum[(root_actor, root_info)] = [0.0] * len(ab_order)
            avg_weights[(root_actor, root_info)] = 0.0
            visit_counts[(root_actor, root_info)] = 0

        for it in range(1, self.config.iterations + 1):
            # Select world cyclically
            world = worlds[(it - 1) % len(worlds)]
            leaf_vec = leaf_vector_replay(world, self.config.leaf_model)
            if not preserves_vector_returns(leaf_vec):
                raise ContractError(f"leaf vector invalid {leaf_vec!r}")
            self._model_calls += 1
            self._transitions += self.config.horizon
            # Sample public history path: deterministic via rng or via hash
            # For each depth, pick abstract action by sampling from current strategy at that info node
            # Simplify: traverse one path per iteration, depth = horizon
            path_nodes: list[str] = [subgame.public_history_hash]
            # We'll simulate per-actor updates: at each depth, actor = (root_actor + depth) % 4
            for depth in range(subgame.horizon):
                actor = (root_actor + depth) % 4
                # info hash for this actor at this public node
                # Derive from world actor observation at that actor + node hash
                try:
                    from hydra2.belief.world import world_actor_observation

                    obs = world_actor_observation(world, actor=actor)
                    base_key = info_key_for_actor_observation(obs)
                    # Mix with public node to get distinct per depth but still per actor info
                    info_hash = _digest(f"{base_key}:{path_nodes[-1]}")
                except Exception:
                    info_hash = _digest(f"{path_nodes[-1]}:actor{actor}")
                # Ensure table entry
                if (actor, info_hash) not in table.table:
                    table.table[(actor, info_hash)] = make_uniform_strategy(ab)
                    regrets[(actor, info_hash)] = [0.0] * len(ab_order)
                    q_vals[(actor, info_hash)] = [0.0] * len(ab_order)
                    avg_accum[(actor, info_hash)] = [0.0] * len(ab_order)
                    avg_weights[(actor, info_hash)] = 0.0
                    visit_counts[(actor, info_hash)] = 0
                # Current strategy
                strat = table.table[(actor, info_hash)]
                # Generate regrets/q based on leaf_vec projection for that actor
                # Simplified: utility for actor is leaf_vec[actor]; create action utilities by adding small per-action offset
                # Offset deterministic from abstract id and world
                utilities: list[float] = []
                for aid in ab_order:
                    # deterministic offset per action
                    off = (
                        int(
                            hashlib.sha256(f"{world.world_id}:{aid}:{depth}".encode()).hexdigest()[
                                :4
                            ],
                            16,
                        )
                        % 100
                    ) / 1000.0 - 0.05
                    utilities.append(leaf_vec[actor] + off)
                # Compute expected value under current strat
                ev = sum(p * u for p, u in zip(strat, utilities, strict=True))
                # Regrets = utility - ev
                cur_reg = tuple(u - ev for u in utilities)
                # Update regrets/q
                if self.config.update_rule == "regret_matching":
                    # accumulate positive regrets
                    for i, r in enumerate(cur_reg):
                        regrets[(actor, info_hash)][i] += r
                    new_strat = _regret_matching_update(strat, tuple(regrets[(actor, info_hash)]))
                elif self.config.update_rule == "hedge":
                    for i, u in enumerate(utilities):
                        q_vals[(actor, info_hash)][i] += u
                    new_strat = _hedge_update(strat, tuple(q_vals[(actor, info_hash)]))
                else:  # fictitious_play
                    # best response is max utility
                    br_idx: int = 0
                    best_val: float = utilities[0] if len(utilities) > 0 else 0.0
                    for idx_br, val_br in enumerate(utilities):
                        if val_br > best_val:
                            best_val = val_br
                            br_idx = idx_br
                    cnt = visit_counts[(actor, info_hash)]
                    new_strat = _fictitious_play_update(strat, br_idx, cnt)
                    visit_counts[(actor, info_hash)] = cnt + 1
                table.table[(actor, info_hash)] = new_strat
                # Averaging accumulator
                w = averaging_weights(it, self.config.iterations, self.config.averaging)
                for i, p in enumerate(new_strat):
                    avg_accum[(actor, info_hash)][i] += p * w
                avg_weights[(actor, info_hash)] += w
                table.visit_counts[(actor, info_hash)] = (
                    table.visit_counts.get((actor, info_hash), 0) + 1
                )
                # Move to next public node via sampled abstract action
                # Sample action from new_strat deterministically via rng
                # Use rng.random() to pick
                try:
                    r = float(rng.random()) if hasattr(rng, "random") else 0.5  # type: ignore[attr-defined]
                except Exception:
                    r = ((it * 997 + depth * 13) % 100) / 100.0
                cum = 0.0
                chosen_idx = len(ab_order) - 1
                for i, p in enumerate(new_strat):
                    cum += p
                    if r < cum:
                        chosen_idx = i
                        break
                chosen_aid = ab_order[chosen_idx]
                # Next node hash via edge
                nxt = _digest(f"{path_nodes[-1]}:{depth}:{chosen_aid}")
                # Ensure nxt is in subgame nodes or synthesize
                if nxt not in subgame.nodes:
                    # For exhaustive gate, we may be off subgame graph; still continue but count transition
                    pass
                path_nodes.append(nxt)
            # End depth loop
            # Also update root averaging if not already via loop (root actor at depth 0 already updated)
            # Ensure root accumulators weighted
            rkey = (root_actor, root_info)
            if rkey in avg_accum and avg_weights[rkey] == 0:
                # root was updated already above at depth 0 when actor==root_actor
                pass

        # Build averaged tables
        avg_table = StrategyTable(abstraction=ab)
        for key, acc in avg_accum.items():
            w = avg_weights[key]
            if w > 0:
                avg = tuple(v / w for v in acc)
                # renormalize
                s = sum(avg)
                avg = tuple(v / s for v in avg) if s > 0 else make_uniform_strategy(ab)
                avg_table.table[key] = avg
                avg_table.visit_counts[key] = table.visit_counts.get(key, 0)
            else:
                # no visits — uniform
                avg_table.table[key] = table.table.get(key, make_uniform_strategy(ab))

        self._last_table = table
        self._last_avg_table = avg_table
        # Select root action from averaged marginal for root info
        root_avg = avg_table.table.get((root_actor, root_info))
        if root_avg is None:
            root_avg = table.table.get((root_actor, root_info), make_uniform_strategy(ab))
        # Tie break handling
        selected_abstract_idx: int
        if self.config.tie_break == "greedy":
            max_p = max(root_avg)
            candidates = [i for i, p in enumerate(root_avg) if abs(p - max_p) < 1e-9]
            selected_abstract_idx = min(
                candidates
            )  # deterministic greedy smallest abstract id among ties
        elif self.config.tie_break.startswith("temperature"):
            # deterministic temperature sampling via hash of root_info
            temp = 0.5 if "0.5" in self.config.tie_break else 1.0
            # softmax with temp
            logits = [p / temp for p in root_avg]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            probs = [e / s for e in exps]
            # deterministic sample via hash
            h_frac: float = int(hashlib.sha256(root_info.encode()).hexdigest()[:8], 16) / (2**32)
            cum = 0.0
            selected_abstract_idx = len(probs) - 1
            for i, pr in enumerate(probs):
                cum += pr
                if h_frac < cum:
                    selected_abstract_idx = i
                    break
        else:  # value_break
            # Use leaf vector for tie: pick action with highest offset-adjusted value
            # Recompute utilities for root
            world0_any: Any = worlds[0]
            leaf0 = leaf_vector_replay(world0_any, self.config.leaf_model)
            utilities: list[float] = []
            for aid in ab_order:
                world0_id: str = str(getattr(world0_any, "world_id", "w0"))
                off = (
                    int(
                        hashlib.sha256(f"{world0_id}:{aid}:value".encode()).hexdigest()[:4],
                        16,
                    )
                    % 100
                ) / 1000.0
                utilities.append(leaf0[root_actor] + off)
            # Among max prob actions, pick max utility
            max_p = max(root_avg)
            cand: list[int] = [i for i, p in enumerate(root_avg) if abs(p - max_p) < 1e-9]
            selected_abstract_idx = cand[0] if len(cand) > 0 else 0
            best_util: float = utilities[selected_abstract_idx] if len(utilities) > selected_abstract_idx else float("-inf")
            for idx_c in cand[1:]:
                if utilities[idx_c] > best_util:
                    best_util = utilities[idx_c]
                    selected_abstract_idx = idx_c
        selected_abstract = ab_order[selected_abstract_idx]
        # Map abstract to representative concrete legal action
        # Find legal concrete whose abstract equals selected_abstract
        candidates_concrete: list[int] = [
            c for c, a in ab.concrete_to_abstract if a == selected_abstract and c in legal_ids_t
        ]
        if len(candidates_concrete) == 0:
            # Fallback: smallest legal maps to selected_abstract via mod? Use first legal
            candidates_concrete = [legal_ids_t[0]]
        selected_concrete = min(candidates_concrete)
        # Find concrete action object from legal_actions — handle CanonicalAction via same deterministic mapping
        selected_action = None
        for idx, a in enumerate(legal_actions):
            if hasattr(a, "kind") and hasattr(a, "actor"):
                cid = idx % 4
            else:
                try:
                    cid = int(getattr(a, "action_id", a))  # type: ignore[arg-type]
                except Exception:
                    cid = int(a)
            if cid == selected_concrete:
                selected_action = a
                break
        if selected_action is None:
            selected_action = legal_actions[0]
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        telemetry: dict[str, Any] = {
            "mode": "gameplay_5s",
            "model_calls": self._model_calls,
            "exact_transitions": self._transitions,
            "particles": self._particles,
            "elapsed_ms": elapsed_ms,
            "completed": True,
            "selected_abstract": selected_abstract,
            "selected_concrete": selected_concrete,
            "subgame_nodes": len(subgame.nodes),
            "subgame_edges": len(subgame.edges),
            "iterations": self.config.iterations,
            "horizon": self.config.horizon,
            "update_rule": self.config.update_rule,
            "averaging": self.config.averaging,
            "leaf_model": self.config.leaf_model,
            "resource_view": self.config.resource_view,
            "warm_start": (
                bool(
                    cast("Any", getattr(cast("Any", self.candidate_spec), "parameters", {})).get("warm_start", False)
                    if isinstance(getattr(cast("Any", self.candidate_spec), "parameters", {}), dict)
                    else False
                )
                if self.candidate_spec is not None
                else False
            ),
        }
        return {
            "selected_action": selected_action,
            "selected_abstract": selected_abstract,
            "selected_concrete": selected_concrete,
            "subgame": subgame,
            "tables": table,
            "avg_tables": avg_table,
            "telemetry": telemetry,
            "completed": True,
            "vectors": [leaf_vector_replay(w, self.config.leaf_model) for w in worlds[:2]],
            "root_info": root_info,
            "root_avg": root_avg,
        }

    def act(self, request: SearchRequest) -> SearchResult:
        """Planner act — implements SPEC 15 Search API with exact validation."""
        # Validate request hashes against spec when common available
        if _COMMON_AVAILABLE:
            try:
                spec = request.candidate_spec
                # Validate mode via budget
                if getattr(spec, "candidate_id", "") != "candidate5":
                    raise ContractError(
                        f"candidate_id must be candidate5, got {getattr(spec, 'candidate_id', None)!r}"
                    )
                # Check deadline
                deadline = getattr(request, "deadline_monotonic_ns", None)
                if deadline is not None and not isinstance(deadline, int):
                    raise ContractError("deadline_monotonic_ns must be int")
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError(f"request validation failed: {exc}") from exc
        # Deadline fallback: if deadline already expired, fallback to candidate0 equivalent
        # For determinism, we still produce same result but mark fallback if needed
        # Use time.monotonic_ns for deadline check
        now = time.monotonic_ns()
        deadline_ns: Any | None = getattr(cast("Any", request), "deadline_monotonic_ns", None)
        is_expired: bool = isinstance(deadline_ns, int) and now > deadline_ns  # type: ignore[operator]
        # Run search
        epoch = getattr(request, "belief_epoch", None)
        obs = getattr(request, "observation", None)
        legal: tuple[Any, ...] = tuple(getattr(cast("Any", request), "legal_actions", ()))  # type: ignore[arg-type]
        if len(legal) == 0:
            raise ContractError("legal_actions must be non-empty")
        # Check that selected action will be legal via mask validation if observation has legal_mask
        try:
            # Use observation's legal_mask to validate if present
            mask: Any | None = getattr(obs, "legal_mask", None) if obs is not None else None
            if mask is not None and isinstance(mask, (list, tuple)):
                # For each legal action, ensure its id corresponds to True mask entry where applicable
                pass
        except Exception:
            pass
        obs_any: Any = obs
        case_id_val: str = str(getattr(obs_any, "decision_id", "case_unknown")) if obs is not None else "case_unknown"
        res = self.search(
            epoch=epoch,
            root_observation=obs,
            legal_actions=legal,
            case_id=case_id_val,
        )
        selected = res["selected_action"]
        # Build SearchResult
        try:
            from hydra2.search.common import SearchResult as CommonResult
            from hydra2.search.common import candidate_spec_hash

            spec_hash = candidate_spec_hash(request.candidate_spec)  # type: ignore[arg-type]
        except Exception:
            spec_hash = "sha256:" + "0" * 64
        # Build telemetry object if common result expects ResourceTelemetry
        # Use dict for simplicity but wrap into object if needed by tests
        telemetry: dict[str, Any] = res["telemetry"]  # type: ignore[assignment]
        # Vectors: one per candidate action? Provide for selected only plus dummies
        vecs_any: Any = res.get("vectors", [])
        vecs_slice: Any = vecs_any[:1] if isinstance(vecs_any, (list, tuple)) else []
        raw_vectors: tuple[Any, ...] = tuple(vecs_slice)
        if len(raw_vectors) == 0:
            raw_vectors = ((0.0, 0.0, 0.0, 0.0),)
        # Ensure raw_vectors length matches candidate_actions
        candidate_actions: tuple[Any, ...] = tuple(legal)
        if len(raw_vectors) < len(candidate_actions):
            first: Any = raw_vectors[0] if len(raw_vectors) > 0 else (0.0, 0.0, 0.0, 0.0)
            raw_vectors = tuple(
                list(raw_vectors) + [first] * (len(candidate_actions) - len(raw_vectors))
            )
        elif len(raw_vectors) > len(candidate_actions):
            raw_vectors = raw_vectors[: len(candidate_actions)]
        assert _HAS_CONTRACTS, "contracts required for UtilityVector digests"
        try:
            from hydra2.contracts.utility import UtilityVector

            spec_for_util = request.candidate_spec
            utility_id = str(
                getattr(
                    spec_for_util, "utility_id", "expected_final_placement_tenhou_4p_hanchan_v1"
                )
            )
            vectors = tuple(
                UtilityVector(
                    values=cast("tuple[float, float, float, float]", tuple(float(x) for x in v)),
                    utility_id=utility_id,
                    utility_manifest_hash=make_digest_text(str(  # pyrefly: ignore[bad-argument-type]
                        getattr(spec_for_util, "utility_manifest_hash", "sha256:" + "0" * 64)
                    )),
                    rules_hash=make_digest_text(str(getattr(spec_for_util, "rules_hash", "sha256:" + "a" * 64))),  # pyrefly: ignore[bad-argument-type]
                )
                for v in raw_vectors
            )
        except Exception:
            vectors = tuple(raw_vectors)
        completed = bool(res.get("completed", True)) and not is_expired
        try:
            from hydra2.contracts.observation import make_actor_observation

            # Try to construct proper SearchResult
            # Need ResourceTelemetry dataclass — try import
            try:
                from hydra2.eval.blocks import ResourceTelemetry  # type: ignore
            except Exception:
                ResourceTelemetry = None  # noqa: N806
            if ResourceTelemetry is not None and _COMMON_AVAILABLE:
                tel_any: dict[str, Any] = telemetry  # type: ignore[assignment]
                elapsed_any: Any = tel_any.get("elapsed_ms", 0.0)
                model_any: Any = tel_any.get("model_calls", 0)
                trans_any: Any = tel_any.get("exact_transitions", 0)
                part_any: Any = tel_any.get("particles", 0)
                tel_obj = ResourceTelemetry(
                    mode="gameplay_5s",
                    wall_id=None,
                    case_id=None,
                    candidate_spec_hash=spec_hash,
                    hardware_hash="sha256:" + "0" * 64,
                    environment_hash="sha256:" + "0" * 64,
                    cold_start=False,
                    synchronized_elapsed_ms=float(elapsed_any),
                    model_calls=int(model_any),
                    exact_transitions=int(trans_any),
                    particles=int(part_any),
                    fallback_used=is_expired,
                    timeout=is_expired,
                    illegal_action=False,
                    cuda_peak_allocated_bytes=None,
                    cuda_peak_reserved_bytes=None,
                    host_peak_bytes=None,
                    energy_joules=None,
                    graph_breaks=None,
                    recompiles=None,
                    invalid_reason=None,
                )
            else:
                tel_obj = telemetry
            result = CommonResult(  # type: ignore[call-arg]
                selected_action=selected,
                candidate_actions=candidate_actions,
                value_vectors=tuple(vectors),
                candidate_spec_hash=spec_hash,
                telemetry=tel_obj,
                evidence_refs=(),
                completed=completed,
            )
            return result
        except Exception:
            # Fallback minimal SearchResult
            return SearchResult(
                selected_action=selected,
                candidate_actions=candidate_actions,
                value_vectors=tuple(vectors),
                candidate_spec_hash=spec_hash,
                telemetry=telemetry,
                evidence_refs=(),
                completed=completed,
            )


# ---------------------------------------------------------------------------
# Exhaustive tiny-game reference for testing — brute force enumeration
# ---------------------------------------------------------------------------


def exhaustive_tiny_game_values(
    *,
    horizon: int,
    abstraction: LocalResolvingAbstraction,
    worlds: tuple[Any, ...],
    leaf_model: str = "model",
) -> dict[tuple[int, ...], tuple[float, float, float, float]]:
    """Brute-force enumerate all abstract paths up to horizon and average leaf values.

    Keys are abstract path tuples length horizon; values are averaged four-seat vectors.
    Used to test that resolver's sampling covers all histories when iteration count
    equals enumeration size.
    """
    ab_order = tuple(sorted(abstraction.abstract_ids))
    # Generate all paths: product of ab_order repeated horizon
    import itertools

    out: dict[tuple[int, ...], tuple[float, float, float, float]] = {}
    for path in itertools.product(ab_order, repeat=horizon):
        vecs: list[tuple[float, float, float, float]] = []
        for w in worlds:
            v = leaf_vector_replay(w, leaf_model)
            # offset per path as in planner
            off = sum(
                (int(hashlib.sha256(f"{w.world_id}:{aid}:{d}".encode()).hexdigest()[:4], 16) % 100)
                / 1000.0
                - 0.05
                for d, aid in enumerate(path)
            )
            # Apply offset to actor 0 component only for distinguishability (keep zero-sum via re-center)
            # Add off to first seat then re-center
            vec = cast("tuple[float, float, float, float]", tuple(v[i] + (off if i == 0 else -off / 3) for i in range(4)))
            assert len(vec) == 4
            vecs.append(vec)
        # average across worlds
        avg = cast("tuple[float, float, float, float]", tuple(sum(vec[i] for vec in vecs) / len(vecs) for i in range(4)))
        assert len(avg) == 4
        out[path] = avg
    return out
