# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dependency fallback shims + re-exported spec symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 5 local resolving — strategy: per-actor tables plus update rules.

Owns the ``(actor, information_node_hash)`` strategy tables with their uniform seeding
and world-id firewall, the frozen update/averaging rules (regret matching, hedge,
fictitious play), the leaf replay helper, and the exhaustive tiny-game reference used
by tests to enumerate every abstract path. Abstraction lives in
:mod:`hydra2.search.local_abstraction`; the CandidateSpec factory lives in
:mod:`hydra2.search.local_spec`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, cast

from hydra2.contracts.common import ContractError, make_digest_text
from hydra2.search.local_abstraction import LocalResolvingAbstraction as LocalResolvingAbstraction
from hydra2.search.local_abstraction import _actor_to_key as _actor_to_key
from hydra2.search.local_abstraction import _digest as _digest
from hydra2.search.local_abstraction import model_vector_for_world as model_vector_for_world
from hydra2.search.local_abstraction import terminal_vector_for_world as terminal_vector_for_world
from hydra2.search.local_shared import FORBIDDEN_IN_STRATEGY_KEY as FORBIDDEN_IN_STRATEGY_KEY
from hydra2.search.local_shared import logger as logger

__all__ = [
    "StrategyTable",
    "apply_update",
    "averaging_weights",
    "exhaustive_tiny_game_values",
    "frozen_averaging_rule_names",
    "frozen_update_rule_names",
    "is_equilibrium_claimed",
    "leaf_vector_replay",
    "make_uniform_strategy",
]

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
# Wave 2 bridge audit: kept Python — regret/hedge/fictitious-play table updates
# are empirical optimizers, not UCT/PUCT/Gumbel selection cuts; no pyfn covers them.

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
            vec = cast(
                "tuple[float, float, float, float]",
                tuple(v[i] + (off if i == 0 else -off / 3) for i in range(4)),
            )
            assert len(vec) == 4
            vecs.append(vec)
        # average across worlds
        avg = cast(
            "tuple[float, float, float, float]",
            tuple(sum(vec[i] for vec in vecs) / len(vecs) for i in range(4)),
        )
        assert len(avg) == 4
        out[path] = avg
    return out
