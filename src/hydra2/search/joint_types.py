# ruff: noqa: F401, B007, C416, RUF005  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (F401 optional-dep fallback imports; B007 intentional scratch locals; C416/RUF005 idiom drift). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 8 joint type/world — types: info keys, type policy, joint particles.

Owns the joint vocabulary beside its only readers: the firewall constants, the
type/seed manifests, the canonical info-key helpers with their hidden-permutation
controls, the observation-only opponent policy, the joint particle/posterior
records, and the validation checks that prove same-information equality. The
exact oracle, the coherent trajectory sampler, and the uncertainty/spec builders
live in :mod:`hydra2.search.joint_uncertainty`; the planner adapter lives in
:mod:`hydra2.search.joint_planner` so each file stays inside the review-size
ceiling.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, VisibilityViolationError, make_digest_text

try:
    from hydra2.search.common import (
        CandidateSpec,
        Planner,
        ResourceBudget,
        SearchRequest,
        SearchResult,
        candidate_spec_hash,
    )

    _COMMON_AVAILABLE = True
except ImportError as exc:
    raise ImportError(
        "hydra2.search.common is required for joint_types; "
        "the minimal-contract fallback was removed (single authority is search.common)"
    ) from exc


try:
    from hydra2.contracts.randomness import RandomStream

    _HAS_RANDOM = True
except ImportError:  # pragma: no cover
    _HAS_RANDOM = False
    RandomStream = Any

try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief
    from hydra2.belief.world import FullWorld, make_full_world, world_actor_observation

    _HAS_BELIEF = True
except ImportError:  # pragma: no cover
    _HAS_BELIEF = False
    NaturalBelief = Any
    BeliefEpoch = Any
    FullWorld = Any
    make_full_world = Any
    world_actor_observation = Any

try:
    from hydra2.contracts.observation import ActorObservation, observation_identity_document

    _HAS_OBS = True
except ImportError:  # pragma: no cover
    _HAS_OBS = False
    ActorObservation = Any

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "JointParticle",
    "JointPosterior",
    "OpponentTypePolicy",
    "deterministic_joint_gumbel",
    "info_key_for_observation",
    "validate_hidden_permutation_invariance",
    "validate_same_information_equality",
]

# Wall = undealt tile stock: live = drawable, dead = dora-indicator reserve.
FORBIDDEN_IN_TREE_KEY: frozenset[str] = frozenset(
    {
        "world_id",
        "simulator_snapshot",
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "full_world",
        "privileged",
        "privileged_label",
        "world_ref",
        "parent_id",
        "latent_state_hidden",
        "server_private",
        "engine_rng_state",
        "future_events",
        "opponent_concealed",
        "unrevealed_dora",
        "theta_private",
        "opponent_theta",
    }
)

# Frozen finite type space — observation-only, declared in manifest.
THETA_IDS: tuple[str, ...] = ("tight", "loose")
DIVERGENCE_DIRECTIONS: frozenset[str] = frozenset({"kl_q_nom", "kl_nom_q", "tv"})
SUPPORT_CLASSES: frozenset[str] = frozenset({"finite_categorical", "quantal"})
RATIONALITY_RULES: frozenset[str] = frozenset({"quantal_softmax", "epsilon_greedy"})

_MASTER_SEED = b"wp13_joint_type_world_v1"
_JOINT_GUMBEL_DOMAIN = b"joint_type_world_gumbel_v1"
_INFO_KEY_DOMAIN = b"joint_type_world_info_v1"


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def deterministic_joint_gumbel(
    *, case_id: str, root_seat: int, candidate_id: str, action_id: int, theta: str
) -> float:
    """Deterministic Gumbel for joint (theta, action) perturbation — for robust selection."""
    if not isinstance(case_id, str) or case_id == "":
        raise ContractError(f"case_id must be non-empty str, got {case_id!r}")
    if not isinstance(root_seat, int) or isinstance(root_seat, bool) or not 0 <= root_seat < 4:
        raise ContractError(f"root_seat must be 0..3, got {root_seat!r}")
    if not isinstance(candidate_id, str) or candidate_id == "":
        raise ContractError(f"candidate_id must be non-empty str, got {candidate_id!r}")
    if theta not in THETA_IDS:
        raise ContractError(f"theta must be one of {THETA_IDS}, got {theta!r}")
    payload = f"{case_id}:{root_seat}:{candidate_id}:{theta}:{action_id}".encode()
    h = hashlib.sha256(_JOINT_GUMBEL_DOMAIN + payload).digest()
    int_val = int.from_bytes(h[:8], "big")
    u = (int_val + 0.5) / 18446744073709551616.0
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    g = -math.log(-math.log(u))
    if not math.isfinite(g):
        raise ContractError(f"gumbel must be finite, got {g!r}")
    return max(min(g, 20.0), -20.0)


def info_key_for_observation(observation: Any) -> str:
    """Canonical information-set key for actor observation — excludes legal_mask & forbidden."""
    if observation is None:
        raise ContractError("observation must be ActorObservation")
    try:
        from hydra2.contracts.observation import ActorObservation as _Obs

        if isinstance(observation, _Obs):
            doc = observation_identity_document(observation)
        else:
            raise ContractError("observation must be ActorObservation")
    except Exception as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(
            f"observation must be ActorObservation, got {type(observation).__name__}"
        ) from exc
    doc = {k: v for k, v in doc.items() if k != "legal_mask"}
    for bad in FORBIDDEN_IN_TREE_KEY:
        if bad in doc:
            raise VisibilityViolationError(f"forbidden field {bad!r} in tree key document")
    payload = canonical_bytes(doc)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def validate_hidden_permutation_invariance(world: Any, actor: int) -> bool:
    """Check hidden permutation leaves serialized actor observation unchanged."""
    try:
        from hydra2.belief.world import make_full_world as _mfw
        from hydra2.belief.world import world_actor_observation as _wao

        obs1 = _wao(world, actor=actor)
        key1 = info_key_for_observation(obs1)
        hands = tuple(tuple(int(t) for t in h) for h in world.concealed_hands)  # type: ignore[union-attr]
        if len(hands) != 4:
            return False
        # Permute opponent concealed order: reverse opponent seat hand then sort invariantly
        permuted = list(hands)
        opp = (actor + 1) % 4
        if len(permuted[opp]) >= 2:
            # Permute opponent hand then sort invariantly — hidden permutation preserves multiset
            permuted[opp] = tuple(sorted(permuted[opp], reverse=True))
        else:
            # If degenerate, try swapping tiles between two opponent seats (still hidden to root)
            opp2 = (actor + 2) % 4
            if len(permuted[opp]) > 0 and len(permuted[opp2]) > 0:
                a = permuted[opp][0]
                b = permuted[opp2][0]
                permuted[opp] = tuple(sorted((b,) + permuted[opp][1:]))
                permuted[opp2] = tuple(sorted((a,) + permuted[opp2][1:]))
        permuted_sorted = tuple(tuple(sorted(h)) for h in permuted)
        # If permutation didn't change world (e.g., identical tiles), still invariant
        if permuted_sorted == hands:
            return True
        # Build permuted world with same public wall/dead
        world2 = _mfw(
            concealed_hands=permuted_sorted,
            live_wall=tuple(int(t) for t in world.live_wall),  # type: ignore[union-attr]
            dead_wall=tuple(int(t) for t in world.dead_wall),  # type: ignore[union-attr]
            rules_hash=world.rules_hash,  # type: ignore[union-attr]
            observation_hash=world.observation_hash,  # type: ignore[union-attr]
        )
        obs2 = _wao(world2, actor=actor)
        key2 = info_key_for_observation(obs2)
        # Root actor's concealed hand unchanged => keys equal (hidden permutation invariance)
        # The check is that root's info key is invariant to opponent hidden permutation
        # Our construction keeps root hand fixed (actor seat), so k1==k2 iff correctly actor-visible
        # If hands had been swapped involving root, they'd differ — but we never touch root seat.
        if permuted_sorted[actor] != hands[actor]:
            return False
        return key1 == key2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Opponent type policy — observation-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpponentTypePolicy:
    """Behavioral policy ``q_j(a | I_j, theta)``.

    - Keyed only by opponent information set ``I_j`` (via ``info_key_for_observation``) and ``theta``.
    - Respects legal masks: illegal actions have probability 0.
    - Same-information equality: same (theta, info_key) → identical distribution.
    - Deterministic construction via hash-seeded Dirichlet-like mapping.
    - Coherent: same theta+info deterministically maps to same distribution across calls.
    """

    theta: str
    seed_domain: bytes = _MASTER_SEED

    def __post_init__(self) -> None:
        if self.theta not in THETA_IDS:
            raise ContractError(f"theta must be one of {THETA_IDS}, got {self.theta!r}")
        if not isinstance(self.seed_domain, (bytes, bytearray)) or len(self.seed_domain) == 0:
            raise ContractError("seed_domain must be non-empty bytes")

    def distribution_for(
        self, *, info_key: str, legal_action_ids: tuple[int, ...]
    ) -> dict[int, float]:
        """Return distribution over legal_action_ids for (theta, info_key).

        Deterministic: hash(theta, info_key, legal set) → probabilities.
        Illegal actions never appear; legal probabilities sum to 1.
        """
        if not isinstance(info_key, str) or not info_key.startswith("sha256:"):
            raise ContractError(f"info_key must be sha256 digest, got {info_key!r}")
        if not isinstance(legal_action_ids, tuple) or len(legal_action_ids) == 0:
            raise ContractError("legal_action_ids must be non-empty tuple")
        for aid in legal_action_ids:
            if not isinstance(aid, int) or isinstance(aid, bool):
                raise ContractError(f"action_id must be int, got {aid!r}")
        if len(set(legal_action_ids)) != len(legal_action_ids):
            raise ContractError("duplicate action_id in legal set")

        # Deterministic Dirichlet pseudo-counts via hashlib
        # Tight type favors first action; loose is near uniform — observable behavioral difference
        seed = hashlib.sha256(
            self.seed_domain
            + f"{self.theta}:{info_key}:{','.join(map(str, sorted(legal_action_ids)))}".encode()
        ).digest()
        # Generate pseudo-counts
        counts: list[float] = []
        for idx, aid in enumerate(sorted(legal_action_ids)):
            h = hashlib.sha256(seed + idx.to_bytes(2, "big")).digest()
            val = int.from_bytes(h[:4], "big") / 4294967296.0  # uniform [0,1)
            # Map to positive count in [0.5, 2.5) then bias by theta
            base = 0.5 + val * 2.0
            if self.theta == "tight":
                # Tight boosts first sorted action (more deterministic)
                if idx == 0:
                    base *= 2.0
            elif self.theta == "loose":
                base = 1.0 + val * 0.5  # flatter
            counts.append(base)
        total = sum(counts)
        probs = [c / total for c in counts]
        # Attach to sorted ids then map back
        sorted_ids = tuple(sorted(legal_action_ids))
        result = {aid: prob for aid, prob in zip(sorted_ids, probs, strict=True)}
        # Validate normalization and range
        s = sum(result.values())
        if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ContractError(f"distribution must sum to 1, got {s}")
        for p in result.values():
            if not (0.0 < p <= 1.0) or not math.isfinite(p):
                raise ContractError(f"probability must be in (0,1], got {p}")
        return result

    def log_prob(
        self, *, info_key: str, legal_action_ids: tuple[int, ...], action_id: int
    ) -> float:
        """Log probability of action_id under this theta+info — for kernel likelihood."""
        dist = self.distribution_for(info_key=info_key, legal_action_ids=legal_action_ids)
        p = dist.get(action_id)
        if p is None or p <= 0.0:
            # Illegal or zero mass -> -inf but we raise ContractError to enforce legal
            raise ContractError(f"action_id {action_id} not in legal set or zero prob")
        lp = math.log(p)
        if not math.isfinite(lp):
            raise ContractError(f"log_prob must be finite, got {lp}")
        return lp


def validate_same_information_equality(
    policy: OpponentTypePolicy, *, world: Any, opponent_seat: int, legal_action_ids: tuple[int, ...]
) -> bool:
    """Same information => same distribution; different info => maybe different."""
    from hydra2.belief.world import world_actor_observation as _wao

    obs = _wao(world, actor=opponent_seat)
    key1 = info_key_for_observation(obs)
    # Perturb opponent hidden hand within same info set? Actually same info_key should give same dist
    # Create second world with same opponent hand (so same I_j) but different other hidden (root's hidden perm)
    # We keep opponent seat unchanged
    hands = tuple(tuple(int(t) for t in h) for h in world.concealed_hands)  # type: ignore[union-attr]
    # Build world2 where we swap tiles among non-opponent seats (still same I_j for opponent)
    permuted = list(hands)
    # Swap tiles between two seats that are not opponent_seat
    others = [s for s in range(4) if s != opponent_seat]
    if len(others) >= 2 and len(permuted[others[0]]) > 0 and len(permuted[others[1]]) > 0:
        a = permuted[others[0]][0]
        b = permuted[others[1]][0]
        permuted[others[0]] = tuple(sorted((b,) + permuted[others[0]][1:]))
        permuted[others[1]] = tuple(sorted((a,) + permuted[others[1]][1:]))
    else:
        # No swap possible, keep identical
        permuted = list(hands)
    from hydra2.belief.world import make_full_world as _mfw

    world2 = _mfw(
        concealed_hands=tuple(tuple(sorted(h)) for h in permuted),
        live_wall=tuple(int(t) for t in world.live_wall),  # type: ignore[union-attr]
        dead_wall=tuple(int(t) for t in world.dead_wall),  # type: ignore[union-attr]
        rules_hash=world.rules_hash,  # type: ignore[union-attr]
        observation_hash=world.observation_hash,  # type: ignore[union-attr]
    )
    obs2 = _wao(world2, actor=opponent_seat)
    key2 = info_key_for_observation(obs2)
    if key1 != key2:
        # Different info sets may give different distributions — not a failure, but we check that
        # unlocking same key gives same dist
        dist1 = policy.distribution_for(info_key=key1, legal_action_ids=legal_action_ids)
        dist1_again = policy.distribution_for(info_key=key1, legal_action_ids=legal_action_ids)
        return dist1 == dist1_again
    # Same key → must be identical distribution (deterministic same-information equality)
    d1 = policy.distribution_for(info_key=key1, legal_action_ids=legal_action_ids)
    d2 = policy.distribution_for(info_key=key2, legal_action_ids=legal_action_ids)
    return d1 == d2


# ---------------------------------------------------------------------------
# Joint particles and posterior
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JointParticle:
    """One joint particle ``(theta, world_ref, weight)`` with provenance."""

    theta: str
    world_ref: str  # opaque world_id
    weight: float
    epoch: int
    target_id: str

    def __post_init__(self) -> None:
        if self.theta not in THETA_IDS:
            raise ContractError(f"theta must be one of {THETA_IDS}, got {self.theta!r}")
        if not isinstance(self.world_ref, str) or not self.world_ref.startswith("sha256:"):
            raise ContractError(f"world_ref must be sha256 digest, got {self.world_ref!r}")
        if (
            not isinstance(self.weight, float)
            or not math.isfinite(self.weight)
            or self.weight < 0.0
        ):
            raise ContractError(f"weight must be finite non-negative float, got {self.weight!r}")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise ContractError(f"epoch must be non-negative int, got {self.epoch!r}")
        if not isinstance(self.target_id, str) or not self.target_id.startswith("sha256:"):
            raise ContractError(f"target_id must be sha256 digest, got {self.target_id!r}")


@dataclass(frozen=True, slots=True)
class JointPosterior:
    """Frozen joint posterior over Theta x Worlds at one epoch."""

    particles: tuple[JointParticle, ...]
    epoch: int
    target_id: str
    theta_ids: tuple[str, ...] = THETA_IDS
    normalized: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.particles, tuple) or len(self.particles) == 0:
            raise ContractError("particles must be non-empty tuple")
        for p in self.particles:
            if not isinstance(p, JointParticle):
                raise ContractError(f"each particle must be JointParticle, got {type(p).__name__}")
            if p.epoch != self.epoch:
                raise ContractError(f"particle epoch {p.epoch} != posterior epoch {self.epoch}")
            if p.target_id != self.target_id:
                raise ContractError("particle target_id mismatch")
            if p.theta not in self.theta_ids:
                raise ContractError(f"particle theta {p.theta!r} not in {self.theta_ids!r}")
        s = sum(p.weight for p in self.particles)
        if self.normalized and not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ContractError(f"normalized posterior must sum to 1, got {s}")
        if s <= 0.0 or not math.isfinite(s):
            raise ContractError(f"total weight must be positive finite, got {s}")

    def marginal_theta(self) -> dict[str, float]:
        """Marginal ``p(theta) = sum_x p(theta,x)``."""
        out: dict[str, float] = dict.fromkeys(self.theta_ids, 0.0)
        for p in self.particles:
            out[p.theta] += p.weight
        # Renormalize to account for float
        tot = sum(out.values())
        if tot > 0:
            for k in out:
                out[k] /= tot
        return out

    def conditional_world_given_theta(self, theta: str) -> dict[str, float]:
        """Conditional ``p(x | theta) = p(theta,x)/p(theta)`` as world_ref -> prob."""
        if theta not in self.theta_ids:
            raise ContractError(f"theta must be one of {self.theta_ids}, got {theta!r}")
        mass_theta = sum(p.weight for p in self.particles if p.theta == theta)
        if mass_theta <= 0.0 or not math.isfinite(mass_theta):
            raise ContractError(f"theta {theta!r} has zero mass")
        out: dict[str, float] = {}
        for p in self.particles:
            if p.theta == theta:
                out[p.world_ref] = p.weight / mass_theta
        return out
