# ruff: noqa: N806
"""WP-07A tiny finite world corpus with exact probabilities (oracle).

Provides a deterministic, fully enumerated set of worlds consistent with an
actor observation, with uniform exact distribution suitable for particle vs oracle
comparison. Tile conservation and red-aware handling are simplified for the tiny
domain but preserve the required invariants for testing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2.belief.world import FullWorld, make_full_world
from hydra2.contracts.common import DigestText, make_digest_text

if TYPE_CHECKING:
    from hydra2.contracts.observation import ActorObservation

__all__ = [
    "TinyCorpus",
    "build_tiny_corpus",
    "enumerate_worlds",
    "exact_log_prob",
]


@dataclass(frozen=True, slots=True)
class TinyCorpus:
    observation_hash: DigestText
    rules_hash: DigestText
    worlds: tuple[FullWorld, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.worlds) != len(self.probabilities):
            raise ValueError("worlds and probabilities length mismatch")
        total = sum(self.probabilities)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        for p in self.probabilities:
            if not math.isfinite(p) or p < 0:
                raise ValueError(f"probability {p} must be finite nonnegative")
        # Ensure worlds observation_hash matches corpus observation_hash
        for w in self.worlds:
            if w.observation_hash != self.observation_hash:
                raise ValueError("world observation_hash mismatch corpus")
            if w.rules_hash != self.rules_hash:
                raise ValueError("world rules_hash mismatch corpus")

    def log_prob(self, world_id: str) -> float:
        for w, p in zip(self.worlds, self.probabilities, strict=False):
            if w.world_id == world_id:
                return math.log(p) if p > 0 else float("-inf")
        return float("-inf")

    def prob(self, world_id: str) -> float:
        for w, p in zip(self.worlds, self.probabilities, strict=False):
            if w.world_id == world_id:
                return p
        return 0.0


def build_tiny_corpus(
    *,
    observation: ActorObservation | None = None,
    observation_hash: DigestText | None = None,
    rules_hash: DigestText | None = None,
    root_hand: tuple[int, ...] = (0, 1),
    size: int = 4,
) -> TinyCorpus:
    """Build deterministic tiny corpus of ``size`` worlds.

    If observation is provided, its hashes are used; otherwise hashes must be supplied.
    Worlds share the same root observation (hidden permutation invariance) but differ
    in opponent assignments, providing the oracle exact distribution (uniform).
    """
    if observation is not None:
        obs_hash = make_digest_text(observation.observation_hash)  # type: ignore[arg-type]
        r_hash = make_digest_text(observation.rules_hash)
    else:
        if observation_hash is None or rules_hash is None:
            raise ValueError("must supply observation or hashes")
        obs_hash = make_digest_text(observation_hash)
        r_hash = make_digest_text(rules_hash)

    # Fixed tile pool for tiny domain — 0..11 as in natural harness
    base_options = [
        ((0, 1), (2, 3), (4, 5), (6, 7)),
        ((0, 1), (2, 4), (3, 5), (6, 7)),
        ((0, 1), (2, 5), (3, 4), (6, 7)),
        ((0, 1), (2, 6), (3, 4), (5, 7)),
        ((0, 1), (2, 7), (3, 4), (5, 6)),
        ((0, 1), (3, 6), (2, 4), (5, 7)),
    ]
    opts = base_options[:size]
    wall = (8, 9, 10, 11)
    worlds: list[FullWorld] = []
    for idx, hands in enumerate(opts):
        # Override root hand if supplied differs from (0,1) — keep root consistent with observation
        # If observation supplied, its concealed_hand should equal root_hand; we enforce.
        if observation is not None:
            exp_hand = tuple(int(t) for t in observation.concealed_hand)
            if tuple(hands[0]) != exp_hand:
                # Replace root hand to match observation while preserving other seats
                hands = (exp_hand, hands[1], hands[2], hands[3])
        else:
            # Use supplied root_hand
            if tuple(hands[0]) != tuple(root_hand):
                hands = (tuple(root_hand), hands[1], hands[2], hands[3])
        w = make_full_world(
            concealed_hands=hands,
            live_wall=wall,
            dead_wall=(),
            latent_state={
                "corpus_idx": idx,
                "tag": hashlib.sha256(f"{obs_hash}:{idx}".encode()).hexdigest()[:8],
            },
            rules_hash=r_hash,
            observation_hash=obs_hash,
            simulator_snapshot=f"tiny_corpus:{obs_hash[:8]}:{idx}",
        )
        worlds.append(w)
    worlds_sorted = tuple(sorted(worlds, key=lambda w: w.world_id))
    # Uniform exact probabilities
    K = len(worlds_sorted)
    probs = tuple(1.0 / K for _ in range(K))
    return TinyCorpus(
        observation_hash=obs_hash,
        rules_hash=r_hash,
        worlds=worlds_sorted,
        probabilities=probs,
    )


def enumerate_worlds(corpus: TinyCorpus) -> tuple[FullWorld, ...]:
    return corpus.worlds


def exact_log_prob(corpus: TinyCorpus, world_id: str) -> float:
    return corpus.log_prob(world_id)
