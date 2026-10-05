"""WP-07A tiny finite world corpus with exact probabilities (oracle).

Provides a deterministic, fully enumerated set of worlds consistent with an
actor observation, with uniform exact distribution suitable for particle vs oracle
comparison. Tile conservation and red-aware handling are simplified for the tiny
domain but preserve the required invariants for testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import sha256_digest
from hydra2.belief.world import FullWorld, make_full_world

if TYPE_CHECKING:
    from hydra2.contracts.common import DigestText
    from hydra2.contracts.observation_actor import ActorObservation

__all__ = [
    "TinyCorpus",
    "build_tiny_corpus",
    "enumerate_worlds",
    "exact_log_prob",
]

#: Frozen tiny-domain tables (single source: ``hydra2._native.contracts``;
#: this module keeps the builder + record roots, ``__all__`` unchanged).
_BASE_OPTIONS: tuple[tuple[tuple[int, ...], ...], ...] = _bridge_contracts.TINY_CORPUS_OPTIONS
_TINY_WALL: tuple[int, ...] = _bridge_contracts.TINY_CORPUS_WALL
_DEFAULT_ROOT_HAND: tuple[int, ...] = _bridge_contracts.TINY_CORPUS_DEFAULT_ROOT_HAND
_DEFAULT_SIZE: int = _bridge_contracts.TINY_CORPUS_DEFAULT_SIZE


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
        for prob in self.probabilities:
            if not math.isfinite(prob) or prob < 0:
                raise ValueError(f"probability {prob} must be finite nonnegative")
        # Invariant: every world binds the corpus hashes; mismatch fails closed.
        for world in self.worlds:
            if world.observation_hash != self.observation_hash:
                raise ValueError("world observation_hash mismatch corpus")
            if world.rules_hash != self.rules_hash:
                raise ValueError("world rules_hash mismatch corpus")

    def log_prob(self, world_id: str) -> float:
        log_prob_value: float = _bridge_contracts.corpus_log_prob_for(
            tuple(world.world_id for world in self.worlds), self.probabilities, world_id
        )
        return log_prob_value


def build_tiny_corpus(
    *,
    observation: ActorObservation | None = None,
    observation_hash: DigestText | None = None,
    rules_hash: DigestText | None = None,
    root_hand: tuple[int, ...] = _DEFAULT_ROOT_HAND,
    size: int = _DEFAULT_SIZE,
) -> TinyCorpus:
    """Build deterministic tiny corpus of ``size`` worlds.

    If observation is provided, its hashes are used; otherwise hashes must be supplied.
    Worlds share the same root observation (hidden permutation invariance) but differ
    in opponent assignments, providing the oracle exact distribution (uniform).
    """
    if observation is not None:
        obs_hash: DigestText = _bridge_contracts.make_digest_text(observation.observation_hash)
        r_hash: DigestText = _bridge_contracts.make_digest_text(observation.rules_hash)
    else:
        if observation_hash is None or rules_hash is None:
            raise ValueError("must supply observation or hashes")
        obs_hash = _bridge_contracts.make_digest_text(observation_hash)
        r_hash = _bridge_contracts.make_digest_text(rules_hash)
    # Invariant: tiny tile pool 0..11 matches the natural harness exactly.
    hand_options: tuple[tuple[tuple[int, ...], ...], ...] = _BASE_OPTIONS[:size]
    wall = _TINY_WALL
    worlds: list[FullWorld] = []
    for idx, hands in enumerate(hand_options):
        # Keep root hand consistent with the observation that binds the corpus.
        # If observation supplied, its concealed_hand should equal root_hand; we enforce.
        if observation is not None:
            # Prefer the observation hand: corpus must share the actor view.
            exp_hand = tuple(int(t) for t in observation.concealed_hand)
            if tuple(hands[0]) != exp_hand:
                # Replace root hand to match observation while preserving other seats
                hands = (exp_hand, hands[1], hands[2], hands[3])
        else:
            # No observation binds the corpus, so fall back to supplied root_hand.
            if tuple(hands[0]) != tuple(root_hand):
                hands = (tuple(root_hand), hands[1], hands[2], hands[3])
        world = make_full_world(
            concealed_hands=hands,
            live_wall=wall,
            dead_wall=(),
            latent_state={
                "corpus_idx": idx,
                # Invariant: tag via canon bridge; rejects raise, never silent.
                # Corpus order (sorted world_id) untouched.
                "tag": str(sha256_digest(f"{obs_hash}:{idx}".encode())).removeprefix("sha256:")[:8],
            },
            rules_hash=r_hash,
            observation_hash=obs_hash,
            simulator_snapshot=f"tiny_corpus:{obs_hash[:8]}:{idx}",
        )
        worlds.append(world)
    worlds_sorted = tuple(sorted(worlds, key=lambda world: world.world_id))
    # Uniform exact probabilities over the sorted corpus.
    num_worlds = len(worlds_sorted)
    probs = tuple(1.0 / num_worlds for _ in range(num_worlds))
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
