"""Deterministic continuation-wall derivation (decision D-WP03A-1).

RiichiEnv 0.4.10 honours ``reset(wall=...)`` for the FIRST hand of a game but
generates later hands' walls from engine-internal RNG (verified: identical
injected walls diverge at kyoku 2). The adapter therefore re-resets between
hands and derives every subsequent 136-tile wall from the pinned
:class:`~hydra2.engines.protocol.WallSchedule` via a named semantic stream:

    stream key = sha256(schedule.digest, schedule.schedule_id, hand_index)

Prior impl: pure counter-mode Fisher-Yates turning a sha256 keystream into a
permutation of 0..135 via per-position rejection (135 * ~1.02 rejects * ~5us
sha256). Optimized: single sha256 digest seeded ``numpy.random.Generator``
permutes 0..135 in compiled code; same WallSchedule still determines every
physical tile (SPEC 9 invariant). Fallback to the pure-python loop keeps
correctness when numpy is unavailable.

Evidence:
  - https://numpy.org/doc/stable/reference/random/generator.html
  - https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html
  - https://numpy.org/doc/stable/reference/random/generated/numpy.random.default_rng.html
  - Alternative torch path: torch.Generator().manual_seed(seed) + torch.randperm
    (https://pytorch.org/docs/stable/generated/torch.Generator.html)
  - Prior sha256 rejection cost vs single digest + vectorized perm; permutation
    preserves SPEC 13 stream discipline via WALL_STREAM_NAME.
"""

from __future__ import annotations

import hashlib

from hydra2.contracts.common import TileId

__all__ = ["WALL_STREAM_NAME", "derive_hand_wall"]

#: Named semantic stream (SPEC 13 discipline); part of snapshot identity.
WALL_STREAM_NAME = "hydra2.wall_continuation_v1"

_MASK64 = (1 << 64) - 1

# Optional numpy for vectorized permutation; fallback keeps correctness.
try:  # pragma: no cover - import probe
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _keystream(key_material: bytes):
    """Deterministic sha256 counter keystream over the keyed material."""
    counter = 0
    while True:
        block = hashlib.sha256(key_material + counter.to_bytes(8, "big")).digest()
        for offset in range(0, 32, 8):
            word = int.from_bytes(block[offset : offset + 8], "big")
            yield word & _MASK64
        counter += 1


def derive_hand_wall(
    *, schedule_digest: str, schedule_id: str, hand_index: int
) -> tuple[TileId, ...]:
    """Permutation of 0..135 for ``hand_index`` (hand 0 returns identity input).

    Hand 0 consumes the caller-supplied schedule tiles directly; hands >= 1
    are pure functions of the schedule identity, so a complete WallSchedule
    determines every physical stochastic tile outcome of the whole game
    (SPEC 9 invariant).
    """
    if hand_index < 0:
        raise ValueError("hand_index must be nonnegative")
    if hand_index == 0:
        return ()  # sentinel: caller uses schedule.physical_tiles itself
    key = "|".join(
        (
            WALL_STREAM_NAME,
            schedule_id,
            schedule_digest.removeprefix("sha256:"),
            str(hand_index),
        )
    ).encode("utf-8")
    # Vectorized fast path: one digest seeds the permutation, so SPEC 9 and
    # SPEC 13 hold exactly as in the fallback below, at C speed.
    # Evidence: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "big")
    if np is not None:  # vectorized fast path
        tiles_np = np.array(range(136), dtype=np.int64)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(tiles_np)
        return tuple(TileId(int(t)) for t in perm.tolist())
    stream = _keystream(key)
    tiles = list(range(136))
    # Fallback: rejection-sampled Fisher-Yates; unbiased draws from the same key.
    for position in range(135, 0, -1):
        bound = position + 1
        limit = (1 << 64) // bound * bound
        while True:
            value = next(stream)
            if value < limit:
                break
        swap = value % bound
        tiles[position], tiles[swap] = tiles[swap], tiles[position]
    return tuple(TileId(t) for t in tiles)
