# ruff: noqa: E501
"""Natural tiny-corpus helpers (deterministic consistent worlds).

Single home for `_validate_finite` plus `_build_tiny_corpus_for_epoch`.
Bridge tiny-corpus tables decide the hands and wall; Python only filters
by observation hash and registers by world_id. Failure mode is fail-closed
`ContractError` on empty corpus downstream, never silent worlds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError as ContractError

if TYPE_CHECKING:
    from hydra2.belief.natural import BeliefEpoch as BeliefEpoch
    from hydra2.belief.world import FullWorld as FullWorld


def _validate_finite(value: float, *, name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ContractError(f"{name} must be finite float, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Tiny corpus generation — deterministic consistent worlds
# ---------------------------------------------------------------------------


def _build_tiny_corpus_for_epoch(
    epoch: BeliefEpoch,
    *,
    registry: dict[str, FullWorld],
) -> list[FullWorld]:
    """Return worlds consistent with epoch.observation_hash from registry.

    Registry is the belief's world store. If epoch has no stored corpus,
    lazily generate a deterministic tiny corpus of 4 worlds sharing the same
    root observation. This is used for uniform natural law.
    """
    # Filter registry by observation_hash and rules_hash
    consistent = [
        w
        for w in registry.values()
        if w.observation_hash == epoch.observation_hash and w.rules_hash == epoch.rules_hash
    ]
    if len(consistent) > 0:
        # Deterministic order by world_id
        consistent.sort(key=lambda w: w.world_id)
        return consistent
    # No stored corpus: synthesize 4 worlds consistent by construction with
    # the same root hand, then bind their observation_hash to the epoch so
    # the generated worlds match the epoch observation.
    # Tiny-domain tile pool 0..11: seats hold variants over 0..7, the wall
    # holds 8..11. Root hand stays fixed for hidden-permutation invariance.
    # Hands/wall sourced from the shared contracts tables (single source with
    # belief.corpus): the first NATURAL_EPOCH_CORPUS_SIZE rows of
    # TINY_CORPUS_OPTIONS plus TINY_CORPUS_WALL, both owned by the bridge.
    corpus_size: int = _bridge_contracts.NATURAL_EPOCH_CORPUS_SIZE
    raw_options: tuple[tuple[tuple[int, ...], ...], ...] = _bridge_contracts.TINY_CORPUS_OPTIONS
    base_hands_options: list[tuple[tuple[int, ...], ...]] = list(raw_options[:corpus_size])
    raw_wall: tuple[int, ...] = _bridge_contracts.TINY_CORPUS_WALL
    wall: tuple[int, ...] = raw_wall
    worlds: list[FullWorld] = []
    for idx, hands in enumerate(base_hands_options):
        # Deterministically perturb latent_state with idx to keep world_id unique even if hands repeat
        from hydra2.belief.world import make_full_world

        w = make_full_world(
            concealed_hands=hands,
            live_wall=wall,
            dead_wall=(),
            latent_state={"corpus_idx": idx},
            rules_hash=epoch.rules_hash,
            observation_hash=epoch.observation_hash,
            simulator_snapshot=f"tiny:{epoch.target_id}:{idx}",
        )
        worlds.append(w)
    # Register them into provided dict (caller will extend)
    for w in worlds:
        registry[w.world_id] = w
    worlds.sort(key=lambda w: w.world_id)
    return worlds
