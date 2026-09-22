"""Gumbel precompute helpers — frozen policy plus Rust-batch envelope (deterministic root Gumbels from (case_id, root_seat, candidate_id, action_id); sequential-halving rounds and visit allocations spec-supplied; model-call/transition matched comparator; every transition exact).

Single home for `UniformContinuationPolicy` plus `_require_driver_bridge`,
`_halving_slots`, `_dry_float_need`, and `_exact_sum_for_mean` shared by the
Gumbel search driver. Python builds ONE batch per search; halving rounds,
rollout descent, backup, cuts, and selection live in
`hydra-search:gumbel_halving_batch` with GIL released. Failure mode is
fail-closed `ContractError` on bad policy or RNG, `ImportError` with
`pixi run build-ext` hint on stale bridge, never silent fallback.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from hydra2.contracts.common import (
    ContractError as ContractError,
)
from hydra2.contracts.common import (
    VisibilityViolationError as VisibilityViolationError,
)
from hydra2.search.gumbel_core import _require_random_stream as _require_random_stream
from hydra2.search.gumbel_core import _require_search_bridge as _require_search_bridge


class UniformContinuationPolicy:
    """Frozen continuation for non-root seats — actor observation only."""

    def __init__(self, *, bias_strength: float = 0.2) -> None:
        if not isinstance(bias_strength, float) or not 0 <= bias_strength < 0.5:
            raise ContractError("bias_strength must be float in [0,0.5)")
        self._bias = bias_strength

    def _distribution_for(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        if len(legal) == 0:
            raise ContractError("legal must be non-empty")
        if len(legal) == 1:
            return (1.0,)
        try:
            _h2: Any | None = getattr(observation, "observation_hash", None)
            h: str = _h2 if isinstance(_h2, str) and _h2 != "" else ""
            digest = hashlib.sha256(h.encode()).digest()
            direction = digest[0] & 1
        except Exception:
            direction = 0
        n = len(legal)
        if n == 2:
            p0 = 0.5 + self._bias if direction == 0 else 0.5 - self._bias
            return (p0, 1.0 - p0)
        w = 1.0 / n
        return tuple(w for _ in legal)

    def distribution(self, observation: Any, legal: tuple[int, ...]) -> tuple[float, ...]:
        if observation is not None:
            try:
                from hydra2.contracts.observation_actor import ActorObservation as _Obs
            except ImportError as exc:
                raise ImportError(
                    "hydra2.contracts.observation not importable "
                    f"({exc}); build the bridge with `pixi run build-ext` before Gumbel search"
                ) from exc
            if not isinstance(observation, _Obs):
                raise ContractError(
                    f"policy input must be ActorObservation, got {type(observation).__name__}"
                )
            if hasattr(observation, "world_id"):
                raise VisibilityViolationError(
                    "policy input contains world_id [PBRF_VIS_POLICY_WORLD]"
                )
            if hasattr(observation, "concealed_hands"):
                raise VisibilityViolationError(
                    "policy input contains concealed_hands (FullWorld) [PBRF_VIS_POLICY_HANDS]"
                )
        return self._distribution_for(observation, legal)

    def sample(self, observation: Any, legal: tuple[int, ...], rng: Any) -> int:
        if not isinstance(legal, tuple) or len(legal) == 0:
            raise ContractError("legal must be non-empty tuple")
        dist = self.distribution(observation, legal)
        _require_random_stream()
        if not hasattr(rng, "random_float"):
            raise ContractError("gumbel: rng must expose random_float; hash%1000 fallback removed")
        r: float = float(rng.random_float())  # type: ignore[explicit-any]
        cum = 0.0
        for idx, p in enumerate(dist):
            cum += p
            if r < cum:
                return legal[idx]
        return legal[-1]


def _require_driver_bridge() -> Any:
    """Import the built ``search`` bridge surface with ``gumbel_halving`` (fail closed)."""
    try:
        mod = _require_search_bridge()
    except ImportError:
        raise
    if not hasattr(mod, "gumbel_halving"):
        raise ImportError(
            "hydra2._native.search.gumbel_halving missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return mod


def _halving_slots(num_candidates: int, halving_rounds: int) -> list[int]:
    """Deterministic slot counts per round (mirrors Rust: M, ceil(M/2), ...)."""
    slots: list[int] = []
    width = num_candidates
    for _ in range(halving_rounds):
        slots.append(width)
        width = (width + 1) // 2
    return slots


def _dry_float_need(
    worlds_live_lens: list[int],
    slots: list[int],
    visits_per_round: tuple[int, ...],
    max_depth: int,
    max_transitions: int | None,
) -> int:
    """Replicate the Rust dry-run float demand (``gumbel_driver.rs:589-639``)."""
    need = 0
    walked = 0
    ord_ = 0
    dr = 0
    dry_done = False
    rounds = len(slots)
    while dr < rounds and not dry_done:
        if slots[dr] <= 1:
            break
        visits = visits_per_round[dr]
        ds = 0
        while ds < slots[dr] and not dry_done:
            dv = 0
            while dv < visits:
                if max_transitions is not None and walked >= max_transitions:
                    dry_done = True
                    break
                ord_ += 1
                walked += 1
                step = 1
                live = worlds_live_lens[ord_ - 1]
                while step < max_depth and live > 0:
                    need += 1
                    if max_transitions is not None and walked >= max_transitions:
                        break
                    live -= 1
                    step += 1
                    walked += 1
                    if max_transitions is not None and walked >= max_transitions:
                        break
                dv += 1
            ds += 1
        dr += 1
    return need


def _exact_sum_for_mean(mean: float, visits: int) -> float:
    """Sum ``s`` with ``s / visits == mean`` exactly (IEEE-754 bit-identical)."""
    cand = mean * visits
    if cand / visits == mean:
        return cand
    lo = math.nextafter(cand, math.inf)
    hi = math.nextafter(cand, -math.inf)
    for _ in range(64):
        if lo / visits == mean:
            return lo
        if hi / visits == mean:
            return hi
        lo = math.nextafter(lo, math.inf)
        hi = math.nextafter(hi, -math.inf)
    return lo
