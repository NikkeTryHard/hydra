"""SPEC 18.3 uncertainty machinery — blocks, bootstrap, sign-flip, CS.

The independent unit of confirmation is the COMPLETE WALL BLOCK; games inside
a wall share the wall deck and are therefore never independent units
(SPEC 18.1/18.3). Everything here resamples whole blocks:

* :func:`bootstrap_blocks` — percentile bootstrap over block contrasts;
* :func:`sign_flip_interval` — random Rademacher sign flips of block
  contrasts (symmetric-null interval);
* :func:`fixed_n_samples` — the SPEC fixed sample size formula
  ``N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)**2)``;
* :func:`hedged_confidence_sequence` / :func:`hedged_cs_path` — a concrete,
  named time-uniform confidence sequence: the hedged-betting capital process
  of Waudby-Smith & Ramdas (2023) with predictable lambdas and a grid union
  bound over means, valid simultaneously at every peek time;
* :func:`sequential_design_guard` — adaptive peeking without a declared
  sequential design invalidates confirmation (SPEC 18.3);
* :func:`cluster_bootstrap` — clustered diagnostics resample game/player
  groups, never decisions (grouping is a two-value literal by construction).

All randomness enters through a semantic :class:`RandomStream`, so every
interval is reproducible from its key.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, Literal

import numpy as np

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream
from hydra2.eval.blocks import (
    BlockTolerance,
    ExcludedBlock,
    WallBlock,
    aggregate_blocks,
    aggregate_wall_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hydra2.eval.telemetry import ResourceTelemetry

__all__ = [
    "ClusterGrouping",
    "SelectionConfig",
    "bootstrap_blocks",
    "ci_covers",
    "cluster_bootstrap",
    "fixed_n_samples",
    "hedged_confidence_sequence",
    "hedged_cs_path",
    "placement_block_contrast",
    "score_selection",
    "selection_gate_check",
    "sequential_design_guard",
    "sign_flip_interval",
]

ClusterGrouping = Literal["game", "player"]

_STD = NormalDist()


def _validate_blocks(block_values: Sequence[float]) -> list[float]:
    values = list(block_values)
    if len(values) < 2:
        raise ContractError("need at least two blocks for any interval")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContractError(f"block contrast must be finite, got {value!r}")
    return list(values)


def _validate_alpha_resamples(alpha: float, resamples: int) -> None:
    if not 0.0 < alpha < 0.5:
        raise ContractError("alpha must lie in (0, 0.5)")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ContractError("resamples must be an int >= 100")


def placement_block_contrast(blocks: tuple[WallBlock, ...]) -> tuple[float, ...]:
    """Collapse wall blocks to placement-contrast means, wall order stable.

    Each :class:`WallBlock` must already carry expected-final-placement
    contrasts (``case.PRIMARY_METRIC``); this helper only collapses via
    :func:`aggregate_wall_block` sorted by wall_id. Feed the result — never
    per-game contrasts — to :func:`bootstrap_blocks` / sign-flip / CS.

    Placement contrast is the mean expected-final-placement difference for
    the declared arm pair (lower-better S1); wall order stable.
    """
    if len(blocks) == 0:
        raise ContractError("need at least one wall block for a contrast")
    ordered = sorted(blocks, key=lambda block: block.wall_id)
    contrasts = tuple(aggregate_wall_block(block) for block in ordered)
    for value in contrasts:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"block contrast must be finite, got {value!r}")
        if not math.isfinite(float(value)):
            raise ContractError(f"block contrast must be finite, got {value!r}")
    return contrasts


def fixed_n_samples(*, s: float, delta: float, alpha: float, beta: float) -> int:
    """SPEC 18.3: ``N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)^2)``."""
    for name, value in (("s", s), ("delta", delta)):
        if not math.isfinite(value) or value <= 0:
            raise ContractError(f"{name} must be positive and finite")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ContractError("alpha and beta must lie in (0, 1)")
    z_a = _STD.inv_cdf(1.0 - alpha)
    z_b = _STD.inv_cdf(1.0 - beta)
    return math.ceil(((z_a + z_b) * s / delta) ** 2)


def _bootstrap_percentile_indexes(alpha: float, resamples: int) -> tuple[int, int]:
    """Percentile indexes verbatim (Rust ``eval::statistics::percentile_indexes``).

    ``low = floor(alpha / 2 * R)``, ``high = ceil((1 - alpha / 2) * R) - 1``
    (T3 pins ``R=2000, alpha=0.05`` → ``(50, 1949)``). Draw-free assembly is
    Rust-owned; this mirror keeps the numpy-draw lane (M2) byte-identical
    until a bridge surface lands with a bit-parity KAT.
    """
    _validate_alpha_resamples(alpha, resamples)
    low = math.floor(alpha / 2 * resamples)
    high = math.ceil((1 - alpha / 2) * resamples) - 1
    return low, high


def bootstrap_blocks(
    block_values: Sequence[float],
    *,
    stream: RandomStream,
    resamples: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Whole-block percentile bootstrap; returns (estimate, low, high).

    Blocks are resampled WITH replacement as atomic units — individual games
    within a block never separate.

    Vectorized over a seeded numpy Generator derived from RandomStream
    (counter-based, reproducible); resamples whole blocks with replacement
    as atomic units.

    Rust-first (port-B, M2 measure-first): numpy PCG64 keeps the draws —
    the ONLY seed source is the legacy CTR stream
    (``seed = u64BE(stream.get_bytes(8))`` into ``np.default_rng``); the
    draw-free assembly (block-mean estimate + percentile indexes) mirrors
    Rust ``eval::statistics::bootstrap_interval``/``percentile_indexes``
    verbatim via :func:`_bootstrap_percentile_indexes`. No PCG64
    reimplementation lives anywhere (port-A only behind measurement +
    bit-parity KAT, never silently). B1/B2: no Gumbel or split draws here —
    whole-block resampling only; held-out splits stay the torch.randperm
    oracle behind KAT.
    """
    values = _validate_blocks(block_values)
    _validate_alpha_resamples(alpha, resamples)
    count = len(values)
    # Seed derives from the stream (consumes 8 bytes); the numpy PCG64
    # Generator then draws resample indices reproducibly.
    seed = int.from_bytes(stream.get_bytes(8), "big")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    # (resamples, count) indices with replacement — equivalent to
    # [stream.random_below(count) for _ in range(count)] per resample
    indices = rng.integers(0, count, size=(resamples, count), dtype=np.intp)
    sample_means = np.take(arr, indices).mean(axis=1)
    sample_means.sort()
    low_index, high_index = _bootstrap_percentile_indexes(alpha, resamples)
    low = float(sample_means[low_index])
    high = float(sample_means[high_index])
    return float(arr.mean()), low, high


def sign_flip_interval(
    block_values: Sequence[float],
    *,
    stream: RandomStream,
    resamples: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Sign-flip (Rademacher) resampling of block contrasts.

    Under the symmetric null each block's contrast is equally likely +/-;
    flipping whole-block signs preserves the paired-block structure while
    generating the null distribution of the mean.

    Vectorized over a seeded numpy Generator derived from RandomStream
    (counter-based, reproducible).
    """
    values = _validate_blocks(block_values)
    _validate_alpha_resamples(alpha, resamples)
    count = len(values)
    seed = int.from_bytes(stream.get_bytes(8), "big")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    # Rademacher: random_below(2)!=0 -> -1 else +1. Use integers 0/1 then map.
    rands = rng.integers(0, 2, size=(resamples, count), dtype=np.int8)
    signs = np.where(rands == 0, 1.0, -1.0)
    # Broadcast arr (count,) across rows (resamples, count)
    stats = (signs * arr).mean(axis=1)
    stats.sort()
    observed = float(arr.mean())
    # Center the null distribution on the observed mean (symmetric-null shift).
    center = float(stats.mean())
    low = observed + (float(stats[math.floor(alpha / 2 * resamples)]) - center)
    high = observed + (float(stats[math.ceil((1 - alpha / 2) * resamples) - 1]) - center)
    return observed, low, high


def ci_covers(bounds: tuple[float, float], truth: float) -> bool:
    """Gate helper: does the interval cover ``truth``?"""
    return bounds[0] <= truth <= bounds[1]


def _hedged_capital_rejected(scaled: Sequence[float], theta: float, threshold: float) -> bool:
    """One hedged capital path; True once either side ever crosses.

    Two-sided hedged capital (Waudby-Smith & Ramdas 2023): per step the
    process multiplies BOTH one-sided bets ``1 +/- lam_t * (x_t - theta)``
    with predictable lambdas using only information up to ``t - 1``, so each
    side is a nonnegative supermartingale under ANY constant mean theta;
    Ville's inequality gives time-uniform level alpha_j per theta, and a
    theta dies when EITHER side's wealth reaches ``threshold``.
    """
    lam_max = min(1.0 / (2.0 * theta), 1.0 / (2.0 * (1.0 - theta)))
    wealth_up = 1.0
    wealth_down = 1.0
    running_sum = 0.0
    for index, observation in enumerate(scaled, start=1):
        mu_prev = running_sum / (index - 1) if index > 1 else 0.5
        variance_term = max(mu_prev * (1.0 - mu_prev), 1e-6)
        # Predictable lambdas use only t-1 info; removing the variance cap
        # or lookback breaks time-uniform validity (Ville).
        lam = min(
            lam_max,
            math.sqrt(
                8.0 * math.log(2.0 * threshold) / (variance_term * index * math.log(index + 1))
            ),
        )
        deviation = observation - theta
        wealth_up *= max(0.0, 1.0 + lam * deviation)
        wealth_down *= max(0.0, 1.0 - lam * deviation)
        if wealth_up >= threshold or wealth_down >= threshold:
            return True
        running_sum += observation
    return False


def _scale_to_unit_interval(values: Sequence[float], bounds: tuple[float, float]) -> list[float]:
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ContractError("bounds must be finite with high > low")
    width = high - low
    scaled = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContractError(f"value must be finite, got {value!r}")
        position = (float(value) - low) / width
        scaled.append(min(1.0, max(0.0, position)))
    return scaled


def hedged_cs_path(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    bounds: tuple[float, float] = (0.0, 1.0),
    grid_size: int = 48,
    peek_times: Sequence[int] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Time-uniform hedged-CS intervals at ``peek_times`` (default all t).

    Returns one (low, high) pair per peek time, each valid simultaneously for
    ALL earlier stopping times (any-time valid). The mean grid uses a union
    bound alpha/grid_size; a grid point is dead once its capital ever crossed
    1/(alpha_j); the surviving set maps back through ``bounds``. An empty
    surviving set (evidence against every grid point) yields low > high.
    """
    _validate_alpha_resamples(alpha, 100)
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 4 <= grid_size <= 512:
        raise ContractError("grid_size must be an int in [4, 512]")
    scaled = _scale_to_unit_interval(values, bounds)
    count = len(scaled)
    times = tuple(sorted(set(peek_times))) if peek_times is not None else (count,)
    for moment in times:
        if isinstance(moment, bool) or not isinstance(moment, int) or not 1 <= moment <= count:
            raise ContractError(f"peek times must be ints in [1, {count}]")

    alpha_j = alpha / grid_size
    threshold = 1.0 / alpha_j
    thetas = [0.02 + (0.98 - 0.02) * index / (grid_size - 1) for index in range(grid_size)]
    rejected = [False] * grid_size

    intervals: list[tuple[float, float]] = []
    low_span, high_span = bounds
    width = high_span - low_span
    prefix = scaled
    for moment in times:
        for slot, theta in enumerate(thetas):
            if rejected[slot]:
                continue
            if _hedged_capital_rejected(prefix[:moment], theta, threshold):
                rejected[slot] = True
        survivors = [thetas[slot] for slot in range(grid_size) if not rejected[slot]]
        if len(survivors) != 0:
            low = low_span + min(survivors) * width
            high = low_span + max(survivors) * width
        else:
            low, high = math.inf, -math.inf
        intervals.append((low, high))

    return tuple(intervals)


def hedged_confidence_sequence(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    bounds: tuple[float, float] = (0.0, 1.0),
    grid_size: int = 48,
) -> tuple[float, float]:
    """Current hedged-CS interval (final peek of :func:`hedged_cs_path`)."""
    return hedged_cs_path(values, alpha=alpha, bounds=bounds, grid_size=grid_size)[-1]


def sequential_design_guard(
    *, design: Literal["fixed_n", "time_uniform_cs"], declared_peeks: Sequence[int]
) -> None:
    """Adaptive peeking without a declared sequential design is fatal.

    ``fixed_n`` allows exactly one look at the precommitted N; anything else
    raises. ``time_uniform_cs`` declares its peek schedule up front.
    """
    peeks = list(declared_peeks)
    if design == "fixed_n":
        if len(peeks) != 1:
            raise ContractError(
                "fixed-N design forbids intermediate peeks; declare 'time_uniform_cs' instead"
            )
        return
    if design == "time_uniform_cs":
        if len(peeks) == 0 or peeks != sorted(set(peeks)) or peeks[0] < 1:
            raise ContractError("declared CS peek schedule must be nonempty, strictly increasing")
        return
    raise ContractError(f"unknown design {design!r}")


def cluster_bootstrap(
    records: Sequence[Mapping[str, object]],
    *,
    grouping: ClusterGrouping,
    value_of: Callable[[Mapping[str, object]], float],
    stream: RandomStream,
    resamples: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Clustered diagnostic CI; resamples game/player groups, never decisions.

    The grouping parameter is the two-literal :data:`ClusterGrouping`; there
    is no decision-level option to misuse. Groups are the atomic resampling
    units; the statistic is the unweighted mean over group means.

    Vectorized over group means via ``numpy`` seeded ``Generator``.
    """
    key = f"{grouping}_id"
    groups: dict[str, list[float]] = {}
    for record in records:
        group_id = record.get(key)
        if not isinstance(group_id, str) or group_id == "":
            raise ContractError(f"every record needs a nonempty '{key}'")
        value = value_of(record)
        if not math.isfinite(value):
            raise ContractError("record values must be finite")
        groups.setdefault(group_id, []).append(value)
    if len(groups) < 2:
        raise ContractError("clustering needs at least two groups")
    _validate_alpha_resamples(alpha, resamples)

    ids = sorted(groups)
    means = [fmean(groups[group_id]) for group_id in ids]
    count = len(ids)
    seed = int.from_bytes(stream.get_bytes(8), "big")
    rng = np.random.default_rng(seed)
    arr = np.asarray(means, dtype=np.float64)
    indices = rng.integers(0, count, size=(resamples, count), dtype=np.intp)
    stats = np.take(arr, indices).mean(axis=1)
    stats.sort()
    estimate = float(arr.mean())
    low = float(stats[math.floor(alpha / 2 * resamples)])
    high = float(stats[math.ceil((1 - alpha / 2) * resamples) - 1])
    return estimate, low, high


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Frozen selection parameters; the ONLY home for N/s/delta/alpha/beta.

    Frozen before any confirmation result exists. :meth:`validate` runs on
    construction (``__post_init__``). ``declared_peeks`` holds 1-based
    wall-block counts: exactly one entry ``(N,)`` by convention for
    ``fixed_n``, a strictly increasing schedule for ``time_uniform_cs``.
    Peek discipline is enforced by :func:`selection_gate_check` (gate only;
    no promotion math lives here). Uncertainty is whole-wall-block only.
    """

    N: int = 30
    pilot_s: float = 1.5
    delta: float = 0.5
    alpha: float = 0.05
    beta: float = 0.2
    design: Literal["fixed_n", "time_uniform_cs"] = "fixed_n"
    declared_peeks: tuple[int, ...] = (30,)
    margin: float = 0.0
    uncertainty_unit: Literal["wall_block"] = "wall_block"
    resamples: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Freeze-time checks; raises :class:`ContractError` when violated."""
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ContractError(f"N must be a positive int, got {self.N!r}")
        for name, value in (("pilot_s", self.pilot_s), ("delta", self.delta)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ContractError(f"{name} must be positive and finite, got {value!r}")
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < float(value) < 1.0
            ):
                raise ContractError(f"{name} must lie in (0, 1), got {value!r}")
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(float(self.margin))
        ):
            raise ContractError(f"margin must be finite, got {self.margin!r}")
        if self.uncertainty_unit != "wall_block":
            raise ContractError(
                f"uncertainty_unit must be 'wall_block', got {self.uncertainty_unit!r}"
            )
        if (
            isinstance(self.resamples, bool)
            or not isinstance(self.resamples, int)
            or self.resamples < 1
        ):
            raise ContractError(f"resamples must be a positive int, got {self.resamples!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError(f"seed must be a non-negative int, got {self.seed!r}")
        peeks = self.declared_peeks
        if not isinstance(peeks, tuple):
            raise ContractError("declared_peeks must be a tuple of positive ints")
        for peek in peeks:
            if isinstance(peek, bool) or not isinstance(peek, int) or peek < 1:
                raise ContractError(f"declared peeks must be positive ints, got {peek!r}")
        sequential_design_guard(design=self.design, declared_peeks=list(peeks))


def selection_gate_check(config: SelectionConfig, n_blocks_observed: int) -> None:
    """Enforce frozen peek discipline at ``n_blocks_observed`` wall blocks.

    Passes only at a declared peek: the frozen ``design`` plus the declared
    schedule sliced to ``<= n_blocks_observed`` is re-checked through
    :func:`sequential_design_guard`, and any look outside the declared
    schedule (early, between, or beyond frozen ``N``) raises.
    """
    config.validate()
    if (
        isinstance(n_blocks_observed, bool)
        or not isinstance(n_blocks_observed, int)
        or n_blocks_observed < 1
    ):
        raise ContractError(f"n_blocks_observed must be a positive int, got {n_blocks_observed!r}")
    if n_blocks_observed > config.N:
        raise ContractError(f"extra look at {n_blocks_observed} blocks exceeds frozen N={config.N}")
    if n_blocks_observed not in config.declared_peeks:
        raise ContractError(
            f"extra look at {n_blocks_observed} blocks "
            f"outside declared peeks {list(config.declared_peeks)!r}"
        )
    elapsed = tuple(peek for peek in config.declared_peeks if peek <= n_blocks_observed)
    sequential_design_guard(design=config.design, declared_peeks=list(elapsed))


def score_selection(
    blocks: tuple[WallBlock, ...],
    telemetry_by_game: Mapping[str, ResourceTelemetry],
    config: SelectionConfig,
    peek_index: int,
    *,
    tolerance: BlockTolerance | None = None,
) -> tuple[float, tuple[float, float, float], tuple[ExcludedBlock, ...]]:
    """Manual-gate selection score over VALID wall blocks only (no auto-wire).

    Never called by training: the caller scores a wall-disjoint held-out
    split by hand (e.g. from :func:`split_blocks_held_out`), then promotes
    by hand. Telemetry policy is enforced through :func:`aggregate_blocks`
    with ``tolerance`` (default strict fail-closed); invalid walls are
    excluded and never silently averaged. Valid walls collapse via
    :func:`placement_block_contrast` (wall order stable), peek discipline
    via :func:`selection_gate_check` on ``peek_index``, uncertainty via
    :func:`bootstrap_blocks` with ``config.alpha``/``config.resamples`` and
    a stream derived from ``config.seed``.

    ``peek_index`` is the declared peek being scored: it MUST equal the
    number of VALID blocks (exclusions change the scored count, so scoring
    ``len(blocks)`` while reporting fewer valid raises). Returns
    ``(metric, (est, low, high), excluded_blocks)`` where ``metric`` is the
    mean valid contrast (lower-better S1), ``(est, low, high)`` is the
    whole-block bootstrap interval, and ``excluded_blocks`` is the
    :func:`aggregate_blocks` exclusion report. The manual caller MUST carry
    ``excluded_blocks`` to the promotion record's ``excluded_blocks`` —
    dropping them misrepresents the scored evidence.

    Raises :class:`ContractError` when no wall block is valid, when
    ``peek_index`` differs from the valid count (excluded beyond tolerance
    for this peek), or when the peek violates the frozen guard.
    """
    if not isinstance(blocks, tuple):
        raise ContractError(f"blocks must be a tuple of WallBlock, got {type(blocks).__name__}")
    if not isinstance(config, SelectionConfig):
        raise ContractError(f"config must be a SelectionConfig, got {type(config).__name__}")
    if tolerance is None:
        effective = BlockTolerance()
    elif isinstance(tolerance, BlockTolerance):
        effective = tolerance
    else:
        raise ContractError(f"tolerance must be a BlockTolerance, got {type(tolerance).__name__}")
    if isinstance(telemetry_by_game, dict):
        telemetry_dict = telemetry_by_game
    else:
        try:
            telemetry_dict = dict(telemetry_by_game)
        except Exception as exc:
            raise ContractError(f"telemetry_by_game must map game_id to row: {exc}") from exc
    result = aggregate_blocks(blocks, telemetry_by_game=telemetry_dict, tolerance=effective)
    if len(result.valid) == 0:
        raise ContractError(
            f"no valid wall blocks to score ({len(result.excluded)} excluded beyond tolerance)"
        )
    if isinstance(peek_index, bool) or not isinstance(peek_index, int) or peek_index < 1:
        raise ContractError(f"peek_index must be a positive int, got {peek_index!r}")
    # Exclusions change the scored count; scoring len(blocks) while reporting
    # fewer valid misrepresents evidence — hence the equality gate.
    if peek_index != len(result.valid):
        raise ContractError(
            f"peek_index {peek_index} != {len(result.valid)} valid wall blocks "
            f"({len(result.excluded)} excluded beyond tolerance)"
        )
    valid_ids = {wall_id for wall_id, _ in result.valid}
    valid_blocks = tuple(block for block in blocks if block.wall_id in valid_ids)
    contrasts = placement_block_contrast(valid_blocks)
    selection_gate_check(config, peek_index)
    seed_bytes = hashlib.sha256(f"score-selection-v1:{config.seed}".encode()).digest()
    stream = RandomStream(seed_bytes)
    est, low, high = bootstrap_blocks(
        contrasts, stream=stream, resamples=config.resamples, alpha=config.alpha
    )
    return fmean(contrasts), (est, low, high), result.excluded
