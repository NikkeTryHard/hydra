"""Uncertainty machinery — blocks, bootstrap, sign-flip, CS.

The independent unit of confirmation is the COMPLETE WALL BLOCK; primary
block outcome is the declared expected-final-placement contrast; alpha,
beta, pilot s, practical margin delta, multiplicity, maximum blocks, and
fixed-N versus named time-uniform CS are frozen blind to arm labels.

Everything here resamples whole blocks after collapsing each block to a
single contrast value; games inside a wall share the wall deck and are
therefore never independent units:

* :func:`bootstrap_blocks` — percentile bootstrap over block contrasts;
* :func:`sign_flip_interval` — random Rademacher sign flips of block
  contrasts (symmetric-null interval);
* :func:`fixed_n_samples` — the fixed sample size formula
  ``N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)**2)`` with alpha, beta,
  pilot s, and practical margin delta frozen blind to arm labels;
* :func:`hedged_confidence_sequence` / :func:`hedged_cs_path` — a concrete,
  named time-uniform confidence sequence: the hedged-betting capital process
  of Waudby-Smith & Ramdas (2023) with predictable lambdas and a grid union
  bound over means, valid simultaneously at every peek time;
* :func:`sequential_design_guard` — adaptive peeking without a declared
  sequential design invalidates confirmation;
* :func:`cluster_bootstrap` — clustered diagnostics resample game/player
  groups, never decisions (grouping is a two-value literal by construction).

All randomness enters through a semantic :class:`RandomStream`, so every
interval is reproducible from its key.
"""

from __future__ import annotations

import math
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from hydra2._native import eval as _bridge_eval  # pyrefly: ignore[missing-import]
from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from hydra2.contracts.randomness import RandomStream
    from hydra2.eval.blocks import WallBlock

__all__ = [
    "ClusterGrouping",
    "bootstrap_blocks",
    "ci_covers",
    "cluster_bootstrap",
    "fixed_n_samples",
    "hedged_confidence_sequence",
    "hedged_cs_path",
    "placement_block_contrast",
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
    contrasts = tuple(_bridge_eval.aggregate_wall_block(block) for block in ordered)
    for value in contrasts:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"block contrast must be finite, got {value!r}")
        if not math.isfinite(float(value)):
            raise ContractError(f"block contrast must be finite, got {value!r}")
    return contrasts


def fixed_n_samples(*, s: float, delta: float, alpha: float, beta: float) -> int:
    """Fixed sample size ``N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)^2)``
    with alpha, beta, pilot s, and practical margin delta frozen blind to arm
    labels (bridge-checked)."""
    try:
        leaf = _bridge_search.eval_stats_fixed_n_samples  # type: ignore[attr-defined]  # reason: eval_stats leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            n: int = leaf(s, delta, alpha, beta)
            return n
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(str(exc)) from exc
    for name, value in (("s", s), ("delta", delta)):
        if not math.isfinite(value) or value <= 0:
            raise ContractError(f"{name} must be positive and finite")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ContractError("alpha and beta must lie in (0, 1)")
    z_a = _STD.inv_cdf(1.0 - alpha)
    z_b = _STD.inv_cdf(1.0 - beta)
    return math.ceil(((z_a + z_b) * s / delta) ** 2)


def _bootstrap_percentile_indexes(alpha: float, resamples: int) -> tuple[int, int]:
    """Percentile indexes verbatim (bridge-checked, bit-exact).

    ``low = floor(alpha / 2 * R)``, ``high = ceil((1 - alpha / 2) * R) - 1``
    (T3 pins ``R=2000, alpha=0.05`` → ``(50, 1949)``). Index math executes in
    ``hydra2._native.search`` (``eval_stats_percentile_indexes``); the
    numpy-draw lane (M2) stays Python.
    """
    try:
        leaf = _bridge_search.eval_stats_percentile_indexes  # type: ignore[attr-defined]  # reason: eval_stats leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            raw: tuple[int, int] = leaf(alpha, resamples)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(str(exc)) from exc
        low, high = raw
        return low, high
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
    """Gate helper: does the interval cover ``truth``? (bridge-checked, bit-exact)."""
    try:
        leaf = _bridge_search.eval_stats_ci_covers  # type: ignore[attr-defined]  # reason: eval_stats leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        covers: bool = leaf(bounds[0], bounds[1], truth)
        return covers
    return bounds[0] <= truth <= bounds[1]


def _require_eval_bridge() -> Any:
    """Import the built ``eval`` bridge surface with ``hedged_cs_path`` (fail closed)."""
    if not hasattr(_bridge_eval, "hedged_cs_path"):
        raise ImportError(
            "hydra2._native.eval.hedged_cs_path missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return _bridge_eval


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

    Draw-free capital math rides the bridge
    (``hydra-search::eval::statistics::hedged_cs_path``, GIL released);
    caller-side type gates stay here — ``bool`` has no ``f64`` analogue and
    would silently coerce across the boundary. Bit-identical to the retired
    oracle (pinned by the /tmp throwaway strict probe: 7 shapes, exact f64
    bits); 10-12x faster on the hottest shape.
    """
    _validate_alpha_resamples(alpha, 100)
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 4 <= grid_size <= 512:
        raise ContractError("grid_size must be an int in [4, 512]")
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ContractError("bounds must be finite with high > low")
    clean: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ContractError(f"value must be finite, got {value!r}")
        clean.append(float(value))
    if peek_times is not None:
        peeks = list(peek_times)
        count = len(clean)
        for moment in peeks:
            if isinstance(moment, bool) or not isinstance(moment, int) or not 1 <= moment <= count:
                raise ContractError(f"peek times must be ints in [1, {count}]")
    else:
        peeks = None
    try:
        out: list[tuple[float, float]] = _require_eval_bridge().hedged_cs_path(
            clean, alpha, low, high, grid_size, peeks
        )
    except ImportError:
        raise
    except Exception as exc:
        raise ContractError(f"eval bridge hedged_cs_path failed: {exc}") from exc
    return tuple(out)


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
    """Adaptive peeking without a declared sequential design is fatal (bridge-checked).

    ``fixed_n`` allows exactly one look at the precommitted N; anything else
    raises. ``time_uniform_cs`` declares its peek schedule up front.
    """
    peeks = list(declared_peeks)
    try:
        leaf = _bridge_search.eval_stats_sequential_design_guard  # type: ignore[attr-defined]  # reason: eval_stats leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            leaf(design, peeks)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(str(exc)) from exc
        return

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
