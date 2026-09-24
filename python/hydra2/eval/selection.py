"""Manual-gate selection over wall blocks (no auto-wire)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Literal

from hydra2._native import eval as _bridge_eval  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream
from hydra2.eval.blocks import (
    BlockTolerance,
    ExcludedBlock,
    WallBlock,
    aggregate_blocks,
)
from hydra2.eval.statistics import (
    bootstrap_blocks as bootstrap_blocks,
)
from hydra2.eval.statistics import (
    placement_block_contrast as placement_block_contrast,
)
from hydra2.eval.statistics import (
    sequential_design_guard as sequential_design_guard,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.eval.telemetry import ResourceTelemetry

__all__ = [
    "SelectionConfig",
    "score_selection",
    "selection_gate_check",
]


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
    seed_bytes_oracle = hashlib.sha256(f"score-selection-v1:{config.seed}".encode()).digest()
    try:
        seed_leaf = _bridge_eval.eval_score_selection_seed_bytes  # type: ignore[attr-defined]  # reason: eval_leaves2 leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        seed_bytes = seed_bytes_oracle
    else:
        try:
            bridged_seed: bytes = seed_leaf(config.seed)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(str(exc)) from exc
        if bridged_seed != seed_bytes_oracle:
            raise ContractError("score_selection seed: Rust bytes != Python oracle")
        seed_bytes = bridged_seed
    stream = RandomStream(seed_bytes)
    est, low, high = bootstrap_blocks(
        contrasts, stream=stream, resamples=config.resamples, alpha=config.alpha
    )
    return fmean(contrasts), (est, low, high), result.excluded
