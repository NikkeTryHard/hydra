"""Hydra2 evaluation schemas and synthetic statistics (WP-03B, SPEC 18).

Public surface: schedule commitment (:mod:`hydra2.eval.schedule`), case
declarations (:mod:`hydra2.eval.case`), whole-wall-block aggregation and
invalid-block policy (:mod:`hydra2.eval.blocks`), block-level uncertainty —
bootstrap, sign-flip, fixed-N, hedged confidence sequence, clustered
diagnostics (:mod:`hydra2.eval.statistics`), resource telemetry
(:mod:`hydra2.eval.telemetry`), and promotion records
(:mod:`hydra2.eval.promotion`).
"""

from hydra2.eval.blocks import (
    BlockAggregateResult,
    BlockTolerance,
    ExcludedBlock,
    WallBlock,
    aggregate_blocks,
    aggregate_wall_block,
)
from hydra2.eval.case import (
    PRIMARY_METRIC,
    EvalCase,
    case_manifest_hash,
    make_eval_case,
)
from hydra2.eval.promotion import PromotionRecord, make_promotion_record, promotion_digest
from hydra2.eval.schedule import (
    TOTAL_GAMES_PER_WALL,
    MatchSchedule,
    build_match_schedule,
    schedule_commitment_hash,
    seat_pair_placements_exact,
)
from hydra2.eval.statistics import (
    bootstrap_blocks,
    ci_covers,
    cluster_bootstrap,
    fixed_n_samples,
    hedged_confidence_sequence,
    hedged_cs_path,
    sequential_design_guard,
    sign_flip_interval,
)
from hydra2.eval.telemetry import (
    ResourceTelemetry,
    TelemetryTolerance,
    make_resource_telemetry,
    telemetry_invalid_reason,
)

__all__ = [
    "PRIMARY_METRIC",
    "TOTAL_GAMES_PER_WALL",
    "BlockAggregateResult",
    "BlockTolerance",
    "EvalCase",
    "ExcludedBlock",
    "MatchSchedule",
    "PromotionRecord",
    "ResourceTelemetry",
    "TelemetryTolerance",
    "WallBlock",
    "aggregate_blocks",
    "aggregate_wall_block",
    "bootstrap_blocks",
    "build_match_schedule",
    "case_manifest_hash",
    "ci_covers",
    "cluster_bootstrap",
    "fixed_n_samples",
    "hedged_confidence_sequence",
    "hedged_cs_path",
    "make_eval_case",
    "make_promotion_record",
    "make_resource_telemetry",
    "promotion_digest",
    "schedule_commitment_hash",
    "seat_pair_placements_exact",
    "sequential_design_guard",
    "sign_flip_interval",
    "telemetry_invalid_reason",
]
