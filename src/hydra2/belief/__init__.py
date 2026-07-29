"""Hydra2 belief package — natural harness (WP-07A) + oracle distillation (WP-07B).

WP-07A owns: natural.py, kernel.py, corpus.py, confirmation.py, world.py
WP-07B owns: oracle_loader.py, oracle_distillation.py

This __init__ re-exports both namespaces without cross-import leakage.
Natural imports are deferred to avoid hard dependency before WP-07A lands.
"""

from __future__ import annotations

# WP-07B oracle exports (always available)
try:
    from hydra2.belief.oracle_distillation import (
        BrierScoreResult,
        CalibrationResult,
        DistillationConfig,
        DistillationMetrics,
        DuplicateBlockComparison,
        OracleTeacher,
        ProperScoreResult,
        StudentBeliefModel,
        brier_score,
        calibration_ece,
        compare_duplicate_blocks,
        compute_proper_scores,
        distillation_loss,
        expected_calibration_error,
        hidden_permutation_invariance_check,
    )
    from hydra2.belief.oracle_loader import (
        AUTHORIZED_TRAIN_SPLIT,
        FORBIDDEN_IN_ACTOR_KEYS,
        PRIVILEGED_KEYS,
        OracleTarget,
        PrivilegedOracleLoader,
        assert_privileged_loader_isolated_from_encoder,
        check_split_disjoint,
        check_wall_leakage,
        load_oracle_batch_in_subprocess,
        validate_actor_batch_no_privileged,
    )
except ImportError:
    pass

# WP-07A natural harness — deferred, optional until WP-07A lands
try:
    from hydra2.belief.confirmation import (
        ConfirmationCase,
        ConfirmationResult,
        NaturalConfirmationRunner,
    )
    from hydra2.belief.corpus import TinyCorpus, build_tiny_corpus
    from hydra2.belief.kernel import NaturalPacketKernel, PacketSuccessor
    from hydra2.belief.natural import BeliefEpoch, Particle, PolicySet, ProposalSpec
except ImportError:
    pass

try:  # noqa: SIM105
    from hydra2.belief.world import FullWorld
except ImportError:
    pass
__all__ = [
    "AUTHORIZED_TRAIN_SPLIT",
    "FORBIDDEN_IN_ACTOR_KEYS",
    "PRIVILEGED_KEYS",
    "BeliefEpoch",
    "BrierScoreResult",
    "CalibrationResult",
    "ConfirmationCase",
    "ConfirmationResult",
    "DistillationConfig",
    "DistillationMetrics",
    "DuplicateBlockComparison",
    "FullWorld",
    "NaturalConfirmationRunner",
    "NaturalPacketKernel",
    "OracleTarget",
    "OracleTeacher",
    "PacketSuccessor",
    "Particle",
    "PolicySet",
    "PrivilegedOracleLoader",
    "ProperScoreResult",
    "ProposalSpec",
    "StudentBeliefModel",
    "TinyCorpus",
    "assert_privileged_loader_isolated_from_encoder",
    "brier_score",
    "build_tiny_corpus",
    "calibration_ece",
    "check_split_disjoint",
    "check_wall_leakage",
    "compare_duplicate_blocks",
    "compute_proper_scores",
    "distillation_loss",
    "expected_calibration_error",
    "hidden_permutation_invariance_check",
    "load_oracle_batch_in_subprocess",
    "validate_actor_batch_no_privileged",
]
