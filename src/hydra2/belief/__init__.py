"""Hydra2 belief package — natural harness (WP-07A) + oracle distillation (WP-07B).

WP-07A owns: natural.py, kernel.py, corpus.py, confirmation.py, world.py
WP-07B owns: oracle_loader.py, oracle_distillation.py

This __init__ re-exports both namespaces without cross-import leakage.
Fail-closed: the package stays importable when optional deps are absent,
but attribute use raises ImportError with a `pixi run build-ext` hint
instead of leaving names undefined (never fail open).
"""

from __future__ import annotations

_ORACLE_NAMES = frozenset(
    {
        "AUTHORIZED_TRAIN_SPLIT",
        "BrierScoreResult",
        "CalibrationResult",
        "DistillationConfig",
        "DistillationMetrics",
        "DuplicateBlockComparison",
        "FORBIDDEN_IN_ACTOR_KEYS",
        "OracleTarget",
        "OracleTeacher",
        "PRIVILEGED_KEYS",
        "PrivilegedOracleLoader",
        "ProperScoreResult",
        "StudentBeliefModel",
        "assert_privileged_loader_isolated_from_encoder",
        "brier_score",
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
    }
)
_NATURAL_NAMES = frozenset(
    {
        "BeliefEpoch",
        "ConfirmationCase",
        "ConfirmationResult",
        "NaturalConfirmationRunner",
        "NaturalPacketKernel",
        "PacketSuccessor",
        "Particle",
        "PolicySet",
        "ProposalSpec",
        "TinyCorpus",
        "build_tiny_corpus",
    }
)
_WORLD_NAMES = frozenset({"FullWorld"})
# WP-07B oracle exports (fail closed on use)
try:
    from hydra2.belief.oracle_guard import (
        AUTHORIZED_TRAIN_SPLIT,
        FORBIDDEN_IN_ACTOR_KEYS,
        PRIVILEGED_KEYS,
        check_split_disjoint,
        check_wall_leakage,
        validate_actor_batch_no_privileged,
    )
    from hydra2.belief.oracle_join import (
        assert_privileged_loader_isolated_from_encoder,
        load_oracle_batch_in_subprocess,
    )
    from hydra2.belief.oracle_models import (
        DistillationConfig,
        OracleTeacher,
        StudentBeliefModel,
        distillation_loss,
    )
    from hydra2.belief.oracle_scores import (
        BrierScoreResult,
        CalibrationResult,
        DistillationMetrics,
        DuplicateBlockComparison,
        ProperScoreResult,
        brier_score,
        calibration_ece,
        compare_duplicate_blocks,
        compute_proper_scores,
        expected_calibration_error,
        hidden_permutation_invariance_check,
    )
    from hydra2.belief.oracle_store import PrivilegedOracleLoader
    from hydra2.belief.oracle_targets import OracleTarget
except ImportError as _oracle_exc:
    _ORACLE_IMPORT_ERROR: ImportError | None = _oracle_exc
else:
    _ORACLE_IMPORT_ERROR = None
# WP-07A natural harness (fail closed on use)
try:
    from hydra2.belief.confirmation import (
        ConfirmationCase,
        ConfirmationResult,
        NaturalConfirmationRunner,
    )
    from hydra2.belief.corpus import TinyCorpus, build_tiny_corpus
    from hydra2.belief.kernel import NaturalPacketKernel, PacketSuccessor
    from hydra2.belief.natural import BeliefEpoch, Particle, PolicySet, ProposalSpec
except ImportError as _natural_exc:
    _NATURAL_IMPORT_ERROR: ImportError | None = _natural_exc
else:
    _NATURAL_IMPORT_ERROR = None
try:
    from hydra2.belief.world import FullWorld
except ImportError as _world_exc:
    _WORLD_IMPORT_ERROR: ImportError | None = _world_exc
else:
    _WORLD_IMPORT_ERROR = None


def __getattr__(name: str) -> object:
    """Lazy fail-closed attribute access (never leave names undefined)."""
    if name in _ORACLE_NAMES and _ORACLE_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief oracle submodule not importable "
            f"({_ORACLE_IMPORT_ERROR}); "
            "build the bridge with `pixi run build-ext` before using "
            f"{name!r}"
        ) from _ORACLE_IMPORT_ERROR
    if name in _NATURAL_NAMES and _NATURAL_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief natural harness submodule not importable "
            f"({_NATURAL_IMPORT_ERROR}); "
            "build the bridge with `pixi run build-ext` before using "
            f"{name!r}"
        ) from _NATURAL_IMPORT_ERROR
    if name in _WORLD_NAMES and _WORLD_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief.world submodule missing "
            f"({_WORLD_IMPORT_ERROR}); "
            "rebuild the bridge with `pixi run build-ext` before using "
            f"{name!r}"
        ) from _WORLD_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
