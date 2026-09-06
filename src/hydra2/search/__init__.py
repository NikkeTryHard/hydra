"""Hydra2 search package — frozen candidate and baselines (Wave 8-9).

Wave 8 owns: common (CandidateSpec/Search API), candidate0 (frozen policy),
ismcts_natural, despot_natural (peers own the latter two).
Wave 9A owns: pbrf (Candidate 3 PBRF core).
Wave 9B owns: modules (Candidate 4 modules one-at-a-time).
Wave 9C owns: persistence_factorial (B/F/R/P/C factorial).
Wave 9D owns: local_resolving (Candidate 5).
Wave 9E owns: gumbel (Candidate 6).
This __init__ re-exports the shared search surface without hard dependency on
torch until candidate0 is imported.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from hydra2.search.candidate0 import (
        FrozenCandidate0,
        candidate0,
        frozen_choice,
        make_candidate0_spec,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.ismcts_natural import (  # noqa: F401
        FORBIDDEN_IN_TREE_KEY,
        InformationSetNode,
        NaturalISMCTSConfig,
        NaturalISMCTSPlanner,
        UniformContinuationPolicy,
        double_weighting_oracle_detects_correction,
        info_key_for_observation,
        is_redeterminization_enabled,
        model_vector_for_world,
        scalarize_vector,
        terminal_vector_for_world,
        validate_tree_keys_contain_no_world_id,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.despot_natural import (
        DespotConfig,
        NaturalDespotPlanner,
        NaturalScenario,
        make_despot_candidate_spec,
        packet_aliasing_rejected,
        proposal_reversal_fixture,
        validate_packet_partition,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.common import (
        CandidateSpec,
        Planner,
        ResourceBudget,
        SearchRequest,
        SearchResult,
        candidate_spec_hash,
        candidate_spec_to_json,
        resource_budget_to_json,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.pbrf import (
        ChildEntry,
        CommitDisposition,
        ImmutableForest,
        PbrfConfig,
        PbrfPlanner,
        build_pbrf,
        commit,
        fixed_allocate,
        make_pbrf_candidate_spec,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.modules import (  # noqa: F401
        MODULE_REGISTRY,
        PERSISTENCE_GATE_MODULE,
        VALID_MODULE_IDS,
        ControlledSMCModule,
        DefensiveMISModule,
        FixedMLMCModule,
        PbrfContext,
        PersistentForestModule,
        PrimalDualPruningModule,
        RaoBlackwellModule,
        RQMCModule,
        ScenarioCoresetModule,
        StructuralCRNModule,
        VOCRoutingModule,
        apply_module,
        make_candidate4_spec,
        make_core_control_spec,
        module_evidence,
        validate_one_at_a_time,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.persistence_factorial import (
        ARM_DEFS,
        FactorialContrasts,
        FactorialReport,
        FinitePacket,
        ForestState,
        PersistenceArm,
        PersistencePlanner,
        commit_equals_rebuild,
        compute_packet_id,
        deterministic_gumbel_for_arm,
        enumerate_packets_for,
        factorial_contrasts,
        fresh_rebuild_epoch,
        generate_factorial_report,
        make_persistence_arm,
        make_persistence_candidate_spec,
        stratify_surprise_miss_recovery,
        validate_deadline_and_fallback,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.local_resolving import (
        LocalResolvingConfig,
        LocalResolvingPlanner,
        PublicSubgame,
        StrategyTable,
        abstraction_round_trip,
        build_public_subgame,
        detect_cycle,
        info_key_for_actor_observation,
        is_equilibrium_claimed,
        leaf_vector_replay,
        make_candidate5_spec,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.gumbel import (
        GumbelSearchConfig,
        GumbelSearchPlanner,
        PuctBaselinePlanner,
        PuctConfig,
        cached_full_history_agreement,
        deterministic_gumbel,
        deterministic_root_gumbels,
        exact_transition,
        learned_rules_transition_rejected,
        make_gumbel_candidate_spec,
        make_puct_candidate_spec,
        validate_hidden_permutation_invariance,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.profiles import (
        PROFILES,
        CandidateProfile,
        admit,
        compare_gumbel_puct,
        jobs_for,
        transitions_bound,
    )

with contextlib.suppress(ImportError):
    from hydra2.search.joint_type_world import (
        JointParticle,
        JointPosterior,
        JointTypeWorldConfig,
        JointTypeWorldPlanner,
        OpponentTypePolicy,
        UncertaintySet,
        coherent_trajectory,
        deterministic_joint_gumbel,
        exact_joint_posterior_oracle,
        hidden_marginalization,
        make_joint_type_world_candidate_spec,
        preserve_correlation_check,
        sequential_joint_update,
        validate_same_information_equality,
    )
    from hydra2.search.joint_type_world import (
        info_key_for_observation as joint_info_key_for_observation,
    )

__all__ = [
    "ARM_DEFS",
    "MODULE_REGISTRY",
    "PERSISTENCE_GATE_MODULE",
    "PROFILES",
    "VALID_MODULE_IDS",
    "CandidateProfile",
    "CandidateSpec",
    "ChildEntry",
    "CommitDisposition",
    "ControlledSMCModule",
    "DefensiveMISModule",
    "DespotConfig",
    "FactorialContrasts",
    "FactorialReport",
    "FinitePacket",
    "FixedMLMCModule",
    "ForestState",
    "FrozenCandidate0",
    "GumbelSearchConfig",
    "GumbelSearchPlanner",
    "ImmutableForest",
    "InformationSetNode",
    "JointParticle",
    "JointPosterior",
    "JointTypeWorldConfig",
    "JointTypeWorldPlanner",
    "LocalResolvingConfig",
    "LocalResolvingPlanner",
    "NaturalDespotPlanner",
    "NaturalISMCTSConfig",
    "NaturalISMCTSPlanner",
    "NaturalScenario",
    "OpponentTypePolicy",
    "PbrfConfig",
    "PbrfContext",
    "PbrfPlanner",
    "PersistenceArm",
    "PersistencePlanner",
    "Planner",
    "PublicSubgame",
    "PuctBaselinePlanner",
    "PuctConfig",
    "RQMCModule",
    "RaoBlackwellModule",
    "ResourceBudget",
    "ScenarioCoresetModule",
    "SearchRequest",
    "SearchResult",
    "StrategyTable",
    "StructuralCRNModule",
    "UncertaintySet",
    "UniformContinuationPolicy",
    "VOCRoutingModule",
    "abstraction_round_trip",
    "admit",
    "apply_module",
    "build_pbrf",
    "build_public_subgame",
    "cached_full_history_agreement",
    "candidate0",
    "candidate_spec_hash",
    "candidate_spec_to_json",
    "coherent_trajectory",
    "commit",
    "commit_equals_rebuild",
    "compare_gumbel_puct",
    "compute_packet_id",
    "detect_cycle",
    "deterministic_gumbel",
    "deterministic_gumbel_for_arm",
    "deterministic_joint_gumbel",
    "deterministic_root_gumbels",
    "double_weighting_oracle_detects_correction",
    "enumerate_packets_for",
    "exact_joint_posterior_oracle",
    "exact_transition",
    "factorial_contrasts",
    "fixed_allocate",
    "fresh_rebuild_epoch",
    "frozen_choice",
    "generate_factorial_report",
    "hidden_marginalization",
    "info_key_for_actor_observation",
    "info_key_for_observation",
    "is_equilibrium_claimed",
    "is_redeterminization_enabled",
    "jobs_for",
    "joint_info_key_for_observation",
    "leaf_vector_replay",
    "learned_rules_transition_rejected",
    "make_candidate0_spec",
    "make_candidate4_spec",
    "make_candidate5_spec",
    "make_core_control_spec",
    "make_despot_candidate_spec",
    "make_gumbel_candidate_spec",
    "make_joint_type_world_candidate_spec",
    "make_pbrf_candidate_spec",
    "make_persistence_arm",
    "make_persistence_candidate_spec",
    "make_puct_candidate_spec",
    "model_vector_for_world",
    "module_evidence",
    "packet_aliasing_rejected",
    "preserve_correlation_check",
    "proposal_reversal_fixture",
    "resource_budget_to_json",
    "scalarize_vector",
    "sequential_joint_update",
    "stratify_surprise_miss_recovery",
    "terminal_vector_for_world",
    "transitions_bound",
    "validate_deadline_and_fallback",
    "validate_hidden_permutation_invariance",
    "validate_one_at_a_time",
    "validate_packet_partition",
    "validate_same_information_equality",
    "validate_tree_keys_contain_no_world_id",
]
