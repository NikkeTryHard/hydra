"""Shared test configuration and deterministic helpers.

CUBLAS_WORKSPACE_CONFIG MUST be set before any CUDA context exists for
torch.use_deterministic_algorithms to cover GEMMs; this module is imported
before test modules by pytest.

Single source of truth for work-package selection AND the BUILD §1 report
writer (WP-03A/WP-03C cutover): ``--package WP-ID`` (or ``HYDRA2_TEST_PACKAGE``)
keeps only tests marked ``@pytest.mark.contract_package("<WP-ID>")``, and the
per-module outcomes are aggregated into named checklist fields and published
atomically under ``$HYDRA2_ARTIFACT_ROOT/reports/<package-or-ALL>/``. The
writer lives at this root level because pytest scopes ``logreport`` hookimpls
to their conftest's subtree; a suite-local conftest never sees reports for
packages whose tests live elsewhere (observed with WP-03C under tests/engines/).
The option previously existed only in the contracts suite's own conftest and
was invisible to repo-wide runs.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch


def _default_inductor_cache_dir() -> str:
    # Persistent torch.inductor disk cache, version-keyed so torch/triton
    # upgrades cannot poison it. setdefault: explicit env always wins.
    # Read by torch._inductor (async_compile/autotune); fx_graph_cache
    # defaults on in torch 2.14 — no other inductor knobs are flipped here.
    safe_ver = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in str(torch.__version__)
    )
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "hydra2", f"inductor-torch{safe_ver}")


os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _default_inductor_cache_dir())

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

#: module basename -> named report fields asserted by that module. Moved
#: verbatim from tests/contracts/conftest.py during the single-source cutover.
CHECKLIST_FIELDS_BY_MODULE = {
    "test_art_canon_001_golden.py": (
        "fixture_ART-CANON-001",
        "rfc8785_canonical_bytes",
        "sha256_independent_recomputation",
    ),
    "test_art_canon_001_edges.py": (
        "numeric_and_unicode_edges",
        "canonical_domain_rejections",
    ),
    "test_art_atomic_001_publication.py": (
        "fixture_ART-ATOMIC-001",
        "atomic_publication",
    ),
    "test_registry_wp02a.py": (
        "artifact_envelope_identity",
        "immutable_registry_rejections",
        "compatibility_and_migration_metadata",
    ),
    "test_bootstrap_supersede.py": ("bootstrap_supersede_shim",),
    "test_rules_wp02b.py": (
        "manifest_canonical_hash_stable",
        "complete_flag_set",
        "rank_tie_break_encoded",
        "owner_decisions_explicit",
    ),
    "test_utility_wp02b.py": (
        "raw_utility_identity_round_trip",
        "seat_permutation_invariance",
        "malformed_settlement_rejection",
    ),
    # WP-02C (Wp02cBuilder): fields owned by that module.
    "test_action_wp02c.py": (
        "action_kind_ordinals_frozen",
        "canonical_action_invariants",
        "red_five_identity_preserved",
        "table_generation_order_stable",
        "encode_decode_bijection",
        "golden_bytes_engine_independent",
    ),
    # WP-02D: BUILD §5 checklist verbatim + SPEC §22 fixtures.
    "test_events_obs_wp02d.py": (
        "fixture_OBS-DORA-005",
        "fixture_OBS-DRAW-PRIVATE-001",
        "fixture_OBS-HIDDEN-PERM-001",
        "fixture_OBS-CANARY-001",
        "hidden_permutation_stability",
        "forbidden_canary_isolation",
        "concealed_draw_actor_only",
        "public_events_all_seats",
        "server_private_rejected_everywhere",
        "dora_shape_five_never_padded",
        "sentinel_and_contiguity",
        "mask_alignment_action_table_digest",
        "sequence_monotonicity",
        "call_window_single_successor_grouping",
        "packet_partition_exhaustive_nonempty",
    ),
    # WP-03B: BUILD §6 WP-03B checklist verbatim.
    "test_randomness_wp03b.py": (
        "semantic_seed_derivation",
        "wall_seat_latency_schedules_and_hashes",
        "four_seat_rotation_and_six_two_v_two_allocations",
        "final_evaluation_seeds_isolated_from_selection",
    ),
    "test_statistics_wp03b.py": (
        "whole_block_bootstrap_and_sign_flip",
        "fixed_n_formula_and_time_uniform_cs_boundary",
        "known_zero_effect_recovered",
        "known_nonzero_effect_recovered",
        "per_game_resampling_negative_test_fails",
    ),
    "test_blocks_telemetry_wp03b.py": (
        "wall_block_aggregation_games_not_independent",
        "resource_telemetry_schema",
        "invalid_block_policy_excluded_and_reported",
    ),
    "test_promotion_case_wp03b.py": ("promotion_record_schema",),
    "test_schedule_wp03b.py": (
        "schedule_replay_identical",
        "seat_balance_exact_under_schedule",
        "schedule_committed_before_results",
        "invalid_block_excluded_and_reported",
    ),
    # WP-03C (Wp03cBuilder): BUILD lines 381-393 checklist mapped 1:1.
    "test_mahjax_shell_wp03c.py": (
        "runtime_sha_verification",
        "environment_tuple_captured",
        "quarantined_default_state",
        "fail_closed_consumption",
        "token_gate_proven_with_test_only_fabrication",
    ),
    # WP-04B Authoritative Data Lineage — BUILD 428-443 plus hard failures
    "test_data_lineage_wp04b.py": (
        "raw_object_join_immutable",
        "ingest_via_packager_zstd_manifest",
        "decode_one_game_per_object",
        "validate_structure_event_order",
        "validate_tile_conservation_red",
        "validate_legality_calls_scores_termination",
        "quarantine_invalid_with_lineage",
        "partition_whole_games_before_expansion",
        "partition_grouping_and_duplicates",
        "walls_disjoint",
        "arrow_parquet_actor_privileged_separation",
        "privileged_leakage_hard_failure",
        "dora_shape_five_not_four",
        "content_addressed_tensor_caches",
        "loader_hash_legal_mask_verification",
        "fresh_process_batch_load",
        "hard_failures_silent_skip_partial_split_corrupt",
    ),
    # WP-05B Supervised Loop — BUILD §8: masked BC, auxiliary, optimizer/sched/accum/ckpt,
    # plain+Fabric identical, resume, local artifacts, reporting, deterministic synthetic parquet, no privileged.
    "test_supervised_loop_wp05b.py": (
        "masked_behavior_cloning_objective",
        "value_event_auxiliary_with_explicit_weights",
        "project_owned_optimizer_scheduler_accumulation_checkpoint",
        "plain_and_fabric_identical_loop_state",
        "resume_restores_model_optimizer_scheduler_step_rng_sampler_manifest",
        "local_artifacts_authoritative_wandb_mirror_does_not_overwrite",
        "reports_masked_nll_topk_calibration_support_confusion_strata_legal_uniform",
        "deterministic_training_over_authoritative_synthetic_parquet",
        "no_privileged_fields",
    ),
    # WP-05C Baseline Qualification — BUILD §8 Wave 5: tiny overfit, deterministic resume,
    # fresh-process inference, reference games, hidden/canary, eager oracle, compile ladder,
    # shape arm not_activated.
    "test_baseline_wp05c.py": (
        "baseline_metrics",
        "held_out_eval",
        "deterministic",
        "report",
    ),
    # WP-05A Model and Inference Contract — BUILD §8 / SPEC 11.
    "test_model_inference_wp05a.py": (
        "actor_visible_tensor_encoder_no_privileged_fields",
        "padded_bucketed_histories_with_explicit_masks",
        "model_contract_inference_contract_deterministic_shapes_masks",
        "sdpa_dense_attention_eval_dropout_zero",
        "dense_legal_policy_head",
        "four_seat_value_distribution_vector_head",
        "event_likelihood_heads_for_belief",
        "legal_mask_before_selection_loss",
        "diagnostics_without_hidden_fields",
        "cache_full_history_encoding_agreement",
        "optional_shape_features_excluded",
    ),
    # WP-06 Duplicate-Block Match Qualification — BUILD §9 / SPEC 18.
    "test_duplicate_block_wp06.py": (
        "exact_and_near_duplicate_detection",
        "disjoint_wall_sets_enforced",
        "block_splitting_whole_walls",
        "whole_block_aggregation",
        "invalid_block_excluded_and_reported",
        "seat_balance_audit",
        "telemetry_completeness_report",
        "held_out_partition_hidden",
        "fresh_process_block_load",
    ),
    # WP-07B Oracle Belief Distillation — BUILD §10 / SPEC 14-18 teacher-student deterministic
    "test_oracle_distillation_wp07b.py": (
        "separate_privileged_loader_namespace_process_boundary",
        "train_belief_value_targets_only_from_authorized_train_split",
        "never_expose_privileged_fields_to_inference_encoder",
        "report_proper_scores_calibration_on_held_out_data",
        "compare_duplicate_blocks_without_changing_frozen_supervised_gate",
        "hidden_permutation_and_split_wall_leakage_tests",
        "teacher_student_deterministic",
    ),
    # WP-07A Natural Belief Harness — BUILD §10 / SPEC 14 (natural, packet kernel, deterministic)
    "test_belief_natural_wp07a.py": (
        "belief_epoch_immutable_target_identity",
        "natural_world_law_consistent_with_actor_observation",
        "scoreable_proposal_samples_with_log_target_proposal",
        "actor_conditional_sampler_with_immutable_constraints",
        "disjoint_next_actor_visible_packet_kernel",
        "physical_transition_and_actor_policy_likelihood",
        "exact_pushforward_then_condition",
        "epoch_increment_after_committed_transition",
        "stale_provenance_epoch_target_rejection",
        "tiny_finite_world_corpus_with_exact_probabilities",
        "natural_full_fidelity_confirmation_runner",
        "packet_mass_one",
        "no_duplicate_missing_packet",
        "parent_only_reweight_negative_fixture",
        "pushforward_equals_rebuild",
        "hidden_permutation_invariance",
        "density_normalization_support",
        "deterministic_confirmation_replay",
    ),
    "test_belief_confirmation_wp07a.py": (
        "belief_epoch_immutable_target_identity",
        "natural_world_law_consistent_with_actor_observation",
        "scoreable_proposal_samples_with_log_target_proposal",
        "actor_conditional_sampler_with_immutable_constraints",
        "disjoint_next_actor_visible_packet_kernel",
        "physical_transition_and_actor_policy_likelihood",
        "exact_pushforward_then_condition",
        "epoch_increment_after_committed_transition",
        "stale_provenance_epoch_target_rejection",
        "tiny_finite_world_corpus_with_exact_probabilities",
        "natural_full_fidelity_confirmation_runner",
        "packet_mass_one",
        "no_duplicate_missing_packet",
        "parent_only_reweight_negative_fixture",
        "pushforward_equals_rebuild",
        "hidden_permutation_invariance",
        "density_normalization_support",
        "deterministic_confirmation_replay",
    ),
    # WP-08A Candidate 0 Frozen Policy — BUILD §11 / SPEC 15-16.1 / Blueprint §7
    "test_candidate0_wp08a.py": (
        "frozen_policy_baseline",
        "deterministic",
        "report",
        "exact_blueprint_candidate0_api",
        "one_model_call_no_search",
        "greedy_frozen_temperature_value_arms",
        "deadline_fallback_is_candidate0",
        "zero_legality_leak_replay",
        "candidate_spec_result_promotion_bound",
    ),
    # WP-08B Candidate 1 Natural ISMCTS — BUILD §11 / SPEC 16.2 / Blueprint §8
    "test_ismcts_natural_wp08b.py": (
        "ismcts_natural",
        "determinism",
        "budget",
        "report",
        "natural_worlds_only_no_importance_ratios",
        "root_tree_keys_use_root_information_set_only",
        "non_root_policies_consume_actor_observation_in_sandbox",
        "carry_vector_values_scalarize_only_root_selection",
        "freeze_uct_depth_budget_continuation_policies_rng_semantics",
        "re_determinization_disabled",
        "candidate1_tests_from_blueprint",
        "confirm_naturally_under_matched_resources",
    ),
    # WP-08C Candidate 2 Natural DESPOT — BUILD §11 / SPEC 16.3 / Blueprint §9
    "test_despot_natural_wp08c.py": (
        "despot_natural_scenarios_only",
        "despot_no_proposal_weights",
        "despot_feasible_lower_not_bound",
        "despot_priority_proxy_not_upper_bound",
        "despot_packet_partition",
        "despot_proposal_reversal",
        "despot_packet_aliasing_rejected",
        "despot_determinism",
        "despot_budget_enforcement",
        "despot_resource_views",
        "despot_candidate_spec_hash_stable",
        "despot_report",
    ),
    # WP-09A Candidate 3 PBRF Core — BUILD §12 / SPEC 16.4 / Blueprint §10
    "test_pbrf_wp09a.py": (
        "pbrf_core",
        "determinism",
        "report",
        "natural_immutable_parent_population",
        "freeze_root_candidate_generator_before_search_evidence",
        "exhaustively_enumerate_immediate_disjoint_packet_kernel_per_parent_action",
        "store_parent_id_successor_delta_raw_weight_provenance",
        "require_child_normalizer_partition_within_tolerance",
        "allocate_fixed_search_batches",
        "freeze_candidates_before_natural_confirmation",
        "commit_only_authoritative_realized_child",
        "increment_belief_epoch_squash_incompatible_siblings_statistics",
        "missing_packet_mass_is_hard_failure",
        "stale_child_is_hard_failure",
        "confirmation_reversal_is_hard_failure",
        "no_hidden_state_leak",
    ),
    # WP-09B Candidate 4 Modules — one at a time (BUILD §12 / SPEC 16.5 / Blueprint §11.1-11.10)
    "test_modules_wp09b.py": (
        "modules_one_at_a_time_candidate",
        "determinism",
        "report",
    ),
    # WP-09C Persistence Factorial — BUILD §12 / SPEC 17 / Blueprint §11.11
    "test_persistence_factorial_wp09c.py": (
        "persistence_factorial",
        "determinism",
        "report",
        "b_frozen_policy_no_search_state_no_ponder",
        "f_fresh_search_discard_state_no_ponder",
        "r_retain_compatible_no_opponent_compute",
        "p_retain_and_ponder_only_between_action_and_packet",
        "c_laboratory_fresh_extended_budget_never_deployable",
        "own_deadline_and_fallback_margin_enforced",
        "actual_resource_logging_never_claim_equality",
        "packet_commit_rebuild_equality",
        "determinism_across_replay",
        "p_f_r_f_p_r_p_c_contrasts_with_uncertainty",
        "surprise_miss_recovery_stratified",
        "exact_b_f_r_p_c_state_machine_fixtures",
        "whole_block_factorial_report_frozen",
    ),
    # WP-09D Candidate 5 Local Resolving — BUILD §12 / SPEC 16.6 / Blueprint §12
    "test_local_resolving_wp09d.py": (
        "local_resolving_candidate",
        "determinism",
        "report",
        "same_information_strategy_keying",
        "settlement_utility_vector_preservation",
        "exhaustive_tiny_game",
        "cycle_detection",
        "leaf_replay",
        "abstraction_round_trip",
        "frozen_update_averaging",
        "pbrf_warm_start_comparison",
        "no_equilibrium_claim",
        "build_declared_subgame_horizon_abstraction",
        "strategies_keyed_by_actor_information_nodes",
        "resource_budget_enforcement",
    ),
    # WP-09E Candidate 6 Gumbel Search — BUILD §12 / SPEC 16.7 / Blueprint §13
    "test_gumbel_wp09e.py": (
        "gumbel_search_candidate",
        "determinism",
        "report",
        "exact_rule_parity",
        "cache_full_history_equality",
        "hidden_permutation_invariance",
        "deterministic_gumbel_replay",
        "vector_backup_scalarize_only_root",
        "accounting_model_calls_transitions",
        "learned_rules_negative_control",
        "puct_comparator_matched",
        "sequential_halving_declared",
    ),
    # WP-10 Candidate 7 Teacher Distillation — BUILD §13 / SPEC 16.8 (M10)
    "test_teacher_distillation_wp10.py": (
        "teacher_distillation",
        "determinism",
        "report",
        "five_gate_teacher_selection",
        "teacher_selection_justification_before_trajectory",
        "actor_visible_record_with_search_policy_vector_return",
        "privileged_labels_isolated_namespace",
        "behavior_cloning_anchors_and_legal_mask",
        "frozen_train_split_checkpoint_calibration",
        "compare_five_arms",
        "split_wall_seed_leakage_audits",
        "duplicate_block_promotion_gate",
        "teacher_replacement_invalidates",
    ),
    # WP-11 Actor-Learner Replay — BUILD §14 / SPEC 20 (M11) — optional
    "test_replay_wp11.py": (
        "actor_learner_replay_over_authorized_data",
        "deterministic_replay",
        "no_privileged_fields",
    ),
    # WP-12 Analysis Qualification — BUILD §15 / SPEC 15+18.2 (M12)
    "test_analysis_wp12.py": (
        "freeze_finite_analysis_budgets_and_resource_caps",
        "reuse_identical_semantics",
        "permit_only_additional_charged_compute",
        "deterministic_replay_across_modes",
        "compare_actions_values_fallback",
        "reject_hidden_fields_altered_rules_changed_estimator_uncharged_work",
        "generate_hashed_analysis_report",
    ),
    # WP-13 Candidate 8 Joint Type/World — BUILD §16 / SPEC 16.9 / Blueprint §15
    "test_joint_type_world_wp13.py": (
        "joint_type_world_candidate",
        "determinism",
        "hidden_permutation_invariance",
        "report",
        "observation_only_type_policy",
        "joint_posterior_exact_oracle",
        "type_world_correlation_preserved",
        "coherent_uncertainty_set",
        "feasibility_nonempty_contains_nominal",
        "calibration_and_no_leakage",
    ),
}
_MODULE_STATS: dict[str, dict[str, int]] = {}


class TinyMLP(torch.nn.Module):
    """Minimal deterministic MLP for runtime probes (not the model contract)."""

    def __init__(self, features: int = 16, hidden: int = 32, outputs: int = 4) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(features, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def unwrap_model(model):
    """Peek through adapter wrappers to reach the parameter-bearing module."""
    seen = model
    for _ in range(4):
        inner = getattr(seen, "_fabric_module", None) or getattr(seen, "module", None)
        if inner is None:
            break
        seen = inner
    return seen


def make_model_and_optimizer(
    seed: int, *, lr: float = 1e-2, device: str | None = None
) -> tuple[TinyMLP, torch.optim.AdamW]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
    if device is not None:
        model = model.to(device)
    return model, optimizer


def make_batch(seed: int, rows: int = 8, features: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, features, generator=generator)
    y = torch.randn(rows, 4, generator=generator)
    return x, y


def run_supervised_steps(handle, x: torch.Tensor, y: torch.Tensor, steps: int) -> list[float]:
    """Eager supervised loop through a RuntimeHandle; returns per-step losses."""
    losses: list[float] = []
    x = x.to(handle.device)
    y = y.to(handle.device)
    for _ in range(steps):
        handle.optimizer.zero_grad(set_to_none=True)
        out = handle.model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        handle.backward(loss)
        handle.optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def capture_grads(handle, x: torch.Tensor, y: torch.Tensor):
    """One forward/backward without stepping; returns (loss, named grads)."""
    x = x.to(handle.device)
    y = y.to(handle.device)
    handle.optimizer.zero_grad(set_to_none=True)
    out = handle.model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    handle.backward(loss)
    grads = {
        name: param.grad.detach().clone()
        for name, param in unwrap_model(handle.model).named_parameters()
        if param.grad is not None
    }
    return loss.detach(), grads


def state_snapshot(obj) -> dict:
    """Device-detached clone of a module/optimizer state dict for comparisons."""
    raw = obj.state_dict()

    def detach(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: detach(v) for k, v in value.items()}
        if isinstance(value, list):
            return [detach(v) for v in value]
        return value

    return {key: detach(value) for key, value in raw.items()}


def assert_states_bitwise_equal(left: dict, right: dict, *, context: str) -> None:
    def walk(a, b, path):
        if isinstance(a, torch.Tensor):
            assert b is not None and a.shape == b.shape, f"{context}:{path} shape mismatch"
            assert a.dtype == b.dtype, f"{context}:{path} dtype differs"
            assert torch.equal(a.cpu(), b.cpu()), f"{context}:{path} differs bitwise"
        elif isinstance(a, dict):
            assert set(a) == set(b), f"{context}:{path} keys differ"
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), f"{context}:{path} length differs"
            for i, (ai, bi) in enumerate(zip(a, b, strict=True)):
                walk(ai, bi, f"{path}[{i}]")
        else:
            assert a == b, f"{context}:{path} scalar differs: {a!r} vs {b!r}"

    walk(left, right, "")


@pytest.fixture(autouse=True)
def _deterministic_algorithms():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--package",
        action="store",
        default=None,
        help="run only the tests marked for this work package (e.g. WP-03A)",
    )
    parser.addoption(
        "--candidate",
        action="store",
        default=None,
        help="alias for --package for Wave 8 search candidates (candidate0->WP-08A)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "contract_package(wp_id): test belongs to this work package's gate",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    requested = config.getoption("--package") or os.environ.get("HYDRA2_TEST_PACKAGE")
    candidate = config.getoption("--candidate")
    if candidate and not requested:
        mapping = {
            "candidate0": "WP-08A",
            "candidate1": "WP-08B",
            "candidate2": "WP-08C",
            "candidate3": "WP-09A",
            "candidate4": "WP-09B",
            "candidate5": "WP-09D",
            "candidate6": "WP-09E",
            "candidate7": "WP-10",
            "candidate8": "WP-13",
            "persistence-factorial": "WP-09C",
        }
        requested = mapping.get(candidate, candidate)
    if not requested:
        return
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        marker = item.get_closest_marker("contract_package")
        matches = marker is not None and bool(marker.args) and marker.args[0] == requested
        (selected if matches else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when not in ("call", "setup"):
        return
    module = report.nodeid.split("::", 1)[0]
    bucket = _MODULE_STATS.setdefault(module, {"passed": 0, "failed": 0, "skipped": 0})
    if report.failed:
        bucket["failed"] += 1
    elif report.when == "call":
        bucket["passed"] += 1
    elif report.skipped:
        bucket["skipped"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    try:
        from hydra2.artifacts.atomic import atomic_replace_bytes
        from hydra2.artifacts.canonical import canonical_bytes
        from hydra2.config import artifact_root
    except Exception:
        return

    fixtures: dict[str, dict[str, object]] = {}
    for module_name, stats in sorted(_MODULE_STATS.items()):
        basename = os.path.basename(module_name)
        fields = CHECKLIST_FIELDS_BY_MODULE.get(basename, (os.path.splitext(basename)[0],))
        entry = {
            "module": basename,
            "fields": list(fields),
            "status": "passed" if stats["failed"] == 0 else "failed",
            **stats,
        }
        for field in fields:
            fixtures[field] = entry
    report_document = {
        "work_package": (
            session.config.getoption("--package") or os.environ.get("HYDRA2_TEST_PACKAGE") or "ALL"
        ),
        "kind": "hydra2.contract_test_report",
        "status": "passed" if int(exitstatus) == 0 else "failed",
        "finished_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fixtures": fixtures,
        "totals": {
            "modules": len(_MODULE_STATS),
            "passed": sum(s["passed"] for s in _MODULE_STATS.values()),
            "failed": sum(s["failed"] for s in _MODULE_STATS.values()),
            "skipped": sum(s["skipped"] for s in _MODULE_STATS.values()),
        },
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        artifact_root() / "reports" / report_document["work_package"] / run_id / "report.json"
    )
    try:
        atomic_replace_bytes(destination, canonical_bytes(report_document))
    except OSError:
        return


def pytest_terminal_summary(terminalreporter) -> None:
    if not _MODULE_STATS:
        return
    total_failed = sum(s["failed"] for s in _MODULE_STATS.values())
    terminalreporter.section("contract report")
    terminalreporter.write_line(f"status: {'passed' if total_failed == 0 else 'failed'}")


@pytest.fixture(scope="session")
def require_cuda():
    """GPU probes hard-fail without CUDA; silent CPU fallback is forbidden."""
    if not torch.cuda.is_available():
        pytest.fail("BLOCKER: CUDA device unavailable; GPU probes cannot be qualified on CPU")
    return torch.device("cuda")
