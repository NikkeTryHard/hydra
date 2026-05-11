use super::*;

#[test]
fn benchmark_gate_evaluation_matches_thresholds() {
    let metrics = BenchmarkGateMetrics {
        afbs_on_turn_ms: 120.0,
        ct_smc_dp_ms: 0.8,
        endgame_exact_ms: 80.0,
        self_play_games_per_sec: 25.0,
        distill_kl_drift: 0.02,
    };
    assert!(evaluate_benchmark_gates(&metrics, 0.05).passed);

    let failed = evaluate_benchmark_gates(
        &BenchmarkGateMetrics {
            self_play_games_per_sec: 19.0,
            ..metrics
        },
        0.05,
    );
    assert!(!failed.passed);
    assert!(failed.failures.contains(&"throughput_self_play"));
}

#[test]
fn validation_gate_evaluation_checks_g0_to_g3() {
    let metrics = ValidationGateMetrics {
        mean_decision_improvement: 0.02,
        negative_decision_fraction: 0.35,
        opponent_kl_p95: 0.08,
        opponent_kl_p95_limit: 0.10,
        hunter_overfold_reduction: 0.01,
        danger_underestimate_rate: 0.02,
        max_danger_underestimate_rate: 0.05,
        saf_advantage_over_shallow: 0.03,
    };
    assert!(evaluate_validation_gates(&metrics).passed);

    let failed = evaluate_validation_gates(&ValidationGateMetrics {
        negative_decision_fraction: 0.45,
        ..metrics
    });
    assert!(!failed.passed);
    assert!(failed.failures.contains(&"g0_negative_fraction"));
}

#[test]
fn benchmark_phase_can_advance_early_once_gates_pass() {
    let mut state = PipelineState::default();
    state.tick_gpu_hours(12.0);
    let report = GateReport {
        passed: true,
        failures: Vec::new(),
    };
    assert!(phase_advance_report(&state, Some(&report), None).passed);
}

#[test]
fn maintenance_plan_enables_mid_phase2_exit_and_rebase() {
    let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
        phase: TrainingPhase::DrdaAchSelfPlay,
        phase_progress: 0.75125,
        should_advance_phase: false,
        rebase_due: true,
        distill_due: true,
        distill_should_warn: false,
    });
    assert!(plan.should_rebase);
    assert!(plan.should_distill);
    assert!(plan.shallow_exit_enabled);
    assert!(!plan.deep_exit_enabled);
    assert!(!plan.distill_warning);
}

#[test]
fn maintenance_plan_keeps_benchmark_phase_idle() {
    let plan = maintenance_plan_from_inputs(OrchestratorPlanInputs {
        phase: TrainingPhase::BenchmarkGates,
        phase_progress: 0.0,
        should_advance_phase: false,
        rebase_due: true,
        distill_due: true,
        distill_should_warn: false,
    });
    assert!(!plan.should_rebase);
    assert!(!plan.should_distill);
    assert!(!plan.shallow_exit_enabled);
    assert!(!plan.deep_exit_enabled);
}
