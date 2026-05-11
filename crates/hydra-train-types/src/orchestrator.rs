//! Backend-independent orchestration gate DTOs and pure phase helpers.
//!
//! Tensor-specific train-step orchestration remains in `hydra-train`; this
//! module owns the scalar reports and pure decisions shared by training crates.

use crate::phase::{PipelineState, TrainingPhase};

/// Scalar benchmark gate measurements for phase-0 acceptance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkGateMetrics {
    pub afbs_on_turn_ms: f32,
    pub ct_smc_dp_ms: f32,
    pub endgame_exact_ms: f32,
    pub self_play_games_per_sec: f32,
    pub distill_kl_drift: f32,
}

/// Scalar validation metrics for rollout gates G0-G3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationGateMetrics {
    pub mean_decision_improvement: f32,
    pub negative_decision_fraction: f32,
    pub opponent_kl_p95: f32,
    pub opponent_kl_p95_limit: f32,
    pub hunter_overfold_reduction: f32,
    pub danger_underestimate_rate: f32,
    pub max_danger_underestimate_rate: f32,
    pub saf_advantage_over_shallow: f32,
}

/// Gate evaluation result with stable failure reason strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateReport {
    pub passed: bool,
    pub failures: Vec<&'static str>,
}

/// Scalar result emitted by a phase train-step wrapper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseTrainReport {
    pub phase: TrainingPhase,
    pub skipped: bool,
    pub loss: Option<f64>,
    pub effective_lr: f64,
    pub oracle_keep_prob: Option<f32>,
    pub kept_oracle_fraction: Option<f32>,
    pub exit_weight: Option<f32>,
}

/// Backend-independent inputs for maintenance planning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrchestratorPlanInputs {
    pub phase: TrainingPhase,
    pub phase_progress: f32,
    pub should_advance_phase: bool,
    pub rebase_due: bool,
    pub distill_due: bool,
    pub distill_should_warn: bool,
}

/// Scalar maintenance decisions for phase-aware orchestration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaintenancePlan {
    pub should_rebase: bool,
    pub should_distill: bool,
    pub distill_warning: bool,
    pub shallow_exit_enabled: bool,
    pub deep_exit_enabled: bool,
}

/// Evaluates phase-0 benchmark gates using archive threshold strings.
pub fn evaluate_benchmark_gates(
    metrics: &BenchmarkGateMetrics,
    max_distill_kl_drift: f32,
) -> GateReport {
    let mut failures = Vec::new();
    if metrics.afbs_on_turn_ms >= 150.0 {
        failures.push("latency_afbs_on_turn");
    }
    if metrics.ct_smc_dp_ms >= 1.0 {
        failures.push("latency_ct_smc_dp");
    }
    if metrics.endgame_exact_ms >= 100.0 {
        failures.push("latency_endgame_exact");
    }
    if metrics.self_play_games_per_sec <= 20.0 {
        failures.push("throughput_self_play");
    }
    if metrics.distill_kl_drift > max_distill_kl_drift {
        failures.push("distill_kl_drift");
    }
    GateReport {
        passed: failures.is_empty(),
        failures,
    }
}

/// Evaluates rollout validation gates G0-G3.
pub fn evaluate_validation_gates(metrics: &ValidationGateMetrics) -> GateReport {
    let mut failures = Vec::new();
    if metrics.mean_decision_improvement <= 0.0 {
        failures.push("g0_mean_decision_improvement");
    }
    if metrics.negative_decision_fraction >= 0.40 {
        failures.push("g0_negative_fraction");
    }
    if metrics.opponent_kl_p95 > metrics.opponent_kl_p95_limit {
        failures.push("g1_robustness_calibration");
    }
    if metrics.hunter_overfold_reduction <= 0.0 {
        failures.push("g2_hunter_overfold_reduction");
    }
    if metrics.danger_underestimate_rate > metrics.max_danger_underestimate_rate {
        failures.push("g2_danger_underestimate_rate");
    }
    if metrics.saf_advantage_over_shallow <= 0.0 {
        failures.push("g3_saf_amortization");
    }
    GateReport {
        passed: failures.is_empty(),
        failures,
    }
}

/// Computes whether the pipeline may advance from the current phase.
pub fn phase_advance_report(
    state: &PipelineState,
    benchmark_report: Option<&GateReport>,
    validation_report: Option<&GateReport>,
) -> GateReport {
    let mut failures = Vec::new();
    match state.phase {
        TrainingPhase::BenchmarkGates => match benchmark_report {
            Some(report) if report.passed => {}
            Some(report) => failures.extend(report.failures.iter().copied()),
            None => failures.push("missing_benchmark_report"),
        },
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
            if !state.should_advance_phase() {
                failures.push("phase_budget_incomplete");
            }
            match validation_report {
                Some(report) if report.passed => {}
                Some(report) => failures.extend(report.failures.iter().copied()),
                None => failures.push("missing_validation_report"),
            }
        }
        _ => {
            if !state.should_advance_phase() {
                failures.push("phase_budget_incomplete");
            }
        }
    }
    GateReport {
        passed: failures.is_empty(),
        failures,
    }
}

/// Applies a passing phase-advance report to the pipeline state.
pub fn maybe_advance_phase(state: &mut PipelineState, advance_report: &GateReport) -> bool {
    if advance_report.passed {
        state.advance_phase();
        true
    } else {
        false
    }
}

/// Builds scalar maintenance decisions from backend-specific status DTOs.
pub fn maintenance_plan_from_inputs(inputs: OrchestratorPlanInputs) -> MaintenancePlan {
    let shallow_exit_enabled = match inputs.phase {
        TrainingPhase::DrdaAchSelfPlay => inputs.phase_progress > 0.5,
        TrainingPhase::ExitPondering => true,
        _ => false,
    };
    let deep_exit_enabled = matches!(inputs.phase, TrainingPhase::ExitPondering);
    let should_rebase = matches!(
        inputs.phase,
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering
    ) && inputs.rebase_due;
    let should_distill = match inputs.phase {
        TrainingPhase::BenchmarkGates => false,
        TrainingPhase::BcWarmStart => inputs.should_advance_phase,
        TrainingPhase::OracleGuiding => false,
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => inputs.distill_due,
    };

    MaintenancePlan {
        should_rebase,
        should_distill,
        distill_warning: inputs.distill_should_warn,
        shallow_exit_enabled,
        deep_exit_enabled,
    }
}

#[cfg(test)]
mod tests;
