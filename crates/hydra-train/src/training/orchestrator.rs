//! Phase-aware training orchestration and gate evaluation.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::config::{OracleGuidingConfig, PipelineState, TrainingPhase};
use crate::model::HydraModel;
use crate::training::bc::{bc_train_step, oracle_guiding_train_step, phase_learning_rate};
use crate::training::distill::{DistillConfig, DistillState};
use crate::training::drda::RebaseTracker;
use crate::training::exit::ExitConfig;
use crate::training::live_exit::LiveExitConfig;
use crate::training::losses::{HydraLoss, HydraTargets};
use crate::training::rl::{RlBatch, RlConfig, rl_step_with_phase_progress};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkGateMetrics {
    pub afbs_on_turn_ms: f32,
    pub ct_smc_dp_ms: f32,
    pub endgame_exact_ms: f32,
    pub self_play_games_per_sec: f32,
    pub distill_kl_drift: f32,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateReport {
    pub passed: bool,
    pub failures: Vec<&'static str>,
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaintenancePlan {
    pub should_rebase: bool,
    pub should_distill: bool,
    pub distill_warning: bool,
    pub shallow_exit_enabled: bool,
    pub deep_exit_enabled: bool,
}

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

pub fn maybe_advance_phase(state: &mut PipelineState, advance_report: &GateReport) -> bool {
    if advance_report.passed {
        state.advance_phase();
        true
    } else {
        false
    }
}

pub fn maintenance_plan(
    state: &PipelineState,
    rebase_tracker: &RebaseTracker,
    distill_state: &DistillState,
    distill_cfg: &DistillConfig,
    elapsed_secs: u64,
    max_distill_kl_drift: f32,
) -> MaintenancePlan {
    let phase_progress = state.phase_progress();
    let shallow_exit_enabled = match state.phase {
        TrainingPhase::DrdaAchSelfPlay => phase_progress > 0.5,
        TrainingPhase::ExitPondering => true,
        _ => false,
    };
    let deep_exit_enabled = matches!(state.phase, TrainingPhase::ExitPondering);
    let should_rebase = matches!(
        state.phase,
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering
    ) && rebase_tracker.should_rebase();
    let should_distill = match state.phase {
        TrainingPhase::BenchmarkGates => false,
        TrainingPhase::BcWarmStart => state.should_advance_phase(),
        TrainingPhase::OracleGuiding => false,
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
            distill_state.should_distill(distill_cfg, elapsed_secs)
        }
    };

    MaintenancePlan {
        should_rebase,
        should_distill,
        distill_warning: distill_state.should_warn(max_distill_kl_drift),
        shallow_exit_enabled,
        deep_exit_enabled,
    }
}

/// Builds the live ExIt producer config from the current maintenance plan.
///
/// Returns a [`LiveExitConfig`] with `enabled` set according to the plan's
/// exit flags. The producer remains default-off when neither shallow nor
/// deep exit is active.
pub fn live_exit_config_from_plan(plan: &MaintenancePlan) -> LiveExitConfig {
    LiveExitConfig {
        enabled: plan.shallow_exit_enabled || plan.deep_exit_enabled,
        exit_config: ExitConfig::default_phase3(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn supervised_phase_train_step<B: AutodiffBackend>(
    state: &PipelineState,
    model: HydraModel<B>,
    obs: Tensor<B, 3>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
    oracle_cfg: &OracleGuidingConfig,
    step: usize,
    total_steps: usize,
    importance_weight: f32,
    max_importance_weight: f32,
    rng_values: &[f32],
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    match state.phase {
        TrainingPhase::BenchmarkGates => Ok((
            model,
            PhaseTrainReport {
                phase: state.phase,
                skipped: true,
                loss: None,
                effective_lr: 0.0,
                oracle_keep_prob: None,
                kept_oracle_fraction: None,
                exit_weight: None,
            },
        )),
        TrainingPhase::BcWarmStart => {
            let lr = phase_learning_rate(state.phase, step, total_steps);
            let (model, loss) = bc_train_step(model, obs, targets, loss_fn, lr, optimizer);
            Ok((
                model,
                PhaseTrainReport {
                    phase: state.phase,
                    skipped: false,
                    loss: Some(loss),
                    effective_lr: lr,
                    oracle_keep_prob: None,
                    kept_oracle_fraction: None,
                    exit_weight: None,
                },
            ))
        }
        TrainingPhase::OracleGuiding => {
            let (model, stats) = oracle_guiding_train_step(
                model,
                obs,
                targets,
                loss_fn,
                phase_learning_rate(state.phase, step, total_steps),
                oracle_cfg,
                step,
                total_steps,
                importance_weight,
                max_importance_weight,
                rng_values,
                optimizer,
            );
            Ok((
                model,
                PhaseTrainReport {
                    phase: state.phase,
                    skipped: stats.skipped,
                    loss: stats.loss,
                    effective_lr: stats.effective_lr,
                    oracle_keep_prob: Some(stats.oracle_keep_prob),
                    kept_oracle_fraction: Some(stats.kept_oracle_fraction),
                    exit_weight: None,
                },
            ))
        }
        _ => Err("supervised_phase_train_step only supports benchmark/bc/oracle phases"),
    }
}

pub fn rl_phase_train_step<B: AutodiffBackend>(
    state: &PipelineState,
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    match state.phase {
        TrainingPhase::DrdaAchSelfPlay | TrainingPhase::ExitPondering => {
            let exit_phase = state.phase.exit_schedule_phase();
            let progress = state.phase_progress();
            let exit_weight = cfg.effective_exit_weight(exit_phase, progress);
            let (model, loss) = rl_step_with_phase_progress(
                model, batch, cfg, exit_phase, progress, loss_fn, optimizer,
            );
            Ok((
                model,
                PhaseTrainReport {
                    phase: state.phase,
                    skipped: false,
                    loss: Some(loss),
                    effective_lr: cfg.lr,
                    oracle_keep_prob: None,
                    kept_oracle_fraction: None,
                    exit_weight: Some(exit_weight),
                },
            ))
        }
        _ => Err("rl_phase_train_step only supports self-play phases"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::INPUT_CHANNELS;
    use crate::model::HydraModelConfig;
    use crate::training::losses::HydraLossConfig;
    use burn::backend::Autodiff;
    use burn::optim::AdamConfig;

    type AB = Autodiff<burn::backend::NdArray<f32>>;

    fn dummy_targets<B: Backend>(device: &B::Device, batch: usize) -> HydraTargets<B> {
        HydraTargets {
            policy_target: Tensor::ones([batch, 46], device) / 46.0,
            legal_mask: Tensor::ones([batch, 46], device),
            value_target: Tensor::zeros([batch], device),
            grp_target: Tensor::ones([batch, 24], device) / 24.0,
            tenpai_target: Tensor::ones([batch, 3], device) / 3.0,
            danger_target: Tensor::zeros([batch, 3, 34], device),
            danger_mask: Tensor::ones([batch, 3, 34], device),
            opp_next_target: Tensor::ones([batch, 3, 34], device) / 34.0,
            score_pdf_target: Tensor::ones([batch, 64], device) / 64.0,
            score_cdf_target: Tensor::zeros([batch, 64], device),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: None,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
        }
    }

    #[test]
    fn benchmark_gate_evaluation_matches_hydra_final_thresholds() {
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
        let state = PipelineState {
            phase: TrainingPhase::DrdaAchSelfPlay,
            gpu_hours_used: 1001.0,
            ..PipelineState::default()
        };
        let mut rebase = RebaseTracker::default_phase2();
        rebase.tick(40.0);
        let distill = DistillState {
            last_kl_drift: 0.03,
            ..DistillState::default()
        };
        let cfg = DistillConfig::fast_distill();

        let plan = maintenance_plan(&state, &rebase, &distill, &cfg, 30, 0.05);
        assert!(plan.should_rebase);
        assert!(plan.should_distill);
        assert!(plan.shallow_exit_enabled);
        assert!(!plan.deep_exit_enabled);
        assert!(!plan.distill_warning);
    }

    #[test]
    fn maintenance_plan_keeps_benchmark_phase_idle() {
        let state = PipelineState::default();
        let rebase = RebaseTracker::default_phase2();
        let distill = DistillState::default();
        let cfg = DistillConfig::fast_distill();

        let plan = maintenance_plan(&state, &rebase, &distill, &cfg, 120, 0.05);
        assert!(!plan.should_rebase);
        assert!(!plan.should_distill);
        assert!(!plan.shallow_exit_enabled);
        assert!(!plan.deep_exit_enabled);
    }

    #[test]
    fn live_exit_config_from_plan_disabled_when_no_exit() {
        let plan = MaintenancePlan {
            should_rebase: false,
            should_distill: false,
            distill_warning: false,
            shallow_exit_enabled: false,
            deep_exit_enabled: false,
        };
        let cfg = live_exit_config_from_plan(&plan);
        assert!(!cfg.enabled);
    }

    #[test]
    fn live_exit_config_from_plan_enabled_on_shallow_exit() {
        let plan = MaintenancePlan {
            should_rebase: false,
            should_distill: false,
            distill_warning: false,
            shallow_exit_enabled: true,
            deep_exit_enabled: false,
        };
        let cfg = live_exit_config_from_plan(&plan);
        assert!(cfg.enabled);
    }

    #[test]
    fn live_exit_config_from_plan_enabled_on_deep_exit() {
        let plan = MaintenancePlan {
            should_rebase: false,
            should_distill: false,
            distill_warning: false,
            shallow_exit_enabled: false,
            deep_exit_enabled: true,
        };
        let cfg = live_exit_config_from_plan(&plan);
        assert!(cfg.enabled);
    }

    #[test]
    fn rl_phase_uses_phase_local_progress_for_exit_ramp() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let batch = RlBatch {
            obs: Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device),
            actions: Tensor::<AB, 1, Int>::zeros([2], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.5], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -1.0], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: dummy_targets::<AB>(&device, 2),
            exit_target: Some(Tensor::<AB, 2>::ones([2, 46], &device) / 46.0),
            exit_mask: Some(Tensor::<AB, 2>::ones([2, 46], &device)),
        };
        let cfg = RlConfig::default_phase2();
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new());
        let mut optimizer = AdamConfig::new().init();
        let state = PipelineState {
            phase: TrainingPhase::DrdaAchSelfPlay,
            gpu_hours_used: 1000.0,
            ..PipelineState::default()
        };

        let (_, report) =
            rl_phase_train_step(&state, model, &batch, &cfg, &loss_fn, &mut optimizer)
                .expect("rl step");
        assert!((report.exit_weight.expect("exit weight") - 0.25).abs() < 1e-6);
    }

    #[test]
    fn supervised_phase_routes_oracle_guiding() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let obs = Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device);
        let mut targets = dummy_targets::<AB>(&device, 2);
        targets.oracle_target = Some(Tensor::<AB, 2>::ones([2, 4], &device));
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new().with_w_oracle_critic(1.0));
        let mut optimizer = AdamConfig::new().init();
        let state = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: 300.0,
            ..PipelineState::default()
        };

        let (_, report) = supervised_phase_train_step(
            &state,
            model,
            obs,
            &targets,
            &loss_fn,
            &mut optimizer,
            &OracleGuidingConfig::default(),
            50,
            100,
            1.0,
            2.0,
            &[0.0, 0.9],
        )
        .expect("oracle step");

        assert!(!report.skipped);
        assert!((report.oracle_keep_prob.expect("keep") - 0.5).abs() < 1e-6);
        assert!((report.kept_oracle_fraction.expect("frac") - 0.5).abs() < 1e-6);
    }
}
