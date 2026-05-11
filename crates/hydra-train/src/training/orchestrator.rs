//! Phase-aware training orchestration and gate evaluation.

use burn::tensor::backend::AutodiffBackend;

use crate::config::PipelineState;
use crate::model::HydraModel;
use crate::training::distill::{DistillConfig, DistillState};
use crate::training::drda::RebaseTracker;
use crate::training::live_exit::LiveExitConfig;
use crate::training::losses::HydraLoss;
use crate::training::rl::{RlBatch, RlConfig};
use hydra_search_labels::exit::ExitConfig;
pub use hydra_train_exec::orchestrator::SupervisedPhaseTrainRequest;
pub use hydra_train_exec::rl_step::RlPhaseTrainRequest;
pub use hydra_train_types::orchestrator::{
    BenchmarkGateMetrics, GateReport, MaintenancePlan, OrchestratorPlanInputs, PhaseTrainReport,
    ValidationGateMetrics, evaluate_benchmark_gates, evaluate_validation_gates,
    maintenance_plan_from_inputs, maybe_advance_phase, phase_advance_report,
};

pub fn maintenance_plan(
    state: &PipelineState,
    rebase_tracker: &RebaseTracker,
    distill_state: &DistillState,
    distill_cfg: &DistillConfig,
    elapsed_secs: u64,
    max_distill_kl_drift: f32,
) -> MaintenancePlan {
    maintenance_plan_from_inputs(OrchestratorPlanInputs {
        phase: state.phase,
        phase_progress: state.phase_progress(),
        should_advance_phase: state.should_advance_phase(),
        rebase_due: rebase_tracker.should_rebase(),
        distill_due: distill_state.should_distill(distill_cfg, elapsed_secs),
        distill_should_warn: distill_state.should_warn(max_distill_kl_drift),
    })
}

pub fn live_exit_config_from_plan(plan: &MaintenancePlan) -> LiveExitConfig {
    LiveExitConfig {
        enabled: plan.shallow_exit_enabled || plan.deep_exit_enabled,
        exit_config: ExitConfig::default_phase3(),
    }
}

pub fn supervised_phase_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    request: SupervisedPhaseTrainRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    hydra_train_exec::orchestrator::supervised_phase_train_step(model, request, optimizer)
}

pub fn rl_phase_train_step<B: AutodiffBackend>(
    state: &PipelineState,
    model: HydraModel<B>,
    batch: &RlBatch<B>,
    cfg: &RlConfig,
    loss_fn: &HydraLoss<B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    hydra_train_exec::rl_step::rl_phase_train_step(state, model, batch, cfg, loss_fn, optimizer)
}

pub fn rl_phase_train_step_with_controller<B: AutodiffBackend>(
    model: HydraModel<B>,
    request: RlPhaseTrainRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    hydra_train_exec::rl_step::rl_phase_train_step_with_controller(model, request, optimizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{INPUT_CHANNELS, OracleGuidingConfig, TrainingPhase};
    use crate::model::{HydraModelConfig, HydraModelInit};
    use crate::training::head_gates::{
        AdvancedHead, HeadActivationConfig, HeadActivationController, HeadState,
    };
    use crate::training::losses::{HydraLossConfig, HydraTargets};
    use burn::backend::Autodiff;
    use burn::optim::AdamConfig;
    use burn::prelude::*;

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
            delta_q_mask: None,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
            target_presence: None,
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
    fn rl_phase_with_controller_keeps_delta_q_off_by_default() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let mut batch = RlBatch {
            obs: Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device),
            actions: Tensor::<AB, 1, Int>::zeros([2], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.5], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -1.0], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: dummy_targets::<AB>(&device, 2),
            exit_target: None,
            exit_mask: None,
        };
        batch.targets.delta_q_target = Some(Tensor::<AB, 2>::ones([2, 46], &device));
        batch.targets.delta_q_mask = Some(Tensor::<AB, 2>::ones([2, 46], &device));
        let cfg = RlConfig::default_phase3();
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new().with_w_delta_q(0.25));
        let mut optimizer = AdamConfig::new().init();
        let state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            gpu_hours_used: 1500.0,
            ..PipelineState::default()
        };
        let mut controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let (_, report) = rl_phase_train_step_with_controller(
            model,
            RlPhaseTrainRequest {
                state: &state,
                batch: &batch,
                cfg: &cfg,
                loss_fn: &loss_fn,
                controller: Some(&mut controller),
            },
            &mut optimizer,
        )
        .expect("rl step with controller");

        assert!(!report.skipped);
        assert_eq!(controller.head_state(AdvancedHead::DeltaQ), HeadState::Off);
    }

    #[test]
    fn rl_phase_with_controller_can_enter_delta_q_warmup() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let mut batch = RlBatch {
            obs: Tensor::<AB, 3>::zeros([2, INPUT_CHANNELS, 34], &device),
            actions: Tensor::<AB, 1, Int>::zeros([2], &device),
            pi_old: Tensor::<AB, 1>::from_floats([0.5, 0.5], &device),
            advantages: Tensor::<AB, 1>::from_floats([1.0, -1.0], &device),
            base_logits: Tensor::<AB, 2>::zeros([2, 46], &device),
            targets: dummy_targets::<AB>(&device, 2),
            exit_target: None,
            exit_mask: None,
        };
        batch.targets.delta_q_target = Some(Tensor::<AB, 2>::ones([2, 46], &device));
        batch.targets.delta_q_mask = Some(Tensor::<AB, 2>::ones([2, 46], &device));
        let cfg = RlConfig::default_phase3();
        let loss_fn = HydraLoss::<AB>::new(HydraLossConfig::new().with_w_delta_q(0.25));
        let mut optimizer = AdamConfig::new().init();
        let state = PipelineState {
            phase: TrainingPhase::ExitPondering,
            gpu_hours_used: 1500.0,
            ..PipelineState::default()
        };
        let mut gate_cfg = HeadActivationConfig::default_with_params(1);
        gate_cfg.min_eval_samples = 1;
        gate_cfg.min_sparse_spp = 1.0;
        gate_cfg.warmup_steps = 1;
        let mut controller = HeadActivationController::new(gate_cfg);
        let presence =
            crate::training::head_gates::borrow_or_extract_target_presence(&batch.targets);
        controller.record_batch(&presence);
        controller.try_activate(AdvancedHead::DeltaQ);

        let (_, report) = rl_phase_train_step_with_controller(
            model,
            RlPhaseTrainRequest {
                state: &state,
                batch: &batch,
                cfg: &cfg,
                loss_fn: &loss_fn,
                controller: Some(&mut controller),
            },
            &mut optimizer,
        )
        .expect("rl step with controller");

        assert!(!report.skipped);
        assert_eq!(
            controller.head_state(AdvancedHead::DeltaQ),
            HeadState::Warmup
        );
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
            model,
            SupervisedPhaseTrainRequest {
                state: &state,
                obs,
                targets: &targets,
                loss_fn: &loss_fn,
                oracle_cfg: &OracleGuidingConfig::default(),
                step: 50,
                total_steps: 100,
                importance_weight: 1.0,
                max_importance_weight: 2.0,
                rng_values: &[0.0, 0.9],
            },
            &mut optimizer,
        )
        .expect("oracle step");

        assert!(!report.skipped);
        assert!((report.oracle_keep_prob.expect("keep") - 0.5).abs() < 1e-6);
        assert!((report.kept_oracle_fraction.expect("frac") - 0.5).abs() < 1e-6);
    }
}
