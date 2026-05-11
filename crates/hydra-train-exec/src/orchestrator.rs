//! Phase-aware supervised train-step orchestration below the train facade.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use hydra_model::model::HydraModel;
use hydra_train_types::config::OracleGuidingConfig;
use hydra_train_types::losses::HydraTargets;
use hydra_train_types::orchestrator::PhaseTrainReport;
use hydra_train_types::phase::{PipelineState, TrainingPhase};

use crate::bc_runtime::{
    BcExitConfig, BcTrainBatchInput, BcTrainStepContext, OracleGuidingBatchInput,
    OracleGuidingStepSchedule, bc_train_step, oracle_guiding_train_step, phase_learning_rate,
};
use crate::losses::HydraLoss;

pub use hydra_train_types::config::OracleGuidingConfig as SupervisedOracleGuidingConfig;

/// Supervised phase train-step inputs owned by the execution layer.
pub struct SupervisedPhaseTrainRequest<'a, B: Backend> {
    /// Current scalar pipeline state.
    pub state: &'a PipelineState,
    /// Observation tensor for the batch.
    pub obs: Tensor<B, 3>,
    /// Multi-head supervised targets.
    pub targets: &'a HydraTargets<B>,
    /// Loss adapter.
    pub loss_fn: &'a HydraLoss<B>,
    /// Oracle-guiding schedule.
    pub oracle_cfg: &'a OracleGuidingConfig,
    /// Current global/schedule step.
    pub step: usize,
    /// Total scheduled steps.
    pub total_steps: usize,
    /// Batch importance weight for oracle-guiding rejection.
    pub importance_weight: f32,
    /// Maximum accepted importance weight after oracle dropout completes.
    pub max_importance_weight: f32,
    /// Pre-sampled RNG values used for deterministic oracle target dropout.
    pub rng_values: &'a [f32],
}

/// Runs one supervised phase train step and emits a scalar report.
pub fn supervised_phase_train_step<B: AutodiffBackend>(
    model: HydraModel<B>,
    request: SupervisedPhaseTrainRequest<'_, B>,
    optimizer: &mut impl burn::optim::Optimizer<HydraModel<B>, B>,
) -> Result<(HydraModel<B>, PhaseTrainReport), &'static str> {
    match request.state.phase {
        TrainingPhase::BenchmarkGates => Ok((
            model,
            PhaseTrainReport {
                phase: request.state.phase,
                skipped: true,
                loss: None,
                effective_lr: 0.0,
                oracle_keep_prob: None,
                kept_oracle_fraction: None,
                exit_weight: None,
            },
        )),
        TrainingPhase::BcWarmStart => {
            let lr = phase_learning_rate(request.state.phase, request.step, request.total_steps);
            let empty_batch = empty_bc_batch(&request.obs);
            let (model, loss) = bc_train_step(
                model,
                BcTrainBatchInput {
                    obs: request.obs,
                    batch: &empty_batch,
                    targets: request.targets,
                },
                BcTrainStepContext {
                    loss_fn: request.loss_fn,
                    exit_cfg: &BcExitConfig::default(),
                    use_amp: false,
                    lr,
                },
                optimizer,
            );
            Ok((
                model,
                PhaseTrainReport {
                    phase: request.state.phase,
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
                OracleGuidingBatchInput {
                    obs: request.obs,
                    targets: request.targets,
                    loss_fn: request.loss_fn,
                    importance_weight: request.importance_weight,
                    max_importance_weight: request.max_importance_weight,
                    rng_values: request.rng_values,
                },
                OracleGuidingStepSchedule {
                    base_lr: phase_learning_rate(
                        request.state.phase,
                        request.step,
                        request.total_steps,
                    ),
                    oracle_cfg: request.oracle_cfg,
                    step: request.step,
                    total_steps: request.total_steps,
                },
                optimizer,
            );
            Ok((
                model,
                PhaseTrainReport {
                    phase: request.state.phase,
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

fn empty_bc_batch<B: Backend>(obs: &Tensor<B, 3>) -> crate::data::sample::MjaiBcBatch<B> {
    let batch_size = obs.dims()[0];
    let device = obs.device();
    crate::data::sample::MjaiBcBatch {
        actions: Tensor::zeros([batch_size], &device),
        exit_target: None,
        exit_mask: None,
    }
}
