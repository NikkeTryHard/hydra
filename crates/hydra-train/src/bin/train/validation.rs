use burn::module::AutodiffModule;
use burn::prelude::*;
use indicatif::ProgressBar;

use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_val_pass};
use hydra_train::data::sample::collate_samples;
use hydra_train::model::{HydraModel, HydraOutput};
use hydra_train::training::bc::{policy_agreement, target_actions_from_policy_target};
use hydra_train::training::losses::{HydraLoss, HydraTargets};

use super::config::{TrainConfig, validation_microbatch_size, validation_sample_limit};
use super::progress::{BatchStats, ScalarAverages, batch_stats_from_breakdown};
use super::resume::BestValidation;
use super::{TrainBackend, ValidBackend};

#[derive(Clone, Copy)]
pub(super) struct ValidationSummary {
    pub(super) total_loss: f64,
    pub(super) policy_loss: f64,
    pub(super) agreement: f64,
    pub(super) samples: usize,
}

pub(super) fn validation_batch_stats<B: Backend>(
    sample_count: usize,
    output: &HydraOutput<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
) -> BatchStats {
    let target_actions = target_actions_from_policy_target(targets.policy_target.clone());
    let agreement = policy_agreement(
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        target_actions,
    );
    let breakdown = loss_fn.total_loss(output, targets);
    batch_stats_from_breakdown(sample_count, agreement, &breakdown)
}

pub(super) fn is_better_validation(
    summary: ValidationSummary,
    best: Option<BestValidation>,
) -> bool {
    match best {
        None => true,
        Some(best) => {
            summary.policy_loss < best.policy_loss
                || ((summary.policy_loss - best.policy_loss).abs() <= f64::EPSILON
                    && summary.agreement > best.agreement)
        }
    }
}

pub(super) fn run_validation(
    model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    device: &<ValidBackend as Backend>::Device,
    loss_fn: &HydraLoss<ValidBackend>,
    progress: Option<&ProgressBar>,
) -> Result<ValidationSummary, String> {
    let model_valid = model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut stats = ScalarAverages::default();
    let mut total_samples = 0usize;

    for buffer_result in stream_val_pass(manifest, loader_config, progress) {
        let buffer = buffer_result.map_err(|err| format!("validation stream failed: {err}"))?;
        for chunk in buffer.chunks(validation_batch_size) {
            if let Some(limit) = validation_sample_limit
                && total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk
            };
            if capped_chunk.is_empty() {
                break;
            }
            let Some((obs, targets)) = collate_samples::<ValidBackend>(capped_chunk, false, device)
            else {
                continue;
            };
            let output = model_valid.forward(obs);
            let batch_stats =
                validation_batch_stats(capped_chunk.len(), &output, &targets, loss_fn);
            stats.record_batch(batch_stats);
            total_samples += capped_chunk.len();
        }
        if let Some(limit) = validation_sample_limit
            && total_samples >= limit
        {
            break;
        }
    }

    if total_samples == 0 {
        Ok(ValidationSummary {
            total_loss: 0.0,
            policy_loss: 0.0,
            agreement: 0.0,
            samples: 0,
        })
    } else {
        let stats = stats.finalize();
        Ok(ValidationSummary {
            total_loss: stats.total_loss,
            policy_loss: stats.loss_policy,
            agreement: stats.policy_agreement,
            samples: total_samples,
        })
    }
}
