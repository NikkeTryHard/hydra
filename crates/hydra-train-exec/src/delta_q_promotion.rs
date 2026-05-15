//! Delta-Q promotion mode execution helpers.
//!
//! This module owns the promotion-mode body below the train facade. It preserves
//! the train binary's artifact schema, gate ordering, and baseline checkpoint
//! handling while keeping the compatibility binary as dispatch glue only.

use std::path::{Path, PathBuf};

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::prelude::{Backend, Module};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use colored::Colorize;
use hydra_core::arena::compute_placements;
use hydra_model::model::{HydraModel, HydraModelConfig, HydraModelInit, HydraOutput};
use hydra_selfplay::run_mixed_policy_game_scores;
use hydra_train_runtime::config::{PrecisionMode, TrainConfig};
use hydra_train_types::delta_q_promotion::{
    DeltaQArenaConfirmationRequest, DeltaQArenaReport, DeltaQPolicyTransferReport,
    DeltaQPolicyTransferSliceInputs, DeltaQPromotionRecommendation, DeltaQPromotionReport,
    DeltaQPromotionSliceInputs, collect_policy_transfer_metrics_from_slices,
    collect_promotion_metrics_from_slices,
};
use hydra_train_types::eval::{PairedArenaEvalConfig, PairedArenaEvalResult};
use hydra_train_types::losses::HydraTargets;

use crate::artifacts::{PersistedDeltaQPromotionArtifact, write_delta_q_promotion_artifact};
use crate::bootstrap::{
    TrainingBootstrap, TrainingRuntime, initialize_training_bootstrap_for_backend,
};
use crate::presentation::timestamped;
use crate::resume::checkpoint_base_from_path;
use crate::validation::{DeltaQPolicyTransferSnapshot, DeltaQPromotionSnapshot};
use crate::validation_runner::{
    ValidationContext, ValidationRuntime, run_validation_with_policy_baseline,
};

/// Collects Delta-Q promotion metrics from Burn model output tensors.
#[must_use]
pub fn collect_promotion_metrics_from_outputs<B: Backend>(
    output: &HydraOutput<B>,
    targets: &HydraTargets<B>,
    high_gap_quantile: f64,
) -> DeltaQPromotionReport {
    let Some(delta_q_target) = &targets.delta_q_target else {
        return DeltaQPromotionReport::new();
    };
    let Some(delta_q_mask) = &targets.delta_q_mask else {
        return DeltaQPromotionReport::new();
    };

    let (policy, policy_rows, policy_width) = tensor_flat_f32(output.policy_logits.clone());
    let (delta_q, delta_q_rows, delta_q_width) = tensor_flat_f32(output.delta_q.clone());
    let (target, target_rows, target_width) = tensor_flat_f32(delta_q_target.clone());
    let (mask, mask_rows, mask_width) = tensor_flat_f32(delta_q_mask.clone());
    let (legal, legal_rows, legal_width) = tensor_flat_f32(targets.legal_mask.clone());

    collect_promotion_metrics_from_slices(
        DeltaQPromotionSliceInputs {
            policy_logits: &policy,
            policy_rows,
            policy_width,
            candidate_delta_q: &delta_q,
            candidate_delta_q_rows: delta_q_rows,
            candidate_delta_q_width: delta_q_width,
            teacher_target: &target,
            teacher_target_rows: target_rows,
            teacher_target_width: target_width,
            teacher_mask: &mask,
            teacher_mask_rows: mask_rows,
            teacher_mask_width: mask_width,
            legal_mask: &legal,
            legal_mask_rows: legal_rows,
            legal_mask_width: legal_width,
        },
        high_gap_quantile,
    )
}

/// Collects Delta-Q policy-transfer metrics from candidate and baseline policy logits.
#[must_use]
pub fn collect_policy_transfer_metrics_from_policy_outputs<B: Backend>(
    candidate_policy_logits: Tensor<B, 2>,
    baseline_policy_logits: Tensor<B, 2>,
    targets: &HydraTargets<B>,
) -> DeltaQPolicyTransferReport {
    let Some(delta_q_target) = &targets.delta_q_target else {
        return DeltaQPolicyTransferReport::new();
    };
    let Some(delta_q_mask) = &targets.delta_q_mask else {
        return DeltaQPolicyTransferReport::new();
    };

    let (candidate, candidate_rows, candidate_width) = tensor_flat_f32(candidate_policy_logits);
    let (baseline, baseline_rows, baseline_width) = tensor_flat_f32(baseline_policy_logits);
    let (target, target_rows, target_width) = tensor_flat_f32(delta_q_target.clone());
    let (mask, mask_rows, mask_width) = tensor_flat_f32(delta_q_mask.clone());
    let (legal, legal_rows, legal_width) = tensor_flat_f32(targets.legal_mask.clone());

    collect_policy_transfer_metrics_from_slices(DeltaQPolicyTransferSliceInputs {
        candidate_policy_logits: &candidate,
        candidate_policy_rows: candidate_rows,
        candidate_policy_width: candidate_width,
        baseline_policy_logits: &baseline,
        baseline_policy_rows: baseline_rows,
        baseline_policy_width: baseline_width,
        teacher_target: &target,
        teacher_target_rows: target_rows,
        teacher_target_width: target_width,
        teacher_mask: &mask,
        teacher_mask_rows: mask_rows,
        teacher_mask_width: mask_width,
        legal_mask: &legal,
        legal_mask_rows: legal_rows,
        legal_mask_width: legal_width,
    })
}

fn tensor_flat_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> (Vec<f32>, usize, usize) {
    let data = tensor.to_data().convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .expect("promotion metrics require f32 tensor data")
        .to_vec();
    let dims = data.shape;
    let rows = dims.first().copied().unwrap_or(0);
    let row_width = dims.iter().skip(1).product::<usize>();
    (values, rows, row_width)
}

/// Result of paired Delta-Q arena confirmation.
#[derive(Debug, Clone)]
pub struct DeltaQArenaEvalOutcome {
    /// Paired candidate-vs-baseline arena result.
    pub paired_result: PairedArenaEvalResult,
    /// Lower bootstrap confidence bound for candidate-baseline mean placement.
    pub lower_confidence_bound_mean_placement: f32,
}

/// Converts a paired arena result into the persisted Delta-Q arena report.
#[must_use]
pub fn delta_q_arena_report_from_paired_eval(
    result: &PairedArenaEvalResult,
    lower_confidence_bound_mean_placement: f32,
) -> DeltaQArenaReport {
    DeltaQArenaReport::from_paired_eval(result, lower_confidence_bound_mean_placement)
}

/// Runs a paired candidate-vs-baseline Delta-Q arena confirmation.
pub fn run_paired_delta_q_arena_confirmation<B: Backend>(
    candidate_model: &HydraModel<B>,
    baseline_model: &HydraModel<B>,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    config: &PairedArenaEvalConfig,
    temperature: f32,
) -> DeltaQArenaEvalOutcome {
    let mut candidate_placements = Vec::with_capacity(config.min_games);
    let mut baseline_placements = Vec::with_capacity(config.min_games);

    for game_idx in 0..config.min_games {
        let challenger_seat = if config.same_seat_rotation_schedule {
            (game_idx % 4) as u8
        } else {
            0
        };
        let game_seed = if config.same_seeds {
            config.seed.wrapping_add(game_idx as u64)
        } else {
            config
                .seed
                .wrapping_add(game_idx as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        };
        let rng_seed = game_seed ^ 0xA5A5_A5A5_5A5A_5A5A;

        let baseline_seats = [
            baseline_model,
            baseline_model,
            baseline_model,
            baseline_model,
        ];
        let mut candidate_seats = baseline_seats;
        candidate_seats[challenger_seat as usize] = candidate_model;

        let candidate_scores =
            run_mixed_policy_game_scores(game_seed, temperature, rng_seed, candidate_seats, device);
        let baseline_scores = if std::ptr::eq(candidate_model, baseline_model) {
            candidate_scores
        } else {
            run_mixed_policy_game_scores(game_seed, temperature, rng_seed, baseline_seats, device)
        };

        candidate_placements.push(compute_placements(candidate_scores)[challenger_seat as usize]);
        baseline_placements.push(compute_placements(baseline_scores)[challenger_seat as usize]);
    }

    let (lower_ci, upper_ci) = paired_bootstrap_mean_placement_ci(
        &candidate_placements,
        &baseline_placements,
        config.seed,
        1024,
    );

    DeltaQArenaEvalOutcome {
        paired_result: paired_arena_result_from_placements(
            &candidate_placements,
            &baseline_placements,
            upper_ci,
        ),
        lower_confidence_bound_mean_placement: lower_ci,
    }
}

fn paired_bootstrap_mean_placement_ci(
    candidate_placements: &[u8],
    baseline_placements: &[u8],
    seed: u64,
    resamples: usize,
) -> (f32, f32) {
    let count = candidate_placements.len().min(baseline_placements.len());
    if count == 0 {
        return (0.0, 0.0);
    }

    let deltas: Vec<f32> = candidate_placements
        .iter()
        .zip(baseline_placements.iter())
        .take(count)
        .map(|(&candidate, &baseline)| candidate as f32 - baseline as f32)
        .collect();
    if deltas.len() == 1 {
        return (deltas[0], deltas[0]);
    }

    let mut rng = seed.max(1);
    let mut means = Vec::with_capacity(resamples.max(1));
    for _ in 0..resamples.max(1) {
        let mut sample_sum = 0.0f32;
        for _ in 0..deltas.len() {
            let idx = next_bootstrap_index(&mut rng, deltas.len());
            sample_sum += deltas[idx];
        }
        means.push(sample_sum / deltas.len() as f32);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((means.len() as f32 - 1.0) * 0.025).floor() as usize;
    let upper_idx = ((means.len() as f32 - 1.0) * 0.975).ceil() as usize;
    let upper_idx = upper_idx.min(means.len() - 1);
    (means[lower_idx], means[upper_idx])
}

fn next_bootstrap_index(state: &mut u64, len: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as usize) % len.max(1)
}

/// Builds a paired arena result from candidate and baseline placements.
#[must_use]
pub fn paired_arena_result_from_placements(
    candidate_placements: &[u8],
    baseline_placements: &[u8],
    upper_confidence_bound_mean_placement: f32,
) -> PairedArenaEvalResult {
    let candidate_mean_placement = compute_mean_placement(candidate_placements);
    let baseline_mean_placement = compute_mean_placement(baseline_placements);
    let candidate_stable_dan = compute_stable_dan(candidate_mean_placement);
    let baseline_stable_dan = compute_stable_dan(baseline_mean_placement);
    PairedArenaEvalResult {
        candidate_mean_placement,
        baseline_mean_placement,
        delta_mean_placement: candidate_mean_placement - baseline_mean_placement,
        candidate_stable_dan,
        baseline_stable_dan,
        delta_stable_dan: candidate_stable_dan - baseline_stable_dan,
        upper_confidence_bound_mean_placement,
        compared_games: candidate_placements.len().min(baseline_placements.len()),
    }
}

/// Computes the stable-dan approximation from mean placement.
#[must_use]
pub fn compute_stable_dan(mean_placement: f32) -> f32 {
    (10.0 - (mean_placement - 1.0) * 4.0).clamp(0.0, 12.0)
}

/// Computes mean 1-based placement, returning 2.5 for no games.
#[must_use]
pub fn compute_mean_placement(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 2.5;
    }
    placements.iter().map(|&p| p as f32 + 1.0).sum::<f32>() / placements.len() as f32
}

/// Computes the pre-arena Delta-Q promotion recommendation.
#[must_use]
pub fn pre_arena_recommendation(
    offline_gate_passed: bool,
    transfer_gate_passed: Option<bool>,
) -> DeltaQPromotionRecommendation {
    if offline_gate_passed && transfer_gate_passed.unwrap_or(true) {
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    } else {
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    }
}

/// Returns the default arena confirmation request for requires-confirmation decisions.
#[must_use]
pub fn default_arena_confirmation_request(
    recommendation: DeltaQPromotionRecommendation,
) -> Option<DeltaQArenaConfirmationRequest> {
    (recommendation == DeltaQPromotionRecommendation::RequiresArenaConfirmation)
        .then_some(Default::default())
}

/// Returns the persisted Delta-Q promotion stage string.
#[must_use]
pub const fn delta_q_promotion_stage(has_arena_report: bool) -> &'static str {
    if has_arena_report {
        "offline_transfer_and_arena_gate"
    } else {
        "offline_and_policy_transfer_gate"
    }
}

/// Formats the arena confirmation requirement summary.
#[must_use]
pub fn delta_q_arena_requirement_summary(
    request: Option<&DeltaQArenaConfirmationRequest>,
) -> String {
    request
        .map(DeltaQArenaConfirmationRequest::summary)
        .unwrap_or_else(|| "n/a".to_string())
}

/// Formats the Delta-Q offline gate status line.
#[must_use]
pub fn format_delta_q_offline_gate_message(
    samples: usize,
    snapshot: DeltaQPromotionSnapshot,
    recommendation: DeltaQPromotionRecommendation,
    arena_requirement: &str,
    artifact_path: &Path,
) -> String {
    timestamped(format!(
        "{} samples={} compared={} dq_lift={:.4} dq_regret={:.4}/{:.4} dq_win={:.2}% dq_offline_gate={} next={} arena_req='{}' artifact={}",
        "DeltaQ offline gate".bold().magenta(),
        samples,
        snapshot.compared_states,
        snapshot.mean_decision_lift,
        snapshot.candidate_mean_regret,
        snapshot.baseline_mean_regret,
        snapshot.regret_beats_baseline_rate * 100.0,
        snapshot.passed,
        recommendation,
        arena_requirement,
        artifact_path.display(),
    ))
}

/// Formats the Delta-Q policy-transfer holdout status line.
#[must_use]
pub fn format_delta_q_policy_holdout_message(snapshot: DeltaQPolicyTransferSnapshot) -> String {
    timestamped(format!(
        "{} compared={} policy_regret={:.4}/{:.4} policy_top1={:.2}%/{:.2}% policy_beats_baseline={:.2}% candidate_worse_rate={:.2}%",
        "DeltaQ policy-vs-teacher holdout".bold().blue(),
        snapshot.compared_states,
        snapshot.candidate_policy_mean_teacher_regret,
        snapshot.baseline_policy_mean_teacher_regret,
        snapshot.candidate_policy_top1_to_teacher * 100.0,
        snapshot.baseline_policy_top1_to_teacher * 100.0,
        snapshot.candidate_beats_baseline_rate * 100.0,
        snapshot.negative_transfer_fraction * 100.0,
    ))
}

/// Formats the Delta-Q policy-transfer gate status line.
#[must_use]
pub fn format_delta_q_policy_transfer_gate_message(
    passed: bool,
    next: DeltaQPromotionRecommendation,
) -> String {
    timestamped(format!(
        "{} pass={} next={}",
        "DeltaQ policy transfer gate".bold().blue(),
        passed,
        next,
    ))
}

/// Runs Delta-Q promotion mode below the train facade.
pub fn handle_delta_q_promotion_mode<B>(
    config_path: &Path,
    config: TrainConfig,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    <B as AutodiffBackend>::InnerBackend: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    if matches!(config.precision_mode, PrecisionMode::Bf16Autocast) {
        return Err(
            "precision_mode=bf16_autocast is not supported for delta_q promotion yet".to_string(),
        );
    }
    let (bootstrap, runtime, _readers) =
        initialize_training_bootstrap_for_backend::<B>(config_path, config)?;
    run_delta_q_promotion_mode_for_bootstrap::<B>(bootstrap, runtime, baseline_checkpoint)
}

fn run_delta_q_promotion_mode_for_bootstrap<B>(
    bootstrap: TrainingBootstrap<B>,
    runtime: TrainingRuntime<B>,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    <B as AutodiffBackend>::InnerBackend: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let TrainingBootstrap {
        config,
        artifacts,
        loader_config,
        manifest,
        model_config,
        device_name,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        ..
    } = bootstrap;
    let TrainingRuntime {
        model,
        mut head_controller,
        ..
    } = runtime;
    let baseline_checkpoint = baseline_checkpoint.as_ref().ok_or_else(|| {
        "delta_q promotion mode requires --delta-q-baseline-checkpoint for arena confirmation"
            .to_string()
    })?;
    let checkpoint_base = checkpoint_base_from_path(baseline_checkpoint);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let baseline_model = HydraModelConfig::learner()
        .init::<B>(&train_device)
        .load_file(&checkpoint_base, &recorder, &train_device)
        .map_err(|err| {
            format!(
                "failed to load delta_q baseline checkpoint {}: {err}",
                checkpoint_base.display()
            )
        })?;

    println!(
        "{}",
        timestamped(format!(
            "{} device={} artifacts={} model={}",
            "Hydra DeltaQ offline/transfer gate".bold().cyan(),
            device_name,
            artifacts.root.display(),
            if model_config.is_learner() {
                "learner"
            } else {
                "actor"
            },
        ))
    );

    let summary = run_validation_with_policy_baseline(
        &model,
        &baseline_model,
        ValidationContext {
            config: &config,
            loader: &crate::data_pipeline::TrainValidationLoader {
                config: &loader_config,
            },
            manifest: &manifest,
            cached_samples: None,
            device: &train_device,
            loss_fn: &valid_loss_fn,
            exit_cfg: &bc_exit_cfg,
        },
        ValidationRuntime {
            head_controller: Some(&mut head_controller),
            progress: None,
        },
    )?;

    let (Some(report), Some(result), Some(snapshot), transfer_result) = (
        summary.delta_q_promotion.as_ref(),
        summary.delta_q_promotion_result.as_ref(),
        summary.delta_q_promotion_snapshot,
        summary.delta_q_policy_transfer_result.as_ref(),
    ) else {
        return Err(
            "delta_q promotion mode requires active delta_q targets in validation batches"
                .to_string(),
        );
    };
    let pre_arena_recommendation =
        pre_arena_recommendation(result.passed, transfer_result.map(|r| r.passed));

    let arena_confirmation_request = default_arena_confirmation_request(pre_arena_recommendation);
    let arena_config = arena_confirmation_request.as_ref().map(|request| {
        PairedArenaEvalConfig::new()
            .with_min_games(request.min_games as usize)
            .with_seed(config.seed)
            .with_same_seeds(request.same_seeds)
            .with_same_seat_rotation_schedule(request.same_seat_rotation_schedule)
            .with_same_search_budget(request.same_search_budget)
            .with_same_temperature(request.same_temperature)
            .with_same_frozen_opponent_pool(request.same_frozen_opponent_pool)
    });
    let arena_eval = arena_config.as_ref().map(|arena_config| {
        run_paired_delta_q_arena_confirmation(
            &model,
            &baseline_model,
            &train_device,
            arena_config,
            config.rl.as_ref().map(|rl| rl.temperature).unwrap_or(1.0),
        )
    });
    let arena_report = arena_eval.as_ref().map(|outcome| {
        delta_q_arena_report_from_paired_eval(
            &outcome.paired_result,
            outcome.lower_confidence_bound_mean_placement,
        )
    });
    let arena_decision = arena_eval.as_ref().map(|outcome| {
        outcome.paired_result.recommendation(
            arena_config
                .as_ref()
                .expect("arena config exists when arena eval exists"),
        )
    });

    write_delta_q_promotion_artifact(
        &artifacts.delta_q_promotion_path,
        &PersistedDeltaQPromotionArtifact {
            scope: "promotion_mode",
            step_or_epoch: 0,
            recommendation: pre_arena_recommendation,
            stage: delta_q_promotion_stage(arena_report.is_some()),
            arena_confirmation: arena_confirmation_request.clone(),
            arena_decision,
            arena_report: arena_report.as_ref(),
            report,
            result,
            policy_transfer: summary.delta_q_policy_transfer.as_ref(),
            policy_transfer_result: transfer_result,
        },
    )?;

    println!(
        "{}",
        format_delta_q_offline_gate_message(
            summary.samples,
            snapshot,
            pre_arena_recommendation,
            &delta_q_arena_requirement_summary(arena_confirmation_request.as_ref()),
            &artifacts.delta_q_promotion_path,
        )
    );
    if let Some(outcome) = arena_eval.as_ref() {
        println!(
            "{}",
            timestamped(format!(
                "{} {} lower_ci={:.3}",
                "DeltaQ arena confirmation".bold().green(),
                outcome.paired_result.summary(
                    arena_config
                        .as_ref()
                        .expect("arena config exists when arena eval exists"),
                ),
                outcome.lower_confidence_bound_mean_placement,
            ))
        );
        if let Some(decision) = arena_decision {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "DeltaQ arena decision".bold().green(),
                    decision.summary(),
                ))
            );
        }
    }
    if let Some(transfer) = summary.delta_q_policy_transfer_snapshot {
        println!("{}", format_delta_q_policy_holdout_message(transfer));
    }
    if let Some(transfer_result) = transfer_result {
        println!(
            "{}",
            format_delta_q_policy_transfer_gate_message(
                transfer_result.passed,
                transfer_result.recommendation(),
            )
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests;
