use colored::Colorize;
use std::path::PathBuf;
use std::time::Instant;

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::prelude::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use hydra_train::eval::{PairedArenaEvalConfig, run_paired_delta_q_arena_confirmation};
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::ProbeKind;
use hydra_train::training::delta_q_promotion::{
    DeltaQArenaConfirmationRequest, DeltaQPromotionRecommendation,
    delta_q_arena_report_from_paired_eval,
};

use super::TrainBackend;
use super::advisory::{AdvisoryDeduper, AdvisoryEvent, startup_runtime_advisories};
use super::artifacts::{
    BcArtifactPaths, PersistedDeltaQPromotionArtifact, write_delta_q_promotion_artifact,
};
use super::bootstrap::TrainingReaders;
use super::bootstrap::{RlTrainingBootstrap, RlTrainingRuntime, initialize_rl_training_bootstrap};
use super::bootstrap::{TrainingBootstrap, TrainingRuntime, initialize_training_bootstrap};
use super::config::{TrainConfig, configure_threads, device_label, validate_config};
use super::epoch_runner::{EpochRunnerContext, EpochRuntimeMut, run_epoch};
use super::preflight_runtime::{run_preflight, run_probe_ladder_only, run_rl_preflight};
use super::presentation::{
    bc_hyperparam_summary_input, explicit_preflight_recommendation, explicit_preflight_summary,
    format_advisory_line, format_preflight_selection_line, format_preflight_summary_line,
    format_probe_results_table, format_status_line, format_timed_phase_message,
    format_warning_line, print_banner, print_preflight_banner, timestamped,
};
use super::probe_request::ProbeRequest;
use super::probe_summary::{best_probe_summary, format_probe_selection_summary, probe_kind_name};
use super::resume::checkpoint_base_from_path;
use super::rl_runner::run_rl_training_loop;
use super::validation::materialize_validation_samples;
use super::validation::{
    ValidationContext, ValidationRuntime, run_validation_with_policy_baseline, validation_loader,
};

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

fn run_bc_training_mode_for_backend<B>(
    bootstrap: TrainingBootstrap<B>,
    runtime: TrainingRuntime<B>,
    readers: TrainingReaders,
) -> Result<(), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
{
    let TrainingBootstrap {
        config,
        resume,
        artifacts,
        loader_config,
        manifest,
        train_cfg,
        model_config,
        device_name,
        train_device,
        current_runtime,
        microbatch_explicitness,
        session_start_global_step,
        total_steps,
        microbatch_size,
        use_amp,
        banner_stats,
        loss_fn,
        valid_loss_fn,
        bc_exit_cfg,
    } = bootstrap;
    let TrainingRuntime {
        model,
        mut optimizer,
        mut best_validation,
        mut global_step,
        run_start,
        mut last_log_step,
        mut last_log_time,
        mut tb,
        mut training_log,
        mut step_log,
        mut head_controller,
    } = runtime;
    let TrainingReaders = readers;

    print_banner(
        &model_config,
        &config,
        &artifacts,
        &device_name,
        &banner_stats,
        bc_hyperparam_summary_input(&train_cfg),
    );
    resume.print_banner_with_effective_runtime(Some(current_runtime));
    let mut advisory_deduper = AdvisoryDeduper::new();
    let startup_advisories =
        advisory_deduper.retain_new(startup_runtime_advisories(&config, microbatch_explicitness));
    for advisory in &startup_advisories {
        println!("{}", format_advisory_line(advisory));
    }
    if !startup_advisories.is_empty() {
        super::artifacts::append_advisory_event_to_writer(
            &mut step_log,
            &AdvisoryEvent::startup(&startup_advisories),
        )?;
    }
    let cached_validation_samples = if config.bc_shards_manifest_path.is_some() {
        None
    } else {
        materialize_validation_samples(&config, &validation_loader(&loader_config), &manifest)?
    };
    let mut model = Some(model);

    for epoch in resume.start_epoch..config.num_epochs {
        let outcome = run_epoch(
            EpochRunnerContext {
                epoch,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &bc_exit_cfg,
                train_device: &train_device,
                session_start_global_step,
                steps_to_skip: resume.steps_to_skip_for_epoch(epoch),
                microbatch_size,
                use_amp,
                total_steps,
                current_runtime,
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: cached_validation_samples.as_deref(),
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )?;
        if outcome.stop_after_epoch {
            break;
        }
    }

    if std::env::var_os("HYDRA_BENCHMARK_QUIET").is_none() {
        println!(
            "{}",
            timestamped(format!(
                "{} {}",
                "Finished BC training. Best validation policy CE:"
                    .bold()
                    .cyan(),
                format_best_validation_summary(best_validation.as_ref())
                    .bold()
                    .green()
            ))
        );
    }

    Ok(())
}

pub(super) fn handle_preflight_mode(
    config_path: &std::path::Path,
    config: &TrainConfig,
) -> Result<(), String> {
    let preflight_wall_start = Instant::now();
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    if config.rl.is_some() {
        let train_device = super::config::train_device(&config.device)?;
        let device_name = device_label(&config.device);
        print_preflight_banner("Hydra RL preflight", config, &device_name);
        let preflight = run_rl_preflight(config_path, config, &train_device)?;
        println!(
            "{}",
            format_rl_preflight_selection_message(
                preflight.selected_games_per_batch,
                preflight.selected_microbatch_size,
            )
        );
        print_probe_table(
            "RL preflight games table",
            ProbeKind::RlGames,
            &preflight.rl_games_probe_results,
            preflight.selected_games_per_batch,
        );
        print_probe_table(
            "RL preflight microbatch table",
            ProbeKind::RlMicrobatch,
            &preflight.rl_microbatch_probe_results,
            preflight.selected_microbatch_size,
        );
        println!(
            "{}",
            format_timed_phase_message(
                "preflight_wall_clock",
                "total elapsed including output",
                preflight_wall_start.elapsed().as_secs_f64(),
            )
        );
        return Ok(());
    }
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    let device_name = device_label(&config.device);
    print_preflight_banner("Hydra preflight", config, &device_name);
    let preflight = run_preflight(
        config_path,
        config,
        &HydraModelConfig::learner(),
        &device_name,
        &artifacts,
    )?;
    println!(
        "{}",
        format_bc_preflight_selection_message(preflight.runtime, preflight.explicit)
    );
    if let Some(benchmark) = preflight.benchmark.as_ref() {
        println!(
            "{}",
            format_preflight_selection_line(format!(
                "benchmark winner mode={:?} wall_clock_effective={:.2} samples/s train_only={:.2} train_mb={} val_mb={} loader=({}, {}, {}, {:?})",
                benchmark.metadata.mode,
                benchmark.score.wall_clock_samples_per_second,
                benchmark.score.train_only_samples_per_second,
                benchmark.runtime.train_microbatch_size,
                benchmark.runtime.validation_microbatch_size,
                benchmark.runtime.loader.archive_queue_bound,
                benchmark.runtime.loader.buffer_samples,
                benchmark.runtime.loader.buffer_games,
                benchmark.runtime.loader.num_threads,
            ))
        );
    }
    for advisory in &preflight.advisories {
        println!("{}", format_advisory_line(advisory));
    }
    print_probe_table(
        "Preflight train table",
        ProbeKind::Train,
        &preflight.train_probe_results,
        preflight.runtime.selected.train_microbatch_size,
    );
    print_probe_table(
        "Preflight validation table",
        ProbeKind::Validation,
        &preflight.validation_probe_results,
        preflight.runtime.selected.validation_microbatch_size,
    );
    println!(
        "{}",
        format_timed_phase_message(
            "preflight_wall_clock",
            "total elapsed including output",
            preflight_wall_start.elapsed().as_secs_f64(),
        )
    );
    Ok(())
}

pub(super) fn handle_probe_mode(
    config_path: &std::path::Path,
    config: &TrainConfig,
    request: ProbeRequest,
) -> Result<(), String> {
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    print_preflight_banner("Hydra probe-only", config, &device_label(&config.device));
    println!("{}", format_probe_only_status_message(request));
    let (selected, results) = run_probe_ladder_only(config_path, config, &artifacts, request)?;
    let selected_summary = best_probe_summary(&results).ok_or_else(|| {
        format!(
            "no stable {} probe result found",
            probe_kind_name(request.kind)
        )
    })?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            request.kind,
            &selected_summary,
        ))
    );
    println!(
        "{}",
        format_status_line(
            "Probe best candidate:",
            format_probe_best_candidate_detail(request.kind, selected)
        )
    );
    print_probe_table("Probe final table", request.kind, &results, selected);
    Ok(())
}

pub(super) fn handle_training_mode(
    config_path: &std::path::Path,
    config: TrainConfig,
) -> Result<(), String> {
    println!(
        "{}",
        format_warning_line(explicit_preflight_recommendation())
    );
    if let Some(rl_cfg) = config.rl.clone() {
        if matches!(
            config.precision_mode,
            crate::config::PrecisionMode::Bf16Autocast
        ) {
            return Err(
                "precision_mode=bf16_autocast is not supported for RL training yet".to_string(),
            );
        }
        let (bootstrap, runtime) = initialize_rl_training_bootstrap(config_path, config, rl_cfg)?;
        let RlTrainingBootstrap {
            config: _,
            rl_config,
            artifacts,
            model_config,
            device_name,
            ..
        } = &bootstrap;
        println!(
            "{}",
            timestamped(format!(
                "{} mode=rl phase={:?} games_per_batch={} device={} artifacts={} model={} ",
                "Hydra RL training".bold().cyan(),
                rl_config.phase,
                rl_config.games_per_batch,
                device_name,
                artifacts.root.display(),
                if model_config.is_learner() {
                    "learner"
                } else {
                    "actor"
                },
            ))
        );
        let _runtime: RlTrainingRuntime = runtime;
        return run_rl_training_loop(bootstrap, _runtime);
    }
    let (bootstrap, runtime, readers) = initialize_training_bootstrap(config_path, config)?;
    run_bc_training_mode_for_backend::<TrainBackend>(bootstrap, runtime, readers)
}

pub(super) fn handle_delta_q_promotion_mode(
    config_path: &std::path::Path,
    config: TrainConfig,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String> {
    if matches!(
        config.precision_mode,
        crate::config::PrecisionMode::Bf16Autocast
    ) {
        return Err(
            "precision_mode=bf16_autocast is not supported for delta_q promotion yet".to_string(),
        );
    }
    let (bootstrap, runtime, _readers) = initialize_training_bootstrap(config_path, config)?;
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
        .init::<super::TrainBackend>(&train_device)
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
            loader: &validation_loader(&loader_config),
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

fn format_probe_only_status_detail(request: ProbeRequest) -> String {
    format!(
        "kind={} candidate_mb={} warmup_steps={} measure_steps={}",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        request.warmup_steps,
        request.measure_steps,
    )
}

fn format_probe_only_status_message(request: ProbeRequest) -> String {
    format_status_line("Probe-only:", format_probe_only_status_detail(request))
}

fn format_rl_preflight_selection_message(
    selected_games_per_batch: usize,
    selected_microbatch_size: usize,
) -> String {
    format_preflight_summary_line(
        "Preflight:",
        format!(
            "selected rl.games_per_batch={} rl.microbatch_size={}",
            selected_games_per_batch, selected_microbatch_size,
        ),
    )
}

fn format_bc_preflight_selection_message(
    runtime: hydra_train::preflight::EffectiveRuntimeConfig,
    explicit: hydra_train::preflight::ExplicitSettings,
) -> String {
    format_preflight_summary_line("Preflight:", explicit_preflight_summary(runtime, explicit))
}

fn format_probe_best_candidate_detail(kind: ProbeKind, selected: usize) -> String {
    format!("{}={}", probe_kind_name(kind), selected)
}

fn format_best_validation_summary(
    best_validation: Option<&super::resume::BestValidation>,
) -> String {
    if let Some(best_validation) = best_validation {
        format!(
            "{:.4} (agree {:.2}%)",
            best_validation.policy_loss,
            best_validation.agreement * 100.0
        )
    } else {
        "n/a".to_string()
    }
}

fn pre_arena_recommendation(
    offline_gate_passed: bool,
    transfer_gate_passed: Option<bool>,
) -> DeltaQPromotionRecommendation {
    if offline_gate_passed && transfer_gate_passed.unwrap_or(true) {
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    } else {
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    }
}

fn default_arena_confirmation_request(
    recommendation: DeltaQPromotionRecommendation,
) -> Option<DeltaQArenaConfirmationRequest> {
    (recommendation == DeltaQPromotionRecommendation::RequiresArenaConfirmation)
        .then_some(Default::default())
}

fn delta_q_promotion_stage(has_arena_report: bool) -> &'static str {
    if has_arena_report {
        "offline_transfer_and_arena_gate"
    } else {
        "offline_and_policy_transfer_gate"
    }
}

fn delta_q_arena_requirement_summary(request: Option<&DeltaQArenaConfirmationRequest>) -> String {
    request
        .map(DeltaQArenaConfirmationRequest::summary)
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_delta_q_offline_gate_message(
    samples: usize,
    snapshot: super::validation::DeltaQPromotionSnapshot,
    recommendation: DeltaQPromotionRecommendation,
    arena_requirement: &str,
    artifact_path: &std::path::Path,
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

fn format_delta_q_policy_holdout_message(
    snapshot: super::validation::DeltaQPolicyTransferSnapshot,
) -> String {
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

fn format_delta_q_policy_transfer_gate_message(
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

fn format_probe_table_message(
    title: &str,
    kind: ProbeKind,
    results: &[hydra_train::preflight::ProbeResult],
    selected: usize,
) -> String {
    timestamped(format!(
        "{}\n{}",
        title.bold().cyan(),
        format_probe_results_table(kind, results, Some(selected))
    ))
}

fn print_probe_table(
    title: &str,
    kind: ProbeKind,
    results: &[hydra_train::preflight::ProbeResult],
    selected: usize,
) {
    println!(
        "{}",
        format_probe_table_message(title, kind, results, selected)
    );
}

#[cfg(test)]
mod tests {
    use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};
    use hydra_train::training::delta_q_promotion::DeltaQPromotionRecommendation;
    use std::path::{Path, PathBuf};

    use super::super::config::{RlTrainConfig, TrainConfig};
    use super::super::resume::BestValidation;
    use super::*;
    use crate::test_support::{dummy_train_config, unique_test_path as shared_unique_test_path};

    fn dummy_config() -> TrainConfig {
        let mut config = dummy_train_config();
        config.num_threads = Some(1);
        config
    }

    fn dummy_probe_request(kind: ProbeKind) -> ProbeRequest {
        ProbeRequest {
            kind,
            candidate_microbatch: 192,
            warmup_steps: 4,
            measure_steps: 8,
        }
    }

    fn dummy_probe_result(
        kind: ProbeKind,
        candidate_microbatch: usize,
        selected: bool,
    ) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(if selected { 512.0 } else { 384.0 }),
            elapsed_seconds: Some(if selected { 1.5 } else { 2.0 }),
            detail: String::new(),
        }
    }

    fn dummy_best_validation(policy_loss: f64, agreement: f64) -> BestValidation {
        BestValidation {
            policy_loss,
            agreement,
        }
    }

    fn unique_test_path(label: &str) -> PathBuf {
        shared_unique_test_path("hydra-modes-test", label)
    }

    #[test]
    fn format_probe_only_status_detail_is_stable() {
        assert_eq!(
            format_probe_only_status_detail(dummy_probe_request(ProbeKind::RlMicrobatch)),
            "kind=rl_microbatch candidate_mb=192 warmup_steps=4 measure_steps=8"
        );
    }

    #[test]
    fn format_probe_best_candidate_detail_uses_kind_name() {
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::Validation, 96),
            "validation=96"
        );
    }

    #[test]
    fn format_probe_table_message_supports_rl_games_rows() {
        let message = format_probe_table_message(
            "RL games probe table",
            ProbeKind::RlGames,
            &[dummy_probe_result(ProbeKind::RlGames, 24, true)],
            24,
        );

        assert!(message.contains("RL games probe table"));
        assert!(message.contains("rl_games"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("yes       24"));
    }

    #[test]
    fn format_best_validation_summary_formats_metrics_and_none_case() {
        let summary = dummy_best_validation(0.125, 0.875);
        assert_eq!(
            format_best_validation_summary(Some(&summary)),
            "0.1250 (agree 87.50%)"
        );
        assert_eq!(format_best_validation_summary(None), "n/a");
    }

    #[test]
    fn pre_arena_recommendation_requires_both_offline_and_transfer_gate() {
        assert_eq!(
            pre_arena_recommendation(true, Some(true)),
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        );
        assert_eq!(
            pre_arena_recommendation(true, None),
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        );
        assert_eq!(
            pre_arena_recommendation(true, Some(false)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            pre_arena_recommendation(false, Some(true)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
    }

    #[test]
    fn default_arena_confirmation_request_tracks_recommendation() {
        let request = default_arena_confirmation_request(
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        )
        .expect("arena confirmation request should exist");
        assert!(request.same_seeds);
        assert_eq!(request.min_games, 10_000);
        assert!(
            default_arena_confirmation_request(DeltaQPromotionRecommendation::RejectAtOfflineGate,)
                .is_none()
        );
    }

    #[test]
    fn delta_q_stage_and_requirement_summary_follow_arena_presence() {
        assert_eq!(
            delta_q_promotion_stage(true),
            "offline_transfer_and_arena_gate"
        );
        assert_eq!(
            delta_q_promotion_stage(false),
            "offline_and_policy_transfer_gate"
        );

        let request = DeltaQArenaConfirmationRequest::default();
        let summary = delta_q_arena_requirement_summary(Some(&request));
        assert!(summary.contains("same_seeds=true"));
        assert!(summary.contains("min_games=10000"));
        assert_eq!(delta_q_arena_requirement_summary(None), "n/a");
    }

    #[test]
    fn format_probe_table_message_includes_title_selection_and_rows() {
        let selected = 64;
        let message = format_probe_table_message(
            "Probe final table",
            ProbeKind::Train,
            &[
                dummy_probe_result(ProbeKind::Train, selected, true),
                dummy_probe_result(ProbeKind::Train, 48, false),
            ],
            selected,
        );

        assert!(message.contains("Probe final table"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("train        yes       64"));
        assert!(message.contains("train        no        48"));
    }

    #[test]
    fn handle_preflight_mode_returns_validation_errors_before_runtime_work() {
        let mut config = dummy_config();
        config.num_epochs = 0;

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid config should fail before preflight runtime");
        assert_eq!(err, "num_epochs must be greater than 0");
    }

    #[test]
    fn handle_preflight_mode_rl_branch_still_validates_before_device_or_runtime_work() {
        let mut config = dummy_config();
        config.num_epochs = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid config should fail before RL preflight setup");

        assert_eq!(err, "num_epochs must be greater than 0");
    }

    #[test]
    fn handle_preflight_mode_bc_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-bc-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 BC preflight data dir");
        let output_dir = unique_test_path("bf16-bc-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        config.precision_mode = crate::config::PrecisionMode::Bf16Autocast;
        let config_path =
            unique_test_path("bf16-bc-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 BC preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 BC preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("BF16 BC preflight should fall through the mode gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_returns_validation_errors_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Train),
        )
        .expect_err("invalid config should fail before probe runtime");
        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn handle_probe_mode_validates_rl_probe_requests_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("invalid config should fail before RL probe wrapper work");

        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn handle_probe_mode_bc_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-bc-probe-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 BC probe data dir");
        let output_dir = unique_test_path("bf16-bc-probe-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        config.precision_mode = crate::config::PrecisionMode::Bf16Autocast;
        let config_path = unique_test_path("bf16-bc-probe-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 BC probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 BC probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("BF16 BC probe should fall through the mode gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_rl_branch_allows_bf16_past_top_level_gate() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.precision_mode = crate::config::PrecisionMode::Bf16Autocast;
        config.device = "definitely-not-a-device".to_string();
        config.data_dir = unique_test_path("missing-rl-bf16-probe-data");
        config.output_dir = unique_test_path("missing-rl-bf16-probe-out");

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("RL probe mode should fall through the top-level gate");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_preflight_mode_rl_branch_allows_bf16_past_top_level_gate() {
        let data_dir = unique_test_path("bf16-rl-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BF16 RL preflight data dir");
        let output_dir = unique_test_path("bf16-rl-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.rl = Some(RlTrainConfig::default());
        config.precision_mode = crate::config::PrecisionMode::Bf16Autocast;
        let config_path =
            unique_test_path("bf16-rl-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BF16 RL preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BF16 RL preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("BF16 RL preflight should fall through the top-level gate");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_rejects_invalid_device_before_rl_runtime() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.device = "definitely-not-a-device".to_string();

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("invalid device should fail before rl preflight runtime");
        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    }

    #[test]
    fn handle_training_mode_returns_validation_errors_from_bootstrap() {
        let mut config = dummy_config();
        config.archive_queue_bound = 0;

        let err = handle_training_mode(Path::new("config.yaml"), config)
            .expect_err("invalid config should fail before training bootstrap work");
        assert_eq!(err, "archive_queue_bound must be greater than 0");
    }

    #[test]
    fn handle_training_mode_rl_branch_rejects_invalid_device_before_runtime_work() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.device = "definitely-not-a-device".to_string();

        let err = handle_training_mode(Path::new("config.yaml"), config)
            .expect_err("invalid RL device should fail before bootstrap runtime work");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
    }

    #[test]
    fn handle_training_mode_bc_branch_bubbles_bootstrap_errors_before_device_runtime_work() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-bc-train-data");
        config.output_dir = unique_test_path("bc-train-out");

        let err = handle_training_mode(Path::new("config.yaml"), config)
            .expect_err("missing BC data should fail while bootstrap initializes training mode");

        assert!(
            err.contains("failed to read data dir") || err.contains("failed to scan MJAI data")
        );
    }

    #[test]
    fn handle_delta_q_promotion_mode_returns_validation_errors_from_bootstrap() {
        let mut config = dummy_config();
        config.buffer_samples = 0;

        let err = handle_delta_q_promotion_mode(Path::new("config.yaml"), config, None)
            .expect_err("invalid config should fail before promotion runtime");
        assert_eq!(err, "buffer_samples must be greater than 0");
    }

    #[test]
    fn handle_delta_q_promotion_mode_requires_baseline_checkpoint_after_bootstrap() {
        let mut config = dummy_config();
        let data_dir = unique_test_path("promotion-data");
        let output_dir = unique_test_path("promotion-out");
        std::fs::create_dir_all(&data_dir).expect("create empty promotion data dir");
        std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();

        let err = handle_delta_q_promotion_mode(Path::new("config.yaml"), config, None)
            .expect_err("promotion mode should require a baseline checkpoint after bootstrap");

        assert_eq!(
            err,
            "delta_q promotion mode requires --delta-q-baseline-checkpoint for arena confirmation"
        );
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
    }

    #[test]
    fn handle_delta_q_promotion_mode_bubbles_baseline_checkpoint_load_errors() {
        let mut config = dummy_config();
        let data_dir = unique_test_path("promotion-load-error-data");
        let output_dir = unique_test_path("promotion-load-error-out");
        let baseline_checkpoint = unique_test_path("missing-baseline-checkpoint");
        std::fs::create_dir_all(&data_dir).expect("create empty promotion data dir");
        std::fs::create_dir_all(&output_dir).expect("create promotion output dir");
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();

        let err = handle_delta_q_promotion_mode(
            Path::new("config.yaml"),
            config,
            Some(baseline_checkpoint.clone()),
        )
        .expect_err("missing baseline checkpoint should fail during load");

        assert!(err.contains("failed to load delta_q baseline checkpoint"));
        assert!(err.contains(baseline_checkpoint.to_string_lossy().as_ref()));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
    }

    #[test]
    fn format_probe_table_message_preserves_selected_candidate_even_without_rows() {
        let message = format_probe_table_message("Empty probe table", ProbeKind::RlGames, &[], 256);
        assert!(message.contains("Empty probe table"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("selected"));
    }

    #[test]
    fn preflight_and_probe_status_message_helpers_render_expected_labels() {
        let probe_message = format_probe_only_status_message(dummy_probe_request(ProbeKind::Train));
        assert!(probe_message.contains("Probe-only:"));
        assert!(probe_message.contains("kind=train"));

        let rl_message = format_rl_preflight_selection_message(64, 16);
        assert!(rl_message.contains("Preflight:"));
        assert!(rl_message.contains("selected rl.games_per_batch=64 rl.microbatch_size=16"));

        let runtime = hydra_train::preflight::EffectiveRuntimeConfig {
            selected: hydra_train::preflight::SelectedRuntimeConfig {
                train_microbatch_size: 64,
                validation_microbatch_size: 32,
                accum_steps: 4,
            },
            loader: hydra_train::preflight::LoaderRuntimeConfig {
                num_threads: Some(6),
                buffer_games: 16,
                buffer_samples: 128,
                archive_queue_bound: 8,
            },
        };
        let explicit = hydra_train::preflight::ExplicitSettings {
            train_microbatch_explicit: false,
            validation_microbatch_explicit: true,
        };
        let bc_message = format_bc_preflight_selection_message(runtime, explicit);
        assert!(bc_message.contains("Preflight:"));
        assert!(bc_message.contains("saved train_mb=64 val_mb=32"));
        assert!(bc_message.contains("accum_steps=4"));
        assert!(bc_message.contains("threads=6"));
        assert!(bc_message.contains("explicit(train=false, val=true)"));
    }

    #[test]
    fn probe_mode_helpers_cover_all_probe_kinds() {
        assert_eq!(
            format_probe_only_status_detail(dummy_probe_request(ProbeKind::Train)),
            "kind=train candidate_mb=192 warmup_steps=4 measure_steps=8"
        );
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::RlGames, 512),
            "rl_games=512"
        );
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::RlMicrobatch, 32),
            "rl_microbatch=32"
        );
    }

    #[test]
    fn print_probe_table_message_shape_stays_stable_for_validation_kind() {
        let message = format_probe_table_message(
            "Validation probe table",
            ProbeKind::Validation,
            &[dummy_probe_result(ProbeKind::Validation, 32, true)],
            32,
        );

        assert!(message.contains("Validation probe table"));
        assert!(message.contains("validation"));
        assert!(message.contains("candidate_mb"));
    }

    #[test]
    fn format_best_validation_summary_rounds_and_handles_zero_agreement() {
        let summary = dummy_best_validation(1.0 / 3.0, 0.0);
        assert_eq!(
            format_best_validation_summary(Some(&summary)),
            "0.3333 (agree 0.00%)"
        );
    }

    #[test]
    fn pre_arena_and_stage_helpers_cover_all_rejecting_paths() {
        assert_eq!(
            pre_arena_recommendation(false, None),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            pre_arena_recommendation(false, Some(false)),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
        assert_eq!(
            delta_q_promotion_stage(true),
            "offline_transfer_and_arena_gate"
        );
        assert_eq!(
            delta_q_promotion_stage(false),
            "offline_and_policy_transfer_gate"
        );
    }

    #[test]
    fn delta_q_arena_requirement_summary_reports_custom_request_fields() {
        let request = DeltaQArenaConfirmationRequest {
            min_games: 256,
            same_seeds: false,
            same_seat_rotation_schedule: false,
            same_search_budget: false,
            same_temperature: false,
            same_frozen_opponent_pool: false,
        };
        let summary = delta_q_arena_requirement_summary(Some(&request));
        assert!(summary.contains("same_seeds=false"));
        assert!(summary.contains("min_games=256"));
    }

    #[test]
    fn delta_q_promotion_formatters_cover_offline_holdout_and_gate_messages() {
        let offline = format_delta_q_offline_gate_message(
            64,
            crate::validation::DeltaQPromotionSnapshot {
                compared_states: 12,
                candidate_top1_agreement: 0.75,
                candidate_mean_regret: 0.2,
                baseline_mean_regret: 0.3,
                mean_decision_lift: 0.1,
                negative_lift_fraction: 0.25,
                regret_beats_baseline_rate: 0.8,
                top1_beats_baseline_rate: 0.7,
                passed: true,
            },
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
            "same_seeds=true min_games=10000",
            Path::new("/tmp/delta_q.json"),
        );
        assert!(offline.contains("DeltaQ offline gate"));
        assert!(offline.contains("samples=64"));
        assert!(offline.contains("compared=12"));
        assert!(offline.contains("next=requires_arena_confirmation"));
        assert!(offline.contains("artifact=/tmp/delta_q.json"));

        let holdout = format_delta_q_policy_holdout_message(
            crate::validation::DeltaQPolicyTransferSnapshot {
                compared_states: 20,
                candidate_policy_top1_to_teacher: 0.6,
                baseline_policy_top1_to_teacher: 0.5,
                candidate_policy_mean_teacher_regret: 0.2,
                baseline_policy_mean_teacher_regret: 0.25,
                candidate_beats_baseline_rate: 0.7,
                negative_transfer_fraction: 0.1,
            },
        );
        assert!(holdout.contains("DeltaQ policy-vs-teacher holdout"));
        assert!(holdout.contains("compared=20"));
        assert!(holdout.contains("policy_top1=60.00%/50.00%"));

        let gate = format_delta_q_policy_transfer_gate_message(
            true,
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        );
        assert!(gate.contains("DeltaQ policy transfer gate"));
        assert!(gate.contains("pass=true"));
        assert!(gate.contains("next=requires_arena_confirmation"));
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_runtime_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-bc-data");
        config.output_dir = unique_test_path("bc-out");

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("missing dataset should fail during BC preflight runtime");

        assert!(err.contains("failed to read config config.yaml"));
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("bc-preflight-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("file-backed output path should fail BC artifact dir creation");

        assert!(err.contains("failed to create BC artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_preflight_mode_bc_branch_bubbles_no_stable_train_result() {
        let data_dir = unique_test_path("bc-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty BC preflight data dir");
        let output_dir = unique_test_path("bc-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("bc-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid BC preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid BC preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("all-failing BC preflight should bubble the no-stable train error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_bubbles_config_read_errors() {
        let mut config = dummy_config();
        config.rl = Some(RlTrainConfig::default());
        config.output_dir = unique_test_path("rl-out");

        let err = handle_preflight_mode(Path::new("config.txt"), &config)
            .expect_err("invalid config extension should fail during RL preflight runtime");

        assert!(err.contains("failed to read config config.txt"));
    }

    #[test]
    fn handle_preflight_mode_rl_branch_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("rl-preflight-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();
        config.rl = Some(RlTrainConfig::default());

        let err = handle_preflight_mode(Path::new("config.yaml"), &config)
            .expect_err("file-backed output path should fail RL artifact dir creation");

        assert!(err.contains("failed to create RL artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_preflight_mode_rl_branch_rejects_invalid_device_before_slow_rl_preflight_work() {
        let data_dir = unique_test_path("rl-preflight-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty RL preflight data dir");
        let output_dir = unique_test_path("rl-preflight-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = false;
        config.preflight.required_successes = 1;
        config.rl = Some(RlTrainConfig::default());
        let config_path = unique_test_path("rl-preflight-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid RL preflight config");
        std::fs::write(&config_path, config_yaml).expect("write valid RL preflight config");

        let err = handle_preflight_mode(&config_path, &config)
            .expect_err("invalid RL device should fail before expensive RL preflight ladder work");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_probe_ladder_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-probe-data");
        config.output_dir = unique_test_path("probe-out");

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Validation),
        )
        .expect_err("missing dataset should fail during probe ladder setup");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_probe_mode_bubbles_rl_probe_ladder_scan_errors() {
        let mut config = dummy_config();
        config.data_dir = unique_test_path("missing-rl-probe-data");
        config.output_dir = unique_test_path("rl-probe-out");
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlMicrobatch),
        )
        .expect_err("missing dataset should fail during RL probe ladder setup");

        assert!(err.starts_with("failed to scan preflight data from "));
        assert!(err.contains(config.data_dir.to_string_lossy().as_ref()));
    }

    #[test]
    fn handle_probe_mode_bubbles_artifact_dir_creation_error() {
        let output_path = unique_test_path("probe-artifact-file");
        std::fs::write(&output_path, "not a directory").expect("write artifact blocker file");
        let mut config = dummy_config();
        config.output_dir = output_path.clone();

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::Train),
        )
        .expect_err("file-backed output path should fail probe artifact dir creation");

        assert!(err.contains("failed to create BC artifact dir"));
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_no_stable_result_when_ladder_returns_only_failures() {
        let data_dir = unique_test_path("probe-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty probe data dir");
        let output_dir = unique_test_path("probe-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("probe-no-stable-config").with_extension("yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 32,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("all-failing probe ladder should bubble the no-stable-result error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn handle_probe_mode_bubbles_no_stable_train_result_when_ladder_returns_only_failures() {
        let data_dir = unique_test_path("probe-train-no-stable-data");
        std::fs::create_dir_all(&data_dir).expect("create empty train probe data dir");
        let output_dir = unique_test_path("probe-train-no-stable-out");
        let mut config = dummy_config();
        config.data_dir = data_dir.clone();
        config.output_dir = output_dir.clone();
        config.device = "definitely-not-a-device".to_string();
        config.preflight.allow_override_explicit_microbatch = true;
        config.preflight.required_successes = 1;
        let config_path = unique_test_path("probe-train-no-stable-config").with_extension("yaml");
        let config_yaml =
            serde_yaml::to_string(&config).expect("serialize valid train probe config");
        std::fs::write(&config_path, config_yaml).expect("write valid train probe config");

        let err = handle_probe_mode(
            &config_path,
            &config,
            ProbeRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 64,
                warmup_steps: 1,
                measure_steps: 1,
            },
        )
        .expect_err("all-failing train probe ladder should bubble the no-stable-result error");

        assert!(err.contains("unsupported HYDRA_TRAIN_DEVICE=definitely-not-a-device"));
        let _ = std::fs::remove_dir_all(data_dir);
        let _ = std::fs::remove_dir_all(output_dir);
        let _ = std::fs::remove_file(config_path);
    }

    #[test]
    fn format_probe_only_and_rl_selection_helpers_render_exact_details() {
        let probe_message =
            format_probe_only_status_message(dummy_probe_request(ProbeKind::Validation));
        assert!(probe_message.contains("Probe-only:"));
        assert!(
            probe_message
                .contains("kind=validation candidate_mb=192 warmup_steps=4 measure_steps=8")
        );

        let rl_message = format_rl_preflight_selection_message(32, 8);
        assert!(rl_message.contains("Preflight:"));
        assert!(rl_message.contains("selected rl.games_per_batch=32 rl.microbatch_size=8"));
    }

    #[test]
    fn format_probe_table_message_supports_rl_microbatch_rows() {
        let message = format_probe_table_message(
            "RL microbatch probe table",
            ProbeKind::RlMicrobatch,
            &[dummy_probe_result(ProbeKind::RlMicrobatch, 16, true)],
            16,
        );

        assert!(message.contains("RL microbatch probe table"));
        assert!(message.contains("rl_microbatch"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("yes       16"));
    }

    #[test]
    fn delta_q_policy_transfer_gate_and_offline_messages_cover_reject_paths() {
        let gate = format_delta_q_policy_transfer_gate_message(
            false,
            DeltaQPromotionRecommendation::RejectAtOfflineGate,
        );
        assert!(gate.contains("pass=false"));
        assert!(gate.contains("next=reject_at_offline_gate"));

        let offline = format_delta_q_offline_gate_message(
            8,
            crate::validation::DeltaQPromotionSnapshot {
                compared_states: 4,
                candidate_top1_agreement: 0.25,
                candidate_mean_regret: 0.5,
                baseline_mean_regret: 0.4,
                mean_decision_lift: -0.1,
                negative_lift_fraction: 0.75,
                regret_beats_baseline_rate: 0.25,
                top1_beats_baseline_rate: 0.1,
                passed: false,
            },
            DeltaQPromotionRecommendation::RejectAtOfflineGate,
            "n/a",
            Path::new("/tmp/reject.json"),
        );
        assert!(offline.contains("dq_offline_gate=false"));
        assert!(offline.contains("next=reject_at_offline_gate"));
        assert!(offline.contains("artifact=/tmp/reject.json"));
    }

    #[test]
    fn default_arena_confirmation_request_returns_default_request_for_requires_confirmation() {
        let request = default_arena_confirmation_request(
            DeltaQPromotionRecommendation::RequiresArenaConfirmation,
        )
        .expect("requires-confirmation should create a default arena request");

        assert_eq!(
            request.min_games,
            DeltaQArenaConfirmationRequest::default().min_games
        );
        assert_eq!(
            request.same_seeds,
            DeltaQArenaConfirmationRequest::default().same_seeds
        );
    }

    #[test]
    fn format_rl_and_bc_preflight_selection_messages_cover_small_values() {
        let rl_message = format_rl_preflight_selection_message(1, 2);
        assert!(rl_message.contains("selected rl.games_per_batch=1 rl.microbatch_size=2"));

        let runtime = hydra_train::preflight::EffectiveRuntimeConfig {
            selected: hydra_train::preflight::SelectedRuntimeConfig {
                train_microbatch_size: 8,
                validation_microbatch_size: 4,
                accum_steps: 1,
            },
            loader: hydra_train::preflight::LoaderRuntimeConfig {
                num_threads: None,
                buffer_games: 2,
                buffer_samples: 16,
                archive_queue_bound: 1,
            },
        };
        let explicit = hydra_train::preflight::ExplicitSettings {
            train_microbatch_explicit: true,
            validation_microbatch_explicit: false,
        };

        let bc_message = format_bc_preflight_selection_message(runtime, explicit);
        assert!(bc_message.contains("saved train_mb=8 val_mb=4"));
        assert!(bc_message.contains("accum_steps=1"));
        assert!(bc_message.contains("explicit(train=true, val=false)"));
    }

    #[test]
    fn format_probe_best_candidate_detail_supports_rl_games_kind() {
        let detail = format_probe_best_candidate_detail(ProbeKind::RlGames, 8);

        assert_eq!(detail, "rl_games=8");
    }

    #[test]
    fn handle_probe_mode_rl_games_request_still_validates_before_probe_runtime() {
        let mut config = dummy_config();
        config.batch_size = 0;
        config.rl = Some(RlTrainConfig::default());

        let err = handle_probe_mode(
            Path::new("config.yaml"),
            &config,
            dummy_probe_request(ProbeKind::RlGames),
        )
        .expect_err("invalid config should fail before RL games probe wrapper work");

        assert_eq!(err, "batch_size must be greater than 0");
    }

    #[test]
    fn format_probe_only_status_message_supports_rl_games_kind() {
        let message = format_probe_only_status_message(dummy_probe_request(ProbeKind::RlGames));

        assert!(message.contains("Probe-only:"));
        assert!(message.contains("kind=rl_games candidate_mb=192 warmup_steps=4 measure_steps=8"));
    }

    #[test]
    fn format_probe_best_candidate_detail_supports_train_kind() {
        assert_eq!(
            format_probe_best_candidate_detail(ProbeKind::Train, 48),
            "train=48"
        );
    }
}
