use colored::Colorize;

use hydra_train::model::HydraModelConfig;
use hydra_train_exec::presentation::{
    BcHyperparamSummaryInput, bc_hyperparam_summary as exec_bc_hyperparam_summary,
};
use hydra_train_types::config::BCTrainerConfig;

use super::artifacts::BcArtifactPaths;
use super::config::{TrainConfig, display_num_threads};
use super::progress::BannerStats;

pub(crate) use hydra_train_exec::presentation::{
    explicit_preflight_recommendation, format_advisory_line, format_progress_message,
    format_warning_line, make_bar, make_spinner, phase_label, timestamped,
};

pub(super) fn model_kind(config: &HydraModelConfig) -> &'static str {
    if config.is_learner() {
        "learner"
    } else {
        "actor"
    }
}

pub(super) fn bc_hyperparam_summary(input: BcHyperparamSummaryInput) -> String {
    exec_bc_hyperparam_summary(input)
}

pub(super) fn bc_hyperparam_summary_input(train_cfg: &BCTrainerConfig) -> BcHyperparamSummaryInput {
    BcHyperparamSummaryInput {
        lr: train_cfg.lr,
        min_learning_rate: train_cfg.min_learning_rate,
        weight_decay: train_cfg.weight_decay.into(),
        grad_clip_norm: train_cfg.grad_clip_norm.into(),
        warmup_steps: train_cfg.warmup_steps,
    }
}

pub(super) fn optimized_path_summary(config: &TrainConfig) -> String {
    let shard_input = config.bc_shards_manifest_path.is_some();
    let pinned_staging = cfg!(feature = "cuda-graph") && shard_input;
    let preallocated_tensors = pinned_staging;
    let copy_compute_overlap = if pinned_staging {
        "unproven-single-buffer"
    } else {
        "off"
    };
    format!(
        "input={} pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay={} copy_compute_overlap={}",
        if shard_input {
            "bc_shards"
        } else {
            "raw_replay"
        },
        if pinned_staging { "on" } else { "off" },
        if preallocated_tensors { "on" } else { "off" },
        hydra_train_exec::presentation::cuda_graph_replay_label(),
        copy_compute_overlap,
    )
}

pub(super) fn print_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &BannerStats,
    train_hyperparams: BcHyperparamSummaryInput,
) {
    hydra_train_exec::presentation::print_header_block("Hydra BC trainer");
    hydra_train_exec::presentation::print_banner_field(
        "Model",
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green(),
    );
    hydra_train_exec::presentation::print_banner_field("Device", device_name.green());
    hydra_train_exec::presentation::print_banner_field(
        "Dataset",
        if stats.counts_exact {
            format!(
                "{} ({} sources, {} games)",
                config.data_dir.display(),
                stats.total_sources,
                stats.total_games
            )
        } else {
            format!(
                "{} ({} sources, archive counts deferred)",
                config.data_dir.display(),
                stats.total_sources,
            )
        }
        .green(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Train",
        if stats.counts_exact {
            format!(
                "{} games | Val: {} games",
                stats.train_count, stats.val_count
            )
        } else {
            "streaming split, counts estimated while loading".to_string()
        }
        .green(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Buffer",
        format!(
            "{} samples (max {} games, archive_queue_bound={}, threads={})",
            config.buffer_samples,
            config.buffer_games,
            config.archive_queue_bound,
            display_num_threads(config.num_threads)
        )
        .yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Optimizer batch",
        format!(
            "{} ({} x {} accum)",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps
        )
        .yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Optimized path",
        optimized_path_summary(config).yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "BC hyperparams",
        bc_hyperparam_summary(train_hyperparams).yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Epochs",
        config.num_epochs.to_string().yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Schedule",
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            train_hyperparams.warmup_steps,
            config
                .max_train_steps
                .map(|steps| steps.to_string())
                .unwrap_or_else(|| "epoch-derived".to_string())
        )
        .yellow(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "Output",
        artifacts.root.display().to_string().green(),
    );
    hydra_train_exec::presentation::print_banner_field(
        "TBoard",
        if config.tensorboard {
            artifacts.tb_session_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        },
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::{bc_hyperparam_summary, model_kind, optimized_path_summary};
    use hydra_train::model::HydraModelConfig;
    use hydra_train::preflight::{
        EffectiveRuntimeConfig, ExplicitSettings, LoaderRuntimeConfig, ProbeKind, ProbeResult,
        ProbeStatus, SelectedRuntimeConfig,
    };
    use hydra_train_exec::presentation::{
        BcHyperparamSummaryInput, explicit_preflight_recommendation, explicit_preflight_summary,
        format_advisory_line, format_preflight_selection_line, format_preflight_summary_line,
        format_probe_progress_line, format_probe_results_table, format_probe_spinner_message,
        format_probe_status_line, format_progress_message, format_runtime_tuning_message,
        format_status_line, format_timed_phase_message, format_warning_line, make_bar,
        make_spinner, phase_label, preflight_phase_label, timestamped, with_utc_timestamp,
    };

    fn strip_ansi(input: &str) -> String {
        let mut output = String::with_capacity(input.len());
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\u{1b}' && matches!(chars.peek(), Some('[')) {
                chars.next();
                for next in chars.by_ref() {
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            output.push(ch);
        }
        output
    }

    fn probe_result(
        kind: ProbeKind,
        candidate_microbatch: usize,
        status: ProbeStatus,
        measured_samples_per_second: Option<f64>,
        elapsed_seconds: Option<f64>,
        detail: &str,
    ) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status,
            measured_samples_per_second,
            elapsed_seconds,
            detail: detail.to_string(),
        }
    }

    fn assert_timestamped_message(rendered: &str, expected_message: &str) {
        let stripped = strip_ansi(rendered);
        let (prefix, message) = stripped
            .split_once("] ")
            .expect("timestamped message should contain bracketed prefix");
        assert!(prefix.starts_with('['));
        assert!(prefix.contains('T'));
        assert!(prefix.ends_with('Z'));
        assert_eq!(message, expected_message);
    }

    #[test]
    fn bc_hyperparam_summary_includes_resolved_values() {
        let input = BcHyperparamSummaryInput {
            lr: 2.5e-4,
            min_learning_rate: 1e-6,
            weight_decay: 1e-5,
            grad_clip_norm: 1.0,
            warmup_steps: 1000,
        };
        assert_eq!(
            bc_hyperparam_summary(input),
            "lr=2.50e-4 min_lr=1.00e-6 wd=1.0e-5 clip=1.00 warmup_steps=1000"
        );
    }

    #[test]
    fn progress_bar_builders_report_success_and_template_errors() {
        assert!(make_bar(32, "{bar:40.cyan/blue} {pos}/{len}").is_ok());
        assert!(make_spinner("{spinner} probing").is_ok());

        let bar_err = make_bar(8, "{bar:bogus}").expect_err("invalid bar template should fail");
        assert!(bar_err.contains("failed to build progress style"));

        let spinner_err =
            make_spinner("{spinner:bogus}").expect_err("invalid spinner template should fail");
        assert!(spinner_err.contains("failed to build spinner style"));
    }

    #[test]
    fn optimized_path_summary_reports_raw_replay_defaults() {
        let mut config = crate::test_support::dummy_train_config();
        config.bc_shards_manifest_path = None;

        assert_eq!(
            optimized_path_summary(&config),
            "input=raw_replay pinned_h2d=off prealloc_gpu_tensors=off cuda_graph_replay=production_off_probe_only copy_compute_overlap=off"
        );
    }

    #[test]
    fn optimized_path_summary_reports_shard_path() {
        let mut config = crate::test_support::dummy_train_config();
        config.bc_shards_manifest_path = Some(std::path::PathBuf::from("/shards/manifest.json"));

        let summary = optimized_path_summary(&config);
        assert!(summary.contains("input=bc_shards"));
        assert!(summary.contains("cuda_graph_replay=production_off_probe_only"));
    }

    #[test]
    fn timestamp_helpers_preserve_message_after_stripping_ansi() {
        assert_timestamped_message(&with_utc_timestamp("hello".to_string()), "hello");
        assert_timestamped_message(&timestamped(42), "42");
    }

    #[test]
    fn advisory_line_renders_key_and_message() {
        let advisory = crate::advisory::RuntimeAdvisory::warning(
            "steady_state_cuda_bc_uses_loose_replay",
            "use shards for steady-state CUDA BC",
        );

        assert_timestamped_message(
            &format_advisory_line(&advisory),
            "Warning steady_state_cuda_bc_uses_loose_replay: use shards for steady-state CUDA BC",
        );
    }

    #[test]
    fn phase_and_progress_helpers_render_expected_text() {
        assert_eq!(preflight_phase_label("scan"), "preflight scan");
        assert_eq!(phase_label("epoch", 0, 1), "epoch");
        assert_eq!(phase_label("epoch", 1, 3), "epoch 2/3");
        assert_eq!(
            format_progress_message(0.12345, 0.875, "lr=2.5e-4", 48.678),
            "loss=0.1235 agree=87.50% steps/s=48.68 lr=2.5e-4"
        );
    }

    #[test]
    fn model_kind_distinguishes_actor_and_learner_configs() {
        assert_eq!(model_kind(&HydraModelConfig::actor()), "actor");
        assert_eq!(model_kind(&HydraModelConfig::learner()), "learner");
    }

    #[test]
    fn formatted_status_lines_keep_core_text_when_stripped() {
        let runtime = strip_ansi(&format_runtime_tuning_message(
            "train",
            "64".to_string(),
            0,
            0,
        ));
        assert!(runtime.contains("[preflight:runtime]"));
        assert!(runtime.contains("phase=train"));
        assert!(runtime.contains("candidate=64 option=1/1"));

        let timed = strip_ansi(&format_timed_phase_message("scan", "done", 12.345));
        assert!(timed.contains("[preflight:timing]"));
        assert!(timed.contains("phase=scan"));
        assert!(timed.contains("done elapsed=12.35s"));

        let summary = strip_ansi(&format_preflight_summary_line("Status", "ok"));
        assert!(summary.contains("Status ok"));

        let selection = strip_ansi(&format_preflight_selection_line("train_mb=64"));
        assert!(selection.contains("Preflight selected: train_mb=64"));

        let status = strip_ansi(&format_status_line("Device", "cuda:0"));
        assert!(status.contains("Device cuda:0"));

        let warning = strip_ansi(&format_warning_line("watch memory"));
        assert!(warning.contains("Warning: watch memory"));
    }

    #[test]
    fn format_probe_progress_line_covers_each_supported_phase_and_fallbacks() {
        let scan_start = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_start",
            )
            .expect("scan_start should render"),
        );
        assert!(
            scan_start.contains("[preflight:train] candidate_mb=64 phase=scan dataset=streaming")
        );

        let scan_complete_exact = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_complete sources=8 total_games=320 counts_exact=true",
            )
            .expect("scan_complete exact should render"),
        );
        assert!(scan_complete_exact.contains("sources=8 games=320"));

        let scan_complete_streaming = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=scan_complete sources=8 counts_exact=false",
            )
            .expect("scan_complete streaming should render"),
        );
        assert!(scan_complete_streaming.contains("sources=8 games=streaming"));

        let starting = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=10",
            )
            .expect("starting should render"),
        );
        assert!(starting.contains("phase=probe warmup=10 measure=?"));

        let warmup = strip_ansi(
            &format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=warmup")
                .expect("warmup should render"),
        );
        assert!(warmup.contains("phase=warmup step=?"));

        let measure = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=measure step=3",
            )
            .expect("measure should render"),
        );
        assert!(measure.contains("phase=measure step=3 throughput=0.00 samples/s"));

        let measure_start = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=measure_start",
            )
            .expect("measure_start should render"),
        );
        assert!(measure_start.contains("phase=measure_start total_steps=?"));

        let rl_selfplay = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=rl_games candidate_mb=16 phase=rl_selfplay",
            )
            .expect("rl_selfplay should render"),
        );
        assert!(
            rl_selfplay.contains("phase=rl_selfplay running cooperative self-play + learner step")
        );

        let done = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=done elapsed=1.25",
            )
            .expect("done should render"),
        );
        assert!(done.contains("phase=done throughput=0.00 samples/s elapsed=1.25s"));

        assert!(
            format_probe_progress_line("probe_progress kind=train candidate_mb=64 phase=unknown")
                .is_none()
        );
        assert!(format_probe_progress_line("probe_progress kind=train phase=measure").is_none());
    }

    #[test]
    fn format_probe_progress_line_covers_init_sub_stages() {
        let init_model = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_model",
            )
            .expect("init_model should render"),
        );
        assert!(init_model.contains("[preflight:train] candidate_mb=64 phase=init_model"));
        assert!(init_model.contains("initializing backbone + heads"));

        let init_optimizer = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_optimizer",
            )
            .expect("init_optimizer should render"),
        );
        assert!(init_optimizer.contains("phase=init_optimizer creating optimizer"));

        let init_loss = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_loss",
            )
            .expect("init_loss should render"),
        );
        assert!(init_loss.contains("phase=init_loss building loss functions"));

        let init_cuda_staging = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_cuda_staging",
            )
            .expect("init_cuda_staging should render"),
        );
        assert!(init_cuda_staging.contains("phase=init_cuda_staging"));
        assert!(init_cuda_staging.contains("allocating CUDA staging buffers"));

        let init_ready = strip_ansi(
            &format_probe_progress_line(
                "probe_progress kind=train candidate_mb=64 phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8",
            )
            .expect("init_ready should render"),
        );
        assert!(init_ready.contains("phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8"));

        let starting = format_probe_progress_line(
            "probe_progress kind=train candidate_mb=64 phase=starting warmup_steps=2 measure_steps=3",
        );
        assert!(starting.is_some());
    }

    #[test]
    fn format_probe_spinner_message_covers_init_sub_stages() {
        let model = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_model",
        )
        .expect("spinner init_model");
        assert!(model.contains("initializing model (backbone + heads)"));

        let optimizer = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_optimizer",
        )
        .expect("spinner init_optimizer");
        assert!(optimizer.contains("creating optimizer"));

        let loss = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_loss",
        )
        .expect("spinner init_loss");
        assert!(loss.contains("building loss functions"));

        let cuda = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_cuda_staging",
        )
        .expect("spinner init_cuda_staging");
        assert!(cuda.contains("CUDA staging"));

        let ready = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=init_ready model_ms=142 optimizer_ms=23 loss_ms=8",
        )
        .expect("spinner init_ready");
        assert!(ready.contains("init complete"));
        assert!(ready.contains("model=142ms"));

        let starting = format_probe_spinner_message(
            "probe_progress kind=train candidate_mb=64 phase=starting",
        )
        .expect("spinner starting backward compat");
        assert!(starting.contains("building model"));
    }

    #[test]
    fn format_probe_status_line_handles_success_oom_and_backend_error_cases() {
        let success = strip_ansi(&format_probe_status_line(&probe_result(
            ProbeKind::Validation,
            32,
            ProbeStatus::Success,
            Some(456.789),
            Some(2.345),
            "",
        )));
        assert!(success.contains(
            "[validation] candidate_mb=32 outcome=success throughput=456.79 samples/s elapsed=2.35s"
        ));

        let oom = strip_ansi(&format_probe_status_line(&probe_result(
            ProbeKind::RlGames,
            8,
            ProbeStatus::Oom,
            None,
            None,
            "",
        )));
        assert!(oom.contains(
            "[rl_games] candidate_mb=8 outcome=oom(generic) next=smaller_microbatch detail=n/a"
        ));

        let backend = strip_ansi(&format_probe_status_line(&probe_result(
            ProbeKind::RlMicrobatch,
            4,
            ProbeStatus::BackendError,
            None,
            None,
            "probe process status child=137",
        )));
        assert!(backend.contains(
            "[rl_microbatch] candidate_mb=4 outcome=backend_error(child_exit) detail=probe process status child=137"
        ));
    }

    #[test]
    fn format_probe_results_table_renders_selection_averages_and_missing_metrics() {
        let table = format_probe_results_table(
            ProbeKind::Train,
            &[
                probe_result(
                    ProbeKind::Train,
                    64,
                    ProbeStatus::Success,
                    Some(400.0),
                    Some(2.0),
                    "",
                ),
                probe_result(
                    ProbeKind::Train,
                    64,
                    ProbeStatus::Success,
                    Some(500.0),
                    Some(4.0),
                    "",
                ),
                probe_result(
                    ProbeKind::Train,
                    32,
                    ProbeStatus::DataError,
                    None,
                    None,
                    "bad archive",
                ),
            ],
            Some(64),
        );

        let lines: Vec<_> = table.lines().collect();
        assert!(lines[0].contains("kind         selected  candidate_mb"));
        assert!(lines[2].contains("train        yes       64"));
        assert!(lines[2].contains("success"));
        assert!(lines[2].contains("450.00"));
        assert!(lines[2].contains("3.00"));
        assert!(lines[3].contains("train        no        32"));
        assert!(lines[3].contains("data_error"));
        assert!(lines[3].contains("-"));
    }

    #[test]
    fn explicit_preflight_helpers_render_saved_runtime_and_recommendation() {
        let summary = explicit_preflight_summary(
            EffectiveRuntimeConfig {
                selected: SelectedRuntimeConfig {
                    train_microbatch_size: 64,
                    validation_microbatch_size: 32,
                    accum_steps: 4,
                },
                loader: LoaderRuntimeConfig {
                    num_threads: Some(6),
                    buffer_games: 128,
                    buffer_samples: 4096,
                    archive_queue_bound: 16,
                },
            },
            ExplicitSettings {
                train_microbatch_explicit: true,
                validation_microbatch_explicit: false,
            },
        );
        assert_eq!(
            summary,
            "saved train_mb=64 val_mb=32 accum_steps=4 threads=6 buffer_games=128 buffer_samples=4096 archive_queue_bound=16 explicit(train=true, val=false)"
        );
        assert_eq!(
            explicit_preflight_recommendation(),
            "using config runtime except epoch-boundary selected-runtime reuse; run train <config.yaml> --preflight to tune this machine before training"
        );
    }
}
