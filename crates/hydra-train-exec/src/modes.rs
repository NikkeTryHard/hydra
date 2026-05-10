//! Train binary mode dispatch facade.
//!
//! This module owns the CLI mode selection order for the train executable without
//! depending on the compatibility `hydra-train` crate. The binary supplies the
//! concrete handlers while execution internals continue moving into this crate.

use std::path::{Path, PathBuf};
use std::time::Instant;

use colored::Colorize;
use hydra_model::model::HydraModelConfig;
use hydra_train_runtime::config::{
    TrainCli, TrainConfig, configure_threads, device_label, display_num_threads, train_device,
    validate_config,
};
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, ProbeKind, ProbeResult,
};
use hydra_train_runtime::probe_request::{ProbeRequest, probe_request_from_cli};

use crate::artifacts::BcArtifactPaths;
use crate::preflight_runtime::{run_preflight, run_rl_preflight};
use crate::presentation::{
    explicit_preflight_summary, format_advisory_line, format_preflight_selection_line,
    format_preflight_summary_line, format_probe_results_table, format_timed_phase_message,
    print_banner_field, print_header_block, timestamped,
};

/// Mode handlers supplied by the train binary during the cutover.
///
/// These callbacks are the temporary compatibility seam: dispatch order and CLI
/// interpretation live here, while individual mode bodies can keep moving from
/// bin-private modules into `hydra-train-exec` independently.
pub trait TrainModeHandlers {
    /// Runs explicit preflight mode.
    fn handle_preflight_mode(
        &mut self,
        config_path: &Path,
        config: &TrainConfig,
    ) -> Result<(), String>;

    /// Runs explicit probe-only mode.
    fn handle_probe_mode(
        &mut self,
        config_path: &Path,
        config: &TrainConfig,
        request: ProbeRequest,
    ) -> Result<(), String>;

    /// Runs Delta-Q promotion mode.
    fn handle_delta_q_promotion_mode(
        &mut self,
        config_path: &Path,
        config: TrainConfig,
        baseline_checkpoint: Option<PathBuf>,
    ) -> Result<(), String>;

    /// Runs the default training mode.
    fn handle_training_mode(
        &mut self,
        config_path: &Path,
        config: TrainConfig,
    ) -> Result<(), String>;
}

/// Runs explicit preflight mode for BC or RL training.
pub fn handle_preflight_mode(config_path: &Path, config: &TrainConfig) -> Result<(), String> {
    let preflight_wall_start = Instant::now();
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    if config.rl.is_some() {
        let train_device = train_device(&config.device)?;
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

/// Prints the preflight banner shared by BC, RL, and probe-only execution.
pub fn print_preflight_banner(title: &str, config: &TrainConfig, device_name: &str) {
    print_header_block(title);
    print_banner_field("Device", device_name.green());
    print_banner_field("Dataset", config.data_dir.display().to_string().green());
    print_banner_field(
        "Optimizer batch",
        format!("{} samples", config.batch_size).yellow(),
    );
    print_banner_field(
        "Runtime defaults",
        format!(
            "train_mb={} val_mb={} threads={} buffer_games={} buffer_samples={} archive_queue_bound={}",
            config.microbatch_size.unwrap_or(config.batch_size),
            config
                .validation_microbatch_size
                .unwrap_or(config.microbatch_size.unwrap_or(config.batch_size)),
            display_num_threads(config.num_threads),
            config.buffer_games,
            config.buffer_samples,
            config.archive_queue_bound,
        )
        .yellow(),
    );
    println!();
}

/// Formats the RL preflight selected runtime line.
pub fn format_rl_preflight_selection_message(
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

/// Formats the BC preflight selected runtime line.
pub fn format_bc_preflight_selection_message(
    runtime: EffectiveRuntimeConfig,
    explicit: ExplicitSettings,
) -> String {
    format_preflight_summary_line("Preflight:", explicit_preflight_summary(runtime, explicit))
}

/// Formats a preflight/probe table with title and selected candidate marker.
pub fn format_probe_table_message(
    title: &str,
    kind: ProbeKind,
    results: &[ProbeResult],
    selected: usize,
) -> String {
    timestamped(format!(
        "{}\n{}",
        title.bold().cyan(),
        format_probe_results_table(kind, results, Some(selected))
    ))
}

fn print_probe_table(title: &str, kind: ProbeKind, results: &[ProbeResult], selected: usize) {
    println!(
        "{}",
        format_probe_table_message(title, kind, results, selected)
    );
}

/// Dispatches the parsed train CLI into the selected execution mode.
///
/// The order preserves the previous train binary behavior:
/// preflight, Delta-Q promotion, probe-only, then default training. Probe-only
/// request defaults are resolved against the already-loaded config here so the
/// binary no longer owns mode selection semantics.
pub fn run_train_modes<H>(
    cli: TrainCli,
    config: TrainConfig,
    handlers: &mut H,
) -> Result<(), String>
where
    H: TrainModeHandlers,
{
    if cli.preflight {
        return handlers.handle_preflight_mode(&cli.config_path, &config);
    }
    if cli.delta_q_promotion {
        return handlers.handle_delta_q_promotion_mode(
            &cli.config_path,
            config,
            cli.delta_q_baseline_checkpoint,
        );
    }
    if let Some(request) = probe_request_from_cli(&config, cli.probe_only)? {
        return handlers.handle_probe_mode(&cli.config_path, &config, request);
    }
    handlers.handle_training_mode(&cli.config_path, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_runtime::config::{
        BcHyperparamConfig, ProbeCliRequest, TrainConfig, ValidationGateConfig,
    };
    use hydra_train_runtime::preflight::{
        EffectiveRuntimeConfig, ExplicitSettings, LoaderRuntimeConfig, PreflightConfig, ProbeKind,
        ProbeResult, ProbeStatus, SelectedRuntimeConfig,
    };

    #[derive(Default)]
    struct RecordingHandlers {
        calls: Vec<String>,
    }

    impl TrainModeHandlers for RecordingHandlers {
        fn handle_preflight_mode(
            &mut self,
            config_path: &Path,
            _config: &TrainConfig,
        ) -> Result<(), String> {
            self.calls
                .push(format!("preflight:{}", config_path.display()));
            Ok(())
        }

        fn handle_probe_mode(
            &mut self,
            config_path: &Path,
            _config: &TrainConfig,
            request: ProbeRequest,
        ) -> Result<(), String> {
            self.calls.push(format!(
                "probe:{}:{}:{}:{}",
                config_path.display(),
                request.kind as u8,
                request.candidate_microbatch,
                request.warmup_steps,
            ));
            Ok(())
        }

        fn handle_delta_q_promotion_mode(
            &mut self,
            config_path: &Path,
            _config: TrainConfig,
            baseline_checkpoint: Option<PathBuf>,
        ) -> Result<(), String> {
            self.calls.push(format!(
                "delta_q:{}:{}",
                config_path.display(),
                baseline_checkpoint
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
            ));
            Ok(())
        }

        fn handle_training_mode(
            &mut self,
            config_path: &Path,
            _config: TrainConfig,
        ) -> Result<(), String> {
            self.calls.push(format!("train:{}", config_path.display()));
            Ok(())
        }
    }

    fn cli() -> TrainCli {
        TrainCli {
            config_path: PathBuf::from("config.yaml"),
            preflight: false,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: None,
        }
    }

    fn config() -> TrainConfig {
        TrainConfig {
            data_dir: std::env::temp_dir().join("hydra-test-data"),
            output_dir: std::env::temp_dir().join("hydra-test-out"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_data_core::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            validation_gates: ValidationGateConfig::default(),
            rl: None,
            bc: BcHyperparamConfig::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: None,
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
            precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
        }
    }

    fn probe_result(kind: ProbeKind, candidate_microbatch: usize, selected: bool) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(if selected { 512.0 } else { 384.0 }),
            elapsed_seconds: Some(if selected { 1.5 } else { 2.0 }),
            detail: String::new(),
        }
    }

    #[test]
    fn formats_rl_and_bc_preflight_selection_messages() {
        let rl_message = format_rl_preflight_selection_message(32, 8);
        assert!(rl_message.contains("Preflight:"));
        assert!(rl_message.contains("selected rl.games_per_batch=32 rl.microbatch_size=8"));

        let bc_message = format_bc_preflight_selection_message(
            EffectiveRuntimeConfig {
                selected: SelectedRuntimeConfig {
                    train_microbatch_size: 64,
                    validation_microbatch_size: 32,
                    accum_steps: 4,
                },
                loader: LoaderRuntimeConfig {
                    num_threads: Some(6),
                    buffer_games: 64,
                    buffer_samples: 512,
                    archive_queue_bound: 8,
                },
            },
            ExplicitSettings {
                train_microbatch_explicit: false,
                validation_microbatch_explicit: true,
            },
        );
        assert!(bc_message.contains("Preflight:"));
        assert!(bc_message.contains("saved train_mb=64 val_mb=32"));
        assert!(bc_message.contains("accum_steps=4"));
        assert!(bc_message.contains("threads=6"));
        assert!(bc_message.contains("explicit(train=false, val=true)"));
    }

    #[test]
    fn formats_probe_table_message_with_selected_candidate() {
        let message = format_probe_table_message(
            "Probe final table",
            ProbeKind::Train,
            &[
                probe_result(ProbeKind::Train, 64, true),
                probe_result(ProbeKind::Train, 48, false),
            ],
            64,
        );

        assert!(message.contains("Probe final table"));
        assert!(message.contains("candidate_mb"));
        assert!(message.contains("train        yes       64"));
        assert!(message.contains("train        no        48"));
    }

    #[test]
    fn dispatches_preflight_before_other_modes() {
        let mut cli = cli();
        cli.preflight = true;
        cli.delta_q_promotion = true;
        let mut handlers = RecordingHandlers::default();

        run_train_modes(cli, config(), &mut handlers).expect("dispatch should succeed");

        assert_eq!(handlers.calls, ["preflight:config.yaml"]);
    }

    #[test]
    fn dispatches_delta_q_before_probe_only() {
        let mut cli = cli();
        cli.delta_q_promotion = true;
        cli.delta_q_baseline_checkpoint = Some(PathBuf::from("baseline.mpk"));
        cli.probe_only = Some(ProbeCliRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 128,
            warmup_steps: Some(2),
            measure_steps: Some(3),
        });
        let mut handlers = RecordingHandlers::default();

        run_train_modes(cli, config(), &mut handlers).expect("dispatch should succeed");

        assert_eq!(handlers.calls, ["delta_q:config.yaml:baseline.mpk"]);
    }

    #[test]
    fn resolves_probe_defaults_before_dispatching_probe_mode() {
        let mut cli = cli();
        cli.probe_only = Some(ProbeCliRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 64,
            warmup_steps: None,
            measure_steps: Some(5),
        });
        let mut config = config();
        config.preflight.warmup_steps = 7;
        let mut handlers = RecordingHandlers::default();

        run_train_modes(cli, config, &mut handlers).expect("dispatch should succeed");

        assert_eq!(handlers.calls, ["probe:config.yaml:1:64:7"]);
    }

    #[test]
    fn dispatches_default_training_mode() {
        let mut handlers = RecordingHandlers::default();

        run_train_modes(cli(), config(), &mut handlers).expect("dispatch should succeed");

        assert_eq!(handlers.calls, ["train:config.yaml"]);
    }

    #[test]
    fn returns_probe_resolution_errors_before_handler_call() {
        let mut cli = cli();
        cli.probe_only = Some(ProbeCliRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 0,
            warmup_steps: Some(1),
            measure_steps: Some(1),
        });
        let mut handlers = RecordingHandlers::default();

        let err = run_train_modes(cli, config(), &mut handlers)
            .expect_err("invalid probe request should fail");

        assert_eq!(err, "--probe-candidate-microbatch must be greater than 0");
        assert!(handlers.calls.is_empty());
    }
}
