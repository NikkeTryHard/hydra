//! Train binary mode dispatch facade.
//!
//! This module owns the CLI mode selection order for the train executable without
//! depending on the compatibility `hydra-train` crate. The binary supplies the
//! concrete handlers while execution internals continue moving into this crate.

use std::path::{Path, PathBuf};

use hydra_train_runtime::config::{TrainCli, TrainConfig};
use hydra_train_runtime::probe_request::{ProbeRequest, probe_request_from_cli};

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
    use hydra_train_runtime::preflight::{PreflightConfig, ProbeKind};

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
