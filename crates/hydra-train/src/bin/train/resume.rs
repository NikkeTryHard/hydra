use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use colored::Colorize;

use hydra_train::config::PipelineState;

use super::config::{PrecisionMode, RlPhaseConfig, RlTrainConfig};

use super::config::{TrainConfig, train_microbatch_size, validation_microbatch_size};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeResumeContract {
    pub(crate) batch_size: usize,
    pub(crate) train_microbatch_size: usize,
    pub(crate) validation_microbatch_size: usize,
    pub(crate) accum_steps: usize,
    pub(crate) precision_mode: PrecisionMode,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BestValidation {
    pub(crate) policy_loss: f64,
    pub(crate) agreement: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) enum ResumeSemantics {
    RestoreOptimizerSkipSeenSamples,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BcResumeState {
    pub(crate) schema_version: u32,
    pub(crate) resume_semantics: ResumeSemantics,
    pub(crate) next_epoch: usize,
    pub(crate) skip_optimizer_steps_in_epoch: usize,
    pub(crate) global_step: usize,
    pub(crate) best_validation: Option<BestValidation>,
    pub(crate) runtime: RuntimeResumeContract,
    pub(crate) saved_at_unix_s: u64,
}

pub(crate) struct ResumeContext {
    pub(crate) checkpoint_base: Option<PathBuf>,
    pub(crate) state: Option<BcResumeState>,
    pub(crate) optimizer_base: Option<PathBuf>,
    pub(crate) session_start_global_step: usize,
    pub(crate) start_epoch: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct RlRuntimeResumeContract {
    pub(crate) games_per_batch: usize,
    pub(crate) microbatch_size: usize,
    pub(crate) phase: RlPhaseConfig,
    pub(crate) precision_mode: PrecisionMode,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) enum RlResumeSemantics {
    RestoreOptimizerFreshSelfPlay,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RlResumeState {
    pub(crate) schema_version: u32,
    pub(crate) resume_semantics: RlResumeSemantics,
    pub(crate) global_step: usize,
    pub(crate) pipeline_state: PipelineState,
    pub(crate) runtime: RlRuntimeResumeContract,
    pub(crate) saved_at_unix_s: u64,
}

pub(crate) struct RlResumeContext {
    pub(crate) checkpoint_base: Option<PathBuf>,
    pub(crate) state: Option<RlResumeState>,
    pub(crate) optimizer_base: Option<PathBuf>,
    pub(crate) session_start_global_step: usize,
}

impl ResumeContext {
    pub(crate) fn load(config: &TrainConfig) -> Result<Self, String> {
        let checkpoint_base = config
            .resume_checkpoint
            .as_ref()
            .map(|path| checkpoint_base_from_path(path));
        let state = checkpoint_base
            .as_ref()
            .and_then(|base| latest_state_path_for_checkpoint_base(base))
            .filter(|path| path.exists())
            .map(|path| read_resume_state(&path))
            .transpose()?;
        let optimizer_base = checkpoint_base
            .as_ref()
            .and_then(|base| latest_optimizer_base_for_checkpoint_base(base))
            .filter(|path| path.with_extension("bin").exists());
        let session_start_global_step = state.as_ref().map(|state| state.global_step).unwrap_or(0);
        let start_epoch = state.as_ref().map(|state| state.next_epoch).unwrap_or(0);
        Ok(Self {
            checkpoint_base,
            state,
            optimizer_base,
            session_start_global_step,
            start_epoch,
        })
    }

    pub(crate) fn best_validation(&self) -> Option<BestValidation> {
        self.state.as_ref().and_then(|state| state.best_validation)
    }

    pub(crate) fn steps_to_skip_for_epoch(&self, epoch: usize) -> usize {
        self.state
            .as_ref()
            .filter(|state| state.next_epoch == epoch)
            .map(|state| state.skip_optimizer_steps_in_epoch)
            .unwrap_or(0)
    }

    pub(crate) fn print_banner_with_effective_runtime(
        &self,
        effective_runtime: Option<RuntimeResumeContract>,
    ) {
        if let Some(state) = self.state.as_ref() {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "Resume:".bold().cyan(),
                    resume_banner_message(state, effective_runtime).yellow(),
                ))
            );
        }
    }
}

impl RlResumeContext {
    pub(crate) fn load(config: &TrainConfig) -> Result<Self, String> {
        let checkpoint_base = config
            .resume_checkpoint
            .as_ref()
            .map(|path| checkpoint_base_from_path(path));
        let state = checkpoint_base
            .as_ref()
            .and_then(|base| latest_state_path_for_checkpoint_base(base))
            .filter(|path| path.exists())
            .map(|path| read_rl_resume_state(&path))
            .transpose()?;
        let optimizer_base = checkpoint_base
            .as_ref()
            .and_then(|base| latest_optimizer_base_for_checkpoint_base(base))
            .filter(|path| path.with_extension("bin").exists());
        let session_start_global_step = state.as_ref().map(|state| state.global_step).unwrap_or(0);
        Ok(Self {
            checkpoint_base,
            state,
            optimizer_base,
            session_start_global_step,
        })
    }

    pub(crate) fn restores_optimizer_state(&self) -> bool {
        self.state.is_some()
    }

    pub(crate) fn print_banner(&self) {
        if let Some(state) = self.state.as_ref() {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "RL Resume:".bold().cyan(),
                    rl_resume_banner_message(state).yellow(),
                ))
            );
        }
    }
}

impl ResumeContext {
    pub(crate) fn restores_optimizer_state(&self) -> bool {
        self.state.as_ref().is_some_and(|state| {
            matches!(
                state.resume_semantics,
                ResumeSemantics::RestoreOptimizerSkipSeenSamples
            )
        })
    }
}

pub(crate) struct EpochContinuation {
    pub(crate) next_epoch: usize,
    pub(crate) skip_optimizer_steps_in_epoch: usize,
    pub(crate) epoch_completed: bool,
}

pub(crate) fn current_timestamp_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn checkpoint_base_from_path(path: &Path) -> PathBuf {
    if path.extension() == Some(OsStr::new("mpk")) {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

pub(crate) fn latest_state_path_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_state.yaml"))
}

pub(crate) fn latest_optimizer_base_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_optimizer"))
}

pub(crate) fn read_resume_state(path: &Path) -> Result<BcResumeState, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read resume state {}: {err}", path.display()))?;
    let state: BcResumeState = serde_yaml::from_str(&raw)
        .map_err(|err| format!("failed to parse resume state {}: {err}", path.display()))?;
    if state.schema_version != 3 {
        return Err(format!(
            "unsupported resume schema_version {} in {}; expected 3",
            state.schema_version,
            path.display()
        ));
    }
    if state.resume_semantics != ResumeSemantics::RestoreOptimizerSkipSeenSamples {
        return Err(format!(
            "unsupported resume semantics {:?} in {}; expected RestoreOptimizerSkipSeenSamples",
            state.resume_semantics,
            path.display()
        ));
    }
    Ok(state)
}

pub(crate) fn read_rl_resume_state(path: &Path) -> Result<RlResumeState, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read RL resume state {}: {err}", path.display()))?;
    let state: RlResumeState = serde_yaml::from_str(&raw)
        .map_err(|err| format!("failed to parse RL resume state {}: {err}", path.display()))?;
    if state.schema_version != 1 {
        return Err(format!(
            "unsupported RL resume schema_version {} in {}; expected 1",
            state.schema_version,
            path.display()
        ));
    }
    Ok(state)
}

pub(crate) fn runtime_resume_contract(config: &TrainConfig) -> RuntimeResumeContract {
    let train_microbatch_size = train_microbatch_size(config);
    RuntimeResumeContract {
        batch_size: config.batch_size,
        train_microbatch_size,
        validation_microbatch_size: validation_microbatch_size(config),
        accum_steps: config.batch_size.div_ceil(train_microbatch_size).max(1),
        precision_mode: config.precision_mode,
    }
}

pub(crate) fn rl_runtime_resume_contract(rl: &RlTrainConfig) -> RlRuntimeResumeContract {
    RlRuntimeResumeContract {
        games_per_batch: rl.games_per_batch,
        microbatch_size: rl
            .microbatch_size
            .unwrap_or(hydra_train::training::rl::DEFAULT_RL_MICROBATCH_SIZE),
        phase: rl.phase,
        precision_mode: PrecisionMode::Fp32,
    }
}

pub(crate) fn validate_rl_resume_runtime_compatibility(
    state: &RlResumeState,
    current: RlRuntimeResumeContract,
) -> Result<(), String> {
    if state.runtime != current {
        return Err(format!(
            "RL resume runtime mismatch: checkpoint games_per_batch={} microbatch_size={} phase={:?} current games_per_batch={} microbatch_size={} phase={:?}",
            state.runtime.games_per_batch,
            state.runtime.microbatch_size,
            state.runtime.phase,
            current.games_per_batch,
            current.microbatch_size,
            current.phase,
        ));
    }
    Ok(())
}

pub(crate) fn validate_resume_runtime_compatibility(
    state: &BcResumeState,
    current: RuntimeResumeContract,
) -> Result<(), String> {
    if state.runtime.batch_size != current.batch_size {
        return Err(format!(
            "resume batch_size mismatch: checkpoint={} current={}",
            state.runtime.batch_size, current.batch_size
        ));
    }

    if state.skip_optimizer_steps_in_epoch > 0 && state.runtime != current {
        return Err(format!(
            "partial-epoch resume requires identical runtime contract; checkpoint train_mb={} val_mb={} accum_steps={} current train_mb={} val_mb={} accum_steps={}",
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
            current.train_microbatch_size,
            current.validation_microbatch_size,
            current.accum_steps,
        ));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn test_runtime_resume_contract(
    batch_size: usize,
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
) -> RuntimeResumeContract {
    RuntimeResumeContract {
        batch_size,
        train_microbatch_size,
        validation_microbatch_size,
        accum_steps: batch_size.div_ceil(train_microbatch_size).max(1),
        precision_mode: PrecisionMode::Fp32,
    }
}

pub(crate) fn write_resume_state(path: &Path, state: &BcResumeState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(state)
        .map_err(|err| format!("failed to serialize resume state {}: {err}", path.display()))?;
    fs::write(path, yaml)
        .map_err(|err| format!("failed to write resume state {}: {err}", path.display()))
}

pub(crate) fn build_resume_state(
    next_epoch: usize,
    skip_optimizer_steps_in_epoch: usize,
    global_step: usize,
    best_validation: Option<BestValidation>,
    runtime: RuntimeResumeContract,
) -> BcResumeState {
    BcResumeState {
        schema_version: 3,
        resume_semantics: ResumeSemantics::RestoreOptimizerSkipSeenSamples,
        next_epoch,
        skip_optimizer_steps_in_epoch,
        global_step,
        best_validation,
        runtime,
        saved_at_unix_s: current_timestamp_s(),
    }
}

pub(crate) fn build_rl_resume_state(
    global_step: usize,
    pipeline_state: PipelineState,
    runtime: RlRuntimeResumeContract,
) -> RlResumeState {
    RlResumeState {
        schema_version: 1,
        resume_semantics: RlResumeSemantics::RestoreOptimizerFreshSelfPlay,
        global_step,
        pipeline_state,
        runtime,
        saved_at_unix_s: current_timestamp_s(),
    }
}

pub(crate) fn paused_training_message(continuation: &EpochContinuation) -> String {
    format!(
        "resume_epoch={} skipped_optimizer_steps_in_epoch={} optimizer_state=restored sample_cursor=reconstructed_from_logical_batch_count partial_epoch_requires_matching_runtime",
        continuation.next_epoch + 1,
        continuation.skip_optimizer_steps_in_epoch
    )
}

pub(crate) fn resume_banner_message(
    state: &BcResumeState,
    effective_runtime: Option<RuntimeResumeContract>,
) -> String {
    if state.skip_optimizer_steps_in_epoch > 0 {
        format!(
            "global_step={} semantics={:?} skipping {} completed optimizer steps worth of samples in epoch {} before new updates runtime=train_mb:{} val_mb:{} accum_steps:{}",
            state.global_step,
            state.resume_semantics,
            state.skip_optimizer_steps_in_epoch,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
        )
    } else {
        let base = format!(
            "global_step={} semantics={:?} resuming at epoch {} with new updates immediately runtime=train_mb:{} val_mb:{} accum_steps:{}",
            state.global_step,
            state.resume_semantics,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
        );
        match effective_runtime.filter(|runtime| runtime != &state.runtime) {
            Some(runtime) => format!(
                "{} effective_runtime=train_mb:{} val_mb:{} accum_steps:{}",
                base,
                runtime.train_microbatch_size,
                runtime.validation_microbatch_size,
                runtime.accum_steps,
            ),
            None => base,
        }
    }
}

pub(crate) fn rl_resume_banner_message(state: &RlResumeState) -> String {
    format!(
        "global_step={} semantics={:?} phase={:?} games={} samples={} runtime=games_per_batch:{} microbatch_size:{}",
        state.global_step,
        state.resume_semantics,
        state.pipeline_state.phase,
        state.pipeline_state.total_games,
        state.pipeline_state.total_samples,
        state.runtime.games_per_batch,
        state.runtime.microbatch_size,
    )
}
use super::presentation::timestamped;

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train::config::TrainingPhase;
    use std::fs;

    use crate::config::{BcHyperparamConfig, PrecisionMode, TrainConfig};

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir: PathBuf::from("/tmp/out"),
            num_epochs: 4,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            train_fraction: 0.9,
            source_filters: hydra_train::data::pipeline::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 7,
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
            device: "cpu".to_string(),
            precision_mode: PrecisionMode::Fp32,
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(1),
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
            preflight: Default::default(),
        }
    }

    fn dummy_best_validation() -> BestValidation {
        BestValidation {
            policy_loss: 0.25,
            agreement: 0.8,
        }
    }

    fn unique_test_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra-resume-test-{label}-{unique}"))
    }

    fn write_yaml_file(label: &str, contents: &str) -> PathBuf {
        let path = unique_test_path(label).with_extension("yaml");
        fs::write(&path, contents).expect("yaml fixture should be writable");
        path
    }

    #[test]
    fn checkpoint_base_and_latest_path_helpers_cover_latest_and_non_latest_names() {
        let latest = Path::new("/tmp/latest_model.mpk");
        let other = Path::new("/tmp/epoch_1_model.mpk");

        assert_eq!(
            checkpoint_base_from_path(latest),
            PathBuf::from("/tmp/latest_model")
        );
        assert_eq!(
            checkpoint_base_from_path(other),
            PathBuf::from("/tmp/epoch_1_model")
        );
        assert_eq!(
            checkpoint_base_from_path(Path::new("/tmp/latest_model")),
            PathBuf::from("/tmp/latest_model")
        );

        assert_eq!(
            latest_state_path_for_checkpoint_base(Path::new("/tmp/latest_model")),
            Some(PathBuf::from("/tmp/latest_state.yaml"))
        );
        assert_eq!(
            latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/latest_model")),
            Some(PathBuf::from("/tmp/latest_optimizer"))
        );
        assert!(latest_state_path_for_checkpoint_base(Path::new("/tmp/epoch_1_model")).is_none());
        assert!(
            latest_optimizer_base_for_checkpoint_base(Path::new("/tmp/epoch_1_model")).is_none()
        );
    }

    #[test]
    fn runtime_contract_helpers_compute_expected_values() {
        let config = dummy_config();
        let bc = runtime_resume_contract(&config);
        assert_eq!(bc.batch_size, 256);
        assert_eq!(bc.train_microbatch_size, 64);
        assert_eq!(bc.validation_microbatch_size, 32);
        assert_eq!(bc.accum_steps, 4);
        assert_eq!(bc.precision_mode, PrecisionMode::Fp32);

        let rl = rl_runtime_resume_contract(&RlTrainConfig::default());
        assert_eq!(rl.games_per_batch, RlTrainConfig::default().games_per_batch);
        assert_eq!(
            rl.microbatch_size,
            hydra_train::training::rl::DEFAULT_RL_MICROBATCH_SIZE
        );
        assert_eq!(rl.phase, RlTrainConfig::default().phase);
        assert_eq!(rl.precision_mode, PrecisionMode::Fp32);
    }

    #[test]
    fn validate_resume_runtime_compatibility_checks_batch_and_partial_epoch_contracts() {
        let current = test_runtime_resume_contract(256, 64, 32);
        let mut state = build_resume_state(2, 0, 12, Some(dummy_best_validation()), current);
        assert_eq!(
            validate_resume_runtime_compatibility(&state, current),
            Ok(())
        );

        let mismatched_batch = test_runtime_resume_contract(128, 64, 32);
        let err = validate_resume_runtime_compatibility(&state, mismatched_batch)
            .expect_err("batch size mismatch should be rejected");
        assert!(err.contains("resume batch_size mismatch"));

        state.skip_optimizer_steps_in_epoch = 3;
        let mismatched_partial = test_runtime_resume_contract(256, 32, 32);
        let err = validate_resume_runtime_compatibility(&state, mismatched_partial)
            .expect_err("partial epoch resume should require identical runtime contract");
        assert!(err.contains("partial-epoch resume requires identical runtime contract"));

        state.skip_optimizer_steps_in_epoch = 1;
        let mut mismatched_precision = current;
        mismatched_precision.precision_mode = PrecisionMode::Bf16Autocast;
        let err = validate_resume_runtime_compatibility(&state, mismatched_precision)
            .expect_err("partial epoch resume should reject precision mode mismatch");
        assert!(err.contains("partial-epoch resume requires identical runtime contract"));
    }

    #[test]
    fn validate_rl_resume_runtime_compatibility_rejects_mismatched_runtime() {
        let state = build_rl_resume_state(
            10,
            PipelineState {
                phase: TrainingPhase::ExitPondering,
                ..PipelineState::default()
            },
            RlRuntimeResumeContract {
                games_per_batch: 8,
                microbatch_size: 16,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: PrecisionMode::Fp32,
            },
        );

        let err = validate_rl_resume_runtime_compatibility(
            &state,
            RlRuntimeResumeContract {
                games_per_batch: 16,
                microbatch_size: 16,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: PrecisionMode::Fp32,
            },
        )
        .expect_err("RL runtime mismatch should be rejected");

        assert!(err.contains("RL resume runtime mismatch"));
    }

    #[test]
    fn read_resume_state_rejects_schema_and_semantics_mismatches() {
        let schema_path = write_yaml_file(
            "bad-bc-schema",
            r#"schema_version: 2
resume_semantics: RestoreOptimizerSkipSeenSamples
next_epoch: 1
skip_optimizer_steps_in_epoch: 0
global_step: 4
best_validation: null
runtime:
  batch_size: 256
  train_microbatch_size: 64
  validation_microbatch_size: 32
  accum_steps: 4
  precision_mode: fp32
saved_at_unix_s: 1
"#,
        );
        let schema_err = read_resume_state(&schema_path).expect_err("schema mismatch should fail");
        assert!(schema_err.contains("unsupported resume schema_version 2"));

        let semantics_path = write_yaml_file(
            "bad-bc-semantics",
            r#"schema_version: 3
resume_semantics: RestoreOptimizerFreshSelfPlay
next_epoch: 1
skip_optimizer_steps_in_epoch: 0
global_step: 4
best_validation: null
runtime:
  batch_size: 256
  train_microbatch_size: 64
  validation_microbatch_size: 32
  accum_steps: 4
  precision_mode: fp32
saved_at_unix_s: 1
"#,
        );
        let semantics_err =
            read_resume_state(&semantics_path).expect_err("semantics mismatch should fail");
        assert!(semantics_err.contains("failed to parse resume state"));
    }

    #[test]
    fn read_rl_resume_state_rejects_schema_mismatch() {
        let path = write_yaml_file(
            "bad-rl-schema",
            r#"schema_version: 2
resume_semantics: RestoreOptimizerFreshSelfPlay
global_step: 7
pipeline_state:
  phase: ExitPondering
  total_games: 0
  total_samples: 0
  gpu_hours_used: 0.0
  learner_version: 0
runtime:
  games_per_batch: 8
  microbatch_size: 16
  phase: ExitPondering
  precision_mode: fp32
saved_at_unix_s: 1
"#,
        );
        let err = read_rl_resume_state(&path).expect_err("schema mismatch should fail");
        assert!(err.contains("failed to parse RL resume state"));
    }

    #[test]
    fn resume_context_helpers_cover_state_access_and_restore_flags() {
        let runtime = test_runtime_resume_contract(256, 64, 32);
        let state = build_resume_state(3, 2, 11, Some(dummy_best_validation()), runtime);
        let ctx = ResumeContext {
            checkpoint_base: None,
            state: Some(state.clone()),
            optimizer_base: None,
            session_start_global_step: state.global_step,
            start_epoch: state.next_epoch,
        };

        assert_eq!(ctx.best_validation(), Some(dummy_best_validation()));
        assert_eq!(ctx.steps_to_skip_for_epoch(3), 2);
        assert_eq!(ctx.steps_to_skip_for_epoch(1), 0);
        assert!(ctx.restores_optimizer_state());

        let empty = ResumeContext {
            checkpoint_base: None,
            state: None,
            optimizer_base: None,
            session_start_global_step: 0,
            start_epoch: 0,
        };
        assert_eq!(empty.best_validation(), None);
        assert_eq!(empty.steps_to_skip_for_epoch(0), 0);
        assert!(!empty.restores_optimizer_state());
    }

    #[test]
    fn rl_resume_context_restore_flag_tracks_presence_of_state() {
        let ctx = RlResumeContext {
            checkpoint_base: None,
            state: None,
            optimizer_base: None,
            session_start_global_step: 0,
        };
        assert!(!ctx.restores_optimizer_state());

        let ctx = RlResumeContext {
            checkpoint_base: None,
            state: Some(build_rl_resume_state(
                5,
                PipelineState::default(),
                RlRuntimeResumeContract {
                    games_per_batch: 8,
                    microbatch_size: 16,
                    phase: RlPhaseConfig::ExitPondering,
                    precision_mode: PrecisionMode::Fp32,
                },
            )),
            optimizer_base: None,
            session_start_global_step: 5,
        };
        assert!(ctx.restores_optimizer_state());
    }

    #[test]
    fn banner_and_pause_messages_include_runtime_details() {
        let state = build_resume_state(
            1,
            3,
            9,
            Some(dummy_best_validation()),
            test_runtime_resume_contract(256, 64, 32),
        );
        let resume_banner = resume_banner_message(&state, None);
        assert!(resume_banner.contains("global_step=9"));
        assert!(resume_banner.contains("skipping 3 completed optimizer steps"));
        assert!(resume_banner.contains("runtime=train_mb:64 val_mb:32 accum_steps:4"));

        let immediate_state =
            build_resume_state(0, 0, 1, None, test_runtime_resume_contract(256, 64, 32));
        let immediate_banner = resume_banner_message(&immediate_state, None);
        assert!(immediate_banner.contains("resuming at epoch 1 with new updates immediately"));

        let effective_banner = resume_banner_message(
            &immediate_state,
            Some(test_runtime_resume_contract(256, 32, 16)),
        );
        assert!(effective_banner.contains("runtime=train_mb:64 val_mb:32 accum_steps:4"));
        assert!(effective_banner.contains("effective_runtime=train_mb:32 val_mb:16 accum_steps:8"));

        let rl_banner = rl_resume_banner_message(&build_rl_resume_state(
            10,
            PipelineState {
                phase: TrainingPhase::ExitPondering,
                total_games: 12,
                total_samples: 128,
                ..PipelineState::default()
            },
            RlRuntimeResumeContract {
                games_per_batch: 8,
                microbatch_size: 16,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: PrecisionMode::Fp32,
            },
        ));
        assert!(rl_banner.contains("phase=ExitPondering"));
        assert!(rl_banner.contains("games=12 samples=128"));
        assert!(rl_banner.contains("runtime=games_per_batch:8 microbatch_size:16"));

        let paused = paused_training_message(&EpochContinuation {
            next_epoch: 2,
            skip_optimizer_steps_in_epoch: 4,
            epoch_completed: false,
        });
        assert!(paused.contains("resume_epoch=3"));
        assert!(paused.contains("skipped_optimizer_steps_in_epoch=4"));
    }

    #[test]
    fn write_resume_state_round_trips_and_load_helpers_detect_latest_files() {
        let root = unique_test_path("resume-roundtrip");
        fs::create_dir_all(&root).expect("temp root should be creatable");
        let checkpoint = root.join("latest_model.mpk");
        fs::write(&checkpoint, b"model").expect("checkpoint marker should be writable");
        let latest_state = root.join("latest_state.yaml");
        let latest_optimizer = root.join("latest_optimizer.bin");
        fs::write(&latest_optimizer, b"optimizer").expect("optimizer marker should be writable");

        let state = build_resume_state(
            2,
            1,
            7,
            Some(dummy_best_validation()),
            test_runtime_resume_contract(256, 64, 32),
        );
        write_resume_state(&latest_state, &state).expect("resume state should write");

        let loaded = read_resume_state(&latest_state).expect("written resume state should parse");
        assert_eq!(loaded, state);

        let mut config = dummy_config();
        config.resume_checkpoint = Some(checkpoint);
        let ctx = ResumeContext::load(&config).expect("resume context should load latest files");
        assert_eq!(ctx.session_start_global_step, 7);
        assert_eq!(ctx.start_epoch, 2);
        assert_eq!(ctx.optimizer_base, Some(root.join("latest_optimizer")));
        assert_eq!(ctx.state, Some(state));
    }

    #[test]
    fn rl_resume_context_load_detects_latest_state_and_optimizer() {
        let root = unique_test_path("rl-resume-load");
        fs::create_dir_all(&root).expect("temp root should be creatable");
        let checkpoint = root.join("latest_model.mpk");
        fs::write(&checkpoint, b"model").expect("checkpoint marker should be writable");
        let latest_state = root.join("latest_state.yaml");
        let latest_optimizer = root.join("latest_optimizer.bin");
        fs::write(&latest_optimizer, b"optimizer").expect("optimizer marker should be writable");

        let state = build_rl_resume_state(
            5,
            PipelineState {
                phase: TrainingPhase::ExitPondering,
                ..PipelineState::default()
            },
            RlRuntimeResumeContract {
                games_per_batch: 8,
                microbatch_size: 16,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: PrecisionMode::Fp32,
            },
        );
        let yaml = serde_yaml::to_string(&state).expect("RL resume state should serialize");
        fs::write(&latest_state, yaml).expect("RL resume state should write");

        let mut config = dummy_config();
        config.resume_checkpoint = Some(checkpoint);
        let ctx =
            RlResumeContext::load(&config).expect("RL resume context should load latest files");
        assert_eq!(ctx.session_start_global_step, 5);
        assert_eq!(ctx.optimizer_base, Some(root.join("latest_optimizer")));
        assert_eq!(ctx.state, Some(state));
    }
}
