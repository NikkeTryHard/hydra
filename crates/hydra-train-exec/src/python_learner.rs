//! Python BC learner process launcher.
//!
//! Rust owns CLI/config and process boundary validation. Python owns BC training.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use hydra_train_runtime::config::{PythonLearnerCliOptions, PythonLearnerInput};
use serde::Deserialize;

const PYTHON_LEARNER_SCRIPT: &str = "scripts/hydra_pytorch_oracle.py";
const PYTHON_LEARNER_RESULT: &str = "python_learner_result.json";

/// Built command for launching the Python BC learner through Pixi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonLearnerCommand {
    /// Executable name.
    pub program: String,
    /// Argument vector, excluding executable.
    pub args: Vec<String>,
    /// JSON result path passed to Python.
    pub result_path: PathBuf,
}

impl PythonLearnerCommand {
    fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        command
    }
}

/// Minimal success report parsed from Python learner JSON output.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonLearnerReport {
    /// JSON result path consumed by this report.
    pub result_path: PathBuf,
    /// Measured samples per second.
    pub samples_per_second: f64,
    /// Final global step after this run.
    pub global_step: u64,
    /// Optional checkpoint path emitted by Python.
    pub checkpoint_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct PythonLearnerJson {
    summary: PythonLearnerSummaryJson,
    global_step: u64,
    checkpoint_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct PythonLearnerSummaryJson {
    samples_per_s: f64,
}

/// Process runner seam for unit tests and the real OS process boundary.
pub trait PythonLearnerRunner {
    /// Runs a fully built command and returns its exit status.
    fn run(&self, command: &PythonLearnerCommand) -> Result<ExitStatus, String>;
}

/// OS-backed Python learner process runner.
#[derive(Debug, Clone, Copy, Default)]
pub struct OsPythonLearnerRunner;

impl PythonLearnerRunner for OsPythonLearnerRunner {
    fn run(&self, command: &PythonLearnerCommand) -> Result<ExitStatus, String> {
        command
            .command()
            .status()
            .map_err(|err| format!("failed to spawn Python learner through pixi: {err}"))
    }
}

/// Builds the strict Pixi/Python command for the Python BC learner.
pub fn build_python_learner_command(options: &PythonLearnerCliOptions) -> PythonLearnerCommand {
    let result_path = options.output_dir.join(PYTHON_LEARNER_RESULT);
    let mut args = vec![
        "run".to_string(),
        "-e".to_string(),
        "py-train".to_string(),
        "python".to_string(),
        PYTHON_LEARNER_SCRIPT.to_string(),
        "--variant".to_string(),
        options.variant.as_str().to_string(),
        "--residual-profile".to_string(),
        options.residual_profile.as_str().to_string(),
        "--batch".to_string(),
        options.batch_size.to_string(),
        "--microbatch".to_string(),
        options.microbatch_size.to_string(),
        "--warmup".to_string(),
        options.warmup_steps.to_string(),
        "--steps".to_string(),
        options.steps.to_string(),
        "--out".to_string(),
        result_path.display().to_string(),
        "--quiet".to_string(),
        "--w-oracle-critic".to_string(),
        options.oracle_critic_weight.to_string(),
        "--w-safety-residual".to_string(),
        options.safety_residual_weight.to_string(),
    ];
    match &options.input {
        PythonLearnerInput::BcShards { manifest } => {
            args.push("--manifest".to_string());
            args.push(manifest.display().to_string());
        }
        PythonLearnerInput::RawMjai {
            data_dir,
            max_games,
            max_samples,
            train_fraction,
            augment,
            transport,
        } => {
            args.push("--raw-mjai-data-dir".to_string());
            args.push(data_dir.display().to_string());
            args.push("--raw-mjai-worker-threads".to_string());
            args.push("20".to_string());
            args.push("--raw-mjai-train-fraction".to_string());
            args.push(train_fraction.to_string());
            args.push("--raw-mjai-transport".to_string());
            args.push(transport.as_str().to_string());
            if let Some(max_games) = max_games {
                args.push("--raw-mjai-max-games".to_string());
                args.push(max_games.to_string());
            }
            if let Some(max_samples) = max_samples {
                args.push("--raw-mjai-max-samples".to_string());
                args.push(max_samples.to_string());
            }
            if *augment {
                args.push("--raw-mjai-augment".to_string());
            }
        }
    }
    if options.compile_fullgraph_check {
        args.push("--compile-fullgraph-check".to_string());
    }
    if let Some(path) = options.checkpoint_out.as_ref() {
        args.push("--checkpoint-out".to_string());
        args.push(path.display().to_string());
    }
    if let Some(path) = options.resume.as_ref() {
        args.push("--resume".to_string());
        args.push(path.display().to_string());
    }
    if options.checkpoint_every_steps != 0 {
        args.push("--checkpoint-every-steps".to_string());
        args.push(options.checkpoint_every_steps.to_string());
    }
    PythonLearnerCommand {
        program: "pixi".to_string(),
        args,
        result_path,
    }
}

/// Runs the Python learner after validating Rust-owned launch contracts.
pub fn run_python_learner(
    options: &PythonLearnerCliOptions,
) -> Result<PythonLearnerReport, String> {
    run_python_learner_with_runner(options, &OsPythonLearnerRunner)
}

/// Runs the Python learner with an injectable process runner.
pub fn run_python_learner_with_runner(
    options: &PythonLearnerCliOptions,
    runner: &impl PythonLearnerRunner,
) -> Result<PythonLearnerReport, String> {
    match &options.input {
        PythonLearnerInput::BcShards { manifest } if !manifest.is_file() => {
            return Err(format!(
                "Python BC learner manifest does not exist or is not a file: {}",
                manifest.display()
            ));
        }
        PythonLearnerInput::RawMjai { data_dir, .. } if !data_dir.exists() => {
            return Err(format!(
                "Python BC learner raw MJAI data dir does not exist: {}",
                data_dir.display()
            ));
        }
        _ => {}
    }
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create Python BC learner output dir {}: {err}",
            options.output_dir.display()
        )
    })?;
    let command = build_python_learner_command(options);
    let status = runner.run(&command)?;
    if !status.success() {
        return Err(format!(
            "Python BC learner failed with status {status}; JSON result path: {}",
            command.result_path.display()
        ));
    }
    parse_python_learner_report(&command.result_path)
}

/// Parses the minimal JSON fields reported to Rust users.
pub fn parse_python_learner_report(path: &Path) -> Result<PythonLearnerReport, String> {
    let text = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read Python learner result {}: {err}",
            path.display()
        )
    })?;
    let parsed: PythonLearnerJson = serde_json::from_str(&text).map_err(|err| {
        format!(
            "failed to parse Python learner result {}: {err}",
            path.display()
        )
    })?;
    if !parsed.summary.samples_per_s.is_finite() {
        return Err(format!(
            "Python BC learner result {} has non-finite samples_per_s",
            path.display()
        ));
    }
    Ok(PythonLearnerReport {
        result_path: path.to_path_buf(),
        samples_per_second: parsed.summary.samples_per_s,
        global_step: parsed.global_step,
        checkpoint_path: parsed.checkpoint_path,
    })
}

/// Runs a Python BC learner benchmark for one batch/microbatch candidate.
pub fn run_python_learner_benchmark_row(
    base: &PythonLearnerCliOptions,
    batch_size: usize,
    microbatch_size: usize,
    warmup_steps: usize,
    measure_steps: usize,
    runner: &impl PythonLearnerRunner,
) -> Result<PythonLearnerReport, String> {
    let mut options = base.clone();
    options.batch_size = batch_size;
    options.microbatch_size = microbatch_size;
    options.warmup_steps = warmup_steps;
    options.steps = measure_steps.max(1);
    options.checkpoint_out = None;
    options.resume = None;
    options.checkpoint_every_steps = 0;
    run_python_learner_with_runner(&options, runner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_runtime::config::PythonLearnerVariant;
    use std::os::unix::process::ExitStatusExt;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Clone, Copy)]
    struct FakeRunner {
        status: ExitStatus,
    }

    impl PythonLearnerRunner for FakeRunner {
        fn run(&self, _command: &PythonLearnerCommand) -> Result<ExitStatus, String> {
            Ok(self.status)
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra-python-learner-{name}-{nonce}"))
    }

    fn options(root: &Path) -> PythonLearnerCliOptions {
        PythonLearnerCliOptions {
            bc_shards_manifest: root.join("manifest.json"),
            input: PythonLearnerInput::BcShards {
                manifest: root.join("manifest.json"),
            },
            output_dir: root.join("out"),
            device: "cuda:0".to_string(),
            batch_size: 2048,
            microbatch_size: 1024,
            variant: PythonLearnerVariant::CompileDefault,
            residual_profile: hydra_train_runtime::config::PythonResidualProfileConfig::ReluSe,
            warmup_steps: 1,
            steps: 3,
            checkpoint_out: Some(root.join("ckpt.pt")),
            resume: Some(root.join("resume.pt")),
            checkpoint_every_steps: 7,
            compile_fullgraph_check: true,
            oracle_critic_weight: 0.25,
            safety_residual_weight: 0.5,
        }
    }

    #[test]
    fn command_preserves_paths_and_compile_default_args() {
        let root = PathBuf::from("/tmp/hydra py launcher");
        let opts = options(&root);
        let command = build_python_learner_command(&opts);
        assert_eq!(command.program, "pixi");
        assert_eq!(
            command.result_path,
            root.join("out/python_learner_result.json")
        );
        assert_eq!(
            command.args[0..5],
            ["run", "-e", "py-train", "python", PYTHON_LEARNER_SCRIPT]
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--manifest", "/tmp/hydra py launcher/manifest.json"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--variant", "compile_default"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--residual-profile", "relu_se"])
        );
        assert!(command.args.windows(2).any(|w| w
            == [
                "--out",
                "/tmp/hydra py launcher/out/python_learner_result.json"
            ]));
        assert!(
            command
                .args
                .contains(&"--compile-fullgraph-check".to_string())
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--checkpoint-out", "/tmp/hydra py launcher/ckpt.pt"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--resume", "/tmp/hydra py launcher/resume.pt"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--checkpoint-every-steps", "7"])
        );
    }

    #[test]
    fn command_passes_raw_mjai_input_when_manifest_absent() {
        let root = PathBuf::from("/tmp/hydra raw launcher");
        let mut opts = options(&root);
        opts.input = PythonLearnerInput::RawMjai {
            data_dir: root.join("mjai"),
            max_games: Some(5),
            max_samples: Some(4096),
            train_fraction: 0.8,
            augment: true,
            transport: hydra_train_runtime::config::PythonRawMjaiTransportConfig::PinnedPyo3,
        };
        let command = build_python_learner_command(&opts);
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--raw-mjai-data-dir", "/tmp/hydra raw launcher/mjai"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--raw-mjai-max-games", "5"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--raw-mjai-max-samples", "4096"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--raw-mjai-train-fraction", "0.8"])
        );
        assert!(
            command
                .args
                .windows(2)
                .any(|w| w == ["--raw-mjai-transport", "pinned_pyo3"])
        );
        assert!(command.args.contains(&"--raw-mjai-augment".to_string()));
        assert!(!command.args.contains(&"--manifest".to_string()));
    }

    #[test]
    fn missing_manifest_hard_errors_before_spawn() {
        let root = temp_dir("missing-manifest");
        let opts = options(&root);
        let err = run_python_learner_with_runner(
            &opts,
            &FakeRunner {
                status: ExitStatus::from_raw(0),
            },
        )
        .expect_err("missing manifest should fail");
        assert!(err.contains("manifest does not exist"));
    }

    #[test]
    fn failure_status_becomes_hard_error_with_result_path() {
        let root = temp_dir("failed-status");
        fs::create_dir_all(&root).expect("temp root should be created");
        fs::write(root.join("manifest.json"), b"{}").expect("manifest fixture should write");
        let opts = options(&root);
        let err = run_python_learner_with_runner(
            &opts,
            &FakeRunner {
                status: ExitStatus::from_raw(1 << 8),
            },
        )
        .expect_err("nonzero Python status should fail");
        assert!(err.contains("Python BC learner failed"));
        assert!(err.contains("python_learner_result.json"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn success_parses_minimal_result() {
        let root = temp_dir("success");
        fs::create_dir_all(root.join("out")).expect("output dir should be created");
        fs::write(root.join("manifest.json"), b"{}").expect("manifest fixture should write");
        fs::write(
            root.join("out/python_learner_result.json"),
            br#"{"summary":{"samples_per_s":123.5},"global_step":9,"checkpoint_path":"ckpt.pt"}"#,
        )
        .expect("result fixture should write");
        let opts = options(&root);
        let report = run_python_learner_with_runner(
            &opts,
            &FakeRunner {
                status: ExitStatus::from_raw(0),
            },
        )
        .expect("successful status and JSON should parse");
        assert_eq!(report.samples_per_second, 123.5);
        assert_eq!(report.global_step, 9);
        assert_eq!(report.checkpoint_path, Some(PathBuf::from("ckpt.pt")));
        let _ = fs::remove_dir_all(root);
    }
}
