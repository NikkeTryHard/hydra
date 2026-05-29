use super::{
    build_python_learner_command,
    command::PythonLearnerCommand,
    command_builder::{PYTHON_LEARNER_SCRIPT, build_python_learner_command_for_run_dir},
    run_python_learner_with_runner,
    runner::PythonLearnerRunner,
    tensorboard::{tensorboard_port_for_run_dir, tensorboard_url},
};
use hydra_train_runtime::config::{
    PythonLearnerCliOptions, PythonLearnerInput, PythonLearnerVariant,
};
use std::fs::{self, File};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct FakeRunner {
    status: ExitStatus,
}

impl PythonLearnerRunner for FakeRunner {
    fn run(&self, _command: &PythonLearnerCommand) -> Result<ExitStatus, String> {
        Ok(self.status)
    }

    fn spawn_background(
        &self,
        _command: &PythonLearnerCommand,
        _stdout: File,
        _stderr: File,
    ) -> Result<u32, String> {
        Ok(12345)
    }
}

struct ResultWritingRunner;

impl PythonLearnerRunner for ResultWritingRunner {
    fn run(&self, command: &PythonLearnerCommand) -> Result<ExitStatus, String> {
        fs::write(
            &command.result_path,
            br#"{"summary":{"samples_per_s":123.5},"global_step":9,"checkpoint_path":"ckpt.pt"}"#,
        )
        .map_err(|err| err.to_string())?;
        Ok(ExitStatus::from_raw(0))
    }

    fn spawn_background(
        &self,
        _command: &PythonLearnerCommand,
        _stdout: File,
        _stderr: File,
    ) -> Result<u32, String> {
        Ok(12345)
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
        stage: None,
        run_name: None,
        device: "cuda:0".to_string(),
        batch_size: 2048,
        microbatch_size: 1024,
        variant: PythonLearnerVariant::CompileDefault,
        conv_memory_format: hydra_train_runtime::config::PythonConvMemoryFormatConfig::Contiguous,
        backbone_profile: hydra_train_runtime::config::PythonBackboneProfileConfig::Conv2dLocal3,
        residual_profile: hydra_train_runtime::config::PythonResidualProfileConfig::ReluSe,
        hidden: 256,
        blocks: 10,
        bottleneck: 64,
        warmup_steps: 1,
        steps: Some(3),
        full_epoch: false,
        validation_steps: 0,
        validation_max_samples: None,
        validation_every: 0,
        raw_mjai_validation_augment: false,
        validation_source_mode: "fixed".to_string(),
        checkpoint_out: Some(root.join("ckpt.pt")),
        resume: Some(root.join("resume.pt")),
        checkpoint_every_steps: 7,
        log_every_steps: 2,
        keep_step_checkpoints: true,
        tensorboard: true,
        launch_tensorboard: false,
        tensorboard_host: "127.0.0.1".to_string(),
        tensorboard_port: 6006,
        background: false,
        learning_rate: 1.0e-4,
        min_learning_rate: 1.0e-6,
        lr_warmup_steps: 11,
        lr_schedule: "cosine".to_string(),
        schedule_total_steps: Some(99),
        schedule_target_games: None,
        grad_clip_norm: 1.25,
        weight_decay: 2.0e-5,
        ema_enabled: true,
        ema_decay: 0.99,
        ema_start_step: 5,
        ema_update_every_steps: 2,
        ema_device: hydra_train_runtime::config::EmaDeviceConfig::Cuda,
        adamw_fused: hydra_train_runtime::config::PythonAdamwFlagConfig::On,
        adamw_foreach: hydra_train_runtime::config::PythonAdamwFlagConfig::Auto,
        compile_fullgraph_check: true,
        oracle_critic_weight: 0.25,
        safety_residual_weight: 0.5,
        exit_weight: 0.125,
        deltaq_weight: 0.0,
    }
}

fn run_dir(root: &Path, stage: &str, run_id: &str) -> PathBuf {
    root.join("out")
        .join("stages")
        .join(stage)
        .join("runs")
        .join(run_id)
}
#[test]
fn command_preserves_paths_and_compile_default_args() {
    let root = PathBuf::from("/tmp/hydra py launcher");
    let opts = options(&root);
    let command =
        build_python_learner_command_for_run_dir(&opts, &run_dir(&root, "bc_baseline", "run-test"));
    assert_eq!(command.program, "pixi");
    assert_eq!(
        command.result_path,
        root.join("out/stages/bc_baseline/runs/run-test/python_learner_result.json")
    );
    assert_eq!(
        command.args[0..6],
        ["run", "--frozen", "-e", "py-train", "python", PYTHON_LEARNER_SCRIPT]
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
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--conv-memory-format", "contiguous"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--backbone-profile", "conv2d_local3"])
    );
    assert!(command.args.windows(2).any(|w| w == ["--lr", "0.0001"]));
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--min-lr", "0.000001"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--lr-warmup-steps", "11"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--lr-schedule", "cosine"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--schedule-total-steps", "99"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--grad-clip-norm", "1.25"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--adamw-fused", "on"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--adamw-foreach", "auto"])
    );
    assert!(command.args.contains(&"--ema-enabled".to_string()));
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--ema-decay", "0.99"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--ema-start-step", "5"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--ema-update-every-steps", "2"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--ema-device", "cuda"])
    );
    assert!(command.args.windows(2).any(|w| w
        == [
            "--out",
            "/tmp/hydra py launcher/out/stages/bc_baseline/runs/run-test/python_learner_result.json"
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
    assert!(command.args.windows(2).any(|w| w == ["--w-exit", "0.125"]));
    assert!(command.args.windows(2).any(|w| w == ["--w-deltaq", "0"]));
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--resume", "/tmp/hydra py launcher/resume.pt"])
    );
    assert!(!command.args.contains(&"--checkpoint-dir".to_string()));
    assert!(command.args.windows(2).any(|w| w
        == [
            "--log-dir",
            "/tmp/hydra py launcher/out/stages/bc_baseline/runs/run-test/logs"
        ]));
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--log-every-steps", "2"])
    );
    assert!(
        !command
            .args
            .contains(&"--keep-step-checkpoints".to_string())
    );
    assert!(command.args.windows(2).any(|w| w
        == [
            "--tensorboard-dir",
            "/tmp/hydra py launcher/out/stages/bc_baseline/runs/run-test/tensorboard"
        ]));
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--tensorboard-url", "http://127.0.0.1:6006/"])
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
        data_dirs: vec![root.join("mjai"), root.join("mjai-2")],
        max_games: Some(5),
        max_samples: Some(4096),
        skip_games: 3,
        train_fraction: 0.8,
        augment: true,
        transport: hydra_train_runtime::config::PythonRawMjaiTransportConfig::PinnedPyo3,
    };
    let command = build_python_learner_command(&opts);
    let raw_dirs: Vec<&str> = command
        .args
        .windows(2)
        .filter(|w| w[0] == "--raw-mjai-data-dir")
        .map(|w| w[1].as_str())
        .collect();
    assert_eq!(
        raw_dirs,
        [
            "/tmp/hydra raw launcher/mjai",
            "/tmp/hydra raw launcher/mjai-2"
        ]
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
            .any(|w| w == ["--raw-mjai-skip-games", "3"])
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
    assert!(
        !command
            .args
            .contains(&"--raw-mjai-validation-augment".to_string())
    );
}

#[test]
fn command_passes_validation_source_controls() {
    let root = PathBuf::from("/tmp/hydra validation launcher");
    let mut opts = options(&root);
    opts.validation_steps = 2;
    opts.validation_max_samples = Some(65_536);
    opts.validation_every = 5;
    opts.raw_mjai_validation_augment = true;
    opts.validation_source_mode = "streaming".to_string();
    let command = build_python_learner_command(&opts);

    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--validation-steps", "2"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--validation-max-samples", "65536"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--validation-every", "5"])
    );
    assert!(
        command
            .args
            .windows(2)
            .any(|w| w == ["--validation-source-mode", "streaming"])
    );
    assert!(
        command
            .args
            .contains(&"--raw-mjai-validation-augment".to_string())
    );
}

#[test]
fn command_passes_checkpoint_dir_when_checkpoint_out_absent() {
    let root = PathBuf::from("/tmp/hydra dir checkpoint launcher");
    let mut opts = options(&root);
    opts.checkpoint_out = None;
    opts.keep_step_checkpoints = true;
    let command =
        build_python_learner_command_for_run_dir(&opts, &run_dir(&root, "bc_baseline", "run-test"));
    assert!(command.args.windows(2).any(|w| w
        == [
            "--checkpoint-out",
            "/tmp/hydra dir checkpoint launcher/out/stages/bc_baseline/runs/run-test/checkpoints/latest.pt"
        ]));
    assert!(
        !command
            .args
            .contains(&"--keep-step-checkpoints".to_string())
    );
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
    assert!(err.contains("stages/bc_baseline/runs/"));
    assert!(err.contains("python_learner_result.json"));
    assert!(root.join("out/campaign.json").is_file());
    assert!(root.join("out/registry/baselines.jsonl").is_file());
    assert!(root.join("out/stages/bc_baseline/latest_run").is_file());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn success_parses_minimal_result() {
    let root = temp_dir("success");
    fs::create_dir_all(&root).expect("temp root should be created");
    fs::write(root.join("manifest.json"), b"{}").expect("manifest fixture should write");
    let opts = options(&root);
    let report = run_python_learner_with_runner(&opts, &ResultWritingRunner)
        .expect("successful status and JSON should parse");
    assert_eq!(report.samples_per_second, 123.5);
    assert_eq!(report.global_step, 9);
    assert_eq!(report.checkpoint_path, Some(PathBuf::from("ckpt.pt")));
    assert!(
        report
            .result_path
            .starts_with(root.join("out/stages/bc_baseline/runs"))
    );
    let _ = fs::remove_dir_all(root);
}
#[test]
fn tensorboard_port_enumerates_when_preferred_is_busy() {
    let root = temp_dir("tensorboard-port");
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).expect("test port bind should work");
    let busy_port = listener
        .local_addr()
        .expect("local addr should exist")
        .port();
    let mut opts = options(&root);
    opts.tensorboard_port = busy_port;

    let selected = tensorboard_port_for_run_dir(&opts, &opts.output_dir)
        .expect("next port should be selected");

    assert!(selected > busy_port);
    assert_eq!(
        tensorboard_url(&opts, selected),
        format!("http://127.0.0.1:{selected}/")
    );
}
