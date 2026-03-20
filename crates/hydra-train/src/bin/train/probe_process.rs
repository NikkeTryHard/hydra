use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use super::artifacts::{BcArtifactPaths, RlArtifactPaths};
use super::presentation::{format_probe_progress_line, with_utc_timestamp};
use super::probe_request::ProbeRequest;
use super::probe_summary::probe_kind_name;

fn interrupt_flag() -> Result<Arc<AtomicBool>, String> {
    static INTERRUPTED: OnceLock<Arc<AtomicBool>> = OnceLock::new();
    static HANDLER_INSTALLED: OnceLock<()> = OnceLock::new();
    let flag = INTERRUPTED
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();
    if HANDLER_INSTALLED.get().is_none() {
        ctrlc::set_handler({
            let flag = flag.clone();
            move || {
                flag.store(true, Ordering::SeqCst);
            }
        })
        .map_err(|err| format!("failed to install preflight interrupt handler: {err}"))?;
        let _ = HANDLER_INSTALLED.set(());
    }
    Ok(flag)
}

fn should_suppress_probe_output_line(line: &str) -> bool {
    let lowered = line.to_ascii_lowercase();
    lowered.contains("thread 'main'")
        || lowered.contains("called `result::unwrap()`")
        || lowered.contains("called `result::unwrap()")
        || lowered.contains("note: run with `rust_backtrace=1`")
        || lowered.contains("stack backtrace")
        || lowered.contains("frame #")
        || lowered.contains("exception raised from malloc")
        || lowered.contains("/pytorch/")
        || lowered.contains("/opt/conda/lib/python")
        || lowered.contains("cudacachingallocator")
        || lowered.contains("skipping ")
}

fn normalized_probe_output_line(line: &str) -> Option<String> {
    if let Some(formatted) = format_probe_progress_line(line) {
        return Some(formatted);
    }
    if line.trim_start().starts_with("probe_progress ") {
        return None;
    }
    if should_suppress_probe_output_line(line) {
        return None;
    }
    Some(line.trim().to_string())
}

fn spawn_output_forwarder<R>(reader: R, stderr: bool) -> thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut collected = Vec::new();
        let mut buffered = BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = buffered
                .read_until(b'\n', &mut line)
                .map_err(|err| format!("failed reading preflight probe output: {err}"))?;
            if read == 0 {
                break;
            }
            collected.extend_from_slice(&line);
            let text = String::from_utf8_lossy(&line);
            if let Some(formatted) = normalized_probe_output_line(&text) {
                if stderr {
                    writeln!(std::io::stderr(), "{formatted}").map_err(|err| {
                        format!("failed forwarding preflight probe stderr: {err}")
                    })?;
                    std::io::stderr()
                        .flush()
                        .map_err(|err| format!("failed flushing preflight probe stderr: {err}"))?;
                } else {
                    writeln!(std::io::stdout(), "{formatted}").map_err(|err| {
                        format!("failed forwarding preflight probe stdout: {err}")
                    })?;
                    std::io::stdout()
                        .flush()
                        .map_err(|err| format!("failed flushing preflight probe stdout: {err}"))?;
                }
            }
        }
        Ok(collected)
    })
}

fn spawn_probe_heartbeat(
    interrupted: Arc<AtomicBool>,
    kind: ProbeKind,
    candidate_microbatch: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let started = Instant::now();
        while !interrupted.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_secs(5));
            if interrupted.load(Ordering::SeqCst) {
                break;
            }
            let line = with_utc_timestamp(format!(
                "[preflight:{}] candidate_mb={} phase=heartbeat elapsed={:.1}s still_running",
                probe_kind_name(kind),
                candidate_microbatch,
                started.elapsed().as_secs_f64(),
            ));
            let _ = writeln!(std::io::stdout(), "{line}");
            let _ = std::io::stdout().flush();
        }
    })
}

fn summarize_probe_failure_output(output: &str) -> String {
    let mut lines = Vec::new();
    for line in output.lines() {
        if should_suppress_probe_output_line(line) {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("probe_progress ") {
            continue;
        }
        lines.push(trimmed.to_string());
        if lines.len() >= 3 {
            break;
        }
    }
    lines.join(" | ")
}

fn probe_failure_detail(
    status: ProbeStatus,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
) -> String {
    match status {
        ProbeStatus::Oom => format!(
            "probe process status={exit_code:?} detail=libtorch/cuda oom during preflight probe; raw panic output suppressed"
        ),
        _ => {
            let summary = summarize_probe_failure_output(stderr);
            let fallback = if summary.is_empty() {
                summarize_probe_failure_output(stdout)
            } else {
                summary
            };
            if fallback.is_empty() {
                format!(
                    "probe process status={exit_code:?} detail=probe child failed without structured result"
                )
            } else {
                format!("probe process status={exit_code:?} detail={fallback}")
            }
        }
    }
}

fn join_output_forwarder(
    handle: thread::JoinHandle<Result<Vec<u8>, String>>,
    stream_name: &str,
) -> Result<Vec<u8>, String> {
    handle
        .join()
        .map_err(|_| format!("preflight probe {stream_name} forwarder panicked"))?
}

fn child_output(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> std::process::Output {
    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

pub(super) fn mem_available_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub(super) fn mem_total_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub(super) fn rl_probe_required_free_bytes(config: &super::config::TrainConfig) -> Option<u64> {
    if config.preflight.rl_probe_min_free_memory_bytes == 0
        && config.preflight.rl_probe_memory_headroom_ratio <= 0.0
    {
        return None;
    }
    let total = mem_total_bytes()?;
    let ratio_floor =
        ((total as f64) * config.preflight.rl_probe_memory_headroom_ratio.max(0.0)).ceil() as u64;
    Some(
        config
            .preflight
            .rl_probe_min_free_memory_bytes
            .max(ratio_floor),
    )
}

fn wait_for_probe_child(
    child: &mut Child,
    interrupted: &AtomicBool,
) -> Result<Option<ExitStatus>, String> {
    loop {
        if interrupted.load(Ordering::SeqCst) {
            child.kill().ok();
            child.wait().ok();
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(err) => {
                child.kill().ok();
                child.wait().ok();
                return Err(format!(
                    "failed while waiting for preflight probe child: {err}"
                ));
            }
        }
    }
}

pub(super) fn write_probe_result(path: &Path, result: &ProbeResult) -> Result<(), String> {
    let json = serde_json::to_string(result)
        .map_err(|err| format!("failed to serialize probe result {}: {err}", path.display()))?;
    fs::write(path, json)
        .map_err(|err| format!("failed to write probe result {}: {err}", path.display()))
}

pub(super) fn read_probe_result(path: &Path) -> Result<ProbeResult, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read probe result {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse probe result {}: {err}", path.display()))
}

pub(super) fn probe_result_path(
    artifacts: &BcArtifactPaths,
    kind: ProbeKind,
    candidate_microbatch: usize,
    attempt: usize,
) -> PathBuf {
    artifacts.root.join(format!(
        "preflight_probe_{}_{}_{}.json",
        probe_kind_name(kind),
        candidate_microbatch,
        attempt
    ))
}

pub(super) fn rl_probe_result_path(
    artifacts: &RlArtifactPaths,
    kind: ProbeKind,
    candidate_microbatch: usize,
    attempt: usize,
) -> PathBuf {
    artifacts.root.join(format!(
        "preflight_probe_{}_{}_{}.json",
        probe_kind_name(kind),
        candidate_microbatch,
        attempt
    ))
}

pub(super) fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
    classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<ProbeResult, String> {
    let _config = super::config::read_config(config_path)?;
    fs::remove_file(result_path).ok();
    let interrupted = interrupt_flag()?;
    interrupted.store(false, Ordering::SeqCst);
    let mut child =
        Command::new(env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?)
            .arg(config_path)
            .arg("--probe-kind")
            .arg(probe_kind_name(request.kind))
            .arg("--probe-candidate-microbatch")
            .arg(request.candidate_microbatch.to_string())
            .arg("--probe-warmup-steps")
            .arg(request.warmup_steps.to_string())
            .arg("--probe-measure-steps")
            .arg(request.measure_steps.to_string())
            .arg("--probe-result-path")
            .arg(result_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| format!("failed to spawn preflight probe child: {err}"))?;
    let stdout_handle = child
        .stdout
        .take()
        .map(|stdout| spawn_output_forwarder(stdout, false));
    let stderr_handle = child
        .stderr
        .take()
        .map(|stderr| spawn_output_forwarder(stderr, true));
    let heartbeat_handle = spawn_probe_heartbeat(
        interrupted.clone(),
        request.kind,
        request.candidate_microbatch,
    );
    if wait_for_probe_child(&mut child, interrupted.as_ref())?.is_none() {
        fs::remove_file(result_path).ok();
        interrupted.store(true, Ordering::SeqCst);
        let _ = heartbeat_handle.join();
        if let Some(handle) = stdout_handle {
            let _ = join_output_forwarder(handle, "stdout");
        }
        if let Some(handle) = stderr_handle {
            let _ = join_output_forwarder(handle, "stderr");
        }
        return Err("preflight interrupted; probe child terminated".to_string());
    }
    interrupted.store(true, Ordering::SeqCst);
    let _ = heartbeat_handle.join();
    let stdout = match stdout_handle {
        Some(handle) => join_output_forwarder(handle, "stdout")?,
        None => Vec::new(),
    };
    let stderr = match stderr_handle {
        Some(handle) => join_output_forwarder(handle, "stderr")?,
        None => Vec::new(),
    };
    let status = child
        .try_wait()
        .map_err(|err| format!("failed to query preflight probe child status: {err}"))?
        .ok_or_else(|| "preflight probe child exited without final status".to_string())?;
    let output = child_output(status, stdout, stderr);

    if result_path.exists() {
        let result = read_probe_result(result_path)?;
        fs::remove_file(result_path).ok();
        return Ok(result);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stdout = stdout.trim();
    let stderr = stderr.trim();
    let combined = format!("stdout={stdout} stderr={stderr}");
    let status = classify_probe_detail(&combined);
    let detail = probe_failure_detail(status.clone(), stdout, stderr, output.status.code());
    Ok(ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::io::Cursor;
    use std::process::Command;

    use hydra_train::preflight::PreflightConfig;

    use super::super::artifacts::{BcArtifactPaths, RlArtifactPaths};
    use super::super::config::{
        default_archive_queue_bound, default_augment, default_batch_size, default_buffer_games,
        default_buffer_samples, default_checkpoint_every_n_steps, default_device,
        default_log_every_n_steps, default_max_skip_logs_per_source,
        default_max_validation_samples, default_seed, default_tensorboard, default_train_fraction,
        default_validate_every_n_steps, default_validation_every_n_epochs, BcHyperparamConfig,
        TrainConfig,
    };
    use super::*;

    fn test_train_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/hydra-data"),
            output_dir: PathBuf::from("/tmp/hydra-output"),
            num_epochs: 1,
            batch_size: default_batch_size(),
            microbatch_size: None,
            validation_microbatch_size: None,
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            train_fraction: default_train_fraction(),
            augment: default_augment(),
            resume_checkpoint: None,
            seed: default_seed(),
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
            device: default_device(),
            buffer_games: default_buffer_games(),
            buffer_samples: default_buffer_samples(),
            num_threads: None,
            tensorboard: default_tensorboard(),
            archive_queue_bound: default_archive_queue_bound(),
            validation_every_n_epochs: default_validation_every_n_epochs(),
            max_skip_logs_per_source: default_max_skip_logs_per_source(),
            log_every_n_steps: default_log_every_n_steps(),
            validate_every_n_steps: default_validate_every_n_steps(),
            checkpoint_every_n_steps: default_checkpoint_every_n_steps(),
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: default_max_validation_samples(),
            preflight: PreflightConfig::default(),
        }
    }

    fn unique_test_path(name: &str, extension: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hydra_probe_process_{name}_{}_{}.{}",
            std::process::id(),
            nanos,
            extension
        ))
    }

    #[test]
    fn suppressor_filters_known_panic_and_backend_noise() {
        assert!(should_suppress_probe_output_line("thread 'main' panicked"));
        assert!(should_suppress_probe_output_line("stack backtrace:"));
        assert!(should_suppress_probe_output_line(
            "/pytorch/allocator/file.cpp"
        ));
        assert!(should_suppress_probe_output_line(
            "CudaCachingAllocator failure"
        ));
        assert!(should_suppress_probe_output_line(
            "NOTE: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"
        ));
        assert!(should_suppress_probe_output_line(
            "Frame #12: allocator::oom"
        ));
        assert!(should_suppress_probe_output_line(
            "skipping malformed archive entry"
        ));
        assert!(!should_suppress_probe_output_line(
            "normal training status line"
        ));
    }

    #[test]
    fn normalized_output_formats_progress_and_drops_raw_progress_and_noise() {
        let formatted = normalized_probe_output_line(
            "probe_progress kind=train candidate_mb=64 phase=measure throughput=123.45",
        )
        .expect("formatted progress line should survive normalization");
        assert!(formatted.contains("[preflight:train]"));
        assert!(formatted.contains("candidate_mb=64"));
        assert!(formatted.contains("throughput=123.45 samples/s"));

        assert!(normalized_probe_output_line("probe_progress kind=train phase=start").is_none());
        assert!(normalized_probe_output_line("thread 'main' panicked").is_none());
        assert_eq!(
            normalized_probe_output_line(" useful line  "),
            Some("useful line".to_string())
        );
    }

    #[test]
    fn output_forwarder_collects_bytes_and_filters_without_error() {
        let handle = spawn_output_forwarder(
            Cursor::new(
                b"useful line\nthread 'main' panicked\nprobe_progress kind=train phase=start\n",
            ),
            false,
        );

        let collected = join_output_forwarder(handle, "stdout").expect("forwarder should succeed");
        assert_eq!(
            String::from_utf8(collected).expect("cursor bytes should stay utf8"),
            "useful line\nthread 'main' panicked\nprobe_progress kind=train phase=start\n"
        );
    }

    #[test]
    fn summarize_probe_failure_output_keeps_first_three_useful_lines() {
        let output = "thread 'main' panicked\n\nfirst useful\nprobe_progress foo\nsecond useful\nthird useful\nfourth useful\n";
        assert_eq!(
            summarize_probe_failure_output(output),
            "first useful | second useful | third useful"
        );
        assert!(summarize_probe_failure_output("probe_progress foo\n").is_empty());
    }

    #[test]
    fn probe_failure_detail_prefers_oom_then_stderr_then_stdout_then_fallback() {
        let oom = probe_failure_detail(ProbeStatus::Oom, "", "", Some(9));
        assert!(oom.contains("libtorch/cuda oom"));

        let stderr = probe_failure_detail(
            ProbeStatus::BackendError,
            "stdout useful",
            "stderr useful\nextra",
            Some(2),
        );
        assert!(stderr.contains("stderr useful"));

        let stdout = probe_failure_detail(ProbeStatus::BackendError, "stdout useful", "", Some(3));
        assert!(stdout.contains("stdout useful"));

        let fallback = probe_failure_detail(ProbeStatus::BackendError, "", "", None);
        assert!(fallback.contains("probe child failed without structured result"));
    }

    #[test]
    fn child_output_preserves_status_and_streams() {
        let status = Command::new("true")
            .status()
            .expect("true command should exit successfully");
        let output = child_output(status, b"stdout".to_vec(), b"stderr".to_vec());

        assert!(output.status.success());
        assert_eq!(output.stdout, b"stdout");
        assert_eq!(output.stderr, b"stderr");
    }

    #[test]
    fn rl_probe_required_free_bytes_handles_disabled_and_active_thresholds() {
        let mut config = test_train_config();
        config.preflight.rl_probe_min_free_memory_bytes = 0;
        config.preflight.rl_probe_memory_headroom_ratio = 0.0;
        assert_eq!(rl_probe_required_free_bytes(&config), None);

        config.preflight.rl_probe_memory_headroom_ratio = 0.25;
        let required_by_ratio = rl_probe_required_free_bytes(&config);
        if let Some(total) = mem_total_bytes() {
            let expected = ((total as f64) * 0.25).ceil() as u64;
            assert_eq!(required_by_ratio, Some(expected));
        } else {
            assert_eq!(required_by_ratio, None);
        }

        config.preflight.rl_probe_min_free_memory_bytes = u64::MAX;
        assert_eq!(rl_probe_required_free_bytes(&config), Some(u64::MAX));
    }

    #[test]
    fn wait_for_probe_child_returns_status_for_fast_exit() {
        let mut child = Command::new("true")
            .spawn()
            .expect("true command should spawn");
        let interrupted = AtomicBool::new(false);

        let status = wait_for_probe_child(&mut child, &interrupted)
            .expect("wait should succeed")
            .expect("child should exit normally");

        assert!(status.success());
    }

    #[test]
    fn wait_for_probe_child_kills_process_when_interrupted() {
        let mut child = Command::new("sleep")
            .arg("1")
            .spawn()
            .expect("sleep command should spawn");
        let interrupted = AtomicBool::new(true);

        let status = wait_for_probe_child(&mut child, &interrupted).expect("wait should succeed");

        assert!(status.is_none());
    }

    #[test]
    fn write_and_read_probe_result_round_trip_and_error_details() {
        let path = unique_test_path("round_trip", "json");
        let result = ProbeResult {
            kind: ProbeKind::Validation,
            candidate_microbatch: 96,
            status: ProbeStatus::Success,
            measured_samples_per_second: Some(321.5),
            elapsed_seconds: Some(1.25),
            detail: "all good".to_string(),
        };

        write_probe_result(&path, &result).expect("probe result should write");
        let parsed = read_probe_result(&path).expect("probe result should read");
        assert_eq!(parsed, result);

        fs::remove_file(&path).expect("round-trip result file should be removable");

        let missing_err = read_probe_result(&path).expect_err("missing file should error");
        assert!(missing_err.contains("failed to read probe result"));
        assert!(missing_err.contains(path.to_string_lossy().as_ref()));

        let invalid_path = unique_test_path("invalid_json", "json");
        fs::write(&invalid_path, "not-json").expect("invalid json fixture should write");
        let parse_err = read_probe_result(&invalid_path).expect_err("invalid json should error");
        assert!(parse_err.contains("failed to parse probe result"));
        assert!(parse_err.contains(invalid_path.to_string_lossy().as_ref()));
        fs::remove_file(&invalid_path).expect("invalid fixture file should be removable");
    }

    #[test]
    fn probe_result_paths_embed_kind_candidate_and_attempt() {
        let bc = BcArtifactPaths::new(Path::new("/tmp/hydra-out"), 0);
        let rl = RlArtifactPaths::new(Path::new("/tmp/hydra-out"), 0);

        let bc_path = probe_result_path(&bc, ProbeKind::Train, 128, 2);
        let rl_path = rl_probe_result_path(&rl, ProbeKind::RlGames, 64, 1);

        assert!(bc_path.ends_with("preflight_probe_train_128_2.json"));
        assert!(rl_path.ends_with("preflight_probe_rl_games_64_1.json"));
    }

    #[test]
    fn execute_probe_request_fails_fast_for_unsupported_config_extension() {
        let config_path = unique_test_path("config_extension", "txt");
        fs::write(&config_path, "ignored").expect("config fixture should write");
        let result_path = unique_test_path("probe_result", "json");
        let request = ProbeRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 32,
            warmup_steps: 1,
            measure_steps: 1,
        };

        let err = execute_probe_request(&config_path, request, &result_path, |_| {
            panic!("classify_probe_detail should not run on config read failure")
        })
        .expect_err("unsupported config extension should fail before spawning child");

        assert!(err.contains("unsupported config extension"));
        assert_eq!(config_path.extension(), Some(OsStr::new("txt")));

        fs::remove_file(&config_path).expect("config fixture should be removable");
        if result_path.exists() {
            fs::remove_file(&result_path).expect("unexpected result fixture should be removable");
        }
    }

    #[test]
    fn normalized_output_line_keeps_blank_lines_trimmed_to_empty_string() {
        assert_eq!(normalized_probe_output_line("   \n"), Some(String::new()));
    }

    #[test]
    fn probe_failure_detail_uses_stdout_when_stderr_has_only_suppressed_noise() {
        let detail = probe_failure_detail(
            ProbeStatus::BackendError,
            "stdout useful",
            "thread 'main' panicked\nstack backtrace:",
            Some(7),
        );
        assert!(detail.contains("stdout useful"));
    }
}
