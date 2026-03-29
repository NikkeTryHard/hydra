#[cfg(not(test))]
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::process::Command;
use std::process::ExitStatus;
#[cfg(not(test))]
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
#[cfg(not(test))]
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;
#[cfg(not(test))]
use std::time::Instant;

use serde::{Deserialize, Serialize};

use hydra_train::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use super::artifacts::{BcArtifactPaths, RlArtifactPaths};
use super::presentation::format_probe_progress_line;
#[cfg(not(test))]
use super::presentation::with_utc_timestamp;
use super::probe_request::{ProbeBatchRequest, ProbeRequest};
use super::probe_summary::probe_kind_name;

#[cfg(not(test))]
fn probe_child_executable() -> Result<PathBuf, String> {
    #[cfg(target_os = "linux")]
    {
        let proc_self = PathBuf::from("/proc/self/exe");
        if proc_self.exists() {
            return Ok(proc_self);
        }
    }

    let current = env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?;
    if current.exists() {
        Ok(current)
    } else {
        Err(format!(
            "current executable path does not exist: {}",
            current.display()
        ))
    }
}

#[cfg(not(test))]
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

#[cfg(not(test))]
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

fn build_probe_failure_result(
    request: ProbeRequest,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> ProbeResult {
    let combined = format!("stdout={stdout} stderr={stderr}");
    let status = classify_probe_detail(&combined);
    let detail = probe_failure_detail(status.clone(), stdout, stderr, exit_code);
    ProbeResult {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        status,
        measured_samples_per_second: None,
        elapsed_seconds: None,
        detail,
    }
}

fn recover_probe_batch_results(
    batch: ProbeBatchRequest,
    results_path: &Path,
    status: ExitStatus,
    stdout: &[u8],
    stderr: &[u8],
    classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<Vec<ProbeResult>, String> {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    let stdout = stdout.trim();
    let stderr = stderr.trim();

    if results_path.exists() {
        let artifact = read_probe_batch_artifact(results_path)?;
        fs::remove_file(results_path).ok();
        if status.success() || artifact.is_finished() {
            return Ok(artifact.replay_ordered_results().cloned().collect());
        }

        let mut recovered = artifact.completed_results_before_failure().to_vec();
        let has_recorded_failure = recovered
            .last()
            .is_some_and(|result| result.status != ProbeStatus::Success);
        if !has_recorded_failure && recovered.len() < batch.attempts.max(1) {
            recovered.push(build_probe_failure_result(
                batch.request,
                stdout,
                stderr,
                status.code(),
                classify_probe_detail,
            ));
        }
        return Ok(recovered);
    }

    Ok(vec![build_probe_failure_result(
        batch.request,
        stdout,
        stderr,
        status.code(),
        classify_probe_detail,
    )])
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct ProbeBatchArtifact {
    pub(super) finished: bool,
    pub(super) results: Vec<ProbeResult>,
}

impl ProbeBatchArtifact {
    pub(super) fn pending() -> Self {
        Self {
            finished: false,
            results: Vec::new(),
        }
    }

    pub(super) fn is_finished(&self) -> bool {
        self.finished
    }

    pub(super) fn push_result(&mut self, result: ProbeResult) {
        if !self.finished {
            self.results.push(result);
        }
    }

    pub(super) fn mark_finished(&mut self) {
        self.finished = true;
    }

    #[cfg(test)]
    pub(super) fn from_results(results: Vec<ProbeResult>, finished: bool) -> Self {
        Self { finished, results }
    }

    pub(super) fn replay_ordered_results(&self) -> impl Iterator<Item = &ProbeResult> {
        self.results.iter()
    }

    pub(super) fn completed_results_before_failure(&self) -> &[ProbeResult] {
        let len = self
            .results
            .iter()
            .position(|result| result.status != ProbeStatus::Success)
            .map(|index| index + 1)
            .unwrap_or(self.results.len());
        &self.results[..len]
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

pub(super) fn write_probe_batch_artifact(
    path: &Path,
    artifact: &ProbeBatchArtifact,
) -> Result<(), String> {
    let json = serde_json::to_string(artifact).map_err(|err| {
        format!(
            "failed to serialize probe batch artifact {}: {err}",
            path.display()
        )
    })?;
    fs::write(path, json).map_err(|err| {
        format!(
            "failed to write probe batch artifact {}: {err}",
            path.display()
        )
    })
}

pub(super) fn read_probe_batch_artifact(path: &Path) -> Result<ProbeBatchArtifact, String> {
    let raw = fs::read_to_string(path).map_err(|err| {
        format!(
            "failed to read probe batch artifact {}: {err}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse probe batch artifact {}: {err}",
            path.display()
        )
    })
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

pub(super) fn probe_batch_results_path(result_path: &Path) -> PathBuf {
    match result_path.extension() {
        Some(extension) => {
            result_path.with_extension(format!("batch.{}", extension.to_string_lossy()))
        }
        None => result_path.with_extension("batch"),
    }
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
    #[cfg_attr(test, allow(unused_variables))] classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<ProbeResult, String> {
    #[cfg(test)]
    {
        let config = super::config::read_config(config_path)?;
        fs::remove_file(result_path).ok();
        let use_actor_model = matches!(
            config.precision_mode,
            super::config::PrecisionMode::Bf16Autocast
        ) && matches!(request.kind, ProbeKind::Train | ProbeKind::Validation);
        if use_actor_model {
            super::preflight_runtime::run_probe_only_with_test_model_config_result(
                &config,
                &hydra_train::model::HydraModelConfig::new(1)
                    .with_input_channels(hydra_train::config::INPUT_CHANNELS)
                    .with_hidden_channels(4)
                    .with_num_groups(4)
                    .with_se_bottleneck(1),
                request,
            )
        } else {
            let child_request =
                super::config::ProbeChildRequest::Single(super::config::ProbeSingleChildRequest {
                    request: super::config::ProbeCliRequest {
                        kind: request.kind,
                        candidate_microbatch: request.candidate_microbatch,
                        warmup_steps: Some(request.warmup_steps),
                        measure_steps: Some(request.measure_steps),
                    },
                    result_path: result_path.to_path_buf(),
                    manifest_cache_path: Some(
                        super::artifacts::PreflightPaths::new(&BcArtifactPaths::new(
                            &config.output_dir,
                            0,
                        ))
                        .manifest_cache_path,
                    ),
                });
            super::preflight_runtime::run_probe_child_mode_result(&config, Some(child_request))?
                .ok_or_else(|| "internal probe child request missing in test execution".to_string())
        }
    }

    #[cfg(not(test))]
    {
        #[cfg(not(test))]
        let _config = super::config::read_config(config_path)?;
        let manifest_cache_path =
            super::artifacts::PreflightPaths::new(&BcArtifactPaths::new(&_config.output_dir, 0))
                .manifest_cache_path;

        fs::remove_file(result_path).ok();
        let interrupted = interrupt_flag()?;
        interrupted.store(false, Ordering::SeqCst);
        let child_exe = probe_child_executable()?;
        let mut child = Command::new(&child_exe)
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
            .arg("--probe-manifest-cache-path")
            .arg(&manifest_cache_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| {
                format!(
                    "failed to spawn preflight probe child {}: {err}",
                    child_exe.display()
                )
            })?;
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
        Ok(build_probe_failure_result(
            request,
            stdout,
            stderr,
            output.status.code(),
            classify_probe_detail,
        ))
    }
}

pub(super) fn execute_probe_request_batch(
    config_path: &Path,
    batch: ProbeBatchRequest,
    results_path: &Path,
    #[cfg_attr(test, allow(unused_variables))] classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<Vec<ProbeResult>, String> {
    #[cfg(test)]
    {
        let config = super::config::read_config(config_path)?;
        fs::remove_file(results_path).ok();
        let child_request =
            super::config::ProbeChildRequest::Batch(super::config::ProbeBatchChildRequest {
                request: super::config::ProbeCliRequest {
                    kind: batch.request.kind,
                    candidate_microbatch: batch.request.candidate_microbatch,
                    warmup_steps: Some(batch.request.warmup_steps),
                    measure_steps: Some(batch.request.measure_steps),
                },
                attempts: batch.attempts,
                results_path: results_path.to_path_buf(),
                manifest_cache_path: Some(
                    super::artifacts::PreflightPaths::new(&BcArtifactPaths::new(
                        &config.output_dir,
                        0,
                    ))
                    .manifest_cache_path,
                ),
            });
        let artifact = super::preflight_runtime::run_probe_child_batch_mode_result(
            &config,
            Some(child_request),
        )?
        .ok_or_else(|| {
            "internal probe batch child request missing in test execution".to_string()
        })?;
        let success_status = Command::new("true")
            .status()
            .map_err(|err| format!("failed to build synthetic batch success status: {err}"))?;
        if !results_path.exists() {
            write_probe_batch_artifact(results_path, &artifact)?;
        }
        recover_probe_batch_results(
            batch,
            results_path,
            success_status,
            &[],
            &[],
            classify_probe_detail,
        )
    }

    #[cfg(not(test))]
    {
        let config = super::config::read_config(config_path)?;
        let manifest_cache_path =
            super::artifacts::PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0))
                .manifest_cache_path;

        fs::remove_file(results_path).ok();
        let interrupted = interrupt_flag()?;
        interrupted.store(false, Ordering::SeqCst);
        let child_exe = probe_child_executable()?;
        let mut child = Command::new(&child_exe)
            .arg(config_path)
            .arg("--probe-kind")
            .arg(probe_kind_name(batch.request.kind))
            .arg("--probe-candidate-microbatch")
            .arg(batch.request.candidate_microbatch.to_string())
            .arg("--probe-warmup-steps")
            .arg(batch.request.warmup_steps.to_string())
            .arg("--probe-measure-steps")
            .arg(batch.request.measure_steps.to_string())
            .arg("--probe-attempts")
            .arg(batch.attempts.to_string())
            .arg("--probe-results-path")
            .arg(results_path)
            .arg("--probe-manifest-cache-path")
            .arg(&manifest_cache_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| {
                format!(
                    "failed to spawn preflight probe child {}: {err}",
                    child_exe.display()
                )
            })?;
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
            batch.request.kind,
            batch.request.candidate_microbatch,
        );
        if wait_for_probe_child(&mut child, interrupted.as_ref())?.is_none() {
            fs::remove_file(results_path).ok();
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
        recover_probe_batch_results(
            batch,
            results_path,
            status,
            &stdout,
            &stderr,
            classify_probe_detail,
        )
    }
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
    use crate::test_loose_replay_fixtures::write_real_probe_fixture;

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
            precision_mode: crate::config::PrecisionMode::Fp32,
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

    fn write_probe_config(
        name: &str,
        precision_mode: crate::config::PrecisionMode,
        replay_path: &Path,
        output_dir: &Path,
    ) -> PathBuf {
        let mut config = test_train_config();
        config.data_dir = replay_path.to_path_buf();
        config.output_dir = output_dir.to_path_buf();
        config.batch_size = 1;
        config.train_fraction = 1.0;
        config.device = "cpu".to_string();
        config.precision_mode = precision_mode;

        let config_path = unique_test_path(name, "yaml");
        let config_yaml = serde_yaml::to_string(&config).expect("serialize valid probe config");
        fs::write(&config_path, config_yaml).expect("write valid probe config");
        config_path
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

    fn sample_probe_result(
        kind: ProbeKind,
        candidate_microbatch: usize,
        status: ProbeStatus,
        detail: &str,
    ) -> ProbeResult {
        ProbeResult {
            kind,
            candidate_microbatch,
            status,
            measured_samples_per_second: Some(candidate_microbatch as f64 * 10.0),
            elapsed_seconds: Some(1.0),
            detail: detail.to_string(),
        }
    }

    fn sample_probe_batch_request(
        kind: ProbeKind,
        candidate_microbatch: usize,
    ) -> ProbeBatchRequest {
        ProbeBatchRequest {
            request: ProbeRequest {
                kind,
                candidate_microbatch,
                warmup_steps: 1,
                measure_steps: 1,
            },
            attempts: 3,
        }
    }

    #[test]
    fn probe_batch_artifact_round_trip_preserves_partial_progress_shape() {
        let path = unique_test_path("batch_round_trip", "json");
        let mut artifact = ProbeBatchArtifact::pending();
        artifact.push_result(sample_probe_result(
            ProbeKind::Train,
            128,
            ProbeStatus::Success,
            "attempt 1 ok",
        ));

        write_probe_batch_artifact(&path, &artifact).expect("batch artifact should write");

        let raw = fs::read_to_string(&path).expect("batch artifact json should read");
        assert_eq!(
            raw,
            r#"{"finished":false,"results":[{"kind":"train","candidate_microbatch":128,"status":"success","measured_samples_per_second":1280.0,"elapsed_seconds":1.0,"detail":"attempt 1 ok"}]}"#
        );

        let parsed = read_probe_batch_artifact(&path).expect("batch artifact should parse");
        assert_eq!(parsed, artifact);
        assert!(!parsed.is_finished());
        fs::remove_file(&path).expect("batch artifact file should be removable");
    }

    #[test]
    fn probe_batch_artifact_round_trip_preserves_finished_marker() {
        let path = unique_test_path("batch_finished_round_trip", "json");
        let mut artifact = ProbeBatchArtifact::pending();
        artifact.push_result(sample_probe_result(
            ProbeKind::Validation,
            96,
            ProbeStatus::Success,
            "attempt 1 ok",
        ));
        artifact.mark_finished();

        write_probe_batch_artifact(&path, &artifact).expect("finished batch artifact should write");
        let parsed =
            read_probe_batch_artifact(&path).expect("finished batch artifact should parse");

        assert!(parsed.is_finished());
        assert_eq!(parsed.results.len(), 1);
        fs::remove_file(&path).expect("finished batch artifact file should be removable");
    }

    #[test]
    fn probe_batch_artifact_replays_results_in_original_order() {
        let artifact = ProbeBatchArtifact::from_results(
            vec![
                sample_probe_result(ProbeKind::Train, 64, ProbeStatus::Success, "first"),
                sample_probe_result(ProbeKind::Train, 64, ProbeStatus::Success, "second"),
                sample_probe_result(ProbeKind::Train, 64, ProbeStatus::Success, "third"),
            ],
            true,
        );

        let details = artifact
            .replay_ordered_results()
            .map(|result| result.detail.as_str())
            .collect::<Vec<_>>();

        assert_eq!(details, vec!["first", "second", "third"]);
    }

    #[test]
    fn probe_batch_artifact_completed_results_before_failure_stops_at_first_failure() {
        let artifact = ProbeBatchArtifact::from_results(
            vec![
                sample_probe_result(ProbeKind::Train, 64, ProbeStatus::Success, "attempt 1 ok"),
                sample_probe_result(ProbeKind::Train, 64, ProbeStatus::Success, "attempt 2 ok"),
                sample_probe_result(
                    ProbeKind::Train,
                    64,
                    ProbeStatus::BackendError,
                    "attempt 3 failed",
                ),
                sample_probe_result(
                    ProbeKind::Train,
                    64,
                    ProbeStatus::Success,
                    "attempt 4 later",
                ),
            ],
            false,
        );

        let visible = artifact
            .completed_results_before_failure()
            .iter()
            .map(|result| result.detail.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            visible,
            vec!["attempt 1 ok", "attempt 2 ok", "attempt 3 failed"]
        );
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
    fn probe_batch_results_path_avoids_legacy_single_attempt_filename_collisions() {
        let bc = BcArtifactPaths::new(Path::new("/tmp/hydra-out"), 0);
        let legacy_attempt_zero = probe_result_path(&bc, ProbeKind::Train, 128, 0);
        let legacy_attempt_one = probe_result_path(&bc, ProbeKind::Train, 128, 1);
        let batch_path = probe_batch_results_path(&legacy_attempt_zero);

        assert!(batch_path.ends_with("preflight_probe_train_128_0.batch.json"));
        assert_ne!(batch_path, legacy_attempt_zero);
        assert_ne!(batch_path, legacy_attempt_one);
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
    fn execute_probe_request_returns_direct_success_result_for_loose_replay_and_bf16_paths() {
        for (label, precision_mode) in [
            ("fp32", crate::config::PrecisionMode::Fp32),
            ("bf16", crate::config::PrecisionMode::Bf16Autocast),
        ] {
            let (root, replay_path, result_path) = write_real_probe_fixture(label);
            let output_dir = root.join(format!("out-{label}"));
            let config_path = write_probe_config(
                &format!("execute_probe_request_{label}"),
                precision_mode,
                &replay_path,
                &output_dir,
            );

            let result = execute_probe_request(
                &config_path,
                ProbeRequest {
                    kind: ProbeKind::Train,
                    candidate_microbatch: 1,
                    warmup_steps: 1,
                    measure_steps: 1,
                },
                &result_path,
                |_| panic!("classify_probe_detail should not run on success"),
            )
            .expect("real loose replay probe should succeed");

            assert_eq!(result.kind, ProbeKind::Train);
            assert_eq!(result.status, ProbeStatus::Success);
            assert_eq!(result.candidate_microbatch, 1);
            assert!(result.measured_samples_per_second.is_some());
            assert!(result.elapsed_seconds.is_some());
            assert_eq!(result.detail, "stable train probe on real dataset");
            assert!(!result_path.exists());

            fs::remove_file(&config_path).expect("config fixture should be removable");
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn recover_probe_batch_results_preserves_partial_successes_and_one_synthesized_failure() {
        let results_path = unique_test_path("batch_partial_recovery", "json");
        let batch = sample_probe_batch_request(ProbeKind::Validation, 64);
        write_probe_batch_artifact(
            &results_path,
            &ProbeBatchArtifact::from_results(
                vec![
                    sample_probe_result(
                        ProbeKind::Validation,
                        64,
                        ProbeStatus::Success,
                        "attempt 1 ok",
                    ),
                    sample_probe_result(
                        ProbeKind::Validation,
                        64,
                        ProbeStatus::Success,
                        "attempt 2 ok",
                    ),
                ],
                false,
            ),
        )
        .expect("partial batch artifact should write");

        let failed_status = Command::new("false")
            .status()
            .expect("false command should exit with failure");
        let recovered = recover_probe_batch_results(
            batch,
            &results_path,
            failed_status,
            b"stdout says replay loader failed",
            b"stderr says replay loader failed",
            |_| ProbeStatus::DataError,
        )
        .expect("partial batch recovery should succeed");

        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].status, ProbeStatus::Success);
        assert_eq!(recovered[1].status, ProbeStatus::Success);
        assert_eq!(recovered[2].status, ProbeStatus::DataError);
        assert!(recovered[2]
            .detail
            .contains("stderr says replay loader failed"));
        assert!(recovered[..2]
            .iter()
            .all(|result| result.status == ProbeStatus::Success));
        assert!(!results_path.exists());
    }

    #[test]
    fn execute_probe_request_batch_uses_finished_artifact_order_without_classifying_failure() {
        let config_path = unique_test_path("batch_ordered_config", "yaml");
        fs::write(
            &config_path,
            serde_yaml::to_string(&test_train_config()).expect("serialize dummy config"),
        )
        .expect("dummy config should write");
        let results_path = unique_test_path("batch_ordered_results", "json");
        let batch = sample_probe_batch_request(ProbeKind::Train, 96);
        write_probe_batch_artifact(
            &results_path,
            &ProbeBatchArtifact::from_results(
                vec![
                    sample_probe_result(ProbeKind::Train, 96, ProbeStatus::Success, "first"),
                    sample_probe_result(ProbeKind::Train, 96, ProbeStatus::Success, "second"),
                    sample_probe_result(ProbeKind::Train, 96, ProbeStatus::Success, "third"),
                ],
                true,
            ),
        )
        .expect("finished batch artifact should write");

        let recovered = recover_probe_batch_results(
            batch,
            &results_path,
            Command::new("true")
                .status()
                .expect("true command should exit successfully"),
            b"",
            b"",
            |_| panic!("finished artifact recovery should not classify failure"),
        )
        .expect("finished batch recovery should replay ordered results");

        assert_eq!(
            recovered
                .iter()
                .map(|result| result.detail.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second", "third"]
        );
        assert!(!results_path.exists());
        fs::remove_file(&config_path).expect("dummy config should be removable");
    }

    #[test]
    fn normalized_output_line_keeps_blank_lines_trimmed_to_empty_string() {
        assert_eq!(normalized_probe_output_line("   \n"), Some(String::new()));
    }

    #[test]
    fn normalized_output_line_suppresses_indented_raw_probe_progress() {
        assert!(normalized_probe_output_line("   probe_progress kind=train phase=start").is_none());
    }

    #[test]
    fn normalized_output_line_trims_non_progress_text_without_suppressing_it() {
        assert_eq!(
            normalized_probe_output_line("   useful stderr line   \n"),
            Some("useful stderr line".to_string())
        );
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

    #[test]
    fn probe_failure_detail_uses_filtered_stderr_summary_when_stdout_is_more_verbose() {
        let detail = probe_failure_detail(
            ProbeStatus::BackendError,
            "stdout useful\nstdout extra",
            "probe_progress kind=train phase=start\n\nfirst useful\nthread 'main' panicked\nsecond useful\nthird useful\nfourth useful",
            Some(11),
        );

        assert!(detail.contains("first useful | second useful | third useful"));
        assert!(!detail.contains("fourth useful"));
        assert!(!detail.contains("stdout useful"));
    }

    #[test]
    fn probe_failure_detail_falls_back_to_process_status_summary_when_outputs_are_empty() {
        let detail = probe_failure_detail(ProbeStatus::BackendError, "", "", Some(17));

        assert_eq!(
            detail,
            "probe process status=Some(17) detail=probe child failed without structured result"
        );
    }

    #[test]
    fn probe_failure_detail_uses_process_status_summary_when_outputs_and_exit_code_are_missing() {
        let detail = probe_failure_detail(ProbeStatus::DataError, "", "", None);

        assert_eq!(
            detail,
            "probe process status=None detail=probe child failed without structured result"
        );
    }

    #[test]
    fn join_output_forwarder_reports_panic_with_stream_name() {
        let handle = thread::spawn(|| -> Result<Vec<u8>, String> {
            panic!("boom");
        });

        let err = join_output_forwarder(handle, "stderr")
            .expect_err("panicking forwarder should surface a stream-specific error");

        assert_eq!(err, "preflight probe stderr forwarder panicked");
    }
}
