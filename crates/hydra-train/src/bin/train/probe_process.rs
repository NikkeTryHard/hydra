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
