#![allow(
    missing_docs,
    reason = "migrated train binary helpers preserve existing internal surface"
)]

#[cfg(not(test))]
use colored::Colorize;
#[cfg(not(test))]
use std::env;
#[cfg(all(not(test), target_os = "linux"))]
use std::ffi::c_int;
#[cfg(any(not(test), feature = "cuda-graph", target_os = "linux"))]
use std::fs;
#[cfg(not(test))]
use std::io::{BufRead, BufReader, Read};
#[cfg(all(not(test), target_os = "linux"))]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
#[cfg(not(test))]
use std::process::Child;
#[cfg(not(test))]
use std::process::Command;
#[cfg(not(test))]
use std::process::ExitStatus;
#[cfg(not(test))]
use std::process::Stdio;
#[cfg(not(test))]
use std::sync::OnceLock;
#[cfg(not(test))]
use std::sync::atomic::AtomicBool;
#[cfg(not(test))]
use std::sync::atomic::Ordering;
#[cfg(not(test))]
use std::sync::{Arc, Mutex};
#[cfg(not(test))]
use std::thread;
#[cfg(not(test))]
use std::time::Duration;
#[cfg(not(test))]
use std::time::Instant;

#[cfg(not(test))]
use indicatif::{ProgressBar, ProgressStyle};

#[cfg(not(test))]
use super::system_metrics::read_gpu_telemetry;
#[cfg(any(not(test), test))]
use hydra_train_runtime::config::read_config;
use hydra_train_runtime::preflight::PreflightConfig;
#[cfg(not(test))]
use hydra_train_runtime::preflight::ProbeKind;
use hydra_train_runtime::preflight::{ProbeResult, ProbeStatus};

#[cfg(not(test))]
use super::artifacts::{BcArtifactPaths, PreflightPaths};
#[cfg(all(test, feature = "cuda-graph"))]
use super::presentation::format_probe_progress_line;
#[cfg(not(test))]
use super::presentation::{
    format_probe_spinner_finish_message, format_probe_spinner_message, make_spinner,
};
#[cfg(not(test))]
use super::probe_summary::probe_kind_name;
#[cfg(not(test))]
use super::probe_transport::recover_probe_batch_results;
#[cfg(all(test, feature = "cuda-graph"))]
use super::probe_transport::should_suppress_probe_output_line;
#[cfg(test)]
use super::probe_transport::write_probe_batch_artifact;
#[cfg(not(test))]
use super::probe_transport::{build_probe_failure_result, read_probe_result};
use hydra_train_runtime::probe_request::{ProbeBatchRequest, ProbeRequest};

#[cfg(not(test))]
const HYDRA_PROBE_CHILD_EXE: &str = "HYDRA_PROBE_CHILD_EXE";

#[cfg(not(test))]
fn probe_child_executable() -> Result<PathBuf, String> {
    if let Ok(path) = env::var(HYDRA_PROBE_CHILD_EXE) {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "{HYDRA_PROBE_CHILD_EXE} points to missing executable: {}",
            path.display()
        ));
    }

    let current = env::current_exe().map_err(|err| format!("current_exe failed: {err}"))?;
    if let Some(train_bin) = train_binary_sibling(&current) {
        return Ok(train_bin);
    }

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
fn train_binary_sibling(current: &Path) -> Option<PathBuf> {
    let deps_dir = current.parent()?;
    if deps_dir.file_name().and_then(|name| name.to_str()) != Some("deps") {
        return None;
    }
    let train_bin = deps_dir.parent()?.join("train");
    train_bin.exists().then_some(train_bin)
}

#[cfg(not(test))]
fn propagate_probe_runtime_env(cmd: &mut Command) {
    const PASSTHROUGH_VARS: &[&str] = &[
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "LIBTORCH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PATH",
    ];

    for key in PASSTHROUGH_VARS {
        if let Ok(value) = env::var(key) {
            cmd.env(key, value);
        }
    }

    cmd.env("OMP_NUM_THREADS", "1");
    cmd.env("MKL_NUM_THREADS", "1");
    if let Ok(v) = env::var("CUDA_BINARY_LOADER_THREAD_COUNT") {
        cmd.env("CUDA_BINARY_LOADER_THREAD_COUNT", v);
    } else {
        cmd.env("CUDA_BINARY_LOADER_THREAD_COUNT", "2");
    }
    if let Ok(v) = env::var("CUDA_CACHE_MAXSIZE") {
        cmd.env("CUDA_CACHE_MAXSIZE", v);
    } else {
        cmd.env("CUDA_CACHE_MAXSIZE", "4294967296");
    }
}

#[cfg(not(test))]
pub fn interrupt_flag() -> Result<Arc<AtomicBool>, String> {
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

#[cfg(all(not(test), target_os = "linux"))]
const PR_SET_PDEATHSIG: c_int = 1;
#[cfg(all(not(test), target_os = "linux"))]
const SIGKILL: c_int = 9;
#[cfg(all(not(test), target_os = "linux"))]
const SIGTERM: c_int = 15;

#[cfg(all(not(test), target_os = "linux"))]
unsafe extern "C" {
    fn kill(pid: c_int, sig: c_int) -> c_int;
    fn prctl(option: c_int, arg2: c_int, arg3: c_int, arg4: c_int, arg5: c_int) -> c_int;
    fn setpgid(pid: c_int, pgid: c_int) -> c_int;
}

#[cfg(all(not(test), target_os = "linux"))]
fn configure_probe_child_process_group() -> std::io::Result<()> {
    unsafe {
        if setpgid(0, 0) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(all(not(test), target_os = "linux"))]
fn terminate_probe_child_process_group(pid: u32) {
    let pgid = -(pid as c_int);
    unsafe {
        kill(pgid, SIGTERM);
    }
    thread::sleep(Duration::from_millis(50));
    unsafe {
        kill(pgid, SIGKILL);
    }
}

#[cfg(not(test))]
struct ProbeChildGuard {
    child: Option<Child>,
}

#[cfg(not(test))]
impl ProbeChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn child_mut(&mut self) -> Result<&mut Child, String> {
        self.child
            .as_mut()
            .ok_or_else(|| "preflight probe child guard missing child".to_string())
    }

    fn terminate(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        #[cfg(target_os = "linux")]
        terminate_probe_child_process_group(child.id());
        child.kill().ok();
        child.wait().ok();
    }

    fn disarm(mut self) -> Result<Child, String> {
        self.child
            .take()
            .ok_or_else(|| "preflight probe child guard missing child".to_string())
    }
}

#[cfg(not(test))]
impl Drop for ProbeChildGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(all(not(test), target_os = "linux"))]
pub fn spawn_guarded_child_for_test(cmd: &mut Command) -> Result<u32, String> {
    unsafe {
        cmd.pre_exec(configure_probe_child_process_group);
    }
    let child = cmd
        .spawn()
        .map_err(|err| format!("failed to spawn test child: {err}"))?;
    let pid = child.id();
    let _guard = ProbeChildGuard::new(child);
    Ok(pid)
}

#[cfg(all(test, target_os = "linux"))]
mod test_child_guard {
    use std::ffi::c_int;
    use std::process::{Child, Command};
    use std::thread;
    use std::time::Duration;

    const SIGKILL: c_int = 9;
    const SIGTERM: c_int = 15;

    unsafe extern "C" {
        fn kill(pid: c_int, sig: c_int) -> c_int;
    }

    fn terminate_probe_child_process_group(pid: u32) {
        let pgid = -(pid as c_int);
        unsafe {
            kill(pgid, SIGTERM);
        }
        thread::sleep(Duration::from_millis(50));
        unsafe {
            kill(pgid, SIGKILL);
        }
    }

    struct ProbeChildGuard {
        child: Option<Child>,
    }

    impl ProbeChildGuard {
        fn new(child: Child) -> Self {
            Self { child: Some(child) }
        }

        fn terminate(&mut self) {
            let Some(child) = self.child.as_mut() else {
                return;
            };
            terminate_probe_child_process_group(child.id());
            child.kill().ok();
            child.wait().ok();
        }
    }

    impl Drop for ProbeChildGuard {
        fn drop(&mut self) {
            self.terminate();
        }
    }

    pub fn spawn_guarded_child_for_test(cmd: &mut Command) -> Result<u32, String> {
        let child = cmd
            .spawn()
            .map_err(|err| format!("failed to spawn test child: {err}"))?;
        let pid = child.id();
        let _guard = ProbeChildGuard::new(child);
        Ok(pid)
    }
}

#[cfg(all(test, target_os = "linux"))]
pub use test_child_guard::spawn_guarded_child_for_test;

#[cfg(all(test, feature = "cuda-graph"))]
#[cfg_attr(
    all(test, feature = "cuda-graph"),
    allow(
        dead_code,
        reason = "feature-gated output normalizer is exercised by train-bin tests"
    )
)]
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

#[cfg(not(test))]
fn spawn_output_forwarder_with_spinner<R>(
    reader: R,
    stderr: bool,
    spinner_message: Arc<Mutex<String>>,
) -> thread::JoinHandle<Result<Vec<u8>, String>>
where
    R: Read + Send + 'static,
{
    spawn_output_forwarder_inner(reader, stderr, Some(spinner_message))
}

#[cfg(not(test))]
fn spawn_output_forwarder_inner<R>(
    reader: R,
    _stderr: bool,
    spinner_message: Option<Arc<Mutex<String>>>,
) -> thread::JoinHandle<Result<Vec<u8>, String>>
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
            if text.trim_start().starts_with("probe_progress ")
                && let Some(message) = format_probe_spinner_message(&text)
                && let Some(spinner_message) = spinner_message.as_ref()
            {
                set_probe_spinner_message(spinner_message, message)?;
            }
        }
        Ok(collected)
    })
}

#[cfg(not(test))]
fn set_probe_spinner_message(
    spinner_message: &Arc<Mutex<String>>,
    message: String,
) -> Result<(), String> {
    *spinner_message
        .lock()
        .map_err(|_| "preflight probe spinner state lock poisoned".to_string())? = message;
    Ok(())
}

#[cfg(not(test))]
fn sync_probe_spinner_message(
    spinner: &ProgressBar,
    spinner_message: &Arc<Mutex<String>>,
) -> Result<(), String> {
    let message = spinner_message
        .lock()
        .map_err(|_| "preflight probe spinner state lock poisoned".to_string())?
        .clone();
    spinner.set_message(message);
    Ok(())
}

#[cfg(not(test))]
fn finish_probe_spinner(spinner: &ProgressBar, message: String) {
    spinner.set_style(
        ProgressStyle::with_template("{msg}").expect("static probe finish template is valid"),
    );
    spinner.finish_with_message(message);
}

#[cfg(not(test))]
fn spawn_probe_spinner(
    kind: ProbeKind,
    candidate_microbatch: usize,
) -> Result<(Arc<Mutex<String>>, ProgressBar), String> {
    let spinner_message = Arc::new(Mutex::new(format!(
        "[preflight:{}] mb={} loading data...",
        probe_kind_name(kind),
        candidate_microbatch,
    )));
    let spinner = make_spinner("{spinner:.cyan} {msg} {elapsed_precise}")?;
    spinner.enable_steady_tick(Duration::from_millis(100));
    sync_probe_spinner_message(&spinner, &spinner_message)?;
    Ok((spinner_message, spinner))
}

#[cfg(not(test))]
fn wait_for_probe_child_with_spinner(
    child: &mut Child,
    interrupted: &AtomicBool,
    spinner: &ProgressBar,
    spinner_message: &Arc<Mutex<String>>,
) -> Result<Option<ExitStatus>, String> {
    loop {
        sync_probe_spinner_message(spinner, spinner_message)?;
        if interrupted.load(Ordering::SeqCst) {
            child.kill().ok();
            child.wait().ok();
            return Ok(None);
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                sync_probe_spinner_message(spinner, spinner_message)?;
                return Ok(Some(status));
            }
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

#[cfg(not(test))]
fn join_output_forwarder(
    handle: thread::JoinHandle<Result<Vec<u8>, String>>,
    stream_name: &str,
) -> Result<Vec<u8>, String> {
    handle
        .join()
        .map_err(|_| format!("preflight probe {stream_name} forwarder panicked"))?
}

#[cfg(not(test))]
fn format_gpu_memory_snapshot(gpu: Option<super::system_metrics::GpuTelemetry>) -> String {
    match gpu {
        Some(gpu) => format!(
            "gpu_mem_used_mb={:?} gpu_mem_free_mb={:?}",
            gpu.mem_used_mb, gpu.mem_free_mb
        ),
        None => "gpu_mem=unavailable".to_string(),
    }
}

#[cfg(not(test))]
fn append_probe_cleanup_log(
    stderr: &mut Vec<u8>,
    pid: u32,
    status: ExitStatus,
    elapsed_seconds: f64,
    gpu_before: Option<super::system_metrics::GpuTelemetry>,
    gpu_after: Option<super::system_metrics::GpuTelemetry>,
) {
    use std::io::Write as _;

    let _ = writeln!(
        stderr,
        "probe_cleanup pid={} status={} elapsed_seconds={:.3} before={} after={}",
        pid,
        status,
        elapsed_seconds,
        format_gpu_memory_snapshot(gpu_before),
        format_gpu_memory_snapshot(gpu_after),
    );
}

#[cfg(not(test))]
fn child_output(status: ExitStatus, stdout: Vec<u8>, stderr: Vec<u8>) -> std::process::Output {
    std::process::Output {
        status,
        stdout,
        stderr,
    }
}

#[cfg(not(test))]
struct ProbeChildRunOutput {
    output: std::process::Output,
    elapsed_seconds: f64,
    spinner: ProgressBar,
}

#[cfg(not(test))]
struct ProbeChildRunRequest<'a> {
    config_path: &'a Path,
    kind: ProbeKind,
    candidate_microbatch: usize,
    warmup_steps: usize,
    measure_steps: usize,
    result_flag: &'a str,
    result_path: &'a Path,
    attempts: Option<usize>,
    manifest_cache_path: &'a Path,
    discovery_summary_path: &'a Path,
    discovery_index_path: &'a Path,
}

#[cfg(not(test))]
fn run_probe_child_process(
    request: ProbeChildRunRequest<'_>,
) -> Result<ProbeChildRunOutput, String> {
    let interrupted = interrupt_flag()?;
    interrupted.store(false, Ordering::SeqCst);
    let probe_started = Instant::now();
    let (spinner_message, spinner) =
        spawn_probe_spinner(request.kind, request.candidate_microbatch)?;
    let child_exe = probe_child_executable()?;
    let mut child_cmd = Command::new(&child_exe);
    child_cmd
        .arg(request.config_path)
        .arg("--probe-kind")
        .arg(probe_kind_name(request.kind))
        .arg("--probe-candidate-microbatch")
        .arg(request.candidate_microbatch.to_string())
        .arg("--probe-warmup-steps")
        .arg(request.warmup_steps.to_string())
        .arg("--probe-measure-steps")
        .arg(request.measure_steps.to_string());
    if let Some(attempts) = request.attempts {
        child_cmd.arg("--probe-attempts").arg(attempts.to_string());
    }
    child_cmd
        .arg(request.result_flag)
        .arg(request.result_path)
        .arg("--probe-manifest-cache-path")
        .arg(request.manifest_cache_path)
        .arg("--probe-discovery-summary-path")
        .arg(request.discovery_summary_path)
        .arg("--probe-discovery-index-path")
        .arg(request.discovery_index_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(target_os = "linux")]
    unsafe {
        child_cmd.pre_exec(configure_probe_child_process_group);
    }
    propagate_probe_runtime_env(&mut child_cmd);
    let gpu_before_spawn = read_gpu_telemetry();
    let child = child_cmd.spawn().map_err(|err| {
        format!(
            "failed to spawn preflight probe child {}: {err}",
            child_exe.display()
        )
    })?;
    let child_pid = child.id();
    let mut child_guard = ProbeChildGuard::new(child);
    let stdout_handle =
        child_guard.child_mut()?.stdout.take().map(|stdout| {
            spawn_output_forwarder_with_spinner(stdout, false, spinner_message.clone())
        });
    let stderr_handle =
        child_guard.child_mut()?.stderr.take().map(|stderr| {
            spawn_output_forwarder_with_spinner(stderr, true, spinner_message.clone())
        });

    let interrupted_run = wait_for_probe_child_with_spinner(
        child_guard.child_mut()?,
        interrupted.as_ref(),
        &spinner,
        &spinner_message,
    )?
    .is_none();
    if interrupted_run {
        fs::remove_file(request.result_path).ok();
        interrupted.store(true, Ordering::SeqCst);
        child_guard.terminate();
        if let Some(handle) = stdout_handle {
            let _ = join_output_forwarder(handle, "stdout");
        }
        if let Some(handle) = stderr_handle {
            let _ = join_output_forwarder(handle, "stderr");
        }
        finish_probe_spinner(
            &spinner,
            format!(
                "{} [{}] mb={} interrupted ({:.1}s)",
                "✘".red(),
                probe_kind_name(request.kind),
                request.candidate_microbatch,
                probe_started.elapsed().as_secs_f64(),
            ),
        );
        return Err("preflight interrupted; probe child terminated".to_string());
    }

    interrupted.store(false, Ordering::SeqCst);
    let mut child = child_guard.disarm()?;
    let stdout = match stdout_handle {
        Some(handle) => join_output_forwarder(handle, "stdout")?,
        None => Vec::new(),
    };
    let mut stderr = match stderr_handle {
        Some(handle) => join_output_forwarder(handle, "stderr")?,
        None => Vec::new(),
    };
    let status = child
        .wait()
        .map_err(|err| format!("failed to reap preflight probe child {child_pid}: {err}"))?;
    let elapsed_seconds = probe_started.elapsed().as_secs_f64();
    let gpu_after_exit = read_gpu_telemetry();
    append_probe_cleanup_log(
        &mut stderr,
        child_pid,
        status,
        elapsed_seconds,
        gpu_before_spawn,
        gpu_after_exit,
    );
    let output = child_output(status, stdout, stderr);
    Ok(ProbeChildRunOutput {
        output,
        elapsed_seconds,
        spinner,
    })
}

pub fn mem_available_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub fn mem_total_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

pub fn rl_probe_required_free_bytes(preflight: &PreflightConfig) -> Option<u64> {
    if preflight.rl_probe_min_free_memory_bytes == 0
        && preflight.rl_probe_memory_headroom_ratio <= 0.0
    {
        return None;
    }
    let total = mem_total_bytes()?;
    let ratio_floor =
        ((total as f64) * preflight.rl_probe_memory_headroom_ratio.max(0.0)).ceil() as u64;
    Some(preflight.rl_probe_min_free_memory_bytes.max(ratio_floor))
}

pub fn execute_probe_request(
    config_path: &Path,
    request: ProbeRequest,
    result_path: &Path,
    #[cfg_attr(
        test,
        allow(
            unused_variables,
            reason = "test path uses the in-process probe runner instead of stderr classification"
        )
    )]
    classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<ProbeResult, String> {
    #[cfg(test)]
    {
        let _ = (config_path, request, result_path, classify_probe_detail);
        Err(
            "hydra-train-exec in-process probe execution is not available in crate-local tests"
                .to_string(),
        )
    }

    #[cfg(not(test))]
    {
        #[cfg(not(test))]
        let _config = read_config(config_path)?;
        let preflight_paths = PreflightPaths::new(&BcArtifactPaths::new(&_config.output_dir, 0));

        fs::remove_file(result_path).ok();
        let child_run = run_probe_child_process(ProbeChildRunRequest {
            config_path,
            kind: request.kind,
            candidate_microbatch: request.candidate_microbatch,
            warmup_steps: request.warmup_steps,
            measure_steps: request.measure_steps,
            result_flag: "--probe-result-path",
            result_path,
            attempts: None,
            manifest_cache_path: &preflight_paths.manifest_cache_path,
            discovery_summary_path: &preflight_paths.discovery_summary_path,
            discovery_index_path: &preflight_paths.discovery_index_path,
        })?;
        let output = child_run.output;

        let result = if result_path.exists() {
            let result = read_probe_result(result_path)?;
            fs::remove_file(result_path).ok();
            result
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stdout = stdout.trim();
            let stderr = stderr.trim();
            build_probe_failure_result(
                request,
                stdout,
                stderr,
                output.status.code(),
                classify_probe_detail,
            )
        };
        finish_probe_spinner(
            &child_run.spinner,
            format_probe_spinner_finish_message(&result, child_run.elapsed_seconds),
        );
        Ok(result)
    }
}

pub fn execute_probe_request_batch(
    config_path: &Path,
    batch: ProbeBatchRequest,
    results_path: &Path,
    #[cfg_attr(
        test,
        allow(
            unused_variables,
            reason = "test path uses the in-process probe runner instead of stderr classification"
        )
    )]
    classify_probe_detail: impl Fn(&str) -> ProbeStatus,
) -> Result<Vec<ProbeResult>, String> {
    #[cfg(test)]
    {
        let _ = (config_path, batch, results_path, classify_probe_detail);
        Err("hydra-train-exec in-process probe batch execution is not available in crate-local tests".to_string())
    }

    #[cfg(not(test))]
    {
        let config = read_config(config_path)?;
        let preflight_paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
        let kind = batch.request.kind;
        let candidate_microbatch = batch.request.candidate_microbatch;

        fs::remove_file(results_path).ok();
        let child_run = run_probe_child_process(ProbeChildRunRequest {
            config_path,
            kind,
            candidate_microbatch,
            warmup_steps: batch.request.warmup_steps,
            measure_steps: batch.request.measure_steps,
            result_flag: "--probe-results-path",
            result_path: results_path,
            attempts: Some(batch.attempts),
            manifest_cache_path: &preflight_paths.manifest_cache_path,
            discovery_summary_path: &preflight_paths.discovery_summary_path,
            discovery_index_path: &preflight_paths.discovery_index_path,
        })?;
        let output = child_run.output;
        let results = recover_probe_batch_results(
            batch,
            results_path,
            output.status,
            &output.stdout,
            &output.stderr,
            classify_probe_detail,
        )?;
        let finish_result = results.last().cloned().unwrap_or_else(|| ProbeResult {
            kind,
            candidate_microbatch,
            status: ProbeStatus::BackendError,
            measured_samples_per_second: None,
            elapsed_seconds: None,
            detail: "probe batch completed without recorded results".to_string(),
        });
        finish_probe_spinner(
            &child_run.spinner,
            format_probe_spinner_finish_message(&finish_result, child_run.elapsed_seconds),
        );
        Ok(results)
    }
}

pub fn execute_probe_request_window(
    config_path: &Path,
    batches: Vec<ProbeBatchRequest>,
    results_paths: &[PathBuf],
    #[cfg_attr(
        test,
        allow(
            unused_variables,
            reason = "test path uses deterministic simulated child-window recovery"
        )
    )]
    classify_probe_detail: impl Fn(&str) -> ProbeStatus + Copy,
) -> Result<Vec<Vec<ProbeResult>>, String> {
    if batches.len() != results_paths.len() {
        return Err(format!(
            "probe window request/result path mismatch: requests={} paths={}",
            batches.len(),
            results_paths.len()
        ));
    }

    #[cfg(test)]
    {
        let _ = classify_probe_detail;
        let config = read_config(config_path)?;
        let unsupported_device = !matches!(config.device.as_str(), "cpu" | "cuda")
            && !config.device.starts_with("cuda:");
        let mut recovered = Vec::with_capacity(batches.len());
        for (batch, path) in batches.into_iter().zip(results_paths) {
            let candidate = batch.request.candidate_microbatch;
            let mut artifact = super::probe_transport::ProbeBatchArtifact::pending();
            if unsupported_device {
                let result = ProbeResult {
                    kind: batch.request.kind,
                    candidate_microbatch: candidate,
                    status: ProbeStatus::BackendError,
                    measured_samples_per_second: None,
                    elapsed_seconds: Some(0.01),
                    detail: format!("unsupported HYDRA_TRAIN_DEVICE={}", config.device),
                };
                artifact.push_result(result);
                write_probe_batch_artifact(path, &artifact)?;
                recovered.push(artifact.completed_results_before_failure().to_vec());
                break;
            }
            if candidate >= 512 {
                let result = ProbeResult {
                    kind: batch.request.kind,
                    candidate_microbatch: candidate,
                    status: ProbeStatus::Oom,
                    measured_samples_per_second: None,
                    elapsed_seconds: Some(0.01),
                    detail: "simulated OOM in reused child".to_string(),
                };
                artifact.push_result(result);
                write_probe_batch_artifact(path, &artifact)?;
                recovered.push(artifact.completed_results_before_failure().to_vec());
                break;
            }
            for _ in 0..batch.attempts.max(1) {
                artifact.push_result(ProbeResult {
                    kind: batch.request.kind,
                    candidate_microbatch: candidate,
                    status: ProbeStatus::Success,
                    measured_samples_per_second: Some(candidate as f64),
                    elapsed_seconds: Some(0.01),
                    detail: "simulated success in reused child".to_string(),
                });
            }
            artifact.mark_finished();
            write_probe_batch_artifact(path, &artifact)?;
            recovered.push(artifact.replay_ordered_results().cloned().collect());
        }
        Ok(recovered)
    }

    #[cfg(not(test))]
    {
        let config = read_config(config_path)?;
        let preflight_paths = PreflightPaths::new(&BcArtifactPaths::new(&config.output_dir, 0));
        let mut recovered = Vec::with_capacity(batches.len());
        for (batch, results_path) in batches.into_iter().zip(results_paths) {
            fs::remove_file(results_path).ok();
            let child_run = run_probe_child_process(ProbeChildRunRequest {
                config_path,
                kind: batch.request.kind,
                candidate_microbatch: batch.request.candidate_microbatch,
                warmup_steps: batch.request.warmup_steps,
                measure_steps: batch.request.measure_steps,
                result_flag: "--probe-results-path",
                result_path: results_path,
                attempts: Some(batch.attempts),
                manifest_cache_path: &preflight_paths.manifest_cache_path,
                discovery_summary_path: &preflight_paths.discovery_summary_path,
                discovery_index_path: &preflight_paths.discovery_index_path,
            })?;
            let output = child_run.output;
            let finish_kind = batch.request.kind;
            let finish_candidate = batch.request.candidate_microbatch;
            let candidate_results = recover_probe_batch_results(
                batch,
                results_path,
                output.status,
                &output.stdout,
                &output.stderr,
                classify_probe_detail,
            )?;
            let finish_result = candidate_results
                .last()
                .cloned()
                .unwrap_or_else(|| ProbeResult {
                    kind: finish_kind,
                    candidate_microbatch: finish_candidate,
                    status: ProbeStatus::BackendError,
                    measured_samples_per_second: None,
                    elapsed_seconds: None,
                    detail: "probe window completed without recorded results".to_string(),
                });
            finish_probe_spinner(
                &child_run.spinner,
                format_probe_spinner_finish_message(&finish_result, child_run.elapsed_seconds),
            );
            let reset_after_failure = candidate_results
                .last()
                .is_some_and(|result| result.status != ProbeStatus::Success);
            recovered.push(candidate_results);
            if reset_after_failure {
                break;
            }
        }
        Ok(recovered)
    }
}
