//! Env-gated LibTorch/Kineto profiler control for focused train-step traces.

use std::env;
use std::ffi::{CStr, CString, c_char, c_int};
use std::path::PathBuf;

const PROFILER_PATH_ENV: &str = "HYDRA_TORCH_PROFILER";
const PROFILER_START_STEP_ENV: &str = "HYDRA_TORCH_PROFILER_START_STEP";
const PROFILER_STOP_STEP_ENV: &str = "HYDRA_TORCH_PROFILER_STOP_STEP";
const PROFILER_RECORD_SHAPES_ENV: &str = "HYDRA_TORCH_PROFILER_RECORD_SHAPES";
const PROFILER_MODE_ENV: &str = "HYDRA_TORCH_PROFILER_MODE";

unsafe extern "C" {
    fn hydra_torch_profiler_start(record_shapes: c_int) -> c_int;
    fn hydra_torch_profiler_stop_and_save(path: *const c_char) -> c_int;
    fn hydra_torch_profiler_start_nvtx(record_shapes: c_int) -> c_int;
    fn hydra_torch_profiler_stop_nvtx() -> c_int;
    fn hydra_torch_profiler_last_exception_message() -> *const c_char;
}

/// Static profiler config parsed once from environment.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum TorchProfilerMode {
    KinetoTrace,
    Nvtx,
}

#[derive(Debug, Clone)]
struct TorchProfilerConfig {
    output_path: PathBuf,
    start_step: usize,
    stop_step: Option<usize>,
    record_shapes: bool,
    mode: TorchProfilerMode,
}

impl TorchProfilerConfig {
    fn from_env() -> Result<Option<Self>, String> {
        let Some(output_path) = env_path(PROFILER_PATH_ENV) else {
            return Ok(None);
        };
        let start_step = env_usize(PROFILER_START_STEP_ENV)?.unwrap_or(10).max(1);
        let stop_step = env_usize(PROFILER_STOP_STEP_ENV)?.unwrap_or(start_step + 5);
        if stop_step < start_step {
            return Err(format!(
                "{PROFILER_STOP_STEP_ENV}={stop_step} must be >= {PROFILER_START_STEP_ENV}={start_step}"
            ));
        }
        let record_shapes = env_bool(PROFILER_RECORD_SHAPES_ENV)?.unwrap_or(true);
        let mode = env_mode()?;
        Ok(Some(Self {
            output_path,
            start_step,
            stop_step: Some(stop_step),
            record_shapes,
            mode,
        }))
    }

    fn should_start_before(&self, session_step: usize) -> bool {
        session_step >= self.start_step && self.stop_step.is_none_or(|stop| session_step <= stop)
    }

    fn should_stop_after(&self, session_step: usize) -> bool {
        self.stop_step.is_some_and(|stop| session_step >= stop)
    }
}

/// Runtime profiler session. Disabled path is a pair of booleans and no FFI calls.
pub struct TorchProfilerSession {
    config: Option<TorchProfilerConfig>,
    active: bool,
    finished: bool,
}

impl TorchProfilerSession {
    /// Builds a profiler session from environment variables.
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            config: TorchProfilerConfig::from_env()?,
            active: false,
            finished: false,
        })
    }

    /// Starts profiling before `session_step` when the configured window opens.
    pub fn maybe_start_before_step(&mut self, session_step: usize) -> Result<(), String> {
        let Some(config) = self.config.as_ref() else {
            return Ok(());
        };
        if self.active || self.finished || !config.should_start_before(session_step) {
            return Ok(());
        }
        let status = match config.mode {
            TorchProfilerMode::KinetoTrace => unsafe {
                hydra_torch_profiler_start(i32::from(config.record_shapes))
            },
            TorchProfilerMode::Nvtx => unsafe {
                hydra_torch_profiler_start_nvtx(i32::from(config.record_shapes))
            },
        };
        if status != 0 {
            return Err(format!(
                "failed to start PyTorch profiler at session_step={session_step}: {}",
                last_error_message()
            ));
        }
        self.active = true;
        Ok(())
    }

    /// Stops profiling after `session_step` when the configured window closes.
    pub fn maybe_stop_after_step(&mut self, session_step: usize) -> Result<(), String> {
        let Some(config) = self.config.as_ref() else {
            return Ok(());
        };
        if self.active && config.should_stop_after(session_step) {
            self.stop_active()
        } else {
            Ok(())
        }
    }

    /// Stops an active profiler, used at loop exit or before returning a train error.
    pub fn stop_if_active(&mut self) -> Result<(), String> {
        if self.active {
            self.stop_active()
        } else {
            Ok(())
        }
    }

    fn stop_active(&mut self) -> Result<(), String> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| "PyTorch profiler active without config".to_string())?;
        let status = match config.mode {
            TorchProfilerMode::KinetoTrace => {
                let path = config.output_path.to_string_lossy();
                let path = CString::new(path.as_bytes()).map_err(|_| {
                    "PyTorch profiler output path contains interior NUL byte".to_string()
                })?;
                unsafe { hydra_torch_profiler_stop_and_save(path.as_ptr()) }
            }
            TorchProfilerMode::Nvtx => unsafe { hydra_torch_profiler_stop_nvtx() },
        };
        if status != 0 {
            return Err(format!(
                "failed to stop/save PyTorch profiler to {}: {}",
                config.output_path.display(),
                last_error_message()
            ));
        }
        self.active = false;
        self.finished = true;
        Ok(())
    }
}

impl Drop for TorchProfilerSession {
    fn drop(&mut self) {
        let _ = self.stop_if_active();
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

fn env_usize(name: &str) -> Result<Option<usize>, String> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };
    value
        .to_string_lossy()
        .parse::<usize>()
        .map(Some)
        .map_err(|err| format!("{name} must be a positive integer: {err}"))
}

fn env_bool(name: &str) -> Result<Option<bool>, String> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };
    let value = value.to_string_lossy();
    match value.as_ref() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(Some(true)),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(Some(false)),
        _ => Err(format!("{name} must be boolean, got {value}")),
    }
}

fn env_mode() -> Result<TorchProfilerMode, String> {
    let Some(value) = env::var_os(PROFILER_MODE_ENV) else {
        return Ok(TorchProfilerMode::KinetoTrace);
    };
    let value = value.to_string_lossy();
    match value.as_ref() {
        "kineto" | "trace" | "chrome" => Ok(TorchProfilerMode::KinetoTrace),
        "nvtx" => Ok(TorchProfilerMode::Nvtx),
        _ => Err(format!(
            "{PROFILER_MODE_ENV} must be kineto or nvtx, got {value}"
        )),
    }
}

fn last_error_message() -> String {
    let ptr = unsafe { hydra_torch_profiler_last_exception_message() };
    if ptr.is_null() {
        return "no C++ error message".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
