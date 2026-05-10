#![allow(
    missing_docs,
    reason = "migrated train binary helpers preserve existing internal surface"
)]

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use hydra_train_runtime::preflight::{ProbeKind, ProbeResult, ProbeStatus};

use super::artifacts::{BcArtifactPaths, RlArtifactPaths, atomic_write_text};
use super::probe_summary::probe_kind_name;
use hydra_train_runtime::probe_request::{ProbeBatchRequest, ProbeRequest};

pub fn should_suppress_probe_output_line(line: &str) -> bool {
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

pub fn summarize_probe_failure_output(output: &str) -> String {
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

pub fn probe_failure_detail(
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

pub fn build_probe_failure_result(
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeBatchArtifact {
    pub finished: bool,
    pub results: Vec<ProbeResult>,
}

impl ProbeBatchArtifact {
    pub fn pending() -> Self {
        Self {
            finished: false,
            results: Vec::new(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn push_result(&mut self, result: ProbeResult) {
        if !self.finished {
            self.results.push(result);
        }
    }

    pub fn mark_finished(&mut self) {
        self.finished = true;
    }

    #[cfg(test)]
    pub fn from_results(results: Vec<ProbeResult>, finished: bool) -> Self {
        Self { finished, results }
    }

    pub fn replay_ordered_results(&self) -> impl Iterator<Item = &ProbeResult> {
        self.results.iter()
    }

    pub fn completed_results_before_failure(&self) -> &[ProbeResult] {
        let len = self
            .results
            .iter()
            .position(|result| result.status != ProbeStatus::Success)
            .map(|index| index + 1)
            .unwrap_or(self.results.len());
        &self.results[..len]
    }
}

pub fn write_probe_result(path: &Path, result: &ProbeResult) -> Result<(), String> {
    let json = serde_json::to_string(result)
        .map_err(|err| format!("failed to serialize probe result {}: {err}", path.display()))?;
    atomic_write_text(path, &json, "probe result")
}

pub fn read_probe_result(path: &Path) -> Result<ProbeResult, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read probe result {}: {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse probe result {}: {err}", path.display()))
}

pub fn write_probe_batch_artifact(
    path: &Path,
    artifact: &ProbeBatchArtifact,
) -> Result<(), String> {
    let json = serde_json::to_string(artifact).map_err(|err| {
        format!(
            "failed to serialize probe batch artifact {}: {err}",
            path.display()
        )
    })?;
    atomic_write_text(path, &json, "probe batch artifact")
}

pub fn read_probe_batch_artifact(path: &Path) -> Result<ProbeBatchArtifact, String> {
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

pub fn probe_result_path(
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

pub fn probe_batch_results_path(result_path: &Path) -> PathBuf {
    match result_path.extension() {
        Some(extension) => {
            result_path.with_extension(format!("batch.{}", extension.to_string_lossy()))
        }
        None => result_path.with_extension("batch"),
    }
}

pub fn rl_probe_result_path(
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

pub fn recover_probe_batch_results(
    batch: ProbeBatchRequest,
    results_path: &Path,
    status: std::process::ExitStatus,
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
