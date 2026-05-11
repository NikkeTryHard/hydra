use super::*;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

#[test]
fn report_serializes_probe_event() {
    let report = CudaGraphProbeReport::failure("stage", "message", 0.25);
    let json = serde_json::to_value(&report).expect("report should serialize");
    assert_eq!(json["event"], "cuda_graph_probe");
    assert_eq!(json["status"], "failure");
    assert_eq!(json["stage"], "stage");
}

#[cfg(unix)]
#[test]
fn summarize_child_failure_keeps_useful_lines() {
    let output = std::process::Output {
        status: std::process::ExitStatus::from_raw(1 << 8),
        stdout: b"warning: noisy\nuseful stdout\n".to_vec(),
        stderr: b"thread panicked\nFinished `release`\nsecond\n".to_vec(),
    };
    let summary = summarize_child_failure(&output);
    assert!(summary.contains("thread panicked"));
    assert!(summary.contains("second"));
    assert!(summary.contains("useful stdout"));
    assert!(!summary.contains("warning: noisy"));
}
