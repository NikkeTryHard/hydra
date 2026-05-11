use std::path::PathBuf;

use crate::preflight::ProbeKind;

use super::*;
use crate::test_support::dummy_train_config;

fn dummy_config() -> TrainConfig {
    dummy_train_config()
}

#[test]
fn probe_request_from_cli_uses_probe_overrides() {
    let config = dummy_config();

    let request = probe_request_from_cli(
        &config,
        Some(ProbeCliRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 192,
            warmup_steps: Some(7),
            measure_steps: Some(9),
        }),
    )
    .expect("probe request should parse")
    .expect("probe request should be present");
    assert_eq!(request.kind, ProbeKind::Validation);
    assert_eq!(request.candidate_microbatch, 192);
    assert_eq!(request.warmup_steps, 7);
    assert_eq!(request.measure_steps, 9);
}

#[test]
fn probe_request_from_cli_falls_back_to_preflight_defaults() {
    let mut config = dummy_config();
    config.preflight.warmup_steps = 11;
    config.preflight.measure_steps = 13;
    let request = probe_request_from_cli(
        &config,
        Some(ProbeCliRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 256,
            warmup_steps: None,
            measure_steps: None,
        }),
    )
    .expect("probe request should parse")
    .expect("probe request should be present");
    assert_eq!(request.warmup_steps, 11);
    assert_eq!(request.measure_steps, 13);
}

#[test]
fn probe_request_from_cli_rejects_zero_values() {
    let config = dummy_config();

    let err = probe_request_from_cli(
        &config,
        Some(ProbeCliRequest {
            kind: ProbeKind::Train,
            candidate_microbatch: 0,
            warmup_steps: Some(0),
            measure_steps: Some(0),
        }),
    )
    .expect_err("zero candidate should fail");
    assert!(err.contains("--probe-candidate-microbatch"));
}

#[test]
fn probe_child_request_from_cli_parses_child_probe_inputs() {
    let (request, path, manifest_cache_path) =
        probe_child_request_from_cli(Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 192,
                warmup_steps: Some(4),
                measure_steps: Some(12),
            },
            result_path: PathBuf::from("/tmp/probe.json"),
            manifest_cache_path: Some(PathBuf::from("/tmp/manifest.json")),
        })))
        .expect("child request should parse")
        .expect("child request should be present");
    assert_eq!(request.kind, ProbeKind::Train);
    assert_eq!(request.candidate_microbatch, 192);
    assert_eq!(request.warmup_steps, 4);
    assert_eq!(request.measure_steps, 12);
    assert_eq!(path, PathBuf::from("/tmp/probe.json"));
    assert_eq!(
        manifest_cache_path,
        Some(PathBuf::from("/tmp/manifest.json"))
    );
}

#[test]
fn probe_child_request_from_cli_ignores_batch_child_inputs() {
    let child = ProbeChildRequest::Batch(ProbeBatchChildRequest {
        request: ProbeCliRequest {
            kind: ProbeKind::Validation,
            candidate_microbatch: 64,
            warmup_steps: Some(2),
            measure_steps: Some(3),
        },
        attempts: 2,
        results_path: PathBuf::from("/tmp/probe-results.json"),
        manifest_cache_path: None,
    });

    let parsed = probe_child_request_from_cli(Some(child)).expect("batch child should parse");
    assert!(parsed.is_none());
}

#[test]
fn probe_batch_child_request_from_cli_parses_batch_probe_inputs() {
    let (batch, path, manifest_cache_path) = probe_batch_child_request_from_cli(Some(
        ProbeChildRequest::Batch(ProbeBatchChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Validation,
                candidate_microbatch: 256,
                warmup_steps: Some(5),
                measure_steps: Some(9),
            },
            attempts: 3,
            results_path: PathBuf::from("/tmp/probe-results.json"),
            manifest_cache_path: Some(PathBuf::from("/tmp/manifest.json")),
        }),
    ))
    .expect("batch child request should parse")
    .expect("batch child request should be present");

    assert_eq!(batch.request.kind, ProbeKind::Validation);
    assert_eq!(batch.request.candidate_microbatch, 256);
    assert_eq!(batch.request.warmup_steps, 5);
    assert_eq!(batch.request.measure_steps, 9);
    assert_eq!(batch.attempts, 3);
    assert_eq!(path, PathBuf::from("/tmp/probe-results.json"));
    assert_eq!(
        manifest_cache_path,
        Some(PathBuf::from("/tmp/manifest.json"))
    );
}

#[test]
fn probe_batch_child_request_from_cli_rejects_zero_attempts() {
    let err = probe_batch_child_request_from_cli(Some(ProbeChildRequest::Batch(
        ProbeBatchChildRequest {
            request: ProbeCliRequest {
                kind: ProbeKind::Train,
                candidate_microbatch: 128,
                warmup_steps: Some(2),
                measure_steps: Some(4),
            },
            attempts: 0,
            results_path: PathBuf::from("/tmp/probe-results.json"),
            manifest_cache_path: None,
        },
    )))
    .expect_err("zero attempts should fail");

    assert_eq!(err, "internal probe batch child missing positive attempts");
}
