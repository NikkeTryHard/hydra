use std::io::Cursor;

use super::{LogScope, TimingMetricsOptions, extract_rows_from_reader, finish_report};

fn complete_line(global_step: usize, producer_wait: f64) -> String {
    format!(
        r#"{{"global_step":{global_step},"epoch":2,"window_steps":4,"window_samples":512,"steps_per_second":8.0,"samples_per_second":1024.0,"profiling":{{"stage":"bc_interval","elapsed_seconds":10.0,"children":[{{"stage":"train","elapsed_seconds":8.0,"children":[{{"stage":"producer_wait","elapsed_seconds":{producer_wait}}},{{"stage":"collation","elapsed_seconds":1.0}},{{"stage":"h2d_transfer","elapsed_seconds":2.0,"children":[{{"stage":"h2d_pageable_to_pinned","elapsed_seconds":0.5}},{{"stage":"h2d_tensor_materialize","elapsed_seconds":0.75}},{{"stage":"h2d_stream_sync","elapsed_seconds":0.25}}]}},{{"stage":"forward","elapsed_seconds":1.0}},{{"stage":"loss","elapsed_seconds":0.5}},{{"stage":"metric_readback","elapsed_seconds":0.2}},{{"stage":"backward","elapsed_seconds":1.5}},{{"stage":"optimizer_step","elapsed_seconds":1.0}}]}},{{"stage":"validation","elapsed_seconds":0.6}},{{"stage":"checkpoint","elapsed_seconds":0.3}},{{"stage":"logging","elapsed_seconds":0.1}}]}}}}"#
    )
}

#[test]
fn extracts_complete_bc_interval_metrics() {
    let input = complete_line(10, 0.5);
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "fixture.jsonl",
        LogScope::Step,
        &TimingMetricsOptions {
            run_id: Some("run-a".to_string()),
            ..TimingMetricsOptions::default()
        },
    )
    .expect("fixture should parse");

    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.run_id.as_deref(), Some("run-a"));
    assert_eq!(row.path, "fixture.jsonl");
    assert_eq!(row.line_number, 1);
    assert_eq!(row.global_step, Some(10));
    assert_eq!(row.epoch, Some(2));
    assert!(row.complete_for_gate);
    assert!(row.missing_stages.is_empty());
    assert_eq!(row.elapsed_seconds, 10.0);
    assert_eq!(row.train_seconds, Some(8.0));
    assert_eq!(row.producer_wait_seconds, Some(0.5));
    assert_eq!(row.collation_seconds, Some(1.0));
    assert_eq!(row.h2d_transfer_seconds, Some(2.0));
    assert_eq!(row.h2d_pageable_to_pinned_seconds, Some(0.5));
    assert_eq!(row.h2d_tensor_materialize_seconds, Some(0.75));
    assert_eq!(row.h2d_stream_sync_seconds, Some(0.25));
    assert_eq!(row.forward_seconds, Some(1.0));
    assert_eq!(row.loss_seconds, Some(0.5));
    assert_eq!(row.backward_seconds, Some(1.5));
    assert_eq!(row.optimizer_step_seconds, Some(1.0));
    assert_eq!(row.metric_readback_seconds, Some(0.2));
    assert_eq!(row.validation_seconds, Some(0.6));
    assert_eq!(row.checkpoint_seconds, Some(0.3));
    assert_eq!(row.logging_seconds, Some(0.1));
    assert_eq!(row.producer_wait_pct, Some(5.0));
    assert_eq!(row.collation_pct, Some(10.0));
    assert_eq!(row.h2d_transfer_pct, Some(20.0));
    assert_eq!(row.input_starvation_pct, Some(35.0));
    assert_eq!(row.compute_pct, Some(40.0));
    assert_eq!(row.metric_readback_pct, Some(2.0));
    assert_eq!(row.validation_pct, Some(6.0));
    assert_eq!(row.checkpoint_pct, Some(3.0));
    assert_eq!(row.logging_pct, Some(1.0));
    assert_eq!(row.h2d_pageable_to_pinned_pct_of_h2d, Some(25.0));
    assert_eq!(row.h2d_tensor_materialize_pct_of_h2d, Some(37.5));
    assert_eq!(row.h2d_stream_sync_pct_of_h2d, Some(12.5));
    assert_eq!(row.ring_occupancy_min, None);
    assert_eq!(row.ring_occupancy_avg, None);
    assert_eq!(row.window_steps, Some(4));
    assert_eq!(row.window_samples, Some(512));
    assert_eq!(row.steps_per_second, Some(8.0));
    assert_eq!(row.samples_per_second, Some(1024.0));
}

#[test]
fn skips_advisory_only_lines() {
    let input = format!(
        "{}\n{}",
        r#"{"event":"runtime_advisories","scope":"interval","advisories":["x"]}"#,
        complete_line(11, 0.1)
    );
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "step_log.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].global_step, Some(11));
    assert_eq!(rows[0].line_number, 2);
}

#[test]
fn missing_gate_stage_marks_row_incomplete() {
    let input = serde_json::json!({
        "global_step": 1,
        "epoch": 0,
        "profiling": {
            "stage": "bc_interval",
            "elapsed_seconds": 10.0,
            "children": [{
                "stage": "train",
                "elapsed_seconds": 8.0,
                "children": [
                    {"stage":"producer_wait","elapsed_seconds":0.5},
                    {"stage":"collation","elapsed_seconds":1.0},
                    {"stage":"forward","elapsed_seconds":1.0},
                    {"stage":"loss","elapsed_seconds":0.5},
                    {"stage":"metric_readback","elapsed_seconds":0.2},
                    {"stage":"backward","elapsed_seconds":1.5},
                    {"stage":"optimizer_step","elapsed_seconds":1.0}
                ]
            }]
        }
    })
    .to_string();
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "missing.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");

    let row = &rows[0];
    assert!(!row.complete_for_gate);
    assert!(
        row.missing_stages
            .iter()
            .any(|stage| stage == "h2d_transfer")
    );
    assert_eq!(row.h2d_transfer_seconds, None);
    assert_eq!(row.h2d_transfer_pct, None);
    assert_eq!(row.input_starvation_pct, None);
}

#[test]
fn nested_stage_does_not_satisfy_direct_child_requirement() {
    let input = serde_json::json!({
        "global_step": 1,
        "epoch": 0,
        "profiling": {
            "stage": "bc_interval",
            "elapsed_seconds": 10.0,
            "children": [{
                "stage": "validation",
                "elapsed_seconds": 8.0,
                "children": [{
                    "stage": "train",
                    "elapsed_seconds": 8.0,
                    "children": [
                        {"stage":"producer_wait","elapsed_seconds":0.5},
                        {"stage":"collation","elapsed_seconds":1.0},
                        {"stage":"h2d_transfer","elapsed_seconds":2.0,"children":[
                            {"stage":"h2d_pageable_to_pinned","elapsed_seconds":0.5},
                            {"stage":"h2d_tensor_materialize","elapsed_seconds":0.75},
                            {"stage":"h2d_stream_sync","elapsed_seconds":0.25}
                        ]},
                        {"stage":"forward","elapsed_seconds":1.0},
                        {"stage":"loss","elapsed_seconds":0.5},
                        {"stage":"metric_readback","elapsed_seconds":0.2},
                        {"stage":"backward","elapsed_seconds":1.5},
                        {"stage":"optimizer_step","elapsed_seconds":1.0}
                    ]
                }]
            }]
        }
    })
    .to_string();
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "nested.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");

    let row = &rows[0];
    assert!(!row.complete_for_gate);
    assert!(row.missing_stages.iter().any(|stage| stage == "train"));
    assert_eq!(row.train_seconds, None);
    assert_eq!(row.producer_wait_seconds, None);
    assert_eq!(row.input_starvation_pct, None);
}

#[test]
fn zero_denominator_emits_null_percentages() {
    let input = serde_json::json!({
        "global_step": 1,
        "epoch": 0,
        "profiling": {
            "stage": "bc_interval",
            "elapsed_seconds": 0.0,
            "children": [{
                "stage": "train",
                "elapsed_seconds": 8.0,
                "children": [
                    {"stage":"producer_wait","elapsed_seconds":0.5},
                    {"stage":"collation","elapsed_seconds":1.0},
                    {"stage":"h2d_transfer","elapsed_seconds":0.0,"children":[
                        {"stage":"h2d_pageable_to_pinned","elapsed_seconds":0.5},
                        {"stage":"h2d_tensor_materialize","elapsed_seconds":0.75},
                        {"stage":"h2d_stream_sync","elapsed_seconds":0.25}
                    ]},
                    {"stage":"forward","elapsed_seconds":1.0},
                    {"stage":"loss","elapsed_seconds":0.5},
                    {"stage":"metric_readback","elapsed_seconds":0.2},
                    {"stage":"backward","elapsed_seconds":1.5},
                    {"stage":"optimizer_step","elapsed_seconds":1.0}
                ]
            }]
        }
    })
    .to_string();
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "zero.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");

    let row = &rows[0];
    assert!(row.complete_for_gate);
    assert_eq!(row.producer_wait_pct, None);
    assert_eq!(row.collation_pct, None);
    assert_eq!(row.h2d_transfer_pct, None);
    assert_eq!(row.input_starvation_pct, None);
    assert_eq!(row.compute_pct, None);
    assert_eq!(row.h2d_pageable_to_pinned_pct_of_h2d, None);
    assert_eq!(row.h2d_tensor_materialize_pct_of_h2d, None);
    assert_eq!(row.h2d_stream_sync_pct_of_h2d, None);
}

#[test]
fn summary_uses_post_warmup_rows() {
    let input = [
        complete_line(1, 9.0),
        complete_line(2, 0.1),
        complete_line(3, 0.2),
        complete_line(4, 0.3),
    ]
    .join("\n");
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "warmup.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");
    let report = finish_report(
        rows,
        &TimingMetricsOptions {
            skip_initial_rows: 0,
            min_global_step: Some(2),
            ..TimingMetricsOptions::default()
        },
    )
    .expect("report should summarize");

    assert_eq!(report.rows.len(), 3);
    assert_eq!(report.summary.producer_wait_pct.count, 3);
    assert_eq!(report.summary.producer_wait_pct.median, Some(2.0));
    assert_eq!(report.summary.producer_wait_pct.p95, Some(3.0));
}

#[test]
fn summary_reports_interval_throughput_and_cv() {
    let input = [
        complete_line(1, 0.1),
        complete_line(2, 0.2).replace("1024.0", "2048.0"),
        complete_line(3, 0.3),
    ]
    .join("\n");
    let rows = extract_rows_from_reader(
        Cursor::new(input),
        "throughput.jsonl",
        LogScope::Step,
        &TimingMetricsOptions::default(),
    )
    .expect("fixture should parse");
    let report =
        finish_report(rows, &TimingMetricsOptions::default()).expect("report should summarize");

    assert_eq!(report.summary.steps_per_second.median, Some(8.0));
    assert_eq!(report.summary.samples_per_second.median, Some(1024.0));
    assert_eq!(report.summary.samples_per_second.p95, Some(2048.0));
    assert_eq!(report.summary.interval_seconds.median, Some(10.0));
    let cv = report
        .summary
        .samples_per_second_cv
        .expect("cv should be present");
    assert!(cv > 0.35 && cv < 0.36, "unexpected cv {cv}");
}
