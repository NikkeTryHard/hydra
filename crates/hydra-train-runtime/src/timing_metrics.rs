//! Timing metric extraction for BC training profiling JSONL logs.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_BC_EPOCH, PROFILING_STAGE_BC_INTERVAL,
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED, PROFILING_STAGE_H2D_STREAM_SYNC,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_LOSS, PROFILING_STAGE_METRIC_READBACK, PROFILING_STAGE_OPTIMIZER_STEP,
    PROFILING_STAGE_PRODUCER_WAIT, PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION,
    ProfilingEnvelope,
};

/// Input filters for BC timing extraction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimingMetricsOptions {
    pub run_id: Option<String>,
    pub skip_initial_rows: usize,
    pub min_global_step: Option<usize>,
}

/// Full timing extractor report.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TimingMetricsReport {
    pub run_id: Option<String>,
    pub rows: Vec<TimingMetricsRow>,
    pub summary: TimingMetricsSummary,
}

/// One accepted profiled JSONL entry.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TimingMetricsRow {
    pub run_id: Option<String>,
    pub path: String,
    pub line_number: usize,
    pub scope: LogScope,
    pub root_stage: String,
    pub global_step: Option<usize>,
    pub epoch: Option<usize>,
    pub complete_for_gate: bool,
    pub missing_stages: Vec<String>,
    pub elapsed_seconds: f64,
    pub train_seconds: Option<f64>,
    pub producer_wait_seconds: Option<f64>,
    pub collation_seconds: Option<f64>,
    pub h2d_transfer_seconds: Option<f64>,
    pub h2d_pageable_to_pinned_seconds: Option<f64>,
    pub h2d_tensor_materialize_seconds: Option<f64>,
    pub h2d_stream_sync_seconds: Option<f64>,
    pub forward_seconds: Option<f64>,
    pub loss_seconds: Option<f64>,
    pub backward_seconds: Option<f64>,
    pub optimizer_step_seconds: Option<f64>,
    pub metric_readback_seconds: Option<f64>,
    pub validation_seconds: Option<f64>,
    pub checkpoint_seconds: Option<f64>,
    pub logging_seconds: Option<f64>,
    pub producer_wait_pct: Option<f64>,
    pub collation_pct: Option<f64>,
    pub h2d_transfer_pct: Option<f64>,
    pub input_starvation_pct: Option<f64>,
    pub compute_pct: Option<f64>,
    pub metric_readback_pct: Option<f64>,
    pub validation_pct: Option<f64>,
    pub checkpoint_pct: Option<f64>,
    pub logging_pct: Option<f64>,
    pub h2d_pageable_to_pinned_pct_of_h2d: Option<f64>,
    pub h2d_tensor_materialize_pct_of_h2d: Option<f64>,
    pub h2d_stream_sync_pct_of_h2d: Option<f64>,
    pub ring_occupancy_min: Option<usize>,
    pub ring_occupancy_avg: Option<f64>,
    pub window_steps: Option<usize>,
    pub window_samples: Option<usize>,
    pub steps_per_second: Option<f64>,
    pub samples_per_second: Option<f64>,
}

struct TimingMetricsLogFields<'a> {
    path: &'a str,
    line_number: usize,
    scope: LogScope,
    global_step: Option<usize>,
    epoch: Option<usize>,
    window_steps: Option<usize>,
    window_samples: Option<usize>,
    steps_per_second: Option<f64>,
    samples_per_second: Option<f64>,
}

/// Source scope for a parsed log row.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogScope {
    Step,
    Epoch,
}

/// Aggregate timing metrics over accepted rows.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct TimingMetricsSummary {
    pub row_count: usize,
    pub complete_row_count: usize,
    pub incomplete_row_count: usize,
    pub producer_wait_pct: MetricStats,
    pub collation_pct: MetricStats,
    pub h2d_transfer_pct: MetricStats,
    pub input_starvation_pct: MetricStats,
    pub compute_pct: MetricStats,
    pub metric_readback_pct: MetricStats,
    pub validation_pct: MetricStats,
    pub checkpoint_pct: MetricStats,
    pub logging_pct: MetricStats,
    pub steps_per_second: MetricStats,
    pub samples_per_second: MetricStats,
    pub interval_seconds: MetricStats,
    pub samples_per_second_cv: Option<f64>,
    pub producer_wait_pass: Option<bool>,
    pub collation_pass: Option<bool>,
    pub h2d_pass: Option<bool>,
    pub input_starvation_pass: Option<bool>,
    pub compute_share_pass: Option<bool>,
    pub ring_occupancy_available: bool,
}

/// Median/p95 for one percentage metric.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricStats {
    pub median: Option<f64>,
    pub p95: Option<f64>,
    pub count: usize,
    pub complete: bool,
}

/// Extracts timing metrics from step and training log files.
pub fn extract_timing_metrics_from_paths(
    step_logs: &[PathBuf],
    training_logs: &[PathBuf],
    options: &TimingMetricsOptions,
) -> Result<TimingMetricsReport, String> {
    let mut rows = Vec::new();
    for path in step_logs {
        rows.extend(read_log_path(path, LogScope::Step, options)?);
    }
    for path in training_logs {
        rows.extend(read_log_path(path, LogScope::Epoch, options)?);
    }
    finish_report(rows, options)
}

fn read_log_path(
    path: &Path,
    scope: LogScope,
    options: &TimingMetricsOptions,
) -> Result<Vec<TimingMetricsRow>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = BufReader::new(file);
    let path_display = path.display().to_string();
    extract_rows_from_reader(reader, &path_display, scope, options)
}

fn extract_rows_from_reader<R: BufRead>(
    reader: R,
    path: &str,
    scope: LogScope,
    options: &TimingMetricsOptions,
) -> Result<Vec<TimingMetricsRow>, String> {
    let mut rows = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|err| format!("failed to read {path}:{line_number}: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|err| format!("failed to parse {path}:{line_number}: {err}"))?;
        let Some(row) = row_from_value(value, path, line_number, scope, options)? else {
            continue;
        };
        rows.push(row);
    }
    Ok(rows)
}

fn finish_report(
    mut rows: Vec<TimingMetricsRow>,
    options: &TimingMetricsOptions,
) -> Result<TimingMetricsReport, String> {
    if let Some(min_global_step) = options.min_global_step {
        rows.retain(|row| row.global_step.is_some_and(|step| step >= min_global_step));
    }
    if options.skip_initial_rows > 0 {
        rows = rows.into_iter().skip(options.skip_initial_rows).collect();
    }
    if rows.is_empty() {
        return Err("no parseable profiled entries remain after filters".to_string());
    }
    let summary = summarize_rows(&rows);
    Ok(TimingMetricsReport {
        run_id: options.run_id.clone(),
        rows,
        summary,
    })
}

fn row_from_value(
    value: Value,
    path: &str,
    line_number: usize,
    scope: LogScope,
    options: &TimingMetricsOptions,
) -> Result<Option<TimingMetricsRow>, String> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    let global_step = object
        .get("global_step")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let epoch = object
        .get("epoch")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let window_steps = object
        .get("window_steps")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let window_samples = object
        .get("window_samples")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let steps_per_second = object.get("steps_per_second").and_then(Value::as_f64);
    let samples_per_second = object.get("samples_per_second").and_then(Value::as_f64);
    if matches!(scope, LogScope::Step) && (global_step.is_none() || epoch.is_none()) {
        return Ok(None);
    }
    let Some(profiling_value) = object.get("profiling") else {
        return Ok(None);
    };
    let profiling: ProfilingEnvelope = serde_json::from_value(profiling_value.clone())
        .map_err(|err| format!("failed to parse profiling at {path}:{line_number}: {err}"))?;
    if profiling.stage != PROFILING_STAGE_BC_INTERVAL && profiling.stage != PROFILING_STAGE_BC_EPOCH
    {
        return Ok(None);
    }
    let fields = TimingMetricsLogFields {
        path,
        line_number,
        scope,
        global_step,
        epoch,
        window_steps,
        window_samples,
        steps_per_second,
        samples_per_second,
    };
    Ok(Some(row_from_profiling(profiling, fields, options)))
}

fn row_from_profiling(
    profiling: ProfilingEnvelope,
    fields: TimingMetricsLogFields<'_>,
    options: &TimingMetricsOptions,
) -> TimingMetricsRow {
    let train = find_direct_child(&profiling, PROFILING_STAGE_TRAIN);
    let validation_seconds =
        find_direct_child(&profiling, PROFILING_STAGE_VALIDATION).map(stage_seconds);
    let checkpoint_seconds =
        find_direct_child(&profiling, PROFILING_STAGE_CHECKPOINT).map(stage_seconds);
    let logging_seconds = find_direct_child(&profiling, PROFILING_STAGE_LOGGING).map(stage_seconds);

    let mut missing_stages = Vec::new();
    let train_seconds = match train {
        Some(train) => Some(train.elapsed_seconds),
        None => {
            missing_stages.push(PROFILING_STAGE_TRAIN.to_string());
            None
        }
    };

    let producer_wait_seconds =
        required_child(train, PROFILING_STAGE_PRODUCER_WAIT, &mut missing_stages);
    let collation_seconds = required_child(train, PROFILING_STAGE_COLLATION, &mut missing_stages);
    let h2d = match train.and_then(|train| find_direct_child(train, PROFILING_STAGE_H2D_TRANSFER)) {
        Some(h2d) => Some(h2d),
        None => {
            missing_stages.push(PROFILING_STAGE_H2D_TRANSFER.to_string());
            None
        }
    };
    let h2d_transfer_seconds = h2d.map(stage_seconds);
    let forward_seconds = required_child(train, PROFILING_STAGE_FORWARD, &mut missing_stages);
    let loss_seconds = required_child(train, PROFILING_STAGE_LOSS, &mut missing_stages);
    let metric_readback_seconds =
        required_child(train, PROFILING_STAGE_METRIC_READBACK, &mut missing_stages);
    let backward_seconds = required_child(train, PROFILING_STAGE_BACKWARD, &mut missing_stages);
    let optimizer_step_seconds =
        required_child(train, PROFILING_STAGE_OPTIMIZER_STEP, &mut missing_stages);

    let h2d_pageable_to_pinned_seconds = required_child(
        h2d,
        PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED,
        &mut missing_stages,
    );
    let h2d_tensor_materialize_seconds = required_child(
        h2d,
        PROFILING_STAGE_H2D_TENSOR_MATERIALIZE,
        &mut missing_stages,
    );
    let h2d_stream_sync_seconds =
        required_child(h2d, PROFILING_STAGE_H2D_STREAM_SYNC, &mut missing_stages);

    missing_stages.dedup();

    let elapsed_seconds = profiling.elapsed_seconds;
    let producer_wait_pct = pct(producer_wait_seconds, Some(elapsed_seconds));
    let collation_pct = pct(collation_seconds, Some(elapsed_seconds));
    let h2d_transfer_pct = pct(h2d_transfer_seconds, Some(elapsed_seconds));
    let input_starvation_pct = sum_pct(
        &[
            producer_wait_seconds,
            collation_seconds,
            h2d_transfer_seconds,
        ],
        elapsed_seconds,
    );
    let compute_pct = sum_pct(
        &[
            forward_seconds,
            loss_seconds,
            backward_seconds,
            optimizer_step_seconds,
        ],
        elapsed_seconds,
    );
    let metric_readback_pct = pct(metric_readback_seconds, Some(elapsed_seconds));
    let validation_pct = pct(validation_seconds, Some(elapsed_seconds));
    let checkpoint_pct = pct(checkpoint_seconds, Some(elapsed_seconds));
    let logging_pct = pct(logging_seconds, Some(elapsed_seconds));
    let h2d_pageable_to_pinned_pct_of_h2d =
        pct(h2d_pageable_to_pinned_seconds, h2d_transfer_seconds);
    let h2d_tensor_materialize_pct_of_h2d =
        pct(h2d_tensor_materialize_seconds, h2d_transfer_seconds);
    let h2d_stream_sync_pct_of_h2d = pct(h2d_stream_sync_seconds, h2d_transfer_seconds);

    TimingMetricsRow {
        run_id: options.run_id.clone(),
        path: fields.path.to_string(),
        line_number: fields.line_number,
        scope: fields.scope,
        root_stage: profiling.stage,
        global_step: fields.global_step,
        epoch: fields.epoch,
        complete_for_gate: missing_stages.is_empty(),
        missing_stages,
        elapsed_seconds,
        train_seconds,
        producer_wait_seconds,
        collation_seconds,
        h2d_transfer_seconds,
        h2d_pageable_to_pinned_seconds,
        h2d_tensor_materialize_seconds,
        h2d_stream_sync_seconds,
        forward_seconds,
        loss_seconds,
        backward_seconds,
        optimizer_step_seconds,
        metric_readback_seconds,
        validation_seconds,
        checkpoint_seconds,
        logging_seconds,
        producer_wait_pct,
        collation_pct,
        h2d_transfer_pct,
        input_starvation_pct,
        compute_pct,
        metric_readback_pct,
        validation_pct,
        checkpoint_pct,
        logging_pct,
        h2d_pageable_to_pinned_pct_of_h2d,
        h2d_tensor_materialize_pct_of_h2d,
        h2d_stream_sync_pct_of_h2d,
        ring_occupancy_min: None,
        ring_occupancy_avg: None,
        window_steps: fields.window_steps,
        window_samples: fields.window_samples,
        steps_per_second: fields.steps_per_second,
        samples_per_second: fields.samples_per_second,
    }
}

fn required_child(
    parent: Option<&ProfilingEnvelope>,
    stage: &str,
    missing_stages: &mut Vec<String>,
) -> Option<f64> {
    let seconds = parent
        .and_then(|parent| find_direct_child(parent, stage))
        .map(stage_seconds);
    if seconds.is_none() {
        missing_stages.push(stage.to_string());
    }
    seconds
}

fn find_direct_child<'a>(
    root: &'a ProfilingEnvelope,
    stage: &str,
) -> Option<&'a ProfilingEnvelope> {
    root.children.iter().find(|child| child.stage == stage)
}

fn stage_seconds(stage: &ProfilingEnvelope) -> f64 {
    stage.elapsed_seconds
}

fn pct(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let numerator = numerator?;
    let denominator = denominator?;
    if denominator <= 0.0 || !denominator.is_finite() || !numerator.is_finite() {
        return None;
    }
    Some(100.0 * numerator / denominator)
}

fn sum_pct(numerators: &[Option<f64>], denominator: f64) -> Option<f64> {
    if denominator <= 0.0 || !denominator.is_finite() {
        return None;
    }
    let mut sum = 0.0;
    for numerator in numerators {
        let value = (*numerator)?;
        if !value.is_finite() {
            return None;
        }
        sum += value;
    }
    Some(100.0 * sum / denominator)
}

fn summarize_rows(rows: &[TimingMetricsRow]) -> TimingMetricsSummary {
    let producer_wait_pct = stats(rows.iter().filter_map(|row| row.producer_wait_pct));
    let collation_pct = stats(rows.iter().filter_map(|row| row.collation_pct));
    let h2d_transfer_pct = stats(rows.iter().filter_map(|row| row.h2d_transfer_pct));
    let input_starvation_pct = stats(rows.iter().filter_map(|row| row.input_starvation_pct));
    let compute_pct = stats(rows.iter().filter_map(|row| row.compute_pct));
    let metric_readback_pct = stats(rows.iter().filter_map(|row| row.metric_readback_pct));
    let validation_pct = stats(rows.iter().filter_map(|row| row.validation_pct));
    let checkpoint_pct = stats(rows.iter().filter_map(|row| row.checkpoint_pct));
    let logging_pct = stats(rows.iter().filter_map(|row| row.logging_pct));
    let steps_per_second = stats(rows.iter().filter_map(|row| row.steps_per_second));
    let samples_per_second = stats(rows.iter().filter_map(|row| row.samples_per_second));
    let interval_seconds = stats(rows.iter().map(|row| row.elapsed_seconds));
    let samples_per_second_cv =
        coefficient_of_variation(rows.iter().filter_map(|row| row.samples_per_second));

    TimingMetricsSummary {
        row_count: rows.len(),
        complete_row_count: rows.iter().filter(|row| row.complete_for_gate).count(),
        incomplete_row_count: rows.iter().filter(|row| !row.complete_for_gate).count(),
        producer_wait_pass: pass_le(&producer_wait_pct, 2.0, 5.0),
        collation_pass: pass_le(&collation_pct, 5.0, 10.0),
        h2d_pass: pass_le(&h2d_transfer_pct, 8.0, 15.0),
        input_starvation_pass: pass_le(&input_starvation_pct, 10.0, 20.0),
        compute_share_pass: pass_ge_median(&compute_pct, 70.0),
        producer_wait_pct,
        collation_pct,
        h2d_transfer_pct,
        input_starvation_pct,
        compute_pct,
        metric_readback_pct,
        validation_pct,
        checkpoint_pct,
        logging_pct,
        steps_per_second,
        samples_per_second,
        interval_seconds,
        samples_per_second_cv,
        ring_occupancy_available: false,
    }
}

fn stats(values: impl Iterator<Item = f64>) -> MetricStats {
    let mut values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    values.sort_by(f64::total_cmp);
    let count = values.len();
    if count == 0 {
        return MetricStats {
            median: None,
            p95: None,
            count,
            complete: false,
        };
    }
    let median = if count.is_multiple_of(2) {
        Some((values[count / 2 - 1] + values[count / 2]) * 0.5)
    } else {
        Some(values[count / 2])
    };
    let p95_index = ((0.95 * count as f64).ceil() as usize)
        .saturating_sub(1)
        .min(count - 1);
    MetricStats {
        median,
        p95: Some(values[p95_index]),
        count,
        complete: true,
    }
}

fn coefficient_of_variation(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean <= 0.0 || !mean.is_finite() {
        return None;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt() / mean)
}

fn pass_le(stats: &MetricStats, median_limit: f64, p95_limit: f64) -> Option<bool> {
    Some(stats.median? <= median_limit && stats.p95? <= p95_limit)
}

fn pass_ge_median(stats: &MetricStats, median_limit: f64) -> Option<bool> {
    Some(stats.median? >= median_limit)
}

#[cfg(test)]
#[path = "timing_metrics/tests.rs"]
mod tests;
