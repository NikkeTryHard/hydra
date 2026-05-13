#![allow(
    missing_docs,
    reason = "lightweight internal telemetry seam shared by preflight parent and child"
)]

use std::fs;
use std::process::Command;
use std::time::Instant;

use hydra_train_runtime::preflight::{ProbeKind, SystemMetricEventKind, SystemMetricsEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSample {
    pub process_ticks: u64,
    pub total_ticks: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskSample {
    pub read_bytes: u64,
    pub write_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuTelemetry {
    pub util_percent: Option<f64>,
    pub mem_used_mb: Option<u64>,
    pub mem_free_mb: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ResourceSampler {
    started_at: Instant,
    last_at: Instant,
    last_cpu: Option<CpuSample>,
    last_disk: Option<DiskSample>,
    gpu_oom_count: u64,
}

impl Default for ResourceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSampler {
    #[must_use]
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            started_at: now,
            last_at: now,
            last_cpu: read_cpu_sample(),
            last_disk: read_disk_sample(),
            gpu_oom_count: 0,
        }
    }

    #[must_use]
    pub fn snapshot(&mut self, phase: impl Into<String>) -> SystemMetricsEvent {
        let now = Instant::now();
        let elapsed_since_last = now.saturating_duration_since(self.last_at).as_secs_f64();
        let cpu = read_cpu_sample();
        let disk = read_disk_sample();
        let gpu = read_gpu_telemetry();
        let event = resource_snapshot_event(
            phase,
            now.saturating_duration_since(self.started_at).as_secs_f64(),
            cpu_percent_between(self.last_cpu, cpu),
            read_process_rss_bytes(),
            disk_rates_between(self.last_disk, disk, elapsed_since_last),
            gpu,
            self.gpu_oom_count,
        );
        self.last_at = now;
        self.last_cpu = cpu;
        self.last_disk = disk;
        event
    }

    pub fn record_gpu_oom(&mut self) {
        self.gpu_oom_count = self.gpu_oom_count.saturating_add(1);
    }
}

pub fn probe_host_memory_event(
    kind: ProbeKind,
    candidate_microbatch: usize,
    mem_available_bytes: Option<u64>,
    mem_total_bytes: Option<u64>,
) -> SystemMetricsEvent {
    SystemMetricsEvent {
        kind: SystemMetricEventKind::ProbeHostMemory,
        probe_kind: Some(kind),
        candidate_microbatch: Some(candidate_microbatch),
        mem_available_bytes,
        mem_total_bytes,
        ..SystemMetricsEvent::default()
    }
}

pub fn probe_child_init_event(
    kind: ProbeKind,
    candidate_microbatch: usize,
    model_init_ms: u128,
    optimizer_init_ms: u128,
    loss_init_ms: u128,
) -> SystemMetricsEvent {
    SystemMetricsEvent {
        kind: SystemMetricEventKind::ProbeChildInit,
        probe_kind: Some(kind),
        candidate_microbatch: Some(candidate_microbatch),
        model_init_ms: Some(model_init_ms),
        optimizer_init_ms: Some(optimizer_init_ms),
        loss_init_ms: Some(loss_init_ms),
        ..SystemMetricsEvent::default()
    }
}

#[must_use]
pub fn progress_event(
    phase: impl Into<String>,
    completed: u64,
    planned: u64,
    elapsed_seconds: f64,
    files_per_second: Option<f64>,
    samples_per_second: Option<f64>,
) -> SystemMetricsEvent {
    SystemMetricsEvent {
        kind: SystemMetricEventKind::Progress,
        phase: Some(phase.into()),
        completed: Some(completed),
        planned: Some(planned),
        elapsed_seconds: Some(elapsed_seconds),
        files_per_second,
        samples_per_second,
        ..SystemMetricsEvent::default()
    }
}

#[must_use]
pub fn pipeline_stage_event(
    phase: impl Into<String>,
    stage: impl Into<String>,
    elapsed_seconds: f64,
    samples: Option<u64>,
) -> SystemMetricsEvent {
    let samples_per_second = samples.and_then(|count| rate_per_second(count, elapsed_seconds));
    SystemMetricsEvent {
        kind: SystemMetricEventKind::PipelineStage,
        phase: Some(phase.into()),
        stage: Some(stage.into()),
        completed: samples,
        elapsed_seconds: Some(elapsed_seconds),
        samples_per_second,
        ..SystemMetricsEvent::default()
    }
}

#[must_use]
pub fn resource_snapshot_event(
    phase: impl Into<String>,
    elapsed_seconds: f64,
    cpu_percent: Option<f64>,
    process_rss_bytes: Option<u64>,
    disk_rates: Option<(f64, f64)>,
    gpu: Option<GpuTelemetry>,
    gpu_oom_count: u64,
) -> SystemMetricsEvent {
    SystemMetricsEvent {
        kind: SystemMetricEventKind::ResourceSnapshot,
        phase: Some(phase.into()),
        elapsed_seconds: Some(elapsed_seconds),
        cpu_percent,
        process_rss_bytes,
        disk_read_mb_per_sec: disk_rates.map(|rates| rates.0),
        disk_write_mb_per_sec: disk_rates.map(|rates| rates.1),
        gpu_util_percent: gpu.and_then(|value| value.util_percent),
        gpu_mem_used_mb: gpu.and_then(|value| value.mem_used_mb),
        gpu_mem_free_mb: gpu.and_then(|value| value.mem_free_mb),
        gpu_oom_count: Some(gpu_oom_count),
        ..SystemMetricsEvent::default()
    }
}

#[must_use]
pub fn rate_per_second(count: u64, elapsed_seconds: f64) -> Option<f64> {
    if count == 0 || elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        None
    } else {
        Some(count as f64 / elapsed_seconds)
    }
}

#[must_use]
pub fn cpu_percent_between(previous: Option<CpuSample>, current: Option<CpuSample>) -> Option<f64> {
    let previous = previous?;
    let current = current?;
    let process_delta = current.process_ticks.checked_sub(previous.process_ticks)?;
    let total_delta = current.total_ticks.checked_sub(previous.total_ticks)?;
    if total_delta == 0 {
        None
    } else {
        Some((process_delta as f64 / total_delta as f64) * 100.0)
    }
}

#[must_use]
pub fn disk_rates_between(
    previous: Option<DiskSample>,
    current: Option<DiskSample>,
    elapsed_seconds: f64,
) -> Option<(f64, f64)> {
    if elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return None;
    }
    let previous = previous?;
    let current = current?;
    let read_delta = current.read_bytes.checked_sub(previous.read_bytes)?;
    let write_delta = current.write_bytes.checked_sub(previous.write_bytes)?;
    Some((
        read_delta as f64 / 1_048_576.0 / elapsed_seconds,
        write_delta as f64 / 1_048_576.0 / elapsed_seconds,
    ))
}

#[must_use]
pub fn parse_proc_stat_total_ticks(contents: &str) -> Option<u64> {
    let first = contents.lines().next()?;
    let rest = first.strip_prefix("cpu ")?;
    rest.split_whitespace()
        .map(str::parse::<u64>)
        .try_fold(0u64, |acc, value| {
            value.ok().and_then(|v| acc.checked_add(v))
        })
}

#[must_use]
pub fn parse_proc_self_stat_process_ticks(contents: &str) -> Option<u64> {
    let end_comm = contents.rfind(") ")?;
    let after = contents.get(end_comm + 2..)?;
    let fields: Vec<&str> = after.split_whitespace().collect();
    let utime = fields.get(11)?.parse::<u64>().ok()?;
    let stime = fields.get(12)?.parse::<u64>().ok()?;
    utime.checked_add(stime)
}

#[must_use]
pub fn parse_proc_self_status_rss_bytes(contents: &str) -> Option<u64> {
    for line in contents.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let kb = fields.next()?.parse::<u64>().ok()?;
        return kb.checked_mul(1024);
    }
    None
}

#[must_use]
pub fn parse_proc_self_io(contents: &str) -> Option<DiskSample> {
    let mut read_bytes = None;
    let mut write_bytes = None;
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("read_bytes:") {
            read_bytes = value.trim().parse::<u64>().ok();
        } else if let Some(value) = line.strip_prefix("write_bytes:") {
            write_bytes = value.trim().parse::<u64>().ok();
        }
    }
    Some(DiskSample {
        read_bytes: read_bytes?,
        write_bytes: write_bytes?,
    })
}

#[must_use]
pub fn parse_nvidia_smi_csv(contents: &str) -> Option<GpuTelemetry> {
    let mut fields = contents.trim().split(',').map(str::trim);
    Some(GpuTelemetry {
        util_percent: fields.next()?.parse::<f64>().ok(),
        mem_used_mb: fields.next()?.parse::<u64>().ok(),
        mem_free_mb: fields.next()?.parse::<u64>().ok(),
    })
}

fn read_cpu_sample() -> Option<CpuSample> {
    let process_ticks =
        parse_proc_self_stat_process_ticks(&fs::read_to_string("/proc/self/stat").ok()?)?;
    let total_ticks = parse_proc_stat_total_ticks(&fs::read_to_string("/proc/stat").ok()?)?;
    Some(CpuSample {
        process_ticks,
        total_ticks,
    })
}

fn read_process_rss_bytes() -> Option<u64> {
    parse_proc_self_status_rss_bytes(&fs::read_to_string("/proc/self/status").ok()?)
}

fn read_disk_sample() -> Option<DiskSample> {
    parse_proc_self_io(&fs::read_to_string("/proc/self/io").ok()?)
}

fn read_gpu_telemetry() -> Option<GpuTelemetry> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_nvidia_smi_csv(std::str::from_utf8(&output.stdout).ok()?)
}

pub fn format_system_metrics_event(event: &SystemMetricsEvent) -> String {
    serde_json::to_string(event).unwrap_or_else(|err| {
        format!(
            "{{\"kind\":\"system_metrics_serialize_error\",\"detail\":\"{}\"}}",
            err
        )
    })
}

pub fn emit_system_metrics_event(event: &SystemMetricsEvent) {
    println!("system_metrics {}", format_system_metrics_event(event));
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_runtime::preflight::SystemMetricEventKind;

    #[test]
    fn probe_child_init_event_serializes_sparse_fields() {
        let event = probe_child_init_event(ProbeKind::Train, 64, 10, 2, 1);
        let encoded = format_system_metrics_event(&event);

        assert!(encoded.contains("\"kind\":\"probe_child_init\""));
        assert!(encoded.contains("\"probe_kind\":\"train\""));
        assert!(encoded.contains("\"candidate_microbatch\":64"));
        assert!(encoded.contains("\"model_init_ms\":10"));
        assert!(!encoded.contains("mem_available_bytes"));
    }

    #[test]
    fn host_memory_event_keeps_optional_memory_values() {
        let event = probe_host_memory_event(ProbeKind::Validation, 128, Some(1024), Some(2048));

        assert_eq!(event.kind, SystemMetricEventKind::ProbeHostMemory);
        assert_eq!(event.probe_kind, Some(ProbeKind::Validation));
        assert_eq!(event.candidate_microbatch, Some(128));
        assert_eq!(event.mem_available_bytes, Some(1024));
        assert_eq!(event.mem_total_bytes, Some(2048));
    }

    #[test]
    fn resource_snapshot_serializes_null_gpu_fields_as_absent() {
        let event = resource_snapshot_event("scan", 1.0, Some(12.5), Some(4096), None, None, 0);
        let encoded = format_system_metrics_event(&event);

        assert!(encoded.contains("\"kind\":\"resource_snapshot\""));
        assert!(encoded.contains("\"cpu_percent\":12.5"));
        assert!(encoded.contains("\"process_rss_bytes\":4096"));
        assert!(encoded.contains("\"gpu_oom_count\":0"));
        assert!(!encoded.contains("gpu_util_percent"));
        assert!(!encoded.contains("gpu_mem_used_mb"));
        assert!(!encoded.contains("gpu_mem_free_mb"));
    }

    #[test]
    fn progress_event_schema_is_stable() {
        let event = progress_event("manifest_scan", 25, 100, 2.0, Some(12.5), None);
        let encoded = format_system_metrics_event(&event);

        assert!(encoded.contains("\"kind\":\"progress\""));
        assert!(encoded.contains("\"phase\":\"manifest_scan\""));
        assert!(encoded.contains("\"completed\":25"));
        assert!(encoded.contains("\"planned\":100"));
        assert!(encoded.contains("\"files_per_second\":12.5"));
        assert!(!encoded.contains("samples_per_second"));
    }

    #[test]
    fn cpu_and_disk_parsers_handle_proc_fixtures() {
        assert_eq!(
            parse_proc_stat_total_ticks("cpu  1 2 3 4 5 6 7 8 9 10\n"),
            Some(55)
        );
        assert_eq!(
            parse_proc_self_stat_process_ticks(
                "123 (name with space) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
            ),
            Some(23)
        );
        assert_eq!(
            parse_proc_self_status_rss_bytes("Name:\tx\nVmRSS:\t 1234 kB\n"),
            Some(1_263_616)
        );
        assert_eq!(
            parse_proc_self_io("rchar: 1\nread_bytes: 1048576\nwrite_bytes: 2097152\n"),
            Some(DiskSample {
                read_bytes: 1_048_576,
                write_bytes: 2_097_152,
            })
        );
        assert_eq!(
            disk_rates_between(
                Some(DiskSample {
                    read_bytes: 0,
                    write_bytes: 0
                }),
                Some(DiskSample {
                    read_bytes: 1_048_576,
                    write_bytes: 2_097_152
                }),
                2.0,
            ),
            Some((0.5, 1.0))
        );
    }

    #[test]
    fn nvidia_smi_parser_accepts_sparse_csv_fixture() {
        assert_eq!(
            parse_nvidia_smi_csv("17, 1024, 23000\n"),
            Some(GpuTelemetry {
                util_percent: Some(17.0),
                mem_used_mb: Some(1024),
                mem_free_mb: Some(23000),
            })
        );
    }
}
