//! Device timeline occupancy uses unions of measured CUDA intervals; summing
//! concurrent streams would overstate work and cannot prove SM utilization.

use serde_json::{Value, json};

fn union_length(intervals: &mut [(f64, f64)]) -> f64 {
    intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let mut end = f64::NEG_INFINITY;
    for &(start, stop) in intervals.iter() {
        if start >= end {
            total += stop - start;
        } else if stop > end {
            total += stop - end;
        }
        end = end.max(stop);
    }
    total
}

/// A named closed trace window includes launch gaps and starvation; excluding
/// these gaps would turn a timeline occupancy result into a misleading claim.
pub fn analyze_trace(document: &Value, scope: &str) -> Result<Value, String> {
    let events = document.get("traceEvents").and_then(Value::as_array)
        .ok_or_else(|| "CUDA trace lacks traceEvents array".to_string())?;
    let mut windows = Vec::new();
    for event in events {
        if event.get("name").and_then(Value::as_str) != Some(scope)
            || event.get("ph").and_then(Value::as_str) != Some("X") {
            continue;
        }
        let start = event.get("ts").and_then(Value::as_f64)
            .ok_or_else(|| "trace window lacks timestamp".to_string())?;
        let duration = event.get("dur").and_then(Value::as_f64)
            .ok_or_else(|| "trace window lacks duration".to_string())?;
        if !start.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err("trace window requires finite positive duration".to_string());
        }
        windows.push((start, start + duration));
    }
    if windows.len() != 1 {
        return Err(format!("expected one closed {scope:?} window, found {}", windows.len()));
    }
    let (begin, end) = windows[0];
    let mut kernels = Vec::new();
    let mut copies = Vec::new();
    let mut kernel_count = 0usize;
    for event in events {
        let category = event.get("cat").and_then(Value::as_str).unwrap_or("");
        if event.get("ph").and_then(Value::as_str) != Some("X")
            || !matches!(category, "kernel" | "gpu_memcpy" | "gpu_memset") {
            continue;
        }
        let start = event.get("ts").and_then(Value::as_f64)
            .ok_or_else(|| "device trace event lacks timestamp".to_string())?;
        let duration = event.get("dur").and_then(Value::as_f64)
            .ok_or_else(|| "device trace event lacks duration".to_string())?;
        if !start.is_finite() || !duration.is_finite() || duration < 0.0 {
            return Err("device trace intervals must be finite and nonnegative".to_string());
        }
        let left = start.max(begin);
        let right = (start + duration).min(end);
        if right <= left {
            continue;
        }
        if category == "kernel" {
            kernels.push((left, right));
            kernel_count += 1;
        } else {
            copies.push((left, right));
        }
    }
    if kernels.is_empty() {
        return Err("trace window contains no measured CUDA kernels".to_string());
    }
    let kernel_us = union_length(&mut kernels);
    let copy_us = union_length(&mut copies);
    kernels.extend_from_slice(&copies);
    let work_us = union_length(&mut kernels);
    let window_us = end - begin;
    Ok(json!({
        "scope": scope,
        "window_us": window_us,
        "kernel_union_us": kernel_us,
        "copy_union_us": copy_us,
        "device_work_union_us": work_us,
        "kernel_occupancy": kernel_us / window_us,
        "device_work_occupancy": work_us / window_us,
        "idle_fraction": 1.0 - work_us / window_us,
        "kernel_count": kernel_count,
        "sm_utilization": null,
        "tensorcore_utilization": null,
        "acceptance_kernel_occupancy_95": kernel_us / window_us >= 0.95,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_kernels_are_counted_once_and_launch_gaps_remain() {
        let trace = json!({"traceEvents": [
            {"name":"steady", "ph":"X", "ts":100.0,"dur":100.0},
            {"cat":"kernel", "ph":"X", "ts":90.0,"dur":40.0},
            {"cat":"kernel", "ph":"X", "ts":120.0,"dur":50.0},
            {"cat":"kernel", "ph":"X", "ts":180.0,"dur":40.0},
            {"cat":"gpu_memcpy", "ph":"X", "ts":170.0,"dur":10.0}
        ]});
        let result = analyze_trace(&trace, "steady").unwrap();
        assert_eq!(result["kernel_union_us"], json!(90.0));
        assert_eq!(result["copy_union_us"], json!(10.0));
        assert_eq!(result["device_work_union_us"], json!(100.0));
        assert_eq!(result["kernel_occupancy"], json!(0.9));
        assert_eq!(result["acceptance_kernel_occupancy_95"], json!(false));
        assert_eq!(result["sm_utilization"], Value::Null);
    }

    #[test]
    fn an_empty_or_ambiguous_window_cannot_claim_saturation() {
        assert!(analyze_trace(&json!({"traceEvents": []}), "steady").is_err());
        let trace = json!({"traceEvents": [
            {"name":"steady", "ph":"X", "ts":0.0,"dur":10.0},
            {"name":"steady", "ph":"X", "ts":20.0,"dur":10.0}
        ]});
        assert!(analyze_trace(&trace, "steady").is_err());
    }
}
