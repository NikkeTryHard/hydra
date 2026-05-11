use std::path::PathBuf;

use crate::preflight::ProbeKind;

use crate::config::{
    ProbeBatchChildRequest, ProbeChildRequest, ProbeCliRequest, ProbeSingleChildRequest,
    TrainConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeRequest {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub warmup_steps: usize,
    pub measure_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeBatchRequest {
    pub request: ProbeRequest,
    pub attempts: usize,
}

pub fn probe_request_from_cli(
    config: &TrainConfig,
    probe: Option<ProbeCliRequest>,
) -> Result<Option<ProbeRequest>, String> {
    let Some(probe) = probe else {
        return Ok(None);
    };
    let warmup_steps = probe.warmup_steps.unwrap_or(config.preflight.warmup_steps);
    let measure_steps = probe
        .measure_steps
        .unwrap_or(config.preflight.measure_steps);
    if probe.candidate_microbatch == 0 {
        return Err("--probe-candidate-microbatch must be greater than 0".to_string());
    }
    if warmup_steps == 0 {
        return Err("--probe-warmup-steps must be greater than 0".to_string());
    }
    if measure_steps == 0 {
        return Err("--probe-measure-steps must be greater than 0".to_string());
    }
    Ok(Some(ProbeRequest {
        kind: probe.kind,
        candidate_microbatch: probe.candidate_microbatch,
        warmup_steps,
        measure_steps,
    }))
}

pub fn probe_child_request_from_cli(
    child: Option<ProbeChildRequest>,
) -> Result<Option<(ProbeRequest, PathBuf, Option<PathBuf>)>, String> {
    let Some(child) = child else {
        return Ok(None);
    };
    match child {
        ProbeChildRequest::Single(child) => Ok(Some(resolve_probe_single_child_request(child)?)),
        ProbeChildRequest::Batch(_) => Ok(None),
    }
}

pub fn probe_batch_child_request_from_cli(
    child: Option<ProbeChildRequest>,
) -> Result<Option<(ProbeBatchRequest, PathBuf, Option<PathBuf>)>, String> {
    let Some(child) = child else {
        return Ok(None);
    };
    match child {
        ProbeChildRequest::Single(_) => Ok(None),
        ProbeChildRequest::Batch(child) => Ok(Some(resolve_probe_batch_child_request(child)?)),
    }
}

pub fn probe_candidate_ceiling(request: ProbeRequest) -> usize {
    request.candidate_microbatch.max(1)
}

pub fn resolve_probe_request(request: ProbeCliRequest) -> Result<ProbeRequest, String> {
    Ok(ProbeRequest {
        kind: request.kind,
        candidate_microbatch: request.candidate_microbatch,
        warmup_steps: request
            .warmup_steps
            .ok_or_else(|| "internal probe child missing resolved warmup steps".to_string())?,
        measure_steps: request
            .measure_steps
            .ok_or_else(|| "internal probe child missing resolved measure steps".to_string())?,
    })
}

pub fn resolve_probe_single_child_request(
    child: ProbeSingleChildRequest,
) -> Result<(ProbeRequest, PathBuf, Option<PathBuf>), String> {
    Ok((
        resolve_probe_request(child.request)?,
        child.result_path,
        child.manifest_cache_path,
    ))
}

pub fn resolve_probe_batch_child_request(
    child: ProbeBatchChildRequest,
) -> Result<(ProbeBatchRequest, PathBuf, Option<PathBuf>), String> {
    if child.attempts == 0 {
        return Err("internal probe batch child missing positive attempts".to_string());
    }
    Ok((
        ProbeBatchRequest {
            request: resolve_probe_request(child.request)?,
            attempts: child.attempts,
        },
        child.results_path,
        child.manifest_cache_path,
    ))
}

#[cfg(test)]
mod tests;
