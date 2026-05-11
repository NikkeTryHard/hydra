use std::fs;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::prelude::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use hydra_replay_sidecar::source_net_hash_from_checkpoint_identity;
use hydra_search_labels::exit::ExitConfig;
use hydra_train_exec::model::{HydraModel, HydraModelConfig, HydraModelInit};
use riichienv_core::replay::{MjaiEvent, load_mjai_events_from_path, read_mjai_events};
use serde::Serialize;

pub(super) type Backend = LibTorch<f32>;

#[derive(Debug)]
pub(super) struct ReplaySidecarCli {
    pub input: PathBuf,
    pub checkpoint: PathBuf,
    pub output: PathBuf,
    pub source_version: u32,
    pub min_visits: Option<u32>,
    pub hard_state_threshold: Option<f32>,
    pub max_kl: Option<f32>,
}

pub(super) fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --input <replay.json|replay.json.gz> --checkpoint <model_base> --output <sidecar.jsonl> --source-version <u32> [--min-visits <u32>] [--hard-state-threshold <f32>] [--max-kl <f32>]"
    )
}

pub(super) fn parse_args<I>(program: &str, args: I) -> Result<ReplaySidecarCli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let _ = args.next();

    let mut input = None;
    let mut checkpoint = None;
    let mut output = None;
    let mut source_version = None;
    let mut min_visits = None;
    let mut hard_state_threshold = None;
    let mut max_kl = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--checkpoint" => checkpoint = args.next().map(PathBuf::from),
            "--output" => output = args.next().map(PathBuf::from),
            "--source-version" => {
                source_version = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --source-version".to_string())?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --source-version: {err}"))?,
                )
            }
            "--min-visits" => {
                min_visits = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --min-visits".to_string())?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --min-visits: {err}"))?,
                )
            }
            "--hard-state-threshold" => {
                hard_state_threshold = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --hard-state-threshold".to_string())?
                        .parse::<f32>()
                        .map_err(|err| format!("invalid --hard-state-threshold: {err}"))?,
                )
            }
            "--max-kl" => {
                max_kl = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --max-kl".to_string())?
                        .parse::<f32>()
                        .map_err(|err| format!("invalid --max-kl: {err}"))?,
                )
            }
            _ => return Err(usage(program)),
        }
    }

    Ok(ReplaySidecarCli {
        input: input.ok_or_else(|| usage(program))?,
        checkpoint: checkpoint.ok_or_else(|| usage(program))?,
        output: output.ok_or_else(|| usage(program))?,
        source_version: source_version.ok_or_else(|| usage(program))?,
        min_visits,
        hard_state_threshold,
        max_kl,
    })
}

pub(super) fn read_events(path: &Path) -> Result<Vec<MjaiEvent>, String> {
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open replay {}: {err}", path.display()))?;
    let reader = BufReader::new(file);
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".json.gz"))
    {
        let gz = flate2::read::GzDecoder::new(reader);
        read_mjai_events(BufReader::new(gz))
            .map_err(|err| format!("failed to parse gz replay {}: {err}", path.display()))
    } else {
        load_mjai_events_from_path(path)
            .map_err(|err| format!("failed to parse replay {}: {err}", path.display()))
    }
}

pub(super) fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), String> {
    let mut file = fs::File::create(path)
        .map_err(|err| format!("failed to create sidecar {}: {err}", path.display()))?;
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|err| format!("failed to serialize sidecar row: {err}"))?;
        writeln!(file, "{line}")
            .map_err(|err| format!("failed to write sidecar {}: {err}", path.display()))?;
    }
    Ok(())
}

pub(super) fn source_identity_from_input(input: &Path) -> Result<&str, String> {
    input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid replay filename {}", input.display()))
}

pub(super) fn success_message(
    record_count: usize,
    output: &Path,
    report_path: &Path,
    record_label: &str,
) -> String {
    format!(
        "Wrote {record_count} {record_label} to {} (report: {})",
        output.display(),
        report_path.display()
    )
}

pub(super) struct ReplaySidecarWriteRequest<'a> {
    pub input: &'a Path,
    pub checkpoint: &'a Path,
    pub output: &'a Path,
    pub source_version: u32,
    pub lane_name: &'a str,
    pub record_label: &'a str,
}

pub(super) fn write_sidecar_with<Record, Report, Generate, WriteJsonl, WriteReport>(
    request: ReplaySidecarWriteRequest<'_>,
    generate: Generate,
    write_jsonl_fn: WriteJsonl,
    write_report_fn: WriteReport,
) -> Result<String, String>
where
    Record: Serialize,
    Report: Serialize,
    Generate: FnOnce(&str, u64, u32) -> Result<(Vec<Record>, Report), String>,
    WriteJsonl: FnOnce(&Path, &[Record]) -> Result<(), String>,
    WriteReport: FnOnce(&Path, &Report) -> Result<PathBuf, String>,
{
    let source_identity = source_identity_from_input(request.input)?;
    let source_net_hash = source_net_hash_from_checkpoint(request.checkpoint);
    let (records, report) = generate(source_identity, source_net_hash, request.source_version)
        .map_err(|err| format!("failed to generate {}: {err}", request.lane_name))?;

    write_jsonl_fn(request.output, &records)?;
    let report_path = write_report_fn(request.output, &report)?;

    Ok(success_message(
        records.len(),
        request.output,
        &report_path,
        request.record_label,
    ))
}

pub(super) fn load_model(
    checkpoint: &Path,
    device: &LibTorchDevice,
) -> Result<HydraModel<Backend>, String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    HydraModelConfig::learner()
        .init::<Backend>(device)
        .load_file(checkpoint, &recorder, device)
        .map_err(|err| format!("failed to load checkpoint {}: {err}", checkpoint.display()))
}

pub(super) fn build_exit_config(cli: &ReplaySidecarCli) -> ExitConfig {
    let mut exit_cfg = ExitConfig::default_phase3();
    if let Some(min_visits) = cli.min_visits {
        exit_cfg.min_visits = min_visits;
    }
    if let Some(hard_state_threshold) = cli.hard_state_threshold {
        exit_cfg.hard_state_threshold = hard_state_threshold;
    }
    if let Some(max_kl) = cli.max_kl {
        exit_cfg.safety_valve_max_kl = max_kl;
    }
    exit_cfg
}

pub(super) fn source_net_hash_from_checkpoint(path: &Path) -> u64 {
    source_net_hash_from_checkpoint_identity(&path.display().to_string())
}

pub(super) fn write_report<T: Serialize>(output: &Path, report: &T) -> Result<PathBuf, String> {
    let report_path = output.with_extension("report.json");
    let report_json = serde_json::to_string_pretty(report)
        .map_err(|err| format!("failed to serialize report: {err}"))?;
    fs::write(&report_path, report_json)
        .map_err(|err| format!("failed to write report {}: {err}", report_path.display()))?;
    Ok(report_path)
}

#[cfg(test)]
#[path = "replay_sidecar_common/tests.rs"]
mod tests;
