use std::fs;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::prelude::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

use hydra_train::model::HydraModelConfig;
use hydra_train::training::exit::ExitConfig;
use hydra_train::training::replay_exit::{
    ReplayExitRecordV1, replay_exit_records_for_identity, source_net_hash_from_checkpoint_identity,
};
use riichienv_core::replay::{MjaiEvent, load_mjai_events_from_path, read_mjai_events};

type Backend = LibTorch<f32>;

struct Cli {
    input: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
    source_version: u32,
    min_visits: Option<u32>,
    hard_state_threshold: Option<f32>,
    max_kl: Option<f32>,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --input <replay.json|replay.json.gz> --checkpoint <model_base> --output <sidecar.jsonl> --source-version <u32> [--min-visits <u32>] [--hard-state-threshold <f32>] [--max-kl <f32>]"
    )
}

fn parse_args<I>(args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args
        .next()
        .unwrap_or_else(|| "build_replay_exit_sidecar".to_string());

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
            _ => return Err(usage(&program)),
        }
    }

    Ok(Cli {
        input: input.ok_or_else(|| usage(&program))?,
        checkpoint: checkpoint.ok_or_else(|| usage(&program))?,
        output: output.ok_or_else(|| usage(&program))?,
        source_version: source_version.ok_or_else(|| usage(&program))?,
        min_visits,
        hard_state_threshold,
        max_kl,
    })
}

fn read_events(path: &Path) -> Result<Vec<MjaiEvent>, String> {
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

fn write_jsonl(path: &Path, records: &[ReplayExitRecordV1]) -> Result<(), String> {
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

fn source_net_hash_from_checkpoint(path: &Path) -> u64 {
    source_net_hash_from_checkpoint_identity(&path.display().to_string())
}

fn run() -> Result<(), String> {
    let cli = parse_args(std::env::args())?;
    let device = LibTorchDevice::Cpu;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = HydraModelConfig::learner()
        .init::<Backend>(&device)
        .load_file(&cli.checkpoint, &recorder, &device)
        .map_err(|err| {
            format!(
                "failed to load checkpoint {}: {err}",
                cli.checkpoint.display()
            )
        })?;

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

    let events = read_events(&cli.input)?;
    let source_identity = cli
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid replay filename {}", cli.input.display()))?;
    let source_net_hash = source_net_hash_from_checkpoint(&cli.checkpoint);
    let (records, report) = replay_exit_records_for_identity(
        source_identity,
        &events,
        &model,
        &device,
        &exit_cfg,
        source_net_hash,
        cli.source_version,
    )
    .map_err(|err| format!("failed to generate replay ExIt sidecar: {err}"))?;

    write_jsonl(&cli.output, &records)?;
    let report_path = cli.output.with_extension("report.json");
    let report_json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize report: {err}"))?;
    fs::write(&report_path, report_json)
        .map_err(|err| format!("failed to write report {}: {err}", report_path.display()))?;

    println!(
        "Wrote {} replay ExIt records to {} (report: {})",
        records.len(),
        cli.output.display(),
        report_path.display()
    );
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
