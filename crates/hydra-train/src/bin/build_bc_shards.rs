use std::path::PathBuf;
use std::sync::Arc;

use hydra_bc_shards::BcShardSplitMode;
use hydra_replay_loader::mjai_loader::SidecarProvenance;
use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};
use hydra_train_exec::bc_shard_builder::{
    BcShardBuildOutput, BuildBcShardsConfig, build_bc_shards,
};
use hydra_train_exec::data_pipeline::scan_data_sources_with_progress;
use indicatif::{ProgressBar, ProgressStyle};

const DEFAULT_QUEUE_BOUND: usize = 128;
const DEFAULT_CHUNK_GAMES: usize = 10_000;
const DEFAULT_REPORT_NAME: &str = "bc_shard_build_report.json";
const DEFAULT_MAX_ERROR_EXAMPLES: usize = 32;

#[derive(Debug)]
struct Cli {
    input: PathBuf,
    output_dir: PathBuf,
    manifest_name: String,
    shard_samples: usize,
    train_fraction: f32,
    split_mode: BcShardSplitMode,
    exit_sidecar: Option<PathBuf>,
    exit_source_net_hash: Option<u64>,
    exit_source_version: Option<u32>,
    delta_q_sidecar: Option<PathBuf>,
    delta_q_source_net_hash: Option<u64>,
    delta_q_source_version: Option<u32>,
    num_threads: Option<usize>,
    queue_bound: usize,
    resume: bool,
    resume_dir: Option<PathBuf>,
    chunk_games: usize,
    report_name: Option<String>,
    progress_jsonl_name: Option<String>,
    dry_scan_only: bool,
    max_error_examples: usize,
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --input <dir|archive|replay> --output-dir <dir> [--manifest-name <file>] [--shard-samples <usize>] [--train-fraction <f32>] [--split train|val|both] [--exit-sidecar <path> --exit-source-net-hash <u64> --exit-source-version <u32>] [--delta-q-sidecar <path> --delta-q-source-net-hash <u64> --delta-q-source-version <u32>] [--num-threads <usize>] [--queue-bound <usize>] [--resume] [--resume-dir <dir>] [--chunk-games <usize>] [--report-name <file>|--no-report] [--progress-jsonl <file>] [--dry-scan-only] [--max-error-examples <usize>]"
    )
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    match args.next() {
        Some(value) if !value.starts_with("--") => Ok(value),
        _ => Err(format!("missing value for {flag}")),
    }
}

fn parse_usize_flag(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    next_value(args, flag)?
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag}: {err}"))
}

fn parse_nonzero_usize_flag(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<usize, String> {
    let value = parse_usize_flag(args, flag)?;
    if value == 0 {
        return Err(format!("{flag} must be > 0"));
    }
    Ok(value)
}

fn parse_split(value: &str) -> Result<BcShardSplitMode, String> {
    match value {
        "both" => Ok(BcShardSplitMode::Both),
        "train" => Ok(BcShardSplitMode::Train),
        "val" | "validation" => Ok(BcShardSplitMode::Validation),
        _ => Err(format!("invalid --split value: {value}")),
    }
}

fn parse_args<I>(program: &str, args: I) -> Result<Cli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let _ = args.next();

    let mut input = None;
    let mut output_dir = None;
    let mut manifest_name = "bc_shards_manifest.json".to_string();
    let mut shard_samples = 10_000usize;
    let mut train_fraction = 0.9f32;
    let mut split_mode = BcShardSplitMode::Both;
    let mut exit_sidecar = None;
    let mut exit_source_net_hash = None;
    let mut exit_source_version = None;
    let mut delta_q_sidecar = None;
    let mut delta_q_source_net_hash = None;
    let mut delta_q_source_version = None;
    let mut num_threads = None;
    let mut queue_bound = DEFAULT_QUEUE_BOUND;
    let mut resume = false;
    let mut resume_dir = None;
    let mut chunk_games = DEFAULT_CHUNK_GAMES;
    let mut report_name = Some(DEFAULT_REPORT_NAME.to_string());
    let mut progress_jsonl_name = None;
    let mut dry_scan_only = false;
    let mut max_error_examples = DEFAULT_MAX_ERROR_EXAMPLES;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(next_value(&mut args, "--input")?)),
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_value(&mut args, "--output-dir")?))
            }
            "--manifest-name" => manifest_name = next_value(&mut args, "--manifest-name")?,
            "--shard-samples" => {
                shard_samples = parse_usize_flag(&mut args, "--shard-samples")?;
            }
            "--train-fraction" => {
                train_fraction = next_value(&mut args, "--train-fraction")?
                    .parse::<f32>()
                    .map_err(|err| format!("invalid --train-fraction: {err}"))?;
            }
            "--split" => {
                split_mode = parse_split(&next_value(&mut args, "--split")?)?;
            }
            "--exit-sidecar" => {
                exit_sidecar = Some(PathBuf::from(next_value(&mut args, "--exit-sidecar")?))
            }
            "--exit-source-net-hash" => {
                exit_source_net_hash = Some(
                    next_value(&mut args, "--exit-source-net-hash")?
                        .parse::<u64>()
                        .map_err(|err| format!("invalid --exit-source-net-hash: {err}"))?,
                )
            }
            "--exit-source-version" => {
                exit_source_version = Some(
                    next_value(&mut args, "--exit-source-version")?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --exit-source-version: {err}"))?,
                )
            }
            "--delta-q-sidecar" => {
                delta_q_sidecar = Some(PathBuf::from(next_value(&mut args, "--delta-q-sidecar")?))
            }
            "--delta-q-source-net-hash" => {
                delta_q_source_net_hash = Some(
                    next_value(&mut args, "--delta-q-source-net-hash")?
                        .parse::<u64>()
                        .map_err(|err| format!("invalid --delta-q-source-net-hash: {err}"))?,
                )
            }
            "--delta-q-source-version" => {
                delta_q_source_version = Some(
                    next_value(&mut args, "--delta-q-source-version")?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --delta-q-source-version: {err}"))?,
                )
            }
            "--num-threads" => {
                num_threads = Some(parse_nonzero_usize_flag(&mut args, "--num-threads")?);
            }
            "--queue-bound" => {
                queue_bound = parse_nonzero_usize_flag(&mut args, "--queue-bound")?;
            }
            "--resume" => resume = true,
            "--resume-dir" => {
                resume_dir = Some(PathBuf::from(next_value(&mut args, "--resume-dir")?));
            }
            "--chunk-games" => {
                chunk_games = parse_nonzero_usize_flag(&mut args, "--chunk-games")?;
            }
            "--report-name" => report_name = Some(next_value(&mut args, "--report-name")?),
            "--no-report" => report_name = None,
            "--progress-jsonl" => {
                progress_jsonl_name = Some(next_value(&mut args, "--progress-jsonl")?);
            }
            "--dry-scan-only" => dry_scan_only = true,
            "--max-error-examples" => {
                max_error_examples = parse_usize_flag(&mut args, "--max-error-examples")?;
            }
            _ => return Err(usage(program)),
        }
    }

    let cli = Cli {
        input: input.ok_or_else(|| usage(program))?,
        output_dir: output_dir.ok_or_else(|| usage(program))?,
        manifest_name,
        shard_samples,
        train_fraction,
        split_mode,
        exit_sidecar,
        exit_source_net_hash,
        exit_source_version,
        delta_q_sidecar,
        delta_q_source_net_hash,
        delta_q_source_version,
        num_threads,
        queue_bound,
        resume,
        resume_dir,
        chunk_games,
        report_name,
        progress_jsonl_name,
        dry_scan_only,
        max_error_examples,
    };

    validate_sidecar_args(&cli)?;
    Ok(cli)
}

fn validate_sidecar_args(cli: &Cli) -> Result<(), String> {
    validate_one_sidecar(
        cli.exit_sidecar.as_ref(),
        cli.exit_source_net_hash,
        cli.exit_source_version,
        "exit",
    )?;
    validate_one_sidecar(
        cli.delta_q_sidecar.as_ref(),
        cli.delta_q_source_net_hash,
        cli.delta_q_source_version,
        "delta-q",
    )?;
    Ok(())
}

fn validate_one_sidecar(
    path: Option<&PathBuf>,
    source_net_hash: Option<u64>,
    source_version: Option<u32>,
    name: &str,
) -> Result<(), String> {
    match (path, source_net_hash, source_version) {
        (None, None, None) => Ok(()),
        (Some(_), Some(_), Some(_)) => Ok(()),
        _ => Err(format!(
            "{name} sidecar requires path, source-net-hash, and source-version together"
        )),
    }
}

fn load_exit_sidecar(path: Option<&PathBuf>) -> Result<Option<Arc<ExitSidecarIndex>>, String> {
    path.map(|path| ExitSidecarIndex::from_jsonl_path(path).map(Arc::new))
        .transpose()
        .map_err(|err| format!("failed to load exit sidecar: {err}"))
}

fn load_delta_q_sidecar(path: Option<&PathBuf>) -> Result<Option<Arc<DeltaQSidecarIndex>>, String> {
    path.map(|path| DeltaQSidecarIndex::from_jsonl_path(path).map(Arc::new))
        .transpose()
        .map_err(|err| format!("failed to load delta-q sidecar: {err}"))
}

fn make_progress_bar(len: usize) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} scan")
            .expect("progress style template should be valid"),
    );
    pb
}

fn summary(output: &BcShardBuildOutput) -> String {
    format!(
        "Wrote {} shard(s), {} sample(s) to {} (manifest: {}, skipped={}, empty={}, report={})",
        output.manifest.totals.shard_count,
        output.manifest.totals.sample_count,
        output.manifest.output_dir,
        output.manifest_path.display(),
        output.manifest.totals.skipped_games,
        output.manifest.totals.empty_games,
        output
            .report_path
            .as_ref()
            .map_or_else(|| "disabled".to_string(), |path| path.display().to_string())
    )
}

fn write_scan_report(
    cli: &Cli,
    scan: &hydra_train_exec::data_pipeline::DataManifest,
) -> Result<Option<PathBuf>, String> {
    let Some(report_name) = &cli.report_name else {
        return Ok(None);
    };
    std::fs::create_dir_all(&cli.output_dir)
        .map_err(|err| format!("failed to create output dir for scan report: {err}"))?;
    let report_path = cli.output_dir.join(report_name);
    let file = std::fs::File::create(&report_path)
        .map_err(|err| format!("failed to create scan report: {err}"))?;
    serde_json::to_writer_pretty(file, scan)
        .map_err(|err| format!("failed to write scan report: {err}"))?;
    Ok(Some(report_path))
}

fn builder_field_blocker(_cli: &Cli) -> Option<&'static str> {
    None
}

fn run() -> Result<(), String> {
    let program = "build_bc_shards";
    let cli = parse_args(program, std::env::args())?;

    let pb = make_progress_bar(0);
    let scan = scan_data_sources_with_progress(
        &cli.input,
        cli.train_fraction,
        &hydra_train_exec::data_pipeline::SourceFilterConfig::default(),
        Some(&pb),
    )
    .map_err(|err| format!("failed to scan replay sources: {err}"))?;
    pb.finish_and_clear();

    if cli.dry_scan_only {
        let report_path = write_scan_report(&cli, &scan)?;
        println!(
            "Scanned {} source(s), total_hint={} (report={})",
            scan.sources.len(),
            scan.total_games,
            report_path
                .as_ref()
                .map_or_else(|| "disabled".to_string(), |path| path.display().to_string())
        );
        return Ok(());
    }

    if let Some(blocker) = builder_field_blocker(&cli) {
        eprintln!(
            "warning: {blocker}; parsed CLI flags cannot be wired until builder support lands"
        );
    }

    let exit_sidecar = load_exit_sidecar(cli.exit_sidecar.as_ref())?;
    let delta_q_sidecar = load_delta_q_sidecar(cli.delta_q_sidecar.as_ref())?;

    let output = build_bc_shards(&BuildBcShardsConfig {
        input: cli.input,
        output_dir: cli.output_dir,
        manifest_name: cli.manifest_name,
        train_fraction: cli.train_fraction,
        shard_samples: cli.shard_samples,
        split_mode: cli.split_mode,
        source_manifest: Some(scan.clone()),
        exit_sidecar,
        exit_sidecar_path: cli.exit_sidecar,
        exit_provenance: SidecarProvenance::new(cli.exit_source_net_hash, cli.exit_source_version),
        delta_q_sidecar,
        delta_q_sidecar_path: cli.delta_q_sidecar,
        delta_q_provenance: SidecarProvenance::new(
            cli.delta_q_source_net_hash,
            cli.delta_q_source_version,
        ),
        num_threads: cli.num_threads,
        queue_bound: cli.queue_bound,
        resume: cli.resume,
        resume_dir: cli.resume_dir,
        chunk_games: cli.chunk_games,
        report_name: cli.report_name,
        progress_jsonl_name: cli.progress_jsonl_name,
        max_error_examples: cli.max_error_examples,
    })
    .map_err(|err| format!("failed to build BC shards: {err}"))?;

    println!(
        "{} (sources={}, total_hint={})",
        summary(&output),
        scan.sources.len(),
        scan.total_games
    );
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "build_bc_shards/tests.rs"]
mod tests;
