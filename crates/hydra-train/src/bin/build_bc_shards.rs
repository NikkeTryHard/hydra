use std::path::PathBuf;
use std::sync::Arc;

use hydra_replay_sidecar::{DeltaQSidecarIndex, ExitSidecarIndex};
use hydra_train::data::bc_shards::{
    BcShardBuildOutput, BcShardSplitMode, BuildBcShardsConfig, build_bc_shards,
};
use hydra_train::data::mjai_loader::SidecarProvenance;
use hydra_train::data::pipeline::scan_data_sources_with_progress;
use indicatif::{ProgressBar, ProgressStyle};

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
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} --input <dir|archive|replay> --output-dir <dir> [--manifest-name <file>] [--shard-samples <usize>] [--train-fraction <f32>] [--split train|val|both] [--exit-sidecar <path> --exit-source-net-hash <u64> --exit-source-version <u32>] [--delta-q-sidecar <path> --delta-q-source-net-hash <u64> --delta-q-source-version <u32>]"
    )
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

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => input = args.next().map(PathBuf::from),
            "--output-dir" => output_dir = args.next().map(PathBuf::from),
            "--manifest-name" => {
                manifest_name = args
                    .next()
                    .ok_or_else(|| "missing value for --manifest-name".to_string())?
            }
            "--shard-samples" => {
                shard_samples = args
                    .next()
                    .ok_or_else(|| "missing value for --shard-samples".to_string())?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --shard-samples: {err}"))?;
            }
            "--train-fraction" => {
                train_fraction = args
                    .next()
                    .ok_or_else(|| "missing value for --train-fraction".to_string())?
                    .parse::<f32>()
                    .map_err(|err| format!("invalid --train-fraction: {err}"))?;
            }
            "--split" => {
                split_mode = parse_split(
                    &args
                        .next()
                        .ok_or_else(|| "missing value for --split".to_string())?,
                )?;
            }
            "--exit-sidecar" => exit_sidecar = args.next().map(PathBuf::from),
            "--exit-source-net-hash" => {
                exit_source_net_hash = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --exit-source-net-hash".to_string())?
                        .parse::<u64>()
                        .map_err(|err| format!("invalid --exit-source-net-hash: {err}"))?,
                )
            }
            "--exit-source-version" => {
                exit_source_version = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --exit-source-version".to_string())?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --exit-source-version: {err}"))?,
                )
            }
            "--delta-q-sidecar" => delta_q_sidecar = args.next().map(PathBuf::from),
            "--delta-q-source-net-hash" => {
                delta_q_source_net_hash = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --delta-q-source-net-hash".to_string())?
                        .parse::<u64>()
                        .map_err(|err| format!("invalid --delta-q-source-net-hash: {err}"))?,
                )
            }
            "--delta-q-source-version" => {
                delta_q_source_version = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --delta-q-source-version".to_string())?
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --delta-q-source-version: {err}"))?,
                )
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
        "Wrote {} shard(s), {} sample(s) to {} (manifest: {})",
        output.manifest.totals.shard_count,
        output.manifest.totals.sample_count,
        output.manifest.output_dir,
        output.manifest_path.display()
    )
}

fn run() -> Result<(), String> {
    let program = "build_bc_shards";
    let cli = parse_args(program, std::env::args())?;

    let pb = make_progress_bar(0);
    let scan = scan_data_sources_with_progress(
        &cli.input,
        cli.train_fraction,
        &hydra_train::data::pipeline::SourceFilterConfig::default(),
        Some(&pb),
    )
    .map_err(|err| format!("failed to scan replay sources: {err}"))?;
    pb.finish_and_clear();

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
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_minimal_required_flags() {
        let cli = parse_args(
            "build_bc_shards",
            vec![
                "build_bc_shards".to_string(),
                "--input".to_string(),
                "replays".to_string(),
                "--output-dir".to_string(),
                "out".to_string(),
            ],
        )
        .expect("args should parse");

        assert_eq!(cli.input, PathBuf::from("replays"));
        assert_eq!(cli.output_dir, PathBuf::from("out"));
        assert_eq!(cli.shard_samples, 10_000);
        assert!((cli.train_fraction - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_args_rejects_partial_sidecar_provenance() {
        let err = parse_args(
            "build_bc_shards",
            vec![
                "build_bc_shards".to_string(),
                "--input".to_string(),
                "replays".to_string(),
                "--output-dir".to_string(),
                "out".to_string(),
                "--exit-sidecar".to_string(),
                "exit.jsonl".to_string(),
            ],
        )
        .expect_err("partial sidecar flags should fail");

        assert!(err.contains("exit sidecar requires path"));
    }

    #[test]
    fn parse_split_accepts_aliases() {
        assert!(matches!(parse_split("both"), Ok(BcShardSplitMode::Both)));
        assert!(matches!(parse_split("train"), Ok(BcShardSplitMode::Train)));
        assert!(matches!(
            parse_split("validation"),
            Ok(BcShardSplitMode::Validation)
        ));
    }
}
