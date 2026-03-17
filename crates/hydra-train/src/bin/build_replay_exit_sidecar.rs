use burn::backend::libtorch::LibTorchDevice;
#[path = "common/replay_sidecar_common.rs"]
mod replay_sidecar_common;

use hydra_train::training::replay_exit::replay_exit_records_for_identity;

use self::replay_sidecar_common::{
    build_exit_config, load_model, parse_args, read_events, source_net_hash_from_checkpoint,
    write_jsonl, write_report,
};

fn run() -> Result<(), String> {
    let cli = parse_args("build_replay_exit_sidecar", std::env::args())?;
    let device = LibTorchDevice::Cpu;
    let model = load_model(&cli.checkpoint, &device)?;
    let exit_cfg = build_exit_config(&cli);

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
    let report_path = write_report(&cli.output, &report)?;

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
