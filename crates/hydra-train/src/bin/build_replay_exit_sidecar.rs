use burn::backend::libtorch::LibTorchDevice;
#[path = "common/replay_sidecar_common.rs"]
mod replay_sidecar_common;

use self::replay_sidecar_common::{
    ReplaySidecarWriteRequest, build_exit_config, load_model, parse_args, read_events, write_jsonl,
    write_report, write_sidecar_with,
};
use hydra_train::training::replay_exit::replay_exit_records_for_identity;

fn run() -> Result<(), String> {
    let cli = parse_args("build_replay_exit_sidecar", std::env::args())?;
    let device = LibTorchDevice::Cpu;
    let model = load_model(&cli.checkpoint, &device)?;
    let exit_cfg = build_exit_config(&cli);

    let events = read_events(&cli.input)?;
    let summary = write_sidecar_with(
        ReplaySidecarWriteRequest {
            input: &cli.input,
            checkpoint: &cli.checkpoint,
            output: &cli.output,
            source_version: cli.source_version,
            lane_name: "replay ExIt sidecar",
            record_label: "replay ExIt records",
        },
        |source_identity, source_net_hash, source_version| {
            replay_exit_records_for_identity(
                source_identity,
                &events,
                &model,
                &device,
                &exit_cfg,
                source_net_hash,
                source_version,
            )
            .map_err(|err| err.to_string())
        },
        write_jsonl,
        write_report,
    )?;

    println!("{summary}");
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
    use super::run;

    #[test]
    fn run_reports_usage_without_required_args() {
        let err = run().expect_err("run without args should fail under test harness");
        assert!(err.contains("Usage:"));
    }
}
