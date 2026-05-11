use burn::backend::libtorch::LibTorchDevice;
#[path = "common/replay_sidecar_common.rs"]
mod replay_sidecar_common;

use self::replay_sidecar_common::{
    ReplaySidecarWriteRequest, build_exit_config, load_model, parse_args, read_events, write_jsonl,
    write_report, write_sidecar_with,
};
use hydra_search_labels::replay_delta_q::replay_delta_q_records_for_identity;

fn validate_source_version(source_version: u32) -> Result<(), String> {
    if source_version != 1 {
        return Err(format!(
            "unsupported --source-version {}; replay delta_q sidecars currently require source-version 1 to match train-side lookup",
            source_version
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let cli = parse_args("build_replay_delta_q_sidecar", std::env::args())?;
    validate_source_version(cli.source_version)?;
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
            lane_name: "replay delta_q sidecar",
            record_label: "replay delta_q records",
        },
        |source_identity, source_net_hash, source_version| {
            replay_delta_q_records_for_identity(
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
    use super::validate_source_version;

    #[test]
    fn delta_q_source_version_one_is_accepted() {
        validate_source_version(1).expect("source-version 1 should be accepted");
    }

    #[test]
    fn delta_q_source_version_other_values_are_rejected() {
        let err = validate_source_version(2).expect_err("non-1 source-version should fail");
        assert!(err.contains("source-version 1"));
    }
}
