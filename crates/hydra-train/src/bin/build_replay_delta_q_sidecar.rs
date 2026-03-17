use burn::backend::libtorch::LibTorchDevice;
#[path = "common/replay_sidecar_common.rs"]
mod replay_sidecar_common;

use hydra_train::training::replay_delta_q::replay_delta_q_records_for_identity;

use self::replay_sidecar_common::{
    build_exit_config, load_model, parse_args, read_events, source_net_hash_from_checkpoint,
    write_jsonl, write_report,
};

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
    let source_identity = cli
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid replay filename {}", cli.input.display()))?;
    let source_net_hash = source_net_hash_from_checkpoint(&cli.checkpoint);
    let (records, report) = replay_delta_q_records_for_identity(
        source_identity,
        &events,
        &model,
        &device,
        &exit_cfg,
        source_net_hash,
        cli.source_version,
    )
    .map_err(|err| format!("failed to generate replay delta_q sidecar: {err}"))?;

    write_jsonl(&cli.output, &records)?;
    let report_path = write_report(&cli.output, &report)?;

    println!(
        "Wrote {} replay delta_q records to {} (report: {})",
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
