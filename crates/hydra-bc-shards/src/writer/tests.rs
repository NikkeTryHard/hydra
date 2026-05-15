use super::*;
use crate::{
    BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS, BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION, BcShardBuildTotals, BcShardManifest, BcShardSidecarManifest,
    BcShardSplitManifest, load_bc_shard_reader,
};
use crate::{BC_SHARD_HEADER_SIZE, FLAG_EXIT, FLAG_SAFETY_RESIDUAL, record_size_for_flags};

fn dummy_sample() -> MjaiSample {
    let mut legal_mask = [0.0; HYDRA_ACTION_SPACE];
    legal_mask[3] = 1.0;
    let mut safety = [0.0; HYDRA_ACTION_SPACE];
    safety[1] = 0.5;
    let mut safety_mask = [0.0; HYDRA_ACTION_SPACE];
    safety_mask[1] = 1.0;
    let mut exit_target = [0.0; HYDRA_ACTION_SPACE];
    exit_target[3] = 0.75;
    let mut exit_mask = [0.0; HYDRA_ACTION_SPACE];
    exit_mask[3] = 1.0;
    MjaiSample {
        obs: [0.25; OBS_SIZE],
        action: 3,
        legal_mask,
        placement: 1,
        score_delta: 1200,
        grp_label: 7,
        oracle_target: Some([0.1, 0.2, 0.3, 0.4]),
        tenpai: [1.0, 0.0, 1.0],
        opp_next: [3, 8, 255],
        danger: [0.0; SPATIAL_TARGET_SIZE],
        danger_mask: [1.0; SPATIAL_TARGET_SIZE],
        safety_residual: Some(safety),
        safety_residual_mask: Some(safety_mask),
        exit_target: Some(exit_target),
        exit_mask: Some(exit_mask),
        delta_q_target: None,
        delta_q_mask: None,
        belief_fields: Some([0.0; 16 * 34]),
        mixture_weights: Some([0.25; 4]),
        belief_fields_present: true,
        mixture_weights_present: true,
    }
}

fn temp_output_dir(test_name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "hydra-bc-shards-{test_name}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("unnamed")
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("temp output dir should be created");
    path
}

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn manifest_for_descriptor(descriptor: BcShardDescriptor) -> BcShardManifest {
    BcShardManifest {
        manifest_version: BC_SHARD_MANIFEST_VERSION,
        shard_version: BC_SHARD_VERSION,
        shard_header_size: BC_SHARD_HEADER_SIZE,
        base_record_size: BC_BASE_RECORD_SIZE,
        max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        obs_size: OBS_SIZE,
        num_channels: NUM_CHANNELS,
        action_space: HYDRA_ACTION_SPACE,
        train_fraction: 1.0,
        shard_samples: descriptor.sample_count as usize,
        augment_runtime: false,
        input: "test".to_string(),
        output_dir: "test".to_string(),
        created_at: "test".to_string(),
        source_count: 1,
        source_total_games_hint: 1,
        source_train_count_hint: 1,
        source_val_count_hint: 0,
        source_counts_exact: true,
        exit_sidecar: None::<BcShardSidecarManifest>,
        delta_q_sidecar: None::<BcShardSidecarManifest>,
        totals: BcShardBuildTotals {
            sample_count: descriptor.sample_count,
            skipped_games: 0,
            empty_games: 0,
            shard_count: 1,
        },
        splits: vec![BcShardSplitManifest {
            split: descriptor.split,
            shard_count: 1,
            sample_count: descriptor.sample_count,
            feature_flags: descriptor.feature_flags,
            record_size: descriptor.record_size,
            shards: vec![descriptor],
        }],
    }
}

#[test]
fn compact_header_size_constant_matches_written_bytes() {
    let mut bytes = Vec::new();
    write_shard_header(
        &mut bytes,
        BcShardSplit::Train,
        2,
        10,
        100,
        FLAG_SAFETY_RESIDUAL,
        record_size_for_flags(FLAG_SAFETY_RESIDUAL),
    )
    .expect("header write should succeed");
    assert_eq!(bytes.len(), BC_SHARD_HEADER_SIZE as usize);
}

#[test]
fn compact_record_size_constant_matches_written_bytes() {
    let sample = dummy_sample();
    let flags = FLAG_SAFETY_RESIDUAL | FLAG_EXIT;
    let mut bytes = Vec::new();
    write_sample_record(&mut bytes, &sample, flags).expect("sample write should succeed");
    assert_eq!(bytes.len(), record_size_for_flags(flags) as usize);
}

#[test]
fn active_shard_writer_new_named_uses_custom_file_name_and_header_values() {
    let output_dir = temp_output_dir("new-named");
    let file_name = "chunk-0007-val-final.hydra-bc".to_string();
    let flags = FLAG_SAFETY_RESIDUAL | FLAG_EXIT;
    let sample = dummy_sample();
    let mut writer = ActiveShardWriter::new_named(
        &output_dir,
        BcShardSplit::Validation,
        7,
        42,
        flags,
        file_name.clone(),
    )
    .expect("named shard writer should open");
    writer
        .write_samples(std::slice::from_ref(&sample))
        .expect("sample should write");
    let descriptor = writer.finish().expect("writer should finish");

    let path = output_dir.join(&file_name);
    assert!(path.exists());
    assert_eq!(descriptor.file_name, file_name);
    assert_eq!(descriptor.shard_index, 7);
    assert_eq!(descriptor.sample_count, 1);
    assert_eq!(descriptor.first_sample_index, 42);
    assert_eq!(descriptor.feature_flags, flags);
    assert_eq!(descriptor.record_size, record_size_for_flags(flags));

    let bytes = std::fs::read(path).expect("shard file should be readable");
    assert_eq!(read_u32_at(&bytes, 16), record_size_for_flags(flags));
    assert_eq!(read_u32_at(&bytes, 20), BcShardSplit::Validation.split_id());
    assert_eq!(read_u32_at(&bytes, 24), 7);
    assert_eq!(read_u64_at(&bytes, 28), 1);
    assert_eq!(read_u64_at(&bytes, 48), 42);
    assert_eq!(read_u32_at(&bytes, 56), flags);

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn rewrite_shard_header_for_descriptor_updates_index_and_first_sample() {
    let output_dir = temp_output_dir("rewrite-header");
    let file_name = "chunk-0003-train-final.hydra-bc".to_string();
    let flags = FLAG_EXIT;
    let samples = [dummy_sample(), dummy_sample()];
    let mut writer = ActiveShardWriter::new_named(
        &output_dir,
        BcShardSplit::Train,
        0,
        0,
        flags,
        file_name.clone(),
    )
    .expect("named shard writer should open");
    writer
        .write_samples(&samples)
        .expect("samples should write");
    let mut descriptor = writer.finish().expect("writer should finish");
    descriptor.shard_index = 3;
    descriptor.first_sample_index = 17;

    let path = output_dir.join(&file_name);
    rewrite_shard_header_for_descriptor(&path, &descriptor).expect("header rewrite should succeed");

    let bytes = std::fs::read(&path).expect("shard file should be readable");
    assert_eq!(read_u32_at(&bytes, 16), record_size_for_flags(flags));
    assert_eq!(read_u32_at(&bytes, 20), BcShardSplit::Train.split_id());
    assert_eq!(read_u32_at(&bytes, 24), 3);
    assert_eq!(read_u64_at(&bytes, 28), 2);
    assert_eq!(read_u64_at(&bytes, 48), 17);
    assert_eq!(read_u32_at(&bytes, 56), flags);

    let mut reader_descriptor = descriptor.clone();
    reader_descriptor.shard_index = 0;
    reader_descriptor.first_sample_index = 0;
    let manifest = manifest_for_descriptor(reader_descriptor);
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");
    let reader = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .expect("reader should accept rewritten header");
    assert_eq!(reader.sample_count(), 2);
    assert_eq!(reader.feature_flags(), flags);

    let _ = std::fs::remove_dir_all(output_dir);
}
