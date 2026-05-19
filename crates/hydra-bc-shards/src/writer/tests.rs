use super::*;
use crate::manifest::COMPACT_OBS_BYTES;
use crate::{
    BC_BASE_RECORD_SIZE, BC_RECORD_SIZE_WITH_ALL_OPTIONALS, BC_SHARD_MANIFEST_VERSION,
    BC_SHARD_VERSION, BcShardBuildTotals, BcShardManifest, BcShardSidecarManifest,
    BcShardSplitManifest, load_bc_shard_reader,
};
use crate::{
    BC_DENSE_SHARD_MAGIC, BC_SHARD_HEADER_SIZE, DENSE_REBUILD_MESSAGE, FLAG_EXIT,
    FLAG_SAFETY_RESIDUAL, PACKED_ACTION_MASK_BYTES, record_size_for_flags,
};
use hydra_data_core::sample::{
    COMPACT_MISSING_TILE, CompactObservationFacts, CompactPlayerDiscards, CompactPlayerMelds,
    CompactSafetyFacts,
};

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
    let mut obs = [0.0; OBS_SIZE];
    for channel in [47, 51, 59, 60] {
        obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(0.25);
    }
    for channel in [71, 77] {
        obs[channel * TILE_COUNT] = 0.25;
    }
    let compact_facts = dense_oracle_compact_facts(&obs);
    MjaiSample {
        obs,
        compact_facts: Some(compact_facts),
        action: 3,
        legal_mask,
        placement: 1,
        score_delta: 1200,
        grp_label: 7,
        oracle_target: Some([0.1, 0.2, 0.3, 0.4]),
        tenpai: [1.0, 0.0, 1.0],
        opp_next: [3, 8, 255],
        danger: [0.0; SPATIAL_TARGET_SIZE],
        danger_mask: [0.0; SPATIAL_TARGET_SIZE],
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

fn dense_oracle_compact_facts(obs: &[f32; OBS_SIZE]) -> CompactObservationFacts {
    let mut facts = CompactObservationFacts {
        hand_counts: counts_from_thresholds(obs, 0),
        open_meld_counts: counts_from_thresholds(obs, 4),
        drawn_tile: one_hot_tile(obs, 8).unwrap_or(COMPACT_MISSING_TILE),
        shanten_base: 0,
        shanten_discard: [hydra_data_core::sample::COMPACT_MISSING_SHANTEN; TILE_COUNT],
        discards: [CompactPlayerDiscards::default(); 4],
        melds: [CompactPlayerMelds::default(); 4],
        dora_indicators: [0; 5],
        dora_indicator_count: 0,
        aka_flags: [false; 3],
        riichi: [false; 4],
        scores: [0; 4],
        kyoku_index: 0,
        honba: 0,
        kyotaku: 0,
        safety: CompactSafetyFacts {
            genbutsu_all: [0; 3],
            genbutsu_tedashi: [0; 3],
            genbutsu_riichi_era: [0; 3],
            suji: [[0.0; TILE_COUNT]; 3],
            half_suji: [0; 3],
            matagi: [[0.0; TILE_COUNT]; 3],
            kabe: 0,
            one_chance: 0,
            visible_counts: [0; TILE_COUNT],
            opponent_riichi: [false; 3],
            cached_tenpai_prob: [0.0; 3],
        },
        advanced_tail: None,
    };
    for (idx, dst) in facts.shanten_discard.iter_mut().enumerate() {
        let keep = obs[9 * TILE_COUNT + idx] != 0.0;
        let next = obs[10 * TILE_COUNT + idx] != 0.0;
        *dst = if next {
            -1
        } else if keep {
            0
        } else {
            hydra_data_core::sample::COMPACT_MISSING_SHANTEN
        };
    }
    for player in 0..4usize {
        let presence_ch = 11 + player * 3;
        let tedashi_ch = presence_ch + 1;
        let mut len = 0usize;
        for tile in 0..TILE_COUNT {
            if obs[presence_ch * TILE_COUNT + tile] != 0.0 {
                facts.discards[player].discards[len].tile = tile as u8;
                facts.discards[player].discards[len].is_tedashi =
                    obs[tedashi_ch * TILE_COUNT + tile] != 0.0;
                facts.discards[player].discards[len].turn = 0;
                len += 1;
            }
        }
        facts.discards[player].len = len as u8;
    }
    for player in 0..4usize {
        let base = 23 + player * 3;
        let mut len = 0usize;
        for kind in 0..3usize {
            for tile in 0..TILE_COUNT {
                if obs[(base + kind) * TILE_COUNT + tile] != 0.0 {
                    facts.melds[player].melds[len].tiles[0] = tile as u8;
                    facts.melds[player].melds[len].tile_count = 1;
                    facts.melds[player].melds[len].meld_type = match kind {
                        0 => hydra_data_core::sample::CompactMeldType::Chi,
                        1 => hydra_data_core::sample::CompactMeldType::Pon,
                        _ => hydra_data_core::sample::CompactMeldType::Kan,
                    };
                    len += 1;
                }
            }
        }
        facts.melds[player].len = len as u8;
    }
    let dora_counts = counts_from_dora(obs);
    for (tile, &count) in dora_counts.iter().enumerate() {
        for _ in 0..count {
            facts.dora_indicators[facts.dora_indicator_count as usize] = tile as u8;
            facts.dora_indicator_count += 1;
        }
    }
    for suit in 0..3usize {
        facts.aka_flags[suit] = obs[(40 + suit) * TILE_COUNT] != 0.0;
    }
    for player in 0..4usize {
        facts.riichi[player] = obs[(43 + player) * TILE_COUNT] != 0.0;
        facts.scores[player] = (obs[(47 + player) * TILE_COUNT] * 100_000.0) as i32;
    }
    facts.kyoku_index = (obs[59 * TILE_COUNT] * 8.0) as u8;
    facts.honba = (obs[60 * TILE_COUNT] * 10.0) as u8;
    facts.kyotaku = (obs[61 * TILE_COUNT] * 10.0) as u8;
    for opp in 0..3usize {
        facts.safety.genbutsu_all[opp] = bitset_from_channel(obs, 62 + opp);
        facts.safety.genbutsu_tedashi[opp] = bitset_from_channel(obs, 65 + opp);
        facts.safety.genbutsu_riichi_era[opp] = bitset_from_channel(obs, 68 + opp);
        facts.safety.suji[opp]
            .copy_from_slice(&obs[(71 + opp) * TILE_COUNT..(72 + opp) * TILE_COUNT]);
        facts.safety.half_suji[opp] = bitset_from_channel(obs, 74 + opp);
        facts.safety.matagi[opp]
            .copy_from_slice(&obs[(77 + opp) * TILE_COUNT..(78 + opp) * TILE_COUNT]);
        facts.safety.opponent_riichi[opp] = obs[(82 + opp) * TILE_COUNT] != 0.0;
    }
    facts.safety.kabe = bitset_from_channel(obs, 80);
    facts.safety.one_chance = bitset_from_channel(obs, 81);
    facts
}

fn counts_from_thresholds(obs: &[f32; OBS_SIZE], channel_start: usize) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for tile in 0..TILE_COUNT {
        for threshold in 0..4usize {
            counts[tile] += u8::from(obs[(channel_start + threshold) * TILE_COUNT + tile] != 0.0);
        }
    }
    counts
}

fn one_hot_tile(obs: &[f32; OBS_SIZE], channel: usize) -> Option<u8> {
    (0..TILE_COUNT)
        .find(|&tile| obs[channel * TILE_COUNT + tile] != 0.0)
        .map(|tile| tile as u8)
}

fn counts_from_dora(obs: &[f32; OBS_SIZE]) -> [u8; TILE_COUNT] {
    let mut counts = [0u8; TILE_COUNT];
    for tile in 0..TILE_COUNT {
        for threshold in 0..5usize {
            counts[tile] += u8::from(obs[(35 + threshold) * TILE_COUNT + tile] != 0.0);
        }
    }
    counts
}

fn bitset_from_channel(obs: &[f32; OBS_SIZE], channel: usize) -> u64 {
    let mut bits = 0u64;
    for tile in 0..TILE_COUNT {
        if obs[channel * TILE_COUNT + tile] != 0.0 {
            bits |= 1u64 << tile;
        }
    }
    bits
}

fn dense_obs_from_compact_facts(
    original: &[f32; OBS_SIZE],
    facts: &CompactObservationFacts,
) -> [f32; OBS_SIZE] {
    let mut obs = [0.0f32; OBS_SIZE];
    for (tile, &count) in facts.hand_counts.iter().enumerate() {
        for threshold in 0..4usize {
            if count as usize > threshold {
                obs[threshold * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    for (tile, &count) in facts.open_meld_counts.iter().enumerate() {
        for threshold in 0..4usize {
            if count as usize > threshold {
                obs[(4 + threshold) * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    if facts.drawn_tile != COMPACT_MISSING_TILE {
        obs[8 * TILE_COUNT + facts.drawn_tile as usize] = 1.0;
    }
    for (tile, &after) in facts.shanten_discard.iter().enumerate() {
        if after != hydra_data_core::sample::COMPACT_MISSING_SHANTEN {
            if after <= facts.shanten_base {
                obs[9 * TILE_COUNT + tile] = 1.0;
            }
            if after < facts.shanten_base {
                obs[10 * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    let dora_counts = counts_from_dora(original);
    for (tile, &count) in dora_counts.iter().enumerate() {
        for threshold in 0..5usize {
            if count as usize > threshold {
                obs[(35 + threshold) * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    for suit in 0..3usize {
        if facts.aka_flags[suit] {
            obs[(40 + suit) * TILE_COUNT..(41 + suit) * TILE_COUNT].fill(1.0);
        }
    }
    for channel in 43..47usize {
        if facts.riichi[channel - 43] {
            obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(1.0);
        }
    }
    for channel in 47..51usize {
        obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT]
            .fill(facts.scores[channel - 47] as f32 / 100_000.0);
    }
    let shanten = facts.shanten_base.clamp(0, 3) as usize;
    obs[(55 + shanten) * TILE_COUNT..(56 + shanten) * TILE_COUNT].fill(1.0);
    let my_score = facts.scores[0];
    for channel in 51..55usize {
        obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT]
            .fill((my_score - facts.scores[channel - 51]) as f32 / 30_000.0);
    }
    obs[59 * TILE_COUNT..60 * TILE_COUNT].fill(facts.kyoku_index as f32 / 8.0);
    obs[60 * TILE_COUNT..61 * TILE_COUNT].fill(facts.honba as f32 / 10.0);
    obs[61 * TILE_COUNT..62 * TILE_COUNT].fill(facts.kyotaku as f32 / 10.0);
    for channel in 62..85usize {
        obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT]
            .copy_from_slice(&original[channel * TILE_COUNT..(channel + 1) * TILE_COUNT]);
    }

    obs
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

fn assert_dense_eq(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (idx, (&a, &b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "dense mismatch at index {idx} channel {} tile {}: {a} != {b}",
            idx / TILE_COUNT,
            idx % TILE_COUNT
        );
    }
}

const NON_BINARY_METADATA_SCALAR_CHANNELS: [usize; 11] =
    [47, 48, 49, 50, 51, 52, 53, 54, 59, 60, 61];

fn set_non_binary_metadata_scalars(obs: &mut [f32; OBS_SIZE]) {
    for (idx, &channel) in NON_BINARY_METADATA_SCALAR_CHANNELS.iter().enumerate() {
        let value = 0.125 + idx as f32 * 0.03125;
        obs[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(value);
    }
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
        split_mode: "train".to_string(),
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
        storage_layout: crate::STORAGE_LAYOUT_COMPACT.to_string(),
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
fn compact_record_accepts_non_binary_metadata_scalars_and_matches_size() {
    let mut sample = dummy_sample();
    set_non_binary_metadata_scalars(&mut sample.obs);
    sample.compact_facts = Some(dense_oracle_compact_facts(&sample.obs));

    let mut bytes = Vec::new();
    write_sample_record(&mut bytes, &sample, 0)
        .expect("sample write should accept metadata scalars");

    assert_eq!(bytes.len(), record_size_for_flags(0) as usize);
}

#[test]
fn compact_record_hard_errors_when_advanced_observation_is_nonzero() {
    let mut sample = dummy_sample();
    sample.obs[150 * TILE_COUNT] = 1.0;

    let mut bytes = Vec::new();
    let err = write_sample_record(&mut bytes, &sample, 0).expect_err("advanced obs should reject");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("advanced observation channels"),
        "unexpected error: {err}"
    );
}

#[test]
fn compact_record_hard_errors_when_compact_facts_missing() {
    let mut sample = dummy_sample();
    sample.compact_facts = None;

    let mut bytes = Vec::new();
    let err = write_sample_record(&mut bytes, &sample, 0).expect_err("missing facts should reject");

    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string()
            .contains("compact observation facts required"),
        "unexpected error: {err}"
    );
}

#[test]
fn compact_record_packs_masks() {
    let sample = dummy_sample();
    let mut bytes = Vec::new();
    write_sample_record(&mut bytes, &sample, 0).expect("sample write should succeed");
    let legal_offset = COMPACT_OBS_BYTES + 1;
    assert_eq!(
        bytes[legal_offset..legal_offset + PACKED_ACTION_MASK_BYTES].len(),
        PACKED_ACTION_MASK_BYTES
    );
    assert_eq!(bytes[legal_offset], 0b0000_1000);
}

#[test]
fn compact_record_round_trips_observation_to_dense_host_batch() {
    let output_dir = temp_output_dir("obs-roundtrip");
    let file_name = "train-00000.hydra-bc".to_string();
    let mut sample = dummy_sample();
    sample.obs[0] = 1.0;
    sample.obs[8] = 1.0;
    sample.obs[71 * TILE_COUNT + 33] = 0.5;
    set_non_binary_metadata_scalars(&mut sample.obs);
    sample.compact_facts = Some(dense_oracle_compact_facts(&sample.obs));
    sample.obs = dense_obs_from_compact_facts(&sample.obs, sample.compact_facts.as_ref().unwrap());

    let mut writer =
        ActiveShardWriter::new_named(&output_dir, BcShardSplit::Train, 0, 0, 0, file_name.clone())
            .expect("writer should open");
    writer
        .write_samples(std::slice::from_ref(&sample))
        .expect("sample should write");
    let descriptor = writer.finish().expect("writer should finish");
    let manifest = manifest_for_descriptor(descriptor);
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let reader = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .expect("reader should load compact shard");
    let batch = reader
        .collate_host_batch_range(0, 1, false)
        .expect("batch should decode");
    assert_dense_eq(&batch.obs_flat[..OBS_SIZE], &sample.obs);
    assert_eq!(batch.legal_mask_flat[3], 1.0);
    assert_eq!(batch.actions[0], 3);

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn compact_record_augments_observation_suits_from_dense_oracle() {
    let output_dir = temp_output_dir("obs-augment");
    let file_name = "train-00000.hydra-bc".to_string();
    let mut sample = dummy_sample();
    sample.obs[0] = 1.0;
    sample.obs[9] = 1.0;
    sample.obs[71 * TILE_COUNT + 1] = 0.125;

    set_non_binary_metadata_scalars(&mut sample.obs);
    sample.compact_facts = Some(dense_oracle_compact_facts(&sample.obs));
    sample.obs = dense_obs_from_compact_facts(&sample.obs, sample.compact_facts.as_ref().unwrap());

    let mut writer =
        ActiveShardWriter::new_named(&output_dir, BcShardSplit::Train, 0, 0, 0, file_name.clone())
            .expect("writer should open");
    writer
        .write_samples(&[dummy_sample(), sample.clone()])
        .expect("samples should write");
    let descriptor = writer.finish().expect("writer should finish");
    let manifest = manifest_for_descriptor(descriptor);
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let reader = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .expect("reader should load compact shard");
    let batch = reader
        .collate_host_batch(&[1], true)
        .expect("batch should decode");
    let mut expected = [0.0f32; OBS_SIZE];
    for channel in 0..hydra_data_core::sample::COMPACT_BASELINE_CHANNELS {
        let src = channel * TILE_COUNT;
        let dst = src;
        expected[dst..dst + 9].copy_from_slice(&sample.obs[src..src + 9]);
        expected[dst + 9..dst + 18].copy_from_slice(&sample.obs[src + 18..src + 27]);
        expected[dst + 18..dst + 27].copy_from_slice(&sample.obs[src + 9..src + 18]);
        expected[dst + 27..dst + TILE_COUNT]
            .copy_from_slice(&sample.obs[src + 27..src + TILE_COUNT]);
    }
    assert_dense_eq(&batch.obs_flat[..OBS_SIZE], &expected);

    let _ = std::fs::remove_dir_all(output_dir);
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
    let manifest = manifest_for_descriptor(reader_descriptor);
    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");
    let err = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .err()
        .expect("reader should reject non-contiguous descriptor");
    assert!(
        err.contains("expected contiguous start"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(output_dir);
}

fn compact_reader_test_shard(
    test_name: &str,
) -> (std::path::PathBuf, std::path::PathBuf, BcShardDescriptor) {
    let output_dir = temp_output_dir(test_name);
    let file_name = "chunk-0000-train-final.hydra-bc".to_string();
    let samples = [dummy_sample(), dummy_sample()];
    let mut writer =
        ActiveShardWriter::new_named(&output_dir, BcShardSplit::Train, 0, 0, 0, file_name)
            .expect("test shard writer should open");
    writer
        .write_samples(&samples)
        .expect("test samples should write");
    let descriptor = writer.finish().expect("test shard should finish");
    let manifest_path = output_dir.join("manifest.json");
    (output_dir, manifest_path, descriptor)
}

fn write_manifest_for_descriptor(manifest_path: &std::path::Path, descriptor: BcShardDescriptor) {
    let manifest = manifest_for_descriptor(descriptor);
    std::fs::write(
        manifest_path,
        serde_json::to_vec(&manifest).expect("manifest should serialize"),
    )
    .expect("manifest should write");
}

fn corrupt_u64(path: &std::path::Path, offset: u64, value: u64) {
    use std::io::{Seek, SeekFrom, Write};

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("shard should open for corruption");
    file.seek(SeekFrom::Start(offset))
        .expect("corruption seek should succeed");
    file.write_all(&value.to_le_bytes())
        .expect("corruption write should succeed");
}

fn corrupt_magic(path: &std::path::Path, magic: [u8; 8]) {
    use std::io::{Seek, SeekFrom, Write};

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("shard should open for magic corruption");
    file.seek(SeekFrom::Start(0))
        .expect("magic corruption seek should succeed");
    file.write_all(&magic)
        .expect("magic corruption write should succeed");
}

#[test]
fn compact_reader_rejects_header_sample_count_mismatch() {
    let (output_dir, manifest_path, descriptor) =
        compact_reader_test_shard("header-sample-count-mismatch");
    let path = output_dir.join(&descriptor.file_name);
    write_manifest_for_descriptor(&manifest_path, descriptor);
    corrupt_u64(&path, 28, 1);

    let err = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .err()
        .expect("reader should reject mismatched header sample count");
    assert!(
        err.contains("sample count mismatch"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn compact_reader_rejects_header_first_sample_index_mismatch() {
    let (output_dir, manifest_path, descriptor) =
        compact_reader_test_shard("header-first-index-mismatch");
    let path = output_dir.join(&descriptor.file_name);
    write_manifest_for_descriptor(&manifest_path, descriptor);
    corrupt_u64(&path, 48, 12);

    let err = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .err()
        .expect("reader should reject mismatched first sample index");
    assert!(
        err.contains("first sample index mismatch"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn compact_reader_rejects_descriptor_byte_len_mismatch() {
    let (output_dir, manifest_path, mut descriptor) =
        compact_reader_test_shard("descriptor-byte-len-mismatch");
    descriptor.byte_len += 1;
    write_manifest_for_descriptor(&manifest_path, descriptor);

    let err = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .err()
        .expect("reader should reject descriptor byte length mismatch");
    assert!(err.contains("byte_len"), "unexpected error: {err}");

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn compact_reader_rejects_dense_magic() {
    let (output_dir, manifest_path, descriptor) = compact_reader_test_shard("dense-magic");
    let path = output_dir.join(&descriptor.file_name);
    write_manifest_for_descriptor(&manifest_path, descriptor);
    corrupt_magic(&path, BC_DENSE_SHARD_MAGIC);

    let err = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .err()
        .expect("reader should reject dense shard magic");
    assert_eq!(err, DENSE_REBUILD_MESSAGE);

    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn compact_reader_round_trips_current_compact_record() {
    let (output_dir, manifest_path, descriptor) =
        compact_reader_test_shard("current-compact-roundtrip");
    write_manifest_for_descriptor(&manifest_path, descriptor);

    let reader = load_bc_shard_reader(&manifest_path, BcShardSplit::Train)
        .expect("reader should load current compact record");
    let batch = reader
        .collate_host_batch_range(0, 2, false)
        .expect("current compact records should collate");
    let mut expected = dummy_sample();
    expected.obs = dense_obs_from_compact_facts(
        &expected.obs,
        expected
            .compact_facts
            .as_ref()
            .expect("compact facts should exist"),
    );
    assert_eq!(reader.sample_count(), 2);
    assert_dense_eq(&batch.obs_flat[..OBS_SIZE], &expected.obs);
    assert_dense_eq(&batch.obs_flat[OBS_SIZE..OBS_SIZE * 2], &expected.obs);
    assert_eq!(batch.actions[0], i64::from(expected.action));
    assert_eq!(batch.actions[1], i64::from(expected.action));

    let _ = std::fs::remove_dir_all(output_dir);
}
