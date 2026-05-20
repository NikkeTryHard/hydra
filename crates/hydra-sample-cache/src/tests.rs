use super::*;
use std::fs;
use std::io::{Cursor, ErrorKind};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after epoch")
        .as_nanos();
    PathBuf::from("/home/cachybtw/tmp")
        .join(format!("hydra_parsed_sample_cache_{label}_{unique}.cache"))
}

fn sample_with_optionals(action: u8) -> MjaiSample {
    let mut obs = [0.0f32; OBS_SIZE];
    obs[0] = 0.25;
    obs[OBS_SIZE - 1] = 0.75;

    let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
    legal_mask[action as usize] = 1.0;
    legal_mask[HYDRA_ACTION_SPACE - 1] = 1.0;

    let mut danger = [0.0f32; DANGER_TARGET_SIZE];
    danger[1] = 0.5;
    danger[51] = 0.25;
    let mut danger_mask = [0.0f32; DANGER_TARGET_SIZE];
    danger_mask[1] = 1.0;
    danger_mask[90] = 1.0;

    let mut oracle_target = [0.0f32; 4];
    oracle_target[0] = 0.4;
    oracle_target[3] = -0.2;

    let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual[3] = 0.8;
    let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
    safety_residual_mask[3] = 1.0;

    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    exit_target[2] = 1.0;
    let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
    exit_mask[2] = 1.0;

    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_target[5] = -0.25;
    let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
    delta_q_mask[5] = 1.0;

    let mut belief_fields = [0.0f32; BELIEF_FIELD_SIZE];
    belief_fields[0] = 0.1;
    belief_fields[BELIEF_FIELD_SIZE - 1] = 0.9;

    let mut mixture_weights = [0.0f32; 4];
    mixture_weights[0] = 0.7;
    mixture_weights[1] = 0.3;

    MjaiSample {
        obs,
        compact_facts: None,
        action,
        legal_mask,
        placement: 2,
        score_delta: -1_200,
        grp_label: 7,
        oracle_target: Some(oracle_target),
        tenpai: [0.1, 0.2, 0.3],
        opp_next: [1, 17, 255],
        danger,
        danger_mask,
        safety_residual: Some(safety_residual),
        safety_residual_mask: Some(safety_residual_mask),
        exit_target: Some(exit_target),
        exit_mask: Some(exit_mask),
        delta_q_target: Some(delta_q_target),
        delta_q_mask: Some(delta_q_mask),
        belief_fields: Some(belief_fields),
        mixture_weights: Some(mixture_weights),
        belief_fields_present: true,
        mixture_weights_present: true,
    }
}

fn valid_header_bytes(sample_count: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(PARSED_SAMPLE_CACHE_MAGIC);
    bytes.extend_from_slice(&PARSED_SAMPLE_CACHE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&sample_count.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes
}

fn valid_sample_bytes(
    flags: u16,
    belief_fields_present: u8,
    mixture_weights_present: u8,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_f32_array(&mut bytes, &[0.0; OBS_SIZE]).expect("obs should encode");
    write_u8(&mut bytes, 0).expect("action should encode");
    write_f32_array(&mut bytes, &[0.0; HYDRA_ACTION_SPACE]).expect("mask should encode");
    write_u8(&mut bytes, 0).expect("placement should encode");
    write_i32(&mut bytes, 0).expect("score should encode");
    write_u8(&mut bytes, 0).expect("grp should encode");
    write_f32_array(&mut bytes, &[0.0; 3]).expect("tenpai should encode");
    bytes.extend_from_slice(&[0u8; 3]);
    write_f32_array(&mut bytes, &[0.0; DANGER_TARGET_SIZE]).expect("danger should encode");
    write_f32_array(&mut bytes, &[0.0; DANGER_TARGET_SIZE]).expect("danger mask should encode");
    write_u8(&mut bytes, belief_fields_present).expect("belief bool should encode");
    write_u8(&mut bytes, mixture_weights_present).expect("mixture bool should encode");
    write_u16(&mut bytes, flags).expect("flags should encode");
    if flags & FLAG_BELIEF_FIELDS != 0 {
        write_f32_array(&mut bytes, &[0.0; BELIEF_FIELD_SIZE]).expect("belief should encode");
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        write_f32_array(&mut bytes, &[0.0; 4]).expect("mixture should encode");
    }
    bytes
}

fn assert_invalid_data<T>(result: io::Result<T>) {
    let err = match result {
        Ok(_) => panic!("corruption should fail"),
        Err(err) => err,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidData);
}

#[test]
fn writer_rejects_oversized_metadata_strings() {
    let path = unique_temp_path("oversized_metadata_write");
    let game = ParsedSampleCacheGame {
        samples: Vec::new(),
        final_scores: [25_000; 4],
    };
    let oversized = "x".repeat(MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN + 1);

    let result = write_parsed_sample_cache(&path, Path::new("source.json"), &oversized, &game);

    let err = result.expect_err("oversized metadata should fail before writing payload");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("metadata string length"));
    let _ = fs::remove_file(path);
}

fn assert_sample_eq(lhs: &MjaiSample, rhs: &MjaiSample) {
    assert_eq!(lhs.obs, rhs.obs);
    assert_eq!(lhs.action, rhs.action);
    assert_eq!(lhs.legal_mask, rhs.legal_mask);
    assert_eq!(lhs.placement, rhs.placement);
    assert_eq!(lhs.score_delta, rhs.score_delta);
    assert_eq!(lhs.grp_label, rhs.grp_label);
    assert_eq!(lhs.oracle_target, rhs.oracle_target);
    assert_eq!(lhs.tenpai, rhs.tenpai);
    assert_eq!(lhs.opp_next, rhs.opp_next);
    assert_eq!(lhs.danger, rhs.danger);
    assert_eq!(lhs.danger_mask, rhs.danger_mask);
    assert_eq!(lhs.safety_residual, rhs.safety_residual);
    assert_eq!(lhs.safety_residual_mask, rhs.safety_residual_mask);
    assert_eq!(lhs.exit_target, rhs.exit_target);
    assert_eq!(lhs.exit_mask, rhs.exit_mask);
    assert_eq!(lhs.delta_q_target, rhs.delta_q_target);
    assert_eq!(lhs.delta_q_mask, rhs.delta_q_mask);
    assert_eq!(lhs.belief_fields, rhs.belief_fields);
    assert_eq!(lhs.mixture_weights, rhs.mixture_weights);
    assert_eq!(lhs.belief_fields_present, rhs.belief_fields_present);
    assert_eq!(lhs.mixture_weights_present, rhs.mixture_weights_present);
}

#[test]
fn parsed_sample_cache_round_trips_game_and_metadata() {
    let path = unique_temp_path("round_trip");
    let game = ParsedSampleCacheGame {
        samples: vec![sample_with_optionals(3), sample_with_optionals(9)],
        final_scores: [31_000, 27_000, 23_000, 19_000],
    };
    let original_source_path = PathBuf::from("/data/raw/league_a/game_0001.mjai.json.gz");
    let original_identity = "league_a/game_0001.mjai.json.gz";

    write_parsed_sample_cache(&path, &original_source_path, original_identity, &game)
        .expect("cache write should succeed");

    let metadata =
        read_parsed_sample_cache_metadata(&path).expect("cache metadata read should succeed");
    assert_eq!(metadata.original_source_path, original_source_path);
    assert_eq!(metadata.original_identity, original_identity);
    assert_eq!(metadata.sample_count, game.samples.len());

    let loaded = load_parsed_sample_cache(&path).expect("cache load should succeed");
    assert_eq!(loaded.metadata, metadata);
    assert_eq!(loaded.game.final_scores, game.final_scores);
    assert_eq!(loaded.game.samples.len(), game.samples.len());
    for (lhs, rhs) in loaded.game.samples.iter().zip(game.samples.iter()) {
        assert_sample_eq(lhs, rhs);
    }

    fs::remove_file(path).ok();
}

#[test]
fn parsed_sample_cache_file_name_rewrites_mjai_suffix() {
    let file_name = parsed_sample_cache_file_name(Path::new("/data/game_0001.mjai.json.gz"))
        .expect("cache filename should build");
    assert_eq!(file_name, "game_0001.mjai.samples.cache");
    assert!(is_parsed_sample_cache_file(Path::new(&file_name)));
}

#[test]
fn parsed_sample_cache_rejects_header_magic_mismatch() {
    let mut bytes = valid_header_bytes(0);
    bytes[0] = b'X';

    assert_invalid_data(read_header_internal(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_unsupported_version() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(PARSED_SAMPLE_CACHE_MAGIC);
    bytes.extend_from_slice(&(PARSED_SAMPLE_CACHE_VERSION + 1).to_le_bytes());

    assert_invalid_data(read_header_internal(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_excessive_metadata_string_length() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(PARSED_SAMPLE_CACHE_MAGIC);
    bytes.extend_from_slice(&PARSED_SAMPLE_CACHE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(
        &((MAX_PARSED_SAMPLE_CACHE_METADATA_STRING_LEN as u32) + 1).to_le_bytes(),
    );

    assert_invalid_data(read_header_internal(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_excessive_sample_count() {
    let bytes = valid_header_bytes(MAX_PARSED_SAMPLE_CACHE_SAMPLES + 1);

    assert_invalid_data(read_header_internal(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_invalid_bool_in_sample() {
    let bytes = valid_sample_bytes(0, 2, 0);

    assert_invalid_data(read_sample(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_unknown_optional_flag_bits() {
    let bytes = valid_sample_bytes(1 << 15, 0, 0);

    assert_invalid_data(read_sample(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_belief_presence_flag_mismatch() {
    let bytes = valid_sample_bytes(0, 1, 0);

    assert_invalid_data(read_sample(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_mixture_presence_flag_mismatch() {
    let bytes = valid_sample_bytes(FLAG_MIXTURE_WEIGHTS, 0, 0);

    assert_invalid_data(read_sample(&mut Cursor::new(bytes)));
}

#[test]
fn parsed_sample_cache_rejects_writer_sample_count_over_format_limit() {
    let err = checked_sample_count((MAX_PARSED_SAMPLE_CACHE_SAMPLES as usize) + 1)
        .expect_err("oversized writer sample count should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn parsed_sample_cache_rejects_writer_sample_count_over_u32_capacity() {
    let err = checked_sample_count((u32::MAX as usize) + 1)
        .expect_err("u32-overflowing writer sample count should fail");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
