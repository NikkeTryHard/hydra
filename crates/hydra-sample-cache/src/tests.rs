use super::*;
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
