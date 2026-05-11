use super::*;
use crate::replay_targets::{StageABeliefAuditSummary, StageABeliefTarget};
use flate2::Compression;
use flate2::write::GzEncoder;
use hydra_replay_sidecar::{
    DeltaQSidecarIndex, ExitSidecarIndex, ReplayDecisionKey, ReplayDeltaQRecordV1,
    ReplayExitRecordV1, legal_mask_digest_from_f32,
};
use riichienv_core::action::{ActionType, Phase};
use riichienv_core::replay::read_mjai_events;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn dummy_game() -> MjaiGame {
    MjaiGame {
        samples: Vec::new(),
        final_scores: [25_000; 4],
    }
}

fn play_game_with_mjai_log(seed: u64) -> (Vec<String>, [i32; 4]) {
    let mut state = GameState::new(0, false, Some(seed), 0, GameRule::default_tenhou());
    let mut steps = 0u32;
    while !state.is_done && steps < 10_000 {
        if state.needs_initialize_next_round {
            state.step(&HashMap::new());
            continue;
        }
        let mut actions = HashMap::new();
        match state.phase {
            Phase::WaitAct => {
                let obs = state.get_observation(state.current_player);
                let legal = obs.legal_actions_method();
                if let Some(action) = legal.first().cloned() {
                    actions.insert(state.current_player, action);
                }
            }
            Phase::WaitResponse => {
                let active_players =
                    state.active_players[..state.active_player_count as usize].to_vec();
                for pid in active_players {
                    let obs = state.get_observation(pid);
                    if let Some(action) = obs.legal_actions_method().first().cloned() {
                        actions.insert(pid, action);
                    }
                }
            }
        }
        state.step(&actions);
        steps += 1;
    }
    (
        state.mjai_log.clone(),
        [
            state.players[0].score,
            state.players[1].score,
            state.players[2].score,
            state.players[3].score,
        ],
    )
}

#[test]
fn empty_dataset() {
    let ds = MjaiDataset::new(0.95);
    assert_eq!(ds.num_samples(), 0);
    let (train, eval) = ds.train_split();
    assert!(train.is_empty());
    assert!(eval.is_empty());
}

#[test]
fn train_fraction_is_clamped_in_constructor() {
    let ds = MjaiDataset::new(1.5);
    assert_eq!(ds.train_fraction, 1.0);
    let ds = MjaiDataset::new(-0.25);
    assert_eq!(ds.train_fraction, 0.0);
}

#[test]
fn train_split_clamps_mutated_fraction() {
    let mut ds = MjaiDataset::new(0.5);
    ds.add_game(dummy_game());
    ds.add_game(dummy_game());
    ds.add_game(dummy_game());
    ds.train_fraction = 2.0;
    let (train, eval) = ds.train_split();
    assert_eq!(train.len(), 3);
    assert_eq!(eval.len(), 0);
    ds.train_fraction = -1.0;
    let (train, eval) = ds.train_split();
    assert_eq!(train.len(), 0);
    assert_eq!(eval.len(), 3);
}

#[test]
fn train_split_handles_nan_fraction() {
    let mut ds = MjaiDataset::new(0.5);
    ds.add_game(dummy_game());
    ds.add_game(dummy_game());
    ds.train_fraction = f32::NAN;
    let (train, eval) = ds.train_split();
    assert_eq!(train.len(), 0);
    assert_eq!(eval.len(), 2);
}

#[test]
fn load_game_from_reader_extracts_samples() {
    let (log, final_scores) = play_game_with_mjai_log(0);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    assert_eq!(game.final_scores, final_scores);
    assert!(game.samples.len() > 50, "expected a real replay sample set");
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.legal_mask[sample.action as usize] > 0.0)
    );
}

#[test]
fn load_game_from_reader_populates_oracle_targets_from_final_scores() {
    let (log, final_scores) = play_game_with_mjai_log(7);
    let game = load_game_from_reader_with_sidecar(
        "game-7",
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        ReplayTargetProfile::with_optional_heads(true, false, false, false, false, false),
        Cursor::new(log.join("\n")),
        None,
        None,
    )
    .expect("load game");
    let expected = oracle_target_from_scores(final_scores);
    assert!(
        !game.samples.is_empty(),
        "expected replay to produce samples"
    );
    for sample in game.samples.iter().take(8) {
        let got_target = sample
            .oracle_target
            .expect("oracle target should be present");
        for (got, want) in got_target.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "oracle target mismatch: {got} vs {want}"
            );
        }
    }
}

#[test]
fn load_game_from_reader_keeps_delta_q_absent_in_replay_samples() {
    let (log, _) = play_game_with_mjai_log(23);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    assert!(
        !game.samples.is_empty(),
        "expected replay loader to produce samples"
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_target.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_mask.is_none())
    );
}

#[test]
fn load_game_from_reader_with_sidecar_keeps_delta_q_absent_when_sidecar_not_configured() {
    let (log, _) = play_game_with_mjai_log(29);
    let game = load_game_from_reader_with_sidecar(
        "game-29",
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::default(),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, false, true),
        Cursor::new(log.join("\n")),
        None,
        None,
    )
    .expect("load game");
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_target.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_mask.is_none())
    );
}

fn replay_sidecar_guardrail_log() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

fn replay_guardrail_decisions(
    source_identity: &str,
) -> Vec<(ReplayDecisionKey, u8, [f32; HYDRA_ACTION_SPACE])> {
    let events =
        read_mjai_events(Cursor::new(replay_sidecar_guardrail_log())).expect("parse events");
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut decisions = Vec::new();

    for (idx, event) in events.iter().enumerate() {
        if let Some(decision) = prepare_replay_decision(event, &mut state, &safety, &mut encoder)
            .expect("prepare replay decision")
        {
            decisions.push((
                ReplayDecisionKey {
                    source_hash: source_hash_from_identity(source_identity),
                    event_index: idx as u32,
                    actor: decision.actor as u8,
                    obs_hash: obs_hash(&decision.obs_encoded),
                },
                decision.action_id,
                decision.legal_mask_f32,
            ));
        }
        update_safety(&mut safety, event).expect("update safety");
        state.apply_mjai_event(event.clone());
    }

    decisions
}

fn synthetic_exit_records(
    source_identity: &str,
    source_net_hash: u64,
    source_version: u32,
) -> Vec<ReplayExitRecordV1> {
    replay_guardrail_decisions(source_identity)
        .into_iter()
        .take(2)
        .map(|(key, action, legal_mask)| {
            let mut target = [0.0f32; HYDRA_ACTION_SPACE];
            let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
            mask[action as usize] = 1.0;
            target[action as usize] = 1.0;
            ReplayExitRecordV1 {
                version: 1,
                semantics: hydra_replay_sidecar::REPLAY_EXIT_SEMANTICS_V1.to_string(),
                provenance: hydra_replay_sidecar::REPLAY_EXIT_PROVENANCE.to_string(),
                key,
                action,
                legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                source_net_hash,
                source_version,
                root_visit_count: 64,
                legal_discard_count: legal_mask[..=DISCARD_END as usize]
                    .iter()
                    .filter(|&&value| value > 0.0)
                    .count() as u8,
                supported_actions: 1,
                coverage: 1.0,
                kl_to_base: 0.0,
                target: target.to_vec(),
                mask: mask.to_vec(),
            }
        })
        .collect()
}

fn synthetic_delta_q_records(
    source_identity: &str,
    source_net_hash: u64,
    source_version: u32,
) -> Vec<ReplayDeltaQRecordV1> {
    replay_guardrail_decisions(source_identity)
        .into_iter()
        .take(2)
        .map(|(key, action, legal_mask)| {
            let mut target = [0.0f32; HYDRA_ACTION_SPACE];
            let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
            mask[action as usize] = 1.0;
            target[action as usize] = 0.25;
            ReplayDeltaQRecordV1 {
                version: 1,
                semantics: hydra_replay_sidecar::REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                provenance: hydra_replay_sidecar::REPLAY_DELTA_Q_PROVENANCE.to_string(),
                key,
                action,
                legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                source_net_hash,
                source_version,
                target: target.to_vec(),
                mask: mask.to_vec(),
            }
        })
        .collect()
}

fn unique_loader_temp_path(prefix: &str, file_name: &str) -> PathBuf {
    let base = std::env::temp_dir();
    fs::create_dir_all(&base).expect("create loader temp root");
    base.join(format!(
        "{prefix}_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ))
    .join(file_name)
}

#[test]
fn loader_replay_key_parity_matches_exit_and_delta_q_sidecars() {
    let log = replay_sidecar_guardrail_log();
    let events = read_mjai_events(Cursor::new(log)).expect("parse events");
    let exit_records = synthetic_exit_records("game-1", 123, 1);
    let delta_q_records = synthetic_delta_q_records("game-1", 123, 1);

    assert!(
        !exit_records.is_empty() || !delta_q_records.is_empty(),
        "expected at least one search-derived replay record"
    );

    let exit_keys: std::collections::BTreeSet<_> = exit_records
        .iter()
        .map(|record| {
            (
                record.key.source_hash,
                record.key.event_index,
                record.key.actor,
                record.key.obs_hash,
                record.action,
            )
        })
        .collect();
    let delta_q_keys: std::collections::BTreeSet<_> = delta_q_records
        .iter()
        .map(|record| {
            (
                record.key.source_hash,
                record.key.event_index,
                record.key.actor,
                record.key.obs_hash,
                record.action,
            )
        })
        .collect();

    let game = load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(123), Some(1)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    )
    .expect("load with both sidecars");

    let mut loader_state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut exit_joined = std::collections::BTreeSet::new();
    let mut delta_q_joined = std::collections::BTreeSet::new();
    for (idx, event) in read_mjai_events(Cursor::new(replay_sidecar_guardrail_log()))
        .expect("parse events for parity")
        .iter()
        .enumerate()
    {
        if let Some(decision) =
            prepare_replay_decision(event, &mut loader_state, &safety, &mut encoder)
                .expect("prepare replay decision")
        {
            let tuple = (
                source_hash_from_identity("game-1"),
                idx as u32,
                decision.actor as u8,
                obs_hash(&decision.obs_encoded),
                decision.action_id,
            );
            if exit_keys.contains(&tuple) {
                exit_joined.insert(tuple);
            }
            if delta_q_keys.contains(&tuple) {
                delta_q_joined.insert(tuple);
            }
        }
        update_safety(&mut safety, event).expect("update safety");
        loader_state.apply_mjai_event(event.clone());
    }

    assert_eq!(
        exit_joined, exit_keys,
        "loader replay keys should match exit sidecar keys"
    );
    assert_eq!(
        delta_q_joined, delta_q_keys,
        "loader replay keys should match delta_q sidecar keys"
    );
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.exit_target.is_some())
    );
    assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some())
    );
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_mask.is_some())
    );
}

#[test]
fn mismatched_obs_hash_prevents_sidecar_hydration() {
    let log = replay_sidecar_guardrail_log();
    let events = read_mjai_events(Cursor::new(log)).expect("parse events");
    let mut exit_records = synthetic_exit_records("game-1", 123, 1);
    let mut delta_q_records = synthetic_delta_q_records("game-1", 123, 1);

    assert!(!exit_records.is_empty(), "expected exit sidecar records");
    assert!(
        !delta_q_records.is_empty(),
        "expected delta_q sidecar records"
    );

    for record in &mut exit_records {
        record.key.obs_hash = record.key.obs_hash.wrapping_add(1);
    }
    for record in &mut delta_q_records {
        record.key.obs_hash = record.key.obs_hash.wrapping_add(1);
    }

    let game = load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(123), Some(1)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    )
    .expect("load with mismatched obs_hash sidecars");

    assert!(
        game.samples
            .iter()
            .all(|sample| sample.exit_target.is_none())
    );
    assert!(game.samples.iter().all(|sample| sample.exit_mask.is_none()));
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_target.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_mask.is_none())
    );
}

#[test]
fn mismatched_exit_provenance_does_not_block_delta_q_hydration() {
    let log = replay_sidecar_guardrail_log();
    let events = read_mjai_events(Cursor::new(log)).expect("parse events");
    let exit_records = synthetic_exit_records("game-1", 123, 1);
    let delta_q_records = synthetic_delta_q_records("game-1", 456, 2);

    let game = load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(999), Some(99)),
        SidecarProvenance::new(Some(456), Some(2)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    )
    .expect("load with mismatched exit provenance");

    assert!(
        game.samples
            .iter()
            .all(|sample| sample.exit_target.is_none())
    );
    assert!(game.samples.iter().all(|sample| sample.exit_mask.is_none()));
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some())
    );
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_mask.is_some())
    );
}

#[test]
fn mismatched_delta_q_provenance_does_not_block_exit_hydration() {
    let log = replay_sidecar_guardrail_log();
    let events = read_mjai_events(Cursor::new(log)).expect("parse events");
    let exit_records = synthetic_exit_records("game-1", 123, 1);
    let delta_q_records = synthetic_delta_q_records("game-1", 456, 2);

    let game = load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(999), Some(99)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    )
    .expect("load with mismatched delta_q provenance");

    assert!(
        game.samples
            .iter()
            .any(|sample| sample.exit_target.is_some())
    );
    assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_target.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_mask.is_none())
    );
}

#[test]
fn load_game_from_reader_uses_minimal_bc_profile_by_default() {
    let (log, _) = play_game_with_mjai_log(11);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    let sample = game
        .samples
        .iter()
        .find(|s| s.action <= DISCARD_END)
        .expect("discard sample");
    assert!(sample.safety_residual.is_none());
    assert!(sample.safety_residual_mask.is_none());

    let mask_offset = hydra_core::encoder::HAND_EV_MASK_CHANNEL * 34;
    assert_eq!(
        sample.obs[mask_offset], 0.0,
        "default BC loader path should leave Hand-EV mask disabled"
    );

    let hand_ev_payload = &sample.obs[hydra_core::encoder::HAND_EV_CHANNEL_START * 34..mask_offset];
    assert!(
        hand_ev_payload.iter().all(|&v| v == 0.0),
        "default BC loader path should zero Hand-EV payload"
    );

    assert_eq!(sample.tenpai, [0.0; 3]);
    assert_eq!(sample.opp_next, [MISSING_TILE_TARGET; 3]);
    assert!(sample.danger.iter().all(|&v| v == 0.0));
    assert!(sample.danger_mask.iter().all(|&v| v == 0.0));
}

#[test]
fn load_game_from_path_with_policy_uses_file_name_identity_for_sidecars() {
    let path = unique_loader_temp_path("loader_policy_path", "game-1.mjai.json");
    let parent = path.parent().expect("temp file parent");
    fs::create_dir_all(parent).expect("create temp parent");
    fs::write(&path, replay_sidecar_guardrail_log()).expect("write replay log");

    let exit_records = synthetic_exit_records("game-1.mjai.json", 123, 1);
    let delta_q_records = synthetic_delta_q_records("game-1.mjai.json", 456, 2);
    let exit_index = ExitSidecarIndex::from_records(exit_records);
    let delta_q_index = DeltaQSidecarIndex::from_records(delta_q_records);
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(456), Some(2)),
        Some(&exit_index),
        Some(&delta_q_index),
    );

    let game = load_game_from_path_with_policy(&path, Some(&policy)).expect("load with policy");
    fs::remove_file(&path).ok();
    fs::remove_dir_all(parent).ok();

    assert!(
        game.samples
            .iter()
            .any(|sample| sample.exit_target.is_some())
    );
    assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some())
    );
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_mask.is_some())
    );
}

#[test]
fn load_game_from_stream_with_policy_uses_explicit_source_identity() {
    let source_identity = "archive.tar.zst/game-1.mjai.json";
    let exit_records = synthetic_exit_records(source_identity, 123, 1);
    let delta_q_records = synthetic_delta_q_records(source_identity, 456, 2);
    let exit_index = ExitSidecarIndex::from_records(exit_records);
    let delta_q_index = DeltaQSidecarIndex::from_records(delta_q_records);
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(456), Some(2)),
        Some(&exit_index),
        Some(&delta_q_index),
    );

    let game = load_game_from_stream_with_policy(
        source_identity,
        Cursor::new(replay_sidecar_guardrail_log()),
        Some(&policy),
    )
    .expect("load stream with policy");

    assert!(
        game.samples
            .iter()
            .any(|sample| sample.exit_target.is_some())
    );
    assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some())
    );
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.delta_q_mask.is_some())
    );
}

#[test]
fn load_game_from_stream_with_empty_policy_falls_back_to_default_loader() {
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        None,
        None,
    );

    let game = load_game_from_stream_with_policy(
        "archive.tar.zst/game-1.mjai.json",
        Cursor::new(replay_sidecar_guardrail_log()),
        Some(&policy),
    )
    .expect("load stream without sidecars");

    assert!(
        game.samples
            .iter()
            .all(|sample| sample.exit_target.is_none())
    );
    assert!(game.samples.iter().all(|sample| sample.exit_mask.is_none()));
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_target.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.delta_q_mask.is_none())
    );
}

#[test]
fn build_safety_residual_targets_uses_signed_exact_safety_correction() {
    let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
    legal_mask[0] = 1.0;
    legal_mask[1] = 1.0;
    legal_mask[2] = 1.0;
    legal_mask[AKA_5M as usize] = 1.0;

    let mut safety = SafetyInfo::default();
    hydra_core::safety::bit_set(&mut safety.genbutsu_all[0], 1);
    hydra_core::safety::bit_set(&mut safety.genbutsu_all[0], 4);

    let mut wait_sets = [[0.0f32; 34]; 3];
    wait_sets[1][4] = 1.0;

    let (target, mask) = build_safety_residual_targets(&legal_mask, &safety, &wait_sets);

    assert!(
        (target[0] - 1.0).abs() < 1e-6,
        "safe tile with public score 0 should become +1 residual"
    );
    assert!(
        target[1].abs() < 1e-6,
        "safe tile with public score 1 should have zero residual"
    );
    assert!(
        (target[2] - 1.0).abs() < 1e-6,
        "safe tile with public score 0 should become +1 residual"
    );
    assert!(
        (target[AKA_5M as usize] + 1.0).abs() < 1e-6,
        "aka tile should map to base tile before residual computation"
    );
    assert_eq!(mask[0], 1.0);
    assert_eq!(mask[1], 1.0);
    assert_eq!(mask[2], 1.0);
    assert_eq!(mask[AKA_5M as usize], 1.0);
    assert!(
        target
            .iter()
            .zip(mask.iter())
            .all(|(&value, &mask_value)| { mask_value <= 0.0 || (-1.0..=1.0).contains(&value) })
    );
}

#[test]
fn exact_waits_returns_empty_waits_for_furiten_tenpai() {
    let mut state = GameState::new(0, false, Some(0), 0, GameRule::default_tenhou());
    let hand = [0u8, 4, 8, 12, 16, 20, 36, 40, 44, 72, 76, 80, 108];
    state.players[0].hand[..hand.len()].copy_from_slice(&hand);
    state.players[0].hand_len = hand.len() as u8;
    state.players[0].discards[0] = 109;
    state.players[0].discard_len = 1;

    let (waits, tenpai) = exact_waits(&state, 0);
    assert!(tenpai, "furiten hand should still register as tenpai");
    assert!(waits.iter().all(|&value| value == 0.0));
}

#[test]
fn load_game_from_reader_keeps_stage_a_belief_targets_truthful_when_emitted() {
    for seed in 0..1u64 {
        let (log, _) = play_game_with_mjai_log(seed);
        let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
        for sample in game.samples {
            match sample.belief_fields {
                Some(belief) => {
                    assert_eq!(belief.len(), 16 * 34);
                    assert!(sample.belief_fields_present);
                }
                None => assert!(!sample.belief_fields_present),
            }
            assert!(sample.mixture_weights.is_none());
            assert!(!sample.mixture_weights_present);
        }
    }
}

#[test]
fn load_game_from_reader_keeps_stage_a_mixture_targets_default_off() {
    let (log, _) = play_game_with_mjai_log(19);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    assert!(
        !game.samples.is_empty(),
        "expected replay to produce samples"
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.mixture_weights.is_none())
    );
    assert!(
        game.samples
            .iter()
            .all(|sample| !sample.mixture_weights_present)
    );
}

#[test]
fn should_sample_replay_event_skips_reach_and_hora() {
    let dahai = MjaiEvent::Dahai {
        actor: 0,
        pai: "1m".to_string(),
        tsumogiri: false,
    };
    let reach = MjaiEvent::Reach { actor: 0 };
    let hora = MjaiEvent::Hora {
        actor: 0,
        target: 1,
        pai: Some("1m".to_string()),
        uradora_markers: None,
        yaku: None,
        fu: None,
        han: None,
        scores: None,
        delta: None,
    };

    assert!(should_sample_replay_event(&dahai));
    assert!(!should_sample_replay_event(&reach));
    assert!(!should_sample_replay_event(&hora));
}

#[test]
fn stage_a_belief_audit_summary_tracks_real_coverage() {
    let (log, _) = play_game_with_mjai_log(17);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    let mut audit = StageABeliefAuditSummary::default();
    for sample in &game.samples {
        let target = match (sample.belief_fields, sample.mixture_weights) {
            (Some(belief_fields), mixture_weights) => Some(StageABeliefTarget {
                belief_fields,
                mixture_weights,
                trust: 1.0,
                ess: 1.0,
                entropy: 0.0,
            }),
            _ => None,
        };
        audit.record(target.as_ref());
    }
    assert!(audit.total > 0);
    assert!(audit.belief_coverage() >= 0.0 && audit.belief_coverage() <= 1.0);
}

#[test]
fn load_game_from_gzip_path_extracts_samples() {
    let (log, final_scores) = play_game_with_mjai_log(1);
    let path = std::env::temp_dir().join(format!(
        "hydra_mjai_loader_{}_{}.json.gz",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let file = File::create(&path).expect("create gzip log");
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(log.join("\n").as_bytes())
        .expect("write gzip log");
    encoder.finish().expect("finish gzip log");

    let game = load_game_from_path(&path).expect("load gz game");
    std::fs::remove_file(&path).expect("cleanup temp log");

    assert_eq!(game.final_scores, final_scores);
    assert!(game.samples.len() > 50);
}

#[test]
fn load_game_from_reader_accepts_valid_kakan_replay_with_class_only_tiles() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","4p","4p","4p","5s","6s"],["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","E"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","5p","6p","7p","8p","9p","S"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"5s","tsumogiri":false}"#,
        r#"{"type":"pon","actor":0,"target":0,"pai":"4p","consumed":["4p","4p"]}"#,
        r#"{"type":"dahai","actor":0,"pai":"6s","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"E"}"#,
        r#"{"type":"dahai","actor":1,"pai":"E","tsumogiri":true}"#,
        r#"{"type":"tsumo","actor":2,"pai":"8m"}"#,
        r#"{"type":"dahai","actor":2,"pai":"8m","tsumogiri":true}"#,
        r#"{"type":"tsumo","actor":3,"pai":"S"}"#,
        r#"{"type":"dahai","actor":3,"pai":"S","tsumogiri":true}"#,
        r#"{"type":"tsumo","actor":0,"pai":"4p"}"#,
        r#"{"type":"kakan","actor":0,"pai":"4p"}"#,
        r#"{"type":"tsumo","actor":0,"pai":"7s"}"#,
        r#"{"type":"dahai","actor":0,"pai":"7s","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ];

    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");

    assert!(!game.samples.is_empty());
    assert!(
        game.samples
            .iter()
            .any(|sample| sample.action < HYDRA_ACTION_SPACE as u8)
    );
}

#[test]
fn load_game_from_reader_accepts_duplicate_plain_tiles_in_start_kyoku() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["6m","6m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p","7p","8p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"8p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ];

    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");

    assert!(!game.samples.is_empty());
}

#[test]
fn load_game_from_reader_resolves_skipped_pon_window_without_pass_sample() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["5m","5m","1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S"],["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"]]}"#,
        r#"{"type":"tsumo","actor":0,"pai":"5p"}"#,
        r#"{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ];

    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");

    assert!(
        !game
            .samples
            .iter()
            .any(|sample| sample.action == hydra_core::action::PASS),
        "current engine resolves a skipped pon response window at the following tsumo boundary without emitting a synthetic pass sample"
    );
}

#[test]
fn prepare_replay_decision_resolves_skipped_response_before_tsumo_boundary() {
    let events = read_mjai_events(Cursor::new(
        [
            r#"{"type":"start_game","names":["a","b","c","d"]}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["5m","5m","1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S"],["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"]]}"#,
            r#"{"type":"tsumo","actor":0,"pai":"5p"}"#,
            r#"{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        ]
        .join("\n"),
    ))
    .expect("parse events");
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();

    for event in events.iter().take(4) {
        update_safety(&mut safety, event).expect("update safety");
        state.apply_mjai_event(event.clone());
    }

    assert_eq!(state.phase, riichienv_core::action::Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1]);

    let decisions = prepare_replay_decisions(&events[4], &mut state, &safety, &mut encoder)
        .expect("prepare replay decisions");

    assert!(decisions.is_empty());
    assert_eq!(state.phase, riichienv_core::action::Phase::WaitAct);
    assert!(state.active_player_slice().is_empty());
}

#[test]
fn prepare_replay_decision_keeps_riichi_dahai_as_discard_action() {
    let events = read_mjai_events(Cursor::new(
        [
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p","5p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"reach","actor":0}"#,
            r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        ]
        .join("\n"),
    ))
    .expect("parse events");
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();

    for event in events.iter().take(2) {
        update_safety(&mut safety, event).expect("update safety");
        state.apply_mjai_event(event.clone());
    }

    let decision = prepare_replay_decision(&events[2], &mut state, &safety, &mut encoder)
        .expect("prepare replay decision should succeed")
        .expect("riichi discard should still emit a replay decision");

    assert_eq!(decision.actor, 0);
    assert_ne!(decision.action_id, hydra_core::action::RIICHI);
    assert!(decision.action_id <= hydra_core::action::DISCARD_END);
    assert!(decision.legal_mask[decision.action_id as usize]);
}

#[test]
fn prepare_replay_decision_resolves_wait_response_before_terminal_event() {
    let events = read_mjai_events(Cursor::new(
        [
            r#"{"type":"start_game","names":["a","b","c","d"]}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["5m","5m","1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S"],["1p","2p","3p","4p","5p","6p","7p","8p","9p","1s","2s","3s","4s"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"]]}"#,
            r#"{"type":"tsumo","actor":0,"pai":"5p"}"#,
            r#"{"type":"dahai","actor":0,"pai":"5m","tsumogiri":false}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n"),
    ))
    .expect("parse events");
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();

    for event in events.iter().take(4) {
        update_safety(&mut safety, event).expect("update safety");
        state.apply_mjai_event(event.clone());
    }

    assert_eq!(state.phase, riichienv_core::action::Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1]);

    let decisions = prepare_replay_decisions(&events[4], &mut state, &safety, &mut encoder)
        .expect("prepare replay decisions should resolve terminal boundary");

    assert!(decisions.is_empty());
    assert_eq!(state.phase, riichienv_core::action::Phase::WaitAct);
    assert!(state.active_player_slice().is_empty());
}

#[test]
fn prepare_replay_decision_allows_implicit_pass_alongside_hora_response() {
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    state.phase = riichienv_core::action::Phase::WaitResponse;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[0] = 1;
    state.current_claims[0][0] = EngineAction::new(ActionType::Ron, None, &[], Some(0));
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = EngineAction::new(ActionType::Ron, None, &[], Some(1));
    state.last_discard = Some((3, 48));

    let safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let decisions = prepare_replay_decisions(
        &MjaiEvent::Hora {
            actor: 1,
            target: 3,
            pai: None,
            uradora_markers: None,
            yaku: None,
            fu: None,
            han: None,
            scores: None,
            delta: Some(vec![0, 2000, 0, -2000]),
        },
        &mut state,
        &safety,
        &mut encoder,
    )
    .expect("prepare replay decisions");

    assert!(
        decisions.iter().any(|decision| {
            decision.actor == 0 && decision.action_id == hydra_core::action::PASS
        })
    );
    assert!(state.players[0].missed_agari_doujun);
    assert_eq!(state.phase, riichienv_core::action::Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[0, 1]);
}
