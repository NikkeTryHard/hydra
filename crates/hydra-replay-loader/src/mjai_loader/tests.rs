use super::*;
use crate::replay_targets::{StageABeliefAuditSummary, StageABeliefTarget};
use flate2::Compression;
use flate2::write::GzEncoder;
use hydra_data_core::{
    COMPACT_BASELINE_CHANNELS, COMPACT_MISSING_SHANTEN, COMPACT_MISSING_TILE, CompactMeldType,
    CompactObservationFacts,
};
use hydra_replay_sidecar::{
    DeltaQSidecarIndex, ExitSidecarIndex, ReplayDecisionKey, ReplayDeltaQRecordV1,
    ReplayExitRecordV1, legal_mask_digest_from_f32,
};
use riichienv_core::action::{ActionType, Phase};
use riichienv_core::replay::read_mjai_events;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Cursor, ErrorKind, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use zstd::stream::write::Encoder as ZstdEncoder;

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
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.compact_facts.is_some())
    );
}

struct CollectRecordSink {
    samples: Vec<MjaiSample>,
}

impl ReplaySampleSink for CollectRecordSink {
    fn push_sample(&mut self, sample: ReplaySampleRecord) -> io::Result<()> {
        self.samples.push(sample.into_sample());
        Ok(())
    }
}

struct CollectReplayRecordSink {
    records: Vec<ReplaySampleRecord>,
}

impl ReplaySampleSink for CollectReplayRecordSink {
    fn push_sample(&mut self, sample: ReplaySampleRecord) -> io::Result<()> {
        self.records.push(sample);
        Ok(())
    }
}

#[test]
fn load_game_from_reader_into_sink_matches_public_loader_samples() {
    let (log, final_scores) = play_game_with_mjai_log(3);
    let payload = log.join("\n");
    let public = load_game_from_reader(Cursor::new(payload.as_bytes())).expect("load game");
    let mut sink = CollectRecordSink {
        samples: Vec::new(),
    };

    let sink_scores =
        load_game_from_reader_into_sink("game-3", Cursor::new(payload.as_bytes()), None, &mut sink)
            .expect("sink loader should succeed");

    assert_eq!(public.final_scores, final_scores);
    assert_eq!(sink_scores, public.final_scores);
    assert_eq!(sink.samples.len(), public.samples.len());
    assert!(sink.samples.len() > 50, "expected real replay decisions");
    for (from_sink, from_public) in sink.samples.iter().zip(public.samples.iter()) {
        assert_replay_sample_eq(from_sink, from_public);
    }
}

fn assert_replay_sample_eq(from_sink: &MjaiSample, from_public: &MjaiSample) {
    assert_eq!(from_sink.obs, from_public.obs);
    assert_eq!(from_sink.action, from_public.action);
    assert_eq!(from_sink.legal_mask, from_public.legal_mask);
    assert_eq!(from_sink.placement, from_public.placement);
    assert_eq!(from_sink.score_delta, from_public.score_delta);
    assert_eq!(from_sink.grp_label, from_public.grp_label);
    assert_eq!(from_sink.oracle_target, from_public.oracle_target);
    assert_eq!(from_sink.tenpai, from_public.tenpai);
    assert_eq!(from_sink.opp_next, from_public.opp_next);
    assert_eq!(from_sink.danger, from_public.danger);
    assert_eq!(from_sink.danger_mask, from_public.danger_mask);
    assert_eq!(from_sink.safety_residual, from_public.safety_residual);
    assert_eq!(
        from_sink.safety_residual_mask,
        from_public.safety_residual_mask
    );
    assert_eq!(from_sink.exit_target, from_public.exit_target);
    assert_eq!(from_sink.exit_mask, from_public.exit_mask);
    assert_eq!(from_sink.delta_q_target, from_public.delta_q_target);
    assert_eq!(from_sink.delta_q_mask, from_public.delta_q_mask);
    assert_eq!(from_sink.belief_fields, from_public.belief_fields);
    assert_eq!(from_sink.mixture_weights, from_public.mixture_weights);
    assert_eq!(
        from_sink.belief_fields_present,
        from_public.belief_fields_present
    );
    assert_eq!(
        from_sink.mixture_weights_present,
        from_public.mixture_weights_present
    );
    assert_eq!(from_sink.compact_facts, from_public.compact_facts);
}

fn representative_replay_payload() -> &'static str {
    include_str!("../../../../RiichiEnv/tests/data/126_204_0_mjai.jsonl")
}

fn assert_representative_replay_events(payload: &str) {
    let events = read_mjai_events(Cursor::new(payload)).expect("parse representative replay");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MjaiEvent::Hora { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MjaiEvent::Pon { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MjaiEvent::Chi { .. }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        MjaiEvent::Kan { .. } | MjaiEvent::Ankan { .. } | MjaiEvent::Kakan { .. }
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MjaiEvent::Reach { .. }))
    );
}

fn load_representative_records(
    payload: &str,
    observation_profile: ReplayObservationProfile,
) -> io::Result<([i32; 4], Vec<ReplaySampleRecord>)> {
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(true, true, true, true, false, false),
        observation_profile,
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        None,
        None,
    );
    let mut sink = CollectReplayRecordSink {
        records: Vec::new(),
    };
    let scores = load_game_from_reader_into_sink(
        "126_204_0_mjai.jsonl",
        Cursor::new(payload.as_bytes()),
        Some(&policy),
        &mut sink,
    )?;
    Ok((scores, sink.records))
}

fn assert_record_targets_eq(row: usize, bc: &ReplaySampleRecord, full: &ReplaySampleRecord) {
    assert_eq!(bc.placement, full.placement, "row {row} placement");
    assert_eq!(bc.score_delta, full.score_delta, "row {row} score_delta");
    assert_eq!(bc.grp_label, full.grp_label, "row {row} grp_label");
    assert_eq!(bc.oracle_target, full.oracle_target, "row {row} oracle");
    assert_eq!(bc.tenpai, full.tenpai, "row {row} tenpai");
    assert_eq!(bc.opp_next, full.opp_next, "row {row} opp_next");
    assert_eq!(bc.danger, full.danger, "row {row} danger");
    assert_eq!(bc.danger_mask, full.danger_mask, "row {row} danger_mask");
    assert_eq!(bc.safety_residual, full.safety_residual, "row {row} safety");
    assert_eq!(
        bc.safety_residual_mask, full.safety_residual_mask,
        "row {row} safety mask"
    );
    assert_eq!(bc.exit_target, None, "row {row} exit target");
    assert_eq!(full.exit_target, None, "row {row} full exit target");
    assert_eq!(bc.exit_mask, None, "row {row} exit mask");
    assert_eq!(full.exit_mask, None, "row {row} full exit mask");
    assert_eq!(bc.delta_q_target, None, "row {row} delta-q target");
    assert_eq!(full.delta_q_target, None, "row {row} full delta-q target");
    assert_eq!(bc.delta_q_mask, None, "row {row} delta-q mask");
    assert_eq!(full.delta_q_mask, None, "row {row} full delta-q mask");
    assert_eq!(bc.belief_fields, full.belief_fields, "row {row} belief");
    assert_eq!(
        bc.mixture_weights, full.mixture_weights,
        "row {row} mixture"
    );
    assert_eq!(
        bc.belief_fields_present, full.belief_fields_present,
        "row {row} belief present"
    );
    assert_eq!(
        bc.mixture_weights_present, full.mixture_weights_present,
        "row {row} mixture present"
    );
}

fn assert_compact_facts_eq(
    row: usize,
    bc: &CompactObservationFacts,
    full: &CompactObservationFacts,
) {
    assert_eq!(
        bc.hand_counts, full.hand_counts,
        "row {row} compact hand_counts"
    );
    assert_eq!(
        bc.open_meld_counts, full.open_meld_counts,
        "row {row} compact open_meld_counts"
    );
    assert_eq!(
        bc.drawn_tile, full.drawn_tile,
        "row {row} compact drawn_tile"
    );
    assert_eq!(
        bc.shanten_base, full.shanten_base,
        "row {row} compact shanten_base"
    );
    assert_eq!(
        bc.shanten_discard, full.shanten_discard,
        "row {row} compact shanten_discard"
    );
    assert_eq!(bc.discards, full.discards, "row {row} compact discards");
    assert_eq!(bc.melds, full.melds, "row {row} compact melds");
    assert_eq!(
        bc.dora_indicators, full.dora_indicators,
        "row {row} compact dora"
    );
    assert_eq!(
        bc.dora_indicator_count, full.dora_indicator_count,
        "row {row} compact dora count"
    );
    assert_eq!(bc.aka_flags, full.aka_flags, "row {row} compact aka");
    assert_eq!(bc.riichi, full.riichi, "row {row} compact riichi");
    assert_eq!(bc.scores, full.scores, "row {row} compact scores");
    assert_eq!(bc.kyoku_index, full.kyoku_index, "row {row} compact kyoku");
    assert_eq!(bc.honba, full.honba, "row {row} compact honba");
    assert_eq!(bc.kyotaku, full.kyotaku, "row {row} compact kyotaku");
    assert_eq!(bc.safety, full.safety, "row {row} compact safety");
    assert_eq!(
        bc.advanced_tail, full.advanced_tail,
        "row {row} compact advanced tail"
    );
}

#[test]
fn representative_replay_duplicate_fact_extraction_matches_single_extract_rows() {
    let payload = representative_replay_payload();
    assert_representative_replay_events(payload);

    let events = read_mjai_events(Cursor::new(payload)).expect("parse representative replay");
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut old_encoder = ObservationEncoder::new();
    let mut checked_rows = 0usize;
    let mut saw_pon = false;
    let mut saw_chi = false;
    let mut saw_kan = false;
    let mut saw_pass = false;

    for (event_idx, event) in events.iter().enumerate() {
        let decisions = prepare_replay_decisions_with_options(
            event,
            &mut state,
            &safety,
            &mut encoder,
            ReplayDecisionOptions {
                observation_profile: ReplayObservationProfile::BcMinimal,
            },
        )
        .expect("prepare representative replay decision");

        for (decision_idx, decision) in decisions.iter().enumerate() {
            let row = checked_rows;
            assert!(
                decision.use_ref_targets,
                "event {event_idx} decision {decision_idx} ref path"
            );
            assert!(
                decision.legal_mask[decision.action_id as usize],
                "event {event_idx} decision {decision_idx} chosen legal"
            );
            let obs_ref = state.observe(decision.actor as u8);
            let old_obs_encoded = hydra_core::bridge::encode_observation_ref_with_profile(
                &mut old_encoder,
                &obs_ref,
                &safety[decision.actor],
                BridgeEncodeProfile::bc_minimal(),
            );
            let old_extracted_facts = extract_observation_facts_ref(&obs_ref);
            let old_compact_facts = CompactObservationFacts::from_encoder_inputs(
                old_extracted_facts.hand,
                old_extracted_facts.open_meld_counts,
                old_extracted_facts.drawn_tile,
                old_extracted_facts.shanten_batch.base,
                old_extracted_facts.shanten_batch.discard,
                &old_extracted_facts.discards,
                &old_extracted_facts.melds,
                &old_extracted_facts.dora,
                &old_extracted_facts.meta,
                &safety[decision.actor],
                &old_obs_encoded,
                false,
            );

            assert_eq!(
                decision.obs_encoded, old_obs_encoded,
                "event {event_idx} decision {decision_idx} obs"
            );
            assert_compact_facts_eq(row, &decision.compact_facts, &old_compact_facts);

            saw_pon |= decision.action_id == hydra_core::action::PON;
            saw_chi |= matches!(
                decision.action_id,
                hydra_core::action::CHI_LEFT
                    | hydra_core::action::CHI_MID
                    | hydra_core::action::CHI_RIGHT
            );
            saw_kan |= decision.action_id == hydra_core::action::KAN;
            saw_pass |= decision.action_id == hydra_core::action::PASS;
            checked_rows += 1;
        }

        update_safety(&mut safety, event).expect("update safety");
        state
            .try_apply_mjai_event(event.clone())
            .expect("apply representative replay event");
    }

    assert!(
        checked_rows > 600,
        "expected representative replay decisions"
    );
    assert!(saw_pon, "expected representative pon decision");
    assert!(saw_chi, "expected representative chi decision");
    assert!(saw_kan, "expected representative kan decision");
    assert!(saw_pass, "expected representative implicit pass decision");
}

#[test]
fn representative_replay_full_and_bc_minimal_targets_stay_in_order() {
    let payload = representative_replay_payload();
    let (bc_scores, bc_rows) =
        load_representative_records(payload, ReplayObservationProfile::BcMinimal)
            .expect("load representative bc replay");
    let (full_scores, full_rows) =
        load_representative_records(payload, ReplayObservationProfile::Full)
            .expect("load representative full replay");

    assert_eq!(bc_scores, full_scores);
    assert_eq!(bc_rows.len(), full_rows.len());
    assert!(bc_rows.len() > 600, "expected representative replay rows");
    for (row, (bc, full)) in bc_rows.iter().zip(full_rows.iter()).enumerate() {
        assert_eq!(bc.action, full.action, "row {row} action");
        assert_eq!(bc.legal_mask, full.legal_mask, "row {row} legal mask");
        assert!(
            bc.legal_mask[bc.action as usize] > 0.0,
            "row {row} chosen legal"
        );
        assert_record_targets_eq(row, bc, full);
    }
}

#[test]
fn load_game_from_reader_into_sink_matches_public_loader_with_optional_targets() {
    let source_identity = "game-1";
    let log = replay_sidecar_guardrail_log();
    let exit_records = synthetic_exit_records(source_identity, 123, 1);
    let delta_q_records = synthetic_delta_q_records(source_identity, 123, 1);
    let exit_index = ExitSidecarIndex::from_records(exit_records);
    let delta_q_index = DeltaQSidecarIndex::from_records(delta_q_records);
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(true, true, true, true, true, true),
        ReplayObservationProfile::BcMinimal,
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(123), Some(1)),
        Some(&exit_index),
        Some(&delta_q_index),
    );
    let public = load_game_from_stream_with_policy(
        source_identity,
        Cursor::new(log.as_bytes()),
        Some(&policy),
    )
    .expect("public policy loader should succeed");
    let mut sink = CollectRecordSink {
        samples: Vec::new(),
    };

    let sink_scores = load_game_from_reader_into_sink(
        source_identity,
        Cursor::new(log.as_bytes()),
        Some(&policy),
        &mut sink,
    )
    .expect("sink policy loader should succeed");

    assert_eq!(sink_scores, public.final_scores);
    assert_eq!(sink.samples.len(), public.samples.len());
    assert!(
        sink.samples
            .iter()
            .any(|sample| sample.oracle_target.is_some()),
        "oracle targets should be present under optional-head profile"
    );
    assert!(
        sink.samples
            .iter()
            .any(|sample| sample.exit_target.is_some()),
        "joined exit sidecar rows should hydrate"
    );
    assert!(
        sink.samples
            .iter()
            .any(|sample| sample.delta_q_target.is_some()),
        "joined delta-Q sidecar rows should hydrate"
    );
    for (from_sink, from_public) in sink.samples.iter().zip(public.samples.iter()) {
        assert_replay_sample_eq(from_sink, from_public);
    }
}

#[test]
fn real_replay_compact_facts_decode_matches_dense_baseline() {
    let (log, _) = play_game_with_mjai_log(11);
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    let sample = game
        .samples
        .iter()
        .find(|sample| sample.compact_facts.is_some())
        .expect("real replay should produce compact facts");
    let facts = sample
        .compact_facts
        .as_ref()
        .expect("compact facts should exist");
    let mut encoder = ObservationEncoder::new();
    let decoded = hydra_core::bridge::encode_extracted_observation_facts_with_profile(
        &mut encoder,
        &hydra_core::bridge::ExtractedObservationFacts {
            hand: facts.hand_counts,
            drawn_tile: (facts.drawn_tile != COMPACT_MISSING_TILE).then_some(facts.drawn_tile),
            open_meld_counts: facts.open_meld_counts,
            discards: std::array::from_fn(|player| {
                let src = &facts.discards[player];
                let mut dst = hydra_core::encoder::PlayerDiscards::new();
                for entry in src.discards.iter().take(src.len as usize) {
                    dst.push(hydra_core::encoder::DiscardEntry {
                        tile: entry.tile,
                        is_tedashi: entry.is_tedashi,
                        turn: entry.turn,
                    });
                }
                dst
            }),
            melds: std::array::from_fn(|player| {
                let src = &facts.melds[player];
                let mut dst = hydra_core::encoder::PlayerMelds::new();
                for meld in src.melds.iter().take(src.len as usize) {
                    dst.push(hydra_core::encoder::MeldInfo {
                        tiles: meld.tiles,
                        tile_count: meld.tile_count,
                        meld_type: match meld.meld_type {
                            CompactMeldType::Chi => hydra_core::encoder::MeldType::Chi,
                            CompactMeldType::Pon => hydra_core::encoder::MeldType::Pon,
                            CompactMeldType::Kan => hydra_core::encoder::MeldType::Kan,
                        },
                    });
                }
                dst
            }),
            dora: hydra_core::encoder::DoraInfo {
                indicators: facts.dora_indicators,
                indicator_count: facts.dora_indicator_count,
                aka_flags: facts.aka_flags,
            },
            meta: hydra_core::encoder::GameMetadata {
                riichi: facts.riichi,
                scores: facts.scores,
                shanten: facts.shanten_base,
                kyoku_index: facts.kyoku_index,
                honba: facts.honba,
                kyotaku: facts.kyotaku,
            },
            shanten_batch: hydra_core::shanten_batch::BatchShantenResult {
                base: facts.shanten_base,
                discard: std::array::from_fn(|tile| {
                    (facts.shanten_discard[tile] != COMPACT_MISSING_SHANTEN)
                        .then_some(facts.shanten_discard[tile])
                }),
            },
        },
        &SafetyInfo {
            genbutsu_all: facts.safety.genbutsu_all,
            genbutsu_tedashi: facts.safety.genbutsu_tedashi,
            genbutsu_riichi_era: facts.safety.genbutsu_riichi_era,
            suji: facts.safety.suji,
            half_suji: facts.safety.half_suji,
            matagi: facts.safety.matagi,
            kabe: facts.safety.kabe,
            one_chance: facts.safety.one_chance,
            visible_counts: facts.safety.visible_counts,
            opponent_riichi: facts.safety.opponent_riichi,
            cached_tenpai_prob: facts.safety.cached_tenpai_prob,
        },
        BridgeEncodeProfile::bc_minimal(),
    );

    assert_eq!(
        &decoded[..COMPACT_BASELINE_CHANNELS * 34],
        &sample.obs[..COMPACT_BASELINE_CHANNELS * 34]
    );
    assert_eq!(facts.advanced_tail, None);
}

#[test]
fn replay_materialization_stats_record_parse_update_encode_and_mask() {
    let _ = drain_replay_materialization_stats();
    let game = load_game_from_reader(Cursor::new(replay_sidecar_guardrail_log()))
        .expect("guardrail replay should load");
    assert!(!game.samples.is_empty());

    let stats = drain_replay_materialization_stats();
    assert!(stats.json_parse_ns > 0);
    assert!(stats.replay_update_ns > 0);
    assert!(stats.observation_encode_ns > 0);
    assert!(stats.mask_build_ns > 0);
    assert!(stats.event_count > 0);
    assert!(stats.decision_count > 0);
}

#[test]
fn load_game_from_reader_populates_oracle_targets_from_final_scores() {
    let (log, final_scores) = play_game_with_mjai_log(7);
    let game = load_game_from_reader_with_sidecar(
        "game-7",
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        ReplayTargetProfile::with_optional_heads(true, false, false, false, false, false),
        ReplayObservationProfile::BcMinimal,
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
        ReplayObservationProfile::BcMinimal,
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
        ReplayObservationProfile::BcMinimal,
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
fn present_sidecar_with_incomplete_provenance_errors() {
    let events =
        read_mjai_events(Cursor::new(replay_sidecar_guardrail_log())).expect("parse events");
    let exit_records = synthetic_exit_records("game-1", 123, 1);

    let err = match load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(123), None),
        SidecarProvenance::default(),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, false),
        ReplayObservationProfile::BcMinimal,
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        None,
    ) {
        Ok(_) => panic!("present sidecar without complete provenance must fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("complete source_net_hash"));
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
        ReplayObservationProfile::BcMinimal,
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

    let err = match load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(999), Some(99)),
        SidecarProvenance::new(Some(456), Some(2)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        ReplayObservationProfile::BcMinimal,
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    ) {
        Ok(_) => panic!("mismatched exit provenance should hard-error"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(
        err.to_string()
            .contains("replay ExIt sidecar source net hash mismatch")
    );
}

#[test]
fn mismatched_delta_q_provenance_does_not_block_exit_hydration() {
    let log = replay_sidecar_guardrail_log();
    let events = read_mjai_events(Cursor::new(log)).expect("parse events");
    let exit_records = synthetic_exit_records("game-1", 123, 1);
    let delta_q_records = synthetic_delta_q_records("game-1", 456, 2);

    let err = match load_game_from_events_with_sidecar(
        "game-1",
        SidecarProvenance::new(Some(123), Some(1)),
        SidecarProvenance::new(Some(999), Some(99)),
        ReplayTargetProfile::with_optional_heads(false, false, false, false, true, true),
        ReplayObservationProfile::BcMinimal,
        events,
        Some(&ExitSidecarIndex::from_records(exit_records)),
        Some(&DeltaQSidecarIndex::from_records(delta_q_records)),
    ) {
        Ok(_) => panic!("mismatched delta_q provenance should hard-error"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(
        err.to_string()
            .contains("replay delta_q sidecar source net hash mismatch")
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
        ReplayObservationProfile::BcMinimal,
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
        ReplayObservationProfile::BcMinimal,
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
fn load_game_from_stream_with_empty_policy_honors_non_sidecar_targets() {
    let policy = ReplayLoadPolicy::new(
        ReplayTargetProfile::with_optional_heads(true, false, false, false, true, true),
        ReplayObservationProfile::BcMinimal,
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
            .any(|sample| sample.oracle_target.is_some()),
        "non-sidecar optional heads should be materialized even when sidecars are absent"
    );
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
fn bc_minimal_belief_targets_use_real_observation_facts() {
    let (log, _) = play_game_with_mjai_log(37);
    let game = load_game_from_reader_with_sidecar(
        "game-37",
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        ReplayTargetProfile::with_optional_heads(false, false, true, false, false, false),
        ReplayObservationProfile::BcMinimal,
        Cursor::new(log.join("\n")),
        None,
        None,
    )
    .expect("load game with belief targets");

    let sample = game
        .samples
        .iter()
        .find(|sample| sample.belief_fields_present)
        .expect("BcMinimal path should produce belief targets");
    let compact_facts = sample
        .compact_facts
        .as_ref()
        .expect("compact facts should be attached");
    assert!(
        compact_facts.hand_counts.iter().any(|&count| count > 0),
        "regression guard: test must exercise non-empty replay facts"
    );
    let belief_fields = sample
        .belief_fields
        .as_ref()
        .expect("belief fields should be attached");
    assert!(
        belief_fields.iter().any(|&value| value > 0.0),
        "belief target should not be built from empty observation facts"
    );
    assert!(sample.mixture_weights.is_none());
    assert!(!sample.mixture_weights_present);
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
fn replay_materialization_stats_record_gzip_decompression() {
    let _ = drain_replay_materialization_stats();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(replay_sidecar_guardrail_log().as_bytes())
        .unwrap();
    let compressed = encoder.finish().unwrap();

    let game = load_game_from_stream(Cursor::new(compressed)).expect("gzip replay should load");
    assert!(!game.samples.is_empty());

    let stats = drain_replay_materialization_stats();
    assert!(stats.decompress_ns > 0);
    assert!(stats.json_parse_ns > 0);
}

#[test]
fn replay_materialization_stats_record_zstd_decompression_separately_from_json_parse() {
    let _ = drain_replay_materialization_stats();
    let mut encoder = ZstdEncoder::new(Vec::new(), 0).expect("create zstd encoder");
    encoder
        .write_all(replay_sidecar_guardrail_log().as_bytes())
        .expect("write zstd log");
    let compressed = encoder.finish().expect("finish zstd log");

    let game = load_game_from_stream(Cursor::new(compressed)).expect("zstd replay should load");
    assert!(!game.samples.is_empty());

    let stats = drain_replay_materialization_stats();
    assert!(stats.decompress_ns > 0);
    assert!(stats.json_parse_ns > 0);
    assert_ne!(
        stats.decompress_ns, stats.json_parse_ns,
        "zstd decompression timing must be recorded independently from JSON parse timing"
    );
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
fn load_game_from_reader_rejects_fifth_start_tile_copy() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["6m","6m","6m","6m","6m","9m","1p","2p","3p","4p","5p","6p","7p","8p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"ryukyoku"}"#,
    ];

    let err = match load_game_from_reader(Cursor::new(log.join("\n"))) {
        Ok(_) => panic!("fifth tile copy should reject"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("more than four copies"));
}

#[test]
fn load_game_from_reader_rejects_unsupported_event_type() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"totally_unknown"}"#,
    ];

    let err = match load_game_from_reader(Cursor::new(log.join("\n"))) {
        Ok(_) => panic!("unknown event type should reject"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("unsupported MJAI event type"));
}

#[test]
fn load_game_from_reader_rejects_hora_scores_delta_mismatch() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"hora","actor":1,"target":0,"pai":"4p","scores":[25000,26000,25000,24000],"deltas":[-1000,1000,0,0]}"#,
    ];

    let err = match load_game_from_reader(Cursor::new(log.join("\n"))) {
        Ok(_) => panic!("mismatched terminal score fields should reject"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("scores do not match delta"));
}

#[test]
fn load_game_from_reader_rejects_hora_han_fu_point_mismatch() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"hora","actor":1,"target":0,"pai":"4p","fu":30,"han":1,"deltas":[-2000,2000,0,0]}"#,
    ];

    let err = match load_game_from_reader(Cursor::new(log.join("\n"))) {
        Ok(_) => panic!("wrong han/fu payment should reject"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("points do not match"));
}

#[test]
fn load_game_from_reader_accepts_hora_han_fu_point_match() {
    let log = [
        r#"{"type":"start_game","names":["a","b","c","d"]}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"hora","actor":1,"target":0,"pai":"4p","fu":30,"han":1,"deltas":[-1000,1000,0,0]}"#,
    ];

    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("valid terminal payment");

    assert!(!game.samples.is_empty());
    assert_eq!(game.final_scores, [24_000, 26_000, 25_000, 25_000]);
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

fn implicit_pass_hora_event() -> MjaiEvent {
    MjaiEvent::Hora {
        actor: 1,
        target: 3,
        pai: None,
        uradora_markers: None,
        yaku: None,
        fu: None,
        han: None,
        scores: None,
        delta: Some(vec![0, 2000, 0, -2000]),
    }
}

fn implicit_pass_ron_state() -> GameState {
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    state.phase = riichienv_core::action::Phase::WaitResponse;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[0] = 1;
    state.current_claims[0][0] = EngineAction::new(ActionType::Ron, None, &[], Some(0));
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = EngineAction::new(ActionType::Ron, None, &[], Some(1));
    state.last_discard = Some((3, 48));
    state
}

#[test]
fn implicit_pass_bc_minimal_matches_full_action_and_legal_mask() {
    let safety = array::from_fn(|_| SafetyInfo::default());
    let mut bc_state = implicit_pass_ron_state();
    let mut full_state = implicit_pass_ron_state();
    let mut bc_encoder = ObservationEncoder::new();
    let mut full_encoder = ObservationEncoder::new();

    let bc_decisions = prepare_replay_decisions_with_options(
        &implicit_pass_hora_event(),
        &mut bc_state,
        &safety,
        &mut bc_encoder,
        ReplayDecisionOptions {
            observation_profile: ReplayObservationProfile::BcMinimal,
        },
    )
    .expect("prepare bc minimal implicit pass");
    let full_decisions = prepare_replay_decisions_with_options(
        &implicit_pass_hora_event(),
        &mut full_state,
        &safety,
        &mut full_encoder,
        ReplayDecisionOptions {
            observation_profile: ReplayObservationProfile::Full,
        },
    )
    .expect("prepare full implicit pass");

    assert_eq!(bc_decisions.len(), 1);
    assert_eq!(full_decisions.len(), 1);
    let bc = &bc_decisions[0];
    let full = &full_decisions[0];
    assert_eq!(bc.actor, full.actor);
    assert_eq!(bc.action_id, hydra_core::action::PASS);
    assert_eq!(bc.action_id, full.action_id);
    assert_eq!(bc.legal_mask, full.legal_mask);
    assert_eq!(bc.legal_mask_f32, full.legal_mask_f32);
    assert!(bc.legal_mask[hydra_core::action::PASS as usize]);
    assert_eq!(bc.legal_mask_f32[hydra_core::action::PASS as usize], 1.0);
    assert!(bc.compact_facts.hand_counts.iter().any(|&count| count != 0));
    assert!(bc.use_ref_targets);
    assert!(!full.use_ref_targets);
}

#[test]
fn prepare_replay_decision_marks_riichi_missed_agari_on_implicit_pass() {
    let mut state = implicit_pass_ron_state();
    state.players[0].riichi_declared = true;
    let safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();

    let decisions = prepare_replay_decisions(
        &implicit_pass_hora_event(),
        &mut state,
        &safety,
        &mut encoder,
    )
    .expect("prepare replay decisions");

    assert_eq!(decisions.len(), 1);
    let decision = &decisions[0];
    assert_eq!(decision.actor, 0);
    assert_eq!(decision.action_id, hydra_core::action::PASS);
    assert!(decision.legal_mask[hydra_core::action::PASS as usize]);
    assert_eq!(
        decision.legal_mask_f32[hydra_core::action::PASS as usize],
        1.0
    );
    assert!(
        decision
            .compact_facts
            .hand_counts
            .iter()
            .any(|&count| count != 0)
    );
    assert!(state.players[0].missed_agari_doujun);
    assert!(state.players[0].missed_agari_riichi);
    assert_eq!(state.phase, riichienv_core::action::Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[0, 1]);
}
