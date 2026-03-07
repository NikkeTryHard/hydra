//! MJAI `.json` / `.json.gz` loader for behavioral cloning data.

use crate::data::sample::{MjaiSample, score_to_placement, scores_to_grp_index};
use crate::teacher::belief::{StageABeliefConfig, build_stage_a_teacher};
use crate::training::losses::oracle_target_from_scores;
use hydra_core::action::{
    AKA_5M, AKA_5P, AKA_5S, ActionPhase, DISCARD_END, HYDRA_ACTION_SPACE, build_legal_mask,
    riichienv_to_hydra,
};
use hydra_core::bridge::{encode_observation, extract_public_remaining_counts};
use hydra_core::encoder::ObservationEncoder;
use hydra_core::safety::SafetyInfo;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::replay::{
    MjaiEvent, load_mjai_events_from_path, mjai_event_actor, mjai_event_to_action, read_mjai_events,
};
use riichienv_core::rule::GameRule;
use riichienv_core::shanten::calc_shanten_from_counts;
use riichienv_core::state::GameState;
use std::array;
use std::io::{self, BufRead};
use std::path::Path;

const MISSING_TILE_TARGET: u8 = 255;

pub struct MjaiGame {
    pub samples: Vec<MjaiSample>,
    pub final_scores: [i32; 4],
}

impl MjaiGame {
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

pub struct MjaiDataset {
    pub games: Vec<MjaiGame>,
    pub train_fraction: f32,
}

#[inline]
fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[inline]
fn tile136_to_type(tile136: u8) -> u8 {
    tile136 / 4
}

fn mjai_tile(tile: &str) -> io::Result<u8> {
    mjai_to_tid(tile).ok_or_else(|| invalid_data(format!("invalid mjai tile: {tile}")))
}

fn mjai_tile_type(tile: &str) -> io::Result<u8> {
    Ok(tile136_to_type(mjai_tile(tile)?))
}

fn rel_opp(observer: usize, actor: usize) -> Option<usize> {
    let idx = ((actor + 4 - observer) % 4).wrapping_sub(1);
    (idx < 3).then_some(idx)
}

fn abs_opp(observer: usize, rel: usize) -> usize {
    (observer + rel + 1) % 4
}

fn update_safety(safety: &mut [SafetyInfo; 4], event: &MjaiEvent) -> io::Result<()> {
    match event {
        MjaiEvent::StartKyoku { dora_marker, .. } => {
            *safety = array::from_fn(|_| SafetyInfo::default());
            let dora = mjai_tile_type(dora_marker)?;
            for info in safety.iter_mut() {
                info.on_dora_revealed(dora);
            }
        }
        MjaiEvent::Dora { dora_marker } => {
            let dora = mjai_tile_type(dora_marker)?;
            for info in safety.iter_mut() {
                info.on_dora_revealed(dora);
            }
        }
        MjaiEvent::Reach { actor } => {
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor
                    && let Some(opp) = rel_opp(observer, *actor)
                {
                    info.on_riichi(opp);
                }
            }
        }
        MjaiEvent::Dahai {
            actor,
            pai,
            tsumogiri,
        } => {
            let tile = mjai_tile_type(pai)?;
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor
                    && let Some(opp) = rel_opp(observer, *actor)
                {
                    info.on_discard(tile, opp, !*tsumogiri);
                }
            }
        }
        MjaiEvent::Pon {
            actor, consumed, ..
        }
        | MjaiEvent::Chi {
            actor, consumed, ..
        }
        | MjaiEvent::Kan {
            actor, consumed, ..
        }
        | MjaiEvent::Ankan { actor, consumed } => {
            let tiles = consumed
                .iter()
                .map(|tile| mjai_tile_type(tile))
                .collect::<io::Result<Vec<_>>>()?;
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor && rel_opp(observer, *actor).is_some() {
                    info.on_call(&tiles);
                }
            }
        }
        MjaiEvent::Kakan { actor, pai } => {
            let tiles = [mjai_tile_type(pai)?];
            for (observer, info) in safety.iter_mut().enumerate() {
                if observer != *actor && rel_opp(observer, *actor).is_some() {
                    info.on_call(&tiles);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn next_discards_after(events: &[MjaiEvent]) -> io::Result<Vec<[Option<u8>; 4]>> {
    let mut out = vec![[None; 4]; events.len()];
    let mut next = [None; 4];
    for (idx, event) in events.iter().enumerate().rev() {
        out[idx] = next;
        if let MjaiEvent::Dahai { actor, pai, .. } = event {
            next[*actor] = Some(mjai_tile_type(pai)?);
        }
    }
    Ok(out)
}

fn final_scores(events: &[MjaiEvent]) -> [i32; 4] {
    let mut scores = [25_000; 4];
    for event in events {
        match event {
            MjaiEvent::StartKyoku { scores: round, .. } => {
                for (dst, src) in scores.iter_mut().zip(round.iter().copied()) {
                    *dst = src;
                }
            }
            MjaiEvent::ReachAccepted { actor } => {
                scores[*actor] -= 1_000;
            }
            MjaiEvent::Hora {
                scores: Some(after),
                ..
            }
            | MjaiEvent::Ryukyoku {
                scores: Some(after),
                ..
            } => {
                for (dst, src) in scores.iter_mut().zip(after.iter().copied()) {
                    *dst = src;
                }
            }
            MjaiEvent::Hora {
                delta: Some(delta), ..
            }
            | MjaiEvent::Ryukyoku {
                delta: Some(delta), ..
            } => {
                for (dst, src) in scores.iter_mut().zip(delta.iter().copied()) {
                    *dst += src;
                }
            }
            _ => {}
        }
    }
    scores
}

fn exact_waits(state: &GameState, player: usize) -> ([f32; 34], bool) {
    let mut counts = [0u8; 34];
    for &tile in state.players[player].hand_slice() {
        counts[tile136_to_type(tile) as usize] += 1;
    }
    let hand_total: u8 = counts.iter().sum();
    let tenpai = calc_shanten_from_counts(&counts, hand_total / 3) == 0;
    if !tenpai {
        return ([0.0; 34], false);
    }

    let mut waits = [0.0; 34];
    for tile in 0..34usize {
        if counts[tile] >= 4 {
            continue;
        }
        if state.players[player]
            .discards_slice()
            .iter()
            .any(|&discard| tile136_to_type(discard) as usize == tile)
        {
            continue;
        }
        counts[tile] += 1;
        let complete = calc_shanten_from_counts(&counts, (hand_total + 1) / 3) == -1;
        counts[tile] -= 1;
        if complete {
            waits[tile] = 1.0;
        }
    }
    (waits, true)
}

fn bool_mask_to_f32(mask: [bool; HYDRA_ACTION_SPACE]) -> [f32; HYDRA_ACTION_SPACE] {
    mask.map(|is_legal| if is_legal { 1.0 } else { 0.0 })
}

fn public_safety_score(safety: &SafetyInfo, tile: u8) -> f32 {
    let t = tile as usize;
    let mut score = 0.0f32;
    for opp in 0..3usize {
        if hydra_core::safety::bit_test(safety.genbutsu_all[opp], t) {
            score += 1.0;
        }
        score += 0.35 * safety.suji[opp][t];
        if hydra_core::safety::bit_test(safety.half_suji[opp], t) {
            score += 0.1;
        }
        score -= 0.25 * safety.matagi[opp][t];
        if safety.opponent_riichi[opp] || safety.cached_tenpai_prob[opp] > 0.5 {
            score -= 0.1;
        }
    }
    if hydra_core::safety::bit_test(safety.kabe, t) {
        score += 0.4;
    }
    if hydra_core::safety::bit_test(safety.one_chance, t) {
        score += 0.2;
    }
    score.clamp(0.0, 1.0)
}

fn exact_dealin_risk_from_waits(wait_sets: &[[f32; 34]; 3], tile: u8) -> f32 {
    let t = tile as usize;
    if wait_sets.iter().any(|waits| waits[t] > 0.0) {
        1.0
    } else {
        0.0
    }
}

fn build_safety_residual_targets(
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    safety: &SafetyInfo,
    wait_sets: &[[f32; 34]; 3],
) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    for action in 0..=DISCARD_END {
        let action_idx = action as usize;
        if legal_mask[action_idx] <= 0.0 {
            continue;
        }
        let tile = match action {
            AKA_5M => 4,
            AKA_5P => 13,
            AKA_5S => 22,
            _ => action,
        };
        let public_score = public_safety_score(safety, tile);
        let exact_risk = exact_dealin_risk_from_waits(wait_sets, tile);
        target[action_idx] = (public_score - exact_risk).clamp(0.0, 1.0);
        mask[action_idx] = 1.0;
    }
    (target, mask)
}

fn build_stage_a_belief_targets(
    state: &GameState,
    actor: usize,
    obs: &riichienv_core::observation::Observation,
) -> (Option<[f32; 16 * 34]>, Option<[f32; 4]>, bool, bool) {
    let hand = hydra_core::bridge::extract_hand(obs);
    let discards = hydra_core::bridge::extract_discards(obs);
    let melds = hydra_core::bridge::extract_melds(obs);
    let dora = hydra_core::bridge::extract_dora(obs);
    let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
    let hidden_tiles = state
        .players
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != actor)
        .map(|(_, p)| p.hand_len as usize)
        .sum::<usize>()
        + state.wall.remaining();
    let target = build_stage_a_teacher(&remaining, hidden_tiles, StageABeliefConfig::default());
    match target {
        Some(target) => (
            Some(target.belief_fields),
            target.mixture_weights,
            true,
            target.mixture_weights.is_some(),
        ),
        None => (None, None, false, false),
    }
}

fn load_game_from_events(events: Vec<MjaiEvent>) -> io::Result<MjaiGame> {
    let final_scores = final_scores(&events);
    let oracle_target = oracle_target_from_scores(final_scores);
    let next_discards = next_discards_after(&events)?;
    let grp_label = scores_to_grp_index(final_scores).map_err(invalid_data)?;
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut samples = Vec::new();

    for (idx, event) in events.iter().enumerate() {
        let env_action = mjai_event_to_action(event)
            .map_err(|err| invalid_data(format!("replay action conversion failed: {err}")))?;
        if let (Some(actor), Some(env_action)) = (mjai_event_actor(event), env_action) {
            let obs = state
                .get_observation_for_replay(actor as u8, &env_action, &env_action.to_mjai())
                .map_err(|err| invalid_data(format!("replay observation failed: {err}")))?;
            let hydra_action = riichienv_to_hydra(&env_action)
                .map_err(|err| invalid_data(format!("hydra action mapping failed: {err}")))?;
            let legal = obs.legal_actions_method();
            let phase = if matches!(event, MjaiEvent::Dahai { .. })
                && state.players[actor].riichi_declared
            {
                ActionPhase::RiichiSelect
            } else {
                ActionPhase::Normal
            };
            let legal_mask = bool_mask_to_f32(build_legal_mask(&legal, phase));
            if legal_mask[hydra_action.id() as usize] > 0.0 {
                let obs_encoded = encode_observation(
                    &mut encoder,
                    &obs,
                    &safety[actor],
                    state.drawn_tile.map(tile136_to_type),
                );
                let mut tenpai = [0.0; 3];
                let mut opp_next = [MISSING_TILE_TARGET; 3];
                let mut danger = [0.0; 102];
                let mut danger_mask = [0.0; 102];
                let mut wait_sets = [[0.0f32; 34]; 3];
                for rel in 0..3usize {
                    let opp = abs_opp(actor, rel);
                    let (waits, is_tenpai) = exact_waits(&state, opp);
                    wait_sets[rel] = waits;
                    tenpai[rel] = if is_tenpai { 1.0 } else { 0.0 };
                    opp_next[rel] = next_discards[idx][opp].unwrap_or(MISSING_TILE_TARGET);
                    let start = rel * 34;
                    danger[start..start + 34].copy_from_slice(&wait_sets[rel]);
                    if is_tenpai {
                        danger_mask[start..start + 34].fill(1.0);
                    }
                }
                let (safety_residual, safety_residual_mask) = build_safety_residual_targets(
                    &legal_mask,
                    &safety[actor],
                    &wait_sets,
                );
                let (belief_fields, mixture_weights, belief_fields_present, mixture_weights_present) =
                    build_stage_a_belief_targets(&state, actor, &obs);
                samples.push(MjaiSample {
                    obs: obs_encoded,
                    action: hydra_action.id(),
                    legal_mask,
                    placement: score_to_placement(final_scores, actor as u8),
                    score_delta: final_scores[actor] - state.players[actor].score,
                    grp_label,
                    oracle_target: Some(oracle_target),
                    tenpai,
                    opp_next,
                    danger,
                    danger_mask,
                    safety_residual: Some(safety_residual),
                    safety_residual_mask: Some(safety_residual_mask),
                    belief_fields,
                    mixture_weights,
                    belief_fields_present,
                    mixture_weights_present,
                });
            }
        }

        update_safety(&mut safety, event)?;
        state.apply_mjai_event(event.clone());
    }

    Ok(MjaiGame {
        samples,
        final_scores,
    })
}

pub fn load_game_from_reader<R: BufRead>(reader: R) -> io::Result<MjaiGame> {
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    load_game_from_events(events)
}

pub fn load_game_from_path(path: impl AsRef<Path>) -> io::Result<MjaiGame> {
    let events = load_mjai_events_from_path(path)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))?;
    load_game_from_events(events)
}

pub fn load_dataset_from_paths<P: AsRef<Path>>(
    paths: &[P],
    train_fraction: f32,
) -> io::Result<MjaiDataset> {
    let mut dataset = MjaiDataset::new(train_fraction);
    for path in paths {
        dataset.add_game(load_game_from_path(path)?);
    }
    Ok(dataset)
}

impl MjaiDataset {
    pub fn new(train_fraction: f32) -> Self {
        Self {
            games: Vec::new(),
            train_fraction: normalized_train_fraction(train_fraction),
        }
    }

    pub fn add_game(&mut self, game: MjaiGame) {
        self.games.push(game);
    }

    pub fn num_samples(&self) -> usize {
        self.games.iter().map(MjaiGame::num_samples).sum()
    }

    pub fn num_games(&self) -> usize {
        self.games.len()
    }

    pub fn summary(&self) -> String {
        format!(
            "dataset(games={}, samples={})",
            self.num_games(),
            self.num_samples()
        )
    }

    pub fn train_split(&self) -> (&[MjaiGame], &[MjaiGame]) {
        let fraction = normalized_train_fraction(self.train_fraction);
        let n = (self.games.len() as f32 * fraction) as usize;
        (&self.games[..n], &self.games[n..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher::belief::StageABeliefAuditSummary;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use riichienv_core::action::Phase;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{Cursor, Write};
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
        let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
        let expected = oracle_target_from_scores(final_scores);
        assert!(!game.samples.is_empty(), "expected replay to produce samples");
        for sample in game.samples.iter().take(8) {
            let got_target = sample.oracle_target.expect("oracle target should be present");
            for (got, want) in got_target.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-6, "oracle target mismatch: {got} vs {want}");
            }
        }
    }

    #[test]
    fn load_game_from_reader_populates_safety_residual_for_discards_only() {
        let (log, _) = play_game_with_mjai_log(11);
        let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
        let sample = game.samples.iter().find(|s| s.action <= DISCARD_END).expect("discard sample");
        let target = sample.safety_residual.expect("safety residual target");
        let mask = sample.safety_residual_mask.expect("safety residual mask");
        assert_eq!(target.len(), HYDRA_ACTION_SPACE);
        assert_eq!(mask.len(), HYDRA_ACTION_SPACE);
        let masked_discards: f32 = mask[..=DISCARD_END as usize].iter().sum();
        assert!(masked_discards > 0.0, "expected at least one discard action to be labeled");
        let masked_non_discards: f32 = mask[(DISCARD_END as usize + 1)..].iter().sum();
        assert!(masked_non_discards.abs() < 1e-6, "non-discard actions should be masked out");
    }

    #[test]
    fn load_game_from_reader_can_emit_stage_a_belief_targets() {
        let (log, _) = play_game_with_mjai_log(13);
        let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
        let sample = game
            .samples
            .iter()
            .find(|s| s.belief_fields.is_some())
            .expect("expected at least one belief-target sample");
        let belief = sample.belief_fields.expect("belief fields");
        assert_eq!(belief.len(), 16 * 34);
        assert!(sample.belief_fields_present);
    }

    #[test]
    fn stage_a_belief_audit_summary_tracks_real_coverage() {
        let (log, _) = play_game_with_mjai_log(17);
        let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
        let mut audit = StageABeliefAuditSummary::default();
        for sample in &game.samples {
            let target = match (sample.belief_fields, sample.mixture_weights) {
                (Some(belief_fields), mixture_weights) => Some(crate::teacher::belief::StageABeliefTarget {
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
}
