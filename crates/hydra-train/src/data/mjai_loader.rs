//! MJAI `.json` / `.json.gz` loader for behavioral cloning data.

use crate::data::replay_targets::{
    build_safety_residual_targets, build_stage_a_belief_targets, exact_waits,
};
use crate::data::sample::{MjaiSample, score_to_placements, scores_to_grp_index};
use crate::training::losses::oracle_target_from_scores;
use crate::training::replay_delta_q::DeltaQSidecarIndex;
use crate::training::replay_exit::{
    ExitSidecarIndex, ReplayDecisionKey, source_hash_from_identity,
};
use flate2::read::GzDecoder;
#[cfg(test)]
use hydra_core::action::{AKA_5M, DISCARD_END};
use hydra_core::action::{ActionPhase, HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::bridge::{
    BridgeEncodeProfile, encode_observation, encode_observation_with_profile,
};
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action as EngineAction, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::replay::{
    MjaiEvent, load_mjai_events_from_path, mjai_event_actor, mjai_event_to_action, read_mjai_events,
};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use std::array;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const MISSING_TILE_TARGET: u8 = 255;

#[derive(Clone, Copy, Default)]
struct ReplayDecisionOptions {
    use_bc_minimal_encode: bool,
}

#[derive(Default)]
struct ReplayProfileStats {
    parse_events_ns: u128,
    precompute_ns: u128,
    prepare_decisions_ns: u128,
    implicit_pass_ns: u128,
    replay_observation_ns: u128,
    legal_mask_build_ns: u128,
    encode_observation_ns: u128,
    legal_mask_convert_ns: u128,
    opponent_targets_ns: u128,
    exact_waits_ns: u128,
    safety_residual_ns: u128,
    belief_targets_ns: u128,
    sidecar_lookup_ns: u128,
    sample_push_ns: u128,
    update_safety_ns: u128,
    apply_event_ns: u128,
    event_count: usize,
    decision_count: usize,
}

static REPLAY_PROFILE_PRINTED: AtomicBool = AtomicBool::new(false);
static REPLAY_IMPLICIT_PASS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAY_OBSERVATION_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAY_LEGAL_MASK_BUILD_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static REPLAY_ENCODE_OBS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn maybe_print_replay_profile(stats: &ReplayProfileStats) {
    if REPLAY_PROFILE_PRINTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let total_ns = stats.parse_events_ns
        + stats.precompute_ns
        + stats.prepare_decisions_ns
        + stats.implicit_pass_ns
        + stats.replay_observation_ns
        + stats.legal_mask_build_ns
        + stats.encode_observation_ns
        + stats.legal_mask_convert_ns
        + stats.opponent_targets_ns
        + stats.safety_residual_ns
        + stats.belief_targets_ns
        + stats.sidecar_lookup_ns
        + stats.sample_push_ns
        + stats.update_safety_ns
        + stats.apply_event_ns;
    let pct = |part: u128| -> f64 {
        if total_ns == 0 {
            0.0
        } else {
            part as f64 * 100.0 / total_ns as f64
        }
    };
    eprintln!(
        "[replay-profile] total={:.3}s parse={:.1}% precompute={:.1}% prepare={:.1}% implicit_pass={:.1}% replay_obs={:.1}% legal_mask_build={:.1}% encode_obs={:.1}% opp_targets={:.1}% exact_waits={:.1}% legal_mask_f32={:.1}% safety={:.1}% belief={:.1}% sidecar={:.1}% sample_push={:.1}% update_safety={:.1}% apply_event={:.1}% events={} decisions={}",
        total_ns as f64 / 1_000_000_000.0,
        pct(stats.parse_events_ns),
        pct(stats.precompute_ns),
        pct(stats.prepare_decisions_ns),
        pct(stats.implicit_pass_ns),
        pct(stats.replay_observation_ns),
        pct(stats.legal_mask_build_ns),
        pct(stats.encode_observation_ns),
        pct(stats.opponent_targets_ns),
        pct(stats.exact_waits_ns),
        pct(stats.legal_mask_convert_ns),
        pct(stats.safety_residual_ns),
        pct(stats.belief_targets_ns),
        pct(stats.sidecar_lookup_ns),
        pct(stats.sample_push_ns),
        pct(stats.update_safety_ns),
        pct(stats.apply_event_ns),
        stats.event_count,
        stats.decision_count,
    );
}

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

pub(crate) use crate::data::replay_targets::bool_mask_to_f32;

#[inline]
pub(crate) fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[inline]
pub(crate) fn tile136_to_type(tile136: u8) -> u8 {
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

pub(crate) fn update_safety(safety: &mut [SafetyInfo; 4], event: &MjaiEvent) -> io::Result<()> {
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

pub(crate) fn next_discards_after(events: &[MjaiEvent]) -> io::Result<Vec<[Option<u8>; 4]>> {
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

pub(crate) fn final_scores(events: &[MjaiEvent]) -> [i32; 4] {
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

pub(crate) fn should_sample_replay_event(event: &MjaiEvent) -> bool {
    matches!(
        event,
        MjaiEvent::Dahai { .. }
            | MjaiEvent::Pon { .. }
            | MjaiEvent::Chi { .. }
            | MjaiEvent::Kan { .. }
            | MjaiEvent::Ankan { .. }
            | MjaiEvent::Kakan { .. }
    )
}

pub(crate) struct PreparedReplayDecision {
    pub actor: usize,
    pub obs: Observation,
    pub action_id: u8,
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub legal_mask_f32: [f32; HYDRA_ACTION_SPACE],
    pub obs_encoded: [f32; OBS_SIZE],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayTargetProfile {
    pub oracle: bool,
    pub safety_residual: bool,
    pub belief: bool,
    pub mixture: bool,
    pub exit: bool,
    pub delta_q: bool,
}

impl ReplayTargetProfile {
    pub const fn minimal_bc() -> Self {
        Self {
            oracle: false,
            safety_residual: false,
            belief: false,
            mixture: false,
            exit: false,
            delta_q: false,
        }
    }

    pub const fn with_optional_heads(
        oracle: bool,
        safety_residual: bool,
        belief: bool,
        mixture: bool,
        exit: bool,
        delta_q: bool,
    ) -> Self {
        Self {
            oracle,
            safety_residual,
            belief,
            mixture,
            exit,
            delta_q,
        }
    }
}

pub(crate) struct ReplayLoadPolicy<'a> {
    pub profile: ReplayTargetProfile,
    pub exit_provenance: SidecarProvenance,
    pub delta_q_provenance: SidecarProvenance,
    pub exit_sidecar: Option<&'a ExitSidecarIndex>,
    pub delta_q_sidecar: Option<&'a DeltaQSidecarIndex>,
}

impl<'a> ReplayLoadPolicy<'a> {
    pub const fn new(
        profile: ReplayTargetProfile,
        exit_provenance: SidecarProvenance,
        delta_q_provenance: SidecarProvenance,
        exit_sidecar: Option<&'a ExitSidecarIndex>,
        delta_q_sidecar: Option<&'a DeltaQSidecarIndex>,
    ) -> Self {
        Self {
            profile,
            exit_provenance,
            delta_q_provenance,
            exit_sidecar,
            delta_q_sidecar,
        }
    }

    fn has_joined_sidecars(&self) -> bool {
        self.exit_sidecar.is_some() || self.delta_q_sidecar.is_some()
    }
}

#[derive(Clone, Copy)]
struct OpponentEventTarget {
    tenpai: f32,
    next_discard: u8,
    waits: [f32; 34],
}

struct EventOpponentTargetCache {
    opp_next_abs: [u8; 4],
    targets: [Option<OpponentEventTarget>; 4],
    actor_relative: [Option<ActorRelativeOpponentTargets>; 4],
    exact_waits_ns: u128,
}

#[derive(Clone, Copy)]
struct ActorRelativeOpponentTargets {
    wait_sets: [[f32; 34]; 3],
    tenpai: [f32; 3],
    opp_next: [u8; 3],
    danger: [f32; 102],
    danger_mask: [f32; 102],
}

impl Default for ActorRelativeOpponentTargets {
    fn default() -> Self {
        Self {
            wait_sets: [[0.0; 34]; 3],
            tenpai: [0.0; 3],
            opp_next: [MISSING_TILE_TARGET; 3],
            danger: [0.0; 102],
            danger_mask: [0.0; 102],
        }
    }
}

impl EventOpponentTargetCache {
    fn new(next_discards: &[[Option<u8>; 4]], event_index: usize) -> Self {
        let mut opp_next_abs = [MISSING_TILE_TARGET; 4];
        for player in 0..4usize {
            opp_next_abs[player] =
                next_discards[event_index][player].unwrap_or(MISSING_TILE_TARGET);
        }
        Self {
            opp_next_abs,
            targets: [None; 4],
            actor_relative: [None; 4],
            exact_waits_ns: 0,
        }
    }

    fn target_for(&mut self, state: &GameState, player: usize) -> OpponentEventTarget {
        if let Some(target) = self.targets[player] {
            return target;
        }

        let t_waits = Instant::now();
        let (waits, is_tenpai) = exact_waits(state, player);
        self.exact_waits_ns += t_waits.elapsed().as_nanos();
        let target = OpponentEventTarget {
            tenpai: if is_tenpai { 1.0 } else { 0.0 },
            next_discard: self.opp_next_abs[player],
            waits,
        };
        self.targets[player] = Some(target);
        target
    }
}

fn actor_relative_opponent_targets(
    actor: usize,
    event_targets: &mut EventOpponentTargetCache,
    state: &GameState,
) -> ActorRelativeOpponentTargets {
    if let Some(targets) = event_targets.actor_relative[actor] {
        return targets;
    }
    let mut tenpai = [0.0; 3];
    let mut opp_next = [MISSING_TILE_TARGET; 3];
    let mut danger = [0.0; 102];
    let mut danger_mask = [0.0; 102];
    let mut wait_sets = [[0.0f32; 34]; 3];

    for rel in 0..3usize {
        let opp = abs_opp(actor, rel);
        let target = event_targets.target_for(state, opp);
        wait_sets[rel] = target.waits;
        tenpai[rel] = target.tenpai;
        opp_next[rel] = target.next_discard;
        let start = rel * 34;
        danger[start..start + 34].copy_from_slice(&wait_sets[rel]);
        if tenpai[rel] > 0.0 {
            danger_mask[start..start + 34].fill(1.0);
        }
    }

    let targets = ActorRelativeOpponentTargets {
        wait_sets,
        tenpai,
        opp_next,
        danger,
        danger_mask,
    };
    event_targets.actor_relative[actor] = Some(targets);
    targets
}

fn replay_phase_for_event(event: &MjaiEvent, state: &GameState, actor: usize) -> ActionPhase {
    if matches!(event, MjaiEvent::Dahai { .. })
        && (state.players[actor].riichi_stage || state.players[actor].riichi_declared)
    {
        ActionPhase::RiichiSelect
    } else {
        ActionPhase::Normal
    }
}

fn analyze_replay_legal_actions(
    legal: &[EngineAction],
    _phase: ActionPhase,
    chosen_action_id: u8,
) -> (
    [bool; HYDRA_ACTION_SPACE],
    [f32; HYDRA_ACTION_SPACE],
    bool,
    bool,
) {
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    let mut legal_mask_f32 = [0.0; HYDRA_ACTION_SPACE];
    let mut chosen_is_legal = false;
    let mut had_ron = false;

    for action in legal {
        had_ron |= action.action_type == ActionType::Ron;
        let Ok(hydra) = riichienv_to_hydra(action) else {
            continue;
        };
        let idx = hydra.id() as usize;
        if idx >= HYDRA_ACTION_SPACE {
            continue;
        }
        // Replay legality should mirror the engine's own action acceptance as closely as
        // possible. Phase-specific masking is useful for model targets, but it must not be
        // allowed to reject the chosen replay action here or we spuriously mark valid replays
        // as desyncs.
        legal_mask[idx] = true;
        legal_mask_f32[idx] = 1.0;
        chosen_is_legal |= hydra.id() == chosen_action_id;
    }

    (legal_mask, legal_mask_f32, chosen_is_legal, had_ron)
}

#[allow(
    clippy::too_many_arguments,
    reason = "replay decision finalization needs the full state context"
)]
fn finalize_prepared_replay_decision(
    actor: usize,
    env_action: EngineAction,
    obs: Observation,
    phase: ActionPhase,
    state: &GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    legal: &[EngineAction],
    options: ReplayDecisionOptions,
) -> io::Result<Option<PreparedReplayDecision>> {
    let hydra_action = riichienv_to_hydra(&env_action)
        .map_err(|err| invalid_data(format!("hydra action mapping failed: {err}")))?;
    let t_legal = Instant::now();
    let (legal_mask, legal_mask_f32, chosen_is_legal, _) =
        analyze_replay_legal_actions(legal, phase, hydra_action.id());
    REPLAY_LEGAL_MASK_BUILD_NS.fetch_add(t_legal.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if !chosen_is_legal {
        return Ok(None);
    }

    let t_encode = Instant::now();
    let obs_encoded = if options.use_bc_minimal_encode {
        // BC-minimal disables Group D Hand-EV inputs only; legality and replay filtering
        // are unchanged. Replay acceptance is decided by analyze_replay_legal_actions()
        // above, which operates on legal move masks, not the observation tensor.
        // The fixed-superset encoder zero-fills Hand-EV channels and sets the presence
        // mask to 0 when hand_ev is None, so the model sees a clean absent signal.
        encode_observation_with_profile(
            encoder,
            &obs,
            &safety[actor],
            state.drawn_tile.map(tile136_to_type),
            BridgeEncodeProfile::bc_minimal(),
        )
    } else {
        encode_observation(
            encoder,
            &obs,
            &safety[actor],
            state.drawn_tile.map(tile136_to_type),
        )
    };
    REPLAY_ENCODE_OBS_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok(Some(PreparedReplayDecision {
        actor,
        obs,
        action_id: hydra_action.id(),
        legal_mask,
        legal_mask_f32,
        obs_encoded,
    }))
}

fn observation_for_replay_event(
    state: &mut GameState,
    actor: usize,
    env_action: &EngineAction,
) -> io::Result<Observation> {
    let t_obs = Instant::now();
    let obs = state
        .get_observation_for_replay(actor as u8, env_action, &env_action.to_mjai())
        .map_err(|err| invalid_data(format!("replay observation failed: {err}")))?;
    REPLAY_OBSERVATION_NS.fetch_add(t_obs.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(obs)
}

fn observation_for_implicit_pass(state: &mut GameState, actor: u8) -> io::Result<Observation> {
    let t_obs = Instant::now();
    let obs = state.get_observation(actor);
    REPLAY_OBSERVATION_NS.fetch_add(t_obs.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(obs)
}

fn prepare_implicit_pass_decisions(
    next_event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    options: ReplayDecisionOptions,
) -> io::Result<Vec<PreparedReplayDecision>> {
    let t_pass = Instant::now();
    let mut decisions = Vec::new();
    if state.phase != Phase::WaitResponse {
        REPLAY_IMPLICIT_PASS_NS.fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(decisions);
    }

    if !matches!(
        next_event,
        MjaiEvent::Dahai { .. }
            | MjaiEvent::Pon { .. }
            | MjaiEvent::Chi { .. }
            | MjaiEvent::Kan { .. }
            | MjaiEvent::Ankan { .. }
            | MjaiEvent::Kakan { .. }
            | MjaiEvent::Reach { .. }
            | MjaiEvent::Hora { .. }
    ) {
        state.resolve_replay_all_passes();
        REPLAY_IMPLICIT_PASS_NS.fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(decisions);
    }

    let responding_actor = mjai_event_actor(next_event)
        .filter(|actor| state.active_player_slice().contains(&(*actor as u8)));
    let resolve_all_passes = responding_actor.is_none();

    let active_players = state.active_player_slice().to_vec();
    for pid in active_players {
        if Some(pid as usize) == responding_actor {
            continue;
        }

        let pass_action = EngineAction::new(ActionType::Pass, None, &[], Some(pid));
        let obs = observation_for_implicit_pass(state, pid)?;
        let legal = obs.legal_actions_method();
        let (_, _, _, had_ron) =
            analyze_replay_legal_actions(&legal, ActionPhase::Normal, hydra_core::action::PASS);
        if let Some(decision) = finalize_prepared_replay_decision(
            pid as usize,
            pass_action,
            obs,
            ActionPhase::Normal,
            state,
            safety,
            encoder,
            &legal,
            options,
        )? {
            decisions.push(decision);
        }

        if had_ron {
            state.players[pid as usize].missed_agari_doujun = true;
            if state.players[pid as usize].riichi_declared {
                state.players[pid as usize].missed_agari_riichi = true;
            }
        }
    }

    if resolve_all_passes {
        state.resolve_replay_all_passes();
    }

    REPLAY_IMPLICIT_PASS_NS.fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok(decisions)
}

#[cfg(test)]
pub(crate) fn prepare_replay_decisions(
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
) -> io::Result<Vec<PreparedReplayDecision>> {
    prepare_replay_decisions_with_options(
        event,
        state,
        safety,
        encoder,
        ReplayDecisionOptions::default(),
    )
}

fn prepare_replay_decisions_with_options(
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    options: ReplayDecisionOptions,
) -> io::Result<Vec<PreparedReplayDecision>> {
    let mut decisions = prepare_implicit_pass_decisions(event, state, safety, encoder, options)?;
    if !should_sample_replay_event(event) {
        return Ok(decisions);
    }

    let env_action = mjai_event_to_action(event)
        .map_err(|err| invalid_data(format!("replay action conversion failed: {err}")))?;
    let (Some(actor), Some(env_action)) = (mjai_event_actor(event), env_action) else {
        return Ok(decisions);
    };

    let obs = observation_for_replay_event(state, actor, &env_action)?;
    let legal = obs.legal_actions_method();
    if let Some(decision) = finalize_prepared_replay_decision(
        actor,
        env_action,
        obs,
        replay_phase_for_event(event, state, actor),
        state,
        safety,
        encoder,
        &legal,
        options,
    )? {
        decisions.push(decision);
    }

    Ok(decisions)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SidecarProvenance {
    pub source_net_hash: Option<u64>,
    pub source_version: Option<u32>,
}

impl SidecarProvenance {
    pub const fn new(source_net_hash: Option<u64>, source_version: Option<u32>) -> Self {
        Self {
            source_net_hash,
            source_version,
        }
    }

    fn complete(self) -> Option<(u64, u32)> {
        self.source_net_hash.zip(self.source_version)
    }
}

pub(crate) fn prepare_replay_decision(
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
) -> io::Result<Option<PreparedReplayDecision>> {
    Ok(prepare_replay_decisions_with_options(
        event,
        state,
        safety,
        encoder,
        ReplayDecisionOptions::default(),
    )?
    .into_iter()
    .find(|decision| decision.action_id != hydra_core::action::PASS))
}

fn lookup_joined_label<T, F>(
    sidecar: Option<&T>,
    replay_key: Option<ReplayDecisionKey>,
    action: u8,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    provenance: SidecarProvenance,
    lookup: F,
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>
where
    F: FnOnce(
        &T,
        &ReplayDecisionKey,
        u8,
        &[f32; HYDRA_ACTION_SPACE],
        u64,
        u32,
    ) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>,
{
    let replay_key = replay_key?;
    let (source_net_hash, source_version) = provenance.complete()?;
    let sidecar = sidecar?;
    lookup(
        sidecar,
        &replay_key,
        action,
        legal_mask,
        source_net_hash,
        source_version,
    )
}

fn load_game_from_events_internal(
    source_hash: Option<u64>,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let mut stats = ReplayProfileStats::default();
    let t_precompute = Instant::now();
    let final_scores = final_scores(&events);
    let placements = score_to_placements(final_scores);
    let oracle_target = profile
        .oracle
        .then(|| oracle_target_from_scores(final_scores));
    let next_discards = next_discards_after(&events)?;
    let grp_label = scores_to_grp_index(final_scores).map_err(invalid_data)?;
    stats.precompute_ns += t_precompute.elapsed().as_nanos();
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut samples = Vec::with_capacity(events.len());
    let needs_exit_lookup =
        profile.exit && exit_sidecar.is_some() && exit_provenance.complete().is_some();
    let needs_delta_q_lookup =
        profile.delta_q && delta_q_sidecar.is_some() && delta_q_provenance.complete().is_some();
    let needs_replay_key = source_hash.is_some() && (needs_exit_lookup || needs_delta_q_lookup);
    let decision_options = ReplayDecisionOptions {
        use_bc_minimal_encode: profile == ReplayTargetProfile::minimal_bc(),
    };

    for (idx, event) in events.iter().enumerate() {
        stats.event_count += 1;
        let t_prepare = Instant::now();
        let decisions = prepare_replay_decisions_with_options(
            event,
            &mut state,
            &safety,
            &mut encoder,
            decision_options,
        )?;
        stats.prepare_decisions_ns += t_prepare.elapsed().as_nanos();
        let mut event_targets = EventOpponentTargetCache::new(&next_discards, idx);
        for decision in decisions {
            stats.decision_count += 1;
            let actor = decision.actor;
            let legal_mask = decision.legal_mask_f32;
            let actor_targets = if profile == ReplayTargetProfile::minimal_bc() {
                ActorRelativeOpponentTargets::default()
            } else {
                let t_opp = Instant::now();
                let actor_targets =
                    actor_relative_opponent_targets(actor, &mut event_targets, &state);
                stats.opponent_targets_ns += t_opp.elapsed().as_nanos();
                stats.exact_waits_ns += event_targets.exact_waits_ns;
                event_targets.exact_waits_ns = 0;
                actor_targets
            };
            let (safety_residual, safety_residual_mask) = if profile.safety_residual {
                let t_safety = Instant::now();
                let (values, mask) = build_safety_residual_targets(
                    &legal_mask,
                    &safety[actor],
                    &actor_targets.wait_sets,
                );
                stats.safety_residual_ns += t_safety.elapsed().as_nanos();
                (Some(values), Some(mask))
            } else {
                (None, None)
            };
            let (belief_fields, mixture_weights, belief_fields_present, mixture_weights_present) =
                if profile.belief || profile.mixture {
                    let t_belief = Instant::now();
                    let (belief_fields, mixture_weights, belief_present, mixture_present) =
                        build_stage_a_belief_targets(&state, actor, &decision.obs);
                    stats.belief_targets_ns += t_belief.elapsed().as_nanos();
                    (
                        if profile.belief { belief_fields } else { None },
                        if profile.mixture {
                            mixture_weights
                        } else {
                            None
                        },
                        profile.belief && belief_present,
                        profile.mixture && mixture_present,
                    )
                } else {
                    (None, None, false, false)
                };
            let t_sidecar = Instant::now();
            let replay_key = needs_replay_key.then(|| ReplayDecisionKey {
                source_hash: source_hash.expect("needs_replay_key implies source hash"),
                event_index: idx as u32,
                actor: actor as u8,
                obs_hash: crate::training::live_exit::obs_hash(&decision.obs_encoded),
            });
            let joined_exit = lookup_joined_label(
                if needs_exit_lookup {
                    exit_sidecar
                } else {
                    None
                },
                replay_key,
                decision.action_id,
                &legal_mask,
                exit_provenance,
                |sidecar, key, action, legal_mask, source_net_hash, source_version| {
                    sidecar.lookup_label(key, action, legal_mask, source_net_hash, source_version)
                },
            );
            let joined_delta_q = lookup_joined_label(
                if needs_delta_q_lookup {
                    delta_q_sidecar
                } else {
                    None
                },
                replay_key,
                decision.action_id,
                &legal_mask,
                delta_q_provenance,
                |sidecar, key, action, legal_mask, source_net_hash, source_version| {
                    sidecar.lookup_label(key, action, legal_mask, source_net_hash, source_version)
                },
            );
            stats.sidecar_lookup_ns += t_sidecar.elapsed().as_nanos();
            let t_push = Instant::now();
            samples.push(MjaiSample {
                obs: decision.obs_encoded,
                action: decision.action_id,
                legal_mask,
                placement: placements[actor],
                score_delta: final_scores[actor] - state.players[actor].score,
                grp_label,
                oracle_target,
                tenpai: actor_targets.tenpai,
                opp_next: actor_targets.opp_next,
                danger: actor_targets.danger,
                danger_mask: actor_targets.danger_mask,
                safety_residual,
                safety_residual_mask,
                exit_target: joined_exit.map(|(target, _)| target),
                exit_mask: joined_exit.map(|(_, mask)| mask),
                delta_q_target: joined_delta_q.map(|(target, _)| target),
                delta_q_mask: joined_delta_q.map(|(_, mask)| mask),
                belief_fields,
                mixture_weights,
                belief_fields_present,
                mixture_weights_present,
            });
            stats.sample_push_ns += t_push.elapsed().as_nanos();
        }

        let t_update = Instant::now();
        update_safety(&mut safety, event)?;
        stats.update_safety_ns += t_update.elapsed().as_nanos();
        let t_apply = Instant::now();
        state.apply_mjai_event(event.clone());
        stats.apply_event_ns += t_apply.elapsed().as_nanos();
    }

    stats.implicit_pass_ns = REPLAY_IMPLICIT_PASS_NS.swap(0, Ordering::Relaxed) as u128;
    stats.replay_observation_ns = REPLAY_OBSERVATION_NS.swap(0, Ordering::Relaxed) as u128;
    stats.legal_mask_build_ns = REPLAY_LEGAL_MASK_BUILD_NS.swap(0, Ordering::Relaxed) as u128;
    stats.encode_observation_ns = REPLAY_ENCODE_OBS_NS.swap(0, Ordering::Relaxed) as u128;

    maybe_print_replay_profile(&stats);

    Ok(MjaiGame {
        samples,
        final_scores,
    })
}

fn load_game_from_events(events: Vec<MjaiEvent>) -> io::Result<MjaiGame> {
    load_game_from_events_internal(
        None,
        SidecarProvenance::default(),
        SidecarProvenance::default(),
        ReplayTargetProfile::minimal_bc(),
        events,
        None,
        None,
    )
}

pub fn load_game_from_events_with_sidecar(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let source_hash = source_hash_from_identity(source_identity);
    load_game_from_events_internal(
        Some(source_hash),
        exit_provenance,
        delta_q_provenance,
        profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_reader<R: BufRead>(reader: R) -> io::Result<MjaiGame> {
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    let game = load_game_from_events(events)?;
    if !game.samples.is_empty() {
        let stats = ReplayProfileStats {
            parse_events_ns: t_parse.elapsed().as_nanos(),
            ..ReplayProfileStats::default()
        };
        maybe_print_replay_profile(&stats);
    }
    Ok(game)
}

pub fn debug_first_replay_failure_from_reader<R: BufRead>(reader: R) -> io::Result<Option<String>> {
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;

    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let mut legal_buf = Vec::with_capacity(64);

    for (idx, event) in events.iter().enumerate() {
        match prepare_replay_decision(event, &mut state, &safety, &mut encoder) {
            Ok(_) => {}
            Err(err) => {
                let actor = mjai_event_actor(event).map(|actor| actor as u8);
                let env_action = mjai_event_to_action(event).map_err(|conv| {
                    invalid_data(format!("replay action conversion failed: {conv}"))
                })?;
                let legal_actions = if let Some(actor) = actor {
                    state.get_legal_actions_into(actor, &mut legal_buf);
                    format!("{:?}", legal_buf)
                } else {
                    "<actor unavailable>".to_string()
                };
                return Ok(Some(format!(
                    "EVENT_INDEX: {idx}\nEVENT: {:?}\nEVENT_ACTOR: {:?}\nENV_ACTION: {:?}\nSTATE_PHASE: {:?}\nSTATE_DRAWN: {:?}\nACTIVE_PLAYERS: {:?}\nLEGAL_ACTIONS: {}\nERROR: {}",
                    event,
                    actor,
                    env_action,
                    state.phase,
                    state.drawn_tile,
                    state.active_player_slice(),
                    legal_actions,
                    err
                )));
            }
        }

        update_safety(&mut safety, event)?;
        state.apply_mjai_event(event.clone());
    }

    Ok(None)
}

pub fn load_game_from_reader_with_sidecar<R: BufRead>(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    reader: R,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    load_game_from_events_with_sidecar(
        source_identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_stream<R: Read>(reader: R) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let is_gzip = {
        let buf = reader
            .fill_buf()
            .map_err(|err| invalid_data(format!("failed to inspect MJAI stream: {err}")))?;
        buf.starts_with(&[0x1f, 0x8b])
    };

    if is_gzip {
        return load_game_from_reader(BufReader::new(GzDecoder::new(reader)));
    }

    load_game_from_reader(reader)
}

pub fn load_game_from_stream_with_sidecar<R: Read>(
    source_identity: &str,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    reader: R,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let is_gzip = {
        let buf = reader
            .fill_buf()
            .map_err(|err| invalid_data(format!("failed to inspect MJAI stream: {err}")))?;
        buf.starts_with(&[0x1f, 0x8b])
    };

    if is_gzip {
        return load_game_from_reader_with_sidecar(
            source_identity,
            exit_provenance,
            delta_q_provenance,
            profile,
            BufReader::new(GzDecoder::new(reader)),
            exit_sidecar,
            delta_q_sidecar,
        );
    }

    load_game_from_reader_with_sidecar(
        source_identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        reader,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_path(path: impl AsRef<Path>) -> io::Result<MjaiGame> {
    let events = load_mjai_events_from_path(path)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))?;
    load_game_from_events(events)
}

pub fn load_game_from_path_with_sidecar(
    path: impl AsRef<Path>,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
) -> io::Result<MjaiGame> {
    let path = path.as_ref();
    let identity = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data(format!("invalid filename {}", path.display())))?;
    let events = load_mjai_events_from_path(path)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))?;
    load_game_from_events_with_sidecar(
        identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        events,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub(crate) fn load_game_from_path_with_policy(
    path: impl AsRef<Path>,
    policy: Option<&ReplayLoadPolicy<'_>>,
) -> io::Result<MjaiGame> {
    let path = path.as_ref();
    match policy.filter(|policy| policy.has_joined_sidecars()) {
        Some(policy) => load_game_from_path_with_sidecar(
            path,
            policy.exit_provenance,
            policy.delta_q_provenance,
            policy.profile,
            policy.exit_sidecar,
            policy.delta_q_sidecar,
        ),
        None => load_game_from_path(path),
    }
}

pub(crate) fn load_game_from_stream_with_policy<R: Read>(
    source_identity: &str,
    reader: R,
    policy: Option<&ReplayLoadPolicy<'_>>,
) -> io::Result<MjaiGame> {
    match policy.filter(|policy| policy.has_joined_sidecars()) {
        Some(policy) => load_game_from_stream_with_sidecar(
            source_identity,
            policy.exit_provenance,
            policy.delta_q_provenance,
            policy.profile,
            reader,
            policy.exit_sidecar,
            policy.delta_q_sidecar,
        ),
        None => load_game_from_stream(reader),
    }
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
    use crate::training::replay_delta_q::{DeltaQSidecarIndex, ReplayDeltaQRecordV1};
    use crate::training::replay_exit::{
        ExitSidecarIndex, ReplayDecisionKey, ReplayExitRecordV1, legal_mask_digest_from_f32,
    };
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use riichienv_core::action::Phase;
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
            if let Some(decision) =
                prepare_replay_decision(event, &mut state, &safety, &mut encoder)
                    .expect("prepare replay decision")
            {
                decisions.push((
                    ReplayDecisionKey {
                        source_hash: source_hash_from_identity(source_identity),
                        event_index: idx as u32,
                        actor: decision.actor as u8,
                        obs_hash: crate::training::live_exit::obs_hash(&decision.obs_encoded),
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
                    semantics: crate::training::replay_exit::REPLAY_EXIT_SEMANTICS_V1.to_string(),
                    provenance: crate::training::replay_exit::REPLAY_EXIT_PROVENANCE.to_string(),
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
                    semantics: crate::training::replay_delta_q::REPLAY_DELTA_Q_SEMANTICS_V1
                        .to_string(),
                    provenance: crate::training::replay_delta_q::REPLAY_DELTA_Q_PROVENANCE
                        .to_string(),
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
        let base = PathBuf::from("/home/nikketryhard/tmp");
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
                    crate::training::live_exit::obs_hash(&decision.obs_encoded),
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

        let hand_ev_payload =
            &sample.obs[hydra_core::encoder::HAND_EV_CHANNEL_START * 34..mask_offset];
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
            target.iter().zip(mask.iter()).all(|(&value, &mask_value)| {
                mask_value <= 0.0 || (-1.0..=1.0).contains(&value)
            })
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
                (Some(belief_fields), mixture_weights) => {
                    Some(crate::teacher::belief::StageABeliefTarget {
                        belief_fields,
                        mixture_weights,
                        trust: 1.0,
                        ess: 1.0,
                        entropy: 0.0,
                    })
                }
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
    fn load_game_from_reader_emits_pass_sample_for_skipped_pon_window() {
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
            game.samples
                .iter()
                .any(|sample| sample.action == hydra_core::action::PASS),
            "expected replay loader to emit a pass sample for the skipped pon window"
        );
    }

    #[test]
    fn prepare_replay_decision_emits_pass_without_mutating_response_state() {
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

        let response_before = state.active_player_slice().to_vec();
        let decisions = prepare_replay_decisions(&events[4], &mut state, &safety, &mut encoder)
            .expect("prepare replay decisions");

        let pass = decisions
            .iter()
            .find(|decision| decision.action_id == hydra_core::action::PASS)
            .expect("pass decision should exist");
        assert_eq!(pass.actor, 1);
        assert!(pass.legal_mask[hydra_core::action::PASS as usize]);
        assert!(response_before.is_empty() || response_before.as_slice() == [1]);
        assert!(state.active_player_slice().is_empty());
        assert_eq!(state.phase, riichienv_core::action::Phase::WaitAct);
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

        assert!(decisions.iter().any(|decision| {
            decision.actor == 0 && decision.action_id == hydra_core::action::PASS
        }));
        assert!(state.players[0].missed_agari_doujun);
        assert_eq!(state.phase, riichienv_core::action::Phase::WaitResponse);
        assert_eq!(state.active_player_slice(), &[0, 1]);
    }
}
