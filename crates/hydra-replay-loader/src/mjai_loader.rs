//! MJAI `.json` / `.json.gz` / `.json.zst` loader for behavioral cloning data.

use crate::replay_targets::{
    build_safety_residual_targets, build_stage_a_belief_targets, exact_waits,
};
use crate::target_helpers::{obs_hash, oracle_target_from_scores};
use flate2::read::GzDecoder;
#[cfg(test)]
use hydra_core::action::{AKA_5M, DISCARD_END};
use hydra_core::action::{ActionPhase, HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::bridge::{
    BridgeEncodeProfile, encode_observation, encode_observation_with_profile,
};
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use hydra_data_core::{MjaiSample, score_to_placements, scores_to_grp_index};
use hydra_replay_sidecar::{
    DeltaQSidecarIndex, ExitSidecarIndex, ReplayDecisionKey, source_hash_from_identity,
};
use riichienv_core::action::{Action as EngineAction, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::replay::{MjaiEvent, mjai_event_actor, mjai_event_to_action, read_mjai_events};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use std::array;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use zstd::stream::read::Decoder as ZstdDecoder;

const MISSING_TILE_TARGET: u8 = 255;

#[derive(Clone, Copy, Default)]
struct ReplayDecisionOptions {
    use_bc_minimal_encode: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayMaterializationStats {
    pub decompress_ns: u128,
    pub json_parse_ns: u128,
    pub replay_update_ns: u128,
    pub observation_encode_ns: u128,
    pub mask_build_ns: u128,
    pub event_count: usize,
    pub decision_count: usize,
}

impl ReplayMaterializationStats {
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(
            self.decompress_ns
                .saturating_add(self.json_parse_ns)
                .saturating_add(self.replay_update_ns)
                .saturating_add(self.observation_encode_ns)
                .saturating_add(self.mask_build_ns)
                .min(u64::MAX as u128) as u64,
        )
    }

    pub fn merge_assign(&mut self, other: ReplayMaterializationStats) {
        self.decompress_ns = self.decompress_ns.saturating_add(other.decompress_ns);
        self.json_parse_ns = self.json_parse_ns.saturating_add(other.json_parse_ns);
        self.replay_update_ns = self.replay_update_ns.saturating_add(other.replay_update_ns);
        self.observation_encode_ns = self
            .observation_encode_ns
            .saturating_add(other.observation_encode_ns);
        self.mask_build_ns = self.mask_build_ns.saturating_add(other.mask_build_ns);
        self.event_count = self.event_count.saturating_add(other.event_count);
        self.decision_count = self.decision_count.saturating_add(other.decision_count);
    }
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
static REPLAY_IMPLICIT_PASS_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_OBSERVATION_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_LEGAL_MASK_BUILD_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_ENCODE_OBS_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_MATERIALIZATION_TOTALS: Mutex<ReplayMaterializationStats> =
    Mutex::new(ReplayMaterializationStats {
        decompress_ns: 0,
        json_parse_ns: 0,
        replay_update_ns: 0,
        observation_encode_ns: 0,
        mask_build_ns: 0,
        event_count: 0,
        decision_count: 0,
    });

fn replay_materialization_totals() -> MutexGuard<'static, ReplayMaterializationStats> {
    REPLAY_MATERIALIZATION_TOTALS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn drain_replay_materialization_stats() -> ReplayMaterializationStats {
    let mut totals = replay_materialization_totals();
    let stats = *totals;
    *totals = ReplayMaterializationStats::default();
    stats
}

pub fn peek_replay_materialization_stats() -> ReplayMaterializationStats {
    *replay_materialization_totals()
}

fn record_replay_materialization_stats(stats: ReplayMaterializationStats) {
    replay_materialization_totals().merge_assign(stats);
}
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

#[inline]
pub fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
pub fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[inline]
pub fn tile136_to_type(tile136: u8) -> u8 {
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

pub fn update_safety(safety: &mut [SafetyInfo; 4], event: &MjaiEvent) -> io::Result<()> {
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

pub fn next_discards_after(events: &[MjaiEvent]) -> io::Result<Vec<[Option<u8>; 4]>> {
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

pub fn final_scores(events: &[MjaiEvent]) -> [i32; 4] {
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

pub fn should_sample_replay_event(event: &MjaiEvent) -> bool {
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

pub struct PreparedReplayDecision {
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

pub struct ReplayLoadPolicy<'a> {
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
pub fn prepare_replay_decisions(
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

pub fn prepare_replay_decision(
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
                obs_hash: obs_hash(&decision.obs_encoded),
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
    record_replay_materialization_stats(ReplayMaterializationStats {
        decompress_ns: 0,
        json_parse_ns: 0,
        replay_update_ns: stats.update_safety_ns.saturating_add(stats.apply_event_ns),
        observation_encode_ns: stats
            .replay_observation_ns
            .saturating_add(stats.encode_observation_ns),
        mask_build_ns: stats.legal_mask_build_ns,
        event_count: stats.event_count,
        decision_count: stats.decision_count,
    });

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
    let parse_ns = t_parse.elapsed().as_nanos();
    let game = load_game_from_events(events)?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: parse_ns,
        ..ReplayMaterializationStats::default()
    });
    if !game.samples.is_empty() {
        let stats = ReplayProfileStats {
            parse_events_ns: parse_ns,
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
    let t_parse = Instant::now();
    let events = read_mjai_events(reader)
        .map_err(|err| invalid_data(format!("failed to parse MJAI events: {err}")))?;
    record_replay_materialization_stats(ReplayMaterializationStats {
        json_parse_ns: t_parse.elapsed().as_nanos(),
        ..ReplayMaterializationStats::default()
    });
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamCompression {
    Plain,
    Gzip,
    Zstd,
}

struct TimedRead<R> {
    inner: R,
    elapsed_ns: Arc<AtomicU64>,
}

impl<R> TimedRead<R> {
    fn new(inner: R) -> (Self, Arc<AtomicU64>) {
        let elapsed_ns = Arc::new(AtomicU64::new(0));
        (
            Self {
                inner,
                elapsed_ns: Arc::clone(&elapsed_ns),
            },
            elapsed_ns,
        )
    }
}

impl<R: Read> Read for TimedRead<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let start = Instant::now();
        let result = self.inner.read(buf);
        let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let _ = self
            .elapsed_ns
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(elapsed))
            });
        result
    }
}

fn inspect_stream_compression<R: BufRead>(reader: &mut R) -> io::Result<StreamCompression> {
    let buf = reader
        .fill_buf()
        .map_err(|err| invalid_data(format!("failed to inspect MJAI stream: {err}")))?;
    if buf.starts_with(&[0x1f, 0x8b]) {
        Ok(StreamCompression::Gzip)
    } else if buf.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Ok(StreamCompression::Zstd)
    } else {
        Ok(StreamCompression::Plain)
    }
}

fn record_decompression_result<T>(
    result: &io::Result<T>,
    elapsed_ns: &AtomicU64,
    compression: StreamCompression,
) {
    if result.is_ok() && !matches!(compression, StreamCompression::Plain) {
        record_replay_materialization_stats(ReplayMaterializationStats {
            decompress_ns: u128::from(elapsed_ns.load(Ordering::Relaxed)),
            ..ReplayMaterializationStats::default()
        });
    }
}

pub fn load_game_from_stream<R: Read>(reader: R) -> io::Result<MjaiGame> {
    let mut reader = BufReader::new(reader);
    let compression = inspect_stream_compression(&mut reader)?;

    match compression {
        StreamCompression::Gzip => {
            let (timed, elapsed_ns) = TimedRead::new(GzDecoder::new(reader));
            let result = load_game_from_reader(BufReader::new(timed));
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Zstd => {
            let zstd = ZstdDecoder::new(reader)
                .map_err(|err| invalid_data(format!("failed to open zstd MJAI stream: {err}")))?;
            let (timed, elapsed_ns) = TimedRead::new(zstd);
            let result = load_game_from_reader(BufReader::new(timed));
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Plain => load_game_from_reader(reader),
    }
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
    let compression = inspect_stream_compression(&mut reader)?;

    match compression {
        StreamCompression::Gzip => {
            let (timed, elapsed_ns) = TimedRead::new(GzDecoder::new(reader));
            let result = load_game_from_reader_with_sidecar(
                source_identity,
                exit_provenance,
                delta_q_provenance,
                profile,
                BufReader::new(timed),
                exit_sidecar,
                delta_q_sidecar,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Zstd => {
            let zstd = ZstdDecoder::new(reader)
                .map_err(|err| invalid_data(format!("failed to open zstd MJAI stream: {err}")))?;
            let (timed, elapsed_ns) = TimedRead::new(zstd);
            let result = load_game_from_reader_with_sidecar(
                source_identity,
                exit_provenance,
                delta_q_provenance,
                profile,
                BufReader::new(timed),
                exit_sidecar,
                delta_q_sidecar,
            );
            record_decompression_result(&result, elapsed_ns.as_ref(), compression);
            result
        }
        StreamCompression::Plain => load_game_from_reader_with_sidecar(
            source_identity,
            exit_provenance,
            delta_q_provenance,
            profile,
            reader,
            exit_sidecar,
            delta_q_sidecar,
        ),
    }
}

pub fn load_game_from_path(path: impl AsRef<Path>) -> io::Result<MjaiGame> {
    let file = fs::File::open(path)?;
    load_game_from_stream(file)
        .map_err(|err| invalid_data(format!("failed to load MJAI events: {err}")))
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
    let file = fs::File::open(path)?;
    load_game_from_stream_with_sidecar(
        identity,
        exit_provenance,
        delta_q_provenance,
        profile,
        file,
        exit_sidecar,
        delta_q_sidecar,
    )
}

pub fn load_game_from_path_with_policy(
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

pub fn load_game_from_stream_with_policy<R: Read>(
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
mod tests;
