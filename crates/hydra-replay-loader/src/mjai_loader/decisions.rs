use super::*;

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

pub fn mjai_event_type_name(event: &MjaiEvent) -> &'static str {
    match event {
        MjaiEvent::StartGame { .. } => "start_game",
        MjaiEvent::StartKyoku { .. } => "start_kyoku",
        MjaiEvent::Tsumo { .. } => "tsumo",
        MjaiEvent::Dahai { .. } => "dahai",
        MjaiEvent::Reach { .. } => "reach",
        MjaiEvent::ReachAccepted { .. } => "reach_accepted",
        MjaiEvent::Pon { .. } => "pon",
        MjaiEvent::Chi { .. } => "chi",
        MjaiEvent::Kan { .. } => "kan",
        MjaiEvent::Ankan { .. } => "ankan",
        MjaiEvent::Kakan { .. } => "kakan",
        MjaiEvent::Dora { .. } => "dora",
        MjaiEvent::Kita { .. } => "kita",
        MjaiEvent::Hora { .. } => "hora",
        MjaiEvent::Ryukyoku { .. } => "ryukyoku",
        MjaiEvent::EndKyoku => "end_kyoku",
        MjaiEvent::EndGame => "end_game",
        MjaiEvent::Other => "other",
    }
}

fn replay_log_action_str<'a>(event: &'a MjaiEvent, env_action: &EngineAction) -> Cow<'a, str> {
    match event {
        MjaiEvent::Dahai { pai, .. }
        | MjaiEvent::Pon { pai, .. }
        | MjaiEvent::Chi { pai, .. }
        | MjaiEvent::Kan { pai, .. }
        | MjaiEvent::Kakan { pai, .. } => Cow::Borrowed(pai.as_str()),
        MjaiEvent::Ankan { consumed, .. } => consumed
            .first()
            .map_or_else(|| Cow::Borrowed(""), |pai| Cow::Borrowed(pai.as_str())),
        _ => Cow::Owned(env_action.to_mjai()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDecisionKind {
    ImplicitPass,
    SampledEvent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayDecisionPhase {
    Normal,
    RiichiSelect,
    KanSelect,
}

impl From<ActionPhase> for ReplayDecisionPhase {
    fn from(phase: ActionPhase) -> Self {
        match phase {
            ActionPhase::Normal => Self::Normal,
            ActionPhase::RiichiSelect => Self::RiichiSelect,
            ActionPhase::KanSelect => Self::KanSelect,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayDecisionTrace {
    pub event_index: usize,
    pub source_event_type: &'static str,
    pub actor: usize,
    pub kind: ReplayDecisionKind,
    pub phase: ReplayDecisionPhase,
    pub kyoku: u8,
    pub honba: u8,
    pub kyotaku: u32,
    pub oya: u8,
    pub round_wind: u8,
    pub response_target: Option<u8>,
}

pub struct PreparedReplayDecision {
    pub trace: ReplayDecisionTrace,
    pub actor: usize,
    pub obs: Observation,
    pub action_id: u8,
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub legal_mask_f32: [f32; HYDRA_ACTION_SPACE],
    pub obs_encoded: [f32; OBS_SIZE],
    pub compact_facts: CompactObservationFacts,
    pub use_ref_targets: bool,
}

#[derive(Clone, Copy)]
struct OpponentEventTarget {
    pub(super) tenpai: f32,
    next_discard: u8,
    waits: [f32; 34],
}

pub(super) struct EventOpponentTargetCache {
    opp_next_abs: [u8; 4],
    targets: [Option<OpponentEventTarget>; 4],
    actor_relative: [Option<ActorRelativeOpponentTargets>; 4],
    pub(super) exact_waits_ns: u128,
}

#[derive(Clone, Copy)]
pub(super) struct ActorRelativeOpponentTargets {
    pub(super) wait_sets: [[f32; 34]; 3],
    pub(super) tenpai: [f32; 3],
    pub(super) opp_next: [u8; 3],
    pub(super) danger: [f32; 102],
    pub(super) danger_mask: [f32; 102],
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
    pub(super) fn new(next_discards: &[[Option<u8>; 4]], event_index: usize) -> Self {
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

pub(super) fn actor_relative_opponent_targets(
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

fn replay_action_phase_for_event(
    event: &MjaiEvent,
    state: &GameState,
    actor: usize,
) -> ActionPhase {
    if matches!(event, MjaiEvent::Dahai { .. })
        && (state.players[actor].riichi_stage || state.players[actor].riichi_declared)
    {
        ActionPhase::RiichiSelect
    } else {
        ActionPhase::Normal
    }
}

pub(super) fn analyze_replay_legal_actions(
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

pub(super) fn replay_decision_trace(
    event_index: usize,
    source_event_type: &'static str,
    actor: usize,
    kind: ReplayDecisionKind,
    phase: ActionPhase,
    state: &GameState,
) -> ReplayDecisionTrace {
    ReplayDecisionTrace {
        event_index,
        source_event_type,
        actor,
        kind,
        phase: phase.into(),
        kyoku: state.kyoku_idx,
        honba: state.honba,
        kyotaku: state.riichi_sticks,
        oya: state.oya,
        round_wind: state.round_wind,
        response_target: state.last_discard.map(|(player, _)| player),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "replay decision finalization needs the full state context"
)]
pub(super) fn finalize_prepared_replay_decision(
    trace: ReplayDecisionTrace,
    actor: usize,
    env_action: EngineAction,
    obs: Observation,
    phase: ActionPhase,
    state: &GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    options: ReplayDecisionOptions,
) -> io::Result<Option<PreparedReplayDecision>> {
    let hydra_action = riichienv_to_hydra(&env_action)
        .map_err(|err| invalid_data(format!("hydra action mapping failed: {err}")))?;
    let t_legal = Instant::now();
    let legal = obs.legal_actions_ref();
    let (legal_mask, legal_mask_f32, chosen_is_legal, _) =
        analyze_replay_legal_actions(legal, phase, hydra_action.id());
    REPLAY_LEGAL_MASK_BUILD_NS.fetch_add(t_legal.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if !chosen_is_legal {
        return Ok(None);
    }

    let t_encode = Instant::now();
    let drawn_tile = state.drawn_tile.map(tile136_to_type);
    let extracted_facts = extract_observation_facts(&obs, drawn_tile);
    let encode_profile = options.observation_profile.bridge_profile();
    let obs_encoded = encode_extracted_observation_facts_with_profile(
        encoder,
        &extracted_facts,
        &safety[actor],
        encode_profile,
    );
    let compact_facts = CompactObservationFacts::from_encoder_inputs(
        extracted_facts.hand,
        extracted_facts.open_meld_counts,
        extracted_facts.drawn_tile,
        extracted_facts.shanten_batch.base,
        extracted_facts.shanten_batch.discard,
        &extracted_facts.discards,
        &extracted_facts.melds,
        &extracted_facts.dora,
        &extracted_facts.meta,
        &safety[actor],
        &obs_encoded,
        false,
    );
    REPLAY_ENCODE_OBS_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok(Some(PreparedReplayDecision {
        trace,
        actor,
        obs,
        action_id: hydra_action.id(),
        legal_mask,
        legal_mask_f32,
        obs_encoded,
        compact_facts,
        use_ref_targets: false,
    }))
}

#[allow(
    clippy::too_many_arguments,
    reason = "replay decision finalization needs the full state context"
)]
pub(super) fn finalize_prepared_replay_decision_ref(
    trace: ReplayDecisionTrace,
    actor: usize,
    env_action: EngineAction,
    phase: ActionPhase,
    state: &GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    legal: &[EngineAction],
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
    let obs_ref = state.observe(actor as u8);
    let extracted_facts = extract_observation_facts_ref(&obs_ref);
    let obs_encoded = encode_extracted_observation_facts_with_profile(
        encoder,
        &extracted_facts,
        &safety[actor],
        BridgeEncodeProfile::bc_minimal(),
    );
    let compact_facts = CompactObservationFacts::from_encoder_inputs(
        extracted_facts.hand,
        extracted_facts.open_meld_counts,
        extracted_facts.drawn_tile,
        extracted_facts.shanten_batch.base,
        extracted_facts.shanten_batch.discard,
        &extracted_facts.discards,
        &extracted_facts.melds,
        &extracted_facts.dora,
        &extracted_facts.meta,
        &safety[actor],
        &obs_encoded,
        false,
    );
    REPLAY_ENCODE_OBS_NS.fetch_add(t_encode.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok(Some(PreparedReplayDecision {
        trace,
        actor,
        obs: empty_replay_observation(actor),
        action_id: hydra_action.id(),
        legal_mask,
        legal_mask_f32,
        obs_encoded,
        compact_facts,
        use_ref_targets: true,
    }))
}

fn empty_replay_observation(actor: usize) -> Observation {
    Observation::new(
        actor as u8,
        std::array::from_fn(|_| Vec::new()),
        std::array::from_fn(|_| Vec::new()),
        std::array::from_fn(|_| Vec::new()),
        Vec::new(),
        [0; 4],
        [false; 4],
        Vec::new(),
        Vec::new(),
        0,
        0,
        0,
        0,
        0,
        Vec::new(),
        false,
        [None; 4],
        [None; 4],
        None,
    )
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

#[cfg(test)]
pub fn prepare_replay_decisions(
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
) -> io::Result<Vec<PreparedReplayDecision>> {
    prepare_replay_decisions_with_options(
        0,
        event,
        state,
        safety,
        encoder,
        ReplayDecisionOptions::default(),
    )
}

fn validate_strict_replay_event(event: &MjaiEvent, state: &mut GameState) -> io::Result<()> {
    if matches!(event, MjaiEvent::Hora { pai: None, .. }) {
        return Ok(());
    }

    let env_action = state
        .replay_action_for_mjai_event(event)
        .map_err(|err| invalid_data(format!("strict replay action conversion failed: {err}")))?;
    let (Some(actor), Some(env_action)) = (mjai_event_actor(event), env_action) else {
        return Ok(());
    };

    let mut legal = Vec::new();
    let log_action_str = replay_log_action_str(event, &env_action);
    state
        .get_replay_legal_actions_into(actor as u8, &env_action, &log_action_str, &mut legal)
        .map_err(|err| invalid_data(format!("strict replay legality failed: {err}")))?;
    Ok(())
}

pub(super) fn prepare_replay_decisions_with_options(
    event_index: usize,
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    options: ReplayDecisionOptions,
) -> io::Result<Vec<PreparedReplayDecision>> {
    let source_event_type = mjai_event_type_name(event);
    let mut decisions = prepare_implicit_pass_decisions(
        event,
        state,
        safety,
        encoder,
        options,
        event_index,
        source_event_type,
    )?;
    if options.strict_replay_legality {
        validate_strict_replay_event(event, state)?;
    }
    if !should_sample_replay_event(event) {
        return Ok(decisions);
    }

    let env_action = state
        .replay_action_for_mjai_event(event)
        .map_err(|err| invalid_data(format!("replay action conversion failed: {err}")))?;
    let (Some(actor), Some(env_action)) = (mjai_event_actor(event), env_action) else {
        return Ok(decisions);
    };

    if options.observation_profile.uses_ref_observation() {
        let mut legal = Vec::new();
        let log_action_str = replay_log_action_str(event, &env_action);
        let t_obs = Instant::now();
        state
            .get_replay_legal_actions_into(actor as u8, &env_action, &log_action_str, &mut legal)
            .map_err(|err| invalid_data(format!("replay observation failed: {err}")))?;
        REPLAY_OBSERVATION_NS.fetch_add(t_obs.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let phase = replay_action_phase_for_event(event, state, actor);
        if let Some(decision) = finalize_prepared_replay_decision_ref(
            replay_decision_trace(
                event_index,
                source_event_type,
                actor,
                ReplayDecisionKind::SampledEvent,
                phase,
                state,
            ),
            actor,
            env_action,
            phase,
            state,
            safety,
            encoder,
            &legal,
        )? {
            decisions.push(decision);
        }
        return Ok(decisions);
    }

    let obs = observation_for_replay_event(state, actor, &env_action)?;
    let phase = replay_action_phase_for_event(event, state, actor);
    if let Some(decision) = finalize_prepared_replay_decision(
        replay_decision_trace(
            event_index,
            source_event_type,
            actor,
            ReplayDecisionKind::SampledEvent,
            phase,
            state,
        ),
        actor,
        env_action,
        obs,
        phase,
        state,
        safety,
        encoder,
        options,
    )? {
        decisions.push(decision);
    }

    Ok(decisions)
}

pub fn prepare_replay_decision(
    event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
) -> io::Result<Option<PreparedReplayDecision>> {
    Ok(prepare_replay_decisions_with_options(
        0,
        event,
        state,
        safety,
        encoder,
        ReplayDecisionOptions::default(),
    )?
    .into_iter()
    .find(|decision| decision.action_id != hydra_core::action::PASS))
}
