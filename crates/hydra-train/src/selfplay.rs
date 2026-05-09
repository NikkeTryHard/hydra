use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use hydra_core::action::{
    ActionPhase, GameContext, HYDRA_ACTION_SPACE, HydraAction, build_legal_mask,
    hydra_to_riichienv, riichienv_to_hydra,
};
use hydra_core::afbs::{AfbsTree, NodeIdx};
use hydra_core::arena::{
    Trajectory, TrajectoryDeltaQLabel, TrajectoryExitLabel, TrajectoryStep,
    sample_action_with_temperature,
};
use hydra_core::bridge::{encode_observation, encode_observation_ref};
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::observation_ref::ObservationRef;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

use crate::model::HydraModel;
use crate::selfplay_batch::{RlBatchScratch, finalize_rewards, trajectories_to_rl_batch_reuse};
use crate::training::exit::{
    build_delta_q_from_afbs_tree, build_exit_from_afbs_tree, compatible_discard_state,
    is_hard_state,
};
use crate::training::gae::GaeConfig;
use crate::training::live_exit::{
    ExitSearchAdapter, LiveExitConfig, SelfPlayExitAdapter, TrajectorySearchLabels,
    base_pi_from_logits, budget_from_legal_count, legal_discard_actions, make_live_exit_fn,
    seed_root_children_all_legal,
};
use crate::training::rl::RlBatch;
use hydra_selfplay::cooperative_state::{
    ExitChildRequest, ExitSearchState, GameAdvance, PendingExitStep, PendingPolicyRequest,
    PendingTurnState, PreparedExitSearch,
};

pub use crate::selfplay_batch::{default_gae_config, trajectories_to_rl_batch};
pub use hydra_train_types::selfplay::{RootDecisionContext, StepRecord};

const DEFAULT_GAME_MODE: u8 = 0;
#[cfg(not(test))]
const MAX_SELF_PLAY_STEPS: u32 = 50_000;
#[cfg(test)]
const MAX_SELF_PLAY_STEPS: u32 = 500;
const NUM_OPPONENTS: usize = 3;

#[derive(Clone, Copy)]
struct PendingContext {
    phase: ActionPhase,
    last_discard: Option<u8>,
    hand: [u8; 14],
    hand_len: u8,
}

struct DecisionEnv<'a, F>
where
    F: FnMut(&[f32; OBS_SIZE]) -> [f32; HYDRA_ACTION_SPACE],
{
    state: &'a mut GameState,
    selector: &'a mut NnActionSelector,
    legal_buf: &'a mut Vec<Action>,
    trajectory: &'a mut Trajectory,
    infer_fn: &'a mut F,
    chosen_actions: &'a mut [Option<Action>; 4],
}

pub struct NnActionSelector {
    encoder: ObservationEncoder,
    safety: [SafetyInfo; 4],
    temperature: f32,
    rng_state: u64,
    last_step: Option<StepRecord>,
    pending_logits: Option<[f32; HYDRA_ACTION_SPACE]>,
    pending_obs: Option<[f32; OBS_SIZE]>,
    pending_context: Option<PendingContext>,
}

impl NnActionSelector {
    pub fn new(temperature: f32, seed: u64) -> Self {
        Self {
            encoder: ObservationEncoder::new(),
            safety: std::array::from_fn(|_| SafetyInfo::new()),
            temperature: temperature.max(1e-3),
            rng_state: seed.max(1),
            last_step: None,
            pending_logits: None,
            pending_obs: None,
            pending_context: None,
        }
    }

    pub fn set_logits(&mut self, logits: [f32; HYDRA_ACTION_SPACE]) {
        self.pending_logits = Some(logits);
    }

    pub fn encode_observation(
        &mut self,
        obs: &Observation,
        player: u8,
        drawn_tile: Option<u8>,
    ) -> [f32; OBS_SIZE] {
        let encoded = encode_observation(
            &mut self.encoder,
            obs,
            &self.safety[player as usize],
            drawn_tile,
        );
        self.pending_obs = Some(encoded);
        self.pending_context = Some(PendingContext {
            phase: infer_action_phase(obs.legal_actions_ref()),
            last_discard: obs.last_discard.and_then(|tile| u8::try_from(tile).ok()),
            hand: hand_from_observation(obs, player),
            hand_len: obs.hands[player as usize].len().min(14) as u8,
        });
        encoded
    }

    pub fn encode_observation_ref(
        &mut self,
        obs: &ObservationRef<'_>,
        legal_actions: &[Action],
        player: u8,
    ) -> [f32; OBS_SIZE] {
        let encoded = encode_observation_ref(&mut self.encoder, obs, &self.safety[player as usize]);
        self.pending_obs = Some(encoded);
        self.pending_context = Some(PendingContext {
            phase: infer_action_phase(legal_actions),
            last_discard: obs.discards[(player as usize + 3) % 4].last().copied(),
            hand: hand_from_observation_ref(obs),
            hand_len: obs.observer_hand.len().min(14) as u8,
        });
        encoded
    }

    pub fn update_safety_from_discard(&mut self, tile: u8, opp: usize, tedashi: bool) {
        self.safety[0].on_discard(tile, opp, tedashi);
    }

    pub fn update_safety_from_riichi(&mut self, opp: usize) {
        self.safety[0].on_riichi(opp);
    }

    pub fn reset_safety(&mut self) {
        for safety in &mut self.safety {
            safety.reset();
        }
    }

    pub fn reset_for_new_game(&mut self, temperature: f32, seed: u64) {
        self.reset_safety();
        self.temperature = temperature.max(1e-3);
        self.rng_state = seed.max(1);
        self.last_step = None;
        self.pending_logits = None;
        self.pending_obs = None;
        self.pending_context = None;
    }

    pub fn safety(&self, player: u8) -> &SafetyInfo {
        &self.safety[player as usize]
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn take_last_step(&mut self) -> Option<StepRecord> {
        self.last_step.take()
    }

    fn next_rng_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64 / u64::MAX as f64) as f32
    }

    fn select_matching_legal_action(
        &self,
        hydra_action: u8,
        legal_actions: &[Action],
    ) -> Option<Action> {
        legal_actions.iter().copied().find(|action| {
            riichienv_to_hydra(action)
                .map(|mapped| mapped.id() == hydra_action)
                .unwrap_or(false)
        })
    }

    fn fallback_action_from_context(&self, hydra_action: u8) -> Option<Action> {
        let context = self.pending_context?;
        let hydra = HydraAction::new(hydra_action)?;
        hydra_to_riichienv(
            hydra,
            &GameContext {
                last_discard: context.last_discard,
                phase: context.phase,
                hand: context.hand,
                hand_len: context.hand_len,
            },
        )
        .ok()
    }

    fn track_action(&mut self, actor: u8, drawn_tile: Option<u8>, action: &Action) {
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile136) = action.tile {
                    let tile_type = tile136 / 4;
                    let is_tsumogiri = drawn_tile == Some(tile136);
                    let is_tedashi = !is_tsumogiri;
                    for observer in 0..4u8 {
                        if observer == actor {
                            continue;
                        }
                        let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                        if opp_idx < NUM_OPPONENTS {
                            self.safety[observer as usize]
                                .on_discard(tile_type, opp_idx, is_tedashi);
                        }
                    }
                }
            }
            ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                let mut tile_types = [0u8; 4];
                let count = action.consume_count as usize;
                for (idx, &tile) in action.consume_slice().iter().enumerate() {
                    tile_types[idx] = tile / 4;
                }
                for safety in &mut self.safety {
                    safety.on_call(&tile_types[..count]);
                }
            }
            ActionType::Riichi => {
                for observer in 0..4u8 {
                    if observer == actor {
                        continue;
                    }
                    let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                    if opp_idx < NUM_OPPONENTS {
                        self.safety[observer as usize].on_riichi(opp_idx);
                    }
                }
            }
            _ => {}
        }
    }
}

impl hydra_core::game_loop::ActionSelector for NnActionSelector {
    fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action {
        let legal_mask = build_legal_mask(legal_actions, ActionPhase::Normal);
        let logits = self
            .pending_logits
            .take()
            .unwrap_or([0.0; HYDRA_ACTION_SPACE]);
        let obs = self.pending_obs.unwrap_or([0.0; OBS_SIZE]);
        let (hydra_action, pi_old) = sample_action_with_temperature(
            &logits,
            &legal_mask,
            self.temperature,
            self.next_rng_f32(),
        );

        self.last_step = Some(StepRecord {
            obs,
            action: hydra_action,
            policy_logits: logits,
            pi_old,
            legal_mask,
            player_id: player,
        });

        if let Some(action) = self.select_matching_legal_action(hydra_action, legal_actions) {
            return action;
        }
        if let Some(action) = self.fallback_action_from_context(hydra_action) {
            return action;
        }
        legal_actions[0]
    }
}

pub fn run_self_play_game<F>(
    game_seed: u64,
    temperature: f32,
    rng_seed: u64,
    infer_fn: F,
) -> Trajectory
where
    F: FnMut(&[f32; OBS_SIZE]) -> [f32; HYDRA_ACTION_SPACE],
{
    run_self_play_game_with_exit_labels(
        game_seed,
        temperature,
        rng_seed,
        infer_fn,
        |_, _, _, _, _| None,
    )
}

pub fn run_self_play_game_with_exit_labels<F, E>(
    game_seed: u64,
    temperature: f32,
    rng_seed: u64,
    mut infer_fn: F,
    mut exit_label_fn: E,
) -> Trajectory
where
    F: FnMut(&[f32; OBS_SIZE]) -> [f32; HYDRA_ACTION_SPACE],
    E: FnMut(
        &GameState,
        &Observation,
        &StepRecord,
        &SafetyInfo,
        u32,
    ) -> Option<TrajectorySearchLabels>,
{
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(DEFAULT_GAME_MODE, true, Some(game_seed), 0, rule);
    let mut selector = NnActionSelector::new(temperature, rng_seed);
    let mut legal_buf = Vec::with_capacity(HYDRA_ACTION_SPACE);
    let mut trajectory = Trajectory::new(0, game_seed);
    let mut total_steps = 0u32;

    while !state.is_done && total_steps < MAX_SELF_PLAY_STEPS {
        if state.needs_initialize_next_round {
            state.step_unchecked(&[None; 4]);
            selector.reset_safety();
            continue;
        }

        let mut chosen_actions = [None; 4];
        match state.phase {
            Phase::WaitAct => {
                let pid = state.current_player;
                run_player_decision(
                    &mut DecisionEnv {
                        state: &mut state,
                        selector: &mut selector,
                        legal_buf: &mut legal_buf,
                        trajectory: &mut trajectory,
                        infer_fn: &mut infer_fn,
                        chosen_actions: &mut chosen_actions,
                    },
                    pid,
                    total_steps,
                    &mut exit_label_fn,
                );
            }
            Phase::WaitResponse => {
                let active_players = state.active_player_slice().to_vec();
                for pid in active_players {
                    run_player_decision(
                        &mut DecisionEnv {
                            state: &mut state,
                            selector: &mut selector,
                            legal_buf: &mut legal_buf,
                            trajectory: &mut trajectory,
                            infer_fn: &mut infer_fn,
                            chosen_actions: &mut chosen_actions,
                        },
                        pid,
                        total_steps,
                        &mut exit_label_fn,
                    );
                }
            }
        }

        state.step_unchecked(&chosen_actions);
        total_steps = total_steps.saturating_add(1);
    }

    trajectory.final_scores = std::array::from_fn(|idx| state.players[idx].score);
    finalize_rewards(&mut trajectory);
    if let Some(last_step) = trajectory.steps.last_mut() {
        last_step.done = true;
    }
    trajectory
}

pub fn run_mixed_policy_game_scores<B: Backend>(
    game_seed: u64,
    temperature: f32,
    rng_seed: u64,
    seat_models: [&HydraModel<B>; 4],
    device: &B::Device,
) -> [i32; 4] {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(DEFAULT_GAME_MODE, true, Some(game_seed), 0, rule);
    let mut selector = NnActionSelector::new(temperature, rng_seed);
    let mut legal_buf = Vec::with_capacity(HYDRA_ACTION_SPACE);
    let mut total_steps = 0u32;

    while !state.is_done && total_steps < MAX_SELF_PLAY_STEPS {
        if state.needs_initialize_next_round {
            state.step_unchecked(&[None; 4]);
            selector.reset_safety();
            continue;
        }

        let mut chosen_actions = [None; 4];
        match state.phase {
            Phase::WaitAct => {
                let pid = state.current_player;
                run_mixed_player_decision(
                    &mut state,
                    &mut selector,
                    &mut legal_buf,
                    &seat_models,
                    device,
                    &mut chosen_actions,
                    pid,
                );
            }
            Phase::WaitResponse => {
                let active_players = state.active_player_slice().to_vec();
                for pid in active_players {
                    run_mixed_player_decision(
                        &mut state,
                        &mut selector,
                        &mut legal_buf,
                        &seat_models,
                        device,
                        &mut chosen_actions,
                        pid,
                    );
                }
            }
        }

        state.step_unchecked(&chosen_actions);
        total_steps = total_steps.saturating_add(1);
    }

    std::array::from_fn(|idx| state.players[idx].score)
}

/// Raw output from a batch of self-play games before RL batch collation.
///
/// Separates trajectory generation from batch construction so that tests
/// and future arena buffering can inspect individual game results.
pub struct SelfPlayBatchSource {
    /// Completed game trajectories, one per seed.
    pub trajectories: Vec<Trajectory>,
    /// Per-step value baselines for each trajectory, used by GAE.
    pub values: Vec<Vec<f32>>,
}

/// Generates self-play trajectories with optional live ExIt labels.
///
/// Runs one game per seed using the provided model for both action
/// selection and value estimation. When `live_exit_cfg.enabled` is true,
/// the ExIt producer attempts to generate search-distillation labels at
/// each decision point (subject to the producer's internal gates).
///
/// The returned [`SelfPlayBatchSource`] contains raw trajectories and
/// per-step value baselines suitable for [`trajectories_to_rl_batch`].
struct CooperativeGameRunner {
    state: GameState,
    selector: NnActionSelector,
    trajectory: Trajectory,
    legal_buf: Vec<Action>,
    total_steps: u32,
    max_steps: u32,
    done: bool,
    pending_policy_obs: Option<PendingPolicyRequest>,
    pending_exit_search: Option<ExitSearchState>,
    turn_state: Option<PendingTurnState>,
    exit_adapter: SelfPlayExitAdapter,
    live_exit_cfg: LiveExitConfig,
    step_values: Vec<f32>,
    exit_child_requests: Vec<(NodeIdx, [f32; OBS_SIZE])>,
}

impl CooperativeGameRunner {
    fn new(game_seed: u64, temperature: f32, rng_seed: u64, live_exit_cfg: LiveExitConfig) -> Self {
        let rule = GameRule::default_tenhou();
        Self {
            state: GameState::new(DEFAULT_GAME_MODE, true, Some(game_seed), 0, rule),
            selector: NnActionSelector::new(temperature, rng_seed),
            trajectory: Trajectory::new(0, game_seed),
            legal_buf: Vec::with_capacity(HYDRA_ACTION_SPACE),
            total_steps: 0,
            max_steps: MAX_SELF_PLAY_STEPS,
            done: false,
            pending_policy_obs: None,
            pending_exit_search: None,
            turn_state: None,
            exit_adapter: SelfPlayExitAdapter::new(),
            live_exit_cfg,
            step_values: Vec::new(),
            exit_child_requests: Vec::new(),
        }
    }

    fn reset_for_new_game(&mut self, game_seed: u64, temperature: f32, rng_seed: u64) {
        self.state.reset_for_new_game(Some(game_seed));
        self.selector.reset_for_new_game(temperature, rng_seed);
        self.trajectory = Trajectory::new(self.trajectory.game_id, game_seed);
        self.legal_buf.clear();
        self.total_steps = 0;
        self.done = false;
        self.pending_policy_obs = None;
        self.pending_exit_search = None;
        self.turn_state = None;
        self.exit_adapter.reset();
        self.step_values.clear();
        self.exit_child_requests.clear();
    }

    fn is_finished(&self) -> bool {
        self.done
    }

    fn take_trajectory_and_values(&mut self) -> (Trajectory, Vec<f32>) {
        if !self.done {
            self.finalize();
        }

        let next_step_capacity = self.trajectory.steps.capacity();
        let game_id = self.trajectory.game_id;
        let seed = self.trajectory.seed;
        let trajectory = std::mem::replace(
            &mut self.trajectory,
            Trajectory {
                steps: Vec::with_capacity(next_step_capacity),
                final_scores: [0; 4],
                game_id,
                seed,
            },
        );
        let next_value_capacity = self.step_values.capacity();
        let values = std::mem::replace(
            &mut self.step_values,
            Vec::with_capacity(next_value_capacity),
        );

        (trajectory, values)
    }

    fn advance_until_inference_needed(&mut self) -> GameAdvance {
        loop {
            if self.done {
                return GameAdvance::default();
            }

            if self.pending_policy_obs.is_some() {
                return GameAdvance { needs_policy: true };
            }

            if self.state.is_done || self.total_steps >= self.max_steps {
                self.finalize();
                return GameAdvance::default();
            }

            if self.state.needs_initialize_next_round {
                self.state.step_unchecked(&[None; 4]);
                self.selector.reset_safety();
                continue;
            }

            if self.turn_state.is_none() {
                let players = match self.state.phase {
                    Phase::WaitAct => vec![self.state.current_player],
                    Phase::WaitResponse => self.state.active_player_slice().to_vec(),
                };
                self.turn_state = Some(PendingTurnState::new(players, self.total_steps));
            }

            if self
                .turn_state
                .as_ref()
                .is_some_and(|turn| turn.next_index >= turn.players.len())
            {
                if self.has_pending_exit_search() {
                    return GameAdvance::default();
                }
                self.flush_turn();
                continue;
            }

            let (pid, turn) = {
                let turn_state = self.turn_state.as_ref().expect("pending turn state");
                (turn_state.players[turn_state.next_index], turn_state.turn)
            };

            self.state.get_legal_actions_into(pid, &mut self.legal_buf);
            if self.legal_buf.is_empty() {
                self.turn_state
                    .as_mut()
                    .expect("pending turn state")
                    .next_index += 1;
                continue;
            }

            let obs = self.state.observe(pid);
            let obs_encoded = self
                .selector
                .encode_observation_ref(&obs, &self.legal_buf, pid);
            self.pending_policy_obs = Some(PendingPolicyRequest {
                pid,
                obs_encoded,
                drawn_tile_before_action: self.state.drawn_tile,
                turn,
            });
            return GameAdvance { needs_policy: true };
        }
    }

    fn pending_policy_obs(&self) -> Option<[f32; OBS_SIZE]> {
        self.pending_policy_obs
            .as_ref()
            .map(|pending| pending.obs_encoded)
    }

    fn has_pending_exit_search(&self) -> bool {
        self.pending_exit_search
            .as_ref()
            .is_some_and(|pending| !pending.is_empty())
    }

    fn pending_exit_child_count(&self) -> usize {
        self.pending_exit_search
            .as_ref()
            .map_or(0, |pending| pending.child_requests.len())
    }

    fn append_pending_exit_obs(&self, batch_observations: &mut Vec<[f32; OBS_SIZE]>) {
        if let Some(pending) = &self.pending_exit_search {
            for child in &pending.child_requests {
                batch_observations.push(child.obs);
            }
        }
    }

    fn provide_policy_result(&mut self, logits: [f32; HYDRA_ACTION_SPACE], value: f32) {
        let pending = self
            .pending_policy_obs
            .take()
            .expect("pending policy request");
        self.selector.set_logits(logits);

        self.state
            .get_legal_actions_into(pending.pid, &mut self.legal_buf);
        if self.legal_buf.is_empty() {
            self.turn_state
                .as_mut()
                .expect("pending turn state")
                .next_index += 1;
            return;
        }

        let action = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
            &mut self.selector,
            pending.pid,
            &self.legal_buf,
        );
        self.selector
            .track_action(pending.pid, pending.drawn_tile_before_action, &action);

        {
            let turn_state = self.turn_state.as_mut().expect("pending turn state");
            turn_state.chosen_actions[pending.pid as usize] = Some(action);
            turn_state.next_index += 1;
        }

        if let Some(step_record) = self.selector.take_last_step() {
            let output_index = {
                let turn_state = self.turn_state.as_mut().expect("pending turn state");
                turn_state.pending_steps.push(None);
                turn_state.pending_values.push(value);
                turn_state.pending_steps.len() - 1
            };

            if let Some(prepared) = self.prepare_exit_search(&step_record, pending.turn) {
                let pending_exit = self
                    .pending_exit_search
                    .get_or_insert_with(ExitSearchState::new);
                let child_offset = pending_exit.child_requests.len();
                let child_count = prepared.child_requests.len();

                for (child_idx, obs) in prepared.child_requests {
                    pending_exit
                        .child_requests
                        .push(ExitChildRequest { child_idx, obs });
                }

                let mut step = prepared.step;
                step.child_offset = child_offset;
                step.child_count = child_count;
                step.output_index = output_index;
                pending_exit.steps.push(step);
            } else {
                let trajectory_step = self.build_trajectory_step(
                    step_record,
                    pending.turn,
                    TrajectorySearchLabels::default(),
                );
                self.turn_state
                    .as_mut()
                    .expect("pending turn state")
                    .pending_steps[output_index] = Some(trajectory_step);
            }
        }
    }

    fn finalize_pending_exit_search(&mut self, child_values: &[f32]) {
        let Some(pending_exit) = self.pending_exit_search.take() else {
            return;
        };

        let mut finalized_steps = Vec::with_capacity(pending_exit.steps.len());
        for mut exit_step in pending_exit.steps {
            let start = exit_step.child_offset;
            let end = start + exit_step.child_count;
            let mut labels = TrajectorySearchLabels::default();

            if let (Some(child_slice), Some(value_slice)) = (
                pending_exit.child_requests.get(start..end),
                child_values.get(start..end),
            ) {
                let mut valid = true;

                for (_child, &value) in child_slice.iter().zip(value_slice.iter()) {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    exit_step.tree.run_search_iterations(
                        exit_step.root,
                        exit_step.budget,
                        &|child_idx| {
                            child_slice
                                .iter()
                                .zip(value_slice.iter())
                                .find_map(|(child, &value)| {
                                    (child.child_idx == child_idx).then_some(value)
                                })
                                .unwrap_or(0.0)
                        },
                    );

                    let exit = build_exit_from_afbs_tree(
                        &exit_step.tree,
                        exit_step.root,
                        &exit_step.base_pi,
                        &exit_step.legal_f32,
                        exit_step.budget,
                        self.live_exit_cfg.exit_config.safety_valve_max_kl,
                    )
                    .and_then(|(target, mask)| TrajectoryExitLabel::from_slices(&target, &mask));
                    let delta_q = build_delta_q_from_afbs_tree(
                        &exit_step.tree,
                        exit_step.root,
                        &exit_step.legal_f32,
                    )
                    .and_then(|(target, mask)| TrajectoryDeltaQLabel::from_slices(&target, &mask));

                    if exit.is_some() || delta_q.is_some() {
                        labels = TrajectorySearchLabels { exit, delta_q };
                    }
                }
            }

            finalized_steps.push((
                exit_step.output_index,
                self.build_trajectory_step(exit_step.step_record, exit_step.turn, labels),
            ));
        }

        let turn_state = self.turn_state.as_mut().expect("pending turn state");
        for (output_index, step) in finalized_steps {
            turn_state.pending_steps[output_index] = Some(step);
        }
    }

    fn prepare_exit_search(
        &mut self,
        step_record: &StepRecord,
        turn: u32,
    ) -> Option<PreparedExitSearch> {
        if !self.live_exit_cfg.enabled {
            return None;
        }

        let ctx = RootDecisionContext::from_step(step_record);
        let legal_f32 = ctx
            .legal_mask
            .map(|is_legal| if is_legal { 1.0 } else { 0.0 });
        if !compatible_discard_state(&legal_f32) {
            return None;
        }

        let legal_discards = legal_discard_actions(step_record);
        if legal_discards.len() < 2 {
            return None;
        }

        let base_pi = base_pi_from_logits(step_record);
        let hard_slice: Vec<f32> = legal_discards
            .iter()
            .map(|&action| base_pi[action])
            .collect();
        if !is_hard_state(
            &hard_slice,
            self.live_exit_cfg.exit_config.hard_state_threshold,
        ) {
            return None;
        }

        let budget = budget_from_legal_count(&self.live_exit_cfg.exit_config, legal_discards.len());
        let root_hash = self
            .exit_adapter
            .root_hash(&self.state, ctx.player_id, &ctx.obs_encoded);
        let mut tree = AfbsTree::new();
        let root = tree.add_node(root_hash, 1.0, false);
        let priors: Vec<(u8, f32)> = legal_discards
            .iter()
            .map(|&action| (action as u8, base_pi[action]))
            .collect();
        seed_root_children_all_legal(&mut tree, root, root_hash, &priors);

        let player_safety = self.selector.safety(step_record.player_id).clone();
        self.exit_child_requests.clear();
        self.exit_child_requests
            .reserve(tree.nodes[root as usize].children.len());
        for &(action, child_idx) in &tree.nodes[root as usize].children.clone() {
            let child_obs = self.exit_adapter.child_public_obs_after_discard_ref(
                &self.state,
                ctx.player_id,
                action,
                &player_safety,
            )?;
            self.exit_child_requests.push((child_idx, child_obs));
        }

        Some(PreparedExitSearch {
            step: PendingExitStep {
                step_record: *step_record,
                turn,
                tree,
                root,
                base_pi,
                legal_f32,
                budget,
                child_offset: 0,
                child_count: 0,
                output_index: 0,
            },
            child_requests: std::mem::take(&mut self.exit_child_requests),
        })
    }

    fn build_trajectory_step(
        &self,
        step_record: StepRecord,
        turn: u32,
        search_labels: TrajectorySearchLabels,
    ) -> TrajectoryStep {
        TrajectoryStep {
            obs: step_record.obs,
            action: step_record.action,
            pi_old: step_record.pi_old,
            legal_mask: step_record.legal_mask,
            exit_label: search_labels.exit,
            delta_q_label: search_labels.delta_q,
            reward: 0.0,
            done: false,
            player_id: step_record.player_id,
            game_id: self.trajectory.game_id,
            turn: turn.min(u16::MAX as u32) as u16,
            temperature: self.selector.temperature(),
        }
    }

    fn flush_turn(&mut self) {
        debug_assert!(self.pending_policy_obs.is_none());
        debug_assert!(!self.has_pending_exit_search());

        let turn_state = self.turn_state.take().expect("pending turn state");
        self.trajectory.steps.extend(
            turn_state
                .pending_steps
                .into_iter()
                .map(|step| step.expect("pending turn step must be finalized before flush")),
        );
        self.step_values.extend(turn_state.pending_values);
        self.state.step_unchecked(&turn_state.chosen_actions);
        self.total_steps = self.total_steps.saturating_add(1);
    }

    fn finalize(&mut self) {
        if self.done {
            return;
        }

        self.trajectory.final_scores = std::array::from_fn(|idx| self.state.players[idx].score);
        finalize_rewards(&mut self.trajectory);
        if let Some(last_step) = self.trajectory.steps.last_mut() {
            last_step.done = true;
        }
        self.pending_policy_obs = None;
        self.pending_exit_search = None;
        self.turn_state = None;
        self.done = true;
    }
}

pub struct CooperativeSelfPlayCoordinator {
    games: Vec<CooperativeGameRunner>,
    batch_game_indices: Vec<usize>,
    batch_observations: Vec<[f32; OBS_SIZE]>,
    batch_outputs: Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
    exit_game_indices: Vec<usize>,
    exit_counts: Vec<usize>,
    exit_observations: Vec<[f32; OBS_SIZE]>,
    exit_values: Vec<f32>,
    flat_buf: Vec<f32>,
    rl_batch_scratch: RlBatchScratch,
    max_steps: u32,
}

pub struct CooperativeSelfPlayRequest<'a> {
    pub game_seeds: &'a [u64],
    pub temperature: f32,
    pub rng_seed: u64,
    pub live_exit_cfg: LiveExitConfig,
}

impl CooperativeSelfPlayCoordinator {
    pub fn new() -> Self {
        Self {
            games: Vec::new(),
            batch_game_indices: Vec::new(),
            batch_observations: Vec::new(),
            batch_outputs: Vec::new(),
            exit_game_indices: Vec::new(),
            exit_counts: Vec::new(),
            exit_observations: Vec::new(),
            exit_values: Vec::new(),
            flat_buf: Vec::new(),
            rl_batch_scratch: RlBatchScratch::default(),
            max_steps: MAX_SELF_PLAY_STEPS,
        }
    }

    #[cfg(test)]
    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    fn prepare_games(
        &mut self,
        game_seeds: &[u64],
        temperature: f32,
        rng_seed: u64,
        live_exit_cfg: &LiveExitConfig,
    ) {
        if self.games.len() < game_seeds.len() {
            for (idx, &seed) in game_seeds.iter().enumerate().skip(self.games.len()) {
                let mut game = CooperativeGameRunner::new(
                    seed,
                    temperature,
                    rng_seed.wrapping_add(idx as u64),
                    live_exit_cfg.clone(),
                );
                game.max_steps = self.max_steps;
                self.games.push(game);
            }
        }

        for (idx, game) in self.games.iter_mut().take(game_seeds.len()).enumerate() {
            game.live_exit_cfg = live_exit_cfg.clone();
            game.max_steps = self.max_steps;
            game.reset_for_new_game(
                game_seeds[idx],
                temperature,
                rng_seed.wrapping_add(idx as u64),
            );
        }

        let n = game_seeds.len();
        self.batch_game_indices.clear();
        self.batch_observations.clear();
        self.exit_game_indices.clear();
        self.exit_counts.clear();
        self.exit_observations.clear();

        if self.batch_game_indices.capacity() < n {
            self.batch_game_indices
                .reserve(n - self.batch_game_indices.capacity());
        }
        if self.batch_observations.capacity() < n {
            self.batch_observations
                .reserve(n - self.batch_observations.capacity());
        }
        if self.exit_game_indices.capacity() < n {
            self.exit_game_indices
                .reserve(n - self.exit_game_indices.capacity());
        }
        if self.exit_counts.capacity() < n {
            self.exit_counts.reserve(n - self.exit_counts.capacity());
        }
        let exit_capacity = n.saturating_mul(14);
        if self.exit_observations.capacity() < exit_capacity {
            self.exit_observations
                .reserve(exit_capacity - self.exit_observations.capacity());
        }
        let flat_capacity = n.saturating_mul(OBS_SIZE);
        if self.flat_buf.capacity() < flat_capacity {
            self.flat_buf
                .reserve(flat_capacity - self.flat_buf.capacity());
        }
    }

    fn run<B: Backend>(
        &mut self,
        game_seeds: &[u64],
        temperature: f32,
        rng_seed: u64,
        model: &HydraModel<B>,
        device: &B::Device,
        live_exit_cfg: LiveExitConfig,
    ) -> SelfPlayBatchSource {
        self.prepare_games(game_seeds, temperature, rng_seed, &live_exit_cfg);
        let active_games = &mut self.games[..game_seeds.len()];

        while active_games.iter().any(|game| !game.is_finished()) {
            loop {
                self.batch_game_indices.clear();
                self.batch_observations.clear();

                for (game_idx, game) in active_games.iter_mut().enumerate() {
                    let advance = game.advance_until_inference_needed();
                    if advance.needs_policy
                        && let Some(obs) = game.pending_policy_obs()
                    {
                        self.batch_game_indices.push(game_idx);
                        self.batch_observations.push(obs);
                    }
                }

                if self.batch_game_indices.is_empty() {
                    break;
                }

                model.fill_batch_policy_value_cpu(
                    &self.batch_observations,
                    device,
                    &mut self.flat_buf,
                    &mut self.batch_outputs,
                );
                for (game_idx, (policy_logits, value)) in self
                    .batch_game_indices
                    .drain(..)
                    .zip(self.batch_outputs.drain(..))
                {
                    active_games[game_idx].provide_policy_result(policy_logits, value);
                }
            }

            self.exit_game_indices.clear();
            self.exit_counts.clear();
            self.exit_observations.clear();
            for (game_idx, game) in active_games.iter().enumerate() {
                let child_count = game.pending_exit_child_count();
                if child_count > 0 {
                    self.exit_game_indices.push(game_idx);
                    self.exit_counts.push(child_count);
                    game.append_pending_exit_obs(&mut self.exit_observations);
                }
            }

            if self.exit_game_indices.is_empty() {
                continue;
            }

            model.fill_batch_value_cpu(
                &self.exit_observations,
                device,
                &mut self.flat_buf,
                &mut self.exit_values,
            );
            let mut offset = 0usize;
            for (game_idx, child_count) in self
                .exit_game_indices
                .drain(..)
                .zip(self.exit_counts.drain(..))
            {
                active_games[game_idx]
                    .finalize_pending_exit_search(&self.exit_values[offset..offset + child_count]);
                offset += child_count;
            }
        }

        let mut trajectories = Vec::with_capacity(active_games.len());
        let mut all_values = Vec::with_capacity(active_games.len());
        for game in active_games.iter_mut() {
            let (trajectory, values) = game.take_trajectory_and_values();
            trajectories.push(trajectory);
            all_values.push(values);
        }

        SelfPlayBatchSource {
            trajectories,
            values: all_values,
        }
    }
}

impl Default for CooperativeSelfPlayCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn generate_self_play_batch_source<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    live_exit_cfg: LiveExitConfig,
) -> SelfPlayBatchSource {
    let mut trajectories = Vec::with_capacity(game_seeds.len());
    let mut all_values = Vec::with_capacity(game_seeds.len());

    for (idx, &seed) in game_seeds.iter().enumerate() {
        let game_rng = rng_seed.wrapping_add(idx as u64);

        let infer_fn =
            |obs: &[f32; OBS_SIZE]| -> [f32; HYDRA_ACTION_SPACE] { model.policy_cpu(obs, device) };

        let exit_cfg = live_exit_cfg.clone();
        let exit_fn = make_live_exit_fn(exit_cfg, |obs: &[f32; OBS_SIZE]| {
            model.policy_and_value_cpu(obs, device)
        });

        let trajectory =
            run_self_play_game_with_exit_labels(seed, temperature, game_rng, infer_fn, exit_fn);

        let step_values: Vec<f32> = trajectory
            .steps
            .iter()
            .map(|step| model.value_cpu(&step.obs, device))
            .collect();

        all_values.push(step_values);
        trajectories.push(trajectory);
    }

    SelfPlayBatchSource {
        trajectories,
        values: all_values,
    }
}

pub fn generate_self_play_batch_source_cooperative<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    live_exit_cfg: LiveExitConfig,
) -> SelfPlayBatchSource {
    let mut coordinator = CooperativeSelfPlayCoordinator::new();
    coordinator.run(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        live_exit_cfg,
    )
}

pub fn generate_self_play_batch_source_cooperative_reuse<B: Backend>(
    coordinator: &mut CooperativeSelfPlayCoordinator,
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    live_exit_cfg: LiveExitConfig,
) -> SelfPlayBatchSource {
    coordinator.run(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        live_exit_cfg,
    )
}

pub fn generate_self_play_batch_source_batched<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    _inference_device: &B::Device,
    live_exit_cfg: LiveExitConfig,
) -> SelfPlayBatchSource {
    generate_self_play_batch_source_cooperative(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        live_exit_cfg,
    )
}

/// Generates a complete RL training batch from self-play games.
///
/// Self-play inference runs on the inner (non-autodiff) backend via
/// `model.valid()` to skip autograd graph construction. The final
/// `RlBatch` tensors are built on the autodiff backend for backprop.
pub fn generate_self_play_rl_batch<B: AutodiffBackend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    gae_config: &GaeConfig,
    live_exit_cfg: LiveExitConfig,
) -> RlBatch<B> {
    let valid_model = model.valid();
    let source = generate_self_play_batch_source_cooperative(
        game_seeds,
        temperature,
        rng_seed,
        &valid_model,
        device,
        live_exit_cfg,
    );
    trajectories_to_rl_batch(&source.trajectories, &source.values, gae_config, device)
}

pub fn generate_self_play_rl_batch_reuse<B: AutodiffBackend>(
    coordinator: &mut CooperativeSelfPlayCoordinator,
    request: CooperativeSelfPlayRequest<'_>,
    model: &HydraModel<B>,
    device: &B::Device,
    gae_config: &GaeConfig,
) -> RlBatch<B> {
    let valid_model = model.valid();
    let source = generate_self_play_batch_source_cooperative_reuse(
        coordinator,
        request.game_seeds,
        request.temperature,
        request.rng_seed,
        &valid_model,
        device,
        request.live_exit_cfg,
    );
    trajectories_to_rl_batch_reuse(
        &source.trajectories,
        &source.values,
        gae_config,
        device,
        &mut coordinator.rl_batch_scratch,
    )
}

fn run_player_decision<F, E>(
    env: &mut DecisionEnv<'_, F>,
    pid: u8,
    turn: u32,
    exit_label_fn: &mut E,
) where
    F: FnMut(&[f32; OBS_SIZE]) -> [f32; HYDRA_ACTION_SPACE],
    E: FnMut(
        &GameState,
        &Observation,
        &StepRecord,
        &SafetyInfo,
        u32,
    ) -> Option<TrajectorySearchLabels>,
{
    env.state.get_legal_actions_into(pid, env.legal_buf);
    if env.legal_buf.is_empty() {
        return;
    }

    let obs = env.state.observe(pid);
    let encoded = env
        .selector
        .encode_observation_ref(&obs, env.legal_buf, pid);
    let logits = (env.infer_fn)(&encoded);
    env.selector.set_logits(logits);

    let drawn_tile_before_action = env.state.drawn_tile;
    let action = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        env.selector,
        pid,
        env.legal_buf,
    );
    env.selector
        .track_action(pid, drawn_tile_before_action, &action);
    env.chosen_actions[pid as usize] = Some(action);

    if let Some(step_record) = env.selector.take_last_step() {
        let player_safety = env.selector.safety(pid);
        let owned_obs = env.state.get_observation(pid);
        let search_labels = exit_label_fn(env.state, &owned_obs, &step_record, player_safety, turn)
            .unwrap_or_default();
        env.trajectory.steps.push(TrajectoryStep {
            obs: step_record.obs,
            action: step_record.action,
            pi_old: step_record.pi_old,
            legal_mask: step_record.legal_mask,
            exit_label: search_labels.exit,
            delta_q_label: search_labels.delta_q,
            reward: 0.0,
            done: false,
            player_id: step_record.player_id,
            game_id: env.trajectory.game_id,
            turn: turn.min(u16::MAX as u32) as u16,
            temperature: env.selector.temperature(),
        });
    }
}

fn run_mixed_player_decision<B: Backend>(
    state: &mut GameState,
    selector: &mut NnActionSelector,
    legal_buf: &mut Vec<Action>,
    seat_models: &[&HydraModel<B>; 4],
    device: &B::Device,
    chosen_actions: &mut [Option<Action>; 4],
    pid: u8,
) {
    state.get_legal_actions_into(pid, legal_buf);
    if legal_buf.is_empty() {
        return;
    }

    let obs = state.observe(pid);
    let encoded = selector.encode_observation_ref(&obs, legal_buf, pid);
    let logits = seat_models[pid as usize].policy_cpu(&encoded, device);
    selector.set_logits(logits);

    let drawn_tile_before_action = state.drawn_tile;
    let action = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
        selector, pid, legal_buf,
    );
    selector.track_action(pid, drawn_tile_before_action, &action);
    chosen_actions[pid as usize] = Some(action);
}

fn infer_action_phase(legal_actions: &[Action]) -> ActionPhase {
    if legal_actions.iter().any(|action| {
        matches!(
            action.action_type,
            ActionType::Pass
                | ActionType::Ron
                | ActionType::Chi
                | ActionType::Pon
                | ActionType::Daiminkan
        )
    }) {
        ActionPhase::KanSelect
    } else {
        ActionPhase::Normal
    }
}

fn hand_from_observation(obs: &Observation, player: u8) -> [u8; 14] {
    let mut hand = [0u8; 14];
    for (idx, &tile) in obs.hands[player as usize].iter().take(14).enumerate() {
        if let Ok(tile_u8) = u8::try_from(tile) {
            hand[idx] = tile_u8;
        }
    }
    hand
}

fn hand_from_observation_ref(obs: &ObservationRef<'_>) -> [u8; 14] {
    let mut hand = [0u8; 14];
    for (idx, &tile) in obs.observer_hand.iter().take(14).enumerate() {
        hand[idx] = tile;
    }
    hand
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GAE_GAMMA, GAE_LAMBDA};
    use crate::model::HydraModelConfig;
    use crate::selfplay_batch::default_gae_config;
    use burn::backend::NdArray;
    use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};

    type B = NdArray<f32>;

    fn discard_actions() -> [Action; 2] {
        [
            Action::new(ActionType::Discard, Some(0), &[], None),
            Action::new(ActionType::Discard, Some(4), &[], None),
        ]
    }

    fn make_test_trajectory() -> Trajectory {
        let mut trajectory = Trajectory::new(7, 999);
        trajectory.final_scores = [32000, 26000, 22000, 20000];
        for idx in 0..4u8 {
            let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
            pi_old[idx as usize] = 0.7;
            pi_old[(idx as usize + 1) % HYDRA_ACTION_SPACE] = 0.3;
            trajectory.steps.push(TrajectoryStep {
                obs: [idx as f32; OBS_SIZE],
                action: idx,
                pi_old,
                legal_mask: {
                    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
                    legal_mask[idx as usize] = true;
                    legal_mask[(idx as usize + 1) % HYDRA_ACTION_SPACE] = true;
                    legal_mask
                },
                exit_label: None,
                delta_q_label: None,
                reward: if idx % 2 == 0 { 1.0 } else { -1.0 },
                done: idx == 3,
                player_id: idx % 4,
                game_id: 7,
                turn: idx as u16,
                temperature: 1.0,
            });
        }
        trajectory
    }

    fn small_test_model_config() -> HydraModelConfig {
        HydraModelConfig::new(1)
            .with_hidden_channels(2)
            .with_se_bottleneck(1)
            .with_num_groups(1)
    }

    fn make_test_step_record(player_id: u8, action: u8) -> StepRecord {
        let mut policy_logits = [0.0f32; HYDRA_ACTION_SPACE];
        policy_logits[0] = 1.0;
        policy_logits[1] = 0.5;
        let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
        pi_old[0] = 0.6;
        pi_old[1] = 0.4;
        let mut legal_mask = [false; HYDRA_ACTION_SPACE];
        legal_mask[0] = true;
        legal_mask[1] = true;
        StepRecord {
            obs: [player_id as f32; OBS_SIZE],
            action,
            policy_logits,
            pi_old,
            legal_mask,
            player_id,
        }
    }

    fn make_test_trajectory_step(player_id: u8, action: u8, turn: u16) -> TrajectoryStep {
        let record = make_test_step_record(player_id, action);
        TrajectoryStep {
            obs: record.obs,
            action: record.action,
            pi_old: record.pi_old,
            legal_mask: record.legal_mask,
            exit_label: None,
            delta_q_label: None,
            reward: 0.0,
            done: false,
            player_id,
            game_id: 99,
            turn,
            temperature: 1.0,
        }
    }

    #[test]
    fn test_nn_action_selector_selects_legal() {
        let mut selector = NnActionSelector::new(1.0, 42);
        selector.pending_obs = Some([0.0; OBS_SIZE]);
        selector.pending_context = Some(PendingContext {
            phase: ActionPhase::Normal,
            last_discard: None,
            hand: [0; 14],
            hand_len: 0,
        });
        let mut logits = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[1] = 1.0;
        selector.set_logits(logits);

        let legal_actions = discard_actions();
        let chosen = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
            &mut selector,
            0,
            &legal_actions,
        );

        assert!(legal_actions.contains(&chosen));
    }

    #[test]
    fn test_nn_action_selector_temperature() {
        let legal_actions = discard_actions();

        let mut low_temp = NnActionSelector::new(0.1, 7);
        low_temp.pending_obs = Some([0.0; OBS_SIZE]);
        low_temp.pending_context = Some(PendingContext {
            phase: ActionPhase::Normal,
            last_discard: None,
            hand: [0; 14],
            hand_len: 0,
        });
        let mut logits = [0.0; HYDRA_ACTION_SPACE];
        logits[0] = 3.0;
        logits[1] = 1.0;
        low_temp.set_logits(logits);
        let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
            &mut low_temp,
            0,
            &legal_actions,
        );
        let low_probs = low_temp
            .take_last_step()
            .map(|step| step.pi_old)
            .unwrap_or([0.0; HYDRA_ACTION_SPACE]);

        let mut high_temp = NnActionSelector::new(10.0, 7);
        high_temp.pending_obs = Some([0.0; OBS_SIZE]);
        high_temp.pending_context = Some(PendingContext {
            phase: ActionPhase::Normal,
            last_discard: None,
            hand: [0; 14],
            hand_len: 0,
        });
        high_temp.set_logits(logits);
        let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
            &mut high_temp,
            0,
            &legal_actions,
        );
        let high_probs = high_temp
            .take_last_step()
            .map(|step| step.pi_old)
            .unwrap_or([0.0; HYDRA_ACTION_SPACE]);

        assert!(low_probs[0] > high_probs[0]);
    }

    #[test]
    fn test_step_record_captured() {
        let mut selector = NnActionSelector::new(1.0, 11);
        selector.pending_obs = Some([1.0; OBS_SIZE]);
        selector.pending_context = Some(PendingContext {
            phase: ActionPhase::Normal,
            last_discard: None,
            hand: [0; 14],
            hand_len: 0,
        });
        let mut logits = [0.0; HYDRA_ACTION_SPACE];
        logits[0] = 2.0;
        selector.set_logits(logits);

        let _ = <NnActionSelector as hydra_core::game_loop::ActionSelector>::select_action(
            &mut selector,
            2,
            &discard_actions(),
        );
        let record = selector.take_last_step();

        assert!(record.is_some());
        let record = record.unwrap_or_else(|| unreachable!());
        assert_eq!(record.player_id, 2);
        assert_eq!(record.action, 0);
        assert_eq!(record.obs[0], 1.0);
        assert_eq!(record.policy_logits[0], 2.0);
        assert!(record.legal_mask[0]);
    }

    #[test]
    fn test_run_self_play_game_basic() {
        let trajectory = run_self_play_game(42, 1.0, 123, |_| [0.0; HYDRA_ACTION_SPACE]);
        assert!(!trajectory.steps.is_empty());
        assert!(trajectory.is_complete());
        assert!(trajectory.validate().is_ok());
    }

    #[test]
    fn test_generate_self_play_batch_source_without_exit() {
        let device = Default::default();
        let model = small_test_model_config().init::<B>(&device);
        let seeds = [42u64];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let source = generate_self_play_batch_source(&seeds, 1.0, 40, &model, &device, cfg);

        assert_eq!(source.trajectories.len(), 1);
        assert_eq!(source.values.len(), 1);
        for (traj, vals) in source.trajectories.iter().zip(source.values.iter()) {
            assert_eq!(traj.steps.len(), vals.len());
            assert!(traj.steps.iter().all(|s| s.exit_label.is_none()));
        }
    }

    #[test]
    fn test_batched_source_matches_serial() {
        let device = Default::default();
        let model = small_test_model_config().init::<B>(&device);
        let seeds = [42u64];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let serial = generate_self_play_batch_source(&seeds, 1.0, 40, &model, &device, cfg.clone());
        let batched =
            generate_self_play_batch_source_batched(&seeds, 1.0, 40, &model, &device, &device, cfg);

        assert_eq!(serial.trajectories.len(), batched.trajectories.len());
        assert_eq!(serial.values.len(), batched.values.len());

        for (idx, (s_traj, b_traj)) in serial
            .trajectories
            .iter()
            .zip(batched.trajectories.iter())
            .enumerate()
        {
            assert_eq!(
                s_traj.steps.len(),
                b_traj.steps.len(),
                "trajectory {idx} step count mismatch"
            );
            assert_eq!(s_traj.seed, b_traj.seed, "trajectory {idx} seed mismatch");
            assert_eq!(
                s_traj.final_scores, b_traj.final_scores,
                "trajectory {idx} final_scores mismatch"
            );
            for (step_idx, (s_step, b_step)) in
                s_traj.steps.iter().zip(b_traj.steps.iter()).enumerate()
            {
                assert_eq!(
                    s_step.action, b_step.action,
                    "trajectory {idx} step {step_idx} action mismatch"
                );
                assert_eq!(
                    s_step.player_id, b_step.player_id,
                    "trajectory {idx} step {step_idx} player_id mismatch"
                );
                assert_eq!(
                    s_step.legal_mask, b_step.legal_mask,
                    "trajectory {idx} step {step_idx} legal_mask mismatch"
                );
                assert_eq!(
                    s_step.exit_label.is_some(),
                    b_step.exit_label.is_some(),
                    "trajectory {idx} step {step_idx} exit_label presence mismatch"
                );
                assert_eq!(
                    s_step.delta_q_label.is_some(),
                    b_step.delta_q_label.is_some(),
                    "trajectory {idx} step {step_idx} delta_q_label presence mismatch"
                );
            }
        }

        for (idx, (s_vals, b_vals)) in serial.values.iter().zip(batched.values.iter()).enumerate() {
            assert_eq!(
                s_vals.len(),
                b_vals.len(),
                "trajectory {idx} value count mismatch"
            );
            for (step_idx, (s_val, b_val)) in s_vals.iter().zip(b_vals.iter()).enumerate() {
                assert!(
                    (s_val - b_val).abs() < 1e-5,
                    "trajectory {idx} step {step_idx} value mismatch: serial={s_val} batched={b_val}"
                );
            }
        }
    }

    fn assert_batch_sources_match(expected: &SelfPlayBatchSource, actual: &SelfPlayBatchSource) {
        assert_eq!(expected.trajectories.len(), actual.trajectories.len());
        assert_eq!(expected.values.len(), actual.values.len());

        for (idx, (lhs, rhs)) in expected
            .trajectories
            .iter()
            .zip(actual.trajectories.iter())
            .enumerate()
        {
            assert_eq!(lhs.seed, rhs.seed, "trajectory {idx} seed mismatch");
            assert_eq!(
                lhs.final_scores, rhs.final_scores,
                "trajectory {idx} final score mismatch"
            );
            assert_eq!(
                lhs.steps.len(),
                rhs.steps.len(),
                "trajectory {idx} step count mismatch"
            );

            for (step_idx, (lhs_step, rhs_step)) in
                lhs.steps.iter().zip(rhs.steps.iter()).enumerate()
            {
                assert_eq!(
                    lhs_step.action, rhs_step.action,
                    "trajectory {idx} step {step_idx} action mismatch"
                );
                assert_eq!(
                    lhs_step.player_id, rhs_step.player_id,
                    "trajectory {idx} step {step_idx} player mismatch"
                );
                assert_eq!(
                    lhs_step.turn, rhs_step.turn,
                    "trajectory {idx} step {step_idx} turn mismatch"
                );
                assert_eq!(
                    lhs_step.done, rhs_step.done,
                    "trajectory {idx} step {step_idx} done mismatch"
                );
                assert_eq!(
                    lhs_step.legal_mask, rhs_step.legal_mask,
                    "trajectory {idx} step {step_idx} legal mask mismatch"
                );
                assert_eq!(
                    lhs_step.exit_label, rhs_step.exit_label,
                    "trajectory {idx} step {step_idx} exit label mismatch"
                );
                assert_eq!(
                    lhs_step.delta_q_label, rhs_step.delta_q_label,
                    "trajectory {idx} step {step_idx} delta q mismatch"
                );
                assert_eq!(
                    lhs_step.obs, rhs_step.obs,
                    "trajectory {idx} step {step_idx} obs mismatch"
                );
                assert_eq!(
                    lhs_step.pi_old, rhs_step.pi_old,
                    "trajectory {idx} step {step_idx} pi_old mismatch"
                );
            }
        }

        for (idx, (lhs, rhs)) in expected.values.iter().zip(actual.values.iter()).enumerate() {
            assert_eq!(lhs.len(), rhs.len(), "values {idx} len mismatch");
            for (step_idx, (&lhs_value, &rhs_value)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert!(
                    (lhs_value - rhs_value).abs() < 1e-6,
                    "values {idx} step {step_idx} mismatch: {lhs_value} vs {rhs_value}"
                );
            }
        }
    }

    #[test]
    fn cooperative_reuse_matches_fresh_batches() {
        let device = Default::default();
        let model = small_test_model_config().init::<B>(&device);
        let seeds = [42u64];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let fresh = generate_self_play_batch_source_cooperative(
            &seeds,
            1.0,
            4,
            &model,
            &device,
            cfg.clone(),
        );
        let mut coordinator = CooperativeSelfPlayCoordinator::new();
        let reused = generate_self_play_batch_source_cooperative_reuse(
            &mut coordinator,
            &seeds,
            1.0,
            4,
            &model,
            &device,
            cfg,
        );

        assert_batch_sources_match(&fresh, &reused);
    }

    #[test]
    fn cooperative_reuse_does_not_leak_state_across_batches() {
        let device = Default::default();
        let model = small_test_model_config().init::<B>(&device);
        let first_seeds = [100u64];
        let second_seeds = [200u64];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let mut coordinator = CooperativeSelfPlayCoordinator::new();
        let first_reused = generate_self_play_batch_source_cooperative_reuse(
            &mut coordinator,
            &first_seeds,
            0.85,
            7,
            &model,
            &device,
            cfg.clone(),
        );
        let second_reused = generate_self_play_batch_source_cooperative_reuse(
            &mut coordinator,
            &second_seeds,
            0.85,
            9,
            &model,
            &device,
            cfg.clone(),
        );

        let first_fresh = generate_self_play_batch_source_cooperative(
            &first_seeds,
            0.85,
            7,
            &model,
            &device,
            cfg.clone(),
        );
        let second_fresh = generate_self_play_batch_source_cooperative(
            &second_seeds,
            0.85,
            9,
            &model,
            &device,
            cfg,
        );

        assert_batch_sources_match(&first_fresh, &first_reused);
        assert_batch_sources_match(&second_fresh, &second_reused);
        assert_eq!(
            second_reused
                .trajectories
                .iter()
                .map(|trajectory| trajectory.seed)
                .collect::<Vec<_>>(),
            second_seeds
        );
    }

    #[test]
    fn mixed_policy_runner_matches_single_model_selfplay_when_all_seats_share_model() {
        let device = Default::default();
        let model = small_test_model_config().init::<B>(&device);

        let trajectory = run_self_play_game(77, 1.0, 1234, |obs| model.policy_cpu(obs, &device));
        let scores =
            run_mixed_policy_game_scores(77, 1.0, 1234, [&model, &model, &model, &model], &device);

        assert_eq!(trajectory.final_scores, scores);
    }

    #[test]
    fn test_generate_self_play_rl_batch_produces_valid_batch() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let model = small_test_model_config().init::<AB>(&device);
        let seeds = [42u64];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };
        let gae = GaeConfig {
            gamma: GAE_GAMMA,
            lambda: GAE_LAMBDA,
        };

        let batch = generate_self_play_rl_batch(&seeds, 1.0, 40, &model, &device, &gae, cfg);
        let [steps, action_dim] = batch.targets.policy_target.dims();
        assert!(steps > 0);
        assert_eq!(action_dim, HYDRA_ACTION_SPACE);
    }

    #[test]
    fn cooperative_runner_waits_for_pending_exit_search_before_flushing_turn() {
        let mut runner = CooperativeGameRunner::new(42, 1.0, 123, LiveExitConfig::default());
        let mut tree = AfbsTree::new();
        let root = tree.add_node(7, 1.0, false);
        let mut turn_state = PendingTurnState::new(vec![0], 7);
        turn_state.next_index = turn_state.players.len();
        turn_state.pending_steps.push(None);
        turn_state.pending_values.push(0.25);
        runner.turn_state = Some(turn_state);
        runner.pending_exit_search = Some(ExitSearchState {
            steps: vec![PendingExitStep {
                step_record: make_test_step_record(0, 0),
                turn: 7,
                tree,
                root,
                base_pi: [0.0; HYDRA_ACTION_SPACE],
                legal_f32: [0.0; HYDRA_ACTION_SPACE],
                budget: 0,
                child_offset: 0,
                child_count: 1,
                output_index: 0,
            }],
            child_requests: vec![ExitChildRequest {
                child_idx: root,
                obs: [0.0; OBS_SIZE],
            }],
        });

        let advance = runner.advance_until_inference_needed();

        assert!(!advance.needs_policy);
        assert_eq!(runner.trajectory.steps.len(), 0);
        assert_eq!(runner.step_values.len(), 0);
        assert_eq!(runner.total_steps, 0);
        assert!(runner.turn_state.is_some());
        assert!(runner.has_pending_exit_search());
    }

    #[test]
    fn cooperative_runner_finalizes_pending_exit_step_into_original_slot() {
        let mut runner = CooperativeGameRunner::new(42, 1.0, 123, LiveExitConfig::default());
        let preserved_a_action = 0u8;
        let preserved_b_action = 1u8;
        let mut turn_state = PendingTurnState::new(vec![0, 1, 2], 3);
        turn_state.pending_steps = vec![
            Some(make_test_trajectory_step(0, preserved_a_action, 3)),
            None,
            Some(make_test_trajectory_step(2, preserved_b_action, 3)),
        ];
        turn_state.pending_values = vec![0.1, 0.2, 0.3];
        runner.turn_state = Some(turn_state);

        let mut tree = AfbsTree::new();
        let root = tree.add_node(11, 1.0, false);
        let delayed_record = make_test_step_record(1, 1);
        runner.pending_exit_search = Some(ExitSearchState {
            steps: vec![PendingExitStep {
                step_record: delayed_record,
                turn: 3,
                tree,
                root,
                base_pi: [0.0; HYDRA_ACTION_SPACE],
                legal_f32: [0.0; HYDRA_ACTION_SPACE],
                budget: 0,
                child_offset: 0,
                child_count: 1,
                output_index: 1,
            }],
            child_requests: vec![ExitChildRequest {
                child_idx: root,
                obs: [0.0; OBS_SIZE],
            }],
        });

        runner.finalize_pending_exit_search(&[]);

        assert!(runner.pending_exit_search.is_none());
        let turn_state = runner.turn_state.as_ref().expect("turn state");
        assert_eq!(turn_state.pending_values, vec![0.1, 0.2, 0.3]);
        assert_eq!(
            turn_state.pending_steps[0].as_ref().map(|s| s.action),
            Some(preserved_a_action)
        );
        let inserted = turn_state.pending_steps[1]
            .as_ref()
            .expect("delayed step inserted");
        assert_eq!(inserted.action, delayed_record.action);
        assert_eq!(inserted.player_id, delayed_record.player_id);
        assert_eq!(inserted.turn, 3);
        assert!(inserted.exit_label.is_none());
        assert!(inserted.delta_q_label.is_none());
        assert_eq!(
            turn_state.pending_steps[2].as_ref().map(|s| s.action),
            Some(preserved_b_action)
        );
    }

    #[test]
    fn test_trajectories_to_rl_batch_shapes() {
        let device = Default::default();
        let trajectory = make_test_trajectory();
        let values = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let batch =
            trajectories_to_rl_batch::<B>(&[trajectory], &values, &default_gae_config(), &device);

        assert_eq!(batch.obs.dims(), [4, NUM_CHANNELS, NUM_TILES]);
        assert_eq!(batch.actions.dims(), [4]);
        assert_eq!(batch.pi_old.dims(), [4]);
        assert_eq!(batch.advantages.dims(), [4]);
        assert_eq!(batch.base_logits.dims(), [4, HYDRA_ACTION_SPACE]);
        assert_eq!(batch.targets.legal_mask.dims(), [4, HYDRA_ACTION_SPACE]);
        assert!(batch.exit_target.is_none());
        assert!(batch.exit_mask.is_none());
    }

    #[test]
    fn test_trajectories_to_rl_batch_advantages_normalized() {
        let device = Default::default();
        let trajectory_a = make_test_trajectory();
        let mut trajectory_b = make_test_trajectory();
        trajectory_b.game_id = 8;
        trajectory_b.seed = 1000;
        trajectory_b.final_scores = [18000, 24000, 26000, 32000];

        let batch = trajectories_to_rl_batch::<B>(
            &[trajectory_a, trajectory_b],
            &[vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.3, 0.2, 0.1]],
            &default_gae_config(),
            &device,
        );

        let advantage_data = batch
            .advantages
            .to_data()
            .as_slice::<f32>()
            .expect("advantages")
            .to_vec();
        let mean = advantage_data.iter().sum::<f32>() / advantage_data.len() as f32;
        let variance = advantage_data
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / advantage_data.len() as f32;

        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
        assert!(
            (variance - 1.0).abs() < 0.1,
            "variance should be ~1, got {variance}"
        );
    }

    #[test]
    fn test_trajectories_to_rl_batch_collates_exit_labels() {
        let device = Default::default();
        let mut trajectory = make_test_trajectory();

        let exit_label = TrajectoryExitLabel::from_slices(
            &{
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                target[0] = 0.625;
                target[1] = 0.375;
                target
            },
            &{
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                mask[0] = 1.0;
                mask[1] = 1.0;
                mask
            },
        )
        .expect("valid exit label");

        trajectory.steps[0].exit_label = Some(exit_label);

        let batch = trajectories_to_rl_batch::<B>(
            &[trajectory],
            &[vec![0.1, 0.2, 0.3, 0.4]],
            &default_gae_config(),
            &device,
        );

        let exit_target = batch.exit_target.expect("exit target");
        let exit_mask = batch.exit_mask.expect("exit mask");
        assert_eq!(exit_target.dims(), [4, HYDRA_ACTION_SPACE]);
        assert_eq!(exit_mask.dims(), [4, HYDRA_ACTION_SPACE]);

        let target_data = exit_target
            .to_data()
            .as_slice::<f32>()
            .expect("exit target slice")
            .to_vec();
        let mask_data = exit_mask
            .to_data()
            .as_slice::<f32>()
            .expect("exit mask slice")
            .to_vec();

        assert!((target_data[0] - 0.625).abs() < 1e-6);
        assert!((target_data[1] - 0.375).abs() < 1e-6);
        assert_eq!(mask_data[0], 1.0);
        assert_eq!(mask_data[1], 1.0);

        let second_row_offset = HYDRA_ACTION_SPACE;
        assert!(
            target_data[second_row_offset..second_row_offset + HYDRA_ACTION_SPACE]
                .iter()
                .all(|value| value.abs() < 1e-6)
        );
        assert!(
            mask_data[second_row_offset..second_row_offset + HYDRA_ACTION_SPACE]
                .iter()
                .all(|value| value.abs() < 1e-6)
        );
    }

    #[test]
    fn test_run_self_play_game_with_exit_labels_persists_hook_output() {
        let trajectory = run_self_play_game_with_exit_labels(
            42,
            1.0,
            123,
            |_| [0.0; HYDRA_ACTION_SPACE],
            |_, _, step, _, _| {
                let legal_f32 = step
                    .legal_mask
                    .map(|is_legal| if is_legal { 1.0 } else { 0.0 });
                if !crate::training::exit::compatible_discard_state(&legal_f32) {
                    return None;
                }
                let legal_actions = step
                    .legal_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &is_legal)| {
                        (is_legal && idx <= hydra_core::action::DISCARD_END as usize).then_some(idx)
                    })
                    .collect::<Vec<_>>();
                if legal_actions.len() < 2 {
                    return None;
                }
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                target[legal_actions[0]] = 0.5;
                target[legal_actions[1]] = 0.5;
                mask[legal_actions[0]] = 1.0;
                mask[legal_actions[1]] = 1.0;
                TrajectoryExitLabel::from_slices(&target, &mask).map(|exit| {
                    crate::training::live_exit::TrajectorySearchLabels {
                        exit: Some(exit),
                        delta_q: None,
                    }
                })
            },
        );

        assert!(
            trajectory
                .steps
                .iter()
                .any(|step| step.exit_label.is_some())
        );
        assert!(trajectory.validate().is_ok());
    }
}
