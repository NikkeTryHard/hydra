use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use hydra_core::action::{
    build_legal_mask, hydra_to_riichienv, riichienv_to_hydra, ActionPhase, GameContext,
    HydraAction, HYDRA_ACTION_SPACE,
};
use hydra_core::afbs::{AfbsTree, NodeIdx};
use hydra_core::arena::{
    sample_action_with_temperature, Trajectory, TrajectoryDeltaQLabel, TrajectoryExitLabel,
    TrajectoryStep,
};
use hydra_core::bridge::encode_observation;
use hydra_core::encoder::{ObservationEncoder, NUM_CHANNELS, OBS_SIZE};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

use crate::config::{GAE_GAMMA, GAE_LAMBDA};
use crate::model::HydraModel;
use crate::training::exit::{
    build_delta_q_from_afbs_tree, build_exit_from_afbs_tree, collate_delta_q_targets,
    collate_exit_targets, compatible_discard_state, is_hard_state,
};
use crate::training::gae::{compute_per_player_gae, normalize_advantages, GaeConfig};
use crate::training::live_exit::{
    base_pi_from_logits, budget_from_legal_count, legal_discard_actions, make_live_exit_fn,
    seed_root_children_all_legal, ExitSearchAdapter, LiveExitConfig, RootDecisionContext,
    SelfPlayExitAdapter, TrajectorySearchLabels,
};
use crate::training::losses::HydraTargets;
use crate::training::rl::RlBatch;

const DEFAULT_GAME_MODE: u8 = 0;
const MAX_SELF_PLAY_STEPS: u32 = 50_000;
const SCORE_BINS: usize = 64;
const GRP_CLASSES: usize = 24;
const NUM_OPPONENTS: usize = 3;
const NUM_TILES: usize = 34;

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

#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub policy_logits: [f32; HYDRA_ACTION_SPACE],
    pub pi_old: [f32; HYDRA_ACTION_SPACE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub player_id: u8,
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

pub fn trajectories_to_rl_batch<B: Backend>(
    trajectories: &[Trajectory],
    values: &[Vec<f32>],
    gae_config: &GaeConfig,
    device: &B::Device,
) -> RlBatch<B> {
    let total_steps: usize = trajectories
        .iter()
        .map(|trajectory| trajectory.steps.len())
        .sum();

    let mut obs_flat = Vec::with_capacity(total_steps * OBS_SIZE);
    let mut actions = Vec::with_capacity(total_steps);
    let mut pi_old = Vec::with_capacity(total_steps);
    let mut advantages = Vec::with_capacity(total_steps);
    let mut legal_mask = Vec::with_capacity(total_steps * HYDRA_ACTION_SPACE);
    let mut policy_target = vec![0.0f32; total_steps * HYDRA_ACTION_SPACE];
    let mut value_target = Vec::with_capacity(total_steps);
    let mut grp_target = vec![0.0f32; total_steps * GRP_CLASSES];
    let tenpai_target = vec![0.0f32; total_steps * NUM_OPPONENTS];
    let danger_target = vec![0.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let danger_mask = vec![1.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let mut opp_next_target = vec![0.0f32; total_steps * NUM_OPPONENTS * NUM_TILES];
    let mut score_pdf_target = vec![0.0f32; total_steps * SCORE_BINS];
    let mut score_cdf_target = vec![0.0f32; total_steps * SCORE_BINS];
    let base_logits = vec![0.0f32; total_steps * HYDRA_ACTION_SPACE];
    let mut exit_samples = Vec::with_capacity(total_steps);
    let mut delta_q_samples = Vec::with_capacity(total_steps);

    let mut global_step = 0usize;
    for (trajectory_idx, trajectory) in trajectories.iter().enumerate() {
        let trajectory_values = values.get(trajectory_idx).map_or(&[][..], Vec::as_slice);
        let trajectory_advantages =
            compute_trajectory_advantages(trajectory, trajectory_values, gae_config);

        for (step_idx, step) in trajectory.steps.iter().enumerate() {
            obs_flat.extend_from_slice(&step.obs);
            actions.push(step.action as i32);
            pi_old.push(step.pi_old[step.action as usize]);
            advantages.push(trajectory_advantages[step_idx]);

            for action_idx in 0..HYDRA_ACTION_SPACE {
                legal_mask.push(if step.legal_mask[action_idx] {
                    1.0
                } else {
                    0.0
                });
            }
            exit_samples.push(step.exit_label.map(TrajectoryExitLabel::to_vec_pair));
            delta_q_samples.push(step.delta_q_label.map(TrajectoryDeltaQLabel::to_vec_pair));

            policy_target[global_step * HYDRA_ACTION_SPACE + step.action as usize] = 1.0;
            value_target.push(step.reward);

            let placement_class = trajectory.placement_for(step.player_id) as usize;
            if placement_class < GRP_CLASSES {
                grp_target[global_step * GRP_CLASSES + placement_class] = 1.0;
            }

            for opponent in 0..NUM_OPPONENTS {
                opp_next_target[global_step * NUM_OPPONENTS * NUM_TILES + opponent * NUM_TILES] =
                    1.0;
            }

            let score_bin = score_to_bin(trajectory.final_scores[step.player_id as usize]);
            score_pdf_target[global_step * SCORE_BINS + score_bin] = 1.0;
            for bin in score_bin..SCORE_BINS {
                score_cdf_target[global_step * SCORE_BINS + bin] = 1.0;
            }

            global_step += 1;
        }
    }

    normalize_advantages(&mut advantages);
    let (exit_target, exit_mask) = collate_exit_targets::<B>(&exit_samples, device);
    let (delta_q_target, delta_q_mask) = collate_delta_q_targets::<B>(&delta_q_samples, device);

    RlBatch {
        obs: Tensor::<B, 1>::from_floats(obs_flat.as_slice(), device).reshape([
            total_steps,
            NUM_CHANNELS,
            NUM_TILES,
        ]),
        actions: Tensor::<B, 1, Int>::from_ints(actions.as_slice(), device),
        pi_old: Tensor::<B, 1>::from_floats(pi_old.as_slice(), device),
        advantages: Tensor::<B, 1>::from_floats(advantages.as_slice(), device),
        base_logits: Tensor::<B, 1>::from_floats(base_logits.as_slice(), device)
            .reshape([total_steps, HYDRA_ACTION_SPACE]),
        targets: HydraTargets {
            policy_target: Tensor::<B, 1>::from_floats(policy_target.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            legal_mask: Tensor::<B, 1>::from_floats(legal_mask.as_slice(), device)
                .reshape([total_steps, HYDRA_ACTION_SPACE]),
            value_target: Tensor::<B, 1>::from_floats(value_target.as_slice(), device),
            grp_target: Tensor::<B, 1>::from_floats(grp_target.as_slice(), device)
                .reshape([total_steps, GRP_CLASSES]),
            tenpai_target: Tensor::<B, 1>::from_floats(tenpai_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS]),
            danger_target: Tensor::<B, 1>::from_floats(danger_target.as_slice(), device).reshape([
                total_steps,
                NUM_OPPONENTS,
                NUM_TILES,
            ]),
            danger_mask: Tensor::<B, 1>::from_floats(danger_mask.as_slice(), device).reshape([
                total_steps,
                NUM_OPPONENTS,
                NUM_TILES,
            ]),
            opp_next_target: Tensor::<B, 1>::from_floats(opp_next_target.as_slice(), device)
                .reshape([total_steps, NUM_OPPONENTS, NUM_TILES]),
            score_pdf_target: Tensor::<B, 1>::from_floats(score_pdf_target.as_slice(), device)
                .reshape([total_steps, SCORE_BINS]),
            score_cdf_target: Tensor::<B, 1>::from_floats(score_cdf_target.as_slice(), device)
                .reshape([total_steps, SCORE_BINS]),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target,
            delta_q_mask,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
        },
        exit_target,
        exit_mask,
    }
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
#[derive(Clone)]
struct PendingPolicyRequest {
    pid: u8,
    obs: Observation,
    obs_encoded: [f32; OBS_SIZE],
    drawn_tile_before_action: Option<u8>,
    turn: u32,
}

struct ExitChildRequest {
    child_idx: NodeIdx,
    obs: [f32; OBS_SIZE],
}

struct PendingExitStep {
    step_record: StepRecord,
    turn: u32,
    tree: AfbsTree,
    root: NodeIdx,
    base_pi: [f32; HYDRA_ACTION_SPACE],
    legal_f32: [f32; HYDRA_ACTION_SPACE],
    budget: u32,
    child_offset: usize,
    child_count: usize,
    output_index: usize,
}

struct ExitSearchState {
    steps: Vec<PendingExitStep>,
    child_requests: Vec<ExitChildRequest>,
}

impl ExitSearchState {
    fn new() -> Self {
        Self {
            steps: Vec::new(),
            child_requests: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

struct PreparedExitSearch {
    step: PendingExitStep,
    child_requests: Vec<(NodeIdx, [f32; OBS_SIZE])>,
}

struct PendingTurnState {
    chosen_actions: [Option<Action>; 4],
    players: Vec<u8>,
    next_index: usize,
    turn: u32,
    pending_steps: Vec<Option<TrajectoryStep>>,
    pending_values: Vec<f32>,
}

impl PendingTurnState {
    fn new(players: Vec<u8>, turn: u32) -> Self {
        let pending_steps = Vec::with_capacity(players.len());
        Self {
            chosen_actions: [None; 4],
            players,
            next_index: 0,
            turn,
            pending_steps,
            pending_values: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct GameAdvance {
    needs_policy: bool,
}

struct CooperativeGameRunner {
    state: GameState,
    selector: NnActionSelector,
    trajectory: Trajectory,
    legal_buf: Vec<Action>,
    total_steps: u32,
    done: bool,
    pending_policy_obs: Option<PendingPolicyRequest>,
    pending_exit_search: Option<ExitSearchState>,
    turn_state: Option<PendingTurnState>,
    exit_adapter: SelfPlayExitAdapter,
    live_exit_cfg: LiveExitConfig,
    step_values: Vec<f32>,
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
            done: false,
            pending_policy_obs: None,
            pending_exit_search: None,
            turn_state: None,
            exit_adapter: SelfPlayExitAdapter::new(),
            live_exit_cfg,
            step_values: Vec::new(),
        }
    }

    fn is_finished(&self) -> bool {
        self.done
    }

    fn into_trajectory_and_values(mut self) -> (Trajectory, Vec<f32>) {
        if !self.done {
            self.finalize();
        }
        (self.trajectory, self.step_values)
    }

    fn advance_until_inference_needed(&mut self) -> GameAdvance {
        loop {
            if self.done {
                return GameAdvance::default();
            }

            if self.pending_policy_obs.is_some() {
                return GameAdvance { needs_policy: true };
            }

            if self.state.is_done || self.total_steps >= MAX_SELF_PLAY_STEPS {
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

            let obs = self.state.get_observation(pid);
            if obs.legal_actions_ref().is_empty() {
                self.turn_state
                    .as_mut()
                    .expect("pending turn state")
                    .next_index += 1;
                continue;
            }

            let drawn_tile = self.state.drawn_tile.map(|tile| tile / 4);
            let obs_encoded = self.selector.encode_observation(&obs, pid, drawn_tile);
            self.pending_policy_obs = Some(PendingPolicyRequest {
                pid,
                obs,
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

            if let Some(prepared) =
                self.prepare_exit_search(&pending.obs, &step_record, pending.turn)
            {
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
                let mut value_by_child = std::collections::HashMap::<NodeIdx, f32>::new();
                let mut valid = true;

                for (child, &value) in child_slice.iter().zip(value_slice.iter()) {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    value_by_child.insert(child.child_idx, value);
                }

                if valid {
                    exit_step.tree.run_search_iterations(
                        exit_step.root,
                        exit_step.budget,
                        &|child_idx| value_by_child.get(&child_idx).copied().unwrap_or(0.0),
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
        obs: &Observation,
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
        let mut child_requests = Vec::with_capacity(tree.nodes[root as usize].children.len());
        for &(action, child_idx) in &tree.nodes[root as usize].children.clone() {
            let child_obs = self.exit_adapter.child_public_obs_after_discard(
                &self.state,
                obs,
                ctx.player_id,
                action,
                &player_safety,
            )?;
            child_requests.push((child_idx, child_obs));
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
            child_requests,
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

fn batch_policy_value_cpu<B: Backend>(
    model: &HydraModel<B>,
    observations: &[[f32; OBS_SIZE]],
    device: &B::Device,
) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
    model.batch_policy_value_cpu(observations, device)
}

fn batched_trajectory_values<B: Backend>(
    trajectories: &[Trajectory],
    model: &HydraModel<B>,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    let total_steps: usize = trajectories
        .iter()
        .map(|trajectory| trajectory.steps.len())
        .sum();
    if total_steps == 0 {
        return trajectories.iter().map(|_| Vec::new()).collect();
    }

    let mut observations = Vec::with_capacity(total_steps);
    let mut step_counts = Vec::with_capacity(trajectories.len());
    for trajectory in trajectories {
        step_counts.push(trajectory.steps.len());
        for step in &trajectory.steps {
            observations.push(step.obs);
        }
    }

    let outputs = batch_policy_value_cpu(model, &observations, device);
    let mut values = Vec::with_capacity(trajectories.len());
    let mut offset = 0usize;
    for step_count in step_counts {
        let mut per_trajectory = Vec::with_capacity(step_count);
        for (_, value) in &outputs[offset..offset + step_count] {
            per_trajectory.push(*value);
        }
        values.push(per_trajectory);
        offset += step_count;
    }
    values
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

        let infer_fn = |obs: &[f32; OBS_SIZE]| -> [f32; HYDRA_ACTION_SPACE] {
            let (logits, _) = model.policy_value_cpu(obs, device);
            logits
        };

        let exit_cfg = live_exit_cfg.clone();
        let exit_fn = make_live_exit_fn(exit_cfg, |obs: &[f32; OBS_SIZE]| {
            model.policy_value_cpu(obs, device)
        });

        let trajectory =
            run_self_play_game_with_exit_labels(seed, temperature, game_rng, infer_fn, exit_fn);

        let step_values: Vec<f32> = trajectory
            .steps
            .iter()
            .map(|step| {
                let (_, value) = model.policy_value_cpu(&step.obs, device);
                value
            })
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
    let mut games: Vec<CooperativeGameRunner> = game_seeds
        .iter()
        .enumerate()
        .map(|(idx, &seed)| {
            CooperativeGameRunner::new(
                seed,
                temperature,
                rng_seed.wrapping_add(idx as u64),
                live_exit_cfg.clone(),
            )
        })
        .collect();

    let n = games.len();
    let mut batch_game_indices: Vec<usize> = Vec::with_capacity(n);
    let mut batch_observations: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n);
    let mut exit_game_indices: Vec<usize> = Vec::with_capacity(n);
    let mut exit_counts: Vec<usize> = Vec::with_capacity(n);
    let mut exit_observations: Vec<[f32; OBS_SIZE]> = Vec::with_capacity(n * 14);
    let mut flat_buf: Vec<f32> = Vec::with_capacity(n * OBS_SIZE);

    while games.iter().any(|game| !game.is_finished()) {
        loop {
            batch_game_indices.clear();
            batch_observations.clear();

            for (game_idx, game) in games.iter_mut().enumerate() {
                let advance = game.advance_until_inference_needed();
                if advance.needs_policy {
                    if let Some(obs) = game.pending_policy_obs() {
                        batch_game_indices.push(game_idx);
                        batch_observations.push(obs);
                    }
                }
            }

            if batch_game_indices.is_empty() {
                break;
            }

            let batch_outputs =
                model.batch_policy_value_cpu_reuse(&batch_observations, device, &mut flat_buf);
            for (game_idx, (policy_logits, value)) in
                batch_game_indices.drain(..).zip(batch_outputs)
            {
                games[game_idx].provide_policy_result(policy_logits, value);
            }
        }

        exit_game_indices.clear();
        exit_counts.clear();
        exit_observations.clear();
        for (game_idx, game) in games.iter().enumerate() {
            let child_count = game.pending_exit_child_count();
            if child_count > 0 {
                exit_game_indices.push(game_idx);
                exit_counts.push(child_count);
                game.append_pending_exit_obs(&mut exit_observations);
            }
        }

        if exit_game_indices.is_empty() {
            continue;
        }

        let exit_outputs =
            model.batch_policy_value_cpu_reuse(&exit_observations, device, &mut flat_buf);
        let mut offset = 0usize;
        for (game_idx, child_count) in exit_game_indices.drain(..).zip(exit_counts.drain(..)) {
            let child_values: Vec<f32> = exit_outputs[offset..offset + child_count]
                .iter()
                .map(|(_, value)| *value)
                .collect();
            games[game_idx].finalize_pending_exit_search(&child_values);
            offset += child_count;
        }
    }

    let mut trajectories = Vec::with_capacity(games.len());
    let mut all_values = Vec::with_capacity(games.len());
    for game in games {
        let (trajectory, values) = game.into_trajectory_and_values();
        trajectories.push(trajectory);
        all_values.push(values);
    }

    SelfPlayBatchSource {
        trajectories,
        values: all_values,
    }
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
    let obs = env.state.get_observation(pid);
    if obs.legal_actions_ref().is_empty() {
        return;
    }

    let drawn_tile = env.state.drawn_tile.map(|tile| tile / 4);
    let encoded = env.selector.encode_observation(&obs, pid, drawn_tile);
    let logits = (env.infer_fn)(&encoded);
    env.selector.set_logits(logits);

    env.state.get_legal_actions_into(pid, env.legal_buf);
    if env.legal_buf.is_empty() {
        return;
    }

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
        let search_labels =
            exit_label_fn(env.state, &obs, &step_record, player_safety, turn).unwrap_or_default();
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

fn finalize_rewards(trajectory: &mut Trajectory) {
    let mut steps_per_player = [0usize; 4];
    for step in &trajectory.steps {
        steps_per_player[step.player_id as usize] += 1;
    }

    for step in &mut trajectory.steps {
        let player = step.player_id as usize;
        let count = steps_per_player[player].max(1) as f32;
        step.reward = trajectory.final_scores[player] as f32 / 100_000.0 / count;
    }
}

fn compute_trajectory_advantages(
    trajectory: &Trajectory,
    values: &[f32],
    gae_config: &GaeConfig,
) -> Vec<f32> {
    let mut advantages = vec![0.0f32; trajectory.steps.len()];

    for player in 0..4u8 {
        let player_indices: Vec<usize> = trajectory
            .steps
            .iter()
            .enumerate()
            .filter_map(|(idx, step)| (step.player_id == player).then_some(idx))
            .collect();
        if player_indices.is_empty() {
            continue;
        }

        let mut player_rewards = Vec::with_capacity(player_indices.len());
        let mut player_values = Vec::with_capacity(player_indices.len() + 1);
        let dones = vec![false; player_indices.len() - 1]
            .into_iter()
            .chain(std::iter::once(true))
            .collect::<Vec<_>>();

        for &idx in &player_indices {
            let mut reward_row = [0.0f32; 4];
            reward_row[player as usize] = trajectory.steps[idx].reward;
            player_rewards.push(reward_row);

            let mut value_row = [0.0f32; 4];
            value_row[player as usize] = values.get(idx).copied().unwrap_or(0.0);
            player_values.push(value_row);
        }
        player_values.push([0.0; 4]);

        let player_advantages = compute_per_player_gae(
            &player_rewards,
            &player_values,
            &dones,
            gae_config.gamma,
            gae_config.lambda,
        );

        for (local_idx, &global_idx) in player_indices.iter().enumerate() {
            advantages[global_idx] = player_advantages[local_idx][player as usize];
        }
    }

    advantages
}

fn score_to_bin(score: i32) -> usize {
    let normalized = ((score as f32 / 1000.0) + 32.0).floor();
    normalized.clamp(0.0, (SCORE_BINS - 1) as f32) as usize
}

pub fn default_gae_config() -> GaeConfig {
    GaeConfig {
        gamma: GAE_GAMMA,
        lambda: GAE_LAMBDA,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::HydraModelConfig;
    use burn::backend::NdArray;

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
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<B>(&device);
        let seeds = [42u64, 43];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let source = generate_self_play_batch_source(&seeds, 1.0, 100, &model, &device, cfg);

        assert_eq!(source.trajectories.len(), 2);
        assert_eq!(source.values.len(), 2);
        for (traj, vals) in source.trajectories.iter().zip(source.values.iter()) {
            assert_eq!(traj.steps.len(), vals.len());
            assert!(traj.steps.iter().all(|s| s.exit_label.is_none()));
        }
    }

    #[test]
    fn test_batched_source_matches_serial() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<B>(&device);
        let seeds = [42u64, 43, 44];
        let cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        let serial =
            generate_self_play_batch_source(&seeds, 1.0, 100, &model, &device, cfg.clone());
        let batched = generate_self_play_batch_source_batched(
            &seeds, 1.0, 100, &model, &device, &device, cfg,
        );

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

    #[test]
    fn test_generate_self_play_rl_batch_produces_valid_batch() {
        type AB = burn::backend::Autodiff<B>;
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<AB>(&device);
        let seeds = [42u64];
        let cfg = LiveExitConfig::default();
        let gae = GaeConfig {
            gamma: GAE_GAMMA,
            lambda: GAE_LAMBDA,
        };

        let batch = generate_self_play_rl_batch(&seeds, 1.0, 100, &model, &device, &gae, cfg);
        let [steps, action_dim] = batch.targets.policy_target.dims();
        assert!(steps > 0);
        assert_eq!(action_dim, HYDRA_ACTION_SPACE);
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

        assert!(trajectory
            .steps
            .iter()
            .any(|step| step.exit_label.is_some()));
        assert!(trajectory.validate().is_ok());
    }
}
