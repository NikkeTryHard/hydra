//! Gym-like environment wrapper for Python.
//!
//! Provides `HydraEnv` (single) and `HydraVectorEnv` (batched) wrappers
//! around riichienv-core's game state with Hydra's 85x34 observation encoding.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArrayMethods;

use hydra_core::action::{
    hydra_to_riichienv, riichienv_to_hydra, ActionPhase, GameContext,
    HydraAction, HYDRA_ACTION_SPACE,
};
use hydra_core::bridge::encode_observation;
use hydra_core::encoder::ObservationEncoder;
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

/// Player seat controlled by the Python agent.
const AGENT_SEAT: u8 = 0;

/// Maximum auto-advance steps before bailing (prevents infinite loops).
const MAX_AUTO_STEPS: u32 = 50_000;

/// Return type for `HydraEnv::step`.
type StepResult<'py> = (Bound<'py, numpy::PyArray1<f32>>, f32, bool, Py<PyAny>);

/// Return type for `HydraVectorEnv::step`.
type VecStepResult<'py> = (Bound<'py, numpy::PyArray2<f32>>, Vec<f32>, Vec<bool>, Vec<Py<PyAny>>);

/// Gym-like single-environment wrapper for Riichi Mahjong.
#[pyclass]
pub struct HydraEnv {
    state: GameState,
    encoder: ObservationEncoder,
    safety: SafetyInfo,
    game_mode: u8,
    seed: Option<u64>,
}

impl HydraEnv {
    /// Auto-advance through non-agent turns until agent needs to act or game ends.
    fn advance_to_agent(&mut self) {
        for _ in 0..MAX_AUTO_STEPS {
            if self.state.is_done {
                return;
            }
            if self.state.needs_initialize_next_round {
                self.state.step(&HashMap::new());
                self.safety.reset();
                continue;
            }
            match self.state.phase {
                Phase::WaitAct => {
                    if self.state.current_player == AGENT_SEAT {
                        return;
                    }
                    self.auto_play_current();
                }
                Phase::WaitResponse => {
                    let active = self.state.active_players.clone();
                    if active.contains(&AGENT_SEAT) {
                        return;
                    }
                    self.auto_pass_all(&active);
                }
            }
        }
    }
}

impl HydraEnv {
    /// Auto-play the current player (non-agent) with their first legal action.
    fn auto_play_current(&mut self) {
        let pid = self.state.current_player;
        let obs = self.state.get_observation(pid);
        let legal = obs.legal_actions_method();
        if let Some(action) = legal.into_iter().next() {
            let mut actions = HashMap::new();
            actions.insert(pid, action);
            self.state.step(&actions);
        }
    }

    /// Auto-pass all active players during WaitResponse.
    fn auto_pass_all(&mut self, active: &[u8]) {
        let mut actions = HashMap::new();
        let pass = Action::new(ActionType::Pass, None, vec![], None);
        for &pid in active {
            actions.insert(pid, pass.clone());
        }
        self.state.step(&actions);
    }
}

#[pymethods]
impl HydraEnv {
    #[new]
    #[pyo3(signature = (game_mode=0, seed=None))]
    fn new(game_mode: u8, seed: Option<u64>) -> Self {
        let rule = GameRule::default_tenhou();
        let state = GameState::new(game_mode, true, seed, 0, rule);
        Self {
            state,
            encoder: ObservationEncoder::new(),
            safety: SafetyInfo::new(),
            game_mode,
            seed,
        }
    }

    /// Reset the environment. Returns initial observation as numpy array.
    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        let rule = GameRule::default_tenhou();
        self.state = GameState::new(self.game_mode, true, self.seed, 0, rule);
        self.safety.reset();
        self.encoder.clear();
        self.advance_to_agent();
        self.get_observation(py)
    }

    /// Get current observation as numpy array [2890].
    fn get_observation<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        let obs = self.state.get_observation(AGENT_SEAT);
        let drawn = self.state.drawn_tile.map(|t| t / 4);
        let tensor = encode_observation(&mut self.encoder, &obs, &self.safety, drawn);
        numpy::PyArray1::from_slice(py, &tensor)
    }

    /// Get legal actions as a list of Hydra action ids (0-45).
    fn legal_actions(&mut self) -> Vec<u8> {
        let obs = self.state.get_observation(AGENT_SEAT);
        let legal = obs.legal_actions_method();
        let mut result = Vec::new();
        for action in &legal {
            if let Ok(hydra) = riichienv_to_hydra(action) {
                result.push(hydra.id());
            }
        }
        result
    }

    /// Check if game is done.
    #[getter]
    fn done(&self) -> bool {
        self.state.is_done
    }

    /// Get current scores for all 4 players.
    #[getter]
    fn scores(&self) -> [i32; 4] {
        std::array::from_fn(|i| self.state.players[i].score)
    }

    /// Step the environment with a Hydra action (0-45).
    /// Returns (observation, reward, done, info_dict).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: u8,
    ) -> PyResult<StepResult<'py>> {
        let hydra = HydraAction::new(action).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                format!("invalid action {action}, must be 0-{}", HYDRA_ACTION_SPACE - 1),
            )
        })?;
        let ctx = self.build_context();
        let riichienv_action = hydra_to_riichienv(hydra, &ctx).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;
        self.submit_agent_action(riichienv_action);
        self.advance_to_agent();
        let done = self.state.is_done;
        let reward = if done {
            let score = self.state.players[AGENT_SEAT as usize].score;
            (score as f32 - 25000.0) / 25000.0
        } else {
            0.0
        };
        let obs = self.get_observation(py);
        let info = PyDict::new(py);
        Ok((obs, reward, done, info.into_any().unbind()))
    }
}

impl HydraEnv {
    /// Build a GameContext for action conversion.
    fn build_context(&mut self) -> GameContext {
        let obs = self.state.get_observation(AGENT_SEAT);
        let phase = match self.state.phase {
            Phase::WaitAct => ActionPhase::Normal,
            Phase::WaitResponse => ActionPhase::RiichiSelect,
        };
        let hand: Vec<u8> = obs.hands[AGENT_SEAT as usize]
            .iter()
            .map(|&t| t as u8)
            .collect();
        let last_discard = obs.last_discard.map(|t| t as u8);
        GameContext { last_discard, phase, hand }
    }
}

impl HydraEnv {
    /// Submit the agent's action and auto-pass other players.
    fn submit_agent_action(&mut self, action: Action) {
        let mut actions = HashMap::new();
        match self.state.phase {
            Phase::WaitAct => {
                actions.insert(AGENT_SEAT, action);
            }
            Phase::WaitResponse => {
                let active = self.state.active_players.clone();
                let pass = Action::new(ActionType::Pass, None, vec![], None);
                for &pid in &active {
                    if pid == AGENT_SEAT {
                        actions.insert(pid, action.clone());
                    } else {
                        actions.insert(pid, pass.clone());
                    }
                }
            }
        }
        self.state.step(&actions);
    }
}

/// Batched environment for vectorized training.
#[pyclass]
pub struct HydraVectorEnv {
    envs: Vec<HydraEnv>,
}

#[pymethods]
impl HydraVectorEnv {
    #[new]
    #[pyo3(signature = (num_envs, game_mode=0, base_seed=None))]
    fn new(num_envs: usize, game_mode: u8, base_seed: Option<u64>) -> Self {
        let envs = (0..num_envs)
            .map(|i| {
                let seed = base_seed.map(|s| s.wrapping_add(i as u64));
                HydraEnv::new(game_mode, seed)
            })
            .collect();
        Self { envs }
    }

    /// Reset all environments. Returns stacked observations [num_envs, 2890].
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let rows: Vec<Vec<f32>> = self.envs
            .iter_mut()
            .map(|env| {
                let obs = env.reset(py);
                let ro = obs.readonly();
                ro.as_slice().expect("contiguous").to_vec()
            })
            .collect();
        numpy::PyArray2::from_vec2(py, &rows).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Step all environments. Returns (obs, rewards, dones, infos).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> PyResult<VecStepResult<'py>> {
        if actions.len() != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "expected {} actions, got {}",
                    self.envs.len(),
                    actions.len()
                ),
            ));
        }
        let mut obs_rows = Vec::with_capacity(self.envs.len());
        let mut rewards = Vec::with_capacity(self.envs.len());
        let mut dones = Vec::with_capacity(self.envs.len());
        let mut infos = Vec::with_capacity(self.envs.len());
        for (env, &action) in self.envs.iter_mut().zip(actions.iter()) {
            let (obs, r, d, info) = env.step(py, action)?;
            let ro = obs.readonly();
            obs_rows.push(ro.as_slice().expect("contiguous").to_vec());
            rewards.push(r);
            dones.push(d);
            infos.push(info);
        }
        let obs2d = numpy::PyArray2::from_vec2(py, &obs_rows).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?;
        Ok((obs2d, rewards, dones, infos))
    }

    /// Get legal actions for all environments.
    fn legal_actions(&mut self) -> Vec<Vec<u8>> {
        self.envs.iter_mut().map(|env| env.legal_actions()).collect()
    }

    /// Number of environments.
    #[getter]
    fn num_envs(&self) -> usize {
        self.envs.len()
    }
}
