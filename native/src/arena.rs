//! Python Arena facade: schedule creation, batched rows, and atomic steps.
//!
//! The class owns every scheduled match for its lifetime. `batch` stages one
//! row per active game; `step` consumes exactly those rows. Errors that would
//! leave a game in an unknown state poison the arena permanently.

mod batch;
mod buffers;
mod game;
mod step;
mod telemetry;

use std::collections::HashSet;
use std::time::Instant;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use self::batch::PreparedBatch;
use self::game::Game;
use self::step::StagedRow;

#[pyclass(module = "hydra3._arena")]
pub struct Arena {
    games: Vec<Game>,
    pool: rayon::ThreadPool,
    staged: Vec<StagedRow>,
    failed: Option<String>,
    telemetry: Option<telemetry::Telemetry>,
}

#[pymethods]
impl Arena {
    #[new]
    #[pyo3(signature = (seeds, workers = 4))]
    fn new(py: Python<'_>, seeds: Vec<u64>, workers: usize) -> PyResult<Self> {
        if workers == 0 {
            return Err(PyValueError::new_err("workers must be at least 1"));
        }
        let mut seen = HashSet::with_capacity(seeds.len());
        for seed in &seeds {
            if !seen.insert(*seed) {
                return Err(PyValueError::new_err(format!(
                    "duplicate arena seed {seed}"
                )));
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|error| PyRuntimeError::new_err(format!("rayon pool failed: {error}")))?;
        let opened: Vec<Result<Game, String>> = if workers == 1 {
            py.detach(|| seeds.into_iter().map(Game::open).collect::<Vec<_>>())
        } else {
            py.detach(|| pool.install(|| seeds.into_par_iter().map(Game::open).collect::<Vec<_>>()))
        };
        let mut games = Vec::with_capacity(opened.len());
        for game in opened {
            games.push(game.map_err(PyRuntimeError::new_err)?);
        }
        Ok(Self {
            games,
            pool,
            staged: Vec::new(),
            failed: None,
            telemetry: None,
        })
    }

    /// Start a fresh telemetry window without changing arena state.
    fn enable_telemetry(&mut self) {
        self.telemetry = Some(telemetry::Telemetry::default());
    }

    /// Return counters accumulated since telemetry was enabled.
    fn telemetry(&self) -> Option<[(&'static str, u64); 7]> {
        self.telemetry.as_ref().map(telemetry::Telemetry::snapshot)
    }

    /// Stage one row per active game and return four little-endian byte arrays.
    fn batch(&mut self, py: Python<'_>) -> PyResult<buffers::WireBuffers> {
        self.ensure_live()?;
        let start = self.telemetry.as_ref().map(|_| Instant::now());
        let prepared = match py.detach(|| batch::prepare(self)) {
            Ok(prepared) => prepared,
            Err(message) => return Err(self.fail_closed(message)),
        };
        let PreparedBatch { rows, staged } = prepared;
        self.staged = staged;
        if let (Some(stats), Some(start)) = (&mut self.telemetry, start) {
            stats.batch_calls += 1;
            stats.rows += rows.len() as u64;
            stats.prepare_ns += telemetry::elapsed(start);
        }
        let start = self.telemetry.as_ref().map(|_| Instant::now());
        let packed = buffers::pack(py, &rows);
        if let (Some(stats), Some(start)) = (&mut self.telemetry, start) {
            stats.pack_ns += telemetry::elapsed(start);
        }
        packed.map_err(|error| self.fail_closed(error.to_string()))
    }

    /// Apply one canonical action per staged row after validating all of them.
    /// `ids` must be a C-contiguous int32 buffer; it is copied once under the
    /// GIL and never borrowed while native work runs detached.
    fn step(&mut self, py: Python<'_>, ids: PyBuffer<i32>) -> PyResult<()> {
        self.ensure_live()?;
        if ids.dimensions() != 1 {
            return Err(PyValueError::new_err("step ids must be one-dimensional"));
        }
        if !ids.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "step ids must be a contiguous int32 buffer",
            ));
        }
        if ids.item_count() != self.staged.len() {
            return Err(PyValueError::new_err(format!(
                "step expected {} ids but received {}",
                self.staged.len(),
                ids.item_count()
            )));
        }
        let start = self.telemetry.as_ref().map(|_| Instant::now());
        let values = ids.to_vec(py)?;
        drop(ids);
        let chosen = match step::validate(&self.staged, &values) {
            Ok(chosen) => chosen,
            Err(message) => return Err(PyValueError::new_err(message)),
        };
        drop(values);
        let staged = std::mem::take(&mut self.staged);
        if let (Some(stats), Some(start)) = (&mut self.telemetry, start) {
            stats.validate_ns += telemetry::elapsed(start);
        }
        let start = self.telemetry.as_ref().map(|_| Instant::now());
        let applied = {
            let games = &mut self.games;
            let pool = &self.pool;
            py.detach(|| step::apply(&staged, &chosen, games, pool))
        };
        if let (Some(stats), Some(start)) = (&mut self.telemetry, start) {
            stats.step_calls += 1;
            stats.apply_ns += telemetry::elapsed(start);
        }
        applied.map_err(|message| self.fail_closed(message))
    }

    /// Terminal matches in slot order: slot, final scores, resolved ranks.
    fn outcomes(&self) -> PyResult<Vec<(i32, Vec<i32>, Vec<i32>)>> {
        self.ensure_live()?;
        let mut rows = Vec::new();
        for (slot, game) in self.games.iter().enumerate() {
            match game.outcome_row(slot) {
                Ok(Some(row)) => rows.push(row),
                Ok(None) => {}
                Err(message) => return Err(PyRuntimeError::new_err(message)),
            }
        }
        Ok(rows)
    }
}

impl Arena {
    fn ensure_live(&self) -> PyResult<()> {
        self.failed.as_ref().map_or(Ok(()), |message| {
            Err(PyRuntimeError::new_err(format!(
                "arena is failed closed: {message}"
            )))
        })
    }

    fn fail_closed(&mut self, message: String) -> PyErr {
        self.failed = Some(message.clone());
        PyRuntimeError::new_err(message)
    }
}
