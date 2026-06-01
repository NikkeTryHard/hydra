use std::cmp::Ordering;

use super::cache::{PonderCache, PonderResult, TrustLevel};

pub struct PonderTask {
    pub info_state_hash: u64,
    pub priority_score: f32,
    pub game_state_snapshot: GameStateSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameStateSnapshot {
    pub info_state_hash: u64,
    pub top2_policy_gap: f32,
    pub risk_score: f32,
    pub particle_ess: f32,
}

impl Eq for PonderTask {}

impl PartialEq for PonderTask {
    fn eq(&self, other: &Self) -> bool {
        self.info_state_hash == other.info_state_hash
            && self.priority_score.to_bits() == other.priority_score.to_bits()
    }
}

impl Ord for PonderTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score
            .partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.info_state_hash.cmp(&other.info_state_hash))
    }
}

impl PartialOrd for PonderTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn compute_ponder_priority(top2_gap: f32, risk_score: f32, particle_ess: f32) -> f32 {
    let gap_term = (0.1 - top2_gap).max(0.0) * 10.0;
    let risk_term = risk_score.max(0.0);
    let ess_term = (1.0 - particle_ess).max(0.0);
    gap_term + risk_term + ess_term
}

pub struct PonderManager {
    pub cache: PonderCache,
    pub priority_queue: std::collections::BinaryHeap<PonderTask>,
    pub worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl PonderManager {
    pub fn new() -> Self {
        Self {
            cache: PonderCache::new(),
            priority_queue: std::collections::BinaryHeap::new(),
            worker_handle: None,
        }
    }

    pub fn enqueue(&mut self, task: PonderTask) {
        self.priority_queue.push(task);
    }

    pub fn enqueue_snapshot(&mut self, snapshot: GameStateSnapshot) {
        let priority_score = compute_ponder_priority(
            snapshot.top2_policy_gap,
            snapshot.risk_score,
            snapshot.particle_ess,
        );
        self.enqueue(PonderTask {
            info_state_hash: snapshot.info_state_hash,
            priority_score,
            game_state_snapshot: snapshot,
        });
    }

    pub fn pop_task(&mut self) -> Option<PonderTask> {
        self.priority_queue.pop()
    }

    pub fn cache_result(&self, hash: u64, result: PonderResult) {
        self.cache.insert(hash, result);
    }

    pub fn lookup(&self, hash: u64) -> Option<PonderResult> {
        self.cache.get(hash)
    }

    pub fn lookup_trusted(&self, hash: u64, min_trust: TrustLevel) -> Option<PonderResult> {
        self.cache.get_trusted(hash, min_trust)
    }

    /// Invalidates all cached entries (e.g. on checkpoint change).
    pub fn invalidate_cache(&self) -> u64 {
        self.cache.invalidate()
    }

    pub fn queue_len(&self) -> usize {
        self.priority_queue.len()
    }

    pub fn has_worker(&self) -> bool {
        self.worker_handle.is_some()
    }
}

impl Default for PonderManager {
    fn default() -> Self {
        Self::new()
    }
}
