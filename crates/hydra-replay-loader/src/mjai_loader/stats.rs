use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayMaterializationStats {
    pub decompress_ns: u128,
    pub json_parse_ns: u128,
    pub replay_update_ns: u128,
    pub observation_encode_ns: u128,
    pub mask_build_ns: u128,
    pub target_synthesis_ns: u128,
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
                .saturating_add(self.target_synthesis_ns)
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
        self.target_synthesis_ns = self
            .target_synthesis_ns
            .saturating_add(other.target_synthesis_ns);
        self.event_count = self.event_count.saturating_add(other.event_count);
        self.decision_count = self.decision_count.saturating_add(other.decision_count);
    }
}

#[derive(Default)]
pub(super) struct ReplayProfileStats {
    pub(super) parse_events_ns: u128,
    pub(super) precompute_ns: u128,
    pub(super) prepare_decisions_ns: u128,
    pub(super) implicit_pass_ns: u128,
    pub(super) replay_observation_ns: u128,
    pub(super) legal_mask_build_ns: u128,
    pub(super) encode_observation_ns: u128,
    pub(super) legal_mask_convert_ns: u128,
    pub(super) opponent_targets_ns: u128,
    pub(super) exact_waits_ns: u128,
    pub(super) safety_residual_ns: u128,
    pub(super) belief_targets_ns: u128,
    pub(super) sidecar_lookup_ns: u128,
    pub(super) sample_push_ns: u128,
    pub(super) update_safety_ns: u128,
    pub(super) apply_event_ns: u128,
    pub(super) event_count: usize,
    pub(super) decision_count: usize,
}

static REPLAY_PROFILE_PRINTED: AtomicBool = AtomicBool::new(false);
pub(super) static REPLAY_IMPLICIT_PASS_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static REPLAY_OBSERVATION_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static REPLAY_LEGAL_MASK_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static REPLAY_ENCODE_OBS_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_MATERIALIZATION_TOTALS: Mutex<ReplayMaterializationStats> =
    Mutex::new(ReplayMaterializationStats {
        decompress_ns: 0,
        json_parse_ns: 0,
        replay_update_ns: 0,
        observation_encode_ns: 0,
        mask_build_ns: 0,
        target_synthesis_ns: 0,
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

pub(super) fn record_replay_materialization_stats(stats: ReplayMaterializationStats) {
    replay_materialization_totals().merge_assign(stats);
}
pub(super) fn maybe_print_replay_profile(stats: &ReplayProfileStats) {
    if std::env::var_os("HYDRA_REPLAY_PROFILE").is_none() {
        return;
    }
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
