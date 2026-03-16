//! Generates Stage A projected belief teacher targets.

use hydra_core::sinkhorn::MixtureSib;

pub const BELIEF_COMPONENTS: usize = 4;
pub const BELIEF_ZONES: usize = 4;
pub const BELIEF_TILES: usize = 34;
pub const BELIEF_FIELDS_SIZE: usize = BELIEF_COMPONENTS * BELIEF_ZONES * BELIEF_TILES;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageABeliefTarget {
    pub belief_fields: [f32; BELIEF_FIELDS_SIZE],
    pub mixture_weights: Option<[f32; BELIEF_COMPONENTS]>,
    pub trust: f32,
    pub ess: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StageABeliefAuditSummary {
    pub total: usize,
    pub emitted_belief: usize,
    pub emitted_mixture: usize,
    pub trust_sum: f32,
    pub ess_sum: f32,
    pub entropy_sum: f32,
}

impl StageABeliefAuditSummary {
    pub fn record(&mut self, target: Option<&StageABeliefTarget>) {
        self.total += 1;
        if let Some(target) = target {
            self.emitted_belief += 1;
            if target.mixture_weights.is_some() {
                self.emitted_mixture += 1;
            }
            self.trust_sum += target.trust;
            self.ess_sum += target.ess;
            self.entropy_sum += target.entropy;
        }
    }

    pub fn belief_coverage(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.emitted_belief as f32 / self.total as f32
        }
    }

    pub fn mixture_coverage(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.emitted_mixture as f32 / self.total as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageABeliefConfig {
    pub num_components: u8,
    pub trust_threshold: f32,
    pub mixture_entropy_threshold: f32,
}

impl Default for StageABeliefConfig {
    fn default() -> Self {
        Self {
            num_components: BELIEF_COMPONENTS as u8,
            trust_threshold: 0.55,
            mixture_entropy_threshold: -1.0,
        }
    }
}

pub fn build_uniform_kernel() -> [f64; BELIEF_TILES * BELIEF_ZONES] {
    [1.0; BELIEF_TILES * BELIEF_ZONES]
}

pub fn project_public_remaining_to_row_sums(
    remaining: &[f32; BELIEF_TILES],
) -> [f64; BELIEF_TILES] {
    let mut row_sums = [0.0f64; BELIEF_TILES];
    for (dst, &value) in row_sums.iter_mut().zip(remaining.iter()) {
        *dst = value.max(0.0) as f64;
    }
    row_sums
}

pub fn project_hidden_counts_to_col_sums(
    hidden_counts: &[usize; BELIEF_ZONES],
) -> [f64; BELIEF_ZONES] {
    let mut col_sums = [0.0f64; BELIEF_ZONES];
    for (dst, &value) in col_sums.iter_mut().zip(hidden_counts.iter()) {
        *dst = value as f64;
    }
    col_sums
}

pub fn build_stage_a_teacher(
    remaining: &[f32; BELIEF_TILES],
    hidden_counts: &[usize; BELIEF_ZONES],
    config: StageABeliefConfig,
) -> Option<StageABeliefTarget> {
    if config.num_components as usize != BELIEF_COMPONENTS {
        return None;
    }
    let hidden_tiles: usize = hidden_counts.iter().sum();
    if hidden_tiles == 0 {
        return None;
    }

    let row_sums = project_public_remaining_to_row_sums(remaining);
    let total_row: f64 = row_sums.iter().sum();
    if total_row <= 0.0 {
        return None;
    }

    let col_sums = project_hidden_counts_to_col_sums(hidden_counts);
    let total_col: f64 = col_sums.iter().sum();
    if (total_row - total_col).abs() > 0.5 {
        return None;
    }
    let kernel = build_uniform_kernel();
    let mixture = MixtureSib::new(config.num_components, &kernel, &row_sums, &col_sums);
    let weights = mixture.weights();
    let entropy = mixture.weight_entropy() as f32;
    let ess = mixture.ess() as f32;
    let trust = ((ess / config.num_components as f32).clamp(0.0, 1.0) * 0.7
        + (1.0 - (entropy / 1.3863).clamp(0.0, 1.0)) * 0.3)
        .clamp(0.0, 1.0);

    if trust < config.trust_threshold {
        return None;
    }

    let mut belief_fields = [0.0f32; BELIEF_FIELDS_SIZE];
    for component in 0..BELIEF_COMPONENTS {
        for zone in 0..BELIEF_ZONES {
            let channel = component * BELIEF_ZONES + zone;
            for tile in 0..BELIEF_TILES {
                belief_fields[channel * BELIEF_TILES + tile] =
                    mixture.components[component].belief[tile * BELIEF_ZONES + zone] as f32;
            }
        }
    }

    let mixture_weights =
        if config.mixture_entropy_threshold >= 0.0 && entropy <= config.mixture_entropy_threshold {
            let mut out = [0.0f32; BELIEF_COMPONENTS];
            for (dst, src) in out.iter_mut().zip(weights.iter().copied()) {
                *dst = src as f32;
            }
            Some(out)
        } else {
            None
        };

    Some(StageABeliefTarget {
        belief_fields,
        mixture_weights,
        trust,
        ess,
        entropy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_a_teacher_returns_none_without_hidden_tiles() {
        let remaining = [1.0f32; BELIEF_TILES];
        assert!(build_stage_a_teacher(
            &remaining,
            &[0; BELIEF_ZONES],
            StageABeliefConfig::default()
        )
        .is_none());
    }

    #[test]
    fn stage_a_teacher_produces_projected_belief_fields() {
        let remaining = [1.0f32; BELIEF_TILES];
        let target =
            build_stage_a_teacher(&remaining, &[8, 8, 8, 10], StageABeliefConfig::default())
                .expect("teacher target");
        assert!(target.trust >= StageABeliefConfig::default().trust_threshold);
        assert!(target
            .belief_fields
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn stage_a_teacher_can_emit_mixture_weights() {
        let remaining = [1.0f32; BELIEF_TILES];
        let cfg = StageABeliefConfig {
            mixture_entropy_threshold: 10.0,
            ..StageABeliefConfig::default()
        };
        let target =
            build_stage_a_teacher(&remaining, &[8, 8, 8, 10], cfg).expect("teacher target");
        assert!(target.mixture_weights.is_some());
    }

    #[test]
    fn stage_a_teacher_keeps_mixture_default_off() {
        let remaining = [1.0f32; BELIEF_TILES];
        let target =
            build_stage_a_teacher(&remaining, &[8, 8, 8, 10], StageABeliefConfig::default())
                .expect("teacher target");
        assert!(target.mixture_weights.is_none());
    }

    #[test]
    fn stage_a_teacher_returns_none_when_public_remaining_disagrees_with_hidden_mass() {
        let remaining = [0.0f32; BELIEF_TILES];
        assert!(
            build_stage_a_teacher(&remaining, &[8, 8, 8, 10], StageABeliefConfig::default(),)
                .is_none()
        );
    }

    #[test]
    fn stage_a_teacher_rejects_noncanonical_component_count() {
        let remaining = [1.0f32; BELIEF_TILES];
        let cfg = StageABeliefConfig {
            num_components: 2,
            ..StageABeliefConfig::default()
        };
        assert!(build_stage_a_teacher(&remaining, &[8, 8, 8, 10], cfg).is_none());
    }

    #[test]
    fn stage_a_audit_summary_tracks_coverage() {
        let remaining = [1.0f32; BELIEF_TILES];
        let target =
            build_stage_a_teacher(&remaining, &[8, 8, 8, 10], StageABeliefConfig::default());
        let mut audit = StageABeliefAuditSummary::default();
        audit.record(target.as_ref());
        audit.record(None);
        assert_eq!(audit.total, 2);
        assert!(audit.belief_coverage() <= 1.0);
        assert!(audit.mixture_coverage() <= 1.0);
    }
}
