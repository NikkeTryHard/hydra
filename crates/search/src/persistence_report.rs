//! Persistence factorial facade + frozen whole-block report (EvalControl-owned).
//!
//! Rust port of `persistence_factorial.py` (71 ln, `__all__` facade surface
//! as constructor aliases over the split modules) + `persistence_report.py`
//! (223 ln, frozen whole-block contrasts and strata).
//! - `factorial_contrasts` (`:68-97`): P-F, R-F, P-R, P-C contrasts as
//!   per-block mean differences with caller-supplied (numpy PCG64)
//!   bootstrap triples. Stream seeds are `sha256("persistence-{name}-v1")`
//!   (`:94`); the P-F seed bytes are pinned (`PERSIST_PF_B0` golden).
//!   Estimate goldens (FEST) pin the means — draw-independent.
//! - `stratify_surprise_miss_recovery` (`:100-127`): per-packet hit /
//!   miss / recovery strata with hit/miss rates (empty logs -> zero
//!   rates, never NaN: `total = raw or 1`, `:119`).
//! - `generate_factorial_report` (`:130-223`): frozen deterministic
//!   report — per-arm means, contrasts, block-manifest hash over canon
//!   `{block_ids, placements_by_arm}`, resource-equality refusal (P vs F
//!   identical calls raise, `:186-190`), fixed UTC-shape timestamp is
//!   caller-supplied (no clock reads in this pure module).

use std::collections::HashMap;

use crate::SearchError;

use crate::eval::{canon_bytes_of, hex_of, neumaier_sum, sha256_bytes};
use crate::persistence_kernel::ArmId;
use crate::persistence_planner::CommitEntry;

// ---------------------------------------------------------------------------
// Report items (full family surface lives in `persistence_factorial`).
// ---------------------------------------------------------------------------

/// Contrast estimate with its caller-supplied interval
/// (`FactorialContrasts`, `:42-48`).
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialContrasts {
    /// Mean per-block difference.
    pub estimate: f64,
    /// Bootstrap low (numpy lane).
    pub ci_low: f64,
    /// Bootstrap high (numpy lane).
    pub ci_high: f64,
    /// Always `wall_block`.
    pub unit: String,
}

/// Contrast pairs computed (`:86`).
pub const CONTRAST_PAIRS: [(&str, ArmId, ArmId); 4] = [
    ("P-F", ArmId::P, ArmId::F),
    ("R-F", ArmId::R, ArmId::F),
    ("P-R", ArmId::P, ArmId::R),
    ("P-C", ArmId::P, ArmId::C),
];

/// Persistence stream seed (`:94`): `sha256("persistence-{name}-v1")`.
pub fn persistence_contrast_seed(name: &str) -> [u8; 32] {
    sha256_bytes(format!("persistence-{name}-v1").as_bytes())
}

/// Per-block differences for one contrast pair (`_block_diffs`,
/// `:81-84`): lengths must match.
pub fn block_diffs(a: &[f64], b: &[f64]) -> Result<Vec<f64>, SearchError> {
    if a.len() != b.len() {
        return Err(SearchError::InvalidArg {
            detail: "block count mismatch",
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
}

/// Compute P-F, R-F, P-R, P-C contrasts (`factorial_contrasts`,
/// `:68-97`). `intervals` carries the numpy-lane bootstrap triples
/// keyed by contrast name; estimates are the per-block mean
/// differences (FEST goldens pin all four).
pub fn factorial_contrasts(
    placements_by_arm: &HashMap<String, Vec<f64>>,
    intervals: &HashMap<String, (f64, f64)>,
) -> Result<HashMap<String, FactorialContrasts>, SearchError> {
    let mut out = HashMap::new();
    for (name, a, b) in CONTRAST_PAIRS {
        let a_vals = placements_by_arm
            .get(a.name())
            .ok_or(SearchError::InvalidArg {
                detail: "missing arm placements",
            })?;
        let b_vals = placements_by_arm
            .get(b.name())
            .ok_or(SearchError::InvalidArg {
                detail: "missing arm placements",
            })?;
        let diffs = block_diffs(a_vals, b_vals)?;
        if diffs.len() < 2 {
            return Err(SearchError::InvalidArg {
                detail: "need at least two blocks for any interval",
            });
        }
        let (low, high) = intervals.get(name).copied().unwrap_or((f64::NAN, f64::NAN));
        // proof: diff-slice len < 2^53 in practice, exact; mean tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let diffs_len_f: f64 = diffs.len() as f64;
        out.insert(
            name.to_string(),
            FactorialContrasts {
                estimate: neumaier_sum(&diffs) / diffs_len_f,
                ci_low: low,
                ci_high: high,
                unit: "wall_block".to_string(),
            },
        );
    }
    Ok(out)
}

/// Per-arm outcome strata (`stratify_surprise_miss_recovery`,
/// `:100-127`).
#[derive(Debug, Clone, PartialEq)]
pub struct ArmStrata {
    /// Counts by outcome.
    pub counts: HashMap<String, usize>,
    /// Summed ponder calls by outcome.
    pub ponder_calls_by_outcome: HashMap<String, usize>,
    /// `hit / total` (`total = raw or 1` — never NaN).
    pub hit_rate: f64,
    /// `(miss_recovery + rebuild_no_forest) / total`.
    pub miss_rate: f64,
    /// Total packets (min 1 in the rate denominator).
    pub total_packets: usize,
}

/// Stratify per-packet outcomes (`:100-127`).
pub fn stratify_surprise_miss_recovery(
    commit_logs_by_arm: &HashMap<String, Vec<CommitEntry>>,
) -> HashMap<String, ArmStrata> {
    let mut strata = HashMap::new();
    for (arm, logs) in commit_logs_by_arm {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut ponder: HashMap<String, usize> = HashMap::new();
        for entry in logs {
            *counts.entry(entry.outcome.clone()).or_insert(0) += 1;
            *ponder.entry(entry.outcome.clone()).or_insert(0) += entry.ponder_calls;
        }
        let hit = counts.get("hit").copied().unwrap_or(0);
        let miss = counts.get("miss_recovery").copied().unwrap_or(0)
            + counts.get("rebuild_no_forest").copied().unwrap_or(0);
        let raw: usize = counts.values().sum();
        let total = if raw != 0 { raw } else { 1 };
        // proof: packet counts are small (< 2^53), exact; rate tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let hit_f: f64 = hit as f64;
        #[allow(clippy::cast_precision_loss)]
        let miss_f: f64 = miss as f64;
        #[allow(clippy::cast_precision_loss)]
        let total_f: f64 = total as f64;
        strata.insert(
            arm.clone(),
            ArmStrata {
                counts,
                ponder_calls_by_outcome: ponder,
                hit_rate: hit_f / total_f,
                miss_rate: miss_f / total_f,
                total_packets: total,
            },
        );
    }
    strata
}

/// Resource sample per block (`resource_samples_by_arm` rows, `:174-184`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceSample {
    /// Model calls.
    pub model_calls: u64,
    /// Exact transitions.
    pub exact_transitions: u64,
    /// Energy joules.
    pub energy_joules: f64,
}

/// Frozen whole-block factorial report (`FactorialReport`, `:51-65`).
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialReport {
    /// Report id.
    pub report_id: String,
    /// Caller-supplied UTC timestamp (`%Y-%m-%dT%H:%M:%SZ` shape).
    pub generated_at_utc: String,
    /// Block-manifest hash over canon `{block_ids, placements_by_arm}`.
    pub block_manifest_hash: String,
    /// Block count.
    pub num_blocks: usize,
    /// Per-arm means.
    pub per_arm_mean: HashMap<String, f64>,
    /// Contrasts.
    pub contrasts: HashMap<String, FactorialContrasts>,
    /// Resource samples by arm.
    pub resource_samples: HashMap<String, Vec<ResourceSample>>,
    /// Outcome strata by arm.
    pub strata: HashMap<String, ArmStrata>,
}

/// Generate the frozen deterministic factorial report
/// (`generate_factorial_report`, `:130-223`). Equal block counts
/// required; `block_ids` default to `wall-block-{i:04d}`; P vs F
/// identical `model_calls` raise (`:186-190` — never claim resource
/// equality); the timestamp is caller-supplied (pure module, no clock).
pub fn generate_factorial_report(
    placements_by_arm: &HashMap<String, Vec<f64>>,
    block_ids: Option<Vec<String>>,
    resource_samples_by_arm: &HashMap<String, Vec<ResourceSample>>,
    commit_logs_by_arm: &HashMap<String, Vec<CommitEntry>>,
    intervals: &HashMap<String, (f64, f64)>,
    report_id: &str,
    generated_at_utc: &str,
) -> Result<FactorialReport, SearchError> {
    if placements_by_arm.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "placements_by_arm must not be empty",
        });
    }
    let mut lengths: Vec<usize> = placements_by_arm.values().map(Vec::len).collect();
    lengths.sort_unstable();
    lengths.dedup();
    if lengths.len() != 1 {
        return Err(SearchError::InvalidArg {
            detail: "each arm must have same block count",
        });
    }
    let n = lengths[0];
    if n == 0 {
        return Err(SearchError::InvalidArg {
            detail: "must have at least one block",
        });
    }
    let ids = match block_ids {
        Some(ids) => {
            if ids.len() != n {
                return Err(SearchError::InvalidArg {
                    detail: "block_ids length mismatch",
                });
            }
            ids
        }
        None => (0..n).map(|i| format!("wall-block-{i:04}")).collect(),
    };
    let mut per_arm_mean = HashMap::new();
    for (arm, vals) in placements_by_arm {
        // proof: placement-slice len < 2^53 in practice, exact; mean tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let vals_len_f: f64 = vals.len() as f64;
        per_arm_mean.insert(arm.clone(), neumaier_sum(vals) / vals_len_f);
    }
    let contrasts = factorial_contrasts(placements_by_arm, intervals)?;
    let payload = serde_json::json!({
        "block_ids": ids,
        "placements_by_arm": placements_by_arm,
    });
    let bytes = canon_bytes_of(&payload, "persistence:report:blocks")?;
    let block_manifest_hash = format!("sha256:{}", hex_of(&sha256_bytes(&bytes)));
    if let (Some(p), Some(f)) = (
        resource_samples_by_arm.get("P"),
        resource_samples_by_arm.get("F"),
    ) {
        let p_calls: Vec<u64> = p.iter().map(|s| s.model_calls).collect();
        let f_calls: Vec<u64> = f.iter().map(|s| s.model_calls).collect();
        if p_calls == f_calls {
            return Err(SearchError::InvalidArg {
                detail: "P and F must not claim identical resource equality",
            });
        }
    }
    Ok(FactorialReport {
        report_id: report_id.to_string(),
        generated_at_utc: generated_at_utc.to_string(),
        block_manifest_hash,
        num_blocks: n,
        per_arm_mean,
        contrasts,
        resource_samples: resource_samples_by_arm.clone(),
        strata: stratify_surprise_miss_recovery(commit_logs_by_arm),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placements() -> HashMap<String, Vec<f64>> {
        [
            ("B".to_string(), vec![0.1, 0.2]),
            ("F".to_string(), vec![0.0, 0.1]),
            ("R".to_string(), vec![-0.1, 0.0]),
            ("P".to_string(), vec![-0.2, -0.1]),
            ("C".to_string(), vec![-0.25, -0.15]),
        ]
        .into_iter()
        .collect()
    }

    /// FEST goldens (`/tmp/eval_goldens3.py`): mean per-block
    /// differences — draw-independent, so they pin the assembly
    /// without any PCG64 port. P-F seed preimage golden (the oracle's
    /// `PERSIST_PF_B0` records the first 8 bytes `36ddb6520588a78a` of
    /// the full `sha256("persistence-P-F-v1")` below).
    #[test]
    fn factorial_estimate_goldens() {
        assert_eq!(
            hex_of(&persistence_contrast_seed("P-F")),
            "84ffb831308eb027f903420744139de37bd8a2a5e88e41135b4bc5d7275b44f8"
        );
        let contrasts = factorial_contrasts(&placements(), &HashMap::new()).unwrap();
        assert_eq!(contrasts["P-F"].estimate, -0.2);
        assert_eq!(contrasts["R-F"].estimate, -0.1);
        assert_eq!(contrasts["P-R"].estimate, -0.1);
        assert!((contrasts["P-C"].estimate - 0.04999999999999999).abs() < 1e-15);
        for name in ["P-F", "R-F", "P-R", "P-C"] {
            assert_eq!(contrasts[name].unit, "wall_block");
        }
    }

    /// FACTOR_BLOCK_HASH golden (`/tmp/eval_goldens2.py`) over
    /// `{block_ids: [wb-0, wb-1], placements}`; P mean golden.
    #[test]
    fn factorial_block_hash_golden() {
        let samples: HashMap<String, Vec<ResourceSample>> = [
            (
                "P".to_string(),
                vec![
                    ResourceSample {
                        model_calls: 36,
                        exact_transitions: 144,
                        energy_joules: 1.5,
                    },
                    ResourceSample {
                        model_calls: 37,
                        exact_transitions: 148,
                        energy_joules: 1.6,
                    },
                ],
            ),
            (
                "F".to_string(),
                vec![
                    ResourceSample {
                        model_calls: 32,
                        exact_transitions: 128,
                        energy_joules: 1.3,
                    },
                    ResourceSample {
                        model_calls: 33,
                        exact_transitions: 132,
                        energy_joules: 1.4,
                    },
                ],
            ),
        ]
        .into_iter()
        .collect();
        let report = generate_factorial_report(
            &placements(),
            Some(vec!["wb-0".to_string(), "wb-1".to_string()]),
            &samples,
            &HashMap::new(),
            &HashMap::new(),
            "persistence-factorial-whole-block-v1",
            "2026-09-16T00:00:00Z",
        )
        .unwrap();
        assert_eq!(
            report.block_manifest_hash,
            "sha256:fc29ea7e0fa476253a80bd85b8e30e39f2683e079f17737f923e2658c54cd719"
        );
        assert_eq!(report.per_arm_mean["P"], -0.15000000000000002);
        assert_eq!(report.num_blocks, 2);
        // Identical P/F calls raise (never claim resource equality).
        let mut same = samples.clone();
        same.insert(
            "P".to_string(),
            vec![
                ResourceSample {
                    model_calls: 32,
                    exact_transitions: 128,
                    energy_joules: 1.3,
                },
                ResourceSample {
                    model_calls: 33,
                    exact_transitions: 132,
                    energy_joules: 1.4,
                },
            ],
        );
        assert!(
            generate_factorial_report(
                &placements(),
                Some(vec!["wb-0".to_string(), "wb-1".to_string()]),
                &same,
                &HashMap::new(),
                &HashMap::new(),
                "r",
                "2026-09-16T00:00:00Z",
            )
            .is_err()
        );
    }

    /// Strata: hit/miss rates over the commit log; empty logs give
    /// zero rates (never NaN — `total = raw or 1`).
    #[test]
    fn strata_rates() {
        let logs: HashMap<String, Vec<CommitEntry>> = [(
            "P".to_string(),
            vec![
                CommitEntry {
                    arm: ArmId::P,
                    packet_id: "p1".to_string(),
                    outcome: "hit".to_string(),
                    ponder_calls: 4,
                },
                CommitEntry {
                    arm: ArmId::P,
                    packet_id: "p2".to_string(),
                    outcome: "miss_recovery".to_string(),
                    ponder_calls: 2,
                },
                CommitEntry {
                    arm: ArmId::P,
                    packet_id: "p3".to_string(),
                    outcome: "rebuild_no_forest".to_string(),
                    ponder_calls: 0,
                },
            ],
        )]
        .into_iter()
        .collect();
        let strata = stratify_surprise_miss_recovery(&logs);
        assert_eq!(strata["P"].total_packets, 3);
        assert!((strata["P"].hit_rate - 1.0 / 3.0).abs() < 1e-15);
        assert!((strata["P"].miss_rate - 2.0 / 3.0).abs() < 1e-15);
        assert_eq!(strata["P"].ponder_calls_by_outcome["hit"], 4);
        let empty: HashMap<String, Vec<CommitEntry>> =
            [("P".to_string(), Vec::new())].into_iter().collect();
        let strata = stratify_surprise_miss_recovery(&empty);
        assert_eq!(strata["P"].hit_rate, 0.0);
        assert_eq!(strata["P"].total_packets, 1);
    }

    /// Split-module surface resolves (B4 family ships as one surface;
    /// full facade aliases live in `persistence_factorial`).
    #[test]
    fn split_surface() {
        let arm = crate::persistence_kernel::make_persistence_arm(ArmId::R);
        assert!(arm.retain_state && !arm.opponent_time_compute);
        let planner = crate::persistence_planner::PersistencePlanner::new(
            ArmId::F,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(planner.max_model_calls, 32);
        let spec = crate::persistence_spec::make_persistence_candidate_spec(
            ArmId::B,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(spec.candidate_id, "persistence-B");
    }
}
