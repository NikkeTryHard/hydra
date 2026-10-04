//! rl/lifecycle_report: scheduling choice + PPO report reduction (pure).
//!
//! DAG: `super::lifecycle_common` + `serde_json` + `hydra_feed::canon`.
//! FORBIDS: PyO3, wall-clock, Torch access, policy/source/shuffle ids in
//! any report (resource/queue/generation counters belong to
//! control/evidence, never to checkpoint-adjacent metrics).

use serde_json::Value;

use super::lifecycle_common::{lc_err, LifecycleError};

/// Next useful GPU work class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    /// One eligible actor inference ticket.
    Inference,
    /// One available learner microstep.
    Learner,
    /// No eligible work in either class (caller counts the stall by reason).
    Idle,
}

impl WorkKind {
    /// Stable wire name (`inference` / `learner` / `idle`).
    pub fn as_str(self) -> &'static str {
        match self {
            WorkKind::Inference => "inference",
            WorkKind::Learner => "learner",
            WorkKind::Idle => "idle",
        }
    }
}

/// Alternating choice between one eligible inference ticket and one learner
/// microstep (plan section 5: at most one of each, immediate fallback when
/// the preferred class lacks work).
///
/// Invariants: the preferred class is served whenever eligible; the other
/// class is served immediately when the preferred class is idle; the
/// preference toggles exactly when work is served (an `Idle` verdict never
/// flips it), so two continuously-eligible classes strictly alternate and
/// any continuously-eligible class waits at most one verdict.
pub fn choose_work(
    preferred_inference: bool,
    inference_eligible: bool,
    learner_eligible: bool,
) -> (WorkKind, bool) {
    if preferred_inference {
        if inference_eligible {
            (WorkKind::Inference, false)
        } else if learner_eligible {
            (WorkKind::Learner, false)
        } else {
            (WorkKind::Idle, true)
        }
    } else if learner_eligible {
        (WorkKind::Learner, true)
    } else if inference_eligible {
        (WorkKind::Inference, true)
    } else {
        (WorkKind::Idle, false)
    }
}

/// Accumulated PPO buffer layout (23 f32, `PpoTerms` field order in
/// `training/rl_objective.py`): Torch accumulates N-normalized micro shares
/// by summation — each micro's `loss` already divides by the TRUE complete
/// count N (never the micro size), so summing micro losses IS the complete
/// loss — and native divides the raw sums by N (value terms by 4N).
/// Index map: 0 loss, 1 ratio_sum, 2 clip_count, 3 entropy_sum,
/// 4 value_squared_sum, 5 bc_kl_sum, 6 approximate_kl_sum,
/// 7..11 residual_sum_by_seat, 11..15 residual_squared_sum_by_seat,
/// 15..19 return_sum_by_seat, 19..23 return_squared_sum_by_seat.
pub const PPO_REPORT_LEN: usize = 23;

/// Reduce one accumulated PPO buffer into finite report means plus per-seat
/// explained variance (`EV = 1 - popVar(residual)/popVar(return)` with
/// population variances from the accumulated sums). A seat whose return
/// variance is `<= 0` reports `{value: null, defined: false}` and increments
/// `zero_variance_count` — never NaN, never a fabricated 0. Any non-finite
/// input or nonpositive count fails the whole report (no partial metrics).
pub fn ppo_report(count: u64, stats: &[f32]) -> Result<Value, LifecycleError> {
    if count == 0 {
        return Err(lc_err("rl lifecycle ppo_report count must be positive"));
    }
    if stats.len() != PPO_REPORT_LEN {
        return Err(lc_err(format!(
            "rl lifecycle ppo_report needs {PPO_REPORT_LEN} scalars, got {}",
            stats.len()
        )));
    }
    let mut value = [0.0f64; PPO_REPORT_LEN];
    for (i, scalar) in stats.iter().enumerate() {
        let v = f64::from(*scalar);
        if !v.is_finite() {
            return Err(lc_err(format!("rl lifecycle ppo_report stats[{i}] is non-finite")));
        }
        value[i] = v;
    }
    let n = count as f64;
    let mean = |sum: f64, denom: f64| sum / denom;
    let mut per_seat = Vec::with_capacity(4);
    let mut zero_variance_count: u64 = 0;
    for seat in 0..4 {
        let residual_sum = value[7 + seat];
        let residual_sq_sum = value[11 + seat];
        let return_sum = value[15 + seat];
        let return_sq_sum = value[19 + seat];
        let mean_residual = mean(residual_sum, n);
        let mean_return = mean(return_sum, n);
        let residual_var = mean(residual_sq_sum, n) - mean_residual * mean_residual;
        let return_var = mean(return_sq_sum, n) - mean_return * mean_return;
        for (name, v) in [
            ("mean_residual", mean_residual),
            ("mean_return", mean_return),
            ("residual_variance", residual_var),
            ("return_variance", return_var),
        ] {
            if !v.is_finite() {
                return Err(lc_err(format!(
                    "rl lifecycle ppo_report seat {seat} {name} is non-finite"
                )));
            }
        }
        let explained = if return_var <= 0.0 {
            zero_variance_count += 1;
            serde_json::json!({ "value": null, "defined": false })
        } else {
            let ev = 1.0 - residual_var / return_var;
            if !ev.is_finite() {
                return Err(lc_err(format!(
                    "rl lifecycle ppo_report seat {seat} explained variance is non-finite"
                )));
            }
            serde_json::json!({ "value": ev, "defined": true })
        };
        per_seat.push(serde_json::json!({
            "seat": seat as u64,
            "mean_residual": mean_residual,
            "mean_return": mean_return,
            "residual_variance": residual_var,
            "return_variance": return_var,
            "explained_variance": explained,
        }));
    }
    let report = serde_json::json!({
        "count": count,
        "loss": value[0],
        "mean_ratio": mean(value[1], n),
        "clip_fraction": mean(value[2], n),
        "mean_entropy": mean(value[3], n),
        "value_loss": mean(value[4], 4.0 * n),
        "mean_bc_kl": mean(value[5], n),
        "mean_approximate_kl": mean(value[6], n),
        "per_seat": per_seat,
        "zero_variance_count": zero_variance_count,
    });
    // Finite-output gate (loss/means over finite inputs with positive N are
    // finite; re-check rather than assume).
    if !report_numbers_finite(&report) {
        return Err(lc_err("rl lifecycle ppo_report outputs are non-finite"));
    }
    Ok(report)
}

fn report_numbers_finite(value: &Value) -> bool {
    match value {
        Value::Number(n) => n.as_f64().is_some_and(|f| f.is_finite()),
        Value::Array(items) => items.iter().all(report_numbers_finite),
        Value::Object(map) => map.values().all(report_numbers_finite),
        Value::Null | Value::Bool(_) | Value::String(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_alternates_when_both_eligible_and_never_flips_on_idle() {
        // Both eligible: serve preferred, then strictly alternate.
        assert_eq!(choose_work(true, true, true), (WorkKind::Inference, false));
        assert_eq!(choose_work(false, true, true), (WorkKind::Learner, true));
        // Fallback serves the live class and toggles (served work flips).
        assert_eq!(choose_work(true, false, true), (WorkKind::Learner, false));
        assert_eq!(choose_work(false, true, false), (WorkKind::Inference, true));
        // Idle never flips the preference.
        assert_eq!(choose_work(true, false, false), (WorkKind::Idle, true));
        assert_eq!(choose_work(false, false, false), (WorkKind::Idle, false));
    }

    #[test]
    fn ppo_report_divides_by_true_counts_and_nulls_degenerate_seats() {
        // N=4: ratio_sum 4 -> mean 1; value_squared_sum 16 -> 16/16=1.
        let mut stats = [0.0f32; PPO_REPORT_LEN];
        stats[0] = 0.5;
        stats[1] = 4.0;
        stats[2] = 1.0;
        stats[4] = 16.0;
        // Seat 0: residual sums 0; return sum 10, sq 30 ->
        // var = 30/4 - 2.5^2 = 1.25 > 0 -> defined EV = 1.
        stats[15] = 10.0;
        stats[19] = 30.0;
        let report = ppo_report(4, &stats).unwrap();
        assert_eq!(report["count"], Value::from(4u64));
        assert_eq!(report["mean_ratio"], Value::from(1.0));
        assert_eq!(report["value_loss"], Value::from(1.0));
        assert_eq!(report["clip_fraction"], Value::from(0.25));
        assert_eq!(
            report["per_seat"][0]["explained_variance"]["defined"],
            Value::Bool(true)
        );
        // Seats 1..3 have zero return variance -> null, counted.
        assert_eq!(report["zero_variance_count"], Value::from(3u64));
        assert_eq!(
            report["per_seat"][1]["explained_variance"],
            serde_json::json!({"value": null, "defined": false})
        );
        // Non-finite input fails the whole report; zero count fails.
        let mut bad = [0.0f32; PPO_REPORT_LEN];
        bad[1] = f32::INFINITY;
        assert!(ppo_report(4, &bad).is_err());
        assert!(ppo_report(0, &[0.0f32; PPO_REPORT_LEN]).is_err());
    }
}
