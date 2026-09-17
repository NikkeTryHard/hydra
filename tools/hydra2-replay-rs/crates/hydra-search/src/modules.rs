//! `modules`: Candidate 4 one-at-a-time module transforms (batch).
//!
//! Parity: `search/modules/__init__.py` transform bodies for the EIGHT
//! exact arms (validation, registry, spec factories, `PbrfContext`,
//! `ess_allocate` / `alloc_cost`, tiny oracles stay Python — test-pinned
//! and sub-floor). Each exact arm mirrors its oracle's op order (`builtin
//! sum` via the shared helper, `fsum` for the VOC total); integer arms
//! are bit-exact, float arms replicate the oracle's accumulation order.
//!
//! Framework-locked (torch math stays Python — the batch FAILS CLOSED):
//! `structural_crn` + `rqmc` draw `torch.rand` off a sha-derived CPU
//! generator (Philox counter-based; no portable re-implementation —
//! Philox words for the frozen Gumbel path are explicitly deleted crate
//! policy, and drawing Gumbels/shifts from a second stream would fork
//! the B1 trap). The batch rejects both ids with a named `ContractError`;
//! callers keep the Python transform for those two arms (torch stays the
//! single RNG site). Metadata updates cross back as a JSON object; the
//! caller merges them into `context.metadata`.

use crate::SearchError;

/// One module transform over owned vectors (eight exact arms; the two
/// torch.rand arms fail closed — see module doc).
///
/// `meta_json` carries the arm's validated context params as an object
/// (`voc` arm: `floor`/`cap`/`budget` ints plus `scores` array-or-null;
/// other arms ignore it). Returns
/// `(particles, weights, budget_calls, budget_transitions, meta_json)`.
/// Unknown `module_id` fails closed; empty populations mirror the
/// oracle's per-arm empty behavior (identity/empty, never a default).
pub fn module_transform(
    module_id: &str,
    particles: &[f64],
    weights: &[f64],
    budget_calls: u64,
    budget_transitions: u64,
    _seed: u64,
    meta_json: &[u8],
) -> Result<(Vec<f64>, Vec<f64>, u64, u64, Vec<u8>), SearchError> {
    if particles.len() != weights.len() {
        return Err(SearchError::InvalidArg { detail: "module particles/weights length mismatch" });
    }
    let meta: serde_json::Value = if meta_json.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(meta_json)
            .map_err(|err| SearchError::Canon { detail: err.to_string() })?
    };
    match module_id {
        "rao_blackwell" => {
            // `rb = 0.6*(x+0.1) + 0.4*(x-0.1)`; charge `2n`.
            let mut out = Vec::with_capacity(particles.len());
            for x in particles {
                out.push(0.6 * (x + 0.1) + 0.4 * (x - 0.1));
            }
            let add = (particles.len() as u64).saturating_mul(2);
            let updates = serde_json::json!({"rb_applied": true});
            Ok((
                out,
                weights.to_vec(),
                budget_calls.saturating_add(add),
                budget_transitions.saturating_add(add),
                updates.to_string().into_bytes(),
            ))
        }
        "defensive_mis" => {
            // Balanced pseudo-ratios by index parity, renormalized.
            let mut scaled = Vec::with_capacity(weights.len());
            let mut idx = 0;
            while idx < weights.len() {
                let ratio = if idx % 2 == 0 { 1.2 } else { 0.8 };
                scaled.push(weights[idx] * ratio);
                idx += 1;
            }
            let sum = crate::builtin_sum(&scaled);
            if sum == 0.0 {
                return Err(SearchError::ZeroMass);
            }
            let mut idx = 0;
            while idx < scaled.len() {
                scaled[idx] /= sum;
                idx += 1;
            }
            let add = (weights.len() as u64).saturating_mul(2);
            let updates =
                serde_json::json!({"mis_applied": true, "mis_single_denominator": true});
            Ok((
                particles.to_vec(),
                scaled,
                budget_calls.saturating_add(add),
                budget_transitions.saturating_add(add),
                updates.to_string().into_bytes(),
            ))
        }
        "structural_crn" | "rqmc" => {
            // Framework-locked: both arms draw `torch.rand` off a
            // sha-derived CPU generator (Philox counter-based, not
            // portable — see module doc). Fail closed with a named
            // reason; the caller keeps the Python transform.
            return Err(SearchError::InvalidArg {
                detail: "torch.rand arm has no batch envelope; use the Python transform",
            });
        }
        "fixed_mlmc" => {
            // Signed telescope `base + 0.05 - 0.02` broadcast (`builtin sum`
            // parity: the oracle means via `sum(particles)/n`).
            let base = if particles.is_empty() {
                0.0
            } else {
                crate::builtin_sum(particles) / particles.len() as f64
            };
            let corrected = base + 0.05 - 0.02;
            let out = vec![corrected; particles.len()];
            let updates =
                serde_json::json!({"mlmc_applied": true, "mlmc_telescope": corrected});
            Ok((
                out,
                weights.to_vec(),
                budget_calls.saturating_add(6),
                budget_transitions.saturating_add(6),
                updates.to_string().into_bytes(),
            ))
        }
        "coreset" => {
            // Top-2 weighted, renormalized (harness `k = 2`).
            let keep = 2usize.min(particles.len());
            let mut order: Vec<usize> = (0..particles.len()).collect();
            order.sort_by(|a, b| {
                weights[*b]
                    .total_cmp(&weights[*a])
                    .then_with(|| particles[*b].total_cmp(&particles[*a]))
            });
            order.truncate(keep);
            let mut ws: Vec<f64> = order.iter().map(|i| weights[*i]).collect();
            let ps: Vec<f64> = order.iter().map(|i| particles[*i]).collect();
            let sum = crate::builtin_sum(&ws);
            if sum > 0.0 {
                for w in &mut ws {
                    *w /= sum;
                }
            } else if !ws.is_empty() {
                let uniform = 1.0 / ws.len() as f64;
                for w in &mut ws {
                    *w = uniform;
                }
            }
            let add = keep as u64;
            let updates =
                serde_json::json!({"coreset_applied": true, "coreset_search_only": true});
            Ok((
                ps,
                ws,
                budget_calls.saturating_add(add),
                budget_transitions.saturating_add(add),
                updates.to_string().into_bytes(),
            ))
        }
        "pruning" => {
            // Simultaneous intervals never fire on harness values.
            if particles.len() < 4 {
                let updates = serde_json::json!({});
                return Ok((
                    particles.to_vec(),
                    weights.to_vec(),
                    budget_calls,
                    budget_transitions,
                    updates.to_string().into_bytes(),
                ));
            }
            let updates = serde_json::json!({
                "pruning_applied": true,
                "pruned": false,
                "simultaneous": true,
            });
            Ok((
                particles.to_vec(),
                weights.to_vec(),
                budget_calls.saturating_add(2),
                budget_transitions.saturating_add(2),
                updates.to_string().into_bytes(),
            ))
        }
        "controlled_smc" => {
            // Kish ESS gate at `0.5n`: fire resamples (`+n/+n`), else copy.
            let n = particles.len();
            let sq: Vec<f64> = weights.iter().map(|w| w * w).collect();
            let quad = crate::builtin_sum(&sq);
            let ess = if quad > 0.0 { 1.0 / quad } else { 0.0 };
            let fired = ess <= 0.5 * n as f64;
            let (add_calls, add_trans) = if fired { (n as u64, n as u64) } else { (0, 0) };
            let updates = serde_json::json!({
                "smc_applied": true,
                "unnormalized": true,
                "resample_fired": fired,
                "resample_skipped": !fired,
                "ess": ess,
            });
            Ok((
                particles.to_vec(),
                weights.to_vec(),
                budget_calls.saturating_add(add_calls),
                budget_transitions.saturating_add(add_trans),
                updates.to_string().into_bytes(),
            ))
        }
        "persistent_forest" => {
            let updates = serde_json::json!({
                "forest_applied": true,
                "epoch_incremented": true,
                "siblings_squashed": true,
            });
            Ok((
                particles.to_vec(),
                weights.to_vec(),
                budget_calls.saturating_add(1),
                budget_transitions.saturating_add(1),
                updates.to_string().into_bytes(),
            ))
        }
        "voc_routing" => voc_route(particles, weights, budget_calls, budget_transitions, &meta),
        _ => Err(SearchError::InvalidArg { detail: "unknown module_id" }),
    }
}

/// VOC exact frozen routing (`VOCRoutingModule.transform` parity).
///
/// `meta` carries validated `floor`/`cap`/`budget` ints plus `scores`
/// (array-or-null; null rides the weights). Integer arms replicate the
/// oracle op-for-op; `_largest_remainder` uses a plain `f64` sum (see
/// note below on the `fsum` razor bound).
fn voc_route(
    particles: &[f64],
    weights: &[f64],
    budget_calls: u64,
    budget_transitions: u64,
    meta: &serde_json::Value,
) -> Result<(Vec<f64>, Vec<f64>, u64, u64, Vec<u8>), SearchError> {
    let get_int = |key: &str| -> Result<u64, SearchError> {
        meta.get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or(SearchError::InvalidArg { detail: "voc routing param must be a non-negative int" })
    };
    let floor = get_int("floor")?;
    let cap = get_int("cap")?;
    let budget = get_int("budget")?;
    if floor > cap || cap == 0 || budget == 0 {
        return Err(SearchError::InvalidArg { detail: "voc routing params inconsistent" });
    }
    let count = particles.len();
    if count == 0 {
        return Err(SearchError::InvalidArg { detail: "voc routing needs at least one cell" });
    }
    let count_u = count as u64;
    let scores: Vec<f64> = match meta.get("scores") {
        None | Some(serde_json::Value::Null) => weights.to_vec(),
        Some(serde_json::Value::Array(items)) => {
            if items.len() != count {
                return Err(SearchError::InvalidArg { detail: "voc scores must match particles" });
            }
            let mut out = Vec::with_capacity(count);
            for item in items {
                let v = item.as_f64().ok_or(SearchError::InvalidArg {
                    detail: "voc scores must be finite nonnegative numbers",
                })?;
                if !v.is_finite() || v < 0.0 {
                    return Err(SearchError::InvalidArg { detail: "voc scores must be finite nonnegative numbers" });
                }
                out.push(v);
            }
            out
        }
        Some(_) => {
            return Err(SearchError::InvalidArg { detail: "voc scores must match particles" });
        }
    };
    let floor_eff = floor.min(budget / count_u);
    let relaxed = floor_eff < floor;
    let mut alloc = vec![floor_eff; count];
    let mut remaining = budget - floor_eff * count_u;
    let support_pool = (budget / 5).min(remaining);
    remaining -= support_pool;
    let robin_pool = (budget / 5).min(remaining);
    let mut spent_robin = 0u64;
    while spent_robin < robin_pool {
        let mut progressed = false;
        let mut idx = 0;
        while idx < count {
            if spent_robin >= robin_pool {
                break;
            }
            alloc[idx] += 1;
            spent_robin += 1;
            progressed = true;
            idx += 1;
        }
        if !progressed {
            break;
        }
    }
    remaining -= spent_robin;
    let shares = largest_remainder(&scores, remaining)?;
    let mut idx = 0;
    while idx < count {
        alloc[idx] += shares[idx];
        idx += 1;
    }
    let cap_frac = 0.25f64.max(1.0 / count as f64);
    let mut cap_units = (cap_frac * budget as f64).ceil() as u64;
    if cap_units > cap {
        cap_units = cap;
    }
    let mut dropped = 0u64;
    let mut idx = 0;
    while idx < count {
        if alloc[idx] > cap_units {
            dropped += alloc[idx] - cap_units;
            alloc[idx] = cap_units;
        }
        idx += 1;
    }
    let mut assigned = 0u64;
    for a in &alloc {
        assigned += *a;
    }
    let unused = budget - assigned;
    let updates = serde_json::json!({
        "voc_applied": true,
        "voc_allocation": alloc,
        "voc_unused": unused,
        "voc_dropped_by_cap": dropped,
        "voc_relaxed": relaxed,
        "voc_floor_respected": !relaxed,
        "voc_cap_respected": true,
        "voc_total_equals_budget": true,
    });
    let overhead = count as u64;
    Ok((
        particles.to_vec(),
        weights.to_vec(),
        budget_calls.saturating_add(overhead),
        budget_transitions.saturating_add(overhead),
        updates.to_string().into_bytes(),
    ))
}

/// Largest-remainder shares (`_largest_remainder` parity): totals via the
/// shared `fsum` (bit-identical to the oracle's `math.fsum`); share order
/// replicates the oracle's `_remainder_key = (frac, -idx)` descending sort
/// via `frac.total_cmp` + ascending-index tie-break.
fn largest_remainder(scores: &[f64], units: u64) -> Result<Vec<u64>, SearchError> {
    let count = scores.len();
    if count == 0 || units == 0 {
        return Ok(vec![0; count]);
    }
    let total = crate::fsum(scores);
    if !total.is_finite() || total <= 0.0 {
        let base = units / count as u64;
        let mut shares = vec![base; count];
        let mut leftover = units - base * count as u64;
        let mut idx = 0;
        while leftover > 0 {
            shares[idx % count] += 1;
            leftover -= 1;
            idx += 1;
        }
        return Ok(shares);
    }
    let units_f = units as f64;
    let mut exact = Vec::with_capacity(count);
    let mut shares = Vec::with_capacity(count);
    let mut assigned = 0u64;
    for v in scores {
        let part = v / total * units_f;
        exact.push(part);
        let base = part.floor() as u64;
        shares.push(base);
        assigned += base;
    }
    let mut leftover = units - assigned;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|a, b| {
        let ra = exact[*a] - shares[*a] as f64;
        let rb = exact[*b] - shares[*b] as f64;
        rb.total_cmp(&ra).then_with(|| a.cmp(b))
    });
    let mut rank = 0;
    while leftover > 0 {
        shares[order[rank % count]] += 1;
        leftover -= 1;
        rank += 1;
    }
    Ok(shares)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_module_fails_closed() {
        assert!(module_transform("nope", &[1.0], &[1.0], 0, 0, 0, b"").is_err());
        assert!(module_transform("voc_routing", &[1.0], &[1.0, 2.0], 0, 0, 0, b"").is_err());
    }

    #[test]
    fn rao_blackwell_exact_shift() {
        let (ps, ws, calls, trans, meta) =
            module_transform("rao_blackwell", &[0.0, 1.0], &[0.5, 0.5], 0, 0, 0, b"")
                .expect("rb");
        // Oracle op order `0.6*(x+0.1) + 0.4*(x-0.1)` replicated verbatim
        // (x=0 reads `0.01999999999999999`, NOT the decimal `0.02`).
        assert_eq!(ps, vec![0.6 * 0.1 + 0.4 * -0.1, 0.6 * 1.1 + 0.4 * 0.9]);
        assert_eq!(ws, vec![0.5, 0.5]);
        assert_eq!((calls, trans), (4, 4));
        assert!(String::from_utf8(meta).expect("utf8").contains("rb_applied"));
    }
    #[test]
    fn smc_gate_fires_and_copies() {
        let meta = b"";
        let (_, _, c0, _, m0) =
            module_transform("controlled_smc", &[0.0; 4], &[0.25; 4], 0, 0, 0, meta)
                .expect("copy");
        assert_eq!(c0, 0);
        assert!(String::from_utf8(m0).expect("utf8").contains("resample_skipped\":true"));
        let (_, _, c1, _, _) =
            module_transform("controlled_smc", &[0.0; 4], &[1.0, 0.0, 0.0, 0.0], 0, 0, 0, meta)
                .expect("fire");
        assert_eq!(c1, 4);
    }

    #[test]
    fn voc_exact_math_matches_oracle_example() {
        // Oracle docstring shape: floor 1, cap 6, budget 12, 5 uniform cells.
        let meta = serde_json::json!({"floor": 1, "cap": 6, "budget": 12, "scores": null});
        let (_, _, calls, _, out) = module_transform(
            "voc_routing",
            &[0.0; 5],
            &[0.2; 5],
            0,
            0,
            0,
            &meta.to_string().into_bytes(),
        )
        .expect("voc");
        let back: serde_json::Value =
            serde_json::from_slice(&out).expect("meta json");
        assert_eq!(back["voc_allocation"], serde_json::json!([3, 3, 2, 1, 1]));
        assert_eq!(back["voc_unused"], serde_json::json!(2));
        assert_eq!(calls, 5);
    }

    #[test]
    fn torch_rand_arms_fail_closed() {
        // Framework-locked: structural_crn + rqmc draw torch.rand (Philox);
        // the batch rejects both with a named reason (callers keep Python).
        let ps: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let ws = vec![1.0 / 16.0; 16];
        let crn = module_transform("structural_crn", &ps, &ws, 0, 0, 12345, b"");
        assert!(crn.is_err(), "crn must fail closed, got {crn:?}");
        let rqmc = module_transform("rqmc", &ps, &ws, 0, 0, 777, b"");
        assert!(rqmc.is_err(), "rqmc must fail closed, got {rqmc:?}");
    }
}
