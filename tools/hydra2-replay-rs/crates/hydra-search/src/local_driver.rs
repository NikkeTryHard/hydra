//! `local_driver`: local-resolving batch driver (resolving loop, no selection).
//!
//! Owns the `local_search.py:234-348` resolving loop: per-iteration world
//! rotation, per-depth actor traversal, regret/hedge/fictitious-play updates,
//! averaging accumulation, deterministic path sampling, and next-node mixing.
//! Same batch shape as `ismcts_driver`/`gumbel_driver` (envelope in, table
//! dumps out, one bridge call per search, GIL released, no per-step crossing).
//!
//! Python precomputes the envelope ONCE per search (sampled worlds in oracle
//! iteration order, memoized per-(world, actor) base info keys, per-world leaf
//! vectors, seeded init-table entries, per-step sampling floats) and Rust
//! replays the oracle loop bit-identically: iterations `1..=N`, depths
//! `0..horizon`, `actor = (root + depth) % 4`.
//!
//! Oracle-fidelity notes (`local_search.py` / `local_strategy.py` line refs):
//! - Base info keys (`247-259`): `obs = world_actor_observation(world, actor)`
//!   then `info_key_for_actor_observation(obs)`, mixed per depth as
//!   `_digest(f"{base}:{path}")` (`257`). All actor-observation construction
//!   stays Python (contracts-owned); only the resulting `sha256:` strings
//!   cross — the firewall (ActorObservation-only, info-keys-only) holds.
//! - Leaf vectors (`237-239`): `leaf_vector_replay(world, leaf_model)` with
//!   the `preserves_vector_returns` gate. Python validates once per distinct
//!   world and crosses the 4 floats; this driver never builds a vector.
//! - Utility offsets (`273-285`): `int(sha256(f"{wid}:{aid}:{depth}")
//!   .hexdigest()[:4], 16) % 100 / 1000.0 - 0.05` added to `leaf[actor]`.
//!   First-two-digest-bytes big-endian replicated verbatim.
//! - Updates (`291-310`): regret accumulates `u - ev` then normalizes positive
//!   regrets (`s > 1e-12`, else uniform); hedge accumulates `u` into `q` then
//!   max-subtract softmax with `eta = 1.0`; fictitious play tracks the
//!   first-max best response with `count`/`w = 1.0` then renormalizes
//!   (redundant `/ s` included — same op order, same bits).
//! - Averaging (`313-316`): weight `1.0` uniform / `float(it)` linear,
//!   accumulated per key alongside the loop.
//! - Sampling (`323-333`): fixed float per step (the oracle reads
//!   `rng.random()` when present else `0.5`; Python resolves the duck-type
//!   and crosses the floats), first index with `cum > r` wins, default last.
//! - Next node (`336`): `_digest(f"{path}:{depth}:{chosen_aid}")`.
//! - Averaged tables (`350-363`): per-key `acc / w`, renormalize `/ s`
//!   (`s > 0`, else uniform); untouched seeds keep the seeded dist with NO
//!   visit-count entry. Visit counts mirror `table.visit_counts` exactly.
//! - Tie-break selection (`375-427`: greedy/temperature/value_break) stays
//!   Python: it runs once per search over the returned root average (zero
//!   hotspot), and the temperature softmax `exp` would fork 1 ulp across
//!   libm implementations. Counters/telemetry assembly stay Python too.
//!
//! Every shape violation fails closed (`SearchError`, never a default). No
//! clock, no global RNG, no Python API, no canon (keys cross as strings).
//! Table dumps serialize crate-side (serde_json lives here, never in the
//! bridge — same convention as the ISMCTS `tree_json` dump).

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::local::{LOCAL_MAX_HORIZON, LOCAL_MAX_ITERS};

/// Regret uniform-fallback gate (`local_strategy.py:147`: `s > 1e-12`).
const REGRET_EPS: f64 = 1e-12;
/// Action-utility offset denominator (`local_search.py:283-284`).
const OFFSET_MOD: u16 = 100;

/// One iteration's precomputed world: identity, leaf vector, base info keys.
#[derive(Debug, Clone, Deserialize)]
struct LocalWorldJson {
    /// World id text (offset salt).
    world_id: String,
    /// Validated 4-seat leaf vector for this world's `leaf_model`.
    leaf: Vec<f64>,
    /// Base info keys for actors 0..3 (`info_key_for_actor_observation`);
    /// `null` where the actor observation build fails — the oracle then
    /// falls back to `_digest(f"{path}:actor{actor}")` with the live path.
    base: Vec<Option<String>>,
}

/// One seeded init-table entry: `(actor, info_hash)` + starting distribution.
#[derive(Debug, Clone, Deserialize)]
struct LocalInitJson {
    /// Actor seat 0..3.
    actor: u32,
    /// Information-set hash (`sha256:` text).
    info: String,
    /// Starting distribution over `ab_order` (seeded, unchecked values —
    /// the oracle assigns seeds without validation).
    dist: Vec<f64>,
}

/// Full resolving batch envelope: worlds, seeds, floats, frozen params.
#[derive(Debug, Clone, Deserialize)]
struct LocalBatchJson {
    /// Abstract action ids in distribution order (sorted).
    ab_order: Vec<u32>,
    /// Frozen horizon (`1..=16`).
    horizon: u32,
    /// Frozen iteration count (`1..=1024`).
    iterations: u32,
    /// `regret_matching` | `hedge` | `fictitious_play`.
    update_rule: String,
    /// `uniform` | `linear`.
    averaging: String,
    /// Root seat 0..3.
    root_actor: u32,
    /// Root information-set hash (ensured without a visit-count entry —
    /// the oracle seeds `table.table` only; `table.visit_counts` gains the
    /// key solely on loop encounter).
    root_info: String,
    /// One world per iteration in oracle iteration order.
    iters: Vec<LocalWorldJson>,
    /// Seeded `(actor, info)` entries (visits start at 0, entry present).
    init: Vec<LocalInitJson>,
    /// Per-step sampling floats, iteration-major (`iterations * horizon`).
    sampling: Vec<f64>,
    /// Root public-history hash (path start).
    public_history_hash: String,
}

/// One dumped table row: `(actor, info)` distribution + visits.
#[derive(Debug, Clone)]
pub struct LocalTableRow {
    /// Actor seat.
    pub actor: u32,
    /// Information-set hash.
    pub info: String,
    /// Distribution over `ab_order`.
    pub dist: Vec<f64>,
    /// Visit count (`None` for averaged rows the oracle leaves unset).
    pub visits: Option<u64>,
}

/// Resolving outcome: current + averaged table dumps in first-touch order.
#[derive(Debug, Clone)]
pub struct LocalResolvingOut {
    /// Current strategy table rows.
    pub table: Vec<LocalTableRow>,
    /// Averaged table rows.
    pub avg: Vec<LocalTableRow>,
}

/// Lowercase hex of a digest (nibble loop, mirrors `belief::hex_of`).
fn hex_of(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    let mut i = 0;
    while i < digest.len() {
        let byte = digest[i];
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0F) as usize] as char);
        i += 1;
    }
    out
}

/// `_digest(s)` parity (`local_abstraction.py:63-64`): plain sha256 hex with
/// the `sha256:` prefix (digest-shape validation is vacuous — hex output).
fn digest_str(text: &str) -> String {
    format!("sha256:{}", hex_of(Sha256::digest(text.as_bytes()).as_slice()))
}

/// Action-utility offset (`local_search.py:276-284`): first two digest bytes
/// big-endian `% 100 / 1000.0 - 0.05` over `f"{wid}:{aid}:{depth}"`.
fn utility_offset(world_id: &str, aid: u32, depth: u32) -> f64 {
    let payload = format!("{world_id}:{aid}:{depth}");
    let sum = Sha256::digest(payload.as_bytes());
    let word = u16::from_be_bytes([sum[0], sum[1]]);
    (word % OFFSET_MOD) as f64 / 1000.0 - 0.05
}

/// Uniform distribution over `n` actions.
fn uniform(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// Key index lookup over the parallel key/state vectors.
fn key_index(keys: &[(u32, String)], actor: u32, info: &str) -> Option<usize> {
    let mut i = 0;
    while i < keys.len() {
        if keys[i].0 == actor && keys[i].1 == info {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Parse + validate the envelope (fail-closed shapes only, never defaults).
fn parse_batch(raw: &[u8]) -> Result<LocalBatchJson, SearchError> {
    let batch: LocalBatchJson =
        serde_json::from_slice(raw).map_err(|_| SearchError::InvalidArg {
            detail: "local batch must be a JSON envelope",
        })?;
    let n = batch.ab_order.len();
    if n == 0 {
        return Err(SearchError::InvalidArg {
            detail: "ab_order must be non-empty",
        });
    }
    if batch.horizon == 0 || batch.horizon > LOCAL_MAX_HORIZON {
        return Err(SearchError::InvalidArg {
            detail: "horizon must be 1..16",
        });
    }
    if batch.iterations == 0 || batch.iterations > LOCAL_MAX_ITERS {
        return Err(SearchError::InvalidArg {
            detail: "iterations must be 1..1024",
        });
    }
    if batch.update_rule != "regret_matching"
        && batch.update_rule != "hedge"
        && batch.update_rule != "fictitious_play"
    {
        return Err(SearchError::InvalidArg {
            detail: "unknown update_rule",
        });
    }
    if batch.averaging != "uniform" && batch.averaging != "linear" {
        return Err(SearchError::InvalidArg {
            detail: "unknown averaging rule",
        });
    }
    if batch.root_actor >= 4 {
        return Err(SearchError::InvalidSeat {
            seat: batch.root_actor,
        });
    }
    if batch.iters.len() as u32 != batch.iterations {
        return Err(SearchError::InvalidArg {
            detail: "iters must carry one world per iteration",
        });
    }
    if batch.sampling.len() as u64 != u64::from(batch.iterations) * u64::from(batch.horizon) {
        return Err(SearchError::InvalidArg {
            detail: "sampling must carry one float per step",
        });
    }
    for world in &batch.iters {
        if world.leaf.len() != 4 {
            return Err(SearchError::InvalidArg {
                detail: "leaf must hold 4 entries",
            });
        }
        for v in &world.leaf {
            if !v.is_finite() {
                return Err(SearchError::NonFinite {
                    context: "local leaf",
                });
            }
        }
        if world.base.len() != 4 {
            return Err(SearchError::InvalidArg {
                detail: "base must hold 4 actor keys",
            });
        }
    }
    for entry in &batch.init {
        if entry.actor >= 4 {
            return Err(SearchError::InvalidSeat {
                seat: entry.actor,
            });
        }
        if entry.dist.len() != n {
            return Err(SearchError::InvalidArg {
                detail: "init dist must match ab_order",
            });
        }
    }
    Ok(batch)
}

/// Run the resolving loop over one validated envelope (`local_search.py:234-348`
/// fidelity; tie-break selection is caller-side, see module docs).
pub fn local_resolving_batch(raw: &[u8]) -> Result<LocalResolvingOut, SearchError> {
    let batch = parse_batch(raw)?;
    let n = batch.ab_order.len();
    let horizon = batch.horizon;
    let iterations = batch.iterations;
    let regret = batch.update_rule == "regret_matching";
    let hedge = !regret && batch.update_rule == "hedge";
    let linear = batch.averaging == "linear";
    let root = batch.root_actor;

    // Per-key state, first-touch ordered (mirrors the oracle dicts).
    let mut keys: Vec<(u32, String)> = Vec::new();
    let mut table: Vec<Vec<f64>> = Vec::new();
    let mut regrets: Vec<Vec<f64>> = Vec::new();
    let mut quals: Vec<Vec<f64>> = Vec::new();
    let mut accum: Vec<Vec<f64>> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut visits: Vec<Option<u64>> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    // Seed from the init entries (`_init_tables` output verbatim; visits
    // entries start present at 0).
    for entry in &batch.init {
        if key_index(&keys, entry.actor, &entry.info).is_none() {
            keys.push((entry.actor, entry.info.clone()));
            table.push(entry.dist.clone());
            regrets.push(vec![0.0; n]);
            quals.push(vec![0.0; n]);
            accum.push(vec![0.0; n]);
            weights.push(0.0);
            visits.push(Some(0));
            counts.push(0);
        }
    }
    // Ensure the root entry (oracle seeds `table.table` only — no
    // visit-count entry until a loop encounter touches the key).
    if key_index(&keys, root, &batch.root_info).is_none() {
        keys.push((root, batch.root_info.clone()));
        table.push(uniform(n));
        regrets.push(vec![0.0; n]);
        quals.push(vec![0.0; n]);
        accum.push(vec![0.0; n]);
        weights.push(0.0);
        visits.push(None);
        counts.push(0);
    }

    let mut it = 1u32;
    while it <= iterations {
        let world = &batch.iters[(it as usize - 1) % batch.iters.len()];
        let mut path = batch.public_history_hash.clone();
        let mut depth = 0u32;
        while depth < horizon {
            let actor = (root + depth) % 4;
            let info = match &world.base[actor as usize] {
                Some(base) => digest_str(&format!("{base}:{path}")),
                None => digest_str(&format!("{path}:actor{actor}")),
            };
            let k = match key_index(&keys, actor, &info) {
                Some(k) => k,
                None => {
                    keys.push((actor, info.clone()));
                    table.push(uniform(n));
                    regrets.push(vec![0.0; n]);
                    quals.push(vec![0.0; n]);
                    accum.push(vec![0.0; n]);
                    weights.push(0.0);
                    visits.push(None);
                    counts.push(0);
                    keys.len() - 1
                }
            };
            let strata = table[k].clone();
            let leaf_actor = world.leaf[actor as usize];
            // Utilities + expected value under the current strategy.
            let mut utils = Vec::with_capacity(n);
            let mut ev = 0.0;
            let mut i = 0;
            while i < n {
                let u = leaf_actor + utility_offset(&world.world_id, batch.ab_order[i], depth);
                utils.push(u);
                ev += strata[i] * u;
                i += 1;
            }
            // Frozen update rule (`local_strategy.py:139-206` verbatim).
            let mut next = Vec::with_capacity(n);
            if regret {
                let reg = &mut regrets[k];
                let mut i = 0;
                while i < n {
                    reg[i] += utils[i] - ev;
                    i += 1;
                }
                let mut s = 0.0;
                for r in reg.iter() {
                    s += r.max(0.0);
                }
                if s > REGRET_EPS {
                    for r in reg.iter() {
                        next.push(r.max(0.0) / s);
                    }
                } else {
                    next = uniform(n);
                }
            } else if hedge {
                let q = &mut quals[k];
                let mut i = 0;
                while i < n {
                    q[i] += utils[i];
                    i += 1;
                }
                let mut m = f64::NEG_INFINITY;
                for v in q.iter() {
                    if !v.is_finite() {
                        return Err(SearchError::NonFinite {
                            context: "local hedge q",
                        });
                    }
                    m = m.max(*v);
                }
                let mut exps = Vec::with_capacity(n);
                let mut s = 0.0;
                for v in q.iter() {
                    let e = (1.0 * (*v - m)).exp();
                    if !e.is_finite() {
                        return Err(SearchError::NonFinite {
                            context: "local hedge exp",
                        });
                    }
                    exps.push(e);
                    s += e;
                }
                for e in exps {
                    next.push(e / s);
                }
            } else {
                // Fictitious play: first-max best response, count-weighted.
                let mut br = 0;
                let mut best = utils[0];
                let mut i = 1;
                while i < n {
                    if utils[i] > best {
                        best = utils[i];
                        br = i;
                    }
                    i += 1;
                }
                let cnt = counts[k] as f64;
                let total = cnt + 1.0;
                let mut s = 0.0;
                let mut i = 0;
                while i < n {
                    let prev = if counts[k] > 0 { strata[i] * cnt / total } else { 0.0 };
                    let add = if i == br { 1.0 / total } else { 0.0 };
                    next.push(prev + add);
                    s += prev + add;
                    i += 1;
                }
                if s > 0.0 {
                    let mut i = 0;
                    while i < n {
                        next[i] /= s;
                        i += 1;
                    }
                } else {
                    next = uniform(n);
                }
                counts[k] += 1;
            }
            table[k] = next.clone();
            // Averaging accumulator (`313-316`).
            let w = if linear { it as f64 } else { 1.0 };
            let mut i = 0;
            while i < n {
                accum[k][i] += next[i] * w;
                i += 1;
            }
            weights[k] += w;
            visits[k] = Some(visits[k].unwrap_or(0) + 1);
            // Deterministic path sample (`323-333`): first `cum > r`, else last.
            let r = batch.sampling[(it as usize - 1) * horizon as usize + depth as usize];
            let mut cum = 0.0;
            let mut chosen = n - 1;
            let mut i = 0;
            while i < n {
                cum += next[i];
                if r < cum {
                    chosen = i;
                    break;
                }
                i += 1;
            }
            let aid = batch.ab_order[chosen];
            path = digest_str(&format!("{path}:{depth}:{aid}"));
            depth += 1;
        }
        it += 1;
    }

    // Dump current + averaged tables (`350-363` fidelity).
    let mut table_rows = Vec::with_capacity(keys.len());
    let mut avg_rows = Vec::with_capacity(keys.len());
    let mut k = 0;
    while k < keys.len() {
        table_rows.push(LocalTableRow {
            actor: keys[k].0,
            info: keys[k].1.clone(),
            dist: table[k].clone(),
            visits: visits[k],
        });
        if weights[k] > 0.0 {
            let mut avg = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                avg.push(accum[k][i] / weights[k]);
                i += 1;
            }
            let s: f64 = avg.iter().sum();
            let dist = if s > 0.0 {
                avg.into_iter().map(|v| v / s).collect()
            } else {
                uniform(n)
            };
            avg_rows.push(LocalTableRow {
                actor: keys[k].0,
                info: keys[k].1.clone(),
                dist,
                visits: visits[k],
            });
        } else {
            avg_rows.push(LocalTableRow {
                actor: keys[k].0,
                info: keys[k].1.clone(),
                dist: table[k].clone(),
                visits: None,
            });
        }
        k += 1;
    }
    Ok(LocalResolvingOut {
        table: table_rows,
        avg: avg_rows,
    })
}

/// Serialize one table dump to JSON rows (`[actor, info, [p...], visits]`;
/// visits renders `null` where the oracle leaves it unset).
fn dump_rows(rows: &[LocalTableRow]) -> serde_json::Value {
    let mut out: Vec<serde_json::Value> = Vec::with_capacity(rows.len());
    let mut i = 0;
    while i < rows.len() {
        let row = &rows[i];
        out.push(serde_json::json!([row.actor, row.info, row.dist, row.visits]));
        i += 1;
    }
    serde_json::Value::Array(out)
}

/// Batch entry point for the bridge: parse + replay + serialize the dumps.
/// Returns `(table_json, avg_json)`; the bridge only wraps the strings.
pub fn local_resolving_batch_json(raw: &[u8]) -> Result<(String, String), SearchError> {
    let out = local_resolving_batch(raw)?;
    Ok((
        dump_rows(&out.table).to_string(),
        dump_rows(&out.avg).to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(rule: &str) -> Vec<u8> {
        let worlds: Vec<String> = (0..4)
            .map(|i| {
                format!(
                    "{{\"world_id\":\"w{i}\",\"leaf\":[0.1,0.2,0.3,0.4],\
                     \"base\":[\"sha256:{:064x}\",\"sha256:{:064x}\",\"sha256:{:064x}\",\"sha256:{:064x}\"]}}",
                    i, i, i, i
                )
            })
            .collect();
        format!(
            "{{\"ab_order\":[0,1],\"horizon\":2,\"iterations\":4,\"update_rule\":\"{rule}\",\
             \"averaging\":\"uniform\",\"root_actor\":0,\"root_info\":\"sha256:{:064x}\",\"iters\":[{}],\
             \"init\":[{{\"actor\":0,\"info\":\"sha256:{:064x}\",\"dist\":[0.5,0.5]}}],\
             \"sampling\":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],\
             \"public_history_hash\":\"sha256:{:064x}\"}}",
            5,
            worlds.join(","),
            9,
            7,
        )
        .into_bytes()
    }

    #[test]
    fn offset_matches_oracle_arithmetic() {
        // int(sha256("w0:1:2").hexdigest()[:4], 16) % 100 / 1000.0 - 0.05.
        let sum = Sha256::digest(b"w0:1:2");
        let word = u16::from_be_bytes([sum[0], sum[1]]);
        let expected = (word % 100) as f64 / 1000.0 - 0.05;
        assert!((utility_offset("w0", 1, 2) - expected).abs() == 0.0);
    }

    #[test]
    fn digest_shape_matches_digest_text() {
        let d = digest_str("abc");
        assert!(d.starts_with("sha256:"));
        assert_eq!(d.len(), 7 + 64);
    }

    #[test]
    fn rejects_bad_envelopes() {
        assert!(parse_batch(b"not json").is_err());
        let mut bad = envelope("bogus_rule");
        assert!(parse_batch(&bad).is_err());
        bad = envelope("regret_matching");
        let mut v: serde_json::Value = serde_json::from_slice(&bad).unwrap();
        v["horizon"] = serde_json::Value::from(17);
        assert!(parse_batch(&serde_json::to_vec(&v).unwrap()).is_err());
        v["horizon"] = serde_json::Value::from(2);
        v["sampling"] = serde_json::Value::Array(vec![]);
        assert!(parse_batch(&serde_json::to_vec(&v).unwrap()).is_err());
    }

    #[test]
    fn loop_runs_all_three_rules() {
        for rule in ["regret_matching", "hedge", "fictitious_play"] {
            let out = local_resolving_batch(&envelope(rule)).expect(rule);
            assert_eq!(out.table.len(), out.avg.len());
            assert!(!out.table.is_empty());
            for row in out.table.iter().chain(out.avg.iter()) {
                assert_eq!(row.dist.len(), 2);
                let s: f64 = row.dist.iter().sum();
                assert!((s - 1.0).abs() < 1e-9, "row {row:?} sums to {s}");
            }
        }
    }

    #[test]
    fn untouched_keys_mirror_oracle_presence() {
        let out = local_resolving_batch(&envelope("regret_matching")).expect("run");
        let has = |actor: u32, tail: char| {
            let info = format!("sha256:{:064x}", tail as u8);
            let cur = out
                .table
                .iter()
                .find(|r| r.actor == actor && r.info == info)
                .expect("seed present in table");
            let avg = out
                .avg
                .iter()
                .find(|r| r.actor == actor && r.info == info)
                .expect("seed present in avg");
            (cur.visits, avg.visits)
        };
        // Init seed untouched: table entry present at 0, avg entry unset.
        assert_eq!(has(0, 9 as char), (Some(0), None));
        // Root ensure untouched: table entry unset, avg entry unset.
        assert_eq!(has(0, 5 as char), (None, None));
    }
}
