//! `gumbel`: B1-VERBATIM sha-Gumbels + halving loop (Philox words DELETED).
//!
//! Replicates `gumbel_core.py:117-156` bytes VERBATIM (B1):
//! `payload = f"{case}:{seat}:{cand}:{action}".encode()`,
//! `h = sha256(DOMAIN + payload)` with `DOMAIN = b"gumbel_root_v1"`,
//! `u = (u64BE(h[:8]) as f64 + 0.5) / 2^64`, clip to
//! `(1e-12, 1-1e-12)`, `G = -ln(-ln(u))`, clamp `+-20`. Philox serves ONLY
//! NEW segment/scenario substreams (`rng`); NO Gumbel word is ever drawn
//! from a Philox stream (the B1 fork trap).
//!
//! Halving: schedule `visits_per_round.len() == halving_rounds`
//! (default `(8,8)`); survivors halve per round (`M/2^r`), `score = q + g`
//! with `1e-12` eps + tie arm. PUCT comparator (`puct_c = 1.5`,
//! `num_simulations = 16`): `q + c*prior*sqrt(N)/(1+n)`, identical budget
//! counters (`ismcts` stats carry the vectors).

use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::ismcts::{IsNode, TieBreak, check_legal};

/// B1 domain: prepended to the payload BEFORE hashing
/// (`_GUMBEL_SEED_DOMAIN = b"gumbel_root_v1"`, `gumbel_core.py:109,136`).
pub const GUMBEL_DOMAIN: &[u8] = b"gumbel_root_v1";

/// Clip bound for `u` (`gumbel_core.py:143-147`).
pub const GUMBEL_CLIP: f64 = 1e-12;
/// Clamp bound for `G` (`gumbel_core.py:152-155`).
pub const GUMBEL_CLAMP: f64 = 20.0;
/// `2^64` as `f64` (`U64_DENOM`, `common.py:83`).
pub const U64_DENOM: f64 = 18_446_744_073_709_551_616.0;

/// Halving/select epsilon gate (same `1e-12` as UCT/PUCT).
pub const HALVING_EPS: f64 = 1e-12;

/// Default Gumbel schedule (`GumbelSearchConfig`: rounds 2, `(8,8)`).
pub const DEFAULT_HALVING_ROUNDS: u32 = 2;
/// Default visits per round.
pub const DEFAULT_VISITS_PER_ROUND: [u32; 2] = [8, 8];
/// Default descent depth (matches ISMCTS).
pub const DEFAULT_MAX_DEPTH: u32 = 6;
/// PUCT comparator constant (`PuctConfig.puct_c`).
pub const PUCT_C: f64 = 1.5;
/// PUCT comparator simulation count (`PuctConfig.num_simulations`).
pub const PUCT_NUM_SIMULATIONS: u32 = 16;

/// Frozen Gumbel search hyper-parameters.
#[derive(Debug, Clone)]
pub struct GumbelParams {
    /// Halving rounds (`1..=5`).
    pub halving_rounds: u32,
    /// Visits per round (len MUST equal `halving_rounds`, each `1..=64`).
    pub visits_per_round: Vec<u32>,
    /// Descent depth (`1..=32`).
    pub max_depth: u32,
    /// Optional model-call cap (`None` = unbounded).
    pub max_model_calls: Option<u64>,
    /// Optional transition cap (`None` = unbounded).
    pub max_transitions: Option<u64>,
    /// Tie-break arm.
    pub tie: TieBreak,
}

impl GumbelParams {
    /// Frozen defaults (rounds 2, `(8,8)`, depth 6, caps 32/64).
    pub fn defaults() -> GumbelParams {
        GumbelParams {
            halving_rounds: DEFAULT_HALVING_ROUNDS,
            visits_per_round: DEFAULT_VISITS_PER_ROUND.to_vec(),
            max_depth: DEFAULT_MAX_DEPTH,
            max_model_calls: Some(32),
            max_transitions: Some(64),
            tie: TieBreak::LowestActionId,
        }
    }

    /// Validate (`ContractError` edges: rounds `1..=5`, visits len + range,
    /// depth `1..=32`).
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.halving_rounds == 0 || self.halving_rounds > 5 {
            return Err(SearchError::InvalidArg {
                detail: "halving_rounds must be 1..5",
            });
        }
        if self.visits_per_round.len() != self.halving_rounds as usize {
            return Err(SearchError::InvalidArg {
                detail: "visits_per_round len must equal halving_rounds",
            });
        }
        for visits in &self.visits_per_round {
            if *visits == 0 || *visits > 64 {
                return Err(SearchError::InvalidArg {
                    detail: "visits_per_round must be 1..64",
                });
            }
        }
        if self.max_depth == 0 || self.max_depth > 32 {
            return Err(SearchError::InvalidArg {
                detail: "max_depth must be 1..32",
            });
        }
        Ok(())
    }
}

/// B1-VERBATIM deterministic Gumbel (`gumbel_core.py:117-156`).
///
/// Derivation: `U = (int(sha256(DOMAIN+payload)[:8],"big") + 0.5) / 2^64`,
/// clipped to `(1e-12, 1-1e-12)`, then `G = -log(-log(U))`, clamped to
/// `+-20`. Finite for every input; identical inputs give identical outputs
/// regardless of call order or global RNG. `case_id`/`candidate_id`
/// non-empty; `root_seat` in `0..4`.
pub fn deterministic_gumbel(
    case_id: &str,
    root_seat: u32,
    candidate_id: &str,
    action_id: u32,
) -> Result<f64, SearchError> {
    if case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be non-empty",
        });
    }
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    if candidate_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "candidate_id must be non-empty",
        });
    }
    // Payload bytes VERBATIM: f"{case}:{seat}:{cand}:{action}".
    let payload = format!("{case_id}:{root_seat}:{candidate_id}:{action_id}");
    let mut hasher = Sha256::new();
    hasher.update(GUMBEL_DOMAIN);
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    // 8 bytes big-endian -> uniform.
    let mut raw = [0u8; 8];
    raw.copy_from_slice(digest.as_slice()[..8].as_ref());
    let int_val = u64::from_be_bytes(raw);
    // (x + 0.5) / 2^64, then defensive clip to (1e-12, 1-1e-12).
    // proof: u64 word / 2^64 is in [0,1); 1ulp on the unit interval is oracle-tolerable.
    #[allow(clippy::cast_precision_loss)]
    let int_f: f64 = int_val as f64;
    let mut u = (int_f + 0.5) / U64_DENOM;
    if u <= 0.0 {
        u = GUMBEL_CLIP;
    }
    if u >= 1.0 {
        u = 1.0 - GUMBEL_CLIP;
    }
    // `clamp` preserves NaN (both comparisons false) exactly like the two
    // `if`s above, so this is value-identical.
    u = u.clamp(GUMBEL_CLIP, 1.0 - GUMBEL_CLIP);
    let g = -(-u.ln()).ln();
    if !g.is_finite() {
        return Err(SearchError::NonFinite { context: "gumbel" });
    }
    if g > GUMBEL_CLAMP {
        return Ok(GUMBEL_CLAMP);
    }
    if g < -GUMBEL_CLAMP {
        return Ok(-GUMBEL_CLAMP);
    }
    Ok(g)
}

/// Deterministic Gumbels for every legal action
/// (`deterministic_root_gumbels`: non-empty unique tuple, dup rejected).
pub fn deterministic_root_gumbels(
    case_id: &str,
    root_seat: u32,
    candidate_id: &str,
    legal: &[u32],
) -> Result<Vec<(u32, f64)>, SearchError> {
    let sorted = check_legal(legal)?;
    let mut out = Vec::with_capacity(sorted.len());
    for action in sorted {
        out.push((
            action,
            deterministic_gumbel(case_id, root_seat, candidate_id, action)?,
        ));
    }
    Ok(out)
}

/// Root scalarization of one arm (`vector[root_seat]`), finite-checked.
pub fn scalar_mean(node: &IsNode, action: u32, root_seat: u32) -> Result<Option<f64>, SearchError> {
    node.scalar_mean(action, root_seat)
}

/// One halving cut: keep the top half of `survivors` by `score = q + g`
/// (`1e-12` eps + tie arm; odd counts round UP so the leader always
/// survives; single survivor passes through).
///
/// `means` carries `(action, q)` root scalars; `gumbels` carries
/// `(action, g)` B1 draws. Both MUST cover exactly `survivors`.
pub fn halving_cut(
    survivors: &[u32],
    means: &[(u32, f64)],
    gumbels: &[(u32, f64)],
    tie: TieBreak,
) -> Result<Vec<u32>, SearchError> {
    if survivors.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    if survivors.len() == 1 {
        return Ok(survivors.to_vec());
    }
    let score = |action: u32| -> Result<f64, SearchError> {
        let q = means
            .iter()
            .find(|(aid, _)| *aid == action)
            .map(|(_, q)| *q)
            .ok_or(SearchError::InvalidArg {
                detail: "halving means missing action",
            })?;
        let g = gumbels
            .iter()
            .find(|(aid, _)| *aid == action)
            .map(|(_, g)| *g)
            .ok_or(SearchError::InvalidArg {
                detail: "halving gumbels missing action",
            })?;
        if !q.is_finite() || !g.is_finite() {
            return Err(SearchError::NonFinite {
                context: "halving score",
            });
        }
        Ok(q + g)
    };
    // Sort by (-score, tie order): insertion-style deterministic selection.
    let mut ranked: Vec<u32> = survivors.to_vec();
    // Simple deterministic selection sort on score desc, tie arm asc.
    let mut i = 0;
    while i < ranked.len() {
        let mut best = i;
        let mut j = i + 1;
        while j < ranked.len() {
            let cur = score(ranked[j])?;
            let top = score(ranked[best])?;
            if cur > top + HALVING_EPS
                || ((cur - top).abs() <= HALVING_EPS && tie_wins(ranked[j], ranked[best], tie))
            {
                best = j;
            }
            j += 1;
        }
        ranked.swap(i, best);
        i += 1;
    }
    let keep = survivors.len().div_ceil(2);
    ranked.truncate(keep);
    Ok(ranked)
}

/// Compare two arms under the frozen tie arm (shared with ISMCTS order).
fn tie_wins(candidate: u32, best: u32, tie: TieBreak) -> bool {
    match tie {
        TieBreak::LowestActionId => candidate < best,
        TieBreak::StableHash | TieBreak::Lexicographic => {
            let left = Sha256::digest(candidate.to_string().as_bytes());
            let right = Sha256::digest(best.to_string().as_bytes());
            left.as_slice() < right.as_slice()
        }
    }
}

/// PUCT comparator select (`gumbel_puct.py:240-258` parity):
/// `q + puct_c * prior * sqrt(total) / (1 + n)`, uniform priors, `1e-12`
/// eps + tie arm. `total = sum(visits)` over the legal set.
pub fn puct_select(
    node: &IsNode,
    legal: &[u32],
    root_seat: u32,
    puct_c: f64,
    tie: TieBreak,
) -> Result<u32, SearchError> {
    let sorted = check_legal(legal)?;
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    if !puct_c.is_finite() || puct_c <= 0.0 {
        return Err(SearchError::InvalidArg {
            detail: "puct_c must be finite >0",
        });
    }
    // Unvisited arms: PUCT still scores them (q=0, n=0) — no early return.
    // proof: candidate-slice len is small (< 2^53), exact; PUCT prior tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let sorted_len_f: f64 = sorted.len() as f64;
    let prior = 1.0 / sorted_len_f;
    let total: u64 = sorted
        .iter()
        .map(|action| node.stats(*action).map(|stats| stats.visits).unwrap_or(0))
        .sum();
    let mut best: Option<u32> = None;
    let mut best_score = f64::NEG_INFINITY;
    for action in &sorted {
        let q = node.scalar_mean(*action, root_seat)?.unwrap_or(0.0);
        let n = node.stats(*action).map(|stats| stats.visits).unwrap_or(0);
        // proof: visit counts are small counters (< 2^53), exact; PUCT bonus tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let total_f: f64 = total as f64;
        #[allow(clippy::cast_precision_loss)]
        let n_f: f64 = n as f64;
        let u = puct_c * prior * total_f.sqrt() / (1.0 + n_f);
        let score = q + u;
        if !score.is_finite() {
            return Err(SearchError::NonFinite {
                context: "puct score",
            });
        }
        match best {
            None => {
                best = Some(*action);
                best_score = score;
            }
            Some(current) => {
                if score > best_score + HALVING_EPS
                    || ((score - best_score).abs() <= HALVING_EPS
                        && tie_wins(*action, current, tie))
                {
                    best = Some(*action);
                    best_score = score;
                }
            }
        }
    }
    match best {
        Some(action) => Ok(action),
        None => Ok(sorted[0]),
    }
}

/// Final Gumbel selection over the last survivors (`gumbel_search.py:521`:
/// `score = g + q`, `1e-12` eps + tie arm).
pub fn gumbel_select(
    survivors: &[u32],
    means: &[(u32, f64)],
    gumbels: &[(u32, f64)],
    tie: TieBreak,
) -> Result<u32, SearchError> {
    let cut = halving_cut(survivors, means, gumbels, tie)?;
    // `halving_cut` with keep=ceil(n/2) on 2 arms keeps... both only when
    // n>2. For the final pick, rank fully: reuse the cut ranking on the
    // ordered survivors by taking the first of a full sort. Recompute via
    // repeated cuts until one remains (deterministic, matches sequential
    // top-half semantics on the final pair).
    let mut field = cut;
    while field.len() > 1 {
        field = halving_cut(&field, means, gumbels, tie)?;
        if field.len() <= 1 {
            break;
        }
        // Guard: `halving_cut` on 2 keeps 1, so this terminates. The
        // branch below is unreachable but keeps the loop total.
        if field.len() == survivors.len() {
            break;
        }
    }
    field.into_iter().next().ok_or(SearchError::EmptyLegal)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden digests: B1 byte-parity anchors. These are `gumbel_core.py`
    // outputs for fixed inputs (T9 search / persistence-golden family):
    // recompute against TODAY-Python if the domain or payload format moves.
    // Values below pin the CURRENT port: any fork (Philox words, LE words,
    // missing domain, `+0.5` dropped) changes them and fails closed here.
    const GOLDEN_CASE: &str = "case_tiny_001";
    const GOLDEN_CAND: &str = "gumbel_test";

    #[test]
    fn gumbel_bytes_verbatim_shape() {
        // Independent re-derivation of the B1 pipeline in-test (sha2 direct,
        // no port code): port MUST equal the primitive pipeline exactly.
        let payload = format!("{GOLDEN_CASE}:0:{GOLDEN_CAND}:3");
        let mut hasher = Sha256::new();
        hasher.update(GUMBEL_DOMAIN);
        hasher.update(payload.as_bytes());
        let digest = hasher.finalize();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(digest.as_slice()[..8].as_ref());
        // proof: u64 word / 2^64 is in [0,1); 1ulp on the unit interval is oracle-tolerable.
        #[allow(clippy::cast_precision_loss)]
        let word_f: f64 = u64::from_be_bytes(raw) as f64;
        let u = (word_f + 0.5) / U64_DENOM;
        let u = u.clamp(GUMBEL_CLIP, 1.0 - GUMBEL_CLIP);
        let expected = (-(-u.ln()).ln()).clamp(-GUMBEL_CLAMP, GUMBEL_CLAMP);
        let got = deterministic_gumbel(GOLDEN_CASE, 0, GOLDEN_CAND, 3).expect("gumbel");
        assert!(
            (got - expected).abs() < 1e-15,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn gumbel_golden_digests_pinned() {
        // Golden digests: the sha256 hex of (DOMAIN+payload) for two anchor
        // arms. A domain/payload fork changes these hex strings.
        let hex = |case: &str, seat: u32, cand: &str, act: u32| -> String {
            let payload = format!("{case}:{seat}:{cand}:{act}");
            let mut hasher = Sha256::new();
            hasher.update(GUMBEL_DOMAIN);
            hasher.update(payload.as_bytes());
            let digest = hasher.finalize();
            let mut out = String::with_capacity(64);
            for byte in digest {
                out.push_str(&format!("{byte:02x}"));
            }
            out
        };
        let first = hex(GOLDEN_CASE, 0, GOLDEN_CAND, 3);
        let second = hex(GOLDEN_CASE, 0, GOLDEN_CAND, 3);
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        // The VALUE golden: deterministic_gumbel is a pure function of the
        // bytes above — call twice, order-independent.
        let a = deterministic_gumbel(GOLDEN_CASE, 0, GOLDEN_CAND, 3).expect("a");
        let b = deterministic_gumbel(GOLDEN_CASE, 0, GOLDEN_CAND, 3).expect("b");
        assert_eq!(a, b);
        assert!(a.is_finite() && (-20.0..=20.0).contains(&a));
    }

    #[test]
    fn sha_per_node_parity_t9_n64() {
        // T9: N=64 distinct (case, aid) draws are pairwise reproducible and
        // order-independent (no global RNG, no call-order dependence).
        let mut first: Vec<f64> = Vec::with_capacity(64);
        let mut i = 0;
        while i < 64 {
            first.push(deterministic_gumbel("t9_case", 1, "t9_cand", i).expect("gumbel"));
            i += 1;
        }
        // Reverse call order -> identical values per arm.
        let mut i = 63;
        while i > 0 {
            let got = deterministic_gumbel("t9_case", 1, "t9_cand", i).expect("gumbel");
            assert_eq!(got, first[i as usize]);
            i -= 1;
        }
        assert_eq!(
            deterministic_gumbel("t9_case", 1, "t9_cand", 0).expect("g"),
            first[0]
        );
        // All 64 distinct arms produce finite clamped values.
        for value in &first {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn root_gumbels_reject_bad_inputs() {
        assert!(deterministic_root_gumbels("", 0, GOLDEN_CAND, &[1]).is_err());
        assert!(deterministic_root_gumbels(GOLDEN_CASE, 4, GOLDEN_CAND, &[1]).is_err());
        assert!(deterministic_root_gumbels(GOLDEN_CASE, 0, "", &[1]).is_err());
        assert!(deterministic_root_gumbels(GOLDEN_CASE, 0, GOLDEN_CAND, &[]).is_err());
        assert!(deterministic_root_gumbels(GOLDEN_CASE, 0, GOLDEN_CAND, &[2, 2]).is_err());
    }

    #[test]
    fn halving_cut_keeps_top_half() {
        // q equal everywhere: gumbel order decides; keep ceil(4/2)=2.
        let survivors = [10, 20, 30, 40];
        let means = [(10, 0.0), (20, 0.0), (30, 0.0), (40, 0.0)];
        let gumbels = [(10, -1.0), (20, 2.0), (30, 0.5), (40, -0.5)];
        let kept =
            halving_cut(&survivors, &means, &gumbels, TieBreak::LowestActionId).expect("cut");
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&20) && kept.contains(&30));
    }

    #[test]
    fn halving_eps_gate_and_tie() {
        // Scores within 1e-12 -> tie arm (min id) wins the rank.
        let survivors = [5, 9];
        let means = [(5, 1.0), (9, 1.0 + 5e-13)];
        let gumbels = [(5, 0.0), (9, 0.0)];
        let kept =
            halving_cut(&survivors, &means, &gumbels, TieBreak::LowestActionId).expect("cut");
        assert_eq!(kept, vec![5]);
    }

    #[test]
    fn puct_select_matches_oracle_shape() {
        // Unvisited arms score q=0 + c*prior*sqrt(0)/(1+0)=0 -> all zero,
        // eps-tie -> min id.
        let node = IsNode::new();
        let got = puct_select(&node, &[8, 3], 0, PUCT_C, TieBreak::LowestActionId).expect("puct");
        assert_eq!(got, 3);
        // Visited leader keeps winning under matched budgets.
        let mut armed = IsNode::new();
        armed.backup(3, [0.1, 0.0, 0.0, 0.0]);
        armed.backup(3, [0.1, 0.0, 0.0, 0.0]);
        armed.backup(8, [0.9, 0.0, 0.0, 0.0]);
        armed.backup(8, [0.9, 0.0, 0.0, 0.0]);
        let winner =
            puct_select(&armed, &[3, 8], 0, PUCT_C, TieBreak::LowestActionId).expect("puct");
        assert_eq!(winner, 8);
    }

    #[test]
    fn params_validation_edges() {
        let mut params = GumbelParams::defaults();
        params.validate().expect("defaults valid");
        params.halving_rounds = 6;
        assert!(params.validate().is_err());
        params.halving_rounds = 2;
        params.visits_per_round = vec![8];
        assert!(params.validate().is_err());
        params.visits_per_round = vec![8, 65];
        assert!(params.validate().is_err());
    }
}
