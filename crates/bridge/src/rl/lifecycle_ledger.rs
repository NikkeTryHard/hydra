//! rl/lifecycle_ledger: deterministic source-ledger generation.
//!
//! DAG: `super::lifecycle_common` + `serde_json` +
//! `hydra_feed::digest::ctr_block` (counter bytes, single owner) +
//! `hydra_search::eval::semantic_seed` (stream seeds, single owner) +
//! `hydra_shard::parity::wall_schedule_digest` (schedule digests, single
//! owner). FORBIDS: PyO3, wall-clock, core auto-walls, cross-game coupling.
//!
//! The ledger is a pure function of (master seed, experiment, split,
//! wall_id, replicate, attempt): per-game `wall`-purpose seeds, tiles via
//! descending Fisher-Yates, then one rotation draw (documented stream
//! order), schedule digests via the shard law.

use serde_json::{Map, Value};

use super::lifecycle_common::{canonical_digest, lc_err, LifecycleError};

/// Wall-id naming law: `rl-g{02}-game{03}` (opaque caller string; the pool
/// passes it through to `GameRow.wall_id` verbatim).
pub fn wall_id_for(generation: u64, game: u64) -> String {
    format!("rl-g{generation:02}-game{game:03}")
}

/// Ledger split scope law: epoch-independent collection scope
/// (`selection_training:g={g}`); the shuffle key extends it with `:epoch=`.
pub fn ledger_split_for(generation: u64) -> String {
    format!("selection_training:g={generation}")
}

/// Seekable counter stream over the feed CTR block (same window law as the
/// replay-index order builder: byte `p` depends only on `(seed, p/32)`).
struct CtrStream {
    seed: Vec<u8>,
    cursor: u64,
}

impl CtrStream {
    fn get_bytes(&mut self, count: usize) -> Result<Vec<u8>, LifecycleError> {
        if self.seed.is_empty() {
            return Err(lc_err("rl lifecycle stream seed must be nonempty"));
        }
        const BLOCK: u64 = 32;
        let end = (self.cursor as usize)
            .checked_add(count)
            .ok_or_else(|| lc_err("rl lifecycle stream window overflows"))?;
        let first_block = self.cursor / BLOCK;
        let offset = (self.cursor % BLOCK) as usize;
        let mut acc = Vec::with_capacity(((end as u64).div_ceil(BLOCK) as usize) * 32);
        let mut block = first_block;
        for _ in 0..(end as u64).div_ceil(BLOCK) {
            acc.extend_from_slice(&hydra_feed::digest::ctr_block(&self.seed, block));
            block = block
                .checked_add(1)
                .ok_or_else(|| lc_err("rl lifecycle stream block counter overflows"))?;
        }
        let out = acc
            .get(offset..offset + count)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| lc_err("rl lifecycle stream window outside accumulated blocks"))?;
        self.cursor = self
            .cursor
            .checked_add(count as u64)
            .ok_or_else(|| lc_err("rl lifecycle stream cursor overflows"))?;
        Ok(out)
    }

    /// Exact `random_below` rejection law (oracle `randomness.py:193-201`):
    /// minimal whole bytes covering the bound,
    /// `limit = 256^nbytes / bound * bound` floor, BE draw, redraw while
    /// `value >= limit`, result `value % bound`.
    fn random_below(&mut self, bound: u64) -> Result<u64, LifecycleError> {
        if bound < 2 {
            return Err(lc_err("rl lifecycle random_below bound must be >= 2"));
        }
        let bits = 64 - (bound - 1).leading_zeros() as usize;
        let nbytes = bits / 8 + 1;
        if nbytes > 8 {
            return Err(lc_err("rl lifecycle random_below bound exceeds u64 draws"));
        }
        let base: u128 = 256;
        let limit = base.pow(nbytes as u32) / bound as u128 * bound as u128;
        loop {
            let raw = self.get_bytes(nbytes)?;
            let mut value: u128 = 0;
            for b in raw {
                value = (value << 8) | b as u128;
            }
            if value < limit {
                return Ok((value % bound as u128) as u64);
            }
        }
    }
}

/// Descending Fisher-Yates over `0..n` (`j = random_below(i+1)`,
/// swap `i <-> j` for `i = n-1 .. 1`).
fn fisher_yates(n: usize, stream: &mut CtrStream) -> Result<Vec<u8>, LifecycleError> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = stream.random_below(i as u64 + 1)? as usize;
        perm.swap(i, j);
    }
    perm.into_iter()
        .map(|v| {
            u8::try_from(v).map_err(|_| lc_err("rl lifecycle ledger permutation overflows u8"))
        })
        .collect()
}

/// Oracle key-shape gate for the `wall` purpose (`randomness.py:162-185` +
/// `validate_key:243-290`): `wall` requires `wall_id`, allows no optional
/// fields, is not game-scoped (so `case_id` must stay null), and always
/// carries the nonempty `experiment_id`/`split_id` core plus explicit
/// `replicate_id`/`attempt_id`. Every other field must be null.
fn validate_wall_key(key: &Map<String, Value>) -> Result<(), LifecycleError> {
    let str_nonempty = |name: &str| -> Result<(), LifecycleError> {
        match key.get(name) {
            Some(Value::String(text)) if !text.is_empty() => Ok(()),
            _ => Err(lc_err(format!(
                "rl lifecycle wall key field {name} must be a nonempty str"
            ))),
        }
    };
    match key.get("purpose") {
        Some(Value::String(p)) if p == "wall" => {}
        other => {
            return Err(lc_err(format!(
                "rl lifecycle wall key purpose must be 'wall', got {other:?}"
            )))
        }
    }
    str_nonempty("experiment_id")?;
    str_nonempty("split_id")?;
    str_nonempty("wall_id")?;
    // Authority gate (`randomness.rs:authority_gate`): selection_training
    // may feed `wall` (it is not a final-evaluation-only purpose).
    for name in ["replicate_id", "attempt_id"] {
        match key.get(name) {
            Some(Value::Number(n)) if !n.is_f64() && n.as_u64().is_some() => {}
            _ => {
                return Err(lc_err(format!(
                    "rl lifecycle wall key field {name} must be a nonnegative int"
                )))
            }
        }
    }
    for name in [
        "candidate_id",
        "case_id",
        "root_seat",
        "belief_epoch",
        "parent_id",
        "action_id",
        "packet_id",
        "fidelity_level",
        "population_id",
        "scramble_id",
        "visit_index",
    ] {
        if key.get(name) != Some(&Value::Null) {
            return Err(lc_err(format!("rl lifecycle wall key field {name} must be null")));
        }
    }
    Ok(())
}

/// Semantic `wall`-purpose seed for one ledger game. Single seed owner is
/// `hydra_search::eval`; the key shape is gated above against the oracle.
fn ledger_seed(
    master: &[u8],
    experiment_id: &str,
    generation: u64,
    wall_id: &str,
    replicate_id: u64,
    attempt_id: u64,
) -> Result<Vec<u8>, LifecycleError> {
    let mut key = Map::with_capacity(17);
    key.insert("purpose".to_string(), Value::String("wall".to_string()));
    key.insert("experiment_id".to_string(), Value::String(experiment_id.to_string()));
    key.insert("split_id".to_string(), Value::String(ledger_split_for(generation)));
    key.insert("candidate_id".to_string(), Value::Null);
    key.insert("case_id".to_string(), Value::Null);
    key.insert("wall_id".to_string(), Value::String(wall_id.to_string()));
    for null_field in [
        "root_seat",
        "belief_epoch",
        "parent_id",
        "action_id",
        "packet_id",
        "fidelity_level",
        "population_id",
    ] {
        key.insert(null_field.to_string(), Value::Null);
    }
    key.insert("replicate_id".to_string(), Value::from(replicate_id));
    key.insert("scramble_id".to_string(), Value::Null);
    key.insert("visit_index".to_string(), Value::Null);
    key.insert("attempt_id".to_string(), Value::from(attempt_id));
    validate_wall_key(&key)?;
    hydra_search::eval::semantic_seed(master, &Value::Object(key))
        .map(|seed| seed.to_vec())
        .map_err(|e| lc_err(format!("rl lifecycle ledger seed rejected: {e}")))
}

/// One ledger game: deterministic wall + seating + schedule digest.
pub(crate) struct LedgerGame {
    pub wall_id: String,
    pub schedule_id: String,
    pub schedule_digest: String,
    pub schedule_tiles: Vec<u8>,
    pub seat_perm: [u8; 4],
}

pub(crate) fn generate_ledger(
    master_seed: &[u8; 32],
    experiment_id: &str,
    replicate_id: u64,
    attempt_id: u64,
    generations: u64,
    games_per_generation: u64,
) -> Result<Vec<Vec<LedgerGame>>, LifecycleError> {
    let mut out: Vec<Vec<LedgerGame>> = Vec::with_capacity(generations as usize);
    for generation in 0..generations {
        let mut cohort = Vec::with_capacity(games_per_generation as usize);
        for game in 0..games_per_generation {
            let wall_id = wall_id_for(generation, game);
            // Stream order per game (fixed: tiles first, then rotation).
            // Both draws come from the SAME per-game wall-purpose seed so
            // the ledger is a pure function of (master, experiment, split,
            // wall_id, replicate, attempt) with no cross-game coupling.
            let seed = ledger_seed(
                master_seed,
                experiment_id,
                generation,
                &wall_id,
                replicate_id,
                attempt_id,
            )?;
            let mut stream = CtrStream { seed, cursor: 0 };
            let tiles = fisher_yates(136, &mut stream)?;
            let rotation = stream.random_below(4)? as u8;
            let seat_perm =
                [rotation % 4, (rotation + 1) % 4, (rotation + 2) % 4, (rotation + 3) % 4];
            // Source digest law (existing authority
            // `hydra_shard::parity::wall_schedule_digest`).
            let schedule_digest = hydra_shard::parity::wall_schedule_digest(&wall_id, &tiles);
            cohort.push(LedgerGame {
                schedule_id: wall_id.clone(),
                wall_id,
                schedule_digest,
                schedule_tiles: tiles,
                seat_perm,
            });
        }
        out.push(cohort);
    }
    Ok(out)
}

pub(crate) fn ledger_digest(ledger: &[Vec<LedgerGame>]) -> Result<String, LifecycleError> {
    let generations: Vec<Value> = ledger
        .iter()
        .enumerate()
        .map(|(generation, cohort)| {
            let games: Vec<Value> = cohort
                .iter()
                .map(|game| {
                    serde_json::json!({
                        "wall_id": game.wall_id,
                        "schedule_id": game.schedule_id,
                        "schedule_digest": game.schedule_digest,
                        "schedule_tiles": game.schedule_tiles.iter().map(|t| Value::from(*t as u64)).collect::<Vec<_>>(),
                        "seat_perm": game.seat_perm.iter().map(|s| Value::from(*s as u64)).collect::<Vec<_>>(),
                    })
                })
                .collect();
            serde_json::json!({
                "generation": generation as u64,
                "games": games,
            })
        })
        .collect();
    canonical_digest(&serde_json::json!({ "generations": generations }), "rl:wall-ledger")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_key_gate_matches_oracle_shape() {
        // A well-formed wall key passes; setting case_id fails closed.
        let mut key = Map::new();
        key.insert("purpose".to_string(), Value::String("wall".to_string()));
        key.insert("experiment_id".to_string(), Value::String("exp".to_string()));
        key.insert(
            "split_id".to_string(),
            Value::String("selection_training:g=0".to_string()),
        );
        key.insert("candidate_id".to_string(), Value::Null);
        key.insert("case_id".to_string(), Value::Null);
        key.insert("wall_id".to_string(), Value::String("rl-g00-game000".to_string()));
        for null_field in [
            "root_seat",
            "belief_epoch",
            "parent_id",
            "action_id",
            "packet_id",
            "fidelity_level",
            "population_id",
            "scramble_id",
            "visit_index",
        ] {
            key.insert(null_field.to_string(), Value::Null);
        }
        key.insert("replicate_id".to_string(), Value::from(0u64));
        key.insert("attempt_id".to_string(), Value::from(0u64));
        assert!(validate_wall_key(&key).is_ok());
        key.insert("case_id".to_string(), Value::String("game".to_string()));
        assert!(validate_wall_key(&key).is_err());
    }

    #[test]
    fn ledger_rows_are_permutations_with_cyclic_seats() {
        // Consumer-visible invariant: every ledger game carries a 0..135
        // permutation plus a cyclic seat rotation (pool admission shape).
        let intake_seed = [0x5au8; 32];
        let ledger =
            generate_ledger(&intake_seed, "exp-smoke", 0, 0, 1, 4).expect("ledger builds");
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger[0].len(), 4);
        for game in &ledger[0] {
            let mut tiles = game.schedule_tiles.clone();
            tiles.sort_unstable();
            assert_eq!(tiles, (0..136u8).collect::<Vec<_>>());
            let mut seats = game.seat_perm.to_vec();
            seats.sort_unstable();
            assert_eq!(seats, vec![0, 1, 2, 3]);
            assert!(game.schedule_digest.starts_with("sha256:"));
        }
        // Deterministic: same inputs rebuild byte-identical tiles.
        let again = generate_ledger(&intake_seed, "exp-smoke", 0, 0, 1, 4).expect("rebuilds");
        assert_eq!(ledger[0][0].schedule_tiles, again[0][0].schedule_tiles);
        assert_eq!(ledger[0][0].schedule_digest, again[0][0].schedule_digest);
    }
}
