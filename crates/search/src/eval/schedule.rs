//! Schedule commitment — pre-results pure function (EvalControl-owned).
//!
//! Rust owner of `schedule.py:41-194,197-239` allocation + latency +
//! commitment half (the `RandomStreamKey` shapes stay Python-side; this
//! module owns the math + hashes).
//! B3 canon-wins: all hashes (`walls_hash`, `latency_schedule_hash`,
//! `seed_protocol_hash`, `schedule_commitment_hash`) call `feed::canon` +
//! `feed::digest` via `super::canon_digest`. No second printer here.
//!
//! M3 (`%3` vs Lemire fork): `_latency_draw` (`:187-189`)
//! `u64BE(semantic_seed(master,key)[:8]) % 3` stays `%3` VERBATIM.
//! Lemire-`below` is ONLY for NEW streams, never this draw — `2^64 mod 3
//! != 0`, so the rejection tail would fork `latency_schedule_hash`.
//! The `%3` latency KAT pins 20 draws on `bytes(range(32))`.
//!
//! - Seat tables: `_SEAT_PAIRS` (`:46-53`) + `symmetric_pair_allocations`
//!   (`:115-126`) + `focal_rotation_allocations` (`:129-138`) ported
//!   verbatim; per wall 6 symmetric + 4 rotation = 10 games.
//! - `build_match_schedule` (`:141-184`): commits walls, seats, latency
//!   classes (`LATENCY_CLASSES`), and the seed-protocol payload (`:55-64`
//!   verbatim incl. `schedule_purposes` + `commitment_order`).
//! - `seat_pair_placements_exact` (`:197-239`): a constructor
//!   post-condition — pair `[3,3,3,3]` seat hits + focal `[0,1,2,3]`.
//! - Commitment SHA gate: `schedule_commitment_hash` (`:192-194`) is
//!   `of_canonical(to_json)`; rebuilds from the same inputs are
//!   byte-identical. T2 pins the COMMIT golden.

use crate::SearchError;

use super::{canon_digest, semantic_seed};

/// Latency classes (`LATENCY_CLASSES`, `:41`).
pub const LATENCY_CLASSES: [&str; 3] = ["low", "moderate", "high"];

/// Symmetric allocations per wall (`:42`).
pub const SYMMETRIC_ALLOCATIONS_PER_WALL: usize = 6;

/// Rotation allocations per wall (`:43`).
pub const ROTATION_ALLOCATIONS_PER_WALL: usize = 4;

/// Total games per wall = 6 + 4 = 10 (`:44`).
pub const TOTAL_GAMES_PER_WALL: usize =
    SYMMETRIC_ALLOCATIONS_PER_WALL + ROTATION_ALLOCATIONS_PER_WALL;

/// Unordered seat pairs (`_SEAT_PAIRS`, `:46-53`).
pub const SEAT_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// All six placements of `pair` across unordered seat pairs
/// (`symmetric_pair_allocations`, `:115-126`).
pub fn symmetric_pair_allocations(pair: (&str, &str), field: (&str, &str)) -> Vec<[String; 4]> {
    let mut rows = Vec::with_capacity(SYMMETRIC_ALLOCATIONS_PER_WALL);
    for (low_seat, high_seat) in SEAT_PAIRS {
        let mut row: [String; 4] = [String::new(), String::new(), String::new(), String::new()];
        row[low_seat] = pair.0.to_string();
        row[high_seat] = pair.1.to_string();
        let mut remaining = [0usize; 2];
        let mut k = 0;
        for seat in 0..4 {
            if seat != low_seat && seat != high_seat {
                remaining[k] = seat;
                k += 1;
            }
        }
        row[remaining[0]] = field.0.to_string();
        row[remaining[1]] = field.1.to_string();
        rows.push(row);
    }
    rows
}

/// Four 1-v-3 diagnostics; focal sits at each seat exactly once
/// (`focal_rotation_allocations`, `:129-138`): `row = list(others)`,
/// `row.insert(seat, focal)`.
pub fn focal_rotation_allocations(focal: &str, others: &[String; 3]) -> Vec<[String; 4]> {
    let mut rows = Vec::with_capacity(ROTATION_ALLOCATIONS_PER_WALL);
    for seat in 0..4 {
        let mut row = vec![others[0].clone(), others[1].clone(), others[2].clone()];
        row.insert(seat, focal.to_string());
        rows.push([
            row[0].clone(),
            row[1].clone(),
            row[2].clone(),
            row[3].clone(),
        ]);
    }
    rows
}

/// SPEC 18.1 schedule; frozen before results exist (`MatchSchedule`,
/// `:67-104`). Field order matches `to_json` (`:96-104`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchSchedule {
    /// Committed wall ids (nonempty, unique).
    pub wall_ids: Vec<String>,
    /// Digest over the wall-id list.
    pub walls_hash: String,
    /// 10 allocation rows per wall (each a 4-label permutation).
    pub seat_allocations: Vec<[String; 4]>,
    /// Digest over the latency rows.
    pub latency_schedule_hash: String,
    /// Rules digest.
    pub rules_hash: String,
    /// Digest over the seed-protocol payload.
    pub seed_protocol_hash: String,
}

impl MatchSchedule {
    /// Constructor post-condition (`__post_init__`, `:78-94`): nonempty +
    /// unique walls, 10 rows per wall, digest shapes, 4-label-permutation
    /// rows. Exactness gates run in [`build_match_schedule`].
    pub fn check(&self) -> Result<(), SearchError> {
        if self.wall_ids.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "schedule needs at least one wall",
            });
        }
        let mut seen = std::collections::HashSet::new();
        for wall_id in &self.wall_ids {
            if wall_id.is_empty() || !seen.insert(wall_id.as_str()) {
                return Err(SearchError::InvalidArg {
                    detail: "wall_ids must be unique",
                });
            }
        }
        if self.seat_allocations.len() != self.wall_ids.len() * TOTAL_GAMES_PER_WALL {
            return Err(SearchError::InvalidArg {
                detail: "seat_allocations must carry 10 rows per wall",
            });
        }
        for digest in [
            &self.walls_hash,
            &self.latency_schedule_hash,
            &self.rules_hash,
            &self.seed_protocol_hash,
        ] {
            if !super::is_digest_text(digest) {
                return Err(SearchError::InvalidArg {
                    detail: "schedule digest malformed",
                });
            }
        }
        for row in &self.seat_allocations {
            let mut uniq = std::collections::HashSet::new();
            for label in row {
                uniq.insert(label.as_str());
            }
            if uniq.len() != 4 {
                return Err(SearchError::InvalidArg {
                    detail: "allocation row is not a 4-label permutation",
                });
            }
        }
        Ok(())
    }

    /// Canonical JSON projection (`to_json`, `:96-104`).
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "wall_ids": self.wall_ids,
            "walls_hash": self.walls_hash,
            "seat_allocations": self.seat_allocations.iter().map(|r| vec![&r[0], &r[1], &r[2], &r[3]]).collect::<Vec<_>>(),
            "latency_schedule_hash": self.latency_schedule_hash,
            "rules_hash": self.rules_hash,
            "seed_protocol_hash": self.seed_protocol_hash,
        })
    }
}

/// Validate four distinct nonempty labels (`_validate_labels`, `:107-112`).
fn require_labels(labels: &[String]) -> Result<[String; 4], SearchError> {
    if labels.len() != 4 || labels.iter().any(String::is_empty) {
        return Err(SearchError::InvalidArg {
            detail: "labels must be four nonempty strings",
        });
    }
    let uniq: std::collections::HashSet<&str> = labels.iter().map(String::as_str).collect();
    if uniq.len() != 4 {
        return Err(SearchError::InvalidArg {
            detail: "labels must be distinct",
        });
    }
    Ok([
        labels[0].clone(),
        labels[1].clone(),
        labels[2].clone(),
        labels[3].clone(),
    ])
}

/// Seed-protocol payload, VERBATIM (`_SEED_PROTOCOL_PAYLOAD`, `:55-64`):
/// protocol + version + derivation sentence + `schedule_purposes` +
/// `commitment_order`.
pub fn seed_protocol_payload() -> serde_json::Value {
    serde_json::json!({
        "protocol": "hydra2_rng_v1",
        "seed_protocol_version": 1,
        "derivation": "sha256(canonical_json({protocol, master_seed.hex, key})) with purpose-discriminated RandomStreamKey",
        "schedule_purposes": ["wall", "evaluation_schedule"],
        "commitment_order": ["walls", "seats", "latency", "protocol"],
    })
}

/// Full SPEC-13 `RandomStreamKey` JSON for an `evaluation_schedule` draw
/// (field order irrelevant — JCS sorts; nulls preserved per
/// `key_to_json`, `randomness.py:228-230`). All non-core fields are null
/// for this purpose (`_REQUIRED_BY_PURPOSE` + `_OPTIONAL_BY_PURPOSE`).
pub fn schedule_stream_key(
    experiment_id: &str,
    split_id: &str,
    replicate_id: u64,
) -> serde_json::Value {
    serde_json::json!({
        "purpose": "evaluation_schedule",
        "experiment_id": experiment_id,
        "split_id": split_id,
        "candidate_id": null,
        "case_id": null,
        "wall_id": null,
        "root_seat": null,
        "belief_epoch": null,
        "parent_id": null,
        "action_id": null,
        "packet_id": null,
        "fidelity_level": null,
        "population_id": null,
        "replicate_id": replicate_id,
        "scramble_id": null,
        "visit_index": null,
        "attempt_id": 0,
    })
}

/// Latency draw (`_latency_draw`, `:187-189`):
/// `u64BE(semantic_seed(master,key)[:8]) % 3` — `%3` VERBATIM (M3).
/// Lemire-`bounded(3)` is barred here: it would fork
/// `latency_schedule_hash`.
pub fn latency_draw(master_seed: &[u8], key: &serde_json::Value) -> Result<usize, SearchError> {
    let seed = semantic_seed(master_seed, key)?;
    let mut n = 0u64;
    for b in &seed[..8] {
        n = (n << 8) | u64::from(*b);
    }
    // proof: `LATENCY_CLASSES.len()` = 3 (frozen const), exact in both casts; `%3` verbatim per M3.
    #[allow(clippy::cast_possible_truncation)]
    let cls_u: u64 = LATENCY_CLASSES.len() as u64;
    // proof: `n % 3` in 0..3, fits `usize`.
    #[allow(clippy::cast_possible_truncation)]
    let idx: usize = (n % cls_u) as usize;
    Ok(idx)
}

/// Commit walls, seats, latency classes, and seed protocol up front
/// (`build_match_schedule`, `:141-184`).
pub fn build_match_schedule(
    wall_ids: &[String],
    labels: &[String],
    rules_hash: &str,
    master_seed: &[u8],
    experiment_id: &str,
    split_id: &str,
) -> Result<MatchSchedule, SearchError> {
    if wall_ids.is_empty() || wall_ids.iter().any(String::is_empty) {
        return Err(SearchError::InvalidArg {
            detail: "wall_ids must be nonempty strings",
        });
    }
    let uniq: std::collections::HashSet<&str> = wall_ids.iter().map(String::as_str).collect();
    if uniq.len() != wall_ids.len() {
        return Err(SearchError::InvalidArg {
            detail: "wall_ids must be unique",
        });
    }
    if !super::is_digest_text(rules_hash) {
        return Err(SearchError::InvalidArg {
            detail: "rules_hash must be sha256:<hex>",
        });
    }
    let lab = require_labels(labels)?;
    let pair = (&lab[0], &lab[1]);
    let field = (&lab[2], &lab[3]);
    let others = [lab[1].clone(), lab[2].clone(), lab[3].clone()];

    let mut allocations: Vec<[String; 4]> = Vec::new();
    let mut latency_rows: Vec<serde_json::Value> = Vec::new();
    for (wall_index, wall_id) in wall_ids.iter().enumerate() {
        let mut wall_rows = symmetric_pair_allocations(
            (pair.0.as_str(), pair.1.as_str()),
            (field.0.as_str(), field.1.as_str()),
        );
        wall_rows.extend(focal_rotation_allocations(lab[0].as_str(), &others));
        for (slot, _) in wall_rows.iter().enumerate().take(TOTAL_GAMES_PER_WALL) {
            let key = schedule_stream_key(
                experiment_id,
                split_id,
                (wall_index * TOTAL_GAMES_PER_WALL + slot) as u64,
            );
            let class_index = latency_draw(master_seed, &key)?;
            latency_rows.push(serde_json::json!([
                wall_id,
                slot,
                LATENCY_CLASSES[class_index]
            ]));
        }
        allocations.extend(wall_rows);
    }

    let walls_hash = canon_digest(&serde_json::json!(wall_ids), "eval:schedule:walls")?;
    let latency_schedule_hash =
        canon_digest(&serde_json::json!(latency_rows), "eval:schedule:latency")?;
    let seed_protocol_hash = canon_digest(&seed_protocol_payload(), "eval:schedule:proto")?;

    let schedule = MatchSchedule {
        wall_ids: wall_ids.to_vec(),
        walls_hash,
        seat_allocations: allocations,
        latency_schedule_hash,
        rules_hash: rules_hash.to_string(),
        seed_protocol_hash,
    };
    seat_pair_placements_exact(&schedule)?;
    schedule.check()?;
    Ok(schedule)
}

/// Single pre-results commitment binding every schedule facet
/// (`schedule_commitment_hash`, `:192-194`).
pub fn schedule_commitment_hash(schedule: &MatchSchedule) -> Result<String, SearchError> {
    canon_digest(&schedule.to_json_value(), "eval:schedule:commitment")
}

pub fn seat_pair_placements_exact(schedule: &MatchSchedule) -> Result<(), SearchError> {
    if schedule.seat_allocations.len() < schedule.wall_ids.len() * TOTAL_GAMES_PER_WALL
        || schedule.seat_allocations.len() <= SYMMETRIC_ALLOCATIONS_PER_WALL
    {
        return Err(SearchError::InvalidArg {
            detail: "seat_allocations must carry 10 rows per wall",
        });
    }
    let pair_members: std::collections::HashSet<&str> = [
        schedule.seat_allocations[0][0].as_str(),
        schedule.seat_allocations[0][1].as_str(),
    ]
    .into_iter()
    .collect();
    let focal = schedule.seat_allocations[SYMMETRIC_ALLOCATIONS_PER_WALL][0].clone();
    for (wall_index, wall_id) in schedule.wall_ids.iter().enumerate() {
        let base = wall_index * TOTAL_GAMES_PER_WALL;
        let symmetric = &schedule.seat_allocations[base..base + SYMMETRIC_ALLOCATIONS_PER_WALL];
        let rotations = &schedule.seat_allocations
            [base + SYMMETRIC_ALLOCATIONS_PER_WALL..base + TOTAL_GAMES_PER_WALL];
        let mut placements: Vec<(usize, usize)> = symmetric
            .iter()
            .map(|row| {
                let mut seats: Vec<usize> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, label)| pair_members.contains(label.as_str()))
                    .map(|(index, _)| index)
                    .collect();
                seats.sort_unstable();
                (seats[0], seats[1])
            })
            .collect();
        placements.sort_unstable();
        let mut want = SEAT_PAIRS.to_vec();
        want.sort_unstable();
        if placements != want {
            return Err(SearchError::InvalidArg {
                detail: "A-pair placements not exact",
            });
        }
        let mut hits = [0usize; 4];
        for row in symmetric {
            for (index, label) in row.iter().enumerate() {
                if pair_members.contains(label.as_str()) {
                    hits[index] += 1;
                }
            }
        }
        if hits != [3, 3, 3, 3] {
            return Err(SearchError::InvalidArg {
                detail: "seat coverage != [3,3,3,3]",
            });
        }
        let mut rotation_seats: Vec<usize> =
            rotations
                .iter()
                .flat_map(|row| {
                    row.iter().enumerate().filter_map(|(index, label)| {
                        if *label == focal { Some(index) } else { None }
                    })
                })
                .collect();
        rotation_seats.sort_unstable();
        if rotation_seats != vec![0, 1, 2, 3] {
            return Err(SearchError::InvalidArg {
                detail: "focal rotation != all seats",
            });
        }
        let _ = wall_id;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        ["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect()
    }

    fn master() -> Vec<u8> {
        (0u8..32).collect()
    }

    /// T2 goldens (torch 2.14 oracle): commitment, walls/latency/proto hashes for the 2-wall
    /// KAT schedule.
    #[test]
    fn commitment_goldens() {
        let walls = vec!["w-001".to_string(), "w-002".to_string()];
        let schedule = build_match_schedule(
            &walls,
            &labels(),
            &format!("sha256:{}", "a".repeat(64)),
            &master(),
            "exp-kat",
            "split-kat",
        )
        .unwrap();
        assert_eq!(
            schedule_commitment_hash(&schedule).unwrap(),
            "sha256:7e4bd2c5a1e5b8d054e1471e6df6a07945ffd08cbcc6b551848958243250783d"
        );
        assert_eq!(
            schedule.walls_hash,
            "sha256:a64106a6d9ee1186b6ead2341571119dba0ac8a5c49ac2e3c9eaf813cdef1f78"
        );
        assert_eq!(
            schedule.latency_schedule_hash,
            "sha256:18f4b186c7745ba2320b91bb8e53d9eeacd6b5cd79568bb1906769f272e5b2a2"
        );
        assert_eq!(
            schedule.seed_protocol_hash,
            "sha256:30538109867443d98bfc50220c1011b4fee656a702d348f2690632e6f9fa63b4"
        );
    }

    /// `%3` KAT (M3): 20 verbatim draws on `bytes(range(32))` — a
    /// Lemire-`bounded(3)` port would fork index 0 (`0xde… % 3 == 1` only
    /// via `%`; the rejection tail differs).
    #[test]
    fn latency_percent3_kat() {
        let want = vec![1, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 1, 2, 1, 1, 2, 2, 0, 0, 2];
        for (slot, class_index) in want.iter().enumerate() {
            let key = schedule_stream_key("exp-kat", "split-kat", slot as u64);
            assert_eq!(
                latency_draw(&master(), &key).unwrap(),
                *class_index,
                "slot {slot}"
            );
        }
    }

    /// Semantic-seed spot golden: the first schedule draw's seed bytes
    /// prove the key projection (with nulls) matches
    /// `make_random_stream_key` + `semantic_seed`.
    #[test]
    fn semantic_seed_spot() {
        let key = schedule_stream_key("exp-kat", "split-kat", 0);
        let seed = semantic_seed(&master(), &key).unwrap();
        assert_eq!(
            super::super::hex_of(&seed),
            "de584f1e7a5df3c0febcd02f247fe41a1dbd8b49299f11cdb16f3247e9467ebe"
        );
    }

    /// Allocation geometry: 6+4 per wall, 10 rows per wall, first label
    /// set fixed positions (pair `[0][:2]`, focal `[6][0]`).
    #[test]
    fn allocation_geometry() {
        let walls = vec!["w-001".to_string()];
        let schedule = build_match_schedule(
            &walls,
            &labels(),
            &format!("sha256:{}", "a".repeat(64)),
            &master(),
            "exp-kat",
            "split-kat",
        )
        .unwrap();
        assert_eq!(schedule.seat_allocations.len(), TOTAL_GAMES_PER_WALL);
        assert_eq!(symmetric_pair_allocations(("A", "B"), ("C", "D")).len(), 6);
        assert_eq!(
            focal_rotation_allocations("A", &["B".to_string(), "C".to_string(), "D".to_string()])
                .len(),
            4
        );
        seat_pair_placements_exact(&schedule).unwrap();
    }

    /// Guards: empty/dup walls, bad labels, bad digest all fail closed.
    #[test]
    fn schedule_constructor_guards() {
        let empty: Vec<String> = Vec::new();
        assert!(
            build_match_schedule(
                &empty,
                &labels(),
                &format!("sha256:{}", "a".repeat(64)),
                &master(),
                "e",
                "s"
            )
            .is_err()
        );
        let dup = vec!["w".to_string(), "w".to_string()];
        assert!(
            build_match_schedule(
                &dup,
                &labels(),
                &format!("sha256:{}", "a".repeat(64)),
                &master(),
                "e",
                "s"
            )
            .is_err()
        );
        let walls = vec!["w".to_string()];
        let bad_labels = vec![
            "A".to_string(),
            "A".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        assert!(
            build_match_schedule(
                &walls,
                &bad_labels,
                &format!("sha256:{}", "a".repeat(64)),
                &master(),
                "e",
                "s"
            )
            .is_err()
        );
        assert!(
            build_match_schedule(&walls, &labels(), "not-a-digest", &master(), "e", "s").is_err()
        );
    }
}
