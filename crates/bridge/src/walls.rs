//! walls: WP-04A deterministic wall-schedule tables + pure validators.
//!
//! Thin SPEC-conformance boundary over the frozen RiichiEnv 0.4.10 layout
//! facts (`python/hydra2/conformance/walls.py`, WP-04A): haipai slot order,
//! live/dead-wall index bases, honor first-copy ids, and the cycle-safe
//! wall-assignment solver. No IO, no wall-clock, no RNG: every input is an
//! explicit placement, every output a deterministic 136-tile wall.
//!
//! DAG: this module depends on pyo3 ONLY (pure integer math, no feed/shard/
//! search owner — lesson-2 grep found no existing owner for the WP-04A
//! schedule tables; the `wall_hash`/`partition` wall-id owners are a
//! different domain). FORBIDS: live-object orchestration (the mutable
//! `WallPlan` builder stays Python-side), canon re-printing, wall-clock or
//! RNG seeding.
//!
//! Concurrency pattern: validate attached, compute under ONE
//! `py.detach(|| ...)` with zero Python API inside, wrap attached —
//! precedent `crates/bridge/src/tiles.rs:50` (`py.detach` around the codec
//! call, `PyValueError` mapping attached). Registration precedent
//! `tiles.rs:126-129` (`wrap_pyfunction!` + `sub.add_function`); frozen
//! tuple-const export precedent `contracts.rs:1156`
//! (`PyTuple::new(py, [...])`); keyword-only signature precedent
//! `resume.rs:555` (`#[pyo3(signature = (*, ...))]`); test-harness
//! precedent `tiles.rs:140-141` (`Python::initialize();` +
//! `Python::attach`).
//!
//! Single-cdylib tree: registers onto the EXISTING `hydra2._native.contracts`
//! submodule via `register(&sub)` — no new top-level submodule (MAIN wires
//! one call; see wiring note on [`register`]). Fail-closed: every shape
//! violation is `ValueError` with the oracle's message text, never a
//! default or fallback.
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - `hands` keys must be plain-int dicts (`BTreeMap` extraction): the
//!   oracle silently ignores non-int seat keys while the bridge raises
//!   `TypeError`. Well-formed callers (`Scenario.hands:
//!   dict[int, dict[int, int]]`) are unaffected; out-of-range int seats are
//!   still ignored exactly like the oracle's `hands.get(seat, {})`.
//! - Multi-type supply/conflict renders sort by key (`BTreeMap`/`BTreeSet`
//!   `Debug`); the oracle renders dict/set insertion order. Single-entry
//!   renders (every observed guard) are byte-identical (`{27: 5}`);
//!   multi-entry renders keep the identical prefix text with sorted tails.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

/// First index consumed by a live (non-rinshan) draw (`walls.py:76`).
const LIVE_DRAW_BASE: u8 = 52;
/// Dealer's opening 14th tile (`walls.py:78`).
const DEALER_DRAW_INDEX: u8 = 52;
/// Index of the initially revealed dora indicator (`walls.py:80`).
const FIRST_DORA_INDICATOR_INDEX: u8 = 131;
/// Highest rinshan tile index (draws descend from here) (`walls.py:82`).
const RINSHAN_TOP_INDEX: u8 = 135;

/// Honor physical ids, first copy of each (`walls.py:85-91`).
const TILE_E: u8 = 108;
/// South first copy (`walls.py:86`).
const TILE_S: u8 = 112;
/// West first copy (`walls.py:87`).
const TILE_W: u8 = 116;
/// North first copy (`walls.py:88`).
const TILE_N: u8 = 120;
/// Haku first copy (`walls.py:89`).
const TILE_P: u8 = 124;
/// Hatsu first copy (`walls.py:90`).
const TILE_F: u8 = 128;
/// Chun first copy (`walls.py:91`).
const TILE_C: u8 = 132;

/// Seat-major haipai slot order (`walls.py:34-52`): seat `k` holds
/// `[4k..4k+3, 16+4k..19+4k, 32+4k..35+4k, 48+k]`.
const fn build_haipai_slots() -> [u8; 52] {
    let mut slots = [0u8; 52];
    let mut seat: i64 = 0;
    while seat < 4 {
        let mut position: i64 = 0;
        while position < 13 {
            let row = position / 4;
            let column = position - row * 4;
            let slot = if row < 3 {
                16 * row + 4 * seat + column
            } else {
                48 + seat
            };
            // proof: `seat` in 0..4 and `position` in 0..13 (loop bounds), so index in 0..52, fits `usize`.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let idx: usize = (seat * 13 + position) as usize;
            // proof: `slot` in 0..52 (haipai layout), fits `u8`.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let slot_b: u8 = slot as u8;
            slots[idx] = slot_b;
            position += 1;
        }
        seat += 1;
    }
    slots
}

/// Frozen haipai slot order, seat-major (`walls.py:34-52`).
const HAIPAI_SLOTS: [u8; 52] = build_haipai_slots();

/// Logical tile type of a physical id (`TileId = 4 * TileType + copy` with
/// copy in 0..3, so the type is `physical // 4`; `walls.py:94-96`).
///
/// `div_euclid` reproduces Python `//` floor semantics exactly (positive
/// divisor), including for negative inputs where truncating `/` diverges.
#[pyfunction]
fn type_id(py: Python<'_>, physical: i64) -> i64 {
    py.detach(|| physical.div_euclid(4))
}

/// The four physical ids of `tile_type` in ascending order
/// (`walls.py:99-101`).
///
/// Wrapping arithmetic keeps absurd inputs panic-free (the oracle's bigints
/// never overflow; every in-domain type `0..33` is exact).
#[pyfunction]
fn copies(py: Python<'_>, tile_type: i64) -> PyResult<Bound<'_, PyTuple>> {
    let base = tile_type.wrapping_mul(4);
    let ids = py.detach(|| {
        [
            base,
            base.wrapping_add(1),
            base.wrapping_add(2),
            base.wrapping_add(3),
        ]
    });
    PyTuple::new(py, ids)
}

/// Shared haipai-slot math (`walls.py:104-113`): validates attached,
/// computes the slot detached.
fn haipai_slot(seat: i64, position: i64) -> Result<u8, String> {
    if !(0..=3).contains(&seat) {
        return Err("seat must be 0..3".to_string());
    }
    if !(0..=12).contains(&position) {
        return Err("position must be 0..12".to_string());
    }
    let (row, column) = (position.div_euclid(4), position.rem_euclid(4));
    let slot = if row < 3 {
        16 * row + 4 * seat + column
    } else {
        48 + seat
    };
    u8::try_from(slot).map_err(|_| format!("walls haipai slot {slot} out of range"))
}

/// Wall index dealt as `position` (0-based) of `seat`'s opening hand
/// (`walls.py:104-113`).
#[pyfunction]
fn haipai_index(py: Python<'_>, seat: i64, position: i64) -> PyResult<u8> {
    py.detach(|| haipai_slot(seat, position))
        .map_err(PyValueError::new_err)
}

/// Pin `index` to physical tile `physical` (`WallPlan.require`,
/// `walls.py:123-127`).
fn plan_require(
    requirements: &mut BTreeMap<u8, BTreeSet<i64>>,
    index: i64,
    physical: i64,
) -> Result<(), String> {
    if !(0..=135).contains(&index) {
        return Err(format!("wall index {index} out of range"));
    }
    let slot = u8::try_from(index).map_err(|_| format!("wall index {index} out of range"))?;
    requirements.entry(slot).or_default().insert(physical);
    Ok(())
}

/// Pin an EXACT multiset `{physical_id: count}` onto haipai slots
/// (`WallPlan.require_hand`, `walls.py:129-154`): literal physical ids are
/// preserved verbatim, per-type supply is enforced at resolve time, and an
/// empty hand leaves every slot free (including seat validation, which the
/// oracle only reaches via `haipai_index` for nonempty hands).
fn plan_require_hand(
    requirements: &mut BTreeMap<u8, BTreeSet<i64>>,
    seat: i64,
    hand: &BTreeMap<i64, i64>,
) -> Result<(), String> {
    let mut wanted: Vec<i64> = Vec::new();
    for (&physical, &count) in hand.iter() {
        if !(0..=135).contains(&physical) {
            return Err(format!("seat {seat}: physical id {physical} out of range"));
        }
        // `[pid] * count` is empty for `count <= 0`; `0..count` mirrors that.
        for _ in 0..count {
            wanted.push(physical);
        }
    }
    if wanted.len() > 13 {
        return Err(format!(
            "seat {seat} hand exceeds 13 tiles: {}",
            wanted.len()
        ));
    }
    if wanted.is_empty() {
        return Ok(());
    }
    wanted.sort_unstable();
    for (position, physical) in wanted.iter().enumerate() {
        let slot = haipai_slot(
            seat,
            i64::try_from(position)
                .map_err(|_| format!("seat {seat} hand exceeds 13 tiles: {}", wanted.len()))?,
        )?;
        plan_require(requirements, i64::from(slot), *physical)?;
    }
    Ok(())
}

/// Solve the assignment and return the full 136-tile wall
/// (`WallPlan.resolve`, `walls.py:156-193`): whole-plan four-copy supply
/// check, then conflict check, then the cycle-safe swap walk (200-pass
/// bound, same pass-local skip of slots settled earlier in the pass).
fn plan_resolve(requirements: &BTreeMap<u8, BTreeSet<i64>>) -> Result<Vec<u8>, String> {
    let mut supply: BTreeMap<i64, usize> = BTreeMap::new();
    for ids in requirements.values() {
        for &physical in ids {
            *supply.entry(physical.div_euclid(4)).or_default() += 1;
        }
    }
    let over: BTreeMap<i64, usize> = supply.into_iter().filter(|(_, count)| *count > 4).collect();
    if !over.is_empty() {
        return Err(format!("tile types exceed four-copy supply: {over:?}"));
    }
    let conflicts: BTreeMap<u8, &BTreeSet<i64>> = requirements
        .iter()
        .filter(|(_, ids)| ids.len() > 1)
        .map(|(index, ids)| (*index, ids))
        .collect();
    if !conflicts.is_empty() {
        return Err(format!("conflicting wall requirements: {conflicts:?}"));
    }
    let mut wall: Vec<u8> = (0..=135u8).collect();
    let mut converged = false;
    for _ in 0..200 {
        let misplaced: Vec<(u8, i64)> = requirements
            .iter()
            .filter_map(|(index, ids)| {
                let physical = *ids.iter().next()?;
                if i64::from(wall[usize::from(*index)]) != physical {
                    Some((*index, physical))
                } else {
                    None
                }
            })
            .collect();
        if misplaced.is_empty() {
            converged = true;
            break;
        }
        for (index, physical) in misplaced {
            if i64::from(wall[usize::from(index)]) == physical {
                continue;
            }
            match wall.iter().position(|tile| i64::from(*tile) == physical) {
                Some(source) => wall.swap(source, usize::from(index)),
                // `list.index` wording, reached only for out-of-range ids.
                None => return Err(format!("{physical} is not in list")),
            }
        }
    }
    if !converged {
        return Err("wall assignment did not converge".to_string());
    }
    let remaining: BTreeMap<u8, i64> = requirements
        .iter()
        .filter_map(|(index, ids)| {
            let physical = *ids.iter().next()?;
            if i64::from(wall[usize::from(*index)]) != physical {
                Some((*index, physical))
            } else {
                None
            }
        })
        .collect();
    if !remaining.is_empty() {
        return Err(format!("unresolved wall slots: {remaining:?}"));
    }
    Ok(wall)
}

/// Assemble placement requirements in oracle call order
/// (`build_wall`, `walls.py:196-225`): seats `0..3` (unknown seats ignored
/// via the fixed loop, matching `hands.get(seat, {})`), dealer draw, sorted
/// live draws (`>= 52`), sorted dead-wall entries (`>= 120`).
fn build_wall_core(
    hands: &BTreeMap<i64, BTreeMap<i64, i64>>,
    dealer_draw: Option<i64>,
    live_draws: Option<&BTreeMap<i64, i64>>,
    dead_wall: Option<&BTreeMap<i64, i64>>,
) -> Result<Vec<u8>, String> {
    let mut requirements: BTreeMap<u8, BTreeSet<i64>> = BTreeMap::new();
    let empty: BTreeMap<i64, i64> = BTreeMap::new();
    for seat in 0..4i64 {
        let hand = hands.get(&seat).unwrap_or(&empty);
        plan_require_hand(&mut requirements, seat, hand)?;
    }
    if let Some(draw) = dealer_draw {
        plan_require(&mut requirements, i64::from(DEALER_DRAW_INDEX), draw)?;
    }
    if let Some(live) = live_draws {
        for (index, physical) in live.iter() {
            if *index < i64::from(LIVE_DRAW_BASE) {
                return Err("live draws start at index 52".to_string());
            }
            plan_require(&mut requirements, *index, *physical)?;
        }
    }
    if let Some(dead) = dead_wall {
        for (index, physical) in dead.iter() {
            if *index < 120 {
                return Err("dead-wall indices start at 120".to_string());
            }
            plan_require(&mut requirements, *index, *physical)?;
        }
    }
    plan_resolve(&requirements)
}

/// Build a complete wall from per-seat hands plus extra placements
/// (`walls.py:196-225`).
#[pyfunction]
#[pyo3(signature = (*, hands, dealer_draw=None, live_draws=None, dead_wall=None))]
fn build_wall(
    py: Python<'_>,
    hands: BTreeMap<i64, BTreeMap<i64, i64>>,
    dealer_draw: Option<i64>,
    live_draws: Option<BTreeMap<i64, i64>>,
    dead_wall: Option<BTreeMap<i64, i64>>,
) -> PyResult<Vec<u8>> {
    py.detach(|| build_wall_core(&hands, dealer_draw, live_draws.as_ref(), dead_wall.as_ref()))
        .map_err(PyValueError::new_err)
}

/// Register the WP-04A wall tables onto the EXISTING `contracts` submodule
/// (mirrors `tiles::register` onto its own submodule; here the caller
/// passes the already-created `contracts` submodule — MAIN wiring, one
/// line inside `contracts::register` before `m.add_submodule(&sub)`:
/// `crate::walls::register(&sub)?;` plus `mod walls;` in `lib.rs`).
/// Compute detached, wrap attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("LIVE_DRAW_BASE", LIVE_DRAW_BASE)?;
    sub.add("DEALER_DRAW_INDEX", DEALER_DRAW_INDEX)?;
    sub.add("FIRST_DORA_INDICATOR_INDEX", FIRST_DORA_INDICATOR_INDEX)?;
    sub.add("RINSHAN_TOP_INDEX", RINSHAN_TOP_INDEX)?;
    sub.add("TILE_E", TILE_E)?;
    sub.add("TILE_S", TILE_S)?;
    sub.add("TILE_W", TILE_W)?;
    sub.add("TILE_N", TILE_N)?;
    sub.add("TILE_P", TILE_P)?;
    sub.add("TILE_F", TILE_F)?;
    sub.add("TILE_C", TILE_C)?;
    sub.add("HAIPAI_SLOTS", PyTuple::new(py, HAIPAI_SLOTS)?)?;
    sub.add_function(wrap_pyfunction!(type_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(copies, sub)?)?;
    sub.add_function(wrap_pyfunction!(haipai_index, sub)?)?;
    sub.add_function(wrap_pyfunction!(build_wall, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle() {
        Python::initialize();
        // Oracle values hand-traced from `walls.py:34-91`.
        assert_eq!(LIVE_DRAW_BASE, 52);
        assert_eq!(DEALER_DRAW_INDEX, 52);
        assert_eq!(FIRST_DORA_INDICATOR_INDEX, 131);
        assert_eq!(RINSHAN_TOP_INDEX, 135);
        assert_eq!(
            (TILE_E, TILE_S, TILE_W, TILE_N, TILE_P, TILE_F, TILE_C),
            (108, 112, 116, 120, 124, 128, 132)
        );
        // Seat 0 opens the table: 0,1,2,3,16,17,18,19,32,33,34,35,48.
        assert_eq!(
            &HAIPAI_SLOTS[0..13],
            &[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48]
        );
        // Last seat closes it: seat 3 rows plus 48+3.
        assert_eq!(
            &HAIPAI_SLOTS[39..52],
            &[12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 51]
        );
        assert_eq!(HAIPAI_SLOTS.len(), 52);
        let mut sorted = HAIPAI_SLOTS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 52);
    }

    #[test]
    fn scalar_leaves_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // `physical // 4` incl. the honor band and the floor edge.
            assert_eq!(type_id(py, 0), 0);
            assert_eq!(type_id(py, 16), 4);
            assert_eq!(type_id(py, 108), 27);
            assert_eq!(type_id(py, 135), 33);
            assert_eq!(type_id(py, -1), -1);
            // Full four-copy blocks, ascending.
            let first: Vec<i64> = copies(py, 0).unwrap().extract().unwrap();
            assert_eq!(first, vec![0, 1, 2, 3]);
            let chun: Vec<i64> = copies(py, 33).unwrap().extract().unwrap();
            assert_eq!(chun, vec![132, 133, 134, 135]);
            // Slot geometry hand-traced from `haipai_index`.
            assert_eq!(haipai_index(py, 0, 0).unwrap(), 0);
            assert_eq!(haipai_index(py, 0, 12).unwrap(), 48);
            assert_eq!(haipai_index(py, 1, 4).unwrap(), 20);
            assert_eq!(haipai_index(py, 2, 11).unwrap(), 43);
            assert_eq!(haipai_index(py, 3, 12).unwrap(), 51);
            // Fail-closed guards keep oracle texts.
            assert!(haipai_index(py, 4, 0).is_err());
            assert!(haipai_index(py, 0, 13).is_err());
            assert!(haipai_index(py, -1, 0).is_err());
        });
    }

    #[test]
    fn build_wall_identity_and_placements() {
        Python::initialize();
        Python::attach(|py| {
            // Empty plan resolves to the identity wall.
            let empty: BTreeMap<i64, BTreeMap<i64, i64>> = BTreeMap::new();
            let wall = build_wall_core(&empty, None, None, None).unwrap();
            let identity: Vec<u8> = (0..=135u8).collect();
            assert_eq!(wall, identity);
            // One pinned haipai tile swaps into slot 0 cycle-safely:
            // wall[0] == 108, wall[108] == 0, everything else identity.
            let hands = BTreeMap::from([(0, BTreeMap::from([(108, 1)]))]);
            let wall = build_wall_core(&hands, None, None, None).unwrap();
            assert_eq!(wall[0], 108);
            assert_eq!(wall[108], 0);
            assert_eq!(wall.len(), 136);
            for (index, tile) in wall.iter().enumerate() {
                if index != 0 && index != 108 {
                    // proof: test `index` in 0..136, fits `u8`.
                    #[allow(clippy::cast_possible_truncation)]
                    let idx_b: u8 = index as u8;
                    assert_eq!(*tile, idx_b);
                }
            }
            // Live draw below the base and dead-wall below 120 fail closed.
            let live = BTreeMap::from([(51, 0)]);
            assert_eq!(
                build_wall_core(&empty, None, Some(&live), None).unwrap_err(),
                "live draws start at index 52"
            );
            let dead = BTreeMap::from([(119, 0)]);
            assert_eq!(
                build_wall_core(&empty, None, None, Some(&dead)).unwrap_err(),
                "dead-wall indices start at 120"
            );
            // Five copies of one type across seats exhaust supply.
            let over = BTreeMap::from([
                (0, BTreeMap::from([(108, 4)])),
                (1, BTreeMap::from([(109, 1)])),
            ]);
            assert!(
                build_wall_core(&over, None, None, None)
                    .unwrap_err()
                    .starts_with("tile types exceed four-copy supply: ")
            );
            // Dealer draw and live draw disagreeing on index 52 conflict.
            let live = BTreeMap::from([(52, 8)]);
            assert!(
                build_wall_core(&empty, Some(7), Some(&live), None)
                    .unwrap_err()
                    .starts_with("conflicting wall requirements: ")
            );
            // Same value twice is one requirement, not a conflict.
            let live = BTreeMap::from([(52, 7)]);
            let wall = build_wall_core(&empty, Some(7), Some(&live), None).unwrap();
            assert_eq!(wall[52], 7);
            // The pyfn defaults mirror the oracle keyword-only call shape.
            let wall = build_wall(py, empty.clone(), None, None, None).unwrap();
            assert_eq!(wall, identity);
        });
    }
}
