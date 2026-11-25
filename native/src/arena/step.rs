//! Atomic validation and application for one staged action per active game.
//!
//! The batch keeps each row's canonical legal actions, so step validation
//! only matches ids against that stored view and application never rebuilds
//! a legal view or rescans the action census. Invalid ids are rejected before
//! any game is advanced. Independent games apply through the arena's bounded
//! pool when it has more than one worker; a single row or a single-worker
//! arena applies directly to avoid a needless pool wakeup.

use hydra_rules::rl::actions::CanonicalAction;
use rayon::prelude::*;

use super::game::Game;

pub(super) struct StagedRow {
    pub(super) slot: usize,
    pub(super) seat: u8,
    pub(super) legal: Vec<(u32, CanonicalAction)>,
}

/// Match one incoming id to each staged row and return the chosen action
/// index inside that row's stored legal view.
pub(super) fn validate(staged: &[StagedRow], ids: &[i32]) -> Result<Vec<usize>, String> {
    if ids.len() != staged.len() {
        return Err(format!(
            "step expected {} ids but received {}",
            staged.len(),
            ids.len()
        ));
    }
    let mut chosen = Vec::with_capacity(ids.len());
    for (row, raw) in staged.iter().zip(ids) {
        let id = u32::try_from(*raw).map_err(|_| {
            format!(
                "slot {} seat {} action id {raw} is negative",
                row.slot, row.seat
            )
        })?;
        let index = row
            .legal
            .binary_search_by_key(&id, |(candidate, _)| *candidate)
            .map_err(|_| {
                format!(
                    "slot {} seat {} action id {id} is not in the staged legal view",
                    row.slot, row.seat
                )
            })?;
        chosen.push(index);
    }
    Ok(chosen)
}

/// Apply a fully validated selection. Games not present in the staged batch
/// are untouched. The caller poisons the arena on an error from this phase.
pub(super) fn apply(
    staged: &[StagedRow],
    chosen: &[usize],
    games: &mut [Game],
    pool: &rayon::ThreadPool,
) -> Result<(), String> {
    if staged.is_empty() {
        return Ok(());
    }
    if chosen.len() != staged.len() {
        return Err("staged action selection length mismatch".to_owned());
    }
    if staged.windows(2).any(|pair| pair[0].slot >= pair[1].slot) {
        return Err("staged rows are not strictly slot ordered".to_owned());
    }
    if staged.iter().any(|row| row.slot >= games.len()) {
        return Err("staged slot is outside the arena".to_owned());
    }

    if staged.len() == 1 {
        let row = &staged[0];
        let game = games
            .get_mut(row.slot)
            .ok_or_else(|| format!("slot {} is outside the arena", row.slot))?;
        return apply_one(row, chosen[0], game);
    }

    if pool.current_num_threads() == 1 {
        for (row, choice) in staged.iter().zip(chosen) {
            let game = games
                .get_mut(row.slot)
                .ok_or_else(|| format!("slot {} is outside the arena", row.slot))?;
            apply_one(row, *choice, game)?;
        }
        return Ok(());
    }

    let results: Vec<Result<(), String>> = pool.install(|| {
        games
            .par_iter_mut()
            .enumerate()
            .map(
                |(slot, game)| match staged.binary_search_by_key(&slot, |row| row.slot) {
                    Ok(index) => apply_one(&staged[index], chosen[index], game),
                    Err(_) => Ok(()),
                },
            )
            .collect()
    });
    for result in results {
        result?;
    }
    Ok(())
}

fn apply_one(row: &StagedRow, chosen: usize, game: &mut Game) -> Result<(), String> {
    let (_, action) = row.legal.get(chosen).ok_or_else(|| {
        format!(
            "slot {} selected action index {chosen} is outside the stored view",
            row.slot
        )
    })?;
    game.apply(row.seat, action)
        .map_err(|error| format!("slot {} seat {} apply failed: {error}", row.slot, row.seat))
}
