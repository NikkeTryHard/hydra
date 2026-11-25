//! Parallel row staging for one arena batch.
//!
//! One row per active game is observed from the reference legal view and
//! observation builder, then encoded exactly once by the shared feature
//! module. The same legal view is retained in the staged row so a later step
//! can validate and apply without rebuilding it. Rows are collected in slot
//! order so both the wire batch and the staged selection are stable.

use rayon::prelude::*;

use crate::features;

use super::Arena;
use super::game::Game;
use super::step::StagedRow;

pub(super) struct PreparedRow {
    pub(super) slot: usize,
    pub(super) seat: u8,
    pub(super) legal_count: usize,
    pub(super) tokens: Vec<f32>,
    pub(super) candidates: Vec<i32>,
}

pub(super) struct PreparedBatch {
    pub(super) rows: Vec<PreparedRow>,
    pub(super) staged: Vec<StagedRow>,
}

pub(super) fn prepare(arena: &mut Arena) -> Result<PreparedBatch, String> {
    let pool = &arena.pool;
    let games = &mut arena.games;
    let serial = pool.current_num_threads() == 1
        || games.iter().filter(|game| !game.is_terminal()).count() <= 1;

    if serial {
        let mut rows = Vec::with_capacity(games.len());
        let mut staged = Vec::with_capacity(games.len());
        for (slot, game) in games.iter_mut().enumerate() {
            if game.is_terminal() {
                continue;
            }
            let (row, staged_row) = prepare_row(game, slot)?;
            rows.push(row);
            staged.push(staged_row);
        }
        return Ok(PreparedBatch { rows, staged });
    }

    let results: Vec<Result<Option<(PreparedRow, StagedRow)>, String>> = pool.install(|| {
        games
            .par_iter_mut()
            .enumerate()
            .map(prepare_game_row)
            .collect()
    });

    let mut rows = Vec::with_capacity(results.len());
    let mut staged = Vec::with_capacity(results.len());
    for result in results {
        if let Some((row, staged_row)) = result? {
            rows.push(row);
            staged.push(staged_row);
        }
    }
    Ok(PreparedBatch { rows, staged })
}

fn prepare_game_row(
    (slot, game): (usize, &mut Game),
) -> Result<Option<(PreparedRow, StagedRow)>, String> {
    if game.is_terminal() {
        Ok(None)
    } else {
        prepare_row(game, slot).map(Some)
    }
}

fn prepare_row(game: &mut Game, slot: usize) -> Result<(PreparedRow, StagedRow), String> {
    let seat = game.pending_seat()?;
    let legal = game
        .legal_view(seat)
        .map_err(|error| format!("slot {slot} legal view failed: {error}"))?;
    if legal.is_empty() {
        return Err(format!("slot {slot} seat {seat} has no legal actions"));
    }
    let observation = game
        .observe(seat, &legal)
        .map_err(|error| format!("slot {slot} observation failed: {error}"))?;

    let token_len = features::TOKENS * features::WIDTH;
    let candidate_len = legal.len() * features::ACTION;
    let mut tokens = vec![0.0f32; token_len];
    let mut candidates = vec![0i32; candidate_len];
    features::encode(
        &observation,
        game.arena(),
        &legal,
        &mut tokens,
        &mut candidates,
    )
    .map_err(|error| format!("slot {slot} feature encode failed: {error}"))?;
    if tokens.len() != token_len || candidates.len() != candidate_len {
        return Err(format!(
            "slot {slot} feature encoder changed buffer geometry"
        ));
    }

    let legal_count = legal.len();
    let staged = StagedRow { slot, seat, legal };
    let row = PreparedRow {
        slot,
        seat,
        legal_count,
        tokens,
        candidates,
    };
    Ok((row, staged))
}
