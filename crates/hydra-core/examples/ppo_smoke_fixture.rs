use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{DecisionRecord, FirstActionSelector, GameRunner, StepOutcome};
use serde::Serialize;

#[derive(Serialize)]
struct Fixture<'a> {
    seed: u64,
    final_scores: [i32; 4],
    placements: [u8; 4],
    rows: &'a [FixtureRow],
}

#[derive(Serialize)]
struct FixtureRow {
    obs: Vec<f32>,
    legal_mask: Vec<bool>,
    action: u8,
    legal_count: u8,
    player_id: u8,
    seat_id: u8,
    turn: u32,
    game_id: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let _program = args.next();
    let Some(output) = args.next() else {
        return Err("usage: ppo_smoke_fixture <output-json> [seed]".into());
    };
    if args.len() > 1 {
        return Err("usage: ppo_smoke_fixture <output-json> [seed]".into());
    }
    let seed = match args.next() {
        Some(value) => value.to_string_lossy().parse::<u64>()?,
        None => 20260524,
    };

    let mut runner = GameRunner::new(Some(seed), 0);
    let mut selector = FirstActionSelector;
    let mut rows = Vec::new();
    while !runner.is_done() {
        let outcome = runner.step_once_recording(&mut selector, &mut |record: DecisionRecord| {
            rows.push(row_from_record(seed, record));
        });
        match outcome {
            StepOutcome::Advanced => {}
            StepOutcome::Complete => break,
            StepOutcome::StepLimitExceeded => return Err("game step limit exceeded".into()),
            StepOutcome::NoLegalAction { player } => {
                return Err(format!("no legal action for player {player}").into());
            }
        }
    }
    if !runner.is_done() {
        return Err("game did not complete".into());
    }
    if rows.is_empty() {
        return Err("game produced no decision rows".into());
    }

    let final_scores = runner.scores();
    let placements = compute_placements(final_scores);
    let fixture = Fixture {
        seed,
        final_scores,
        placements,
        rows: &rows,
    };
    let path = PathBuf::from(output);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &fixture)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn row_from_record(game_id: u64, record: DecisionRecord) -> FixtureRow {
    debug_assert_eq!(record.obs.len(), OBS_SIZE);
    debug_assert_eq!(record.legal_mask.len(), HYDRA_ACTION_SPACE);
    FixtureRow {
        obs: record.obs.to_vec(),
        legal_mask: record.legal_mask.to_vec(),
        action: record.action,
        legal_count: record.legal_count,
        player_id: record.player_id,
        seat_id: record.seat_id,
        turn: record.turn,
        game_id,
    }
}
