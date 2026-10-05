// CLI smoke binary: stdout timing/size report is its interface (same as feed_duel.rs).
#![allow(clippy::print_stdout, clippy::print_stderr)]
use std::{error::Error, fs, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("expected immutable tape blob path")?;
    let bytes = fs::read(path)?;
    let started = Instant::now();
    let full: serde_json::Value = serde_json::from_slice(&bytes)?;
    let full_seconds = started.elapsed().as_secs_f64();
    let expected = full.as_object().ok_or("tape root must be an object")?;
    let expected = ["row_count", "event_ref"]
        .into_iter()
        .filter_map(|key| {
            expected
                .get(key)
                .map(|value| (key.to_string(), value.clone()))
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    let walls = full
        .get("games")
        .and_then(serde_json::Value::as_array)
        .ok_or("tape games absent")?
        .iter()
        .map(|game| game.get("wall_id").cloned().ok_or("wall id absent"))
        .collect::<Result<Vec<_>, _>>()?;
    drop(full);
    let started = Instant::now();
    let projected =
        hydra_feed::json_projection::object_fields(&bytes, &["row_count", "event_ref"])?;
    let projected_seconds = started.elapsed().as_secs_f64();
    if expected != projected {
        return Err("projected runtime metadata differs from full tape parsing".into());
    }
    let started = Instant::now();
    let games = hydra_feed::json_projection::object_array_fields(&bytes, "games", &["wall_id"])?;
    let game_seconds = started.elapsed().as_secs_f64();
    let actual = games
        .get("games")
        .and_then(serde_json::Value::as_array)
        .ok_or("projected games absent")?
        .iter()
        .map(|game| game.get("wall_id").cloned().ok_or("projected wall absent"))
        .collect::<Result<Vec<_>, _>>()?;
    if actual != walls || game_seconds > full_seconds {
        return Err("wall projection parity or timing gate failed".into());
    }
    println!(
        "wall_projection_seconds={game_seconds:.6} walls={}",
        actual.len()
    );
    if projected_seconds > full_seconds {
        return Err("projection timing gate failed".into());
    }
    println!(
        "bytes={} full_seconds={full_seconds:.6} projected_seconds={projected_seconds:.6} metadata={}",
        bytes.len(),
        serde_json::to_string(&projected)?
    );
    Ok(())
}
