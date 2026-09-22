#![allow(clippy::print_stderr)] // CLI dump binary: stderr progress is its interface
//! `hydra2_replay_dump`: framed per-game JSONL -> decision rows JSONL.
//!
//! Reads a directory of per-game framed MJAI files (one JSON object per
//! line, as staged by the Step-0 oracle dumper; plain `.jsonl` or
//! compressed `.jsonl.zst` / `.jsonl.gz` since the corpus ships
//! `.mjai.json.zst`), replays each game through the wall-less v1 driver,
//! and emits one JSON object per decision row. Whole-game quarantines are
//! collected into a sidecar JSON file instead of failing the run (mirrors
//! the oracle harness quarantine table): the CLI exits 0 with
//! `<out>.quarantine.json` always written (override with
//! `--quarantine-out`), even when every game replays clean.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use hydra_shard::{ActionTable, Quarantine};

#[derive(Debug, Parser)]
#[command(
    name = "hydra2_replay_dump",
    about = "Wall-less v1 replay: per-game framed MJAI -> decision rows JSONL"
)]
struct Args {
    /// Directory of per-game framed `.jsonl` input files.
    #[arg(long)]
    inputs: PathBuf,
    /// Output rows file (one JSON object per decision row).
    #[arg(long)]
    out: PathBuf,
    /// Published action-table artifact
    /// (`configs/contracts/action_table_v1.json`).
    #[arg(long)]
    action_table: PathBuf,
    /// Quarantine sidecar output (JSON map game_id -> reason).
    /// Defaults to `<out>.quarantine.json`.
    #[arg(long)]
    quarantine_out: Option<PathBuf>,
    /// Optional manifest mapping input file names to stream object ids
    /// (`{"games": [{"file": ..., "object_id": ...}]}`).
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Engine backend: `v1` (pinned drained-oracle, default) or `v2`
    /// (single-pass live-engine semantics: kuikae-strict, complete-chi,
    /// kyushu, take conservation).
    #[arg(long, default_value = "v1")]
    engine_version: String,
}

fn object_id_for(file_name: &str, manifest: &BTreeMap<String, String>) -> String {
    if let Some(object_id) = manifest.get(file_name) {
        return object_id.clone();
    }
    // Fall back to the file stem (mirrors `stem_of` closely enough for the
    // probe layout `gNNN-<game-id>.jsonl`, where the game id is explicit in
    // the framed bytes only when the log carries it). Compressed framings
    // keep the same stem with a codec suffix.
    for suffix in [".jsonl.zst", ".jsonl.gz", ".jsonl"] {
        if let Some(stem) = file_name.strip_suffix(suffix) {
            return stem.to_string();
        }
    }
    file_name.to_string()
}

/// Framed per-game inputs: plain `.jsonl` or compressed `.jsonl.zst` /
/// `.jsonl.gz` (corpus ships `.mjai.json.zst`; zstd+flate2 decode only,
/// never re-generated here).
fn is_game_file(file_name: &str) -> bool {
    file_name.ends_with(".jsonl")
        || file_name.ends_with(".jsonl.zst")
        || file_name.ends_with(".jsonl.gz")
}

/// Read one framed input, transparently decoding `.zst` (zstd) / `.gz`
/// (flate2) framings to UTF-8 text for the strict JSONL framer.
fn read_input_text(path: &std::path::Path, file_name: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("input read {file_name}: {e}"))?;
    let raw: Vec<u8> = if file_name.ends_with(".zst") {
        zstd::decode_all(&bytes[..]).map_err(|e| format!("input zstd decode {file_name}: {e}"))?
    } else if file_name.ends_with(".gz") {
        use std::io::Read as _;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| format!("input gzip decode {file_name}: {e}"))?;
        out
    } else {
        bytes
    };
    String::from_utf8(raw).map_err(|e| format!("input utf8 {file_name}: {e}"))
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let table_text =
        fs::read_to_string(&args.action_table).map_err(|e| format!("action table read: {e}"))?;
    let table =
        ActionTable::load_json(&table_text).map_err(|e| format!("action table load: {e}"))?;
    eprintln!("action table: {} templates", table.len);

    let manifest: BTreeMap<String, String> = match &args.manifest {
        None => BTreeMap::new(),
        Some(path) => {
            let text = fs::read_to_string(path).map_err(|e| format!("manifest read: {e}"))?;
            let doc: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("manifest parse: {e}"))?;
            let mut map = BTreeMap::new();
            if let Some(games) = doc.get("games").and_then(|v| v.as_array()) {
                for entry in games {
                    if let (Some(file), Some(object_id)) = (
                        entry.get("file").and_then(|v| v.as_str()),
                        entry.get("object_id").and_then(|v| v.as_str()),
                    ) {
                        map.insert(file.to_string(), object_id.to_string());
                    }
                }
            }
            map
        }
    };

    let mut files: Vec<PathBuf> = fs::read_dir(&args.inputs)
        .map_err(|e| format!("inputs read: {e}"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("inputs list: {e}"))?;
    files.sort();
    eprintln!("input files: {}", files.len());

    let quarantine_path = args.quarantine_out.clone().unwrap_or_else(|| {
        let mut name = args.out.as_os_str().to_owned();
        name.push(".quarantine.json");
        PathBuf::from(name)
    });

    let out_file = fs::File::create(&args.out).map_err(|e| format!("out create: {e}"))?;
    let mut writer = std::io::BufWriter::new(out_file);
    use std::io::Write as _;

    let mut n_rows: u64 = 0;
    let mut n_ok: u64 = 0;
    let mut quarantines: BTreeMap<String, Quarantine> = BTreeMap::new();
    for path in &files {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if !is_game_file(&file_name) {
            continue;
        }
        let text = read_input_text(path, &file_name)?;
        let object_id = object_id_for(&file_name, &manifest);
        let version = match args.engine_version.as_str() {
            "v2" => hydra_shard::EngineVersion::V2,
            "v1" => hydra_shard::EngineVersion::V1,
            other => return Err(format!("unknown --engine-version {other:?} (want v1|v2)")),
        };
        match hydra_shard::replay_game_text_with_version(&text, &object_id, &table, version) {
            Ok(rows) => {
                n_ok += 1;
                for row in &rows {
                    let line = serde_json::to_string(row)
                        .map_err(|e| format!("row encode: {e}"))?;
                    writeln!(writer, "{line}").map_err(|e| format!("out write: {e}"))?;
                    n_rows += 1;
                }
            }
            Err(quarantine) => {
                quarantines.insert(file_name.clone(), quarantine);
            }
        }
    }
    drop(writer);
    let quarantine_json: BTreeMap<&String, &Quarantine> =
        quarantines.iter().collect();
    fs::write(
        &quarantine_path,
        serde_json::to_string_pretty(&quarantine_json).map_err(|e| format!("quarantine encode: {e}"))?,
    )
    .map_err(|e| format!("quarantine write: {e}"))?;
    eprintln!("games ok: {n_ok} quarantined: {} rows: {n_rows}", quarantines.len());
    for (file, quarantine) in &quarantines {
        eprintln!("quarantine {file}: {}", quarantine.reason_code);
    }
    Ok(())
}
