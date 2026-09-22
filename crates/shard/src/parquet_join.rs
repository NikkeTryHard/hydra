//! Privileged parquet cold join (P4-B, cold only — NEVER hot).
//!
//! The Python parquet owner (`src/hydra2/data/parquet.py`) writes privileged
//! shards with the hoisted `_PRIVILEGED_SCHEMA`: `decision_id: string`,
//! `privileged_label: string` (opaque JSON `{"ranks": [..], "split": ..,
//! "wall_id"?}`), `full_world: string`. Rust never writes parquet — it only
//! reads those three string columns here (by NAME, null-tolerant) and joins
//! them to collated rows on the opaque `decision_id`.
//!
//! Label validation reuses the cold authorities: ranks pin to the strict
//! 1..4 permutation via [`crate::parity::validate_privileged_ranks`] and the
//! opaque label shape via [`crate::parity::check_privileged_label`].

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

use crate::parity::validate_privileged_ranks;

/// One privileged label joined to its actor row by opaque `decision_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivilegedJoin {
    pub decision_id: String,
    pub ranks: [u8; 4],
    pub split: String,
    pub wall_id: Option<String>,
}

/// Privileged-join failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum ParquetJoinError {
    Io(std::io::Error),
    Parquet(parquet::errors::ParquetError),
    MissingColumn { name: &'static str },
    EmptyDecisionId { row: usize },
    BadLabel { row: usize, msg: String },
    DuplicateDecisionId { decision_id: String },
}

impl core::fmt::Display for ParquetJoinError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParquetJoinError::Io(e) => write!(f, "privileged join io: {e}"),
            ParquetJoinError::Parquet(e) => write!(f, "privileged join parquet: {e}"),
            ParquetJoinError::MissingColumn { name } => {
                write!(f, "privileged join missing column: {name}")
            }
            ParquetJoinError::EmptyDecisionId { row } => {
                write!(f, "privileged join empty decision_id at row {row}")
            }
            ParquetJoinError::BadLabel { row, msg } => {
                write!(f, "privileged join bad label at row {row}: {msg}")
            }
            ParquetJoinError::DuplicateDecisionId { decision_id } => {
                write!(f, "privileged join duplicate decision_id: {decision_id}")
            }
        }
    }
}

impl std::error::Error for ParquetJoinError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParquetJoinError::Io(e) => Some(e),
            ParquetJoinError::Parquet(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ParquetJoinError {
    fn from(e: std::io::Error) -> Self {
        ParquetJoinError::Io(e)
    }
}

impl From<parquet::errors::ParquetError> for ParquetJoinError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        ParquetJoinError::Parquet(e)
    }
}

fn field_to_string(field: &Field) -> Option<String> {
    match field {
        Field::Str(s) => Some(s.clone()),
        Field::Bytes(b) => String::from_utf8(b.data().to_vec()).ok(),
        _ => None,
    }
}

/// Parse one opaque `privileged_label` JSON value into rank/split/wall parts.
///
/// Mirrors `write_privileged_ranks`: `ranks` is the authoritative strict 1..4
/// permutation (validated here), `split` is provenance passthrough,
/// `wall_id` rides only on walled rows.
pub fn parse_privileged_label(
    decision_id: &str,
    label_text: &str,
) -> Result<([u8; 4], String, Option<String>), String> {
    let label: serde_json::Value =
        serde_json::from_str(label_text).map_err(|e| format!("label JSON: {e}"))?;
    crate::parity::check_privileged_label(&label)
        .map_err(|e| format!("label shape: {e}"))?;
    let ranks_value = label.get("ranks").ok_or_else(|| "label missing ranks".to_string())?;
    let ranks_arr = ranks_value.as_array().ok_or_else(|| "ranks not an array".to_string())?;
    let ranks = validate_privileged_ranks(ranks_arr, decision_id)
        .map_err(|e| format!("ranks: {e}"))?;
    let split = label
        .get("split")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "label missing split".to_string())?
        .to_string();
    let wall_id = label
        .get("wall_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok((ranks, split, wall_id))
}

/// Read a Python-written privileged shard (`privileged-*.parquet`) into join
/// rows: `decision_id` + parsed opaque label per row, file order preserved.
/// `full_world` is carried by the file but never interpreted here (belief
/// support is deferred; the join only needs ranks/split/wall_id).
pub fn read_privileged_parquet(path: &Path) -> Result<Vec<PrivilegedJoin>, ParquetJoinError> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let mut has_decision = false;
    let mut has_label = false;
    for col in schema.columns() {
        if col.name() == "decision_id" {
            has_decision = true;
        }
        if col.name() == "privileged_label" {
            has_label = true;
        }
    }
    if !has_decision {
        return Err(ParquetJoinError::MissingColumn { name: "decision_id" });
    }
    if !has_label {
        return Err(ParquetJoinError::MissingColumn {
            name: "privileged_label",
        });
    }
    let mut out = Vec::new();
    let mut seen: HashMap<String, ()> = HashMap::new();
    let iter = reader.get_row_iter(None)?;
    for (row_idx, row) in iter.enumerate() {
        let row = row?;
        let mut decision_id: Option<String> = None;
        let mut label_text: Option<String> = None;
        for (name, field) in row.get_column_iter() {
            if name == "decision_id" {
                decision_id = field_to_string(field);
            } else if name == "privileged_label" {
                label_text = field_to_string(field);
            }
        }
        let decision_id = decision_id.unwrap_or_default();
        if decision_id.is_empty() {
            return Err(ParquetJoinError::EmptyDecisionId { row: row_idx });
        }
        if seen.contains_key(&decision_id) {
            return Err(ParquetJoinError::DuplicateDecisionId { decision_id });
        }
        let label_text = label_text.ok_or_else(|| ParquetJoinError::BadLabel {
            row: row_idx,
            msg: "missing privileged_label".to_string(),
        })?;
        let (ranks, split, wall_id) =
            parse_privileged_label(&decision_id, &label_text).map_err(|msg| {
                ParquetJoinError::BadLabel { row: row_idx, msg }
            })?;
        seen.insert(decision_id.clone(), ());
        out.push(PrivilegedJoin {
            decision_id,
            ranks,
            split,
            wall_id,
        });
    }
    Ok(out)
}

/// Hash-join collated row keys to privileged labels: for each `decision_id`
/// in row order, the index into `labels` (`None` = no privileged row, e.g. a
/// quarantined-then-replayed game that never earned a rank row).
pub fn join_privileged(
    decision_ids: &[String],
    labels: &[PrivilegedJoin],
) -> Vec<Option<usize>> {
    let mut by_id: HashMap<&str, usize> = HashMap::with_capacity(labels.len());
    let mut i = 0usize;
    while i < labels.len() {
        by_id.entry(labels[i].decision_id.as_str()).or_insert(i);
        i += 1;
    }
    let mut out = Vec::with_capacity(decision_ids.len());
    for id in decision_ids {
        out.push(by_id.get(id.as_str()).copied());
    }
    out
}
